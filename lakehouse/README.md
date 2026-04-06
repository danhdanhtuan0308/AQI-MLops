# AQI Lakehouse

The data lakehouse stores all raw sensor readings and model predictions on AWS S3 using Apache Iceberg as the table format and Athena as the query engine. Data is organized into four layers: two Bronze layers for raw data, one Silver layer for the unified cleaned dataset, and one Gold layer for model predictions.

---

## Data Layers

| Layer | S3 Location | What it contains |
|-------|-------------|------------------|
| Bronze | s3://weather-bulk/data-pipeline/ | Raw historical Parquet files from the one-time bulk ingest, never modified |
| Bronze | s3://weather-bulk/hourly/ | Raw live hourly Parquet files written by Lambda A, one file per hour |
| Silver | s3://weather-bulk/processed/aqi_unified/ | Cleaned, deduplicated, unified Iceberg table with all data from both Bronze sources |
| Gold | s3://weather-bulk/predictions/ | Hourly model predictions written by Lambda C, one Parquet file per flush |

---

## How Each Component Works

**Glue Data Catalog**

Stores metadata only and holds zero rows of actual data. It tracks table names, column types, S3 locations, file formats, Iceberg snapshot pointers, and partition definitions. Athena must look up Glue before every query to know the schema and where the files live.

**Athena**

A serverless query engine with no storage of its own. Every query goes to Glue for schema information then reads or writes Parquet files directly on S3. Handles the Bronze-to-Silver MERGE and the Gold INSERT from Lambda C.

**S3**

The only place actual data lives. Bronze files are written directly by the pipeline. Silver files are Iceberg-managed files rewritten by Athena's MERGE operations. Athena query results are written to s3://weather-bulk/athena-results/ as temporary CSV files.

**Apache Iceberg (Silver and Gold tables)**

Enables MERGE INTO (upsert) operations on S3 Parquet files, which is not possible with plain Parquet. Every write creates a new Iceberg snapshot. Glue stores the pointer to the latest snapshot. Both the Silver and Gold tables are partitioned by month for efficient time-range queries.

---

## Lambda Functions

**Lambda A (aqi-hourly-ingest)**

Runs every hour via EventBridge at :00. Fetches live AQI and pollutant readings for all 99 cities from OpenWeatherMap. Writes all 99 rows as a single Parquet file to s3://weather-bulk/hourly/{year}/{month}/{day}/. Then invokes Lambda B asynchronously.

**Lambda B (aqi-iceberg-merge)**

Triggered by Lambda A. Runs a MERGE INTO query in Athena against aqi_db.aqi_unified scoped to the exact new file. This means only the 99 new rows are ever scanned, not the full table. After the merge succeeds, it sends a POST /warm-cache HTTP request to the FastAPI service on EC2. FastAPI then pulls the latest data from Athena, runs the model on all 99 cities, and populates Redis.

Required environment variables for Lambda B:
- ATHENA_DB: aqi_db
- ATHENA_TABLE: aqi_unified
- ATHENA_RESULTS: s3://weather-bulk/athena-results/
- S3_BUCKET: weather-bulk
- PREDICT_API_URL: http://3.94.115.44

**Lambda C (aqi-prediction-flush)**

Runs every hour at :05 via EventBridge, 5 minutes after Lambda A so the warm-cache cycle has time to complete. Reads all records from the aqi:pred_buffer Redis list (one record per city, roughly 99 records per run). Writes them as a single Parquet file to s3://weather-bulk/predictions/year={year}/month={month}/. Then runs an INSERT INTO on aqi_db.predictions to register the records in Iceberg permanently.

Required environment variables for Lambda C:
- REDIS_URL: the Redis Cloud connection string (set manually in AWS console)
- S3_BUCKET: weather-bulk
- ATHENA_DB: aqi_db
- ATHENA_RESULTS: s3://weather-bulk/athena-results/

---

## Data Flow Step by Step

1. One-time backfill: backfill.py reads all historical Bronze pipeline files with pandas, normalises them, and runs INSERT INTO aqi_unified. This seeded the Silver table with 836,115 rows from 12 months of data.

2. Every hour at :00: Lambda A fetches 99 cities from OpenWeatherMap and writes raw Parquet to s3://weather-bulk/hourly/. It then invokes Lambda B.

3. Every hour (triggered by Lambda A): Lambda B runs MERGE INTO aqi_unified scoped to the exact new file. Only 99 rows are scanned. After the merge, it POSTs /warm-cache to FastAPI.

4. After Lambda B triggers it: FastAPI queries Athena for the last 340 hours of all 99 cities in one query. It builds features, runs the XGBoost model, stores predictions in Redis with a 2-hour TTL, and appends each prediction to aqi:pred_buffer.

5. Every hour at :05: Lambda C reads aqi:pred_buffer from Redis, writes a Parquet file to s3://weather-bulk/predictions/, and runs INSERT INTO aqi_db.predictions.

---

## Athena Tables

| Table | Type | Description |
|-------|------|-------------|
| aqi_db.raw_pipeline | External (Hive Parquet) | Bronze historical data from bulk ingest |
| aqi_db.raw_hourly | External (Hive Parquet, partitioned) | Bronze live hourly data from Lambda A |
| aqi_db.aqi_unified | Iceberg (Silver) | Deduplicated unified table, partitioned by month(timestamp) |
| aqi_db.predictions | Iceberg (Gold) | Hourly model predictions, partitioned by month(forecast_for) |

---

## Key Files

| File | Purpose |
|------|---------|
| setup.sql | Athena DDL to create all four tables |
| backfill.py | One-time historical load of 836,115 rows into aqi_unified |
| merge_handler.py | Lambda B handler: MERGE INTO aqi_unified then trigger /warm-cache |
| prediction_flush_handler.py | Lambda C handler: flush Redis pred_buffer to S3 and Athena predictions |
| deploy_lakehouse.sh | Deploys Lambda B and Lambda C, creates IAM roles, runs Athena DDL, sets up EventBridge rule for Lambda C |

---

## Deploying the Lakehouse

Run once from the project root with AWS credentials available:

```bash
bash lakehouse/deploy_lakehouse.sh
```

This script will:
1. Create the aqi_db database in Athena if it does not exist
2. Create all four tables (raw_pipeline, raw_hourly, aqi_unified, predictions)
3. Create the IAM role for Lambda B
4. Deploy Lambda B and set its environment variables
5. Grant Lambda A permission to invoke Lambda B
6. Deploy Lambda C with pyarrow and redis runtime dependencies
7. Create an EventBridge rule to trigger Lambda C every hour at :05

After the script completes, set REDIS_URL on Lambda C manually in the AWS console. Then run the one-time backfill:

```bash
uv run python lakehouse/backfill.py
```

---

## Sample Queries

```sql
-- Row counts by source in Silver table
SELECT source, COUNT(*) AS rows
FROM aqi_db.aqi_unified
GROUP BY source;

-- Latest sensor readings
SELECT city, aqi, timestamp
FROM aqi_db.aqi_unified
ORDER BY timestamp DESC
LIMIT 10;

-- Predicted vs actual AQI for accuracy tracking
SELECT
  p.city_slug,
  p.forecast_for,
  p.predicted,
  p.confidence,
  u.aqi AS actual
FROM aqi_db.predictions p
JOIN aqi_db.aqi_unified u
  ON p.city_slug = u.city_slug
 AND p.forecast_for = u.timestamp
ORDER BY p.forecast_for DESC
LIMIT 20;
```

Before running queries in the Athena console, set the query result location to s3://weather-bulk/athena-results/ and set the database to aqi_db.
