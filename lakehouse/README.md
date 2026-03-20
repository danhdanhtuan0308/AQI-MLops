# AQI Lakehouse

## Medallion Layers

| Layer | Where data lives | What it is |
|-------|-----------------|------------|
| **Bronze** | `s3://weather-bulk/data-pipeline/` | Raw historical AQI, Parquet per city/month, unchanged |
| **Bronze** | `s3://weather-bulk/hourly/` | Raw live AQI, 99 cities per hour, unchanged |
| **Silver** | `s3://weather-bulk/processed/aqi_unified/` | Cleaned, deduplicated, unified Iceberg table |
| **Gold** | `s3://weather-bulk/processed/gold_*/` | Use-case specific tables (ML features, daily averages, etc.) |

## How Each Component Works

### Glue Data Catalog
- Stores **metadata only** — zero rows of actual data
- Holds: table name, column names + types, S3 location, file format, Iceberg snapshot pointer, partition spec
- Athena **must** look up Glue before every query to know the schema and where the files are

### Athena (Query Engine)
- Has no storage of its own
- Every query: (1) look up Glue for schema + S3 path → (2) read/write Parquet files on S3 directly
- Handles Bronze→Silver (INSERT/MERGE) and Silver→Gold (CREATE TABLE AS SELECT)

### S3
- The **only place actual data lives** — all Parquet files
- Bronze: raw files as written by the pipeline/Lambda
- Silver: Iceberg-managed files written by Athena's INSERT/MERGE
- Athena result temp output: `s3://weather-bulk/athena-results/` (not a data layer — just query output CSVs)

### Iceberg (table format on Silver)
- Enables MERGE INTO (upsert) on S3 files — not possible with plain Parquet
- Every write creates a new snapshot; Glue stores the pointer to the latest snapshot
- Partitioned by `month(timestamp)` for efficient time-range queries

## Data Flow

1. **Backfill (one-time)** — `backfill.py` reads all 1,287 Bronze pipeline files with pandas (handles mixed types), normalises, writes one clean Parquet to S3, then Athena `INSERT INTO aqi_unified` → 836,115 rows
2. **Hourly ingestion** — Lambda A fetches live AQI for 99 cities → writes Parquet to `s3://weather-bulk/hourly/` → invokes Lambda B async
3. **Hourly merge** — Lambda B runs `MERGE INTO aqi_unified` scoped to the exact new file (`$path` filter) — scans only 99 rows (~50 KB), never the full table
4. **Gold** — query `aqi_unified` in Athena with `CREATE TABLE AS SELECT` for specific use cases (ML training, dashboards, reporting)

## Key Files

| File | Purpose |
|------|---------|
| `setup.sql` | Athena DDL — creates database, Bronze external tables, Silver Iceberg table |
| `backfill.py` | One-time historical load into `aqi_unified` |
| `merge_handler.py` | Lambda B — hourly MERGE INTO `aqi_unified` |
| `deploy_lakehouse.sh` | Deploys Lambda B, IAM role, runs Athena setup |

## Querying

```sql
-- Row counts by source
SELECT source, COUNT(*) AS rows FROM aqi_db.aqi_unified GROUP BY source;

-- Latest readings
SELECT city, aqi, timestamp FROM aqi_db.aqi_unified
ORDER BY timestamp DESC LIMIT 10;

-- Gold example: daily city averages
CREATE TABLE aqi_db.gold_daily_avg AS
SELECT city, DATE_TRUNC('day', timestamp) AS date, AVG(aqi) AS avg_aqi
FROM aqi_db.aqi_unified
GROUP BY city, DATE_TRUNC('day', timestamp);
```

> **Athena console setup**: set query result location to `s3://weather-bulk/athena-results/` and database to `aqi_db` before running queries.
