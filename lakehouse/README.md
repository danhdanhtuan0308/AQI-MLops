# AQI Lakehouse

## Layers

| Layer | Location | Description |
|-------|----------|-------------|
| **Bronze** | `s3://weather-bulk/data-pipeline/` | 1 year of historical AQI, Parquet per city/month |
| **Bronze** | `s3://weather-bulk/hourly/` | Live data, 99 cities written every hour by Lambda |
| **Silver** | Glue `aqi_db.aqi_unified` | Unified Iceberg table, deduplicated on `city_slug + timestamp` |

## How It Works

1. **Backfill (one-time)** — `backfill.py` reads all Bronze pipeline files, normalises types, and does `INSERT INTO aqi_unified`
2. **Hourly ingestion** — Lambda A (`aqi-hourly-ingest`) fetches live AQI for 99 cities → writes Parquet to `s3://weather-bulk/hourly/` → invokes Lambda B async
3. **Merge** — Lambda B (`aqi-iceberg-merge`) runs `MERGE INTO aqi_unified` using the exact S3 key (scans only 99 rows, ~50 KB)

## Key Files

| File | Purpose |
|------|---------|
| `setup.sql` | Athena DDL — creates database, Bronze external tables, Silver Iceberg table |
| `backfill.py` | One-time historical load into `aqi_unified` |
| `merge_handler.py` | Lambda B — hourly upsert into `aqi_unified` |
| `deploy_lakehouse.sh` | Deploys Lambda B, IAM role, and runs Athena setup |

## Query the Table

```sql
-- Row counts by source
SELECT source, COUNT(*) AS rows FROM aqi_db.aqi_unified GROUP BY source;

-- Latest readings
SELECT city, aqi, timestamp FROM aqi_db.aqi_unified
ORDER BY timestamp DESC LIMIT 10;
```

> **Athena settings**: set query result location to `s3://weather-bulk/athena-results/` before running queries in the console.
