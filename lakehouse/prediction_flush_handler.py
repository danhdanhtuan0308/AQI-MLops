"""
Lambda C — prediction_flush_handler
====================================
Triggered by EventBridge hourly (+5 min after Lambda B).

Flow:
  1. LRANGE aqi:pred_buffer 0 -1   → read all buffered prediction records
  2. DEL aqi:pred_buffer            → atomically clear the buffer
  3. Write records as Parquet       → s3://weather-bulk/predictions/year=YYYY/month=MM/
  4. INSERT INTO aqi_db.predictions → register in Iceberg via Athena

Env vars required:
  REDIS_URL       — redis://default:...@host:port
  S3_BUCKET       — weather-bulk
  ATHENA_DB       — aqi_db
  ATHENA_RESULTS  — s3://weather-bulk/athena-results/
  AWS_REGION      — us-east-1 (default)
"""

import io
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone

import boto3
import redis as redis_lib

logger = logging.getLogger()
logger.setLevel(logging.INFO)

REDIS_URL      = os.environ["REDIS_URL"]
S3_BUCKET      = os.environ.get("S3_BUCKET", "weather-bulk")
ATHENA_DB      = os.environ.get("ATHENA_DB", "aqi_db")
ATHENA_RESULTS = os.environ.get("ATHENA_RESULTS", f"s3://{S3_BUCKET}/athena-results/")
REGION         = os.environ.get("AWS_REGION", "us-east-1")

_redis_client = None


def _get_redis():
    global _redis_client
    if _redis_client is None:
        _redis_client = redis_lib.from_url(REDIS_URL, decode_responses=True, socket_timeout=10)
    return _redis_client


def _athena_run(client, sql: str, label: str) -> None:
    exec_id = client.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": ATHENA_DB, "Catalog": "AwsDataCatalog"},
        ResultConfiguration={"OutputLocation": ATHENA_RESULTS},
        WorkGroup="primary",
    )["QueryExecutionId"]
    logger.info(f"{label} → {exec_id}")
    for _ in range(60):
        resp  = client.get_query_execution(QueryExecutionId=exec_id)
        state = resp["QueryExecution"]["Status"]["State"]
        if state == "SUCCEEDED":
            logger.info(f"✅ {label}")
            return
        if state in ("FAILED", "CANCELLED"):
            reason = resp["QueryExecution"]["Status"].get("StateChangeReason", "")
            raise RuntimeError(f"Athena {label} {state}: {reason}")
        time.sleep(5)
    raise TimeoutError(f"Athena {label} timed out")


def handler(event, context):
    r = _get_redis()

    # 1. Atomically read and clear the buffer
    raw_records = r.lrange("aqi:pred_buffer", 0, -1)
    if not raw_records:
        logger.info("pred_buffer empty — nothing to flush")
        return {"status": "ok", "flushed": 0}

    r.delete("aqi:pred_buffer")
    logger.info(f"Flushed {len(raw_records)} records from pred_buffer")

    # 2. Parse records
    records = []
    for raw in raw_records:
        try:
            records.append(json.loads(raw))
        except Exception as e:
            logger.warning(f"Skipping malformed record: {e}")

    if not records:
        return {"status": "ok", "flushed": 0}

    # 3. Write Parquet to S3 using pyarrow
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        now = datetime.now(timezone.utc)
        year_str  = now.strftime("%Y")
        month_str = now.strftime("%m")

        table = pa.table({
            "forecast_for": pa.array([r["forecast_for"] for r in records], type=pa.string()),
            "city_slug":    pa.array([r["city_slug"]    for r in records], type=pa.string()),
            "predicted":    pa.array([r["predicted"]    for r in records], type=pa.int32()),
            "confidence":   pa.array([float(r["confidence"]) for r in records], type=pa.float64()),
            "as_of":        pa.array([r["as_of"]        for r in records], type=pa.string()),
        })

        buf = io.BytesIO()
        pq.write_table(table, buf, compression="snappy")
        buf.seek(0)

        s3_key = (
            f"predictions/year={year_str}/month={month_str}/"
            f"pred_{now.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}.parquet"
        )
        s3 = boto3.client("s3", region_name=REGION)
        s3.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=buf.read())
        logger.info(f"Wrote s3://{S3_BUCKET}/{s3_key}")

    except Exception as e:
        logger.error(f"Failed to write Parquet to S3: {e}")
        raise

    # 4. INSERT INTO Athena predictions Iceberg table
    # Build VALUES clause — cast strings to TIMESTAMP
    values_rows = []
    for rec in records:
        forecast_for = rec["forecast_for"].replace("T", " ")[:19]
        as_of        = rec["as_of"].replace("T", " ")[:19]
        predicted    = int(rec["predicted"])
        confidence   = float(rec["confidence"])
        city_slug    = rec["city_slug"].replace("'", "''")  # escape single quotes
        values_rows.append(
            f"(TIMESTAMP '{forecast_for}', '{city_slug}', {predicted}, {confidence:.6f}, TIMESTAMP '{as_of}')"
        )

    insert_sql = (
        f"INSERT INTO {ATHENA_DB}.predictions (forecast_for, city_slug, predicted, confidence, as_of)\n"
        f"VALUES\n" + ",\n".join(values_rows)
    )

    try:
        athena = boto3.client("athena", region_name=REGION)
        _athena_run(athena, insert_sql, "insert predictions")
    except Exception as e:
        logger.error(f"Athena INSERT failed: {e}")
        raise

    return {"status": "ok", "flushed": len(records), "s3_key": s3_key}
