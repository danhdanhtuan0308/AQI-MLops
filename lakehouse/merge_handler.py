"""Lambda B — triggered by Lambda A with the S3 key of a new hourly file.
Runs MERGE INTO aqi_unified targeting only that file (99 rows, ~50 KB).
"""
import json
import logging
import os
import time

import boto3

log = logging.getLogger()
log.setLevel(logging.INFO)

ATHENA_DB      = os.environ.get("ATHENA_DB",      "aqi_db")
ATHENA_TABLE   = os.environ.get("ATHENA_TABLE",   "aqi_unified")
ATHENA_RESULTS = os.environ.get("ATHENA_RESULTS", "s3://weather-bulk/athena-results/")
ATHENA_WG      = os.environ.get("ATHENA_WG",      "primary")
S3_BUCKET      = os.environ.get("S3_BUCKET",      "weather-bulk")

athena = boto3.client("athena", region_name=os.environ.get("AWS_REGION", "us-east-1"))


def run_query(sql: str) -> str:
    """Start Athena query and return execution ID."""
    resp = athena.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": ATHENA_DB, "Catalog": "AwsDataCatalog"},
        ResultConfiguration={"OutputLocation": ATHENA_RESULTS},
        WorkGroup=ATHENA_WG,
    )
    return resp["QueryExecutionId"]


def wait_query(execution_id: str, timeout: int = 600) -> str:
    """Poll until query succeeds/fails. Returns final state."""
    deadline = time.time() + timeout
    delay = 2
    while time.time() < deadline:
        resp = athena.get_query_execution(QueryExecutionId=execution_id)
        state = resp["QueryExecution"]["Status"]["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            if state != "SUCCEEDED":
                reason = resp["QueryExecution"]["Status"].get("StateChangeReason", "")
                log.error("Query %s — %s: %s", execution_id, state, reason)
            return state
        time.sleep(delay)
        delay = min(delay * 2, 30)  # exponential backoff, max 30s
    raise TimeoutError(f"Athena query {execution_id} timed out after {timeout}s")


def parse_partition(s3_key: str) -> tuple[str, str, str]:
    """Extract year, month, day from hourly S3 key.

    Key format: hourly/year=YYYY/month=MM/day=DD/aqi_YYYY-MM-DD_HH.parquet
    """
    parts = {}
    for segment in s3_key.split("/"):
        if "=" in segment:
            k, v = segment.split("=", 1)
            parts[k] = v
    return parts["year"], parts["month"], parts["day"]


def build_add_partition_sql(s3_key: str) -> str:
    """ALTER TABLE to register the exact partition containing this file."""
    year, month, day = parse_partition(s3_key)
    partition_dir = f"s3://{S3_BUCKET}/hourly/year={year}/month={month}/day={day}"
    return (
        f"ALTER TABLE {ATHENA_DB}.raw_hourly "
        f"ADD IF NOT EXISTS "
        f"PARTITION (year='{year}', month='{month}', day='{day}') "
        f"LOCATION '{partition_dir}'"
    )


def build_merge_sql(s3_key: str) -> str:
    s3_path = f"s3://{S3_BUCKET}/{s3_key}"
    return f"""
MERGE INTO {ATHENA_DB}.{ATHENA_TABLE} AS t
USING (
    SELECT
        timestamp, city, city_slug, aqi,
        co, no, no2, o3, so2, pm2_5, pm10, nh3,
        'hourly' AS source
    FROM {ATHENA_DB}.raw_hourly
    WHERE "$path" = '{s3_path}'
) AS s
ON t.city_slug = s.city_slug
   AND t.timestamp = s.timestamp
WHEN MATCHED THEN
    UPDATE SET
        aqi    = s.aqi,
        co     = s.co,
        no     = s.no,
        no2    = s.no2,
        o3     = s.o3,
        so2    = s.so2,
        pm2_5  = s.pm2_5,
        pm10   = s.pm10,
        nh3    = s.nh3,
        source = s.source
WHEN NOT MATCHED THEN
    INSERT (timestamp, city, city_slug, aqi, co, no, no2, o3, so2, pm2_5, pm10, nh3, source)
    VALUES (s.timestamp, s.city, s.city_slug, s.aqi, s.co, s.no, s.no2, s.o3,
            s.so2, s.pm2_5, s.pm10, s.nh3, 'hourly')
"""


def handler(event, context):
    s3_key = event.get("s3_key")
    if not s3_key:
        log.error("No s3_key in event: %s", event)
        return {"statusCode": 400, "body": "missing s3_key"}

    log.info("MERGE from s3://%s/%s", S3_BUCKET, s3_key)

    # Register the exact partition for this file (ALTER TABLE is reliable; MSCK REPAIR can miss new partitions)
    add_part_sql = build_add_partition_sql(s3_key)
    log.info("Adding partition: %s", add_part_sql)
    part_id = run_query(add_part_sql)
    part_state = wait_query(part_id)
    log.info("ADD PARTITION → %s", part_state)

    sql = build_merge_sql(s3_key)
    exec_id = run_query(sql)
    log.info("Athena query started: %s", exec_id)

    state = wait_query(exec_id)
    log.info("Athena query %s → %s", exec_id, state)

    # Proactively warm Redis cache — one bulk Athena query for all cities
    predict_url = os.environ.get("PREDICT_API_URL", "").rstrip("/")
    if predict_url and state == "SUCCEEDED":
        try:
            import urllib.request
            req = urllib.request.Request(
                f"{predict_url}/warm-cache",
                method="POST",
                headers={"Content-Type": "application/json"},
                data=b"{}",
            )
            urllib.request.urlopen(req, timeout=10)
            log.info("Cache warm triggered at %s/warm-cache", predict_url)
        except Exception as e:
            log.warning("Cache warm failed (non-fatal): %s", e)

    return {
        "statusCode": 200 if state == "SUCCEEDED" else 500,
        "body": json.dumps({"state": state, "execution_id": exec_id, "s3_key": s3_key}),
    }
