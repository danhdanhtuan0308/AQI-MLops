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

    sql = build_merge_sql(s3_key)
    exec_id = run_query(sql)
    log.info("Athena query started: %s", exec_id)

    state = wait_query(exec_id)
    log.info("Athena query %s → %s", exec_id, state)

    return {
        "statusCode": 200 if state == "SUCCEEDED" else 500,
        "body": json.dumps({"state": state, "execution_id": exec_id, "s3_key": s3_key}),
    }
