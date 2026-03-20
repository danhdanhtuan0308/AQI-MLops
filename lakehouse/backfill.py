"""
backfill.py — ONE-TIME script
Reads all data-pipeline/ parquet files directly with pandas (handles mixed types),
writes a single clean consolidated file to S3, then Athena inserts it into aqi_unified.
Run once after deploy_lakehouse.sh has been executed.
Usage:  uv run python lakehouse/backfill.py
"""
import io
import os
import time
import logging
from dotenv import load_dotenv
import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BUCKET         = "weather-bulk"
ATHENA_DB      = "aqi_db"
ATHENA_RESULTS = f"s3://{BUCKET}/athena-results/"
CLEAN_KEY      = "processed/backfill_clean/pipeline_clean.parquet"
REGION         = os.getenv("AWS_REGION", "us-east-1")

s3 = boto3.client("s3", region_name=REGION,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))
athena = boto3.client("athena", region_name=REGION,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))


def list_keys(prefix):
    keys, token = [], None
    while True:
        kwargs = dict(Bucket=BUCKET, Prefix=prefix)
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        keys += [o["Key"] for o in resp.get("Contents", []) if o["Key"].endswith(".parquet")]
        if not resp.get("IsTruncated"):
            break
        token = resp["NextContinuationToken"]
    return keys


def athena_run(sql, label):
    resp = athena.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": ATHENA_DB, "Catalog": "AwsDataCatalog"},
        ResultConfiguration={"OutputLocation": ATHENA_RESULTS},
        WorkGroup="primary")
    eid = resp["QueryExecutionId"]
    log.info("  %s → %s", label, eid)
    delay = 5
    for _ in range(120):
        r = athena.get_query_execution(QueryExecutionId=eid)
        state = r["QueryExecution"]["Status"]["State"]
        if state == "SUCCEEDED":
            scanned = r["QueryExecution"].get("Statistics", {}).get("DataScannedInBytes", 0)
            log.info("  ✅ %s  (%.1f MB scanned)", label, scanned / 1e6)
            return eid
        if state in ("FAILED", "CANCELLED"):
            log.error("  ❌ %s: %s", label, r["QueryExecution"]["Status"].get("StateChangeReason", ""))
            return None
        time.sleep(delay)
        delay = min(delay * 2, 30)


# Step 1: read all data-pipeline/ files
log.info("Reading all data-pipeline/ parquet files from S3...")
pipeline_keys = list_keys("data-pipeline/")
log.info("Found %d pipeline files", len(pipeline_keys))

dfs = []
for i, key in enumerate(pipeline_keys):
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    df = pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()
    dfs.append(df)
    if (i + 1) % 50 == 0:
        log.info("  Read %d / %d files...", i + 1, len(pipeline_keys))

pipeline_df = pd.concat(dfs, ignore_index=True)
log.info("Total pipeline rows: %d", len(pipeline_df))

# Step 2: normalise types
COLS = ["timestamp", "city", "city_slug", "aqi", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]
for c in COLS:
    if c not in pipeline_df.columns:
        pipeline_df[c] = None

pipeline_df["timestamp"] = pd.to_datetime(pipeline_df["timestamp"], utc=True)
pipeline_df["aqi"]       = pd.to_numeric(pipeline_df["aqi"], errors="coerce").round().astype("Int32")
for c in ["co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]:
    pipeline_df[c] = pd.to_numeric(pipeline_df[c], errors="coerce")
pipeline_df["source"] = "pipeline"

clean_df = pipeline_df[COLS + ["source"]].drop_duplicates(
    subset=["city_slug", "timestamp"]
).dropna(subset=["timestamp", "city_slug"])

log.info("Clean pipeline rows (deduplicated): %d", len(clean_df))

# Step 3: write clean parquet to S3
log.info("Uploading clean file to s3://%s/%s ...", BUCKET, CLEAN_KEY)
schema = pa.schema([
    ("timestamp", pa.timestamp("us", tz="UTC")),
    ("city",      pa.string()),
    ("city_slug", pa.string()),
    ("aqi",       pa.int32()),
    ("co",        pa.float64()), ("no",   pa.float64()), ("no2",  pa.float64()),
    ("o3",        pa.float64()), ("so2",  pa.float64()), ("pm2_5",pa.float64()),
    ("pm10",      pa.float64()), ("nh3",  pa.float64()),
    ("source",    pa.string()),
])
table = pa.Table.from_pandas(clean_df, schema=schema, preserve_index=False)
buf = io.BytesIO()
pq.write_table(table, buf, compression="snappy")
buf.seek(0)
s3.put_object(Bucket=BUCKET, Key=CLEAN_KEY, Body=buf.getvalue())
log.info("  ✅ Uploaded (%.1f MB)", buf.tell() / 1e6)

# Step 4: register as Athena external table
athena_run("DROP TABLE IF EXISTS aqi_db.raw_pipeline_clean", "drop old clean table")
athena_run(f"""
CREATE EXTERNAL TABLE aqi_db.raw_pipeline_clean (
  timestamp TIMESTAMP, city STRING, city_slug STRING, aqi INT,
  co DOUBLE, no DOUBLE, no2 DOUBLE, o3 DOUBLE,
  so2 DOUBLE, pm2_5 DOUBLE, pm10 DOUBLE, nh3 DOUBLE, source STRING
)
STORED AS PARQUET
LOCATION 's3://{BUCKET}/processed/backfill_clean/'
TBLPROPERTIES ('parquet.compress'='SNAPPY')
""", "create raw_pipeline_clean table")

# Step 5: INSERT into aqi_unified
athena_run("""
INSERT INTO aqi_db.aqi_unified
    (timestamp, city, city_slug, aqi, co, no, no2, o3, so2, pm2_5, pm10, nh3, source)
SELECT timestamp, city, city_slug, aqi, co, no, no2, o3, so2, pm2_5, pm10, nh3, source
FROM aqi_db.raw_pipeline_clean
""", "INSERT pipeline → aqi_unified")

# Step 6: verify row counts
eid = athena_run(
    "SELECT source, COUNT(*) AS rows FROM aqi_db.aqi_unified GROUP BY source",
    "final row count"
)
if eid:
    time.sleep(5)
    res = athena.get_query_results(QueryExecutionId=eid)
    log.info("\n── aqi_unified row counts:")
    for row in res["ResultSet"]["Rows"][1:]:
        vals = [c.get("VarCharValue", "") for c in row["Data"]]
        log.info("   %-12s %s rows", vals[0], vals[1])

log.info("Backfill complete ✅")
