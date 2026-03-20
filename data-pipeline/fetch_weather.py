import logging
import os
import time
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import yaml
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

OWM_API_KEY = os.environ["OWM_API_KEY"]
AWS_REGION  = os.getenv("AWS_REGION", "us-east-1")

with open(Path(__file__).parent / "config.yaml") as f:
    CFG = yaml.safe_load(f)

START_DT    = datetime.fromisoformat(CFG["start_dt"]).replace(tzinfo=timezone.utc)
END_DT      = datetime.fromisoformat(CFG["end_dt"]).replace(tzinfo=timezone.utc)
CHUNK_DAYS  = CFG["chunk_days"]
SLEEP_SEC   = CFG["sleep_sec"]
S3_BUCKET   = CFG["s3_bucket"]
S3_PREFIX   = CFG["s3_prefix"]
HISTORY_URL = CFG["history_url"]
AQI_URL     = CFG["aqi_url"]
CITIES      = CFG["cities"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

WEATHER_PLAN_OK = True  # flipped to False on first 401 — plan-level, not transient


def build_chunks() -> list[tuple[int, int]]:
    chunks, cursor = [], START_DT
    while cursor < END_DT:
        end = min(cursor + timedelta(days=CHUNK_DAYS), END_DT)
        chunks.append((int(cursor.timestamp()), int(end.timestamp())))
        cursor = end
    return chunks


def fetch_weather(lat: float, lon: float, start: int, end: int) -> list[dict]:
    r = requests.get(
        HISTORY_URL,
        params={"lat": lat, "lon": lon, "type": "hour",
                "start": start, "end": end, "units": "metric", "appid": OWM_API_KEY},
        timeout=30,
    )
    r.raise_for_status()
    return r.json().get("list", [])


def fetch_aqi(lat: float, lon: float, start: int, end: int) -> list[dict]:
    r = requests.get(
        AQI_URL,
        params={"lat": lat, "lon": lon, "start": start, "end": end, "appid": OWM_API_KEY},
        timeout=30,
    )
    r.raise_for_status()
    return r.json().get("list", [])


def parse_weather(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        m, wind = r.get("main", {}), r.get("wind", {})
        w        = r.get("weather", [{}])[0]
        temp, rh = m.get("temp"), m.get("humidity")
        dew = round(temp - ((100 - rh) / 5.0), 2) if temp is not None and rh else None
        rows.append({
            "dt":             r["dt"],
            "timestamp":      datetime.fromtimestamp(r["dt"], tz=timezone.utc),
            "temp_c":         temp,
            "feels_like_c":   m.get("feels_like"),
            "temp_min_c":     m.get("temp_min"),
            "temp_max_c":     m.get("temp_max"),
            "pressure_hpa":   m.get("pressure"),
            "sea_level_hpa":  m.get("sea_level"),
            "grnd_level_hpa": m.get("grnd_level"),
            "humidity_pct":   rh,
            "dew_point_c":    dew,
            "wind_speed_ms":  wind.get("speed"),
            "wind_deg":       wind.get("deg"),
            "wind_gust_ms":   wind.get("gust"),
            "clouds_pct":     r.get("clouds", {}).get("all"),
            "visibility_m":   r.get("visibility"),
            "rain_1h_mm":     r.get("rain", {}).get("1h", 0.0),
            "snow_1h_mm":     r.get("snow", {}).get("1h", 0.0),
            "weather_id":     w.get("id"),
            "weather_main":   w.get("main"),
            "weather_desc":   w.get("description"),
        })
    return pd.DataFrame(rows)


def parse_aqi(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        c = r.get("components", {})
        rows.append({
            "dt":        r["dt"],
            "timestamp": datetime.fromtimestamp(r["dt"], tz=timezone.utc),
            "aqi":       r.get("main", {}).get("aqi"),
            "co":    c.get("co"),
            "no":    c.get("no"),
            "no2":   c.get("no2"),
            "o3":    c.get("o3"),
            "so2":   c.get("so2"),
            "pm2_5": c.get("pm2_5"),
            "pm10":  c.get("pm10"),
            "nh3":   c.get("nh3"),
        })
    return pd.DataFrame(rows)


def ensure_bucket(s3, bucket: str) -> None:
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchBucket"):
            kwargs: dict = {"Bucket": bucket}
            if AWS_REGION != "us-east-1":
                kwargs["CreateBucketConfiguration"] = {"LocationConstraint": AWS_REGION}
            s3.create_bucket(**kwargs)
            log.info("Created bucket: %s", bucket)
        else:
            raise


def upload_parquet(s3, df: pd.DataFrame, bucket: str, key: str) -> None:
    buf = BytesIO()
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), buf, compression="snappy")
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    log.info("  uploaded s3://%s/%s  (%d rows)", bucket, key, len(df))


def fetch_city(city: list, chunks: list[tuple[int, int]]) -> pd.DataFrame | None:
    global WEATHER_PLAN_OK
    name, slug, lat, lon = city
    log.info("── %s", name)

    wx_frames, aqi_frames = [], []
    for start_ts, end_ts in chunks:
        if WEATHER_PLAN_OK:
            try:
                wx = fetch_weather(lat, lon, start_ts, end_ts)
                if wx:
                    wx_frames.append(parse_weather(wx))
            except requests.HTTPError as e:
                if e.response.status_code == 401:
                    log.warning("  [weather] HTTP 401 — History API requires a paid OWM plan. Skipping weather for all cities.")
                    WEATHER_PLAN_OK = False
                else:
                    log.warning("  [weather] %s skipped: HTTP %s", name, e.response.status_code)
            time.sleep(SLEEP_SEC)

        try:
            aqi = fetch_aqi(lat, lon, start_ts, end_ts)
            if aqi:
                aqi_frames.append(parse_aqi(aqi))
                log.info("  [aqi] %s  chunk %d records", name, len(aqi))
        except requests.HTTPError as e:
            log.warning("  [aqi] %s skipped: HTTP %s", name, e.response.status_code)
        time.sleep(SLEEP_SEC)

    if not aqi_frames and not wx_frames:
        log.warning("  No data at all for %s — skipping.", name)
        return None

    if wx_frames and aqi_frames:
        df_wx  = pd.concat(wx_frames).drop_duplicates("dt").sort_values("dt").reset_index(drop=True)
        df_aqi = pd.concat(aqi_frames).drop_duplicates("dt")
        df = df_wx.merge(df_aqi, on="dt", how="left")
    elif aqi_frames:
        df = pd.concat(aqi_frames).drop_duplicates("dt").sort_values("dt").reset_index(drop=True)
    else:
        df = pd.concat(wx_frames).drop_duplicates("dt").sort_values("dt").reset_index(drop=True)

    df["city"]      = name
    df["city_slug"] = slug
    return df.sort_values("dt").reset_index(drop=True)


def run_pipeline() -> None:
    log.info("Pipeline start: %s → %s | %d cities", START_DT.date(), END_DT.date(), len(CITIES))
    s3 = boto3.client("s3", region_name=AWS_REGION)
    ensure_bucket(s3, S3_BUCKET)
    chunks = build_chunks()

    for city in CITIES:
        df = fetch_city(city, chunks)
        if df is None:
            continue
        slug = city[1]
        df["_ym"] = df["timestamp"].dt.to_period("M").astype(str)
        for ym, grp in df.groupby("_ym"):
            key = f"{S3_PREFIX}/{slug}/year={ym[:4]}/month={ym[5:]}/weather_aqi_{ym}.parquet"
            upload_parquet(s3, grp.drop(columns="_ym"), S3_BUCKET, key)

    if not WEATHER_PLAN_OK:
        log.warning("Collected AQI-only data. Upgrade to OWM Professional plan to include weather history.")
    log.info("Pipeline complete.")


if __name__ == "__main__":
    run_pipeline()
