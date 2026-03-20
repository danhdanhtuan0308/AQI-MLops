import io
import json
import logging
import os
import time
from datetime import datetime, timezone

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

log = logging.getLogger()
log.setLevel(logging.INFO)

OWM_API_KEY = os.environ["OWM_API_KEY"]
S3_BUCKET   = os.environ.get("S3_BUCKET", "weather-bulk")
S3_PREFIX   = os.environ.get("S3_PREFIX", "hourly")
SLEEP_SEC   = float(os.environ.get("SLEEP_SEC", "0.2"))
AQI_URL     = "http://api.openweathermap.org/data/2.5/air_pollution"

# 99 cities inline — mirrors config.yaml
CITIES = [
    ["Tokyo","tokyo",35.6762,139.6503],["Delhi","delhi",28.6139,77.209],
    ["Shanghai","shanghai",31.2304,121.4737],["Sao Paulo","sao-paulo",-23.5505,-46.6333],
    ["Mexico City","mexico-city",19.4326,-99.1332],["Cairo","cairo",30.0444,31.2357],
    ["Mumbai","mumbai",19.076,72.8777],["Beijing","beijing",39.9042,116.4074],
    ["Dhaka","dhaka",23.8103,90.4125],["Osaka","osaka",34.6937,135.5023],
    ["New York","new-york",40.7128,-74.006],["Karachi","karachi",24.8607,67.0011],
    ["Buenos Aires","buenos-aires",-34.6037,-58.3816],["Chongqing","chongqing",29.4316,106.9123],
    ["Istanbul","istanbul",41.0082,28.9784],["Kolkata","kolkata",22.5726,88.3639],
    ["Manila","manila",14.5995,120.9842],["Lagos","lagos",6.5244,3.3792],
    ["Rio de Janeiro","rio-de-janeiro",-22.9068,-43.1729],["Tianjin","tianjin",39.3434,117.3616],
    ["Kinshasa","kinshasa",-4.4419,15.2663],["Guangzhou","guangzhou",23.1291,113.2644],
    ["Los Angeles","los-angeles",34.0522,-118.2437],["Moscow","moscow",55.7558,37.6173],
    ["Shenzhen","shenzhen",22.5431,114.0579],["Lahore","lahore",31.5204,74.3587],
    ["Bangalore","bangalore",12.9716,77.5946],["Paris","paris",48.8566,2.3522],
    ["Bogota","bogota",4.711,-74.0721],["Jakarta","jakarta",-6.2088,106.8456],
    ["Chennai","chennai",13.0827,80.2707],["Lima","lima",-12.0464,-77.0428],
    ["Bangkok","bangkok",13.7563,100.5018],["Seoul","seoul",37.5665,126.978],
    ["Nagoya","nagoya",35.1815,136.9066],["Hyderabad","hyderabad",17.385,78.4867],
    ["London","london",51.5074,-0.1278],["Tehran","tehran",35.6892,51.389],
    ["Chicago","chicago",41.8781,-87.6298],["Chengdu","chengdu",30.5728,104.0668],
    ["Nanjing","nanjing",32.0603,118.7969],["Wuhan","wuhan",30.5928,114.3055],
    ["Ho Chi Minh City","ho-chi-minh-city",10.8231,106.6297],["Luanda","luanda",-8.839,13.2894],
    ["Ahmedabad","ahmedabad",23.0225,72.5714],["Kuala Lumpur","kuala-lumpur",3.139,101.6869],
    ["Xian","xian",34.3416,108.9398],["Hong Kong","hong-kong",22.3193,114.1694],
    ["Dongguan","dongguan",23.0207,113.7519],["Hangzhou","hangzhou",30.2741,120.1551],
    ["Foshan","foshan",23.0219,113.1215],["Shenyang","shenyang",41.8057,123.4315],
    ["Riyadh","riyadh",24.7136,46.6753],["Baghdad","baghdad",33.3152,44.3661],
    ["Santiago","santiago",-33.4489,-70.6693],["Surat","surat",21.1702,72.8311],
    ["Madrid","madrid",40.4168,-3.7038],["Suzhou","suzhou",31.2989,120.5853],
    ["Pune","pune",18.5204,73.8567],["Harbin","harbin",45.8038,126.535],
    ["Houston","houston",29.7604,-95.3698],["Dallas","dallas",32.7767,-96.797],
    ["Toronto","toronto",43.6532,-79.3832],["Dar es Salaam","dar-es-salaam",-6.7924,39.2083],
    ["Miami","miami",25.7617,-80.1918],["Belo Horizonte","belo-horizonte",-19.9167,-43.9345],
    ["Singapore","singapore",1.3521,103.8198],["Philadelphia","philadelphia",39.9526,-75.1652],
    ["Atlanta","atlanta",33.749,-84.388],["Fukuoka","fukuoka",33.5902,130.4017],
    ["Khartoum","khartoum",15.5007,32.5599],["Barcelona","barcelona",41.3851,2.1734],
    ["Johannesburg","johannesburg",-26.2041,28.0473],["Saint Petersburg","saint-petersburg",59.9311,30.3609],
    ["Qingdao","qingdao",36.0671,120.3826],["Jeddah","jeddah",21.4858,39.1925],
    ["Abidjan","abidjan",5.36,-4.0083],["Zhengzhou","zhengzhou",34.7466,113.6253],
    ["Nairobi","nairobi",-1.2921,36.8219],["Alexandria","alexandria",31.2001,29.9187],
    ["Casablanca","casablanca",33.5731,-7.5898],["Kabul","kabul",34.5553,69.2075],
    ["Accra","accra",5.6037,-0.187],["Cape Town","cape-town",-33.9249,18.4241],
    ["Sydney","sydney",-33.8688,151.2093],["Melbourne","melbourne",-37.8136,144.9631],
    ["Rome","rome",41.9028,12.4964],["Berlin","berlin",52.52,13.405],
    ["Addis Ababa","addis-ababa",9.032,38.7469],["Yangon","yangon",16.8661,96.1951],
    ["Kathmandu","kathmandu",27.7172,85.324],["Ankara","ankara",39.9334,32.8597],
    ["Athens","athens",37.9838,23.7275],["Taipei","taipei",25.033,121.5654],
    ["Amsterdam","amsterdam",52.3676,4.9041],["Dubai","dubai",25.2048,55.2708],
    ["Caracas","caracas",10.4806,-66.9036],["Guadalajara","guadalajara",20.6597,-103.3496],
    ["Monterrey","monterrey",25.6866,-100.3161],
]


def fetch_current_aqi(lat: float, lon: float) -> dict | None:
    try:
        r = requests.get(
            AQI_URL,
            params={"lat": lat, "lon": lon, "appid": OWM_API_KEY},
            timeout=10,
        )
        r.raise_for_status()
        items = r.json().get("list", [])
        return items[0] if items else None
    except Exception as e:
        log.warning("AQI fetch failed for (%s, %s): %s", lat, lon, e)
        return None


def handler(event, context):
    now = datetime.now(tz=timezone.utc)
    s3 = boto3.client("s3")
    rows = []

    for name, slug, lat, lon in CITIES:
        record = fetch_current_aqi(lat, lon)
        if record:
            c = record.get("components", {})
            rows.append({
                "dt":        record["dt"],
                "timestamp": datetime.fromtimestamp(record["dt"], tz=timezone.utc),
                "city":      name,
                "city_slug": slug,
                "aqi":       record.get("main", {}).get("aqi"),
                "co":        c.get("co"),
                "no":        c.get("no"),
                "no2":       c.get("no2"),
                "o3":        c.get("o3"),
                "so2":       c.get("so2"),
                "pm2_5":     c.get("pm2_5"),
                "pm10":      c.get("pm10"),
                "nh3":       c.get("nh3"),
            })
            log.info("  ✓ %s  aqi=%s", name, record.get("main", {}).get("aqi"))
        time.sleep(SLEEP_SEC)

    if not rows:
        log.error("No data collected — aborting.")
        return {"statusCode": 500, "body": "no data"}

    df = pd.DataFrame(rows)
    ts  = now.strftime("%Y-%m-%d_%H")
    key = (f"{S3_PREFIX}/year={now.year}/month={now.month:02d}"
           f"/day={now.day:02d}/aqi_{ts}.parquet")

    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), buf, compression="snappy")
    buf.seek(0)
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())

    log.info("Uploaded s3://%s/%s  (%d cities)", S3_BUCKET, key, len(df))
    return {"statusCode": 200, "body": json.dumps({"cities": len(df), "key": key})}
