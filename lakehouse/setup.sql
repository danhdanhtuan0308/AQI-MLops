-- AQI Lakehouse Setup
-- Run sequentially in Athena, or: bash lakehouse/deploy_lakehouse.sh

-- Step 1: database
CREATE DATABASE IF NOT EXISTS aqi_db
COMMENT 'AQI MLOps lakehouse — raw + unified Iceberg table';


-- Step 2: Bronze — bulk pipeline (read-only)
CREATE EXTERNAL TABLE IF NOT EXISTS aqi_db.raw_pipeline (
  dt          BIGINT,
  timestamp   TIMESTAMP,
  aqi         DOUBLE,
  co          DOUBLE,
  no          DOUBLE,
  no2         DOUBLE,
  o3          DOUBLE,
  so2         DOUBLE,
  pm2_5       DOUBLE,
  pm10        DOUBLE,
  nh3         DOUBLE,
  city        STRING,
  city_slug   STRING
)
STORED AS PARQUET
LOCATION 's3://weather-bulk/data-pipeline/'
TBLPROPERTIES ('parquet.compress' = 'SNAPPY');


-- Step 3: Bronze — hourly live data (read-only, Hive-partitioned)
CREATE EXTERNAL TABLE IF NOT EXISTS aqi_db.raw_hourly (
  dt          BIGINT,
  timestamp   TIMESTAMP,
  city        STRING,
  city_slug   STRING,
  aqi         INT,
  co          DOUBLE,
  no          DOUBLE,
  no2         DOUBLE,
  o3          DOUBLE,
  so2         DOUBLE,
  pm2_5       DOUBLE,
  pm10        DOUBLE,
  nh3         DOUBLE
)
PARTITIONED BY (year STRING, month STRING, day STRING)
STORED AS PARQUET
LOCATION 's3://weather-bulk/hourly/'
TBLPROPERTIES ('parquet.compress' = 'SNAPPY');

MSCK REPAIR TABLE aqi_db.raw_hourly;


-- Step 4: Silver — unified Iceberg table (partitioned by month)
CREATE TABLE IF NOT EXISTS aqi_db.aqi_unified (
  timestamp   TIMESTAMP,
  city        STRING,
  city_slug   STRING,
  aqi         INT,
  co          DOUBLE,
  no          DOUBLE,
  no2         DOUBLE,
  o3          DOUBLE,
  so2         DOUBLE,
  pm2_5       DOUBLE,
  pm10        DOUBLE,
  nh3         DOUBLE,
  source      STRING
)
PARTITIONED BY (month(timestamp))
LOCATION 's3://weather-bulk/processed/aqi_unified/'
TBLPROPERTIES (
  'table_type'        = 'ICEBERG',
  'format'            = 'parquet',
  'write_compression' = 'snappy'
);


-- Step 5: Gold — model predictions (written by Lambda C hourly)
CREATE TABLE IF NOT EXISTS aqi_db.predictions (
  forecast_for  TIMESTAMP,
  city_slug     STRING,
  predicted     INT,
  confidence    DOUBLE,
  as_of         TIMESTAMP
)
PARTITIONED BY (month(forecast_for))
LOCATION 's3://weather-bulk/predictions/'
TBLPROPERTIES (
  'table_type'        = 'ICEBERG',
  'format'            = 'parquet',
  'write_compression' = 'snappy'
);
