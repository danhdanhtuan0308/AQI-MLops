#!/usr/bin/env bash
# deploy_lakehouse.sh
# 1. Runs setup.sql in Athena (create DB, external tables, Iceberg table)
# 2. Deploys Lambda B (aqi-iceberg-merge)
# 3. Updates Lambda A IAM role to allow invoking Lambda B
# 4. Updates Lambda A code to invoke Lambda B after each hourly write
set -euo pipefail

source <(grep -v '^#' .env | grep '=' | sed 's/^/export /')

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

LAMBDA_A="aqi-hourly-ingest"
LAMBDA_B="aqi-iceberg-merge"
ROLE_A="aqi-lambda-role"         # existing Lambda A role
ROLE_B="aqi-merge-lambda-role"   # new role for Lambda B
S3_BUCKET="weather-bulk"
ATHENA_RESULTS="s3://${S3_BUCKET}/athena-results/"
BUILD_B="/tmp/aqi-merge-build"
ZIP_B="/tmp/aqi-merge.zip"

echo "── Account: $ACCOUNT_ID  Region: $REGION"

# helper: run Athena query and wait
athena_run() {
    local sql="$1"
    local label="${2:-query}"
    local exec_id
    exec_id=$(aws athena start-query-execution \
        --query-string "$sql" \
        --query-execution-context Database=aqi_db,Catalog=AwsDataCatalog \
        --result-configuration OutputLocation="$ATHENA_RESULTS" \
        --work-group primary \
        --region "$REGION" \
        --query QueryExecutionId --output text)
    echo "   $label → $exec_id"
    for i in $(seq 1 60); do
        state=$(aws athena get-query-execution \
            --query-execution-id "$exec_id" \
            --region "$REGION" \
            --query "QueryExecution.Status.State" --output text)
        if [[ "$state" == "SUCCEEDED" ]]; then echo "   ✅ $label"; return 0; fi
        if [[ "$state" == "FAILED" || "$state" == "CANCELLED" ]]; then
            reason=$(aws athena get-query-execution \
                --query-execution-id "$exec_id" \
                --region "$REGION" \
                --query "QueryExecution.Status.StateChangeReason" --output text 2>/dev/null || echo "")
            echo "   ❌ $label — $state: $reason"; return 1
        fi
        sleep 5
    done
    echo "   ⏱ $label timed out"; return 1
}

# 1. ensure Athena results prefix exists
echo "── Ensuring Athena results prefix exists..."
aws s3api put-object --bucket "$S3_BUCKET" --key "athena-results/" --region "$REGION" > /dev/null || true

# 2. run Athena setup (create DB + tables)
echo "── Running Athena setup..."

athena_run "CREATE DATABASE IF NOT EXISTS aqi_db COMMENT 'AQI MLOps lakehouse'" "create database"

athena_run "$(cat <<'EOF'
CREATE EXTERNAL TABLE IF NOT EXISTS aqi_db.raw_pipeline (
  dt BIGINT, timestamp TIMESTAMP, aqi DOUBLE,
  co DOUBLE, no DOUBLE, no2 DOUBLE, o3 DOUBLE,
  so2 DOUBLE, pm2_5 DOUBLE, pm10 DOUBLE, nh3 DOUBLE,
  city STRING, city_slug STRING
)
STORED AS PARQUET
LOCATION 's3://weather-bulk/data-pipeline/'
TBLPROPERTIES ('parquet.compress'='SNAPPY')
EOF
)" "create raw_pipeline table"

athena_run "$(cat <<'EOF'
CREATE EXTERNAL TABLE IF NOT EXISTS aqi_db.raw_hourly (
  dt BIGINT, timestamp TIMESTAMP,
  city STRING, city_slug STRING, aqi INT,
  co DOUBLE, no DOUBLE, no2 DOUBLE, o3 DOUBLE,
  so2 DOUBLE, pm2_5 DOUBLE, pm10 DOUBLE, nh3 DOUBLE
)
PARTITIONED BY (year STRING, month STRING, day STRING)
STORED AS PARQUET
LOCATION 's3://weather-bulk/hourly/'
TBLPROPERTIES ('parquet.compress'='SNAPPY')
EOF
)" "create raw_hourly table"

athena_run "MSCK REPAIR TABLE aqi_db.raw_hourly" "repair hourly partitions"

athena_run "$(cat <<'EOF'
CREATE TABLE IF NOT EXISTS aqi_db.aqi_unified (
  timestamp TIMESTAMP, city STRING, city_slug STRING, aqi INT,
  co DOUBLE, no DOUBLE, no2 DOUBLE, o3 DOUBLE,
  so2 DOUBLE, pm2_5 DOUBLE, pm10 DOUBLE, nh3 DOUBLE, source STRING
)
PARTITIONED BY (month(timestamp))
LOCATION 's3://weather-bulk/processed/aqi_unified/'
TBLPROPERTIES ('table_type'='ICEBERG','format'='parquet','write_compression'='snappy')
EOF
)" "create aqi_unified Iceberg table"

echo "── Athena setup complete"

# 3. IAM role for Lambda B
ROLE_B_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_B}"
if ! aws iam get-role --role-name "$ROLE_B" &>/dev/null; then
    echo "── Creating IAM role: $ROLE_B..."
    aws iam create-role --role-name "$ROLE_B" \
        --assume-role-policy-document '{
            "Version":"2012-10-17",
            "Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},
            "Action":"sts:AssumeRole"}]
        }' > /dev/null
    aws iam attach-role-policy --role-name "$ROLE_B" \
        --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
    aws iam put-role-policy --role-name "$ROLE_B" \
        --policy-name "aqi-merge-policy" \
        --policy-document "{
            \"Version\":\"2012-10-17\",
            \"Statement\":[
                {\"Effect\":\"Allow\",
                 \"Action\":[\"athena:StartQueryExecution\",\"athena:GetQueryExecution\",\"athena:GetQueryResults\"],
                 \"Resource\":\"*\"},
                {\"Effect\":\"Allow\",
                 \"Action\":[\"s3:GetObject\",\"s3:PutObject\",\"s3:DeleteObject\",\"s3:ListBucket\"],
                 \"Resource\":[\"arn:aws:s3:::${S3_BUCKET}\",\"arn:aws:s3:::${S3_BUCKET}/*\"]},
                {\"Effect\":\"Allow\",
                 \"Action\":[\"glue:GetDatabase\",\"glue:GetTable\",\"glue:GetPartitions\",\"glue:UpdateTable\"],
                 \"Resource\":\"*\"}
            ]
        }"
    echo "   Waiting for role propagation..."
    sleep 15
else
    echo "── IAM role already exists: $ROLE_B"
fi

# 4. deploy Lambda B
echo "── Building Lambda B package..."
rm -rf "$BUILD_B" && mkdir -p "$BUILD_B"
cp lakehouse/merge_handler.py "$BUILD_B/handler.py"
cd "$BUILD_B" && zip -r "$ZIP_B" . > /dev/null && cd - > /dev/null
echo "   Package: $ZIP_B ($(du -sh $ZIP_B | cut -f1))"

ENV_B="Variables={ATHENA_DB=aqi_db,ATHENA_TABLE=aqi_unified,ATHENA_RESULTS=${ATHENA_RESULTS},S3_BUCKET=${S3_BUCKET}}"

if aws lambda get-function --function-name "$LAMBDA_B" --region "$REGION" &>/dev/null; then
    echo "── Updating Lambda B..."
    aws lambda update-function-code --function-name "$LAMBDA_B" \
        --zip-file "fileb://$ZIP_B" --region "$REGION" > /dev/null
    aws lambda wait function-updated --function-name "$LAMBDA_B" --region "$REGION"
    aws lambda update-function-configuration --function-name "$LAMBDA_B" \
        --environment "$ENV_B" --region "$REGION" > /dev/null
else
    echo "── Creating Lambda B: $LAMBDA_B..."
    aws lambda create-function \
        --function-name "$LAMBDA_B" \
        --runtime python3.12 \
        --role "$ROLE_B_ARN" \
        --handler "handler.handler" \
        --zip-file "fileb://$ZIP_B" \
        --timeout 600 \
        --memory-size 256 \
        --environment "$ENV_B" \
        --region "$REGION" > /dev/null
    aws lambda wait function-active --function-name "$LAMBDA_B" --region "$REGION"
fi
echo "   ✅ Lambda B ready"

# 5. grant Lambda A invoke permission on Lambda B
echo "── Granting Lambda A invoke permission on Lambda B..."
aws iam put-role-policy --role-name "$ROLE_A" \
    --policy-name "invoke-merge-lambda" \
    --policy-document "{
        \"Version\":\"2012-10-17\",
        \"Statement\":[{\"Effect\":\"Allow\",
        \"Action\":\"lambda:InvokeFunction\",
        \"Resource\":\"arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${LAMBDA_B}\"}]
    }"

echo ""
echo "✅ Lakehouse deployment complete!"
echo ""
echo "── Next: run the one-time backfill:"
echo "   uv run python lakehouse/backfill.py"
echo ""
echo "── From now on, every hour Lambda A writes hourly file → invokes Lambda B"
echo "   → MERGE INTO aqi_db.aqi_unified (only 99 rows scanned)"
echo ""
echo "── Query your unified table in Athena:"
echo "   SELECT source, COUNT(*) FROM aqi_db.aqi_unified GROUP BY source;"
