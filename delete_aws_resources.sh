#!/usr/bin/env bash
set -e

REGION="${AWS_REGION:-us-east-1}"

echo "Checking AWS credentials..."
if ! aws sts get-caller-identity > /dev/null; then
    echo "Error: AWS credentials expired or invalid. Please refresh them first."
    exit 1
fi
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo "Deleting EventBridge targets and rules..."
aws events remove-targets --rule "aqi-hourly-trigger" --ids "aqi-lambda" --region "$REGION" 2>/dev/null || true
aws events delete-rule --name "aqi-hourly-trigger" --region "$REGION" 2>/dev/null || true

aws events remove-targets --rule "aqi-prediction-flush-hourly" --ids "lambda-c" --region "$REGION" 2>/dev/null || true
aws events delete-rule --name "aqi-prediction-flush-hourly" --region "$REGION" 2>/dev/null || true

echo "Deleting Lambda functions..."
aws lambda delete-function --function-name "aqi-hourly-ingest" --region "$REGION" 2>/dev/null || true
aws lambda delete-function --function-name "aqi-iceberg-merge" --region "$REGION" 2>/dev/null || true
aws lambda delete-function --function-name "aqi-prediction-flush" --region "$REGION" 2>/dev/null || true

echo "Deleting IAM Roles and policies..."
for role in "aqi-lambda-role" "aqi-merge-lambda-role"; do
    aws iam detach-role-policy --role-name "$role" --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole 2>/dev/null || true
    
    POLICIES=$(aws iam list-role-policies --role-name "$role" --query PolicyNames --output text 2>/dev/null || echo "")
    for policy in $POLICIES; do
        aws iam delete-role-policy --role-name "$role" --policy-name "$policy" 2>/dev/null || true
    done
    
    aws iam delete-role --role-name "$role" 2>/dev/null || true
done

echo "Deleting Athena Database..."
aws athena start-query-execution \
    --query-string "DROP DATABASE IF EXISTS aqi_db CASCADE;" \
    --result-configuration OutputLocation="s3://weather-bulk/athena-results-cleanup/" \
    --work-group primary \
    --region "$REGION" > /dev/null 2>&1 || true

echo "Waiting for Athena to process..."
sleep 5

echo "Emptying and deleting S3 Bucket weather-bulk..."
aws s3 rm s3://weather-bulk --recursive 2>/dev/null || true
aws s3api delete-bucket --bucket weather-bulk --region "$REGION" 2>/dev/null || true

echo "AWS resources deletion completed."
