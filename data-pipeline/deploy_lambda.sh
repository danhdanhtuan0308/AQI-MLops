#!/usr/bin/env bash
# deploy_lambda.sh — creates Lambda + EventBridge hourly trigger for AQI ingestion
set -euo pipefail

# load .env
source <(grep -v '^#' .env | sed 's/^/export /')

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

FUNCTION_NAME="aqi-hourly-ingest"
ROLE_NAME="aqi-lambda-role"
RULE_NAME="aqi-hourly-trigger"
S3_BUCKET="weather-bulk"
S3_PREFIX="hourly"
S3_ZIP_KEY="lambda-deploy/aqi-lambda.zip"   # staging area for the zip
RUNTIME="python3.12"
BUILD_DIR="/tmp/aqi-lambda-build"
ZIP_FILE="/tmp/aqi-lambda.zip"

echo "── Account: $ACCOUNT_ID  Region: $REGION"

# 1. build package
echo "── Building package..."
rm -rf "$BUILD_DIR" && mkdir -p "$BUILD_DIR"
cp data-pipeline/lambda/handler.py "$BUILD_DIR/"

pip install requests pandas pyarrow \
    --target "$BUILD_DIR" \
    --python-version 3.12 \
    --only-binary=:all: \
    --platform manylinux2014_x86_64 \
    --quiet

# Strip pyarrow test/doc/misc data to shrink zip
find "$BUILD_DIR" -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true
find "$BUILD_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$BUILD_DIR" -name "*.pyi" -delete 2>/dev/null || true
# Remove pyarrow datasets/flight/gandiva (heavy, unused)
rm -rf "$BUILD_DIR"/pyarrow/tests \
       "$BUILD_DIR"/pyarrow/gandiva* \
       "$BUILD_DIR"/pyarrow/dataset* \
       "$BUILD_DIR"/pyarrow/flight* \
       "$BUILD_DIR"/pyarrow/plasma* 2>/dev/null || true

cd "$BUILD_DIR" && zip -r "$ZIP_FILE" . -x "*.pyc" -x "*/__pycache__/*" > /dev/null
cd - > /dev/null
echo "   Package: $ZIP_FILE ($(du -sh $ZIP_FILE | cut -f1))"

# 2. upload zip to S3 (bypasses 50MB direct-upload limit)
echo "── Uploading package to s3://$S3_BUCKET/$S3_ZIP_KEY ..."
aws s3 cp "$ZIP_FILE" "s3://$S3_BUCKET/$S3_ZIP_KEY" --region "$REGION"
echo "   Package uploaded"

# 3. IAM role
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"
if ! aws iam get-role --role-name "$ROLE_NAME" &>/dev/null; then
    echo "── Creating IAM role: $ROLE_NAME..."
    aws iam create-role \
        --role-name "$ROLE_NAME" \
        --assume-role-policy-document '{
            "Version":"2012-10-17",
            "Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},
            "Action":"sts:AssumeRole"}]
        }' > /dev/null

    aws iam attach-role-policy \
        --role-name "$ROLE_NAME" \
        --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

    aws iam put-role-policy \
        --role-name "$ROLE_NAME" \
        --policy-name "aqi-s3-write" \
        --policy-document "{
            \"Version\":\"2012-10-17\",
            \"Statement\":[{\"Effect\":\"Allow\",\"Action\":[\"s3:PutObject\",\"s3:GetObject\"],
            \"Resource\":[\"arn:aws:s3:::weather-bulk/hourly/*\",
                         \"arn:aws:s3:::weather-bulk/lambda-deploy/*\"]}]
        }"
    echo "   Waiting for role propagation..."
    sleep 15
else
    echo "── IAM role already exists: $ROLE_NAME"
fi

# ── 3. Create or update Lambda ─────────────────────────────────────────────────
ENV_VARS="Variables={OWM_API_KEY=${OWM_API_KEY},S3_BUCKET=${S3_BUCKET},S3_PREFIX=${S3_PREFIX},SLEEP_SEC=0.2}"

if aws lambda get-function --function-name "$FUNCTION_NAME" --region "$REGION" &>/dev/null; then
    echo "── Updating Lambda function..."
    aws lambda update-function-code \
        --function-name "$FUNCTION_NAME" \
        --s3-bucket "$S3_BUCKET" \
        --s3-key "$S3_ZIP_KEY" \
        --region "$REGION" > /dev/null
    aws lambda wait function-updated \
        --function-name "$FUNCTION_NAME" \
        --region "$REGION"
    aws lambda update-function-configuration \
        --function-name "$FUNCTION_NAME" \
        --environment "$ENV_VARS" \
        --region "$REGION" > /dev/null
else
    echo "── Creating Lambda function: $FUNCTION_NAME..."
    aws lambda create-function \
        --function-name "$FUNCTION_NAME" \
        --runtime "$RUNTIME" \
        --role "$ROLE_ARN" \
        --handler "handler.handler" \
        --code "S3Bucket=$S3_BUCKET,S3Key=$S3_ZIP_KEY" \
        --timeout 300 \
        --memory-size 512 \
        --environment "$ENV_VARS" \
        --region "$REGION" > /dev/null
    aws lambda wait function-active \
        --function-name "$FUNCTION_NAME" \
        --region "$REGION"
fi
echo "   Lambda ready"

LAMBDA_ARN="arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${FUNCTION_NAME}"

# 5. EventBridge rule — every hour
echo "── Creating EventBridge rule: $RULE_NAME..."
RULE_ARN=$(aws events put-rule \
    --name "$RULE_NAME" \
    --schedule-expression "rate(1 hour)" \
    --state ENABLED \
    --description "Trigger AQI hourly ingest for 99 cities" \
    --region "$REGION" \
    --query RuleArn --output text)
echo "   Rule ARN: $RULE_ARN"

# 6. EventBridge → Lambda permission
aws lambda remove-permission \
    --function-name "$FUNCTION_NAME" \
    --statement-id "eventbridge-hourly" \
    --region "$REGION" 2>/dev/null || true

aws lambda add-permission \
    --function-name "$FUNCTION_NAME" \
    --statement-id "eventbridge-hourly" \
    --action "lambda:InvokeFunction" \
    --principal "events.amazonaws.com" \
    --source-arn "$RULE_ARN" \
    --region "$REGION" > /dev/null

# 7. set Lambda as EventBridge target
aws events put-targets \
    --rule "$RULE_NAME" \
    --targets "Id=aqi-lambda,Arn=${LAMBDA_ARN}" \
    --region "$REGION" > /dev/null

echo ""
echo "Deployment complete!"
echo "   Lambda  : $LAMBDA_ARN"
echo "   Trigger : every hour  (rate(1 hour))"
echo "   S3 path : s3://${S3_BUCKET}/${S3_PREFIX}/year=YYYY/month=MM/day=DD/aqi_YYYY-MM-DD_HH.parquet"
echo ""
echo "── Test invoke now:"
echo "   aws lambda invoke --function-name $FUNCTION_NAME --region $REGION /tmp/aqi-test-out.json && cat /tmp/aqi-test-out.json"
