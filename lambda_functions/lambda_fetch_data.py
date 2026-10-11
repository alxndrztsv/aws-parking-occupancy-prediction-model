"""
Download one month of parking transactions from the city API into the bronze layer.

Invoked as the first step of the Step Functions pipeline. With no year/month in the
event it targets the previous calendar month, which is what the monthly EventBridge
schedule on the third day wants; passing year and month re-runs a specific period.

Uses only the standard library plus boto3, so the function ships without a
dependency layer.
"""

import calendar
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")
ssm = boto3.client("ssm")

BRONZE_BUCKET = os.environ["BRONZE_BUCKET"]
API_URL = os.environ["API_URL"]
API_CITY_CODE = os.environ["API_CITY_CODE"]

API_TIMEOUT_SECONDS = 60

API_LOGIN_SSM_PATH = os.environ.get("API_LOGIN_SSM_PATH", "")
API_PASSWORD_SSM_PATH = os.environ.get("API_PASSWORD_SSM_PATH", "")


def get_secrets():
    """Fetch API credentials from AWS SSM Parameter Store at runtime."""
    login = ssm.get_parameter(Name=API_LOGIN_SSM_PATH, WithDecryption=True)["Parameter"]["Value"]

    password = ssm.get_parameter(Name=API_PASSWORD_SSM_PATH, WithDecryption=True)["Parameter"][
        "Value"
    ]

    return login, password


def get_month_date_range(year=None, month=None):
    """Return (year, month, startdate, enddate) for one calendar month."""
    if year and month:
        first_day = datetime(year, month, 1, tzinfo=UTC)
        last_day_num = calendar.monthrange(year, month)[1]
        last_day = datetime(year, month, last_day_num, tzinfo=UTC)
    else:
        now = datetime.now(UTC)
        last_day = now.replace(day=1) - timedelta(days=1)
        first_day = last_day.replace(day=1)
        year, month = first_day.year, first_day.month

    return (
        year,
        month,
        first_day.strftime("%Y%m%d000000"),
        last_day.strftime("%Y%m%d235959"),
    )


def fetch_transactions(login, password, start_date, end_date):
    """POST the transaction_history report request and return the open response."""
    payload = urllib.parse.urlencode(
        {
            "login": login,
            "password": password,
            "report": "transaction_history",
            "startdate": start_date,
            "enddate": end_date,
            "city": API_CITY_CODE,
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        API_URL,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "Mozilla/5.0 (Compatible; AWS-Lambda-Ingestion/1.0)",
        },
    )

    return urllib.request.urlopen(request, timeout=API_TIMEOUT_SECONDS)


def lambda_handler(event, context):
    event = event or {}

    # Fetch credentials at runtime to ensure we always have the latest secrets
    try:
        API_LOGIN, API_PASSWORD = get_secrets()
    except Exception as e:
        logger.error(f"Failed to retrieve secrets from SSM: {e}")
        raise

    # Determine target month
    if "year" in event and "month" in event:
        year, month, start_date, end_date = get_month_date_range(
            int(event["year"]), int(event["month"])
        )
    else:
        year, month, start_date, end_date = get_month_date_range()

    # Generate Sagemaker training job name
    run_id = f"{year}{month:02d}-{datetime.now(UTC):%Y%m%d%H%M%S}"
    s3_key = f"year={year}/month={month:02d}/transactions.csv"

    logger.info("Fetching %s-%02d into s3://%s/%s", year, month, BRONZE_BUCKET, s3_key)

    try:
        with fetch_transactions(API_LOGIN, API_PASSWORD, start_date, end_date) as response:
            csv_data = response.read()
            s3.put_object(Bucket=BRONZE_BUCKET, Key=s3_key, Body=csv_data)

    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"API returned status {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error calling API: {exc.reason}") from exc

    return {
        "statusCode": 200,
        "year": str(year),
        "month": str(month),
        "run_id": run_id,
        "body": f"Successfully uploaded {s3_key}",
    }
