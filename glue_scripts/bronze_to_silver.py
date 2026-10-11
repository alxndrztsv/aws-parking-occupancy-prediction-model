import sys
from typing import Final

from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql.functions import (
    broadcast,
    col,
    floor,
    lit,
    lower,
    month,
    to_timestamp,
    trim,
    when,
    year,
)

# Initialize Glue Context
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)

RAW_DATE_FORMAT: Final[str] = "yyyy-MM-dd HH:mm:ss.SSSSSS"

COLUMN_RENAME_MAP_RAW: Final[dict[str, str]] = {
    "PAYMENT_MEAN": "payment_mean",
    "METER_DATE": "start_date",
    "METER_CODE": "terminal_code",
    "PAID_DURATION": "paid_duration_mins",
    "TOTAL_DURATION": "total_duration_mins",
    "SYSTEM_ID": "system_id",
    "PRINTED_ID": "printed_id",
    "CARD_NAME": "card_type",
    "ZONE_DESC": "zone_desc",
    "CIRCUIT_DESC": "circuit_desc",
    "PARK_CODE": "park_code",
    "PARK_NAME": "park_name",
    "ADDRESS": "address",
    "AMOUNT": "amount",
    "END_DATE": "end_date",
    "FREE_DURATION": "free_duration_mins",
    "CURRENCY": "currency",
}

NORMALIZED_COLUMNS: Final[list[str]] = [
    "system_id",
    "printed_id",
    "terminal_code",
    "start_date",
    "end_date",
    "paid_duration_mins",
    "total_duration_mins",
    "free_duration_mins",
    "amount",
    "currency",
    "payment_mean",
    "card_type",
    "zone_desc",
    "circuit_desc",
    "address",
    "park_code",
    "park_name",
]

# Columns that must never be null
CRITICAL_COLUMNS: Final[list[str]] = [
    "system_id",
    "terminal_code",
    "start_date",
]

ENRICHED_REQUIRED_COLUMNS: Final[list[str]] = [
    "system_id",
    "terminal_code",
    "start_date",
    "latitude",
    "longitude",
    "total_spaces",
]

# Get parameters
args = getResolvedOptions(
    sys.argv, ["JOB_NAME", "source_bucket", "target_bucket", "artifacts_bucket", "year", "month"]
)

job.init(args["JOB_NAME"], args)  # initialize the Glue Job
source_bucket = args["source_bucket"]
target_bucket = args["target_bucket"]
artifacts_bucket = args["artifacts_bucket"]
year_number = int(args["year"])
month_number = int(args["month"])

# The ingestion Lambda writes the key with a zero-padded month (month=09), so the path must match
source_path = f"s3://{source_bucket}/year={year_number}/month={month_number:02d}/transactions.csv"
target_path = f"s3://{target_bucket}/processed"
coords_path = f"s3://{artifacts_bucket}/reference/coordinates.csv"
spaces_path = f"s3://{artifacts_bucket}/reference/total_spaces.csv"

print(f"Reading raw data from: {source_path}")

# Read from Bronze Layer
df = spark.read.option("header", "true").csv(source_path)

# Check 1: Input must not be empty
assert df.limit(1).count() > 0, "Bronze input dataset is empty."

# --- Step 1: Normalize - rename column names and select only important ones ---
# Rename columns to standard schema
for old_name, new_name in COLUMN_RENAME_MAP_RAW.items():
    if old_name in df.columns:
        df = df.withColumnRenamed(old_name, new_name)

# Parse dates into native Spark Timestamps
df = df.withColumn("start_date", to_timestamp(col("start_date"), RAW_DATE_FORMAT)).withColumn(
    "end_date", to_timestamp(col("end_date"), RAW_DATE_FORMAT)
)

# Check for unparsed dates
unparsed_dates = df.filter(col("start_date").isNull()).count()
assert unparsed_dates == 0, (
    f"{unparsed_dates} rows have a start_date that does not match "
    f"RAW_DATE_FORMAT {RAW_DATE_FORMAT!r}"
)

# Filter to keep only records within the target month
df = df.filter(
    (year(col("start_date")) == year_number) & (month(col("start_date")) == month_number)
)

# Deduplicate by system_id (keeping first encountered)
df = df.dropDuplicates(["system_id"])

# Process data: casting, formatting, and cleaning
df = (
    df.withColumn("system_id", col("system_id").cast("int"))
    .withColumn("printed_id", col("printed_id").cast("int"))
    .withColumn("terminal_code", col("terminal_code").cast("int"))
    .withColumn(
        "paid_duration_mins", floor(col("paid_duration_mins").cast("long") / 60).cast("long")
    )
    .withColumn(
        "total_duration_mins", floor(col("total_duration_mins").cast("long") / 60).cast("long")
    )
    .withColumn(
        "free_duration_mins", floor(col("free_duration_mins").cast("long") / 60).cast("long")
    )
    .withColumn("amount", col("amount").cast("float"))
    .withColumn("currency", trim(col("currency")))
    .withColumn(
        "payment_mean", when(lower(col("payment_mean")) == "coins", "coins").otherwise("card")
    )
    .withColumn("park_code", col("park_code").cast("long"))
)

# Ensure all normalized columns exist
for c in NORMALIZED_COLUMNS:
    if c not in df.columns:
        df = df.withColumn(c, lit(None).cast("string"))

df = df.select(NORMALIZED_COLUMNS)

# --- Step 2: Clean normalized data ---
# Filter essential non-null fields
df = df.filter(col("terminal_code").isNotNull() & col("start_date").isNotNull())

# --- Step 3: Enrich cleaned data with coordinates and total spaces ---
# Load reference data
coords_df = spark.read.option("header", "true").option("inferSchema", "true").csv(coords_path)
spaces_df = spark.read.option("header", "true").option("inferSchema", "true").csv(spaces_path)

# Filter and cast reference tables
coords_df = (
    coords_df.filter(
        col("terminal_code").isNotNull()
        & col("latitude").isNotNull()
        & col("longitude").isNotNull()
    )
    .withColumn("terminal_code", col("terminal_code").cast("int"))
    .withColumn("latitude", col("latitude").cast("double"))
    .withColumn("longitude", col("longitude").cast("double"))
)

spaces_df = (
    spaces_df.filter(col("terminal_code").isNotNull() & col("total_spaces").isNotNull())
    .withColumn("terminal_code", col("terminal_code").cast("int"))
    .withColumn("total_spaces", col("total_spaces").cast("int"))
)

# Inner join reference tables
ref_df = broadcast(coords_df.join(spaces_df, on="terminal_code", how="inner"))

# Join with main df
df = df.join(ref_df, on="terminal_code", how="inner")

# --- Step 4: Validate enriched data for any wrong values ---
df = df.cache()

# Check required columns
missing_columns = [c for c in ENRICHED_REQUIRED_COLUMNS if c not in df.columns]
assert not missing_columns, f"Missing required enriched columns: {missing_columns}"

assert df.count() > 0, "Enriched dataset is empty"

# Check negative amounts
neg_amount = df.filter(col("amount") < 0).count()
assert neg_amount == 0, f"Found {neg_amount} rows with negative amounts"

# Check invalid spaces
invalid_spaces = df.filter(col("total_spaces") <= 0).count()
assert invalid_spaces == 0, f"Found {invalid_spaces} rows with invalid total_spaces (<= 0)"

# --- Step 5: Write to Silver Layer ---
print(f"Writing processed data to: {target_path}")

# Add partition columns
df = df.withColumn("year", lit(year_number)).withColumn("month", lit(month_number))

# Overwrite only partitions that appear in df
spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")

# Write parquet to Silver Layer
df.write.mode("overwrite").partitionBy("year", "month").parquet(target_path)

# Commit Job
job.commit()
print("Bronze to Silver job completed successfully!")
