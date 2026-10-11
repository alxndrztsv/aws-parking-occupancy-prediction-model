import operator
import sys
from datetime import timedelta
from functools import reduce

from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql.functions import (
    broadcast,
    coalesce,
    col,
    count,
    dayofweek,
    explode,
    expr,
    first,
    hour,
    lag,
    lead,
    lit,
    max,
    mean,
    min,
    minute,
    month,
    stddev,
    when,
)
from pyspark.sql.window import Window

# Initialize Glue Context
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)

# Resolve parameters
args = getResolvedOptions(
    sys.argv, ["JOB_NAME", "source_bucket", "target_bucket", "bank_holidays", "horizons_minutes"]
)

job.init(args["JOB_NAME"], args)

source_bucket = args["source_bucket"]
target_bucket = args["target_bucket"]

# Business logic
INTERVAL_MINUTES = 10
OPEN_HOUR = 9
CLOSE_HOUR = 20
TOTAL_BINS_PER_DAY = (24 * 60) // INTERVAL_MINUTES
TEST_DAYS = 30
VALIDATION_DAYS = 60

bank_holidays_str = args.get("bank_holidays", "")
BANK_HOLIDAYS = [d.strip() for d in bank_holidays_str.split(",") if d.strip()]

horizons_str = args.get("horizons_minutes", "h1h:60,h2h:120,h3h:180,h24h:1440")
HORIZONS_MINUTES = {}
for item in horizons_str.split(","):
    name, mins = item.split(":")
    HORIZONS_MINUTES[name.strip()] = int(mins.strip())

# Paths
silver_path = f"s3://{source_bucket}/processed"
gold_operational_path = f"s3://{target_bucket}/operational"
gold_continuous_path = f"s3://{target_bucket}/continuous"
gold_features_path = f"s3://{target_bucket}/features"
gold_train_path = f"s3://{target_bucket}/splits/train"
gold_valid_path = f"s3://{target_bucket}/splits/valid"
gold_test_path = f"s3://{target_bucket}/splits/test"

print(f"Reading silver data from: {silver_path}")
df = spark.read.parquet(silver_path)

# --- Build operational occupancy table ---
print("Building operational occupancy table...")

# Spark's INTERVAL <value> <unit> only accepts a literal, so scale a literal interval by the column
df = df.withColumn(
    "end_time", col("start_date") + col("paid_duration_mins") * expr("INTERVAL 1 MINUTE")
)
df = df.withColumn("day_start", expr(f"date_trunc('day', start_date) + INTERVAL {OPEN_HOUR} HOURS"))
df = df.withColumn("day_end", expr(f"date_trunc('day', start_date) + INTERVAL {CLOSE_HOUR} HOURS"))

# Cap start and end times to operational hours
df = df.withColumn(
    "start_time",
    when(col("start_date") > col("day_start"), col("start_date")).otherwise(col("day_start")),
)
df = df.withColumn(
    "end_time_capped",
    when(col("end_time") < col("day_end"), col("end_time")).otherwise(col("day_end")),
)

# Filter out tickets completely outside operational hours
df = df.filter(col("start_time") < col("end_time_capped"))

# Bin truncation: floor(unix_timestamp / bin_seconds) * bin_seconds
bin_seconds = INTERVAL_MINUTES * 60
df = df.withColumn(
    "first_bin_ts",
    expr(f"to_timestamp(floor(unix_timestamp(start_time) / {bin_seconds}) * {bin_seconds})"),
)
df = df.withColumn(
    "last_bin_ts",
    expr(
        f"to_timestamp(floor(unix_timestamp(end_time_capped - INTERVAL 1 SECOND)"
        f" / {bin_seconds}) * {bin_seconds})"
    ),
)
df = df.withColumn(
    "num_bins",
    expr(
        f"cast((unix_timestamp(last_bin_ts) - unix_timestamp(first_bin_ts))"
        f" / {bin_seconds} + 1 as int)"
    ),
)

# Generate sequence of timestamps and explode
df = df.withColumn(
    "timestamps", expr(f"sequence(first_bin_ts, last_bin_ts, interval {INTERVAL_MINUTES} minutes)")
)
df_exploded = df.select(
    "terminal_code", "total_spaces", explode(col("timestamps")).alias("timestamp")
)

operational_table = (
    df_exploded.groupBy("terminal_code", "timestamp")
    .agg(count("*").alias("active_sessions"), first("total_spaces").alias("total_spaces"))
    .withColumn(
        "occupancy_rate",
        expr("least(greatest(cast(active_sessions as double) / total_spaces, 0.0), 1.0)"),
    )
    .select(
        col("terminal_code").cast("int"),
        col("timestamp").cast("timestamp"),
        col("active_sessions").cast("int"),
        col("total_spaces").cast("int"),
        col("occupancy_rate").cast("double"),
    )
)

print(f"Writing operational table to: {gold_operational_path}")
operational_table.write.mode("overwrite").parquet(gold_operational_path)

# --- Build continuous 24-hour occupancy table ---
print("Building continuous occupancy table...")

# Get min/max dates per terminal
bounds = operational_table.groupBy("terminal_code").agg(
    min(expr("cast(timestamp as date)")).alias("min_date"),
    max(expr("cast(timestamp as date)")).alias("max_date"),
    first("total_spaces").alias("total_spaces"),
)

# Filter out invalid bounds (matches Polars logic)
bounds = bounds.filter(
    col("min_date").isNotNull()
    & col("max_date").isNotNull()
    & col("total_spaces").isNotNull()
    & (col("total_spaces") > 0)
)

# Generate all dates for each terminal
bounds = bounds.withColumn("date_seq", expr("sequence(min_date, max_date, interval 1 day)"))
exploded_dates = bounds.select(
    "terminal_code", "total_spaces", explode(col("date_seq")).alias("date")
)

# Generate 24h slots (0 to TOTAL_BINS_PER_DAY - 1)
slots_df = (
    spark.range(TOTAL_BINS_PER_DAY)
    .withColumnRenamed("id", "slot_24h")
    .withColumn("minutes_from_midnight", col("slot_24h") * INTERVAL_MINUTES)
)

# Broadcast the small slots_df to prevent a massive shuffle
grid = exploded_dates.crossJoin(broadcast(slots_df))
grid = grid.withColumn(
    "timestamp",
    col("date").cast("timestamp") + col("minutes_from_midnight") * expr("INTERVAL 1 MINUTE"),
)

# Drop intermediate column
grid = grid.drop("minutes_from_midnight")

# Left join with actual operational data
continuous_table = grid.join(
    operational_table.select("terminal_code", "timestamp", "active_sessions", "occupancy_rate"),
    on=["terminal_code", "timestamp"],
    how="left",
)

# Fill nulls with 0 (closed/empty bins)
continuous_table = continuous_table.withColumn(
    "active_sessions", coalesce(col("active_sessions"), lit(0)).cast("int")
)
continuous_table = continuous_table.withColumn(
    "occupancy_rate", coalesce(col("occupancy_rate"), lit(0.0)).cast("double")
)

# Calendar features
continuous_table = continuous_table.withColumn("hour", hour("timestamp"))
continuous_table = continuous_table.withColumn("minute", minute("timestamp"))
# Spark's dayofweek() is 1=Sun..7=Sat; the reference polars weekday() is 1=Mon..7=Sun
continuous_table = continuous_table.withColumn(
    "day_of_week", ((dayofweek("timestamp") + 5) % 7) + 1
)
continuous_table = continuous_table.withColumn("date", expr("cast(timestamp as date)"))
continuous_table = continuous_table.withColumn("is_weekend", col("day_of_week").isin([6, 7]))

if BANK_HOLIDAYS:
    continuous_table = continuous_table.withColumn(
        "is_bank_holiday", col("date").isin(BANK_HOLIDAYS)
    )
else:
    continuous_table = continuous_table.withColumn("is_bank_holiday", lit(False))

continuous_table = continuous_table.withColumn(
    "is_closed_day", col("is_weekend") | col("is_bank_holiday")
)

# Force closed days to 0 occupancy
continuous_table = continuous_table.withColumn(
    "active_sessions", when(col("is_closed_day"), lit(0)).otherwise(col("active_sessions"))
)
continuous_table = continuous_table.withColumn(
    "occupancy_rate", when(col("is_closed_day"), lit(0.0)).otherwise(col("occupancy_rate"))
)

print(f"Writing continuous table to: {gold_continuous_path}")
continuous_table.write.mode("overwrite").parquet(gold_continuous_path)

# --- Build feature table ---
print("Building feature table...")

window_spec = Window.partitionBy("terminal_code").orderBy("timestamp")
df_feat = continuous_table

# Build all lag/lead expressions
lag_exprs = [
    lag("occupancy_rate", bins).over(window_spec).alias(f"occupancy_lag_{bins}")
    for bins in [1, 2, 3, 6, 12, 18]
]
same_time_exprs = [
    lag("occupancy_rate", days * TOTAL_BINS_PER_DAY)
    .over(window_spec)
    .alias(f"occupancy_same_time_{days}d")
    for days in [1, 7, 14]
]

horizon_exprs = []
for horizon_name, horizon_minutes in HORIZONS_MINUTES.items():
    horizon_bins = horizon_minutes // INTERVAL_MINUTES
    horizon_exprs.append(
        lead("occupancy_rate", horizon_bins).over(window_spec).alias(f"y_{horizon_name}")
    )
    naive_shift = TOTAL_BINS_PER_DAY - horizon_bins
    horizon_exprs.append(
        lag("occupancy_rate", naive_shift)
        .over(window_spec)
        .alias(f"pred_seasonal_naive_{horizon_name}")
    )

# Rolling features (rowsBetween is inclusive)
window_rolling_6 = window_spec.rowsBetween(-5, 0)  # 6 rows total
window_rolling_12 = window_spec.rowsBetween(-11, 0)  # 12 rows total
window_rolling_18 = window_spec.rowsBetween(-17, 0)  # 18 rows total


df_feat = df_feat.select(
    "*",
    *lag_exprs,
    *same_time_exprs,
    *horizon_exprs,
    when(
        count("*").over(window_rolling_6) == 6, mean("occupancy_rate").over(window_rolling_6)
    ).alias("occupancy_rolling_mean_1h"),
    when(
        count("*").over(window_rolling_6) == 6, stddev("occupancy_rate").over(window_rolling_6)
    ).alias("occupancy_rolling_std_1h"),
    when(
        count("*").over(window_rolling_12) == 12, mean("occupancy_rate").over(window_rolling_12)
    ).alias("occupancy_rolling_mean_2h"),
    when(
        count("*").over(window_rolling_18) == 18, mean("occupancy_rate").over(window_rolling_18)
    ).alias("occupancy_rolling_mean_3h"),
    month("timestamp").alias("month"),
    (
        (col("hour") >= OPEN_HOUR)
        & (col("hour") < CLOSE_HOUR)
        & ~col("is_weekend")
        & ~col("is_bank_holiday")
    ).alias("is_training_origin"),
)

# Same-time last 7 days mean
tmp_cols = [f"tmp_lag_{d}d" for d in range(1, 8)]
last_7d_lags = [
    lag("occupancy_rate", d * TOTAL_BINS_PER_DAY).over(window_spec).alias(c)
    for d, c in zip(range(1, 8), tmp_cols, strict=True)
]
df_feat = df_feat.select("*", *last_7d_lags)

sum_expr = reduce(operator.add, [coalesce(col(c), lit(0.0)) for c in tmp_cols])
count_expr = reduce(operator.add, [col(c).isNotNull().cast("int") for c in tmp_cols])

df_feat = df_feat.withColumn(
    "occupancy_same_time_last_7d_mean",
    when(count_expr > 0, sum_expr / count_expr).otherwise(lit(None).cast("double")),
).drop(*tmp_cols)

# Current state features
df_feat = (
    df_feat.withColumn("current_occupancy_rate", col("occupancy_rate"))
    .withColumn("current_active_sessions", col("active_sessions"))
    .withColumn(
        "current_free_spaces",
        when(col("active_sessions") >= col("total_spaces"), lit(0)).otherwise(
            col("total_spaces") - col("active_sessions")
        ),
    )
    .drop("occupancy_rate", "active_sessions")
)  # prevent data leakage

print("Caching feature table to optimize writes...")

df_feat = df_feat.cache()
df_feat.count()  # trigger the cache

print(f"Writing features table to: {gold_features_path}")

df_feat.write.mode("overwrite").parquet(gold_features_path)

# --- Split data (train / valid / test) ---
print("Splitting data into train, valid, test...")

target_cols = [f"y_{h}" for h in HORIZONS_MINUTES]
for tc in target_cols:
    df_feat = df_feat.filter(col(tc).isNotNull())

df_feat = df_feat.filter(col("is_training_origin"))

# Get max date dynamically
max_date_row = df_feat.select(max(expr("cast(timestamp as date)"))).collect()[0]
max_date = max_date_row[0]

if max_date is None:
    raise ValueError(
        "No valid data found after filtering. Check your date ranges and bank holidays."
    )

test_start = max_date - timedelta(days=TEST_DAYS - 1)
valid_start = test_start - timedelta(days=VALIDATION_DAYS)

print(f"Split boundaries: valid_start={valid_start}, test_start={test_start}, max_date={max_date}")

train_df = df_feat.filter(expr(f"cast(timestamp as date) < '{valid_start}'"))
valid_df = df_feat.filter(
    expr(f"cast(timestamp as date) >= '{valid_start}' and cast(timestamp as date) < '{test_start}'")
)
test_df = df_feat.filter(expr(f"cast(timestamp as date) >= '{test_start}'"))

print(f"Writing train split to: {gold_train_path}")
train_df.write.mode("overwrite").parquet(gold_train_path)

print(f"Writing valid split to: {gold_valid_path}")
valid_df.write.mode("overwrite").parquet(gold_valid_path)

print(f"Writing test split to: {gold_test_path}")
test_df.write.mode("overwrite").parquet(gold_test_path)

# Clean up cache
df_feat.unpersist()

job.commit()
print("Silver to Gold job completed successfully!")
