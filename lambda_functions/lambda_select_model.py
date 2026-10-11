"""
Pick the best model for every horizon after a Sagemaker training job.

Reads the model_metrics.csv that the training job uploaded and selects best model
per horizon by the lowest validation WAPE. The training job has already re-scored
the previously retained models on this run's validation split, so both candidate
sets are in that one file and are directly comparable - this function does not
compare metrics across runs.

WAPE rather than MAE: MAE is not comparable across horizons because longer
horizons have a lower mean target and a larger share of zeros. Selection uses the
validation split so the test split stays an honest held-out estimate.

The best model is promoted to a stable historical_models/ key, or simply retained
there if it already won a previous run.
"""

import csv
import io
import json
import os
from datetime import UTC, datetime

import boto3
from botocore.exceptions import ClientError

s3 = boto3.client("s3")

SELECTION_METRIC = os.environ.get("SELECTION_METRIC", "valid_wape")
METRICS_FILENAME = "model_metrics.csv"
OUTPUT_FILENAME = "best_models.json"

# "h1h:60,h2h:120,..." - the same value the Glue job is given.
HORIZONS_MINUTES = os.environ.get("HORIZONS_MINUTES", "")

REPORTED_METRICS = (
    "valid_mae",
    "valid_rmse",
    "valid_wape",
    "valid_mase",
    "test_mae",
    "test_rmse",
    "test_wape",
    "test_mase",
)

# Stable location each horizon's best model is promoted to, and that the next training
# run re-scores against. The per-run candidates live under current_models/, but this
# function learns their keys from the artifact_s3_uri column rather than rebuilding them.
HISTORICAL_MODELS_PREFIX = "historical_models"


def parse_s3_uri(uri):
    """Split s3://bucket/key into (bucket, key)."""
    if not uri.startswith("s3://"):
        raise ValueError(f"Not an S3 URI: {uri!r}")

    bucket, _, key = uri[len("s3://") :].partition("/")

    if not bucket:
        raise ValueError(f"S3 URI has no bucket: {uri!r}")

    return bucket, key


def expected_horizons():
    """Horizon names from HORIZONS_MINUTES, or an empty set when it is not configured."""
    names = set()
    for item in HORIZONS_MINUTES.split(","):
        name = item.partition(":")[0].strip()
        if name:
            names.add(name)
    return names


def _metric_value(row, metric):
    """Read a metric as a float, mapping missing/unparsable/NaN to +inf so it loses."""
    try:
        value = float(row.get(metric))
    except (TypeError, ValueError):
        return float("inf")

    return value if value == value else float("inf")


def _source_of(row):
    """Where a candidate came from. Rows without the column predate it and are current."""
    return (row.get("source") or "current").strip() or "current"


def _as_number(value):
    """Safely convert a CSV string or JSON value to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def select_best(rows, metric=SELECTION_METRIC):
    """Return {horizon: best model row}, lowest metric first, ties broken by model name."""
    best_models = {}

    for row in rows:
        horizon = (row.get("horizon") or "").strip()
        model = (row.get("model") or "").strip()
        if not horizon or not model:
            continue

        rank = (_metric_value(row, metric), model)
        current_model = best_models.get(horizon)
        if current_model is None or rank < current_model[0]:
            best_models[horizon] = (rank, row)

    return {horizon: row for horizon, (rank, row) in best_models.items()}


def read_metrics(bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    text = obj["Body"].read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def lambda_handler(event, context):
    metrics_s3_uri = event["metrics_s3_uri"]
    run_id = event.get("run_id", "")

    bucket, metrics_prefix = parse_s3_uri(metrics_s3_uri)
    metrics_key = f"{metrics_prefix.rstrip('/')}/{METRICS_FILENAME}"

    print(f"Reading s3://{bucket}/{metrics_key}")
    rows = read_metrics(bucket, metrics_key)
    if not rows:
        raise ValueError(f"{metrics_key} contained no metric rows")

    for row in rows:
        print(
            f"  {row.get('horizon'):<6} {row.get('model'):<10} {_source_of(row):<10} "
            f"{SELECTION_METRIC}={row.get(SELECTION_METRIC)} test_mae={row.get('test_mae')}"
        )

    best_models = select_best(rows)

    expected = expected_horizons()
    missing = expected - set(best_models)
    if missing:
        raise ValueError(
            f"No trained model reported metrics for horizon(s) {sorted(missing)}. "
            f"Expected {sorted(expected)}, got {sorted(best_models)}."
        )

    final_horizons = {}

    for horizon in sorted(best_models):
        row = best_models[horizon]
        model = row["model"].strip()
        source = _source_of(row)
        artifact_uri = (row.get("artifact_s3_uri") or "").strip()
        promoted_key = f"{HISTORICAL_MODELS_PREFIX}/{model}_{horizon}.joblib"

        if source == "historical":
            # Already sitting at the stable location; re-copying would be a no-op.
            final_uri = artifact_uri or f"s3://{bucket}/{promoted_key}"
            print(f"  {horizon}: retaining {model} ({_metric_value(row, SELECTION_METRIC):.4f})")
        else:
            if not artifact_uri:
                raise ValueError(
                    f"{horizon}: best model {model} has no artifact_s3_uri in {metrics_key}"
                )
            src_bucket, src_key = parse_s3_uri(artifact_uri)
            try:
                s3.copy_object(
                    Bucket=bucket,
                    CopySource={"Bucket": src_bucket, "Key": src_key},
                    Key=promoted_key,
                )
            except ClientError as exc:
                raise RuntimeError(
                    f"{horizon}: failed to promote s3://{src_bucket}/{src_key} "
                    f"to s3://{bucket}/{promoted_key}: {exc}"
                ) from exc
            final_uri = f"s3://{bucket}/{promoted_key}"
            print(f"  {horizon}: promoted {model} ({_metric_value(row, SELECTION_METRIC):.4f})")

        final_horizons[horizon] = {
            "model": model,
            "source": source,
            "artifact_s3_uri": final_uri,
            **{name: _as_number(row.get(name)) for name in REPORTED_METRICS},
        }

    payload = {
        "run_id": run_id,
        "selected_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "selection_metric": SELECTION_METRIC,
        "source_metrics": f"s3://{bucket}/{metrics_key}",
        "candidates": len(rows),
        "horizons": final_horizons,
    }

    body = json.dumps(payload, indent=2).encode("utf-8")

    # The stable registry the next training run re-scores against.
    registry_key = f"{HISTORICAL_MODELS_PREFIX}/{OUTPUT_FILENAME}"
    s3.put_object(Bucket=bucket, Key=registry_key, Body=body, ContentType="application/json")
    print(f"Updated registry at s3://{bucket}/{registry_key}")

    # Per-run copy for audit: what this execution decided, and from what.
    audit_key = f"{metrics_prefix.rstrip('/')}/{OUTPUT_FILENAME}"
    s3.put_object(Bucket=bucket, Key=audit_key, Body=body, ContentType="application/json")

    return payload
