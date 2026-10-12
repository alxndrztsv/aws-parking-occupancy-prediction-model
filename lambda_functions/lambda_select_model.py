"""
Decide what to do with a freshly trained model set, and ask a person.

Reads the model_metrics.csv the training job uploaded. That file already contains
both candidate sets scored on the same validation window, because the training job
re-scores the retained models on this run's splits - so comparing them here is
comparing like with like, not comparing two different months.

WAPE rather than MAE: MAE is not comparable across horizons because longer horizons
have a lower mean target and a larger share of zeros. Selection uses the validation
split so the test split stays an honest held-out estimate.

This function writes historical_models/pending.json, publishes a summary with
a single-use link, and returns. Production keeps serving the retained models
until lambda_apply_decision is invoked through that link.
"""

import csv
import io
import json
import os
import secrets
import urllib.parse
from datetime import UTC, datetime, timedelta

import boto3

s3 = boto3.client("s3")
sns = boto3.client("sns")

SELECTION_METRIC = os.environ.get("SELECTION_METRIC", "valid_wape")
METRICS_FILENAME = "model_metrics.csv"
PENDING_FILENAME = "pending.json"

# "h1h:60,h2h:120,..." - the same value the Glue job is given.
HORIZONS_MINUTES = os.environ.get("HORIZONS_MINUTES", "")

# A current model only wins if it clears this relative improvement. Without a band the
# pipeline promotes on noise: h2h once scored 0.0482 against 0.0483, a statistical
# tie, and flipping winners every month makes the retained set meaningless.
PROMOTION_THRESHOLD = float(os.environ.get("PROMOTION_THRESHOLD", "0.01"))

# Must stay below the 90-day lifecycle on current_models/, or an approved link
# would point at an object that no longer exists.
LINK_TTL_DAYS = int(os.environ.get("LINK_TTL_DAYS", "30"))

MODEL_DECISION_TOPIC_ARN = os.environ.get("MODEL_DECISION_TOPIC_ARN", "")
APPROVAL_URL = os.environ.get("APPROVAL_URL", "")

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

# Stable location the retained models live in, and that the next training run
# re-scores against. Per-run candidates live under current_models/; this function
# learns their keys from the artifact_s3_uri column rather than rebuilding them.
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


def select_best(rows, metric=SELECTION_METRIC, source=None):
    """Return {horizon: best row}, lowest metric first, ties broken by model name."""
    best_models = {}

    for row in rows:
        if source is not None and _source_of(row) != source:
            continue

        horizon = (row.get("horizon") or "").strip()
        model = (row.get("model") or "").strip()
        if not horizon or not model:
            continue

        rank = (_metric_value(row, metric), model)
        current_model = best_models.get(horizon)
        if current_model is None or rank < current_model[0]:
            best_models[horizon] = (rank, row)

    return {horizon: row for horizon, (rank, row) in best_models.items()}


def _candidate(row):
    """The part of a metric row that survives into pending.json."""
    if row is None:
        return None

    return {
        "model": row["model"].strip(),
        "artifact_s3_uri": (row.get("artifact_s3_uri") or "").strip(),
        **{name: _as_number(row.get(name)) for name in REPORTED_METRICS},
    }


def build_decision(rows, metric=SELECTION_METRIC, threshold=PROMOTION_THRESHOLD):
    """
    Compare this run's best against the retained model, per horizon.

    Returns {horizon: {current, historical, improvement, recommendation, reason}}.
    """
    currents = select_best(rows, metric=metric, source="current")
    historicals = select_best(rows, metric=metric, source="historical")

    horizons = {}

    for horizon, current_row in currents.items():
        historical_row = historicals.get(horizon)
        current_value = _metric_value(current_row, metric)
        entry = {
            "current": _candidate(current_row),
            "historical": _candidate(historical_row),
        }

        if historical_row is None:
            entry.update(
                improvement=None,
                recommendation="replace",
                reason="no model is retained for this horizon yet",
            )
            horizons[horizon] = entry
            continue

        historical_value = _metric_value(historical_row, metric)

        if historical_value in (0.0, float("inf")) or current_value == float("inf"):
            entry.update(
                improvement=None,
                recommendation="keep",
                reason=f"{metric} is unusable for one of the two candidates",
            )
            horizons[horizon] = entry
            continue

        improvement = (historical_value - current_value) / historical_value
        entry["improvement"] = round(improvement, 6)

        if current_value <= historical_value * (1 - threshold):
            entry["recommendation"] = "replace"
            entry["reason"] = (
                f"{metric} improved {improvement * 100:.1f}%, above the {threshold * 100:.1f}% band"
            )
        else:
            entry["recommendation"] = "keep"
            entry["reason"] = (
                f"{metric} moved {improvement * 100:+.1f}%, "
                f"inside the {threshold * 100:.1f}% noise band"
            )

        horizons[horizon] = entry

    return horizons


def read_metrics(bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    text = obj["Body"].read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def build_email(pending, approval_url):
    """Render the plain-text SNS message and its subject."""
    horizons = pending["horizons"]
    replace_count = sum(1 for e in horizons.values() if e["recommendation"] == "replace")

    subject = f"Models ready: replace {replace_count}/{len(horizons)} (run {pending['run_id']})"

    lines = [
        f"Run {pending['run_id']} finished.",
        f"Metric: {pending['selection_metric']}",
        "",
        # "improve" not "delta": the number is (historical - current) / historical,
        # so positive always means the new model is better. A header called "delta"
        # reads as a change in WAPE, where positive would look like a regression.
        f"{'horizon':<8} {'retained':<22} {'new best':<22} {'improve':>8}",
    ]

    for horizon, entry in sorted(horizons.items()):
        historical = entry["historical"]
        current = entry["current"]
        retained = (
            f"{historical['model']} {_fmt(historical['valid_wape'])}" if historical else "(none)"
        )
        lines.append(
            f"{horizon:<8} {retained:<22} "
            f"{current['model']} {_fmt(current['valid_wape']):<15} "
            f"{_fmt_delta(entry['improvement']):>8}"
        )

    lines += [
        "",
        f"Metrics: {pending['source_metrics']}",
        "",
        "Nothing has been promoted. Production still serves the retained models.",
        "",
        "Review and decide (nothing changes until you submit):",
        f"  {approval_url}?{urllib.parse.urlencode(_link_query(pending))}",
        "",
        f"Link expires {pending['expires_at']}. A newer run supersedes it.",
    ]

    return subject[:100], "\n".join(lines)


def _link_query(pending):
    return {"run_id": pending["run_id"], "token": pending["token"]}


def _fmt(value):
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "?"


def _fmt_delta(value):
    try:
        return f"{float(value) * 100:+.1f}%"
    except (TypeError, ValueError):
        return "?"


def lambda_handler(event, context):
    metrics_s3_uri = event["metrics_s3_uri"]
    run_id = event.get("run_id", "")

    bucket, metrics_prefix = parse_s3_uri(metrics_s3_uri)
    metrics_prefix = metrics_prefix.rstrip("/")
    metrics_key = f"{metrics_prefix}/{METRICS_FILENAME}"

    print(f"Reading s3://{bucket}/{metrics_key}")
    rows = read_metrics(bucket, metrics_key)
    if not rows:
        raise ValueError(f"{metrics_key} contained no metric rows")

    for row in rows:
        print(
            f"  {row.get('horizon'):<6} {row.get('model'):<10} {_source_of(row):<10} "
            f"{SELECTION_METRIC}={row.get(SELECTION_METRIC)} test_mae={row.get('test_mae')}"
        )

    currents = select_best(rows, source="current")

    expected = expected_horizons()
    missing = expected - set(currents)
    if missing:
        raise ValueError(
            f"No trained model reported metrics for horizon(s) {sorted(missing)}. "
            f"Expected {sorted(expected)}, got {sorted(currents)}."
        )

    now = datetime.now(UTC)
    pending = {
        "run_id": run_id,
        "status": "pending",
        "created_at": now.isoformat(timespec="seconds"),
        "expires_at": (now + timedelta(days=LINK_TTL_DAYS)).isoformat(timespec="seconds"),
        "token": secrets.token_urlsafe(24),  # single-use token to secure unauthenticated endpoint
        "selection_metric": SELECTION_METRIC,
        "threshold": PROMOTION_THRESHOLD,
        "source_metrics": f"s3://{bucket}/{metrics_key}",
        "candidates": len(rows),
        "horizons": build_decision(rows),
    }

    body = json.dumps(pending, indent=2).encode("utf-8")

    # Fixed key, overwritten every run: "the current decision" is always here, and a
    # link from an older run stops verifying because its run_id no longer matches.
    pending_key = f"{HISTORICAL_MODELS_PREFIX}/{PENDING_FILENAME}"
    s3.put_object(Bucket=bucket, Key=pending_key, Body=body, ContentType="application/json")
    print(f"Wrote s3://{bucket}/{pending_key}")

    # Per-run copy for audit.
    audit_key = f"{metrics_prefix}/{PENDING_FILENAME}"
    s3.put_object(Bucket=bucket, Key=audit_key, Body=body, ContentType="application/json")

    subject, message = build_email(pending, APPROVAL_URL.rstrip("/"))

    if MODEL_DECISION_TOPIC_ARN:
        sns.publish(
            TopicArn=MODEL_DECISION_TOPIC_ARN,
            Subject=subject,
            Message=message,
        )
        print(f"Published to {MODEL_DECISION_TOPIC_ARN}: {subject}")
    else:
        print("MODEL_DECISION_TOPIC_ARN is not set, skipping the notification.")
        print(message)

    base_url = APPROVAL_URL.rstrip("/")
    approval_link = f"{base_url}?{urllib.parse.urlencode(_link_query(pending))}"

    return {
        "run_id": run_id,
        "status": "pending",
        "replace": sorted(
            h for h, e in pending["horizons"].items() if e["recommendation"] == "replace"
        ),
        "keep": sorted(h for h, e in pending["horizons"].items() if e["recommendation"] == "keep"),
        "pending_s3_uri": f"s3://{bucket}/{pending_key}",
        "approval_url": approval_link,
    }
