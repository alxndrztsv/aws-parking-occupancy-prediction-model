import csv
import io
import json
import math

import lambda_select_model as sel
import pytest

ARTIFACTS = "parking-occupancy-prediction-dev-artifacts"
METRICS_URI = f"s3://{ARTIFACTS}/metrics/run-1/"


def metric_row(model, horizon, wape, source="current", mae=0.1, with_artifact=True):
    row = {
        "model": model,
        "horizon": horizon,
        "target": f"y_{horizon}",
        "valid_mae": str(mae),
        "valid_rmse": "0.2",
        "valid_wape": str(wape),
        "valid_mase": "0.9",
        "test_mae": str(mae),
        "test_rmse": "0.2",
        "test_wape": str(wape),
        "test_mase": "0.9",
        "source": source,
    }
    if with_artifact:
        prefix = "historical_models" if source == "historical" else "current_models/run-1"
        row["artifact_s3_uri"] = f"s3://{ARTIFACTS}/{prefix}/{model}_{horizon}.joblib"
    return row


class FakeS3:
    def __init__(self, body=""):
        self.body = body.encode("utf-8") if isinstance(body, str) else body
        self.puts = []
        self.copies = []

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.body)}

    def put_object(self, Bucket, Key, Body, ContentType):
        self.puts.append((Bucket, Key, Body, ContentType))

    def copy_object(self, Bucket, CopySource, Key):
        self.copies.append((Bucket, CopySource["Bucket"], CopySource["Key"], Key))


def csv_body(rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def run_handler(monkeypatch, rows, **event):
    fake = FakeS3(csv_body(rows))
    monkeypatch.setattr(sel, "s3", fake)
    payload = sel.lambda_handler({"metrics_s3_uri": METRICS_URI, **event}, None)
    return payload, fake


def registry_of(fake):
    """The stable best_models.json the next training run will read back."""
    for _bucket, key, body, _ in fake.puts:
        if key == "historical_models/best_models.json":
            return json.loads(body)
    raise AssertionError("no registry written")


# --- pure helpers ---


def test_parse_s3_uri_splits_bucket_and_key():
    assert sel.parse_s3_uri("s3://bucket/metrics/run/") == ("bucket", "metrics/run/")
    assert sel.parse_s3_uri("s3://bucket") == ("bucket", "")


def test_parse_s3_uri_rejects_other_schemes():
    for bad in ("https://bucket/key", "bucket/key", "s3://"):
        with pytest.raises(ValueError):
            sel.parse_s3_uri(bad)


def test_lowest_wape_wins_per_horizon():
    rows = [
        metric_row("lightgbm", "h1h", 0.30),
        metric_row("xgboost", "h1h", 0.31),
        metric_row("ensemble", "h1h", 0.29),
        metric_row("xgboost", "h3h", 0.20),
        metric_row("lightgbm", "h3h", 0.25),
    ]

    got = sel.select_best(rows)

    assert got["h1h"]["model"] == "ensemble"
    assert got["h3h"]["model"] == "xgboost"


def test_ties_are_broken_by_model_name_so_the_result_is_deterministic():
    rows = [metric_row("xgboost", "h1h", 0.3), metric_row("lightgbm", "h1h", 0.3)]

    assert sel.select_best(rows)["h1h"]["model"] == "lightgbm"
    assert sel.select_best(list(reversed(rows)))["h1h"]["model"] == "lightgbm"


@pytest.mark.parametrize("bad", ["", "nan", "NaN", None, "not-a-number"])
def test_unusable_metrics_lose_to_a_real_number(bad):
    rows = [metric_row("broken", "h1h", bad), metric_row("working", "h1h", 0.9)]

    assert sel.select_best(rows)["h1h"]["model"] == "working"


def test_rows_without_a_model_or_horizon_are_skipped():
    rows = [
        metric_row("lightgbm", "h1h", 0.3),
        {"model": "", "horizon": "h1h", "valid_wape": "0.1"},
        {"model": "ghost", "horizon": "", "valid_wape": "0.1"},
    ]

    assert sel.select_best(rows)["h1h"]["model"] == "lightgbm"


def test_the_selection_metric_can_be_overridden():
    rows = [metric_row("cheap", "h1h", 0.9, mae=0.01), metric_row("dear", "h1h", 0.1, mae=0.5)]

    assert sel.select_best(rows, metric="valid_wape")["h1h"]["model"] == "dear"
    assert sel.select_best(rows, metric="valid_mae")["h1h"]["model"] == "cheap"


def test_expected_horizons_parses_the_terraform_variable(monkeypatch):
    monkeypatch.setattr(sel, "HORIZONS_MINUTES", "h1h:60,h2h:120, h24h:1440")
    assert sel.expected_horizons() == {"h1h", "h2h", "h24h"}

    monkeypatch.setattr(sel, "HORIZONS_MINUTES", "")
    assert sel.expected_horizons() == set()


def test_metric_value_maps_nan_to_infinity():
    assert sel._metric_value({"valid_wape": "nan"}, "valid_wape") == math.inf
    assert sel._metric_value({}, "valid_wape") == math.inf
    assert sel._metric_value({"valid_wape": "0.25"}, "valid_wape") == 0.25


def test_source_defaults_to_current_for_rows_that_predate_the_column():
    assert sel._source_of({"model": "x", "horizon": "h1h"}) == "current"
    assert sel._source_of({"source": ""}) == "current"
    assert sel._source_of({"source": "historical"}) == "historical"


# --- handler ---


def test_a_current_winner_is_promoted_into_historical_models(monkeypatch):
    rows = [metric_row("lightgbm", "h1h", 0.30), metric_row("xgboost", "h1h", 0.31)]

    payload, fake = run_handler(monkeypatch, rows, run_id="run-1")

    assert fake.copies == [
        (
            ARTIFACTS,
            ARTIFACTS,
            "current_models/run-1/lightgbm_h1h.joblib",
            "historical_models/lightgbm_h1h.joblib",
        )
    ]

    entry = payload["horizons"]["h1h"]
    assert entry["model"] == "lightgbm"
    assert entry["source"] == "current"
    assert entry["artifact_s3_uri"] == f"s3://{ARTIFACTS}/historical_models/lightgbm_h1h.joblib"
    assert entry["valid_wape"] == 0.30
    assert payload["run_id"] == "run-1"
    assert payload["selection_metric"] == "valid_wape"
    assert payload["candidates"] == 2
    assert payload["source_metrics"] == f"s3://{ARTIFACTS}/metrics/run-1/model_metrics.csv"


def test_a_historical_winner_is_retained_without_copying(monkeypatch):
    """The whole point of re-scoring in the training job: an older model can win."""
    rows = [
        metric_row("lightgbm", "h1h", 0.30, source="current"),
        metric_row("catboost", "h1h", 0.28, source="historical"),
    ]

    payload, fake = run_handler(monkeypatch, rows, run_id="run-1")

    assert fake.copies == []
    assert payload["horizons"]["h1h"]["model"] == "catboost"
    assert payload["horizons"]["h1h"]["source"] == "historical"
    assert payload["horizons"]["h1h"]["artifact_s3_uri"] == (
        f"s3://{ARTIFACTS}/historical_models/catboost_h1h.joblib"
    )


def test_registry_and_audit_copy_are_both_written(monkeypatch):
    payload, fake = run_handler(monkeypatch, [metric_row("lightgbm", "h1h", 0.3)], run_id="run-1")

    keys = [key for _, key, _, _ in fake.puts]
    assert keys == [
        "historical_models/best_models.json",
        "metrics/run-1/best_models.json",
    ]
    assert all(content_type == "application/json" for *_, content_type in fake.puts)
    assert json.loads(fake.puts[0][2]) == payload
    assert registry_of(fake) == payload


def test_each_horizon_is_decided_independently(monkeypatch):
    rows = [
        metric_row("lightgbm", "h1h", 0.30, source="current"),
        metric_row("catboost", "h1h", 0.35, source="historical"),
        metric_row("xgboost", "h24h", 0.40, source="current"),
        metric_row("ensemble", "h24h", 0.20, source="historical"),
    ]

    payload, fake = run_handler(monkeypatch, rows, run_id="run-1")

    assert payload["horizons"]["h1h"]["model"] == "lightgbm"
    assert payload["horizons"]["h1h"]["source"] == "current"
    assert payload["horizons"]["h24h"]["model"] == "ensemble"
    assert payload["horizons"]["h24h"]["source"] == "historical"
    assert len(fake.copies) == 1


def test_a_current_winner_without_an_artifact_uri_fails_loudly(monkeypatch):
    rows = [metric_row("lightgbm", "h1h", 0.3, with_artifact=False)]

    with pytest.raises(ValueError, match="no artifact_s3_uri"):
        run_handler(monkeypatch, rows, run_id="run-1")


def test_handler_fails_when_a_configured_horizon_was_never_trained(monkeypatch):
    monkeypatch.setattr(sel, "HORIZONS_MINUTES", "h1h:60,h2h:120")

    with pytest.raises(ValueError, match="h2h"):
        run_handler(monkeypatch, [metric_row("lightgbm", "h1h", 0.3)], run_id="run-1")


def test_handler_fails_on_an_empty_metrics_file(monkeypatch):
    fake = FakeS3("")
    monkeypatch.setattr(sel, "s3", fake)

    with pytest.raises(ValueError, match="no metric rows"):
        sel.lambda_handler({"metrics_s3_uri": METRICS_URI}, None)
