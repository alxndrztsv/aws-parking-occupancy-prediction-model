import csv
import io
import json
import math
import urllib.parse

import lambda_select_model as sel
import pytest

ARTIFACTS = "parking-ml-pipeline-dev-artifacts"
METRICS_URI = f"s3://{ARTIFACTS}/metrics/run-1/"
APPROVAL_URL = "https://abc123.lambda-url.eu-west-1.on.aws/"


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

    def copy_object(self, **kwargs):
        self.copies.append(kwargs)


class FakeSNS:
    def __init__(self):
        self.published = []

    def publish(self, **kwargs):
        self.published.append(kwargs)


def csv_body(rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


@pytest.fixture
def fake_aws(monkeypatch):
    monkeypatch.setattr(sel, "HORIZONS_MINUTES", "")
    monkeypatch.setattr(sel, "APPROVAL_URL", APPROVAL_URL)
    monkeypatch.setattr(sel, "MODEL_DECISION_TOPIC_ARN", "arn:aws:sns:eu-west-1:1:model-decisions")
    sns = FakeSNS()
    monkeypatch.setattr(sel, "sns", sns)
    return sns


def run_handler(monkeypatch, rows, **event):
    fake = FakeS3(csv_body(rows))
    monkeypatch.setattr(sel, "s3", fake)
    payload = sel.lambda_handler({"metrics_s3_uri": METRICS_URI, **event}, None)
    return payload, fake


def pending_of(fake, key="historical_models/pending.json"):
    for _bucket, stored_key, body, _ in fake.puts:
        if stored_key == key:
            return json.loads(body)
    raise AssertionError(f"{key} was not written")


def make_pending(monkeypatch, rows):
    return sel.build_decision(rows)


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
    ]

    assert sel.select_best(rows)["h1h"]["model"] == "ensemble"


def test_ties_are_broken_by_model_name_so_the_result_is_deterministic():
    rows = [metric_row("xgboost", "h1h", 0.3), metric_row("lightgbm", "h1h", 0.3)]

    assert sel.select_best(rows)["h1h"]["model"] == "lightgbm"
    assert sel.select_best(list(reversed(rows)))["h1h"]["model"] == "lightgbm"


@pytest.mark.parametrize("bad", ["", "nan", "NaN", None, "not-a-number"])
def test_unusable_metrics_lose_to_a_real_number(bad):
    rows = [metric_row("broken", "h1h", bad), metric_row("working", "h1h", 0.9)]

    assert sel.select_best(rows)["h1h"]["model"] == "working"


def test_source_filter_separates_currents_from_historicals():
    rows = [
        metric_row("lightgbm", "h1h", 0.30, source="current"),
        metric_row("catboost", "h1h", 0.20, source="historical"),
    ]

    assert sel.select_best(rows, source="current")["h1h"]["model"] == "lightgbm"
    assert sel.select_best(rows, source="historical")["h1h"]["model"] == "catboost"
    # unfiltered still sees both and picks the lower one
    assert sel.select_best(rows)["h1h"]["model"] == "catboost"


def test_rows_without_a_model_or_horizon_are_skipped():
    rows = [
        metric_row("lightgbm", "h1h", 0.3),
        {"model": "", "horizon": "h1h", "valid_wape": "0.1", "source": "current"},
        {"model": "ghost", "horizon": "", "valid_wape": "0.1", "source": "current"},
    ]

    assert list(sel.select_best(rows)) == ["h1h"]


def test_expected_horizons_parses_the_terraform_variable(monkeypatch):
    monkeypatch.setattr(sel, "HORIZONS_MINUTES", "h1h:60,h2h:120, h24h:1440")
    assert sel.expected_horizons() == {"h1h", "h2h", "h24h"}

    monkeypatch.setattr(sel, "HORIZONS_MINUTES", "")
    assert sel.expected_horizons() == set()


def test_metric_value_maps_nan_to_infinity():
    assert sel._metric_value({"valid_wape": "nan"}, "valid_wape") == math.inf
    assert sel._metric_value({}, "valid_wape") == math.inf
    assert sel._metric_value({"valid_wape": "0.25"}, "valid_wape") == 0.25


# --- recommendation ---


def test_first_run_with_no_historical_recommends_replace():
    entry = sel.build_decision([metric_row("lightgbm", "h1h", 0.30)])["h1h"]

    assert entry["recommendation"] == "replace"
    assert entry["historical"] is None
    assert entry["improvement"] is None
    assert "no model is retained" in entry["reason"]


def test_improvement_beyond_the_threshold_recommends_replace():
    rows = [
        metric_row("lightgbm", "h1h", 0.0441, source="current"),
        metric_row("catboost", "h1h", 0.0462, source="historical"),
    ]

    entry = sel.build_decision(rows, threshold=0.01)["h1h"]

    assert entry["recommendation"] == "replace"
    # (0.0462 - 0.0441) / 0.0462 = 4.545%
    assert entry["improvement"] == pytest.approx(0.045455, abs=1e-5)
    assert entry["current"]["model"] == "lightgbm"
    assert entry["historical"]["model"] == "catboost"


def test_improvement_inside_the_noise_band_keeps_the_historical():
    """h2h once scored 0.0482 vs 0.0483 - a statistical tie must not promote."""
    rows = [
        metric_row("ensemble", "h2h", 0.0481, source="current"),
        metric_row("ensemble", "h2h", 0.0483, source="historical"),
    ]

    entry = sel.build_decision(rows, threshold=0.01)["h2h"]

    assert entry["recommendation"] == "keep"
    assert entry["improvement"] == pytest.approx(0.00414, abs=1e-4)
    assert "noise band" in entry["reason"]


def test_a_worse_current_keeps_the_historical():
    rows = [
        metric_row("lightgbm", "h1h", 0.0500, source="current"),
        metric_row("catboost", "h1h", 0.0440, source="historical"),
    ]

    entry = sel.build_decision(rows, threshold=0.01)["h1h"]

    assert entry["recommendation"] == "keep"
    assert entry["improvement"] < 0


def test_threshold_zero_promotes_on_any_improvement():
    rows = [
        metric_row("a", "h1h", 0.04829, source="current"),
        metric_row("b", "h1h", 0.04830, source="historical"),
    ]

    assert sel.build_decision(rows, threshold=0.0)["h1h"]["recommendation"] == "replace"
    assert sel.build_decision(rows, threshold=0.01)["h1h"]["recommendation"] == "keep"


def test_exactly_at_the_threshold_still_replaces():
    rows = [
        metric_row("a", "h1h", 0.099, source="current"),
        metric_row("b", "h1h", 0.100, source="historical"),
    ]

    assert sel.build_decision(rows, threshold=0.01)["h1h"]["recommendation"] == "replace"


def test_a_perfect_historical_is_never_replaced():
    """Division-by-zero guard: an historical at 0.0 has no meaningful improvement."""
    rows = [
        metric_row("a", "h1h", 0.0, source="current"),
        metric_row("b", "h1h", 0.0, source="historical"),
    ]

    entry = sel.build_decision(rows, threshold=0.01)["h1h"]

    assert entry["recommendation"] == "keep"
    assert entry["improvement"] is None
    assert "unusable" in entry["reason"]


def test_an_unusable_current_keeps_the_historical():
    rows = [
        metric_row("a", "h1h", "nan", source="current"),
        metric_row("b", "h1h", 0.05, source="historical"),
    ]

    entry = sel.build_decision(rows, threshold=0.01)["h1h"]

    assert entry["recommendation"] == "keep"
    assert entry["improvement"] is None


def test_decision_is_made_per_horizon():
    rows = [
        metric_row("a", "h1h", 0.040, source="current"),
        metric_row("b", "h1h", 0.050, source="historical"),
        metric_row("c", "h24h", 0.050, source="current"),
        metric_row("d", "h24h", 0.050, source="historical"),
    ]

    got = sel.build_decision(rows, threshold=0.01)

    assert got["h1h"]["recommendation"] == "replace"
    assert got["h24h"]["recommendation"] == "keep"


def test_candidate_records_carry_the_artifact_uri_and_metrics():
    current = sel.build_decision([metric_row("lightgbm", "h1h", 0.0441, mae=0.0438)])["h1h"][
        "current"
    ]

    assert current["model"] == "lightgbm"
    assert current["artifact_s3_uri"] == (
        f"s3://{ARTIFACTS}/current_models/run-1/lightgbm_h1h.joblib"
    )
    assert current["valid_wape"] == 0.0441
    assert current["valid_mae"] == 0.0438
    assert set(current) >= set(sel.REPORTED_METRICS)


# --- email ---


def _pending(rows, run_id="run-1"):
    return {
        "run_id": run_id,
        "token": "tok",
        "selection_metric": "valid_wape",
        "threshold": 0.01,
        "expires_at": "2026-11-09T06:00:00+00:00",
        "source_metrics": f"s3://{ARTIFACTS}/metrics/run-1/model_metrics.csv",
        "horizons": sel.build_decision(rows),
    }


def test_email_lists_every_horizon_with_its_improvement():
    pending = _pending(
        [
            metric_row("lightgbm", "h1h", 0.0441, source="current"),
            metric_row("catboost", "h1h", 0.0462, source="historical"),
            metric_row("ensemble", "h2h", 0.0481, source="current"),
            metric_row("ensemble", "h2h", 0.0483, source="historical"),
        ]
    )

    subject, message = sel.build_email(pending, APPROVAL_URL.rstrip("/"))

    # The per-horizon recommendation lives on the confirmation page; the email
    # carries the count in the subject and the improvement percentages in the table.
    assert "replace 1/2" in subject
    assert len(subject) <= 100, "SNS rejects a Subject longer than 100 characters"
    assert "h1h" in message and "h2h" in message
    assert "catboost" in message and "lightgbm" in message
    assert "+4.5%" in message, "h1h: (0.0462 - 0.0441) / 0.0462"
    assert "+0.4%" in message, "h2h is inside the noise band but still shown"

    link = [ln.strip() for ln in message.splitlines() if ln.strip().startswith("https://")][0]
    query = urllib.parse.parse_qs(urllib.parse.urlparse(link).query)
    assert query == {"run_id": ["run-1"], "token": ["tok"]}


def test_email_explains_that_nothing_was_promoted_yet():
    _subject, message = sel.build_email(
        _pending([metric_row("lightgbm", "h1h", 0.0441)]), APPROVAL_URL.rstrip("/")
    )

    assert "Nothing has been promoted" in message
    assert "Link expires 2026-11-09" in message


def test_a_long_run_id_cannot_produce_an_invalid_subject():
    subject, _ = sel.build_email(
        _pending([metric_row("a", "h1h", 0.04)], run_id="x" * 200), APPROVAL_URL.rstrip("/")
    )

    assert len(subject) <= 100


# --- handler ---


def test_handler_promotes_nothing(fake_aws, monkeypatch):
    """The whole point of the gate: no copy_object, no best_models.json."""
    rows = [
        metric_row("lightgbm", "h1h", 0.040, source="current"),
        metric_row("catboost", "h1h", 0.050, source="historical"),
    ]

    _payload, fake = run_handler(monkeypatch, rows, run_id="run-1")

    assert fake.copies == []
    assert all(key != "historical_models/best_models.json" for _b, key, *_ in fake.puts)


def test_handler_writes_pending_to_the_fixed_key_and_the_audit_copy(fake_aws, monkeypatch):
    _payload, fake = run_handler(monkeypatch, [metric_row("lightgbm", "h1h", 0.040)], run_id="r")

    assert [key for _b, key, *_ in fake.puts] == [
        "historical_models/pending.json",
        "metrics/run-1/pending.json",
    ]

    pending = pending_of(fake)
    assert pending == pending_of(fake, "metrics/run-1/pending.json")
    assert pending["run_id"] == "r"
    assert pending["status"] == "pending"
    assert len(pending["token"]) >= 32
    assert pending["threshold"] == pytest.approx(sel.PROMOTION_THRESHOLD)
    assert pending["source_metrics"] == f"s3://{ARTIFACTS}/metrics/run-1/model_metrics.csv"


def test_handler_publishes_one_email_containing_the_link(fake_aws, monkeypatch):
    payload, _fake = run_handler(monkeypatch, [metric_row("lightgbm", "h1h", 0.040)], run_id="r")

    assert len(fake_aws.published) == 1
    published = fake_aws.published[0]
    assert published["TopicArn"] == "arn:aws:sns:eu-west-1:1:model-decisions"
    assert len(published["Subject"]) <= 100
    assert payload["approval_url"] in published["Message"]

    # The trailing slash is stripped before the query string is appended, and the
    # link must carry exactly the run id and the token stored in pending.json.
    parsed = urllib.parse.urlparse(payload["approval_url"])
    assert f"{parsed.scheme}://{parsed.netloc}/" == APPROVAL_URL
    assert urllib.parse.parse_qs(parsed.query) == {
        "run_id": ["r"],
        "token": [pending_of(_fake)["token"]],
    }


def test_handler_reports_which_horizons_it_would_replace(fake_aws, monkeypatch):
    rows = [
        metric_row("a", "h1h", 0.040, source="current"),
        metric_row("b", "h1h", 0.050, source="historical"),
        metric_row("c", "h24h", 0.050, source="current"),
        metric_row("d", "h24h", 0.050, source="historical"),
    ]

    payload, _fake = run_handler(monkeypatch, rows, run_id="r")

    assert payload["replace"] == ["h1h"]
    assert payload["keep"] == ["h24h"]
    assert payload["status"] == "pending"


def test_each_run_gets_a_fresh_token(fake_aws, monkeypatch):
    rows = [metric_row("lightgbm", "h1h", 0.040)]

    _p1, fake1 = run_handler(monkeypatch, rows, run_id="r1")
    _p2, fake2 = run_handler(monkeypatch, rows, run_id="r2")

    assert pending_of(fake1)["token"] != pending_of(fake2)["token"]


def test_handler_fails_when_a_configured_horizon_was_never_trained(fake_aws, monkeypatch):
    monkeypatch.setattr(sel, "HORIZONS_MINUTES", "h1h:60,h2h:120")

    with pytest.raises(ValueError, match="h2h"):
        run_handler(monkeypatch, [metric_row("lightgbm", "h1h", 0.3)], run_id="r")


def test_handler_fails_on_an_empty_metrics_file(fake_aws, monkeypatch):
    monkeypatch.setattr(sel, "s3", FakeS3(""))

    with pytest.raises(ValueError, match="no metric rows"):
        sel.lambda_handler({"metrics_s3_uri": METRICS_URI}, None)


def test_handler_still_works_without_a_topic_configured(monkeypatch):
    monkeypatch.setattr(sel, "MODEL_DECISION_TOPIC_ARN", "")
    monkeypatch.setattr(sel, "APPROVAL_URL", APPROVAL_URL)

    _payload, fake = run_handler(monkeypatch, [metric_row("a", "h1h", 0.04)], run_id="r")

    assert len(fake.puts) == 2
