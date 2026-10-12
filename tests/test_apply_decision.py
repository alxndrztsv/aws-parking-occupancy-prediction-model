import base64
import io
import json
import urllib.parse
from datetime import UTC, datetime, timedelta

import lambda_apply_decision as app
import pytest

BUCKET = "parking-ml-pipeline-dev-artifacts"
PENDING_KEY = "historical_models/pending.json"
REGISTRY_KEY = "historical_models/best_models.json"
PREV_REGISTRY_KEY = "historical_models/best_models.prev.json"
TOKEN = "a-token-with-enough-entropy-to-be-unguessable"


class FakeS3:
    """Dict-backed S3, so read-modify-write across calls behaves like the real thing."""

    def __init__(self, store=None):
        self.store = dict(store or {})
        self.copies = []

    def get_object(self, Bucket, Key):
        if Key not in self.store:
            raise app.ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "GetObject"
            )
        return {"Body": io.BytesIO(self.store[Key])}

    def put_object(self, Bucket, Key, Body, ContentType=None):
        self.store[Key] = Body if isinstance(Body, bytes) else Body.encode()

    def copy_object(self, Bucket, CopySource, Key):
        src = CopySource["Key"]
        if src not in self.store:
            raise app.ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "CopyObject"
            )
        self.copies.append((CopySource["Bucket"], src, Key))
        self.store[Key] = self.store[src]


def candidate(model, horizon, wape, source):
    prefix = "historical_models" if source == "historical" else "current_models/run-1"
    return {
        "model": model,
        "artifact_s3_uri": f"s3://{BUCKET}/{prefix}/{model}_{horizon}.joblib",
        "valid_wape": wape,
        "valid_mae": wape,
        "test_wape": wape,
    }


def pending_record(recommendations=None, **overrides):
    """A pending.json as lambda_select_model would have written it."""
    recommendations = recommendations or {"h1h": "replace", "h24h": "keep"}
    horizons = {}
    for horizon, recommendation in recommendations.items():
        horizons[horizon] = {
            "recommendation": recommendation,
            "improvement": 0.05 if recommendation == "replace" else 0.001,
            "reason": "test",
            "current": candidate("lightgbm", horizon, 0.040, "current"),
            "historical": candidate("catboost", horizon, 0.050, "historical"),
        }

    record = {
        "run_id": "run-1",
        "status": "pending",
        "created_at": "2026-10-03T06:00:00+00:00",
        "expires_at": (datetime.now(UTC) + timedelta(days=30)).isoformat(timespec="seconds"),
        "token": TOKEN,
        "selection_metric": "valid_wape",
        "threshold": 0.01,
        "source_metrics": f"s3://{BUCKET}/metrics/run-1/model_metrics.csv",
        "candidates": 20,
        "horizons": horizons,
    }
    record.update(overrides)
    return record


@pytest.fixture
def store():
    """A bucket holding the pending record plus every artifact it points at."""
    record = pending_record()
    objects = {PENDING_KEY: json.dumps(record).encode()}
    for horizon in record["horizons"]:
        for side in ("current", "historical"):
            model = record["horizons"][horizon][side]["model"]
            prefix = "historical_models" if side == "historical" else "current_models/run-1"
            objects[f"{prefix}/{model}_{horizon}.joblib"] = f"bundle-{side}-{horizon}".encode()
    objects[REGISTRY_KEY] = json.dumps({"horizons": {"old": True}}).encode()
    return objects


@pytest.fixture
def fake(store, monkeypatch):
    fake = FakeS3(store)
    monkeypatch.setattr(app, "s3", fake)
    return fake


def url_event(method="GET", run_id="run-1", token=TOKEN, decision=None):
    query = {"run_id": run_id, "token": token}
    if decision:
        query["decision"] = decision
    body = urllib.parse.urlencode({"decision": decision}) if decision else None
    return {
        "version": "2.0",
        "requestContext": {"http": {"method": method, "sourceIp": "203.0.113.7"}},
        "queryStringParameters": query,
        "body": body,
        "isBase64Encoded": False,
    }


# --- link validation ---


def test_a_wrong_token_is_refused(fake):
    with pytest.raises(app.LinkError, match="not valid"):
        app.validate(pending_record(), "run-1", "wrong-token")


def test_an_empty_token_is_refused_even_when_the_record_has_none(fake):
    """compare_digest must not turn 'both empty' into a match."""
    with pytest.raises(app.LinkError, match="not valid"):
        app.validate(pending_record(token=""), "run-1", "")


def test_a_superseded_run_is_refused_and_names_the_current_one(fake):
    """Defence in depth: a stale run_id is rejected even if the token somehow matches."""
    with pytest.raises(app.LinkError, match="superseded by run run-2"):
        app.validate(pending_record(run_id="run-2"), "run-1", TOKEN)


def test_a_real_stale_link_is_refused_by_the_token_check_first(fake):
    """What actually happens in production: a newer run overwrites pending.json with a
    fresh token, so an old link fails on the token and gets the generic message. The
    token is checked before the run id on purpose - an unauthenticated caller must
    not be able to learn which run is current, or whether one was approved."""
    newer = pending_record(run_id="run-2", token="a-completely-different-token")

    with pytest.raises(app.LinkError, match="not valid"):
        app.validate(newer, "run-1", TOKEN)


def test_an_already_decided_link_is_refused(fake):
    record = pending_record(status="approved", decided_at="2026-10-04T09:00:00+00:00")

    with pytest.raises(app.LinkError, match="already approved at 2026-10-04"):
        app.validate(record, "run-1", TOKEN)


def test_an_expired_link_is_refused(fake):
    past = (datetime.now(UTC) - timedelta(days=1)).isoformat(timespec="seconds")

    with pytest.raises(app.LinkError, match="expired"):
        app.validate(pending_record(expires_at=past), "run-1", TOKEN)


def test_a_link_within_its_ttl_is_accepted(fake):
    app.validate(pending_record(), "run-1", TOKEN)


def test_a_missing_pending_file_reports_no_pending_decision(monkeypatch):
    monkeypatch.setattr(app, "s3", FakeS3({}))

    with pytest.raises(app.LinkError, match="no pending decision"):
        app.load_pending()


# --- applying the decision ---


def test_approve_copies_only_the_recommended_horizons(fake):
    promoted = app.apply_decision(pending_record(), app.APPROVE)

    assert promoted == [("h1h", "lightgbm", "historical_models/lightgbm_h1h.joblib")]
    assert fake.copies == [
        (
            BUCKET,
            "current_models/run-1/lightgbm_h1h.joblib",
            "historical_models/lightgbm_h1h.joblib",
        )
    ]


def test_reject_copies_nothing_and_leaves_the_registry_alone(fake):
    before = fake.store[REGISTRY_KEY]

    promoted = app.apply_decision(pending_record(), app.REJECT)

    assert promoted == []
    assert fake.copies == []
    assert fake.store[REGISTRY_KEY] == before
    assert PREV_REGISTRY_KEY not in fake.store


def test_the_new_registry_keeps_every_horizon_including_the_kept_ones(fake):
    app.apply_decision(pending_record(), app.APPROVE)

    registry = json.loads(fake.store[REGISTRY_KEY])

    # h24h was "keep", but it must still be listed or the next training run would
    # have no historical to re-score for that horizon.
    assert set(registry["horizons"]) == {"h1h", "h24h"}
    assert registry["horizons"]["h1h"]["model"] == "lightgbm"
    assert registry["horizons"]["h1h"]["source"] == "current"
    assert registry["horizons"]["h24h"]["model"] == "catboost"
    assert registry["horizons"]["h24h"]["source"] == "historical"
    assert registry["run_id"] == "run-1"


def test_registry_uris_always_point_into_historical_models(fake):
    app.apply_decision(pending_record(), app.APPROVE)

    registry = json.loads(fake.store[REGISTRY_KEY])

    for horizon, entry in registry["horizons"].items():
        assert entry["artifact_s3_uri"] == (
            f"s3://{BUCKET}/historical_models/{entry['model']}_{horizon}.joblib"
        )


def test_the_previous_registry_is_saved_for_rollback(fake):
    app.apply_decision(pending_record(), app.APPROVE)

    assert json.loads(fake.store[PREV_REGISTRY_KEY]) == {"horizons": {"old": True}}


def test_a_first_run_without_a_previous_registry_still_works(monkeypatch):
    store = {PENDING_KEY: json.dumps(pending_record()).encode()}
    record = pending_record()
    for horizon in record["horizons"]:
        for side in ("current", "historical"):
            model = record["horizons"][horizon][side]["model"]
            prefix = "historical_models" if side == "historical" else "current_models/run-1"
            store[f"{prefix}/{model}_{horizon}.joblib"] = b"bundle"
    fake = FakeS3(store)
    monkeypatch.setattr(app, "s3", fake)

    app.apply_decision(pending_record(), app.APPROVE)

    assert PREV_REGISTRY_KEY not in fake.store
    assert json.loads(fake.store[REGISTRY_KEY])["horizons"].keys() == {"h1h", "h24h"}


def test_a_missing_bundle_surfaces_instead_of_writing_a_broken_registry(fake):
    del fake.store["current_models/run-1/lightgbm_h1h.joblib"]

    with pytest.raises(app.ClientError):
        app.apply_decision(pending_record(), app.APPROVE)


# --- the Function URL handler ---


def test_get_renders_a_confirmation_page_and_changes_nothing(fake):
    before = dict(fake.store)

    response = app.lambda_handler(url_event("GET"), None)

    assert response["statusCode"] == 200
    assert response["headers"]["Content-Type"].startswith("text/html")
    assert "REPLACE" in response["body"]
    assert "lightgbm" in response["body"] and "catboost" in response["body"]
    # two submit buttons, and no state change whatsoever
    assert response["body"].count('method="POST"') == 2
    assert fake.store == before
    assert fake.copies == []


def test_a_mail_scanner_following_the_link_cannot_promote_anything(fake):
    """Outlook SafeLinks and friends issue GETs; only POST may act."""
    response = app.lambda_handler(url_event("GET", decision="approve"), None)

    assert response["statusCode"] == 200
    assert fake.copies == []
    assert json.loads(fake.store[PENDING_KEY])["status"] == "pending"


def test_post_approve_promotes_and_records_the_decision(fake):
    response = app.lambda_handler(url_event("POST", decision="approve"), None)

    assert response["statusCode"] == 200
    assert len(fake.copies) == 1
    decided = json.loads(fake.store[PENDING_KEY])
    assert decided["status"] == "approved"
    assert decided["decided_by_ip"] == "203.0.113.7"
    assert "Promoted 1 model" in response["body"]
    assert PREV_REGISTRY_KEY in fake.store


def test_post_reject_records_the_decision_without_touching_models(fake):
    registry_before = fake.store[REGISTRY_KEY]

    response = app.lambda_handler(url_event("POST", decision="reject"), None)

    assert response["statusCode"] == 200
    assert fake.copies == []
    assert json.loads(fake.store[PENDING_KEY])["status"] == "rejected"
    assert fake.store[REGISTRY_KEY] == registry_before
    assert "Nothing was promoted" in response["body"]


def test_the_decision_is_single_use(fake):
    app.lambda_handler(url_event("POST", decision="approve"), None)

    second = app.lambda_handler(url_event("POST", decision="approve"), None)

    assert second["statusCode"] == 403
    assert "already approved" in second["body"]
    assert len(fake.copies) == 1


def test_an_unknown_decision_is_a_client_error(fake):
    response = app.lambda_handler(url_event("POST", decision="maybe"), None)

    assert response["statusCode"] == 400
    assert fake.copies == []
    assert json.loads(fake.store[PENDING_KEY])["status"] == "pending"


def test_a_bad_token_never_reaches_the_action(fake):
    response = app.lambda_handler(url_event("POST", token="nope", decision="approve"), None)

    assert response["statusCode"] == 403
    assert fake.copies == []
    assert json.loads(fake.store[PENDING_KEY])["status"] == "pending"


def test_a_superseded_link_explains_which_run_replaced_it(fake):
    fake.store[PENDING_KEY] = json.dumps(pending_record(run_id="run-9")).encode()

    response = app.lambda_handler(url_event("POST", decision="approve"), None)

    assert response["statusCode"] == 403
    assert "run-9" in response["body"]
    assert fake.copies == []


def test_a_missing_pending_file_renders_a_page_not_a_stack_trace(monkeypatch):
    monkeypatch.setattr(app, "s3", FakeS3({}))

    response = app.lambda_handler(url_event("GET"), None)

    assert response["statusCode"] == 403
    assert "no pending decision" in response["body"].lower()


def test_a_base64_encoded_form_body_is_decoded(fake):
    event = url_event("POST")
    event["body"] = base64.b64encode(b"decision=approve").decode()
    event["isBase64Encoded"] = True

    response = app.lambda_handler(event, None)

    assert response["statusCode"] == 200
    assert len(fake.copies) == 1
    assert json.loads(fake.store[PENDING_KEY])["status"] == "approved"


def test_split_s3_uri_rejects_incomplete_uris():
    for bad in ("https://b/k", "b/k", "s3://", "s3://bucket-only"):
        with pytest.raises(app.LinkError):
            app._split_s3_uri(bad)

    assert app._split_s3_uri(f"s3://{BUCKET}/a/b.joblib") == (BUCKET, "a/b.joblib")
