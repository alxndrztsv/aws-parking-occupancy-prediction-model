import numpy as np
import pandas as pd
import pytest

from src.models import artifacts, inference
from tests.test_train_models import StubEstimator

FEATURES = ["hour", "current_occupancy_rate", "terminal_id"]


@pytest.fixture
def terminal_map():
    return pd.DataFrame(
        {
            "terminal_id": np.array([0, 1], dtype="int64"),
            "terminal_code": np.array([100, 200], dtype="int32"),
        }
    )


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "terminal_code": np.array([100, 200, 100], dtype="int32"),
            "hour": np.array([9, 10, 11], dtype="int32"),
            "current_occupancy_rate": np.array([0.2, 0.4, 0.6], dtype="float32"),
            "y_h1h": np.array([0.3, 0.5, 0.7], dtype="float32"),
        }
    )


@pytest.fixture
def bundle_path(tmp_path, terminal_map):
    return artifacts.save_model_bundle(
        model_dir=tmp_path,
        model=StubEstimator(mean=0.25),
        model_name="lightgbm",
        horizon_name="h1h",
        target="y_h1h",
        feature_columns=FEATURES,
        terminal_map=terminal_map,
    )


def test_load_bundle_accepts_a_complete_bundle(bundle_path):
    bundle = inference.load_bundle(bundle_path)

    assert bundle["model_name"] == "lightgbm"
    assert bundle["horizon"] == "h1h"
    assert bundle["feature_columns"] == FEATURES


def test_load_bundle_rejects_an_incomplete_one(tmp_path):
    import joblib

    path = tmp_path / "broken.joblib"
    joblib.dump({"model": object(), "model_name": "lightgbm"}, path)

    with pytest.raises(ValueError, match="missing bundle keys"):
        inference.load_bundle(path)


def test_predict_frame_returns_predictions_aligned_with_the_input(bundle_path, frame):
    bundle = inference.load_bundle(bundle_path)

    scored = inference.predict_frame(bundle, frame)

    assert len(scored) == 3
    assert scored["pred"].tolist() == [0.25, 0.25, 0.25]
    # the target survives so the caller can score against it
    assert scored["y_h1h"].tolist() == pytest.approx([0.3, 0.5, 0.7])


def test_predict_frame_drops_terminals_the_bundle_never_saw(bundle_path, frame):
    """A retained model meets new terminals every month; that must not be fatal."""
    bundle = inference.load_bundle(bundle_path)
    wider = pd.concat([frame, frame.head(1).assign(terminal_code=np.int32(999))], ignore_index=True)

    scored = inference.predict_frame(bundle, wider)

    assert len(scored) == 3
    assert 999 not in set(scored["terminal_code"])


def test_predict_frame_hands_the_estimator_an_int64_terminal_id(bundle_path, frame):
    """Regression: the left merge widens terminal_id to float64, which makes
    XGBoost match it against int64 categories and silently emit missing-category
    predictions for every row."""
    seen = {}

    class Recording(StubEstimator):
        def predict(self, X):
            seen["dtype"] = X["terminal_id"].dtype
            return super().predict(X)

    bundle = inference.load_bundle(bundle_path)
    bundle["model"] = Recording(mean=0.25)

    inference.predict_frame(bundle, frame)

    assert seen["dtype"] == np.int64


def test_predict_frame_names_a_missing_feature(bundle_path, frame):
    bundle = inference.load_bundle(bundle_path)

    with pytest.raises(ValueError, match="missing required features"):
        inference.predict_frame(bundle, frame.drop(columns=["hour"]))


def test_predict_frame_clips_to_the_valid_occupancy_range(bundle_path, frame):
    bundle = inference.load_bundle(bundle_path)
    bundle["model"] = StubEstimator(mean=5.0, clip=False)

    scored = inference.predict_frame(bundle, frame)

    assert scored["pred"].max() <= 1.0
    assert scored["pred"].min() >= 0.0


def test_rescore_bundle_produces_a_metrics_row(bundle_path, frame):
    bundle = inference.load_bundle(bundle_path)

    row = inference.rescore_bundle(bundle, frame, frame, mase_scale=0.5)

    assert row["model"] == "lightgbm"
    assert row["horizon"] == "h1h"
    assert row["target"] == "y_h1h"
    assert set(row) == {
        "model",
        "horizon",
        "target",
        "valid_mae",
        "valid_rmse",
        "valid_wape",
        "valid_mase",
        "test_mae",
        "test_rmse",
        "test_wape",
        "test_mase",
    }
    # StubEstimator predicts 0.25 everywhere: |0.3-.25| + |0.5-.25| + |0.7-.25| = 0.75
    assert row["valid_mae"] == pytest.approx(0.25, abs=1e-4)
    assert row["valid_mase"] == pytest.approx(row["valid_mae"] / 0.5, abs=1e-4)


def test_rescore_bundle_skips_when_the_feature_schema_drifted(bundle_path, frame):
    """A schema change between runs must skip the candidate, not fail a 3-hour job."""
    bundle = inference.load_bundle(bundle_path)
    drifted = frame.drop(columns=["current_occupancy_rate"])

    assert inference.rescore_bundle(bundle, drifted, drifted, 0.5) is None


def test_rescore_bundle_skips_when_no_terminal_matches(bundle_path, frame):
    bundle = inference.load_bundle(bundle_path)
    strangers = frame.assign(terminal_code=np.int32(999))

    assert inference.rescore_bundle(bundle, strangers, strangers, 0.5) is None


def test_rescore_bundle_skips_when_the_target_disappeared(bundle_path, frame):
    bundle = inference.load_bundle(bundle_path)

    assert inference.rescore_bundle(bundle, frame.drop(columns=["y_h1h"]), frame, 0.5) is None


def test_rescore_bundle_survives_an_unpredictable_model(bundle_path, frame):
    class Exploding(StubEstimator):
        def predict(self, X):
            raise RuntimeError("unpickled under a different library version")

    bundle = inference.load_bundle(bundle_path)
    bundle["model"] = Exploding()

    assert inference.rescore_bundle(bundle, frame, frame, 0.5) is None
