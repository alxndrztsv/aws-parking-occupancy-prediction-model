import numpy as np
import pandas as pd
import pytest

from src.models import artifacts
from src.models.estimators.base import BaseEstimator


class StubEstimator(BaseEstimator):
    """Stands in for a fitted booster so the artifact format can be tested alone."""

    def __init__(self, name="stub", params=None, importance_type="split"):
        super().__init__(name, params or {})
        self.importance_type = importance_type
        self.model = object()

    def fit(self, X_train, y_train, X_valid, y_valid, categorical_features):
        raise NotImplementedError

    def predict(self, X):
        return np.zeros(len(X))

    def get_feature_importances(self, feature_columns):
        return np.arange(len(feature_columns), dtype=float), self.importance_type


@pytest.fixture
def terminal_map():
    return pd.DataFrame(
        {
            "terminal_id": np.array([0, 1], dtype="int64"),
            "terminal_code": np.array([100, 200], dtype="int32"),
        }
    )


def test_bundle_carries_everything_inference_needs(tmp_path, terminal_map):
    features = ["hour", "terminal_id"]

    path = artifacts.save_model_bundle(
        model_dir=tmp_path,
        model=StubEstimator(),
        model_name="lightgbm",
        horizon_name="h1h",
        target="y_h1h",
        feature_columns=features,
        terminal_map=terminal_map,
    )

    assert path.name == "lightgbm_h1h.joblib"

    import joblib

    bundle = joblib.load(path)

    assert set(bundle) == {
        "model",
        "model_name",
        "horizon",
        "target",
        "feature_columns",
        "terminal_map",
        "categorical_features",
    }
    assert bundle["model_name"] == "lightgbm"
    assert bundle["horizon"] == "h1h"
    assert bundle["target"] == "y_h1h"
    assert bundle["feature_columns"] == features
    assert bundle["categorical_features"] == ("terminal_id",)
    # the estimator, not the raw library model: XGBoost needs its category mapping
    assert isinstance(bundle["model"], StubEstimator)
    assert list(bundle["terminal_map"].columns) == ["terminal_id", "terminal_code"]


def test_bundle_names_are_unique_per_model_and_horizon(tmp_path, terminal_map):
    for model_name in ("lightgbm", "ensemble"):
        for horizon in ("h1h", "h24h"):
            artifacts.save_model_bundle(
                model_dir=tmp_path,
                model=StubEstimator(),
                model_name=model_name,
                horizon_name=horizon,
                target=f"y_{horizon}",
                feature_columns=["hour"],
                terminal_map=terminal_map,
            )

    assert sorted(p.name for p in tmp_path.glob("*.joblib")) == [
        "ensemble_h1h.joblib",
        "ensemble_h24h.joblib",
        "lightgbm_h1h.joblib",
        "lightgbm_h24h.joblib",
    ]


def test_feature_importance_is_sorted_descending_and_labelled(tmp_path):
    features = ["a", "b", "c"]

    path = artifacts.save_feature_importance(
        report_dir=tmp_path,
        estimator=StubEstimator(importance_type="gain"),
        model_name="xgboost",
        horizon_name="h2h",
        feature_columns=features,
    )

    assert path.name == "xgboost_h2h_feature_importance.csv"

    table = pd.read_csv(path)

    assert list(table.columns) == ["feature", "importance", "importance_type"]
    assert table["feature"].tolist() == ["c", "b", "a"]
    assert table["importance"].is_monotonic_decreasing
    assert set(table["importance_type"]) == {"gain"}


def test_feature_importance_rejects_a_length_mismatch(tmp_path):
    class ShortImportance(StubEstimator):
        def get_feature_importances(self, feature_columns):
            return np.array([1.0]), "split"

    with pytest.raises(ValueError, match="importance values"):
        artifacts.save_feature_importance(
            report_dir=tmp_path,
            estimator=ShortImportance(),
            model_name="lightgbm",
            horizon_name="h1h",
            feature_columns=["a", "b"],
        )


def test_no_csv_index_column_is_written(tmp_path):
    """Spark/pandas round-trips break if an unnamed index column sneaks in."""
    artifacts.save_feature_importance(
        report_dir=tmp_path,
        estimator=StubEstimator(),
        model_name="catboost",
        horizon_name="h3h",
        feature_columns=["a"],
    )

    raw = (tmp_path / "catboost_h3h_feature_importance.csv").read_text().splitlines()

    assert raw[0] == "feature,importance,importance_type"
