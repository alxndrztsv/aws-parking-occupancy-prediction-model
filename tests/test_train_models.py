import numpy as np
import pandas as pd
import pytest

from src import config
from src.models import train_models
from src.models.estimators.base import BaseEstimator
from src.models.estimators.ensemble import EnsembleEstimator

HORIZONS = list(config.HORIZONS_MINUTES)


class StubEstimator(BaseEstimator):
    """Predicts a constant, so metrics are deterministic and non-trivial.

    Defaults to the training mean; pass `mean` to fix the prediction instead, which
    is what the inference tests need. `clip=False` lets a test drive an out-of-range
    value through a caller that is supposed to clip it.
    """

    def __init__(self, name="stub", params=None, mean=None, clip=True):
        super().__init__(name, params or {})
        self._fixed_mean = mean
        self.clip = clip
        # Set here too, not only in fit(): the inference tests pickle a stub into a
        # bundle and unpickle it without ever fitting, and it must still predict.
        if mean is not None:
            self.mean = float(mean)

    def fit(self, X_train, y_train, X_valid, y_valid, categorical_features):
        self.mean = (
            float(np.asarray(y_train).mean())
            if self._fixed_mean is None
            else float(self._fixed_mean)
        )
        self.seen_features = list(X_train.columns)
        self.seen_categorical = list(categorical_features)
        self.model = object()

    def predict(self, X):
        preds = np.full(len(X), getattr(self, "mean", 0.0))
        return np.clip(preds, 0.0, 1.0) if self.clip else preds

    def get_feature_importances(self, feature_columns):
        return np.ones(len(feature_columns), dtype=float), "stub"


@pytest.fixture
def stub_registry(monkeypatch):
    """Swap the booster classes for stubs; the ensemble stays real."""
    created = []

    def fake_get_estimator(model_name):
        if model_name == "ensemble":
            est = EnsembleEstimator(
                name="ensemble",
                params={"models": ["lightgbm", "xgboost", "catboost"], "weights": {}},
            )
        else:
            est = StubEstimator(name=model_name, params={})
        created.append(est)
        return est

    monkeypatch.setattr(train_models, "get_estimator", fake_get_estimator)
    return created


def test_trains_every_model_on_every_horizon(tmp_path, splits, stub_registry):
    train, valid, test = splits
    model_dir, report_dir = tmp_path / "models", tmp_path / "reports"

    metrics_table = train_models.train_all_models(
        train, valid, test, model_dir=model_dir, report_dir=report_dir
    )

    expected_rows = (len(config.MODEL_TYPES) + 1) * len(HORIZONS)
    assert len(metrics_table) == expected_rows == 16
    assert sorted(metrics_table["model"].unique()) == [
        "catboost",
        "ensemble",
        "lightgbm",
        "xgboost",
    ]
    assert sorted(metrics_table["horizon"].unique()) == sorted(HORIZONS)
    assert list(metrics_table.columns) == [
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
    ]


def test_writes_one_bundle_and_one_importance_report_per_combination(
    tmp_path, splits, stub_registry
):
    train, valid, test = splits
    model_dir, report_dir = tmp_path / "models", tmp_path / "reports"

    train_models.train_all_models(train, valid, test, model_dir=model_dir, report_dir=report_dir)

    bundles = sorted(p.name for p in model_dir.glob("*.joblib"))
    assert len(bundles) == 16
    assert "lightgbm_h1h.joblib" in bundles
    assert "ensemble_h24h.joblib" in bundles

    importance = sorted(p.name for p in report_dir.glob("*_feature_importance.csv"))
    assert len(importance) == 16

    assert (report_dir / "model_metrics.csv").exists()
    written = pd.read_csv(report_dir / "model_metrics.csv")
    assert len(written) == 16
    # no unnamed pandas index column
    assert not any(c.startswith("Unnamed") for c in written.columns)


def test_metrics_are_finite_and_bounded_for_every_combination(tmp_path, splits, stub_registry):
    train, valid, test = splits

    metrics_table = train_models.train_all_models(
        train, valid, test, model_dir=tmp_path / "m", report_dir=tmp_path / "r"
    )

    for column in metrics_table.columns:
        if column in ("model", "horizon", "target"):
            continue
        values = metrics_table[column].to_numpy(dtype=float)
        assert np.isfinite(values).all(), f"{column} has non-finite values"
        assert (values >= 0).all(), f"{column} went negative"


def test_estimators_receive_the_categorical_feature_and_the_full_feature_list(
    tmp_path, splits, stub_registry
):
    train, valid, test = splits

    train_models.train_all_models(
        train,
        valid,
        test,
        model_names=("lightgbm",),
        model_dir=tmp_path / "m",
        report_dir=tmp_path / "r",
    )

    stub = next(e for e in stub_registry if isinstance(e, StubEstimator))
    assert stub.seen_categorical == ["terminal_id"]
    assert stub.seen_features[-1] == "terminal_id"
    assert len(stub.seen_features) == 24
    assert not any(c.startswith(("y_", "pred_")) for c in stub.seen_features)


def test_ensemble_reuses_fitted_members_instead_of_refitting(tmp_path, splits, stub_registry):
    train, valid, test = splits

    train_models.train_all_models(
        train, valid, test, model_dir=tmp_path / "m", report_dir=tmp_path / "r"
    )

    ensembles = [e for e in stub_registry if isinstance(e, EnsembleEstimator)]
    assert len(ensembles) == len(HORIZONS)
    for ensemble in ensembles:
        assert sorted(ensemble.estimators) == ["catboost", "lightgbm", "xgboost"]
        assert all(isinstance(member, StubEstimator) for member in ensemble.estimators.values())
    # 3 base models x 4 horizons; the ensemble itself fits nothing
    assert sum(1 for e in stub_registry if isinstance(e, StubEstimator)) == 3 * len(HORIZONS)


def test_model_names_argument_restricts_what_is_trained(tmp_path, splits, stub_registry):
    train, valid, test = splits

    metrics_table = train_models.train_all_models(
        train, valid, test, model_names=("lightgbm",), model_dir=tmp_path / "m"
    )

    assert len(metrics_table) == len(HORIZONS)
    assert set(metrics_table["model"]) == {"lightgbm"}


def test_directories_are_created_when_missing(tmp_path, splits, stub_registry):
    train, valid, test = splits
    model_dir = tmp_path / "deep" / "models"
    report_dir = tmp_path / "deep" / "reports"
    assert not model_dir.exists()

    train_models.train_all_models(train, valid, test, model_dir=model_dir, report_dir=report_dir)

    assert model_dir.is_dir() and report_dir.is_dir()


def test_directories_are_optional(tmp_path, splits, stub_registry):
    train, valid, test = splits

    metrics_table = train_models.train_all_models(train, valid, test, model_names=("lightgbm",))

    assert len(metrics_table) == len(HORIZONS)
    assert not list(tmp_path.iterdir())


def test_a_missing_target_column_raises_a_named_error(tmp_path, splits, stub_registry):
    train, valid, test = splits
    broken = train.drop(columns=[f"y_{HORIZONS[0]}"])

    with pytest.raises(KeyError):
        train_models.train_all_models(broken, valid, test, model_dir=tmp_path / "m")
