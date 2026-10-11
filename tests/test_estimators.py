import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")
pytest.importorskip("xgboost")
pytest.importorskip("catboost")
pytest.importorskip("sklearn")

from src.models.estimators.catboost import CatBoostEstimator  # noqa: E402
from src.models.estimators.ensemble import EnsembleEstimator  # noqa: E402
from src.models.estimators.lightgbm import LightGBMEstimator  # noqa: E402
from src.models.estimators.xgboost import XGBoostEstimator  # noqa: E402

FAST_XGB = dict(
    objective="reg:absoluteerror",
    eval_metric="mae",
    tree_method="hist",
    enable_categorical=True,
    n_estimators=25,
    max_depth=3,
    random_state=42,
    n_jobs=1,
    early_stopping_rounds=10,
)
FAST_CB = dict(
    loss_function="MAE",
    iterations=25,
    depth=3,
    task_type="CPU",
    thread_count=1,
    random_state=42,
    early_stopping_rounds=10,
)
FAST_LGB = dict(
    objective="regression_l1",
    metric="mae",
    n_estimators=25,
    learning_rate=0.1,
    num_leaves=15,
    verbose=-1,
    n_jobs=1,
    early_stopping_rounds=7,
)


def frame(terminals, n, seed):
    r = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "f0": r.random(n),
            "terminal_id": r.choice(terminals, n).astype("int64"),
        }
    )
    return X, pd.Series(r.random(n))


@pytest.fixture(scope="module")
def splits():
    X_tr, y_tr = frame([0, 1, 2], 300, 1)
    X_va, y_va = frame([0, 1], 120, 2)
    X_te, _ = frame([0, 2], 120, 3)
    return X_tr, y_tr, X_va, y_va, X_te


@pytest.mark.parametrize(
    "estimator_cls,params",
    [
        (XGBoostEstimator, FAST_XGB),
        (CatBoostEstimator, FAST_CB),
        (LightGBMEstimator, FAST_LGB),
    ],
)
def test_predict_works_when_a_split_misses_a_terminal(estimator_cls, params, splits):
    """Regression: terminal_id reached predict() as int64 while fit() saw a category."""
    X_tr, y_tr, X_va, y_va, X_te = splits
    est = estimator_cls(name=estimator_cls.__name__, params=dict(params))

    est.fit(X_tr, y_tr, X_va, y_va, ["terminal_id"])

    for X in (X_va, X_te):
        pred = est.predict(X)
        assert pred.shape == (len(X),)
        assert np.isfinite(pred).all()
        assert pred.min() >= 0.0
        assert pred.max() <= 1.0


@pytest.mark.parametrize(
    "estimator_cls,params,expected_type",
    [
        (LightGBMEstimator, FAST_LGB, "split"),
        (XGBoostEstimator, FAST_XGB, "gain"),
        (CatBoostEstimator, FAST_CB, "PredictionValuesChange"),
    ],
)
def test_feature_importances_match_the_feature_count(estimator_cls, params, expected_type, splits):
    X_tr, y_tr, X_va, y_va, _ = splits
    est = estimator_cls(name=estimator_cls.__name__, params=dict(params))
    est.fit(X_tr, y_tr, X_va, y_va, ["terminal_id"])

    importance, importance_type = est.get_feature_importances(list(X_tr.columns))

    assert importance_type == expected_type
    assert len(importance) == len(X_tr.columns)


def test_xgboost_category_mapping_is_frozen_at_training(splits):
    """XGBoost encodes a category by its position, so all frames must share one list."""
    X_tr, y_tr, X_va, y_va, _ = splits
    est = XGBoostEstimator(name="xgboost", params=dict(FAST_XGB))

    est.fit(X_tr, y_tr, X_va, y_va, ["terminal_id"])

    assert est.categories["terminal_id"] == [0, 1, 2]
    # valid holds only terminals 0 and 1, yet the mapping still spans all three
    assert est._encode_categories(X_va)["terminal_id"].cat.categories.tolist() == [0, 1, 2]


def test_catboost_reuses_the_categorical_columns_from_fit(splits):
    """Regression: predict() hardcoded the column name instead of reusing fit()."""
    X_tr, y_tr, X_va, y_va, _ = splits
    est = CatBoostEstimator(name="catboost", params=dict(FAST_CB))

    est.fit(X_tr, y_tr, X_va, y_va, ["terminal_id", "absent_column"])

    assert est.cat_features == ["terminal_id"]
    assert est._encode_categories(X_va)["terminal_id"].dtype == np.int64


def test_lightgbm_fit_leaves_the_shared_params_intact(splits):
    """Regression: popping early_stopping_rounds from self.params broke a refit."""
    X_tr, y_tr, X_va, y_va, _ = splits
    est = LightGBMEstimator(name="lightgbm", params=dict(FAST_LGB))

    est.fit(X_tr, y_tr, X_va, y_va, ["terminal_id"])
    first = est.predict(X_va)

    assert est.params["early_stopping_rounds"] == 7

    est.fit(X_tr, y_tr, X_va, y_va, ["terminal_id"])
    second = est.predict(X_va)

    assert np.allclose(first, second)


def test_ensemble_blends_attached_members_without_refitting(splits):
    X_tr, y_tr, X_va, y_va, _ = splits

    members = {}
    for cls, params, name in (
        (LightGBMEstimator, FAST_LGB, "lightgbm"),
        (XGBoostEstimator, FAST_XGB, "xgboost"),
    ):
        est = cls(name=name, params=dict(params))
        est.fit(X_tr, y_tr, X_va, y_va, ["terminal_id"])
        members[name] = est

    ensemble = EnsembleEstimator(name="ensemble", params={"weights": {}})
    for name, est in members.items():
        ensemble.add_estimator(name, est)

    blended = ensemble.predict(X_va)
    expected = np.clip(
        (members["lightgbm"].predict(X_va) + members["xgboost"].predict(X_va)) / 2, 0.0, 1.0
    )

    assert np.allclose(blended, expected)
    assert ensemble.get_feature_importances(list(X_tr.columns))[1] == "average_normalized"


def test_ensemble_without_members_refuses_to_predict():
    with pytest.raises(ValueError, match="no fitted estimators"):
        EnsembleEstimator().predict(pd.DataFrame({"f0": [0.5]}))
