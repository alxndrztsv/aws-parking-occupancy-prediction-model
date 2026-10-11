from typing import Final

from src.models.estimators.base import BaseEstimator
from src.models.estimators.catboost import CatBoostEstimator
from src.models.estimators.ensemble import EnsembleEstimator
from src.models.estimators.lightgbm import LightGBMEstimator
from src.models.estimators.xgboost import XGBoostEstimator
from src.models.hyperparams import load_hyperparameters

CATEGORICAL_FEATURES: Final[tuple[str, ...]] = ("terminal_id",)
_HYPERPARAMS = load_hyperparameters()

ESTIMATOR_CLASSES: Final[dict[str, type[BaseEstimator]]] = {
    "lightgbm": LightGBMEstimator,
    "xgboost": XGBoostEstimator,
    "catboost": CatBoostEstimator,
    "ensemble": EnsembleEstimator,
}
MODEL_NAMES: Final[tuple[str, ...]] = tuple(ESTIMATOR_CLASSES.keys())


def get_estimator(model_name: str) -> BaseEstimator:
    if model_name not in ESTIMATOR_CLASSES:
        raise ValueError(f"Unknown model: {model_name}. Available: {MODEL_NAMES}")

    models_config = _HYPERPARAMS.get("models", {})
    params = dict(models_config.get(model_name, {}))

    if "random_state" not in params and "random_state" in _HYPERPARAMS:
        params["random_state"] = _HYPERPARAMS["random_state"]
    if "early_stopping_rounds" not in params and "early_stopping_rounds" in _HYPERPARAMS:
        params["early_stopping_rounds"] = _HYPERPARAMS["early_stopping_rounds"]

    return ESTIMATOR_CLASSES[model_name](name=model_name, params=params)
