from typing import Any

import numpy as np
import pandas as pd

from src.models.estimators.base import BaseEstimator


class EnsembleEstimator(BaseEstimator):
    """Ensemble blend that combines predictions from multiple estimators."""

    def __init__(self, name: str = "ensemble", params: dict[str, Any] | None = None):
        super().__init__(name, params or {})
        self.estimators: dict[str, BaseEstimator] = {}
        self.weights: dict[str, float] = self.params.get("weights", {})

    def add_estimator(self, name: str, estimator: BaseEstimator, weight: float = 1.0) -> None:
        """Attach an already-trained estimator to the ensemble."""
        self.estimators[name] = estimator
        if name not in self.weights:
            self.weights[name] = weight

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        categorical_features: list[str],
    ) -> None:
        """Fit all constituent estimators if they are not already attached."""
        # Deferred on purpose: registry.py imports this module to build
        # ESTIMATOR_CLASSES, so a module-level import of get_estimator here
        # closes a cycle and neither module finishes initialising.
        from src.models.registry import get_estimator

        base_models = self.params.get("models", ["lightgbm", "xgboost", "catboost"])
        for model_name in base_models:
            if model_name not in self.estimators:
                est = get_estimator(model_name)
                est.fit(X_train, y_train, X_valid, y_valid, categorical_features)
                self.add_estimator(model_name, est)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Weighted average of the members' predictions, clipped to [0, 1]."""
        if not self.estimators:
            raise ValueError("Ensemble has no fitted estimators.")

        preds = []
        total_weight = 0.0

        for name, est in self.estimators.items():
            w = float(self.weights.get(name, 1.0))
            preds.append(w * est.predict(X))
            total_weight += w

        if total_weight <= 0:
            total_weight = float(len(preds))

        blended = np.sum(preds, axis=0) / total_weight
        return np.clip(blended, 0.0, 1.0)

    def get_feature_importances(self, feature_columns: list[str]) -> tuple[np.ndarray, str]:
        """Return the average normalized feature importance across all estimators."""
        if not self.estimators:
            return np.zeros(len(feature_columns), dtype=float), "average_normalized"

        normalized_list = []
        for est in self.estimators.values():
            imp, _ = est.get_feature_importances(feature_columns)
            total = np.sum(imp)
            normalized_list.append(imp / total if total > 0 else imp)

        return np.mean(normalized_list, axis=0), "average_normalized"
