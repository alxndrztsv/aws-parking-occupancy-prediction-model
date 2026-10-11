from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class BaseEstimator(ABC):
    def __init__(self, name: str, params: dict[str, Any]):
        self.name = name
        self.params = params
        self.model = None

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        categorical_features: list[str],
    ) -> None:
        """Fit model with early stopping on validation data."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict values clipped to [0, 1]."""
        pass

    @abstractmethod
    def get_feature_importances(self, feature_columns: list[str]) -> tuple[np.ndarray, str]:
        """Return (importance_array, importance_type)."""
        pass
