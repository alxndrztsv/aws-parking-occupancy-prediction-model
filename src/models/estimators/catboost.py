import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.models.estimators.base import BaseEstimator


class CatBoostEstimator(BaseEstimator):
    """CatBoost regression adapter implementing BaseEstimator interface."""

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        categorical_features: list[str],
    ) -> None:
        params = self.params.copy()
        early_stopping_rounds = params.pop("early_stopping_rounds", 50)

        self.cat_features = [col for col in categorical_features if col in X_train.columns]

        self.model = CatBoostRegressor(
            **params,
            early_stopping_rounds=early_stopping_rounds,
            verbose=False,
        )

        self.model.fit(
            self._encode_categories(X_train),
            y_train,
            eval_set=(self._encode_categories(X_valid), y_valid),
            cat_features=self.cat_features,
            use_best_model=True,
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(self._encode_categories(X))
        return np.clip(preds, 0.0, 1.0)

    def _encode_categories(self, X: pd.DataFrame) -> pd.DataFrame:
        X_encoded = X.copy()
        for col in self.cat_features:
            X_encoded[col] = X_encoded[col].astype("int64")
        return X_encoded

    def get_feature_importances(self, feature_columns):
        importances = np.asarray(self.model.get_feature_importance(), dtype=float)
        return importances, "PredictionValuesChange"
