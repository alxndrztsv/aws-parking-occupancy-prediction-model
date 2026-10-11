import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.models.estimators.base import BaseEstimator


class XGBoostEstimator(BaseEstimator):
    def fit(self, X_train, y_train, X_valid, y_valid, categorical_features):
        # XGBoost encodes a categorical value as its position in the category
        # list, so every frame - train, validation, test and later inference -
        # must be encoded against the categories seen during training.
        self.categories = {
            col: sorted(X_train[col].dropna().unique())
            for col in categorical_features
            if col in X_train.columns
        }

        self.model = XGBRegressor(**self.params)
        self.model.fit(
            self._encode_categories(X_train),
            y_train,
            eval_set=[(self._encode_categories(X_valid), y_valid)],
            verbose=False,
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(self._encode_categories(X))
        return np.clip(preds, 0.0, 1.0)

    def _encode_categories(self, X: pd.DataFrame) -> pd.DataFrame:
        X_encoded = X.copy()
        for col, categories in self.categories.items():
            X_encoded[col] = pd.Categorical(X_encoded[col], categories=categories)
        return X_encoded

    def get_feature_importances(self, feature_columns):
        return np.asarray(self.model.feature_importances_, dtype=float), "gain"
