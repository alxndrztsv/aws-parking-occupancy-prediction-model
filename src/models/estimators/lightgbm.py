import lightgbm as lgb
import numpy as np
import pandas as pd

from src.models.estimators.base import BaseEstimator


class LightGBMEstimator(BaseEstimator):
    def fit(self, X_train, y_train, X_valid, y_valid, categorical_features):
        params = self.params.copy()
        early_stopping_rounds = params.pop("early_stopping_rounds", 50)
        self.model = lgb.LGBMRegressor(**params)

        self.model.fit(
            X_train,
            y_train,
            eval_X=X_valid,
            eval_y=y_valid,
            categorical_feature=categorical_features,
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                lgb.log_evaluation(0),
            ],
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        best_iter = getattr(self.model, "best_iteration_", None)
        preds = (
            self.model.predict(X, num_iteration=best_iter) if best_iter else self.model.predict(X)
        )
        return np.clip(preds, 0.0, 1.0)

    def get_feature_importances(self, feature_columns):
        return np.asarray(self.model.feature_importances_, dtype=float), "split"
