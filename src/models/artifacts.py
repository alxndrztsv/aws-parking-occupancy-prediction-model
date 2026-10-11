from pathlib import Path

import joblib
import pandas as pd

from src.models.estimators.base import BaseEstimator
from src.models.registry import CATEGORICAL_FEATURES


def save_model_bundle(
    model_dir: Path,
    model: BaseEstimator,
    model_name: str,
    horizon_name: str,
    target: str,
    feature_columns: list[str],
    terminal_map: pd.DataFrame,
) -> Path:
    """Persist a trained estimator together with everything needed to reuse it."""
    artifact_path = model_dir / f"{model_name}_{horizon_name}.joblib"

    joblib.dump(
        {
            "model": model,
            "model_name": model_name,
            "horizon": horizon_name,
            "target": target,
            "feature_columns": feature_columns,
            "terminal_map": terminal_map,
            "categorical_features": CATEGORICAL_FEATURES,
        },
        artifact_path,
    )

    return artifact_path


def save_feature_importance(
    report_dir: Path,
    estimator: BaseEstimator,
    model_name: str,
    horizon_name: str,
    feature_columns: list[str],
) -> Path:
    """Write per-feature importance, sorted descending."""
    importance, importance_type = estimator.get_feature_importances(feature_columns)

    if len(importance) != len(feature_columns):
        raise ValueError(
            f"{model_name}/{horizon_name}: got {len(importance)} importance "
            f"values for {len(feature_columns)} feature columns"
        )

    table = (
        pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": importance,
                "importance_type": [importance_type] * len(feature_columns),
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    report_path = report_dir / f"{model_name}_{horizon_name}_feature_importance.csv"
    table.to_csv(report_path, index=False)

    return report_path
