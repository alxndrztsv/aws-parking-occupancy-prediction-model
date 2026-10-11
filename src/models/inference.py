"""
Inference contract for the bundles written by artifacts.save_model_bundle.

Used by the training job to re-score the model retained from a previous run on
the *current* validation split. That re-scoring is the only way the two are
comparable: each monthly run re-derives its split boundaries from max(timestamp),
so a WAPE recorded last month was measured on a different window.

A bundle is only valid inside the domain it was trained on: weekday operational
bins, and terminals present in its stored terminal_map.
"""

from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd

from src.models.metrics import evaluate_regression

REQUIRED_BUNDLE_KEYS: Final[tuple[str, ...]] = (
    "model",
    "model_name",
    "horizon",
    "target",
    "feature_columns",
    "terminal_map",
    "categorical_features",
)


def load_bundle(path: Path) -> dict[str, Any]:
    """Load a saved model bundle and check that it is complete."""
    bundle = joblib.load(path)

    missing = [key for key in REQUIRED_BUNDLE_KEYS if key not in bundle]
    if missing:
        raise ValueError(f"{path} is missing bundle keys: {missing}")

    return bundle


def predict_frame(
    bundle: dict[str, Any],
    df: pd.DataFrame,
    column_name: str = "pred",
) -> pd.DataFrame:
    """
    Score the rows of df the bundle can handle, returning them plus a prediction.

    Terminals absent from the bundle's terminal_map are dropped rather than raised
    on: re-scoring an old model on a newer month legitimately meets new terminals.
    The returned frame keeps the surviving rows' original columns, so the caller
    can score predictions against the matching target.
    """
    feature_columns = list(bundle["feature_columns"])
    terminal_map = bundle["terminal_map"]

    missing = [c for c in feature_columns if c != "terminal_id" and c not in df.columns]
    if missing:
        raise ValueError(f"Frame is missing required features: {missing}")

    encoded = df.merge(terminal_map, on="terminal_code", how="left")
    scored = encoded.dropna(subset=["terminal_id"]).copy()

    # The merge widens terminal_id to float64 wherever a terminal did not match.
    # XGBoost matches it against the int64 categories frozen during training, so
    # leaving it as float64 would turn every prediction into a missing-category one.
    scored["terminal_id"] = scored["terminal_id"].astype("int64")

    predictions = bundle["model"].predict(scored[feature_columns])
    scored[column_name] = np.clip(np.asarray(predictions, dtype=float), 0.0, 1.0)

    return scored


def rescore_bundle(
    bundle: dict[str, Any],
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mase_scale: float | None = None,
) -> dict[str, Any] | None:
    """
    Score one retained bundle on this run's splits.

    Returns a row in the same schema train_all_models produces, or None when the
    bundle cannot be applied - typically because the feature schema drifted
    between runs, which is a reason to skip the candidate, not to fail the job.
    """
    horizon = bundle["horizon"]
    target = bundle["target"]

    feature_columns = list(bundle["feature_columns"])
    required = [c for c in feature_columns if c != "terminal_id"]
    for frame in (valid_df, test_df):
        absent = [c for c in required if c not in frame.columns]
        if absent:
            print(f"  skipping {horizon}: frame is missing features {absent}")
            return None
        if target not in frame.columns:
            print(f"  skipping {horizon}: frame has no target {target}")
            return None

    try:
        valid_scored = predict_frame(bundle, valid_df)
        test_scored = predict_frame(bundle, test_df)
    except Exception as exc:  # a stale bundle must not kill the run
        print(f"  skipping {horizon}: could not score the retained bundle ({exc})")
        return None

    if valid_scored.empty:
        print(f"  skipping {horizon}: no validation row matched the retained terminal_map")
        return None

    valid_metrics = evaluate_regression(
        valid_scored[target], valid_scored["pred"].to_numpy(), mase_scale
    )
    test_metrics = (
        evaluate_regression(test_scored[target], test_scored["pred"].to_numpy(), mase_scale)
        if not test_scored.empty
        else {name: float("nan") for name in ("mae", "rmse", "wape", "mase")}
    )

    return {
        "model": bundle["model_name"],
        "horizon": horizon,
        "target": target,
        "valid_mae": valid_metrics["mae"],
        "valid_rmse": valid_metrics["rmse"],
        "valid_wape": valid_metrics["wape"],
        "valid_mase": valid_metrics["mase"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_wape": test_metrics["wape"],
        "test_mase": test_metrics["mase"],
    }
