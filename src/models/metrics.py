from typing import Final

import numpy as np
import pandas as pd

NAIVE_COLUMNS: Final[tuple[str, ...]] = (
    "occupancy_same_time_1d",
    "current_occupancy_rate",
)

METRIC_COLUMNS: Final[tuple[str, ...]] = ("mae", "rmse", "wape", "mase")


def evaluate_regression(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    mase_scale: float | None = None,
) -> dict[str, float]:
    """Calculate MAE, RMSE, WAPE and MASE."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    error = y_t - y_p
    abs_error = np.abs(error)

    mae = float(np.mean(abs_error))
    rmse = float(np.sqrt(np.mean(error**2)))

    true_total = float(np.sum(np.abs(y_t)))
    wape = float(np.sum(abs_error) / true_total) if true_total > 0 else float("nan")

    mase = mae / float(mase_scale) if mase_scale and mase_scale > 0 else float("nan")

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "wape": round(wape, 4),
        "mase": round(mase, 4),
    }


def evaluate_prediction_column(
    df: pd.DataFrame,
    target: str,
    prediction_column: str,
    mase_scale: float | None = None,
) -> dict[str, float]:
    """Score one prediction column of a frame with evaluate_regression."""
    pair = df[[target, prediction_column]].dropna()

    if pair.empty:
        return {name: float("nan") for name in METRIC_COLUMNS}

    return evaluate_regression(
        pair[target].to_numpy(),
        pair[prediction_column].to_numpy(),
        mase_scale,
    )


def naive_scale(
    df: pd.DataFrame,
    target: str,
    naive_columns: tuple[str, ...] = NAIVE_COLUMNS,
) -> float:
    """In-sample MAE of the naive forecast, used as the MASE denominator."""
    horizon_name = target.removeprefix("y_")
    candidates = (f"pred_seasonal_naive_{horizon_name}",) + tuple(naive_columns)
    naive_column = next((col for col in candidates if col in df.columns), None)

    if naive_column is None:
        return float("nan")

    pair = df[[target, naive_column]].dropna()

    if pair.empty:
        return float("nan")

    scale = (pair[target] - pair[naive_column]).abs().mean()

    return float(scale) if pd.notna(scale) else float("nan")


def target_profile(
    df: pd.DataFrame,
    target: str,
    mase_scale: float | None = None,
) -> dict[str, float | int]:
    """Describe the target on one split: row count, share of zeros, mean, naive MAE."""
    series = df[target].dropna()
    naive_mae = float(mase_scale) if mase_scale else float("nan")

    if series.empty:
        return {
            "n_rows": 0,
            "zero_share": float("nan"),
            "mean_target": float("nan"),
            "naive_mae": round(naive_mae, 4),
        }

    return {
        "n_rows": int(len(series)),
        "zero_share": round(float((series == 0.0).mean()), 4),
        "mean_target": round(float(series.mean()), 4),
        "naive_mae": round(naive_mae, 4),
    }


def build_comparison_table(
    model_metrics: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Stack model and baseline scores into one tidy table."""
    frames = [
        pd.DataFrame(
            {
                "kind": "model",
                "split": split_name,
                "model": model_metrics["model"].to_numpy(),
                "horizon": model_metrics["horizon"].to_numpy(),
                "mae": model_metrics[f"{split_name}_mae"].to_numpy(),
                "rmse": model_metrics[f"{split_name}_rmse"].to_numpy(),
                "wape": model_metrics[f"{split_name}_wape"].to_numpy(),
                "mase": model_metrics[f"{split_name}_mase"].to_numpy(),
            }
        )
        for split_name in ("valid", "test")
    ]
    frames.append(baseline_metrics)

    comparison = pd.concat(frames, ignore_index=True)
    comparison = comparison[["kind", "split", "model", "horizon", *METRIC_COLUMNS]]

    return comparison.sort_values(
        ["horizon", "split", "mae"], na_position="last", ignore_index=True
    )
