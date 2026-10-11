from typing import Final

import pandas as pd

from src.models import metrics

BASELINE_MODELS: Final[tuple[str, ...]] = (
    "pred_current",
    "pred_yesterday",
    "pred_last_week",
)

BASELINE_SOURCES: Final[dict[str, str]] = {
    "pred_current": "current_occupancy_rate",
    "pred_yesterday": "occupancy_same_time_1d",
    "pred_last_week": "occupancy_same_time_7d",
}

TIDY_COLUMNS: Final[list[str]] = [
    "kind",
    "split",
    "model",
    "horizon",
    "mae",
    "rmse",
    "wape",
    "mase",
]


def evaluate_baselines(
    df: pd.DataFrame,
    target: str,
    prediction_columns: tuple[str, ...] = BASELINE_MODELS,
    mase_scale: float | None = None,
) -> pd.DataFrame:
    """Evaluate baseline predictions with the shared regression metrics."""
    scored = df.assign(
        **{
            baseline: df[source]
            for baseline, source in BASELINE_SOURCES.items()
            if source in df.columns
        }
    )

    rows = [
        {
            "model": prediction,
            **metrics.evaluate_prediction_column(scored, target, prediction, mase_scale),
        }
        for prediction in prediction_columns
    ]

    return pd.DataFrame(rows)


def evaluate_baselines_by_split(
    splits: dict[str, pd.DataFrame],
    target: str,
    horizon_name: str,
    mase_scale: float | None = None,
) -> pd.DataFrame:
    """
    Score every baseline on every split in tidy long format.

    The MASE denominator is computed once on the training split and passed in, so
    baseline and model scores are directly comparable.
    """
    frames = []

    for split_name, df in splits.items():
        part = evaluate_baselines(df, target=target, mase_scale=mase_scale)
        part.insert(0, "kind", "baseline")
        part.insert(1, "split", split_name)
        part["horizon"] = horizon_name
        frames.append(part[TIDY_COLUMNS])

    return pd.concat(frames, ignore_index=True)
