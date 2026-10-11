import math

import numpy as np
import pandas as pd
import pytest

from src.models import baselines, metrics


def test_perfect_prediction_scores_zero_everywhere():
    got = metrics.evaluate_regression([0.25, 0.75], np.array([0.25, 0.75]), 0.5)

    assert got == {"mae": 0.0, "rmse": 0.0, "wape": 0.0, "mase": 0.0}


def test_mase_is_mae_divided_by_the_naive_scale():
    got = metrics.evaluate_regression([0.0, 1.0], np.array([0.5, 0.5]), 0.25)

    assert got["mae"] == 0.5
    assert got["mase"] == 2.0


def test_wape_and_mase_are_nan_when_undefined():
    all_zero = metrics.evaluate_regression([0.0, 0.0], np.array([0.1, 0.0]))

    assert all_zero["mae"] == 0.05
    assert math.isnan(all_zero["wape"])
    assert math.isnan(all_zero["mase"])

    no_scale = metrics.evaluate_regression([0.4, 0.6], np.array([0.5, 0.5]))

    assert no_scale["wape"] == 0.2
    assert math.isnan(no_scale["mase"])


def test_metrics_are_rounded_to_four_decimals():
    got = metrics.evaluate_regression([0.0], [1 / 3])

    assert got["mae"] == 0.3333
    assert got["rmse"] == 0.3333


def test_prediction_column_skips_rows_with_a_null_side():
    df = pd.DataFrame(
        {
            "y_h1h": [0.5, 0.5, np.nan, 0.5],
            "pred": [0.5, np.nan, 0.9, 0.7],
        }
    )

    got = metrics.evaluate_prediction_column(df, "y_h1h", "pred")

    # only rows (0.5, 0.5) and (0.5, 0.7) survive -> (0.0 + 0.2) / 2
    assert got["mae"] == pytest.approx(0.1)
    assert got["wape"] == pytest.approx(0.2)


def test_prediction_column_without_usable_rows_is_all_nan():
    df = pd.DataFrame({"y_h1h": [np.nan, np.nan], "pred": [0.5, 0.6]})

    got = metrics.evaluate_prediction_column(df, "y_h1h", "pred")

    assert all(math.isnan(value) for value in got.values())


def test_naive_scale_prefers_the_seasonal_column():
    df = pd.DataFrame(
        {
            "y_h1h": [0.5, 0.4, 0.9, np.nan],
            "occupancy_same_time_1d": [0.4, 0.4, 0.7, 0.2],
            "current_occupancy_rate": [0.1, 0.1, 0.1, 0.1],
        }
    )

    assert metrics.naive_scale(df, "y_h1h") == pytest.approx(0.1)
    assert metrics.naive_scale(df, "y_h1h", ("current_occupancy_rate",)) == pytest.approx(0.5)
    assert math.isnan(metrics.naive_scale(df, "y_h1h", ("absent",)))


def test_naive_scale_prefers_the_horizon_specific_column():
    """The denominator must follow the horizon, otherwise MASE is not comparable."""
    df = pd.DataFrame(
        {
            "y_h1h": [0.6, 0.4],
            "pred_seasonal_naive_h1h": [0.5, 0.4],
            "occupancy_same_time_1d": [0.0, 0.0],
        }
    )

    # horizon-specific column wins: (|0.6 - 0.5| + |0.4 - 0.4|) / 2 = 0.05,
    # while the generic fallback would have given 0.3
    assert metrics.naive_scale(df, "y_h1h") == pytest.approx(0.05)


def test_seasonal_naive_baseline_scores_exactly_one_mase():
    """The MASE denominator and the baseline must be the same quantity."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "y_h1h": rng.uniform(0, 1, n),
            "current_occupancy_rate": rng.uniform(0, 1, n),
            "occupancy_same_time_1d": rng.uniform(0, 1, n),
            "occupancy_same_time_7d": rng.uniform(0, 1, n),
        }
    )

    scale = metrics.naive_scale(df, "y_h1h")
    got = baselines.evaluate_baselines(df, target="y_h1h", mase_scale=scale)

    row = got.loc[got["model"] == "pred_yesterday"].iloc[0]

    assert row["mase"] == pytest.approx(1.0)
    assert list(got.columns) == ["model", "mae", "rmse", "wape", "mase"]


def test_baselines_by_split_stays_tidy_and_labels_every_split():
    frame = pd.DataFrame(
        {
            "y_h1h": [0.4, 0.6],
            "current_occupancy_rate": [0.3, 0.5],
            "occupancy_same_time_1d": [0.5, 0.5],
            "occupancy_same_time_7d": [0.2, 0.7],
        }
    )

    got = baselines.evaluate_baselines_by_split(
        {"valid": frame, "test": frame}, target="y_h1h", horizon_name="h1h", mase_scale=0.2
    )

    assert list(got.columns) == baselines.TIDY_COLUMNS
    assert len(got) == 6
    assert sorted(got["split"].unique()) == ["test", "valid"]
    assert (got["kind"] == "baseline").all()
    assert (got["horizon"] == "h1h").all()


def test_target_profile_reports_the_zero_share():
    df = pd.DataFrame({"y_h1h": [0.0, 0.0, 0.5, 1.0, np.nan]})

    got = metrics.target_profile(df, "y_h1h", 0.25)

    assert got == {
        "n_rows": 4,
        "zero_share": 0.5,
        "mean_target": 0.375,
        "naive_mae": 0.25,
    }


def test_target_profile_on_an_empty_split():
    df = pd.DataFrame({"y_h1h": [np.nan, np.nan]})

    got = metrics.target_profile(df, "y_h1h")

    assert got["n_rows"] == 0
    assert math.isnan(got["zero_share"])
    assert math.isnan(got["naive_mae"])


def test_comparison_table_stacks_models_over_baselines():
    model_metrics = pd.DataFrame(
        {
            "model": ["lightgbm", "xgboost"],
            "horizon": ["h1h", "h1h"],
            "target": ["y_h1h", "y_h1h"],
            "valid_mae": [0.10, 0.12],
            "valid_rmse": [0.20, 0.21],
            "valid_wape": [0.30, 0.31],
            "valid_mase": [0.80, 0.90],
            "test_mae": [0.11, 0.13],
            "test_rmse": [0.20, 0.21],
            "test_wape": [0.30, 0.31],
            "test_mase": [0.85, 0.95],
        }
    )
    baseline_metrics = pd.DataFrame(
        {
            "kind": ["baseline", "baseline"],
            "split": ["valid", "test"],
            "model": ["pred_current", "pred_current"],
            "horizon": ["h1h", "h1h"],
            "mae": [0.20, 0.21],
            "rmse": [0.30, 0.31],
            "wape": [0.50, 0.51],
            "mase": [1.02, 1.03],
        }
    )

    got = metrics.build_comparison_table(model_metrics, baseline_metrics)

    assert list(got.columns) == [
        "kind",
        "split",
        "model",
        "horizon",
        "mae",
        "rmse",
        "wape",
        "mase",
    ]
    assert len(got) == 6
    assert sorted(got["kind"].unique()) == ["baseline", "model"]
    assert got["mae"].notna().all()
    # sorted by horizon, split, then ascending mae
    assert got.loc[got["split"] == "valid", "model"].tolist() == [
        "lightgbm",
        "xgboost",
        "pred_current",
    ]
