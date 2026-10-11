import numpy as np
import pandas as pd
import pytest

from src import config


def build_feature_frame(n_rows=2000, n_terminals=4, seed=0):
    """A frame shaped like the Glue silver_to_gold feature table output.

    Column order matches what Spark writes, because get_feature_columns derives the
    model input order from it.
    """
    rng = np.random.default_rng(seed)
    per_terminal = n_rows // n_terminals
    n_rows = per_terminal * n_terminals

    # Terminals are interleaved rather than blocked, so every split sees every
    # terminal - as the real data does. encode_terminal_codes drops rows whose
    # terminal was never seen in train, so a blocked layout would empty valid/test.
    terminal_index = np.arange(n_rows) % n_terminals

    frame = pd.DataFrame(
        {
            "terminal_code": (100 + terminal_index).astype("int32"),
            "total_spaces": rng.integers(5, 50, n_terminals)[terminal_index].astype("int32"),
            "date": pd.to_datetime("2026-01-01") + pd.to_timedelta(np.arange(n_rows) // 144, "D"),
            "slot_24h": (np.arange(n_rows) % 144).astype("int32"),
            "timestamp": pd.to_datetime("2026-01-01")
            + pd.to_timedelta(np.arange(n_rows) * 10, "m"),
            "hour": ((np.arange(n_rows) % 144) // 6).astype("int32"),
            "minute": (((np.arange(n_rows) % 144) % 6) * 10).astype("int32"),
            "day_of_week": rng.integers(1, 8, n_rows).astype("int32"),
            "is_weekend": rng.integers(0, 2, n_rows).astype(bool),
            "is_bank_holiday": np.zeros(n_rows, dtype=bool),
            "is_closed_day": rng.integers(0, 2, n_rows).astype(bool),
        }
    )

    for bins in (1, 2, 3, 6, 12, 18):
        frame[f"occupancy_lag_{bins}"] = rng.random(n_rows).astype("float32")
    for days in (1, 7, 14):
        frame[f"occupancy_same_time_{days}d"] = rng.random(n_rows).astype("float32")
    frame["occupancy_same_time_last_7d_mean"] = rng.random(n_rows).astype("float32")
    for name in (
        "occupancy_rolling_mean_1h",
        "occupancy_rolling_std_1h",
        "occupancy_rolling_mean_2h",
        "occupancy_rolling_mean_3h",
    ):
        frame[name] = rng.random(n_rows).astype("float32")

    frame["month"] = np.ones(n_rows, dtype="int32")
    frame["is_training_origin"] = rng.integers(0, 2, n_rows).astype(bool)

    for horizon in config.HORIZONS_MINUTES:
        frame[f"y_{horizon}"] = rng.random(n_rows).astype("float32")
        frame[f"pred_seasonal_naive_{horizon}"] = rng.random(n_rows).astype("float32")

    frame["current_occupancy_rate"] = rng.random(n_rows).astype("float32")
    frame["current_active_sessions"] = rng.integers(0, 20, n_rows).astype("int32")
    frame["current_free_spaces"] = rng.integers(0, 20, n_rows).astype("int32")

    return frame


# Exactly the 24 columns the reference pipeline feeds to the models, in order.
EXPECTED_FEATURES = [
    "total_spaces",
    "slot_24h",
    "hour",
    "minute",
    "day_of_week",
    "occupancy_lag_1",
    "occupancy_lag_2",
    "occupancy_lag_3",
    "occupancy_lag_6",
    "occupancy_lag_12",
    "occupancy_lag_18",
    "occupancy_same_time_1d",
    "occupancy_same_time_7d",
    "occupancy_same_time_14d",
    "occupancy_same_time_last_7d_mean",
    "occupancy_rolling_mean_1h",
    "occupancy_rolling_std_1h",
    "occupancy_rolling_mean_2h",
    "occupancy_rolling_mean_3h",
    "month",
    "current_occupancy_rate",
    "current_active_sessions",
    "current_free_spaces",
    "terminal_id",
]


@pytest.fixture(scope="session")
def expected_features():
    return list(EXPECTED_FEATURES)


@pytest.fixture(scope="session")
def splits():
    """train / valid / test frames covering the same terminals."""
    frame = build_feature_frame()
    return (
        frame.iloc[:1200].reset_index(drop=True),
        frame.iloc[1200:1600].reset_index(drop=True),
        frame.iloc[1600:].reset_index(drop=True),
    )
