from typing import Final

import numpy as np
import pandas as pd

NON_FEATURE_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "terminal_code",
        "timestamp",
        "date",
        "is_training_origin",
        "is_weekend",
        "is_bank_holiday",
        "is_closed_day",
    }
)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric and boolean feature column names for modeling."""
    feature_columns = []

    for name in df.columns:
        if name in NON_FEATURE_COLUMNS:
            continue

        if name.startswith(("y_", "pred_")):
            continue

        dtype = df[name].dtype
        if pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
            feature_columns.append(name)

    return feature_columns


def _encode_with(
    df: pd.DataFrame,
    terminal_map: pd.DataFrame,
    drop_unseen: bool,
) -> pd.DataFrame:
    encoded = df.merge(terminal_map, on="terminal_code", how="left")

    if drop_unseen:
        encoded = encoded.dropna(subset=["terminal_id"])

    encoded["terminal_id"] = encoded["terminal_id"].astype("int64")

    return encoded


def encode_terminal_codes(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Encode terminal_code into integer terminal_id using training terminals only."""
    terminal_codes = (
        pd.Series(train["terminal_code"].dropna().unique(), name="terminal_code")
        .sort_values(kind="stable")
        .reset_index(drop=True)
    )
    terminal_map = pd.DataFrame({"terminal_code": terminal_codes})
    terminal_map.insert(0, "terminal_id", np.arange(len(terminal_map), dtype="int64"))

    train_enc = _encode_with(train, terminal_map, drop_unseen=False)
    valid_enc = _encode_with(valid, terminal_map, drop_unseen=True)
    test_enc = _encode_with(test, terminal_map, drop_unseen=True)

    return train_enc, valid_enc, test_enc, terminal_map


def prepare_pandas_split(
    df: pd.DataFrame,
    feature_columns: list[str],
    target: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Slice one split into the estimator's (X, y) pair."""
    return df[feature_columns], df[target]
