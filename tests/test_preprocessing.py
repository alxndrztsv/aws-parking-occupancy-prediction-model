import numpy as np
import pandas as pd

from src.models import preprocessing as model_input


def test_get_feature_columns_keeps_only_numeric_and_boolean():
    df = pd.DataFrame(
        {
            "terminal_code": pd.Series([1], dtype="int32"),
            "timestamp": pd.to_datetime(["2026-01-01"]),
            "date": pd.to_datetime(["2026-01-01"]),
            "is_weekend": pd.Series([True]),
            "is_training_origin": pd.Series([True]),
            "address": pd.Series(["street"]),
            "zone_desc": pd.Series(["z"], dtype="category"),
            "hour": pd.Series([9], dtype="int8"),
            "occupancy_rate": pd.Series([0.5], dtype="float64"),
            "is_free": pd.Series([True]),
            "terminal_id": pd.Series([7], dtype="int64"),
            "y_h1h": pd.Series([0.5]),
            "pred_current": pd.Series([0.5]),
        }
    )

    got = model_input.get_feature_columns(df)

    assert got == ["hour", "occupancy_rate", "is_free", "terminal_id"]


def test_feature_table_yields_exactly_the_expected_features(splits, expected_features):
    """Regression guard: a stray numeric column silently becomes a 25th feature."""
    train, _, _ = splits
    train_enc, _, _, _ = model_input.encode_terminal_codes(train, train, train)

    got = model_input.get_feature_columns(train_enc)

    assert got == expected_features
    assert len(got) == 24
    assert got[-1] == "terminal_id"


def test_minutes_from_midnight_would_leak_into_the_features():
    """silver_to_gold.py builds timestamps from this helper column; if it is not
    dropped it is numeric and lands in the model input."""
    df = pd.DataFrame({"hour": [1], "minutes_from_midnight": [60]})

    assert model_input.get_feature_columns(df) == ["hour", "minutes_from_midnight"]


def test_encode_terminal_codes_maps_only_training_terminals():
    train = pd.DataFrame({"terminal_code": np.array([300, 100, 200, 100], dtype="int32")})
    valid = pd.DataFrame({"terminal_code": np.array([100, 999], dtype="int32")})
    test = pd.DataFrame({"terminal_code": np.array([200], dtype="int32")})

    train_enc, valid_enc, test_enc, terminal_map = model_input.encode_terminal_codes(
        train, valid, test
    )

    assert terminal_map["terminal_code"].tolist() == [100, 200, 300]
    assert terminal_map["terminal_id"].tolist() == [0, 1, 2]
    # with_row_index in the reference put terminal_id first; the bundle must match.
    assert list(terminal_map.columns) == ["terminal_id", "terminal_code"]
    assert terminal_map["terminal_id"].dtype == np.int64

    # train keeps every row, unseen terminals are dropped from valid and test
    assert len(train_enc) == 4
    assert sorted(train_enc["terminal_id"].unique()) == [0, 1, 2]
    assert valid_enc["terminal_code"].tolist() == [100]
    assert valid_enc["terminal_id"].tolist() == [0]
    assert test_enc["terminal_id"].tolist() == [1]


def test_terminal_id_stays_int64_after_dropping_unseen_terminals():
    """Regression: the left merge widens terminal_id to float64, which makes
    XGBoost match it against int64 categories and turn every value into NaN."""
    train = pd.DataFrame({"terminal_code": np.array([100, 200], dtype="int32")})
    valid = pd.DataFrame({"terminal_code": np.array([100, 999], dtype="int32")})

    train_enc, valid_enc, test_enc, _ = model_input.encode_terminal_codes(train, valid, valid)

    for frame in (train_enc, valid_enc, test_enc):
        assert frame["terminal_id"].dtype == np.int64


def test_encode_terminal_codes_keeps_existing_columns():
    train = pd.DataFrame({"terminal_code": np.array([100], dtype="int32"), "occupancy_rate": [0.5]})

    train_enc, _, _, _ = model_input.encode_terminal_codes(train, train, train)

    assert set(train_enc.columns) == {"terminal_code", "occupancy_rate", "terminal_id"}


def test_prepare_pandas_split_returns_the_requested_columns(splits):
    train, _, _ = splits
    train_enc, _, _, _ = model_input.encode_terminal_codes(train, train, train)
    features = model_input.get_feature_columns(train_enc)

    X, y = model_input.prepare_pandas_split(train_enc, features, "y_h1h")

    assert list(X.columns) == features
    assert y.name == "y_h1h"
    assert len(X) == len(y) == len(train_enc)
