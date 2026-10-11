import importlib.util
import io
import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("boto3")

from src import config  # noqa: E402
from src.models import train_models  # noqa: E402
from src.models.estimators.ensemble import EnsembleEstimator  # noqa: E402
from tests.test_train_models import StubEstimator  # noqa: E402

TRAIN_SCRIPT = Path(__file__).resolve().parents[1] / "sagemaker" / "train.py"


def load_train_module():
    spec = importlib.util.spec_from_file_location("sm_train", TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sm_train():
    return load_train_module()


def write_spark_channel(directory, frame, parts=2):
    """Mimic the Glue output: several part files plus a _SUCCESS marker."""
    directory.mkdir(parents=True, exist_ok=True)
    step = -(-len(frame) // parts)
    for index in range(parts):
        chunk = frame.iloc[index * step : (index + 1) * step]
        chunk.to_parquet(directory / f"part-{index:05d}.parquet", index=False)
    (directory / "_SUCCESS").write_text("")
    return directory


def test_load_split_reads_every_part_and_ignores_success(tmp_path, splits, sm_train):
    train, _, _ = splits
    channel = write_spark_channel(tmp_path / "train", train)

    got = sm_train.load_split(str(channel))

    assert len(got) == len(train)
    assert list(got.columns) == list(train.columns)


def test_load_split_rejects_a_missing_or_empty_channel(tmp_path, sm_train):
    with pytest.raises(ValueError, match="channel is not set"):
        sm_train.load_split(None)

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        sm_train.load_split(str(empty))


def test_run_stage_baseline_covers_every_horizon_and_split(splits, sm_train):
    train, valid, test = splits

    baseline_metrics, target_profiles, mase_scales = sm_train.run_stage_baseline(train, valid, test)

    horizons = list(config.HORIZONS_MINUTES)
    assert sorted(baseline_metrics["horizon"].unique()) == sorted(horizons)
    assert len(baseline_metrics) == len(horizons) * 2 * 3
    assert list(baseline_metrics.columns) == [
        "kind",
        "split",
        "model",
        "horizon",
        "mae",
        "rmse",
        "wape",
        "mase",
    ]

    assert len(target_profiles) == len(horizons) * 3
    assert sorted(target_profiles["split"].unique()) == ["test", "train", "valid"]
    assert (target_profiles["n_rows"] > 0).all()

    # the historical stage reuses these, so they must cover every horizon
    assert sorted(mase_scales) == sorted(horizons)
    assert all(scale > 0 for scale in mase_scales.values())


def test_run_stage_baseline_names_a_missing_horizon(splits, sm_train):
    train, valid, test = splits
    broken = train.drop(columns=["y_h1h"])

    with pytest.raises(KeyError, match="var.horizons_minutes"):
        sm_train.run_stage_baseline(broken, valid, test)


def test_run_stage_compare_joins_the_zero_share(splits, sm_train, tmp_path, monkeypatch):
    train, valid, test = splits

    baseline_metrics, target_profiles, _ = sm_train.run_stage_baseline(train, valid, test)

    def fake_get_estimator(model_name):
        if model_name == "ensemble":
            return EnsembleEstimator(
                name="ensemble", params={"models": ["lightgbm"], "weights": {}}
            )
        return StubEstimator(name=model_name, params={})

    monkeypatch.setattr(train_models, "get_estimator", fake_get_estimator)
    model_metrics = train_models.train_all_models(
        train, valid, test, model_names=("lightgbm",), model_dir=tmp_path / "m"
    )

    comparison = sm_train.run_stage_compare(model_metrics, baseline_metrics, target_profiles)

    assert list(comparison.columns) == [
        "kind",
        "split",
        "model",
        "horizon",
        "mae",
        "rmse",
        "wape",
        "mase",
        "zero_share",
    ]
    assert comparison["zero_share"].notna().all()
    assert set(comparison["kind"]) == {"model", "baseline"}


def test_parse_args_falls_back_to_the_sagemaker_environment(monkeypatch, sm_train):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--train",
            "/opt/ml/input/data/train",
            "--valid",
            "/opt/ml/input/data/valid",
            "--test",
            "/opt/ml/input/data/test",
            "--run-id",
            "202609-20261003060012",
            "--metrics-s3-uri",
            "s3://art/metrics/run/",
        ],
    )

    args = sm_train.parse_args()

    # argparse must accept the hyphenated names Step Functions sends as hyperparameters
    assert args.train == "/opt/ml/input/data/train"
    assert args.run_id == "202609-20261003060012"
    assert args.metrics_s3_uri == "s3://art/metrics/run/"


def test_upload_reports_uses_the_given_prefix(tmp_path, sm_train, monkeypatch):
    (tmp_path / "model_metrics.csv").write_text("model\n")
    (tmp_path / "not_a_report.txt").write_text("ignore me\n")

    calls = []
    monkeypatch.setattr(
        sm_train.s3,
        "upload_file",
        lambda path, bucket, key: calls.append((path, bucket, key)),
    )

    sm_train.upload_reports(tmp_path, "s3://artifacts-bucket/metrics/run-1/")

    assert calls == [
        (
            str(tmp_path / "model_metrics.csv"),
            "artifacts-bucket",
            "metrics/run-1/model_metrics.csv",
        )
    ]


# --- retained-model re-scoring ---

HISTORICAL_URI = "s3://artifacts-bucket/historical_models/"
RETAINED_KEY = "historical_models/lightgbm_h1h.joblib"


def test_split_s3_uri_keeps_object_keys_and_prefixes_apart(sm_train):
    """Regression: normalising a prefix by appending "/" turned the artifact key
    "x.joblib" into "x.joblib/", which is a different, nonexistent object."""
    assert sm_train._split_s3_uri("s3://b/historical_models/") == ("b", "historical_models/")
    assert sm_train._split_s3_uri("s3://b/historical_models") == ("b", "historical_models")
    assert sm_train._split_s3_uri("s3://b/h/m.joblib") == ("b", "h/m.joblib")
    assert sm_train._split_s3_uri("s3://b") == ("b", "")

    for bad in ("https://b/k", "b/k", "s3://"):
        with pytest.raises(ValueError):
            sm_train._split_s3_uri(bad)


@pytest.fixture
def retained_bundle(tmp_path, splits):
    """A real joblib bundle standing in for one promoted by a previous run."""
    from src.models import artifacts

    train, _, _ = splits
    terminal_map = (
        train[["terminal_code"]]
        .drop_duplicates()
        .sort_values("terminal_code")
        .reset_index(drop=True)
    )
    terminal_map.insert(0, "terminal_id", range(len(terminal_map)))
    terminal_map["terminal_id"] = terminal_map["terminal_id"].astype("int64")

    return artifacts.save_model_bundle(
        model_dir=tmp_path,
        model=StubEstimator(mean=0.25),
        model_name="lightgbm",
        horizon_name="h1h",
        target="y_h1h",
        feature_columns=["hour", "current_occupancy_rate", "terminal_id"],
        terminal_map=terminal_map,
    )


class HistoricalS3:
    """Minimal stand-in for the two calls run_stage_historical makes."""

    def __init__(self, registry, bundle_path=None, registry_error=None):
        self.registry = registry
        self.bundle_path = bundle_path
        self.registry_error = registry_error
        self.downloaded = []

    def get_object(self, Bucket, Key):
        if self.registry_error:
            raise self.registry_error
        return {"Body": io.BytesIO(json.dumps(self.registry).encode("utf-8"))}

    def download_file(self, Bucket, Key, Filename):
        self.downloaded.append((Bucket, Key))
        shutil.copyfile(self.bundle_path, Filename)


def no_such_key():
    from botocore.exceptions import ClientError

    return ClientError({"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "GetObject")


def test_historical_stage_is_a_no_op_on_the_first_run(sm_train, splits, monkeypatch):
    _, valid, test = splits
    monkeypatch.setattr(sm_train, "s3", HistoricalS3({}, registry_error=no_such_key()))

    got = sm_train.run_stage_historical(HISTORICAL_URI, valid, test, {"h1h": 0.5})

    assert got.empty


def test_historical_stage_rescores_the_retained_bundle(
    sm_train, splits, monkeypatch, retained_bundle
):
    _, valid, test = splits
    registry = {
        "horizons": {
            "h1h": {
                "model": "lightgbm",
                "source": "current",
                "artifact_s3_uri": f"s3://artifacts-bucket/{RETAINED_KEY}",
            }
        }
    }
    fake = HistoricalS3(registry, retained_bundle)
    monkeypatch.setattr(sm_train, "s3", fake)

    got = sm_train.run_stage_historical(HISTORICAL_URI, valid, test, {"h1h": 0.5})

    assert fake.downloaded == [("artifacts-bucket", RETAINED_KEY)]
    assert len(got) == 1
    row = got.iloc[0]
    assert row["model"] == "lightgbm"
    assert row["horizon"] == "h1h"
    assert row["source"] == "historical"
    assert row["artifact_s3_uri"] == f"s3://artifacts-bucket/{RETAINED_KEY}"
    # StubEstimator predicts 0.25 everywhere, so the score comes from this run's data
    assert row["valid_mae"] > 0
    assert row["valid_mase"] == pytest.approx(row["valid_mae"] / 0.5, abs=1e-3)


def test_historical_stage_skips_a_bundle_it_cannot_load(sm_train, splits, monkeypatch, tmp_path):
    """A retained bundle unpickled under newer library versions must not kill the run."""
    _, valid, test = splits
    corrupt = tmp_path / "corrupt.joblib"
    corrupt.write_bytes(b"not a joblib file")

    registry = {
        "horizons": {
            "h1h": {"model": "lightgbm", "artifact_s3_uri": f"s3://artifacts-bucket/{RETAINED_KEY}"}
        }
    }
    monkeypatch.setattr(sm_train, "s3", HistoricalS3(registry, corrupt))

    assert sm_train.run_stage_historical(HISTORICAL_URI, valid, test, {"h1h": 0.5}).empty


def test_historical_stage_skips_a_registry_entry_without_an_artifact(
    sm_train, splits, monkeypatch, retained_bundle
):
    _, valid, test = splits
    registry = {"horizons": {"h1h": {"model": "lightgbm"}}}
    fake = HistoricalS3(registry, retained_bundle)
    monkeypatch.setattr(sm_train, "s3", fake)

    assert sm_train.run_stage_historical(HISTORICAL_URI, valid, test, {"h1h": 0.5}).empty
    assert fake.downloaded == []
