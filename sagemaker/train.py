"""
SageMaker entry point: train every model on every horizon in one job.

The script reads the three splits, scores the naive baselines, trains all base
estimators plus their per-horizon ensemble, and writes the model bundlesm and
metric reports out.

Runs under the SageMaker scikit-learn container in File mode: each input channel
is downloaded to /opt/ml/input/data/<channel>/ as a set of Spark part-*.parquet
files, models go to SM_MODEL_DIR and reports to SM_OUTPUT_DATA_DIR.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import boto3
import pandas as pd
from botocore.exceptions import ClientError

# train.py sits next to src/ and configs/ inside the unpacked training archive
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config
from src.models import baselines, inference, metrics, train_models

s3 = boto3.client("s3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--valid", default=os.environ.get("SM_CHANNEL_VALID"))
    parser.add_argument("--test", default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "models"))
    parser.add_argument(
        "--output-data-dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", "reports")
    )
    parser.add_argument("--artifacts-bucket", default=os.environ.get("ARTIFACTS_BUCKET"))

    # s3://<artifacts bucket>/metrics/<run id>/ - where the SelectModel step
    # reads model_metrics.csv from. Optional so the script stays runnable locally.
    parser.add_argument("--metrics-s3-uri", default=None)
    # s3://<artifacts bucket>/historical_models/ - the best models retained by earlier
    # runs. Re-scored on this run's splits so the comparison is apples to apples.
    parser.add_argument("--historical-models-uri", default=None)
    parser.add_argument("--run-id", default="local")

    return parser.parse_args()


def load_split(channel_path: str) -> pd.DataFrame:
    """Read every Spark part file of one channel into a single frame."""
    if not channel_path:
        raise ValueError("Split channel is not set; pass --train/--valid/--test.")

    files = sorted(Path(channel_path).rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {channel_path}")

    frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"{channel_path}: {len(files)} part file(s), {len(frame)} rows")
    return frame


def run_stage_baseline(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """
    Describe each target and score the naive baselines on valid and test.

    Also returns the per-horizon MASE denominators: the historical re-scoring
    stage needs the same ones, so retained models and fresh models report MASE
    on an identical scale.
    """
    baseline_rows = []
    profile_rows = []
    mase_scales: dict[str, float] = {}

    for horizon_name in config.HORIZONS_MINUTES:
        target = f"y_{horizon_name}"
        if target not in train_df.columns:
            raise KeyError(
                f"{target} is missing from the train split. The Glue job was given "
                f"horizons {sorted(config.HORIZONS_MINUTES)}; check that "
                "terraform's var.horizons_minutes still matches src/config.py."
            )

        # One denominator per horizon, computed in-sample on train, shared by the
        # baselines and the models so their MASE values are comparable.
        mase_scale = metrics.naive_scale(train_df, target)
        mase_scales[horizon_name] = mase_scale

        baseline_rows.append(
            baselines.evaluate_baselines_by_split(
                splits={"valid": valid_df, "test": test_df},
                target=target,
                horizon_name=horizon_name,
                mase_scale=mase_scale,
            )
        )

        for split_name, split_df in (
            ("train", train_df),
            ("valid", valid_df),
            ("test", test_df),
        ):
            profile_rows.append(
                {
                    "horizon": horizon_name,
                    "split": split_name,
                    **metrics.target_profile(split_df, target, mase_scale),
                }
            )

    return (
        pd.concat(baseline_rows, ignore_index=True),
        pd.DataFrame(profile_rows),
        mase_scales,
    )


def run_stage_compare(
    model_metrics: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    target_profiles: pd.DataFrame,
) -> pd.DataFrame:
    """Stack model and baseline scores, annotated with each target's zero share."""
    comparison = metrics.build_comparison_table(model_metrics, baseline_metrics)

    return comparison.merge(
        target_profiles[["horizon", "split", "zero_share"]],
        on=["horizon", "split"],
        how="left",
    )


def run_stage_historical(
    historical_models_uri: str,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mase_scales: dict[str, float],
) -> pd.DataFrame:
    """
    Re-score the models retained by previous runs on this run's splits.

    Comparing a stored valid_wape against a freshly computed one is meaningless:
    every monthly run re-derives its split boundaries from max(timestamp), so the
    two numbers describe different validation windows. Loading the retained bundle
    and scoring it here is what makes the comparison fair.
    """
    empty = pd.DataFrame()

    bucket, prefix = _split_s3_uri(historical_models_uri)
    registry_key = f"{prefix.rstrip('/')}/best_models.json".lstrip("/")

    try:
        body = s3.get_object(Bucket=bucket, Key=registry_key)["Body"].read().decode("utf-8")
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
            print(f"No {registry_key} yet - first run, nothing to compare against.")
            return empty
        raise

    registry = json.loads(body).get("horizons", {})
    if not registry:
        print(f"{registry_key} lists no horizons, nothing to re-score.")
        return empty

    rows = []
    with tempfile.TemporaryDirectory() as work_dir:
        for horizon, entry in sorted(registry.items()):
            artifact_uri = entry.get("artifact_s3_uri")
            if not artifact_uri:
                print(f"  skipping {horizon}: no artifact_s3_uri in the registry")
                continue

            dest = Path(work_dir) / f"{horizon}.joblib"
            try:
                art_bucket, art_key = _split_s3_uri(artifact_uri)
                s3.download_file(art_bucket, art_key, str(dest))
                bundle = inference.load_bundle(dest)
            except Exception as exc:
                # Deliberately broad. A bundle written by an older container can fail
                # to unpickle (pickle raises UnpicklingError, joblib can raise
                # ModuleNotFoundError), and losing one candidate is far cheaper than
                # throwing away a multi-hour training run at its last step.
                print(f"  skipping {horizon}: could not load {artifact_uri} ({exc})")
                continue

            row = inference.rescore_bundle(bundle, valid_df, test_df, mase_scales.get(horizon))
            if row is None:
                continue

            row["source"] = "historical"
            row["artifact_s3_uri"] = artifact_uri
            rows.append(row)
            print(f"  re-scored {horizon}: retained {row['model']} valid_wape={row['valid_wape']}")

    if not rows:
        return empty

    return pd.DataFrame(rows)


def _split_s3_uri(uri: str) -> tuple[str, str]:
    """
    Split s3://bucket/some/key into (bucket, "some/key").

    The key is returned exactly as written - no trailing slash is added - because
    this serves both prefixes and object keys, and "x.joblib/" is a different
    (nonexistent) object from "x.joblib".
    """
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {uri}")

    bucket, _, key = uri[len("s3://") :].partition("/")
    if not bucket:
        raise ValueError(f"S3 URI has no bucket: {uri}")

    return bucket, key.lstrip("/")


def upload_models(model_dir: Path, artifacts_bucket: str, run_id: str) -> None:
    """Upload individual .joblib files to the current_models prefix for the selection Lambda."""
    for path in sorted(model_dir.glob("*.joblib")):
        key = f"current_models/{run_id}/{path.name}"
        s3.upload_file(str(path), artifacts_bucket, key)
        print(f"Uploaded s3://{artifacts_bucket}/{key}")


def upload_reports(report_dir: Path, metrics_s3_uri: str) -> None:
    """Put the metric CSVs at a stable S3 prefix for the selection of the best model."""
    if not metrics_s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {metrics_s3_uri}")

    parts = metrics_s3_uri.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    prefix = prefix.strip("/") + "/" if prefix else ""

    for path in sorted(report_dir.glob("*.csv")):
        key = f"{prefix}{path.name}"
        s3.upload_file(str(path), bucket, key)
        print(f"Uploaded s3://{bucket}/{key}")


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    report_dir = Path(args.output_data_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    print("=== Training configuration ===")
    print(f"Run id:        {args.run_id}")
    print(f"Horizons:      {config.HORIZONS_MINUTES}")
    print(f"Models config: {config.MODELS_CONFIG}")
    print(f"Model dir:     {model_dir}")
    print(f"Report dir:    {report_dir}")

    train_df = load_split(args.train)
    valid_df = load_split(args.valid)
    test_df = load_split(args.test)

    print("--- Baselines ---")
    baseline_metrics, target_profiles, mase_scales = run_stage_baseline(train_df, valid_df, test_df)
    baseline_metrics.to_csv(report_dir / "baseline_metrics.csv", index=False)
    target_profiles.to_csv(report_dir / "target_profile.csv", index=False)
    print(target_profiles.to_string(index=False))

    print("--- Training ---")
    model_metrics = train_models.train_all_models(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        model_dir=model_dir,
        report_dir=report_dir,
    )

    if args.artifacts_bucket:
        upload_models(model_dir, args.artifacts_bucket, args.run_id)

    # Tell the SelectModel step where each candidate lives and where it came from,
    # so it never has to reconstruct an S3 key from a model name.
    model_metrics["source"] = "current"
    model_metrics["artifact_s3_uri"] = [
        (
            f"s3://{args.artifacts_bucket}/current_models/{args.run_id}/"
            f"{row.model}_{row.horizon}.joblib"
        )
        if args.artifacts_bucket
        else f"{model_dir}/{row.model}_{row.horizon}.joblib"
        for row in model_metrics.itertuples()
    ]
    print(model_metrics.to_string(index=False))

    print("--- Retained models ---")
    if args.historical_models_uri:
        historical_metrics = run_stage_historical(
            args.historical_models_uri, valid_df, test_df, mase_scales
        )
    else:
        historical_metrics = pd.DataFrame()
        print("No --historical-models-uri given, skipping the re-scoring of retained models.")

    # Overwrites the current-run-only copy train_all_models wrote: this is the file
    # the selection step reads, so it must carry both candidate sets.
    selection_metrics = pd.concat([model_metrics, historical_metrics], ignore_index=True)
    selection_metrics.to_csv(report_dir / "model_metrics.csv", index=False)
    if not historical_metrics.empty:
        print(historical_metrics.to_string(index=False))

    print("--- Comparison ---")
    comparison = run_stage_compare(model_metrics, baseline_metrics, target_profiles)
    comparison.to_csv(report_dir / "model_comparison.csv", index=False)
    print(comparison.to_string(index=False))

    if args.metrics_s3_uri:
        upload_reports(report_dir, args.metrics_s3_uri)
    else:
        print("No --metrics-s3-uri given, skipping the S3 report upload.")

    print(
        f"=== Training complete: {len(model_metrics)} new + "
        f"{len(historical_metrics)} retained candidates ==="
    )


if __name__ == "__main__":
    main()
