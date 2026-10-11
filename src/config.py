import os
from pathlib import Path
from typing import Final

BASE_DIR: Final[Path] = Path(__file__).resolve().parents[1]

# Sagemaker unpacks the training archive at /opt/ml/code, so BASE_DIR resolves to
# the archive root.
MODELS_CONFIG: Final[Path] = Path(
    os.environ.get("MODELS_CONFIG", BASE_DIR / "configs" / "models.yaml")
)

# Forecast horizons in minutes. These names become the y_<name> target columns and
# the <model>_<horizon>.joblib artifact names, so they must stay in sync with
# terraform's var.horizons_minutes.
HORIZONS_MINUTES: Final[dict[str, int]] = {
    "h1h": 60,
    "h2h": 120,
    "h3h": 180,
    "h24h": 1440,
}

# Base estimators
MODEL_TYPES: Final[list[str]] = ["lightgbm", "xgboost", "catboost"]
