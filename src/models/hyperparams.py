import os
from pathlib import Path
from typing import Any, Final

import yaml

from src import config

# Every library names the compute device differently, so one environment variable
# drives all three. SageMaker training instances are CPU-only unless a GPU instance
# type is chosen, and the default in models.yaml is already cpu.
DEVICE_OVERRIDES: Final[dict[str, dict[str, str]]] = {
    "cpu": {
        "lightgbm:device_type": "cpu",
        "xgboost:device": "cpu",
        "catboost:task_type": "CPU",
    },
    "gpu": {
        "lightgbm:device_type": "gpu",
        "xgboost:device": "cuda",
        "catboost:task_type": "GPU",
    },
}


def apply_device_override(data: dict[str, Any]) -> dict[str, Any]:
    """Force every model onto one device when ML_DEVICE is set."""
    requested = os.getenv("ML_DEVICE", "").strip().lower()

    if not requested:
        return data

    overrides = DEVICE_OVERRIDES.get(requested)
    if overrides is None:
        raise ValueError(f"ML_DEVICE must be one of {sorted(DEVICE_OVERRIDES)}, got {requested!r}")

    for target, value in overrides.items():
        model_name, _, key = target.partition(":")
        params = data["models"].get(model_name)
        if isinstance(params, dict):
            params[key] = value

    return data


def load_hyperparameters(path: Path = config.MODELS_CONFIG) -> dict[str, Any]:
    """Read model hyperparameters from the YAML config."""
    if not path.exists():
        raise FileNotFoundError(f"Hyperparameter config not found: {path}")

    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a mapping, got {type(data).__name__}")

    for key in ("random_state", "early_stopping_rounds", "models"):
        if key not in data:
            raise ValueError(f"{path} is missing required key: {key}")

    models = data["models"]

    if not isinstance(models, dict) or not models:
        raise ValueError(f"{path}: 'models' must be a non-empty mapping")

    for model_name, params in models.items():
        if not isinstance(params, dict):
            raise TypeError(f"{path}: model '{model_name}' must map to its parameters")

    return apply_device_override(data)
