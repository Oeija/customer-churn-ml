"""Configuration loader with path validation."""

import os
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load project configuration from YAML.

    Args:
        config_path: Relative path from project root to the config file.

    Returns:
        Nested dictionary with configuration values.

    Raises:
        FileNotFoundError: If the config file or a referenced data path does not exist.
    """
    full_path = PROJECT_ROOT / config_path
    if not full_path.exists():
        raise FileNotFoundError(f"Config file not found: {full_path}")

    with open(full_path, "r") as f:
        config = yaml.safe_load(f)

    # Resolve relative paths to absolute paths
    paths = config.get("paths", {})
    for key, rel_path in paths.items():
        abs_path = PROJECT_ROOT / rel_path
        # Only validate existence for paths that are expected to exist already
        if key in ("raw_data",) and not abs_path.exists():
            raise FileNotFoundError(f"Required path '{key}' does not exist: {abs_path}")
        config["paths"][key] = str(abs_path)

    # Resolve MLflow URI to absolute (default to project-local mlruns)
    mlflow_cfg = config.setdefault("mlflow", {})
    mlruns = mlflow_cfg.get("mlruns_uri", "")
    if not mlruns or mlruns.startswith("file://./"):
        mlflow_cfg["mlruns_uri"] = f"file://{PROJECT_ROOT}/mlruns"

    return config
