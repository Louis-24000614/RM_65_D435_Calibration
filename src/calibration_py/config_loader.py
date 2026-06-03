"""Load calibration configuration from YAML."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "calibration.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        path = DEFAULT_CONFIG
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg
