"""IO utilities for calibration data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_intrinsics_from_json(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    obj = load_json(path)
    K = np.array(obj["camera_matrix"], dtype=float).reshape(3, 3)
    dist = np.array(obj.get("dist_coeffs", [0, 0, 0, 0, 0]), dtype=float).reshape(-1, 1)
    return K, dist, obj
