#!/usr/bin/env python3
"""Solve eye-in-hand calibration from collected ChArUco samples.

Usage:
  python3 scripts/03_solve_handeye.py --config config/calibration.yaml
  python3 scripts/03_solve_handeye.py --session-dir ./data/samples/20260603_120000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "calibration_py"))

from config_loader import load_config
from handeye_solver import solve_from_session
from io_utils import save_json


def main() -> int:
    args = parse_args()
    result = solve_from_session(args.session_dir, min_corners=args.min_corners, max_reprojection=args.max_reprojection)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, result)

    best_T = np.array(result["T_link6_camera"])
    print(f"[OK] Best method: {result['method']}")
    print(f"[OK] Samples used: {result['sample_count']}")
    print(f"[OK] Mean target consistency: {result['metrics']['target_translation_mean_error_m'] * 1000.0:.2f} mm")
    print("[OK] T_link6_camera:")
    print(np.array2string(best_T, precision=6, suppress_small=False))
    print(f"[OK] translation xyz m: {result['translation_xyz_m']}")
    print(f"[OK] quaternion xyzw: {result['quaternion_xyzw']}")
    print(f"[OK] Saved: {args.output}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Solve eye-in-hand calibration from collected ChArUco samples.")
    p.add_argument("--config", default=None, help="Path to calibration.yaml")
    p.add_argument("--session-dir", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--min-corners", type=int, default=None)
    p.add_argument("--max-reprojection", type=float, default=None)
    cli = p.parse_args()

    cfg = load_config(cli.config) if cli.config else load_config()
    paths = cfg.get("paths", {})
    ch = cfg.get("charuco", {})

    session_dir = cli.session_dir
    if session_dir is None:
        # Pick the most recent samples directory
        samples_dir = Path(paths.get("samples_dir", "./data/samples"))
        if samples_dir.exists():
            dirs = [d for d in samples_dir.iterdir() if d.is_dir() and (d / "session.json").exists()]
            if dirs:
                session_dir = max(dirs, key=lambda d: d.stat().st_mtime)
        if session_dir is None:
            session_dir = Path("./data/samples/latest")

    output = cli.output
    if output is None:
        output = Path(paths.get("handeye_result", "./data/results/charuco_eye_in_hand.json"))

    args = argparse.Namespace()
    args.session_dir = session_dir
    args.output = output
    args.min_corners = cli.min_corners if cli.min_corners is not None else ch.get("min_corners", 20)
    args.max_reprojection = cli.max_reprojection if cli.max_reprojection is not None else 2.0
    return args


if __name__ == "__main__":
    raise SystemExit(main())
