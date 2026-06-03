"""Hand-eye calibration solver and evaluation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from transforms import load_matrix, make_T, matrix_to_quat

METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def rotation_angle(R: np.ndarray) -> float:
    v = (np.trace(R) - 1.0) * 0.5
    return float(np.arccos(np.clip(v, -1.0, 1.0)))


def evaluate(T_gripper_camera: np.ndarray, samples: list[dict]) -> dict:
    base_targets = []
    for s in samples:
        T_base_gripper = load_matrix(s, "T_base_gripper")
        T_camera_target = load_matrix(s, "T_target_camera")
        base_targets.append(T_base_gripper @ T_gripper_camera @ T_camera_target)

    translations = np.array([T[:3, 3] for T in base_targets], dtype=float)
    mean_t = np.mean(translations, axis=0)
    trans_err = np.linalg.norm(translations - mean_t, axis=1)

    R_ref = base_targets[0][:3, :3]
    rot_err = np.array([rotation_angle(R_ref.T @ T[:3, :3]) for T in base_targets], dtype=float)
    return {
        "target_translation_std_m": float(np.std(trans_err)),
        "target_translation_mean_error_m": float(np.mean(trans_err)),
        "target_translation_max_error_m": float(np.max(trans_err)),
        "target_rotation_mean_error_deg": float(np.rad2deg(np.mean(rot_err))),
        "target_rotation_max_error_deg": float(np.rad2deg(np.max(rot_err))),
    }


def solve(samples: list[dict]) -> tuple[np.ndarray, str, dict, list[dict]]:
    """Solve AX=XB from samples.

    Returns:
        best_T: 4x4 gripper-to-camera transform
        best_name: method name
        best_metrics: evaluation metrics dict
        all_results: list of dicts for all successful methods
    """
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []
    for s in samples:
        T_base_gripper = load_matrix(s, "T_base_gripper")
        T_target_camera = load_matrix(s, "T_target_camera")
        R_gripper2base.append(T_base_gripper[:3, :3])
        t_gripper2base.append(T_base_gripper[:3, 3].reshape(3, 1))
        R_target2cam.append(T_target_camera[:3, :3])
        t_target2cam.append(T_target_camera[:3, 3].reshape(3, 1))

    results = []
    for name, method in METHODS.items():
        try:
            R, t = cv2.calibrateHandEye(
                R_gripper2base,
                t_gripper2base,
                R_target2cam,
                t_target2cam,
                method=method,
            )
            T = make_T(R, t)
            metrics = evaluate(T, samples)
            results.append((metrics["target_translation_mean_error_m"], name, T, metrics))
        except cv2.error as exc:
            print(f"[WARN] {name} failed: {exc}")

    if not results:
        raise RuntimeError("No hand-eye method produced a result")

    results.sort(key=lambda x: x[0])
    best_error, best_name, best_T, best_metrics = results[0]

    all_results = [
        {"method": name, "metrics": metrics, "T_link6_camera": T.tolist()}
        for _, name, T, metrics in results
    ]
    return best_T, best_name, best_metrics, all_results


def solve_from_session(session_dir: Path, min_corners: int = 20, max_reprojection: float = 2.0) -> dict:
    import json

    session = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    samples = [
        json.loads((session_dir / name).read_text(encoding="utf-8"))
        for name in session.get("samples", [])
    ]
    samples = [
        s
        for s in samples
        if int(s.get("charuco_corners", 0)) >= min_corners
        and float(s.get("reprojection_mean_px", 999.0)) <= max_reprojection
    ]
    if len(samples) < 5:
        raise RuntimeError(f"Need at least 5 good samples, got {len(samples)}")

    best_T, best_name, best_metrics, all_results = solve(samples)
    return {
        "method": best_name,
        "sample_count": len(samples),
        "T_link6_camera": best_T.tolist(),
        "translation_xyz_m": [float(v) for v in best_T[:3, 3]],
        "quaternion_xyzw": matrix_to_quat(best_T[:3, :3]),
        "metrics": best_metrics,
        "all_methods": all_results,
        "session": str(session_dir),
    }
