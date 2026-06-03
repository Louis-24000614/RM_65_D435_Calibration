#!/usr/bin/env python3
"""Verify eye-in-hand calibration by locating ChArUco board center in base.

Usage:
  ./run.sh 04_verify_handeye.py --config config/calibration.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "calibration_py"))

from charuco_board import aruco_dictionary, decode_ros_image, detect_one, draw_detection, make_board
from config_loader import load_config
from io_utils import load_json
from ros_utils import BaseCollectorNode
from transforms import matrix_from_transform


class VerifyNode(BaseCollectorNode):
    def __init__(self, image_topic: str, base_frame: str, gripper_frame: str):
        super().__init__(image_topic, base_frame, gripper_frame, node_name="verify_charuco_hover")


def load_intrinsics(session_json: Path) -> tuple[np.ndarray, np.ndarray]:
    obj = load_json(session_json)
    K = np.array(obj["camera_matrix"], dtype=float).reshape(3, 3)
    dist = np.array(obj.get("dist_coeffs", [0, 0, 0, 0, 0]), dtype=float).reshape(-1, 1)
    return K, dist


def load_handeye(handeye_json: Path) -> np.ndarray:
    obj = load_json(handeye_json)
    return np.array(obj["T_link6_camera"], dtype=float).reshape(4, 4)


def homog(point_xyz: np.ndarray) -> np.ndarray:
    return np.array([float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2]), 1.0], dtype=float)


def fmt_vec(v: np.ndarray) -> str:
    return f"({v[0]: .6f}, {v[1]: .6f}, {v[2]: .6f})"


def estimate_charuco_pose(image_bgr: np.ndarray, args, K: np.ndarray, dist: np.ndarray):
    board = make_board(args.dictionary, args.squares_x, args.squares_y, args.square_mm / 1000.0, args.marker_mm / 1000.0)
    result = detect_one(
        image_bgr,
        args.dictionary,
        args.squares_x,
        args.squares_y,
        args.square_mm / 1000.0,
        args.marker_mm / 1000.0,
        args.legacy_pattern,
    )
    if result.charuco_count < args.min_corners:
        return None, result, None

    all_obj = board.getChessboardCorners().astype(np.float32)
    ids = result.charuco_ids.reshape(-1).astype(int)
    obj_pts = all_obj[ids].reshape(-1, 1, 3)
    img_pts = result.charuco_corners.astype(np.float32).reshape(-1, 1, 2)
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, result, None

    projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    reproj = np.linalg.norm(projected.reshape(-1, 2) - img_pts.reshape(-1, 2), axis=1)
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    stats = {
        "reprojection_mean_px": float(np.mean(reproj)),
        "reprojection_max_px": float(np.max(reproj)),
    }
    return T, result, stats


def main() -> int:
    args = parse_args()
    K, dist = load_intrinsics(args.session_json)
    T_gripper_camera = load_handeye(args.handeye_json)
    board = make_board(args.dictionary, args.squares_x, args.squares_y, args.square_mm / 1000.0, args.marker_mm / 1000.0)
    right_bottom = np.array(board.getRightBottomCorner(), dtype=float).reshape(3)
    board_center = np.array([right_bottom[0] * 0.5, right_bottom[1] * 0.5, 0.0], dtype=float)

    rclpy.init()
    node = VerifyNode(args.ros_topic, args.base_frame, args.gripper_frame)
    try:
        deadline = time.time() + args.timeout
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.latest_image is None:
                continue
            try:
                T_base_gripper = node.lookup_base_to_gripper()
            except Exception as exc:
                print(f"[WARN] TF not ready: {exc}")
                continue

            image = node.latest_image.copy()
            T_camera_board, detection, stats = estimate_charuco_pose(image, args, K, dist)
            if T_camera_board is None:
                print(f"[WARN] Board not ready: markers={detection.marker_count}, charuco={detection.charuco_count}")
                continue

            T_base_camera = T_base_gripper @ T_gripper_camera
            T_base_board = T_base_camera @ T_camera_board
            board_center_base = (T_base_board @ homog(board_center))[:3]
            camera_base = T_base_camera[:3, 3]
            toward_camera = camera_base - board_center_base
            norm = float(np.linalg.norm(toward_camera))
            if norm < 1e-9:
                raise RuntimeError("Camera and board center are numerically identical")
            hover_base = board_center_base + toward_camera / norm * float(args.hover_m)

            print("\n[OK] ChArUco eye-in-hand verification snapshot")
            print(f"image: {image.shape[1]}x{image.shape[0]}")
            print(f"markers={detection.marker_count}, charuco={detection.charuco_count}")
            print(f"reprojection_mean={stats['reprojection_mean_px']:.3f}px")
            print(f"T_base_{args.gripper_frame} translation: {fmt_vec(T_base_gripper[:3, 3])} m")
            print(f"T_{args.gripper_frame}_camera translation: {fmt_vec(T_gripper_camera[:3, 3])} m")
            print(f"camera position in base: {fmt_vec(camera_base)} m")
            print(f"board center in board frame: {fmt_vec(board_center)} m")
            print(f"board center in base: {fmt_vec(board_center_base)} m")
            print(f"safe hover point ({args.hover_m * 1000:.0f} mm toward camera): {fmt_vec(hover_base)} m")
            print("\nMove Link6 near the safe hover point slowly if you want a physical check.")
            return 0
        print("[ERROR] Timed out waiting for image, TF, and ChArUco detection.")
        return 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Print board center and safe hover point in base using ChArUco hand-eye.")
    p.add_argument("--config", default=None, help="Path to calibration.yaml")
    p.add_argument("--ros-topic", default=None)
    p.add_argument("--base-frame", default=None)
    p.add_argument("--gripper-frame", default=None)
    p.add_argument("--handeye-json", type=Path, default=None)
    p.add_argument("--session-json", type=Path, default=None)
    p.add_argument("--dictionary", default=None)
    p.add_argument("--squares-x", type=int, default=None)
    p.add_argument("--squares-y", type=int, default=None)
    p.add_argument("--square-mm", type=float, default=None)
    p.add_argument("--marker-mm", type=float, default=None)
    p.add_argument("--legacy-pattern", action="store_true")
    p.add_argument("--min-corners", type=int, default=None)
    p.add_argument("--hover-m", type=float, default=0.05)
    p.add_argument("--timeout", type=float, default=10.0)
    cli = p.parse_args()

    cfg = load_config(cli.config) if cli.config else load_config()
    ch = cfg.get("charuco", {})
    ros = cfg.get("ros", {})
    paths = cfg.get("paths", {})

    args = argparse.Namespace()
    args.ros_topic = cli.ros_topic if cli.ros_topic is not None else ros.get("image_topic", "/vision/raw")
    args.base_frame = cli.base_frame if cli.base_frame is not None else ros.get("base_frame", "base_link")
    args.gripper_frame = cli.gripper_frame if cli.gripper_frame is not None else ros.get("gripper_frame", "Link6")

    handeye_json = cli.handeye_json
    if handeye_json is None:
        handeye_json = Path(paths.get("handeye_result", "./data/results/charuco_eye_in_hand.json"))
    args.handeye_json = handeye_json

    session_json = cli.session_json
    if session_json is None:
        # Find the most recent session
        samples_dir = Path(paths.get("samples_dir", "./data/samples"))
        if samples_dir.exists():
            dirs = [d for d in samples_dir.iterdir() if d.is_dir() and (d / "session.json").exists()]
            if dirs:
                session_json = max(dirs, key=lambda d: d.stat().st_mtime) / "session.json"
        if session_json is None:
            session_json = Path("./data/samples/latest/session.json")
    args.session_json = session_json

    args.dictionary = cli.dictionary if cli.dictionary is not None else ch.get("dictionary", "DICT_5X5_100")
    args.squares_x = cli.squares_x if cli.squares_x is not None else ch.get("squares_x", 12)
    args.squares_y = cli.squares_y if cli.squares_y is not None else ch.get("squares_y", 9)
    args.square_mm = cli.square_mm if cli.square_mm is not None else ch.get("square_mm", 30.0)
    args.marker_mm = cli.marker_mm if cli.marker_mm is not None else ch.get("marker_mm", 22.5)
    args.legacy_pattern = bool(cli.legacy_pattern) or ch.get("legacy_pattern", False)
    args.min_corners = cli.min_corners if cli.min_corners is not None else ch.get("min_corners", 20)
    args.hover_m = cli.hover_m
    args.timeout = cli.timeout

    aruco_dictionary(args.dictionary)
    return args


if __name__ == "__main__":
    raise SystemExit(main())
