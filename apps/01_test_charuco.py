#!/usr/bin/env python3
"""Test ChArUco board detection from ROS topic, image file, or USB camera."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "calibration_py"))

from charuco_board import (
    DEFAULT_DICTIONARIES,
    aruco_dictionary,
    close_windows,
    detect_best,
    draw_detection,
    show_image,
)
from config_loader import load_config


def print_result(result, shape) -> None:
    height, width = shape[:2]
    print(
        f"[INFO] image={width}x{height} dictionary={result.dictionary_name} "
        f"board={result.squares_x}x{result.squares_y} "
        f"legacy={int(result.legacy_pattern)} markers={result.marker_count} "
        f"charuco_corners={result.charuco_count}"
    )


def maybe_save(args, vis: np.ndarray) -> None:
    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), vis)
        print(f"[INFO] Saved annotated image: {out}")


def run_on_image(args) -> int:
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        print(f"[ERROR] Could not read image: {args.image}", file=sys.stderr)
        return 2
    result = detect_best(
        image, args.dictionary_names, args.squares_x, args.squares_y, args.square_m, args.marker_m, args.legacy_pattern
    )
    print_result(result, image.shape)
    vis = draw_detection(image, result)
    maybe_save(args, vis)
    if not args.no_display:
        show_image("charuco detection", vis, 0)
    return 0 if result.charuco_count > 0 else 1


def run_on_camera(args) -> int:
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {args.camera}", file=sys.stderr)
        return 2
    if args.camera_width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    if args.camera_height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)

    last_print = 0.0
    last_vis = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Failed to read camera frame", file=sys.stderr)
                continue
            result = detect_best(
                frame, args.dictionary_names, args.squares_x, args.squares_y, args.square_m, args.marker_m, args.legacy_pattern
            )
            last_vis = draw_detection(frame, result)
            now = time.time()
            if now - last_print >= args.print_period:
                print_result(result, frame.shape)
                last_print = now
            if not args.no_display:
                key = show_image("charuco detection", last_vis, 1) & 0xFF
                if key in (27, ord("q")):
                    break
    finally:
        cap.release()
        if last_vis is not None:
            maybe_save(args, last_vis)
        close_windows()
    return 0


def run_on_ros(args) -> int:
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image
    except Exception as exc:
        print(f"[ERROR] ROS2 Python imports failed: {exc}", file=sys.stderr)
        return 2

    from charuco_board import decode_ros_image

    class CharucoTestNode(Node):
        def __init__(self):
            super().__init__("test_charuco_detection")
            self.latest_vis = None
            self.last_print = 0.0
            self.create_subscription(Image, args.ros_topic, self.cb, 10)
            self.get_logger().info(f"Subscribed to {args.ros_topic}")

        def cb(self, msg):
            frame = decode_ros_image(msg)
            if frame is None:
                return
            result = detect_best(
                frame, args.dictionary_names, args.squares_x, args.squares_y, args.square_m, args.marker_m, args.legacy_pattern
            )
            self.latest_vis = draw_detection(frame, result)
            now = time.time()
            if now - self.last_print >= args.print_period:
                print_result(result, frame.shape)
                self.last_print = now
            if not args.no_display:
                show_image("charuco detection", self.latest_vis, 1)

    rclpy.init()
    node = CharucoTestNode()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if not args.no_display:
                key = show_image("charuco detection", node.latest_vis, 1) & 0xFF if node.latest_vis is not None else 0
                if key in (27, ord("q")):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if node.latest_vis is not None:
            maybe_save(args, node.latest_vis)
        node.destroy_node()
        rclpy.shutdown()
        close_windows()
    return 0


def build_args_from_config(cfg: dict, overrides: argparse.Namespace) -> argparse.Namespace:
    ch = cfg.get("charuco", {})
    ros = cfg.get("ros", {})
    cam = cfg.get("camera", {})

    args = argparse.Namespace()
    args.ros_topic = overrides.ros_topic if overrides.ros_topic is not None else ros.get("image_topic", "/vision/raw")
    args.image = overrides.image
    args.camera = overrides.camera
    args.squares_x = overrides.squares_x if overrides.squares_x is not None else ch.get("squares_x", 12)
    args.squares_y = overrides.squares_y if overrides.squares_y is not None else ch.get("squares_y", 9)
    args.square_mm = overrides.square_mm if overrides.square_mm is not None else ch.get("square_mm", 30.0)
    args.marker_mm = overrides.marker_mm if overrides.marker_mm is not None else ch.get("marker_mm", 22.5)
    args.legacy_pattern = overrides.legacy_pattern
    args.no_auto_layout = overrides.no_auto_layout
    args.camera_width = overrides.camera_width if overrides.camera_width is not None else cam.get("width", 1280)
    args.camera_height = overrides.camera_height if overrides.camera_height is not None else cam.get("height", 720)
    args.print_period = overrides.print_period
    args.save = overrides.save
    args.no_display = overrides.no_display
    args.dictionary = overrides.dictionary

    args.square_m = args.square_mm / 1000.0
    args.marker_m = args.marker_mm / 1000.0
    if args.dictionary == "auto":
        args.dictionary_names = DEFAULT_DICTIONARIES
    else:
        args.dictionary_names = (args.dictionary,)
        aruco_dictionary(args.dictionary)
    if args.no_auto_layout and args.legacy_pattern is None:
        args.legacy_pattern = False
    return args


def main() -> int:
    parser = argparse.ArgumentParser(description="Test whether a custom ChArUco board can be detected.")
    parser.add_argument("--config", default=None, help="Path to calibration.yaml")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ros-topic", default=None, help="ROS2 sensor_msgs/Image topic, e.g. /vision/raw")
    source.add_argument("--image", type=Path, help="Path to a saved image")
    source.add_argument("--camera", type=int, help="OpenCV camera index")

    parser.add_argument("--squares-x", type=int, default=None)
    parser.add_argument("--squares-y", type=int, default=None)
    parser.add_argument("--square-mm", type=float, default=None)
    parser.add_argument("--marker-mm", type=float, default=None)
    parser.add_argument("--dictionary", default="auto")
    parser.add_argument("--legacy-pattern", action="store_true", default=None)
    parser.add_argument("--no-auto-layout", action="store_true")
    parser.add_argument("--camera-width", type=int, default=None)
    parser.add_argument("--camera-height", type=int, default=None)
    parser.add_argument("--print-period", type=float, default=0.5)
    parser.add_argument("--save", default="")
    parser.add_argument("--no-display", action="store_true")
    overrides = parser.parse_args()

    cfg = load_config(overrides.config) if overrides.config else {}
    args = build_args_from_config(cfg, overrides)

    if args.image is not None:
        return run_on_image(args)
    if args.camera is not None:
        return run_on_camera(args)
    return run_on_ros(args)


if __name__ == "__main__":
    raise SystemExit(main())
