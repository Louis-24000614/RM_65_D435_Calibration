#!/usr/bin/env python3
"""Collect ChArUco eye-in-hand calibration samples.

Usage:
  ./run.sh 02_collect_handeye.py --config config/calibration.yaml
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "calibration_py"))

from charuco_board import (
    aruco_dictionary,
    decode_ros_image,
    detect_one,
    draw_detection,
    make_board,
)
from config_loader import load_config
from io_utils import save_json
from ros_utils import BaseCollectorNode
from transforms import matrix_from_transform, quat_to_matrix


class PreviewState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.jpeg: Optional[bytes] = None
        self.status: dict = {}

    def update(self, image_bgr: np.ndarray, status: dict) -> None:
        ok, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return
        with self.lock:
            self.jpeg = encoded.tobytes()
            self.status = dict(status)

    def snapshot(self) -> tuple[Optional[bytes], dict]:
        with self.lock:
            return self.jpeg, dict(self.status)


def make_preview_handler(state: PreviewState):
    class PreviewHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                html = (
                    "<!doctype html><html><head><meta charset='utf-8'>"
                    "<title>ChArUco Preview</title>"
                    "<style>body{margin:0;background:#111;color:#eee;font-family:sans-serif;}"
                    "#bar{padding:8px 12px;background:#222;font-size:14px;}"
                    "img{display:block;max-width:100vw;max-height:calc(100vh - 36px);margin:auto;}</style>"
                    "</head><body><div id='bar'>ChArUco live preview - refresh if the stream stalls</div>"
                    "<img src='/stream.mjpg'></body></html>"
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
                return

            if self.path == "/status.json":
                _, status = state.snapshot()
                body = json.dumps(status, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path != "/stream.mjpg":
                self.send_error(404)
                return

            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            while True:
                frame, _ = state.snapshot()
                if frame is None:
                    time.sleep(0.1)
                    continue
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(frame)))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    break
                time.sleep(0.05)

    return PreviewHandler


def start_preview_server(host: str, port: int, state: PreviewState):
    server = ThreadingHTTPServer((host, port), make_preview_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def query_realsense_color_intrinsics(width: int, height: int, fps: int) -> tuple[np.ndarray, np.ndarray, dict]:
    import pyrealsense2 as rs

    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        raise RuntimeError("No RealSense device found")

    for dev in devices:
        for sensor in dev.query_sensors():
            for profile in sensor.get_stream_profiles():
                stream = profile.stream_type()
                if stream != rs.stream.color:
                    continue
                video = profile.as_video_stream_profile()
                if video.width() != width or video.height() != height or profile.fps() != fps:
                    continue
                intr = video.get_intrinsics()
                K = np.array(
                    [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
                    dtype=float,
                )
                dist = np.array(intr.coeffs, dtype=float).reshape(-1, 1)
                meta = {
                    "source": "pyrealsense2_profile",
                    "width": int(intr.width),
                    "height": int(intr.height),
                    "fx": float(intr.fx),
                    "fy": float(intr.fy),
                    "cx": float(intr.ppx),
                    "cy": float(intr.ppy),
                    "model": str(intr.model),
                    "coeffs": [float(v) for v in intr.coeffs],
                }
                return K, dist, meta
    raise RuntimeError(f"No matching RealSense color profile {width}x{height}@{fps}")


def intrinsics_from_args(args) -> tuple[np.ndarray, np.ndarray, dict]:
    if args.intrinsics_json:
        obj = json.loads(Path(args.intrinsics_json).read_text(encoding="utf-8"))
        K = np.array(obj["camera_matrix"], dtype=float).reshape(3, 3)
        dist = np.array(obj.get("dist_coeffs", [0, 0, 0, 0, 0]), dtype=float).reshape(-1, 1)
        return K, dist, {"source": str(args.intrinsics_json), **obj}

    if all(v is not None for v in (args.fx, args.fy, args.cx, args.cy)):
        K = np.array([[args.fx, 0.0, args.cx], [0.0, args.fy, args.cy], [0.0, 0.0, 1.0]], dtype=float)
        dist = np.zeros((5, 1), dtype=float)
        return K, dist, {
            "source": "cli",
            "width": int(args.width),
            "height": int(args.height),
            "fx": float(args.fx),
            "fy": float(args.fy),
            "cx": float(args.cx),
            "cy": float(args.cy),
            "coeffs": [0.0] * 5,
        }

    K, dist, meta = query_realsense_color_intrinsics(args.width, args.height, args.fps)
    return K, dist, meta


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
        "rvec": [float(v) for v in rvec.reshape(3)],
        "tvec": [float(v) for v in tvec.reshape(3)],
    }
    return T, result, stats


def preview_loop(
    node: BaseCollectorNode,
    args,
    K: np.ndarray,
    dist: np.ndarray,
    preview_state: PreviewState,
    stop_event: threading.Event,
) -> None:
    last_print = 0.0
    while not stop_event.is_set():
        image = node.latest_image
        if image is None:
            stop_event.wait(args.preview_period)
            continue

        image = image.copy()
        _, detection, stats = estimate_charuco_pose(image, args, K, dist)
        vis = draw_detection(image, detection)
        h, w = image.shape[:2]
        expected_w = int(args.width)
        expected_h = int(args.height)
        if (w, h) != (expected_w, expected_h):
            warn_text = f"IMAGE {w}x{h} != INTRINSICS {expected_w}x{expected_h}"
            cv2.putText(vis, warn_text, (18, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.preview), vis)

        status = {
            "updated_at": time.time(),
            "image": str(args.preview),
            "image_width": int(w),
            "image_height": int(h),
            "intrinsics_width": expected_w,
            "intrinsics_height": expected_h,
            "dictionary": detection.dictionary_name,
            "board": f"{detection.squares_x}x{detection.squares_y}",
            "markers": detection.marker_count,
            "charuco_corners": detection.charuco_count,
            "accepted_now": detection.charuco_count >= args.min_corners,
            "reprojection_mean_px": None if stats is None else stats["reprojection_mean_px"],
        }
        preview_state.update(vis, status)
        save_json(args.preview.with_suffix(".json"), status)

        now = time.time()
        if now - last_print >= args.preview_print_period:
            reproj = "n/a" if stats is None else f"{stats['reprojection_mean_px']:.3f}px"
            print(
                f"\n[preview] http://{args.preview_host}:{args.preview_port} "
                f"image={w}x{h} markers={detection.marker_count} "
                f"charuco={detection.charuco_count} reproj={reproj}"
            )
            last_print = now
        stop_event.wait(args.preview_period)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    K, dist, intr_meta = intrinsics_from_args(args)
    session = {
        "created_at": time.time(),
        "image_topic": args.ros_topic,
        "base_frame": args.base_frame,
        "gripper_frame": args.gripper_frame,
        "target_frame": "charuco_board",
        "board": {
            "dictionary": args.dictionary,
            "squares_x": args.squares_x,
            "squares_y": args.squares_y,
            "square_mm": args.square_mm,
            "marker_mm": args.marker_mm,
            "legacy_pattern": bool(args.legacy_pattern),
        },
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.reshape(-1).tolist(),
        "intrinsics": intr_meta,
        "samples": [],
    }
    save_json(args.out_dir / "session.json", session)

    rclpy.init()
    node = BaseCollectorNode(args.ros_topic, args.base_frame, args.gripper_frame, node_name="charuco_handeye_collect")
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    stop_preview = threading.Event()
    preview_thread = None
    preview_state = PreviewState()
    preview_server = None
    preview_server_thread = None
    if args.preview_port > 0:
        preview_server, preview_server_thread = start_preview_server(args.preview_host, args.preview_port, preview_state)
    if args.preview:
        preview_thread = threading.Thread(
            target=preview_loop,
            args=(node, args, K, dist, preview_state, stop_preview),
            daemon=True,
        )
        preview_thread.start()

    print("[INFO] Move the arm to varied poses. Press Enter to capture, or q + Enter to finish.")
    print(f"[INFO] Saving samples under: {args.out_dir}")
    if args.preview:
        print(f"[INFO] Live annotated preview is written to: {args.preview}")
    if args.preview_port > 0:
        print(f"[INFO] Browser live preview: http://{args.preview_host}:{args.preview_port}")
    sample_idx = 0
    try:
        while True:
            cmd = input("[capture] Enter=save, q=quit > ").strip().lower()
            if cmd in {"q", "quit", "exit"}:
                break
            if node.latest_image is None:
                print("[WARN] No image received yet.")
                continue
            try:
                T_base_gripper = node.lookup_base_to_gripper()
            except Exception as exc:
                print(f"[WARN] Could not read TF: {exc}")
                continue

            image = node.latest_image.copy()
            T_target_camera, detection, stats = estimate_charuco_pose(image, args, K, dist)
            if T_target_camera is None:
                print(
                    f"[WARN] Rejected: markers={detection.marker_count}, "
                    f"charuco={detection.charuco_count}, need >= {args.min_corners}"
                )
                continue

            sample_idx += 1
            stem = f"sample_{sample_idx:03d}"
            image_path = args.out_dir / f"{stem}.png"
            json_path = args.out_dir / f"{stem}.json"
            cv2.imwrite(str(image_path), image)
            sample = {
                "index": sample_idx,
                "image": image_path.name,
                "stamp": node.latest_stamp,
                "base_frame": args.base_frame,
                "gripper_frame": args.gripper_frame,
                "T_base_gripper": T_base_gripper.tolist(),
                "T_target_camera": T_target_camera.tolist(),
                "dictionary": detection.dictionary_name,
                "charuco_corners": detection.charuco_count,
                "markers": detection.marker_count,
                **stats,
            }
            save_json(json_path, sample)
            session["samples"].append(json_path.name)
            save_json(args.out_dir / "session.json", session)
            print(
                f"[OK] {stem}: markers={detection.marker_count}, "
                f"charuco={detection.charuco_count}, reproj={stats['reprojection_mean_px']:.3f}px"
            )
    finally:
        stop_preview.set()
        if preview_server is not None:
            preview_server.shutdown()
            preview_server.server_close()
        if preview_server_thread is not None:
            preview_server_thread.join(timeout=1.0)
        if preview_thread is not None:
            preview_thread.join(timeout=1.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect ChArUco samples for eye-in-hand calibration.")
    parser.add_argument("--config", default=None, help="Path to calibration.yaml")
    parser.add_argument("--ros-topic", default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--base-frame", default=None)
    parser.add_argument("--gripper-frame", default=None)
    parser.add_argument("--dictionary", default=None)
    parser.add_argument("--squares-x", type=int, default=None)
    parser.add_argument("--squares-y", type=int, default=None)
    parser.add_argument("--square-mm", type=float, default=None)
    parser.add_argument("--marker-mm", type=float, default=None)
    parser.add_argument("--legacy-pattern", action="store_true")
    parser.add_argument("--min-corners", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--intrinsics-json", default="")
    parser.add_argument("--fx", type=float)
    parser.add_argument("--fy", type=float)
    parser.add_argument("--cx", type=float)
    parser.add_argument("--cy", type=float)
    parser.add_argument("--preview", type=Path, default=None)
    parser.add_argument("--preview-period", type=float, default=0.5)
    parser.add_argument("--preview-print-period", type=float, default=2.0)
    parser.add_argument("--preview-host", default="127.0.0.1")
    parser.add_argument("--preview-port", type=int, default=None)
    cli = parser.parse_args()

    cfg = load_config(cli.config) if cli.config else load_config()
    ch = cfg.get("charuco", {})
    ros = cfg.get("ros", {})
    cam = cfg.get("camera", {})
    prv = cfg.get("preview", {})
    paths = cfg.get("paths", {})

    # Use CLI override if given, else config, else hardcoded defaults
    out_dir = cli.out_dir
    if out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(paths.get("samples_dir", "./data/samples")) / ts

    args = argparse.Namespace()
    args.ros_topic = cli.ros_topic if cli.ros_topic is not None else ros.get("image_topic", "/vision/raw")
    args.out_dir = out_dir
    args.base_frame = cli.base_frame if cli.base_frame is not None else ros.get("base_frame", "base_link")
    args.gripper_frame = cli.gripper_frame if cli.gripper_frame is not None else ros.get("gripper_frame", "Link6")
    args.dictionary = cli.dictionary if cli.dictionary is not None else ch.get("dictionary", "DICT_5X5_100")
    args.squares_x = cli.squares_x if cli.squares_x is not None else ch.get("squares_x", 12)
    args.squares_y = cli.squares_y if cli.squares_y is not None else ch.get("squares_y", 9)
    args.square_mm = cli.square_mm if cli.square_mm is not None else ch.get("square_mm", 30.0)
    args.marker_mm = cli.marker_mm if cli.marker_mm is not None else ch.get("marker_mm", 22.5)
    args.legacy_pattern = bool(cli.legacy_pattern) or ch.get("legacy_pattern", False)
    args.min_corners = cli.min_corners if cli.min_corners is not None else ch.get("min_corners", 20)
    args.width = cli.width if cli.width is not None else cam.get("width", 1280)
    args.height = cli.height if cli.height is not None else cam.get("height", 720)
    args.fps = cli.fps if cli.fps is not None else cam.get("fps", 30)
    args.intrinsics_json = cli.intrinsics_json or cam.get("intrinsics_json", "")
    args.fx = cli.fx if cli.fx is not None else cam.get("fx")
    args.fy = cli.fy if cli.fy is not None else cam.get("fy")
    args.cx = cli.cx if cli.cx is not None else cam.get("cx")
    args.cy = cli.cy if cli.cy is not None else cam.get("cy")

    preview_path = cli.preview
    if preview_path is None:
        preview_path = Path(prv.get("image_path", "./data/samples/preview.png"))
    args.preview = preview_path
    args.preview_period = cli.preview_period if cli.preview_period is not None else prv.get("period", 0.5)
    args.preview_print_period = cli.preview_print_period if cli.preview_print_period is not None else prv.get("print_period", 2.0)
    args.preview_host = cli.preview_host if cli.preview_host is not None else prv.get("host", "127.0.0.1")
    preview_port = cli.preview_port
    if preview_port is None:
        preview_port = prv.get("port", 8088)
    args.preview_port = preview_port

    aruco_dictionary(args.dictionary)
    return args


if __name__ == "__main__":
    raise SystemExit(main())
