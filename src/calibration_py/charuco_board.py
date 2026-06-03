"""ChArUco board detection and drawing utilities."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

DEFAULT_DICTIONARIES = ("DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000")
_GUI_AVAILABLE: Optional[bool] = None


@dataclass
class DetectionResult:
    dictionary_name: str
    squares_x: int
    squares_y: int
    legacy_pattern: bool
    marker_corners: tuple
    marker_ids: Optional[np.ndarray]
    charuco_corners: Optional[np.ndarray]
    charuco_ids: Optional[np.ndarray]

    @property
    def marker_count(self) -> int:
        return 0 if self.marker_ids is None else int(len(self.marker_ids))

    @property
    def charuco_count(self) -> int:
        return 0 if self.charuco_ids is None else int(len(self.charuco_ids))


def aruco_dictionary(name: str):
    if not hasattr(cv2.aruco, name):
        valid = ", ".join(DEFAULT_DICTIONARIES)
        raise ValueError(f"Unknown dictionary {name!r}. Expected one of: {valid}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def make_board(dictionary_name: str, squares_x: int, squares_y: int, square_m: float, marker_m: float):
    dictionary = aruco_dictionary(dictionary_name)
    board = cv2.aruco.CharucoBoard((int(squares_x), int(squares_y)), float(square_m), float(marker_m), dictionary)
    return board


def detect_one(
    image_bgr: np.ndarray,
    dictionary_name: str,
    squares_x: int,
    squares_y: int,
    square_m: float,
    marker_m: float,
    legacy_pattern: bool,
) -> DetectionResult:
    board = make_board(dictionary_name, squares_x, squares_y, square_m, marker_m)
    if legacy_pattern and hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(True)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    if hasattr(cv2.aruco, "CharucoDetector"):
        charuco_params = cv2.aruco.CharucoParameters()
        detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
    else:
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
            gray, aruco_dictionary(dictionary_name), parameters=detector_params
        )
        if marker_ids is None or len(marker_ids) == 0:
            charuco_corners, charuco_ids = None, None
        else:
            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, gray, board
            )

    marker_corners = tuple(marker_corners) if marker_corners is not None else tuple()
    return DetectionResult(
        dictionary_name,
        int(squares_x),
        int(squares_y),
        bool(legacy_pattern),
        marker_corners,
        marker_ids,
        charuco_corners,
        charuco_ids,
    )


def detect_best(
    image_bgr: np.ndarray,
    dictionary_names: Iterable[str],
    squares_x: int,
    squares_y: int,
    square_m: float,
    marker_m: float,
    legacy_pattern: bool,
) -> DetectionResult:
    layouts = [(squares_x, squares_y)]
    if squares_x != squares_y:
        layouts.append((squares_y, squares_x))
    legacy_values = (legacy_pattern,)
    if legacy_pattern is None:
        legacy_values = (False, True)

    results = []
    for name in dictionary_names:
        for sx, sy in layouts:
            for legacy in legacy_values:
                results.append(detect_one(image_bgr, name, sx, sy, square_m, marker_m, bool(legacy)))
    return max(results, key=lambda r: (r.charuco_count, r.marker_count))


def draw_detection(image_bgr: np.ndarray, result: DetectionResult) -> np.ndarray:
    vis = image_bgr.copy()
    if result.marker_count > 0:
        cv2.aruco.drawDetectedMarkers(vis, result.marker_corners, result.marker_ids)
    if result.charuco_count > 0:
        cv2.aruco.drawDetectedCornersCharuco(vis, result.charuco_corners, result.charuco_ids, (0, 255, 0))

    text = (
        f"{result.dictionary_name}  markers={result.marker_count}  "
        f"charuco={result.charuco_count}  board={result.squares_x}x{result.squares_y}"
        f"{' legacy' if result.legacy_pattern else ''}"
    )
    cv2.rectangle(vis, (8, 8), (min(vis.shape[1] - 8, 650), 44), (0, 0, 0), -1)
    cv2.putText(vis, text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
    return vis


def show_image(window_name: str, image: np.ndarray, delay_ms: int = 1) -> int:
    global _GUI_AVAILABLE
    if _GUI_AVAILABLE is False:
        return -1
    try:
        cv2.imshow(window_name, image)
        key = cv2.waitKey(delay_ms)
        _GUI_AVAILABLE = True
        return key
    except cv2.error as exc:
        _GUI_AVAILABLE = False
        print(f"[WARN] OpenCV GUI is unavailable; continuing with console output only: {exc}", file=sys.stderr)
        return -1


def close_windows() -> None:
    if _GUI_AVAILABLE is False:
        return
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


def decode_ros_image(msg) -> Optional[np.ndarray]:
    try:
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        if msg.encoding.lower() in ("bgr8", "rgb8"):
            arr = arr.reshape(msg.height, msg.width, 3)
            if msg.encoding.lower() == "rgb8":
                return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            return arr.copy()
        if msg.encoding.lower() in ("mono8", "8uc1"):
            return arr.reshape(msg.height, msg.width)
    except Exception as exc:
        print(f"[WARN] Failed to decode ROS image: {exc}", file=sys.stderr)
    return None
