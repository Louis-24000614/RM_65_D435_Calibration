"""4x4 homogeneous transform utilities."""

from __future__ import annotations

import numpy as np


def quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3, dtype=float)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - yy - zz, xy - wz, xz + wy],
            [xy + wz, 1.0 - xx - zz, yz - wx],
            [xz - wy, yz + wx, 1.0 - xx - yy],
        ],
        dtype=float,
    )


def matrix_from_transform(t) -> np.ndarray:
    """Convert TF transform message to 4x4 matrix."""
    qx = float(t.transform.rotation.x)
    qy = float(t.transform.rotation.y)
    qz = float(t.transform.rotation.z)
    qw = float(t.transform.rotation.w)
    tx = float(t.transform.translation.x)
    ty = float(t.transform.translation.y)
    tz = float(t.transform.translation.z)
    R = quat_to_matrix(qx, qy, qz, qw)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


def invert(T: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=float)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ T[:3, 3]
    return out


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.array(R, dtype=float).reshape(3, 3)
    T[:3, 3] = np.array(t, dtype=float).reshape(3)
    return T


def load_matrix(obj: dict, key: str) -> np.ndarray:
    return np.array(obj[key], dtype=float).reshape(4, 4)


def matrix_to_quat(R: np.ndarray) -> list[float]:
    m = np.array(R, dtype=float).reshape(3, 3)
    tr = float(np.trace(m))
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(m)))
        if idx == 0:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif idx == 1:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=float)
    q /= np.linalg.norm(q)
    return [float(v) for v in q]
