"""平台运动学用的四元数运算。顺序 wxyz，与 config_gen / 网页演示一致。"""

from __future__ import annotations

import numpy as np

Z_AXIS = np.array([0.0, 0.0, 1.0])


def mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return np.stack(
        (
            vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx),
        ),
        axis=-1,
    )


def hinge_z(angles: np.ndarray) -> np.ndarray:
    half = np.asarray(angles, dtype=np.float64) * 0.5
    out = np.zeros(half.shape + (4,), dtype=np.float64)
    out[..., 0] = np.cos(half)
    out[..., 3] = np.sin(half)
    return out


def axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    half = float(angle) * 0.5
    sine = np.sin(half)
    return np.array([np.cos(half), axis[0] * sine, axis[1] * sine, axis[2] * sine], dtype=np.float64)


def quat_matrix(q: np.ndarray) -> np.ndarray:
    """单个四元数 wxyz 转 3×3 旋转矩阵。"""
    w, x, y, z = [float(value) for value in q]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def so3_vee(rotation: np.ndarray) -> np.ndarray:
    return np.array(
        [rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]],
        dtype=np.float64,
    ) * 0.5


def axis_angle_x(angle: float) -> np.ndarray:
    half = float(angle) * 0.5
    return np.array([np.cos(half), np.sin(half), 0.0, 0.0], dtype=np.float64)
