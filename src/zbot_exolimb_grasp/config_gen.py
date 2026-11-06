"""把离散连接朝向变成与 ZBOT6.xml 一致的运动学。

六个模块之间有五个连接。每个连接朝向取 0 或 π/2，绕该连接轴
（b 连杆原点指向下一 a 连杆原点的方向）右旋。全 0 时位置与四元数
和 `assets/zbot_6/ZBOT6.xml` 相同，关节轴仍是各连杆局部的 `0 0 1`。

变体只改中间模块：从肩部数起的第 4 个（XML 里的 `base`）相对上一模块转动 π/2。
公式与 `web/src/kinematics.js` 的 `attachmentQuat` / `footOrigin` 一致。四元数为 wxyz。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

MODULE_COUNT = 6
INTERFACE_COUNT = 5
# 0 起算的连接序号：第 4 个模块的入连接。
MIDDLE_INTERFACE = 2

SHOULDER = np.array([0.0, 0.0, 0.40])
A_TO_B_POS = np.array([-0.00021213, 0.0, 0.054788])
A_TO_B_QUAT = np.array([0.923879181, 0.0, 0.382684281, 0.0])
B_TO_A_POS = np.array([-0.038890846, 0.0, 0.03919071])
B_TO_A_QUAT = np.array([0.0, -0.382684281, 0.0, 0.923879181])
CONNECTION_AXIS = B_TO_A_POS / np.linalg.norm(B_TO_A_POS)

BASELINE: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
MIDDLE: tuple[float, ...] = tuple(
    math.pi / 2 if index == MIDDLE_INTERFACE else 0.0 for index in range(INTERFACE_COUNT)
)

CONFIGS: dict[str, dict[str, object]] = {
    "baseline": {"label": "基准构型", "orientations": BASELINE},
    "middle": {"label": "中间模块旋转 90°", "orientations": MIDDLE},
}


def _orientations(orientations: Sequence[float]) -> np.ndarray:
    phi = np.asarray(tuple(orientations), dtype=np.float64)
    if phi.shape != (INTERFACE_COUNT,):
        raise ValueError(f"连接朝向应为 {INTERFACE_COUNT} 个数，每个接口取 0 或 π/2")
    return phi


def _mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
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


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
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


def attachment_quaternions(orientations: Sequence[float]) -> np.ndarray:
    """每个连接的 b→a 四元数，形状 (5, 4)。朝向为 0 时等于 XML 里的四元数。"""
    phi = _orientations(orientations)
    half = phi * 0.5
    sine = np.sin(half)
    twist = np.stack(
        (
            np.cos(half),
            CONNECTION_AXIS[0] * sine,
            CONNECTION_AXIS[1] * sine,
            CONNECTION_AXIS[2] * sine,
        ),
        axis=-1,
    )
    return _mul(twist, np.broadcast_to(B_TO_A_QUAT, twist.shape))


def _as_batch(q: np.ndarray) -> tuple[np.ndarray, bool]:
    joints = np.asarray(q, dtype=np.float64)
    if joints.ndim == 1:
        if joints.shape != (MODULE_COUNT,):
            raise ValueError("关节角应为 6 个数")
        return joints.reshape(1, MODULE_COUNT), True
    if joints.ndim != 2 or joints.shape[1] != MODULE_COUNT:
        raise ValueError("关节角应为 (N, 6)")
    return joints, False


def _hinge_z(angles: np.ndarray) -> np.ndarray:
    half = angles * 0.5
    out = np.zeros((angles.shape[0], 4), dtype=np.float64)
    out[:, 0] = np.cos(half)
    out[:, 3] = np.sin(half)
    return out


def forward_kinematics(
    q: np.ndarray, orientations: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """末端（foot_1 原点）位置，以及平移雅可比。

    单个姿态返回 (3,) 和 (3, 6)；一批姿态返回 (N, 3) 和 (N, 3, 6)。
    雅可比只有线速度三行：第 i 列是关节轴与「轴上一点到末端」的叉积。
    关节 6 穿过末端原点，对应列为 0。
    """
    joints, single = _as_batch(q)
    count = joints.shape[0]
    attach = attachment_quaternions(orientations)
    pos = np.empty((count, 3), dtype=np.float64)
    pos[:] = SHOULDER
    quat = np.zeros((count, 4), dtype=np.float64)
    quat[:, 0] = 1.0
    origins: list[np.ndarray] = []
    axes: list[np.ndarray] = []
    a_pos = np.broadcast_to(A_TO_B_POS, (count, 3))
    a_quat = np.broadcast_to(A_TO_B_QUAT, (count, 4))
    b_pos = np.broadcast_to(B_TO_A_POS, (count, 3))
    local_axis = np.broadcast_to(np.array([0.0, 0.0, 1.0]), (count, 3))
    for index in range(MODULE_COUNT):
        pos = pos + _rotate(quat, a_pos)
        quat = _mul(quat, a_quat)
        origins.append(pos)
        axes.append(_rotate(quat, local_axis))
        quat = _mul(quat, _hinge_z(joints[:, index]))
        if index == MODULE_COUNT - 1:
            break
        pos = pos + _rotate(quat, b_pos)
        quat = _mul(quat, np.broadcast_to(attach[index], (count, 4)))
    ee = origins[-1]
    jacobian = np.empty((count, 3, MODULE_COUNT), dtype=np.float64)
    for index in range(MODULE_COUNT):
        jacobian[:, :, index] = np.cross(axes[index], ee - origins[index])
    if single:
        return ee[0], jacobian[0]
    return ee, jacobian


def translational_manipulability(jacobian: np.ndarray) -> np.ndarray:
    """Yoshikawa 平移可操作度 sqrt(det(J J^T))，J 为 3×6 线速度雅可比。"""
    singular = np.linalg.svd(jacobian, compute_uv=False)
    return np.prod(singular, axis=-1)


def module_point(
    q: np.ndarray, orientations: Sequence[float], module_index: int, local: np.ndarray
) -> np.ndarray:
    """第 module_index 个 b 连杆（0 起算，关节角已作用）上局部点的世界坐标。"""
    if module_index < 0 or module_index >= MODULE_COUNT:
        raise ValueError("模块序号超出 0..5")
    joints, _ = _as_batch(np.asarray(q, dtype=np.float64))
    attach = attachment_quaternions(orientations)
    pos = SHOULDER.copy()
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    local = np.asarray(local, dtype=np.float64)
    for index in range(MODULE_COUNT):
        pos = pos + _rotate(quat, A_TO_B_POS)
        quat = _mul(quat, A_TO_B_QUAT)
        quat = _mul(quat, _hinge_z(joints[:, index])[0])
        if index == module_index:
            return pos + _rotate(quat, local)
        pos = pos + _rotate(quat, B_TO_A_POS)
        quat = _mul(quat, attach[index])
    raise RuntimeError("未找到模块")
