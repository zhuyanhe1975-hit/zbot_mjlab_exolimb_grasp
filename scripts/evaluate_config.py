"""比较基准构型和中间模块旋转 90° 的工作空间与平移可操作度。

对两种构型使用同一批关节角（默认 20000 组，均匀取自 ±6.28 rad）。
工作空间用末端原点占据的体素个数作体积代理；可操作度是平移雅可比
（只要线速度三行）的 Yoshikawa 度量 sqrt(det(J J^T)) 的中位数。

用法（在仓库根目录，需要 NumPy）::

    python scripts/evaluate_config.py

结果打印成表，并写入 web/public/config_metrics.json。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401

import numpy as np

from zbot_exolimb_grasp.config_gen import (
    BASELINE,
    B_TO_A_QUAT,
    CONFIGS,
    MIDDLE,
    attachment_quaternions,
    forward_kinematics,
    module_point,
    translational_manipulability,
)
from zbot_exolimb_grasp.constants import NOMINAL_JOINT_POS, STRAIGHT_JOINT_POS

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "web" / "public" / "config_metrics.json"

SAMPLE_COUNT = 20_000
SEED = 0
VOXEL_SIZE_M = 0.05
JOINT_LIMIT = 6.28

# 与 web/check_kinematics.mjs 中 MuJoCo 核对过的 foot_1 原点一致。
STRAIGHT_EE = np.array([2.12137298e-4, 0.0, 1.00478799])
NOMINAL_EE = np.array([2.66836957e-4, 0.124544078, 0.455248373])


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"构型评估失败：{message}")


def check_baseline() -> None:
    """全 0 朝向必须复现现有模型，变体必须让远端模块真正转开。"""
    identity = attachment_quaternions(BASELINE)
    _check(np.allclose(identity, B_TO_A_QUAT, atol=0.0), "基准连接四元数偏离了 XML")
    twisted = attachment_quaternions(MIDDLE)
    _check(np.allclose(twisted[:2], B_TO_A_QUAT, atol=0.0), "变体改动了中间模块之前的连接")
    _check(not np.allclose(twisted[2], B_TO_A_QUAT, atol=1e-8), "中间连接没有旋转")

    straight = np.asarray(STRAIGHT_JOINT_POS, dtype=np.float64)
    nominal = np.asarray(NOMINAL_JOINT_POS, dtype=np.float64)
    straight_ee, straight_jac = forward_kinematics(straight, BASELINE)
    nominal_ee, _ = forward_kinematics(nominal, BASELINE)
    _check(np.allclose(straight_ee, STRAIGHT_EE, atol=1e-8), f"伸直末端 {straight_ee} 与模型不符")
    _check(np.allclose(nominal_ee, NOMINAL_EE, atol=1e-8), f"名义末端 {nominal_ee} 与模型不符")
    _check(np.linalg.norm(straight_jac[:, 5]) < 1e-12, "关节 6 不应移动末端原点")

    variant_ee, _ = forward_kinematics(straight, MIDDLE)
    _check(np.linalg.norm(variant_ee - straight_ee) > 5e-4, "变体伸直末端与基准重合")
    local = np.array([-0.07, 0.0, 0.03])
    base_tip = module_point(straight, BASELINE, 5, local)
    variant_tip = module_point(straight, MIDDLE, 5, local)
    proximal = module_point(straight, BASELINE, 2, local)
    proximal_variant = module_point(straight, MIDDLE, 2, local)
    _check(np.allclose(proximal, proximal_variant, atol=1e-12), "中间模块之前的连杆被带动了")
    shift = float(np.linalg.norm(variant_tip - base_tip))
    _check(shift > 0.03, f"远端模块只移动了 {shift:.4f} m，旋转看不出来")
    base_angle = float(np.arctan2(base_tip[1], base_tip[0]))
    variant_angle = float(np.arctan2(variant_tip[1], variant_tip[0]))
    turn = abs(variant_angle - base_angle)
    turn = min(turn, 2.0 * np.pi - turn)
    _check(abs(turn - np.pi / 2) < np.deg2rad(20), f"远端连接不是大约 90°，转过 {np.rad2deg(turn):.1f}°")

    rng = np.random.default_rng(1)
    sample = rng.uniform(-JOINT_LIMIT, JOINT_LIMIT, size=6)
    for orientations in (BASELINE, MIDDLE):
        ee, jac = forward_kinematics(sample, orientations)
        numeric = np.zeros((3, 6))
        step = 1e-6
        for index in range(6):
            bumped = sample.copy()
            bumped[index] += step
            ee_bumped, _ = forward_kinematics(bumped, orientations)
            numeric[:, index] = (ee_bumped - ee) / step
        error = float(np.max(np.abs(jac - numeric)))
        _check(error < 1e-5, f"平移雅可比与差分不一致，最大误差 {error}")


def occupied_voxels(points: np.ndarray, voxel_size: float) -> int:
    indices = np.floor(points / voxel_size).astype(np.int64)
    return int(np.unique(indices, axis=0).shape[0])


def evaluate() -> dict[str, object]:
    check_baseline()
    rng = np.random.default_rng(SEED)
    samples = rng.uniform(-JOINT_LIMIT, JOINT_LIMIT, size=(SAMPLE_COUNT, 6))
    configs = []
    for config_id, spec in CONFIGS.items():
        orientations = spec["orientations"]
        assert isinstance(orientations, tuple)
        positions, jacobian = forward_kinematics(samples, orientations)
        manipulability = translational_manipulability(jacobian)
        voxels = occupied_voxels(positions, VOXEL_SIZE_M)
        configs.append(
            {
                "id": config_id,
                "label": spec["label"],
                "orientations_rad": [float(value) for value in orientations],
                "occupied_voxels": voxels,
                "workspace_volume_m3": round(voxels * VOXEL_SIZE_M**3, 6),
                "median_translational_manipulability": round(float(np.median(manipulability)), 8),
            }
        )
    return {
        "sample_count": SAMPLE_COUNT,
        "seed": SEED,
        "voxel_size_m": VOXEL_SIZE_M,
        "joint_range_rad": [-JOINT_LIMIT, JOINT_LIMIT],
        "jacobian": "translational_rows",
        "manipulability": "yoshikawa_sqrt_det_jjt",
        "description": "六个模块、五个连接朝向。基准全为 0。变体把第 4 个模块（XML 中的 base）相对上一模块绕连接轴转动 π/2。",
        "configs": configs,
    }


def print_table(payload: dict[str, object]) -> None:
    configs = payload["configs"]
    assert isinstance(configs, list)
    print(
        f"采样 {payload['sample_count']} 组，种子 {payload['seed']}，"
        f"体素边长 {payload['voxel_size_m']} m，关节 ±{JOINT_LIMIT} rad"
    )
    print("构型 | 体素占用 | 体积估计 m³ | 平移可操作度中位数")
    for config in configs:
        print(
            f"{config['label']} | {config['occupied_voxels']} | "
            f"{config['workspace_volume_m3']:.5f} | "
            f"{config['median_translational_manipulability']:.6f}"
        )
    baseline, variant = configs
    print(
        "差值（变体 − 基准）| "
        f"体素 {variant['occupied_voxels'] - baseline['occupied_voxels']:+d} | "
        f"体积 {variant['workspace_volume_m3'] - baseline['workspace_volume_m3']:+.5f} | "
        f"可操作度 {variant['median_translational_manipulability'] - baseline['median_translational_manipulability']:+.6f}"
    )


def main() -> None:
    payload = evaluate()
    print_table(payload)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已写入 {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
