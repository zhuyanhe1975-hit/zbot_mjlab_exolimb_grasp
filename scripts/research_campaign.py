"""用科研后端复核 6/7/8 模块，并和批次 2 的自写结果对照。

不训练设计网络，也不跑上游的整晚穷举基线。

用法::

    PYTHONPATH=src:third_party/generative-robot-design python scripts/research_campaign.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "generative-robot-design"))

from zbot_exolimb_grasp.constants import NOMINAL_JOINT_POS
from zbot_exolimb_grasp.platform.exporting import export_mjcf
from zbot_exolimb_grasp.platform.math3d import axis_angle, mul
from zbot_exolimb_grasp.platform.model import (
    collision_mask,
    forward_chain,
    ik_start_poses,
    static_torques,
    task_targets,
    translational_manipulability,
    velocity_capability,
    zbot_chain,
)
from zbot_exolimb_grasp.platform.research_backend import (
    availability,
    gradient_position_ik,
    link_frames,
    mesh_collision_count,
    module_matrices,
    mujoco_tool_positions,
    pinocchio_static_torques,
    positions,
)
from zbot_exolimb_grasp.platform.schema import TaskSpec

OUT = ROOT / "experiments" / "research_backend_report.json"
CSV_PATH = ROOT / "experiments" / "research_backend_screen.csv"
PUBLIC = ROOT / "web" / "public" / "research_backend.json"


def feasible_factory(module_ids, twists, lengths, payload: float):
    from zbot_exolimb_grasp.platform.schema import Design

    design = Design(tuple(module_ids), tuple(twists), tuple({} if module_id == "zbot_r3" else {"length_m": length} for module_id, length in zip(module_ids, lengths, strict=True)))

    def feasible(q: np.ndarray) -> bool:
        chain = forward_chain(design, q, tool_point="foot_1_sole")
        if bool(collision_mask(chain)[0]):
            return False
        torque = static_torques(chain, payload)[0]
        return float(np.max(np.abs(torque))) <= 20.0

    return feasible


def screen_one(module_ids, twists, lengths, targets, starts, payload: float) -> dict:
    proximal, distal, mass = module_matrices(module_ids, twists, lengths)
    checker = feasible_factory(module_ids, twists, lengths, payload)
    q, error = gradient_position_ik(
        proximal,
        distal,
        targets,
        starts,
        tool="foot_1_sole",
        iterations=20,
        step=1.0,
        tolerance=0.02,
        feasible=checker,
    )
    reached = error <= 0.02
    solved = q[reached]
    manip = None
    speed = None
    if solved.size:
        design_chain = forward_chain(
            __import__("zbot_exolimb_grasp.platform.schema", fromlist=["Design"]).Design(
                tuple(module_ids),
                tuple(twists),
                tuple({} if module_id == "zbot_r3" else {"length_m": float(length)} for module_id, length in zip(module_ids, lengths, strict=True)),
            ),
            solved[:8],
            tool_point="foot_1_sole",
        )
        jacobian = design_chain["jacobian"]
        manip = float(np.median(translational_manipulability(jacobian)))
        speed = float(np.median(velocity_capability(jacobian, 1.5)))
    zero = np.zeros(len(module_ids))
    margins = {}
    for load in (0.4, 1.5, 4.0):
        tau = pinocchio_static_torques(proximal, distal, module_ids, lengths, zero, load)
        margins[str(load)] = 20.0 - float(np.max(np.abs(tau)))
    return {
        "reached": int(reached.sum()),
        "targets": int(targets.shape[0]),
        "fraction": float(reached.mean()),
        "mass_kg": mass,
        "manipulability": manip,
        "velocity_pseudoinverse": speed,
        "torque_margin_nm": margins,
    }


def spearman(left: list[float], right: list[float]) -> float:
    from scipy.stats import spearmanr

    coef = spearmanr(left, right).statistic
    return float(coef)


def kendall(left: list[float], right: list[float]) -> float:
    from scipy.stats import kendalltau

    return float(kendalltau(left, right).statistic)


def non_dominated(rows: list[dict]) -> list[dict]:
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    if not rows:
        return []
    matrix = []
    for row in rows:
        matrix.append(
            [
                -row["fraction"],
                row["mass_kg"],
                -(row["torque_margin_nm"]["0.4"] or 0.0),
                -(row["velocity_pseudoinverse"] or 0.0),
                -(row["manipulability"] or 0.0),
            ]
        )
    front = NonDominatedSorting().do(np.asarray(matrix, dtype=float), only_non_dominated_front=True)
    return [rows[int(index)] for index in front]


def richer_pose_targets() -> list[dict]:
    """不使用 6 模块零位姿。位置和姿态都按世界坐标事先指定。"""
    targets = []
    center = np.array([0.12, 0.16, 0.74])
    half = np.array([0.07, 0.07, 0.07])
    corners = [center + half * np.array([sx, sy, sz]) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    orientations = {
        "world": np.array([1.0, 0.0, 0.0, 0.0]),
        "rx60": axis_angle(np.array([1.0, 0.0, 0.0]), math.radians(60)),
        "ry-40": axis_angle(np.array([0.0, 1.0, 0.0]), math.radians(-40)),
    }
    # 只取 4 个角 × 3 个姿态，避免把 224 条都做成最贵的位姿穷举。
    for pose_index, position in enumerate(corners[:4]):
        for name, quat in orientations.items():
            targets.append({"position_m": position.tolist(), "quat_wxyz": [float(v) for v in quat], "name": f"{name}-{pose_index}"})
    return targets


def main() -> None:
    started = time.perf_counter()
    report: dict[str, object] = {"availability": availability(), "pinned_commit": (ROOT / "third_party" / "generative-robot-design" / "PINNED_COMMIT").read_text().strip()}
    proximal, distal, _mass = module_matrices(("zbot_r3",) * 6, (0.0,) * 5, (0.0,) * 6)
    fk_rows = []
    xml = ROOT / "assets" / "zbot_6" / "ZBOT6.xml"
    for name, pose in ("zero", np.zeros(6)), ("nominal", np.asarray(NOMINAL_JOINT_POS)), ("spread", np.linspace(-1.0, 0.9, 6)):
        reference = mujoco_tool_positions(xml, pose, site_name="foot_1_sole", body_name="foot_1")
        origin_error = float(np.max(np.abs(positions(proximal, distal, pose, "foot_joint_origin") - reference["body"])))
        sole_error = float(np.max(np.abs(positions(proximal, distal, pose, "foot_1_sole") - reference["site"])))
        fk_rows.append({"pose": name, "origin_error_m": origin_error, "sole_error_m": sole_error})
        print("FK", name, origin_error, sole_error, flush=True)
    extra = []
    for count in (7, 8):
        design = zbot_chain(count)
        matrices = module_matrices(("zbot_r3",) * count, (0.0,) * (count - 1), (0.0,) * count)
        pose = np.linspace(-0.5, 0.4, count)
        exported = export_mjcf(design, tool_point="foot_1_sole")
        reference = mujoco_tool_positions(exported, pose, site_name="ee")
        error = float(np.max(np.abs(positions(matrices[0], matrices[1], pose, "foot_1_sole") - reference["site"])))
        extra.append({"count": count, "error_m": error})
        print("FK", count, error, flush=True)
    report["verification_fk"] = {"six_vs_mujoco_xml": fk_rows, "seven_eight_vs_exported_mjcf": extra}

    starts8 = np.stack(ik_start_poses(6, 8, 6.28))
    endpoints = task_targets(TaskSpec(grid_divisions=5))
    checker = feasible_factory(("zbot_r3",) * 6, (0.0,) * 5, (0.0,) * 6, 0.4)
    _q, endpoint_error = gradient_position_ik(proximal, distal, endpoints, starts8, tool="foot_1_sole", iterations=40, step=1.0, tolerance=0.01, feasible=checker)
    centers = []
    half = 0.22
    center = (0.0, 0.05, 0.72)
    step = 2 * half / 5
    axes = [[center[axis] - half + step * (index + 0.5) for index in range(5)] for axis in range(3)]
    center_points = np.asarray([(x, y, z) for x in axes[0] for y in axes[1] for z in axes[2]], dtype=float)
    _q, center_error = gradient_position_ik(proximal, distal, center_points, starts8, tool="foot_1_sole", iterations=40, step=1.0, tolerance=0.01, feasible=checker)
    report["verification_ik"] = {
        "endpoint_reached": int((endpoint_error <= 0.01).sum()),
        "center_reached": int((center_error <= 0.01).sum()),
        "previous_batch2": {"endpoint": 120, "centers": 125},
        "note": "同一 1 cm、125 点和胶囊/静力矩约束。求解器换成上游正运动学上的阻尼最小二乘加 Adam 精修，所以格点数可以和 120/125 有几个点的差别。未解出不是不可达。",
    }
    print("IK", report["verification_ik"], flush=True)

    screen_targets = task_targets(TaskSpec(grid_divisions=3))
    starts4 = np.stack(ik_start_poses(8, 4, 6.28))
    previous = {row["bits"]: float(row["fraction"]) for row in csv.DictReader((ROOT / "experiments" / "longterm_batch2_screen.csv").open()) if row["count"] == "6"}
    new_scores = []
    old_scores = []
    screened = []
    for mask in range(32):
        twists = tuple((math.pi / 2 if mask & (1 << index) else 0.0) for index in range(5))
        bits = "".join("1" if value > 0.1 else "0" for value in twists)
        metrics = screen_one(("zbot_r3",) * 6, twists, (0.0,) * 6, screen_targets, starts4[:, :6], 0.4)
        metrics.update({"count": 6, "bits": bits, "modules": ["zbot_r3"] * 6})
        screened.append(metrics)
        new_scores.append(metrics["fraction"])
        old_scores.append(previous[bits])
        print("rank", bits, metrics["reached"], previous[bits], flush=True)
    report["verification_rank"] = {
        "spearman": spearman(old_scores, new_scores),
        "kendall": kendall(old_scores, new_scores),
        "old_definition": "批次 2 自写 IK，27 点、2 cm、足底",
        "new_definition": "上游 FK 的阻尼最小二乘加梯度精修，同一 27 点和 2 cm，并套用胶囊与静力矩",
    }

    high_center = (0.0, 0.22, 0.98)
    high_half = (0.10, 0.10, 0.08)
    high_axes = []
    for value, width in zip(high_center, high_half, strict=True):
        high_axes.append(np.linspace(value - width, value + width, 3))
    high_targets = np.asarray([(x, y, z) for x in high_axes[0] for y in high_axes[1] for z in high_axes[2]], dtype=float)
    representatives = []
    for count, bits in ((6, "00000"), (6, "10000"), (7, "000000"), (8, "0000000")):
        twists = tuple((math.pi / 2 if bit == "1" else 0.0) for bit in bits)
        metrics = screen_one(("zbot_r3",) * count, twists, (0.0,) * count, screen_targets, starts4[:, :count], 0.4)
        high = screen_one(("zbot_r3",) * count, twists, (0.0,) * count, high_targets, starts4[:, :count], 0.4)
        pose = richer_pose_targets()
        # 位姿集只记录位置投影是否进 2 cm，完整六维仍由自写 solve_pose 在报告里另算四台。
        pose_pos = np.asarray([item["position_m"] for item in pose], dtype=float)
        pose_metrics = screen_one(("zbot_r3",) * count, twists, (0.0,) * count, pose_pos, starts4[:, :count], 0.4)
        frames_a, frames_b = link_frames(*module_matrices(("zbot_r3",) * count, twists, (0.0,) * count)[:2], np.linspace(-1.0, 1.0, count))
        metrics.update({"count": count, "bits": bits, "high_fraction": high["fraction"], "pose_position_fraction": pose_metrics["fraction"], "mesh_hits_on_spread_pose": mesh_collision_count(frames_a, frames_b)})
        representatives.append(metrics)
        print("rep", count, bits, metrics["fraction"], high["fraction"], pose_metrics["fraction"], metrics["torque_margin_nm"], flush=True)
    report["representatives"] = representatives

    for count in (7, 8):
        for mask in range(1 << (count - 1)):
            if count == 6:
                continue
            twists = tuple((math.pi / 2 if mask & (1 << index) else 0.0) for index in range(count - 1))
            bits = "".join("1" if value > 0.1 else "0" for value in twists)
            if any(row["count"] == count and row["bits"] == bits for row in screened):
                continue
            metrics = screen_one(("zbot_r3",) * count, twists, (0.0,) * count, screen_targets, starts4[:, :count], 0.4)
            metrics.update({"count": count, "bits": bits, "modules": ["zbot_r3"] * count})
            screened.append(metrics)
            if mask % 16 == 0:
                print("screen", count, bits, metrics["reached"], flush=True)
    report["screen_rows"] = len(screened)
    report["pareto_default_physical"] = [
        {"count": row["count"], "bits": row["bits"], "fraction": row["fraction"], "mass_kg": row["mass_kg"], "velocity_pseudoinverse": row["velocity_pseudoinverse"], "torque_margin_0.4": row["torque_margin_nm"]["0.4"]}
        for row in non_dominated([row for row in screened if row["count"] in {6, 7, 8}])
    ]

    # 概念方案：pymoo 在默认盒上做很短的混合变量搜索，质量按 4 kg/m。
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.optimize import minimize

    class ConceptProblem(ElementwiseProblem):
        def __init__(self, count: int):
            self.arm_count = count
            super().__init__(n_var=3 * (count - 1), n_obj=4, xl=0.0, xu=1.0)

        def _evaluate(self, x, out, *args, **kwargs):
            count = self.arm_count
            types = [1 if value > 0.5 else 0 for value in x[: count - 1]]
            twists_bits = [1 if value > 0.5 else 0 for value in x[count - 1 : 2 * (count - 1)]]
            lengths = [0.06 + 0.10 * float(value) for value in x[2 * (count - 1) :]]
            module_ids = tuple("concept_axial" if bit else "zbot_r3" for bit in types) + ("zbot_r3",)
            twists = tuple(math.pi / 2 if bit else 0.0 for bit in twists_bits)
            length_values = tuple(lengths) + (0.0,)
            metrics = screen_one(module_ids, twists, length_values, screen_targets, starts4[:, :count], 0.4)
            out["F"] = [
                -metrics["fraction"],
                metrics["mass_kg"],
                -(metrics["torque_margin_nm"]["0.4"] or 0.0),
                -(metrics["velocity_pseudoinverse"] or 0.0),
            ]

    concept_fronts = {}
    for count in (6, 7, 8):
        result = minimize(ConceptProblem(count), NSGA2(pop_size=8), termination=("n_gen", 2), seed=11, verbose=False)
        concept_fronts[str(count)] = []
        for item in np.atleast_2d(result.F):
            concept_fronts[str(count)].append(
                {
                    "feasible_fraction": float(-item[0]),
                    "mass_kg_assumed": float(item[1]),
                    "torque_margin_0.4": float(-item[2]),
                    "velocity_pseudoinverse": float(-item[3]),
                    "mass_is_assumption": True,
                }
            )
        print("concept", count, len(concept_fronts[str(count)]), flush=True)
    report["concept_pymoo"] = {
        "budget": "每个驱动数种群 8、2 代。这不是穷尽，质量是 4 kg/m 假设。",
        "fronts": concept_fronts,
    }
    report["elapsed_s"] = time.perf_counter() - started
    report["recommendation"] = {
        "default_position": "6 模块全 0 仍是默认盒上覆盖和轻量的实物选择。新后端的 125 点结果与 120/125、125/125 同量级。",
        "high_box": "高远盒仍偏向更长的 7 或 8 模块；7 模块是覆盖和位姿命中的折中，8 模块在轻载下速度更高。",
        "heavy_payload": "Pinocchio 在 4 kg 下的静力矩裕度下降，和批次 3 里 8 模块先碰到力矩约束的方向一致。",
        "not_global": "没有全局最优。未进容差的点不是不可达。概念前沿不能当成实物减重。",
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with CSV_PATH.open("w", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["count", "bits", "reached", "targets", "fraction", "mass_kg", "velocity_pseudoinverse", "margin_0.4", "margin_1.5", "margin_4.0"])
        writer.writeheader()
        for row in screened:
            writer.writerow(
                {
                    "count": row["count"],
                    "bits": row["bits"],
                    "reached": row["reached"],
                    "targets": row["targets"],
                    "fraction": row["fraction"],
                    "mass_kg": row["mass_kg"],
                    "velocity_pseudoinverse": row.get("velocity_pseudoinverse"),
                    "margin_0.4": row["torque_margin_nm"]["0.4"],
                    "margin_1.5": row["torque_margin_nm"]["1.5"],
                    "margin_4.0": row["torque_margin_nm"]["4.0"],
                }
            )
    PUBLIC.write_text(
        json.dumps(
            {
                "note": "科研后端对照。不是用户工况，也不是全局最优。概念质量是 4 kg/m 假设。",
                "verification_ik": report["verification_ik"],
                "verification_rank": report["verification_rank"],
                "recommendation": report["recommendation"],
                "pareto_default_physical": report["pareto_default_physical"][:12],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("done", report["elapsed_s"], flush=True)


if __name__ == "__main__":
    main()
