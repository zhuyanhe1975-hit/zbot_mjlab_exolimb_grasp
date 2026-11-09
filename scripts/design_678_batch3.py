"""批次 3：足底点上的场景/载荷扫描，以及少量六维位姿目标。

场景在运行前写死，不是用户实测工况。只复核已有的 6/7/8 实物代表。
默认位置盒的 125 点结果仍用批次 2，不在这里重算。

用法::

    python scripts/design_678_batch3.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401

import numpy as np

from zbot_exolimb_grasp.platform.math3d import axis_angle, mul
from zbot_exolimb_grasp.platform.model import (
    evaluate_design,
    evaluate_pose_targets,
    forward_chain,
    sole_site,
    solve_pose,
    zbot_chain,
)
from zbot_exolimb_grasp.platform.schema import TaskSpec

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "experiments" / "cache" / "longterm_batch3.json"
STATE = ROOT / "experiments" / "longterm_search_state.json"
OUT = ROOT / "experiments" / "longterm_batch3.json"
PUBLIC = ROOT / "web" / "public" / "longterm_batch3.json"
CSV_PATH = ROOT / "experiments" / "longterm_batch3_scenarios.csv"
VERSION = "batch3-v1"

# 与批次 2 复核同一套足底点和种子。这里用每轴 4 份、64 点，四台机器同一预算。
SENS = dict(
    grid_divisions=4,
    position_tolerance_m=0.01,
    ik_starts=6,
    ik_iterations=30,
    seed=11,
    tool_point="foot_1_sole",
    compute_velocity_lp=True,
    compute_velocity_lp_fixed14=True,
)
POSE_POS_TOL = 0.01
POSE_ROT_TOL = math.radians(5.0)
POSE_POS_RELAXED = 0.02
POSE_ROT_RELAXED = math.radians(15.0)

# 事先固定。不是用户工况。
SCENARIOS = (
    {"id": "default_p0.4", "center_m": (0.0, 0.05, 0.72), "half_m": (0.22, 0.22, 0.22), "payload_kg": 0.4, "note": "与批次 2 相同的默认盒，本批只用 64 点预算做横向对照"},
    {"id": "default_p1.5", "center_m": (0.0, 0.05, 0.72), "half_m": (0.22, 0.22, 0.22), "payload_kg": 1.5, "note": "默认盒，载荷高于 0.4 kg"},
    {"id": "default_p4.0", "center_m": (0.0, 0.05, 0.72), "half_m": (0.22, 0.22, 0.22), "payload_kg": 4.0, "note": "默认盒，更重载荷，用来看力矩上限"},
    {"id": "high_p0.4", "center_m": (0.0, 0.22, 0.98), "half_m": (0.10, 0.10, 0.08), "payload_kg": 0.4, "note": "更远更高的小盒"},
    {"id": "near_p0.4", "center_m": (0.15, 0.12, 0.48), "half_m": (0.10, 0.10, 0.08), "payload_kg": 0.4, "note": "更近更低的小盒"},
    {"id": "side_p0.4", "center_m": (0.24, -0.05, 0.64), "half_m": (0.10, 0.10, 0.10), "payload_kg": 0.4, "note": "偏到一侧的小盒"},
    {"id": "high_p1.5", "center_m": (0.0, 0.22, 0.98), "half_m": (0.10, 0.10, 0.08), "payload_kg": 1.5, "note": "高处小盒加重载荷"},
)

CANDIDATES = (
    {"count": 6, "bits": "00000", "role": "all_zero"},
    {"count": 6, "bits": "10000", "role": "screen_coverage_batch2"},
    {"count": 7, "bits": "000000", "role": "all_zero"},
    {"count": 8, "bits": "0000000", "role": "all_zero"},
)


def identity() -> str:
    site = sole_site()
    pos = ",".join(f"{value:.10g}" for value in site.pos_m)
    quat = ",".join(f"{value:.10g}" for value in site.quat_wxyz)
    return f"{site.source_commit}:{site.name}:{pos}:{quat}"


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def design_of(spec: dict):
    twists = tuple((math.pi / 2 if bit == "1" else 0.0) for bit in spec["bits"])
    return zbot_chain(spec["count"], twists)


def task_for(scenario: dict, **overrides) -> TaskSpec:
    data = dict(
        center_m=scenario["center_m"],
        half_m=scenario["half_m"],
        payload_kg=scenario["payload_kg"],
        effort_limit_nm=20.0,
        velocity_limit_rad_s=1.5,
        sample_count=0,
        **SENS,
    )
    data.update(overrides)
    return TaskSpec(**data)


def cell_centers(scenario: dict, divisions: int = 4) -> np.ndarray:
    axes = []
    for center, half in zip(scenario["center_m"], scenario["half_m"], strict=True):
        step = 2.0 * half / divisions
        axes.append([center - half + step * (index + 0.5) for index in range(divisions)])
    return np.asarray([(x, y, z) for x in axes[0] for y in axes[1] for z in axes[2]], dtype=float)


def compact(metrics: dict) -> dict:
    keys = [
        "task_targets_reached",
        "task_target_count",
        "task_feasible_fraction",
        "task_manipulability_median",
        "task_velocity_median",
        "task_velocity_lp_median_m_s",
        "task_velocity_lp_fixed14_median_m_s",
        "task_torque_margin_median_nm",
        "q0_collision",
        "q0_peak_torque_nm",
        "mass_kg",
        "driven_joint_count",
    ]
    return {key: metrics.get(key) for key in keys}


def evaluate_cached(cache: dict, kind: str, scenario: dict, candidate: dict, *, centers: bool = False) -> dict:
    key = f"{VERSION}:{identity()}:{kind}:{scenario['id']}:{candidate['count']}:{candidate['bits']}"
    if key in cache:
        print(f"缓存 {kind} {scenario['id']} {candidate['count']} {candidate['bits']}", flush=True)
        return cache[key]
    design = design_of(candidate)
    task = task_for(scenario)
    started = time.perf_counter()
    if centers:
        metrics = evaluate_design(design, task, targets=cell_centers(scenario), target_set="cell_centers")
    else:
        metrics = evaluate_design(design, task)
    row = {"scenario": scenario["id"], "candidate": candidate, "metrics": compact(metrics), "elapsed_s": time.perf_counter() - started}
    cache[key] = row
    save_json(CACHE, cache)
    reached = metrics["task_targets_reached"]
    print(
        f"{kind} {scenario['id']} {candidate['count']}模块 {candidate['bits']}: {reached}/{metrics['task_target_count']} "
        f"轴LP {metrics['task_velocity_lp_median_m_s']} 14向 {metrics['task_velocity_lp_fixed14_median_m_s']}",
        flush=True,
    )
    return row


def pose_orientation_pair() -> tuple[np.ndarray, np.ndarray]:
    """工具姿态取 6 模块全 0、关节为 0 时的 foot_1_sole 世界姿态，再加绕世界 Z 转 90°。"""
    straight = forward_chain(zbot_chain(6), np.zeros(6), tool_point="foot_1_sole")
    base = np.asarray(straight["ee_quat"][0], dtype=np.float64)
    yaw = axis_angle(np.array([0.0, 0.0, 1.0]), math.pi / 2)
    return base, mul(yaw, base)


def independent_pose_targets() -> list[dict]:
    quat_a, quat_b = pose_orientation_pair()
    center = (0.0, 0.05, 0.62)
    half = (0.08, 0.08, 0.08)
    axes = []
    for value, width in zip(center, half, strict=True):
        axes.append((value - width, value + width))
    targets = []
    index = 0
    for name, quat in (("sole_zero", quat_a), ("sole_yaw90", quat_b)):
        for x in axes[0]:
            for y in axes[1]:
                for z in axes[2]:
                    targets.append(
                        {
                            "id": f"{name}-{index}",
                            "position_m": [x, y, z],
                            "quat_wxyz": [float(value) for value in quat],
                            "orientation_name": name,
                        }
                    )
                    index += 1
    return targets


def regression_pose(cache: dict) -> list[dict]:
    rows = []
    task = TaskSpec(tool_point="foot_1_sole", ik_starts=4, ik_iterations=30, payload_kg=0.4, effort_limit_nm=20.0, sample_count=0)
    for count in (6, 7, 8):
        key = f"{VERSION}:{identity()}:pose_regression:{count}"
        if key not in cache:
            true_q = np.linspace(-0.8, 0.7, count)
            design = zbot_chain(count)
            chain = forward_chain(design, true_q, tool_point="foot_1_sole")
            solved = solve_pose(
                design,
                chain["ee"][0],
                chain["ee_quat"][0],
                task,
                position_tolerance_m=POSE_POS_TOL,
                rotation_tolerance_rad=POSE_ROT_TOL,
                extra_starts=[true_q + 0.15],
            )
            cache[key] = {
                "count": count,
                "solved": bool(solved["solved"]),
                "position_error_m": solved["position_error_m"],
                "rotation_error_rad": solved["rotation_error_rad"],
                "source": "该臂自身的一组已知关节角做正运动学，再从扰动起点求回",
            }
            save_json(CACHE, cache)
        rows.append(cache[key])
        print(f"六维回归 {count} 模块 {cache[key]['solved']}", flush=True)
    return rows


def pose_compare(cache: dict) -> dict:
    targets = independent_pose_targets()
    task = TaskSpec(tool_point="foot_1_sole", ik_starts=8, ik_iterations=40, payload_kg=0.4, effort_limit_nm=20.0, sample_count=0, seed=11)
    rows = []
    for candidate in CANDIDATES:
        key = f"{VERSION}:{identity()}:pose16:{candidate['count']}:{candidate['bits']}"
        if key not in cache:
            strict = evaluate_pose_targets(
                design_of(candidate),
                targets,
                task,
                position_tolerance_m=POSE_POS_TOL,
                rotation_tolerance_rad=POSE_ROT_TOL,
            )
            relaxed = evaluate_pose_targets(
                design_of(candidate),
                targets,
                task,
                position_tolerance_m=POSE_POS_RELAXED,
                rotation_tolerance_rad=POSE_ROT_RELAXED,
            )
            cache[key] = {"candidate": candidate, "strict": strict, "relaxed_diagnostic": {k: relaxed[k] for k in ("targets_reached", "feasible_fraction", "position_tolerance_m", "rotation_tolerance_rad")}}
            save_json(CACHE, cache)
            print(f"六维 {candidate['count']} {candidate['bits']}: {strict['targets_reached']}/16，放宽诊断 {relaxed['targets_reached']}/16", flush=True)
        rows.append(cache[key])
    quat_a, quat_b = pose_orientation_pair()
    return {
        "position_tolerance_m": POSE_POS_TOL,
        "rotation_tolerance_rad": POSE_ROT_TOL,
        "rotation_tolerance_deg": 5.0,
        "relaxed_is_not_acceptance": True,
        "orientation_source": "6 模块全 0、关节为 0 时 foot_1_sole 的世界姿态；另一组是该姿态再绕世界 +Z 转 90°。工具系就是 XML 里的 foot_1_sole。",
        "position_source": "中心 (0, 0.05, 0.62) m、半尺寸 0.08 m 的立方体 8 个角，两种姿态，共 16 个目标。",
        "reference_quat_wxyz": [float(value) for value in quat_a],
        "yaw90_quat_wxyz": [float(value) for value in quat_b],
        "ik": {"starts": 8, "iterations": 40, "characteristic_length_m": 0.1, "seed_note": "起点由 ik_start_poses 确定，不用随机数"},
        "rows": rows,
    }


def matrix_rows(endpoint_rows: list[dict], center_rows: list[dict]) -> list[dict]:
    centers = {(row["scenario"], row["candidate"]["count"], row["candidate"]["bits"]): row for row in center_rows}
    table = []
    for row in endpoint_rows:
        key = (row["scenario"], row["candidate"]["count"], row["candidate"]["bits"])
        other = centers.get(key)
        metrics = row["metrics"]
        table.append(
            {
                "scenario": row["scenario"],
                "count": row["candidate"]["count"],
                "bits": row["candidate"]["bits"],
                "role": row["candidate"]["role"],
                "endpoint_reached": metrics["task_targets_reached"],
                "endpoint_count": metrics["task_target_count"],
                "centers_reached": None if other is None else other["metrics"]["task_targets_reached"],
                "velocity_lp_axis": metrics["task_velocity_lp_median_m_s"],
                "velocity_lp_fixed14": metrics["task_velocity_lp_fixed14_median_m_s"],
                "velocity_pseudoinverse": metrics["task_velocity_median"],
                "manipulability": metrics["task_manipulability_median"],
                "torque_margin_nm": metrics["task_torque_margin_median_nm"],
                "q0_collision": metrics["q0_collision"],
                "mass_kg": metrics["mass_kg"],
            }
        )
    return table


def choose(table: list[dict], scenario_id: str) -> dict:
    group = [row for row in table if row["scenario"] == scenario_id and row["endpoint_reached"] is not None]
    if not group:
        return {"scenario": scenario_id}
    best_cover = max(group, key=lambda row: (row["endpoint_reached"], -(row["count"])))
    baseline = next(row for row in group if row["count"] == 6 and row["bits"] == "00000")
    gain = (best_cover["endpoint_reached"] - baseline["endpoint_reached"]) / float(best_cover["endpoint_count"])
    centers_ok = True
    center_gain = None
    if best_cover["centers_reached"] is not None and baseline["centers_reached"] is not None:
        center_gain = (best_cover["centers_reached"] - baseline["centers_reached"]) / float(best_cover["endpoint_count"])
        centers_ok = center_gain >= 0.03
    cover_choice = best_cover if gain >= 0.03 and centers_ok else baseline
    speed_pool = [row for row in group if row["endpoint_reached"] >= best_cover["endpoint_reached"] - 0.08 * best_cover["endpoint_count"]]
    speed = max(speed_pool, key=lambda row: row["velocity_lp_fixed14"] or -1)
    leader_ratio = (speed["velocity_lp_fixed14"] or 0) / max(baseline["velocity_lp_fixed14"] or 1e-9, 1e-9)
    cover_drop = (baseline["endpoint_reached"] - speed["endpoint_reached"]) / float(baseline["endpoint_count"])
    speed_choice = speed if leader_ratio > 1.05 and cover_drop <= 0.03 else baseline
    chosen_ratio = (speed_choice["velocity_lp_fixed14"] or 0) / max(baseline["velocity_lp_fixed14"] or 1e-9, 1e-9)
    return {
        "scenario": scenario_id,
        "coverage": {
            "count": cover_choice["count"],
            "bits": cover_choice["bits"],
            "endpoint_gain_vs_6_zero": gain,
            "center_gain_vs_6_zero": center_gain,
        },
        "speed_fixed14": {
            "count": speed_choice["count"],
            "bits": speed_choice["bits"],
            "ratio_vs_6_zero": chosen_ratio,
            "leader_count": speed["count"],
            "leader_bits": speed["bits"],
            "leader_ratio_vs_6_zero": leader_ratio,
            "leader_endpoint_drop_vs_6_zero": cover_drop,
        },
        "light": {"count": 6, "bits": "00000", "mass_kg": baseline["mass_kg"]},
    }


def main() -> None:
    started = time.perf_counter()
    cache = load_json(CACHE)
    endpoint_rows = [evaluate_cached(cache, "endpoint64", scenario, candidate) for scenario in SCENARIOS for candidate in CANDIDATES]
    # 只对端点上有人比 6 模块全 0 更高的场景，再跑同一预算的单元中心。最多三个场景。
    interesting = []
    for scenario in SCENARIOS:
        group = [row for row in endpoint_rows if row["scenario"] == scenario["id"]]
        base = next(row for row in group if row["candidate"]["bits"] == "00000" and row["candidate"]["count"] == 6)
        if any(row["metrics"]["task_targets_reached"] > base["metrics"]["task_targets_reached"] for row in group):
            interesting.append(scenario)
    center_rows = [evaluate_cached(cache, "centers64", scenario, candidate, centers=True) for scenario in interesting for candidate in CANDIDATES]
    regressions = regression_pose(cache)
    poses = pose_compare(cache)
    table = matrix_rows(endpoint_rows, center_rows)
    choices = [choose(table, scenario["id"]) for scenario in SCENARIOS]
    elapsed = time.perf_counter() - started
    pose_rows = poses["rows"]
    strict_hits = [row["strict"]["targets_reached"] for row in pose_rows]
    relaxed_hits = [row["relaxed_diagnostic"]["targets_reached"] for row in pose_rows]
    solver_failed = all(value == 0 for value in strict_hits) and all(value == 0 for value in relaxed_hits)
    previous = load_json(STATE)
    batch2_elapsed = float((previous.get("batch2") or {}).get("elapsed_s") or 0.0)
    prior = float((previous.get("batch3") or {}).get("elapsed_s") or 0.0) if previous.get("checkpoint_version", 0) >= 3 else 0.0
    batch3_elapsed = elapsed if prior == 0.0 else prior + elapsed
    site = sole_site()
    next_batch = {
        "not_started": True,
        "intent": (
            "16 个六维目标在 1 cm/5° 和 2 cm/15° 诊断下都没有命中。下一批只把这 4 台、这 16 个目标的 IK 加到 12 个起点和 80 次迭代，不新增场景，也不加长模块搜索。"
            if solver_failed
            else "已测示例场景上的 6/7/8 实物规则可以交付，但这不是全局最优，长期状态也不标完成。没有证据支持继续加概念长度预算。下一批只应在出现一个用户给出的工况时，用同一足底点和这四台代表重核那一个工况。"
        ),
    }
    state = {
        "checkpoint_version": 3,
        "status": "batch3_complete",
        "longterm_status": "open",
        "bounded_physical_selection_ready": not solver_failed,
        "suggest_close": False,
        "suggest_close_scope": {
            "default_box_position_only": True,
            "note": "批次 2 的 suggest_close 只收束默认位置盒上的加长搜索。它不收束载荷、任务区和六维位姿，也不表示长期任务完成。",
        },
        "completed_batches": ["batch1_physical_6_7_8", "batch2_sole_and_concept", "batch3_scenarios_and_pose"],
        "elapsed_s": float(previous.get("elapsed_s") or 0.0) - prior + batch3_elapsed,
        "batch1": previous.get("batch1"),
        "batch2": previous.get("batch2"),
        "batch3": {
            "batch_id": "batch3_scenarios_and_pose",
            "status": "batch3_complete",
            "elapsed_s": batch3_elapsed,
            "not_a_measured_user_task": True,
            "tool_point": {
                "name": site.name,
                "body": site.body,
                "pos_m": list(site.pos_m),
                "quat_wxyz": list(site.quat_wxyz),
                "source_commit": site.source_commit,
                "source_path": site.source_path,
            },
            "scenarios": [dict(item) for item in SCENARIOS],
            "shared_budget": SENS,
            "candidates": [dict(item) for item in CANDIDATES],
            "center_followup_scenarios": [item["id"] for item in interesting],
            "table": table,
            "choices": choices,
            "pose_regression": regressions,
            "pose_targets": poses,
            "concept_search_limit": "批次 2 对每个驱动数只做了种群 16、4 代、约 80 次评价。本批没有扩大概念搜索。",
            "velocity_names": {
                "task_velocity_lp_median_m_s": "六个坐标轴方向的最小值",
                "task_velocity_lp_fixed14_median_m_s": "六个坐标轴加八个体对角线，共 14 个预固定方向的最小值",
            },
            "solver_failed_on_pose_set": solver_failed,
        },
        "next_batch": next_batch,
    }
    # 长期总状态保持 open。默认盒子任务的收束单独记在 batch2。
    if isinstance(state["batch2"], dict):
        state["batch2"]["suggest_close_scope"] = "default_box_position_subtask_only"
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUT.write_text(json.dumps(state["batch3"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["scenario,count,bits,endpoint_reached,endpoint_count,centers_reached,lp_axis,lp_fixed14,pseudoinverse,manipulability,torque_margin_nm,mass_kg\n"]
    for row in table:
        lines.append(
            f"{row['scenario']},{row['count']},{row['bits']},{row['endpoint_reached']},{row['endpoint_count']},{row['centers_reached']},"
            f"{row['velocity_lp_axis']},{row['velocity_lp_fixed14']},{row['velocity_pseudoinverse']},{row['manipulability']},{row['torque_margin_nm']},{row['mass_kg']}\n"
        )
    CSV_PATH.write_text("".join(lines), encoding="utf-8")
    PUBLIC.write_text(
        json.dumps(
            {
                "tool_point": "foot_1_sole",
                "note": "批次 3 是示例场景和六维位姿，不是用户实测工况，也不是全局最优。",
                "scenarios": [dict(item) for item in SCENARIOS],
                "choices": choices,
                "pose_summary": [
                    {
                        "count": row["candidate"]["count"],
                        "bits": row["candidate"]["bits"],
                        "strict_reached": row["strict"]["targets_reached"],
                        "relaxed_reached": row["relaxed_diagnostic"]["targets_reached"],
                    }
                    for row in pose_rows
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"批次 3 用时 {elapsed:.1f} s，六维求解失败 {solver_failed}", flush=True)


if __name__ == "__main__":
    main()
