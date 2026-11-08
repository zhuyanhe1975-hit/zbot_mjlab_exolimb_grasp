"""首批 6/7/8 模块实物枚举初筛，并对少量候选做 1 cm、125 点复核。

7、8 模块是位置任务的冗余臂，不是恰好 6 驱动。实物尺寸不缩放。
结果写入缓存，重复运行会跳过已完成的键。

用法::

    python scripts/design_678_batch1.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401

from zbot_exolimb_grasp.platform.exporting import export_mjcf
from zbot_exolimb_grasp.platform.model import evaluate_design, generic_rank_report, zbot_chain
from zbot_exolimb_grasp.platform.schema import TaskSpec
from zbot_exolimb_grasp.platform.search import design_to_dict

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "experiments" / "cache" / "longterm_batch1_screen.json"
REVIEW_CACHE = ROOT / "experiments" / "cache" / "longterm_batch1_review.json"
STATE = ROOT / "experiments" / "longterm_search_state.json"
OUT = ROOT / "experiments" / "longterm_batch1_678.json"
PUBLIC = ROOT / "web" / "public" / "longterm_batch1.json"
CACHE_VERSION = "batch1-screen-v1"
REVIEW_VERSION = "batch1-review-v1"

SCREEN = dict(grid_divisions=3, position_tolerance_m=0.02, ik_starts=3, ik_iterations=10, seed=11)
REVIEW = dict(grid_divisions=5, position_tolerance_m=0.01, ik_starts=8, ik_iterations=50, seed=11, compute_velocity_lp=True)


def task_of(spec: dict) -> TaskSpec:
    return TaskSpec(
        center_m=(0.0, 0.05, 0.72),
        half_m=(0.22, 0.22, 0.22),
        payload_kg=0.4,
        effort_limit_nm=20.0,
        velocity_limit_rad_s=1.5,
        sample_count=0,
        **spec,
    )


def cell_centers(divisions: int = 5):
    import numpy as np

    center = (0.0, 0.05, 0.72)
    half = (0.22, 0.22, 0.22)
    axes = []
    for value, width in zip(center, half, strict=True):
        step = 2.0 * width / divisions
        axes.append([value - width + step * (index + 0.5) for index in range(divisions)])
    points = [(x, y, z) for x in axes[0] for y in axes[1] for z in axes[2]]
    return np.asarray(points, dtype=float)


def bits(twists) -> str:
    return "".join("1" if value > 0.1 else "0" for value in twists)


def compact(metrics: dict) -> dict:
    keys = [
        "driven_joint_count",
        "passive_body_count",
        "mass_kg",
        "task_feasible_fraction",
        "task_targets_reached",
        "task_target_count",
        "position_tolerance_m",
        "grid_divisions",
        "ik_starts",
        "ik_iterations",
        "task_manipulability_median",
        "task_manipulability_valid",
        "task_velocity_median",
        "task_velocity_valid",
        "task_velocity_lp_median_m_s",
        "task_torque_margin_median_nm",
        "position_rank_deficient_fraction",
        "task_pose_spatial_rank_deficient_fraction",
        "q0_collision",
        "q0_peak_torque_nm",
    ]
    return {key: metrics.get(key) for key in keys}


def load_cache() -> dict:
    if not CACHE.is_file():
        return {}
    return json.loads(CACHE.read_text(encoding="utf-8"))


def save_cache(cache: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache), encoding="utf-8")


def screen_all(cache: dict) -> list[dict]:
    budget = task_of(SCREEN)
    rows = []
    for count, total in ((6, 32), (7, 64), (8, 128)):
        if total != 1 << (count - 1):
            raise SystemExit("接口规则与 0/π/2 枚举数量不一致")
        for mask in range(total):
            twists = tuple((math.pi / 2 if mask & (1 << index) else 0.0) for index in range(count - 1))
            label = bits(twists)
            key = f"{CACHE_VERSION}:{count}:{label}"
            if key not in cache:
                metrics = compact(evaluate_design(zbot_chain(count, twists), budget))
                cache[key] = metrics
                if mask % 16 == 0:
                    save_cache(cache)
                    print(f"筛选 {count} {label} {metrics['task_targets_reached']}/{metrics['task_target_count']}", flush=True)
            rows.append({"count": count, "bits": label, "metrics": cache[key]})
    save_cache(cache)
    write_screen_csv(rows)
    return rows


def write_screen_csv(rows: list[dict]) -> None:
    path = ROOT / "experiments" / "longterm_batch1_screen.csv"
    header = "count,bits,mass_kg,reached,targets,fraction,manipulability,velocity_pseudoinverse,torque_margin_nm,q0_collision\n"
    lines = [header]
    for row in rows:
        metrics = row["metrics"]
        lines.append(
            f"{row['count']},{row['bits']},{metrics['mass_kg']},{metrics['task_targets_reached']},"
            f"{metrics['task_target_count']},{metrics['task_feasible_fraction']},{metrics['task_manipulability_median']},"
            f"{metrics['task_velocity_median']},{metrics['task_torque_margin_median_nm']},{metrics['q0_collision']}\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def screen_summary(rows: list[dict]) -> dict:
    summary = {}
    for count in (6, 7, 8):
        group = [row for row in rows if row["count"] == count]
        best = max(group, key=lambda row: row["metrics"]["task_feasible_fraction"])
        summary[str(count)] = {
            "sequences": len(group),
            "best_bits": best["bits"],
            "best_reached": best["metrics"]["task_targets_reached"],
            "best_fraction": best["metrics"]["task_feasible_fraction"],
            "mass_kg": best["metrics"]["mass_kg"],
        }
    return summary


def pick(rows: list[dict]) -> list[tuple[int, str, str]]:
    """全 0 是每种模块数的代表。末位 90° 只作对照：它几乎不移动 foot_1 原点。

    6 模块另保留 11000。它是 2 cm 初筛里 22/27 的并列之一，末位为 0，不是唯一初筛冠军。
    """
    del rows
    return [
        (6, "00000", "all_zero"),
        (6, "11000", "screen_coverage_tie"),
        (7, "000000", "all_zero"),
        (7, "000001", "last_interface_same_point"),
        (8, "0000000", "all_zero"),
        (8, "0000001", "last_interface_same_point"),
    ]


def review(chosen: list[tuple[int, str, str]]) -> list[dict]:
    strict = task_of(REVIEW)
    centers = cell_centers(5)
    cache = json.loads(REVIEW_CACHE.read_text(encoding="utf-8")) if REVIEW_CACHE.is_file() else {}
    reviewed = []
    for count, label, reason in chosen:
        key = f"{REVIEW_VERSION}:{count}:{label}"
        if key in cache:
            row = cache[key]
            row["role"] = reason
            reviewed.append(row)
            print(f"复核缓存 {count}关节 {label}", flush=True)
            continue
        twists = tuple((math.pi / 2 if bit == "1" else 0.0) for bit in label)
        design = zbot_chain(count, twists)
        started = time.perf_counter()
        endpoint = evaluate_design(design, strict)
        centers_metrics = evaluate_design(design, strict, targets=centers, target_set="cell_centers")
        rank = generic_rank_safe(design)
        endpoint_view = compact(endpoint)
        endpoint_view["example_q"] = endpoint["example_q"]
        row = {
            "role": reason,
            "count": count,
            "bits": label,
            "driven_joint_count": count,
            "passive_body_count": count,
            "mass_kg": endpoint["mass_kg"],
            "design": design_to_dict(design),
            "endpoint_125": endpoint_view,
            "cell_centers_125": compact(centers_metrics),
            "generic_rank": rank,
            "elapsed_s": time.perf_counter() - started,
        }
        cache[key] = row
        REVIEW_CACHE.parent.mkdir(parents=True, exist_ok=True)
        REVIEW_CACHE.write_text(json.dumps(cache), encoding="utf-8")
        reviewed.append(row)
        print(
            f"复核 {count}关节 {label} {reason}: 端点 {endpoint['task_targets_reached']}/125，"
            f"中心 {centers_metrics['task_targets_reached']}/125，"
            f"伪逆速度 {endpoint['task_velocity_median']}，线性规划速度 {endpoint['task_velocity_lp_median_m_s']}",
            flush=True,
        )
    return reviewed


def generic_rank_safe(design):
    from zbot_exolimb_grasp.platform.model import generic_rank_report

    report = generic_rank_report(design, samples=8, seed=11)
    return {
        "structurally_degenerate": report["structurally_degenerate"],
        "by_length": report["by_length"],
    }


def goal_notes(reviewed: list[dict]) -> dict:
    representatives = [row for row in reviewed if row["role"] == "all_zero"]
    baseline = next(row for row in representatives if row["count"] == 6)
    def both_gains(row: dict) -> tuple[float, float]:
        return (
            row["endpoint_125"]["task_feasible_fraction"] - baseline["endpoint_125"]["task_feasible_fraction"],
            row["cell_centers_125"]["task_feasible_fraction"] - baseline["cell_centers_125"]["task_feasible_fraction"],
        )
    coverage = max(representatives, key=lambda row: (min(both_gains(row)), row["endpoint_125"]["task_feasible_fraction"]))
    endpoint_gain, center_gain = both_gains(coverage)
    reliable_coverage = endpoint_gain >= 0.03 and center_gain >= 0.03
    speed = max(representatives, key=lambda row: row["endpoint_125"]["task_velocity_lp_median_m_s"] or -1)
    base_speed = baseline["endpoint_125"]["task_velocity_lp_median_m_s"] or 0.0
    speed_value = speed["endpoint_125"]["task_velocity_lp_median_m_s"] or 0.0
    speed_endpoint, speed_center = both_gains(speed)
    speed_keeps_coverage = speed_endpoint >= -0.03 and speed_center >= -0.03
    return {
        "coverage_priority": {
            "count": coverage["count"] if reliable_coverage else 6,
            "bits": coverage["bits"] if reliable_coverage else "00000",
            "reliable_improvement_on_both_point_sets": reliable_coverage,
            "highest_worse_set_count": coverage["count"],
            "highest_worse_set_bits": coverage["bits"],
            "endpoint_gain": endpoint_gain,
            "cell_center_gain": center_gain,
            "why": "可靠改进必须在端点和单元中心两套 125 点上都至少高 0.03。只在一套上变多，记成点集取舍，不叫覆盖优胜。末位接口 90° 与全 0 的 foot_1 原点几乎重合，不单独推荐。",
        },
        "speed_priority": {
            "count": speed["count"] if speed_value > base_speed * 1.05 and speed_keeps_coverage else 6,
            "bits": speed["bits"] if speed_value > base_speed * 1.05 and speed_keeps_coverage else "00000",
            "reliable_improvement": speed_value > base_speed * 1.05 and speed_keeps_coverage,
            "best_reviewed_count": speed["count"],
            "best_reviewed_bits": speed["bits"],
            "endpoint_lp_m_s": speed_value,
            "baseline_lp_m_s": base_speed,
            "why": "速度用坐标轴线性规划，不用伪逆。更长的臂在同样 1.5 rad/s 关节限速下末端线速度更高，这是力臂尺度，不是软件把冗余臂算得更快。平移可操作度同样偏向更多关节。",
        },
        "light_priority": {
            "count": 6,
            "bits": "00000",
            "mass_kg": baseline["mass_kg"],
            "why": "实物 ZBot 不缩放，质量只随模块数增加。轻量优先是 6 模块全 0。",
        },
        "point_set_tradeoff": [
            {
                "count": row["count"],
                "bits": row["bits"],
                "endpoint_reached": row["endpoint_125"]["task_targets_reached"],
                "centers_reached": row["cell_centers_125"]["task_targets_reached"],
                "velocity_lp_m_s": row["endpoint_125"]["task_velocity_lp_median_m_s"],
                "mass_kg": row["mass_kg"],
            }
            for row in representatives
        ],
        "unique_optimum": False,
        "distal_joint_note": "跟踪点是 foot_1 关节原点。末关节的平移雅可比为 0，末位接口方位几乎不改变这个点。7 模块有 6 个能移动该点的关节，8 模块有 7 个。足底 site 相对关节有偏移，本批没有改成足底点。",
    }


def main() -> None:
    started = time.perf_counter()
    prior_elapsed = 0.0
    review_keys_before = 0
    if STATE.is_file():
        prior_elapsed = float(json.loads(STATE.read_text(encoding="utf-8")).get("elapsed_s") or 0.0)
    if REVIEW_CACHE.is_file():
        review_keys_before = len(json.loads(REVIEW_CACHE.read_text(encoding="utf-8")))
    cache = load_cache()
    rows = screen_all(cache)
    chosen = pick(rows)
    print("复核名单", chosen, flush=True)
    reviewed = review(chosen)
    review_keys_after = len(json.loads(REVIEW_CACHE.read_text(encoding="utf-8"))) if REVIEW_CACHE.is_file() else 0
    elapsed = prior_elapsed + (time.perf_counter() - started if review_keys_after > review_keys_before else 0.0)
    baseline = next(row for row in reviewed if row["bits"] == "0" * 5 and row["count"] == 6)
    goals = goal_notes(reviewed)
    state = {
        "checkpoint_version": 1,
        "cache_version": CACHE_VERSION,
        "cache_path": str(CACHE.relative_to(ROOT)),
        "batch_id": "batch1_physical_6_7_8",
        "status": "batch1_complete",
        "model": {
            "module": "zbot_r3",
            "kind": "physical",
            "size_locked": True,
            "interface_angles_rad": [0.0, math.pi / 2],
            "driven_joints_per_module": 1,
            "passive_bodies_per_module": 1,
            "legal_sequence_counts": {"6": 32, "7": 64, "8": 128},
            "note": "7 和 8 模块是 7、8 个驱动关节。跟踪点是 foot_1 关节原点，末关节平移雅可比为 0，所以能移动该点的关节数分别是 5、6、7。末位接口几乎不移动这个点。足底 site 有偏移，本批没有把任务点改到足底。",
        },
        "task": {
            "origin": "默认示例任务，不是用户给定工况，也不是通用最优。",
            "center_m": [0.0, 0.05, 0.72],
            "half_m": [0.22, 0.22, 0.22],
            "payload_kg": 0.4,
            "effort_limit_nm": 20.0,
            "velocity_limit_rad_s": 1.5,
            "velocity_limit_is_assumption": True,
        },
        "budgets": {
            "screen": SCREEN,
            "review": REVIEW,
            "review_also": "125 个单元中心点，同一 1 cm、8 起点、50 迭代",
        },
        "completed_batches": ["batch1_physical_6_7_8"],
        "elapsed_s": elapsed,
        "screen_rows": len(rows),
        "screen_summary": screen_summary(rows),
        "recommendations_by_goal": goals,
        "baseline_6": {"count": 6, "bits": "00000", "endpoint_125": baseline["endpoint_125"], "cell_centers_125": baseline["cell_centers_125"]},
        "reviewed": [
            {
                "count": row["count"],
                "bits": row["bits"],
                "role": row["role"],
                "endpoint_reached": row["endpoint_125"]["task_targets_reached"],
                "centers_reached": row["cell_centers_125"]["task_targets_reached"],
                "manipulability": row["endpoint_125"]["task_manipulability_median"],
                "velocity_pseudoinverse": row["endpoint_125"]["task_velocity_median"],
                "velocity_lp": row["endpoint_125"]["task_velocity_lp_median_m_s"],
                "mass_kg": row["mass_kg"],
                "spatial_rank_deficient_on_task_solutions": row["endpoint_125"]["task_pose_spatial_rank_deficient_fraction"],
                "generic_rank_degenerate": row["generic_rank"]["structurally_degenerate"],
            }
            for row in reviewed
        ],
        "unverified_assumptions": [
            "任务盒不是用户工况。",
            "关节速度 1.5 rad/s 无 R3 实测。",
            "胶囊碰撞和质点静力矩是近似。",
            "未解出的格点不是不可达证明。",
            "概念模块长度和质量尚未进入本批，留给下一批。",
            "六维秩只诊断瞬时运动，不是六维任务全通过。",
        ],
        "stop_criteria": [
            "两套独立 125 点、1 cm 容差下，候选的可行比例都不比 6 模块全 0 基准高 0.03，且线性规划速度也不更高，则不为该目标宣称改进。",
            "若连续两批在覆盖优先和速度优先上都没有过这个门槛，就停在 6 模块全 0 作为该默认任务的实物参考，不再加长枚举。",
        ],
        "next_batch": {
            "intent": "在已排除结构退化的概念模块上，做固定 6、7 或 8 驱动关节的长度与 0/π/2 连接联合优化，并与本批实物复核用同一 1 cm、125 点预算比较。",
            "not_started": True,
            "must_separate_from_hardware": True,
            "mass_assumption": "概念连杆继续标明 4 kg/m，不能写成 ZBot 实测减重。",
        },
    }
    # reviewed details are large; full rows saved beside the checkpoint
    OUT.write_text(json.dumps({"screen_top_note": "screen metrics are in the cache", "reviewed": reviewed, "goals": goals}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    public = {
        "note": "6/7/8 模块实物首批复核。默认示例任务，不是用户工况，也没有唯一全局最优。7 和 8 是冗余驱动，不是恰好 6 关节。",
        "limitations": [
            "平移可操作度是 sqrt(det(Jv Jv^T))，更多关节会改变尺度。",
            "伪逆速度是关节限速下的保守下界，不是冗余臂的最优速度。",
            "线性规划速度只取六个坐标轴方向的最小值，不是速度多面体的内切球。",
            "六维归一化秩只诊断瞬时运动，不是六维位姿任务全通过。",
            "实物尺寸锁定。速度上限 1.5 rad/s 没有实测。",
        ],
        "goals": goals,
        "baseline": {"design": baseline["design"], "metrics": baseline["endpoint_125"], "bits": baseline["bits"], "role": "baseline"},
        "feasible_pareto": [
            {
                "design": row["design"],
                "metrics": row["endpoint_125"],
                "centers": row["cell_centers_125"],
                "bits": row["bits"],
                "role": row["role"],
                "violation": 0.0,
                "generic_rank_degenerate": row["generic_rank"]["structurally_degenerate"],
            }
            for row in reviewed
        ],
    }
    PUBLIC.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC.write_text(json.dumps(public, ensure_ascii=False) + "\n", encoding="utf-8")
    mjcf_dir = ROOT / "experiments" / "dof678_mjcf"
    mjcf_dir.mkdir(parents=True, exist_ok=True)
    for row in reviewed:
        design = zbot_chain(row["count"], tuple(math.pi / 2 if bit == "1" else 0.0 for bit in row["bits"]))
        (mjcf_dir / f"zbot_{row['count']}_{row['bits']}.xml").write_text(export_mjcf(design), encoding="utf-8")
    print(f"用时 {elapsed_from(started):.1f} s")
    print(json.dumps(goals, ensure_ascii=False, indent=2))


def elapsed_from(started: float) -> float:
    return time.perf_counter() - started


if __name__ == "__main__":
    main()
