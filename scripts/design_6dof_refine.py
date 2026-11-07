"""更严的固定 6 驱动关节复核：六维秩、1 cm 容差和 125 个任务点。

不把每轴 2 份或 5 cm 初筛当成最终推荐。概念链若在随机一般位形上六维秩退化，
不进入完整位姿推荐。默认任务盒仍是示例，不是用户工况。

用法::

    python scripts/design_6dof_refine.py --out experiments/dof6_refine.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401

import numpy as np

from zbot_exolimb_grasp.platform.exporting import export_mjcf
from zbot_exolimb_grasp.platform.model import (
    actuation_report,
    baseline_design,
    evaluate_design,
    generic_rank_report,
    zbot_chain,
)
from zbot_exolimb_grasp.platform.schema import Design, TaskSpec
from zbot_exolimb_grasp.platform.search import design_from_dict, run_search, spec_from_dict


def example_task(**overrides) -> TaskSpec:
    values = dict(
        center_m=(0.0, 0.05, 0.72),
        half_m=(0.22, 0.22, 0.22),
        payload_kg=0.4,
        effort_limit_nm=20.0,
        velocity_limit_rad_s=1.5,
        sample_count=0,
        seed=1,
        position_tolerance_m=0.05,
        grid_divisions=3,
        ik_starts=4,
        ik_iterations=12,
    )
    values.update(overrides)
    return TaskSpec(**values)


def cell_centers(task: TaskSpec, divisions: int = 5) -> np.ndarray:
    """与端点格点错开的单元中心，用来检查结论是不是某一套格点的偶然。"""
    points = []
    axes = []
    for center, half in zip(task.center_m, task.half_m, strict=True):
        step = 2.0 * float(half) / divisions
        axes.append([float(center) - float(half) + step * (index + 0.5) for index in range(divisions)])
    for x in axes[0]:
        for y in axes[1]:
            for z in axes[2]:
                points.append((x, y, z))
    return np.asarray(points, dtype=np.float64)


def bits(twists) -> str:
    return "".join("1" if value > 0.1 else "0" for value in twists)


def design_dict(design: Design, label: str) -> dict:
    modules = []
    for index, module_id in enumerate(design.module_ids):
        modules.append(
            {
                "id": module_id,
                "kind": "physical" if module_id == "zbot_r3" else "conceptual",
                "parameters": dict(design.parameters[index]),
                "twist_to_next_rad": None if index == len(design.module_ids) - 1 else design.twists_rad[index],
            }
        )
    return {
        "topology": "serial",
        "label": label,
        "base_position_m": list(design.base_position_m),
        "modules": modules,
    }


def brief(metrics: dict) -> str:
    manip = metrics["task_manipulability_median"]
    speed = metrics["task_velocity_median"]
    spatial = metrics["task_pose_spatial_rank_deficient_fraction"]
    return (
        f"{metrics['task_targets_reached']}/{metrics['task_target_count']} "
        f"容差 {metrics['position_tolerance_m']} m，"
        f"可操作度 {'无效' if manip is None else f'{manip:.5f}'}，"
        f"速度 {'无效' if speed is None else f'{speed:.4f}'}，"
        f"任务解六维秩不足 {'无效' if spatial is None else f'{spatial:.2f}'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="6 驱动关节的秩诊断和更严位置复核")
    parser.add_argument("--out", type=Path, default=Path("experiments/dof6_refine.json"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--concept-pop", type=int, default=10)
    parser.add_argument("--concept-gen", type=int, default=3)
    args = parser.parse_args()
    started = time.perf_counter()
    tight = example_task(position_tolerance_m=0.01, grid_divisions=3, ik_starts=4, ik_iterations=20, seed=args.seed)
    print("1 cm、27 点重筛全部 32 种实物连接。这仍是筛选，不是最终推荐。")
    physical = []
    ranks = []
    for mask in range(32):
        twists = tuple((math.pi / 2 if mask & (1 << index) else 0.0) for index in range(5))
        design = zbot_chain(6, twists)
        metrics = evaluate_design(design, tight)
        rank = generic_rank_report(design, samples=8, seed=args.seed)
        label = bits(twists)
        physical.append({"label": label, "family": "physical", "design": design_dict(design, label), "metrics": metrics, "generic_rank": rank})
        ranks.append(rank["structurally_degenerate"])
        print(f"实物 {label} {brief(metrics)} 结构退化 {rank['structurally_degenerate']}")
    if any(ranks):
        raise SystemExit("出现结构退化的纯 ZBot 连接，需要先停下来检查模型")

    print("概念搜索：固定 6 驱动关节，长度和连接一起变。结构退化的链不进可行前沿。")
    concept = run_search(
        spec_from_dict(
            {
                "mode": "mixed",
                "count_min": 6,
                "count_max": 6,
                "count_lock": 6,
                "population": args.concept_pop,
                "generations": args.concept_gen,
                "objectives": ["coverage", "manipulability", "velocity"],
                "seed": args.seed,
                "length_lower_m": 0.08,
                "length_upper_m": 0.16,
                "task": {
                    "center_m": tight.center_m,
                    "half_m": tight.half_m,
                    "payload_kg": tight.payload_kg,
                    "effort_limit_nm": tight.effort_limit_nm,
                    "velocity_limit_rad_s": tight.velocity_limit_rad_s,
                    "sample_count": 0,
                    "seed": args.seed,
                    "grid_divisions": 3,
                    "position_tolerance_m": 0.01,
                    "ik_starts": 4,
                    "ik_iterations": 20,
                    "reject_structural_degeneracy": True,
                },
            }
        ),
        start=baseline_design(),
    )
    conceptual = []
    for row in concept["feasible_pareto"]:
        design = design_from_dict(row["design"])
        if actuation_ok(design) and not row["metrics"].get("structurally_degenerate"):
            if all(item["id"] == "zbot_r3" for item in row["design"]["modules"]):
                continue
            rank = generic_rank_report(design, samples=12, seed=args.seed + 4)
            if rank["structurally_degenerate"]:
                print("丢弃结构退化概念链", [item["id"] for item in row["design"]["modules"]])
                continue
            conceptual.append({"family": "conceptual", "design": row["design"], "metrics": row["metrics"], "generic_rank": rank})
    conceptual.sort(key=lambda row: (-row["metrics"]["task_feasible_fraction"], -(row["metrics"]["task_manipulability_median"] or -1)))

    ordered = sorted(physical, key=lambda row: (-row["metrics"]["task_feasible_fraction"], -(row["metrics"]["task_manipulability_median"] or -1), row["label"]))
    shortlist = []
    for label in ("00000", ordered[0]["label"], "11000"):
        row = next(item for item in physical if item["label"] == label)
        if all(item["label"] != label for item in shortlist):
            shortlist.append(row)
    if conceptual:
        shortlist.append(conceptual[0])

    strict = example_task(position_tolerance_m=0.01, grid_divisions=5, ik_starts=8, ik_iterations=50, seed=args.seed)
    loose = example_task(position_tolerance_m=0.05, grid_divisions=5, ik_starts=8, ik_iterations=50, seed=args.seed)
    shorter = example_task(position_tolerance_m=0.01, grid_divisions=5, ik_starts=8, ik_iterations=30, seed=args.seed)
    alternate = cell_centers(strict, 5)
    reviews = []
    for row in shortlist:
        design = design_from_dict(row["design"])
        label = row["design"].get("label") or row.get("label") or "concept"
        item = {"label": label, "family": row["family"], "design": row["design"]}
        for name, budget, targets in (
            ("tol_0.01_125_iter50", strict, None),
            ("tol_0.05_125_iter50", loose, None),
            ("tol_0.01_125_iter30", shorter, None),
            ("tol_0.01_cell_centers_iter50", strict, alternate),
        ):
            metrics = evaluate_design(
                design,
                budget,
                targets=targets,
                target_set="cell_centers" if targets is not None else "linspace_endpoints",
            )
            item[name] = metrics
            print(f"{name} {row['family']} {label} {brief_metrics(metrics)}")
        item["generic_rank"] = generic_rank_report(design, samples=16, seed=args.seed)
        reviews.append(item)

    baseline = next(item for item in reviews if item["label"] == "00000")
    physical_reviews = [item for item in reviews if item["family"] == "physical" and item["label"] != "00000"]
    concept_reviews = [item for item in reviews if item["family"] == "conceptual"]

    def gain(item: dict, key: str = "tol_0.01_125_iter50") -> float:
        return float(item[key]["task_feasible_fraction"]) - float(baseline[key]["task_feasible_fraction"])

    credible = []
    for item in physical_reviews:
        primary = gain(item)
        other = gain(item, "tol_0.01_cell_centers_iter50")
        manip_ok = ratio(item, baseline, "task_manipulability_median") >= 0.9
        speed_ok = ratio(item, baseline, "task_velocity_median") >= 0.9
        stable = primary >= 0.03 and other >= 0.03 and manip_ok and speed_ok
        item["credible_improvement"] = stable
        if stable:
            credible.append(item)
    elapsed = time.perf_counter() - started
    payload = {
        "task_origin": "默认示例任务，不是用户给定工况，也不是通用最优。",
        "elapsed_s": elapsed,
        "seed": args.seed,
        "physical_tight_screen": physical,
        "concept_search_evaluations": concept["evaluations"],
        "concept_elapsed_s": concept["elapsed_s"],
        "conceptual_rank6": conceptual,
        "reviews": reviews,
        "conclusion": {
            "all_32_zbot_spatially_full_rank_on_screen": not any(ranks),
            "credible_physical_improvement": [item["label"] for item in credible],
            "baseline_remains_best_physical": not credible,
            "note": (
                "可信改进要求 1 cm、125 点主格点和另一套单元中心点都至少高 0.03，"
                "且可操作度与速度都不低于基准的 90%。"
                "概念链若一般位形六维秩退化，不在此列。"
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for item in reviews:
        design = design_from_dict(item["design"])
        safe = str(item["label"]).replace(" ", "_")
        path = args.out.parent / "dof6_refine_mjcf" / f"{item['family']}_{safe}.xml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(export_mjcf(design), encoding="utf-8")
    print(f"用时 {elapsed:.1f} s，已写入 {args.out}")
    print("结论", payload["conclusion"])


def actuation_ok(design: Design) -> bool:
    return actuation_report(design)["driven_joint_count"] == 6


def ratio(item: dict, baseline: dict, key: str) -> float:
    value = item["tol_0.01_125_iter50"][key]
    base = baseline["tol_0.01_125_iter50"][key]
    if value is None or not base:
        return 0.0
    return float(value) / float(base)


def brief_metrics(metrics: dict) -> str:
    return (
        f"{metrics['task_targets_reached']}/{metrics['task_target_count']} "
        f"可操作度 {metrics['task_manipulability_median']} 速度 {metrics['task_velocity_median']} "
        f"六维秩不足 {metrics['task_pose_spatial_rank_deficient_fraction']}"
    )


if __name__ == "__main__":
    main()
