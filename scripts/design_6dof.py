"""固定 6 个驱动关节的构型搜索，并做更密格点的独立复核。

实物候选只来自六个 ZBot R3 模块。相邻接口只能是 0 或 π/2，一共 32 条，全部评价。
概念候选把轴向概念连杆插进同样的 6 关节链，长度是假设，不能当成现有 ZBot 的减重。

默认任务盒没有用户给定的工况来源。

用法::

    python scripts/design_6dof.py --out experiments/dof6_search.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401

from zbot_exolimb_grasp.platform.exporting import candidates_csv, export_mjcf
from zbot_exolimb_grasp.platform.model import actuation_report, baseline_design, evaluate_design, zbot_chain
from zbot_exolimb_grasp.platform.schema import TaskSpec
from zbot_exolimb_grasp.platform.search import design_from_dict, run_search, spec_from_dict


def screen_task() -> TaskSpec:
    return TaskSpec(
        center_m=(0.0, 0.05, 0.72),
        half_m=(0.22, 0.22, 0.22),
        payload_kg=0.4,
        effort_limit_nm=20.0,
        velocity_limit_rad_s=1.5,
        sample_count=0,
        seed=1,
        grid_divisions=3,
        position_tolerance_m=0.05,
        ik_starts=4,
        ik_iterations=12,
    )


def review_task(kind: str) -> TaskSpec:
    task = screen_task()
    if kind == "dense":
        task.grid_divisions = 4
        task.ik_starts = 8
        task.ik_iterations = 16
    elif kind == "more_ik":
        task.grid_divisions = 3
        task.ik_starts = 8
        task.ik_iterations = 16
    else:
        raise ValueError(kind)
    return task


def bits_of(twists: list[float] | tuple[float, ...]) -> str:
    return "".join("1" if value > 0.1 else "0" for value in twists)


def public_row(design_dict: dict, metrics: dict, family: str) -> dict:
    return {"family": family, "design": design_dict, "metrics": metrics, "violation": 0.0}


def summarize(metrics: dict) -> str:
    manip = "无效" if metrics["task_manipulability_median"] is None else f"{metrics['task_manipulability_median']:.5f}"
    speed = "无效" if metrics["task_velocity_median"] is None else f"{metrics['task_velocity_median']:.4f}"
    margin = "无效" if metrics["task_torque_margin_median_nm"] is None else f"{metrics['task_torque_margin_median_nm']:.2f}"
    rank = "无效" if metrics["position_rank_deficient_fraction"] is None else f"{metrics['position_rank_deficient_fraction']:.2f}"
    return (
        f"驱动关节 {metrics['driven_joint_count']}，被动 a 连杆 {metrics['passive_body_count']}，"
        f"质量 {metrics['mass_kg']:.3f} kg，格点 {metrics['task_targets_reached']}/{metrics['task_target_count']}"
        f" = {metrics['task_feasible_fraction']:.3f}，可操作度 {manip}，速度 {speed} m/s，"
        f"静载裕度 {margin} N·m，平移秩不足比例 {rank}"
    )


def better(left: dict, right: dict) -> bool:
    """复核时先看可行格点，再看任务可操作度，再看速度。质量不参与实物之间的名次。"""
    keys = (
        ("task_feasible_fraction", True),
        ("task_manipulability_median", True),
        ("task_velocity_median", True),
    )
    for key, higher in keys:
        a = left["metrics"][key]
        b = right["metrics"][key]
        a = -1.0 if a is None else float(a)
        b = -1.0 if b is None else float(b)
        if abs(a - b) > 1e-9:
            return a > b if higher else a < b
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="固定 6 驱动关节的实物枚举、概念搜索和独立复核")
    parser.add_argument("--out", type=Path, default=Path("experiments/dof6_search.json"))
    parser.add_argument("--concept-pop", type=int, default=12)
    parser.add_argument("--concept-gen", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    task = screen_task()
    task.seed = args.seed
    print(
        "默认示例任务，不是用户工况：中心 "
        f"{task.center_m} m，半尺寸 {task.half_m} m，载荷 {task.payload_kg} kg，"
        f"力矩 {task.effort_limit_nm} N·m，速度假设 {task.velocity_limit_rad_s} rad/s，"
        f"容差 {task.position_tolerance_m} m"
    )
    print(f"筛选预算：每轴 {task.grid_divisions} 份格点，IK 起点 {task.ik_starts}，迭代 {task.ik_iterations}")

    physical = []
    for mask in range(32):
        twists = tuple((math.pi / 2 if mask & (1 << index) else 0.0) for index in range(5))
        design = zbot_chain(6, twists)
        report = actuation_report(design)
        if report["driven_joint_count"] != 6:
            raise SystemExit("实物枚举出现了不是 6 驱动关节的设计")
        metrics = evaluate_design(design, task)
        physical.append(public_row(
            {
                "topology": "serial",
                "base_position_m": list(design.base_position_m),
                "modules": [
                    {
                        "id": "zbot_r3",
                        "kind": "physical",
                        "parameters": {},
                        "twist_to_next_rad": None if index == 5 else twists[index],
                    }
                    for index in range(6)
                ],
                "bits": bits_of(twists),
            },
            metrics,
            "physical",
        ))
        print(f"实物 {bits_of(twists)} {summarize(metrics)}")

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
                "task": {
                    "center_m": task.center_m,
                    "half_m": task.half_m,
                    "payload_kg": task.payload_kg,
                    "effort_limit_nm": task.effort_limit_nm,
                    "velocity_limit_rad_s": task.velocity_limit_rad_s,
                    "sample_count": 0,
                    "seed": args.seed,
                    "grid_divisions": task.grid_divisions,
                    "position_tolerance_m": task.position_tolerance_m,
                    "ik_starts": task.ik_starts,
                    "ik_iterations": task.ik_iterations,
                },
            }
        ),
        start=baseline_design(),
    )
    conceptual_rows = []
    for row in concept["feasible_pareto"]:
        if actuation_report(design_from_dict(row["design"]))["driven_joint_count"] != 6:
            continue
        if all(item["id"] == "zbot_r3" for item in row["design"]["modules"]):
            continue
        conceptual_rows.append(public_row(row["design"], row["metrics"], "conceptual"))
    conceptual_rows.sort(key=lambda row: (
        -float(row["metrics"]["task_feasible_fraction"]),
        -(row["metrics"]["task_manipulability_median"] or -1),
    ))

    def pick_physical(limit: int) -> list[dict]:
        ordered = sorted(physical, key=lambda row: (
            -float(row["metrics"]["task_feasible_fraction"]),
            -(row["metrics"]["task_manipulability_median"] or -1),
            -(row["metrics"]["task_velocity_median"] or -1),
            row["design"]["bits"],
        ))
        chosen = []
        seen = set()
        for row in ordered:
            if row["design"]["bits"] in seen:
                continue
            chosen.append(row)
            seen.add(row["design"]["bits"])
            if len(chosen) >= limit:
                break
        return chosen

    shortlist = []
    for row in pick_physical(3):
        shortlist.append(("physical", row))
    if not any(item[1]["design"].get("bits") == "00000" for item in shortlist):
        baseline_row = next(row for row in physical if row["design"]["bits"] == "00000")
        shortlist.append(("physical", baseline_row))
    for row in conceptual_rows[:2]:
        shortlist.append(("conceptual", row))

    def rerun(kind: str) -> list[dict]:
        budget = review_task(kind)
        budget.seed = args.seed
        rows = []
        for family, row in shortlist:
            design = design_from_dict(row["design"])
            metrics = evaluate_design(design, budget)
            label = row["design"].get("bits") or " ".join(item["id"] for item in row["design"]["modules"])
            print(f"复核 {kind} {family} {label} {summarize(metrics)}")
            rows.append({"family": family, "label": label, "design": row["design"], "metrics": metrics, "budget": kind})
        return rows

    print("独立复核：更密格点，不是把筛选结果再打印一次")
    dense = rerun("dense")
    print("敏感性：同一 27 点格点，但 IK 起点和迭代多于筛选")
    more_ik = rerun("more_ik")

    def family_best(rows: list[dict], family: str) -> dict | None:
        pool = [row for row in rows if row["family"] == family]
        if not pool:
            return None
        return max(pool, key=lambda row: (
            float(row["metrics"]["task_feasible_fraction"]),
            row["metrics"]["task_manipulability_median"] or -1,
            row["metrics"]["task_velocity_median"] or -1,
        ))

    baseline_dense = next(row for row in dense if row["label"] == "00000")
    physical_dense = [row for row in dense if row["family"] == "physical"]
    physical_winner = max(physical_dense, key=lambda row: (
        float(row["metrics"]["task_feasible_fraction"]),
        row["metrics"]["task_manipulability_median"] or -1,
        row["metrics"]["task_velocity_median"] or -1,
    ))
    concept_winner = family_best(dense, "conceptual")

    def ratio(row: dict, key: str) -> float:
        value = row["metrics"][key]
        base = baseline_dense["metrics"][key]
        if value is None or base in (None, 0):
            return 0.0
        return float(value) / float(base)

    gain = float(physical_winner["metrics"]["task_feasible_fraction"]) - float(
        baseline_dense["metrics"]["task_feasible_fraction"]
    )
    clear_win = (
        physical_winner["label"] != "00000"
        and gain >= 0.03
        and ratio(physical_winner, "task_manipulability_median") >= 0.9
        and ratio(physical_winner, "task_velocity_median") >= 0.9
    )
    print(
        "密网格复核后，实物"
        + ("有方位在格点、可操作度和速度上同时明显优于全 0 基准。" if clear_win else "没有方位同时明显优于全 0 基准。")
    )

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_origin": "默认示例任务，不是用户给定工况，也不是通用最优。",
        "task": {
            "center_m": list(task.center_m),
            "half_m": list(task.half_m),
            "payload_kg": task.payload_kg,
            "effort_limit_nm": task.effort_limit_nm,
            "velocity_limit_rad_s": task.velocity_limit_rad_s,
            "position_tolerance_m": task.position_tolerance_m,
        },
        "screen_budget": {
            "grid_divisions": task.grid_divisions,
            "ik_starts": task.ik_starts,
            "ik_iterations": task.ik_iterations,
            "physical_count": 32,
        },
        "physical_screen": physical,
        "conceptual_search": {
            "seed": args.seed,
            "population": args.concept_pop,
            "generations": args.concept_gen,
            "note": "概念连杆质量按 4 kg/m 假设，不能解释成现有 ZBot 减重。驱动关节仍是 6 个。",
            "rows": conceptual_rows,
        },
        "independent_review": {"dense": dense, "more_ik": more_ik},
        "conclusion": {
            "physical_beats_baseline_on_dense_grid": clear_win,
            "coverage_gain_vs_baseline": gain,
            "note": "明显优于要求格点比例至少高 0.03，且可操作度和速度都不低于基准的 90%。差一个 64 格点不算明显改进。",
            "baseline_bits": "00000",
            "physical_winner_bits": physical_winner["label"],
            "conceptual_winner": None if concept_winner is None else concept_winner["label"],
        },
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    table = {
        "baseline": {"design": baseline_dense["design"], "metrics": baseline_dense["metrics"]},
        "feasible_pareto": [
            {"design": row["design"], "metrics": row["metrics"], "violation": 0.0}
            for row in dense
        ],
    }
    out.with_suffix(".csv").write_text(candidates_csv(table), encoding="utf-8")
    mjcf_dir = out.parent / "dof6_mjcf"
    mjcf_dir.mkdir(parents=True, exist_ok=True)
    for row in dense:
        design = design_from_dict(row["design"])
        safe = row["label"].replace(" ", "_")
        (mjcf_dir / f"{row['family']}_{safe}.xml").write_text(export_mjcf(design), encoding="utf-8")
    print(f"已写入 {out}")


if __name__ == "__main__":
    main()
