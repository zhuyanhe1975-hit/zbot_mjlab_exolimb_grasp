"""批次 2：足底任务点重核 6/7/8 实物，并在固定驱动数下搜索概念模块。

旧的 foot_1 关节原点结果留在批次 1，不拿来当足底分数。
缓存键包含工具点、模型提交和 site 变换。重复运行会跳过已完成的键。

用法::

    python scripts/design_678_batch2.py
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

from zbot_exolimb_grasp.platform.exporting import export_mjcf
from zbot_exolimb_grasp.platform.model import evaluate_design, generic_rank_report, sole_site, zbot_chain
from zbot_exolimb_grasp.platform.schema import SearchSpec, TaskSpec
from zbot_exolimb_grasp.platform.search import design_from_dict, design_to_dict, run_search

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "experiments" / "cache" / "longterm_batch2_screen.json"
REVIEW_CACHE = ROOT / "experiments" / "cache" / "longterm_batch2_review.json"
CONCEPT_CACHE = ROOT / "experiments" / "cache" / "longterm_batch2_concept.json"
STATE = ROOT / "experiments" / "longterm_search_state.json"
OUT = ROOT / "experiments" / "longterm_batch2.json"
PUBLIC = ROOT / "web" / "public" / "longterm_batch2.json"
SCREEN_VERSION = "batch2-sole-screen-v1"
REVIEW_VERSION = "batch2-sole-review-v1"
CONCEPT_VERSION = "batch2-sole-concept-v1"

SCREEN = dict(grid_divisions=3, position_tolerance_m=0.02, ik_starts=3, ik_iterations=10, seed=11, tool_point="foot_1_sole")
REVIEW = dict(
    grid_divisions=5,
    position_tolerance_m=0.01,
    ik_starts=8,
    ik_iterations=50,
    seed=11,
    compute_velocity_lp=True,
    tool_point="foot_1_sole",
)
SEARCH_TASK = dict(
    grid_divisions=3,
    position_tolerance_m=0.02,
    ik_starts=3,
    ik_iterations=10,
    seed=11,
    tool_point="foot_1_sole",
    reject_structural_degeneracy=True,
)


def sole_identity() -> str:
    site = sole_site()
    pos = ",".join(f"{value:.10g}" for value in site.pos_m)
    quat = ",".join(f"{value:.10g}" for value in site.quat_wxyz)
    return f"{site.source_commit}:{site.name}@{site.body}:{pos}:{quat}"


def task_of(spec: dict, **overrides) -> TaskSpec:
    merged = dict(spec)
    merged.update(overrides)
    return TaskSpec(
        center_m=(0.0, 0.05, 0.72),
        half_m=(0.22, 0.22, 0.22),
        payload_kg=0.4,
        effort_limit_nm=20.0,
        velocity_limit_rad_s=1.5,
        sample_count=0,
        **merged,
    )


def cell_centers(divisions: int = 5) -> np.ndarray:
    center = (0.0, 0.05, 0.72)
    half = (0.22, 0.22, 0.22)
    axes = []
    for value, width in zip(center, half, strict=True):
        step = 2.0 * width / divisions
        axes.append([value - width + step * (index + 0.5) for index in range(divisions)])
    return np.asarray([(x, y, z) for x in axes[0] for y in axes[1] for z in axes[2]], dtype=float)


def bits_of(twists) -> str:
    return "".join("1" if value > 0.1 else "0" for value in twists)


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def compact(metrics: dict) -> dict:
    keys = [
        "driven_joint_count",
        "passive_body_count",
        "mass_kg",
        "tool_point",
        "tool_point_applied",
        "tool_site",
        "distal_linear_jacobian_norm",
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
        "structurally_degenerate",
    ]
    return {key: metrics.get(key) for key in keys}


def screen_physical(identity: str) -> list[dict]:
    cache = load_json(CACHE)
    budget = task_of(SCREEN)
    rows = []
    for count, total in ((6, 32), (7, 64), (8, 128)):
        if total != 1 << (count - 1):
            raise SystemExit("0/π/2 枚举数量和接口规则不一致")
        for mask in range(total):
            twists = tuple((math.pi / 2 if mask & (1 << index) else 0.0) for index in range(count - 1))
            label = bits_of(twists)
            key = f"{SCREEN_VERSION}:{identity}:{count}:{label}"
            if key not in cache:
                metrics = compact(evaluate_design(zbot_chain(count, twists), budget))
                cache[key] = metrics
                if mask % 16 == 0:
                    save_json(CACHE, cache)
                    print(f"足底初筛 {count} {label} {metrics['task_targets_reached']}/{metrics['task_target_count']}", flush=True)
            rows.append({"count": count, "bits": label, "kind": "physical", "metrics": cache[key]})
    save_json(CACHE, cache)
    return rows


def write_screen_csv(rows: list[dict]) -> None:
    path = ROOT / "experiments" / "longterm_batch2_screen.csv"
    header = "count,bits,mass_kg,reached,targets,fraction,manipulability,velocity_pseudoinverse,distal_jacobian_norm,q0_collision\n"
    lines = [header]
    for row in rows:
        metrics = row["metrics"]
        lines.append(
            f"{row['count']},{row['bits']},{metrics['mass_kg']},{metrics['task_targets_reached']},"
            f"{metrics['task_target_count']},{metrics['task_feasible_fraction']},{metrics['task_manipulability_median']},"
            f"{metrics['task_velocity_median']},{metrics['distal_linear_jacobian_norm']},{metrics['q0_collision']}\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def pick_physical(rows: list[dict]) -> list[tuple[int, str, str]]:
    chosen: list[tuple[int, str, str]] = []

    def add(count: int, label: str, reason: str) -> None:
        if (count, label) not in {(item[0], item[1]) for item in chosen}:
            chosen.append((count, label, reason))

    for count in (6, 7, 8):
        group = [row for row in rows if row["count"] == count]
        add(count, "0" * (count - 1), "all_zero")
        best = max(group, key=lambda row: (row["metrics"]["task_feasible_fraction"], row["metrics"]["task_velocity_median"] or -1))
        if best["bits"] != "0" * (count - 1):
            add(count, best["bits"], "screen_coverage")
        floor = best["metrics"]["task_feasible_fraction"] - 0.08
        speedy = max(group, key=lambda row: row["metrics"]["task_velocity_median"] or -1)
        if speedy["metrics"]["task_feasible_fraction"] >= floor:
            add(count, speedy["bits"], "screen_speed")
    return chosen


def review_design(design, label: str, reason: str, kind: str, cache: dict) -> dict:
    identity = sole_identity()
    key = f"{REVIEW_VERSION}:{identity}:{kind}:{label}"
    if key in cache:
        row = cache[key]
        row["role"] = reason
        print(f"复核缓存 {label}", flush=True)
        return row
    strict = task_of(REVIEW)
    started = time.perf_counter()
    endpoint = evaluate_design(design, strict)
    centers = evaluate_design(design, strict, targets=cell_centers(5), target_set="cell_centers")
    rank = generic_rank_report(design, samples=8, seed=11, tool_point="foot_1_sole")
    view = compact(endpoint)
    view["example_q"] = endpoint["example_q"]
    modules = [module_id for module_id in design.module_ids]
    assumed = any(module_id != "zbot_r3" for module_id in modules)
    if assumed:
        view["mass_is_assumption"] = True
    row = {
        "role": reason,
        "kind": kind,
        "label": label,
        "count": len(modules),
        "bits": bits_of(design.twists_rad),
        "modules": modules,
        "mass_kg": endpoint["mass_kg"],
        "mass_is_assumption": assumed,
        "design": design_to_dict(design),
        "endpoint_125": view,
        "cell_centers_125": compact(centers),
        "generic_rank_degenerate": bool(rank["structurally_degenerate"]),
        "elapsed_s": time.perf_counter() - started,
    }
    cache[key] = row
    save_json(REVIEW_CACHE, cache)
    print(
        f"复核 {kind} {label}: 端点 {endpoint['task_targets_reached']}/125，"
        f"中心 {centers['task_targets_reached']}/125，"
        f"伪逆 {endpoint['task_velocity_median']}，轴 LP {endpoint['task_velocity_lp_median_m_s']}",
        flush=True,
    )
    return row


def review_physical(chosen: list[tuple[int, str, str]]) -> list[dict]:
    cache = load_json(REVIEW_CACHE)
    rows = []
    for count, label, reason in chosen:
        twists = tuple((math.pi / 2 if bit == "1" else 0.0) for bit in label)
        rows.append(review_design(zbot_chain(count, twists), f"zbot{count}:{label}", reason, "physical", cache))
    return rows


def concept_search() -> list[dict]:
    cache = load_json(CONCEPT_CACHE)
    identity = sole_identity()
    found = []
    for count in (6, 7, 8):
        key = f"{CONCEPT_VERSION}:{identity}:{count}"
        if key not in cache:
            spec = SearchSpec(
                mode="mixed",
                count_min=count,
                count_max=count,
                count_lock=count,
                population=16,
                generations=4,
                objectives=("coverage", "velocity", "mass"),
                seed=11,
                distal_module_id="zbot_r3",
                task=task_of(SEARCH_TASK),
            )
            result = run_search(spec, zbot_chain(count))
            cache[key] = {
                "evaluations": result["evaluations"],
                "elapsed_s": result["elapsed_s"],
                "infeasible_in_population": result["infeasible_in_population"],
                "pareto": result["feasible_pareto"],
            }
            save_json(CONCEPT_CACHE, cache)
            print(f"概念搜索 {count} 驱动：评价 {result['evaluations']}，前沿 {len(result['feasible_pareto'])}", flush=True)
        found.append({"count": count, **cache[key]})
    return found


def pick_concept(searches: list[dict]) -> list[dict]:
    chosen = []
    for search in searches:
        rows = []
        for item in search["pareto"]:
            metrics = item["metrics"]
            if metrics.get("tool_point_applied") != "foot_1_sole":
                continue
            if not metrics.get("task_manipulability_valid") and metrics.get("task_feasible_fraction", 0) <= 0:
                continue
            rows.append(item)
        if not rows:
            continue
        best = max(rows, key=lambda item: item["metrics"]["task_feasible_fraction"])
        floor = best["metrics"]["task_feasible_fraction"] - 0.08
        eligible = [item for item in rows if item["metrics"]["task_feasible_fraction"] >= floor]
        speedy = max(eligible, key=lambda item: item["metrics"].get("task_velocity_median") or -1)
        light = min(eligible, key=lambda item: item["metrics"]["mass_kg"])
        seen = set()
        for item, reason in ((best, "concept_coverage"), (speedy, "concept_speed"), (light, "concept_light")):
            signature = json.dumps(item["design"], sort_keys=True)
            if signature in seen:
                continue
            seen.add(signature)
            chosen.append({"reason": reason, "count": search["count"], "item": item})
    return chosen


def review_concept(chosen: list[dict]) -> list[dict]:
    cache = load_json(REVIEW_CACHE)
    rows = []
    for index, pick in enumerate(chosen):
        design = design_from_dict(pick["item"]["design"])
        label = f"concept{pick['count']}:{index}:{pick['reason']}"
        rows.append(review_design(design, label, pick["reason"], "concept", cache))
    return rows


def iteration_check(rows: list[dict]) -> list[dict]:
    cache = load_json(REVIEW_CACHE)
    identity = sole_identity()
    checked = []
    for row in rows:
        if row["kind"] != "physical" or row["role"] != "all_zero":
            continue
        key = f"{REVIEW_VERSION}:{identity}:iter30:{row['label']}"
        if key not in cache:
            design = zbot_chain(row["count"], tuple(0.0 for _ in range(row["count"] - 1)))
            metrics = evaluate_design(design, task_of(REVIEW, ik_iterations=30, compute_velocity_lp=False))
            cache[key] = {"reached": metrics["task_targets_reached"], "fraction": metrics["task_feasible_fraction"]}
            save_json(REVIEW_CACHE, cache)
        reached = cache[key]["reached"]
        checked.append(
            {
                "label": row["label"],
                "iter50": row["endpoint_125"]["task_targets_reached"],
                "iter30": reached,
                "same_count": reached == row["endpoint_125"]["task_targets_reached"],
            }
        )
        print(f"迭代核对 {row['label']}: 30 次 {reached}/125，50 次 {row['endpoint_125']['task_targets_reached']}/125", flush=True)
    return checked


def gains(row: dict, baseline: dict) -> tuple[float, float]:
    return (
        row["endpoint_125"]["task_feasible_fraction"] - baseline["endpoint_125"]["task_feasible_fraction"],
        row["cell_centers_125"]["task_feasible_fraction"] - baseline["cell_centers_125"]["task_feasible_fraction"],
    )


def goal_block(candidates: list[dict], baseline: dict, *, concept: bool) -> dict:
    if concept:
        candidates = [row for row in candidates if row["mass_is_assumption"]]
    if not candidates:
        return {"available": False, "why": "没有带概念模块的复核候选。混合种群里的全实物链不算概念改进。"}
    coverage = max(candidates, key=lambda row: min(gains(row, baseline)))
    endpoint_gain, center_gain = gains(coverage, baseline)
    reliable_coverage = endpoint_gain >= 0.03 and center_gain >= 0.03
    speed = max(candidates, key=lambda row: row["endpoint_125"]["task_velocity_lp_median_m_s"] or -1)
    base_speed = baseline["endpoint_125"]["task_velocity_lp_median_m_s"] or 0.0
    speed_value = speed["endpoint_125"]["task_velocity_lp_median_m_s"] or 0.0
    speed_gains = gains(speed, baseline)
    speed_ok = speed_value > base_speed * 1.05 and speed_gains[0] >= -0.03 and speed_gains[1] >= -0.03
    light = min(candidates, key=lambda row: row["mass_kg"])
    light_gains = gains(light, baseline)
    light_ok = concept and light_gains[0] >= -0.03 and light_gains[1] >= -0.03 and light["mass_kg"] < baseline["mass_kg"] - 0.05
    prefix = "概念质量来自 4 kg/m 假设，不是 ZBot 实测减重。" if concept else "实物尺寸锁定，质量只随模块数增加。"
    return {
        "available": True,
        "mass_is_assumption": concept,
        "coverage_priority": {
            "label": coverage["label"] if reliable_coverage else baseline["label"],
            "count": coverage["count"] if reliable_coverage else baseline["count"],
            "bits": coverage["bits"] if reliable_coverage else baseline["bits"],
            "modules": coverage["modules"] if reliable_coverage else baseline["modules"],
            "reliable_improvement_on_both_point_sets": reliable_coverage,
            "endpoint_gain": endpoint_gain,
            "cell_center_gain": center_gain,
            "why": "可靠覆盖改进要在端点和单元中心两套 125 点上都至少高 0.03。只在一套上变多，记成点集取舍。",
        },
        "speed_priority": {
            "label": speed["label"] if speed_ok else baseline["label"],
            "count": speed["count"] if speed_ok else baseline["count"],
            "bits": speed["bits"] if speed_ok else baseline["bits"],
            "modules": speed["modules"] if speed_ok else baseline["modules"],
            "reliable_improvement": speed_ok,
            "endpoint_lp_m_s": speed_value,
            "baseline_lp_m_s": base_speed,
            "why": "速度用六个坐标轴方向的线性规划最小值，不是全方向最快，也不是伪逆。更长的臂在 1.5 rad/s 假设下末端更快，这是力臂尺度。",
        },
        "light_priority": {
            "label": light["label"] if light_ok else baseline["label"],
            "count": light["count"] if light_ok else baseline["count"],
            "bits": light["bits"] if light_ok else baseline["bits"],
            "modules": light["modules"] if light_ok else baseline["modules"],
            "mass_kg": light["mass_kg"] if light_ok else baseline["mass_kg"],
            "mass_is_assumption": bool(light_ok),
            "reliable_improvement": light_ok,
            "why": prefix + "更轻的概念链还要在两套点上比实物基准少的可行比例不超过 0.03，才进入轻量取舍。",
        },
    }


def summarize(row: dict) -> dict:
    return {
        "label": row["label"],
        "kind": row["kind"],
        "role": row["role"],
        "count": row["count"],
        "bits": row["bits"],
        "modules": row["modules"],
        "mass_kg": row["mass_kg"],
        "mass_is_assumption": row["mass_is_assumption"],
        "endpoint_reached": row["endpoint_125"]["task_targets_reached"],
        "centers_reached": row["cell_centers_125"]["task_targets_reached"],
        "manipulability": row["endpoint_125"]["task_manipulability_median"],
        "velocity_pseudoinverse": row["endpoint_125"]["task_velocity_median"],
        "velocity_lp": row["endpoint_125"]["task_velocity_lp_median_m_s"],
        "torque_margin_nm": row["endpoint_125"]["task_torque_margin_median_nm"],
        "q0_collision": row["endpoint_125"]["q0_collision"],
        "distal_linear_jacobian_norm": row["endpoint_125"]["distal_linear_jacobian_norm"],
        "spatial_rank_deficient_on_task_solutions": row["endpoint_125"]["task_pose_spatial_rank_deficient_fraction"],
        "generic_rank_degenerate": row["generic_rank_degenerate"],
    }


def main() -> None:
    started = time.perf_counter()
    identity = sole_identity()
    site = sole_site()
    print("足底", site.name, site.pos_m, site.quat_wxyz, flush=True)
    rows = screen_physical(identity)
    write_screen_csv(rows)
    chosen = pick_physical(rows)
    print("实物复核", chosen, flush=True)
    physical = review_physical(chosen)
    searches = concept_search()
    concept_picks = pick_concept(searches)
    concept = review_concept(concept_picks)
    checked = iteration_check(physical)
    baseline = next(row for row in physical if row["count"] == 6 and row["role"] == "all_zero")
    physical_goals = goal_block([row for row in physical if row["role"] == "all_zero" or row["kind"] == "physical"], baseline, concept=False)
    concept_goals = goal_block(concept, baseline, concept=True)
    stable = bool(checked) and all(item["same_count"] for item in checked)
    elapsed = time.perf_counter() - started
    previous = load_json(STATE)
    batch1 = previous.get("batch1") if previous.get("checkpoint_version") == 2 else previous
    batch1_elapsed = float((batch1 or {}).get("elapsed_s") or 0.0)
    prior_batch2 = float((previous.get("batch2") or {}).get("elapsed_s") or 0.0) if previous.get("checkpoint_version") == 2 else 0.0
    batch2_elapsed = elapsed if prior_batch2 == 0.0 else prior_batch2
    site_record = {
        "name": site.name,
        "body": site.body,
        "pos_m": list(site.pos_m),
        "quat_wxyz": list(site.quat_wxyz),
        "source_commit": site.source_commit,
        "source_path": site.source_path,
        "identity": identity,
    }
    suggest_close = stable
    state = {
        "checkpoint_version": 2,
        "status": "batch2_complete",
        "completed_batches": ["batch1_physical_6_7_8", "batch2_sole_and_concept"],
        "elapsed_s": batch1_elapsed + batch2_elapsed,
        "batch1": batch1,
        "batch2": {
            "batch_id": "batch2_sole_and_concept",
            "status": "batch2_complete",
            "elapsed_s": batch2_elapsed,
            "tool_point": site_record,
            "legacy_tool_point": "foot_joint_origin",
            "legacy_results": "保留在 batch1，不与足底分数混用。",
            "task": {
                "origin": "默认示例任务，不是用户工况。",
                "center_m": [0.0, 0.05, 0.72],
                "half_m": [0.22, 0.22, 0.22],
                "payload_kg": 0.4,
                "effort_limit_nm": 20.0,
                "velocity_limit_rad_s": 1.5,
                "velocity_limit_is_assumption": True,
                "position_tolerance_m": 0.01,
                "point_sets": ["125 linspace endpoints", "125 cell centers"],
                "ik_starts": 8,
                "ik_iterations": 50,
                "seed": 11,
            },
            "budgets": {"screen": SCREEN, "review": REVIEW, "concept_search": {"population": 16, "generations": 4, "task": SEARCH_TASK}},
            "screen_rows": len(rows),
            "physical_reviewed": [summarize(row) for row in physical],
            "concept_reviewed": [summarize(row) for row in concept],
            "concept_search": [
                {"count": item["count"], "evaluations": item["evaluations"], "elapsed_s": item["elapsed_s"], "pareto": len(item["pareto"])}
                for item in searches
            ],
            "iteration_check": checked,
            "recommendations": {"physical_sole": physical_goals, "concept_assumed_mass": concept_goals, "unique_optimum": False},
            "verified": [
                "foot_1_sole 的名称、局部位置和四元数来自 ZBOT6.xml，不是手写 4 cm。",
                "足底点的平移和转动雅可比与有限差分一致，MJCF 末端使用同一 site。",
                "末关节平移列与轴叉乘工具偏置一致；关节原点模式下该列为 0。",
                "0/π/2 实物序列在足底点下重新初筛，缓存键含模型提交和 site 变换。",
            ],
            "unverified_assumptions": [
                "任务盒不是用户工况。",
                "关节速度 1.5 rad/s 没有实测。轴线性规划只覆盖六个坐标轴方向。",
                "胶囊碰撞和质点静力矩是近似。",
                "未解出的格点不是不可达证明。",
                "概念线密度 4 kg/m、胶囊半径 0.025 m 是假设。",
                "六维秩只诊断瞬时运动，不是六维姿态任务。",
                "概念搜索是种群 16、4 代的抽样，不是长度和接口的完全枚举。",
            ],
            "suggest_close": suggest_close,
            "stop_criteria": [
                "足底任务点上，候选须在两套 125 点都比 6 模块全 0 高至少 0.03，才称覆盖改进。",
                "速度改进用轴线性规划，并且两套点的可行比例下降不超过 0.03。",
                "概念质量不写成硬件减重。",
                "30 次与 50 次迭代的端点可行数一致，才把 IK 预算当成稳定。",
            ],
        },
        "recommendations_by_goal": {
            "tool_point": "foot_1_sole",
            "physical": physical_goals,
            "concept": concept_goals,
            "unique_optimum": False,
            "batch1_joint_origin_kept": True,
        },
        "next_batch": {
            "not_started": True,
            "suggest_close": suggest_close,
            "intent": (
                "足底默认盒子上的条件化选型已经可以交付，建议父端暂停例行加长搜索。"
                if suggest_close
                else "全 0 实物在 30 次和 50 次迭代的端点可行数不一致。下一批只对不一致的代表把迭代加到 80，并用同一两套 125 点重核，不新开模块搜索。"
            ),
        },
    }
    OUT.write_text(json.dumps({"physical": physical, "concept": concept, "goals": state["recommendations_by_goal"]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    public_rows = physical + concept
    PUBLIC.write_text(
        json.dumps(
            {
                "tool_point": "foot_1_sole",
                "note": "批次 2 使用 ZBOT6.xml 的 foot_1_sole。关节原点结果仍在首批文件里。概念质量是 4 kg/m 假设。没有唯一最优。",
                "limitations": state["batch2"]["unverified_assumptions"],
                "goals": {
                    "coverage_priority": physical_goals["coverage_priority"],
                    "speed_priority": physical_goals["speed_priority"],
                    "light_priority": physical_goals["light_priority"],
                    "concept_coverage_priority": concept_goals.get("coverage_priority"),
                    "concept_speed_priority": concept_goals.get("speed_priority"),
                    "concept_light_priority": concept_goals.get("light_priority"),
                },
                "baseline": {"design": baseline["design"], "metrics": baseline["endpoint_125"], "bits": baseline["bits"], "role": "all_zero"},
                "feasible_pareto": [
                    {
                        "design": row["design"],
                        "metrics": row["endpoint_125"],
                        "centers": row["cell_centers_125"],
                        "bits": row["bits"],
                        "role": f"{row['kind']}:{row['role']}",
                        "modules": row["modules"],
                        "violation": 0.0,
                    }
                    for row in public_rows
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    mjcf_dir = ROOT / "experiments" / "dof678_sole_mjcf"
    mjcf_dir.mkdir(parents=True, exist_ok=True)
    for row in public_rows:
        design = design_from_dict(row["design"])
        (mjcf_dir / f"{row['kind']}_{row['count']}_{row['bits']}_{row['role']}.xml").write_text(
            export_mjcf(design, tool_point="foot_1_sole"),
            encoding="utf-8",
        )
    print(f"本批用时 {elapsed:.1f} s，建议收束 {suggest_close}", flush=True)


if __name__ == "__main__":
    main()
