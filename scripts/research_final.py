"""收尾批次：1 cm 分层评价全部 224 种实物接口，并加大概念搜索。

粗筛是 64 点。每个模块数的前几名再用 125 个端点和 125 个单元中心复核。
位姿用锥角内的随机采样，两种子。概念搜索用 pymoo，多种子，质量仍是 4 kg/m 假设。

用法::

    PYTHONPATH=src:third_party/generative-robot-design python scripts/research_final.py
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "generative-robot-design"))

from zbot_exolimb_grasp.platform.exporting import export_mjcf
from zbot_exolimb_grasp.platform.math3d import axis_angle, mul
from zbot_exolimb_grasp.platform.model import (
    collision_mask,
    forward_chain,
    ik_start_poses,
    solve_pose,
    static_torques,
    translational_manipulability,
    velocity_capability,
    zbot_chain,
)
from zbot_exolimb_grasp.platform.research_backend import gradient_position_ik, module_matrices, pinocchio_static_torques
from zbot_exolimb_grasp.platform.schema import Design

CACHE = ROOT / "experiments" / "cache" / "research_final.json"
OUT = ROOT / "experiments" / "research_final.json"
CSV_PATH = ROOT / "experiments" / "research_final_screen.csv"
PUBLIC = ROOT / "web" / "public" / "research_final.json"
MJCF_DIR = ROOT / "experiments" / "final_selection_mjcf"
STATE = ROOT / "experiments" / "longterm_search_state.json"

DEFAULT_CENTER = (0.0, 0.05, 0.72)
DEFAULT_HALF = (0.22, 0.22, 0.22)
HIGH_CENTER = (0.0, 0.22, 0.98)
HIGH_HALF = (0.10, 0.10, 0.08)
TOL = 0.01
CONE = math.radians(25.0)


def load_cache() -> dict:
    if CACHE.is_file():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache), encoding="utf-8")


def grid(center, half, divisions: int, centers: bool) -> np.ndarray:
    axes = []
    for value, width in zip(center, half, strict=True):
        if centers:
            step = 2.0 * width / divisions
            axes.append([value - width + step * (index + 0.5) for index in range(divisions)])
        else:
            axes.append(np.linspace(value - width, value + width, divisions).tolist())
    return np.asarray([(x, y, z) for x in axes[0] for y in axes[1] for z in axes[2]], dtype=float)


def design_of(count: int, bits: str) -> Design:
    twists = tuple(math.pi / 2 if bit == "1" else 0.0 for bit in bits)
    return Design(tuple("zbot_r3" for _ in range(count)), twists, tuple({} for _ in range(count)))


def evaluate_modules(module_ids, twists, lengths, targets: np.ndarray, *, starts: int, iterations: int, payload: float) -> dict:
    design = Design(tuple(module_ids), tuple(twists), tuple({} if module_id == "zbot_r3" else {"length_m": float(length)} for module_id, length in zip(module_ids, lengths, strict=True)))
    proximal, distal, mass = module_matrices(module_ids, twists, lengths)
    seed = np.stack(ik_start_poses(len(module_ids), starts, 6.28))
    q, error = gradient_position_ik(
        proximal,
        distal,
        targets,
        seed,
        tool="foot_1_sole",
        iterations=iterations,
        step=1.0,
        tolerance=TOL,
    )
    margins = []
    speeds = []
    manips = []
    reached = 0
    for index in range(len(targets)):
        if error[index] > TOL:
            continue
        chain = forward_chain(design, q[index], tool_point="foot_1_sole")
        if float(np.linalg.norm(chain["ee"][0] - targets[index])) > TOL:
            continue
        if bool(collision_mask(chain)[0]):
            continue
        torque = static_torques(chain, payload)[0]
        peak = float(np.max(np.abs(torque)))
        if peak > 20.0:
            continue
        reached += 1
        margins.append(20.0 - peak)
        manips.append(float(translational_manipulability(chain["jacobian"])))
        speeds.append(float(velocity_capability(chain["jacobian"], 1.5)))
    bits = "".join("1" if abs(value - math.pi / 2) < 0.1 else "0" for value in twists)
    return {
        "count": len(module_ids),
        "bits": bits,
        "interfaces_base_to_tip": [0 if bit == "0" else 90 for bit in bits],
        "reached": reached,
        "targets": int(len(targets)),
        "fraction": reached / float(len(targets)),
        "min_torque_margin_nm": None if not margins else float(min(margins)),
        "median_torque_margin_nm": None if not margins else float(np.median(margins)),
        "median_speed_m_s": None if not speeds else float(np.median(speeds)),
        "median_manipulability": None if not manips else float(np.median(manips)),
        "mass_kg": mass,
        "payload_kg": payload,
        "mass_is_assumption": any(module_id != "zbot_r3" for module_id in module_ids),
    }


def evaluate_position(count: int, bits: str, targets: np.ndarray, *, starts: int, iterations: int, payload: float) -> dict:
    twists = tuple(math.pi / 2 if bit == "1" else 0.0 for bit in bits)
    return evaluate_modules(("zbot_r3",) * count, twists, (0.0,) * count, targets, starts=starts, iterations=iterations, payload=payload)


def cached_eval(cache: dict, key: str, fn):
    if key in cache:
        return cache[key]
    value = fn()
    cache[key] = value
    save_cache(cache)
    return value


def all_bits(count: int):
    for mask in range(1 << (count - 1)):
        yield "".join("1" if mask & (1 << index) else "0" for index in range(count - 1))


def rank_key(row: dict):
    return (
        row["fraction"],
        row["min_torque_margin_nm"] if row["min_torque_margin_nm"] is not None else -1.0,
        row["median_speed_m_s"] if row["median_speed_m_s"] is not None else -1.0,
    )


def sample_cone(n: int, seed: int) -> list[dict]:
    """盒内均匀位置。工具 z 轴落在世界 -Z 周围 25° 锥内，滚转 ±90°。不是 6 模块零位姿。"""
    rng = np.random.default_rng(seed)
    ref = np.array([0.0, 1.0, 0.0, 0.0])
    poses = []
    low = np.asarray(DEFAULT_CENTER) - np.asarray(DEFAULT_HALF)
    high = np.asarray(DEFAULT_CENTER) + np.asarray(DEFAULT_HALF)
    for index in range(n):
        position = rng.uniform(low, high)
        azimuth = float(rng.uniform(0.0, 2.0 * math.pi))
        cosine = float(rng.uniform(math.cos(CONE), 1.0))
        polar = math.acos(cosine)
        tilt_axis = np.array([math.cos(azimuth), math.sin(azimuth), 0.0])
        tilt = axis_angle(tilt_axis, polar)
        roll = axis_angle(np.array([0.0, 0.0, -1.0]), float(rng.uniform(-math.pi / 2, math.pi / 2)))
        quat = mul(roll, mul(tilt, ref))
        poses.append({"position_m": position.tolist(), "quat_wxyz": [float(v) for v in quat], "id": f"s{seed}-{index}"})
    return poses


def pose_fraction(count: int, bits: str, poses: list[dict]) -> dict:
    from zbot_exolimb_grasp.platform.schema import TaskSpec

    design = design_of(count, bits)
    task = TaskSpec(tool_point="foot_1_sole", ik_starts=4, ik_iterations=16, payload_kg=0.4, effort_limit_nm=20.0, sample_count=0)
    hit = 0
    for pose in poses:
        solved = solve_pose(
            design,
            np.asarray(pose["position_m"]),
            np.asarray(pose["quat_wxyz"]),
            task,
            position_tolerance_m=0.01,
            rotation_tolerance_rad=math.radians(10.0),
        )
        hit += int(solved["solved"])
    return {"count": count, "bits": bits, "reached": hit, "targets": len(poses), "fraction": hit / float(len(poses))}


def main() -> None:
    started = time.perf_counter()
    cache = load_cache()
    coarse_targets = {
        "default": grid(DEFAULT_CENTER, DEFAULT_HALF, 4, False),
        "high": grid(HIGH_CENTER, HIGH_HALF, 4, False),
    }
    screened = {"default": [], "high": []}
    for box, targets in coarse_targets.items():
        for count in (6, 7, 8):
            for bits in all_bits(count):
                key = f"coarse64-1cm:{box}:{count}:{bits}"
                def run(count=count, bits=bits, targets=targets):
                    row = evaluate_position(count, bits, targets, starts=4, iterations=12, payload=0.4)
                    print(f"粗筛 {box} {count} {bits} {row['reached']}/{row['targets']} 裕度 {row['min_torque_margin_nm']}", flush=True)
                    return row
                screened[box].append(cached_eval(cache, key, run))
    tops = {}
    for box, rows in screened.items():
        tops[box] = {}
        for count in (6, 7, 8):
            group = [row for row in rows if row["count"] == count]
            group.sort(key=rank_key, reverse=True)
            tops[box][str(count)] = group[:4]
            print("前四", box, count, [(row["bits"], row["reached"], row["min_torque_margin_nm"]) for row in group[:4]], flush=True)

    review = []
    chosen = {}
    for box in ("default", "high"):
        picks = []
        for count in (6, 7, 8):
            picks.extend(tops[box][str(count)])
            zero = "0" * (count - 1)
            if all(row["bits"] != zero for row in tops[box][str(count)]):
                picks.append({"count": count, "bits": zero})
        seen = set()
        for row in picks:
            item = (row["count"], row["bits"])
            if item in seen:
                continue
            seen.add(item)
            for name, centers in (("endpoints", False), ("centers", True)):
                divisions = 5
                targets = grid(DEFAULT_CENTER if box == "default" else HIGH_CENTER, DEFAULT_HALF if box == "default" else HIGH_HALF, divisions, centers)
                key = f"review125-1cm:{box}:{name}:{row['count']}:{row['bits']}"
                def run(count=row["count"], bits=row["bits"], targets=targets):
                    result = evaluate_position(count, bits, targets, starts=6, iterations=20, payload=0.4)
                    print(f"复核 {box} {name} {count} {bits} {result['reached']}/{result['targets']}", flush=True)
                    return result
                result = cached_eval(cache, key, run)
                result = dict(result)
                result["box"] = box
                result["set"] = name
                review.append(result)
        chosen[box] = list(seen)

    heavy = []
    for count, bits in chosen["default"]:
        targets = grid(DEFAULT_CENTER, DEFAULT_HALF, 4, False)
        key = f"heavy4kg-64:{count}:{bits}"
        def run(count=count, bits=bits, targets=targets):
            result = evaluate_position(count, bits, targets, starts=4, iterations=12, payload=4.0)
            print(f"重载 {count} {bits} {result['reached']}/{result['targets']} 裕度 {result['min_torque_margin_nm']}", flush=True)
            return result
        heavy.append(cached_eval(cache, key, run))

    pose_sets = {str(seed): sample_cone(12, seed) for seed in (11, 23)}
    pose_rows = []
    pose_candidates = []
    for count in (6, 7, 8):
        pose_candidates.append((count, "0" * (count - 1)))
        for row in tops["default"][str(count)][:2]:
            pose_candidates.append((row["count"], row["bits"]))
    pose_seen = []
    for count, bits in pose_candidates:
        if (count, bits) in pose_seen:
            continue
        pose_seen.append((count, bits))
        for seed, poses in pose_sets.items():
            key = f"pose-cone:{seed}:{count}:{bits}"
            def run(count=count, bits=bits, poses=poses):
                result = pose_fraction(count, bits, poses)
                print(f"位姿 种子{seed} {count} {bits} {result['reached']}/{result['targets']}", flush=True)
                return result
            result = cached_eval(cache, key, run)
            result = dict(result)
            result["seed"] = int(seed)
            pose_rows.append(result)

    concept = concept_search(cache)
    physical = recommendations(review, heavy, pose_rows, screened)
    elapsed = time.perf_counter() - started
    report = {
        "status": "final_example_selection",
        "not_a_user_task": True,
        "not_global_optimum": True,
        "elapsed_s": elapsed,
        "budget": {
            "coarse": "全部 224 种，默认盒和高远盒，1 cm，每轴 4 份共 64 点，4 起点，12 次迭代",
            "review": "每个模块数粗筛前 4 名加全 0，1 cm，125 端点与 125 单元中心，6 起点，20 次迭代",
            "pose": "12 个锥角位姿，种子 11 和 23，位置 1 cm，旋转 10°，4 起点，16 次迭代",
            "concept": concept["budget"],
            "why_not_125_for_all": "64 点已能拉开力矩和速度；125 点只用于前段，避免 224 乘两套盒子超出本次时间。",
        },
        "coarse_spread": {
            box: {
                str(count): {
                    "min_reached": min(row["reached"] for row in screened[box] if row["count"] == count),
                    "max_reached": max(row["reached"] for row in screened[box] if row["count"] == count),
                    "n": sum(row["count"] == count for row in screened[box]),
                }
                for count in (6, 7, 8)
            }
            for box in ("default", "high")
        },
        "tops": {box: {count: [{"bits": row["bits"], "reached": row["reached"], "min_torque_margin_nm": row["min_torque_margin_nm"], "median_speed_m_s": row["median_speed_m_s"]} for row in rows] for count, rows in groups.items()} for box, groups in tops.items()},
        "review": review,
        "heavy_4kg": heavy,
        "pose": pose_rows,
        "concept": concept,
        "recommendations": physical,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(screened)
    write_mjcf(physical)
    write_public(report)
    update_state(report)
    print("完成", round(elapsed, 1), "s", flush=True)


def concept_search(cache: dict) -> dict:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.optimize import minimize

    targets = grid(DEFAULT_CENTER, DEFAULT_HALF, 3, False)
    baseline = cached_eval(
        cache,
        "concept-baseline-27-1cm",
        lambda: evaluate_position(6, "00000", targets, starts=4, iterations=12, payload=0.4),
    )

    class Problem(ElementwiseProblem):
        def __init__(self, count: int):
            self.arm_count = count
            super().__init__(n_var=3 * (count - 1), n_obj=2, xl=0.0, xu=1.0)

        def _evaluate(self, x, out, *args, **kwargs):
            count = self.arm_count
            types = [1 if value > 0.5 else 0 for value in x[: count - 1]]
            twist_bits = [1 if value > 0.5 else 0 for value in x[count - 1 : 2 * (count - 1)]]
            lengths = [0.06 + 0.10 * float(value) for value in x[2 * (count - 1) :]]
            module_ids = tuple("concept_axial" if bit else "zbot_r3" for bit in types) + ("zbot_r3",)
            twists = tuple(math.pi / 2 if bit else 0.0 for bit in twist_bits)
            length_values = tuple(lengths) + (0.0,)
            key = "concept:" + json.dumps([module_ids, twists, [round(v, 3) for v in length_values]])
            def run(module_ids=module_ids, twists=twists, length_values=length_values, count=count):
                return evaluate_modules(module_ids, twists, length_values, targets, starts=3, iterations=10, payload=0.4)
            metrics = cached_eval(cache, key, run)
            out["F"] = [-metrics["fraction"], metrics["mass_kg"]]

    fronts = []
    for count in (6, 7, 8):
        for seed in (11, 23):
            key = f"pymoo-summary:{count}:{seed}"
            if key in cache:
                fronts.append(cache[key])
                continue
            result = minimize(Problem(count), NSGA2(pop_size=16), termination=("n_gen", 5), seed=seed, verbose=False)
            rows = []
            for item in np.atleast_2d(result.F):
                rows.append({"feasible_fraction": float(-item[0]), "mass_kg_assumed": float(item[1]), "mass_is_assumption": True, "count": count, "seed": seed})
            best = max(rows, key=lambda row: (row["feasible_fraction"], -row["mass_kg_assumed"]))
            record = {"count": count, "seed": seed, "best": best, "front_size": len(rows)}
            cache[key] = record
            save_cache(cache)
            fronts.append(record)
            print("概念", count, seed, best, flush=True)
    beats = []
    for row in fronts:
        best = row["best"]
        if best["feasible_fraction"] >= baseline["fraction"] - 0.03 and best["mass_kg_assumed"] < baseline["mass_kg"] - 0.3:
            beats.append(best)
    return {
        "budget": "每个驱动数、每个种子：种群 16、5 代。评价是 27 点、1 cm、3 起点、10 次迭代。两种子 11 和 23。",
        "physical_baseline_same_budget": baseline,
        "runs": fronts,
        "credible_mass_and_coverage_win": beats,
        "conclusion": "没有同时在覆盖上接近实物、又在假设质量上轻超过 0.3 kg 的概念候选。" if not beats else "出现了需要和实物分开报告的概念候选。",
    }


def recommendations(review, heavy, pose_rows, screened) -> dict:
    def both(box: str, count: int, bits: str) -> tuple[dict, dict]:
        ends = next(row for row in review if row["box"] == box and row["set"] == "endpoints" and row["count"] == count and row["bits"] == bits)
        centers = next(row for row in review if row["box"] == box and row["set"] == "centers" and row["count"] == count and row["bits"] == bits)
        return ends, centers

    def worse(box, count, bits):
        ends, centers = both(box, count, bits)
        return min(ends["fraction"], centers["fraction"]), ends, centers

    default_candidates = {(row["count"], row["bits"]) for row in review if row["box"] == "default" and row["set"] == "endpoints"}
    high_candidates = {(row["count"], row["bits"]) for row in review if row["box"] == "high" and row["set"] == "endpoints"}

    def best_coverage(box, pool):
        return max(pool, key=lambda item: (worse(box, *item)[0], -item[0]))

    near = best_coverage("default", default_candidates)
    # 轻量优先在覆盖不比最优差 0.03 的实物里取模块数最少，通常是 6。
    best_default = worse("default", *near)[0]
    light_pool = [item for item in default_candidates if worse("default", *item)[0] >= best_default - 0.03]
    light = min(light_pool, key=lambda item: (item[0], item[1]))
    speed_pool = [item for item in default_candidates if worse("default", *item)[0] >= best_default - 0.03]
    speed = max(speed_pool, key=lambda item: both("default", *item)[0]["median_speed_m_s"] or -1)
    high = best_coverage("high", high_candidates)
    pose_by_arm: dict[tuple[int, str], list[float]] = {}
    for row in pose_rows:
        pose_by_arm.setdefault((row["count"], row["bits"]), []).append(row["fraction"])
    pose = max(pose_by_arm, key=lambda item: (min(pose_by_arm[item]), -item[0]))
    heavy_best = max(heavy, key=lambda row: (row["fraction"], row["min_torque_margin_nm"] or -1, -row["count"]))

    def pack(count, bits, box):
        ends, centers = both(box, count, bits)
        base_ends, base_centers = both(box, 6, "00000")
        return {
            "count": count,
            "bits": bits,
            "interfaces_base_to_tip_deg": [0 if bit == "0" else 90 for bit in bits],
            "modules": ["zbot_r3"] * count,
            "mass_kg": ends["mass_kg"],
            "endpoints": f"{ends['reached']}/{ends['targets']}",
            "centers": f"{centers['reached']}/{centers['targets']}",
            "min_torque_margin_nm": ends["min_torque_margin_nm"],
            "median_speed_m_s": ends["median_speed_m_s"],
            "median_manipulability": ends["median_manipulability"],
            "gain_vs_6_zero_endpoints": ends["fraction"] - base_ends["fraction"],
            "gain_vs_6_zero_centers": centers["fraction"] - base_centers["fraction"],
            "mjcf": f"experiments/final_selection_mjcf/zbot_{count}_{bits}.xml",
        }

    return {
        "near_light": {**pack(*light, "default"), "why": "默认盒上覆盖不落后，模块数最少，质量 4.279 kg。"},
        "high_far": {**pack(*high, "high"), "why": "高远盒两套 125 点上可行比例最高；并列时取更少模块。"},
        "pose_6d": {
            "count": pose[0],
            "bits": pose[1],
            "interfaces_base_to_tip_deg": [0 if bit == "0" else 90 for bit in pose[1]],
            "seeds": pose_by_arm[pose],
            "why": "两种子锥角位姿里较差的一种子最高。这组姿态不是 6 模块零位姿。",
            "mjcf": f"experiments/final_selection_mjcf/zbot_{pose[0]}_{pose[1]}.xml",
        },
        "light_load_speed": {**pack(*speed, "default"), "why": "默认可覆盖不掉过 0.03 的候选里，可行点伪逆速度中位数最高。它是保守下界，不是全方向最快。"},
        "heavy_4kg": {
            "count": heavy_best["count"],
            "bits": heavy_best["bits"],
            "interfaces_base_to_tip_deg": heavy_best["interfaces_base_to_tip"],
            "reached": f"{heavy_best['reached']}/{heavy_best['targets']}",
            "min_torque_margin_nm": heavy_best["min_torque_margin_nm"],
            "mass_kg": heavy_best["mass_kg"],
            "why": "4 kg、64 点、1 cm 上可行比例最高，再看最小力矩裕度。",
            "mjcf": f"experiments/final_selection_mjcf/zbot_{heavy_best['count']}_{heavy_best['bits']}.xml",
        },
        "concept_alternative": None,
        "manipulability_note": "平移可操作度随关节数变大，没有用来决定这些选型。",
    }


def write_csv(screened) -> None:
    lines = ["box,count,bits,reached,targets,fraction,min_torque_margin_nm,median_speed_m_s,median_manipulability,mass_kg\n"]
    for box, rows in screened.items():
        for row in rows:
            lines.append(
                f"{box},{row['count']},{row['bits']},{row['reached']},{row['targets']},{row['fraction']},"
                f"{row['min_torque_margin_nm']},{row['median_speed_m_s']},{row['median_manipulability']},{row['mass_kg']}\n"
            )
    CSV_PATH.write_text("".join(lines), encoding="utf-8")


def write_mjcf(physical: dict) -> None:
    MJCF_DIR.mkdir(parents=True, exist_ok=True)
    seen = set()
    for item in physical.values():
        if not isinstance(item, dict) or "count" not in item:
            continue
        key = (item["count"], item["bits"])
        if key in seen:
            continue
        seen.add(key)
        design = zbot_chain(item["count"], tuple(math.pi / 2 if bit == "1" else 0.0 for bit in item["bits"]))
        (MJCF_DIR / f"zbot_{item['count']}_{item['bits']}.xml").write_text(export_mjcf(design, tool_point="foot_1_sole"), encoding="utf-8")


def write_public(report: dict) -> None:
    PUBLIC.write_text(
        json.dumps(
            {
                "note": "最终示例选型。不是用户工况，也不是全局最优。概念质量若出现，仍是 4 kg/m 假设。",
                "recommendations": report["recommendations"],
                "concept_conclusion": report["concept"]["conclusion"],
                "coarse_spread": report["coarse_spread"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def update_state(report: dict) -> None:
    state = json.loads(STATE.read_text(encoding="utf-8"))
    state["checkpoint_version"] = 4
    state["status"] = "final_example_selection_delivered"
    state["longterm_status"] = "delivered_example_selection"
    state["suggest_close"] = True
    state["suggest_close_scope"] = {
        "delivered": "示例任务上的条件化实物选型",
        "not_included": "用户实测工况、全局最优、概念长度穷尽、全方向速度、完整 SO(3) 工作空间",
    }
    previous_final = float((state.get("final_selection") or {}).get("elapsed_s") or 0.0)
    state["final_selection"] = {
        "source": "experiments/research_final.json",
        "recommendations": report["recommendations"],
        "concept": report["concept"]["conclusion"],
        "elapsed_s": report["elapsed_s"],
        "budget": report["budget"],
    }
    if "research_final" not in state["completed_batches"]:
        state["completed_batches"].append("research_final")
    state["elapsed_s"] = float(state.get("elapsed_s") or 0.0) - previous_final + float(report["elapsed_s"])
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
