"""模块选型、数量、次序、连接方位和连续参数的联合搜索。"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from .library import module_spec
from .model import assembly_errors, baseline_design, evaluate_design
from .nsga2 import Individual, feasible_front, nsga2
from .schema import MISSING_OBJECTIVE, OBJECTIVE_LABELS, OBJECTIVE_SENSE, COUNT_MAX, Design, SearchSpec, TaskSpec


@dataclass
class Genome:
    count: int
    module_ids: list[str]
    twists: list[float]
    lengths: list[float]

    def to_design(self, base: tuple[float, float, float]) -> Design:
        ids = tuple(self.module_ids[: self.count])
        parameters = []
        for module_id, length in zip(ids, self.lengths, strict=False):
            spec = module_spec(module_id)
            if spec.parameters:
                parameters.append({spec.parameters[0].name: float(length)})
            else:
                parameters.append({})
        return Design(
            module_ids=ids,
            twists_rad=tuple(self.twists[: self.count - 1]),
            parameters=tuple(parameters),
            base_position_m=base,
        )


def validate_search(spec: SearchSpec) -> list[str]:
    errors: list[str] = []
    if spec.mode not in {"physical", "mixed"}:
        errors.append("模式只能是实物装配或概念混合")
    if not spec.count_min <= spec.count_max:
        errors.append("模块数量下界不能大于上界")
    if spec.count_min < 3 or spec.count_max > COUNT_MAX:
        errors.append(f"模块数量范围须落在 3–{COUNT_MAX}")
    if spec.count_lock is not None and not spec.count_min <= spec.count_lock <= spec.count_max:
        errors.append("锁定的模块数量不在范围内")
    if spec.population < 4 or spec.population > 80 or spec.population % 2:
        errors.append("种群须为 4–80 的偶数")
    if spec.generations < 1 or spec.generations > 80:
        errors.append("代数须为 1–80")
    if spec.task.sample_count < 0 or spec.task.sample_count > 20000:
        errors.append("末端落盒诊断采样数须为 0–20000，0 表示不算这项")
    if spec.task.grid_divisions < 1 or spec.task.grid_divisions > 6:
        errors.append("任务格点每轴份数须为 1–6")
    if spec.task.ik_starts < 1 or spec.task.ik_starts > 16:
        errors.append("IK 起点数须为 1–16")
    if spec.task.ik_iterations < 1 or spec.task.ik_iterations > 40:
        errors.append("IK 迭代数须为 1–40")
    if not 0.005 <= spec.task.position_tolerance_m <= 0.2:
        errors.append("位置容差须在 0.005–0.2 m")
    if spec.task.payload_kg < 0 or spec.task.payload_kg > 20:
        errors.append("载荷质量须在 0–20 kg")
    if spec.task.velocity_limit_rad_s <= 0 or spec.task.velocity_limit_rad_s > 20:
        errors.append("关节速度上限须在 (0, 20] rad/s，且这是假设值")
    if spec.task.effort_limit_nm <= 0 or spec.task.effort_limit_nm > 200:
        errors.append("力矩上限须在 (0, 200] N·m")
    if any(value <= 0.01 for value in spec.task.half_m):
        errors.append("任务区半尺寸必须大于 0.01 m")
    if not spec.length_lower_m < spec.length_upper_m:
        errors.append("概念连杆长度下界必须小于上界")
    if spec.length_lower_m < 0.06 or spec.length_upper_m > 0.16:
        errors.append("概念连杆长度搜索范围须在 0.06–0.16 m 内")
    if len(spec.objectives) < 2 or len(spec.objectives) > 4:
        errors.append("至少选择 2 个、至多 4 个目标")
    for name in spec.objectives:
        if name not in OBJECTIVE_SENSE:
            errors.append(f"未知目标 {name}")
    return errors


def _clip_length(spec: SearchSpec, value: float) -> float:
    return float(np.clip(value, spec.length_lower_m, spec.length_upper_m))


def _repair(genome: Genome, spec: SearchSpec) -> Genome:
    allowed = list(spec.allowed_module_ids())
    count = genome.count
    if spec.count_lock is not None:
        count = spec.count_lock
    count = int(np.clip(count, spec.count_min, spec.count_max))
    modules = list(genome.module_ids)
    while len(modules) < COUNT_MAX:
        modules.append(allowed[0])
    twists = list(genome.twists)
    while len(twists) < COUNT_MAX - 1:
        twists.append(0.0)
    lengths = list(genome.lengths)
    while len(lengths) < COUNT_MAX:
        lengths.append(0.5 * (spec.length_lower_m + spec.length_upper_m))
    for index in range(COUNT_MAX):
        if modules[index] not in allowed:
            modules[index] = allowed[0]
        lengths[index] = _clip_length(spec, lengths[index])
        if index < COUNT_MAX - 1:
            twists[index] = 0.0 if twists[index] < math.pi / 4 else math.pi / 2
    return Genome(count=count, module_ids=modules[:COUNT_MAX], twists=twists[: COUNT_MAX - 1], lengths=lengths[:COUNT_MAX])


def _random_genome(rng: np.random.Generator, spec: SearchSpec) -> Genome:
    allowed = list(spec.allowed_module_ids())
    count = spec.count_lock if spec.count_lock is not None else int(rng.integers(spec.count_min, spec.count_max + 1))
    modules = [allowed[int(rng.integers(0, len(allowed)))] for _ in range(COUNT_MAX)]
    twists = [0.0 if rng.random() < 0.5 else math.pi / 2 for _ in range(COUNT_MAX - 1)]
    lengths = rng.uniform(spec.length_lower_m, spec.length_upper_m, size=COUNT_MAX).tolist()
    return _repair(Genome(count, modules, twists, lengths), spec)


def _genome_from_design(design: Design, spec: SearchSpec) -> Genome:
    modules = list(design.module_ids) + ["zbot_r3"] * COUNT_MAX
    twists = list(design.twists_rad) + [0.0] * COUNT_MAX
    lengths = []
    for index in range(COUNT_MAX):
        params = design.parameters[index] if index < len(design.parameters) else {}
        lengths.append(float(params.get("length_m", 0.5 * (spec.length_lower_m + spec.length_upper_m))))
    return _repair(
        Genome(count=len(design.module_ids), module_ids=modules[:COUNT_MAX], twists=twists[: COUNT_MAX - 1], lengths=lengths),
        spec,
    )


def _sbx(left: float, right: float, rng: np.random.Generator, low: float, high: float, eta: float = 15.0) -> tuple[float, float]:
    if abs(left - right) < 1e-12 or rng.random() > 0.9:
        return left, right
    y1, y2 = (left, right) if left < right else (right, left)
    rand = float(rng.random())
    beta = 1.0 + 2.0 * (y1 - low) / (y2 - y1)
    alpha = 2.0 - beta ** (-(eta + 1.0))
    if rand <= 1.0 / alpha:
        betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
    else:
        betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
    c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
    beta = 1.0 + 2.0 * (high - y2) / (y2 - y1)
    alpha = 2.0 - beta ** (-(eta + 1.0))
    if rand <= 1.0 / alpha:
        betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
    else:
        betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
    c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))
    c1 = float(np.clip(c1, low, high))
    c2 = float(np.clip(c2, low, high))
    if rng.random() < 0.5:
        return c1, c2
    return c2, c1


def _poly_mutation(value: float, rng: np.random.Generator, low: float, high: float, eta: float = 20.0) -> float:
    if rng.random() > 1.0 / COUNT_MAX:
        return value
    delta = high - low
    if delta <= 0:
        return value
    rand = float(rng.random())
    if rand < 0.5:
        xy = 1.0 - (value - low) / delta
        val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1.0)
        deltaq = val ** (1.0 / (eta + 1.0)) - 1.0
    else:
        xy = 1.0 - (high - value) / delta
        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1.0)
        deltaq = 1.0 - val ** (1.0 / (eta + 1.0))
    return float(np.clip(value + deltaq * delta, low, high))


def _vary(left: Genome, right: Genome, rng: np.random.Generator, spec: SearchSpec) -> tuple[Genome, Genome]:
    allowed = list(spec.allowed_module_ids())
    count = int(left.count if rng.random() < 0.5 else right.count)
    other = int(right.count if count == left.count else left.count)
    if rng.random() < 0.2 and spec.count_lock is None:
        count = int(np.clip(count + int(rng.integers(-1, 2)), spec.count_min, spec.count_max))
        other = int(np.clip(other + int(rng.integers(-1, 2)), spec.count_min, spec.count_max))
    modules_a, modules_b = [], []
    twists_a, twists_b = [], []
    lengths_a, lengths_b = [], []
    for index in range(COUNT_MAX):
        if rng.random() < 0.5:
            modules_a.append(left.module_ids[index])
            modules_b.append(right.module_ids[index])
        else:
            modules_a.append(right.module_ids[index])
            modules_b.append(left.module_ids[index])
        if rng.random() < 1.0 / max(count, 1) and len(allowed) > 1:
            modules_a[index] = allowed[int(rng.integers(0, len(allowed)))]
        if rng.random() < 1.0 / max(other, 1) and len(allowed) > 1:
            modules_b[index] = allowed[int(rng.integers(0, len(allowed)))]
        if index < COUNT_MAX - 1:
            if rng.random() < 0.5:
                twists_a.append(left.twists[index])
                twists_b.append(right.twists[index])
            else:
                twists_a.append(right.twists[index])
                twists_b.append(left.twists[index])
            if rng.random() < 1.0 / max(count, 1):
                twists_a[index] = 0.0 if twists_a[index] > 0.1 else math.pi / 2
            if rng.random() < 1.0 / max(other, 1):
                twists_b[index] = 0.0 if twists_b[index] > 0.1 else math.pi / 2
        child_a, child_b = _sbx(left.lengths[index], right.lengths[index], rng, spec.length_lower_m, spec.length_upper_m)
        lengths_a.append(_poly_mutation(child_a, rng, spec.length_lower_m, spec.length_upper_m))
        lengths_b.append(_poly_mutation(child_b, rng, spec.length_lower_m, spec.length_upper_m))
    return (
        _repair(Genome(count, modules_a, twists_a, lengths_a), spec),
        _repair(Genome(other, modules_b, twists_b, lengths_b), spec),
    )


def _objectives(metrics: dict[str, object], names: tuple[str, ...]) -> list[float]:
    manip = (
        -float(metrics["task_manipulability_median"])
        if metrics.get("task_manipulability_valid")
        else MISSING_OBJECTIVE
    )
    speed = (
        -float(metrics["task_velocity_median"])
        if metrics.get("task_velocity_valid")
        else MISSING_OBJECTIVE
    )
    values = {
        "coverage": -float(metrics["task_feasible_fraction"]),
        "manipulability": manip,
        "mass": float(metrics["mass_kg"]),
        "velocity": speed,
    }
    return [values[name] for name in names]


def _public_metrics(metrics: dict[str, object]) -> dict[str, object]:
    hidden = {"cloud"}
    return {key: value for key, value in metrics.items() if key not in hidden}


def run_search(
    spec: SearchSpec,
    start: Design | None = None,
    *,
    on_progress=None,
    should_stop=None,
    cloud_front: bool = False,
) -> dict[str, object]:
    errors = validate_search(spec)
    if errors:
        raise ValueError("；".join(errors))
    if start is not None:
        assembly = assembly_errors(start)
        if assembly:
            raise ValueError("；".join(assembly))
    started = time.perf_counter()
    base_position = start.base_position_m if start is not None else (0.0, 0.0, 0.40)
    evaluations = {"count": 0}

    def evaluate_genomes(genomes: list[Genome]):
        rows = []
        for genome in genomes:
            design = genome.to_design(base_position)
            # 每个个体使用同一任务种子的独立样本流，避免个体序号改变别人的随机数。
            local = TaskSpec(**{**spec.task.__dict__})
            local.seed = spec.task.seed + _genome_salt(genome)
            metrics = evaluate_design(design, local)
            evaluations["count"] += 1
            rows.append((_objectives(metrics, spec.objectives), float(metrics["violation"]), _public_metrics(metrics)))
        return rows

    def initialize(rng: np.random.Generator, population: int) -> list[Genome]:
        seeds = []
        if start is not None:
            seeds.append(_genome_from_design(start, spec))
        reference = _genome_from_design(baseline_design(), spec)
        if spec.count_min <= 6 <= spec.count_max or spec.count_lock == 6:
            seeds.append(reference)
        while len(seeds) < population:
            seeds.append(_random_genome(rng, spec))
        return [_repair(item, spec) for item in seeds[:population]]

    def vary(left: Genome, right: Genome, rng: np.random.Generator):
        return _vary(left, right, rng, spec)

    history: list[dict[str, int]] = []

    def progress(generation: int, pop: list) -> None:
        feasible = sum(1 for item in pop if item.violation == 0.0)
        record = {"generation": generation, "evaluations": evaluations["count"], "feasible": feasible}
        history.append(record)
        if on_progress is not None:
            on_progress(record)

    pop = nsga2(
        population=spec.population,
        generations=spec.generations,
        seed=spec.seed,
        initialize=initialize,
        vary=vary,
        evaluate=evaluate_genomes,
        on_progress=progress,
        should_stop=should_stop,
    )
    front = feasible_front(pop)
    candidates = [_candidate(item, base_position, spec, cloud_front) for item in front]
    reference_task = TaskSpec(**{**spec.task.__dict__})
    reference_task.seed = spec.task.seed
    reference = evaluate_design(baseline_design(), reference_task, include_cloud=cloud_front)
    return {
        "platform_version": "0.1.0",
        "mode": spec.mode,
        "objectives": [
            {"id": name, "label": OBJECTIVE_LABELS[name], "sense": OBJECTIVE_SENSE[name]}
            for name in spec.objectives
        ],
        "seed": spec.seed,
        "population": spec.population,
        "generations": spec.generations,
        "sample_count": spec.task.sample_count,
        "evaluations": evaluations["count"],
        "elapsed_s": time.perf_counter() - started,
        "progress": history,
        "baseline": {
            "design": design_to_dict(baseline_design()),
            "metrics": _public_metrics(reference) if not cloud_front else reference,
            "note": "六模块 ZBot R3、连接方位全 0。用同一次任务设置评价，不参与被搜索个体的随机样本。",
        },
        "feasible_pareto": candidates,
        "infeasible_in_population": sum(1 for item in pop if item.violation > 0.0),
        "assumptions": _assumptions(spec),
    }


def _candidate(item: Individual, base: tuple[float, float, float], spec: SearchSpec, cloud: bool) -> dict[str, object]:
    genome: Genome = item.genome
    design = genome.to_design(base)
    metrics = dict(item.payload or {})
    if cloud:
        local = TaskSpec(**{**spec.task.__dict__})
        local.seed = spec.task.seed + _genome_salt(genome)
        metrics = evaluate_design(design, local, include_cloud=True)
    return {
        "design": design_to_dict(design),
        "objectives_minimized": item.objectives,
        "violation": item.violation,
        "metrics": metrics,
    }


def _genome_salt(genome: Genome) -> int:
    text = f"{genome.count}|{'/'.join(genome.module_ids[: genome.count])}|{','.join(f'{v:.5f}' for v in genome.twists[: genome.count - 1])}|{','.join(f'{v:.5f}' for v in genome.lengths[: genome.count])}"
    total = 0
    for char in text:
        total = (total * 131 + ord(char)) % 100000
    return total


def design_to_dict(design: Design) -> dict[str, object]:
    rows = []
    for index, module_id in enumerate(design.module_ids):
        spec = module_spec(module_id)
        rows.append(
            {
                "id": module_id,
                "label": spec.label,
                "kind": spec.kind,
                "in_family": spec.in_family,
                "out_family": spec.out_family,
                "parameters": dict(design.parameters[index]),
                "twist_to_next_rad": design.twists_rad[index] if index < len(design.twists_rad) else None,
            }
        )
    return {
        "topology": "serial",
        "base_position_m": list(design.base_position_m),
        "modules": rows,
    }


def design_from_dict(payload: dict[str, object]) -> Design:
    if payload.get("topology") not in {None, "serial"}:
        raise ValueError("这一版只接受串联拓扑")
    modules = payload["modules"]
    if not isinstance(modules, list) or not modules:
        raise ValueError("设计里没有模块")
    ids = []
    parameters = []
    twists = []
    for index, row in enumerate(modules):
        if not isinstance(row, dict):
            raise ValueError("模块记录格式不对")
        ids.append(str(row["id"]))
        parameters.append({str(key): float(value) for key, value in dict(row.get("parameters") or {}).items()})
        twist = row.get("twist_to_next_rad")
        if index < len(modules) - 1:
            if twist is None:
                raise ValueError("除末模块外都要有连接方位")
            twists.append(float(twist))
    base = tuple(float(v) for v in payload.get("base_position_m", (0.0, 0.0, 0.4)))
    design = Design(tuple(ids), tuple(twists), tuple(parameters), base)
    errors = assembly_errors(design)
    if errors:
        raise ValueError("；".join(errors))
    return design


def _assumptions(spec: SearchSpec) -> list[str]:
    notes = [
        "串联链。不生成树或闭环。",
        "ZBot R3 的连杆尺寸、质量和 ±6.28 rad 行程来自既有 XML，实物模式不缩放。",
        "轴向概念连杆的线密度 4 kg/m、胶囊半径 0.025 m 是假设，不是第二套硬件。",
        f"关节速度上限 {spec.task.velocity_limit_rad_s} rad/s 是假设，仓库里没有 R3 实测速度。",
        "力矩上限默认 20 N·m，来自该项目位置伺服的力矩上限；界面改过则视为用户覆盖。",
        "自碰撞是非相邻胶囊的近似。静力矩是质点重力模型。",
        (
            f"任务指标用每轴 {spec.task.grid_divisions} 份格点、容差 {spec.task.position_tolerance_m} m、"
            f"{spec.task.ik_starts} 个 IK 起点、{spec.task.ik_iterations} 次迭代。"
            "未解出不是严格不可达。末端落盒比例如果计算，只是关节采样诊断。"
        ),
        "零位碰撞不否决设计。可操作度和速度能力只在可行任务解上取中位数，没有可行解就用惩罚。",
        "demo_flange 与 zbot_flange 没有装配规则，混合链会被拒绝。",
    ]
    if spec.task.effort_is_override:
        notes.append(f"本次力矩上限已由用户改为 {spec.task.effort_limit_nm} N·m。")
    return notes


def spec_to_dict(spec: SearchSpec) -> dict[str, object]:
    task = spec.task
    return {
        "mode": spec.mode,
        "count_min": spec.count_min,
        "count_max": spec.count_max,
        "count_lock": spec.count_lock,
        "population": spec.population,
        "generations": spec.generations,
        "objectives": list(spec.objectives),
        "length_lower_m": spec.length_lower_m,
        "length_upper_m": spec.length_upper_m,
        "seed": spec.seed,
        "task": {
            "center_m": list(task.center_m),
            "half_m": list(task.half_m),
            "payload_kg": task.payload_kg,
            "effort_limit_nm": task.effort_limit_nm,
            "effort_is_override": task.effort_is_override,
            "velocity_limit_rad_s": task.velocity_limit_rad_s,
            "sample_count": task.sample_count,
            "seed": task.seed,
            "grid_divisions": task.grid_divisions,
            "position_tolerance_m": task.position_tolerance_m,
            "ik_starts": task.ik_starts,
            "ik_iterations": task.ik_iterations,
            "ik_damping": task.ik_damping,
        },
    }


def spec_from_dict(payload: dict[str, object]) -> SearchSpec:
    task_payload = dict(payload.get("task") or {})
    task = TaskSpec(
        center_m=tuple(task_payload.get("center_m", (0.0, 0.05, 0.72))),
        half_m=tuple(task_payload.get("half_m", (0.22, 0.22, 0.22))),
        payload_kg=float(task_payload.get("payload_kg", 0.4)),
        effort_limit_nm=float(task_payload.get("effort_limit_nm", 20.0)),
        effort_is_override=bool(task_payload.get("effort_is_override", False)),
        velocity_limit_rad_s=float(task_payload.get("velocity_limit_rad_s", 1.5)),
        sample_count=int(task_payload.get("sample_count", 0)),
        seed=int(task_payload.get("seed", payload.get("seed", 1))),
        grid_divisions=int(task_payload.get("grid_divisions", 2)),
        position_tolerance_m=float(task_payload.get("position_tolerance_m", 0.05)),
        ik_starts=int(task_payload.get("ik_starts", 3)),
        ik_iterations=int(task_payload.get("ik_iterations", 8)),
        ik_damping=float(task_payload.get("ik_damping", 0.08)),
        reject_structural_degeneracy=bool(task_payload.get("reject_structural_degeneracy", False)),
    )
    lock = payload.get("count_lock")
    return SearchSpec(
        mode=str(payload.get("mode", "physical")),
        count_min=int(payload.get("count_min", 3)),
        count_max=int(payload.get("count_max", 6)),
        count_lock=None if lock in {None, ""} else int(lock),
        population=int(payload.get("population", 12)),
        generations=int(payload.get("generations", 6)),
        objectives=tuple(payload.get("objectives", ("coverage", "manipulability", "mass"))),
        length_lower_m=float(payload.get("length_lower_m", 0.06)),
        length_upper_m=float(payload.get("length_upper_m", 0.16)),
        seed=int(payload.get("seed", 1)),
        task=task,
    )
