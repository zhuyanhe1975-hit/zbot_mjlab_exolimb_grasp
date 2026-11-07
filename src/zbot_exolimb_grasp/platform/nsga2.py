"""约束支配的 NSGA-II。

实现遵循 Deb 等人 2002 的非支配排序、拥挤距离和精英保留。
离散基因由外部的交叉变异函数处理。同一随机种子下结果可重复。
不可行个体用约束违反量参与支配，不会进入 violation 为 0 的可行前沿。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class Individual:
    genome: object
    objectives: list[float]
    violation: float
    rank: int = 0
    crowding: float = 0.0
    payload: dict | None = None


def dominates(a: Individual, b: Individual) -> bool:
    if a.violation == 0.0 and b.violation > 0.0:
        return True
    if a.violation > 0.0 and b.violation == 0.0:
        return False
    if a.violation > 0.0 and b.violation > 0.0:
        return a.violation < b.violation - 1e-15
    not_worse = all(left <= right + 1e-15 for left, right in zip(a.objectives, b.objectives, strict=True))
    better = any(left < right - 1e-15 for left, right in zip(a.objectives, b.objectives, strict=True))
    return not_worse and better


def fast_non_dominated_sort(pop: list[Individual]) -> list[list[int]]:
    size = len(pop)
    dominated = [0] * size
    dominates_set: list[list[int]] = [[] for _ in range(size)]
    fronts: list[list[int]] = [[]]
    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            if dominates(pop[i], pop[j]):
                dominates_set[i].append(j)
            elif dominates(pop[j], pop[i]):
                dominated[i] += 1
        if dominated[i] == 0:
            pop[i].rank = 0
            fronts[0].append(i)
    index = 0
    while fronts[index]:
        nxt: list[int] = []
        for i in fronts[index]:
            for j in dominates_set[i]:
                dominated[j] -= 1
                if dominated[j] == 0:
                    pop[j].rank = index + 1
                    nxt.append(j)
        index += 1
        fronts.append(nxt)
    fronts.pop()
    return fronts


def crowding_distance(pop: list[Individual], front: list[int]) -> None:
    if not front:
        return
    for index in front:
        pop[index].crowding = 0.0
    if len(front) <= 2:
        for index in front:
            pop[index].crowding = float("inf")
        return
    width = len(pop[front[0]].objectives)
    for obj in range(width):
        ordered = sorted(front, key=lambda item: pop[item].objectives[obj])
        pop[ordered[0]].crowding = float("inf")
        pop[ordered[-1]].crowding = float("inf")
        low = pop[ordered[0]].objectives[obj]
        high = pop[ordered[-1]].objectives[obj]
        span = high - low
        if span < 1e-15:
            continue
        for pos in range(1, len(ordered) - 1):
            previous = pop[ordered[pos - 1]].objectives[obj]
            nxt = pop[ordered[pos + 1]].objectives[obj]
            pop[ordered[pos]].crowding += (nxt - previous) / span


def _better(pop: list[Individual], left: int, right: int) -> int:
    a, b = pop[left], pop[right]
    if a.rank < b.rank:
        return left
    if b.rank < a.rank:
        return right
    if a.crowding > b.crowding:
        return left
    if b.crowding > a.crowding:
        return right
    return left if left < right else right


def nsga2(
    *,
    population: int,
    generations: int,
    seed: int,
    initialize: Callable[[np.random.Generator, int], list[object]],
    vary: Callable[[object, object, np.random.Generator], tuple[object, object]],
    evaluate: Callable[[list[object]], list[tuple[list[float], float, dict]]],
    on_progress: Callable[[int, list[Individual]], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> list[Individual]:
    if population < 4 or population % 2 != 0:
        raise ValueError("种群大小须为不小于 4 的偶数")
    if generations < 1:
        raise ValueError("代数至少为 1")
    rng = np.random.default_rng(seed)
    genomes = initialize(rng, population)
    scored = evaluate(genomes)
    pop = [
        Individual(genome=genome, objectives=list(obj), violation=float(viol), payload=payload)
        for genome, (obj, viol, payload) in zip(genomes, scored, strict=True)
    ]
    _assign(pop)
    for generation in range(generations):
        if should_stop is not None and should_stop():
            break
        parents = _select_parents(pop, rng)
        children_genomes: list[object] = []
        for index in range(0, population, 2):
            left, right = vary(parents[index].genome, parents[index + 1].genome, rng)
            children_genomes.extend((left, right))
        child_scores = evaluate(children_genomes)
        children = [
            Individual(genome=genome, objectives=list(obj), violation=float(viol), payload=payload)
            for genome, (obj, viol, payload) in zip(children_genomes, child_scores, strict=True)
        ]
        combined = pop + children
        _assign(combined)
        pop = _truncate(combined, population)
        if on_progress is not None:
            on_progress(generation + 1, pop)
    return pop


def feasible_front(pop: list[Individual]) -> list[Individual]:
    return [item for item in pop if item.violation == 0.0 and item.rank == 0]


def _assign(pop: list[Individual]) -> None:
    fronts = fast_non_dominated_sort(pop)
    for front in fronts:
        crowding_distance(pop, front)


def _select_parents(pop: list[Individual], rng: np.random.Generator) -> list[Individual]:
    size = len(pop)
    chosen: list[Individual] = []
    for _ in range(size):
        left, right = rng.integers(0, size, size=2)
        chosen.append(pop[_better(pop, int(left), int(right))])
    return chosen


def _truncate(pop: list[Individual], population: int) -> list[Individual]:
    fronts = fast_non_dominated_sort(pop)
    nxt: list[Individual] = []
    for front in fronts:
        if len(nxt) + len(front) <= population:
            nxt.extend(pop[index] for index in front)
            continue
        ordered = sorted(front, key=lambda index: pop[index].crowding, reverse=True)
        for index in ordered:
            if len(nxt) >= population:
                break
            nxt.append(pop[index])
        break
    return nxt
