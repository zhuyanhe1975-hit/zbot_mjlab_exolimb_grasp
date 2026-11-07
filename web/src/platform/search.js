/** 浏览器里的 NSGA-II。目标函数与 Python 相同；随机数流不同，所以同一种子不会逐个个体重合。 */

import { COUNT_MAX, LIBRARY, assemblyErrors, baselineDesign, evaluateDesign, mulberry32 } from "./model.js";

const SENSES = { coverage: "max", manipulability: "max", mass: "min", velocity: "max" };

export function validateSpec(spec) {
  const errors = [];
  if (!["physical", "mixed"].includes(spec.mode)) errors.push("模式只能是实物装配或概念混合");
  if (spec.count_min > spec.count_max) errors.push("模块数量下界不能大于上界");
  if (spec.count_min < 3 || spec.count_max > COUNT_MAX) errors.push(`模块数量范围须落在 3–${COUNT_MAX}`);
  if (spec.population < 4 || spec.population > 24 || spec.population % 2) errors.push("在线种群须为 4–24 的偶数");
  if (spec.generations < 1 || spec.generations > 20) errors.push("在线代数须为 1–20");
  if (spec.task.sample_count < 32 || spec.task.sample_count > 800) errors.push("在线采样须为 32–800，更大预算请用 Python");
  if (spec.task.payload_kg < 0 || spec.task.payload_kg > 20) errors.push("载荷质量须在 0–20 kg");
  if (!(spec.task.velocity_limit_rad_s > 0 && spec.task.velocity_limit_rad_s <= 20)) errors.push("关节速度上限须在 (0, 20] rad/s");
  if (!(spec.task.effort_limit_nm > 0 && spec.task.effort_limit_nm <= 200)) errors.push("力矩上限须在 (0, 200] N·m");
  if (spec.task.half_m.some((value) => value <= 0.01)) errors.push("任务区半尺寸必须大于 0.01 m");
  if (!(spec.length_lower_m < spec.length_upper_m)) errors.push("概念连杆长度下界必须小于上界");
  if (spec.length_lower_m < 0.06 || spec.length_upper_m > 0.16) errors.push("概念连杆长度搜索范围须在 0.06–0.16 m 内");
  if (spec.objectives.length < 2 || spec.objectives.length > 4) errors.push("至少选择 2 个、至多 4 个目标");
  return errors;
}

function allowed(spec) {
  return spec.mode === "physical" ? ["zbot_r3"] : ["zbot_r3", "concept_axial"];
}

function repair(genome, spec) {
  const types = allowed(spec);
  const count = spec.count_lock ?? Math.min(spec.count_max, Math.max(spec.count_min, genome.count));
  const modules = genome.module_ids.slice();
  const twists = genome.twists.slice();
  const lengths = genome.lengths.slice();
  while (modules.length < COUNT_MAX) modules.push(types[0]);
  while (twists.length < COUNT_MAX - 1) twists.push(0);
  while (lengths.length < COUNT_MAX) lengths.push(0.5 * (spec.length_lower_m + spec.length_upper_m));
  for (let index = 0; index < COUNT_MAX; index += 1) {
    if (!types.includes(modules[index])) modules[index] = types[0];
    lengths[index] = Math.min(spec.length_upper_m, Math.max(spec.length_lower_m, lengths[index]));
    if (index < COUNT_MAX - 1) twists[index] = twists[index] < Math.PI / 4 ? 0 : Math.PI / 2;
  }
  return { count, module_ids: modules.slice(0, COUNT_MAX), twists: twists.slice(0, COUNT_MAX - 1), lengths: lengths.slice(0, COUNT_MAX) };
}

function toDesign(genome, base) {
  const moduleIds = genome.module_ids.slice(0, genome.count);
  return {
    module_ids: moduleIds,
    twists_rad: genome.twists.slice(0, genome.count - 1),
    parameters: moduleIds.map((moduleId, index) => (
      LIBRARY[moduleId].parameters.length ? { length_m: genome.lengths[index] } : {}
    )),
    base_position_m: base,
  };
}

function salt(genome) {
  const text = `${genome.count}|${genome.module_ids.slice(0, genome.count).join("/")}|${genome.twists.slice(0, genome.count - 1).map((v) => v.toFixed(5)).join(",")}|${genome.lengths.slice(0, genome.count).map((v) => v.toFixed(5)).join(",")}`;
  let total = 0;
  for (const char of text) total = (total * 131 + char.charCodeAt(0)) % 100000;
  return total;
}

function objectivesOf(metrics, names) {
  const values = {
    coverage: -metrics.task_coverage,
    manipulability: -metrics.median_translational_manipulability,
    mass: metrics.mass_kg,
    velocity: -metrics.median_velocity_capability_m_s,
  };
  return names.map((name) => values[name]);
}

function dominates(a, b) {
  if (a.violation === 0 && b.violation > 0) return true;
  if (a.violation > 0 && b.violation === 0) return false;
  if (a.violation > 0 && b.violation > 0) return a.violation < b.violation - 1e-15;
  const notWorse = a.objectives.every((value, index) => value <= b.objectives[index] + 1e-15);
  const better = a.objectives.some((value, index) => value < b.objectives[index] - 1e-15);
  return notWorse && better;
}

function sortFronts(pop) {
  const dominated = Array(pop.length).fill(0);
  const sets = pop.map(() => []);
  const fronts = [[]];
  pop.forEach((_, i) => {
    pop.forEach((__, j) => {
      if (i === j) return;
      if (dominates(pop[i], pop[j])) sets[i].push(j);
      else if (dominates(pop[j], pop[i])) dominated[i] += 1;
    });
    if (dominated[i] === 0) {
      pop[i].rank = 0;
      fronts[0].push(i);
    }
  });
  let index = 0;
  while (fronts[index].length) {
    const next = [];
    fronts[index].forEach((i) => {
      sets[i].forEach((j) => {
        dominated[j] -= 1;
        if (dominated[j] === 0) {
          pop[j].rank = index + 1;
          next.push(j);
        }
      });
    });
    index += 1;
    fronts.push(next);
  }
  fronts.pop();
  fronts.forEach((front) => {
    front.forEach((i) => { pop[i].crowding = 0; });
    if (front.length <= 2) {
      front.forEach((i) => { pop[i].crowding = Infinity; });
      return;
    }
    const width = pop[front[0]].objectives.length;
    for (let obj = 0; obj < width; obj += 1) {
      const ordered = front.slice().sort((a, b) => pop[a].objectives[obj] - pop[b].objectives[obj]);
      pop[ordered[0]].crowding = Infinity;
      pop[ordered[ordered.length - 1]].crowding = Infinity;
      const span = pop[ordered[ordered.length - 1]].objectives[obj] - pop[ordered[0]].objectives[obj];
      if (span < 1e-15) continue;
      for (let pos = 1; pos < ordered.length - 1; pos += 1) {
        pop[ordered[pos]].crowding += (pop[ordered[pos + 1]].objectives[obj] - pop[ordered[pos - 1]].objectives[obj]) / span;
      }
    }
  });
  return fronts;
}

function better(pop, left, right) {
  if (pop[left].rank !== pop[right].rank) return pop[left].rank < pop[right].rank ? left : right;
  if (pop[left].crowding !== pop[right].crowding) return pop[left].crowding > pop[right].crowding ? left : right;
  return left < right ? left : right;
}

function randomGenome(rand, spec) {
  const types = allowed(spec);
  const count = spec.count_lock ?? spec.count_min + Math.floor(rand() * (spec.count_max - spec.count_min + 1));
  return repair({
    count,
    module_ids: Array.from({ length: COUNT_MAX }, () => types[Math.floor(rand() * types.length)]),
    twists: Array.from({ length: COUNT_MAX - 1 }, () => (rand() < 0.5 ? 0 : Math.PI / 2)),
    lengths: Array.from({ length: COUNT_MAX }, () => spec.length_lower_m + rand() * (spec.length_upper_m - spec.length_lower_m)),
  }, spec);
}

function fromDesign(design, spec) {
  const modules = design.module_ids.slice();
  const twists = design.twists_rad.slice();
  const lengths = design.parameters.map((params) => params.length_m ?? 0.5 * (spec.length_lower_m + spec.length_upper_m));
  while (modules.length < COUNT_MAX) modules.push("zbot_r3");
  while (twists.length < COUNT_MAX - 1) twists.push(0);
  while (lengths.length < COUNT_MAX) lengths.push(0.5 * (spec.length_lower_m + spec.length_upper_m));
  return repair({ count: design.module_ids.length, module_ids: modules, twists, lengths }, spec);
}

function vary(left, right, rand, spec) {
  const types = allowed(spec);
  let count = rand() < 0.5 ? left.count : right.count;
  let other = count === left.count ? right.count : left.count;
  if (rand() < 0.2 && spec.count_lock == null) {
    count = Math.min(spec.count_max, Math.max(spec.count_min, count + Math.floor(rand() * 3) - 1));
    other = Math.min(spec.count_max, Math.max(spec.count_min, other + Math.floor(rand() * 3) - 1));
  }
  const child = (flipCount) => {
    const modules = [];
    const twists = [];
    const lengths = [];
    for (let index = 0; index < COUNT_MAX; index += 1) {
      modules.push(rand() < 0.5 ? left.module_ids[index] : right.module_ids[index]);
      if (rand() < 1 / Math.max(flipCount, 1) && types.length > 1) modules[index] = types[Math.floor(rand() * types.length)];
      if (index < COUNT_MAX - 1) {
        twists.push(rand() < 0.5 ? left.twists[index] : right.twists[index]);
        if (rand() < 1 / Math.max(flipCount, 1)) twists[index] = twists[index] > 0.1 ? 0 : Math.PI / 2;
      }
      const low = spec.length_lower_m;
      const high = spec.length_upper_m;
      let length = 0.5 * (left.lengths[index] + right.lengths[index]);
      if (rand() < 1 / COUNT_MAX) length += (rand() - 0.5) * (high - low) * 0.3;
      lengths.push(Math.min(high, Math.max(low, length)));
    }
    return repair({ count: flipCount, module_ids: modules, twists, lengths }, spec);
  };
  return [child(count), child(other)];
}

function score(genome, spec, base) {
  const design = toDesign(genome, base);
  const samples = sampleFor(genome, spec);
  const metrics = evaluateDesign(design, spec.task, samples, false);
  return { genome, objectives: objectivesOf(metrics, spec.objectives), violation: metrics.violation, metrics, design };
}

function sampleFor(genome, spec) {
  const design = toDesign(genome, spec.base_position_m ?? [0, 0, 0.4]);
  const rand = mulberry32((spec.task.seed + salt(genome)) >>> 0);
  return Array.from({ length: spec.task.sample_count }, () =>
    Array.from({ length: design.module_ids.length }, () => (rand() * 2 - 1) * 6.28));
}

export async function runSearch(spec, start, hooks = {}) {
  const errors = validateSpec(spec);
  if (errors.length) throw new Error(errors.join("；"));
  if (start) {
    const assembly = assemblyErrors(start);
    if (assembly.length) throw new Error(assembly.join("；"));
  }
  const base = start?.base_position_m ?? [0, 0, 0.4];
  spec.base_position_m = base;
  const rand = mulberry32((spec.seed + 17) >>> 0);
  let pop = [];
  if (start) pop.push(score(fromDesign(start, spec), spec, base));
  if ((spec.count_min <= 6 && 6 <= spec.count_max) || spec.count_lock === 6) {
    pop.push(score(fromDesign(baselineDesign(), spec), spec, base));
  }
  while (pop.length < spec.population) pop.push(score(randomGenome(rand, spec), spec, base));
  pop = pop.slice(0, spec.population);
  const progress = [];
  for (let generation = 1; generation <= spec.generations; generation += 1) {
    if (hooks.yield) await new Promise((resolve) => setTimeout(resolve, 0));
    if (hooks.shouldStop?.()) break;
    sortFronts(pop);
    const parents = Array.from({ length: spec.population }, () => {
      const left = Math.floor(rand() * pop.length);
      const right = Math.floor(rand() * pop.length);
      return pop[better(pop, left, right)];
    });
    const children = [];
    for (let index = 0; index < spec.population; index += 2) {
      vary(parents[index].genome, parents[index + 1].genome, rand, spec).forEach((genome) => {
        children.push(score(genome, spec, base));
      });
    }
    pop = truncate(pop.concat(children), spec.population);
    const record = {
      generation,
      evaluations: generation * spec.population + spec.population,
      feasible: pop.filter((item) => item.violation === 0).length,
    };
    progress.push(record);
    hooks.onProgress?.(record);
  }
  sortFronts(pop);
  const front = pop.filter((item) => item.violation === 0 && item.rank === 0);
  const baselineRand = mulberry32(spec.task.seed >>> 0);
  const baselinePose = Array.from({ length: spec.task.sample_count }, () => Array.from({ length: 6 }, () => (baselineRand() * 2 - 1) * 6.28));
  const baselineMetrics = evaluateDesign(baselineDesign(), spec.task, baselinePose, true);
  return {
    platform_version: "0.1.0",
    mode: spec.mode,
    objectives: spec.objectives.map((id) => ({ id, label: labelOf(id), sense: SENSES[id] })),
    seed: spec.seed,
    population: spec.population,
    generations: spec.generations,
    sample_count: spec.task.sample_count,
    evaluations: spec.population * (progress.length + 1),
    progress,
    baseline: {
      design: baselineDesign(),
      metrics: baselineMetrics,
      note: "六模块 ZBot R3、连接方位全 0。同一任务设置下的对照。",
    },
    feasible_pareto: front.map((item) => ({
      design: item.design,
      objectives_minimized: item.objectives,
      violation: item.violation,
      metrics: { ...item.metrics, cloud: evaluateDesign(item.design, spec.task, sampleFor(item.genome, spec), true).cloud },
    })),
    infeasible_in_population: pop.filter((item) => item.violation > 0).length,
    assumptions: [
      "浏览器内的 NSGA-II 与 Python 使用同一目标函数和约束支配。随机数流不同，同一种子不会逐个个体相同。",
      "更大预算请用 python scripts/design_optimize.py。",
      `关节速度上限 ${spec.task.velocity_limit_rad_s} rad/s 是假设。`,
      "自碰撞是胶囊近似，静力矩是质点模型。",
    ],
  };
}

function labelOf(id) {
  return { coverage: "任务区覆盖率", manipulability: "平移可操作度中位数", mass: "质量", velocity: "速度能力中位数" }[id];
}

function truncate(pop, population) {
  const fronts = sortFronts(pop);
  const next = [];
  for (const front of fronts) {
    if (next.length + front.length <= population) {
      front.forEach((index) => next.push(pop[index]));
      continue;
    }
    front.slice().sort((a, b) => pop[b].crowding - pop[a].crowding).forEach((index) => {
      if (next.length < population) next.push(pop[index]);
    });
    break;
  }
  return next;
}

export { SENSES };
