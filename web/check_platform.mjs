import { readFileSync } from "node:fs";
import { exportMjcf } from "./src/platform/export.js";
import { evaluateDesign, forwardDesign } from "./src/platform/model.js";
import { runSearch } from "./src/platform/search.js";

const golden = JSON.parse(readFileSync(new URL("./public/platform_golden.json", import.meta.url)));

function near(actual, expected, tol, label) {
  const flatA = actual.flat ? actual.flat() : actual;
  const flatE = expected.flat ? expected.flat() : expected;
  const err = Math.max(...flatA.map((value, index) => Math.abs(value - flatE[index])));
  if (err > tol) throw new Error(`${label} 误差 ${err}`);
}

const baseline = {
  module_ids: Array(6).fill("zbot_r3"),
  twists_rad: Array(5).fill(0),
  parameters: Array.from({ length: 6 }, () => ({})),
  base_position_m: [0, 0, 0.4],
};
near(forwardDesign(baseline, [0, 0, 0, 0, 0, 0]).ee, golden.baseline_straight.ee, 1e-8, "伸直");
near(forwardDesign(baseline, [0.312, 0.837, -2.02, 2.02, -0.837, -0.312]).ee, golden.baseline_nominal.ee, 1e-8, "名义");
const mixedChain = forwardDesign(golden.mixed.design, golden.mixed.q);
near(mixedChain.ee, golden.mixed.ee, 1e-8, "混合链末端");
const jacRows = [0, 1, 2].map((row) => mixedChain.jacobian.map((column) => column[row]));
near(jacRows, golden.mixed.jacobian, 1e-7, "混合链雅可比");
if (Math.abs(mixedChain.mass - golden.mixed.mass_kg) > 1e-6) {
  throw new Error(`质量 ${mixedChain.mass} != ${golden.mixed.mass_kg}`);
}

const task = golden.diagnostic_task;
const metrics = evaluateDesign(baseline, task, golden.baseline_samples.q);
if (Math.abs(metrics.pose_hit_rate - golden.baseline_samples.pose_hit_rate) > 1e-9) {
  throw new Error(`落盒比例 ${metrics.pose_hit_rate}`);
}
if (Math.abs(metrics.posture_manipulability_median - golden.baseline_samples.posture_manipulability_median) > 1e-6) {
  throw new Error(`全姿态可操作度 ${metrics.posture_manipulability_median}`);
}
const ik = evaluateDesign(baseline, golden.ik_task, null);
if (ik.task_targets_reached !== golden.baseline_ik.task_targets_reached) {
  throw new Error(`格点可行数 ${ik.task_targets_reached} != ${golden.baseline_ik.task_targets_reached}`);
}
if (Math.abs(ik.task_feasible_fraction - golden.baseline_ik.task_feasible_fraction) > 1e-9) {
  throw new Error(`格点可行比例 ${ik.task_feasible_fraction}`);
}

const spec = {
  mode: "mixed",
  count_min: 3,
  count_max: 4,
  count_lock: null,
  population: 4,
  generations: 1,
  objectives: ["coverage", "mass"],
  length_lower_m: 0.08,
  length_upper_m: 0.12,
  seed: 2,
  task: {
    center_m: [0, 0.05, 0.72],
    half_m: [0.22, 0.22, 0.22],
    payload_kg: 0.2,
    effort_limit_nm: 20,
    velocity_limit_rad_s: 1.5,
    sample_count: 0,
    seed: 2,
    grid_divisions: 2,
    position_tolerance_m: 0.05,
    ik_starts: 2,
    ik_iterations: 4,
    ik_damping: 0.08,
  },
};
const wide = await runSearch({ ...spec, length_upper_m: 0.16, length_lower_m: 0.06 }, baseline);
const narrow = await runSearch(spec, {
  ...baseline,
  module_ids: ["zbot_r3", "concept_axial", "zbot_r3"],
  twists_rad: [0, 0],
  parameters: [{}, { length_m: 0.08 }, {}],
});
if (!wide.baseline?.metrics?.mass_kg || wide.evaluations < 4) throw new Error("搜索没有真实评价");
const wideMasses = wide.feasible_pareto.map((row) => row.metrics.mass_kg);
const narrowMasses = narrow.feasible_pareto.map((row) => row.metrics.mass_kg);
if (wideMasses.length && narrowMasses.length && wideMasses.join(",") === narrowMasses.join(",")) {
  throw new Error("改变长度范围后前沿完全相同，搜索像是写死的");
}
for (const design of [baseline, golden.mixed.design]) {
  const xml = exportMjcf(design);
  const open = (xml.match(/<body /g) ?? []).length;
  const close = (xml.match(/<\/body>/g) ?? []).length;
  const jointCount = (xml.match(/<joint /g) ?? []).length;
  if (open !== close || jointCount !== design.module_ids.length || !xml.includes('axis="0 0 1"') || !xml.includes('name="ee"')) {
    throw new Error(`MJCF 结构不对 open ${open} close ${close} joints ${jointCount}`);
  }
}
console.log(`平台模型核对通过。格点可行 ${ik.task_targets_reached}/${ik.task_target_count}，在线搜索评价 ${wide.evaluations} 次。`);
