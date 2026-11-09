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

const sole = golden.foot_1_sole;
if (sole.name !== "foot_1_sole") throw new Error(`足底名称 ${sole.name}`);
for (const count of [6, 7, 8]) {
  const expected = sole.arms[String(count)];
  const arm = {
    module_ids: Array(count).fill("zbot_r3"),
    twists_rad: Array(count - 1).fill(0),
    parameters: Array.from({ length: count }, () => ({})),
    base_position_m: [0, 0, 0.4],
  };
  const chain = forwardDesign(arm, expected.pose, "foot_1_sole");
  const origin = forwardDesign(arm, expected.pose, "foot_joint_origin");
  if (chain.toolPointApplied !== "foot_1_sole") throw new Error("浏览器没有使用足底点");
  if (chain.jacobian.length !== count) throw new Error(`足底雅可比列数 ${chain.jacobian.length}`);
  near(chain.ee, expected.ee, 1e-8, `${count} 模块足底`);
  near(origin.ee, expected.origin_ee, 1e-8, `${count} 模块关节原点`);
  const distal = chain.jacobian[count - 1];
  const distalNorm = Math.hypot(...distal);
  if (Math.abs(distalNorm - expected.distal_column_norm) > 1e-6) throw new Error(`末关节平移列 ${distalNorm}`);
  const originNorm = Math.hypot(...origin.jacobian[count - 1]);
  if (originNorm > 1e-8) throw new Error("关节原点模式的末关节平移列不是 0");
}

for (const count of [7, 8]) {
  const arm = {
    module_ids: Array(count).fill("zbot_r3"),
    twists_rad: Array(count - 1).fill(0),
    parameters: Array.from({ length: count }, () => ({})),
    base_position_m: [0, 0, 0.4],
  };
  const pose = Array.from({ length: count }, (_, index) => -1.2 + (2.3 * index) / (count - 1));
  const straight = forwardDesign(arm, Array(count).fill(0));
  const chain = forwardDesign(arm, pose);
  if (chain.jacobian.length !== count) throw new Error(`${count} 模块雅可比被截成 ${chain.jacobian.length}`);
  if (chain.axes.length !== count) throw new Error(`${count} 模块关节轴被截成 ${chain.axes.length}`);
  near(straight.ee, golden[`zbot_${count}`].straight_ee, 1e-8, `${count} 模块伸直`);
  near(chain.ee, golden[`zbot_${count}`].ee, 1e-8, `${count} 模块末端`);
  const task = {
    center_m: straight.ee,
    half_m: [0, 0, 0],
    payload_kg: 0.4,
    effort_limit_nm: 20,
    velocity_limit_rad_s: 1.5,
    sample_count: 0,
    seed: 1,
    grid_divisions: 1,
    position_tolerance_m: 0.01,
    ik_starts: 1,
    ik_iterations: 2,
    ik_damping: 0.08,
    joint_sample_limit: 6.28,
  };
  const solved = evaluateDesign(arm, task, null);
  if (solved.driven_joint_count !== count) throw new Error(`驱动数 ${solved.driven_joint_count}`);
  if (solved.example_q.length !== count) throw new Error(`IK 解被截成 ${solved.example_q.length}`);
  if (solved.task_targets_reached !== 1) throw new Error(`${count} 模块没有解回自身末端`);
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
