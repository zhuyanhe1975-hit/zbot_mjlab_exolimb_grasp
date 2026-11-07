/** 浏览器侧 MJCF。结构与 Python exporting.export_mjcf 一致，供下载后用同一运动学核对。 */

import { zbotAttachment } from "./model.js";

const A_POS = [-0.00021213, 0, 0.054788];
const A_QUAT = [0.923879181, 0, 0.382684281, 0];
const B_POS = [-0.038890846, 0, 0.03919071];
const COM_A = [-0.0022495, -0.000029534, 0.042888];
const COM_B = [-0.02292, -0.00016188, 0.03226];

function fmt(values) {
  return values.map((value) => Number(value).toPrecision(8)).join(" ");
}

function axisAngleX(angle) {
  const half = angle * 0.5;
  return [Math.cos(half), Math.sin(half), 0, 0];
}

export function exportMjcf(design) {
  const lines = ['<mujoco model="modular_arm">', '  <compiler angle="radian"/>', "  <worldbody>"];
  const indent = (level, text) => lines.push(`${"  ".repeat(level)}${text}`);
  design.module_ids.forEach((moduleId, index) => {
    const last = index === design.module_ids.length - 1;
    const twist = last ? 0 : design.twists_rad[index];
    const level = 2 + index * 2;
    if (moduleId === "zbot_r3") {
      const pos = index === 0 ? ` pos="${fmt(design.base_position_m)}" quat="1 0 0 0"` : "";
      indent(level, `<body name="m${index}_a"${pos}>`);
      indent(level + 1, `<inertial pos="${fmt(COM_A)}" mass="0.54171"/>`);
      indent(level + 1, `<body name="m${index}_b" pos="${fmt(A_POS)}" quat="${fmt(A_QUAT)}">`);
      indent(level + 2, `<joint name="joint${index + 1}" type="hinge" axis="0 0 1" range="-6.28 6.28"/>`);
      indent(level + 2, `<inertial pos="${fmt(COM_B)}" mass="0.17138"/>`);
      if (last) indent(level + 2, '<site name="ee" pos="0 0 0"/>');
      else {
        indent(level + 2, `<body name="m${index + 1}_mount" pos="${fmt(B_POS)}" quat="${fmt(zbotAttachment(twist))}">`);
      }
    } else {
      const length = design.parameters[index].length_m;
      const mass = 4 * length;
      const pos = index === 0 ? ` pos="${fmt(design.base_position_m)}" quat="1 0 0 0"` : "";
      indent(level, `<body name="m${index}_axial"${pos}>`);
      indent(level + 1, `<joint name="joint${index + 1}" type="hinge" axis="0 0 1" range="-6.28 6.28"/>`);
      indent(level + 1, `<inertial pos="${fmt([length / 2, 0, 0])}" mass="${mass.toPrecision(8)}"/>`);
      if (last) indent(level + 1, `<site name="ee" pos="${fmt([length, 0, 0])}"/>`);
      else indent(level + 1, `<body name="m${index + 1}_mount" pos="${fmt([length, 0, 0])}" quat="${fmt(axisAngleX(twist))}">`);
    }
  });
  for (let index = design.module_ids.length - 1; index >= 0; index -= 1) {
    const level = 2 + index * 2;
    const last = index === design.module_ids.length - 1;
    if (!last) indent(level + 2, "</body>");
    if (design.module_ids[index] === "zbot_r3") {
      indent(level + 1, "</body>");
      indent(level, "</body>");
    } else indent(level, "</body>");
  }
  lines.push("  </worldbody>", "</mujoco>", "");
  return lines.join("\n");
}

export function candidatesCsv(result) {
  const header = ["index", "modules", "mass_kg", "task_feasible_fraction", "task_target_count", "task_targets_reached", "manipulability", "velocity", "pose_hit_rate"];
  const rows = [result.baseline, ...result.feasible_pareto].map((row, index) => [
    index,
    row.design.module_ids ? row.design.module_ids.join(" ") : row.design.modules.map((item) => item.id).join(" "),
    row.metrics.mass_kg.toFixed(6),
    row.metrics.task_feasible_fraction.toFixed(6),
    row.metrics.task_target_count,
    row.metrics.task_targets_reached,
    row.metrics.task_manipulability_median == null ? "" : row.metrics.task_manipulability_median.toFixed(8),
    row.metrics.task_velocity_median == null ? "" : row.metrics.task_velocity_median.toFixed(6),
    row.metrics.pose_hit_rate == null ? "" : row.metrics.pose_hit_rate.toFixed(6),
  ]);
  return [header.join(","), ...rows.map((row) => row.join(","))].join("\n") + "\n";
}
