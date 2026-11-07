/** 与 Python platform.model 对齐的串联模型。四元数 wxyz。浏览器里做真实评价，不读取预计算表。 */

export const COUNT_MIN = 3;
export const COUNT_MAX = 8;
export const JOINT_LIMIT = 6.28;
export const MASS_A = 0.54171;
export const MASS_B = 0.17138;
export const CONCEPT_DENSITY = 4;
export const CONCEPT_RADIUS = 0.025;
export const ZBOT_RADIUS = 0.04;
const A_POS = [-0.00021213, 0, 0.054788];
const A_QUAT = [0.923879181, 0, 0.382684281, 0];
const B_POS = [-0.038890846, 0, 0.03919071];
const B_QUAT = [0, -0.382684281, 0, 0.923879181];
const COM_A = [-0.0022495, -0.000029534, 0.042888];
const COM_B = [-0.02292, -0.00016188, 0.03226];
const Z = [0, 0, 1];

export const LIBRARY = {
  zbot_r3: {
    id: "zbot_r3",
    label: "ZBot R3 关节模块",
    kind: "physical",
    summary: "尺寸、质量和行程来自 R3 的 ZBOT6。实物模式锁定尺寸。",
    in_family: "zbot_flange",
    out_family: "zbot_flange",
    parameters: [],
  },
  concept_axial: {
    id: "concept_axial",
    label: "轴向概念连杆",
    kind: "conceptual",
    summary: "接口族按 zbot_flange 建模，所以可以插入 ZBot 链。长度和质量是假设，不是第二套硬件。",
    in_family: "zbot_flange",
    out_family: "zbot_flange",
    parameters: [{ name: "length_m", unit: "m", lower: 0.06, upper: 0.16, default: 0.1, label: "连杆长度" }],
  },
  demo_unreleased: {
    id: "demo_unreleased",
    label: "未接入接口示例",
    kind: "conceptual",
    summary: "demo_flange 与 zbot_flange 没有装配规则，不能和前两个模块串联。",
    in_family: "demo_flange",
    out_family: "demo_flange",
    parameters: [{ name: "length_m", unit: "m", lower: 0.08, upper: 0.12, default: 0.1, label: "示例长度" }],
  },
};

const RULES = [
  { parent: "zbot_flange", child: "zbot_flange", angles: [0, Math.PI / 2] },
  { parent: "demo_flange", child: "demo_flange", angles: [0, Math.PI / 2] },
];

export const UNSUPPORTED = [
  "网格精确碰撞",
  "完整逆动力学与加速度能力",
  "转动可操作度（不与平移指标混单位）",
  "树形或闭环拓扑",
];

function add(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}
function sub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
function scale(v, s) {
  return [v[0] * s, v[1] * s, v[2] * s];
}
function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function norm(v) {
  return Math.hypot(v[0], v[1], v[2]);
}
export function mulQuat(a, b) {
  const [aw, ax, ay, az] = a;
  const [bw, bx, by, bz] = b;
  return [
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
  ];
}
export function rotate(q, v) {
  const [, x, y, z] = q;
  const [vx, vy, vz] = v;
  const tx = 2 * (y * vz - z * vy);
  const ty = 2 * (z * vx - x * vz);
  const tz = 2 * (x * vy - y * vx);
  return [
    vx + q[0] * tx + (y * tz - z * ty),
    vy + q[0] * ty + (z * tx - x * tz),
    vz + q[0] * tz + (x * ty - y * tx),
  ];
}
function hinge(angle) {
  const half = angle * 0.5;
  return [Math.cos(half), 0, 0, Math.sin(half)];
}
function axisAngleX(angle) {
  const half = angle * 0.5;
  return [Math.cos(half), Math.sin(half), 0, 0];
}
function normalize(v) {
  const n = norm(v);
  return [v[0] / n, v[1] / n, v[2] / n];
}
const CONNECTION_AXIS = normalize(B_POS);

export function zbotAttachment(phi) {
  const half = phi * 0.5;
  const sine = Math.sin(half);
  const twist = [Math.cos(half), CONNECTION_AXIS[0] * sine, CONNECTION_AXIS[1] * sine, CONNECTION_AXIS[2] * sine];
  return mulQuat(twist, B_QUAT);
}

function mate(parentId, childId) {
  const parent = LIBRARY[parentId];
  const child = LIBRARY[childId];
  if (!parent || !child) return null;
  return RULES.find((rule) => rule.parent === parent.out_family && rule.child === child.in_family) ?? null;
}

export function assemblyErrors(design) {
  const errors = [];
  const ids = design.module_ids ?? [];
  const count = ids.length;
  if (count < COUNT_MIN || count > COUNT_MAX) errors.push(`模块数量 ${count} 超出 ${COUNT_MIN}–${COUNT_MAX}`);
  if ((design.twists_rad ?? []).length !== Math.max(count - 1, 0)) errors.push("连接方位的数量必须比模块少 1");
  if ((design.parameters ?? []).length !== count) errors.push("每个模块都要有一份参数");
  ids.forEach((moduleId, index) => {
    const spec = LIBRARY[moduleId];
    if (!spec) {
      errors.push(`第 ${index + 1} 个模块 ${moduleId} 不在模块库中`);
      return;
    }
    const params = design.parameters[index] ?? {};
    if (spec.kind === "physical" && Object.keys(params).length) errors.push(`${spec.label} 是实物模块，不能带尺寸参数`);
    spec.parameters.forEach((item) => {
      const value = params[item.name];
      if (!(value >= item.lower - 1e-9 && value <= item.upper + 1e-9)) {
        errors.push(`第 ${index + 1} 个模块的 ${item.label} 超出 ${item.lower}–${item.upper} ${item.unit}`);
      }
    });
    if (index < count - 1) {
      const rule = mate(moduleId, ids[index + 1]);
      if (!rule) errors.push(`第 ${index + 1} 与 ${index + 2} 个模块的接口族不能连接`);
      else if (!rule.angles.some((angle) => Math.abs(angle - design.twists_rad[index]) < 1e-8)) {
        errors.push(`第 ${index + 1} 个连接方位不在合法集合内`);
      }
    }
  });
  return errors;
}

export function forwardDesign(design) {
  const errors = assemblyErrors(design);
  if (errors.length) throw new Error(errors.join("；"));
  const q = arguments[1];
  const count = design.module_ids.length;
  let pos = design.base_position_m.slice();
  let quat = [1, 0, 0, 0];
  const origins = [];
  const axes = [];
  const bodies = [];
  const capsules = [];
  let mass = 0;
  let ee = null;
  for (let index = 0; index < count; index += 1) {
    const moduleId = design.module_ids[index];
    const hasChild = index < count - 1;
    const twist = hasChild ? design.twists_rad[index] : 0;
    if (moduleId === "zbot_r3") {
      const origin = add(pos, rotate(quat, A_POS));
      const jointQuat = mulQuat(quat, A_QUAT);
      const axis = rotate(jointQuat, Z);
      const swung = mulQuat(jointQuat, hinge(q[index]));
      const comA = add(pos, rotate(quat, COM_A));
      const comB = add(origin, rotate(swung, COM_B));
      const bEnd = add(origin, rotate(swung, B_POS));
      origins.push(origin);
      axes.push(axis);
      bodies.push({ mass: MASS_A, com: comA, module: index, after: false });
      bodies.push({ mass: MASS_B, com: comB, module: index, after: true });
      mass += MASS_A + MASS_B;
      capsules.push({ p0: pos.slice(), p1: origin, r: ZBOT_RADIUS, module: index });
      capsules.push({ p0: origin, p1: bEnd, r: ZBOT_RADIUS, module: index });
      if (hasChild) {
        pos = bEnd;
        quat = mulQuat(swung, zbotAttachment(twist));
      } else ee = origin;
    } else {
      const length = design.parameters[index].length_m;
      const origin = pos.slice();
      const axis = rotate(quat, Z);
      const post = mulQuat(quat, hinge(q[index]));
      const tip = add(origin, rotate(post, [length, 0, 0]));
      const com = add(origin, rotate(post, [length * 0.5, 0, 0]));
      const linkMass = CONCEPT_DENSITY * length;
      origins.push(origin);
      axes.push(axis);
      bodies.push({ mass: linkMass, com, module: index, after: true });
      mass += linkMass;
      capsules.push({ p0: origin, p1: tip, r: CONCEPT_RADIUS, module: index });
      if (hasChild) {
        pos = tip;
        quat = mulQuat(post, axisAngleX(twist));
      } else ee = tip;
    }
  }
  const jacobian = origins.map((origin, index) => cross(axes[index], sub(ee, origin)));
  return { ee, jacobian, mass, origins, axes, bodies, capsules };
}

function segmentDistance(p0, p1, q0, q1) {
  const d1 = sub(p1, p0);
  const d2 = sub(q1, q0);
  const r = sub(p0, q0);
  const a = dot(d1, d1);
  const e = dot(d2, d2);
  const f = dot(d2, r);
  const c = dot(d1, r);
  const b = dot(d1, d2);
  const eps = 1e-12;
  let s = 0;
  let t = 0;
  const denom = a * e - b * b;
  if (a <= eps && e <= eps) return norm(sub(p0, q0));
  if (a <= eps) t = Math.min(1, Math.max(0, f / e));
  else if (e <= eps) s = Math.min(1, Math.max(0, -c / a));
  else {
    if (denom > eps) s = Math.min(1, Math.max(0, (b * f - c * e) / denom));
    t = (b * s + f) / e;
    if (t < 0) {
      t = 0;
      s = Math.min(1, Math.max(0, -c / a));
    } else if (t > 1) {
      t = 1;
      s = Math.min(1, Math.max(0, (b - c) / a));
    }
  }
  const c1 = add(p0, scale(d1, s));
  const c2 = add(q0, scale(d2, t));
  return norm(sub(c1, c2));
}

function collides(chain) {
  for (let i = 0; i < chain.capsules.length; i += 1) {
    for (let j = i + 1; j < chain.capsules.length; j += 1) {
      const a = chain.capsules[i];
      const b = chain.capsules[j];
      if (Math.abs(a.module - b.module) < 2) continue;
      if (segmentDistance(a.p0, a.p1, b.p0, b.p1) < a.r + b.r - 1e-6) return true;
    }
  }
  return false;
}

function torques(chain, payload, gravity = 9.81) {
  const weight = [0, 0, -gravity];
  return chain.origins.map((origin, joint) => {
    let tau = 0;
    chain.bodies.forEach((body) => {
      const distal = body.module > joint || (body.module === joint && body.after);
      if (!distal) return;
      tau += dot(chain.axes[joint], cross(sub(body.com, origin), scale(weight, body.mass)));
    });
    tau += dot(chain.axes[joint], cross(sub(chain.ee, origin), scale(weight, payload)));
    return tau;
  });
}

function det3(m) {
  return (
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
    m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
    m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
  );
}

function manipulability(jacobian) {
  const n = jacobian.length ? jacobian[0].length : 0;
  const g = [0, 1, 2].map((row) => [0, 1, 2].map((col) => {
    let sum = 0;
    for (let k = 0; k < n; k += 1) sum += jacobian[row][k] * jacobian[col][k];
    return sum;
  }));
  return Math.sqrt(Math.max(det3(g), 0));
}

function solve3(m, b) {
  const det = det3(m);
  if (Math.abs(det) < 1e-12) return null;
  const replace = (col) => m.map((row, index) => row.map((value, j) => (j === col ? b[index] : value)));
  return [0, 1, 2].map((col) => det3(replace(col)) / det);
}

function velocityCapability(jacobian, limit) {
  const columns = jacobian[0]?.length ?? 0;
  const rows = jacobian;
  let worst = Infinity;
  for (let axis = 0; axis < 3; axis += 1) {
    const direction = [0, 0, 0];
    direction[axis] = 1;
    const gram = [0, 1, 2].map((row) => [0, 1, 2].map((col) => {
      let sum = 0;
      for (let k = 0; k < columns; k += 1) sum += rows[row][k] * rows[col][k];
      return sum;
    }));
    const lambda = solve3(gram, direction);
    if (!lambda) return 0;
    const qdot = Array.from({ length: columns }, (_, index) =>
      rows[0][index] * lambda[0] + rows[1][index] * lambda[1] + rows[2][index] * lambda[2]);
    let residual = 0;
    for (let row = 0; row < 3; row += 1) {
      let acc = 0;
      for (let k = 0; k < columns; k += 1) acc += rows[row][k] * qdot[k];
      residual += (acc - direction[row]) ** 2;
    }
    if (Math.sqrt(residual) > 1e-4) return 0;
    const peak = Math.max(...qdot.map((value) => Math.abs(value)));
    worst = Math.min(worst, peak < 1e-12 ? 0 : limit / peak);
  }
  return Number.isFinite(worst) ? worst : 0;
}

function insideBox(point, task) {
  return task.center_m.every((value, index) => Math.abs(point[index] - value) <= task.half_m[index] + 1e-12);
}

function jacobianMatrix(chain) {
  const n = chain.origins.length;
  return [0, 1, 2].map((row) => chain.jacobian.map((column) => column[row]).slice(0, n));
}

export function evaluateDesign(design, task, samples, includeCloud = false) {
  const count = design.module_ids.length;
  const poses = samples ?? [Array(count).fill(0)];
  const straight = forwardDesign(design, Array(count).fill(0));
  const torque0 = torques(straight, task.payload_kg);
  const peak0 = Math.max(...torque0.map((value) => Math.abs(value)));
  const collide0 = collides(straight);
  const torque0Ok = peak0 <= task.effort_limit_nm + 1e-9;
  let insideCount = 0;
  let collideCount = 0;
  let torqueFail = 0;
  const manips = [];
  const speeds = [];
  let example = Array(count).fill(0);
  let found = false;
  const cloud = [];
  poses.forEach((pose, index) => {
    const chain = forwardDesign(design, pose);
    const hit = collides(chain);
    const tau = torques(chain, task.payload_kg);
    const badTorque = tau.some((value) => Math.abs(value) > task.effort_limit_nm + 1e-9);
    const inside = insideBox(chain.ee, task);
    if (inside) insideCount += 1;
    if (hit) collideCount += 1;
    if (badTorque) torqueFail += 1;
    const matrix = jacobianMatrix(chain);
    manips.push(manipulability(matrix));
    speeds.push(velocityCapability(matrix, task.velocity_limit_rad_s));
    if (!found && inside && !hit && !badTorque) {
      example = pose.slice();
      found = true;
    }
    if (includeCloud && index % Math.max(1, Math.floor(poses.length / 120)) === 0 && cloud.length < 120) {
      cloud.push({ p: chain.ee.slice(), inside, collide: hit, torque_ok: !badTorque });
    }
  });
  const n = poses.length;
  const coverage = insideCount / n;
  const collisionRate = collideCount / n;
  const torqueFailRate = torqueFail / n;
  const limit = task.rate_limit ?? 0.85;
  let violation = (collide0 ? 1 : 0) + (torque0Ok ? 0 : 1);
  violation += Math.max(0, collisionRate - limit) + Math.max(0, torqueFailRate - limit);
  const sortedManip = manips.slice().sort((a, b) => a - b);
  const sortedSpeed = speeds.slice().sort((a, b) => a - b);
  const mid = (values) => values.length % 2 ? values[(values.length - 1) / 2] : 0.5 * (values[values.length / 2 - 1] + values[values.length / 2]);
  return {
    mass_kg: straight.mass,
    task_coverage: coverage,
    coverage_se: Math.sqrt(Math.max(coverage * (1 - coverage), 0) / n),
    sample_count: n,
    median_translational_manipulability: mid(sortedManip),
    median_velocity_capability_m_s: mid(sortedSpeed),
    collision_rate: collisionRate,
    torque_fail_rate: torqueFailRate,
    q0_collision: collide0,
    q0_torque_ok: torque0Ok,
    q0_peak_torque_nm: peak0,
    violation,
    feasible: violation === 0,
    example_q: example,
    ee_straight: straight.ee,
    unsupported: UNSUPPORTED.slice(),
    cloud: includeCloud ? cloud : undefined,
  };
}

export function baselineDesign() {
  return {
    module_ids: Array(6).fill("zbot_r3"),
    twists_rad: Array(5).fill(0),
    parameters: Array.from({ length: 6 }, () => ({})),
    base_position_m: [0, 0, 0.4],
  };
}

export function mulberry32(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

export function uniformSamples(count, joints, limit, seed) {
  const rand = mulberry32(seed >>> 0);
  return Array.from({ length: count }, () => Array.from({ length: joints }, () => (rand() * 2 - 1) * limit));
}
