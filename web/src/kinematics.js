/** 与 assets/zbot_6/ZBOT6.xml 一致的固定基座运动学。四元数顺序为 wxyz。
 *  五个模块间连接各取朝向 0 或 π/2。全 0 即基准构型，与 XML 的变换相同。
 *  公式与 src/zbot_exolimb_grasp/config_gen.py 一致。
 */

export const SHOULDER = [0, 0, 0.4];
export const STRAIGHT = [0, 0, 0, 0, 0, 0];
export const NOMINAL = [0.312, 0.837, -2.02, 2.02, -0.837, -0.312];
export const TOOL_POS = [-0.0003285042625797094, 0.12478776399807348, 0.3800370725014494];
export const TOOL_RADIUS = 0.022;
export const GRASP_DISTANCE = 0.08;
export const TRAY_HALF = [0.18, 0.14, 0.008];
export const TRAY_POS = [
  TOOL_POS[0],
  TOOL_POS[1],
  TOOL_POS[2] - TOOL_RADIUS - TRAY_HALF[2],
];

const A_TO_B_POS = [-0.00021213, 0, 0.054788];
const A_TO_B_QUAT = [0.923879181, 0, 0.382684281, 0];
const B_TO_A_POS = [-0.038890846, 0, 0.03919071];
const B_TO_A_QUAT = [0, -0.382684281, 0, 0.923879181];

function normalize(v) {
  const length = Math.hypot(v[0], v[1], v[2]);
  return [v[0] / length, v[1] / length, v[2] / length];
}

const CONNECTION_AXIS = normalize(B_TO_A_POS);

/** 从肩部数起第 4 个模块（XML 的 base）的入连接。 */
export const MIDDLE_INTERFACE = 2;
export const BASELINE_ORIENTATION = [0, 0, 0, 0, 0];
export const MIDDLE_ORIENTATION = [0, 0, Math.PI / 2, 0, 0];

export function add3(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
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

export function axisAngle(axis, angle) {
  const half = angle * 0.5;
  const sine = Math.sin(half);
  return [Math.cos(half), axis[0] * sine, axis[1] * sine, axis[2] * sine];
}

/** 连接朝向 φ。φ = 0 时返回 XML 中的 b→a 四元数。 */
export function attachmentQuat(phi) {
  return mulQuat(axisAngle(CONNECTION_AXIS, phi), B_TO_A_QUAT);
}

/** foot_1 机体原点。关节 6 绕该原点转动，不改变原点位置。 */
export function footOrigin(q, orientation = BASELINE_ORIENTATION) {
  let pos = SHOULDER.slice();
  let quat = [1, 0, 0, 0];
  for (let i = 0; i < 6; i += 1) {
    pos = add3(pos, rotate(quat, A_TO_B_POS));
    quat = mulQuat(quat, A_TO_B_QUAT);
    if (i === 5) {
      return pos;
    }
    quat = mulQuat(quat, hinge(q[i]));
    pos = add3(pos, rotate(quat, B_TO_A_POS));
    quat = mulQuat(quat, attachmentQuat(orientation[i]));
  }
  return pos;
}

/** 第 moduleIndex 个 b 连杆上的局部点（0 起算，关节角已作用）。 */
export function modulePoint(q, orientation, moduleIndex, local) {
  let pos = SHOULDER.slice();
  let quat = [1, 0, 0, 0];
  for (let i = 0; i < 6; i += 1) {
    pos = add3(pos, rotate(quat, A_TO_B_POS));
    quat = mulQuat(quat, A_TO_B_QUAT);
    quat = mulQuat(quat, hinge(q[i]));
    if (i === moduleIndex) {
      return add3(pos, rotate(quat, local));
    }
    pos = add3(pos, rotate(quat, B_TO_A_POS));
    quat = mulQuat(quat, attachmentQuat(orientation[i]));
  }
  return pos;
}

export function distanceToTool(q, orientation = BASELINE_ORIENTATION) {
  const foot = footOrigin(q, orientation);
  const dx = foot[0] - TOOL_POS[0];
  const dy = foot[1] - TOOL_POS[1];
  const dz = foot[2] - TOOL_POS[2];
  return Math.hypot(dx, dy, dz);
}

export const A_TO_B = { pos: A_TO_B_POS, quat: A_TO_B_QUAT };
export const B_TO_A = { pos: B_TO_A_POS, quat: B_TO_A_QUAT };
