import {
  B_TO_A,
  BASELINE_ORIENTATION,
  GRASP_DISTANCE,
  MIDDLE_ORIENTATION,
  NOMINAL,
  STRAIGHT,
  attachmentQuat,
  distanceToTool,
  footOrigin,
  modulePoint,
} from "./src/kinematics.js";

function near(actual, expected, tol, label) {
  const err = Math.max(...actual.map((value, index) => Math.abs(value - expected[index])));
  if (err > tol) {
    throw new Error(`${label} 误差 ${err}，实际 ${actual}，期望 ${expected}`);
  }
}

near(footOrigin(STRAIGHT), [2.12137298e-4, 0, 1.00478799], 1e-6, "伸直");
near(footOrigin(NOMINAL), [2.66836957e-4, 0.124544078, 0.455248373], 1e-6, "名义姿态");
const straight = distanceToTool(STRAIGHT);
const nominal = distanceToTool(NOMINAL);
if (!(straight > GRASP_DISTANCE && nominal < GRASP_DISTANCE)) {
  throw new Error(`距离不对：伸直 ${straight}，名义 ${nominal}`);
}

const baselineQuat = attachmentQuat(0);
near(baselineQuat, B_TO_A.quat, 0, "基准连接四元数");
const twisted = attachmentQuat(Math.PI / 2);
if (Math.max(...twisted.map((value, index) => Math.abs(value - B_TO_A.quat[index]))) < 1e-8) {
  throw new Error("中间连接没有旋转");
}

const variantFoot = footOrigin(STRAIGHT, MIDDLE_ORIENTATION);
const footShift = Math.hypot(
  variantFoot[0] - footOrigin(STRAIGHT)[0],
  variantFoot[1] - footOrigin(STRAIGHT)[1],
  variantFoot[2] - footOrigin(STRAIGHT)[2],
);
if (footShift < 5e-4) {
  throw new Error(`变体末端与基准重合：${footShift}`);
}
const local = [-0.07, 0, 0.03];
const proximal = modulePoint(STRAIGHT, BASELINE_ORIENTATION, 2, local);
near(modulePoint(STRAIGHT, MIDDLE_ORIENTATION, 2, local), proximal, 1e-12, "近端模块");
const baseTip = modulePoint(STRAIGHT, BASELINE_ORIENTATION, 5, local);
const variantTip = modulePoint(STRAIGHT, MIDDLE_ORIENTATION, 5, local);
const tipShift = Math.hypot(
  variantTip[0] - baseTip[0],
  variantTip[1] - baseTip[1],
  variantTip[2] - baseTip[2],
);
if (tipShift < 0.03) {
  throw new Error(`远端模块几乎没动：${tipShift}`);
}
const baseAngle = Math.atan2(baseTip[1], baseTip[0]);
const variantAngle = Math.atan2(variantTip[1], variantTip[0]);
let turn = Math.abs(variantAngle - baseAngle);
turn = Math.min(turn, 2 * Math.PI - turn);
if (Math.abs(turn - Math.PI / 2) > (20 * Math.PI) / 180) {
  throw new Error(`远端连接不是大约 90°，转过 ${(turn * 180) / Math.PI}°`);
}
console.log(`运动学核对通过。伸直 ${straight.toFixed(4)} m，名义 ${nominal.toFixed(4)} m，远端差 ${tipShift.toFixed(3)} m。`);
