import { distanceToTool, footOrigin, NOMINAL, STRAIGHT, GRASP_DISTANCE } from "./src/kinematics.js";

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
console.log(`运动学核对通过。伸直 ${straight.toFixed(4)} m，名义 ${nominal.toFixed(4)} m。`);
