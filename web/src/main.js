import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";
import {
  A_TO_B,
  B_TO_A,
  GRASP_DISTANCE,
  NOMINAL,
  SHOULDER,
  STRAIGHT,
  TOOL_POS,
  TOOL_RADIUS,
  TRAY_HALF,
  TRAY_POS,
  distanceToTool,
} from "./kinematics.js";

const statusEl = document.querySelector("#status");
const distanceEl = document.querySelector("#distance");
const jointRoot = document.querySelector("#joints");
const nominalButton = document.querySelector("#nominal");
const straightButton = document.querySelector("#straight");
const playButton = document.querySelector("#play");
const viewport = document.querySelector("#viewport");

const q = STRAIGHT.slice();
const sliders = [];
let playing = false;
let playStart = 0;

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
viewport.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xd5dde6);

const camera = new THREE.PerspectiveCamera(42, window.innerWidth / window.innerHeight, 0.01, 20);
camera.up.set(0, 0, 1);
camera.position.set(1.35, -1.45, 0.78);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0.0, 0.04, 0.68);
controls.enableDamping = true;
controls.update();

scene.add(new THREE.AmbientLight(0xffffff, 0.72));
const key = new THREE.DirectionalLight(0xffffff, 1.35);
key.position.set(0.8, -0.6, 1.6);
key.castShadow = true;
scene.add(key);
scene.add(new THREE.DirectionalLight(0xd6e4ff, 0.45).translateX(-1).translateZ(0.4));

const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(3.2, 3.2),
  new THREE.MeshStandardMaterial({ color: 0xd7dee6, roughness: 0.95 }),
);
floor.receiveShadow = true;
scene.add(floor);

const grid = new THREE.GridHelper(3.2, 32, 0xb7c2ce, 0xc9d2dc);
grid.rotation.x = Math.PI / 2;
scene.add(grid);

const tray = new THREE.Mesh(
  new THREE.BoxGeometry(TRAY_HALF[0] * 2, TRAY_HALF[1] * 2, TRAY_HALF[2] * 2),
  new THREE.MeshStandardMaterial({ color: 0x73563d, roughness: 0.72 }),
);
tray.position.set(...TRAY_POS);
tray.castShadow = true;
tray.receiveShadow = true;
scene.add(tray);

const tool = new THREE.Mesh(
  new THREE.SphereGeometry(TOOL_RADIUS, 48, 32),
  new THREE.MeshStandardMaterial({ color: 0xff7314, roughness: 0.38, metalness: 0.05 }),
);
tool.position.set(...TOOL_POS);
tool.castShadow = true;
scene.add(tool);

const aMaterial = new THREE.MeshStandardMaterial({
  color: new THREE.Color(0.89804, 0.92157, 0.92941),
  roughness: 0.42,
  metalness: 0.08,
});
const bMaterial = new THREE.MeshStandardMaterial({
  color: new THREE.Color(0.11702, 0.17021, 1),
  roughness: 0.38,
  metalness: 0.12,
});

let joints = [];

function quatOf(wxyz) {
  return new THREE.Quaternion(wxyz[1], wxyz[2], wxyz[3], wxyz[0]);
}

function addFrame(parent, pos, quat, geometry, material) {
  const frame = new THREE.Group();
  frame.position.set(pos[0], pos[1], pos[2]);
  frame.quaternion.copy(quatOf(quat));
  if (geometry) {
    const mesh = new THREE.Mesh(geometry, material);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    frame.add(mesh);
  }
  parent.add(frame);
  return frame;
}

function buildArm(aGeometry, bGeometry) {
  const root = new THREE.Group();
  let parent = addFrame(root, SHOULDER, [1, 0, 0, 0], aGeometry, aMaterial);
  joints = [];
  for (let index = 0; index < 6; index += 1) {
    const body = addFrame(parent, A_TO_B.pos, A_TO_B.quat, null, null);
    const joint = new THREE.Group();
    body.add(joint);
    const mesh = new THREE.Mesh(bGeometry, bMaterial);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    joint.add(mesh);
    joints.push(joint);
    if (index < 5) {
      parent = addFrame(joint, B_TO_A.pos, B_TO_A.quat, aGeometry, aMaterial);
    }
  }
  scene.add(root);
}

function setJoints(next, updateSliders = true) {
  for (let index = 0; index < 6; index += 1) {
    q[index] = next[index];
    if (joints[index]) {
      joints[index].rotation.z = next[index];
    }
    if (updateSliders && sliders[index]) {
      sliders[index].value = String(next[index]);
      sliders[index].nextElementSibling.value = next[index].toFixed(3);
    }
  }
  renderStatus();
}

function renderStatus() {
  const distance = distanceToTool(q);
  distanceEl.textContent = `末端原点距离 ${distance.toFixed(3)} m`;
  const inside = distance < GRASP_DISTANCE;
  statusEl.textContent = inside ? "进入抓取距离" : "尚未进入抓取距离";
  statusEl.classList.toggle("in-range", inside);
}

function stopPlayback() {
  playing = false;
  playButton.textContent = "播放接近演示";
}

function poseFromSliders() {
  setJoints(sliders.map((slider) => Number(slider.value)), false);
}

for (let index = 0; index < 6; index += 1) {
  const row = document.createElement("label");
  row.className = "joint";
  const name = document.createElement("span");
  name.textContent = `关节 ${index + 1}`;
  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = "-6.28";
  slider.max = "6.28";
  slider.step = "0.001";
  slider.value = "0";
  const output = document.createElement("output");
  output.value = "0.000";
  slider.addEventListener("input", () => {
    stopPlayback();
    output.value = Number(slider.value).toFixed(3);
    poseFromSliders();
  });
  row.append(name, slider, output);
  jointRoot.append(row);
  sliders.push(slider);
}

nominalButton.addEventListener("click", () => {
  stopPlayback();
  setJoints(NOMINAL);
});
straightButton.addEventListener("click", () => {
  stopPlayback();
  setJoints(STRAIGHT);
});
playButton.addEventListener("click", () => {
  if (playing) {
    stopPlayback();
    return;
  }
  playing = true;
  playStart = performance.now();
  playButton.textContent = "停止演示";
});

function frame(now) {
  if (playing) {
    const t = Math.min(1, (now - playStart) / 3200);
    const s = t * t * (3 - 2 * t);
    const next = STRAIGHT.map((start, index) => start + (NOMINAL[index] - start) * s);
    setJoints(next);
    if (t >= 1) {
      stopPlayback();
    }
  }
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}

const loader = new STLLoader();
const base = import.meta.env.BASE_URL;
Promise.all([
  loader.loadAsync(`${base}meshes/a.STL`),
  loader.loadAsync(`${base}meshes/b.STL`),
]).then(([aGeometry, bGeometry]) => {
  aGeometry.computeVertexNormals();
  bGeometry.computeVertexNormals();
  buildArm(aGeometry, bGeometry);
  setJoints(STRAIGHT);
  statusEl.textContent = "尚未进入抓取距离";
}).catch(() => {
  statusEl.textContent = "模块网格加载失败";
});

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

requestAnimationFrame(frame);
setJoints(STRAIGHT);
