import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";
import { candidatesCsv, exportMjcf } from "./export.js";
import { LIBRARY, assemblyErrors, forwardDesign, zbotAttachment } from "./model.js";

const A_POS = [-0.00021213, 0, 0.054788];
const A_QUAT = [0.923879181, 0, 0.382684281, 0];
const B_POS = [-0.038890846, 0, 0.03919071];

const libraryEl = document.querySelector("#library");
const chainEl = document.querySelector("#chain");
const warnEl = document.querySelector("#assembly-warn");
const modeHelp = document.querySelector("#mode-help");
const progressEl = document.querySelector("#progress");
const candidatesEl = document.querySelector("#candidates");
const detailEl = document.querySelector("#detail");
const jointRoot = document.querySelector("#joints");
const paretoCanvas = document.querySelector("#pareto");
const runButton = document.querySelector("#run");
const cancelButton = document.querySelector("#cancel");

const state = {
  count: 4,
  rows: [
    { id: "zbot_r3", twist: 0, length: 0.1 },
    { id: "zbot_r3", twist: 0, length: 0.1 },
    { id: "concept_axial", twist: 0, length: 0.1 },
    { id: "zbot_r3", twist: 0, length: 0.1 },
  ],
  q: [0, 0, 0, 0],
  result: null,
  selected: -1,
};

let aGeometry = null;
let bGeometry = null;
let armRoot = null;
let cloudPoints = null;
let worker = null;
const pivots = [];

function mode() {
  return document.querySelector("input[name=mode]:checked").value;
}
function num(id) {
  return Number(document.querySelector(id).value);
}
function designFromState() {
  const count = state.count;
  const rows = state.rows.slice(0, count);
  return {
    module_ids: rows.map((row) => (mode() === "physical" ? "zbot_r3" : row.id)),
    twists_rad: rows.slice(0, -1).map((row) => row.twist),
    parameters: rows.map((row) => (mode() === "mixed" && LIBRARY[row.id]?.parameters.length ? { length_m: row.length } : {})),
    base_position_m: [0, 0, 0.4],
  };
}
function specFromState() {
  const objectives = [...document.querySelectorAll("#objectives input:checked")].map((input) => input.value);
  return {
    mode: mode(),
    count_min: num("#count-min"),
    count_max: num("#count-max"),
    count_lock: document.querySelector("#count-lock").checked ? state.count : null,
    population: num("#pop"),
    generations: num("#gen"),
    objectives,
    length_lower_m: num("#len-min"),
    length_upper_m: num("#len-max"),
    seed: num("#seed"),
    task: {
      center_m: [num("#cx"), num("#cy"), num("#cz")],
      half_m: [num("#hx"), num("#hy"), num("#hz")],
      payload_kg: num("#payload"),
      effort_limit_nm: num("#effort"),
      velocity_limit_rad_s: num("#velocity"),
      sample_count: num("#samples"),
      seed: num("#seed"),
      rate_limit: 0.85,
    },
  };
}

function renderLibrary() {
  libraryEl.replaceChildren();
  Object.values(LIBRARY).forEach((spec) => {
    const card = document.createElement("article");
    card.className = "card";
    const kind = spec.kind === "physical" ? "实物" : "概念";
    card.innerHTML = `<span class="badge ${spec.kind === "conceptual" ? "concept" : ""}">${kind}</span><strong>${spec.label}</strong><span>${spec.in_family} → ${spec.out_family}</span><p class="help">${spec.summary}</p>`;
    libraryEl.append(card);
  });
}

function renderChain() {
  chainEl.replaceChildren();
  const physical = mode() === "physical";
  modeHelp.textContent = physical
    ? "只使用 ZBot R3。搜索可以改变模块数量和连接方位，不能改变实物尺寸。"
    : "可以在 ZBot R3 和轴向概念连杆之间选择类型、次序和长度。未接入接口示例不能放进这条链。";
  for (let index = 0; index < state.count; index += 1) {
    const row = state.rows[index];
    const box = document.createElement("div");
    box.className = "module-row";
    const type = document.createElement("select");
    ["zbot_r3", "concept_axial"].forEach((id) => {
      const option = document.createElement("option");
      option.value = id;
      option.textContent = LIBRARY[id].label;
      type.append(option);
    });
    type.value = physical ? "zbot_r3" : row.id;
    type.disabled = physical;
    type.addEventListener("change", () => {
      row.id = type.value;
      refresh();
    });
    const twist = document.createElement("select");
    [["0", "连接 0°"], [String(Math.PI / 2), "连接 90°"]].forEach(([value, label]) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      twist.append(option);
    });
    twist.value = String(row.twist);
    twist.disabled = index === state.count - 1;
    twist.addEventListener("change", () => {
      row.twist = Number(twist.value);
      refresh();
    });
    const editableLength = !physical && row.id === "concept_axial";
    let lengthField;
    if (editableLength) {
      lengthField = document.createElement("input");
      lengthField.type = "number";
      lengthField.min = "0.06";
      lengthField.max = "0.16";
      lengthField.step = "0.01";
      lengthField.value = String(row.length);
      lengthField.addEventListener("change", () => {
        row.length = Number(lengthField.value);
        refresh();
      });
    } else {
      lengthField = document.createElement("span");
      lengthField.className = "locked-size";
      lengthField.textContent = "尺寸锁定";
    }
    box.append(type, twist, lengthField);
    chainEl.append(box);
  }
}

function renderJoints() {
  jointRoot.replaceChildren();
  for (let index = 0; index < state.count; index += 1) {
    const row = document.createElement("label");
    row.className = "joint";
    const name = document.createElement("span");
    name.textContent = `关节 ${index + 1}`;
    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = "-6.28";
    slider.max = "6.28";
    slider.step = "0.001";
    slider.value = String(state.q[index] ?? 0);
    const output = document.createElement("output");
    output.value = Number(slider.value).toFixed(3);
    slider.addEventListener("input", () => {
      state.q[index] = Number(slider.value);
      output.value = state.q[index].toFixed(3);
      poseArm();
    });
    row.append(name, slider, output);
    jointRoot.append(row);
  }
}

function currentDesign() {
  const design = designFromState();
  const errors = assemblyErrors(design);
  warnEl.hidden = errors.length === 0;
  warnEl.textContent = errors.join(" ");
  return { design, errors };
}

function refresh() {
  document.querySelector("#count-now").value = String(state.count);
  while (state.rows.length < state.count) state.rows.push({ id: "zbot_r3", twist: 0, length: 0.1 });
  state.q = Array.from({ length: state.count }, (_, index) => state.q[index] ?? 0);
  renderChain();
  renderJoints();
  mountArm();
}

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
    frame.add(mesh);
  }
  parent.add(frame);
  return frame;
}

const aMaterial = new THREE.MeshStandardMaterial({ color: new THREE.Color(0.89804, 0.92157, 0.92941), roughness: 0.42 });
const bMaterial = new THREE.MeshStandardMaterial({ color: new THREE.Color(0.11702, 0.17021, 1), roughness: 0.38 });
const conceptMaterial = new THREE.MeshStandardMaterial({ color: 0xd9782d, roughness: 0.55 });

function mountArm() {
  if (!aGeometry) return;
  const { design, errors } = currentDesign();
  if (armRoot) scene.remove(armRoot);
  armRoot = new THREE.Group();
  if (!errors.length) {
    let parent = new THREE.Group();
    parent.position.set(...design.base_position_m);
    armRoot.add(parent);
    pivots.length = 0;
    design.module_ids.forEach((moduleId, index) => {
      const last = index === design.module_ids.length - 1;
      if (moduleId === "zbot_r3") {
        addFrame(parent, [0, 0, 0], [1, 0, 0, 0], aGeometry, aMaterial);
        const body = addFrame(parent, A_POS, A_QUAT);
        const joint = new THREE.Group();
        body.add(joint);
        const mesh = new THREE.Mesh(bGeometry, bMaterial);
        mesh.castShadow = true;
        joint.add(mesh);
        pivots.push(joint);
        if (!last) parent = addFrame(joint, B_POS, zbotAttachment(design.twists_rad[index]), null, null);
      } else {
        const length = design.parameters[index].length_m;
        const joint = new THREE.Group();
        parent.add(joint);
        const geometry = new THREE.CylinderGeometry(0.025, 0.025, length, 20);
        geometry.rotateZ(Math.PI / 2);
        geometry.translate(length / 2, 0, 0);
        const mesh = new THREE.Mesh(geometry, conceptMaterial);
        mesh.castShadow = true;
        joint.add(mesh);
        pivots.push(joint);
        if (!last) {
          const frame = new THREE.Group();
          frame.position.set(length, 0, 0);
          frame.quaternion.copy(quatOf([Math.cos(design.twists_rad[index] / 2), Math.sin(design.twists_rad[index] / 2), 0, 0]));
          joint.add(frame);
          parent = frame;
        }
      }
    });
  }
  scene.add(armRoot);
  poseArm();
  drawTask();
}

function poseArm() {
  pivots.forEach((joint, index) => {
    joint.rotation.z = state.q[index] ?? 0;
  });
}

function drawTask() {
  const old = scene.getObjectByName("task");
  if (old) scene.remove(old);
  const box = new THREE.Mesh(
    new THREE.BoxGeometry(num("#hx") * 2, num("#hy") * 2, num("#hz") * 2),
    new THREE.MeshBasicMaterial({ color: 0x1e2bff, wireframe: true, transparent: true, opacity: 0.7 }),
  );
  box.name = "task";
  box.position.set(num("#cx"), num("#cy"), num("#cz"));
  scene.add(box);
}

function showCloud(metrics) {
  if (cloudPoints) scene.remove(cloudPoints);
  const cloud = metrics?.cloud ?? [];
  const positions = new Float32Array(cloud.length * 3);
  const colors = new Float32Array(cloud.length * 3);
  cloud.forEach((point, index) => {
    positions.set(point.p, index * 3);
    const color = point.collide ? [0.8, 0.25, 0.2] : point.inside ? [0.15, 0.45, 0.95] : [0.55, 0.6, 0.66];
    colors.set(color, index * 3);
  });
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  cloudPoints = new THREE.Points(geometry, new THREE.PointsMaterial({ size: 0.012, vertexColors: true }));
  scene.add(cloudPoints);
}

function describe(metrics) {
  if (!metrics) return "还没有采样指标。";
  return [
    `质量 ${metrics.mass_kg.toFixed(3)} kg`,
    `覆盖率 ${(metrics.task_coverage * 100).toFixed(1)}%（标准误差 ±${(metrics.coverage_se * 100).toFixed(1)} 个百分点）`,
    `平移可操作度中位数 ${metrics.median_translational_manipulability.toFixed(5)}`,
    `速度能力中位数 ${metrics.median_velocity_capability_m_s.toFixed(3)} m/s（速度上限是假设）`,
    `自碰撞样本 ${(metrics.collision_rate * 100).toFixed(0)}%，静力矩超限 ${(metrics.torque_fail_rate * 100).toFixed(0)}%`,
    metrics.feasible ? "约束满足，可以进入可行前沿。" : "约束不满足，不会进入可行前沿。",
    metrics.q0_collision ? "伸直姿态胶囊相交。" : "伸直姿态没有非相邻胶囊相交。",
    `伸直姿态峰值静力矩 ${metrics.q0_peak_torque_nm.toFixed(3)} N·m。`,
  ].join("\n");
}

function drawPareto() {
  const ctx = paretoCanvas.getContext("2d");
  const width = paretoCanvas.width;
  const height = paretoCanvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#f8f9fb";
  ctx.fillRect(0, 0, width, height);
  const rows = state.result?.feasible_pareto ?? [];
  if (!rows.length) {
    ctx.fillStyle = "#5c6775";
    ctx.font = "13px sans-serif";
    ctx.fillText("可行前沿会画在这里", 16, 28);
    return;
  }
  const xs = rows.map((row) => row.metrics.task_coverage);
  const ys = rows.map((row) => row.metrics.mass_kg);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const px = (value) => 28 + ((value - minX) / (maxX - minX || 1)) * (width - 44);
  const py = (value) => height - 24 - ((value - minY) / (maxY - minY || 1)) * (height - 40);
  ctx.strokeStyle = "#c5ced8";
  ctx.strokeRect(28, 12, width - 40, height - 32);
  rows.forEach((row, index) => {
    ctx.beginPath();
    ctx.fillStyle = index === state.selected ? "#1e2bff" : "#e36a12";
    ctx.arc(px(row.metrics.task_coverage), py(row.metrics.mass_kg), 5, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.fillStyle = "#5c6775";
  ctx.font = "11px sans-serif";
  ctx.fillText("覆盖率 →", width - 70, height - 6);
  ctx.fillText("质量", 4, 16);
  paretoCanvas.onclick = (event) => {
    const rect = paretoCanvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) * (paretoCanvas.width / rect.width);
    const y = (event.clientY - rect.top) * (paretoCanvas.height / rect.height);
    let best = 0;
    let bestDist = Infinity;
    rows.forEach((row, index) => {
      const dx = px(row.metrics.task_coverage) - x;
      const dy = py(row.metrics.mass_kg) - y;
      const dist = dx * dx + dy * dy;
      if (dist < bestDist) {
        bestDist = dist;
        best = index;
      }
    });
    if (bestDist < 400) selectCandidate(best);
  };
}

function selectCandidate(index) {
  state.selected = index;
  const row = state.result?.feasible_pareto?.[index];
  drawPareto();
  [...candidatesEl.children].forEach((child, childIndex) => child.classList.toggle("active", childIndex === index));
  if (!row) return;
  const modules = row.design.module_ids ?? row.design.modules.map((item) => item.id);
  state.count = modules.length;
  state.rows = modules.map((id, moduleIndex) => ({
    id,
    twist: row.design.twists_rad ? row.design.twists_rad[moduleIndex] ?? 0 : row.design.modules[moduleIndex].twist_to_next_rad ?? 0,
    length: row.design.parameters ? row.design.parameters[moduleIndex]?.length_m ?? 0.1 : row.design.modules[moduleIndex].parameters?.length_m ?? 0.1,
  }));
  state.q = (row.metrics.example_q ?? Array(state.count).fill(0)).slice();
  refresh();
  showCloud(row.metrics);
  detailEl.textContent = `${modules.join(" · ")}\n${describe(row.metrics)}`;
}

function renderResult() {
  candidatesEl.replaceChildren();
  const rows = state.result?.feasible_pareto ?? [];
  if (!rows.length) {
    candidatesEl.textContent = state.result ? "这次预算下没有可行前沿。可以放宽任务区、减小载荷，或增加采样。" : "";
  }
  rows.forEach((row, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "candidate";
    const modules = row.design.module_ids ?? row.design.modules.map((item) => item.id);
    button.innerHTML = `<strong>${modules.map((id) => LIBRARY[id]?.label ?? id).join(" / ")}</strong><span>覆盖 ${(row.metrics.task_coverage * 100).toFixed(0)}% · 质量 ${row.metrics.mass_kg.toFixed(2)} kg · 可操作度 ${row.metrics.median_translational_manipulability.toFixed(4)}</span>`;
    button.addEventListener("click", () => selectCandidate(index));
    candidatesEl.append(button);
  });
  drawPareto();
  if (rows.length) selectCandidate(0);
  else detailEl.textContent = describe(state.result?.baseline?.metrics);
}

function download(filename, text, type) {
  const blob = new Blob([text], { type });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  link.click();
  URL.revokeObjectURL(link.href);
}

function normalizeImported(payload) {
  if (payload.feasible_pareto) return payload;
  if (payload.design) return { feasible_pareto: [payload], baseline: null };
  throw new Error("文件里没有设计或搜索结果");
}

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xd5dde6);
const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 20);
camera.up.set(0, 0, 1);
camera.position.set(1.35, -1.45, 0.9);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
document.querySelector("#viewport").append(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0.05, 0.05, 0.7);
controls.enableDamping = true;
scene.add(new THREE.AmbientLight(0xffffff, 0.75));
const key = new THREE.DirectionalLight(0xffffff, 1.2);
key.position.set(0.8, -0.5, 1.5);
scene.add(key);
const floor = new THREE.Mesh(new THREE.PlaneGeometry(3, 3), new THREE.MeshStandardMaterial({ color: 0xd7dee6 }));
scene.add(floor);
const grid = new THREE.GridHelper(3, 24, 0xb7c2ce, 0xc9d2dc);
grid.rotation.x = Math.PI / 2;
scene.add(grid);

function frame() {
  const viewport = document.querySelector("#viewport");
  const width = viewport.clientWidth;
  const height = viewport.clientHeight;
  if (renderer.domElement.width !== width || renderer.domElement.height !== height) {
    renderer.setSize(width, height, false);
    camera.aspect = width / Math.max(height, 1);
    camera.updateProjectionMatrix();
  }
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

document.querySelector("#count-now").addEventListener("change", () => {
  state.count = Math.min(8, Math.max(3, num("#count-now")));
  document.querySelector("#count-now").value = String(state.count);
  refresh();
});
document.querySelectorAll("input[name=mode]").forEach((input) => input.addEventListener("change", refresh));
["#count-min", "#count-max", "#cx", "#cy", "#cz", "#hx", "#hy", "#hz"].forEach((id) => {
  document.querySelector(id).addEventListener("change", () => {
    state.count = Math.min(num("#count-max"), Math.max(num("#count-min"), state.count));
    refresh();
  });
});
document.querySelector("#count-lock").addEventListener("change", () => {
  if (document.querySelector("#count-lock").checked) {
    document.querySelector("#count-min").value = String(state.count);
    document.querySelector("#count-max").value = String(state.count);
  }
});

runButton.addEventListener("click", () => {
  const { design, errors } = currentDesign();
  if (errors.length) {
    progressEl.textContent = errors.join(" ");
    return;
  }
  const spec = specFromState();
  runButton.disabled = true;
  cancelButton.disabled = false;
  progressEl.textContent = "正在本页计算，不连接外部求解服务。";
  worker?.terminate();
  worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  worker.onmessage = (event) => {
    if (event.data.type === "progress") {
      progressEl.textContent = `第 ${event.data.generation} 代 · 可行个体 ${event.data.feasible} · 评价约 ${event.data.evaluations} 次`;
    } else if (event.data.type === "done") {
      state.result = event.data.result;
      progressEl.textContent = event.data.cancelled
        ? "已取消。下面是停下来之前的可行前沿。"
        : `完成 ${state.result.evaluations} 次评价。基准覆盖率 ${(state.result.baseline.metrics.task_coverage * 100).toFixed(1)}%，可行前沿 ${state.result.feasible_pareto.length} 条。`;
      renderResult();
      runButton.disabled = false;
      cancelButton.disabled = true;
    } else if (event.data.type === "error") {
      progressEl.textContent = event.data.message;
      runButton.disabled = false;
      cancelButton.disabled = true;
    }
  };
  worker.postMessage({ type: "start", spec, design });
});
cancelButton.addEventListener("click", () => worker?.postMessage({ type: "cancel" }));

document.querySelector("#export-json").addEventListener("click", () => {
  const payload = state.result ?? { design: designFromState(), note: "尚未搜索，只导出当前装配" };
  download("arm-design.json", JSON.stringify(payload, null, 2), "application/json");
});
document.querySelector("#export-csv").addEventListener("click", () => {
  if (!state.result) return;
  download("arm-design.csv", candidatesCsv(state.result), "text/csv");
});
document.querySelector("#export-mjcf").addEventListener("click", () => {
  const { design, errors } = currentDesign();
  if (errors.length) return;
  download("arm-design.xml", exportMjcf(design), "application/xml");
});
document.querySelector("#import-json").addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  const payload = normalizeImported(JSON.parse(await file.text()));
  state.result = payload.feasible_pareto ? payload : null;
  renderResult();
});

new STLLoader().load(`${import.meta.env.BASE_URL}meshes/a.STL`, (geometry) => {
  geometry.computeVertexNormals();
  aGeometry = geometry;
  new STLLoader().load(`${import.meta.env.BASE_URL}meshes/b.STL`, (other) => {
    other.computeVertexNormals();
    bGeometry = other;
    mountArm();
  });
});

renderLibrary();
refresh();
drawPareto();
