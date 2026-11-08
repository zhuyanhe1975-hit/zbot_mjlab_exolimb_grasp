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
      grid_divisions: num("#grid"),
      position_tolerance_m: num("#tolerance"),
      ik_starts: num("#ik-starts"),
      ik_iterations: num("#ik-iters"),
      ik_damping: 0.08,
      joint_sample_limit: 6.28,
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
    const color = point.reached ? [0.12, 0.42, 0.95] : [0.62, 0.66, 0.7];
    colors.set(color, index * 3);
  });
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  cloudPoints = new THREE.Points(geometry, new THREE.PointsMaterial({ size: 0.012, vertexColors: true }));
  scene.add(cloudPoints);
}

function describe(metrics) {
  if (!metrics) return "还没有指标。";
  const manip = metrics.task_manipulability_valid
    ? metrics.task_manipulability_median.toFixed(5)
    : "无效（没有可行任务解，优化中按惩罚处理）";
  const speed = metrics.task_velocity_valid
    ? `${metrics.task_velocity_median.toFixed(3)} m/s`
    : "无效";
  const hit = metrics.pose_hit_rate == null
    ? "未计算"
    : `${(metrics.pose_hit_rate * 100).toFixed(1)}%（关节样本比例，标准误差 ±${((metrics.pose_hit_se || 0) * 100).toFixed(1)} 个百分点，不是任务覆盖）`;
  return [
    `质量 ${metrics.mass_kg.toFixed(3)} kg`,
    `任务格点可行 ${metrics.task_targets_reached}/${metrics.task_target_count} = ${(metrics.task_feasible_fraction * 100).toFixed(1)}%`,
    `格点每轴 ${metrics.grid_divisions}，容差 ${metrics.position_tolerance_m} m，IK 起点 ${metrics.ik_starts}，迭代 ${metrics.ik_iterations}`,
    "未解出的格点只表示这个预算没找到，不是严格不可达。",
    `可行解上的平移可操作度 ${manip}（sqrt(det(Jv Jv^T))，更多关节会改变尺度）`,
    `可行解上的伪逆速度 ${speed}（冗余臂的保守下界，不是限速最优；速度上限是假设）`,
    metrics.task_velocity_lp_median_m_s == null
      ? "坐标轴线性规划速度 未计算"
      : `坐标轴线性规划速度 ${metrics.task_velocity_lp_median_m_s.toFixed(3)} m/s（六个轴方向的最小值，不是速度多面体内切球）`,
    metrics.task_torque_margin_median_nm == null
      ? "静载裕度 无效"
      : `可行解静载裕度中位数 ${metrics.task_torque_margin_median_nm.toFixed(2)} N·m`,
    metrics.position_rank_deficient_fraction == null
      ? "平移雅可比秩 无效"
      : `平移雅可比秩不足的可行解 ${(metrics.position_rank_deficient_fraction * 100).toFixed(0)}%（只看位置 3 行，不是姿态 6 维）`,
    `驱动关节 ${metrics.driven_joint_count ?? "—"}，ZBot 被动 a 连杆 ${metrics.passive_body_count ?? "—"}`,
    (metrics.driven_joint_count ?? 0) > 6 ? "驱动数大于 6，这是位置任务的冗余臂，浏览器没有把它截成 6 个关节。" : "",
    metrics.structurally_degenerate
      ? "随机一般位形的六维几何雅可比结构退化，不能当作完整位姿 6DOF。"
      : (metrics.task_pose_spatial_rank_deficient_fraction == null
        ? ""
        : `可行任务解上六维秩不足 ${(metrics.task_pose_spatial_rank_deficient_fraction * 100).toFixed(0)}%（特征长度 0.1 m，只作诊断，不是分数）。`),
    `末端落盒比例 ${hit}`,
    metrics.q0_collision ? "零位有胶囊相交，仅作诊断，不否决这条设计。" : "零位没有非相邻胶囊相交。这也只是诊断。",
    `零位峰值静力矩 ${metrics.q0_peak_torque_nm.toFixed(3)} N·m。`,
  ].filter(Boolean).join("\n");
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
  const xs = rows.map((row) => row.metrics.task_feasible_fraction);
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
    ctx.arc(px(row.metrics.task_feasible_fraction), py(row.metrics.mass_kg), 5, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.fillStyle = "#5c6775";
  ctx.font = "11px sans-serif";
  ctx.fillText("格点可行比例 →", width - 108, height - 6);
  ctx.fillText("质量", 4, 16);
  paretoCanvas.onclick = (event) => {
    const rect = paretoCanvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) * (paretoCanvas.width / rect.width);
    const y = (event.clientY - rect.top) * (paretoCanvas.height / rect.height);
    let best = 0;
    let bestDist = Infinity;
    rows.forEach((row, index) => {
      const dx = px(row.metrics.task_feasible_fraction) - x;
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
  const bits = row.bits ?? (row.design.modules ?? []).slice(0, -1).map((item) => (item.twist_to_next_rad > 0.1 ? "1" : "0")).join("");
  const centers = row.centers
    ? `\n单元中心点 ${row.centers.task_targets_reached}/${row.centers.task_target_count}，伪逆速度 ${row.centers.task_velocity_median?.toFixed?.(3) ?? "—"} m/s，线性规划速度 ${row.centers.task_velocity_lp_median_m_s?.toFixed?.(3) ?? "—"} m/s。`
    : "";
  const base = state.result?.baseline?.metrics;
  const versus = base
    ? `\n相对 6 模块全 0 基准：可行比例 ${(row.metrics.task_feasible_fraction * 100).toFixed(1)}% 对 ${(base.task_feasible_fraction * 100).toFixed(1)}%，质量 ${row.metrics.mass_kg.toFixed(3)} kg 对 ${base.mass_kg.toFixed(3)} kg。`
    : "";
  detailEl.textContent = `${row.role ? `${row.role} · ` : ""}${modules.length} 个模块 · 接口 ${bits || "—"}\n${modules.join(" · ")}\n${describe(row.metrics)}${centers}${versus}`;
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
    const driven = row.metrics.driven_joint_count ?? modules.length;
    const bits = row.bits ?? "—";
    const manip = row.metrics.task_manipulability_valid ? row.metrics.task_manipulability_median.toFixed(4) : "无效";
    const lp = row.metrics.task_velocity_lp_median_m_s == null ? "未算" : `${row.metrics.task_velocity_lp_median_m_s.toFixed(3)} m/s`;
    button.innerHTML = `<strong>${driven} 驱动 · 接口 ${bits} · ${row.role ?? "候选"}</strong><span>格点可行 ${(row.metrics.task_feasible_fraction * 100).toFixed(0)}% · 质量 ${row.metrics.mass_kg.toFixed(2)} kg · 可操作度 ${manip} · 轴速度 ${lp}</span>`;
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

function lockCount(count) {
  state.count = count;
  document.querySelector("#count-min").value = String(count);
  document.querySelector("#count-max").value = String(count);
  document.querySelector("#count-lock").checked = true;
  refresh();
}
document.querySelector("#lock6").addEventListener("click", () => lockCount(6));
document.querySelector("#lock7").addEventListener("click", () => lockCount(7));
document.querySelector("#lock8").addEventListener("click", () => lockCount(8));
document.querySelector("#load-batch").addEventListener("click", async () => {
  progressEl.textContent = "正在读取首批 6/7/8 复核。";
  const response = await fetch(`${import.meta.env.BASE_URL}longterm_batch1.json`);
  if (!response.ok) {
    progressEl.textContent = "还没有首批结果文件。先运行 python scripts/design_678_batch1.py。";
    return;
  }
  const payload = normalizeImported(await response.json());
  state.result = payload;
  const goals = payload.goals ?? {};
  const coverage = goals.coverage_priority;
  const speed = goals.speed_priority;
  const light = goals.light_priority;
  progressEl.textContent = [
    payload.note,
    coverage ? `覆盖优先：${coverage.count} 模块 ${coverage.bits}。${coverage.why}` : "",
    speed ? `速度优先：${speed.count} 模块 ${speed.bits}。${speed.why}` : "",
    light ? `轻量优先：${light.count} 模块 ${light.bits}。${light.why}` : "",
  ].filter(Boolean).join(" ");
  renderResult();
});
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
  document.querySelector("#reeval").disabled = true;
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
        : `完成 ${state.result.evaluations} 次评价。基准格点可行 ${(state.result.baseline.metrics.task_feasible_fraction * 100).toFixed(1)}%（${state.result.baseline.metrics.task_targets_reached}/${state.result.baseline.metrics.task_target_count}），前沿 ${state.result.feasible_pareto.length} 条。`;
      renderResult();
      runButton.disabled = false;
      cancelButton.disabled = true;
      document.querySelector("#reeval").disabled = false;
    } else if (event.data.type === "reeval") {
      event.data.rows.forEach((metrics, index) => {
        if (index === 0) state.result.baseline.metrics = metrics;
        else if (state.result.feasible_pareto[index - 1]) state.result.feasible_pareto[index - 1].metrics = metrics;
      });
      const reviewed = event.data.rows[0];
      progressEl.textContent = `复核完成。格点每轴 ${reviewed.grid_divisions}，基准可行 ${reviewed.task_targets_reached}/${reviewed.task_target_count}。没找到的格点不是严格不可达。`;
      renderResult();
      runButton.disabled = false;
      cancelButton.disabled = true;
      document.querySelector("#reeval").disabled = false;
    } else if (event.data.type === "error") {
      progressEl.textContent = event.data.message;
      runButton.disabled = false;
      cancelButton.disabled = true;
    }
  };
  worker.postMessage({ type: "start", spec, design });
});
cancelButton.addEventListener("click", () => worker?.postMessage({ type: "cancel" }));
document.querySelector("#reeval").addEventListener("click", () => {
  if (!state.result || !worker) return;
  const spec = specFromState();
  const task = {
    ...spec.task,
    grid_divisions: Math.min(4, spec.task.grid_divisions + 1),
    ik_starts: Math.min(8, spec.task.ik_starts + 2),
    ik_iterations: Math.min(20, spec.task.ik_iterations + 4),
    sample_count: 0,
  };
  const designs = [state.result.baseline.design, ...state.result.feasible_pareto.slice(0, 3).map((row) => row.design)]
    .map(normalizeDesign);
  progressEl.textContent = `正在用每轴 ${task.grid_divisions} 份格点、${task.ik_starts} 个起点复核，不把漏检当成不可达。`;
  document.querySelector("#reeval").disabled = true;
  worker.postMessage({ type: "reeval", task, designs });
});

function normalizeDesign(design) {
  if (design.module_ids) return design;
  return {
    module_ids: design.modules.map((item) => item.id),
    twists_rad: design.modules.slice(0, -1).map((item) => item.twist_to_next_rad ?? 0),
    parameters: design.modules.map((item) => item.parameters ?? {}),
    base_position_m: design.base_position_m ?? [0, 0, 0.4],
  };
}

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
