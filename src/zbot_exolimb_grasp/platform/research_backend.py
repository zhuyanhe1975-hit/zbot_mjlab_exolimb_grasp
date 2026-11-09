"""可选科研后端。

正运动学走 generative-robot-design 的 ``Custom(proximal, distal)``：
近端变换、绕 z 的 θ、远端变换。现有 NumPy 模型保留，不从这里替换。

Pinocchio 负责静力矩，hpp-fcl 负责网格复核，pymoo 负责多目标筛选。
设计网络训练不在这里做。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from zbot_exolimb_grasp.config_gen import A_TO_B_POS, A_TO_B_QUAT, B_TO_A_POS
from zbot_exolimb_grasp.constants import LINK_A_MASS, LINK_B_MASS, SHOULDER_POS
from zbot_exolimb_grasp.platform.library import COM_A, COM_B, CONCEPT_LINEAR_DENSITY_KG_PER_M
from zbot_exolimb_grasp.platform.model import sole_site, zbot_attachment

ROOT = Path(__file__).resolve().parents[3]
GRD_ROOT = ROOT / "third_party" / "generative-robot-design"
MESH_A = ROOT / "assets" / "zbot_6" / "meshes" / "a.STL"
MESH_B = ROOT / "assets" / "zbot_6" / "meshes" / "b.STL"
JOINT_LIMIT = 6.28


def ensure_grd_path() -> None:
    text = str(GRD_ROOT)
    if text not in sys.path:
        sys.path.insert(0, text)


def availability() -> dict[str, object]:
    status: dict[str, object] = {"grd_root": str(GRD_ROOT), "grd_present": GRD_ROOT.is_dir()}
    for name, imported in (
        ("torch", "torch"),
        ("mujoco", "mujoco"),
        ("pinocchio", "pinocchio"),
        ("hppfcl", "hppfcl"),
        ("pymoo", "pymoo"),
    ):
        try:
            module = __import__(imported)
            status[name] = getattr(module, "__version__", "ok")
        except Exception as exc:  # 可选依赖缺失时只记录，不打断自写后端。
            status[name] = f"unavailable: {exc.__class__.__name__}: {exc}"
    return status


def _need(name: str):
    status = availability()
    value = status.get(name)
    if not isinstance(value, str) or value.startswith("unavailable"):
        raise RuntimeError(f"科研后端缺少 {name}：{value}")
    return value


def _quat_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in quat]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def homog(pos, quat) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = _quat_matrix(np.asarray(quat, dtype=np.float64))
    out[:3, 3] = np.asarray(pos, dtype=np.float64)
    return out


def zbot_proximal() -> np.ndarray:
    return homog(A_TO_B_POS, A_TO_B_QUAT)


def zbot_distal(twist: float) -> np.ndarray:
    return homog(B_TO_A_POS, zbot_attachment(twist))


def sole_matrix() -> np.ndarray:
    site = sole_site()
    return homog(site.pos_m, site.quat_wxyz)


def concept_proximal() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def concept_distal(length: float, twist: float) -> np.ndarray:
    distal = np.eye(4, dtype=np.float64)
    distal[0, 3] = float(length)
    half = 0.5 * float(twist)
    # 绕下一模块的 x 轴，与自写模型的 axis_angle_x 一致。
    distal[:3, :3] = _quat_matrix(np.array([math.cos(half), math.sin(half), 0.0, 0.0]))
    return distal


def module_matrices(module_ids: tuple[str, ...], twists: tuple[float, ...], lengths: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray, float]:
    """每个模块的近端、远端 4×4，以及总质量。末端模块的远端不会进入工具点。"""
    count = len(module_ids)
    proximal = np.zeros((count, 4, 4), dtype=np.float64)
    distal = np.zeros((count, 4, 4), dtype=np.float64)
    mass = 0.0
    for index, module_id in enumerate(module_ids):
        twist = float(twists[index]) if index < count - 1 else 0.0
        if module_id == "zbot_r3":
            proximal[index] = zbot_proximal()
            distal[index] = zbot_distal(twist)
            mass += LINK_A_MASS + LINK_B_MASS
        elif module_id == "concept_axial":
            length = float(lengths[index])
            proximal[index] = concept_proximal()
            distal[index] = concept_distal(length, twist)
            mass += CONCEPT_LINEAR_DENSITY_KG_PER_M * length
        else:
            raise ValueError(f"科研后端还没有模块 {module_id}")
    return proximal, distal, mass


def _custom_modules(proximal: np.ndarray, distal: np.ndarray):
    """把矩阵装进上游的 Custom。标量 θ 走他们的 transform()。"""
    ensure_grd_path()
    _need("torch")
    import torch
    from robot_modules.module import Custom

    modules = []
    for index in range(proximal.shape[0]):
        modules.append(
            Custom(
                torch.tensor(proximal[index], dtype=torch.float64),
                torch.tensor(distal[index], dtype=torch.float64),
            )
        )
    return modules


def scalar_module_matches_product(proximal: np.ndarray, distal: np.ndarray, theta: float) -> float:
    """确认批量公式和上游 Module.transform 在单个关节角上一致。"""
    ensure_grd_path()
    import torch
    from transforms.batched import Rotate

    module = _custom_modules(proximal[:1], distal[:1])[0]
    official = module.transform(torch.tensor(theta, dtype=torch.float64)).detach().numpy()
    rotation = Rotate.axis_angle(torch.tensor(theta, dtype=torch.float64), "z").get_matrix().detach().numpy()
    manual = proximal[0] @ rotation @ distal[0]
    return float(np.max(np.abs(official - manual)))


def forward_details(proximal: np.ndarray, distal: np.ndarray, q, tool: str = "foot_1_sole"):
    """可微正运动学。返回末端 4×4、关节原点和平移雅可比。q 形状 (..., n)。"""
    ensure_grd_path()
    import torch
    from transforms.batched import Rotate

    theta = torch.as_tensor(q, dtype=torch.float64)
    count = proximal.shape[0]
    base = torch.eye(4, dtype=torch.float64)
    base[:3, 3] = torch.tensor(SHOULDER_POS, dtype=torch.float64)
    frame = base.expand(*theta.shape[:-1], 4, 4).clone()
    prox = torch.tensor(proximal, dtype=torch.float64)
    dist = torch.tensor(distal, dtype=torch.float64)
    origins = []
    axes = []
    swung = None
    joint_origin = None
    for index in range(count):
        frame = frame @ prox[index]
        joint_origin = frame
        origins.append(frame[..., :3, 3])
        axes.append(frame[..., :3, 2])
        frame = frame @ Rotate.axis_angle(theta[..., index], "z").get_matrix()
        swung = frame
        if index < count - 1:
            frame = frame @ dist[index]
    if tool == "foot_joint_origin":
        end = joint_origin
    elif tool == "foot_1_sole":
        sole = torch.tensor(sole_matrix(), dtype=torch.float64)
        end = swung @ sole
    else:
        raise ValueError(tool)
    ee = end[..., :3, 3]
    jacobian = torch.stack([torch.linalg.cross(axis, ee - origin) for axis, origin in zip(axes, origins, strict=True)], dim=-1)
    axis_stack = torch.stack(axes, dim=-1)
    return end, jacobian, axis_stack


def forward_chain_torch(proximal: np.ndarray, distal: np.ndarray, q, tool: str = "foot_1_sole"):
    end, _jacobian, _axes = forward_details(proximal, distal, q, tool)
    return end


def positions(proximal, distal, q, tool: str = "foot_1_sole") -> np.ndarray:
    pose = forward_chain_torch(proximal, distal, q, tool)
    import torch

    return pose.detach().numpy()[..., :3, 3]


def gradient_position_ik(
    proximal,
    distal,
    targets: np.ndarray,
    starts: np.ndarray,
    *,
    tool: str,
    iterations: int,
    step: float,
    tolerance: float,
    feasible=None,
) -> tuple[np.ndarray, np.ndarray]:
    """用上游正运动学的几何雅可比做阻尼最小二乘，再对仍超差的解做 Adam 精修。

    feasible(q) 为真才算这个起点可用。返回 (best_q, best_error)。没进容差不是不可达。
    """
    import torch

    goals = torch.tensor(targets, dtype=torch.float64)
    seeds = torch.tensor(starts, dtype=torch.float64)
    count = proximal.shape[0]
    q = seeds[:, None, :].expand(seeds.shape[0], goals.shape[0], count).clone()
    desired = goals[None, :, :]
    damping = 0.08
    eye = torch.eye(3, dtype=torch.float64)
    for _ in range(int(iterations)):
        _end, jacobian, _axes = forward_details(proximal, distal, q, tool)
        ee = _end[..., :3, 3]
        error = desired - ee
        gram = jacobian @ jacobian.transpose(-1, -2) + (damping ** 2) * eye
        lagrange = torch.linalg.solve(gram, error.unsqueeze(-1)).squeeze(-1)
        delta = (jacobian.transpose(-1, -2) @ lagrange.unsqueeze(-1)).squeeze(-1)
        q = (q + float(step) * delta).clamp(-JOINT_LIMIT, JOINT_LIMIT)
    with torch.no_grad():
        error = torch.linalg.norm(forward_details(proximal, distal, q, tool)[0][..., :3, 3] - desired, dim=-1)
    q_np = q.detach().numpy()
    err_np = error.detach().numpy()
    picked = np.zeros((goals.shape[0], count), dtype=np.float64)
    picked_error = np.zeros(goals.shape[0], dtype=np.float64)
    refine_rows = []
    for target_index in range(goals.shape[0]):
        order = np.argsort(err_np[:, target_index])
        chosen = int(order[0])
        for start_index in order:
            if err_np[start_index, target_index] > tolerance:
                break
            if feasible is None or feasible(q_np[start_index, target_index]):
                chosen = int(start_index)
                break
        picked[target_index] = q_np[chosen, target_index]
        picked_error[target_index] = err_np[chosen, target_index]
        if picked_error[target_index] > tolerance:
            refine_rows.append(target_index)
    if refine_rows:
        import torch

        subset = torch.tensor(picked[refine_rows], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([subset], lr=0.03)
        desired_sub = goals[refine_rows]
        for _ in range(10):
            ee = forward_chain_torch(proximal, distal, subset, tool)[..., :3, 3]
            loss = torch.sum((ee - desired_sub) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                subset.clamp_(-JOINT_LIMIT, JOINT_LIMIT)
        with torch.no_grad():
            refined = subset.detach().numpy()
            refined_error = torch.linalg.norm(
                forward_chain_torch(proximal, distal, subset, tool)[..., :3, 3] - desired_sub, dim=-1
            ).detach().numpy()
        for local, target_index in enumerate(refine_rows):
            if refined_error[local] < picked_error[target_index]:
                if feasible is None or feasible(refined[local]) or refined_error[local] > tolerance:
                    picked[target_index] = refined[local]
                    picked_error[target_index] = refined_error[local]
    return picked, picked_error


def mujoco_tool_positions(xml: str | Path, q: np.ndarray, *, site_name: str, body_name: str | None = None) -> dict[str, np.ndarray]:
    _need("mujoco")
    import mujoco

    if isinstance(xml, Path) or (isinstance(xml, str) and xml.endswith(".xml") and Path(xml).is_file()):
        model = mujoco.MjModel.from_xml_path(str(xml))
    else:
        model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qpos[: q.size] = np.asarray(q, dtype=np.float64)
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"MJCF 没有 site {site_name}")
    out = {"site": data.site_xpos[site_id].copy()}
    if body_name:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"MJCF 没有 body {body_name}")
        out["body"] = data.xpos[body_id].copy()
    return out


def pinocchio_static_torques(proximal, distal, module_ids, lengths, q: np.ndarray, payload_kg: float) -> np.ndarray:
    """用 RNEA 在零速度、零加速度下算重力静力矩。载荷加在足底点。"""
    _need("pinocchio")
    import pinocchio as pin

    model = pin.Model()
    parent = 0
    shoulder = pin.SE3(np.eye(3), np.asarray(SHOULDER_POS, dtype=np.float64))
    for index, module_id in enumerate(module_ids):
        proximal_pose = pin.SE3(proximal[index][:3, :3], proximal[index][:3, 3])
        if index == 0:
            joint_placement = shoulder * proximal_pose
        else:
            previous = pin.SE3(distal[index - 1][:3, :3], distal[index - 1][:3, 3])
            joint_placement = previous * proximal_pose
        joint_id = model.addJoint(parent, pin.JointModelRZ(), joint_placement, f"joint{index + 1}")
        if module_id == "zbot_r3":
            model.appendBodyToJoint(
                joint_id,
                pin.Inertia(LINK_B_MASS, np.asarray(COM_B, dtype=np.float64), np.diag([1e-4, 1e-4, 1e-4])),
                pin.SE3.Identity(),
            )
            inertia_a = pin.Inertia(LINK_A_MASS, np.asarray(COM_A, dtype=np.float64), np.diag([1e-4, 1e-4, 1e-4]))
            if index == 0:
                model.appendBodyToJoint(0, inertia_a, shoulder)
            else:
                model.appendBodyToJoint(parent, inertia_a, pin.SE3(distal[index - 1][:3, :3], distal[index - 1][:3, 3]))
        else:
            length = float(lengths[index])
            mass = CONCEPT_LINEAR_DENSITY_KG_PER_M * length
            model.appendBodyToJoint(
                joint_id,
                pin.Inertia(mass, np.array([0.5 * length, 0.0, 0.0]), np.diag([1e-4, 1e-4, 1e-4])),
                pin.SE3.Identity(),
            )
        parent = joint_id
    sole = sole_matrix()
    model.appendBodyToJoint(
        parent,
        pin.Inertia(float(payload_kg), np.zeros(3), np.diag([1e-6, 1e-6, 1e-6])),
        pin.SE3(sole[:3, :3], sole[:3, 3]),
    )
    data = model.createData()
    q_full = np.asarray(q, dtype=np.float64).reshape(model.nq)
    tau = pin.rnea(model, data, q_full, np.zeros(model.nv), np.zeros(model.nv))
    return np.asarray(tau, dtype=np.float64)


def mesh_collision_count(frames_a: list[np.ndarray], frames_b: list[np.ndarray]) -> int:
    """非相邻模块的 a/b 网格碰撞次数。frames 是各连杆的世界 4×4。"""
    _need("hppfcl")
    import hppfcl

    loader = hppfcl.MeshLoader()
    mesh_a = loader.load(str(MESH_A))
    mesh_b = loader.load(str(MESH_B))
    objects = []
    for frame in frames_a:
        objects.append(hppfcl.CollisionObject(mesh_a, hppfcl.Transform3f(frame[:3, :3], frame[:3, 3])))
    for frame in frames_b:
        objects.append(hppfcl.CollisionObject(mesh_b, hppfcl.Transform3f(frame[:3, :3], frame[:3, 3])))
    # 前 count 个是 a，后 count 个是 b。同一模块的 a 与 b 相邻，不查。
    count = len(frames_a)
    hits = 0
    request = hppfcl.CollisionRequest()
    for i in range(count):
        for j in range(i + 1, count):
            if j - i < 2:
                continue
            for left, right in ((i, j), (i, count + j), (count + i, j), (count + i, count + j)):
                result = hppfcl.CollisionResult()
                hppfcl.collide(objects[left], objects[right], request, result)
                hits += int(result.isCollision())
    return hits


def link_frames(proximal, distal, q: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """每个 ZBot 模块的 a 连杆世界系和关节后的 b 连杆世界系。概念模块两套都放在关节后。"""
    import torch
    from transforms.batched import Rotate

    ensure_grd_path()
    theta = torch.tensor(q, dtype=torch.float64)
    base = torch.eye(4, dtype=torch.float64)
    base[:3, 3] = torch.tensor(SHOULDER_POS, dtype=torch.float64)
    frame = base.clone()
    prox = torch.tensor(proximal, dtype=torch.float64)
    dist = torch.tensor(distal, dtype=torch.float64)
    frames_a = []
    frames_b = []
    for index in range(theta.shape[0]):
        frames_a.append(frame.detach().numpy().copy())
        frame = frame @ prox[index]
        frame = frame @ Rotate.axis_angle(theta[index], "z").get_matrix()
        frames_b.append(frame.detach().numpy().copy())
        if index < theta.shape[0] - 1:
            frame = frame @ dist[index]
    return frames_a, frames_b
