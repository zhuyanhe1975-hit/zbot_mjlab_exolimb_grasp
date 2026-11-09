"""串联设计的统一模型：装配校验、FK、平移雅可比和近似评价。

不支持网格精确碰撞、完整动力学和加速度能力。这些缺口在结果里写成
unsupported，不计入分数。
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from zbot_exolimb_grasp.config_gen import (
    A_TO_B_POS,
    A_TO_B_QUAT,
    B_TO_A_POS,
    B_TO_A_QUAT,
    CONNECTION_AXIS,
)
from zbot_exolimb_grasp.constants import LINK_A_MASS, LINK_B_MASS, SOURCE_COMMIT

from .library import (
    CONCEPT_CAPSULE_RADIUS_M,
    CONCEPT_LINEAR_DENSITY_KG_PER_M,
    COM_A,
    COM_B,
    ZBOT_CAPSULE_RADIUS_M,
    mate_rule,
    module_spec,
)
from .math3d import Z_AXIS, axis_angle_x, hinge_z, mul, quat_matrix, rotate, so3_vee
from .schema import COUNT_MAX, COUNT_MIN, JOINT_LIMIT, Design, TaskSpec

UNSUPPORTED = (
    "网格精确碰撞",
    "完整逆动力学与加速度能力",
    "转动可操作度（不与平移指标混单位）",
    "树形或闭环拓扑",
)

_A_POS = np.asarray(A_TO_B_POS, dtype=np.float64)
_A_QUAT = np.asarray(A_TO_B_QUAT, dtype=np.float64)
_B_POS = np.asarray(B_TO_A_POS, dtype=np.float64)
_B_QUAT = np.asarray(B_TO_A_QUAT, dtype=np.float64)
_COM_A = np.asarray(COM_A, dtype=np.float64)
_COM_B = np.asarray(COM_B, dtype=np.float64)
_AXIS = np.asarray(CONNECTION_AXIS, dtype=np.float64)


def zbot_attachment(phi: float) -> np.ndarray:
    """与 config_gen.attachment_quaternions 同一公式，φ=0 时回到 XML。"""
    half = float(phi) * 0.5
    sine = math.sin(half)
    twist = np.array([math.cos(half), _AXIS[0] * sine, _AXIS[1] * sine, _AXIS[2] * sine])
    return mul(twist, _B_QUAT)


def assembly_errors(design: Design) -> list[str]:
    errors: list[str] = []
    count = len(design.module_ids)
    if count < COUNT_MIN or count > COUNT_MAX:
        errors.append(f"模块数量 {count} 超出 {COUNT_MIN}–{COUNT_MAX}")
    if len(design.twists_rad) != max(count - 1, 0):
        errors.append("连接方位的数量必须比模块少 1")
    if len(design.parameters) != count:
        errors.append("每个模块都要有一份参数")
    if len(design.base_position_m) != 3 or not all(math.isfinite(v) for v in design.base_position_m):
        errors.append("安装点必须是有限的三维坐标")
    for index, module_id in enumerate(design.module_ids):
        try:
            spec = module_spec(module_id)
        except KeyError:
            errors.append(f"第 {index + 1} 个模块 {module_id} 不在模块库中")
            continue
        if spec.kind == "physical" and design.parameters[index]:
            errors.append(f"{spec.label} 是实物模块，不能带尺寸参数")
        expected = {item.name for item in spec.parameters}
        given = set(design.parameters[index])
        if spec.kind != "physical" and given != expected:
            errors.append(f"第 {index + 1} 个模块的参数应为 {sorted(expected)}")
        for item in spec.parameters:
            value = design.parameters[index].get(item.name)
            if value is None or not math.isfinite(value) or not item.lower - 1e-9 <= value <= item.upper + 1e-9:
                errors.append(f"第 {index + 1} 个模块的 {item.label} 超出 {item.lower}–{item.upper} {item.unit}")
        if index < count - 1:
            try:
                child_id = design.module_ids[index + 1]
                rule = mate_rule(module_id, child_id)
            except KeyError:
                continue
            if rule is None:
                errors.append(
                    f"第 {index + 1} 与 {index + 2} 个模块的接口族不能连接"
                )
            else:
                twist = design.twists_rad[index]
                if not any(abs(twist - allowed) < 1e-8 for allowed in rule.angles_rad):
                    allowed_deg = ", ".join(str(round(math.degrees(v))) for v in rule.angles_rad)
                    errors.append(f"第 {index + 1} 个连接方位不在合法集合（{allowed_deg}°）内")
    return errors


def require_assembly(design: Design) -> None:
    errors = assembly_errors(design)
    if errors:
        raise ValueError("；".join(errors))


def _as_batch(q: np.ndarray, count: int) -> np.ndarray:
    joints = np.asarray(q, dtype=np.float64)
    if joints.ndim == 1:
        joints = joints.reshape(1, -1)
    if joints.ndim != 2 or joints.shape[1] != count:
        raise ValueError(f"关节角应为 ({count},) 或 (样本, {count})")
    return joints


def _segment_distance(p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    """两线段距离，形状与样本轴一致。"""
    d1 = p1 - p0
    d2 = q1 - q0
    r = p0 - q0
    a = np.sum(d1 * d1, axis=-1)
    e = np.sum(d2 * d2, axis=-1)
    f = np.sum(d2 * r, axis=-1)
    c = np.sum(d1 * r, axis=-1)
    b = np.sum(d1 * d2, axis=-1)
    eps = 1e-12
    denom = a * e - b * b
    s = np.zeros_like(a)
    parallel = denom <= eps
    s = np.where(~parallel, np.clip((b * f - c * e) / np.maximum(denom, eps), 0.0, 1.0), 0.0)
    t = np.where(e > eps, (b * s + f) / np.maximum(e, eps), 0.0)
    low = t < 0.0
    high = t > 1.0
    s = np.where(low & (a > eps), np.clip(-c / np.maximum(a, eps), 0.0, 1.0), s)
    t = np.where(low, 0.0, t)
    s = np.where(high & (a > eps), np.clip((b - c) / np.maximum(a, eps), 0.0, 1.0), s)
    t = np.where(high, 1.0, t)
    degenerate = (a <= eps) & (e <= eps)
    s = np.where(a <= eps, 0.0, s)
    t = np.where(e <= eps, 0.0, t)
    t = np.where((a <= eps) & (e > eps), np.clip(f / np.maximum(e, eps), 0.0, 1.0), t)
    c1 = p0 + s[..., None] * d1
    c2 = q0 + t[..., None] * d2
    dist = np.linalg.norm(c1 - c2, axis=-1)
    return np.where(degenerate, np.linalg.norm(p0 - q0, axis=-1), dist)


_ZBOT_XML = Path(__file__).resolve().parents[3] / "assets" / "zbot_6" / "ZBOT6.xml"
TOOL_JOINT_ORIGIN = "foot_joint_origin"
TOOL_SOLE = "foot_1_sole"


@dataclass(frozen=True)
class ToolSite:
    name: str
    body: str
    pos_m: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]
    source_commit: str
    source_path: str


_SOLE_SITE: ToolSite | None = None


def sole_site() -> ToolSite:
    """从权威 ZBOT6.xml 读取 foot_1 上的足底 site。数值不在这里手写。"""
    global _SOLE_SITE
    if _SOLE_SITE is not None:
        return _SOLE_SITE
    if not _ZBOT_XML.is_file():
        raise FileNotFoundError(f"找不到权威模型 {_ZBOT_XML}")
    foot = None
    for body in ET.parse(_ZBOT_XML).iter("body"):
        if body.get("name") == "foot_1":
            foot = body
            break
    if foot is None:
        raise ValueError("ZBOT6.xml 没有 body foot_1")
    sites = [site for site in list(foot) if site.tag == "site" and "sole" in (site.get("name") or "")]
    if len(sites) != 1:
        raise ValueError(f"foot_1 上的足底 site 数量是 {len(sites)}，无法确定工具点")
    site = sites[0]
    pos = tuple(float(part) for part in site.get("pos", "").split())
    quat = tuple(float(part) for part in site.get("quat", "").split())
    if len(pos) != 3 or len(quat) != 4:
        raise ValueError(f"足底 site 变换不完整：pos={site.get('pos')} quat={site.get('quat')}")
    _SOLE_SITE = ToolSite(
        name=str(site.get("name")),
        body="foot_1",
        pos_m=(float(pos[0]), float(pos[1]), float(pos[2])),
        quat_wxyz=(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
        source_commit=SOURCE_COMMIT,
        source_path="assets/zbot_6/ZBOT6.xml",
    )
    return _SOLE_SITE


def forward_chain(design: Design, q: np.ndarray, *, tool_point: str = TOOL_JOINT_ORIGIN) -> dict[str, object]:
    """批量正运动学。默认末端仍是 foot_1 关节原点，足底模式另标。"""
    if tool_point not in {TOOL_JOINT_ORIGIN, TOOL_SOLE}:
        raise ValueError(f"未知任务点 {tool_point}")
    require_assembly(design)
    count = len(design.module_ids)
    joints = _as_batch(q, count)
    samples = joints.shape[0]
    pos = np.broadcast_to(np.asarray(design.base_position_m, dtype=np.float64), (samples, 3)).copy()
    quat = np.zeros((samples, 4), dtype=np.float64)
    quat[:, 0] = 1.0
    origins: list[np.ndarray] = []
    axes: list[np.ndarray] = []
    masses: list[float] = []
    coms: list[np.ndarray] = []
    after_joint: list[bool] = []
    module_of_body: list[int] = []
    capsules: list[dict[str, object]] = []
    total_mass = 0.0
    for index, module_id in enumerate(design.module_ids):
        has_child = index < count - 1
        twist = float(design.twists_rad[index]) if has_child else 0.0
        if module_id == "zbot_r3":
            origin = pos + rotate(quat, _A_POS)
            joint_quat = mul(quat, _A_QUAT)
            axis = rotate(joint_quat, Z_AXIS)
            swung = mul(joint_quat, hinge_z(joints[:, index]))
            com_a = pos + rotate(quat, _COM_A)
            com_b = origin + rotate(swung, _COM_B)
            b_end = origin + rotate(swung, _B_POS)
            origins.append(origin)
            axes.append(axis)
            masses.extend([LINK_A_MASS, LINK_B_MASS])
            coms.extend([com_a, com_b])
            after_joint.extend([False, True])
            module_of_body.extend([index, index])
            total_mass += LINK_A_MASS + LINK_B_MASS
            capsules.append({"p0": pos.copy(), "p1": origin, "r": ZBOT_CAPSULE_RADIUS_M, "module": index})
            capsules.append({"p0": origin, "p1": b_end, "r": ZBOT_CAPSULE_RADIUS_M, "module": index})
            if has_child:
                pos = b_end
                quat = mul(swung, zbot_attachment(twist))
            else:
                ee = origin
                ee_quat = swung
                applied = TOOL_JOINT_ORIGIN
                if tool_point == TOOL_SOLE:
                    site = sole_site()
                    ee = origin + rotate(swung, np.asarray(site.pos_m, dtype=np.float64))
                    ee_quat = mul(swung, np.asarray(site.quat_wxyz, dtype=np.float64))
                    applied = site.name
        elif module_id in {"concept_axial", "demo_unreleased"}:
            length = float(design.parameters[index]["length_m"])
            origin = pos.copy()
            axis = rotate(quat, Z_AXIS)
            post = mul(quat, hinge_z(joints[:, index]))
            tip = origin + rotate(post, np.array([length, 0.0, 0.0]))
            com = origin + rotate(post, np.array([length * 0.5, 0.0, 0.0]))
            mass = CONCEPT_LINEAR_DENSITY_KG_PER_M * length
            origins.append(origin)
            axes.append(axis)
            masses.append(mass)
            coms.append(com)
            after_joint.append(True)
            module_of_body.append(index)
            total_mass += mass
            capsules.append({"p0": origin, "p1": tip, "r": CONCEPT_CAPSULE_RADIUS_M, "module": index})
            if has_child:
                pos = tip
                quat = mul(post, axis_angle_x(twist))
            else:
                ee = tip
                ee_quat = post
                applied = "concept_tip" if tool_point == TOOL_JOINT_ORIGIN else "concept_tip_no_sole"
        else:
            raise ValueError(f"没有运动学实现：{module_id}")
    jacobian = np.empty((samples, 3, count), dtype=np.float64)
    for index in range(count):
        jacobian[:, :, index] = np.cross(axes[index], ee - origins[index])
    return {
        "ee": ee,
        "ee_quat": ee_quat,
        "tool_point": tool_point,
        "tool_point_applied": applied,
        "jacobian": jacobian,
        "origins": origins,
        "axes": axes,
        "masses": masses,
        "coms": coms,
        "after_joint": after_joint,
        "module_of_body": module_of_body,
        "capsules": capsules,
        "mass_kg": total_mass,
    }


def static_torques(chain: dict[str, object], payload_kg: float, gravity: float = 9.81) -> np.ndarray:
    """质点静力矩。重力沿 -Z。这不是逆动力学。"""
    ee = chain["ee"]
    origins = chain["origins"]
    axes = chain["axes"]
    samples = ee.shape[0]
    count = len(origins)
    torque = np.zeros((samples, count), dtype=np.float64)
    weight = np.array([0.0, 0.0, -gravity], dtype=np.float64)
    for joint_index in range(count):
        for body_index, mass in enumerate(chain["masses"]):
            distal = chain["module_of_body"][body_index] > joint_index or (
                chain["module_of_body"][body_index] == joint_index and chain["after_joint"][body_index]
            )
            if not distal:
                continue
            lever = chain["coms"][body_index] - origins[joint_index]
            torque[:, joint_index] += np.sum(axes[joint_index] * np.cross(lever, mass * weight), axis=-1)
        lever = ee - origins[joint_index]
        torque[:, joint_index] += np.sum(
            axes[joint_index] * np.cross(lever, payload_kg * weight),
            axis=-1,
        )
    return torque


def collision_mask(chain: dict[str, object]) -> np.ndarray:
    """非相邻模块胶囊相交。相邻模块不查，避免把正常接头当成碰撞。"""
    capsules = chain["capsules"]
    samples = chain["ee"].shape[0]
    hit = np.zeros(samples, dtype=bool)
    for i in range(len(capsules)):
        for j in range(i + 1, len(capsules)):
            if abs(int(capsules[i]["module"]) - int(capsules[j]["module"])) < 2:
                continue
            dist = _segment_distance(capsules[i]["p0"], capsules[i]["p1"], capsules[j]["p0"], capsules[j]["p1"])
            limit = float(capsules[i]["r"]) + float(capsules[j]["r"])
            hit |= dist < limit - 1e-6
    return hit


def geometric_jacobian(chain: dict[str, object]) -> np.ndarray:
    """6×n 几何雅可比。前 3 行是 m，后 3 行是 rad，不能直接相乘当 Yoshikawa 分数。"""
    linear = chain["jacobian"]
    angular = np.stack(chain["axes"], axis=-1)
    return np.concatenate([linear, angular], axis=-2)


def normalized_singular_values(jacobian6: np.ndarray, characteristic_length_m: float) -> np.ndarray:
    """用特征长度把平移行缩成无量纲，只用于秩和最小奇异值，不作为优化分数。"""
    scale = np.array([1.0 / characteristic_length_m] * 3 + [1.0] * 3, dtype=np.float64)
    scaled = jacobian6 * scale[:, None]
    return np.linalg.svd(scaled, compute_uv=False)


def rank_at_length(singular_values: np.ndarray, relative_threshold: float = 1e-4) -> int:
    peak = float(singular_values[0]) if singular_values.size else 0.0
    if peak <= 0.0:
        return 0
    return int(np.sum(singular_values > relative_threshold * peak))


def angular_jacobian_finite_difference(
    design: Design,
    q: np.ndarray,
    step: float = 1e-6,
    *,
    tool_point: str = TOOL_JOINT_ORIGIN,
) -> np.ndarray:
    """世界系角速度列。R(q+δ) R(q)^T 的 vee / δ。足底 site 固定在足体上，角速度与关节体相同。"""
    pose = np.asarray(q, dtype=np.float64).reshape(-1)
    base = forward_chain(design, pose, tool_point=tool_point)
    rotation = quat_matrix(base["ee_quat"][0])
    numeric = np.zeros((3, pose.size), dtype=np.float64)
    for index in range(pose.size):
        bumped = pose.copy()
        bumped[index] += step
        nxt = quat_matrix(forward_chain(design, bumped, tool_point=tool_point)["ee_quat"][0])
        numeric[:, index] = so3_vee(nxt @ rotation.T) / step
    return numeric


def generic_rank_report(
    design: Design,
    *,
    samples: int = 24,
    seed: int = 0,
    lengths_m: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5),
    relative_threshold: float = 1e-4,
    tool_point: str = TOOL_JOINT_ORIGIN,
) -> dict[str, object]:
    """随机一般位形上的 6×n 秩。满秩上限是 min(6, n)。多数样本不满秩才记为结构退化。"""
    count = len(design.module_ids)
    poses = np.random.default_rng(seed).uniform(-JOINT_LIMIT, JOINT_LIMIT, size=(samples, count))
    chain = forward_chain(design, poses, tool_point=tool_point)
    full = geometric_jacobian(chain)
    by_length = []
    deficient = None
    for length in lengths_m:
        ranks = []
        minima = []
        for index in range(samples):
            values = normalized_singular_values(full[index], length)
            ranks.append(rank_at_length(values, relative_threshold))
            minima.append(float(values[-1]))
        fraction = float(np.mean(np.asarray(ranks) < min(6, count)))
        record = {
            "characteristic_length_m": length,
            "relative_threshold": relative_threshold,
            "median_min_singular_value": float(np.median(minima)),
            "rank_deficient_fraction": fraction,
        }
        by_length.append(record)
        if abs(length - 0.1) < 1e-12:
            deficient = fraction
    structurally_degenerate = deficient is not None and deficient > 0.9
    return {
        "driven_joint_count": count,
        "samples": samples,
        "seed": seed,
        "structurally_degenerate": structurally_degenerate,
        "by_length": by_length,
        "note": (
            "秩用特征长度归一化后的奇异值，阈值是最大奇异值的 1e-4。"
            "6×n 矩阵的满秩上限是 min(6, n)，不能拿关节数当满秩。"
            "这不是混单位的 6 行可操作度分数，也不是六维任务可达。"
            "结构退化表示随机位形里绝大多数不满秩，换制造不能凭空补上缺失的瞬时运动。"
        ),
    }


def _axis_directions() -> tuple[np.ndarray, ...]:
    directions = []
    for sign in (1.0, -1.0):
        for axis in range(3):
            direction = np.zeros(3)
            direction[axis] = sign
            directions.append(direction)
    return tuple(directions)


def _octant_directions() -> tuple[np.ndarray, ...]:
    directions = []
    for x in (-1.0, 1.0):
        for y in (-1.0, 1.0):
            for z in (-1.0, 1.0):
                direction = np.array([x, y, z], dtype=np.float64)
                directions.append(direction / np.linalg.norm(direction))
    return tuple(directions)


AXIS_VELOCITY_DIRECTIONS = _axis_directions()
FIXED14_VELOCITY_DIRECTIONS = AXIS_VELOCITY_DIRECTIONS + _octant_directions()


def velocity_lp_direction_min(jacobian: np.ndarray, speed_limit: float, directions: tuple[np.ndarray, ...]) -> float:
    """预固定方向上、|qdot| 不超过上限时的最大速度，再取这些方向里的最小。

    六个坐标轴是一组；再加上八个体对角线后仍不是速度多面体的内切球，也不是全方向最快。
    """
    from scipy.optimize import linprog

    width = jacobian.shape[1]
    worst = float("inf")
    for direction in directions:
        cost = np.zeros(width + 1)
        cost[-1] = -1.0
        equalities = np.zeros((3, width + 1))
        equalities[:, :width] = jacobian
        equalities[:, -1] = -np.asarray(direction, dtype=np.float64)
        bounds = [(-speed_limit, speed_limit)] * width + [(0.0, None)]
        solved = linprog(cost, A_eq=equalities, b_eq=np.zeros(3), bounds=bounds, method="highs")
        if not solved.success:
            return 0.0
        worst = min(worst, float(solved.x[-1]))
    return 0.0 if not np.isfinite(worst) else worst


def velocity_lp_axis_min(jacobian: np.ndarray, speed_limit: float) -> float:
    """六个坐标轴方向上的限速速度最小值。不是全方向速度。"""
    return velocity_lp_direction_min(jacobian, speed_limit, AXIS_VELOCITY_DIRECTIONS)


def translational_manipulability(jacobian: np.ndarray) -> np.ndarray:
    """sqrt(det(Jv Jv^T))。冗余关节会改变这个尺度，不能单独当成速度。"""
    singular = np.linalg.svd(jacobian, compute_uv=False)
    return np.prod(singular, axis=-1)


def velocity_capability(jacobian: np.ndarray, speed_limit: float) -> np.ndarray:
    """伪逆把单位任务速度缩放到关节速度上限内。

    对 7、8 关节这种位置任务冗余臂，这不是关节限速下的最大速度，只是保守下界。
    最终候选另用坐标轴方向的线性规划速度，见 velocity_lp_axis_min。
    """
    samples = jacobian.shape[0]
    capability = np.empty(samples, dtype=np.float64)
    directions = np.eye(3)
    pinv = np.linalg.pinv(jacobian)
    for sample in range(samples):
        worst = np.inf
        jac = jacobian[sample]
        for direction in directions:
            qdot = pinv[sample] @ direction
            residual = float(np.linalg.norm(jac @ qdot - direction))
            if residual > 1e-4:
                worst = 0.0
                break
            peak = float(np.max(np.abs(qdot)))
            speed = 0.0 if peak < 1e-12 else speed_limit / peak
            worst = min(worst, speed)
        capability[sample] = 0.0 if not np.isfinite(worst) else worst
    return capability


def points_in_box(points: np.ndarray, task: TaskSpec) -> np.ndarray:
    """末端是否落在任务盒里。这是几何包含，不是可行到达。"""
    center = np.asarray(task.center_m, dtype=np.float64)
    half = np.asarray(task.half_m, dtype=np.float64)
    return np.all(np.abs(points - center) <= half + 1e-12, axis=-1)


def task_targets(task: TaskSpec) -> np.ndarray:
    """任务盒上的固定格点。与设计无关，分母对所有机构相同。"""
    divisions = int(task.grid_divisions)
    axes = []
    for center, half in zip(task.center_m, task.half_m, strict=True):
        if divisions <= 1:
            axes.append(np.array([float(center)], dtype=np.float64))
        else:
            axes.append(np.linspace(float(center) - float(half), float(center) + float(half), divisions))
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    return np.ascontiguousarray(grid.reshape(-1, 3))


def ik_start_poses(count: int, starts: int, limit: float) -> list[np.ndarray]:
    """确定性的多起点。不用随机数，便于 Python 和浏览器对齐。"""
    poses = [np.zeros(count, dtype=np.float64)]
    for index in range(1, int(starts)):
        pose = np.array(
            [limit * 0.55 * math.sin((index + 1) * (joint + 1) * 0.7) for joint in range(count)],
            dtype=np.float64,
        )
        poses.append(np.clip(pose, -limit, limit))
    return poses


def _damped_step(jacobian: np.ndarray, error: np.ndarray, damping: float) -> np.ndarray:
    gram = jacobian @ jacobian.T + (damping**2) * np.eye(3)
    try:
        lagrange = np.linalg.solve(gram, error)
    except np.linalg.LinAlgError:
        return np.zeros(jacobian.shape[1], dtype=np.float64)
    return jacobian.T @ lagrange


def _pose_is_feasible(chain: dict[str, object], task: TaskSpec) -> bool:
    if bool(collision_mask(chain)[0]):
        return False
    torque = static_torques(chain, task.payload_kg, task.gravity_m_s2)[0]
    return float(np.max(np.abs(torque))) <= float(task.effort_limit_nm) + 1e-9


def solve_target(design: Design, target: np.ndarray, task: TaskSpec) -> tuple[np.ndarray | None, np.ndarray | None]:
    """多起点阻尼最小二乘。成功必须同时满足位置容差、关节限位、无胶囊碰撞和静力矩。"""
    count = len(design.module_ids)
    limit = float(task.joint_sample_limit)
    target = np.asarray(target, dtype=np.float64)
    for start in ik_start_poses(count, task.ik_starts, limit):
        pose = start.copy()
        for _ in range(int(task.ik_iterations)):
            chain = forward_chain(design, pose.reshape(1, -1), tool_point=task.tool_point)
            error = target - chain["ee"][0]
            if float(np.linalg.norm(error)) <= float(task.position_tolerance_m) and _pose_is_feasible(chain, task):
                return pose, chain["jacobian"][0]
            step = _damped_step(chain["jacobian"][0], error, float(task.ik_damping))
            pose = np.clip(pose + step, -limit, limit)
    return None, None


def rotation_angle_rad(current_quat: np.ndarray, desired_quat: np.ndarray) -> float:
    """当前工具系到目标工具系的旋转角，单位 rad。"""
    relative = quat_matrix(desired_quat) @ quat_matrix(current_quat).T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.arccos(cosine))


def so3_log_world(current_quat: np.ndarray, desired_quat: np.ndarray) -> np.ndarray:
    """把当前工具姿态转到目标姿态的世界系旋转向量。

    几何雅可比的转动行是世界系角速度，所以这个误差和工具 site 的世界姿态一致。
    site 相对足体的四元数是常量，不另造一套角速度。
    """
    relative = quat_matrix(desired_quat) @ quat_matrix(current_quat).T
    sine_axis = so3_vee(relative)
    theta = rotation_angle_rad(current_quat, desired_quat)
    sine = float(np.sin(theta))
    if abs(sine) < 1e-8:
        return sine_axis
    return sine_axis * (theta / sine)


def _damped_step_rows(jacobian: np.ndarray, error: np.ndarray, damping: float) -> np.ndarray:
    gram = jacobian @ jacobian.T + (damping**2) * np.eye(jacobian.shape[0])
    try:
        lagrange = np.linalg.solve(gram, error)
    except np.linalg.LinAlgError:
        return np.zeros(jacobian.shape[1], dtype=np.float64)
    return jacobian.T @ lagrange


def solve_pose(
    design: Design,
    target_position: np.ndarray,
    target_quat: np.ndarray,
    task: TaskSpec,
    *,
    position_tolerance_m: float = 0.01,
    rotation_tolerance_rad: float = math.radians(5.0),
    characteristic_length_m: float = 0.1,
    extra_starts: list[np.ndarray] | None = None,
) -> dict[str, object]:
    """多起点六维阻尼最小二乘。位置和 SO(3) 容差分开。

    成功还要同时满足关节限位、胶囊无碰撞和静力矩。没解出不是不可达证明。
    特征长度只用来配平平移误差和转动误差的步长，不进入通过判定。
    """
    count = len(design.module_ids)
    limit = float(task.joint_sample_limit)
    target_position = np.asarray(target_position, dtype=np.float64)
    target_quat = np.asarray(target_quat, dtype=np.float64)
    starts = ik_start_poses(count, task.ik_starts, limit)
    for extra in extra_starts or []:
        starts.append(np.clip(np.asarray(extra, dtype=np.float64), -limit, limit))
    best_pos = float("inf")
    best_rot = float("inf")
    for start in starts:
        pose = start.copy()
        for _ in range(int(task.ik_iterations)):
            chain = forward_chain(design, pose.reshape(1, -1), tool_point=task.tool_point)
            position_error = target_position - chain["ee"][0]
            rotation_error = so3_log_world(chain["ee_quat"][0], target_quat)
            position_miss = float(np.linalg.norm(position_error))
            rotation_miss = rotation_angle_rad(chain["ee_quat"][0], target_quat)
            best_pos = min(best_pos, position_miss)
            best_rot = min(best_rot, rotation_miss)
            if position_miss <= position_tolerance_m and rotation_miss <= rotation_tolerance_rad and _pose_is_feasible(chain, task):
                return {
                    "solved": True,
                    "q": pose,
                    "position_error_m": position_miss,
                    "rotation_error_rad": rotation_miss,
                    "jacobian6": geometric_jacobian(chain)[0],
                }
            scaled = np.concatenate([position_error / characteristic_length_m, rotation_error])
            jacobian6 = geometric_jacobian(chain)[0]
            jacobian6 = jacobian6.copy()
            jacobian6[:3, :] /= characteristic_length_m
            step = _damped_step_rows(jacobian6, scaled, float(task.ik_damping))
            pose = np.clip(pose + step, -limit, limit)
    return {
        "solved": False,
        "q": None,
        "position_error_m": None if not np.isfinite(best_pos) else best_pos,
        "rotation_error_rad": None if not np.isfinite(best_rot) else best_rot,
        "jacobian6": None,
    }


def evaluate_pose_targets(
    design: Design,
    targets: list[dict[str, object]],
    task: TaskSpec,
    *,
    position_tolerance_m: float = 0.01,
    rotation_tolerance_rad: float = math.radians(5.0),
) -> dict[str, object]:
    """固定六维目标集上的可行比例。分母是目标数。秩满不计入通过。"""
    rows = []
    for target in targets:
        solved = solve_pose(
            design,
            np.asarray(target["position_m"], dtype=np.float64),
            np.asarray(target["quat_wxyz"], dtype=np.float64),
            task,
            position_tolerance_m=position_tolerance_m,
            rotation_tolerance_rad=rotation_tolerance_rad,
        )
        rows.append(
            {
                "id": target.get("id"),
                "solved": bool(solved["solved"]),
                "position_error_m": solved["position_error_m"],
                "rotation_error_rad": solved["rotation_error_rad"],
            }
        )
    reached = sum(1 for row in rows if row["solved"])
    return {
        "tool_point": task.tool_point,
        "position_tolerance_m": position_tolerance_m,
        "rotation_tolerance_rad": rotation_tolerance_rad,
        "target_count": len(targets),
        "targets_reached": reached,
        "feasible_fraction": reached / float(len(targets)) if targets else None,
        "targets": rows,
        "definition": (
            "位置误差和 SO(3) 旋转角分开判定。通过还要满足关节限位、胶囊无碰撞和静力矩。"
            "未解出只表示这个多起点预算没找到，不是不可达。六维秩不作为通过条件。"
        ),
    }


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def evaluate_design(
    design: Design,
    task: TaskSpec,
    q: np.ndarray | None = None,
    *,
    include_cloud: bool = False,
    targets: np.ndarray | None = None,
    target_set: str = "linspace_endpoints",
) -> dict[str, object]:
    count = len(design.module_ids)
    if targets is None:
        targets = task_targets(task)
    else:
        targets = np.asarray(targets, dtype=np.float64)
        target_set = "custom" if target_set == "linspace_endpoints" else target_set
    reached_flags: list[bool] = []
    task_manips: list[float] = []
    task_speeds: list[float] = []
    task_speeds_lp: list[float] = []
    task_speeds_lp14: list[float] = []
    torque_margins: list[float] = []
    rank_deficient = 0
    spatial_deficient = 0
    example: list[float] | None = None
    for target in targets:
        pose, jacobian = solve_target(design, target, task)
        reached = pose is not None and jacobian is not None
        reached_flags.append(reached)
        if not reached:
            continue
        if example is None:
            example = pose.tolist()
        manip = float(translational_manipulability(jacobian.reshape(1, 3, count))[0])
        task_manips.append(manip)
        task_speeds.append(float(velocity_capability(jacobian.reshape(1, 3, count), task.velocity_limit_rad_s)[0]))
        if task.compute_velocity_lp:
            task_speeds_lp.append(velocity_lp_axis_min(jacobian.reshape(3, count), task.velocity_limit_rad_s))
        if task.compute_velocity_lp_fixed14:
            task_speeds_lp14.append(velocity_lp_direction_min(jacobian.reshape(3, count), task.velocity_limit_rad_s, FIXED14_VELOCITY_DIRECTIONS))
        solved = forward_chain(design, pose.reshape(1, -1), tool_point=task.tool_point)
        peak = float(np.max(np.abs(static_torques(solved, task.payload_kg, task.gravity_m_s2)[0])))
        torque_margins.append(float(task.effort_limit_nm) - peak)
        if manip < 1e-8:
            rank_deficient += 1
        spatial = normalized_singular_values(geometric_jacobian(solved)[0], 0.1)
        if rank_at_length(spatial) < min(6, count):
            spatial_deficient += 1
    reached_count = int(sum(reached_flags))
    fraction = reached_count / float(targets.shape[0])
    manip_valid = bool(task_manips)
    speed_valid = bool(task_speeds)
    straight = forward_chain(design, np.zeros((1, count)), tool_point=task.tool_point)
    torque0 = static_torques(straight, task.payload_kg, task.gravity_m_s2)[0]
    torque0_peak = float(np.max(np.abs(torque0)))
    pose_hit_rate = None
    pose_hit_se = None
    posture_manip = None
    posture_speed = None
    collision_rate = None
    torque_fail_rate = None
    if task.sample_count > 0:
        if q is None:
            rng = np.random.default_rng(task.seed)
            q = rng.uniform(-task.joint_sample_limit, task.joint_sample_limit, size=(task.sample_count, count))
        joints = _as_batch(q, count)
        chain = forward_chain(design, joints, tool_point=task.tool_point)
        torque = static_torques(chain, task.payload_kg, task.gravity_m_s2)
        collide = collision_mask(chain)
        torque_bad = np.any(np.abs(torque) > float(task.effort_limit_nm) + 1e-9, axis=1)
        inside = points_in_box(chain["ee"], task)
        pose_hit_rate = float(np.mean(inside))
        pose_hit_se = math.sqrt(max(pose_hit_rate * (1.0 - pose_hit_rate), 0.0) / joints.shape[0])
        collision_rate = float(np.mean(collide))
        torque_fail_rate = float(np.mean(torque_bad))
        posture_manip = float(np.median(translational_manipulability(chain["jacobian"])))
        posture_speed = float(np.median(velocity_capability(chain["jacobian"], task.velocity_limit_rad_s)))
    site = sole_site() if task.tool_point == TOOL_SOLE else None
    result: dict[str, object] = {
        "mass_kg": straight["mass_kg"],
        "tool_point": task.tool_point,
        "tool_point_applied": straight["tool_point_applied"],
        "tool_site": None
        if site is None
        else {
            "name": site.name,
            "body": site.body,
            "pos_m": list(site.pos_m),
            "quat_wxyz": list(site.quat_wxyz),
            "source_commit": site.source_commit,
            "source_path": site.source_path,
        },
        "distal_linear_jacobian_norm": float(np.linalg.norm(straight["jacobian"][0][:, -1])),
        "task_feasible_fraction": fraction,
        "task_target_count": int(targets.shape[0]),
        "task_targets_reached": reached_count,
        "grid_divisions": int(task.grid_divisions),
        "target_set": target_set,
        "position_tolerance_m": float(task.position_tolerance_m),
        "ik_starts": int(task.ik_starts),
        "ik_iterations": int(task.ik_iterations),
        "task_coverage_definition": (
            "固定任务格点中，在位置容差、关节限位、胶囊自碰撞和静力矩约束下找到可行解的比例。"
            "分母是格点数。未解出只表示该预算下没找到，不是严格不可达，也不是工作空间体积。"
        ),
        "pose_hit_rate": pose_hit_rate,
        "pose_hit_se": pose_hit_se,
        "pose_hit_definition": "随机关节样本里末端落入任务盒的比例，含碰撞和超力矩姿态。不是任务覆盖率。",
        "sample_count": 0 if pose_hit_rate is None else int(np.asarray(q).reshape(-1, count).shape[0]),
        "task_manipulability_median": _median(task_manips) if manip_valid else None,
        "task_manipulability_valid": manip_valid,
        "task_velocity_median": _median(task_speeds) if speed_valid else None,
        "task_velocity_valid": speed_valid,
        "task_velocity_is_pseudoinverse_lower_bound": True,
        "task_velocity_lp_median_m_s": _median(task_speeds_lp) if task_speeds_lp else None,
        "task_velocity_lp_fixed14_median_m_s": _median(task_speeds_lp14) if task_speeds_lp14 else None,
        "task_velocity_lp_fixed14_definition": "六个坐标轴加八个预固定体对角线，共 14 个方向上的限速速度再取最小。不是全方向速度。",
        "task_velocity_lp_definition": (
            "六个坐标轴正负方向上，关节速度盒约束线性规划的最大速度，再取最小。"
            "只在复核候选上计算。它仍不是整个速度多面体的内切球。"
        ),
        "task_torque_margin_median_nm": _median(torque_margins) if torque_margins else None,
        "position_rank_deficient_fraction": (rank_deficient / reached_count) if reached_count else None,
        "position_rank_note": "这是平移雅可比 3 行的秩是否不足 3。不是姿态 6 维的秩。",
        "task_pose_spatial_rank_deficient_fraction": (spatial_deficient / reached_count) if reached_count else None,
        "spatial_rank_definition": (
            "6 行几何雅可比用特征长度 0.1 m 把平移行除开后做 SVD，"
            "最小奇异值低于最大奇异值的 1e-4 视为不满秩。"
            "只在已经找到的可行任务解上统计。随机一般位形另见 structurally_degenerate。"
        ),
        "task_metric_domain": "只统计该预算下找到的可行任务解。没有可行解时指标无效，优化里用惩罚，不用 NaN。",
        **actuation_report(design),
        "posture_manipulability_median": posture_manip,
        "posture_velocity_median": posture_speed,
        "posture_metric_definition": "全体关节样本的中位数，包含任务区外和不可行姿态，不是任务性能。",
        "collision_rate": collision_rate,
        "torque_fail_rate": torque_fail_rate,
        "q0_collision": bool(collision_mask(straight)[0]),
        "q0_torque_ok": torque0_peak <= float(task.effort_limit_nm) + 1e-9,
        "q0_peak_torque_nm": torque0_peak,
        "q0_is_design_veto": False,
        "q0_note": "零位碰撞或超力矩只作诊断。可重构装配的零位不必是工作姿态，不能据此判定整机不可用。",
        "violation": 0.0,
        "feasible": True,
        "structurally_degenerate": False,
        "example_q": example if example is not None else [0.0] * count,
        "ee_straight": straight["ee"][0].tolist(),
        "unsupported": list(UNSUPPORTED),
        "approximations": [
            "自碰撞用胶囊，半径是网格外接尺寸的近似，不是 STL 精确碰撞",
            "静力矩把连杆质量集中在质心，只平衡重力与末端载荷，不含惯性力",
            "速度能力是伪逆下界，关节速度上限由用户给定，R3 没有实测速度",
            "任务格点可行比例取决于格点分辨率、位置容差和 IK 起点/迭代预算，漏检不等于不可达",
            "末端落盒比例若计算，其二项标准误差只描述关节采样，不能挪用来当任务空间覆盖的误差",
        ],
    }
    if task.reject_structural_degeneracy:
        rank = generic_rank_report(design, samples=8, seed=task.seed, tool_point=task.tool_point)
        result["structurally_degenerate"] = bool(rank["structurally_degenerate"])
        result["generic_rank"] = rank
        if rank["structurally_degenerate"]:
            result["violation"] = 1.0
            result["feasible"] = False
    if include_cloud:
        result["cloud"] = [
            {"p": target.tolist(), "reached": reached}
            for target, reached in zip(targets, reached_flags, strict=True)
        ]
    return result


def baseline_design() -> Design:
    """六模块 ZBot、连接方位全 0。这是 R3 外肢体的对照，不是搜索空间的上界。"""
    return Design(
        module_ids=tuple("zbot_r3" for _ in range(6)),
        twists_rad=tuple(0.0 for _ in range(5)),
        parameters=tuple({} for _ in range(6)),
    )


def actuation_report(design: Design) -> dict[str, object]:
    """驱动关节数不等于随便数模块。当前库里每个模块恰有一个驱动转动关节。"""
    driven = 0
    passive = 0
    for module_id in design.module_ids:
        spec = module_spec(module_id)
        driven += len(spec.joints)
        if module_id == "zbot_r3":
            passive += 1
    return {
        "driven_joint_count": driven,
        "module_count": len(design.module_ids),
        "passive_body_count": passive,
        "actuation_note": (
            "每个已注册模块有 1 个驱动转动关节，所以驱动数等于模块数。"
            "7 或 8 个模块就是 7 或 8 个驱动关节，对三维位置任务是冗余臂，不是恰好 6 驱动。"
            "ZBot 的 a 连杆没有自己的关节，计为被动结构件。"
            "概念轴向连杆随自身关节运动，不计被动件。"
            "位置任务只用平移 3 行。六维几何雅可比满秩上限是 min(6, 驱动数)，秩诊断不是姿态任务可达。"
        ),
    }


def zbot_chain(count: int, twists: tuple[float, ...] | None = None) -> Design:
    if twists is None:
        twists = tuple(0.0 for _ in range(count - 1))
    return Design(
        module_ids=tuple("zbot_r3" for _ in range(count)),
        twists_rad=twists,
        parameters=tuple({} for _ in range(count)),
    )
