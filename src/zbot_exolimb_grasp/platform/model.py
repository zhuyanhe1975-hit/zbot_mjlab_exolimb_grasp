"""串联设计的统一模型：装配校验、FK、平移雅可比和近似评价。

不支持网格精确碰撞、完整动力学和加速度能力。这些缺口在结果里写成
unsupported，不计入分数。
"""

from __future__ import annotations

import math

import numpy as np

from zbot_exolimb_grasp.config_gen import (
    A_TO_B_POS,
    A_TO_B_QUAT,
    B_TO_A_POS,
    B_TO_A_QUAT,
    CONNECTION_AXIS,
)
from zbot_exolimb_grasp.constants import LINK_A_MASS, LINK_B_MASS

from .library import (
    CONCEPT_CAPSULE_RADIUS_M,
    CONCEPT_LINEAR_DENSITY_KG_PER_M,
    COM_A,
    COM_B,
    ZBOT_CAPSULE_RADIUS_M,
    mate_rule,
    module_spec,
)
from .math3d import Z_AXIS, axis_angle_x, hinge_z, mul, rotate
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


def forward_chain(design: Design, q: np.ndarray) -> dict[str, object]:
    """批量正运动学。末端定义写在模块实现里，ZBot 末模块与 foot_1 原点一致。"""
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
        else:
            raise ValueError(f"没有运动学实现：{module_id}")
    jacobian = np.empty((samples, 3, count), dtype=np.float64)
    for index in range(count):
        jacobian[:, :, index] = np.cross(axes[index], ee - origins[index])
    return {
        "ee": ee,
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


def translational_manipulability(jacobian: np.ndarray) -> np.ndarray:
    singular = np.linalg.svd(jacobian, compute_uv=False)
    return np.prod(singular, axis=-1)


def velocity_capability(jacobian: np.ndarray, speed_limit: float) -> np.ndarray:
    """伪逆把单位任务速度缩放到关节速度上限内。这是下界，不是速度多面体半径。"""
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


def _coverage(points: np.ndarray, task: TaskSpec) -> np.ndarray:
    center = np.asarray(task.center_m, dtype=np.float64)
    half = np.asarray(task.half_m, dtype=np.float64)
    return np.all(np.abs(points - center) <= half + 1e-12, axis=-1)


def evaluate_design(
    design: Design,
    task: TaskSpec,
    q: np.ndarray | None = None,
    *,
    include_cloud: bool = False,
) -> dict[str, object]:
    count = len(design.module_ids)
    if q is None:
        rng = np.random.default_rng(task.seed)
        q = rng.uniform(-task.joint_sample_limit, task.joint_sample_limit, size=(task.sample_count, count))
    joints = _as_batch(q, count)
    chain = forward_chain(design, joints)
    straight = forward_chain(design, np.zeros((1, count)))
    torque = static_torques(chain, task.payload_kg, task.gravity_m_s2)
    torque0 = static_torques(straight, task.payload_kg, task.gravity_m_s2)[0]
    collide = collision_mask(chain)
    collide0 = bool(collision_mask(straight)[0])
    limit = float(task.effort_limit_nm)
    torque_bad = np.any(np.abs(torque) > limit + 1e-9, axis=1)
    torque0_peak = float(np.max(np.abs(torque0)))
    torque0_ok = torque0_peak <= limit + 1e-9
    inside = _coverage(chain["ee"], task)
    coverage = float(np.mean(inside))
    se = math.sqrt(max(coverage * (1.0 - coverage), 0.0) / joints.shape[0])
    manip = translational_manipulability(chain["jacobian"])
    velocity = velocity_capability(chain["jacobian"], task.velocity_limit_rad_s)
    collision_rate = float(np.mean(collide))
    torque_fail_rate = float(np.mean(torque_bad))
    violation = 0.0
    if collide0:
        violation += 1.0
    if not torque0_ok:
        violation += 1.0
    violation += max(0.0, collision_rate - task.rate_limit)
    violation += max(0.0, torque_fail_rate - task.rate_limit)
    good = inside & ~collide & ~torque_bad
    if np.any(good):
        example = joints[int(np.argmax(good))].tolist()
    elif np.any(inside):
        example = joints[int(np.argmax(inside))].tolist()
    else:
        example = [0.0] * count
    result: dict[str, object] = {
        "mass_kg": chain["mass_kg"],
        "task_coverage": coverage,
        "coverage_se": se,
        "sample_count": int(joints.shape[0]),
        "median_translational_manipulability": float(np.median(manip)),
        "median_velocity_capability_m_s": float(np.median(velocity)),
        "collision_rate": collision_rate,
        "torque_fail_rate": torque_fail_rate,
        "q0_collision": collide0,
        "q0_torque_ok": torque0_ok,
        "q0_peak_torque_nm": torque0_peak,
        "violation": violation,
        "feasible": violation == 0.0,
        "example_q": example,
        "ee_straight": straight["ee"][0].tolist(),
        "unsupported": list(UNSUPPORTED),
        "approximations": [
            "自碰撞用胶囊，半径是网格外接尺寸的近似，不是 STL 精确碰撞",
            "静力矩把连杆质量集中在质心，只平衡重力与末端载荷，不含惯性力",
            "速度能力是伪逆下界，关节速度上限由用户给定，R3 没有实测速度",
            "覆盖率是蒙特卡洛比例，标准误差 sqrt(p(1-p)/n)，不是体积证书",
        ],
    }
    if include_cloud:
        step = max(1, joints.shape[0] // 120)
        chosen = np.arange(0, joints.shape[0], step)[:120]
        result["cloud"] = [
            {
                "p": chain["ee"][int(i)].tolist(),
                "inside": bool(inside[int(i)]),
                "collide": bool(collide[int(i)]),
                "torque_ok": not bool(torque_bad[int(i)]),
            }
            for i in chosen
        ]
    return result


def baseline_design() -> Design:
    """六模块 ZBot、连接方位全 0。这是 R3 外肢体的对照，不是搜索空间的上界。"""
    return Design(
        module_ids=tuple("zbot_r3" for _ in range(6)),
        twists_rad=tuple(0.0 for _ in range(5)),
        parameters=tuple({} for _ in range(6)),
    )


def zbot_chain(count: int, twists: tuple[float, ...] | None = None) -> Design:
    if twists is None:
        twists = tuple(0.0 for _ in range(count - 1))
    return Design(
        module_ids=tuple("zbot_r3" for _ in range(count)),
        twists_rad=twists,
        parameters=tuple({} for _ in range(count)),
    )
