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
            chain = forward_chain(design, pose.reshape(1, -1))
            error = target - chain["ee"][0]
            if float(np.linalg.norm(error)) <= float(task.position_tolerance_m) and _pose_is_feasible(chain, task):
                return pose, chain["jacobian"][0]
            step = _damped_step(chain["jacobian"][0], error, float(task.ik_damping))
            pose = np.clip(pose + step, -limit, limit)
    return None, None


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def evaluate_design(
    design: Design,
    task: TaskSpec,
    q: np.ndarray | None = None,
    *,
    include_cloud: bool = False,
) -> dict[str, object]:
    count = len(design.module_ids)
    targets = task_targets(task)
    reached_flags: list[bool] = []
    task_manips: list[float] = []
    task_speeds: list[float] = []
    torque_margins: list[float] = []
    rank_deficient = 0
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
        solved = forward_chain(design, pose.reshape(1, -1))
        peak = float(np.max(np.abs(static_torques(solved, task.payload_kg, task.gravity_m_s2)[0])))
        torque_margins.append(float(task.effort_limit_nm) - peak)
        if manip < 1e-8:
            rank_deficient += 1
    reached_count = int(sum(reached_flags))
    fraction = reached_count / float(targets.shape[0])
    manip_valid = bool(task_manips)
    speed_valid = bool(task_speeds)
    straight = forward_chain(design, np.zeros((1, count)))
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
        chain = forward_chain(design, joints)
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
    result: dict[str, object] = {
        "mass_kg": straight["mass_kg"],
        "task_feasible_fraction": fraction,
        "task_target_count": int(targets.shape[0]),
        "task_targets_reached": reached_count,
        "grid_divisions": int(task.grid_divisions),
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
        "task_torque_margin_median_nm": _median(torque_margins) if torque_margins else None,
        "position_rank_deficient_fraction": (rank_deficient / reached_count) if reached_count else None,
        "position_rank_note": "这是平移雅可比 3 行的秩是否不足 3。不是姿态 6 维的秩，6 个关节也不表示任意姿态可控。",
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
            "ZBot 的 a 连杆没有自己的关节，计为被动结构件。"
            "概念轴向连杆随自身关节运动，不计被动件。"
            "6 个驱动关节只说明位置任务的雅可比最多 3 行；本轮不保证姿态 6 维可控。"
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
