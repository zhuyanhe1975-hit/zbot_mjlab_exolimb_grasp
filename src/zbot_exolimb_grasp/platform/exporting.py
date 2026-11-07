"""设计与结果的 JSON、CSV、MJCF 导出。MJCF 只用本平台能解释的子集。"""

from __future__ import annotations

import csv
import io
import xml.etree.ElementTree as ET

import numpy as np

from zbot_exolimb_grasp.config_gen import A_TO_B_POS, A_TO_B_QUAT, B_TO_A_POS
from zbot_exolimb_grasp.constants import LINK_A_MASS, LINK_B_MASS

from .library import COM_A, COM_B, CONCEPT_LINEAR_DENSITY_KG_PER_M
from .math3d import Z_AXIS, mul, rotate
from .model import forward_chain, zbot_attachment
from .schema import JOINT_LIMIT, Design


def _fmt(values) -> str:
    return " ".join(f"{float(value):.8g}" for value in values)


def _quat_attr(quat: np.ndarray) -> str:
    return _fmt(quat)


def export_mjcf(design: Design) -> str:
    root = ET.Element("mujoco", model="modular_arm")
    ET.SubElement(root, "compiler", angle="radian")
    world = ET.SubElement(root, "worldbody")
    parent = world
    for index, module_id in enumerate(design.module_ids):
        last = index == len(design.module_ids) - 1
        twist = design.twists_rad[index] if not last else 0.0
        if index == 0:
            pos = design.base_position_m
            quat = (1.0, 0.0, 0.0, 0.0)
        else:
            pos = None
            quat = None
        if module_id == "zbot_r3":
            body = ET.SubElement(parent, "body", name=f"m{index}_a")
            if pos is not None:
                body.set("pos", _fmt(pos))
                body.set("quat", _quat_attr(np.array(quat)))
            ET.SubElement(body, "inertial", pos=_fmt(COM_A), mass=f"{LINK_A_MASS:.8g}")
            joint_body = ET.SubElement(
                body,
                "body",
                name=f"m{index}_b",
                pos=_fmt(A_TO_B_POS),
                quat=_quat_attr(np.asarray(A_TO_B_QUAT)),
            )
            ET.SubElement(
                joint_body,
                "joint",
                name=f"joint{index + 1}",
                type="hinge",
                axis="0 0 1",
                range=f"{-JOINT_LIMIT:.4g} {JOINT_LIMIT:.4g}",
            )
            ET.SubElement(joint_body, "inertial", pos=_fmt(COM_B), mass=f"{LINK_B_MASS:.8g}")
            if last:
                ET.SubElement(joint_body, "site", name="ee", pos="0 0 0")
                parent = joint_body
            else:
                child = ET.SubElement(
                    joint_body,
                    "body",
                    name=f"m{index + 1}_mount",
                    pos=_fmt(B_TO_A_POS),
                    quat=_quat_attr(zbot_attachment(twist)),
                )
                parent = child
        else:
            length = float(design.parameters[index]["length_m"])
            mass = CONCEPT_LINEAR_DENSITY_KG_PER_M * length
            body = ET.SubElement(parent, "body", name=f"m{index}_axial")
            if pos is not None:
                body.set("pos", _fmt(pos))
                body.set("quat", _quat_attr(np.array(quat)))
            ET.SubElement(
                body,
                "joint",
                name=f"joint{index + 1}",
                type="hinge",
                axis="0 0 1",
                range=f"{-JOINT_LIMIT:.4g} {JOINT_LIMIT:.4g}",
            )
            ET.SubElement(body, "inertial", pos=_fmt((length * 0.5, 0.0, 0.0)), mass=f"{mass:.8g}")
            if last:
                ET.SubElement(body, "site", name="ee", pos=_fmt((length, 0.0, 0.0)))
                parent = body
            else:
                from .math3d import axis_angle_x

                child = ET.SubElement(
                    body,
                    "body",
                    name=f"m{index + 1}_mount",
                    pos=_fmt((length, 0.0, 0.0)),
                    quat=_quat_attr(axis_angle_x(twist)),
                )
                parent = child
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


def _parse_vec(text: str | None, width: int) -> np.ndarray:
    if not text:
        return np.array([1.0, 0.0, 0.0, 0.0][:width] if width == 4 else [0.0] * width, dtype=np.float64)
    values = [float(part) for part in text.split()]
    if len(values) != width:
        raise ValueError(f"向量长度应为 {width}：{text}")
    return np.asarray(values, dtype=np.float64)


def mjcf_end_effector(xml: str, q: np.ndarray) -> np.ndarray:
    """用本平台对 MJCF 子集的解释求末端。用来核对导出，不调用 MuJoCo。"""
    root = ET.fromstring(xml)
    world = root.find("worldbody")
    if world is None:
        raise ValueError("MJCF 缺少 worldbody")
    joints: list[ET.Element] = []

    def collect(body: ET.Element) -> None:
        for child in list(body):
            if child.tag == "joint":
                joints.append(child)
            elif child.tag == "body":
                collect(child)

    for body in list(world):
        if body.tag == "body":
            collect(body)
    if len(joints) != len(q):
        raise ValueError(f"MJCF 关节数 {len(joints)} 与姿态 {len(q)} 不一致")
    cursor = {"i": 0}

    def walk(body: ET.Element, parent_pos: np.ndarray, parent_quat: np.ndarray) -> np.ndarray | None:
        pos = parent_pos + rotate(parent_quat, _parse_vec(body.get("pos"), 3))
        quat = mul(parent_quat, _parse_vec(body.get("quat"), 4))
        joint = body.find("joint")
        if joint is not None:
            angle = float(q[cursor["i"]])
            cursor["i"] += 1
            axis = _parse_vec(joint.get("axis"), 3)
            if not np.allclose(axis, Z_AXIS):
                raise ValueError("这一版校验只接受关节轴 0 0 1")
            quat = mul(quat, np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]))
        site = body.find("site")
        if site is not None and site.get("name") == "ee":
            return pos + rotate(quat, _parse_vec(site.get("pos"), 3))
        for child in list(body):
            if child.tag == "body":
                found = walk(child, pos, quat)
                if found is not None:
                    return found
        return None

    identity_pos = np.zeros(3)
    identity_quat = np.array([1.0, 0.0, 0.0, 0.0])
    for body in list(world):
        if body.tag == "body":
            found = walk(body, identity_pos, identity_quat)
            if found is not None:
                return found
    raise ValueError("MJCF 没有名为 ee 的 site")


def assert_mjcf_matches(design: Design, samples: int = 5, seed: int = 0) -> None:
    xml = export_mjcf(design)
    rng = np.random.default_rng(seed)
    count = len(design.module_ids)
    poses = [np.zeros(count)]
    poses.extend(rng.uniform(-1.0, 1.0, size=(samples, count)))
    for pose in poses:
        exported = mjcf_end_effector(xml, pose)
        model = forward_chain(design, pose.reshape(1, -1))["ee"][0]
        if not np.allclose(exported, model, atol=1e-6):
            raise AssertionError(f"MJCF 末端 {exported} 与内部模型 {model} 不一致")


def candidates_csv(result: dict[str, object]) -> str:
    buffer = io.StringIO()
    fields = [
        "index",
        "feasible",
        "modules",
        "twists_rad",
        "mass_kg",
        "task_coverage",
        "coverage_se",
        "median_translational_manipulability",
        "median_velocity_capability_m_s",
        "collision_rate",
        "torque_fail_rate",
        "violation",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    rows = list(result.get("feasible_pareto") or [])
    baseline = result.get("baseline")
    if isinstance(baseline, dict):
        rows = [{"design": baseline["design"], "metrics": baseline["metrics"], "violation": 0.0}] + rows
    for index, row in enumerate(rows):
        metrics = row["metrics"]
        modules = row["design"]["modules"]
        writer.writerow(
            {
                "index": index,
                "feasible": int(float(row.get("violation", metrics.get("violation", 0.0))) == 0.0),
                "modules": " ".join(item["id"] for item in modules),
                "twists_rad": " ".join(
                    "" if item["twist_to_next_rad"] is None else f"{item['twist_to_next_rad']:.6g}"
                    for item in modules[:-1]
                ),
                "mass_kg": f"{metrics['mass_kg']:.6f}",
                "task_coverage": f"{metrics['task_coverage']:.6f}",
                "coverage_se": f"{metrics['coverage_se']:.6f}",
                "median_translational_manipulability": f"{metrics['median_translational_manipulability']:.8f}",
                "median_velocity_capability_m_s": f"{metrics['median_velocity_capability_m_s']:.6f}",
                "collision_rate": f"{metrics['collision_rate']:.6f}",
                "torque_fail_rate": f"{metrics['torque_fail_rate']:.6f}",
                "violation": f"{float(row.get('violation', 0.0)):.6f}",
            }
        )
    return buffer.getvalue()
