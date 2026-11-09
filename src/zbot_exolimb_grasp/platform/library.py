"""可扩展模块库。

实物数据只来自已核对的 ZBot R3。概念模块用独立 id、kind 和假设说明，
不能被当成第二套硬件。demo_flange 用来证明没有装配规则就不能串联。
"""

from __future__ import annotations

import math

from zbot_exolimb_grasp.config_gen import (
    A_TO_B_POS,
    A_TO_B_QUAT,
    B_TO_A_POS,
    B_TO_A_QUAT,
    CONNECTION_AXIS,
)
from zbot_exolimb_grasp.constants import EFFORT_LIMIT, LINK_A_MASS, LINK_B_MASS, SOURCE_COMMIT

from .schema import COUNT_MAX, JOINT_LIMIT, JointSpec, MateRule, ModuleSpec, ParameterSpec

ZBOT_FAMILY = "zbot_flange"
DEMO_FAMILY = "demo_flange"

# 概念轴向连杆的线密度。不是测量值。
CONCEPT_LINEAR_DENSITY_KG_PER_M = 4.0
CONCEPT_CAPSULE_RADIUS_M = 0.025
ZBOT_CAPSULE_RADIUS_M = 0.04

COM_A = (-0.0022495, -0.000029534, 0.042888)
COM_B = (-0.02292, -0.00016188, 0.03226)

_ZBOT_JOINT = JointSpec(
    name="revolute",
    axis=(0.0, 0.0, 1.0),
    lower=-JOINT_LIMIT,
    upper=JOINT_LIMIT,
    effort_limit_nm=EFFORT_LIMIT,
    velocity_limit_rad_s=None,
    velocity_is_assumption=True,
)

MODULES: dict[str, ModuleSpec] = {
    "zbot_r3": ModuleSpec(
        id="zbot_r3",
        label="ZBot R3 关节模块",
        kind="physical",
        summary="a/b 连杆、质量和关节行程来自 R3 的 ZBOT6。尺寸锁定，不能在实物模式里缩放。",
        in_family=ZBOT_FAMILY,
        out_family=ZBOT_FAMILY,
        joints=(_ZBOT_JOINT,),
        parameters=(),
        source=f"zbot_rl_mjlab_R3@{SOURCE_COMMIT} assets/zbot_6/ZBOT6.xml",
    ),
    "concept_axial": ModuleSpec(
        id="concept_axial",
        label="轴向概念连杆",
        kind="conceptual",
        summary="绕根部 Z 轴转动、连杆沿关节后的 X 向伸出。接口族故意做成 zbot_flange，用来演示可插入 ZBot 链；几何和质量是假设。",
        in_family=ZBOT_FAMILY,
        out_family=ZBOT_FAMILY,
        joints=(_ZBOT_JOINT,),
        parameters=(
            ParameterSpec("length_m", "m", 0.06, 0.16, 0.10, "连杆长度"),
        ),
        source="概念设计，不是实物模块",
    ),
    "demo_unreleased": ModuleSpec(
        id="demo_unreleased",
        label="未接入接口示例",
        kind="conceptual",
        summary="另一接口族的占位模块。没有与 zbot_flange 的装配规则，不能和 ZBot 或轴向概念连杆串联。",
        in_family=DEMO_FAMILY,
        out_family=DEMO_FAMILY,
        joints=(_ZBOT_JOINT,),
        parameters=(
            ParameterSpec("length_m", "m", 0.08, 0.12, 0.10, "示例长度"),
        ),
        source="规则示例，不是实物模块",
    ),
}

MATE_RULES: tuple[MateRule, ...] = (
    MateRule(ZBOT_FAMILY, ZBOT_FAMILY, (0.0, math.pi / 2)),
    MateRule(DEMO_FAMILY, DEMO_FAMILY, (0.0, math.pi / 2)),
)


def module_spec(module_id: str) -> ModuleSpec:
    try:
        return MODULES[module_id]
    except KeyError as exc:
        raise KeyError(f"模块库里没有 {module_id}") from exc


def mate_rule(parent_id: str, child_id: str) -> MateRule | None:
    parent = module_spec(parent_id)
    child = module_spec(child_id)
    for rule in MATE_RULES:
        if rule.parent_family == parent.out_family and rule.child_family == child.in_family:
            return rule
    return None


def library_catalog() -> list[dict[str, object]]:
    """给界面和导出用的纯数据目录。"""
    items = []
    for spec in MODULES.values():
        items.append(
            {
                "id": spec.id,
                "label": spec.label,
                "kind": spec.kind,
                "summary": spec.summary,
                "in_family": spec.in_family,
                "out_family": spec.out_family,
                "source": spec.source,
                "parameters": [
                    {
                        "name": item.name,
                        "unit": item.unit,
                        "lower": item.lower,
                        "upper": item.upper,
                        "default": item.default,
                        "label": item.label,
                    }
                    for item in spec.parameters
                ],
                "effort_limit_nm": spec.joints[0].effort_limit_nm,
                "joint_range_rad": [spec.joints[0].lower, spec.joints[0].upper],
                "velocity_is_assumption": spec.joints[0].velocity_is_assumption,
            }
        )
    return items


def default_parameters(module_id: str) -> dict[str, float]:
    spec = module_spec(module_id)
    return {item.name: item.default for item in spec.parameters}


def geometry_constants() -> dict[str, object]:
    return {
        "a_to_b_pos": [float(v) for v in A_TO_B_POS],
        "a_to_b_quat": [float(v) for v in A_TO_B_QUAT],
        "b_to_a_pos": [float(v) for v in B_TO_A_POS],
        "b_to_a_quat": [float(v) for v in B_TO_A_QUAT],
        "connection_axis": [float(v) for v in CONNECTION_AXIS],
        "com_a": list(COM_A),
        "com_b": list(COM_B),
        "mass_a": LINK_A_MASS,
        "mass_b": LINK_B_MASS,
        "zbot_capsule_radius_m": ZBOT_CAPSULE_RADIUS_M,
        "concept_linear_density_kg_per_m": CONCEPT_LINEAR_DENSITY_KG_PER_M,
        "concept_capsule_radius_m": CONCEPT_CAPSULE_RADIUS_M,
        "count_max": COUNT_MAX,
    }
