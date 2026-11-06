"""固定基座六模块 Z-Bot 外肢体。

执行器与 zbot_rl_mjlab_R3 ``src/zbot_rl_mjlab/robot.py`` 中实际用于仿真的
位置伺服一致：kp 15、kd 1.5、力矩上限 20 N·m、armature 0.001。
不采用 USD 转换质量，也不把 armature 改成 0.0036。
"""

from __future__ import annotations

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from .constants import (
    ARMATURE,
    EFFORT_LIMIT,
    JOINT_NAMES,
    PD_DAMPING,
    PD_STIFFNESS,
    STRAIGHT_JOINT_POS,
)

ASSET_PATH = Path(__file__).resolve().parents[2] / "assets" / "zbot_6" / "ZBOT6.xml"


def get_spec() -> mujoco.MjSpec:
    """读取权威 XML，并关掉手臂自碰撞。

    连杆质量、关节阻尼、armature 与行程仍以 XML 为准。
    名义折姿态下相邻模块网格会重叠；抓取任务只需要末端与工具、托盘接触，
    因此手臂几何体彼此不碰撞，但仍与工具和托盘碰撞。
    """
    if not ASSET_PATH.is_file():
        raise FileNotFoundError(f"找不到机械臂模型：{ASSET_PATH}")
    spec = mujoco.MjSpec.from_file(str(ASSET_PATH))
    free_joints = [joint for joint in spec.joints if joint.type == mujoco.mjtJoint.mjJNT_FREE]
    if free_joints:
        raise ValueError("外肢体模型不应再包含自由关节，肩部必须固定。")
    for geom in spec.geoms:
        # contype=1、conaffinity=2：手臂模块彼此不碰撞，
        # 但与 contype=2 的工具、托盘和地面碰撞。
        geom.contype = 1
        geom.conaffinity = 2
    return spec


def robot_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=get_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(
                BuiltinPositionActuatorCfg(
                    target_names_expr=JOINT_NAMES,
                    stiffness=PD_STIFFNESS,
                    damping=PD_DAMPING,
                    effort_limit=EFFORT_LIMIT,
                    armature=ARMATURE,
                ),
            )
        ),
        init_state=EntityCfg.InitialStateCfg(
            # 肩高写在 XML 的 foot_0 上。MJLab 会再套一层 mocap，
            # 这里的根位置必须是原点，否则肩部会被再抬高 0.40 m。
            pos=(0.0, 0.0, 0.0),
            joint_pos=dict(zip(JOINT_NAMES, STRAIGHT_JOINT_POS, strict=True)),
            joint_vel={".*": 0.0},
        ),
    )
