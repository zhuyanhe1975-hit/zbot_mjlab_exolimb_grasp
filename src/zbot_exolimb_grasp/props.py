"""托盘、地面和橙色工具球。"""

from __future__ import annotations

import mujoco
from mjlab.entity import EntityCfg

from .constants import TOOL_MASS, TOOL_POS, TOOL_RADIUS, TRAY_HALF, TRAY_POS

# 手臂是 contype=1、conaffinity=2，因此手臂之间不碰撞。
# 工具、托盘和地面用 contype=2、conaffinity=3：既托住工具，也挡住手臂。
_PROP_CONTYPE = 2
_PROP_CONAFFINITY = 3


def get_tool_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="tool", pos=TOOL_POS)
    body.add_freejoint(name="tool_free")
    body.add_geom(
        name="tool_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=(TOOL_RADIUS, 0.0, 0.0),
        mass=TOOL_MASS,
        rgba=(1.0, 0.45, 0.05, 1.0),
        friction=(1.2, 0.05, 0.01),
        condim=3,
        contype=_PROP_CONTYPE,
        conaffinity=_PROP_CONAFFINITY,
    )
    return spec


def get_stage_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    tray = spec.worldbody.add_body(name="tray", pos=TRAY_POS)
    tray.add_geom(
        name="tray_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=TRAY_HALF,
        rgba=(0.45, 0.34, 0.24, 1.0),
        friction=(1.0, 0.05, 0.01),
        condim=3,
        contype=_PROP_CONTYPE,
        conaffinity=_PROP_CONAFFINITY,
    )
    floor = spec.worldbody.add_body(name="floor", pos=(0.0, 0.0, 0.0))
    floor.add_geom(
        name="floor_geom",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=(1.6, 1.6, 0.1),
        rgba=(0.82, 0.84, 0.86, 1.0),
        contype=_PROP_CONTYPE,
        conaffinity=_PROP_CONAFFINITY,
    )
    return spec


def tool_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=get_tool_spec,
        init_state=EntityCfg.InitialStateCfg(pos=TOOL_POS),
    )


def stage_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=get_stage_spec,
        # 托盘和地面的位置已经写在规格里。mocap 根保持在原点，避免再平移一次。
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
