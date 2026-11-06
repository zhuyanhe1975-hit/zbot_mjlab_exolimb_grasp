"""外肢体抓取的观测、奖励和复位。

成功条件：末端模块 ``foot_1`` 与工具接触，并且两者原点距离小于 0.08 m。
抓住之后锁存该事件，并奖励工具线速度变小。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from .constants import GRASP_DISTANCE

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot", body_names=("foot_1",))


def _env_ids(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    return env_ids


def _ee_position(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot: Entity = env.scene[asset_cfg.name]
    position = robot.data.body_link_pos_w[:, asset_cfg.body_ids]
    if position.ndim == 3:
        position = position.squeeze(1)
    return position


def ee_tool_distance(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    tool: Entity = env.scene["tool"]
    return torch.linalg.norm(_ee_position(env, asset_cfg) - tool.data.root_link_pos_w, dim=-1)


def ee_to_tool(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """工具原点相对末端原点的向量，形状 (环境数, 3)。"""
    tool: Entity = env.scene["tool"]
    return tool.data.root_link_pos_w - _ee_position(env, asset_cfg)


def tool_lin_vel(env: ManagerBasedRlEnv) -> torch.Tensor:
    tool: Entity = env.scene["tool"]
    return tool.data.root_link_lin_vel_w


def in_tool_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    found = sensor.data.found
    if found is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return found.reshape(env.num_envs, -1).amax(dim=-1) > 0


def grasp_success_mask(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROBOT,
    sensor_name: str = "ee_tool_contact",
    distance: float = GRASP_DISTANCE,
) -> torch.Tensor:
    return in_tool_contact(env, sensor_name) & (ee_tool_distance(env, asset_cfg) < distance)


def _latch(env: ManagerBasedRlEnv) -> torch.Tensor:
    latch = getattr(env, "_exolimb_grasp_latched", None)
    if latch is None or latch.shape[0] != env.num_envs or latch.device != torch.device(env.device):
        latch = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._exolimb_grasp_latched = latch
    return latch


def grasp_latched(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROBOT,
    sensor_name: str = "ee_tool_contact",
    distance: float = GRASP_DISTANCE,
) -> torch.Tensor:
    latch = _latch(env)
    latch |= grasp_success_mask(env, asset_cfg, sensor_name, distance)
    return latch


def clear_grasp_latch(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
    latch = getattr(env, "_exolimb_grasp_latched", None)
    if latch is None:
        return
    latch[_env_ids(env, env_ids)] = False


def approach_reward(
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    distance = ee_tool_distance(env, asset_cfg)
    return torch.exp(-torch.square(distance) / std**2)


def contact_reward(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    return in_tool_contact(env, sensor_name).float()


def success_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROBOT,
    sensor_name: str = "ee_tool_contact",
    distance: float = GRASP_DISTANCE,
) -> torch.Tensor:
    return grasp_success_mask(env, asset_cfg, sensor_name, distance).float()


def tool_velocity_damping(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROBOT,
    sensor_name: str = "ee_tool_contact",
    distance: float = GRASP_DISTANCE,
    vel_std: float = 0.25,
) -> torch.Tensor:
    """抓住之后奖励工具变慢。未抓住时该项为零。"""
    latched = grasp_latched(env, asset_cfg, sensor_name, distance).float()
    speed = torch.linalg.norm(tool_lin_vel(env), dim=-1)
    return latched * torch.exp(-speed / vel_std)


def reset_tool_xy(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    xy_noise: float,
) -> None:
    """在默认球心附近微扰工具的水平位置，并保持竖直高度。"""
    ids = _env_ids(env, env_ids)
    tool: Entity = env.scene["tool"]
    root = tool.data.default_root_state[ids].clone()
    if xy_noise > 0.0:
        noise = (torch.rand(ids.shape[0], 2, device=env.device) * 2.0 - 1.0) * xy_noise
        root[:, 0:2] += noise
    root[:, 0:3] += env.scene.env_origins[ids]
    tool.write_root_state_to_sim(root, env_ids=ids)


def tool_dropped(
    env: ManagerBasedRlEnv,
    minimum_height: float,
) -> torch.Tensor:
    tool: Entity = env.scene["tool"]
    return tool.data.root_link_pos_w[:, 2] < minimum_height
