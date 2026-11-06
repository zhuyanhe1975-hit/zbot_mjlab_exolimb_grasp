"""Mjlab-Zbot-Exolimb-Grasp 环境配置。"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import joint_pos_rel, joint_vel_rel, last_action
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.events import reset_joints_by_offset, reset_scene_to_default
from mjlab.envs.mdp.terminations import time_out
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from . import mdp
from .props import stage_cfg, tool_cfg
from .robot import robot_cfg


def exolimb_grasp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    joint_noise = Unoise(n_min=-0.01, n_max=0.01)
    cartesian_noise = Unoise(n_min=-0.005, n_max=0.005)
    ee_cfg = SceneEntityCfg("robot", body_names=("foot_1",))

    terms = {
        "joint_pos": ObservationTermCfg(func=joint_pos_rel, noise=joint_noise),
        "joint_vel": ObservationTermCfg(
            func=joint_vel_rel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "ee_to_tool": ObservationTermCfg(
            func=mdp.ee_to_tool,
            params={"asset_cfg": ee_cfg},
            noise=cartesian_noise,
        ),
        "tool_lin_vel": ObservationTermCfg(func=mdp.tool_lin_vel),
        "last_action": ObservationTermCfg(func=last_action),
    }
    observations = {
        "actor": ObservationGroupCfg(terms, enable_corruption=not play),
        "critic": ObservationGroupCfg(terms, enable_corruption=False),
    }

    actions = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"),
            scale=2.0,
            use_default_offset=True,
        )
    }

    joint_offset = (0.0, 0.0) if play else (-0.15, 0.15)
    xy_noise = 0.0 if play else 0.01
    events = {
        "reset_scene_to_default": EventTermCfg(func=reset_scene_to_default, mode="reset"),
        "reset_robot_joints": EventTermCfg(
            func=reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": joint_offset,
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=("joint.*",)),
            },
        ),
        "reset_tool": EventTermCfg(
            func=mdp.reset_tool_xy,
            mode="reset",
            params={"xy_noise": xy_noise},
        ),
        "clear_grasp": EventTermCfg(func=mdp.clear_grasp_latch, mode="reset"),
    }

    rewards = {
        "approach": RewardTermCfg(
            func=mdp.approach_reward,
            weight=1.0,
            params={"std": 0.25, "asset_cfg": ee_cfg},
        ),
        "contact": RewardTermCfg(
            func=mdp.contact_reward,
            weight=2.0,
            params={"sensor_name": "ee_tool_contact"},
        ),
        "success": RewardTermCfg(
            func=mdp.success_reward,
            weight=8.0,
            params={
                "asset_cfg": ee_cfg,
                "sensor_name": "ee_tool_contact",
                "distance": 0.08,
            },
        ),
        "tool_velocity_damping": RewardTermCfg(
            func=mdp.tool_velocity_damping,
            weight=1.5,
            params={
                "asset_cfg": ee_cfg,
                "sensor_name": "ee_tool_contact",
                "distance": 0.08,
                "vel_std": 0.25,
            },
        ),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=time_out, time_out=True),
        "tool_dropped": TerminationTermCfg(
            func=mdp.tool_dropped,
            params={"minimum_height": 0.05},
        ),
    }

    contact = ContactSensorCfg(
        name="ee_tool_contact",
        primary=ContactMatch(mode="subtree", pattern="foot_1", entity="robot"),
        secondary=ContactMatch(mode="geom", pattern="tool_geom", entity="tool"),
        fields=("found", "force"),
        reduce="maxforce",
        num_slots=1,
    )

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=1 if play else 8,
            env_spacing=2.0,
            entities={
                "robot": robot_cfg(),
                "tool": tool_cfg(),
                "stage": stage_cfg(),
            },
            sensors=(contact,),
        ),
        observations=observations,
        actions=actions,
        events=events,
        rewards=rewards,
        terminations=terminations,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="foot_0",
            distance=1.4,
            elevation=-15.0,
            azimuth=135.0,
        ),
        sim=SimulationCfg(
            nconmax=80,
            njmax=250,
            mujoco=MujocoCfg(
                timestep=0.005,
                integrator="implicitfast",
                iterations=8,
                ls_iterations=12,
            ),
        ),
        decimation=4,
        episode_length_s=8.0,
    )
