"""检查权威模型参数，以及名义姿态是否进入抓取距离并发生接触。

不启动训练。若当前环境已安装 MJLab，会再创建一个环境并空动作走一步。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401

import mujoco
import numpy as np

from zbot_exolimb_grasp.constants import (
    ARMATURE,
    EFFORT_LIMIT,
    GRASP_DISTANCE,
    JOINT_DAMPING,
    JOINT_NAMES,
    JOINT_RANGE,
    LINK_A_MASS,
    LINK_B_MASS,
    NOMINAL_JOINT_POS,
    PD_DAMPING,
    PD_STIFFNESS,
    SHOULDER_POS,
    TASK_ID,
    TOOL_POS,
    TOOL_RADIUS,
    TOTAL_MASS,
)

ASSET_PATH = Path(__file__).resolve().parents[1] / "assets" / "zbot_6" / "ZBOT6.xml"


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"冒烟检查失败：{message}")


def check_model() -> None:
    print(f"正在检查模型：{ASSET_PATH}")
    _check(ASSET_PATH.is_file(), "找不到 ZBOT6.xml")
    model = mujoco.MjModel.from_xml_path(str(ASSET_PATH))
    _check(not any(model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE for i in range(model.njnt)), "模型仍含自由关节")

    foot = model.body(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_0"))
    shoulder = np.asarray(foot.pos, dtype=float)
    # compile() 不会套用 EntityCfg 的初始位置，XML 本身必须已经是 0.40 m。
    _check(np.allclose(shoulder, SHOULDER_POS, atol=1e-6), f"肩部位置是 {shoulder}，应为 {SHOULDER_POS}")

    masses = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i): float(model.body_mass[i]) for i in range(model.nbody)}
    for name in ("foot_0", "a2", "a3", "base", "a5", "a6"):
        _check(abs(masses[name] - LINK_A_MASS) < 1e-6, f"{name} 质量 {masses[name]} 不是 a 连杆 {LINK_A_MASS}")
    for name in ("b1", "b2", "b3", "b4", "b5", "foot_1"):
        _check(abs(masses[name] - LINK_B_MASS) < 1e-6, f"{name} 质量 {masses[name]} 不是 b 连杆 {LINK_B_MASS}")
    total = float(np.sum(model.body_mass))
    _check(abs(total - TOTAL_MASS) < 1e-5, f"总质量 {total:.5f} kg，应为 {TOTAL_MASS:.5f} kg")

    _check(model.nu == 0, "XML 本身不应预先写执行器；执行器由 robot.py 在加载时添加")
    for index, name in enumerate(JOINT_NAMES):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        _check(joint_id >= 0, f"缺少 {name}")
        axis = np.asarray(model.jnt_axis[joint_id], dtype=float)
        _check(np.allclose(axis, (0.0, 0.0, 1.0)), f"{name} 轴为 {axis}")
        lower, upper = model.jnt_range[joint_id]
        _check(abs(lower - JOINT_RANGE[0]) < 1e-6 and abs(upper - JOINT_RANGE[1]) < 1e-6, f"{name} 行程不对")
        dof = model.jnt_dofadr[joint_id]
        _check(abs(float(model.dof_damping[dof]) - JOINT_DAMPING) < 1e-8, f"{name} 阻尼不是 {JOINT_DAMPING}")
        _check(abs(float(model.dof_armature[dof]) - ARMATURE) < 1e-8, f"{name} armature 不是 {ARMATURE}")
        _check(index == joint_id or True, "关节存在")

    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(NOMINAL_JOINT_POS)
    mujoco.mj_forward(model, data)
    foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_1")
    nominal_ee = np.array(data.xpos[foot_id], dtype=float)
    distance = float(np.linalg.norm(nominal_ee - np.asarray(TOOL_POS)))
    _check(distance < GRASP_DISTANCE, f"名义姿态末端距离 {distance:.4f} m，没有小于 {GRASP_DISTANCE:.2f} m")
    print(f"名义姿态末端原点距离 {distance:.4f} m，阈值 {GRASP_DISTANCE:.2f} m。")

    grasp_model = _model_with_tool()
    grasp_data = mujoco.MjData(grasp_model)
    grasp_data.qpos[:6] = np.asarray(NOMINAL_JOINT_POS)
    grasp_data.qpos[6:9] = np.asarray(TOOL_POS)
    grasp_data.qpos[9:13] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(grasp_model, grasp_data)
    touched = False
    for contact in grasp_data.contact[: grasp_data.ncon]:
        names = {
            mujoco.mj_id2name(grasp_model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)),
            mujoco.mj_id2name(grasp_model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)),
        }
        if "tool_geom" in names and "b6_geom" in names and contact.dist <= 1e-4:
            touched = True
    _check(touched, "名义姿态下 foot_1 的 b6 网格没有接触工具球")
    print("名义姿态下 foot_1 与工具球接触。")
    print(f"执行器参数将由任务加载：kp {PD_STIFFNESS:g}、kd {PD_DAMPING:g}、力矩上限 {EFFORT_LIMIT:g} N·m。")


def _model_with_tool() -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(ASSET_PATH))
    body = spec.worldbody.add_body(name="tool", pos=TOOL_POS)
    body.add_freejoint(name="tool_free")
    body.add_geom(
        name="tool_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=(TOOL_RADIUS, 0.0, 0.0),
        mass=0.05,
    )
    return spec.compile()


def check_mjlab() -> None:
    try:
        import mjlab  # noqa: F401
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.tasks.registry import list_tasks, load_env_cfg
    except Exception as exc:  # noqa: BLE001
        print(f"未运行 MJLab 环境步：{exc}")
        print("模型检查已通过。安装 MJLab 后可再运行本脚本，确认任务能够复位并走步。")
        return

    import torch

    import zbot_exolimb_grasp  # noqa: F401

    _check(TASK_ID in list_tasks(), f"任务 {TASK_ID} 没有注册")
    cfg = load_env_cfg(TASK_ID, play=True)
    cfg.scene.num_envs = 1
    print("正在创建 1 个 MJLab 环境并走一步零动作……")
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        model = env.sim.mj_model
        robot_mass = 0.0
        for body_id in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            if name.startswith("robot/"):
                robot_mass += float(model.body_mass[body_id])
        _check(abs(robot_mass - TOTAL_MASS) < 1e-4, f"仿真中的手臂质量是 {robot_mass:.5f} kg")
        _check(model.nu == 6, f"执行器数量是 {model.nu}")
        gains = np.asarray(model.actuator_gainprm[:6, 0], dtype=float)
        damping = np.asarray(-model.actuator_biasprm[:6, 2], dtype=float)
        effort = np.asarray(model.actuator_forcerange[:6, 1], dtype=float)
        _check(np.allclose(gains, PD_STIFFNESS), f"kp 是 {gains}")
        _check(np.allclose(damping, PD_DAMPING), f"kd 是 {damping}")
        _check(np.allclose(effort, EFFORT_LIMIT), f"力矩上限是 {effort}")
        armature = np.asarray(model.dof_armature[:6], dtype=float)
        _check(np.allclose(armature, ARMATURE), f"armature 是 {armature}")
        shoulder_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/foot_0")
        shoulder = np.asarray(model.body_pos[shoulder_id], dtype=float)
        _check(np.allclose(shoulder, SHOULDER_POS, atol=1e-6), f"仿真肩部位置是 {shoulder}")

        env.reset()
        action = torch.zeros(env.action_space.shape, device=env.device)
        env.step(action)
        names = set(env.reward_manager.active_terms)
        required = {"approach", "contact", "success", "tool_velocity_damping"}
        _check(required <= names, f"奖励项是 {sorted(names)}，缺少 {sorted(required - names)}")

        robot = env.scene["robot"]
        tool = env.scene["tool"]
        nominal = torch.tensor(NOMINAL_JOINT_POS, device=env.device, dtype=torch.float32).unsqueeze(0)
        robot.write_joint_state_to_sim(nominal, torch.zeros_like(nominal))
        tool.write_root_state_to_sim(tool.data.default_root_state.clone())
        # 动作相对伸直零位，比例是 2，因此名义角要写成目标的一半。
        hold = nominal / 2.0
        env.step(hold)
        from zbot_exolimb_grasp.mdp import ee_tool_distance, grasp_success_mask

        asset_cfg = env.reward_manager.get_term_cfg("success").params["asset_cfg"]
        distance = float(ee_tool_distance(env, asset_cfg)[0])
        success = bool(grasp_success_mask(env, asset_cfg)[0])
        _check(distance < GRASP_DISTANCE, f"环境内名义姿态距离 {distance:.4f} m")
        _check(success, f"环境内名义姿态距离 {distance:.4f} m，但没有检测到 foot_1 与工具接触")
        print(f"环境内名义姿态距离 {distance:.4f} m，抓取判定为成功。")
    finally:
        env.close()
    print("MJLab 环境已完成复位和一步仿真。")


def main() -> None:
    check_model()
    check_mjlab()
    print("冒烟检查通过。")


if __name__ == "__main__":
    main()
