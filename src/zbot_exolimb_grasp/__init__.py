"""注册 Mjlab-Zbot-Exolimb-Grasp。

导入本包即可把任务放进 MJLab 注册表。安装后，MJLab 也会通过
``mjlab.tasks`` 入口点自动导入。尚未安装 MJLab 时，导入常量不会失败。
"""

from .constants import TASK_ID


def _register() -> None:
    try:
        from mjlab.tasks.registry import list_tasks, register_mjlab_task
    except ModuleNotFoundError as exc:
        if exc.name == "mjlab" or (exc.name or "").startswith("mjlab."):
            return
        raise
    if TASK_ID in list_tasks():
        return
    from .env_cfg import exolimb_grasp_env_cfg
    from .rl_cfg import exolimb_grasp_ppo_runner_cfg

    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=exolimb_grasp_env_cfg(),
        play_env_cfg=exolimb_grasp_env_cfg(play=True),
        rl_cfg=exolimb_grasp_ppo_runner_cfg(),
    )


_register()

__all__ = ["TASK_ID"]
