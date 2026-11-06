"""在本地训练 Mjlab-Zbot-Exolimb-Grasp。

浏览器里的页面只做运动学演示，训练请在这台机器上运行。没有 GPU 时自动使用 CPU。
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401

import torch

from zbot_exolimb_grasp.constants import TASK_ID


def main() -> None:
    parser = argparse.ArgumentParser(description="训练固定六模块 Z-Bot 外肢体抓取任务。")
    parser.add_argument("--num-envs", type=int, default=None, help="并行环境数。CPU 上建议 4 到 16。")
    parser.add_argument("--max-iterations", type=int, default=None, help="PPO 迭代次数。试跑可先用 20。")
    parser.add_argument("--seed", type=int, default=None, help="随机种子。")
    parser.add_argument("--cpu", action="store_true", help="即使检测到 GPU 也使用 CPU。")
    args = parser.parse_args()

    use_cpu = args.cpu or not torch.cuda.is_available()
    if use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device_text = "CPU"
    else:
        device_text = "GPU"

    import zbot_exolimb_grasp  # noqa: F401
    from mjlab.scripts.train import TrainConfig, launch_training

    cfg = TrainConfig.from_task(TASK_ID)
    if args.num_envs is not None:
        cfg.env.scene.num_envs = args.num_envs
    if args.max_iterations is not None:
        cfg.agent.max_iterations = args.max_iterations
    if args.seed is not None:
        cfg.agent.seed = args.seed
        cfg.env.seed = args.seed
    if use_cpu:
        cfg = replace(cfg, gpu_ids=None)

    print(f"任务：{TASK_ID}")
    print(f"设备：{device_text}，并行环境 {cfg.env.scene.num_envs}，迭代 {cfg.agent.max_iterations}")
    print("日志写在 logs/rsl_rl/zbot_exolimb_grasp/。")
    launch_training(TASK_ID, cfg)


if __name__ == "__main__":
    main()
