"""回放 Mjlab-Zbot-Exolimb-Grasp。

没有检查点时默认使用零动作，方便先看场景。训练完成后把检查点传给 --checkpoint。
无图形界面时会打开 Viser；本机有显示器时使用 MuJoCo 窗口。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401

import torch

from zbot_exolimb_grasp.constants import TASK_ID


def main() -> None:
    parser = argparse.ArgumentParser(description="回放固定六模块 Z-Bot 外肢体抓取任务。")
    parser.add_argument("--checkpoint", type=str, default=None, help="model_*.pt 检查点路径。")
    parser.add_argument(
        "--agent",
        choices=("trained", "zero", "random"),
        default="zero",
        help="trained 需要检查点；zero 保持伸直附近；random 随机动作。",
    )
    parser.add_argument("--viewer", choices=("auto", "native", "viser"), default="auto")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU。")
    args = parser.parse_args()

    use_cpu = args.cpu or not torch.cuda.is_available()
    if use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import zbot_exolimb_grasp  # noqa: F401
    from mjlab.scripts.play import PlayConfig, run_play

    if args.agent == "trained" and not args.checkpoint:
        raise SystemExit("回放训练策略时请传入 --checkpoint logs/rsl_rl/zbot_exolimb_grasp/.../model_*.pt")

    print(f"任务：{TASK_ID}")
    print(f"策略：{args.agent}，查看器：{args.viewer}，设备：{'CPU' if use_cpu else 'GPU'}")
    run_play(
        TASK_ID,
        PlayConfig(
            agent=args.agent,
            checkpoint_file=args.checkpoint,
            viewer=args.viewer,
            num_envs=args.num_envs,
            device="cpu" if use_cpu else None,
        ),
    )


if __name__ == "__main__":
    main()
