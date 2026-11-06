"""让 scripts/ 在未安装包时也能导入任务，并在无显示环境下选择软件渲染。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# 在导入 MJLab 之前指定。MJLab 在 Linux 上会默认改成 egl；
# 没有 EGL 的 CPU 机器用 glfw 才能把 MuJoCo 导入进来。需要 EGL 时自行导出 MUJOCO_GL=egl。
os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("MJLAB_WARP_QUIET", "1")
