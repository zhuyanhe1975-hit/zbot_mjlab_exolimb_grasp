"""抓取任务里与网页演示共用的几何常量。

名义关节角来自 zbot_rl_mjlab_R3 `robot.py` 的 ``DEFAULT_JOINT_POS``。
工具球放在该姿态下 ``foot_1`` 足端外侧，使末端原点距离小于 0.08 m，
并且 ``b6`` 网格与球接触。数值由 MuJoCo 正运动学核对。
"""

from __future__ import annotations

TASK_ID = "Mjlab-Zbot-Exolimb-Grasp"

# 权威模型：zhuyanhe1975-hit/zbot_rl_mjlab_R3 @ 0c0b686c69f8f78e79bf9f875dead868f25e1cdc
SOURCE_COMMIT = "0c0b686c69f8f78e79bf9f875dead868f25e1cdc"

SHOULDER_POS = (0.0, 0.0, 0.40)
JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
STRAIGHT_JOINT_POS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
NOMINAL_JOINT_POS = (0.312, 0.837, -2.02, 2.02, -0.837, -0.312)

# a 连杆 0.54171 kg × 6 + b 连杆 0.17138 kg × 6
LINK_A_MASS = 0.54171
LINK_B_MASS = 0.17138
TOTAL_MASS = 4.27854

JOINT_DAMPING = 0.05
ARMATURE = 0.001
JOINT_RANGE = (-6.28, 6.28)
PD_STIFFNESS = 15.0
PD_DAMPING = 1.5
EFFORT_LIMIT = 20.0

GRASP_DISTANCE = 0.08
TOOL_RADIUS = 0.022
TOOL_MASS = 0.05
# 名义姿态下 foot_1 原点到球心约 0.075 m，且 b6 与球面接触。
TOOL_POS = (-0.0003285042625797094, 0.12478776399807348, 0.3800370725014494)
TRAY_HALF = (0.18, 0.14, 0.008)
TRAY_POS = (
    TOOL_POS[0],
    TOOL_POS[1],
    TOOL_POS[2] - TOOL_RADIUS - TRAY_HALF[2],
)
