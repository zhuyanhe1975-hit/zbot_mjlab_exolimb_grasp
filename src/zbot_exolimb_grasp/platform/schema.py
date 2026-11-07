"""模块化可重构臂的通用数据结构。

平台只生成串联链，不生成树或闭环。模块通过接口族和装配规则连接；
规则里没有的族对，不能装在一起。ZBot R3 是注册进这个 schema 的第一套实物，
不是写死在求值器里的唯一机构。
"""

from __future__ import annotations

from dataclasses import dataclass, field

COUNT_MIN = 3
COUNT_MAX = 8
JOINT_LIMIT = 6.28
PLATFORM_VERSION = "0.1.0"


@dataclass(frozen=True)
class ParameterSpec:
    """连续结构参数。实物模块可以没有参数。"""

    name: str
    unit: str
    lower: float
    upper: float
    default: float
    label: str


@dataclass(frozen=True)
class JointSpec:
    """模块内部的一个转动关节。速度上限不是实测值时必须标明假设。"""

    name: str
    axis: tuple[float, float, float]
    lower: float
    upper: float
    effort_limit_nm: float
    velocity_limit_rad_s: float | None
    velocity_is_assumption: bool


@dataclass(frozen=True)
class ModuleSpec:
    id: str
    label: str
    kind: str  # physical | conceptual
    summary: str
    in_family: str
    out_family: str
    joints: tuple[JointSpec, ...]
    parameters: tuple[ParameterSpec, ...] = ()
    source: str = ""


@dataclass(frozen=True)
class MateRule:
    """父模块输出口与子模块输入口的合法连接。angles_rad 是绕法兰轴的离散方位。"""

    parent_family: str
    child_family: str
    angles_rad: tuple[float, ...]


@dataclass(frozen=True)
class Design:
    """一条串联设计。twists 长度是 len(module_ids) - 1。"""

    module_ids: tuple[str, ...]
    twists_rad: tuple[float, ...]
    parameters: tuple[dict[str, float], ...]
    base_position_m: tuple[float, float, float] = (0.0, 0.0, 0.40)

    def __post_init__(self) -> None:
        object.__setattr__(self, "module_ids", tuple(self.module_ids))
        object.__setattr__(self, "twists_rad", tuple(float(v) for v in self.twists_rad))
        object.__setattr__(
            self,
            "parameters",
            tuple(dict(item) for item in self.parameters),
        )
        object.__setattr__(self, "base_position_m", tuple(float(v) for v in self.base_position_m))


@dataclass
class TaskSpec:
    center_m: tuple[float, float, float] = (0.0, 0.05, 0.72)
    half_m: tuple[float, float, float] = (0.22, 0.22, 0.22)
    payload_kg: float = 0.4
    gravity_m_s2: float = 9.81
    effort_limit_nm: float = 20.0
    effort_is_override: bool = False
    velocity_limit_rad_s: float = 1.5
    sample_count: int = 256
    seed: int = 0
    joint_sample_limit: float = JOINT_LIMIT
    # 采样里自碰撞或静力矩失败的比例超过该值，则整条设计不可行。
    rate_limit: float = 0.85

    def __post_init__(self) -> None:
        self.center_m = tuple(float(v) for v in self.center_m)  # type: ignore[assignment]
        self.half_m = tuple(float(v) for v in self.half_m)  # type: ignore[assignment]


@dataclass
class SearchSpec:
    mode: str = "physical"  # physical | mixed
    count_min: int = 3
    count_max: int = 6
    count_lock: int | None = None
    population: int = 12
    generations: int = 6
    objectives: tuple[str, ...] = ("coverage", "manipulability", "mass")
    length_lower_m: float = 0.06
    length_upper_m: float = 0.16
    seed: int = 1
    task: TaskSpec = field(default_factory=TaskSpec)

    def allowed_module_ids(self) -> tuple[str, ...]:
        if self.mode == "physical":
            return ("zbot_r3",)
        if self.mode == "mixed":
            return ("zbot_r3", "concept_axial")
        raise ValueError(f"未知模式 {self.mode}")


OBJECTIVE_LABELS = {
    "coverage": "任务区覆盖率",
    "manipulability": "平移可操作度中位数",
    "mass": "质量",
    "velocity": "速度能力中位数",
}

OBJECTIVE_SENSE = {
    "coverage": "max",
    "manipulability": "max",
    "mass": "min",
    "velocity": "max",
}
