"""模块化可重构臂综合设计与参数优化。

通用对象是模块、接口族、装配规则和串联设计。ZBot R3 是第一套实物模块，
概念模块用来证明库可以扩展，不代表存在第二套已测量硬件。
"""

from .schema import PLATFORM_VERSION

__all__ = ["PLATFORM_VERSION"]
