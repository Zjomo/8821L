"""obstacle_avoidance: 单球避障导航与多球聚拢（自包含项目，默认 dryrun 仿真）。

坐标契约（见 OBSTACLE_AVOID_PLAN.md 第 1 节）：
- 像素坐标: (x, y)，x 向右, y 向下，原点为帧左上角。
- 物理坐标: 毫米 (mm)，方向与像素一致，比例 px_per_mm。
- 位移台步数: dryrun 适配层直接以 mm 位移执行（真实驱动可替换 XYStageProtocol）。
"""

__version__ = "0.1.0"
