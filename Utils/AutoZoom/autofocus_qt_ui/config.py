"""
autofocus_qt_ui 默认配置。
"""

from __future__ import annotations

from pathlib import Path

# 输出目录
DEFAULT_OUTPUT_DIR: str = "focus_output"

# 运行模式："screen" | "window" | "usb" | "sim"
DEFAULT_CAPTURE_MODE: str = "sim"

# 默认窗口标题（window 模式）
DEFAULT_WINDOW_TITLE: str = "NIS"

# 默认 USB 设备
DEFAULT_USB_DEVICE: int = 0

# 默认截图区域 (left, top, width, height)
DEFAULT_CAPTURE_AREA: tuple[int, int, int, int] = (116, 98, 1112, 886)

# 默认 ROI (x, y, w, h)
DEFAULT_FOCUS_ROI: tuple[int, int, int, int] = (0, 0, 300, 300)

# 默认循环参数
DEFAULT_CYCLES: int = 30
DEFAULT_INTERVAL_S: float = 1.5

# 默认 Z 轴参数
DEFAULT_Z_ENABLED: bool = True
DEFAULT_Z_AXIS: int = 1
DEFAULT_Z_SPEED: int = 100
DEFAULT_Z_ACCEL: int = 100

# 默认补焦触发阈值
DEFAULT_TRIGGER_RATIO: float = 0.90
DEFAULT_STOP_RATIO: float = 0.95
DEFAULT_TRIGGER_COUNT: int = 3

# 默认搜索策略
DEFAULT_SEARCH_STRATEGY: str = "hill_climb"

# 默认本地视频文件路径
DEFAULT_LOCAL_VIDEO_PATH: str = ""
