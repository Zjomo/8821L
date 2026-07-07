"""
自动对焦/聚焦指标 配置参数。
从 measurement_autofocus_shg_closed_loop.py 的 MeasurementConfig 中提取聚焦相关字段。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class AutofocusConfig:
    """
    聚焦指标计算 + 自动补焦全部配置参数。
    """

    capture_mode: str = "screen_region"
    """采集模式：screen_region / window_roi / usb_camera"""

    window_title: str = ""
    """窗口标题关键字（window_roi 模式使用）"""

    window_match_mode: str = "contains"
    """窗口匹配方式：contains / exact"""

    window_padding: Tuple[int, int, int, int] = (0, 0, 0, 0)
    """窗口截图边缘补偿 (left, top, right, bottom)"""

    capture_area: Tuple[int, int, int, int] = (116, 98, 1112, 886)
    """屏幕截图区域 (left, top, width, height)"""

    focus_roi: Tuple[int, int, int, int] = (0, 0, 300, 300)
    """相对于截图左上角的 ROI (x, y, w, h)"""

    # USB 相机配置（usb_camera 模式使用）
    usb_device_index: int = 0
    """USB 相机设备索引"""

    usb_resolution: str = "AUTO"
    """USB 相机分辨率：PAL / NTSC / AUTO"""

    usb_pixel_format: Optional[str] = None
    """USB 相机像素格式：AUTO / MJPG / YUY2 / YUYV / UYVY"""

    # -------------------- 聚焦指标计算参数 --------------------
    focus_fft_low_radius_ratio: float = 0.10
    """FFT 高频能量计算时，低频圆盘半径 = ratio * min(h, w)"""

    focus_edge_profile_half_width: int = 60
    """边缘轮廓分析的半宽度 (px)"""

    # -------------------- FocusScore 权重 --------------------
    focus_weight_highfreq: float = 0.35
    focus_weight_tenengrad: float = 0.25
    focus_weight_brenner: float = 0.25
    focus_weight_red_blue: float = 0.15

    # 新增常用聚焦指标权重（默认 0，手动启用）
    focus_weight_modified_laplacian: float = 0.0
    focus_weight_dct_energy: float = 0.0
    focus_weight_smd: float = 0.0
    focus_weight_entropy: float = 0.0

    # -------------------- 自动补焦触发阈值 --------------------
    autofocus_enabled: bool = True
    """是否启用自动补焦"""

    autofocus_focus_trigger_ratio: float = 0.90
    """FocusScore_ratio 低于此值开始计数"""

    autofocus_focus_trigger_count: int = 3
    """连续 N 轮低于触发值则启动补焦"""

    autofocus_stop_ratio: float = 0.95
    """闭环补焦的目标 FocusScore_ratio，达到即停止"""

    # -------------------- SHG 补焦触发阈值（可选） --------------------
    autofocus_shg_trigger_ratio: float = 0.90
    autofocus_shg_trigger_count: int = 2
    autofocus_shg_hard_ratio: float = 0.85

    # -------------------- Z 轴 (Newport 8742 Picomotor) --------------------
    z_enabled: bool = True
    z_axis: int = 1
    z_speed: int = 100
    z_accel: int = 100
    z_probe_steps: int = 10
    """试探步数"""

    z_search_steps: int = 10
    """搜索步数"""

    z_settle_time_s: float = 0.20
    """移动后稳定等待时间 (秒)"""

    z_min_improve_ratio: float = 0.005
    """最小提升比例：新分数 > 最佳分数 * (1 + min_improve) 才算进步"""

    z_max_iter: int = 40
    """最大搜索迭代次数"""

    z_max_total_steps: int = 400
    """最大总移动步数绝对值"""

    z_patience: int = 3
    """连续无进步的耐心轮数"""

    # -------------------- 搜索策略（新增） --------------------
    z_search_strategy: str = "hill_climb"
    """搜索策略：hill_climb / full_sweep / golden_section / curve_fit"""

    z_sweep_range_steps: int = 100
    """全扫/拟合策略的半范围（步数）"""

    z_curve_fit_points: int = 7
    """曲线拟合策略的采样点数（奇数 >= 5）"""

    z_golden_section_tol: int = 3
    """黄金分割搜索收敛容差（步数）"""

    z_adaptive_step_decay: float = 1.0
    """爬坡策略中步长衰减系数（1.0 表示不衰减；<1.0 时每次无进步会缩小步长）"""

    # -------------------- 参考建立 --------------------
    focus_reference_capture_count: int = 5
    """建立聚焦参考时采集次数"""