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

    local_video_path: str = ""
    """本地视频文件路径（local_video_file 模式使用）"""

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

    autofocus_focus_trigger_ratio: float = 0.95
    """FocusScore_ratio 触发阈值（对称区间下限）。
    以 1.0 为中心，允许区间为 [ratio, 2 - ratio]。
    默认 0.95 表示允许区间 [0.95, 1.05]，超出即计数。"""

    autofocus_trigger_absolute: bool = True
    """触发阈值是否使用以 1.0 为中心的对称区间（绝对值模式）。
    True  ：[trigger_ratio, 2 - trigger_ratio]，例如 [0.95, 1.05]。
    False ：仅单边下限，score < trigger_ratio 时触发。"""

    autofocus_detection_only: bool = False
    """FocusScore 检测模式。
    True 时触发补焦条件后只记录当前图像和分数，不执行 Z 轴闭环搜索等硬件操作。"""

    autofocus_focus_trigger_count: int = 3
    """连续 N 轮 FocusScore_ratio 超出允许区间则启动补焦"""

    autofocus_stop_ratio: float = 0.95
    """闭环补焦的目标 FocusScore_ratio；搜索策略内部以此为达标线；
    建议与 autofocus_focus_trigger_ratio 保持一致"""

    autofocus_passive_mode: bool = False
    """被动补焦模式：触发后连续搜索，直到分数回到允许区间内（或达到最大尝试次数）"""

    autofocus_passive_max_attempts: int = 10
    """被动补焦单轮最大连续尝试次数"""

    autofocus_passive_consecutive_good: int = 3
    """被动补焦模式下，FocusScore 连续多少轮回到允许区间内后自动停止循环"""

    autofocus_passive_disable_auto_stop: bool = False
    """被动补焦模式下是否关闭“连续达标次数”自动停止逻辑。
    True 时被动补焦将持续运行，只有手动点击停止才会结束。"""

    # -------------------- SHG 补焦触发阈值（可选） --------------------
    autofocus_shg_trigger_ratio: float = 0.90
    autofocus_shg_trigger_count: int = 2
    autofocus_shg_hard_ratio: float = 0.85

    # -------------------- Z 轴 (Newport 8742 Picomotor) --------------------
    z_enabled: bool = True
    z_axis: int = 1
    z_speed: int = 100
    z_accel: int = 100

    # Picomotor 控制器连接参数（解决 conn=0 硬编码导致的连接失败）
    z_picomotor_conn: int = 1
    #z_picomotor_conn: str = r"8742-100100"
    """控制器索引（对应 Newport.Picomotor8742 的 conn 参数）"""

    z_picomotor_backend: str = "auto"
    #z_picomotor_backend: str = "network"
    """连接后端：auto / pyusb / serial"""

    z_picomotor_timeout: float = 5.0
    """连接/通信超时（秒）"""

    z_picomotor_multiaddr: bool = False
    """是否启用多地址模式"""

    z_picomotor_scan: bool = True
    """是否扫描可用轴"""

    z_picomotor_velocity: Optional[int] = None
    """电机速度（None 时使用 z_speed）"""

    z_picomotor_acceleration: Optional[int] = None
    """电机加速度（None 时使用 z_accel）"""

    z_probe_steps: int = 10
    """试探步数"""

    z_direction_probe_steps: tuple[int, ...] = (10, 20, 30)
    """方向判断的双向采样步长序列。"""

    z_direction_probe_samples: int = 3
    """每个采样点的重复采样次数。"""

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

    z_local_refine_enabled: bool = True
    """爬坡粗搜结束后，是否回到 best_pos 左右做局部细搜。"""

    z_local_refine_decay: float = 0.5
    """局部细搜步长衰减系数，例如 0.5 表示 10 -> 5 -> 2 -> 1。"""

    z_local_refine_min_step: int = 1
    """局部细搜最小步长。"""

    z_local_refine_max_rounds: int = 4
    """局部细搜最大轮数。每轮会在 best_pos 左右各测一次。"""

    # -------------------- 参考建立 --------------------
    focus_reference_capture_count: int = 5
    """建立聚焦参考时采集次数"""
