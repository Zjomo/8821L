from __future__ import annotations

import sys
import os
import csv
import json
import time
import math
import threading
import traceback
import copy
import re
import pyautogui
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog

import numpy as np
import cv2
import matplotlib.pyplot as plt

try:
    from pylablib.devices import Newport, Thorlabs
except Exception as _pylablib_import_error:
    Newport = None  # type: ignore
    Thorlabs = None  # type: ignore

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


try:
    from openpyxl import Workbook, load_workbook
    from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
except ImportError as e:
    raise ImportError("需要先安装 openpyxl 才能生成 xlsx 汇总文件：pip install openpyxl") from e

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 黑体
plt.rcParams["axes.unicode_minus"] = False   # 正常显示负号

# ============================================================
# 路径与外部模块导入
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# YOLO-OBB angle detector: fixed region screenshot -> YOLO OBB -> long-edge angle.
YOLO_CONFIG_DIR = PROJECT_ROOT / ".ultralytics"
try:
    YOLO_CONFIG_DIR.mkdir(exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))
except Exception:
    pass
try:
    import mss
except Exception:
    mss = None  # type: ignore
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None  # type: ignore

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SAM2_REPO_ROOT = PROJECT_ROOT / "sam2-main"
if SAM2_REPO_ROOT.exists() and str(SAM2_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM2_REPO_ROOT))

# ============================================================
# GUI 默认参数配置
# ============================================================
# 优先读取同目录 config.py；如果不存在，则兼容旧的
# config_conditional_second_detect.py；都不存在时使用代码内置默认值。
from config_angle_repair_fixed import DEFAULT_CONFIG  # GUI 左侧所有输入项的默认值

# 本版本要求 real/virtual 共用同一套 Step1/Step7 YOLO-OBB 检测节奏。
# 这里强制覆盖旧配置文件中可能残留的慢检测/每帧保存 overlay 设置，
# 避免只改代码默认值但被 config_angle_repair_fixed.py 中的旧值覆盖。
DEFAULT_CONFIG["rule_ab_angle_watch_interval_s"] = 0.5
DEFAULT_CONFIG["rule_ab_realtime_save_every_angle_frame"] = False

# 新增：光谱仪设备选择配置（不影响原有代码环境）
DEFAULT_CONFIG["spectrometer_backend"] = "labview_tcp"  # 选项: labview_tcp, picam, picam_demo
DEFAULT_CONFIG["picam_dll_path"] = None
DEFAULT_CONFIG["picam_camera_index"] = 0
DEFAULT_CONFIG["picam_exposure"] = 0.1
DEFAULT_CONFIG["picam_temperature"] = -25.0
DEFAULT_CONFIG["picam_roi_x"] = 0
DEFAULT_CONFIG["picam_roi_y"] = 0
DEFAULT_CONFIG["picam_roi_width"] = 1024
DEFAULT_CONFIG["picam_roi_height"] = 256
# 新增：IsoPlane 单色仪配置
DEFAULT_CONFIG["isoplane_dll_path"] = None
DEFAULT_CONFIG["isoplane_device_index"] = 0
DEFAULT_CONFIG["center_wavelength_nm"] = 550.0
DEFAULT_CONFIG["grating_index"] = 1
DEFAULT_CONFIG["entrance_slit_um"] = 50
DEFAULT_CONFIG["exit_slit_um"] = 50

def _cfg(name: str, default: Any) -> Any:
    return DEFAULT_CONFIG.get(name, default)

# ============================================================
# 光谱仪/TCP 配置
# ============================================================
# 当前版本恢复真实 LabVIEW TCP 光谱仪流程：
#   1) 启动/等待/关闭 LabVIEW TCP Server；
#   2) 完整测量中通过 TCP 请求真实光谱数据；
#   3) 保存真实 raw / filtered / median / fit 峰值。
# 若以后需要临时脱机测试，可手动改成 True。
SPECTROMETER_TCP_DISABLED = False
DUMMY_SPECTRUM_PEAK = 1000.0

# 新增：PI 光谱仪直接控制适配器（设备可选项）
# 将 PrincetonInstruments/Project 加入路径，确保能导入 patches/pi_spectrometer_adapter
_PI_PROJECT_DIR = Path(__file__).resolve().parent / "Utils" / "PrincetonInstruments" / "Project"
if str(_PI_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PI_PROJECT_DIR))
try:
    from patches.pi_spectrometer_adapter import PISpectrometerAdapter
except Exception as _pi_adapter_import_error:
    PISpectrometerAdapter = None  # type: ignore

from control.illumination_relay import IlluminationRelay
from control.signal_generator_rigol import RigolDG4062Controller
try:
    from control.labview_tcp_server import LabVIEWTCPServer
except Exception as _labview_tcp_import_error:
    LabVIEWTCPServer = None  # type: ignore

from vision.screen_capture import CaptureArea
try:
    from vision.screen_capture import FixedRegionScreenCapture
except Exception:
    FixedRegionScreenCapture = None

try:
    # 推荐使用 strict_c 版本：完整测量显式指定 C 后，不允许外部模块回退旧 C。
    from logic.actual_nano_boundary_following_sam2VideoAB_xyStep_detailed_pathfix_strict_c import RuntimeConfig as RuleABRuntimeConfig
    from logic.actual_nano_boundary_following_sam2VideoAB_xyStep_detailed_pathfix_strict_c import ActualNanoBoundaryFollower
except Exception:
    # 兼容旧文件名。若完整测量仍加载旧 C，请把 strict_c 文件复制到 logic 目录。
    from logic.actual_nano_boundary_following_sam2VideoAB_xyStep_detailed_pathfix import RuntimeConfig as RuleABRuntimeConfig
    from logic.actual_nano_boundary_following_sam2VideoAB_xyStep_detailed_pathfix import ActualNanoBoundaryFollower
from logic.rule_ac_fixed import RuleACConfig, RuleACOverlapController

from Focus.config import AutofocusConfig
from Focus.controller import AutofocusController, focus_score_ratio_in_tolerance
from Focus.metrics import FocusMetricsCalculator
from Focus.scorer import FocusScorer
from Focus.simulator import create_demo_environment
from Focus.z_axis import ZAxisController
from spectrum_autofocus_loop import SpectrumAutofocusLoop
from log_manager import LogManager


@dataclass
class MeasurementConfig:
    # 硬件模式："real" 连接 Newport / Thorlabs / LabVIEW；"virtual" 不连接任何真实硬件。
    # virtual 模式保留完整测量主逻辑、标定、Step1/Step7/Step9、角度检测与保存字段；
    # 只是把硬件动作替换为日志成功返回，把光谱仪替换为虚拟光谱或回放光谱。
    hardware_mode: str = str(_cfg("hardware_mode", "real"))
    virtual_spectrum_mode: str = str(_cfg("virtual_spectrum_mode", "gaussian"))  # gaussian / replay_csv
    virtual_spectrum_replay_csv: str = str(_cfg("virtual_spectrum_replay_csv", ""))
    virtual_spectrum_points: int = int(_cfg("virtual_spectrum_points", 1024))
    virtual_spectrum_center_nm: float = float(_cfg("virtual_spectrum_center_nm", 516.0))
    virtual_spectrum_sigma: float = float(_cfg("virtual_spectrum_sigma", 18.0))
    virtual_spectrum_noise: float = float(_cfg("virtual_spectrum_noise", 15.0))

    max_cycles: int = int(_cfg("max_cycles", 1))

    # 新循环结构：
    # max_cycles固定为1，表示只执行一次主循环框架
    # sub_loop_iterations_per_cycle控制实际的循环次数（理解为主循环次数）
    # 序号分配：0=初始光谱采集，1-N=实际主循环（原子循环）
    sub_loop_iterations_per_cycle: int = int(_cfg("sub_loop_iterations_per_cycle", 1))

    # 这个不是照明光时间，而是信号发生器 CH1 ON 的持续时间
    signal_on_time_ms: float = float(_cfg("signal_on_time_ms", 50.0))

    # 照明光打开后等待时间
    stable_wait_ms: int = int(_cfg("stable_wait_ms", 1200))

    # Newport 8743-CL 激光开关控制参数
    # 说明：单轮流程中的 Step 6 / Step 8 不再使用 Rigol CH1，
    # 而是用 Newport 轴运动实现激光打开/关闭。
    laser_axis: int = int(_cfg("laser_axis", 2))
    laser_on_steps: int = int(_cfg("laser_on_steps", 400))
    laser_off_steps: int = int(_cfg("laser_off_steps", 400))
    laser_speed: int = int(_cfg("laser_speed", 5000))
    laser_accel: int = int(_cfg("laser_accel", 5000))

    # angle_after - angle_before 的目标范围
    angle_delta_min_deg: float = float(_cfg("angle_delta_min_deg", 1.0))
    angle_delta_max_deg: float = float(_cfg("angle_delta_max_deg", 4.0))

    # 开光时间调节倍乘系数
    signal_time_factor: float = float(_cfg("signal_time_factor", 1.5))

    save_root: str = str(_cfg("save_root", "measurement_output"))

    light_port: str = str(_cfg("light_port", "COM20"))

    rigol_visa: str = str(_cfg("rigol_visa", "USB0::0x1AB1::0x0641::DG4E192200870::INSTR"))
    rigol_timeout_ms: int = int(_cfg("rigol_timeout_ms", 3000))
    rigol_command_delay_s: float = float(_cfg("rigol_command_delay_s", 0.05))

    ch1_low_v: float = float(_cfg("ch1_low_v", 0.0))
    ch1_high_v: float = float(_cfg("ch1_high_v", 4.0))
    ch1_freq_hz: float = float(_cfg("ch1_freq_hz", 6000.0))
    ch1_duty_percent: float = float(_cfg("ch1_duty_percent", 3.0))
    ch1_delay_s: float = float(_cfg("ch1_delay_s", 0.0))

    ch2_low_v: float = float(_cfg("ch2_low_v", 0.0))
    ch2_high_v: float = float(_cfg("ch2_high_v", 0.002))
    ch2_freq_hz: float = float(_cfg("ch2_freq_hz", 6000.0))
    ch2_duty_percent: float = float(_cfg("ch2_duty_percent", 3.0))
    ch2_delay_s: float = float(_cfg("ch2_delay_s", 0.0))

    # 角度检测：使用 YOLO-OBB，方式与第二个代码一致：固定截图区域 -> OBB 四角 -> 最长边角度。
    angle_model_path: str = str(_cfg("angle_model_path", r".\vision\best_wan12.2.pt"))
    capture_area: Tuple[int, int, int, int] = tuple(_cfg("capture_area", (116, 100, 1112, 850)))  # type: ignore
    angle_output_dir: str = str(_cfg("angle_output_dir", "outputs/captured_frames"))
    angle_conf: float = float(_cfg("angle_conf", 0.5))
    angle_iou: float = float(_cfg("angle_iou", 0.7))
    angle_imgsz: int = int(_cfg("angle_imgsz", 640))
    angle_device: str = str(_cfg("angle_device", ""))
    # 旧字段保留兼容 GUI/旧配置；当前 YOLO-OBB 角度检测不再使用 num/cw。
    angle_num: int = int(_cfg("angle_num", 0))
    angle_cw: int = int(_cfg("angle_cw", 0))

    # 条件二次检测：第一次角度落在这些范围内时，临时启用 use_second_detect 再检测一次。
    # 这不是原来的全局“启用二次检测”复选框，而是自动触发的保护逻辑。
    enable_conditional_second_detect: bool = bool(_cfg("enable_conditional_second_detect", True))
    conditional_second_detect_ranges: Tuple[Tuple[float, float], ...] = tuple(
        tuple(x) for x in _cfg("conditional_second_detect_ranges", ((0.0, 20.0), (90.0, 110.0)))
    )  # type: ignore

    tcp_host: str = str(_cfg("tcp_host", "127.0.0.1"))
    tcp_port: int = int(_cfg("tcp_port", 65432))
    tcp_output_dir: str = str(_cfg("tcp_output_dir", "labview_csv_output"))
    tcp_command: str = str(_cfg("tcp_command", "MEASURE"))

    # 新增：光谱仪设备选择（不影响原有代码环境）
    spectrometer_backend: str = str(_cfg("spectrometer_backend", "labview_tcp"))  # labview_tcp / picam / picam_demo
    picam_dll_path: Optional[str] = _cfg("picam_dll_path", None)
    picam_camera_index: int = int(_cfg("picam_camera_index", 0))
    picam_exposure: float = float(_cfg("picam_exposure", 0.1))
    picam_temperature: float = float(_cfg("picam_temperature", -25.0))
    picam_roi_x: int = int(_cfg("picam_roi_x", 0))
    picam_roi_y: int = int(_cfg("picam_roi_y", 0))
    picam_roi_width: int = int(_cfg("picam_roi_width", 1024))
    picam_roi_height: int = int(_cfg("picam_roi_height", 256))
    # IsoPlane 单色仪参数
    isoplane_dll_path: Optional[str] = _cfg("isoplane_dll_path", None)
    isoplane_device_index: int = int(_cfg("isoplane_device_index", 0))
    center_wavelength_nm: float = float(_cfg("center_wavelength_nm", 550.0))
    grating_index: int = int(_cfg("grating_index", 1))
    entrance_slit_um: int = int(_cfg("entrance_slit_um", 50))
    exit_slit_um: int = int(_cfg("exit_slit_um", 50))

    # TCP 原始数据处理参数
    # 第一步：删除高于该阈值的原始数据点；第二步：对剩余数据做 5 点中值滤波
    raw_remove_above: float = float(_cfg("raw_remove_above", 3000.0))
    median_filter_window: int = int(_cfg("median_filter_window", 5))

    x_axis_xlsx_path: str = str(_cfg("x_axis_xlsx_path", "516中心波长.xlsx"))

    enable_delta_w_judge: bool = bool(_cfg("enable_delta_w_judge", False))
    delta_w_threshold: float = float(_cfg("delta_w_threshold", 0.0))
    stop_when_delta_w_not_enough: bool = bool(_cfg("stop_when_delta_w_not_enough", False))

    # A 推动 B：调用 logic/rule_ab.py，每一步后检测 B 对边角度。
    rule_ab_enable_stage: bool = bool(_cfg("rule_ab_enable_stage", True))
    rule_ab_max_steps: int = int(_cfg("rule_ab_max_steps", 15))
    rule_ab_angle_delta_min_deg: float = float(_cfg("rule_ab_angle_delta_min_deg", 3.5))
    rule_ab_angle_delta_max_deg: float = float(_cfg("rule_ab_angle_delta_max_deg", 6.0))

    # 当前版本：Step1/Step7 不做任何角度修复，不做候选边替换，不使用旧版 6° 过冲修复判据。
    # Step1 baseline = YOLO-OBB 当前帧长边角度；Step7 final = YOLO-OBB 当前帧长边角度。
    # Step7 只判断 abs(current_angle - Step1 baseline) >= rule_ab_angle_delta_min_deg。

    # A/B 分割全局重试：
    # 只要完整测量中需要重新分割/跟踪 A、B，遇到 A/B temporal 候选质量不足、
    # infer_abc 失败、capture_and_build_scene 失败，都不应直接终止完整测量；
    # 而是等待下一帧后重试。
    rule_ab_ab_seg_retry_attempts: int = int(_cfg("rule_ab_ab_seg_retry_attempts", 3))
    rule_ab_ab_seg_retry_interval_s: float = float(_cfg("rule_ab_ab_seg_retry_interval_s", 0.25))
    rule_ab_ab_seg_reset_bad_scene_cache: bool = bool(_cfg("rule_ab_ab_seg_reset_bad_scene_cache", True))

    # Step1 / 完整测量启动阶段的截图/首帧初始化容错。
    # 遇到 PIL/ImageGrab/mss/显卡后端偶发的半帧读取错误，例如：
    #   backend exception: 'read returned less data than expected'
    # 不立即终止完整测量，而是释放本次 RuleAB follower 后等待下一帧重试。
    rule_ab_startup_init_retry_attempts: int = int(_cfg("rule_ab_startup_init_retry_attempts", 5))
    rule_ab_startup_init_retry_interval_s: float = float(_cfg("rule_ab_startup_init_retry_interval_s", 0.8))
    rule_ab_frame_read_retry_attempts: int = int(_cfg("rule_ab_frame_read_retry_attempts", 5))
    rule_ab_frame_read_retry_interval_s: float = float(_cfg("rule_ab_frame_read_retry_interval_s", 0.15))

    rule_ab_loop_interval_s: float = float(_cfg("rule_ab_loop_interval_s", 0.15))
    rule_ab_stage_conn: str = str(_cfg("rule_ab_stage_conn", "97101208"))
    rule_ab_stage_x_channel: int = int(_cfg("rule_ab_stage_x_channel", 3))
    rule_ab_stage_y_channel: int = int(_cfg("rule_ab_stage_y_channel", 4))


    rule_ab_ab_close_threshold_px: float = float(_cfg("rule_ab_ab_close_threshold_px", 35.0))  # 旧字段，保留兼容，不再用于单独测试
    rule_ab_ab_overlap_min_area_px: float = float(_cfg("rule_ab_ab_overlap_min_area_px", 1.0))
    rule_ab_ac_target_clearance_px: float = float(_cfg("rule_ab_ac_target_clearance_px", 90.0))
    rule_ab_ac_min_clearance_px: float = float(_cfg("rule_ab_ac_min_clearance_px", 80.0))
    rule_ab_ac_max_clearance_px: float = float(_cfg("rule_ab_ac_max_clearance_px", 100.0))
    # GUI 中选择 C 路线方向后同步到这里：+1=屏幕视觉逆时针，-1=屏幕视觉顺时针。
    rule_ab_follow_c_direction: int = int(_cfg("rule_ab_follow_c_direction", 1))
    rule_ab_ac_correction_mode: str = str(_cfg("rule_ab_ac_correction_mode", "nearest_normal"))

    # Step7 新方案：不再由实时 RuleAB 策略决定 A 的运动方向，而是在 C 标定时
    # 根据 C 的近似四边形四条边预生成 A 的运动路线；Step7 每一帧都让 A 追踪这条路线。
    rule_ab_use_c_edge_route: bool = bool(_cfg("rule_ab_use_c_edge_route", True))
    # Step7 路线来源：
    #   "quad"：默认使用 C 近似四边形四条边生成路线；
    #   "sam2_outer_edge"：可选，使用 SAM2 原始 C mask 最大外轮廓像素生成路线。
    rule_ab_c_edge_route_source: str = str(_cfg("rule_ab_c_edge_route_source", "quad"))
    rule_ab_c_edge_route_spacing_px: float = float(_cfg("rule_ab_c_edge_route_spacing_px", 12.0))
    # 路线本身默认就是近似四边形边界，因此默认不外扩。
    # 安全距离由 A 中心到路线的实时距离判断实现。
    rule_ab_c_edge_route_safe_clearance_px: float = float(_cfg("rule_ab_c_edge_route_safe_clearance_px", 0.0))

    # Step7 路线安全带：A 中心到“C 近似四边形路线”的距离必须保持在 [min, max]。
    # 默认直接复用 A-C 的安全距离参数，即 A 中心到 C 近似四边形边界保持在 min/target/max 范围。
    # 若距离 < min：先远离路线，回到安全带中间；
    # 若距离 > max：先靠近路线，回到安全带中间；
    # 只有在安全带内，才沿 C 近似四边形路线切向前进。
    rule_ab_route_min_distance_px: float = float(_cfg("rule_ab_route_min_distance_px", _cfg("rule_ab_ac_min_clearance_px", 80.0)))
    rule_ab_route_max_distance_px: float = float(_cfg("rule_ab_route_max_distance_px", _cfg("rule_ab_ac_max_clearance_px", 100.0)))
    rule_ab_route_safe_target_distance_px: float = float(_cfg("rule_ab_route_safe_target_distance_px", _cfg("rule_ab_ac_target_clearance_px", 90.0)))
    rule_ab_route_target_tolerance_px: float = float(_cfg("rule_ab_route_target_tolerance_px", 8.0))
    rule_ab_route_lookahead_points: int = int(_cfg("rule_ab_route_lookahead_points", 1))
    rule_ab_route_loop: bool = bool(_cfg("rule_ab_route_loop", True))
    # Step7 固定路线索引推进模式：只有超过硬安全余量时才允许纯 SAFE 法向修正；
    # 轻微越界时切向推进仍保留，用切向 + 法向合成，避免局部 UP/DOWN 振荡。
    rule_ab_route_hard_safety_margin_px: float = float(_cfg("rule_ab_route_hard_safety_margin_px", 15.0))

    # A/B mask 运行时后处理：当前版本不再约束 B，也不再回退上一帧 B。
    # 当前帧检测结果会被用于 Step7 路线运动和角度检测；不再执行角度修复。
    # 这里只保留 A = A - dilate(B_current)，避免 A mask 把当前帧 B 包进去。
    rule_ab_enable_ab_mask_guard: bool = bool(_cfg("rule_ab_enable_ab_mask_guard", True))
    rule_ab_b_area_ratio_min: float = float(_cfg("rule_ab_b_area_ratio_min", 0.75))
    rule_ab_b_area_ratio_max: float = float(_cfg("rule_ab_b_area_ratio_max", 1.35))
    rule_ab_b_center_jump_max_px: float = float(_cfg("rule_ab_b_center_jump_max_px", 40.0))
    rule_ab_b_angle_jump_max_deg: float = float(_cfg("rule_ab_b_angle_jump_max_deg", 12.0))
    rule_ab_a_exclude_b_dilate_px: int = int(_cfg("rule_ab_a_exclude_b_dilate_px", 1))
    rule_ab_a_min_area_after_b_exclude_px: int = int(_cfg("rule_ab_a_min_area_after_b_exclude_px", 5))

    # RuleAB：第一帧 SAM2 分割确认与 C 四边形拟合
    rule_ab_confirm_first_frame_segmentation: bool = bool(_cfg("rule_ab_confirm_first_frame_segmentation", True))
    rule_ab_confirm_window_scale: float = float(_cfg("rule_ab_confirm_window_scale", 0.85))
    rule_ab_c_contour_mode: str = str(_cfg("rule_ab_c_contour_mode", "quadrilateral"))
    rule_ab_c_quadrilateral_method: str = str(_cfg("rule_ab_c_quadrilateral_method", "body_edge"))

    # 固定 C map 复用/加载。第一次确认 C 后保存；后续同一 GUI 或重启 GUI 后可只点击 A/B。
    rule_ab_static_c_map_name: str = str(_cfg("rule_ab_static_c_map_name", "default_static_c_map"))
    rule_ab_static_c_map_dir: str = str(_cfg("rule_ab_static_c_map_dir", ""))
    rule_ab_reuse_static_c_map_in_memory: bool = bool(_cfg("rule_ab_reuse_static_c_map_in_memory", True))
    rule_ab_load_static_c_map_if_exists: bool = bool(_cfg("rule_ab_load_static_c_map_if_exists", True))
    rule_ab_save_static_c_map_library: bool = bool(_cfg("rule_ab_save_static_c_map_library", True))
    rule_ab_force_reselect_c_each_run: bool = bool(_cfg("rule_ab_force_reselect_c_each_run", False))

    # A 后续帧实时分割：选择最像上一帧的候选 mask。
    # A 的形状基本不变，但像素位置会随运动而移动。
    rule_ab_a_use_temporal_mask_selection: bool = bool(_cfg("rule_ab_a_use_temporal_mask_selection", True))
    rule_ab_a_temporal_score_weight: float = float(_cfg("rule_ab_a_temporal_score_weight", 0.20))
    rule_ab_a_temporal_iou_weight: float = float(_cfg("rule_ab_a_temporal_iou_weight", 0.00))
    rule_ab_a_temporal_shape_iou_weight: float = float(_cfg("rule_ab_a_temporal_shape_iou_weight", 0.00))
    rule_ab_a_temporal_area_weight: float = float(_cfg("rule_ab_a_temporal_area_weight", 0.45))
    rule_ab_a_temporal_center_weight: float = float(_cfg("rule_ab_a_temporal_center_weight", 0.45))
    rule_ab_a_temporal_aspect_weight: float = float(_cfg("rule_ab_a_temporal_aspect_weight", 0.00))
    rule_ab_a_temporal_bbox_size_weight: float = float(_cfg("rule_ab_a_temporal_bbox_size_weight", 0.00))
    rule_ab_a_temporal_c_leak_weight: float = float(_cfg("rule_ab_a_temporal_c_leak_weight", 0.00))
    rule_ab_a_temporal_center_sigma_px: float = float(_cfg("rule_ab_a_temporal_center_sigma_px", 80.0))
    rule_ab_a_temporal_area_ratio_min: float = float(_cfg("rule_ab_a_temporal_area_ratio_min", 0.35))
    rule_ab_a_temporal_area_ratio_max: float = float(_cfg("rule_ab_a_temporal_area_ratio_max", 2.80))
    rule_ab_a_temporal_min_shape_iou: float = float(_cfg("rule_ab_a_temporal_min_shape_iou", 0.00))
    rule_ab_a_temporal_min_aspect_similarity: float = float(_cfg("rule_ab_a_temporal_min_aspect_similarity", 0.00))
    rule_ab_a_temporal_min_bbox_size_similarity: float = float(_cfg("rule_ab_a_temporal_min_bbox_size_similarity", 0.00))
    rule_ab_a_temporal_max_c_unique_leak_ratio: float = float(_cfg("rule_ab_a_temporal_max_c_unique_leak_ratio", 1.00))
    rule_ab_a_temporal_fallback_to_last_if_bad: bool = bool(_cfg("rule_ab_a_temporal_fallback_to_last_if_bad", True))
    rule_ab_a_temporal_min_accept_score: float = float(_cfg("rule_ab_a_temporal_min_accept_score", 0.05))

    # B 后续帧实时分割：选择最像上一帧的候选 mask。
    # B 的形状基本不变，但像素位置会随 A 推动而移动。
    rule_ab_b_use_temporal_mask_selection: bool = bool(_cfg("rule_ab_b_use_temporal_mask_selection", True))
    rule_ab_b_temporal_score_weight: float = float(_cfg("rule_ab_b_temporal_score_weight", 0.20))
    rule_ab_b_temporal_iou_weight: float = float(_cfg("rule_ab_b_temporal_iou_weight", 0.00))
    rule_ab_b_temporal_shape_iou_weight: float = float(_cfg("rule_ab_b_temporal_shape_iou_weight", 0.00))
    rule_ab_b_temporal_area_weight: float = float(_cfg("rule_ab_b_temporal_area_weight", 0.45))
    rule_ab_b_temporal_center_weight: float = float(_cfg("rule_ab_b_temporal_center_weight", 0.45))
    rule_ab_b_temporal_aspect_weight: float = float(_cfg("rule_ab_b_temporal_aspect_weight", 0.00))
    rule_ab_b_temporal_bbox_size_weight: float = float(_cfg("rule_ab_b_temporal_bbox_size_weight", 0.00))
    rule_ab_b_temporal_c_leak_weight: float = float(_cfg("rule_ab_b_temporal_c_leak_weight", 0.00))
    rule_ab_b_temporal_center_sigma_px: float = float(_cfg("rule_ab_b_temporal_center_sigma_px", 80.0))
    rule_ab_b_temporal_area_ratio_min: float = float(_cfg("rule_ab_b_temporal_area_ratio_min", 0.35))
    rule_ab_b_temporal_area_ratio_max: float = float(_cfg("rule_ab_b_temporal_area_ratio_max", 2.80))
    rule_ab_b_temporal_min_shape_iou: float = float(_cfg("rule_ab_b_temporal_min_shape_iou", 0.00))
    rule_ab_b_temporal_min_aspect_similarity: float = float(_cfg("rule_ab_b_temporal_min_aspect_similarity", 0.00))
    rule_ab_b_temporal_min_bbox_size_similarity: float = float(_cfg("rule_ab_b_temporal_min_bbox_size_similarity", 0.00))
    rule_ab_b_temporal_max_c_unique_leak_ratio: float = float(_cfg("rule_ab_b_temporal_max_c_unique_leak_ratio", 1.00))
    rule_ab_b_temporal_fallback_to_last_if_bad: bool = bool(_cfg("rule_ab_b_temporal_fallback_to_last_if_bad", True))
    rule_ab_b_temporal_min_accept_score: float = float(_cfg("rule_ab_b_temporal_min_accept_score", 0.05))

    rule_ab_temporal_search_margin_px: float = float(_cfg("rule_ab_temporal_search_margin_px", 60.0))
    rule_ab_temporal_use_center_point_only: bool = bool(_cfg("rule_ab_temporal_use_center_point_only", True))

    # A/B 位置跟踪：先通过上一帧图像块模板匹配预测当前帧位置，再交给 SAM2。
    rule_ab_temporal_position_match_enable: bool = bool(_cfg("rule_ab_temporal_position_match_enable", True))
    rule_ab_temporal_position_match_margin_px: float = float(_cfg("rule_ab_temporal_position_match_margin_px", 220.0))
    rule_ab_temporal_position_match_min_score: float = float(_cfg("rule_ab_temporal_position_match_min_score", 0.15))
    rule_ab_temporal_position_match_use_last_image_template: bool = bool(_cfg("rule_ab_temporal_position_match_use_last_image_template", True))
    rule_ab_temporal_stop_on_track_fail: bool = bool(_cfg("rule_ab_temporal_stop_on_track_fail", True))

    # SAM2 video predictor：仅保留首帧模板建立相关入口；完整测量后续帧不再启用 SAM2 video tracking。
    # 后续帧 Bmask 只能来自 feature tracker，失败即记录失败，不 fallback。
    rule_ab_use_sam2_video_tracking: bool = bool(_cfg("rule_ab_use_sam2_video_tracking", False))
    rule_ab_sam2_video_tracking_mode: str = str(_cfg("rule_ab_sam2_video_tracking_mode", "two_frame_anchor"))
    rule_ab_sam2_video_temp_dir: str = str(_cfg("rule_ab_sam2_video_temp_dir", "outputs/sam2_video_tracking_tmp"))
    rule_ab_sam2_video_prompt_max_points: int = int(_cfg("rule_ab_sam2_video_prompt_max_points", 5))
    rule_ab_sam2_video_area_ratio_min: float = float(_cfg("rule_ab_sam2_video_area_ratio_min", 0.35))
    rule_ab_sam2_video_area_ratio_max: float = float(_cfg("rule_ab_sam2_video_area_ratio_max", 2.80))
    rule_ab_sam2_video_center_jump_max_px: float = float(_cfg("rule_ab_sam2_video_center_jump_max_px", 220.0))
    rule_ab_sam2_video_fallback_to_template: bool = bool(_cfg("rule_ab_sam2_video_fallback_to_template", False))


    # ============================================================
    # 新方案：SAM2 只用于首帧三目标分割；后续用颜色 + 形状模板自动识别 A/B/C
    # ============================================================
    # 逻辑：
    #   1) 完整标定/初始化时，仍然允许 SAM2 根据 A/B/C 正点负点分割一次；
    #   2) 从这一次 mask 内提取每个目标的 HSV 颜色模板、面积/中心/外接框/形状模板；
    #   3) 后续完整循环测量不再每帧调用 SAM2，而是用模板自动识别 A/B/C；
    #   4) 当 A/B 部分重叠时，按 A/B 各自颜色距离逐像素分配，尽量把二者分开。
    rule_ab_use_feature_tracker_after_sam2_init: bool = bool(_cfg("rule_ab_use_feature_tracker_after_sam2_init", True))
    rule_ab_feature_tracker_save_dir: str = str(_cfg("rule_ab_feature_tracker_save_dir", ""))
    rule_ab_feature_h_tol: int = int(_cfg("rule_ab_feature_h_tol", 14))
    rule_ab_feature_s_tol: int = int(_cfg("rule_ab_feature_s_tol", 75))
    rule_ab_feature_v_tol: int = int(_cfg("rule_ab_feature_v_tol", 75))
    rule_ab_feature_min_area_px: int = int(_cfg("rule_ab_feature_min_area_px", 20))
    rule_ab_feature_morph_kernel: int = int(_cfg("rule_ab_feature_morph_kernel", 5))
    rule_ab_feature_area_ratio_min: float = float(_cfg("rule_ab_feature_area_ratio_min", 0.25))
    rule_ab_feature_area_ratio_max: float = float(_cfg("rule_ab_feature_area_ratio_max", 3.50))
    rule_ab_feature_center_sigma_px: float = float(_cfg("rule_ab_feature_center_sigma_px", 180.0))
    rule_ab_feature_shape_weight: float = float(_cfg("rule_ab_feature_shape_weight", 0.25))
    rule_ab_feature_color_weight: float = float(_cfg("rule_ab_feature_color_weight", 0.45))
    rule_ab_feature_center_weight: float = float(_cfg("rule_ab_feature_center_weight", 0.30))
    rule_ab_feature_ab_color_separation: bool = bool(_cfg("rule_ab_feature_ab_color_separation", True))
    rule_ab_feature_update_template_each_frame: bool = bool(_cfg("rule_ab_feature_update_template_each_frame", False))
    rule_ab_feature_template_update_alpha: float = float(_cfg("rule_ab_feature_template_update_alpha", 0.08))

    # Bmask 多颜色模板 + feature-only 稳定检测。
    # 说明：B 可能是多色/半透明/光照变化目标，不能只用单 mean_hsv。
    # 标定 B 正点时会提取 HSV seed；首帧 clean Bmask 内颜色与 seed 联合生成多 HSV center。
    # 后续帧只允许 feature tracker 作为 Bmask 来源；失败即记录失败，不调用 SAM2 fallback，不 hold_last。
    rule_ab_b_positive_color_seed_radius_px: int = int(_cfg("rule_ab_b_positive_color_seed_radius_px", 5))
    rule_ab_b_hsv_center_count: int = int(_cfg("rule_ab_b_hsv_center_count", 8))
    rule_ab_b_color_min_s: int = int(_cfg("rule_ab_b_color_min_s", 20))
    rule_ab_b_color_min_v: int = int(_cfg("rule_ab_b_color_min_v", 20))
    rule_ab_b_feature_h_tol: int = int(_cfg("rule_ab_b_feature_h_tol", _cfg("rule_ab_feature_h_tol", 14)))
    rule_ab_b_feature_s_tol: int = int(_cfg("rule_ab_b_feature_s_tol", _cfg("rule_ab_feature_s_tol", 75)))
    rule_ab_b_feature_v_tol: int = int(_cfg("rule_ab_b_feature_v_tol", _cfg("rule_ab_feature_v_tol", 75)))
    rule_ab_feature_enable_sam2_fallback: bool = bool(_cfg("rule_ab_feature_enable_sam2_fallback", False))
    rule_ab_bmask_allow_hold_last: bool = bool(_cfg("rule_ab_bmask_allow_hold_last", False))
    rule_ab_bmask_hold_last_max_frames: int = int(_cfg("rule_ab_bmask_hold_last_max_frames", 3))
    rule_ab_bmask_fail_max_consecutive: int = int(_cfg("rule_ab_bmask_fail_max_consecutive", 10))
    rule_ab_bmask_min_valid_area_px: int = int(_cfg("rule_ab_bmask_min_valid_area_px", 20))
    rule_ab_bmask_max_center_jump_px: float = float(_cfg("rule_ab_bmask_max_center_jump_px", _cfg("rule_ab_sam2_video_center_jump_max_px", 220.0)))
    rule_ab_b_feature_area_ratio_min: float = float(_cfg("rule_ab_b_feature_area_ratio_min", 0.03))
    rule_ab_b_feature_area_ratio_max: float = float(_cfg("rule_ab_b_feature_area_ratio_max", 8.0))

    # ============================================================
    # 指定 B 边 KLT 光流跟踪：首帧人工选边，Step1/Step7 跨轮次连续跟踪
    # ============================================================
    rule_ab_use_klt_selected_b_edge_angle: bool = bool(_cfg("rule_ab_use_klt_selected_b_edge_angle", False))
    # 完整循环测量每次开始时，是否强制重新选择 KLT 指定 B 边。
    # True：不复用 calibration.json 中的 selected_b_edge_endpoints，也不复用上一轮 b_edge_klt_state；
    #      每次点击“运行完整循环测量”后，在首次生成 A/B/C 运行模板时弹窗重新选择 B 指定边。
    # False：兼容旧逻辑，优先复用标定包中保存的 KLT 边。
    rule_ab_force_reselect_klt_edge_each_run: bool = bool(_cfg("rule_ab_force_reselect_klt_edge_each_run", False))
    rule_ab_klt_num_points: int = int(_cfg("rule_ab_klt_num_points", 20))
    rule_ab_klt_min_good_points: int = int(_cfg("rule_ab_klt_min_good_points", 6))
    rule_ab_klt_max_fit_residual_px: float = float(_cfg("rule_ab_klt_max_fit_residual_px", 4.0))
    rule_ab_klt_max_angle_jump_deg: float = float(_cfg("rule_ab_klt_max_angle_jump_deg", 25.0))
    rule_ab_klt_max_fb_error_px: float = float(_cfg("rule_ab_klt_max_fb_error_px", 2.0))
    rule_ab_klt_relocate_max_failures: int = int(_cfg("rule_ab_klt_relocate_max_failures", 3))
    rule_ab_klt_relocate_stable_required: int = int(_cfg("rule_ab_klt_relocate_stable_required", 2))
    rule_ab_klt_relocate_stable_tol_deg: float = float(_cfg("rule_ab_klt_relocate_stable_tol_deg", 2.0))
    rule_ab_klt_overlay_dir_name: str = str(_cfg("rule_ab_klt_overlay_dir_name", "ab_angle_klt_edge"))

    # ============================================================
    # 新方案：指定 B 边动态一维灰度导数扫描（替代 KLT 光流）
    # ============================================================
    # 旧 KLT/Profile 指定边角度方案已停用；当前 Step1/Step7 只使用 YOLO-OBB 长边角度。
    rule_ab_use_profile_selected_b_edge_angle: bool = bool(_cfg("rule_ab_use_profile_selected_b_edge_angle", False))
    rule_ab_profile_num_lines: int = int(_cfg("rule_ab_profile_num_lines", _cfg("rule_ab_klt_num_points", 20)))
    rule_ab_profile_half_width_px: float = float(_cfg("rule_ab_profile_half_width_px", 30.0))
    rule_ab_profile_line_spacing_px: float = float(_cfg("rule_ab_profile_line_spacing_px", 8.0))
    rule_ab_profile_smooth_kernel: int = int(_cfg("rule_ab_profile_smooth_kernel", 5))
    # <=0 表示自动阈值；>0 表示导数峰值绝对阈值。
    rule_ab_profile_gradient_threshold: float = float(_cfg("rule_ab_profile_gradient_threshold", 0.0))
    # "auto"：首帧自动判断；"positive"：暗到亮；"negative"：亮到暗；"abs"：只看绝对峰值。
    rule_ab_profile_edge_polarity: str = str(_cfg("rule_ab_profile_edge_polarity", "auto"))
    rule_ab_profile_min_good_points: int = int(_cfg("rule_ab_profile_min_good_points", _cfg("rule_ab_klt_min_good_points", 6)))
    rule_ab_profile_max_point_shift_px: float = float(_cfg("rule_ab_profile_max_point_shift_px", 20.0))
    rule_ab_profile_max_fit_residual_px: float = float(_cfg("rule_ab_profile_max_fit_residual_px", _cfg("rule_ab_klt_max_fit_residual_px", 4.0)))
    rule_ab_profile_max_angle_jump_deg: float = float(_cfg("rule_ab_profile_max_angle_jump_deg", _cfg("rule_ab_klt_max_angle_jump_deg", 25.0)))
    rule_ab_profile_fail_max: int = int(_cfg("rule_ab_profile_fail_max", 3))
    rule_ab_profile_overlay_dir_name: str = str(_cfg("rule_ab_profile_overlay_dir_name", "ab_angle_profile_edge"))

    # ============================================================
    # 当前方案：Bmask 最长边角度检测
    # ============================================================
    rule_ab_bmask_longest_edge_overlay_dir_name: str = str(_cfg("rule_ab_bmask_longest_edge_overlay_dir_name", "ab_angle_bmask_longest_edge"))
    rule_ab_bmask_longest_edge_min_area_px: int = int(_cfg("rule_ab_bmask_longest_edge_min_area_px", 20))
    rule_ab_bmask_longest_edge_morph_kernel: int = int(_cfg("rule_ab_bmask_longest_edge_morph_kernel", 3))

    # ============================================================
    # Step7 实时双线程：控制线程执行 A 沿 C 路线运动，角度线程在 move/pause 期间持续监控 Bmask 最长边角度
    # ============================================================
    rule_ab_realtime_step7_enable: bool = bool(_cfg("rule_ab_realtime_step7_enable", True))
    # Step7 角度监控间隔：real/virtual 使用同一套逻辑。
    # 默认 0.05 s 以降低“角度已经超过阈值但下一帧才检测到”的过冲概率。
    rule_ab_angle_watch_interval_s: float = float(_cfg("rule_ab_angle_watch_interval_s", 0.5))
    rule_ab_pause_check_interval_s: float = float(_cfg("rule_ab_pause_check_interval_s", 0.1))
    rule_ab_realtime_overlay_dir_name: str = str(_cfg("rule_ab_realtime_overlay_dir_name", "ab_angle_realtime_bmask_longest_edge"))
    # 每帧保存 overlay 会显著拖慢 YOLO-OBB 检测。默认关闭；需要排查时再手动打开。
    rule_ab_realtime_save_every_angle_frame: bool = bool(_cfg("rule_ab_realtime_save_every_angle_frame", False))
    rule_ab_realtime_max_duration_s: float = float(_cfg("rule_ab_realtime_max_duration_s", 0.0))  # <=0 表示不额外限制，由停止/阈值决定

    # Stage34 默认参数。为了兼容 logic.rule_ab.RuntimeConfig，仍保留单组默认速度/加速度。
    rule_ab_stage_velocity: int = int(_cfg("rule_ab_stage_velocity", 10))
    rule_ab_stage_acceleration: int = int(_cfg("rule_ab_stage_acceleration", 10))
    rule_ab_stage_max_voltage: int = int(_cfg("rule_ab_stage_max_voltage", 100))

    # 3/4 通道分别设置速度和加速度。
    rule_ab_stage_ch3_velocity: int = int(_cfg("rule_ab_stage_ch3_velocity", 10))
    rule_ab_stage_ch3_acceleration: int = int(_cfg("rule_ab_stage_ch3_acceleration", 10))
    rule_ab_stage_ch4_velocity: int = int(_cfg("rule_ab_stage_ch4_velocity", 10))
    rule_ab_stage_ch4_acceleration: int = int(_cfg("rule_ab_stage_ch4_acceleration", 10))
    rule_ab_stage_ch3_max_voltage: int = int(_cfg("rule_ab_stage_ch3_max_voltage", 50))
    rule_ab_stage_ch4_max_voltage: int = int(_cfg("rule_ab_stage_ch4_max_voltage", 50))

    # 如果速度/加速度已经设为 1 仍过快，优先减小这些步数。
    rule_ab_stage_step_x: int = int(_cfg("rule_ab_stage_step_x", 20))
    rule_ab_stage_step_y: int = int(_cfg("rule_ab_stage_step_y", 20))
    rule_ab_action_step: int = int(_cfg("rule_ab_action_step", 20))

    # CH3/CH4 每次动作后的强制停顿。
    # 需求：
    #   CH3：不论 LEFT / RIGHT，执行一次运动后都等待 2 s，再进入下一帧闭环。
    #   CH4：不论 UP / DOWN，执行一次运动后都等待 2 s，再进入下一帧闭环。
    rule_ab_stage_ch3_pause_after_move_s: float = float(_cfg("rule_ab_stage_ch3_pause_after_move_s", 2.0))
    rule_ab_stage_ch4_pause_after_move_s: float = float(_cfg("rule_ab_stage_ch4_pause_after_move_s", 2.0))

    rule_ab_stage_x_sign: int = int(_cfg("rule_ab_stage_x_sign", -1))
    rule_ab_stage_y_sign: int = int(_cfg("rule_ab_stage_y_sign", 1))

    # Step9：B 和 C 的重合区域中心对齐控制。
    # 旧版本是面积阈值；现在改为：
    #   计算 B_mask ∩ C_mask 的中心点；
    #   与提前选定的目标点比较；
    #   偏差超过容差时，用 1/2 通道的四个对角方向把重合中心拉回目标点。
    rule_ac_area_threshold_px: float = float(_cfg("rule_ac_area_threshold_px", 100.0))  # 旧字段，保留兼容，不再作为 Step9 判据
    rule_ac_enable_stage: bool = bool(_cfg("rule_ac_enable_stage", True))
    rule_ac_max_cycles: int = int(_cfg("rule_ac_max_cycles", 50))
    rule_ac_stage_axis: str = str(_cfg("rule_ac_stage_axis", "horizontal"))             # 旧字段，保留兼容
    rule_ac_stage_direction: int = int(_cfg("rule_ac_stage_direction", 1))              # 旧字段，保留兼容
    rule_ac_stage_step_size: int = int(_cfg("rule_ac_stage_step_size", 20))
    rule_ac_stage_device_id: str = str(_cfg("rule_ac_stage_device_id", "97101208"))

    # Step9 目标点与控制参数。target_x/y 为截图区域内坐标，不是屏幕全局坐标。
    # 小于 0 表示尚未人工选择；单独测试时会弹窗选择，完整测量时会跳过/提示。
    rule_ac_target_x_px: float = float(_cfg("rule_ac_target_x_px", -1.0))
    rule_ac_target_y_px: float = float(_cfg("rule_ac_target_y_px", -1.0))
    rule_ac_center_tolerance_px: float = float(_cfg("rule_ac_center_tolerance_px", 5.0))
    rule_ac_loop_interval_s: float = float(_cfg("rule_ac_loop_interval_s", 0.10))
    rule_ac_stage12_velocity: int = int(_cfg("rule_ac_stage12_velocity", 10))
    rule_ac_stage12_acceleration: int = int(_cfg("rule_ac_stage12_acceleration", 10))
    rule_ac_stage12_max_voltage: int = int(_cfg("rule_ac_stage12_max_voltage", 50))

    # Step9：颜色检测区域中心对齐。
    # include：直接选择“需要的区域”的颜色，只保留接近该颜色的区域。
    # exclude：选择“其他/背景颜色”，凡是不接近该颜色的区域都视作需要的区域。
    rule_ac_color_mode: str = str(_cfg("rule_ac_color_mode", "include"))
    rule_ac_color_h: int = int(_cfg("rule_ac_color_h", -1))
    rule_ac_color_s: int = int(_cfg("rule_ac_color_s", -1))
    rule_ac_color_v: int = int(_cfg("rule_ac_color_v", -1))
    rule_ac_color_h_tol: int = int(_cfg("rule_ac_color_h_tol", 12))
    rule_ac_color_s_tol: int = int(_cfg("rule_ac_color_s_tol", 70))
    rule_ac_color_v_tol: int = int(_cfg("rule_ac_color_v_tol", 70))
    rule_ac_color_min_area_px: int = int(_cfg("rule_ac_color_min_area_px", 50))
    rule_ac_color_morph_kernel: int = int(_cfg("rule_ac_color_morph_kernel", 5))


    # 完整测量前统一标定包。
    # 该文件用于保存：角度检测B点/检测边、RuleAB的A点、全局C点/static C、Step9目标点与颜色。
    calibration_path: str = str(_cfg("calibration_path", ""))
    require_full_calibration_before_run: bool = bool(_cfg("require_full_calibration_before_run", True))

    # -------------------- 完整循环测量补焦参数（与光谱补焦循环 UI 共用） --------------------
    focus_roi: Tuple[int, int, int, int] = tuple(_cfg("focus_roi", (0, 0, 300, 300)))  # type: ignore

    # 光谱补焦循环专用截图区域与 ROI，独立于标定/角度检测使用的 capture_area
    saf_capture_area: Tuple[int, int, int, int] = tuple(_cfg("saf_capture_area", (116, 100, 1112, 850)))  # type: ignore
    saf_focus_roi: Tuple[int, int, int, int] = tuple(_cfg("saf_focus_roi", (0, 0, 300, 300)))  # type: ignore

    focus_trigger_ratio: float = float(_cfg("focus_trigger_ratio", 0.95))
    focus_stop_ratio: float = float(_cfg("focus_stop_ratio", 0.95))
    focus_trigger_count: int = int(_cfg("focus_trigger_count", 3))
    focus_trigger_absolute: bool = bool(_cfg("focus_trigger_absolute", True))
    focus_detection_only: bool = bool(_cfg("focus_detection_only", False))
    focus_z_enabled: bool = bool(_cfg("focus_z_enabled", True))
    focus_z_axis: int = int(_cfg("focus_z_axis", 1))
    focus_z_speed: int = int(_cfg("focus_z_speed", 100))
    focus_z_accel: int = int(_cfg("focus_z_accel", 100))
    focus_search_strategy: str = str(_cfg("focus_search_strategy", "hill_climb"))
    focus_max_checks_per_cycle: int = int(_cfg("focus_max_checks_per_cycle", 20))
    focus_check_interval_s: float = float(_cfg("focus_check_interval_s", 0.2))
    focus_z_search_steps: int = int(_cfg("focus_z_search_steps", 10))
    focus_z_patience: int = int(_cfg("focus_z_patience", 3))
    focus_z_direction_probe_steps: Tuple[int, ...] = tuple(_cfg("focus_z_direction_probe_steps", (10, 20, 30)))  # type: ignore
    focus_z_direction_probe_stage_count: int = int(_cfg("focus_z_direction_probe_stage_count", 3))
    focus_z_direction_probe_step_interval: int = int(_cfg("focus_z_direction_probe_step_interval", 10))
    focus_z_direction_probe_samples: int = int(_cfg("focus_z_direction_probe_samples", 3))
    focus_z_direction_probe_points_per_step: int = int(_cfg("focus_z_direction_probe_points_per_step", 5))
    focus_z_local_refine_enabled: bool = bool(_cfg("focus_z_local_refine_enabled", True))
    focus_z_local_refine_decay: float = float(_cfg("focus_z_local_refine_decay", 0.5))
    focus_z_local_refine_min_step: int = int(_cfg("focus_z_local_refine_min_step", 1))
    focus_z_local_refine_max_rounds: int = int(_cfg("focus_z_local_refine_max_rounds", 4))

    def __post_init__(self):
        """
        初始化后若补焦专用区域仍为默认值，则默认同步为标定/角度检测区域，
        保证 backwards compatibility。

        重要：saf_capture_area / saf_focus_roi 是补焦专用区域，与 capture_area /
        focus_roi 完全独立。用户通过 GUI「光谱补焦循环」面板或运行时的全屏 ROI
        选择修改它们，不会反向影响标定 ABC、角度检测等画面区域。
        """
        default_saf_capture = (116, 100, 1112, 850)
        default_saf_focus = (0, 0, 300, 300)
        if getattr(self, "saf_capture_area", None) == default_saf_capture:
            self.saf_capture_area = tuple(int(v) for v in self.capture_area)
        if getattr(self, "saf_focus_roi", None) == default_saf_focus:
            self.saf_focus_roi = tuple(int(v) for v in self.focus_roi)

    # 运行日志实时保存到文件
    save_log_to_file: bool = bool(_cfg("save_log_to_file", True))
    log_dir: str = str(_cfg("log_dir", "./Log"))


# ============================================================
# 完整测量前统一标定状态
# ============================================================

@dataclass
class CalibrationState:
    """
    完整循环测量前的统一标定包。

    设计目的：
        1. 所有需要人工点选/标定的内容，在初始化实验设备之前完成；
        2. 标定结果保存为 JSON，后续运行完整循环测量时直接加载；
        3. 完整测量启动前先做 preflight 检查，缺少任何关键标定则禁止开始。

    坐标约定：
        所有点坐标均为 capture_area 截图区域内坐标，不是屏幕全局坐标。
    """
    calibration_version: int = 1
    created_at: str = ""
    updated_at: str = ""
    capture_area: List[int] = field(default_factory=list)
    angle_model_path: str = ""

    # 角度检测：B mask 正点/负点，以及要检测的边。
    angle_b_positive_points: List[List[float]] = field(default_factory=list)
    angle_b_negative_points: List[List[float]] = field(default_factory=list)
    angle_edge_index: int = -1
    angle_edge_name: str = ""
    angle_num: int = 0
    angle_cw: int = 0

    # A 推动 B：A mask 正点/负点。必要时也可保存 B 点，便于后续扩展。
    rule_ab_a_positive_points: List[List[float]] = field(default_factory=list)
    rule_ab_a_negative_points: List[List[float]] = field(default_factory=list)
    rule_ab_b_positive_points: List[List[float]] = field(default_factory=list)
    rule_ab_b_negative_points: List[List[float]] = field(default_factory=list)

    # 整体测量前：C mask 正点/负点，以及保存好的 static C mask/map。
    global_c_positive_points: List[List[float]] = field(default_factory=list)
    global_c_negative_points: List[List[float]] = field(default_factory=list)
    static_c_map_dir: str = ""
    static_c_mask_path: str = ""
    static_c_edge_route_path: str = ""

    # 新方案：首帧 SAM2 分割后保存的 A/B/C 颜色与形状模板。
    # feature_profile_json 保存完整模板元数据；feature_profile_dir 保存初始 mask / overlay。
    feature_profile_dir: str = ""
    feature_profile_json: str = ""

    # B 多颜色模板：标定 B 正点附近提取的 HSV 种子，后续用于生成 b_hsv_centers。
    b_positive_hsv_seeds: List[List[float]] = field(default_factory=list)
    b_positive_hsv_seed_count: int = 0

    # 指定 B 边 KLT 光流跟踪：首帧人工选边后保存的物理边描述。
    selected_b_edge_endpoints: List[List[float]] = field(default_factory=list)
    selected_b_edge_angle: float = float("nan")
    selected_b_edge_length: float = float("nan")
    selected_b_edge_midpoint: List[float] = field(default_factory=list)
    selected_b_edge_descriptor: Dict[str, Any] = field(default_factory=dict)

    # Step9：颜色区域中心对齐。
    step9_target_x_px: float = -1.0
    step9_target_y_px: float = -1.0
    step9_color_mode: str = "include"
    step9_color_h: int = -1
    step9_color_s: int = -1
    step9_color_v: int = -1
    step9_color_h_tol: int = 12
    step9_color_s_tol: int = 70
    step9_color_v_tol: int = 70
    step9_color_min_area_px: int = 50
    step9_color_morph_kernel: int = 5
    step9_center_tolerance_px: float = 5.0

    # 手动排除 ROI：多个多边形，每个 ROI 是闭合顶点序列。
    # 用于排除显微镜视野中的干扰图案（如表情包）。
    exclude_roi_polygons: List[List[List[float]]] = field(default_factory=list)

    notes: str = ""

    @staticmethod
    def _points_to_lists(points: Any) -> List[List[float]]:
        out: List[List[float]] = []
        if not points:
            return out
        for p in points:
            try:
                out.append([float(p[0]), float(p[1])])
            except Exception:
                continue
        return out

    @staticmethod
    def _normalize_roi_polygons(raw: Any) -> List[List[List[float]]]:
        """把 ROI 多边形描述归一化为 List[List[List[float]]]。"""
        out: List[List[List[float]]] = []
        if not raw:
            return out
        if isinstance(raw, np.ndarray):
            raw = raw.tolist()
        if not isinstance(raw, (list, tuple)):
            return out

        # 如果 raw 是二维点列表，包成单个多边形
        if len(raw) > 0 and not isinstance(raw[0], (list, tuple, dict)):
            try:
                poly = [[float(p[0]), float(p[1])] for p in raw]
                if len(poly) >= 3:
                    out.append(poly)
            except Exception:
                pass
            return out

        for item in raw:
            if item is None:
                continue
            if isinstance(item, np.ndarray):
                item = item.tolist()
            if not isinstance(item, (list, tuple)) or len(item) == 0:
                continue
            try:
                poly: List[List[float]] = []
                for p in item:
                    if isinstance(p, dict):
                        x = float(p.get("x", p.get("X", 0.0)))
                        y = float(p.get("y", p.get("Y", 0.0)))
                    else:
                        x, y = float(p[0]), float(p[1])
                    poly.append([x, y])
                if len(poly) >= 3:
                    out.append(poly)
            except Exception:
                continue
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrationState":
        kwargs = {}
        for k in cls.__dataclass_fields__.keys():  # type: ignore[attr-defined]
            if k in data:
                kwargs[k] = data[k]
        state = cls(**kwargs)
        state.angle_b_positive_points = cls._points_to_lists(state.angle_b_positive_points)
        state.angle_b_negative_points = cls._points_to_lists(state.angle_b_negative_points)
        state.rule_ab_a_positive_points = cls._points_to_lists(state.rule_ab_a_positive_points)
        state.rule_ab_a_negative_points = cls._points_to_lists(state.rule_ab_a_negative_points)
        state.rule_ab_b_positive_points = cls._points_to_lists(state.rule_ab_b_positive_points)
        state.rule_ab_b_negative_points = cls._points_to_lists(state.rule_ab_b_negative_points)
        state.global_c_positive_points = cls._points_to_lists(state.global_c_positive_points)
        state.global_c_negative_points = cls._points_to_lists(state.global_c_negative_points)
        # ROI 多边形归一化：兼容 list/tuple/dict 等多种描述。
        state.exclude_roi_polygons = cls._normalize_roi_polygons(state.exclude_roi_polygons)
        # HSV seed 是 [h,s,v] 三元组，不能用 _points_to_lists 截成二维点。
        hsv_seeds: List[List[float]] = []
        for p in (state.b_positive_hsv_seeds or []):
            try:
                hsv_seeds.append([float(p[0]), float(p[1]), float(p[2])])
            except Exception:
                continue
        state.b_positive_hsv_seeds = hsv_seeds
        try:
            state.b_positive_hsv_seed_count = int(len(state.b_positive_hsv_seeds))
        except Exception:
            state.b_positive_hsv_seed_count = 0
        return state

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def missing_items(self) -> List[str]:
        missing: List[str] = []
        # 当前 Step1/Step7 角度来源为 YOLO-OBB，不再需要标定 Bmask、人工选 B 边、RuleAB-B 点。
        if not self.rule_ab_a_positive_points:
            missing.append("A推动B：A mask 正点")
        if not self.global_c_positive_points and not self.static_c_map_dir and not self.static_c_mask_path:
            missing.append("整体测量：C mask 正点或 static C map")
        if self.static_c_map_dir:
            cdir = Path(str(self.static_c_map_dir))
            if not ((cdir / "static_c_mask.png").exists() or (cdir / "static_c_reference" / "static_c_mask.png").exists()):
                missing.append("整体测量：static_c_map_dir 中未找到 static_c_mask.png")
        if self.step9_target_x_px < 0 or self.step9_target_y_px < 0:
            missing.append("Step9：颜色区域中心目标点")
        # 当前 Step9 使用颜色检测区域中心对齐，因此仍需要颜色 HSV 标定。
        if self.step9_color_h < 0 or self.step9_color_s < 0 or self.step9_color_v < 0:
            missing.append("Step9：颜色 HSV")
        return missing


# ============================================================
# 虚拟硬件适配层：只替换硬件 I/O，不改完整测量主逻辑
# ============================================================

def _is_virtual_mode_value(value: Any) -> bool:
    return str(value or "real").strip().lower() in ("virtual", "sim", "simulation", "dryrun", "dry-run")


class VirtualIlluminationRelay:
    def __init__(self, port: str = "", log_func=None):
        self.port = port
        self.log_func = log_func
        self.is_open = False
        self.is_on = False

    def _log(self, msg: str):
        if callable(self.log_func):
            self.log_func(msg)

    def open(self):
        self.is_open = True
        self._log(f"[虚拟照明] open(port={self.port})：未连接真实照明继电器")
        return True

    def on(self):
        self.is_on = True
        self._log("[虚拟照明] ON：未控制真实硬件")
        return True

    def off(self):
        self.is_on = False
        self._log("[虚拟照明] OFF：未控制真实硬件")
        return True

    def close(self):
        self.is_open = False
        self.is_on = False
        self._log("[虚拟照明] close：未释放真实硬件")
        return True

    def release(self):
        return self.close()


class VirtualNewportPicomotor8742:
    def __init__(self, log_func=None):
        self.log_func = log_func
        self.axes = (1, 2, 3, 4)
        self.positions = {int(a): 0 for a in self.axes}
        self.last_query = []

    def _log(self, msg: str):
        if callable(self.log_func):
            self.log_func(msg)

    def get_id(self):
        return "Virtual Newport/Picomotor8742"

    def get_all_axes(self):
        return list(self.axes)

    def query(self, cmd: str, axis: Optional[int] = None):
        self.last_query.append((cmd, axis))
        self._log(f"[虚拟Newport] query(axis={axis}, cmd={cmd})：未连接真实控制器")
        return "OK"

    def move_by(self, *args, **kwargs):
        axis = kwargs.get("axis", None)
        steps = kwargs.get("steps", kwargs.get("distance", None))
        if axis is None and len(args) >= 2:
            # 兼容 move_by(distance, channel=...) 之外的少数位置参数形式
            steps = args[0]
        if axis is None:
            axis = kwargs.get("channel", 1)
        if steps is None and args:
            steps = args[0]
        axis = int(axis)
        steps = float(steps or 0)
        self.positions[axis] = self.positions.get(axis, 0.0) + steps
        self._log(f"[虚拟Newport] move_by(axis={axis}, steps={steps})，virtual_pos={self.positions[axis]}；未控制真实硬件")
        return True

    def wait_move(self, axis: Optional[int] = None):
        self._log(f"[虚拟Newport] wait_move(axis={axis})：立即完成")
        return True

    def stop(self, *args, **kwargs):
        axis = kwargs.get("axis", kwargs.get("channel", None))
        self._log(f"[虚拟Newport] stop(axis={axis})")
        return True

    def stop_all(self):
        self._log("[虚拟Newport] stop_all()")
        return True

    def close(self):
        self._log("[虚拟Newport] close()：未释放真实硬件")
        return True


class VirtualKinesisPiezoMotor:
    def __init__(self, serial: str = "", log_func=None):
        self.serial = str(serial)
        self.log_func = log_func
        self.positions = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        self.channel_settings: Dict[int, Dict[str, Any]] = {}
        self.action_history: List[Dict[str, Any]] = []

    def _log(self, msg: str):
        if callable(self.log_func):
            self.log_func(msg)

    def setup_channel(self, channel: int, max_voltage: int = 50, velocity: int = 10, acceleration: int = 10):
        self.channel_settings[int(channel)] = {
            "max_voltage": int(max_voltage),
            "velocity": int(velocity),
            "acceleration": int(acceleration),
        }
        self._log(f"[虚拟Thorlabs] setup_channel(CH{int(channel)}, V={max_voltage}, v={velocity}, a={acceleration})")
        return True

    def setup_drive(self, max_voltage: int = 50, velocity: int = 10, acceleration: int = 10, channel: int = 1):
        return self.setup_channel(channel=channel, max_voltage=max_voltage, velocity=velocity, acceleration=acceleration)

    def _get_channel_setting(self, channel: int) -> Dict[str, Any]:
        return dict(self.channel_settings.get(int(channel), {}))

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return float(default)
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _to_int_or_none(value: Any) -> Optional[int]:
        try:
            if value is None:
                return None
            return int(value)
        except Exception:
            return None

    @staticmethod
    def _direction_to_sign(value: Any, default: int = 1) -> int:
        if isinstance(value, str):
            s = value.strip().lower()
            if s in ("-", "-1", "neg", "negative", "minus", "left", "down", "reverse", "backward", "ccw-"):
                return -1
            if s in ("+", "+1", "1", "pos", "positive", "plus", "right", "up", "forward", "cw+"):
                return 1
        try:
            return 1 if float(value) >= 0 else -1
        except Exception:
            return int(default)

    def move_by(self, *args, **kwargs):
        channel = kwargs.get("channel", kwargs.get("axis", kwargs.get("ch", None)))
        distance = kwargs.get("distance", kwargs.get("steps", kwargs.get("step", None)))
        if distance is None and args:
            distance = args[0]
        if channel is None:
            channel = 1
        channel = int(channel)
        distance = float(distance or 0.0)
        self.positions[channel] = self.positions.get(channel, 0.0) + distance
        self._log(f"[虚拟Thorlabs] move_by(CH{channel}, distance={distance})，virtual_pos={self.positions[channel]}；未控制真实运动台")
        return True

    def execute_rule_action(self, *args, **kwargs) -> Dict[str, Any]:
        """
        兼容 logic.actual_nano_boundary_following...ActualNanoBoundaryFollower.execute_action()
        对真实 Stage 封装的调用。

        真实 RuleAB 里通常会先把 action_id 映射成 channel/direction/move_step，
        然后调用 stage.execute_rule_action(...). virtual 模式不能缺这个接口，
        否则 Step7 控制线程会因为 AttributeError 停止完整测量。

        本函数只更新虚拟 position 和写日志，不连接、不控制真实 Thorlabs。
        为了兼容不同版本的 RuleAB stage wrapper，这里接受 *args/**kwargs，
        支持两类调用：
          1) execute_rule_action(channel=3, direction=1, move_step=20, ...)
          2) execute_rule_action(action_id) 作为兜底。
        """
        action_id = kwargs.get("action_id", None)
        if action_id is None and args:
            # 某些旧调用可能直接传 action_id；若后面还传 channel，则 channel 优先。
            try:
                action_id = int(args[0])
            except Exception:
                action_id = None

        channel = kwargs.get("channel", kwargs.get("axis", kwargs.get("ch", kwargs.get("stage_channel", None))))
        channel_name = kwargs.get("channel_name", kwargs.get("axis_name", ""))
        direction = kwargs.get("direction", kwargs.get("sign", kwargs.get("move_direction", None)))
        move_step = kwargs.get(
            "move_step",
            kwargs.get("step", kwargs.get("steps", kwargs.get("step_size", kwargs.get("distance", kwargs.get("amount", None))))),
        )
        pause_s = self._to_float(kwargs.get("pause_s", kwargs.get("pause_after_move_s", kwargs.get("wait_s", 0.0))), 0.0)

        # 如果 RuleAB 没有传 channel，只传了 action_id，则做一个兜底映射。
        # 当前完整测量 _route_action_from_error 语义：0=STAY, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT。
        # 旧版 RuleAB 有时用 ±1/±2/±10/±20 表示不同方向，这里也兼容。
        fallback_desc = ""
        if channel is None:
            aid = self._to_int_or_none(action_id)
            if aid in (0, None):
                info = {
                    "ok": True, "dry_run": True, "virtual": True, "action_id": aid or 0,
                    "channel": None, "channel_name": "STAY", "move_step": 0.0,
                    "pause_s": 0.0, "voltage": None, "velocity": None, "acceleration": None,
                    "reason": "stay_or_no_channel",
                }
                self.action_history.append(info)
                self._log(f"[虚拟Thorlabs] execute_rule_action(action_id={aid}) -> STAY；未控制真实运动台")
                return info
            # 图像方向兜底：左右走 CH3，上下走 CH4。
            if aid == 1:      # UP
                channel, direction, fallback_desc = 4, 1, "UP/CH4+"
            elif aid == 2:    # DOWN
                channel, direction, fallback_desc = 4, -1, "DOWN/CH4-"
            elif aid == 3:    # LEFT
                channel, direction, fallback_desc = 3, -1, "LEFT/CH3-"
            elif aid == 4:    # RIGHT
                channel, direction, fallback_desc = 3, 1, "RIGHT/CH3+"
            elif aid == -1:
                channel, direction, fallback_desc = 3, -1, "legacy -1/CH3-"
            elif aid == -2:
                channel, direction, fallback_desc = 4, -1, "legacy -2/CH4-"
            elif aid == 10:
                channel, direction, fallback_desc = 3, 1, "legacy RIGHT/CH3+"
            elif aid == -10:
                channel, direction, fallback_desc = 3, -1, "legacy LEFT/CH3-"
            elif aid == 20:
                channel, direction, fallback_desc = 4, 1, "legacy UP/CH4+"
            elif aid == -20:
                channel, direction, fallback_desc = 4, -1, "legacy DOWN/CH4-"
            else:
                channel, direction, fallback_desc = 3, 1, f"unknown_action_{aid}/fallback_CH3+"

        channel = int(channel)
        sign = self._direction_to_sign(direction, default=1)
        step = abs(self._to_float(move_step, 20.0))
        signed_step = float(sign) * float(step)

        # 更新虚拟位置。真实硬件不动作。
        self.positions[channel] = self.positions.get(channel, 0.0) + signed_step
        setting = self._get_channel_setting(channel)
        voltage = kwargs.get("voltage", kwargs.get("max_voltage", setting.get("max_voltage", "?")))
        velocity = kwargs.get("velocity", setting.get("velocity", "?"))
        acceleration = kwargs.get("acceleration", setting.get("acceleration", "?"))
        if not channel_name:
            channel_name = f"CH{channel}"

        info = {
            "ok": True,
            "dry_run": True,
            "virtual": True,
            "action_id": self._to_int_or_none(action_id),
            "channel": int(channel),
            "channel_name": str(channel_name),
            "direction": int(sign),
            "move_step": float(signed_step),
            "requested_step_abs": float(step),
            "pause_s": float(pause_s),
            "voltage": voltage,
            "velocity": velocity,
            "acceleration": acceleration,
            "virtual_position": float(self.positions[channel]),
            "fallback_desc": fallback_desc,
            "raw_args": [str(a) for a in args],
            "raw_kwargs_keys": sorted([str(k) for k in kwargs.keys()]),
        }
        self.action_history.append(info)
        self._log(
            f"[虚拟Thorlabs] execute_rule_action(action_id={info['action_id']}, "
            f"channel=CH{channel}, direction={sign}, move_step={signed_step})，"
            f"virtual_pos={self.positions[channel]}，pause_s={pause_s}，"
            f"V={voltage}, v={velocity}, a={acceleration}；未控制真实 Stage"
            + (f"；fallback={fallback_desc}" if fallback_desc else "")
        )
        return info

    def wait_move(self, *args, **kwargs):
        channel = kwargs.get("channel", kwargs.get("axis", None))
        self._log(f"[虚拟Thorlabs] wait_move(channel={channel})：立即完成")
        return True

    def wait_for_stop(self, *args, **kwargs):
        channel = kwargs.get("channel", kwargs.get("axis", None))
        self._log(f"[虚拟Thorlabs] wait_for_stop(channel={channel})：立即完成")
        return True

    def get_position(self, channel: int = 1):
        return float(self.positions.get(int(channel), 0.0))

    def stop(self, *args, **kwargs):
        channel = kwargs.get("channel", kwargs.get("axis", None))
        self._log(f"[虚拟Thorlabs] stop(channel={channel})")
        return True

    def stop_all(self):
        self._log("[虚拟Thorlabs] stop_all()")
        return True

    def close(self):
        self._log(f"[虚拟Thorlabs] close(serial={self.serial})")
        return True


class VirtualLabVIEWTCPServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 65432, output_dir: str = "labview_csv_output", log_func=None):
        self.host = host
        self.port = int(port)
        self.output_dir = output_dir
        self.log_func = log_func
        self.ready = True
        self.connected = True
        self.last_result: Optional[Dict[str, Any]] = None

    def _log(self, msg: str):
        if callable(self.log_func):
            self.log_func(msg)

    def start_server_async(self):
        self.ready = True
        self.connected = True
        self._log(f"[虚拟TCP] start_server_async(host={self.host}, port={self.port})：未打开真实端口")
        return True

    def wait_ready(self):
        self.ready = True
        self.connected = True
        self._log("[虚拟TCP] wait_ready()：立即 READY")
        return True

    def request_measure(self, command: str = "MEASURE", index: int = 0, save_csv: bool = True):
        # 返回状态包即可；真正光谱由 MeasurementWorkflow 的 virtual branch 生成，保证与原保存字段一致。
        self.last_result = {"ok": True, "index": int(index), "command": command, "method": "virtual_request_measure", "num_points": 0}
        self._log(f"[虚拟TCP] request_measure(command={command}, index={index})：未请求真实 LabVIEW")
        return self.last_result

    def close(self):
        self.connected = False
        self.ready = False
        self._log("[虚拟TCP] close()")
        return True


class MeasurementWorkflow:
    def __init__(self, cfg: MeasurementConfig, on_log=None, on_update=None):
        self.cfg = cfg
        self.on_log = on_log
        self.on_update = on_update

        self.light: Optional[IlluminationRelay] = None
        self.signal_generator: Optional[RigolDG4062Controller] = None
        self.laser_stage: Optional[Newport.Picomotor8742] = None
        self.angle_module: Optional[Any] = None
        self.tcp_server: Optional[LabVIEWTCPServer] = None
        self.rule_ab_follower: Optional[ActualNanoBoundaryFollower] = None
        self.rule_ac_controller: Optional[RuleACOverlapController] = None
        self.stage12_device: Optional[Any] = None

        # Step9：颜色检测区域中心对齐专用运行状态。
        # 单独测试 Step9 时，C 从“选择C文件夹”加载 static_c_mask.png；
        # B 由 SAM2 分割/跟踪得到。停止按钮会把 step9_stop_requested 置 True，
        # 并尝试停止 1/2 通道。
        self.step9_stop_requested: bool = False
        self.step9_b_positive_points: List[Tuple[float, float]] = []
        self.step9_b_negative_points: List[Tuple[float, float]] = []
        self.step9_b_last_mask: Optional[np.ndarray] = None
        self.step9_b_last_center: Optional[Tuple[float, float]] = None
        self.step9_b_last_box: Optional[Tuple[float, float, float, float]] = None
        self.step9_c_mask_cache: Optional[np.ndarray] = None
        self.step9_c_mask_path: Optional[str] = None
        # Step9 旧版 dynamic C 状态保留兼容；当前颜色中心对齐不再使用 B∩C_dynamic。
        self.step9_dynamic_c_last_mask: Optional[np.ndarray] = None
        self.step9_dynamic_c_last_raw_mask: Optional[np.ndarray] = None
        self.step9_dynamic_c_last_center: Optional[Tuple[float, float]] = None
        self.step9_dynamic_c_last_box: Optional[Tuple[float, float, float, float]] = None
        self.step9_dynamic_c_last_geometry: Optional[Dict[str, Any]] = None
        self.step9_sam2_predictor: Optional[Any] = None
        self.step9_sam2_model_info: Optional[Dict[str, Any]] = None

        # Step9：颜色检测区域中心对齐专用状态。
        self.step9_color_mode: str = str(getattr(cfg, "rule_ac_color_mode", "include"))
        self.step9_color_hsv: Optional[Tuple[int, int, int]] = None
        if int(getattr(cfg, "rule_ac_color_h", -1)) >= 0:
            self.step9_color_hsv = (
                int(getattr(cfg, "rule_ac_color_h", -1)),
                int(getattr(cfg, "rule_ac_color_s", -1)),
                int(getattr(cfg, "rule_ac_color_v", -1)),
            )
        self.step9_color_last_mask: Optional[np.ndarray] = None
        self.step9_color_last_center: Optional[Tuple[float, float]] = None

        # Step9 单独测试每次点击都会创建一个独立运行文件夹。
        # 运行过程中所有截图、B/C/color/region mask、overlay、CSV 日志都写入该文件夹。
        self.step9_current_run_dir: Optional[Path] = None
        self.step9_history_csv_path: Optional[Path] = None

        # GUI 会在单独测试 Step9 前注入该回调。
        # Step9 循环每一轮开始前调用它，把 GUI 当前输入同步到 workflow.cfg，
        # 因此运动过程中修改 Step9 输入框，下一轮就会使用新数值。
        self.step9_config_sync_callback: Optional[Any] = None

        # GUI 会在完整测量 Step7 前注入该回调。
        # RuleAB 循环每一步开始前调用它，把 GUI 当前输入同步到 workflow.cfg，
        # 因此运行完整循环测量时修改 RuleAB/RuleAC 单独测试面板中的数值，
        # 下一步/下一帧就会按新数值执行，不需要停止程序重开。
        self.rule_ab_config_sync_callback: Optional[Any] = None

        self.is_measuring: bool = False
        self.stop_requested: bool = False

        # 完整测量运行中途暂停重标定状态机。
        # 目标：不中止完整测量总线程，只暂停当前 cycle 的 Stage34/Step9 运动，
        # 人工重新标定 A/B/C/Step9 后，让当前 cycle 从 Step1 重新开始。
        self.pause_for_recalibration_requested: bool = False
        self.recalibration_in_progress: bool = False
        self.resume_after_recalibration_requested: bool = False
        self.restart_current_cycle_after_recalibration: bool = False
        self.midrun_recalibration_cycle_index: Optional[int] = None

        self.save_threads: List[threading.Thread] = []
        self.save_threads_lock = threading.Lock()
        self.save_io_lock = threading.Lock()

        # Step7 实时双线程用锁：普通 3/4 通道 move 只允许控制线程发；
        # angle_monitor_thread 只允许设置 stop_event 和 emergency stop Stage34；不允许直接关闭激光。
        self.rule_ab_stage34_lock = threading.Lock()
        self.rule_ab_laser_lock = threading.Lock()
        self.rule_ab_vision_lock = threading.Lock()

        self.run_session_dir: Optional[Path] = None
        self.run_session_name: Optional[str] = None

        self.previous_cycle_angle: Optional[float] = None

        # 指定 B 边 KLT 光流跨轮次跟踪状态。
        # 首帧人工指定一次 B 的目标物理边后，Step1/Step7 都从这里继续跟踪。
        self.b_edge_klt_state: Dict[str, Any] = {
            "initialized": False,
            "selected": False,
            # 兼容旧字段：新 profile 方法不再使用 prev_pts / calcOpticalFlowPyrLK。
            "prev_gray": None,
            "prev_pts": None,
            "last_gray": None,
            "last_edge_endpoints": None,
            "last_angle": None,
            "last_valid": False,
            "descriptor": None,
            "profile_polarity": None,
            "failure_count": 0,
            "relocate_stable_count": 0,
            "last_relocated_angle": None,
            "last_label": "",
        }

        self.context: Dict[str, Any] = {
            "cycle_index": 0,
            "light_on": False,
            "laser_on": False,
            "signal_ch1_on": False,
            "signal_ch2_on": False,
            "tcp_started": False,
            "labview_ready": False,

            "last_angle_result": None,
            "angle_before": None,
            "angle_after": None,
            "save_angle_deg": None,           # 本轮真正写入 CSV/XLSX/绘图的角度；当前固定为 Step1 最终角度
            "angle_delta": None,              # 本轮角度 - 上一轮角度
            "angle_before_after_delta": None, # 同一轮 before/after 差值，仅保留用于查看，不参与 Step 11
            "previous_cycle_angle": None,
            "signal_on_time_used_ms": None,
            "signal_on_time_next_ms": self.cfg.signal_on_time_ms,
            "signal_time_adjust_action": None,

            "last_tcp_result": None,
            "last_rule_ab_result": None,
            "last_rule_ac_result": None,
            "raw_values": None,
            "raw_filtered_values": None,
            "raw_median_values": None,
            "raw_original_peak": None,
            "raw_filtered_peak": None,
            "raw_median_peak": None,
            "raw_filter_threshold": self.cfg.raw_remove_above,
            "median_filter_window": self.cfg.median_filter_window,
            "x_axis_xlsx_path": self.cfg.x_axis_xlsx_path,
            "x_axis_values": None,
            "raw_peak": None,
            "fit_peak": None,
            "w0": None,
            "w": None,
            "delta_w": None,
            "valid_change": None,

            "plot_points": [],

            # 完整测量专用 C 锁定状态：
            # full_measurement_mode=True 后，RuleAB/Step9 只能使用本次 run_session 下
            # full_calibration_static_c/ 目录，禁止再从 RuleAB/RuleAC 单独测试面板读取残留 C。
            "full_measurement_mode": False,
            "strict_full_calibration_c_locked": False,
            "strict_full_calibration_c_dir": "",
            "strict_full_calibration_c_mask_path": "",

            "midrun_recalibration_requested": False,
            "midrun_recalibration_in_progress": False,
            "restart_current_cycle_after_recalibration": False,
            "midrun_recalibration_cycle_index": None,
        }

        self.plot_points: List[Dict[str, Any]] = []
        self.context["plot_points"] = self.plot_points

        self.output_root = Path(cfg.save_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

        # 日志管理器（实时保存运行日志到文件）
        # 必须在 _ensure_summary_xlsx 等可能调用 self.log() 的方法之前初始化。
        self._log_manager: Optional[LogManager] = None
        self._init_log_manager()

        self.summary_date_dir = self._get_today_save_dir()
        self.summary_csv_path = self.summary_date_dir / "measurement_summary.csv"

        self.summary_xlsx_created_at = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.summary_xlsx_path = self.summary_date_dir / f"measurement_summary_{self.summary_xlsx_created_at}.xlsx"
        self.summary_xlsx_headers = [
            "序号（第几次测量）",
            "本次角度/deg",
            "本次原始数据峰值",
            "本次拟合数据峰值",
            "文件名",
            "信号发生器打开时间/ms",
        ]
        self._ensure_summary_xlsx()

        # --------------------------------------------------------
        # 聚焦参考图与补焦组件（完整循环测量使用）
        # --------------------------------------------------------
        self._focus_reference_image: Optional[np.ndarray] = None
        self._focus_reference_ready: bool = False
        self._focus_roi_selected: bool = False
        self._focus_metrics_calc: Optional[FocusMetricsCalculator] = None
        self._focus_scorer: Optional[FocusScorer] = None
        self._focus_controller: Optional[AutofocusController] = None
        self._focus_simulator: Optional[Any] = None
        self._focus_consecutive_low_count: int = 0
        self._last_autofocus_search_failed: bool = False
        self._focus_autofocus_event_counter: int = 0

    def _init_log_manager(self) -> None:
        """初始化日志管理器。"""
        if getattr(self.cfg, "save_log_to_file", True):
            log_dir = str(getattr(self.cfg, "log_dir", "./Log"))
            # 不再把 self.on_log 透传给 LogManager，避免双写：
            # self.log() 已经负责把内容同时写入 LogManager 和 self.on_log。
            self._log_manager = LogManager(
                log_dir=log_dir,
                enabled=True,
                on_log=None,
            )
            self._log_manager.start()

    def _shutdown_log_manager(self) -> None:
        """关闭日志管理器。"""
        if self._log_manager is not None:
            self._log_manager.stop()
            self._log_manager = None

    def _get_today_save_dir(self) -> Path:
        date_dir = datetime.now().strftime("%m.%d")
        summary_dir = self.output_root / "save" / date_dir
        summary_dir.mkdir(parents=True, exist_ok=True)
        return summary_dir

    def _refresh_summary_paths_for_output_root(self):
        """
        当 GUI 中保存目录发生变化时，重新计算 CSV/XLSX 汇总文件路径。
        """
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.summary_date_dir = self._get_today_save_dir()
        self.summary_csv_path = self.summary_date_dir / "measurement_summary.csv"
        self.summary_xlsx_path = (
            self.summary_date_dir /
            f"measurement_summary_{self.summary_xlsx_created_at}.xlsx"
        )
        self._ensure_summary_xlsx()

    def log(self, msg: str):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{now}] {msg}"

        # 写入日志管理器（实时保存到文件）
        if self._log_manager is not None:
            self._log_manager.log(line)

        # 原有回调或打印
        if self.on_log is not None:
            self.on_log(line)
        else:
            print(line)

    def notify_update(self):
        if self.on_update is not None:
            self.on_update(self)


    # --------------------------------------------------------
    # 完整测量前统一标定包：加载、保存、应用、启动前检查
    # --------------------------------------------------------

    def get_calibration_path(self) -> Path:
        """返回当前完整测量使用的统一标定 JSON 路径。"""
        p = str(getattr(self.cfg, "calibration_path", "") or "").strip()
        if p:
            return Path(p).expanduser().resolve()
        return (Path(str(getattr(self.cfg, "save_root", "measurement_output"))) / "calibration" / "current_calibration.json").resolve()

    def load_calibration_state(self, required: bool = False) -> Optional[CalibrationState]:
        """从 JSON 读取统一标定包，并自动同步到 cfg。"""
        path = self.get_calibration_path()
        if not path.exists():
            if required:
                raise RuntimeError(f"未找到完整测量标定文件：{path}")
            self.log(f"[完整测量标定] 标定文件不存在：{path}")
            return None

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        state = CalibrationState.from_dict(data)
        self.apply_calibration_state(state)
        self.context["calibration_path"] = str(path)
        self.context["calibration_state"] = state.to_dict()
        self.log(f"[完整测量标定] 已加载标定文件：{path}")
        return state

    def save_calibration_state(self, state: CalibrationState) -> Path:
        """保存统一标定包到 JSON。"""
        state = self._complete_calibration_state_for_runtime(state)

        path = self.get_calibration_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not state.created_at:
            state.created_at = now
        state.updated_at = now
        if not state.capture_area:
            state.capture_area = [int(v) for v in self.cfg.capture_area]
        state.angle_model_path = str(getattr(self.cfg, "angle_model_path", ""))

        with path.open("w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)

        self.context["calibration_path"] = str(path)
        self.context["calibration_state"] = state.to_dict()
        self.log(f"[完整测量标定] 已保存标定文件：{path}")
        return path

    @staticmethod
    def _calib_points_to_tuples(points: Any) -> List[Tuple[float, float]]:
        """把标定包里的 [[x,y], ...] 统一转成 [(x,y), ...]。"""
        out: List[Tuple[float, float]] = []
        if not points:
            return out
        for p in points:
            try:
                out.append((float(p[0]), float(p[1])))
            except Exception:
                continue
        return out

    def _get_loaded_calibration_state(self) -> Optional[CalibrationState]:
        """从 context 中取已经加载过的完整标定包。"""
        data = self.context.get("calibration_state")
        if isinstance(data, CalibrationState):
            return data
        if isinstance(data, dict):
            try:
                return CalibrationState.from_dict(data)
            except Exception:
                return None
        return None

    def _complete_calibration_state_for_runtime(self, state: CalibrationState) -> CalibrationState:
        """
        补全运行时必须用到的标定点。

        关键修复：
            1. 原完整标定只强制检查 RuleAB-A 点，没有强制检查 RuleAB-B 点；
            2. Step7 的 RuleAB 初始化需要 A/B/C 三类对象；
            3. 如果 RuleAB-B 未单独标定，则自动复用“角度检测 B”的正负点作为 RuleAB-B。
        """
        if state is None:
            return state

        # RuleAB-B 没有单独标定时，复用角度检测 B 点。
        if not state.rule_ab_b_positive_points and state.angle_b_positive_points:
            state.rule_ab_b_positive_points = self._points_to_plain_lists(state.angle_b_positive_points)
            state.rule_ab_b_negative_points = self._points_to_plain_lists(state.angle_b_negative_points)
            self.log(
                "[完整测量标定] RuleAB-B 未单独标定，已自动复用角度检测 B 点作为 RuleAB-B："
                f"B+={len(state.rule_ab_b_positive_points)}, B-={len(state.rule_ab_b_negative_points)}"
            )

        # 反向兜底：没有点击“1 标定角度B/边”中的 B 点，但已经单独标定了 RuleAB-B，
        # 则角度检测 B 也复用 RuleAB-B，避免 B 缺失导致 ABC 初始化/SAM2 prompt 顺序不稳定。
        if not state.angle_b_positive_points and state.rule_ab_b_positive_points:
            state.angle_b_positive_points = self._points_to_plain_lists(state.rule_ab_b_positive_points)
            state.angle_b_negative_points = self._points_to_plain_lists(state.rule_ab_b_negative_points)
            self.log(
                "[完整测量标定] 角度B未单独标定，已自动复用 RuleAB-B 点作为角度检测 B："
                f"B+={len(state.angle_b_positive_points)}, B-={len(state.angle_b_negative_points)}"
            )

        # static C 文件夹规范化。
        resolved_c_dir = self._resolve_static_c_dir_from_state(state)
        if resolved_c_dir:
            state.static_c_map_dir = str(resolved_c_dir)
            mask_path = Path(resolved_c_dir) / "static_c_mask.png"
            if mask_path.exists():
                state.static_c_mask_path = str(mask_path.resolve())
            route_path = Path(resolved_c_dir) / "static_c_edge_route.json"
            if route_path.exists():
                state.static_c_edge_route_path = str(route_path.resolve())

        return state

    @staticmethod
    def _points_to_plain_lists(points: Any) -> List[List[float]]:
        out: List[List[float]] = []
        if not points:
            return out
        for p in points:
            try:
                out.append([float(p[0]), float(p[1])])
            except Exception:
                continue
        return out

    @staticmethod
    def _safe_setattr(obj: Any, name: str, value: Any) -> bool:
        """兼容普通对象、dataclass、slots 对象；能写就写，不能写就跳过。"""
        if obj is None:
            return False
        try:
            setattr(obj, name, value)
            return True
        except Exception:
            return False

    def _resolve_static_c_dir_from_state(self, state: CalibrationState) -> str:
        """从标定包解析后续 RuleAB/Step9 应加载的 static C 文件夹。"""
        candidates: List[Path] = []
        if state.static_c_map_dir:
            p = Path(str(state.static_c_map_dir)).expanduser()
            candidates.extend([p, p / "static_c_reference"])
        if state.static_c_mask_path:
            mp = Path(str(state.static_c_mask_path)).expanduser()
            candidates.extend([mp.parent, mp.parent / "static_c_reference"])
        if getattr(state, "static_c_edge_route_path", ""):
            rp = Path(str(state.static_c_edge_route_path)).expanduser()
            candidates.extend([rp.parent, rp.parent / "static_c_reference"])

        for c in candidates:
            try:
                if c.exists() and c.is_dir() and (c / "static_c_mask.png").exists():
                    return str(c.resolve())
            except Exception:
                continue
        return str(state.static_c_map_dir or "")

    def _get_strict_full_calibration_c_dir(self) -> str:
        """
        返回完整测量运行时被锁定的 C 目录。

        只要 strict_full_calibration_c_locked=True，后续 RuleAB/Step9 不允许再使用
        GUI 单独测试区的 rule_ab_static_c_map_dir，而必须使用这个目录。
        """
        if not bool(self.context.get("strict_full_calibration_c_locked", False)):
            return ""
        c_dir = str(self.context.get("strict_full_calibration_c_dir", "") or "").strip()
        if not c_dir:
            return ""
        p = Path(c_dir).expanduser()
        try:
            if p.exists() and p.is_dir() and (p / "static_c_mask.png").exists():
                return str(p.resolve())
        except Exception:
            return ""
        return ""

    @staticmethod
    def _find_contours_compat(image: np.ndarray, mode: int, method: int) -> List[np.ndarray]:
        """
        兼容 OpenCV 3.x 与 4.x 的 findContours 返回值差异。

        OpenCV 3.x: image, contours, hierarchy = cv2.findContours(...)
        OpenCV 4.x: contours, hierarchy = cv2.findContours(...)
        """
        result = cv2.findContours(image, mode, method)
        if len(result) == 3:
            _, contours, _ = result
        else:
            contours, _ = result
        return list(contours)

    def _force_cfg_to_strict_full_calibration_c(self, reason: str = "") -> str:
        """
        把 workflow.cfg 强制锁定到本次完整测量专用 C。

        这个函数是阻断旧 C 污染的核心：
            - sync_config_from_ui_to_workflow() 每轮都会从 GUI 读参数；
            - GUI 中 RuleAB/RuleAC 单独测试区可能残留旧 C 文件夹；
            - 因此每次同步后都必须重新把 cfg.rule_ab_static_c_map_dir 改回
              full_calibration_static_c。
        """
        c_dir = self._get_strict_full_calibration_c_dir()
        if not c_dir:
            return ""

        self.cfg.rule_ab_static_c_map_dir = c_dir
        self.cfg.rule_ab_load_static_c_map_if_exists = True
        self.cfg.rule_ab_force_reselect_c_each_run = False
        self.cfg.rule_ab_reuse_static_c_map_in_memory = True
        self.cfg.rule_ab_confirm_first_frame_segmentation = False

        mask_path = str((Path(c_dir) / "static_c_mask.png").resolve())
        self.context["strict_full_calibration_c_mask_path"] = mask_path

        # 同步 context 中的标定包，防止后续 _get_loaded_calibration_state() 再读到旧 C。
        data = self.context.get("calibration_state")
        if isinstance(data, dict):
            data = dict(data)
            data["static_c_map_dir"] = c_dir
            data["static_c_mask_path"] = mask_path
            self.context["calibration_state"] = data

        if reason:
            self.log(f"[完整测量标定] 已锁定完整测量专用 C：{c_dir}；reason={reason}")
        return c_dir

    def apply_calibration_state(self, state: CalibrationState):
        """
        把完整标定包同步到后续运行配置和 workflow 缓存。

        关键修复：
            旧版只同步了 angle_num/cw、static C、Step9 target/HSV；
            现在会把 A/B/C 正点负点也保存到 context，并准备注入 RuleAB/SAM2 跟踪模块。
        """
        if state is None:
            return

        state = self._complete_calibration_state_for_runtime(state)

        # 保存完整标定状态，后续 ensure_rule_ab_follower()/init_angle_module()/Step9 都从这里取。
        self.context["calibration_state"] = state.to_dict()
        self.context["calibration_loaded_for_runtime"] = True

        # 坐标区域一致性提示，不直接阻止运行；真正缺项由 preflight_check_calibration() 判断。
        try:
            if state.capture_area and tuple(int(v) for v in state.capture_area) != tuple(int(v) for v in self.cfg.capture_area):
                self.log(
                    "[完整测量标定] 警告：标定包 capture_area="
                    f"{state.capture_area} 与当前 GUI capture_area={self.cfg.capture_area} 不一致；"
                    "点坐标仍按截图区域内坐标使用，请确认截图区域没有改变。"
                )
        except Exception:
            pass

        # 角度检测的边选择仍通过现有 angle_num / angle_cw 进入 ScreenAngleDetector。
        if state.angle_num is not None:
            self.cfg.angle_num = int(state.angle_num)
        if state.angle_cw is not None:
            self.cfg.angle_cw = int(state.angle_cw)

        # C map：完整测量中的 RuleAB/Step9 只允许加载完整标定 C。
        # 如果已经进入完整测量并创建了 full_calibration_static_c，则必须优先使用锁定目录，
        # 禁止被 GUI 单独测试区残留的 C 文件夹覆盖。
        resolved_c_dir = self._get_strict_full_calibration_c_dir() or self._resolve_static_c_dir_from_state(state)
        if resolved_c_dir:
            self.cfg.rule_ab_static_c_map_dir = str(resolved_c_dir)
            self.cfg.rule_ab_load_static_c_map_if_exists = True
            self.cfg.rule_ab_force_reselect_c_each_run = False
            self.cfg.rule_ab_reuse_static_c_map_in_memory = True
            state.static_c_map_dir = str(resolved_c_dir)
            mask_path = Path(str(resolved_c_dir)) / "static_c_mask.png"
            if mask_path.exists():
                state.static_c_mask_path = str(mask_path.resolve())

        # 完整标定已加载时，不允许后续完整测量再弹第一帧确认/重选 C。
        # 单独测试按钮如果确实需要重新标点，可重新点击对应标定按钮。
        self.cfg.rule_ab_confirm_first_frame_segmentation = False
        self.cfg.rule_ab_force_reselect_c_each_run = False
        self.cfg.rule_ab_load_static_c_map_if_exists = True

        # Step9 颜色区域中心对齐。
        self.cfg.rule_ac_target_x_px = float(state.step9_target_x_px)
        self.cfg.rule_ac_target_y_px = float(state.step9_target_y_px)
        self.cfg.rule_ac_color_mode = str(state.step9_color_mode or "include")
        self.cfg.rule_ac_color_h = int(state.step9_color_h)
        self.cfg.rule_ac_color_s = int(state.step9_color_s)
        self.cfg.rule_ac_color_v = int(state.step9_color_v)
        self.cfg.rule_ac_color_h_tol = int(state.step9_color_h_tol)
        self.cfg.rule_ac_color_s_tol = int(state.step9_color_s_tol)
        self.cfg.rule_ac_color_v_tol = int(state.step9_color_v_tol)
        self.cfg.rule_ac_color_min_area_px = int(state.step9_color_min_area_px)
        self.cfg.rule_ac_color_morph_kernel = int(state.step9_color_morph_kernel)
        self.cfg.rule_ac_center_tolerance_px = float(state.step9_center_tolerance_px)

        self.step9_color_mode = self.cfg.rule_ac_color_mode
        if self.cfg.rule_ac_color_h >= 0:
            self.step9_color_hsv = (
                int(self.cfg.rule_ac_color_h),
                int(self.cfg.rule_ac_color_s),
                int(self.cfg.rule_ac_color_v),
            )

        # Step9 如果以后仍需要 B 点/SAM2，则这里也预置，避免单独准备时再弹 B 点选择。
        self.step9_b_positive_points = self._calib_points_to_tuples(state.rule_ab_b_positive_points)
        self.step9_b_negative_points = self._calib_points_to_tuples(state.rule_ab_b_negative_points)

        self.log(
            "[完整测量标定] 已同步到运行配置："
            f"angle_num={self.cfg.angle_num}, cw={self.cfg.angle_cw}, "
            f"A点={len(state.rule_ab_a_positive_points)}/{len(state.rule_ab_a_negative_points)}, "
            f"B点={len(state.rule_ab_b_positive_points)}/{len(state.rule_ab_b_negative_points)}, "
            f"C点={len(state.global_c_positive_points)}/{len(state.global_c_negative_points)}, "
            f"static_C={self.cfg.rule_ab_static_c_map_dir}, "
            f"Step9 target=({self.cfg.rule_ac_target_x_px:.1f},{self.cfg.rule_ac_target_y_px:.1f}), "
            f"HSV=({self.cfg.rule_ac_color_h},{self.cfg.rule_ac_color_s},{self.cfg.rule_ac_color_v})"
        )

    def _materialize_full_calibration_c_for_runtime(self, state: CalibrationState) -> CalibrationState:
        """
        为完整循环测量创建“本次运行专用 C 目录”。

        为什么要这样做：
            RuleAB/RuleAC 单独测试面板里可能残留旧的 C 文件夹；
            外部 RuleAB 模块加载 static C 时还会把来源标成 legacy_reference。
            为避免完整测量误用旧 C，这里把完整标定 JSON 中的 static_c_mask.png
            复制到本次 run_session 下的 full_calibration_static_c/，后续 RuleAB 只允许加载这个目录。
        """
        if state is None:
            return state

        src_mask: Optional[Path] = None
        if state.static_c_mask_path:
            p0 = Path(str(state.static_c_mask_path)).expanduser()
            if p0.exists() and p0.is_file():
                src_mask = p0

        if src_mask is None and state.static_c_map_dir:
            p1 = Path(str(state.static_c_map_dir)).expanduser()
            for c in (p1, p1 / "static_c_reference"):
                if (c / "static_c_mask.png").exists():
                    src_mask = c / "static_c_mask.png"
                    break

        if src_mask is None or not src_mask.exists():
            raise RuntimeError(
                "完整标定包中的 C 不可用：没有找到 static_c_mask.png。"
                f" static_c_map_dir={state.static_c_map_dir}, static_c_mask_path={state.static_c_mask_path}"
            )

        base_dir = self.run_session_dir if self.run_session_dir is not None else (self.output_root / "calibration")
        runtime_dir = Path(base_dir) / "full_calibration_static_c"
        runtime_dir.mkdir(parents=True, exist_ok=True)

        dst_mask = runtime_dir / "static_c_mask.png"
        mask = cv2.imread(str(src_mask), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"读取完整标定 C mask 失败：{src_mask}")
        # 防御性处理：确保 mask 为单通道二维数组，避免某些 OpenCV/图像格式导致维度异常。
        if len(mask.shape) == 3:
            if mask.shape[2] == 1:
                mask = np.squeeze(mask)
            else:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if len(mask.shape) != 2:
            mask = np.squeeze(mask)
            if len(mask.shape) != 2:
                raise RuntimeError(f"完整标定 C mask 维度异常：{mask.shape}")
        if int(np.count_nonzero(mask > 0)) <= 0:
            raise RuntimeError(f"完整标定 C mask 为空：{src_mask}")
        cv2.imwrite(str(dst_mask), mask)

        # 同步原始 SAM2 C mask。static_c_mask.png 可能仍是四边形 mask；
        # 仍同步原始 SAM2 C mask，便于排查；Step7 默认路线使用近似四边形。
        try:
            src_sam2_mask = src_mask.parent / "static_c_sam2_mask.png"
            if src_sam2_mask.exists():
                import shutil
                shutil.copy2(str(src_sam2_mask), str(runtime_dir / "static_c_sam2_mask.png"))
        except Exception as e:
            self.log(f"[完整测量标定] 复制原始 SAM2 C mask 失败，将使用 static_c_mask.png 兜底：{e}")

        # 同步 C 边沿路线：完整测量 Step7 会优先读取本运行目录下的 static_c_edge_route.json。
        src_route: Optional[Path] = None
        if getattr(state, "static_c_edge_route_path", ""):
            rp = Path(str(state.static_c_edge_route_path)).expanduser()
            if rp.exists() and rp.is_file():
                src_route = rp
        if src_route is None and src_mask is not None:
            candidate = src_mask.parent / "static_c_edge_route.json"
            if candidate.exists():
                src_route = candidate
        if src_route is not None:
            try:
                import shutil
                shutil.copy2(str(src_route), str(runtime_dir / "static_c_edge_route.json"))
                state.static_c_edge_route_path = str((runtime_dir / "static_c_edge_route.json").resolve())
            except Exception as e:
                self.log(f"[完整测量标定] 复制 C 边沿路线失败，将尝试重建：{e}")
        # 无论原 C 文件夹里是否已有旧路线，完整测量运行目录都重新生成一次路线。
        # 当前版本默认从 C 近似四边形四条边生成。
        try:
            route_source_rt = str(getattr(self.cfg, "rule_ab_c_edge_route_source", "quad") or "quad").lower().strip()
            if route_source_rt in ("quad", "quadrilateral"):
                route_rt = self._build_and_save_quad_edge_route(runtime_dir)
            else:
                route_rt = self._build_and_save_sam2_outer_edge_route(runtime_dir, prefer_sam2_mask=True)
                if not route_rt:
                    route_rt = self._build_and_save_sam2_outer_edge_route(runtime_dir, prefer_sam2_mask=False)
            if route_rt:
                state.static_c_edge_route_path = str((runtime_dir / "static_c_edge_route.json").resolve())
                self.log(
                    f"[完整测量标定] 已按 {route_source_rt} 重建 Step7 A 路线："
                    f"points={len(route_rt)}, path={state.static_c_edge_route_path}"
                )
        except Exception as e:
            self.log(f"[完整测量标定] 重建 C 四边形路线失败：{e}")

        # 同步保存 npy/csv，兼容外部模块可能读取这些辅助文件。
        yx = np.column_stack(np.where(mask > 0))
        try:
            np.save(str(runtime_dir / "static_c_pixels_yx.npy"), yx.astype(np.int32))
            with (runtime_dir / "static_c_reference.csv").open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["index", "y", "x"])
                for idx, (y, x) in enumerate(yx):
                    writer.writerow([idx, int(y), int(x)])
        except Exception as e:
            self.log(f"[完整测量标定] 保存运行时 C 辅助文件失败，但 static_c_mask.png 已保存：{e}")

        # 生成外部 ActualNanoBoundaryFollower 能识别的 static_c_map_meta.json。
        # 旧版只写 static_c_meta.json，外部模块会把它当成 legacy_reference，
        # overlay 上显示 C=static_c_map_loaded_from_legacy_reference。
        mask_bool = mask > 0
        bbox = None
        center = None
        contour_xy = []
        area_float = float(np.count_nonzero(mask_bool))
        try:
            ys, xs = np.where(mask_bool)
            bbox = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
            m = cv2.moments((mask_bool.astype(np.uint8) * 255))
            if abs(m.get("m00", 0.0)) > 1e-8:
                center = [float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])]
            else:
                center = [float(np.mean(xs)), float(np.mean(ys))]

            contours = self._find_contours_compat((mask_bool.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                peri = float(cv2.arcLength(contour, True))
                approx = cv2.approxPolyDP(contour, max(1.0, 0.012 * peri), True)
                contour_xy = [[float(x), float(y)] for x, y in approx.reshape(-1, 2)]
                if len(contour_xy) < 3:
                    contour_xy = [[float(x), float(y)] for x, y in contour.reshape(-1, 2)]
                area_float = float(cv2.contourArea(contour)) or area_float
        except Exception as e:
            self.log(f"[完整测量标定] 生成 C meta 几何信息失败，将使用 bbox 兜底：{e}")

        if not contour_xy and bbox is not None:
            x1, y1, x2, y2 = bbox
            contour_xy = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        if center is None and bbox is not None:
            x1, y1, x2, y2 = bbox
            center = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]

        meta = {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "created_time": time.time(),
            "source": "full_calibration_static_c",
            "source_static_c_mask": str(src_mask.resolve()),
            "runtime_static_c_dir": str(runtime_dir.resolve()),
            "static_c_map_name": "full_calibration_static_c",
            "mask_path": str(dst_mask.resolve()),
            "edge_route_path": str((runtime_dir / "static_c_edge_route.json").resolve()) if (runtime_dir / "static_c_edge_route.json").exists() else "",
            "pixels_path": str((runtime_dir / "static_c_pixels_yx.npy").resolve()),
            "bbox": bbox,
            "center": center,
            "area": area_float,
            "mask_area_px": int(np.count_nonzero(mask_bool)),
            "contour": contour_xy,
            "purpose": "strict_full_calibration_c_for_measurement_run",
            "note": "完整测量只允许 RuleAB/Step9 加载本目录，不允许使用 GUI 单独测试区域残留的 C 文件夹。",
        }
        try:
            with (runtime_dir / "static_c_map_meta.json").open("w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            with (runtime_dir / "static_c_meta.json").open("w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        state.static_c_map_dir = str(runtime_dir.resolve())
        state.static_c_mask_path = str(dst_mask.resolve())
        if (runtime_dir / "static_c_edge_route.json").exists():
            state.static_c_edge_route_path = str((runtime_dir / "static_c_edge_route.json").resolve())
        self.context["full_measurement_mode"] = True
        self.context["strict_full_calibration_c_locked"] = True
        self.context["strict_full_calibration_c_dir"] = str(runtime_dir.resolve())
        self.context["strict_full_calibration_c_mask_path"] = str(dst_mask.resolve())
        self.context["strict_full_calibration_c_runtime_dir"] = str(runtime_dir.resolve())
        self.context["calibration_state"] = state.to_dict()
        self.cfg.rule_ab_static_c_map_dir = str(runtime_dir.resolve())
        self.cfg.rule_ab_load_static_c_map_if_exists = True
        self.cfg.rule_ab_force_reselect_c_each_run = False
        self.cfg.rule_ab_confirm_first_frame_segmentation = False
        self.log(
            "[完整测量标定] 已创建并锁定本次完整测量专用 C 目录："
            f"{runtime_dir.resolve()}；来源={src_mask.resolve()}；area={meta['mask_area_px']} px；"
            "外部 RuleAB 应显示 source=full_calibration_static_c"
        )
        return state

    def prepare_runtime_from_full_calibration(self, state: CalibrationState):
        """
        完整测量启动前调用：把完整标定强制应用到所有运行模块。
        """
        state = self._complete_calibration_state_for_runtime(state)
        state = self._materialize_full_calibration_c_for_runtime(state)
        self.apply_calibration_state(state)

        # 已经存在的模块可能是旧参数创建的；完整测量前清空，下一次创建时注入标定点。
        self.rule_ab_follower = None
        self.rule_ac_controller = None

        # 已初始化的角度模块不需要重新加载 YOLO，但要同步运行参数和标定点属性。
        if self.angle_module is not None:
            try:
                if hasattr(self.angle_module, "num"):
                    self.angle_module.num = int(self.cfg.angle_num)
                if hasattr(self.angle_module, "cw"):
                    self.angle_module.cw = int(self.cfg.angle_cw)
                self._inject_calibration_into_angle_module(self.angle_module, state)
            except Exception as e:
                self.log(f"[完整测量标定] 向角度模块注入标定失败，将继续运行：{e}")

    def _inject_calibration_into_angle_module(self, angle_module: Any, state: CalibrationState):
        """把角度检测 B 点/边编号写入 ScreenAngleDetector 可能支持的属性。"""
        if angle_module is None or state is None:
            return

        b_pos = self._calib_points_to_tuples(state.angle_b_positive_points)
        b_neg = self._calib_points_to_tuples(state.angle_b_negative_points)
        attrs = {
            "precalibration_state": state.to_dict(),
            "calibration_state": state.to_dict(),
            "angle_b_positive_points": b_pos,
            "angle_b_negative_points": b_neg,
            "b_positive_points": b_pos,
            "b_negative_points": b_neg,
            "preloaded_positive_points": b_pos,
            "preloaded_negative_points": b_neg,
            "edge_index": int(state.angle_edge_index),
            "angle_edge_index": int(state.angle_edge_index),
            "angle_edge_name": str(state.angle_edge_name),
            "skip_interactive_selection": True,
            "use_precalibration": True,
        }
        for k, v in attrs.items():
            self._safe_setattr(angle_module, k, v)

    def _inject_calibration_into_rule_ab_follower(self, follower: Any, state: CalibrationState):
        """
        把完整标定的 A/B/C 点和 static C 目录写入 ActualNanoBoundaryFollower 及其内部 segmenter/cfg。

        这里采用“宽兼容注入”：不同版本的 ActualNanoBoundaryFollower 字段名可能不同，
        因此同时写入多组常见字段名；不存在的字段会自动跳过。
        """
        if follower is None or state is None:
            return

        a_pos = self._calib_points_to_tuples(state.rule_ab_a_positive_points)
        a_neg = self._calib_points_to_tuples(state.rule_ab_a_negative_points)
        b_pos = self._calib_points_to_tuples(state.rule_ab_b_positive_points)
        b_neg = self._calib_points_to_tuples(state.rule_ab_b_negative_points)
        c_pos = self._calib_points_to_tuples(state.global_c_positive_points)
        c_neg = self._calib_points_to_tuples(state.global_c_negative_points)
        c_dir = self._resolve_static_c_dir_from_state(state)
        payload = state.to_dict()

        common_attrs = {
            "precalibration_state": payload,
            "calibration_state": payload,
            "full_calibration_state": payload,
            "use_precalibration": True,
            "use_full_calibration": True,
            "skip_interactive_selection": True,
            "skip_first_frame_confirmation": True,
            "confirm_first_frame_segmentation": False,
            "force_reselect_c_each_run": False,
            "load_static_c_map_if_exists": True,
            "static_c_map_dir": c_dir,
            "static_c_mask_path": str(state.static_c_mask_path or ""),
            "a_positive_points": a_pos,
            "a_negative_points": a_neg,
            "b_positive_points": b_pos,
            "b_negative_points": b_neg,
            "c_positive_points": c_pos,
            "c_negative_points": c_neg,
            "initial_a_positive_points": a_pos,
            "initial_a_negative_points": a_neg,
            "initial_b_positive_points": b_pos,
            "initial_b_negative_points": b_neg,
            "initial_c_positive_points": c_pos,
            "initial_c_negative_points": c_neg,
            "rule_ab_a_positive_points": a_pos,
            "rule_ab_a_negative_points": a_neg,
            "rule_ab_b_positive_points": b_pos,
            "rule_ab_b_negative_points": b_neg,
            "global_c_positive_points": c_pos,
            "global_c_negative_points": c_neg,
            "preloaded_a_positive_points": a_pos,
            "preloaded_a_negative_points": a_neg,
            "preloaded_b_positive_points": b_pos,
            "preloaded_b_negative_points": b_neg,
            "preloaded_c_positive_points": c_pos,
            "preloaded_c_negative_points": c_neg,
            "exclude_roi_polygons": state.exclude_roi_polygons,
        }

        targets = [follower, getattr(follower, "cfg", None), getattr(follower, "abc_segmenter", None), getattr(follower, "segmenter", None)]
        for obj in targets:
            if obj is None:
                continue
            for k, v in common_attrs.items():
                self._safe_setattr(obj, k, v)

        # 某些版本把点存成 dict。
        points_dict = {
            "A": {"positive": a_pos, "negative": a_neg},
            "B": {"positive": b_pos, "negative": b_neg},
            "C": {"positive": c_pos, "negative": c_neg},
            "a": {"positive": a_pos, "negative": a_neg},
            "b": {"positive": b_pos, "negative": b_neg},
            "c": {"positive": c_pos, "negative": c_neg},
        }
        for obj in targets:
            self._safe_setattr(obj, "preloaded_points", points_dict)
            self._safe_setattr(obj, "calibration_points", points_dict)
            self._safe_setattr(obj, "prompt_points", points_dict)

    def _call_initialize_abc_with_loaded_calibration(self, follower: Any, state: CalibrationState):
        """
        初始化 RuleAB 的 A/B/C。

        优先尝试使用“带标定点参数”的初始化接口；如果外部模块版本没有这些接口，
        就先注入属性，再调用原 initialize_abc_with_first_frame()。
        """
        if follower is None:
            raise RuntimeError("RuleAB follower=None，不能初始化 A/B/C。")

        self._inject_calibration_into_rule_ab_follower(follower, state)

        a_pos = self._calib_points_to_tuples(state.rule_ab_a_positive_points)
        a_neg = self._calib_points_to_tuples(state.rule_ab_a_negative_points)
        b_pos = self._calib_points_to_tuples(state.rule_ab_b_positive_points)
        b_neg = self._calib_points_to_tuples(state.rule_ab_b_negative_points)
        c_pos = self._calib_points_to_tuples(state.global_c_positive_points)
        c_neg = self._calib_points_to_tuples(state.global_c_negative_points)
        c_dir = self._resolve_static_c_dir_from_state(state)

        kwargs = {
            "calibration_state": state,
            "calibration": state,
            "state": state,
            "a_positive_points": a_pos,
            "a_negative_points": a_neg,
            "b_positive_points": b_pos,
            "b_negative_points": b_neg,
            "c_positive_points": c_pos,
            "c_negative_points": c_neg,
            "static_c_map_dir": c_dir,
            "confirm": False,
            "interactive": False,
            "skip_interactive": True,
        }

        # 新版外部模块如果提供专门接口，优先调用。
        for method_name in (
            "initialize_abc_with_calibration",
            "initialize_abc_from_calibration",
            "initialize_abc_with_points",
            "initialize_from_calibration",
            "initialize_with_calibration",
            "initialize_abc_precalibrated",
        ):
            m = getattr(follower, method_name, None)
            if not callable(m):
                continue
            try:
                import inspect
                sig = inspect.signature(m)
                accepted = {}
                for name, p in sig.parameters.items():
                    if name in kwargs:
                        accepted[name] = kwargs[name]
                    elif p.kind == p.VAR_KEYWORD:
                        accepted = kwargs
                        break
                m(**accepted)
                self.log(f"[完整测量标定] 已通过 {method_name}() 初始化 RuleAB A/B/C，未弹窗。")
                return
            except TypeError:
                continue
            except Exception as e:
                self.log(f"[完整测量标定] {method_name}() 初始化失败，尝试其他方式：{e}")

        # 旧版接口：先把点和 skip 标志注入 follower/follower.cfg/segmenter，再调用原初始化。
        cfg_obj = getattr(follower, "cfg", None)
        if cfg_obj is not None:
            self._safe_setattr(cfg_obj, "confirm_first_frame_segmentation", False)
            self._safe_setattr(cfg_obj, "force_reselect_c_each_run", False)
            self._safe_setattr(cfg_obj, "load_static_c_map_if_exists", True)
            self._safe_setattr(cfg_obj, "static_c_map_dir", c_dir)

        # 最后的兼容分支：旧版外部模块只有 initialize_abc_with_first_frame()。
        # 这里先尝试用“禁止交互”标志和预置点运行；如果外部模块仍然弹窗，说明该模块内部
        # 没有读取这些字段，需要同步修改 logic/actual_...py。为避免完整测量卡在窗口中，
        # 本文件会在日志中明确提示当前使用的是兼容分支。
        try:
            self.log(
                "[完整测量标定] 外部 RuleAB 模块没有发现专用非交互初始化接口，"
                "将尝试用预注入 A/B/C 点调用 initialize_abc_with_first_frame()；"
                "如果仍弹窗，需要修改 logic/actual_nano_boundary_following...py 读取 preloaded 点。"
            )
            follower.initialize_abc_with_first_frame()
            self.log("[完整测量标定] 已调用 initialize_abc_with_first_frame()；标定点已提前注入 follower/cfg/segmenter。")
        except Exception as e:
            raise RuntimeError(
                "RuleAB 初始化失败。当前外部 ActualNanoBoundaryFollower 版本可能仍只支持交互式点选，"
                "需要在该模块中读取 follower/cfg 的 preloaded_a/b/c_positive_points 或 calibration_state。"
            ) from e

    def preflight_check_calibration(self) -> CalibrationState:
        """
        完整测量启动前强制检查标定包。

        必须在 connect_measurement_devices() 之前调用。
        这样可以保证所有人工标定都发生在初始化设备之前。
        """
        required = bool(getattr(self.cfg, "require_full_calibration_before_run", True))
        state = self.load_calibration_state(required=required)
        if state is None:
            raise RuntimeError("未加载完整测量标定包，不能开始完整循环测量。")

        state = self._complete_calibration_state_for_runtime(state)

        missing = state.missing_items()
        if missing:
            msg = "完整测量标定不完整，不能开始运行。缺少：\n" + "\n".join(f"- {x}" for x in missing)
            self.log("[完整测量标定] " + msg.replace("\n", "；"))
            raise RuntimeError(msg)

        self.apply_calibration_state(state)
        self.log(
            "[完整测量标定] 启动前检查通过："
            f"angle_source=Bmask最长边, "
            f"A点={len(state.rule_ab_a_positive_points)}/{len(state.rule_ab_a_negative_points)}, "
            f"C点={len(state.global_c_positive_points)}/{len(state.global_c_negative_points)}, "
            f"Step9 target=({state.step9_target_x_px:.1f}, {state.step9_target_y_px:.1f}), "
            f"HSV=({state.step9_color_h}, {state.step9_color_s}, {state.step9_color_v})"
        )
        return state

    def _capture_current_rule_ab_frame(self, output_dir: Path) -> np.ndarray:
        """
        截取当前固定屏幕区域，返回 RGB 图像。

        该函数现在带有截图后端重试：
            - FixedRegionScreenCapture / PIL.ImageGrab 偶发可能返回不完整帧；
            - 遇到半帧读取错误时等待下一帧再试；
            - 成功返回前检查图像尺寸与 dtype，避免把坏帧送入 A/B/C 跟踪。
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        capture_area = tuple(int(v) for v in self.cfg.capture_area)
        left, top, width, height = capture_area
        attempts = max(1, int(getattr(self.cfg, "rule_ab_frame_read_retry_attempts", 5)))
        interval_s = max(0.0, float(getattr(self.cfg, "rule_ab_frame_read_retry_interval_s", 0.15)))
        last_exc: Optional[BaseException] = None

        for attempt in range(1, attempts + 1):
            if self.stop_requested:
                raise RuntimeError("stop_requested")
            try:
                if FixedRegionScreenCapture is not None:
                    capturer = FixedRegionScreenCapture(
                        capture_area=capture_area,
                        output_dir=output_dir / "step9_capture_tmp",
                        save_image=False,
                    )
                    frame = capturer.capture()
                    arr = np.asarray(frame).astype(np.uint8)
                else:
                    # 兜底：不用 vision.screen_capture 时，使用 PIL.ImageGrab。
                    try:
                        from PIL import ImageGrab
                    except Exception as e:
                        raise RuntimeError(
                            "无法导入 FixedRegionScreenCapture，也无法导入 PIL.ImageGrab，不能截图。"
                        ) from e

                    img = ImageGrab.grab(
                        bbox=(left, top, left + width, top + height)
                    ).convert("RGB")
                    arr = np.array(img, dtype=np.uint8)

                if arr is None or arr.size <= 0:
                    raise RuntimeError("截图返回空数组")
                if arr.ndim != 3 or arr.shape[2] < 3:
                    raise RuntimeError(f"截图通道数异常：shape={getattr(arr, 'shape', None)}")
                if int(arr.shape[0]) != int(height) or int(arr.shape[1]) != int(width):
                    raise RuntimeError(
                        f"截图尺寸异常：got={arr.shape[:2]}, expected={(height, width)}"
                    )

                if arr.shape[2] > 3:
                    arr = arr[:, :, :3]
                if attempt > 1:
                    self.log(f"[截图重试] capture_area={capture_area} 第 {attempt}/{attempts} 次成功。")
                return np.ascontiguousarray(arr, dtype=np.uint8)

            except Exception as e:
                last_exc = e
                self.log(
                    f"[截图重试] capture_area={capture_area} 失败 attempt={attempt}/{attempts}: {e}"
                )
                if attempt < attempts:
                    time.sleep(interval_s)

        raise RuntimeError(
            f"截图连续失败 {attempts} 次，最后错误：{last_exc}"
        ) from last_exc

    def begin_step9_run_session(self) -> Path:
        """
        每次点击“单独测试Step9对齐”时新建一个独立文件夹。

        保存位置：
            <save_root>/step9_color_center_align/run_YYYYMMDD_HHMMSS_mmm/

        该文件夹用于保存：
            1. 每轮截图；
            2. 每轮颜色区域 mask；
            3. 每轮可选保存的 A/B mask；
            4. 每轮 overlay 检查图；
            5. step9_history.csv；
            6. step9_meta.json。
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        run_dir = self.output_root / "step9_color_center_align" / f"run_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)

        for sub in ("frames", "masks", "overlays", "color_initial", "ab_masks"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)

        self.step9_current_run_dir = run_dir
        self.step9_history_csv_path = run_dir / "step9_history.csv"

        meta = {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "capture_area": list(self.cfg.capture_area),
            "target_x_px": float(getattr(self.cfg, "rule_ac_target_x_px", -1.0)),
            "target_y_px": float(getattr(self.cfg, "rule_ac_target_y_px", -1.0)),
            "tolerance_px": float(getattr(self.cfg, "rule_ac_center_tolerance_px", 5.0)),
            "stage_step_size": int(getattr(self.cfg, "rule_ac_stage_step_size", 20)),
            "stage12_velocity": int(getattr(self.cfg, "rule_ac_stage12_velocity", 10)),
            "stage12_acceleration": int(getattr(self.cfg, "rule_ac_stage12_acceleration", 10)),
            "stage12_max_voltage": int(getattr(self.cfg, "rule_ac_stage12_max_voltage", 50)),
            "enable_stage": bool(getattr(self.cfg, "rule_ac_enable_stage", False)),
            "step9_detection_mode": "color_region_center",
            "color_mode": str(getattr(self.cfg, "rule_ac_color_mode", "include")),
            "color_hsv": [
                int(getattr(self.cfg, "rule_ac_color_h", -1)),
                int(getattr(self.cfg, "rule_ac_color_s", -1)),
                int(getattr(self.cfg, "rule_ac_color_v", -1)),
            ],
            "color_tol_hsv": [
                int(getattr(self.cfg, "rule_ac_color_h_tol", 12)),
                int(getattr(self.cfg, "rule_ac_color_s_tol", 70)),
                int(getattr(self.cfg, "rule_ac_color_v_tol", 70)),
            ],
            "color_min_area_px": int(getattr(self.cfg, "rule_ac_color_min_area_px", 50)),
            "static_c_map_dir_legacy": str(getattr(self.cfg, "rule_ab_static_c_map_dir", "")),
        }
        try:
            with (run_dir / "step9_meta.json").open("w", encoding="utf-8") as f:
                json.dump(self._json_safe(meta), f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.log(f"[Step9-颜色中心] 保存 Step9 meta 失败：{e}")

        self.log(f"[Step9-颜色中心] 新建本次 Step9 保存文件夹：{run_dir}")
        return run_dir

    def _get_step9_output_dir(self, subdir: Optional[str] = None) -> Path:
        """
        获取 Step9 当前运行文件夹；如果还没有创建，则自动创建。
        """
        if self.step9_current_run_dir is None:
            self.begin_step9_run_session()

        assert self.step9_current_run_dir is not None
        if subdir:
            p = self.step9_current_run_dir / str(subdir)
            p.mkdir(parents=True, exist_ok=True)
            return p
        return self.step9_current_run_dir

    def _append_step9_history_csv(self, row: Dict[str, Any]):
        """
        追加 Step9 单独测试/完整流程的每轮记录。
        """
        if self.step9_history_csv_path is None:
            self._get_step9_output_dir()

        assert self.step9_history_csv_path is not None
        fieldnames = [
            "cycle",
            "time",
            "ok",
            "aligned",
            "reason",
            "center_x",
            "center_y",
            "target_x",
            "target_y",
            "dx",
            "dy",
            "tolerance_px",
            "color_area_px",
            "a_mask_area_px",
            "b_mask_area_px",
            "action_code",
            "action_name",
            "moved",
            "dry_run",
            "ch1_distance",
            "ch2_distance",
            "overlay_path",
            "frame_path",
            "b_mask_path",
            "color_mask_path",
            "a_mask_path",
            "b_mask_path",
        ]
        file_exists = self.step9_history_csv_path.exists()
        with self.step9_history_csv_path.open("a", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    def _is_virtual_hardware_mode(self) -> bool:
        return _is_virtual_mode_value(getattr(self.cfg, "hardware_mode", "real"))

    # --------------------------------------------------------
    # 设备连接
    # --------------------------------------------------------

    def connect_measurement_devices(self):
        self.log("========== 初始化实验设备 ==========")

        self.connect_light()
        # 不再初始化 Rigol 信号发生器。
        # 当前流程的激光开/关由 Newport 8743-CL / Picomotor 轴运动完成。
        self.connect_laser_controller()
        self.init_angle_module()

        self.log("========== 实验设备初始化完成：照明光 + 激光控制器 + 角度检测 ==========")
        self.notify_update()

    def connect_all(self):
        self.connect_measurement_devices()

    def connect_light(self):
        # 如果已有连接但串口号变化，先关闭旧连接并重新创建
        if self.light is not None:
            current_port = getattr(self.light, "port", None)
            if current_port != self.cfg.light_port:
                self.log(
                    f"[照明光] 串口变更：{current_port} -> {self.cfg.light_port}，"
                    f"关闭旧连接并重新连接"
                )
                try:
                    self.light.close()
                except Exception as e:
                    self.log(f"[照明光] 关闭旧连接失败：{e}")
                self.light = None
            else:
                self.log("[照明光] 已连接")
                return

        if self._is_virtual_hardware_mode():
            self.log(f"[照明光][virtual] 创建虚拟照明对象：port={self.cfg.light_port}；不连接真实硬件")
            self.light = VirtualIlluminationRelay(port=self.cfg.light_port, log_func=self.log)
            self.light.open()
            self.log("[照明光][virtual] 连接成功")
            return

        self.log(f"[照明光] 正在连接：{self.cfg.light_port}")
        self.light = IlluminationRelay(port=self.cfg.light_port)
        self.light.open()
        self.log("[照明光] 连接成功")

    def connect_laser_controller(self):
        """
        连接 Newport 8743-CL，并设置激光开关轴的速度/加速度。

        这里沿用你单独测试成功的方式：
            Newport.Picomotor8742()
            stage.query("VA...", axis=axis)
            stage.query("AC...", axis=axis)

        不使用 setup_velocity()，避免 8743-CL 在 pyLabLib 中回读 VA?/AC? 异常。
        """
        if self.laser_stage is not None:
            self.log("[激光开关] Newport 8743-CL 已连接")
            return

        if self._is_virtual_hardware_mode():
            self.log("[激光开关][virtual] 创建虚拟 Newport/Picomotor8742；不连接真实硬件")
            self.laser_stage = VirtualNewportPicomotor8742(log_func=self.log)
            self.set_laser_velocity_accel(
                axis=int(self.cfg.laser_axis),
                speed=int(self.cfg.laser_speed),
                accel=int(self.cfg.laser_accel),
            )
            self.context["laser_on"] = False
            self.log("[激光开关][virtual] 虚拟激光控制器连接成功")
            return

        if Newport is None:
            raise RuntimeError("当前环境无法导入 pylablib.devices.Newport；请安装 pylablib 或切换 hardware_mode=virtual。")

        self.log("[激光开关] 正在连接 Newport 8743-CL / Picomotor")
        self.laser_stage = Newport.Picomotor8742(0)

        try:
            self.log(f"[激光开关] ID: {self.laser_stage.get_id()}")
        except Exception as e:
            self.log(f"[激光开关] 读取 ID 失败，但继续使用：{e}")

        try:
            self.log(f"[激光开关] Axes: {self.laser_stage.get_all_axes()}")
        except Exception as e:
            self.log(f"[激光开关] 读取轴信息失败，但继续使用：{e}")

        self.set_laser_velocity_accel(
            axis=int(self.cfg.laser_axis),
            speed=int(self.cfg.laser_speed),
            accel=int(self.cfg.laser_accel),
        )
        self.context["laser_on"] = False
        self.log("[激光开关] Newport 8743-CL 连接成功")

    def set_laser_velocity_accel(self, axis: int, speed: int, accel: int):
        """
        设置 Newport 8743-CL 激光开关轴速度和加速度。
        """
        if self.laser_stage is None:
            raise RuntimeError("激光开关 Newport 控制器未连接")

        axis = int(axis)
        speed = int(speed)
        accel = int(accel)

        cmd_speed = f"VA{speed}"
        self.log(f"[激光开关] SET axis={axis}, cmd={cmd_speed}")
        self.laser_stage.query(cmd_speed, axis=axis)
        time.sleep(0.05)

        cmd_accel = f"AC{accel}"
        self.log(f"[激光开关] SET axis={axis}, cmd={cmd_accel}")
        self.laser_stage.query(cmd_accel, axis=axis)
        time.sleep(0.10)

        self.log(f"[激光开关] OK axis={axis}, speed={speed}, accel={accel}")

    def laser_move_and_wait(self, axis: int, steps: int, wait: bool = True):
        """
        Newport 激光开关轴运动指定步数。
        """
        if self.laser_stage is None:
            raise RuntimeError("激光开关 Newport 控制器未连接")

        axis = int(axis)
        steps = int(steps)

        self.log(f"[激光开关] MOVE axis={axis}, steps={steps}")
        self.laser_stage.move_by(axis=axis, steps=steps)

        if wait:
            try:
                self.laser_stage.wait_move(axis=axis)
            except Exception as e:
                self.log(f"[激光开关] wait_move 失败，短暂等待后继续：{e}")
                time.sleep(0.2)

    def laser_on(self):
        """
        打开激光：替代原单轮流程 Step 6 的 Rigol CH1 ON。

        约定：
            cfg.laser_on_steps 只填写步数幅值。
            打开激光时固定使用正向运动 +abs(laser_on_steps)。
            若 context 中激光已经是 ON 状态，则跳过重复打开，防止子循环中多次调用导致轴过冲。
        """
        if bool(self.context.get("laser_on", False)):
            self.log("[激光开关] 激光已经是 ON 状态，跳过重复打开")
            return

        steps = abs(int(self.cfg.laser_on_steps))

        self.laser_move_and_wait(
            axis=int(self.cfg.laser_axis),
            steps=steps,
            wait=True,
        )
        self.context["laser_on"] = True
        self.context["laser_last_on_steps"] = int(steps)
        self.log(
            f"[激光开关] ON 完成：axis={self.cfg.laser_axis}, "
            f"steps={steps}"
        )
        self.notify_update()

    def laser_off(self):
        """
        关闭激光：替代原单轮流程 Step 8 的 Rigol CH1 OFF。

        约定：
            cfg.laser_off_steps 只填写步数幅值。
            关闭激光必须相对于打开激光反向运动，
            因此固定使用 -abs(laser_off_steps)。

        这样 config.py 中继续写：
            laser_on_steps = 400
            laser_off_steps = 400
        运行日志会显示：
            ON  steps=400
            OFF steps=-400
        """
        steps = -abs(int(self.cfg.laser_off_steps))

        self.laser_move_and_wait(
            axis=int(self.cfg.laser_axis),
            steps=steps,
            wait=True,
        )
        self.context["laser_on"] = False
        self.context["laser_last_off_steps"] = int(steps)
        self.log(
            f"[激光开关] OFF 完成：axis={self.cfg.laser_axis}, "
            f"steps={steps}"
        )
        self.notify_update()

    def connect_signal_generator(self):
        """
        兼容旧代码入口。

        现在完整测量不再使用 Rigol 信号发生器，
        原来的“连接信号发生器”动作改为连接 Newport 激光开关控制器。
        """
        self.log("[激光开关] 当前版本不再连接 Rigol；改为连接 Newport 激光控制器")
        self.connect_laser_controller()

    def configure_signal_generator(self):
        """
        兼容旧代码入口。

        现在不再配置 Rigol CH1/CH2，只重新设置 Newport 激光开关轴速度/加速度。
        """
        self.log("[激光开关] 当前版本不再配置 Rigol；改为设置 Newport 激光轴速度/加速度")
        self.connect_laser_controller()
        self.set_laser_velocity_accel(
            axis=int(self.cfg.laser_axis),
            speed=int(self.cfg.laser_speed),
            accel=int(self.cfg.laser_accel),
        )
        self.context["signal_ch1_on"] = bool(self.context.get("laser_on", False))
        self.context["signal_ch2_on"] = False

    def init_angle_module(self):
        """初始化 YOLO-OBB 角度检测模型。"""
        if self.angle_module is not None:
            self.log("[YOLO-OBB角度] 已初始化")
            return
        if YOLO is None:
            raise RuntimeError("无法导入 ultralytics.YOLO，请先安装：pip install ultralytics")
        if mss is None:
            raise RuntimeError("无法导入 mss，请先安装：pip install mss")
        if not self.cfg.angle_model_path:
            raise ValueError("未设置 angle_model_path，请在 GUI 中选择 YOLO-OBB .pt 模型路径")
        model_path = Path(str(self.cfg.angle_model_path)).expanduser()
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO-OBB模型不存在：{model_path}")
        self.log("[YOLO-OBB角度] 正在初始化 YOLO-OBB 模型")
        self.log(f"[YOLO-OBB角度] model_path={model_path}")
        self.log(f"[YOLO-OBB角度] capture_area={self.cfg.capture_area}")
        self.log(f"[YOLO-OBB角度] conf={getattr(self.cfg, 'angle_conf', 0.5)}, iou={getattr(self.cfg, 'angle_iou', 0.7)}, imgsz={getattr(self.cfg, 'angle_imgsz', 640)}, device={getattr(self.cfg, 'angle_device', '')!r}")
        self.angle_module = YOLO(str(model_path))
        self.context["angle_detector_type"] = "yolo_obb_long_edge"
        self.context["angle_model_path_resolved"] = str(model_path)
        self.log("[YOLO-OBB角度] 初始化完成；Step1/Step7 将使用 OBB 四角长边角度")

    def _create_labview_tcp_server(self):
        """创建光谱仪后端服务器。

        本版本新增设备选择：
            - labview_tcp : 原有 LabVIEW TCP 中间层
            - picam       : PI 真实相机（PICam SDK 直接控制）
            - picam_demo  : PI 软件模拟相机
        """
        backend = str(self.cfg.spectrometer_backend).strip().lower()

        if backend in ("picam", "picam_demo"):
            if PISpectrometerAdapter is None:
                raise RuntimeError(
                    "无法导入 patches.pi_spectrometer_adapter.PISpectrometerAdapter。"
                    "请确认 PrincetonInstruments/Project/patches/ 目录存在且在 PYTHONPATH 中。"
                )
            from pi_spectrometer.core.types import ROI
            roi = ROI(
                x=int(self.cfg.picam_roi_x),
                y=int(self.cfg.picam_roi_y),
                width=int(self.cfg.picam_roi_width),
                height=int(self.cfg.picam_roi_height),
            )
            self.log(f"[光谱仪] 使用 PI 直接控制后端: {backend}")
            return PISpectrometerAdapter(
                backend_type=backend,
                output_dir=str(self.cfg.tcp_output_dir),
                on_log=lambda msg: self.log(msg),
                dll_path=self.cfg.picam_dll_path or None,
                camera_index=int(self.cfg.picam_camera_index),
                exposure=float(self.cfg.picam_exposure),
                sensor_temperature=float(self.cfg.picam_temperature),
                roi=roi,
                # IsoPlane 单色仪参数
                isoplane_dll_path=self.cfg.isoplane_dll_path or None,
                isoplane_device_index=int(self.cfg.isoplane_device_index),
                center_wavelength_nm=float(self.cfg.center_wavelength_nm),
                grating_index=int(self.cfg.grating_index),
                entrance_slit_um=int(self.cfg.entrance_slit_um),
                exit_slit_um=int(self.cfg.exit_slit_um),
            )

        if self._is_virtual_hardware_mode():
            return VirtualLabVIEWTCPServer(
                host=str(self.cfg.tcp_host),
                port=int(self.cfg.tcp_port),
                output_dir=str(self.cfg.tcp_output_dir),
                log_func=self.log,
            )

        if LabVIEWTCPServer is None:
            raise RuntimeError(
                "无法导入 control.labview_tcp_server.LabVIEWTCPServer。"
                "请确认 control/labview_tcp_server.py 存在，并且其中定义了 LabVIEWTCPServer。"
            )

        host = str(self.cfg.tcp_host)
        port = int(self.cfg.tcp_port)
        output_dir = str(self.cfg.tcp_output_dir)

        constructors = [
            lambda: LabVIEWTCPServer(host=host, port=port, output_dir=output_dir),
            lambda: LabVIEWTCPServer(host, port, output_dir),
            lambda: LabVIEWTCPServer(port=port, host=host, output_dir=output_dir),
            lambda: LabVIEWTCPServer(port, output_dir),
            lambda: LabVIEWTCPServer(port=port, output_dir=output_dir),
            lambda: LabVIEWTCPServer(),
        ]

        last_exc: Optional[BaseException] = None
        for make in constructors:
            try:
                server = make()
                # 如果无参构造成功，尽量把配置写进去，兼容外部类用属性读取。
                for attr, value in (
                    ("host", host),
                    ("tcp_host", host),
                    ("port", port),
                    ("tcp_port", port),
                    ("output_dir", output_dir),
                    ("save_dir", output_dir),
                ):
                    try:
                        if not hasattr(server, attr) or getattr(server, attr) in (None, ""):
                            setattr(server, attr, value)
                    except Exception:
                        pass
                return server
            except Exception as e:
                last_exc = e

        raise RuntimeError(f"创建 LabVIEWTCPServer 失败：{last_exc}")

    def _start_labview_tcp_server_compat(self, server: Any):
        """
        兼容启动不同版本的 LabVIEWTCPServer。

        修复点：
            旧代码只认 start/start_server；
            但你的 control.labview_tcp_server.LabVIEWTCPServer 可能把 Server
            在 __init__ 中就启动了，或者使用 run/serve_forever/listen 等方法名。
            因此这里不再因为缺少 start/start_server 就报错。
        """
        start_method_names = (
            # 你的 control/labview_tcp_server.py 当前公开的是 start_server_async / start_server_blocking。
            # 先尝试 async，避免 GUI 被阻塞；没有 async 时再把 blocking 放入守护线程。
            "start_server_async",
            "start_server_blocking",
            "start",
            "start_server",
            "run",
            "serve_forever",
            "serve",
            "listen",
            "open",
        )

        for name in start_method_names:
            method = getattr(server, name, None)
            if not callable(method):
                continue

            def _target():
                try:
                    method()
                except Exception as e:
                    self.context["tcp_started"] = False
                    self.context["labview_ready"] = False
                    self.context["tcp_server_thread_error"] = str(e)
                    self.log(f"[TCP] Server线程异常：{e}")
                    self.log(traceback.format_exc())

            if name == "start_server_async":
                # async 方法本身应该创建监听线程；直接调用，避免套两层线程导致状态不一致。
                try:
                    method()
                except Exception as e:
                    self.context["tcp_started"] = False
                    self.context["labview_ready"] = False
                    self.context["tcp_server_thread_error"] = str(e)
                    raise
                self.context["tcp_server_start_method"] = name
                self.context["tcp_server_thread_name"] = "managed_by_start_server_async"
                self.log(f"[TCP] 已调用 LabVIEWTCPServer.{name}() 启动 Server")
                return

            # blocking/run/serve_forever 等可能阻塞；统一放到守护线程里，避免卡住 GUI。
            t = threading.Thread(target=_target, name=f"LabVIEWTCPServer-{name}", daemon=True)
            t.start()
            self.context["tcp_server_start_method"] = name
            self.context["tcp_server_thread_name"] = t.name
            self.log(f"[TCP] 已调用 LabVIEWTCPServer.{name}() 启动 Server 线程")
            return

        # 关键修复：没有 start/start_server 不再直接失败。
        # 很多自定义 TCP Server 在 __init__ 里已经 bind/listen，或者只提供 request/wait 接口。
        methods = [m for m in dir(server) if not m.startswith("_")]
        self.context["tcp_server_start_method"] = "constructor_or_external"
        self.log(
            "[TCP] LabVIEWTCPServer 未提供 start_server_async/start_server_blocking/start/start_server/"
            "run/serve_forever/listen/open；已保留实例并继续运行。若该类不是构造即启动，"
            "请在 control/labview_tcp_server.py 中增加 start_server_async() 或 start() 方法。"
        )
        self.log(f"[TCP] 当前 LabVIEWTCPServer 可用公开成员：{methods}")

    def start_tcp_server(self):
        """启动 LabVIEW TCP Server；virtual 模式下只创建虚拟 Server，不打开真实端口。"""
        if self._is_virtual_hardware_mode():
            self.log("========== 启动虚拟 LabVIEW TCP Server ==========" )
            if self.tcp_server is None:
                self.tcp_server = self._create_labview_tcp_server()
                self._start_labview_tcp_server_compat(self.tcp_server)
            self.context["tcp_started"] = True
            self.context["labview_ready"] = True
            self.notify_update()
            return {"ok": True, "virtual": True}

        if bool(globals().get("SPECTROMETER_TCP_DISABLED", False)):
            self.tcp_server = None
            self.context["tcp_started"] = False
            self.context["labview_ready"] = True
            self.log("[TCP] SPECTROMETER_TCP_DISABLED=True：跳过真实 LabVIEW TCP Server。")
            self.notify_update()
            return {"ok": True, "disabled": True, "message": "spectrometer_tcp_disabled"}

        self.log("========== 启动 LabVIEW TCP Server ==========")
        self.log(f"[TCP] host={self.cfg.tcp_host}, port={self.cfg.tcp_port}, output_dir={self.cfg.tcp_output_dir}")

        if self.tcp_server is not None:
            self.log("[TCP] Server 已存在，不重复创建")
            self.context["tcp_started"] = True
            self.notify_update()
            return {"ok": True, "already_started": True}

        server = self._create_labview_tcp_server()
        self.tcp_server = server
        self._start_labview_tcp_server_compat(server)

        self.context["tcp_started"] = True
        self.context["labview_ready"] = False
        self.notify_update()
        return {"ok": True, "disabled": False}


    def wait_labview_ready(self):
        """等待 LabVIEW READY；兼容不同版本 server 的 ready 方法。"""
        if self._is_virtual_hardware_mode():
            if self.tcp_server is None:
                self.start_tcp_server()
            self.context["tcp_started"] = True
            self.context["labview_ready"] = True
            self.log("[TCP][virtual] LabVIEW READY：虚拟模式立即就绪")
            self.notify_update()
            return {"ok": True, "virtual": True}

        if bool(globals().get("SPECTROMETER_TCP_DISABLED", False)):
            self.tcp_server = None
            self.context["tcp_started"] = False
            self.context["labview_ready"] = True
            self.log("[TCP] SPECTROMETER_TCP_DISABLED=True：跳过 LabVIEW READY 等待。")
            self.notify_update()
            return {"ok": True, "disabled": True}

        if self.tcp_server is None:
            self.start_tcp_server()

        server = self.tcp_server
        if server is None:
            raise RuntimeError("TCP Server 未创建，无法等待 LabVIEW READY")

        self.log("========== 等待 LabVIEW READY ==========")

        # 先等待 LabVIEW 真实建立 TCP 连接（start_server_async 只是启动监听，
        # 还需等 LabVIEW 客户端连上来）。
        t0_conn = time.time()
        timeout_conn = 60.0
        while True:
            if self.stop_requested:
                raise RuntimeError("stop_requested")
            if getattr(server, "is_connected", False):
                break
            if time.time() - t0_conn > timeout_conn:
                raise RuntimeError(
                    "LabVIEW 在 60 秒内未连接到 TCP Server，"
                    "请检查 LabVIEW 客户端是否已启动并尝试连接 127.0.0.1:65432。"
                )
            time.sleep(0.1)

        # 优先使用外部 Server 已实现的方法。
        for name in ("wait_ready", "wait_for_ready", "wait_labview_ready", "wait_client_ready"):
            method = getattr(server, name, None)
            if callable(method):
                try:
                    result = method()
                except TypeError:
                    result = method(timeout=None)

                # 关键修复：必须确认返回结果表示成功，才把 labview_ready 置为 True。
                if isinstance(result, dict) and not result.get("ok", True):
                    reason = result.get("reason", "unknown")
                    raise RuntimeError(
                        f"LabVIEW 等待 READY 失败（{name}() 返回 ok=False）：{reason}"
                    )

                self.context["labview_ready"] = True
                self.log(f"[TCP] LabVIEW READY：由 {name}() 返回")
                self.notify_update()
                return {"ok": True, "method": name, "result": result}

        # 兼容属性型 READY 标志。
        ready_attrs = ("ready", "is_ready", "labview_ready", "client_ready", "connected", "is_connected")
        t0 = time.time()
        timeout_s = 300.0
        while time.time() - t0 < timeout_s:
            if self.stop_requested:
                raise RuntimeError("stop_requested")
            for attr in ready_attrs:
                try:
                    value = getattr(server, attr, None)
                    value = value() if callable(value) else value
                    if bool(value):
                        self.context["labview_ready"] = True
                        self.log(f"[TCP] LabVIEW READY：检测到 server.{attr}=True")
                        self.notify_update()
                        return {"ok": True, "method": f"attr:{attr}"}
                except Exception:
                    pass
            time.sleep(0.1)

        # 如果外部类没有 READY 接口，不阻塞完整测量；采集时再由 request 报真实错误。
        self.context["labview_ready"] = True
        self.log("[TCP] 未找到 READY 接口，已跳过显式 READY 等待；采集阶段会继续验证 TCP 可用性。")
        self.notify_update()
        return {"ok": True, "method": "no_ready_interface_skip"}


    def signal_ch1_on(self):
        """
        兼容旧名称：原 Rigol CH1 ON 现在等价为“打开激光”。
        """
        self.laser_on()
        self.context["signal_ch1_on"] = True
        self.context["signal_ch2_on"] = False

    def signal_ch1_off(self):
        """
        兼容旧名称：原 Rigol CH1 OFF 现在等价为“关闭激光”。
        """
        self.laser_off()
        self.context["signal_ch1_on"] = False
        self.context["signal_ch2_on"] = False

    def signal_ch2_on(self):
        """
        兼容旧名称：当前不再使用 Rigol CH2。
        为避免旧按钮/旧流程误连 Rigol，这里改为提示并不执行任何硬件动作。
        """
        self.context["signal_ch2_on"] = False
        self.log("[激光开关] 当前版本不再使用 Rigol CH2；该按钮不执行动作")
        self.notify_update()

    def signal_ch2_off(self):
        """
        兼容旧名称：当前不再使用 Rigol CH2。
        """
        self.context["signal_ch2_on"] = False
        self.log("[激光开关] 当前版本不再使用 Rigol CH2；CH2 保持未使用状态")
        self.notify_update()

    def signal_all_on(self):
        """
        兼容旧调用：现在等价为打开激光。
        """
        self.signal_ch1_on()

    def signal_all_off(self):
        """
        兼容旧调用：现在等价为关闭激光。
        """
        self.signal_ch1_off()

    def light_on(self):
        if self.light is None:
            raise RuntimeError("照明光未连接")

        self.light.on()
        self.context["light_on"] = True
        self.log("[照明光] ON")
        self.notify_update()

    def light_off(self):
        if self.light is None:
            raise RuntimeError("照明光未连接")

        self.light.off()
        self.context["light_on"] = False
        self.log("[照明光] OFF")

        time.sleep(self.cfg.stable_wait_ms/1000.0)

        self.notify_update()

    # --------------------------------------------------------
    # 角度检测与信号时间自适应
    # --------------------------------------------------------

    @staticmethod
    def angle_diff_deg(a: float, b: float) -> float:
        """
        0~180 方向角的最小差值。
        """
        diff = abs(float(a) - float(b)) % 180.0
        if diff > 90.0:
            diff = 180.0 - diff
        return float(diff)

    @staticmethod
    def _yolo_obb_normalize_angle(angle_degrees: float) -> float:
        return float(((float(angle_degrees) + 90.0) % 180.0) - 90.0)

    @classmethod
    def _yolo_obb_long_edge_angle(cls, points: np.ndarray) -> float:
        pts = np.asarray(points, dtype=float)
        if pts.shape != (4, 2):
            raise ValueError(f"Expected OBB points shape (4, 2), got {pts.shape}")
        edges = []
        for i in range(4):
            p1 = pts[i]
            p2 = pts[(i + 1) % 4]
            dx = float(p2[0] - p1[0])
            dy = float(p2[1] - p1[1])
            length = math.hypot(dx, dy)
            edges.append((length, dx, dy, p1, p2))
        _, dx, dy, _, _ = max(edges, key=lambda item: item[0])
        return cls._yolo_obb_normalize_angle(math.degrees(math.atan2(dy, dx)))

    @staticmethod
    def _format_yolo_obb_points(points: np.ndarray) -> str:
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        return ";".join(f"{x:.2f},{y:.2f}" for x, y in pts)

    def _yolo_obb_output_dir(self) -> Path:
        base = self.run_session_dir if self.run_session_dir is not None else self.output_root
        override_dir_name = self.context.get("bmask_longest_edge_overlay_dir_override")
        dir_name = str(override_dir_name or getattr(self.cfg, "rule_ab_realtime_overlay_dir_name", "ab_angle_yolo_obb"))
        dir_name = dir_name.replace("bmask_longest_edge", "yolo_obb").replace("Bmask", "YOLO_OBB")
        out = Path(base) / dir_name
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _capture_yolo_obb_frame_bgr(self) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        if mss is None:
            raise RuntimeError("mss 未安装，无法按第二个代码方式进行屏幕截图")
        left, top, width, height = tuple(int(v) for v in self.cfg.capture_area)
        with mss.mss() as sct:
            screen = np.array(sct.grab({"left": left, "top": top, "width": width, "height": height}))
        return cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR), (left, top, width, height)

    def _run_angle_detector_once_raw(self, use_second_detect: bool = False, label: str = "yolo_obb") -> Dict[str, Any]:
        if self.angle_module is None:
            self.init_angle_module()
        if self.angle_module is None:
            raise RuntimeError("YOLO-OBB角度模块未初始化")

        frame_bgr, (left, top, width, height) = self._capture_yolo_obb_frame_bgr()
        timestamp = time.time()
        kwargs = {"source": frame_bgr, "conf": float(getattr(self.cfg, "angle_conf", 0.5)), "iou": float(getattr(self.cfg, "angle_iou", 0.7)), "imgsz": int(getattr(self.cfg, "angle_imgsz", 640)), "verbose": False}
        device = str(getattr(self.cfg, "angle_device", "") or "").strip()
        if device:
            kwargs["device"] = device
        result = self.angle_module.predict(**kwargs)[0]
        out_dir = self._yolo_obb_output_dir()
        safe_label = "".join(ch if (ch.isalnum() or ch in "_-.()") else "_" for ch in str(label))[:90]
        stem = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}_{safe_label}"
        raw_path = out_dir / f"{stem}_raw.png"
        cv2.imwrite(str(raw_path), frame_bgr)
        if result.obb is None or len(result.obb) == 0:
            return {"ok": False, "angle_ok": False, "angle_deg": None, "angle_deg_raw": None, "reason": "yolo_obb_no_detection", "angle_source": "yolo_obb_no_detection", "edge_selection_mode": "yolo_obb_long_edge", "image_path": str(raw_path), "raw_image_path": str(raw_path), "detection_count": 0, "timestamp": timestamp}

        points_array = result.obb.xyxyxyxy.cpu().numpy()
        confs = result.obb.conf.cpu().numpy()
        classes = result.obb.cls.cpu().numpy().astype(int)
        names = getattr(result, "names", {}) or {}
        best_idx = int(np.argmax(confs))
        points = np.asarray(points_array[best_idx], dtype=float).reshape(4, 2)
        confidence = float(confs[best_idx])
        class_id = int(classes[best_idx])
        class_name = str(names.get(class_id, class_id))
        angle = float(self._yolo_obb_long_edge_angle(points))
        center = points.mean(axis=0)
        screen_points = points + np.array([left, top], dtype=float)
        screen_center = center + np.array([left, top], dtype=float)
        edge_items = []
        for i in range(4):
            p1e = points[i]
            p2e = points[(i + 1) % 4]
            edge_items.append((float(np.linalg.norm(p2e - p1e)), p1e, p2e))
        edge_length, p1, p2 = max(edge_items, key=lambda item: item[0])

        overlay_path = out_dir / f"{stem}_YOLO_OBB_overlay.png"
        meta_path = out_dir / f"{stem}_YOLO_OBB_meta.json"
        try:
            annotated = result.plot()
        except Exception:
            annotated = frame_bgr.copy()
        cv2.line(annotated, (int(round(p1[0])), int(round(p1[1]))), (int(round(p2[0])), int(round(p2[1]))), (0, 0, 255), 3, cv2.LINE_AA)
        cv2.putText(annotated, f"{angle:.2f} deg conf={confidence:.3f}", (int(round(center[0])) + 8, int(round(center[1])) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(overlay_path), annotated)
        meta = {"ok": True, "label": label, "method": "yolo_obb_long_edge_from_second_code", "angle_deg": angle, "angle_deg_raw": angle, "confidence": confidence, "class_id": class_id, "class_name": class_name, "center_region": [float(center[0]), float(center[1])], "center_screen": [float(screen_center[0]), float(screen_center[1])], "points_region": points.tolist(), "points_screen": screen_points.tolist(), "edge_length_px": float(edge_length), "capture_area": [int(left), int(top), int(width), int(height)], "detection_count": int(len(points_array)), "timestamp": timestamp, "raw_image_path": str(raw_path), "overlay_image_path": str(overlay_path)}
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(self._json_safe(meta), f, ensure_ascii=False, indent=2)
        return {"ok": True, "angle_ok": True, "angle_deg": angle, "angle_deg_raw": angle, "label": label, "reason": "ok", "angle_source": "yolo_obb_long_edge", "edge_selection_mode": "yolo_obb_long_edge", "allow_close": True, "confidence": confidence, "class_id": class_id, "class_name": class_name, "center": [float(center[0]), float(center[1])], "center_screen": [float(screen_center[0]), float(screen_center[1])], "points_region": self._format_yolo_obb_points(points), "points_screen": self._format_yolo_obb_points(screen_points), "obb_points": points.tolist(), "obb_points_screen": screen_points.tolist(), "endpoints": [[float(p1[0]), float(p1[1])], [float(p2[0]), float(p2[1])]], "selected_b_edge": {"angle_deg": angle, "length_px": float(edge_length), "edge_length_px": float(edge_length), "endpoints": [[float(p1[0]), float(p1[1])], [float(p2[0]), float(p2[1])]], "angle_source": "yolo_obb_long_edge"}, "edge_length_px": float(edge_length), "detection_count": int(len(points_array)), "timestamp": timestamp, "image_path": str(raw_path), "raw_image_path": str(raw_path), "overlay_path": str(overlay_path), "overlay_image_path": str(overlay_path), "meta_path": str(meta_path)}

    @staticmethod
    def _normalize_angle_result(result: Any) -> Dict[str, Any]:
        """
        统一角度检测返回结果，避免后续保存、日志、CSV 逻辑因为缺字段报错。
        """
        if isinstance(result, dict):
            result.setdefault("angle_deg_raw", result.get("angle_deg"))
            result.setdefault("center", None)
            result.setdefault("center_norm", None)
            result.setdefault("bbox", None)
            result.setdefault("bbox_xywhn", None)
            result.setdefault("area_px", None)
            result.setdefault("area", None)
            result.setdefault("stats", {})
            return result

        return {
            "ok": False,
            "angle_deg": None,
            "reason": "angle_module_return_not_dict",
            "error": f"角度检测模块返回类型错误：{type(result)}",
        }

    def _should_run_conditional_second_detect(self, angle: Optional[float]) -> bool:
        """
        判断是否需要触发条件二次检测。

        默认规则：第一次基础检测角度满足
            0 <= angle <= 20 或 90 <= angle <= 110
        时，临时启用 use_second_detect=True 再检测一次。
        """
        if not bool(getattr(self.cfg, "enable_conditional_second_detect", True)):
            return False

        if angle is None:
            return False

        try:
            a = float(angle)
        except Exception:
            return False

        ranges = getattr(
            self.cfg,
            "conditional_second_detect_ranges",
            ((0.0, 20.0), (90.0, 110.0)),
        )

        for lo, hi in ranges:
            if float(lo) <= a <= float(hi):
                return True
        return False

    def detect_angle_once(self, label: str, allow_fail: bool = True) -> Dict[str, Any]:
        """单次 YOLO-OBB 长边角度检测。"""
        self.log(f"========== YOLO-OBB角度检测：{label} ==========")
        try:
            result = self._normalize_angle_result(self._run_angle_detector_once_raw(use_second_detect=False, label=label))
        except Exception as e:
            result = {"ok": False, "angle_ok": False, "angle_deg": None, "angle_deg_raw": None, "reason": f"yolo_obb_angle_exception: {e}", "angle_source": "yolo_obb_failed", "error": str(e)}
            self.log(f"[YOLO-OBB角度-{label}] 检测异常：{e}")
            self.log(traceback.format_exc())
            if not allow_fail:
                raise
        self.context["last_angle_result"] = self._json_safe(result)
        if not result.get("ok", False):
            self.log(f"[YOLO-OBB角度-{label}] 检测失败：reason={result.get('reason')}")
            if not allow_fail:
                raise RuntimeError(f"{label} YOLO-OBB角度检测失败：{result}")
            return result
        self.log(f"[YOLO-OBB角度-{label}] angle={result.get('angle_deg')}, conf={result.get('confidence')}, class={result.get('class_name')}, overlay={result.get('overlay_image_path') or result.get('overlay_path')}")
        return result

    def adjust_signal_time_by_angle_delta(self, angle_delta: float):
        old_ms = float(self.cfg.signal_on_time_ms)
        factor = float(self.cfg.signal_time_factor)

        if factor <= 1.0:
            self.log("[信号时间调节] 倍乘系数 <= 1，保持不变")
            self.context["signal_time_adjust_action"] = "keep_factor_invalid"
            return

        if angle_delta < self.cfg.angle_delta_min_deg:
            new_ms = old_ms * factor
            action = "multiply"
            reason = f"角度差 {angle_delta:.6g} < {self.cfg.angle_delta_min_deg}"
        elif angle_delta > self.cfg.angle_delta_max_deg:
            new_ms = old_ms / factor
            action = "divide"
            reason = f"角度差 {angle_delta:.6g} > {self.cfg.angle_delta_max_deg}"
        else:
            new_ms = old_ms
            action = "keep"
            reason = f"角度差 {angle_delta:.6g} 在 [{self.cfg.angle_delta_min_deg}, {self.cfg.angle_delta_max_deg}] 内"

        self.cfg.signal_on_time_ms = float(new_ms)

        self.context["signal_on_time_next_ms"] = float(new_ms)
        self.context["signal_time_adjust_action"] = action

        self.log(
            f"[信号时间调节] {reason}，动作={action}，"
            f"本轮使用={old_ms:.3f} ms，下一轮={new_ms:.3f} ms，factor={factor}"
        )

        self.notify_update()

    # --------------------------------------------------------
    # Rule AB / Rule AC 模块
    # --------------------------------------------------------


    @staticmethod
    def _is_transient_frame_read_error(exc: BaseException) -> bool:
        """
        判断是否属于截图/图像后端偶发的半帧读取错误。

        典型日志：
            backend exception: 'read returned less data than expected'
        这类错误通常不是 A/B/C 标定错误，而是屏幕截图、mss/PIL、显卡/显示后端在某一帧
        没有返回完整图像。处理方式应该是释放本次 follower，等待下一帧后重试。
        """
        msg = str(exc).lower()
        transient_keywords = (
            "read returned less data than expected",
            "less data than expected",
            "backend exception",
            "imagegrab",
            "grab",
            "mss",
            "screen",
            "capture",
            "bitmap",
            "cannot identify image file",
            "truncated",
            "incomplete",
        )
        return any(k in msg for k in transient_keywords)

    def _reset_rule_ab_follower_after_transient_error(self, reason: str = "") -> None:
        """
        Step1 首帧初始化失败后释放 RuleAB follower，让下一次重试重新取帧、重新初始化 A/B/C。

        注意：
            - 不关闭照明/激光；
            - 只清理 RuleAB follower 与特征跟踪状态；
            - 尽量关闭 follower 内部 stage 连接，避免残留线程/USB状态影响下一次重试。
        """
        follower = getattr(self, "rule_ab_follower", None)
        if follower is not None:
            for method_name in ("close", "shutdown", "release"):
                m = getattr(follower, method_name, None)
                if callable(m):
                    try:
                        m()
                        break
                    except Exception:
                        pass
            try:
                if hasattr(follower, "_abc_feature_tracker_installed"):
                    setattr(follower, "_abc_feature_tracker_installed", False)
                if hasattr(follower, "_abc_feature_tracker_state"):
                    setattr(follower, "_abc_feature_tracker_state", {})
            except Exception:
                pass
        self.rule_ab_follower = None
        self.context.pop("feature_tracker_scene", None)
        self.context.pop("abc_feature_tracker", None)
        if reason:
            self.log(f"[RuleAB启动重试] 已释放本次 RuleAB follower，准备下一帧重试；reason={reason}")

    def _ensure_rule_ab_follower_with_startup_retry(self, label: str = "startup") -> ActualNanoBoundaryFollower:
        """
        Step1 baseline 前的 RuleAB 初始化容错入口。

        原 ensure_rule_ab_follower() 内部会调用 ActualNanoBoundaryFollower 初始化与首帧 A/B/C 建立。
        如果截图后端偶发返回半帧，会抛出：
            backend exception: 'read returned less data than expected'
        旧逻辑会直接停止完整测量；这里改为释放 follower 后等待下一帧重试。
        """
        attempts = max(1, int(getattr(self.cfg, "rule_ab_startup_init_retry_attempts", 5)))
        interval_s = max(0.0, float(getattr(self.cfg, "rule_ab_startup_init_retry_interval_s", 0.8)))
        last_exc: Optional[BaseException] = None

        for attempt in range(1, attempts + 1):
            if self.stop_requested:
                raise RuntimeError("stop_requested")
            try:
                follower = self.ensure_rule_ab_follower()
                if attempt > 1:
                    self.log(f"[RuleAB启动重试] {label}: 第 {attempt}/{attempts} 次初始化成功。")
                return follower
            except Exception as e:
                last_exc = e
                is_transient = self._is_transient_frame_read_error(e)
                self.log(
                    f"[RuleAB启动重试] {label}: 初始化/首帧读取失败 attempt={attempt}/{attempts}: {e}；"
                    f"transient_frame_read_error={is_transient}"
                )
                self._reset_rule_ab_follower_after_transient_error(reason=f"{label}/attempt={attempt}: {e}")
                if attempt < attempts:
                    time.sleep(interval_s)
                # 即使不是半帧错误，也允许按配置重试；多次失败后再抛出，方便应对偶发 CUDA/截图后端异常。

        raise RuntimeError(
            f"{label}: RuleAB 初始化/首帧读取连续失败 {attempts} 次，最后错误：{last_exc}"
        ) from last_exc

    def ensure_rule_ab_follower(self) -> ActualNanoBoundaryFollower:
        if self.rule_ab_follower is not None:
            return self.rule_ab_follower

        # 完整测量中，RuleAB 的 C 来源必须优先来自完整标定包，不能来自 GUI 单独测试区域残留值。
        state = self._get_loaded_calibration_state()
        if state is not None:
            state = self._complete_calibration_state_for_runtime(state)
            resolved_c_dir = self._resolve_static_c_dir_from_state(state)
            if resolved_c_dir:
                self.cfg.rule_ab_static_c_map_dir = str(resolved_c_dir)
                self.cfg.rule_ab_load_static_c_map_if_exists = True
                self.cfg.rule_ab_force_reselect_c_each_run = False
                self.cfg.rule_ab_reuse_static_c_map_in_memory = True
            self.cfg.rule_ab_confirm_first_frame_segmentation = False
            self.cfg.rule_ab_enable_stage = False if self._is_virtual_hardware_mode() else True
            self._force_cfg_to_strict_full_calibration_c(reason="ensure_rule_ab_follower")

        cfg = RuleABRuntimeConfig(
            capture_area=self.cfg.capture_area,
            output_dir=str(self.output_root / "rule_ab_from_measurement"),
            sam2_device="cuda",
            enable_stage=(False if self._is_virtual_hardware_mode() else bool(self.cfg.rule_ab_enable_stage)),
            # 完整测量不再用 rule_ab_max_steps=15 作为 Step7 结束条件；
            # Step7 只由角度变化是否进入目标范围、是否超上限、用户停止或 RuleAB 返回 False 决定。
            max_cycles=10_000_000,
            loop_interval_s=float(self.cfg.rule_ab_loop_interval_s),

            # 完整标定加载后不再弹第一帧确认窗口；直接使用标定包中的 A/B/C 点和 static C。
            confirm_first_frame_segmentation=bool(self.cfg.rule_ab_confirm_first_frame_segmentation),
            confirm_window_scale=float(self.cfg.rule_ab_confirm_window_scale),
            c_contour_mode=str(self.cfg.rule_ab_c_contour_mode),
            c_quadrilateral_method=str(self.cfg.rule_ab_c_quadrilateral_method),

            static_c_map_name=str(self.cfg.rule_ab_static_c_map_name),
            static_c_map_dir=str(self.cfg.rule_ab_static_c_map_dir),
            reuse_static_c_map_in_memory=bool(self.cfg.rule_ab_reuse_static_c_map_in_memory),
            load_static_c_map_if_exists=bool(self.cfg.rule_ab_load_static_c_map_if_exists),
            save_static_c_map_library=bool(self.cfg.rule_ab_save_static_c_map_library),
            force_reselect_c_each_run=bool(self.cfg.rule_ab_force_reselect_c_each_run),

            # A 后续帧使用“最像上一帧 A”的候选 mask，而不是只选 SAM2 最高分。
            a_use_temporal_mask_selection=False if bool(getattr(self.cfg, "rule_ab_use_feature_tracker_after_sam2_init", True)) else bool(self.cfg.rule_ab_a_use_temporal_mask_selection),
            a_temporal_score_weight=float(self.cfg.rule_ab_a_temporal_score_weight),
            a_temporal_iou_weight=float(self.cfg.rule_ab_a_temporal_iou_weight),
            a_temporal_shape_iou_weight=float(self.cfg.rule_ab_a_temporal_shape_iou_weight),
            a_temporal_area_weight=float(self.cfg.rule_ab_a_temporal_area_weight),
            a_temporal_center_weight=float(self.cfg.rule_ab_a_temporal_center_weight),
            a_temporal_aspect_weight=float(self.cfg.rule_ab_a_temporal_aspect_weight),
            a_temporal_bbox_size_weight=float(self.cfg.rule_ab_a_temporal_bbox_size_weight),
            a_temporal_c_leak_weight=float(self.cfg.rule_ab_a_temporal_c_leak_weight),
            a_temporal_center_sigma_px=float(self.cfg.rule_ab_a_temporal_center_sigma_px),
            a_temporal_area_ratio_min=float(self.cfg.rule_ab_a_temporal_area_ratio_min),
            a_temporal_area_ratio_max=float(self.cfg.rule_ab_a_temporal_area_ratio_max),
            a_temporal_min_shape_iou=float(self.cfg.rule_ab_a_temporal_min_shape_iou),
            a_temporal_min_aspect_similarity=float(self.cfg.rule_ab_a_temporal_min_aspect_similarity),
            a_temporal_min_bbox_size_similarity=float(self.cfg.rule_ab_a_temporal_min_bbox_size_similarity),
            a_temporal_max_c_unique_leak_ratio=float(self.cfg.rule_ab_a_temporal_max_c_unique_leak_ratio),
            a_temporal_fallback_to_last_if_bad=bool(self.cfg.rule_ab_a_temporal_fallback_to_last_if_bad),
            a_temporal_min_accept_score=float(self.cfg.rule_ab_a_temporal_min_accept_score),

            # B 后续帧使用“最像上一帧 B”的候选 mask，而不是只选 SAM2 最高分。
            b_use_temporal_mask_selection=False if bool(getattr(self.cfg, "rule_ab_use_feature_tracker_after_sam2_init", True)) else bool(self.cfg.rule_ab_b_use_temporal_mask_selection),
            b_temporal_score_weight=float(self.cfg.rule_ab_b_temporal_score_weight),
            b_temporal_iou_weight=float(self.cfg.rule_ab_b_temporal_iou_weight),
            b_temporal_shape_iou_weight=float(self.cfg.rule_ab_b_temporal_shape_iou_weight),
            b_temporal_area_weight=float(self.cfg.rule_ab_b_temporal_area_weight),
            b_temporal_center_weight=float(self.cfg.rule_ab_b_temporal_center_weight),
            b_temporal_aspect_weight=float(self.cfg.rule_ab_b_temporal_aspect_weight),
            b_temporal_bbox_size_weight=float(self.cfg.rule_ab_b_temporal_bbox_size_weight),
            b_temporal_c_leak_weight=float(self.cfg.rule_ab_b_temporal_c_leak_weight),
            b_temporal_center_sigma_px=float(self.cfg.rule_ab_b_temporal_center_sigma_px),
            b_temporal_area_ratio_min=float(self.cfg.rule_ab_b_temporal_area_ratio_min),
            b_temporal_area_ratio_max=float(self.cfg.rule_ab_b_temporal_area_ratio_max),
            b_temporal_min_shape_iou=float(self.cfg.rule_ab_b_temporal_min_shape_iou),
            b_temporal_min_aspect_similarity=float(self.cfg.rule_ab_b_temporal_min_aspect_similarity),
            b_temporal_min_bbox_size_similarity=float(self.cfg.rule_ab_b_temporal_min_bbox_size_similarity),
            b_temporal_max_c_unique_leak_ratio=float(self.cfg.rule_ab_b_temporal_max_c_unique_leak_ratio),
            b_temporal_fallback_to_last_if_bad=bool(self.cfg.rule_ab_b_temporal_fallback_to_last_if_bad),
            b_temporal_min_accept_score=float(self.cfg.rule_ab_b_temporal_min_accept_score),
            temporal_search_margin_px=float(self.cfg.rule_ab_temporal_search_margin_px),
            temporal_use_center_point_only=bool(self.cfg.rule_ab_temporal_use_center_point_only),
            temporal_position_match_enable=bool(self.cfg.rule_ab_temporal_position_match_enable),
            temporal_position_match_margin_px=float(self.cfg.rule_ab_temporal_position_match_margin_px),
            temporal_position_match_min_score=float(self.cfg.rule_ab_temporal_position_match_min_score),
            temporal_position_match_use_last_image_template=bool(self.cfg.rule_ab_temporal_position_match_use_last_image_template),
            temporal_stop_on_track_fail=bool(self.cfg.rule_ab_temporal_stop_on_track_fail),
            use_sam2_video_tracking=False if bool(getattr(self.cfg, "rule_ab_use_feature_tracker_after_sam2_init", True)) else bool(self.cfg.rule_ab_use_sam2_video_tracking),
            sam2_video_tracking_mode=str(self.cfg.rule_ab_sam2_video_tracking_mode),
            sam2_video_temp_dir=str(self.cfg.rule_ab_sam2_video_temp_dir),
            sam2_video_prompt_max_points=int(self.cfg.rule_ab_sam2_video_prompt_max_points),
            sam2_video_area_ratio_min=float(self.cfg.rule_ab_sam2_video_area_ratio_min),
            sam2_video_area_ratio_max=float(self.cfg.rule_ab_sam2_video_area_ratio_max),
            sam2_video_center_jump_max_px=float(self.cfg.rule_ab_sam2_video_center_jump_max_px),
            sam2_video_fallback_to_template=bool(self.cfg.rule_ab_sam2_video_fallback_to_template),

            # 新 RuleAB：沿 C 边界绕行并维持 A-C 间隙；A/B 覆盖不再强制停止。
            a_c_target_clearance=float(self.cfg.rule_ab_ac_target_clearance_px),
            a_c_min_clearance=float(self.cfg.rule_ab_ac_min_clearance_px),
            a_c_max_clearance=float(self.cfg.rule_ab_ac_max_clearance_px),
            a_c_correction_mode=str(self.cfg.rule_ab_ac_correction_mode),
            follow_c_direction=int(self.cfg.rule_ab_follow_c_direction),
            a_b_overlap_stop=False,
            a_b_overlap_min_area_px=float(self.cfg.rule_ab_ab_overlap_min_area_px),

            # Thorlabs KinesisPiezoMotor 控制器与通道参数。
            # 这里显式传入，避免 logic/rule_ab.py 使用旧默认序列号。
            stage_conn=str(self.cfg.rule_ab_stage_conn),
            stage_x_channel=int(self.cfg.rule_ab_stage_x_channel),
            stage_y_channel=int(self.cfg.rule_ab_stage_y_channel),
            stage_default_velocity=int(self.cfg.rule_ab_stage_velocity),
            stage_default_acceleration=int(self.cfg.rule_ab_stage_acceleration),
            stage_default_max_voltage=int(self.cfg.rule_ab_stage_max_voltage),
            stage_ch3_velocity=int(self.cfg.rule_ab_stage_ch3_velocity),
            stage_ch3_acceleration=int(self.cfg.rule_ab_stage_ch3_acceleration),
            stage_ch3_max_voltage=int(self.cfg.rule_ab_stage_ch3_max_voltage),
            stage_ch4_velocity=int(self.cfg.rule_ab_stage_ch4_velocity),
            stage_ch4_acceleration=int(self.cfg.rule_ab_stage_ch4_acceleration),
            stage_ch4_max_voltage=int(self.cfg.rule_ab_stage_ch4_max_voltage),
            stage_step_x=int(self.cfg.rule_ab_stage_step_x),
            stage_step_y=int(self.cfg.rule_ab_stage_step_y),
            action_step=int(self.cfg.rule_ab_action_step),
            stage_ch3_pause_after_move_s=float(self.cfg.rule_ab_stage_ch3_pause_after_move_s),
            stage_ch4_pause_after_move_s=float(self.cfg.rule_ab_stage_ch4_pause_after_move_s),
            stage_x_sign=int(self.cfg.rule_ab_stage_x_sign),
            stage_y_sign=int(self.cfg.rule_ab_stage_y_sign),
            exclude_roi_polygons=(state.exclude_roi_polygons if state is not None else []),
        )
        # 如果已经加载完整标定，强制关闭 RuleAB 的交互式第一帧确认和 C 重选。
        if state is not None:
            try:
                cfg.confirm_first_frame_segmentation = False
                cfg.force_reselect_c_each_run = False
                cfg.load_static_c_map_if_exists = True
                cfg.reuse_static_c_map_in_memory = True
                resolved_c_dir = self._resolve_static_c_dir_from_state(state)
                if resolved_c_dir:
                    cfg.static_c_map_dir = resolved_c_dir
            except Exception:
                pass

        self.log(
            "[RuleAB] 初始化 A推动B 模块："
            f"enable_stage={cfg.enable_stage}, max_steps={cfg.max_cycles}, "
            f"full_calibration_loaded={state is not None}, "
            f"confirm_first_frame_segmentation={getattr(cfg, 'confirm_first_frame_segmentation', None)}, "
            f"static_c_map_dir={getattr(cfg, 'static_c_map_dir', '')}"
        )
        self.rule_ab_follower = ActualNanoBoundaryFollower(cfg)
        if self._is_virtual_hardware_mode():
            try:
                self.rule_ab_follower.stage = VirtualKinesisPiezoMotor(
                    serial=str(self.cfg.rule_ab_stage_conn),
                    log_func=self.log,
                )
                self.log("[RuleAB][virtual] 已把 follower.stage 替换为虚拟 Stage34；后续硬件动作只写日志。")
            except Exception as e:
                self.log(f"[RuleAB][virtual] 注入虚拟 Stage34 失败：{e}")

        if state is not None:
            self._call_initialize_abc_with_loaded_calibration(self.rule_ab_follower, state)
        else:
            self.rule_ab_follower.initialize_abc_with_first_frame()

        # 新方案：SAM2 只负责初始化分割；后续完整循环测量统一走颜色 + 形状特征跟踪。
        self._install_feature_tracker_on_rule_ab_follower(self.rule_ab_follower, state)

        return self.rule_ab_follower

    def _setup_stage34_channels_from_current_cfg(self, follower: Any, reason: str = ""):
        """
        根据当前 cfg 重新下发 CH3/CH4 速度、加速度和最大电压。

        作用：
            运行完整循环测量时，用户修改左侧 RuleAB/RuleAC 面板里的
            CH3/CH4 速度、加速度、最大电压后，不需要停止程序；
            下一次 RuleAB 动作前会尝试把新参数写到底层控制器。
        """
        if follower is None:
            return
        stage = getattr(follower, "stage", None)
        if stage is None:
            return

        vals = {
            "ch3_velocity": int(getattr(self.cfg, "rule_ab_stage_ch3_velocity", 10)),
            "ch3_acceleration": int(getattr(self.cfg, "rule_ab_stage_ch3_acceleration", 10)),
            "ch3_max_voltage": int(getattr(self.cfg, "rule_ab_stage_ch3_max_voltage", 50)),
            "ch4_velocity": int(getattr(self.cfg, "rule_ab_stage_ch4_velocity", 10)),
            "ch4_acceleration": int(getattr(self.cfg, "rule_ab_stage_ch4_acceleration", 10)),
            "ch4_max_voltage": int(getattr(self.cfg, "rule_ab_stage_ch4_max_voltage", 50)),
        }
        sig = tuple(vals[k] for k in sorted(vals.keys()))
        old_sig = getattr(follower, "_measurement_stage34_setup_signature", None)
        if old_sig == sig:
            return

        try:
            if hasattr(stage, "setup_channel"):
                stage.setup_channel(
                    channel=3,
                    max_voltage=vals["ch3_max_voltage"],
                    velocity=vals["ch3_velocity"],
                    acceleration=vals["ch3_acceleration"],
                )
                stage.setup_channel(
                    channel=4,
                    max_voltage=vals["ch4_max_voltage"],
                    velocity=vals["ch4_velocity"],
                    acceleration=vals["ch4_acceleration"],
                )
            else:
                dev = getattr(stage, "dev", None)
                if dev is not None and hasattr(dev, "setup_drive"):
                    dev.setup_drive(
                        max_voltage=vals["ch3_max_voltage"],
                        velocity=vals["ch3_velocity"],
                        acceleration=vals["ch3_acceleration"],
                        channel=3,
                    )
                    dev.setup_drive(
                        max_voltage=vals["ch4_max_voltage"],
                        velocity=vals["ch4_velocity"],
                        acceleration=vals["ch4_acceleration"],
                        channel=4,
                    )
            setattr(follower, "_measurement_stage34_setup_signature", sig)
            self.log(
                f"[RuleAB][运行中参数更新] 已下发 Stage34 参数："
                f"CH3(v={vals['ch3_velocity']},a={vals['ch3_acceleration']},V={vals['ch3_max_voltage']}), "
                f"CH4(v={vals['ch4_velocity']},a={vals['ch4_acceleration']},V={vals['ch4_max_voltage']}); reason={reason}"
            )
        except Exception as e:
            self.log(f"[RuleAB][运行中参数更新] 下发 Stage34 参数失败：{e}")

    def apply_runtime_rule_ab_params_to_follower(self, follower: Optional[Any] = None, reason: str = ""):
        """
        把当前 self.cfg 中的 RuleAB 运行参数写入已经初始化好的 follower。

        这解决的问题是：完整测量正在运行 Step7 时，用户在 GUI 左侧第 8 区域
        修改 A-C 距离、CH3/CH4 速度/加速度/步数/停顿、角度目标范围等参数，
        程序应在下一步立即按新参数执行，而不是必须停止后重新运行。
        """
        follower = follower or self.rule_ab_follower
        if follower is None:
            return
        fc = getattr(follower, "cfg", None)
        if fc is None:
            return

        feature_tracker_enabled = bool(getattr(self.cfg, "rule_ab_use_feature_tracker_after_sam2_init", True))

        # 完整测量时仍然锁定完整标定 C，防止 GUI 单独测试区旧 C 污染。
        if bool(self.context.get("strict_full_calibration_c_locked", False)):
            c_dir = self._force_cfg_to_strict_full_calibration_c(reason=f"runtime_rule_ab_params/{reason}")
        else:
            c_dir = str(getattr(self.cfg, "rule_ab_static_c_map_dir", ""))

        updates = {
            # 真动开关和循环节奏
            "enable_stage": bool(getattr(self.cfg, "rule_ab_enable_stage", True)),
            "loop_interval_s": float(getattr(self.cfg, "rule_ab_loop_interval_s", 0.15)),

            # A-C / A-B 规则参数
            "a_c_target_clearance": float(getattr(self.cfg, "rule_ab_ac_target_clearance_px", 90.0)),
            "a_c_min_clearance": float(getattr(self.cfg, "rule_ab_ac_min_clearance_px", 80.0)),
            "a_c_max_clearance": float(getattr(self.cfg, "rule_ab_ac_max_clearance_px", 100.0)),
            "a_c_correction_mode": str(getattr(self.cfg, "rule_ab_ac_correction_mode", "nearest_normal")),
            "follow_c_direction": int(getattr(self.cfg, "rule_ab_follow_c_direction", 1)),
            "a_b_overlap_stop": False,
            "a_b_overlap_min_area_px": float(getattr(self.cfg, "rule_ab_ab_overlap_min_area_px", 1.0)),

            # Stage34 参数
            "stage_x_channel": int(getattr(self.cfg, "rule_ab_stage_x_channel", 3)),
            "stage_y_channel": int(getattr(self.cfg, "rule_ab_stage_y_channel", 4)),
            "stage_default_velocity": int(getattr(self.cfg, "rule_ab_stage_velocity", 10)),
            "stage_default_acceleration": int(getattr(self.cfg, "rule_ab_stage_acceleration", 10)),
            "stage_default_max_voltage": int(getattr(self.cfg, "rule_ab_stage_max_voltage", 100)),
            "stage_ch3_velocity": int(getattr(self.cfg, "rule_ab_stage_ch3_velocity", 10)),
            "stage_ch3_acceleration": int(getattr(self.cfg, "rule_ab_stage_ch3_acceleration", 10)),
            "stage_ch3_max_voltage": int(getattr(self.cfg, "rule_ab_stage_ch3_max_voltage", 50)),
            "stage_ch4_velocity": int(getattr(self.cfg, "rule_ab_stage_ch4_velocity", 10)),
            "stage_ch4_acceleration": int(getattr(self.cfg, "rule_ab_stage_ch4_acceleration", 10)),
            "stage_ch4_max_voltage": int(getattr(self.cfg, "rule_ab_stage_ch4_max_voltage", 50)),
            "stage_step_x": int(getattr(self.cfg, "rule_ab_stage_step_x", 20)),
            "stage_step_y": int(getattr(self.cfg, "rule_ab_stage_step_y", 20)),
            "action_step": int(getattr(self.cfg, "rule_ab_action_step", 20)),
            "stage_ch3_pause_after_move_s": float(getattr(self.cfg, "rule_ab_stage_ch3_pause_after_move_s", 2.0)),
            "stage_ch4_pause_after_move_s": float(getattr(self.cfg, "rule_ab_stage_ch4_pause_after_move_s", 2.0)),
            "stage_x_sign": int(getattr(self.cfg, "rule_ab_stage_x_sign", -1)),
            "stage_y_sign": int(getattr(self.cfg, "rule_ab_stage_y_sign", 1)),

            # C 相关：完整测量时 c_dir 是运行专用 C；单独测试时是 GUI 当前 C。
            "static_c_map_dir": c_dir,
            "load_static_c_map_if_exists": bool(getattr(self.cfg, "rule_ab_load_static_c_map_if_exists", True)),
            "force_reselect_c_each_run": bool(getattr(self.cfg, "rule_ab_force_reselect_c_each_run", False)),
            "reuse_static_c_map_in_memory": bool(getattr(self.cfg, "rule_ab_reuse_static_c_map_in_memory", True)),
        }

        for k, v in updates.items():
            self._safe_setattr(fc, k, v)
            self._safe_setattr(follower, k, v)

        # A/B 跟踪和 SAM2 video tracking 运行参数。
        tracking_map = {
            "temporal_search_margin_px": "rule_ab_temporal_search_margin_px",
            "temporal_use_center_point_only": "rule_ab_temporal_use_center_point_only",
            "temporal_position_match_enable": "rule_ab_temporal_position_match_enable",
            "temporal_position_match_margin_px": "rule_ab_temporal_position_match_margin_px",
            "temporal_position_match_min_score": "rule_ab_temporal_position_match_min_score",
            "temporal_position_match_use_last_image_template": "rule_ab_temporal_position_match_use_last_image_template",
            "temporal_stop_on_track_fail": "rule_ab_temporal_stop_on_track_fail",
            "use_sam2_video_tracking": "rule_ab_use_sam2_video_tracking",
            "sam2_video_tracking_mode": "rule_ab_sam2_video_tracking_mode",
            "sam2_video_temp_dir": "rule_ab_sam2_video_temp_dir",
            "sam2_video_prompt_max_points": "rule_ab_sam2_video_prompt_max_points",
            "sam2_video_area_ratio_min": "rule_ab_sam2_video_area_ratio_min",
            "sam2_video_area_ratio_max": "rule_ab_sam2_video_area_ratio_max",
            "sam2_video_center_jump_max_px": "rule_ab_sam2_video_center_jump_max_px",
            "sam2_video_fallback_to_template": "rule_ab_sam2_video_fallback_to_template",
        }
        seg = getattr(follower, "abc_segmenter", None) or getattr(follower, "segmenter", None)
        for target_attr, cfg_attr in tracking_map.items():
            if not hasattr(self.cfg, cfg_attr):
                continue
            value = getattr(self.cfg, cfg_attr)
            if feature_tracker_enabled:
                # 完整循环测量启用“首帧 SAM2 建模板 + 颜色/形状跟踪”后，
                # 旧的 temporal mask selection / SAM2 video tracking 只能用于首帧初始化，
                # 不能在 Step7、角度检测、overlay 保存时重新接管 A/B。
                if target_attr in (
                    "temporal_position_match_enable",
                    "temporal_position_match_use_last_image_template",
                    "temporal_stop_on_track_fail",
                    "use_sam2_video_tracking",
                    "sam2_video_fallback_to_template",
                ):
                    value = False
            if target_attr == "sam2_video_temp_dir":
                value = Path(str(value))
            self._safe_setattr(fc, target_attr, value)
            if seg is not None:
                self._safe_setattr(seg, target_attr, value)

        if feature_tracker_enabled:
            forced_tracking_off = {
                "a_use_temporal_mask_selection": False,
                "b_use_temporal_mask_selection": False,
                "use_sam2_video_tracking": False,
                "temporal_position_match_enable": False,
                "temporal_position_match_use_last_image_template": False,
                "temporal_stop_on_track_fail": False,
                "sam2_video_fallback_to_template": False,
            }
            for k, v in forced_tracking_off.items():
                self._safe_setattr(fc, k, v)
                self._safe_setattr(follower, k, v)
                if seg is not None:
                    self._safe_setattr(seg, k, v)
            # 防止外部模块保存 overlay 时仍把 A/B 标成 SAM2。
            self._safe_setattr(follower, "abc_identity_source", "feature_tracker")
            self._safe_setattr(follower, "a_source", "feature_tracker")
            self._safe_setattr(follower, "b_source", "feature_tracker")

        self._setup_stage34_channels_from_current_cfg(follower, reason=reason)

    @staticmethod
    def _normalize_angle_0_180(angle: float) -> float:
        """把任意方向角归一化到 [0, 180)。"""
        a = float(angle) % 180.0
        if a < 0:
            a += 180.0
        return float(a)

    def _bmask_edge_angle_candidates_from_mask(
        self,
        mask: np.ndarray,
        min_edge_length_px: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        从 B mask 的外轮廓计算所有候选边角度。

        返回列表元素：
            {
                "angle_deg": 0~180方向角,
                "length_px": 边长,
                "p1": (x1,y1),
                "p2": (x2,y2)
            }
        """
        if mask is None:
            return []

        m = np.asarray(mask).astype(bool)
        if m.size <= 0 or not np.any(m):
            return []

        if min_edge_length_px is None:
            min_edge_length_px = 8.0
        min_len = max(1.0, float(min_edge_length_px))

        m_u8 = (m.astype(np.uint8) * 255)
        contours = self._find_contours_compat(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        contour = max(contours, key=cv2.contourArea)
        if contour is None or len(contour) < 2:
            return []

        peri = float(cv2.arcLength(contour, True))
        candidates_pts = []

        # 先尝试多边形近似；如果点太少或退化，则回退到凸包/原轮廓抽样。
        for eps_ratio in (0.006, 0.010, 0.015, 0.025, 0.040):
            eps = max(1.0, eps_ratio * peri)
            approx = cv2.approxPolyDP(contour, eps, True)
            if approx is not None and len(approx) >= 3:
                candidates_pts = approx.reshape(-1, 2).astype(np.float32).tolist()
                break

        if len(candidates_pts) < 3:
            hull = cv2.convexHull(contour)
            if hull is not None and len(hull) >= 3:
                candidates_pts = hull.reshape(-1, 2).astype(np.float32).tolist()

        if len(candidates_pts) < 3:
            pts = contour.reshape(-1, 2).astype(np.float32)
            if len(pts) >= 3:
                stride = max(1, len(pts) // 20)
                candidates_pts = pts[::stride].tolist()

        if len(candidates_pts) < 2:
            return []

        out: List[Dict[str, Any]] = []
        n = len(candidates_pts)
        for i in range(n):
            x1, y1 = candidates_pts[i]
            x2, y2 = candidates_pts[(i + 1) % n]
            dx = float(x2) - float(x1)
            dy = float(y2) - float(y1)
            length = math.hypot(dx, dy)
            if length < min_len:
                continue
            angle = math.degrees(math.atan2(dy, dx))
            angle = self._normalize_angle_0_180(angle)
            out.append({
                "angle_deg": float(angle),
                "length_px": float(length),
                "p1": (float(x1), float(y1)),
                "p2": (float(x2), float(y2)),
            })

        # 去掉非常接近的重复角度，保留较长边。
        out.sort(key=lambda d: float(d.get("length_px", 0.0)), reverse=True)
        unique: List[Dict[str, Any]] = []
        for item in out:
            a = float(item["angle_deg"])
            if all(self.angle_diff_deg(a, float(u["angle_deg"])) > 1.0 for u in unique):
                unique.append(item)
        unique.sort(key=lambda d: float(d.get("angle_deg", 0.0)))
        return unique

    @staticmethod
    def _distance_point_to_segment_xy(
        point_xy: Tuple[float, float],
        p1_xy: Tuple[float, float],
        p2_xy: Tuple[float, float],
    ) -> Tuple[float, Tuple[float, float]]:
        """计算点到线段的最短距离，并返回线段上的最近点。"""
        px, py = float(point_xy[0]), float(point_xy[1])
        x1, y1 = float(p1_xy[0]), float(p1_xy[1])
        x2, y2 = float(p2_xy[0]), float(p2_xy[1])
        vx, vy = x2 - x1, y2 - y1
        wx, wy = px - x1, py - y1
        vv = vx * vx + vy * vy
        if vv <= 1e-12:
            return float(math.hypot(px - x1, py - y1)), (x1, y1)
        t = (wx * vx + wy * vy) / vv
        t = max(0.0, min(1.0, float(t)))
        qx = x1 + t * vx
        qy = y1 + t * vy
        return float(math.hypot(px - qx, py - qy)), (float(qx), float(qy))

    def _select_b_edge_nearest_to_a_center(
        self,
        candidates: List[Dict[str, Any]],
        a_center_xy: Tuple[float, float],
    ) -> Optional[Dict[str, Any]]:
        """
        从 B mask 的所有候选边中选择 A 中心距离最近的那一条边。

        这是 Step1/Step7 新的角度定义：
            角度边 = 当前 A 中心到 B 轮廓候选边距离最短的那条边。
        不再固定使用 GUI num 指定的旧边；当前 Step1/Step7 只使用 YOLO-OBB 长边角度。
        """
        if not candidates or a_center_xy is None:
            return None
        best: Optional[Dict[str, Any]] = None
        for c in candidates:
            try:
                p1 = c.get("p1")
                p2 = c.get("p2")
                if p1 is None or p2 is None:
                    continue
                dist, nearest = self._distance_point_to_segment_xy(a_center_xy, p1, p2)
                item = dict(c)
                item["distance_to_a_center_px"] = float(dist)
                item["nearest_point_on_edge_xy"] = [float(nearest[0]), float(nearest[1])]
                if best is None:
                    best = item
                    continue
                # 主排序：A中心到边的距离越小越好；辅排序：边越长越稳定。
                if float(item["distance_to_a_center_px"]) < float(best["distance_to_a_center_px"]) - 1e-6:
                    best = item
                elif abs(float(item["distance_to_a_center_px"]) - float(best["distance_to_a_center_px"])) <= 1e-6:
                    if float(item.get("length_px", 0.0)) > float(best.get("length_px", 0.0)):
                        best = item
            except Exception:
                continue
        return best

    def _save_ab_nearest_edge_angle_debug(
        self,
        image_rgb: np.ndarray,
        a_mask: Optional[np.ndarray],
        b_mask: Optional[np.ndarray],
        a_center: Optional[Tuple[float, float]],
        b_center: Optional[Tuple[float, float]],
        candidates: List[Dict[str, Any]],
        selected_edge: Optional[Dict[str, Any]],
        label: str,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """保存每次 A/B 分割结果和“离 A 中心最近的 B 边”角度调试图。"""
        saved: Dict[str, str] = {}
        try:
            if self.run_session_dir is not None:
                out_dir = Path(self.run_session_dir) / "ab_angle_nearest_edge"
            else:
                out_dir = self.output_root / "ab_angle_nearest_edge" / datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir.mkdir(parents=True, exist_ok=True)

            safe_label = "".join(ch if (ch.isalnum() or ch in "_-.") else "_" for ch in str(label))[:80]
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            stem = f"{ts}_{safe_label}"

            img = np.asarray(image_rgb).copy()
            if img.ndim == 2:
                canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                canvas = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if a_mask is not None:
                a_bool = np.asarray(a_mask).astype(bool)
                overlay = canvas.copy()
                overlay[a_bool] = (0, 255, 255)  # A: yellow
                canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)
                p_a = out_dir / f"{stem}_A_mask.png"
                cv2.imwrite(str(p_a), a_bool.astype(np.uint8) * 255)
                saved["a_mask_path"] = str(p_a)

            if b_mask is not None:
                b_bool = np.asarray(b_mask).astype(bool)
                overlay = canvas.copy()
                overlay[b_bool] = (0, 128, 255)  # B: orange
                canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)
                p_b = out_dir / f"{stem}_B_mask.png"
                cv2.imwrite(str(p_b), b_bool.astype(np.uint8) * 255)
                saved["b_mask_path"] = str(p_b)

            # 画所有候选边，选中边加粗。
            for c in candidates:
                try:
                    x1, y1 = c["p1"]
                    x2, y2 = c["p2"]
                    cv2.line(canvas, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), (180, 180, 180), 1, cv2.LINE_AA)
                except Exception:
                    pass

            if selected_edge is not None:
                try:
                    x1, y1 = selected_edge["p1"]
                    x2, y2 = selected_edge["p2"]
                    cv2.line(canvas, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), (255, 0, 255), 3, cv2.LINE_AA)
                    q = selected_edge.get("nearest_point_on_edge_xy")
                    if q is not None:
                        qx, qy = float(q[0]), float(q[1])
                        cv2.circle(canvas, (int(round(qx)), int(round(qy))), 5, (255, 0, 255), -1, cv2.LINE_AA)
                        if a_center is not None:
                            cv2.line(canvas, (int(round(a_center[0])), int(round(a_center[1]))), (int(round(qx)), int(round(qy))), (255, 0, 255), 1, cv2.LINE_AA)
                except Exception:
                    pass

            if a_center is not None:
                ax, ay = int(round(a_center[0])), int(round(a_center[1]))
                cv2.circle(canvas, (ax, ay), 6, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.putText(canvas, "A center", (ax + 8, ay - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
            if b_center is not None:
                bx, by = int(round(b_center[0])), int(round(b_center[1]))
                cv2.circle(canvas, (bx, by), 5, (0, 128, 255), -1, cv2.LINE_AA)
                cv2.putText(canvas, "B center", (bx + 8, by - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 128, 255), 2, cv2.LINE_AA)

            if selected_edge is not None:
                text = (
                    f"angle={float(selected_edge.get('angle_deg', 0.0)):.3f} deg, "
                    f"A-edge={float(selected_edge.get('distance_to_a_center_px', 0.0)):.2f}px"
                )
                cv2.putText(canvas, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(canvas, "selected B edge = nearest to feature_tracker_scene.a center", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

            try:
                src_txt = str((extra_info or {}).get("scene_source") or (extra_info or {}).get("source") or self.context.get("abc_feature_tracker_last_source") or "feature_tracker")
                cv2.putText(canvas, f"A/B source={src_txt}", (12, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            except Exception:
                pass

            p_overlay = out_dir / f"{stem}_AB_nearest_B_edge_overlay.png"
            cv2.imwrite(str(p_overlay), canvas)
            saved["overlay_path"] = str(p_overlay)

            meta = {
                "label": label,
                "a_center_xy": list(a_center) if a_center is not None else None,
                "b_center_xy": list(b_center) if b_center is not None else None,
                "selected_edge": self._json_safe(selected_edge),
                "candidates": self._json_safe(candidates),
                "extra_info": self._json_safe(extra_info or {}),
                "saved": saved,
            }
            p_json = out_dir / f"{stem}_AB_nearest_B_edge_meta.json"
            with p_json.open("w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            saved["meta_path"] = str(p_json)
        except Exception as e:
            self.log(f"[AB最近边角度] 保存 A/B 分割调试图失败：{e}")
        return saved

    def _clear_rule_ab_bad_scene_cache(self, follower: Any, reason: str = ""):
        """
        清理 RuleAB/SAM2 A/B 跟踪失败后可能残留的坏 scene 缓存。

        注意：这里只清理运行时 scene/mask 临时缓存，不清空完整测量标定点，
        不重新弹窗，不改变 A/B/C prompt。这样下一次 capture_and_build_scene()
        会重新取当前帧并重新分割/跟踪 A、B。
        """
        if follower is None:
            return
        if not bool(getattr(self.cfg, "rule_ab_ab_seg_reset_bad_scene_cache", True)):
            return
        objs = [follower, getattr(follower, "abc_segmenter", None), getattr(follower, "segmenter", None)]
        cache_attrs = (
            "last_scene", "current_scene", "scene",
            "last_image_rgb", "current_image_rgb",
            "last_a_mask", "current_a_mask", "a_mask",
            "last_b_mask", "current_b_mask", "b_mask",
            "last_a_center", "current_a_center", "last_b_center", "current_b_center",
        )
        for obj in objs:
            if obj is None:
                continue
            for attr in cache_attrs:
                try:
                    if hasattr(obj, attr):
                        setattr(obj, attr, None)
                except Exception:
                    pass
        if reason:
            self.log(f"[RuleAB-A/B重试] 已清理坏 scene 缓存；reason={reason}")

    def _capture_and_build_scene_with_ab_retry(
        self,
        follower: Any,
        label: str,
        max_attempts: Optional[int] = None,
        allow_fail: bool = False,
    ) -> Tuple[Optional[np.ndarray], Any, Dict[str, Any]]:
        """
        完整测量中所有需要 A/B 分割的地方统一通过这个函数取帧。

        解决的问题：
            - A temporal 候选质量不足；
            - B temporal 候选质量不足；
            - infer_abc() 抛 RuntimeError；
            - capture_and_build_scene() 抛异常。

        行为：
            失败时不直接终止完整测量，而是等待下一帧后重新 capture + infer_abc。
            达到重试次数仍失败时：
                allow_fail=True  -> 返回 (None, None, info)
                allow_fail=False -> 抛 RuntimeError，交给上层决定是否继续 while。
        """
        attempts = int(max_attempts if max_attempts is not None else getattr(self.cfg, "rule_ab_ab_seg_retry_attempts", 3))
        attempts = max(1, attempts)
        interval_s = max(0.0, float(getattr(self.cfg, "rule_ab_ab_seg_retry_interval_s", 0.25)))
        last_exc: Optional[BaseException] = None
        info: Dict[str, Any] = {
            "ok": False,
            "label": label,
            "attempts": attempts,
            "errors": [],
        }

        if follower is None:
            msg = "RuleAB follower 为空，无法取帧分割 A/B"
            info["reason"] = msg
            if allow_fail:
                return None, None, info
            raise RuntimeError(msg)

        # 完整测量/特征跟踪模式下，所有 A/B/C 取帧入口都必须先被 feature tracker 接管。
        if bool(getattr(self.cfg, "rule_ab_use_feature_tracker_after_sam2_init", True)):
            self._ensure_rule_ab_feature_tracker_installed(follower, self._get_loaded_calibration_state())

        for attempt in range(1, attempts + 1):
            if self.stop_requested:
                msg = "stop_requested"
                info["reason"] = msg
                if allow_fail:
                    return None, None, info
                raise RuntimeError(msg)
            try:
                image_rgb, scene = follower.capture_and_build_scene()
                if bool(getattr(self.cfg, "rule_ab_use_feature_tracker_after_sam2_init", True)):
                    src = str(getattr(scene, "segmentation_source", getattr(scene, "source", "")) or "") if scene is not None else ""
                    if isinstance(scene, dict):
                        src = str(scene.get("segmentation_source", scene.get("source", src)) or src)
                    # 不是硬性依赖 source 字段是否存在，但必须把当前 scene 记录为 feature_tracker_scene，
                    # 这样 Step7 / 角度检测 / overlay 保存读取的是同一个 scene.a。
                    self.context["feature_tracker_scene"] = scene
                    info["scene_source"] = src or "feature_tracker"
                info.update({"ok": True, "success_attempt": attempt, "reason": "ok"})
                if attempt > 1:
                    self.log(f"[RuleAB-A/B重试] {label}: 第 {attempt}/{attempts} 次重新取帧分割成功。")
                return image_rgb, scene, info
            except Exception as e:
                last_exc = e
                msg = str(e)
                info["errors"].append({"attempt": attempt, "error": msg})
                self.log(
                    f"[RuleAB-A/B重试] {label}: A/B 分割失败 attempt={attempt}/{attempts}: {msg}；"
                    "不停止完整测量，准备重新取下一帧。"
                )
                self._clear_rule_ab_bad_scene_cache(follower, reason=f"{label}/attempt={attempt}: {msg}")
                if attempt < attempts:
                    time.sleep(interval_s)

        final_msg = f"{label}: A/B 分割连续失败 {attempts} 次，最后错误：{last_exc}"
        info["reason"] = final_msg
        if allow_fail:
            return None, None, info
        raise RuntimeError(final_msg)


    # --------------------------------------------------------
    # 指定 B 边 KLT 光流跟踪：首帧人工选边，Step1/Step7 跨轮次连续跟踪
    # --------------------------------------------------------

    @staticmethod
    def _edge_angle_from_endpoints(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        dx = float(p2[0]) - float(p1[0])
        dy = float(p2[1]) - float(p1[1])
        a = math.degrees(math.atan2(dy, dx)) % 180.0
        if a < 0:
            a += 180.0
        return float(a)

    @staticmethod
    def _edge_length_midpoint(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, Tuple[float, float]]:
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        return float(math.hypot(x2 - x1, y2 - y1)), ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    def _sample_points_on_edge(self, p1: Tuple[float, float], p2: Tuple[float, float], n: Optional[int] = None) -> np.ndarray:
        npts = int(n if n is not None else getattr(self.cfg, "rule_ab_klt_num_points", 20))
        npts = max(2, npts)
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        pts = []
        for i in range(npts):
            t = i / max(1, npts - 1)
            pts.append([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])
        return np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)

    def _b_edge_klt_output_dir(self) -> Path:
        base = self.run_session_dir if self.run_session_dir is not None else self.output_root
        out = Path(base) / str(getattr(self.cfg, "rule_ab_klt_overlay_dir_name", "ab_angle_klt_edge"))
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _build_selected_b_edge_descriptor(self, b_mask: np.ndarray, p1: Tuple[float, float], p2: Tuple[float, float]) -> Dict[str, Any]:
        center = self._mask_center_xy(b_mask)
        length, mid = self._edge_length_midpoint(p1, p2)
        angle = self._edge_angle_from_endpoints(p1, p2)
        rel_mid = [0.0, 0.0]
        if center is not None:
            rel_mid = [float(mid[0] - center[0]), float(mid[1] - center[1])]
        return {
            "p1": [float(p1[0]), float(p1[1])],
            "p2": [float(p2[0]), float(p2[1])],
            "angle_deg": float(angle),
            "length_px": float(length),
            "midpoint_xy": [float(mid[0]), float(mid[1])],
            "b_center_xy": [float(center[0]), float(center[1])] if center is not None else None,
            "relative_midpoint_xy": rel_mid,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _candidate_edge_score_for_descriptor(
        self,
        candidate: Dict[str, Any],
        descriptor: Dict[str, Any],
        b_center: Optional[Tuple[float, float]],
        last_angle: Optional[float] = None,
    ) -> float:
        try:
            p1 = candidate.get("p1")
            p2 = candidate.get("p2")
            cand_len = float(candidate.get("length_px", 1.0))
            cand_angle = float(candidate.get("angle_deg"))
            _, cand_mid = self._edge_length_midpoint(p1, p2)
            ref_len = max(1.0, float(descriptor.get("length_px", cand_len)))
            ref_angle = float(last_angle if last_angle is not None else descriptor.get("angle_deg", cand_angle))
            angle_cost = self.angle_diff_deg(cand_angle, ref_angle) / 15.0
            len_cost = abs(math.log(max(cand_len, 1.0) / ref_len))
            rel_cost = 0.0
            if b_center is not None and descriptor.get("relative_midpoint_xy") is not None:
                rx, ry = descriptor.get("relative_midpoint_xy", [0.0, 0.0])
                pred_mid = (float(b_center[0]) + float(rx), float(b_center[1]) + float(ry))
                rel_cost = math.hypot(cand_mid[0] - pred_mid[0], cand_mid[1] - pred_mid[1]) / 120.0
            # 分数越低越好。
            return float(angle_cost * 1.8 + len_cost * 1.0 + rel_cost * 0.7)
        except Exception:
            return 1e9

    def _relocate_selected_b_edge_from_mask(self, b_mask: np.ndarray, descriptor: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        descriptor = descriptor or self.b_edge_klt_state.get("descriptor") or {}
        if b_mask is None or not np.any(np.asarray(b_mask).astype(bool)):
            return {"ok": False, "reason": "empty_b_mask"}
        candidates = self._bmask_edge_angle_candidates_from_mask(b_mask)
        if not candidates:
            return {"ok": False, "reason": "no_b_edge_candidates"}
        b_center = self._mask_center_xy(b_mask)
        last_angle = self.b_edge_klt_state.get("last_angle")
        scored = []
        for c in candidates:
            score = self._candidate_edge_score_for_descriptor(c, descriptor, b_center, last_angle=last_angle)
            scored.append((score, c))
        scored.sort(key=lambda x: x[0])
        best = dict(scored[0][1])
        p1 = tuple(best["p1"])
        p2 = tuple(best["p2"])
        angle = self._edge_angle_from_endpoints(p1, p2)
        length, mid = self._edge_length_midpoint(p1, p2)
        best.update({
            "angle_deg": float(angle),
            "length_px": float(length),
            "midpoint_xy": [float(mid[0]), float(mid[1])],
            "score": float(scored[0][0]),
        })
        return {
            "ok": True,
            "edge": best,
            "angle_deg": float(angle),
            "endpoints": [list(map(float, p1)), list(map(float, p2))],
            "candidate_count": len(candidates),
            "descriptor": self._json_safe(descriptor),
            "reason": "ok",
        }

    def _reset_b_edge_klt_state_for_new_run(
        self,
        state: Optional[CalibrationState] = None,
        reason: str = "new_full_measurement_run",
    ) -> Optional[CalibrationState]:
        """
        完整循环测量启动时清空“指定 B 边”运行状态。

        注意：函数名保留 klt，是为了不改 GUI/run_workflow 的调用链；
        当前实现已经改为动态一维灰度导数扫描（profile edge probe），不再使用 KLT 光流。
        """
        if state is None:
            state = self._get_loaded_calibration_state()

        if state is not None:
            state.selected_b_edge_endpoints = []
            state.selected_b_edge_angle = float("nan")
            state.selected_b_edge_length = float("nan")
            state.selected_b_edge_midpoint = []
            state.selected_b_edge_descriptor = {}
            self.context["calibration_state"] = state.to_dict()

        self.b_edge_klt_state.update({
            "initialized": False,
            "selected": False,
            "prev_gray": None,
            "prev_pts": None,
            "last_gray": None,
            "last_edge_endpoints": None,
            "last_angle": None,
            "last_valid": False,
            "descriptor": None,
            "profile_polarity": None,
            "failure_count": 0,
            "relocate_stable_count": 0,
            "last_relocated_angle": None,
            "last_label": "",
        })

        self.context["selected_b_edge_descriptor"] = None
        self.context["force_reselect_klt_edge_this_run"] = True
        self.context["b_edge_profile_state"] = {
            k: self._json_safe(v)
            for k, v in self.b_edge_klt_state.items()
            if k not in ("prev_gray", "prev_pts", "last_gray")
        }

        self.log(
            "[Profile指定边] 已清空旧指定边与动态扫描状态："
            f"reason={reason}；本次完整测量将重新弹窗选择 B 指定边。"
        )
        return state

    def _select_initial_b_edge_interactively(
        self,
        image_rgb: np.ndarray,
        b_mask: np.ndarray,
        state: Optional[CalibrationState] = None,
    ) -> Dict[str, Any]:
        """首帧 cleaned B mask 上人工点击两个端点，确定需要跨轮次 profile 扫描跟踪的 B 物理边。"""
        force_reselect_this_run = bool(self.context.get("force_reselect_klt_edge_this_run", False))
        try:
            endpoints = getattr(state, "selected_b_edge_endpoints", None) if state is not None else None
            if (not force_reselect_this_run) and endpoints and len(endpoints) >= 2:
                p1 = (float(endpoints[0][0]), float(endpoints[0][1]))
                p2 = (float(endpoints[1][0]), float(endpoints[1][1]))
                desc = getattr(state, "selected_b_edge_descriptor", {}) or self._build_selected_b_edge_descriptor(b_mask, p1, p2)
                return {"ok": True, "p1": p1, "p2": p2, "descriptor": desc, "source": "calibration_json"}
            if force_reselect_this_run and endpoints and len(endpoints) >= 2:
                self.log("[Profile指定边] 本次完整测量已要求重新选边，忽略 calibration.json 中保存的旧 selected_b_edge_endpoints。")
        except Exception:
            pass

        img = np.asarray(image_rgb).astype(np.uint8)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        mask_bool = np.asarray(b_mask).astype(bool)
        overlay = bgr.copy()
        overlay[mask_bool] = (0, 255, 0)
        show = cv2.addWeighted(overlay, 0.35, bgr, 0.65, 0)

        candidates = self._bmask_edge_angle_candidates_from_mask(mask_bool)
        for idx, c in enumerate(candidates):
            try:
                p1 = tuple(int(round(float(v))) for v in c["p1"])
                p2 = tuple(int(round(float(v))) for v in c["p2"])
                cv2.line(show, p1, p2, (255, 0, 255), 2, cv2.LINE_AA)
                mx, my = int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)
                cv2.putText(show, str(idx), (mx + 4, my - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            except Exception:
                continue
        cv2.putText(show, "Click two endpoints of selected B edge, ENTER confirm, ESC cancel", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        clicks: List[Tuple[float, float]] = []
        win = "Select B edge endpoints for PROFILE"

        def _mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 2:
                clicks.append((float(x), float(y)))

        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, _mouse)
        while True:
            canvas = show.copy()
            for i, (x, y) in enumerate(clicks):
                cv2.circle(canvas, (int(round(x)), int(round(y))), 6, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.putText(canvas, f"P{i+1}", (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            if len(clicks) == 2:
                cv2.line(canvas, (int(clicks[0][0]), int(clicks[0][1])), (int(clicks[1][0]), int(clicks[1][1])), (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow(win, canvas)
            key = cv2.waitKey(30) & 0xFF
            if key in (13, 10) and len(clicks) == 2:
                break
            if key == 27:
                cv2.destroyWindow(win)
                raise RuntimeError("用户取消了 B 指定边 profile 端点选择。")
            if key in (8, 127) and clicks:
                clicks.pop()
        cv2.destroyWindow(win)

        p1, p2 = clicks[0], clicks[1]
        desc = self._build_selected_b_edge_descriptor(mask_bool, p1, p2)
        return {"ok": True, "p1": p1, "p2": p2, "descriptor": desc, "source": "manual_click"}

    def _profile_output_dir(self) -> Path:
        base = self.run_session_dir if self.run_session_dir is not None else self.output_root
        out = Path(base) / str(getattr(self.cfg, "rule_ab_profile_overlay_dir_name", "ab_angle_profile_edge"))
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _edge_unit_vectors(self, endpoints: Any) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        ep = np.asarray(endpoints, dtype=np.float32).reshape(2, 2)
        v = ep[1] - ep[0]
        length = float(np.linalg.norm(v))
        if length < 2.0:
            raise RuntimeError(f"selected edge too short: {length:.3f}px")
        t = v / length
        n = np.asarray([-t[1], t[0]], dtype=np.float32)
        mid = (ep[0] + ep[1]) * 0.5
        return t, n, length, mid

    def _profile_gray(self, image_rgb: np.ndarray) -> np.ndarray:
        img = np.asarray(image_rgb).astype(np.uint8)
        if img.ndim == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def _sample_profile_line(self, gray: np.ndarray, center: np.ndarray, normal: np.ndarray, half_width: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        half = max(3.0, float(half_width))
        count = int(round(half * 2.0)) + 1
        offsets = np.linspace(-half, half, count).astype(np.float32)
        coords = center.reshape(1, 2) + offsets.reshape(-1, 1) * normal.reshape(1, 2)
        xs = coords[:, 0].astype(np.float32).reshape(-1, 1)
        ys = coords[:, 1].astype(np.float32).reshape(-1, 1)
        prof = cv2.remap(gray, xs, ys, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).reshape(-1).astype(np.float32)
        return offsets, prof, coords.astype(np.float32)

    def _smooth_profile(self, profile: np.ndarray) -> np.ndarray:
        k = int(getattr(self.cfg, "rule_ab_profile_smooth_kernel", 5))
        if k <= 1:
            return profile.astype(np.float32)
        if k % 2 == 0:
            k += 1
        k = max(3, k)
        return cv2.GaussianBlur(profile.reshape(1, -1).astype(np.float32), (k, 1), 0).reshape(-1)

    def _choose_profile_peak(self, grad: np.ndarray, offsets: np.ndarray, polarity: str) -> Tuple[Optional[int], float, str]:
        pol = str(polarity or "auto").lower()
        if pol not in ("positive", "negative", "abs", "auto"):
            pol = "auto"
        if pol == "positive":
            scores = grad.copy()
            idx = int(np.argmax(scores))
            amp = float(scores[idx])
            used = "positive"
        elif pol == "negative":
            scores = -grad.copy()
            idx = int(np.argmax(scores))
            amp = float(scores[idx])
            used = "negative"
        else:
            scores = np.abs(grad)
            idx = int(np.argmax(scores))
            amp = float(scores[idx])
            used = "positive" if float(grad[idx]) >= 0 else "negative"
            if pol == "abs":
                used = "abs"
        if not np.isfinite(amp):
            return None, 0.0, used
        return idx, amp, used

    def _auto_gradient_threshold(self, abs_grad_values: List[float]) -> float:
        manual = float(getattr(self.cfg, "rule_ab_profile_gradient_threshold", 0.0))
        if manual > 0:
            return manual
        arr = np.asarray([v for v in abs_grad_values if np.isfinite(v)], dtype=np.float32)
        if arr.size == 0:
            return 4.0
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        p70 = float(np.percentile(arr, 70))
        return max(3.0, med + 2.5 * mad, p70 * 0.35)

    def _detect_profile_points(
        self,
        image_rgb: np.ndarray,
        predicted_endpoints: Any,
        polarity: str,
    ) -> Dict[str, Any]:
        gray = self._profile_gray(image_rgb)
        t, n, length, mid = self._edge_unit_vectors(predicted_endpoints)
        requested_lines = int(getattr(self.cfg, "rule_ab_profile_num_lines", 20))
        requested_lines = max(3, requested_lines)
        spacing = float(getattr(self.cfg, "rule_ab_profile_line_spacing_px", 8.0))
        if spacing > 0:
            max_by_spacing = max(3, int(math.floor(length / max(1.0, spacing))) + 1)
            n_lines = max(3, min(requested_lines, max_by_spacing))
        else:
            n_lines = requested_lines
        # 避开端点 5%，减少端点附近圆角/遮挡对直线拟合的影响。
        ts = np.linspace(0.05, 0.95, n_lines).astype(np.float32)
        ep = np.asarray(predicted_endpoints, dtype=np.float32).reshape(2, 2)
        centers = ep[0].reshape(1, 2) + ts.reshape(-1, 1) * (ep[1] - ep[0]).reshape(1, 2)
        half_width = float(getattr(self.cfg, "rule_ab_profile_half_width_px", 30.0))
        max_shift = float(getattr(self.cfg, "rule_ab_profile_max_point_shift_px", 20.0))

        raw_candidates: List[Dict[str, Any]] = []
        amps: List[float] = []
        for i, c in enumerate(centers):
            offsets, prof, coords = self._sample_profile_line(gray, c.astype(np.float32), n, half_width)
            smooth = self._smooth_profile(prof)
            grad = np.gradient(smooth).astype(np.float32)
            idx, amp, used_pol = self._choose_profile_peak(grad, offsets, polarity)
            if idx is None:
                raw_candidates.append({"line_index": i, "ok": False, "reason": "no_peak"})
                continue
            shift = float(offsets[idx])
            pt = coords[idx].astype(float).tolist()
            raw_candidates.append({
                "line_index": int(i),
                "ok": True,
                "center": [float(c[0]), float(c[1])],
                "point": [float(pt[0]), float(pt[1])],
                "shift_px": shift,
                "amp": float(amp),
                "polarity": used_pol,
                "scan_p1": [float(coords[0, 0]), float(coords[0, 1])],
                "scan_p2": [float(coords[-1, 0]), float(coords[-1, 1])],
            })
            amps.append(abs(float(amp)))

        thr = self._auto_gradient_threshold(amps)
        good: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        for r in raw_candidates:
            if not r.get("ok", False):
                rejected.append(r)
                continue
            if abs(float(r.get("shift_px", 0.0))) > max_shift:
                r["reject_reason"] = f"shift>{max_shift}"
                rejected.append(r)
                continue
            if abs(float(r.get("amp", 0.0))) < thr:
                r["reject_reason"] = f"amp<thr({thr:.3f})"
                rejected.append(r)
                continue
            good.append(r)

        return {
            "ok": True,
            "gray": gray,
            "predicted_endpoints": np.asarray(predicted_endpoints, dtype=np.float32).reshape(2, 2).tolist(),
            "tangent": t.tolist(),
            "normal": n.tolist(),
            "length_px": float(length),
            "polarity": polarity,
            "threshold": float(thr),
            "good": good,
            "rejected": rejected,
            "raw_candidates": raw_candidates,
        }

    def _fit_profile_edge_from_points(self, points: np.ndarray, predicted_endpoints: Any) -> Dict[str, Any]:
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        min_good = int(getattr(self.cfg, "rule_ab_profile_min_good_points", 6))
        if len(pts) < min_good:
            return {"ok": False, "reason": f"good_points<{min_good}: {len(pts)}"}
        max_res = float(getattr(self.cfg, "rule_ab_profile_max_fit_residual_px", 4.0))
        pred = np.asarray(predicted_endpoints, dtype=np.float32).reshape(2, 2)
        pred_t, _pred_n, pred_len, _pred_mid = self._edge_unit_vectors(pred)

        best_inliers = None
        best_score = (-1, float("inf"))
        rng = np.random.default_rng(12345)
        pairs = []
        n = len(pts)
        # 先枚举相邻较远点，再补随机，保证小样本稳定。
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(pts[j] - pts[i]) >= max(5.0, pred_len * 0.15):
                    pairs.append((i, j))
        if len(pairs) > 120:
            pairs = [pairs[int(k)] for k in np.linspace(0, len(pairs) - 1, 120)]
        for _ in range(60):
            if n >= 2:
                i, j = rng.choice(n, size=2, replace=False)
                if np.linalg.norm(pts[j] - pts[i]) >= 3.0:
                    pairs.append((int(i), int(j)))
        for i, j in pairs:
            p0 = pts[i]
            v = pts[j] - pts[i]
            norm = float(np.linalg.norm(v))
            if norm < 1e-6:
                continue
            v = v / norm
            # 方向与上一帧边保持一致，避免 180° 翻转。
            if float(np.dot(v, pred_t)) < 0:
                v = -v
            d = np.abs((pts[:, 0] - p0[0]) * (-v[1]) + (pts[:, 1] - p0[1]) * v[0])
            inliers = d <= max_res
            cnt = int(np.count_nonzero(inliers))
            mean_res = float(np.mean(d[inliers])) if cnt > 0 else float("inf")
            score = (cnt, -mean_res)
            if cnt > best_score[0] or (cnt == best_score[0] and mean_res < -best_score[1]):
                best_score = (cnt, -mean_res)
                best_inliers = inliers
        if best_inliers is None or int(np.count_nonzero(best_inliers)) < min_good:
            return {"ok": False, "reason": f"ransac_inliers<{min_good}"}

        in_pts = pts[best_inliers]
        mean = np.mean(in_pts, axis=0)
        centered = in_pts - mean.reshape(1, 2)
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            t = vh[0].astype(np.float32)
        except Exception:
            t = pred_t.astype(np.float32)
        if float(np.dot(t, pred_t)) < 0:
            t = -t
        nvec = np.asarray([-t[1], t[0]], dtype=np.float32)
        residuals = np.abs((in_pts[:, 0] - mean[0]) * nvec[0] + (in_pts[:, 1] - mean[1]) * nvec[1])
        residual = float(np.mean(residuals)) if residuals.size else 0.0

        # 用当前检测点的投影中位数作为边中心，但保持首帧/上一帧的物理长度，避免遮挡导致端点收缩。
        proj = np.dot(in_pts - pred[0].reshape(1, 2), t)
        center_proj = float(np.median(proj))
        center = pred[0] + center_proj * t
        half_len = pred_len * 0.5
        p1 = center - half_len * t
        p2 = center + half_len * t
        angle = self._edge_angle_from_endpoints((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1])))
        outlier_pts = pts[~best_inliers]
        return {
            "ok": True,
            "p1": [float(p1[0]), float(p1[1])],
            "p2": [float(p2[0]), float(p2[1])],
            "angle_deg": float(angle),
            "good_points": in_pts.astype(float).tolist(),
            "rejected_points": outlier_pts.astype(float).tolist(),
            "good_count": int(len(in_pts)),
            "rejected_count": int(len(outlier_pts)),
            "fit_residual_px": residual,
            "line_center": [float(center[0]), float(center[1])],
        }

    def _infer_initial_profile_polarity(self, image_rgb: np.ndarray, endpoints: Any) -> str:
        cfg_pol = str(getattr(self.cfg, "rule_ab_profile_edge_polarity", "auto") or "auto").lower()
        if cfg_pol in ("positive", "negative", "abs"):
            return cfg_pol
        scan = self._detect_profile_points(image_rgb, endpoints, polarity="abs")
        signs = []
        for r in scan.get("raw_candidates", []):
            if not r.get("ok", False):
                continue
            pol = str(r.get("polarity", ""))
            if pol == "positive":
                signs.append(1)
            elif pol == "negative":
                signs.append(-1)
        if not signs:
            return "abs"
        return "positive" if sum(signs) >= 0 else "negative"

    def _initialize_b_edge_profile(
        self,
        image_rgb: np.ndarray,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        descriptor: Optional[Dict[str, Any]] = None,
        source: str = "manual",
    ) -> Dict[str, Any]:
        gray = self._profile_gray(image_rgb)
        angle = self._edge_angle_from_endpoints(p1, p2)
        length, mid = self._edge_length_midpoint(p1, p2)
        descriptor = dict(descriptor or {})
        descriptor.update({
            "p1": [float(p1[0]), float(p1[1])],
            "p2": [float(p2[0]), float(p2[1])],
            "angle_deg": float(angle),
            "length_px": float(length),
            "midpoint_xy": [float(mid[0]), float(mid[1])],
            "tracker": "dynamic_1d_gray_derivative_profile",
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        polarity = descriptor.get("profile_polarity") or self._infer_initial_profile_polarity(image_rgb, [p1, p2])
        descriptor["profile_polarity"] = str(polarity)
        self.b_edge_klt_state.update({
            "initialized": True,
            "selected": True,
            "prev_gray": None,
            "prev_pts": None,
            "last_gray": gray,
            "last_edge_endpoints": [[float(p1[0]), float(p1[1])], [float(p2[0]), float(p2[1])]],
            "last_angle": float(angle),
            "last_valid": True,
            "descriptor": descriptor,
            "profile_polarity": str(polarity),
            "last_label": str(source),
        })
        self.context["selected_b_edge_descriptor"] = self._json_safe(descriptor)
        self.context["b_edge_profile_state"] = {k: self._json_safe(v) for k, v in self.b_edge_klt_state.items() if k not in ("prev_gray", "prev_pts", "last_gray")}
        return {"ok": True, "angle_deg": float(angle), "endpoints": self.b_edge_klt_state["last_edge_endpoints"], "source": source, "profile_polarity": str(polarity)}

    def _ensure_initial_b_edge_klt_selected_and_initialized(
        self,
        image_rgb: np.ndarray,
        b_mask: np.ndarray,
        state: Optional[CalibrationState] = None,
    ) -> Dict[str, Any]:
        """兼容旧函数名：初始化 profile 指定边，不再初始化 KLT 光流点。"""
        if not bool(getattr(self.cfg, "rule_ab_use_profile_selected_b_edge_angle", True)):
            return {"ok": False, "reason": "profile_selected_b_edge_disabled"}
        if self.b_edge_klt_state.get("initialized", False):
            return {"ok": True, "reason": "already_initialized", "angle_deg": self.b_edge_klt_state.get("last_angle")}
        sel = self._select_initial_b_edge_interactively(image_rgb, b_mask, state=state)
        if not sel.get("ok", False):
            return sel
        p1 = tuple(sel["p1"])
        p2 = tuple(sel["p2"])
        desc = sel.get("descriptor") or self._build_selected_b_edge_descriptor(b_mask, p1, p2)
        init = self._initialize_b_edge_profile(image_rgb, p1, p2, descriptor=desc, source="initial_selected_b_edge_profile")
        self.context["force_reselect_klt_edge_this_run"] = False
        self.log(
            f"[Profile指定边] 已初始化：angle={float(init.get('angle_deg')):.6f}°, "
            f"polarity={init.get('profile_polarity')}, endpoints={init.get('endpoints')}, source={sel.get('source')}"
        )
        return init

    def _save_b_edge_klt_overlay(
        self,
        image_rgb: np.ndarray,
        label: str,
        result: Dict[str, Any],
        b_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, str]:
        """兼容旧函数名：保存 profile 指定边检测 overlay 到 ab_angle_profile_edge。"""
        try:
            out_dir = self._profile_output_dir()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            stem = f"{ts}_{label}_B_PROFILE_edge"
            overlay_path = out_dir / f"{stem}.png"
            json_path = out_dir / f"{stem}.json"
            img = np.asarray(image_rgb).astype(np.uint8)
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            canvas = bgr.copy()
            if b_mask is not None:
                try:
                    mb = np.asarray(b_mask).astype(bool)
                    tint = canvas.copy()
                    tint[mb] = (0, 120, 0)
                    canvas = cv2.addWeighted(tint, 0.25, canvas, 0.75, 0)
                except Exception:
                    pass

            def _pt(p):
                return (int(round(float(p[0]))), int(round(float(p[1]))))

            pred = result.get("predicted_endpoints") or result.get("previous_endpoints")
            if pred is not None:
                try:
                    cv2.line(canvas, _pt(pred[0]), _pt(pred[1]), (255, 255, 0), 2, cv2.LINE_AA)
                except Exception:
                    pass
            for r in result.get("scan_lines", []) or []:
                try:
                    cv2.line(canvas, _pt(r["scan_p1"]), _pt(r["scan_p2"]), (120, 120, 120), 1, cv2.LINE_AA)
                except Exception:
                    continue
            for p in result.get("rejected_points", []) or []:
                try:
                    cv2.circle(canvas, _pt(p), 3, (0, 0, 255), -1, cv2.LINE_AA)
                except Exception:
                    pass
            for p in result.get("good_points", []) or []:
                try:
                    cv2.circle(canvas, _pt(p), 4, (0, 255, 255), -1, cv2.LINE_AA)
                except Exception:
                    pass
            ep = result.get("endpoints") or result.get("repaired_edge_endpoints")
            if ep is not None:
                try:
                    cv2.line(canvas, _pt(ep[0]), _pt(ep[1]), (0, 0, 255), 3, cv2.LINE_AA)
                    cv2.circle(canvas, _pt(ep[0]), 5, (0, 0, 255), -1, cv2.LINE_AA)
                    cv2.circle(canvas, _pt(ep[1]), 5, (0, 0, 255), -1, cv2.LINE_AA)
                except Exception:
                    pass
            lines = [
                f"PROFILE selected B edge: {label}",
                f"status={result.get('profile_status', result.get('klt_status'))}, source={result.get('angle_source')}",
                f"angle={result.get('angle_deg')}, delta={result.get('delta')}",
                f"good={result.get('good_count')}, residual={result.get('fit_residual_px')}, polarity={result.get('profile_polarity')}",
                f"allow_close={result.get('allow_close')}, reason={result.get('reason')}",
            ]
            y = 24
            for line in lines:
                cv2.putText(canvas, str(line), (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
                y += 24
            cv2.imwrite(str(overlay_path), canvas)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(self._json_safe(result), f, ensure_ascii=False, indent=2)
            return {"overlay_path": str(overlay_path), "json_path": str(json_path)}
        except Exception as e:
            self.log(f"[Profile指定边] 保存 overlay 失败：{e}")
            return {}

    def _track_selected_b_edge_klt_on_image(
        self,
        image_rgb: np.ndarray,
        b_mask: Optional[np.ndarray],
        label: str,
        baseline_angle: Optional[float] = None,
        allow_relocate: bool = True,
        allow_close_from_relocation: bool = False,
    ) -> Dict[str, Any]:
        """兼容旧函数名：用动态一维灰度导数扫描跟踪指定 B 物理边。"""
        result: Dict[str, Any] = {
            "ok": False,
            "label": label,
            "angle_deg": None,
            "baseline_angle": baseline_angle,
            "profile_status": "not_run",
            "klt_status": "disabled_replaced_by_profile",
            "angle_source": "failed_no_close",
            "allow_close": False,
            "method": "dynamic_1d_gray_derivative_profile",
        }
        st = self.b_edge_klt_state
        curr_gray = self._profile_gray(image_rgb)
        if not st.get("initialized", False) or st.get("last_edge_endpoints") is None:
            if b_mask is not None and np.any(np.asarray(b_mask).astype(bool)):
                init = self._ensure_initial_b_edge_klt_selected_and_initialized(image_rgb, np.asarray(b_mask).astype(bool), state=self._get_loaded_calibration_state())
                if not init.get("ok", False):
                    result.update({"profile_status": "not_initialized", "reason": init.get("reason", "profile_not_initialized")})
                    paths = self._save_b_edge_klt_overlay(image_rgb, label, result, b_mask=b_mask)
                    result.update(paths)
                    return result
            else:
                result.update({"profile_status": "not_initialized", "reason": "profile_not_initialized_and_no_initial_b_mask"})
                paths = self._save_b_edge_klt_overlay(image_rgb, label, result, b_mask=b_mask)
                result.update(paths)
                return result

        predicted = np.asarray(st.get("last_edge_endpoints"), dtype=np.float32).reshape(2, 2)
        polarity = str(st.get("profile_polarity") or (st.get("descriptor") or {}).get("profile_polarity") or getattr(self.cfg, "rule_ab_profile_edge_polarity", "auto"))
        if polarity == "auto":
            polarity = "abs"
        result["previous_endpoints"] = predicted.astype(float).tolist()
        result["predicted_endpoints"] = predicted.astype(float).tolist()
        result["profile_polarity"] = polarity
        try:
            scan = self._detect_profile_points(image_rgb, predicted, polarity=polarity)
            good_records = scan.get("good", [])
            points = np.asarray([r["point"] for r in good_records], dtype=np.float32).reshape(-1, 2) if good_records else np.empty((0, 2), dtype=np.float32)
            fit = self._fit_profile_edge_from_points(points, predicted)
            if not fit.get("ok", False):
                raise RuntimeError(str(fit.get("reason", "profile_fit_failed")))
            angle = float(fit["angle_deg"])
            last_angle = st.get("last_angle")
            if last_angle is not None:
                jump = self.angle_diff_deg(angle, float(last_angle))
                max_jump = float(getattr(self.cfg, "rule_ab_profile_max_angle_jump_deg", 25.0))
                if jump > max_jump:
                    raise RuntimeError(f"profile_angle_jump>{max_jump}: {jump:.3f}")
            endpoints = [fit["p1"], fit["p2"]]
            self._initialize_b_edge_profile(image_rgb, tuple(endpoints[0]), tuple(endpoints[1]), descriptor=st.get("descriptor"), source=label)
            delta = self.angle_diff_deg(angle, float(baseline_angle)) if baseline_angle is not None else None
            result.update({
                "ok": True,
                "angle_deg": float(angle),
                "angle_deg_raw": float(angle),
                "endpoints": endpoints,
                "baseline_angle": baseline_angle,
                "delta": delta,
                "profile_status": "success",
                "klt_status": "disabled_replaced_by_profile_success",
                "angle_source": "profile_success",
                "scan_lines": [{k: r.get(k) for k in ("line_index", "scan_p1", "scan_p2", "center", "shift_px", "amp", "polarity")} for r in scan.get("raw_candidates", []) if r.get("ok", False)],
                "good_points": fit.get("good_points", []),
                "rejected_points": fit.get("rejected_points", []) + [r.get("point") for r in scan.get("rejected", []) if r.get("point") is not None],
                "good_count": int(fit.get("good_count", 0)),
                "rejected_count": int(fit.get("rejected_count", 0)) + int(len(scan.get("rejected", []))),
                "fit_residual_px": float(fit.get("fit_residual_px", 0.0)),
                "gradient_threshold": float(scan.get("threshold", 0.0)),
                "allow_close": True,
                "reason": "ok",
            })
            st["failure_count"] = 0
            st["last_gray"] = curr_gray
        except Exception as e:
            st["failure_count"] = int(st.get("failure_count", 0) or 0) + 1
            fail_count = int(st.get("failure_count", 0))
            fail_max = int(getattr(self.cfg, "rule_ab_profile_fail_max", 3))
            last_ep = st.get("last_edge_endpoints")
            last_angle = st.get("last_angle")
            delta = self.angle_diff_deg(float(last_angle), float(baseline_angle)) if (baseline_angle is not None and last_angle is not None) else None
            # 短暂失败时只保持上一帧预测边，不允许用该角度触发关激光，避免误关。
            if last_ep is not None and last_angle is not None and fail_count <= fail_max:
                result.update({
                    "ok": True,
                    "angle_deg": float(last_angle),
                    "angle_deg_raw": float(last_angle),
                    "endpoints": self._json_safe(last_ep),
                    "baseline_angle": baseline_angle,
                    "delta": delta,
                    "profile_status": "hold_last_after_fail",
                    "klt_status": "disabled_replaced_by_profile_hold_last",
                    "angle_source": "profile_hold_last_no_close",
                    "good_count": 0,
                    "fit_residual_px": None,
                    "allow_close": False,
                    "failure_count": fail_count,
                    "reason": f"profile_failed_hold_last: {e}",
                })
            else:
                result.update({
                    "ok": False,
                    "angle_deg": None,
                    "profile_status": "failed",
                    "klt_status": "disabled_replaced_by_profile_failed",
                    "angle_source": "failed_no_close",
                    "allow_close": False,
                    "failure_count": fail_count,
                    "reason": f"profile_failed: {e}",
                })
        paths = self._save_b_edge_klt_overlay(image_rgb, label, result, b_mask=b_mask)
        result.update(paths)
        self.context["b_edge_profile_state"] = {k: self._json_safe(v) for k, v in self.b_edge_klt_state.items() if k not in ("prev_gray", "prev_pts", "last_gray")}
        # 兼容旧界面/日志字段。
        self.context["b_edge_klt_state"] = self.context["b_edge_profile_state"]
        return result

    def _bmask_longest_edge_output_dir(self) -> Path:
        """Bmask 最长边角度检测的统一输出目录。"""
        base = self.run_session_dir if self.run_session_dir is not None else self.output_root
        override_dir_name = self.context.get("bmask_longest_edge_overlay_dir_override")
        dir_name = str(override_dir_name or getattr(self.cfg, "rule_ab_bmask_longest_edge_overlay_dir_name", "ab_angle_bmask_longest_edge"))
        out = Path(base) / dir_name
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _clean_bmask_for_longest_edge(self, b_mask: np.ndarray) -> np.ndarray:
        """把 B mask 转成干净的 uint8 二值图，只服务于最长边几何检测。"""
        if b_mask is None:
            return np.zeros((1, 1), dtype=np.uint8)
        m = np.asarray(b_mask)
        if m.ndim == 3:
            m = m[..., 0]
        m = (m.astype(bool).astype(np.uint8) * 255)
        if m.size <= 0:
            return m.astype(np.uint8)
        k = int(getattr(self.cfg, "rule_ab_bmask_longest_edge_morph_kernel", 3))
        if k >= 2:
            if k % 2 == 0:
                k += 1
            kernel = np.ones((k, k), dtype=np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
        return m.astype(np.uint8)

    def _select_longest_edge_from_bmask(self, b_mask: np.ndarray) -> Dict[str, Any]:
        """
        从当前帧 Bmask 最大外轮廓中选择最长边；poly 失败时用 minAreaRect 兜底。
        返回字段满足 Step1 / Step7 / simple_angle_loop 统一角度入口使用。
        """
        m = self._clean_bmask_for_longest_edge(b_mask)
        if m is None or m.size <= 0 or int(np.count_nonzero(m)) <= 0:
            return {"ok": False, "reason": "bmask_empty"}

        min_area = int(getattr(self.cfg, "rule_ab_bmask_longest_edge_min_area_px", 20))
        area_px = int(np.count_nonzero(m > 0))
        if area_px < max(1, min_area):
            return {"ok": False, "reason": f"bmask_area_too_small:{area_px}< {min_area}", "b_mask_area_px": area_px}

        contours = self._find_contours_compat(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {"ok": False, "reason": "no_external_contour", "b_mask_area_px": area_px}
        contour = max(contours, key=cv2.contourArea)
        contour_area = float(cv2.contourArea(contour))
        if contour is None or len(contour) < 2 or contour_area <= 0:
            return {"ok": False, "reason": "invalid_largest_contour", "b_mask_area_px": area_px, "contour_area_px": contour_area}

        def _edge_item(p1: Any, p2: Any, source: str) -> Optional[Dict[str, Any]]:
            x1, y1 = float(p1[0]), float(p1[1])
            x2, y2 = float(p2[0]), float(p2[1])
            length = float(math.hypot(x2 - x1, y2 - y1))
            if length <= 1e-6:
                return None
            angle = self._edge_angle_from_endpoints((x1, y1), (x2, y2))
            return {
                "angle_deg": float(angle),
                "length_px": float(length),
                "edge_length_px": float(length),
                "p1": (x1, y1),
                "p2": (x2, y2),
                "endpoints": [[x1, y1], [x2, y2]],
                "angle_source": source,
            }

        candidates: List[Dict[str, Any]] = []
        peri = float(cv2.arcLength(contour, True))
        approx_used = None
        if peri > 0:
            for eps_ratio in (0.006, 0.010, 0.015, 0.025, 0.040, 0.060):
                approx = cv2.approxPolyDP(contour, max(1.0, eps_ratio * peri), True)
                if approx is not None and len(approx) >= 3:
                    approx_used = approx.reshape(-1, 2).astype(np.float32)
                    break
        if approx_used is not None and len(approx_used) >= 3:
            n = len(approx_used)
            for i in range(n):
                item = _edge_item(approx_used[i], approx_used[(i + 1) % n], "bmask_longest_edge_polygon")
                if item is not None:
                    candidates.append(item)

        if candidates:
            selected = max(candidates, key=lambda d: float(d.get("length_px", 0.0)))
            return {
                "ok": True,
                "reason": "ok",
                "angle_deg": float(selected["angle_deg"]),
                "angle_deg_raw": float(selected["angle_deg"]),
                "edge_length_px": float(selected["length_px"]),
                "endpoints": self._json_safe(selected["endpoints"]),
                "selected_b_edge": self._json_safe(selected),
                "candidate_edges": self._json_safe(candidates),
                "candidate_count": len(candidates),
                "angle_source": "bmask_longest_edge_polygon",
                "edge_selection_mode": "bmask_longest_edge",
                "b_mask_area_px": area_px,
                "contour_area_px": contour_area,
                "approx_vertex_count": int(len(approx_used)) if approx_used is not None else 0,
                "clean_mask": m,
                "largest_contour": contour,
            }

        # 兜底：多边形近似失败或没有可用边时，用 minAreaRect 四条边。
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect).astype(np.float32)
        rect_candidates: List[Dict[str, Any]] = []
        for i in range(4):
            item = _edge_item(box[i], box[(i + 1) % 4], "bmask_longest_edge_minAreaRect_fallback")
            if item is not None:
                rect_candidates.append(item)
        if not rect_candidates:
            return {"ok": False, "reason": "minAreaRect_no_valid_edge", "b_mask_area_px": area_px, "contour_area_px": contour_area}
        selected = max(rect_candidates, key=lambda d: float(d.get("length_px", 0.0)))
        return {
            "ok": True,
            "reason": "ok_minAreaRect_fallback",
            "angle_deg": float(selected["angle_deg"]),
            "angle_deg_raw": float(selected["angle_deg"]),
            "edge_length_px": float(selected["length_px"]),
            "endpoints": self._json_safe(selected["endpoints"]),
            "selected_b_edge": self._json_safe(selected),
            "candidate_edges": self._json_safe(rect_candidates),
            "candidate_count": len(rect_candidates),
            "angle_source": "bmask_longest_edge_minAreaRect_fallback",
            "edge_selection_mode": "bmask_longest_edge",
            "b_mask_area_px": area_px,
            "contour_area_px": contour_area,
            "approx_vertex_count": 0,
            "clean_mask": m,
            "largest_contour": contour,
        }

    def _save_bmask_longest_edge_overlay(
        self,
        image_rgb: np.ndarray,
        b_mask: Optional[np.ndarray],
        edge_result: Dict[str, Any],
        label: str,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """保存当前帧原图、Bmask、最长边 overlay 和 JSON 元数据。"""
        saved: Dict[str, str] = {}
        out_dir = self._bmask_longest_edge_output_dir()
        safe_label = "".join(ch if (ch.isalnum() or ch in "_-.()") else "_" for ch in str(label))[:90]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        stem = f"{ts}_{safe_label}"

        img = np.asarray(image_rgb).copy()
        if img.ndim == 2:
            canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            raw_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            canvas = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            raw_rgb = img

        raw_path = out_dir / f"{stem}_raw.png"
        cv2.imwrite(str(raw_path), cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR))
        saved["raw_image_path"] = str(raw_path)
        saved["image_path"] = str(raw_path)

        if b_mask is not None:
            clean = self._clean_bmask_for_longest_edge(b_mask)
            mask_path = out_dir / f"{stem}_B_mask.png"
            cv2.imwrite(str(mask_path), clean)
            saved["b_mask_path"] = str(mask_path)
            overlay = canvas.copy()
            overlay[clean.astype(bool)] = (0, 128, 255)
            canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)
            contours = self._find_contours_compat(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(canvas, contours, -1, (0, 255, 255), 1, cv2.LINE_AA)

        # 所有候选边：灰色；当前最长边：红色加粗。
        for c in edge_result.get("candidate_edges") or []:
            try:
                p1 = c.get("p1") or (c.get("endpoints") or [None, None])[0]
                p2 = c.get("p2") or (c.get("endpoints") or [None, None])[1]
                if p1 is None or p2 is None:
                    continue
                cv2.line(canvas, (int(round(float(p1[0]))), int(round(float(p1[1])))),
                         (int(round(float(p2[0]))), int(round(float(p2[1])))), (180, 180, 180), 1, cv2.LINE_AA)
            except Exception:
                pass

        endpoints = edge_result.get("endpoints") or []
        if len(endpoints) >= 2:
            try:
                p1, p2 = endpoints[0], endpoints[1]
                x1, y1 = int(round(float(p1[0]))), int(round(float(p1[1])))
                x2, y2 = int(round(float(p2[0]))), int(round(float(p2[1])))
                cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 255), 3, cv2.LINE_AA)
                cv2.circle(canvas, (x1, y1), 5, (255, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas, (x2, y2), 5, (255, 0, 255), -1, cv2.LINE_AA)
            except Exception:
                pass

        lines = [
            f"label={label}",
            f"ok={bool(edge_result.get('ok', False))}",
            f"angle_deg={edge_result.get('angle_deg')}",
            f"edge_length_px={edge_result.get('edge_length_px')}",
            f"source={edge_result.get('angle_source')}",
            f"reason={edge_result.get('reason')}",
        ]
        y = 24
        for line in lines:
            cv2.putText(canvas, str(line), (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(canvas, str(line), (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
            y += 22

        overlay_path = out_dir / f"{stem}_B_LONGEST_EDGE_overlay.png"
        cv2.imwrite(str(overlay_path), canvas)
        saved["overlay_image_path"] = str(overlay_path)
        saved["overlay_path"] = str(overlay_path)

        meta = dict(edge_result)
        meta.pop("clean_mask", None)
        meta.pop("largest_contour", None)
        meta.update({"label": label, "saved_paths": saved})
        if extra_info:
            meta.update({k: self._json_safe(v) for k, v in extra_info.items()})
        meta_path = out_dir / f"{stem}_B_LONGEST_EDGE_meta.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(self._json_safe(meta), f, ensure_ascii=False, indent=2)
        saved["meta_path"] = str(meta_path)
        return saved

    def detect_step7_yolo_obb_angle_once(
        self,
        follower: Any = None,
        label: str = "step_yolo_obb",
        baseline_angle: Optional[float] = None,
        allow_fail: bool = True,
        allow_close_from_relocation: bool = False,
        save_overlay: bool = True,
        **_unused: Any,
    ) -> Dict[str, Any]:
        """Step1/Step7 唯一角度入口：YOLO-OBB 四角长边角度。

        当前版本不再使用 Bmask 最长边、AB 最近边、KLT、Profile 或任何角度修复。
        follower 参数仅保留调用兼容，不参与角度计算。
        """
        result = self.detect_angle_once(label=str(label).replace("bmask_longest_edge", "yolo_obb"), allow_fail=allow_fail)
        if result.get("ok", False) and baseline_angle is not None and result.get("angle_deg") is not None:
            try:
                result["baseline_angle"] = float(baseline_angle)
                result["delta"] = self.angle_diff_deg(float(result["angle_deg"]), float(baseline_angle))
                result["delta_from_baseline"] = result["delta"]
                result["final_delta_to_baseline"] = result["delta"]
            except Exception:
                pass
        result["angle_source"] = "yolo_obb_long_edge" if result.get("ok", False) else result.get("angle_source", "yolo_obb_failed")
        result["edge_selection_mode"] = "yolo_obb_long_edge"
        result["allow_close"] = bool(result.get("ok", False))
        result["final_angle"] = result.get("angle_deg")
        return result

    def _stop_rule_ab_stage34_if_possible(self, follower: Any, reason: str = "") -> bool:
        """Step7 实时保护：尽最大可能立即停止 3/4 通道控制器。"""
        try:
            stage = getattr(follower, "stage", None) if follower is not None else None
            if stage is None and self.rule_ab_follower is not None:
                stage = getattr(self.rule_ab_follower, "stage", None)
            if stage is None:
                self.log(f"[Step7实时] 未找到 Stage34，无法 emergency stop；reason={reason}")
                return False
            with self.rule_ab_stage34_lock:
                if hasattr(stage, "stop_all") and callable(getattr(stage, "stop_all")):
                    stage.stop_all()
                    self.log(f"[Step7实时] 已发送 stage.stop_all()；reason={reason}")
                    return True
                m = getattr(stage, "stop", None)
                if callable(m):
                    for ch in (int(getattr(self.cfg, "rule_ab_stage_x_channel", 3)), int(getattr(self.cfg, "rule_ab_stage_y_channel", 4))):
                        try:
                            m(channel=ch)
                        except TypeError:
                            try:
                                m(ch)
                            except Exception:
                                pass
                    self.log(f"[Step7实时] 已发送 stage.stop(channel=3/4)；reason={reason}")
                    return True
            self.log(f"[Step7实时] Stage34 不支持 stop_all/stop；reason={reason}")
            return False
        except Exception as e:
            self.log(f"[Step7实时] Stage34 emergency stop 失败：{e}")
            return False

    def _is_midrun_recalibration_requested(self) -> bool:
        """完整测量中途重标定请求是否已经置位。"""
        return bool(
            getattr(self, "pause_for_recalibration_requested", False)
            or getattr(self, "recalibration_in_progress", False)
            or getattr(self, "restart_current_cycle_after_recalibration", False)
        )

    def request_midrun_recalibration_pause(self, cycle_index: Optional[int] = None) -> None:
        """
        GUI 点击“暂停运动并重新标定ABC”时调用。

        只停止当前运动与视觉运行态，不关闭照明/激光控制器/TCP，不结束完整测量总线程。
        完整测量线程会在当前 Step7/Step9/单轮流程检测到该标志后，等待用户重标定完成。
        """
        self.pause_for_recalibration_requested = True
        self.recalibration_in_progress = True
        self.resume_after_recalibration_requested = False
        self.restart_current_cycle_after_recalibration = True
        if cycle_index is not None:
            self.midrun_recalibration_cycle_index = int(cycle_index)
        else:
            try:
                self.midrun_recalibration_cycle_index = int(self.context.get("cycle_index") or 0) or None
            except Exception:
                self.midrun_recalibration_cycle_index = None

        self.context["midrun_recalibration_requested"] = True
        self.context["midrun_recalibration_in_progress"] = True
        self.context["restart_current_cycle_after_recalibration"] = True
        self.context["midrun_recalibration_cycle_index"] = self.midrun_recalibration_cycle_index

        # 立即尝试停止 3/4 和 1/2 通道，但不置 stop_requested。
        try:
            self._stop_rule_ab_stage34_if_possible(self.rule_ab_follower, reason="midrun_recalibration_requested")
        except Exception as e:
            self.log(f"[中途重标定] 停止 Stage34 失败：{e}")
        try:
            self.stop_step9_stage12()
        except Exception as e:
            self.log(f"[中途重标定] 停止 Step9 Stage12 失败：{e}")

        self.log(
            "[中途重标定] 已请求暂停当前 cycle 并重新标定 A/B/C/Step9；"
            "完整测量线程不会结束，当前 cycle 将在重标定后从 Step1 重新开始。"
        )
        self.notify_update()

    def _reset_b_edge_runtime_state(self) -> None:
        """重置 B 边/角度检测运行态。"""
        try:
            self.b_edge_klt_state.update({
                "initialized": False,
                "selected": False,
                "prev_gray": None,
                "prev_pts": None,
                "last_gray": None,
                "last_edge_endpoints": None,
                "last_angle": None,
                "last_valid": False,
                "descriptor": None,
                "profile_polarity": None,
                "failure_count": 0,
                "relocate_stable_count": 0,
                "last_relocated_angle": None,
                "last_label": "",
            })
        except Exception:
            self.b_edge_klt_state = {
                "initialized": False,
                "selected": False,
                "prev_gray": None,
                "prev_pts": None,
                "last_gray": None,
                "last_edge_endpoints": None,
                "last_angle": None,
                "last_valid": False,
                "descriptor": None,
                "profile_polarity": None,
                "failure_count": 0,
                "relocate_stable_count": 0,
                "last_relocated_angle": None,
                "last_label": "",
            }

    def prepare_runtime_for_midrun_recalibration_waiting(self) -> None:
        """
        当前 cycle 已经暂停后，进入“等待用户重新标定”的安全状态。

        不关闭全部设备，不结束完整测量总线程；只释放会污染后续识别的运行态缓存。
        """
        self.recalibration_in_progress = True
        self.pause_for_recalibration_requested = False
        self.restart_current_cycle_after_recalibration = True
        self.resume_after_recalibration_requested = False
        self.stop_requested = False
        self.step9_stop_requested = False
        self.is_measuring = True
        self.context["is_measuring"] = True
        self.context["midrun_recalibration_requested"] = False
        self.context["midrun_recalibration_in_progress"] = True
        self.context["restart_current_cycle_after_recalibration"] = True

        # 释放旧 RuleAB / RuleAC / Step9 运行态，避免旧 SAM2 prompt、feature template、static C 路线污染新标定。
        for attr, label in (
            ("rule_ab_follower", "RuleAB"),
            ("rule_ac_controller", "RuleAC"),
        ):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    if hasattr(obj, "close"):
                        obj.close()
                    self.log(f"[中途重标定] 已释放旧 {label} 运行态")
                except Exception as e:
                    self.log(f"[中途重标定] 释放旧 {label} 运行态失败：{e}")
                try:
                    setattr(self, attr, None)
                except Exception:
                    pass

        try:
            self.reset_step9_tracking_state(clear_b_points=True)
        except Exception:
            pass
        self.step9_b_last_mask = None
        self.step9_b_last_center = None
        self.step9_b_last_box = None
        self.step9_color_last_mask = None
        self.step9_color_last_center = None
        self.step9_current_run_dir = None
        self.step9_history_csv_path = None
        self.step9_sam2_predictor = None
        self.step9_sam2_model_info = None
        self.step9_dynamic_c_last_mask = None
        self.step9_dynamic_c_last_raw_mask = None
        self.step9_dynamic_c_last_center = None
        self.step9_dynamic_c_last_box = None
        self.step9_dynamic_c_last_geometry = None

        self._reset_b_edge_runtime_state()
        self.context.pop("step7_c_edge_route_state", None)
        self.context["last_rule_ab_result"] = None
        self.context["last_rule_ac_result"] = None
        self.log("[中途重标定] 已进入等待重标定状态：请重新标定 A/B/C/Step9，完成后点击“完成重标定并继续测量”。")
        self.notify_update()

    def resume_after_midrun_recalibration(self, state: Optional[CalibrationState] = None) -> None:
        """
        用户完成 A/B/C/Step9 标定后调用。重新加载标定并准备从当前 cycle 的 Step1 重跑。
        """
        if state is None:
            state = self.load_calibration_state(required=True)
        if state is None:
            raise RuntimeError("无法继续：未获得有效完整标定包。")
        state = self._complete_calibration_state_for_runtime(state)
        missing = state.missing_items()
        if missing:
            raise RuntimeError("无法继续完整测量，中途重标定仍缺少：\n" + "\n".join(f"- {x}" for x in missing))

        self.prepare_runtime_from_full_calibration(state)
        self._force_cfg_to_strict_full_calibration_c("resume_after_midrun_recalibration")
        self._reset_b_edge_runtime_state()
        self.stop_requested = False
        self.step9_stop_requested = False
        self.is_measuring = True
        self.context["is_measuring"] = True
        self.pause_for_recalibration_requested = False
        self.recalibration_in_progress = False
        self.resume_after_recalibration_requested = False
        self.restart_current_cycle_after_recalibration = False
        self.context["midrun_recalibration_requested"] = False
        self.context["midrun_recalibration_in_progress"] = False
        self.context["restart_current_cycle_after_recalibration"] = False
        self.log("[中途重标定] 已加载新标定并重建运行态；当前 cycle 将从 Step1 重新开始。")
        self.notify_update()

    def _laser_off_once_threadsafe(self, reason: str = "") -> bool:
        """
        兼容旧代码入口，但 Step7 实时角度线程不再允许直接关闭激光。

        激光关闭必须回到完整测量主流程中的正常 Step8/关激光步骤执行；
        Step7 只负责设置 stop_event 并停止/暂停 Stage34。
        """
        self.log(
            f"[Step7实时] 已禁止角度线程直接 laser_off；reason={reason}；"
            "激光将由完整测量主流程的正常关激光步骤统一处理。"
        )
        return False


    def _interruptible_pause(self, seconds: float, stop_event: Optional[threading.Event], phase_state: Optional[Dict[str, Any]] = None) -> bool:
        """保留 pause 总时长，但分段 sleep 并反复检查 stop_event。"""
        seconds = max(0.0, float(seconds or 0.0))
        if seconds <= 0:
            return bool(stop_event is not None and stop_event.is_set())
        interval = max(0.005, float(getattr(self.cfg, "rule_ab_pause_check_interval_s", 0.02)))
        end_t = time.time() + seconds
        if phase_state is not None:
            phase_state["phase"] = "pause"
            phase_state["pause_s"] = float(seconds)
        interrupted = False
        while time.time() < end_t:
            if stop_event is not None and stop_event.is_set():
                interrupted = True
                break
            time.sleep(min(interval, max(0.0, end_t - time.time())))
        if phase_state is not None:
            phase_state["pause_interrupted"] = bool(interrupted)
        return bool(interrupted)

    def _execute_rule_ab_action_with_watchable_pause(
        self,
        follower: Any,
        action_id: int,
        stop_event: Optional[threading.Event] = None,
        phase_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        保持 follower.execute_action() 的原始方向映射；临时禁止其内部整段 sleep，
        再由 workflow 执行可被 stop_event 打断的分段 pause。
        """
        if stop_event is not None and stop_event.is_set():
            return {"ok": False, "reason": "stop_event_set_before_execute_action", "action_id": int(action_id)}
        fc = getattr(follower, "cfg", None)
        old_ch3 = getattr(fc, "stage_ch3_pause_after_move_s", float(getattr(self.cfg, "rule_ab_stage_ch3_pause_after_move_s", 2.0))) if fc is not None else float(getattr(self.cfg, "rule_ab_stage_ch3_pause_after_move_s", 2.0))
        old_ch4 = getattr(fc, "stage_ch4_pause_after_move_s", float(getattr(self.cfg, "rule_ab_stage_ch4_pause_after_move_s", 2.0))) if fc is not None else float(getattr(self.cfg, "rule_ab_stage_ch4_pause_after_move_s", 2.0))
        try:
            with self.rule_ab_stage34_lock:
                if fc is not None:
                    try:
                        fc.stage_ch3_pause_after_move_s = 0.0
                        fc.stage_ch4_pause_after_move_s = 0.0
                    except Exception:
                        pass
                if phase_state is not None:
                    phase_state["phase"] = "moving"
                    phase_state["controller_action"] = f"action_id={int(action_id)}"
                move_info = follower.execute_action(int(action_id)) or {}
        finally:
            if fc is not None:
                try:
                    fc.stage_ch3_pause_after_move_s = old_ch3
                    fc.stage_ch4_pause_after_move_s = old_ch4
                except Exception:
                    pass
        ch = move_info.get("channel", None)
        ch_name = str(move_info.get("channel_name", "") or "").upper()
        x_ch = int(getattr(self.cfg, "rule_ab_stage_x_channel", 3))
        y_ch = int(getattr(self.cfg, "rule_ab_stage_y_channel", 4))
        try:
            ch_i = int(ch)
        except Exception:
            ch_i = None
        if ch_i == x_ch or ch_name in ("X", "CH3", "LEFT", "RIGHT"):
            pause_s = float(getattr(self.cfg, "rule_ab_stage_ch3_pause_after_move_s", old_ch3))
        elif ch_i == y_ch or ch_name in ("Y", "CH4", "UP", "DOWN"):
            pause_s = float(getattr(self.cfg, "rule_ab_stage_ch4_pause_after_move_s", old_ch4))
        else:
            pause_s = max(float(getattr(self.cfg, "rule_ab_stage_ch3_pause_after_move_s", old_ch3)), float(getattr(self.cfg, "rule_ab_stage_ch4_pause_after_move_s", old_ch4)))
        move_info["watchable_pause_s"] = float(pause_s)
        move_info["pause_interrupted"] = bool(self._interruptible_pause(pause_s, stop_event=stop_event, phase_state=phase_state))
        return self._json_safe(move_info)

    def _write_step7_realtime_records_csv(self, records: List[Dict[str, Any]], csv_path: Path) -> None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "timestamp", "frame_index", "step", "phase",
            "raw_angle", "final_angle", "angle_deg", "baseline_angle",
            "last_confirmed_angle_before", "raw_delta_to_last",
            "delta_from_baseline", "final_delta_to_baseline",
            "angle_source",
            "edge_length_px", "endpoints", "trigger_type", "controller_action",
            "raw_image_path", "overlay_image_path", "reason",
        ]
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in records:
                rr = dict(r)
                for key in ("endpoints",):
                    if isinstance(rr.get(key), (list, dict)):
                        rr[key] = json.dumps(self._json_safe(rr.get(key)), ensure_ascii=False)
                writer.writerow(rr)

    def run_rule_ab_realtime_until_angle_delta(
        self,
        baseline_angle: Optional[float],
        min_delta_deg: Optional[float] = None,
        max_delta_deg: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Step7 实时双线程版本：控制线程运动，角度线程在 moving/pause 期间持续检测。"""
        self.log("========== RuleAB：Step7实时双线程，Bmask最长边监控角度 ==========")
        min_delta = float(self.cfg.rule_ab_angle_delta_min_deg if min_delta_deg is None else min_delta_deg)
        max_delta = float(self.cfg.rule_ab_angle_delta_max_deg if max_delta_deg is None else max_delta_deg)
        follower = self.ensure_rule_ab_follower()
        self._ensure_rule_ab_feature_tracker_installed(follower, self._get_loaded_calibration_state())
        self.apply_runtime_rule_ab_params_to_follower(follower, reason="step7_realtime_start")
        stop_event = threading.Event()
        controller_done = threading.Event()
        records: List[Dict[str, Any]] = []
        records_lock = threading.Lock()
        phase_state: Dict[str, Any] = {"phase": "starting", "controller_action": "none", "step": 0}
        result_box: Dict[str, Any] = {"trigger_type": None, "stop_reason": "not_started", "final_angle": None, "final_delta": None, "controller_error": None, "angle_error": None}
        # 上一帧角度仅用于 CSV 诊断；不做角度修复或候选边替换。
        last_confirmed_angle_box: Dict[str, Any] = {"value": float(baseline_angle) if baseline_angle is not None else None}
        step_counter = {"value": 0}
        start_t = time.time()
        max_duration_s = float(getattr(self.cfg, "rule_ab_realtime_max_duration_s", 0.0) or 0.0)
        save_every = bool(getattr(self.cfg, "rule_ab_realtime_save_every_angle_frame", True))
        out_base = self.run_session_dir if self.run_session_dir is not None else self.output_root
        realtime_dir = Path(out_base) / str(getattr(self.cfg, "rule_ab_realtime_overlay_dir_name", "ab_angle_realtime_bmask_longest_edge"))
        realtime_dir.mkdir(parents=True, exist_ok=True)
        csv_path = realtime_dir / "step7_realtime_angle_records.csv"
        self.context.pop("step7_c_edge_route_state", None)

        def _append_record(rec: Dict[str, Any]) -> None:
            with records_lock:
                records.append(self._json_safe(rec))

        def angle_monitor_thread() -> None:
            frame_idx = 0
            old_override = self.context.get("bmask_longest_edge_overlay_dir_override")
            self.context["bmask_longest_edge_overlay_dir_override"] = str(getattr(self.cfg, "rule_ab_realtime_overlay_dir_name", "ab_angle_realtime_bmask_longest_edge"))
            try:
                while not stop_event.is_set() and not self.stop_requested and not self._is_midrun_recalibration_requested():
                    frame_idx += 1
                    try:
                        with self.rule_ab_vision_lock:
                            angle_result = self.detect_step7_yolo_obb_angle_once(
                                follower=follower,
                                label=f"step7_realtime_frame_{frame_idx:06d}",
                                baseline_angle=baseline_angle,
                                allow_fail=True,
                                allow_close_from_relocation=False,
                                save_overlay=save_every,
                            )
                        angle_ok = bool(angle_result.get("ok", False) and angle_result.get("angle_deg") is not None)
                        angle_deg = float(angle_result["angle_deg"]) if angle_ok else None
                        raw_angle = angle_deg
                        last_before = last_confirmed_angle_box.get("value")
                        delta = self.angle_diff_deg(float(angle_deg), float(baseline_angle)) if angle_deg is not None and baseline_angle is not None else None
                        raw_delta_to_last = self.angle_diff_deg(float(angle_deg), float(last_before)) if angle_deg is not None and last_before is not None else None
                        rec = {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                            "frame_index": int(frame_idx),
                            "step": int(phase_state.get("step", step_counter.get("value", 0)) or 0),
                            "phase": str(phase_state.get("phase", "unknown")),
                            "raw_angle": raw_angle,
                            "final_angle": angle_deg,
                            "angle_deg": angle_deg,
                            "baseline_angle": baseline_angle,
                            "last_confirmed_angle_before": last_before,
                            "raw_delta_to_last": raw_delta_to_last,
                            "delta_from_baseline": delta,
                            "final_delta_to_baseline": delta,
                            "angle_source": angle_result.get("angle_source"),
                            "edge_length_px": angle_result.get("edge_length_px"),
                            "endpoints": angle_result.get("endpoints"),
                            "trigger_type": "",
                            "controller_action": phase_state.get("controller_action", ""),
                            "raw_image_path": angle_result.get("raw_image_path") or angle_result.get("image_path"),
                            "overlay_image_path": angle_result.get("overlay_image_path") or angle_result.get("overlay_path"),
                            "reason": angle_result.get("reason"),
                        }
                        if not angle_ok:
                            rec["trigger_type"] = "angle_invalid"
                            _append_record(rec)
                            self.log("[Step7实时角度] 本帧 YOLO-OBB 角度无效；继续下一帧。")
                            self._interruptible_pause(float(getattr(self.cfg, "rule_ab_angle_watch_interval_s", 0.5)), stop_event, phase_state=None)
                            continue

                        # 成功得到 YOLO-OBB 原始角度后，更新上一帧角度，仅用于日志诊断，不参与修复。
                        last_confirmed_angle_box["value"] = float(angle_deg)

                        if angle_ok and delta is not None:
                            # Step7 当前逻辑：只使用当前轮 Step1 baseline 作为基准；
                            # 只判断 delta >= min_delta。max_delta / 6° 不再参与停止判定，
                            # 也不再把 delta >= max_delta 单独标记为 overshoot。
                            if float(delta) >= float(min_delta):
                                rec["trigger_type"] = "target_reached"
                                result_box.update({
                                    "trigger_type": "target_reached",
                                    "stop_reason": "realtime_bmask_longest_edge_delta_ge_target_min_continue_measurement",
                                    "final_angle": float(angle_deg),
                                    "final_delta": float(delta),
                                })
                                _append_record(rec)
                                self.log(
                                    f"[Step7实时角度] 角度变化已达到阈值：final_angle={angle_deg:.6f}°, "
                                    f"baseline={baseline_angle}, delta={delta:.3f}° >= {min_delta:.3f}°；"
                                    "停止/暂停Stage34当前推动，不再使用6°上限/overshoot判定；"
                                    "不在Step7角度线程关闭激光，后续由完整测量主流程正常关激光，并继续完整测量。"
                                )
                                stop_event.set()
                                self._stop_rule_ab_stage34_if_possible(follower, reason="angle_delta_ge_target_min_no_laser_off")
                                break
                        _append_record(rec)
                    except Exception as e:
                        result_box["angle_error"] = str(e)
                        _append_record({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], "frame_index": int(frame_idx), "step": int(phase_state.get("step", 0) or 0), "phase": str(phase_state.get("phase", "unknown")), "angle_deg": None, "baseline_angle": baseline_angle, "delta_from_baseline": None, "angle_source": "bmask_longest_edge_failed", "edge_length_px": None, "endpoints": None, "trigger_type": "", "controller_action": phase_state.get("controller_action", ""), "raw_image_path": "", "overlay_image_path": "", "reason": f"angle_monitor_exception:{e}"})
                    self._interruptible_pause(float(getattr(self.cfg, "rule_ab_angle_watch_interval_s", 0.5)), stop_event, phase_state=None)
            finally:
                if old_override is None:
                    self.context.pop("bmask_longest_edge_overlay_dir_override", None)
                else:
                    self.context["bmask_longest_edge_overlay_dir_override"] = old_override

        def controller_thread() -> None:
            try:
                step_idx = 0
                while not stop_event.is_set() and not self.stop_requested and not self._is_midrun_recalibration_requested():
                    if max_duration_s > 0 and (time.time() - start_t) >= max_duration_s:
                        result_box["stop_reason"] = "realtime_max_duration_reached"
                        stop_event.set()
                        break
                    if callable(getattr(self, "rule_ab_config_sync_callback", None)):
                        try:
                            self.rule_ab_config_sync_callback()
                        except Exception as e:
                            self.log(f"[Step7实时] 运行中同步 GUI 参数失败，本步继续使用旧参数：{e}")
                    self.apply_runtime_rule_ab_params_to_follower(follower, reason=f"step7_realtime_controller_step_{step_idx + 1}")
                    step_idx += 1
                    step_counter["value"] = int(step_idx)
                    phase_state.update({"step": int(step_idx), "phase": "planning"})
                    route_points = self._load_or_build_step7_c_edge_route(follower)
                    self.log(f"[Step7实时控制] step={step_idx}: 执行一次 C边沿路线运动；route_points={len(route_points)}；pause保留但可被stop_event打断。")
                    ok, route_info = self._run_one_c_edge_route_cycle(follower=follower, step_idx=step_idx, route_points=route_points, stop_event=stop_event, phase_state=phase_state)
                    _append_record({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], "frame_index": "", "step": int(step_idx), "phase": str(phase_state.get("phase", "unknown")), "angle_deg": None, "baseline_angle": baseline_angle, "delta_from_baseline": None, "angle_source": "controller", "edge_length_px": None, "endpoints": None, "trigger_type": "", "controller_action": phase_state.get("controller_action", ""), "raw_image_path": "", "overlay_image_path": route_info.get("route_overlay_path") if isinstance(route_info, dict) else "", "reason": route_info.get("reason", "controller_step_ok") if isinstance(route_info, dict) else "controller_step_ok"})
                    if stop_event.is_set():
                        break
                    if not ok:
                        result_box["stop_reason"] = "route_or_ab_seg_failed_retry_next_frame"
                        self.log(f"[Step7实时控制] 路线运动/A-B分割失败，继续等待下一帧：{route_info}")
                    phase_state["phase"] = "loop_interval"
                    self._interruptible_pause(float(getattr(self.cfg, "rule_ab_loop_interval_s", 0.15)), stop_event, phase_state=phase_state)
            except Exception as e:
                result_box["controller_error"] = str(e)
                result_box["stop_reason"] = f"controller_thread_exception:{e}"
                self.log(f"[Step7实时控制] controller_thread 异常：{e}")
                self.log(traceback.format_exc())
                stop_event.set()
            finally:
                phase_state["phase"] = "stopped"
                controller_done.set()

        t_angle = threading.Thread(target=angle_monitor_thread, name="step7_angle_monitor_thread", daemon=True)
        t_controller = threading.Thread(target=controller_thread, name="step7_controller_thread", daemon=True)
        t_angle.start()
        t_controller.start()
        while not controller_done.is_set() and not self.stop_requested:
            if self._is_midrun_recalibration_requested():
                result_box["stop_reason"] = "midrun_recalibration_requested"
                stop_event.set()
                self._stop_rule_ab_stage34_if_possible(follower, reason="midrun_recalibration_requested")
                break
            if stop_event.is_set():
                break
            time.sleep(0.02)
        if self.stop_requested:
            result_box["stop_reason"] = "user_stop_requested"
            stop_event.set()
            self._stop_rule_ab_stage34_if_possible(follower, reason="user_stop_requested")
        elif self._is_midrun_recalibration_requested():
            result_box["stop_reason"] = "midrun_recalibration_requested"
            stop_event.set()
            self._stop_rule_ab_stage34_if_possible(follower, reason="midrun_recalibration_requested")
        t_controller.join(timeout=3.0)
        stop_event.set()
        t_angle.join(timeout=3.0)
        try:
            self._write_step7_realtime_records_csv(records, csv_path)
        except Exception as e:
            self.log(f"[Step7实时] 保存实时角度 CSV 失败：{e}")
        trigger_type = result_box.get("trigger_type")
        ok = bool(trigger_type == "target_reached")
        overshoot = False
        allow_continue = bool(ok)
        result = {"ok": bool(allow_continue), "step7_success_in_target_range": bool(ok), "allow_continue_measurement": bool(allow_continue), "overshoot": bool(overshoot), "reason": str(result_box.get("stop_reason") or "stopped"), "baseline_angle": baseline_angle, "final_angle": result_box.get("final_angle"), "final_delta": result_box.get("final_delta"), "target_min": min_delta, "target_max": None, "angle_source": "bmask_longest_edge_realtime_thread", "records": self._json_safe(records), "realtime_csv_path": str(csv_path), "realtime_dir": str(realtime_dir), "controller_error": result_box.get("controller_error"), "angle_error": result_box.get("angle_error")}
        self.context["last_rule_ab_result"] = result
        self.notify_update()
        self.log(f"[Step7实时] 结束：ok={ok}, overshoot={overshoot}, reason={result['reason']}, final_angle={result['final_angle']}, final_delta={result['final_delta']}, records={len(records)}, csv={csv_path}")
        return result

    def run_rule_ab_until_angle_delta(
        self,
        baseline_angle: Optional[float],
        min_delta_deg: Optional[float] = None,
        max_delta_deg: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Step7 主循环：当前版本只使用“当前帧 Bmask 最长边”作为角度来源。

        不再使用 KLT、Profile 动态灰度扫描、上一帧 hold_last 或人工指定边。
        """
        if bool(getattr(self.cfg, "rule_ab_realtime_step7_enable", True)):
            return self.run_rule_ab_realtime_until_angle_delta(
                baseline_angle=baseline_angle,
                min_delta_deg=min_delta_deg,
                max_delta_deg=max_delta_deg,
            )

        self.log("========== RuleAB：A推动B，Bmask最长边检测B角度（旧串行兼容模式） ==========")

        min_delta = float(self.cfg.rule_ab_angle_delta_min_deg if min_delta_deg is None else min_delta_deg)
        max_delta = float(self.cfg.rule_ab_angle_delta_max_deg if max_delta_deg is None else max_delta_deg)

        follower = self.ensure_rule_ab_follower()
        self._ensure_rule_ab_feature_tracker_installed(follower, self._get_loaded_calibration_state())
        self.apply_runtime_rule_ab_params_to_follower(follower, reason="step7_start")
        try:
            st = getattr(follower, "stage", None)
            self.log(
                f"[RuleAB] Stage34状态：enable_stage={getattr(follower.cfg, 'enable_stage', None)}, "
                f"stage_is_none={st is None}, x_channel={getattr(follower.cfg, 'stage_x_channel', None)}, "
                f"y_channel={getattr(follower.cfg, 'stage_y_channel', None)}, "
                f"step_x={getattr(follower.cfg, 'stage_step_x', None)}, step_y={getattr(follower.cfg, 'stage_step_y', None)}"
            )
        except Exception:
            pass

        records: List[Dict[str, Any]] = []
        reached = False
        allow_continue_measurement = False
        stop_reason = "not_started"
        final_angle: Optional[float] = None
        final_delta: Optional[float] = None

        # Step7 每次进入都重新用当前 A center 找最近 route index 作为起点；
        # 后续帧只按 current_route_index + direction 推进，不再每帧跳到最近线段。
        self.context.pop("step7_c_edge_route_state", None)

        # 串行兼容模式中上一帧角度也仅用于诊断；不做角度修复。
        last_confirmed_angle: Optional[float] = float(baseline_angle) if baseline_angle is not None else None

        step_idx = 0
        while True:
            if self.stop_requested:
                stop_reason = "user_stop_requested"
                break
            if self._is_midrun_recalibration_requested():
                stop_reason = "midrun_recalibration_requested"
                self._stop_rule_ab_stage34_if_possible(follower, reason="midrun_recalibration_requested")
                break

            if callable(getattr(self, "rule_ab_config_sync_callback", None)):
                try:
                    self.rule_ab_config_sync_callback()
                except Exception as e:
                    self.log(f"[RuleAB] 运行中同步 GUI 参数失败，本步继续使用旧参数：{e}")

            if min_delta_deg is None:
                min_delta = float(getattr(self.cfg, "rule_ab_angle_delta_min_deg", min_delta))
            if max_delta_deg is None:
                max_delta = float(getattr(self.cfg, "rule_ab_angle_delta_max_deg", max_delta))

            follower = self.ensure_rule_ab_follower()
            self._ensure_rule_ab_feature_tracker_installed(follower, self._get_loaded_calibration_state())
            self.apply_runtime_rule_ab_params_to_follower(follower, reason=f"step7_step_{step_idx + 1}")

            step_idx += 1
            use_route = bool(getattr(self.cfg, "rule_ab_use_c_edge_route", True))
            route_info: Dict[str, Any] = {"route_enabled": use_route}

            if use_route:
                route_points = self._load_or_build_step7_c_edge_route(follower)
                self.log(
                    f"[RuleAB] step={step_idx}: 执行一次 C边沿路线运动；route_points={len(route_points)}；"
                    f"目标角度范围=[{min_delta}, {max_delta}]；角度来源=当前帧Bmask最长边。"
                )
                try:
                    ok, route_info = self._run_one_c_edge_route_cycle(follower, step_idx, route_points)
                except Exception as e:
                    ok = False
                    route_info = {
                        "route_enabled": True,
                        "reason": "route_cycle_exception_retry_next_frame",
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }
                if not ok:
                    self.log(f"[RuleAB-Route] 路线运动/A-B分割失败：{route_info}；不停止完整测量，重新取下一帧继续。")
                    records.append({
                        "step": step_idx,
                        "rule_ab_ok": False,
                        "angle_deg": None,
                        "angle_delta_from_baseline": None,
                        "angle_source": "skip_angle_after_route_ab_seg_fail",
                        "angle_result": {"ok": False, "reason": "skip_angle_after_route_ab_seg_fail"},
                        "c_edge_route": self._json_safe(route_info),
                        "retry_next_frame": True,
                    })
                    stop_reason = "route_or_ab_seg_failed_retry_next_frame"
                    time.sleep(float(self.cfg.rule_ab_loop_interval_s))
                    continue
            else:
                self.log(
                    f"[RuleAB] step={step_idx}: 执行一次 rule_ab.run_one_cycle()；"
                    f"目标角度范围=[{min_delta}, {max_delta}]；角度来源=当前帧Bmask最长边。"
                )
                try:
                    ok = follower.run_one_cycle()
                except Exception as e:
                    ok = False
                    route_info = {
                        "route_enabled": False,
                        "reason": "run_one_cycle_exception_retry_next_frame",
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }
                if not ok:
                    self.log(f"[RuleAB] run_one_cycle/A-B分割失败：{route_info}；重新取下一帧继续。")
                    records.append({
                        "step": step_idx,
                        "rule_ab_ok": False,
                        "angle_deg": None,
                        "angle_delta_from_baseline": None,
                        "angle_source": "skip_angle_after_run_one_cycle_fail",
                        "angle_result": {"ok": False, "reason": "skip_angle_after_run_one_cycle_fail"},
                        "c_edge_route": self._json_safe(route_info),
                        "retry_next_frame": True,
                    })
                    stop_reason = "run_one_cycle_failed_retry_next_frame"
                    time.sleep(float(self.cfg.rule_ab_loop_interval_s))
                    continue

            # 每一步运动后，重新取当前帧 Bmask，并直接使用最大外轮廓最长边计算角度；不使用上一帧 hold_last。
            angle_result = self.detect_step7_yolo_obb_angle_once(
                follower=follower,
                label=f"rule_ab_bmask_longest_edge_step_{step_idx}",
                baseline_angle=baseline_angle,
                allow_fail=True,
                allow_close_from_relocation=False,
            )

            angle_ok = bool(angle_result.get("ok", False) and angle_result.get("angle_deg") is not None)
            raw_angle = float(angle_result["angle_deg"]) if angle_ok else None
            current_angle = raw_angle
            angle_source = str(angle_result.get("angle_source", "yolo_obb_failed"))
            allow_close_by_angle = bool(angle_result.get("allow_close", False))
            final_angle = current_angle
            final_delta = None
            if current_angle is not None and baseline_angle is not None:
                final_delta = self.angle_diff_deg(float(current_angle), float(baseline_angle))
            if angle_ok:
                last_confirmed_angle = float(current_angle)

            record = {
                "step": step_idx,
                "rule_ab_ok": bool(ok),
                "raw_angle": raw_angle,
                "final_angle": current_angle,
                "angle_deg": current_angle,
                "angle_delta_from_baseline": final_delta,
                "final_delta_to_baseline": final_delta,
                "baseline_angle": baseline_angle,
                "last_confirmed_angle_before": last_confirmed_angle,
                "raw_delta_to_last": self.angle_diff_deg(float(raw_angle), float(last_confirmed_angle)) if raw_angle is not None and last_confirmed_angle is not None else None,
                "angle_source": angle_source,
                "edge_length_px": angle_result.get("edge_length_px"),
                "endpoints": angle_result.get("endpoints"),
                "bmask_longest_edge_allow_close": allow_close_by_angle,
                "angle_result": self._json_safe(angle_result),
                "c_edge_route": self._json_safe(route_info),
                "target_min": min_delta,
                "target_max": max_delta,
            }
            records.append(record)

            self.log(
                f"[RuleAB-Bmask最长边] step={step_idx}, angle={current_angle}, baseline={baseline_angle}, "
                f"delta={final_delta}, source={angle_source}, length={angle_result.get('edge_length_px')}, "
                f"allow_close={allow_close_by_angle}"
            )

            if not angle_ok:
                stop_reason = "bmask_longest_edge_angle_failed_retry_next_frame_no_close"
                self.log("[RuleAB-Bmask最长边] 本帧 Bmask 最长边角度无效，不使用上一帧角度；继续下一步推动/重新取帧。")
                time.sleep(float(self.cfg.rule_ab_loop_interval_s))
                continue

            # 串行兼容模式同样只判断 delta >= min_delta；不再使用 max_delta / 6° 上限。
            close_hit = bool(
                allow_close_by_angle
                and final_delta is not None
                and float(final_delta) >= float(min_delta)
            )
            if close_hit:
                reached = True
                allow_continue_measurement = True
                stop_reason = "bmask_longest_edge_delta_ge_target_min_continue_measurement"
                self.log(
                    f"[RuleAB-Bmask最长边] 角度变化已达到阈值：delta={float(final_delta):.3f}° >= "
                    f"{float(min_delta):.3f}°；停止本次 Stage34 推动；不再使用6°上限/overshoot判定；"
                    "不在Step7分支直接关闭激光，后续由完整测量主流程正常关激光并继续完整测量。"
                )
                break

            if final_delta is not None and float(final_delta) < float(min_delta):
                stop_reason = "bmask_longest_edge_delta_below_target_continue"
                self.log(
                    f"[RuleAB-Bmask最长边] 当前角度变化 {float(final_delta):.3f}° 未达到关闭阈值 "
                    f"{float(min_delta):.3f}°；继续下一步推动。"
                )
            else:
                stop_reason = "bmask_longest_edge_angle_valid_continue"
                self.log("[RuleAB-Bmask最长边] 本帧角度有效但未满足关闭条件；继续下一步。")

            time.sleep(float(self.cfg.rule_ab_loop_interval_s))

        result = {
            "ok": bool(reached or allow_continue_measurement),
            "step7_success_in_target_range": bool(reached),
            "allow_continue_measurement": bool(allow_continue_measurement),
            "reason": stop_reason,
            "baseline_angle": baseline_angle,
            "final_angle": final_angle,
            "final_delta": final_delta,
            "target_min": min_delta,
            "target_max": None,
            "angle_source": "bmask_longest_edge",
            "records": records,
        }
        self.context["last_rule_ab_result"] = result
        self.notify_update()
        return result

    # --------------------------------------------------------
    # Step7：沿 C 四边形边沿预生成路线运动
    # --------------------------------------------------------

    @staticmethod
    def _order_route_points_by_nearest(points: List[Tuple[float, float]], start_xy: Optional[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """把闭合路线旋转到距离当前 A 最近的点开始。"""
        if not points or start_xy is None:
            return points
        sx, sy = float(start_xy[0]), float(start_xy[1])
        d2 = [(float(x) - sx) ** 2 + (float(y) - sy) ** 2 for x, y in points]
        idx = int(np.argmin(np.asarray(d2)))
        return list(points[idx:]) + list(points[:idx])

    @staticmethod
    def _load_c_edge_route_from_dir(c_dir: str) -> List[Tuple[float, float]]:
        """从 C 标定目录读取 static_c_edge_route.json 的目标中线 route_points_xy。"""
        if not c_dir:
            return []
        route_path = Path(str(c_dir)) / "static_c_edge_route.json"
        if not route_path.exists():
            return []
        with route_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        pts = data.get("route_points_xy") or data.get("target_route_points_xy") or data.get("points_xy") or []
        out: List[Tuple[float, float]] = []
        for p in pts:
            try:
                out.append((float(p[0]), float(p[1])))
            except Exception:
                continue
        return out

    @staticmethod
    def _load_c_edge_route_band_from_dir(c_dir: str) -> Dict[str, Any]:
        """
        从 C 标定目录读取 Step7 三线安全带。

        返回内容：
            base_route_points_xy: C 近似四边形边界线；
            min_route_points_xy:  内侧安全边界线，外扩 min；
            route_points_xy:      中间目标线，外扩 target；
            max_route_points_xy:  外侧安全边界线，外扩 max。
        """
        out: Dict[str, Any] = {
            "ok": False,
            "base_route_points_xy": [],
            "min_route_points_xy": [],
            "route_points_xy": [],
            "max_route_points_xy": [],
        }
        if not c_dir:
            out["reason"] = "empty_c_dir"
            return out
        route_path = Path(str(c_dir)) / "static_c_edge_route.json"
        if not route_path.exists():
            out["reason"] = "route_json_missing"
            return out
        try:
            with route_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            out["reason"] = f"read_failed:{e}"
            return out

        def _pts(key: str) -> List[Tuple[float, float]]:
            arr = data.get(key) or []
            ans: List[Tuple[float, float]] = []
            for p0 in arr:
                try:
                    ans.append((float(p0[0]), float(p0[1])))
                except Exception:
                    continue
            return ans

        target = _pts("route_points_xy") or _pts("target_route_points_xy") or _pts("points_xy")
        base = _pts("base_route_points_xy")
        min_route = _pts("min_route_points_xy") or _pts("inner_route_points_xy")
        max_route = _pts("max_route_points_xy") or _pts("outer_route_points_xy")
        out.update({
            "ok": bool(target),
            "source": data.get("source", ""),
            "spacing_px": data.get("spacing_px", None),
            "follow_direction": data.get("follow_direction", None),
            "follow_direction_name": data.get("follow_direction_name", ""),
            "min_distance_px": data.get("min_distance_px", None),
            "target_distance_px": data.get("target_distance_px", data.get("safe_clearance_px", None)),
            "max_distance_px": data.get("max_distance_px", None),
            "base_route_points_xy": base,
            "min_route_points_xy": min_route,
            "route_points_xy": target,
            "max_route_points_xy": max_route,
        })
        if not target:
            out["reason"] = "target_route_empty"
        return out

    def _load_or_build_step7_c_edge_route(self, follower: Any) -> List[Tuple[float, float]]:
        """
        Step7 加载 C 近似四边形路线。

        当前版本默认使用 C 近似四边形四条边生成路线：
            static_c_quad_points.json -> 四边形边界 -> 按 spacing 采样。
        如果确实要恢复原始 SAM2 mask 外轮廓路线，可在 config.py 显式设置
        rule_ab_c_edge_route_source="sam2_outer_edge"。
        """
        c_dir = self._get_strict_full_calibration_c_dir() or str(getattr(self.cfg, "rule_ab_static_c_map_dir", "") or "")
        if not c_dir:
            return []

        c_path = Path(c_dir)
        route_source = str(getattr(self.cfg, "rule_ab_c_edge_route_source", "quad") or "quad").lower().strip()
        if route_source not in ("sam2_outer_edge", "mask_outer_edge", "sam2_mask", "quad", "quadrilateral"):
            route_source = "quad"

        required_clearance = float(getattr(self.cfg, "rule_ab_c_edge_route_safe_clearance_px", getattr(self.cfg, "rule_ab_ac_target_clearance_px", 90.0)))
        required_spacing = float(getattr(self.cfg, "rule_ab_c_edge_route_spacing_px", 12.0))
        required_follow = int(getattr(self.cfg, "rule_ab_follow_c_direction", 1))
        route_path = c_path / "static_c_edge_route.json"

        if route_path.exists():
            try:
                with route_path.open("r", encoding="utf-8") as f:
                    route_data = json.load(f)
                existing_clearance = route_data.get("safe_clearance_px", None)
                existing_source = str(route_data.get("source", "") or "").lower()
                existing_spacing = route_data.get("spacing_px", None)
                existing_follow = route_data.get("follow_direction", None)
                existing_min = route_data.get("min_distance_px", None)
                existing_target = route_data.get("target_distance_px", existing_clearance)
                existing_max = route_data.get("max_distance_px", None)
                route = self._load_c_edge_route_from_dir(str(c_path))

                source_ok = False
                if route_source in ("quad", "quadrilateral"):
                    source_ok = "quad" in existing_source
                else:
                    source_ok = ("sam2_outer" in existing_source) or ("mask_outer" in existing_source) or ("sam2_mask" in existing_source)

                clearance_ok = existing_clearance is not None and abs(float(existing_clearance) - required_clearance) <= 1e-6
                spacing_ok = existing_spacing is None or abs(float(existing_spacing) - required_spacing) <= 1e-6
                follow_ok = existing_follow is not None and int(existing_follow) == required_follow
                required_min = float(getattr(self.cfg, "rule_ab_route_min_distance_px", getattr(self.cfg, "rule_ab_ac_min_clearance_px", 80.0)))
                required_target = float(getattr(self.cfg, "rule_ab_route_safe_target_distance_px", required_clearance))
                required_max = float(getattr(self.cfg, "rule_ab_route_max_distance_px", getattr(self.cfg, "rule_ab_ac_max_clearance_px", 100.0)))
                band_param_ok = True
                if route_source in ("quad", "quadrilateral"):
                    band_param_ok = (
                        existing_min is not None and existing_target is not None and existing_max is not None
                        and abs(float(existing_min) - required_min) <= 1e-6
                        and abs(float(existing_target) - required_target) <= 1e-6
                        and abs(float(existing_max) - required_max) <= 1e-6
                    )
                band_ok = True
                if route_source in ("quad", "quadrilateral"):
                    band_ok = bool(
                        route_data.get("base_route_points_xy")
                        and route_data.get("min_route_points_xy")
                        and route_data.get("max_route_points_xy")
                    )

                if route and source_ok and clearance_ok and spacing_ok and follow_ok and band_ok and band_param_ok:
                    self.context["step7_c_edge_route_band"] = self._load_c_edge_route_band_from_dir(str(c_path))
                    return route
                if route:
                    self.log(
                        f"[RuleAB-Route] 检测到旧路线/路线来源不一致/安全距离不一致："
                        f"source={existing_source}, required_source={route_source}, "
                        f"clearance={existing_clearance}, required={required_clearance:.2f}px；将重新生成。"
                    )
            except Exception:
                pass

        route: List[Tuple[float, float]] = []
        if route_source in ("quad", "quadrilateral"):
            route = self._build_and_save_quad_edge_route(c_path)
        else:
            route = self._build_and_save_sam2_outer_edge_route(c_path)
            if not route:
                self.log("[RuleAB-Route] SAM2 原始 mask 外轮廓路线生成失败，尝试从 static_c_mask.png 外轮廓兜底生成。")
                route = self._build_and_save_sam2_outer_edge_route(c_path, prefer_sam2_mask=False)
            if not route:
                self.log("[RuleAB-Route] mask 外轮廓路线仍失败，最后回退到四边形路线。")
                route = self._build_and_save_quad_edge_route(c_path)
        self.context["step7_c_edge_route_band"] = self._load_c_edge_route_band_from_dir(str(c_path))
        return route

    def _build_and_save_sam2_outer_edge_route(self, c_path: Path, prefer_sam2_mask: bool = True) -> List[Tuple[float, float]]:
        """从 C 原始 SAM2 mask 的最大外轮廓像素生成并保存 Step7 路线。"""
        c_path = Path(c_path)
        mask_candidates: List[Path] = []
        if prefer_sam2_mask:
            mask_candidates.extend([
                c_path / "static_c_sam2_mask.png",
                c_path / "static_c_raw_mask.png",
                c_path / "static_c_original_mask.png",
            ])
        mask_candidates.append(c_path / "static_c_mask.png")

        mask_bool = None
        mask_path_used: Optional[Path] = None
        for mp in mask_candidates:
            if not mp.exists():
                continue
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if m is not None and int(np.count_nonzero(m > 0)) > 0:
                mask_bool = (m > 0)
                mask_path_used = mp
                break

        if mask_bool is None:
            return []

        spacing = float(getattr(self.cfg, "rule_ab_c_edge_route_spacing_px", 12.0))
        clearance = float(getattr(self.cfg, "rule_ab_c_edge_route_safe_clearance_px", getattr(self.cfg, "rule_ab_ac_target_clearance_px", 90.0)))
        follow = int(getattr(self.cfg, "rule_ab_follow_c_direction", 1))
        route, contour_points = self._generate_c_edge_route_points_from_mask_outer_edge(
            mask_bool,
            spacing_px=spacing,
            follow_direction=follow,
            safe_clearance_px=clearance,
            image_shape=mask_bool.shape[:2],
        )
        if not route:
            return []

        out_path = c_path / "static_c_edge_route.json"
        csv_path = c_path / "static_c_edge_route.csv"
        contour_json_path = c_path / "static_c_outer_edge_pixels.json"
        contour_csv_path = c_path / "static_c_outer_edge_pixels.csv"

        with out_path.open("w", encoding="utf-8") as f:
            json.dump({
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source": "sam2_outer_edge_pixels" if prefer_sam2_mask else "mask_outer_edge_pixels_fallback",
                "source_mask_path": str(mask_path_used.resolve()) if mask_path_used else "",
                "spacing_px": spacing,
                "safe_clearance_px": clearance,
                "follow_direction": follow,
                "follow_direction_name": "顺时针" if int(follow) < 0 else "逆时针",
                "route_loop": bool(getattr(self.cfg, "rule_ab_route_loop", True)),
                "route_target_tolerance_px": float(getattr(self.cfg, "rule_ab_route_target_tolerance_px", 8.0)),
                "outer_edge_point_count": len(contour_points),
                "route_points_xy": [[float(x), float(y)] for x, y in route],
            }, f, ensure_ascii=False, indent=2)
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "x", "y"])
            for idx, (x, y) in enumerate(route):
                writer.writerow([idx, float(x), float(y)])
        with contour_json_path.open("w", encoding="utf-8") as f:
            json.dump({
                "source_mask_path": str(mask_path_used.resolve()) if mask_path_used else "",
                "points_xy": [[float(x), float(y)] for x, y in contour_points],
            }, f, ensure_ascii=False, indent=2)
        with contour_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "x", "y"])
            for idx, (x, y) in enumerate(contour_points):
                writer.writerow([idx, float(x), float(y)])
        return route

    @staticmethod
    def _get_mask_contour_for_shape_judge(mask_bool: np.ndarray) -> Optional[np.ndarray]:
        """取 mask 最大外轮廓；用于判断 C 是否近似圆形。"""
        try:
            m = np.asarray(mask_bool).astype(bool)
            if m.size <= 0 or not np.any(m):
                return None
            contours = MeasurementWorkflow._find_contours_compat((m.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
            contour = max(contours, key=cv2.contourArea)
            if contour is None or len(contour) < 5:
                return None
            return contour
        except Exception:
            return None

    @staticmethod
    def _judge_c_mask_circle_like(mask_bool: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        """
        判断 C mask 是否近似圆形。

        判据尽量保守：
            1) circularity = 4*pi*area/perimeter^2 较高；
            2) 外接框宽高比接近 1；
            3) mask 面积占最小外接圆面积比例不能太低；
            4) approxPolyDP 在较小 epsilon 下不是稳定 4 点多边形。

        返回：
            is_circle_like, debug_info
        """
        debug: Dict[str, Any] = {
            "is_circle_like": False,
            "reason": "not_checked",
        }
        contour = MeasurementWorkflow._get_mask_contour_for_shape_judge(mask_bool)
        if contour is None:
            debug["reason"] = "no_valid_contour"
            return False, debug

        area = float(cv2.contourArea(contour))
        peri = float(cv2.arcLength(contour, True))
        if area <= 1.0 or peri <= 1e-6:
            debug.update({"reason": "area_or_perimeter_too_small", "area": area, "perimeter": peri})
            return False, debug

        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = float(max(1, int(w) * int(h)))
        bbox_aspect = float(w) / float(max(1, h))
        bbox_aspect_norm = min(bbox_aspect, 1.0 / max(bbox_aspect, 1e-6))
        circularity = float(4.0 * math.pi * area / max(peri * peri, 1e-6))

        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        circle_area = float(math.pi * max(float(radius), 1e-6) ** 2)
        circle_fill = float(area / max(circle_area, 1e-6))
        bbox_fill = float(area / max(bbox_area, 1e-6))

        approx_vertex_count = -1
        try:
            eps = max(1.0, 0.02 * peri)
            approx = cv2.approxPolyDP(contour, eps, True)
            approx_vertex_count = int(len(approx))
        except Exception:
            approx_vertex_count = -1

        # 圆形/近圆形：圆度高、宽高接近、外接圆填充合理。
        # 对正方形，circularity≈0.785，通常不会通过 0.82 阈值；
        # 对明显椭圆，bbox_aspect_norm 会降低。
        is_circle_like = (
            circularity >= 0.82
            and bbox_aspect_norm >= 0.72
            and circle_fill >= 0.58
            and bbox_fill >= 0.45
            and approx_vertex_count != 4
        )
        debug.update({
            "is_circle_like": bool(is_circle_like),
            "reason": "circle_like" if is_circle_like else "not_circle_like",
            "area": area,
            "perimeter": peri,
            "circularity": circularity,
            "bbox_xywh": [int(x), int(y), int(w), int(h)],
            "bbox_aspect": bbox_aspect,
            "bbox_aspect_norm": bbox_aspect_norm,
            "bbox_fill": bbox_fill,
            "min_enclosing_circle": [float(cx), float(cy), float(radius)],
            "circle_fill": circle_fill,
            "approx_vertex_count_eps_0p02": approx_vertex_count,
        })
        return bool(is_circle_like), debug

    @staticmethod
    def _build_external_bbox_quad_from_mask(mask_bool: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        对近似圆形的 C mask 生成外接矩形/四边形。

        这里使用图像坐标轴对齐的 boundingRect，得到圆形的稳定外接矩形路径：
            左上 -> 右上 -> 右下 -> 左下
        返回 quad_points、quad_mask、debug。
        """
        m = np.asarray(mask_bool).astype(bool)
        if m.size <= 0 or not np.any(m):
            raise RuntimeError("C mask 为空，不能生成圆形外接矩形路径。")
        h_img, w_img = m.shape[:2]
        ys, xs = np.where(m)
        x0 = int(np.clip(xs.min(), 0, max(0, w_img - 1)))
        x1 = int(np.clip(xs.max(), 0, max(0, w_img - 1)))
        y0 = int(np.clip(ys.min(), 0, max(0, h_img - 1)))
        y1 = int(np.clip(ys.max(), 0, max(0, h_img - 1)))
        if x1 <= x0 or y1 <= y0:
            raise RuntimeError(f"C mask 外接矩形退化：x0={x0}, x1={x1}, y0={y0}, y1={y1}")
        quad = np.asarray(
            [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
            dtype=np.float32,
        )
        quad_mask_u8 = np.zeros((h_img, w_img), dtype=np.uint8)
        quad_i32 = np.round(quad).astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(quad_mask_u8, [quad_i32], 255)
        debug = {
            "bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
            "bbox_width_px": int(x1 - x0 + 1),
            "bbox_height_px": int(y1 - y0 + 1),
            "bbox_area_px": int((x1 - x0 + 1) * (y1 - y0 + 1)),
            "mask_area_px": int(np.count_nonzero(m)),
        }
        return quad, quad_mask_u8.astype(bool), debug

    @staticmethod
    def _labels_near_prompt_point(labels_im: np.ndarray, x: float, y: float, radius_px: int = 5) -> Dict[int, int]:
        """返回某个 prompt 点附近半径内出现的连通域 label 计数。"""
        out: Dict[int, int] = {}
        try:
            h, w = labels_im.shape[:2]
            cx = int(round(float(x)))
            cy = int(round(float(y)))
            r = max(0, int(radius_px))
            x0 = max(0, cx - r)
            x1 = min(w - 1, cx + r)
            y0 = max(0, cy - r)
            y1 = min(h - 1, cy + r)
            if x1 < x0 or y1 < y0:
                return out
            patch = labels_im[y0:y1 + 1, x0:x1 + 1]
            vals, counts = np.unique(patch, return_counts=True)
            for v, c in zip(vals, counts):
                iv = int(v)
                if iv <= 0:
                    continue
                out[iv] = int(c)
        except Exception:
            return {}
        return out

    @staticmethod
    def _clean_c_mask_by_prompt_connected_component(
        mask_bool: np.ndarray,
        positive_points: Optional[List[Tuple[float, float]]] = None,
        negative_points: Optional[List[Tuple[float, float]]] = None,
        point_radius_px: int = 5,
        min_component_area_px: int = 8,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        用 C 的正/负 prompt 点清理 SAM2 返回的 C mask。

        目的：
            SAM2 可能把蓝色正点附近的目标和旁边粘连的结构一起分出来。
            这里先做连通域分析，只保留“命中正点最多、命中负点最少”的主体连通域。

        说明：
            - 蓝色/正点只用于选择主体连通域，不改变后续路线/安全带逻辑；
            - 如果 C mask 只有一个连通域，则不能靠连通域切开粘连区域，会保留原 mask 并记录原因；
            - 如果没有任何正点落在 mask 连通域上，也保留原 mask，避免误删 C。
        """
        m = np.asarray(mask_bool).astype(bool)
        debug: Dict[str, Any] = {
            "enabled": True,
            "reason": "not_checked",
            "input_area_px": int(np.count_nonzero(m)),
            "output_area_px": int(np.count_nonzero(m)),
            "positive_point_count": int(len(positive_points or [])),
            "negative_point_count": int(len(negative_points or [])),
        }
        if m.size <= 0 or not np.any(m):
            debug["reason"] = "empty_mask"
            return m, debug

        try:
            num_labels, labels_im, stats, _centroids = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
        except Exception as e:
            debug["reason"] = f"connected_components_failed:{e}"
            return m, debug

        components: List[Dict[str, Any]] = []
        for lab in range(1, int(num_labels)):
            area = int(stats[lab, cv2.CC_STAT_AREA]) if stats is not None else int(np.count_nonzero(labels_im == lab))
            if area < int(min_component_area_px):
                continue
            components.append({
                "label": int(lab),
                "area_px": int(area),
                "pos_hits": 0,
                "neg_hits": 0,
                "pos_near_pixels": 0,
                "neg_near_pixels": 0,
                "score": 0.0,
            })
        debug["component_count"] = int(len(components))
        if not components:
            debug["reason"] = "no_valid_component"
            return m, debug

        comp_by_label = {int(c["label"]): c for c in components}
        for x, y in (positive_points or []):
            near = MeasurementWorkflow._labels_near_prompt_point(labels_im, float(x), float(y), int(point_radius_px))
            if not near:
                continue
            # 一个正点只给附近像素最多的那个 label 记一次命中，避免一个点同时奖励多个粘连边缘。
            lab = max(near.items(), key=lambda kv: kv[1])[0]
            if lab in comp_by_label:
                comp_by_label[lab]["pos_hits"] += 1
                comp_by_label[lab]["pos_near_pixels"] += int(near.get(lab, 0))
        for x, y in (negative_points or []):
            near = MeasurementWorkflow._labels_near_prompt_point(labels_im, float(x), float(y), int(point_radius_px))
            if not near:
                continue
            lab = max(near.items(), key=lambda kv: kv[1])[0]
            if lab in comp_by_label:
                comp_by_label[lab]["neg_hits"] += 1
                comp_by_label[lab]["neg_near_pixels"] += int(near.get(lab, 0))

        for c in components:
            # 正点命中是最高优先级；负点命中强惩罚；面积仅作为同分时的弱兜底。
            c["score"] = (
                float(c["pos_hits"]) * 100000.0
                + float(c["pos_near_pixels"]) * 100.0
                + math.log1p(float(c["area_px"]))
                - float(c["neg_hits"]) * 200000.0
                - float(c["neg_near_pixels"]) * 100.0
            )

        components_sorted = sorted(components, key=lambda d: float(d.get("score", -1e18)), reverse=True)
        debug["components"] = components_sorted[:12]
        best = components_sorted[0]
        debug["selected_component"] = dict(best)

        if int(best.get("pos_hits", 0)) <= 0 and positive_points:
            debug["reason"] = "no_positive_prompt_hit_any_component_keep_original"
            return m, debug

        cleaned = labels_im == int(best["label"])
        if int(np.count_nonzero(cleaned)) <= 0:
            debug["reason"] = "selected_component_empty_keep_original"
            return m, debug

        debug["reason"] = "kept_component_with_most_positive_prompts"
        debug["output_area_px"] = int(np.count_nonzero(cleaned))
        debug["area_ratio_output_over_input"] = float(debug["output_area_px"] / max(1, debug["input_area_px"]))
        return cleaned.astype(bool), debug

    @staticmethod
    def _select_and_clean_c_mask_from_sam2_candidates(
        masks: Any,
        scores: Any,
        positive_points: List[Tuple[float, float]],
        negative_points: List[Tuple[float, float]],
    ) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        """
        从 SAM2 multimask 输出中选择最符合 C 正点/负点的候选，并进行连通域清理。
        """
        scores_arr = np.asarray(scores).reshape(-1) if scores is not None else np.zeros((len(masks),), dtype=np.float32)
        candidates: List[Dict[str, Any]] = []
        best_mask: Optional[np.ndarray] = None
        best_idx = 0
        best_total = -1e30
        for idx0, cand in enumerate(masks):
            raw = np.asarray(cand).astype(bool)
            if int(np.count_nonzero(raw)) <= 0:
                continue
            cleaned, clean_debug = MeasurementWorkflow._clean_c_mask_by_prompt_connected_component(
                raw,
                positive_points=positive_points,
                negative_points=negative_points,
                point_radius_px=5,
                min_component_area_px=8,
            )
            selected = clean_debug.get("selected_component") or {}
            pos_hits = int(selected.get("pos_hits", 0)) if isinstance(selected, dict) else 0
            neg_hits = int(selected.get("neg_hits", 0)) if isinstance(selected, dict) else 0
            pos_near = int(selected.get("pos_near_pixels", 0)) if isinstance(selected, dict) else 0
            neg_near = int(selected.get("neg_near_pixels", 0)) if isinstance(selected, dict) else 0
            area_px = int(np.count_nonzero(cleaned))
            sam_score = float(scores_arr[idx0]) if idx0 < len(scores_arr) else 0.0
            total = (
                float(pos_hits) * 100000.0
                + float(pos_near) * 100.0
                + float(sam_score) * 10.0
                + math.log1p(float(area_px))
                - float(neg_hits) * 200000.0
                - float(neg_near) * 100.0
            )
            item = {
                "candidate_index": int(idx0),
                "sam2_score": sam_score,
                "raw_area_px": int(np.count_nonzero(raw)),
                "cleaned_area_px": int(area_px),
                "pos_hits": int(pos_hits),
                "neg_hits": int(neg_hits),
                "pos_near_pixels": int(pos_near),
                "neg_near_pixels": int(neg_near),
                "total_score": float(total),
                "clean_reason": str(clean_debug.get("reason", "")),
            }
            candidates.append(item)
            if total > best_total:
                best_total = total
                best_idx = int(idx0)
                best_mask = cleaned.astype(bool)
                best_debug = clean_debug

        if best_mask is None:
            # 兜底回到 SAM2 原始分数最高的候选。
            idx_fallback = int(np.argmax(scores_arr)) if len(scores_arr) else 0
            raw = np.asarray(masks[idx_fallback]).astype(bool)
            cleaned, best_debug = MeasurementWorkflow._clean_c_mask_by_prompt_connected_component(
                raw,
                positive_points=positive_points,
                negative_points=negative_points,
                point_radius_px=5,
                min_component_area_px=8,
            )
            best_mask = cleaned.astype(bool)
            best_idx = idx_fallback
            best_total = float(scores_arr[idx_fallback]) if len(scores_arr) else 0.0

        debug: Dict[str, Any] = {
            "selected_index": int(best_idx),
            "selected_total_score": float(best_total),
            "selected_sam2_score": float(scores_arr[best_idx]) if best_idx < len(scores_arr) else None,
            "candidates": sorted(candidates, key=lambda d: float(d.get("total_score", -1e18)), reverse=True),
            "component_clean_debug": best_debug if 'best_debug' in locals() else {},
        }
        return best_mask.astype(bool), int(best_idx), debug

    def _load_preferred_c_mask_for_route_shape(self, c_path: Path) -> Tuple[Optional[np.ndarray], str]:
        """路线生成时优先使用原始 SAM2 C mask 判断形状；没有时再用 static_c_mask.png 兜底。"""
        c_path = Path(c_path)
        for mp in (
            c_path / "static_c_sam2_mask.png",
            c_path / "static_c_raw_mask.png",
            c_path / "static_c_original_mask.png",
            c_path / "static_c_mask.png",
        ):
            if not mp.exists():
                continue
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if m is not None and int(np.count_nonzero(m > 0)) > 0:
                return (m > 0), str(mp)
        return None, ""

    @staticmethod
    def _fit_circle_from_mask(mask_bool: np.ndarray) -> Optional[Dict[str, Any]]:
        """从 C mask 最大外轮廓拟合圆，返回圆心和半径。"""
        m = np.asarray(mask_bool).astype(np.uint8)
        if m.size <= 0 or int(np.count_nonzero(m)) <= 0:
            return None
        cnts, _ = cv2.findContours((m > 0).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            return None
        cnt = max(cnts, key=cv2.contourArea)
        area = float(cv2.contourArea(cnt))
        if area <= 1.0 or len(cnt) < 5:
            return None
        (cx, cy), radius = cv2.minEnclosingCircle(cnt.astype(np.float32))
        peri = float(cv2.arcLength(cnt, True))
        circularity = float(4.0 * math.pi * area / max(peri * peri, 1e-9)) if peri > 0 else 0.0
        return {
            "center_xy": [float(cx), float(cy)],
            "radius_px": float(radius),
            "area_px": float(area),
            "perimeter_px": float(peri),
            "circularity": float(circularity),
            "point_count": int(len(cnt)),
        }

    @staticmethod
    def _generate_octagon_route_points(
        center_xy: Tuple[float, float],
        radius_px: float,
        follow_direction: int = 1,
        image_shape: Optional[Tuple[int, int]] = None,
        start_angle_deg: float = 22.5,
    ) -> List[Tuple[float, float]]:
        """生成 8 个八边形顶点。offset=22.5deg 时包含水平/垂直边和斜边。"""
        cx, cy = float(center_xy[0]), float(center_xy[1])
        r = max(1.0, float(radius_px))
        # 图像坐标 y 向下；与圆路线保持一致：follow>=0 视觉逆时针。
        sign = -1.0 if int(follow_direction) >= 0 else 1.0
        h = w = None
        if image_shape is not None and len(image_shape) >= 2:
            h, w = int(image_shape[0]), int(image_shape[1])
        pts: List[Tuple[float, float]] = []
        start = math.radians(float(start_angle_deg))
        for i in range(8):
            theta = start + sign * (2.0 * math.pi * float(i) / 8.0)
            x = cx + r * math.cos(theta)
            y = cy + r * math.sin(theta)
            if w is not None and h is not None:
                x = max(0.0, min(float(w - 1), float(x)))
                y = max(0.0, min(float(h - 1), float(y)))
            pts.append((float(x), float(y)))
        return pts

    def _build_and_save_circle_edge_route(self, c_path: Path) -> List[Tuple[float, float]]:
        """
        从 C mask 拟合圆并保存 Step7 圆周路线。

        注意：本模式下 rule_ab_route_min/safe_target/max_distance_px 表示
        A center 到 C 圆心的半径距离，而不是 A 到 C 边界或 A 到路线最近点的距离。
        """
        c_path = Path(c_path)
        mask_bool = None
        mask_path_used: Optional[Path] = None
        try:
            mask_bool, mask_path_used = self._load_preferred_c_mask_for_route_shape(c_path)
            if mask_path_used:
                mask_path_used = Path(mask_path_used)
        except Exception:
            mask_bool = None
            mask_path_used = None
        if mask_bool is None:
            for mp in [c_path / "static_c_sam2_mask.png", c_path / "static_c_raw_mask.png", c_path / "static_c_original_mask.png", c_path / "static_c_mask.png"]:
                if not mp.exists():
                    continue
                m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                if m is not None and int(np.count_nonzero(m > 0)) > 0:
                    mask_bool = (m > 0)
                    mask_path_used = mp
                    break
        if mask_bool is None:
            return []

        circle = self._fit_circle_from_mask(mask_bool)
        if not circle:
            return []
        cx, cy = float(circle["center_xy"][0]), float(circle["center_xy"][1])
        c_radius = float(circle["radius_px"])

        spacing = float(getattr(self.cfg, "rule_ab_c_edge_route_spacing_px", 12.0))
        follow = int(getattr(self.cfg, "rule_ab_follow_c_direction", 1))
        min_d = max(0.0, float(getattr(self.cfg, "rule_ab_route_min_distance_px", getattr(self.cfg, "rule_ab_ac_min_clearance_px", 80.0))))
        max_d = max(min_d, float(getattr(self.cfg, "rule_ab_route_max_distance_px", getattr(self.cfg, "rule_ab_ac_max_clearance_px", 100.0))))
        target_d = float(getattr(self.cfg, "rule_ab_route_safe_target_distance_px", getattr(self.cfg, "rule_ab_c_edge_route_safe_clearance_px", getattr(self.cfg, "rule_ab_ac_target_clearance_px", 90.0))))
        if target_d < 0:
            target_d = 0.5 * (min_d + max_d)
        target_d = max(min_d, min(max_d, target_d))

        image_shape = mask_bool.shape[:2]
        oct_start = float(getattr(self.cfg, "rule_ab_octagon_start_angle_deg", 22.5))
        base_route = self._generate_octagon_route_points((cx, cy), c_radius, follow, image_shape=image_shape, start_angle_deg=oct_start)
        min_route = self._generate_octagon_route_points((cx, cy), min_d, follow, image_shape=image_shape, start_angle_deg=oct_start)
        route = self._generate_octagon_route_points((cx, cy), target_d, follow, image_shape=image_shape, start_angle_deg=oct_start)
        max_route = self._generate_octagon_route_points((cx, cy), max_d, follow, image_shape=image_shape, start_angle_deg=oct_start)
        if not route:
            return []

        out_path = c_path / "static_c_edge_route.json"
        csv_path = c_path / "static_c_edge_route.csv"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source": "circle_fit_center_radius_route",
                "source_mask": str(mask_path_used.resolve()) if mask_path_used else "",
                "safety_reference": "circle_center_radius",
                "route_geometry_mode": "circle",
                "execution_route_mode": "octagon_state_machine",
                "route_shape": "octagon",
                "octagon_vertex_count": 8,
                "octagon_start_angle_deg": float(oct_start),
                "circle_center_xy": [float(cx), float(cy)],
                "circle_base_radius_px": float(c_radius),
                "circle_fit_debug": circle,
                "c_mask_shape_for_route": "fit_circle",
                "spacing_px": float(spacing),
                "safe_clearance_px": float(target_d),
                "min_distance_px": float(min_d),
                "target_distance_px": float(target_d),
                "max_distance_px": float(max_d),
                "follow_direction": int(follow),
                "follow_direction_name": "顺时针" if int(follow) < 0 else "逆时针",
                "route_loop": bool(getattr(self.cfg, "rule_ab_route_loop", True)),
                "route_target_tolerance_px": float(getattr(self.cfg, "rule_ab_route_target_tolerance_px", 8.0)),
                "base_route_points_xy": [[float(x), float(y)] for x, y in base_route],
                "min_route_points_xy": [[float(x), float(y)] for x, y in min_route],
                "route_points_xy": [[float(x), float(y)] for x, y in route],
                "target_route_points_xy": [[float(x), float(y)] for x, y in route],
                "max_route_points_xy": [[float(x), float(y)] for x, y in max_route],
                "execution_base_route_points_xy": [[float(x), float(y)] for x, y in base_route],
                "execution_min_route_points_xy": [[float(x), float(y)] for x, y in min_route],
                "execution_route_points_xy": [[float(x), float(y)] for x, y in route],
                "execution_target_route_points_xy": [[float(x), float(y)] for x, y in route],
                "execution_max_route_points_xy": [[float(x), float(y)] for x, y in max_route],
                "octagon_diagonal_chunk_steps": int(getattr(self.cfg, "rule_ab_octagon_diagonal_chunk_steps", 3)),
                "note": "Step7 八边形状态机路线：route_points_xy 为 8 个八边形关键点；水平段只左右、垂直段只上下、斜边按 X/Y 方向分块交替近似。安全距离判据为 distance(A_center, circle_center)。",
            }, f, ensure_ascii=False, indent=2)
        try:
            with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["index", "x", "y"])
                for idx0, (x, y) in enumerate(route):
                    writer.writerow([idx0, float(x), float(y)])
        except Exception:
            pass
        self.log(
            f"[RuleAB-Route] 已生成 Step7 八边形状态机路线：center=({cx:.2f},{cy:.2f}), "
            f"C_radius={c_radius:.2f}px, octagon_radius={target_d:.2f}px, "
            f"safe_band_radius=[{min_d:.2f},{max_d:.2f}], points={len(route)}, path={out_path}"
        )
        return route

    def _build_and_save_quad_edge_route(self, c_path: Path) -> List[Tuple[float, float]]:
        """
        从 C 几何生成 Step7 三线安全带路线。

        新增圆形 C mask 分支：
            - 优先读取 static_c_sam2_mask.png 判断原始 C 是否近似圆形；
            - 如果近似圆形，则不再用 approxPolyDP 生成不稳定四边形，
              而是使用 C 圆形 mask 的外接矩形/四边形作为路线基准；
            - 如果不是圆形，则保持旧逻辑：static_c_quad_points.json 或 static_c_mask.png 四边形兜底。
        """
        c_path = Path(c_path)

        circle_like = False
        circle_debug: Dict[str, Any] = {}
        circle_bbox_debug: Dict[str, Any] = {}
        circle_mask_path_used = ""
        circle_quad: Optional[np.ndarray] = None

        # 先判断原始 C mask 是否近似圆形。即使已有旧的 static_c_quad_points.json，
        # 只要原始 SAM2 C 是圆形，也强制使用圆形外接矩形生成路径。
        try:
            mask_for_shape, mask_path_used = self._load_preferred_c_mask_for_route_shape(c_path)
            if mask_for_shape is not None:
                circle_like, circle_debug = MeasurementWorkflow._judge_c_mask_circle_like(mask_for_shape)
                circle_mask_path_used = str(mask_path_used)
                if circle_like:
                    circle_quad, _circle_quad_mask, circle_bbox_debug = MeasurementWorkflow._build_external_bbox_quad_from_mask(mask_for_shape)
                    with (c_path / "static_c_circle_bbox_points.json").open("w", encoding="utf-8") as f:
                        json.dump({
                            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "method": "circle_like_external_axis_aligned_bbox",
                            "source_mask_path": circle_mask_path_used,
                            "circle_debug": circle_debug,
                            "bbox_debug": circle_bbox_debug,
                            "points_xy": [[float(x), float(y)] for x, y in circle_quad.tolist()],
                        }, f, ensure_ascii=False, indent=2)
                    self.log(
                        "[RuleAB-Route] 检测到 C mask 近似圆形，Step7 路线改用圆形外接矩形/四边形："
                        f"circularity={float(circle_debug.get('circularity', 0.0)):.3f}, "
                        f"bbox={circle_bbox_debug.get('bbox_xyxy')}, source={circle_mask_path_used}"
                    )
        except Exception as e:
            circle_like = False
            self.log(f"[RuleAB-Route] C 圆形判断失败，继续使用原四边形路线逻辑：{e}")

        quad_path = c_path / "static_c_quad_points.json"
        if not circle_like and not quad_path.exists():
            meta_path = c_path / "static_c_meta.json"
            if meta_path.exists():
                try:
                    with meta_path.open("r", encoding="utf-8") as f:
                        meta = json.load(f)
                    pts = meta.get("quad_points_xy") or meta.get("contour") or []
                    if pts:
                        with quad_path.open("w", encoding="utf-8") as f:
                            json.dump({"points_xy": pts}, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
        if not circle_like and not quad_path.exists():
            mask_path = c_path / "static_c_mask.png"
            if mask_path.exists():
                m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if m is not None and int(np.count_nonzero(m > 0)) > 0:
                    try:
                        quad, _quad_mask, method = MeasurementWorkflowGUI._fit_quadrilateral_from_mask(m > 0)
                        with quad_path.open("w", encoding="utf-8") as f:
                            json.dump({"method": method, "points_xy": [[float(x), float(y)] for x, y in quad.tolist()]}, f, ensure_ascii=False, indent=2)
                    except Exception:
                        pass
        if not circle_like and not quad_path.exists():
            return []
        try:
            if circle_like and circle_quad is not None:
                quad = np.asarray(circle_quad, dtype=np.float32).reshape(4, 2)
                route_payload_source = "circle_bbox_quad_points_three_line_safety_band"
                route_source_mask = circle_mask_path_used
                route_shape_note = "检测到 C mask 近似圆形：Step7 不沿圆周走，而是沿圆形外接矩形/四边形路径走。"
            else:
                with quad_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                quad = np.asarray(data.get("points_xy") or data.get("quad_points_xy"), dtype=np.float32).reshape(-1, 2)
                if quad.shape[0] < 4:
                    return []
                quad = quad[:4]
                route_payload_source = "quad_points_three_line_safety_band"
                route_source_mask = str(quad_path)
                route_shape_note = "使用 C 近似四边形四条边生成 Step7 路线。"

            spacing = float(getattr(self.cfg, "rule_ab_c_edge_route_spacing_px", 12.0))
            follow = int(getattr(self.cfg, "rule_ab_follow_c_direction", 1))
            min_d = max(0.0, float(getattr(self.cfg, "rule_ab_route_min_distance_px", getattr(self.cfg, "rule_ab_ac_min_clearance_px", 80.0))))
            max_d = max(min_d, float(getattr(self.cfg, "rule_ab_route_max_distance_px", getattr(self.cfg, "rule_ab_ac_max_clearance_px", 100.0))))
            target_d = float(getattr(self.cfg, "rule_ab_route_safe_target_distance_px", getattr(self.cfg, "rule_ab_c_edge_route_safe_clearance_px", getattr(self.cfg, "rule_ab_ac_target_clearance_px", 90.0))))
            if target_d < 0:
                target_d = 0.5 * (min_d + max_d)
            target_d = max(min_d, min(max_d, target_d))

            base_route = self._generate_c_edge_route_points_from_quad(
                quad,
                spacing_px=spacing,
                follow_direction=follow,
                safe_clearance_px=0.0,
            )
            min_route = self._generate_c_edge_route_points_from_quad(
                quad,
                spacing_px=spacing,
                follow_direction=follow,
                safe_clearance_px=min_d,
            )
            route = self._generate_c_edge_route_points_from_quad(
                quad,
                spacing_px=spacing,
                follow_direction=follow,
                safe_clearance_px=target_d,
            )
            max_route = self._generate_c_edge_route_points_from_quad(
                quad,
                spacing_px=spacing,
                follow_direction=follow,
                safe_clearance_px=max_d,
            )

            with (c_path / "static_c_edge_route.json").open("w", encoding="utf-8") as f:
                json.dump({
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "source": route_payload_source,
                    "source_mask": route_source_mask,
                    "c_mask_shape_for_route": "circle_like_external_bbox" if circle_like else "quadrilateral",
                    "circle_like": bool(circle_like),
                    "circle_debug": circle_debug,
                    "circle_bbox_debug": circle_bbox_debug,
                    "spacing_px": spacing,
                    "safe_clearance_px": target_d,
                    "min_distance_px": min_d,
                    "target_distance_px": target_d,
                    "max_distance_px": max_d,
                    "follow_direction": follow,
                    "follow_direction_name": "顺时针" if int(follow) < 0 else "逆时针",
                    "route_loop": bool(getattr(self.cfg, "rule_ab_route_loop", True)),
                    "route_target_tolerance_px": float(getattr(self.cfg, "rule_ab_route_target_tolerance_px", 8.0)),
                    "quad_points_xy": [[float(x), float(y)] for x, y in quad.tolist()],
                    "base_route_points_xy": [[float(x), float(y)] for x, y in base_route],
                    "min_route_points_xy": [[float(x), float(y)] for x, y in min_route],
                    "route_points_xy": [[float(x), float(y)] for x, y in route],
                    "target_route_points_xy": [[float(x), float(y)] for x, y in route],
                    "max_route_points_xy": [[float(x), float(y)] for x, y in max_route],
                    "note": route_shape_note + " Step7 三线安全带：min/max 是边界线，target 是中间方向参考线；A 在 min-max 之间时只沿 target 切向运动，越界才法向修正。",
                }, f, ensure_ascii=False, indent=2)

            # CSV 仍保存中间目标线，便于兼容旧检查脚本。
            try:
                with (c_path / "static_c_edge_route.csv").open("w", encoding="utf-8-sig", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["index", "x", "y"])
                    for idx0, (x, y) in enumerate(route):
                        writer.writerow([idx0, float(x), float(y)])
            except Exception:
                pass
            return route
        except Exception as e:
            self.log(f"[RuleAB-Route] 从 C 四边形/圆形外接矩形重建三线安全带路线失败：{e}")
            return []

    @staticmethod
    def _resample_closed_polyline_points(points_xy: np.ndarray, spacing_px: float) -> List[Tuple[float, float]]:
        """把闭合折线/轮廓按近似等间距重采样。"""
        pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
        if len(pts) == 0:
            return []
        if len(pts) == 1:
            return [(float(pts[0, 0]), float(pts[0, 1]))]
        spacing = max(1.0, float(spacing_px))
        closed = np.vstack([pts, pts[0:1]])
        seg = closed[1:] - closed[:-1]
        lens = np.linalg.norm(seg, axis=1)
        total = float(np.sum(lens))
        if total <= 1e-6:
            return [(float(pts[0, 0]), float(pts[0, 1]))]
        n_samples = max(3, int(math.ceil(total / spacing)))
        targets = np.linspace(0.0, total, n_samples, endpoint=False)
        cum = np.concatenate([[0.0], np.cumsum(lens)])
        out: List[Tuple[float, float]] = []
        j = 0
        for d in targets:
            while j < len(lens) - 1 and cum[j + 1] < d:
                j += 1
            length = float(lens[j])
            if length <= 1e-9:
                p = closed[j].copy()
            else:
                t = (float(d) - float(cum[j])) / length
                p = closed[j] * (1.0 - t) + closed[j + 1] * t
            out.append((float(p[0]), float(p[1])))
        return out

    @staticmethod
    def _generate_c_edge_route_points_from_mask_outer_edge(
        mask_bool: np.ndarray,
        spacing_px: float = 12.0,
        follow_direction: int = 1,
        safe_clearance_px: float = 0.0,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        根据 C 原始 mask 最大外轮廓的边缘像素点生成 A 的闭合运动路线。

        contour_points 是 C mask 外侧边界像素点；route_points 是把 contour_points
        按 spacing_px 重采样后，再沿远离 mask 质心方向偏移 safe_clearance_px 得到的 A 路线。
        """
        m = np.asarray(mask_bool).astype(bool)
        if m.size <= 0 or not np.any(m):
            return [], []
        h, w = m.shape[:2]
        contours = self._find_contours_compat((m.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return [], []
        contour = max(contours, key=cv2.contourArea)
        contour_pts = contour.reshape(-1, 2).astype(np.float32)
        if len(contour_pts) < 3:
            return [], []

        # 统一方向；OpenCV 图像坐标下 contourArea(oriented=True) 的符号只用于一致化顺序。
        if int(follow_direction) < 0:
            contour_pts = contour_pts[::-1].copy()

        mmt = cv2.moments((m.astype(np.uint8) * 255))
        if abs(float(mmt.get("m00", 0.0))) > 1e-8:
            center = np.array([float(mmt["m10"] / mmt["m00"]), float(mmt["m01"] / mmt["m00"])], dtype=np.float32)
        else:
            center = np.mean(contour_pts, axis=0).astype(np.float32)

        edge_points = MeasurementWorkflow._resample_closed_polyline_points(contour_pts, spacing_px)
        clearance = max(0.0, float(safe_clearance_px))
        route: List[Tuple[float, float]] = []
        for x, y in edge_points:
            pnt = np.array([float(x), float(y)], dtype=np.float32)
            v = pnt - center
            norm = float(np.linalg.norm(v))
            if norm <= 1e-6:
                # 极少数退化点：用局部切线法线兜底。
                v = np.array([0.0, -1.0], dtype=np.float32)
                norm = 1.0
            outward = v / norm
            q = pnt + clearance * outward
            if image_shape is not None:
                hh, ww = int(image_shape[0]), int(image_shape[1])
                q[0] = np.clip(q[0], 0, max(0, ww - 1))
                q[1] = np.clip(q[1], 0, max(0, hh - 1))
            route.append((float(q[0]), float(q[1])))

        cleaned: List[Tuple[float, float]] = []
        for x, y in route:
            if not cleaned or math.hypot(float(x) - cleaned[-1][0], float(y) - cleaned[-1][1]) >= 0.5:
                cleaned.append((float(x), float(y)))
        contour_list = [(float(x), float(y)) for x, y in edge_points]
        return cleaned, contour_list

    @staticmethod
    def _generate_c_edge_route_points_from_quad(
        quad_points: np.ndarray,
        spacing_px: float = 12.0,
        follow_direction: int = 1,
        safe_clearance_px: float = 0.0,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> List[Tuple[float, float]]:
        """
        根据 C 四边形四条边生成 A 的闭合运动路线。

        注意：这里生成的不是 C 的边界线本身，而是从 C 边界向外法线方向偏移
        safe_clearance_px 后的 A 目标路线。这样 A 按路线运动时会与 C 保持安全距离。
        """
        pts = np.asarray(quad_points, dtype=np.float32).reshape(4, 2)

        # 图像坐标系 y 轴向下，普通几何坐标里的顺/逆时针符号与屏幕视觉方向相反。
        # 约定：follow_direction >= 0 表示屏幕上逆时针绕行；follow_direction < 0 表示屏幕上顺时针绕行。
        # 对于图像坐标，shoelace 面积 < 0 才是屏幕视觉逆时针。
        try:
            signed_area = 0.5 * float(np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - pts[:, 1] * np.roll(pts[:, 0], -1)))
            is_screen_ccw = signed_area < 0.0
            want_screen_ccw = int(follow_direction) >= 0
            if is_screen_ccw != want_screen_ccw:
                pts = pts[::-1].copy()
        except Exception:
            if int(follow_direction) < 0:
                pts = pts[::-1].copy()

        spacing = max(1.0, float(spacing_px))
        clearance = max(0.0, float(safe_clearance_px))
        center = np.mean(pts, axis=0).astype(np.float32)
        route: List[Tuple[float, float]] = []
        for i in range(4):
            p0 = pts[i]
            p1 = pts[(i + 1) % 4]
            edge = p1 - p0
            length = float(np.linalg.norm(edge))
            if length <= 1e-6:
                continue
            # 两个候选法线，选“远离四边形中心”的那个作为外法线。
            n1 = np.array([-edge[1], edge[0]], dtype=np.float32) / length
            n2 = -n1
            mid = 0.5 * (p0 + p1)
            normal = n1 if float(np.dot(mid + n1 - center, mid + n1 - center)) >= float(np.dot(mid + n2 - center, mid + n2 - center)) else n2
            n = max(1, int(math.ceil(length / spacing)))
            for k in range(n):
                t = float(k) / float(n)
                p = (1.0 - t) * p0 + t * p1 + clearance * normal
                if image_shape is not None:
                    h, w = int(image_shape[0]), int(image_shape[1])
                    p[0] = np.clip(p[0], 0, max(0, w - 1))
                    p[1] = np.clip(p[1], 0, max(0, h - 1))
                route.append((float(p[0]), float(p[1])))
        # 去掉连续重复点
        cleaned: List[Tuple[float, float]] = []
        for x, y in route:
            if not cleaned or math.hypot(float(x) - cleaned[-1][0], float(y) - cleaned[-1][1]) >= 0.5:
                cleaned.append((float(x), float(y)))
        return cleaned

    @staticmethod
    def _route_visual_order_info(route_points: List[Tuple[float, float]]) -> Dict[str, Any]:
        """
        判断闭合 route_points 的索引增大方向在屏幕图像坐标系下是视觉顺时针还是视觉逆时针。

        约定：
            - 图像坐标系 y 轴向下；
            - shoelace signed_area > 0 表示 route index 增大方向为 visual_CW；
            - shoelace signed_area < 0 表示 route index 增大方向为 visual_CCW。

        返回字段：
            signed_area: shoelace 有符号面积；
            route_index_order: visual_CW / visual_CCW / visual_unknown。
        """
        try:
            pts = np.asarray(route_points, dtype=np.float64).reshape(-1, 2)
            if pts.shape[0] < 3:
                return {"signed_area": 0.0, "route_index_order": "visual_unknown"}
            x = pts[:, 0]
            y = pts[:, 1]
            signed_area = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
            if signed_area > 1e-6:
                order = "visual_CW"
            elif signed_area < -1e-6:
                order = "visual_CCW"
            else:
                order = "visual_unknown"
            return {"signed_area": signed_area, "route_index_order": order}
        except Exception:
            return {"signed_area": 0.0, "route_index_order": "visual_unknown"}

    @classmethod
    def _resolve_route_actual_index_step(cls, route_points: List[Tuple[float, float]], follow_direction: int) -> Dict[str, Any]:
        """
        把 GUI 里的视觉方向语义转换为实际 route index 步进方向。

        GUI 语义保持不变：
            rule_ab_follow_c_direction = +1 -> 屏幕视觉逆时针 visual_CCW；
            rule_ab_follow_c_direction = -1 -> 屏幕视觉顺时针 visual_CW。

        route_points 自身索引顺序可能是 visual_CW，也可能是 visual_CCW，
        因此不能再直接把 +1/-1 当作实际 index step。
        """
        info = cls._route_visual_order_info(route_points)
        route_order = str(info.get("route_index_order", "visual_unknown"))
        user_desired = "visual_CCW" if int(follow_direction) >= 0 else "visual_CW"

        if route_order in ("visual_CW", "visual_CCW"):
            actual_step = 1 if route_order == user_desired else -1
            fallback = False
        else:
            # 退化路线无法可靠判断方向时，保留旧行为作为兜底，但明确标记。
            actual_step = 1 if int(follow_direction) >= 0 else -1
            fallback = True

        out = dict(info)
        out.update({
            "user_desired_direction": user_desired,
            "actual_index_step": int(actual_step),
            "direction_fallback_used": bool(fallback),
        })
        return out

    def _save_step7_route_overlay_image(
        self,
        image_rgb: np.ndarray,
        route_points: List[Tuple[float, float]],
        a_center: Optional[Tuple[float, float]],
        target_xy: Optional[Tuple[float, float]],
        route_index: int,
        step_idx: int,
        action_name: str,
        error_dist: Optional[float] = None,
        min_route_points: Optional[List[Tuple[float, float]]] = None,
        max_route_points: Optional[List[Tuple[float, float]]] = None,
        base_route_points: Optional[List[Tuple[float, float]]] = None,
        target_idx: Optional[int] = None,
        direction: int = 1,
        route_loop: bool = True,
        route_distance_px: Optional[float] = None,
        safe_band: Optional[Tuple[float, float]] = None,
        target_mode: str = "",
        nearest_route_xy: Optional[Tuple[float, float]] = None,
        safe_target_xy: Optional[Tuple[float, float]] = None,
        final_move_target_xy: Optional[Tuple[float, float]] = None,
        image_action: str = "",
        stage_action: str = "",
        oscillation_detected: bool = False,
        route_index_order: str = "",
        user_desired_direction: str = "",
        actual_index_step: Optional[int] = None,
        route_signed_area: Optional[float] = None,
    ) -> str:
        """
        保存 Step7 当前帧的 A 路线叠加图。新版显示固定路线索引推进状态与安全带诊断：
            - A center、current target、nearest_route_xy、safe_target_xy、final_move_target 使用不同颜色；
            - 同时显示 route_distance_px 与 error_dist，避免把二者混淆；
            - 同时显示 image_action 与 stage_action，避免图像方向和实际轴动作混淆。
        """
        try:
            if image_rgb is None:
                return ""
            if self.run_session_dir is not None:
                out_dir = Path(self.run_session_dir) / "step7_a_route_overlays"
            else:
                out_dir = self.output_root / "step7_a_route_overlays" / datetime.now().strftime("run_%Y%m%d_%H%M%S")
            out_dir.mkdir(parents=True, exist_ok=True)

            canvas = cv2.cvtColor(np.asarray(image_rgb).copy(), cv2.COLOR_RGB2BGR)
            h, w = canvas.shape[:2]

            def _pt(xy: Tuple[float, float]) -> Tuple[int, int]:
                x = int(round(float(xy[0])))
                y = int(round(float(xy[1])))
                return int(np.clip(x, 0, max(0, w - 1))), int(np.clip(y, 0, max(0, h - 1)))

            # 三线安全带可视化：base=灰色C边界，min=绿色内边界，target=紫色中线，max=红色外边界。
            for _pts, _color, _thick in (
                (base_route_points, (160, 160, 160), 1),
                (min_route_points, (0, 255, 0), 2),
                (max_route_points, (0, 0, 255), 2),
            ):
                if _pts:
                    arr = np.asarray(_pts, dtype=np.float32).reshape(-1, 2)
                    arr[:, 0] = np.clip(arr[:, 0], 0, max(0, w - 1))
                    arr[:, 1] = np.clip(arr[:, 1], 0, max(0, h - 1))
                    arr_i32 = np.round(arr).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(canvas, [arr_i32], isClosed=True, color=_color, thickness=_thick)

            if route_points:
                route_arr = np.asarray(route_points, dtype=np.float32).reshape(-1, 2)
                route_arr[:, 0] = np.clip(route_arr[:, 0], 0, max(0, w - 1))
                route_arr[:, 1] = np.clip(route_arr[:, 1], 0, max(0, h - 1))
                route_i32 = np.round(route_arr).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(canvas, [route_i32], isClosed=True, color=(255, 0, 255), thickness=2)

                sx, sy = route_points[0]
                cv2.circle(canvas, _pt((sx, sy)), 6, (0, 255, 0), -1)
                cv2.putText(canvas, "route start", (_pt((sx, sy))[0] + 8, _pt((sx, sy))[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2, cv2.LINE_AA)

                sample_step = max(1, len(route_points) // 30)
                for rx, ry in route_points[::sample_step]:
                    cv2.circle(canvas, _pt((rx, ry)), 2, (255, 0, 255), -1)

                if 0 <= int(route_index) < len(route_points):
                    cv2.circle(canvas, _pt(route_points[int(route_index)]), 5, (255, 255, 255), 2)
                    cv2.putText(canvas, f"cur_idx {int(route_index)}", (_pt(route_points[int(route_index)])[0] + 8, _pt(route_points[int(route_index)])[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # 颜色约定：A center=青色；current target=红色；nearest_route_xy=橙色；safe_target_xy=绿色；final_move_target=蓝色。
            if target_xy is not None:
                cv2.drawMarker(canvas, _pt(target_xy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
                cv2.circle(canvas, _pt(target_xy), 7, (0, 0, 255), 2)
                tlabel = f"target #{int(target_idx if target_idx is not None else route_index)}"
                cv2.putText(canvas, tlabel, (_pt(target_xy)[0] + 10, _pt(target_xy)[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

            if nearest_route_xy is not None:
                cv2.circle(canvas, _pt(nearest_route_xy), 6, (0, 165, 255), -1)
                cv2.putText(canvas, "nearest_route", (_pt(nearest_route_xy)[0] + 8, _pt(nearest_route_xy)[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 165, 255), 2, cv2.LINE_AA)

            if safe_target_xy is not None:
                cv2.drawMarker(canvas, _pt(safe_target_xy), (0, 255, 0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=16, thickness=2)
                cv2.putText(canvas, "safe_target", (_pt(safe_target_xy)[0] + 8, _pt(safe_target_xy)[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 2, cv2.LINE_AA)

            if final_move_target_xy is not None:
                cv2.drawMarker(canvas, _pt(final_move_target_xy), (255, 0, 0), markerType=cv2.MARKER_DIAMOND, markerSize=18, thickness=2)
                cv2.putText(canvas, "final_move", (_pt(final_move_target_xy)[0] + 8, _pt(final_move_target_xy)[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 0, 0), 2, cv2.LINE_AA)

            if a_center is not None:
                cv2.circle(canvas, _pt(a_center), 7, (255, 255, 0), -1)
                cv2.putText(canvas, "A center", (_pt(a_center)[0] + 10, _pt(a_center)[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)
                if final_move_target_xy is not None:
                    cv2.arrowedLine(canvas, _pt(a_center), _pt(final_move_target_xy), (0, 255, 255), 2, tipLength=0.18)
                elif target_xy is not None:
                    cv2.arrowedLine(canvas, _pt(a_center), _pt(target_xy), (0, 255, 255), 2, tipLength=0.18)

            safe_text = ""
            if safe_band is not None:
                safe_text = f" safe_band=[{float(safe_band[0]):.1f},{float(safe_band[1]):.1f}]"
            route_dist_text = "None" if route_distance_px is None else f"{float(route_distance_px):.2f}"
            err_text = "None" if error_dist is None else f"{float(error_dist):.2f}"
            actual_step_text = str(int(actual_index_step)) if actual_index_step is not None else str(int(direction))
            loop_text = "loop" if route_loop else "no-loop"
            osc_text = " oscillation=True" if oscillation_detected else ""
            area_text = "None" if route_signed_area is None else f"{float(route_signed_area):.2f}"

            lines = [
                f"Step7 fixed-index route: step={step_idx}, cur_idx={route_index}/{max(0, len(route_points)-1)}, target_idx={target_idx}, {loop_text}",
                f"route_index_order={route_index_order or 'unknown'}, user_desired_direction={user_desired_direction or 'unknown'}, actual_index_step={actual_step_text}",
                f"route_signed_area={area_text}, mode={target_mode}{osc_text}",
                f"route_distance_px={route_dist_text}, error_dist={err_text},{safe_text}",
                f"image_action={image_action or action_name}, stage_action={stage_action or 'not_executed'}",
            ]
            y0 = 26
            for i, line in enumerate(lines):
                cv2.putText(canvas, line, (12, y0 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 255), 2, cv2.LINE_AA)

            out_target_idx = int(target_idx if target_idx is not None else route_index)
            out_path = out_dir / f"step7_a_route_step_{int(step_idx):04d}_cur_{int(route_index):04d}_target_{out_target_idx:04d}.png"
            cv2.imwrite(str(out_path), canvas)
            return str(out_path)
        except Exception as e:
            try:
                self.log(f"[RuleAB-Route] 保存 Step7 A 路线叠加图失败：{e}")
            except Exception:
                pass
            return ""

    def _get_scene_a_center_xy(self, scene: Any) -> Optional[Tuple[float, float]]:
        """从 scene.a 中提取 A 的中心；优先 mask 中心，失败时兼容 center/bbox 字段。"""
        a_obj = None
        try:
            a_obj = getattr(scene, "a", None)
        except Exception:
            pass
        if a_obj is None and isinstance(scene, dict):
            a_obj = scene.get("a") or scene.get("A")
        mask = self._extract_mask_from_scene_object(a_obj)
        center = self._mask_center_xy(mask) if mask is not None else None
        if center is not None:
            return center
        for key in ("center", "center_xy", "xy"):
            try:
                val = getattr(a_obj, key, None)
                if val is not None and len(val) >= 2:
                    return float(val[0]), float(val[1])
            except Exception:
                pass
            try:
                if isinstance(a_obj, dict) and a_obj.get(key) is not None:
                    val = a_obj.get(key)
                    return float(val[0]), float(val[1])
            except Exception:
                pass
        try:
            bbox = getattr(a_obj, "bbox", None)
            if bbox is None and isinstance(a_obj, dict):
                bbox = a_obj.get("bbox")
            if bbox is not None and len(bbox) >= 4:
                x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
                return (x1 + x2) / 2.0, (y1 + y2) / 2.0
        except Exception:
            pass
        return None

    @staticmethod
    def _route_action_from_error(dx: float, dy: float, tol: float) -> int:
        """
        根据 A 中心到目标点的误差选择动作。
        action_id 与原 RuleAB 保持一致：0 STAY, 1 UP, 2 DOWN, 3 LEFT, 4 RIGHT。
        """
        if abs(dx) <= tol and abs(dy) <= tol:
            return 0
        if abs(dx) >= abs(dy):
            return 4 if dx > 0 else 3
        return 2 if dy > 0 else 1

    @staticmethod
    def _nearest_point_on_route_polyline(
        point_xy: Tuple[float, float],
        route_points: List[Tuple[float, float]],
        closed: bool = True,
    ) -> Dict[str, Any]:
        """
        计算点到路线折线的最近点和距离。

        返回：
            nearest_xy: 路线上离 point_xy 最近的点；
            distance_px: A 中心到路线的垂直/端点最小距离；
            segment_index: 最近点所在的线段起点索引；
            segment_t: 最近点在线段上的归一化参数 [0,1]；
            segment_unit_xy: 最近线段方向单位向量；
            away_unit_xy: 从路线最近点指向 A 中心的单位向量。
        """
        if not route_points:
            return {"ok": False, "reason": "empty_route"}

        px, py = float(point_xy[0]), float(point_xy[1])
        pts = [(float(x), float(y)) for x, y in route_points]
        n = len(pts)
        if n == 1:
            rx, ry = pts[0]
            vx, vy = px - rx, py - ry
            dist = math.hypot(vx, vy)
            if dist > 1e-9:
                away = (vx / dist, vy / dist)
            else:
                away = (0.0, -1.0)
            return {
                "ok": True,
                "nearest_xy": (rx, ry),
                "distance_px": float(dist),
                "segment_index": 0,
                "segment_t": 0.0,
                "segment_unit_xy": (1.0, 0.0),
                "away_unit_xy": away,
            }

        seg_count = n if closed else n - 1
        best: Optional[Dict[str, Any]] = None
        for i in range(seg_count):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            sx, sy = x2 - x1, y2 - y1
            seg_len2 = sx * sx + sy * sy
            if seg_len2 <= 1e-12:
                t = 0.0
                qx, qy = x1, y1
                seg_unit = (1.0, 0.0)
            else:
                t = ((px - x1) * sx + (py - y1) * sy) / seg_len2
                t = max(0.0, min(1.0, float(t)))
                qx = x1 + t * sx
                qy = y1 + t * sy
                seg_len = math.sqrt(seg_len2)
                seg_unit = (sx / seg_len, sy / seg_len)

            vx, vy = px - qx, py - qy
            dist = math.hypot(vx, vy)
            if dist > 1e-9:
                away = (vx / dist, vy / dist)
            else:
                # A 正好落在线上时，用线段法向作为远离方向。
                away = (-seg_unit[1], seg_unit[0])

            item = {
                "ok": True,
                "nearest_xy": (float(qx), float(qy)),
                "distance_px": float(dist),
                "segment_index": int(i),
                "segment_t": float(t),
                "segment_unit_xy": (float(seg_unit[0]), float(seg_unit[1])),
                "away_unit_xy": (float(away[0]), float(away[1])),
            }
            if best is None or item["distance_px"] < best["distance_px"]:
                best = item

        return best if best is not None else {"ok": False, "reason": "no_valid_segment"}

    def _route_safety_target_from_distance(
        self,
        a_center: Tuple[float, float],
        route_points: List[Tuple[float, float]],
    ) -> Dict[str, Any]:
        """
        根据 A 中心到路线的距离判断是否需要先回到安全带。

        安全带定义：
            rule_ab_route_min_distance_px <= distance(A_center, route_polyline) <= rule_ab_route_max_distance_px

        返回中的 should_correct=True 时，Step7 不推进路线目标，
        而是先把 A 往 safe_target_xy 移动，使 A 回到安全带中间位置。
        """
        min_d = max(0.0, float(getattr(self.cfg, "rule_ab_route_min_distance_px", 5.0)))
        max_d = float(getattr(self.cfg, "rule_ab_route_max_distance_px", max(15.0, min_d + 1.0)))
        if max_d < min_d:
            max_d = min_d
        target_d = float(getattr(self.cfg, "rule_ab_route_safe_target_distance_px", -1.0))
        if target_d < 0:
            target_d = 0.5 * (min_d + max_d)
        target_d = max(min_d, min(max_d, target_d))

        near = self._nearest_point_on_route_polyline(a_center, route_points, closed=True)
        if not near.get("ok", False):
            return {
                "ok": False,
                "should_correct": False,
                "reason": near.get("reason", "nearest_failed"),
                "min_distance_px": min_d,
                "max_distance_px": max_d,
                "safe_target_distance_px": target_d,
            }

        dist = float(near.get("distance_px", 0.0))
        qx, qy = near["nearest_xy"]
        ux, uy = near.get("away_unit_xy", (0.0, -1.0))

        if dist < min_d:
            mode = "too_close_to_route_move_out"
            should = True
        elif dist > max_d:
            mode = "too_far_from_route_move_in"
            should = True
        else:
            mode = "inside_route_safe_band"
            should = False

        safe_target = (float(qx) + float(ux) * target_d, float(qy) + float(uy) * target_d)
        return {
            "ok": True,
            "should_correct": bool(should),
            "mode": mode,
            "route_distance_px": dist,
            "min_distance_px": float(min_d),
            "max_distance_px": float(max_d),
            "safe_target_distance_px": float(target_d),
            "nearest_route_xy": [float(qx), float(qy)],
            "away_unit_xy": [float(ux), float(uy)],
            "safe_target_xy": [float(safe_target[0]), float(safe_target[1])],
            "nearest_segment_index": int(near.get("segment_index", -1)),
            "nearest_segment_t": float(near.get("segment_t", 0.0)),
        }

    @staticmethod
    def _mask_to_2d_bool(mask: Any, target_shape: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
        if mask is None:
            return None
        arr = np.asarray(mask)
        if arr.ndim == 3:
            channels = int(arr.shape[2])
            if channels == 1:
                arr = arr[:, :, 0]
            elif channels == 4:
                arr = arr[:, :, 3]
            else:
                arr = np.any(arr, axis=2) if arr.dtype == np.bool_ else np.max(arr, axis=2)
        elif arr.ndim != 2:
            return None

        m = arr if arr.dtype == np.bool_ else (arr > 0)
        m = np.asarray(m, dtype=bool)
        if target_shape is not None and tuple(m.shape[:2]) != tuple(target_shape):
            m = cv2.resize(
                m.astype(np.uint8),
                (int(target_shape[1]), int(target_shape[0])),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        return m

    def _preflight_mask_metrics(self, mask: Any) -> Dict[str, Any]:
        if mask is None:
            return {
                "ok": False,
                "area_px": 0,
                "center_x": float("nan"),
                "center_y": float("nan"),
                "center_xy": None,
                "bbox": [],
                "bbox_xyxy": None,
                "reason": "mask_is_none",
            }

        m = self._mask_to_2d_bool(mask)
        if m is None:
            arr = np.asarray(mask)
            return {
                "ok": False,
                "area_px": 0,
                "center_x": float("nan"),
                "center_y": float("nan"),
                "center_xy": None,
                "bbox": [],
                "bbox_xyxy": None,
                "reason": f"invalid_mask_ndim_{arr.ndim}",
            }

        ys, xs = np.where(m)

        if xs.size <= 0:
            return {
                "ok": False,
                "area_px": 0,
                "center_x": float("nan"),
                "center_y": float("nan"),
                "center_xy": None,
                "bbox": [],
                "bbox_xyxy": None,
                "reason": "empty_mask",
            }

        area = int(xs.size)
        cx = float(xs.mean())
        cy = float(ys.mean())
        x0 = int(xs.min())
        y0 = int(ys.min())
        x1 = int(xs.max())
        y1 = int(ys.max())

        return {
            "ok": True,
            "area_px": area,
            "center_x": cx,
            "center_y": cy,
            "center_xy": [cx, cy],
            "bbox": [x0, y0, x1, y1],
            "bbox_xyxy": [x0, y0, x1, y1],
            "reason": "ok",
        }

    def _find_preflight_a_template_mask(self, state: CalibrationState) -> Optional[Path]:
        """优先从 feature_profile_dir / feature_profile_json 附近寻找 a_sam2_initial_mask_cleaned.png。"""
        candidates: List[Path] = []
        for raw in (getattr(state, "feature_profile_dir", ""), getattr(state, "feature_profile_json", "")):
            if not raw:
                continue
            p = Path(str(raw)).expanduser()
            if p.is_file():
                candidates.extend([p.parent / "a_sam2_initial_mask_cleaned.png", p.parent / "a_sam2_initial_mask.png"])
            else:
                candidates.extend([p / "a_sam2_initial_mask_cleaned.png", p / "a_sam2_initial_mask.png"])
        for c in candidates:
            try:
                if c.exists() and c.is_file():
                    return c.resolve()
            except Exception:
                continue
        return None

    def _save_step7_preflight_overlay(
        self,
        image_rgb: np.ndarray,
        a_mask: Optional[np.ndarray],
        a_template_mask: Optional[np.ndarray],
        c_mask: Optional[np.ndarray],
        route_points: List[Tuple[float, float]],
        a_center: Optional[Tuple[float, float]],
        nearest_xy: Optional[Tuple[float, float]],
        out_path: Path,
        passed: bool,
    ) -> str:
        """保存 Step7 A/C 标定预检 overlay。"""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img = np.asarray(image_rgb).copy()
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        canvas = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if c_mask is not None:
            cm = self._mask_to_2d_bool(c_mask, target_shape=canvas.shape[:2])
            if cm is not None:
                overlay = canvas.copy()
                overlay[cm] = (255, 0, 0)
                canvas = cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0)
        if a_template_mask is not None:
            tm = self._mask_to_2d_bool(a_template_mask, target_shape=canvas.shape[:2])
            if tm is not None:
                overlay = canvas.copy()
                overlay[tm] = (0, 165, 255)
                canvas = cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0)
        if a_mask is not None:
            am = self._mask_to_2d_bool(a_mask, target_shape=canvas.shape[:2])
            if am is not None:
                overlay = canvas.copy()
                overlay[am] = (0, 255, 255)
                canvas = cv2.addWeighted(overlay, 0.30, canvas, 0.70, 0)
        if route_points:
            pts = np.asarray([[int(round(x)), int(round(y))] for x, y in route_points], dtype=np.int32)
            if len(pts) >= 2:
                cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], isClosed=True, color=(0, 0, 255), thickness=2)
        if nearest_xy is not None:
            nx, ny = int(round(nearest_xy[0])), int(round(nearest_xy[1]))
            cv2.circle(canvas, (nx, ny), 6, (0, 165, 255), -1)
            cv2.putText(canvas, "nearest_route", (nx + 8, ny - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 165, 255), 2, cv2.LINE_AA)
        if a_center is not None:
            ax, ay = int(round(a_center[0])), int(round(a_center[1]))
            cv2.circle(canvas, (ax, ay), 7, (0, 255, 255), -1)
            cv2.putText(canvas, "A current", (ax + 8, ay + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2, cv2.LINE_AA)
            if nearest_xy is not None:
                cv2.line(canvas, (ax, ay), (int(round(nearest_xy[0])), int(round(nearest_xy[1]))), (0, 255, 255), 1)
        status = "PASS" if passed else "FAIL"
        cv2.putText(canvas, f"Step7 A/C preflight: {status}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(canvas, f"Step7 A/C preflight: {status}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_path), canvas)
        return str(out_path)

    def validate_step7_ac_calibration_preflight(self) -> Dict[str, Any]:
        """
        完整循环测量前的 Step7 A/C 标定预检。

        只做视觉链路检查，不打开激光、不采光谱、不执行 Stage34 运动。
        尽量复用 Step7 实际路径：ensure_rule_ab_follower -> capture_and_build_scene ->
        _capture_and_build_scene_with_ab_retry -> _postprocess_scene_ab_masks -> _extract_mask_from_scene_object，
        并加载完整标定 C 的 static_c_mask / static_c_edge_route。
        """
        self.log("========== Step7 A/C 标定预检：只检查视觉链路，不执行 Stage34 运动 ==========")
        state = self.preflight_check_calibration()
        state = self.prepare_runtime_from_full_calibration(state)
        state = self._get_loaded_calibration_state() or state
        self.context["full_measurement_mode"] = True
        self.context["strict_full_calibration_c_locked"] = True
        c_dir = self._force_cfg_to_strict_full_calibration_c(reason="step7_ac_preflight") or self._resolve_static_c_dir_from_state(state)
        if not c_dir:
            raise RuntimeError("[Step7预检] 未找到完整标定 C 文件夹。")
        c_dir_path = Path(c_dir)
        c_mask_path = c_dir_path / "static_c_mask.png"
        if not c_mask_path.exists():
            raise RuntimeError(f"[Step7预检] static_c_mask.png 不存在：{c_mask_path}")
        c_mask_u8 = cv2.imread(str(c_mask_path), cv2.IMREAD_GRAYSCALE)
        if c_mask_u8 is None or int(np.count_nonzero(c_mask_u8 > 0)) <= 0:
            raise RuntimeError(f"[Step7预检] static_c_mask.png 读取失败或为空：{c_mask_path}")
        c_mask = (c_mask_u8 > 0)

        route_path = c_dir_path / "static_c_edge_route.json"
        route_needs_rebuild = not route_path.exists()
        if route_path.exists():
            try:
                with route_path.open("r", encoding="utf-8") as f:
                    route_existing = json.load(f)
                route_source_existing = str(route_existing.get("source", "") or "").lower()
                execution_mode_existing = str(route_existing.get("execution_route_mode", "") or "")
                if "quad" in route_source_existing and (
                    str(route_existing.get("route_geometry_mode", "") or "") != "mitered_rectangle"
                    or execution_mode_existing != "axis_aligned_hv_chunked_manhattan"
                ):
                    route_needs_rebuild = True
            except Exception:
                route_needs_rebuild = True
        if route_needs_rebuild:
            self.log(f"[Step7预检] static_c_edge_route.json 缺失或仍是旧路径模式，尝试按当前配置重建：{route_path}")
            route_source = str(getattr(self.cfg, "rule_ab_c_edge_route_source", "circle") or "circle").lower().strip()
            if route_source in ("circle", "c_circle", "fit_circle"):
                self._build_and_save_circle_edge_route(c_dir_path)
            elif route_source in ("quad", "quadrilateral"):
                self._build_and_save_quad_edge_route(c_dir_path)
            else:
                route_rt = self._build_and_save_sam2_outer_edge_route(c_dir_path, prefer_sam2_mask=True)
                if not route_rt:
                    self._build_and_save_sam2_outer_edge_route(c_dir_path, prefer_sam2_mask=False)
        route_band = self._load_c_edge_route_band_from_dir(str(c_dir_path))
        route_points = (
            route_band.get("execution_route_points_xy")
            or route_band.get("execution_target_route_points_xy")
            or route_band.get("route_points_xy")
            or route_band.get("target_route_points_xy")
            or route_band.get("execution_base_route_points_xy")
            or route_band.get("base_route_points_xy")
            or []
        )
        if not route_points:
            route_points = self._load_c_edge_route_from_dir(str(c_dir_path))
        if not route_points:
            raise RuntimeError(f"[Step7预检] C route 为空或不可读取：{route_path}")

        out_dir = (self.run_session_dir if self.run_session_dir is not None else self.output_root) / "step7_ac_preflight"
        out_dir.mkdir(parents=True, exist_ok=True)

        old_enable_stage = bool(getattr(self.cfg, "rule_ab_enable_stage", False))
        old_stop = bool(self.stop_requested)
        old_feature_tracker = bool(getattr(self.cfg, "rule_ab_use_feature_tracker_after_sam2_init", True))
        self.stop_requested = False
        seg_retry_info: Dict[str, Any] = {}
        try:
            # 预检禁止 Stage34 运动。follower 仍按 Step7 实际视觉路径初始化/截帧/建 scene。
            self.cfg.rule_ab_enable_stage = False
            # 预检只需要 A/C，不需要 B 模板匹配；关闭 feature tracker 避免被 B 失败阻塞。
            self.cfg.rule_ab_use_feature_tracker_after_sam2_init = False
            self._reset_rule_ab_follower_after_transient_error(reason="step7_ac_preflight_fresh_visual_check")
            follower = self.ensure_rule_ab_follower()
            image_rgb, scene, seg_retry_info = self._capture_and_build_scene_with_ab_retry(
                follower=follower,
                label="Step7-A/C标定预检",
                allow_fail=False,
            )
            try:
                self._postprocess_scene_ab_masks(follower=follower, scene=scene, tag="step7_ac_preflight")
            except Exception as e:
                self.log(f"[Step7预检] A/B mask 后处理失败，将继续检查原始 scene mask：{e}")
            a_mask = self._extract_mask_from_scene_object(getattr(scene, "a", None))
        finally:
            self.cfg.rule_ab_enable_stage = old_enable_stage
            self.cfg.rule_ab_use_feature_tracker_after_sam2_init = old_feature_tracker
            self.stop_requested = old_stop

        a_mask = self._mask_to_2d_bool(a_mask)
        if a_mask is None or int(np.count_nonzero(a_mask)) <= 0:
            raise RuntimeError("[Step7预检] 当前帧 A mask 为空：A 标定点或 A feature template 可能不可用。")
        a_metrics = self._preflight_mask_metrics(a_mask)
        c_metrics = self._preflight_mask_metrics(c_mask)
        a_center_list = a_metrics.get("center_xy")
        a_center = (float(a_center_list[0]), float(a_center_list[1])) if a_center_list else None

        template_path = self._find_preflight_a_template_mask(state)
        a_template_mask = None
        a_template_metrics = {"area_px": "", "center_xy": None, "bbox_xyxy": None}
        template_center_distance_px = ""
        area_ratio_to_template = ""
        if template_path is not None:
            t_u8 = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
            if t_u8 is not None and int(np.count_nonzero(t_u8 > 0)) > 0:
                a_template_mask = (t_u8 > 0)
                if a_template_mask.shape != a_mask.shape:
                    a_template_mask = cv2.resize(a_template_mask.astype(np.uint8), (a_mask.shape[1], a_mask.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                a_template_metrics = self._preflight_mask_metrics(a_template_mask)
                tc = a_template_metrics.get("center_xy")
                if tc and a_center is not None:
                    template_center_distance_px = float(math.hypot(float(a_center[0]) - float(tc[0]), float(a_center[1]) - float(tc[1])))
                if float(a_template_metrics.get("area_px") or 0) > 0:
                    area_ratio_to_template = float(a_metrics["area_px"]) / float(a_template_metrics["area_px"])

        safety_info: Dict[str, Any] = {}
        nearest_xy = None
        in_safe_band = False
        if a_center is not None:
            safety_info = self._route_safety_target_from_distance(a_center, route_points)
            if safety_info.get("nearest_route_xy") is not None:
                nr = safety_info.get("nearest_route_xy")
                nearest_xy = (float(nr[0]), float(nr[1]))
            dist = safety_info.get("route_distance_px")
            if dist is not None:
                in_safe_band = float(safety_info.get("min_distance_px", 0.0)) <= float(dist) <= float(safety_info.get("max_distance_px", 0.0))

        pass_items = {
            "a_positive_points_ok": bool(state.rule_ab_a_positive_points),
            "c_static_mask_ok": bool(c_metrics.get("area_px", 0) > 0),
            "c_route_ok": bool(len(route_points) >= 2),
            "current_a_mask_ok": bool(a_metrics.get("area_px", 0) > 0),
            "a_to_route_distance_measured": bool(safety_info.get("ok", False) and safety_info.get("route_distance_px") is not None),
        }
        passed = all(pass_items.values())
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        overlay_path = self._save_step7_preflight_overlay(
            image_rgb=image_rgb,
            a_mask=a_mask,
            a_template_mask=a_template_mask,
            c_mask=c_mask,
            route_points=route_points,
            a_center=a_center,
            nearest_xy=nearest_xy,
            out_path=out_dir / f"preflight_step7_ac_overlay_{ts}.png",
            passed=passed,
        )
        result = {
            "ok": bool(passed),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "pass_items": pass_items,
            "reason": "ok" if passed else "preflight_failed",
            "calibration_path": str(self.get_calibration_path()),
            "c_dir": str(c_dir_path),
            "c_mask_path": str(c_mask_path),
            "route_path": str(route_path),
            "route_points_count": int(len(route_points)),
            "a_metrics_current": a_metrics,
            "a_template_mask_path": str(template_path) if template_path else "",
            "a_metrics_template": a_template_metrics,
            "a_template_center_distance_px": template_center_distance_px,
            "a_area_ratio_to_template": area_ratio_to_template,
            "c_metrics_static": c_metrics,
            "a_to_c_route_safety": self._json_safe(safety_info),
            "a_to_route_distance_in_safe_band": bool(in_safe_band),
            "a_to_route_distance_is_diagnostic_only": True,
            "seg_retry_info": self._json_safe(seg_retry_info),
            "overlay_path": overlay_path,
        }
        json_path = out_dir / f"preflight_step7_ac_result_{ts}.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(self._json_safe(result), f, ensure_ascii=False, indent=2)
        result["json_path"] = str(json_path)

        dist_txt = safety_info.get("route_distance_px", None)
        if passed:
            self.log(
                f"[Step7预检通过] A mask area={a_metrics.get('area_px')}, "
                f"C area={c_metrics.get('area_px')}, route_points={len(route_points)}, "
                f"A到C路线距离={dist_txt}, in_safe_band={in_safe_band}（仅诊断；越界时 Step7 运行时会先做安全带修正），overlay={overlay_path}"
            )
        else:
            failed = [k for k, v in pass_items.items() if not v]
            self.log(
                f"[Step7预检失败] failed={failed}; A area={a_metrics.get('area_px')}, "
                f"C area={c_metrics.get('area_px')}, route_points={len(route_points)}, "
                f"A到C路线距离={dist_txt}，overlay={overlay_path}"
            )
        self.context["last_step7_ac_preflight_result"] = result
        return result

    def _run_one_c_edge_route_cycle(
        self,
        follower: Any,
        step_idx: int,
        route_points: List[Tuple[float, float]],
        stop_event: Optional[threading.Event] = None,
        phase_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """执行一次 Step7“固定路线索引推进模式”的 A route 运动。"""
        if not route_points:
            return False, {"route_enabled": True, "reason": "empty_route"}

        # 捕获当前帧并获得 A/B/C scene；不再调用 follower.rule_policy(scene)。
        # A/B 分割失败时不允许异常直接冒泡终止完整测量，而是返回 False，
        # 由 run_rule_ab_until_angle_delta() 重新取下一帧继续 Step7。
        try:
            with self.rule_ab_vision_lock:
                image_rgb, scene, seg_retry_info = self._capture_and_build_scene_with_ab_retry(
                    follower=follower,
                    label=f"Step7固定索引路线运动/step={int(step_idx)}",
                    allow_fail=False,
                )
        except Exception as e:
            return False, {
                "route_enabled": True,
                "reason": "ab_segmentation_failed_retry_next_frame",
                "error": str(e),
                "step": int(step_idx),
            }
        try:
            follower.cycle_index = int(getattr(follower, "cycle_index", 0) or 0) + 1
        except Exception:
            pass

        # A/B 分割保护：先保护 B，再从 A 中排除 B；后续 A 中心、路线控制使用修正后的 mask。
        with self.rule_ab_vision_lock:
            ab_guard_info = self._postprocess_scene_ab_masks(
                follower=follower,
                scene=scene,
                tag=f"step7_fixed_route_{int(step_idx)}",
            )

        a_center = self._get_scene_a_center_xy(scene)
        if a_center is None:
            return False, {"route_enabled": True, "reason": "a_center_missing"}

        n_route = len(route_points)
        tol = max(1.0, float(getattr(self.cfg, "rule_ab_route_target_tolerance_px", 8.0)))
        loop_route = bool(getattr(self.cfg, "rule_ab_route_loop", True))
        follow_direction = int(getattr(self.cfg, "rule_ab_follow_c_direction", 1))
        direction_info = self._resolve_route_actual_index_step(route_points, follow_direction)
        direction = int(direction_info.get("actual_index_step", 1) or 1)
        route_index_order = str(direction_info.get("route_index_order", "visual_unknown"))
        user_desired_direction = str(direction_info.get("user_desired_direction", "visual_CCW" if follow_direction >= 0 else "visual_CW"))
        route_signed_area = float(direction_info.get("signed_area", 0.0) or 0.0)
        direction_fallback_used = bool(direction_info.get("direction_fallback_used", False))
        lookahead = max(1, int(getattr(self.cfg, "rule_ab_route_lookahead_points", 1)))
        hard_margin = max(0.0, float(getattr(self.cfg, "rule_ab_route_hard_safety_margin_px", 15.0)))

        route_state = self.context.setdefault("step7_c_edge_route_state", {})
        route_sig = (
            n_route,
            tuple(route_points[0]),
            tuple(route_points[-1]),
            int(direction),
            str(route_index_order),
            str(user_desired_direction),
            bool(loop_route),
        )
        route_sig_old = route_state.get("route_signature")

        def _nearest_route_index_by_point(pt: Tuple[float, float]) -> int:
            best_idx = 0
            best_dist = float("inf")
            px, py = float(pt[0]), float(pt[1])
            for ii, (rx, ry) in enumerate(route_points):
                dd = (px - float(rx)) ** 2 + (py - float(ry)) ** 2
                if dd < best_dist:
                    best_dist = dd
                    best_idx = ii
            return int(best_idx)

        def _advance_index(idx0: int, delta: int) -> int:
            if n_route <= 0:
                return 0
            if loop_route:
                return int((int(idx0) + int(delta)) % n_route)
            return int(max(0, min(n_route - 1, int(idx0) + int(delta))))

        # 每次进入 Step7 时由 run_rule_ab_until_angle_delta() 清空 route_state。
        # 这里只在 state 为空或路线签名变化时，根据当前 A center 找最近 route index 作为起点；
        # 后续不再每帧重跳最近 index。
        if route_sig_old != route_sig or route_state.get("initialized") is not True:
            start_idx = _nearest_route_index_by_point(a_center)
            route_state.clear()
            route_state.update({
                "initialized": True,
                "route_signature": route_sig,
                "current_route_index": int(start_idx),
                "last_advanced_step": int(step_idx),
                "action_history": [],
                "route_index_history": [],
                "oscillation_count": 0,
            })

        current_route_index = int(route_state.get("current_route_index", 0) or 0)
        current_route_index = int(current_route_index % n_route) if loop_route else int(max(0, min(n_route - 1, current_route_index)))

        target_idx = _advance_index(current_route_index, direction * lookahead)
        current_target_xy = (float(route_points[target_idx][0]), float(route_points[target_idx][1]))
        target_distance_px = float(math.hypot(float(current_target_xy[0] - a_center[0]), float(current_target_xy[1] - a_center[1])))

        advanced_this_step = False
        if target_distance_px <= tol:
            current_route_index = _advance_index(current_route_index, direction)
            route_state["current_route_index"] = int(current_route_index)
            route_state["last_advanced_step"] = int(step_idx)
            advanced_this_step = True
            target_idx = _advance_index(current_route_index, direction * lookahead)
            current_target_xy = (float(route_points[target_idx][0]), float(route_points[target_idx][1]))
            target_distance_px = float(math.hypot(float(current_target_xy[0] - a_center[0]), float(current_target_xy[1] - a_center[1])))
        else:
            route_state["current_route_index"] = int(current_route_index)

        # 三线安全带：base_route = C 近似四边形边界；min/max 只用于显示；
        # 距离判定使用 base_route 的真实 route_distance_px。
        band_info = self.context.get("step7_c_edge_route_band") or {}

        def _band_pts(key: str) -> List[Tuple[float, float]]:
            arr = band_info.get(key) if isinstance(band_info, dict) else None
            pts2: List[Tuple[float, float]] = []
            if arr:
                for p0 in arr:
                    try:
                        pts2.append((float(p0[0]), float(p0[1])))
                    except Exception:
                        continue
            return pts2

        base_route_points = _band_pts("base_route_points_xy") or route_points
        min_route_points = _band_pts("min_route_points_xy")
        max_route_points = _band_pts("max_route_points_xy")

        safety_info = self._route_safety_target_from_distance(a_center, base_route_points)
        route_distance_px = safety_info.get("route_distance_px")
        min_d = float(safety_info.get("min_distance_px", getattr(self.cfg, "rule_ab_route_min_distance_px", 0.0)) or 0.0)
        max_d = float(safety_info.get("max_distance_px", getattr(self.cfg, "rule_ab_route_max_distance_px", min_d)) or min_d)
        nearest_route_xy = None
        if safety_info.get("nearest_route_xy") is not None:
            try:
                nr = safety_info.get("nearest_route_xy")
                nearest_route_xy = (float(nr[0]), float(nr[1]))
            except Exception:
                nearest_route_xy = None
        safe_target_xy = None
        if safety_info.get("safe_target_xy") is not None:
            try:
                stxy = safety_info.get("safe_target_xy")
                safe_target_xy = (float(stxy[0]), float(stxy[1]))
            except Exception:
                safe_target_xy = None

        # 固定索引切向：按 current_route_index -> target_idx 的索引方向推进。
        cur_pt = route_points[current_route_index]
        tgt_pt = route_points[target_idx]
        tvx, tvy = float(tgt_pt[0] - cur_pt[0]), float(tgt_pt[1] - cur_pt[1])
        tn = math.hypot(tvx, tvy)
        if tn <= 1e-9:
            next_idx = _advance_index(current_route_index, direction)
            tvx, tvy = float(route_points[next_idx][0] - cur_pt[0]), float(route_points[next_idx][1] - cur_pt[1])
            tn = math.hypot(tvx, tvy)
        tangent_vec = (tvx / tn, tvy / tn) if tn > 1e-9 else (1.0, 0.0)

        # 主路线目标向量：真正追踪 target_idx 对应的 route point，确保 current_route_index 能推进。
        rdx = float(current_target_xy[0] - a_center[0])
        rdy = float(current_target_xy[1] - a_center[1])
        rn = math.hypot(rdx, rdy)
        route_target_vec = (rdx / rn, rdy / rn) if rn > 1e-9 else tangent_vec

        # 法向修正：太近则沿 away_unit 远离；太远则沿 -away_unit 靠近。
        normal_correction_vec = (0.0, 0.0)
        safety_state = "inside"
        severe_safety = False
        slight_safety = False
        if safety_info.get("ok", False) and route_distance_px is not None:
            dist_val = float(route_distance_px)
            aux, auy = safety_info.get("away_unit_xy", (0.0, -1.0))
            aux, auy = float(aux), float(auy)
            if dist_val < min_d:
                safety_state = "too_close"
                normal_correction_vec = (aux, auy)
                severe_safety = bool(dist_val < (min_d - hard_margin))
                slight_safety = not severe_safety
            elif dist_val > max_d:
                safety_state = "too_far"
                normal_correction_vec = (-aux, -auy)
                severe_safety = bool(dist_val > (max_d + hard_margin))
                slight_safety = not severe_safety
            else:
                safety_state = "inside"

        # 振荡检测：连续 UP/DOWN 往返且 current_route_index 未推进。
        hist_actions = list(route_state.get("action_history") or [])
        hist_indices = list(route_state.get("route_index_history") or [])
        oscillation_detected = False
        if len(hist_actions) >= 4 and len(hist_indices) >= 4:
            last4_actions = [str(x).replace("SAFE_", "") for x in hist_actions[-4:]]
            last4_indices = [int(x) for x in hist_indices[-4:]]
            up_down_only = all(x in ("UP", "DOWN") for x in last4_actions)
            alternating = all(last4_actions[i] != last4_actions[i - 1] for i in range(1, len(last4_actions)))
            no_index_progress = len(set(last4_indices)) == 1 and int(last4_indices[-1]) == int(current_route_index)
            oscillation_detected = bool(up_down_only and alternating and no_index_progress and not advanced_this_step)

        target_mode = "inside_band_fixed_index_tangent"
        tangent_weight = 1.0
        normal_weight = 0.0
        final_move_target_xy: Tuple[float, float]

        if severe_safety and safe_target_xy is not None:
            # 只有严重越界时允许纯 SAFE 修正。
            target_mode = f"hard_safe_{safety_state}_normal_only"
            mvx, mvy = normal_correction_vec
            image_action_prefix = "SAFE_"
            final_step_px = max(float(getattr(self.cfg, "rule_ab_c_edge_route_spacing_px", 12.0)), tol * 2.0, 10.0)
            final_move_target_xy = (float(a_center[0]) + float(mvx) * final_step_px, float(a_center[1]) + float(mvy) * final_step_px)
        else:
            image_action_prefix = ""
            if slight_safety:
                target_mode = f"soft_safe_{safety_state}_tangent_plus_normal"
                tangent_weight = 1.0
                normal_weight = 0.45
            else:
                target_mode = "inside_band_fixed_index_target"
                tangent_weight = 1.0
                normal_weight = 0.0

            if oscillation_detected:
                # 振荡时强制加大索引方向分量，并把 target_idx 临时前看一格。
                route_state["oscillation_count"] = int(route_state.get("oscillation_count", 0) or 0) + 1
                target_idx = _advance_index(current_route_index, direction * max(lookahead + 1, 2))
                current_target_xy = (float(route_points[target_idx][0]), float(route_points[target_idx][1]))
                cur_pt = route_points[current_route_index]
                tgt_pt = route_points[target_idx]
                tvx, tvy = float(tgt_pt[0] - cur_pt[0]), float(tgt_pt[1] - cur_pt[1])
                tn = math.hypot(tvx, tvy)
                tangent_vec = (tvx / tn, tvy / tn) if tn > 1e-9 else tangent_vec
                rdx = float(current_target_xy[0] - a_center[0])
                rdy = float(current_target_xy[1] - a_center[1])
                rn = math.hypot(rdx, rdy)
                route_target_vec = (rdx / rn, rdy / rn) if rn > 1e-9 else tangent_vec
                tangent_weight = max(2.0, tangent_weight * 2.0)
                normal_weight = min(0.15, normal_weight * 0.25)
                target_mode += "_oscillation_force_tangent"

            # 非严重越界时，主分量使用“指向 target_idx 路线点”的向量；
            # 这样 A 会真正接近 current target，达到 tolerance 后推进 current_route_index。
            main_route_vec = route_target_vec
            mvx = float(main_route_vec[0]) * float(tangent_weight) + float(normal_correction_vec[0]) * float(normal_weight)
            mvy = float(main_route_vec[1]) * float(tangent_weight) + float(normal_correction_vec[1]) * float(normal_weight)
            mn = math.hypot(mvx, mvy)
            if mn <= 1e-9:
                mvx, mvy = tangent_vec
                mn = math.hypot(mvx, mvy)
            mvx, mvy = float(mvx) / max(mn, 1e-9), float(mvy) / max(mn, 1e-9)
            final_step_px = max(float(getattr(self.cfg, "rule_ab_c_edge_route_spacing_px", 12.0)), tol * 2.0, 10.0)
            final_move_target_xy = (float(a_center[0]) + mvx * final_step_px, float(a_center[1]) + mvy * final_step_px)

        dx = float(final_move_target_xy[0] - a_center[0])
        dy = float(final_move_target_xy[1] - a_center[1])
        action_id = self._route_action_from_error(dx, dy, tol)
        action_name_map = {0: "STAY", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT"}
        image_action = image_action_prefix + action_name_map.get(action_id, f"ACTION_{action_id}")

        status = {}
        try:
            if hasattr(follower, "get_boundary_rule_status"):
                status = follower.get_boundary_rule_status(scene) or {}
        except Exception:
            status = {}

        move_info: Dict[str, Any] = {}
        stage_action = "STAY/no_stage_move"
        if action_id != 0:
            if stop_event is not None and stop_event.is_set():
                # 即使 stop_event 已设置，也保存当前帧的路线 overlay 便于排查
                _early_overlay_path = ""
                try:
                    _early_overlay_path = self._save_step7_route_overlay_image(
                        image_rgb=image_rgb,
                        route_points=route_points,
                        a_center=a_center,
                        target_xy=current_target_xy,
                        route_index=current_route_index,
                        step_idx=step_idx,
                        action_name=image_action,
                        error_dist=float(math.hypot(float(current_target_xy[0] - a_center[0]), float(current_target_xy[1] - a_center[1]))),
                        min_route_points=min_route_points,
                        max_route_points=max_route_points,
                        base_route_points=base_route_points,
                        target_idx=target_idx,
                        direction=direction,
                        route_loop=loop_route,
                        route_distance_px=float(route_distance_px) if route_distance_px is not None else None,
                        safe_band=(float(min_d), float(max_d)),
                        target_mode=target_mode,
                        nearest_route_xy=nearest_route_xy,
                        safe_target_xy=safe_target_xy,
                        final_move_target_xy=final_move_target_xy,
                        image_action=image_action,
                        stage_action="STAY/stop_event_set",
                        oscillation_detected=oscillation_detected,
                        route_index_order=route_index_order,
                        user_desired_direction=user_desired_direction,
                        actual_index_step=direction,
                        route_signed_area=route_signed_area,
                    )
                except Exception as _e:
                    self.log(f"[Step7实时控制] stop_event 提前返回时保存 overlay 失败：{_e}")
                return False, {
                    "route_enabled": True,
                    "reason": "stop_event_set_before_stage34_move",
                    "step": int(step_idx),
                    "action_id": int(action_id),
                    "action_name": image_action,
                    "route_overlay_path": _early_overlay_path,
                }
            if phase_state is not None:
                phase_state["phase"] = "moving"
                phase_state["controller_action"] = str(image_action)
            move_info = self._execute_rule_ab_action_with_watchable_pause(
                follower=follower,
                action_id=int(action_id),
                stop_event=stop_event,
                phase_state=phase_state,
            ) or {}
            ch = move_info.get("channel", "?")
            ch_name = move_info.get("channel_name", "?")
            move_step = move_info.get("move_step", "?")
            dry_run = bool(move_info.get("dry_run", False))
            interrupted = bool(move_info.get("pause_interrupted", False))
            stage_action = f"{ch_name}/CH{ch}: move_step={move_step}, dry_run={dry_run}, pause_interrupted={interrupted}"

        # 更新振荡历史：使用图像方向和当前路线索引。
        hist_actions.append(str(image_action))
        hist_indices.append(int(current_route_index))
        route_state["action_history"] = hist_actions[-8:]
        route_state["route_index_history"] = hist_indices[-8:]
        route_state["last_image_action"] = str(image_action)
        route_state["last_stage_action"] = str(stage_action)
        route_state["last_target_idx"] = int(target_idx)

        # 尽量保存外部 follower 的标注图，便于后续从 annotated_frames 追溯角度来源。
        try:
            if hasattr(follower, "save_annotated_image"):
                follower.save_annotated_image(image_rgb, scene, status, action_id=action_id, action_name=f"ROUTE_{image_action}")
        except TypeError:
            try:
                follower.save_annotated_image(image_rgb, scene, status, action_id, f"ROUTE_{image_action}")
            except Exception:
                pass
        except Exception:
            pass

        route_overlay_path = self._save_step7_route_overlay_image(
            image_rgb=image_rgb,
            route_points=route_points,
            a_center=a_center,
            target_xy=current_target_xy,
            route_index=current_route_index,
            step_idx=step_idx,
            action_name=image_action,
            error_dist=float(math.hypot(float(current_target_xy[0] - a_center[0]), float(current_target_xy[1] - a_center[1]))),
            min_route_points=min_route_points,
            max_route_points=max_route_points,
            base_route_points=base_route_points,
            target_idx=target_idx,
            direction=direction,
            route_loop=loop_route,
            route_distance_px=float(route_distance_px) if route_distance_px is not None else None,
            safe_band=(float(min_d), float(max_d)),
            target_mode=target_mode,
            nearest_route_xy=nearest_route_xy,
            safe_target_xy=safe_target_xy,
            final_move_target_xy=final_move_target_xy,
            image_action=image_action,
            stage_action=stage_action,
            oscillation_detected=oscillation_detected,
            route_index_order=route_index_order,
            user_desired_direction=user_desired_direction,
            actual_index_step=direction,
            route_signed_area=route_signed_area,
        )

        info = {
            "route_enabled": True,
            "step": int(step_idx),
            "current_route_index": int(current_route_index),
            "route_index": int(current_route_index),
            "target_idx": int(target_idx),
            "route_len": int(n_route),
            "direction": int(direction),
            "actual_index_step": int(direction),
            "route_index_order": str(route_index_order),
            "user_desired_direction": str(user_desired_direction),
            "route_signed_area": float(route_signed_area),
            "direction_fallback_used": bool(direction_fallback_used),
            "direction_name": str(user_desired_direction),
            "route_loop": bool(loop_route),
            "lookahead_points": int(lookahead),
            "a_center_xy": [float(a_center[0]), float(a_center[1])],
            "current_target_xy": [float(current_target_xy[0]), float(current_target_xy[1])],
            "target_xy": [float(current_target_xy[0]), float(current_target_xy[1])],
            "nearest_route_xy": [float(nearest_route_xy[0]), float(nearest_route_xy[1])] if nearest_route_xy is not None else None,
            "safe_target_xy": [float(safe_target_xy[0]), float(safe_target_xy[1])] if safe_target_xy is not None else None,
            "final_move_target_xy": [float(final_move_target_xy[0]), float(final_move_target_xy[1])],
            "error_dx": float(current_target_xy[0] - a_center[0]),
            "error_dy": float(current_target_xy[1] - a_center[1]),
            "error_dist": float(math.hypot(float(current_target_xy[0] - a_center[0]), float(current_target_xy[1] - a_center[1]))),
            "move_dx": dx,
            "move_dy": dy,
            "move_dist": float(math.hypot(dx, dy)),
            "tolerance_px": tol,
            "action_id": int(action_id),
            "image_action": image_action,
            "stage_action": stage_action,
            "action_name": image_action,
            "target_mode": target_mode,
            "route_safety": self._json_safe(safety_info),
            "route_distance_px": float(route_distance_px) if route_distance_px is not None else None,
            "safe_band": [float(min_d), float(max_d)],
            "route_min_distance_px": float(min_d),
            "route_max_distance_px": float(max_d),
            "route_hard_safety_margin_px": float(hard_margin),
            "safety_state": safety_state,
            "severe_safety": bool(severe_safety),
            "slight_safety": bool(slight_safety),
            "tangent_vec": [float(tangent_vec[0]), float(tangent_vec[1])],
            "route_target_vec": [float(route_target_vec[0]), float(route_target_vec[1])],
            "normal_correction_vec": [float(normal_correction_vec[0]), float(normal_correction_vec[1])],
            "tangent_weight": float(tangent_weight),
            "normal_weight": float(normal_weight),
            "oscillation_detected": bool(oscillation_detected),
            "oscillation_count": int(route_state.get("oscillation_count", 0) or 0),
            "advanced_this_step": bool(advanced_this_step),
            "route_overlay_path": route_overlay_path,
            "ab_mask_guard": self._json_safe(ab_guard_info),
            "seg_retry_info": self._json_safe(seg_retry_info),
            "move_info": self._json_safe(move_info),
            "status": self._json_safe(status),
        }
        self.log(
            f"[RuleAB-Route] step={step_idx}, cur_idx={current_route_index}/{n_route-1}, "
            f"target_idx={target_idx}, route_index_order={route_index_order}, "
            f"user_desired_direction={user_desired_direction}, actual_index_step={direction}, "
            f"route_signed_area={route_signed_area:.2f}, loop={loop_route}, mode={target_mode}, "
            f"route_distance_px={info['route_distance_px']}, safe_band=[{min_d:.2f},{max_d:.2f}], "
            f"nearest_route_xy={info['nearest_route_xy']}, safe_target_xy={info['safe_target_xy']}, "
            f"A=({a_center[0]:.1f},{a_center[1]:.1f}), "
            f"current_target=({current_target_xy[0]:.1f},{current_target_xy[1]:.1f}), "
            f"final_move_target=({final_move_target_xy[0]:.1f},{final_move_target_xy[1]:.1f}), "
            f"error_dist={info['error_dist']:.2f}px, move_vec=({dx:.1f},{dy:.1f}), "
            f"image_action={image_action}, stage_action={stage_action}, "
            f"oscillation_detected={oscillation_detected}"
        )
        return True, info

    @staticmethod
    def _mask_area_center_angle(mask: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
        """计算 mask 的面积、中心和主轴角度，用于 B 形状保护。"""
        if mask is None:
            return None
        m = np.asarray(mask).astype(bool)
        if m.size <= 0 or not np.any(m):
            return None
        ys, xs = np.where(m)
        area = int(len(xs))
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        angle = 0.0
        try:
            pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
            if len(pts) >= 5:
                pts0 = pts - pts.mean(axis=0, keepdims=True)
                cov = np.cov(pts0.T)
                vals, vecs = np.linalg.eigh(cov)
                v = vecs[:, int(np.argmax(vals))]
                angle = math.degrees(math.atan2(float(v[1]), float(v[0]))) % 180.0
        except Exception:
            angle = 0.0
        return {"area": area, "center": (cx, cy), "angle_deg": float(angle)}

    @staticmethod
    def _set_mask_to_scene_object(obj: Any, mask: np.ndarray) -> bool:
        """把修正后的 mask 写回 scene.a/scene.b 对象。"""
        if obj is None or mask is None:
            return False
        m = np.asarray(mask).astype(bool)
        try:
            if isinstance(obj, dict):
                obj["mask"] = m
                obj["area"] = int(np.count_nonzero(m))
                ys, xs = np.where(m)
                if len(xs) > 0:
                    obj["center"] = (float(np.mean(xs)), float(np.mean(ys)))
                return True
        except Exception:
            pass
        ok = False
        try:
            setattr(obj, "mask", m)
            ok = True
        except Exception:
            pass
        try:
            setattr(obj, "area", int(np.count_nonzero(m)))
        except Exception:
            pass
        try:
            ys, xs = np.where(m)
            if len(xs) > 0:
                cx, cy = float(np.mean(xs)), float(np.mean(ys))
                setattr(obj, "center", (cx, cy))
                setattr(obj, "center_xy", (cx, cy))
                setattr(obj, "bbox", (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())))
        except Exception:
            pass
        return ok

    def _postprocess_scene_ab_masks(self, follower: Any, scene: Any, tag: str = "") -> Dict[str, Any]:
        """
        对当前 scene 的 A/B mask 做运行时后处理。

        当前版本按你的要求修改为：
            1. B mask 不再做面积比例、中心跳变、主轴角度跳变约束；
            2. B mask 不再因为异常而回退上一帧；
            3. 后续运动控制、角度检测直接使用当前帧结果；
            4. 仍保留 A = A - dilate(B_current) 的处理，避免 A mask 把当前帧 B 包进去。

        注意：rule_ab_enable_ab_mask_guard 现在只表示是否执行 A/B 后处理。
        即使开启，也不会再替换 B 为上一帧 B。
        """
        info: Dict[str, Any] = {
            "enabled": bool(getattr(self.cfg, "rule_ab_enable_ab_mask_guard", True)),
            "tag": tag,
            "b_guard_used_previous": False,
            "b_current_used_directly": False,
            "b_constraint_checked": False,
            "a_b_excluded": False,
            "reason": "not_checked",
        }
        if not info["enabled"]:
            info["reason"] = "disabled"
            return info

        try:
            a_obj = getattr(scene, "a", None) if scene is not None else None
            b_obj = getattr(scene, "b", None) if scene is not None else None
            if isinstance(scene, dict):
                a_obj = scene.get("a") or scene.get("A")
                b_obj = scene.get("b") or scene.get("B")
        except Exception:
            a_obj = b_obj = None

        a_mask = self._extract_mask_from_scene_object(a_obj)
        b_mask = self._extract_mask_from_scene_object(b_obj)
        if b_mask is None:
            info["reason"] = "b_mask_missing"
            return info

        b_mask = np.asarray(b_mask).astype(bool)
        b_feat = self._mask_area_center_angle(b_mask)
        if b_feat is None:
            info["reason"] = "b_mask_empty"
            return info

        # 关键修改：不再检查 area_ratio / center_jump / angle_jump，也不再回退上一帧 B。
        # 这里只把当前帧 B 写回 scene.b，并更新诊断缓存；后续角度和运动全部使用当前帧 B。
        self._set_mask_to_scene_object(b_obj, b_mask)
        guard = self.context.setdefault("rule_ab_ab_mask_guard", {})
        if guard.get("b_ref_feat") is None:
            guard["b_ref_feat"] = self._json_safe(b_feat)
        guard["b_last_feat"] = self._json_safe(b_feat)
        guard["b_last_mask"] = b_mask.copy()

        info.update({
            "reason": "b_current_used_directly_no_constraint_no_fallback",
            "b_guard_used_previous": False,
            "b_current_used_directly": True,
            "b_current_feat": self._json_safe(b_feat),
            "b_ref_feat": self._json_safe(guard.get("b_ref_feat")),
        })

        # A/B 已经由颜色距离逐像素分离时，不再额外做 A = A - dilate(B)。
        # 否则 A/B 接触或部分重叠时，会把已经分给 A 的像素再次删除，导致 A center 漂移。
        if bool(getattr(self.cfg, "rule_ab_use_feature_tracker_after_sam2_init", True)) and bool(getattr(self.cfg, "rule_ab_feature_ab_color_separation", True)):
            info["a_b_excluded"] = False
            info["a_exclude_skipped"] = True
            info["a_exclude_reason"] = "feature_tracker_color_separation_already_applied"
            return info

        # A 排除当前帧 B：这里只修正 A，不约束/替换 B。
        if a_mask is not None and b_mask is not None:
            a_bool = np.asarray(a_mask).astype(bool)
            b_bool = np.asarray(b_mask).astype(bool)
            dilate_px = max(0, int(getattr(self.cfg, "rule_ab_a_exclude_b_dilate_px", 5)))
            if dilate_px > 0:
                k = 2 * dilate_px + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                b_exclude = cv2.dilate(b_bool.astype(np.uint8), kernel, iterations=1).astype(bool)
            else:
                b_exclude = b_bool

            a_clean = a_bool & (~b_exclude)
            clean_area = int(np.count_nonzero(a_clean))
            min_area = int(getattr(self.cfg, "rule_ab_a_min_area_after_b_exclude_px", 5))
            overlap_before = int(np.count_nonzero(a_bool & b_bool))
            info["a_b_overlap_before_exclude_px"] = overlap_before
            info["a_area_before_exclude_px"] = int(np.count_nonzero(a_bool))
            info["a_area_after_exclude_px"] = clean_area

            if clean_area >= max(1, min_area):
                self._set_mask_to_scene_object(a_obj, a_clean)
                info["a_b_excluded"] = True
                # 同步可能存在的 segmenter/follower 缓存字段。
                for obj in (follower, getattr(follower, "abc_segmenter", None), getattr(follower, "segmenter", None)):
                    if obj is None:
                        continue
                    for attr in ("last_a_mask", "current_a_mask", "a_mask", "last_mask_a"):
                        try:
                            if hasattr(obj, attr):
                                setattr(obj, attr, a_clean.copy())
                        except Exception:
                            pass
            else:
                info["a_b_excluded"] = False
                info["a_exclude_reason"] = f"clean_area_too_small<{min_area}"

        return info


    # --------------------------------------------------------
    # 新方案：SAM2 首帧分割 -> 保存 A/B/C 颜色与形状模板 -> 后续特征识别
    # --------------------------------------------------------

    def _get_scene_object_by_label(self, scene: Any, label: str) -> Any:
        """兼容 scene.a / scene['a'] / scene.A 等字段。"""
        if scene is None:
            return None
        keys = [str(label).lower(), str(label).upper()]
        for k in keys:
            try:
                if hasattr(scene, k):
                    return getattr(scene, k)
            except Exception:
                pass
        if isinstance(scene, dict):
            for k in keys:
                if k in scene:
                    return scene.get(k)
        return None

    @staticmethod
    def _json_safe_simple(obj: Any) -> Any:
        """保存模板 JSON 时使用的轻量安全转换。"""
        try:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, dict):
                return {str(k): MeasurementWorkflow._json_safe_simple(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [MeasurementWorkflow._json_safe_simple(v) for v in obj]
        except Exception:
            pass
        return obj

    def _feature_tracker_output_dir(self) -> Path:
        """A/B/C 颜色形状模板保存目录。"""
        custom = str(getattr(self.cfg, "rule_ab_feature_tracker_save_dir", "") or "").strip()
        if custom:
            out = Path(custom).expanduser()
        elif self.run_session_dir is not None:
            out = Path(self.run_session_dir) / "abc_feature_templates"
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = self.output_root / "abc_feature_templates" / f"run_{ts}"
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _extract_b_positive_hsv_seeds_from_points(
        self,
        image_rgb: np.ndarray,
        positive_points: Any,
        label: str = "B",
    ) -> List[List[float]]:
        """从 B 正点附近小窗口提取 HSV median 颜色种子。"""
        img = np.asarray(image_rgb).astype(np.uint8)
        if img.ndim != 3 or img.shape[2] < 3:
            return []
        hsv = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        r = max(1, int(getattr(self.cfg, "rule_ab_b_positive_color_seed_radius_px", 5)))
        seeds: List[List[float]] = []
        pts = self._normalize_points_xy(positive_points)
        for i, (x, y) in enumerate(pts):
            cx, cy = int(round(float(x))), int(round(float(y)))
            x1, x2 = max(0, cx - r), min(w, cx + r + 1)
            y1, y2 = max(0, cy - r), min(h, cy + r + 1)
            if x2 <= x1 or y2 <= y1:
                self.log(f"[BColorSeed] {label} positive point {i} 窗口越界/为空，跳过。")
                continue
            patch = hsv[y1:y2, x1:x2].reshape(-1, 3)
            if patch.shape[0] < 3:
                self.log(f"[BColorSeed] {label} positive point {i} 窗口像素太少 n={patch.shape[0]}，跳过。")
                continue
            med = np.median(patch.astype(np.float32), axis=0)
            seed = [float(med[0]), float(med[1]), float(med[2])]
            seeds.append(seed)
            self.log(f"[BColorSeed] {label} positive point {i} hsv_seed=({seed[0]:.1f},{seed[1]:.1f},{seed[2]:.1f})")
        self.log(f"[BColorSeed] saved b_positive_hsv_seeds count={len(seeds)}")
        return seeds

    @staticmethod
    def _dedupe_hsv_centers(centers: List[List[float]], h_tol: float = 3.0, sv_tol: float = 12.0) -> List[List[float]]:
        """去掉过近的 HSV center；H 按 OpenCV 0~179 环形距离处理。"""
        out: List[List[float]] = []
        for c in centers:
            try:
                h, s, v = float(c[0]) % 180.0, float(c[1]), float(c[2])
            except Exception:
                continue
            duplicate = False
            for e in out:
                dh = abs(h - float(e[0]))
                dh = min(dh, 180.0 - dh)
                ds = abs(s - float(e[1]))
                dv = abs(v - float(e[2]))
                if dh <= h_tol and ds <= sv_tol and dv <= sv_tol:
                    duplicate = True
                    break
            if not duplicate:
                out.append([float(h), float(s), float(v)])
        return out

    def _build_b_hsv_centers_from_seeds_and_mask(
        self,
        image_rgb: np.ndarray,
        b_mask: np.ndarray,
        b_positive_hsv_seeds: Optional[List[List[float]]] = None,
    ) -> Tuple[List[List[float]], Dict[str, Any]]:
        """用 B 正点颜色 seed + B clean mask 内颜色联合生成多 HSV center。"""
        img = np.asarray(image_rgb).astype(np.uint8)
        hsv = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
        m = np.asarray(b_mask).astype(bool)
        target_k = max(1, int(getattr(self.cfg, "rule_ab_b_hsv_center_count", 8)))
        min_s = int(getattr(self.cfg, "rule_ab_b_color_min_s", 20))
        min_v = int(getattr(self.cfg, "rule_ab_b_color_min_v", 20))
        seeds = []
        for c in (b_positive_hsv_seeds or []):
            try:
                seeds.append([float(c[0]) % 180.0, float(c[1]), float(c[2])])
            except Exception:
                continue
        seeds = self._dedupe_hsv_centers(seeds)

        pix = hsv[m] if np.any(m) else np.empty((0, 3), dtype=np.uint8)
        if pix.size > 0:
            keep = (pix[:, 1].astype(np.int16) >= min_s) & (pix[:, 2].astype(np.int16) >= min_v)
            pix = pix[keep]
        debug: Dict[str, Any] = {
            "mask_pixel_count": int(np.count_nonzero(m)),
            "valid_color_pixel_count": int(len(pix)),
            "seed_count": int(len(seeds)),
            "target_center_count": int(target_k),
            "min_s": int(min_s),
            "min_v": int(min_v),
        }

        centers: List[List[float]] = []
        # 正点 seed 优先保留，避免 B 多色被整块均值冲淡。
        for c in seeds:
            if len(centers) < target_k:
                centers.append(c)
        need = max(0, target_k - len(centers))
        if need > 0 and len(pix) > 0:
            # 为处理 H 环形距离，把 H 映射到 cos/sin，再联合 S/V 做 cv2.kmeans。
            sample = pix.astype(np.float32)
            if len(sample) > 5000:
                idx = np.linspace(0, len(sample) - 1, 5000).astype(np.int64)
                sample = sample[idx]
            h_rad = sample[:, 0] / 180.0 * 2.0 * np.pi
            feat = np.column_stack([
                np.cos(h_rad) * 90.0,
                np.sin(h_rad) * 90.0,
                sample[:, 1],
                sample[:, 2],
            ]).astype(np.float32)
            k = int(min(need, len(feat)))
            if k > 0:
                try:
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
                    _compactness, labels, km_centers = cv2.kmeans(feat, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
                    labels = labels.reshape(-1)
                    for j in range(k):
                        members = sample[labels == j]
                        if len(members) <= 0:
                            continue
                        # 每个聚类的 H 用向量均值恢复，S/V 用中位数。
                        rr = members[:, 0] / 180.0 * 2.0 * np.pi
                        ang = math.atan2(float(np.mean(np.sin(rr))), float(np.mean(np.cos(rr))))
                        if ang < 0:
                            ang += 2.0 * math.pi
                        hh = (ang / (2.0 * math.pi) * 180.0) % 180.0
                        ss, vv = np.median(members[:, 1].astype(np.float32)), np.median(members[:, 2].astype(np.float32))
                        centers.append([float(hh), float(ss), float(vv)])
                except Exception as e:
                    debug["kmeans_error"] = str(e)
                    med = np.median(sample, axis=0)
                    centers.append([float(med[0]), float(med[1]), float(med[2])])
        centers = self._dedupe_hsv_centers(centers)[:target_k]
        if not centers and len(pix) > 0:
            med = np.median(pix.astype(np.float32), axis=0)
            centers = [[float(med[0]), float(med[1]), float(med[2])]]
        debug["center_count"] = int(len(centers))
        debug["centers"] = centers
        debug["source"] = "positive_seeds+clean_mask"
        self.log(f"[BColorCenter] center_count={len(centers)}, source=positive_seeds+clean_mask, b_hsv_centers={centers}")
        return centers, debug

    def _extract_feature_profile_from_mask(self, image_rgb: np.ndarray, mask: np.ndarray, label: str) -> Dict[str, Any]:
        """从 SAM2 首帧 mask 内提取颜色、面积、中心、bbox、形状模板。"""
        m = np.asarray(mask).astype(bool)
        if not np.any(m):
            raise RuntimeError(f"{label} 初始 mask 为空，不能建立颜色/形状模板。")

        img = np.asarray(image_rgb).astype(np.uint8)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        pix_hsv = hsv[m]
        pix_rgb = img[m]

        ys, xs = np.where(m)
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        cx, cy = float(xs.mean()), float(ys.mean())
        area = int(np.count_nonzero(m))

        crop = m[y1:y2 + 1, x1:x2 + 1].astype(np.uint8) * 255
        shape_template = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_NEAREST) > 0

        # Hue 是环形变量，这里保存普通均值作为阈值中心；若材料颜色跨 0/179，可把 h_tol 调大一点。
        mean_hsv = [float(x) for x in np.mean(pix_hsv, axis=0)]
        std_hsv = [float(x) for x in np.std(pix_hsv, axis=0)]
        mean_rgb = [float(x) for x in np.mean(pix_rgb, axis=0)]
        std_rgb = [float(x) for x in np.std(pix_rgb, axis=0)]

        contours = self._find_contours_compat(m.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hu = [0.0] * 7
        if contours:
            c = max(contours, key=cv2.contourArea)
            mom = cv2.moments(c)
            hu_raw = cv2.HuMoments(mom).reshape(-1)
            hu = [float(-np.sign(v) * np.log10(abs(v) + 1e-12)) for v in hu_raw]

        return {
            "label": str(label).lower(),
            "area_px": int(area),
            "center_xy": [float(cx), float(cy)],
            "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
            "bbox_wh": [float(x2 - x1 + 1), float(y2 - y1 + 1)],
            "mean_hsv": mean_hsv,
            "std_hsv": std_hsv,
            "mean_rgb": mean_rgb,
            "std_rgb": std_rgb,
            "hu_log": hu,
            "shape_template_64": shape_template.astype(np.uint8).tolist(),
            "last_center_xy": [float(cx), float(cy)],
            "last_bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
        }

    def _save_feature_templates_debug(
        self,
        image_rgb: np.ndarray,
        profiles: Dict[str, Dict[str, Any]],
        masks: Dict[str, np.ndarray],
        out_dir: Path,
        raw_masks: Optional[Dict[str, np.ndarray]] = None,
        clean_debug: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        保存首帧模板、mask 和 overlay。

        约定：
            - *_sam2_initial_mask_raw.png：原始 SAM2 输出；
            - *_sam2_initial_mask_cleaned.png：按正点连通域清理后的 mask；
            - *_sam2_initial_mask.png：兼容旧文件名，保存 cleaned mask；
            - feature tracker 只使用 cleaned mask 建模板。
        """
        paths: Dict[str, str] = {}
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            cv2.imwrite(str(out_dir / "initial_frame_rgb.png"), cv2.cvtColor(np.asarray(image_rgb), cv2.COLOR_RGB2BGR))
            canvas = cv2.cvtColor(np.asarray(image_rgb).copy(), cv2.COLOR_RGB2BGR)
            color_map = {"a": (0, 255, 255), "b": (0, 255, 0), "c": (255, 0, 255)}
            overlay = canvas.copy()
            raw_masks = raw_masks or {}

            for lab in ("a", "b", "c"):
                if lab in raw_masks:
                    rb = np.asarray(raw_masks[lab]).astype(bool)
                    raw_path = out_dir / f"{lab}_sam2_initial_mask_raw.png"
                    cv2.imwrite(str(raw_path), rb.astype(np.uint8) * 255)
                    paths[f"{lab}_raw_mask"] = str(raw_path)

                if lab not in masks:
                    continue
                mb = np.asarray(masks[lab]).astype(bool)
                cleaned_path = out_dir / f"{lab}_sam2_initial_mask_cleaned.png"
                legacy_path = out_dir / f"{lab}_sam2_initial_mask.png"
                cv2.imwrite(str(cleaned_path), mb.astype(np.uint8) * 255)
                cv2.imwrite(str(legacy_path), mb.astype(np.uint8) * 255)
                paths[f"{lab}_cleaned_mask"] = str(cleaned_path)
                paths[f"{lab}_legacy_mask"] = str(legacy_path)

                overlay[mb] = color_map.get(lab, (255, 255, 0))
                center = profiles.get(lab, {}).get("center_xy")
                if center:
                    cx, cy = int(round(center[0])), int(round(center[1]))
                    cv2.circle(canvas, (cx, cy), 6, color_map.get(lab, (255, 255, 0)), -1)
                    cv2.putText(canvas, f"{lab.upper()}-clean", (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_map.get(lab, (255, 255, 0)), 2, cv2.LINE_AA)

            canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)
            overlay_path = out_dir / "abc_sam2_initial_feature_overlay.png"
            cv2.imwrite(str(overlay_path), canvas)
            paths["overlay_path"] = str(overlay_path)

            if clean_debug is not None:
                clean_debug_path = out_dir / "abc_initial_mask_clean_debug.json"
                with clean_debug_path.open("w", encoding="utf-8") as f:
                    json.dump(self._json_safe_simple(clean_debug), f, ensure_ascii=False, indent=2)
                paths["clean_debug_json"] = str(clean_debug_path)

            profiles_path = out_dir / "abc_feature_profiles.json"
            with profiles_path.open("w", encoding="utf-8") as f:
                json.dump(self._json_safe_simple(profiles), f, ensure_ascii=False, indent=2)
            paths["profiles_json"] = str(profiles_path)
        except Exception as e:
            self.log(f"[ABC特征跟踪] 保存首帧模板调试文件失败：{e}")
        return paths

    def _hsv_color_distance_to_profile(self, image_rgb: np.ndarray, profile: Dict[str, Any]) -> np.ndarray:
        """计算每个像素到模板 HSV 中心的归一化颜色距离，越小越像。

        对 B 优先使用 b_hsv_centers 的最小距离；兼容旧 mean_hsv。
        """
        img = np.asarray(image_rgb).astype(np.uint8)
        hsv = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
        label = str(profile.get("label", "")).lower()
        if label == "b" and profile.get("b_hsv_centers"):
            centers = profile.get("b_hsv_centers") or []
            h_tol = max(1.0, float(getattr(self.cfg, "rule_ab_b_feature_h_tol", getattr(self.cfg, "rule_ab_feature_h_tol", 14))))
            s_tol = max(1.0, float(getattr(self.cfg, "rule_ab_b_feature_s_tol", getattr(self.cfg, "rule_ab_feature_s_tol", 75))))
            v_tol = max(1.0, float(getattr(self.cfg, "rule_ab_b_feature_v_tol", getattr(self.cfg, "rule_ab_feature_v_tol", 75))))
            best = np.full(hsv.shape[:2], 9999.0, dtype=np.float32)
            for c in centers:
                try:
                    h0, s0, v0 = float(c[0]), float(c[1]), float(c[2])
                except Exception:
                    continue
                dh = self._hue_distance_180(hsv[:, :, 0], int(round(h0))).astype(np.float32) / h_tol
                ds = np.abs(hsv[:, :, 1].astype(np.float32) - s0) / s_tol
                dv = np.abs(hsv[:, :, 2].astype(np.float32) - v0) / v_tol
                dist = np.sqrt(dh * dh + ds * ds + dv * dv)
                best = np.minimum(best, dist.astype(np.float32))
            return best

        h0, s0, v0 = [float(v) for v in profile.get("mean_hsv", [0, 0, 0])]
        h_tol = max(1.0, float(getattr(self.cfg, "rule_ab_feature_h_tol", 14)))
        s_tol = max(1.0, float(getattr(self.cfg, "rule_ab_feature_s_tol", 75)))
        v_tol = max(1.0, float(getattr(self.cfg, "rule_ab_feature_v_tol", 75)))
        dh = self._hue_distance_180(hsv[:, :, 0], int(round(h0))).astype(np.float32) / h_tol
        ds = np.abs(hsv[:, :, 1].astype(np.float32) - s0) / s_tol
        dv = np.abs(hsv[:, :, 2].astype(np.float32) - v0) / v_tol
        return np.sqrt(dh * dh + ds * ds + dv * dv)

    def _raw_color_mask_from_profile(self, image_rgb: np.ndarray, profile: Dict[str, Any]) -> np.ndarray:
        """根据模板颜色阈值得到粗 mask。"""
        dist = self._hsv_color_distance_to_profile(image_rgb, profile)
        # dist<=sqrt(3) 等价于 H/S/V 都在容差量级内；适当收紧到 1.65 减少串色。
        raw = dist <= 1.65
        k = int(getattr(self.cfg, "rule_ab_feature_morph_kernel", 5))
        if k > 1:
            if k % 2 == 0:
                k += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            raw = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
            raw = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
        return raw.astype(bool)

    @staticmethod
    def _shape_similarity_64(mask: np.ndarray, profile: Dict[str, Any]) -> float:
        """候选连通域与首帧形状模板的 IoU，相似度 0~1。"""
        try:
            m = np.asarray(mask).astype(bool)
            if not np.any(m):
                return 0.0
            ys, xs = np.where(m)
            crop = m[int(ys.min()):int(ys.max()) + 1, int(xs.min()):int(xs.max()) + 1]
            cur = cv2.resize(crop.astype(np.uint8), (64, 64), interpolation=cv2.INTER_NEAREST).astype(bool)
            ref = np.asarray(profile.get("shape_template_64", []), dtype=np.uint8).astype(bool)
            if ref.shape != cur.shape:
                return 0.0
            inter = int(np.count_nonzero(cur & ref))
            uni = int(np.count_nonzero(cur | ref))
            return float(inter / max(1, uni))
        except Exception:
            return 0.0

    def _select_component_by_color_shape_center(self, raw_mask: np.ndarray, image_rgb: np.ndarray, profile: Dict[str, Any]) -> np.ndarray:
        """从颜色粗 mask 的连通域中选择最符合面积/中心/形状的主体。"""
        raw = np.asarray(raw_mask).astype(np.uint8)
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(raw, connectivity=8)
        out = np.zeros(raw.shape, dtype=bool)
        if num <= 1:
            return out

        ref_area = max(1.0, float(profile.get("area_px", 1.0)))
        label = str(profile.get("label", "")).lower()
        if label == "b":
            min_ratio = float(getattr(self.cfg, "rule_ab_b_feature_area_ratio_min", 0.03))
            max_ratio = float(getattr(self.cfg, "rule_ab_b_feature_area_ratio_max", 8.0))
        else:
            min_ratio = float(getattr(self.cfg, "rule_ab_feature_area_ratio_min", 0.25))
            max_ratio = float(getattr(self.cfg, "rule_ab_feature_area_ratio_max", 3.5))
        min_area = max(int(getattr(self.cfg, "rule_ab_feature_min_area_px", 20)), int(ref_area * min_ratio))
        max_area = int(ref_area * max_ratio)
        last_center = profile.get("last_center_xy") or profile.get("center_xy") or [0.0, 0.0]
        sigma = max(1.0, float(getattr(self.cfg, "rule_ab_feature_center_sigma_px", 180.0)))
        w_shape = float(getattr(self.cfg, "rule_ab_feature_shape_weight", 0.25))
        w_color = float(getattr(self.cfg, "rule_ab_feature_color_weight", 0.45))
        w_center = float(getattr(self.cfg, "rule_ab_feature_center_weight", 0.30))
        color_dist = self._hsv_color_distance_to_profile(image_rgb, profile)

        best_score = -1e18
        best_lab = -1
        for lab in range(1, num):
            area = int(stats[lab, cv2.CC_STAT_AREA])
            if area < min_area or area > max(1, max_area):
                continue
            comp = labels == lab
            cx, cy = float(centroids[lab][0]), float(centroids[lab][1])
            d_center = math.hypot(cx - float(last_center[0]), cy - float(last_center[1]))
            center_score = math.exp(-(d_center * d_center) / (2.0 * sigma * sigma))
            shape_score = self._shape_similarity_64(comp, profile)
            mean_color = float(np.mean(color_dist[comp])) if np.any(comp) else 999.0
            color_score = math.exp(-0.5 * mean_color * mean_color)
            area_score = math.exp(-abs(math.log(max(area, 1) / ref_area)))
            score = w_color * color_score + w_center * center_score + w_shape * shape_score + 0.10 * area_score
            if score > best_score:
                best_score = score
                best_lab = lab

        if best_lab >= 0:
            out = labels == best_lab
        return out.astype(bool)

    def _separate_ab_masks_by_color(self, image_rgb: np.ndarray, masks: Dict[str, np.ndarray], profiles: Dict[str, Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """A/B 部分重叠时，按颜色距离逐像素分配给 A 或 B。"""
        if not bool(getattr(self.cfg, "rule_ab_feature_ab_color_separation", True)):
            return masks
        if "a" not in masks or "b" not in masks or "a" not in profiles or "b" not in profiles:
            return masks

        a = np.asarray(masks["a"]).astype(bool)
        b = np.asarray(masks["b"]).astype(bool)
        overlap = a & b
        if not np.any(overlap):
            return masks

        da = self._hsv_color_distance_to_profile(image_rgb, profiles["a"])
        db = self._hsv_color_distance_to_profile(image_rgb, profiles["b"])
        assign_a = overlap & (da <= db)
        assign_b = overlap & (db < da)
        a2 = (a & ~overlap) | assign_a
        b2 = (b & ~overlap) | assign_b
        masks = dict(masks)
        masks["a"] = a2.astype(bool)
        masks["b"] = b2.astype(bool)
        return masks

    def _update_feature_profile_runtime(self, image_rgb: np.ndarray, profile: Dict[str, Any], mask: np.ndarray) -> Dict[str, Any]:
        """后续帧可选地缓慢更新颜色均值和 last_center，适应光照轻微变化。"""
        m = np.asarray(mask).astype(bool)
        if not np.any(m):
            return profile
        ys, xs = np.where(m)
        profile["last_center_xy"] = [float(xs.mean()), float(ys.mean())]
        profile["last_bbox_xyxy"] = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
        if bool(getattr(self.cfg, "rule_ab_feature_update_template_each_frame", False)):
            alpha = float(getattr(self.cfg, "rule_ab_feature_template_update_alpha", 0.08))
            alpha = max(0.0, min(1.0, alpha))
            hsv = cv2.cvtColor(cv2.cvtColor(np.asarray(image_rgb).astype(np.uint8), cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
            cur = np.mean(hsv[m], axis=0)
            old = np.asarray(profile.get("mean_hsv", cur), dtype=np.float32)
            # Hue 简单线性更新；若实验颜色稳定，alpha 很小即可。
            new = (1.0 - alpha) * old + alpha * cur.astype(np.float32)
            profile["mean_hsv"] = [float(x) for x in new]
        return profile

    def _track_abc_by_saved_features(
        self,
        image_rgb: np.ndarray,
        profiles: Dict[str, Dict[str, Any]],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        """用保存的 A/B/C 颜色和形状模板识别当前帧三目标。"""
        raw_masks: Dict[str, np.ndarray] = {}
        selected: Dict[str, np.ndarray] = {}
        debug: Dict[str, Dict[str, Any]] = {}
        for lab in ("a", "b", "c"):
            if lab not in profiles:
                continue
            raw = self._raw_color_mask_from_profile(image_rgb, profiles[lab])
            raw_masks[lab] = raw

        # A/B 先按颜色分离，再各自选主体；C 通常独立处理。
        if "a" in raw_masks and "b" in raw_masks:
            sep = self._separate_ab_masks_by_color(image_rgb, {"a": raw_masks["a"], "b": raw_masks["b"]}, profiles)
            raw_masks["a"], raw_masks["b"] = sep["a"], sep["b"]

        for lab, raw in raw_masks.items():
            m = self._select_component_by_color_shape_center(raw, image_rgb, profiles[lab])
            if not np.any(m):
                # 若连通域筛选失败，保留 raw 中最大连通域兜底。
                num, labels, stats, _ = cv2.connectedComponentsWithStats(raw.astype(np.uint8), connectivity=8)
                if num > 1:
                    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                    m = labels == best
            selected[lab] = m.astype(bool)
            center = self._mask_center_xy(m) if np.any(m) else None
            debug[lab] = {
                "raw_area_px": int(np.count_nonzero(raw)),
                "selected_area_px": int(np.count_nonzero(m)),
                "center_xy": [float(center[0]), float(center[1])] if center is not None else None,
                "score": None,
            }
            if np.any(m):
                profiles[lab] = self._update_feature_profile_runtime(image_rgb, profiles[lab], m)

        return selected, debug

    def _is_bmask_valid_for_feature_tracker(
        self,
        b_mask: Optional[np.ndarray],
        profile: Optional[Dict[str, Any]],
        debug_b: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str, float, Optional[Tuple[float, float]], float]:
        """判断本帧 Bmask 是否足够可信。返回 ok, reason, area, center, center_jump。"""
        if b_mask is None:
            return False, "bmask_none", 0.0, None, float("nan")
        m = np.asarray(b_mask).astype(bool)
        area = float(np.count_nonzero(m))
        min_area = max(1, int(getattr(self.cfg, "rule_ab_bmask_min_valid_area_px", 20)))
        if area < min_area:
            return False, f"area_too_small<{min_area}", area, None, float("nan")
        center = self._mask_center_xy(m)
        if center is None:
            return False, "center_none", area, None, float("nan")
        last_center = None
        if profile is not None:
            last_center = profile.get("last_center_xy") or profile.get("center_xy")
        center_jump = 0.0
        if last_center:
            try:
                center_jump = math.hypot(float(center[0]) - float(last_center[0]), float(center[1]) - float(last_center[1]))
                max_jump = float(getattr(self.cfg, "rule_ab_bmask_max_center_jump_px", getattr(self.cfg, "rule_ab_sam2_video_center_jump_max_px", 220.0)))
                if center_jump > max_jump:
                    return False, f"center_jump_too_large>{max_jump:.1f}", area, center, center_jump
            except Exception:
                center_jump = float("nan")
        return True, "ok", area, center, center_jump

    def _save_bmask_runtime_debug(
        self,
        tracker_state: Dict[str, Any],
        image_rgb: np.ndarray,
        frame_index: int,
        bmask_source: str,
        b_feature_raw_mask: Optional[np.ndarray] = None,
        b_feature_clean_mask: Optional[np.ndarray] = None,
        b_feature_selected_component: Optional[np.ndarray] = None,
        b_sam2_fallback_mask: Optional[np.ndarray] = None,
        b_final_mask: Optional[np.ndarray] = None,
        feature_area_px: int = 0,
        final_area_px: int = 0,
        center_x: Optional[float] = None,
        center_y: Optional[float] = None,
        center_jump_px: Optional[float] = None,
        score: Optional[float] = None,
        fallback_used: bool = False,
        hold_last_used: bool = False,
        fail_reason: str = "",
        profiles: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, str]:
        """保存每一帧 Bmask 检测 debug 图和 bmask_debug.csv。"""
        paths: Dict[str, str] = {}
        try:
            base = Path(str(tracker_state.get("out_dir", self._feature_tracker_output_dir()))) / "bmask_runtime_debug"
            base.mkdir(parents=True, exist_ok=True)
            prefix = f"cycle_{int(self.context.get('cycle_index', 0)):04d}_frame_{int(frame_index):06d}"
            img_bgr = cv2.cvtColor(np.asarray(image_rgb).astype(np.uint8), cv2.COLOR_RGB2BGR)
            raw_path = base / f"{prefix}_raw_frame.png"
            cv2.imwrite(str(raw_path), img_bgr)
            paths["raw_image_path"] = str(raw_path)

            def _write_mask(name: str, mask: Optional[np.ndarray]) -> str:
                if mask is None:
                    return ""
                path = base / f"{prefix}_{name}.png"
                cv2.imwrite(str(path), np.asarray(mask).astype(bool).astype(np.uint8) * 255)
                return str(path)

            paths["feature_mask_path"] = _write_mask("b_feature_raw_mask", b_feature_raw_mask)
            paths["feature_clean_mask_path"] = _write_mask("b_feature_clean_mask", b_feature_clean_mask)
            paths["feature_selected_component_path"] = _write_mask("b_feature_selected_component", b_feature_selected_component)
            paths["sam2_fallback_mask_path"] = _write_mask("b_sam2_fallback_mask", b_sam2_fallback_mask)
            paths["final_mask_path"] = _write_mask("b_final_mask", b_final_mask)

            overlay = img_bgr.copy()
            if b_feature_clean_mask is not None:
                overlay[np.asarray(b_feature_clean_mask).astype(bool)] = (0, 160, 255)
            if b_sam2_fallback_mask is not None:
                overlay[np.asarray(b_sam2_fallback_mask).astype(bool)] = (255, 0, 255)
            if b_final_mask is not None:
                overlay[np.asarray(b_final_mask).astype(bool)] = (0, 255, 0)
            overlay = cv2.addWeighted(overlay, 0.35, img_bgr, 0.65, 0)
            cv2.putText(overlay, f"B source={bmask_source} area={final_area_px} reason={fail_reason}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
            if center_x is not None and center_y is not None:
                cv2.circle(overlay, (int(round(center_x)), int(round(center_y))), 6, (0, 0, 255), -1)
            overlay_path = base / f"{prefix}_b_final_overlay.png"
            cv2.imwrite(str(overlay_path), overlay)
            paths["overlay_path"] = str(overlay_path)

            prof_b = (profiles or {}).get("b", {}) if isinstance(profiles, dict) else {}
            row = {
                "cycle_index": int(self.context.get("cycle_index", 0)),
                "frame_index": int(frame_index),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "bmask_source": bmask_source,
                "b_hsv_center_count": int(prof_b.get("b_hsv_center_count", len(prof_b.get("b_hsv_centers", []) or [])) or 0),
                "b_positive_hsv_seed_count": int(prof_b.get("b_positive_hsv_seed_count", len(prof_b.get("b_positive_hsv_seeds", []) or [])) or 0),
                "feature_area_px": int(feature_area_px),
                "final_area_px": int(final_area_px),
                "center_x": "" if center_x is None else float(center_x),
                "center_y": "" if center_y is None else float(center_y),
                "center_jump_px": "" if center_jump_px is None or (isinstance(center_jump_px, float) and math.isnan(center_jump_px)) else float(center_jump_px),
                "score": "" if score is None else float(score),
                "fallback_used": bool(fallback_used),
                "hold_last_used": bool(hold_last_used),
                "fail_reason": str(fail_reason),
                "raw_image_path": paths.get("raw_image_path", ""),
                "feature_mask_path": paths.get("feature_mask_path", ""),
                "sam2_fallback_mask_path": paths.get("sam2_fallback_mask_path", ""),
                "final_mask_path": paths.get("final_mask_path", ""),
                "overlay_path": paths.get("overlay_path", ""),
            }
            csv_path = base / "bmask_debug.csv"
            write_header = not csv_path.exists()
            with csv_path.open("a", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
            paths["bmask_debug_csv"] = str(csv_path)
        except Exception as e:
            self.log(f"[Bmask] 保存 runtime debug 失败：{e}")
        return paths

    def _apply_feature_tracker_masks_to_scene(
        self,
        follower: Any,
        scene: Any,
        image_rgb: np.ndarray,
        masks: Dict[str, np.ndarray],
        debug: Optional[Dict[str, Dict[str, Any]]] = None,
        tracker_state: Optional[Dict[str, Any]] = None,
        source: str = "feature_tracker",
    ) -> Any:
        """
        把颜色/形状模板识别到的 A/B/C mask 写回同一个 scene。

        关键点：
            - Step7 路线运动读取 scene.a.center；
            - AB 最近边角度读取同一个 scene.a.center 和 scene.b.mask；
            - overlay 保存也读取同一个 scene；
          因此这里是统一 A 身份的唯一出口，禁止再从旧 SAM2 scene 单独取 A。
        """
        debug = debug or {}
        if scene is None:
            return scene
        for lab, mask in masks.items():
            obj = self._get_scene_object_by_label(scene, lab)
            if obj is None:
                continue
            mb = np.asarray(mask).astype(bool)
            self._set_mask_to_scene_object(obj, mb)
            center = self._mask_center_xy(mb)
            # 给外部 save_annotated_image / 日志 / meta 一个明确来源。
            try:
                setattr(obj, "source", source)
                setattr(obj, "mask_source", source)
                setattr(obj, "segmentation_source", source)
                setattr(obj, "label", str(lab).upper())
                if center is not None:
                    setattr(obj, "center", (float(center[0]), float(center[1])))
                    setattr(obj, "center_xy", (float(center[0]), float(center[1])))
            except Exception:
                pass
            try:
                if isinstance(obj, dict):
                    obj["source"] = source
                    obj["mask_source"] = source
                    obj["segmentation_source"] = source
                    obj["label"] = str(lab).upper()
                    if center is not None:
                        obj["center"] = (float(center[0]), float(center[1]))
                        obj["center_xy"] = (float(center[0]), float(center[1]))
            except Exception:
                pass

            # 同步常见缓存字段，避免外部模块仍读取旧 SAM2 last_a_mask/current_a_mask。
            for holder in (follower, getattr(follower, "abc_segmenter", None), getattr(follower, "segmenter", None)):
                if holder is None:
                    continue
                for attr in (f"last_{lab}_mask", f"current_{lab}_mask", f"{lab}_mask", f"last_mask_{lab}"):
                    try:
                        setattr(holder, attr, mb.copy())
                    except Exception:
                        pass
                if center is not None:
                    for attr in (f"last_{lab}_center", f"current_{lab}_center", f"{lab}_center"):
                        try:
                            setattr(holder, attr, (float(center[0]), float(center[1])))
                        except Exception:
                            pass

        try:
            setattr(scene, "source", source)
            setattr(scene, "mask_source", source)
            setattr(scene, "segmentation_source", source)
            setattr(scene, "a_source", source)
            setattr(scene, "b_source", source)
            setattr(scene, "c_source", source)
        except Exception:
            pass
        if isinstance(scene, dict):
            try:
                scene["source"] = source
                scene["mask_source"] = source
                scene["segmentation_source"] = source
                scene["a_source"] = source
                scene["b_source"] = source
                scene["c_source"] = source
            except Exception:
                pass

        try:
            setattr(follower, "last_scene", scene)
            setattr(follower, "current_scene", scene)
            setattr(follower, "abc_feature_tracker_debug", debug)
            setattr(follower, "abc_identity_source", source)
            setattr(follower, "a_source", source)
            setattr(follower, "b_source", source)
        except Exception:
            pass

        try:
            self.context["feature_tracker_scene"] = scene
            self.context["feature_tracker_a_center"] = list(self._get_scene_a_center_xy(scene) or [])
            self.context["feature_tracker_debug"] = self._json_safe(debug)
            self.context["abc_feature_tracker_last_source"] = source
            if tracker_state is not None:
                tracker_state["last_scene"] = scene
                tracker_state["last_image_rgb"] = np.asarray(image_rgb).copy()
                tracker_state["last_debug"] = debug
        except Exception:
            pass
        return scene

    def _ensure_rule_ab_feature_tracker_installed(self, follower: Any, state: Optional[CalibrationState] = None) -> None:
        """完整测量中每次取 A/B 前都确保 capture_and_build_scene 已被特征跟踪器接管。"""
        if follower is None:
            return
        if not bool(getattr(self.cfg, "rule_ab_use_feature_tracker_after_sam2_init", True)):
            return
        if not getattr(follower, "_abc_feature_tracker_installed", False):
            self._install_feature_tracker_on_rule_ab_follower(follower, state or self._get_loaded_calibration_state())

    def _install_feature_tracker_on_rule_ab_follower(self, follower: Any, state: Optional[CalibrationState] = None):
        """
        把 RuleAB follower 的 capture_and_build_scene() 替换成：
            首次：使用原 SAM2 scene 建立 A/B/C 颜色+形状模板；
            后续：只截图并用模板识别 A/B/C，不再调用 SAM2。
        """
        if follower is None:
            return
        if not bool(getattr(self.cfg, "rule_ab_use_feature_tracker_after_sam2_init", True)):
            self.log("[ABC特征跟踪] 已关闭 rule_ab_use_feature_tracker_after_sam2_init，仍使用原 SAM2/temporal 路径。")
            return
        if getattr(follower, "_abc_feature_tracker_installed", False):
            return
        if not hasattr(follower, "capture_and_build_scene"):
            self.log("[ABC特征跟踪] follower 没有 capture_and_build_scene，无法安装特征跟踪器。")
            return

        original_capture = follower.capture_and_build_scene
        tracker_state: Dict[str, Any] = {
            "initialized": False,
            "profiles": {},
            "base_scene": None,
            "frame_index": 0,
            "out_dir": str(self._feature_tracker_output_dir()),
            "original_capture": original_capture,
        }

        def _build_templates_from_original_scene() -> Tuple[np.ndarray, Any]:
            # 只在这里允许调用一次原始 SAM2：用于首帧 A/B/C 建模板。
            # 但截图/显示后端偶发可能返回半帧：
            #   backend exception: 'read returned less data than expected'
            # 因此首帧 original_capture 需要局部重试，不能一次失败就终止完整测量。
            attempts = max(1, int(getattr(self.cfg, "rule_ab_frame_read_retry_attempts", 5)))
            interval_s = max(0.0, float(getattr(self.cfg, "rule_ab_frame_read_retry_interval_s", 0.15)))
            last_exc: Optional[BaseException] = None
            for attempt in range(1, attempts + 1):
                try:
                    image_rgb, sam2_scene = original_capture()
                    if attempt > 1:
                        self.log(f"[ABC特征跟踪] 首帧 original_capture 第 {attempt}/{attempts} 次成功。")
                    break
                except Exception as e:
                    last_exc = e
                    self.log(
                        f"[ABC特征跟踪] 首帧 original_capture 失败 attempt={attempt}/{attempts}: {e}；"
                        "等待下一帧重试。"
                    )
                    if attempt < attempts:
                        time.sleep(interval_s)
            else:
                raise RuntimeError(
                    f"[ABC特征跟踪] 首帧 original_capture 连续失败 {attempts} 次，最后错误：{last_exc}"
                ) from last_exc
            profiles: Dict[str, Dict[str, Any]] = {}
            raw_sam2_masks: Dict[str, np.ndarray] = {}
            for lab in ("a", "b", "c"):
                obj = self._get_scene_object_by_label(sam2_scene, lab)
                mask = self._extract_mask_from_scene_object(obj)
                if mask is None or not np.any(np.asarray(mask).astype(bool)):
                    raise RuntimeError(f"首帧 SAM2 没有得到 {lab.upper()} mask，不能建立特征模板。")
                raw_sam2_masks[lab] = np.asarray(mask).astype(bool)

            # 关键修复：不能直接用 raw SAM2 mask 建模板。
            # raw A mask 可能混入 B、比例尺、文字或边框；必须先按对应正点连通域清理。
            positive_points = self._feature_initial_positive_points_from_state(state)
            cleaned_masks, clean_debug = self._clean_initial_abc_masks_by_positive_components(
                image_rgb=image_rgb,
                raw_masks=raw_sam2_masks,
                positive_points=positive_points,
            )

            # 首帧 cleaned B mask 已得到：如果启用 KLT 指定边角度，
            # 在这里人工指定一次 B 物理边，并初始化跨轮次 KLT 状态。
            if bool(getattr(self.cfg, "rule_ab_use_klt_selected_b_edge_angle", True)):
                self._ensure_initial_b_edge_klt_selected_and_initialized(image_rgb, cleaned_masks["b"], state=state)

            for lab in ("a", "b", "c"):
                profiles[lab] = self._extract_feature_profile_from_mask(image_rgb, cleaned_masks[lab], lab)
                profiles[lab]["initial_mask_source"] = "sam2_raw_cleaned_by_positive_component"
                profiles[lab]["positive_points_used_for_cleaning"] = [
                    [float(x), float(y)] for x, y in positive_points.get(lab, [])
                ]
                profiles[lab]["clean_debug"] = clean_debug.get("labels", {}).get(lab, {})

            # B 多颜色模板：优先使用完整标定里保存的 B 正点 HSV seed；
            # 若旧标定文件没有 seed，则从当前首帧 B 正点附近即时提取。
            b_seeds: List[List[float]] = []
            try:
                if state is not None:
                    b_seeds = [
                        [float(v[0]), float(v[1]), float(v[2])]
                        for v in getattr(state, "b_positive_hsv_seeds", [])
                        if len(v) >= 3
                    ]
            except Exception:
                b_seeds = []
            if not b_seeds:
                b_seeds = self._extract_b_positive_hsv_seeds_from_points(image_rgb, positive_points.get("b", []), label="B")
                if state is not None:
                    try:
                        state.b_positive_hsv_seeds = [[float(x), float(y), float(z)] for x, y, z in b_seeds]
                        state.b_positive_hsv_seed_count = int(len(b_seeds))
                    except Exception:
                        pass
            b_centers, b_color_debug = self._build_b_hsv_centers_from_seeds_and_mask(
                image_rgb=image_rgb,
                b_mask=cleaned_masks["b"],
                b_positive_hsv_seeds=b_seeds,
            )
            profiles["b"]["b_positive_hsv_seeds"] = b_seeds
            profiles["b"]["b_positive_hsv_seed_count"] = int(len(b_seeds))
            profiles["b"]["b_hsv_centers"] = b_centers
            profiles["b"]["b_hsv_center_count"] = int(len(b_centers))
            profiles["b"]["b_hsv_center_source"] = "positive_seeds+clean_mask"
            profiles["b"]["b_hsv_center_tolerance"] = {
                "h": int(getattr(self.cfg, "rule_ab_b_feature_h_tol", getattr(self.cfg, "rule_ab_feature_h_tol", 14))),
                "s": int(getattr(self.cfg, "rule_ab_b_feature_s_tol", getattr(self.cfg, "rule_ab_feature_s_tol", 75))),
                "v": int(getattr(self.cfg, "rule_ab_b_feature_v_tol", getattr(self.cfg, "rule_ab_feature_v_tol", 75))),
            }
            profiles["b"]["b_color_center_debug"] = b_color_debug

            out_dir = Path(tracker_state["out_dir"])
            saved = self._save_feature_templates_debug(
                image_rgb=image_rgb,
                profiles=profiles,
                masks=cleaned_masks,
                out_dir=out_dir,
                raw_masks=raw_sam2_masks,
                clean_debug=clean_debug,
            )

            # 首帧直接返回 cleaned mask 写回的 feature_tracker_scene。
            # 不在首帧再跑一次颜色阈值，否则可能重新把比例尺/文字等颜色相近区域选入 A。
            debug = {
                lab: {
                    "raw_sam2_area_px": int(np.count_nonzero(raw_sam2_masks.get(lab, np.zeros_like(next(iter(cleaned_masks.values())))))),
                    "cleaned_area_px": int(np.count_nonzero(cleaned_masks[lab])),
                    "used_initial_cleaned_mask": True,
                    "clean_debug": clean_debug.get("labels", {}).get(lab, {}),
                }
                for lab in ("a", "b", "c")
            }
            feature_masks = {lab: cleaned_masks[lab].copy() for lab in ("a", "b", "c")}

            feature_scene = copy.deepcopy(sam2_scene)
            feature_scene = self._apply_feature_tracker_masks_to_scene(
                follower=follower,
                scene=feature_scene,
                image_rgb=image_rgb,
                masks=feature_masks,
                debug=debug,
                tracker_state=tracker_state,
                source="feature_tracker",
            )

            tracker_state["profiles"] = profiles
            tracker_state["base_scene"] = copy.deepcopy(feature_scene)
            tracker_state["initialized"] = True
            tracker_state["saved"] = saved
            tracker_state["first_frame_from_sam2_then_feature_scene"] = True
            try:
                self.context["abc_feature_tracker"] = {
                    "enabled": True,
                    "profiles_json": saved.get("profiles_json", ""),
                    "overlay_path": saved.get("overlay_path", ""),
                    "out_dir": str(out_dir),
                    "mode": "sam2_once_then_color_shape_tracking",
                    "scene_source": "feature_tracker",
                }
                if state is not None and isinstance(state, CalibrationState):
                    state.feature_profile_dir = str(out_dir)
                    state.feature_profile_json = saved.get("profiles_json", "")
                    self.context["calibration_state"] = state.to_dict()
            except Exception:
                pass
            self.log(
                "[ABC特征跟踪] 已用首帧 SAM2 建立 A/B/C 模板；首帧开始即返回 feature_tracker_scene，"
                "后续完整测量仅使用 feature tracker；Bmask 失败不 SAM2 fallback、不 hold_last；"
                f"模板目录={out_dir}"
            )
            return image_rgb, feature_scene

        def _feature_capture_and_build_scene():
            if not tracker_state.get("initialized", False):
                return _build_templates_from_original_scene()

            out_dir = Path(tracker_state["out_dir"]) / "feature_runtime_capture"
            image_rgb = self._capture_current_rule_ab_frame(output_dir=out_dir)
            profiles = tracker_state.get("profiles", {})
            masks, debug = self._track_abc_by_saved_features(image_rgb, profiles)
            tracker_state["profiles"] = profiles
            tracker_state["frame_index"] = int(tracker_state.get("frame_index", 0)) + 1
            frame_index = int(tracker_state.get("frame_index", 0))

            # -------- Bmask feature-only 流程：feature -> failed --------
            # 关键约束：
            #   1) 后续帧 Bmask 只能来自 feature tracker；
            #   2) feature tracker 失败时，本帧 Bmask 置空，角度检测失败；
            #   3) 不调用原始 SAM2/original_capture，不使用 SAM2 fallback；
            #   4) 不使用上一帧 last_valid_b_mask / hold_last；
            #   5) 当前版本不再执行角度修复；Step1/Step7 使用 YOLO-OBB 原始角度。
            if not isinstance(masks, dict):
                masks = {}
            b_profile = profiles.get("b", {}) if isinstance(profiles, dict) else {}
            b_raw_mask = self._raw_color_mask_from_profile(image_rgb, b_profile) if b_profile else None
            b_clean_mask = b_raw_mask.copy() if b_raw_mask is not None else None
            b_feature_mask = masks.get("b") if isinstance(masks, dict) else None
            ok_b, reason_b, area_b, center_b, center_jump_b = self._is_bmask_valid_for_feature_tracker(
                b_feature_mask, b_profile, debug.get("b", {}) if isinstance(debug, dict) else {}
            )

            final_b_mask = np.asarray(b_feature_mask).astype(bool) if ok_b and b_feature_mask is not None else None
            bmask_source = "feature" if ok_b else "failed"
            fallback_used = False
            hold_last_used = False
            sam2_fallback_mask = None
            fail_reason = "" if ok_b else reason_b

            if ok_b:
                self.log(f"[Bmask] source=feature, area={int(area_b)}, center_jump={center_jump_b}")
                masks["b"] = final_b_mask.astype(bool)
                # 可以记录最近一次有效 Bmask 供调试查看，但后续失败时绝不读取它参与检测。
                tracker_state["last_valid_b_mask"] = final_b_mask.copy()
                tracker_state["last_valid_b_center"] = self._mask_center_xy(final_b_mask)
                tracker_state["bmask_hold_count"] = 0
                tracker_state["bmask_consecutive_fail_count"] = 0
                if "b" in profiles:
                    profiles["b"] = self._update_feature_profile_runtime(image_rgb, profiles["b"], final_b_mask)
            else:
                tracker_state["bmask_consecutive_fail_count"] = int(tracker_state.get("bmask_consecutive_fail_count", 0)) + 1
                fail_count = int(tracker_state.get("bmask_consecutive_fail_count", 0))
                bmask_source = "failed"
                self.log(
                    f"[Bmask] source=failed, reason={fail_reason}, "
                    f"feature_area={int(area_b)}, consecutive_fail={fail_count}; "
                    "不调用 SAM2 fallback，不使用 hold_last/上一帧 Bmask。"
                )
                # 给 scene 写入空 Bmask，让角度检测得到 angle_ok=False，
                # 而不是读取旧 Bmask 或上一帧 Bmask。
                masks["b"] = np.zeros(image_rgb.shape[:2], dtype=bool)
                # 按需求：feature tracker 失败只记录本帧失败，不因为连续失败直接终止完整测量。
                max_fail = int(getattr(self.cfg, "rule_ab_bmask_fail_max_consecutive", 10))
                if fail_count > max_fail:
                    self.log(
                        f"[Bmask] 连续失败 {fail_count} 帧，已超过阈值 {max_fail}；"
                        "仍不终止完整测量，仅继续记录 angle_ok=False。"
                    )

            final_center = self._mask_center_xy(final_b_mask) if final_b_mask is not None else None
            final_area = int(np.count_nonzero(final_b_mask)) if final_b_mask is not None else 0
            self._save_bmask_runtime_debug(
                tracker_state=tracker_state,
                image_rgb=image_rgb,
                frame_index=frame_index,
                bmask_source=bmask_source,
                b_feature_raw_mask=b_raw_mask,
                b_feature_clean_mask=b_clean_mask,
                b_feature_selected_component=b_feature_mask,
                b_sam2_fallback_mask=sam2_fallback_mask,
                b_final_mask=final_b_mask,
                feature_area_px=int(np.count_nonzero(b_feature_mask)) if b_feature_mask is not None else 0,
                final_area_px=final_area,
                center_x=None if final_center is None else float(final_center[0]),
                center_y=None if final_center is None else float(final_center[1]),
                center_jump_px=center_jump_b,
                score=debug.get("b", {}).get("score") if isinstance(debug, dict) else None,
                fallback_used=fallback_used,
                hold_last_used=hold_last_used,
                fail_reason=fail_reason,
                profiles=profiles,
            )
            if isinstance(debug, dict):
                debug.setdefault("b", {})["bmask_source"] = bmask_source
                debug.setdefault("b", {})["fallback_used"] = bool(fallback_used)
                debug.setdefault("b", {})["hold_last_used"] = bool(hold_last_used)
                debug.setdefault("b", {})["fail_reason"] = fail_reason
                debug.setdefault("b", {})["final_area_px"] = final_area

            scene = copy.deepcopy(tracker_state.get("base_scene"))
            if scene is None:
                raise RuntimeError("ABC feature tracker base_scene 缺失；为避免 A/B 回到旧 SAM2，本帧不允许回退原始 SAM2。")

            scene = self._apply_feature_tracker_masks_to_scene(
                follower=follower,
                scene=scene,
                image_rgb=image_rgb,
                masks=masks,
                debug=debug,
                tracker_state=tracker_state,
                source=f"feature_tracker_b_{bmask_source}",
            )
            return image_rgb, scene

        follower._original_capture_and_build_scene_before_feature_tracker = original_capture
        follower.capture_and_build_scene = _feature_capture_and_build_scene
        follower._abc_feature_tracker_installed = True
        follower._abc_feature_tracker_state = tracker_state
        self.log("[ABC特征跟踪] 已安装：SAM2 仅用于首帧模板建立；后续帧仅使用 feature tracker；Bmask 失败不 fallback、不 hold_last；角度修复已禁用，直接使用 YOLO-OBB 原始角度。")

    def ensure_rule_ac_controller(self) -> RuleACOverlapController:
        """
        旧 RuleAC 面积阈值控制器入口，保留兼容。

        当前 Step9 已经改为“颜色检测区域中心对齐”，
        完整流程和 GUI 单独测试默认调用 run_rule_ac_until_threshold()
        中的新中心对齐逻辑，不再依赖 RuleACOverlapController 的面积阈值。
        """
        if self.rule_ac_controller is not None:
            return self.rule_ac_controller

        cfg = RuleACConfig(
            area_threshold_px=float(self.cfg.rule_ac_area_threshold_px),
            use_um2_threshold=False,
            um_per_px=None,
            capture_area=self.cfg.capture_area,
            output_dir=str(self.output_root / "rule_ac_from_measurement"),
            sample_config_path="sample_config_rule_ac_measurement.json",
            enable_stage=bool(self.cfg.rule_ac_enable_stage),
            stage_device_id=str(self.cfg.rule_ac_stage_device_id),
            stage_axis=str(self.cfg.rule_ac_stage_axis),
            stage_direction=int(self.cfg.rule_ac_stage_direction),
            stage_step_size=int(self.cfg.rule_ac_stage_step_size),
            max_cycles=int(self.cfg.rule_ac_max_cycles),
            stop_when_threshold_reached=True,
        )
        self.log(
            "[RuleAC-legacy] 初始化旧面积阈值控制模块："
            f"threshold_px={cfg.area_threshold_px}, enable_stage={cfg.enable_stage}"
        )
        self.rule_ac_controller = RuleACOverlapController(cfg)
        return self.rule_ac_controller

    @staticmethod
    def _extract_mask_from_scene_object(obj: Any) -> Optional[np.ndarray]:
        """
        从 scene.a / scene.b / scene.c 这类对象里提取 mask。

        兼容：
            obj.mask
            obj["mask"]
            直接传入 np.ndarray
        """
        if obj is None:
            return None

        if isinstance(obj, np.ndarray):
            return obj.astype(bool)

        try:
            m = getattr(obj, "mask", None)
            if m is not None:
                return np.asarray(m).astype(bool)
        except Exception:
            pass

        try:
            if isinstance(obj, dict) and obj.get("mask") is not None:
                return np.asarray(obj.get("mask")).astype(bool)
        except Exception:
            pass

        return None

    @staticmethod
    def _mask_center_xy(mask: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        计算二值 mask 的像素中心，返回截图区域内坐标 (x, y)。
        """
        if mask is None:
            return None
        m = np.asarray(mask).astype(bool)
        if m.size <= 0 or not np.any(m):
            return None
        ys, xs = np.where(m)
        if len(xs) <= 0:
            return None
        return float(np.mean(xs)), float(np.mean(ys))

    @staticmethod
    def _normalize_points_xy(points: Any) -> List[Tuple[float, float]]:
        """把 [[x,y], ...] / [(x,y), ...] 统一为 [(x,y), ...]。"""
        out: List[Tuple[float, float]] = []
        if not points:
            return out
        for p in points:
            try:
                out.append((float(p[0]), float(p[1])))
            except Exception:
                continue
        return out

    def _feature_initial_positive_points_from_state(
        self,
        state: Optional[CalibrationState],
    ) -> Dict[str, List[Tuple[float, float]]]:
        """
        取 A/B/C 初始模板清理用的正点。

        A：RuleAB-A 正点；
        B：RuleAB-B 正点，若没有则复用 Angle-B 正点；
        C：全局 C 正点。
        """
        if state is None:
            state = self._get_loaded_calibration_state()
        pts: Dict[str, List[Tuple[float, float]]] = {"a": [], "b": [], "c": []}
        if state is None:
            return pts
        pts["a"] = self._normalize_points_xy(getattr(state, "rule_ab_a_positive_points", []))
        pts["b"] = self._normalize_points_xy(getattr(state, "rule_ab_b_positive_points", []))
        if not pts["b"]:
            pts["b"] = self._normalize_points_xy(getattr(state, "angle_b_positive_points", []))
        pts["c"] = self._normalize_points_xy(getattr(state, "global_c_positive_points", []))
        return pts

    @staticmethod
    def _component_hit_count_for_points(labels: np.ndarray, comp_id: int, points_xy: List[Tuple[float, float]]) -> int:
        """统计某个连通域覆盖了多少个正点。"""
        if labels is None or not points_xy:
            return 0
        h, w = labels.shape[:2]
        hits = 0
        for x, y in points_xy:
            xi, yi = int(round(float(x))), int(round(float(y)))
            if 0 <= xi < w and 0 <= yi < h and int(labels[yi, xi]) == int(comp_id):
                hits += 1
        return hits

    @staticmethod
    def _component_distance_to_points(
        centroid_xy: Tuple[float, float],
        points_xy: List[Tuple[float, float]],
    ) -> float:
        """连通域中心到正点均值的距离；没有正点时返回 0。"""
        if not points_xy:
            return 0.0
        px = float(np.mean([p[0] for p in points_xy]))
        py = float(np.mean([p[1] for p in points_xy]))
        return float(math.hypot(float(centroid_xy[0]) - px, float(centroid_xy[1]) - py))

    def _select_initial_component_by_positive_points(
        self,
        raw_mask: np.ndarray,
        points_xy: List[Tuple[float, float]],
        label: str,
        min_area_px: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        从 SAM2 初始 mask 的多个连通域中，只保留对应目标正点所在的主体连通域。

        选择规则：
            1) 优先保留覆盖正点数量最多的连通域；
            2) 若没有任何连通域覆盖正点，则保留离正点均值最近的连通域；
            3) 只返回一个连通域，避免 A 模板混入 B、比例尺、文字或边框。
        """
        raw = np.asarray(raw_mask).astype(bool)
        out = np.zeros(raw.shape, dtype=bool)
        debug: Dict[str, Any] = {
            "label": str(label),
            "raw_area_px": int(np.count_nonzero(raw)),
            "positive_points": [[float(x), float(y)] for x, y in points_xy],
            "selected_component": -1,
            "selected_area_px": 0,
            "selected_pos_hits": 0,
            "component_count": 0,
            "reason": "",
        }
        if raw.size <= 0 or not np.any(raw):
            debug["reason"] = "empty_raw_mask"
            return out, debug

        min_area = int(min_area_px if min_area_px is not None else getattr(self.cfg, "rule_ab_feature_min_area_px", 20))
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(raw.astype(np.uint8), connectivity=8)
        debug["component_count"] = int(max(0, num - 1))
        if num <= 1:
            debug["reason"] = "no_foreground_component"
            return out, debug

        best_id = -1
        best_key: Optional[Tuple[int, float, int]] = None
        component_debug: List[Dict[str, Any]] = []
        for comp_id in range(1, num):
            area = int(stats[comp_id, cv2.CC_STAT_AREA])
            if area < max(1, min_area):
                continue
            cx, cy = float(centroids[comp_id][0]), float(centroids[comp_id][1])
            hits = self._component_hit_count_for_points(labels, comp_id, points_xy)
            dist = self._component_distance_to_points((cx, cy), points_xy)
            component_debug.append({
                "id": int(comp_id),
                "area_px": area,
                "centroid_xy": [cx, cy],
                "pos_hits": int(hits),
                "dist_to_pos_mean_px": float(dist),
            })
            # tuple 越大越好：正点命中优先；距离越小越好；面积越大越好
            key = (int(hits), -float(dist), int(area))
            if best_key is None or key > best_key:
                best_key = key
                best_id = comp_id

        debug["components"] = component_debug
        if best_id < 0:
            debug["reason"] = "all_components_below_min_area"
            return out, debug

        out = labels == best_id
        debug["selected_component"] = int(best_id)
        debug["selected_area_px"] = int(np.count_nonzero(out))
        debug["selected_pos_hits"] = int(self._component_hit_count_for_points(labels, best_id, points_xy))
        if points_xy and debug["selected_pos_hits"] <= 0:
            debug["reason"] = "selected_nearest_component_no_positive_hit"
        else:
            debug["reason"] = "selected_component_with_most_positive_hits"
        return out.astype(bool), debug

    @staticmethod
    def _dilate_bool_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
        """二值 mask 膨胀。radius_px<=0 时原样返回。"""
        m = np.asarray(mask).astype(bool)
        r = int(radius_px)
        if r <= 0 or not np.any(m):
            return m
        k = 2 * r + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.dilate(m.astype(np.uint8), kernel, iterations=1).astype(bool)

    def _clean_initial_abc_masks_by_positive_components(
        self,
        image_rgb: np.ndarray,
        raw_masks: Dict[str, np.ndarray],
        positive_points: Dict[str, List[Tuple[float, float]]],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        首帧 SAM2 建模板前清理 A/B/C 初始 mask。

        关键目的：
            - A/B/C 都只保留对应正点所在主体连通域；
            - A 额外排除 B/C，防止 A 模板混入 B/C 接触区域；
            - 远离正点的比例尺、文字、边框等连通域会被删除；
            - cleaned mask 异常时直接 raise，禁止继续 Step7。
        """
        cleaned: Dict[str, np.ndarray] = {}
        debug: Dict[str, Any] = {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "labels": {},
        }

        min_area = int(getattr(self.cfg, "rule_ab_feature_min_area_px", 20))

        # 先清理 B/C。A 后续要排除 cleaned B/C。
        for lab in ("b", "c"):
            if lab not in raw_masks:
                continue
            pts = positive_points.get(lab, [])
            cm, info = self._select_initial_component_by_positive_points(raw_masks[lab], pts, lab, min_area)
            if not np.any(cm):
                raise RuntimeError(
                    f"[ABC特征跟踪] 首帧 {lab.upper()} cleaned mask 为空，不能建立特征模板；"
                    f"debug={info}"
                )
            cleaned[lab] = cm.astype(bool)
            debug["labels"][lab] = info

        # A 必须排除 B/C 后，再按 A 正点选择连通域。
        if "a" in raw_masks:
            a_raw = np.asarray(raw_masks["a"]).astype(bool)
            exclude = np.zeros(a_raw.shape, dtype=bool)
            exclude_radius = int(getattr(self.cfg, "rule_ab_a_exclude_b_dilate_px", 5))
            for other in ("b", "c"):
                if other in cleaned:
                    exclude |= self._dilate_bool_mask(cleaned[other], exclude_radius)

            a_preclean = a_raw & (~exclude)
            # 跳过b/c膨胀
            # a_preclean = a_raw

            a_pts = positive_points.get("a", [])
            a_clean, a_info = self._select_initial_component_by_positive_points(a_preclean, a_pts, "a", min_area)
            a_info["raw_area_before_exclude_px"] = int(np.count_nonzero(a_raw))
            a_info["area_after_exclude_bc_px"] = int(np.count_nonzero(a_preclean))
            a_info["exclude_bc_dilate_px"] = int(exclude_radius)

            
            if not np.any(a_clean):
                raise RuntimeError(
                    "[ABC特征跟踪] 首帧 A cleaned mask 为空。"
                    "A 初始 SAM2 mask 可能没有覆盖 A 正点，或被 B/C 排除后无有效区域；"
                    f"debug={a_info}"
                )

        '''
            # A 与 B/C 的重叠硬检查。
            overlap_area = 0
            for other in ("b", "c"):
                if other in cleaned:
                    overlap_area += int(np.count_nonzero(a_clean & cleaned[other]))
            a_area = int(np.count_nonzero(a_clean))
            overlap_ratio = float(overlap_area / max(1, a_area))
            max_overlap_ratio = float(getattr(self.cfg, "rule_ab_feature_initial_a_overlap_max_ratio", 0.03))
            a_info["overlap_with_cleaned_bc_area_px"] = int(overlap_area)
            a_info["overlap_with_cleaned_bc_ratio"] = float(overlap_ratio)
            a_info["max_allowed_overlap_ratio"] = float(max_overlap_ratio)
            
            if overlap_ratio > max_overlap_ratio:
                raise RuntimeError(
                    f"[ABC特征跟踪] 首帧 A cleaned mask 与 B/C 重叠过大："
                    f"overlap_ratio={overlap_ratio:.4f} > {max_overlap_ratio:.4f}；"
                    "为避免 A 模板污染，停止完整测量。"
                )
            

            # A 面积硬检查：过小或过大都不允许进入 Step7。
            min_a_area = int(getattr(self.cfg, "rule_ab_feature_initial_a_min_area_px", min_area))
            max_a_area = int(getattr(self.cfg, "rule_ab_feature_initial_a_max_area_px", 200000))
            a_info["min_allowed_area_px"] = int(min_a_area)
            a_info["max_allowed_area_px"] = int(max_a_area)
            if a_area < min_a_area or a_area > max_a_area:
                raise RuntimeError(
                    f"[ABC特征跟踪] 首帧 A cleaned mask 面积异常：area={a_area}, "
                    f"allowed=[{min_a_area}, {max_a_area}]；停止完整测量。"
                )

            cleaned["a"] = a_clean.astype(bool)
            debug["labels"]["a"] = a_info
        '''
        

        # 如果还有缺失标签，按原始 mask 清理一次并要求有效。
        for lab in ("a", "b", "c"):
            if lab in cleaned:
                continue
            if lab not in raw_masks:
                raise RuntimeError(f"[ABC特征跟踪] 首帧缺少 {lab.upper()} raw mask，不能建立特征模板。")
            pts = positive_points.get(lab, [])
            cm, info = self._select_initial_component_by_positive_points(raw_masks[lab], pts, lab, min_area)
            if not np.any(cm):
                raise RuntimeError(f"[ABC特征跟踪] 首帧 {lab.upper()} cleaned mask 为空；debug={info}")
            cleaned[lab] = cm.astype(bool)
            debug["labels"][lab] = info

        '''
        # 三个 cleaned mask 两两重叠检查，防止模板互相污染。
        for lab1, lab2 in (("a", "b"), ("a", "c"), ("b", "c")):
            if lab1 in cleaned and lab2 in cleaned:
                area1 = int(np.count_nonzero(cleaned[lab1]))
                area2 = int(np.count_nonzero(cleaned[lab2]))
                ov = int(np.count_nonzero(cleaned[lab1] & cleaned[lab2]))
                ratio = float(ov / max(1, min(area1, area2)))
                debug[f"overlap_{lab1}_{lab2}"] = {"area_px": ov, "ratio_to_min_area": ratio}
                if lab1 == "a" and ov > 0:
                    raise RuntimeError(
                        f"[ABC特征跟踪] 首帧 cleaned A 与 {lab2.upper()} 仍有重叠：overlap={ov}px；"
                        "为避免 A 模板污染，停止完整测量。"
                    )
        '''

        return cleaned, debug

    def _get_or_create_step9_follower_for_bc(self) -> ActualNanoBoundaryFollower:
        """
        Step9 需要 B/C mask，因此复用 RuleAB 的 SAM2 A/B/C 分割模块。

        如果当前已经有 rule_ab_follower，就直接复用；
        如果没有，则初始化一个 follower。是否真动 Stage34 仍由 rule_ab_enable_stage 控制，
        但 Step9 本身只会调用 1/2 通道，不会调用 3/4 通道动作。
        """
        return self.ensure_rule_ab_follower()

    def _get_stage12_device(self):
        """
        获取用于 Step9 的 1/2 通道控制对象。

        优先复用 RuleAB follower 中已经打开的同一个 Kinesis 控制器，
        避免同一个序列号被重复打开。若没有可复用对象，则单独打开
        Thorlabs.KinesisPiezoMotor。
        """
        # 1) 优先复用 RuleAB follower 的 Stage34 封装。
        try:
            if self.rule_ab_follower is not None:
                st = getattr(self.rule_ab_follower, "stage", None)
                if st is not None:
                    return st, False
        except Exception:
            pass

        # 2) 复用本 workflow 已经打开的 Stage12。
        if self.stage12_device is not None:
            return self.stage12_device, True

        # 3) 单独打开控制器。
        serial = str(getattr(self.cfg, "rule_ac_stage_device_id", "97101208"))
        if self._is_virtual_hardware_mode():
            self.log(f"[Step9-颜色中心][virtual] 创建虚拟 1/2 通道控制器：{serial}；不连接真实硬件")
            self.stage12_device = VirtualKinesisPiezoMotor(serial=serial, log_func=self.log)
        else:
            if Thorlabs is None:
                raise RuntimeError("当前环境无法导入 pylablib.devices.Thorlabs；请安装 pylablib 或切换 hardware_mode=virtual。")
            self.log(f"[Step9-颜色中心] 正在连接 1/2 通道控制器：{serial}")
            self.stage12_device = Thorlabs.KinesisPiezoMotor(serial)

        # 初始化 1/2 通道参数。
        for ch in (1, 2):
            self._setup_stage_channel_if_possible(
                self.stage12_device,
                channel=ch,
                max_voltage=int(getattr(self.cfg, "rule_ac_stage12_max_voltage", 50)),
                velocity=int(getattr(self.cfg, "rule_ac_stage12_velocity", 10)),
                acceleration=int(getattr(self.cfg, "rule_ac_stage12_acceleration", 10)),
            )

        self.log("[Step9-颜色中心] 1/2 通道控制器连接成功")
        return self.stage12_device, True

    def _setup_stage_channel_if_possible(
        self,
        stage_obj: Any,
        channel: int,
        max_voltage: int,
        velocity: int,
        acceleration: int,
    ):
        """
        兼容 Stage34 封装和 pylablib 原生 KinesisPiezoMotor 的通道参数设置。
        """
        if stage_obj is None:
            return

        channel = int(channel)
        max_voltage = int(max_voltage)
        velocity = int(velocity)
        acceleration = int(acceleration)

        try:
            if hasattr(stage_obj, "setup_channel"):
                stage_obj.setup_channel(
                    channel=channel,
                    max_voltage=max_voltage,
                    velocity=velocity,
                    acceleration=acceleration,
                )
                return
        except Exception as e:
            self.log(f"[Step9-颜色中心] setup_channel CH{channel} 失败，尝试 setup_drive：{e}")

        try:
            if hasattr(stage_obj, "setup_drive"):
                stage_obj.setup_drive(
                    max_voltage=max_voltage,
                    velocity=velocity,
                    acceleration=acceleration,
                    channel=channel,
                )
        except Exception as e:
            self.log(f"[Step9-颜色中心] setup_drive CH{channel} 失败：{e}")

    def _move_stage_channel(self, stage_obj: Any, channel: int, distance: float):
        """
        兼容 Stage34 封装和 pylablib 原生 KinesisPiezoMotor 的单通道相对运动。
        """
        if stage_obj is None:
            return

        channel = int(channel)
        distance = float(distance)

        # Stage34: move_by(channel=..., distance=...)
        try:
            stage_obj.move_by(channel=channel, distance=distance)
            return
        except TypeError:
            pass

        # pylablib KinesisPiezoMotor: move_by(distance=..., channel=...)
        try:
            stage_obj.move_by(distance=distance, channel=channel)
            return
        except TypeError:
            pass

        # 部分封装可能是位置参数形式。
        stage_obj.move_by(distance, channel=channel)

    def _stop_stage12_if_possible(self, stage_obj: Any):
        """
        尝试停止 1/2 通道。不同封装支持的 stop 接口可能不同，所以做兼容处理。
        """
        if stage_obj is None:
            return
        for method_name in ("stop_all", "stop"):
            try:
                m = getattr(stage_obj, method_name, None)
                if callable(m):
                    try:
                        m()
                    except TypeError:
                        try:
                            m(channel=1)
                            m(channel=2)
                        except Exception:
                            pass
                    return
            except Exception:
                pass

    def stop_step9_stage12(self):
        """
        外部按钮调用：请求停止 Step9，并尝试立即停止 1/2 通道运动。
        """
        self.step9_stop_requested = True
        self.log("[Step9-颜色中心] 已请求停止 1/2 通道运动")
        try:
            stage_obj = None
            if self.stage12_device is not None:
                stage_obj = self.stage12_device
            elif self.rule_ab_follower is not None:
                stage_obj = getattr(self.rule_ab_follower, "stage", None)
            self._stop_stage12_if_possible(stage_obj)
            self.log("[Step9-颜色中心] 已尝试发送 1/2 通道 stop 命令")
        except Exception as e:
            self.log(f"[Step9-颜色中心] 停止 1/2 通道失败：{e}")

    def _step9_direction_from_error(self, dx: float, dy: float, tol: float) -> int:
        """
        根据“当前检测区域中心 -> 目标点”的误差选择 1/2 通道运动命令。

        坐标约定：
            图像 x 正方向 = 向右；
            图像 y 正方向 = 向下。

        dx = target_x - current_x
        dy = target_y - current_y

        你的 1/2 通道真实屏幕运动方向：
            CH1 -  = 左下
            CH1 +  = 右上
            CH2 -  = 右下
            CH2 +  = 左上

        因此：
            需要向右：CH1 + 与 CH2 - 叠加
            需要向左：CH1 - 与 CH2 + 叠加
            需要向上：CH1 + 与 CH2 + 叠加
            需要向下：CH1 - 与 CH2 - 叠加

        action_code 定义：
             1  = 右上，只动 CH1+
            -1  = 左下，只动 CH1-
             2  = 左上，只动 CH2+
            -2  = 右下，只动 CH2-
            10  = 右，CH1+ + CH2-
           -10  = 左，CH1- + CH2+
            20  = 上，CH1+ + CH2+
           -20  = 下，CH1- + CH2-
             0  = 已在容差内，不需要运动
        """
        tol = abs(float(tol))

        sx = 0
        sy = 0
        if abs(float(dx)) > tol:
            sx = 1 if float(dx) > 0 else -1
        if abs(float(dy)) > tol:
            # 图像 y 轴向下为正：dy > 0 表示需要向下，dy < 0 表示需要向上。
            sy = 1 if float(dy) > 0 else -1

        if sx == 0 and sy == 0:
            return 0

        # 纯水平/纯竖直方向：由两个斜轴叠加得到。
        if sx == 1 and sy == 0:
            return 10      # 右 = CH1+ + CH2-
        if sx == -1 and sy == 0:
            return -10     # 左 = CH1- + CH2+
        if sx == 0 and sy == -1:
            return 20      # 上 = CH1+ + CH2+
        if sx == 0 and sy == 1:
            return -20     # 下 = CH1- + CH2-

        # 对角方向：直接用单个斜轴即可。
        if sx == 1 and sy == -1:
            return 1       # 右上 = CH1+
        if sx == -1 and sy == 1:
            return -1      # 左下 = CH1-
        if sx == -1 and sy == -1:
            return 2       # 左上 = CH2+
        return -2          # 右下 = CH2-

    def _execute_stage12_diagonal_action(self, action_code: int) -> Dict[str, Any]:
        """
        按真实 1/2 通道屏幕运动方向执行 Step9 运动。

        你的硬件观察结果：
            1- = 左下，1+ = 右上；
            2- = 右下，2+ = 左上。

        所以这里不再把 1/2 通道当作普通水平/垂直轴，也不再使用旧版
        “CH1/CH2 同号就是右上/左下”的错误假设，而是按两个斜轴叠加：
            向右 = CH1+ + CH2-
            向左 = CH1- + CH2+
            向上 = CH1+ + CH2+
            向下 = CH1- + CH2-
        """
        step = abs(float(getattr(self.cfg, "rule_ac_stage_step_size", 20)))

        # action_name 表示“希望检测区域中心在图像中移动的方向”。
        # ch1_distance/ch2_distance 是实际发送给 1/2 通道的相对步数。
        # 注意：这里的 action_name 表示“希望检测区域中心在图像中移动的方向”。
        # 1/2 轴实际推动的是背景/样品台，图像中的检测区域中心通常会沿相反方向变化。
        # 旧版把 stage 运动方向当成了图像中心运动方向，导致中心离目标越来越远；
        # 因此这里对 CH1/CH2 命令做反向映射，使“图像检测中心”按 action_name 靠近目标。
        mapping = {
            1:   (-step,  0.0,   "右上"),
            -1:  (step,   0.0,   "左下"),
            2:   (0.0,  -step,  "左上"),
            -2:  (0.0,   step,  "右下"),
            10:  (-step, step,   "右"),
            -10: (step, -step,   "左"),
            20:  (-step, -step,  "上"),
            -20: (step,  step,   "下"),
            0:   (0.0,   0.0,   "不动"),
        }

        action_code = int(action_code)
        if action_code not in mapping:
            return {"moved": False, "action_code": action_code, "reason": "unknown_action"}

        d1, d2, name = mapping[action_code]
        if action_code == 0 or (abs(d1) < 1e-12 and abs(d2) < 1e-12):
            return {
                "moved": False,
                "dry_run": False,
                "action_code": action_code,
                "action_name": name,
                "ch1_distance": d1,
                "ch2_distance": d2,
                "reason": "no_motion_needed",
            }

        if not bool(getattr(self.cfg, "rule_ac_enable_stage", False)):
            self.log(
                f"[Step9-颜色中心] DRY-RUN：希望中心向{name}移动，"
                f"应发送 CH1={d1}, CH2={d2}；但 RuleAC真动Stage12 未勾选"
            )
            return {
                "moved": False,
                "dry_run": True,
                "action_code": action_code,
                "action_name": name,
                "ch1_distance": d1,
                "ch2_distance": d2,
            }

        stage_obj, _owned = self._get_stage12_device()

        # 每次运动前同步 1/2 通道速度/加速度，运行中改 GUI 下一轮生效。
        for ch in (1, 2):
            self._setup_stage_channel_if_possible(
                stage_obj,
                channel=ch,
                max_voltage=int(getattr(self.cfg, "rule_ac_stage12_max_voltage", 50)),
                velocity=int(getattr(self.cfg, "rule_ac_stage12_velocity", 10)),
                acceleration=int(getattr(self.cfg, "rule_ac_stage12_acceleration", 10)),
            )

        self.log(
            f"[Step9-颜色中心] 希望中心向{name}移动："
            f"CH1={d1}, CH2={d2} "
            f"[已按图像中心运动方向反向映射：让检测中心靠近目标]"
        )

        if abs(d1) > 1e-12:
            self._move_stage_channel(stage_obj, channel=1, distance=d1)
        if abs(d2) > 1e-12:
            self._move_stage_channel(stage_obj, channel=2, distance=d2)

        return {
            "moved": True,
            "dry_run": False,
            "action_code": action_code,
            "action_name": name,
            "ch1_distance": d1,
            "ch2_distance": d2,
        }


    def _load_step9_c_mask_from_config(self, image_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Step9 加载 C mask。

        优先从 GUI 的“C标定文件夹”读取 static_c_mask.png。
        这个 mask 是“单独分割C”或“选择C文件夹”得到的固定 C 区域。
        """
        # 完整测量时 Step9 也必须使用 full_calibration_static_c，
        # 不能再从 RuleAB/RuleAC 单独测试区域残留的 C 文件夹读取。
        strict_c_dir = self._force_cfg_to_strict_full_calibration_c(reason="_load_step9_c_mask_from_config")
        c_dir_text = str(strict_c_dir or getattr(self.cfg, "rule_ab_static_c_map_dir", "")).strip()
        c_dir = None
        if c_dir_text:
            p0 = Path(c_dir_text.strip('"').strip("'"))
            candidates = [p0, p0 / "static_c_reference"]
            for c in candidates:
                if c.exists() and c.is_dir() and (c / "static_c_mask.png").exists():
                    c_dir = c
                    break

        if c_dir is None:
            raise RuntimeError(
                "Step9 需要先加载 C mask：请先点击“单独分割C”生成 C 标定文件夹，"
                "或点击“选择C文件夹”选择包含 static_c_mask.png 的文件夹。"
            )

        mask_path = c_dir / "static_c_mask.png"
        mask_path_str = str(mask_path.resolve())

        if self.step9_c_mask_cache is not None and self.step9_c_mask_path == mask_path_str:
            c_mask = self.step9_c_mask_cache.copy()
        else:
            m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if m is None:
                raise RuntimeError(f"Step9 加载 C mask 失败：{mask_path}")
            c_mask = (m > 0)
            if int(np.count_nonzero(c_mask)) <= 0:
                raise RuntimeError(f"Step9 加载到的 C mask 为空：{mask_path}")
            self.step9_c_mask_cache = c_mask.copy()
            self.step9_c_mask_path = mask_path_str
            self.log(f"[Step9-颜色中心] 已加载 C mask：{mask_path}，area={int(np.count_nonzero(c_mask))} px")

        if image_shape is not None:
            h, w = int(image_shape[0]), int(image_shape[1])
            if c_mask.shape[:2] != (h, w):
                self.log(
                    f"[Step9-颜色中心] C mask 尺寸 {c_mask.shape[:2]} 与当前截图 {(h, w)} 不一致，"
                    "将按最近邻 resize。建议确认 C 文件夹和当前截图区域一致。"
                )
                c_mask = cv2.resize(c_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

        return c_mask.astype(bool)

    def _get_step9_sam2_predictor(self):
        """
        Step9 专用 SAM2 ImagePredictor。

        新方案下 Step9 不再主动调用 SAM2；Step9 只使用颜色区域中心对齐。
        保留该函数只是为了兼容旧按钮/旧调用，一旦误调用会明确报错，避免完整测量又走回逐帧 SAM2。

        单独测试 Step9 时只需要分割/跟踪 B，因此这里单独缓存一个 predictor，
        避免每一帧重复加载模型。
        """
        if bool(getattr(self.cfg, "rule_ab_use_feature_tracker_after_sam2_init", True)):
            raise RuntimeError("当前已启用 SAM2首帧+颜色形状特征跟踪方案，Step9 不再使用 SAM2 分割；请使用 Step9 目标/颜色标定。")

        if self.step9_sam2_predictor is not None:
            return self.step9_sam2_predictor

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception as e:
            raise ImportError(
                "Step9 需要调用 SAM2 分割 B，但当前环境无法导入 sam2。"
                "请确认 sam2-main 已安装，并且可导入 build_sam2 / SAM2ImagePredictor。"
            ) from e

        try:
            tmp_cfg = RuleABRuntimeConfig()
        except Exception:
            tmp_cfg = None

        sam2_cfg = str(getattr(tmp_cfg, "sam2_cfg", "configs/sam2.1/sam2.1_hiera_t.yaml"))
        sam2_checkpoint = str(
            getattr(
                tmp_cfg,
                "sam2_checkpoint",
                str(SAM2_REPO_ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt"),
            )
        )
        sam2_device = str(getattr(tmp_cfg, "sam2_device", "cuda"))

        self.log(
            f"[Step9-颜色中心] 正在加载 SAM2 分割 B：cfg={sam2_cfg}, "
            f"checkpoint={sam2_checkpoint}, device={sam2_device}"
        )
        model = build_sam2(sam2_cfg, sam2_checkpoint, device=sam2_device)
        predictor = SAM2ImagePredictor(model)
        self.step9_sam2_predictor = predictor
        self.step9_sam2_model_info = {
            "cfg": sam2_cfg,
            "checkpoint": sam2_checkpoint,
            "device": sam2_device,
        }
        return predictor

    def _select_step9_b_points_interactively(
        self,
        image_rgb: np.ndarray,
        window_name: str = "Step9 select B: left positive, right negative",
        scale: float = 0.85,
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Step9 单独测试前，用 SAM2 分割 B 的点提示。

        左键：B 正点，可以多个；
        右键：B 负点，可以多个；
        Enter/N：确认；
        R：清空；
        ESC/Q：取消。
        """
        if scale <= 0:
            scale = 1.0

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        h, w = image_bgr.shape[:2]
        show_w = max(1, int(w * scale))
        show_h = max(1, int(h * scale))
        display = cv2.resize(image_bgr, (show_w, show_h), interpolation=cv2.INTER_AREA)

        pos: List[Tuple[int, int]] = []
        neg: List[Tuple[int, int]] = []

        def redraw() -> np.ndarray:
            canvas = display.copy()
            lines = [
                "Step9: select B for SAM2 segmentation",
                "Left click: B positive point | Right click: B negative point",
                "N/Enter: confirm | R: reset | ESC/Q: cancel",
                f"B positive={len(pos)}, negative={len(neg)}",
            ]
            for i, s in enumerate(lines):
                cv2.putText(canvas, s, (18, 28 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA)

            for j, (x, y) in enumerate(pos):
                cv2.circle(canvas, (x, y), 6, (0, 255, 0), -1)
                cv2.putText(canvas, f"B+{j + 1}", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2, cv2.LINE_AA)
            for j, (x, y) in enumerate(neg):
                cv2.circle(canvas, (x, y), 7, (0, 0, 255), 2)
                cv2.line(canvas, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 255), 2)
                cv2.line(canvas, (x - 5, y + 5), (x + 5, y - 5), (0, 0, 255), 2)
                cv2.putText(canvas, f"B-{j + 1}", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 255), 2, cv2.LINE_AA)
            return canvas

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                pos.append((int(x), int(y)))
            elif event == cv2.EVENT_RBUTTONDOWN:
                neg.append((int(x), int(y)))

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, show_w, show_h)
        cv2.setMouseCallback(window_name, on_mouse)

        try:
            while True:
                cv2.imshow(window_name, redraw())
                key = cv2.waitKey(30) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    raise RuntimeError("用户取消了 Step9 的 B 分割。")
                if key in (ord("r"), ord("R")):
                    pos.clear()
                    neg.clear()
                if key in (13, 10, ord("n"), ord("N")):
                    if len(pos) <= 0:
                        self.log("[Step9-颜色中心] B 分割至少需要 1 个正点。")
                        continue
                    break
        finally:
            try:
                cv2.destroyWindow(window_name)
            except Exception:
                pass

        def to_original(points: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
            return [(float(x) / scale, float(y) / scale) for x, y in points]

        return to_original(pos), to_original(neg)

    @staticmethod
    def _mask_bbox_xyxy(mask: np.ndarray, margin: float = 20.0) -> Optional[Tuple[float, float, float, float]]:
        """由 mask 计算 xyxy bbox，并添加 margin。"""
        if mask is None:
            return None
        m = np.asarray(mask).astype(bool)
        if not np.any(m):
            return None
        ys, xs = np.where(m)
        h, w = m.shape[:2]
        x1 = max(0.0, float(xs.min()) - float(margin))
        y1 = max(0.0, float(ys.min()) - float(margin))
        x2 = min(float(w - 1), float(xs.max()) + float(margin))
        y2 = min(float(h - 1), float(ys.max()) + float(margin))
        return (x1, y1, x2, y2)

    def _predict_step9_b_mask_by_sam2(
        self,
        image_rgb: np.ndarray,
        positive_points: List[Tuple[float, float]],
        negative_points: Optional[List[Tuple[float, float]]] = None,
        box_xyxy: Optional[Tuple[float, float, float, float]] = None,
    ) -> np.ndarray:
        """
        调用 SAM2 ImagePredictor 分割 B。

        初始帧使用人工选择的 B 正负点；
        后续帧优先使用上一帧 B 中心点 + 上一帧 bbox 更新，保证每帧都调用 SAM2。
        """
        import torch

        negative_points = negative_points or []
        predictor = self._get_step9_sam2_predictor()
        predictor.set_image(image_rgb)

        coords: List[List[float]] = []
        labels: List[int] = []
        for x, y in positive_points:
            coords.append([float(x), float(y)])
            labels.append(1)
        for x, y in negative_points:
            coords.append([float(x), float(y)])
            labels.append(0)

        if not coords or 1 not in labels:
            raise ValueError("Step9 B 分割至少需要一个正点。")

        point_coords = np.array(coords, dtype=np.float32)
        point_labels = np.array(labels, dtype=np.int32)
        box = None
        if box_xyxy is not None:
            box = np.array([float(v) for v in box_xyxy], dtype=np.float32)

        with torch.inference_mode():
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=True,
            )

        if masks is None or len(masks) == 0:
            raise RuntimeError("SAM2 没有返回 B mask。")

        scores_arr = np.asarray(scores).reshape(-1)
        best_idx = int(np.argmax(scores_arr))
        b_mask = masks[best_idx].astype(bool)
        area = int(np.count_nonzero(b_mask))
        if area <= 0:
            raise RuntimeError("SAM2 返回的 B mask 为空。")

        center = self._mask_center_xy(b_mask)
        bbox = self._mask_bbox_xyxy(b_mask, margin=30.0)
        self.step9_b_last_mask = b_mask.copy()
        self.step9_b_last_center = center
        self.step9_b_last_box = bbox

        self.log(
            f"[Step9-颜色中心] SAM2 B 分割完成：score={float(scores_arr[best_idx]):.6f}, "
            f"area={area} px, center={center}, bbox={bbox}"
        )
        return b_mask

    def prepare_step9_b_segmentation(self):
        """
        单独测试 Step9 前准备 B 的 SAM2 分割。

        这一步会：
            1. 截取当前固定区域；
            2. 让用户用正/负点标记 B；
            3. 调用 SAM2 分割 B；
            4. 缓存 B 的初始 mask/center/bbox，后续每帧继续用 SAM2 更新。
        """
        output_dir = self._get_step9_output_dir("b_initial")
        output_dir.mkdir(parents=True, exist_ok=True)
        image_rgb = self._capture_current_rule_ab_frame(output_dir=output_dir)

        pos, neg = self._select_step9_b_points_interactively(
            image_rgb=image_rgb,
            window_name="Step9 B segmentation: left positive, right negative",
            scale=float(getattr(self.cfg, "rule_ab_confirm_window_scale", 0.85)),
        )
        self.step9_b_positive_points = list(pos)
        self.step9_b_negative_points = list(neg)

        b_mask = self._predict_step9_b_mask_by_sam2(
            image_rgb=image_rgb,
            positive_points=self.step9_b_positive_points,
            negative_points=self.step9_b_negative_points,
            box_xyxy=None,
        )

        # 保存初始 B 分割检查图
        try:
            out_dir = self._get_step9_output_dir("b_initial")
            out_dir.mkdir(parents=True, exist_ok=True)
            canvas = cv2.cvtColor(np.asarray(image_rgb).copy(), cv2.COLOR_RGB2BGR)
            overlay = canvas.copy()
            overlay[b_mask.astype(bool)] = (0, 255, 0)
            canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)
            if self.step9_b_last_center is not None:
                cx, cy = int(round(self.step9_b_last_center[0])), int(round(self.step9_b_last_center[1]))
                cv2.circle(canvas, (cx, cy), 6, (0, 0, 255), -1)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            cv2.imwrite(str(out_dir / f"step9_b_initial_{ts}.png"), canvas)
        except Exception as e:
            self.log(f"[Step9-颜色中心] 保存 B 初始分割图失败：{e}")

        self.log(
            f"[Step9-颜色中心] B 初始分割已准备：positive={len(pos)}, negative={len(neg)}, "
            f"center={self.step9_b_last_center}"
        )

    def _select_step9_color_interactively(
        self,
        image_rgb: np.ndarray,
        default_mode: str = "include",
        window_name: str = "Step9 select color: I=include, E=exclude",
        scale: float = 0.85,
    ) -> Dict[str, Any]:
        """
        Step9 颜色检测区域选择。

        两种模式：
            include：直接点击“需要检测区域”的颜色，只保留接近该颜色的区域；
            exclude：点击“其他/背景颜色”，凡是不接近该颜色的区域都视作需要检测区域。

        操作：
            左键：选择颜色点，可以重复点击，使用最后一次点击的颜色；
            I：include 模式；
            E：exclude 模式；
            Enter/N：确认；
            R：清空重选；
            ESC/Q：取消。
        """
        if scale <= 0:
            scale = 1.0
        mode = str(default_mode or "include").lower().strip()
        if mode not in ("include", "exclude"):
            mode = "include"

        image_bgr = cv2.cvtColor(np.asarray(image_rgb), cv2.COLOR_RGB2BGR)
        image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        h, w = image_bgr.shape[:2]
        show_w = max(1, int(w * scale))
        show_h = max(1, int(h * scale))
        display = cv2.resize(image_bgr, (show_w, show_h), interpolation=cv2.INTER_AREA)

        selected: List[Tuple[int, int]] = []
        selected_hsv: List[Tuple[int, int, int]] = []

        def _preview_mask(sample_hsv: Optional[Tuple[int, int, int]], mode_text: str) -> Optional[np.ndarray]:
            if sample_hsv is None:
                return None
            m = self._build_step9_color_mask_from_hsv(
                image_rgb=image_rgb,
                sample_hsv=sample_hsv,
                mode=mode_text,
                h_tol=int(getattr(self.cfg, "rule_ac_color_h_tol", 12)),
                s_tol=int(getattr(self.cfg, "rule_ac_color_s_tol", 70)),
                v_tol=int(getattr(self.cfg, "rule_ac_color_v_tol", 70)),
                min_area_px=int(getattr(self.cfg, "rule_ac_color_min_area_px", 50)),
                morph_kernel=int(getattr(self.cfg, "rule_ac_color_morph_kernel", 5)),
                prefer_center=None,
            )
            return m

        def redraw() -> np.ndarray:
            canvas = display.copy()
            sample = selected_hsv[-1] if selected_hsv else None
            if sample is not None:
                try:
                    pm = _preview_mask(sample, mode)
                    if pm is not None:
                        pm_show = cv2.resize(pm.astype(np.uint8), (show_w, show_h), interpolation=cv2.INTER_NEAREST).astype(bool)
                        overlay = canvas.copy()
                        overlay[pm_show] = (0, 255, 255)
                        canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)
                except Exception:
                    pass

            lines = [
                "Step9 color detection region",
                "Left click: sample color | I: include selected color | E: exclude selected color",
                "Enter/N: confirm | R: reset | ESC/Q: cancel",
                f"mode={mode}  HSV={sample if sample is not None else None}",
                "include: detect pixels close to clicked color; exclude: detect pixels NOT close to clicked color",
            ]
            for i, text in enumerate(lines):
                cv2.putText(canvas, text, (18, 28 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 255, 255), 2, cv2.LINE_AA)
            if selected:
                x, y = selected[-1]
                cv2.drawMarker(canvas, (x, y), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
                cv2.putText(canvas, "sample", (x + 10, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 255), 2, cv2.LINE_AA)
            return canvas

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                ox = int(round(float(x) / scale))
                oy = int(round(float(y) / scale))
                ox = max(0, min(w - 1, ox))
                oy = max(0, min(h - 1, oy))
                hsv = tuple(int(v) for v in image_hsv[oy, ox].tolist())
                selected.clear()
                selected.append((int(x), int(y)))
                selected_hsv.clear()
                selected_hsv.append(hsv)  # type: ignore[arg-type]

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, show_w, show_h)
        cv2.setMouseCallback(window_name, on_mouse)

        try:
            while True:
                cv2.imshow(window_name, redraw())
                key = cv2.waitKey(30) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    raise RuntimeError("用户取消了 Step9 颜色选择。")
                if key in (ord("r"), ord("R")):
                    selected.clear()
                    selected_hsv.clear()
                if key in (ord("i"), ord("I")):
                    mode = "include"
                if key in (ord("e"), ord("E")):
                    mode = "exclude"
                if key in (13, 10, ord("n"), ord("N")):
                    if not selected_hsv:
                        self.log("[Step9-颜色中心] 请先左键点击一个颜色采样点。")
                        continue
                    break
        finally:
            try:
                cv2.destroyWindow(window_name)
            except Exception:
                pass

        hsv = selected_hsv[-1]
        return {
            "mode": mode,
            "h": int(hsv[0]),
            "s": int(hsv[1]),
            "v": int(hsv[2]),
            "hsv": (int(hsv[0]), int(hsv[1]), int(hsv[2])),
        }

    @staticmethod
    def _hue_distance_180(h_arr: np.ndarray, h0: int) -> np.ndarray:
        """OpenCV HSV hue 范围为 0~179，计算环形距离。"""
        d = np.abs(h_arr.astype(np.int16) - int(h0))
        return np.minimum(d, 180 - d)

    def _build_step9_color_mask_from_hsv(
        self,
        image_rgb: np.ndarray,
        sample_hsv: Tuple[int, int, int],
        mode: str,
        h_tol: int,
        s_tol: int,
        v_tol: int,
        min_area_px: int,
        morph_kernel: int,
        prefer_center: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """
        根据采样 HSV 生成 Step9 检测区域 mask。

        include：mask = 接近采样颜色的像素；
        exclude：mask = 不接近采样颜色的像素。
        """
        img = np.asarray(image_rgb)
        if img.ndim == 2:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        h0, s0, v0 = [int(x) for x in sample_hsv]
        h_tol = max(0, int(h_tol))
        s_tol = max(0, int(s_tol))
        v_tol = max(0, int(v_tol))

        dh = self._hue_distance_180(hsv[:, :, 0], h0)
        ds = np.abs(hsv[:, :, 1].astype(np.int16) - int(s0))
        dv = np.abs(hsv[:, :, 2].astype(np.int16) - int(v0))
        close = (dh <= h_tol) & (ds <= s_tol) & (dv <= v_tol)

        mode = str(mode or "include").lower().strip()
        if mode == "exclude":
            mask = ~close
        else:
            mask = close

        mask = mask.astype(np.uint8)
        k = int(morph_kernel)
        if k > 1:
            if k % 2 == 0:
                k += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 连通域筛选：首帧取最大区域；后续优先取离上一帧中心最近的区域。
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        if num <= 1:
            return np.zeros(mask.shape, dtype=bool)

        min_area = max(1, int(min_area_px))
        candidates = []
        for lab in range(1, num):
            area = int(stats[lab, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            cx, cy = float(centroids[lab][0]), float(centroids[lab][1])
            if prefer_center is not None:
                dist = math.hypot(cx - float(prefer_center[0]), cy - float(prefer_center[1]))
                # 面积仍参与评分，避免选到极小噪点。
                score = -dist + 0.002 * area
            else:
                score = float(area)
            candidates.append((score, lab, area, cx, cy))

        if not candidates:
            return np.zeros(mask.shape, dtype=bool)

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_lab = candidates[0][1]
        return (labels == best_lab)

    def prepare_step9_color_region_detection(self) -> Dict[str, Any]:
        """
        单独测试 Step9 前选择颜色检测区域。

        不再选择 B/C mask 正点负点；改为选择一个颜色采样点，并指定 include/exclude 模式。
        """
        # 允许“停止测量/关闭全部设备”之后立即重新选择 Step9 颜色区域。
        # 旧的 stop_requested=True 只表示上一轮已停止，不应阻止新的人工标定截图。
        self.stop_requested = False
        self.step9_stop_requested = False

        output_dir = self._get_step9_output_dir("color_initial")
        output_dir.mkdir(parents=True, exist_ok=True)
        image_rgb = self._capture_current_rule_ab_frame(output_dir=output_dir)

        default_mode = str(getattr(self.cfg, "rule_ac_color_mode", "include"))
        color_info = self._select_step9_color_interactively(
            image_rgb=image_rgb,
            default_mode=default_mode,
            window_name="Step9 color region: click sample, I include, E exclude",
            scale=float(getattr(self.cfg, "rule_ab_confirm_window_scale", 0.85)),
        )
        hsv = color_info["hsv"]
        mode = str(color_info["mode"])

        self.step9_color_mode = mode
        self.step9_color_hsv = (int(hsv[0]), int(hsv[1]), int(hsv[2]))
        self.cfg.rule_ac_color_mode = mode
        self.cfg.rule_ac_color_h = int(hsv[0])
        self.cfg.rule_ac_color_s = int(hsv[1])
        self.cfg.rule_ac_color_v = int(hsv[2])

        mask = self._build_step9_color_mask_from_hsv(
            image_rgb=image_rgb,
            sample_hsv=self.step9_color_hsv,
            mode=mode,
            h_tol=int(getattr(self.cfg, "rule_ac_color_h_tol", 12)),
            s_tol=int(getattr(self.cfg, "rule_ac_color_s_tol", 70)),
            v_tol=int(getattr(self.cfg, "rule_ac_color_v_tol", 70)),
            min_area_px=int(getattr(self.cfg, "rule_ac_color_min_area_px", 50)),
            morph_kernel=int(getattr(self.cfg, "rule_ac_color_morph_kernel", 5)),
            prefer_center=None,
        )
        center = self._mask_center_xy(mask)
        self.step9_color_last_mask = mask.copy()
        self.step9_color_last_center = center

        try:
            canvas = cv2.cvtColor(np.asarray(image_rgb).copy(), cv2.COLOR_RGB2BGR)
            overlay = canvas.copy()
            overlay[mask.astype(bool)] = (0, 255, 255)
            canvas = cv2.addWeighted(overlay, 0.40, canvas, 0.60, 0)
            if center is not None:
                cx, cy = int(round(center[0])), int(round(center[1]))
                cv2.circle(canvas, (cx, cy), 7, (0, 0, 255), -1)
                cv2.putText(canvas, "color center", (cx + 10, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            cv2.imwrite(str(output_dir / f"step9_color_initial_{ts}.png"), canvas)
            cv2.imwrite(str(output_dir / f"step9_color_initial_{ts}_mask.png"), mask.astype(np.uint8) * 255)
        except Exception as e:
            self.log(f"[Step9-颜色中心] 保存颜色初始检测图失败：{e}")

        self.log(
            f"[Step9-颜色中心] 颜色检测已准备：mode={mode}, HSV={self.step9_color_hsv}, "
            f"area={int(np.count_nonzero(mask))} px, center={center}"
        )
        return {
            "mode": mode,
            "hsv": self.step9_color_hsv,
            "center": center,
            "area_px": int(np.count_nonzero(mask)),
        }

    def reset_step9_tracking_state(self, clear_b_points: bool = False):
        """
        重置 Step9 的 B/C 缓存。

        clear_b_points=True 时，下次单独测试会重新要求人工选择 B 点。
        """
        if clear_b_points:
            self.step9_b_positive_points = []
            self.step9_b_negative_points = []
        self.step9_b_last_mask = None
        self.step9_b_last_center = None
        self.step9_b_last_box = None
        self.step9_c_mask_cache = None
        self.step9_c_mask_path = None
        self.step9_dynamic_c_last_mask = None
        self.step9_dynamic_c_last_raw_mask = None
        self.step9_dynamic_c_last_center = None
        self.step9_dynamic_c_last_box = None
        self.step9_dynamic_c_last_geometry = None

    def _get_step9_global_c_prompt_points(self) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """读取完整标定包里的 C 正/负点，供 Step9 每帧动态 C 分割使用。"""
        state = self._get_loaded_calibration_state()
        if state is None:
            try:
                state = self.load_calibration_state(required=False)
            except Exception:
                state = None
        pos: List[Tuple[float, float]] = []
        neg: List[Tuple[float, float]] = []
        if state is not None:
            pos = self._calib_points_to_tuples(getattr(state, "global_c_positive_points", []))
            neg = self._calib_points_to_tuples(getattr(state, "global_c_negative_points", []))
        return pos, neg

    def _predict_step9_dynamic_c_mask_by_sam2(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        """
        Step9 每帧动态分割 C，并生成用于 B∩C 计算的外接矩形/四边形 mask。

        重要分工：
            - Step7 的运动路线仍然使用 full_calibration_static_c/static_c_edge_route.json；
            - 这里得到的 C_dynamic 只用于 Step9/Step8 的 颜色检测区域 重合区域中心计算。
        """
        import torch

        pos, neg = self._get_step9_global_c_prompt_points()
        predictor = self._get_step9_sam2_predictor()
        predictor.set_image(image_rgb)

        coords: List[List[float]] = []
        labels: List[int] = []
        prompt_mode = "calibration_c_points"

        # 后续帧优先用上一帧 颜色检测 的中心点 + bbox 更新，避免平台运动后原始 C 正点坐标失效。
        box = None
        if self.step9_dynamic_c_last_center is not None and self.step9_dynamic_c_last_box is not None:
            cx, cy = self.step9_dynamic_c_last_center
            coords.append([float(cx), float(cy)])
            labels.append(1)
            box = np.array([float(v) for v in self.step9_dynamic_c_last_box], dtype=np.float32)
            prompt_mode = "last_dynamic_c_center_and_bbox"
        else:
            for x, y in pos:
                coords.append([float(x), float(y)])
                labels.append(1)
            for x, y in neg:
                coords.append([float(x), float(y)])
                labels.append(0)

        if not coords or 1 not in labels:
            # 没有 C 正点时，退回 static C，只作为兜底，避免 Step9 直接崩溃。
            static_c = self._load_step9_c_mask_from_config(image_shape=image_rgb.shape[:2])
            center = self._mask_center_xy(static_c)
            bbox = self._mask_bbox_xyxy(static_c, margin=30.0)
            self.step9_dynamic_c_last_mask = static_c.copy()
            self.step9_dynamic_c_last_raw_mask = static_c.copy()
            self.step9_dynamic_c_last_center = center
            self.step9_dynamic_c_last_box = bbox
            self.step9_dynamic_c_last_geometry = {"source": "fallback_static_c_no_prompt"}
            return {
                "ok": True,
                "raw_c_mask": static_c,
                "c_mask_for_overlap": static_c,
                "center": center,
                "bbox": bbox,
                "geometry": {"source": "fallback_static_c_no_prompt"},
                "prompt_mode": "fallback_static_c_no_prompt",
                "reason": "no_global_c_prompt_points",
            }

        point_coords = np.array(coords, dtype=np.float32)
        point_labels = np.array(labels, dtype=np.int32)

        with torch.inference_mode():
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=True,
            )

        if masks is None or len(masks) == 0:
            raise RuntimeError("SAM2 没有返回 Step9 颜色检测 mask。")

        scores_arr = np.asarray(scores).reshape(-1)
        # 初帧用完整 C 正/负点清理；后续帧只有上一帧中心点时，先按 SAM2 score 选，再取最大主体连通域。
        clean_debug: Dict[str, Any]
        if prompt_mode == "calibration_c_points":
            raw_clean, best_idx, clean_debug = MeasurementWorkflow._select_and_clean_c_mask_from_sam2_candidates(
                masks,
                scores,
                positive_points=pos,
                negative_points=neg,
            )
        else:
            best_idx = int(np.argmax(scores_arr))
            raw_best = masks[best_idx].astype(bool)
            raw_clean, clean_debug = MeasurementWorkflow._clean_c_mask_by_prompt_connected_component(
                raw_best,
                positive_points=[self.step9_dynamic_c_last_center] if self.step9_dynamic_c_last_center is not None else [],
                negative_points=[],
                point_radius_px=8,
                min_component_area_px=8,
            )

        raw_c = np.asarray(raw_clean).astype(bool)
        if int(np.count_nonzero(raw_c)) <= 0:
            raise RuntimeError("Step9 颜色检测 清理后为空。")

        # 生成用于重合区域的几何 C：圆形 -> 水平外接矩形；非圆形 -> 四边形拟合。
        circle_like, circle_debug = MeasurementWorkflow._judge_c_mask_circle_like(raw_c)
        geometry: Dict[str, Any] = {
            "prompt_mode": prompt_mode,
            "sam2_selected_idx": int(best_idx),
            "sam2_score": float(scores_arr[int(best_idx)]) if scores_arr.size > int(best_idx) else None,
            "clean_debug": self._json_safe(clean_debug),
            "circle_debug": self._json_safe(circle_debug),
        }
        if circle_like:
            geom_points, c_for_overlap, bbox_debug = MeasurementWorkflow._build_external_bbox_quad_from_mask(raw_c)
            geom_method = "circle_like_external_axis_aligned_bbox"
            geometry["bbox_debug"] = self._json_safe(bbox_debug)
        else:
            geom_points, c_for_overlap, geom_method = self._fit_quadrilateral_from_mask(raw_c)

        center = self._mask_center_xy(c_for_overlap)
        bbox_dyn = self._mask_bbox_xyxy(c_for_overlap, margin=35.0)
        geometry.update({
            "source": "dynamic_c_for_overlap_only",
            "method": geom_method,
            "circle_like": bool(circle_like),
            "raw_area_px": int(np.count_nonzero(raw_c)),
            "geometry_area_px": int(np.count_nonzero(c_for_overlap)),
            "geometry_points_xy": [[float(x), float(y)] for x, y in np.asarray(geom_points).reshape(-1, 2)],
        })

        self.step9_dynamic_c_last_raw_mask = raw_c.copy()
        self.step9_dynamic_c_last_mask = c_for_overlap.astype(bool).copy()
        self.step9_dynamic_c_last_center = center
        self.step9_dynamic_c_last_box = bbox_dyn
        self.step9_dynamic_c_last_geometry = geometry

        self.log(
            f"[Step9-颜色中心] 颜色检测 分割完成：mode={prompt_mode}, "
            f"score={geometry.get('sam2_score')}, raw_area={geometry['raw_area_px']} px, "
            f"geom_area={geometry['geometry_area_px']} px, method={geom_method}, center={center}"
        )

        return {
            "ok": True,
            "raw_c_mask": raw_c,
            "c_mask_for_overlap": c_for_overlap.astype(bool),
            "center": center,
            "bbox": bbox_dyn,
            "geometry": geometry,
            "prompt_mode": prompt_mode,
            "reason": "ok",
        }

    def _capture_step9_color_center_once(self) -> Dict[str, Any]:
        """
        Step9 当前帧检测：颜色检测区域中心对齐。

        当前版本不再计算 颜色检测区域，也不再每帧动态分割 C。
        Step9/Step8 的位置对齐只使用颜色检测区域中心；A/B mask 仅作为调试结果每帧保存，
        不参与 Step9 的运动方向和停止条件。
        """
        image_rgb = None
        a_mask = None
        b_mask = None
        ab_saved: Dict[str, Optional[str]] = {}

        # 优先复用 RuleAB follower 捕获同一帧，这样可以顺便保存 A/B 分割结果。
        # 如果 A/B 跟踪失败，不允许影响 Step9 颜色中心检测；此时退回普通截图。
        try:
            follower = self.ensure_rule_ab_follower()
            image_rgb, scene, _seg_retry_info = self._capture_and_build_scene_with_ab_retry(
                follower=follower,
                label="Step9颜色中心保存A/B",
                allow_fail=False,
            )
            try:
                self._postprocess_scene_ab_masks(
                    follower=follower,
                    scene=scene,
                    tag="step9_color_center_save_ab",
                )
            except Exception as e:
                self.log(f"[Step9-颜色中心] Step9 A/B mask 后处理失败，仅保存原始 scene mask：{e}")
            a_mask = self._extract_mask_from_scene_object(getattr(scene, "a", None))
            b_mask = self._extract_mask_from_scene_object(getattr(scene, "b", None))
        except Exception as e:
            self.log(f"[Step9-颜色中心] 当前帧 A/B 分割失败，本轮仍继续颜色中心检测：{e}")
            try:
                # 使用Step9输出目录作为临时目录
                step9_output_dir = self._get_step9_output_dir()
                image_rgb = self._capture_current_rule_ab_frame(output_dir=step9_output_dir)
            except Exception as ee:
                return {
                    "ok": False,
                    "reason": f"capture_failed:{ee}",
                    "image_rgb": None,
                    "color_mask": None,
                    "color_area_px": 0,
                    "a_mask": None,
                    "b_mask": None,
                    "ab_saved": ab_saved,
                }

        if image_rgb is None:
            return {
                "ok": False,
                "reason": "image_rgb_none",
                "image_rgb": None,
                "color_mask": None,
                "color_area_px": 0,
                "a_mask": a_mask,
                "b_mask": b_mask,
                "ab_saved": ab_saved,
            }

        # 颜色 HSV 必须已经通过“Step9目标/颜色”标定或手动填写。
        hsv_tuple = self.step9_color_hsv
        if hsv_tuple is None:
            h = int(getattr(self.cfg, "rule_ac_color_h", -1))
            s = int(getattr(self.cfg, "rule_ac_color_s", -1))
            v = int(getattr(self.cfg, "rule_ac_color_v", -1))
            if h >= 0 and s >= 0 and v >= 0:
                hsv_tuple = (h, s, v)
                self.step9_color_hsv = hsv_tuple
        if hsv_tuple is None:
            return {
                "ok": False,
                "reason": "step9_color_hsv_not_set",
                "image_rgb": image_rgb,
                "color_mask": None,
                "color_area_px": 0,
                "a_mask": a_mask,
                "b_mask": b_mask,
                "ab_saved": ab_saved,
            }

        try:
            color_mask = self._build_step9_color_mask_from_hsv(
                image_rgb=image_rgb,
                sample_hsv=hsv_tuple,
                mode=str(getattr(self.cfg, "rule_ac_color_mode", self.step9_color_mode or "include")),
                h_tol=int(getattr(self.cfg, "rule_ac_color_h_tol", 12)),
                s_tol=int(getattr(self.cfg, "rule_ac_color_s_tol", 70)),
                v_tol=int(getattr(self.cfg, "rule_ac_color_v_tol", 70)),
                min_area_px=int(getattr(self.cfg, "rule_ac_color_min_area_px", 50)),
                morph_kernel=int(getattr(self.cfg, "rule_ac_color_morph_kernel", 5)),
                prefer_center=self.step9_color_last_center,
            )
        except Exception as e:
            return {
                "ok": False,
                "reason": f"color_mask_failed:{e}",
                "image_rgb": image_rgb,
                "color_mask": None,
                "color_area_px": 0,
                "a_mask": a_mask,
                "b_mask": b_mask,
                "ab_saved": ab_saved,
            }

        center = self._mask_center_xy(color_mask)
        color_area = int(np.count_nonzero(color_mask))
        if center is None or color_area <= 0:
            return {
                "ok": False,
                "reason": "empty_color_region_mask",
                "image_rgb": image_rgb,
                "color_mask": color_mask,
                "color_area_px": color_area,
                "a_mask": a_mask,
                "b_mask": b_mask,
                "ab_saved": ab_saved,
            }

        self.step9_color_last_mask = color_mask.copy()
        self.step9_color_last_center = center
        return {
            "ok": True,
            "reason": "ok_color_region_center",
            "image_rgb": image_rgb,
            "color_mask": color_mask,
            "color_center": center,
            "color_area_px": color_area,
            "a_mask": a_mask,
            "b_mask": b_mask,
            "a_mask_area_px": int(np.count_nonzero(a_mask)) if a_mask is not None else "",
            "b_mask_area_px": int(np.count_nonzero(b_mask)) if b_mask is not None else "",
            "ab_saved": ab_saved,
        }

    def _save_step9_color_overlay(
        self,
        image_rgb: Optional[np.ndarray],
        color_mask: Optional[np.ndarray],
        center: Optional[Tuple[float, float]],
        target: Optional[Tuple[float, float]],
        cycle_idx: int,
        action_code: int,
        a_mask: Optional[np.ndarray] = None,
        b_mask: Optional[np.ndarray] = None,
        c_mask: Optional[np.ndarray] = None,
        c_raw_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Optional[str]]:
        """
        保存 Step9 颜色检测区域中心对齐的检查图和 A/B/C 分割调试结果。

        当前 Step9 使用颜色检测区域中心对齐；A/B/C mask 仅保存和叠加显示，
        方便检查每帧分割是否正确。这里的 c_mask 是完整测量锁定的 static C
        （也就是“3 标定全局C”后保存到本轮 full_calibration_static_c 的 static_c_mask.png），
        c_raw_mask 若存在则保存原始/动态 C 调试 mask。
        """
        saved = {
            "overlay_path": None,
            "frame_path": None,
            "color_mask_path": None,
            "a_mask_path": None,
            "b_mask_path": None,
            "c_mask_path": None,
            "c_raw_mask_path": None,
        }

        try:
            overlay_dir = self._get_step9_output_dir("overlays")
            frame_dir = self._get_step9_output_dir("frames")
            mask_dir = self._get_step9_output_dir("masks")
            ab_dir = self._get_step9_output_dir("ab_masks")

            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            stem = f"step9_cycle_{cycle_idx:04d}_{ts}_action_{action_code}"

            if image_rgb is None:
                return saved
            img = np.asarray(image_rgb).copy()
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            frame_path = frame_dir / f"{stem}_frame.png"
            cv2.imwrite(str(frame_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            saved["frame_path"] = str(frame_path)

            if color_mask is not None:
                p_color = mask_dir / f"{stem}_color_region_mask.png"
                cv2.imwrite(str(p_color), (np.asarray(color_mask).astype(np.uint8) * 255))
                saved["color_mask_path"] = str(p_color)

            if a_mask is not None:
                p_a = ab_dir / f"{stem}_a_mask.png"
                cv2.imwrite(str(p_a), (np.asarray(a_mask).astype(np.uint8) * 255))
                saved["a_mask_path"] = str(p_a)
            if b_mask is not None:
                p_b = ab_dir / f"{stem}_b_mask.png"
                cv2.imwrite(str(p_b), (np.asarray(b_mask).astype(np.uint8) * 255))
                saved["b_mask_path"] = str(p_b)
            if c_mask is not None:
                p_c = ab_dir / f"{stem}_c_mask.png"
                cv2.imwrite(str(p_c), (np.asarray(c_mask).astype(np.uint8) * 255))
                saved["c_mask_path"] = str(p_c)
            if c_raw_mask is not None:
                p_c_raw = ab_dir / f"{stem}_c_raw_mask.png"
                cv2.imwrite(str(p_c_raw), (np.asarray(c_raw_mask).astype(np.uint8) * 255))
                saved["c_raw_mask_path"] = str(p_c_raw)

            canvas = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # 调试叠加：C 蓝色，A 黄色，B 绿色，颜色检测区域青色。A/B/C 只用于观察，不参与 Step9 对齐。
            if c_mask is not None:
                cm_static = np.asarray(c_mask).astype(bool)
                overlay = canvas.copy()
                overlay[cm_static] = (255, 0, 0)
                canvas = cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0)
            if c_raw_mask is not None:
                cm_raw = np.asarray(c_raw_mask).astype(bool)
                overlay = canvas.copy()
                overlay[cm_raw] = (255, 128, 0)
                canvas = cv2.addWeighted(overlay, 0.12, canvas, 0.88, 0)
            if a_mask is not None:
                am = np.asarray(a_mask).astype(bool)
                overlay = canvas.copy()
                overlay[am] = (0, 255, 255)
                canvas = cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0)
            if b_mask is not None:
                bm = np.asarray(b_mask).astype(bool)
                overlay = canvas.copy()
                overlay[bm] = (0, 255, 0)
                canvas = cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0)
            if color_mask is not None:
                cm = np.asarray(color_mask).astype(bool)
                overlay = canvas.copy()
                overlay[cm] = (255, 255, 0)
                canvas = cv2.addWeighted(overlay, 0.42, canvas, 0.58, 0)

            if center is not None:
                cx, cy = int(round(center[0])), int(round(center[1]))
                cv2.circle(canvas, (cx, cy), 7, (0, 0, 255), -1)
                cv2.putText(canvas, "color center", (cx + 10, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

            if target is not None:
                tx, ty = int(round(target[0])), int(round(target[1]))
                cv2.drawMarker(canvas, (tx, ty), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
                cv2.putText(canvas, "target", (tx + 10, ty + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2, cv2.LINE_AA)

            overlay_path = overlay_dir / f"{stem}_overlay.png"
            cv2.imwrite(str(overlay_path), canvas)
            saved["overlay_path"] = str(overlay_path)
        except Exception as e:
            self.log(f"[Step9-颜色中心] 保存检查图/分割结果失败：{e}")

        return saved

    def run_rule_ac_until_threshold(self) -> Dict[str, Any]:
        """
        Step9 新逻辑：颜色检测区域中心对齐。

        本版本支持：
            1. 每次单独测试 Step9 生成独立保存文件夹；
            2. 每轮保存原始截图、颜色区域 mask、A/B mask、overlay 和 CSV；
            3. 运动过程中可修改 RuleAC/Step9 输入，下一轮立即读取新值；
            4. 停止按钮可请求停止 1/2 通道运动。
        """
        self.log("========== Step9：颜色检测区域中心对齐，必要时移动1/2通道 ==========")
        self.step9_stop_requested = False

        # 如果是完整测量流程中直接调用 Step9，可能尚未创建 Step9 run 文件夹。
        self._get_step9_output_dir()

        records: List[Dict[str, Any]] = []
        final_ok = False
        last_record: Dict[str, Any] = {}

        cycle_idx = 1
        while True:
            # 运行过程中同步 GUI 最新 RuleAC 输入。
            # 单独测试时 GUI 会设置 self.step9_config_sync_callback；
            # 完整测量流程中没有该回调时则使用当前 self.cfg。
            if callable(getattr(self, "step9_config_sync_callback", None)):
                try:
                    self.step9_config_sync_callback()
                except Exception as e:
                    self.log(f"[Step9-颜色中心] 运行中同步 GUI 参数失败，本轮继续使用旧参数：{e}")

            target_x = float(getattr(self.cfg, "rule_ac_target_x_px", -1.0))
            target_y = float(getattr(self.cfg, "rule_ac_target_y_px", -1.0))
            tol = float(getattr(self.cfg, "rule_ac_center_tolerance_px", 5.0))
            max_cycles = int(getattr(self.cfg, "rule_ac_max_cycles", 50))
            loop_interval_s = float(getattr(self.cfg, "rule_ac_loop_interval_s", 0.10))

            if target_x < 0 or target_y < 0:
                self.log(
                    "[Step9-颜色中心] 未设置目标点 rule_ac_target_x/y。"
                    "请先在 GUI 中点击“选定Step9目标”，或在目标x/y输入框中填写截图区域内坐标。"
                )
                result = {
                    "ok": False,
                    "reason": "target_not_set",
                    "target": None,
                    "records": records,
                    "run_dir": str(self.step9_current_run_dir) if self.step9_current_run_dir else "",
                }
                self.context["last_rule_ac_result"] = result
                self.notify_update()
                return result

            # 当前版本不再用“最大循环次数”作为 Step9 的结束条件。
            # Step9 只在以下情况退出：
            #   1) 检测区域中心与选定目标点在容差内重合；
            #   2) 用户点击“停止1/2轴运动”或停止测量；
            #   3) 检测失败。
            # max_cycles 仍从 GUI 读取并记录到日志中，保留为参数兼容，但不用于停止循环。

            if self.stop_requested or self.step9_stop_requested or self._is_midrun_recalibration_requested():
                if self._is_midrun_recalibration_requested():
                    self.log("[Step9-颜色中心] 收到中途重标定请求，停止 1/2 通道并退出 Step9。")
                else:
                    self.log("[Step9-颜色中心] 收到停止请求，停止 1/2 通道并退出 Step9。")
                try:
                    stage_obj, _ = self._get_stage12_device() if self.stage12_device is not None else (None, False)
                    self._stop_stage12_if_possible(stage_obj)
                except Exception:
                    pass
                break

            det = self._capture_step9_color_center_once()
            if not det.get("ok", False):
                action_code = 0
                save_info = self._save_step9_color_overlay(
                    det.get("image_rgb"),
                    det.get("color_mask"),
                    None,
                    (target_x, target_y),
                    cycle_idx,
                    action_code,
                    a_mask=det.get("a_mask"),
                    b_mask=det.get("b_mask"),
                )
                last_record = {
                    "cycle": cycle_idx,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "ok": False,
                    "aligned": False,
                    "reason": det.get("reason"),
                    "center_x": "",
                    "center_y": "",
                    "target_x": target_x,
                    "target_y": target_y,
                    "dx": "",
                    "dy": "",
                    "tolerance_px": tol,
                    "color_area_px": det.get("color_area_px"),
                    "a_mask_area_px": det.get("a_mask_area_px"),
                    "b_mask_area_px": det.get("b_mask_area_px"),
                    "action_code": action_code,
                    "action_name": "",
                    "move_info": {"moved": False, "reason": det.get("reason")},
                    **save_info,
                }
                records.append(last_record)
                self._append_step9_history_csv({
                    **last_record,
                    "moved": False,
                    "dry_run": "",
                    "ch1_distance": "",
                    "ch2_distance": "",
                })
                self.log(f"[Step9-颜色中心] cycle={cycle_idx}, 检测失败：{last_record}")
                break

            cx, cy = det["color_center"]
            dx = target_x - float(cx)
            dy = target_y - float(cy)
            abs_dx = abs(dx)
            abs_dy = abs(dy)

            if abs_dx <= tol and abs_dy <= tol:
                action_code = 0
                final_ok = True
                move_info = {"moved": False, "reason": "center_aligned"}
                save_info = self._save_step9_color_overlay(
                    det.get("image_rgb"),
                    det.get("color_mask"),
                    (cx, cy),
                    (target_x, target_y),
                    cycle_idx,
                    action_code,
                    a_mask=det.get("a_mask"),
                    b_mask=det.get("b_mask"),
                )
                last_record = {
                    "cycle": cycle_idx,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "ok": True,
                    "aligned": True,
                    "reason": "center_aligned",
                    "center_x": cx,
                    "center_y": cy,
                    "target_x": target_x,
                    "target_y": target_y,
                    "dx": dx,
                    "dy": dy,
                    "tolerance_px": tol,
                    "color_area_px": det.get("color_area_px"),
                    "a_mask_area_px": det.get("a_mask_area_px"),
                    "b_mask_area_px": det.get("b_mask_area_px"),
                    "action_code": action_code,
                    "action_name": "",
                    "move_info": move_info,
                    **save_info,
                }
                records.append(last_record)
                self._append_step9_history_csv({
                    **last_record,
                    "moved": False,
                    "dry_run": "",
                    "ch1_distance": "",
                    "ch2_distance": "",
                })
                self.log(
                    f"[Step9-颜色中心] cycle={cycle_idx}, 已对齐："
                    f"center=({cx:.2f},{cy:.2f}), target=({target_x:.2f},{target_y:.2f}), "
                    f"dx={dx:.2f}, dy={dy:.2f}, tol={tol}"
                )
                break

            action_code = self._step9_direction_from_error(dx=dx, dy=dy, tol=tol)
            save_info = self._save_step9_color_overlay(
                det.get("image_rgb"),
                det.get("color_mask"),
                (cx, cy),
                (target_x, target_y),
                cycle_idx,
                action_code,
                b_mask=det.get("b_mask"),
                c_mask=det.get("c_mask"),
                c_raw_mask=det.get("c_raw_mask"),
            )

            if self.step9_stop_requested:
                self.log("[Step9-颜色中心] 运动前收到停止请求，跳过本次 1/2 通道动作。")
                move_info = {"moved": False, "reason": "step9_stop_requested_before_move"}
            else:
                # _execute_stage12_diagonal_action 会在执行前读取当前 self.cfg，
                # 所以 step_size / velocity / acceleration / enable_stage 的 GUI 修改会在本轮生效。
                move_info = self._execute_stage12_diagonal_action(action_code)

            action_name = move_info.get("action_name")
            last_record = {
                "cycle": cycle_idx,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ok": True,
                "aligned": False,
                "reason": "moving",
                "center_x": cx,
                "center_y": cy,
                "target_x": target_x,
                "target_y": target_y,
                "dx": dx,
                "dy": dy,
                "tolerance_px": tol,
                "color_area_px": det.get("color_area_px"),
                "a_mask_area_px": det.get("a_mask_area_px"),
                "b_mask_area_px": det.get("b_mask_area_px"),
                "action_code": action_code,
                "action_name": action_name,
                "move_info": move_info,
                **save_info,
            }
            records.append(last_record)
            self._append_step9_history_csv({
                **last_record,
                "moved": move_info.get("moved"),
                "dry_run": move_info.get("dry_run"),
                "ch1_distance": move_info.get("ch1_distance"),
                "ch2_distance": move_info.get("ch2_distance"),
            })

            self.log(
                f"[Step9-颜色中心] cycle={cycle_idx}, "
                f"center=({cx:.2f},{cy:.2f}), target=({target_x:.2f},{target_y:.2f}), "
                f"dx={dx:.2f}, dy={dy:.2f}, tol={tol}, "
                f"color_area={det.get('color_area_px')}, "
                f"step={getattr(self.cfg, 'rule_ac_stage_step_size', None)}, "
                f"max_cycles={max_cycles}(ignored), enable_stage={getattr(self.cfg, 'rule_ac_enable_stage', None)}, "
                f"action={action_code}({action_name}), moved={move_info.get('moved')}"
            )

            cycle_idx += 1
            time.sleep(loop_interval_s)

        result = {
            "ok": bool(final_ok),
            "reason": ("aligned" if final_ok else ("midrun_recalibration_requested" if self._is_midrun_recalibration_requested() else "not_aligned_or_stopped_or_detect_failed")),
            "target": (
                float(getattr(self.cfg, "rule_ac_target_x_px", -1.0)),
                float(getattr(self.cfg, "rule_ac_target_y_px", -1.0)),
            ),
            "tolerance_px": float(getattr(self.cfg, "rule_ac_center_tolerance_px", 5.0)),
            "history_len": len(records),
            "last_row": last_record,
            "records": records,
            "csv_path": str(self.step9_history_csv_path) if self.step9_history_csv_path else "",
            "run_dir": str(self.step9_current_run_dir) if self.step9_current_run_dir else "",
        }
        self.context["last_rule_ac_result"] = result
        self.notify_update()
        return result

    # --------------------------------------------------------
    # LabVIEW 光谱与保存
    # --------------------------------------------------------

    @staticmethod
    def _to_float_list(values: Any) -> Optional[List[float]]:
        """把 LabVIEW/TCP 返回的列表、元组、numpy 数组或字符串统一转成 float 列表。"""
        if values is None:
            return None

        if isinstance(values, np.ndarray):
            values = values.tolist()

        # 字符串里可能带有 DATA/RAW_Y/END 等前缀，不能简单 split 后 float，
        # 这里直接用正则提取所有数字。
        if isinstance(values, str):
            nums = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", values)
            out = []
            for x in nums:
                try:
                    fx = float(x)
                    if math.isfinite(fx):
                        out.append(fx)
                except Exception:
                    continue
            return out

        # 标量数值也允许作为单点光谱/峰值兜底。
        if isinstance(values, (int, float, np.integer, np.floating)):
            try:
                fx = float(values)
                return [fx] if math.isfinite(fx) else []
            except Exception:
                return []

        out: List[float] = []
        try:
            for x in values:
                try:
                    if x is None:
                        continue
                    # nested list/dict 不在这里展开，交给 _extract_spectrum_arrays_from_any；
                    # 这里仅处理一维列表。
                    if isinstance(x, (list, tuple, dict, np.ndarray)):
                        continue
                    fx = float(x)
                    if math.isfinite(fx):
                        out.append(fx)
                except Exception:
                    continue
        except TypeError:
            return None

        return out

    @staticmethod
    def _numeric_from_any(v: Any) -> Optional[float]:
        """把单个对象转成有限 float；失败返回 None。"""
        try:
            if v is None:
                return None
            if isinstance(v, str):
                text = v.strip()
                if not text:
                    return None
                m = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
                if not m:
                    return None
                v = m.group(0)
            fx = float(v)
            return fx if math.isfinite(fx) else None
        except Exception:
            return None

    @classmethod
    def _extract_spectrum_arrays_from_any(cls, obj: Any) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        """
        从 LabVIEWTCPServer 返回的任意结构里提取 wavelength/intensity。

        重点兼容这些情况：
            1) values 是一维数值列表；
            2) values 是二维行数据：[[x,y], [x,y], ...]；
            3) values 是字符串："DATA,1,2,3" 或多行 "x,y"；
            4) values 是 dict：{"wavelength": [...], "intensity": [...]}；
            5) result 本身含 raw_y/intensity/data/spectrum 等字段。
        """
        if obj is None:
            return None, None

        if isinstance(obj, np.ndarray):
            obj = obj.tolist()

        # dict：优先按明确字段取 y，再取 x。
        if isinstance(obj, dict):
            x = None
            y = None
            for x_key in ("wavelength", "lambda", "x", "x_axis", "center_wavelength", "波长", "中心波长"):
                if x is None and x_key in obj:
                    x = cls._to_float_list(obj.get(x_key))
            for y_key in ("raw_y", "RAW_Y", "raw_values", "intensity", "values", "data", "y", "spectrum"):
                if y is None and y_key in obj:
                    val = obj.get(y_key)
                    if isinstance(val, dict):
                        x2, y2 = cls._extract_spectrum_arrays_from_any(val)
                        x = x or x2
                        y = y2
                    else:
                        x2, y2 = cls._extract_spectrum_arrays_from_any(val)
                        x = x or x2
                        y = y2
            if y:
                return x, y

            # 没有标准 key 时，尝试所有 value，取最长的一组数值作为 y。
            best_x, best_y = None, None
            for val in obj.values():
                x2, y2 = cls._extract_spectrum_arrays_from_any(val)
                if y2 and (best_y is None or len(y2) > len(best_y)):
                    best_x, best_y = x2, y2
            return best_x, best_y

        # 字符串：按行判断是否是 x,y；否则提取所有数字作为 y。
        if isinstance(obj, str):
            lines = [ln.strip() for ln in obj.replace(";", "\n").splitlines() if ln.strip()]
            x_vals: List[float] = []
            y_vals: List[float] = []
            row_like_count = 0
            for ln in lines:
                nums = [float(m) for m in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", ln)]
                if len(nums) >= 2:
                    row_like_count += 1
                    x_vals.append(float(nums[0]))
                    y_vals.append(float(nums[-1]))
                elif len(nums) == 1:
                    y_vals.append(float(nums[0]))
            if row_like_count >= 2 and len(y_vals) >= row_like_count:
                return x_vals[:row_like_count], y_vals[:row_like_count]
            flat = cls._to_float_list(obj)
            return None, flat

        # 标量。
        if isinstance(obj, (int, float, np.integer, np.floating)):
            y = cls._to_float_list(obj)
            return None, y

        # list/tuple：先判断二维表格；否则递归展开，取最长可用数据。
        if isinstance(obj, (list, tuple)):
            if len(obj) == 0:
                return None, None

            # 二维行数据：每行至少 2 个数，按第一列 x、最后一列 y。
            x_vals: List[float] = []
            y_vals: List[float] = []
            row_count = 0
            for row in obj:
                if isinstance(row, np.ndarray):
                    row = row.tolist()
                if isinstance(row, (list, tuple)):
                    nums = [cls._numeric_from_any(v) for v in row]
                    nums = [v for v in nums if v is not None]
                    if len(nums) >= 2:
                        row_count += 1
                        x_vals.append(float(nums[0]))
                        y_vals.append(float(nums[-1]))
                elif isinstance(row, dict):
                    x2, y2 = cls._extract_spectrum_arrays_from_any(row)
                    if y2 and len(y2) >= 2:
                        return x2, y2
            if row_count >= 2:
                return x_vals, y_vals

            # 一维数值列表。
            direct = cls._to_float_list(obj)
            if direct:
                return None, direct

            # 混合嵌套结构：递归找最长 y。
            best_x, best_y = None, None
            for item in obj:
                x2, y2 = cls._extract_spectrum_arrays_from_any(item)
                if y2 and (best_y is None or len(y2) > len(best_y)):
                    best_x, best_y = x2, y2
            return best_x, best_y

        return None, None

    def _preview_labview_value(self, value: Any, max_chars: int = 500) -> str:
        """日志用：只显示 TCP 返回 values 的前几百字符，避免刷屏。"""
        try:
            text = repr(value)
        except Exception:
            text = f"<unreprable {type(value).__name__}>"
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        return text

    @staticmethod
    def _remove_above_threshold(values: Optional[List[float]], threshold: Optional[float]) -> List[float]:
        """删除高于 threshold 的数据点；threshold 为空或非有限值时不删除。"""
        if not values:
            return []

        try:
            th = float(threshold)
        except Exception:
            return list(values)

        if not math.isfinite(th):
            return list(values)

        return [float(v) for v in values if float(v) <= th]

    @staticmethod
    def _median_filter_1d(values: Optional[List[float]], window: int = 5) -> List[float]:
        """
        1D 中值滤波，默认 5 点窗口。输出长度与输入长度一致。
        边缘处使用可用窗口，不补零，避免人为引入极端值。
        """
        if not values:
            return []

        try:
            k = int(window)
        except Exception:
            k = 5

        if k <= 1:
            return [float(v) for v in values]

        if k % 2 == 0:
            k += 1

        half = k // 2
        arr = [float(v) for v in values]
        filtered: List[float] = []

        for i in range(len(arr)):
            left = max(0, i - half)
            right = min(len(arr), i + half + 1)
            filtered.append(float(np.median(arr[left:right])))

        return filtered

    def _resolve_existing_path(self, path_text: Optional[str]) -> Optional[Path]:
        """
        解析 GUI 里填写的文件路径。

        支持：
            1. 绝对路径；
            2. 相对于当前主程序所在目录的路径；
            3. 相对于当前工作目录的路径。
        """
        if path_text is None:
            return None

        s = str(path_text).strip().strip('"').strip("'")
        if not s:
            return None

        candidates = []
        p = Path(s)
        candidates.append(p)
        if not p.is_absolute():
            candidates.append(PROJECT_ROOT / s)
            candidates.append(Path.cwd() / s)

        for c in candidates:
            try:
                if c.exists() and c.is_file():
                    return c
            except Exception:
                continue
        return None

    def load_x_axis_values_from_xlsx(self) -> List[float]:
        """
        从 cfg.x_axis_xlsx_path 指向的 .xlsx 文件中读取横坐标。

        读取规则：
            - 优先读取第一张包含数值的工作表；
            - 默认取第 1 列中的所有数值；
            - 跳过空值和不能转成 float 的单元格；
            - 读取失败或文件为空时返回 []，绘图时会自动退回 index。
        """
        xlsx_path = self._resolve_existing_path(self.cfg.x_axis_xlsx_path)
        if xlsx_path is None:
            self.log(f"[横坐标xlsx] 文件不存在或未设置：{self.cfg.x_axis_xlsx_path}，图2/图3将使用 index")
            return []

        try:
            wb = load_workbook(xlsx_path, data_only=True, read_only=True)
        except Exception as e:
            self.log(f"[横坐标xlsx] 读取失败：{xlsx_path}，原因：{e}，图2/图3将使用 index")
            return []

        values: List[float] = []
        sheet_name = None

        try:
            for ws in wb.worksheets:
                tmp: List[float] = []
                for row in ws.iter_rows(min_row=1, max_col=1, values_only=True):
                    if not row:
                        continue
                    v = row[0]
                    if v is None:
                        continue
                    try:
                        fv = float(v)
                    except Exception:
                        continue
                    if math.isfinite(fv):
                        tmp.append(fv)

                if tmp:
                    values = tmp
                    sheet_name = ws.title
                    break
        finally:
            try:
                wb.close()
            except Exception:
                pass

        if values:
            self.log(f"[横坐标xlsx] 已读取：{xlsx_path}，sheet={sheet_name}，点数={len(values)}")
        else:
            self.log(f"[横坐标xlsx] 未读取到有效数值：{xlsx_path}，图2/图3将使用 index")

        self.context["x_axis_xlsx_path"] = str(xlsx_path)
        self.context["x_axis_values"] = values
        return values

    @staticmethod
    def _make_x_values_for_plot(y_values: List[float], x_values: Optional[List[float]]) -> List[float]:
        """
        为一组 y 数据生成横坐标。
        如果 xlsx 横坐标数量足够，使用 xlsx；否则退回 0,1,2,...。
        """
        n = len(y_values or [])
        if n <= 0:
            return []

        if x_values is not None and len(x_values) >= n:
            return [float(v) for v in x_values[:n]]

        return list(range(n))

    def _get_expected_spectrum_point_count_quiet(self) -> Optional[int]:
        """
        返回预期光谱点数，用来区分真正 1024 点光谱和 TCP 状态/日志字段。

        你的横坐标 xlsx 是 1024 点，因此完整测量 Step 4 只有拿到接近 1024
        的 y 数组时，才认为它和“单次光谱采集”是同一类真实光谱数据。
        这里不调用 load_x_axis_values_from_xlsx()，避免在候选 CSV 扫描时反复刷日志。
        """
        try:
            cached = self.context.get("x_axis_values")
            if isinstance(cached, (list, tuple)) and len(cached) >= 100:
                return int(len(cached))
        except Exception:
            pass

        try:
            xlsx_path = self._resolve_existing_path(getattr(self.cfg, "x_axis_xlsx_path", ""))
            if xlsx_path is None or not xlsx_path.exists():
                return None
            wb = load_workbook(xlsx_path, data_only=True, read_only=True)
            try:
                best = 0
                for ws in wb.worksheets:
                    n = 0
                    for row in ws.iter_rows(min_row=1, max_col=1, values_only=True):
                        if not row:
                            continue
                        v = row[0]
                        if v is None:
                            continue
                        try:
                            fv = float(v)
                            if math.isfinite(fv):
                                n += 1
                        except Exception:
                            continue
                    best = max(best, n)
                return best if best >= 100 else None
            finally:
                try:
                    wb.close()
                except Exception:
                    pass
        except Exception:
            return None

    def _min_usable_spectrum_points(self) -> int:
        """
        判定真实光谱数组的最小点数。

        若横坐标 xlsx 是 1024 点，则要求 y 至少达到 80% 点数，即约 819 点。
        这样可以防止把 [20260616, 2026, 6, 16, ...] 这类日期/状态字段误判为光谱。
        """
        expected = self._get_expected_spectrum_point_count_quiet()
        if expected is not None and expected >= 100:
            return max(32, int(expected * 0.80))
        return 32

    def _labview_result_has_usable_spectrum(self, result: Any) -> bool:
        """
        判断 LabVIEW 返回对象里是否真的包含可用于绘图/保存的一组光谱点。

        关键修复：
            - 不能再用 len(y)>=2 判定真实光谱；
            - 你的 LabVIEW 光谱应为 1024 点，9 点数据通常是日期/状态/日志字段；
            - 因此必须用“最小有效点数”过滤掉非光谱状态包。
        """
        if result is None:
            return False

        min_points = self._min_usable_spectrum_points()

        def _ok(y: Optional[List[float]]) -> bool:
            return bool(y and len(y) >= min_points)

        try:
            if isinstance(result, (str, Path)):
                fp = Path(str(result))
                if not fp.exists() or not fp.is_file():
                    return False
                _, y = self._load_spectrum_from_csv(str(fp))
                return _ok(y)

            if not isinstance(result, dict):
                _, y = self._extract_spectrum_arrays_from_any(result)
                return _ok(y)

            csv_path = str(result.get("csv_path") or "").strip()
            if csv_path:
                fp = Path(csv_path)
                if fp.exists() and fp.is_file():
                    _, y = self._load_spectrum_from_csv(str(fp))
                    if _ok(y):
                        return True

            # 只检查明确可能承载光谱的字段，不再递归扫描整个 result。
            # 否则 ok/reason/index/日期/文件名里的数字会被拼成 9 个点，误判成光谱。
            for key in ("raw_y", "RAW_Y", "raw_values", "intensity", "spectrum", "y", "data", "values", "fit_y", "FIT_Y"):
                if key not in result:
                    continue
                _, y = self._extract_spectrum_arrays_from_any(result.get(key))
                if _ok(y):
                    return True

            return False
        except Exception:
            return False

    def _normalize_labview_result(self, value: Any, cycle_index: int, command: str, method_name: str) -> Dict[str, Any]:
        """把任意 LabVIEW 返回值统一包装成 result dict。"""
        if isinstance(value, (str, Path)):
            result = {"csv_path": str(value)}
        elif isinstance(value, dict):
            result = dict(value)
        else:
            result = {"values": value}
        result.setdefault("ok", True)
        result.setdefault("index", cycle_index)
        result.setdefault("command", command)
        result.setdefault("method", method_name)
        return result

    def _collect_labview_result_after_request(
        self,
        server: Any,
        cycle_index: int,
        command: str,
        method_name: str = "",
        request_started_at: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        request_measure() 等接口可能只负责触发采集，真正数据保存在 server 属性或 CSV 中。

        修改点：
            1. 不再“读到任何 attr 就返回”；必须确认 attr/CSV 里有真实光谱数组；
            2. 若 attr 只是 ok/reason/index/num_points 这类状态包，则继续找其它 attr 或最新 CSV；
            3. latest_csv 优先取本次请求之后生成/更新的文件，避免误读很久以前的旧 CSV。
        """
        status_candidate: Optional[Dict[str, Any]] = None

        # 1) 优先读 server 暴露的结果属性。
        for attr in (
            "last_result",
            "result",
            "latest_result",
            "last_data",
            "data",
            "last_csv_path",
            "csv_path",
            "latest_csv_path",
            "last_file",
        ):
            try:
                value = getattr(server, attr, None)
                value = value() if callable(value) else value
                if value is None:
                    continue

                result = self._normalize_labview_result(
                    value, cycle_index, command, f"{method_name}+attr:{attr}"
                )
                if self._labview_result_has_usable_spectrum(result):
                    self.log(f"[TCP] 已从 LabVIEWTCPServer.{attr} 获取真实光谱结果")
                    return result

                if status_candidate is None:
                    status_candidate = result
            except Exception:
                continue

        # 2) 再从 output_dir/save_dir 中找最新 CSV。LabVIEWTCPServer 日志里 output_dir=labview_csv_output。
        candidate_dirs: List[Path] = []
        for attr in ("output_dir", "save_dir"):
            try:
                d = getattr(server, attr, None)
                if d:
                    candidate_dirs.append(Path(str(d)))
            except Exception:
                pass
        try:
            if getattr(self.cfg, "tcp_output_dir", ""):
                candidate_dirs.append(Path(str(self.cfg.tcp_output_dir)))
        except Exception:
            pass

        newest_csv: Optional[Path] = None
        newest_mtime = -1.0
        min_mtime = None
        if request_started_at is not None:
            # 留 2 秒余量，兼容 Windows 文件时间戳和 LabVIEW 写文件延迟。
            min_mtime = float(request_started_at) - 2.0

        seen_dirs = set()
        for d in candidate_dirs:
            try:
                d = d.expanduser().resolve()
                if str(d) in seen_dirs:
                    continue
                seen_dirs.add(str(d))
                if not d.exists():
                    continue
                for fp in d.rglob("*.csv"):
                    try:
                        mt = fp.stat().st_mtime
                    except Exception:
                        continue
                    if min_mtime is not None and mt < min_mtime:
                        continue
                    if mt > newest_mtime:
                        candidate = {
                            "ok": True,
                            "index": cycle_index,
                            "command": command,
                            "method": f"{method_name}+latest_csv",
                            "csv_path": str(fp),
                        }
                        if self._labview_result_has_usable_spectrum(candidate):
                            newest_mtime = mt
                            newest_csv = fp
            except Exception:
                continue

        if newest_csv is not None:
            self.log(f"[TCP] 已从最新 CSV 获取真实光谱结果：{newest_csv}")
            return {
                "ok": True,
                "index": cycle_index,
                "command": command,
                "method": f"{method_name}+latest_csv",
                "csv_path": str(newest_csv),
            }

        return status_candidate


    def _request_labview_spectrum_compat(self, cycle_index: int) -> Dict[str, Any]:
        """
        通过 LabVIEWTCPServer 请求一次真实光谱，兼容不同方法名。

        关键修复：完整测量 Step 4 和 GUI“单次光谱采集”共用 request_labview_spectrum()。
        request_measure() 若只返回 ok/reason/index/num_points 状态包，不允许 Step 4 直接结束；
        必须继续按单次采集可成功的路径去读 last_result/latest_result/last_data/data 或最新 CSV。
        """
        if self.tcp_server is None:
            self.start_tcp_server()
        if not bool(self.context.get("labview_ready", False)):
            self.wait_labview_ready()

        server = self.tcp_server
        if server is None:
            raise RuntimeError("TCP Server 未创建，无法采集光谱")

        command = str(self.cfg.tcp_command or "MEASURE")
        request_method_names = (
            # 你的 control/labview_tcp_server.py 当前公开的是 request_measure。
            "request_measure",
            "request_measurement",
            "measure",
            "request_spectrum",
            "get_spectrum",
            "acquire_spectrum",
            "acquire",
            "send_command_and_wait",
            "send_command",
            "request",
        )

        last_exc: Optional[BaseException] = None
        last_status_result: Optional[Dict[str, Any]] = None

        for name in request_method_names:
            method = getattr(server, name, None)
            if not callable(method):
                continue
            try:
                request_started_at = time.time()

                # 依次尝试常见签名。优先传 save_csv=True，保证完整测量和单次采集
                # 都能让 LabVIEWTCPServer 保存/暴露同一份真实 1024 点 CSV。
                try:
                    raw = method(command=command, index=cycle_index, save_csv=True)
                except TypeError:
                    try:
                        raw = method(command=command, index=cycle_index)
                    except TypeError:
                        try:
                            raw = method(command, cycle_index, True)
                        except TypeError:
                            try:
                                raw = method(command)
                            except TypeError:
                                try:
                                    raw = method(cycle_index)
                                except TypeError:
                                    raw = method()

                result = None
                if raw is not None:
                    result = self._normalize_labview_result(raw, cycle_index, command, name)
                    if self._labview_result_has_usable_spectrum(result):
                        self.log(f"[TCP] 已通过 LabVIEWTCPServer.{name}() 直接获取真实光谱结果")
                        return result
                    last_status_result = result
                    self.log(
                        f"[TCP] LabVIEWTCPServer.{name}() 返回状态包/非光谱数据，"
                        "继续读取 last_result/latest_result/last_data/data 或最新 CSV。"
                    )

                fallback = self._collect_labview_result_after_request(
                    server,
                    cycle_index,
                    command,
                    method_name=name,
                    request_started_at=request_started_at,
                )
                if fallback is not None and self._labview_result_has_usable_spectrum(fallback):
                    fallback.setdefault("trigger_method", name)
                    self.log(f"[TCP] 已通过 {fallback.get('method')} 获取真实光谱结果")
                    return fallback

                if fallback is not None and last_status_result is None:
                    last_status_result = fallback

                # 当前方法已经成功触发过 LabVIEW，但没有拿到数组；不要再无意义地调用其它触发方法，
                # 避免连续发送多次 MEASURE。其它方法名通常只是同一功能的别名。
                if raw is not None:
                    break

            except Exception as e:
                last_exc = e
                self.log(f"[TCP] 调用 {name}() 失败，尝试下一个接口：{e}")

        # 所有触发方法都没给出可用数组时，再做一次全局兜底读取。
        fallback = self._collect_labview_result_after_request(
            server, cycle_index, command, method_name="final_fallback", request_started_at=None
        )
        if fallback is not None and self._labview_result_has_usable_spectrum(fallback):
            self.log(f"[TCP] 已通过 {fallback.get('method')} 获取真实光谱结果")
            return fallback

        if last_status_result is not None:
            # 返回状态包给 request_labview_spectrum()，由那里生成包含 values_preview/debug_json 的明确报错。
            return last_status_result

        methods = [m for m in dir(server) if not m.startswith("_")]
        raise RuntimeError(
            "LabVIEWTCPServer 未提供可用的采集接口。已尝试："
            f"{request_method_names} 和 last_result/result/latest_result/last_data/data/latest_csv；"
            f"最后错误：{last_exc}；当前公开成员：{methods}"
        )


    def _make_virtual_spectrum_result(self, cycle_index: int) -> Dict[str, Any]:
        """
        virtual 模式固定光谱结果。

        关键点：
            1. 不调用真实 LabVIEW/TCP；
            2. 不调用真实光谱处理链路中的阈值过滤、中值滤波、拟合函数；
            3. 直接给固定峰值 DUMMY_SPECTRUM_PEAK，避免 virtual 模式因为光谱处理函数缺失而中断；
            4. 仍然填充完整测量后续保存/绘图常用字段，保证 CSV/XLSX/records 不缺键。
        """
        fixed_peak = float(globals().get("DUMMY_SPECTRUM_PEAK", 1000.0))
        try:
            n = int(getattr(self.cfg, "virtual_spectrum_points", 1024) or 1024)
        except Exception:
            n = 1024
        n = max(1, n)

        # virtual 固定值模式不再读取 x_axis_xlsx，也不再做任何真实光谱处理。
        # 这里给一个 index 横坐标和常数光谱数组，仅用于兼容后续保存/绘图字段。
        plot_x_values = [float(i) for i in range(n)]
        raw_values = [fixed_peak for _ in range(n)]
        threshold_values = list(raw_values)
        median_values = list(raw_values)
        fit_values = list(raw_values)

        raw_original_peak = fixed_peak
        raw_filtered_peak = fixed_peak
        raw_median_peak = fixed_peak
        raw_peak = fixed_peak
        fit_peak = fixed_peak

        self.context["wavelength"] = plot_x_values
        self.context["x_axis_values"] = plot_x_values
        self.context["x_axis_xlsx_path"] = ""
        self.context["intensity"] = median_values
        self.context["raw_values"] = raw_values
        self.context["raw_filtered_values"] = threshold_values
        self.context["raw_median_values"] = median_values
        self.context["raw_original_peak"] = raw_original_peak
        self.context["raw_filtered_peak"] = raw_filtered_peak
        self.context["raw_median_peak"] = raw_median_peak
        self.context["raw_filter_threshold"] = None
        self.context["median_filter_window"] = None
        self.context["raw_peak"] = raw_peak
        self.context["fit_peak"] = fit_peak
        self.context["fit_values"] = fit_values
        self.context["fit_params"] = {
            "mode": "virtual_fixed_value",
            "fixed_peak": fixed_peak,
            "note": "virtual mode does not call hardware spectrum processing",
        }
        self.context["labview_csv_path"] = ""

        result = {
            "ok": True,
            "virtual": True,
            "index": int(cycle_index),
            "command": str(self.cfg.tcp_command or "MEASURE"),
            "method": "virtual_fixed_spectrum",
            "num_points": n,
            "wavelength": plot_x_values,
            "x_axis_values": plot_x_values,
            "raw_values": raw_values,
            "intensity": raw_values,
            "threshold_filtered_intensity": threshold_values,
            "median_filtered_intensity": median_values,
            "fit_values": fit_values,
            "raw_original_peak": raw_original_peak,
            "raw_filtered_peak": raw_filtered_peak,
            "raw_median_peak": raw_median_peak,
            "raw_peak": raw_peak,
            "fit_peak": fit_peak,
            "raw_filter_threshold": None,
            "median_filter_window": None,
            "x_axis_xlsx_path": "",
            "csv_path": None,
        }

        self.context["last_tcp_result"] = result
        self.log(
            f"[光谱][virtual] 固定值光谱结果：num_points={n}, "
            f"raw_original_peak={raw_original_peak}, raw_filtered_peak={raw_filtered_peak}, "
            f"raw_median_peak={raw_median_peak}, fit_peak={fit_peak}；"
            "未调用 LabVIEW/TCP/阈值滤波/中值滤波/拟合处理"
        )
        self.notify_update()
        return result

    def request_labview_spectrum(self, cycle_index: int) -> Dict[str, Any]:
        """
        请求并解析一次真实 LabVIEW 光谱。

        这个函数是唯一的光谱采集入口：
            - GUI“单次光谱采集”调用它；
            - “运行完整循环测量”的 Step 4 也调用它。

        因此完整测量和单次采集会使用同一套 TCP 触发、last_result/CSV 兜底、
        横坐标 xlsx、阈值滤波、中值滤波、峰值/拟合峰值计算逻辑。
        """
        self.log("========== LabVIEW 采集光谱 ==========")

        if self._is_virtual_hardware_mode():
            return self._make_virtual_spectrum_result(cycle_index)

        if bool(globals().get("SPECTROMETER_TCP_DISABLED", False)):
            raise RuntimeError(
                "SPECTROMETER_TCP_DISABLED=True，当前被设置为跳过真实光谱仪。"
                "本版本不再用1000伪造光谱数据；请把 SPECTROMETER_TCP_DISABLED 改为 False。"
            )

        # 关键修复：不要直接调用 self.tcp_server.request_measure() 后就解析状态包。
        # request_measure() 在你的 LabVIEWTCPServer 中可能只返回 ok/index/num_points 等状态，
        # 真正的 1024 点光谱可能在 last_result/latest_result/last_data 或最新 CSV 中。
        # 这里统一走兼容采集函数，和单次光谱采集完全共用同一条链路。
        result = self._request_labview_spectrum_compat(cycle_index=cycle_index)
        self.context["last_tcp_result"] = result

        if not isinstance(result, dict):
            result = self._normalize_labview_result(result, cycle_index, str(self.cfg.tcp_command or "MEASURE"), "request_labview_spectrum")

        if not result.get("ok", False):
            raise RuntimeError(f"LabVIEW 采集失败：{result}")

        parsed = self._parse_labview_result(result)

        raw_values = parsed.get("raw_values") or []
        threshold_values = parsed.get("threshold_filtered_intensity") or []
        median_values = parsed.get("median_filtered_intensity") or []
        fit_values = parsed.get("fit_values") or []

        min_points = self._min_usable_spectrum_points()
        if len(raw_values) < min_points:
            raise RuntimeError(
                "LabVIEW 返回结果中没有可用的完整真实光谱数组："
                f"raw_points={len(raw_values)}, min_required={min_points}, "
                f"result_keys={list(result.keys())}, "
                f"csv_path={result.get('csv_path')}, "
                f"method={result.get('method')}, "
                f"preview={self._preview_labview_value(result)}"
            )

        # 图2/图3/保存 CSV 横坐标：优先使用 GUI 选择的 xlsx 第一列；
        # 这一步与“单次光谱采集”完全一致。读取失败时绘图/保存再退回 index 或 LabVIEW wavelength。
        x_axis_values = self.load_x_axis_values_from_xlsx()
        plot_x_values = self._make_x_values_for_plot(raw_values, x_axis_values)

        # 若 LabVIEW 只给了原始 y，没有 fit_y，则用中值滤波曲线作为显示/保存的拟合曲线兜底；
        # fit_peak 仍按 _parse_labview_result() 中的逻辑优先使用 LabVIEW fit_peak，
        # 其次 fit_values 最大值，最后 raw_peak。
        if not fit_values:
            fit_values = list(median_values)
            parsed["fit_values"] = fit_values

        self.context["wavelength"] = parsed.get("wavelength")
        self.context["x_axis_values"] = x_axis_values
        self.context["x_axis_xlsx_path"] = self.cfg.x_axis_xlsx_path
        self.context["intensity"] = median_values
        self.context["raw_values"] = raw_values
        self.context["raw_filtered_values"] = threshold_values
        self.context["raw_median_values"] = median_values
        self.context["raw_original_peak"] = parsed.get("raw_original_peak")
        self.context["raw_filtered_peak"] = parsed.get("raw_filtered_peak")
        self.context["raw_median_peak"] = parsed.get("raw_median_peak")
        self.context["raw_filter_threshold"] = self.cfg.raw_remove_above
        self.context["median_filter_window"] = self.cfg.median_filter_window
        self.context["raw_peak"] = parsed.get("raw_peak")
        self.context["fit_peak"] = parsed.get("fit_peak")
        self.context["fit_values"] = fit_values
        self.context["fit_params"] = parsed.get("fit_params")

        # 把解析后的真实峰值也写回 result，异步保存快照/GUI显示直接用同一组值。
        result["num_points"] = len(raw_values)
        result["raw_original_peak"] = parsed.get("raw_original_peak")
        result["raw_filtered_peak"] = parsed.get("raw_filtered_peak")
        result["raw_median_peak"] = parsed.get("raw_median_peak")
        result["raw_peak"] = parsed.get("raw_peak")
        result["fit_peak"] = parsed.get("fit_peak")
        result["raw_filter_threshold"] = self.cfg.raw_remove_above
        result["median_filter_window"] = self.cfg.median_filter_window
        result["x_axis_xlsx_path"] = self.cfg.x_axis_xlsx_path

        self.log(
            f"[LabVIEW] 采集完成：index={result.get('index')}, "
            f"method={result.get('method')}, num_points={len(raw_values)}, csv_path={result.get('csv_path')}"
        )
        self.log(
            f"[光谱] 真实采集完成：num_points={len(raw_values)}, "
            f"x_points={len(x_axis_values)}, plot_x_points={len(plot_x_values)}, "
            f"raw_original_peak={self.context['raw_original_peak']}, "
            f"raw_filtered_peak={self.context['raw_filtered_peak']}, "
            f"raw_median_peak={self.context['raw_median_peak']}, "
            f"fit_peak={self.context['fit_peak']}"
        )
        self.log(
            f"[光谱] 阈值去除后点数={len(threshold_values)}，阈值={self.cfg.raw_remove_above}；"
            f"{self.cfg.median_filter_window}点中值滤波后点数={len(median_values)}"
        )

        self.notify_update()
        return result

    def save_single_spectrum_to_xlsx(
        self,
        save_dir: Optional[Path] = None,
        tag: str = "",
    ) -> Optional[Path]:
        """
        将当前 context 中的光谱数据保存为 xlsx。

        文件路径：save_dir / measurement_summary_{tag}_{timestamp}.xlsx
        若 save_dir 为 None，默认使用 {output_root}/save/{MM.DD}。
        工作表：光谱数据，列包括波长/索引、原始强度、阈值滤波、中值滤波、拟合曲线。
        """
        try:
            raw_values = self.context.get("raw_values") or []
            threshold_values = self.context.get("raw_filtered_values") or []
            median_values = self.context.get("raw_median_values") or []
            fit_values = self.context.get("fit_values") or []
            x_axis_values = self.context.get("x_axis_values") or []

            if not raw_values:
                self.log("[保存光谱数据] 无原始光谱数据，跳过保存")
                return None

            if save_dir is None:
                save_dir = self._get_today_save_dir()
            else:
                save_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tag_part = f"{tag}_" if tag else ""
            xlsx_path = save_dir / f"measurement_summary_{tag_part}{timestamp}.xlsx"

            wb = Workbook()
            ws = wb.active
            ws.title = "光谱数据"
            ws.append(["波长/索引", "原始强度", "阈值滤波", "中值滤波", "拟合曲线"])

            n = len(raw_values)
            for i in range(n):
                x_val = x_axis_values[i] if i < len(x_axis_values) else i
                row = [
                    x_val,
                    raw_values[i],
                    threshold_values[i] if i < len(threshold_values) else None,
                    median_values[i] if i < len(median_values) else None,
                    fit_values[i] if i < len(fit_values) else None,
                ]
                ws.append(row)

            # 简单样式
            header_fill = PatternFill("solid", fgColor="1F4E78")
            header_font = Font(bold=True, color="FFFFFF")
            center_alignment = Alignment(horizontal="center", vertical="center")
            for cell in ws[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = center_alignment

            wb.save(xlsx_path)
            self.log(f"[保存光谱数据] 已保存：{xlsx_path}")
            return xlsx_path
        except Exception as e:
            self.log(f"[保存光谱数据] 保存失败：{e}")
            self.log(traceback.format_exc())
            return None


    def _parse_labview_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析 LabVIEW 返回的真实光谱。

        关键修复：
            1. 不再先递归扫描整个 result；
               否则 ok/reason/index/日期/文件名里的数字可能被误拼成 9 个点。
            2. 只从明确的光谱字段或可读取 CSV 中取数据。
            3. 若横坐标 xlsx 是 1024 点，则低于有效点数阈值的候选数组会被拒绝。
        """
        wavelength = None
        raw_values = None
        fit_values = None

        min_points = self._min_usable_spectrum_points()

        def _accept_y(y: Optional[List[float]]) -> bool:
            return bool(y and len(y) >= min_points)

        def _try_set_from_value(value: Any, label: str) -> bool:
            nonlocal wavelength, raw_values
            x2, y2 = self._extract_spectrum_arrays_from_any(value)
            if _accept_y(y2):
                wavelength = x2 or wavelength
                raw_values = y2
                return True
            if y2:
                self.log(
                    f"[光谱] 已忽略非完整光谱候选：source={label}, "
                    f"points={len(y2)}, min_required={min_points}, "
                    f"preview={self._preview_labview_value(y2[:10])}"
                )
            return False

        # 1) 明确字段优先：这些字段才允许作为光谱数组来源。
        if isinstance(result, dict):
            if "wavelength" in result:
                w = self._to_float_list(result.get("wavelength"))
                if w:
                    wavelength = w

            for key in ("raw_y", "RAW_Y", "raw_values", "intensity", "y", "spectrum", "data", "values"):
                if key in result and _try_set_from_value(result.get(key), key):
                    break

            # 2) CSV 次之。只有 CSV 里读到足够点数才接受。
            if not raw_values and result.get("csv_path"):
                w2, y2 = self._load_spectrum_from_csv(str(result["csv_path"]))
                if _accept_y(y2):
                    wavelength = w2 or wavelength
                    raw_values = y2
                elif y2:
                    self.log(
                        f"[光谱] 已忽略点数不足的 CSV 候选：csv={result.get('csv_path')}, "
                        f"points={len(y2)}, min_required={min_points}"
                    )

            # 3) 拟合数据也必须满足点数要求才作为曲线；fit_peak 标量仍可单独使用。
            for key in ("fit_y", "FIT_Y", "fit_values"):
                if key in result:
                    _, fy = self._extract_spectrum_arrays_from_any(result.get(key))
                    if _accept_y(fy):
                        fit_values = fy
                    elif fy:
                        self.log(
                            f"[光谱] 已忽略非完整拟合曲线候选：source={key}, "
                            f"points={len(fy)}, min_required={min_points}"
                        )
                    break

        else:
            _try_set_from_value(result, "non_dict_result")

        raw_values = raw_values or []
        fit_values = fit_values or []

        threshold_values = self._remove_above_threshold(raw_values, self.cfg.raw_remove_above)
        median_values = self._median_filter_1d(threshold_values, self.cfg.median_filter_window)

        raw_original_peak = max(raw_values) if raw_values else None
        raw_filtered_peak = max(threshold_values) if threshold_values else None
        raw_median_peak = max(median_values) if median_values else None

        # raw_peak 用处理后的结果；如果处理后没有点，则退回阈值后/原始峰值
        raw_peak = raw_median_peak
        if raw_peak is None:
            raw_peak = raw_filtered_peak
        if raw_peak is None:
            raw_peak = raw_original_peak

        fit_peak = result.get("fit_peak") if isinstance(result, dict) else None
        fit_peak = self._numeric_from_any(fit_peak) if fit_peak is not None else None
        if fit_peak is None and fit_values:
            fit_peak = max(fit_values)
        if fit_peak is None:
            fit_peak = raw_peak

        return {
            "wavelength": wavelength,
            "raw_values": raw_values,
            "threshold_filtered_intensity": threshold_values,
            "median_filtered_intensity": median_values,
            "raw_original_peak": raw_original_peak,
            "raw_filtered_peak": raw_filtered_peak,
            "raw_median_peak": raw_median_peak,
            "raw_peak": raw_peak,
            "fit_values": fit_values,
            "fit_peak": fit_peak,
            "fit_params": result.get("fit_params") if isinstance(result, dict) else None,
        }


    def _load_spectrum_from_csv(self, csv_path: str) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        """
        从 LabVIEW 保存的 CSV 中读取 wavelength 与 intensity。

        兼容格式：
            1) 无表头：一列 intensity，或两列 wavelength,intensity；
            2) 有表头：wavelength/lambda/x + raw_y/intensity/values/y；
            3) 多列数据：优先找 raw_y/intensity/values，否则取最后一个数值列作为 intensity。
        """
        path = Path(csv_path)

        if not path.exists():
            self.log(f"[光谱] CSV 不存在，无法读取：{csv_path}")
            return None, None

        rows: List[List[str]] = []
        for enc in ("utf-8-sig", "gbk", "utf-8"):
            try:
                with path.open("r", encoding=enc, newline="") as f:
                    rows = [row for row in csv.reader(f) if row]
                break
            except UnicodeDecodeError:
                continue

        if not rows:
            self.log(f"[光谱] CSV 为空：{csv_path}")
            return None, None

        def _try_float(v: Any) -> Optional[float]:
            try:
                s = str(v).strip()
                if s == "":
                    return None
                fv = float(s)
                return fv if math.isfinite(fv) else None
            except Exception:
                return None

        def _norm_header(s: Any) -> str:
            return str(s).strip().lower().replace(" ", "").replace("-", "_")

        first_numeric = [_try_float(x) for x in rows[0]]
        has_header = not any(v is not None for v in first_numeric)

        wavelength: List[float] = []
        intensity: List[float] = []

        if has_header:
            headers = [_norm_header(x) for x in rows[0]]
            data_rows = rows[1:]

            x_names = {"wavelength", "lambda", "x", "x_axis", "center_wavelength", "波长", "中心波长"}
            y_names = {"raw_y", "raw", "intensity", "values", "value", "data", "y", "spectrum", "光强", "强度"}

            x_idx = next((i for i, h in enumerate(headers) if h in x_names or "wavelength" in h), None)
            y_idx = next((i for i, h in enumerate(headers) if h in y_names or "raw_y" in h or "intensity" in h), None)

            for row in data_rows:
                nums = [_try_float(v) for v in row]
                if y_idx is None:
                    numeric_indices = [i for i, v in enumerate(nums) if v is not None]
                    if not numeric_indices:
                        continue
                    y_i = numeric_indices[-1]
                else:
                    y_i = y_idx

                yv = nums[y_i] if y_i < len(nums) else None
                if yv is None:
                    continue
                intensity.append(float(yv))

                if x_idx is not None and x_idx < len(nums) and nums[x_idx] is not None:
                    wavelength.append(float(nums[x_idx]))
        else:
            for row in rows:
                nums = [_try_float(v) for v in row]
                numeric = [v for v in nums if v is not None]
                if not numeric:
                    continue
                if len(numeric) >= 2:
                    wavelength.append(float(numeric[0]))
                    intensity.append(float(numeric[1]))
                else:
                    intensity.append(float(numeric[0]))

        if len(wavelength) == 0:
            wavelength_out = None
        else:
            wavelength_out = wavelength
        if len(intensity) == 0:
            intensity_out = None
        else:
            intensity_out = intensity

        self.log(
            f"[光谱] 已从CSV读取：{csv_path}；"
            f"wavelength点数={0 if wavelength_out is None else len(wavelength_out)}；"
            f"intensity点数={0 if intensity_out is None else len(intensity_out)}"
        )
        return wavelength_out, intensity_out

    def begin_new_run_session(self):
        """
        每次点击“运行完整循环测量”时调用一次。

        作用：
            1. 创建本次点击对应的总文件夹，文件夹名用点击时间；
            2. 清空本次 GUI 右侧“角度-拟合峰值列表”的历史点；
            3. 重置上一轮角度，使第一轮不进行 Step 11 自适应调节；
            4. 重置补焦 ROI 与参考图状态，确保每次运行都重新选择补焦基准图 ROI。
        """
        now = datetime.now()
        date_dir = now.strftime("%m.%d")
        self.run_session_name = now.strftime("%Y%m%d_%H%M%S")
        self.run_session_dir = self.output_root / "save" / date_dir / self.run_session_name
        self.run_session_dir.mkdir(parents=True, exist_ok=True)

        self.previous_cycle_angle = None
        self.context["previous_cycle_angle"] = None
        self.context["angle_delta"] = None
        self.context["angle_before_after_delta"] = None
        self.context["save_angle_deg"] = None

        self.plot_points.clear()
        self.context["plot_points"] = self.plot_points

        self.context["current_paths"] = {
            "run_session_dir": str(self.run_session_dir),
            "run_session_name": self.run_session_name,
        }

        # 每次新运行都重新选择补焦基准图 ROI 并更新参考图
        self._focus_roi_selected = False
        self._focus_reference_ready = False
        self._focus_reference_image = None
        self.log(
            "[聚焦参考] 新运行会话开始，已重置补焦 ROI 与参考图状态，"
            "将在 Step 1.5 重新选择补焦基准图 ROI"
        )

        self.log(f"[运行批次] 新建本次测量文件夹：{self.run_session_dir}")
        self.notify_update()

    def build_save_path(self, cycle_index: int) -> Dict[str, str]:
        self.log("========== 生成最终存储路径 ==========")

        if self.run_session_dir is None or self.run_session_name is None:
            # 兼容直接调用 run_one_cycle 的情况。
            self.begin_new_run_session()

        assert self.run_session_dir is not None
        assert self.run_session_name is not None

        timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]
        spectrum_filename = f"cycle_{cycle_index:04d}_{timestamp}_spectrum.csv"

        paths = {
            "run_session_dir": str(self.run_session_dir),
            "run_session_name": self.run_session_name,
            "spectrum_filename": spectrum_filename,
            "spectrum_csv": str(self.run_session_dir / spectrum_filename),
        }

        self.context["current_paths"] = paths
        self.log(f"[保存路径] run_session_dir={paths['run_session_dir']}")
        self.log(f"[保存路径] spectrum_csv={paths['spectrum_csv']}")
        self.notify_update()

        return paths

    def _make_save_context_snapshot(self) -> Dict[str, Any]:
        """
        为异步保存创建轻量级 context 快照。

        注意：这里不能 deepcopy 整个 self.context。
        原来的全量 deepcopy 会把 plot_points、last_tcp_result、历史光谱数组等内容全部复制，
        导致 Step 12 在“启动保存线程之前”就被卡住数秒。

        这里只复制本轮保存真正需要的字段：
            1. 角度结果摘要；
            2. 本轮光谱数组；
            3. 峰值和滤波参数；
            4. 本轮保存路径。
        这样 Step 12 只需要很短时间即可提交保存线程，然后主测量流程可以继续进入下一轮。
        """
        ctx = self.context

        def _copy_float_list(value: Any) -> List[float]:
            if value is None:
                return []
            try:
                return [float(v) for v in value if v is not None]
            except Exception:
                return []

        return {
            "cycle_index": ctx.get("cycle_index"),

            "angle_before": ctx.get("angle_before"),
            "angle_after": ctx.get("angle_after"),
            "save_angle_deg": ctx.get("save_angle_deg"),
            "angle_delta": ctx.get("angle_delta"),

            "signal_on_time_used_ms": ctx.get("signal_on_time_used_ms"),
            "signal_on_time_next_ms": ctx.get("signal_on_time_next_ms"),
            "signal_time_adjust_action": ctx.get("signal_time_adjust_action"),

            "wavelength": _copy_float_list(ctx.get("wavelength")),
            "raw_values": _copy_float_list(ctx.get("raw_values")),
            "raw_filtered_values": _copy_float_list(ctx.get("raw_filtered_values")),
            "raw_median_values": _copy_float_list(ctx.get("raw_median_values")),
            "fit_values": _copy_float_list(ctx.get("fit_values")),
            "x_axis_values": _copy_float_list(ctx.get("x_axis_values")),

            # 保存 Step 4 已经解析出的真实 TCP 光谱峰值。
            # 只有 SPECTROMETER_TCP_DISABLED=True 的脱机模式才会在 request_labview_spectrum()
            # 中把这些字段设为 DUMMY_SPECTRUM_PEAK=1000。
            "raw_original_peak": ctx.get("raw_original_peak"),
            "raw_filtered_peak": ctx.get("raw_filtered_peak"),
            "raw_median_peak": ctx.get("raw_median_peak"),
            "raw_peak": ctx.get("raw_peak"),
            "fit_peak": ctx.get("fit_peak"),
            "fit_params": self._json_safe(ctx.get("fit_params")),

            "raw_filter_threshold": ctx.get("raw_filter_threshold"),
            "median_filter_window": ctx.get("median_filter_window"),
            "x_axis_xlsx_path": ctx.get("x_axis_xlsx_path"),
            "labview_csv_path": (ctx.get("last_tcp_result") or {}).get("csv_path") if isinstance(ctx.get("last_tcp_result"), dict) else "",

            "current_paths": dict(ctx.get("current_paths") or {}),
        }

    @staticmethod
    def _make_angle_result_snapshot(result: Dict[str, Any]) -> Dict[str, Any]:
        """
        为异步保存创建角度检测结果快照。

        不复制可能很大的中间对象，只保留保存和排查需要的关键字段。
        """
        if not isinstance(result, dict):
            return {"ok": False, "reason": "angle_result_not_dict", "raw": str(result)}

        keep_keys = [
            "ok",
            "angle_deg",
            "angle_deg_raw",
            "reason",
            "image_path",
            "capture_area",
            "timestamp",
            "center",
            "center_norm",
            "bbox",
            "bbox_xywhn",
            "area_px",
            "area",
            "stats",
        ]
        return {k: MeasurementWorkflow._json_safe(result.get(k)) for k in keep_keys if k in result}

    @staticmethod
    def _make_labview_result_snapshot(result: Dict[str, Any]) -> Dict[str, Any]:
        """
        为异步保存创建 LabVIEW 返回结果快照。

        不复制 raw_data_text 或 values 等长字段，因为本轮光谱数组已经在 context_snapshot 中保存。
        """
        if not isinstance(result, dict):
            return {"ok": False, "raw": str(result)}

        keep_keys = [
            "ok",
            "index",
            "num_points",
            "csv_path",
            "message",
            "error",
            "method",
            "raw_original_peak",
            "raw_filtered_peak",
            "raw_median_peak",
            "raw_peak",
            "fit_peak",
        ]
        return {k: MeasurementWorkflow._json_safe(result.get(k)) for k in keep_keys if k in result}

    def start_save_cycle_result_async(
        self,
        cycle_index: int,
        paths: Dict[str, str],
        angle_before_result: Dict[str, Any],
        angle_after_result: Dict[str, Any],
        labview_result: Dict[str, Any],
        append_plot_point: bool = True,
    ) -> threading.Thread:
        """
        启动独立线程保存本轮数据。

        该函数只负责创建轻量快照并提交保存任务，立即返回；
        真正的 CSV/JSON/XLSX 写入在后台线程中完成。
        """
        t0 = time.time()

        context_snapshot = self._make_save_context_snapshot()
        paths_snapshot = dict(paths)
        angle_before_snapshot = self._make_angle_result_snapshot(angle_before_result)
        angle_after_snapshot = self._make_angle_result_snapshot(angle_after_result)
        labview_snapshot = self._make_labview_result_snapshot(labview_result)

        t = threading.Thread(
            target=self._save_cycle_result_thread_entry,
            args=(
                cycle_index,
                paths_snapshot,
                angle_before_snapshot,
                angle_after_snapshot,
                labview_snapshot,
                context_snapshot,
                append_plot_point,
            ),
            name=f"save-cycle-{cycle_index:04d}",
            daemon=False,
        )

        with self.save_threads_lock:
            self.save_threads.append(t)

        t.start()
        self.log(
            f"[异步保存] 第 {cycle_index} 轮保存线程已启动：{t.name}；"
            f"提交准备耗时={time.time() - t0:.3f} s"
        )
        return t

    def _save_cycle_result_thread_entry(
        self,
        cycle_index: int,
        paths: Dict[str, str],
        angle_before_result: Dict[str, Any],
        angle_after_result: Dict[str, Any],
        labview_result: Dict[str, Any],
        context_snapshot: Dict[str, Any],
        append_plot_point: bool = True,
    ):
        current = threading.current_thread()
        try:
            self.save_cycle_result(
                cycle_index=cycle_index,
                paths=paths,
                angle_before_result=angle_before_result,
                angle_after_result=angle_after_result,
                labview_result=labview_result,
                context_snapshot=context_snapshot,
                append_plot_point=append_plot_point,
            )
            self.log(f"[异步保存] 第 {cycle_index} 轮保存完成")
        except Exception as e:
            self.log(f"[异步保存错误] 第 {cycle_index} 轮保存失败：{e}")
            self.log(traceback.format_exc())

            try:
                error_dir = self.output_root / "errors"
                error_dir.mkdir(parents=True, exist_ok=True)
                error_path = error_dir / (
                    f"save_error_cycle_{cycle_index:04d}_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                error_info = {
                    "cycle_index": cycle_index,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "paths": self._json_safe(paths),
                    "context_snapshot": self._json_safe(context_snapshot),
                }
                with error_path.open("w", encoding="utf-8") as f:
                    json.dump(error_info, f, ensure_ascii=False, indent=2)
                self.log(f"[异步保存错误] 错误信息已保存：{error_path}")
            except Exception as ee:
                self.log(f"[异步保存错误] 保存错误日志失败：{ee}")
        finally:
            with self.save_threads_lock:
                self.save_threads = [t for t in self.save_threads if t is not current]

    def wait_for_pending_saves(self, timeout_per_join: Optional[float] = None):
        """
        等待所有后台保存线程结束。

        在停止测量、流程结束、关闭 GUI 前调用，确保已经提交的保存任务落盘，
        但不会影响每一轮 Step 12 提交后立即进入下一轮的速度。
        """
        while True:
            with self.save_threads_lock:
                threads = [
                    t for t in self.save_threads
                    if t is not threading.current_thread() and t.is_alive()
                ]

            if not threads:
                return

            self.log(f"[异步保存] 等待 {len(threads)} 个保存线程完成...")
            for t in threads:
                t.join(timeout=timeout_per_join)

    def save_cycle_result(
        self,
        cycle_index: int,
        paths: Dict[str, str],
        angle_before_result: Dict[str, Any],
        angle_after_result: Dict[str, Any],
        labview_result: Dict[str, Any],
        context_snapshot: Optional[Dict[str, Any]] = None,
        append_plot_point: bool = True,
    ):
        self.log("========== 保存本轮测量结果 ==========")

        ctx = context_snapshot if context_snapshot is not None else self.context

        with self.save_io_lock:
            # 每轮保存的 CSV 中，wavelength 列优先写入 516中心波长.xlsx
            # 读取到的中心波长序列；如果该文件没有读到有效数值，再退回
            # LabVIEW/TCP 返回的 wavelength。
            wavelength = ctx.get("x_axis_values") or ctx.get("wavelength")
            raw_values = ctx.get("raw_values") or []
            filtered_values = ctx.get("raw_filtered_values") or []
            median_values = ctx.get("raw_median_values") or []

            # 每轮光谱数据只保存为 CSV，不再保存 JSON，也不再为每轮创建子文件夹。
            if raw_values or filtered_values or median_values:
                # 本轮保存角度：固定使用 Step1 最终角度 save_angle_deg。
                # 只有旧数据/旧调用没有 save_angle_deg 时，才退回 angle_before。
                angle_value = ctx.get("save_angle_deg")
                if angle_value is None:
                    angle_value = ctx.get("angle_before")

                max_len = max(
                    len(raw_values),
                    len(filtered_values),
                    len(median_values),
                )
                with open(paths["spectrum_csv"], "w", encoding="utf-8-sig", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "index",
                        "wavelength",
                        "raw_y",
                        "threshold_filtered_y",
                        "median_filtered_y",
                        "angle_deg",
                    ])
                    for i in range(max_len):
                        x = wavelength[i] if wavelength is not None and i < len(wavelength) else ""
                        raw_y = raw_values[i] if i < len(raw_values) else ""
                        filt_y = filtered_values[i] if i < len(filtered_values) else ""
                        med_y = median_values[i] if i < len(median_values) else ""
                        writer.writerow([i, x, raw_y, filt_y, med_y, angle_value])
                self.log(
                    f"[保存] 光谱 CSV：{paths['spectrum_csv']}；"
                    f"wavelength列已写入中心波长；已写入本轮角度 angle_deg={angle_value}"
                )
            else:
                self.log(
                    "[保存] 本轮没有可保存的光谱数组：不会再用1000伪造光谱；"
                    "请检查 Step 4 的 TCP 返回是否包含 raw_y/intensity/values 或有效 csv_path"
                )
            
            # 保存当前Summary实时数据
            self._append_summary_csv(
                cycle_index=cycle_index,
                angle_before_result=angle_before_result,
                angle_after_result=angle_after_result,
                labview_result=labview_result,
                paths=paths,
                context_snapshot=ctx,
            )

            self._append_summary_xlsx(
                cycle_index=cycle_index,
                angle_before_result=angle_before_result,
                angle_after_result=angle_after_result,
                labview_result=labview_result,
                paths=paths,
                context_snapshot=ctx,
            )

            if append_plot_point:
                self._append_plot_point(cycle_index, context_snapshot=ctx)

        self.notify_update()

    def _append_summary_csv(
        self,
        cycle_index: int,
        angle_before_result: Dict[str, Any],
        angle_after_result: Dict[str, Any],
        labview_result: Dict[str, Any],
        paths: Dict[str, str],
        context_snapshot: Optional[Dict[str, Any]] = None,
    ):
        ctx = context_snapshot if context_snapshot is not None else self.context
        file_exists = self.summary_csv_path.exists()

        with self.summary_csv_path.open("a", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow([
                    "cycle_index",
                    "time",
                    "angle_before",
                    "angle_after",
                    "angle_before_after_delta",
                    "previous_cycle_angle",
                    "angle_delta_vs_previous_cycle",
                    "signal_on_time_used_ms",
                    "signal_on_time_next_ms",
                    "signal_time_adjust_action",
                    "angle_before_center",
                    "angle_after_center",
                    "raw_original_peak",
                    "raw_peak",
                    "fit_peak",
                    "run_session_name",
                    "spectrum_filename",
                    "labview_csv_path",
                    "spectrum_csv_path",
                ])

            writer.writerow([
                cycle_index,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ctx.get("angle_before"),
                ctx.get("angle_after"),
                ctx.get("angle_before_after_delta"),
                ctx.get("previous_cycle_angle"),
                ctx.get("angle_delta"),
                ctx.get("signal_on_time_used_ms"),
                ctx.get("signal_on_time_next_ms"),
                ctx.get("signal_time_adjust_action"),
                angle_before_result.get("center"),
                angle_after_result.get("center"),
                ctx.get("raw_original_peak"),
                ctx.get("raw_peak"),
                ctx.get("fit_peak"),
                paths.get("run_session_name"),
                paths.get("spectrum_filename"),
                labview_result.get("csv_path") or ctx.get("labview_csv_path") or "",
                paths.get("spectrum_csv"),
            ])

        self.log(f"[保存] Summary CSV 已更新：{self.summary_csv_path}")

    def _append_plot_point(self, cycle_index: int, context_snapshot: Optional[Dict[str, Any]] = None):
        """
        记录用于 GUI 实时绘图/右侧列表的数据点。

        异步保存时使用 context_snapshot，确保该数据点属于当前 cycle，
        不会被下一轮 self.context 覆盖。
        """
        ctx = context_snapshot if context_snapshot is not None else self.context

        angle_value = ctx.get("save_angle_deg")
        if angle_value is None:
            angle_value = ctx.get("angle_before")

        # 右侧列表/绘图使用 Step 4 真实 TCP 解析得到的 fit_peak。
        # 只有脱机模式下 request_labview_spectrum() 才会把 fit_peak 设为 1000。
        fit_peak = ctx.get("fit_peak")

        point = {
            "cycle_index": int(cycle_index),
            "angle_deg": angle_value,
            "fit_peak": fit_peak,
            "raw_values": list(ctx.get("raw_values") or []),
            "raw_median_values": list(ctx.get("raw_median_values") or []),
            "x_axis_values": list(ctx.get("x_axis_values") or []),
        }

        self.plot_points.append(point)
        self.context["plot_points"] = self.plot_points

        self.log(
            f"[绘图] 已追加数据点：cycle={cycle_index}, "
            f"angle={angle_value}, fit_peak={fit_peak}"
        )

    # --------------------------------------------------------
    # Excel 汇总文件：每轮追加一行
    # --------------------------------------------------------

    def _ensure_summary_xlsx(self):
        """
        创建本次运行的 xlsx 汇总文件。

        该文件在 workflow 初始化时就生成，后续每一轮测量完成后，
        通过 _append_summary_xlsx() 追加一行。
        """
        self.summary_date_dir.mkdir(parents=True, exist_ok=True)

        if self.summary_xlsx_path.exists():
            return

        wb = Workbook()
        ws = wb.active
        ws.title = "测量汇总"

        ws.append(self.summary_xlsx_headers)

        header_fill = PatternFill("solid", fgColor="1F4E78")
        header_font = Font(bold=True, color="FFFFFF")
        center_alignment = Alignment(horizontal="center", vertical="center")
        thin_side = Side(style="thin", color="D9E2F3")
        border = Border(left=thin_side, right=thin_side, top=thin_side, bottom=thin_side)

        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = center_alignment
            cell.border = border

        ws.freeze_panes = "A2"
        ws.column_dimensions["A"].width = 20
        ws.column_dimensions["B"].width = 16
        ws.column_dimensions["C"].width = 20
        ws.column_dimensions["D"].width = 20
        ws.column_dimensions["E"].width = 34
        ws.column_dimensions["F"].width = 24

        wb.save(self.summary_xlsx_path)
        self.log(f"[保存] Excel汇总文件已创建：{self.summary_xlsx_path}")

    def _append_summary_xlsx(
        self,
        cycle_index: int,
        angle_before_result: Dict[str, Any],
        angle_after_result: Dict[str, Any],
        labview_result: Dict[str, Any],
        paths: Dict[str, str],
        context_snapshot: Optional[Dict[str, Any]] = None,
    ):
        """
        向 Excel 汇总文件追加一轮结果。
        表头：
            序号、本次角度、本次原始数据峰值、拟合数据峰值、文件名、信号发生器打开时间。
        """
        ctx = context_snapshot if context_snapshot is not None else self.context
        self._ensure_summary_xlsx()

        # “本次角度”采用 Step1 最终角度 save_angle_deg；旧快照无该字段时退回 angle_before。
        angle_value = ctx.get("save_angle_deg")
        if angle_value is None:
            angle_value = ctx.get("angle_before")

        row = [
            int(cycle_index),
            angle_value,
            ctx.get("raw_original_peak"),
            ctx.get("fit_peak"),
            paths.get("spectrum_filename") or Path(paths.get("spectrum_csv", "")).name,
            ctx.get("signal_on_time_used_ms"),
        ]

        wb = load_workbook(self.summary_xlsx_path)
        ws = wb["测量汇总"] if "测量汇总" in wb.sheetnames else wb.active

        # 如果旧文件已经存在但表头还是 5 列，则自动升级表头。
        expected_headers = self.summary_xlsx_headers
        current_headers = [ws.cell(row=1, column=i).value for i in range(1, len(expected_headers) + 1)]
        if current_headers != expected_headers:
            for i, header in enumerate(expected_headers, start=1):
                ws.cell(row=1, column=i, value=header)

        ws.append(row)

        row_idx = ws.max_row
        center_alignment = Alignment(horizontal="center", vertical="center")
        thin_side = Side(style="thin", color="D9E2F3")
        border = Border(left=thin_side, right=thin_side, top=thin_side, bottom=thin_side)

        for col_idx in range(1, 7):
            cell = ws.cell(row=row_idx, column=col_idx)
            cell.alignment = center_alignment
            cell.border = border

        ws.cell(row=row_idx, column=1).number_format = "0"
        ws.cell(row=row_idx, column=2).number_format = "0.000000"
        ws.cell(row=row_idx, column=3).number_format = "0.000000"
        ws.cell(row=row_idx, column=4).number_format = "0.000000"
        ws.cell(row=row_idx, column=6).number_format = "0.000"

        wb.save(self.summary_xlsx_path)
        self.log(f"[保存] Excel汇总已追加第 {cycle_index} 轮：{self.summary_xlsx_path}")

    def save_summary_xlsx(self, wait_pending: bool = True):
        """
        测量结束或停止时再确认保存一次 Excel 汇总文件。

        wait_pending=True 时会先等待后台保存线程完成，防止程序关闭时
        还有 CSV/JSON/XLSX 未写完。
        """
        try:
            if wait_pending:
                self.wait_for_pending_saves()
            with self.save_io_lock:
                self._ensure_summary_xlsx()
            self.log(f"[保存] Excel汇总文件已保存：{self.summary_xlsx_path}")
        except Exception as e:
            self.log(f"[保存] Excel汇总文件保存失败：{e}")

    @staticmethod
    def _json_safe(obj: Any) -> Any:
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, dict):
            return {str(k): MeasurementWorkflow._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [MeasurementWorkflow._json_safe(v) for v in obj]
        return str(obj)

    # --------------------------------------------------------
    # Δw 判断
    # --------------------------------------------------------

    def judge_delta_w(self) -> bool:
        self.log("========== 判断 Δw 是否继续 ==========")

        if not self.cfg.enable_delta_w_judge:
            self.log("[Δw] 未启用 Δw 判断，默认继续")
            return True

        if SPECTROMETER_TCP_DISABLED:
            self.context["w0"] = DUMMY_SPECTRUM_PEAK
            self.context["w"] = DUMMY_SPECTRUM_PEAK
            self.context["delta_w"] = 0.0
            self.context["valid_change"] = True
            self.log("[Δw] 光谱仪已禁用，峰值固定为1000；跳过Δw停止判断，默认继续")
            self.notify_update()
            return True

        fit_peak = self.context.get("fit_peak")

        if fit_peak is None:
            self.log("[Δw] fit_peak 为空，无法判断")
            return not self.cfg.stop_when_delta_w_not_enough

        if self.context.get("w0") is None:
            self.context["w0"] = fit_peak
            self.context["w"] = fit_peak
            self.context["delta_w"] = 0.0
            self.context["valid_change"] = True
            self.log(f"[Δw] 初始化 w0={fit_peak}，默认继续")
            self.notify_update()
            return True

        w0 = float(self.context["w0"])
        w = float(fit_peak)
        delta_w = abs(w - w0)

        self.context["w"] = w
        self.context["delta_w"] = delta_w
        self.context["valid_change"] = delta_w > self.cfg.delta_w_threshold

        self.log(f"[Δw] w0={w0}, w={w}, delta_w={delta_w}, threshold={self.cfg.delta_w_threshold}")
        self.log(f"[Δw] valid_change={self.context['valid_change']}")

        self.notify_update()

        if not self.context["valid_change"] and self.cfg.stop_when_delta_w_not_enough:
            self.log("[Δw] 变化量不足，停止循环")
            return False

        return True

    # --------------------------------------------------------
    # Bmask/角度失败非致命判断与本轮跳过处理
    # --------------------------------------------------------

    def _is_nonfatal_bmask_angle_failure_reason(self, reason: Any) -> bool:
        """
        判断 Step1/Step7 中的 Bmask/角度检测失败是否只应跳过本轮，而不是终止完整测量。

        设计原则：
            1. Bmask 为空、面积太小、轮廓失败、最长边失败、feature tracker 当前帧失败：非致命；
            2. 未达到角度阈值但没有硬件/用户停止异常：非致命；
            3. 用户主动停止、中途重标定、硬件运动异常、LabVIEW/光谱异常：仍按原逻辑处理。
        """
        text = str(reason or "").lower()
        if not text:
            return False

        fatal_keys = (
            "user_stop",
            "stop_requested",
            "manual_stop",
            "midrun_recalibration",
            "controller_error",
            "stage_error",
            "stage_exception",
            "hardware",
            "laser_off_failed",
            "tcp",
            "labview",
            "spectrum",
        )
        if any(k in text for k in fatal_keys):
            return False

        nonfatal_keys = (
            "bmask",
            "b_mask",
            "b mask",
            "longest_edge",
            "longest edge",
            "angle_invalid",
            "angle failed",
            "angle_failed",
            "angle_monitor_exception",
            "feature_tracker",
            "track fail",
            "track_failed",
            "no_external_contour",
            "invalid_largest_contour",
            "area_too_small",
            "bmask_empty",
            "minarearect_no_valid_edge",
            "delta_below_target_continue",
            "not_reached",
            "realtime_max_duration_reached",
        )
        return any(k in text for k in nonfatal_keys)

    def _skip_current_cycle_without_stopping_measurement(self, cycle_index: int, phase: str, reason: Any, close_laser_if_on: bool = False) -> bool:
        """
        Bmask/角度检测失败时跳过当前 cycle，但不把完整测量置为失败。

        返回 True 是为了让外层完整循环继续进入下一轮。
        close_laser_if_on=True 只关闭当前 Step6 打开的激光，避免 Step7 失败后激光悬空；
        不调用 finish()，不 close_all_devices()，不关闭 TCP，不把 stop_requested 置 True。
        """
        self.context["last_skipped_cycle_index"] = int(cycle_index)
        self.context["last_skipped_cycle_phase"] = str(phase)
        self.context["last_skipped_cycle_reason"] = str(reason or "")
        self.context["rule_ab_failed_stop"] = False
        self.context["rule_ab_failed_reason"] = str(reason or "")

        self.log(
            f"[完整测量-跳过本轮] 第 {cycle_index} 轮在 {phase} 发生 Bmask/角度检测失败；"
            f"不结束完整循环、不关闭全部设备，进入下一轮。reason={reason}"
        )

        if close_laser_if_on:
            try:
                if bool(self.context.get("laser_on", False)):
                    self.laser_off()
                    self.log("[完整测量-跳过本轮] Step7 未成功完成，已仅关闭当前激光；未关闭其它设备/TCP。")
            except Exception as e:
                # 激光关闭失败属于安全相关异常，不能继续循环。
                self.log(f"[完整测量-跳过本轮] 关闭当前激光失败，出于安全停止完整测量：{e}")
                self.context["laser_off_failed_after_step7"] = True
                self.context["laser_off_failed_reason"] = str(e)
                self.stop_requested = True
                self.notify_update()
                return False

        self.notify_update()
        return True

    # --------------------------------------------------------
    # 聚焦参考图与补焦（完整循环测量）
    # --------------------------------------------------------

    def _make_focus_config(self) -> AutofocusConfig:
        """根据 GUI 当前值构造完整循环测量使用的 AutofocusConfig。

        使用 saf_capture_area / saf_focus_roi，与标定/角度检测的 capture_area 完全解耦。
        当通过全屏 ROI 选择建立参考时，saf_capture_area 通常为 (0, 0, screen_w, screen_h)，
        saf_focus_roi 为相对于全屏的 ROI。
        """
        capture_area = tuple(
            int(v) for v in self._parse_focus_roi_text(self.cfg.saf_capture_area)
        )
        focus_roi = tuple(
            int(v) for v in self._parse_focus_roi_text(self.cfg.saf_focus_roi)
        )
        return AutofocusConfig(
            capture_mode="screen_region",
            capture_area=capture_area,
            focus_roi=focus_roi,
            autofocus_enabled=True,
            autofocus_focus_trigger_ratio=float(self.cfg.focus_trigger_ratio),
            autofocus_stop_ratio=float(self.cfg.focus_stop_ratio),
            autofocus_focus_trigger_count=int(self.cfg.focus_trigger_count),
            autofocus_trigger_absolute=bool(self.cfg.focus_trigger_absolute),
            autofocus_detection_only=bool(self.cfg.focus_detection_only),
            z_enabled=bool(self.cfg.focus_z_enabled),
            z_axis=int(self.cfg.focus_z_axis),
            z_speed=int(self.cfg.focus_z_speed),
            z_accel=int(self.cfg.focus_z_accel),
            z_search_strategy=str(self.cfg.focus_search_strategy),
            z_search_steps=int(getattr(self.cfg, "focus_z_search_steps", 10)),
            z_patience=int(getattr(self.cfg, "focus_z_patience", 3)),
            z_direction_probe_steps=tuple(
                int(v) for v in getattr(self.cfg, "focus_z_direction_probe_steps", (10, 20, 30))
            ),
            z_direction_probe_stage_count=int(
                getattr(self.cfg, "focus_z_direction_probe_stage_count", 3)
            ),
            z_direction_probe_step_interval=int(
                getattr(self.cfg, "focus_z_direction_probe_step_interval", 10)
            ),
            z_direction_probe_samples=int(getattr(self.cfg, "focus_z_direction_probe_samples", 3)),
            z_direction_probe_points_per_step=int(
                getattr(self.cfg, "focus_z_direction_probe_points_per_step", 5)
            ),
            z_local_refine_enabled=bool(getattr(self.cfg, "focus_z_local_refine_enabled", True)),
            z_local_refine_decay=float(getattr(self.cfg, "focus_z_local_refine_decay", 0.5)),
            z_local_refine_min_step=int(getattr(self.cfg, "focus_z_local_refine_min_step", 1)),
            z_local_refine_max_rounds=int(getattr(self.cfg, "focus_z_local_refine_max_rounds", 4)),
        )

    def _ensure_focus_components(self) -> None:
        """延迟初始化 Focus 评分/控制组件。"""
        if self._focus_metrics_calc is None:
            cfg = self._make_focus_config()
            self._focus_metrics_calc = FocusMetricsCalculator(cfg)
        if self._focus_scorer is None:
            self._focus_scorer = FocusScorer(
                self._focus_metrics_calc.cfg, self._focus_metrics_calc
            )

    def _capture_current_focus_frame(self) -> Optional[np.ndarray]:
        """截取当前 SAF 补焦截图区域画面（RGB），用于聚焦参考或评分。使用 SAF 独立截图区域。"""
        try:
            self._ensure_focus_components()
            result = self._focus_metrics_calc.capture_live()
            image_rgb = result.get("full_rgb")
            if image_rgb is not None and image_rgb.size > 0:
                return image_rgb
        except Exception as e:
            self.log(f"[聚焦] 截图失败：{e}")
        return None

    def _capture_full_screen_frame(self) -> Optional[np.ndarray]:
        """截取整个屏幕画面（RGB），用于补焦基准图 ROI 交互式选择。

        返回的图像尺寸即为屏幕分辨率；后续 saf_capture_area 应同步为 (0, 0, W, H)，
        saf_focus_roi 为相对于全屏的 ROI，从而与 capture_area / focus_roi 完全解耦。
        """
        try:
            if FixedRegionScreenCapture is not None:
                # region=None 时 pyautogui.screenshot 截取全屏
                frame = pyautogui.screenshot(region=None)
                image_rgb = np.array(frame)
                return image_rgb
        except Exception as e:
            self.log(f"[聚焦] 全屏截图失败（pyautogui）：{e}")

        try:
            from PIL import ImageGrab

            img = ImageGrab.grab().convert("RGB")
            return np.array(img)
        except Exception as e:
            self.log(f"[聚焦] 全屏截图失败（PIL）：{e}")

        return None

    def _select_focus_roi_interactively(self) -> bool:
        """
        当聚焦参考无法建立时，自动弹出 ROI 选择窗口让用户框选补焦区域。

        使用整个屏幕采集一帧作为底图（不再局限于 capture_area），在 OpenCV 窗口中
        拖拽矩形，Enter/N 确认、R 重置、ESC/Q 取消。选择成功后 ROI 坐标直接相对于
        全屏，self.cfg.saf_capture_area 同步为 (0, 0, screen_w, screen_h)，
        self.cfg.saf_focus_roi 同步为用户选择的 ROI，从而与 capture_area / focus_roi
        完全解耦，不影响标定 ABC、角度检测等画面区域。

        关键修复：
            - 窗口创建后尝试置顶（WND_PROP_TOPMOST）并移到 (0,0)，避免被其他窗口遮挡；
            - 用户确认后通过 cv2.waitKey + destroyWindow + destroyAllWindows 彻底关闭窗口，
              解决 ROI 界面框残留、看似"卡住"的问题；
            - 循环内检查 self.stop_requested，允许外部停止请求中断选择；
            - 对过小选区（<2px）进行拦截，要求重新拖拽。
        """
        if cv2 is None:
            self.log("[聚焦ROI] OpenCV 不可用，无法交互式选择 ROI")
            return False

        try:
            self.log("[聚焦ROI] 即将弹出全屏 ROI 选择窗口；请拖拽矩形框选补焦区域后按 Enter/N 确认")
            output_dir = self.output_root / "focus_roi_select"
            output_dir.mkdir(parents=True, exist_ok=True)

            # 使用全屏截图作为 ROI 选择底图
            image_rgb = self._capture_full_screen_frame()
            if image_rgb is None or image_rgb.size == 0:
                self.log("[聚焦ROI] 全屏截图失败，无法选择 ROI")
                return False

            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            h, w = image_bgr.shape[:2]
            scale = 0.85
            show_w = max(1, int(w * scale))
            show_h = max(1, int(h * scale))
            display = cv2.resize(image_bgr, (show_w, show_h), interpolation=cv2.INTER_AREA)

            rect: List[Tuple[int, int]] = []
            drawing = False

            def redraw() -> np.ndarray:
                canvas = display.copy()
                lines = [
                    "Select Focus ROI on FULL SCREEN: drag rectangle",
                    "Enter/N: confirm | R: reset | ESC/Q: cancel",
                ]
                for i, s in enumerate(lines):
                    cv2.putText(
                        canvas,
                        s,
                        (18, 28 + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.62,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                if len(rect) == 2:
                    x0, y0 = rect[0]
                    x1, y1 = rect[1]
                    cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 255, 0), 2)
                return canvas

            def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
                nonlocal drawing
                if event == cv2.EVENT_LBUTTONDOWN:
                    drawing = True
                    rect.clear()
                    rect.append((x, y))
                    rect.append((x, y))
                elif event == cv2.EVENT_MOUSEMOVE and drawing:
                    if rect:
                        rect[-1] = (x, y)
                elif event == cv2.EVENT_LBUTTONUP:
                    drawing = False
                    if rect:
                        rect[-1] = (x, y)

            window_name = "Select Focus ROI (Full Screen)"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, show_w, show_h)
            try:
                # 让 ROI 选择窗口置顶并获取焦点，避免被其他窗口遮挡导致按键无响应。
                cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
            except Exception:
                pass
            try:
                cv2.moveWindow(window_name, 0, 0)
            except Exception:
                pass
            cv2.setMouseCallback(window_name, on_mouse)

            try:
                while True:
                    if self.stop_requested:
                        self.log("[聚焦ROI] 收到停止请求，取消 ROI 选择")
                        return False
                    cv2.imshow(window_name, redraw())
                    key = cv2.waitKey(30) & 0xFF
                    if key in (27, ord("q"), ord("Q")):
                        self.log("[聚焦ROI] 用户取消了 ROI 选择")
                        return False
                    if key in (ord("r"), ord("R")):
                        rect.clear()
                    if key in (13, 10, ord("n"), ord("N")):
                        if len(rect) != 2:
                            self.log("[聚焦ROI] 请先拖拽选择一个矩形区域")
                            continue
                        x0, y0 = rect[0]
                        x1, y1 = rect[1]
                        if abs(x1 - x0) < 2 or abs(y1 - y0) < 2:
                            self.log("[聚焦ROI] 选区过小，请重新拖拽选择更大区域")
                            continue
                        break
            finally:
                try:
                    # 先刷新一帧让事件循环处理绘制，再彻底销毁窗口，
                    # 避免 ROI 界面框残留在屏幕上导致"卡住"的观感。
                    cv2.imshow(window_name, redraw())
                    cv2.waitKey(100)
                    cv2.destroyWindow(window_name)
                    cv2.waitKey(300)
                    cv2.destroyAllWindows()
                    cv2.waitKey(100)
                except Exception:
                    pass

            x0, y0 = rect[0]
            x1, y1 = rect[1]
            x_min, x_max = sorted((x0, x1))
            y_min, y_max = sorted((y0, y1))

            # 将显示窗口坐标反缩放，得到全屏坐标系下的 ROI
            roi_x = int(x_min / scale)
            roi_y = int(y_min / scale)
            roi_w = int((x_max - x_min) / scale)
            roi_h = int((y_max - y_min) / scale)

            # 裁剪到屏幕边界
            roi_x = max(0, min(roi_x, w - 1))
            roi_y = max(0, min(roi_y, h - 1))
            roi_w = max(1, min(roi_w, w - roi_x))
            roi_h = max(1, min(roi_h, h - roi_y))

            # 更新 SAF 专用区域，使后续聚焦截图与 ROI 都基于全屏
            self.cfg.saf_capture_area = (0, 0, w, h)
            self.cfg.saf_focus_roi = (roi_x, roi_y, roi_w, roi_h)

            # 重置 Focus 组件，强制使用新 ROI 重建
            self._focus_metrics_calc = None
            self._focus_scorer = None
            self._focus_controller = None
            self._focus_simulator = None

            self._focus_roi_selected = True
            self.log(
                f"[聚焦ROI] 全屏 ROI 已选择：({roi_x}, {roi_y}, {roi_w}, {roi_h})，"
                f"saf_capture_area 已同步为全屏 (0, 0, {w}, {h})；"
                f"capture_area / focus_roi 保持不变"
            )
            return True
        except Exception as e:
            self.log(f"[聚焦ROI] 交互式选择 ROI 失败：{e}")
            self.log(traceback.format_exc())
            return False

    def _show_roi_selection_required_dialog(self) -> None:
        """显示必须选择 ROI 的模态对话框，阻塞直到用户确认。"""
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            messagebox.showerror(
                "必须选择 ROI",
                "必须完成“补焦基准图”的 ROI 选择才能继续测量。\n"
                "请在弹出的 ROI 选择窗口中框选区域并按 Enter 确认。",
                parent=root,
            )
            root.destroy()
        except Exception as e:
            self.log(f"[聚焦参考] ROI 必须选择提示框失败：{e}")

    def capture_focus_reference(self, cycle_index: int) -> bool:
        """
        在第一轮循环照明光 OFF 前，截取当前 ROI 画面并建立聚焦参考图。

        强制要求：只有用户完成“补焦基准图”的 ROI 交互式选择后，才能继续下一步；
        否则循环等待并反复提示，避免流程因 ROI 未选而卡死或继续执行。

        参考图建立成功后直接保存到文件，不再弹出 "Focus Reference Baseline Image"
        实时预览窗口，避免 OpenCV 窗口事件循环导致的"未响应"问题。
        """
        self.log("========== Step 1.5：建立聚焦参考图 ==========")
        try:
            self._ensure_focus_components()

            # 强制完成 ROI 选择，用户取消则阻塞并再次提示
            while not self._focus_roi_selected:
                if self.stop_requested:
                    self.log("[聚焦参考] 收到停止请求，取消参考建立")
                    return False
                self.log("[聚焦参考] 尚未选择补焦基准图 ROI，即将弹出 ROI 选择窗口")
                if self._select_focus_roi_interactively():
                    self._focus_roi_selected = True
                    break
                self.log("[聚焦参考] ROI 选择未完成，必须选择后才能继续测量")
                self._show_roi_selection_required_dialog()

            # ROI 已确认，尝试采集参考图
            image_rgb = self._capture_current_focus_frame()
            if image_rgb is None or image_rgb.size == 0:
                self.log("[聚焦参考] ROI 选择后仍无法获取有效图像，跳过参考建立")
                return False

            # 补焦参考基准图预览窗口已移除：
            # 之前弹出的 "Focus Reference Baseline Image" OpenCV 窗口容易引起"未响应"，
            # 且 ROI 选择窗口已足够确认补焦区域，因此直接保存参考图到文件即可。
            self.log("[聚焦参考] 参考图已建立并保存，不再弹出实时预览窗口")

            ref_dir = self.output_root / "focus_reference"
            ref_dir.mkdir(parents=True, exist_ok=True)
            ref = self._focus_scorer.build_reference_from_image(
                image_rgb=image_rgb,
                output_root=ref_dir,
                on_log=self.log,
            )
            self._focus_reference_image = ref.get("full_rgb")
            self._focus_reference_ready = True
            self._focus_consecutive_low_count = 0
            self.log(
                f"[聚焦参考] 第 {cycle_index} 轮已建立，ROI="
                f"{self._focus_metrics_calc.cfg.focus_roi}"
            )
            return True
        except Exception as e:
            self.log(f"[聚焦参考] 建立失败：{e}")
            self.log(traceback.format_exc())
            return False

    def compute_current_focus_score(self) -> Optional[float]:
        """截取当前画面并计算与参考图的 FocusScore_ratio。"""
        if not self._focus_reference_ready or self._focus_scorer is None:
            return None
        image_rgb = self._capture_current_focus_frame()
        if image_rgb is None or image_rgb.size == 0:
            return None
        roi = self._focus_metrics_calc.clamp_roi(
            tuple(self._focus_metrics_calc.cfg.focus_roi), image_rgb.shape
        )
        x, y, rw, rh = roi
        roi_rgb = image_rgb[y : y + rh, x : x + rw].copy()
        roi_metrics = self._focus_metrics_calc.compute_for_image(roi_rgb)
        score, _ = self._focus_scorer.score_ratio(roi_metrics)
        return score

    def _ensure_focus_controller(self) -> AutofocusController:
        """获取或创建 AutofocusController；virtual 模式使用 simulator。"""
        if self._focus_controller is not None:
            return self._focus_controller

        self._ensure_focus_components()
        cfg = self._focus_metrics_calc.cfg

        if self._is_virtual_hardware_mode():
            self.log("[聚焦补焦] virtual 模式：使用 FocusSimulator")
            simulator, metrics_calc, scorer, controller = create_demo_environment(
                cfg,
                peak_z=50,
                blur_scale=0.3,
                image_size=(400, 400),
                initial_z=0,
            )
            self._focus_metrics_calc = metrics_calc
            self._focus_scorer = scorer
            self._focus_simulator = simulator
        else:
            z_axis = ZAxisController(cfg)
            controller = AutofocusController(
                cfg, self._focus_scorer, self._focus_metrics_calc, z_axis
            )

        controller.on_log = self.log
        controller.should_stop = lambda: self.stop_requested
        self._focus_controller = controller
        return controller

    def run_autofocus_if_needed(self, cycle_index: int) -> Dict[str, Any]:
        """
        计算当前 FocusScore_ratio，若连续超出阈值则执行补焦。
        返回 {"score": float|None, "triggered": bool, "autofocus_ok": bool}。
        若实际执行了闭环补焦，score 会更新为补焦后的最终分数，
        原始触发前分数保存在 pre_score。
        """
        result = {
            "score": None,
            "pre_score": None,
            "final_score": None,
            "triggered": False,
            "autofocus_ok": False,
        }
        score = self.compute_current_focus_score()
        result["score"] = score
        result["pre_score"] = score

        if score is None:
            self.log("[聚焦补焦] 当前 FocusScore 为空，跳过")
            return result

        trigger_ratio = float(self.cfg.focus_trigger_ratio)
        absolute = bool(self.cfg.focus_trigger_absolute)
        lower = min(trigger_ratio, 2.0 - trigger_ratio) if absolute else trigger_ratio
        upper = max(trigger_ratio, 2.0 - trigger_ratio) if absolute else float("inf")

        if focus_score_ratio_in_tolerance(score, trigger_ratio, absolute):
            self._focus_consecutive_low_count = 0
            self.log(
                f"[聚焦补焦] FocusScore_ratio={score:.4f} 在允许区间"
                f"[{lower:.4f}, {upper:.4f}] 内，不触发补焦"
            )
            return result

        self._focus_consecutive_low_count += 1
        self.log(
            f"[聚焦补焦] FocusScore_ratio={score:.4f} 超出允许区间"
            f"[{lower:.4f}, {upper:.4f}]，连续低分计数="
            f"{self._focus_consecutive_low_count}/{self.cfg.focus_trigger_count}"
        )

        if self._focus_consecutive_low_count < int(self.cfg.focus_trigger_count):
            return result

        result["triggered"] = True
        if not bool(self.cfg.focus_z_enabled):
            self.log("[聚焦补焦] 已触发但 Z 轴禁用，不执行补焦")
            return result

        if bool(self.cfg.focus_detection_only):
            self.log("[聚焦补焦] FocusScore检测模式：仅记录，不执行 Z 轴闭环")
            return result

        try:
            self.log("[聚焦补焦] 触发补焦，启动闭环搜索")
            controller = self._ensure_focus_controller()

            # 策略降级机制：如果上一次 hill_climb 结束但分数未达标，本次改用 full_sweep
            current_strategy = str(getattr(self.cfg, "focus_search_strategy", "hill_climb"))
            _last_search_failed = getattr(self, "_last_autofocus_search_failed", False)
            if _last_search_failed and current_strategy != "full_sweep":
                self.log(f"[聚焦补焦] 上次 {current_strategy} 搜索未达标，本次切换为 full_sweep")
                original_strategy = self.cfg.focus_search_strategy
                self.cfg.focus_search_strategy = "full_sweep"
                try:
                    autofocus_result = controller.run_closed_loop(initial_focus_score=score)
                finally:
                    self.cfg.focus_search_strategy = original_strategy
            else:
                autofocus_result = controller.run_closed_loop(initial_focus_score=score)

            result["autofocus_ok"] = bool(autofocus_result.get("ok", False))
            final_score = autofocus_result.get("best_score")
            result["final_score"] = final_score
            if final_score is not None:
                result["score"] = final_score
            self.log(
                f"[聚焦补焦] 闭环结束：ok={result['autofocus_ok']}, "
                f"best_score={final_score}"
            )
            if final_score is not None and focus_score_ratio_in_tolerance(
                final_score, trigger_ratio, absolute
            ):
                self._focus_consecutive_low_count = 0
                self._last_autofocus_search_failed = False
            else:
                # 搜索正常结束但分数未达标，标记下次需要使用更宽范围策略
                self._last_autofocus_search_failed = True
                self.log(f"[聚焦补焦] 搜索未达标（best={final_score}），下次将尝试更大范围搜索")
        except Exception as e:
            self.log(f"[聚焦补焦] 闭环补焦异常：{e}")
            self.log(traceback.format_exc())

        return result

    @staticmethod
    def _parse_focus_roi_text(value: Any) -> Tuple[int, int, int, int]:
        """把 'x,y,w,h' 字符串或四元组解析为整数元组；失败返回 (0,0,300,300)。"""
        try:
            if isinstance(value, (tuple, list)) and len(value) == 4:
                return tuple(int(v) for v in value)  # type: ignore
            parts = [int(v.strip().strip("()[]")) for v in str(value).split(",")]
            if len(parts) == 4:
                return tuple(parts)  # type: ignore
        except Exception:
            pass
        return (0, 0, 300, 300)

    def _set_midrun_recalibration(self, cycle_index: int, phase: str) -> None:
        """统一设置中途重标定状态并记录日志。"""
        self.restart_current_cycle_after_recalibration = True
        self.midrun_recalibration_cycle_index = int(cycle_index)
        self.context["restart_current_cycle_after_recalibration"] = True
        self.context["midrun_recalibration_cycle_index"] = int(cycle_index)
        self.log(f"[中途重标定] 第 {cycle_index} 轮 {phase} 检测到请求，暂停并等待重标定。")

    def _acquire_and_save_spectrum(
        self,
        cycle_index: int,
        paths: Dict[str, Any],
        angle_result: Dict[str, Any],
        off_label: str,
        acquire_label: str,
        on_label: str,
        save_label: str,
        append_plot_point: bool = True,
    ) -> bool:
        """
        标准光谱采集保存块：照明光 OFF → LabVIEW 光谱采集 → 照明光 ON → 异步保存。

        参数中的 *_label 用于日志区分 Step 3/4/5/6 与 Step 11/12/13/14。
        保存角度固定使用 Step1 传入的 angle_result。
        append_plot_point 控制是否向 angle-fit 列表追加绘图点，用于避免一轮多次保存导致列表翻倍。
        返回 False 表示流程被停止或触发中途重标定。
        """
        # 照明光 OFF，等待稳定
        self.log(f"========== {off_label}：照明光 OFF，等待稳定 ==========")
        self.light_off()

        wait_after_off = float(self.cfg.stable_wait_ms) / 1000.0
        if wait_after_off > 0:
            self.log(f"[照明光] OFF 后稳定等待：{wait_after_off:.3f} s（由“稳定等待/ms”控制）")
            time.sleep(wait_after_off)
        else:
            self.log("[照明光] OFF 后稳定等待：0 s，立即进入 LabVIEW 光谱采集")

        if self.stop_requested:
            return False
        if self._is_midrun_recalibration_requested():
            self._set_midrun_recalibration(cycle_index, f"{off_label} 后")
            return False

        # LabVIEW 光谱采集
        self.log(f"========== {acquire_label}：LabVIEW 光谱采集 ==========")
        labview_result = self.request_labview_spectrum(cycle_index)

        if self.stop_requested:
            return False
        if self._is_midrun_recalibration_requested():
            self._set_midrun_recalibration(cycle_index, f"{acquire_label} 后")
            return False

        # 照明光 ON
        self.log(f"========== {on_label}：照明光 ON ==========")
        self.light_on()

        if self.stop_requested:
            return False
        if self._is_midrun_recalibration_requested():
            self._set_midrun_recalibration(cycle_index, f"{on_label} 后")
            return False

        # 保存数据
        self.log(f"========== {save_label}：保存数据（异步线程，使用 Step1 最终角度） ==========")
        angle_after_result_for_save = {
            "ok": False,
            "angle_deg": None,
            "reason": "save_uses_step1_final_angle",
        }
        t_save_submit0 = time.time()
        self.start_save_cycle_result_async(
            cycle_index=cycle_index,
            paths=paths,
            angle_before_result=angle_result,
            angle_after_result=angle_after_result_for_save,
            labview_result=labview_result,
            append_plot_point=append_plot_point,
        )
        self.log(f"[{save_label}] 异步保存任务提交耗时：{time.time() - t_save_submit0:.3f} s；保存角度=Step1最终角度；append_plot_point={append_plot_point}")
        return True

    # --------------------------------------------------------
    # 单轮完整流程
    # --------------------------------------------------------

    def run_one_cycle(self, cycle_index: int) -> bool:
        """
        单轮循环测量流程：
        Step 1：角度检测一次，得到 current_angle
        Step 1.5（仅第一轮）：在照明光 OFF 前，截取当前 ROI 画面建立聚焦参考图
        Step 2：生成保存路径
        Step 3：照明光 OFF，并按 stable_wait_ms 等待稳定
        Step 4：LabVIEW 光谱采集
        Step 5：照明光 ON
        Step 6：保存本轮数据；保存角度使用 Step1 的 YOLO-OBB baseline 原始角度



        Step 7：打开激光
        Step 8：A推动B（logic/rule_ab.py），每次运动后检测B对边角度；
                Step7 角度只用于关闭激光/是否继续推动，不再覆盖本轮保存角度。
        Step 9：检测颜色区域中心，与提前选定位置对齐；偏离过大则移动1/2通道
        Step 10：补焦判断。计算当前 ROI 与参考图的 FocusScore_ratio；
                  若触发阈值，执行补焦。
        Step 11：照明光 OFF，并按 stable_wait_ms 等待稳定
        Step 12：LabVIEW 光谱采集
        Step 13：照明光 ON
        Step 14：保存本轮数据；保存角度使用 Step1 的 YOLO-OBB baseline 原始角度

        Step 15：继续 Step 7 到 Step 14 的循环（固定次数，由 sub_loop_iterations_per_cycle 控制）
        """
        self.context["cycle_index"] = cycle_index
        self.notify_update()

        if self._is_midrun_recalibration_requested():
            self._set_midrun_recalibration(cycle_index, "开始前")
            return False

        self.log("")
        self.log("############################################################")
        self.log(f"开始第 {cycle_index} 轮循环测量")
        self.log("############################################################")

        try:
            self.context["signal_on_time_used_ms"] = None
            self.context["signal_on_time_next_ms"] = None
            self.context["signal_time_adjust_action"] = None

            self.log("========== Step 1：Bmask最长边角度检测 current_angle ==========")
            angle_result = {"ok": False, "angle_deg": None, "reason": "bmask_longest_edge_not_run"}
            follower_for_angle = None
            try:
                follower_for_angle = self._ensure_rule_ab_follower_with_startup_retry(
                    label="step1_bmask_longest_edge_angle"
                )
                self.apply_runtime_rule_ab_params_to_follower(follower_for_angle, reason="step1_bmask_longest_edge_angle")
                self._ensure_rule_ab_feature_tracker_installed(follower_for_angle, self._get_loaded_calibration_state())
                angle_result = self.detect_step7_yolo_obb_angle_once(
                    follower=follower_for_angle,
                    label="current_angle_bmask_longest_edge",
                    baseline_angle=None,
                    allow_fail=True,
                    allow_close_from_relocation=False,
                )
                if angle_result.get("ok", False) and angle_result.get("angle_deg") is not None:
                    self.log(
                        f"[角度检测-current_angle] 已使用当前帧 Bmask 最长边角度作为本轮 baseline："
                        f"{angle_result.get('angle_deg')}°；source={angle_result.get('angle_source')}"
                    )
                else:
                    self.log(
                        f"[角度检测-current_angle] Bmask 最长边 baseline 计算失败，"
                        f"不再回退 A 最近边作为主角度来源；reason={angle_result.get('reason')}"
                    )
            except Exception as e:
                self.log(f"[角度检测-current_angle] Bmask 最长边 baseline 计算异常：{e}")
                angle_result = {"ok": False, "angle_deg": None, "reason": str(e), "angle_source": "failed_no_close"}

            if angle_result.get("ok", False) and angle_result.get("angle_deg") is not None:
                current_angle = float(angle_result["angle_deg"])
            else:
                current_angle = None
                reason = angle_result.get("reason")
                self.log(
                    "[角度检测-current_angle] 本轮 Bmask 最长边角度无效；"
                    "不再把它当作整次测量失败，本轮不进入 Step7，直接跳过并继续下一轮。"
                    f" reason={reason}"
                )
                self.context["last_angle_result"] = angle_result
                return self._skip_current_cycle_without_stopping_measurement(
                    cycle_index=cycle_index,
                    phase="Step1_Bmask_longest_edge_baseline",
                    reason=reason,
                    close_laser_if_on=False,
                )

            # 兼容旧显示/保存字段：angle_before 作为本轮 Step1 baseline，angle_after 不再使用。
            self.context["angle_before"] = current_angle
            self.context["angle_after"] = None
            self.context["save_angle_deg"] = current_angle
            self.context["angle_before_after_delta"] = None
            self.context["last_angle_result"] = angle_result
            self.context["previous_cycle_angle"] = self.previous_cycle_angle
            if self.previous_cycle_angle is not None and current_angle is not None:
                self.context["angle_delta"] = self.angle_diff_deg(current_angle, self.previous_cycle_angle)
            else:
                self.context["angle_delta"] = None
            self.notify_update()

            if current_angle is not None:
                self.previous_cycle_angle = float(current_angle)

            # 仅在第一轮且未建立参考图时建立聚焦参考；必须完成 ROI 选择
            if cycle_index == 1 and not self._focus_reference_ready:
                if not self.capture_focus_reference(cycle_index):
                    self.log("[流程] 聚焦参考图建立失败，无法继续测量")
                    self.stop_requested = True
                    return False

            self.log("========== Step 2：生成保存路径 ==========")
            # 初始光谱采集使用序号0（基准测量）
            paths = self.build_save_path(0)

            # Step 3-6：初始光谱采集（基准测量，序号0）
            # 这是所有后续测量的基准参考。
            # 无论是否存在实际主循环，都向角度-拟合峰值列表追加序号 0 的初始化点，
            # 确保右侧列表显示"第一轮初始化"的角度与拟合峰值。
            sub_loop_count = max(0, int(getattr(self.cfg, "sub_loop_iterations_per_cycle", 1)))
            initial_append_plot = True

            if not self._acquire_and_save_spectrum(
                cycle_index=0,  # 初始光谱采集使用序号0（基准）
                paths=paths,
                angle_result=angle_result,
                off_label="Step 3",
                acquire_label="Step 4",
                on_label="Step 5",
                save_label="Step 6",
                append_plot_point=initial_append_plot,
            ):
                return False



            # Step 7-14 实际主循环（原子循环）
            # 每次子循环使用序号1, 2, 3...作为实际的主循环序号
            self.log(f"[流程] 本周期实际主循环次数：{sub_loop_count}（序号1-{sub_loop_count}）")

            last_rule_ab_result: Dict[str, Any] = {}
            last_rule_ac_result: Dict[str, Any] = {}
            last_autofocus_result: Dict[str, Any] = {}

            # 实际主循环（原子循环）
            # 保存框架主循环序号，用于循环结束后恢复（框架主循环序号始终为1）
            main_cycle_index = cycle_index

            for sub_idx in range(1, sub_loop_count + 1):
                if self.stop_requested:
                    return False
                if self._is_midrun_recalibration_requested():
                    self._set_midrun_recalibration(cycle_index, f"实际主循环第 {sub_idx} 次开始前")
                    return False

                # 在实际主循环内部，将cycle_index更新为实际主循环序号（1, 2, 3...）
                cycle_index = sub_idx
                self.context["cycle_index"] = cycle_index
                self.notify_update()

                self.log(f"========== 实际主循环第 {sub_idx}/{sub_loop_count} 次开始 ==========")
                
                '''
                self.log("========== Step 1：Bmask最长边角度检测 current_angle ==========")
                angle_result = {"ok": False, "angle_deg": None, "reason": "bmask_longest_edge_not_run"}
                follower_for_angle = None
                try:
                    follower_for_angle = self._ensure_rule_ab_follower_with_startup_retry(
                        label="step1_bmask_longest_edge_angle"
                    )
                    self.apply_runtime_rule_ab_params_to_follower(follower_for_angle, reason="step1_bmask_longest_edge_angle")
                    self._ensure_rule_ab_feature_tracker_installed(follower_for_angle, self._get_loaded_calibration_state())
                    angle_result = self.detect_step7_yolo_obb_angle_once(
                        follower=follower_for_angle,
                        label="current_angle_bmask_longest_edge",
                        baseline_angle=None,
                        allow_fail=True,
                        allow_close_from_relocation=False,
                    )
                    if angle_result.get("ok", False) and angle_result.get("angle_deg") is not None:
                        self.log(
                            f"[角度检测-current_angle] 已使用当前帧 Bmask 最长边角度作为本轮 baseline："
                            f"{angle_result.get('angle_deg')}°；source={angle_result.get('angle_source')}"
                        )
                    else:
                        self.log(
                            f"[角度检测-current_angle] Bmask 最长边 baseline 计算失败，"
                            f"不再回退 A 最近边作为主角度来源；reason={angle_result.get('reason')}"
                        )
                except Exception as e:
                    self.log(f"[角度检测-current_angle] Bmask 最长边 baseline 计算异常：{e}")
                    angle_result = {"ok": False, "angle_deg": None, "reason": str(e), "angle_source": "failed_no_close"}

                if angle_result.get("ok", False) and angle_result.get("angle_deg") is not None:
                    current_angle = float(angle_result["angle_deg"])
                else:
                    current_angle = None
                    reason = angle_result.get("reason")
                    self.log(
                        "[角度检测-current_angle] 本轮 Bmask 最长边角度无效；"
                        "不再把它当作整次测量失败，本轮不进入 Step7，直接跳过并继续下一轮。"
                        f" reason={reason}"
                    )
                    self.context["last_angle_result"] = angle_result
                    _skip_ok = self._skip_current_cycle_without_stopping_measurement(
                        cycle_index=cycle_index,
                        phase="Step1_Bmask_longest_edge_baseline",
                        reason=reason,
                        close_laser_if_on=False,
                    )
                    if not _skip_ok:
                        return False
                    # 跳过本轮剩余步骤（Step11-14 光谱采集），继续子循环的下一轮
                    continue

                # 兼容旧显示/保存字段：angle_before 作为本轮 Step1 baseline，angle_after 不再使用。
                self.context["angle_before"] = current_angle
                self.context["angle_after"] = None
                self.context["save_angle_deg"] = current_angle
                self.context["angle_before_after_delta"] = None
                self.context["last_angle_result"] = angle_result
                self.context["previous_cycle_angle"] = self.previous_cycle_angle
                if self.previous_cycle_angle is not None and current_angle is not None:
                    self.context["angle_delta"] = self.angle_diff_deg(current_angle, self.previous_cycle_angle)
                else:
                    self.context["angle_delta"] = None
                self.notify_update()

                if current_angle is not None:
                    self.previous_cycle_angle = float(current_angle)

                # 仅在第一轮主循环且未建立参考图时建立聚焦参考；必须完成 ROI 选择
                # 注意：这里检查框架主循环序号(main_cycle_index=1)和实际主循环序号(sub_idx=1)
                if main_cycle_index == 1 and sub_idx == 1 and not self._focus_reference_ready:
                    if not self.capture_focus_reference(cycle_index):
                        self.log("[流程] 聚焦参考图建立失败，无法继续测量")
                        self.stop_requested = True
                        return False

                self.log(f"========== 实际主循环第 {sub_idx} 次生成保存路径 ==========")
                paths = self.build_save_path(cycle_index)
                '''

                # Step 7：打开激光
                self.log("========== Step 7：打开激光 ==========")
                self.laser_on()

                # Step 8：A推动B
                self.log("========== Step 8：A推动B，并用当前帧Bmask最长边检测B角度 ==========")

                # 完整测量 Step8 的目标就是让 3/4 通道推动 A。
                # 如果 GUI 中“RuleAB真动Stage34”没有勾选，旧代码会 dry-run，表现为 3/4 通道没有任何变化。
                # 这里在完整测量中自动打开真动 Stage34；单独测试按钮仍可通过 GUI 选择 dry-run。
                if self._is_virtual_hardware_mode():
                    self.cfg.rule_ab_enable_stage = False
                    if self.rule_ab_follower is not None and getattr(self.rule_ab_follower, "cfg", None) is not None:
                        try:
                            self.rule_ab_follower.cfg.enable_stage = False
                        except Exception:
                            pass
                    self.log("[RuleAB][virtual] 完整测量 Step8 使用真实视觉/角度逻辑，但 Stage34 为虚拟动作：不连接、不控制真实 3/4 通道。")
                elif not bool(getattr(self.cfg, "rule_ab_enable_stage", False)):
                    self.log("[RuleAB] 完整测量 Step8 检测到 rule_ab_enable_stage=False，已自动改为 True，避免3/4通道 dry-run。")
                    self.cfg.rule_ab_enable_stage = True
                    if self.rule_ab_follower is not None and getattr(self.rule_ab_follower, "cfg", None) is not None:
                        try:
                            self.rule_ab_follower.cfg.enable_stage = True
                        except Exception:
                            pass

                rule_ab_result = self.run_rule_ab_until_angle_delta(
                    baseline_angle=current_angle,
                    min_delta_deg=None,
                    max_delta_deg=None,
                )
                last_rule_ab_result = rule_ab_result

                if self._is_midrun_recalibration_requested() or str(rule_ab_result.get("reason", "")) == "midrun_recalibration_requested":
                    self._set_midrun_recalibration(cycle_index, f"实际主循环第 {sub_idx} 次 RuleAB")
                    self.context["last_rule_ab_result"] = rule_ab_result
                    self.notify_update()
                    return False

                self.context["last_rule_ab_result"] = rule_ab_result
                self.notify_update()

                if self.stop_requested:
                    return False

                # Step8 判定：
                # 只有 YOLO-OBB 检测角度变化 >= min_delta，Step8 才算成功并允许继续完整测量。
                # 当前版本不再使用 max_delta / 6° 上限；达到 3.5° 阈值即进入下一步。
                # 修改：Bmask/角度检测类失败不再当作整次测量失败；只跳过当前 cycle，完整循环继续。
                # 但用户主动停止、硬件/激光关闭失败、中途重标定等仍按原逻辑停止或暂停。
                if not bool(rule_ab_result.get("ok", False)):
                    final_delta = rule_ab_result.get("final_delta")
                    target_min = rule_ab_result.get("target_min")
                    target_max = rule_ab_result.get("target_max")
                    reason = rule_ab_result.get("reason")

                    if self._is_nonfatal_bmask_angle_failure_reason(reason):
                        self.log(
                            "[RuleAB] Step8 未达到目标角度或 Bmask/角度检测失败；"
                            "本次不再停止完整循环，只跳过当前轮并继续下一轮："
                            f"final_delta={final_delta}, target=[{target_min}, {target_max}], "
                            f"reason={reason}, records={len(rule_ab_result.get('records', []))}"
                        )
                        self.context["last_rule_ab_result"] = rule_ab_result
                        _skip_ok = self._skip_current_cycle_without_stopping_measurement(
                            cycle_index=cycle_index,
                            phase="Step8_Bmask_angle_or_delta_not_ready",
                            reason=reason,
                            close_laser_if_on=True,
                        )
                        if not _skip_ok:
                            return False
                        # 跳过本轮剩余步骤（Step9-14），继续子循环的下一轮
                        continue

                    self.log(
                        "[RuleAB] Step8 发生非 Bmask/角度类失败，仍停止完整循环测量："
                        f"final_delta={final_delta}, target=[{target_min}, {target_max}], "
                        f"reason={reason}, records={len(rule_ab_result.get('records', []))}"
                    )
                    self.context["rule_ab_failed_stop"] = True
                    self.context["rule_ab_failed_reason"] = reason
                    self.stop_requested = True
                    self.notify_update()
                    return False
                
                # Step8 成功：角度检测通过，判断是否已达到阈值并关闭激光
                final_delta = rule_ab_result.get("final_delta", 0)
                target_min = rule_ab_result.get("target_min")
                if target_min is not None and final_delta >= target_min:
                    # 角度达到阈值，尝试关闭激光
                    if self.context.get("laser_on", False):
                        try:
                            self.laser_off()
                            self.log(f"[RuleAB] Step8 角度达到阈值={target_min}，激光已关闭")
                        except Exception as e:
                            self.log(f"[RuleAB] Step8 角度达到阈值={target_min}，但关闭激光失败：{e}")
                            self.context["laser_off_failed_after_step8"] = True
                            self.context["laser_off_failed_reason"] = str(e)
                else:
                    self.log(f"[RuleAB] Step8 角度检测通过但最终_delta={final_delta} < 阈值={target_min}")


                


                # Step 9：检测颜色区域中心，必要时移动1/2通道
                self.log("========== Step 9：检测颜色区域中心，必要时移动1/2通道 ==========")
                rule_ac_result = self.run_rule_ac_until_threshold()
                last_rule_ac_result = rule_ac_result
                self.context["last_rule_ac_result"] = rule_ac_result
                self.notify_update()

                if self._is_midrun_recalibration_requested() or str(rule_ac_result.get("reason", "")) == "midrun_recalibration_requested":
                    self._set_midrun_recalibration(cycle_index, f"实际主循环第 {sub_idx} 次 RuleAC")
                    self.notify_update()
                    return False

                # Step 10：补焦判断（循环等待直到补焦完成或分数达标）
                self.log("========== Step 10：补焦判断（循环等待） ==========")
                autofocus_check_count = 0
                max_autofocus_checks = int(self.cfg.focus_max_checks_per_cycle) if hasattr(self.cfg, 'focus_max_checks_per_cycle') else 20
                while autofocus_check_count < max_autofocus_checks:
                    autofocus_check_count += 1
                    try:
                        autofocus_result = self.run_autofocus_if_needed(cycle_index)
                        last_autofocus_result = autofocus_result
                        self.context["last_autofocus_result"] = autofocus_result
                        self.notify_update()

                        # 仅以分数达标作为退出条件（autofocus_ok 只代表搜索正常结束，不代表达标）
                        score = autofocus_result.get("score")
                        triggered = autofocus_result.get("triggered", False)
                        autofocus_ok = autofocus_result.get("autofocus_ok", False)

                        if score is not None and focus_score_ratio_in_tolerance(
                            score,
                            float(self.cfg.focus_trigger_ratio),
                            bool(self.cfg.focus_trigger_absolute)
                        ):
                            self.log(f"[补焦判断] FocusScore 已达标，退出等待循环（第 {autofocus_check_count} 次检查）")
                            break

                        # 未达标，记录当前状态并继续等待
                        if triggered and not autofocus_ok:
                            self.log(f"[补焦判断] 补焦已触发但未成功（score={score}），继续等待（第 {autofocus_check_count} 次检查）")
                        elif triggered and autofocus_ok:
                            self.log(f"[补焦判断] 补焦搜索正常结束但分数未达标（score={score}），继续等待（第 {autofocus_check_count} 次检查）")
                        else:
                            self.log(f"[补焦判断] 分数未达标（score={score}），继续等待（第 {autofocus_check_count} 次检查）")

                        # 等待一个间隔后再次检查
                        if autofocus_check_count < max_autofocus_checks:
                            wait_s = float(self.cfg.focus_check_interval_s) if hasattr(self.cfg, 'focus_check_interval_s') else 1.5
                            self.log(f"[补焦判断] 等待 {wait_s:.1f}s 后再次检查...")
                            time.sleep(wait_s)
                            
                    except Exception as e:
                        self.log(f"[补焦判断] 异常：{e}")
                        self.log(traceback.format_exc())
                        last_autofocus_result = {"score": None, "triggered": False, "autofocus_ok": False, "error": str(e)}
                        self.context["last_autofocus_result"] = last_autofocus_result
                        # 异常后等待一会再重试
                        time.sleep(1.0)
                        continue
                    
                    if self.stop_requested:
                        return False
                    if self._is_midrun_recalibration_requested():
                        self._set_midrun_recalibration(cycle_index, f"实际主循环第 {sub_idx} 次补焦")
                        return False
                
                if autofocus_check_count >= max_autofocus_checks:
                    self.log(f"[补焦判断] 已达到最大检查次数 {max_autofocus_checks}，继续进入下一步")

                if self.stop_requested:
                    return False
                if self._is_midrun_recalibration_requested():
                    self._set_midrun_recalibration(cycle_index, f"实际主循环第 {sub_idx} 次补焦检查")
                    return False
                
                # 角度测量 + 光谱角度数据保存
                self.log("========== Step 1：Bmask最长边角度检测 current_angle ==========")
                angle_result = {"ok": False, "angle_deg": None, "reason": "bmask_longest_edge_not_run"}
                follower_for_angle = None
                try:
                    follower_for_angle = self._ensure_rule_ab_follower_with_startup_retry(
                        label="step1_bmask_longest_edge_angle"
                    )
                    self.apply_runtime_rule_ab_params_to_follower(follower_for_angle, reason="step1_bmask_longest_edge_angle")
                    self._ensure_rule_ab_feature_tracker_installed(follower_for_angle, self._get_loaded_calibration_state())
                    angle_result = self.detect_step7_yolo_obb_angle_once(
                        follower=follower_for_angle,
                        label="current_angle_bmask_longest_edge",
                        baseline_angle=None,
                        allow_fail=True,
                        allow_close_from_relocation=False,
                    )
                    if angle_result.get("ok", False) and angle_result.get("angle_deg") is not None:
                        self.log(
                            f"[角度检测-current_angle] 已使用当前帧 Bmask 最长边角度作为本轮 baseline："
                            f"{angle_result.get('angle_deg')}°；source={angle_result.get('angle_source')}"
                        )
                    else:
                        self.log(
                            f"[角度检测-current_angle] Bmask 最长边 baseline 计算失败，"
                            f"不再回退 A 最近边作为主角度来源；reason={angle_result.get('reason')}"
                        )
                except Exception as e:
                    self.log(f"[角度检测-current_angle] Bmask 最长边 baseline 计算异常：{e}")
                    angle_result = {"ok": False, "angle_deg": None, "reason": str(e), "angle_source": "failed_no_close"}

                if angle_result.get("ok", False) and angle_result.get("angle_deg") is not None:
                    current_angle = float(angle_result["angle_deg"])
                else:
                    current_angle = None
                    reason = angle_result.get("reason")
                    self.log(
                        "[角度检测-current_angle] 本轮 Bmask 最长边角度无效；"
                        "不再把它当作整次测量失败，本轮不进入 Step7，直接跳过并继续下一轮。"
                        f" reason={reason}"
                    )
                    self.context["last_angle_result"] = angle_result
                    _skip_ok = self._skip_current_cycle_without_stopping_measurement(
                        cycle_index=cycle_index,
                        phase="Step1_Bmask_longest_edge_baseline",
                        reason=reason,
                        close_laser_if_on=False,
                    )
                    if not _skip_ok:
                        return False
                    # 跳过本轮剩余步骤（Step11-14 光谱采集），继续子循环的下一轮
                    continue

                # 兼容旧显示/保存字段：angle_before 作为本轮 Step1 baseline，angle_after 不再使用。
                self.context["angle_before"] = current_angle
                self.context["angle_after"] = None
                self.context["save_angle_deg"] = current_angle
                self.context["angle_before_after_delta"] = None
                self.context["last_angle_result"] = angle_result
                self.context["previous_cycle_angle"] = self.previous_cycle_angle
                if self.previous_cycle_angle is not None and current_angle is not None:
                    self.context["angle_delta"] = self.angle_diff_deg(current_angle, self.previous_cycle_angle)
                else:
                    self.context["angle_delta"] = None
                self.notify_update()

                if current_angle is not None:
                    self.previous_cycle_angle = float(current_angle)

                # 仅在第一轮主循环且未建立参考图时建立聚焦参考；必须完成 ROI 选择
                # 注意：这里检查框架主循环序号(main_cycle_index=1)和实际主循环序号(sub_idx=1)
                if main_cycle_index == 1 and sub_idx == 1 and not self._focus_reference_ready:
                    if not self.capture_focus_reference(cycle_index):
                        self.log("[流程] 聚焦参考图建立失败，无法继续测量")
                        self.stop_requested = True
                        return False

                self.log(f"========== 实际主循环第 {sub_idx} 次生成保存路径 ==========")
                paths = self.build_save_path(cycle_index)                


                # Step 11-14：光谱采集与保存
                # 每轮实际主循环都向角度-拟合峰值列表追加绘图点（序号1, 2, 3...）
                if not self._acquire_and_save_spectrum(
                    cycle_index=cycle_index,
                    paths=paths,
                    angle_result=angle_result,
                    off_label="Step 11",
                    acquire_label="Step 12",
                    on_label="Step 13",
                    save_label="Step 14",
                    append_plot_point=True,  # 每轮实际主循环都追加绘图点
                ):
                    return False

                self.log(f"========== 实际主循环第 {sub_idx}/{sub_loop_count} 次结束 ==========")

            # 实际主循环结束后恢复框架主循环序号
            cycle_index = main_cycle_index
            self.context["cycle_index"] = cycle_index
            self.notify_update()

            # 实际主循环结束后关闭激光（安全兜底）
            self.log("========== 实际主循环结束：关闭激光 ==========")
            try:
                if bool(self.context.get("laser_on", False)):
                    self.laser_off()
                    self.log("[激光开关] 实际主循环结束，主流程正常关闭激光")
                else:
                    self.log("[激光开关] 实际主循环结束，但 laser_on=False，主流程跳过重复关激光")
            except Exception as e:
                self.log(f"[激光开关] 实际主循环结束后正常关闭激光失败：{e}")
                self.context["laser_off_failed_after_step7"] = True
                self.context["laser_off_failed_reason"] = str(e)
                self.stop_requested = True
                self.notify_update()
                return False

            # 使用最后一轮子循环的结果更新上下文（用于排查/GUI显示）
            final_step7_angle = last_rule_ab_result.get("final_angle")
            final_step7_delta = last_rule_ab_result.get("final_delta")
            last_records = last_rule_ab_result.get("records") or []
            last_record = last_records[-1] if last_records else {}
            last_angle_result = last_record.get("angle_result") if isinstance(last_record.get("angle_result"), dict) else {}

            self.context["angle_before"] = current_angle
            self.context["angle_after"] = final_step7_angle
            self.context["angle_before_after_delta"] = final_step7_delta
            # 不修改 self.context["save_angle_deg"]：它必须保持 Step1 最终角度。
            self.context["last_step7_angle_result"] = last_angle_result or {}
            self.context["last_rule_ab_final_angle"] = final_step7_angle
            self.context["last_rule_ab_final_delta"] = final_step7_delta
            self.context["last_rule_ab_uses_raw_yolo_angle"] = True
            self.notify_update()

            self.log(
                f"[流程] 第 {cycle_index} 轮数据已提交保存；"
                f"保存角度=Step1最终角度 {current_angle}；"
                f"Step8最终角度={final_step7_angle} 仅用于关闭激光/排查，不覆盖保存角度。"
            )
            self.notify_update()
            return True

        except Exception as e:
            # 判断是否在子循环中发生的错误
            in_sub_loop = 'main_cycle_index' in locals() and cycle_index != main_cycle_index
            if in_sub_loop:
                # 子循环内的错误，cycle_index是子循环序号
                self.log(f"[错误] 主循环第 {main_cycle_index} 轮，子循环第 {cycle_index} 次测量失败：{e}")
                error_path = error_dir / f"error_cycle_{main_cycle_index:04d}_sub_{cycle_index:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                error_info = {
                    "main_cycle_index": main_cycle_index,
                    "sub_cycle_index": cycle_index,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "context": self._safe_context_for_json(),
                }
            else:
                # 主循环中的错误
                self.log(f"[错误] 第 {cycle_index} 轮测量失败：{e}")
                error_path = error_dir / f"error_cycle_{cycle_index:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                error_info = {
                    "cycle_index": cycle_index,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "context": self._safe_context_for_json(),
                }

            self.log(traceback.format_exc())

            error_dir = self.output_root / "errors"
            error_dir.mkdir(parents=True, exist_ok=True)

            with error_path.open("w", encoding="utf-8") as f:
                json.dump(error_info, f, ensure_ascii=False, indent=2)

            self.log(f"[错误] 错误信息已保存：{error_path}")

            self.finish()
            return False

    def _safe_context_for_json(self) -> Dict[str, Any]:
        return {k: self._json_safe(v) for k, v in self.context.items()}

    def run(self):
        self.connect_measurement_devices()

        # 完整测量恢复真实光谱仪 TCP：运行前确保 Server 已启动并完成 READY。
        if self._is_virtual_hardware_mode():
            self.start_tcp_server()
            self.wait_labview_ready()
            self.log("[流程][virtual] 已启用虚拟 TCP/虚拟光谱；完整测量 Step4 仍走 request_labview_spectrum() 统一入口。")
        elif bool(globals().get("SPECTROMETER_TCP_DISABLED", False)):
            self.context["tcp_started"] = False
            self.context["labview_ready"] = True
            self.log("[流程] SPECTROMETER_TCP_DISABLED=True：跳过完整测量前的 TCP/READY 检查")
        else:
            if self.tcp_server is None or not bool(self.context.get("tcp_started", False)):
                self.start_tcp_server()
            if not bool(self.context.get("labview_ready", False)):
                self.wait_labview_ready()
            self.log("[流程] LabVIEW TCP 已启动并完成 READY 检查")

        self.is_measuring = True
        self.stop_requested = False
        self.context["is_measuring"] = True
        self.notify_update()

        for cycle_index in range(1, self.cfg.max_cycles + 1):
            if not self.is_measuring or self.stop_requested:
                break

            should_continue = self.run_one_cycle(cycle_index)

            if not should_continue:
                break

        self.finish()

    def run_simple_angle_loop_one_cycle(self, cycle_index: int, state: Optional[CalibrationState] = None) -> bool:
        """
        简单角度循环单轮：只截图/分割当前帧 Bmask，并用 Bmask 最长边计算角度。
        不连接/控制 Newport，不执行 RuleAB 推动，不使用 KLT/Profile。
        """
        self.context["cycle_index"] = cycle_index
        try:
            follower = self.ensure_rule_ab_follower()
            if state is not None:
                self.apply_calibration_state(state)
            self.apply_runtime_rule_ab_params_to_follower(follower, reason=f"simple_angle_loop_bmask_longest_edge_{cycle_index}")
            self._ensure_rule_ab_feature_tracker_installed(follower, self._get_loaded_calibration_state())
            result = self.detect_step7_yolo_obb_angle_once(
                image_rgb=None,
                b_mask=None,
                label=f"simple_angle_loop_cycle_{cycle_index:04d}",
                baseline_angle=None,
                allow_fail=True,
                follower=follower,
            )
            self.context["last_angle_result"] = self._json_safe(result)
            self.context["angle_before"] = result.get("angle_deg") if result.get("ok") else None
            self.context["save_angle_deg"] = result.get("angle_deg") if result.get("ok") else None
            self.notify_update()
            return bool(result.get("ok", False))
        except Exception as e:
            self.log(f"[SimpleAngleLoop-Bmask最长边] 第 {cycle_index} 轮失败：{e}")
            self.context["last_angle_result"] = {"ok": False, "reason": str(e), "angle_source": "bmask_longest_edge_failed"}
            self.notify_update()
            return False

    def request_stop(self):
        self.stop_requested = True
        self.is_measuring = False
        self.log("[流程] 收到停止请求")

        # 停止测量后，兜底关闭激光；当前版本不再操作 Rigol。
        try:
            if self.context.get("laser_on", False):
                self.laser_off()
                self.log("[流程] 停止测量：激光已 OFF")
        except Exception as e:
            self.log(f"[流程] 停止测量时关闭激光失败：{e}")

        # 兜底：将当前未完成轮的已有数据刷入 measurement_summary.csv，防止数据丢失
        try:
            self._flush_unsaved_cycle_to_csv()
        except Exception as e:
            self.log(f"[流程] 停止测量时刷新未保存 CSV 失败：{e}")

        self.save_summary_xlsx()
        self.notify_update()

    def reset_runtime_state_for_new_calibration(self, close_followers: bool = True):
        """
        停止/失败/关闭设备之后，允许重新标定 A/B/C/Step9 并重新运行完整测量。

        只重置“运行态”和“停止态”，不删除 calibration/current_calibration.json。
        目的：
            1. 清掉 stop_requested / step9_stop_requested，避免停止后再点 Step9 目标可以选，
               但进入颜色/面积选择时被 _capture_current_rule_ab_frame() 的 stop_requested 拦截；
            2. 清掉 full_measurement_mode / strict_full_calibration_c_locked，避免上一轮完整测量
               复制出来的 full_calibration_static_c 继续锁住下一次重新标定；
            3. 关闭并释放上一轮 RuleAB/RuleAC/Stage12 缓存，避免旧 SAM2 prompt、旧 mask、
               旧 C 路线污染新一轮。
        """
        self.stop_requested = False
        self.is_measuring = False
        self.step9_stop_requested = False
        self.pause_for_recalibration_requested = False
        self.recalibration_in_progress = False
        self.resume_after_recalibration_requested = False
        self.restart_current_cycle_after_recalibration = False
        self.midrun_recalibration_cycle_index = None

        self.context["is_measuring"] = False
        self.context["midrun_recalibration_requested"] = False
        self.context["midrun_recalibration_in_progress"] = False
        self.context["restart_current_cycle_after_recalibration"] = False
        self.context["midrun_recalibration_cycle_index"] = None
        self.context["full_measurement_mode"] = False
        self.context["strict_full_calibration_c_locked"] = False
        self.context["strict_full_calibration_c_dir"] = ""
        self.context["strict_full_calibration_c_mask_path"] = ""
        self.context["rule_ab_failed_stop"] = False
        self.context["rule_ab_failed_reason"] = ""
        self.context["last_rule_ab_result"] = None
        self.context["last_rule_ac_result"] = None

        # 清 Step9 运行缓存，但不清 GUI 已输入/标定文件里的 target/HSV。
        try:
            self.reset_step9_tracking_state(clear_b_points=True)
        except Exception:
            pass
        self.step9_color_last_mask = None
        self.step9_color_last_center = None
        self.step9_current_run_dir = None
        self.step9_history_csv_path = None

        # 清 B 边/角度运行态。
        self._reset_b_edge_runtime_state()

        if close_followers:
            for attr, label in (
                ("rule_ab_follower", "RuleAB"),
                ("rule_ac_controller", "RuleAC"),
                ("stage12_device", "Step9 Stage12"),
            ):
                obj = getattr(self, attr, None)
                if obj is None:
                    continue
                try:
                    if hasattr(obj, "close"):
                        obj.close()
                except Exception as e:
                    self.log(f"[运行态重置] 关闭 {label} 失败：{e}")
                try:
                    setattr(self, attr, None)
                except Exception:
                    pass

        self.log("[运行态重置] 已清理上一轮停止/失败后的运行缓存；可以重新标定并重新运行完整测量。")
        self.notify_update()

    def finish(self):
        self.log("========== 结束测量 ==========")
        self.is_measuring = False
        self.context["is_measuring"] = False

        # 当前版本不再使用 Rigol 信号发生器；流程结束时只兜底关闭激光。

        # 测量结束时，兜底关闭激光。
        try:
            if self.context.get("laser_on", False):
                self.laser_off()
                self.log("[流程] 测量结束，激光已 OFF")
        except Exception as e:
            self.log(f"[流程] 测量结束时关闭激光失败：{e}")

        # 照明光不在本次修改范围内，仍保持测量流程结束时的当前状态。
        try:
            if self.light is not None:
                self.log("[流程] 测量结束，照明光保持当前状态，不额外发送 ON/OFF")
        except Exception as e:
            self.log(f"[流程] 照明光保持状态提示失败：{e}")

        self.save_summary_xlsx()

        self.log("[流程] 正在测量 = False")
        self.notify_update()

    def close_all(self):
        """
        关闭硬件安全相关对象，但保留 LabVIEW TCP Server。

        当前要求：
            1. 关闭运动设备、激光、照明等硬件安全相关对象；
            2. 不关闭 LabVIEW TCP Server；
            3. 不把 self.tcp_server 置 None；
            4. 不把 context["tcp_started"] / context["labview_ready"] 改成 False；
            5. TCP 只由 GUI 的“关闭TCP”按钮单独关闭。
        """
        self.log("========== 关闭全部设备：关闭硬件安全相关对象，保留 LabVIEW TCP ==========")
        self.save_summary_xlsx()

        try:
            if self.light is not None:
                try:
                    if bool(self.context.get("light_on", False)):
                        self.light_off()
                except Exception as e:
                    self.log(f"[照明光] 关闭全部设备时 OFF 失败：{e}")
                try:
                    if hasattr(self.light, "close"):
                        self.light.close()
                    elif hasattr(self.light, "release"):
                        self.light.release()
                except Exception as e:
                    self.log(f"[照明光] 关闭全部设备时释放连接失败：{e}")
                self.light = None
                self.context["light_on"] = False
                self.log("[照明光] 关闭全部设备时已关闭/释放照明光对象")
        except Exception as e:
            self.log(f"[照明光] 关闭全部设备时处理失败：{e}")

        try:
            if self.laser_stage is not None:
                if self.context.get("laser_on", False):
                    self.laser_off()
                self.laser_stage.close()
                self.laser_stage = None
                self.context["laser_on"] = False
                self.log("[激光开关] GUI关闭时已关闭 Newport 控制器连接")
        except Exception as e:
            self.log(f"[激光开关] GUI关闭时关闭 Newport 控制器失败：{e}")

        # 注意：关闭全部设备不再关闭 LabVIEW TCP Server。
        # TCP 连接保持可复用，下一次完整循环测量可直接使用；
        # 只有点击 GUI 的“关闭TCP”按钮时才会真正关闭 TCP。
        if self.tcp_server is not None:
            self.log("[TCP] 关闭全部设备：保留 LabVIEW TCP Server，不关闭、不置 None、不修改 READY 状态")

        try:
            if self.rule_ab_follower is not None:
                self.rule_ab_follower.close()
                self.rule_ab_follower = None
                self.log("[RuleAB] 已关闭")
        except Exception as e:
            self.log(f"[RuleAB] 关闭失败：{e}")

        try:
            if self.rule_ac_controller is not None:
                self.rule_ac_controller.close()
                self.rule_ac_controller = None
                self.log("[RuleAC] 已关闭")
        except Exception as e:
            self.log(f"[RuleAC] 关闭失败：{e}")

        try:
            if self.stage12_device is not None:
                self.stage12_device.close()
                self.stage12_device = None
                self.log("[Step9-颜色中心] Stage12 控制器已关闭")
        except Exception as e:
            self.log(f"[Step9-颜色中心] Stage12 控制器关闭失败：{e}")

        # 关闭 Focus 组件（包含 Z 轴 Picomotor），防止下次补焦启动时 USB 连接冲突
        try:
            if self._focus_controller is not None:
                try:
                    if hasattr(self._focus_controller, "close"):
                        self._focus_controller.close()
                except Exception as e:
                    self.log(f"[聚焦] 关闭控制器失败：{e}")
                self._focus_controller = None
            if self._focus_metrics_calc is not None:
                try:
                    self._focus_metrics_calc.close()
                except Exception as e:
                    self.log(f"[聚焦] 关闭 metrics_calc 失败：{e}")
                self._focus_metrics_calc = None
            if self._focus_scorer is not None:
                self._focus_scorer = None
            if self._focus_simulator is not None:
                self._focus_simulator = None
            self.log("[聚焦] 已释放所有 Focus 组件")
        except Exception as e:
            self.log(f"[聚焦] 释放组件失败：{e}")

        # 关闭全部设备后要允许重新标定/重新完整测量；
        # request_stop() 会把 stop_requested=True，如果不清掉，后续 Step9 颜色/面积选择会被截图函数直接拦截。
        self.reset_runtime_state_for_new_calibration(close_followers=False)

        # 关闭日志管理器
        try:
            if self._log_manager is not None:
                self._shutdown_log_manager()
                self.log("[日志] 日志管理器已关闭")
        except Exception as e:
            print(f"[日志] 关闭日志管理器失败：{e}")

        self.log("========== 关闭全部设备完成：运动/激光/照明/聚焦等硬件安全对象已处理，LabVIEW TCP 保持原状态；运行态已重置 ==========")
        self.notify_update()

# ============================================================
# GUI
# ============================================================

class MeasurementWorkflowGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("循环测量 GUI：可切换 real/virtual 硬件 + YOLO-OBB角度检测 + LabVIEW/虚拟光谱")
        self.root.geometry("1320x1080")

        self.workflow: Optional[MeasurementWorkflow] = None
        self.worker_thread: Optional[threading.Thread] = None
        self.is_busy = False

        # 单独 RuleAB / A沿C绕行线程的停止标志。
        # “停止测量”和“停止A推B控制器”都会把它置 True。
        self.rule_ab_only_stop_requested = False

        # 当前正在运行的 RuleAB follower 引用。
        # 供“停止A推B控制器”按钮在任意时刻直接调用 stage.stop_all()。
        self.rule_ab_active_follower: Optional[ActualNanoBoundaryFollower] = None
        self.rule_ab_controller_stop_requested = False

        # 光谱补焦循环状态
        self.spectrum_autofocus_loop: Optional[SpectrumAutofocusLoop] = None
        self.spectrum_autofocus_thread: Optional[threading.Thread] = None
        self.spectrum_autofocus_stop_requested = False

        self.fig: Optional[Figure] = None
        self.ax_fit_peak = None
        self.ax_raw = None
        self.ax_median = None
        self.plot_canvas: Optional[FigureCanvasTkAgg] = None
        self.plot_status_var = tk.StringVar(value="图像显示：暂无数据")
        self.focus_preview_window: Optional[tk.Toplevel] = None
        self.focus_preview_fig: Optional[Figure] = None
        self.focus_preview_ax = None
        self.focus_preview_canvas: Optional[FigureCanvasTkAgg] = None
        self.focus_preview_status_var = tk.StringVar(value="FocusScore预览：未启动")
        self.focus_preview_history: List[Tuple[float, float]] = []
        self.focus_preview_started_at: Optional[float] = None
        self.focus_preview_after_id: Optional[str] = None
        self.focus_preview_running = False
        self.focus_preview_sample_inflight = False

        # 右侧列表：原“角度-拟合峰值图”的横纵坐标数据
        self.angle_fit_tree = None
        self.angle_fit_list_status_var = tk.StringVar(value="角度-拟合峰值列表：暂无数据")

        # 信号ON时间输入框的运行时同步策略：
        # 1. 默认情况下，下一轮使用程序按角度差计算出的 wf.cfg.signal_on_time_ms；
        # 2. 如果用户在测量过程中手动修改“信号ON/ms”，下一轮优先使用用户修改值；
        # 3. 程序自动把计算值写回输入框时，不应被误判为用户手动修改。
        self.signal_on_time_user_modified = False
        self._programmatic_signal_on_time_update = False
        self.default_cfg = MeasurementConfig()
        self.hardware_mode_var = tk.StringVar(value=str(self.default_cfg.hardware_mode))
        self.virtual_spectrum_mode_var = tk.StringVar(value=str(self.default_cfg.virtual_spectrum_mode))
        self.virtual_spectrum_replay_csv_var = tk.StringVar(value=str(self.default_cfg.virtual_spectrum_replay_csv))
        self._tk_thread_ident = threading.get_ident()
        self.calibration_status_var = tk.StringVar(value="完整测量标定：未加载")
        self.calibration_path_var = tk.StringVar(value="标定文件：未设置")

        self._build_ui()

    # --------------------------------------------------------
    # UI 构建
    # --------------------------------------------------------

    def _build_ui(self):
        """
        重新整理 GUI 布局：
            左侧：流程按钮、参数设置、设备控制；
            中间：图像显示、运行日志；
            右侧：角度-拟合峰值横纵坐标列表。

        这样做的目的：
            1. 左侧高频操作按钮和参数区保持原结构；
            2. 中间集中显示原始数据图、中值滤波图和运行日志；
            3. 右侧用列表替代原“角度-拟合峰值图”。
        """
        self.root.geometry("1500x950")
        self.root.minsize(1280, 820)

        # Tk/ttk 基础样式
        style = ttk.Style()
        try:
            style.configure("Title.TLabel", font=("Microsoft YaHei", 11, "bold"))
            style.configure("Value.TLabel", font=("Microsoft YaHei", 10))
            style.configure("Primary.TButton", padding=(8, 4))
            style.configure("Danger.TButton", padding=(8, 4))
            style.configure("Panel.TLabelframe", padding=8)
            style.configure("Panel.TLabelframe.Label", font=("Microsoft YaHei", 10, "bold"))
        except Exception:
            pass

        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # =====================================================
        # 左侧：可滚动控制面板
        # =====================================================
        left_outer = ttk.Frame(main, width=500)
        center_panel = ttk.Frame(main)
        right_panel = ttk.Frame(main, width=360)
        main.add(left_outer, weight=1)
        main.add(center_panel, weight=3)
        main.add(right_panel, weight=1)

        left_canvas = tk.Canvas(left_outer, highlightthickness=0, width=500)
        left_scrollbar = ttk.Scrollbar(left_outer, orient=tk.VERTICAL, command=left_canvas.yview)
        left_inner = ttk.Frame(left_canvas)

        left_window = left_canvas.create_window((0, 0), window=left_inner, anchor="nw")
        left_canvas.configure(yscrollcommand=left_scrollbar.set)

        def _sync_left_scrollregion(event=None):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))
            left_canvas.itemconfigure(left_window, width=left_canvas.winfo_width())

        left_inner.bind("<Configure>", _sync_left_scrollregion)
        left_canvas.bind("<Configure>", _sync_left_scrollregion)

        def _on_mousewheel(event):
            # Windows: event.delta 通常是 ±120
            left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        left_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # =====================================================
        # 左侧 1：流程控制
        # =====================================================
        flow_frame = ttk.LabelFrame(left_inner, text="1. 自动循环测量流程控制", padding=10, style="Panel.TLabelframe")
        flow_frame.pack(fill=tk.X, padx=4, pady=(0, 8))

        self.flow_status_var = tk.StringVar(value="流程状态：未初始化")

        button_grid = ttk.Frame(flow_frame)
        button_grid.pack(fill=tk.X)
        for i in range(2):
            button_grid.columnconfigure(i, weight=1)

        ttk.Button(button_grid, text="初始化全部设备", command=self.init_all_devices_thread, style="Primary.TButton").grid(row=0, column=0, padx=4, pady=4, sticky="ew")
        ttk.Button(button_grid, text="运行完整循环测量", command=self.run_workflow_thread, style="Primary.TButton").grid(row=0, column=1, padx=4, pady=4, sticky="ew")
        ttk.Button(button_grid, text="停止测量", command=self.request_stop, style="Danger.TButton").grid(row=1, column=0, padx=4, pady=4, sticky="ew")
        ttk.Button(button_grid, text="关闭全部设备", command=self.close_all_thread).grid(row=1, column=1, padx=4, pady=4, sticky="ew")
        ttk.Button(button_grid, text="2 标定RuleAB-A", command=self.calibrate_rule_ab_a_thread).grid(row=2, column=0, padx=4, pady=4, sticky="ew")
        ttk.Button(button_grid, text="3 标定全局C", command=self.calibrate_global_c_thread).grid(row=2, column=1, padx=4, pady=4, sticky="ew")
        ttk.Button(button_grid, text="4a 标定Step9目标", command=self.calibrate_step9_target_thread).grid(row=4, column=0, padx=4, pady=4, sticky="ew")
        ttk.Button(button_grid, text="4b 标定Step9颜色", command=self.calibrate_step9_color_thread).grid(row=4, column=1, padx=4, pady=4, sticky="ew")
        ttk.Button(button_grid, text="保存/检查完整标定", command=self.save_and_check_calibration_thread).grid(row=5, column=0, padx=4, pady=4, sticky="ew")
        ttk.Button(button_grid, text="加载完整标定", command=self.load_calibration_to_gui_thread).grid(row=5, column=1, padx=4, pady=4, sticky="ew")
        ttk.Button(button_grid, text="重置后重新标定", command=self.reset_after_stop_for_recalibration_thread).grid(row=6, column=0, padx=4, pady=4, sticky="ew")
        ttk.Button(button_grid, text="暂停运动并重新标定ABC", command=self.pause_midrun_recalibration_thread).grid(row=6, column=1, padx=4, pady=4, sticky="ew")
        ttk.Button(button_grid, text="完成重标定并继续测量", command=self.resume_midrun_recalibration_thread).grid(row=7, column=0, columnspan=2, padx=4, pady=4, sticky="ew")
        ttk.Button(button_grid, text="Step7 A/C 标定预检", command=self.step7_ac_preflight_thread).grid(row=8, column=0, columnspan=2, padx=4, pady=4, sticky="ew")

        ttk.Label(flow_frame, textvariable=self.flow_status_var, wraplength=460).pack(fill=tk.X, padx=4, pady=(6, 0))
        ttk.Label(flow_frame, textvariable=self.calibration_status_var, wraplength=460, style="Value.TLabel").pack(fill=tk.X, padx=4, pady=(4, 0))
        ttk.Label(flow_frame, textvariable=self.calibration_path_var, wraplength=460, style="Value.TLabel").pack(fill=tk.X, padx=4, pady=(2, 0))

        hardware_frame = ttk.LabelFrame(left_inner, text="硬件模式", padding=10, style="Panel.TLabelframe")
        hardware_frame.pack(fill=tk.X, padx=4, pady=(0, 8))
        hardware_frame.columnconfigure(1, weight=1)
        ttk.Label(hardware_frame, text="hardware_mode").grid(row=0, column=0, padx=4, pady=3, sticky="w")
        ttk.Combobox(hardware_frame, textvariable=self.hardware_mode_var, values=["virtual", "real"], state="readonly", width=16).grid(row=0, column=1, padx=4, pady=3, sticky="ew")
        ttk.Label(hardware_frame, text="virtual spectrum").grid(row=1, column=0, padx=4, pady=3, sticky="w")
        ttk.Combobox(hardware_frame, textvariable=self.virtual_spectrum_mode_var, values=["gaussian", "replay_csv"], state="readonly", width=16).grid(row=1, column=1, padx=4, pady=3, sticky="ew")
        ttk.Label(hardware_frame, text="replay CSV").grid(row=2, column=0, padx=4, pady=3, sticky="w")
        ttk.Entry(hardware_frame, textvariable=self.virtual_spectrum_replay_csv_var).grid(row=2, column=1, padx=4, pady=3, sticky="ew")
        ttk.Label(hardware_frame, text="virtual：不连接 Newport/Thorlabs/LabVIEW；real：连接真实硬件", wraplength=460).grid(row=3, column=0, columnspan=2, padx=4, pady=3, sticky="w")

        # 流程状态后面：当前测量结果
        result_frame = ttk.LabelFrame(left_inner, text="当前测量结果", padding=10, style="Panel.TLabelframe")
        result_frame.pack(fill=tk.X, padx=4, pady=(0, 8))

        self.current_cycle_var = tk.StringVar(value="cycle：0")
        self.current_angle_before_var = tk.StringVar(value="本轮角度：None")
        self.current_angle_after_var = tk.StringVar(value="angle_after：单角度模式未使用")
        self.current_angle_delta_var = tk.StringVar(value="跨轮angle_delta：None")
        self.current_signal_time_var = tk.StringVar(value="下一轮信号ON时间/ms：None")
        self.current_raw_peak_var = tk.StringVar(value="raw_peak：None")
        self.current_fit_peak_var = tk.StringVar(value="fit_peak：None")
        self.current_save_path_var = tk.StringVar(value="保存路径：None")

        # 改成单列多行布局，避免多个长字符串挤在同一行导致被控件覆盖。
        result_frame.columnconfigure(0, weight=1)
        result_labels = [
            self.current_cycle_var,
            self.current_angle_before_var,
            self.current_angle_after_var,
            self.current_angle_delta_var,
            self.current_signal_time_var,
            self.current_raw_peak_var,
            self.current_fit_peak_var,
            self.current_save_path_var,
        ]
        for row_idx, var in enumerate(result_labels):
            ttk.Label(
                result_frame,
                textvariable=var,
                style="Value.TLabel",
                wraplength=460,
                justify=tk.LEFT,
            ).grid(row=row_idx, column=0, padx=6, pady=3, sticky="ew")



        # =====================================================
        # 左侧 2：基本测量参数
        # =====================================================
        basic_frame = ttk.LabelFrame(left_inner, text="2. 基本测量参数", padding=10, style="Panel.TLabelframe")
        basic_frame.pack(fill=tk.X, padx=4, pady=8)

        cfg0 = self.default_cfg
        self.max_cycles_var = tk.IntVar(value=cfg0.max_cycles)
        self.sub_loop_iterations_per_cycle_var = tk.IntVar(value=cfg0.sub_loop_iterations_per_cycle)
        self.signal_on_time_ms_var = tk.DoubleVar(value=cfg0.signal_on_time_ms)
        self.signal_on_time_ms_var.trace_add("write", self._on_signal_on_time_var_changed)
        self.stable_wait_ms_var = tk.IntVar(value=cfg0.stable_wait_ms)
        self.signal_time_factor_var = tk.DoubleVar(value=cfg0.signal_time_factor)
        self.angle_delta_min_var = tk.DoubleVar(value=cfg0.angle_delta_min_deg)
        self.angle_delta_max_var = tk.DoubleVar(value=cfg0.angle_delta_max_deg)
        self.num_var = tk.IntVar(value=cfg0.angle_num)
        self.cw_var = tk.IntVar(value=cfg0.angle_cw)
        self.save_root_var = tk.StringVar(value=cfg0.save_root)
        self.raw_remove_above_var = tk.DoubleVar(value=cfg0.raw_remove_above)
        self.median_filter_window_var = tk.IntVar(value=cfg0.median_filter_window)
        self.x_axis_xlsx_path_var = tk.StringVar(value=cfg0.x_axis_xlsx_path)

        # 基本参数区改成“标签列 + 输入列”的两组布局。
        # 这样比原先四列硬挤更稳定，也便于把角度检测 num/cw 放进同一区域。
        for col in (1, 3):
            basic_frame.columnconfigure(col, weight=1)
        for col in (0, 2):
            basic_frame.columnconfigure(col, weight=0)

        ttk.Label(basic_frame, text="循环次数").grid(row=0, column=0, padx=(4, 6), pady=4, sticky="w")
        ttk.Entry(basic_frame, textvariable=self.max_cycles_var, width=14).grid(row=0, column=1, padx=(0, 10), pady=4, sticky="ew")
        ttk.Label(basic_frame, text="信号ON/ms").grid(row=0, column=2, padx=(4, 6), pady=4, sticky="w")
        ttk.Entry(basic_frame, textvariable=self.signal_on_time_ms_var, width=14).grid(row=0, column=3, padx=(0, 4), pady=4, sticky="ew")

        ttk.Label(basic_frame, text="稳定等待/ms").grid(row=1, column=0, padx=(4, 6), pady=4, sticky="w")
        ttk.Entry(basic_frame, textvariable=self.stable_wait_ms_var, width=14).grid(row=1, column=1, padx=(0, 10), pady=4, sticky="ew")
        ttk.Label(basic_frame, text="倍乘系数").grid(row=1, column=2, padx=(4, 6), pady=4, sticky="w")
        ttk.Entry(basic_frame, textvariable=self.signal_time_factor_var, width=14).grid(row=1, column=3, padx=(0, 10), pady=4, sticky="ew")
        
        ttk.Label(basic_frame, text="角度差下限/deg").grid(row=2, column=0, padx=(4, 6), pady=4, sticky="w")
        ttk.Entry(basic_frame, textvariable=self.angle_delta_min_var, width=14).grid(row=2, column=1, padx=(0, 4), pady=4, sticky="ew")
        ttk.Label(basic_frame, text="角度差上限/deg").grid(row=2, column=2, padx=(4, 6), pady=4, sticky="w")
        ttk.Entry(basic_frame, textvariable=self.angle_delta_max_var, width=14).grid(row=2, column=3, padx=(0, 10), pady=4, sticky="ew")

        ttk.Label(basic_frame, text="角度检测 num").grid(row=3, column=0, padx=(4, 6), pady=4, sticky="w")
        ttk.Entry(basic_frame, textvariable=self.num_var, width=14).grid(row=3, column=1, padx=(0, 4), pady=4, sticky="ew")
        ttk.Label(basic_frame, text="角度检测 cw").grid(row=3, column=2, padx=(4, 6), pady=4, sticky="w")
        ttk.Entry(basic_frame, textvariable=self.cw_var, width=14).grid(row=3, column=3, padx=(0, 10), pady=4, sticky="ew")

        ttk.Label(basic_frame, text="原始数据去除阈值").grid(row=4, column=0, padx=(4, 6), pady=4, sticky="w")
        ttk.Entry(basic_frame, textvariable=self.raw_remove_above_var, width=14).grid(row=4, column=1, padx=(0, 4), pady=4, sticky="ew")
        ttk.Label(basic_frame, text="中值滤波窗口").grid(row=4, column=2, padx=(4, 6), pady=4, sticky="w")
        ttk.Entry(basic_frame, textvariable=self.median_filter_window_var, width=14).grid(row=4, column=3, padx=(0, 10), pady=4, sticky="ew")

        ttk.Label(basic_frame, text="子循环次数").grid(row=5, column=0, padx=(4, 6), pady=4, sticky="w")
        ttk.Entry(basic_frame, textvariable=self.sub_loop_iterations_per_cycle_var, width=14).grid(row=5, column=1, padx=(0, 10), pady=4, sticky="ew")

        ttk.Separator(basic_frame, orient=tk.HORIZONTAL).grid(row=6, column=0, columnspan=4, padx=4, pady=(8, 6), sticky="ew")

        ttk.Label(basic_frame, text="保存目录").grid(row=7, column=0, padx=(4, 6), pady=4, sticky="w")
        ttk.Entry(basic_frame, textvariable=self.save_root_var).grid(row=7, column=1, columnspan=2, padx=(0, 10), pady=4, sticky="ew")
        ttk.Button(basic_frame, text="选择", command=self.select_save_root, width=8).grid(row=7, column=3, padx=(0, 4), pady=4, sticky="ew")

        ttk.Label(basic_frame, text="图2/图3横坐标xlsx").grid(row=8, column=0, padx=(4, 6), pady=4, sticky="w")
        ttk.Entry(basic_frame, textvariable=self.x_axis_xlsx_path_var).grid(row=8, column=1, columnspan=2, padx=(0, 10), pady=4, sticky="ew")
        ttk.Button(basic_frame, text="选择", command=self.select_x_axis_xlsx, width=8).grid(row=8, column=3, padx=(0, 4), pady=4, sticky="ew")

        # =====================================================
        # 左侧 3：照明光
        # =====================================================
        light_frame = ttk.LabelFrame(left_inner, text="3. 照明光控制", padding=10, style="Panel.TLabelframe")
        light_frame.pack(fill=tk.X, padx=4, pady=8)

        self.light_port_var = tk.StringVar(value=cfg0.light_port)
        self.light_status_var = tk.StringVar(value="照明光状态：未连接")

        light_frame.columnconfigure(1, weight=1)
        ttk.Label(light_frame, text="串口").grid(row=0, column=0, padx=4, pady=4, sticky="w")
        ttk.Entry(light_frame, textvariable=self.light_port_var, width=14).grid(row=0, column=1, padx=4, pady=4, sticky="ew")
        ttk.Button(light_frame, text="单独连接", command=self.connect_light_thread).grid(row=0, column=2, padx=4, pady=4, sticky="ew")

        light_buttons = ttk.Frame(light_frame)
        light_buttons.grid(row=1, column=0, columnspan=3, padx=2, pady=4, sticky="ew")
        light_buttons.columnconfigure(0, weight=1)
        light_buttons.columnconfigure(1, weight=1)
        ttk.Button(light_buttons, text="ON", command=self.light_on_thread).grid(row=0, column=0, padx=4, pady=2, sticky="ew")
        ttk.Button(light_buttons, text="OFF", command=self.light_off_thread).grid(row=0, column=1, padx=4, pady=2, sticky="ew")
        ttk.Label(light_frame, textvariable=self.light_status_var, wraplength=450).grid(row=2, column=0, columnspan=3, padx=4, pady=(4, 0), sticky="w")

        # =====================================================
        # 左侧 4：Newport 激光开关控制器
        # =====================================================
        rigol_frame = ttk.LabelFrame(left_inner, text="4. Newport 8743-CL 激光开关控制器", padding=10, style="Panel.TLabelframe")
        rigol_frame.pack(fill=tk.X, padx=4, pady=8)

        self.rigol_visa_var = tk.StringVar(value=cfg0.rigol_visa)
        self.rigol_status_var = tk.StringVar(value="激光控制器状态：未连接")

        rigol_frame.columnconfigure(1, weight=1)
        ttk.Label(rigol_frame, text="原Rigol VISA/现保留不用").grid(row=0, column=0, padx=4, pady=4, sticky="w")
        ttk.Entry(rigol_frame, textvariable=self.rigol_visa_var).grid(row=0, column=1, columnspan=2, padx=4, pady=4, sticky="ew")

        rigol_buttons = ttk.Frame(rigol_frame)
        rigol_buttons.grid(row=1, column=0, columnspan=3, padx=2, pady=4, sticky="ew")
        for i in range(2):
            rigol_buttons.columnconfigure(i, weight=1)
        ttk.Button(rigol_buttons, text="连接激光控制器", command=self.connect_rigol_thread).grid(row=0, column=0, padx=4, pady=3, sticky="ew")
        ttk.Button(rigol_buttons, text="设置激光轴速度/加速度", command=self.configure_rigol_thread).grid(row=0, column=1, padx=4, pady=3, sticky="ew")
        ttk.Button(rigol_buttons, text="激光 ON", command=self.signal_ch1_on_thread).grid(row=1, column=0, padx=4, pady=3, sticky="ew")
        ttk.Button(rigol_buttons, text="激光 OFF", command=self.signal_ch1_off_thread).grid(row=1, column=1, padx=4, pady=3, sticky="ew")
        ttk.Button(rigol_buttons, text="CH2未使用", command=self.signal_ch2_on_thread).grid(row=2, column=0, padx=4, pady=3, sticky="ew")
        ttk.Button(rigol_buttons, text="CH2保持关闭", command=self.signal_ch2_off_thread).grid(row=2, column=1, padx=4, pady=3, sticky="ew")
        ttk.Label(rigol_frame, textvariable=self.rigol_status_var, wraplength=450).grid(row=2, column=0, columnspan=3, padx=4, pady=(4, 0), sticky="w")

        # =====================================================
        # 左侧 5：角度检测
        # =====================================================
        angle_frame = ttk.LabelFrame(left_inner, text="5. YOLO-OBB 长边角度检测", padding=10, style="Panel.TLabelframe")
        angle_frame.pack(fill=tk.X, padx=4, pady=8)

        self.angle_model_path_var = tk.StringVar(value=cfg0.angle_model_path)
        self.capture_area_var = tk.StringVar(value=",".join(str(v) for v in cfg0.capture_area))
        self.angle_status_var = tk.StringVar(value="角度检测状态：未初始化")
        self.angle_result_var = tk.StringVar(value="角度结果：None")

        angle_frame.columnconfigure(1, weight=1)
        ttk.Label(angle_frame, text="YOLO-OBB .pt模型").grid(row=0, column=0, padx=4, pady=4, sticky="w")
        ttk.Entry(angle_frame, textvariable=self.angle_model_path_var).grid(row=0, column=1, padx=4, pady=4, sticky="ew")
        ttk.Button(angle_frame, text="选择模型", command=self.select_angle_model).grid(row=0, column=2, padx=4, pady=4, sticky="ew")

        ttk.Label(angle_frame, text="截图区域").grid(row=1, column=0, padx=4, pady=4, sticky="w")
        ttk.Entry(angle_frame, textvariable=self.capture_area_var).grid(row=1, column=1, padx=4, pady=4, sticky="ew")

        angle_frame.columnconfigure(3, weight=1)

        angle_buttons = ttk.Frame(angle_frame)
        angle_buttons.grid(row=3, column=0, columnspan=4, padx=2, pady=4, sticky="ew")
        angle_buttons.columnconfigure(0, weight=1)
        angle_buttons.columnconfigure(1, weight=1)
        ttk.Button(angle_buttons, text="初始化角度模块", command=self.init_angle_thread).grid(row=0, column=0, padx=4, pady=3, sticky="ew")
        ttk.Button(angle_buttons, text="单次角度检测", command=self.detect_angle_once_thread).grid(row=0, column=1, padx=4, pady=3, sticky="ew")

        ttk.Label(angle_frame, textvariable=self.angle_status_var, wraplength=450).grid(row=5, column=0, columnspan=4, padx=4, pady=(4, 0), sticky="w")
        ttk.Label(angle_frame, textvariable=self.angle_result_var, wraplength=450).grid(row=6, column=0, columnspan=4, padx=4, pady=(4, 0), sticky="w")

        # =====================================================
        # 左侧 6：LabVIEW TCP 通信
        # =====================================================
        tcp_frame = ttk.LabelFrame(left_inner, text="6. 光谱仪通信", padding=10, style="Panel.TLabelframe")
        tcp_frame.pack(fill=tk.X, padx=4, pady=8)

        self.tcp_host_var = tk.StringVar(value=cfg0.tcp_host)
        self.tcp_port_var = tk.IntVar(value=cfg0.tcp_port)
        self.tcp_output_dir_var = tk.StringVar(value=cfg0.tcp_output_dir)
        self.tcp_command_var = tk.StringVar(value=cfg0.tcp_command)
        self.tcp_status_var = tk.StringVar(value="TCP状态：未启动")
        self.tcp_result_var = tk.StringVar(value="最近光谱：None")

        # 日志保存配置
        self.save_log_to_file_var = tk.BooleanVar(value=getattr(cfg0, "save_log_to_file", True))
        self.log_dir_var = tk.StringVar(value=getattr(cfg0, "log_dir", "./Log"))

        # 新增：光谱仪设备选择
        self.spectrometer_backend_var = tk.StringVar(value=cfg0.spectrometer_backend)
        self.picam_exposure_var = tk.DoubleVar(value=cfg0.picam_exposure)
        self.picam_temperature_var = tk.DoubleVar(value=cfg0.picam_temperature)
        self.picam_roi_width_var = tk.IntVar(value=cfg0.picam_roi_width)
        self.picam_roi_height_var = tk.IntVar(value=cfg0.picam_roi_height)

        for i in range(4):
            tcp_frame.columnconfigure(i, weight=1)

        # 新增：光谱仪设备选择下拉框
        ttk.Label(tcp_frame, text="设备后端").grid(row=0, column=0, padx=4, pady=4, sticky="w")
        backend_combo = ttk.Combobox(
            tcp_frame,
            textvariable=self.spectrometer_backend_var,
            values=["labview_tcp", "picam", "picam_demo"],
            state="readonly",
            width=12,
        )
        backend_combo.grid(row=0, column=1, padx=4, pady=4, sticky="ew")
        backend_combo.bind("<<ComboboxSelected>>", self._on_spectrometer_backend_changed)

        ttk.Label(tcp_frame, text="HOST").grid(row=1, column=0, padx=4, pady=4, sticky="w")
        ttk.Entry(tcp_frame, textvariable=self.tcp_host_var).grid(row=1, column=1, padx=4, pady=4, sticky="ew")
        ttk.Label(tcp_frame, text="PORT").grid(row=1, column=2, padx=4, pady=4, sticky="w")
        ttk.Entry(tcp_frame, textvariable=self.tcp_port_var).grid(row=1, column=3, padx=4, pady=4, sticky="ew")

        ttk.Label(tcp_frame, text="命令").grid(row=2, column=0, padx=4, pady=4, sticky="w")
        ttk.Entry(tcp_frame, textvariable=self.tcp_command_var).grid(row=2, column=1, padx=4, pady=4, sticky="ew")
        ttk.Label(tcp_frame, text="CSV目录").grid(row=2, column=2, padx=4, pady=4, sticky="w")
        ttk.Entry(tcp_frame, textvariable=self.tcp_output_dir_var).grid(row=2, column=3, padx=4, pady=4, sticky="ew")

        # 新增：PI 光谱仪参数（仅在 PI 后端时启用）
        pi_frame = ttk.Frame(tcp_frame)
        pi_frame.grid(row=3, column=0, columnspan=4, padx=2, pady=4, sticky="ew")
        for i in range(4):
            pi_frame.columnconfigure(i, weight=1)

        ttk.Label(pi_frame, text="曝光(s)").grid(row=0, column=0, padx=4, pady=2, sticky="w")
        self.picam_exposure_entry = ttk.Entry(pi_frame, textvariable=self.picam_exposure_var, width=8)
        self.picam_exposure_entry.grid(row=0, column=1, padx=4, pady=2, sticky="ew")
        ttk.Label(pi_frame, text="温度(°C)").grid(row=0, column=2, padx=4, pady=2, sticky="w")
        self.picam_temperature_entry = ttk.Entry(pi_frame, textvariable=self.picam_temperature_var, width=8)
        self.picam_temperature_entry.grid(row=0, column=3, padx=4, pady=2, sticky="ew")

        ttk.Label(pi_frame, text="ROI宽").grid(row=1, column=0, padx=4, pady=2, sticky="w")
        self.picam_roi_width_entry = ttk.Entry(pi_frame, textvariable=self.picam_roi_width_var, width=8)
        self.picam_roi_width_entry.grid(row=1, column=1, padx=4, pady=2, sticky="ew")
        ttk.Label(pi_frame, text="ROI高").grid(row=1, column=2, padx=4, pady=2, sticky="w")
        self.picam_roi_height_entry = ttk.Entry(pi_frame, textvariable=self.picam_roi_height_var, width=8)
        self.picam_roi_height_entry.grid(row=1, column=3, padx=4, pady=2, sticky="ew")

        # 初始状态：根据后端类型启用/禁用 PI 参数
        self._update_pi_controls_state()

        tcp_buttons = ttk.Frame(tcp_frame)
        tcp_buttons.grid(row=4, column=0, columnspan=4, padx=2, pady=4, sticky="ew")
        for i in range(2):
            tcp_buttons.columnconfigure(i, weight=1)
        ttk.Button(tcp_buttons, text="启动TCP", command=self.start_tcp_thread).grid(row=0, column=0, padx=4, pady=3, sticky="ew")
        ttk.Button(tcp_buttons, text="等待READY", command=self.wait_ready_thread).grid(row=0, column=1, padx=4, pady=3, sticky="ew")
        ttk.Button(tcp_buttons, text="单次光谱采集", command=self.measure_spectrum_once_thread).grid(row=1, column=0, padx=4, pady=3, sticky="ew")
        ttk.Button(tcp_buttons, text="关闭TCP", command=self.close_tcp_thread).grid(row=1, column=1, padx=4, pady=3, sticky="ew")
        ttk.Button(tcp_buttons, text="保存当前光谱数据（背景光）", command=self.save_current_spectrum_background_light_thread).grid(row=2, column=0, columnspan=2, padx=4, pady=3, sticky="ew")

        # 日志保存配置：复选框 + 日志目录输入框（独占一行，避免拥挤）
        log_save_frame = ttk.Frame(tcp_frame)
        log_save_frame.grid(row=5, column=0, columnspan=4, padx=2, pady=4, sticky="ew")
        log_save_frame.columnconfigure(0, weight=0)
        log_save_frame.columnconfigure(1, weight=0)
        log_save_frame.columnconfigure(2, weight=1)
        ttk.Checkbutton(
            log_save_frame, text="保存运行日志到文件", variable=self.save_log_to_file_var
        ).grid(row=0, column=0, padx=(4, 6), pady=2, sticky="w")
        ttk.Label(log_save_frame, text="日志目录:").grid(row=0, column=1, padx=(8, 2), pady=2, sticky="e")
        log_dir_entry = ttk.Entry(log_save_frame, textvariable=self.log_dir_var)
        log_dir_entry.grid(row=0, column=2, padx=2, pady=2, sticky="ew")
        self.log_dir_entry = log_dir_entry

        ttk.Label(tcp_frame, textvariable=self.tcp_status_var, wraplength=450).grid(row=6, column=0, columnspan=4, padx=4, pady=(4, 0), sticky="w")
        ttk.Label(tcp_frame, textvariable=self.tcp_result_var, wraplength=450).grid(row=7, column=0, columnspan=4, padx=4, pady=(4, 0), sticky="w")

        # =====================================================
        # 左侧 7：Δw 判断
        # =====================================================
        judge_frame = ttk.LabelFrame(left_inner, text="7. Δw 循环判断", padding=10, style="Panel.TLabelframe")
        judge_frame.pack(fill=tk.X, padx=4, pady=(8, 16))

        self.enable_delta_w_judge_var = tk.BooleanVar(value=cfg0.enable_delta_w_judge)
        self.delta_w_threshold_var = tk.DoubleVar(value=cfg0.delta_w_threshold)
        self.stop_when_delta_w_not_enough_var = tk.BooleanVar(value=cfg0.stop_when_delta_w_not_enough)
        self.delta_status_var = tk.StringVar(value="Δw状态：未启用")

        judge_frame.columnconfigure(1, weight=1)
        ttk.Checkbutton(judge_frame, text="启用 Δw 判断", variable=self.enable_delta_w_judge_var).grid(row=0, column=0, columnspan=2, padx=4, pady=4, sticky="w")
        ttk.Label(judge_frame, text="阈值 C").grid(row=1, column=0, padx=4, pady=4, sticky="w")
        ttk.Entry(judge_frame, textvariable=self.delta_w_threshold_var).grid(row=1, column=1, padx=4, pady=4, sticky="ew")
        ttk.Checkbutton(judge_frame, text="Δw不足时停止", variable=self.stop_when_delta_w_not_enough_var).grid(row=2, column=0, columnspan=2, padx=4, pady=4, sticky="w")
        ttk.Label(judge_frame, textvariable=self.delta_status_var, wraplength=450).grid(row=3, column=0, columnspan=2, padx=4, pady=(4, 0), sticky="w")

        # =====================================================
        # 左侧 8：Rule AB / Rule AC 单独测试
        # =====================================================
        rule_frame = ttk.LabelFrame(left_inner, text="8. Rule AB / Rule AC 单独测试", padding=10, style="Panel.TLabelframe")
        rule_frame.pack(fill=tk.X, padx=4, pady=(0, 16))

        self.rule_ab_enable_stage_var = tk.BooleanVar(value=True)

        # 单独 A推动B 不再使用最大推动次数；运行后会持续推动/监测，直到点击“停止测量”。
        # 这里保留内部变量只是为了兼容完整测量流程和旧配置，不在 GUI 中显示。
        self.rule_ab_max_steps_var = tk.IntVar(value=cfg0.rule_ab_max_steps)

        # 旧 A-B 距离阈值变量保留兼容，但单独测试不再使用距离停止。
        self.rule_ab_ab_close_threshold_var = tk.DoubleVar(value=cfg0.rule_ab_ab_close_threshold_px)
        self.rule_ab_ab_overlap_threshold_var = tk.DoubleVar(value=cfg0.rule_ab_ab_overlap_min_area_px)
        self.rule_ab_ac_target_clearance_var = tk.DoubleVar(value=cfg0.rule_ab_ac_target_clearance_px)
        self.rule_ab_ac_min_clearance_var = tk.DoubleVar(value=cfg0.rule_ab_ac_min_clearance_px)
        self.rule_ab_ac_max_clearance_var = tk.DoubleVar(value=cfg0.rule_ab_ac_max_clearance_px)
        # C 路线绕行方向：在 GUI 中明确选择“顺时针/逆时针”。
        # 内部仍写入 rule_ab_follow_c_direction：+1=屏幕视觉逆时针，-1=屏幕视觉顺时针。
        self.rule_ab_follow_c_direction_var = tk.StringVar(
            value=self._rule_ab_follow_direction_to_label(cfg0.rule_ab_follow_c_direction)
        )

        self.rule_ab_ch3_velocity_var = tk.IntVar(value=cfg0.rule_ab_stage_ch3_velocity)
        self.rule_ab_ch3_acceleration_var = tk.IntVar(value=cfg0.rule_ab_stage_ch3_acceleration)
        self.rule_ab_ch4_velocity_var = tk.IntVar(value=cfg0.rule_ab_stage_ch4_velocity)
        self.rule_ab_ch4_acceleration_var = tk.IntVar(value=cfg0.rule_ab_stage_ch4_acceleration)
        self.rule_ab_ch3_max_voltage_var = tk.IntVar(value=cfg0.rule_ab_stage_ch3_max_voltage)
        self.rule_ab_ch4_max_voltage_var = tk.IntVar(value=cfg0.rule_ab_stage_ch4_max_voltage)
        self.rule_ab_stage_step_x_var = tk.IntVar(value=cfg0.rule_ab_stage_step_x)
        self.rule_ab_stage_step_y_var = tk.IntVar(value=cfg0.rule_ab_stage_step_y)
        self.rule_ab_action_step_var = tk.IntVar(value=cfg0.rule_ab_action_step)
        self.rule_ab_ch3_pause_after_move_s_var = tk.DoubleVar(value=cfg0.rule_ab_stage_ch3_pause_after_move_s)
        self.rule_ab_ch4_pause_after_move_s_var = tk.DoubleVar(value=cfg0.rule_ab_stage_ch4_pause_after_move_s)

        self.rule_ab_static_c_map_name_var = tk.StringVar(value=cfg0.rule_ab_static_c_map_name)
        self.rule_ab_static_c_map_dir_var = tk.StringVar(value=cfg0.rule_ab_static_c_map_dir)
        self.rule_ab_load_static_c_map_var = tk.BooleanVar(value=cfg0.rule_ab_load_static_c_map_if_exists)
        self.rule_ab_force_reselect_c_var = tk.BooleanVar(value=cfg0.rule_ab_force_reselect_c_each_run)

        self.rule_ab_delta_min_var = tk.DoubleVar(value=cfg0.rule_ab_angle_delta_min_deg)
        self.rule_ab_delta_max_var = tk.DoubleVar(value=cfg0.rule_ab_angle_delta_max_deg)

        self.rule_ac_enable_stage_var = tk.BooleanVar(value=True)
        self.rule_ac_area_threshold_var = tk.DoubleVar(value=cfg0.rule_ac_area_threshold_px)  # 旧字段，保留兼容
        self.rule_ac_max_cycles_var = tk.IntVar(value=cfg0.rule_ac_max_cycles)
        self.rule_ac_target_x_var = tk.DoubleVar(value=cfg0.rule_ac_target_x_px)
        self.rule_ac_target_y_var = tk.DoubleVar(value=cfg0.rule_ac_target_y_px)
        self.rule_ac_center_tolerance_var = tk.DoubleVar(value=cfg0.rule_ac_center_tolerance_px)
        self.rule_ac_stage_step_size_var = tk.IntVar(value=cfg0.rule_ac_stage_step_size)
        self.rule_ac_stage12_velocity_var = tk.IntVar(value=cfg0.rule_ac_stage12_velocity)
        self.rule_ac_stage12_acceleration_var = tk.IntVar(value=cfg0.rule_ac_stage12_acceleration)
        self.rule_ac_stage12_max_voltage_var = tk.IntVar(value=cfg0.rule_ac_stage12_max_voltage)
        self.rule_ac_color_mode_var = tk.StringVar(value=cfg0.rule_ac_color_mode)
        self.rule_ac_color_h_var = tk.IntVar(value=cfg0.rule_ac_color_h)
        self.rule_ac_color_s_var = tk.IntVar(value=cfg0.rule_ac_color_s)
        self.rule_ac_color_v_var = tk.IntVar(value=cfg0.rule_ac_color_v)
        self.rule_ac_color_h_tol_var = tk.IntVar(value=cfg0.rule_ac_color_h_tol)
        self.rule_ac_color_s_tol_var = tk.IntVar(value=cfg0.rule_ac_color_s_tol)
        self.rule_ac_color_v_tol_var = tk.IntVar(value=cfg0.rule_ac_color_v_tol)
        self.rule_ac_color_min_area_var = tk.IntVar(value=cfg0.rule_ac_color_min_area_px)
        self.rule_ac_color_morph_kernel_var = tk.IntVar(value=cfg0.rule_ac_color_morph_kernel)
        self.rule_module_status_var = tk.StringVar(value="规则模块状态：未测试")

        for col in (1, 3):
            rule_frame.columnconfigure(col, weight=1)

        ttk.Checkbutton(rule_frame, text="RuleAB真动Stage34", variable=self.rule_ab_enable_stage_var).grid(
            row=0, column=0, columnspan=2, padx=4, pady=3, sticky="w"
        )

        ttk.Label(rule_frame, text="A-C距离 min/target/max px").grid(row=1, column=0, padx=4, pady=3, sticky="w")
        ttk.Entry(rule_frame, textvariable=self.rule_ab_ac_min_clearance_var, width=8).grid(row=1, column=1, padx=4, pady=3, sticky="ew")
        ttk.Entry(rule_frame, textvariable=self.rule_ab_ac_target_clearance_var, width=8).grid(row=1, column=2, padx=4, pady=3, sticky="ew")
        ttk.Entry(rule_frame, textvariable=self.rule_ab_ac_max_clearance_var, width=8).grid(row=1, column=3, padx=4, pady=3, sticky="ew")

        ttk.Label(rule_frame, text="AB覆盖阈值px² / C路线方向").grid(row=2, column=0, padx=4, pady=3, sticky="w")
        ttk.Entry(rule_frame, textvariable=self.rule_ab_ab_overlap_threshold_var, width=8).grid(row=2, column=1, padx=4, pady=3, sticky="ew")
        self.rule_ab_follow_c_direction_combo = ttk.Combobox(
            rule_frame,
            textvariable=self.rule_ab_follow_c_direction_var,
            values=("逆时针", "顺时针"),
            state="readonly",
            width=8,
        )
        self.rule_ab_follow_c_direction_combo.grid(row=2, column=2, padx=4, pady=3, sticky="ew")
        ttk.Label(rule_frame, text="A按所选方向沿C绕行").grid(row=2, column=3, padx=4, pady=3, sticky="w")

        ttk.Label(rule_frame, text="CH3 速度/加速度/停顿s").grid(row=3, column=0, padx=4, pady=3, sticky="w")
        ttk.Entry(rule_frame, textvariable=self.rule_ab_ch3_velocity_var, width=8).grid(row=3, column=1, padx=4, pady=3, sticky="ew")
        ttk.Entry(rule_frame, textvariable=self.rule_ab_ch3_acceleration_var, width=8).grid(row=3, column=2, padx=4, pady=3, sticky="ew")
        ttk.Entry(rule_frame, textvariable=self.rule_ab_ch3_pause_after_move_s_var, width=8).grid(row=3, column=3, padx=4, pady=3, sticky="ew")

        ttk.Label(rule_frame, text="CH4 速度/加速度/停顿s").grid(row=4, column=0, padx=4, pady=3, sticky="w")
        ttk.Entry(rule_frame, textvariable=self.rule_ab_ch4_velocity_var, width=8).grid(row=4, column=1, padx=4, pady=3, sticky="ew")
        ttk.Entry(rule_frame, textvariable=self.rule_ab_ch4_acceleration_var, width=8).grid(row=4, column=2, padx=4, pady=3, sticky="ew")
        ttk.Entry(rule_frame, textvariable=self.rule_ab_ch4_pause_after_move_s_var, width=8).grid(row=4, column=3, padx=4, pady=3, sticky="ew")

        ttk.Label(rule_frame, text="CH3/CH4最大电压").grid(row=5, column=0, padx=4, pady=3, sticky="w")
        ttk.Entry(rule_frame, textvariable=self.rule_ab_ch3_max_voltage_var, width=8).grid(row=5, column=1, padx=4, pady=3, sticky="ew")
        ttk.Entry(rule_frame, textvariable=self.rule_ab_ch4_max_voltage_var, width=8).grid(row=5, column=2, padx=4, pady=3, sticky="ew")

        ttk.Label(rule_frame, text="X/Y步数/action步数").grid(row=6, column=0, padx=4, pady=3, sticky="w")
        ttk.Entry(rule_frame, textvariable=self.rule_ab_stage_step_x_var, width=8).grid(row=6, column=1, padx=4, pady=3, sticky="ew")
        ttk.Entry(rule_frame, textvariable=self.rule_ab_stage_step_y_var, width=8).grid(row=6, column=2, padx=4, pady=3, sticky="ew")
        ttk.Entry(rule_frame, textvariable=self.rule_ab_action_step_var, width=8).grid(row=6, column=3, padx=4, pady=3, sticky="ew")

        ttk.Label(rule_frame, text="C地图名/加载/重选C").grid(row=7, column=0, padx=4, pady=3, sticky="w")
        ttk.Entry(rule_frame, textvariable=self.rule_ab_static_c_map_name_var, width=12).grid(row=7, column=1, padx=4, pady=3, sticky="ew")
        ttk.Checkbutton(rule_frame, text="加载C", variable=self.rule_ab_load_static_c_map_var).grid(row=7, column=2, padx=4, pady=3, sticky="w")
        ttk.Checkbutton(rule_frame, text="重选C", variable=self.rule_ab_force_reselect_c_var).grid(row=7, column=3, padx=4, pady=3, sticky="w")

        ttk.Label(rule_frame, text="C标定文件夹").grid(row=8, column=0, padx=4, pady=3, sticky="w")
        ttk.Entry(rule_frame, textvariable=self.rule_ab_static_c_map_dir_var, width=28).grid(row=8, column=1, columnspan=2, padx=4, pady=3, sticky="ew")
        ttk.Button(rule_frame, text="选择C文件夹", command=self.choose_static_c_map_dir).grid(row=8, column=3, padx=4, pady=3, sticky="ew")

        ttk.Label(rule_frame, text="完整测量角度目标min/max").grid(row=9, column=0, padx=4, pady=3, sticky="w")
        ttk.Entry(rule_frame, textvariable=self.rule_ab_delta_min_var, width=8).grid(row=9, column=1, padx=4, pady=3, sticky="ew")
        ttk.Entry(rule_frame, textvariable=self.rule_ab_delta_max_var, width=8).grid(row=9, column=2, padx=4, pady=3, sticky="ew")

        ttk.Checkbutton(rule_frame, text="Step9真动Stage12", variable=self.rule_ac_enable_stage_var).grid(
            row=10, column=0, columnspan=2, padx=4, pady=(8, 3), sticky="w"
        )
        ttk.Label(rule_frame, text="Step9目标x/y(px)").grid(row=11, column=0, padx=4, pady=3, sticky="w")
        ttk.Entry(rule_frame, textvariable=self.rule_ac_target_x_var, width=10).grid(row=11, column=1, padx=4, pady=3, sticky="ew")
        ttk.Entry(rule_frame, textvariable=self.rule_ac_target_y_var, width=10).grid(row=11, column=2, padx=4, pady=3, sticky="ew")
        ttk.Button(rule_frame, text="选定Step9目标", command=self.select_step9_target_thread).grid(row=11, column=3, padx=4, pady=3, sticky="ew")

        ttk.Label(rule_frame, text="Step9容差px/步数/最大循环").grid(row=12, column=0, padx=4, pady=3, sticky="w")
        ttk.Entry(rule_frame, textvariable=self.rule_ac_center_tolerance_var, width=10).grid(row=12, column=1, padx=4, pady=3, sticky="ew")
        ttk.Entry(rule_frame, textvariable=self.rule_ac_stage_step_size_var, width=10).grid(row=12, column=2, padx=4, pady=3, sticky="ew")
        ttk.Entry(rule_frame, textvariable=self.rule_ac_max_cycles_var, width=10).grid(row=12, column=3, padx=4, pady=3, sticky="ew")

        ttk.Label(rule_frame, text="Stage12速度/加速度/电压").grid(row=13, column=0, padx=4, pady=3, sticky="w")
        ttk.Entry(rule_frame, textvariable=self.rule_ac_stage12_velocity_var, width=10).grid(row=13, column=1, padx=4, pady=3, sticky="ew")
        ttk.Entry(rule_frame, textvariable=self.rule_ac_stage12_acceleration_var, width=10).grid(row=13, column=2, padx=4, pady=3, sticky="ew")
        ttk.Entry(rule_frame, textvariable=self.rule_ac_stage12_max_voltage_var, width=10).grid(row=13, column=3, padx=4, pady=3, sticky="ew")

        ttk.Label(rule_frame, text="Step9颜色模式/HSV").grid(row=14, column=0, padx=4, pady=3, sticky="w")
        ttk.Combobox(rule_frame, textvariable=self.rule_ac_color_mode_var, values=("include", "exclude"), width=9, state="readonly").grid(row=14, column=1, padx=4, pady=3, sticky="ew")
        ttk.Entry(rule_frame, textvariable=self.rule_ac_color_h_var, width=6).grid(row=14, column=2, padx=4, pady=3, sticky="ew")
        ttk.Entry(rule_frame, textvariable=self.rule_ac_color_s_var, width=6).grid(row=14, column=3, padx=4, pady=3, sticky="ew")
        ttk.Label(rule_frame, text="HSV容差H/S/V/最小面积").grid(row=15, column=0, padx=4, pady=3, sticky="w")
        ttk.Entry(rule_frame, textvariable=self.rule_ac_color_h_tol_var, width=6).grid(row=15, column=1, padx=4, pady=3, sticky="ew")
        ttk.Entry(rule_frame, textvariable=self.rule_ac_color_s_tol_var, width=6).grid(row=15, column=2, padx=4, pady=3, sticky="ew")
        color_extra = ttk.Frame(rule_frame)
        color_extra.grid(row=15, column=3, padx=4, pady=3, sticky="ew")
        color_extra.columnconfigure(0, weight=1)
        color_extra.columnconfigure(1, weight=1)
        ttk.Entry(color_extra, textvariable=self.rule_ac_color_v_tol_var, width=5).grid(row=0, column=0, padx=(0, 2), sticky="ew")
        ttk.Entry(color_extra, textvariable=self.rule_ac_color_min_area_var, width=5).grid(row=0, column=1, padx=(2, 0), sticky="ew")

        rule_buttons = ttk.Frame(rule_frame)
        rule_buttons.grid(row=16, column=0, columnspan=4, padx=2, pady=6, sticky="ew")
        for i in range(4):
            rule_buttons.columnconfigure(i, weight=1)
        ttk.Button(rule_buttons, text="单独分割C", command=self.segment_c_only_thread).grid(row=0, column=0, padx=4, pady=3, sticky="ew")
        ttk.Button(rule_buttons, text="单独A沿C绕行", command=self.test_rule_ab_thread).grid(row=0, column=1, padx=4, pady=3, sticky="ew")
        ttk.Button(rule_buttons, text="停止A推B控制器", command=self.stop_rule_ab_controller_thread).grid(row=0, column=2, padx=4, pady=3, sticky="ew")
        ttk.Button(rule_buttons, text="单独测试Step9对齐", command=self.test_rule_ac_thread).grid(row=0, column=3, padx=4, pady=3, sticky="ew")
        ttk.Button(rule_buttons, text="停止1/2轴运动", command=self.stop_step9_stage12_thread).grid(row=1, column=0, columnspan=4, padx=4, pady=3, sticky="ew")
        ttk.Label(rule_frame, textvariable=self.rule_module_status_var, wraplength=450).grid(
            row=17, column=0, columnspan=4, padx=4, pady=(4, 0), sticky="w"
        )

        # =====================================================
        # 左侧 9：光谱补焦循环
        # =====================================================
        saf_frame = ttk.LabelFrame(
            left_inner,
            text="9. 光谱补焦循环",
            padding=10,
            style="Panel.TLabelframe",
        )
        saf_frame.pack(fill=tk.X, padx=4, pady=(0, 8))

        self.saf_roi_var = tk.StringVar(value="0,0,300,300")
        self.saf_capture_area_var = tk.StringVar(value=",".join(str(v) for v in cfg0.capture_area))
        self.saf_output_dir_var = tk.StringVar(value="focus_output")
        self.saf_status_var = tk.StringVar(value="光谱补焦循环：未启动")
        self.saf_max_cycles_var = tk.IntVar(value=0)

        # 补焦参数（与 autofocus_qt_ui 对齐）
        self.saf_trigger_ratio_var = tk.DoubleVar(value=0.95)
        self.saf_stop_ratio_var = tk.DoubleVar(value=0.95)
        self.saf_trigger_count_var = tk.IntVar(value=3)
        self.saf_trigger_absolute_var = tk.BooleanVar(value=True)
        self.saf_detection_only_var = tk.BooleanVar(value=False)
        self.saf_passive_mode_var = tk.BooleanVar(value=True)
        self.saf_passive_attempts_var = tk.IntVar(value=10)
        self.saf_passive_good_var = tk.IntVar(value=5)
        self.saf_disable_auto_stop_var = tk.BooleanVar(value=False)
        self.saf_z_enabled_var = tk.BooleanVar(value=True)
        self.saf_z_axis_var = tk.IntVar(value=1)
        self.saf_z_speed_var = tk.IntVar(value=100)
        self.saf_z_accel_var = tk.IntVar(value=100)
        self.saf_search_strategy_var = tk.StringVar(value="hill_climb")
        self.saf_interval_var = tk.DoubleVar(value=1.0)
        self.saf_wait_between_spectrum_var = tk.DoubleVar(value=120.0)
        self.saf_z_search_steps_var = tk.IntVar(value=10)
        self.saf_z_patience_var = tk.IntVar(value=3)
        self.saf_z_direction_probe_stage_count_var = tk.IntVar(value=3)
        self.saf_z_direction_probe_step_interval_var = tk.IntVar(value=10)
        self.saf_z_direction_probe_samples_var = tk.IntVar(value=3)
        self.saf_z_direction_probe_points_var = tk.IntVar(value=5)
        self.saf_z_local_refine_enabled_var = tk.BooleanVar(value=True)
        self.saf_z_local_refine_decay_var = tk.DoubleVar(value=0.5)
        self.saf_z_local_refine_min_step_var = tk.IntVar(value=1)
        self.saf_z_local_refine_max_rounds_var = tk.IntVar(value=4)

        for col_idx in range(4):
            saf_frame.columnconfigure(col_idx, weight=1)

        # Row 0: ROI
        ttk.Label(saf_frame, text="Focus ROI(x,y,w,h)").grid(
            row=0, column=0, padx=4, pady=4, sticky="w"
        )
        ttk.Entry(saf_frame, textvariable=self.saf_roi_var).grid(
            row=0, column=1, columnspan=2, padx=4, pady=4, sticky="ew"
        )
        ttk.Button(saf_frame, text="选择ROI", command=self.select_saf_roi_thread).grid(
            row=0, column=3, padx=4, pady=4, sticky="ew"
        )

        # Row 1: 补焦专用截图区域（独立于标定/角度检测的 capture_area）
        ttk.Label(saf_frame, text="补焦截图区域").grid(
            row=1, column=0, padx=4, pady=4, sticky="w"
        )
        ttk.Entry(saf_frame, textvariable=self.saf_capture_area_var).grid(
            row=1, column=1, columnspan=3, padx=4, pady=4, sticky="ew"
        )

        # Row 2: 输出目录 + 最大轮数
        ttk.Label(saf_frame, text="输出目录").grid(
            row=2, column=0, padx=4, pady=4, sticky="w"
        )
        ttk.Entry(saf_frame, textvariable=self.saf_output_dir_var).grid(
            row=2, column=1, padx=4, pady=4, sticky="ew"
        )
        ttk.Label(saf_frame, text="最大轮数(0=无限)").grid(
            row=2, column=2, padx=4, pady=4, sticky="w"
        )
        ttk.Entry(saf_frame, textvariable=self.saf_max_cycles_var, width=10).grid(
            row=2, column=3, padx=4, pady=4, sticky="w"
        )

        # Row 3: 触发阈值 / 目标阈值
        ttk.Label(saf_frame, text="触发阈值").grid(
            row=3, column=0, padx=4, pady=4, sticky="w"
        )
        ttk.Entry(saf_frame, textvariable=self.saf_trigger_ratio_var, width=10).grid(
            row=3, column=1, padx=4, pady=4, sticky="w"
        )
        ttk.Label(saf_frame, text="目标阈值").grid(
            row=3, column=2, padx=4, pady=4, sticky="w"
        )
        ttk.Entry(saf_frame, textvariable=self.saf_stop_ratio_var, width=10).grid(
            row=3, column=3, padx=4, pady=4, sticky="w"
        )

        # Row 4: 连续触发次数 / 搜索策略
        ttk.Label(saf_frame, text="连续触发次数").grid(
            row=4, column=0, padx=4, pady=4, sticky="w"
        )
        ttk.Entry(saf_frame, textvariable=self.saf_trigger_count_var, width=10).grid(
            row=4, column=1, padx=4, pady=4, sticky="w"
        )
        ttk.Label(saf_frame, text="搜索策略").grid(
            row=4, column=2, padx=4, pady=4, sticky="w"
        )
        ttk.Combobox(
            saf_frame,
            textvariable=self.saf_search_strategy_var,
            values=["hill_climb", "full_sweep", "curve_fit", "golden_section"],
            state="readonly",
            width=12,
        ).grid(row=4, column=3, padx=4, pady=4, sticky="w")

        # Row 5: 模式开关
        ttk.Checkbutton(
            saf_frame, text="绝对值区间", variable=self.saf_trigger_absolute_var
        ).grid(row=5, column=0, padx=4, pady=4, sticky="w")
        ttk.Checkbutton(
            saf_frame, text="FocusScore检测", variable=self.saf_detection_only_var
        ).grid(row=5, column=1, padx=4, pady=4, sticky="w")
        ttk.Checkbutton(
            saf_frame, text="被动补焦", variable=self.saf_passive_mode_var
        ).grid(row=5, column=2, padx=4, pady=4, sticky="w")
        ttk.Checkbutton(
            saf_frame, text="关闭达标阈值", variable=self.saf_disable_auto_stop_var
        ).grid(row=5, column=3, padx=4, pady=4, sticky="w")

        # Row 6: 被动补焦参数
        ttk.Label(saf_frame, text="最大连续补焦").grid(
            row=6, column=0, padx=4, pady=4, sticky="w"
        )
        ttk.Entry(saf_frame, textvariable=self.saf_passive_attempts_var, width=10).grid(
            row=6, column=1, padx=4, pady=4, sticky="w"
        )
        ttk.Label(saf_frame, text="连续达标次数").grid(
            row=6, column=2, padx=4, pady=4, sticky="w"
        )
        ttk.Entry(saf_frame, textvariable=self.saf_passive_good_var, width=10).grid(
            row=6, column=3, padx=4, pady=4, sticky="w"
        )

        # Row 7: Z 轴参数
        ttk.Checkbutton(
            saf_frame, text="启用Z轴", variable=self.saf_z_enabled_var
        ).grid(row=7, column=0, padx=4, pady=4, sticky="w")
        ttk.Label(saf_frame, text="Z轴").grid(
            row=7, column=1, padx=4, pady=4, sticky="w"
        )
        ttk.Entry(saf_frame, textvariable=self.saf_z_axis_var, width=6).grid(
            row=7, column=1, padx=(30, 4), pady=4, sticky="w"
        )
        ttk.Label(saf_frame, text="速度").grid(
            row=7, column=2, padx=4, pady=4, sticky="w"
        )
        ttk.Entry(saf_frame, textvariable=self.saf_z_speed_var, width=8).grid(
            row=7, column=2, padx=(30, 4), pady=4, sticky="w"
        )
        ttk.Label(saf_frame, text="加速度").grid(
            row=7, column=3, padx=4, pady=4, sticky="w"
        )
        ttk.Entry(saf_frame, textvariable=self.saf_z_accel_var, width=8).grid(
            row=7, column=3, padx=(40, 4), pady=4, sticky="w"
        )

        # Row 8: 时间参数
        ttk.Label(saf_frame, text="补焦间隔(s)").grid(
            row=8, column=0, padx=4, pady=4, sticky="w"
        )
        ttk.Entry(saf_frame, textvariable=self.saf_interval_var, width=10).grid(
            row=8, column=1, padx=4, pady=4, sticky="w"
        )
        ttk.Label(saf_frame, text="光谱间等待(s)").grid(
            row=8, column=2, padx=4, pady=4, sticky="w"
        )
        ttk.Entry(saf_frame, textvariable=self.saf_wait_between_spectrum_var, width=10).grid(
            row=8, column=3, padx=4, pady=4, sticky="w"
        )

        # Row 9: Z轴移动步数 / Z轴方向判断连续次数
        ttk.Label(saf_frame, text="Z轴移动步数").grid(
            row=9, column=0, padx=4, pady=4, sticky="w"
        )
        ttk.Entry(saf_frame, textvariable=self.saf_z_search_steps_var, width=10).grid(
            row=9, column=1, padx=4, pady=4, sticky="w"
        )
        ttk.Label(saf_frame, text="方向判断连续次数").grid(
            row=9, column=2, padx=4, pady=4, sticky="w"
        )
        ttk.Entry(saf_frame, textvariable=self.saf_z_patience_var, width=10).grid(
            row=9, column=3, padx=4, pady=4, sticky="w"
        )

        # Row 10: 动态双向采样参数
        ttk.Label(saf_frame, text="阶段数/步数间隔").grid(
            row=10, column=0, padx=4, pady=4, sticky="w"
        )
        probe_stage_frame = ttk.Frame(saf_frame)
        probe_stage_frame.grid(row=10, column=1, padx=4, pady=4, sticky="w")
        ttk.Entry(probe_stage_frame, textvariable=self.saf_z_direction_probe_stage_count_var, width=5).grid(
            row=0, column=0, padx=(0, 4), pady=0, sticky="w"
        )
        ttk.Entry(probe_stage_frame, textvariable=self.saf_z_direction_probe_step_interval_var, width=5).grid(
            row=0, column=1, padx=(4, 0), pady=0, sticky="w"
        )
        ttk.Label(saf_frame, text="重复采样/每档点数").grid(
            row=10, column=2, padx=4, pady=4, sticky="w"
        )
        probe_sample_frame = ttk.Frame(saf_frame)
        probe_sample_frame.grid(row=10, column=3, padx=4, pady=4, sticky="w")
        ttk.Entry(probe_sample_frame, textvariable=self.saf_z_direction_probe_samples_var, width=5).grid(
            row=0, column=0, padx=(0, 4), pady=0, sticky="w"
        )
        ttk.Entry(probe_sample_frame, textvariable=self.saf_z_direction_probe_points_var, width=5).grid(
            row=0, column=1, padx=(4, 0), pady=0, sticky="w"
        )

        # Row 11: 局部细搜参数
        ttk.Checkbutton(
            saf_frame, text="启用局部细搜", variable=self.saf_z_local_refine_enabled_var
        ).grid(row=11, column=0, padx=4, pady=4, sticky="w")
        ttk.Label(saf_frame, text="细搜衰减").grid(
            row=11, column=1, padx=4, pady=4, sticky="w"
        )
        ttk.Entry(saf_frame, textvariable=self.saf_z_local_refine_decay_var, width=8).grid(
            row=11, column=1, padx=(70, 4), pady=4, sticky="w"
        )
        ttk.Label(saf_frame, text="最小步数").grid(
            row=11, column=2, padx=4, pady=4, sticky="w"
        )
        ttk.Entry(saf_frame, textvariable=self.saf_z_local_refine_min_step_var, width=8).grid(
            row=11, column=2, padx=(70, 4), pady=4, sticky="w"
        )
        ttk.Label(saf_frame, text="最大轮数").grid(
            row=11, column=3, padx=4, pady=4, sticky="w"
        )
        ttk.Entry(saf_frame, textvariable=self.saf_z_local_refine_max_rounds_var, width=8).grid(
            row=11, column=3, padx=(70, 4), pady=4, sticky="w"
        )

        # Row 12: 按钮
        saf_buttons = ttk.Frame(saf_frame)
        saf_buttons.grid(row=12, column=0, columnspan=4, padx=2, pady=4, sticky="ew")
        saf_buttons.columnconfigure(0, weight=1)
        saf_buttons.columnconfigure(1, weight=1)
        saf_buttons.columnconfigure(2, weight=1)
        ttk.Button(
            saf_buttons,
            text="开始光谱补焦循环",
            command=self.start_spectrum_autofocus_loop_thread,
            style="Primary.TButton",
        ).grid(row=0, column=0, padx=4, pady=3, sticky="ew")
        ttk.Button(
            saf_buttons,
            text="停止光谱补焦循环",
            command=self.stop_spectrum_autofocus_loop,
            style="Danger.TButton",
        ).grid(row=0, column=1, padx=4, pady=3, sticky="ew")
        ttk.Button(
            saf_buttons,
            text="预览FocusScore曲线",
            command=self.open_focus_score_preview,
        ).grid(row=0, column=2, padx=4, pady=3, sticky="ew")

        # Row 13: 状态
        ttk.Label(saf_frame, textvariable=self.saf_status_var, wraplength=450).grid(
            row=13, column=0, columnspan=4, padx=4, pady=(4, 0), sticky="w"
        )

        # =====================================================
        # 中间：图像显示 + 运行日志
        # =====================================================
        center_panel.columnconfigure(0, weight=1)
        center_panel.rowconfigure(0, weight=3)
        center_panel.rowconfigure(1, weight=2)

        # 中间上部：图像/曲线显示。
        # 现在固定显示三张图：
        #   1. 序号 - 拟合峰值；
        #   2. xlsx横坐标 - 原始数据；
        #   3. xlsx横坐标 - 中值滤波结果。
        # 右侧仍然保留“角度-拟合峰值列表”。
        plot_frame = ttk.LabelFrame(
            center_panel,
            text="图像显示：序号-拟合峰值 / xlsx横坐标-原始数据 / xlsx横坐标-中值滤波结果",
            padding=10,
            style="Panel.TLabelframe",
        )
        plot_frame.grid(row=0, column=0, sticky="nsew", padx=(10, 8), pady=(0, 8))
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)

        self.fig = Figure(figsize=(8.2, 7.2), dpi=100)
        self.ax_fit_peak = self.fig.add_subplot(311)
        self.ax_raw = self.fig.add_subplot(312)
        self.ax_median = self.fig.add_subplot(313)
        self.fig.tight_layout()

        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.plot_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        ttk.Label(plot_frame, textvariable=self.plot_status_var).grid(row=1, column=0, sticky="w", pady=(6, 0))

        # 中间下部：运行日志
        log_frame = ttk.LabelFrame(center_panel, text="运行日志", padding=10, style="Panel.TLabelframe")
        log_frame.grid(row=1, column=0, sticky="nsew", padx=(10, 8), pady=(8, 0))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, height=12, wrap=tk.WORD)
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scrollbar.grid(row=0, column=1, sticky="ns")

        # =====================================================
        # 右侧：角度-拟合峰值列表
        # =====================================================
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(0, weight=1)

        angle_fit_list_frame = ttk.LabelFrame(
            right_panel,
            text="角度-拟合峰值列表",
            padding=10,
            style="Panel.TLabelframe",
        )
        angle_fit_list_frame.grid(row=0, column=0, sticky="nsew", padx=(8, 0), pady=(0, 0))
        angle_fit_list_frame.rowconfigure(0, weight=1)
        angle_fit_list_frame.columnconfigure(0, weight=1)

        columns = ("cycle", "angle", "fit_peak")
        self.angle_fit_tree = ttk.Treeview(
            angle_fit_list_frame,
            columns=columns,
            show="headings",
            height=24,
        )
        self.angle_fit_tree.heading("cycle", text="序号")
        self.angle_fit_tree.heading("angle", text="横坐标：角度/deg")
        self.angle_fit_tree.heading("fit_peak", text="纵坐标：拟合峰值")
        self.angle_fit_tree.column("cycle", width=60, anchor="center", stretch=False)
        self.angle_fit_tree.column("angle", width=130, anchor="center", stretch=True)
        self.angle_fit_tree.column("fit_peak", width=130, anchor="center", stretch=True)

        tree_y_scroll = ttk.Scrollbar(angle_fit_list_frame, orient=tk.VERTICAL, command=self.angle_fit_tree.yview)
        tree_x_scroll = ttk.Scrollbar(angle_fit_list_frame, orient=tk.HORIZONTAL, command=self.angle_fit_tree.xview)
        self.angle_fit_tree.configure(yscrollcommand=tree_y_scroll.set, xscrollcommand=tree_x_scroll.set)

        def _angle_fit_mousewheel(event):
            # Windows: event.delta 每 120 代表一格；这里一次滚动 5 行。
            try:
                units = -5 * int(event.delta / 120)
                if units == 0:
                    units = -5 if event.delta > 0 else 5
                self.angle_fit_tree.yview_scroll(units, "units")
                return "break"
            except Exception:
                return None

        def _angle_fit_scroll_up(event):
            try:
                self.angle_fit_tree.yview_scroll(-5, "units")
                return "break"
            except Exception:
                return None

        def _angle_fit_scroll_down(event):
            try:
                self.angle_fit_tree.yview_scroll(5, "units")
                return "break"
            except Exception:
                return None

        self.angle_fit_tree.bind("<MouseWheel>", _angle_fit_mousewheel)
        self.angle_fit_tree.bind("<Button-4>", _angle_fit_scroll_up)
        self.angle_fit_tree.bind("<Button-5>", _angle_fit_scroll_down)

        self.angle_fit_tree.grid(row=0, column=0, sticky="nsew")
        tree_y_scroll.grid(row=0, column=1, sticky="ns")
        tree_x_scroll.grid(row=1, column=0, sticky="ew")

        ttk.Label(angle_fit_list_frame, textvariable=self.angle_fit_list_status_var, wraplength=320).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )

        # 为所有 UI 参数注册实时同步 trace
        self._bind_ui_parameter_traces()

    # --------------------------------------------------------
    # GUI 工具
    # --------------------------------------------------------

    def log(self, msg: str):
        def _append():
            self.log_text.insert(tk.END, msg + "\n")
            self.log_text.see(tk.END)
            print(msg)
        self.root.after(0, _append)

    def _on_signal_on_time_var_changed(self, *args):
        """
        记录用户是否在测量过程中手动修改了“信号ON/ms”。

        程序自动把计算出的下一轮信号时间写回输入框时，
        _programmatic_signal_on_time_update=True，因此不会触发手动覆盖标记。
        """
        if self._programmatic_signal_on_time_update:
            return

        try:
            if self.workflow is not None and self.workflow.is_measuring:
                self.signal_on_time_user_modified = True
                # 这里只做轻量标记，不在 trace 回调里频繁写日志，避免输入时刷屏。
        except Exception:
            pass

    def _on_ui_parameter_change(self, *args):
        """
        任意 UI 参数变化时触发。

        非运行状态下立即把 GUI 参数同步到 workflow 配置；
        运行状态下本轮不重复同步，由下一轮开始前的 sync_config_from_ui_to_workflow()
        统一读取当前 GUI 值，避免输入过程中频繁重建对象。
        """
        try:
            if self.workflow is not None and not self.workflow.is_measuring:
                self.sync_config_from_ui_to_workflow()
                self.log("[实时更新] UI 参数已同步到 workflow 配置")
        except Exception:
            # 用户输入过程中可能出现临时非法值，不阻断交互
            pass

    def _bind_ui_parameter_traces(self):
        """为所有 UI 参数变量注册 trace，实现非运行状态下的实时同步。"""
        param_vars = [
            self.hardware_mode_var,
            self.virtual_spectrum_mode_var,
            self.virtual_spectrum_replay_csv_var,
            self.max_cycles_var,
            self.sub_loop_iterations_per_cycle_var,
            self.signal_on_time_ms_var,
            self.stable_wait_ms_var,
            self.signal_time_factor_var,
            self.angle_delta_min_var,
            self.angle_delta_max_var,
            self.num_var,
            self.cw_var,
            self.save_root_var,
            self.raw_remove_above_var,
            self.median_filter_window_var,
            self.x_axis_xlsx_path_var,
            self.light_port_var,
            self.rigol_visa_var,
            self.angle_model_path_var,
            self.capture_area_var,
            self.tcp_host_var,
            self.tcp_port_var,
            self.tcp_output_dir_var,
            self.tcp_command_var,
            self.save_log_to_file_var,
            self.log_dir_var,
            self.spectrometer_backend_var,
            self.picam_exposure_var,
            self.picam_temperature_var,
            self.picam_roi_width_var,
            self.picam_roi_height_var,
            self.enable_delta_w_judge_var,
            self.delta_w_threshold_var,
            self.stop_when_delta_w_not_enough_var,
            self.rule_ab_enable_stage_var,
            self.rule_ab_max_steps_var,
            self.rule_ab_ab_close_threshold_var,
            self.rule_ab_ab_overlap_threshold_var,
            self.rule_ab_ac_target_clearance_var,
            self.rule_ab_ac_min_clearance_var,
            self.rule_ab_ac_max_clearance_var,
            self.rule_ab_stage_step_x_var,
            self.rule_ab_stage_step_y_var,
            self.rule_ab_action_step_var,
            self.rule_ab_static_c_map_name_var,
            self.rule_ab_static_c_map_dir_var,
            self.rule_ab_load_static_c_map_var,
            self.rule_ab_force_reselect_c_var,
            self.rule_ab_delta_min_var,
            self.rule_ab_delta_max_var,
            self.rule_ab_follow_c_direction_var,
            self.rule_ab_ch3_velocity_var,
            self.rule_ab_ch3_acceleration_var,
            self.rule_ab_ch4_velocity_var,
            self.rule_ab_ch4_acceleration_var,
            self.rule_ab_ch3_max_voltage_var,
            self.rule_ab_ch4_max_voltage_var,
            self.rule_ab_ch3_pause_after_move_s_var,
            self.rule_ab_ch4_pause_after_move_s_var,
            self.rule_ac_enable_stage_var,
            self.rule_ac_area_threshold_var,
            self.rule_ac_max_cycles_var,
            self.rule_ac_target_x_var,
            self.rule_ac_target_y_var,
            self.rule_ac_center_tolerance_var,
            self.rule_ac_stage_step_size_var,
            self.rule_ac_color_mode_var,
            self.rule_ac_color_h_var,
            self.rule_ac_color_s_var,
            self.rule_ac_color_v_var,
            self.rule_ac_color_h_tol_var,
            self.rule_ac_color_s_tol_var,
            self.rule_ac_color_v_tol_var,
            self.rule_ac_color_min_area_var,
            self.rule_ac_color_morph_kernel_var,
            self.saf_roi_var,
            self.saf_capture_area_var,
            self.saf_output_dir_var,
            self.saf_max_cycles_var,
            self.saf_trigger_ratio_var,
            self.saf_stop_ratio_var,
            self.saf_trigger_count_var,
            self.saf_trigger_absolute_var,
            self.saf_detection_only_var,
            self.saf_passive_mode_var,
            self.saf_passive_attempts_var,
            self.saf_passive_good_var,
            self.saf_disable_auto_stop_var,
            self.saf_z_enabled_var,
            self.saf_z_axis_var,
            self.saf_z_speed_var,
            self.saf_z_accel_var,
            self.saf_search_strategy_var,
            self.saf_z_search_steps_var,
            self.saf_z_patience_var,
            self.saf_z_direction_probe_stage_count_var,
            self.saf_z_direction_probe_step_interval_var,
            self.saf_z_direction_probe_samples_var,
            self.saf_z_direction_probe_points_var,
            self.saf_z_local_refine_enabled_var,
            self.saf_z_local_refine_decay_var,
            self.saf_z_local_refine_min_step_var,
            self.saf_z_local_refine_max_rounds_var,
        ]
        for var in param_vars:
            try:
                var.trace_add("write", self._on_ui_parameter_change)
            except Exception:
                pass

    # 新增：光谱仪设备选择变化处理
    def _on_spectrometer_backend_changed(self, event=None):
        """当用户切换光谱仪后端时，更新 PI 参数输入框的启用状态。"""
        self._update_pi_controls_state()
        backend = str(self.spectrometer_backend_var.get()).strip().lower()
        if backend in ("picam", "picam_demo"):
            self.log(f"[光谱仪] 已切换为 PI 直接控制后端: {backend}")
        else:
            self.log(f"[光谱仪] 已切换为 LabVIEW TCP 后端")

    def _update_pi_controls_state(self):
        """根据当前光谱仪后端启用/禁用 PI 参数输入框。"""
        backend = str(self.spectrometer_backend_var.get()).strip().lower()
        is_pi = backend in ("picam", "picam_demo")
        state = "normal" if is_pi else "disabled"
        try:
            self.picam_exposure_entry.configure(state=state)
            self.picam_temperature_entry.configure(state=state)
            self.picam_roi_width_entry.configure(state=state)
            self.picam_roi_height_entry.configure(state=state)
        except Exception:
            pass

    def set_signal_on_time_var_programmatically(self, value: Any):
        """
        程序内部更新“信号ON/ms”输入框。
        这类更新表示自动计算结果，不等同于用户手动修改。
        """
        def _set():
            self._programmatic_signal_on_time_update = True
            try:
                self.signal_on_time_ms_var.set(value)
            finally:
                self._programmatic_signal_on_time_update = False

        self.root.after(0, _set)

    def set_var(self, var: tk.Variable, value: Any):
        self.root.after(0, lambda: var.set(value))

    def set_var_blocking(self, var: tk.Variable, value: Any, timeout_s: float = 2.0):
        """
        从工作线程同步更新 Tk 变量。

        普通 set_var() 使用 root.after 异步提交，调用后马上读取 GUI 变量时可能还没生效。
        Step9 目标/颜色标定需要“刚选的新目标”立刻进入 cfg 和标定 JSON，
        因此这里提供一个带等待的同步写入方法。
        """
        done = threading.Event()

        def _set():
            try:
                var.set(value)
            finally:
                done.set()

        try:
            self.root.after(0, _set)
            done.wait(max(0.05, float(timeout_s)))
        except Exception:
            try:
                var.set(value)
            except Exception:
                pass

    def run_in_thread(self, target):
        t = threading.Thread(target=target, daemon=True)
        t.start()
        return t

    def select_save_root(self):
        path = filedialog.askdirectory(title="选择保存目录")
        if path:
            self.save_root_var.set(path)

    def select_x_axis_xlsx(self):
        path = filedialog.askopenfilename(
            title="选择图2/图3横坐标 .xlsx 文件",
            filetypes=[("Excel workbook", "*.xlsx"), ("All files", "*.*")],
            initialdir=str(PROJECT_ROOT),
        )
        if path:
            self.x_axis_xlsx_path_var.set(path)

    def select_angle_model(self):
        path = filedialog.askopenfilename(
            title="选择 YOLO .pt 模型",
            filetypes=[("YOLO/PyTorch model", "*.pt *.pth"), ("All files", "*.*")],
            initialdir=str(PROJECT_ROOT),
        )
        if path:
            self.angle_model_path_var.set(path)

    def parse_capture_area(self) -> Tuple[int, int, int, int]:
        try:
            vals = [int(x.strip()) for x in self.capture_area_var.get().split(",")]
            if len(vals) != 4:
                raise ValueError
            return tuple(vals)  # type: ignore
        except Exception:
            raise ValueError("截图区域格式错误，应为 left,top,width,height，例如 116,98,1112,886")

    def parse_int_var(self, var: tk.Variable, name: str) -> int:
        try:
            return int(var.get())
        except Exception:
            raise ValueError(f"{name} 必须是整数，例如 0、1、2")

    def _resolve_static_c_reference_dir(self, folder: str) -> Optional[Path]:
        """
        解析 C 标定文件夹。

        支持两种选择方式：
            1. 直接选择包含 static_c_mask.png 的文件夹；
            2. 选择 run_xxx 文件夹，程序自动使用其 static_c_reference 子文件夹。
        """
        if not folder:
            return None

        p = Path(str(folder).strip().strip('"').strip("'"))
        candidates = [p]
        candidates.append(p / "static_c_reference")

        for c in candidates:
            try:
                if c.exists() and c.is_dir() and (c / "static_c_mask.png").exists():
                    return c.resolve()
            except Exception:
                continue
        return None

    def choose_static_c_map_dir(self):
        """选择已经保存好的 C 标定文件夹，并让后续“单独A沿C绕行”加载该 C 分割结果。"""
        folder = filedialog.askdirectory(title="选择包含 static_c_mask.png 的 C 标定文件夹，或选择 run_xxx 文件夹；新版 static_c_mask.png 为四边形 C mask")
        if not folder:
            return

        resolved = self._resolve_static_c_reference_dir(folder)
        if resolved is None:
            self.rule_ab_static_c_map_dir_var.set(folder)
            self.rule_ab_load_static_c_map_var.set(True)
            self.rule_ab_force_reselect_c_var.set(False)
            msg = (
                "所选文件夹中未找到 static_c_mask.png。\n\n"
                "可以先点击“单独分割C”生成 C 标定文件夹；\n"
                "或者选择包含 static_c_mask.png 的 static_c_reference 文件夹。"
            )
            self.log(f"[RuleAB-C] 已选择 C 文件夹但未找到 static_c_mask.png：{folder}")
            self.root.after(0, lambda: messagebox.showwarning("C标定文件夹无效", msg))
            return

        self.rule_ab_static_c_map_dir_var.set(str(resolved))
        self.rule_ab_load_static_c_map_var.set(True)
        self.rule_ab_force_reselect_c_var.set(False)
        self.log(f"[RuleAB-C] 已选择并加载 C 标定文件夹：{resolved}")

    def segment_c_only_thread(self):
        self.run_in_thread(self.segment_c_only)

    def _capture_current_rule_ab_frame(self, output_dir: Path) -> np.ndarray:
        """截取当前固定屏幕区域，返回 RGB 图像。"""
        cfg_gui = self.build_config_from_ui()
        capture_area = tuple(int(v) for v in cfg_gui.capture_area)

        if FixedRegionScreenCapture is not None:
            capturer = FixedRegionScreenCapture(
                capture_area=capture_area,
                output_dir=output_dir / "c_only_capture_tmp",
                save_image=False,
            )
            return capturer.capture()

        # 兜底：不用 vision.screen_capture 时，使用 PIL.ImageGrab。
        try:
            from PIL import ImageGrab
        except Exception as e:
            raise RuntimeError("无法导入 FixedRegionScreenCapture，也无法导入 PIL.ImageGrab，不能截图。") from e

        left, top, width, height = capture_area
        img = ImageGrab.grab(bbox=(left, top, left + width, top + height)).convert("RGB")
        return np.array(img)

    def _select_c_points_interactively(
        self,
        image_rgb: np.ndarray,
        window_name: str = "SAM2 C only: left=positive, right=negative",
        scale: float = 0.85,
        existing_roi_polygons: Any = None,
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], List[List[List[float]]]]:
        """
        只为 C 进行交互式点提示：
            左键：C 正点，可以多个；
            右键：C 负点，可以多个；
            R：清空；
            Enter/N：完成点选；
            点选完成后按 E 进入 ROI 模式，绘制需要排除的干扰区域；
            ESC：取消。
        返回 C 正负点和 ROI 多边形。
        """
        from logic.roi_exclusion import select_roi_polygons_interactively

        if scale <= 0:
            scale = 1.0

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        h, w = image_bgr.shape[:2]
        show_w = max(1, int(w * scale))
        show_h = max(1, int(h * scale))
        display = cv2.resize(image_bgr, (show_w, show_h), interpolation=cv2.INTER_AREA)

        pos: List[Tuple[int, int]] = []
        neg: List[Tuple[int, int]] = []

        def redraw() -> np.ndarray:
            canvas = display.copy()
            lines = [
                "C-only SAM2 segmentation",
                "Left click: C positive point | Right click: C negative point",
                "N/Enter: finish point selection | R: reset | ESC: cancel",
                "After points, press E to draw ROI polygons for excluded regions",
                f"C positive={len(pos)}, negative={len(neg)}",
            ]
            for i, s in enumerate(lines):
                cv2.putText(canvas, s, (18, 28 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA)

            for j, (x, y) in enumerate(pos):
                cv2.circle(canvas, (x, y), 6, (255, 255, 0), -1)
                cv2.putText(canvas, f"C+{j + 1}", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 0), 2, cv2.LINE_AA)
            for j, (x, y) in enumerate(neg):
                cv2.circle(canvas, (x, y), 7, (0, 0, 255), 2)
                cv2.line(canvas, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 255), 2)
                cv2.line(canvas, (x - 5, y + 5), (x + 5, y - 5), (0, 0, 255), 2)
                cv2.putText(canvas, f"C-{j + 1}", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 255), 2, cv2.LINE_AA)
            return canvas

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                pos.append((int(x), int(y)))
            elif event == cv2.EVENT_RBUTTONDOWN:
                neg.append((int(x), int(y)))

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, show_w, show_h)
        cv2.setMouseCallback(window_name, on_mouse)

        try:
            while True:
                cv2.imshow(window_name, redraw())
                key = cv2.waitKey(30) & 0xFF

                if key == 27:
                    raise RuntimeError("用户取消了 C 分割。")
                if key in (ord("r"), ord("R")):
                    pos.clear()
                    neg.clear()
                if key in (13, 10, ord("n"), ord("N")):
                    if len(pos) <= 0:
                        self.log("[RuleAB-C] C 分割至少需要 1 个正点。")
                        continue
                    break
        finally:
            try:
                cv2.destroyWindow(window_name)
            except Exception:
                pass

        def to_original(points: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
            return [(float(x) / scale, float(y) / scale) for x, y in points]

        original_pos = to_original(pos)
        original_neg = to_original(neg)

        # 点选完成后，进入 ROI 绘制模式（用户可按 E 进入/退出，或直接 Enter 跳过）。
        roi_polygons = select_roi_polygons_interactively(
            image_bgr=image_bgr,
            existing_polygons=existing_roi_polygons,
            window_name="Draw ROI for C segmentation (E=toggle, ESC=cancel)",
            scale=scale,
        )
        return original_pos, original_neg, roi_polygons

    def _predict_c_mask_by_sam2_points(
        self,
        image_rgb: np.ndarray,
        positive_points: List[Tuple[float, float]],
        negative_points: List[Tuple[float, float]],
        roi_polygons: Any = None,
    ) -> np.ndarray:
        """调用 SAM2 ImagePredictor，仅根据 C 的正负点分割 C，并应用 ROI 排除干扰区域。"""
        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception as e:
            raise ImportError(
                "SAM2 未正确安装或当前环境无法导入 sam2。请确认 sam2-main 已安装，并可导入 build_sam2 / SAM2ImagePredictor。"
            ) from e

        # 从 RuleABRuntimeConfig 默认值读取 SAM2 配置，保证与 A沿C绕行模块使用同一套权重。
        try:
            tmp_cfg = RuleABRuntimeConfig()
        except Exception:
            tmp_cfg = None

        sam2_cfg = str(getattr(tmp_cfg, "sam2_cfg", "configs/sam2.1/sam2.1_hiera_t.yaml"))
        sam2_checkpoint = str(
            getattr(
                tmp_cfg,
                "sam2_checkpoint",
                str(SAM2_REPO_ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt"),
            )
        )
        sam2_device = str(getattr(tmp_cfg, "sam2_device", "cuda"))

        self.log(
            f"[RuleAB-C] 正在加载 SAM2 进行 C 单独分割：cfg={sam2_cfg}, checkpoint={sam2_checkpoint}, device={sam2_device}"
        )

        sam2_model = build_sam2(sam2_cfg, sam2_checkpoint, device=sam2_device)
        predictor = SAM2ImagePredictor(sam2_model)
        predictor.set_image(image_rgb)

        coords: List[List[float]] = []
        labels: List[int] = []
        for x, y in positive_points:
            coords.append([float(x), float(y)])
            labels.append(1)
        for x, y in negative_points:
            coords.append([float(x), float(y)])
            labels.append(0)

        if not coords or 1 not in labels:
            raise ValueError("C 分割至少需要一个正点。")

        point_coords = np.array(coords, dtype=np.float32)
        point_labels = np.array(labels, dtype=np.int32)

        with torch.inference_mode():
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=None,
                multimask_output=True,
            )

        if masks is None or len(masks) == 0:
            raise RuntimeError("SAM2 没有返回 C mask。")

        # 不再只按 SAM2 原始 score 选 mask；先根据蓝色 C 正点/红色负点选择最可靠候选，
        # 再只保留“包含最多 C 正点、尽量不包含 C 负点”的主体连通区域。
        try:
            mask, best_idx, prompt_clean_debug = MeasurementWorkflow._select_and_clean_c_mask_from_sam2_candidates(
                masks,
                scores,
                positive_points=positive_points,
                negative_points=negative_points,
            )
        except Exception as e:
            best_idx = int(np.argmax(np.asarray(scores).reshape(-1)))
            mask = masks[best_idx].astype(bool)
            prompt_clean_debug = {"enabled": False, "reason": f"prompt_candidate_selection_failed:{e}"}

        area = int(mask.sum())
        if area <= 0:
            raise RuntimeError("SAM2 返回的 C mask 为空。")

        # 应用手动 ROI 排除区域
        if roi_polygons:
            try:
                from logic.roi_exclusion import polygons_to_mask
                h, w = mask.shape[:2]
                exclude_mask = polygons_to_mask((h, w), roi_polygons)
                if exclude_mask is not None:
                    mask = mask.astype(bool)
                    mask[exclude_mask] = False
                    area = int(mask.sum())
                    self.log(f"[RuleAB-C] 已应用 ROI 排除：polygons={len(roi_polygons)}, remaining_area={area}")
            except Exception as e:
                self.log(f"[RuleAB-C] 应用 ROI 排除失败：{e}")

        sel_score = float(np.asarray(scores).reshape(-1)[best_idx])
        selected_component = (prompt_clean_debug.get("component_clean_debug") or {}).get("selected_component", {}) if isinstance(prompt_clean_debug, dict) else {}
        self.log(
            f"[RuleAB-C] C 单独分割完成：selected_idx={best_idx}, sam2_score={sel_score:.6f}, "
            f"cleaned_area={area} px, pos_hits={selected_component.get('pos_hits', 'NA')}, "
            f"neg_hits={selected_component.get('neg_hits', 'NA')}, "
            f"clean_reason={(prompt_clean_debug.get('component_clean_debug') or {}).get('reason', prompt_clean_debug.get('reason', '')) if isinstance(prompt_clean_debug, dict) else ''}"
        )
        return mask

    @staticmethod
    def _order_quad_points_clockwise(points: np.ndarray) -> np.ndarray:
        """
        将四边形顶点整理为顺时针顺序，并从左上附近的点开始。
        输入/输出 shape=(4, 2)，坐标顺序为 x,y。
        """
        pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
        center = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        ordered = pts[np.argsort(angles)]

        # 让第一个点尽量是左上角，便于后续检查。
        start_idx = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
        ordered = np.roll(ordered, -start_idx, axis=0)
        return ordered.astype(np.float32)

    @staticmethod
    def _find_contours_compat(image: np.ndarray, mode: int, method: int):
        """委托给 MeasurementWorkflow 的同名静态方法，保持调用一致性。"""
        return MeasurementWorkflow._find_contours_compat(image, mode, method)

    def _fit_quadrilateral_from_mask(self, mask_bool: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        将 SAM2 原始 C mask 近似为四边形，并生成四边形 mask。

        返回：
            quad_points: shape=(4,2)，四个顶点，坐标为 x,y；
            quad_mask: bool mask，与输入 mask 同尺寸；
            method: 拟合方法说明。

        拟合策略：
            1. 取最大外轮廓；
            2. 先尝试 approxPolyDP 找到 4 点多边形；
            3. 如果失败，退回 minAreaRect 最小外接旋转矩形。
        """
        c_bool = mask_bool.astype(bool)
        h, w = c_bool.shape[:2]
        mask_u8 = (c_bool.astype(np.uint8) * 255)

        contours = self._find_contours_compat(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise RuntimeError("C mask 中没有找到有效轮廓，无法拟合四边形。")

        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area <= 1.0:
            raise RuntimeError(f"C mask 轮廓面积过小，无法拟合四边形：area={area:.3f}")

        peri = float(cv2.arcLength(contour, True))
        quad_points = None
        method = "minAreaRect_fallback"

        # 先从较小 epsilon 到较大 epsilon 搜索 4 个点的近似多边形。
        for eps_ratio in np.linspace(0.005, 0.12, 40):
            eps = max(1.0, float(eps_ratio) * peri)
            approx = cv2.approxPolyDP(contour, eps, True)
            if len(approx) == 4:
                quad_points = approx.reshape(4, 2).astype(np.float32)
                method = f"approxPolyDP_eps_ratio_{float(eps_ratio):.4f}"
                break

        # 如果直接轮廓没有 4 点，再尝试凸包。
        if quad_points is None:
            hull = cv2.convexHull(contour)
            hull_peri = float(cv2.arcLength(hull, True))
            for eps_ratio in np.linspace(0.005, 0.15, 50):
                eps = max(1.0, float(eps_ratio) * hull_peri)
                approx = cv2.approxPolyDP(hull, eps, True)
                if len(approx) == 4:
                    quad_points = approx.reshape(4, 2).astype(np.float32)
                    method = f"convexHull_approxPolyDP_eps_ratio_{float(eps_ratio):.4f}"
                    break

        # 最后兜底：最小外接旋转矩形。这个一定是四点。
        if quad_points is None:
            rect = cv2.minAreaRect(contour)
            quad_points = cv2.boxPoints(rect).astype(np.float32)
            method = "minAreaRect_fallback"

        quad_points = MeasurementWorkflowGUI._order_quad_points_clockwise(quad_points)

        # 限制点在图像范围内。
        quad_points[:, 0] = np.clip(quad_points[:, 0], 0, w - 1)
        quad_points[:, 1] = np.clip(quad_points[:, 1], 0, h - 1)

        quad_mask_u8 = np.zeros((h, w), dtype=np.uint8)
        quad_i32 = np.round(quad_points).astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(quad_mask_u8, [quad_i32], 255)
        quad_mask = quad_mask_u8.astype(bool)

        if int(quad_mask.sum()) <= 0:
            raise RuntimeError("四边形 mask 为空，C 四边形拟合失败。")

        return quad_points, quad_mask, method

    @staticmethod
    def _generate_c_edge_route_points_from_quad(
        quad_points: np.ndarray,
        spacing_px: float = 12.0,
        follow_direction: int = 1,
        safe_clearance_px: float = 0.0,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> List[Tuple[float, float]]:
        """
        根据 C 四边形四条边生成 A 的闭合运动路线。

        注意：这里生成的不是 C 的边界线本身，而是从 C 边界向外法线方向偏移
        safe_clearance_px 后的 A 目标路线。这样 A 按路线运动时会与 C 保持安全距离。
        """
        pts = np.asarray(quad_points, dtype=np.float32).reshape(4, 2)

        # 图像坐标系 y 轴向下，普通几何坐标里的顺/逆时针符号与屏幕视觉方向相反。
        # 约定：follow_direction >= 0 表示屏幕上逆时针绕行；follow_direction < 0 表示屏幕上顺时针绕行。
        # 对于图像坐标，shoelace 面积 < 0 才是屏幕视觉逆时针。
        try:
            signed_area = 0.5 * float(np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - pts[:, 1] * np.roll(pts[:, 0], -1)))
            is_screen_ccw = signed_area < 0.0
            want_screen_ccw = int(follow_direction) >= 0
            if is_screen_ccw != want_screen_ccw:
                pts = pts[::-1].copy()
        except Exception:
            if int(follow_direction) < 0:
                pts = pts[::-1].copy()

        spacing = max(1.0, float(spacing_px))
        clearance = max(0.0, float(safe_clearance_px))
        center = np.mean(pts, axis=0).astype(np.float32)
        route: List[Tuple[float, float]] = []
        for i in range(4):
            p0 = pts[i]
            p1 = pts[(i + 1) % 4]
            edge = p1 - p0
            length = float(np.linalg.norm(edge))
            if length <= 1e-6:
                continue
            # 两个候选法线，选“远离四边形中心”的那个作为外法线。
            n1 = np.array([-edge[1], edge[0]], dtype=np.float32) / length
            n2 = -n1
            mid = 0.5 * (p0 + p1)
            normal = n1 if float(np.dot(mid + n1 - center, mid + n1 - center)) >= float(np.dot(mid + n2 - center, mid + n2 - center)) else n2
            n = max(1, int(math.ceil(length / spacing)))
            for k in range(n):
                t = float(k) / float(n)
                p = (1.0 - t) * p0 + t * p1 + clearance * normal
                if image_shape is not None:
                    h, w = int(image_shape[0]), int(image_shape[1])
                    p[0] = np.clip(p[0], 0, max(0, w - 1))
                    p[1] = np.clip(p[1], 0, max(0, h - 1))
                route.append((float(p[0]), float(p[1])))
        # 去掉连续重复点
        cleaned: List[Tuple[float, float]] = []
        for x, y in route:
            if not cleaned or math.hypot(float(x) - cleaned[-1][0], float(y) - cleaned[-1][1]) >= 0.5:
                cleaned.append((float(x), float(y)))
        return cleaned

    def _save_static_c_map_result(
        self,
        image_rgb: np.ndarray,
        c_mask: np.ndarray,
        positive_points: List[Tuple[float, float]],
        negative_points: List[Tuple[float, float]],
        save_dir: Path,
        roi_polygons: Any = None,
    ) -> Path:
        """
        保存 C 固定分割结果，供“选择C文件夹”加载。

        重要：
            - static_c_sam2_mask.png 保存 SAM2 原始 C mask；
            - static_c_quad_mask.png 保存四边形近似后的 C mask；
            - static_c_mask.png 也保存四边形 mask，作为后续“加载C/选择C文件夹”的默认加载结果。

        因此后续 A 沿 C 运动时使用的是四边形近似后的 C，而不是原始不规则 SAM2 mask。
        """
        save_dir.mkdir(parents=True, exist_ok=True)

        c_raw_sam2_bool = c_mask.astype(bool)
        if int(c_raw_sam2_bool.sum()) <= 0:
            raise RuntimeError("C 原始 SAM2 mask 为空，不能保存 C 标定结果。")

        # 关键修复：蓝色点是 C 正点，红色点是 C 负点。
        # SAM2 得到原始 C_mask 后，先按正/负点做连通域主体筛选：
        #   只保留“包含最多 C 正点、尽量不包含 C 负点”的主体连通区域，
        #   再进行四边形拟合/圆形判断/路线生成。
        c_sam2_bool, c_prompt_clean_debug = MeasurementWorkflow._clean_c_mask_by_prompt_connected_component(
            c_raw_sam2_bool,
            positive_points=positive_points,
            negative_points=negative_points,
            point_radius_px=5,
            min_component_area_px=8,
        )
        if int(c_sam2_bool.sum()) <= 0:
            raise RuntimeError("C 正点连通域清理后 mask 为空，不能保存 C 标定结果。")
        if int(c_sam2_bool.sum()) != int(c_raw_sam2_bool.sum()):
            self.log(
                "[RuleAB-C] 已按 C 正点/负点清理 SAM2 C mask："
                f"raw_area={int(c_raw_sam2_bool.sum())} px -> cleaned_area={int(c_sam2_bool.sum())} px；"
                f"reason={c_prompt_clean_debug.get('reason', '')}；"
                f"selected={c_prompt_clean_debug.get('selected_component', {})}"
            )
        else:
            self.log(
                "[RuleAB-C] C mask 正点连通域清理未改变面积："
                f"area={int(c_sam2_bool.sum())} px；reason={c_prompt_clean_debug.get('reason', '')}"
            )

        quad_points, c_quad_bool, quad_method = self._fit_quadrilateral_from_mask(c_sam2_bool)

        # 后续加载使用四边形 C mask。
        yx_quad = np.column_stack(np.where(c_quad_bool))
        yx_sam2 = np.column_stack(np.where(c_sam2_bool))

        # 兼容旧加载逻辑的文件：static_c_mask.png / static_c_pixels_yx.npy / static_c_reference.csv
        # 这三个文件现在写入“四边形 C”。
        mask_path = save_dir / "static_c_mask.png"
        npy_path = save_dir / "static_c_pixels_yx.npy"
        csv_path = save_dir / "static_c_reference.csv"

        # 新增：raw SAM2 C、按正点清理后的 SAM2 C、四边形 C 分开保存，便于对比排查。
        raw_sam2_mask_path = save_dir / "static_c_raw_sam2_mask.png"
        sam2_mask_path = save_dir / "static_c_sam2_mask.png"
        sam2_npy_path = save_dir / "static_c_sam2_pixels_yx.npy"
        sam2_csv_path = save_dir / "static_c_sam2_reference.csv"
        quad_mask_path = save_dir / "static_c_quad_mask.png"
        quad_points_path = save_dir / "static_c_quad_points.json"
        quad_csv_path = save_dir / "static_c_quad_reference.csv"
        route_json_path = save_dir / "static_c_edge_route.json"
        route_csv_path = save_dir / "static_c_edge_route.csv"

        meta_path = save_dir / "static_c_meta.json"
        frame_path = save_dir / "static_c_frame.png"
        overlay_path = save_dir / "static_c_overlay.png"
        quad_overlay_path = save_dir / "static_c_quad_overlay.png"
        route_overlay_path = save_dir / "static_c_edge_route_overlay.png"

        # 1. 保存四边形 C：作为默认加载结果。
        cv2.imwrite(str(mask_path), (c_quad_bool.astype(np.uint8) * 255))
        np.save(str(npy_path), yx_quad.astype(np.int32))
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "y", "x"])
            for idx, (y, x) in enumerate(yx_quad):
                writer.writerow([idx, int(y), int(x)])

        # 2. 保存 SAM2 C：raw_sam2=清理前，static_c_sam2_mask=按 C 正点/负点清理后的主体 mask。
        cv2.imwrite(str(raw_sam2_mask_path), (c_raw_sam2_bool.astype(np.uint8) * 255))
        cv2.imwrite(str(sam2_mask_path), (c_sam2_bool.astype(np.uint8) * 255))
        np.save(str(sam2_npy_path), yx_sam2.astype(np.int32))
        with sam2_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "y", "x"])
            for idx, (y, x) in enumerate(yx_sam2):
                writer.writerow([idx, int(y), int(x)])

        # 3. 保存四边形专用文件。
        cv2.imwrite(str(quad_mask_path), (c_quad_bool.astype(np.uint8) * 255))
        with quad_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "y", "x"])
            for idx, (y, x) in enumerate(yx_quad):
                writer.writerow([idx, int(y), int(x)])
        quad_points_payload = {
            "method": quad_method,
            "points_xy": [[float(x), float(y)] for x, y in quad_points.tolist()],
            "raw_sam2_mask_area_px": int(c_raw_sam2_bool.sum()),
            "sam2_mask_area_px": int(c_sam2_bool.sum()),
            "quad_mask_area_px": int(c_quad_bool.sum()),
            "c_prompt_component_clean_debug": c_prompt_clean_debug,
        }
        with quad_points_path.open("w", encoding="utf-8") as f:
            json.dump(quad_points_payload, f, ensure_ascii=False, indent=2)

        # 3.5 根据 C 几何预生成 A 的运动路线。
        # Step7 会直接读取 static_c_edge_route.json；默认路线来源为 quad。
        # 新增：如果原始 C_mask 近似圆形，则路线使用圆形外接矩形/四边形，后续运动逻辑不变。
        cfg_gui_for_route = self.build_config_from_ui()
        route_geometry_points = quad_points
        route_circle_like = False
        route_circle_debug: Dict[str, Any] = {}
        route_circle_bbox_debug: Dict[str, Any] = {}
        try:
            route_circle_like, route_circle_debug = MeasurementWorkflow._judge_c_mask_circle_like(c_sam2_bool)
            if route_circle_like:
                route_geometry_points, _route_bbox_mask, route_circle_bbox_debug = MeasurementWorkflow._build_external_bbox_quad_from_mask(c_sam2_bool)
                self.workflow.log(
                    "[RuleAB-C] 检测到 C_mask 近似圆形：Step7 路线改用圆形外接矩形/四边形；"
                    f"circularity={float(route_circle_debug.get('circularity', 0.0)):.3f}, "
                    f"bbox={route_circle_bbox_debug.get('bbox_xyxy')}"
                )
        except Exception as e:
            route_circle_like = False
            self.workflow.log(f"[RuleAB-C] C_mask 圆形判断失败，继续使用原四边形路线：{e}")
        route_spacing_px = float(getattr(cfg_gui_for_route, "rule_ab_c_edge_route_spacing_px", 12.0))
        route_follow_direction = int(getattr(cfg_gui_for_route, "rule_ab_follow_c_direction", 1))
        route_min_clearance_px = max(0.0, float(getattr(cfg_gui_for_route, "rule_ab_route_min_distance_px", getattr(cfg_gui_for_route, "rule_ab_ac_min_clearance_px", 80.0))))
        route_max_clearance_px = max(route_min_clearance_px, float(getattr(cfg_gui_for_route, "rule_ab_route_max_distance_px", getattr(cfg_gui_for_route, "rule_ab_ac_max_clearance_px", 100.0))))
        route_safe_clearance_px = float(getattr(cfg_gui_for_route, "rule_ab_route_safe_target_distance_px", getattr(cfg_gui_for_route, "rule_ab_c_edge_route_safe_clearance_px", getattr(cfg_gui_for_route, "rule_ab_ac_target_clearance_px", 90.0))))
        if route_safe_clearance_px < 0:
            route_safe_clearance_px = 0.5 * (route_min_clearance_px + route_max_clearance_px)
        route_safe_clearance_px = max(route_min_clearance_px, min(route_max_clearance_px, route_safe_clearance_px))
        route_source = str(getattr(cfg_gui_for_route, "rule_ab_c_edge_route_source", "quad") or "quad").lower().strip()
        base_route_points: List[Tuple[float, float]] = []
        min_route_points: List[Tuple[float, float]] = []
        max_route_points: List[Tuple[float, float]] = []
        if route_source in ("quad", "quadrilateral"):
            base_route_points = self._generate_c_edge_route_points_from_quad(
                route_geometry_points, spacing_px=route_spacing_px, follow_direction=route_follow_direction, safe_clearance_px=0.0, image_shape=image_rgb.shape[:2]
            )
            min_route_points = self._generate_c_edge_route_points_from_quad(
                route_geometry_points, spacing_px=route_spacing_px, follow_direction=route_follow_direction, safe_clearance_px=route_min_clearance_px, image_shape=image_rgb.shape[:2]
            )
            route_points = self._generate_c_edge_route_points_from_quad(
                route_geometry_points, spacing_px=route_spacing_px, follow_direction=route_follow_direction, safe_clearance_px=route_safe_clearance_px, image_shape=image_rgb.shape[:2]
            )
            max_route_points = self._generate_c_edge_route_points_from_quad(
                route_geometry_points, spacing_px=route_spacing_px, follow_direction=route_follow_direction, safe_clearance_px=route_max_clearance_px, image_shape=image_rgb.shape[:2]
            )
            outer_edge_points: List[Tuple[float, float]] = []
            route_payload_source = "circle_bbox_quad_points_three_line_safety_band" if route_circle_like else "static_c_quad_points_three_line_safety_band"
        else:
            route_points, outer_edge_points = MeasurementWorkflow._generate_c_edge_route_points_from_mask_outer_edge(
                c_sam2_bool,
                spacing_px=route_spacing_px,
                follow_direction=route_follow_direction,
                safe_clearance_px=route_safe_clearance_px,
                image_shape=image_rgb.shape[:2],
            )
            route_payload_source = "sam2_outer_edge_pixels"

        route_payload = {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": route_payload_source,
            "source_mask": "static_c_sam2_mask.png" if (route_payload_source == "sam2_outer_edge_pixels" or route_circle_like) else "static_c_quad_points.json",
            "c_mask_shape_for_route": "circle_like_external_bbox" if route_circle_like else ("sam2_outer_edge" if route_payload_source == "sam2_outer_edge_pixels" else "quadrilateral"),
            "circle_like": bool(route_circle_like),
            "circle_debug": route_circle_debug,
            "circle_bbox_debug": route_circle_bbox_debug,
            "spacing_px": route_spacing_px,
            "safe_clearance_px": route_safe_clearance_px,
            "min_distance_px": route_min_clearance_px,
            "target_distance_px": route_safe_clearance_px,
            "max_distance_px": route_max_clearance_px,
            "follow_direction": route_follow_direction,
            "follow_direction_name": "顺时针" if int(route_follow_direction) < 0 else "逆时针",
            "route_loop": bool(getattr(cfg_gui_for_route, "rule_ab_route_loop", True)),
            "route_target_tolerance_px": float(getattr(cfg_gui_for_route, "rule_ab_route_target_tolerance_px", 8.0)),
            "outer_edge_point_count": len(outer_edge_points),
            "quad_points_xy": [[float(x), float(y)] for x, y in route_geometry_points.tolist()],
            "original_quad_points_xy": [[float(x), float(y)] for x, y in quad_points.tolist()],
            "base_route_points_xy": [[float(x), float(y)] for x, y in base_route_points],
            "min_route_points_xy": [[float(x), float(y)] for x, y in min_route_points],
            "route_points_xy": [[float(x), float(y)] for x, y in route_points],
            "target_route_points_xy": [[float(x), float(y)] for x, y in route_points],
            "max_route_points_xy": [[float(x), float(y)] for x, y in max_route_points],
            "note": ("检测到 C_mask 近似圆形：Step7 路线使用圆形外接矩形/四边形。" if route_circle_like else "") + "Step7 三线安全带：min/max 是边界线，target 是中间方向参考线；A 在 min-max 之间时只沿 target 切向运动，越界才法向修正。",
        }
        with route_json_path.open("w", encoding="utf-8") as f:
            json.dump(route_payload, f, ensure_ascii=False, indent=2)
        with route_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "x", "y"])
            for idx, (x, y) in enumerate(route_points):
                writer.writerow([idx, float(x), float(y)])

        # 额外保存 C 原始外轮廓像素点，便于检查路线来源是不是原 mask 外边缘。
        outer_edge_json_path = save_dir / "static_c_outer_edge_pixels.json"
        outer_edge_csv_path = save_dir / "static_c_outer_edge_pixels.csv"
        try:
            with outer_edge_json_path.open("w", encoding="utf-8") as f:
                json.dump({
                    "source": "static_c_sam2_mask.png",
                    "points_xy": [[float(x), float(y)] for x, y in outer_edge_points],
                }, f, ensure_ascii=False, indent=2)
            with outer_edge_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["index", "x", "y"])
                for idx, (x, y) in enumerate(outer_edge_points):
                    writer.writerow([idx, float(x), float(y)])
        except Exception:
            pass

        # 4. 保存原图和叠加图。
        frame_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(frame_path), frame_bgr)

        # overlay：黄色为 SAM2 原始 mask，绿色线为四边形。
        overlay = frame_bgr.copy()
        sam2_layer = overlay.copy()
        sam2_layer[c_sam2_bool] = (255, 255, 0)
        overlay = cv2.addWeighted(sam2_layer, 0.35, overlay, 0.65, 0)
        quad_i32 = np.round(quad_points).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [quad_i32], isClosed=True, color=(0, 255, 0), thickness=3)
        for i, (x, y) in enumerate(positive_points):
            cv2.circle(overlay, (int(round(x)), int(round(y))), 6, (255, 255, 0), -1)
            cv2.putText(overlay, f"C+{i + 1}", (int(round(x)) + 8, int(round(y)) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 0), 2, cv2.LINE_AA)
        for i, (x, y) in enumerate(negative_points):
            xx, yy = int(round(x)), int(round(y))
            cv2.circle(overlay, (xx, yy), 7, (0, 0, 255), 2)
            cv2.line(overlay, (xx - 5, yy - 5), (xx + 5, yy + 5), (0, 0, 255), 2)
            cv2.line(overlay, (xx - 5, yy + 5), (xx + 5, yy - 5), (0, 0, 255), 2)
            cv2.putText(overlay, f"C-{i + 1}", (xx + 8, yy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(overlay_path), overlay)

        # quad_overlay：只显示最终会被加载使用的四边形 mask。
        quad_overlay = frame_bgr.copy()
        quad_layer = quad_overlay.copy()
        quad_layer[c_quad_bool] = (0, 255, 0)
        quad_overlay = cv2.addWeighted(quad_layer, 0.35, quad_overlay, 0.65, 0)
        cv2.polylines(quad_overlay, [quad_i32], isClosed=True, color=(0, 255, 0), thickness=3)
        # 在 quad_overlay 上显示三线安全带：min=绿色，target=紫色，max=红色。
        for _pts, _color, _thick in (
            (min_route_points if 'min_route_points' in locals() else [], (0, 255, 0), 2),
            (max_route_points if 'max_route_points' in locals() else [], (0, 0, 255), 2),
        ):
            if _pts:
                _i32 = np.round(np.asarray(_pts, dtype=np.float32)).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(quad_overlay, [_i32], isClosed=True, color=_color, thickness=_thick)
        if route_points:
            route_i32 = np.round(np.asarray(route_points, dtype=np.float32)).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(quad_overlay, [route_i32], isClosed=True, color=(255, 0, 255), thickness=2)
            for ridx, (rx, ry) in enumerate(route_points[::max(1, len(route_points)//12)]):
                cv2.circle(quad_overlay, (int(round(rx)), int(round(ry))), 3, (255, 0, 255), -1)
        for idx, (x, y) in enumerate(quad_points.tolist()):
            cv2.circle(quad_overlay, (int(round(x)), int(round(y))), 5, (0, 0, 255), -1)
            cv2.putText(quad_overlay, f"Q{idx + 1}", (int(round(x)) + 8, int(round(y)) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(quad_overlay_path), quad_overlay)

        # route_overlay：在“原始截取图像”上单独绘制 A 的完整运动路径。
        # 这个文件用于直接检查 Step7 的 A 路线是否来自 C 近似四边形边界，
        # 以及 A 中心后续是否能被约束在路线安全带内。
        route_overlay = frame_bgr.copy()
        # 黄色线：SAM2 原始 C mask 外轮廓像素；绿色线：近似四边形；紫色线：Step7 实际 A 路线。
        try:
            if outer_edge_points:
                edge_i32 = np.round(np.asarray(outer_edge_points, dtype=np.float32)).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(route_overlay, [edge_i32], isClosed=True, color=(0, 255, 255), thickness=2)
            cv2.polylines(route_overlay, [quad_i32], isClosed=True, color=(0, 255, 0), thickness=1)
        except Exception:
            pass
        # route_overlay：base=灰色 C 边界，min=绿色内边界，target=紫色中线，max=红色外边界。
        for _pts, _color, _thick in (
            (base_route_points if 'base_route_points' in locals() else [], (160, 160, 160), 1),
            (min_route_points if 'min_route_points' in locals() else [], (0, 255, 0), 2),
            (max_route_points if 'max_route_points' in locals() else [], (0, 0, 255), 2),
        ):
            if _pts:
                _i32 = np.round(np.asarray(_pts, dtype=np.float32)).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(route_overlay, [_i32], isClosed=True, color=_color, thickness=_thick)
        if route_points:
            route_i32 = np.round(np.asarray(route_points, dtype=np.float32)).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(route_overlay, [route_i32], isClosed=True, color=(255, 0, 255), thickness=2)
            # 起点/终点和路线采样点。BGR: magenta=路线，green=起点，red=终点。
            sx, sy = route_points[0]
            ex, ey = route_points[-1]
            cv2.circle(route_overlay, (int(round(sx)), int(round(sy))), 7, (0, 255, 0), -1)
            cv2.putText(route_overlay, "A-route START", (int(round(sx)) + 8, int(round(sy)) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(route_overlay, (int(round(ex)), int(round(ey))), 7, (0, 0, 255), -1)
            cv2.putText(route_overlay, "END", (int(round(ex)) + 8, int(round(ey)) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
            sample_step = max(1, len(route_points) // 24)
            for ridx, (rx, ry) in enumerate(route_points[::sample_step]):
                cv2.circle(route_overlay, (int(round(rx)), int(round(ry))), 3, (255, 0, 255), -1)
            cv2.putText(
                route_overlay,
                f"A route from {route_payload_source}: {len(route_points)} pts, spacing={route_spacing_px:.1f}px, min/target/max={route_min_clearance_px:.1f}/{route_safe_clearance_px:.1f}/{route_max_clearance_px:.1f}px",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(route_overlay, "A motion route EMPTY", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(route_overlay_path), route_overlay)

        # 保存手动 ROI 排除区域（如果标定时绘制了）。
        roi_json_path = save_dir / "exclude_roi_polygons.json"
        roi_overlay_path = save_dir / "static_c_roi_overlay.png"
        roi_polys: List[List[List[float]]] = []
        try:
            from logic.roi_exclusion import normalize_roi_polygons, draw_roi_overlay
            roi_polys = normalize_roi_polygons(roi_polygons)
            if roi_polys:
                with roi_json_path.open("w", encoding="utf-8") as f:
                    json.dump({"exclude_roi_polygons": roi_polys}, f, ensure_ascii=False, indent=2)
                roi_overlay = draw_roi_overlay(frame_bgr, roi_polys, color=(0, 0, 255), alpha=0.35)
                cv2.imwrite(str(roi_overlay_path), roi_overlay)
                self.log(f"[RuleAB-C] 已保存 ROI 排除区域：polygons={len(roi_polys)}")
        except Exception as e:
            self.log(f"[RuleAB-C] 保存 ROI 排除区域失败：{e}")

        cfg_gui = self.build_config_from_ui()
        meta = {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "capture_area": list(cfg_gui.capture_area),
            "positive_points": [[float(x), float(y)] for x, y in positive_points],
            "negative_points": [[float(x), float(y)] for x, y in negative_points],
            "saved_static_c_mask_type": "quadrilateral",
            "c_prompt_component_clean_debug": c_prompt_clean_debug,
            "quad_fit_method": quad_method,
            "quad_points_xy": [[float(x), float(y)] for x, y in quad_points.tolist()],
            "edge_route_source": str(route_payload_source),
            "edge_route_points_count": int(len(route_points)),
            "outer_edge_points_count": int(len(outer_edge_points)),
            "edge_route_spacing_px": float(route_spacing_px),
            "edge_route_safe_clearance_px": float(route_safe_clearance_px),
            "edge_route_follow_direction": int(route_follow_direction),
            "raw_sam2_mask_area_px": int(c_raw_sam2_bool.sum()),
            "sam2_mask_area_px": int(c_sam2_bool.sum()),
            "quad_mask_area_px": int(c_quad_bool.sum()),
            "exclude_roi_polygons": roi_polys,
            "c_prompt_component_clean_debug": c_prompt_clean_debug,
            "files": {
                "static_c_mask": str(mask_path.name),
                "static_c_pixels_yx": str(npy_path.name),
                "static_c_reference_csv": str(csv_path.name),
                "static_c_raw_sam2_mask": str(raw_sam2_mask_path.name),
                "static_c_sam2_mask": str(sam2_mask_path.name),
                "static_c_sam2_pixels_yx": str(sam2_npy_path.name),
                "static_c_sam2_reference_csv": str(sam2_csv_path.name),
                "static_c_quad_mask": str(quad_mask_path.name),
                "static_c_quad_points": str(quad_points_path.name),
                "static_c_quad_reference_csv": str(quad_csv_path.name),
                "static_c_edge_route_json": str(route_json_path.name),
                "static_c_edge_route_csv": str(route_csv_path.name),
                "static_c_outer_edge_pixels_json": "static_c_outer_edge_pixels.json",
                "static_c_outer_edge_pixels_csv": "static_c_outer_edge_pixels.csv",
                "static_c_frame": str(frame_path.name),
                "static_c_overlay": str(overlay_path.name),
                "static_c_quad_overlay": str(quad_overlay_path.name),
                "static_c_edge_route_overlay": str(route_overlay_path.name),
                "exclude_roi_polygons_json": str(roi_json_path.name),
                "static_c_roi_overlay": str(roi_overlay_path.name),
            },
        }
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        self.log(
            "[RuleAB-C] C 已四边形近似并保存："
            f"SAM2原始面积={int(c_raw_sam2_bool.sum())} px，"
            f"正点清理后面积={int(c_sam2_bool.sum())} px，"
            f"四边形面积={int(c_quad_bool.sum())} px，"
            f"method={quad_method}；已生成近似四边形A路线点数={len(route_points)}；"
            "后续加载使用 static_c_mask.png=四边形mask，Step7 使用 static_c_edge_route.json（默认由近似四边形四条边生成）"
        )

        return save_dir

    def segment_c_only(self):
        """
        单独分割 C：
            1. 只截取当前固定区域；
            2. 只用 SAM2 对 C 做多正点+多负点分割；
            3. 将 SAM2 C mask 近似为四边形；
            4. 保存 static_c_mask.png 等固定 C 文件，其中 static_c_mask.png 是四边形 mask；
            5. 自动把保存目录写入“C标定文件夹”，后续“单独A沿C绕行”会加载该四边形 C。
        """
        try:
            self.set_var(self.rule_module_status_var, "规则模块状态：正在单独分割C")
            self.log("========== 单独分割C：分割 C → 四边形近似 → 保存为后续加载C，不分割 A/B，不移动控制器 ==========")

            cfg_gui = self.build_config_from_ui()
            save_root = Path(str(getattr(cfg_gui, "save_root", "measurement_output")))
            map_name = str(getattr(cfg_gui, "rule_ab_static_c_map_name", "default_static_c_map")).strip() or "default_static_c_map"
            safe_map_name = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in map_name)

            current_dir_text = str(getattr(cfg_gui, "rule_ab_static_c_map_dir", "")).strip()
            current_dir = Path(current_dir_text) if current_dir_text else None
            if current_dir is not None and current_dir_text:
                # 如果用户已经选择了一个 C 文件夹，则直接覆盖/更新该文件夹中的 C 分割结果。
                save_dir = current_dir
                if save_dir.name != "static_c_reference" and (save_dir / "static_c_reference").exists():
                    save_dir = save_dir / "static_c_reference"
            else:
                save_dir = save_root / "rule_ab_follow_c_only" / "_static_c_map_library" / safe_map_name

            image_rgb = self._capture_current_rule_ab_frame(output_dir=save_dir)
            pos, neg, roi_polygons = self._select_c_points_interactively(
                image_rgb=image_rgb,
                window_name="SAM2 C only: left positive, right negative",
                scale=float(getattr(cfg_gui, "rule_ab_confirm_window_scale", 0.85)),
            )

            self.log(f"[RuleAB-C] C 点提示：positive={len(pos)}, negative={len(neg)}, ROIs={len(roi_polygons)}")
            c_mask = self._predict_c_mask_by_sam2_points(
                image_rgb=image_rgb,
                positive_points=pos,
                negative_points=neg,
                roi_polygons=roi_polygons,
            )
            saved_dir = self._save_static_c_map_result(
                image_rgb=image_rgb,
                c_mask=c_mask,
                positive_points=pos,
                negative_points=neg,
                save_dir=save_dir,
                roi_polygons=roi_polygons,
            )

            self.rule_ab_static_c_map_dir_var.set(str(saved_dir))
            self.rule_ab_load_static_c_map_var.set(True)
            self.rule_ab_force_reselect_c_var.set(False)

            self.set_var(
                self.rule_module_status_var,
                f"规则模块状态：C分割+四边形近似完成，已保存并加载：{saved_dir}",
            )
            self.log(f"[RuleAB-C] C 分割及四边形近似结果已保存，并设置为后续加载目录：{saved_dir}")

        except Exception as e:
            self.set_var(self.rule_module_status_var, "规则模块状态：C分割失败")
            self.log(f"[RuleAB-C] 单独分割C失败：{e}")
            self.log(traceback.format_exc())
            error_msg = str(e)
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("C分割失败", msg))

    @staticmethod
    def _rule_ab_follow_direction_to_label(value: Any) -> str:
        """把内部方向值转换成 GUI 显示文字。+1=逆时针，-1=顺时针。"""
        try:
            return "顺时针" if int(value) < 0 else "逆时针"
        except Exception:
            s = str(value).strip().lower()
            if s in ("cw", "clockwise", "顺时针", "-1", "negative"):
                return "顺时针"
            return "逆时针"

    @staticmethod
    def _rule_ab_follow_direction_label_to_value(value: Any) -> int:
        """把 GUI 显示文字转换成内部方向值。+1=逆时针，-1=顺时针。"""
        s = str(value).strip().lower()
        if s in ("顺时针", "cw", "clockwise", "-1", "negative"):
            return -1
        if s in ("逆时针", "ccw", "counterclockwise", "counter-clockwise", "1", "+1", "positive"):
            return 1
        try:
            return -1 if int(float(s)) < 0 else 1
        except Exception:
            return 1

    def get_rule_ab_follow_c_direction_value(self) -> int:
        """读取 GUI 中的 C 路线方向选择，供路线生成和 Step7 实时运动使用。"""
        return self._rule_ab_follow_direction_label_to_value(self.rule_ab_follow_c_direction_var.get())

    def get_rule_ab_follow_c_direction_label(self) -> str:
        return self._rule_ab_follow_direction_to_label(self.get_rule_ab_follow_c_direction_value())

    def build_config_from_ui(self) -> MeasurementConfig:
        return MeasurementConfig(
            hardware_mode=str(self.hardware_mode_var.get()).strip() or "real",
            virtual_spectrum_mode=str(self.virtual_spectrum_mode_var.get()).strip() or "gaussian",
            virtual_spectrum_replay_csv=str(self.virtual_spectrum_replay_csv_var.get()).strip(),

            max_cycles=int(self.max_cycles_var.get()),
            sub_loop_iterations_per_cycle=int(self.sub_loop_iterations_per_cycle_var.get()),

            signal_on_time_ms=float(self.signal_on_time_ms_var.get()),
            stable_wait_ms=int(self.stable_wait_ms_var.get()),
            signal_time_factor=float(self.signal_time_factor_var.get()),
            angle_delta_min_deg=float(self.angle_delta_min_var.get()),
            angle_delta_max_deg=float(self.angle_delta_max_var.get()),

            save_root=self.save_root_var.get().strip(),

            light_port=self.light_port_var.get().strip(),

            rigol_visa=self.rigol_visa_var.get().strip(),

            angle_model_path=self.angle_model_path_var.get().strip(),
            capture_area=self.parse_capture_area(),
            angle_output_dir="outputs/captured_frames",
            angle_num=self.parse_int_var(self.num_var, "num"),
            angle_cw=self.parse_int_var(self.cw_var, "cw"),

            tcp_host=self.tcp_host_var.get().strip(),
            tcp_port=int(self.tcp_port_var.get()),
            tcp_output_dir=self.tcp_output_dir_var.get().strip(),
            tcp_command=self.tcp_command_var.get().strip(),

            # 日志保存配置
            save_log_to_file=bool(self.save_log_to_file_var.get()),
            log_dir=self.log_dir_var.get().strip(),

            # 新增：PI 光谱仪配置
            spectrometer_backend=str(self.spectrometer_backend_var.get()).strip().lower() or "labview_tcp",
            picam_dll_path=None,
            picam_camera_index=0,
            picam_exposure=float(self.picam_exposure_var.get()),
            picam_temperature=float(self.picam_temperature_var.get()),
            picam_roi_x=0,
            picam_roi_y=0,
            picam_roi_width=int(self.picam_roi_width_var.get()),
            picam_roi_height=int(self.picam_roi_height_var.get()),

            raw_remove_above=float(self.raw_remove_above_var.get()),
            median_filter_window=int(self.median_filter_window_var.get()),
            x_axis_xlsx_path=self.x_axis_xlsx_path_var.get().strip(),

            enable_delta_w_judge=bool(self.enable_delta_w_judge_var.get()),
            delta_w_threshold=float(self.delta_w_threshold_var.get()),
            stop_when_delta_w_not_enough=bool(self.stop_when_delta_w_not_enough_var.get()),

            rule_ab_enable_stage=bool(self.rule_ab_enable_stage_var.get()),

            # 单独A推动B不再使用最大推动次数；该值仅保留给完整测量流程/旧逻辑。
            rule_ab_max_steps=int(self.rule_ab_max_steps_var.get()),
            rule_ab_angle_delta_min_deg=float(self.rule_ab_delta_min_var.get()),
            rule_ab_angle_delta_max_deg=float(self.rule_ab_delta_max_var.get()),

            rule_ab_ab_close_threshold_px=float(self.rule_ab_ab_close_threshold_var.get()),
            rule_ab_ab_overlap_min_area_px=float(self.rule_ab_ab_overlap_threshold_var.get()),
            rule_ab_ac_target_clearance_px=float(self.rule_ab_ac_target_clearance_var.get()),
            rule_ab_ac_min_clearance_px=float(self.rule_ab_ac_min_clearance_var.get()),
            rule_ab_ac_max_clearance_px=float(self.rule_ab_ac_max_clearance_var.get()),
            rule_ab_ac_correction_mode="nearest_normal",
            rule_ab_follow_c_direction=self.get_rule_ab_follow_c_direction_value(),

            # Step7 路线强制使用近似四边形；安全带直接跟随 GUI 的 A-C min/target/max。
            # 因此运行中只改 GUI 里的 A-C距离 min/target/max px，下一步就会按新值运动。
            rule_ab_use_c_edge_route=True,
            rule_ab_c_edge_route_source="quad",
            rule_ab_c_edge_route_safe_clearance_px=float(self.rule_ab_ac_target_clearance_var.get()),
            rule_ab_route_min_distance_px=float(self.rule_ab_ac_min_clearance_var.get()),
            rule_ab_route_safe_target_distance_px=float(self.rule_ab_ac_target_clearance_var.get()),
            rule_ab_route_max_distance_px=float(self.rule_ab_ac_max_clearance_var.get()),

            # 单独 A沿C绕行 / 完整流程 RuleAB 默认使用你的 Thorlabs 控制器 97101208 和 3/4 通道。
            rule_ab_stage_conn="97101208",
            rule_ab_stage_x_channel=3,
            rule_ab_stage_y_channel=4,

            # 兼容 RuntimeConfig 的默认速度/加速度，同时按 CH3/CH4 分别设置实际通道参数。
            rule_ab_stage_velocity=int(self.rule_ab_ch3_velocity_var.get()),
            rule_ab_stage_acceleration=int(self.rule_ab_ch3_acceleration_var.get()),
            rule_ab_stage_max_voltage=max(int(self.rule_ab_ch3_max_voltage_var.get()), int(self.rule_ab_ch4_max_voltage_var.get())),
            rule_ab_stage_ch3_velocity=int(self.rule_ab_ch3_velocity_var.get()),
            rule_ab_stage_ch3_acceleration=int(self.rule_ab_ch3_acceleration_var.get()),
            rule_ab_stage_ch4_velocity=int(self.rule_ab_ch4_velocity_var.get()),
            rule_ab_stage_ch4_acceleration=int(self.rule_ab_ch4_acceleration_var.get()),
            rule_ab_stage_ch3_max_voltage=int(self.rule_ab_ch3_max_voltage_var.get()),
            rule_ab_stage_ch4_max_voltage=int(self.rule_ab_ch4_max_voltage_var.get()),
            rule_ab_stage_step_x=int(self.rule_ab_stage_step_x_var.get()),
            rule_ab_stage_step_y=int(self.rule_ab_stage_step_y_var.get()),
            rule_ab_action_step=int(self.rule_ab_action_step_var.get()),
            rule_ab_stage_ch3_pause_after_move_s=float(self.rule_ab_ch3_pause_after_move_s_var.get()),
            rule_ab_stage_ch4_pause_after_move_s=float(self.rule_ab_ch4_pause_after_move_s_var.get()),

            rule_ab_static_c_map_name=self.rule_ab_static_c_map_name_var.get().strip() or "default_static_c_map",
            rule_ab_static_c_map_dir=self.rule_ab_static_c_map_dir_var.get().strip(),
            rule_ab_reuse_static_c_map_in_memory=True,
            rule_ab_load_static_c_map_if_exists=bool(self.rule_ab_load_static_c_map_var.get()),
            rule_ab_save_static_c_map_library=True,
            rule_ab_force_reselect_c_each_run=bool(self.rule_ab_force_reselect_c_var.get()),

            rule_ab_stage_x_sign=-1,
            rule_ab_stage_y_sign=1,

            rule_ac_area_threshold_px=float(self.rule_ac_area_threshold_var.get()),
            rule_ac_enable_stage=bool(self.rule_ac_enable_stage_var.get()),
            rule_ac_max_cycles=int(self.rule_ac_max_cycles_var.get()),
            rule_ac_stage_step_size=int(self.rule_ac_stage_step_size_var.get()),
            rule_ac_target_x_px=float(self.rule_ac_target_x_var.get()),
            rule_ac_target_y_px=float(self.rule_ac_target_y_var.get()),
            rule_ac_center_tolerance_px=float(self.rule_ac_center_tolerance_var.get()),
            rule_ac_stage12_velocity=int(self.rule_ac_stage12_velocity_var.get()),
            rule_ac_stage12_acceleration=int(self.rule_ac_stage12_acceleration_var.get()),
            rule_ac_stage12_max_voltage=int(self.rule_ac_stage12_max_voltage_var.get()),
            rule_ac_color_mode=str(self.rule_ac_color_mode_var.get()).strip() or "include",
            rule_ac_color_h=int(self.rule_ac_color_h_var.get()),
            rule_ac_color_s=int(self.rule_ac_color_s_var.get()),
            rule_ac_color_v=int(self.rule_ac_color_v_var.get()),
            rule_ac_color_h_tol=int(self.rule_ac_color_h_tol_var.get()),
            rule_ac_color_s_tol=int(self.rule_ac_color_s_tol_var.get()),
            rule_ac_color_v_tol=int(self.rule_ac_color_v_tol_var.get()),
            rule_ac_color_min_area_px=int(self.rule_ac_color_min_area_var.get()),
            rule_ac_color_morph_kernel=int(self.rule_ac_color_morph_kernel_var.get()),

            calibration_path=str((Path(self.save_root_var.get().strip() or "measurement_output") / "calibration" / "current_calibration.json")),
            require_full_calibration_before_run=True,

            focus_roi=self._parse_focus_roi(self.saf_roi_var.get()),
            saf_capture_area=self._parse_focus_roi(self.saf_capture_area_var.get()),
            saf_focus_roi=self._parse_focus_roi(self.saf_roi_var.get()),
            focus_trigger_ratio=float(self.saf_trigger_ratio_var.get()),
            focus_stop_ratio=float(self.saf_stop_ratio_var.get()),
            focus_trigger_count=int(self.saf_trigger_count_var.get()),
            focus_trigger_absolute=bool(self.saf_trigger_absolute_var.get()),
            focus_detection_only=bool(self.saf_detection_only_var.get()),
            focus_z_enabled=bool(self.saf_z_enabled_var.get()),
            focus_z_axis=int(self.saf_z_axis_var.get()),
            focus_z_speed=int(self.saf_z_speed_var.get()),
            focus_z_accel=int(self.saf_z_accel_var.get()),
            focus_search_strategy=str(self.saf_search_strategy_var.get()),
            focus_z_search_steps=int(self.saf_z_search_steps_var.get()),
            focus_z_patience=int(self.saf_z_patience_var.get()),
            focus_z_direction_probe_steps=tuple(
                max(1, int(self.saf_z_direction_probe_step_interval_var.get())) * i
                for i in range(1, max(1, int(self.saf_z_direction_probe_stage_count_var.get())) + 1)
            ),
            focus_z_direction_probe_stage_count=max(1, int(self.saf_z_direction_probe_stage_count_var.get())),
            focus_z_direction_probe_step_interval=max(1, int(self.saf_z_direction_probe_step_interval_var.get())),
            focus_z_direction_probe_samples=int(self.saf_z_direction_probe_samples_var.get()),
            focus_z_direction_probe_points_per_step=int(self.saf_z_direction_probe_points_var.get()),
            focus_z_local_refine_enabled=bool(self.saf_z_local_refine_enabled_var.get()),
            focus_z_local_refine_decay=float(self.saf_z_local_refine_decay_var.get()),
            focus_z_local_refine_min_step=int(self.saf_z_local_refine_min_step_var.get()),
            focus_z_local_refine_max_rounds=int(self.saf_z_local_refine_max_rounds_var.get()),
        )
    
    def sync_config_from_ui_to_workflow(self) -> MeasurementWorkflow:
        """
        把 GUI 当前输入框里的参数同步到 workflow.cfg。

        运行中的同步策略：
            1. 每一轮开始前调用本函数，重新读取 GUI 当前输入框；
            2. 除“信号ON/ms”外，基本测量参数都会在下一轮直接生效；
            3. “信号ON/ms”保留原有策略：用户手动改动优先，否则使用 Step 11 自动计算值；
            4. num/cw 属于角度检测运行参数，运行中修改后下一轮生效，且不重新加载模型；
            5. 只有模型路径或截图区域变化时，才重建 ScreenAngleDetector。
        """
        new_cfg = self.build_config_from_ui()

        # 如果 workflow 还不存在，就新建。
        if self.workflow is None:
            self.workflow = MeasurementWorkflow(
                new_cfg,
                on_log=self.log,
                on_update=self.refresh_result_labels,
            )
            self.log("[配置同步] workflow 不存在，已根据 GUI 当前参数新建 workflow")
            return self.workflow

        signal_time_source = "GUI参数"

        # 运行过程中信号ON时间的优先级：
        # 1. 用户在测量过程中手动修改“信号ON/ms” -> 下一轮使用用户修改值；
        # 2. 用户没有手动修改 -> 下一轮使用 Step 11 自动计算后保存在 workflow.cfg 中的值。
        if self.workflow.is_measuring:
            if self.signal_on_time_user_modified:
                signal_time_source = "用户手动修改"
                self.signal_on_time_user_modified = False
            else:
                signal_time_source = "自动计算下一轮"
                new_cfg.signal_on_time_ms = float(self.workflow.cfg.signal_on_time_ms)
                self.set_signal_on_time_var_programmatically(round(new_cfg.signal_on_time_ms, 3))

        old_cfg = self.workflow.cfg

        save_root_changed = old_cfg.save_root != new_cfg.save_root
        hardware_mode_changed = str(getattr(old_cfg, "hardware_mode", "real")) != str(getattr(new_cfg, "hardware_mode", "real"))
        if hardware_mode_changed:
            self.log(f"[配置同步] hardware_mode 变化：{old_cfg.hardware_mode} -> {new_cfg.hardware_mode}；将重置硬件对象，保留标定/保存逻辑。")
            try:
                self.workflow.close_all()
            except Exception as e:
                self.log(f"[配置同步] 切换 hardware_mode 时关闭旧硬件对象失败：{e}")


        # 这些变化会影响 ScreenAngleDetector 的底层对象，需要重建。
        angle_detector_reinit_changed = (
            old_cfg.angle_model_path != new_cfg.angle_model_path or
            old_cfg.capture_area != new_cfg.capture_area
        )

        # 这些变化只影响检测运行参数，不需要重新加载 YOLO 模型。
        angle_runtime_param_changed = (
            old_cfg.angle_num != new_cfg.angle_num or
            old_cfg.angle_cw != new_cfg.angle_cw
        )

        rigol_param_changed = (
            old_cfg.ch1_low_v != new_cfg.ch1_low_v or
            old_cfg.ch1_high_v != new_cfg.ch1_high_v or
            old_cfg.ch1_freq_hz != new_cfg.ch1_freq_hz or
            old_cfg.ch1_duty_percent != new_cfg.ch1_duty_percent or
            old_cfg.ch1_delay_s != new_cfg.ch1_delay_s or
            old_cfg.ch2_low_v != new_cfg.ch2_low_v or
            old_cfg.ch2_high_v != new_cfg.ch2_high_v or
            old_cfg.ch2_freq_hz != new_cfg.ch2_freq_hz or
            old_cfg.ch2_duty_percent != new_cfg.ch2_duty_percent or
            old_cfg.ch2_delay_s != new_cfg.ch2_delay_s
        )

        rule_ab_param_changed = (
            old_cfg.rule_ab_enable_stage != new_cfg.rule_ab_enable_stage or
            old_cfg.rule_ab_max_steps != new_cfg.rule_ab_max_steps or
            old_cfg.rule_ab_angle_delta_min_deg != new_cfg.rule_ab_angle_delta_min_deg or
            old_cfg.rule_ab_angle_delta_max_deg != new_cfg.rule_ab_angle_delta_max_deg or
            old_cfg.rule_ab_stage_ch4_pause_after_move_s != new_cfg.rule_ab_stage_ch4_pause_after_move_s or
            old_cfg.capture_area != new_cfg.capture_area
        )

        rule_ac_param_changed = (
            old_cfg.rule_ac_area_threshold_px != new_cfg.rule_ac_area_threshold_px or
            old_cfg.rule_ac_enable_stage != new_cfg.rule_ac_enable_stage or
            old_cfg.rule_ac_max_cycles != new_cfg.rule_ac_max_cycles or
            old_cfg.rule_ac_stage_step_size != new_cfg.rule_ac_stage_step_size or
            old_cfg.rule_ac_target_x_px != new_cfg.rule_ac_target_x_px or
            old_cfg.rule_ac_target_y_px != new_cfg.rule_ac_target_y_px or
            old_cfg.rule_ac_center_tolerance_px != new_cfg.rule_ac_center_tolerance_px or
            old_cfg.rule_ac_stage12_velocity != new_cfg.rule_ac_stage12_velocity or
            old_cfg.rule_ac_stage12_acceleration != new_cfg.rule_ac_stage12_acceleration or
            old_cfg.rule_ac_stage12_max_voltage != new_cfg.rule_ac_stage12_max_voltage or
            old_cfg.capture_area != new_cfg.capture_area
        )

        # 核心：把 GUI 参数同步到 workflow.cfg。
        self.workflow.cfg = new_cfg

        # 日志保存配置变化时，重建日志管理器
        log_config_changed = (
            bool(getattr(old_cfg, "save_log_to_file", True)) != bool(getattr(new_cfg, "save_log_to_file", True))
            or str(getattr(old_cfg, "log_dir", "./Log")) != str(getattr(new_cfg, "log_dir", "./Log"))
        )
        if log_config_changed:
            try:
                self.workflow._shutdown_log_manager()
                self.workflow._init_log_manager()
                self.log(f"[日志] 日志配置变化：save_log_to_file={new_cfg.save_log_to_file}, log_dir={new_cfg.log_dir}")
            except Exception as e:
                self.log(f"[日志] 重建日志管理器失败：{e}")

        # 如果已经加载完整标定，GUI 同步后必须重新应用标定包。
        # 否则 build_config_from_ui() 会把 confirm_first_frame_segmentation/static C/Step9 HSV 等恢复成 GUI 默认值，
        # 导致完整测量中后续 RuleAB/SAM2 又弹出标点窗口。
        loaded_state = self.workflow._get_loaded_calibration_state()
        if loaded_state is not None:
            if self.workflow.is_measuring or bool(self.workflow.context.get("full_measurement_mode", False)):
                # 运行中不要再用完整标定包覆盖 GUI 实时修改的 RuleAB/Step9 参数。
                # 这里只保留完整标定上下文，并强制锁定完整测量专用 C。
                # 这样 A-C距离、CH3/CH4速度/加速度/步数/停顿、Step9目标/HSV/容差等
                # 都可以在 GUI 中实时修改并在下一步生效。
                self.workflow.context["calibration_state"] = loaded_state.to_dict()
                self.workflow.context["calibration_loaded_for_runtime"] = True
                self.workflow._force_cfg_to_strict_full_calibration_c(reason="sync_config_from_ui_to_workflow")
            else:
                # 非运行状态下加载标定，仍然把标定值同步到 GUI/运行配置。
                self.workflow.apply_calibration_state(loaded_state)
            new_cfg = self.workflow.cfg

        # Step9 颜色检测参数属于运行时参数。
        # 如果用户在完整测量过程中修改 HSV/mode，下一轮 Step9 检测必须使用新值，
        # 不能继续使用加载标定时缓存的 self.step9_color_hsv。
        try:
            self.workflow.step9_color_mode = str(getattr(new_cfg, "rule_ac_color_mode", "include"))
            h = int(getattr(new_cfg, "rule_ac_color_h", -1))
            s_val = int(getattr(new_cfg, "rule_ac_color_s", -1))
            v = int(getattr(new_cfg, "rule_ac_color_v", -1))
            if h >= 0 and s_val >= 0 and v >= 0:
                self.workflow.step9_color_hsv = (h, s_val, v)
            else:
                self.workflow.step9_color_hsv = None
        except Exception:
            pass

        # 保存目录变化：立即同步 output_root。
        # 如果正在运行且本次点击的 run_session_name 已存在，则把后续轮次保存到新 save_root 下同名批次目录。
        if save_root_changed:
            self.workflow.output_root = Path(new_cfg.save_root)
            self.workflow._refresh_summary_paths_for_output_root()

            if self.workflow.is_measuring and self.workflow.run_session_name is not None:
                date_dir = datetime.now().strftime("%m.%d")
                self.workflow.run_session_dir = (
                    self.workflow.output_root / "save" / date_dir / self.workflow.run_session_name
                )
                self.workflow.run_session_dir.mkdir(parents=True, exist_ok=True)
                self.workflow.context["current_paths"] = {
                    "run_session_dir": str(self.workflow.run_session_dir),
                    "run_session_name": self.workflow.run_session_name,
                }
                self.log(f"[配置同步] 运行中保存目录已变化，后续轮次将保存到：{self.workflow.run_session_dir}")

            self.log(f"[配置同步] 保存目录已更新：{self.workflow.output_root}")
            self.log(f"[配置同步] 当日汇总目录已更新：{self.workflow.summary_date_dir}")
            self.log(f"[配置同步] Excel汇总路径已更新：{self.workflow.summary_xlsx_path}")

        # 当前版本不再使用 Rigol。这里保留旧参数变化检测，但不再连接/配置 Rigol。
        if rigol_param_changed and self.workflow.laser_stage is not None:
            self.log("[配置同步] 原Rigol参数发生变化；当前版本不再使用Rigol，跳过Rigol配置")

        # 模型路径或截图区域变化时才重建角度模块。
        if angle_detector_reinit_changed and self.workflow.angle_module is not None:
            self.log("[配置同步] 角度模型路径或截图区域已变化，重新初始化角度模块")
            self.workflow.angle_module = None
            self.workflow.init_angle_module()

        # num/cw 运行中只更新对象属性，避免重复加载模型。
        elif angle_runtime_param_changed and self.workflow.angle_module is not None:
            if hasattr(self.workflow.angle_module, "num"):
                self.workflow.angle_module.num = int(new_cfg.angle_num)
            if hasattr(self.workflow.angle_module, "cw"):
                self.workflow.angle_module.cw = int(new_cfg.angle_cw)
            self.log(
                "[配置同步] 角度检测运行参数已更新，下一轮检测生效；"
                f"num={new_cfg.angle_num}, cw={new_cfg.angle_cw}；未重新加载模型"
            )

        if rule_ab_param_changed:
            if self.workflow.is_measuring and self.workflow.rule_ab_follower is not None:
                self.workflow.apply_runtime_rule_ab_params_to_follower(
                    self.workflow.rule_ab_follower,
                    reason="sync_config_from_ui_to_workflow"
                )
                self.log("[配置同步] RuleAB 参数已在运行中更新到当前 follower，不重新弹窗、不重新初始化")
            else:
                self.workflow.rule_ab_follower = None
                self.log("[配置同步] RuleAB 参数变化，下一次运行将重新初始化 A推动B 模块")

        if rule_ac_param_changed:
            self.workflow.rule_ac_controller = None
            self.log("[配置同步] RuleAC 参数变化，下一次运行将重新初始化面积控制模块")

        self.log(
            "[配置同步] 已同步当前参数："
            f"max_cycles={new_cfg.max_cycles}, "
            f"signal_on_time_ms={new_cfg.signal_on_time_ms}（来源：{signal_time_source}）, "
            f"stable_wait_ms={new_cfg.stable_wait_ms}, "
            f"factor={new_cfg.signal_time_factor}, "
            f"angle_range=[{new_cfg.angle_delta_min_deg}, {new_cfg.angle_delta_max_deg}], "
            f"angle_num={new_cfg.angle_num}, "
            f"angle_cw={new_cfg.angle_cw}, "
            f"raw_remove_above={new_cfg.raw_remove_above}, "
            f"median_filter_window={new_cfg.median_filter_window}, "
            f"x_axis_xlsx_path={new_cfg.x_axis_xlsx_path}, "
            f"save_root={new_cfg.save_root}, "
            f"rule_ab_ac_clearance=[{new_cfg.rule_ab_ac_min_clearance_px}, {new_cfg.rule_ab_ac_target_clearance_px}, {new_cfg.rule_ab_ac_max_clearance_px}], "
            f"rule_ab_ac_correction_mode={new_cfg.rule_ab_ac_correction_mode}, "
            f"rule_ab_overlap_threshold={new_cfg.rule_ab_ab_overlap_min_area_px}, "
            f"rule_ab_target=[{new_cfg.rule_ab_angle_delta_min_deg}, {new_cfg.rule_ab_angle_delta_max_deg}], "
            f"rule_ac_threshold_px={new_cfg.rule_ac_area_threshold_px}"
        )

        return self.workflow

    def ensure_workflow(self) -> MeasurementWorkflow:
        return self.sync_config_from_ui_to_workflow()

    def schedule_plot_update(self, workflow: Optional[MeasurementWorkflow] = None):
        """
        Tkinter 只能在主线程更新控件，所以绘图更新统一投递到 root.after。
        """
        wf = workflow or self.workflow
        self.root.after(0, lambda: self.update_angle_fit_peak_plot(wf))

    def update_angle_fit_peak_plot(self, workflow: Optional[MeasurementWorkflow] = None):
        """
        实时更新界面：
            1. 右侧列表仍然显示每轮的：
               - 序号；
               - 横坐标：本轮角度 / deg；
               - 纵坐标：拟合峰值。
            2. 中间图像显示区固定显示三张图：
               - 序号 - 拟合峰值；
               - xlsx横坐标 - 原始数据；
               - xlsx横坐标 - 中值滤波结果。
        """
        wf = workflow or self.workflow
        if wf is None:
            return

        points = getattr(wf, "plot_points", []) or wf.context.get("plot_points", []) or []

        # 右侧列表：仍然显示“角度-拟合峰值”的 x/y 数据。
        # 即使角度检测失败，angle_deg 为 None，也保留该轮记录；
        # 列表中角度显示为 "None"，不再跳过该行。
        list_points: List[Tuple[int, Optional[float], Optional[float]]] = []
        fit_plot_x: List[int] = []
        fit_plot_y: List[float] = []

        for p in points:
            try:
                # cycle_index 是完整测量的真实轮次号，遇到 Step1 YOLO-OBB 无检测等情况会跳过保存，
                # 因此真实 cycle_index 可能是 1,7,12...。右侧“序号”从 0 开始连续编号：
                # 初始光谱采集（cycle_index=0）显示为序号 0，后续有效记录依次为 1,2,3...。
                # 真实轮次仍保存在 point["cycle_index"]、CSV/XLSX 和日志中。
                record_index = len(list_points)

                angle = p.get("angle_deg")
                fit_peak = p.get("fit_peak")

                angle_value = None if angle is None else float(angle)
                fit_peak_value = None if fit_peak is None else float(fit_peak)

                list_points.append((record_index, angle_value, fit_peak_value))

                # 图1：有效记录序号 - 拟合峰值。
                # 拟合峰值为空时不画该点，但右侧列表仍保留该条记录。
                if fit_peak_value is not None and math.isfinite(fit_peak_value):
                    fit_plot_x.append(record_index)
                    fit_plot_y.append(float(fit_peak_value))
            except Exception:
                continue

        self.update_angle_fit_xy_list(list_points)

        if self.fig is None or self.plot_canvas is None:
            return
        if self.ax_fit_peak is None or self.ax_raw is None or self.ax_median is None:
            return

        raw_values = [float(v) for v in (wf.context.get("raw_values") or [])]
        median_values = [float(v) for v in (wf.context.get("raw_median_values") or [])]

        x_axis_values = wf.context.get("x_axis_values") or []
        if not x_axis_values:
            try:
                x_axis_values = wf.load_x_axis_values_from_xlsx()
            except Exception as e:
                wf.log(f"[横坐标xlsx] 绘图时读取失败：{e}，图像显示使用 index")
                x_axis_values = []

        raw_x = wf._make_x_values_for_plot(raw_values, x_axis_values)
        median_x = wf._make_x_values_for_plot(median_values, x_axis_values)
        x_label = "xlsx横坐标" if x_axis_values else "index"

        self.ax_fit_peak.clear()
        self.ax_raw.clear()
        self.ax_median.clear()

        # 图1：序号 - 拟合峰值
        self.ax_fit_peak.set_title("序号 - 拟合峰值")
        self.ax_fit_peak.set_xlabel("序号")
        self.ax_fit_peak.set_ylabel("拟合峰值")
        self.ax_fit_peak.grid(True)
        if fit_plot_x and fit_plot_y:
            self.ax_fit_peak.plot(fit_plot_x, fit_plot_y, marker="o")
        else:
            self.ax_fit_peak.text(
                0.5,
                0.5,
                "暂无拟合峰值数据",
                ha="center",
                va="center",
                transform=self.ax_fit_peak.transAxes,
            )

        # 图2：xlsx横坐标 - 原始数据
        self.ax_raw.set_title("xlsx横坐标 - 原始数据")
        self.ax_raw.set_xlabel(x_label)
        self.ax_raw.set_ylabel("RAW_Y")
        self.ax_raw.grid(True)
        if raw_values:
            self.ax_raw.plot(raw_x, raw_values)
        else:
            self.ax_raw.text(0.5, 0.5, "暂无原始数据", ha="center", va="center", transform=self.ax_raw.transAxes)

        # 图3：xlsx横坐标 - 中值滤波结果
        self.ax_median.set_title("xlsx横坐标 - 中值滤波结果")
        self.ax_median.set_xlabel(x_label)
        self.ax_median.set_ylabel("median filtered")
        self.ax_median.grid(True)
        if median_values:
            self.ax_median.plot(median_x, median_values)
        else:
            self.ax_median.text(0.5, 0.5, "暂无中值滤波结果", ha="center", va="center", transform=self.ax_median.transAxes)

        self.fig.tight_layout()
        self.plot_canvas.draw_idle()

        self.plot_status_var.set(
            f"图像显示：序号-拟合峰值点数={len(fit_plot_y)}；"
            f"原始点数={len(raw_values)}；"
            f"中值滤波点数={len(median_values)}；"
            f"光谱横坐标={x_label}"
        )

    @staticmethod
    def _format_optional_float(value: Optional[float], digits: int = 6) -> str:
        """右侧列表用：None 保留为字符串 None，数字按指定小数位显示。"""
        if value is None:
            return "None"
        try:
            return f"{float(value):.{digits}f}"
        except Exception:
            return str(value)

    def update_angle_fit_xy_list(self, valid_points: List[Tuple[int, Optional[float], Optional[float]]]):
        """
        把原“角度-拟合峰值图”的 x/y 数据显示到右侧列表。

        如果某一轮没有检测到角度，仍然保留该轮记录：
            angle 显示为 None；
            fit_peak 如果有效则正常显示，否则也显示 None。
        """
        tree = self.angle_fit_tree
        if tree is None:
            return

        try:
            for item in tree.get_children():
                tree.delete(item)

            for cycle_index, angle, fit_peak in valid_points:
                tree.insert(
                    "",
                    tk.END,
                    values=(
                        cycle_index,
                        self._format_optional_float(angle),
                        self._format_optional_float(fit_peak),
                    ),
                )

            # ttk.Treeview 不能真正滚动到“最后一条之后的空白区域”。
            # 为了让最后一条真实数据可以滚到列表中间，额外追加若干空白占位行。
            # 这些空白行不计入状态栏点数，也不会写入保存文件。
            try:
                visible_rows = int(tree.cget("height"))
            except Exception:
                visible_rows = 24
            padding_rows = max(8, visible_rows // 2)
            for _ in range(padding_rows):
                tree.insert("", tk.END, values=("", "", ""), tags=("padding",))

            self.angle_fit_list_status_var.set(
                f"角度-拟合峰值列表：{len(valid_points)} 个有效记录；"
                "序号=有效记录序号（真实cycle可能跳过），横坐标=本轮角度/deg，纵坐标=拟合峰值"
            )
        except Exception as e:
            self.angle_fit_list_status_var.set(f"角度-拟合峰值列表更新失败：{e}")


    def refresh_result_labels(self, workflow: Optional[MeasurementWorkflow] = None):
        wf = workflow or self.workflow
        if wf is None:
            return

        ctx = wf.context

        self.set_var(self.current_cycle_var, f"cycle：{ctx.get('cycle_index')}")
        self.set_var(self.current_angle_before_var, f"本轮角度：{ctx.get('angle_before')}")
        self.set_var(self.current_angle_after_var, "angle_after：单角度模式未使用")
        self.set_var(
            self.current_angle_delta_var,
            f"跨轮angle_delta：{ctx.get('angle_delta')}；同轮before/after差：{ctx.get('angle_before_after_delta')}"
        )
        self.set_var(
            self.current_signal_time_var,
            f"下一轮信号ON时间/ms：{wf.cfg.signal_on_time_ms:.3f}"
        )
        self.set_var(
            self.current_raw_peak_var,
            f"raw_peak：{ctx.get('raw_peak')}；原始峰值：{ctx.get('raw_original_peak')}"
        )
        self.set_var(self.current_fit_peak_var, f"fit_peak：{ctx.get('fit_peak')}")

        # 运行中信号ON时间输入框的显示/执行策略：
        # - 用户没有手动改输入框：把 Step 11 计算出的下一轮时间写回输入框，
        #   下一轮会按这个计算值运行，例如 62.500 ms；
        # - 用户已经手动改输入框：不覆盖用户输入，下一轮优先按用户改好的值运行。
        if wf.is_measuring and not self.signal_on_time_user_modified:
            self.set_signal_on_time_var_programmatically(
                round(float(wf.cfg.signal_on_time_ms), 3)
            )

        current_paths = ctx.get("current_paths")
        if isinstance(current_paths, dict):
            xlsx_path = getattr(wf, "summary_xlsx_path", None)
            self.set_var(
                self.current_save_path_var,
                f"保存路径：{current_paths.get('run_session_dir')}；当前CSV：{current_paths.get('spectrum_filename')}；Excel汇总：{xlsx_path}"
            )

        self.set_var(
            self.delta_status_var,
            f"Δw：delta={ctx.get('delta_w')} valid={ctx.get('valid_change')}",
        )

        light_state = "ON" if ctx.get("light_on") else "OFF"
        laser_state = "ON" if ctx.get("laser_on") else "OFF"

        if wf.light is not None:
            self.set_var(self.light_status_var, f"照明光状态：已连接 / {light_state}")
        else:
            self.set_var(self.light_status_var, "照明光状态：未连接")

        if wf.laser_stage is not None:
            self.set_var(
                self.rigol_status_var,
                f"激光控制器状态：已连接，激光 {laser_state}，axis={wf.cfg.laser_axis}"
            )
        else:
            self.set_var(self.rigol_status_var, "激光控制器状态：未连接")

        if wf.angle_module is not None:
            self.set_var(
                self.angle_status_var,
                f"角度检测状态：已初始化，模型已加载一次；num={wf.cfg.angle_num}，cw={wf.cfg.angle_cw}"
            )
        else:
            self.set_var(self.angle_status_var, "角度检测状态：未初始化")

        if wf.context.get("labview_ready", False):
            self.set_var(self.tcp_status_var, "TCP状态：READY，可采集")
        elif wf.context.get("tcp_started", False):
            self.set_var(self.tcp_status_var, "TCP状态：Server已启动，等待READY")
        else:
            self.set_var(self.tcp_status_var, "TCP状态：未启动")

        self.schedule_plot_update(wf)

    # --------------------------------------------------------
    # 初始化 / 运行 / 停止
    # --------------------------------------------------------

    def init_all_devices_thread(self):
        self.run_in_thread(self.init_all_devices)

    def init_all_devices(self):
        try:
            self.set_var(self.flow_status_var, "流程状态：正在初始化实验设备...")

            # 重要：通过 sync_config_from_ui_to_workflow() 同步 GUI 参数。
            # 这样 num/cw、模型路径、截图区域等参数在点击按钮时立即生效；
            # 同时不会丢失已经建立好的 TCP 连接和 LabVIEW READY 状态。
            self.workflow = self.sync_config_from_ui_to_workflow()

            self.workflow.connect_measurement_devices()

            self.set_var(self.flow_status_var, "流程状态：实验设备初始化完成")
            self.set_var(self.light_status_var, "照明光状态：已连接")
            self.set_var(self.rigol_status_var, "激光控制器状态：已连接，激光轴速度/加速度已设置")
            self.set_var(self.angle_status_var, "角度检测状态：已初始化，模型已加载一次")

            if self.workflow.context.get("labview_ready", False):
                self.set_var(self.tcp_status_var, "TCP状态：READY，可采集")
            elif self.workflow.context.get("tcp_started", False):
                self.set_var(self.tcp_status_var, "TCP状态：Server已启动，等待READY")
            else:
                self.set_var(self.tcp_status_var, "TCP状态：未启动")

            self.refresh_result_labels()

        except Exception as e:
            self.set_var(self.flow_status_var, "流程状态：初始化失败")
            self.log(f"[GUI错误] 初始化失败：{e}")
            self.log(traceback.format_exc())
            error_msg = str(e)
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("初始化失败", msg))


    # --------------------------------------------------------
    # 完整测量前统一标定 GUI
    # --------------------------------------------------------

    def _ask_string_on_main_thread(self, title: str, prompt: str, initialvalue: str = "") -> Optional[str]:
        """线程安全地弹出 simpledialog.askstring。"""
        if threading.get_ident() == getattr(self, "_tk_thread_ident", None):
            return simpledialog.askstring(title, prompt, initialvalue=initialvalue, parent=self.root)

        box: Dict[str, Any] = {"value": None, "done": False}
        ev = threading.Event()

        def _ask():
            try:
                box["value"] = simpledialog.askstring(title, prompt, initialvalue=initialvalue, parent=self.root)
            finally:
                box["done"] = True
                ev.set()

        self.root.after(0, _ask)
        ev.wait()
        return box.get("value")

    def _get_calibration_path_from_gui(self) -> Path:
        cfg = self.build_config_from_ui()
        p = str(getattr(cfg, "calibration_path", "") or "").strip()
        if p:
            return Path(p).expanduser().resolve()
        return (Path(cfg.save_root) / "calibration" / "current_calibration.json").resolve()

    def _load_calibration_file_raw(self) -> CalibrationState:
        path = self._get_calibration_path_from_gui()
        if not path.exists():
            raise RuntimeError(f"未找到完整测量标定文件：{path}")
        with path.open("r", encoding="utf-8") as f:
            return CalibrationState.from_dict(json.load(f))

    def _save_calibration_file_raw(self, state: CalibrationState) -> Path:
        cfg = self.build_config_from_ui()
        wf = self.workflow or MeasurementWorkflow(cfg, on_log=self.log, on_update=self.refresh_result_labels)
        wf.cfg = cfg
        path = wf.save_calibration_state(state)
        self.workflow = wf
        self.set_var(self.calibration_path_var, f"标定文件：{path}")
        missing = state.missing_items()
        if missing:
            self.set_var(self.calibration_status_var, "完整测量标定：未完成，缺少 " + "；".join(missing))
        else:
            self.set_var(self.calibration_status_var, "完整测量标定：已完成，可运行完整循环测量")
        return path

    def _current_or_new_calibration_state(self) -> CalibrationState:
        try:
            state = self._load_calibration_file_raw()
        except Exception:
            state = CalibrationState()
        cfg = self.build_config_from_ui()
        state.capture_area = [int(v) for v in cfg.capture_area]
        state.angle_model_path = str(cfg.angle_model_path)
        state.angle_num = int(cfg.angle_num)
        state.angle_cw = int(cfg.angle_cw)
        # 关键修复：
        # 不再用 RuleAB/RuleAC 单独测试面板中残留的“C标定文件夹”覆盖完整标定包里的全局 C。
        # 只有当标定包本身还没有 static C 时，才允许从 GUI 的 C 文件夹补一次。
        if not state.static_c_map_dir and not state.static_c_mask_path:
            state.static_c_map_dir = str(cfg.rule_ab_static_c_map_dir or "")
        if state.static_c_map_dir:
            cdir = Path(state.static_c_map_dir)
            if (cdir / "static_c_mask.png").exists():
                state.static_c_mask_path = str((cdir / "static_c_mask.png").resolve())
                if (cdir / "static_c_edge_route.json").exists():
                    state.static_c_edge_route_path = str((cdir / "static_c_edge_route.json").resolve())
            elif (cdir / "static_c_reference" / "static_c_mask.png").exists():
                state.static_c_mask_path = str((cdir / "static_c_reference" / "static_c_mask.png").resolve())
                if (cdir / "static_c_reference" / "static_c_edge_route.json").exists():
                    state.static_c_edge_route_path = str((cdir / "static_c_reference" / "static_c_edge_route.json").resolve())
        state.step9_target_x_px = float(cfg.rule_ac_target_x_px)
        state.step9_target_y_px = float(cfg.rule_ac_target_y_px)
        state.step9_color_mode = str(cfg.rule_ac_color_mode)
        state.step9_color_h = int(cfg.rule_ac_color_h)
        state.step9_color_s = int(cfg.rule_ac_color_s)
        state.step9_color_v = int(cfg.rule_ac_color_v)
        state.step9_color_h_tol = int(cfg.rule_ac_color_h_tol)
        state.step9_color_s_tol = int(cfg.rule_ac_color_s_tol)
        state.step9_color_v_tol = int(cfg.rule_ac_color_v_tol)
        state.step9_color_min_area_px = int(cfg.rule_ac_color_min_area_px)
        state.step9_color_morph_kernel = int(cfg.rule_ac_color_morph_kernel)
        state.step9_center_tolerance_px = float(cfg.rule_ac_center_tolerance_px)

        # B 点双向兜底，避免“不点击 1 标定角度B/边”时 RuleAB-B/角度B 缺失导致 ABC 初始化不稳定。
        if not state.rule_ab_b_positive_points and state.angle_b_positive_points:
            state.rule_ab_b_positive_points = [[float(x), float(y)] for x, y in state.angle_b_positive_points]
            state.rule_ab_b_negative_points = [[float(x), float(y)] for x, y in state.angle_b_negative_points]
        if not state.angle_b_positive_points and state.rule_ab_b_positive_points:
            state.angle_b_positive_points = [[float(x), float(y)] for x, y in state.rule_ab_b_positive_points]
            state.angle_b_negative_points = [[float(x), float(y)] for x, y in state.rule_ab_b_negative_points]
        return state

    def _select_object_points_interactively(
        self,
        image_rgb: np.ndarray,
        object_name: str,
        window_name: str,
        scale: float = 0.85,
        existing_roi_polygons: Any = None,
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], List[List[List[float]]]]:
        """
        通用点选窗口：左键正点，右键负点，Enter/N完成，R重选。
        点选完成后按 E 进入 ROI 模式，绘制需要排除的干扰区域。
        返回截图区域原始坐标和 ROI 多边形。
        """
        from logic.roi_exclusion import select_roi_polygons_interactively

        if scale <= 0:
            scale = 1.0
        image_bgr = cv2.cvtColor(np.asarray(image_rgb), cv2.COLOR_RGB2BGR)
        h, w = image_bgr.shape[:2]
        show_w = max(1, int(w * scale))
        show_h = max(1, int(h * scale))
        display = cv2.resize(image_bgr, (show_w, show_h), interpolation=cv2.INTER_AREA)
        pos: List[Tuple[int, int]] = []
        neg: List[Tuple[int, int]] = []

        def redraw() -> np.ndarray:
            canvas = display.copy()
            lines = [
                f"{object_name} point calibration",
                "Left click: positive point | Right click: negative point",
                "Enter/N: finish point selection | R: reset | ESC/Q: cancel",
                "After points, press E to draw ROI polygons for excluded regions",
                f"positive={len(pos)}, negative={len(neg)}",
            ]
            for i, text in enumerate(lines):
                cv2.putText(canvas, text, (18, 28 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2, cv2.LINE_AA)
            for j, (x, y) in enumerate(pos):
                cv2.circle(canvas, (x, y), 6, (255, 255, 0), -1)
                cv2.putText(canvas, f"+{j+1}", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)
            for j, (x, y) in enumerate(neg):
                cv2.circle(canvas, (x, y), 7, (0, 0, 255), 2)
                cv2.line(canvas, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 255), 2)
                cv2.line(canvas, (x - 5, y + 5), (x + 5, y - 5), (0, 0, 255), 2)
                cv2.putText(canvas, f"-{j+1}", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
            return canvas

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                pos.append((int(x), int(y)))
            elif event == cv2.EVENT_RBUTTONDOWN:
                neg.append((int(x), int(y)))

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, show_w, show_h)
        cv2.setMouseCallback(window_name, on_mouse)
        try:
            while True:
                cv2.imshow(window_name, redraw())
                key = cv2.waitKey(30) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    raise RuntimeError(f"用户取消了 {object_name} 标定。")
                if key in (ord("r"), ord("R")):
                    pos.clear(); neg.clear()
                if key in (13, 10, ord("n"), ord("N")):
                    if len(pos) <= 0:
                        self.log(f"[{object_name}] 至少需要 1 个正点。")
                        continue
                    break
        finally:
            try:
                cv2.destroyWindow(window_name)
            except Exception:
                pass

        to_original = lambda pts: [(float(x) / scale, float(y) / scale) for x, y in pts]
        original_pos = to_original(pos)
        original_neg = to_original(neg)

        # 点选完成后，进入 ROI 绘制模式（用户可按 E 进入/退出，或直接 Enter 跳过）。
        roi_polygons = select_roi_polygons_interactively(
            image_bgr=image_bgr,
            existing_polygons=existing_roi_polygons,
            window_name=f"Draw ROI for {object_name} (E=toggle, ESC=cancel)",
            scale=scale,
        )
        return original_pos, original_neg, roi_polygons

    @staticmethod
    def _normalize_points_xy_for_gui(points: Any) -> List[Tuple[float, float]]:
        """把 GUI 标定点统一转成 [(x, y), ...]，兼容 tuple/list/dict。"""
        out: List[Tuple[float, float]] = []
        if points is None:
            return out
        for p in points:
            try:
                if isinstance(p, dict):
                    x = p.get("x", p.get("X", None))
                    y = p.get("y", p.get("Y", None))
                else:
                    x, y = p[0], p[1]
                out.append((float(x), float(y)))
            except Exception:
                continue
        return out

    def _extract_b_positive_hsv_seeds_from_points(
        self,
        image_rgb: np.ndarray,
        positive_points: Any,
        label: str = "B",
    ) -> List[List[float]]:
        """
        从 B 正点附近小窗口提取 HSV median 颜色种子。

        这个函数需要放在 MeasurementWorkflowGUI 中，因为完整测量标定按钮
        calibrate_angle_b()/calibrate_rule_ab_b() 是 GUI 方法，保存标定文件前
        会直接调用 self._extract_b_positive_hsv_seeds_from_points(...)。
        MeasurementWorkflow 中虽然也有同名函数，但 GUI 对象不能自动访问。
        """
        img = np.asarray(image_rgb).astype(np.uint8)
        if img.ndim != 3 or img.shape[2] < 3:
            return []

        try:
            cfg = self.build_config_from_ui()
            radius_px = int(getattr(cfg, "rule_ab_b_positive_color_seed_radius_px", 5))
        except Exception:
            radius_px = 5
        r = max(1, radius_px)

        hsv = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
        h_img, w_img = hsv.shape[:2]
        seeds: List[List[float]] = []
        pts = self._normalize_points_xy_for_gui(positive_points)

        for i, (x, y) in enumerate(pts):
            cx, cy = int(round(float(x))), int(round(float(y)))
            x1, x2 = max(0, cx - r), min(w_img, cx + r + 1)
            y1, y2 = max(0, cy - r), min(h_img, cy + r + 1)
            if x2 <= x1 or y2 <= y1:
                self.log(f"[BColorSeed] {label} positive point {i} 窗口越界/为空，跳过。")
                continue

            patch = hsv[y1:y2, x1:x2].reshape(-1, 3)
            if patch.shape[0] < 3:
                self.log(f"[BColorSeed] {label} positive point {i} 窗口像素太少 n={patch.shape[0]}，跳过。")
                continue

            med = np.median(patch.astype(np.float32), axis=0)
            seed = [float(med[0]), float(med[1]), float(med[2])]
            seeds.append(seed)
            self.log(
                f"[BColorSeed] {label} positive point {i} "
                f"hsv_seed=({seed[0]:.1f},{seed[1]:.1f},{seed[2]:.1f})"
            )

        self.log(f"[BColorSeed] saved b_positive_hsv_seeds count={len(seeds)}")
        return seeds

    def calibrate_angle_b_thread(self):
        self.log("[YOLO-OBB角度] 当前版本已删除 1 标定角度B/边；Step1/Step7 不再需要 Bmask 标记。")

    def calibrate_angle_b(self):
        self.log("[YOLO-OBB角度] 当前版本已删除 1 标定角度B/边；Step1/Step7 不再需要 Bmask 标记。")

    def calibrate_rule_ab_a_thread(self):
        self.run_in_thread(self.calibrate_rule_ab_a)

    def calibrate_rule_ab_a(self):
        """标定 A推动B 中 A mask 的正点/负点。"""
        try:
            self._prepare_for_manual_calibration(reason="calibrate_rule_ab_a")
            cfg = self.build_config_from_ui()
            out_dir = Path(cfg.save_root) / "calibration" / "rule_ab_a"
            out_dir.mkdir(parents=True, exist_ok=True)
            image_rgb = self._capture_current_rule_ab_frame(output_dir=out_dir)
            state = self._current_or_new_calibration_state()
            pos, neg, roi_polygons = self._select_object_points_interactively(
                image_rgb,
                "RuleAB-A",
                "Calibration RuleAB A: left positive, right negative",
                existing_roi_polygons=state.exclude_roi_polygons,
            )

            state.rule_ab_a_positive_points = [[float(x), float(y)] for x, y in pos]
            state.rule_ab_a_negative_points = [[float(x), float(y)] for x, y in neg]
            state.exclude_roi_polygons = roi_polygons
            self._save_calibration_file_raw(state)
            self.log(f"[完整测量标定] RuleAB-A 标定完成：A+={len(pos)}, A-={len(neg)}, ROIs={len(roi_polygons)}")
        except Exception as e:
            self.log(f"[完整测量标定] RuleAB-A 标定失败：{e}")
            self.log(traceback.format_exc())
            error_msg = str(e)
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("RuleAB-A标定失败", msg))

    def calibrate_rule_ab_b_thread(self):
        self.log("[YOLO-OBB角度] 当前版本已删除 2b 标定RuleAB-B；Step1/Step7 角度不再依赖 Bmask。")

    def calibrate_rule_ab_b(self):
        self.log("[YOLO-OBB角度] 当前版本已删除 2b 标定RuleAB-B；Step1/Step7 角度不再依赖 Bmask。")

    def calibrate_global_c_thread(self):
        self.run_in_thread(self.calibrate_global_c)

    def calibrate_global_c(self):
        """标定整体测量前的 C mask 正点/负点，并保存 static C map。"""
        try:
            self._prepare_for_manual_calibration(reason="calibrate_global_c")
            self.set_var(self.rule_module_status_var, "规则模块状态：正在完整测量前标定C")
            cfg_gui = self.build_config_from_ui()
            save_root = Path(str(getattr(cfg_gui, "save_root", "measurement_output")))
            map_name = str(getattr(cfg_gui, "rule_ab_static_c_map_name", "default_static_c_map")).strip() or "default_static_c_map"
            safe_map_name = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in map_name)
            save_dir = save_root / "rule_ab_follow_c_only" / "_static_c_map_library" / safe_map_name

            image_rgb = self._capture_current_rule_ab_frame(output_dir=save_dir)
            state = self._current_or_new_calibration_state()
            pos, neg, roi_polygons = self._select_c_points_interactively(
                image_rgb=image_rgb,
                window_name="Calibration Global C: left positive, right negative",
                scale=float(getattr(cfg_gui, "rule_ab_confirm_window_scale", 0.85)),
                existing_roi_polygons=state.exclude_roi_polygons,
            )
            c_mask = self._predict_c_mask_by_sam2_points(
                image_rgb=image_rgb,
                positive_points=pos,
                negative_points=neg,
                roi_polygons=roi_polygons,
            )
            saved_dir = self._save_static_c_map_result(
                image_rgb=image_rgb,
                c_mask=c_mask,
                positive_points=pos,
                negative_points=neg,
                save_dir=save_dir,
                roi_polygons=roi_polygons,
            )

            self.rule_ab_static_c_map_dir_var.set(str(saved_dir))
            self.rule_ab_load_static_c_map_var.set(True)
            self.rule_ab_force_reselect_c_var.set(False)

            state.global_c_positive_points = [[float(x), float(y)] for x, y in pos]
            state.global_c_negative_points = [[float(x), float(y)] for x, y in neg]
            state.static_c_map_dir = str(saved_dir)
            state.exclude_roi_polygons = roi_polygons
            c_mask_path = Path(saved_dir) / "static_c_mask.png"
            if c_mask_path.exists():
                state.static_c_mask_path = str(c_mask_path.resolve())
            self._save_calibration_file_raw(state)
            self.log(f"[完整测量标定] 全局C标定完成并保存 static C：{saved_dir}，ROIs={len(roi_polygons)}")
        except Exception as e:
            self.log(f"[完整测量标定] 全局C标定失败：{e}")
            self.log(traceback.format_exc())
            error_msg = str(e)
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("全局C标定失败", msg))

    def calibrate_step9_target_thread(self):
        self.run_in_thread(self.calibrate_step9_target)

    def calibrate_step9_target(self):
        """只标定 Step9 目标点，不重新选择颜色 HSV。"""
        try:
            self._prepare_for_manual_calibration(reason="calibrate_step9_target")

            # 强制清空旧目标，确保本按钮一定重新点选目标。
            self.set_var_blocking(self.rule_ac_target_x_var, -1.0)
            self.set_var_blocking(self.rule_ac_target_y_var, -1.0)

            selected_target = self.select_step9_target()
            if selected_target is None:
                raise RuntimeError("Step9 目标点未完成选择，已取消 Step9 目标标定。")
            target_x, target_y = float(selected_target[0]), float(selected_target[1])

            self.set_var_blocking(self.rule_ac_target_x_var, round(target_x, 3))
            self.set_var_blocking(self.rule_ac_target_y_var, round(target_y, 3))

            wf = self.sync_config_from_ui_to_workflow()
            wf.cfg.rule_ac_target_x_px = target_x
            wf.cfg.rule_ac_target_y_px = target_y

            state = self._current_or_new_calibration_state()
            state.step9_target_x_px = float(target_x)
            state.step9_target_y_px = float(target_y)
            state.step9_center_tolerance_px = float(wf.cfg.rule_ac_center_tolerance_px)
            self._save_calibration_file_raw(state)

            self.log(
                f"[完整测量标定] Step9目标标定完成："
                f"target=({state.step9_target_x_px:.1f},{state.step9_target_y_px:.1f})；"
                "未修改 Step9 颜色 HSV。"
            )
        except Exception as e:
            self.log(f"[完整测量标定] Step9目标标定失败：{e}")
            self.log(traceback.format_exc())
            error_msg = str(e)
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("Step9目标标定失败", msg))

    def calibrate_step9_color_thread(self):
        self.run_in_thread(self.calibrate_step9_color)

    def calibrate_step9_color(self):
        """只标定 Step9 颜色 HSV，不重新选择目标点。"""
        try:
            self._prepare_for_manual_calibration(reason="calibrate_step9_color")

            wf = self.sync_config_from_ui_to_workflow()
            target_x = float(getattr(wf.cfg, "rule_ac_target_x_px", -1.0))
            target_y = float(getattr(wf.cfg, "rule_ac_target_y_px", -1.0))
            if target_x < 0 or target_y < 0:
                raise RuntimeError("请先点击“4a 标定Step9目标”，或在 Step9目标x/y 输入框中填写有效目标点。")

            wf.begin_step9_run_session()
            color_result = wf.prepare_step9_color_region_detection()
            if not isinstance(color_result, dict):
                raise RuntimeError("Step9 颜色检测返回结果无效。")
            hsv = color_result.get("hsv", None)
            mode = str(color_result.get("mode", getattr(wf.cfg, "rule_ac_color_mode", "include")) or "include")
            if hsv is None or len(hsv) < 3:
                raise RuntimeError("Step9 颜色检测未返回有效 HSV。")
            h, s_val, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

            self.set_var_blocking(self.rule_ac_color_mode_var, mode)
            self.set_var_blocking(self.rule_ac_color_h_var, h)
            self.set_var_blocking(self.rule_ac_color_s_var, s_val)
            self.set_var_blocking(self.rule_ac_color_v_var, v)

            wf.cfg.rule_ac_color_mode = mode
            wf.cfg.rule_ac_color_h = h
            wf.cfg.rule_ac_color_s = s_val
            wf.cfg.rule_ac_color_v = v
            wf.step9_color_mode = mode
            wf.step9_color_hsv = (h, s_val, v)

            state = self._current_or_new_calibration_state()
            # 颜色按钮不修改目标；只把当前有效目标同步进标定包，避免保存时目标丢失。
            state.step9_target_x_px = float(target_x)
            state.step9_target_y_px = float(target_y)
            state.step9_color_mode = mode
            state.step9_color_h = h
            state.step9_color_s = s_val
            state.step9_color_v = v
            state.step9_color_h_tol = int(wf.cfg.rule_ac_color_h_tol)
            state.step9_color_s_tol = int(wf.cfg.rule_ac_color_s_tol)
            state.step9_color_v_tol = int(wf.cfg.rule_ac_color_v_tol)
            state.step9_color_min_area_px = int(wf.cfg.rule_ac_color_min_area_px)
            state.step9_color_morph_kernel = int(wf.cfg.rule_ac_color_morph_kernel)
            state.step9_center_tolerance_px = float(wf.cfg.rule_ac_center_tolerance_px)
            self._save_calibration_file_raw(state)

            if self.workflow is not None:
                self.workflow.cfg.rule_ac_color_mode = mode
                self.workflow.cfg.rule_ac_color_h = h
                self.workflow.cfg.rule_ac_color_s = s_val
                self.workflow.cfg.rule_ac_color_v = v
                self.workflow.step9_color_mode = mode
                self.workflow.step9_color_hsv = (h, s_val, v)

            self.log(
                f"[完整测量标定] Step9颜色标定完成："
                f"mode={mode}, HSV=({h},{s_val},{v})；"
                f"目标保持为 ({target_x:.1f},{target_y:.1f})；"
                f"area={color_result.get('area_px')}, center={color_result.get('center')}"
            )
        except Exception as e:
            self.log(f"[完整测量标定] Step9颜色标定失败：{e}")
            self.log(traceback.format_exc())
            error_msg = str(e)
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("Step9颜色标定失败", msg))

    def calibrate_step9_target_color_thread(self):
        self.run_in_thread(self.calibrate_step9_target_color)

    def calibrate_step9_target_color(self):
        """标定 Step9 目标点和颜色检测 HSV。

        关键要求：
            1. 点击“4 标定Step9目标/颜色”时，必须重新标定目标点，
               不能因为 GUI 或旧标定文件里已有 target_x/y 就复用旧目标；
            2. prepare_step9_color_region_detection() 成功返回后，必须立刻把本次点选得到的
               HSV/mode 写回 GUI 变量和 workflow.cfg；
            3. 保存完整标定文件时，Step9 HSV 必须使用 prepare_step9_color_region_detection()
               本次返回的 HSV，而不是再次依赖 sync_config_from_ui_to_workflow() 从 GUI 旧值读取。

        这样可以避免：颜色点选时日志显示 HSV=(64,155,115)，但保存完整标定时又被
        GUI 旧值/默认值覆盖成 HSV=(0,255,255)。
        """
        try:
            self._prepare_for_manual_calibration(reason="calibrate_step9_target_color")

            # 强制清空旧目标，避免旧完整标定文件加载出的 target_x/y 被继续复用。
            self.set_var_blocking(self.rule_ac_target_x_var, -1.0)
            self.set_var_blocking(self.rule_ac_target_y_var, -1.0)

            # 每次点击按钮都必须重新选 Step9 目标点。
            selected_target = self.select_step9_target()
            if selected_target is None:
                raise RuntimeError("Step9 目标点未完成选择，已取消 Step9 目标/颜色标定。")
            target_x, target_y = float(selected_target[0]), float(selected_target[1])

            # select_step9_target 内部会写 GUI 变量；这里再同步写一次并等待，
            # 保证后续 build_config_from_ui()/sync_config_from_ui_to_workflow() 读到的是本次新目标。
            self.set_var_blocking(self.rule_ac_target_x_var, round(target_x, 3))
            self.set_var_blocking(self.rule_ac_target_y_var, round(target_y, 3))

            # 先同步一次，让 workflow 使用新 target 创建 Step9 运行文件夹/截图。
            wf = self.sync_config_from_ui_to_workflow()
            wf.cfg.rule_ac_target_x_px = target_x
            wf.cfg.rule_ac_target_y_px = target_y
            wf.begin_step9_run_session()

            # Step9 使用颜色检测区域中心对齐，因此目标点之后还需要选择一次颜色 HSV。
            # 关键：必须接住本次函数返回的 HSV/mode，不能只依赖 workflow 内部状态，
            # 更不能在没有写回 GUI 的情况下再次 sync_config_from_ui_to_workflow()。
            color_result = wf.prepare_step9_color_region_detection()
            if not isinstance(color_result, dict):
                raise RuntimeError("Step9 颜色检测返回结果无效。")
            hsv = color_result.get("hsv", None)
            mode = str(color_result.get("mode", getattr(wf.cfg, "rule_ac_color_mode", "include")) or "include")
            if hsv is None or len(hsv) < 3:
                raise RuntimeError("Step9 颜色检测未返回有效 HSV。")
            h, s_val, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

            # 立刻把本次点选得到的 HSV/mode 写回 GUI 变量，且使用 blocking，
            # 避免 root.after 异步写入尚未执行时，后续保存又读到旧 HSV。
            self.set_var_blocking(self.rule_ac_color_mode_var, mode)
            self.set_var_blocking(self.rule_ac_color_h_var, h)
            self.set_var_blocking(self.rule_ac_color_s_var, s_val)
            self.set_var_blocking(self.rule_ac_color_v_var, v)

            # 同步写回 workflow.cfg 和 workflow 运行缓存。
            wf.cfg.rule_ac_target_x_px = target_x
            wf.cfg.rule_ac_target_y_px = target_y
            wf.cfg.rule_ac_color_mode = mode
            wf.cfg.rule_ac_color_h = h
            wf.cfg.rule_ac_color_s = s_val
            wf.cfg.rule_ac_color_v = v
            wf.step9_color_mode = mode
            wf.step9_color_hsv = (h, s_val, v)

            self.log(
                "[完整测量标定] Step9颜色已写回 GUI/workflow.cfg："
                f"mode={mode}, HSV=({h},{s_val},{v}), "
                f"target=({target_x:.3f},{target_y:.3f}), "
                f"area={color_result.get('area_px')}, center={color_result.get('center')}"
            )

            # 保存标定文件时，明确使用本次 color_result 的 HSV/mode，
            # 不再再次依赖 sync_config_from_ui_to_workflow() 读 GUI 旧值。
            state = self._current_or_new_calibration_state()
            state.step9_target_x_px = float(target_x)
            state.step9_target_y_px = float(target_y)
            state.step9_color_mode = mode
            state.step9_color_h = h
            state.step9_color_s = s_val
            state.step9_color_v = v
            state.step9_color_h_tol = int(wf.cfg.rule_ac_color_h_tol)
            state.step9_color_s_tol = int(wf.cfg.rule_ac_color_s_tol)
            state.step9_color_v_tol = int(wf.cfg.rule_ac_color_v_tol)
            state.step9_color_min_area_px = int(wf.cfg.rule_ac_color_min_area_px)
            state.step9_color_morph_kernel = int(wf.cfg.rule_ac_color_morph_kernel)
            state.step9_center_tolerance_px = float(wf.cfg.rule_ac_center_tolerance_px)

            self._save_calibration_file_raw(state)

            # _save_calibration_file_raw() 内部会用 build_config_from_ui() 重建 cfg，
            # 因为上面已经 blocking 写回 GUI，所以这里再强制确认 workflow.cfg 仍是本次 HSV。
            if self.workflow is not None:
                self.workflow.cfg.rule_ac_target_x_px = target_x
                self.workflow.cfg.rule_ac_target_y_px = target_y
                self.workflow.cfg.rule_ac_color_mode = mode
                self.workflow.cfg.rule_ac_color_h = h
                self.workflow.cfg.rule_ac_color_s = s_val
                self.workflow.cfg.rule_ac_color_v = v
                self.workflow.step9_color_mode = mode
                self.workflow.step9_color_hsv = (h, s_val, v)

            self.log(
                f"[完整测量标定] Step9目标标定完成："
                f"target=({state.step9_target_x_px:.1f},{state.step9_target_y_px:.1f})；"
                f"颜色={state.step9_color_mode}, HSV=({state.step9_color_h},{state.step9_color_s},{state.step9_color_v})"
            )
        except Exception as e:
            self.log(f"[完整测量标定] Step9目标/颜色标定失败：{e}")
            self.log(traceback.format_exc())
            error_msg = str(e)
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("Step9标定失败", msg))


    def save_and_check_calibration_thread(self):
        self.run_in_thread(self.save_and_check_calibration)

    def save_and_check_calibration(self):
        """保存当前 GUI 中已有的完整标定参数，并检查是否满足完整测量启动条件。"""
        try:
            state = self._current_or_new_calibration_state()
            path = self._save_calibration_file_raw(state)
            missing = state.missing_items()
            if missing:
                msg = "完整测量标定尚未完成，缺少：\n" + "\n".join(f"- {x}" for x in missing)
                self.log("[完整测量标定] " + msg.replace("\n", "；"))
                self.root.after(0, lambda: messagebox.showwarning("标定未完成", msg))
            else:
                self.log(f"[完整测量标定] 检查通过，可运行完整循环测量：{path}")
                self.root.after(0, lambda: messagebox.showinfo("标定完成", f"完整测量标定检查通过。\n{path}"))
        except Exception as e:
            self.log(f"[完整测量标定] 保存/检查失败：{e}")
            self.log(traceback.format_exc())
            error_msg = str(e)
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("保存/检查标定失败", msg))

    def load_calibration_to_gui_thread(self):
        self.run_in_thread(self.load_calibration_to_gui)

    def load_calibration_to_gui(self):
        """加载完整标定 JSON，并同步到 GUI 输入框。"""
        try:
            state = self._load_calibration_file_raw()

            # 同步到当前 workflow，确保“加载完整标定”不只是刷新 GUI 输入框，
            # 也真正进入后续 RuleAB/SAM2/角度/Step9 的运行上下文。
            wf = self.ensure_workflow()
            wf.apply_calibration_state(state)
            wf.prepare_runtime_from_full_calibration(state)

            self.set_var(self.num_var, int(state.angle_num))
            self.set_var(self.cw_var, int(state.angle_cw))
            if state.static_c_map_dir:
                self.set_var(self.rule_ab_static_c_map_dir_var, str(state.static_c_map_dir))
                self.set_var(self.rule_ab_load_static_c_map_var, True)
                self.set_var(self.rule_ab_force_reselect_c_var, False)

            # 每次打开程序/加载完整标定后，默认启用真实运动。
            self.set_var(self.rule_ab_enable_stage_var, True)
            self.set_var(self.rule_ac_enable_stage_var, True)
            self.set_var(self.rule_ac_target_x_var, float(state.step9_target_x_px))
            self.set_var(self.rule_ac_target_y_var, float(state.step9_target_y_px))
            self.set_var(self.rule_ac_color_mode_var, str(state.step9_color_mode or "include"))
            self.set_var(self.rule_ac_color_h_var, int(state.step9_color_h))
            self.set_var(self.rule_ac_color_s_var, int(state.step9_color_s))
            self.set_var(self.rule_ac_color_v_var, int(state.step9_color_v))
            self.set_var(self.rule_ac_color_h_tol_var, int(state.step9_color_h_tol))
            self.set_var(self.rule_ac_color_s_tol_var, int(state.step9_color_s_tol))
            self.set_var(self.rule_ac_color_v_tol_var, int(state.step9_color_v_tol))
            self.set_var(self.rule_ac_color_min_area_var, int(state.step9_color_min_area_px))
            self.set_var(self.rule_ac_color_morph_kernel_var, int(state.step9_color_morph_kernel))
            self.set_var(self.rule_ac_center_tolerance_var, float(state.step9_center_tolerance_px))

            path = self._get_calibration_path_from_gui()
            self.set_var(self.calibration_path_var, f"标定文件：{path}")
            missing = state.missing_items()
            if missing:
                self.set_var(self.calibration_status_var, "完整测量标定：已加载但未完成，缺少 " + "；".join(missing))
            else:
                self.set_var(self.calibration_status_var, "完整测量标定：已加载且完整")
            self.log(f"[完整测量标定] 已加载并同步到 GUI：{path}")
        except Exception as e:
            self.log(f"[完整测量标定] 加载失败：{e}")
            self.log(traceback.format_exc())
            error_msg = str(e)
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("加载标定失败", msg))

    def pause_midrun_recalibration_thread(self):
        self.run_in_thread(self.pause_midrun_recalibration)

    def pause_midrun_recalibration(self):
        """完整测量运行中：暂停当前 Stage34/Step9 运动，进入等待人工重标定状态。"""
        try:
            wf = self.workflow
            if wf is None:
                self.log("[中途重标定] 当前 workflow=None，无法暂停重标定。")
                self.root.after(0, lambda: messagebox.showwarning("无法暂停重标定", "当前没有正在运行/已初始化的完整测量 workflow。"))
                return
            if not bool(getattr(wf, "is_measuring", False)):
                self.log("[中途重标定] 当前完整测量未运行；如已失败/停止，请使用“重置后重新标定”。")
                self.root.after(0, lambda: messagebox.showwarning("完整测量未运行", "完整测量未运行。失败/停止后请使用“重置后重新标定”。"))
                return

            self.rule_ab_controller_stop_requested = True
            self.rule_ab_only_stop_requested = False
            try:
                self._stop_active_rule_ab_stage(reason="中途重标定按钮")
            except Exception:
                pass
            wf.request_midrun_recalibration_pause(cycle_index=wf.context.get("cycle_index"))
            self.set_var(self.flow_status_var, "流程状态：已请求暂停当前cycle，请等待运动停止后重新标定A/B/C/Step9")
            self.set_var(self.rule_module_status_var, "规则模块状态：中途重标定请求已发出，正在停止Stage34/Step9")
            self.log("[GUI] 已请求完整测量中途暂停重标定；请等待状态变为等待重标定后，再点击 A/B/C/Step9 标定按钮。")
        except Exception as e:
            self.log(f"[中途重标定] 请求失败：{e}")
            self.log(traceback.format_exc())
            error_msg = str(e)
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("中途重标定请求失败", msg))

    def resume_midrun_recalibration_thread(self):
        self.run_in_thread(self.resume_midrun_recalibration)

    def resume_midrun_recalibration(self):
        """用户重新标定 A/B/C/Step9 后，通知完整测量线程继续当前 cycle。"""
        try:
            wf = self.workflow
            if wf is None:
                raise RuntimeError("当前 workflow=None，无法继续完整测量。")
            if not bool(getattr(wf, "recalibration_in_progress", False) or getattr(wf, "restart_current_cycle_after_recalibration", False)):
                raise RuntimeError("当前不处于中途重标定等待状态。")

            # 保存并检查当前 GUI/标定文件，确保 A/B/C/Step9 都是最新的。
            state = self._current_or_new_calibration_state()
            self._save_calibration_file_raw(state)
            state = self._load_calibration_file_raw()
            missing = state.missing_items()
            if missing:
                raise RuntimeError("中途重标定尚未完成，缺少：\n" + "\n".join(f"- {x}" for x in missing))

            # 这里只置位“可以继续”的标志；真正重建运行态由完整测量线程执行，避免并发改 follower。
            wf.resume_after_recalibration_requested = True
            wf.stop_requested = False
            wf.step9_stop_requested = False
            wf.context["midrun_recalibration_resume_requested"] = True
            self.rule_ab_controller_stop_requested = False
            self.rule_ab_only_stop_requested = False
            self.set_var(self.flow_status_var, "流程状态：已完成重标定，完整测量将从当前cycle的Step1重新开始")
            self.set_var(self.rule_module_status_var, "规则模块状态：重标定完成，等待完整测量线程重建运行态")
            self.log("[GUI] 已确认中途重标定完成；完整测量线程将重新加载标定，并从当前 cycle 的 Step1 重跑。")
        except Exception as e:
            self.log(f"[中途重标定] 完成并继续失败：{e}")
            self.log(traceback.format_exc())
            error_msg = str(e)
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("中途重标定继续失败", msg))

    def step7_ac_preflight_thread(self):
        self.run_in_thread(self.step7_ac_preflight)

    def step7_ac_preflight(self):
        """GUI 按钮：完整测量前验证 Step7 A/C 标定是否可用。"""
        self.is_busy = True
        try:
            self.set_var(self.flow_status_var, "流程状态：正在执行 Step7 A/C 标定预检...")
            self._prepare_for_manual_calibration(reason="step7_ac_preflight")
            wf = self.ensure_workflow()
            self.sync_config_from_ui_to_workflow()
            result = wf.validate_step7_ac_calibration_preflight()
            ok = bool(result.get("ok", False))
            overlay = str(result.get("overlay_path", ""))
            json_path = str(result.get("json_path", ""))
            if ok:
                msg = f"Step7 A/C 标定预检通过。\nOverlay: {overlay}\nJSON: {json_path}"
                self.log(f"[Step7预检] 通过：overlay={overlay}; json={json_path}")
                self.set_var(self.flow_status_var, "流程状态：Step7 A/C 标定预检通过")
                self.root.after(0, lambda m=msg: messagebox.showinfo("Step7预检通过", m))
            else:
                failed = result.get("pass_items", {})
                msg = f"Step7 A/C 标定预检失败，请重新检查/标定 A 或 C。\n失败项: {failed}\nOverlay: {overlay}\nJSON: {json_path}"
                self.log(f"[Step7预检] 失败：{failed}; overlay={overlay}; json={json_path}")
                self.set_var(self.flow_status_var, "流程状态：Step7 A/C 标定预检失败")
                self.root.after(0, lambda m=msg: messagebox.showwarning("Step7预检失败", m))
        except Exception as e:
            self.log(f"[Step7预检] 执行失败：{e}")
            self.log(traceback.format_exc())
            self.set_var(self.flow_status_var, "流程状态：Step7 A/C 标定预检失败")
            err = str(e)
            self.root.after(0, lambda m=err: messagebox.showerror("Step7预检失败", m))
        finally:
            self.is_busy = False

    def reset_after_stop_for_recalibration_thread(self):
        self.run_in_thread(self.reset_after_stop_for_recalibration)

    def reset_after_stop_for_recalibration(self):
        """
        用户发现检测不好或上一轮失败后，用于重新进入“可标定/可运行”状态。

        推荐流程：
            停止测量 -> 关闭全部设备 -> 重置运行态/重新标定 -> 重新运行完整循环测量。
        这个函数不会删除旧的标定 JSON；后续重新点 A/B/C/Step9 会覆盖对应字段。
        """
        try:
            self.rule_ab_only_stop_requested = False
            self.rule_ab_controller_stop_requested = False
            self.rule_ab_active_follower = None

            if self.workflow is not None:
                self.workflow.reset_runtime_state_for_new_calibration(close_followers=True)

            self.is_busy = False
            self.set_var(self.flow_status_var, "流程状态：已重置运行态，可以重新标定A/B/C/Step9并重新运行")
            self.set_var(self.rule_module_status_var, "规则模块状态：已清理上一轮运行缓存")
            self.log("[GUI] 已重置上一轮停止/失败后的运行态")
        except Exception as e:
            self.log(f"[GUI] 重置运行态失败：{e}")
            self.log(traceback.format_exc())
            error_msg = str(e)
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("重置运行态失败", msg))

    def _prepare_for_manual_calibration(self, reason: str = ""):
        """
        每次人工标定前调用，避免上一轮 stop_requested=True 或 Step9 stop 标志
        阻塞新的截图/颜色选择。
        """
        try:
            self.rule_ab_only_stop_requested = False
            self.rule_ab_controller_stop_requested = False
            if self.workflow is not None:
                midrun_recal = bool(getattr(self.workflow, "recalibration_in_progress", False) or getattr(self.workflow, "restart_current_cycle_after_recalibration", False))
                self.workflow.stop_requested = False
                self.workflow.step9_stop_requested = False
                if not midrun_recal:
                    self.workflow.is_measuring = False
                    self.workflow.context["is_measuring"] = False
                    self.workflow.context["full_measurement_mode"] = False
                    self.workflow.context["strict_full_calibration_c_locked"] = False
                    self.workflow.context["strict_full_calibration_c_dir"] = ""
                    self.workflow.context["strict_full_calibration_c_mask_path"] = ""
                else:
                    self.workflow.is_measuring = True
                    self.workflow.context["is_measuring"] = True
                    self.log("[中途重标定] 当前处于完整测量暂停重标定状态：允许重新标定，但不结束完整测量线程。")
            if reason:
                self.log(f"[GUI] 人工标定前已解除上一轮停止锁：{reason}")
        except Exception as e:
            self.log(f"[GUI] 人工标定前清理运行态失败：{e}")

    def run_workflow_thread(self):
        if self.is_busy:
            messagebox.showwarning("正在运行", "当前已有任务在运行。")
            return
        self.worker_thread = self.run_in_thread(self.run_workflow)

    def run_workflow(self):
        self.is_busy = True

        try:
            # 新一轮完整测量启动前，必须清掉上一轮失败/停止残留。
            # 否则 stop_requested=True 会让截图、Step9 颜色检测或 run_one_cycle 直接退出。
            self._prepare_for_manual_calibration(reason="run_workflow_start")

            self.set_var(self.flow_status_var, "流程状态：正在运行循环测量.")

            # 运行前先同步一次 GUI 参数
            wf = self.ensure_workflow()
            # 完整测量运行中，Step7/Step9 每一步都会调用这些回调，
            # 使左侧第 8 区域的 GUI 修改无需重启即可在下一步生效。
            wf.rule_ab_config_sync_callback = self.sync_config_from_ui_to_workflow
            wf.step9_config_sync_callback = self.sync_config_from_ui_to_workflow

            # 每次点击“运行完整循环测量”都新建一个总文件夹，并清空右侧列表/本次绘图点。
            wf.begin_new_run_session()
            self.root.after(0, lambda: self.update_angle_fit_xy_list([]))

            # 1. 完整测量前先检查统一标定包。
            #    注意：这一步必须在初始化全部设备之前完成，保证人工标定不依赖设备初始化。
            calibration_state = wf.preflight_check_calibration()
            # 当前版本不再使用 KLT/Profile 指定边，因此运行前不再清空/重选指定边。
            wf.context["force_reselect_klt_edge_this_run"] = False
            wf.prepare_runtime_from_full_calibration(calibration_state)
            # prepare_runtime_from_full_calibration() 会把 C 复制到本次运行专用目录，
            # 因此这里必须重新从 workflow.context 取回更新后的 state。
            calibration_state = wf._get_loaded_calibration_state() or calibration_state
            wf.context["full_measurement_mode"] = True
            wf.context["strict_full_calibration_c_locked"] = True
            wf._force_cfg_to_strict_full_calibration_c(reason="run_workflow_after_prepare_runtime")

            # 运行完整循环测量时，完整标定包是唯一的 C 来源。
            # 这里同步 GUI 只是为了显示，不作为 RuleAB 的真实 C 来源。
            # 真实 C 来源已经固定在 wf.cfg.rule_ab_static_c_map_dir。
            self.root.after(0, lambda: self.rule_ab_static_c_map_dir_var.set(str(wf.cfg.rule_ab_static_c_map_dir)))
            self.root.after(0, lambda: self.rule_ab_load_static_c_map_var.set(True))
            self.root.after(0, lambda: self.rule_ab_force_reselect_c_var.set(False))
            # 真动开关的默认值由 config.py 决定；运行中若用户修改，下一步会同步生效。

            self.log(
                f"[流程] 完整测量标定检查通过并已注入后续分割/跟踪模块，标定文件={wf.get_calibration_path()}；"
                f"本次完整测量强制使用运行专用 C：{wf.cfg.rule_ab_static_c_map_dir}；"
                "不会再使用 RuleAB/RuleAC 单独测试区域残留的 C 文件夹。"
            )

            # 2. 确保实验设备已经初始化
            if wf.light is None or wf.laser_stage is None or wf.angle_module is None:
                self.log("[流程] 实验设备尚未完整初始化，先初始化实验设备")
                wf.connect_measurement_devices()

            # 2. 完整测量恢复真实 LabVIEW TCP 光谱仪流程。
            #    virtual 模式不连接真实 LabVIEW，但仍走同一个 request_labview_spectrum() 入口，
            #    让 Step4、单次采集、保存字段保持一致。
            if wf._is_virtual_hardware_mode():
                if wf.tcp_server is None or not bool(wf.context.get("tcp_started", False)):
                    wf.start_tcp_server()
                if not bool(wf.context.get("labview_ready", False)):
                    wf.wait_labview_ready()
                self.set_var(self.tcp_status_var, "TCP状态：virtual READY；完整测量将使用虚拟/回放光谱")
                self.log("[流程][virtual] 虚拟 TCP 已 READY；完整测量 Step4 将与单次光谱采集共用 request_labview_spectrum()")
            else:
                if bool(globals().get("SPECTROMETER_TCP_DISABLED", False)):
                    raise RuntimeError(
                        "SPECTROMETER_TCP_DISABLED=True，完整测量不会再用1000伪造光谱数据；"
                        "请把 SPECTROMETER_TCP_DISABLED 改为 False 后再运行完整循环测量。"
                    )

                if wf.tcp_server is None or not bool(wf.context.get("tcp_started", False)):
                    wf.start_tcp_server()
                if not bool(wf.context.get("labview_ready", False)):
                    wf.wait_labview_ready()
                self.set_var(self.tcp_status_var, "TCP状态：READY，可采集；完整测量将使用真实光谱")
                self.log("[流程] LabVIEW TCP 已启动并完成 READY 检查；完整测量 Step4 将与单次光谱采集共用 request_labview_spectrum()")

            wf.is_measuring = True
            wf.stop_requested = False
            wf.context["is_measuring"] = True
            wf.notify_update()

            # 不再用 for range 固定 max_cycles
            # 改成 while，这样每轮开始前都可以读取 GUI 最新的 max_cycles
            cycle_index = 1

            while True:
                if wf.stop_requested or not wf.is_measuring:
                    break

                wf = self.sync_config_from_ui_to_workflow()
                wf._force_cfg_to_strict_full_calibration_c(reason=f"run_workflow_cycle_{cycle_index}_before_run_one_cycle")

                max_cycles_now = int(wf.cfg.max_cycles)

                if cycle_index > max_cycles_now:
                    self.log(
                    f"[流程] 当前 cycle_index={cycle_index} 已超过 GUI 当前循环次数 "
                    f"max_cycles={max_cycles_now}，结束循环"
                    )
                    break

                self.log(
                    f"[流程] 准备执行第 {cycle_index} 轮；"
                    f"已读取本轮开始前 GUI 最新参数："
                    f"signal_on_time_ms={wf.cfg.signal_on_time_ms}, "
                    f"stable_wait_ms={wf.cfg.stable_wait_ms}, "
                    f"factor={wf.cfg.signal_time_factor}, "
                    f"angle_range=[{wf.cfg.angle_delta_min_deg}, {wf.cfg.angle_delta_max_deg}], "
                    f"num={wf.cfg.angle_num}, cw={wf.cfg.angle_cw}, "
                    f"raw_remove_above={wf.cfg.raw_remove_above}, "
                    f"median_window={wf.cfg.median_filter_window}"
                )

                should_continue = wf.run_one_cycle(cycle_index)

                self.refresh_result_labels(wf)

                if bool(getattr(wf, "restart_current_cycle_after_recalibration", False)):
                    wf.prepare_runtime_for_midrun_recalibration_waiting()
                    self.set_var(
                        self.flow_status_var,
                        f"流程状态：第 {cycle_index} 轮已暂停，等待重新标定A/B/C/Step9；完成后点“完成重标定并继续测量”"
                    )
                    self.set_var(self.rule_module_status_var, "规则模块状态：等待中途重标定完成")
                    while (
                        bool(getattr(wf, "restart_current_cycle_after_recalibration", False))
                        and not bool(getattr(wf, "resume_after_recalibration_requested", False))
                        and not bool(getattr(wf, "stop_requested", False))
                    ):
                        time.sleep(0.1)

                    if bool(getattr(wf, "stop_requested", False)):
                        self.log("[中途重标定] 等待重标定期间收到停止测量请求，结束完整测量。")
                        break

                    state = self._load_calibration_file_raw()
                    wf.resume_after_midrun_recalibration(state)
                    wf.is_measuring = True
                    wf.stop_requested = False
                    wf.context["is_measuring"] = True
                    self.set_var(self.flow_status_var, f"流程状态：中途重标定完成，重新执行第 {cycle_index} 轮 Step1")
                    self.log(f"[中途重标定] 重新执行第 {cycle_index} 轮；cycle_index 不递增。")
                    continue

                if not should_continue:
                    break

                cycle_index += 1

            wf.finish()
            self.set_var(self.flow_status_var, "流程状态：测量结束")

        except Exception as e:
            self.set_var(self.flow_status_var, "流程状态：运行失败")
            self.log(f"[GUI错误] 运行失败：{e}")
            self.log(traceback.format_exc())
            error_msg = str(e)
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("运行失败", msg))

        finally:
            self.is_busy = False

    def request_stop(self):
        # 同时停止完整测量线程和“单独A推动B”线程。
        # 单独A推动B不创建 MeasurementWorkflow，因此必须使用单独的停止标志。
        self.rule_ab_only_stop_requested = True
        self.rule_ab_controller_stop_requested = True

        # 停止光谱补焦循环
        self.spectrum_autofocus_stop_requested = True

        # 立即停止当前 RuleAB 的 Stage34，避免等待循环自然结束。
        self._stop_active_rule_ab_stage(reason="停止测量按钮")

        if self.workflow is not None:
            # 完整测量如果正卡在 Step9 1/2 轴运动里，仅 request_stop() 只能让主循环退出；
            # 这里额外调用 stop_step9_stage12()，立即置位 step9_stop_requested 并尝试发送 Stage12 stop。
            try:
                self.workflow.stop_step9_stage12()
            except Exception as e:
                self.log(f"[GUI] 请求停止时停止Step9 1/2轴失败：{e}")

            self.workflow.request_stop()
        self.set_var(self.flow_status_var, "流程状态：已请求停止")
        self.log("[GUI] 已请求停止测量 / 单独A推动B / Step9 1/2轴 / 光谱补焦循环")

    def _stop_active_rule_ab_stage(self, reason: str = "用户请求") -> bool:
        """
        立即停止当前单独 A沿C/RuleAB 线程里的 Stage34 控制器。

        返回：
            True  表示找到了 stage 并成功调用 stop_all()；
            False 表示当前没有正在运行的 follower/stage，或者 stage 不支持 stop_all()。
        """
        follower = getattr(self, "rule_ab_active_follower", None)
        if follower is None:
            self.log(f"[RuleAB-C] 当前没有正在运行的 RuleAB follower，无法执行控制器停止；原因：{reason}")
            return False

        try:
            stage = getattr(follower, "stage", None)
            if stage is None:
                self.log(f"[RuleAB-C] 当前 follower.stage=None，可能 enable_stage=False；原因：{reason}")
                return False

            if hasattr(stage, "stop_all"):
                stage.stop_all()
                self.log(f"[RuleAB-C] 已立即执行 stage.stop_all()；原因：{reason}")
                return True

            self.log(f"[RuleAB-C] 当前 stage 对象没有 stop_all() 方法；原因：{reason}")
            return False
        except Exception as e:
            self.log(f"[RuleAB-C] 立即停止控制器失败：{e}")
            self.log(traceback.format_exc())
            return False

    def stop_rule_ab_controller_thread(self):
        self.run_in_thread(self.stop_rule_ab_controller)

    def stop_rule_ab_controller(self):
        """
        单独停止 A推B/A沿C 控制器按钮。

        作用：
            1. 立即对当前 RuleAB follower 的 Stage34 调用 stop_all()；
            2. 设置 rule_ab_only_stop_requested=True，让单独 A沿C 循环退出；
            3. 不关闭 Rigol、LabVIEW TCP、照明光，也不刷新光谱图。
        """
        self.rule_ab_controller_stop_requested = True
        self.rule_ab_only_stop_requested = True
        ok = self._stop_active_rule_ab_stage(reason="停止A推B控制器按钮")
        if ok:
            self.set_var(self.rule_module_status_var, "规则模块状态：已请求停止A推B控制器，Stage34已stop_all")
        else:
            self.set_var(self.rule_module_status_var, "规则模块状态：已请求停止A推B控制器，但当前未找到可停止的Stage34")
        self.log("[GUI] 已点击停止A推B控制器按钮")

    def stop_step9_stage12_thread(self):
        """
        “停止1/2轴运动”按钮入口。

        注意：
            _build_ui() 中按钮绑定的是 self.stop_step9_stage12_thread。
            因此 GUI 类里必须有这个方法，否则创建界面时会直接 AttributeError。
        """
        self.run_in_thread(self.stop_step9_stage12)

    def stop_step9_stage12(self):
        """
        停止 Step9 的 1/2 轴运动。

        作用：
            1. 设置 workflow.step9_stop_requested=True，让 Step9 循环在下一轮检测前退出；
            2. 立即尝试对当前 Stage12 控制器发送 stop/stop_all；
            3. 不关闭照明光、Rigol、LabVIEW TCP，也不影响 A推B 的 Stage34。
        """
        try:
            if self.workflow is None:
                self.log("[Step9-颜色中心] 当前 workflow=None，没有正在运行的 1/2 轴可停止")
                self.set_var(self.rule_module_status_var, "规则模块状态：当前没有正在运行的Step9 1/2轴")
                return

            self.workflow.stop_step9_stage12()
            self.set_var(self.rule_module_status_var, "规则模块状态：已请求停止Step9 1/2轴运动")
            self.log("[GUI] 已点击停止1/2轴运动按钮")

        except Exception as e:
            self.log(f"[Step9-颜色中心] GUI停止1/2轴失败：{e}")
            self.log(traceback.format_exc())
            error_msg = str(e)
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("停止1/2轴失败", msg))

    # --------------------------------------------------------
    # 单独设备测试按钮
    # --------------------------------------------------------

    def connect_light_thread(self):
        self.run_in_thread(self.connect_light)

    def connect_light(self):
        try:
            wf = self.ensure_workflow()
            wf.connect_light()
            self.set_var(self.light_status_var, "照明光状态：已连接")
        except Exception as e:
            self.log(f"[照明光] 连接失败：{e}")
            error_msg = str(e)
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("照明光连接失败", msg))

    def light_on_thread(self):
        self.run_in_thread(self.light_on)

    def light_on(self):
        try:
            wf = self.ensure_workflow()
            if wf.light is None:
                wf.connect_light()
            wf.light_on()
        except Exception as e:
            self.log(f"[照明光] ON失败：{e}")

    def light_off_thread(self):
        self.run_in_thread(self.light_off)

    def light_off(self):
        try:
            wf = self.ensure_workflow()
            if wf.light is None:
                wf.connect_light()
            wf.light_off()
        except Exception as e:
            self.log(f"[照明光] OFF失败：{e}")

    def connect_rigol_thread(self):
        self.run_in_thread(self.connect_rigol)

    def connect_rigol(self):
        """兼容旧按钮名称：现在连接 Newport 激光开关控制器。"""
        try:
            wf = self.ensure_workflow()
            wf.connect_laser_controller()
            self.refresh_result_labels(wf)
        except Exception as e:
            self.log(f"[激光开关] 连接失败：{e}")
            error_msg = str(e)
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("激光控制器连接失败", msg))

    def configure_rigol_thread(self):
        self.run_in_thread(self.configure_rigol)

    def configure_rigol(self):
        """兼容旧按钮名称：现在设置 Newport 激光轴速度/加速度。"""
        try:
            wf = self.ensure_workflow()
            wf.connect_laser_controller()
            wf.set_laser_velocity_accel(
                axis=int(wf.cfg.laser_axis),
                speed=int(wf.cfg.laser_speed),
                accel=int(wf.cfg.laser_accel),
            )
            self.refresh_result_labels(wf)
        except Exception as e:
            self.log(f"[激光开关] 设置速度/加速度失败：{e}")

    def signal_ch1_on_thread(self):
        self.run_in_thread(self.signal_ch1_on)

    def signal_ch1_on(self):
        try:
            wf = self.ensure_workflow()
            if wf.laser_stage is None:
                wf.connect_laser_controller()
            wf.laser_on()
            self.refresh_result_labels(wf)
        except Exception as e:
            self.log(f"[激光开关] ON失败：{e}")

    def signal_ch1_off_thread(self):
        self.run_in_thread(self.signal_ch1_off)

    def signal_ch1_off(self):
        try:
            wf = self.ensure_workflow()
            if wf.laser_stage is None:
                wf.connect_laser_controller()
            wf.laser_off()
            self.refresh_result_labels(wf)
        except Exception as e:
            self.log(f"[激光开关] OFF失败：{e}")

    def signal_ch2_on_thread(self):
        self.run_in_thread(self.signal_ch2_on)

    def signal_ch2_on(self):
        try:
            wf = self.ensure_workflow()
            wf.signal_ch2_on()
            self.refresh_result_labels(wf)
        except Exception as e:
            self.log(f"[激光开关] CH2未使用提示失败：{e}")

    def signal_ch2_off_thread(self):
        self.run_in_thread(self.signal_ch2_off)

    def signal_ch2_off(self):
        try:
            wf = self.ensure_workflow()
            wf.signal_ch2_off()
            self.refresh_result_labels(wf)
        except Exception as e:
            self.log(f"[激光开关] CH2保持关闭提示失败：{e}")

    # 兼容旧名称，避免外部代码仍调用时报错。
    def signal_all_on_thread(self):
        self.signal_ch1_on_thread()

    def signal_all_on(self):
        self.signal_ch1_on()

    def signal_all_off_thread(self):
        self.signal_ch1_off_thread()

    def signal_all_off(self):
        self.signal_ch1_off()

    def init_angle_thread(self):
        self.run_in_thread(self.init_angle)

    def init_angle(self):
        try:
            wf = self.ensure_workflow()

            # 点击“初始化角度模块”时，强制按照 GUI 当前参数重新创建 ScreenAngleDetector。
            # 这样 num/cw 修改后不需要重启程序。
            if wf.angle_module is not None:
                wf.angle_module = None

            wf.init_angle_module()
            self.set_var(
                self.angle_status_var,
                f"角度检测状态：已初始化，模型已加载一次；num={wf.cfg.angle_num}，cw={wf.cfg.angle_cw}"
            )
        except Exception as e:
            self.set_var(self.angle_status_var, "角度检测状态：初始化失败")
            self.log(f"[角度检测] 初始化失败：{e}")
            error_msg = str(e)
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("角度检测初始化失败", msg))

    def detect_angle_once_thread(self):
        self.run_in_thread(self.detect_angle_once)

    def detect_angle_once(self):
        try:
            wf = self.ensure_workflow()
            if wf.angle_module is None:
                wf.init_angle_module()

            result = wf.detect_angle_once(label="manual")
            self.set_var(
                self.angle_result_var,
                f"角度结果：{result.get('angle_deg')}°，"
                f"num={wf.cfg.angle_num}，cw={wf.cfg.angle_cw}，"
                f"center={result.get('center')}",
            )
            self.set_var(self.angle_status_var, "角度检测状态：检测成功")
            self.refresh_result_labels(wf)

        except Exception as e:
            self.set_var(self.angle_status_var, "角度检测状态：检测失败")
            self.log(f"[角度检测] 单次检测失败：{e}")
            self.log(traceback.format_exc())

    def start_tcp_thread(self):
        self.run_in_thread(self.start_tcp)

    def start_tcp(self):
        try:
            wf = self.ensure_workflow()

            # 启动 TCP 前更新一次 TCP / 光谱仪配置
            wf.cfg.tcp_host = self.tcp_host_var.get().strip()
            wf.cfg.tcp_port = int(self.tcp_port_var.get())
            wf.cfg.tcp_output_dir = self.tcp_output_dir_var.get().strip()
            wf.cfg.tcp_command = self.tcp_command_var.get().strip()

            # 新增：同步光谱仪设备选择
            wf.cfg.spectrometer_backend = str(self.spectrometer_backend_var.get()).strip().lower() or "labview_tcp"
            wf.cfg.picam_exposure = float(self.picam_exposure_var.get())
            wf.cfg.picam_temperature = float(self.picam_temperature_var.get())
            wf.cfg.picam_roi_width = int(self.picam_roi_width_var.get())
            wf.cfg.picam_roi_height = int(self.picam_roi_height_var.get())

            wf.start_tcp_server()
            self.set_var(self.tcp_status_var, "TCP状态：已启动，等待 LabVIEW 连接")
            self.refresh_result_labels(wf)

        except Exception as e:
            self.log(f"[TCP] 启动失败：{e}")
            error_msg = str(e)
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("TCP启动失败", msg))

    def wait_ready_thread(self):
        self.run_in_thread(self.wait_ready)

    def wait_ready(self):
        try:
            wf = self.ensure_workflow()
            wf.wait_labview_ready()

            self.set_var(self.tcp_status_var, "TCP状态：READY，可采集")
            self.refresh_result_labels(wf)

        except Exception as e:
            self.set_var(self.tcp_status_var, "TCP状态：READY失败")
            self.log(f"[TCP] READY失败：{e}")
            error_msg = str(e)
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("READY失败", msg))

    def measure_spectrum_once_thread(self):
        self.run_in_thread(self.measure_spectrum_once)

    def measure_spectrum_once(self):
        try:
            wf = self.ensure_workflow()
            result = wf.request_labview_spectrum(cycle_index=0)

            self.set_var(
                self.tcp_result_var,
                f"最近光谱：num_points={result.get('num_points')} "
                f"raw_original_peak={wf.context.get('raw_original_peak')} "
                f"raw_filtered_peak={wf.context.get('raw_filtered_peak')} "
                f"raw_median_peak={wf.context.get('raw_median_peak')} "
                f"fit_peak={wf.context.get('fit_peak')} "
                f"csv={result.get('csv_path')}",
            )
            self.set_var(self.tcp_status_var, "TCP状态：采集完成")
            self.refresh_result_labels(wf)

        except Exception as e:
            self.set_var(self.tcp_status_var, "TCP状态：采集失败")
            self.log(f"[TCP] 单次采集失败：{e}")
            self.log(traceback.format_exc())

    def save_current_spectrum_background_light_thread(self):
        self.run_in_thread(self.save_current_spectrum_background_light)

    def save_current_spectrum_background_light(self):
        """将当前 context 中的实时光谱数据保存到 ./save/{MM.DD}。"""
        try:
            wf = self.ensure_workflow()
            date_dir = datetime.now().strftime("%m.%d")
            save_dir = Path("./save") / date_dir
            saved_path = wf.save_single_spectrum_to_xlsx(
                save_dir=save_dir,
                tag="background_light",
            )
            if saved_path is not None:
                self.set_var(self.tcp_status_var, f"已保存背景光光谱：{saved_path}")
                self.log(f"[TCP] 已保存当前光谱数据（背景光）：{saved_path}")
            else:
                self.set_var(self.tcp_status_var, "保存失败：无可用光谱数据")
                self.log("[TCP] 保存当前光谱数据（背景光）失败：无可用光谱数据")
        except Exception as e:
            self.set_var(self.tcp_status_var, "保存失败")
            self.log(f"[TCP] 保存当前光谱数据（背景光）失败：{e}")
            self.log(traceback.format_exc())

    def select_step9_target_thread(self):
        self.run_in_thread(self.select_step9_target)

    def select_step9_target(self):
        """
        人工选定 Step9 的目标位置。

        该位置是截图区域内坐标，用于后续颜色检测区域中心对齐。
        左键点击目标位置，Enter/N 确认，R 重选，ESC 取消。
        """
        try:
            self._prepare_for_manual_calibration(reason="select_step9_target")
            wf = self.ensure_workflow()
            output_dir = Path(wf.output_root) / "step9_target_select"
            output_dir.mkdir(parents=True, exist_ok=True)
            image_rgb = self._capture_current_rule_ab_frame(output_dir=output_dir)

            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            scale = 0.85
            h, w = image_bgr.shape[:2]
            show_w = max(1, int(w * scale))
            show_h = max(1, int(h * scale))
            display = cv2.resize(image_bgr, (show_w, show_h), interpolation=cv2.INTER_AREA)

            selected: List[Tuple[int, int]] = []
            window_name = "Select Step9 target: color region center target"

            def redraw():
                canvas = display.copy()
                lines = [
                    "Select Step9 target point",
                    "Left click: target position",
                    "Enter/N: confirm | R: reset | ESC/Q: cancel",
                ]
                for i, s in enumerate(lines):
                    cv2.putText(canvas, s, (18, 28 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA)

                if selected:
                    x, y = selected[-1]
                    cv2.drawMarker(canvas, (x, y), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=22, thickness=2)
                    cv2.putText(canvas, f"target=({x / scale:.1f},{y / scale:.1f})", (x + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 0, 0), 2, cv2.LINE_AA)
                return canvas

            def on_mouse(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    selected.clear()
                    selected.append((int(x), int(y)))

            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, show_w, show_h)
            cv2.setMouseCallback(window_name, on_mouse)

            try:
                while True:
                    cv2.imshow(window_name, redraw())
                    key = cv2.waitKey(30) & 0xFF

                    if key in (27, ord("q"), ord("Q")):
                        raise RuntimeError("用户取消了 Step9 目标点选择。")
                    if key in (ord("r"), ord("R")):
                        selected.clear()
                    if key in (13, 10, ord("n"), ord("N")):
                        if not selected:
                            self.log("[Step9-颜色中心] 请先左键点击目标位置。")
                            continue
                        break
            finally:
                try:
                    cv2.destroyWindow(window_name)
                except Exception:
                    pass

            sx, sy = selected[-1]
            target_x = float(sx) / float(scale)
            target_y = float(sy) / float(scale)

            self.set_var_blocking(self.rule_ac_target_x_var, round(target_x, 3))
            self.set_var_blocking(self.rule_ac_target_y_var, round(target_y, 3))

            # 只点“选定Step9目标”时，也立即写入 workflow.cfg 和标定包；
            # 后续再点“4 标定Step9目标/颜色”或“单独测试Step9对齐”都能使用新目标。
            try:
                wf.cfg.rule_ac_target_x_px = target_x
                wf.cfg.rule_ac_target_y_px = target_y
                state = self._current_or_new_calibration_state()
                state.step9_target_x_px = float(target_x)
                state.step9_target_y_px = float(target_y)
                self._save_calibration_file_raw(state)
            except Exception as e:
                self.log(f"[Step9-颜色中心] 写入目标点到标定文件失败，但GUI目标已更新：{e}")

            self.log(f"[Step9-颜色中心] 已选定目标位置：x={target_x:.3f}, y={target_y:.3f}（截图区域内坐标）")
            self.set_var(self.rule_module_status_var, f"Step9目标已选定：({target_x:.1f}, {target_y:.1f})")
            return (target_x, target_y)

        except Exception as e:
            self.log(f"[Step9-颜色中心] 目标点选择失败：{e}")
            self.log(traceback.format_exc())
            error_msg = str(e)
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("Step9目标选择失败", msg))
            return None


    def test_rule_ab_thread(self):
        self.run_in_thread(self.test_rule_ab)

    def test_rule_ab(self):
        """
        单独点击按钮时，只执行新的 RuleAB 视觉闭环：

            1. SAM2 每帧分割/更新 A、B、C；
            2. 不再根据 A-B 距离控制；
            3. A 沿 C 的局部切向运动一圈，并持续维持 A-C 距离在 min~max 内；
            4. 每一帧都检测 B 目标边角度；
            5. A 和 B mask 有覆盖时不再强制停止，仍按 A-C 距离和 C 边界规则运动；
            6. A/B 覆盖面积只记录到日志和 CSV，不参与停止判定；
            7. 不连接 Rigol、不采集光谱、不读取 xlsx、不刷新光谱图。
        """
        follower = None

        def _angle_diff_180(a: Optional[float], b: Optional[float]) -> Optional[float]:
            if a is None or b is None:
                return None
            diff = abs(float(a) - float(b)) % 180.0
            if diff > 90.0:
                diff = 180.0 - diff
            return float(diff)

        def _stop_stage_safe(reason: str):
            try:
                stage = getattr(follower, "stage", None)
                if stage is not None and hasattr(stage, "stop_all"):
                    stage.stop_all()
                    self.log(f"[RuleAB-C] 已执行 stage.stop_all()，原因：{reason}")
            except Exception as e:
                self.log(f"[RuleAB-C] stage.stop_all() 失败：{e}")

        def _configure_stage_ch3_ch4_separately(
            ch3_velocity: int,
            ch3_acceleration: int,
            ch4_velocity: int,
            ch4_acceleration: int,
            ch3_max_voltage: int,
            ch4_max_voltage: int,
        ):
            try:
                stage = getattr(follower, "stage", None)
                if stage is None:
                    self.log("[RuleAB-C] enable_stage=False 或 stage=None，跳过 CH3/CH4 单独速度设置")
                    return

                if hasattr(stage, "setup_channel"):
                    stage.setup_channel(
                        channel=3,
                        max_voltage=int(ch3_max_voltage),
                        velocity=int(ch3_velocity),
                        acceleration=int(ch3_acceleration),
                    )
                    stage.setup_channel(
                        channel=4,
                        max_voltage=int(ch4_max_voltage),
                        velocity=int(ch4_velocity),
                        acceleration=int(ch4_acceleration),
                    )
                    self.log(
                        "[RuleAB-C] 已分别设置 Stage34："
                        f"CH3 velocity={ch3_velocity}, acceleration={ch3_acceleration}; "
                        f"CH4 velocity={ch4_velocity}, acceleration={ch4_acceleration}; "
                        f"CH3 max_voltage={ch3_max_voltage}, CH4 max_voltage={ch4_max_voltage}"
                    )
                    return

                dev = getattr(stage, "dev", None)
                if dev is not None and hasattr(dev, "setup_drive"):
                    dev.setup_drive(
                        max_voltage=int(ch3_max_voltage),
                        velocity=int(ch3_velocity),
                        acceleration=int(ch3_acceleration),
                        channel=3,
                    )
                    dev.setup_drive(
                        max_voltage=int(ch4_max_voltage),
                        velocity=int(ch4_velocity),
                        acceleration=int(ch4_acceleration),
                        channel=4,
                    )
                    self.log("[RuleAB-C] 已通过 stage.dev 分别设置 CH3/CH4 速度和加速度")
                    return

                self.log("[RuleAB-C] 当前 stage 对象没有 setup_channel/setup_drive 接口，无法分别设置 CH3/CH4")
            except Exception as e:
                self.log(f"[RuleAB-C] 分别设置 CH3/CH4 速度和加速度失败：{e}")
                self.log(traceback.format_exc())


        # 运行中动态参数同步：
        #   允许用户在 RuleAB 单独测试运行过程中修改 GUI 输入框；
        #   下一帧/下一次运动会读取这些新值并写入 follower.cfg。
        #   注意：C 是否重新选择、截图区域、SAM2模型路径这类初始化级参数仍需重启本次 RuleAB 测试。
        _last_rule_ab_stage_signature = None

        def _apply_rule_ab_runtime_ui_params(frame_tag: str = "") -> Dict[str, Any]:
            nonlocal _last_rule_ab_stage_signature

            live_cfg = self.build_config_from_ui()

            # 同步到 workflow.cfg，保证下次点击测试/完整流程也使用最新 GUI 参数。
            try:
                wf_live = self.workflow
                if wf_live is not None:
                    wf_live.cfg = live_cfg
            except Exception:
                pass

            if follower is None:
                return {
                    "loop_interval_s": float(getattr(live_cfg, "rule_ab_loop_interval_s", 0.15)),
                    "ok": False,
                }

            fc = follower.cfg

            # -------- 运动规则参数：下一帧立即生效 --------
            fc.loop_interval_s = float(getattr(live_cfg, "rule_ab_loop_interval_s", getattr(fc, "loop_interval_s", 0.15)))
            fc.max_cycles = int(getattr(live_cfg, "rule_ab_max_steps", getattr(fc, "max_cycles", 10_000_000)))

            fc.a_c_target_clearance = float(getattr(live_cfg, "rule_ab_ac_target_clearance_px", getattr(fc, "a_c_target_clearance", 90.0)))
            fc.a_c_min_clearance = float(getattr(live_cfg, "rule_ab_ac_min_clearance_px", getattr(fc, "a_c_min_clearance", 80.0)))
            fc.a_c_max_clearance = float(getattr(live_cfg, "rule_ab_ac_max_clearance_px", getattr(fc, "a_c_max_clearance", 100.0)))
            fc.follow_c_direction = int(getattr(live_cfg, "rule_ab_follow_c_direction", getattr(fc, "follow_c_direction", 1)))
            fc.a_c_correction_mode = str(getattr(live_cfg, "rule_ab_ac_correction_mode", getattr(fc, "a_c_correction_mode", "nearest_normal")))
            fc.a_b_overlap_min_area_px = float(getattr(live_cfg, "rule_ab_ab_overlap_min_area_px", getattr(fc, "a_b_overlap_min_area_px", 1.0)))
            fc.a_b_overlap_stop = False

            # -------- Stage 参数：下一次 action 立即生效 --------
            fc.enable_stage = bool(getattr(live_cfg, "rule_ab_enable_stage", getattr(fc, "enable_stage", False)))
            fc.stage_x_channel = int(getattr(live_cfg, "rule_ab_stage_x_channel", getattr(fc, "stage_x_channel", 3)))
            fc.stage_y_channel = int(getattr(live_cfg, "rule_ab_stage_y_channel", getattr(fc, "stage_y_channel", 4)))

            fc.stage_ch3_velocity = int(getattr(live_cfg, "rule_ab_stage_ch3_velocity", getattr(fc, "stage_ch3_velocity", 10)))
            fc.stage_ch3_acceleration = int(getattr(live_cfg, "rule_ab_stage_ch3_acceleration", getattr(fc, "stage_ch3_acceleration", 10)))
            fc.stage_ch3_max_voltage = int(getattr(live_cfg, "rule_ab_stage_ch3_max_voltage", getattr(fc, "stage_ch3_max_voltage", 50)))

            fc.stage_ch4_velocity = int(getattr(live_cfg, "rule_ab_stage_ch4_velocity", getattr(fc, "stage_ch4_velocity", 10)))
            fc.stage_ch4_acceleration = int(getattr(live_cfg, "rule_ab_stage_ch4_acceleration", getattr(fc, "stage_ch4_acceleration", 10)))
            fc.stage_ch4_max_voltage = int(getattr(live_cfg, "rule_ab_stage_ch4_max_voltage", getattr(fc, "stage_ch4_max_voltage", 50)))

            fc.stage_default_velocity = int(getattr(live_cfg, "rule_ab_stage_velocity", getattr(fc, "stage_default_velocity", fc.stage_ch3_velocity)))
            fc.stage_default_acceleration = int(getattr(live_cfg, "rule_ab_stage_acceleration", getattr(fc, "stage_default_acceleration", fc.stage_ch3_acceleration)))
            fc.stage_default_max_voltage = int(getattr(live_cfg, "rule_ab_stage_max_voltage", getattr(fc, "stage_default_max_voltage", max(fc.stage_ch3_max_voltage, fc.stage_ch4_max_voltage))))

            fc.stage_step_x = int(getattr(live_cfg, "rule_ab_stage_step_x", getattr(fc, "stage_step_x", 20)))
            fc.stage_step_y = int(getattr(live_cfg, "rule_ab_stage_step_y", getattr(fc, "stage_step_y", 20)))
            fc.action_step = int(getattr(live_cfg, "rule_ab_action_step", getattr(fc, "action_step", 20)))

            fc.stage_ch3_pause_after_move_s = float(getattr(live_cfg, "rule_ab_stage_ch3_pause_after_move_s", getattr(fc, "stage_ch3_pause_after_move_s", 2.0)))
            fc.stage_ch4_pause_after_move_s = float(getattr(live_cfg, "rule_ab_stage_ch4_pause_after_move_s", getattr(fc, "stage_ch4_pause_after_move_s", 2.0)))

            fc.stage_x_sign = int(getattr(live_cfg, "rule_ab_stage_x_sign", getattr(fc, "stage_x_sign", -1)))
            fc.stage_y_sign = int(getattr(live_cfg, "rule_ab_stage_y_sign", getattr(fc, "stage_y_sign", 1)))

            # 如果用户运行中改了 CH3/CH4 速度/加速度/电压，则下一次运动前重新下发到底层控制器。
            stage_signature = (
                int(fc.stage_ch3_velocity),
                int(fc.stage_ch3_acceleration),
                int(fc.stage_ch3_max_voltage),
                int(fc.stage_ch4_velocity),
                int(fc.stage_ch4_acceleration),
                int(fc.stage_ch4_max_voltage),
            )
            if stage_signature != _last_rule_ab_stage_signature:
                _configure_stage_ch3_ch4_separately(
                    ch3_velocity=int(fc.stage_ch3_velocity),
                    ch3_acceleration=int(fc.stage_ch3_acceleration),
                    ch4_velocity=int(fc.stage_ch4_velocity),
                    ch4_acceleration=int(fc.stage_ch4_acceleration),
                    ch3_max_voltage=int(fc.stage_ch3_max_voltage),
                    ch4_max_voltage=int(fc.stage_ch4_max_voltage),
                )
                _last_rule_ab_stage_signature = stage_signature
                self.log(
                    f"[RuleAB-C][运行中参数更新{frame_tag}] "
                    f"CH3=({fc.stage_ch3_velocity},{fc.stage_ch3_acceleration},{fc.stage_ch3_max_voltage}V,"
                    f"step_x={fc.stage_step_x},pause={fc.stage_ch3_pause_after_move_s}s), "
                    f"CH4=({fc.stage_ch4_velocity},{fc.stage_ch4_acceleration},{fc.stage_ch4_max_voltage}V,"
                    f"step_y={fc.stage_step_y},pause={fc.stage_ch4_pause_after_move_s}s)"
                )

            # -------- A/B 跟踪阈值：下一帧分割立即生效 --------
            for name in (
                "rule_ab_temporal_position_match_enable",
                "rule_ab_temporal_position_match_margin_px",
                "rule_ab_temporal_position_match_min_score",
                "rule_ab_temporal_position_match_use_last_image_template",
                "rule_ab_temporal_stop_on_track_fail",
                "rule_ab_use_sam2_video_tracking",
                "rule_ab_sam2_video_tracking_mode",
                "rule_ab_sam2_video_temp_dir",
                "rule_ab_sam2_video_prompt_max_points",
                "rule_ab_sam2_video_area_ratio_min",
                "rule_ab_sam2_video_area_ratio_max",
                "rule_ab_sam2_video_center_jump_max_px",
                "rule_ab_sam2_video_fallback_to_template",
            ):
                short = name.replace("rule_ab_", "")
                if hasattr(fc, short):
                    value = getattr(live_cfg, name)
                    # follower.cfg 里这个字段也允许保存为 Path，避免后续直接调用 mkdir 出错。
                    if short == "sam2_video_temp_dir":
                        value = Path(str(value))
                    setattr(fc, short, value)

            # 同步到 segmenter 内部。部分属性名与 RuntimeConfig 一致。
            seg = getattr(follower, "abc_segmenter", None)
            if seg is not None:
                mapping = {
                    "temporal_position_match_enable": "rule_ab_temporal_position_match_enable",
                    "temporal_position_match_margin_px": "rule_ab_temporal_position_match_margin_px",
                    "temporal_position_match_min_score": "rule_ab_temporal_position_match_min_score",
                    "temporal_position_match_use_last_image_template": "rule_ab_temporal_position_match_use_last_image_template",
                    "temporal_stop_on_track_fail": "rule_ab_temporal_stop_on_track_fail",
                    "use_sam2_video_tracking": "rule_ab_use_sam2_video_tracking",
                    "sam2_video_tracking_mode": "rule_ab_sam2_video_tracking_mode",
                    "sam2_video_temp_dir": "rule_ab_sam2_video_temp_dir",
                    "sam2_video_prompt_max_points": "rule_ab_sam2_video_prompt_max_points",
                    "sam2_video_area_ratio_min": "rule_ab_sam2_video_area_ratio_min",
                    "sam2_video_area_ratio_max": "rule_ab_sam2_video_area_ratio_max",
                    "sam2_video_center_jump_max_px": "rule_ab_sam2_video_center_jump_max_px",
                    "sam2_video_fallback_to_template": "rule_ab_sam2_video_fallback_to_template",
                }
                for seg_attr, cfg_attr in mapping.items():
                    if hasattr(seg, seg_attr):
                        try:
                            value = getattr(live_cfg, cfg_attr)
                            # 关键修复：
                            # sam2_video_temp_dir 在 segmenter 内部必须是 Path，
                            # 运行中从 GUI 读取时是 str，如果直接 setattr 会导致：
                            # AttributeError: 'str' object has no attribute 'mkdir'
                            if seg_attr == "sam2_video_temp_dir":
                                value = Path(str(value))
                            setattr(seg, seg_attr, value)
                        except Exception:
                            pass

            return {
                "ok": True,
                "loop_interval_s": float(fc.loop_interval_s),
                "stage_step_x": int(fc.stage_step_x),
                "stage_step_y": int(fc.stage_step_y),
                "ch3_pause": float(fc.stage_ch3_pause_after_move_s),
                "ch4_pause": float(fc.stage_ch4_pause_after_move_s),
            }


        def _save_and_record(image_rgb, scene, status, action_id: int, action_name: str, direction: str):
            try:
                if bool(getattr(follower.cfg, "save_annotated_image", True)) and hasattr(follower, "save_annotated_image"):
                    follower.save_annotated_image(
                        image_rgb=image_rgb,
                        scene=scene,
                        status=status,
                        action_name=action_name,
                    )
            except Exception as e:
                self.log(f"[RuleAB-C] 保存标注图失败：{e}")

            try:
                if hasattr(follower, "record_cycle"):
                    follower.record_cycle(
                        scene=scene,
                        status=status,
                        action_id=action_id,
                        action_name=action_name,
                        direction=direction,
                    )
            except Exception as e:
                self.log(f"[RuleAB-C] 写入 RuleAB CSV 失败：{e}")

        try:
            self.rule_ab_only_stop_requested = False
            self.rule_ab_controller_stop_requested = False
            self.rule_ab_active_follower = None
            self.set_var(self.rule_module_status_var, "规则模块状态：A沿C绕行初始化中")
            self.log("========== 单独A沿C绕行(多点SAM2)：覆盖B时不再停；持续维持A-C距离并沿C运动 ==========")

            cfg_gui = self.build_config_from_ui()
            calib_state = None
            try:
                wf_for_calib = self.ensure_workflow()
                calib_state = wf_for_calib._get_loaded_calibration_state()
                if calib_state is not None:
                    wf_for_calib.apply_calibration_state(calib_state)
                    cfg_gui = wf_for_calib.cfg
            except Exception:
                calib_state = None

            capture_area = cfg_gui.capture_area
            save_root = str(getattr(cfg_gui, "save_root", "measurement_output"))
            rule_ab_enable_stage = bool(getattr(cfg_gui, "rule_ab_enable_stage", False))
            rule_ab_loop_interval_s = float(getattr(cfg_gui, "rule_ab_loop_interval_s", 0.15))

            stage_conn = str(getattr(cfg_gui, "rule_ab_stage_conn", "97101208"))
            stage_x_channel = int(getattr(cfg_gui, "rule_ab_stage_x_channel", 3))
            stage_y_channel = int(getattr(cfg_gui, "rule_ab_stage_y_channel", 4))

            stage_ch3_velocity = int(getattr(cfg_gui, "rule_ab_stage_ch3_velocity", 10))
            stage_ch3_acceleration = int(getattr(cfg_gui, "rule_ab_stage_ch3_acceleration", 10))
            stage_ch4_velocity = int(getattr(cfg_gui, "rule_ab_stage_ch4_velocity", 10))
            stage_ch4_acceleration = int(getattr(cfg_gui, "rule_ab_stage_ch4_acceleration", 10))
            stage_ch3_max_voltage = int(getattr(cfg_gui, "rule_ab_stage_ch3_max_voltage", 50))
            stage_ch4_max_voltage = int(getattr(cfg_gui, "rule_ab_stage_ch4_max_voltage", 50))

            stage_default_velocity = int(getattr(cfg_gui, "rule_ab_stage_velocity", stage_ch3_velocity))
            stage_default_acceleration = int(getattr(cfg_gui, "rule_ab_stage_acceleration", stage_ch3_acceleration))
            stage_max_voltage = int(getattr(cfg_gui, "rule_ab_stage_max_voltage", max(stage_ch3_max_voltage, stage_ch4_max_voltage)))

            stage_step_x = int(getattr(cfg_gui, "rule_ab_stage_step_x", 20))
            stage_step_y = int(getattr(cfg_gui, "rule_ab_stage_step_y", 20))
            action_step = int(getattr(cfg_gui, "rule_ab_action_step", min(stage_step_x, stage_step_y)))
            stage_x_sign = int(getattr(cfg_gui, "rule_ab_stage_x_sign", -1))
            stage_y_sign = int(getattr(cfg_gui, "rule_ab_stage_y_sign", 1))

            ac_target = float(getattr(cfg_gui, "rule_ab_ac_target_clearance_px", 90.0))
            ac_min = float(getattr(cfg_gui, "rule_ab_ac_min_clearance_px", 80.0))
            ac_max = float(getattr(cfg_gui, "rule_ab_ac_max_clearance_px", 100.0))
            follow_c_direction = int(getattr(cfg_gui, "rule_ab_follow_c_direction", 1))
            ab_overlap_threshold = float(getattr(cfg_gui, "rule_ab_ab_overlap_min_area_px", 1.0))

            confirm_first_frame_segmentation = bool(
                getattr(cfg_gui, "rule_ab_confirm_first_frame_segmentation", True)
            )
            confirm_window_scale = float(getattr(cfg_gui, "rule_ab_confirm_window_scale", 0.85))
            c_contour_mode = str(getattr(cfg_gui, "rule_ab_c_contour_mode", "quadrilateral"))
            c_quadrilateral_method = str(
                getattr(cfg_gui, "rule_ab_c_quadrilateral_method", "body_edge")
            )

            static_c_map_name = str(getattr(cfg_gui, "rule_ab_static_c_map_name", "default_static_c_map"))
            static_c_map_dir = str(getattr(cfg_gui, "rule_ab_static_c_map_dir", "")).strip()
            load_static_c_map_if_exists = bool(getattr(cfg_gui, "rule_ab_load_static_c_map_if_exists", True))
            force_reselect_c_each_run = bool(getattr(cfg_gui, "rule_ab_force_reselect_c_each_run", False))

            # 如果用户已经通过“选择C文件夹”或“单独分割C”指定了 C 标定目录，
            # 这里先规范化目录，确保后续 ActualNanoBoundaryFollower 能加载到 static_c_mask.png。
            resolved_static_c_dir = self._resolve_static_c_reference_dir(static_c_map_dir) if static_c_map_dir else None
            if resolved_static_c_dir is not None:
                static_c_map_dir = str(resolved_static_c_dir)
                self.rule_ab_static_c_map_dir_var.set(static_c_map_dir)
                self.rule_ab_load_static_c_map_var.set(True)
                self.rule_ab_force_reselect_c_var.set(False)
                load_static_c_map_if_exists = True
                force_reselect_c_each_run = False
                self.log(f"[RuleAB-C] 本次 A沿C绕行将加载固定 C 分割结果：{static_c_map_dir}")

            # 注意：这里不再把时间戳放进 output_dir。ActualNanoBoundaryFollower 内部会自动创建 run_xxx。
            # 这样固定 C map 会保存在稳定目录 rule_ab_follow_c_only/_static_c_map_library 下，
            # GUI 关闭后重新打开仍然可以加载。
            output_dir = str(Path(save_root) / "rule_ab_follow_c_only")

            self.log(
                "[RuleAB-C] 参数："
                f"enable_stage={rule_ab_enable_stage}, loop_interval_s={rule_ab_loop_interval_s}, "
                f"A-C[min,target,max]=[{ac_min}, {ac_target}, {ac_max}] px, "
                f"AB_overlap_threshold={ab_overlap_threshold}px², follow_c_direction={follow_c_direction}, "
                f"a_c_correction_mode=nearest_normal, "
                f"confirm_first_frame_segmentation={confirm_first_frame_segmentation}, "
                f"c_contour_mode={c_contour_mode}, c_quadrilateral_method={c_quadrilateral_method}, "
                f"stage_conn={stage_conn}, x_channel={stage_x_channel}, y_channel={stage_y_channel}, "
                f"CH3 velocity={stage_ch3_velocity}, CH3 acceleration={stage_ch3_acceleration}, "
                f"CH4 velocity={stage_ch4_velocity}, CH4 acceleration={stage_ch4_acceleration}, "
                f"CH3_voltage={stage_ch3_max_voltage}, CH4_voltage={stage_ch4_max_voltage}, "
                f"max_voltage={stage_max_voltage}, step_x={stage_step_x}, step_y={stage_step_y}, action_step={action_step}, "
                f"CH3_pause={float(getattr(cfg_gui, 'rule_ab_stage_ch3_pause_after_move_s', 2.0))}s, "
                f"CH4_pause={float(getattr(cfg_gui, 'rule_ab_stage_ch4_pause_after_move_s', 2.0))}s, "
                f"运动步数规则=LEFT/RIGHT使用step_x，UP/DOWN使用step_y，action_step仅兜底, "
                f"static_c_map_name={static_c_map_name}, static_c_map_dir={static_c_map_dir}, load_C={load_static_c_map_if_exists}, force_reselect_C={force_reselect_c_each_run}, "
                f"x_sign={stage_x_sign}, y_sign={stage_y_sign}, output_dir={output_dir}"
            )

            rule_cfg = RuleABRuntimeConfig(
                capture_area=capture_area,
                output_dir=output_dir,
                sam2_device="cuda",

                enable_stage=rule_ab_enable_stage,
                max_cycles=10_000_000,
                loop_interval_s=rule_ab_loop_interval_s,

                # 第一帧 SAM2 分割确认 + C 四边形轮廓拟合。
                confirm_first_frame_segmentation=confirm_first_frame_segmentation,
                confirm_window_scale=confirm_window_scale,
                c_contour_mode=c_contour_mode,
                c_quadrilateral_method=c_quadrilateral_method,

                static_c_map_name=static_c_map_name,
                static_c_map_dir=static_c_map_dir,
                reuse_static_c_map_in_memory=True,
                load_static_c_map_if_exists=load_static_c_map_if_exists,
                save_static_c_map_library=True,
                force_reselect_c_each_run=force_reselect_c_each_run,

                # A/B 位置跟踪：默认使用第一帧人工确认的初始化模板定位当前帧，
                # 避免上一帧错误模板污染后续帧。匹配失败时拒绝更新 A/B 并暂停本帧运动。
                temporal_position_match_enable=bool(getattr(cfg_gui, "rule_ab_temporal_position_match_enable", True)),
                temporal_position_match_margin_px=float(getattr(cfg_gui, "rule_ab_temporal_position_match_margin_px", 220.0)),
                temporal_position_match_min_score=float(getattr(cfg_gui, "rule_ab_temporal_position_match_min_score", 0.15)),
                temporal_position_match_use_last_image_template=bool(getattr(cfg_gui, "rule_ab_temporal_position_match_use_last_image_template", True)),
                temporal_stop_on_track_fail=bool(getattr(cfg_gui, "rule_ab_temporal_stop_on_track_fail", True)),
                use_sam2_video_tracking=bool(getattr(cfg_gui, "rule_ab_use_sam2_video_tracking", True)),
                sam2_video_tracking_mode=str(getattr(cfg_gui, "rule_ab_sam2_video_tracking_mode", "two_frame_anchor")),
                sam2_video_temp_dir=str(getattr(cfg_gui, "rule_ab_sam2_video_temp_dir", "outputs/sam2_video_tracking_tmp")),
                sam2_video_prompt_max_points=int(getattr(cfg_gui, "rule_ab_sam2_video_prompt_max_points", 5)),
                sam2_video_area_ratio_min=float(getattr(cfg_gui, "rule_ab_sam2_video_area_ratio_min", 0.35)),
                sam2_video_area_ratio_max=float(getattr(cfg_gui, "rule_ab_sam2_video_area_ratio_max", 2.80)),
                sam2_video_center_jump_max_px=float(getattr(cfg_gui, "rule_ab_sam2_video_center_jump_max_px", 220.0)),
                sam2_video_fallback_to_template=bool(getattr(cfg_gui, "rule_ab_sam2_video_fallback_to_template", True)),

                # 新规则：沿 C 绕行 + A-C 距离控制；A/B 覆盖不再强制停。
                a_c_target_clearance=ac_target,
                a_c_min_clearance=ac_min,
                a_c_max_clearance=ac_max,
                # A 可能位于 C 的上/下/左/右任意方向，因此用最近边界法向修正距离。
                # 例：A 在 C 上方，太远则 DOWN 靠近；太近则 UP 远离。
                a_c_correction_mode="nearest_normal",
                follow_c_direction=follow_c_direction,
                a_b_overlap_stop=False,
                a_b_overlap_min_area_px=ab_overlap_threshold,

                stage_conn=stage_conn,
                stage_x_channel=stage_x_channel,
                stage_y_channel=stage_y_channel,
                stage_default_velocity=stage_default_velocity,
                stage_default_acceleration=stage_default_acceleration,
                stage_default_max_voltage=stage_max_voltage,
                stage_ch3_velocity=stage_ch3_velocity,
                stage_ch3_acceleration=stage_ch3_acceleration,
                stage_ch3_max_voltage=stage_ch3_max_voltage,
                stage_ch4_velocity=stage_ch4_velocity,
                stage_ch4_acceleration=stage_ch4_acceleration,
                stage_ch4_max_voltage=stage_ch4_max_voltage,
                stage_step_x=stage_step_x,
                stage_step_y=stage_step_y,
                action_step=action_step,
                stage_ch3_pause_after_move_s=float(getattr(cfg_gui, "rule_ab_stage_ch3_pause_after_move_s", 2.0)),
                stage_ch4_pause_after_move_s=float(getattr(cfg_gui, "rule_ab_stage_ch4_pause_after_move_s", 2.0)),
                stage_x_sign=stage_x_sign,
                stage_y_sign=stage_y_sign,
            )

            if calib_state is not None:
                self.set_var(self.rule_module_status_var, "规则模块状态：已加载完整标定，正在无弹窗初始化 A/B/C")
                try:
                    rule_cfg.confirm_first_frame_segmentation = False
                    rule_cfg.force_reselect_c_each_run = False
                    rule_cfg.load_static_c_map_if_exists = True
                    resolved_c_dir = wf_for_calib._resolve_static_c_dir_from_state(calib_state)
                    if resolved_c_dir:
                        rule_cfg.static_c_map_dir = resolved_c_dir
                except Exception:
                    pass
            else:
                self.set_var(self.rule_module_status_var, "规则模块状态：请在弹出的图像中点击/确认 A、B、C")
            self.log("[RuleAB-C] 正在初始化 ActualNanoBoundaryFollower")
            follower = ActualNanoBoundaryFollower(rule_cfg)
            self.rule_ab_active_follower = follower
            if calib_state is not None:
                wf_for_calib._inject_calibration_into_rule_ab_follower(follower, calib_state)

            _configure_stage_ch3_ch4_separately(
                ch3_velocity=stage_ch3_velocity,
                ch3_acceleration=stage_ch3_acceleration,
                ch4_velocity=stage_ch4_velocity,
                ch4_acceleration=stage_ch4_acceleration,
                ch3_max_voltage=stage_ch3_max_voltage,
                ch4_max_voltage=stage_ch4_max_voltage,
            )

            self.log("[RuleAB-C] 初始化第一帧 A/B/C")
            if calib_state is not None:
                wf_for_calib._call_initialize_abc_with_loaded_calibration(follower, calib_state)
            else:
                follower.initialize_abc_with_first_frame()

            self.set_var(self.rule_module_status_var, "规则模块状态：A沿C绕行中；AB覆盖不再强制停止")
            self.log("[RuleAB-C] 开始持续运行。点击“停止A推B控制器”或“停止测量”可立即停止控制器。")

            frame_idx = 0
            motion_steps = 0
            overlap_monitor_frames = 0  # 仅统计，不再触发停止
            lost_count = 0
            last_b_angle: Optional[float] = None

            action_name_map = {0: "STAY", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT"}
            direction_map = {0: "stay", 1: "up", 2: "down", 3: "left", 4: "right"}

            while not self.rule_ab_only_stop_requested:
                frame_idx += 1

                # 每一帧开始前重新读取 GUI 输入框。
                # 用户在 RuleAB 单独测试过程中修改参数后，从这一帧/下一次动作立即生效。
                live_rule_ab_params = _apply_rule_ab_runtime_ui_params(frame_tag=f"/frame={frame_idx}")
                rule_ab_loop_interval_s = float(live_rule_ab_params.get("loop_interval_s", rule_ab_loop_interval_s))

                try:
                    image_rgb, scene = follower.capture_and_build_scene()
                    # 自定义循环没有调用 follower.run_one_cycle()，所以必须手动更新 cycle_index。
                    # 否则 save_annotated_image() 会反复使用同一个 cycle 编号，
                    # 在 action_name 相同的情况下保存文件会被覆盖，看起来只剩一张图。
                    follower.cycle_index = frame_idx
                    status = follower.get_boundary_rule_status(scene)
                    lost_count = 0
                except Exception as e:
                    lost_count += 1
                    self.log(
                        f"[RuleAB-C] frame={frame_idx} 视觉检测/分割失败：{e}，"
                        f"lost_count={lost_count}/{int(getattr(rule_cfg, 'max_lost_frames', 10))}"
                    )
                    self.log(traceback.format_exc())
                    _stop_stage_safe("视觉检测/分割失败")

                    if lost_count >= int(getattr(rule_cfg, "max_lost_frames", 10)):
                        self.log("[RuleAB-C] 连续视觉失败过多，停止单独A沿C绕行(多点SAM2)")
                        break

                    time.sleep(max(0.10, rule_ab_loop_interval_s))
                    continue

                b_angle = status.get("b_edge_angle_deg")
                d_angle = _angle_diff_180(b_angle, last_b_angle)
                last_b_angle = b_angle

                ab_overlap = bool(status.get("ab_overlap", False))
                ab_overlap_area = float(status.get("ab_overlap_area", 0.0) or 0.0)
                c_clearance = float(status.get("c_clearance", float("nan")))

                # A/B 覆盖不再触发停机；只保留 overlap 面积用于日志、CSV 和可视化。
                if ab_overlap:
                    overlap_monitor_frames += 1
                action_id = int(follower.rule_policy(scene))
                action_name = action_name_map.get(action_id, f"ACTION_{action_id}")
                direction = direction_map.get(action_id, "unknown")

                self.log(
                    f"[RuleAB-C] 绕行判断 frame={frame_idx}, "
                    f"B_edge_angle={b_angle}, d_angle={d_angle}, "
                    f"AB_overlap_area={ab_overlap_area:.1f}px², "
                    f"A-C={c_clearance:.2f}px [{status.get('ac_status')}], "
                    f"policy_action={action_name}"
                )

                _save_and_record(image_rgb, scene, status, action_id, action_name, direction)

                # 用户可能在本帧分割/判断过程中按下“停止A推B控制器”。
                # 这里在真正发送运动命令前再检查一次，防止刚停止又执行下一步运动。
                if self.rule_ab_controller_stop_requested or self.rule_ab_only_stop_requested:
                    _stop_stage_safe("运动命令发送前检测到停止请求")
                    self.log("[RuleAB-C] 已收到停止控制器请求，本帧不再发送运动命令，准备退出循环")
                    break

                if action_id == 0:
                    self.log("[RuleAB-C] 策略输出 STAY，本帧不运动；继续检测")
                else:
                    move_info = follower.execute_action(action_id) or {}
                    motion_steps += 1

                    actual_channel = move_info.get("channel", "?")
                    actual_channel_name = move_info.get("channel_name", "?")
                    actual_step = move_info.get("move_step", "?")
                    actual_pause = move_info.get("pause_s", 0.0)
                    actual_voltage = move_info.get("voltage", "?")
                    actual_velocity = move_info.get("velocity", "?")
                    actual_acceleration = move_info.get("acceleration", "?")
                    dry_run = bool(move_info.get("dry_run", False))

                    self.log(
                        f"[RuleAB-C] 已执行 Stage 动作：{action_name}; "
                        f"channel={actual_channel_name}/{actual_channel}, "
                        f"voltage={actual_voltage}, velocity={actual_velocity}, acceleration={actual_acceleration}, "
                        f"move_step={actual_step}, pause_after_move={float(actual_pause):.3f}s, "
                        f"dry_run={dry_run}, motion_steps={motion_steps}"
                    )

                if rule_ab_loop_interval_s > 0:
                    time.sleep(rule_ab_loop_interval_s)

            _stop_stage_safe("单独A沿C绕行(多点SAM2)停止/结束")

            self.set_var(
                self.rule_module_status_var,
                (
                    "规则模块状态：单独A沿C绕行(多点SAM2)结束，"
                    f"frames={frame_idx}，motion_steps={motion_steps}，"
                    f"overlap_monitor_frames={overlap_monitor_frames}"
                )
            )
            self.log(
                "[RuleAB-C] 单独A沿C绕行(多点SAM2)结束："
                f"frames={frame_idx}, motion_steps={motion_steps}, "
                f"overlap_monitor_frames={overlap_monitor_frames}"
            )

        except Exception as e:
            self.set_var(self.rule_module_status_var, "规则模块状态：单独A沿C绕行(多点SAM2)失败")
            self.log(f"[RuleAB-C] 单独A沿C绕行(多点SAM2)失败：{e}")
            self.log(traceback.format_exc())

        finally:
            if follower is not None:
                try:
                    _stop_stage_safe("单独A沿C绕行(多点SAM2) finally")
                except Exception:
                    pass

                try:
                    if hasattr(follower, "close"):
                        follower.close()
                    self.log("[RuleAB-C] follower 已关闭")
                except Exception as e:
                    self.log(f"[RuleAB-C] follower 关闭失败：{e}")
                finally:
                    if self.rule_ab_active_follower is follower:
                        self.rule_ab_active_follower = None


    def test_rule_ac_thread(self):
        self.run_in_thread(self.test_rule_ac)

    def test_rule_ac(self):
        """
        Step9 单独测试：颜色检测区域中心对齐。

        使用颜色检测区域中心对齐；需要提前选定目标位置和颜色 HSV。
        A/B 分割结果每帧保存用于检查，但不参与 Step9 的运动方向和停止判断。
        """
        try:
            self.set_var(self.rule_module_status_var, "规则模块状态：Step9 颜色检测区域中心对齐测试中")
            self.log("========== Step9 单独测试：颜色检测区域中心对齐 ==========")

            # 如果尚未选定目标点，则先让用户选择。
            try:
                tx = float(self.rule_ac_target_x_var.get())
                ty = float(self.rule_ac_target_y_var.get())
            except Exception:
                tx, ty = -1.0, -1.0

            if tx < 0 or ty < 0:
                self.log("[Step9-颜色中心] 尚未设置目标点，先进入目标点选择。")
                self.select_step9_target()

            # 同步 GUI 参数。
            wf = self.sync_config_from_ui_to_workflow()

            # 每次点击“单独测试Step9对齐”都新建一个独立保存文件夹。
            wf.begin_step9_run_session()

            # 注入运行中同步回调：
            # Step9 循环每一轮开始前都会调用该回调读取 GUI 最新 RuleAC 输入。
            wf.step9_config_sync_callback = self.sync_config_from_ui_to_workflow

            # 重置 Step9 停止状态和动态 C/B 跟踪结果。
            wf.step9_stop_requested = False
            wf.reset_step9_tracking_state(clear_b_points=False)
            wf.step9_color_last_mask = None
            wf.step9_color_last_center = None
            # Step9 使用颜色检测区域中心对齐；如果 HSV 未设置则先选择颜色。

            self.log(
                f"[Step9-颜色中心] 参数：target=({wf.cfg.rule_ac_target_x_px}, {wf.cfg.rule_ac_target_y_px}), "
                f"tol={wf.cfg.rule_ac_center_tolerance_px}, "
                f"step={wf.cfg.rule_ac_stage_step_size}, "
                f"max_cycles={wf.cfg.rule_ac_max_cycles}, "
                f"enable_stage={wf.cfg.rule_ac_enable_stage}, "
                "detection=颜色检测区域, need_color_selection=True, "
                f"Stage12速度/加速度/电压={wf.cfg.rule_ac_stage12_velocity}/"
                f"{wf.cfg.rule_ac_stage12_acceleration}/{wf.cfg.rule_ac_stage12_max_voltage}, "
                f"run_dir={wf.step9_current_run_dir}"
            )

            if int(getattr(wf.cfg, "rule_ac_color_h", -1)) < 0 or int(getattr(wf.cfg, "rule_ac_color_s", -1)) < 0 or int(getattr(wf.cfg, "rule_ac_color_v", -1)) < 0:
                self.log("[Step9-颜色中心] 尚未设置颜色 HSV，先进入颜色选择。")
                wf.prepare_step9_color_region_detection()
                self.sync_workflow_to_ui()

            result = wf.run_rule_ac_until_threshold()
            last_row = result.get("last_row") or {}

            self.set_var(
                self.rule_module_status_var,
                f"规则模块状态：Step9完成，ok={result.get('ok')}，"
                f"center=({last_row.get('center_x')}, {last_row.get('center_y')}), "
                f"target={result.get('target')}, action={last_row.get('action_code')}, "
                f"保存={result.get('run_dir')}"
            )
            self.refresh_result_labels(wf)

        except Exception as e:
            self.set_var(self.rule_module_status_var, "规则模块状态：Step9测试失败")
            self.log(f"[Step9-颜色中心] 单独测试失败：{e}")
            self.log(traceback.format_exc())
            error_msg = str(e)
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("Step9测试失败", msg))

    def close_tcp_thread(self):
        self.run_in_thread(self.close_tcp)

    def close_tcp(self):
        try:
            if self.workflow is not None and self.workflow.tcp_server is not None:
                self.workflow.tcp_server.close()
                self.workflow.tcp_server = None
                self.workflow.context["tcp_started"] = False
                self.workflow.context["labview_ready"] = False
            self.set_var(self.tcp_status_var, "TCP状态：已关闭")
            self.log("[TCP] 已关闭")
        except Exception as e:
            if self.workflow is not None:
                self.workflow.context["tcp_started"] = False
                self.workflow.context["labview_ready"] = False
            self.log(f"[TCP] 关闭失败：{e}")

    # --------------------------------------------------------
    # 关闭
    # --------------------------------------------------------

    def close_all_thread(self):
        self.run_in_thread(self.close_all)

    def close_all(self):
        try:
            if self.workflow is not None:
                self.workflow.request_stop()
                self.workflow.close_all()

            # 关闭全部设备后允许立刻重新标定/重新运行。
            self.rule_ab_only_stop_requested = False
            self.rule_ab_controller_stop_requested = False
            self.rule_ab_active_follower = None
            self.is_busy = False
            if self.workflow is not None:
                self.workflow.reset_runtime_state_for_new_calibration(close_followers=False)

            self.set_var(self.flow_status_var, "流程状态：GUI清理完成，已重置运行态，可重新标定/运行")
            self.set_var(self.light_status_var, "照明光状态：已关闭/释放")
            self.set_var(self.rigol_status_var, "激光控制器状态：已关闭")
            tcp_started = bool(self.workflow.context.get("tcp_started", False)) if self.workflow is not None else False
            labview_ready = bool(self.workflow.context.get("labview_ready", False)) if self.workflow is not None else False
            if tcp_started or labview_ready:
                self.set_var(self.tcp_status_var, f"TCP状态：保持连接/可复用（started={tcp_started}, ready={labview_ready}）")
            else:
                self.set_var(self.tcp_status_var, "TCP状态：未启动或保持原状态；如需关闭请点“关闭TCP”")

        except Exception as e:
            self.log(f"[GUI] 关闭全部设备失败：{e}")

    # --------------------------------------------------------
    # 光谱补焦循环
    # --------------------------------------------------------

    def _parse_focus_roi(self, text: str) -> Tuple[int, int, int, int]:
        """解析 Focus ROI 字符串为整数元组。"""
        try:
            parts = [p.strip() for p in str(text).split(",")]
            if len(parts) != 4:
                raise ValueError("格式应为 x,y,w,h")
            return tuple(int(p) for p in parts)  # type: ignore
        except Exception as e:
            self.log(f"[光谱补焦] ROI 解析失败({text})：{e}，使用默认值")
            return (0, 0, 300, 300)

    def _parse_int_sequence(self, text: str) -> Tuple[int, ...]:
        """解析形如 '10,20,30' 的整数序列。"""
        try:
            values = [int(p.strip()) for p in str(text).split(",") if str(p).strip()]
            values = [v for v in values if v > 0]
            if values:
                return tuple(values)
        except Exception as e:
            self.log(f"[光谱补焦] 序列解析失败({text})：{e}，使用默认值")
        return (10, 20, 30)

    def _make_saf_config(self) -> AutofocusConfig:
        """根据 GUI 当前值构造光谱补焦循环用的 AutofocusConfig。

        使用补焦专用截图区域 saf_capture_area，不再复用标定/角度检测的 capture_area。
        """
        capture_area = tuple(
            int(v) for v in self._parse_focus_roi(self.saf_capture_area_var.get())
        )
        focus_roi = self._parse_focus_roi(self.saf_roi_var.get())
        return AutofocusConfig(
            capture_mode="screen_region",
            capture_area=capture_area,
            focus_roi=focus_roi,
            autofocus_enabled=True,
            autofocus_focus_trigger_ratio=float(self.saf_trigger_ratio_var.get()),
            autofocus_stop_ratio=float(self.saf_stop_ratio_var.get()),
            autofocus_focus_trigger_count=int(self.saf_trigger_count_var.get()),
            autofocus_trigger_absolute=bool(self.saf_trigger_absolute_var.get()),
            autofocus_detection_only=bool(self.saf_detection_only_var.get()),
            autofocus_passive_mode=bool(self.saf_passive_mode_var.get()),
            autofocus_passive_max_attempts=int(self.saf_passive_attempts_var.get()),
            autofocus_passive_consecutive_good=int(self.saf_passive_good_var.get()),
            autofocus_passive_disable_auto_stop=bool(self.saf_disable_auto_stop_var.get()),
            z_enabled=bool(self.saf_z_enabled_var.get()),
            z_axis=int(self.saf_z_axis_var.get()),
            z_speed=int(self.saf_z_speed_var.get()),
            z_accel=int(self.saf_z_accel_var.get()),
            z_search_strategy=str(self.saf_search_strategy_var.get()),
            z_search_steps=int(self.saf_z_search_steps_var.get()),
            z_patience=int(self.saf_z_patience_var.get()),
            z_direction_probe_steps=tuple(
                max(1, int(self.saf_z_direction_probe_step_interval_var.get())) * i
                for i in range(1, max(1, int(self.saf_z_direction_probe_stage_count_var.get())) + 1)
            ),
            z_direction_probe_stage_count=max(1, int(self.saf_z_direction_probe_stage_count_var.get())),
            z_direction_probe_step_interval=max(1, int(self.saf_z_direction_probe_step_interval_var.get())),
            z_direction_probe_samples=int(self.saf_z_direction_probe_samples_var.get()),
            z_direction_probe_points_per_step=int(self.saf_z_direction_probe_points_var.get()),
            z_local_refine_enabled=bool(self.saf_z_local_refine_enabled_var.get()),
            z_local_refine_decay=float(self.saf_z_local_refine_decay_var.get()),
            z_local_refine_min_step=int(self.saf_z_local_refine_min_step_var.get()),
            z_local_refine_max_rounds=int(self.saf_z_local_refine_max_rounds_var.get()),
        )

    def open_focus_score_preview(self):
        """打开实时 FocusScore 折线图预览窗口。"""
        try:
            if (
                self.focus_preview_window is not None
                and self.focus_preview_window.winfo_exists()
            ):
                self.focus_preview_window.lift()
                return
        except Exception:
            self.focus_preview_window = None

        self.focus_preview_history = []
        self.focus_preview_started_at = time.time()
        self.focus_preview_running = True
        self.focus_preview_sample_inflight = False
        self.focus_preview_status_var.set("FocusScore预览：等待采样")

        win = tk.Toplevel(self.root)
        win.title("FocusScore 实时预览")
        win.geometry("760x480")
        win.minsize(560, 360)
        win.protocol("WM_DELETE_WINDOW", self.close_focus_score_preview)
        self.focus_preview_window = win

        frame = ttk.Frame(win, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        self.focus_preview_fig = Figure(figsize=(7.2, 3.8), dpi=100)
        self.focus_preview_ax = self.focus_preview_fig.add_subplot(111)
        self.focus_preview_canvas = FigureCanvasTkAgg(self.focus_preview_fig, master=frame)
        self.focus_preview_canvas.get_tk_widget().grid(row=0, column=0, columnspan=2, sticky="nsew")

        ttk.Label(frame, textvariable=self.focus_preview_status_var).grid(
            row=1, column=0, padx=4, pady=(8, 0), sticky="w"
        )
        ttk.Button(frame, text="关闭预览", command=self.close_focus_score_preview).grid(
            row=1, column=1, padx=4, pady=(8, 0), sticky="e"
        )

        self._redraw_focus_score_preview()
        self._schedule_focus_score_preview_sample(delay_ms=10)

    def close_focus_score_preview(self):
        """关闭 FocusScore 预览窗口并停止后续刷新。"""
        self.focus_preview_running = False
        if self.focus_preview_after_id is not None:
            try:
                self.root.after_cancel(self.focus_preview_after_id)
            except Exception:
                pass
            self.focus_preview_after_id = None
        if self.focus_preview_window is not None:
            try:
                self.focus_preview_window.destroy()
            except Exception:
                pass
        self.focus_preview_window = None
        self.focus_preview_canvas = None
        self.focus_preview_fig = None
        self.focus_preview_ax = None
        self.focus_preview_status_var.set("FocusScore预览：已关闭")

    def _schedule_focus_score_preview_sample(self, delay_ms: int = 1000):
        if not self.focus_preview_running:
            return
        self.focus_preview_after_id = self.root.after(
            max(10, int(delay_ms)), self._start_focus_score_preview_sample
        )

    def _start_focus_score_preview_sample(self):
        if not self.focus_preview_running or self.focus_preview_sample_inflight:
            self._schedule_focus_score_preview_sample()
            return
        self.focus_preview_sample_inflight = True
        self.run_in_thread(self._focus_score_preview_sample_worker)

    def _focus_score_preview_sample_worker(self):
        score: Optional[float] = None
        status = ""
        try:
            score, status = self._compute_focus_score_preview_value()
        except Exception as e:
            status = f"采样失败：{e}"
        self.root.after(0, lambda s=score, m=status: self._finish_focus_score_preview_sample(s, m))

    def _finish_focus_score_preview_sample(self, score: Optional[float], status: str):
        self.focus_preview_sample_inflight = False
        if not self.focus_preview_running:
            return
        if score is not None:
            t0 = self.focus_preview_started_at or time.time()
            elapsed_s = max(0.0, time.time() - t0)
            self.focus_preview_history.append((elapsed_s, float(score)))
            if len(self.focus_preview_history) > 300:
                self.focus_preview_history = self.focus_preview_history[-300:]
            self.focus_preview_status_var.set(
                f"FocusScore预览：score={float(score):.4f}，点数={len(self.focus_preview_history)}，来源={status}"
            )
        else:
            self.focus_preview_status_var.set(f"FocusScore预览：{status or 'score=None'}")
        self._redraw_focus_score_preview()
        self._schedule_focus_score_preview_sample(delay_ms=1000)

    def _compute_focus_score_preview_value(self) -> Tuple[Optional[float], str]:
        """计算当前 FocusScore，优先使用光谱补焦 ROI/参考。"""
        loop = self.spectrum_autofocus_loop
        if loop is not None and bool(getattr(loop, "_reference_ready", False)):
            metrics_calc = loop._get_or_create_metrics_calc()
            scorer = loop._get_or_create_scorer()
            live = metrics_calc.capture_live()
            score, _ = scorer.score_ratio(live.get("roi_metrics"))
            return score, "光谱补焦ROI"

        wf = self.workflow
        if (
            wf is not None
            and bool(getattr(wf, "_focus_reference_ready", False))
            and getattr(wf, "_focus_scorer", None) is not None
        ):
            return wf.compute_current_focus_score(), "完整流程Focus参考"

        return None, "未建立Focus参考，请先选择ROI/建立参考或启动补焦循环"

    def _redraw_focus_score_preview(self):
        if self.focus_preview_ax is None or self.focus_preview_canvas is None:
            return
        ax = self.focus_preview_ax
        ax.clear()
        ax.set_title("FocusScore Ratio")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("score")
        ax.grid(True, alpha=0.3)

        if self.focus_preview_history:
            xs = [p[0] for p in self.focus_preview_history]
            ys = [p[1] for p in self.focus_preview_history]
            ax.plot(xs, ys, marker="o", linewidth=1.6)
            ymin = min(ys)
            ymax = max(ys)
            pad = max(0.01, (ymax - ymin) * 0.2)
            ax.set_ylim(ymin - pad, ymax + pad)
        else:
            ax.text(
                0.5,
                0.5,
                "Waiting for FocusScore...",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        try:
            trigger = float(self.saf_trigger_ratio_var.get())
            ax.axhline(trigger, color="tab:orange", linestyle="--", linewidth=1.0, label="trigger")
            if bool(self.saf_trigger_absolute_var.get()):
                ax.axhline(2.0 - trigger, color="tab:orange", linestyle=":", linewidth=1.0, label="upper")
            ax.legend(loc="best")
        except Exception:
            pass

        self.focus_preview_canvas.draw_idle()

    def select_saf_roi_thread(self):
        self.run_in_thread(self.select_saf_roi)

    def select_saf_roi(self):
        """交互式选择 Focus ROI 并建立参考；仅更新补焦专用截图区域，不影响标定/角度检测。"""
        try:
            wf = self.ensure_workflow()
            # 清除 workflow 停止标志，避免上一轮停止后无法进入 ROI 选择截图
            wf.stop_requested = False
            wf.step9_stop_requested = False
            current_capture_area = self.parse_capture_area()
            cfg = self._make_saf_config()
            loop = SpectrumAutofocusLoop(
                workflow=wf,
                cfg=cfg,
                output_dir=self.saf_output_dir_var.get().strip() or "focus_output",
                on_log=self.log,
                should_stop=lambda: False,
            )
            roi = loop.select_focus_roi_interactively()

            # 将相对 ROI 转换为新的屏幕截图区域，Focus ROI 归一化为 (0,0,w,h)
            new_capture_area, new_focus_roi = (
                SpectrumAutofocusLoop.roi_to_screen_capture_area(
                    current_capture_area, roi
                )
            )

            # 仅同步回光谱补焦专用输入框，不修改主 capture_area_var
            self.saf_capture_area_var.set(",".join(str(v) for v in new_capture_area))
            self.saf_roi_var.set(",".join(str(v) for v in new_focus_roi))

            # 同步 loop 配置；workflow 标定/角度检测使用的主 capture_area 保持不变
            loop.cfg.capture_area = new_capture_area
            loop.cfg.focus_roi = new_focus_roi

            # 清除已建参考，确保启动时按新的 capture_area 重新截图建立参考
            loop._reference_image = None
            loop._reference_ready = False
            loop._metrics_calc = None
            loop._scorer = None

            self.spectrum_autofocus_loop = loop
            self.set_var(
                self.saf_status_var,
                f"ROI 已选择：{new_focus_roi}，补焦截图区域已同步为 {new_capture_area}，启动时将重建参考",
            )
        except Exception as e:
            self.log(f"[光谱补焦] 选择 ROI 失败：{e}")
            self.log(traceback.format_exc())

    def start_spectrum_autofocus_loop_thread(self):
        if self.spectrum_autofocus_thread is not None and self.spectrum_autofocus_thread.is_alive():
            messagebox.showwarning("正在运行", "光谱补焦循环已在运行中。")
            return
        self.spectrum_autofocus_stop_requested = False
        self.spectrum_autofocus_thread = self.run_in_thread(
            self.start_spectrum_autofocus_loop
        )

    def start_spectrum_autofocus_loop(self):
        """启动光谱补焦循环后台线程。"""
        try:
            self.set_var(self.saf_status_var, "光谱补焦循环：运行中")
            wf = self.ensure_workflow()
            # 清除 workflow 停止标志，确保新一轮循环不被旧停止锁拦截
            wf.stop_requested = False
            wf.step9_stop_requested = False
            cfg = self._make_saf_config()

            # 复用已选择 ROI 的 loop 实例，否则新建
            if (
                self.spectrum_autofocus_loop is not None
                and self.spectrum_autofocus_loop._roi_selected
            ):
                loop = self.spectrum_autofocus_loop
                loop.cfg = cfg
                loop.output_dir = Path(
                    self.saf_output_dir_var.get().strip() or "focus_output"
                )
            else:
                loop = SpectrumAutofocusLoop(
                    workflow=wf,
                    cfg=cfg,
                    output_dir=self.saf_output_dir_var.get().strip()
                    or "focus_output",
                    on_log=self.log,
                    should_stop=lambda: self.spectrum_autofocus_stop_requested,
                )
                self.spectrum_autofocus_loop = loop

            loop.should_stop = lambda: self.spectrum_autofocus_stop_requested
            loop.interval_s = float(self.saf_interval_var.get())
            loop.wait_between_spectrum_s = float(
                self.saf_wait_between_spectrum_var.get()
            )
            loop.run(max_cycles=int(self.saf_max_cycles_var.get()))

            self.set_var(self.saf_status_var, "光谱补焦循环：已结束")
        except Exception as e:
            self.log(f"[光谱补焦] 运行失败：{e}")
            self.log(traceback.format_exc())
            self.set_var(self.saf_status_var, "光谱补焦循环：运行失败")
        finally:
            self.spectrum_autofocus_thread = None

    def stop_spectrum_autofocus_loop(self):
        self.spectrum_autofocus_stop_requested = True
        self.set_var(self.saf_status_var, "光谱补焦循环：已请求停止")
        self.log("[光谱补焦] 已请求停止")


# ============================================================
# 主函数
# ============================================================

def main():
    root = tk.Tk()
    app = MeasurementWorkflowGUI(root)

    def on_close():
        if app.workflow is not None and app.workflow.is_measuring:
            ok = messagebox.askyesno("确认退出", "测量流程正在运行，是否停止并退出？")
            if not ok:
                return

        try:
            if app.workflow is not None:
                app.workflow.request_stop()
                app.workflow.close_all()
        except Exception:
            pass

        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
