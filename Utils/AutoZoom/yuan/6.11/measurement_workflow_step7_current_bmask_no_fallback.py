from __future__ import annotations

import sys
import csv
import json
import time
import math
import threading
import traceback
import copy
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog

import numpy as np
import cv2
import matplotlib.pyplot as plt

from pylablib.devices import Newport, Thorlabs
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
from config_angle_repair_fixed import DEFAULT_CONFIG  # 新版：GUI 左侧所有输入项的默认值

def _cfg(name: str, default: Any) -> Any:
    return DEFAULT_CONFIG.get(name, default)

# ============================================================
# 光谱仪/TCP 禁用配置
# ============================================================
# 按当前需求：
#   1) 不再启动/等待/关闭 LabVIEW TCP；
#   2) 完整测量中不再真实采集光谱数据；
#   3) 保存数据里的原始峰值、处理后峰值、拟合峰值统一用 1000 代替。
SPECTROMETER_TCP_DISABLED = True
DUMMY_SPECTRUM_PEAK = 1000.0

from control.illumination_relay import IlluminationRelay
from control.signal_generator_rigol import RigolDG4062Controller
# from control.labview_tcp_server import LabVIEWTCPServer  # 已按需求注释：不再连接/采集光谱仪 TCP

from vision.screen_capture import CaptureArea
try:
    from vision.screen_capture import FixedRegionScreenCapture
except Exception:
    FixedRegionScreenCapture = None

from logic.angle_detect import ScreenAngleDetector
try:
    # 推荐使用 strict_c 版本：完整测量显式指定 C 后，不允许外部模块回退旧 C。
    from logic.actual_nano_boundary_following_sam2VideoAB_xyStep_detailed_pathfix_strict_c import RuntimeConfig as RuleABRuntimeConfig
    from logic.actual_nano_boundary_following_sam2VideoAB_xyStep_detailed_pathfix_strict_c import ActualNanoBoundaryFollower
except Exception:
    # 兼容旧文件名。若完整测量仍加载旧 C，请把 strict_c 文件复制到 logic 目录。
    from logic.actual_nano_boundary_following_sam2VideoAB_xyStep_detailed_pathfix import RuntimeConfig as RuleABRuntimeConfig
    from logic.actual_nano_boundary_following_sam2VideoAB_xyStep_detailed_pathfix import ActualNanoBoundaryFollower
from logic.rule_ac_fixed import RuleACConfig, RuleACOverlapController


@dataclass
class MeasurementConfig:
    max_cycles: int = int(_cfg("max_cycles", 10))

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

    light_port: str = str(_cfg("light_port", "COM17"))

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

    angle_model_path: str = str(_cfg("angle_model_path", r"D:/desktop/train/model/best_wan12.2.pt"))
    capture_area: Tuple[int, int, int, int] = tuple(_cfg("capture_area", (116, 98, 1112, 886)))  # type: ignore
    angle_output_dir: str = str(_cfg("angle_output_dir", "outputs/captured_frames"))
    angle_num: int = int(_cfg("angle_num", 0))              # 传给 logic.angle_detect.ScreenAngleDetector 的 num
    angle_cw: int = int(_cfg("angle_cw", 0))               # 传给 logic.angle_detect.ScreenAngleDetector 的 cw

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

    # TCP 原始数据处理参数
    # 第一步：删除高于该阈值的原始数据点；第二步：对剩余数据做 5 点中值滤波
    raw_remove_above: float = float(_cfg("raw_remove_above", 3000.0))
    median_filter_window: int = int(_cfg("median_filter_window", 5))

    x_axis_xlsx_path: str = str(_cfg("x_axis_xlsx_path", "516中心波长(1).xlsx"))

    enable_delta_w_judge: bool = bool(_cfg("enable_delta_w_judge", False))
    delta_w_threshold: float = float(_cfg("delta_w_threshold", 0.0))
    stop_when_delta_w_not_enough: bool = bool(_cfg("stop_when_delta_w_not_enough", False))

    # A 推动 B：调用 logic/rule_ab.py，每一步后检测 B 对边角度。
    rule_ab_enable_stage: bool = bool(_cfg("rule_ab_enable_stage", True))
    rule_ab_max_steps: int = int(_cfg("rule_ab_max_steps", 15))
    rule_ab_angle_delta_min_deg: float = float(_cfg("rule_ab_angle_delta_min_deg", 3.5))
    rule_ab_angle_delta_max_deg: float = float(_cfg("rule_ab_angle_delta_max_deg", 6.0))

    # Step7 角度修复：
    # 当前检测角度相对 Step1 推动前 baseline 的变化 > rule_ab_angle_delta_max_deg（默认 6°）时，
    # 判定该检测角度不合适，必须用当前 B mask 的全边候选进行修复。
    # 修复角度必须落在上一帧“确定角度”的 ±rule_ab_angle_repair_accept_tolerance_deg 内。
    rule_ab_enable_angle_repair: bool = bool(_cfg("rule_ab_enable_angle_repair", True))
    rule_ab_angle_repair_jump_threshold_deg: float = float(_cfg("rule_ab_angle_repair_jump_threshold_deg", 10.0))  # 仅保留日志/兼容，不作为当前修复触发条件
    rule_ab_angle_repair_accept_tolerance_deg: float = float(_cfg("rule_ab_angle_repair_accept_tolerance_deg", 1.0))
    rule_ab_angle_repair_min_edge_length_px: float = float(_cfg("rule_ab_angle_repair_min_edge_length_px", 8.0))

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

    # A/B mask 运行时后处理：当前版本不再约束 B，也不再回退上一帧 B。
    # 当前帧 Bmask 会被直接用于 Step7 路线运动、角度检测与角度修复。
    # 这里只保留 A = A - dilate(B_current)，避免 A mask 把当前帧 B 包进去。
    rule_ab_enable_ab_mask_guard: bool = bool(_cfg("rule_ab_enable_ab_mask_guard", True))
    rule_ab_b_area_ratio_min: float = float(_cfg("rule_ab_b_area_ratio_min", 0.75))
    rule_ab_b_area_ratio_max: float = float(_cfg("rule_ab_b_area_ratio_max", 1.35))
    rule_ab_b_center_jump_max_px: float = float(_cfg("rule_ab_b_center_jump_max_px", 40.0))
    rule_ab_b_angle_jump_max_deg: float = float(_cfg("rule_ab_b_angle_jump_max_deg", 12.0))
    rule_ab_a_exclude_b_dilate_px: int = int(_cfg("rule_ab_a_exclude_b_dilate_px", 5))
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

    # SAM2 video predictor：用于推动过程中的 A/B 连续帧跟踪。
    # 适合 A/B 同时发生平移和自转；失败时回退到上一帧局部模板 + ImagePredictor。
    rule_ab_use_sam2_video_tracking: bool = bool(_cfg("rule_ab_use_sam2_video_tracking", True))
    rule_ab_sam2_video_tracking_mode: str = str(_cfg("rule_ab_sam2_video_tracking_mode", "two_frame_anchor"))
    rule_ab_sam2_video_temp_dir: str = str(_cfg("rule_ab_sam2_video_temp_dir", "outputs/sam2_video_tracking_tmp"))
    rule_ab_sam2_video_prompt_max_points: int = int(_cfg("rule_ab_sam2_video_prompt_max_points", 5))
    rule_ab_sam2_video_area_ratio_min: float = float(_cfg("rule_ab_sam2_video_area_ratio_min", 0.35))
    rule_ab_sam2_video_area_ratio_max: float = float(_cfg("rule_ab_sam2_video_area_ratio_max", 2.80))
    rule_ab_sam2_video_center_jump_max_px: float = float(_cfg("rule_ab_sam2_video_center_jump_max_px", 220.0))
    rule_ab_sam2_video_fallback_to_template: bool = bool(_cfg("rule_ab_sam2_video_fallback_to_template", True))

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
        return state

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def missing_items(self) -> List[str]:
        missing: List[str] = []

        # 角度检测 B 与 RuleAB-B 现在做双向兜底：
        #   - 点了“1 标定角度B/边”但没点 RuleAB-B：RuleAB-B 复用角度 B；
        #   - 点了 RuleAB-B 但没点“1 标定角度B/边”：角度 B 复用 RuleAB-B。
        # 因此这里不再把“必须点击 1 标定角度B/边中的 B 点”作为硬条件，
        # 只要求至少有一套 B 点，避免 C/ABC 初始化因为 B 缺失而不稳定。
        has_any_b_points = bool(self.angle_b_positive_points or self.rule_ab_b_positive_points)
        if not has_any_b_points:
            missing.append("B mask 正点（可通过 1 标定角度B/边 或 单独 RuleAB-B 标定提供）")
        if self.angle_edge_index < 0 and not self.angle_edge_name:
            missing.append("角度检测：检测边编号/名称")
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


class MeasurementWorkflow:
    def __init__(self, cfg: MeasurementConfig, on_log=None, on_update=None):
        self.cfg = cfg
        self.on_log = on_log
        self.on_update = on_update

        self.light: Optional[IlluminationRelay] = None
        self.signal_generator: Optional[RigolDG4062Controller] = None
        self.laser_stage: Optional[Newport.Picomotor8742] = None
        self.angle_module: Optional[ScreenAngleDetector] = None
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

        self.save_threads: List[threading.Thread] = []
        self.save_threads_lock = threading.Lock()
        self.save_io_lock = threading.Lock()

        self.run_session_dir: Optional[Path] = None
        self.run_session_name: Optional[str] = None

        self.previous_cycle_angle: Optional[float] = None

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
        }

        self.plot_points: List[Dict[str, Any]] = []
        self.context["plot_points"] = self.plot_points

        self.output_root = Path(cfg.save_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

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

            contours, _ = cv2.findContours((mask_bool.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
            f"angle_edge=({state.angle_edge_index}, {state.angle_edge_name}), "
            f"A点={len(state.rule_ab_a_positive_points)}/{len(state.rule_ab_a_negative_points)}, "
            f"C点={len(state.global_c_positive_points)}/{len(state.global_c_negative_points)}, "
            f"Step9 target=({state.step9_target_x_px:.1f}, {state.step9_target_y_px:.1f}), "
            f"HSV=({state.step9_color_h}, {state.step9_color_s}, {state.step9_color_v})"
        )
        return state

    def _capture_current_rule_ab_frame(self, output_dir: Path) -> np.ndarray:
        """
        截取当前固定屏幕区域，返回 RGB 图像。

        说明：
            Step9 单独测试需要先截图给 SAM2 分割 B。
            之前这个函数只写在 GUI 类里，MeasurementWorkflow.prepare_step9_b_segmentation()
            调用不到，所以会报：
                'MeasurementWorkflow' object has no attribute '_capture_current_rule_ab_frame'

            现在把同名函数补到 MeasurementWorkflow 类内部。
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        capture_area = tuple(int(v) for v in self.cfg.capture_area)

        if FixedRegionScreenCapture is not None:
            capturer = FixedRegionScreenCapture(
                capture_area=capture_area,
                output_dir=output_dir / "step9_capture_tmp",
                save_image=False,
            )
            frame = capturer.capture()
            return np.asarray(frame).astype(np.uint8)

        # 兜底：不用 vision.screen_capture 时，使用 PIL.ImageGrab。
        try:
            from PIL import ImageGrab
        except Exception as e:
            raise RuntimeError(
                "无法导入 FixedRegionScreenCapture，也无法导入 PIL.ImageGrab，不能截图。"
            ) from e

        left, top, width, height = capture_area
        img = ImageGrab.grab(
            bbox=(left, top, left + width, top + height)
        ).convert("RGB")
        return np.array(img, dtype=np.uint8)

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
        if self.light is not None:
            self.log("[照明光] 已连接")
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

        self.log("[激光开关] 正在连接 Newport 8743-CL / Picomotor")
        self.laser_stage = Newport.Picomotor8742()

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
        """
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
        if self.angle_module is not None:
            self.log("[角度检测] 已初始化")
            return

        if not self.cfg.angle_model_path:
            raise ValueError("未设置 angle_model_path，请在 GUI 中选择 YOLO .pt 模型路径")

        self.log("[角度检测] 正在初始化 ScreenAngleDetector 角度检测模块")
        self.log(f"[角度检测] model_path={self.cfg.angle_model_path}")
        self.log(f"[角度检测] capture_area={self.cfg.capture_area}")
        self.log(
            f"[角度检测] num={self.cfg.angle_num}, "
            f"cw={self.cfg.angle_cw}"
        )

        self.angle_module = ScreenAngleDetector(
            model_path=self.cfg.angle_model_path,
            capture_area=self.cfg.capture_area,
            output_dir=self.cfg.angle_output_dir,
            save_image=True,
            device=None,
            num=int(self.cfg.angle_num),
            cw=int(self.cfg.angle_cw),
            use_second_detect=False,
        )

        # 如果已经加载完整标定，把角度 B 点/检测边也写入角度模块。
        state = self._get_loaded_calibration_state()
        if state is not None:
            self._inject_calibration_into_angle_module(self.angle_module, state)
            self.log(
                "[完整测量标定] 已向角度模块注入 B 点/检测边："
                f"B+={len(state.angle_b_positive_points)}, B-={len(state.angle_b_negative_points)}, "
                f"edge={state.angle_edge_index}/{state.angle_edge_name}"
            )

        self.log("[角度检测] 初始化完成；已调用 logic.angle_detect.ScreenAngleDetector，YOLO .pt 模型后续循环不会重复加载")

    def start_tcp_server(self):
        """
        光谱仪 TCP 已按需求禁用。

        保留这个函数是为了不破坏 GUI 按钮和旧流程调用；
        调用时只更新状态，不创建 LabVIEWTCPServer，不占用端口。
        """
        self.tcp_server = None
        self.context["tcp_started"] = False
        self.context["labview_ready"] = True
        self.log("[TCP] 光谱仪 TCP 连接已注释/禁用：不启动 Server，不等待 LabVIEW。")
        self.notify_update()
        return {"ok": True, "disabled": True, "message": "spectrometer_tcp_disabled"}

    def wait_labview_ready(self):
        """光谱仪 TCP 已禁用；直接视为可继续完整测量。"""
        self.tcp_server = None
        self.context["tcp_started"] = False
        self.context["labview_ready"] = True
        self.log("[TCP] 光谱仪 TCP 已禁用：跳过 LabVIEW READY 等待。")
        self.notify_update()
        return {"ok": True, "disabled": True, "message": "spectrometer_tcp_disabled"}

    # --------------------------------------------------------
    # 设备基础动作
    # --------------------------------------------------------

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

    def _run_angle_detector_once_raw(self, use_second_detect: bool) -> Dict[str, Any]:
        """
        调用 ScreenAngleDetector 执行一次检测。

        use_second_detect=False：基础检测。
        use_second_detect=True ：临时启用 ScreenAngleDetector 内部的二次检测逻辑。
        """
        if self.angle_module is None:
            raise RuntimeError("角度检测模块未初始化")

        if hasattr(self.angle_module, "use_second_detect"):
            self.angle_module.use_second_detect = bool(use_second_detect)

        if hasattr(self.angle_module, "detect_once_detail"):
            result = self.angle_module.detect_once_detail()
        elif hasattr(self.angle_module, "capture_and_detect_once"):
            result = self.angle_module.capture_and_detect_once()
        else:
            raise RuntimeError(
                "角度检测模块接口不匹配：需要 detect_once_detail() "
                "或 capture_and_detect_once()"
            )

        return result

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
        """
        单次角度检测。

        检测流程：
            1. 先强制 use_second_detect=False，执行一次基础角度检测；
            2. 如果第一次角度落在 0~20° 或 90~110°，则临时设置
               use_second_detect=True，再调用 ScreenAngleDetector 检测一次；
            3. 如果第二次检测成功，使用第二次检测结果；如果第二次失败，保留第一次结果；
            4. 检测结束后重新把 use_second_detect 置回 False。

        这样第二个代码仍然没有 GUI 里的全局“启用二次检测”复选框，
        但具备你需要的“特定角度范围自动旋转/二次检测”能力。
        """
        if self.angle_module is None:
            raise RuntimeError("角度检测模块未初始化")

        self.log(f"========== 角度检测：{label} ==========")

        # 运行时同步 num/cw，不重新加载模型。
        if hasattr(self.angle_module, "num"):
            self.angle_module.num = int(self.cfg.angle_num)
        if hasattr(self.angle_module, "cw"):
            self.angle_module.cw = int(self.cfg.angle_cw)

        self.log(
            f"[角度检测-{label}] 使用参数："
            f"num={self.cfg.angle_num}, "
            f"cw={self.cfg.angle_cw}, "
            f"conditional_second_detect={getattr(self.cfg, 'enable_conditional_second_detect', True)}"
        )

        # 第一次：基础检测，明确关闭 ScreenAngleDetector 内部二次检测。
        first_result = self._normalize_angle_result(
            self._run_angle_detector_once_raw(use_second_detect=False)
        )
        first_angle = first_result.get("angle_deg") if first_result.get("ok", False) else None

        final_result = first_result
        second_triggered = self._should_run_conditional_second_detect(first_angle)

        if second_triggered:
            self.log(
                f"[角度检测-{label}] 第一次角度={first_angle}，"
                "落入条件二次检测范围，临时启用 use_second_detect=True 再检测一次"
            )
            try:
                second_result = self._normalize_angle_result(
                    self._run_angle_detector_once_raw(use_second_detect=True)
                )
                second_result["conditional_second_detect_triggered"] = True
                second_result["first_angle_deg"] = first_angle
                second_result["first_result"] = self._json_safe(first_result)

                if second_result.get("ok", False) and second_result.get("angle_deg") is not None:
                    final_result = second_result
                    self.log(
                        f"[角度检测-{label}] 条件二次检测成功："
                        f"first={first_angle}, second={second_result.get('angle_deg')}"
                    )
                else:
                    first_result["conditional_second_detect_triggered"] = True
                    first_result["conditional_second_detect_failed"] = True
                    first_result["second_result"] = self._json_safe(second_result)
                    final_result = first_result
                    self.log(
                        f"[角度检测-{label}] 条件二次检测失败，保留第一次结果："
                        f"first={first_angle}, reason={second_result.get('reason')}"
                    )
            finally:
                # 第二个代码的默认状态仍然保持“固定关闭二次检测”。
                if hasattr(self.angle_module, "use_second_detect"):
                    self.angle_module.use_second_detect = False
        else:
            first_result["conditional_second_detect_triggered"] = False
            if first_angle is not None:
                self.log(f"[角度检测-{label}] 第一次角度={first_angle}，未触发条件二次检测")

        # 统一补齐字段，保证后续代码可以直接 result.get(...)
        final_result = self._normalize_angle_result(final_result)
        final_result.setdefault("conditional_second_detect_triggered", bool(second_triggered))

        if not final_result.get("ok", False):
            self.log(f"[角度检测-{label}] 检测失败，但允许流程继续")
            self.log(f"[角度检测-{label}] reason={final_result.get('reason')}")
            self.log(f"[角度检测-{label}] image_path={final_result.get('image_path')}")

            if not allow_fail:
                raise RuntimeError(f"{label} 角度检测失败：{final_result}")

            return final_result

        self.log(f"[角度检测-{label}] image_path={final_result.get('image_path')}")
        self.log(f"[角度检测-{label}] angle_deg={final_result.get('angle_deg')}")
        self.log(f"[角度检测-{label}] raw={final_result.get('angle_deg_raw')}")
        self.log(f"[角度检测-{label}] conditional_second_detect_triggered={final_result.get('conditional_second_detect_triggered')}")
        self.log(f"[角度检测-{label}] center={final_result.get('center')} area_px={final_result.get('area_px')}")

        return final_result

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
            self.cfg.rule_ab_enable_stage = True
            self._force_cfg_to_strict_full_calibration_c(reason="ensure_rule_ab_follower")

        cfg = RuleABRuntimeConfig(
            capture_area=self.cfg.capture_area,
            output_dir=str(self.output_root / "rule_ab_from_measurement"),
            sam2_device="cuda",
            enable_stage=bool(self.cfg.rule_ab_enable_stage),
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
            a_use_temporal_mask_selection=bool(self.cfg.rule_ab_a_use_temporal_mask_selection),
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
            b_use_temporal_mask_selection=bool(self.cfg.rule_ab_b_use_temporal_mask_selection),
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
            use_sam2_video_tracking=bool(self.cfg.rule_ab_use_sam2_video_tracking),
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

        if state is not None:
            self._call_initialize_abc_with_loaded_calibration(self.rule_ab_follower, state)
        else:
            self.rule_ab_follower.initialize_abc_with_first_frame()

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
            if target_attr == "sam2_video_temp_dir":
                value = Path(str(value))
            self._safe_setattr(fc, target_attr, value)
            if seg is not None:
                self._safe_setattr(seg, target_attr, value)

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
            min_edge_length_px = float(getattr(self.cfg, "rule_ab_angle_repair_min_edge_length_px", 8.0))
        min_len = max(1.0, float(min_edge_length_px))

        m_u8 = (m.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        不再固定使用 GUI num 指定的旧边，也不再只在角度跳变修复时才查看 B mask。
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
                cv2.putText(canvas, "selected B edge = nearest to A center", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

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

    def detect_ab_nearest_b_edge_angle_once(
        self,
        follower: Any,
        label: str,
        allow_fail: bool = True,
    ) -> Dict[str, Any]:
        """
        用当前帧重新分割 A/B，并选择 A 中心距离 B 边最近的那条边计算角度。

        该函数用于替代 Step7 中原来只调用 ScreenAngleDetector 的角度逻辑，
        使每一次角度判断都具备可追溯的 A/B mask 保存结果。
        """
        self.log(f"========== AB最近边角度检测：{label} ==========")
        try:
            if follower is None:
                raise RuntimeError("RuleAB follower 为空，无法做 A/B 分割角度检测")
            image_rgb, scene = follower.capture_and_build_scene()
            try:
                ab_guard_info = self._postprocess_scene_ab_masks(
                    follower=follower,
                    scene=scene,
                    tag=f"ab_angle_{label}",
                )
            except Exception as e:
                ab_guard_info = {"ok": False, "reason": f"ab_guard_failed: {e}"}

            a_obj = getattr(scene, "a", None)
            b_obj = getattr(scene, "b", None)
            a_mask = self._extract_mask_from_scene_object(a_obj)
            b_mask = self._extract_mask_from_scene_object(b_obj)
            a_center = self._get_scene_a_center_xy(scene)
            b_center = self._mask_center_xy(b_mask) if b_mask is not None else None

            if a_center is None:
                raise RuntimeError("当前帧 A center 为空")
            if b_mask is None or not np.any(np.asarray(b_mask).astype(bool)):
                raise RuntimeError("当前帧 B mask 为空")

            candidates = self._bmask_edge_angle_candidates_from_mask(b_mask)
            selected = self._select_b_edge_nearest_to_a_center(candidates, a_center)
            saved = self._save_ab_nearest_edge_angle_debug(
                image_rgb=image_rgb,
                a_mask=a_mask,
                b_mask=b_mask,
                a_center=a_center,
                b_center=b_center,
                candidates=candidates,
                selected_edge=selected,
                label=label,
                extra_info={"ab_guard": ab_guard_info},
            )
            if selected is None:
                raise RuntimeError("未能从当前 B mask 得到可用候选边")

            angle = float(selected["angle_deg"])
            result = {
                "ok": True,
                "angle_deg": angle,
                "angle_deg_raw": angle,
                "label": label,
                "edge_selection_mode": "nearest_b_edge_to_a_center",
                "a_center": [float(a_center[0]), float(a_center[1])],
                "center": [float(a_center[0]), float(a_center[1])],
                "b_center": [float(b_center[0]), float(b_center[1])] if b_center is not None else None,
                "selected_b_edge": self._json_safe(selected),
                "candidate_count": len(candidates),
                "candidate_edges": self._json_safe(candidates),
                "ab_guard": self._json_safe(ab_guard_info),
                "a_mask_area_px": int(np.count_nonzero(np.asarray(a_mask).astype(bool))) if a_mask is not None else None,
                "b_mask_area_px": int(np.count_nonzero(np.asarray(b_mask).astype(bool))) if b_mask is not None else None,
                "image_path": saved.get("overlay_path"),
                "overlay_path": saved.get("overlay_path"),
                "a_mask_path": saved.get("a_mask_path"),
                "b_mask_path": saved.get("b_mask_path"),
                "meta_path": saved.get("meta_path"),
                "reason": "ok",
            }
            self.log(
                f"[AB最近边角度-{label}] angle={angle:.6f}°, "
                f"A=({a_center[0]:.1f},{a_center[1]:.1f}), "
                f"selected_edge_dist={float(selected.get('distance_to_a_center_px', -1.0)):.3f}px, "
                f"candidates={len(candidates)}, overlay={saved.get('overlay_path')}"
            )
            return result
        except Exception as e:
            msg = f"AB最近边角度检测失败：{e}"
            self.log(f"[AB最近边角度-{label}] {msg}")
            if not allow_fail:
                raise
            return {
                "ok": False,
                "angle_deg": None,
                "angle_deg_raw": None,
                "label": label,
                "edge_selection_mode": "nearest_b_edge_to_a_center",
                "reason": msg,
                "error": str(e),
            }

    def _get_rule_ab_current_bmask_angle_candidates(self, follower: Any) -> List[Dict[str, Any]]:
        """
        用当前 RuleAB/SAM2 跟踪得到的 B mask 计算所有候选边角度。

        注意：这里不弹窗、不重新标定。B 的初始化仍来自“1 标定角度B/边”
        或 RuleAB-B 标定写入的完整标定包；本函数只读取当前跟踪出的 B mask。
        """
        if follower is None:
            return []

        # 优先从当前 follower 再抓一帧 scene，保证拿到当前 B mask。
        try:
            if hasattr(follower, "capture_and_build_scene"):
                _image_rgb, scene = follower.capture_and_build_scene()
                b_mask = self._extract_mask_from_scene_object(getattr(scene, "b", None))
                if b_mask is not None:
                    cands = self._bmask_edge_angle_candidates_from_mask(b_mask)
                    if cands:
                        return cands
        except Exception as e:
            self.log(f"[RuleAB-角度修复] 从当前 RuleAB scene 获取 B mask 失败，尝试缓存字段：{e}")

        # 兼容外部模块可能保存 last/current scene。
        for attr in ("last_scene", "current_scene", "scene"):
            try:
                scene = getattr(follower, attr, None)
                if scene is None:
                    continue
                b_mask = self._extract_mask_from_scene_object(getattr(scene, "b", None))
                if b_mask is not None:
                    cands = self._bmask_edge_angle_candidates_from_mask(b_mask)
                    if cands:
                        return cands
            except Exception:
                pass

        # 兼容 segmenter/follower 直接缓存 B mask 的字段。
        for obj in (follower, getattr(follower, "abc_segmenter", None), getattr(follower, "segmenter", None)):
            if obj is None:
                continue
            for attr in ("last_b_mask", "current_b_mask", "b_mask", "last_mask_b"):
                try:
                    b_mask = getattr(obj, attr, None)
                    if b_mask is not None:
                        cands = self._bmask_edge_angle_candidates_from_mask(np.asarray(b_mask))
                        if cands:
                            return cands
                except Exception:
                    pass

        return []

    def _repair_rule_ab_angle_if_jump(
        self,
        raw_angle: Optional[float],
        previous_angle: Optional[float],
        baseline_angle: Optional[float],
        follower: Any,
        angle_result: Dict[str, Any],
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """
        Step7 角度修复。

        当前固定逻辑：
            1) Step7 每推动一次，就检测一次 A/B 最近边角度 raw_angle；
            2) raw_angle 先和“上一帧确定角度 previous_angle”比较；
            3) 如果 raw_jump <= rule_ab_angle_delta_max_deg（默认 6°），认为本帧检测角度可信，直接使用 raw_angle；
            4) 如果 raw_jump > 6°，认为最近边选错或边发生跳变，必须先修复；
            5) 修复从当前 B mask 的全部候选边角度中选择：
                  - 优先选择距离 previous_angle 在 ±1° 内的候选，认为是成功修复；
                  - 如果没有 ±1° 内候选，则选择距离 previous_angle 最近的候选，标记为 out_of_tolerance；
            6) raw_jump > 6° 后，raw_angle 只用于触发修复，不能直接决定是否关闭激光。
        """
        max_jump = float(getattr(self.cfg, "rule_ab_angle_delta_max_deg", 6.0))
        accept_tol = float(getattr(self.cfg, "rule_ab_angle_repair_accept_tolerance_deg", 1.0))

        info: Dict[str, Any] = {
            "enabled": bool(getattr(self.cfg, "rule_ab_enable_angle_repair", True)),
            "triggered": False,
            "trigger_reason": "none",
            "used_repaired_angle": False,
            "repair_failed": False,
            "repair_out_of_tolerance": False,
            "raw_angle": raw_angle,
            "previous_confirmed_angle": previous_angle,
            "baseline_angle": baseline_angle,
            "max_jump_from_previous_for_repair_deg": max_jump,
            "accept_tolerance_to_previous_deg": accept_tol,
            "raw_delta_from_baseline": None,
            "raw_jump_from_previous_deg": None,
            "candidate_count": 0,
            "candidate_angles": [],
            "candidate_diffs_to_previous": [],
            "eligible_candidate_angles_within_tolerance": [],
            "nearest_candidate_angle": None,
            "nearest_candidate_diff_to_previous_deg": None,
            "repaired_angle": None,
            "repaired_delta_from_baseline": None,
            "repaired_jump_from_previous_deg": None,
            "reason": "not_checked",
        }

        if not info["enabled"]:
            info["reason"] = "disabled_use_raw_angle"
            return raw_angle, info
        if raw_angle is None:
            info["reason"] = "raw_angle_none"
            return raw_angle, info

        raw_delta = None
        if baseline_angle is not None:
            try:
                raw_delta = self.angle_diff_deg(float(raw_angle), float(baseline_angle))
            except Exception:
                raw_delta = None
        info["raw_delta_from_baseline"] = raw_delta

        raw_jump = None
        if previous_angle is not None:
            try:
                raw_jump = self.angle_diff_deg(float(raw_angle), float(previous_angle))
            except Exception:
                raw_jump = None
        info["raw_jump_from_previous_deg"] = raw_jump

        # 没有上一帧确定角度时，无法判断“边跳变”，只能把当前检测角度作为第一帧确定角度。
        if previous_angle is None or raw_jump is None:
            info["reason"] = "previous_confirmed_angle_none_or_jump_none_use_raw_angle"
            try:
                angle_result["angle_repair"] = self._json_safe(info)
                angle_result["angle_deg_raw_before_repair"] = float(raw_angle)
                angle_result["angle_deg"] = float(raw_angle)
                angle_result["used_repaired_angle"] = False
            except Exception:
                pass
            return raw_angle, info

        # 只有“当前检测角度相对上一帧确定角度跳变超过 6°”才修复。
        if raw_jump <= max_jump:
            info["reason"] = "raw_jump_within_max_use_raw_angle"
            self.log(
                f"[RuleAB-角度修复] raw_angle={raw_angle:.6f} 相对上一帧确定角度={previous_angle:.6f} "
                f"跳变 raw_jump={raw_jump:.3f}° <= {max_jump:.3f}°，检测角度可信，不执行修复。"
            )
            try:
                angle_result["angle_repair"] = self._json_safe(info)
                angle_result["angle_deg_raw_before_repair"] = float(raw_angle)
                angle_result["angle_deg"] = float(raw_angle)
                angle_result["used_repaired_angle"] = False
            except Exception:
                pass
            return raw_angle, info

        info["triggered"] = True
        info["trigger_reason"] = "raw_jump_from_previous_above_max"
        self.log(
            f"[RuleAB-角度修复] raw_angle={raw_angle:.6f} 相对上一帧确定角度={previous_angle:.6f} "
            f"跳变 raw_jump={raw_jump:.3f}° > {max_jump:.3f}°，判定检测边可能选错，开始基于 B mask 全边候选修复；"
            f"优先接受范围=±{accept_tol:.3f}°。"
        )

        candidates = self._get_rule_ab_current_bmask_angle_candidates(follower)
        info["candidate_count"] = len(candidates)
        info["candidate_angles"] = [float(c["angle_deg"]) for c in candidates]

        if not candidates:
            info["repair_failed"] = True
            info["reason"] = "no_bmask_edge_candidates_no_close"
            self.log("[RuleAB-角度修复] 未能从当前 B mask 得到候选边角度；本帧不使用异常 raw_angle 关闭激光，继续推动/检测。")
            return None, info

        enriched = []
        for c in candidates:
            try:
                a = float(c["angle_deg"])
                d = self.angle_diff_deg(a, float(previous_angle))
                enriched.append((float(d), a, c))
            except Exception:
                continue
        enriched.sort(key=lambda x: x[0])
        info["candidate_diffs_to_previous"] = [float(x[0]) for x in enriched]

        if not enriched:
            info["repair_failed"] = True
            info["reason"] = "candidate_angles_invalid_no_close"
            self.log("[RuleAB-角度修复] B mask 候选边角度无有效数值；本帧不关闭激光，继续推动/检测。")
            return None, info

        eligible = [x for x in enriched if float(x[0]) <= accept_tol]
        info["eligible_candidate_angles_within_tolerance"] = [float(x[1]) for x in eligible]

        if eligible:
            best_diff, repaired, best = eligible[0]
            reason = "raw_jump_above_max_repaired_by_candidate_within_1deg_to_previous_confirmed_angle"
            out_of_tol = False
        else:
            # 没有 ±1° 内候选时，仍然给出一个“最近候选修复角度”。
            # 后续关闭逻辑会检查 repaired_jump 是否仍 >6°；若仍 >6°，按用户要求立即关闭激光。
            best_diff, repaired, best = enriched[0]
            reason = "raw_jump_above_max_nearest_candidate_out_of_1deg_tolerance"
            out_of_tol = True

        repaired = float(repaired)
        repaired_jump = float(best_diff)
        repaired_delta = None
        if baseline_angle is not None:
            try:
                repaired_delta = self.angle_diff_deg(repaired, float(baseline_angle))
            except Exception:
                repaired_delta = None

        info.update({
            "used_repaired_angle": True,
            "repair_failed": False,
            "repair_out_of_tolerance": bool(out_of_tol),
            "repaired_angle": repaired,
            "repaired_jump_from_previous_deg": repaired_jump,
            "repaired_delta_from_baseline": repaired_delta,
            "nearest_candidate_angle": repaired,
            "nearest_candidate_diff_to_previous_deg": repaired_jump,
            "selected_edge": self._json_safe(best),
            "reason": reason,
        })

        try:
            angle_result["angle_repair"] = self._json_safe(info)
            angle_result["angle_deg_raw_before_repair"] = float(raw_angle)
            angle_result["angle_deg_repaired"] = repaired
            angle_result["angle_deg"] = repaired
            angle_result["used_repaired_angle"] = True
            angle_result["repair_out_of_tolerance"] = bool(out_of_tol)
        except Exception:
            pass

        if out_of_tol:
            self.log(
                f"[RuleAB-角度修复] 候选角度={info['candidate_angles']}；没有 ±{accept_tol:.3f}° 内候选，"
                f"选择最近候选 repaired_angle={repaired:.6f}°，相对上一帧确定角度 repaired_jump={repaired_jump:.3f}°，"
                f"修复后相对Step1基准变化 repaired_delta={repaired_delta}。"
            )
        else:
            self.log(
                f"[RuleAB-角度修复] 候选角度={info['candidate_angles']}；"
                f"±{accept_tol:.3f}° 内候选={info['eligible_candidate_angles_within_tolerance']}；"
                f"选择 repaired_angle={repaired:.6f}°，相对上一帧确定角度差={repaired_jump:.3f}°，"
                f"修复后相对Step1基准变化 repaired_delta={repaired_delta}。"
            )
        return repaired, info

    def run_rule_ab_until_angle_delta(
        self,
        baseline_angle: Optional[float],
        min_delta_deg: Optional[float] = None,
        max_delta_deg: Optional[float] = None,
    ) -> Dict[str, Any]:
        self.log("========== RuleAB：A推动B，检测B对边角度 ==========")

        min_delta = float(
            self.cfg.rule_ab_angle_delta_min_deg
            if min_delta_deg is None else min_delta_deg
        )
        max_delta = float(
            self.cfg.rule_ab_angle_delta_max_deg
            if max_delta_deg is None else max_delta_deg
        )

        follower = self.ensure_rule_ab_follower()
        self.apply_runtime_rule_ab_params_to_follower(follower, reason="step7_start")
        try:
            st = getattr(follower, "stage", None)
            self.log(
                f"[RuleAB] Stage34状态：enable_stage={getattr(follower.cfg, 'enable_stage', None)}, "
                f"stage_is_none={st is None}, "
                f"x_channel={getattr(follower.cfg, 'stage_x_channel', None)}, "
                f"y_channel={getattr(follower.cfg, 'stage_y_channel', None)}, "
                f"step_x={getattr(follower.cfg, 'stage_step_x', None)}, "
                f"step_y={getattr(follower.cfg, 'stage_step_y', None)}"
            )
        except Exception:
            pass

        records: List[Dict[str, Any]] = []
        reached = False
        allow_continue_measurement = False
        stop_reason = "not_started"
        final_angle = None
        final_delta = None
        last_valid_step_angle: Optional[float] = float(baseline_angle) if baseline_angle is not None else None

        step_idx = 0
        while True:
            if self.stop_requested:
                stop_reason = "user_stop_requested"
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
            self.apply_runtime_rule_ab_params_to_follower(follower, reason=f"step7_step_{step_idx + 1}")

            step_idx += 1
            use_route = bool(getattr(self.cfg, "rule_ab_use_c_edge_route", True))
            route_info: Dict[str, Any] = {"route_enabled": use_route}
            if use_route:
                route_points = self._load_or_build_step7_c_edge_route(follower)
                self.log(
                    f"[RuleAB] step={step_idx}: 执行一次 C边沿路线运动；"
                    f"route_points={len(route_points)}，不再调用 rule_ab.run_one_cycle() 的实时策略；"
                    f"目标角度范围=[{min_delta}, {max_delta}]；"
                    f"修复触发条件=相对上一帧确定角度跳变>{max_delta}°，修复优先接受范围=±{getattr(self.cfg, 'rule_ab_angle_repair_accept_tolerance_deg', 1.0)}°"
                )
                ok, route_info = self._run_one_c_edge_route_cycle(follower, step_idx, route_points)
                if not ok:
                    self.log(f"[RuleAB-Route] 路线运动失败：{route_info}")
            else:
                self.log(
                    f"[RuleAB] step={step_idx}: 执行一次 rule_ab.run_one_cycle()；"
                    f"不再受 rule_ab_max_steps 限制；当前目标角度范围=[{min_delta}, {max_delta}]；"
                    f"修复触发条件=相对上一帧确定角度跳变>{max_delta}°，修复优先接受范围=±{getattr(self.cfg, 'rule_ab_angle_repair_accept_tolerance_deg', 1.0)}°"
                )
                ok = follower.run_one_cycle()

            # 每一步运动后，重新分割 A/B；角度边定义为“A中心距离最近的 B 边”。
            angle_result = self.detect_ab_nearest_b_edge_angle_once(
                follower=follower,
                label=f"rule_ab_edge_step_{step_idx}",
                allow_fail=True,
            )

            # 兜底：如果 A/B 分割角度失败，才回退到旧 ScreenAngleDetector，避免直接中断完整测量。
            if not angle_result.get("ok", False):
                self.log(
                    f"[RuleAB] AB最近边角度检测失败，临时回退旧角度检测；"
                    f"reason={angle_result.get('reason')}"
                )
                angle_result = self.detect_angle_once(
                    label=f"rule_ab_edge_step_{step_idx}_fallback_screen",
                    allow_fail=True,
                )

            raw_angle = None
            repaired_info: Dict[str, Any] = {}
            if angle_result.get("ok", False) and angle_result.get("angle_deg") is not None:
                raw_angle = float(angle_result["angle_deg"])

                # 关键修复：即使当前角度来自“A中心最近的B边”，也仍然允许角度修复覆盖最终角度。
                # 否则某一帧最近边误跳变时，阈值判断和保存结果仍会使用 raw_angle，
                # 造成“日志里看到了修复，但最终保存/停光不是修复角度”的问题。
                final_angle, repaired_info = self._repair_rule_ab_angle_if_jump(
                    raw_angle=raw_angle,
                    previous_angle=last_valid_step_angle,
                    baseline_angle=baseline_angle,
                    follower=follower,
                    angle_result=angle_result,
                )
                repair_triggered_now = bool(repaired_info.get("triggered", False)) if isinstance(repaired_info, dict) else False
                repair_used_now = bool(repaired_info.get("used_repaired_angle", False)) if isinstance(repaired_info, dict) else False
                repair_out_of_tolerance_now = bool(repaired_info.get("repair_out_of_tolerance", False)) if isinstance(repaired_info, dict) else False
                repair_failed_now = bool(repair_triggered_now and not repair_used_now)

                # raw_angle 相对上一帧确定角度跳变 >6° 后会进入修复。
                # 若修复失败，本帧 raw_angle 不能作为确定角度，
                # 也不能更新上一帧确定角度；继续沿用 last_valid_step_angle 作为当前显示/保存兜底。
                if final_angle is None:
                    if repair_failed_now and last_valid_step_angle is not None:
                        final_angle = float(last_valid_step_angle)
                    else:
                        final_angle = raw_angle

                try:
                    angle_result["angle_deg_raw"] = float(raw_angle)
                    angle_result["angle_deg"] = float(final_angle)
                    angle_result["angle_repair"] = self._json_safe(repaired_info)
                    angle_result["used_repaired_angle"] = bool(repair_used_now)
                    angle_result["repair_failed"] = bool(repair_failed_now)
                    angle_result["repair_out_of_tolerance"] = bool(repair_out_of_tolerance_now)
                except Exception:
                    pass

                if baseline_angle is not None and final_angle is not None:
                    final_delta = self.angle_diff_deg(float(final_angle), float(baseline_angle))
                else:
                    final_delta = None

                # “上一帧确定角度”只用可信角度更新：
                #   - 未触发修复：raw_angle 相对上一帧没有 >6° 跳变，可信，更新；
                #   - 触发修复且修复角度在上一帧确定角度 ±1° 内：可信，更新；
                #   - 修复失败或最近候选超出 ±1°：不更新，继续保持上一帧确定角度。
                if (not repair_triggered_now) or (repair_used_now and not repair_out_of_tolerance_now):
                    if final_angle is not None:
                        last_valid_step_angle = float(final_angle)
            else:
                final_angle = None
                final_delta = None
                repaired_info = {"enabled": bool(getattr(self.cfg, "rule_ab_enable_angle_repair", True)), "reason": "angle_detect_failed"}

            record = {
                "step": step_idx,
                "rule_ab_ok": ok,
                "angle_deg_raw": raw_angle,
                "angle_deg": final_angle,
                "angle_delta_from_baseline": final_delta,
                "angle_repair": self._json_safe(repaired_info),
                "angle_result": self._json_safe(angle_result),
                "c_edge_route": self._json_safe(route_info),
            }
            records.append(record)

            raw_delta = None
            if raw_angle is not None and baseline_angle is not None:
                try:
                    raw_delta = self.angle_diff_deg(float(raw_angle), float(baseline_angle))
                except Exception:
                    raw_delta = None

            repaired_delta = repaired_info.get("repaired_delta_from_baseline") if isinstance(repaired_info, dict) else None
            if repaired_delta is None and bool(repaired_info.get("used_repaired_angle", False)) and final_delta is not None:
                repaired_delta = final_delta

            # 关闭激光条件（当前明确版）：
            #   1) 每推动一次后先得到 raw_angle；
            #   2) raw_angle 先与上一帧确定角度比较：raw_jump > 6° 时说明检测边可能跳变，必须先修复；
            #   3) 没触发修复：用 raw_angle 相对 Step1 推动前 baseline 的 raw_delta 判断，raw_delta >= 3.5° 立即关闭激光；
            #   4) 触发修复：用 repaired_angle 作为最终角度；
            #        - repaired_delta >= 3.5° 立即关闭激光；
            #        - repaired_delta < 3.5° 继续推动；
            #        - 如果修复后 repaired_jump 相对上一帧确定角度仍 > 6°，也立即关闭激光，避免继续用错误边拖下去；
            #   5) raw_jump > 6° 后，异常 raw_angle 只用于触发修复，不能直接关闭激光。
            close_min_delta = float(min_delta)
            close_target_max_delta = float(max_delta)
            repair_triggered = bool(repaired_info.get("triggered", False)) if isinstance(repaired_info, dict) else False
            repair_used = bool(repaired_info.get("used_repaired_angle", False)) if isinstance(repaired_info, dict) else False
            repair_failed = bool(repair_triggered and not repair_used)
            repair_out_of_tolerance = bool(repaired_info.get("repair_out_of_tolerance", False)) if isinstance(repaired_info, dict) else False
            raw_jump_from_previous = repaired_info.get("raw_jump_from_previous_deg") if isinstance(repaired_info, dict) else None
            repaired_jump_from_previous = repaired_info.get("repaired_jump_from_previous_deg") if isinstance(repaired_info, dict) else None

            close_delta_for_reason = None
            close_source = "none"
            close_hit = False
            close_band = "not_reached"

            if repair_used:
                # 检测角度跳变 >6° 后必须先使用修复角度。
                # 关闭激光首先看 repaired_delta 是否已经相对 Step1 推动前角度达到 3.5°。
                if repaired_delta is not None:
                    close_delta_for_reason = float(repaired_delta)
                elif final_delta is not None:
                    close_delta_for_reason = float(final_delta)

                close_source = "repaired_after_raw_jump_gt6" if repair_triggered else "repaired"
                if close_delta_for_reason is not None and float(close_delta_for_reason) >= close_min_delta:
                    close_hit = True
                    if float(close_delta_for_reason) > close_target_max_delta:
                        close_band = "repaired_delta_above_6_close"
                    else:
                        close_band = "repaired_delta_ge_3p5_close"
                else:
                    close_band = "repaired_delta_below_3p5_continue"

                # 额外保护：如果修复后的候选边相对上一帧确定角度仍然 >6°，说明没有修回原边，按你的要求立即关闭激光。
                try:
                    if repaired_jump_from_previous is not None and float(repaired_jump_from_previous) > close_target_max_delta:
                        close_hit = True
                        close_band = "repaired_jump_still_above_6_close"
                        if close_delta_for_reason is None:
                            close_delta_for_reason = float(repaired_jump_from_previous)
                except Exception:
                    pass
            elif repair_failed:
                # 检测角度跳变 >6° 但没有任何可用候选边时，不能用异常 raw_angle 直接关激光。
                close_delta_for_reason = None
                close_source = "repair_failed_no_raw_close"
                close_hit = False
                close_band = "repair_failed_continue"
            else:
                # 检测角度相对上一帧没有超过 6°，认为检测角度可信；用 raw_delta 相对 Step1 baseline 判断是否关激光。
                if raw_delta is not None:
                    close_delta_for_reason = float(raw_delta)
                elif final_delta is not None:
                    close_delta_for_reason = float(final_delta)
                close_source = "raw_valid"
                close_hit = bool(close_delta_for_reason is not None and float(close_delta_for_reason) >= close_min_delta)
                if close_delta_for_reason is not None:
                    if float(close_delta_for_reason) < close_min_delta:
                        close_band = "raw_delta_below_3p5_continue"
                    elif float(close_delta_for_reason) <= close_target_max_delta:
                        close_band = "raw_delta_ge_3p5_close"
                    else:
                        # 这个分支一般不该出现；若出现，说明 raw 对上一帧没跳变，但相对 Step1 已经很大，也应该关。
                        close_band = "raw_delta_above_6_close"

            try:
                record["angle_delta_raw_from_baseline"] = raw_delta
                record["angle_delta_repaired_from_baseline"] = repaired_delta
                record["close_min_inclusive"] = close_min_delta
                record["target_max_delta"] = close_target_max_delta
                record["close_rule"] = "raw_jump_gt6_repair_first__final_delta_ge3p5_close__repaired_jump_still_gt6_close"
                record["repair_triggered"] = bool(repair_triggered)
                record["repair_used"] = bool(repair_used)
                record["repair_failed"] = bool(repair_failed)
                record["raw_delta_ignored_for_close"] = bool(repair_triggered)
                record["raw_jump_from_previous"] = raw_jump_from_previous
                record["repaired_jump_from_previous"] = repaired_jump_from_previous
                record["repair_out_of_tolerance"] = bool(repair_out_of_tolerance)
                record["close_delta_used"] = close_delta_for_reason
                record["close_hit_raw"] = bool((not repair_triggered) and close_hit and close_source == "raw_valid")
                record["close_hit_repaired"] = bool(repair_used and close_hit and close_source in ("repaired_after_raw_jump_gt6", "repaired"))
                record["close_source"] = close_source
                record["close_band"] = close_band
                record["previous_confirmed_angle_after_step"] = last_valid_step_angle
            except Exception:
                pass

            self.log(
                f"[RuleAB] step={step_idx}, raw_angle={raw_angle}, final_angle={final_angle}, "
                f"raw_delta={raw_delta}, final_delta={final_delta}, repaired_delta={repaired_delta}, "
                f"raw_jump_from_previous={raw_jump_from_previous}, repaired_jump_from_previous={repaired_jump_from_previous}, "
                f"repair_triggered={repair_triggered}, repair_used={repair_used}, repair_failed={repair_failed}, "
                f"repair_out_of_tolerance={repair_out_of_tolerance}, close_source={close_source}, "
                f"close_delta_used={close_delta_for_reason}, close_band={close_band}, "
                f"close_rule=raw_jump>6先修复；检测/修复角度相对Step1基准>=3.5关闭；修复后相对上一帧仍>6也关闭"
            )

            if repair_triggered:
                self.log(
                    f"[RuleAB] 本帧检测角度相对上一帧确定角度 raw_jump={raw_jump_from_previous}° > {close_target_max_delta:.3f}°，"
                    "已判定为检测边可能跳变；raw_angle 只触发修复，不直接参与关闭激光判断。"
                )

            if close_hit:
                reached = True
                allow_continue_measurement = True
                stop_reason = f"{close_source}_{close_band}"

                if close_source == "repaired_after_raw_jump_gt6":
                    self.log(
                        f"[RuleAB] 检测角度相对上一帧确定角度跳变 > {close_target_max_delta:.3f}° 后已完成修复；"
                        f"修复后 close_band={close_band}, close_delta_used={close_delta_for_reason:.3f}°，立即关闭激光。"
                    )
                else:
                    self.log(
                        f"[RuleAB] 检测角度变化进入目标范围："
                        f"source={close_source}, delta={close_delta_for_reason:.3f}°，"
                        f"目标范围=[{close_min_delta:.3f}, {close_target_max_delta:.3f}]；立即关闭激光。"
                    )

                try:
                    if bool(self.context.get("laser_on", False)):
                        self.laser_off()
                        self.log("[RuleAB] 角度关闭条件满足后已立即执行 laser_off()")
                    else:
                        self.log("[RuleAB] 角度关闭条件满足时 context 显示 laser_on=False，跳过 laser_off()")
                except Exception as e:
                    self.log(f"[RuleAB] 角度达到关闭条件后关闭激光失败：{e}")
                break

            if repair_failed:
                stop_reason = "raw_jump_gt6_repair_failed_continue_no_raw_close"
                self.log(
                    "[RuleAB] 本帧 raw_jump > 6° 已触发角度修复，但没有得到可用 B 边候选；"
                    "不使用异常 raw_angle 关闭激光，继续下一步推动/检测。"
                )
            elif repair_used:
                stop_reason = "raw_jump_gt6_repaired_delta_below_3p5_continue"
                self.log(
                    f"[RuleAB] raw_jump > 6° 后已得到修复角度；close_band={close_band}, "
                    f"repaired_delta={close_delta_for_reason}，仍未满足关闭条件；继续下一步推动/检测。"
                )
            elif close_delta_for_reason is not None:
                stop_reason = "raw_delta_not_in_close_range_continue"
                self.log(
                    f"[RuleAB] 当前检测角度变化 {close_delta_for_reason:.3f}° 未进入关闭条件："
                    f"需要 >= {close_min_delta:.3f}°；继续下一步推动/检测。"
                )

            if not ok:
                reached = False
                allow_continue_measurement = False
                stop_reason = "rule_ab_returned_false"
                self.log("[RuleAB] rule_ab 返回 False，停止 A推动B 模块；该情况仍会停止完整测量。")
                break

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
            "target_max": max_delta,
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
            contours, _ = cv2.findContours((m.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        contours, _ = cv2.findContours((m.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
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
    ) -> str:
        """
        保存 Step7 当前帧的 A 路线叠加图。

        图像来源是当前截取帧 image_rgb；叠加内容包括：
            - 紫色：完整 A 运动路线；
            - 绿色：路线起点；
            - 红色：当前 Step7 目标路线点；
            - 青色：当前 A 中心；
            - 黄色线：A 当前中心到目标点的误差向量。
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
                cv2.circle(canvas, (int(round(sx)), int(round(sy))), 6, (0, 255, 0), -1)
                cv2.putText(canvas, "route start", (int(round(sx)) + 8, int(round(sy)) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2, cv2.LINE_AA)

                sample_step = max(1, len(route_points) // 30)
                for ridx, (rx, ry) in enumerate(route_points[::sample_step]):
                    cv2.circle(canvas, (int(round(rx)), int(round(ry))), 2, (255, 0, 255), -1)

            if target_xy is not None:
                tx, ty = target_xy
                cv2.drawMarker(canvas, (int(round(tx)), int(round(ty))), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
                cv2.circle(canvas, (int(round(tx)), int(round(ty))), 7, (0, 0, 255), 2)
                cv2.putText(canvas, f"target #{int(route_index)}", (int(round(tx)) + 10, int(round(ty)) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

            if a_center is not None:
                ax, ay = a_center
                cv2.circle(canvas, (int(round(ax)), int(round(ay))), 7, (255, 255, 0), -1)
                cv2.putText(canvas, "A center", (int(round(ax)) + 10, int(round(ay)) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)
                if target_xy is not None:
                    tx, ty = target_xy
                    cv2.arrowedLine(canvas, (int(round(ax)), int(round(ay))), (int(round(tx)), int(round(ty))), (0, 255, 255), 2, tipLength=0.18)

            dist_text = "" if error_dist is None else f", dist={float(error_dist):.2f}px"
            cv2.putText(
                canvas,
                f"Step7 A route: step={step_idx}, idx={route_index}/{max(0, len(route_points)-1)}, action={action_name}{dist_text}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )

            out_path = out_dir / f"step7_a_route_step_{int(step_idx):04d}_idx_{int(route_index):04d}.png"
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

    def _run_one_c_edge_route_cycle(
        self,
        follower: Any,
        step_idx: int,
        route_points: List[Tuple[float, float]],
    ) -> Tuple[bool, Dict[str, Any]]:
        """执行一次“沿 C 边沿路线 + 路线安全带”的 Step7 运动。"""
        if not route_points:
            return False, {"route_enabled": True, "reason": "empty_route"}

        # 捕获当前帧并获得 A/B/C scene；不再调用 follower.rule_policy(scene)。
        image_rgb, scene = follower.capture_and_build_scene()
        try:
            follower.cycle_index = int(getattr(follower, "cycle_index", 0) or 0) + 1
        except Exception:
            pass

        # A/B 分割保护：先保护 B，再从 A 中排除 B；后续 A 中心、路线控制、角度修复都使用修正后的 mask。
        ab_guard_info = self._postprocess_scene_ab_masks(
            follower=follower,
            scene=scene,
            tag=f"step7_route_{int(step_idx)}",
        )

        a_center = self._get_scene_a_center_xy(scene)
        if a_center is None:
            return False, {"route_enabled": True, "reason": "a_center_missing"}

        route_state = self.context.setdefault("step7_c_edge_route_state", {})
        route_sig = route_state.get("route_signature")
        new_sig = (len(route_points), tuple(route_points[0]), tuple(route_points[-1]))
        if route_sig != new_sig:
            route_points = self._order_route_points_by_nearest(route_points, a_center)
            route_state.clear()
            route_state["route_signature"] = new_sig
            route_state["route_points"] = [[float(x), float(y)] for x, y in route_points]
            route_state["route_index"] = 0

        # 使用旋转后的 target route；如果状态里已有，就保持上次进度。
        stored_route = route_state.get("route_points")
        if stored_route:
            route_points = [(float(p[0]), float(p[1])) for p in stored_route]

        tol = max(1.0, float(getattr(self.cfg, "rule_ab_route_target_tolerance_px", 8.0)))
        loop_route = bool(getattr(self.cfg, "rule_ab_route_loop", True))

        # 三线安全带：
        #   base_route = C 近似四边形边界；
        #   min/max_route = GUI 绿框 min/max 外扩边界；
        #   route_points = GUI 绿框 target 外扩中线，只提供切向方向参考。
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

        # 用 base_route 判断 A 是否在 min/max 安全带内；用 target route 的切向方向决定安全带内的运动方向。
        safety_info = self._route_safety_target_from_distance(a_center, base_route_points)
        use_safety_correction = bool(safety_info.get("should_correct", False))

        near_target = self._nearest_point_on_route_polyline(a_center, route_points, closed=True)
        idx = int(near_target.get("segment_index", route_state.get("route_index", 0) or 0)) if near_target.get("ok", False) else int(route_state.get("route_index", 0) or 0)
        idx = max(0, min(idx, len(route_points) - 1))
        route_state["route_index"] = idx

        action_name_prefix = ""
        if use_safety_correction and safety_info.get("safe_target_xy") is not None:
            # 越界：只做法向安全修正，把 A 拉回 target 中线附近；暂不强制沿切向前进。
            sx, sy = safety_info["safe_target_xy"]
            cur_target = (float(sx), float(sy))
            target_mode = str(safety_info.get("mode", "route_safety_correction"))
            action_name_prefix = "SAFE_"
        else:
            # 在 min/max 两条边界线之间：不再追某个固定 route point，
            # 只沿 target 中线当前位置的切向方向运动。
            if near_target.get("ok", False):
                tx, ty = near_target.get("segment_unit_xy", (1.0, 0.0))
            else:
                # 兜底：用当前 idx 到下一点的方向。
                p0 = route_points[idx]
                p1 = route_points[(idx + 1) % len(route_points)] if loop_route or idx + 1 < len(route_points) else route_points[idx]
                vx, vy = float(p1[0] - p0[0]), float(p1[1] - p0[1])
                nrm = math.hypot(vx, vy)
                tx, ty = (vx / nrm, vy / nrm) if nrm > 1e-9 else (1.0, 0.0)

            virtual_step_px = max(float(getattr(self.cfg, "rule_ab_c_edge_route_spacing_px", 12.0)), tol * 2.0, 10.0)
            cur_target = (float(a_center[0]) + float(tx) * virtual_step_px, float(a_center[1]) + float(ty) * virtual_step_px)
            target_mode = "inside_band_follow_target_tangent"
            safety_info["target_tangent_xy"] = [float(tx), float(ty)]
            safety_info["virtual_tangent_step_px"] = float(virtual_step_px)

        dx = float(cur_target[0] - a_center[0])
        dy = float(cur_target[1] - a_center[1])
        action_id = self._route_action_from_error(dx, dy, tol)
        action_name_map = {0: "STAY", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT"}
        action_name = action_name_prefix + action_name_map.get(action_id, f"ACTION_{action_id}")

        status = {}
        try:
            if hasattr(follower, "get_boundary_rule_status"):
                status = follower.get_boundary_rule_status(scene) or {}
        except Exception:
            status = {}

        move_info: Dict[str, Any] = {}
        if action_id != 0:
            move_info = follower.execute_action(action_id) or {}

        # 尽量保存外部 follower 的标注图，便于后续从 annotated_frames 追溯角度来源。
        try:
            if hasattr(follower, "save_annotated_image"):
                follower.save_annotated_image(image_rgb, scene, status, action_id=action_id, action_name=f"ROUTE_{action_name}")
        except TypeError:
            try:
                follower.save_annotated_image(image_rgb, scene, status, action_id, f"ROUTE_{action_name}")
            except Exception:
                pass
        except Exception:
            pass

        route_overlay_path = self._save_step7_route_overlay_image(
            image_rgb=image_rgb,
            route_points=route_points,
            a_center=a_center,
            target_xy=cur_target,
            route_index=idx,
            step_idx=step_idx,
            action_name=action_name,
            error_dist=float(math.hypot(dx, dy)),
            min_route_points=min_route_points,
            max_route_points=max_route_points,
            base_route_points=base_route_points,
        )

        info = {
            "route_enabled": True,
            "step": int(step_idx),
            "route_index": int(idx),
            "route_len": int(len(route_points)),
            "a_center_xy": [float(a_center[0]), float(a_center[1])],
            "target_xy": [float(cur_target[0]), float(cur_target[1])],
            "error_dx": dx,
            "error_dy": dy,
            "error_dist": float(math.hypot(dx, dy)),
            "tolerance_px": tol,
            "action_id": int(action_id),
            "action_name": action_name,
            "target_mode": target_mode,
            "route_safety": self._json_safe(safety_info),
            "route_distance_px": safety_info.get("route_distance_px"),
            "route_min_distance_px": safety_info.get("min_distance_px"),
            "route_max_distance_px": safety_info.get("max_distance_px"),
            "route_overlay_path": route_overlay_path,
            "ab_mask_guard": self._json_safe(ab_guard_info),
            "move_info": self._json_safe(move_info),
            "status": self._json_safe(status),
        }
        self.log(
            f"[RuleAB-Route] step={step_idx}, route_idx={idx}/{len(route_points)-1}, "
            f"mode={target_mode}, route_dist={safety_info.get('route_distance_px')}, "
            f"safe_band=[{safety_info.get('min_distance_px')},{safety_info.get('max_distance_px')}], "
            f"A=({a_center[0]:.1f},{a_center[1]:.1f}), "
            f"target=({cur_target[0]:.1f},{cur_target[1]:.1f}), "
            f"err=({dx:.1f},{dy:.1f}), dist={info['error_dist']:.2f}px, action={action_name}"
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
                setattr(obj, "center", (float(np.mean(xs)), float(np.mean(ys))))
        except Exception:
            pass
        return ok

    def _postprocess_scene_ab_masks(self, follower: Any, scene: Any, tag: str = "") -> Dict[str, Any]:
        """
        对当前 scene 的 A/B mask 做运行时后处理。

        当前版本按你的要求修改为：
            1. B mask 不再做面积比例、中心跳变、主轴角度跳变约束；
            2. B mask 不再因为异常而回退上一帧；
            3. 后续运动控制、角度检测、角度修复都直接使用当前帧 B mask；
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

        单独测试 Step9 时只需要分割/跟踪 B，因此这里单独缓存一个 predictor，
        避免每一帧重复加载模型。
        """
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
            image_rgb, scene = follower.capture_and_build_scene()
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
                image_rgb = self._capture_current_frame_rgb()
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
    ) -> Dict[str, Optional[str]]:
        """
        保存 Step9 颜色检测区域中心对齐的检查图和 A/B 分割调试结果。

        当前 Step9 不再计算 颜色检测区域；A/B mask 仅保存，方便检查每帧 A/B 分割是否正确。
        """
        saved = {
            "overlay_path": None,
            "frame_path": None,
            "color_mask_path": None,
            "a_mask_path": None,
            "b_mask_path": None,
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

            canvas = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # 调试叠加：A 黄色，B 绿色，颜色检测区域青色。A/B 只用于观察，不参与 Step9 对齐。
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

            if self.stop_requested or self.step9_stop_requested:
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
            "reason": "aligned" if final_ok else "not_aligned_or_stopped_or_detect_failed",
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

        if isinstance(values, str):
            text = (
                values.replace("\\n", ",")
                .replace("\r", ",")
                .replace("\n", ",")
                .replace("\t", ",")
                .replace(";", ",")
                .replace(" ", ",")
            )
            values = [x for x in text.split(",") if x.strip()]

        out: List[float] = []
        try:
            for x in values:
                try:
                    if x is None:
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

    def request_labview_spectrum(self, cycle_index: int) -> Dict[str, Any]:
        """
        光谱仪采集已按需求注释掉。

        原来这里会通过 LabVIEW TCP 请求一次光谱采集；
        现在不再访问 TCP、不读取光谱 CSV，只生成一个占位结果，
        并把所有后续保存/汇总会用到的峰值字段统一写成 1000。
        """
        self.log("========== Step 4：光谱仪采集已禁用，使用占位峰值 1000 ==========")

        result = {
            "ok": True,
            "disabled": True,
            "index": cycle_index,
            "num_points": 0,
            "csv_path": "",
            "message": "spectrometer_tcp_disabled_dummy_peak_1000",
            "raw_original_peak": DUMMY_SPECTRUM_PEAK,
            "raw_peak": DUMMY_SPECTRUM_PEAK,
            "fit_peak": DUMMY_SPECTRUM_PEAK,
        }

        self.context["last_tcp_result"] = result

        # 不再加载/保存真实光谱数组；绘图和每轮光谱 CSV 写入会自动跳过数组。
        self.context["wavelength"] = []
        self.context["x_axis_values"] = []
        self.context["x_axis_xlsx_path"] = self.cfg.x_axis_xlsx_path
        self.context["intensity"] = []
        self.context["raw_values"] = []
        self.context["raw_filtered_values"] = []
        self.context["raw_median_values"] = []
        self.context["fit_values"] = []
        self.context["fit_params"] = None

        # 保存数据中的峰值点统一用 1000 代替。
        self.context["raw_original_peak"] = DUMMY_SPECTRUM_PEAK
        self.context["raw_filtered_peak"] = DUMMY_SPECTRUM_PEAK
        self.context["raw_median_peak"] = DUMMY_SPECTRUM_PEAK
        self.context["raw_peak"] = DUMMY_SPECTRUM_PEAK
        self.context["fit_peak"] = DUMMY_SPECTRUM_PEAK
        self.context["raw_filter_threshold"] = self.cfg.raw_remove_above
        self.context["median_filter_window"] = self.cfg.median_filter_window

        self.log("[光谱] 已跳过真实 TCP 采集；raw_original_peak/raw_peak/fit_peak 均保存为 1000")
        self.notify_update()
        return result

    def _parse_labview_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        兼容两类 TCP 返回：
        1. 旧格式：DATA,... 解析后 result["values"]；
        2. 新格式：RAW_Y,... 和 FIT_Y,... 解析后 result["raw_y"] / result["fit_y"]。

        原始数据处理顺序：
            raw_values -> 删除 > raw_remove_above 的点 -> 5点中值滤波。
        """
        wavelength = None
        raw_values = None
        fit_values = None

        if "wavelength" in result:
            wavelength = self._to_float_list(result.get("wavelength"))

        # 优先读取 RAW_Y，其次读取 intensity/data/values
        for key in ("raw_y", "RAW_Y", "raw_values", "intensity", "values"):
            if raw_values is None and key in result:
                raw_values = self._to_float_list(result.get(key))

        if raw_values is None and "spectrum" in result:
            spectrum = result.get("spectrum")
            if isinstance(spectrum, dict):
                wavelength = wavelength or self._to_float_list(spectrum.get("wavelength"))
                raw_values = self._to_float_list(spectrum.get("intensity") or spectrum.get("data"))
            elif isinstance(spectrum, list):
                raw_values = self._to_float_list(spectrum)

        if raw_values is None and "data" in result:
            data = result.get("data")
            if isinstance(data, dict):
                wavelength = wavelength or self._to_float_list(data.get("wavelength"))
                raw_values = self._to_float_list(data.get("intensity") or data.get("values"))
            elif isinstance(data, list):
                raw_values = self._to_float_list(data)

        if raw_values is None and result.get("csv_path"):
            wavelength, raw_values = self._load_spectrum_from_csv(result["csv_path"])

        for key in ("fit_y", "FIT_Y", "fit_values"):
            if fit_values is None and key in result:
                fit_values = self._to_float_list(result.get(key))

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

        fit_peak = result.get("fit_peak")
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
            "fit_params": result.get("fit_params"),
        }

    def _load_spectrum_from_csv(self, csv_path: str) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        path = Path(csv_path)

        if not path.exists():
            self.log(f"[光谱] CSV 不存在，无法读取：{csv_path}")
            return None, None

        wavelength: List[float] = []
        intensity: List[float] = []

        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                try:
                    if len(row) >= 2:
                        wavelength.append(float(row[0]))
                        intensity.append(float(row[1]))
                    elif len(row) == 1:
                        intensity.append(float(row[0]))
                except Exception:
                    continue

        if len(wavelength) == 0:
            wavelength = None
        if len(intensity) == 0:
            intensity = None

        return wavelength, intensity

    def begin_new_run_session(self):
        """
        每次点击“运行完整循环测量”时调用一次。

        作用：
            1. 创建本次点击对应的总文件夹，文件夹名用点击时间；
            2. 清空本次 GUI 右侧“角度-拟合峰值列表”的历史点；
            3. 重置上一轮角度，使第一轮不进行 Step 11 自适应调节。
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

        self.plot_points.clear()
        self.context["plot_points"] = self.plot_points

        self.context["current_paths"] = {
            "run_session_dir": str(self.run_session_dir),
            "run_session_name": self.run_session_name,
        }

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

            # 光谱仪已禁用：保存快照中的峰值统一固定为 1000。
            "raw_original_peak": DUMMY_SPECTRUM_PEAK,
            "raw_filtered_peak": DUMMY_SPECTRUM_PEAK,
            "raw_median_peak": DUMMY_SPECTRUM_PEAK,
            "raw_peak": DUMMY_SPECTRUM_PEAK,
            "fit_peak": DUMMY_SPECTRUM_PEAK,
            "fit_params": self._json_safe(ctx.get("fit_params")),

            "raw_filter_threshold": ctx.get("raw_filter_threshold"),
            "median_filter_window": ctx.get("median_filter_window"),
            "x_axis_xlsx_path": ctx.get("x_axis_xlsx_path"),

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
        ]
        return {k: MeasurementWorkflow._json_safe(result.get(k)) for k in keep_keys if k in result}

    def start_save_cycle_result_async(
        self,
        cycle_index: int,
        paths: Dict[str, str],
        angle_before_result: Dict[str, Any],
        angle_after_result: Dict[str, Any],
        labview_result: Dict[str, Any],
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
                # 本轮检测角度。
                # 当前版本每轮只检测一次角度，保存时 angle_before 字段就是本轮 current_angle；
                # 为了兼容旧逻辑，如果后续又启用 angle_after，则优先保存 angle_after。
                angle_value = ctx.get("angle_after")
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
                self.log("[保存] 光谱仪已禁用：本轮不保存真实光谱数组；汇总峰值固定写入1000")

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
                DUMMY_SPECTRUM_PEAK,
                DUMMY_SPECTRUM_PEAK,
                DUMMY_SPECTRUM_PEAK,
                paths.get("run_session_name"),
                paths.get("spectrum_filename"),
                "",
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

        angle_value = ctx.get("angle_after")
        if angle_value is None:
            angle_value = ctx.get("angle_before")

        # 光谱仪已禁用：右侧列表/绘图的峰值统一显示 1000。
        fit_peak = DUMMY_SPECTRUM_PEAK

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

        # “本次角度”采用本轮最终角度 angle_after；如果 angle_after 无效，退回 angle_before。
        angle_value = ctx.get("angle_after")
        if angle_value is None:
            angle_value = ctx.get("angle_before")

        row = [
            int(cycle_index),
            angle_value,
            DUMMY_SPECTRUM_PEAK,
            DUMMY_SPECTRUM_PEAK,
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
    # 单轮完整流程
    # --------------------------------------------------------

    def run_one_cycle(self, cycle_index: int) -> bool:
        """
        单轮循环测量流程：
        Step 1：角度检测一次，得到 current_angle
        Step 2：生成保存路径
        Step 3：照明光 OFF，并按 stable_wait_ms 等待稳定
        Step 4：LabVIEW 光谱采集
        Step 5：照明光 ON
        Step 6：打开激光
        Step 7：A推动B（logic/rule_ab.py），每次运动后检测B对边角度；
                角度修复只在 raw_angle 相对上一帧有效角度跳变超过 10° 时执行；
                检测角度变化或修复角度变化一旦 >= min_delta_deg（默认3.5°），就在 Step7 内立即关闭激光；
                max_delta_deg（默认6°）只作为过冲提示，目标是尽可能不要超过6°。
        Step 8：检测颜色区域中心，与提前选定位置对齐；偏离过大则移动1/2通道
        Step 9：保存本轮数据；保存角度使用 Step7 的最终角度；若触发修复则保存修复角度，否则保存检测角度。
        """
        self.context["cycle_index"] = cycle_index
        self.notify_update()

        self.log("")
        self.log("############################################################")
        self.log(f"开始第 {cycle_index} 轮循环测量")
        self.log("############################################################")

        try:
            self.context["signal_on_time_used_ms"] = None
            self.context["signal_on_time_next_ms"] = None
            self.context["signal_time_adjust_action"] = None

            self.log("========== Step 1：角度检测 current_angle ==========")
            # 新逻辑：每次角度计算都重新分割 A/B，并选择 A 中心距离最近的 B 边作为角度边。
            # 先保留旧 ScreenAngleDetector 结果作为兜底，避免 A/B 分割偶发失败时无法继续。
            screen_angle_result = self.detect_angle_once(label="current_angle_screen_fallback", allow_fail=True)
            angle_result = screen_angle_result
            try:
                follower_for_angle = self.ensure_rule_ab_follower()
                self.apply_runtime_rule_ab_params_to_follower(follower_for_angle, reason="step1_ab_nearest_angle")
                ab_angle_result = self.detect_ab_nearest_b_edge_angle_once(
                    follower=follower_for_angle,
                    label="current_angle_ab_nearest",
                    allow_fail=True,
                )
                if ab_angle_result.get("ok", False) and ab_angle_result.get("angle_deg") is not None:
                    angle_result = ab_angle_result
                    self.log(
                        f"[角度检测-current_angle] 已使用 A/B 分割最近边角度作为本轮 baseline："
                        f"{ab_angle_result.get('angle_deg')}°"
                    )
                else:
                    self.log(
                        f"[角度检测-current_angle] A/B 最近边角度失败，保留旧 ScreenAngleDetector 结果；"
                        f"reason={ab_angle_result.get('reason')}"
                    )
            except Exception as e:
                self.log(f"[角度检测-current_angle] A/B 最近边 baseline 计算异常，保留旧 ScreenAngleDetector 结果：{e}")

            if angle_result.get("ok", False) and angle_result.get("angle_deg") is not None:
                current_angle = float(angle_result["angle_deg"])
            else:
                current_angle = None
                self.log("[角度检测-current_angle] 本轮角度无效，流程继续，但本轮不参与跨轮角度差调节")

            # 兼容旧显示/保存字段：angle_before 作为本轮角度，angle_after 不再使用。
            self.context["angle_before"] = current_angle
            self.context["angle_after"] = None
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

            self.log("========== Step 2：生成保存路径 ==========")
            paths = self.build_save_path(cycle_index)

            self.log("========== Step 3：照明光 OFF，等待稳定 ==========")
            self.light_off()
            wait_after_off = float(self.cfg.stable_wait_ms) / 1000.0
            if wait_after_off > 0:
                self.log(f"[照明光] OFF 后稳定等待：{wait_after_off:.3f} s（由“稳定等待/ms”控制）")
                time.sleep(wait_after_off)
            else:
                self.log("[照明光] OFF 后稳定等待：0 s，立即进入 LabVIEW 光谱采集")

            if self.stop_requested:
                return False

            self.log("========== Step 4：LabVIEW 光谱采集 ==========")
            labview_result = self.request_labview_spectrum(cycle_index)

            if self.stop_requested:
                return False

            self.log("========== Step 5：照明光 ON ==========")
            self.light_on()

            # 注意：这里不再提前保存。
            # 保存必须放在 Step7 和 Step8 对齐之后，否则保存线程拿到的 context_snapshot
            # 只有 Step1 baseline 角度，没有 Step7 的最终角度。
            if self.stop_requested:
                return False

            self.log("========== Step 6：打开激光 ==========")
            self.laser_on()

            self.log("========== Step 7：A推动B，并检测B对边角度 ==========")

            # 完整测量 Step7 的目标就是让 3/4 通道推动 A。
            # 如果 GUI 中“RuleAB真动Stage34”没有勾选，旧代码会 dry-run，表现为 3/4 通道没有任何变化。
            # 这里在完整测量中自动打开真动 Stage34；单独测试按钮仍可通过 GUI 选择 dry-run。
            if not bool(getattr(self.cfg, "rule_ab_enable_stage", False)):
                self.log("[RuleAB] 完整测量 Step7 检测到 rule_ab_enable_stage=False，已自动改为 True，避免3/4通道 dry-run。")
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

            self.context["last_rule_ab_result"] = rule_ab_result
            self.notify_update()

            if self.stop_requested:
                return False

            # Step7 判定：
            # 只有检测角度变化或修复角度变化 >= min_delta，Step7 才算成功并允许继续完整测量。
            # max_delta 只是过冲提示，不再阻止关闭激光；越早达到 min_delta 越能避免超过 6°。
            # RuleAB 返回 False、用户停止、角度长期无效等失败：停止完整测量。
            if not bool(rule_ab_result.get("ok", False)):
                final_delta = rule_ab_result.get("final_delta")
                target_min = rule_ab_result.get("target_min")
                target_max = rule_ab_result.get("target_max")
                reason = rule_ab_result.get("reason")

                self.log(
                    "[RuleAB] Step7 未达到可继续完整测量的条件，停止完整循环测量："
                    f"final_delta={final_delta}, target=[{target_min}, {target_max}], "
                    f"reason={reason}, records={len(rule_ab_result.get('records', []))}"
                )
                self.context["rule_ab_failed_stop"] = True
                self.context["rule_ab_failed_reason"] = reason
                self.stop_requested = True
                self.notify_update()
                return False

            self.log("========== Step 8：检测颜色区域中心，必要时移动1/2通道 ==========")
            rule_ac_result = self.run_rule_ac_until_threshold()
            self.context["last_rule_ac_result"] = rule_ac_result
            self.notify_update()

            # Step7 后，用最终角度覆盖本轮保存字段。
            # angle_before 保留 Step1 baseline；angle_after 写 Step7 的最终角度。
            # save_cycle_result() 会优先保存 ctx["angle_after"]，因此 CSV/XLSX/JSON 都会使用修复后的最终角度。
            final_step7_angle = rule_ab_result.get("final_angle")
            final_step7_delta = rule_ab_result.get("final_delta")
            last_records = rule_ab_result.get("records") or []
            last_record = last_records[-1] if last_records else {}
            last_angle_result = last_record.get("angle_result") if isinstance(last_record.get("angle_result"), dict) else {}

            self.context["angle_before"] = current_angle
            self.context["angle_after"] = final_step7_angle
            self.context["angle_before_after_delta"] = final_step7_delta
            self.context["last_angle_result"] = last_angle_result or angle_result
            self.context["last_rule_ab_final_angle"] = final_step7_angle
            self.context["last_rule_ab_final_delta"] = final_step7_delta
            self.context["last_rule_ab_used_repaired_angle"] = bool(
                (last_record.get("angle_repair") or {}).get("used_repaired_angle", False)
            ) if isinstance(last_record.get("angle_repair"), dict) else False
            self.notify_update()

            angle_after_result = {
                "ok": final_step7_angle is not None,
                "angle_deg": final_step7_angle,
                "angle_deg_raw": last_record.get("angle_deg_raw"),
                "angle_delta_from_baseline": final_step7_delta,
                "angle_repair": last_record.get("angle_repair"),
                "used_repaired_angle": bool(
                    (last_record.get("angle_repair") or {}).get("used_repaired_angle", False)
                ) if isinstance(last_record.get("angle_repair"), dict) else False,
                "reason": rule_ab_result.get("reason"),
                "step": last_record.get("step"),
                "angle_result": last_angle_result,
            }

            self.log("========== Step 9：保存数据（异步线程，使用 Step7 最终角度） ==========")
            t_save_submit0 = time.time()
            self.start_save_cycle_result_async(
                cycle_index=cycle_index,
                paths=paths,
                angle_before_result=angle_result,
                angle_after_result=angle_after_result,
                labview_result=labview_result,
            )
            self.log(f"[Step 9] 异步保存任务提交耗时：{time.time() - t_save_submit0:.3f} s")

            self.log(f"[流程] 第 {cycle_index} 轮保存任务已在 Step9 提交，可以进入下一轮")
            self.notify_update()
            return True

        except Exception as e:
            self.log(f"[错误] 第 {cycle_index} 轮测量失败：{e}")
            self.log(traceback.format_exc())

            error_dir = self.output_root / "errors"
            error_dir.mkdir(parents=True, exist_ok=True)

            error_path = error_dir / f"error_cycle_{cycle_index:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            error_info = {
                "cycle_index": cycle_index,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error": str(e),
                "traceback": traceback.format_exc(),
                "context": self._safe_context_for_json(),
            }

            with error_path.open("w", encoding="utf-8") as f:
                json.dump(error_info, f, ensure_ascii=False, indent=2)

            self.log(f"[错误] 错误信息已保存：{error_path}")

            self.finish()
            return False

    def _safe_context_for_json(self) -> Dict[str, Any]:
        return {k: self._json_safe(v) for k, v in self.context.items()}

    def run(self):
        self.connect_measurement_devices()

        # 光谱仪 TCP 已禁用：完整测量不再要求 TCP Server 或 LabVIEW READY。
        self.context["tcp_started"] = False
        self.context["labview_ready"] = True
        self.log("[流程] 光谱仪 TCP 已禁用：跳过完整测量前的 TCP/READY 检查")

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

        self.save_summary_xlsx()
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
        关闭 GUI/流程时的兜底清理。

        当前要求：
            1. 当前版本不再使用 Rigol 信号发生器；
            2. 关闭 GUI 时兜底关闭 Newport 激光控制器；
            3. 照明光不在本次修改范围内，关闭 GUI 时不额外发送 ON/OFF；
            4. TCP 连接需要关闭，释放端口。
        """
        self.log("========== 关闭GUI：关闭激光控制器，照明光保持当前状态 ==========")
        self.save_summary_xlsx()

        try:
            if self.light is not None:
                self.log("[照明光] GUI关闭时不额外发送 ON/OFF，保持当前照明状态")
        except Exception as e:
            self.log(f"[照明光] GUI关闭保持状态提示失败：{e}")

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

        try:
            if self.tcp_server is not None:
                self.tcp_server.close()
                self.tcp_server = None
                self.context["tcp_started"] = False
                self.context["labview_ready"] = False
                self.log("[TCP] 已关闭")
        except Exception as e:
            self.context["tcp_started"] = False
            self.context["labview_ready"] = False
            self.log(f"[TCP] 关闭失败：{e}")

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

        self.log("========== GUI关闭清理完成：激光控制器已关闭，照明光未额外切换 ==========")
        self.notify_update()

# ============================================================
# GUI
# ============================================================

class MeasurementWorkflowGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("循环测量 GUI：照明光 + Newport激光开关 + Rigol + ScreenAngleDetector角度检测 + LabVIEW TCP光谱")
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

        self.fig: Optional[Figure] = None
        self.ax_fit_peak = None
        self.ax_raw = None
        self.ax_median = None
        self.plot_canvas: Optional[FigureCanvasTkAgg] = None
        self.plot_status_var = tk.StringVar(value="图像显示：暂无数据")

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
        ttk.Button(button_grid, text="1 标定角度B/边", command=self.calibrate_angle_b_thread).grid(row=2, column=0, padx=4, pady=4, sticky="ew")
        ttk.Button(button_grid, text="2 标定RuleAB-A", command=self.calibrate_rule_ab_a_thread).grid(row=2, column=1, padx=4, pady=4, sticky="ew")
        ttk.Button(button_grid, text="2b 标定RuleAB-B", command=self.calibrate_rule_ab_b_thread).grid(row=3, column=0, padx=4, pady=4, sticky="ew")
        ttk.Button(button_grid, text="3 标定全局C", command=self.calibrate_global_c_thread).grid(row=3, column=1, padx=4, pady=4, sticky="ew")
        ttk.Button(button_grid, text="4 标定Step9目标/颜色", command=self.calibrate_step9_target_color_thread).grid(row=4, column=0, padx=4, pady=4, sticky="ew")
        ttk.Button(button_grid, text="保存/检查完整标定", command=self.save_and_check_calibration_thread).grid(row=4, column=1, padx=4, pady=4, sticky="ew")
        ttk.Button(button_grid, text="加载完整标定", command=self.load_calibration_to_gui_thread).grid(row=5, column=0, columnspan=2, padx=4, pady=4, sticky="ew")

        ttk.Label(flow_frame, textvariable=self.flow_status_var, wraplength=460).pack(fill=tk.X, padx=4, pady=(6, 0))
        ttk.Label(flow_frame, textvariable=self.calibration_status_var, wraplength=460, style="Value.TLabel").pack(fill=tk.X, padx=4, pady=(4, 0))
        ttk.Label(flow_frame, textvariable=self.calibration_path_var, wraplength=460, style="Value.TLabel").pack(fill=tk.X, padx=4, pady=(2, 0))

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
        angle_frame = ttk.LabelFrame(left_inner, text="5. ScreenAngleDetector 角度检测", padding=10, style="Panel.TLabelframe")
        angle_frame.pack(fill=tk.X, padx=4, pady=8)

        self.angle_model_path_var = tk.StringVar(value=cfg0.angle_model_path)
        self.capture_area_var = tk.StringVar(value=",".join(str(v) for v in cfg0.capture_area))
        self.angle_status_var = tk.StringVar(value="角度检测状态：未初始化")
        self.angle_result_var = tk.StringVar(value="角度结果：None")

        angle_frame.columnconfigure(1, weight=1)
        ttk.Label(angle_frame, text=".pt模型").grid(row=0, column=0, padx=4, pady=4, sticky="w")
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
        tcp_frame = ttk.LabelFrame(left_inner, text="6. LabVIEW TCP 通信", padding=10, style="Panel.TLabelframe")
        tcp_frame.pack(fill=tk.X, padx=4, pady=8)

        self.tcp_host_var = tk.StringVar(value=cfg0.tcp_host)
        self.tcp_port_var = tk.IntVar(value=cfg0.tcp_port)
        self.tcp_output_dir_var = tk.StringVar(value=cfg0.tcp_output_dir)
        self.tcp_command_var = tk.StringVar(value=cfg0.tcp_command)
        self.tcp_status_var = tk.StringVar(value="TCP状态：未启动")
        self.tcp_result_var = tk.StringVar(value="最近光谱：None")

        for i in range(4):
            tcp_frame.columnconfigure(i, weight=1)

        ttk.Label(tcp_frame, text="HOST").grid(row=0, column=0, padx=4, pady=4, sticky="w")
        ttk.Entry(tcp_frame, textvariable=self.tcp_host_var).grid(row=0, column=1, padx=4, pady=4, sticky="ew")
        ttk.Label(tcp_frame, text="PORT").grid(row=0, column=2, padx=4, pady=4, sticky="w")
        ttk.Entry(tcp_frame, textvariable=self.tcp_port_var).grid(row=0, column=3, padx=4, pady=4, sticky="ew")

        ttk.Label(tcp_frame, text="命令").grid(row=1, column=0, padx=4, pady=4, sticky="w")
        ttk.Entry(tcp_frame, textvariable=self.tcp_command_var).grid(row=1, column=1, padx=4, pady=4, sticky="ew")
        ttk.Label(tcp_frame, text="CSV目录").grid(row=1, column=2, padx=4, pady=4, sticky="w")
        ttk.Entry(tcp_frame, textvariable=self.tcp_output_dir_var).grid(row=1, column=3, padx=4, pady=4, sticky="ew")

        tcp_buttons = ttk.Frame(tcp_frame)
        tcp_buttons.grid(row=2, column=0, columnspan=4, padx=2, pady=4, sticky="ew")
        for i in range(2):
            tcp_buttons.columnconfigure(i, weight=1)
        ttk.Button(tcp_buttons, text="启动TCP", command=self.start_tcp_thread).grid(row=0, column=0, padx=4, pady=3, sticky="ew")
        ttk.Button(tcp_buttons, text="等待READY", command=self.wait_ready_thread).grid(row=0, column=1, padx=4, pady=3, sticky="ew")
        ttk.Button(tcp_buttons, text="单次光谱采集", command=self.measure_spectrum_once_thread).grid(row=1, column=0, padx=4, pady=3, sticky="ew")
        ttk.Button(tcp_buttons, text="关闭TCP", command=self.close_tcp_thread).grid(row=1, column=1, padx=4, pady=3, sticky="ew")

        ttk.Label(tcp_frame, textvariable=self.tcp_status_var, wraplength=450).grid(row=3, column=0, columnspan=4, padx=4, pady=(4, 0), sticky="w")
        ttk.Label(tcp_frame, textvariable=self.tcp_result_var, wraplength=450).grid(row=4, column=0, columnspan=4, padx=4, pady=(4, 0), sticky="w")

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
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        只为 C 进行交互式点提示：
            左键：C 正点，可以多个；
            右键：C 负点，可以多个；
            R：清空；
            Enter/N：完成；
            ESC：取消。
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
                "C-only SAM2 segmentation",
                "Left click: C positive point | Right click: C negative point",
                "N/Enter: finish | R: reset | ESC: cancel",
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

        return to_original(pos), to_original(neg)

    def _predict_c_mask_by_sam2_points(
        self,
        image_rgb: np.ndarray,
        positive_points: List[Tuple[float, float]],
        negative_points: List[Tuple[float, float]],
    ) -> np.ndarray:
        """调用 SAM2 ImagePredictor，仅根据 C 的正负点分割 C。"""
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
    def _fit_quadrilateral_from_mask(mask_bool: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
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

        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
            pos, neg = self._select_c_points_interactively(
                image_rgb=image_rgb,
                window_name="SAM2 C only: left positive, right negative",
                scale=float(getattr(cfg_gui, "rule_ab_confirm_window_scale", 0.85)),
            )

            self.log(f"[RuleAB-C] C 点提示：positive={len(pos)}, negative={len(neg)}")
            c_mask = self._predict_c_mask_by_sam2_points(image_rgb=image_rgb, positive_points=pos, negative_points=neg)
            saved_dir = self._save_static_c_map_result(
                image_rgb=image_rgb,
                c_mask=c_mask,
                positive_points=pos,
                negative_points=neg,
                save_dir=save_dir,
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
            self.root.after(0, lambda: messagebox.showerror("C分割失败", str(e)))

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
            max_cycles=int(self.max_cycles_var.get()),

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
                cycle_index = p.get("cycle_index")
                if cycle_index is None:
                    cycle_index = len(list_points) + 1

                angle = p.get("angle_deg")
                fit_peak = p.get("fit_peak")

                angle_value = None if angle is None else float(angle)
                fit_peak_value = None if fit_peak is None else float(fit_peak)

                cycle_index_int = int(cycle_index)
                list_points.append((cycle_index_int, angle_value, fit_peak_value))

                # 图1：序号 - 拟合峰值。
                # 拟合峰值为空时不画该点，但右侧列表仍保留该轮记录。
                if fit_peak_value is not None and math.isfinite(fit_peak_value):
                    fit_plot_x.append(cycle_index_int)
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
                f"角度-拟合峰值列表：{len(valid_points)} 个点；"
                "横坐标=本轮角度/deg（无角度时为None），纵坐标=拟合峰值"
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
            self.root.after(0, lambda: messagebox.showerror("初始化失败", str(e)))


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
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        通用点选窗口：左键正点，右键负点，Enter/N完成，R重选。
        返回截图区域原始坐标。
        """
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
                "Enter/N: finish | R: reset | ESC/Q: cancel",
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
        return to_original(pos), to_original(neg)

    def calibrate_angle_b_thread(self):
        self.run_in_thread(self.calibrate_angle_b)

    def calibrate_angle_b(self):
        """标定角度检测用的 B mask 正点/负点，以及要检测的边。"""
        try:
            cfg = self.build_config_from_ui()
            out_dir = Path(cfg.save_root) / "calibration" / "angle_b"
            out_dir.mkdir(parents=True, exist_ok=True)
            image_rgb = self._capture_current_rule_ab_frame(output_dir=out_dir)
            pos, neg = self._select_object_points_interactively(image_rgb, "Angle-B", "Calibration Angle B: left positive, right negative")

            edge_index_text = self._ask_string_on_main_thread(
                "角度检测边编号",
                "请输入要检测的 B 对边/边编号。\n建议与 angle_detect.py 中 num 对应；不知道就填当前 GUI 的 num。",
                initialvalue=str(self.num_var.get()),
            )
            if edge_index_text is None:
                raise RuntimeError("用户取消了角度边编号输入。")
            try:
                edge_index = int(str(edge_index_text).strip())
            except Exception:
                edge_index = int(self.num_var.get())

            edge_name = self._ask_string_on_main_thread(
                "角度检测边名称",
                "可选：请输入这条边的备注名称，例如 B_opposite_edge。",
                initialvalue="B_opposite_edge",
            ) or "B_opposite_edge"

            state = self._current_or_new_calibration_state()
            state.angle_b_positive_points = [[float(x), float(y)] for x, y in pos]
            state.angle_b_negative_points = [[float(x), float(y)] for x, y in neg]
            state.angle_edge_index = int(edge_index)
            state.angle_edge_name = str(edge_name)
            state.angle_num = int(edge_index)
            state.angle_cw = int(self.cw_var.get())

            # 角度检测 B 和 RuleAB-B 通常是同一个 B。
            # 如果 RuleAB-B 尚未单独标定，自动复用这里的 B 点，避免完整测量 Step7 再弹 B 点选择窗口。
            if not state.rule_ab_b_positive_points:
                state.rule_ab_b_positive_points = [[float(x), float(y)] for x, y in pos]
                state.rule_ab_b_negative_points = [[float(x), float(y)] for x, y in neg]

            self.set_var(self.num_var, int(edge_index))
            self._save_calibration_file_raw(state)
            self.log(f"[完整测量标定] 角度B/边标定完成：B+={len(pos)}, B-={len(neg)}, edge={edge_index}/{edge_name}")
        except Exception as e:
            self.log(f"[完整测量标定] 角度B/边标定失败：{e}")
            self.log(traceback.format_exc())
            self.root.after(0, lambda: messagebox.showerror("角度B/边标定失败", str(e)))

    def calibrate_rule_ab_a_thread(self):
        self.run_in_thread(self.calibrate_rule_ab_a)

    def calibrate_rule_ab_a(self):
        """标定 A推动B 中 A mask 的正点/负点。"""
        try:
            cfg = self.build_config_from_ui()
            out_dir = Path(cfg.save_root) / "calibration" / "rule_ab_a"
            out_dir.mkdir(parents=True, exist_ok=True)
            image_rgb = self._capture_current_rule_ab_frame(output_dir=out_dir)
            pos, neg = self._select_object_points_interactively(image_rgb, "RuleAB-A", "Calibration RuleAB A: left positive, right negative")

            state = self._current_or_new_calibration_state()
            state.rule_ab_a_positive_points = [[float(x), float(y)] for x, y in pos]
            state.rule_ab_a_negative_points = [[float(x), float(y)] for x, y in neg]
            self._save_calibration_file_raw(state)
            self.log(f"[完整测量标定] RuleAB-A 标定完成：A+={len(pos)}, A-={len(neg)}")
        except Exception as e:
            self.log(f"[完整测量标定] RuleAB-A 标定失败：{e}")
            self.log(traceback.format_exc())
            self.root.after(0, lambda: messagebox.showerror("RuleAB-A标定失败", str(e)))

    def calibrate_rule_ab_b_thread(self):
        self.run_in_thread(self.calibrate_rule_ab_b)

    def calibrate_rule_ab_b(self):
        """标定 A推动B 中 B mask 的正点/负点。"""
        try:
            cfg = self.build_config_from_ui()
            out_dir = Path(cfg.save_root) / "calibration" / "rule_ab_b"
            out_dir.mkdir(parents=True, exist_ok=True)
            image_rgb = self._capture_current_rule_ab_frame(output_dir=out_dir)
            pos, neg = self._select_object_points_interactively(
                image_rgb,
                "RuleAB-B",
                "Calibration RuleAB B: left positive, right negative",
            )

            state = self._current_or_new_calibration_state()
            state.rule_ab_b_positive_points = [[float(x), float(y)] for x, y in pos]
            state.rule_ab_b_negative_points = [[float(x), float(y)] for x, y in neg]
            self._save_calibration_file_raw(state)
            self.log(f"[完整测量标定] RuleAB-B 标定完成：B+={len(pos)}, B-={len(neg)}")
        except Exception as e:
            self.log(f"[完整测量标定] RuleAB-B 标定失败：{e}")
            self.log(traceback.format_exc())
            self.root.after(0, lambda: messagebox.showerror("RuleAB-B标定失败", str(e)))

    def calibrate_global_c_thread(self):
        self.run_in_thread(self.calibrate_global_c)

    def calibrate_global_c(self):
        """标定整体测量前的 C mask 正点/负点，并保存 static C map。"""
        try:
            self.set_var(self.rule_module_status_var, "规则模块状态：正在完整测量前标定C")
            cfg_gui = self.build_config_from_ui()
            save_root = Path(str(getattr(cfg_gui, "save_root", "measurement_output")))
            map_name = str(getattr(cfg_gui, "rule_ab_static_c_map_name", "default_static_c_map")).strip() or "default_static_c_map"
            safe_map_name = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in map_name)
            save_dir = save_root / "rule_ab_follow_c_only" / "_static_c_map_library" / safe_map_name

            image_rgb = self._capture_current_rule_ab_frame(output_dir=save_dir)
            pos, neg = self._select_c_points_interactively(
                image_rgb=image_rgb,
                window_name="Calibration Global C: left positive, right negative",
                scale=float(getattr(cfg_gui, "rule_ab_confirm_window_scale", 0.85)),
            )
            c_mask = self._predict_c_mask_by_sam2_points(image_rgb=image_rgb, positive_points=pos, negative_points=neg)
            saved_dir = self._save_static_c_map_result(image_rgb=image_rgb, c_mask=c_mask, positive_points=pos, negative_points=neg, save_dir=save_dir)

            self.rule_ab_static_c_map_dir_var.set(str(saved_dir))
            self.rule_ab_load_static_c_map_var.set(True)
            self.rule_ab_force_reselect_c_var.set(False)

            state = self._current_or_new_calibration_state()
            state.global_c_positive_points = [[float(x), float(y)] for x, y in pos]
            state.global_c_negative_points = [[float(x), float(y)] for x, y in neg]
            state.static_c_map_dir = str(saved_dir)
            c_mask_path = Path(saved_dir) / "static_c_mask.png"
            if c_mask_path.exists():
                state.static_c_mask_path = str(c_mask_path.resolve())
            self._save_calibration_file_raw(state)
            self.log(f"[完整测量标定] 全局C标定完成并保存 static C：{saved_dir}")
        except Exception as e:
            self.log(f"[完整测量标定] 全局C标定失败：{e}")
            self.log(traceback.format_exc())
            self.root.after(0, lambda: messagebox.showerror("全局C标定失败", str(e)))

    def calibrate_step9_target_color_thread(self):
        self.run_in_thread(self.calibrate_step9_target_color)

    def calibrate_step9_target_color(self):
        """标定 Step9 目标点和颜色检测 HSV。

        关键要求：
            点击“4 标定Step9目标/颜色”时，必须重新标定目标点，
            不能因为 GUI 或旧标定文件里已有 target_x/y 就复用旧目标。
        """
        try:
            # 强制清空旧目标，避免旧完整标定文件加载出的 target_x/y 被继续复用。
            self.set_var_blocking(self.rule_ac_target_x_var, -1.0)
            self.set_var_blocking(self.rule_ac_target_y_var, -1.0)

            # 每次点击按钮都必须重新选 Step9 目标点。
            selected_target = self.select_step9_target()
            if selected_target is None:
                raise RuntimeError("Step9 目标点未完成选择，已取消 Step9 目标/颜色标定。")
            target_x, target_y = float(selected_target[0]), float(selected_target[1])

            # select_step9_target 内部会写 GUI 变量；这里再同步写一次并等待，
            # 保证后续 sync_config_from_ui_to_workflow() 读到的是本次新目标。
            self.set_var_blocking(self.rule_ac_target_x_var, round(target_x, 3))
            self.set_var_blocking(self.rule_ac_target_y_var, round(target_y, 3))

            wf = self.sync_config_from_ui_to_workflow()
            wf.cfg.rule_ac_target_x_px = target_x
            wf.cfg.rule_ac_target_y_px = target_y
            wf.begin_step9_run_session()
            # Step9 使用颜色检测区域中心对齐，因此目标点之后还需要选择一次颜色 HSV。
            wf.prepare_step9_color_region_detection()

            # 重新同步 GUI，确保本次新目标点和颜色 HSV 进入 cfg。
            wf = self.sync_config_from_ui_to_workflow()
            wf.cfg.rule_ac_target_x_px = target_x
            wf.cfg.rule_ac_target_y_px = target_y
            state = self._current_or_new_calibration_state()
            state.step9_target_x_px = float(target_x)
            state.step9_target_y_px = float(target_y)
            state.step9_color_mode = str(wf.cfg.rule_ac_color_mode)
            state.step9_color_h = int(wf.cfg.rule_ac_color_h)
            state.step9_color_s = int(wf.cfg.rule_ac_color_s)
            state.step9_color_v = int(wf.cfg.rule_ac_color_v)
            state.step9_color_h_tol = int(wf.cfg.rule_ac_color_h_tol)
            state.step9_color_s_tol = int(wf.cfg.rule_ac_color_s_tol)
            state.step9_color_v_tol = int(wf.cfg.rule_ac_color_v_tol)
            state.step9_color_min_area_px = int(wf.cfg.rule_ac_color_min_area_px)
            state.step9_color_morph_kernel = int(wf.cfg.rule_ac_color_morph_kernel)
            state.step9_center_tolerance_px = float(wf.cfg.rule_ac_center_tolerance_px)
            self._save_calibration_file_raw(state)
            self.log(
                f"[完整测量标定] Step9目标标定完成："
                f"target=({state.step9_target_x_px:.1f},{state.step9_target_y_px:.1f})；"
                f"颜色={self.workflow.cfg.rule_ac_color_mode}, HSV=({self.workflow.cfg.rule_ac_color_h},{self.workflow.cfg.rule_ac_color_s},{self.workflow.cfg.rule_ac_color_v})"
            )
        except Exception as e:
            self.log(f"[完整测量标定] Step9目标/颜色标定失败：{e}")
            self.log(traceback.format_exc())
            self.root.after(0, lambda: messagebox.showerror("Step9标定失败", str(e)))

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
            self.root.after(0, lambda: messagebox.showerror("保存/检查标定失败", str(e)))

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
            self.root.after(0, lambda: messagebox.showerror("加载标定失败", str(e)))

    def run_workflow_thread(self):
        if self.is_busy:
            messagebox.showwarning("正在运行", "当前已有任务在运行。")
            return
        self.worker_thread = self.run_in_thread(self.run_workflow)

    def run_workflow(self):
        self.is_busy = True

        try:
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

            # 2. 光谱仪 TCP 已禁用：完整测量不再检查 TCP Server / LabVIEW READY。
            wf.tcp_server = None
            wf.context["tcp_started"] = False
            wf.context["labview_ready"] = True
            self.set_var(self.tcp_status_var, "TCP状态：光谱仪TCP已禁用，完整测量跳过采集")
            self.log("[流程] 光谱仪 TCP 已禁用：跳过完整测量前的 TCP/READY 检查")

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

                if not should_continue:
                    break

                cycle_index += 1

            wf.finish()
            self.set_var(self.flow_status_var, "流程状态：测量结束")

        except Exception as e:
            self.set_var(self.flow_status_var, "流程状态：运行失败")
            self.log(f"[GUI错误] 运行失败：{e}")
            self.log(traceback.format_exc())
            self.root.after(0, lambda: messagebox.showerror("运行失败", str(e)))

        finally:
            self.is_busy = False

    def request_stop(self):
        # 同时停止完整测量线程和“单独A推动B”线程。
        # 单独A推动B不创建 MeasurementWorkflow，因此必须使用单独的停止标志。
        self.rule_ab_only_stop_requested = True
        self.rule_ab_controller_stop_requested = True

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
        self.log("[GUI] 已请求停止测量 / 单独A推动B / Step9 1/2轴")

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
            self.root.after(0, lambda: messagebox.showerror("停止1/2轴失败", str(e)))

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
            self.root.after(0, lambda: messagebox.showerror("照明光连接失败", str(e)))

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
            self.root.after(0, lambda: messagebox.showerror("激光控制器连接失败", str(e)))

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
            self.root.after(0, lambda: messagebox.showerror("角度检测初始化失败", str(e)))

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

            # 启动 TCP 前更新一次 TCP 配置
            wf.cfg.tcp_host = self.tcp_host_var.get().strip()
            wf.cfg.tcp_port = int(self.tcp_port_var.get())
            wf.cfg.tcp_output_dir = self.tcp_output_dir_var.get().strip()
            wf.cfg.tcp_command = self.tcp_command_var.get().strip()

            wf.start_tcp_server()
            self.set_var(self.tcp_status_var, "TCP状态：已启动，等待 LabVIEW 连接")
            self.refresh_result_labels(wf)

        except Exception as e:
            self.log(f"[TCP] 启动失败：{e}")
            self.root.after(0, lambda: messagebox.showerror("TCP启动失败", str(e)))

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
            self.root.after(0, lambda: messagebox.showerror("READY失败", str(e)))

    def measure_spectrum_once_thread(self):
        self.run_in_thread(self.measure_spectrum_once)

    def measure_spectrum_once(self):
        try:
            wf = self.ensure_workflow()
            result = wf.request_labview_spectrum(cycle_index=0)

            self.set_var(
                self.tcp_result_var,
                f"最近光谱：num_points={result.get('num_points')} "
                f"raw_peak={wf.context.get('raw_peak')} "
                f"fit_peak={wf.context.get('fit_peak')} "
                f"csv={result.get('csv_path')}",
            )
            self.set_var(self.tcp_status_var, "TCP状态：采集完成")
            self.refresh_result_labels(wf)

        except Exception as e:
            self.set_var(self.tcp_status_var, "TCP状态：采集失败")
            self.log(f"[TCP] 单次采集失败：{e}")
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
            self.log(f"[Step9-颜色中心] 已选定目标位置：x={target_x:.3f}, y={target_y:.3f}（截图区域内坐标）")
            self.set_var(self.rule_module_status_var, f"Step9目标已选定：({target_x:.1f}, {target_y:.1f})")
            return (target_x, target_y)

        except Exception as e:
            self.log(f"[Step9-颜色中心] 目标点选择失败：{e}")
            self.log(traceback.format_exc())
            self.root.after(0, lambda: messagebox.showerror("Step9目标选择失败", str(e)))
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
            self.root.after(0, lambda: messagebox.showerror("Step9测试失败", str(e)))

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

            self.set_var(self.flow_status_var, "流程状态：GUI清理完成，激光控制器已关闭")
            self.set_var(self.light_status_var, "照明光状态：GUI关闭不额外切换，保持当前状态")
            self.set_var(self.rigol_status_var, "激光控制器状态：已关闭")
            self.set_var(self.tcp_status_var, "TCP状态：已关闭")

        except Exception as e:
            self.log(f"[GUI] 关闭全部设备失败：{e}")


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
