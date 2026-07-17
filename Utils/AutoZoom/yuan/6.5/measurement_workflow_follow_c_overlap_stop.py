from __future__ import annotations

import sys
import csv
import json
import time
import math
import threading
import traceback
import copy
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import matplotlib.pyplot as plt
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

try:
    from config_conditional_second_detect import DEFAULT_CONFIG
except Exception:
    DEFAULT_CONFIG = {}

def _cfg(name: str, default: Any) -> Any:
    return DEFAULT_CONFIG.get(name, default)

from control.illumination_relay import IlluminationRelay
from control.signal_generator_rigol import RigolDG4062Controller
from control.labview_tcp_server import LabVIEWTCPServer

from vision.screen_capture import CaptureArea

from logic.angle_detect import ScreenAngleDetector
from logic.rule_ab_follow_c_overlap_stop import RuntimeConfig as RuleABRuntimeConfig
from logic.rule_ab_follow_c_overlap_stop import ActualNanoBoundaryFollower
from logic.rule_ac_fixed import RuleACConfig, RuleACOverlapController


@dataclass
class MeasurementConfig:
    max_cycles: int = int(_cfg("max_cycles", 10))

    # 这个不是照明光时间，而是信号发生器 CH1 ON 的持续时间
    signal_on_time_ms: float = float(_cfg("signal_on_time_ms", 50.0))

    # 照明光打开后等待时间
    stable_wait_ms: int = int(_cfg("stable_wait_ms", 1200))

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

    # 图2/图3 的横坐标来源：xlsx 第一张非空工作表的第一列数值
    # 例如 516中心波长(1).xlsx 中的中心波长序列。为空或读取失败时自动退回 index。
    x_axis_xlsx_path: str = str(_cfg("x_axis_xlsx_path", "516中心波长(1).xlsx"))

    enable_delta_w_judge: bool = bool(_cfg("enable_delta_w_judge", False))
    delta_w_threshold: float = float(_cfg("delta_w_threshold", 0.0))
    stop_when_delta_w_not_enough: bool = bool(_cfg("stop_when_delta_w_not_enough", False))

    # A 推动 B：调用 logic/rule_ab.py，每一步后检测 B 对边角度。
    rule_ab_enable_stage: bool = bool(_cfg("rule_ab_enable_stage", False))
    rule_ab_max_steps: int = int(_cfg("rule_ab_max_steps", 15))
    rule_ab_angle_delta_min_deg: float = float(_cfg("rule_ab_angle_delta_min_deg", 4.0))
    rule_ab_angle_delta_max_deg: float = float(_cfg("rule_ab_angle_delta_max_deg", 5.0))
    rule_ab_loop_interval_s: float = float(_cfg("rule_ab_loop_interval_s", 0.15))
    rule_ab_stage_conn: str = str(_cfg("rule_ab_stage_conn", "97101208"))
    rule_ab_stage_x_channel: int = int(_cfg("rule_ab_stage_x_channel", 3))
    rule_ab_stage_y_channel: int = int(_cfg("rule_ab_stage_y_channel", 4))
    # 新 RuleAB：A 沿 C 边界绕行，同时维持 A-C 间隙。
    # 如果 A 和 B 的 SAM2 mask 发生覆盖，则 Stage 停止，只持续检测 B 目标边角度；
    # 一旦 A/B 不再覆盖，又继续沿 C 边界运动。
    rule_ab_ab_close_threshold_px: float = float(_cfg("rule_ab_ab_close_threshold_px", 35.0))  # 旧字段，保留兼容，不再用于单独测试
    rule_ab_ab_overlap_min_area_px: float = float(_cfg("rule_ab_ab_overlap_min_area_px", 1.0))
    rule_ab_ac_target_clearance_px: float = float(_cfg("rule_ab_ac_target_clearance_px", 90.0))
    rule_ab_ac_min_clearance_px: float = float(_cfg("rule_ab_ac_min_clearance_px", 80.0))
    rule_ab_ac_max_clearance_px: float = float(_cfg("rule_ab_ac_max_clearance_px", 100.0))
    rule_ab_follow_c_direction: int = int(_cfg("rule_ab_follow_c_direction", 1))

    # Stage34 默认参数。为了兼容 logic.rule_ab.RuntimeConfig，仍保留单组默认速度/加速度。
    rule_ab_stage_velocity: int = int(_cfg("rule_ab_stage_velocity", 10))
    rule_ab_stage_acceleration: int = int(_cfg("rule_ab_stage_acceleration", 10))
    rule_ab_stage_max_voltage: int = int(_cfg("rule_ab_stage_max_voltage", 100))

    # 3/4 通道分别设置速度和加速度。
    rule_ab_stage_ch3_velocity: int = int(_cfg("rule_ab_stage_ch3_velocity", 10))
    rule_ab_stage_ch3_acceleration: int = int(_cfg("rule_ab_stage_ch3_acceleration", 10))
    rule_ab_stage_ch4_velocity: int = int(_cfg("rule_ab_stage_ch4_velocity", 10))
    rule_ab_stage_ch4_acceleration: int = int(_cfg("rule_ab_stage_ch4_acceleration", 10))

    rule_ab_stage_step_x: int = int(_cfg("rule_ab_stage_step_x", 100))
    rule_ab_stage_step_y: int = int(_cfg("rule_ab_stage_step_y", 100))
    rule_ab_stage_x_sign: int = int(_cfg("rule_ab_stage_x_sign", -1))
    rule_ab_stage_y_sign: int = int(_cfg("rule_ab_stage_y_sign", 1))

    # overlap 面积阈值：调用 logic/rule_ac.py 控制 1/2 通道。
    rule_ac_area_threshold_px: float = float(_cfg("rule_ac_area_threshold_px", 100.0))
    rule_ac_enable_stage: bool = bool(_cfg("rule_ac_enable_stage", False))
    rule_ac_max_cycles: int = int(_cfg("rule_ac_max_cycles", 50))
    rule_ac_stage_axis: str = str(_cfg("rule_ac_stage_axis", "horizontal"))
    rule_ac_stage_direction: int = int(_cfg("rule_ac_stage_direction", 1))
    rule_ac_stage_step_size: int = int(_cfg("rule_ac_stage_step_size", 100))
    rule_ac_stage_device_id: str = str(_cfg("rule_ac_stage_device_id", "97101208"))



# ============================================================
# 测量流程类
# ============================================================

class MeasurementWorkflow:
    def __init__(self, cfg: MeasurementConfig, on_log=None, on_update=None):
        self.cfg = cfg
        self.on_log = on_log
        self.on_update = on_update

        self.light: Optional[IlluminationRelay] = None
        self.signal_generator: Optional[RigolDG4062Controller] = None
        self.angle_module: Optional[ScreenAngleDetector] = None
        self.tcp_server: Optional[LabVIEWTCPServer] = None
        self.rule_ab_follower: Optional[ActualNanoBoundaryFollower] = None
        self.rule_ac_controller: Optional[RuleACOverlapController] = None

        self.is_measuring: bool = False
        self.stop_requested: bool = False

        # 异步保存相关：Step 12 保存数据放到独立线程，避免阻塞循环测量线程。
        # save_io_lock 用于串行化 CSV/XLSX 写入，防止多轮保存同时写同一个汇总文件。
        self.save_threads: List[threading.Thread] = []
        self.save_threads_lock = threading.Lock()
        self.save_io_lock = threading.Lock()

        # 每次点击“运行完整循环测量”都会创建一个 run_session_dir。
        # 本次点击产生的每一轮光谱 CSV 都直接保存在这个目录下，不再为每一轮创建子文件夹，也不再保存 JSON。
        self.run_session_dir: Optional[Path] = None
        self.run_session_name: Optional[str] = None

        # Step 11 使用“本轮角度”和“上一轮角度”的差值调节下一轮信号 ON 时间。
        # 第一轮没有上一轮角度，所以不调节。
        self.previous_cycle_angle: Optional[float] = None

        self.context: Dict[str, Any] = {
            "cycle_index": 0,
            "light_on": False,
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

            # GUI 实时绘图数据：x=本轮角度，y=本轮拟合峰值
            "plot_points": [],
        }

        # 每轮完成后追加：{"cycle_index": ..., "angle_deg": ..., "fit_peak": ...}
        self.plot_points: List[Dict[str, Any]] = []
        self.context["plot_points"] = self.plot_points

        self.output_root = Path(cfg.save_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

        # 当日汇总目录：measurement_output/save/MM.DD
        # 例如：measurement_output/save/05.15
        self.summary_date_dir = self._get_today_save_dir()
        self.summary_csv_path = self.summary_date_dir / "measurement_summary.csv"

        # 本次运行的 Excel 汇总文件：程序启动/创建 workflow 时生成，后续每一轮追加一行
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
    # 设备连接
    # --------------------------------------------------------

    def connect_measurement_devices(self):
        self.log("========== 初始化实验设备 ==========")

        self.connect_light()
        self.connect_signal_generator()
        self.configure_signal_generator()
        self.init_angle_module()

        self.log("========== 实验设备初始化完成 ==========")
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

    def connect_signal_generator(self):
        if self.signal_generator is not None:
            self.log("[Rigol] 已连接")
            return

        self.log(f"[Rigol] 正在连接：{self.cfg.rigol_visa}")
        self.signal_generator = RigolDG4062Controller(
            resource_name=self.cfg.rigol_visa,
            timeout_ms=self.cfg.rigol_timeout_ms,
            command_delay_s=self.cfg.rigol_command_delay_s,
        )
        self.signal_generator.open()
        self.log("[Rigol] 连接成功")

    def configure_signal_generator(self):
        if self.signal_generator is None:
            raise RuntimeError("Rigol 未连接")

        self.log("[Rigol] 配置 CH1")
        self.signal_generator.configure_pulse(
            channel=1,
            low_v=self.cfg.ch1_low_v,
            high_v=self.cfg.ch1_high_v,
            freq_hz=self.cfg.ch1_freq_hz,
            duty_percent=self.cfg.ch1_duty_percent,
            delay_s=self.cfg.ch1_delay_s,
            edge_time_s=None,
            output_on=False,
        )

        self.log("[Rigol] 配置 CH2")
        self.signal_generator.configure_pulse(
            channel=2,
            low_v=self.cfg.ch2_low_v,
            high_v=self.cfg.ch2_high_v,
            freq_hz=self.cfg.ch2_freq_hz,
            duty_percent=self.cfg.ch2_duty_percent,
            delay_s=self.cfg.ch2_delay_s,
            edge_time_s=None,
            output_on=False,
        )

        self.context["signal_ch1_on"] = False
        self.context["signal_ch2_on"] = False
        self.log("[Rigol] CH1 参数配置完成；CH2 参数配置完成")

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

        self.log("[角度检测] 初始化完成；已调用 logic.angle_detect.ScreenAngleDetector，YOLO .pt 模型后续循环不会重复加载")

    def start_tcp_server(self):
        if self.tcp_server is not None and getattr(self.tcp_server, "is_server_running", False):
            self.log("[TCP] Server 已经运行")
            self.context["tcp_started"] = True
            self.notify_update()
            return

        self.log(f"[TCP] 启动 Python TCP Server: {self.cfg.tcp_host}:{self.cfg.tcp_port}")

        self.tcp_server = LabVIEWTCPServer(
            host=self.cfg.tcp_host,
            port=self.cfg.tcp_port,
            output_dir=self.cfg.tcp_output_dir,
            on_log=self.log,
        )

        self.tcp_server.start_server_async()

        self.context["tcp_started"] = True
        self.context["labview_ready"] = False

        self.log("[TCP] Server 已启动，等待 LabVIEW 连接")
        self.notify_update()

    def wait_labview_ready(self):
        if self.tcp_server is None:
            self.context["tcp_started"] = False
            self.context["labview_ready"] = False
            self.notify_update()
            raise RuntimeError("TCP Server 未启动")

        self.log("[TCP] 等待 LabVIEW READY")
        result = self.tcp_server.wait_for_ready()

        if not result.get("ok", False):
            self.context["labview_ready"] = False
            self.notify_update()
            raise RuntimeError(f"LabVIEW READY 失败：{result}")

        self.context["tcp_started"] = True
        self.context["labview_ready"] = True

        self.log("[TCP] LabVIEW READY，可以开始测量")
        self.notify_update()
        return result

    # --------------------------------------------------------
    # 设备基础动作
    # --------------------------------------------------------

    def signal_ch1_on(self):
        if self.signal_generator is None:
            raise RuntimeError("Rigol 未连接")

        self.signal_generator.on(channel=1)
        self.context["signal_ch1_on"] = True
        self.log("[Rigol] CH1 ON")
        self.notify_update()

    def signal_ch1_off(self):
        if self.signal_generator is None:
            raise RuntimeError("Rigol 未连接")

        self.signal_generator.off(channel=1)
        self.context["signal_ch1_on"] = False
        self.log("[Rigol] CH1 OFF")
        self.notify_update()

    def signal_ch2_on(self):
        if self.signal_generator is None:
            raise RuntimeError("Rigol 未连接")

        self.signal_generator.on(channel=2)
        self.context["signal_ch2_on"] = True
        self.log("[Rigol] CH2 ON")
        self.notify_update()

    def signal_ch2_off(self):
        if self.signal_generator is None:
            raise RuntimeError("Rigol 未连接")

        self.signal_generator.off(channel=2)
        self.context["signal_ch2_on"] = False
        self.log("[Rigol] CH2 OFF")
        self.notify_update()

    def signal_all_on(self):
        """
        兼容旧调用：不再输出合并日志，而是顺序执行 CH1 ON、CH2 ON。
        GUI 和完整循环测量不再调用该函数作为主要入口。
        """
        self.signal_ch1_on()
        self.signal_ch2_on()

    def signal_all_off(self):
        """
        兼容旧调用：不再输出合并日志，而是顺序执行 CH1 OFF、CH2 OFF。
        """
        self.signal_ch1_off()
        self.signal_ch2_off()

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

        cfg = RuleABRuntimeConfig(
            capture_area=self.cfg.capture_area,
            output_dir=str(self.output_root / "rule_ab_from_measurement"),
            sam2_device="cuda",
            enable_stage=bool(self.cfg.rule_ab_enable_stage),
            max_cycles=int(self.cfg.rule_ab_max_steps),
            loop_interval_s=float(self.cfg.rule_ab_loop_interval_s),

            # 新 RuleAB：沿 C 边界绕行并维持 A-C 间隙；A/B 覆盖时停止运动只测角度。
            a_c_target_clearance=float(self.cfg.rule_ab_ac_target_clearance_px),
            a_c_min_clearance=float(self.cfg.rule_ab_ac_min_clearance_px),
            a_c_max_clearance=float(self.cfg.rule_ab_ac_max_clearance_px),
            follow_c_direction=int(self.cfg.rule_ab_follow_c_direction),
            a_b_overlap_stop=True,
            a_b_overlap_min_area_px=float(self.cfg.rule_ab_ab_overlap_min_area_px),

            # Thorlabs KinesisPiezoMotor 控制器与通道参数。
            # 这里显式传入，避免 logic/rule_ab.py 使用旧默认序列号。
            stage_conn=str(self.cfg.rule_ab_stage_conn),
            stage_x_channel=int(self.cfg.rule_ab_stage_x_channel),
            stage_y_channel=int(self.cfg.rule_ab_stage_y_channel),
            stage_default_velocity=int(self.cfg.rule_ab_stage_velocity),
            stage_default_acceleration=int(self.cfg.rule_ab_stage_acceleration),
            stage_default_max_voltage=int(self.cfg.rule_ab_stage_max_voltage),
            stage_step_x=int(self.cfg.rule_ab_stage_step_x),
            stage_step_y=int(self.cfg.rule_ab_stage_step_y),
            stage_x_sign=int(self.cfg.rule_ab_stage_x_sign),
            stage_y_sign=int(self.cfg.rule_ab_stage_y_sign),
        )
        self.log(
            "[RuleAB] 初始化 A推动B 模块："
            f"enable_stage={cfg.enable_stage}, max_steps={cfg.max_cycles}"
        )
        self.rule_ab_follower = ActualNanoBoundaryFollower(cfg)
        self.rule_ab_follower.initialize_abc_with_first_frame()
        return self.rule_ab_follower

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
        records: List[Dict[str, Any]] = []
        reached = False
        final_angle = None
        final_delta = None

        for step_idx in range(1, int(self.cfg.rule_ab_max_steps) + 1):
            if self.stop_requested:
                break

            self.log(f"[RuleAB] step={step_idx}: 执行一次 rule_ab.run_one_cycle()")
            ok = follower.run_one_cycle()

            angle_result = self.detect_angle_once(
                label=f"rule_ab_edge_step_{step_idx}",
                allow_fail=True,
            )
            if angle_result.get("ok", False) and angle_result.get("angle_deg") is not None:
                final_angle = float(angle_result["angle_deg"])
                if baseline_angle is not None:
                    final_delta = self.angle_diff_deg(final_angle, float(baseline_angle))
                else:
                    final_delta = None
            else:
                final_angle = None
                final_delta = None

            record = {
                "step": step_idx,
                "rule_ab_ok": ok,
                "angle_deg": final_angle,
                "angle_delta_from_baseline": final_delta,
                "angle_result": self._json_safe(angle_result),
            }
            records.append(record)

            self.log(
                f"[RuleAB] step={step_idx}, angle={final_angle}, "
                f"delta_from_current={final_delta}, target=[{min_delta}, {max_delta}]"
            )

            if final_delta is not None and final_delta >= min_delta:
                reached = True
                if final_delta > max_delta:
                    self.log(
                        f"[RuleAB] 角度变化 {final_delta:.3f}° 已超过上限 {max_delta:.3f}°，立即停止 CH1"
                    )
                else:
                    self.log(f"[RuleAB] 角度变化进入目标范围：{final_delta:.3f}°")
                break

            if not ok:
                self.log("[RuleAB] rule_ab 返回 False，停止 A推动B 模块")
                break

            time.sleep(float(self.cfg.rule_ab_loop_interval_s))

        result = {
            "ok": reached,
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

    def ensure_rule_ac_controller(self) -> RuleACOverlapController:
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
            "[RuleAC] 初始化 overlap面积控制模块："
            f"threshold_px={cfg.area_threshold_px}, enable_stage={cfg.enable_stage}"
        )
        self.rule_ac_controller = RuleACOverlapController(cfg)
        return self.rule_ac_controller

    def run_rule_ac_until_threshold(self) -> Dict[str, Any]:
        self.log("========== RuleAC：检测颜色区域，小于阈值则移动1/2通道 ==========")
        controller = self.ensure_rule_ac_controller()

        last_continue = True
        for _ in range(int(self.cfg.rule_ac_max_cycles)):
            if self.stop_requested:
                break
            last_continue = controller.run_one_cycle()
            if not last_continue:
                break

        result = {
            "ok": not last_continue,
            "history_len": len(controller.history),
            "last_row": controller.history[-1] if controller.history else None,
            "csv_path": str(controller.csv_path),
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
        self.log("========== LabVIEW 采集光谱 ==========")

        if self.tcp_server is None:
            raise RuntimeError("TCP Server 未启动")

        result = self.tcp_server.request_measure(
            command=self.cfg.tcp_command,
            index=cycle_index,
            save_csv=True,
        )

        self.context["last_tcp_result"] = result

        if not result.get("ok", False):
            raise RuntimeError(f"LabVIEW 采集失败：{result}")

        self.log(f"[LabVIEW] 采集完成：index={result.get('index')}")
        self.log(f"[LabVIEW] num_points={result.get('num_points')}")
        self.log(f"[LabVIEW] csv_path={result.get('csv_path')}")

        parsed = self._parse_labview_result(result)

        raw_values = parsed.get("raw_values") or []
        threshold_values = parsed.get("threshold_filtered_intensity") or []
        median_values = parsed.get("median_filtered_intensity") or []

        # 图2/图3 横坐标：优先使用 GUI 选择的 xlsx 第一列；读取失败则绘图时退回 index
        x_axis_values = self.load_x_axis_values_from_xlsx()

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
        self.context["fit_values"] = parsed.get("fit_values")
        self.context["fit_params"] = parsed.get("fit_params")

        self.log(f"[光谱] 原始点数={len(raw_values)}")
        self.log(
            f"[光谱] 阈值去除后点数={len(threshold_values)}，"
            f"阈值={self.cfg.raw_remove_above}"
        )
        self.log(
            f"[光谱] {self.cfg.median_filter_window}点中值滤波后点数={len(median_values)}"
        )
        self.log(f"[光谱] raw_original_peak={self.context['raw_original_peak']}")
        self.log(f"[光谱] raw_peak_after_processing={self.context['raw_peak']}")
        self.log(f"[光谱] fit_peak={self.context['fit_peak']}")

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

            "raw_original_peak": ctx.get("raw_original_peak"),
            "raw_filtered_peak": ctx.get("raw_filtered_peak"),
            "raw_median_peak": ctx.get("raw_median_peak"),
            "raw_peak": ctx.get("raw_peak"),
            "fit_peak": ctx.get("fit_peak"),
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
                self.log("[保存] 本轮没有可保存的光谱数组，跳过光谱 CSV 写入")

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
                ctx.get("raw_original_peak"),
                ctx.get("raw_peak"),
                ctx.get("fit_peak"),
                paths.get("run_session_name"),
                paths.get("spectrum_filename"),
                labview_result.get("csv_path"),
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

        # “本次角度”采用本轮最终角度 angle_after；如果 angle_after 无效，退回 angle_before。
        angle_value = ctx.get("angle_after")
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
        Step 6：信号发生器 CH1 ON
        Step 7：A推动B（logic/rule_ab.py），并检测B对边角度
        Step 8：当对边角度移动 1-2° 时，CH1 OFF
        Step 9：检测颜色区域（logic/rule_ac.py），小于面积阈值则移动1/2通道
        Step 10：保存数据
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
            angle_result = self.detect_angle_once(label="current_angle", allow_fail=True)

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

            if self.stop_requested:
                return False

            self.log("========== Step 6：信号发生器 CH1 ON ==========")
            self.signal_ch1_on()

            self.log("========== Step 7：A推动B，并检测B对边角度 ==========")
            try:
                rule_ab_result = self.run_rule_ab_until_angle_delta(
                    baseline_angle=current_angle,
                    min_delta_deg=self.cfg.rule_ab_angle_delta_min_deg,
                    max_delta_deg=self.cfg.rule_ab_angle_delta_max_deg,
                )
            finally:
                self.log("========== Step 8：信号发生器 CH1 OFF ==========")
                try:
                    self.signal_ch1_off()
                except Exception as e:
                    self.log(f"[Rigol] CH1 OFF 失败：{e}")

            self.context["last_rule_ab_result"] = rule_ab_result
            self.notify_update()

            if self.stop_requested:
                return False

            self.log("========== Step 9：检测颜色区域，必要时移动1/2通道 ==========")
            rule_ac_result = self.run_rule_ac_until_threshold()
            self.context["last_rule_ac_result"] = rule_ac_result
            self.notify_update()

            self.log("========== Step 10：保存数据（异步线程） ==========")
            t_save_submit0 = time.time()
            self.start_save_cycle_result_async(
                cycle_index=cycle_index,
                paths=paths,
                angle_before_result=angle_result,
                angle_after_result={"ok": False, "reason": "single_angle_mode_no_angle_after", "angle_deg": None},
                labview_result=labview_result,
            )
            self.log(f"[Step 10] 异步保存任务提交耗时：{time.time() - t_save_submit0:.3f} s")

            self.log(f"[流程] 第 {cycle_index} 轮保存任务已提交，可以进入下一轮")
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

        if self.tcp_server is None or not self.context.get("tcp_started", False):
            raise RuntimeError("TCP Server 未启动。请先点击“启动TCP”，运行 LabVIEW，并点击“等待READY”。")

        if not self.context.get("labview_ready", False):
            raise RuntimeError("LabVIEW 尚未 READY。请先确认 LabVIEW 已连接，并点击“等待READY”。")

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

        # 用户要求：停止测量后，信号发生器保持关闭状态。
        # 这里立即依次关闭 CH1 和 CH2；如果流程线程随后进入 finish()，finish() 也会再次兜底关闭。
        try:
            if self.signal_generator is not None:
                self.signal_all_off()
                self.log("[流程] 停止测量：Rigol CH1 已 OFF，CH2 已 OFF")
        except Exception as e:
            self.log(f"[流程] 停止测量时关闭 Rigol 失败：{e}")

        self.save_summary_xlsx()
        self.notify_update()

    def finish(self):
        self.log("========== 结束测量 ==========")
        self.is_measuring = False
        self.context["is_measuring"] = False

        # 用户要求：停止测量/流程结束后，信号发生器保持关闭状态。
        # 因此这里不再保持输出打开，而是兜底关闭 CH1 和 CH2。
        try:
            if self.signal_generator is not None:
                self.signal_all_off()
                self.log("[流程] 测量结束，Rigol CH1 已 OFF，CH2 已 OFF")
        except Exception as e:
            self.log(f"[流程] 保持 Rigol OFF 失败：{e}")

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
            1. 关闭 GUI 后，信号发生器 CH1 和 CH2 必须保持 OFF；
            2. 照明光不在本次修改范围内，关闭 GUI 时不额外发送 ON/OFF；
            3. TCP 连接需要关闭，释放端口。
        """
        self.log("========== 关闭GUI：Rigol保持OFF，照明光保持当前状态 ==========")
        self.save_summary_xlsx()

        try:
            if self.signal_generator is not None:
                self.signal_all_off()
                self.log("[Rigol] GUI关闭时已依次发送 CH1 OFF、CH2 OFF，保持关闭状态")
        except Exception as e:
            self.log(f"[Rigol] GUI关闭时关闭输出失败：{e}")

        try:
            if self.light is not None:
                self.log("[照明光] GUI关闭时不额外发送 ON/OFF，保持当前照明状态")
        except Exception as e:
            self.log(f"[照明光] GUI关闭保持状态提示失败：{e}")

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

        self.log("========== GUI关闭清理完成：Rigol已OFF，照明光未额外切换 ==========")
        self.notify_update()

# ============================================================
# GUI
# ============================================================

class MeasurementWorkflowGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("循环测量 GUI：照明光 + Rigol + ScreenAngleDetector角度检测 + LabVIEW TCP光谱")
        self.root.geometry("1320x1080")

        self.workflow: Optional[MeasurementWorkflow] = None
        self.worker_thread: Optional[threading.Thread] = None
        self.is_busy = False
        self.rule_ab_only_stop_requested = False

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

        ttk.Label(flow_frame, textvariable=self.flow_status_var, wraplength=460).pack(fill=tk.X, padx=4, pady=(6, 0))

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
        # 左侧 4：Rigol 信号发生器
        # =====================================================
        rigol_frame = ttk.LabelFrame(left_inner, text="4. Rigol DG4062 信号发生器", padding=10, style="Panel.TLabelframe")
        rigol_frame.pack(fill=tk.X, padx=4, pady=8)

        self.rigol_visa_var = tk.StringVar(value=cfg0.rigol_visa)
        self.rigol_status_var = tk.StringVar(value="Rigol状态：未连接")

        rigol_frame.columnconfigure(1, weight=1)
        ttk.Label(rigol_frame, text="VISA").grid(row=0, column=0, padx=4, pady=4, sticky="w")
        ttk.Entry(rigol_frame, textvariable=self.rigol_visa_var).grid(row=0, column=1, columnspan=2, padx=4, pady=4, sticky="ew")

        rigol_buttons = ttk.Frame(rigol_frame)
        rigol_buttons.grid(row=1, column=0, columnspan=3, padx=2, pady=4, sticky="ew")
        for i in range(2):
            rigol_buttons.columnconfigure(i, weight=1)
        ttk.Button(rigol_buttons, text="单独连接", command=self.connect_rigol_thread).grid(row=0, column=0, padx=4, pady=3, sticky="ew")
        ttk.Button(rigol_buttons, text="配置CH1和CH2", command=self.configure_rigol_thread).grid(row=0, column=1, padx=4, pady=3, sticky="ew")
        ttk.Button(rigol_buttons, text="CH1 ON", command=self.signal_ch1_on_thread).grid(row=1, column=0, padx=4, pady=3, sticky="ew")
        ttk.Button(rigol_buttons, text="CH1 OFF", command=self.signal_ch1_off_thread).grid(row=1, column=1, padx=4, pady=3, sticky="ew")
        ttk.Button(rigol_buttons, text="CH2 ON", command=self.signal_ch2_on_thread).grid(row=2, column=0, padx=4, pady=3, sticky="ew")
        ttk.Button(rigol_buttons, text="CH2 OFF", command=self.signal_ch2_off_thread).grid(row=2, column=1, padx=4, pady=3, sticky="ew")
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

        self.rule_ab_enable_stage_var = tk.BooleanVar(value=cfg0.rule_ab_enable_stage)

        # 单独 A推动B 不再使用最大推动次数；运行后会持续推动/监测，直到点击“停止测量”。
        # 这里保留内部变量只是为了兼容完整测量流程和旧配置，不在 GUI 中显示。
        self.rule_ab_max_steps_var = tk.IntVar(value=cfg0.rule_ab_max_steps)

        # 旧 A-B 距离阈值变量保留兼容，但单独测试不再使用距离停止。
        self.rule_ab_ab_close_threshold_var = tk.DoubleVar(value=cfg0.rule_ab_ab_close_threshold_px)
        self.rule_ab_ab_overlap_threshold_var = tk.DoubleVar(value=cfg0.rule_ab_ab_overlap_min_area_px)
        self.rule_ab_ac_target_clearance_var = tk.DoubleVar(value=cfg0.rule_ab_ac_target_clearance_px)
        self.rule_ab_ac_min_clearance_var = tk.DoubleVar(value=cfg0.rule_ab_ac_min_clearance_px)
        self.rule_ab_ac_max_clearance_var = tk.DoubleVar(value=cfg0.rule_ab_ac_max_clearance_px)
        self.rule_ab_follow_c_direction_var = tk.IntVar(value=cfg0.rule_ab_follow_c_direction)

        self.rule_ab_ch3_velocity_var = tk.IntVar(value=cfg0.rule_ab_stage_ch3_velocity)
        self.rule_ab_ch3_acceleration_var = tk.IntVar(value=cfg0.rule_ab_stage_ch3_acceleration)
        self.rule_ab_ch4_velocity_var = tk.IntVar(value=cfg0.rule_ab_stage_ch4_velocity)
        self.rule_ab_ch4_acceleration_var = tk.IntVar(value=cfg0.rule_ab_stage_ch4_acceleration)

        self.rule_ab_delta_min_var = tk.DoubleVar(value=cfg0.rule_ab_angle_delta_min_deg)
        self.rule_ab_delta_max_var = tk.DoubleVar(value=cfg0.rule_ab_angle_delta_max_deg)

        self.rule_ac_enable_stage_var = tk.BooleanVar(value=cfg0.rule_ac_enable_stage)
        self.rule_ac_area_threshold_var = tk.DoubleVar(value=cfg0.rule_ac_area_threshold_px)
        self.rule_ac_max_cycles_var = tk.IntVar(value=cfg0.rule_ac_max_cycles)
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

        ttk.Label(rule_frame, text="AB覆盖阈值px² / 沿C方向").grid(row=2, column=0, padx=4, pady=3, sticky="w")
        ttk.Entry(rule_frame, textvariable=self.rule_ab_ab_overlap_threshold_var, width=8).grid(row=2, column=1, padx=4, pady=3, sticky="ew")
        ttk.Entry(rule_frame, textvariable=self.rule_ab_follow_c_direction_var, width=8).grid(row=2, column=2, padx=4, pady=3, sticky="ew")
        ttk.Label(rule_frame, text="覆盖时Stage不动；无覆盖时沿C绕行").grid(row=2, column=3, padx=4, pady=3, sticky="w")

        ttk.Label(rule_frame, text="CH3 速度/加速度").grid(row=3, column=0, padx=4, pady=3, sticky="w")
        ttk.Entry(rule_frame, textvariable=self.rule_ab_ch3_velocity_var, width=8).grid(row=3, column=1, padx=4, pady=3, sticky="ew")
        ttk.Entry(rule_frame, textvariable=self.rule_ab_ch3_acceleration_var, width=8).grid(row=3, column=2, padx=4, pady=3, sticky="ew")

        ttk.Label(rule_frame, text="CH4 速度/加速度").grid(row=4, column=0, padx=4, pady=3, sticky="w")
        ttk.Entry(rule_frame, textvariable=self.rule_ab_ch4_velocity_var, width=8).grid(row=4, column=1, padx=4, pady=3, sticky="ew")
        ttk.Entry(rule_frame, textvariable=self.rule_ab_ch4_acceleration_var, width=8).grid(row=4, column=2, padx=4, pady=3, sticky="ew")

        ttk.Label(rule_frame, text="完整测量角度目标min/max").grid(row=5, column=0, padx=4, pady=3, sticky="w")
        ttk.Entry(rule_frame, textvariable=self.rule_ab_delta_min_var, width=8).grid(row=5, column=1, padx=4, pady=3, sticky="ew")
        ttk.Entry(rule_frame, textvariable=self.rule_ab_delta_max_var, width=8).grid(row=5, column=2, padx=4, pady=3, sticky="ew")

        ttk.Checkbutton(rule_frame, text="RuleAC真动Stage12", variable=self.rule_ac_enable_stage_var).grid(
            row=6, column=0, columnspan=2, padx=4, pady=(8, 3), sticky="w"
        )
        ttk.Label(rule_frame, text="面积阈值px²").grid(row=7, column=0, padx=4, pady=3, sticky="w")
        ttk.Entry(rule_frame, textvariable=self.rule_ac_area_threshold_var, width=10).grid(row=7, column=1, padx=4, pady=3, sticky="ew")
        ttk.Label(rule_frame, text="AC最大循环").grid(row=7, column=2, padx=4, pady=3, sticky="w")
        ttk.Entry(rule_frame, textvariable=self.rule_ac_max_cycles_var, width=10).grid(row=7, column=3, padx=4, pady=3, sticky="ew")

        rule_buttons = ttk.Frame(rule_frame)
        rule_buttons.grid(row=9, column=0, columnspan=4, padx=2, pady=6, sticky="ew")
        rule_buttons.columnconfigure(0, weight=1)
        rule_buttons.columnconfigure(1, weight=1)
        ttk.Button(rule_buttons, text="单独A沿C绕行", command=self.test_rule_ab_thread).grid(row=0, column=0, padx=4, pady=3, sticky="ew")
        ttk.Button(rule_buttons, text="单独测试Step9面积", command=self.test_rule_ac_thread).grid(row=0, column=1, padx=4, pady=3, sticky="ew")
        ttk.Label(rule_frame, textvariable=self.rule_module_status_var, wraplength=450).grid(
            row=9, column=0, columnspan=4, padx=4, pady=(4, 0), sticky="w"
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
            rule_ab_follow_c_direction=int(self.rule_ab_follow_c_direction_var.get()),

            # 单独 A沿C绕行 / 完整流程 RuleAB 默认使用你的 Thorlabs 控制器 97101208 和 3/4 通道。
            rule_ab_stage_conn="97101208",
            rule_ab_stage_x_channel=3,
            rule_ab_stage_y_channel=4,

            # 兼容 RuntimeConfig 的默认速度/加速度，同时按 CH3/CH4 分别设置实际通道参数。
            rule_ab_stage_velocity=int(self.rule_ab_ch3_velocity_var.get()),
            rule_ab_stage_acceleration=int(self.rule_ab_ch3_acceleration_var.get()),
            rule_ab_stage_max_voltage=100,
            rule_ab_stage_ch3_velocity=int(self.rule_ab_ch3_velocity_var.get()),
            rule_ab_stage_ch3_acceleration=int(self.rule_ab_ch3_acceleration_var.get()),
            rule_ab_stage_ch4_velocity=int(self.rule_ab_ch4_velocity_var.get()),
            rule_ab_stage_ch4_acceleration=int(self.rule_ab_ch4_acceleration_var.get()),
            rule_ab_stage_step_x=100,
            rule_ab_stage_step_y=100,
            rule_ab_stage_x_sign=-1,
            rule_ab_stage_y_sign=1,

            rule_ac_area_threshold_px=float(self.rule_ac_area_threshold_var.get()),
            rule_ac_enable_stage=bool(self.rule_ac_enable_stage_var.get()),
            rule_ac_max_cycles=int(self.rule_ac_max_cycles_var.get()),
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
            old_cfg.capture_area != new_cfg.capture_area
        )

        rule_ac_param_changed = (
            old_cfg.rule_ac_area_threshold_px != new_cfg.rule_ac_area_threshold_px or
            old_cfg.rule_ac_enable_stage != new_cfg.rule_ac_enable_stage or
            old_cfg.rule_ac_max_cycles != new_cfg.rule_ac_max_cycles or
            old_cfg.capture_area != new_cfg.capture_area
        )

        # 核心：把 GUI 参数同步到 workflow.cfg。
        self.workflow.cfg = new_cfg

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

        # 如果 Rigol 已连接，且 Rigol 参数发生变化，则重新配置 CH1 和 CH2。
        if rigol_param_changed and self.workflow.signal_generator is not None:
            self.log("[配置同步] Rigol 参数已变化，重新配置 CH1 和 CH2")
            self.workflow.configure_signal_generator()

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
        ch1 = "ON" if ctx.get("signal_ch1_on") else "OFF"
        ch2 = "ON" if ctx.get("signal_ch2_on") else "OFF"

        if wf.light is not None:
            self.set_var(self.light_status_var, f"照明光状态：已连接 / {light_state}")
        else:
            self.set_var(self.light_status_var, "照明光状态：未连接")

        if wf.signal_generator is not None:
            self.set_var(self.rigol_status_var, f"Rigol状态：已连接，CH1 {ch1}，CH2 {ch2}")
        else:
            self.set_var(self.rigol_status_var, "Rigol状态：未连接")

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
            self.set_var(self.rigol_status_var, "Rigol状态：已连接，CH1已配置，CH2已配置")
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

            # 每次点击“运行完整循环测量”都新建一个总文件夹，并清空右侧列表/本次绘图点。
            wf.begin_new_run_session()
            self.root.after(0, lambda: self.update_angle_fit_xy_list([]))

            # 1. 确保实验设备已经初始化
            if wf.light is None or wf.signal_generator is None or wf.angle_module is None:
                self.log("[流程] 实验设备尚未完整初始化，先初始化实验设备")
                wf.connect_measurement_devices()

            # 2. TCP 不在这里自动启动，只检查
            if wf.tcp_server is None or not wf.context.get("tcp_started", False):
                raise RuntimeError(
                    "TCP Server 未启动。请先点击“启动TCP”，运行 LabVIEW，并点击“等待READY”。"
                )

            if not wf.context.get("labview_ready", False):
                raise RuntimeError(
                "LabVIEW 尚未 READY。请先确认 LabVIEW 已连接，并点击“等待READY”。"
                )

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

        if self.workflow is not None:
            self.workflow.request_stop()
        self.set_var(self.flow_status_var, "流程状态：已请求停止")
        self.log("[GUI] 已请求停止测量 / 单独A推动B")

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
        try:
            wf = self.ensure_workflow()
            wf.connect_signal_generator()
        except Exception as e:
            self.log(f"[Rigol] 连接失败：{e}")
            self.root.after(0, lambda: messagebox.showerror("Rigol连接失败", str(e)))

    def configure_rigol_thread(self):
        self.run_in_thread(self.configure_rigol)

    def configure_rigol(self):
        try:
            wf = self.ensure_workflow()
            if wf.signal_generator is None:
                wf.connect_signal_generator()
            wf.configure_signal_generator()
            self.refresh_result_labels(wf)
        except Exception as e:
            self.log(f"[Rigol] 配置失败：{e}")

    def signal_ch1_on_thread(self):
        self.run_in_thread(self.signal_ch1_on)

    def signal_ch1_on(self):
        try:
            wf = self.ensure_workflow()
            if wf.signal_generator is None:
                wf.connect_signal_generator()
            wf.signal_ch1_on()
        except Exception as e:
            self.log(f"[Rigol] CH1 ON失败：{e}")

    def signal_ch1_off_thread(self):
        self.run_in_thread(self.signal_ch1_off)

    def signal_ch1_off(self):
        try:
            wf = self.ensure_workflow()
            if wf.signal_generator is None:
                wf.connect_signal_generator()
            wf.signal_ch1_off()
        except Exception as e:
            self.log(f"[Rigol] CH1 OFF失败：{e}")

    def signal_ch2_on_thread(self):
        self.run_in_thread(self.signal_ch2_on)

    def signal_ch2_on(self):
        try:
            wf = self.ensure_workflow()
            if wf.signal_generator is None:
                wf.connect_signal_generator()
            wf.signal_ch2_on()
        except Exception as e:
            self.log(f"[Rigol] CH2 ON失败：{e}")

    def signal_ch2_off_thread(self):
        self.run_in_thread(self.signal_ch2_off)

    def signal_ch2_off(self):
        try:
            wf = self.ensure_workflow()
            if wf.signal_generator is None:
                wf.connect_signal_generator()
            wf.signal_ch2_off()
        except Exception as e:
            self.log(f"[Rigol] CH2 OFF失败：{e}")

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

    def test_rule_ab_thread(self):
        self.run_in_thread(self.test_rule_ab)

    def test_rule_ab(self):
        """
        单独点击按钮时，只执行新的 RuleAB 视觉闭环：

            1. SAM2 每帧分割/更新 A、B、C；
            2. 不再根据 A-B 距离控制；
            3. A 沿 C 的局部切向运动一圈，并持续维持 A-C 距离在 min~max 内；
            4. 每一帧都检测 B 目标边角度；
            5. 如果 A 和 B mask 有覆盖，Stage 立即停止，只检测 B 角度；
            6. 如果 A/B 不再覆盖，则恢复沿 C 绕行和 A-C 距离控制；
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
            max_voltage: int,
        ):
            try:
                stage = getattr(follower, "stage", None)
                if stage is None:
                    self.log("[RuleAB-C] enable_stage=False 或 stage=None，跳过 CH3/CH4 单独速度设置")
                    return

                if hasattr(stage, "setup_channel"):
                    stage.setup_channel(
                        channel=3,
                        max_voltage=int(max_voltage),
                        velocity=int(ch3_velocity),
                        acceleration=int(ch3_acceleration),
                    )
                    stage.setup_channel(
                        channel=4,
                        max_voltage=int(max_voltage),
                        velocity=int(ch4_velocity),
                        acceleration=int(ch4_acceleration),
                    )
                    self.log(
                        "[RuleAB-C] 已分别设置 Stage34："
                        f"CH3 velocity={ch3_velocity}, acceleration={ch3_acceleration}; "
                        f"CH4 velocity={ch4_velocity}, acceleration={ch4_acceleration}; "
                        f"max_voltage={max_voltage}"
                    )
                    return

                dev = getattr(stage, "dev", None)
                if dev is not None and hasattr(dev, "setup_drive"):
                    dev.setup_drive(
                        max_voltage=int(max_voltage),
                        velocity=int(ch3_velocity),
                        acceleration=int(ch3_acceleration),
                        channel=3,
                    )
                    dev.setup_drive(
                        max_voltage=int(max_voltage),
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
            self.set_var(self.rule_module_status_var, "规则模块状态：A沿C绕行初始化中")
            self.log("========== 单独A沿C绕行：覆盖B时停，只检测B角度；无覆盖时维持A-C距离并沿C运动 ==========")

            cfg_gui = self.build_config_from_ui()

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

            stage_default_velocity = int(getattr(cfg_gui, "rule_ab_stage_velocity", stage_ch3_velocity))
            stage_default_acceleration = int(getattr(cfg_gui, "rule_ab_stage_acceleration", stage_ch3_acceleration))
            stage_max_voltage = int(getattr(cfg_gui, "rule_ab_stage_max_voltage", 100))

            stage_step_x = int(getattr(cfg_gui, "rule_ab_stage_step_x", 100))
            stage_step_y = int(getattr(cfg_gui, "rule_ab_stage_step_y", 100))
            stage_x_sign = int(getattr(cfg_gui, "rule_ab_stage_x_sign", -1))
            stage_y_sign = int(getattr(cfg_gui, "rule_ab_stage_y_sign", 1))

            ac_target = float(getattr(cfg_gui, "rule_ab_ac_target_clearance_px", 90.0))
            ac_min = float(getattr(cfg_gui, "rule_ab_ac_min_clearance_px", 80.0))
            ac_max = float(getattr(cfg_gui, "rule_ab_ac_max_clearance_px", 100.0))
            follow_c_direction = int(getattr(cfg_gui, "rule_ab_follow_c_direction", 1))
            ab_overlap_threshold = float(getattr(cfg_gui, "rule_ab_ab_overlap_min_area_px", 1.0))

            output_dir = str(
                Path(save_root)
                / "rule_ab_follow_c_only"
                / datetime.now().strftime("run_%Y%m%d_%H%M%S")
            )

            self.log(
                "[RuleAB-C] 参数："
                f"enable_stage={rule_ab_enable_stage}, loop_interval_s={rule_ab_loop_interval_s}, "
                f"A-C[min,target,max]=[{ac_min}, {ac_target}, {ac_max}] px, "
                f"AB_overlap_threshold={ab_overlap_threshold}px², follow_c_direction={follow_c_direction}, "
                f"stage_conn={stage_conn}, x_channel={stage_x_channel}, y_channel={stage_y_channel}, "
                f"CH3 velocity={stage_ch3_velocity}, CH3 acceleration={stage_ch3_acceleration}, "
                f"CH4 velocity={stage_ch4_velocity}, CH4 acceleration={stage_ch4_acceleration}, "
                f"max_voltage={stage_max_voltage}, step_x={stage_step_x}, step_y={stage_step_y}, "
                f"x_sign={stage_x_sign}, y_sign={stage_y_sign}, output_dir={output_dir}"
            )

            rule_cfg = RuleABRuntimeConfig(
                capture_area=capture_area,
                output_dir=output_dir,
                sam2_device="cuda",

                enable_stage=rule_ab_enable_stage,
                max_cycles=10_000_000,
                loop_interval_s=rule_ab_loop_interval_s,

                # 新规则：沿 C 绕行 + A-C 距离控制；A/B 覆盖时停。
                a_c_target_clearance=ac_target,
                a_c_min_clearance=ac_min,
                a_c_max_clearance=ac_max,
                follow_c_direction=follow_c_direction,
                a_b_overlap_stop=True,
                a_b_overlap_min_area_px=ab_overlap_threshold,

                stage_conn=stage_conn,
                stage_x_channel=stage_x_channel,
                stage_y_channel=stage_y_channel,
                stage_default_velocity=stage_default_velocity,
                stage_default_acceleration=stage_default_acceleration,
                stage_default_max_voltage=stage_max_voltage,
                stage_step_x=stage_step_x,
                stage_step_y=stage_step_y,
                stage_x_sign=stage_x_sign,
                stage_y_sign=stage_y_sign,
            )

            self.set_var(self.rule_module_status_var, "规则模块状态：请在弹出的图像中点击/确认 A、B、C")
            self.log("[RuleAB-C] 正在初始化 ActualNanoBoundaryFollower")
            follower = ActualNanoBoundaryFollower(rule_cfg)

            _configure_stage_ch3_ch4_separately(
                ch3_velocity=stage_ch3_velocity,
                ch3_acceleration=stage_ch3_acceleration,
                ch4_velocity=stage_ch4_velocity,
                ch4_acceleration=stage_ch4_acceleration,
                max_voltage=stage_max_voltage,
            )

            self.log("[RuleAB-C] 初始化第一帧 A/B/C")
            follower.initialize_abc_with_first_frame()

            self.set_var(self.rule_module_status_var, "规则模块状态：A沿C绕行中；AB覆盖时停，只检测B角度")
            self.log("[RuleAB-C] 开始持续运行。点击“停止测量”可停止。")

            frame_idx = 0
            motion_steps = 0
            overlap_monitor_frames = 0
            lost_count = 0
            last_b_angle: Optional[float] = None

            action_name_map = {0: "STAY", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT"}
            direction_map = {0: "stay", 1: "up", 2: "down", 3: "left", 4: "right"}

            while not self.rule_ab_only_stop_requested:
                frame_idx += 1

                try:
                    image_rgb, scene = follower.capture_and_build_scene()
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
                        self.log("[RuleAB-C] 连续视觉失败过多，停止单独A沿C绕行")
                        break

                    time.sleep(max(0.10, rule_ab_loop_interval_s))
                    continue

                b_angle = status.get("b_edge_angle_deg")
                d_angle = _angle_diff_180(b_angle, last_b_angle)
                last_b_angle = b_angle

                ab_overlap = bool(status.get("ab_overlap", False))
                ab_overlap_area = float(status.get("ab_overlap_area", 0.0) or 0.0)
                c_clearance = float(status.get("c_clearance", float("nan")))

                if ab_overlap:
                    _stop_stage_safe(f"A/B覆盖 area={ab_overlap_area:.1f}px²")
                    overlap_monitor_frames += 1
                    action_id = 0
                    action_name = "AB_OVERLAP_ANGLE_MONITOR"
                    direction = "monitor_only"

                    self.log(
                        f"[RuleAB-C] 覆盖监测 frame={frame_idx}, "
                        f"B_edge_angle={b_angle}, d_angle={d_angle}, "
                        f"AB_overlap_area={ab_overlap_area:.1f}px², "
                        f"A-C={c_clearance:.2f}px；Stage不动"
                    )

                    _save_and_record(image_rgb, scene, status, action_id, action_name, direction)
                    time.sleep(max(0.10, rule_ab_loop_interval_s))
                    continue

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

                if action_id == 0:
                    self.log("[RuleAB-C] 策略输出 STAY，本帧不运动；继续检测")
                else:
                    follower.execute_action(action_id)
                    motion_steps += 1
                    self.log(f"[RuleAB-C] 已执行 Stage 动作：{action_name}；motion_steps={motion_steps}")

                if rule_ab_loop_interval_s > 0:
                    time.sleep(rule_ab_loop_interval_s)

            _stop_stage_safe("单独A沿C绕行停止/结束")

            self.set_var(
                self.rule_module_status_var,
                (
                    "规则模块状态：单独A沿C绕行结束，"
                    f"frames={frame_idx}，motion_steps={motion_steps}，"
                    f"overlap_monitor_frames={overlap_monitor_frames}"
                )
            )
            self.log(
                "[RuleAB-C] 单独A沿C绕行结束："
                f"frames={frame_idx}, motion_steps={motion_steps}, "
                f"overlap_monitor_frames={overlap_monitor_frames}"
            )

        except Exception as e:
            self.set_var(self.rule_module_status_var, "规则模块状态：单独A沿C绕行失败")
            self.log(f"[RuleAB-C] 单独A沿C绕行失败：{e}")
            self.log(traceback.format_exc())

        finally:
            if follower is not None:
                try:
                    _stop_stage_safe("单独A沿C绕行 finally")
                except Exception:
                    pass

                try:
                    if hasattr(follower, "close"):
                        follower.close()
                    self.log("[RuleAB-C] follower 已关闭")
                except Exception as e:
                    self.log(f"[RuleAB-C] follower 关闭失败：{e}")


    def test_rule_ac_thread(self):
        self.run_in_thread(self.test_rule_ac)

    def test_rule_ac(self):
        try:
            wf = self.ensure_workflow()
            self.set_var(self.rule_module_status_var, "规则模块状态：RuleAC测试中")
            result = wf.run_rule_ac_until_threshold()
            last_row = result.get("last_row") or {}
            self.set_var(
                self.rule_module_status_var,
                f"规则模块状态：RuleAC完成，ok={result.get('ok')}，"
                f"area_px={last_row.get('area_px')}，csv={result.get('csv_path')}",
            )
            self.refresh_result_labels(wf)

        except Exception as e:
            self.set_var(self.rule_module_status_var, "规则模块状态：RuleAC测试失败")
            self.log(f"[RuleAC] 单独测试失败：{e}")
            self.log(traceback.format_exc())

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

            self.set_var(self.flow_status_var, "流程状态：GUI清理完成，Rigol已OFF")
            self.set_var(self.light_status_var, "照明光状态：GUI关闭不额外切换，保持当前状态")
            self.set_var(self.rigol_status_var, "Rigol状态：CH1已OFF，CH2已OFF")
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
