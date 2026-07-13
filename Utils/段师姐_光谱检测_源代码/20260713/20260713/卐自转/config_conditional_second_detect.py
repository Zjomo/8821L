# -*- coding: utf-8 -*-
"""
循环测量 GUI 默认参数配置文件。

只改默认值时，优先修改本文件，不需要到主 GUI 大代码里查找。
主程序启动时会读取 DEFAULT_CONFIG；GUI 运行过程中手动修改输入框仍然优先于这里的默认值。
"""

DEFAULT_CONFIG = {
    # ---------------- 基本测量参数 ----------------
    "max_cycles": 200,
    "signal_on_time_ms": 1000.0,
    "stable_wait_ms": 1200,
    # 稳定等待时间：关照明光之后到 LabVIEW 采集之前的等待。
    "angle_delta_min_deg": 1.0,
    "angle_delta_max_deg": 4.0,
    "signal_time_factor": 1.5,
    "save_root": "measurement_output",

    # ---------------- 照明光 ----------------
    "light_port": "COM17",

    # ---------------- Rigol DG4062 ----------------
    "rigol_visa": "USB0::0x1AB1::0x0641::DG4E192200870::INSTR",
    "rigol_timeout_ms": 2000,
    "rigol_command_delay_s": 0.05,
    "ch1_low_v": 1.0,
    "ch1_high_v": 5.0,
    "ch1_freq_hz":6000,
    "ch1_duty_percent": 3.0,
    "ch1_delay_s": 0.0,
    "ch2_low_v": 3.0,
    "ch2_high_v": 0.002,
    "ch2_freq_hz": 5000.0,
    "ch2_duty_percent": 3.0,
    "ch2_delay_s": 0.0,

    # ---------------- 角度检测 ----------------
    "angle_model_path": r"D:/desktop/train/model/best_wan12.2.pt",
    "capture_area": (116, 98, 1112, 886),
    "angle_output_dir": "outputs/captured_frames",
    "angle_num": 0,
    "angle_cw": 1,

    # 条件二次检测：第一次基础检测角度落入以下范围时，
    # 临时启用 ScreenAngleDetector.use_second_detect=True 再检测一次。
    # 默认规则：0~20° 或 90~110°。
    "enable_conditional_second_detect": True,
    "conditional_second_detect_ranges": ((0.0, 20.0), (90.0, 110.0)),

    # ---------------- LabVIEW TCP ----------------
    "tcp_host": "127.0.0.1",
    "tcp_port": 65432,
    "tcp_output_dir": "labview_csv_output",
    "tcp_command": "MEASURE",

    # ---------------- 光谱数据处理 ----------------
    "raw_remove_above": 3000.0,
    "median_filter_window": 5,
    "x_axis_xlsx_path": "516中心波长.xlsx",

    # ---------------- Δw 判断 ----------------
    "enable_delta_w_judge": False,
    "delta_w_threshold": 0.0,
    "stop_when_delta_w_not_enough": False,
}
