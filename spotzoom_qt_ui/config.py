from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional


# =============================================================
# 默认配置定义 - 所有默认参数集中管理
# =============================================================

# --- 运行模式 ---
RUN_MODE: str = "simulation"
DETECTOR_BACKEND: str = "classic"
FRAME_SOURCE_IMAGE: Optional[str] = None
FRAME_SOURCE_IMAGE_2: Optional[str] = None

# --- 模拟参数 ---
SIM_JITTER_PX: int = 0
SIM_NOISE_STD: float = 0.0

# --- ROI ---
SELECT_ROI: bool = False
SELECT_ROI_2: bool = False

# --- 运动控制 ---
XY_DRIVER: str = "newport-mrc4"
Z_DRIVER: str = "dryrun"
X_MOVE_STEP: int = 5
Y_MOVE_STEP: int = 5
Z_STEP: float = 1.0

# --- 对准参数 ---
TOLERANCE_PX: int = 6
DETECT_RETRY: int = 6
DETECT_RETRY_INTERVAL: float = 0.25
SETTLE_TIME: float = 0.35
MAX_ALIGN_ROUNDS: int = 60
MAX_ITERATIONS: int = 8
MIN_FOCUS_SCORE: float = 20.0
ADAPTIVE_STEP: bool = False
ENABLE_RECOVERY_SCAN: bool = False

# --- 启动自检 ---
STARTUP_MOTION_CHECK_ENABLED: bool = True
STARTUP_MOTION_CHECK_TIMEOUT: float = 1.0
STARTUP_MOTION_CHECK_XY_STEPS: int = 1
STARTUP_MOTION_CHECK_Z_STEP: float = 1.0

# --- 运行锁与日志 ---
DISABLE_RUN_LOCK: bool = False
RUN_LOCK_FILE: str = str(Path("artifacts") / "spotzoom.lock.json")
EVENT_STREAM_JSONL: str = str(Path("artifacts") / "spotzoom_ui_events.jsonl")
RUN_REPORT_JSON: str = str(Path("artifacts") / "spotzoom_ui_report.json")
LOG_LEVEL: str = "INFO"
NO_PREVIEW: bool = True

# --- 窗口参数 ---
WINDOW_TITLE: str = "NIS"
WINDOW_TITLE_2: Optional[str] = None
WINDOW_WAIT_SECONDS: float = 10.0

# --- 检测器参数 (YOLO) ---
YOLO_PYTHON: Optional[str] = None
MODEL_PATH: Optional[str] = None

# --- 检测器参数 (classic) ---
CLASSIC_METHOD: str = "otsu"
CLASSIC_SELECTION: str = "brightest"
CLASSIC_MIN_AREA: int = 20
CLASSIC_MAX_AREA: int = 0
CLASSIC_MIN_CIRCULARITY: float = 0.2
CLASSIC_MIN_INTENSITY_RATIO: float = 1.5
CLASSIC_MORPH_KERNEL_SIZE: int = 5

# --- Newport / 运动控制器 ---
NEWPORT_CONN: int = 0
NEWPORT_X_AXIS: int = 1
NEWPORT_Y_AXIS: int = 2
NEWPORT_BACKEND: str = "auto"
NEWPORT_TIMEOUT: float = 5.0
NEWPORT_MULTIADDR: bool = False
NEWPORT_NO_SCAN: bool = False
NEWPORT_VELOCITY: Optional[int] = None
NEWPORT_ACCELERATION: Optional[int] = None
NEWPORT_NO_WAIT: bool = False

# --- MRC 4轴参数（轴1、轴2 指定为mirror1；轴3、轴4 指定为mirror2）
MRC_MIRROR1_X_AXIS: int = 1
MRC_MIRROR1_Y_AXIS: int = 2
MRC_MIRROR2_X_AXIS: int = 3
MRC_MIRROR2_Y_AXIS: int = 4
MRC_MIRROR1_X_SIGN: int = 1
MRC_MIRROR1_Y_SIGN: int = 1
MRC_MIRROR2_X_SIGN: int = 1
MRC_MIRROR2_Y_SIGN: int = 1
MRC_VIRTUAL_AXIS_MODE: str = "shared"

# --- Z轴 ---
Z_UP_SIGN: int = 1
Z_PICOMOTOR_CONN: int = 1
Z_PICOMOTOR_AXIS: int = 1
Z_PICOMOTOR_SIGN: int = 1
Z_PICOMOTOR_VELOCITY: Optional[int] = None
Z_PICOMOTOR_ACCELERATION: Optional[int] = None

# --- XPS ---
XPS_IP: str = "192.168.1.100"
XPS_PORT: int = 5001
XPS_USER: str = "Administrator"
XPS_PASSWORD: str = "Administrator"
XPS_GROUP: str = "ILS300LM"

# --- 实验模块 ---
EXPERIMENTAL_MODULE_FILTERS: List[str] = []

# --- 4轴双镜闭环 ---
ALIGNMENT_STRATEGY: str = "dual_detector_4axis"
STAGE1_KP: float = 1.0
STAGE1_KI: float = 0.1
STAGE2_KP: float = 1.0
STAGE2_KI: float = 0.1
COUPLING_C12: float = 0.0
COUPLING_C21: float = 0.0
TOLERANCE_POS_PX: int = 4
TOLERANCE_ANG_PX: int = 4
CONVERGE_STABLE_FRAMES: int = 5

# --- 双探测器 ---
DETECTOR2_FOCAL_LENGTH: float = 200.0
DETECTOR_MODE: str = "single_detector"
CORRECTION_MIRROR: str = "mirror2"  # 本探测器对应的校正镜: mirror1, mirror2, both
STAGE1_GAIN_FACTOR: float = 2.0
SEQUENTIAL_STAGE1_ITERATIONS: int = 3

# --- 探测器对比参数 ---
COMPARISON_MODE: str = "detector_primary"
DETECTOR_WEIGHT: float = 0.7
TOUVIEW_WEIGHT: float = 0.3
DISAGREEMENT_THRESHOLD_PX: float = 10.0
COMPARISON_LOG_INTERVAL: int = 1

# --- UCC CCD相机参数 ---
UCC_DEVICE: Optional[int] = None
UCC_RESOLUTION: str = "PAL"
UCC_EXPOSURE: Optional[float] = None
UCC_GAIN: Optional[float] = None
UCC_BRIGHTNESS: Optional[float] = None
UCC_CONTRAST: Optional[float] = None
UCC_DEVICE_2: Optional[int] = None
UCC_RESOLUTION_2: str = "PAL"

# --- 预览与缓存 ---
PREVIEW: bool = True
FRAME_CACHE_ENABLED: bool = False
FRAME_CACHE_DIR: str = "Tmp_Frames"

# --- 中间帧保存 ---
SAVE_INTERMEDIATE_FRAMES: bool = False
FRAME_TMP_DIR: str = "FrameTmp"

# --- Z轴禁用 ---
DISABLE_Z_AXIS: bool = True