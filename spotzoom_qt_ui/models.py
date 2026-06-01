from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from . import config as cfg_defaults


class UiStatus(str, Enum):
    NORMAL = "normal"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    DISCONNECTED = "disconnected"
    DISABLED = "disabled"
    WARNING = "warning"
    ARMED = "armed"


class RunMode(str, Enum):
    SIMULATION = "simulation"
    REAL = "real"


class AlignmentStrategy(str, Enum):
    Z_SCAN_LEGACY = "z_scan_legacy"
    DUAL_DETECTOR_4AXIS = "dual_detector_4axis"


@dataclass
class RuntimeProfile:
    run_mode: RunMode = RunMode(cfg_defaults.RUN_MODE)
    detector_backend: str = cfg_defaults.DETECTOR_BACKEND
    frame_source_image: Optional[str] = cfg_defaults.FRAME_SOURCE_IMAGE
    frame_source_image_2: Optional[str] = cfg_defaults.FRAME_SOURCE_IMAGE_2
    sim_jitter_px: int = cfg_defaults.SIM_JITTER_PX
    sim_noise_std: float = cfg_defaults.SIM_NOISE_STD
    select_roi: bool = cfg_defaults.SELECT_ROI
    select_roi_2: bool = cfg_defaults.SELECT_ROI_2
    xy_driver: str = cfg_defaults.XY_DRIVER
    z_driver: str = cfg_defaults.Z_DRIVER
    x_move_step: int = cfg_defaults.X_MOVE_STEP
    y_move_step: int = cfg_defaults.Y_MOVE_STEP
    z_step: float = cfg_defaults.Z_STEP
    tolerance_px: int = cfg_defaults.TOLERANCE_PX
    detect_retry: int = cfg_defaults.DETECT_RETRY
    detect_retry_interval: float = cfg_defaults.DETECT_RETRY_INTERVAL
    settle_time: float = cfg_defaults.SETTLE_TIME
    max_align_rounds: int = cfg_defaults.MAX_ALIGN_ROUNDS
    max_iterations: int = cfg_defaults.MAX_ITERATIONS
    min_focus_score: float = cfg_defaults.MIN_FOCUS_SCORE
    adaptive_step: bool = cfg_defaults.ADAPTIVE_STEP
    enable_recovery_scan: bool = cfg_defaults.ENABLE_RECOVERY_SCAN
    startup_motion_check_enabled: bool = cfg_defaults.STARTUP_MOTION_CHECK_ENABLED
    startup_motion_check_timeout: float = cfg_defaults.STARTUP_MOTION_CHECK_TIMEOUT
    startup_motion_check_xy_steps: int = cfg_defaults.STARTUP_MOTION_CHECK_XY_STEPS
    startup_motion_check_z_step: float = cfg_defaults.STARTUP_MOTION_CHECK_Z_STEP
    disable_run_lock: bool = cfg_defaults.DISABLE_RUN_LOCK
    run_lock_file: str = cfg_defaults.RUN_LOCK_FILE
    event_stream_jsonl: str = cfg_defaults.EVENT_STREAM_JSONL
    run_report_json: str = cfg_defaults.RUN_REPORT_JSON
    log_level: str = cfg_defaults.LOG_LEVEL
    no_preview: bool = cfg_defaults.NO_PREVIEW
    window_title: str = cfg_defaults.WINDOW_TITLE
    window_title_2: Optional[str] = cfg_defaults.WINDOW_TITLE_2
    window_wait_seconds: float = cfg_defaults.WINDOW_WAIT_SECONDS
    yolo_python: Optional[str] = cfg_defaults.YOLO_PYTHON
    model_path: Optional[str] = cfg_defaults.MODEL_PATH
    classic_method: str = cfg_defaults.CLASSIC_METHOD
    classic_selection: str = cfg_defaults.CLASSIC_SELECTION
    classic_min_area: int = cfg_defaults.CLASSIC_MIN_AREA
    classic_max_area: int = cfg_defaults.CLASSIC_MAX_AREA
    classic_min_circularity: float = cfg_defaults.CLASSIC_MIN_CIRCULARITY
    classic_min_intensity_ratio: float = cfg_defaults.CLASSIC_MIN_INTENSITY_RATIO
    classic_morph_kernel_size: int = cfg_defaults.CLASSIC_MORPH_KERNEL_SIZE
    newport_conn: int = cfg_defaults.NEWPORT_CONN
    newport_x_axis: int = cfg_defaults.NEWPORT_X_AXIS
    newport_y_axis: int = cfg_defaults.NEWPORT_Y_AXIS
    newport_backend: str = cfg_defaults.NEWPORT_BACKEND
    newport_timeout: float = cfg_defaults.NEWPORT_TIMEOUT
    newport_multiaddr: bool = cfg_defaults.NEWPORT_MULTIADDR
    newport_no_scan: bool = cfg_defaults.NEWPORT_NO_SCAN
    newport_velocity: Optional[int] = cfg_defaults.NEWPORT_VELOCITY
    newport_acceleration: Optional[int] = cfg_defaults.NEWPORT_ACCELERATION
    newport_no_wait: bool = cfg_defaults.NEWPORT_NO_WAIT
    mrc_mirror1_x_axis: int = cfg_defaults.MRC_MIRROR1_X_AXIS
    mrc_mirror1_y_axis: int = cfg_defaults.MRC_MIRROR1_Y_AXIS
    mrc_mirror2_x_axis: int = cfg_defaults.MRC_MIRROR2_X_AXIS
    mrc_mirror2_y_axis: int = cfg_defaults.MRC_MIRROR2_Y_AXIS
    mrc_mirror1_x_sign: int = cfg_defaults.MRC_MIRROR1_X_SIGN
    mrc_mirror1_y_sign: int = cfg_defaults.MRC_MIRROR1_Y_SIGN
    mrc_mirror2_x_sign: int = cfg_defaults.MRC_MIRROR2_X_SIGN
    mrc_mirror2_y_sign: int = cfg_defaults.MRC_MIRROR2_Y_SIGN
    mrc_virtual_axis_mode: str = cfg_defaults.MRC_VIRTUAL_AXIS_MODE
    z_up_sign: int = cfg_defaults.Z_UP_SIGN
    z_picomotor_conn: int = cfg_defaults.Z_PICOMOTOR_CONN
    z_picomotor_axis: int = cfg_defaults.Z_PICOMOTOR_AXIS
    z_picomotor_sign: int = cfg_defaults.Z_PICOMOTOR_SIGN
    z_picomotor_velocity: Optional[int] = cfg_defaults.Z_PICOMOTOR_VELOCITY
    z_picomotor_acceleration: Optional[int] = cfg_defaults.Z_PICOMOTOR_ACCELERATION
    xps_ip: str = cfg_defaults.XPS_IP
    xps_port: int = cfg_defaults.XPS_PORT
    xps_user: str = cfg_defaults.XPS_USER
    xps_password: str = cfg_defaults.XPS_PASSWORD
    xps_group: str = cfg_defaults.XPS_GROUP
    experimental_module_filters: List[str] = field(default_factory=lambda: list(cfg_defaults.EXPERIMENTAL_MODULE_FILTERS))
    # 4轴双镜闭环参数
    alignment_strategy: AlignmentStrategy = AlignmentStrategy(cfg_defaults.ALIGNMENT_STRATEGY)
    stage1_kp: float = cfg_defaults.STAGE1_KP
    stage1_ki: float = cfg_defaults.STAGE1_KI
    stage2_kp: float = cfg_defaults.STAGE2_KP
    stage2_ki: float = cfg_defaults.STAGE2_KI
    coupling_c12: float = cfg_defaults.COUPLING_C12
    coupling_c21: float = cfg_defaults.COUPLING_C21
    tolerance_pos_px: int = cfg_defaults.TOLERANCE_POS_PX
    tolerance_ang_px: int = cfg_defaults.TOLERANCE_ANG_PX
    converge_stable_frames: int = cfg_defaults.CONVERGE_STABLE_FRAMES
    detector2_focal_length: float = cfg_defaults.DETECTOR2_FOCAL_LENGTH
    detector_mode: str = cfg_defaults.DETECTOR_MODE
    stage1_gain_factor: float = cfg_defaults.STAGE1_GAIN_FACTOR
    sequential_stage1_iterations: int = cfg_defaults.SEQUENTIAL_STAGE1_ITERATIONS
    # 探测器效果对比参数
    comparison_mode: str = cfg_defaults.COMPARISON_MODE
    detector_weight: float = cfg_defaults.DETECTOR_WEIGHT
    touview_weight: float = cfg_defaults.TOUVIEW_WEIGHT
    disagreement_threshold_px: float = cfg_defaults.DISAGREEMENT_THRESHOLD_PX
    comparison_log_interval: int = cfg_defaults.COMPARISON_LOG_INTERVAL
    # UCC CCD相机参数
    ucc_device: Optional[int] = cfg_defaults.UCC_DEVICE
    ucc_resolution: str = cfg_defaults.UCC_RESOLUTION
    ucc_exposure: Optional[float] = cfg_defaults.UCC_EXPOSURE
    ucc_gain: Optional[float] = cfg_defaults.UCC_GAIN
    ucc_brightness: Optional[float] = cfg_defaults.UCC_BRIGHTNESS
    ucc_contrast: Optional[float] = cfg_defaults.UCC_CONTRAST
    ucc_device_2: Optional[int] = cfg_defaults.UCC_DEVICE_2
    ucc_resolution_2: str = cfg_defaults.UCC_RESOLUTION_2
    preview: bool = cfg_defaults.PREVIEW
    frame_cache_enabled: bool = cfg_defaults.FRAME_CACHE_ENABLED
    frame_cache_dir: str = cfg_defaults.FRAME_CACHE_DIR
    save_intermediate_frames: bool = cfg_defaults.SAVE_INTERMEDIATE_FRAMES
    frame_tmp_dir: str = cfg_defaults.FRAME_TMP_DIR
    disable_z_axis: bool = cfg_defaults.DISABLE_Z_AXIS


@dataclass
class RuntimeSnapshot:
    mode_label: str
    requested_backend: str
    resolved_backend: str
    xy_driver: str
    z_driver: str
    roi_label: str
    target_label: str
    device_status_label: str
    env_check_status: UiStatus
    startup_check_status: UiStatus
    last_run_status: UiStatus
    run_lock_status: UiStatus
    risk_label: str


@dataclass
class DevicePanelState:
    title: str
    status: UiStatus
    summary: str
    details: Dict[str, str]


@dataclass
class ModuleViewItem:
    version: int
    name: str
    type_key: str
    type_label: str
    title: str
    summary: str
    status: UiStatus
    placement: str
    import_path: str
    source_path: str
    docs_path: str
    primary_symbol: str
    config_symbol: str
    dependency_reason: str


@dataclass
class EventRecord:
    timestamp: str
    event_name: str
    payload_summary: str
    raw_payload: Dict[str, object]


@dataclass
class TestCaseSpec:
    test_id: str
    category: str
    title: str
    description: str


@dataclass
class TestResult:
    test_id: str
    title: str
    status: UiStatus
    duration_ms: int
    message: str
    raw_return: str


@dataclass
class ActionResult:
    status: UiStatus
    message: str
    command_preview: str
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
