from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


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
    run_mode: RunMode = RunMode.SIMULATION
    detector_backend: str = "classic"
    frame_source_image: Optional[str] = None
    frame_source_image_2: Optional[str] = None
    sim_jitter_px: int = 0
    sim_noise_std: float = 0.0
    select_roi: bool = False
    select_roi_2: bool = False
    xy_driver: str = "dryrun"
    z_driver: str = "dryrun"
    x_move_step: int = 500
    y_move_step: int = 500
    z_step: float = 1.0
    tolerance_px: int = 6
    detect_retry: int = 6
    detect_retry_interval: float = 0.25
    settle_time: float = 0.35
    max_align_rounds: int = 60
    max_iterations: int = 8
    min_focus_score: float = 20.0
    adaptive_step: bool = False
    enable_recovery_scan: bool = False
    startup_motion_check_enabled: bool = True
    startup_motion_check_timeout: float = 1.0
    startup_motion_check_xy_steps: int = 1
    startup_motion_check_z_step: float = 1.0
    disable_run_lock: bool = False
    run_lock_file: str = str(Path("artifacts") / "spotzoom.lock.json")
    event_stream_jsonl: str = str(Path("artifacts") / "spotzoom_ui_events.jsonl")
    run_report_json: str = str(Path("artifacts") / "spotzoom_ui_report.json")
    log_level: str = "INFO"
    no_preview: bool = True
    window_title: str = "ToupView"
    window_title_2: Optional[str] = None
    window_wait_seconds: float = 10.0
    yolo_python: Optional[str] = None
    model_path: Optional[str] = None
    classic_method: str = "otsu"
    classic_selection: str = "brightest"
    classic_min_area: int = 20
    classic_max_area: int = 0
    classic_min_circularity: float = 0.2
    classic_min_intensity_ratio: float = 1.5
    classic_morph_kernel_size: int = 5
    newport_conn: int = 0
    newport_x_axis: int = 1
    newport_y_axis: int = 2
    newport_backend: str = "auto"
    newport_timeout: float = 5.0
    newport_multiaddr: bool = False
    newport_no_scan: bool = False
    newport_velocity: Optional[int] = None
    newport_acceleration: Optional[int] = None
    newport_no_wait: bool = False
    mrc_mirror1_x_axis: int = 1
    mrc_mirror1_y_axis: int = 2
    mrc_mirror2_x_axis: int = 3
    mrc_mirror2_y_axis: int = 4
    mrc_mirror1_x_sign: int = 1
    mrc_mirror1_y_sign: int = 1
    mrc_mirror2_x_sign: int = 1
    mrc_mirror2_y_sign: int = 1
    mrc_virtual_axis_mode: str = "shared"
    z_up_sign: int = 1
    z_picomotor_conn: int = 1
    z_picomotor_axis: int = 1
    z_picomotor_sign: int = 1
    z_picomotor_velocity: Optional[int] = None
    z_picomotor_acceleration: Optional[int] = None
    xps_ip: str = "192.168.1.100"
    xps_port: int = 5001
    xps_user: str = "Administrator"
    xps_password: str = "Administrator"
    xps_group: str = "ILS300LM"
    experimental_module_filters: List[str] = field(default_factory=list)
    # 4轴双镜闭环参数
    alignment_strategy: AlignmentStrategy = AlignmentStrategy.Z_SCAN_LEGACY
    stage1_kp: float = 1.0
    stage1_ki: float = 0.1
    stage2_kp: float = 1.0
    stage2_ki: float = 0.1
    coupling_c12: float = 0.0
    coupling_c21: float = 0.0
    tolerance_pos_px: int = 4
    tolerance_ang_px: int = 4
    converge_stable_frames: int = 5
    detector2_focal_length: float = 200.0
    detector_mode: str = "single_detector"
    stage1_gain_factor: float = 2.0
    sequential_stage1_iterations: int = 3
    # 探测器效果对比参数
    comparison_mode: str = "detector_primary"
    detector_weight: float = 0.7
    touview_weight: float = 0.3
    disagreement_threshold_px: float = 10.0
    comparison_log_interval: int = 1
    # UCC CCD相机参数
    ucc_device: Optional[int] = None
    ucc_resolution: str = "PAL"
    ucc_exposure: Optional[float] = None
    ucc_gain: Optional[float] = None
    ucc_brightness: Optional[float] = None
    ucc_contrast: Optional[float] = None
    ucc_device_2: Optional[int] = None
    ucc_resolution_2: str = "PAL"
    preview: bool = True


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
