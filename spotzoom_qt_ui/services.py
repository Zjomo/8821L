from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np

import SpotZoom
from SpotZoom_Machine_Learning_Unified import list_module_specs
from SpotZoom_Machine_Learning_Unified.registry import OptimizationType

from .models import (
    ActionResult,
    AlignmentStrategy,
    DevicePanelState,
    EventRecord,
    ModuleViewItem,
    RunMode,
    RuntimeProfile,
    RuntimeSnapshot,
    TestCaseSpec,
    TestResult,
    UiStatus,
)


TYPE_LABELS: Dict[str, str] = {
    OptimizationType.IMAGE_ENHANCEMENT.value: "图像增强",
    OptimizationType.DETECTION_LOCALIZATION.value: "检测与定位",
    OptimizationType.TRACKING_PREDICTION.value: "跟踪与预测",
    OptimizationType.ALIGNMENT_REGISTRATION.value: "配准与对齐",
    OptimizationType.ADAPTIVE_OPTICS_CONTROL.value: "自适应光学与控制",
    OptimizationType.BEAM_OPTICAL_SIMULATION.value: "光束与光学仿真",
    OptimizationType.SYSTEM_ANALYSIS.value: "系统分析",
    OptimizationType.SYSTEM_DIAGNOSTICS.value: "系统诊断",
    OptimizationType.SYSTEM_IDENTIFICATION.value: "系统辨识",
    OptimizationType.HARDWARE_ABSTRACTION.value: "硬件抽象",
    OptimizationType.GLOBAL_OPTIMIZATION.value: "全局优化",
}


PLACEMENT_HINTS: Dict[str, str] = {
    OptimizationType.IMAGE_ENHANCEMENT.value: "检测前",
    OptimizationType.DETECTION_LOCALIZATION.value: "检测中",
    OptimizationType.TRACKING_PREDICTION.value: "跟踪阶段",
    OptimizationType.ALIGNMENT_REGISTRATION.value: "配准阶段",
    OptimizationType.ADAPTIVE_OPTICS_CONTROL.value: "仅实验模式",
    OptimizationType.BEAM_OPTICAL_SIMULATION.value: "仅实验模式",
    OptimizationType.SYSTEM_ANALYSIS.value: "诊断阶段",
    OptimizationType.SYSTEM_DIAGNOSTICS.value: "诊断阶段",
    OptimizationType.SYSTEM_IDENTIFICATION.value: "仅实验模式",
    OptimizationType.HARDWARE_ABSTRACTION.value: "fallback",
    OptimizationType.GLOBAL_OPTIMIZATION.value: "仅实验模式",
}


def _shorten_payload(payload: Dict[str, object]) -> str:
    if not payload:
        return "-"
    keys = [key for key in payload.keys() if key != "event"]
    preview = ", ".join(f"{key}={payload[key]}" for key in keys[:3])
    return preview if preview else "-"


class RuntimeControlService:
    def __init__(self, repo_root: Optional[Path] = None, python_exe: Optional[str] = None):
        self.repo_root = Path(repo_root or Path(__file__).resolve().parent.parent)
        self.python_exe = python_exe or sys.executable
        self.script_path = self.repo_root / "SpotZoom.py"
        self.artifacts_dir = self.repo_root / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._sample_frame_path = self.artifacts_dir / "spotzoom_qt_sample.png"

    def default_profile(self) -> RuntimeProfile:
        profile = RuntimeProfile()
        profile.frame_source_image = str(self.ensure_sample_frame())
        return profile

    def ensure_sample_frame(self) -> Path:
        if self._sample_frame_path.exists():
            return self._sample_frame_path
        canvas = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.circle(canvas, (256, 256), 28, (255, 255, 255), -1)
        canvas = cv2.GaussianBlur(canvas, (0, 0), 5.0)
        cv2.imwrite(str(self._sample_frame_path), canvas)
        return self._sample_frame_path

    def _normalize_profile(self, profile: RuntimeProfile) -> RuntimeProfile:
        if profile.run_mode == RunMode.SIMULATION and not profile.frame_source_image:
            profile.frame_source_image = str(self.ensure_sample_frame())
        return profile

    def build_command(
        self,
        profile: RuntimeProfile,
        *,
        check_env: bool = False,
        single_step: bool = False,
        startup_check_only: bool = False,
        extra_args: Optional[Iterable[str]] = None,
    ) -> List[str]:
        profile = self._normalize_profile(profile)
        args: List[str] = [self.python_exe, str(self.script_path)]

        if check_env:
            args.append("--check-env")
        if startup_check_only:
            args.append("--startup-motion-check-only")

        args.extend(["--detector-backend", profile.detector_backend])
        args.extend(["--xy-driver", profile.xy_driver, "--z-driver", profile.z_driver])
        args.extend(["--window-title", profile.window_title])
        args.extend(["--window-wait-seconds", str(profile.window_wait_seconds)])

        if profile.run_mode == RunMode.SIMULATION:
            args.extend(["--frame-source-image", str(profile.frame_source_image or self.ensure_sample_frame())])
            args.extend(["--sim-jitter-px", str(profile.sim_jitter_px)])
            args.extend(["--sim-noise-std", str(profile.sim_noise_std)])

        if profile.select_roi:
            args.append("--select-roi")
        else:
            args.append("--skip-roi")

        # 4轴双镜闭环参数
        args.extend(["--alignment-strategy", profile.alignment_strategy.value])
        if profile.window_title_2:
            args.extend(["--window-title-2", profile.window_title_2])
        if profile.frame_source_image_2:
            args.extend(["--frame-source-image-2", profile.frame_source_image_2])
        if profile.select_roi_2:
            args.append("--select-roi-2")
        else:
            args.append("--skip-roi-2")
        args.extend(["--stage1-kp", str(profile.stage1_kp)])
        args.extend(["--stage1-ki", str(profile.stage1_ki)])
        args.extend(["--stage2-kp", str(profile.stage2_kp)])
        args.extend(["--stage2-ki", str(profile.stage2_ki)])
        args.extend(["--coupling-c12", str(profile.coupling_c12)])
        args.extend(["--coupling-c21", str(profile.coupling_c21)])
        args.extend(["--tolerance-pos-px", str(profile.tolerance_pos_px)])
        args.extend(["--tolerance-ang-px", str(profile.tolerance_ang_px)])
        args.extend(["--converge-stable-frames", str(profile.converge_stable_frames)])
        args.extend(["--detector2-focal-length", str(profile.detector2_focal_length)])
        args.extend(["--detector-mode", profile.detector_mode])
        args.extend(["--stage1-gain-factor", str(profile.stage1_gain_factor)])
        args.extend(["--sequential-stage1-iterations", str(profile.sequential_stage1_iterations)])
        args.extend(["--comparison-mode", profile.comparison_mode])
        args.extend(["--detector-weight", str(profile.detector_weight)])
        args.extend(["--touview-weight", str(profile.touview_weight)])
        args.extend(["--disagreement-threshold-px", str(profile.disagreement_threshold_px)])
        args.extend(["--comparison-log-interval", str(profile.comparison_log_interval)])

        # UCC CCD相机参数
        if profile.ucc_device is not None:
            args.extend(["--ucc-device", str(profile.ucc_device)])
            args.extend(["--ucc-resolution", profile.ucc_resolution])
            if profile.ucc_exposure is not None:
                args.extend(["--ucc-exposure", str(profile.ucc_exposure)])
            if profile.ucc_gain is not None:
                args.extend(["--ucc-gain", str(profile.ucc_gain)])
            if profile.ucc_brightness is not None:
                args.extend(["--ucc-brightness", str(profile.ucc_brightness)])
            if profile.ucc_contrast is not None:
                args.extend(["--ucc-contrast", str(profile.ucc_contrast)])
        if profile.ucc_device_2 is not None:
            args.extend(["--ucc-device-2", str(profile.ucc_device_2)])
            args.extend(["--ucc-resolution-2", profile.ucc_resolution_2])

        args.extend(["--tolerance-px", str(profile.tolerance_px)])
        args.extend(["--detect-retry", str(profile.detect_retry)])
        args.extend(["--detect-retry-interval", str(profile.detect_retry_interval)])
        args.extend(["--settle-time", str(profile.settle_time)])
        args.extend(["--max-align-rounds", str(profile.max_align_rounds)])
        args.extend(["--x-move-step", str(profile.x_move_step)])
        args.extend(["--y-move-step", str(profile.y_move_step)])
        args.extend(["--z-step", str(profile.z_step)])
        args.extend(["--min-focus-score", str(profile.min_focus_score)])

        max_iterations = 1 if single_step else profile.max_iterations
        args.extend(["--max-iterations", str(max_iterations)])

        if profile.adaptive_step:
            args.append("--adaptive-step")
        if profile.enable_recovery_scan:
            args.append("--enable-recovery-scan")
        if not profile.startup_motion_check_enabled:
            args.append("--disable-startup-motion-check")
        args.extend(["--startup-motion-check-timeout", str(profile.startup_motion_check_timeout)])
        args.extend(["--startup-motion-check-xy-steps", str(profile.startup_motion_check_xy_steps)])
        args.extend(["--startup-motion-check-z-step", str(profile.startup_motion_check_z_step)])

        if profile.frame_cache_enabled:
            args.append("--frame-cache")
        if profile.disable_z_axis:
            args.append("--disable-z-axis")

        args.extend(["--newport-conn", str(profile.newport_conn)])
        args.extend(["--newport-x-axis", str(profile.newport_x_axis)])
        args.extend(["--newport-y-axis", str(profile.newport_y_axis)])
        args.extend(["--newport-backend", profile.newport_backend])
        args.extend(["--newport-timeout", str(profile.newport_timeout)])
        if profile.newport_multiaddr:
            args.append("--newport-multiaddr")
        if profile.newport_no_scan:
            args.append("--newport-no-scan")
        if profile.newport_no_wait:
            args.append("--newport-no-wait")
        if profile.newport_velocity is not None:
            args.extend(["--newport-velocity", str(profile.newport_velocity)])
        if profile.newport_acceleration is not None:
            args.extend(["--newport-acceleration", str(profile.newport_acceleration)])

        args.extend(["--mrc-mirror1-x-axis", str(profile.mrc_mirror1_x_axis)])
        args.extend(["--mrc-mirror1-y-axis", str(profile.mrc_mirror1_y_axis)])
        args.extend(["--mrc-mirror2-x-axis", str(profile.mrc_mirror2_x_axis)])
        args.extend(["--mrc-mirror2-y-axis", str(profile.mrc_mirror2_y_axis)])
        args.extend(["--mrc-mirror1-x-sign", str(profile.mrc_mirror1_x_sign)])
        args.extend(["--mrc-mirror1-y-sign", str(profile.mrc_mirror1_y_sign)])
        args.extend(["--mrc-mirror2-x-sign", str(profile.mrc_mirror2_x_sign)])
        args.extend(["--mrc-mirror2-y-sign", str(profile.mrc_mirror2_y_sign)])
        virtual_axis_mode_map = {"共享": "shared", "反射镜1": "mirror1", "反射镜2": "mirror2"}
        cli_mode = virtual_axis_mode_map.get(profile.mrc_virtual_axis_mode, profile.mrc_virtual_axis_mode)
        args.extend(["--mrc-virtual-axis-mode", cli_mode])

        args.extend(["--z-picomotor-conn", str(profile.z_picomotor_conn)])
        args.extend(["--z-picomotor-axis", str(profile.z_picomotor_axis)])
        args.extend(["--z-picomotor-sign", str(profile.z_picomotor_sign)])
        if profile.z_picomotor_velocity is not None:
            args.extend(["--z-picomotor-velocity", str(profile.z_picomotor_velocity)])
        if profile.z_picomotor_acceleration is not None:
            args.extend(["--z-picomotor-acceleration", str(profile.z_picomotor_acceleration)])

        args.extend(["--xps-ip", profile.xps_ip])
        args.extend(["--xps-port", str(profile.xps_port)])
        args.extend(["--xps-user", profile.xps_user])
        args.extend(["--xps-password", profile.xps_password])
        args.extend(["--xps-group", profile.xps_group])

        if profile.disable_run_lock:
            args.append("--disable-run-lock")
        else:
            args.extend(["--run-lock-file", str(profile.run_lock_file)])
        args.extend(["--event-stream-jsonl", str(profile.event_stream_jsonl)])
        args.extend(["--run-report-json", str(profile.run_report_json)])
        args.extend(["--log-level", profile.log_level])
        if profile.no_preview:
            args.append("--no-preview")
        if profile.yolo_python:
            args.extend(["--yolo-python", profile.yolo_python])
        if profile.model_path:
            args.extend(["--model-path", profile.model_path])

        if extra_args:
            args.extend(list(extra_args))
        return args

    def command_preview(self, command: Iterable[str]) -> str:
        return subprocess.list2cmdline(list(command))

    def _namespace_for_backend(self, profile: RuntimeProfile) -> SimpleNamespace:
        profile = self._normalize_profile(profile)
        return SimpleNamespace(
            detector_backend=profile.detector_backend,
            yolo_python=profile.yolo_python,
            model_path=profile.model_path,
            target_class_name="lightspot",
            target_class_id=0,
            conf_thres=0.25,
            classic_method=profile.classic_method,
            classic_selection=profile.classic_selection,
            classic_min_area=profile.classic_min_area,
            classic_max_area=profile.classic_max_area,
            classic_min_circularity=profile.classic_min_circularity,
            classic_min_intensity_ratio=profile.classic_min_intensity_ratio,
            classic_morph_kernel_size=profile.classic_morph_kernel_size,
        )

    def resolve_backend(self, profile: RuntimeProfile) -> str:
        try:
            return SpotZoom._resolve_detector_backend(self._namespace_for_backend(profile))
        except Exception:
            return profile.detector_backend

    def build_snapshot(
        self,
        profile: RuntimeProfile,
        *,
        env_status: UiStatus = UiStatus.NORMAL,
        startup_status: UiStatus = UiStatus.NORMAL,
        run_status: UiStatus = UiStatus.NORMAL,
    ) -> RuntimeSnapshot:
        resolved = self.resolve_backend(profile)
        mode_label = "模拟模式" if profile.run_mode == RunMode.SIMULATION else "真实设备"
        roi_label = "启用 ROI" if profile.select_roi else "全幅采集"
        target_label = "P1/P2/P3 闭环"
        device_summary = f"XY={profile.xy_driver} / Z={profile.z_driver}"
        risk_label = "不会驱动真实设备" if profile.run_mode == RunMode.SIMULATION else "会驱动真实设备"
        run_lock_status = UiStatus.DISABLED if profile.disable_run_lock else UiStatus.NORMAL
        return RuntimeSnapshot(
            mode_label=mode_label,
            requested_backend=profile.detector_backend,
            resolved_backend=resolved,
            xy_driver=profile.xy_driver,
            z_driver=profile.z_driver,
            roi_label=roi_label,
            target_label=target_label,
            device_status_label=device_summary,
            env_check_status=env_status,
            startup_check_status=startup_status,
            last_run_status=run_status,
            run_lock_status=run_lock_status,
            risk_label=risk_label,
        )

    def run_command(
        self,
        profile: RuntimeProfile,
        *,
        check_env: bool = False,
        single_step: bool = False,
        startup_check_only: bool = False,
        extra_args: Optional[Iterable[str]] = None,
        timeout: Optional[float] = None,
    ) -> ActionResult:
        command = self.build_command(
            profile,
            check_env=check_env,
            single_step=single_step,
            startup_check_only=startup_check_only,
            extra_args=extra_args,
        )
        started = time.perf_counter()
        try:
            result = subprocess.run(
                command,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        except Exception as exc:
            return ActionResult(
                status=UiStatus.FAILED,
                message=str(exc),
                command_preview=self.command_preview(command),
                stderr=str(exc),
                return_code=1,
            )
        elapsed_ms = int((time.perf_counter() - started) * 1000.0)
        message = f"命令完成，用时 {elapsed_ms} ms，返回码 {result.returncode}"
        status = UiStatus.SUCCESS if result.returncode == 0 else UiStatus.FAILED
        return ActionResult(
            status=status,
            message=message,
            command_preview=self.command_preview(command),
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.returncode,
        )

    def read_recent_events(self, event_stream_path: str, limit: int = 20) -> List[EventRecord]:
        path = Path(event_stream_path)
        if not path.exists():
            return []
        rows: List[EventRecord] = []
        for raw_line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
            try:
                payload = json.loads(raw_line)
            except Exception:
                continue
            rows.append(
                EventRecord(
                    timestamp=str(payload.get("ts", "")),
                    event_name=str(payload.get("event", "")),
                    payload_summary=_shorten_payload(payload),
                    raw_payload=payload,
                )
            )
        return list(reversed(rows))

    def read_run_report(self, report_path: str) -> Dict[str, object]:
        path = Path(report_path)
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}


class DeviceRegistryService:
    def build_panels(self, profile: RuntimeProfile, runtime: RuntimeControlService) -> List[DevicePanelState]:
        mode_label = "仿真帧源" if profile.run_mode == RunMode.SIMULATION else "ToupView"
        image_summary = profile.frame_source_image if profile.run_mode == RunMode.SIMULATION else profile.window_title
        return [
            DevicePanelState(
                title="图像源",
                status=UiStatus.NORMAL,
                summary=mode_label,
                details={
                    "source": image_summary or "-",
                    "roi": "启用" if profile.select_roi else "跳过",
                    "sim_jitter_px": str(profile.sim_jitter_px),
                    "sim_noise_std": str(profile.sim_noise_std),
                },
            ),
            DevicePanelState(
                title="XY 子系统",
                status=UiStatus.NORMAL,
                summary=profile.xy_driver,
                details={
                    "driver": profile.xy_driver,
                    "newport_conn": str(profile.newport_conn),
                    "mrc_virtual_axis_mode": profile.mrc_virtual_axis_mode,
                    "allocation_preview": self.mrc_allocation_preview(profile),
                },
            ),
            DevicePanelState(
                title="Z 子系统",
                status=UiStatus.NORMAL,
                summary=profile.z_driver,
                details={
                    "driver": profile.z_driver,
                    "z_step": str(profile.z_step),
                    "z_picomotor_conn": str(profile.z_picomotor_conn),
                    "z_picomotor_axis": str(profile.z_picomotor_axis),
                },
            ),
            DevicePanelState(
                title="检测后端",
                status=UiStatus.NORMAL,
                summary=profile.detector_backend,
                details={
                    "requested": profile.detector_backend,
                    "resolved": runtime.resolve_backend(profile),
                    "classic_method": profile.classic_method,
                    "classic_selection": profile.classic_selection,
                },
            ),
            DevicePanelState(
                title="环境与启动检查",
                status=UiStatus.NORMAL,
                summary="preflight",
                details={
                    "check_env": "可执行",
                    "startup_motion_check": "启用" if profile.startup_motion_check_enabled else "禁用",
                    "startup_timeout": str(profile.startup_motion_check_timeout),
                    "startup_xy_steps": str(profile.startup_motion_check_xy_steps),
                    "startup_z_step": str(profile.startup_motion_check_z_step),
                },
            ),
        ]

    def mrc_allocation_preview(self, profile: RuntimeProfile) -> str:
        if profile.xy_driver != "newport-mrc4":
            return "当前未启用 MRC 4轴"
        mode = profile.mrc_virtual_axis_mode
        if mode in ("mirror1", "反射镜1"):
            return f"虚拟 X/Y -> ({profile.mrc_mirror1_x_axis}, {profile.mrc_mirror1_y_axis})"
        if mode in ("mirror2", "反射镜2"):
            return f"虚拟 X/Y -> ({profile.mrc_mirror2_x_axis}, {profile.mrc_mirror2_y_axis})"
        return (
            "虚拟 X -> "
            f"({profile.mrc_mirror1_x_axis}, {profile.mrc_mirror2_x_axis}), "
            "虚拟 Y -> "
            f"({profile.mrc_mirror1_y_axis}, {profile.mrc_mirror2_y_axis})"
        )


class ModuleCatalogService:
    def list_modules(self, profile: Optional[RuntimeProfile] = None) -> List[ModuleViewItem]:
        enabled = set(profile.experimental_module_filters if profile else [])
        docs_root = Path("Document") / "SpotZoom_Machine_Learning_Unified"
        rows: List[ModuleViewItem] = []
        for spec in list_module_specs():
            status = UiStatus.DISABLED
            reason = ""
            if spec.module_name in enabled:
                status = UiStatus.RUNNING
            elif spec.primary_symbol:
                status = UiStatus.NORMAL
            else:
                status = UiStatus.WARNING
                reason = "缺少统一主入口"
            rows.append(
                ModuleViewItem(
                    version=spec.version,
                    name=spec.module_name,
                    type_key=spec.optimization_type.value,
                    type_label=TYPE_LABELS.get(spec.optimization_type.value, spec.optimization_type.value),
                    title=spec.title,
                    summary=spec.summary,
                    status=status,
                    placement=PLACEMENT_HINTS.get(spec.optimization_type.value, "仅实验模式"),
                    import_path=spec.import_path,
                    source_path=spec.source_path,
                    docs_path=str(docs_root / f"v{spec.version}_{spec.module_name}.md"),
                    primary_symbol=spec.primary_symbol or "",
                    config_symbol=spec.config_symbol or "",
                    dependency_reason=reason,
                )
            )
        return rows


class DeviceTestService:
    def __init__(self, runtime: RuntimeControlService):
        self.runtime = runtime

    def list_tests(self) -> List[TestCaseSpec]:
        return [
            TestCaseSpec("env_check", "连接测试", "环境检查", "运行 SpotZoom --check-env 并汇总结果"),
            TestCaseSpec("startup_check", "连接测试", "启动运动自检", "仅执行启动运动检测，不进入完整准直"),
            TestCaseSpec("picomotor_usb_count", "连接测试", "枚举 Picomotor USB 数量", "查询 8742 控制器数量"),
            TestCaseSpec("picomotor_axis_available", "连接测试", "测试指定轴可用性", "读取当前控制器可用轴列表"),
            TestCaseSpec("xps_connection", "连接测试", "测试 XPS 连接", "尝试连接并立刻释放 XPS"),
            TestCaseSpec("toupview_capture", "图像测试", "测试 ToupView 捕获", "尝试连接窗口并抓取一帧"),
            TestCaseSpec("simulated_frame", "图像测试", "测试模拟图像源", "加载本地图像作为模拟帧"),
            TestCaseSpec("yolo_backend", "图像测试", "测试 YOLO 后端", "验证 YOLO 依赖与模型路径"),
            TestCaseSpec("classic_backend", "图像测试", "测试 Classic 后端", "验证 classic 检测器可用性"),
            TestCaseSpec("mrc_virtual_x", "运动测试", "MRC shared 虚拟 X 测试", "执行一次虚拟 X 修正"),
            TestCaseSpec("mrc_virtual_y", "运动测试", "MRC shared 虚拟 Y 测试", "执行一次虚拟 Y 修正"),
            TestCaseSpec("picomotor_z_up", "运动测试", "Picomotor Z 上移一步", "执行独立 Z 轴上移"),
            TestCaseSpec("picomotor_z_down", "运动测试", "Picomotor Z 下移一步", "执行独立 Z 轴下移"),
        ]

    def run_test(self, test_id: str, profile: RuntimeProfile) -> TestResult:
        started = time.perf_counter()
        try:
            if test_id == "env_check":
                action = self.runtime.run_command(profile, check_env=True, timeout=180.0)
                return self._from_action(test_id, "环境检查", action, started)
            if test_id == "startup_check":
                action = self.runtime.run_command(profile, startup_check_only=True, timeout=180.0)
                return self._from_action(test_id, "启动运动自检", action, started)
            if test_id == "picomotor_usb_count":
                count = SpotZoom.PicoMotor8742Controller.usb_device_count()
                return self._success(test_id, "枚举 Picomotor USB 数量", started, str(count), f"usb_count={count}")
            if test_id == "picomotor_axis_available":
                controller = SpotZoom.PicoMotor8742Controller(conn=profile.newport_conn).open()
                try:
                    axes = controller.axes()
                finally:
                    controller.close()
                return self._success(test_id, "测试指定轴可用性", started, str(axes), f"axes={axes}")
            if test_id == "xps_connection":
                axis = SpotZoom.XPSZAxis(
                    ip=profile.xps_ip,
                    port=profile.xps_port,
                    username=profile.xps_user,
                    password=profile.xps_password,
                    group_name=profile.xps_group,
                )
                axis.close()
                return self._success(test_id, "测试 XPS 连接", started, "连接成功", "xps_connected=true")
            if test_id == "toupview_capture":
                window = SpotZoom.ToupViewWindow(
                    title_keyword=profile.window_title,
                    wait_timeout_s=profile.window_wait_seconds,
                )
                try:
                    frame = self._capture_frame(window)
                finally:
                    close = getattr(window, "close", None)
                    if callable(close):
                        close()
                return self._success(
                    test_id,
                    "测试 ToupView 捕获",
                    started,
                    f"frame={frame.shape[1]}x{frame.shape[0]}",
                    f"frame_shape={frame.shape}",
                )
            if test_id == "simulated_frame":
                image_path = str(profile.frame_source_image or self.runtime.ensure_sample_frame())
                window = SpotZoom.SimulatedFrameWindow(
                    image_path=image_path,
                    jitter_px=profile.sim_jitter_px,
                    noise_std=profile.sim_noise_std,
                )
                frame = self._capture_frame(window)
                return self._success(
                    test_id,
                    "测试模拟图像源",
                    started,
                    f"frame={frame.shape[1]}x{frame.shape[0]}",
                    image_path,
                )
            if test_id == "yolo_backend":
                ok, message = SpotZoom._check_yolo_backend_prereqs(self.runtime._namespace_for_backend(profile))
                if ok:
                    return self._success(test_id, "测试 YOLO 后端", started, "YOLO 可用", "yolo_ok=true")
                return self._failure(test_id, "测试 YOLO 后端", started, message or "YOLO 不可用")
            if test_id == "classic_backend":
                ok = SpotZoom._classic_backend_available()
                if ok:
                    return self._success(test_id, "测试 Classic 后端", started, "classic 可用", "classic_ok=true")
                return self._failure(test_id, "测试 Classic 后端", started, "classic 检测器不可用")
            if test_id == "mrc_virtual_x":
                return self._run_mrc_motion(test_id, "MRC shared 虚拟 X 测试", profile, axis="x", steps=1, started=started)
            if test_id == "mrc_virtual_y":
                return self._run_mrc_motion(test_id, "MRC shared 虚拟 Y 测试", profile, axis="y", steps=1, started=started)
            if test_id == "picomotor_z_up":
                return self._run_picomotor_z(test_id, "Picomotor Z 上移一步", profile, direction="up", started=started)
            if test_id == "picomotor_z_down":
                return self._run_picomotor_z(test_id, "Picomotor Z 下移一步", profile, direction="down", started=started)
            return self._failure(test_id, test_id, started, "未知测试项")
        except Exception as exc:
            return self._failure(test_id, test_id, started, str(exc))

    def _capture_frame(self, window) -> np.ndarray:
        grab = getattr(window, "grab_frame", None)
        if callable(grab):
            return grab()
        capture = getattr(window, "capture_frame", None)
        if callable(capture):
            return capture()
        raise RuntimeError(f"{window.__class__.__name__} 缺少 grab_frame/capture_frame 接口")

    def _run_mrc_motion(
        self,
        test_id: str,
        title: str,
        profile: RuntimeProfile,
        *,
        axis: str,
        steps: int,
        started: float,
    ) -> TestResult:
        stage = SpotZoom.NewportMRC4MirrorStage(
            conn=profile.newport_conn,
            mirror1_x_axis=profile.mrc_mirror1_x_axis,
            mirror1_y_axis=profile.mrc_mirror1_y_axis,
            mirror2_x_axis=profile.mrc_mirror2_x_axis,
            mirror2_y_axis=profile.mrc_mirror2_y_axis,
            mirror1_x_sign=profile.mrc_mirror1_x_sign,
            mirror1_y_sign=profile.mrc_mirror1_y_sign,
            mirror2_x_sign=profile.mrc_mirror2_x_sign,
            mirror2_y_sign=profile.mrc_mirror2_y_sign,
            backend=profile.newport_backend,
            timeout=profile.newport_timeout,
            multiaddr=profile.newport_multiaddr,
            scan=not profile.newport_no_scan,
            velocity=profile.newport_velocity,
            acceleration=profile.newport_acceleration,
            wait_each_move=not profile.newport_no_wait,
            virtual_axis_mode=profile.mrc_virtual_axis_mode,
        )
        try:
            if axis == "x":
                stage.move_x(steps)
            else:
                stage.move_y(steps)
        finally:
            stage.close()
        return self._success(test_id, title, started, f"{axis}={steps}", f"virtual_{axis}_steps={steps}")

    def _run_picomotor_z(
        self,
        test_id: str,
        title: str,
        profile: RuntimeProfile,
        *,
        direction: str,
        started: float,
    ) -> TestResult:
        axis = SpotZoom.NewportPicomotorZAxis(
            conn=profile.z_picomotor_conn,
            axis=profile.z_picomotor_axis,
            hw_sign=profile.z_picomotor_sign,
            backend=profile.newport_backend,
            timeout=profile.newport_timeout,
            multiaddr=profile.newport_multiaddr,
            scan=not profile.newport_no_scan,
            velocity=profile.z_picomotor_velocity,
            acceleration=profile.z_picomotor_acceleration,
        )
        try:
            if direction == "up":
                axis.move_up(profile.z_step)
            else:
                axis.move_down(profile.z_step)
        finally:
            axis.close()
        return self._success(test_id, title, started, direction, f"z_step={profile.z_step}")

    def _from_action(self, test_id: str, title: str, action: ActionResult, started: float) -> TestResult:
        elapsed = int((time.perf_counter() - started) * 1000.0)
        raw = action.stdout if action.stdout else action.stderr
        return TestResult(
            test_id=test_id,
            title=title,
            status=action.status,
            duration_ms=elapsed,
            message=action.message,
            raw_return=raw or action.command_preview,
        )

    def _success(self, test_id: str, title: str, started: float, message: str, raw_return: str) -> TestResult:
        return TestResult(
            test_id=test_id,
            title=title,
            status=UiStatus.SUCCESS,
            duration_ms=int((time.perf_counter() - started) * 1000.0),
            message=message,
            raw_return=raw_return,
        )

    def _failure(self, test_id: str, title: str, started: float, message: str) -> TestResult:
        return TestResult(
            test_id=test_id,
            title=title,
            status=UiStatus.FAILED,
            duration_ms=int((time.perf_counter() - started) * 1000.0),
            message=message,
            raw_return=message,
        )
