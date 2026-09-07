"""Qt UI（PLAN 第 6 节，PyQt5 兼容层）。

功能：场景选择、dryrun 运行（QThread）、帧+路径预览、状态/置信度面板、
急停/暂停按钮、JSONL 报告回放。全部基于 dryrun，不触碰真实设备。
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import traceback
from typing import Optional

import cv2
import numpy as np

if __package__ in (None, ""):  # 支持直接运行：python app.py
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from obstacle_avoidance import sim_microscope, video_sim
    from obstacle_avoidance.qt_compat import QtCore, QtGui, QtWidgets, Signal, Slot
    from obstacle_avoidance.cli import SCENARIOS, build_scenario
    from obstacle_avoidance.controller import ObstacleAvoidController
    from obstacle_avoidance.models import RunState
    from obstacle_avoidance.motion import (MotionConfigError, MotionLimitError,
                                            MotionStateError)
    from obstacle_avoidance.planner import CollisionModel
    from obstacle_avoidance.reporter import replay, summarize
    from obstacle_avoidance.roi_zones import (LayoutValidationError, RoiConfig,
                                               Zone, rect_contains_rect,
                                               validate_sim_layout)
    from obstacle_avoidance.vision import (get_shared_detector,
                                           try_shared_detector,
                                           warmup_shared_detector_async)
else:
    from . import sim_microscope, video_sim
    from .qt_compat import QtCore, QtGui, QtWidgets, Signal, Slot
    from .cli import SCENARIOS, build_scenario
    from .controller import ObstacleAvoidController
    from .models import RunState
    from .motion import (MotionConfigError, MotionLimitError,
                         MotionStateError)
    from .planner import CollisionModel
    from .reporter import replay, summarize
    from .roi_zones import (LayoutValidationError, RoiConfig, Zone,
                            rect_contains_rect, validate_sim_layout)
    from .vision import (get_shared_detector, try_shared_detector,
                         warmup_shared_detector_async)


# ---------------------------------------------------------------- 错误处理
# 后台日志（不退出）+ 窗口弹窗：任何未捕获异常只记录/提示，绝不终止程序。
LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "artifacts", "ui_error.log")

_logger: Optional[logging.Logger] = None


def get_ui_logger() -> logging.Logger:
    """UI 错误日志：artifacts/ui_error.log（追加）+ 控制台。"""
    global _logger
    if _logger is None:
        _logger = logging.getLogger("obstacle_avoidance.ui")
        if not _logger.handlers:
            _logger.setLevel(logging.INFO)
            os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
            fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
            fh.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s"))
            _logger.addHandler(fh)
            sh = logging.StreamHandler()
            sh.setFormatter(logging.Formatter("[ui-log] %(message)s"))
            _logger.addHandler(sh)
            _logger.propagate = False
    return _logger


def _log_exception(context: str, exc: BaseException) -> None:
    """后台日志记录完整 traceback。"""
    get_ui_logger().error("%s: %s\n%s", context, exc,
                          traceback.format_exc())


def install_global_excepthook() -> None:
    """兜底钩子：主线程异常 -> 日志+弹窗；子线程异常 -> 日志。
    替代 PyQt5 默认的 '未捕获即 qFatal 闪退' 行为。"""
    def _handle(exc_type, exc_value, exc_tb) -> None:
        _log_exception("unhandled exception", exc_value)
        if threading.current_thread() is threading.main_thread():
            try:
                QtWidgets.QMessageBox.critical(
                    None, "程序错误",
                    f"发生未处理异常（已记录日志，程序继续运行）:\n\n"
                    f"{exc_type.__name__}: {exc_value}")
            except Exception:  # noqa: BLE001 - 无 QApplication 时兜底
                pass
        else:
            print(f"[ui-log] background error: "
                  f"{exc_type.__name__}: {exc_value}", file=sys.stderr)
    sys.excepthook = _handle
    if hasattr(threading, "excepthook"):
        threading.excepthook = lambda a: _handle(a.exc_type, a.exc_value,
                                                 a.exc_traceback)


class WorkerThread(QtCore.QThread):
    """在后台线程跑 dryrun 场景，信号推送帧/路径/状态到 UI。"""

    frame_ready = Signal(np.ndarray)         # BGR 帧（含路径叠加）
    state_ready = Signal(str, str)           # state, detail
    metrics_ready = Signal(dict)
    finished_run = Signal(int)               # exit code
    error_occurred = Signal(str)             # 需弹窗的错误信息
    log_ready = Signal(str)                  # 需求5：运行期事件 -> UI 操作日志
    motion_ready = Signal(dict)              # XYZ 运动遥测

    def __init__(self, scenario: str, controller_ref: dict, parent=None,
                 builder=None, sample_spec: Optional[dict] = None) -> None:
        super().__init__(parent)
        self.scenario = scenario
        self._controller_ref = controller_ref
        self._builder = builder   # None -> build_scenario(scenario)；电机模式注入
        self._sample_spec = sample_spec   # sim01 仿真镜头图层配置
        self._stage_sink: list = []   # run 注册真实 stage -> UI 急停直接 stop_all
        self._sim_layout: Optional[dict] = None   # sim01 多衬底 ROI 布局
        self._run_mode: str = "oa"                # "oa" 避障 / "ag" 组装

    def toggle_pause(self) -> bool:
        """Toggle the active controller pause state."""
        ctl = self._controller_ref.get("controller")
        if ctl is None:
            return False
        paused = getattr(ctl, "_pause", None)
        if paused is not None and paused.is_set():
            ctl.resume()
            return False
        ctl.pause()
        return True

    def request_estop(self) -> None:
        """Request controller stop and latch every active stage."""
        ctl = self._controller_ref.get("controller")
        if ctl is not None:
            ctl.request_estop()
        for stage in list(self._stage_sink):
            stop = getattr(stage, "stop_all", None)
            if callable(stop):
                try:
                    stop()
                except Exception as exc:  # noqa: BLE001
                    _log_exception("worker estop failed", exc)

    @property
    def stages(self) -> list:
        """运行期间注册的真实位移台（PicoMotorStage/SerialXYStage）。"""
        return self._stage_sink

    # ---- 帧推送与路径叠加
    def _latest_plan_points(self, rep):
        """从报告事件中取最近一次成功规划的 waypoint（倒序查找）。"""
        for e in reversed(rep.events):
            if e.get("event") == "plan" and e.get("success"):
                return e.get("waypoints_px") or []
        return []

    @staticmethod
    def _draw_overlay(img, plan_pts):
        """BGR 帧上叠加规划路径（绿）与 waypoint（黄点）。返回副本。"""
        import cv2
        out = img.copy()
        if plan_pts:
            pts = np.array([(int(round(x)), int(round(y))) for x, y in plan_pts],
                           dtype=np.int32)
            cv2.polylines(out, [pts], False, (0, 200, 0), 2)
            for p in pts:
                cv2.circle(out, (int(p[0]), int(p[1])), 3, (0, 220, 220), -1)
        return out

    def _run_single_oa(self, world, rep, ball_rect, goal_pt):
        """单球避障：把 ball_rect 对应的球移动到 goal_pt。"""
        if __package__ in (None, ""):
            from obstacle_avoidance.controller import ControllerConfig, ObstacleAvoidController
            from obstacle_avoidance.planner import GridPlanner
            from obstacle_avoidance.vision import VisionPipeline, ParticleTracker
            from obstacle_avoidance.sim_microscope import _initial_detect
            from obstacle_avoidance.models import GoalRegion
        else:
            from .controller import ControllerConfig, ObstacleAvoidController
            from .planner import GridPlanner
            from .vision import VisionPipeline, ParticleTracker
            from .sim_microscope import _initial_detect
            from .models import GoalRegion
        detector = world.make_detector()
        cx = ball_rect[0] + ball_rect[2] / 2
        cy = ball_rect[1] + ball_rect[3] / 2
        snap, tid = _initial_detect(world, detector, hint=(cx, cy),
                                    pipeline=VisionPipeline(
                                        detector,
                                        tracker=ParticleTracker(
                                            max_jump_px=220.0)))
        if tid is None:
            return None
        goal = GoalRegion(center=(goal_pt[0], goal_pt[1]), radius_px=25)
        cfg = ControllerConfig(
            max_step_mm=0.05, tolerance_px=8.0, stable_frames=3,
            max_iterations=400, max_track_jump_px=250.0)
        stage = world.make_stage(tid)
        self._stage_sink.append(stage)
        planner = GridPlanner()
        pipeline = VisionPipeline(
            detector, tracker=ParticleTracker(max_jump_px=220.0))
        ctl = ObstacleAvoidController(stage, pipeline, planner, cfg, rep)
        self._controller_ref["controller"] = ctl
        self._controller_ref["controller"] = ctl
        return ctl.run(snap, tid, goal, task_id="sim01-oa",
                       get_frame=world.render, get_snapshot=world.snapshot)

    def run(self) -> None:  # noqa: D102 - QThread 入口
        if __package__ in (None, ""):
            from obstacle_avoidance.controller import ControllerConfig, ObstacleAvoidController
            from obstacle_avoidance.planner import GridPlanner
            from obstacle_avoidance.vision import VisionPipeline
            from obstacle_avoidance.reporter import RunReporter
            from obstacle_avoidance.aggregation import AggregationConfig, AggregationPlanner
            from obstacle_avoidance.cli import build_scenario as _cli_build
            from obstacle_avoidance.models import GoalRegion
        else:
            from .controller import ControllerConfig, ObstacleAvoidController
            from .planner import GridPlanner
            from .vision import VisionPipeline
            from .reporter import RunReporter
            from .aggregation import AggregationConfig, AggregationPlanner
            from .cli import build_scenario as _cli_build
            from .models import GoalRegion
        try:
            # ---- sim01 多 ground 模式：从 UI 布局构建 dispatcher ----
            if self._sim_layout and self.scenario == "sim01":
                layout = self._sim_layout
                mode = self._run_mode
                try:
                    ball_groups = validate_sim_layout(layout, mode=mode)
                except LayoutValidationError as exc:
                    self.state_ready.emit("ABORTED", str(exc))
                    self.error_occurred.emit(str(exc))
                    self.finished_run.emit(1)
                    return
                world, _ = _cli_build("sim01",
                                      sample_spec=self._sample_spec,
                                      sim_layout=layout)
                rep = RunReporter(None)
                orig_render = world.render
                # ---- 需求5：包装 make_stage，逐条捕获台位移动 -> 日志
                orig_make_stage = world.make_stage

                def make_stage_logged(*a, **k):
                    st = orig_make_stage(*a, **k)
                    orig_move = st.move_by

                    def move_by_logged(dx, dy, *a2, **k2):
                        ok = orig_move(dx, dy, *a2, **k2)
                        self.log_ready.emit(
                            f"move dx={dx * 1000:+.1f}µm dy={dy * 1000:+.1f}µm")
                        return ok
                    st.move_by = move_by_logged
                    return st
                world.make_stage = make_stage_logged
                # ---- 需求5：增量推送运行内部事件（规划/状态机/错误/结束）
                ev_state = {"n": 0}

                def pump_events():
                    try:
                        events = rep.events
                    except Exception:  # noqa: BLE001
                        return
                    for ev in events[ev_state["n"]:]:
                        ev_state["n"] += 1
                        et = ev.get("event", "?")
                        if et == "plan":
                            self.log_ready.emit(
                                f"规划 ok={ev.get('success')} "
                                f"waypoints={ev.get('waypoint_count', '?')}")
                        elif et == "state_change":
                            self.log_ready.emit(
                                f"状态 {ev.get('from')} -> {ev.get('to')}")
                        elif et == "error":
                            self.log_ready.emit(f"错误: {ev.get('reason')}")
                        elif et == "run_end":
                            self.log_ready.emit(
                                f"运行结束: {ev.get('final_state')}")
                        elif et in ("estop", "pause", "detection"):
                            self.log_ready.emit(
                                f"{et}: {ev.get('action', ev.get('reason', ''))}")

                def render_and_emit():
                    img = orig_render()
                    pump_events()
                    telemetry = getattr(getattr(world, "motion_stage", None),
                                        "last_telemetry", None)
                    if telemetry is not None:
                        self.motion_ready.emit(telemetry.to_dict())
                    overlay = self._draw_overlay(img, self._latest_plan_points(rep))
                    self.frame_ready.emit(overlay)
                    return img
                world.render = render_and_emit
                self.frame_ready.emit(self._draw_overlay(orig_render(), []))

                # 球按 ground 分组
                grounds = layout.get("grounds") or []
                balls = layout.get("balls") or []
                goals = layout.get("ground_goals") or [None] * len(grounds)
                goal_ranges = layout.get("ground_goal_ranges") or [None] * len(grounds)

                def _in_ground(bcx, bcy, grect):
                    return (grect[0] <= bcx <= grect[0] + grect[2]
                            and grect[1] <= bcy <= grect[1] + grect[3])

                # ball_groups was produced by validate_sim_layout so each
                # ball belongs to exactly one substrate.

                # 逐 ground 串行执行
                final_state = None
                run_failed = False
                for gi, grect in enumerate(grounds):
                    ball_idx = ball_groups[gi]
                    world.substrate = world.substrate.__class__(
                        polygon=[(grect[0], grect[1]),
                                 (grect[0] + grect[2], grect[1]),
                                 (grect[0] + grect[2], grect[1] + grect[3]),
                                 (grect[0], grect[1] + grect[3])],
                        safety_margin_px=4.0)
                    if not ball_idx:
                        self.state_ready.emit(f"G{gi} 无球，跳过", "")
                        continue
                    self.state_ready.emit(
                        f"G{gi} 运行 [{mode}] 球数={len(ball_idx)}", "")

                    # 用 sim_microscope 构建单球目标（选第一个球），
                    # 多球时用 AggregationPlanner 或逐球串行
                    if mode == "oa":
                        goal_pt = goals[gi]
                        if not goal_pt: continue
                        # 逐球串行：先移动第一个球到目标，然后第二个球...
                        for bi in ball_idx:
                            # 把该球设为目标球，其他球当障碍
                            result = self._run_single_oa(world, rep, balls[bi], goal_pt)
                            if result is None or result.final_state != RunState.COMPLETE:
                                final_state = getattr(result, "final_state", RunState.ABORTED)
                                run_failed = True
                                break
                            pump_events()   # run_end 在最后一次渲染后，需补泵
                    else:  # ag
                        gr = goal_ranges[gi]
                        if not gr: continue
                        cx, cy = gr[0] + gr[2] / 2, gr[1] + gr[3] / 2
                        goal = GoalRegion(center=(cx, cy),
                                          radius_px=max(min(gr[2], gr[3]) / 2, 20))
                        # AggregationPlanner.run 把所有检出球聚到 goal
                        if world.pipeline is None:
                            if __package__ in (None, ""):
                                from obstacle_avoidance.sim_microscope import _initial_detect
                                from obstacle_avoidance.vision import ParticleTracker
                            else:
                                from .sim_microscope import _initial_detect
                                from .vision import ParticleTracker
                            detector = world.make_detector()
                            _initial_detect(
                                world, detector, hint=None,
                                pipeline=VisionPipeline(
                                    detector,
                                    tracker=ParticleTracker(max_jump_px=220.0)))
                        ag = AggregationPlanner(
                            GridPlanner(),
                            world.pipeline or VisionPipeline(),
                            AggregationConfig(
                                required_count=len(ball_idx),
                                controller=ControllerConfig(
                                    max_step_mm=0.05,
                                    tolerance_px=8.0,
                                    stable_frames=3,
                                    max_iterations=400,
                                    max_track_jump_px=250.0,
                                    prefer_track_id=True)),
                            rep, controller_sink=self._controller_ref)
                        result = ag.run(world, world.snapshot(),
                                        goal, task_id=f"sim01-G{gi}-ag")
                        pump_events()   # run_end 在最后一次渲染后，需补泵
                        final_state = result.final_state
                        run_failed = result.final_state != RunState.COMPLETE
                    if run_failed:
                        break
                world.close()
                self.state_ready.emit(
                    final_state.value if final_state else "COMPLETE",
                    f"sim01 [{mode}] 全部 ground 完成")
                self.finished_run.emit(0 if not run_failed and
                                       final_state in (None, RunState.COMPLETE)
                                       else 1)
                return

            # ---- 默认路径（单 ground / 其他场景） ----
            world, run_fn = (self._builder() if self._builder is not None
                             else _cli_build(self.scenario,
                                             sample_spec=self._sample_spec))
            rep = RunReporter(None)
            orig_render = world.render

            def render_and_emit():
                img = orig_render()
                telemetry = getattr(getattr(world, "motion_stage", None),
                                    "last_telemetry", None)
                if telemetry is not None:
                    self.motion_ready.emit(telemetry.to_dict())
                overlay = self._draw_overlay(img, self._latest_plan_points(rep))
                self.frame_ready.emit(overlay)
                return img

            world.render = render_and_emit
            # 先推送一帧初始场景
            self.frame_ready.emit(self._draw_overlay(orig_render(), []))
            result = run_fn(rep=rep, stage_sink=self._stage_sink)
        except Exception as exc:  # noqa: BLE001 - UI 层兜底
            _log_exception("run failed", exc)
            msg = f"{type(exc).__name__}: {exc}"
            self.state_ready.emit("FAULT", msg)
            self.error_occurred.emit(msg)
            self.finished_run.emit(1)
            return
        # 结束帧：run_fn 的 finally 可能已释放场景资源（如 sim01 相机 disable），
        # 渲染失败时降级复用 world 缓存的最后一帧——不得让结束帧异常
        # 逃出 QThread（否则下方状态/指标 emit 全部被跳过，UI 卡运行态）。
        try:
            final_frame = orig_render()
        except Exception as exc:  # noqa: BLE001 - 资源已释放属预期
            _log_exception("final frame render failed", exc)
            final_frame = (getattr(world, "last_frame", None)
                           or getattr(world, "_last_frame", None))
        if final_frame is not None:
            self.frame_ready.emit(self._draw_overlay(
                final_frame, self._latest_plan_points(rep)))
        self.state_ready.emit(result.final_state.value,
                              getattr(result, "detail", "") or "")
        self.metrics_ready.emit(result.to_dict())
        self.finished_run.emit(0 if result.final_state == RunState.COMPLETE
                               or getattr(result, "completed", False) else 1)


def frame_to_pix(bgr: np.ndarray) -> QtGui.QImage:
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    h, w, _ = rgb.shape
    img = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
    return img.copy()


class LiveDetectThread(QtCore.QThread):
    """实时检测推理线程：只做 YOLO 推理（丢帧策略），绝不阻塞 UI。"""

    dets_ready = Signal(list)   # [(center, r, conf), ...]

    def __init__(self, detector_fn, parent=None) -> None:
        super().__init__(parent)
        self._detector_fn = detector_fn    # () -> detector or None
        self._lock = threading.Lock()
        self._frame = None
        self._has_new = False
        self._running = True

    def submit(self, frame: np.ndarray) -> None:
        """非阻塞提交最新帧（覆盖旧帧，推理跟不上时自动丢帧）。"""
        with self._lock:
            self._frame = frame
            self._has_new = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:  # noqa: D102
        while self._running:
            with self._lock:
                frame, new = self._frame, self._has_new
                self._has_new = False
            if not new:
                self.msleep(15)
                continue
            det = self._detector_fn()
            if det is None:
                self.msleep(50)
                continue
            try:
                dets, _ = det.detect_particles(frame)
            except Exception:  # noqa: BLE001 - 单帧失败不中断实时流
                continue
            self.dets_ready.emit(dets)


class PreviewThread(QtCore.QThread):
    """场景预览构建线程：sim01/video 场景构建耗时，后台执行 UI 不冻结。"""

    preview_ready = Signal(np.ndarray)
    preview_failed = Signal(str)

    def __init__(self, scenario: str, parent=None,
                 sample_spec: Optional[dict] = None) -> None:
        super().__init__(parent)
        self.scenario = scenario
        self._sample_spec = sample_spec

    def run(self) -> None:  # noqa: D102
        try:
            world, _ = build_scenario(self.scenario,
                                      sample_spec=self._sample_spec)
            try:
                self.preview_ready.emit(world.render())
            finally:
                close = getattr(world, "close", None)  # sim01 释放相机/DB
                if callable(close):
                    close()
        except Exception as exc:  # noqa: BLE001
            self.preview_failed.emit(f"{type(exc).__name__}: {exc}")


class SimSpecDialog(QtWidgets.QDialog):
    """仿真镜头图层配置：掩码/衬底/障碍物的数量、类别标签与形状。

    每类字段：count 数量 / shape 形状 / label 标签前缀 / size 基准尺寸 /
    custom 自定义多边形顶点（"x,y x,y ..."，shape=custom 时生效）。
    """

    SHAPES = ["ellipse", "blob", "rect", "triangle", "star", "custom"]
    CATS = [("ground", "衬底 ground"), ("mask", "掩码 mask"),
            ("obstacle", "障碍物 obstacle")]

    def __init__(self, spec: dict, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("仿真镜头图层配置")
        form = QtWidgets.QFormLayout(self)
        self._fields: dict = {}
        for cat, zh in self.CATS:
            cfg = spec.get(cat) or {}
            count = QtWidgets.QSpinBox()
            count.setRange(0, 50)
            count.setValue(int(cfg.get("count", 0) or 0))
            shape = QtWidgets.QComboBox()
            shape.addItems(self.SHAPES)
            shape.setCurrentText(str(cfg.get("shape", "ellipse")))
            label = QtWidgets.QLineEdit(str(cfg.get("label", "")))
            size = QtWidgets.QDoubleSpinBox()
            size.setRange(2.0, 500.0)
            size.setValue(float(cfg.get("size", 40.0) or 40.0))
            custom = QtWidgets.QLineEdit(str(cfg.get("custom", "")))
            custom.setToolTip("自定义多边形顶点：\"x,y x,y ...\"（shape=custom）")
            sub = QtWidgets.QGridLayout()
            for col, (name, w) in enumerate((
                    ("数量", count), ("形状", shape), ("标签", label),
                    ("尺寸", size), ("顶点", custom))):
                sub.addWidget(QtWidgets.QLabel(name), 0, col)
                sub.addWidget(w, 1, col)
            group = QtWidgets.QGroupBox(zh)
            group.setLayout(sub)
            form.addRow(group)
            self._fields[cat] = (count, shape, label, size, custom)
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def spec(self) -> dict:
        """收集表单 -> sample_spec（空标签沿用内置默认前缀）。"""
        out = {}
        for cat, (count, shape, label, size, custom) in self._fields.items():
            cfg = {"count": count.value(), "shape": shape.currentText(),
                   "size": size.value()}
            if label.text().strip():
                cfg["label"] = label.text().strip()
            if shape.currentText() == "custom" and custom.text().strip():
                cfg["custom"] = custom.text().strip()
            out[cat] = cfg
        return out


class Canvas(QtWidgets.QLabel):
    """帧预览画布：把鼠标拖拽映射为帧坐标矩形（ROI/区域划分用）。
    
    还支持实时模式下的位移台控制（mouse_dragged 信号）。
    """

    rect_drawn = Signal(int, int, int, int)  # x, y, w, h（帧坐标）
    drag_start = Signal(int, int)            # 帧坐标 x, y（鼠标按下）
    drag_move = Signal(int, int)             # 帧坐标增量 dx, dy（鼠标移动）
    drag_end = Signal()                      # 鼠标释放（拖拽结束）

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._mapper = None            # (qx,qy)->(fx,fy)
        self._origin = None            # 按下时的帧坐标
        self._last_pos = None          # 上次移动时的帧坐标
        self.setMouseTracking(False)   # 默认不跟踪；需时由外部开启

    def setMapper(self, mapper) -> None:
        self._mapper = mapper

    def _to_frame(self, qp: QtCore.QPoint):
        if self._mapper is None:
            return None
        return self._mapper(qp.x(), qp.y())

    @staticmethod
    def _event_pos(ev) -> QtCore.QPoint:
        """Qt5(QMouseEvent.pos)/Qt6(QMouseEvent.position) 兼容。"""
        p = getattr(ev, "position", None)
        return p.toPoint() if p is not None else ev.pos()

    def mousePressEvent(self, ev) -> None:  # noqa: N802
        if ev.button() == QtCore.Qt.LeftButton:
            fp = self._to_frame(self._event_pos(ev))
            self._origin = fp
            self._last_pos = fp
            if fp is not None:
                self.drag_start.emit(int(fp[0]), int(fp[1]))
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev) -> None:  # noqa: N802
        if ev.buttons() & QtCore.Qt.LeftButton and self._last_pos is not None:
            fp = self._to_frame(self._event_pos(ev))
            if fp is not None:
                dx = int(fp[0] - self._last_pos[0])
                dy = int(fp[1] - self._last_pos[1])
                if dx != 0 or dy != 0:
                    self.drag_move.emit(dx, dy)
                    self._last_pos = fp
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev) -> None:  # noqa: N802
        if ev.button() == QtCore.Qt.LeftButton and self._origin is not None:
            end = self._to_frame(self._event_pos(ev))
            if end is not None:
                x0, y0 = self._origin
                x, y = min(x0, end[0]), min(y0, end[1])
                w, h = abs(end[0] - x0), abs(end[1] - y0)
                if w > 4 and h > 4:
                    self.rect_drawn.emit(int(x), int(y), int(w), int(h))
            self._origin = None
            self._last_pos = None
            self.drag_end.emit()
        super().mouseReleaseEvent(ev)


class MainWindow(QtWidgets.QMainWindow):
    usb_count_ready = Signal(int)   # Picomotor USB 设备数（后台检测回填）

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ObstacleAvoid - dryrun 仿真")
        # Keep enough room for the canvas and the parameter panel. The panel
        # is scrollable, so smaller screens can still access every control.
        self.resize(1360, 800)
        self.setMinimumSize(1360, 640)
        self.worker: Optional[WorkerThread] = None
        self._controller_ref: dict = {}
        self._last_frame: Optional[np.ndarray] = None

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # 左：画布
        self.canvas = Canvas()
        self.canvas.setMinimumSize(640, 480)
        self.canvas.setStyleSheet("background:#222;color:#aaa;")
        self.canvas.setAlignment(QtCore.Qt.AlignCenter)
        self.canvas.setMapper(self._map_to_frame)
        self.canvas.rect_drawn.connect(self.on_rect_drawn)
        self.canvas.drag_start.connect(self._on_sim_drag_start)
        self.canvas.drag_move.connect(self._on_sim_drag_move)
        self.canvas.drag_end.connect(self._on_sim_drag_end)
        layout.addWidget(self.canvas, 3)

        # 右：面板
        panel_widget = QtWidgets.QWidget()
        panel_widget.setMinimumWidth(360)
        panel = QtWidgets.QVBoxLayout(panel_widget)
        panel.setContentsMargins(8, 8, 8, 8)
        panel.setSpacing(6)
        panel_scroll = QtWidgets.QScrollArea()
        panel_scroll.setWidget(panel_widget)
        panel_scroll.setWidgetResizable(True)
        panel_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        panel_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        # The XYZ grid contains four labelled columns; reserve enough width
        # for labels and numeric editors so values never overlap or clip.
        panel_scroll.setMinimumWidth(660)
        layout.addWidget(panel_scroll, 2)

        panel.addWidget(QtWidgets.QLabel("场景"))
        self.scenario_combo = QtWidgets.QComboBox()
        # 仅保留 sim01 仿真模拟模式（其余场景仍可经 CLI 使用）
        self.scenario_combo.addItems(["sim01"])
        panel.addWidget(self.scenario_combo)

        self.mode_sel = QtWidgets.QComboBox()
        self.mode_sel.addItems(["虚拟模式 (仿真)", "电机模式 (真实相机+电机)"])
        self.mode_sel.currentIndexChanged.connect(self.on_mode_changed)
        panel.addWidget(self.mode_sel)

        run_row = QtWidgets.QHBoxLayout()
        self.run_oa_btn = QtWidgets.QPushButton("避障运行")
        self.run_oa_btn.clicked.connect(lambda: self.on_run(mode="oa"))
        run_row.addWidget(self.run_oa_btn)
        self.run_ag_btn = QtWidgets.QPushButton("组装运行")
        self.run_ag_btn.clicked.connect(lambda: self.on_run(mode="ag"))
        run_row.addWidget(self.run_ag_btn)
        panel.addLayout(run_row)

        # 电机模式设置
        self.motor_box = QtWidgets.QGroupBox("电机模式设置")
        form = QtWidgets.QFormLayout(self.motor_box)
        self.driver_combo = QtWidgets.QComboBox()
        self.driver_combo.addItems(["8742/8743 Picomotor (USB)",
                                    "串口 G 代码位移台"])
        self.driver_combo.currentIndexChanged.connect(self.on_driver_changed)
        form.addRow("驱动", self.driver_combo)
        # -- Picomotor（pylablib）参数
        self.pico_widgets = []
        self.conn_spin = QtWidgets.QSpinBox()
        self.conn_spin.setRange(0, 10)
        self.conn_spin.setToolTip("USB 设备索引（0 起；可用'检测设备'查看数量）")
        form.addRow("USB 索引", self.conn_spin)
        self.pico_widgets += [self.conn_spin]
        self.pico_count_label = QtWidgets.QLabel("USB 设备: ?")
        form.addRow(self.pico_count_label)
        self.pico_widgets.append(self.pico_count_label)
        self.axis_x_spin = QtWidgets.QSpinBox()
        self.axis_x_spin.setRange(1, 4)
        self.axis_x_spin.setValue(1)
        form.addRow("X 轴号", self.axis_x_spin)
        self.axis_y_spin = QtWidgets.QSpinBox()
        self.axis_y_spin.setRange(1, 4)
        self.axis_y_spin.setValue(2)
        form.addRow("Y 轴号", self.axis_y_spin)
        self.pico_widgets += [self.axis_x_spin, self.axis_y_spin]
        self.spm_spin = QtWidgets.QDoubleSpinBox()
        self.spm_spin.setRange(1.0, 200000.0)
        self.spm_spin.setDecimals(1)
        self.spm_spin.setValue(1000.0)
        self.spm_spin.setToolTip("每毫米步数：须按实际位移台标定\n"
                                 "(MTM 平台 ~30nm/步 ≈ 33333 steps/mm)")
        form.addRow("steps/mm", self.spm_spin)
        self.pico_widgets.append(self.spm_spin)
        self.speed_spin = QtWidgets.QSpinBox()
        self.speed_spin.setRange(0, 5000)
        self.speed_spin.setValue(0)
        self.speed_spin.setToolTip("速度 steps/s；0=不修改控制器当前速度")
        form.addRow("速度(steps/s)", self.speed_spin)
        self.pico_widgets.append(self.speed_spin)
        # -- 串口 G 代码参数
        self.port_edit = QtWidgets.QLineEdit("COM3")
        form.addRow("串口", self.port_edit)
        self.baud_spin = QtWidgets.QSpinBox()
        self.baud_spin.setRange(9600, 256000)
        self.baud_spin.setValue(115200)
        form.addRow("波特率", self.baud_spin)
        self.serial_widgets = [self.port_edit, self.baud_spin]
        # -- 公共参数
        self.cam_spin = QtWidgets.QSpinBox()
        self.cam_spin.setRange(0, 8)
        form.addRow("相机索引", self.cam_spin)
        self.shift_sign_combo = QtWidgets.QComboBox()
        self.shift_sign_combo.addItems(["+1（需现场标定）", "-1（需现场标定）"])
        self.shift_sign_combo.setToolTip(
            "图像位移符号：下发已知小步，观测目标球在画面中的移动方向；\n"
            "实际移动方向与预期相反则改为 -1（防失位检测依赖该符号）")
        form.addRow("图像位移符号", self.shift_sign_combo)
        self.confirm_chk = QtWidgets.QCheckBox(
            "我确认已连接真实电机，并已检查限位与急停")
        self.confirm_chk.setStyleSheet("color:#c00;")
        form.addRow(self.confirm_chk)
        self.motor_box.setVisible(False)
        panel.addWidget(self.motor_box)

        self.pause_btn = QtWidgets.QPushButton("暂停")
        self.pause_btn.clicked.connect(self.on_pause)
        panel.addWidget(self.pause_btn)

        self.estop_btn = QtWidgets.QPushButton("急停")
        self.estop_btn.setStyleSheet("background:#c0392b;color:white;")
        self.estop_btn.clicked.connect(self.on_estop)
        panel.addWidget(self.estop_btn)

        self.xyz_box = QtWidgets.QGroupBox("XYZ stage (um)")
        xyz = QtWidgets.QVBoxLayout(self.xyz_box)
        xyz.setContentsMargins(10, 8, 10, 8)
        xyz.setSpacing(5)
        self.xyz_axis_combo = QtWidgets.QComboBox()
        self.xyz_axis_combo.addItems(["x", "y", "z"])
        self.xyz_axis_combo.setFixedWidth(70)
        self.xyz_step_spin = QtWidgets.QDoubleSpinBox()
        self.xyz_step_spin.setRange(0.001, 1000.0)
        self.xyz_step_spin.setDecimals(3)
        self.xyz_step_spin.setValue(10.0)
        self.xyz_steps_spin = QtWidgets.QDoubleSpinBox()
        self.xyz_steps_spin.setRange(0.001, 1000000.0)
        self.xyz_steps_spin.setValue(1.0)
        self.xyz_speed_spin = QtWidgets.QDoubleSpinBox()
        self.xyz_speed_spin.setRange(0.001, 1000000.0)
        self.xyz_speed_spin.setValue(100.0)
        self.xyz_accel_spin = QtWidgets.QDoubleSpinBox()
        self.xyz_accel_spin.setRange(0.001, 1000000.0)
        self.xyz_accel_spin.setValue(200.0)
        self.xyz_apply_btn = QtWidgets.QPushButton("apply profile")
        self.xyz_apply_btn.clicked.connect(self._apply_xyz_profile)
        self.xyz_jog_minus = QtWidgets.QPushButton("- jog")
        self.xyz_jog_plus = QtWidgets.QPushButton("+ jog")
        self.xyz_jog_minus.clicked.connect(lambda: self._jog_xyz(-1.0))
        self.xyz_jog_plus.clicked.connect(lambda: self._jog_xyz(1.0))
        self.xyz_home_btn = QtWidgets.QPushButton("home")
        self.xyz_home_btn.clicked.connect(self._home_xyz)
        self.xyz_zero_btn = QtWidgets.QPushButton("zero")
        self.xyz_zero_btn.clicked.connect(self._zero_xyz)
        self.xyz_enable_btn = QtWidgets.QPushButton("enable")
        self.xyz_enable_btn.clicked.connect(self._enable_xyz)
        self.xyz_stop_btn = QtWidgets.QPushButton("stop")
        self.xyz_stop_btn.setStyleSheet("background:#c0392b;color:white;")
        self.xyz_stop_btn.clicked.connect(self._stop_xyz)
        # Keep the numeric editors compact; the panel is also used beside the
        # live microscope canvas and should not force a wide horizontal scroll.
        for spin in (self.xyz_step_spin, self.xyz_steps_spin,
                     self.xyz_speed_spin, self.xyz_accel_spin):
            spin.setFixedWidth(92)
        for button in (self.xyz_jog_minus, self.xyz_jog_plus):
            button.setFixedWidth(88)
        for button in (self.xyz_home_btn, self.xyz_zero_btn,
                       self.xyz_enable_btn, self.xyz_stop_btn):
            button.setFixedWidth(78)
        self.xyz_apply_btn.setFixedWidth(108)

        def _row(*widgets):
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)
            for widget in widgets:
                row.addWidget(widget)
            return row

        axis_label = QtWidgets.QLabel("Axis")
        axis_label.setMinimumWidth(34)
        step_label = QtWidgets.QLabel("Step (um)")
        step_label.setMinimumWidth(58)
        xyz.addLayout(_row(axis_label, self.xyz_axis_combo,
                           step_label, self.xyz_step_spin,
                           self.xyz_jog_minus, self.xyz_jog_plus))

        steps_label = QtWidgets.QLabel("Steps/unit")
        steps_label.setMinimumWidth(67)
        speed_label = QtWidgets.QLabel("Max speed")
        speed_label.setMinimumWidth(61)
        accel_label = QtWidgets.QLabel("Accel")
        accel_label.setMinimumWidth(38)
        xyz.addLayout(_row(steps_label, self.xyz_steps_spin,
                           speed_label, self.xyz_speed_spin,
                           accel_label, self.xyz_accel_spin,
                           self.xyz_apply_btn))

        xyz.addLayout(_row(self.xyz_home_btn, self.xyz_zero_btn,
                           self.xyz_enable_btn, self.xyz_stop_btn))
        self.xyz_position_label = QtWidgets.QLabel("x=--  y=--  z=--")
        self.xyz_position_label.setObjectName("xyzPositionLabel")
        self.xyz_position_label.setMinimumHeight(22)
        xyz.addWidget(self.xyz_position_label)
        self.xyz_telemetry_out = QtWidgets.QPlainTextEdit()
        self.xyz_telemetry_out.setReadOnly(True)
        self.xyz_telemetry_out.setMaximumBlockCount(80)
        self.xyz_telemetry_out.setFixedHeight(88)
        self.xyz_telemetry_out.setPlaceholderText("Telemetry will appear here")
        xyz.addWidget(self.xyz_telemetry_out)
        panel.addWidget(self.xyz_box)

        self.state_label = QtWidgets.QLabel("IDLE")
        panel.addWidget(self.state_label)
        self.detail_label = QtWidgets.QLabel("")
        self.detail_label.setWordWrap(True)
        panel.addWidget(self.detail_label)

        # ---- 实时检测 + ROI/区域划分（固定镜头）
        panel.addWidget(QtWidgets.QLabel("—— 实时检测 / 区域划分 ——"))
        self.live_btn = QtWidgets.QPushButton("开始实时检测")
        self.live_btn.setCheckable(True)
        self.live_btn.toggled.connect(self.on_live_toggled)
        panel.addWidget(self.live_btn)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("画框模式"))
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["ROI 视野", "目标区", "障碍区", "自由区",
                                  "圆球(mask)", "衬底(ground)",
                                  "障碍物(obstacle)",
                                  "目标点(避障)", "目标范围(组装)"])
        row.addWidget(self.mode_combo, 1)
        panel.addLayout(row)

        row2 = QtWidgets.QHBoxLayout()
        self.reset_view_btn = QtWidgets.QPushButton("复位视野")
        self.reset_view_btn.clicked.connect(self.on_reset_view)
        row2.addWidget(self.reset_view_btn)
        self.undo_zone_btn = QtWidgets.QPushButton("撤销区域")
        self.undo_zone_btn.clicked.connect(self.on_undo_zone)
        row2.addWidget(self.undo_zone_btn)
        panel.addLayout(row2)

        row4 = QtWidgets.QHBoxLayout()
        row4.addWidget(QtWidgets.QLabel("边界间隙(px)"))
        self.edge_spin = QtWidgets.QSpinBox()
        self.edge_spin.setRange(1, 200)
        self.edge_spin.setValue(39)
        self.edge_spin.setToolTip(
            "球心距视野边界的最小允许间隙(实际下限4px)；障碍碰撞不受影响。"
            "调小可让贴边球作为起点，红带随之变窄。")
        self.edge_spin.valueChanged.connect(self.on_edge_changed)
        row4.addWidget(self.edge_spin)
        panel.addLayout(row4)

        row3 = QtWidgets.QHBoxLayout()
        self.save_cfg_btn = QtWidgets.QPushButton("保存配置")
        self.save_cfg_btn.clicked.connect(self.on_save_config)
        row3.addWidget(self.save_cfg_btn)
        self.load_cfg_btn = QtWidgets.QPushButton("载入配置")
        self.load_cfg_btn.clicked.connect(self.on_load_config)
        row3.addWidget(self.load_cfg_btn)
        panel.addLayout(row3)
        self.zones_label = QtWidgets.QLabel("zones: 0")
        panel.addWidget(self.zones_label)

        # ---- 对象属性实时面板（需求3：位置/顶点/尺寸实时更新）
        panel.addWidget(QtWidgets.QLabel("对象属性 (实时)"))
        self.props_out = QtWidgets.QPlainTextEdit()
        self.props_out.setReadOnly(True)
        self.props_out.setMaximumBlockCount(400)
        self.props_out.setFont(QtGui.QFont("Consolas", 8))
        self.props_out.setFixedHeight(120)
        panel.addWidget(self.props_out)

        # ---- 操作日志框（需求5：移动/新建等全部入框）
        panel.addWidget(QtWidgets.QLabel("操作日志"))
        self.log_out = QtWidgets.QPlainTextEdit()
        self.log_out.setReadOnly(True)
        self.log_out.setMaximumBlockCount(500)
        self.log_out.setFont(QtGui.QFont("Consolas", 8))
        self.log_out.setFixedHeight(120)
        panel.addWidget(self.log_out)

        panel.addWidget(QtWidgets.QLabel("报告回放 (JSONL)"))
        self.report_path = QtWidgets.QLineEdit()
        panel.addWidget(self.report_path)
        self.replay_btn = QtWidgets.QPushButton("回放")
        self.replay_btn.clicked.connect(self.on_replay)
        panel.addWidget(self.replay_btn)
        self.replay_out = QtWidgets.QPlainTextEdit()
        self.replay_out.setReadOnly(True)
        panel.addWidget(self.replay_out, 1)

        # 仿真镜头图层配置入口（mask/ground/obstacle 数量/标签/形状）
        self.sim_spec_btn = QtWidgets.QPushButton("仿真镜头图层...")
        self.sim_spec_btn.clicked.connect(self.on_edit_sim_spec)
        panel.addWidget(self.sim_spec_btn)

        # sim01 运行状态（ROI 布局 / 预览帧 / 实时检测世界）
        # grounds: 衬底对象列表，每个 ground 内嵌 goal(避障) / goal_range(组装)
        self._sim_cfg = {
            "grounds": [],                # [{id, rect:[x,y,w,h], goal, goal_range}]
            "balls": [],                  # [[x,y,w,h], ...]
            "obstacles": []               # [[x,y,w,h], ...]
        }
        self._sim_ids = {"balls": [], "obstacles": []}   # 平行 ID 列表（需求4）
        self._sim_uid = 0                # 全局单调计数器，ID 永不复用
        self._roi_cfg = None             # 提前初始化（_load_sim_config 会刷新属性面板）
        self._live_dets = []             # 提前初始化（同上）
        self._sim_order: list = []       # 画框顺序栈（undo 后画先撤）
        self._sim_preview_frame = None   # sim01 预览原始帧（叠加 ROI 用）
        self._sim_live = None            # sim01 实时检测世界（SimMicroscopeWorld）
        self._load_sim_config()
        self._simlog("UI 就绪")

        # 场景切换 -> 预览初始帧
        self.scenario_combo.currentIndexChanged.connect(self.on_scenario_changed)
        self.on_scenario_changed()

        # 实时检测状态（推理全部在 LiveDetectThread，UI 线程只渲染/标注）
        self._live_timer = QtCore.QTimer(self)
        self._live_timer.setInterval(70)   # ≈14fps（需求6：低延迟实时展示）
        self._live_timer.timeout.connect(self._live_tick)
        self._live_world = None
        self._live_detect: Optional[LiveDetectThread] = None
        self._live_dets = []
        self._tick_n = 0
        self._roi_cfg = None       # RoiConfig（zones 用视频绝对坐标）
        self._view_map = None      # (scale, ox, oy) 帧->画布映射
        self._preview_worker: Optional[PreviewThread] = None
        self._pending_scenario: Optional[str] = None
        self._sim_drag_acc = (0, 0)  # 位移台拖拽累计增量 px
        self._xyz_stage = None

        # 启动即后台预热 YOLO 权重（3-5s），期间 UI 完全可交互
        warmup_shared_detector_async(
            video_sim.WEIGHTS,
            on_error=lambda exc: self.detail_label.setText(
                f"YOLO 权重预热失败: {exc}"))
        self.usb_count_ready.connect(
            lambda n: self.pico_count_label.setText(f"USB 设备: {n}"))

    def closeEvent(self, ev) -> None:  # noqa: N802 - 退出时回收线程
        if self._live_timer.isActive():
            self._live_timer.stop()
        if self._live_detect is not None:
            self._live_detect.stop()
            self._live_detect.wait(2000)
        if self._preview_worker is not None:
            self._preview_worker.wait(2000)
        if self._sim_live is not None:   # sim01 仿真镜头资源
            self._sim_live.close()
            self._sim_live = None
        super().closeEvent(ev)

    # ---------------- 实时检测 / 区域划分
    def _get_xyz_stage(self):
        if self.scenario_combo.currentText() != "sim01":
            raise MotionStateError("XYZ jog is available in virtual sim01 mode")
        world = self._ensure_live()
        self._xyz_stage = world.motion_stage
        return self._xyz_stage

    def _update_xyz_view(self, stage) -> None:
        pos = stage.position
        self.xyz_position_label.setText(
            "x={x:.2f}  y={y:.2f}  z={z:.2f} um [{state}]".format(
                x=pos["x"], y=pos["y"], z=pos["z"],
                state="enabled" if stage.enabled else "stopped"))

    def _jog_xyz(self, sign: float) -> None:
        try:
            stage = self._get_xyz_stage()
            axis = self.xyz_axis_combo.currentText()
            telemetry = stage.move_by({axis: sign * self.xyz_step_spin.value()},
                                      source="ui-jog")
            self._update_xyz_view(stage)
            self.on_motion(telemetry.to_dict())
            self._refresh_sim_preview()
        except (MotionConfigError, MotionLimitError, MotionStateError) as exc:
            self.detail_label.setText(str(exc))

    def _apply_xyz_profile(self) -> None:
        try:
            stage = self._get_xyz_stage()
            axis = self.xyz_axis_combo.currentText()
            stage.configure_axis(axis,
                                 steps_per_unit=self.xyz_steps_spin.value(),
                                 max_speed=self.xyz_speed_spin.value(),
                                 acceleration=self.xyz_accel_spin.value())
            self.detail_label.setText(f"{axis} motion profile applied")
        except (MotionConfigError, MotionLimitError, MotionStateError) as exc:
            self.detail_label.setText(str(exc))

    def _home_xyz(self) -> None:
        try:
            stage = self._get_xyz_stage()
            telemetry = stage.home(source="ui-home")
            self._update_xyz_view(stage)
            self.on_motion(telemetry.to_dict())
            self._refresh_sim_preview()
        except (MotionConfigError, MotionLimitError, MotionStateError) as exc:
            self.detail_label.setText(str(exc))

    def _zero_xyz(self) -> None:
        try:
            stage = self._get_xyz_stage()
            stage.zero()
            self._update_xyz_view(stage)
            self._refresh_sim_preview()
        except (MotionConfigError, MotionLimitError, MotionStateError) as exc:
            self.detail_label.setText(str(exc))

    def _enable_xyz(self) -> None:
        try:
            stage = self._get_xyz_stage()
            stage.enable()
            self._update_xyz_view(stage)
        except MotionStateError as exc:
            self.detail_label.setText(str(exc))

    def _stop_xyz(self) -> None:
        try:
            stage = self._get_xyz_stage()
            stage.stop()
            self._update_xyz_view(stage)
            self.detail_label.setText("XYZ stage stopped; press enable to resume")
        except MotionStateError as exc:
            self.detail_label.setText(str(exc))

    @Slot(dict)
    def on_motion(self, telemetry: dict) -> None:
        pos = telemetry.get("position_after", {})
        if len(pos) == 3:
            self.xyz_position_label.setText(
                "x={x:.2f}  y={y:.2f}  z={z:.2f} um".format(**pos))
        steps = telemetry.get("steps", {})
        speed = telemetry.get("peak_speed", {})
        self.xyz_telemetry_out.appendPlainText(
            "seq={seq} source={source} steps={steps} peak={speed} "
            "duration={duration_s:.4f}s limits={limit_hit}".format(
                seq=telemetry.get("sequence", "?"),
                source=telemetry.get("source", "?"), steps=steps, speed=speed,
                duration_s=float(telemetry.get("duration_s", 0.0)),
                limit_hit=telemetry.get("limit_hit", {})))

    def _map_to_frame(self, qx: int, qy: int):
        if self._view_map is None:
            return None
        s, ox, oy = self._view_map
        return ((qx - ox) / s, (qy - oy) / s)

    # ---------------- sim01 实时模式位移台鼠标拖拽（需求2）
    def _is_draw_mode(self) -> bool:
        """画框模式为 sim01 对象绘制模式时返回 True（拖拽应为画框而非位移台）。"""
        mode = self.mode_combo.currentText()
        return mode in ("圆球(mask)", "衬底(ground)", "障碍物(obstacle)",
                        "目标点(避障)", "目标范围(组装)")

    def _on_sim_drag_start(self, fx: int, fy: int) -> None:
        """鼠标按下：若在 sim01 实时模式且非画框模式，开启跟踪以支持位移台拖拽。"""
        if (self.scenario_combo.currentText() == "sim01"
                and self._sim_live is not None
                and self.live_btn.isChecked()
                and not self._is_draw_mode()):
            self.canvas.setMouseTracking(True)
            self._sim_drag_acc = (0, 0)   # 累计拖拽增量 px

    def _on_sim_drag_move(self, dx: int, dy: int) -> None:
        """鼠标拖拽 -> 位移台移动（自然方向：拖拽方向 = 视野移动方向）。
        
        坐标映射：拖拽 dx_px → 球在画面中应移 -dx_px（即视野移 +dx_px），
        台位需移 +dx_px * pixel_size_um µm。
        """
        if self.scenario_combo.currentText() != "sim01":
            return
        world = self._sim_live
        if world is None:
            return
        px_um = world.pixel_size_um   # 0.5 µm/px
        cx = world.micro_stage.position["x"]
        cy = world.micro_stage.position["y"]
        try:
            world.motion_stage.move_by({"x": dx * px_um, "y": dy * px_um},
                                       source="ui-drag")
            self._update_xyz_view(world.motion_stage)
            if world.motion_stage.last_telemetry:
                self.on_motion(world.motion_stage.last_telemetry.to_dict())
        except (MotionConfigError, MotionLimitError, MotionStateError) as exc:
            self.detail_label.setText(str(exc))
        ax, ay = self._sim_drag_acc
        self._sim_drag_acc = (ax + dx, ay + dy)

    def _on_sim_drag_end(self) -> None:
        """鼠标释放时关闭跟踪并记录累计位移。"""
        if self.scenario_combo.currentText() == "sim01":
            self.canvas.setMouseTracking(False)
            ax, ay = self._sim_drag_acc
            if abs(ax) > 1 or abs(ay) > 1:
                px_um = (self._sim_live.pixel_size_um
                         if self._sim_live else 0.5)
                self._simlog(f"位移台拖拽完成: 累计 ({ax},{ay})px "
                             f"≈ ({ax * px_um:.0f},{ay * px_um:.0f})µm")

    def _ensure_live(self):
        if self.scenario_combo.currentText() == "sim01":
            # 仿真镜头实时流：SimMicroscopeWorld 连续渲染（复用，退出时统一关闭）
            if self._sim_live is None:
                self._sim_live = sim_microscope.SimMicroscopeWorld()
            self._live_world = self._sim_live   # 统一入口（tick 判空/停止复用）
            self._xyz_stage = self._sim_live.motion_stage
            self._update_xyz_view(self._xyz_stage)
            return self._sim_live
        vw, vh = video_sim._video_size()
        if self._live_world is None:
            self._live_world = video_sim.VideoWorld(
                video_path=video_sim.VIDEO, window=(vw, vh), offset=(0.0, 0.0))
        if self._roi_cfg is None:
            self._roi_cfg = RoiConfig(roi=(0, 0, vw, vh), video=video_sim.VIDEO)
        return self._live_world

    @Slot(bool)
    def on_live_toggled(self, on: bool) -> None:
        try:
            if on:
                self._ensure_live()
                self._live_timer.start()
                self.live_btn.setText("停止实时检测")
                self.state_label.setText("LIVE")
            else:
                self._live_timer.stop()
                if self._live_detect is not None:
                    self._live_detect.stop()
                    self._live_detect.wait(2000)
                    self._live_detect = None
                self._live_dets = []
                self.live_btn.setText("开始实时检测")
                if self.state_label.text() == "LIVE":
                    self.state_label.setText("IDLE")
        except Exception as exc:  # noqa: BLE001
            self.live_btn.setChecked(False)
            _log_exception("live detect start failed", exc)
            self.detail_label.setText(f"实时检测启动失败: {exc}")

    def _annotate_live(self, frame: np.ndarray) -> np.ndarray:
        """叠加 ROI、安全缓冲带、区域、YOLO 检测框（视频坐标->窗口坐标）。"""
        out = frame.copy()
        cfg = self._roi_cfg
        ox, oy = self._live_world.offset
        if cfg is not None:
            # 红带 = 实际禁停区：边界间隙 + 衬底内缩10 + 裕量4
            # 局部混合（仅 4 条带区域），避免全帧 copy+addWeighted 的开销
            band = max(2, int(cfg.edge_clearance_px) + 14)
            rx, ry, rw, rh = cfg.roi
            H, W = out.shape[:2]
            for (bx, by, bw, bh) in (
                    (rx - ox, ry - oy, rw, band),                     # 上
                    (rx - ox, ry - oy + rh - band, rw, band),         # 下
                    (rx - ox, ry - oy, band, rh),                     # 左
                    (rx - ox + rw - band, ry - oy, band, rh)):        # 右
                x0, y0 = max(0, int(bx)), max(0, int(by))
                x1 = min(W, int(bx + bw))
                y1 = min(H, int(by + bh))
                if x1 <= x0 or y1 <= y0:
                    continue
                tile = np.full((y1 - y0, x1 - x0, 3), (0, 0, 255),
                               dtype=np.uint8)
                out[y0:y1, x0:x1] = cv2.addWeighted(
                    out[y0:y1, x0:x1], 0.70, tile, 0.30, 0)
            # ROI 边框（黄）
            cv2.rectangle(out, (int(rx - ox), int(ry - oy)),
                          (int(rx - ox + rw), int(ry - oy + rh)),
                          (0, 220, 220), 2)
            colors = {"goal": (0, 180, 0), "obstacle": (0, 0, 180),
                      "free": (180, 120, 0)}
            for z in cfg.zones:
                x, y, w, h = (int(z.rect[0] - ox), int(z.rect[1] - oy),
                              int(z.rect[2]), int(z.rect[3]))
                c = colors[z.kind]
                if z.kind == "obstacle":
                    cv2.rectangle(out, (x, y), (x + w, y + h), c, -1)
                    # 障碍膨胀警戒环
                    cv2.rectangle(out, (x - band, y - band),
                                  (x + w + band, y + h + band), c, 1)
                cv2.rectangle(out, (x, y), (x + w, y + h), c, 2)
                cv2.putText(out, z.name, (x + 2, y + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
        for (cx, cy), r, conf in self._live_dets:
            cv2.rectangle(out, (int(cx - r), int(cy - r)),
                          (int(cx + r), int(cy + r)), (255, 255, 0), 2)
            cv2.putText(out, f"{conf:.2f}", (int(cx - r), int(cy - r) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        # 光斑-球锁定距离（goal 区中心即光斑位）
        if cfg is not None and cfg.goal_zone() is not None and self._live_dets:
            gx, gy = cfg.goal_zone().center()
            gx, gy = gx - ox, gy - oy
            (cx, cy) = self._live_dets[0][0]
            d = ((cx - gx) ** 2 + (cy - gy) ** 2) ** 0.5
            cv2.line(out, (int(cx), int(cy)), (int(gx), int(gy)),
                     (255, 0, 255), 1)
            cv2.putText(out, f"spot-ball {d:.0f}px",
                        (int((cx + gx) / 2) + 4, int((cy + gy) / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 0, 255), 1)
        return out

    @Slot()
    def _live_tick(self) -> None:
        if self._live_world is None:
            return
        frame = self._live_world.render()
        self._tick_n += 1
        if self.scenario_combo.currentText() == "sim01":
            # 仿真镜头：经典检测器（白盘球 + 衬底内缩），推理同走后台丢帧线程
            if self._live_detect is None:
                self._live_detect = LiveDetectThread(
                    self._sim_live.make_detector, parent=self)
                self._live_detect.dets_ready.connect(self._on_live_dets)
                self._live_detect.start()
            self._live_detect.submit(frame)
            out = self._overlay_sim_cfg(frame)
            for (cx, cy), r, conf in self._live_dets:
                cv2.rectangle(out, (int(cx - r), int(cy - r)),
                              (int(cx + r), int(cy + r)), (255, 255, 0), 2)
            self._show_frame(out)
            return
        det = try_shared_detector(video_sim.WEIGHTS)
        if det is not None:
            # 推理在后台线程（丢帧策略）；结果经 dets_ready 回流
            if self._live_detect is None:
                self._live_detect = LiveDetectThread(lambda: det, parent=self)
                self._live_detect.dets_ready.connect(self._on_live_dets)
                self._live_detect.start()
            self._live_detect.submit(frame)
        elif self._tick_n % 25 == 1:   # 权重后台加载中：画面角标提示
            cv2.putText(frame, "loading YOLO model...", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        self._show_frame(self._annotate_live(frame))

    # ---------------- sim01 ROI 布局（窗口坐标：球/衬底/障碍/目标）
    @staticmethod
    def _rect_inside(px: int, py: int, rect) -> bool:
        return rect[0] <= px <= rect[0] + rect[2] and rect[1] <= py <= rect[1] + rect[3]

    def _find_ground_for(self, px: int, py: int) -> int:
        """返回包含点 (px,py) 的第一个 ground 索引；无匹配返回 -1。"""
        for i, g in enumerate(self._sim_cfg["grounds"]):
            if self._rect_inside(px, py, g["rect"]):
                return i
        return -1

    def _find_ground_for_rect(self, rect) -> int:
        """Return the sole ground containing the complete rectangle."""
        owners = [i for i, g in enumerate(self._sim_cfg["grounds"])
                  if rect_contains_rect(g["rect"], rect)]
        return owners[0] if len(owners) == 1 else -1

    def _sim_rect(self, x: int, y: int, w: int, h: int) -> None:
        """sim01：按画框模式记录 ROI；目标点/范围归属到其所在 ground。"""
        mode = self.mode_combo.currentText()
        cx, cy = int(x + w / 2), int(y + h / 2)
        if mode == "衬底(ground)":
            if w <= 10 or h <= 10:
                self.detail_label.setText("衬底框太小（需 > 10px）")
                return
            gid = self._new_id("G")
            self._sim_cfg["grounds"].append({
                "id": gid,
                "rect": [int(x), int(y), int(w), int(h)],
                "goal": None, "goal_range": None})
            self._sim_order.append(("ground_add",))
            self._simlog(f"新建衬底 {gid}: pos=({x},{y}) size={w}x{h} "
                         f"vertices={self._vertices([x, y, w, h])}")
        elif mode == "圆球(mask)":
            if w <= 6 or h <= 6:
                self.detail_label.setText("圆球框太小")
                return
            if self._find_ground_for_rect((x, y, w, h)) < 0:
                self.detail_label.setText(
                    "⚠ 圆球必须在衬底范围内（先画衬底）")
                return
            bid = self._new_id("B")
            self._sim_cfg["balls"].append([int(x), int(y), int(w), int(h)])
            self._sim_ids["balls"].append(bid)
            self._sim_order.append(("ball_add",))
            self._simlog(f"新建圆球 {bid}: pos=({x},{y}) size={w}x{h} "
                         f"center=({cx},{cy})")
        elif mode == "障碍物(obstacle)":
            if w <= 6 or h <= 6:
                self.detail_label.setText("障碍框太小")
                return
            if self._find_ground_for_rect((x, y, w, h)) < 0:
                self.detail_label.setText(
                    "⚠ 障碍物必须在衬底范围内（先画衬底）")
                return
            oid_ = self._new_id("O")
            self._sim_cfg["obstacles"].append([int(x), int(y), int(w), int(h)])
            self._sim_ids["obstacles"].append(oid_)
            self._sim_order.append(("obstacle_add",))
            self._simlog(f"新建障碍物 {oid_}: pos=({x},{y}) size={w}x{h} "
                         f"center=({cx},{cy})")
        elif mode == "目标点(避障)":
            gi = self._find_ground_for(cx, cy)
            if gi < 0:
                self.detail_label.setText(
                    "⚠ 目标点必须在某个衬底范围内")
                return
            if self._sim_cfg["grounds"][gi].get("goal"):
                self.detail_label.setText(
                    f"⚠ 一个衬底有且仅有一个目标点（{self._sim_cfg['grounds'][gi]['id']} 已有目标点）")
                return
            self._sim_cfg["grounds"][gi]["goal"] = [cx, cy]
            self._sim_order.append(("ground_goal", gi))
            self._simlog(f"设定目标点 -> {self._sim_cfg['grounds'][gi]['id']}"
                         f": ({cx},{cy})")
        elif mode == "目标范围(组装)":
            gi = self._find_ground_for_rect((x, y, w, h))
            if gi < 0:
                self.detail_label.setText(
                    "⚠ 目标范围必须在某个衬底范围内")
                return
            if self._sim_cfg["grounds"][gi].get("goal_range"):
                self.detail_label.setText(
                    f"⚠ 一个衬底有且仅有一个目标范围（{self._sim_cfg['grounds'][gi]['id']} 已有）")
                return
            self._sim_cfg["grounds"][gi]["goal_range"] = [
                int(x), int(y), int(w), int(h)]
            self._sim_order.append(("ground_range", gi))
            self._simlog(f"设定目标范围 -> {self._sim_cfg['grounds'][gi]['id']}"
                         f": pos=({x},{y}) size={w}x{h}")
        else:
            self.detail_label.setText(
                "sim01 请选择：圆球/衬底/障碍物/目标点(避障)/目标范围(组装)")
            return
        self._save_sim_config()
        self._update_sim_zones_label()
        self._refresh_sim_preview()

    # ---------------- 日志 / ID / 属性面板（需求 3/4/5）
    def _simlog(self, msg: str) -> None:
        """操作日志：时间戳 + 消息，追加到日志框（自动裁剪到 500 行）。"""
        ts = QtCore.QTime.currentTime().toString("HH:mm:ss.zzz")
        self.log_out.appendPlainText(f"[{ts}] {msg}")

    def _new_id(self, prefix: str) -> str:
        """唯一 ID：单调递增，永不复用（需求4 唯一性）。"""
        self._sim_uid += 1
        return f"{prefix}{self._sim_uid}"

    @staticmethod
    def _vertices(rect) -> list:
        x, y, w, h = rect
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

    def _fmt_obj(self, oid: str, kind: str, rect, extra: str = "") -> str:
        x, y, w, h = rect
        vs = " ".join(f"({px},{py})" for px, py in self._vertices(rect))
        return (f"{oid:<6}{kind:<8} pos=({x:>4},{y:>4}) size={w}x{h} "
                f"center=({x + w // 2},{y + h // 2}) v[{vs}]{extra}")

    def _update_props_panel(self) -> None:
        """需求3：所有对象的绝对位置/顶点/尺寸实时更新到属性面板。"""
        lines = []
        c = self._sim_cfg
        for i, g in enumerate(c["grounds"]):
            extra = ""
            if g.get("goal"):
                gx, gy = g["goal"]
                extra += f" goal=({gx},{gy})"
            if g.get("goal_range"):
                gr = g["goal_range"]
                extra += (f" range=({gr[0]},{gr[1]}) {gr[2]}x{gr[3]} "
                          f"center=({gr[0] + gr[2] // 2},{gr[1] + gr[3] // 2})")
            lines.append(self._fmt_obj(g.get("id", f"G{i}?"),
                                       "ground", g["rect"], extra))
            if g.get("goal"):
                lines.append(f"       goal-pt  pos=({g['goal'][0]},{g['goal'][1]})")
            if g.get("goal_range"):
                gr = g["goal_range"]
                lines.append(self._fmt_obj(g.get("id", "G?") + "-R",
                                           "goalRange", gr))
        for oid, b in zip(self._sim_ids["balls"], c["balls"]):
            lines.append(self._fmt_obj(oid, "ball", b))
        for oid, o in zip(self._sim_ids["obstacles"], c["obstacles"]):
            lines.append(self._fmt_obj(oid, "obstacle", o))
        if self._roi_cfg is not None:
            lines.append(self._fmt_obj("ROI", "view",
                                       list(self._roi_cfg.roi)))
            for z in self._roi_cfg.zones:
                lines.append(self._fmt_obj(z.name, z.kind, list(z.rect)))
        if self._live_dets:
            for i, det in enumerate(self._live_dets[:8]):
                try:
                    (cx, cy), r, conf = det
                except Exception:  # noqa: BLE001 - 兼容其他元组结构
                    continue
                lines.append(f"LIVE{i:<4}{'ball':<8} pos=({cx:.0f},{cy:.0f}) "
                             f"r={r:.0f} conf={conf:.2f}")
        self.props_out.setPlainText("\n".join(lines) or "(无对象)")

    def _update_sim_zones_label(self) -> None:
        c = self._sim_cfg
        g_goals = sum(1 for g in c["grounds"] if g.get("goal"))
        g_ranges = sum(1 for g in c["grounds"] if g.get("goal_range"))
        self.zones_label.setText(
            f"sim ROI: 球{len(c['balls'])} 衬底{len(c['grounds'])} "
            f"障碍{len(c['obstacles'])} "
            f"目标点{g_goals} 目标范围{g_ranges}")
        self._update_props_panel()   # 需求3：任何变更即时刷新属性面板

    def _overlay_sim_cfg(self, frame: np.ndarray) -> np.ndarray:
        """预览帧上叠加 sim01 ROI：衬底=蓝、球=青、障碍=红、目标点=黄十字、目标范围=绿圆。"""
        out = frame.copy()
        c = self._sim_cfg
        for i, g in enumerate(c["grounds"]):
            rx, ry, rw, rh = g["rect"]
            cv2.rectangle(out, (rx, ry), (rx + rw, ry + rh), (255, 120, 0), 1)
            cv2.putText(out, f"G{i}", (rx + 2, ry - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 120, 0), 1)
            if g.get("goal"):
                gx, gy = g["goal"]
                cv2.drawMarker(out, (gx, gy), (0, 255, 255),
                               cv2.MARKER_CROSS, 16, 2)
            if g.get("goal_range"):
                gx, gy, grw, grh = g["goal_range"]
                cv2.rectangle(out, (gx, gy), (gx + grw, gy + grh),
                              (0, 220, 0), 2)
                cx, cy = int(gx + grw / 2), int(gy + grh / 2)
                cv2.circle(out, (cx, cy), 4, (0, 220, 0), -1)
        for x, y, w, h in c["obstacles"]:
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 1)
        for x, y, w, h in c["balls"]:
            cv2.rectangle(out, (x, y), (x + w, y + h), (255, 255, 0), 1)
        return out

    def _refresh_sim_preview(self) -> None:
        if self._sim_live is not None:
            try:
                self._show_frame(self._overlay_sim_cfg(self._sim_live.render()))
                self._update_xyz_view(self._sim_live.motion_stage)
                return
            except Exception as exc:  # pragma: no cover - display fallback
                _log_exception("sim preview refresh failed", exc)
        if self._sim_preview_frame is not None:
            self._show_frame(self._overlay_sim_cfg(self._sim_preview_frame))

    SIM_ROI_CONFIG = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "artifacts", "sim_roi_config.json")
    SIM_SAMPLE_SPEC = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "artifacts", "sim_sample_spec.json")

    def _save_sim_config(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.SIM_ROI_CONFIG), exist_ok=True)
            with open(self.SIM_ROI_CONFIG, "w", encoding="utf-8") as f:
                json.dump(self._sim_cfg, f, ensure_ascii=False, indent=2)
        except Exception as exc:  # noqa: BLE001
            self.detail_label.setText(f"sim ROI 保存失败: {exc}")

    def _load_sim_config(self) -> None:
        try:
            with open(self.SIM_ROI_CONFIG, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if isinstance(cfg, dict):
                # 兼容旧格式：grounds=[[x,y,w,h],...] + 顶层 goal
                old_grounds = cfg.get("grounds")
                if old_grounds and old_grounds and not isinstance(old_grounds[0], dict):
                    cfg["grounds"] = [{"rect": list(g), "goal": None,
                                       "goal_range": None}
                                      for g in old_grounds]
                    if cfg.get("goal") is not None and cfg["grounds"]:
                        cfg["grounds"][0]["goal"] = cfg["goal"]
                self._sim_cfg = {**self._sim_cfg, **cfg}
        except Exception:  # noqa: BLE001 - 无配置/损坏时用默认空布局
            return
        # 需求4：载入的对象必须保有唯一 ID（缺失则补配，不复用当前计数器之前的值）
        for g in self._sim_cfg.get("grounds", []):
            if not g.get("id"):
                g["id"] = self._new_id("G")
        for key, prefix in (("balls", "B"), ("obstacles", "O")):
            need = len(self._sim_cfg.get(key, [])) - \
                len(self._sim_ids.get(key, []))
            for _ in range(max(0, need)):
                self._sim_ids[key].append(self._new_id(prefix))
        self._update_sim_zones_label()
        n = (len(self._sim_cfg.get("grounds", []))
             + len(self._sim_cfg.get("balls", []))
             + len(self._sim_cfg.get("obstacles", [])))
        if n:
            self._simlog(f"载入 sim ROI 配置: {n} 个对象 "
                         f"(衬底{len(self._sim_cfg.get('grounds', []))} "
                         f"球{len(self._sim_cfg.get('balls', []))} "
                         f"障碍{len(self._sim_cfg.get('obstacles', []))})")

    def _load_sample_spec(self) -> Optional[dict]:
        """读取仿真镜头图层配置（不存在/损坏返回 None -> 用相机当前配置）。"""
        try:
            with open(self.SIM_SAMPLE_SPEC, "r", encoding="utf-8") as f:
                spec = json.load(f)
            return spec if isinstance(spec, dict) else None
        except Exception:  # noqa: BLE001
            return None

    @Slot()
    def on_edit_sim_spec(self) -> None:
        """仿真镜头图层配置对话框：掩码/衬底/障碍物的数量/标签/形状。"""
        spec = self._load_sample_spec() or {}
        dlg = SimSpecDialog(spec, self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        new_spec = dlg.spec()
        try:
            os.makedirs(os.path.dirname(self.SIM_SAMPLE_SPEC), exist_ok=True)
            with open(self.SIM_SAMPLE_SPEC, "w", encoding="utf-8") as f:
                json.dump(new_spec, f, ensure_ascii=False, indent=2)
        except Exception as exc:  # noqa: BLE001
            self.detail_label.setText(f"图层配置保存失败: {exc}")
            return
        self.detail_label.setText("图层配置已保存，重新运行 sim01 后生效")

    @Slot(list)
    def _on_live_dets(self, dets: list) -> None:
        self._live_dets = dets
        self._update_props_panel()   # 需求3：实时检测位置进入属性面板

    @Slot(int, int, int, int)
    def on_rect_drawn(self, x: int, y: int, w: int, h: int) -> None:
        """画布拖拽结束：按模式应用 ROI 或区域（窗口坐标 -> 视频绝对坐标）。"""
        if self.scenario_combo.currentText() == "sim01":
            self._sim_rect(x, y, w, h)
            return
        if self._live_world is None or self._roi_cfg is None:
            return
        ox, oy = self._live_world.offset
        vx, vy = int(x + ox), int(y + oy)          # 视频绝对坐标
        mode = self.mode_combo.currentText()
        try:
            if mode == "ROI 视野":
                self._apply_roi((vx, vy, w, h))
            else:
                kind = {"目标区": "goal", "障碍区": "obstacle",
                        "自由区": "free"}[mode]
                # 裁剪进 ROI
                rx, ry, rw, rh = self._roi_cfg.roi
                x2, y2 = (min(vx + w, rx + rw), min(vy + h, ry + rh))
                vx, vy = max(vx, rx), max(vy, ry)
                w, h = x2 - vx, y2 - vy
                if w <= 8 or h <= 8:
                    return
                n = sum(1 for z in self._roi_cfg.zones if z.kind == kind) + 1
                name = {"goal": f"goal_{n}", "obstacle": f"obs_{n}",
                        "free": f"free_{n}"}[kind]
                self._roi_cfg.zones.append(Zone(name=name, kind=kind,
                                                rect=(vx, vy, w, h)))
                self._roi_cfg.validate()
        except ValueError as exc:
            self.detail_label.setText(f"区域非法: {exc}")
            return
        self.zones_label.setText(f"zones: {len(self._roi_cfg.zones)}")

    def _apply_roi(self, rect) -> None:
        if self.scenario_combo.currentText() == "sim01":
            self.detail_label.setText("sim01 为固定仿真镜头，不支持重设 ROI 视野")
            return
        old = self._roi_cfg
        self._roi_cfg = RoiConfig(roi=tuple(rect), video=video_sim.VIDEO,
                                  px_per_mm=old.px_per_mm if old else 100.0,
                                  edge_clearance_px=(old.edge_clearance_px
                                                     if old else 39.0),
                                  zones=old.zones if old else [])
        try:
            self._roi_cfg.validate(video_sim._video_size())
        except ValueError as exc:
            self._roi_cfg = old
            self.detail_label.setText(f"ROI 非法: {exc}")
            return
        self._live_world.set_window((int(rect[2]), int(rect[3])),
                                    (float(rect[0]), float(rect[1])))

    @Slot(int)
    def on_edge_changed(self, v: int) -> None:
        if self._roi_cfg is not None:
            self._roi_cfg.edge_clearance_px = float(v)

    @Slot()
    def on_reset_view(self) -> None:
        if self._live_world is None:
            return
        vw, vh = video_sim._video_size()
        self._live_world.window = (vw, vh)
        self._live_world.offset = [0.0, 0.0]

    @Slot()
    def on_undo_zone(self) -> None:
        if self.scenario_combo.currentText() == "sim01":
            c = self._sim_cfg
            if self._sim_order:          # 后画先撤
                entry = self._sim_order.pop()
                kind = entry[0]
                if kind == "ground_add" and c["grounds"]:
                    g = c["grounds"].pop()
                    self._sim_ids.setdefault("grounds", [])
                    if self._sim_ids.get("grounds"):
                        self._sim_ids["grounds"].pop()
                    self._simlog(f"撤销衬底 {g.get('id','?')}: "
                                 f"pos={tuple(g['rect'][:2])} "
                                 f"size={g['rect'][2]}x{g['rect'][3]}")
                elif kind == "ball_add" and c["balls"]:
                    b = c["balls"].pop()
                    bid = (self._sim_ids["balls"].pop()
                           if self._sim_ids["balls"] else "?")
                    self._simlog(f"撤销圆球 {bid}: pos={tuple(b[:2])} "
                                 f"size={b[2]}x{b[3]}")
                elif kind == "obstacle_add" and c["obstacles"]:
                    o = c["obstacles"].pop()
                    oid_ = (self._sim_ids["obstacles"].pop()
                            if self._sim_ids["obstacles"] else "?")
                    self._simlog(f"撤销障碍物 {oid_}: pos={tuple(o[:2])} "
                                 f"size={o[2]}x{o[3]}")
                elif kind == "ground_goal":
                    gi = entry[1]
                    if 0 <= gi < len(c["grounds"]):
                        gp = c["grounds"][gi].get("goal")
                        c["grounds"][gi]["goal"] = None
                        self._simlog(f"撤销目标点 "
                                     f"{c['grounds'][gi].get('id','?')}"
                                     f": {tuple(gp) if gp else '?'}")
                elif kind == "ground_range":
                    gi = entry[1]
                    if 0 <= gi < len(c["grounds"]):
                        gr = c["grounds"][gi].get("goal_range")
                        c["grounds"][gi]["goal_range"] = None
                        self._simlog(f"撤销目标范围 "
                                     f"{c['grounds'][gi].get('id','?')}"
                                     f": {tuple(gr[:2]) if gr else '?'}")
            else:   # 载入配置无顺序信息 -> 固定优先级
                for gi in range(len(c["grounds"]) - 1, -1, -1):
                    if c["grounds"][gi].get("goal_range"):
                        c["grounds"][gi]["goal_range"] = None; break
                    if c["grounds"][gi].get("goal"):
                        c["grounds"][gi]["goal"] = None; break
                else:
                    if c["balls"]:
                        b = c["balls"].pop()
                        bid = (self._sim_ids["balls"].pop()
                               if self._sim_ids["balls"] else "?")
                        self._simlog(f"撤销圆球 {bid}: pos={tuple(b[:2])} "
                                     f"size={b[2]}x{b[3]}")
                    elif c["obstacles"]:
                        o = c["obstacles"].pop()
                        oid_ = (self._sim_ids["obstacles"].pop()
                                if self._sim_ids["obstacles"] else "?")
                        self._simlog(f"撤销障碍物 {oid_}: pos={tuple(o[:2])} "
                                     f"size={o[2]}x{o[3]}")
                    elif c["grounds"]:
                        g = c["grounds"].pop()
                        self._simlog(f"撤销衬底 {g.get('id','?')}: "
                                     f"pos={tuple(g['rect'][:2])} "
                                     f"size={g['rect'][2]}x{g['rect'][3]}")
            self._save_sim_config()
            self._update_sim_zones_label()
            self._refresh_sim_preview()
            return
        if self._roi_cfg and self._roi_cfg.zones:
            z = self._roi_cfg.zones.pop()
            self.zones_label.setText(f"zones: {len(self._roi_cfg.zones)}")

    @Slot()
    def on_save_config(self) -> None:
        if self._roi_cfg is None:
            return
        try:
            vw, vh = video_sim._video_size()
            self._roi_cfg.validate((vw, vh))
            p = self._roi_cfg.save(video_sim.DEFAULT_ROI_CONFIG)
            self.detail_label.setText(f"配置已保存: {p}")
        except ValueError as exc:
            self.detail_label.setText(f"保存失败: {exc}")

    @Slot()
    def on_load_config(self) -> None:
        try:
            cfg = RoiConfig.load(video_sim.DEFAULT_ROI_CONFIG)
            vw, vh = video_sim._video_size()
            cfg.validate((vw, vh))
        except Exception as exc:  # noqa: BLE001
            self.detail_label.setText(f"载入失败: {exc}")
            return
        self._roi_cfg = cfg
        self.zones_label.setText(f"zones: {len(cfg.zones)}")
        self.edge_spin.blockSignals(True)
        self.edge_spin.setValue(int(cfg.edge_clearance_px))
        self.edge_spin.blockSignals(False)
        if self._live_world is not None:
            self._apply_roi(tuple(cfg.roi))
        self.detail_label.setText("配置已载入")


    # ---------------- preview
    @Slot()
    def on_scenario_changed(self) -> None:
        """显示所选场景初始帧；权重未就绪时 video 场景转后台构建。"""
        sc = self.scenario_combo.currentText()
        # video 场景需 YOLO 权重：未预热完 -> 后台线程构建（UI 不冻结）
        if sc.startswith("video") and try_shared_detector(video_sim.WEIGHTS) is None:
            if self._preview_worker is not None and self._preview_worker.isRunning():
                self._pending_scenario = sc
                return
            self._start_preview(sc)
            return
        try:
            world, _ = build_scenario(sc)
            self._show_frame(world.render())
        except Exception as exc:  # noqa: BLE001
            self.detail_label.setText(f"预览失败: {exc}")

    def _start_preview(self, sc: str) -> None:
        self.state_label.setText(f"场景构建中 {sc} ...")
        w = PreviewThread(sc, parent=self)
        w.preview_ready.connect(self._on_preview_ready)
        w.preview_failed.connect(self._on_preview_failed)
        w.finished.connect(self._on_preview_finished)
        self._preview_worker = w
        w.start()

    @Slot(str)
    def _on_preview_failed(self, msg: str) -> None:
        _log_exception("preview failed", RuntimeError(msg))
        self.detail_label.setText(f"预览失败: {msg}")
        QtWidgets.QMessageBox.warning(
            self, "场景预览失败",
            f"{msg}\n\n（完整 traceback 已写入 {LOG_PATH}）")

    @Slot(np.ndarray)
    def _on_preview_ready(self, frame: np.ndarray) -> None:
        if self.scenario_combo.currentText() == "sim01":
            self._sim_preview_frame = frame   # 缓存原始帧供 ROI 叠加
            frame = self._overlay_sim_cfg(frame)
        self._show_frame(frame)

    @Slot()
    def _on_preview_finished(self) -> None:
        if self.state_label.text().startswith("场景构建中"):
            self.state_label.setText("IDLE")
        pending = self._pending_scenario
        self._pending_scenario = None
        if pending and (self._preview_worker is None
                        or not self._preview_worker.isRunning()
                        or pending != self._preview_worker.scenario):
            self._start_preview(pending)

    def _show_frame(self, bgr: np.ndarray) -> None:
        pix = frame_to_pix(bgr)
        # 记录 帧->画布 映射（KeepAspectRatio），供鼠标坐标换算
        cw, ch = self.canvas.width(), self.canvas.height()
        fh, fw = bgr.shape[:2]
        s = min(cw / fw, ch / fh)
        self._view_map = (s, (cw - fw * s) / 2, (ch - fh * s) / 2)
        self.canvas.setPixmap(QtGui.QPixmap.fromImage(pix).scaled(
            self.canvas.size(), QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation))

    # ---------------- slots
    @Slot(int)
    def on_mode_changed(self, idx: int) -> None:
        motor = idx == 1
        self.motor_box.setVisible(motor)
        # Keep the two explicit run actions in sync with the selected mode.
        self.run_oa_btn.setText("避障运行 (电机)" if motor else "避障运行")
        self.run_ag_btn.setText("组装运行 (电机)" if motor else "组装运行")
        if motor and self.driver_combo.currentIndex() == 0:
            self._refresh_usb_count()

    @Slot(int)
    def on_driver_changed(self, idx: int) -> None:
        pico = idx == 0
        for w in self.pico_widgets:
            w.setVisible(pico)
        for w in self.serial_widgets:
            w.setVisible(not pico)
        if pico:
            self._refresh_usb_count()

    def _refresh_usb_count(self) -> None:
        """后台枚举 Picomotor USB 设备数（pylablib 扫描可能耗时）。"""
        self.pico_count_label.setText("USB 设备: 检测中...")

        def _probe():
            if __package__ in (None, ""):   # 直接运行 app.py 无包上下文
                from obstacle_avoidance.stages import PicoMotorStage
            else:
                from .stages import PicoMotorStage
            try:
                n = PicoMotorStage.usb_device_count()
            except Exception:  # noqa: BLE001 - pylablib 缺失/无设备
                n = -1
            self.usb_count_ready.emit(n)

        threading.Thread(target=_probe, daemon=True,
                         name="pico-usb-probe").start()

    def _motor_builder(self):
        """电机模式场景构建：当前 ROI 配置 + 真实相机 + 真实位移台。"""
        cfg, motor = self._motor_params()
        cfg.validate()   # 相机帧尺寸未知，仅校验区域相对几何
        return video_sim.build_video_scenario("video03", config=cfg,
                                              motor=motor)

    def _motor_params(self):
        """电机模式参数预检：配置完整性 + 硬件确认门控。"""
        cfg = self._roi_cfg
        if cfg is None or cfg.goal_zone() is None:
            raise RuntimeError("电机模式需要含目标区的 ROI 配置："
                               "先在实时检测里画 ROI/目标区并保存")
        if not self.confirm_chk.isChecked():
            raise RuntimeError("安全门控：请勾选'我确认已连接真实电机'后再运行")
        shift_sign = 1 if self.shift_sign_combo.currentIndex() == 0 else -1
        common = {"camera_index": self.cam_spin.value(),
                  "ball_shift_sign": shift_sign, "confirmed": True}
        if self.driver_combo.currentIndex() == 0:   # 8742/8743 Picomotor
            motor = {"driver": "picomotor",
                     "conn": self.conn_spin.value(),
                     "x_axis": self.axis_x_spin.value(),
                     "y_axis": self.axis_y_spin.value(),
                     "steps_per_mm": self.spm_spin.value(),
                     "speed_steps": (self.speed_spin.value() or None),
                     **common}
        else:                                        # 串口 G 代码
            motor = {"driver": "serial",
                     "port": self.port_edit.text().strip(),
                     "baudrate": self.baud_spin.value(),
                     **common}
        cfg_copy = RoiConfig(roi=cfg.roi, zones=list(cfg.zones),
                             video=cfg.video, px_per_mm=cfg.px_per_mm,
                             edge_clearance_px=cfg.edge_clearance_px)
        return cfg_copy, motor

    @Slot(str)
    def on_run(self, mode: str = "oa") -> None:
        """mode='oa' 避障运行；mode='ag' 组装运行。"""
        if self.worker and self.worker.isRunning():
            return
        if self.live_btn.isChecked():
            self.live_btn.setChecked(False)   # 停实时检测，独占推理资源
        scenario = self.scenario_combo.currentText()
        motor = self.mode_sel.currentIndex() == 1
        if motor:
            try:
                roi_cfg, motor_cfg = self._motor_params()
                self._controller_ref = {}
                builder = lambda: video_sim.build_video_scenario(
                    "video03", config=roi_cfg, motor=motor_cfg,
                    task_mode=mode, controller_sink=self._controller_ref)
            except Exception as exc:  # noqa: BLE001
                self.detail_label.setText(str(exc))
                return
            self.state_label.setText(f"RUNNING video03 [{mode}]")
            self.worker = WorkerThread("video03", self._controller_ref,
                                       builder=builder)
            self.worker.frame_ready.connect(self.on_frame)
            self.worker.state_ready.connect(self.on_state)
            self.worker.metrics_ready.connect(self.on_metrics)
            self.worker.motion_ready.connect(self.on_motion)
            self.worker.finished_run.connect(self.on_done)
            self.worker.error_occurred.connect(self.on_error_dialog)
            self.worker.log_ready.connect(self.on_log_line)
            self.worker.start()
            return
        c = self._sim_cfg

        # ---- 校验：每个 ground 对应一个目标
        if not c["grounds"]:
            self.detail_label.setText("⚠ 请先画衬底（至少一个）")
            return
        if mode == "oa":
            missing = [i for i, g in enumerate(c["grounds"]) if not g.get("goal")]
            if missing:
                self.detail_label.setText(
                    f"⚠ G{missing} 缺少目标点（避障模式：一个衬底有且仅有一个目标点）")
                return
            extra = [i for i, g in enumerate(c["grounds"])
                     if g.get("goal_range") and not g.get("goal")]
            if extra:
                self.detail_label.setText(
                    f"⚠ G{extra} 有目标范围但无目标点（避障模式需目标点）")
                return
        else:   # ag
            missing = [i for i, g in enumerate(c["grounds"])
                       if not g.get("goal_range")]
            if missing:
                self.detail_label.setText(
                    f"⚠ G{missing} 缺少目标范围（组装模式：一个衬底有且仅有一个目标范围）")
                return

        # ---- 构建 sim_layout：把 grounds 对象展开成 cli 期望的格式
        # balls / obstacles 保持扁平框列表
        # grounds 提取 rect 列表（sim_microscope.build_sim_scenario 只消费 rect）
        layout = {
            "balls": [list(b) for b in c["balls"]],
            "obstacles": [list(o) for o in c["obstacles"]],
            "grounds": [list(g["rect"]) for g in c["grounds"]],
            "ground_goals": [g.get("goal") for g in c["grounds"]],
            "ground_goal_ranges": [g.get("goal_range") for g in c["grounds"]],
            "run_mode": mode,
        }
        try:
            validate_sim_layout(layout, mode=mode)
        except LayoutValidationError as exc:
            self.detail_label.setText(str(exc))
            return

        if motor:
            self.detail_label.setText("电机模式需真实硬件连接，当前暂不可用")
            return

        self.state_label.setText(f"RUNNING sim01 [{mode}]")
        self._simlog(f"启动运行 sim01 [{mode}]: "
                     f"球{len(layout['balls'])} "
                     f"衬底{len(layout['grounds'])} "
                     f"障碍{len(layout['obstacles'])}")
        self.worker = WorkerThread(scenario, {}, builder=None,
                                   sample_spec=self._load_sample_spec())
        self.worker._sim_layout = layout       # 直接注入 WorkerThread
        self.worker._run_mode = mode
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.state_ready.connect(self.on_state)
        self.worker.metrics_ready.connect(self.on_metrics)
        self.worker.motion_ready.connect(self.on_motion)
        self.worker.finished_run.connect(self.on_done)
        self.worker.error_occurred.connect(self.on_error_dialog)
        self.worker.log_ready.connect(self.on_log_line)   # 需求5
        self.worker.start()

    @Slot(np.ndarray)
    def on_frame(self, bgr: np.ndarray) -> None:
        if self.scenario_combo.currentText() == "sim01":
            bgr = self._overlay_sim_cfg(bgr)
        self._show_frame(bgr)

    @Slot(str)
    def on_log_line(self, line: str) -> None:
        """运行期事件 -> 操作日志（需求5，跨线程信号）。"""
        self._simlog(line)

    @Slot(str)
    def on_error_dialog(self, msg: str) -> None:
        """运行/构建错误：后台日志已记录，这里弹窗告知用户。"""
        _log_exception("worker error", RuntimeError(msg))
        QtWidgets.QMessageBox.critical(
            self, "运行错误",
            f"{msg}\n\n（完整 traceback 已写入 {LOG_PATH}）")

    @Slot()
    def on_pause(self) -> None:
        if self.worker is None or not self.worker.isRunning():
            return
        paused = self.worker.toggle_pause()
        self.pause_btn.setText("继续" if paused else "暂停")
        self._simlog("任务已暂停" if paused else "任务已恢复，重新验证")
        pass  # dryrun 场景由 controller 内部处理；UI 预留

    @Slot()
    def on_estop(self) -> None:
        if self.worker is not None:
            self.worker.request_estop()
        """急停：真实 stage 立即 stop_all（immediate），非阻塞发送。"""
        self.state_label.setText("ESTOP REQUESTED")
        self._simlog("⚠ 急停请求（ESTOP）")
        for st in list(self.worker.stages if self.worker else []):
            def _stop(s=st):
                try:
                    s.stop_all()
                except Exception as exc:  # noqa: BLE001
                    _log_exception("estop stop_all failed", exc)
            threading.Thread(target=_stop, daemon=True,
                             name="estop-stop").start()

    @Slot(str, str)
    def on_state(self, state: str, detail: str) -> None:
        self.state_label.setText(state)
        self.detail_label.setText(detail)
        self._simlog(f"运行状态: {state} {detail}")

    @Slot(dict)
    def on_metrics(self, metrics: dict) -> None:
        self.detail_label.setText(self.detail_label.text() +
                                  f"\nmetrics: {metrics}")
        self._simlog(f"指标: {metrics}")
        motion = metrics.get("motion") if isinstance(metrics, dict) else None
        if motion and motion.get("last_telemetry"):
            self.on_motion(motion["last_telemetry"])

    @Slot(int)
    def on_done(self, code: int) -> None:
        self.pause_btn.setText("暂停")
        self.state_label.setText(self.state_label.text() +
                                 (" [OK]" if code == 0 else " [FAIL]"))
        self._simlog(f"运行完成 exit={code}")

    @Slot()
    def on_replay(self) -> None:
        path = self.report_path.text().strip()
        if not path or not os.path.exists(path):
            self.replay_out.setPlainText("报告文件不存在")
            return
        s = summarize(replay(path))
        self.replay_out.setPlainText("\n".join(
            f"{k}: {v}" for k, v in s.items()))


def run_qt_ui(argv=None) -> int:
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--offscreen" in argv:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        argv.remove("--offscreen")
    install_global_excepthook()   # 未捕获异常 -> 日志+弹窗，绝不闪退
    app = QtWidgets.QApplication(argv)
    win = MainWindow()
    win.show()
    return app.exec_() if hasattr(app, "exec_") else app.exec()


if __name__ == "__main__":
    raise SystemExit(run_qt_ui(sys.argv[1:]))
