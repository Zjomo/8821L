"""Qt UI（PLAN 第 6 节，PyQt5 兼容层）。

功能：场景选择、dryrun 运行（QThread）、帧+路径预览、状态/置信度面板、
急停/暂停按钮、JSONL 报告回放。全部基于 dryrun，不触碰真实设备。
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import threading
import time
import traceback
from typing import Optional

import cv2
import numpy as np

if __package__ in (None, ""):  # 支持直接运行：python app.py
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from obstacle_avoidance import sim_microscope, video_sim
    from obstacle_avoidance.qt_compat import (QtCore, QtGui, QtWidgets, Qt,
                                              Signal, Slot)
    from obstacle_avoidance.cli import SCENARIOS, build_scenario
    from obstacle_avoidance.controller import ObstacleAvoidController
    from obstacle_avoidance.models import RunState
    from obstacle_avoidance.motion import (MotionConfigError, MotionLimitError,
                                            MotionStateError,
                                            Simulated874xController)
    from obstacle_avoidance.planner import CollisionModel
    from obstacle_avoidance.reporter import (aggregate_experiments,
                                             extract_trajectory, replay,
                                             summarize)
    from obstacle_avoidance.roi_zones import (LayoutValidationError, RoiConfig,
                                               Zone, rect_contains_rect,
                                               validate_sim_layout)
    from obstacle_avoidance.vision import (get_shared_detector,
                                           try_shared_detector,
                                           warmup_shared_detector_async)
else:
    from . import sim_microscope, video_sim
    from .qt_compat import QtCore, QtGui, QtWidgets, Qt, Signal, Slot
    from .cli import SCENARIOS, build_scenario
    from .controller import ObstacleAvoidController
    from .models import RunState
    from .motion import (MotionConfigError, MotionLimitError,
                         MotionStateError, Simulated874xController)
    from .planner import CollisionModel
    from .reporter import (aggregate_experiments, extract_trajectory, replay,
                           summarize)
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


def _new_report_path() -> str:
    """每轮运行生成独立 JSONL 报告（Reports/run_时间戳_pid.jsonl），
    供日志/报告页『回放上一轮』使用。"""
    root = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".."))
    d = os.path.join(root, "Reports")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, time.strftime("run_%Y%m%d_%H%M%S")
                        + f"_{os.getpid()}.jsonl")


class WorkerThread(QtCore.QThread):
    """在后台线程跑 dryrun 场景，信号推送帧/路径/状态到 UI。"""

    frame_ready = Signal(np.ndarray)         # BGR 帧（含路径叠加）
    state_ready = Signal(str, str)           # state, detail
    metrics_ready = Signal(dict)
    finished_run = Signal(int)               # exit code
    error_occurred = Signal(str)             # 需弹窗的错误信息
    log_ready = Signal(str)                  # 需求5：运行期事件 -> UI 操作日志
    motion_ready = Signal(dict)              # XYZ 运动遥测
    report_ready = Signal(str)               # 本轮 JSONL 报告路径（回放用）
    stage_ready = Signal(dict)               # 需求3：运行结束时的最终台位（µm）
    plan_ready = Signal(list)                # 最新规划 waypoint（电机模式实时叠加）

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
        self._algorithm: str = "Alg1"
        self._execution_mode: str = "virtual"
        self._motion_params: Optional[dict] = None  # 球步长/速度/加速度
        self._origin_um: Optional[tuple] = None   # 运行视窗原点（WYSIWYG）
        self._virtual_motor: Optional[Simulated874xController] = None
        self.selected_ball_index: Optional[int] = None

    layout_updated = Signal(list)   # 运行成功后回传最终球位置（初始视窗坐标）

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

    def _emit_final_stage(self, world) -> None:
        """需求3：把运行结束时的最终台位回传 UI（虚拟模式）。

        电机模式（真实台位/相机世界）不移动、不回传。
        """
        if self._execution_mode != "virtual":
            return
        stage = getattr(world, "motion_stage", None)
        pos = getattr(stage, "position", None)
        if not isinstance(pos, dict):
            return
        try:
            self.stage_ready.emit({k: float(v) for k, v in pos.items()})
        except Exception as exc:  # noqa: BLE001 - 回传失败不影响结束流程
            _log_exception("emit final stage failed", exc)

    @property
    def stages(self) -> list:
        """运行期间注册的真实位移台（PicoMotorStage/SerialXYStage）。"""
        return self._stage_sink

    # ---- 帧推送与路径叠加
    def _latest_plan_points(self, rep):
        """从报告事件中取最近一次成功规划的 waypoint（倒序查找）。

        list() 快照：运行期控制器（写）与实时预览线程（读）会并发访问。
        """
        for e in reversed(list(rep.events)):
            if e.get("event") == "plan" and e.get("success"):
                return e.get("waypoints_px") or []
        return []

    def _install_plan_push(self, rep) -> None:
        """电机模式：把最新规划路径推给 UI 实时叠加（需求：同虚拟模式实时标注）。

        电机模式下 UI 实时画面（实时检测/自动识别）不是运行世界的 render 帧，
        故用报告事件驱动：每次成功规划即推送 waypoint，UI 画在实时帧上。
        """
        original = rep.log

        def log_and_push(event_type, task_id="", **data):
            record = original(event_type, task_id=task_id, **data)
            if event_type == "plan" and data.get("success"):
                pts = [(float(x), float(y))
                       for x, y in (data.get("waypoints_px") or [])]
                if len(pts) >= 2:
                    self.plan_ready.emit(pts)
            return record

        rep.log = log_and_push

    def _install_virtual_motor_logging(self, world, rep, algorithm: str) -> None:
        """Instrument virtual moves as 8742/8743 channel step events."""
        mp = self._motion_params or {}
        self._virtual_motor = Simulated874xController(
            steps_per_mm=float(mp.get("steps_per_mm", 33333.0)))

        def emit_steps(delta_mm, source, task_id="", track_id=-1):
            records = self._virtual_motor.command(
                delta_mm, source=source, task_id=task_id, track_id=track_id)
            for rec in records:
                rep.log("motor_step", algorithm=algorithm, **rec)
                self.log_ready.emit(
                    "8742/8743 CH{} {}{} {} steps ({:+.3f}um)".format(
                        rec["channel"], rec["axis"], rec["direction"],
                        rec["steps"], rec["delta_um"]))

        # Alg2 uses the XYZ motion model; wrapping this method captures X/Y/Z,
        # including the Z-safe/focus moves.  The values are in micrometres.
        motion = getattr(world, "motion_stage", None)
        if motion is not None and not getattr(motion, "_alg2_motor_wrapped", False):
            original_motion = motion.move_by

            def motion_logged(delta, source="ui"):
                tel = original_motion(delta, source=source)
                emit_steps({axis: value / 1000.0
                             for axis, value in tel.applied_delta.items()},
                            source=tel.source)
                return tel

            motion.move_by = motion_logged
            motion._alg2_motor_wrapped = True

        # Alg1's virtual stage updates the selected particle directly and does
        # not pass through VirtualXYZStage, so instrument its XY protocol too.
        original_make_stage = world.make_stage

        def make_stage_logged(*args, **kwargs):
            stage = original_make_stage(*args, **kwargs)
            original_move = stage.move_by

            def move_logged(dx, dy, *a2, **kw2):
                ok = original_move(dx, dy, *a2, **kw2)
                emit_steps({"x": dx, "y": dy},
                            "alg1-controller" if algorithm == "Alg1"
                            else "alg2-controller",
                            kw2.get("task_id", ""), kw2.get("track_id", -1))
                return ok

            stage.move_by = move_logged
            return stage

        world.make_stage = make_stage_logged

    @staticmethod
    def _draw_overlay(img, plan_pts, segment=None):
        """BGR 帧上叠加规划路径（绿）、waypoint（黄点）、当前目标段（青）。

        segment=(起点, 终点) 为窗口坐标；电机模式对准阶段没有 plan 事件时，
        用它标出球当前正在走的这一段，使实时画面与虚拟模式一样有路径标注。
        """
        import cv2
        out = img.copy()
        if plan_pts:
            pts = np.array([(int(round(x)), int(round(y))) for x, y in plan_pts],
                           dtype=np.int32)
            cv2.polylines(out, [pts], False, (0, 200, 0), 2)
            for p in pts:
                cv2.circle(out, (int(p[0]), int(p[1])), 3, (0, 220, 220), -1)
        if segment is not None:
            cv2.line(out, segment[0], segment[1], (255, 200, 0), 2)
            cv2.circle(out, segment[1], 4, (255, 200, 0), -1)
        return out

    @staticmethod
    def _start_motor_preview(emit_frame, interval_s: float = 0.07):
        """电机模式运行期：后台按固定帧率抓帧推送，画面持续动态更新。

        控制器每轮控制前才取一帧，位移台移动/串口等待期间画面静止；该线程
        与控制器共享同一 frame_source（CameraWorld 内部加锁串行取帧），
        返回 (thread, stop_event)，运行结束由调用方 set() 停止。
        """
        stop = threading.Event()

        def pump():
            while not stop.wait(interval_s):
                try:
                    emit_frame()
                except RuntimeError:
                    return    # 窗口/对象已销毁：静默退出
                except Exception as exc:  # noqa: BLE001 - 预览失败不中断运行
                    _log_exception("motor preview frame failed", exc)

        thread = threading.Thread(target=pump, daemon=True,
                                  name="motor-preview")
        thread.start()
        return thread, stop

    def _motor_target_segment(self, world):
        """电机模式『当前目标段』：目标球心 -> 当前目标点（窗口坐标）。

        Alg2 固定光斑模式在对准阶段把球直线移向光斑，但该阶段不产生 plan
        事件，画面里看不到任何路径——这里按当前目标补画一段。
        """
        if self._execution_mode != "motor" or not self._controller_ref:
            return None
        stage = getattr(self._controller_ref.get("controller"), "stage", None)
        try:
            beam = getattr(stage, "beam_position_px", None)
        except Exception:  # noqa: BLE001 - 光斑未标定：不画目标段
            return None
        pipeline = getattr(world, "pipeline", None)
        if beam is None or pipeline is None:
            return None
        particles = list(pipeline.tracker.active_particles())
        if not particles:
            return None
        track_id = getattr(stage, "track_id", None)
        ball = next((p for p in particles if p.track_id == track_id), None)
        if ball is None:   # 目标尚未锁定：取离光斑最近的一球
            ball = min(particles, key=lambda p: (
                (p.position_px[0] - beam[0]) ** 2
                + (p.position_px[1] - beam[1]) ** 2))
        start = (int(round(ball.position_px[0])), int(round(ball.position_px[1])))
        end = (int(round(beam[0])), int(round(beam[1])))
        return (start, end) if start != end else None

    def _alg2_nearest_ball_idx(self, ball_idx, balls) -> int:
        """Alg2：返回离光斑标定点最近的球序号（需求2，不默认 0 号球）。

        标定激光控制下，用"离标定点最近"的球作为优先移动目标；其余球由
        planner 视为动态障碍。标定点未配置时退化为 ball_idx[0]。
        """
        beam = tuple((self._motion_params or {}).get(
            "beam_position_px", (None, None)))
        best, best_d = None, 1e18
        for bi in ball_idx:
            bcx = balls[bi][0] + balls[bi][2] / 2.0
            bcy = balls[bi][1] + balls[bi][3] / 2.0
            d = (math.dist((bcx, bcy), beam) if beam[0] is not None else 0.0)
            if best is None or d < best_d:
                best, best_d = bi, d
        return best if best is not None else ball_idx[0]

    def _alg2_ring_seat(self, group_idx, balls, goal_pt):
        """Alg2 交互逐个搬运：目标点已有球时，新球绕已就位球环形紧贴就位。

        座位：座位0=目标点中心；座位 m>=1 处于第 (m-1)//6 环、第 (m-1)%6
        方位，环半径 = 2r*(1+环)，球心距保持 2r（相切紧贴）。从中心座位起
        逐个扫描，返回第一个未被本组任何球占用的驻点，天然保证不重叠。
        """
        gx, gy = float(goal_pt[0]), float(goal_pt[1])
        rs = [(balls[i][2] + balls[i][3]) / 4.0 for i in group_idx]
        ball_r = max(rs) if rs else 12.0
        # 现有球心集合（坐标 -> 中心），用于判占位
        centers = [
            (balls[i][0] + balls[i][2] / 2.0,
             balls[i][1] + balls[i][3] / 2.0) for i in group_idx]

        def occupied(x, y, tol=1.5 * ball_r):
            return any(math.dist((x, y), c) <= tol for c in centers)

        if not occupied(gx, gy, tol=ball_r):
            return (gx, gy)
        m = 1
        while True:
            ring = (m - 1) // 6
            slot = (m - 1) % 6
            ang = 2 * math.pi * slot / 6.0
            rr = 2.0 * ball_r * (1 + ring)
            x, y = gx + rr * math.cos(ang), gy + rr * math.sin(ang)
            if not occupied(x, y):
                return (x, y)
            m += 1

    def _run_single_oa(self, world, rep, ball_rect, goal_pt,
                       peer_balls=()):
        """单球避障：把 ball_rect 对应的球移动到 goal_pt。

        需求(接触体积/防误触)：peer_balls 为"本回合其他圆球"（(cx,cy,半径)），
        运行期以真实半径+缓冲转为圆形障碍一并交给规划/控制，使运动中圆球
        主动避让其他球、障碍，并按接触体积（重叠深度）防误触。
        """
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
        # 其他圆球 -> 圆形障碍（避免与目标球自身重复）。
        # 注意：peer 是"另一颗球"而非静态障碍，planner 会对其再叠加 infl
        # （ball_radius + spot_radius + safety_margin）。若直接传物理半径，
        # 最终阻塞半径 = peer_r + (ball_r + spot_r + safety)，对"球-球"避让
        # 过度膨胀（球与球本应只按 peer_r + ball_r 相切即可），会误吞运动球
        # 的光斑起点/目标驻点导致 low_clearance。因此把传入半径扣掉光斑与
        # 安全余量，使实际阻塞半径回到物理合理的 ball_r + peer_r。
        from obstacle_avoidance.planner import CollisionModel
        _cm = CollisionModel()
        _peer_credit = _cm.spot_radius_px + _cm.safety_margin_px
        from obstacle_avoidance.models import Obstacle
        extra = []
        for bcx, bcy, br in peer_balls or ():
            # 保底：只需非负（避免网格里出现负半径）；小 peer 球自然更贴。
            r_eff = max(0.0, br - _peer_credit)
            extra.append(Obstacle(kind="circle", center=(bcx, bcy),
                                  radius=r_eff))
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
        mp = self._motion_params or {}
        is_alg2 = getattr(self, "_algorithm", "Alg1") == "Alg2"
        cfg = ControllerConfig(
            max_step_mm=float(mp.get("step_mm", 0.01)),
            tolerance_px=8.0, stable_frames=3,
            max_iterations=400, max_track_jump_px=250.0,
            # SimMicroscopeWorld translates the stage by the inverse camera
            # shift, so the observed particle displacement has the same sign
            # as the controller command.
            ball_shift_sign=1)
        if is_alg2:
            if __package__ in (None, ""):
                from obstacle_avoidance.algorithm2 import (Alg2Config,
                                                            Alg2Stage,
                                                            FixedBeamController)
            else:
                from .algorithm2 import (Alg2Config, Alg2Stage,
                                         FixedBeamController)
            # 需求(交互逐球)：光束沿用真实标定点（UI 旋钮 beam_position_px），
            # 由 Alg2 先在 CALIBRATING 阶段把光斑对准目标球（标定），随后
            # beam_lock 锁定，TRACKING 阶段由固定光斑带动该球移动（通过反向
            # 移动样品实现），而非让球自己直接飞向目标点。
            beam = tuple((self._motion_params or {}).get(
                "beam_position_px", (world.window[0] / 2.0,
                                     world.window[1] / 2.0)))
            world.set_beam_position(beam)
            stage = Alg2Stage(
                world, tid,
                config=Alg2Config(beam_position_px=beam,
                                  beam_calibration_confidence=1.0,
                                  image_shift_sign=-1))
            stage.prepare_focus()
            rep.log("beam_target", task_id="sim01-oa", track_id=tid,
                    action="select", algorithm="Alg2")
        else:
            stage = world.make_stage(tid)
        self._stage_sink.append(stage)
        planner = GridPlanner()
        pipeline = VisionPipeline(
            detector, tracker=ParticleTracker(max_jump_px=220.0))
        ctl = (FixedBeamController(stage, pipeline, planner, cfg, rep)
               if is_alg2 else
               ObstacleAvoidController(stage, pipeline, planner, cfg, rep))
        self._controller_ref["controller"] = ctl
        self._controller_ref["controller"] = ctl
        return ctl.run(snap, tid, goal, task_id="sim01-oa",
                       extra_obstacles=extra,
                       get_frame=world.render, get_snapshot=world.snapshot)

    def _run_sim_assembly(self, world, rep, gi, ball_idx, balls, center,
                          radius_px, pump_events, final_balls, set_ball_at):
        """多球组装：以 center 为范围中心环形驻点，逐球串行避障移动。

        AG 目标范围与 OA 多球共用一个目标点均走此流程：环形驻点间距按
        有效球半径计算（球间不重叠），每球独立控制器，已就位球作为
        碰撞障碍。返回 (final_state, run_failed)。
        """
        if __package__ in (None, ""):
            from obstacle_avoidance.aggregation import (AggregationConfig,
                                                        AggregationPlanner)
            from obstacle_avoidance.controller import ControllerConfig
            from obstacle_avoidance.models import GoalRegion
            from obstacle_avoidance.planner import GridPlanner
            from obstacle_avoidance.sim_microscope import _initial_detect
            from obstacle_avoidance.vision import (ParticleTracker,
                                                   VisionPipeline)
        else:
            from .aggregation import AggregationConfig, AggregationPlanner
            from .controller import ControllerConfig
            from .models import GoalRegion
            from .planner import GridPlanner
            from .sim_microscope import _initial_detect
            from .vision import ParticleTracker, VisionPipeline
        if world.pipeline is None:
            detector = world.make_detector()
            _initial_detect(
                world, detector, hint=None,
                pipeline=VisionPipeline(
                    detector,
                    tracker=ParticleTracker(max_jump_px=220.0)))
        # 组装序列里两球贴近会造成数帧轮廓粘连（blob 合并）：tracker 的
        # 粘连保护会让相关 track coast 保位，但默认 3 帧就丢 ID，粘连稍久
        # track 即被丢弃、blob 另建新 track 污染后续规划。放宽保留帧数，
        # 粘连结束后按最近邻自然复配。
        _tr = getattr(world.pipeline, "tracker", None)
        if _tr is not None:
            _tr.keep_lost_frames = max(
                int(getattr(_tr, "keep_lost_frames", 0) or 0), 30)
        # 需求(组装编号)：按球心到范围中心距离升序编号（0 = 最近），
        # 0 号球先移到范围中心，其余按序号环形紧贴（驻点由
        # AggregationPlanner.assign 生成：中心 + 2*r_eff 相切环）。
        ball_idx = sorted(
            ball_idx,
            key=lambda bi: math.dist(
                (balls[bi][0] + balls[bi][2] / 2.0,
                 balls[bi][1] + balls[bi][3] / 2.0), center))
        rep.log("ball_numbering", task_id=f"sim01-G{gi}-ag",
                detail="assembly order: nearest ball first (No.0 -> center)",
                ball_order=[int(bi) for bi in ball_idx],
                center=[round(float(center[0]), 1),
                        round(float(center[1]), 1)])
        # 本 ground 的球 -> 检测 track id（贪心最近匹配，不跨 ground 误聚）
        snap = world.snapshot()
        track_ids, used_t = [], set()
        for bi in ball_idx:
            bc = (balls[bi][0] + balls[bi][2] / 2.0,
                  balls[bi][1] + balls[bi][3] / 2.0)
            cands = sorted(
                (p for p in snap.particles if p.track_id not in used_t),
                key=lambda q: math.dist(q.position_px, bc))
            if cands:
                track_ids.append(cands[0].track_id)
                used_t.add(cands[0].track_id)
        goal = GoalRegion(center=center, radius_px=radius_px)
        ag = AggregationPlanner(
            GridPlanner(),
            world.pipeline or VisionPipeline(),
            AggregationConfig(
                required_count=len(ball_idx),
                controller=ControllerConfig(
                    max_step_mm=float(
                        (self._motion_params or {}).get("step_mm", 0.01)),
                    tolerance_px=8.0,
                    stable_frames=3,
                    max_iterations=400,
                    max_track_jump_px=250.0,
                    prefer_track_id=True)),
            rep, controller_sink=self._controller_ref)
        if getattr(self, "_algorithm", "Alg1") == "Alg2":
            world.alg2_mode = True
        result = ag.run(world, world.snapshot(), goal,
                        track_ids=track_ids or None,
                        task_id=f"sim01-G{gi}-ag")
        pump_events()   # run_end 在最后一次渲染后，需补泵
        final_state = result.final_state
        run_failed = result.final_state != RunState.COMPLETE
        if not run_failed:
            # 移动完成：按实际最终检测位置回写各球（环目标位置，
            # 避免多球叠在同一中心）
            try:
                parts = list(world.pipeline.tracker.active_particles())
            except Exception:  # noqa: BLE001
                parts = []
            used = set()
            for p in sorted(parts, key=lambda q: q.track_id):
                best_i, best_d = None, 1e18
                for bi in ball_idx:
                    if bi in used:
                        continue
                    bx = final_balls[bi][0] + final_balls[bi][2] / 2
                    by = final_balls[bi][1] + final_balls[bi][3] / 2
                    d = ((p.position_px[0] - bx) ** 2
                         + (p.position_px[1] - by) ** 2)
                    if d < best_d:
                        best_d, best_i = d, bi
                if best_i is not None:
                    used.add(best_i)
                    set_ball_at(best_i, p.position_px[0], p.position_px[1])
            for bi in ball_idx:   # 兜底：未匹配的球落中心
                if bi not in used:
                    set_ball_at(bi, center[0], center[1])
            self.log_ready.emit(
                f"球{sorted(used)} 移动完成 -> 目标中心 "
                f"({center[0]:.0f},{center[1]:.0f})")
            self.layout_updated.emit(list(final_balls))
        return final_state, run_failed

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
                                      sim_layout=layout,
                                      origin_um=self._origin_um)
                world.alg2_mode = (getattr(self, "_algorithm", "Alg1") == "Alg2")
                if world.alg2_mode:
                    beam = tuple((self._motion_params or {}).get(
                        "beam_position_px", (world.window[0] / 2.0,
                                             world.window[1] / 2.0)))
                    world.set_beam_position(beam)
                # 圆球移动参数：X/Y 轴速度/加速度（步长在控制器里生效）
                mp = self._motion_params or {}
                if mp:
                    for ax in ("x", "y"):
                        try:
                            world.motion_stage.configure_axis(
                                ax, max_speed=float(mp["speed_ums"]),
                                acceleration=float(mp["accel_ums2"]))
                        except Exception:  # noqa: BLE001 - 配置失败不阻断运行
                            pass
                rep = RunReporter(_new_report_path())
                rep.log("run_start", scenario="sim01", algorithm=self._algorithm,
                        task_mode=mode, mode="virtual",
                        controller="8742/8743-sim",
                        steps_per_mm=float((self._motion_params or {}).get(
                            "steps_per_mm", 33333.0)))
                self._install_virtual_motor_logging(world, rep, self._algorithm)
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
                            # 需求(日志)：报错尽量带出可诊断上下文——任务、目标
                            # 球、失败原因与关键指标，避免"光秃秃一条错误"。
                            detail = str(ev.get("reason", ""))
                            ctx = []
                            for k, lbl in (("task_id", "task"),
                                           ("track_id", "球"),
                                           ("final_state", "状态"),
                                           ("iterations", "迭代"),
                                           ("final_error_px", "残余px")):
                                v = ev.get(k)
                                if v not in (None, ""):
                                    ctx.append(f"{lbl}={v}")
                            suffix = (" " + " ".join(ctx)) if ctx else ""
                            self.log_ready.emit(f"错误: {detail}{suffix}")
                        elif et == "stage_command":
                            # 需求(日志)：逐条台位移动指令（µm 位移 / 航点 /
                            # 规划版本），便于回放定位"哪一步动到哪"。
                            self.log_ready.emit(
                                f"台位移动 dx={ev.get('dx_mm', 0) * 1000:+.1f}µm "
                                f"dy={ev.get('dy_mm', 0) * 1000:+.1f}µm "
                                f"wp={ev.get('waypoint_index', '-')} "
                                f"v={ev.get('source_plan_version', '-')}")
                        elif et == "element_offscreen":
                            # 需求2：元素离屏不中止，用外推位置继续定位
                            pos = ev.get("position_px")
                            where = (f"({pos[0]:.0f},{pos[1]:.0f})"
                                     if isinstance(pos, (list, tuple))
                                     and len(pos) == 2 else "")
                            self.log_ready.emit(
                                f"元素离屏外推: {ev.get('role', '?')} "
                                f"track={ev.get('track_id', '-')} {where} "
                                f"{ev.get('detail', '')}".rstrip())
                        elif et == "run_end":
                            # 需求(日志)：结束不止报状态，附失败原因（若有）。
                            fs = ev.get('final_state')
                            if fs in ("COMPLETE", "IDLE"):
                                self.log_ready.emit(f"运行结束: {fs}")
                            else:
                                detail = (ev.get("metrics") or {}).get(
                                    "failure_reason") or \
                                    (ev.get("metrics") or {}).get("detail") \
                                    or ""
                                self.log_ready.emit(
                                    f"运行结束: {fs}"
                                    + (f"（{detail}）" if detail else ""))
                        elif et == "motor_step":
                            self.log_ready.emit(
                                f"8742/8743 {ev.get('axis')}"
                                f"{ev.get('direction')} {ev.get('steps')} steps")
                        elif et == "beam_target":
                            self.log_ready.emit(
                                f"beam target track={ev.get('track_id')}")
                        elif et == "beam_lock":
                            pos = ev.get("position_px") or ("?", "?")
                            self.log_ready.emit(
                                f"Alg2 球{ev.get('track_id')} 已锁定固定光斑 "
                                f"({pos[0]:.1f},{pos[1]:.1f})，样品/镜头移动"
                                if isinstance(pos[0], (int, float)) else
                                "Alg2 目标球已锁定固定光斑，样品/镜头移动")
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
                final_balls = [list(b) for b in layout["balls"]]

                def _set_ball_at(bi, cx, cy):
                    """球 bi 移动完成：中心落到 (cx,cy)，尺寸不变。"""
                    _, _, bw, bh = final_balls[bi]
                    final_balls[bi] = [int(cx - bw / 2), int(cy - bh / 2),
                                       bw, bh]

                for gi, grect in enumerate(grounds):
                    ball_idx = ball_groups[gi]
                    if self.selected_ball_index is not None:
                        ball_idx = [bi for bi in ball_idx
                                    if bi == self.selected_ball_index]
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
                    # 多球共用目标点时按组装逻辑环形驻点
                    if mode == "oa":
                        goal_pt = goals[gi]
                        if not goal_pt: continue
                        is_alg2 = (getattr(self, "_algorithm", "Alg1")
                                   == "Alg2")
                        # 需求(多球 Alg2 交互逐球搬运)：标定激光下存在多球时，
                        # 每回合只移动"被点选的一个球"（selected_ball_index，
                        # 无点选则退化为离标定点/光斑最近的一球），其余球由
                        # planner 作动态障碍；_run_single_oa 会先把标定激光
                        # 移到目标球心、再带球避障到目标点，完成后把最终位置
                        # 回写 UI。若目标点已有球，本轮球的驻点按环形紧贴计算
                        # （第 0 球压中心，后续球环绕相切错开）。单球仍走原路径。
                        if len(ball_idx) == 1 or is_alg2:
                            _bi = (ball_idx[0]
                                   if len(ball_idx) == 1
                                   else self._alg2_nearest_ball_idx(
                                       ball_idx, balls))
                            # 目标驻点：Alg2 多球且在选球时，按本 ground 已就位球
                            # 环形紧贴；否则（单品球 / 无点选）仍压目标点中心。
                            tgt = (self._alg2_ring_seat(
                                       ball_groups[gi], balls, goal_pt)
                                   if is_alg2 else goal_pt)
                            # 需求(防误触)：把本 ground 其他圆球作圆形障碍一并
                            # 交给规划/控制，使运动中圆球主动避让、不误触他球。
                            peer_balls = []
                            for _pb in ball_groups[gi]:
                                if _pb == _bi:
                                    continue
                                pbr = ((balls[_pb][2] + balls[_pb][3]) /
                                       4.0)
                                peer_balls.append(
                                    (balls[_pb][0] + balls[_pb][2] / 2.0,
                                     balls[_pb][1] + balls[_pb][3] / 2.0,
                                     pbr))
                            result = self._run_single_oa(
                                world, rep, balls[_bi], tgt, peer_balls)
                            if result is None or \
                                    result.final_state != RunState.COMPLETE:
                                final_state = getattr(
                                    result, "final_state", RunState.ABORTED)
                                run_failed = True
                                # 需求(日志)：移动失败出具可诊断原因——目标球、
                                # 具体故障码与残余误差，取代"就一句失败"。
                                if result is not None:
                                    md = result.to_dict()
                                    fr = md.get("failure_reason") \
                                        or md.get("detail") or ""
                                    self.log_ready.emit(
                                        f"球{_bi} 移动失败: "
                                        f"status={md.get('final_state')}"
                                        + (f" 原因={fr}" if fr else "")
                                        + (f" 残余={md.get('final_error_px'):.0f}px"
                                           if md.get("final_error_px") is not None
                                           else "")
                                        + f" 迭代={md.get('iterations')}")
                            else:
                                # 移动完成：更新球位置到目标驻点并回传 UI
                                _set_ball_at(_bi, tgt[0], tgt[1])
                                self.log_ready.emit(
                                    f"球{_bi} 移动完成 -> 目标驻点 "
                                    f"({tgt[0]:.0f},{tgt[1]:.0f})")
                                self.layout_updated.emit(list(final_balls))
                                pump_events()   # run_end 补泵
                        else:
                            # 多球共用一个目标点：目标点=组装范围中心，
                            # 环形驻点逐球避障移动（同 AG 逻辑，驻点间距
                            # 按有效球半径保证球间不碰撞，已就位球当障碍）
                            radius_px = 40.0 * math.sqrt(len(ball_idx))
                            final_state, run_failed = self._run_sim_assembly(
                                world, rep, gi, ball_idx, balls,
                                (float(goal_pt[0]), float(goal_pt[1])),
                                radius_px, pump_events, final_balls,
                                _set_ball_at)
                    else:  # ag
                        gr = goal_ranges[gi]
                        if not gr: continue
                        cx, cy = gr[0] + gr[2] / 2, gr[1] + gr[3] / 2
                        # 半径按球数放大：环目标在 0.5R，需保证环上球间
                        # clearance >= inflation(22px)，否则后到球被拒
                        radius_px = max(min(gr[2], gr[3]) / 2,
                                        40.0 * math.sqrt(len(ball_idx)))
                        final_state, run_failed = self._run_sim_assembly(
                            world, rep, gi, ball_idx, balls, (cx, cy),
                            radius_px, pump_events, final_balls,
                            _set_ball_at)
                    if run_failed:
                        break
                # 需求3：运行结束不回起点——把最终台位回传 UI，
                # 由 UI 把实时世界的视窗（台位）同步到运行终点。
                self._emit_final_stage(world)
                world.close()
                self.layout_updated.emit(list(final_balls))
                self.state_ready.emit(
                    final_state.value if final_state else "COMPLETE",
                    f"sim01 [{mode}] 全部 ground 完成")
                rep.finalize_experiment(algorithm=self._algorithm,
                                        task_mode=mode, mode="virtual")
                rep.close()
                self.report_ready.emit(rep.path)
                self.finished_run.emit(0 if not run_failed and
                                       final_state in (None, RunState.COMPLETE)
                                       else 1)
                return

            # ---- 默认路径（单 ground / 其他场景） ----
            world, run_fn = (self._builder() if self._builder is not None
                             else _cli_build(self.scenario,
                                             sample_spec=self._sample_spec))
            rep = RunReporter(_new_report_path())
            rep.log("run_start", scenario=self.scenario,
                    algorithm=self._algorithm,
                    task_mode=self._run_mode,
                    mode=self._execution_mode,
                    controller=("8742/8743-sim" if self._execution_mode == "virtual"
                                else "real-stage"))
            if self._execution_mode == "virtual":
                self._install_virtual_motor_logging(world, rep, self._algorithm)
            else:
                self._install_plan_push(rep)   # 电机模式：规划路径实时推给 UI
            orig_render = world.render

            def render_and_emit():
                img = orig_render()
                telemetry = getattr(getattr(world, "motion_stage", None),
                                    "last_telemetry", None)
                if telemetry is not None:
                    self.motion_ready.emit(telemetry.to_dict())
                overlay = self._draw_overlay(
                    img, self._latest_plan_points(rep),
                    self._motor_target_segment(world))
                self.frame_ready.emit(overlay)
                return img

            world.render = render_and_emit
            # 先推送一帧初始场景
            self.frame_ready.emit(self._draw_overlay(orig_render(), []))

            def emit_preview_frame():
                img = orig_render()
                if img is None:
                    return
                self.frame_ready.emit(self._draw_overlay(
                    img, self._latest_plan_points(rep),
                    self._motor_target_segment(world)))

            # 电机模式：控制器只在每轮控制前取一帧，位移台移动/等待期间画面
            # 是静止的；这里按固定帧率后台抓帧推送，使显微镜画面实时更新。
            preview = (self._start_motor_preview(emit_preview_frame)
                       if self._execution_mode == "motor" else None)
            try:
                result = run_fn(rep=rep, stage_sink=self._stage_sink)
            finally:
                if preview is not None:
                    thread, stop = preview
                    stop.set()
                    thread.join(timeout=1.0)
            # 需求3：虚拟模式运行结束回传最终台位（电机模式不动真实台位）
            self._emit_final_stage(world)
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
        rep.finalize_experiment(algorithm=self._algorithm,
                                task_mode=self._run_mode,
                                mode=self._execution_mode)
        rep.close()
        self.report_ready.emit(rep.path)
        self.finished_run.emit(0 if result.final_state == RunState.COMPLETE
                               or getattr(result, "completed", False) else 1)


def frame_to_pix(bgr: np.ndarray) -> QtGui.QImage:
    """BGR 帧转 QImage：Format_BGR888 免通道交换复制（仅 QImage.copy
    一次持有数据），比 RGB888 路径每帧少一次全帧复制。"""
    h, w, _ = bgr.shape
    src = np.ascontiguousarray(bgr)
    img = QtGui.QImage(src.data, w, h, 3 * w, QtGui.QImage.Format_BGR888)
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


class AutoRecognizeThread(QtCore.QThread):
    """Latest-frame Alg2 workspace recognition without blocking the Qt UI."""

    result_ready = Signal(object)
    recognition_failed = Signal(str)

    def __init__(self, detector_fn, min_confidence: float,
                 parent=None, manual_substrate_polygon=None) -> None:
        super().__init__(parent)
        self._detector_fn = detector_fn
        self._min_confidence = float(min_confidence)
        self._lock = threading.Lock()
        self._frame = None
        self._has_new = False
        self._running = True
        self._frame_id = 0
        self._manual_substrate_polygon = manual_substrate_polygon
        self._substrate_dirty = False

    def submit(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame.copy()
            self._has_new = True

    def set_manual_substrate(self, polygon) -> None:
        """『衬底区域』画框/撤销后换用手动多边形（下一帧生效）。"""
        with self._lock:
            self._manual_substrate_polygon = (
                [(float(x), float(y)) for x, y in polygon]
                if polygon else None)
            self._substrate_dirty = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:  # noqa: D102
        detector = self._detector_fn()
        if detector is None:
            self.recognition_failed.emit("YOLO 模型尚未加载完成")
            return
        try:
            if __package__ in (None, ""):
                from obstacle_avoidance.auto_recognition import (
                    Alg2AutoRecognizer, AutoRecognitionConfig)
            else:
                from .auto_recognition import (Alg2AutoRecognizer,
                                               AutoRecognitionConfig)
            recognizer = Alg2AutoRecognizer(
                particle_detector=detector,
                manual_substrate_polygon=self._manual_substrate_polygon,
                config=AutoRecognitionConfig(
                    minimum_overall_confidence=self._min_confidence,
                    # 衬底/障碍一律手动框定（『衬底区域』/『障碍区』）
                    auto_substrate=False,
                    auto_obstacles=False))
        except Exception as exc:  # noqa: BLE001
            self.recognition_failed.emit(str(exc))
            return
        while self._running:
            with self._lock:
                frame, new = self._frame, self._has_new
                self._has_new = False
                polygon = self._manual_substrate_polygon
                dirty = self._substrate_dirty
                self._substrate_dirty = False
            if dirty:
                # 画框即"当前帧"的衬底：让识别器把位移参考挪到此刻，否则
                # 衬底框会带着画框前累计的位移整体偏移。
                recognizer.set_manual_substrate(polygon)
            if not new:
                self.msleep(15)
                continue
            try:
                result = recognizer.process(frame, self._frame_id, time.time())
                self._frame_id += 1
                self.result_ready.emit(result)
            except Exception as exc:  # noqa: BLE001 - keep live preview alive
                self.recognition_failed.emit(str(exc))


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
    points_drawn = Signal(list)              # 自由多边形 [(x,y),..]（帧坐标）
    drag_start = Signal(int, int)            # 帧坐标 x, y（鼠标按下）
    drag_move = Signal(int, int)             # 帧坐标增量 dx, dy（鼠标移动）
    drag_end = Signal()                      # 鼠标释放（拖拽结束）
    target_clicked = Signal(int, int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._mapper = None            # (qx,qy)->(fx,fy)
        self._origin = None            # 按下时的帧坐标
        self._last_pos = None          # 上次移动时的帧坐标
        self.drawing_enabled = True    # 画框闸门（MainWindow 默认锁定防误触）
        self.setMouseTracking(False)   # 默认不跟踪；需时由外部开启
        # 自由多边形（框定形状=Free）：逐点点击，双击/右键/点击起点闭合
        self.polygon_mode = False
        self._poly_pts = []            # 顶点（widget 坐标）
        self._poly_cursor = None       # 橡皮线端点（widget 坐标）

    def set_polygon_mode(self, on: bool) -> None:
        """切换自由多边形模式（清空未完成的顶点，同步鼠标跟踪）。"""
        if self.polygon_mode != on:
            self.polygon_mode = on
            self.setMouseTracking(on)
        self.cancel_polygon()

    def cancel_polygon(self) -> None:
        self._poly_pts = []
        self._poly_cursor = None
        self.update()

    def _poly_close(self) -> None:
        """顶点 >=3 时闭合多边形并发射 points_drawn（帧坐标）。"""
        mapped = [self._to_frame(p) for p in self._poly_pts]
        pts = [(int(mp[0]), int(mp[1])) for mp in mapped if mp is not None]
        self.cancel_polygon()
        if len(pts) >= 3:
            self.points_drawn.emit(pts)

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
        pos = self._event_pos(ev)
        if self.polygon_mode and self.drawing_enabled:
            if ev.button() == QtCore.Qt.LeftButton:
                if (len(self._poly_pts) >= 3
                        and (pos - self._poly_pts[0]).manhattanLength() <= 12):
                    self._poly_close()          # 点击起点闭合
                else:
                    self._poly_pts.append(pos)
                self.update()
                return
            if ev.button() == QtCore.Qt.RightButton:
                if len(self._poly_pts) >= 3:
                    self._poly_close()          # 右键闭合
                else:
                    self.cancel_polygon()       # 右键取消
                return
        if ev.button() == QtCore.Qt.LeftButton:
            fp = self._to_frame(pos)
            self._origin = fp
            self._last_pos = fp
            if fp is not None:
                self.drag_start.emit(int(fp[0]), int(fp[1]))
        super().mousePressEvent(ev)

    def mouseDoubleClickEvent(self, ev) -> None:  # noqa: N802
        if self.polygon_mode and self.drawing_enabled \
                and len(self._poly_pts) >= 3:
            self._poly_close()                  # 双击闭合
            return
        super().mouseDoubleClickEvent(ev)

    def mouseMoveEvent(self, ev) -> None:  # noqa: N802
        if self.polygon_mode:
            if self._poly_pts:
                self._poly_cursor = self._event_pos(ev)
                self.update()                   # 橡皮线跟随
            return
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
        if self.polygon_mode:
            return
        if ev.button() == QtCore.Qt.LeftButton and self._origin is not None:
            end = self._to_frame(self._event_pos(ev))
            if end is not None:
                x0, y0 = self._origin
                x, y = min(x0, end[0]), min(y0, end[1])
                w, h = abs(end[0] - x0), abs(end[1] - y0)
                if w <= 4 and h <= 4:
                    self.target_clicked.emit(int(end[0]), int(end[1]))
                if w > 4 and h > 4 and self.drawing_enabled:
                    self.rect_drawn.emit(int(x), int(y), int(w), int(h))
            self._origin = None
            self._last_pos = None
            self.drag_end.emit()
        super().mouseReleaseEvent(ev)

    def paintEvent(self, ev) -> None:  # noqa: N802
        super().paintEvent(ev)
        if not self._poly_pts:
            return
        qp = QtGui.QPainter(self)
        # 已落顶点：青色折线 + 白色顶点
        qp.setRenderHint(QtGui.QPainter.Antialiasing, True)
        qp.setPen(QtGui.QPen(QtGui.QColor(0, 220, 220), 2))
        for a, b in zip(self._poly_pts, self._poly_pts[1:]):
            qp.drawLine(a, b)
        qp.setPen(QtGui.QPen(QtGui.QColor(240, 240, 240), 1))
        qp.setBrush(QtGui.QColor(240, 240, 240))
        for p in self._poly_pts:
            qp.drawEllipse(p, 3, 3)
        qp.setBrush(QtCore.Qt.NoBrush)
        if len(self._poly_pts) >= 3:
            # 闭合预览边（绿虚线：起点->末点）
            qp.setPen(QtGui.QPen(QtGui.QColor(0, 210, 90), 1, Qt.DashLine))
            qp.drawLine(self._poly_pts[0], self._poly_pts[-1])
        if self._poly_cursor is not None:
            # 橡皮线（灰虚线：末点->光标）
            qp.setPen(QtGui.QPen(QtGui.QColor(160, 160, 160), 1, Qt.DashLine))
            qp.drawLine(self._poly_pts[-1], self._poly_cursor)


class ScreenRegionOverlay(QtWidgets.QDialog):
    """全屏半透明覆盖层：拖拽框选屏幕区域。

    返回 region=(x,y,w,h) 为 mss 物理像素坐标（虚拟桌面全局）；
    系统显示缩放 !=100% 时按所在屏 devicePixelRatio 自动换算，
    多屏时用物理分辨率匹配显示器原点（兼容混合 DPI）。Esc 取消。
    """

    def __init__(self) -> None:
        super().__init__(None, QtCore.Qt.FramelessWindowHint
                         | QtCore.Qt.WindowStaysOnTopHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setWindowModality(QtCore.Qt.ApplicationModal)
        self.setWindowState(QtCore.Qt.WindowFullScreen)
        self.setGeometry(QtWidgets.QApplication.primaryScreen().virtualGeometry())
        self.setCursor(QtCore.Qt.CrossCursor)
        self._origin = None
        self._rect = None          # 覆盖层局部坐标
        self.region = None         # 物理像素 (x, y, w, h)

    @staticmethod
    def _physical_region(rect, screen, monitors=None, virtual=None):
        """逻辑全局坐标矩形 -> 物理像素区域 (x,y,w,h)。

        优先用 Qt 虚拟桌面逻辑尺寸与 mss 全桌面物理尺寸的比值换算：
        Windows 系统缩放 >100% 时 Qt（DPI-unaware）的 devicePixelRatio()
        恒为 1，逐屏 DPR 换算会得到缩小的错误区域——全局比值同时覆盖
        DPI-aware 与 DPI-unaware 两种情形。换算后裁剪到矩形中心点所在
        显示器（混合 DPI 下防越界/串屏）。
        monitors/virtual 不可用时退回 devicePixelRatio 逐屏换算。
        """
        def _fallback():
            ratio = screen.devicePixelRatio() or 1.0
            dx = int((rect.x() - screen.geometry().x()) * ratio)
            dy = int((rect.y() - screen.geometry().y()) * ratio)
            ox = int(screen.geometry().x() * ratio)
            oy = int(screen.geometry().y() * ratio)
            return (ox + dx, oy + dy,
                    int(rect.width() * ratio), int(rect.height() * ratio))

        if monitors is None:
            try:
                import mss
                sct = mss.MSS() if hasattr(mss, "MSS") else mss.mss()
                monitors = sct.monitors
            except Exception:  # noqa: BLE001 - mss 不可用时退回 DPR 换算
                monitors = []
        if not monitors or len(monitors) < 2:
            return _fallback()
        if virtual is None:
            virtual = screen.virtualGeometry()
        virt = monitors[0]
        rx = virt["width"] / max(1.0, float(virtual.width()))
        ry = virt["height"] / max(1.0, float(virtual.height()))
        px = int(virt["left"] + (rect.x() - virtual.x()) * rx)
        py = int(virt["top"] + (rect.y() - virtual.y()) * ry)
        pw = max(1, int(rect.width() * rx))
        ph = max(1, int(rect.height() * ry))
        # 裁剪到矩形中心点所在显示器
        cx, cy = px + pw // 2, py + ph // 2
        best = None
        for m in monitors[1:]:
            ml, mt = int(m["left"]), int(m["top"])
            mr, mb = ml + int(m["width"]), mt + int(m["height"])
            inside = ml <= cx <= mr and mt <= cy <= mb
            d = ((min(max(cx, ml), mr) - cx) ** 2
                 + (min(max(cy, mt), mb) - cy) ** 2)
            if best is None or (inside and not best[0]) or (
                    inside == best[0] and d < best[1]):
                best = (inside, d, (ml, mt, mr, mb))
        if best is not None:
            ml, mt, mr, mb = best[2]
            pw = min(pw, mr - ml)
            ph = min(ph, mb - mt)
            px = min(max(px, ml), mr - pw)
            py = min(max(py, mt), mb - ph)
        return (px, py, pw, ph)

    def paintEvent(self, ev) -> None:  # noqa: N802
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 110))
        if self._rect is not None:
            p.setPen(QtGui.QPen(QtGui.QColor(0, 220, 220), 2))
            p.drawRect(self._rect)
            p.fillRect(self._rect, QtGui.QColor(0, 180, 255, 40))

    def mousePressEvent(self, ev) -> None:  # noqa: N802
        if ev.button() == QtCore.Qt.LeftButton:
            self._origin = ev.pos()
            self._rect = QtCore.QRect(self._origin, QtCore.QSize())
            self.update()
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev) -> None:  # noqa: N802
        if self._origin is not None:
            self._rect = QtCore.QRect(self._origin, ev.pos()).normalized()
            self.update()
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev) -> None:  # noqa: N802
        if self._origin is not None:
            r = QtCore.QRect(self._origin, ev.pos()).normalized()
            if r.width() > 10 and r.height() > 10:
                g = self.geometry()
                logical = QtCore.QRect(int(r.x() + g.x()), int(r.y() + g.y()),
                                       int(r.width()), int(r.height()))
                scr = (QtWidgets.QApplication.screenAt(logical.center())
                       or QtWidgets.QApplication.primaryScreen())
                self.region = self._physical_region(logical, scr)
            self._origin = None
            self.update()
            self.accept()
        super().mouseReleaseEvent(ev)

    def keyPressEvent(self, ev) -> None:  # noqa: N802
        if ev.key() == QtCore.Qt.Key_Escape:
            self.reject()
            return
        super().keyPressEvent(ev)


class _ReplayCanvas(QtWidgets.QWidget):
    """回放画布：示意化重演圆球移动过程（目标/规划路径/轨迹/实时球位）。

    报告不含原始视频帧，绘制为归一化坐标图：绿=目标点，青=规划路径，
    橙=球心轨迹，白=当前帧全部粒子，黄圈=当前任务球。
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(260)
        self.setMaximumHeight(420)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #101418;")
        self._traj = None
        self._idx = -1
        self._bounds = None

    def set_trajectory(self, traj) -> None:
        self._traj = traj
        self._bounds = self._compute_bounds()
        self.set_index(0)

    def set_index(self, i: int) -> None:
        frames = (self._traj or {}).get("frames") or []
        self._idx = max(0, min(i, len(frames) - 1)) if frames else -1
        self.update()

    def _compute_bounds(self):
        pts = []
        t = self._traj or {}
        for info in (t.get("tasks") or {}).values():
            if info.get("goal"):
                pts.append(tuple(info["goal"]))
            for p in info.get("waypoints") or []:
                pts.append(tuple(p))
        for fr in t.get("frames") or []:
            for p in fr["particles"]:
                pts.append(tuple(p["position_px"]))
        for tr in (t.get("trails") or {}).values():
            for p in tr:
                pts.append(tuple(p))
        if not pts:
            return None
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        pad = max(10.0, (max(xs) - min(xs) + max(ys) - min(ys)) * 0.05)
        return (min(xs) - pad, min(ys) - pad,
                max(xs) + pad, max(ys) + pad)

    def _map(self, p, W, H):
        x0, y0, x1, y1 = self._bounds
        pad = 8
        s = min((W - 2 * pad) / max(1e-6, x1 - x0),
                (H - 2 * pad) / max(1e-6, y1 - y0))
        return (pad + (p[0] - x0) * s, pad + (p[1] - y0) * s, s)

    def paintEvent(self, ev) -> None:  # noqa: N802 - Qt 命名
        qp = QtGui.QPainter(self)
        W, H = self.width(), self.height()
        qp.fillRect(0, 0, W, H, QtGui.QColor("#101418"))
        t = self._traj
        frames = (t or {}).get("frames") or []
        if not t or self._idx < 0 or not frames:
            qp.setPen(QtGui.QColor("#8090a0"))
            qp.drawText(self.rect(), Qt.AlignCenter,
                        "暂无回放数据：先运行一轮避障/组装，再点『回放上一轮』")
            return
        fr = frames[self._idx]

        # 规划路径（青，虚线）
        pen = QtGui.QPen(QtGui.QColor("#40c0d0"), 1, Qt.DashLine)
        qp.setPen(pen)
        for info in (t.get("tasks") or {}).values():
            wps = info.get("waypoints") or []
            if len(wps) >= 2:
                prev = self._map(wps[0], W, H)
                for p in wps[1:]:
                    cur = self._map(p, W, H)
                    qp.drawLine(int(prev[0]), int(prev[1]),
                                int(cur[0]), int(cur[1]))
                    prev = cur
        # 目标点（绿十字+圈）
        qp.setPen(QtGui.QPen(QtGui.QColor("#40d060"), 2))
        for info in (t.get("tasks") or {}).values():
            g = info.get("goal")
            if not g:
                continue
            gx, gy, _ = self._map(g, W, H)
            qp.drawEllipse(QtCore.QPointF(gx, gy), 7, 7)
            qp.drawLine(int(gx) - 11, int(gy), int(gx) + 11, int(gy))
            qp.drawLine(int(gx), int(gy) - 11, int(gx), int(gy) + 11)
        # 球心轨迹（橙）
        qp.setPen(QtGui.QPen(QtGui.QColor("#f08030"), 1))
        for tr in (t.get("trails") or {}).values():
            for a, b in zip(tr, tr[1:]):
                pa, pb = self._map(a, W, H), self._map(b, W, H)
                qp.drawLine(int(pa[0]), int(pa[1]), int(pb[0]), int(pb[1]))
        # 当前帧粒子（白圆），当前任务球（黄圈高亮）
        cur_task = fr.get("task_id")
        task_info = (t.get("tasks") or {}).get(cur_task) or {}
        track = task_info.get("track_id")
        highlight = None
        for p in fr["particles"]:
            x, y, s = self._map(p["position_px"], W, H)
            r = max(3.0, p["radius_px"] * s * 0.5)
            qp.setPen(QtGui.QPen(QtGui.QColor("#e0e0e0"), 1))
            qp.setBrush(QtGui.QColor("#e0e0e0"))
            qp.drawEllipse(QtCore.QPointF(x, y), r, r)
            if (track is not None and p.get("track_id") == track) or \
                    (track is None and highlight is None):
                highlight = (x, y, r, p)
        if highlight is not None:
            x, y, r, _ = highlight
            qp.setBrush(Qt.NoBrush)
            qp.setPen(QtGui.QPen(QtGui.QColor("#f0e040"), 2))
            qp.drawEllipse(QtCore.QPointF(x, y), r + 4, r + 4)
        # 状态条
        qp.setPen(QtGui.QColor("#90a0b0"))
        qp.drawText(8, H - 8,
                    f"检测帧 {self._idx + 1}/{len(frames)}"
                    f"  最终状态: {t.get('final_state') or '?'}")


class _HeightDragHandle(QtWidgets.QFrame):
    """水平拖拽条：上下拖动可连续调整 target 控件的高度。

    target 按“固定高度”方式调整（同时设置 min/max），因此外层
    QScrollArea 的内容高度随之增长，纵向滚动条的滑块自动适配。
    """

    def __init__(self, target: QtWidgets.QWidget, min_h: int = 60,
                 max_h: int = 2000, parent=None) -> None:
        super().__init__(parent)
        self._target = target
        self._min_h = min_h
        self._max_h = max_h
        self._drag_y = 0
        self._drag_h = 0
        self.setFixedHeight(7)
        self.setCursor(QtCore.Qt.SizeVerCursor)
        self.setToolTip("按住并上下拖动，可调整上方内容框高度")
        self.setStyleSheet(
            "background-color: #2a333d; border-radius: 3px;")
        self._resized = False

    def mousePressEvent(self, event) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            self._drag_y = event.globalPos().y()
            self._drag_h = self._target.height()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if not (event.buttons() & QtCore.Qt.LeftButton):
            super().mouseMoveEvent(event)
            return
        new_h = self._drag_h + (event.globalPos().y() - self._drag_y)
        new_h = max(self._min_h, min(self._max_h, int(new_h)))
        self.set_target_height(new_h)
        event.accept()

    def set_target_height(self, height: int) -> None:
        """将目标控件高度固定为 height（首次调用后进入手动尺寸模式）。"""
        self._resized = True
        self._target.setMinimumHeight(height)
        self._target.setMaximumHeight(height)

    def is_user_sized(self) -> bool:
        return self._resized


class _MetricsCanvas(QtWidgets.QWidget):
    """Compact comparison chart for accumulated Alg1/Alg2 experiments."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.setMaximumHeight(360)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Preferred)
        self.setStyleSheet("background-color: #101418;")
        self._data = {}

    def set_data(self, data: dict) -> None:
        self._data = data or {}
        self.update()

    def paintEvent(self, ev) -> None:  # noqa: N802
        qp = QtGui.QPainter(self)
        W, H = self.width(), self.height()
        qp.fillRect(0, 0, W, H, QtGui.QColor("#101418"))
        groups = (self._data or {}).get("by_algorithm") or {}
        if not groups:
            qp.setPen(QtGui.QColor("#8090a0"))
            qp.drawText(self.rect(), Qt.AlignCenter, "暂无多次实验统计")
            return
        metrics = [("success_rate", "成功率"),
                   ("mean_path_efficiency", "路径效率"),
                   ("mean_uncertain_rate", "不确定率")]
        colors = ["#40d060", "#40c0d0", "#f0a040"]
        left, top, right, bottom = 52, 20, 12, 30
        chart_h = max(1, H - top - bottom)
        chart_w = max(1, W - left - right)
        qp.setPen(QtGui.QColor("#8090a0"))
        qp.drawLine(left, top, left, top + chart_h)
        qp.drawLine(left, top + chart_h, left + chart_w, top + chart_h)
        algs = list(groups)
        slot = chart_w / max(1, len(algs))
        for i, alg in enumerate(algs):
            x0 = left + i * slot + slot * 0.12
            bar_w = slot * 0.22
            qp.setPen(QtGui.QColor("#d0d8e0"))
            qp.drawText(int(left + i * slot), H - 10, alg)
            for j, (key, _label) in enumerate(metrics):
                value = groups[alg].get(key)
                value = max(0.0, min(1.0, float(value or 0.0)))
                x = x0 + j * bar_w * 1.2
                y = top + (1.0 - value) * chart_h
                qp.setBrush(QtGui.QColor(colors[j]))
                qp.setPen(Qt.NoPen)
                qp.drawRect(int(x), int(y), max(2, int(bar_w)),
                            max(1, int(top + chart_h - y)))
            qp.setPen(QtGui.QColor("#aab4c0"))
            qp.drawText(int(left + i * slot), 13,
                        f"{groups[alg].get('runs', 0)} runs")
        qp.setPen(QtGui.QColor("#aab4c0"))
        qp.drawText(6, top + 5, "1.0")
        qp.drawText(6, top + chart_h, "0.0")


# 电机模式『画框模式』-> 区域类型/命名前缀（衬底区域=手动可行域，单例）
MOTOR_ZONE_KINDS = {"目标区": "goal", "障碍区": "obstacle",
                    "自由区": "free", "衬底区域": "substrate"}
MOTOR_ZONE_PREFIX = {"goal": "goal_", "obstacle": "obs_", "free": "free_",
                     "substrate": "substrate_"}


class MainWindow(QtWidgets.QMainWindow):
    usb_count_ready = Signal(int)   # Picomotor USB 设备数（后台检测回填）
    controller_scan_ready = Signal(object)  # 控制器 ID/可用轴扫描结果
    kinesis_scan_ready = Signal(object)  # Kinesis serial/description records

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ObstacleAvoid - dryrun 仿真")
        # Keep enough room for the canvas and the parameter panel. Controls
        # are grouped into tabs, so smaller screens can still access them.
        self.resize(1360, 800)
        self.setMinimumSize(1180, 640)
        self.worker: Optional[WorkerThread] = None
        self._controller_ref: dict = {}
        self._ui_closing = False
        self._last_frame: Optional[np.ndarray] = None
        self.selected_ball_index: Optional[int] = None

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
        self.canvas.target_clicked.connect(self._on_target_clicked)
        layout.addWidget(self.canvas, 3)

        # 右：选项卡面板（运行控制 / XYZ 台位 / 检测与ROI / 日志与报告），
        # 运行状态常驻顶部；替代原先的单列长滚动面板。
        panel_widget = QtWidgets.QWidget()
        panel_widget.setMinimumWidth(420)
        panel = QtWidgets.QVBoxLayout(panel_widget)
        panel.setContentsMargins(8, 8, 8, 8)
        panel.setSpacing(6)

        self.state_label = QtWidgets.QLabel("IDLE")
        _st_font = self.state_label.font()
        _st_font.setBold(True)
        self.state_label.setFont(_st_font)
        panel.addWidget(self.state_label)
        self.detail_label = QtWidgets.QLabel("")
        self.detail_label.setWordWrap(True)
        self.detail_label.setMaximumHeight(48)
        panel.addWidget(self.detail_label)

        self.tabs = QtWidgets.QTabWidget()

        self._tab_pages = {}
        self._tab_scrolls = {}

        def _tab_page(name: str = "", scroll: bool = False) -> QtWidgets.QVBoxLayout:
            if not name:
                name = ("run", "xyz", "roi", "log")[len(self._tab_pages)]
                scroll = name == "log"
            page = QtWidgets.QWidget()
            outer = QtWidgets.QVBoxLayout(page)
            outer.setContentsMargins(6, 6, 6, 6)
            outer.setSpacing(0)
            if scroll:
                content = QtWidgets.QWidget()
                vbox = QtWidgets.QVBoxLayout(content)
                vbox.setContentsMargins(4, 4, 10, 8)
                vbox.setSpacing(8)
                area = QtWidgets.QScrollArea()
                area.setObjectName(f"{name}ScrollArea")
                area.setWidgetResizable(True)
                area.setFrameShape(QtWidgets.QFrame.NoFrame)
                area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
                area.setWidget(content)
                outer.addWidget(area)
                self._tab_scrolls[name] = area
            else:
                vbox = QtWidgets.QVBoxLayout()
                vbox.setContentsMargins(6, 6, 6, 6)
                vbox.setSpacing(6)
                outer.addLayout(vbox)
            self.tabs.addTab(page, "")
            self._tab_pages[name] = page
            return vbox

        # 日志/报告页内容较长，明确放入垂直滚动区域；其它页保持紧凑布局。
        self._p_run = _tab_page("run", scroll=False)   # 运行控制
        self._p_xyz = _tab_page("xyz", scroll=False)   # XYZ 台位
        self._p_roi = _tab_page("roi", scroll=False)   # 检测 / ROI
        self._p_log = _tab_page("log", scroll=True)    # 日志 / 报告
        self.tabs.setTabText(0, "运行控制")
        self.tabs.setTabText(1, "XYZ 台位")
        self.tabs.setTabText(2, "检测 / ROI")
        self.tabs.setTabText(3, "日志 / 报告")
        panel.addWidget(self.tabs, 1)
        layout.addWidget(panel_widget, 2)

        self._p_run.addWidget(QtWidgets.QLabel("场景"))
        self.scenario_combo = QtWidgets.QComboBox()
        # 仅保留 sim01 仿真模拟模式（其余场景仍可经 CLI 使用）
        self.scenario_combo.addItems(["sim01"])
        self._p_run.addWidget(self.scenario_combo)

        # ---- 算法选择：打包当前控制算法（避障+组装）为 Alg1，
        # 后续新算法在此注册即可与 Alg1 区分
        self.ALGORITHMS = {
            "Alg1": "避障+组装（样品/小球直接仿真）",
            "Alg2": "固定光束+XYZ 位移台（显微镜仿真）",
        }
        self._p_run.addWidget(QtWidgets.QLabel("算法"))
        self.alg_combo = QtWidgets.QComboBox()
        self.alg_combo.addItems(list(self.ALGORITHMS.keys()))
        self.alg_combo.setToolTip(
            "; ".join(f"{k}: {v}" for k, v in self.ALGORITHMS.items()))
        self._p_run.addWidget(self.alg_combo)

        self.mode_sel = QtWidgets.QComboBox()
        self.mode_sel.addItems(["虚拟模式 (仿真)", "电机模式 (真实相机+电机)"])
        self.mode_sel.currentIndexChanged.connect(self.on_mode_changed)
        self._p_run.addWidget(self.mode_sel)

        run_row = QtWidgets.QHBoxLayout()
        self.run_oa_btn = QtWidgets.QPushButton("避障运行")
        self.run_oa_btn.clicked.connect(lambda: self.on_run(mode="oa"))
        run_row.addWidget(self.run_oa_btn)
        self.run_ag_btn = QtWidgets.QPushButton("组装运行")
        self.run_ag_btn.clicked.connect(lambda: self.on_run(mode="ag"))
        run_row.addWidget(self.run_ag_btn)
        self._p_run.addLayout(run_row)

        # 电机模式设置
        self.motor_box = QtWidgets.QGroupBox("电机模式设置")
        form = QtWidgets.QFormLayout(self.motor_box)
        self.driver_combo = QtWidgets.QComboBox()
        self.driver_combo.addItems(["8742/8743 Picomotor (USB)",
                                    "串口 G 代码位移台",
                                    "Kinesis KIM101 (Thorlabs)"])
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
        self.controller_scan_btn = QtWidgets.QPushButton("检测并映射控制器")
        self.controller_scan_btn.setToolTip(
            "扫描 USB 8742/8743，读取控制器 ID 和可用轴，并自动填入 X/Y/Z 轴号")
        self.controller_scan_btn.clicked.connect(self._refresh_usb_count)
        form.addRow(self.controller_scan_btn)
        self.pico_widgets.append(self.controller_scan_btn)
        self.controller_id_label = QtWidgets.QLabel("控制器: 未检测")
        self.controller_id_label.setStyleSheet("color:#888;")
        form.addRow(self.controller_id_label)
        self.pico_widgets.append(self.controller_id_label)
        self.axis_x_spin = QtWidgets.QSpinBox()
        self.axis_x_spin.setRange(1, 4)
        self.axis_x_spin.setValue(1)
        form.addRow("X 轴号", self.axis_x_spin)
        self.axis_y_spin = QtWidgets.QSpinBox()
        self.axis_y_spin.setRange(1, 4)
        self.axis_y_spin.setValue(2)
        form.addRow("Y 轴号", self.axis_y_spin)
        self.axis_z_spin = QtWidgets.QSpinBox()
        self.axis_z_spin.setRange(1, 4)
        self.axis_z_spin.setValue(3)
        form.addRow("Z 轴号", self.axis_z_spin)
        self.pico_widgets += [self.axis_x_spin, self.axis_y_spin,
                              self.axis_z_spin]
        for _spin in (self.conn_spin, self.axis_x_spin, self.axis_y_spin,
                      self.axis_z_spin):
            _spin.valueChanged.connect(lambda _value: self._close_motor_xyz_stage())
        # -- 步进类驱动（8742/8743 与 Kinesis）共用的运动参数
        # 驱动本身按 step 计量，因此这里统一用 step + steps/mm 换算成 mm
        self.motion_widgets = []
        self.spm_spin = QtWidgets.QDoubleSpinBox()
        self.spm_spin.setRange(1.0, 200000.0)
        self.spm_spin.setDecimals(1)
        self.spm_spin.setValue(1000.0)
        self.spm_spin.setToolTip("每毫米步数：须按实际位移台标定\n"
                                 "(MTM 平台 ~30nm/步 ≈ 33333 steps/mm)")
        form.addRow("steps/mm", self.spm_spin)
        self.motion_widgets.append(self.spm_spin)
        self.step_steps_spin = QtWidgets.QSpinBox()
        self.step_steps_spin.setRange(1, 500000)
        self.step_steps_spin.setValue(300)
        self.step_steps_spin.setToolTip(
            "单步位移（step）：闭环每步向台位下发的最大步数，\n"
            "同时作为驱动侧单步限位（按 steps/mm 换算成 mm 后下发）")
        form.addRow("单步位移(step)", self.step_steps_spin)
        self.motion_widgets.append(self.step_steps_spin)
        self.step_mm_hint = QtWidgets.QLabel()
        self.step_mm_hint.setStyleSheet("color:#888;")
        form.addRow(self.step_mm_hint)
        self.motion_widgets.append(self.step_mm_hint)
        self.speed_spin = QtWidgets.QSpinBox()
        self.speed_spin.setRange(0, 5000)
        self.speed_spin.setValue(0)
        self.speed_spin.setToolTip(
            "位移速度（step/s）：写入控制器步率\n"
            "（KIM101 StepRate / 8742 velocity）；0 = 不修改控制器当前值")
        form.addRow("位移速度(step/s)", self.speed_spin)
        self.motion_widgets.append(self.speed_spin)
        self.accel_steps_spin = QtWidgets.QSpinBox()
        self.accel_steps_spin.setRange(0, 1000000)
        self.accel_steps_spin.setValue(0)
        self.accel_steps_spin.setToolTip(
            "位移加速度（step/s²）：写入控制器步加速度\n"
            "（KIM101 StepAcceleration / 8742 accel）；0 = 不修改")
        form.addRow("位移加速度(step/s²)", self.accel_steps_spin)
        self.motion_widgets.append(self.accel_steps_spin)
        for _spin in (self.spm_spin, self.step_steps_spin):
            _spin.valueChanged.connect(lambda _v: self._update_step_mm_hint())
        self._update_step_mm_hint()
        # -- 串口 G 代码参数（G 代码台位以 mm 计，故单步位移仍用 mm）
        self.port_edit = QtWidgets.QLineEdit("COM3")
        form.addRow("串口", self.port_edit)
        self.baud_spin = QtWidgets.QSpinBox()
        self.baud_spin.setRange(9600, 256000)
        self.baud_spin.setValue(115200)
        form.addRow("波特率", self.baud_spin)
        self.step_mm_spin = QtWidgets.QDoubleSpinBox()
        self.step_mm_spin.setRange(0.0001, 5.0)
        self.step_mm_spin.setDecimals(4)
        self.step_mm_spin.setValue(0.30)
        self.step_mm_spin.setToolTip(
            "单步最大位移 (mm)：G 代码台位每步走这一距离，同时作为单步限位")
        form.addRow("单步位移(mm)", self.step_mm_spin)
        self.serial_widgets = [self.port_edit, self.baud_spin,
                               self.step_mm_spin]
        # -- Thorlabs Kinesis KIM101 参数：一个控制器只使用一个序列号
        self.kinesis_widgets = []
        self.kinesis_serial_edit = QtWidgets.QLineEdit()
        self.kinesis_serial_edit.setPlaceholderText("KIM101 控制器序列号")
        form.addRow("Kinesis 序列号", self.kinesis_serial_edit)
        self.kinesis_widgets.append(self.kinesis_serial_edit)
        self.kinesis_scan_btn = QtWidgets.QPushButton("检测 Kinesis 控制器")
        self.kinesis_scan_btn.setToolTip(
            "调用 Kinesis DeviceManagerCLI，读取已连接 KIM101 序列号")
        self.kinesis_scan_btn.clicked.connect(self._refresh_kinesis_devices)
        form.addRow(self.kinesis_scan_btn)
        self.kinesis_widgets.append(self.kinesis_scan_btn)
        self.kinesis_id_label = QtWidgets.QLabel("Kinesis: 未检测")
        self.kinesis_id_label.setStyleSheet("color:#888;")
        form.addRow(self.kinesis_id_label)
        self.kinesis_widgets.append(self.kinesis_id_label)
        # -- 公共参数
        self.iters_spin = QtWidgets.QSpinBox()
        self.iters_spin.setRange(10, 20000)
        self.iters_spin.setValue(3000)
        self.iters_spin.setToolTip(
            "最大步数：从圆球到目标点的运动步数上限，超过即中止并报未完成")
        form.addRow("最大步数", self.iters_spin)
        self.src_combo = QtWidgets.QComboBox()
        self.src_combo.addItems(["相机", "屏幕区域"])
        self.src_combo.setToolTip(
            "帧源：USB 相机；或截取某个显示屏的一部分区域\n"
            "（如仿真软件画面）。框选坐标自动按系统缩放换算")
        form.addRow("帧源", self.src_combo)
        self.cam_spin = QtWidgets.QSpinBox()
        self.cam_spin.setRange(0, 8)
        form.addRow("相机索引", self.cam_spin)
        self.select_region_btn = QtWidgets.QPushButton("框选屏幕区域...")
        self.select_region_btn.clicked.connect(self.on_select_screen_region)
        self.region_label = QtWidgets.QLabel("未选择")
        self.region_label.setStyleSheet("color:#888;")
        form.addRow(self.select_region_btn, self.region_label)
        self.screen_widgets = [self.select_region_btn, self.region_label]
        self.src_combo.currentIndexChanged.connect(self.on_frame_source_changed)
        self.cam_spin.valueChanged.connect(lambda _: self._invalidate_live())
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
        for _w in self.screen_widgets:   # 默认帧源=相机，屏幕区域控件隐藏
            _w.setVisible(False)
        # 参数控件按当前驱动（默认 8742/8743）初始化可见性
        self._apply_driver_visibility(self.driver_combo.currentIndex())
        self._p_run.addWidget(self.motor_box)

        self.pause_btn = QtWidgets.QPushButton("暂停")
        self.pause_btn.clicked.connect(self.on_pause)
        self._p_run.addWidget(self.pause_btn)

        self.estop_btn = QtWidgets.QPushButton("急停")
        self.estop_btn.setStyleSheet("background:#c0392b;color:white;")
        self.estop_btn.clicked.connect(self.on_estop)
        self._p_run.addWidget(self.estop_btn)

        self.xyz_box = QtWidgets.QGroupBox("XYZ stage (um)")
        # 选项卡布局下面板独占一页：垂直不扩展，保持紧凑（不被拉伸撑满）
        self.xyz_box.setSizePolicy(QtWidgets.QSizePolicy.Preferred,
                                   QtWidgets.QSizePolicy.Maximum)
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
            row.addStretch(1)
            return row

        axis_label = QtWidgets.QLabel("Axis")
        axis_label.setMinimumWidth(34)
        step_label = QtWidgets.QLabel("Step (um)")
        step_label.setMinimumWidth(58)
        xyz.addLayout(_row(axis_label, self.xyz_axis_combo,
                           step_label, self.xyz_step_spin))
        xyz.addLayout(_row(self.xyz_jog_minus, self.xyz_jog_plus))

        steps_label = QtWidgets.QLabel("Steps/unit")
        steps_label.setMinimumWidth(67)
        speed_label = QtWidgets.QLabel("Max speed")
        speed_label.setMinimumWidth(61)
        xyz.addLayout(_row(steps_label, self.xyz_steps_spin,
                           speed_label, self.xyz_speed_spin))
        xyz.addLayout(_row(accel_label := QtWidgets.QLabel("Accel"),
                           self.xyz_accel_spin, self.xyz_apply_btn))

        # 刻度线开关：画布左上角原点的 x/y 刻度尺（实时生效）
        self.xyz_ruler_chk = QtWidgets.QCheckBox("刻度线")
        self.xyz_ruler_chk.setChecked(True)
        self.xyz_ruler_chk.toggled.connect(lambda _on: self._refresh_sim_preview())
        # 锁定视角：禁用鼠标拖拽控制位移台（防止移动过程中误触视角）
        self.view_lock_chk = QtWidgets.QCheckBox("锁定视角")
        self.view_lock_chk.setChecked(True)   # 默认锁定，防止误触视角
        self.view_lock_chk.setToolTip(
            "锁定后禁用鼠标拖拽控制位移台，防止移动/运行过程中误触视角。")
        self.view_lock_chk.toggled.connect(self.on_view_lock_toggled)
        xyz.addLayout(_row(self.xyz_home_btn, self.xyz_zero_btn,
                           self.xyz_enable_btn, self.xyz_stop_btn,
                           self.xyz_ruler_chk, self.view_lock_chk))

        # ---- 圆球移动参数（避障/组装运行：球 -> 目标点）
        self.ball_step_spin = QtWidgets.QDoubleSpinBox()
        self.ball_step_spin.setRange(0.0001, 5.0)
        self.ball_step_spin.setDecimals(4)
        self.ball_step_spin.setValue(0.01)   # 10 um（默认球步长）
        self.ball_step_spin.setSuffix(" mm")
        self.ball_step_spin.setToolTip("圆球每次移动的步长（单步位移）；0.01mm = 10um")
        self.ball_speed_spin = QtWidgets.QDoubleSpinBox()
        self.ball_speed_spin.setRange(0.01, 1000000.0)
        self.ball_speed_spin.setDecimals(2)
        self.ball_speed_spin.setValue(0.1)
        self.ball_speed_spin.setSuffix(" um/s")
        self.ball_speed_spin.setToolTip("圆球移动速度（台位 X/Y 轴最大速度）")
        self.ball_accel_spin = QtWidgets.QDoubleSpinBox()
        self.ball_accel_spin.setRange(0.01, 1000000.0)
        self.ball_accel_spin.setDecimals(2)
        self.ball_accel_spin.setValue(0.1)
        self.ball_accel_spin.setSuffix(" um/s2")
        self.ball_accel_spin.setToolTip("圆球移动加速度（台位 X/Y 轴加速度）")
        for spin in (self.ball_step_spin, self.ball_speed_spin,
                     self.ball_accel_spin):
            spin.setFixedWidth(92)
        xyz.addLayout(_row(QtWidgets.QLabel("球步长"), self.ball_step_spin,
                           QtWidgets.QLabel("速度"), self.ball_speed_spin))
        xyz.addLayout(_row(QtWidgets.QLabel("加速度"), self.ball_accel_spin))
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
        self._p_xyz.addWidget(self.xyz_box)

        # ---- 实时检测 + ROI/区域划分（固定镜头）
        self._p_roi.addWidget(QtWidgets.QLabel("—— 实时检测 / 区域划分 ——"))
        self.live_btn = QtWidgets.QPushButton("开始实时检测")
        self.live_btn.setCheckable(True)
        self.live_btn.toggled.connect(self.on_live_toggled)
        self._p_roi.addWidget(self.live_btn)

        self.auto_recognition_box = QtWidgets.QGroupBox("Alg2 自动识别")
        auto_layout = QtWidgets.QVBoxLayout(self.auto_recognition_box)
        self.substrate_mode_label = QtWidgets.QLabel(
            "衬底：仅由『衬底区域』手动框定（长方形/Free），未标注时用 ROI 内缩；"
            "障碍：仅由『障碍区』手动框定")
        self.substrate_mode_label.setStyleSheet("color:#555;")
        auto_layout.addWidget(self.substrate_mode_label)
        self.auto_recognition_chk = QtWidgets.QCheckBox("自动识别圆球 / 光斑")
        self.auto_recognition_chk.setToolTip(
            "选择 Alg2 后默认启用；点击“开始实时检测”处理相机或屏幕帧。")
        self.auto_recognition_chk.toggled.connect(
            self.on_auto_recognition_toggled)
        auto_layout.addWidget(self.auto_recognition_chk)
        auto_conf_row = QtWidgets.QHBoxLayout()
        auto_conf_row.addWidget(QtWidgets.QLabel("最低置信度"))
        self.auto_confidence_spin = QtWidgets.QDoubleSpinBox()
        self.auto_confidence_spin.setRange(0.10, 0.99)
        self.auto_confidence_spin.setSingleStep(0.05)
        self.auto_confidence_spin.setDecimals(2)
        self.auto_confidence_spin.setValue(0.35)
        self.auto_confidence_spin.valueChanged.connect(
            self.on_auto_confidence_changed)
        auto_conf_row.addWidget(self.auto_confidence_spin)
        self.auto_confirm_btn = QtWidgets.QPushButton("确认识别结果")
        self.auto_confirm_btn.setEnabled(False)
        self.auto_confirm_btn.clicked.connect(self.on_confirm_auto_recognition)
        auto_conf_row.addWidget(self.auto_confirm_btn)
        auto_layout.addLayout(auto_conf_row)
        beam_row = QtWidgets.QHBoxLayout()
        beam_row.addWidget(QtWidgets.QLabel("激光位置"))
        self.beam_x_spin = QtWidgets.QDoubleSpinBox()
        self.beam_y_spin = QtWidgets.QDoubleSpinBox()
        for spin in (self.beam_x_spin, self.beam_y_spin):
            spin.setRange(0.0, 10000.0)
            spin.setDecimals(1)
            spin.setSuffix(" px")
        self.beam_x_spin.setPrefix("X ")
        self.beam_y_spin.setPrefix("Y ")
        self.beam_x_spin.setValue(400.0)
        self.beam_y_spin.setValue(300.0)
        # 手动改光斑坐标后立即重绘当前帧（标定后光斑跟随输入移动）
        self.beam_x_spin.valueChanged.connect(self._refresh_beam_spot)
        self.beam_y_spin.valueChanged.connect(self._refresh_beam_spot)
        beam_row.addWidget(self.beam_x_spin)
        beam_row.addWidget(self.beam_y_spin)
        self.beam_center_btn = QtWidgets.QPushButton("设为画面中心")
        self.beam_center_btn.clicked.connect(self.on_calibrate_beam_center)
        beam_row.addWidget(self.beam_center_btn)
        self.beam_pick_btn = QtWidgets.QPushButton("点击画面标定")
        self.beam_pick_btn.setCheckable(True)
        self.beam_pick_btn.setToolTip("在当前帧点击固定光斑中心，保存为 Alg2 光斑坐标")
        self.beam_pick_btn.toggled.connect(self._on_beam_pick_toggled)
        beam_row.addWidget(self.beam_pick_btn)
        auto_layout.addLayout(beam_row)
        self.beam_status_label = QtWidgets.QLabel("激光位置未标定")
        self.beam_status_label.setStyleSheet("color:#b36b00;")
        auto_layout.addWidget(self.beam_status_label)
        self.auto_status_label = QtWidgets.QLabel("选择 Alg2 后开始实时检测")
        self.auto_status_label.setWordWrap(True)
        self.auto_status_label.setStyleSheet("color:#888;")
        auto_layout.addWidget(self.auto_status_label)
        self._p_run.addWidget(self.auto_recognition_box)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("画框模式"))
        self.mode_combo = QtWidgets.QComboBox()
        self._switch_draw_modes(False)   # 默认虚拟模式：sim 对象绘制选项
        row.addWidget(self.mode_combo, 1)
        # 画框闸门：默认开启，拖拽画框立即生效
        self.draw_gate_btn = QtWidgets.QPushButton("画框:开")
        self.draw_gate_btn.setCheckable(True)
        self.draw_gate_btn.setChecked(True)
        self.draw_gate_btn.setFixedWidth(76)
        self.draw_gate_btn.setToolTip(
            "画框开关：开启后拖拽画框才生效。默认开启，"
            "移动/运行过程中可点击锁定以防误触画框。")
        self.draw_gate_btn.toggled.connect(self.on_draw_gate_toggled)
        row.addWidget(self.draw_gate_btn)
        self.canvas.drawing_enabled = True   # 初始开启
        self._p_roi.addLayout(row)

        # 框定形状：拖拽框按所选形状归一化（圆/正方形取中心+短边）
        sh_row = QtWidgets.QHBoxLayout()
        sh_row.addWidget(QtWidgets.QLabel("框定形状"))
        self.shape_combo = QtWidgets.QComboBox()
        self.shape_combo.addItems(["长方形", "正方形", "圆形", "Free"])
        self.shape_combo.setToolTip(
            "拖拽/点击画框的形状：长方形=原样；正方形/圆形取拖拽框中心、"
            "短边为边长（圆形存储为外接正方形，构建障碍时生成圆）；"
            "Free=自由多边形：逐点左键点击，双击/右键/点击起点闭合"
            "（电机模式区域存真实多边形；虚拟模式对象取外接矩形）。")
        self.shape_combo.currentIndexChanged.connect(
            self._sync_polygon_mode)
        sh_row.addWidget(self.shape_combo, 1)
        self._p_roi.addLayout(sh_row)
        self.canvas.points_drawn.connect(self.on_points_drawn)

        row2 = QtWidgets.QHBoxLayout()
        self.reset_view_btn = QtWidgets.QPushButton("复位视野")
        self.reset_view_btn.clicked.connect(self.on_reset_view)
        row2.addWidget(self.reset_view_btn)
        self.undo_zone_btn = QtWidgets.QPushButton("撤销区域")
        self.undo_zone_btn.clicked.connect(self.on_undo_zone)
        row2.addWidget(self.undo_zone_btn)
        self._p_roi.addLayout(row2)

        row4 = QtWidgets.QHBoxLayout()
        row4.addWidget(QtWidgets.QLabel("边界间隙(px)"))
        self.edge_spin = QtWidgets.QSpinBox()
        self.edge_spin.setRange(1, 200)
        self.edge_spin.setValue(1)
        self.edge_spin.setToolTip(
            "球心距视野边界的最小允许间隙(实际下限4px)；障碍碰撞不受影响。"
            "调小可让贴边球作为起点，红带随之变窄。")
        self.edge_spin.valueChanged.connect(self.on_edge_changed)
        row4.addWidget(self.edge_spin)
        self._p_roi.addLayout(row4)

        row3 = QtWidgets.QHBoxLayout()
        self.save_cfg_btn = QtWidgets.QPushButton("保存配置")
        self.save_cfg_btn.clicked.connect(self.on_save_config)
        row3.addWidget(self.save_cfg_btn)
        self.load_cfg_btn = QtWidgets.QPushButton("载入配置")
        self.load_cfg_btn.clicked.connect(self.on_load_config)
        row3.addWidget(self.load_cfg_btn)
        self._p_roi.addLayout(row3)
        self.zones_label = QtWidgets.QLabel("zones: 0")
        self._p_roi.addWidget(self.zones_label)

        # ---- 日志/报告页 -------------------------------------------------
        # 每个区域单独成组，避免固定高度控件挤在同一个布局中。外层日志
        # 页由 QScrollArea 承载，因此窄窗口也能完整访问下方内容。
        def _report_group(title: str):
            group = QtWidgets.QGroupBox(title)
            group.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                QtWidgets.QSizePolicy.Preferred)
            group.setStyleSheet(
                "QGroupBox {"
                "  font-weight: 600;"
                "  border: 1px solid #3a4652;"
                "  border-radius: 5px;"
                "  margin-top: 8px;"
                "  padding-top: 8px;"
                "}"
                "QGroupBox::title {"
                "  subcontrol-origin: margin;"
                "  left: 8px;"
                "  padding: 0 4px;"
                "}"
            )
            body = QtWidgets.QVBoxLayout(group)
            body.setContentsMargins(8, 8, 8, 8)
            body.setSpacing(6)
            self._p_log.addWidget(group)
            return group, body

        self._props_group, props_layout = _report_group("对象属性（实时）")
        self.props_out = QtWidgets.QPlainTextEdit()
        self.props_out.setReadOnly(True)
        self.props_out.setMaximumBlockCount(400)
        self.props_out.setFont(QtGui.QFont("Consolas", 8))
        self.props_out.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.props_out.setMinimumHeight(80)
        self.props_out.setMaximumHeight(180)
        self.props_out.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                     QtWidgets.QSizePolicy.Preferred)
        props_layout.addWidget(self.props_out)

        self._log_group, log_layout = _report_group("操作日志")
        self.log_out = QtWidgets.QPlainTextEdit()
        self.log_out.setReadOnly(True)
        self.log_out.setMaximumBlockCount(500)
        self.log_out.setFont(QtGui.QFont("Consolas", 8))
        self.log_out.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.log_out.setMinimumHeight(90)
        # 不设最大高度：由下方拖拽条手动拉长，框内滚动条自适应
        self.log_out.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        log_layout.addWidget(self.log_out)
        # 拖拽条：上下拖动调整日志框高度；外层滚动区域随之增长
        self.log_resize_handle = _HeightDragHandle(
            self.log_out, min_h=90, max_h=2000)
        log_layout.addWidget(self.log_resize_handle)
        log_ctl = QtWidgets.QHBoxLayout()
        log_ctl.addStretch(1)
        self.clear_log_btn = QtWidgets.QPushButton("清空日志")
        self.clear_log_btn.setToolTip("清除操作日志的全部内容（不影响报告回放）")
        self.clear_log_btn.clicked.connect(self.on_clear_log)
        log_ctl.addWidget(self.clear_log_btn)
        log_layout.addLayout(log_ctl)

        self._replay_group, replay_layout = _report_group("报告回放（JSONL）")
        rp_row = QtWidgets.QHBoxLayout()
        self.report_path = QtWidgets.QLineEdit()
        self.report_path.setPlaceholderText("Reports/run_*.jsonl（运行后自动填入）")
        rp_row.addWidget(self.report_path, 1)
        self.replay_last_btn = QtWidgets.QPushButton("回放上一轮")
        self.replay_last_btn.setToolTip("自动加载 Reports/ 下最新一轮运行的报告并回放")
        self.replay_last_btn.clicked.connect(self.on_replay_last)
        rp_row.addWidget(self.replay_last_btn)
        self.replay_btn = QtWidgets.QPushButton("回放")
        self.replay_btn.clicked.connect(self.on_replay)
        rp_row.addWidget(self.replay_btn)
        replay_layout.addLayout(rp_row)

        # 回放动画：圆球移动完整过程（目标/路径/轨迹/球位）
        self.replay_canvas = _ReplayCanvas()
        self.replay_canvas.setMinimumHeight(260)
        self.replay_canvas.setMaximumHeight(460)
        replay_layout.addWidget(self.replay_canvas)
        rp_ctl = QtWidgets.QHBoxLayout()
        self.rp_play_btn = QtWidgets.QPushButton("播放")
        self.rp_play_btn.setEnabled(False)
        self.rp_play_btn.clicked.connect(self.on_replay_play)
        rp_ctl.addWidget(self.rp_play_btn)
        self.rp_prev_btn = QtWidgets.QPushButton("上一步")
        self.rp_prev_btn.setEnabled(False)
        self.rp_prev_btn.clicked.connect(lambda: self._rp_set_index(
            self.rp_slider.value() - 1))
        rp_ctl.addWidget(self.rp_prev_btn)
        self.rp_next_btn = QtWidgets.QPushButton("下一步")
        self.rp_next_btn.setEnabled(False)
        self.rp_next_btn.clicked.connect(lambda: self._rp_set_index(
            self.rp_slider.value() + 1))
        rp_ctl.addWidget(self.rp_next_btn)
        self.rp_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.rp_slider.setRange(0, 0)
        self.rp_slider.valueChanged.connect(self._rp_set_index)
        rp_ctl.addWidget(self.rp_slider, 1)
        self.rp_speed = QtWidgets.QComboBox()
        self.rp_speed.addItems(["0.5x", "1x", "2x", "4x"])
        self.rp_speed.setCurrentIndex(1)
        self.rp_label = QtWidgets.QLabel("无回放数据")
        replay_layout.addLayout(rp_ctl)

        # 速度和状态单独放在第二行，避免窄窗口下与三个控制按钮争抢宽度。
        rp_info = QtWidgets.QHBoxLayout()
        rp_info.addWidget(QtWidgets.QLabel("速度"))
        rp_info.addWidget(self.rp_speed)
        rp_info.addStretch(1)
        rp_info.addWidget(self.rp_label)
        replay_layout.addLayout(rp_info)

        self.replay_out = QtWidgets.QPlainTextEdit()
        self.replay_out.setReadOnly(True)
        self.replay_out.setFont(QtGui.QFont("Consolas", 8))
        self.replay_out.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.replay_out.setMinimumHeight(72)
        self.replay_out.setMaximumHeight(160)
        self.replay_out.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                      QtWidgets.QSizePolicy.Preferred)
        replay_layout.addWidget(self.replay_out)

        self._comparison_group, comparison_layout = _report_group(
            "实验统计与算法对比")
        self.experiment_stats_out = QtWidgets.QPlainTextEdit()
        self.experiment_stats_out.setReadOnly(True)
        self.experiment_stats_out.setFont(QtGui.QFont("Consolas", 8))
        self.experiment_stats_out.setLineWrapMode(
            QtWidgets.QPlainTextEdit.NoWrap)
        self.experiment_stats_out.setMinimumHeight(90)
        self.experiment_stats_out.setMaximumHeight(220)
        self.experiment_stats_out.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Preferred)
        comparison_layout.addWidget(self.experiment_stats_out)
        self.experiment_canvas = _MetricsCanvas()
        self.experiment_canvas.setMinimumHeight(190)
        self.experiment_canvas.setMaximumHeight(360)
        comparison_layout.addWidget(self.experiment_canvas)
        self._p_log.addStretch(1)

        # 仿真镜头图层配置（mask/ground/obstacle 数量/标签/形状）——直接内嵌
        # 在"检测 / ROI"界面下方，替代原来的弹窗入口，无需打开对话框即可查看。
        # 赋值给 sim_spec_btn（现为 QGroupBox）以沿用模式切换的可见性控制。
        self.sim_spec_btn = self._build_sim_spec_panel()
        self.sim_spec_btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                        QtWidgets.QSizePolicy.Maximum)
        self._p_roi.addWidget(self.sim_spec_btn)

        # 各选项卡内容顶部对齐（日志页由 replay_out 撑满，无需 stretch）
        self._p_run.addStretch(1)
        self._p_xyz.addStretch(1)
        self._p_roi.addStretch(1)

        # sim01 运行状态（ROI 布局 / 预览帧 / 实时检测世界）
        # grounds: 衬底对象列表，每个 ground 内嵌 goal(避障) / goal_range(组装)
        self._sim_cfg = {
            "grounds": [],                # [{id, rect, poly?, goal, goal_range}]
            "balls": [],                  # [[x,y,w,h], ...]
            "obstacles": [],              # [[x,y,w,h], ...]（rect 为外接框）
            "obstacle_polys": []          # 与 obstacles 平行；[[[x,y],..]|None]
        }                                 # Free 多边形顶点（None=矩形对象）
        self._sim_ids = {"balls": [], "obstacles": []}   # 平行 ID 列表（需求4）
        self._sim_uid = 0                # 全局单调计数器，ID 永不复用
        self._roi_cfg = None             # 提前初始化（_load_sim_config 会刷新属性面板）
        self._live_dets = []             # 提前初始化（同上）
        self._auto_result = None         # 提前初始化（属性面板会读取）
        self._auto_confirmed_result = None
        self._auto_good_streak = 0
        self._auto_recognition_confirmed = False
        self._beam_calibrated = False
        self._beam_calibration_armed = False
        self._selected_auto_track_id: Optional[int] = None
        self._rp_traj = None             # 回放数据（extract_trajectory 结果）
        self._rp_timer = None            # 回放动画定时器
        self._sim_order: list = []       # 画框顺序栈（undo 后画先撤）
        self._sim_preview_frame = None   # sim01 预览原始帧（叠加 ROI 用）
        self._sim_live = None            # sim01 实时检测世界（SimMicroscopeWorld）
        self._pending_view_center = None  # 需求1：待对齐的视窗中心（样本坐标）
        self._final_stage_position = None  # 需求3：本轮运行最终台位（µm）
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
        self._auto_detect: Optional[AutoRecognizeThread] = None
        self._live_dets = []
        self._auto_result = None
        self._auto_confirmed_result = None
        self._auto_good_streak = 0
        self._auto_recognition_confirmed = False
        # 区域框是"样品绑定"的：记录画框时刻的实测位移，运行/移动期间按
        # 实测位移增量平移，框才不会脱离真实衬底/障碍
        self._zone_ref_shift: dict = {}
        self._run_plan_pts: list = []   # 电机模式运行期最新规划路径（实时叠加）
        self._beam_calibrated = False
        self._beam_calibration_armed = False
        self._selected_auto_track_id = None
        self._tick_n = 0
        self._roi_cfg = None       # RoiConfig（zones 用视频绝对坐标）
        self._view_map = None      # (scale, ox, oy) 帧->画布映射
        self._preview_worker: Optional[PreviewThread] = None
        self._pending_scenario: Optional[str] = None
        self._sim_drag_acc = (0, 0)  # 位移台拖拽累计增量 px
        self._xyz_stage = None
        self._motor_xyz_stage = None
        self._controller_scan = []
        self._selected_controller = None
        self._kinesis_devices = []
        self._xyz_axis_profiles = {}
        self._cam_frame_size = None  # 电机模式帧源全画幅 (w,h)，ROI 校验用
        self._screen_region = None   # 屏幕区域帧源 (x,y,w,h)，虚拟桌面全局坐标
        self._live_src_kind = None   # 当前电机模式帧源类型 'camera'/'screen'
        self._live_src_region = None  # 构建当前 live 世界时所用屏幕区域

        # 启动即后台预热 YOLO 权重（3-5s），期间 UI 完全可交互
        warmup_shared_detector_async(
            video_sim.WEIGHTS,
            on_error=lambda exc: self.detail_label.setText(
                f"YOLO 权重预热失败: {exc}"))
        self.usb_count_ready.connect(
            lambda n: self.pico_count_label.setText(f"USB 设备: {n}"))
        self.controller_scan_ready.connect(self._on_controller_scan)
        self.kinesis_scan_ready.connect(self._on_kinesis_scan)
        self.alg_combo.currentTextChanged.connect(self.on_algorithm_changed)
        self.on_algorithm_changed(self.alg_combo.currentText())

    def closeEvent(self, ev) -> None:  # noqa: N802 - 退出时回收线程
        self._ui_closing = True
        if self._live_timer.isActive():
            self._live_timer.stop()
        if self._live_detect is not None:
            self._live_detect.stop()
            self._live_detect.wait(2000)
        if self._auto_detect is not None:
            self._auto_detect.stop()
            self._auto_detect.wait(3000)
        if self._preview_worker is not None:
            self._preview_worker.wait(2000)
        if self._sim_live is not None:   # sim01 仿真镜头资源
            try:
                # 退出前把 UI 台位当前位置同步回 SQLite 持久层，
                # 下次启动自动恢复（zero() 等旁路 on_move 的操作也一并落盘）
                self._sim_live.micro_stage.move_to(
                    dict(self._sim_live.motion_stage.position))
            except Exception:  # noqa: BLE001 - 退出时保存失败不阻断关闭
                pass
            # 需求1：退出前把框定元素与当前视窗一并落盘，避免下次启动
            # 出现"元素位置/对象丢失"（视窗落盘需在 close 之前）。
            self._save_sim_config()
            self._sim_live.close()
            self._sim_live = None
        if isinstance(self._live_world, video_sim.CameraWorld):
            self._live_world.close()     # 电机模式真实相机
            self._live_world = None
        self._close_motor_xyz_stage()
        super().closeEvent(ev)

    # ---------------- 实时检测 / 区域划分
    def _is_motor_mode(self) -> bool:
        """True=电机模式（真实相机+真实位移台+YOLO）；False=虚拟模式（sim01）。"""
        return self.mode_sel.currentIndex() == 1

    def _get_xyz_stage(self):
        if self._is_motor_mode():
            return self._ensure_motor_xyz_stage()
        world = self._ensure_live()
        self._xyz_stage = world.motion_stage
        return self._xyz_stage

    def _ensure_motor_xyz_stage(self):
        """Open the selected hardware controller for manual XYZ operations."""
        if self.driver_combo.currentIndex() == 1:
            raise MotionStateError(
                "串口 G 代码驱动当前仅支持避障/组装 XY；请选择 8742/8743 或 Kinesis")
        if self._motor_xyz_stage is not None:
            return self._motor_xyz_stage
        if not self.confirm_chk.isChecked():
            raise MotionStateError("请先勾选真实电机安全确认，再使用 XYZ 台位")
        if __package__ in (None, ""):   # 直接运行 app.py 无包上下文
            from obstacle_avoidance.stages import (KinesisXYZStage,
                                                   PicoMotorXYZStage)
        else:
            from .stages import KinesisXYZStage, PicoMotorXYZStage
        profiles = dict(self._xyz_axis_profiles)
        default_steps_per_unit = self.spm_spin.value() / 1000.0
        for axis in ("x", "y", "z"):
            profiles.setdefault(axis, {})
            profiles[axis].setdefault("steps_per_unit", default_steps_per_unit)
        # The first profile is also used as the initial calibration for the
        # selected axis; defaults match the existing XYZ panel units (µm).
        try:
            if self.driver_combo.currentIndex() == 2:
                stage = KinesisXYZStage(
                    serial_no=self._kinesis_serial(),
                    steps_per_mm=self.spm_spin.value(),
                    profiles=profiles,
                    speed_steps=(self.speed_spin.value() or None),
                    accel_steps=(self.accel_steps_spin.value() or None),
                    confirmed=True)
            else:
                stage = PicoMotorXYZStage(
                    conn=self.conn_spin.value(),
                    x_axis=self.axis_x_spin.value(),
                    y_axis=self.axis_y_spin.value(),
                    z_axis=self.axis_z_spin.value(),
                    steps_per_mm=self.spm_spin.value(),
                    profiles=profiles,
                    speed_steps=(self.speed_spin.value() or None),
                    accel_steps=(self.accel_steps_spin.value() or None),
                    max_step_mm=self._step_mm(),
                    confirmed=True)
        except Exception as exc:  # noqa: BLE001 - surface driver errors in UI
            raise MotionStateError(f"XYZ 控制器连接失败: {exc}") from exc
        self._motor_xyz_stage = stage
        self._xyz_stage = stage
        self._update_xyz_view(stage)
        if self.driver_combo.currentIndex() == 2:
            mapping = getattr(stage, "channel_map_desc", lambda: "自动")()
            self._simlog("XYZ Kinesis 控制器已连接: {} 通道映射 {}".format(
                self._kinesis_serial(), mapping))
        else:
            self._simlog(
                "XYZ 控制器已连接: USB={} axes=X{} Y{} Z{}".format(
                    self.conn_spin.value(), self.axis_x_spin.value(),
                    self.axis_y_spin.value(), self.axis_z_spin.value()))
        return stage

    def _close_motor_xyz_stage(self) -> None:
        stage = getattr(self, "_motor_xyz_stage", None)
        self._motor_xyz_stage = None
        if stage is not None:
            try:
                stage.close()
            except Exception:  # noqa: BLE001 - shutdown must be best-effort
                pass

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
        except (MotionConfigError, MotionLimitError, MotionStateError,
                RuntimeError) as exc:
            self.detail_label.setText(str(exc))

    def _apply_xyz_profile(self) -> None:
        try:
            stage = self._get_xyz_stage()
            axis = self.xyz_axis_combo.currentText()
            profile = {
                "steps_per_unit": self.xyz_steps_spin.value(),
                "max_speed": self.xyz_speed_spin.value(),
                "acceleration": self.xyz_accel_spin.value(),
            }
            stage.configure_axis(axis, **profile)
            self._xyz_axis_profiles[axis] = profile
            self.detail_label.setText(f"{axis} motion profile applied")
        except (MotionConfigError, MotionLimitError, MotionStateError,
                RuntimeError) as exc:
            self.detail_label.setText(str(exc))

    def _home_xyz(self) -> None:
        try:
            stage = self._get_xyz_stage()
            telemetry = stage.home(source="ui-home")
            self._update_xyz_view(stage)
            self.on_motion(telemetry.to_dict())
            self._refresh_sim_preview()
        except (MotionConfigError, MotionLimitError, MotionStateError,
                RuntimeError) as exc:
            self.detail_label.setText(str(exc))

    def _zero_xyz(self) -> None:
        try:
            stage = self._get_xyz_stage()
            stage.zero()
            self._update_xyz_view(stage)
            self._refresh_sim_preview()
        except (MotionConfigError, MotionLimitError, MotionStateError,
                RuntimeError) as exc:
            self.detail_label.setText(str(exc))

    def _enable_xyz(self) -> None:
        try:
            stage = self._get_xyz_stage()
            stage.enable()
            self._update_xyz_view(stage)
        except (MotionStateError, RuntimeError) as exc:
            self.detail_label.setText(str(exc))

    def _stop_xyz(self) -> None:
        try:
            stage = self._get_xyz_stage()
            stage.stop()
            self._update_xyz_view(stage)
            self.detail_label.setText("XYZ stage stopped; press enable to resume")
        except (MotionStateError, RuntimeError) as exc:
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

    @Slot(bool)
    def on_draw_gate_toggled(self, on: bool) -> None:
        """画框闸门：锁定时拖拽/点击均不产生任何区域（防移动中误触）。"""
        self.canvas.drawing_enabled = on
        self.draw_gate_btn.setText("画框:开" if on else "画框:关")
        self._sync_polygon_mode()
        self._simlog(f"画框{'开启：拖拽画框生效' if on else '锁定（防误触）'}")
        if self._is_draw_mode():
            self.detail_label.setText(
                "画框已开启：拖拽画框生效" if on else
                "画框已锁定：需要画框请先点击\"画框:关\"按钮")

    @Slot()
    def _sync_polygon_mode(self, *_args) -> None:
        """框定形状=Free 时画布进入逐点多边形模式（随闸门/形状联动）。"""
        free = (getattr(self, "shape_combo", None) is not None
                and self.shape_combo.currentText() == "Free")
        self.canvas.set_polygon_mode(free)
        if free and self._is_draw_mode():
            self.detail_label.setText(
                "Free 自由多边形：左键逐点点击，双击/右键/点击起点闭合")

    @Slot(bool)
    def on_view_lock_toggled(self, on: bool) -> None:
        """锁定视角：禁用鼠标拖拽移动台（防止移动/运行中误触拖拽）。"""
        self._simlog("视角锁定：鼠标拖拽控制已禁用" if on
                     else "视角解锁：恢复鼠标拖拽控制")

    def _on_sim_drag_start(self, fx: int, fy: int) -> None:
        """鼠标按下：若在虚拟模式实时流且非画框模式，开启跟踪以支持位移台拖拽。"""
        if (not self._is_motor_mode()
                and self._sim_live is not None
                and self.live_btn.isChecked()
                and not self._is_draw_mode()
                and not self.view_lock_chk.isChecked()):
            self.canvas.setMouseTracking(True)
            self._sim_drag_acc = (0, 0)   # 累计拖拽增量 px

    def _on_sim_drag_move(self, dx: int, dy: int) -> None:
        """鼠标拖拽 -> 位移台移动（自然方向：拖拽方向 = 视野移动方向）。
        
        坐标映射：拖拽 dx_px → 球在画面中应移 -dx_px（即视野移 +dx_px），
        台位需移 +dx_px * pixel_size_um µm。
        """
        if self._is_motor_mode():
            return
        if self.view_lock_chk.isChecked():   # 拖拽中途上锁也立即停走
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
        if not self._is_motor_mode():
            self.canvas.setMouseTracking(False)
            ax, ay = self._sim_drag_acc
            if abs(ax) > 1 or abs(ay) > 1:
                px_um = (self._sim_live.pixel_size_um
                         if self._sim_live else 0.5)
                self._simlog(f"位移台拖拽完成: 累计 ({ax},{ay})px "
                             f"≈ ({ax * px_um:.0f},{ay * px_um:.0f})µm")

    def _ensure_live(self):
        if not self._is_motor_mode():
            # 仿真镜头实时流：SimMicroscopeWorld 连续渲染（复用，退出时统一关闭）
            if self._sim_live is None:
                # recenter=False：恢复上次会话的台位位置（worker 世界保持居中）
                self._sim_live = sim_microscope.SimMicroscopeWorld(
                    recenter=False)
                # 需求1：启动时若框定元素不在视窗内（上次运行把台位带到别处），
                # 把视窗对齐到元素包围盒中心，避免"重启后元素丢失"。
                pending = getattr(self, "_pending_view_center", None)
                if pending is not None:
                    self._pending_view_center = None
                    self._apply_view_center(pending)
            self._live_world = self._sim_live   # 统一入口（tick 判空/停止复用）
            self._xyz_stage = self._sim_live.motion_stage
            self._update_xyz_view(self._xyz_stage)
            return self._sim_live
        # 电机模式：真实相机/屏幕区域 + CameraWorld（YOLO 检测，ROI 配置决定视野）
        kind = "screen" if self._using_screen_source() else "camera"
        if (isinstance(self._live_world, video_sim.CameraWorld)
                and self._live_src_kind == kind):
            return self._live_world
        region_changed = False
        if kind == "screen":
            if self._screen_region is None:
                raise RuntimeError("请先点击'框选屏幕区域'选择显示屏区域")
            region_changed = (self._screen_region != self._live_src_region)
            src = video_sim.screen_source(region=self._screen_region)
            first = src()
            if first is None:
                raise RuntimeError("屏幕区域捕获失败（区域可能超出桌面范围）")
        else:
            src = video_sim.camera_source(self.cam_spin.value())
            first = src()
            if first is None:
                raise RuntimeError(f"相机 index={self.cam_spin.value()} 无画面")
        h, w = first.shape[:2]
        self._cam_frame_size = (w, h)
        self._live_src_kind = kind
        self._live_src_region = (self._screen_region
                                 if kind == "screen" else None)
        if kind == "screen" and region_changed:
            # 新选区域：旧 ROI/区域配置按旧画面划定，直接沿用会裁出
            # 错误子区（"选的内容显示不对"的根源）——重置为全画幅。
            self._roi_cfg = RoiConfig(roi=(0, 0, w, h), video="screen")
            self._roi_defaulted = True
        if self._roi_cfg is None:
            try:   # 优先载入上次保存的 ROI/区域配置
                cfg = RoiConfig.load(video_sim.DEFAULT_ROI_CONFIG)
                cfg.validate((w, h))
                self._roi_cfg = cfg
                self._roi_defaulted = False
            except Exception:  # noqa: BLE001 - 无配置/超尺寸 -> 全画幅
                self._roi_cfg = RoiConfig(roi=(0, 0, w, h), video=kind)
                self._roi_defaulted = True
        elif not region_changed:
            # 沿用现有配置（新选屏幕区域时保留上方重置的全画幅提示）
            self._roi_defaulted = False
        win = (int(self._roi_cfg.roi[2]), int(self._roi_cfg.roi[3]))
        off = (float(self._roi_cfg.roi[0]), float(self._roi_cfg.roi[1]))
        if win[0] <= 0 or win[1] <= 0 or win[0] > w or win[1] > h:
            win, off = (w, h), (0.0, 0.0)
        self._live_world = video_sim.CameraWorld(
            frame_source=src, window=win, offset=off,
            px_per_mm=self._roi_cfg.px_per_mm)
        self._sync_substrate_zone()   # 沿用/载入配置中的『衬底区域』标注
        msg = f"帧源: {self._frame_source_desc()}（{w}x{h}）"
        self.detail_label.setText(msg)
        self._simlog(msg)   # 明确记录本次检测使用的帧源
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
                if self._auto_detect is not None:
                    self._auto_detect.stop()
                    self._auto_detect.wait(3000)
                    self._auto_detect = None
                self._live_dets = []
                self.live_btn.setText("开始实时检测")
                if self.state_label.text() == "LIVE":
                    self.state_label.setText("IDLE")
                if self._is_motor_mode() and isinstance(
                        self._live_world, video_sim.CameraWorld):
                    self._live_world.close()   # 释放真实相机，重启时重新打开
                    self._live_world = None
        except Exception as exc:  # noqa: BLE001
            self.live_btn.setChecked(False)
            _log_exception("live detect start failed", exc)
            self.detail_label.setText(f"实时检测启动失败: {exc}")

    def _annotate_live(self, frame: np.ndarray) -> np.ndarray:
        """叠加 ROI、安全缓冲带、区域、YOLO 检测框（视频坐标->窗口坐标）。"""
        out = frame.copy()
        cfg = self._roi_cfg
        ox, oy = self._live_world.offset
        if getattr(self, "_roi_defaulted", False) and cfg is not None:
            # ROI 未框定：画面角标动态提示（用『ROI 视野』画框后自动消失）
            Hh, Ww = out.shape[:2]
            text = "Draw 'ROI' rect on canvas to set view window"
            cv2.putText(out, text, (12, Hh - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 3)
            cv2.putText(out, text, (12, Hh - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 220), 1)
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
                      "free": (180, 120, 0), "substrate": (255, 0, 255)}
            for z in cfg.zones:
                # 区域框随样品同步：按实测图像位移增量平移（ROI 框不参与）
                sx, sy = self._zone_draw_offset(z.name)
                x, y, w, h = (int(z.rect[0] - ox + sx),
                              int(z.rect[1] - oy + sy),
                              int(z.rect[2]), int(z.rect[3]))
                c = colors[z.kind]
                if z.shape == "free" and z.points:
                    # 自由多边形：实心/警戒环/轮廓均按顶点绘制
                    poly = [(int(px - ox + sx), int(py - oy + sy))
                            for px, py in z.points]
                    pts = np.array(poly, np.int32).reshape(-1, 1, 2)
                    if z.kind == "obstacle":
                        cv2.fillPoly(out, [pts], c)
                    else:
                        cv2.polylines(out, [pts], True, c, 2)
                    lx, ly = poly[0]
                    cv2.putText(out, z.name, (lx + 2, ly + 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
                    continue
                if z.shape == "circle":
                    # 圆形区域：实心/警戒环/外框均按圆绘制
                    cc = (x + w // 2, y + h // 2)
                    r = int(z.radius())
                    if z.kind == "obstacle":
                        cv2.circle(out, cc, r, c, -1)
                        cv2.circle(out, cc, r + band, c, 1)
                    else:
                        cv2.circle(out, cc, r, c, 2)
                    cv2.putText(out, z.name, (cc[0] - r + 2, cc[1] - r + 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
                    continue
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
        if self._auto_recognition_active() and self._auto_result is not None:
            if __package__ in (None, ""):
                from obstacle_avoidance.auto_recognition import Alg2AutoRecognizer
            else:
                from .auto_recognition import Alg2AutoRecognizer
            out = Alg2AutoRecognizer.draw_overlay(out, self._auto_result)
        # 运行期规划路径（电机模式：与虚拟模式一致地实时标注）
        if self._run_plan_pts:
            out = WorkerThread._draw_overlay(out, self._run_plan_pts)
        return out

    @Slot()
    def _live_tick(self) -> None:
        if self._live_world is None:
            return
        frame = self._live_world.render()
        if frame is None:   # CameraWorld 读帧失败（相机断开等）
            return
        self._tick_n += 1
        if not self._is_motor_mode():
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
        if self._auto_recognition_active():
            if det is not None:
                if self._auto_detect is None:
                    # 衬底：优先手动『衬底区域』标注，未标注时用 ROI 视野
                    manual_polygon = self._substrate_window_polygon()
                    if manual_polygon is None and self._roi_cfg is not None:
                        rx, ry, rw, rh = self._roi_cfg.roi
                        manual_polygon = [(0.0, 0.0), (float(rw - 1), 0.0),
                                          (float(rw - 1), float(rh - 1)),
                                          (0.0, float(rh - 1))]
                    self._auto_detect = AutoRecognizeThread(
                        lambda: det, self.auto_confidence_spin.value(), self,
                        manual_substrate_polygon=manual_polygon)
                    self._auto_detect.result_ready.connect(
                        self._on_auto_recognition_result)
                    self._auto_detect.recognition_failed.connect(
                        self._on_auto_recognition_failed)
                    self._auto_detect.start()
                self._auto_detect.submit(frame)
            elif self._tick_n % 25 == 1:
                self.auto_status_label.setText("YOLO 模型加载中...")
            self._show_frame(self._annotate_live(frame))
            return
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

    # ---- 样本绝对坐标转换（修复：绘制对象须固定在仿真样本上，不随视窗平移）
    @staticmethod
    def _sim_initial_origin():
        """全新 SimMicroscopeWorld 的初始视窗原点（样本->窗口用）。"""
        return (sim_microscope.SAMPLE_CENTER[0] - sim_microscope.WINDOW[0] / 2.0,
                sim_microscope.SAMPLE_CENTER[1] - sim_microscope.WINDOW[1] / 2.0)

    def _sim_origin(self):
        """当前视窗原点：样本坐标 = 窗口坐标 + 原点。无世界时用初始原点。"""
        if self._sim_live is not None:
            return tuple(self._sim_live._view_origin())
        return self._sim_initial_origin()

    @staticmethod
    def _rect_w2s(rect, origin):
        """窗口矩形 -> 样本矩形（平移，尺寸不变）。"""
        return [int(rect[0] + origin[0]), int(rect[1] + origin[1]),
                int(rect[2]), int(rect[3])]

    @staticmethod
    def _rect_s2w(rect, origin):
        """样本矩形 -> 窗口矩形。"""
        return [int(round(rect[0] - origin[0])), int(round(rect[1] - origin[1])),
                int(rect[2]), int(rect[3])]

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

    def _sim_rect(self, x: int, y: int, w: int, h: int,
                  poly: Optional[list] = None) -> None:
        """sim01：按画框模式记录 ROI；目标点/范围归属到其所在 ground。

        绘制坐标是当前视窗的窗口坐标，存储统一转为样本绝对坐标
        （对象固定在仿真样本上，不随视窗平移）；显示时再转回窗口坐标。
        poly：Free 多边形顶点（窗口坐标，边数按点位确定）——衬底存入
        ground["poly"]，障碍存入 obstacle_polys[索引]（与 obstacles 平行）；
        圆球物理为圆，取外接圆（bbox），仅记录顶点数。
        """
        mode = self.mode_combo.currentText()
        # 框定形状归一化（窗口坐标，中心不变；目标点不受影响）
        x, y, w, h, _shp = self._apply_draw_shape(x, y, w, h)
        origin = self._sim_origin()
        wx, wy = int(x), int(y)                    # 窗口坐标（日志用）
        x, y, w, h = self._rect_w2s([x, y, w, h], origin)
        poly_s = ([(int(px + origin[0]), int(py + origin[1])) for px, py in poly]
                  if poly and len(poly) >= 3 else None)
        cx, cy = int(x + w / 2), int(y + h / 2)    # 样本坐标
        if mode == "衬底(ground)":
            if w <= 10 or h <= 10:
                self.detail_label.setText("衬底框太小（需 > 10px）")
                return
            gid = self._new_id("G")
            ground = {"id": gid, "rect": [x, y, w, h],
                      "goal": None, "goal_range": None}
            if poly_s:
                ground["poly"] = poly_s    # 真实多边形可行域（边数=顶点数）
            self._sim_cfg["grounds"].append(ground)
            self._sim_order.append(("ground_add",))
            self._simlog(f"新建衬底 {gid}: 窗口=({wx},{wy}) "
                         f"样本=({x},{y}) size={w}x{h} "
                         + (f"poly={len(poly_s)}边形 " if poly_s else "")
                         + f"vertices={self._vertices([x, y, w, h])}")
        elif mode == "圆球(mask)":
            if w <= 6 or h <= 6:
                self.detail_label.setText("圆球框太小")
                return
            if self._find_ground_for_rect((x, y, w, h)) < 0:
                self.detail_label.setText(
                    "⚠ 圆球必须在衬底范围内（先画衬底）")
                return
            bid = self._new_id("B")
            self._sim_cfg["balls"].append([x, y, w, h])
            self._sim_ids["balls"].append(bid)
            self._sim_order.append(("ball_add",))
            self._simlog(f"新建圆球 {bid}: 窗口=({wx},{wy}) "
                         f"样本=({x},{y}) size={w}x{h} center=({cx},{cy})"
                         + (f"（Free {len(poly_s)}边形按外接圆，球物理为圆）"
                            if poly_s else ""))
        elif mode == "障碍物(obstacle)":
            if w <= 6 or h <= 6:
                self.detail_label.setText("障碍框太小")
                return
            if self._find_ground_for_rect((x, y, w, h)) < 0:
                self.detail_label.setText(
                    "⚠ 障碍物必须在衬底范围内（先画衬底）")
                return
            oid_ = self._new_id("O")
            self._sim_cfg["obstacles"].append([x, y, w, h])
            self._sim_ids["obstacles"].append(oid_)
            # Free 多边形障碍：碰撞按真实边界（边数=顶点数）；None=矩形
            self._sim_cfg.setdefault("obstacle_polys", []).append(poly_s)
            self._sim_order.append(("obstacle_add",))
            self._simlog(f"新建障碍物 {oid_}: 窗口=({wx},{wy}) "
                         f"样本=({x},{y}) size={w}x{h} center=({cx},{cy})"
                         + (f" poly={len(poly_s)}边形" if poly_s else ""))
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
            self._sim_cfg["grounds"][gi]["goal_range"] = [x, y, w, h]
            self._sim_order.append(("ground_range", gi))
            self._simlog(f"设定目标范围 -> {self._sim_cfg['grounds'][gi]['id']}"
                         f": 窗口=({wx},{wy}) 样本=({x},{y}) size={w}x{h}")
        else:
            self.detail_label.setText(
                "sim01 请选择：圆球/衬底/障碍物/目标点(避障)/目标范围(组装)")
            return
        self._save_sim_config()
        self._update_sim_zones_label()
        self._refresh_sim_preview()

    # ---------------- 日志 / ID / 属性面板（需求 3/4/5）
    @Slot()
    def on_clear_log(self) -> None:
        """清空操作日志（仅清显示，不影响已保存的报告文件）。"""
        self.log_out.clear()

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
        if self._auto_result is not None:
            result = self._auto_result
            lines.append(
                f"AUTO    substrate conf={result.substrate_confidence:.2f} "
                f"registration={result.registration_confidence:.2f} "
                "source=manual-roi")
            lines.append(
                f"AUTO    workspace shift=({result.registration_shift_px[0]:.1f},"
                f"{result.registration_shift_px[1]:.1f}) "
                f"instant=({result.instant_shift_px[0]:.1f},"
                f"{result.instant_shift_px[1]:.1f}) method={result.motion_method}")
            if result.track_events:
                lines.append("AUTO    tracks " + ", ".join(result.track_events))
            if result.beam_spot_px is not None:
                bx, by = result.beam_spot_px
                lines.append(
                    f"AUTO    beam     pos=({bx:.0f},{by:.0f}) "
                    f"conf={result.beam_confidence:.2f}")
            lines.append(f"AUTO    overall={result.overall_confidence:.2f}")
        self.props_out.setPlainText("\n".join(lines) or "(无对象)")

    @Slot(int, int)
    def _on_target_clicked(self, x: int, y: int) -> None:
        """点击选择目标球：点击坐标为窗口坐标，球存样本绝对坐标，
        须按当前视窗原点换算后再匹配（否则永远选不中）。"""
        if self._beam_calibration_armed:
            self.beam_x_spin.setValue(float(x))
            self.beam_y_spin.setValue(float(y))
            self._beam_calibrated = True
            self._beam_calibration_armed = False
            self.beam_pick_btn.setChecked(False)
            self.beam_status_label.setText(
                f"已标定固定激光位置 ({x:.1f}, {y:.1f}) px")
            self.beam_status_label.setStyleSheet(
                "color:#087f23;font-weight:bold;")
            self._simlog(f"Alg2 激光位置标定: ({x:.1f},{y:.1f}) px")
            self._refresh_beam_spot()
            return
        if (self._is_motor_mode() and self._auto_result is not None and
                self._auto_recognition_active()):
            candidates = sorted(
                (math.dist(p.position_px, (x, y)), p)
                for p in self._auto_result.particles)
            if candidates and candidates[0][0] <= max(
                    12.0, candidates[0][1].radius_px * 1.8):
                self._selected_auto_track_id = candidates[0][1].track_id
                self.detail_label.setText(
                    f"已选择自动识别圆球 track={self._selected_auto_track_id}，"
                    "启动时将先对准固定光斑")
                self._update_props_panel()
            return
        if self.worker is not None and self.worker.isRunning():
            return
        ox, oy = self._sim_origin()
        sx, sy = x + ox, y + oy
        candidates = []
        for index, ball in enumerate(self._sim_cfg.get("balls", [])):
            cx = ball[0] + ball[2] / 2.0
            cy = ball[1] + ball[3] / 2.0
            distance = ((cx - sx) ** 2 + (cy - sy) ** 2) ** 0.5
            if distance <= max(ball[2], ball[3]) * 0.75:
                candidates.append((distance, index))
        if not candidates:
            return
        self.selected_ball_index = min(candidates)[1]
        self.detail_label.setText(
            f"Selected target ball #{self.selected_ball_index + 1}; it will be locked before motion")
        self._update_sim_zones_label()

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
        """预览帧上叠加 sim01 ROI：衬底=蓝、球=青、障碍=红、目标点=黄十字、目标范围=绿圆。

        _sim_cfg 存样本绝对坐标 -> 按当前视窗原点转回窗口坐标绘制
        （对象固定在样本上，视窗平移时随样本反向移动）。
        运行中的 worker 帧来自全新世界（初始视窗），用初始原点。
        """
        out = frame.copy()
        c = self._sim_cfg
        if self.worker is not None and self.worker.isRunning():
            # 运行中 worker 帧来自按运行原点定位的世界（on_run 记录）
            ox, oy = getattr(self, "_run_origin", None) or \
                self._sim_initial_origin()
        else:
            ox, oy = self._sim_origin()
        for i, g in enumerate(c["grounds"]):
            if g.get("poly"):
                # Free 多边形衬底：按顶点绘制（边数=顶点数）
                pts = np.array(
                    [[px - ox, py - oy] for px, py in g["poly"]],
                    np.int32).reshape(-1, 1, 2)
                cv2.polylines(out, [pts], True, (255, 120, 0), 1)
                lx, ly = g["poly"][0]
                cv2.putText(out, f"G{i}", (int(lx - ox + 2), int(ly - oy - 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 120, 0), 1)
            else:
                rx, ry, rw, rh = self._rect_s2w(g["rect"], (ox, oy))
                cv2.rectangle(out, (rx, ry), (rx + rw, ry + rh),
                              (255, 120, 0), 1)
                cv2.putText(out, f"G{i}", (rx + 2, ry - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 120, 0), 1)
            if g.get("goal"):
                gx, gy = g["goal"]
                gx, gy = int(gx - ox), int(gy - oy)
                cv2.drawMarker(out, (gx, gy), (0, 255, 255),
                               cv2.MARKER_CROSS, 16, 2)
            if g.get("goal_range"):
                gx, gy, grw, grh = self._rect_s2w(g["goal_range"], (ox, oy))
                cv2.rectangle(out, (gx, gy), (gx + grw, gy + grh),
                              (0, 220, 0), 2)
                cx, cy = int(gx + grw / 2), int(gy + grh / 2)
                cv2.circle(out, (cx, cy), 4, (0, 220, 0), -1)
        polys = c.get("obstacle_polys") or []
        for i, ob in enumerate(c["obstacles"]):
            poly = polys[i] if i < len(polys) else None
            if poly:
                # Free 多边形障碍：按顶点绘制（边数=顶点数）
                pts = np.array([[px - ox, py - oy] for px, py in poly],
                               np.int32).reshape(-1, 1, 2)
                cv2.polylines(out, [pts], True, (0, 0, 255), 1)
            else:
                x, y, w, h = self._rect_s2w(ob, (ox, oy))
                cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 1)
        for ball_index, b in enumerate(c["balls"]):
            x, y, w, h = self._rect_s2w(b, (ox, oy))
            color = (0, 255, 255) if ball_index == self.selected_ball_index else (255, 255, 0)
            thickness = 3 if ball_index == self.selected_ball_index else 1
            cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
            cv2.putText(out, f"ball-{ball_index + 1}", (x, max(16, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        return out

    def _refresh_sim_preview(self) -> None:
        if self._sim_live is not None:
            try:
                frame = self._sim_live.render()
                # During an active run the simulator already renders the
                # physical object shapes.  The config overlay is an editing
                # aid and would turn those objects back into rectangles.
                if not (self.worker is not None and self.worker.isRunning()):
                    frame = self._overlay_sim_cfg(frame)
                self._show_frame(frame)
                self._update_xyz_view(self._sim_live.motion_stage)
                return
            except Exception as exc:  # pragma: no cover - display fallback
                _log_exception("sim preview refresh failed", exc)
        if self._sim_preview_frame is not None:
            frame = self._sim_preview_frame
            if not (self.worker is not None and self.worker.isRunning()):
                frame = self._overlay_sim_cfg(frame)
            self._show_frame(frame)

    SIM_ROI_CONFIG = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "artifacts", "sim_roi_config.json")
    SIM_SAMPLE_SPEC = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "artifacts", "sim_sample_spec.json")

    def _save_sim_config(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.SIM_ROI_CONFIG), exist_ok=True)
            data = dict(self._sim_cfg)
            data["coords"] = "sample"   # 坐标系标记（样本绝对坐标）
            # 需求1：连同当前视窗一起落盘（origin=样本->窗口平移量），
            # 便于复盘以及元素不在视窗内时自动对齐。
            ox, oy = self._sim_origin()
            view: dict = {"origin": [int(ox), int(oy)]}
            if self._sim_live is not None:
                pos = self._sim_live.micro_stage.position
                view["stage_um"] = [float(pos["x"]), float(pos["y"])]
            data["view"] = view
            with open(self.SIM_ROI_CONFIG, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
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
                # 旧版本存的是窗口坐标（初始视窗）-> 迁移为样本绝对坐标
                if cfg.get("coords") != "sample":
                    o = self._sim_initial_origin()
                    for g in cfg.get("grounds", []):
                        if isinstance(g, dict):
                            g["rect"] = self._rect_w2s(g["rect"], o)
                            if g.get("goal"):
                                g["goal"] = [int(g["goal"][0] + o[0]),
                                             int(g["goal"][1] + o[1])]
                            if g.get("goal_range"):
                                g["goal_range"] = self._rect_w2s(
                                    g["goal_range"], o)
                    cfg["balls"] = [self._rect_w2s(b, o)
                                    for b in cfg.get("balls", [])]
                    cfg["obstacles"] = [self._rect_w2s(ob, o)
                                        for ob in cfg.get("obstacles", [])]
                    cfg["coords"] = "sample"
                self._sim_cfg = {**self._sim_cfg, **cfg}
        except Exception:  # noqa: BLE001 - 无配置/损坏时用默认空布局
            return
        # obstacle_polys 归一化为与 obstacles 平行的列表（None=矩形）；
        # 旧 dict 格式（键=障碍ID）载入后 ID 重新补配、无法对应 rect
        # 顺序 -> 丢弃降级为矩形障碍
        polys = self._sim_cfg.get("obstacle_polys")
        if isinstance(polys, dict) or not isinstance(polys, list):
            polys = []
        obs_n = len(self._sim_cfg.get("obstacles", []))
        self._sim_cfg["obstacle_polys"] = \
            (list(polys) + [None] * obs_n)[:obs_n]
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
        # 需求1：载入后确保元素落在视窗内（不丢对象/不丢位置）
        self._sync_view_to_elements()

    # ---- 需求1：视窗与框定元素对齐（对象固定在样本上，视窗须跟上）
    def _element_bbox_center(self) -> Optional[tuple]:
        """所有框定元素（衬底/球/障碍/目标点/目标范围）样本坐标包围盒中心。"""
        xs0, ys0 = float("inf"), float("inf")
        xs1, ys1 = float("-inf"), float("-inf")

        def add_rect(rect) -> None:
            nonlocal xs0, ys0, xs1, ys1
            if not rect:
                return
            xs0, ys0 = min(xs0, rect[0]), min(ys0, rect[1])
            xs1 = max(xs1, rect[0] + rect[2])
            ys1 = max(ys1, rect[1] + rect[3])

        def add_pt(pt) -> None:
            nonlocal xs0, ys0, xs1, ys1
            if not pt:
                return
            xs0, ys0 = min(xs0, pt[0]), min(ys0, pt[1])
            xs1, ys1 = max(xs1, pt[0]), max(ys1, pt[1])

        c = self._sim_cfg
        for g in c.get("grounds", []):
            if not isinstance(g, dict):
                continue
            add_rect(g.get("rect"))
            add_rect(g.get("goal_range"))
            add_pt(g.get("goal"))
            for p in (g.get("poly") or []):
                add_pt(p)
        for b in c.get("balls", []):
            add_rect(b)
        for ob in c.get("obstacles", []):
            add_rect(ob)
        for poly in (c.get("obstacle_polys") or []):
            for p in (poly or []):
                add_pt(p)
        if xs0 > xs1 or ys0 > ys1:
            return None
        return ((xs0 + xs1) / 2.0, (ys0 + ys1) / 2.0)

    def _sync_view_to_elements(self) -> None:
        """元素不在当前视窗内 -> 对齐视窗（无实时世界时暂存，创建后应用）。"""
        center = self._element_bbox_center()
        if center is None:
            return
        self._apply_view_center(center)

    def _apply_view_center(self, center) -> None:
        """把视窗中心移到 center（样本坐标）；已在视窗内则保持用户位置。"""
        world = self._sim_live
        if world is None:
            self._pending_view_center = center
            return
        w, h = sim_microscope.WINDOW
        ox, oy = world._view_origin()
        wx, wy = center[0] - ox, center[1] - oy
        if 0 <= wx < w and 0 <= wy < h:
            return
        try:
            px_um = world.pixel_size_um
            world.motion_stage.move_to({"x": center[0] * px_um,
                                        "y": center[1] * px_um},
                                       source="restore-element-view")
        except Exception as exc:  # noqa: BLE001 - 对齐失败不阻断启动
            self._simlog(f"视窗对齐失败: {exc}")
            return
        self._simlog(f"视窗已对齐到框定元素中心 "
                     f"({center[0]:.0f},{center[1]:.0f})px（元素保持可见）")

    def _load_sample_spec(self) -> Optional[dict]:
        """读取仿真镜头图层配置（不存在/损坏返回 None -> 用相机当前配置）。"""
        try:
            with open(self.SIM_SAMPLE_SPEC, "r", encoding="utf-8") as f:
                spec = json.load(f)
            return spec if isinstance(spec, dict) else None
        except Exception:  # noqa: BLE001
            return None

    def _build_sim_spec_panel(self) -> QtWidgets.QGroupBox:
        """仿真镜头图层配置面板（内嵌于"检测 / ROI"界面下方）。

        与 SimSpecDialog 同构：衬底/掩码/障碍物三类的
        数量/形状/标签/尺寸/顶点，直接显示并可在面板内保存，无需打开弹窗。
        """
        spec = self._load_sample_spec() or {}
        box = QtWidgets.QGroupBox("仿真镜头图层（衬底 / 掩码 / 障碍物）")
        outer = QtWidgets.QVBoxLayout(box)
        outer.setContentsMargins(6, 6, 6, 6)
        self._sim_spec_fields: dict = {}
        for cat, zh in SimSpecDialog.CATS:
            cfg = spec.get(cat) or {}
            count = QtWidgets.QSpinBox()
            count.setRange(0, 50)
            count.setValue(int(cfg.get("count", 0) or 0))
            shape = QtWidgets.QComboBox()
            shape.addItems(SimSpecDialog.SHAPES)
            shape.setCurrentText(str(cfg.get("shape", "ellipse")))
            label = QtWidgets.QLineEdit(str(cfg.get("label", "")))
            size = QtWidgets.QDoubleSpinBox()
            size.setRange(2.0, 500.0)
            size.setValue(float(cfg.get("size", 40.0) or 40.0))
            custom = QtWidgets.QLineEdit(str(cfg.get("custom", "")))
            custom.setToolTip(
                '自定义多边形顶点："x,y x,y ..."（shape=custom）')
            sub = QtWidgets.QGridLayout()
            for col, (name, w) in enumerate((
                    ("数量", count), ("形状", shape), ("标签", label),
                    ("尺寸", size), ("顶点", custom))):
                sub.addWidget(QtWidgets.QLabel(name), 0, col)
                sub.addWidget(w, 1, col)
            group = QtWidgets.QGroupBox(zh)
            group.setLayout(sub)
            outer.addWidget(group)
            self._sim_spec_fields[cat] = (count, shape, label, size, custom)
        save_btn = QtWidgets.QPushButton("保存图层配置")
        save_btn.setSizePolicy(QtWidgets.QSizePolicy.Maximum,
                               QtWidgets.QSizePolicy.Fixed)
        save_btn.clicked.connect(self._save_inline_sim_spec)
        outer.addWidget(save_btn)
        return box

    def _collect_sim_spec(self) -> dict:
        """收集内嵌面板字段 -> sample_spec（空标签沿用内置默认前缀）。"""
        out = {}
        for cat, (count, shape, label, size, custom) in \
                self._sim_spec_fields.items():
            cfg = {"count": count.value(), "shape": shape.currentText(),
                   "size": size.value()}
            if label.text().strip():
                cfg["label"] = label.text().strip()
            if shape.currentText() == "custom" and custom.text().strip():
                cfg["custom"] = custom.text().strip()
            out[cat] = cfg
        return out

    @Slot()
    def _save_inline_sim_spec(self) -> None:
        """保存内嵌面板的仿真镜头图层配置到磁盘。"""
        new_spec = self._collect_sim_spec()
        try:
            os.makedirs(os.path.dirname(self.SIM_SAMPLE_SPEC), exist_ok=True)
            with open(self.SIM_SAMPLE_SPEC, "w", encoding="utf-8") as f:
                json.dump(new_spec, f, ensure_ascii=False, indent=2)
        except Exception as exc:  # noqa: BLE001
            self.detail_label.setText(f"图层配置保存失败: {exc}")
            return
        self.detail_label.setText(
            "图层配置已保存，重新运行 sim01 后生效")

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

    def _auto_recognition_active(self) -> bool:
        return (self.alg_combo.currentText() == "Alg2" and
                self._is_motor_mode() and
                self.auto_recognition_chk.isChecked())

    @Slot()
    def on_calibrate_beam_center(self) -> None:
        """Calibrate the stationary beam to the active cropped frame center."""

        if self._last_frame is None:
            self.detail_label.setText("请先开始实时检测并获取一帧图像")
            return
        h, w = self._last_frame.shape[:2]
        self.beam_x_spin.setValue(w / 2.0)
        self.beam_y_spin.setValue(h / 2.0)
        self._beam_calibrated = True
        self.beam_status_label.setText(
            f"已标定固定激光位置 ({w / 2.0:.1f}, {h / 2.0:.1f}) px")
        self.beam_status_label.setStyleSheet("color:#087f23;font-weight:bold;")
        self._simlog(
            f"Alg2 激光位置标定: ({w / 2.0:.1f},{h / 2.0:.1f}) px")
        self._refresh_beam_spot()

    @Slot(bool)
    def _on_beam_pick_toggled(self, armed: bool) -> None:
        self._beam_calibration_armed = bool(armed)
        if armed:
            self.beam_status_label.setText("请在画面中点击固定激光光斑中心")
            self.beam_status_label.setStyleSheet("color:#b36b00;font-weight:bold;")
        elif not self._beam_calibrated:
            self.beam_status_label.setText("激光位置未标定")
            self.beam_status_label.setStyleSheet("color:#b36b00;")

    @Slot(str)
    def on_algorithm_changed(self, algorithm: str) -> None:
        is_alg2 = algorithm == "Alg2"
        self.auto_recognition_box.setVisible(is_alg2)
        # Beam calibration is shared by virtual and motor Alg2. Automatic
        # recognition itself is still activated only by the motor live path.
        self.auto_recognition_box.setEnabled(is_alg2)
        if is_alg2:
            self.auto_recognition_chk.setChecked(True)
        else:
            self.auto_recognition_chk.setChecked(False)
        self._reset_auto_recognition(
            "请切换到电机模式并开始实时检测" if not self._is_motor_mode()
            else "开始实时检测后自动识别")

    @Slot(bool)
    def on_auto_recognition_toggled(self, enabled: bool) -> None:
        self._reset_auto_recognition(
            "开始实时检测后自动识别" if enabled else "自动识别已关闭")
        if self.live_btn.isChecked():
            if self._live_detect is not None:
                self._live_detect.stop()
                self._live_detect.wait(2000)
                self._live_detect = None
            if self._auto_detect is not None:
                self._auto_detect.stop()
                self._auto_detect.wait(3000)
                self._auto_detect = None

    @Slot(float)
    def on_auto_confidence_changed(self, _value: float) -> None:
        self._reset_auto_recognition("置信度阈值已变化，请重新检测并确认")
        if self._auto_detect is not None:
            self._auto_detect.stop()
            self._auto_detect.wait(3000)
            self._auto_detect = None

    def _reset_auto_recognition(self, message: str) -> None:
        self._auto_result = None
        # 识别器重开后累计位移从 0 重新计，旧的画框参考位移作废
        self._zone_ref_shift.clear()
        self._auto_confirmed_result = None
        self._selected_auto_track_id = None
        self._auto_good_streak = 0
        self._auto_recognition_confirmed = False
        self.auto_confirm_btn.setEnabled(False)
        self.auto_status_label.setText(message)
        self.auto_status_label.setStyleSheet("color:#888;")

    @Slot(object)
    def _on_auto_recognition_result(self, result) -> None:
        self._auto_result = result
        self._live_dets = [(p.position_px, p.radius_px, p.confidence)
                           for p in result.particles
                           if p.frame_id == result.frame_id]
        threshold = self.auto_confidence_spin.value()
        acceptable = (not result.uncertain and
                      result.overall_confidence >= threshold and
                      bool(result.particles) and
                      bool(result.substrate_polygon))
        self._auto_good_streak = self._auto_good_streak + 1 if acceptable else 0
        if not acceptable:
            self._auto_recognition_confirmed = False
            self._auto_confirmed_result = None
        self.auto_confirm_btn.setEnabled(self._auto_good_streak >= 3)
        inside = sum(result.particles_inside_substrate)
        total = len(result.particles)
        if acceptable:
            message = (f"识别稳定 {self._auto_good_streak}/3 | 球 {total} "
                       f"(衬底内 {inside}) | "
                       f"置信度 {result.overall_confidence:.2f}")
            color = "#087f23" if self._auto_good_streak >= 3 else "#b36b00"
        else:
            reason = result.uncertain_reason or "低于置信度阈值"
            message = (f"识别未通过: {reason} | 球 {total} | "
                       f"置信度 {result.overall_confidence:.2f}")
            color = "#c00"
        self.auto_status_label.setText(message)
        self.auto_status_label.setStyleSheet(f"color:{color};")
        self._update_props_panel()

    def _live_sample_shift(self):
        """实时识别实测的样品图像位移 (px)；无有效实测时返回 None。"""
        result = self._auto_result
        if result is None:
            return None
        shift = getattr(result, "registration_shift_px", None)
        if shift is None or float(getattr(result, "registration_confidence",
                                          0.0)) <= 0.0:
            return None
        return (float(shift[0]), float(shift[1]))

    def _mark_zone_reference(self, name: str) -> None:
        """记录区域框的画框时刻位移：之后按增量平移框。"""
        self._zone_ref_shift[str(name)] = self._live_sample_shift()

    def _zone_draw_offset(self, name: str) -> tuple:
        """区域框相对画框时刻的实测位移增量（无实测/无参考时为 0）。"""
        current = self._live_sample_shift()
        reference = self._zone_ref_shift.get(str(name))
        if current is None or reference is None:
            return (0.0, 0.0)
        return (current[0] - reference[0], current[1] - reference[1])

    @Slot(str)
    def _on_auto_recognition_failed(self, message: str) -> None:
        self._auto_good_streak = 0
        self._auto_recognition_confirmed = False
        self.auto_confirm_btn.setEnabled(False)
        self.auto_status_label.setText(f"自动识别失败: {message}")
        self.auto_status_label.setStyleSheet("color:#c00;")

    @Slot()
    def on_confirm_auto_recognition(self) -> None:
        if self._auto_result is None or self._auto_good_streak < 3:
            self.detail_label.setText("自动识别尚未连续稳定 3 帧")
            return
        self._auto_confirmed_result = self._auto_result
        self._auto_recognition_confirmed = True
        self.auto_status_label.setText(
            f"已确认 frame={self._auto_result.frame_id}，可启动 Alg2")
        self.auto_status_label.setStyleSheet("color:#087f23;font-weight:bold;")
        self._simlog(
            f"Alg2 自动识别已确认: frame={self._auto_result.frame_id} "
            f"球={len(self._auto_result.particles)} "
            f"confidence={self._auto_result.overall_confidence:.3f}")

    def _apply_draw_shape(self, x: int, y: int, w: int, h: int):
        """按『框定形状』归一化拖拽矩形，返回 (x, y, w, h, shape)。

        长方形原样；正方形/圆形取拖拽框中心、短边为边长的正方形；
        圆形 shape='circle'（rect 为外接正方形，障碍构建时生成圆）。
        """
        shape = getattr(self, "shape_combo", None)
        mode = shape.currentText() if shape is not None else "长方形"
        if mode == "长方形":
            return int(x), int(y), int(w), int(h), "rect"
        if mode == "Free":
            # 自由多边形走逐点路径（on_points_drawn）；拖拽矩形不适用
            return int(x), int(y), int(w), int(h), "free"
        side = int(min(w, h))
        cx, cy = x + w / 2, y + h / 2
        nx = int(round(cx - side / 2))
        ny = int(round(cy - side / 2))
        return nx, ny, side, side, ("circle" if mode == "圆形" else "rect")

    @Slot(int, int, int, int)
    def on_rect_drawn(self, x: int, y: int, w: int, h: int) -> None:
        """画布拖拽结束：按模式应用 ROI 或区域（窗口坐标 -> 视频绝对坐标）。"""
        if not self._is_motor_mode():
            self._sim_rect(x, y, w, h)
            return
        if self._live_world is None or self._roi_cfg is None:
            self.detail_label.setText(
                "提示：请先点『开始实时检测』打开帧源，再画框（本次画框未生效）")
            return
        ox, oy = self._live_world.offset
        vx, vy = int(x + ox), int(y + oy)          # 视频绝对坐标
        mode = self.mode_combo.currentText()
        kind = None
        try:
            if mode == "ROI 视野":
                self._apply_roi((vx, vy, w, h))
            else:
                # 非矩形形状归一化（ROI 始终为矩形）
                x, y, w, h, shp = self._apply_draw_shape(x, y, w, h)
                if shp == "free":
                    self.detail_label.setText(
                        "Free 自由多边形：请左键逐点点击，双击/右键/点击起点闭合"
                        "（拖拽不适用）")
                    return
                vx, vy = int(x + ox), int(y + oy)
                kind = MOTOR_ZONE_KINDS[mode]
                if kind == "substrate" and shp == "circle":
                    self.detail_label.setText(
                        "衬底区域不支持圆形：请用长方形/正方形拖拽，"
                        "或选『Free』逐点框定衬底边界")
                    return
                # 裁剪进 ROI
                rx, ry, rw, rh = self._roi_cfg.roi
                x2, y2 = (min(vx + w, rx + rw), min(vy + h, ry + rh))
                vx, vy = max(vx, rx), max(vy, ry)
                w, h = x2 - vx, y2 - vy
                if w <= 8 or h <= 8:
                    self.detail_label.setText(
                        f"提示：{mode} 区域太小或不在 ROI 内，未生效")
                    return
                name = self._append_zone(kind, (vx, vy, w, h), shp)
                self.detail_label.setText(
                    f"已添加 {mode} {name} ({vx},{vy}) {w}x{h}"
                    + ("（圆）" if shp == "circle" else "")
                    + ("：衬底可行域已更新" if kind == "substrate" else ""))
        except ValueError as exc:
            self.detail_label.setText(f"区域非法: {exc}")
            return
        if kind == "substrate":
            self._sync_substrate_zone()
        self.zones_label.setText(f"zones: {len(self._roi_cfg.zones)}")

    @Slot(list)
    def on_points_drawn(self, pts: list) -> None:
        """Free 自由多边形收笔：顶点首尾连通，按当前模式落库。

        电机模式：区域存真实多边形（Zone.points，rect 为外接矩形），
        障碍区构建为多边形障碍；虚拟模式：衬底/障碍存真实多边形
        （圆球物理为圆，取外接）。
        """
        if not pts or len(pts) < 3:
            return
        if not self._is_motor_mode():
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x, y = int(min(xs)), int(min(ys))
            w, h = int(max(xs)) - x, int(max(ys)) - y
            if w > 4 and h > 4:
                # 真实多边形顶点随对象存储（衬底=多边形可行域，
                # 障碍=多边形碰撞边界；圆球物理为圆取外接）
                self._sim_rect(x, y, w, h,
                               poly=[(float(px), float(py)) for px, py in pts])
            return
        if self._live_world is None or self._roi_cfg is None:
            self.detail_label.setText(
                "提示：请先点『开始实时检测』打开帧源，再画多边形")
            return
        mode = self.mode_combo.currentText()
        if mode == "ROI 视野":
            self.detail_label.setText("ROI 视野需矩形拖拽（Free 不适用）")
            return
        ox, oy = self._live_world.offset
        kind = MOTOR_ZONE_KINDS[mode]
        # 顶点转视频绝对坐标并裁剪进 ROI
        rx, ry, rw, rh = self._roi_cfg.roi
        vpts = [(min(max(px + ox, rx), rx + rw),
                 min(max(py + oy, ry), ry + rh)) for px, py in pts]
        vpts = [p for i, p in enumerate(vpts)
                if i == 0 or p != vpts[i - 1]]   # 去连续重复顶点
        xs = [p[0] for p in vpts]
        ys = [p[1] for p in vpts]
        bx, by = min(xs), min(ys)
        bw, bh = max(xs) - bx, max(ys) - by
        if len(vpts) < 3 or bw <= 8 or bh <= 8:
            self.detail_label.setText(
                f"提示：{mode} 多边形太小或不在 ROI 内，未生效")
            return
        try:
            name = self._append_zone(
                kind, (bx, by, bw, bh), "free",
                points=[(round(px, 1), round(py, 1)) for px, py in vpts])
            self.detail_label.setText(
                f"已添加 {mode}(Free) {name}：{len(vpts)} 顶点，"
                f"外接 ({int(bx)},{int(by)}) {int(bw)}x{int(bh)}")
        except ValueError as exc:
            self.detail_label.setText(f"区域非法: {exc}")
            return
        if kind == "substrate":
            self._sync_substrate_zone()
        self.zones_label.setText(f"zones: {len(self._roi_cfg.zones)}")

    def _append_zone(self, kind: str, rect, shape: str,
                     points=None) -> str:
        """区域落库并返回名称：衬底区域为单例（重画即替换），其余累加编号。

        校验失败时回滚本次新增，保证配置始终合法。
        """
        zones = self._roi_cfg.zones
        if kind == "substrate":
            # 可行域只有一块：重画即替换（否则"最多一个"校验会拒绝新框定）
            zones[:] = [z for z in zones if z.kind != "substrate"]
        n = sum(1 for z in zones if z.kind == kind) + 1
        name = f"{MOTOR_ZONE_PREFIX[kind]}{n}"
        zones.append(Zone(name=name, kind=kind, rect=rect, shape=shape,
                          points=points))
        try:
            self._roi_cfg.validate()
        except ValueError:
            zones.pop()
            raise
        self._mark_zone_reference(name)
        return name

    def _substrate_window_polygon(self):
        """手动『衬底区域』（窗口坐标多边形）；未标注返回 None。"""
        cfg = self._roi_cfg
        zone = cfg.substrate_zone() if cfg is not None else None
        if zone is None:
            return None
        ox, oy = getattr(self._live_world, "offset", (0.0, 0.0))
        return [(px - ox, py - oy) for px, py in zone.polygon()]

    def _sync_substrate_zone(self) -> None:
        """把『衬底区域』标注同步到实时世界与识别线程（手动标注优先）。"""
        polygon = self._substrate_window_polygon()
        setter = getattr(self._live_world, "set_substrate_polygon", None)
        if callable(setter):
            setter(polygon)
        # 识别线程持有的是构造时的多边形；画框后必须换掉，否则衬底框仍是
        # 旧多边形（或 ROI 兜底矩形），看起来就像"衬底框跟真实衬底脱开"。
        updater = getattr(self._auto_detect, "set_manual_substrate", None)
        if callable(updater):
            zone = (self._roi_cfg.substrate_zone()
                    if self._roi_cfg is not None else None)
            if zone is not None:
                self._mark_zone_reference(zone.name)
            updater(self._substrate_window_polygon())

    def _config_frame_size(self):
        """ROI 几何校验用的全画幅尺寸：电机=相机帧，虚拟=视频帧。"""
        if self._is_motor_mode():
            return self._cam_frame_size
        return video_sim._video_size()

    def _apply_roi(self, rect) -> None:
        if not self._is_motor_mode():
            self.detail_label.setText("虚拟模式为固定仿真镜头，不支持重设 ROI 视野")
            return
        old = self._roi_cfg
        self._roi_cfg = RoiConfig(roi=tuple(rect), video="camera",
                                  px_per_mm=old.px_per_mm if old else 100.0,
                                  edge_clearance_px=(old.edge_clearance_px
                                                     if old else 1.0),
                                  zones=old.zones if old else [])
        try:
            self._roi_cfg.validate(self._config_frame_size())
        except ValueError as exc:
            self._roi_cfg = old
            self.detail_label.setText(f"ROI 非法: {exc}")
            return
        self._live_world.set_window((int(rect[2]), int(rect[3])),
                                    (float(rect[0]), float(rect[1])))
        self._roi_defaulted = False
        self._sync_substrate_zone()   # ROI 变更后衬底标注按新视野重新映射
        self.detail_label.setText(
            f"ROI 视野已更新 ({rect[0]},{rect[1]}) {rect[2]}x{rect[3]}："
            "用『目标区/障碍区/衬底区域』继续标注，或保存配置")

    @Slot(int)
    def on_edge_changed(self, v: int) -> None:
        if self._roi_cfg is not None:
            self._roi_cfg.edge_clearance_px = float(v)

    @Slot()
    def on_reset_view(self) -> None:
        if self._live_world is None:
            return
        if self._is_motor_mode():
            if self._cam_frame_size is None:
                return
            vw, vh = self._cam_frame_size
            # ROI 配置同步复位为全画幅（此前只复位了世界窗口，运行仍用旧 ROI）
            self._roi_cfg = RoiConfig(
                roi=(0, 0, vw, vh), video=self._roi_cfg.video,
                px_per_mm=self._roi_cfg.px_per_mm,
                edge_clearance_px=self._roi_cfg.edge_clearance_px,
                zones=self._roi_cfg.zones)
            self._roi_defaulted = True
            self.detail_label.setText(
                "视野已复位为全画幅：请用『ROI 视野』重新框定检测视野")
        else:
            vw, vh = video_sim._video_size()
        self._live_world.window = (vw, vh)
        self._live_world.offset = [0.0, 0.0]

    @Slot()
    def on_undo_zone(self) -> None:
        """撤销区域：按当前画框模式过滤，只撤销该类型最新创建的一个对象。

        需求：与"画框模式"指定的对象一致——例如模式为"目标点(避障)"时，
        仅从画框顺序栈中撤销最新建的一个目标点，而非无条件撤销栈顶对象。
        """
        if not self._is_motor_mode():
            c = self._sim_cfg
            # 当前模式 -> 允许撤销的对象类型集合
            wanted = self._undo_kinds_for_mode()
            if self._sim_order:          # 有顺序信息：后画先撤（按类型过滤）
                idx = next(
                    (i for i in range(len(self._sim_order) - 1, -1, -1)
                     if self._sim_order[i][0] in wanted), None)
                if idx is None:
                    self._simlog(
                        f"无可撤销的[{self.mode_combo.currentText()}]对象")
                    return
                entry = self._sim_order.pop(idx)
                self._undo_sim_entry(entry)
            else:   # 载入配置无顺序信息 -> 按当前模式固定优先级删对应类型
                self._undo_sim_loaded(wanted)
            self._save_sim_config()
            self._update_sim_zones_label()
            self._refresh_sim_preview()
            return
        if self._roi_cfg and self._roi_cfg.zones:
            z = self._roi_cfg.zones.pop()
            self._zone_ref_shift.pop(str(z.name), None)
            if z.kind == "substrate":
                self._sync_substrate_zone()   # 撤销衬底标注：恢复默认可行域
            self.zones_label.setText(f"zones: {len(self._roi_cfg.zones)}")

    def _undo_kinds_for_mode(self) -> set:
        """当前画框模式 -> 可撤销对象类型集合。"""
        mode = self.mode_combo.currentText()
        return {
            "衬底(ground)": {"ground_add"},
            "圆球(mask)": {"ball_add"},
            "障碍物(obstacle)": {"obstacle_add"},
            "目标点(避障)": {"ground_goal"},
            "目标范围(组装)": {"ground_range"},
        }.get(mode, {"ground_add", "ball_add", "obstacle_add",
                     "ground_goal", "ground_range"})

    def _undo_sim_entry(self, entry: tuple) -> None:
        """撤销一条画框记录（按类型回退 _sim_cfg 中的对应对象）。"""
        c = self._sim_cfg
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
            # 先同步弹出平行多边形列表，保持与 obstacles 对齐
            polys = c.setdefault("obstacle_polys", [])
            poly = (polys.pop() if len(polys) == len(c["obstacles"])
                    else None)
            o = c["obstacles"].pop()
            oid_ = (self._sim_ids["obstacles"].pop()
                    if self._sim_ids["obstacles"] else "?")
            self._simlog(f"撤销障碍物 {oid_}: pos={tuple(o[:2])} "
                         f"size={o[2]}x{o[3]}"
                         + (f" poly={len(poly)}边形" if poly else ""))
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

    def _undo_sim_loaded(self, wanted: set) -> None:
        """无顺序信息时：按当前模式想要的类型，从后往前删除第一个匹配对象。"""
        c = self._sim_cfg
        for gi in range(len(c["grounds"]) - 1, -1, -1):
            if "ground_goal" in wanted and c["grounds"][gi].get("goal"):
                gp = c["grounds"][gi]["goal"]
                c["grounds"][gi]["goal"] = None
                self._simlog(f"撤销目标点 "
                             f"{c['grounds'][gi].get('id','?')}: "
                             f"{tuple(gp) if gp else '?'}")
                return
            if "ground_range" in wanted and c["grounds"][gi].get("goal_range"):
                gr = c["grounds"][gi]["goal_range"]
                c["grounds"][gi]["goal_range"] = None
                self._simlog(f"撤销目标范围 "
                             f"{c['grounds'][gi].get('id','?')}: "
                             f"{tuple(gr[:2]) if gr else '?'}")
                return
        if "ball_add" in wanted and c["balls"]:
            b = c["balls"].pop()
            bid = (self._sim_ids["balls"].pop()
                   if self._sim_ids["balls"] else "?")
            self._simlog(f"撤销圆球 {bid}: pos={tuple(b[:2])} "
                         f"size={b[2]}x{b[3]}")
            return
        if "obstacle_add" in wanted and c["obstacles"]:
            polys = c.setdefault("obstacle_polys", [])
            poly = (polys.pop() if len(polys) ==
                    len(c["obstacles"]) else None)
            o = c["obstacles"].pop()
            oid_ = (self._sim_ids["obstacles"].pop()
                    if self._sim_ids["obstacles"] else "?")
            self._simlog(f"撤销障碍物 {oid_}: pos={tuple(o[:2])} "
                         f"size={o[2]}x{o[3]}"
                         + (f" poly={len(poly)}边形" if poly else ""))
            return
        if "ground_add" in wanted and c["grounds"]:
            g = c["grounds"].pop()
            self._simlog(f"撤销衬底 {g.get('id','?')}: "
                         f"pos={tuple(g['rect'][:2])} "
                         f"size={g['rect'][2]}x{g['rect'][3]}")
            return
        self._simlog(f"无可撤销的[{self.mode_combo.currentText()}]对象")

    @Slot()
    def on_save_config(self) -> None:
        if self._roi_cfg is None:
            return
        try:
            vsz = self._config_frame_size()
            self._roi_cfg.validate(vsz)
            p = self._roi_cfg.save(video_sim.DEFAULT_ROI_CONFIG)
            self.detail_label.setText(f"配置已保存: {p}")
        except ValueError as exc:
            self.detail_label.setText(f"保存失败: {exc}")

    @Slot()
    def on_load_config(self) -> None:
        roi_defaulted = False
        try:
            cfg = RoiConfig.load(video_sim.DEFAULT_ROI_CONFIG)
            frame_size = self._config_frame_size()
            try:
                cfg.validate(frame_size)
            except ValueError:
                # An empty configuration has no coordinates whose meaning
                # could be lost. If the camera/screen capture size changed
                # since it was saved, adopt the current full frame instead of
                # rejecting a stale ROI. Configurations containing zones stay
                # strict so their absolute annotations are never shifted
                # silently.
                if not cfg.zones and frame_size:
                    fw, fh = (int(frame_size[0]), int(frame_size[1]))
                    cfg = RoiConfig(
                        roi=(0, 0, fw, fh), zones=[], video=cfg.video,
                        px_per_mm=cfg.px_per_mm,
                        edge_clearance_px=cfg.edge_clearance_px)
                    cfg.validate(frame_size)
                    roi_defaulted = True
                    self.detail_label.setText(
                        f"空配置已按当前帧源重置为全画幅 ({fw}x{fh})")
                else:
                    raise
        except Exception as exc:  # noqa: BLE001
            self.detail_label.setText(f"载入失败: {exc}")
            return
        # Do not retain a previous fallback marker after a normal load.
        self._roi_defaulted = roi_defaulted
        self._roi_cfg = cfg
        self.zones_label.setText(f"zones: {len(cfg.zones)}")
        self.edge_spin.blockSignals(True)
        self.edge_spin.setValue(int(cfg.edge_clearance_px))
        self.edge_spin.blockSignals(False)
        if self._live_world is not None:
            self._apply_roi(tuple(cfg.roi))
        if roi_defaulted:
            fw, fh = int(cfg.roi[2]), int(cfg.roi[3])
            self.detail_label.setText(
                f"空配置已载入，ROI按当前帧源重置为全画幅 ({fw}x{fh})")
        else:
            self.detail_label.setText("配置已载入")


    # ---------------- preview
    @Slot()
    def on_scenario_changed(self) -> None:
        """显示所选场景初始帧；权重未就绪时 video 场景转后台构建。"""
        if self._is_motor_mode():
            return   # 电机模式预览由 _refresh_mode_preview 处理
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
        if not self._is_motor_mode():
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

    def _draw_rulers(self, frame: np.ndarray,
                     um_per_px: Optional[float] = None) -> np.ndarray:
        """画布左上角为原点的 x/y 刻度尺：顶部 x 轴、左侧 y 轴。

        主刻度 100px（带数值标签），次刻度 25px；线色黄、字带黑描边，
        深浅背景下均可读。um_per_px 提供像素当量时标签用物理单位（um），
        否则用像素（px）。
        """
        out = frame.copy()
        h, w = out.shape[:2]
        major, minor = 100, 25
        color, txt_bg, txt_fg = (60, 220, 230), (0, 0, 0), (240, 240, 240)

        def _label(v_px: float) -> str:
            if um_per_px:
                return f"{v_px * um_per_px:g}um"
            return f"{v_px:g}px"

        cv2.line(out, (0, 0), (w - 1, 0), color, 1)
        cv2.line(out, (0, 0), (0, h - 1), color, 1)
        for x in range(0, w, minor):
            major_tick = x % major == 0
            cv2.line(out, (x, 0), (x, 12 if major_tick else 6), color, 1)
            if major_tick:
                text = _label(x)
                org = (x + 3, 20)
                cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                            0.32, txt_bg, 2, cv2.LINE_AA)
                cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                            0.32, txt_fg, 1, cv2.LINE_AA)
        for y in range(0, h, minor):
            major_tick = y % major == 0
            cv2.line(out, (0, y), (12 if major_tick else 6, y), color, 1)
            if major_tick:
                text = _label(y)
                org = (14, y + 4)
                cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                            0.32, txt_bg, 2, cv2.LINE_AA)
                cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                            0.32, txt_fg, 1, cv2.LINE_AA)
        return out

    def _ruler_um_per_px(self) -> Optional[float]:
        """刻度尺物理单位换算（um/px）；未知像素当量时返回 None（退回 px）。

        虚拟模式：仿真镜头像素当量固定 0.5um/px（SIM_PIXEL_SIZE）；
        电机模式：取 ROI 配置的 px_per_mm（1px = 1000/px_per_mm um）。
        """
        if not self._is_motor_mode():
            return float(sim_microscope.SIM_PIXEL_SIZE)
        ppm = getattr(getattr(self, "_roi_cfg", None), "px_per_mm", None)
        if ppm:
            return 1000.0 / float(ppm)
        return None

    def _draw_beam_spot(self, bgr: np.ndarray) -> np.ndarray:
        """标定后叠加固定激光光斑（5px 绿色实心点，Alg2）。

        位置即「激光位置」标定值（点击画面标定 / 设为画面中心）。只有光斑
        打在圆球上时控制器才会吸住该球并驱动其移动（FixedBeamController
        的光镊捕获判据），脱靶时先动态再捕获把球心拉回光斑中心。
        """
        combo = getattr(self, "alg_combo", None)
        if (combo is None or combo.currentText() != "Alg2"
                or not getattr(self, "_beam_calibrated", False)):
            return bgr
        x = int(round(self.beam_x_spin.value()))
        y = int(round(self.beam_y_spin.value()))
        h, w = bgr.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return bgr
        cv2.circle(bgr, (x, y), 5, (0, 255, 0), -1)
        return bgr

    def _refresh_beam_spot(self, *_args) -> None:
        """激光位置标定/变更后立即重绘当前帧，让光斑马上出现。"""
        frame = getattr(self, "_last_frame", None)
        if frame is not None:
            self._show_frame(frame)

    def _show_frame(self, bgr: np.ndarray) -> None:
        self._last_frame = bgr.copy()
        bgr = self._draw_beam_spot(bgr)
        if (getattr(self, "xyz_ruler_chk", None) is not None
                and self.xyz_ruler_chk.isChecked()):
            bgr = self._draw_rulers(bgr, self._ruler_um_per_px())
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
        # 切换模式：停止实时检测并丢弃旧世界（虚拟/电机世界不通用）
        if self.live_btn.isChecked():
            self.live_btn.setChecked(False)
        self._live_world = None
        self._live_dets = []
        self.auto_recognition_box.setEnabled(
            self.alg_combo.currentText() == "Alg2")
        self._reset_auto_recognition(
            "开始实时检测后自动识别" if motor else
            "请切换到电机模式并开始实时检测")
        if not motor:
            self._close_motor_xyz_stage()
        self._switch_draw_modes(motor)
        self.sim_spec_btn.setVisible(not motor)   # 仿真图层仅虚拟模式可用
        self._refresh_mode_preview()
        if motor and self.driver_combo.currentIndex() == 0:
            self._refresh_usb_count()
        elif motor and self.driver_combo.currentIndex() == 2:
            self._refresh_kinesis_devices()

    def _using_screen_source(self) -> bool:
        """电机模式帧源是否为屏幕区域捕获。"""
        return self._is_motor_mode() and self.src_combo.currentIndex() == 1

    def _invalidate_live(self) -> None:
        """帧源参数变化：停止实时流并丢弃旧世界（下次开启时重建）。"""
        if self.live_btn.isChecked():
            self.live_btn.setChecked(False)
        self._live_world = None
        self._live_dets = []
        self._reset_auto_recognition("帧源已变化，请重新开始实时检测")

    @Slot(int)
    def on_frame_source_changed(self, idx: int) -> None:
        cam = idx == 0
        self.cam_spin.setVisible(cam)
        for wdg in self.screen_widgets:
            wdg.setVisible(not cam)
        self._invalidate_live()

    @Slot()
    def on_select_screen_region(self) -> None:
        """全屏半透明覆盖层框选屏幕区域（虚拟桌面全局坐标）。"""
        dlg = ScreenRegionOverlay()
        dlg.exec_()
        if dlg.region:
            self._screen_region = dlg.region
            x, y, w, h = dlg.region
            self.region_label.setText(f"({x},{y}) {w}x{h}")
            self.region_label.setStyleSheet("color:#0a0;")
            self._invalidate_live()

    def _switch_draw_modes(self, motor: bool) -> None:
        """画框模式选项按模式过滤：电机=ROI/区域，虚拟=sim 对象绘制。"""
        items = (["ROI 视野", "目标区", "障碍区", "自由区",
                  "衬底区域"] if motor else
                 ["圆球(mask)", "衬底(ground)", "障碍物(obstacle)",
                  "目标点(避障)", "目标范围(组装)"])
        self.mode_combo.blockSignals(True)
        self.mode_combo.clear()
        self.mode_combo.addItems(items)
        self.mode_combo.blockSignals(False)

    def _refresh_mode_preview(self) -> None:
        """模式切换后的画布预览：电机=提示帧，虚拟=sim01 初始帧。"""
        if self._is_motor_mode():
            w, h = self._cam_frame_size or (640, 480)
            frame = np.full((h, w, 3), 40, np.uint8)
            cv2.putText(frame, "Motor mode: start LIVE to preview camera",
                        (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 200, 255), 2)
            self._show_frame(frame)
        else:
            self.on_scenario_changed()

    @Slot(int)
    def on_driver_changed(self, idx: int) -> None:
        self._close_motor_xyz_stage()
        self._apply_driver_visibility(idx)
        if idx == 0:
            self._refresh_usb_count()
        elif idx == 2:
            self._refresh_kinesis_devices()

    def _apply_driver_visibility(self, idx: int) -> None:
        """按驱动显示/隐藏参数控件（0=8742/8743，1=串口，2=Kinesis）。"""
        pico = idx == 0
        kinesis = idx == 2
        for w in self.pico_widgets:
            w.setVisible(pico)
        for w in self.serial_widgets:
            w.setVisible(idx == 1)
        # 步进类驱动（step 计量的 8742/8743 与 KIM101）共用运动参数
        for w in self.motion_widgets:
            w.setVisible(pico or kinesis)
        for w in self.kinesis_widgets:
            w.setVisible(kinesis)

    def _refresh_usb_count(self) -> None:
        """后台枚举 Picomotor 控制器、ID 和可用轴（扫描可能耗时）。"""
        self.pico_count_label.setText("USB 设备: 检测中...")
        self.controller_id_label.setText("控制器: 检测中...")

        def _probe():
            if __package__ in (None, ""):   # 直接运行 app.py 无包上下文
                from obstacle_avoidance.stages import PicoMotorStage
            else:
                from .stages import PicoMotorStage
            try:
                records = PicoMotorStage.scan_usb_controllers()
                n = sum(1 for rec in records if rec.get("ok"))
            except Exception as exc:  # noqa: BLE001 - pylablib missing/no device
                records = [{"conn": 0, "id": "", "axes": [],
                            "ok": False, "error": str(exc)}]
                n = 0
            if not getattr(self, "_ui_closing", False):
                try:
                    self.usb_count_ready.emit(n)
                    self.controller_scan_ready.emit(records)
                except RuntimeError:
                    # The window may have been destroyed while the probe was
                    # still connecting to a controller.
                    pass

        threading.Thread(target=_probe, daemon=True,
                         name="pico-usb-probe").start()

    @Slot(object)
    def _on_controller_scan(self, records) -> None:
        """Apply the first detected controller and map its first 3 axes to XYZ."""
        self._controller_scan = list(records or [])
        good = [rec for rec in self._controller_scan if rec.get("ok")]
        if not good:
            error = (self._controller_scan[0].get("error")
                     if self._controller_scan else "未发现 USB 8742/8743")
            self._selected_controller = None
            self.controller_id_label.setText(f"控制器: 未检测 ({error})")
            self.controller_id_label.setStyleSheet("color:#c00;")
            return
        selected = good[0]
        self._close_motor_xyz_stage()
        self._selected_controller = selected
        self.conn_spin.setValue(int(selected.get("conn", 0)))
        axes = sorted({int(a) for a in selected.get("axes", [])})
        if len(axes) >= 3:
            self.axis_x_spin.setValue(axes[0])
            self.axis_y_spin.setValue(axes[1])
            self.axis_z_spin.setValue(axes[2])
        elif len(axes) >= 2:
            self.axis_x_spin.setValue(axes[0])
            self.axis_y_spin.setValue(axes[1])
            self.controller_id_label.setStyleSheet("color:#c60;")
        self.controller_id_label.setText(
            f"控制器: {selected.get('id') or '8742/8743'} | "
            f"USB {selected.get('conn')} | 可用轴 {axes or '?'}")
        if len(axes) >= 3:
            self.controller_id_label.setStyleSheet("color:#080;")
        self._simlog(
            "控制器检测: id={} USB={} axes={} -> X{} Y{} Z{}".format(
                selected.get("id") or "8742/8743", selected.get("conn"),
                axes or "?", self.axis_x_spin.value(), self.axis_y_spin.value(),
                self.axis_z_spin.value()))

    def _refresh_kinesis_devices(self) -> None:
        """Scan Kinesis DeviceManagerCLI in a worker thread."""
        self.kinesis_id_label.setText("Kinesis: 检测中...")

        def _probe():
            if __package__ in (None, ""):   # 直接运行 app.py 无包上下文
                from obstacle_avoidance.stages import KinesisKIM101Stage
            else:
                from .stages import KinesisKIM101Stage
            records = KinesisKIM101Stage.detect_devices()
            if not getattr(self, "_ui_closing", False):
                try:
                    self.kinesis_scan_ready.emit(records)
                except RuntimeError:
                    pass

        threading.Thread(target=_probe, daemon=True,
                         name="kinesis-device-probe").start()

    @Slot(object)
    def _on_kinesis_scan(self, records) -> None:
        records = list(records or [])
        self._kinesis_devices = records
        good = [r for r in records if r.get("ok") and r.get("serial")]
        if not good:
            error = records[0].get("error") if records else "未发现 Kinesis KIM101"
            self.kinesis_id_label.setText(f"Kinesis: 未检测 ({error})")
            self.kinesis_id_label.setStyleSheet("color:#c00;")
            return
        # KIM101 is a single controller; never interpret multiple detected
        # devices as X/Y/Z serial fields.  Select the first valid device and
        # keep the remaining records available for diagnostics.
        selected = good[0]
        serial = str(selected["serial"])
        self.kinesis_serial_edit.setText(serial)
        desc = ", ".join(str(r.get("serial")) for r in good)
        self.kinesis_id_label.setText(
            f"Kinesis: KIM101 {serial} 通道自动映射 X/Y/Z→Ch1/2/3" +
            (f"；另有 {len(good) - 1} 台" if len(good) > 1 else ""))
        self.kinesis_id_label.setStyleSheet("color:#080;")
        self._simlog(f"Kinesis 检测: {desc} -> 选用控制器 {serial}")

    def _frame_source_desc(self) -> str:
        """当前帧源的可读描述（电机模式）。"""
        if self._using_screen_source():
            if self._screen_region is None:
                return "屏幕区域（未框选）"
            x, y, w, h = self._screen_region
            return f"屏幕区域 ({x},{y}) {w}x{h}"
        return f"相机 index={self.cam_spin.value()}"

    def _kinesis_serial(self) -> str:
        """Return the single KIM101 controller serial configured in the UI."""
        return self.kinesis_serial_edit.text().strip()

    def _step_mm(self) -> float:
        """「单步位移(step)」× steps/mm → mm（下发与单步限位统一用 mm）。"""
        spm = float(self.spm_spin.value())
        if spm <= 0:
            raise RuntimeError("steps/mm 必须大于 0")
        return float(self.step_steps_spin.value()) / spm

    def _update_step_mm_hint(self) -> None:
        spm = float(self.spm_spin.value())
        mm = float(self.step_steps_spin.value()) / spm if spm > 0 else 0.0
        self.step_mm_hint.setText(f"→ {mm:.4f} mm @ {spm:.1f} steps/mm")

    def _step_desc(self, motor_cfg: dict) -> str:
        """运行提示里的单步位移（step 驱动附 step↔mm 换算）。"""
        mm = float(motor_cfg.get("max_step_mm", 0.0))
        if motor_cfg.get("driver") == "serial":
            return f"{mm:.4f}mm"
        return f"{self.step_steps_spin.value()} step ≈ {mm:.4f}mm"

    def _motor_params(self):
        """电机模式参数预检：配置完整性 + 硬件确认门控。"""
        cfg = self._roi_cfg
        if cfg is None or cfg.goal_zone() is None:
            raise RuntimeError("电机模式需要含目标区的 ROI 配置："
                               "先在实时检测里画 ROI/目标区并保存")
        if not self.confirm_chk.isChecked():
            raise RuntimeError("安全门控：请勾选'我确认已连接真实电机'后再运行")
        auto_recognition = self._auto_recognition_active()
        if auto_recognition:
            if not self._auto_recognition_confirmed or self._auto_confirmed_result is None:
                raise RuntimeError(
                    "Alg2 自动识别尚未确认：请开始实时检测，等待连续稳定 3 帧，"
                    "再点击“确认识别结果”")
            result = self._auto_confirmed_result
            if (result.uncertain or
                    result.overall_confidence < self.auto_confidence_spin.value()):
                raise RuntimeError("Alg2 自动识别结果已失效，请重新检测并确认")
        if self.alg_combo.currentText() == "Alg2" and not self._beam_calibrated:
            raise RuntimeError(
                "Alg2 固定激光位置尚未标定：请开始实时检测后点击“设为画面中心”")
        shift_sign = 1 if self.shift_sign_combo.currentIndex() == 0 else -1
        # 单步位移：step 驱动（8742/8743、Kinesis）按 step→mm 换算；
        # 串口 G 代码台位本身以 mm 计，直接用 mm 框。
        serial_driver = self.driver_combo.currentIndex() == 1
        step_mm = (float(self.step_mm_spin.value()) if serial_driver
                   else self._step_mm())
        common = {"ball_shift_sign": shift_sign, "confirmed": True,
                  "algorithm": self.alg_combo.currentText(),
                  "max_step_mm": step_mm,
                  "auto_recognition": auto_recognition,
                  "recognition_min_confidence":
                      float(self.auto_confidence_spin.value())}
        if self.alg_combo.currentText() == "Alg2":
            common.update({
                "beam_position_px": (self.beam_x_spin.value(),
                                     self.beam_y_spin.value()),
                "beam_confidence": 1.0,
                "beam_source": "ui_manual",
            })
            if (self._selected_auto_track_id is not None and
                    self._auto_confirmed_result is not None):
                selected = next(
                    (p for p in self._auto_confirmed_result.particles
                     if p.track_id == self._selected_auto_track_id), None)
                if selected is not None:
                    common["target_hint_px"] = tuple(selected.position_px)
        if self._using_screen_source():
            if self._screen_region is None:
                raise RuntimeError("屏幕区域帧源：请先点击'框选屏幕区域'")
            common["frame_source"] = video_sim.screen_source(
                region=self._screen_region)
        else:
            common["camera_index"] = self.cam_spin.value()
        if self.driver_combo.currentIndex() == 0:   # 8742/8743 Picomotor
            axis_map = (self.axis_x_spin.value(), self.axis_y_spin.value(),
                        self.axis_z_spin.value())
            if len(set(axis_map)) != 3:
                raise RuntimeError("8742/8743 的 X/Y/Z 轴号必须一一对应且不能重复")
            if (self._selected_controller and
                    int(self._selected_controller.get("conn", -1)) ==
                    self.conn_spin.value()):
                available = {int(a) for a in
                             self._selected_controller.get("axes", [])}
                missing = [axis for axis in axis_map if axis not in available]
                if available and missing:
                    raise RuntimeError(
                        f"控制器仅提供轴 {sorted(available)}，映射轴 {missing} 不可用")
            axis_steps = {
                axis: float(self._xyz_axis_profiles.get(axis, {}).get(
                    "steps_per_unit", self.spm_spin.value() / 1000.0)) * 1000.0
                for axis in ("x", "y", "z")}
            motor = {"driver": "picomotor",
                     "conn": self.conn_spin.value(),
                     "x_axis": self.axis_x_spin.value(),
                     "y_axis": self.axis_y_spin.value(),
                     "z_axis": self.axis_z_spin.value(),
                     "steps_per_mm": self.spm_spin.value(),
                     "steps_per_mm_by_axis": axis_steps,
                     "speed_steps": (self.speed_spin.value() or None),
                     "accel_steps": (self.accel_steps_spin.value() or None),
                     **common}
        elif self.driver_combo.currentIndex() == 2:  # Thorlabs Kinesis KIM101
            serial = self._kinesis_serial()
            if not serial:
                raise RuntimeError("请输入或检测 KIM101 控制器序列号")
            detected = {str(r.get("serial")) for r in self._kinesis_devices
                        if r.get("ok") and r.get("serial")}
            if detected and serial not in detected:
                raise RuntimeError(
                    f"Kinesis 序列号未在检测列表中: {serial}")
            # 通道不做手工配置：驱动连接该序列号的控制器后，按可用通道
            # 自动完成 X/Y/Z 映射（详见 KinesisKIM101Stage._detect_channels）。
            motor = {"driver": "kinesis",
                     "serial_no": serial,
                     "steps_per_mm": self.spm_spin.value(),
                     "steps_per_mm_by_axis": {
                         axis: float(self._xyz_axis_profiles.get(axis, {}).get(
                             "steps_per_unit", self.spm_spin.value() / 1000.0)) * 1000.0
                         for axis in ("x", "y", "z")},
                     "speed_steps": (self.speed_spin.value() or None),
                     "accel_steps": (self.accel_steps_spin.value() or None),
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
        self._run_plan_pts = []               # 新一轮运行：清空上轮残余路径
        scenario = self.scenario_combo.currentText()
        motor = self.mode_sel.currentIndex() == 1
        if motor:
            try:
                # Manual XYZ jog and a closed-loop run must not hold the same
                # USB controller concurrently.  Reopen it in WorkerThread.
                self._close_motor_xyz_stage()
                roi_cfg, motor_cfg = self._motor_params()
                self._controller_ref = {}
                builder = lambda: video_sim.build_video_scenario(
                    "video03", config=roi_cfg, motor=motor_cfg,
                    task_mode=mode, controller_sink=self._controller_ref,
                    auto_recognition=bool(
                        motor_cfg.get("auto_recognition", False)),
                    cfg_overrides={
                        "max_step_mm": motor_cfg["max_step_mm"],
                        "max_iterations": self.iters_spin.value()})
            except Exception as exc:  # noqa: BLE001
                self.detail_label.setText(str(exc))
                return
            self.state_label.setText(f"RUNNING video03 [{mode}]")
            driver_desc = ("Kinesis" if motor_cfg.get("driver") == "kinesis"
                           else ("Picomotor" if motor_cfg.get("driver") == "picomotor"
                                 else "串口"))
            if motor_cfg.get("driver") == "kinesis":
                axis_desc = (f"KIM101 {motor_cfg.get('serial_no', '')} "
                             "通道自动映射(X/Y/Z→Ch1/2/3)")
            elif motor_cfg.get("driver") == "picomotor":
                axis_desc = (f"X{motor_cfg.get('x_axis', '相机')} "
                             f"Y{motor_cfg.get('y_axis', '相机')} "
                             f"Z{motor_cfg.get('z_axis', '未用')}")
            else:
                axis_desc = "相机/串口"
            self.detail_label.setText(
                f"驱动: {driver_desc} | 帧源: {self._frame_source_desc()} | "
                f"轴映射 {axis_desc} | "
                f"自动识别 {'ON' if motor_cfg.get('auto_recognition') else 'OFF'} | "
                f"单步 {self._step_desc(motor_cfg)} | "
                f"速度 {motor_cfg.get('speed_steps') or '不改'} step/s | "
                f"加速度 {motor_cfg.get('accel_steps') or '不改'} step/s² | "
                f"最大步数 {self.iters_spin.value()}")
            self._simlog(self.detail_label.text())
            self.worker = WorkerThread("video03", self._controller_ref,
                                       builder=builder)
            self.worker._execution_mode = "motor"
            self.worker._algorithm = self.alg_combo.currentText()
            self.worker.frame_ready.connect(self.on_frame)
            self.worker.state_ready.connect(self.on_state)
            self.worker.metrics_ready.connect(self.on_metrics)
            self.worker.motion_ready.connect(self.on_motion)
            self.worker.finished_run.connect(self.on_done)
            self.worker.error_occurred.connect(self.on_error_dialog)
            self.worker.log_ready.connect(self.on_log_line)
            self.worker.report_ready.connect(self.on_report_ready)
            self.worker.plan_ready.connect(self.on_plan_ready)   # 实时路径标注
            self.worker.start()
            return
        c = self._sim_cfg

        # ---- 校验：每个 ground 对应一个目标
        if not c["grounds"]:
            self.detail_label.setText("⚠ 请先画衬底（至少一个）")
            return
        if not c["balls"]:
            self.detail_label.setText("⚠ 请先画圆球(mask)（至少一个）")
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
        # _sim_cfg 存样本绝对坐标；worker 世界按用户当前视野定位
        # （WYSIWYG：运行窗口 = 所见窗口），因此转回当前视窗窗口坐标。
        # 无实时世界（未进入虚拟模式）时退回初始原点（样本中心）。
        # 需求1：启动后若还挂着"元素不在视窗内"的待对齐请求，先建立实时世界
        # 并对齐视窗，再取运行原点（保证运行窗口 = 所见窗口）。
        if self._sim_live is None and self._pending_view_center is not None:
            try:
                self._ensure_live()
            except Exception as exc:  # noqa: BLE001 - 建世界失败沿用初始原点
                _log_exception("sim live init failed", exc)
        if self._sim_live is not None:
            pos = self._sim_live.micro_stage.position
            origin_um = (float(pos["x"]), float(pos["y"]))
            o = self._sim_origin()
        else:
            origin_um = None
            o = self._sim_initial_origin()
        self._run_origin = o
        layout = {
            "balls": [self._rect_s2w(b, o) for b in c["balls"]],
            "obstacles": [self._rect_s2w(ob, o) for ob in c["obstacles"]],
            "grounds": [self._rect_s2w(g["rect"], o) for g in c["grounds"]],
            # Free 多边形（窗口坐标顶点，平行列表；None=矩形对象）
            "ground_polys": [
                ([[px - o[0], py - o[1]] for px, py in g["poly"]]
                 if g.get("poly") else None) for g in c["grounds"]],
            "obstacle_polys": [
                ([[px - o[0], py - o[1]] for px, py in poly]
                 if poly else None) for poly in (c.get("obstacle_polys") or [])],
            "ground_goals": [
                ([int(g["goal"][0] - o[0]), int(g["goal"][1] - o[1])]
                 if g.get("goal") else None) for g in c["grounds"]],
            "ground_goal_ranges": [
                (self._rect_s2w(g["goal_range"], o)
                 if g.get("goal_range") else None) for g in c["grounds"]],
            "run_mode": mode,
        }
        try:
            validate_sim_layout(layout, mode=mode)
        except LayoutValidationError as exc:
            self.detail_label.setText(str(exc))
            return

        self.state_label.setText(f"RUNNING sim01 [{mode}]")
        alg = self.alg_combo.currentText()
        if alg not in self.ALGORITHMS:
            self.detail_label.setText(f"⚠ 未知算法 {alg!r}（未注册）")
            return
        self._simlog(f"启动运行 sim01 [{mode}] 算法={alg}: "
                     f"球{len(layout['balls'])} "
                     f"衬底{len(layout['grounds'])} "
                     f"障碍{len(layout['obstacles'])}")
        self.worker = WorkerThread(scenario, {}, builder=None,
                                   sample_spec=self._load_sample_spec())
        self.worker.selected_ball_index = self.selected_ball_index
        self.worker._execution_mode = "virtual"
        self.worker._sim_layout = layout       # 直接注入 WorkerThread
        self.worker._run_mode = mode
        self.worker._algorithm = alg
        self.worker._motion_params = {         # 球步长/速度/加速度（XYZ 页）
            "step_mm": self.ball_step_spin.value(),
            "speed_ums": self.ball_speed_spin.value(),
            "accel_ums2": self.ball_accel_spin.value(),
            "steps_per_mm": self.spm_spin.value(),
            "beam_position_px": (self.beam_x_spin.value(),
                                 self.beam_y_spin.value())}
        self.worker._origin_um = origin_um     # 运行视窗原点（WYSIWYG）
        self.worker.layout_updated.connect(self.on_layout_updated)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.state_ready.connect(self.on_state)
        self.worker.metrics_ready.connect(self.on_metrics)
        self.worker.motion_ready.connect(self.on_motion)
        self.worker.finished_run.connect(self.on_done)
        self.worker.error_occurred.connect(self.on_error_dialog)
        self.worker.log_ready.connect(self.on_log_line)   # 需求5
        self.worker.report_ready.connect(self.on_report_ready)  # 回放用
        self.worker.stage_ready.connect(self.on_stage_ready)    # 需求3
        self.worker.start()

    @Slot(str)
    def on_report_ready(self, path: str) -> None:
        """运行结束：自动填入本轮报告路径，回放一键可用。"""
        self.report_path.setText(path)
        self._simlog(f"报告已保存: {path}")

    @Slot(list)
    def on_plan_ready(self, pts: list) -> None:
        """运行期收到最新规划路径：缓存，供实时画面叠加（与虚拟模式一致）。"""
        if pts:
            self._run_plan_pts = [(float(x), float(y)) for x, y in pts]

    @Slot(list)
    def on_layout_updated(self, balls_w: list) -> None:
        """移动完成：把 worker 回传的最终球位置（运行视窗坐标）写回
        _sim_cfg（转样本绝对坐标），预览中圆球出现在目标点。"""
        o = getattr(self, "_run_origin", None) or self._sim_initial_origin()
        changed = 0
        for i, rect in enumerate(balls_w):
            if i < len(self._sim_cfg["balls"]):
                new_rect = self._rect_w2s(rect, o)
                if new_rect != self._sim_cfg["balls"][i]:
                    self._sim_cfg["balls"][i] = new_rect
                    changed += 1
        if changed:
            # 需求1：运行结果立即落盘，避免"关闭程序后球位回退到运行前"
            self._save_sim_config()
            self._refresh_sim_preview()

    @Slot(np.ndarray)
    def on_frame(self, bgr: np.ndarray) -> None:
        if (not self._is_motor_mode() and
                not (self.worker is not None and self.worker.isRunning())):
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

    @Slot(dict)
    def on_stage_ready(self, position: dict) -> None:
        """需求3：暂存本轮运行最终台位，on_done 时同步到实时世界视窗。"""
        self._final_stage_position = dict(position)

    @Slot(int)
    def on_done(self, code: int) -> None:
        self.pause_btn.setText("暂停")
        self.state_label.setText(self.state_label.text() +
                                 (" [OK]" if code == 0 else " [FAIL]"))
        self._simlog(f"运行完成 exit={code}")
        # 需求3：画面停在最终达到的窗口状态（不回运行起点）——虚拟模式下把
        # 实时世界的台位同步到 worker 的最终台位并刷新预览。
        pos = getattr(self, "_final_stage_position", None)
        self._final_stage_position = None
        if not pos or self._is_motor_mode():
            return
        world = self._sim_live
        if world is None:
            return
        try:
            world.motion_stage.move_to(pos, source="run-final")
        except Exception as exc:  # noqa: BLE001 - 同步失败不影响结束
            self._simlog(f"最终窗口同步失败: {exc}")
            return
        self._simlog("运行结束：视窗停在最终窗口 "
                     f"({pos.get('x', 0):.0f},{pos.get('y', 0):.0f})µm")
        self._save_sim_config()    # 需求1：最终视窗与球位一并落盘
        self._refresh_sim_preview()

    @Slot()
    def on_replay(self) -> None:
        """回放：摘要文本 + 圆球移动过程动画（目标/路径/轨迹/球位）。"""
        path = self.report_path.text().strip()
        if not path or not os.path.exists(path):
            self.replay_out.setPlainText("报告文件不存在")
            return
        events = replay(path)
        s = summarize(events)
        self.replay_out.setPlainText("\n".join(
            f"{k}: {v}" for k, v in s.items()))
        reports_root = os.path.dirname(path)
        reports = [os.path.join(reports_root, f)
                   for f in os.listdir(reports_root)
                   if f.endswith(".jsonl")]
        stats = aggregate_experiments(reports)
        self.experiment_stats_out.setPlainText(
            json.dumps(stats, ensure_ascii=False, indent=2))
        self.experiment_canvas.set_data(stats)
        traj = extract_trajectory(events)
        self._rp_traj = traj
        self.replay_canvas.set_trajectory(traj)
        n = len(traj["frames"])
        if self._rp_timer is not None:
            self._rp_timer.stop()
        self.rp_play_btn.setText("播放")
        if n == 0:
            self.rp_slider.setRange(0, 0)
            self.rp_label.setText("报告中无检测帧（detection 事件）")
            self.rp_play_btn.setEnabled(False)
            self.rp_prev_btn.setEnabled(False)
            self.rp_next_btn.setEnabled(False)
            return
        self.rp_slider.setRange(0, n - 1)
        self.rp_slider.setValue(0)
        self.rp_label.setText(f"检测帧 1/{n}")
        self.rp_play_btn.setEnabled(True)
        self.rp_prev_btn.setEnabled(True)
        self.rp_next_btn.setEnabled(True)

    @Slot()
    def on_replay_last(self) -> None:
        """回放上一轮：自动加载 Reports/ 下最新的运行报告。"""
        root = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "Reports"))
        files = ([os.path.join(root, f) for f in os.listdir(root)
                  if f.endswith(".jsonl")] if os.path.isdir(root) else [])
        if not files:
            self.replay_out.setPlainText(
                "Reports/ 下暂无报告：先运行一轮避障/组装（报告自动保存）")
            return
        self.report_path.setText(max(files, key=os.path.getmtime))
        self.on_replay()

    def _rp_set_index(self, i: int) -> None:
        traj = getattr(self, "_rp_traj", None) or {}
        frames = traj.get("frames") or []
        if not frames:
            return
        i = max(0, min(int(i), len(frames) - 1))
        self.replay_canvas.set_index(i)
        self.rp_slider.blockSignals(True)
        self.rp_slider.setValue(i)
        self.rp_slider.blockSignals(False)
        self.rp_label.setText(f"检测帧 {i + 1}/{len(frames)}")

    @Slot()
    def on_replay_play(self) -> None:
        """播放/暂停回放动画；速度由速度下拉框控制。"""
        if self.rp_play_btn.text() == "暂停":
            if self._rp_timer is not None:
                self._rp_timer.stop()
            self.rp_play_btn.setText("播放")
            return
        if not (getattr(self, "_rp_traj", None) or {}).get("frames"):
            return
        if self._rp_timer is None:
            self._rp_timer = QtCore.QTimer(self)
            self._rp_timer.timeout.connect(self._rp_tick)
        interval = {0: 300, 1: 150, 2: 75, 3: 40}[
            self.rp_speed.currentIndex()]
        self._rp_timer.start(interval)
        self.rp_play_btn.setText("暂停")

    def _rp_tick(self) -> None:
        traj = self._rp_traj or {}
        n = len(traj.get("frames") or [])
        nxt = self.rp_slider.value() + 1
        if nxt >= n:
            if self._rp_timer is not None:
                self._rp_timer.stop()
            self.rp_play_btn.setText("播放")
            return
        self._rp_set_index(nxt)


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
