"""Alg2 fixed-beam calibration and stage-motion primitives.

The beam is fixed in camera coordinates.  With the beam off, stage motion moves
the complete sample until the selected ball reaches the calibrated beam point.
With the beam on, the selected ball remains under the beam while substrate,
obstacles and all other balls move with the stage.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from .models import (FailureReason, GoalRegion, Obstacle, Particle, Point,
                     RunState, StageCommand, TargetLock, TargetSelection,
                     WorkspaceSnapshot)
from .simulator import StageError, XYStageProtocol


@dataclass(frozen=True)
class Alg2Config:
    """XYZ setup used by Alg2 (micrometres)."""

    z_safe_um: float = 5.0
    z_focus_um: float = 0.0
    focus_settle_s: float = 0.0
    beam_position_px: Optional[Point] = None
    beam_calibration_confidence: float = 0.0
    image_shift_sign: int = -1
    # 对准前用探针实测『图像位移/样品位移』增益（px/mm），自动修正符号与比例。
    # 这里给的是探针步长的**上限**（mm）；实际步长按"图像位移≈一个球半径"
    # 由 px_per_mm 反推（见 FixedBeamController._probe_step_mm），避免固定
    # mm 步长在 px_per_mm 大的机型上把球一步推出画面。0 表示不标定，退回
    # image_shift_sign * px_per_mm。
    shift_probe_mm: float = 0.2
    # 实测位移小于该增益判为无效（检测抖动量级），该轴退回配置值。
    min_shift_gain_px_per_mm: float = 5.0


@dataclass(frozen=True)
class BeamCalibration:
    """Fixed beam position measured in the cropped camera frame."""

    position_px: Point
    confidence: float = 1.0
    source: str = "manual"
    timestamp: float = field(default_factory=time.time)

    def validate(self, frame_size: Optional[tuple[int, int]] = None) \
            -> "BeamCalibration":
        x, y = map(float, self.position_px)
        if not (math.isfinite(x) and math.isfinite(y)):
            raise StageError("beam calibration contains non-finite coordinates")
        if frame_size is not None:
            w, h = frame_size
            if not (0 <= x < w and 0 <= y < h):
                raise StageError(
                    f"beam position ({x:.1f},{y:.1f}) outside frame {w}x{h}")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise StageError("beam calibration confidence must be in [0, 1]")
        return self


class LaserGate:
    """Laser/shutter state adapter; a callback connects real hardware."""

    def __init__(self, set_enabled: Optional[Callable[[bool], None]] = None,
                 hardware_controlled: bool = False) -> None:
        self._callback = set_enabled
        self.hardware_controlled = bool(hardware_controlled)
        self.enabled = False
        self.history: list[tuple[float, bool]] = []

    def set_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if self._callback is not None:
            self._callback(enabled)
        self.enabled = enabled
        self.history.append((time.time(), enabled))

    def off(self) -> None:
        self.set_enabled(False)

    def on(self) -> None:
        self.set_enabled(True)


class Alg2Stage(XYStageProtocol):
    """XYStageProtocol facade backed by a real/simulated XYZ stage.

    The controller continues to issue small XY moves through the existing
    protocol.  ``prepare_focus`` explicitly exercises the Z axis before a task;
    the underlying microscope stage remains available for UI XYZ jogs and
    telemetry.
    """

    def __init__(self, world, track_id: Optional[int] = None,
                 config: Optional[Alg2Config] = None,
                 target_selection: Optional[TargetSelection] = None,
                 xy_stage=None, laser_gate: Optional[LaserGate] = None) -> None:
        self.world = world
        if target_selection is not None:
            if track_id is not None and int(track_id) != target_selection.track_id:
                raise StageError('track_id conflicts with target selection')
            track_id = target_selection.track_id
        self.track_id = track_id
        self.target_selection = target_selection
        self.target_lock = (TargetLock(target_selection) if target_selection is not None else None)
        self.config = config or Alg2Config()
        maker = getattr(world, "make_alg2_stage", None)
        if xy_stage is None:
            if not callable(maker):
                raise StageError("world does not provide fixed-beam Alg2 stage")
            xy_stage = maker(track_id)
        self._xy = xy_stage
        callback = getattr(world, "set_laser_enabled", None)
        self.laser_gate = laser_gate or LaserGate(
            callback if callable(callback) else None,
            hardware_controlled=callable(callback))
        beam = self.config.beam_position_px
        if beam is None:
            beam = getattr(world, "beam_position_px", None)
        if beam is None and getattr(world, "window", None):
            beam = (world.window[0] / 2.0, world.window[1] / 2.0)
        self.beam = (BeamCalibration(
            position_px=(float(beam[0]), float(beam[1])),
            confidence=float(self.config.beam_calibration_confidence or 1.0),
            source="configured").validate(getattr(world, "window", None))
                     if beam is not None else None)
        self._last: Optional[StageCommand] = None
        self._seq = 0
        self.workspace_shift_px: Point = tuple(
            getattr(world, "workspace_shift_px", (0.0, 0.0)))
        self.fixed_beam_mode = True
        # 实测增益 (px/mm, px/mm)；None 项表示该轴退回 image_shift_sign*px_per_mm
        self.shift_gain: tuple[Optional[float], Optional[float]] = (None, None)

    @classmethod
    def from_particle(cls, world, particle: Particle, config: Optional[Alg2Config] = None) -> 'Alg2Stage':
        return cls(world, target_selection=TargetSelection.from_particle(particle, 'alg2'), config=config)

    def update_target(self, particles: Sequence[Particle]) -> Optional[Particle]:
        if self.target_lock is None:
            return next((p for p in particles if p.track_id == self.track_id), None)
        particle = self.target_lock.update(particles)
        if particle is not None:
            self.track_id = particle.track_id
        return particle

    @property
    def target_state(self) -> Optional[str]:
        return self.target_lock.state.value if self.target_lock else None

    def prepare_focus(self, z_safe_um: Optional[float] = None,
                      z_focus_um: Optional[float] = None) -> None:
        """Z 安全高度/聚焦序列（可选按调用方覆盖高度）。

        电机模式由调用方传入 motor["z_safe_um"]/["z_focus_um"]；缺省时沿用
        config。真实 XYZ 台（PicoMotorStage/KinesisXYZStage）自带 prepare_focus，
        优先委派；仿真世界无该接口，回退到 world.motion_stage 显式走 Z。
        """
        if z_safe_um is None:
            z_safe_um = self.config.z_safe_um
        if z_focus_um is None:
            z_focus_um = self.config.z_focus_um
        delegate = getattr(self._xy, "prepare_focus", None)
        if callable(delegate):
            delegate(z_safe_um=float(z_safe_um), z_focus_um=float(z_focus_um))
            return
        stage = getattr(self.world, "motion_stage", None)
        if stage is None:
            return
        # Safe excursion followed by focus position makes Z motion explicit and
        # deterministic even when the stage starts already at focus.
        stage.enable()
        current = float(stage.position.get("z", 0.0))
        if abs(current - float(z_safe_um)) > 1e-9:
            stage.move_to({"z": float(z_safe_um)}, source="alg2-z-safe")
        if abs(float(z_focus_um) - float(z_safe_um)) > 1e-9:
            stage.move_to({"z": float(z_focus_um)}, source="alg2-z-focus")

    def set_laser_enabled(self, enabled: bool) -> None:
        self.laser_gate.set_enabled(enabled)

    def set_shift_gain(self, gain_px_per_mm) -> None:
        """写入探针实测的图像位移增益 (px/mm)；None/0 表示该轴沿用配置值。"""
        self.shift_gain = tuple(
            (float(g) if g is not None and abs(float(g)) >= 1e-9 else None)
            for g in (gain_px_per_mm[0], gain_px_per_mm[1]))

    @property
    def max_step_mm(self) -> Optional[float]:
        """底层台位的单步限位（UI「单步位移(mm)」）；无该属性的台位返回 None。"""
        return getattr(self._xy, "max_step_mm", None)

    def shift_workspace_by(self, dx_px: float, dy_px: float,
                           px_per_mm: float, **kwargs) -> bool:
        """Move all sample-bound objects by the requested image displacement."""

        sign = int(self.config.image_shift_sign)
        if sign not in (-1, 1):
            raise StageError("Alg2 image_shift_sign must be +1 or -1")
        raw: list[float] = []
        for value, gain in zip((dx_px, dy_px), self.shift_gain):
            # 实测增益优先：现场 image_shift_sign 配反会让球越走越远
            # （对准永远到不了位、电机一路朝同一方向走）。
            raw.append(float(value) / (gain if gain else sign * float(px_per_mm)))
        ok = self._xy.move_by(raw[0], raw[1], **kwargs)
        if ok:
            self._record_command(raw[0], raw[1], kwargs)
            self.note_image_shift(dx_px, dy_px)
        return bool(ok)

    def sample_move_mm(self, dx_mm: float, dy_mm: float, **kwargs) -> bool:
        """探针标定用：按 mm 直接移动样品，不做符号/增益换算。"""
        ok = self._xy.move_by(float(dx_mm), float(dy_mm), **kwargs)
        if ok:
            self._record_command(float(dx_mm), float(dy_mm), kwargs)
        return bool(ok)

    def note_image_shift(self, dx_px: float, dy_px: float) -> None:
        """同步一次『图像位移』台账（指令值或探针实测值）与世界。"""
        register = getattr(self.world, "register_workspace_shift", None)
        if callable(register):
            register(float(dx_px), float(dy_px))
        self.workspace_shift_px = (
            self.workspace_shift_px[0] + float(dx_px),
            self.workspace_shift_px[1] + float(dy_px))

    def _record_command(self, raw_dx: float, raw_dy: float, kwargs) -> None:
        self._seq += 1
        self._last = StageCommand(
            seq=self._seq, timestamp=time.time(), dx_mm=raw_dx,
            dy_mm=raw_dy, task_id=str(kwargs.get("task_id", "")),
            track_id=int(kwargs.get("track_id", -1)),
            waypoint_index=int(kwargs.get("waypoint_index", -1)),
            frame_id=int(kwargs.get("frame_id", -1)),
            source_plan_version=int(kwargs.get("plan_version", -1)))

    @property
    def beam_position_px(self) -> Point:
        if self.beam is None:
            raise StageError("Alg2 beam position has not been calibrated")
        return self.beam.position_px

    def sample_point_to_camera(self, point: Point) -> Point:
        return (float(point[0]) + self.workspace_shift_px[0],
                float(point[1]) + self.workspace_shift_px[1])

    def move_by(self, dx_mm: float, dy_mm: float, **kwargs) -> bool:
        if self.beam is None:
            raise StageError("Alg2 beam position has not been calibrated")
        ppm = float(getattr(getattr(self.world, "transform", None),
                            "px_per_mm", 0.0))
        if ppm <= 0:
            raise StageError("invalid pixel/mm calibration")
        # Moving the trapped ball +d relative to the sample requires the
        # complete sample image to move -d under the stationary beam.
        ok = self.shift_workspace_by(-dx_mm * ppm, -dy_mm * ppm,
                                     ppm, **kwargs)
        return ok

    def stop_all(self) -> None:
        self.laser_gate.off()
        stop = getattr(self._xy, "stop_all", None)
        if callable(stop):
            stop()
        stage = getattr(self.world, "motion_stage", None)
        if stage is not None:
            stage.stop()

    def close(self) -> None:
        self.laser_gate.off()
        close = getattr(self._xy, "close", None)
        if callable(close):
            close()

    @property
    def last_command(self):
        return self._last or self._xy.last_command


class FixedBeamController:
    """Alg2 closed loop in the stationary-beam camera coordinate system."""

    def __init__(self, stage: Alg2Stage, vision, planner, config, reporter) -> None:
        from .controller import RunResult

        self.stage = stage
        self.vision = vision
        self.planner = planner
        self.config = config
        self.reporter = reporter
        self._result_type = RunResult
        self._estop = threading.Event()
        self._pause = threading.Event()
        self._last_frame = None
        self._last_plan = None
        # W1：beam_lock 后钉住的目标球身份。多球场景下若每一帧都按
        # "谁最靠近光斑"重新解析，邻居球偶发贴近光斑就会静默切换目标
        # （真球被误判为障碍），造成多球报错。锁定后除非钉住球持续缺失，
        # 否则不切换目标。
        self._locked_track_id: Optional[int] = None

    @property
    def last_frame(self):
        return self._last_frame

    @property
    def last_plan(self):
        return self._last_plan

    def request_estop(self) -> None:
        self._estop.set()
        self.stage.stop_all()

    def pause(self) -> None:
        self._pause.set()
        self.stage.set_laser_enabled(False)

    def resume(self) -> None:
        self._pause.clear()

    def _capture(self, get_frame, get_snapshot):
        frame = get_frame() if get_frame else None
        snap = get_snapshot() if get_snapshot else None
        if frame is None or snap is None:
            return snap, None
        self._last_frame = frame
        return snap, self.vision.process(
            frame, snap.frame_id,
            allow_offscreen=bool(getattr(self.config,
                                         "allow_offscreen_elements", True)))

    def _offscreen_particle(self, particles, track_id: int, reference: Point):
        """需求2：元素（球/衬底外参考点）离开画面范围时的外推位置。"""
        off = [p for p in particles if not getattr(p, "in_frame", True)]
        if not off:
            return None
        by_id = next((p for p in off if p.track_id == track_id), None)
        if by_id is not None:
            return by_id
        return min(off, key=lambda p: math.dist(p.position_px, reference))

    def _offscreen_goal_plan(self, snap: WorkspaceSnapshot, start: Point,
                             goal: Point, task_id: str, role: str = "goal"):
        """需求2：终点离开画面范围 -> 直行兜底计划（否则返回 None）。"""
        fb = self.planner.direct_plan(snap, start, goal)
        if fb is None or not fb.success:
            return None
        self.reporter.log("element_offscreen", task_id=task_id,
                          stage="planning", role=role, detail=fb.detail)
        return fb

    def _target_particle(self, particles, track_id: int, reference: Point,
                         preferred: Optional[Point] = None,
                         pinned: bool = False):
        """Resolve the trapped ball in camera coordinates.

        During laser-off alignment the whole sample moves and contour order
        can change, so a stale track id is not sufficient. Prefer the id when
        it remains close to the predicted target; otherwise use the unique
        nearest detection. Once the laser is on, the fixed beam is the most
        reliable reference and prevents a crossing neighbour from inheriting
        the target lock.
        """
        if not particles:
            return None
        anchor = preferred if preferred is not None else reference
        if pinned:
            # W1：beam_lock 后目标身份已钉住。多球场景下邻居偶发贴近光斑时，
            # 不得用"谁最靠近光斑"来重新解析，否则会静默切换目标（真球变障碍）。
            # 只按钉住的 track_id 解析：在帧内命中直接返回（即使邻居更靠光斑）。
            by_id = next((p for p in particles if p.track_id == track_id), None)
            if by_id is not None:
                return by_id
            # 钉住球本帧未检出 —— 交由调用方用离屏/外推位置续追；这里不回落
            # 到最近邻，避免邻居伪装成目标。
            return None
        by_id = next((p for p in particles if p.track_id == track_id), None)
        if by_id is not None and math.dist(by_id.position_px, anchor) <= max(
                self.config.max_track_jump_px, 3.0 * by_id.radius_px):
            return by_id
        ranked = sorted((math.dist(p.position_px, anchor), p)
                        for p in particles)
        if not ranked:
            return None
        nearest_d, nearest = ranked[0]
        margin = ranked[1][0] - nearest_d if len(ranked) > 1 else float("inf")
        if nearest_d <= max(self.config.max_track_jump_px,
                            3.0 * nearest.radius_px) and margin >= max(
                                3.0, nearest.radius_px * 0.35):
            return nearest
        return None

    @staticmethod
    def _capture_limit_px(ball, cfg, fallback: float) -> float:
        """光镊捕获判据：光斑中心落在球内且留裕量（dist <= 半径 * 系数）。

        半径不可用（检测异常）时退回中心容差 ``fallback``。
        """
        radius = float(getattr(ball, "radius_px", 0.0) or 0.0)
        limit = radius * float(getattr(cfg, "beam_capture_fraction", 0.8))
        if limit <= 0.0:
            limit = float(fallback)
        return max(limit, 0.5)

    def _move_workspace(self, shift: Point, snap: WorkspaceSnapshot,
                        result, task_id: str, track_id: int,
                        waypoint_index: int, plan_version: int) -> bool:
        ok = self.stage.shift_workspace_by(
            shift[0], shift[1], snap.transform.px_per_mm,
            task_id=task_id, track_id=track_id,
            waypoint_index=waypoint_index, frame_id=snap.frame_id,
            plan_version=plan_version)
        command = self.stage.last_command
        if ok and command is not None:
            result.stage_commands.append(command)
            data = command.to_dict()
            data.pop("task_id", None)
            self.reporter.log("stage_command", task_id=task_id, **data)
        return bool(ok)

    def _probe_ball(self, vis, track_id: int, initial, frame_id: int):
        """探针取帧：只认本帧的真实检测。

        台账外推位置（frame_id 落后一帧）在探针里是致命的——球其实被台位
        带走了，外推值却停在原处，会测出 0 位移、标定直接失效。
        """
        if vis is None or vis.uncertain:
            return None
        fresh = [p for p in vis.particles
                 if int(getattr(p, "frame_id", -1)) == int(frame_id)
                 and getattr(p, "in_frame", True)]
        if not fresh:
            return None
        predicted = (initial.position_px[0] + self.stage.workspace_shift_px[0],
                     initial.position_px[1] + self.stage.workspace_shift_px[1])
        return self._target_particle(fresh, track_id, predicted)

    def _probe_step_mm(self, px_per_mm: float, radius_px: float) -> float:
        """探针步长：让图像位移约等于一个球半径，再折算回 mm。

        固定 mm 步长在不同 px_per_mm 下差异巨大（2000px/mm 的仿真里 0.2mm
        就是 400px，球直接出画），所以按目标图像位移反推，并受
        ``shift_probe_mm`` 上限约束。
        """
        limit_mm = float(getattr(self.stage.config, "shift_probe_mm", 0.0))
        if limit_mm <= 0.0 or px_per_mm <= 0.0:
            return 0.0
        target_px = min(60.0, max(3.0, float(radius_px or 0.0)))
        step = min(limit_mm, target_px / float(px_per_mm))
        # 探针同样是发往真实台位的一步：不得越过台位单步限位（UI「单步位移」
        # mm），否则驱动直接拒绝，标定被整段跳过。
        stage_limit = getattr(self.stage, "max_step_mm", None)
        if stage_limit:
            step = min(step, float(stage_limit))
        return step

    def _calibrate_shift_gain(self, initial, track_id: int, get_frame,
                              get_snapshot, result, task_id: str):
        """探针实测『图像位移 / 样品位移』增益（px/mm），修正符号与比例。

        现场最常见的问题是 image_shift_sign 配反：对准时球越走越远，一圈
        下来到不了光斑（SPOT_SLIP 中止），表现为电机一路朝同一个方向走。
        这里在开光前各发一次 +x/+y 小步，用实测球心位移直接标定增益；
        测不准的轴返回 None，由调用方回退到配置的 sign * px_per_mm。
        """
        # 探针参数属于 Alg2Stage 的 Alg2Config（对准步数等策略参数才在
        # ControllerConfig 里），不能从 self.config 取。
        cfg = self.stage.config
        if float(getattr(cfg, "shift_probe_mm", 0.0)) <= 0.0:
            return (None, None)
        min_gain = float(getattr(cfg, "min_shift_gain_px_per_mm", 5.0))
        retry = max(1, int(getattr(cfg, "shift_probe_retry", 8)))
        gains: list[Optional[float]] = [None, None]
        for axis in range(2):
            current, vis = self._capture(get_frame, get_snapshot)
            if current is None:
                return tuple(gains)
            before = self._probe_ball(vis, track_id, initial, current.frame_id)
            if before is None:
                return tuple(gains)
            ppm = float(current.transform.px_per_mm)
            probe_mm = self._probe_step_mm(ppm, before.radius_px)
            if probe_mm <= 0.0:
                return tuple(gains)
            # 有效响应线：明显大于检测抖动才算"球真的动了"（真机取帧有滞后，
            # 命令后头几帧可能还是旧画面，所以要重试到响应出现为止）。
            min_response_px = max(1.0, 0.2 * probe_mm * ppm)
            step = (probe_mm, 0.0) if axis == 0 else (0.0, probe_mm)
            try:
                ok = self.stage.sample_move_mm(
                    step[0], step[1], task_id=task_id, track_id=track_id,
                    waypoint_index=-2, frame_id=current.frame_id,
                    plan_version=0)
            except (StageError, TimeoutError, OSError) as exc:
                self.reporter.log("shift_calibration_abort", task_id=task_id,
                                  axis=axis, detail=str(exc))
                return tuple(gains)
            if not ok:
                return tuple(gains)
            command = self.stage.last_command
            if command is not None:
                result.stage_commands.append(command)
                data = command.to_dict()
                data.pop("task_id", None)
                self.reporter.log("stage_command", task_id=task_id, **data)
            measured = None
            for _ in range(retry):
                current, vis = self._capture(get_frame, get_snapshot)
                if current is None:
                    continue
                after = self._probe_ball(vis, track_id, initial,
                                         current.frame_id)
                if after is None:
                    continue
                measured = (after.position_px[0] - before.position_px[0],
                            after.position_px[1] - before.position_px[1])
                if abs(measured[axis]) >= min_response_px:
                    break
            if measured is None:
                return tuple(gains)
            # 探针是原始 mm 指令，图像位移只能用实测值入账。
            self.stage.note_image_shift(*measured)
            gain = measured[axis] / probe_mm
            if abs(gain) >= min_gain and abs(measured[axis]) >= min_response_px:
                gains[axis] = gain
            else:
                self.reporter.log("shift_calibration_weak", task_id=task_id,
                                  axis=axis, probe_mm=round(probe_mm, 6),
                                  measured_px=[round(v, 1) for v in measured])
        return tuple(gains)

    def run(self, snap: WorkspaceSnapshot, track_id: int, goal: GoalRegion,
            task_id: str = "oa", extra_obstacles=(), get_frame=None,
            get_snapshot=None):
        result = self._result_type(final_state=RunState.IDLE)
        cfg = self.config
        initial = snap.particle(track_id)
        if initial is None:
            result.final_state = RunState.ABORTED
            result.failure_reason = FailureReason.TARGET_LOST
            result.detail = f"initial target missing track {track_id}"
            return result
        initial_substrate_center = self._substrate_center(snap)
        extra_base_shift = tuple(getattr(self.stage, "workspace_shift_px",
                                        (0.0, 0.0)))
        extra_initial = list(extra_obstacles or ())

        def shifted_extra():
            current_shift = tuple(getattr(self.stage, "workspace_shift_px",
                                           extra_base_shift))
            dx = current_shift[0] - extra_base_shift[0]
            dy = current_shift[1] - extra_base_shift[1]
            out = []
            for obstacle in extra_initial:
                if obstacle.kind == "circle":
                    out.append(Obstacle(
                        kind="circle",
                        center=(obstacle.center[0] + dx,
                                obstacle.center[1] + dy),
                        radius=obstacle.radius,
                        obstacle_id=obstacle.obstacle_id))
                else:
                    out.append(Obstacle(
                        kind="polygon",
                        polygon=[(x + dx, y + dy)
                                 for x, y in obstacle.polygon],
                        obstacle_id=obstacle.obstacle_id))
            return out

        beam = self.stage.beam_position_px
        self.stage.set_laser_enabled(False)
        self.reporter.log("beam_calibration", task_id=task_id,
                          position_px=list(beam),
                          confidence=self.stage.beam.confidence,
                          source=self.stage.beam.source)
        self.reporter.log("laser_state", task_id=task_id, enabled=False,
                          reason="safe_alignment",
                          hardware_controlled=self.stage.laser_gate.hardware_controlled)
        self.reporter.log("state_change", task_id=task_id,
                          **{"from": "IDLE", "to": "CALIBRATING"})

        try:
            aligned = False
            ball_seen = False        # W1.4：对齐阶段是否至少成功解析到目标球
            gains = self._calibrate_shift_gain(
                initial, track_id, get_frame, get_snapshot, result, task_id)
            self.stage.set_shift_gain(gains)
            self.reporter.log(
                "shift_calibration", task_id=task_id,
                probe_mm=float(getattr(self.stage.config, "shift_probe_mm", 0.0)),
                gain_px_per_mm=[None if g is None else round(float(g), 3)
                                for g in gains],
                source="probe" if any(g is not None for g in gains)
                else "configured")
            align_limit = max(1, int(getattr(
                cfg, "beam_alignment_max_steps", 80)))
            align_tolerance = float(getattr(
                cfg, "beam_alignment_tolerance_px", cfg.tolerance_px))
            # 对准预算按『有效步进』计：检测丢帧/低置信度的帧不再吃掉步数，
            # 否则现场抖动几帧就把 80 步耗光（球还没走到光斑就 SPOT_SLIP 中止）。
            align_moves = 0
            align_frames = 0
            last_distance: Optional[float] = None
            last_radius = 0.0
            align_frame_cap = align_limit * max(
                1, int(getattr(cfg, "beam_alignment_frame_factor", 4)))
            while align_moves < align_limit and align_frames < align_frame_cap:
                align_frames += 1
                if self._estop.is_set():
                    result.final_state = RunState.ABORTED
                    result.failure_reason = FailureReason.ESTOP
                    result.detail = "estop during beam alignment"
                    return result
                current, vis = self._capture(get_frame, get_snapshot)
                if current is None or vis is None or vis.uncertain:
                    continue
                predicted = (initial.position_px[0] +
                             self.stage.workspace_shift_px[0],
                             initial.position_px[1] +
                             self.stage.workspace_shift_px[1])
                ball = self._target_particle(vis.particles, track_id,
                                             predicted)
                if ball is None:
                    # 需求2：球已离开画面范围 -> 用外推位置把它拉回视野，
                    # 不直接放弃对齐（否则会误报 SPOT_SLIP 中止）。
                    ball = self._offscreen_particle(vis.particles, track_id,
                                                    predicted)
                    if ball is None:
                        continue
                    self.reporter.log(
                        "element_offscreen", task_id=task_id, stage="alignment",
                        track_id=track_id, role="target_ball",
                        position_px=[round(v, 1) for v in ball.position_px])
                ball_seen = True
                track_id = ball.track_id
                self.stage.track_id = track_id
                set_target = getattr(self.stage.world, "set_alg2_target", None)
                if callable(set_target):
                    set_target(track_id)
                error = (beam[0] - ball.position_px[0],
                         beam[1] - ball.position_px[1])
                distance = math.hypot(*error)
                last_distance = distance
                last_radius = float(getattr(ball, "radius_px", 0.0) or 0.0)
                # 光镊物理：只有光斑打在球上（光斑中心落在球内且留裕量）
                # 才允许开光并控制该球移动，故对齐判据用捕获判据而非中心容差。
                if distance <= self._capture_limit_px(ball, cfg,
                                                      align_tolerance):
                    aligned = True
                    break
                # 对准也交给规划器：球到光斑走绕障路径，而不是直线撞过去
                # （直线会把球顶进 peer/禁区，也可能一路擦着边界走）。
                # 规划不可用时退回直线步进，保证对准总能继续。
                vector = error
                approach = self.planner.plan(current, ball.position_px, beam,
                                             list(shifted_extra()))
                if approach.success and len(approach.waypoints_px) >= 2:
                    ref = approach.waypoints_px[0]
                    nxt = next((p for p in approach.waypoints_px[1:]
                                if math.dist(p, ref) > 1e-6), beam)
                    vector = (nxt[0] - ref[0], nxt[1] - ref[1])
                    self._last_plan = approach
                    result.plan = approach
                    result.replan_count += 1
                    self.reporter.log("plan", task_id=task_id,
                                      stage="alignment", **approach.to_dict())
                length = math.hypot(*vector)
                if length <= 1e-9:
                    continue
                max_px = cfg.max_step_mm * current.transform.px_per_mm
                step_px = min(length, max_px)
                shift = (vector[0] * step_px / length,
                         vector[1] * step_px / length)
                if not self._move_workspace(
                        shift, current, result, task_id, track_id, -1, 0):
                    result.final_state = RunState.FAULT
                    result.failure_reason = FailureReason.COMM_TIMEOUT
                    result.detail = "stage refused beam-alignment command"
                    return result
                align_moves += 1
            if not aligned:
                # W1.4：区分"全程检测不到球"（检测问题）与"球已解析但未对齐"
                #（光斑/收敛问题），避免误报 SPOT_SLIP 掩盖真正的检测失效。
                enter_limit = max(
                    1.0, float(getattr(cfg, "beam_enter_fraction", 1.5))
                    * float(last_radius or 0.0))
                # "尽量对准"即可：球已解析且离光斑不远时不再中止，开光后由
                # tracking 的再捕获（方向已修正）把球心收进光斑；否则才按
                # TARGET_LOST / SPOT_SLIP 安全停止。
                if ball_seen and last_distance is not None and \
                        last_distance <= enter_limit:
                    self.reporter.log(
                        "beam_alignment_relaxed", task_id=task_id,
                        distance_px=round(last_distance, 1),
                        radius_px=round(float(last_radius or 0.0), 1),
                        enter_limit_px=round(enter_limit, 1),
                        detail="entering tracking with ball near beam")
                elif not ball_seen:
                    result.final_state = RunState.ABORTED
                    result.failure_reason = FailureReason.TARGET_LOST
                    result.detail = ("target ball never resolved during "
                                     "beam alignment (track {0}); no ball "
                                     "detected near predicted position".format(track_id))
                    return result
                else:
                    result.final_state = RunState.ABORTED
                    result.failure_reason = FailureReason.SPOT_SLIP
                    result.detail = ("selected ball could not align to "
                                     "calibrated beam within "
                                     f"{align_limit} alignment steps "
                                     f"(moves={align_moves}, "
                                     f"frames={align_frames}, "
                                     f"dist={last_distance:.1f}px)")
                    return result

            self.stage.set_laser_enabled(True)
            self.reporter.log("laser_state", task_id=task_id, enabled=True,
                              reason="ball_aligned" if aligned else "near_beam",
                              hardware_controlled=self.stage.laser_gate.hardware_controlled)
            # In a fixed-beam microscope the trapped ball is stationary in
            # camera coordinates while its sample-coordinate position changes
            # with the stage. Record this explicitly for the UI/replay so the
            # normal fixed-beam behavior is distinguishable from a lost ball.
            self.reporter.log("beam_lock", task_id=task_id,
                              track_id=track_id,
                              position_px=list(beam),
                              camera_locked=True)
            # W1：光斑锁定后钉住目标身份；后续多球判定一律以它为基准，
            # 避免邻居贴近光斑时静默切换目标。
            self._locked_track_id = track_id
            self.reporter.log("state_change", task_id=task_id,
                              **{"from": "CALIBRATING", "to": "TRACKING"})
            stable = 0
            uncertain = 0
            recapture = 0
            offscreen_moves = 0      # 需求2：连续离屏外推控制步数
            # 需求(防卡壳)：连续无进展帧数 / 软重规划剩余次数
            stall_streak = 0
            stall_last_error = float("inf")
            soft_replan_left = max(0, int(getattr(cfg, "soft_replan_limit", 4)))
            for iteration in range(1, cfg.max_iterations + 1):
                result.iterations = iteration
                if self._estop.is_set():
                    result.final_state = RunState.ABORTED
                    result.failure_reason = FailureReason.ESTOP
                    result.detail = "estop"
                    return result
                while self._pause.is_set() and not self._estop.is_set():
                    time.sleep(0.01)
                current, vis = self._capture(get_frame, get_snapshot)
                if current is None or vis is None or vis.uncertain:
                    uncertain += 1
                    if uncertain > cfg.uncertain_retry_limit:
                        result.final_state = RunState.ABORTED
                        result.failure_reason = FailureReason.DETECTION_UNCERTAIN
                        result.detail = "Alg2 dynamic workspace detection uncertain"
                        return result
                    continue
                uncertain = 0
                # W2：每帧记录全部球位置与当前目标，供多球场景复盘诊断。
                self.reporter.log(
                    "detection", task_id=task_id, frame_id=current.frame_id,
                    uncertain=False, reason="",
                    particles=[p.to_dict() for p in vis.particles],
                    target_track_id=track_id)
                ball = self._target_particle(vis.particles, track_id,
                                             beam, preferred=beam,
                                             pinned=True)
                offscreen_ball = False
                if ball is None:
                    # 需求2：目标球离开画面范围/单帧未检出 -> 用台账/跟踪器
                    # 外推位置继续定位运动（把球拉回光斑），不切目标、不直接
                    # ABORTED。
                    ball = self._offscreen_particle(vis.particles, track_id,
                                                    beam)
                    if ball is None:
                        result.final_state = RunState.ABORTED
                        result.failure_reason = FailureReason.TARGET_LOST
                        result.detail = ("selected ball lost after laser capture "
                                         f"(track {track_id})")
                        return result
                    offscreen_ball = True
                    self.reporter.log(
                        "element_offscreen", task_id=task_id, stage="tracking",
                        track_id=track_id, role="target_ball",
                        position_px=[round(v, 1) for v in ball.position_px])
                track_id = ball.track_id
                self.stage.track_id = track_id
                set_target = getattr(self.stage.world, "set_alg2_target", None)
                if callable(set_target):
                    set_target(track_id)
                # 光镊物理：只有光斑打在球上才能控制该球移动。脱靶时不做
                # 硬性中止，而是动态再捕获——把球心推回光斑中心（光斑稳定
                # 在球心）；偏离过远或连续再捕获超限才安全停止。
                offset = (ball.position_px[0] - beam[0],
                          ball.position_px[1] - beam[1])
                slip = math.hypot(*offset)
                if offscreen_ball:
                    # 需求2：元素离屏（外推位置）不做失位中止——直接把样品
                    # 朝外推位置移动把球拉回视野；连续外推超限才安全停止。
                    offscreen_moves += 1
                    if offscreen_moves > max(1, int(getattr(
                            cfg, "offscreen_retry_limit", 120))):
                        result.final_state = RunState.ABORTED
                        result.failure_reason = FailureReason.TARGET_LOST
                        result.detail = ("offscreen target not recovered "
                                         f"(moves={offscreen_moves})")
                        return result
                    max_px = cfg.max_step_mm * current.transform.px_per_mm
                    scale = min(1.0, max_px / max(slip, 1e-9))
                    # 未捕获的球随样品一起漂移（与对准阶段同一物理），要让球心
                    # 回到光斑就必须按 beam - ball = -offset 让图像位移；用
                    # +offset 会把球越推越远，电机一路同向走到 SPOT_SLIP。
                    if not self._move_workspace(
                            (-offset[0] * scale, -offset[1] * scale), current,
                            result, task_id, track_id, 1, 0):
                        result.final_state = RunState.FAULT
                        result.failure_reason = FailureReason.COMM_TIMEOUT
                        result.detail = "stage refused offscreen recovery command"
                        return result
                    continue
                if slip > self._capture_limit_px(
                        ball, cfg, cfg.beam_alignment_tolerance_px):
                    if slip > cfg.slip_abort_px:
                        result.final_state = RunState.ABORTED
                        result.failure_reason = FailureReason.SPOT_SLIP
                        result.detail = (f"beam far off ball {slip:.1f}px "
                                         f"(abort={cfg.slip_abort_px:.0f}px)")
                        return result
                    recapture += 1
                    if recapture > max(1, int(getattr(
                            cfg, "beam_recapture_max", 60))):
                        result.final_state = RunState.ABORTED
                        result.failure_reason = FailureReason.SPOT_SLIP
                        result.detail = (
                            f"beam off ball {slip:.1f}px > r*"
                            f"{cfg.beam_capture_fraction:.2f} "
                            f"(recapture attempts={recapture - 1})")
                        return result
                    self.reporter.log(
                        "beam_recapture", task_id=task_id, track_id=track_id,
                        offset_px=[round(offset[0], 1), round(offset[1], 1)],
                        radius_px=round(float(ball.radius_px), 1),
                        attempt=recapture)
                    max_px = cfg.max_step_mm * current.transform.px_per_mm
                    scale = min(1.0, max_px / max(slip, 1e-9))
                    # 光斑脱靶 = 球没被抓住，此刻球随样品漂移；要让球心回到
                    # 光斑中心必须按 beam - ball = -offset 让图像位移（与对准
                    # 阶段同向），用 +offset 会越修越偏。
                    if not self._move_workspace(
                            (-offset[0] * scale, -offset[1] * scale), current,
                            result, task_id, track_id, 1, 0):
                        result.final_state = RunState.FAULT
                        result.failure_reason = FailureReason.COMM_TIMEOUT
                        result.detail = "stage refused beam re-capture command"
                        return result
                    continue
                recapture = 0
                offscreen_moves = 0

                current_substrate_center = self._substrate_center(current)
                substrate_shift = (
                    current_substrate_center[0] - initial_substrate_center[0],
                    current_substrate_center[1] - initial_substrate_center[1])
                moving_goal = (goal.center[0] + substrate_shift[0],
                               goal.center[1] + substrate_shift[1])
                error_px = math.dist(beam, moving_goal)
                if error_px <= cfg.tolerance_px:
                    stable += 1
                    if stable >= cfg.stable_frames:
                        result.final_state = RunState.COMPLETE
                        result.final_error_px = error_px
                        return result
                    continue
                else:
                    stable = 0

                # ---- 需求(防卡壳)：连续无进展 -> 软重规划 + 卡缝临时禁区
                # 卡壳 = 光斑到目标的距离连续多帧几乎不缩短（球被 peer/障碍
                # 挤在缝里）。先用放宽碰撞模型（去掉 safety margin）+ 把当前
                # 卡住位置列为临时禁区重搜路径；软重规划耗尽仍无进展则明确
                # 中止，不再无限重规划同一条路径（视频复盘：卡缝死循环 9s）。
                stall_frames_eff = max(1, int(getattr(cfg, "stall_frames", 6)))
                progress = stall_last_error - error_px
                stall_last_error = min(stall_last_error, error_px)
                if error_px <= cfg.tolerance_px or progress > max(
                        1.0, float(getattr(cfg, "stall_progress_px", 1.0))):
                    stall_streak = 0
                else:
                    stall_streak += 1
                soft_extra: list = []
                if soft_replan_left <= 0 and \
                        stall_streak >= 2 * stall_frames_eff:
                    result.final_state = RunState.ABORTED
                    result.failure_reason = FailureReason.NO_SAFE_PATH
                    result.detail = (
                        f"stalled {stall_streak} frames without progress "
                        f"(error {error_px:.1f}px); soft replans exhausted")
                    return result
                if stall_streak >= stall_frames_eff and soft_replan_left > 0:
                    soft_replan_left -= 1
                    stall_streak = 0
                    # 卡缝临时禁区：目标还贴得很近时不加（否则堵死终点），
                    # 只有离目标尚远才把光斑处列为软禁区强迫绕行。
                    if error_px > self.planner.config.model.inflation_px + \
                            self.planner.config.model.ball_radius_px:
                        soft_extra = [Obstacle(
                            kind="circle", center=beam,
                            radius=self.planner.config.model.ball_radius_px,
                            obstacle_id="stall-marker")]
                    self.reporter.log(
                        "soft_replan", task_id=task_id, track_id=track_id,
                        detail="stall recovery: soft replan with seam marker",
                        soft_replan_left=soft_replan_left,
                        error_px=round(error_px, 1))
                relax = self.planner.config.model.safety_margin_px
                if soft_extra:
                    self.planner.config.model.safety_margin_px = 0.0
                plan = self.planner.plan(current, beam, moving_goal,
                                         list(shifted_extra()) + soft_extra)
                self.planner.config.model.safety_margin_px = relax
                if (not plan.success or len(plan.waypoints_px) < 2) and \
                        getattr(cfg, "allow_offscreen_elements", True):
                    # 需求2：终点（随样品移动）离开画面范围导致规划失败 ->
                    # 直行兜底，继续定位运动，不直接 ABORTED。
                    fb = self._offscreen_goal_plan(current, beam, moving_goal,
                                                   task_id)
                    if fb is not None:
                        plan = fb
                self._last_plan = plan
                result.plan = plan
                result.replan_count += 1
                self.reporter.log("plan", task_id=task_id, **plan.to_dict())
                if not plan.success or len(plan.waypoints_px) < 2:
                    result.final_state = RunState.ABORTED
                    result.failure_reason = (plan.failure_reason or
                                             FailureReason.NO_SAFE_PATH)
                    result.detail = plan.detail or "Alg2 found no safe path"
                    return result
                # 向量参考取路径自身起点而非光斑：起点常被 nearest_feasible
                # 就近挪出膨胀区（球贴着 peer/边界时），若仍以 beam 为参考，
                # 首段方向会指向禁区边缘而非沿路径前进，造成来回振荡卡壳。
                ref = plan.waypoints_px[0] if plan.waypoints_px else beam
                waypoint = next(
                    (p for p in plan.waypoints_px[1:]
                     if math.dist(p, ref) > 1e-6), moving_goal)
                vector = (waypoint[0] - ref[0], waypoint[1] - ref[1])
                length = math.hypot(*vector)
                step_px = min(length,
                              cfg.max_step_mm * current.transform.px_per_mm)
                world_shift = (-vector[0] * step_px / length,
                               -vector[1] * step_px / length)
                if not self._move_workspace(
                        world_shift, current, result, task_id, track_id, 1,
                        plan.plan_version):
                    result.final_state = RunState.FAULT
                    result.failure_reason = FailureReason.COMM_TIMEOUT
                    result.detail = "stage refused transport command"
                    return result
                result.min_clearance_observed_px = min(
                    result.min_clearance_observed_px, plan.min_clearance_px)

            result.final_state = RunState.FAULT
            result.failure_reason = FailureReason.MAX_STEPS
            result.detail = f"max_iterations={cfg.max_iterations} reached"
            return result
        except (StageError, TimeoutError, OSError, ConnectionError) as exc:
            result.final_state = RunState.FAULT
            result.failure_reason = FailureReason.COMM_TIMEOUT
            result.detail = str(exc)
            return result
        finally:
            self.stage.set_laser_enabled(False)
            self.reporter.log("laser_state", task_id=task_id, enabled=False,
                              reason="task_end",
                              hardware_controlled=self.stage.laser_gate.hardware_controlled)

    @staticmethod
    def _substrate_center(snapshot: WorkspaceSnapshot) -> Point:
        points = snapshot.substrate.polygon
        if not points:
            return (snapshot.frame_size[0] / 2.0,
                    snapshot.frame_size[1] / 2.0)
        # Recognition may re-sample the same boundary with a different number
        # of vertices. An area centroid is invariant to that sampling density,
        # unlike a plain average of vertices.
        if len(points) >= 3:
            twice_area = 0.0
            cx = cy = 0.0
            for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1]):
                cross = x0 * y1 - x1 * y0
                twice_area += cross
                cx += (x0 + x1) * cross
                cy += (y0 + y1) * cross
            if abs(twice_area) > 1e-9:
                return (cx / (3.0 * twice_area),
                        cy / (3.0 * twice_area))
        return (sum(p[0] for p in points) / len(points),
                sum(p[1] for p in points) / len(points))
