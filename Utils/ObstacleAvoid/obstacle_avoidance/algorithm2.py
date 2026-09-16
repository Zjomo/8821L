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

from .models import (FailureReason, GoalRegion, Particle, Point, RunState,
                     StageCommand, TargetLock, TargetSelection,
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

    def prepare_focus(self) -> None:
        stage = getattr(self.world, "motion_stage", None)
        if stage is None:
            return
        # Safe excursion followed by focus position makes Z motion explicit and
        # deterministic even when the stage starts already at focus.
        stage.enable()
        current = float(stage.position.get("z", 0.0))
        if abs(current - self.config.z_safe_um) > 1e-9:
            stage.move_to({"z": self.config.z_safe_um}, source="alg2-z-safe")
        if abs(self.config.z_focus_um - self.config.z_safe_um) > 1e-9:
            stage.move_to({"z": self.config.z_focus_um}, source="alg2-z-focus")

    def set_laser_enabled(self, enabled: bool) -> None:
        self.laser_gate.set_enabled(enabled)

    def shift_workspace_by(self, dx_px: float, dy_px: float,
                           px_per_mm: float, **kwargs) -> bool:
        """Move all sample-bound objects by the requested image displacement."""

        sign = int(self.config.image_shift_sign)
        if sign not in (-1, 1):
            raise StageError("Alg2 image_shift_sign must be +1 or -1")
        raw_dx = float(dx_px) / (sign * float(px_per_mm))
        raw_dy = float(dy_px) / (sign * float(px_per_mm))
        ok = self._xy.move_by(raw_dx, raw_dy, **kwargs)
        if ok:
            self._seq += 1
            self._last = StageCommand(
                seq=self._seq, timestamp=time.time(), dx_mm=raw_dx,
                dy_mm=raw_dy, task_id=str(kwargs.get("task_id", "")),
                track_id=int(kwargs.get("track_id", -1)),
                waypoint_index=int(kwargs.get("waypoint_index", -1)),
                frame_id=int(kwargs.get("frame_id", -1)),
                source_plan_version=int(kwargs.get("plan_version", -1)))
            register = getattr(self.world, "register_workspace_shift", None)
            if callable(register):
                register(float(dx_px), float(dy_px))
            self.workspace_shift_px = (
                self.workspace_shift_px[0] + float(dx_px),
                self.workspace_shift_px[1] + float(dy_px))
        return bool(ok)

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
                         preferred: Optional[Point] = None):
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
            align_limit = max(1, int(getattr(
                cfg, "beam_alignment_max_steps", 80)))
            align_tolerance = float(getattr(
                cfg, "beam_alignment_tolerance_px", cfg.tolerance_px))
            for index in range(align_limit):
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
                track_id = ball.track_id
                self.stage.track_id = track_id
                set_target = getattr(self.stage.world, "set_alg2_target", None)
                if callable(set_target):
                    set_target(track_id)
                error = (beam[0] - ball.position_px[0],
                         beam[1] - ball.position_px[1])
                distance = math.hypot(*error)
                # 光镊物理：只有光斑打在球上（光斑中心落在球内且留裕量）
                # 才允许开光并控制该球移动，故对齐判据用捕获判据而非中心容差。
                if distance <= self._capture_limit_px(ball, cfg,
                                                      align_tolerance):
                    aligned = True
                    break
                max_px = cfg.max_step_mm * current.transform.px_per_mm
                scale = min(1.0, max_px / max(distance, 1e-9))
                shift = (error[0] * scale, error[1] * scale)
                if not self._move_workspace(
                        shift, current, result, task_id, track_id, -1, 0):
                    result.final_state = RunState.FAULT
                    result.failure_reason = FailureReason.COMM_TIMEOUT
                    result.detail = "stage refused beam-alignment command"
                    return result
            if not aligned:
                result.final_state = RunState.ABORTED
                result.failure_reason = FailureReason.SPOT_SLIP
                result.detail = "selected ball could not align to calibrated beam"
                return result

            self.stage.set_laser_enabled(True)
            self.reporter.log("laser_state", task_id=task_id, enabled=True,
                              reason="ball_aligned",
                              hardware_controlled=self.stage.laser_gate.hardware_controlled)
            # In a fixed-beam microscope the trapped ball is stationary in
            # camera coordinates while its sample-coordinate position changes
            # with the stage. Record this explicitly for the UI/replay so the
            # normal fixed-beam behavior is distinguishable from a lost ball.
            self.reporter.log("beam_lock", task_id=task_id,
                              track_id=track_id,
                              position_px=list(beam),
                              camera_locked=True)
            self.reporter.log("state_change", task_id=task_id,
                              **{"from": "CALIBRATING", "to": "TRACKING"})
            stable = 0
            uncertain = 0
            recapture = 0
            offscreen_moves = 0      # 需求2：连续离屏外推控制步数
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
                ball = self._target_particle(vis.particles, track_id,
                                             beam, preferred=beam)
                offscreen_ball = False
                if ball is None:
                    # 需求2：目标球离开画面范围 -> 用台账/跟踪器外推位置
                    # 继续定位运动（把球拉回光斑），不再直接 ABORTED。
                    ball = self._offscreen_particle(vis.particles, track_id,
                                                    beam)
                    if ball is None:
                        result.final_state = RunState.ABORTED
                        result.failure_reason = FailureReason.TARGET_LOST
                        result.detail = "selected ball lost after laser capture"
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

                plan = self.planner.plan(current, beam, moving_goal,
                                         shifted_extra())
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
                waypoint = next(
                    (p for p in plan.waypoints_px[1:]
                     if math.dist(p, beam) > 1e-6), moving_goal)
                vector = (waypoint[0] - beam[0], waypoint[1] - beam[1])
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
