"""闭环位移控制器与安全状态机（PLAN 第 4 节）。

状态机：IDLE -> CALIBRATING -> TRACKING -> PLANNING -> MOVING -> VERIFYING
       -> COMPLETE / ABORTED / FAULT（DETECTION_UNCERTAIN 为安全驻留态）。

安全策略：
- 急停/暂停在每次电机命令前检查；急停后不再发出任何命令。
- 检测不确定 -> DETECTION_UNCERTAIN 驻留，重试超限 -> ABORTED；不盲动。
- 障碍签名变化 -> 立即重规划（OA-05）。
- 恢复（resume）必须重新检测 + 重新规划，不能盲目续跑。
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from .models import (FailureReason, GoalRegion, Obstacle, Point, RunState,
                     StageCommand, WorkspaceSnapshot, now)
from .planner import CollisionModel, GridPlanner, PlanResult
from .reporter import RunReporter
from .simulator import StageError, XYStageProtocol
from .vision import VisionPipeline


@dataclass
class ControllerConfig:
    max_step_mm: float = 0.30          # 单条命令最大位移（小步移动）
    tolerance_px: float = 6.0          # 到达判定距离
    stable_frames: int = 3             # 连续稳定帧数
    max_iterations: int = 400          # 无界循环保护
    uncertain_retry_limit: int = 5     # 检测不确定重试上限
    min_clearance_px: float = 2.0      # 规划最小 clearance 门限
    max_track_jump_px: float = 80.0    # 跨帧匹配最大跳变
    obstacle_replan_quantum_px: float = 2.0  # 忽略检测坐标的亚像素抖动
    model: CollisionModel = field(default_factory=CollisionModel)
    # ---- 光斑-球失位防护（电机/视觉闭环通用）
    # 语义：一条 stage 命令 (dx,dy) 后，球在图像中的预期位移 = sign * (dx,dy)*px_per_mm。
    # sign=+1：球随光斑/载物移动（SimWorld）；sign=-1：衬底反向移动、球锁定衬底（VideoWorld/相机）。
    ball_shift_sign: int = 1
    prefer_track_id: bool = False
    slip_threshold_px: float = 25.0    # 实际位移偏离预期超过该值记一次滑移事件
    slip_abort_px: float = 60.0        # 单次滑移超过该值立即安全停止
    slip_max_events: int = 3           # 累计滑移事件上限，超过则停止


@dataclass
class RunResult:
    final_state: RunState
    failure_reason: Optional[FailureReason] = None
    final_error_px: Optional[float] = None
    iterations: int = 0
    stage_commands: List[StageCommand] = field(default_factory=list)
    commands_after_estop: int = 0
    estop_latency_s: Optional[float] = None
    replan_count: int = 0
    min_clearance_observed_px: float = float("inf")
    plan: Optional[PlanResult] = None
    detail: str = ""

    def to_dict(self) -> dict:
        return {"final_state": self.final_state.value,
                "failure_reason": (self.failure_reason.value
                                   if self.failure_reason else None),
                "final_error_px": self.final_error_px,
                "iterations": self.iterations,
                "stage_command_count": len(self.stage_commands),
                "commands_after_estop": self.commands_after_estop,
                "estop_latency_s": self.estop_latency_s,
                "replan_count": self.replan_count,
                "min_clearance_observed_px": (None if self.min_clearance_observed_px
                                              == float("inf")
                                              else self.min_clearance_observed_px),
                "detail": self.detail}


class ObstacleAvoidController:
    """单球避障导航闭环控制器。"""

    def __init__(self,
                 stage: XYStageProtocol,
                 vision: Optional[VisionPipeline] = None,
                 planner: Optional[GridPlanner] = None,
                 config: Optional[ControllerConfig] = None,
                 reporter: Optional[RunReporter] = None) -> None:
        self.stage = stage
        self.vision = vision or VisionPipeline()
        self.planner = planner or GridPlanner()
        self.config = config or ControllerConfig()
        self.reporter = reporter or RunReporter()
        self.state = RunState.IDLE
        self._estop = threading.Event()
        self._pause = threading.Event()
        self._estop_requested_at: Optional[float] = None
        self._last_frame: Optional[object] = None       # 最近一帧图像（UI 用）
        self._last_plan: Optional[PlanResult] = None    # 最近规划（UI 用）

    # ---------------- external safety controls
    def request_estop(self) -> None:
        self._estop_requested_at = now()
        self._estop.set()
        self.reporter.log("estop", action="requested")

    def pause(self) -> None:
        self._pause.set()
        self.reporter.log("pause", action="pause")

    def resume(self) -> None:
        self._pause.clear()
        self.reporter.log("pause", action="resume_revalidate")

    @property
    def last_frame(self):
        return self._last_frame

    @property
    def last_plan(self) -> Optional[PlanResult]:
        return self._last_plan

    # ---------------- helpers
    def _set_state(self, s: RunState, task_id: str = "") -> None:
        if s != self.state:
            self.reporter.log("state_change", task_id=task_id,
                              **{"from": self.state.value, "to": s.value})
            self.state = s

    def _estop_hit(self, result: RunResult) -> bool:
        if not self._estop.is_set():
            return False
        if self._estop_requested_at is not None:
            result.estop_latency_s = now() - self._estop_requested_at
        result.commands_after_estop = 0
        result.final_state = RunState.ABORTED
        result.failure_reason = FailureReason.ESTOP
        self._set_state(RunState.ABORTED)
        self.reporter.log("run_end", final_state=RunState.ABORTED.value,
                          reason="estop",
                          metrics=result.to_dict())
        return True

    # ---------------- main loop
    def run(self, snap: WorkspaceSnapshot, track_id: int, goal: GoalRegion,
            task_id: str = "oa",
            extra_obstacles: Optional[Sequence[Obstacle]] = None,
            get_frame: Optional[Callable[[], object]] = None,
            get_snapshot: Optional[Callable[[], WorkspaceSnapshot]] = None,
            ) -> RunResult:
        cfg = self.config
        result = RunResult(final_state=RunState.IDLE)
        extra = list(extra_obstacles or [])
        self._estop.clear()
        self._pause.clear()
        self.reporter.log("task_config", task_id=task_id,
                          track_id=track_id, goal=goal.to_dict(),
                          max_step_mm=cfg.max_step_mm,
                          tolerance_px=cfg.tolerance_px,
                          stable_frames=cfg.stable_frames)

        # ---- CALIBRATING: 初始检测 + 任务校验（拒绝层）
        self._set_state(RunState.CALIBRATING, task_id)
        particle = snap.particle(track_id)
        if particle is None:
            result.final_state = RunState.FAULT
            result.failure_reason = FailureReason.DETECTION_UNCERTAIN
            result.detail = f"initial detection missing track {track_id}"
            self.reporter.log("error", task_id=task_id, reason=result.detail)
            return result
        start = particle.position_px
        # 碰撞体积适配：运动球自身实际检测半径并入模型（模型值偏小时
        # 防止低估碰撞体积；只增不减，序列控制的后续球保持保守膨胀）
        self._adapt_ball_radius(particle)
        reason = self.planner.check_point(snap, start, extra)
        kind = "start"
        if reason is None:
            reason = self.planner.check_point(snap, goal.center, extra)
            kind = "goal"
        if reason is not None:
            pt = start if kind == "start" else goal.center
            c = self.planner.clearance_at(snap, pt, extra)
            need = self.planner.config.edge_clearance
            result.final_state = RunState.ABORTED
            result.failure_reason = reason
            result.detail = (f"task rejected: {reason.value} at {kind} "
                             f"({pt[0]:.0f},{pt[1]:.0f}) clearance={c:.0f}px "
                             f"required>={need:.0f}px")
            self.reporter.log("error", task_id=task_id, reason=result.detail)
            return result

        self._set_state(RunState.TRACKING, task_id)
        plan: Optional[PlanResult] = None
        # Camera detections jitter slightly even while the sample is still.
        # Quantize only the control-loop signature so this does not cause a
        # needless replan before every movement command.
        obstacle_sig_q = max(0.0, float(cfg.obstacle_replan_quantum_px))
        last_obstacle_sig = snap.obstacle_signature(obstacle_sig_q)
        stable = 0
        uncertain_streak = 0
        wp_index = 0
        pause_seen = False
        last_pos = start          # 最近位置链（用于跨帧匹配同一球）
        error_px = math.dist(start, goal.center)
        pending_check = None      # (移动前球位置, 预期像素位移)：滑移校验
        slip_events = 0           # 累计滑移事件数

        for it in range(1, cfg.max_iterations + 1):
            result.iterations = it
            if self._estop_hit(result):
                return result
            while self._pause.is_set():
                pause_seen = True
                if self._estop_hit(result):
                    return result
                time.sleep(0.01)
            # 恢复后强制重新检测 + 重新规划（不能盲目续跑）
            force_replan = pause_seen
            pause_seen = False
            # 注意：pending_check 不能在此重置——它在上一迭代移动后设置，
            # 需要在本迭代感知阶段校验（滑移检测）。不确定/丢失路径 continue
            # 时保留它，中间未下发新命令，下次成功检测补验依然有效。

            # ---- TRACKING：感知
            snap = get_snapshot() if get_snapshot else snap
            frame = get_frame() if get_frame else None
            if frame is None:
                vis = None
            else:
                self._last_frame = frame
                vis = self.vision.process(frame, snap.frame_id)
            if vis is None or vis.uncertain:
                uncertain_streak += 1
                self._set_state(RunState.DETECTION_UNCERTAIN, task_id)
                reason = (vis.uncertain_reason if vis is not None
                          else "no_frame_source")
                self.reporter.log("detection", task_id=task_id,
                                  frame_id=snap.frame_id, uncertain=True,
                                  reason=reason, particles=[])
                if uncertain_streak > cfg.uncertain_retry_limit:
                    result.final_state = RunState.ABORTED
                    result.failure_reason = FailureReason.DETECTION_UNCERTAIN
                    result.detail = f"detection uncertain x{uncertain_streak}"
                    self.reporter.log("run_end", final_state="ABORTED",
                                      metrics=result.to_dict())
                    return result
                continue
            uncertain_streak = 0

            self.reporter.log("detection", task_id=task_id,
                              frame_id=snap.frame_id,
                              uncertain=False, reason="",
                              particles=[p.to_dict() for p in vis.particles])

            particle = self._match_particle(
                vis.particles, last_pos,
                track_id if cfg.prefer_track_id else None)
            if particle is None:
                # 跟踪丢失（最近邻匹配失败），按不确定处理
                uncertain_streak += 1
                self._set_state(RunState.DETECTION_UNCERTAIN, task_id)
                if uncertain_streak > cfg.uncertain_retry_limit:
                    result.final_state = RunState.ABORTED
                    result.failure_reason = FailureReason.DETECTION_UNCERTAIN
                    result.detail = "track lost"
                    self.reporter.log("run_end", final_state="ABORTED",
                                      metrics=result.to_dict())
                    return result
                continue
            last_pos = particle.position_px
            pos = particle.position_px
            # 移动中碰撞体积跟随检测：实际半径增大 -> 更新模型并强制重规划
            if self._adapt_ball_radius(particle):
                force_replan = True
            # ---- 光斑-球失位校准：实际位移 vs 预期位移（stage 命令反推）
            if pending_check is not None:
                p0, shift = pending_check
                pending_check = None
                exp = (p0[0] + cfg.ball_shift_sign * shift[0],
                       p0[1] + cfg.ball_shift_sign * shift[1])
                dev = math.dist(pos, exp)
                if dev > cfg.slip_threshold_px:
                    slip_events += 1
                    self.reporter.log(
                        "spot_slip", task_id=task_id,
                        deviation_px=round(dev, 1),
                        expected_px=[round(v, 1) for v in exp],
                        actual_px=[round(v, 1) for v in pos],
                        event_index=slip_events)
                    if dev > cfg.slip_abort_px or slip_events >= cfg.slip_max_events:
                        result.final_state = RunState.ABORTED
                        result.failure_reason = FailureReason.SPOT_SLIP
                        result.detail = (f"spot-ball slip {dev:.0f}px "
                                         f"(events={slip_events}, "
                                         f"threshold={cfg.slip_threshold_px:.0f} "
                                         f"/abort={cfg.slip_abort_px:.0f})")
                        self.reporter.log("run_end", final_state="ABORTED",
                                          metrics=result.to_dict())
                        self._set_state(RunState.ABORTED, task_id)
                        return result
                    # 未达停止线：从实际位置强制重规划（重新锁定）
                    force_replan = True
            error_px = math.dist(pos, goal.center)

            # ---- VERIFYING：完成判定（先于移动，支持零位移场景）
            if error_px <= cfg.tolerance_px:
                stable += 1
                if stable >= cfg.stable_frames:
                    result.final_state = RunState.COMPLETE
                    result.final_error_px = error_px
                    result.plan = plan
                    self._set_state(RunState.COMPLETE, task_id)
                    self.reporter.log("run_end", final_state="COMPLETE",
                                      metrics=result.to_dict())
                    return result
            else:
                stable = 0

            # ---- 障碍突现检测：签名变化 -> 立即重规划（OA-05）
            sig = snap.obstacle_signature(obstacle_sig_q)
            need_replan = (plan is None or force_replan
                           or sig != last_obstacle_sig
                           or wp_index >= len(plan.waypoints_px))
            if need_replan:
                self._set_state(RunState.PLANNING, task_id)
                plan = self.planner.plan(snap, pos, goal.center, extra)
                self._last_plan = plan
                result.replan_count += 1
                wp_index = 0
                self.reporter.log("plan", task_id=task_id, **plan.to_dict())
                if not plan.success:
                    # 拒绝下发任何电机命令
                    result.final_state = RunState.ABORTED
                    result.failure_reason = plan.failure_reason
                    result.detail = plan.detail
                    result.plan = plan
                    self._set_state(RunState.ABORTED, task_id)
                    self.reporter.log("run_end", final_state="ABORTED",
                                      metrics=result.to_dict())
                    return result
                if plan.min_clearance_px < cfg.min_clearance_px:
                    result.final_state = RunState.ABORTED
                    result.failure_reason = FailureReason.LOW_CLEARANCE
                    result.detail = ("plan clearance "
                                     f"{plan.min_clearance_px:.1f}px < "
                                     f"{cfg.min_clearance_px}px")
                    self._set_state(RunState.ABORTED, task_id)
                    self.reporter.log("run_end", final_state="ABORTED",
                                      metrics=result.to_dict())
                    return result

                # A* can map start and goal into the same coarse grid cell.
                # In that case the simplified path contains one point and the
                # old loop kept replanning forever without issuing a command.
                # Preserve a safe direct segment when it is collision-free;
                # otherwise fail explicitly with a useful planning reason.
                if len(plan.waypoints_px) < 2 and error_px > cfg.tolerance_px:
                    if self.planner._line_free(pos, goal.center, snap, extra):
                        plan.waypoints_px = [pos, goal.center]
                        plan.length_px = math.dist(pos, goal.center)
                        plan.min_clearance_px = self.planner._min_clearance(
                            plan.waypoints_px, snap, extra)
                    else:
                        result.final_state = RunState.ABORTED
                        result.failure_reason = FailureReason.NO_SAFE_PATH
                        result.detail = (
                            "planner returned a degenerate path and the "
                            "direct segment is blocked")
                        result.plan = plan
                        self._set_state(RunState.ABORTED, task_id)
                        self.reporter.log("run_end", final_state="ABORTED",
                                          metrics=result.to_dict())
                        return result
                last_obstacle_sig = sig

                # Grid plans include the current position as their first
                # waypoint.  Consume all zero-length leading waypoints now;
                # otherwise a replan triggered by camera jitter can repeatedly
                # select the same start point and never issue a stage command.
                while (wp_index < len(plan.waypoints_px) - 1 and
                       math.dist(pos, plan.waypoints_px[wp_index]) < 1e-6):
                    wp_index += 1

            # ---- MOVING：沿 waypoint 小步移动
            self._set_state(RunState.MOVING, task_id)
            target = plan.waypoints_px[min(wp_index, len(plan.waypoints_px) - 1)]
            seg = math.dist(pos, target)
            if seg < 1e-6:
                wp_index += 1
                continue
            step_px = min(seg, cfg.max_step_mm * snap.transform.px_per_mm)
            ux, uy = (target[0] - pos[0]) / seg, (target[1] - pos[1]) / seg
            dx_mm = ux * step_px / snap.transform.px_per_mm
            dy_mm = uy * step_px / snap.transform.px_per_mm
            if self._estop_hit(result):
                return result
            try:
                ok = self._stage_move(dx_mm, dy_mm, task_id, track_id,
                                      wp_index, snap.frame_id, plan.plan_version)
            except (StageError, TimeoutError, OSError, ConnectionError) as exc:
                # 通信超时/驱动故障 -> FAULT，后续不再发命令（HW-02）
                result.final_state = RunState.FAULT
                result.failure_reason = FailureReason.COMM_TIMEOUT
                result.detail = str(exc)
                self.reporter.log("error", task_id=task_id, reason=result.detail)
                self._set_state(RunState.FAULT, task_id)
                self.reporter.log("run_end", final_state="FAULT",
                                  metrics=result.to_dict())
                return result
            if not ok:
                result.final_state = RunState.FAULT
                result.failure_reason = FailureReason.COMM_TIMEOUT
                result.detail = "stage refused command"
                self._set_state(RunState.FAULT, task_id)
                return result
            result.stage_commands.append(self.stage.last_command)
            cmd_dict = dict(self.stage.last_command.to_dict())
            cmd_dict.pop("task_id", None)  # 避免与 log 参数冲突
            self.reporter.log("stage_command", task_id=task_id, **cmd_dict)
            ppm = snap.transform.px_per_mm
            pending_check = (pos, (dx_mm * ppm, dy_mm * ppm))
            result.min_clearance_observed_px = min(
                result.min_clearance_observed_px, plan.min_clearance_px)
            # VERIFYING：移动后重检
            self._set_state(RunState.VERIFYING, task_id)
            if seg - step_px < 1e-6:
                wp_index += 1
        # for 结束
        result.final_state = RunState.FAULT
        result.failure_reason = FailureReason.MAX_STEPS
        result.detail = f"max_iterations={cfg.max_iterations} reached"
        self.reporter.log("run_end", final_state="FAULT",
                          metrics=result.to_dict())
        return result

    def _adapt_ball_radius(self, particle) -> bool:
        """把运动球实际检测半径并入碰撞模型（只增不减）。

        返回 True 表示模型半径被上调，调用方应强制重规划。
        """
        model = self.planner.config.model
        r = float(getattr(particle, "radius_px", 0.0) or 0.0)
        if r > model.ball_radius_px:
            self.reporter.log("collision_model_update",
                              ball_radius_px=round(model.ball_radius_px, 1),
                              effective_radius_px=round(r, 1))
            model.ball_radius_px = r
            return True
        return False

    def _match_particle(self, particles, last_pos: Point,
                        preferred_track_id: Optional[int] = None):
        """最近位置链匹配：从检测结果中挑出距 last_pos 最近的球。"""
        if preferred_track_id is not None:
            preferred = next((p for p in particles
                              if p.track_id == preferred_track_id), None)
            if preferred is not None:
                d = math.dist(preferred.position_px, last_pos)
                if d <= self.config.max_track_jump_px:
                    return preferred
        best, best_d = None, float("inf")
        for p in particles:
            d = math.dist(p.position_px, last_pos)
            if d < best_d:
                best, best_d = p, d
        if best is None or best_d > self.config.max_track_jump_px:
            return None
        return best

    def _stage_move(self, dx_mm: float, dy_mm: float, task_id: str,
                    track_id: int, wp_index: int, frame_id: int,
                    plan_version: int) -> bool:
        move = self.stage.move_by
        try:
            return move(dx_mm, dy_mm, task_id=task_id, track_id=track_id,
                        waypoint_index=wp_index, frame_id=frame_id,
                        plan_version=plan_version)
        except TypeError:
            # 兼容仅实现最小签名的驱动
            return move(dx_mm, dy_mm)
