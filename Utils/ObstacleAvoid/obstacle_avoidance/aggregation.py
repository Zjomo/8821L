"""聚拢任务与分配策略（PLAN 第 5 节）。

- Hungarian（scipy.linear_sum_assignment）把球分配到聚拢区域内的互不重叠驻点。
- 逐球闭环执行；已到达球作为安全占位（静态障碍）参与后续规划。
- 完成判定（闭环模式）：区域内数量、质心偏差、速度阈值。
- 区域非法（衬底外 / 容量不足 / clearance 不足）-> 拒绝任务。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .controller import ControllerConfig, ObstacleAvoidController, RunResult
from .models import (ConfigError, FailureReason, GoalRegion, Obstacle, Point,
                     RunState, WorkspaceSnapshot)
from .planner import CollisionModel, GridPlanner
from .reporter import RunReporter
from .simulator import SimWorld
from .vision import VisionPipeline


@dataclass
class AggregationConfig:
    required_count: int = 2
    max_centroid_dev_px: float = 30.0     # 成员质心相对区域中心最大偏差
    max_speed_px_per_frame: float = 2.0   # 完成判定速度阈值
    capacity_factor: float = 1.35         # 容量安全系数
    controller: ControllerConfig = field(default_factory=ControllerConfig)


@dataclass
class AggregationResult:
    final_state: RunState
    completed: bool = False
    per_ball: Dict[int, RunResult] = field(default_factory=dict)
    assignment: Dict[int, Point] = field(default_factory=dict)
    failure_reason: Optional[FailureReason] = None
    detail: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"final_state": self.final_state.value,
                "completed": self.completed,
                "assignment": {str(k): list(v) for k, v in self.assignment.items()},
                "failure_reason": (self.failure_reason.value
                                   if self.failure_reason else None),
                "detail": self.detail,
                "per_ball": {str(k): v.to_dict() for k, v in self.per_ball.items()},
                "metrics": self.metrics}


class AggregationPlanner:
    """多球聚拢：分配 -> 逐球闭环 -> 完成判定。"""

    def __init__(self, planner: Optional[GridPlanner] = None,
                 vision: Optional[VisionPipeline] = None,
                 config: Optional[AggregationConfig] = None,
                 reporter: Optional[RunReporter] = None,
                 controller_sink: Optional[dict] = None,
                 stage_factory: Optional[Callable[[], object]] = None) -> None:
        self.grid = planner or GridPlanner()
        self.vision = vision or VisionPipeline()
        self.config = config or AggregationConfig()
        self.reporter = reporter or RunReporter()
        self.controller_sink = controller_sink
        self.stage_factory = stage_factory

    # ---------------- validation
    def validate_region(self, snap: WorkspaceSnapshot,
                        region: GoalRegion, n_balls: int) -> None:
        """AG-04：区域非法时拒绝任务（ConfigError）。"""
        ccfg = self.config.controller
        r_eff = ccfg.model.ball_radius_px
        if not snap.substrate.is_feasible(region.center,
                                          clearance_px=r_eff):
            raise ConfigError(
                f"region center not in feasible substrate ({region.center})")
        # 容量：n 个球的外接圆面积 * factor <= 区域面积
        need = n_balls * math.pi * r_eff * r_eff * self.config.capacity_factor
        have = math.pi * region.radius_px ** 2
        if have < need:
            raise ConfigError(
                f"region capacity insufficient: need {need:.0f}px^2, have {have:.0f}")

    # ---------------- assignment
    def assign(self, snap: WorkspaceSnapshot, region: GoalRegion,
               track_ids: Sequence[int]) -> Dict[int, Point]:
        """Hungarian 分配：区域内按环形排布驻点，间距 >= 2*r_eff。"""
        n = len(track_ids)
        r_eff = self.config.controller.model.ball_radius_px
        targets = _ring_targets(region.center, region.radius_px * 0.5, n,
                                spacing=2 * r_eff * 1.2)
        cost = []
        for tid in track_ids:
            p = snap.particle(tid)
            row = [math.dist(p.position_px, t) for t in targets]
            cost.append(row)
        try:
            from scipy.optimize import linear_sum_assignment
            rows, cols = linear_sum_assignment(cost)
            mapping = {track_ids[r]: targets[c] for r, c in zip(rows, cols)}
        except ImportError:  # pragma: no cover
            mapping = {}
            remaining = set(range(len(targets)))
            for tid in track_ids:
                p = snap.particle(tid)
                c = min(remaining, key=lambda i: math.dist(p.position_px, targets[i]))
                mapping[tid] = targets[c]
                remaining.discard(c)
        return mapping

    # ---------------- execution
    def run(self, world: SimWorld, snap: WorkspaceSnapshot, region: GoalRegion,
            track_ids: Optional[Sequence[int]] = None,
            task_id: str = "ag") -> AggregationResult:
        cfg = self.config
        if track_ids is None:
            track_ids = [p.track_id for p in snap.particles]
        if len(track_ids) < cfg.required_count:
            raise ConfigError(
                f"need {cfg.required_count} balls, got {len(track_ids)}")
        self.validate_region(snap, region, len(track_ids))
        self.reporter.log("task_config", task_id=task_id, kind="aggregation",
                          region=region.to_dict(),
                          track_ids=list(track_ids),
                          required_count=cfg.required_count)

        assignment = self.assign(snap, region, track_ids)
        result = AggregationResult(final_state=RunState.IDLE,
                                   assignment=dict(assignment))

        # 就近优先执行（减少交叉与等待）
        order = sorted(track_ids,
                       key=lambda tid: math.dist(
                           snap.particle(tid).position_px, assignment[tid]))
        placed: List[Obstacle] = []   # 已到达球的安全占位
        for tid in order:
            if self._controller_estopped(result):
                break
            target = assignment[tid]
            goal = GoalRegion(center=target,
                              radius_px=cfg.controller.tolerance_px + 1.0)
            # Worlds that render/track a designated target omit that ball from
            # the dynamic-obstacle list.  Rotate the designation for every
            # sequentially controlled ball; otherwise the previously selected
            # ball could disappear from collision planning in later runs.
            if hasattr(world, "target_track_id"):
                world.target_track_id = tid
            if self.stage_factory is not None:
                stage = self.stage_factory()
            else:
                try:
                    stage = world.make_stage(tid)
                except TypeError:
                    stage = world.make_stage()
            controller = ObstacleAvoidController(
                stage=stage, vision=self.vision, planner=self.grid,
                config=cfg.controller, reporter=self.reporter)
            if self.controller_sink is not None:
                self.controller_sink["controller"] = controller
            run = controller.run(
                world.snapshot(), tid, goal, task_id=f"{task_id}/ball{tid}",
                extra_obstacles=list(placed),
                get_frame=world.render,
                get_snapshot=world.snapshot)
            result.per_ball[tid] = run
            if run.final_state == RunState.COMPLETE:
                pos = world.particle_position(tid)
                placed.append(Obstacle(kind="circle", center=pos,
                                       radius=self._ball_radius(snap, tid),
                                       obstacle_id=f"placed-{tid}"))
            else:
                # 任一球失败 -> 任务中止（安全优先），原因透传
                result.final_state = run.final_state
                result.failure_reason = run.failure_reason
                result.detail = f"ball {tid}: {run.detail}"
                self.reporter.log("run_end", task_id=task_id,
                                  final_state=run.final_state.value,
                                  metrics=result.to_dict())
                return result

        # ---- 完成判定（闭环）
        final = world.snapshot()
        in_region = [p for p in final.particles
                     if region.contains_center(p.position_px, p.radius_px)]
        count_ok = len(in_region) >= cfg.required_count
        centroid_ok, centroid_dev = True, 0.0
        if in_region:
            cx = sum(p.position_px[0] for p in in_region) / len(in_region)
            cy = sum(p.position_px[1] for p in in_region) / len(in_region)
            centroid_dev = math.dist((cx, cy), region.center)
            centroid_ok = centroid_dev <= cfg.max_centroid_dev_px
        speed_ok = all(math.hypot(*p.velocity_px_s) <= cfg.max_speed_px_per_frame
                       for p in final.particles)
        result.metrics = {"count_in_region": float(len(in_region)),
                          "centroid_dev_px": centroid_dev,
                          "max_speed_px_per_frame": max(
                              (math.hypot(*p.velocity_px_s)
                               for p in final.particles), default=0.0)}
        if count_ok and centroid_ok and speed_ok:
            result.final_state = RunState.COMPLETE
            result.completed = True
        else:
            result.final_state = RunState.ABORTED
            result.failure_reason = FailureReason.LOW_CLEARANCE
            result.detail = (f"completion check failed: count_ok={count_ok} "
                             f"centroid_ok={centroid_ok} speed_ok={speed_ok}")
        self.reporter.log("run_end", task_id=task_id,
                          final_state=result.final_state.value,
                          metrics=result.to_dict())
        return result

    # ---------------- helpers
    def _ball_radius(self, snap: WorkspaceSnapshot, tid: int) -> float:
        p = snap.particle(tid)
        return p.radius_px if p else \
            self.config.controller.model.ball_radius_px

    def _controller_estopped(self, result: AggregationResult) -> bool:
        return False  # 占位：外层急停由各 controller 内部处理


def _ring_targets(center: Point, radius: float, n: int,
                  spacing: float) -> List[Point]:
    """环形驻点；n=1 时为中心；间距不足时扩大半径。"""
    if n <= 1:
        return [center]
    r = radius
    min_r = (spacing * n) / (2 * math.pi)
    r = max(r, min_r)
    return [(center[0] + r * math.cos(2 * math.pi * i / n),
             center[1] + r * math.sin(2 * math.pi * i / n)) for i in range(n)]
