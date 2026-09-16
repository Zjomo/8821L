"""聚拢任务与分配策略（PLAN 第 5 节）。

- Hungarian（scipy.linear_sum_assignment）把球分配到聚拢区域内的互不重叠驻点。
- 逐球闭环执行；已到达球作为安全占位（静态障碍）参与后续规划。
- 完成判定（闭环模式）：区域内数量、质心偏差、速度阈值。
- 区域非法（衬底外 / 容量不足 / clearance 不足）-> 拒绝任务。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
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
    def _effective_radius(self, snap: WorkspaceSnapshot,
                          track_ids: Sequence[int]) -> float:
        """碰撞体积有效半径：模型默认值与所有球实际检测半径取最大。

        驻点间距/区域容量按此值计算，保证相邻球驻点不重叠。
        """
        r = self.config.controller.model.ball_radius_px
        for tid in track_ids:
            p = snap.particle(tid)
            if p is not None:
                r = max(r, float(p.radius_px))
        return r

    def validate_region(self, snap: WorkspaceSnapshot,
                        region: GoalRegion, n_balls: int,
                        ball_radius_px: Optional[float] = None) -> None:
        """AG-04：区域非法时拒绝任务（ConfigError）。"""
        ccfg = self.config.controller
        r_eff = (ccfg.model.ball_radius_px if ball_radius_px is None
                 else max(ccfg.model.ball_radius_px, float(ball_radius_px)))
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
        """组装驻点分配：按球到范围中心的距离编号（最近 = 0 号）。

        0 号球驻点 = 范围中心；其余球在中心外围环形紧贴（球心距
        2*r_eff，相切不重叠）。环上驻点与剩余球之间用 Hungarian
        （无 scipy 时贪心）最小化总位移。
        """
        r_eff = self._effective_radius(snap, track_ids)
        # 编号：按球心到范围中心距离升序（0 = 最近）
        ordered = sorted(track_ids,
                         key=lambda tid: math.dist(
                             snap.particle(tid).position_px, region.center))
        first, rest = ordered[0], ordered[1:]
        targets = self._assembly_slots(region.center, len(rest), r_eff)
        # 驻点可行性：紧贴环可能落进障碍/边界膨胀区，把不可行驻点就近
        # 挪到可行点；实在挪不动则保留原值（由控制器给出明确拒绝原因）。
        targets = [
            (t if self.grid.check_point(snap, t) is None
             else (self.grid.nearest_feasible(
                 snap, t,
                 max_r_px=self.grid.config.model.inflation_px + 4.0,
                 step_px=2.0) or t))
            for t in targets]
        mapping = {first: region.center}
        if rest:
            cost = [[math.dist(snap.particle(tid).position_px, t)
                     for t in targets] for tid in rest]
            try:
                from scipy.optimize import linear_sum_assignment
                rows, cols = linear_sum_assignment(cost)
                mapping.update({rest[r]: targets[c]
                                for r, c in zip(rows, cols)})
            except ImportError:  # pragma: no cover
                remaining = set(range(len(targets)))
                for tid in rest:
                    p = snap.particle(tid)
                    c = min(remaining,
                            key=lambda i: math.dist(p.position_px,
                                                    targets[i]))
                    mapping[tid] = targets[c]
                    remaining.discard(c)
        return mapping

    @staticmethod
    def _assembly_slots(center: Point, n_ring: int,
                        r_eff: float) -> List[Point]:
        """中心外的环形紧贴驻点：第 k 层半径 2*r_eff*k + 余量，容量 6k 槽。"""
        # 紧贴驻点的外扩余量：余量同时决定转运时与已就位球的最近距离
        # （见 placed 障碍的球-球信用扣减），余量过小会让轮廓在转运中
        # 重叠粘连，破坏跟踪 ID。9px 保证轮廓分离，视觉上仍近似紧贴。
        slack = 9.0
        slots: List[Point] = []
        k = 1
        while len(slots) < n_ring:
            n_here = min(6 * k, n_ring - len(slots))
            r = 2.0 * r_eff * k + slack
            for i in range(n_here):
                a = 2 * math.pi * i / n_here
                slots.append((center[0] + r * math.cos(a),
                              center[1] + r * math.sin(a)))
            k += 1
        return slots

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
        self.validate_region(snap, region, len(track_ids),
                             ball_radius_px=self._effective_radius(snap, track_ids))
        self.reporter.log("task_config", task_id=task_id, kind="aggregation",
                          region=region.to_dict(),
                          track_ids=list(track_ids),
                          required_count=cfg.required_count)

        assignment = self.assign(snap, region, track_ids)
        result = AggregationResult(final_state=RunState.IDLE,
                                   assignment=dict(assignment))

        # 编号顺序执行（需求：按球到范围中心距离编号，0 号先到中心；
        # 就近优先也减少交叉与等待）
        order = sorted(track_ids,
                       key=lambda tid: math.dist(
                           snap.particle(tid).position_px, region.center))
        placed: List[Obstacle] = []   # 已到达球的安全占位
        for tid in order:
            if self._controller_estopped(result):
                break
            target = assignment[tid]
            # 紧贴驻点可达性：已就位球的实际落位（容差 tolerance_px 内）
            # 叠加检测抖动，可能把理想驻点压进膨胀区，导致后到球被
            # "goal rejected: low_clearance" 中止。把驻点就近微调到当前
            # 可行点（保持环上方位，球仍尽量贴近已就位球）。
            cur = world.snapshot()
            if self.grid.check_point(cur, target, list(placed)) is not None:
                nudged = self.grid.nearest_feasible(
                    cur, target, list(placed),
                    max_r_px=self.grid.config.model.inflation_px + 12.0,
                    step_px=1.0)
                if nudged is not None:
                    self.reporter.log(
                        "seat_adjusted", task_id=task_id, track_id=tid,
                        detail="assembly slot nudged to feasible point",
                        original=[round(target[0], 1), round(target[1], 1)],
                        adjusted=[round(nudged[0], 1), round(nudged[1], 1)])
                    target = nudged
                    assignment[tid] = nudged
            goal = GoalRegion(center=target,
                              radius_px=cfg.controller.tolerance_px + 1.0)
            # 已就位球的位置以 placed（控制器回读的落位点）为准：粘连帧
            # 会把 tracker 里已就位 track 的位置污染到两球中点，快照里的
            # ball-N 障碍随之失真并挡住后到球的驻点。过滤掉这些已就位
            # track 的快照障碍，避免用被污染的位置做碰撞判定。
            placed_ids = {int(ob.obstacle_id.split("-", 1)[1])
                          for ob in placed}

            def _filtered_snapshot(_ids=tuple(placed_ids)):
                s = world.snapshot()
                if _ids:
                    s = replace(s, obstacles=[
                        o for o in s.obstacles
                        if not (o.obstacle_id or "").startswith("ball-")
                        or int(o.obstacle_id.split("-", 1)[1]) not in _ids])
                return s
            # Worlds that render/track a designated target omit that ball from
            # the dynamic-obstacle list.  Rotate the designation for every
            # sequentially controlled ball; otherwise the previously selected
            # ball could disappear from collision planning in later runs.
            if hasattr(world, "target_track_id"):
                world.target_track_id = tid
            if getattr(world, "alg2_mode", False) and hasattr(world, "set_alg2_target"):
                world.set_alg2_target(tid)
                self.reporter.log("beam_target", task_id=f"{task_id}/ball{tid}",
                                  track_id=tid, action="select",
                                  algorithm="Alg2")
            if self.stage_factory is not None:
                stage = self.stage_factory()
            else:
                try:
                    if getattr(world, "alg2_mode", False) and hasattr(world, "make_alg2_stage"):
                        from .algorithm2 import Alg2Stage
                        stage = Alg2Stage(world, tid)
                        stage.prepare_focus()
                    else:
                        stage = world.make_stage(tid)
                except TypeError:
                    stage = world.make_stage()
            # Alg2 uses a stationary camera beam. Every ball therefore gets
            # its own fixed-beam controller: laser-off alignment first, then
            # laser-on transport while the substrate and obstacles drift with
            # the stage. Keep Alg1 on the original controller contract.
            if getattr(world, "alg2_mode", False):
                from .algorithm2 import (Alg2Config, Alg2Stage,
                                         FixedBeamController)
                if not isinstance(stage, Alg2Stage):
                    beam = getattr(world, "beam_position_px", None)
                    stage = Alg2Stage(
                        world, tid,
                        config=Alg2Config(
                            beam_position_px=beam,
                            beam_calibration_confidence=1.0,
                            image_shift_sign=-1),
                        xy_stage=stage)
                controller = FixedBeamController(
                    stage=stage, vision=self.vision, planner=self.grid,
                    config=cfg.controller, reporter=self.reporter)
            else:
                controller = ObstacleAvoidController(
                    stage=stage, vision=self.vision, planner=self.grid,
                    config=cfg.controller, reporter=self.reporter)
            if self.controller_sink is not None:
                self.controller_sink["controller"] = controller
            try:
                run = controller.run(
                    _filtered_snapshot(), tid, goal,
                    task_id=f"{task_id}/ball{tid}",
                    extra_obstacles=list(placed),
                    get_frame=world.render,
                    get_snapshot=_filtered_snapshot)
            finally:
                # Hardware assembly opens a controller per ball so that every
                # task can select a new track.  Release the USB/serial handle
                # before the next ball is started; virtual stages simply have
                # no close method and remain unaffected.
                close = getattr(stage, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:  # noqa: BLE001 - preserve run result
                        pass
            result.per_ball[tid] = run
            if run.final_state == RunState.COMPLETE:
                # The completed ball is released before selecting the next
                # one. Capture one fresh frame after laser-off so all
                # sample-bound objects (including the placed ball) reflect
                # the released stage coordinate, avoiding a false overlap at
                # the fixed beam in the next task.
                off = getattr(stage, "set_laser_enabled", None)
                if callable(off):
                    off(False)
                try:
                    frame = world.render()
                    self.vision.process(frame, int(getattr(world, "frame_counter", 0)))
                except Exception:  # noqa: BLE001 - preserve completed result
                    pass
                resolved_tid = int(getattr(stage, "track_id", tid))
                try:
                    pos = world.particle_position(resolved_tid)
                except KeyError:
                    # A detector may reassign a contour ID while the target
                    # remains physically locked. Use the controller's final
                    # resolved ID when available, otherwise fail safely.
                    result.final_state = RunState.ABORTED
                    result.failure_reason = FailureReason.TARGET_LOST
                    result.detail = f"ball {tid}: final target position unavailable"
                    self.reporter.log("run_end", task_id=task_id,
                                      final_state=result.final_state.value,
                                      metrics=result.to_dict())
                    return result
                placed.append(Obstacle(
                    kind="circle", center=pos,
                    # 已就位球是静态球，组装语义允许后续球靠近。全膨胀
                    # （r + spot + safety）对"球-球"接触过度膨胀；只保留
                    # 2px 信用：转运最近距离 = 2r + slack - 10 ≈ 轮廓刚好
                    # 分离，既不粘连破坏跟踪，也尽量贴近。
                    radius=max(0.0, self._ball_radius(snap, tid) - 2.0),
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
        # 停稳判定：补处理两帧再取快照。tracker 速度按帧间位移计算，
        # 最后一个控制帧常残留 1~2px 量化移动，直接判定会把已就位的
        # 任务误报 speed_ok=False -> ABORTED。
        try:
            fid = int(getattr(world, "frame_counter", 0))
            for _ in range(2):
                fid += 1
                self.vision.process(world.render(), frame_id=fid)
        except Exception:  # noqa: BLE001 - 渲染失败不影响既有快照判定
            pass
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
