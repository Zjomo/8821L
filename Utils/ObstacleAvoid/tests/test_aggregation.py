"""聚拢任务 AG-01..AG-05 黑盒测试。"""
import math

import pytest

from obstacle_avoidance.aggregation import (AggregationConfig,
                                             AggregationPlanner)
from obstacle_avoidance.cli import build_scenario
from obstacle_avoidance.models import (ConfigError, FailureReason, GoalRegion,
                                       RunState)
from obstacle_avoidance.planner import CollisionModel, GridPlanner
from obstacle_avoidance.reporter import RunReporter
from obstacle_avoidance.vision import VisionPipeline

R = CollisionModel().ball_radius_px


def test_ag01_single_ball_aggregation():
    world, run = build_scenario("ag01")
    result = run(rep=RunReporter(None))
    assert result.completed
    assert result.final_state == RunState.COMPLETE
    pos = world.particle_position(1)
    assert math.dist(pos, (470, 240)) <= 60 - R


def test_ag02_multi_ball_crossing_paths():
    world, run = build_scenario("ag02")
    rep = RunReporter(None)
    result = run(rep=rep)
    assert result.completed, result.detail
    assert result.final_state == RunState.COMPLETE
    # 分配唯一且覆盖所有球
    assert len(result.assignment) == 4
    targets = list(result.assignment.values())
    assert len({(round(t[0]), round(t[1])) for t in targets}) == 4
    # 区域内数量与质心偏差指标
    assert result.metrics["count_in_region"] >= 4
    assert result.metrics["centroid_dev_px"] <= 30.0


def test_ag03_overlapping_balls_keep_tracks():
    """两球检测框重叠阶段 -> 不重复计数，ID 保持。"""
    world, _ = build_scenario("ag02")
    vis = VisionPipeline(expected_radius_px=R)
    # 移到接近但不完全重合（间隔 20px < 2R）
    world.move_particle(1, 60, 0)     # (180,120)
    world.move_particle(2, -60, 0)    # (460,360) 不重叠——改为构造局部重叠
    w2 = build_scenario("ag02")[0]
    w2.move_particle(3, 190, -110)    # (310,250) 靠近中心球? 中心无球
    # 直接构造重叠对：ball1 (120,120) -> (300,230)，ball4 (520,120) -> (316,236)
    w3 = build_scenario("ag02")[0]
    w3.move_particle(1, 180, 110)     # (300,230)
    w3.move_particle(4, -204, 116)    # (316,236)
    res = vis.process(w3.render(), frame_id=1)
    # 断言：要么显式 uncertain（粘连），要么仍是两个独立 track
    if not res.uncertain:
        assert len(res.particles) == 2
        assert len({p.track_id for p in res.particles}) == 2


def test_ag04_region_outside_substrate_rejected():
    world, _ = build_scenario("ag01")
    snap = world.snapshot()
    ag = AggregationPlanner(GridPlanner(),
                            VisionPipeline(expected_radius_px=R),
                            AggregationConfig(required_count=1))
    with pytest.raises(ConfigError):
        ag.run(world, snap, GoalRegion(center=(650, 240), radius_px=60),
               task_id="ag04-out")


def test_ag04_region_capacity_insufficient_rejected():
    world, _ = build_scenario("ag02")
    snap = world.snapshot()
    ag = AggregationPlanner(GridPlanner(),
                            VisionPipeline(expected_radius_px=R),
                            AggregationConfig(required_count=4))
    # 半径 12 区域装不下 4 个 r=12 的球
    with pytest.raises(ConfigError):
        ag.run(world, snap, GoalRegion(center=(320, 240), radius_px=12),
               task_id="ag04-cap")


def test_ag05_estop_mid_task_aborts_cleanly():
    """AG-05: 运行中急停 -> 新命令停止，状态 ABORTED。"""
    import threading
    import time as _t
    world, run = build_scenario("ag01")
    rep = RunReporter(None)
    # 包一层：第一次 stage 命令后触发急停
    snap0 = world.snapshot()
    from obstacle_avoidance.controller import (ControllerConfig,
                                                ObstacleAvoidController)
    stage = world.make_stage(1)
    ctl = ObstacleAvoidController(stage, VisionPipeline(expected_radius_px=R),
                                  GridPlanner(), ControllerConfig(), rep)
    orig_move = stage.move_by

    def move_then_estop(*a, **kw):
        r = orig_move(*a, **kw)
        ctl.request_estop()
        return r

    stage.move_by = move_then_estop  # monkey patch（测试专用）
    result = ctl.run(snap0, 1,
                     GoalRegion(center=(470, 240), radius_px=6 + 1),
                     task_id="ag05", get_frame=world.render,
                     get_snapshot=world.snapshot)
    assert result.final_state == RunState.ABORTED
    assert result.failure_reason == FailureReason.ESTOP
    n_cmds = len(stage.commands)
    _t.sleep(0.02)
    assert len(stage.commands) == n_cmds  # 急停后无新命令


def test_aggregation_requires_min_balls():
    world, _ = build_scenario("ag01")
    snap = world.snapshot()
    ag = AggregationPlanner(GridPlanner(),
                            VisionPipeline(expected_radius_px=R),
                            AggregationConfig(required_count=2))
    with pytest.raises(ConfigError):
        ag.run(world, snap, GoalRegion(center=(470, 240), radius_px=60),
               task_id="ag-min")
