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


def _synthetic_snap(radius_px: float, n: int = 4):
    """合成快照：n 个半径 radius_px 的球（用于碰撞体积单元测试）。"""
    from obstacle_avoidance.models import (CoordinateTransform, Particle,
                                           SubstrateRegion,
                                           WorkspaceSnapshot)
    sub = SubstrateRegion(polygon=[(0, 0), (400, 0), (400, 400), (0, 400)])
    parts = [Particle(track_id=i + 1, position_px=(100.0 + i * 10.0, 100.0),
                      radius_px=radius_px) for i in range(n)]
    return WorkspaceSnapshot(frame_id=1, timestamp=0.0, substrate=sub,
                             obstacles=[], particles=parts,
                             transform=CoordinateTransform(),
                             frame_size=(400, 400))


def test_assign_spacing_uses_actual_radius():
    """驻点间距按实际球半径计算：大球（>模型默认）驻点互不重叠。"""
    ag = AggregationPlanner(GridPlanner())
    snap = _synthetic_snap(radius_px=30.0)
    region = GoalRegion(center=(200, 200), radius_px=120)
    mapping = ag.assign(snap, region, [1, 2, 3, 4])
    targets = list(mapping.values())
    # 实际半径 30 > 模型 12：间距必须 >= 2*30*1.2 = 72
    min_d = min(math.dist(a, b) for i, a in enumerate(targets)
                for b in targets[i + 1:])
    assert min_d >= 2 * 30.0 * 1.2 - 1e-6


def test_validate_region_capacity_uses_actual_radius():
    """区域容量校验按实际球半径：大球装不下时拒绝。"""
    ag = AggregationPlanner(GridPlanner(),
                            AggregationConfig(required_count=4))
    snap = _synthetic_snap(radius_px=30.0)
    # 容量按 r=30 校验：need = 4*pi*30^2*1.35 ≈ 15268 > pi*40^2
    with pytest.raises(ConfigError):
        ag.validate_region(snap, GoalRegion(center=(200, 200), radius_px=40),
                           4, ball_radius_px=30.0)


def test_controller_adapts_ball_radius_only_grows():
    """运动球实际半径并入碰撞模型：只增不减，增大返回 True。"""
    from obstacle_avoidance.controller import ObstacleAvoidController
    from obstacle_avoidance.models import Particle
    ctl = ObstacleAvoidController(stage=None)
    p = Particle(track_id=1, position_px=(0.0, 0.0), radius_px=30.0)
    assert ctl._adapt_ball_radius(p) is True
    assert ctl.planner.config.model.ball_radius_px == 30.0
    assert ctl._adapt_ball_radius(p) is False   # 已并入，不重复触发重规划


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


def test_sim_microscope_multi_ball_aggregation_binds_each_target():
    """显微镜仿真聚拢必须逐球绑定，不能把整个位移台视作所有球一起移动。"""
    pytest.importorskip("simulator_app")
    from obstacle_avoidance.controller import ControllerConfig
    from obstacle_avoidance.sim_microscope import _initial_detect
    from obstacle_avoidance.vision import ParticleTracker

    layout = {
        "grounds": [[40, 40, 500, 400]],
        "balls": [[100, 200, 30, 30], [180, 220, 30, 30]],
        "obstacles": [[300, 150, 60, 60]],
    }
    world, _ = build_scenario("sim01", sim_layout=layout)
    try:
        detector = world.make_detector()
        pipeline = VisionPipeline(
            detector, tracker=ParticleTracker(max_jump_px=220.0))
        snap, _ = _initial_detect(world, detector, pipeline=pipeline)
        ag = AggregationPlanner(
            GridPlanner(), pipeline,
            AggregationConfig(
                required_count=2,
                controller=ControllerConfig(
                    max_step_mm=0.05, tolerance_px=8.0, stable_frames=3,
                    max_iterations=400, max_track_jump_px=250.0,
                    prefer_track_id=True)))
        result = ag.run(world, snap, GoalRegion(center=(400, 340), radius_px=70),
                        task_id="sim-ag")
        assert result.completed, result.detail
    finally:
        world.close()
