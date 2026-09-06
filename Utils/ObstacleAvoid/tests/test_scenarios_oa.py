"""OA-01..07 场景黑盒测试（dryrun 仿真，对照 PLAN 测试表）。"""
import math

import numpy as np
import pytest

from obstacle_avoidance.cli import build_scenario
from obstacle_avoidance.controller import (ControllerConfig,
                                            ObstacleAvoidController)
from obstacle_avoidance.models import FailureReason, GoalRegion, RunState
from obstacle_avoidance.planner import CollisionModel, GridPlanner
from obstacle_avoidance.reporter import RunReporter, replay
from obstacle_avoidance.simulator import SimWorld
from obstacle_avoidance.vision import VisionPipeline

R = CollisionModel().ball_radius_px  # 12


def _substrate():
    from obstacle_avoidance.models import SubstrateRegion
    return SubstrateRegion(polygon=[(40, 40), (600, 40), (600, 440), (40, 440)],
                           safety_margin_px=6.0)


def _controller(world, rep=None, **cfg):
    snap = world.snapshot()
    stage = world.make_stage(1)
    ctl = ObstacleAvoidController(
        stage, VisionPipeline(expected_radius_px=R), GridPlanner(),
        ControllerConfig(**cfg), rep)
    return snap, stage, ctl


def test_oa01_straight_path_no_obstacles():
    """OA-01: 无障碍 -> 直线路径闭环到达。"""
    world, run = build_scenario("oa01")
    rep = RunReporter(None)
    result = run(rep=rep)
    assert result.final_state == RunState.COMPLETE
    assert result.final_error_px is not None and result.final_error_px <= 6.0
    assert result.replan_count >= 1
    assert len(result.stage_commands) >= 5  # 330px / 30px 步长


def test_oa02_static_obstacle_detour():
    """OA-02: 静态障碍横跨直线路径 -> 绕行且 clearance 达标。"""
    world, run = build_scenario("oa02")
    result = run(rep=RunReporter(None))
    assert result.final_state == RunState.COMPLETE
    infl = CollisionModel().inflation_px
    assert result.min_clearance_observed_px >= infl - 0.5
    # 实际路径必须绕行：存在非零横向位移分量
    dxs = [c.dx_mm for c in result.stage_commands]
    assert any(abs(d) > 1e-6 for d in dxs)


def test_oa03_no_safe_path_no_motor():
    """OA-03: 障碍封闭目标 -> NO_SAFE_PATH 且无电机命令。"""
    world, run = build_scenario("oa03")
    x0 = 150.0
    result = run(rep=RunReporter(None))
    assert result.failure_reason in (FailureReason.NO_SAFE_PATH,)
    assert result.final_state == RunState.ABORTED
    assert len(result.stage_commands) == 0
    assert math.dist(world.particle_position(1), (x0, 240)) < 1e-6


def test_oa04_goal_out_of_bounds_rejected():
    """OA-04: 终点在衬底外 -> 拒绝任务并给出越界原因。"""
    world, run = build_scenario("oa04")
    result = run(rep=RunReporter(None))
    assert result.failure_reason == FailureReason.OUT_OF_BOUNDS
    assert result.final_state == RunState.ABORTED
    assert len(result.stage_commands) == 0


def test_oa05_dynamic_obstacle_replan():
    """OA-05: 移动中障碍突现 -> 安全停止并重规划（clearance 始终达标）。"""
    world, run = build_scenario("oa05")
    rep = RunReporter(None)
    result = run(rep=rep)
    assert result.final_state == RunState.COMPLETE
    assert result.replan_count >= 2  # 初始规划 + 障碍突现后重规划
    plan_events = [e for e in rep.events if e["event"] == "plan"]
    assert len(plan_events) >= 2
    infl = CollisionModel().inflation_px
    assert result.min_clearance_observed_px >= infl - 0.5


def test_oa06_detection_uncertain_no_blind_move():
    """OA-06: 连续丢帧/低置信度 -> DETECTION_UNCERTAIN，不盲动，恢复后完成。"""
    world, _ = build_scenario("oa01")
    # 帧 3-5 遮挡球
    world.on_frame(lambda fid: world.set_occluded(1, 3 <= fid <= 5))
    rep = RunReporter(None)
    snap, stage, ctl = _controller(world, rep)
    result = ctl.run(snap, 1, GoalRegion(center=(480, 240), radius_px=20),
                     task_id="oa06", get_frame=world.render,
                     get_snapshot=world.snapshot)
    assert result.final_state == RunState.COMPLETE
    # 不确定帧上没有电机命令
    cmd_frames = {c.frame_id for c in result.stage_commands}
    assert not (cmd_frames & {3, 4, 5})


def test_oa07_completion_tolerance_and_stable_frames():
    """OA-07: 距离<=tolerance 且连续稳定帧 -> COMPLETE 并记录最终误差。"""
    world, _ = build_scenario("oa01")
    snap, stage, ctl = _controller(world, rep=None, stable_frames=3,
                                   tolerance_px=6.0)
    result = ctl.run(snap, 1, GoalRegion(center=(480, 240), radius_px=20),
                     task_id="oa07", get_frame=world.render,
                     get_snapshot=world.snapshot)
    assert result.final_state == RunState.COMPLETE
    assert result.final_error_px <= 6.0


def test_estop_latency_and_no_commands_after():
    """补充 OA/AG-05 核心：急停 <=100ms（仿真）且急停后无新命令。"""
    world, _ = build_scenario("oa01")
    snap, stage, ctl = _controller(world)
    orig_move = stage.move_by

    def move_then_estop(*a, **kw):
        r = orig_move(*a, **kw)
        ctl.request_estop()  # 第一条命令后立即急停
        return r

    stage.move_by = move_then_estop
    result = ctl.run(snap, 1, GoalRegion(center=(480, 240), radius_px=20),
                     task_id="estop", get_frame=world.render,
                     get_snapshot=world.snapshot)
    assert result.final_state == RunState.ABORTED
    assert result.failure_reason == FailureReason.ESTOP
    assert result.estop_latency_s is not None
    assert result.estop_latency_s <= 0.1  # 100 ms 仿真门限
    assert result.commands_after_estop == 0
    assert len(result.stage_commands) == 1  # 急停后未再发出命令
