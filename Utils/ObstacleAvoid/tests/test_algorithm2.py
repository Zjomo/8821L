import math

import pytest

from obstacle_avoidance.algorithm2 import (Alg2Config, Alg2Stage,
                                            FixedBeamController)
from obstacle_avoidance.cli import build_scenario
from obstacle_avoidance.controller import ControllerConfig
from obstacle_avoidance.models import RunState
from obstacle_avoidance.motion import MotionStateError
from obstacle_avoidance.models import GoalRegion
from obstacle_avoidance.planner import GridPlanner
from obstacle_avoidance.reporter import RunReporter
from obstacle_avoidance.sim_microscope import build_sim_scenario
from obstacle_avoidance.vision import ParticleTracker, VisionPipeline


def test_alg2_stage_moves_xyz_and_records_z_focus():
    world, _ = build_scenario("sim01")
    try:
        stage = Alg2Stage(world)
        stage.prepare_focus()
        assert world.motion_stage.last_telemetry is not None
        assert world.motion_stage.last_telemetry.source in {"alg2-z-safe", "alg2-z-focus"}
        stage.move_by(0.001, 0.0)
        assert stage.last_command is not None
        assert world.motion_stage.last_telemetry is not None
        assert set(world.motion_stage.last_telemetry.applied_delta) == {"x", "y", "z"}
    finally:
        world.close()


def test_alg2_single_ball_uses_fixed_beam_stage():
    world, run = build_scenario("sim01")
    try:
        # The public scenario remains Alg1-compatible; explicitly exercise the
        # fixed-beam adapter through the same controller contract.
        stage = Alg2Stage(world, track_id=None)
        stage.prepare_focus()
        assert world.motion_stage.position["z"] == pytest.approx(0.0)
    finally:
        world.close()


def test_fixed_beam_controller_aligns_with_laser_off_then_transports():
    world, _ = build_sim_scenario(static_obstacles=[])
    try:
        world.alg2_mode = True
        detector = world.make_detector()
        vision = VisionPipeline(
            detector, tracker=ParticleTracker(max_jump_px=220))
        world.bind_pipeline(vision)
        detected = vision.process(world.render(), 1).particles[0]
        world.set_alg2_target(detected.track_id)
        world.set_beam_position((400.0, 300.0))
        stage = Alg2Stage(
            world, detected.track_id,
            config=Alg2Config(beam_position_px=(400.0, 300.0),
                              image_shift_sign=-1))
        cfg = ControllerConfig(
            max_step_mm=0.005, tolerance_px=8.0, stable_frames=2,
            max_iterations=100, max_track_jump_px=250.0,
            slip_abort_px=80.0)
        report = RunReporter(None)

        result = FixedBeamController(
            stage, vision, GridPlanner(), cfg, report).run(
                world.snapshot(), detected.track_id,
                GoalRegion((500.0, 400.0), 20.0),
                get_frame=world.render, get_snapshot=world.snapshot)

        assert result.final_state.value == "COMPLETE"
        assert stage.laser_gate.history[0][1] is False
        assert any(enabled for _timestamp, enabled in stage.laser_gate.history)
        assert stage.laser_gate.history[-1][1] is False
        assert world.laser_enabled is False
        # Net shift includes both the initial ball-to-beam alignment and the
        # transport phase, so only the final sample-goal/beam relation matters.
        moving_goal = stage.sample_point_to_camera((500.0, 400.0))
        assert moving_goal == pytest.approx((400.0, 300.0), abs=8.0)
    finally:
        world.close()


def _alg2_scenario(beam=(400.0, 300.0), static_obstacles=None, **cfg_kwargs):
    """开一个 sim 世界 + 目标球 + 光斑，返回 (world, vision, detected, stage, cfg)。

    static_obstacles=None 用场景默认障碍（样本坐标，初始落在窗口 (480,435)）；
    传序列则按样本坐标覆盖。
    """
    world, _ = build_sim_scenario(
        **({} if static_obstacles is None
           else {"static_obstacles": list(static_obstacles)}))
    world.alg2_mode = True
    vision = VisionPipeline(world.make_detector(),
                            tracker=ParticleTracker(max_jump_px=220))
    world.bind_pipeline(vision)
    detected = vision.process(world.render(), 1).particles[0]
    world.set_alg2_target(detected.track_id)
    world.set_beam_position(beam)
    stage = Alg2Stage(
        world, detected.track_id,
        config=Alg2Config(
            beam_position_px=beam,
            image_shift_sign=cfg_kwargs.pop("image_shift_sign", -1),
            shift_probe_mm=cfg_kwargs.pop("shift_probe_mm", 0.2)))
    cfg = ControllerConfig(
        max_step_mm=cfg_kwargs.pop("max_step_mm", 0.005),
        tolerance_px=8.0, stable_frames=2,
        max_iterations=100, max_track_jump_px=250.0, slip_abort_px=80.0,
        **cfg_kwargs)
    return world, vision, detected, stage, cfg


def test_alg2_probe_calibrates_shift_gain_and_recovers_flipped_sign():
    """符号配反（image_shift_sign=+1）时，探针实测增益必须把方向纠正过来。

    现场最典型的故障：符号配反 -> 对准时球越走越远 -> 永远对不齐 -> 中止，
    且电机一路朝同一方向走。
    """
    world, vision, detected, stage, cfg = _alg2_scenario(
        image_shift_sign=1, static_obstacles=[])
    try:
        report = RunReporter(None)
        result = FixedBeamController(stage, vision, GridPlanner(), cfg,
                                     report).run(
            world.snapshot(), detected.track_id,
            GoalRegion((500.0, 400.0), 20.0),
            get_frame=world.render, get_snapshot=world.snapshot)

        assert result.final_state.value == "COMPLETE"
        calib = [e for e in report.events if e["event"] == "shift_calibration"]
        assert calib and calib[0]["source"] == "probe"
        # 台位 +mm 让球在画面里朝 -x/-y 走（探针测出的就是真实增益）
        assert stage.shift_gain[0] < 0 and stage.shift_gain[1] < 0
        assert stage.sample_point_to_camera((500.0, 400.0)) == pytest.approx(
            (400.0, 300.0), abs=8.0)
    finally:
        world.close()


def test_alg2_without_probe_flipped_sign_never_aligns():
    """反证：关掉探针且符号配反 -> 球越走越远，对准不可能收敛。"""
    world, vision, detected, stage, cfg = _alg2_scenario(
        image_shift_sign=1, shift_probe_mm=0.0, static_obstacles=[])
    try:
        report = RunReporter(None)
        result = FixedBeamController(stage, vision, GridPlanner(), cfg,
                                     report).run(
            world.snapshot(), detected.track_id,
            GoalRegion((500.0, 400.0), 20.0),
            get_frame=world.render, get_snapshot=world.snapshot)

        assert result.final_state.value != "COMPLETE"
        assert result.failure_reason is not None
        calib = [e for e in report.events if e["event"] == "shift_calibration"]
        assert calib and calib[0]["source"] == "configured"
    finally:
        world.close()


def test_alg2_alignment_plans_around_obstacle():
    """对准阶段也走规划绕障：直线会把球顶进障碍，必须给出绕行路径。

    场景默认障碍（样本 (1580,1135) r=55）初始正落在球 (320,450) 与
    光斑 (620,450) 的连线上。
    """
    world, vision, detected, stage, cfg = _alg2_scenario(beam=(620.0, 450.0))
    try:
        report = RunReporter(None)
        result = FixedBeamController(stage, vision, GridPlanner(), cfg,
                                     report).run(
            world.snapshot(), detected.track_id,
            GoalRegion((620.0, 450.0), 20.0),
            get_frame=world.render, get_snapshot=world.snapshot)

        plans = [e for e in report.events
                 if e["event"] == "plan" and e.get("stage") == "alignment"]
        assert plans, "对准阶段必须产出 plan 事件（UI 靠它画实时路径）"
        first = plans[0]["waypoints_px"]
        assert len(first) >= 3, (
            f"障碍挡路时对准路径必须绕行（>2 个航点），实际 {first}")
        assert result.iterations > 0
    finally:
        world.close()


def test_alg2_near_beam_enters_tracking_and_recaptures():
    """『尽量对准』：对准预算用尽但球已接近光斑 -> 开光进入跟踪并再捕获。

    走这条路必须靠方向正确的 beam_recapture（+offset 把球心推回光斑），
    方向反了球会越飘越远直到 SPOT_SLIP。探针关掉，保证初始几何确定。
    """
    world, vision, detected, stage, cfg = _alg2_scenario(
        beam=(340.0, 450.0), beam_alignment_max_steps=2, max_step_mm=0.0025,
        shift_probe_mm=0.0, static_obstacles=[])
    try:
        report = RunReporter(None)
        result = FixedBeamController(stage, vision, GridPlanner(), cfg,
                                     report).run(
            world.snapshot(), detected.track_id,
            GoalRegion((500.0, 400.0), 20.0),
            get_frame=world.render, get_snapshot=world.snapshot)

        assert [e for e in report.events
                if e["event"] == "beam_alignment_relaxed"], "应放行进入跟踪"
        assert [e for e in report.events
                if e["event"] == "beam_recapture"], "开光后应动态再捕获"
        assert result.final_state.value == "COMPLETE"
    finally:
        world.close()


def test_alg2_every_command_respects_max_step_mm():
    """「单步位移」必须对闭环保真：探针/对准/跟踪下发的每一步都不得超限。

    电机模式下 max_step_mm = 单步位移(step) / steps_per_mm，同时进
    ControllerConfig（决策侧）与驱动（硬限位，超限直接拒绝）。
    """
    world, vision, detected, stage, cfg = _alg2_scenario(
        static_obstacles=[], max_step_mm=0.005, shift_probe_mm=0.0)
    try:
        report = RunReporter(None)
        FixedBeamController(stage, vision, GridPlanner(), cfg, report).run(
            world.snapshot(), detected.track_id,
            GoalRegion((500.0, 400.0), 20.0),
            get_frame=world.render, get_snapshot=world.snapshot)

        commands = stage._xy.commands     # Alg2 台位适配器记录的真实命令
        assert commands, "闭环未下发任何台位命令"
        worst = max(math.hypot(c.dx_mm, c.dy_mm) for c in commands)
        assert worst <= cfg.max_step_mm + 1e-9, (
            f"单步位移超限: {worst:.6f}mm > {cfg.max_step_mm}mm")
    finally:
        world.close()
