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
