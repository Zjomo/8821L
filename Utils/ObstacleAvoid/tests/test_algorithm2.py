import pytest

from obstacle_avoidance.algorithm2 import Alg2Stage
from obstacle_avoidance.cli import build_scenario
from obstacle_avoidance.models import RunState
from obstacle_avoidance.motion import MotionStateError


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

