import cv2
import pytest

from obstacle_avoidance.sim_microscope import build_sim_scenario
from obstacle_avoidance.vision import ParticleTracker, VisionPipeline
from obstacle_avoidance.algorithm2 import Alg2Stage


def test_alg2_renders_green_fixed_beam_spot():
    world, _ = build_sim_scenario()
    try:
        world.alg2_mode = True
        detector = world.make_detector()
        pipeline = VisionPipeline(detector, tracker=ParticleTracker(max_jump_px=220))
        world.bind_pipeline(pipeline)
        vis = pipeline.process(world.render(), frame_id=1)
        assert vis.particles
        world.target_track_id = vis.particles[0].track_id
        world.set_laser_enabled(True)
        frame = world.render()
        # OpenCV BGR green ring/center must be visible in the rendered frame.
        green = (frame[:, :, 1] > 180) & (frame[:, :, 0] < 80) & (frame[:, :, 2] < 80)
        assert int(green.sum()) > 20
    finally:
        world.close()


def test_alg2_stage_moves_entire_workspace_and_keeps_trapped_ball_at_beam():
    world, _ = build_sim_scenario()
    try:
        world.alg2_mode = True
        detector = world.make_detector()
        pipeline = VisionPipeline(detector, tracker=ParticleTracker(max_jump_px=220))
        world.bind_pipeline(pipeline)
        first = pipeline.process(world.render(), frame_id=1)
        ball = first.particles[0]
        world.set_alg2_target(ball.track_id)
        world.set_beam_position(ball.position_px)
        stage = Alg2Stage(world, ball.track_id)

        before_origin = world._view_origin()
        before_ball = world._to_window(world._balls[0][:2])
        stage.set_laser_enabled(True)
        stage.shift_workspace_by(-20.0, 12.0, world.transform.px_per_mm)
        after_origin = world._view_origin()
        after_ball = world._to_window(world._balls[0][:2])

        assert after_origin != before_origin
        assert after_ball == pytest.approx(before_ball)
        assert stage.workspace_shift_px == pytest.approx((-20.0, 12.0))
    finally:
        world.close()


def test_alg2_spot_does_not_break_tracking_after_repeated_moves():
    world, _ = build_sim_scenario(balls=[((1304.0, 936.0), 14.6)])
    try:
        world.alg2_mode = True
        detector = world.make_detector()
        pipeline = VisionPipeline(detector, tracker=ParticleTracker(max_jump_px=220))
        world.bind_pipeline(pipeline)
        stage = Alg2Stage(world, 1)
        stage.prepare_focus()
        uncertain = []
        for frame_id in range(1, 16):
            vis = pipeline.process(world.render(), frame_id=frame_id)
            uncertain.append(vis.uncertain)
            if frame_id < 12:
                stage.move_by(0.0005, 0.00087)
        assert not any(uncertain)
    finally:
        world.close()
