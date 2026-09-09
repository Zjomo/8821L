import cv2

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
        frame = world.render()
        # OpenCV BGR green ring/center must be visible in the rendered frame.
        green = (frame[:, :, 1] > 180) & (frame[:, :, 0] < 80) & (frame[:, :, 2] < 80)
        assert int(green.sum()) > 20
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
