"""Offline Alg2 automatic-recognition tests without loading YOLO weights."""
import cv2
import numpy as np
import pytest

from obstacle_avoidance.auto_recognition import (Alg2AutoRecognizer,
                                                  AutoRecognitionConfig,
                                                  AutoRecognitionPipeline,
                                                  BeamSpotDetector,
                                                  suppress_ui_overlays)


class _FixedParticleDetector:
    min_confidence = 0.45

    def __init__(self, detections):
        self.detections = detections

    def detect_particles(self, _frame, expected_radius_px=None):
        return self.detections, []


def _microscope_frame():
    frame = np.full((240, 320, 3), (158, 156, 152), np.uint8)
    frame[55:180] = (170, 151, 145)
    # Add long outer edges and texture so registration has real signal.
    cv2.line(frame, (0, 55), (319, 55), (145, 137, 139), 2)
    cv2.line(frame, (0, 180), (319, 180), (145, 137, 139), 2)
    cv2.line(frame, (120, 80), (126, 100), (105, 90, 110), 3)
    return frame


def test_recognizer_detects_substrate_and_ball_membership():
    detections = [((80.0, 100.0), 12.0, 0.96),
                  ((230.0, 215.0), 12.0, 0.94)]
    recognizer = Alg2AutoRecognizer(
        particle_detector=_FixedParticleDetector(detections),
        config=AutoRecognitionConfig(substrate_min_area_ratio=0.05))
    result = recognizer.process(_microscope_frame(), frame_id=0)

    assert len(result.particles) == 2
    assert result.particles_inside_substrate == [True, False]
    assert result.substrate_confidence >= 0.45
    assert not result.uncertain


def test_substrate_refinement_preserves_irregular_outer_boundary():
    frame = np.full((240, 320, 3), (158, 156, 152), np.uint8)
    outline = np.asarray([
        (0, 55), (112, 55), (112, 82), (130, 82), (130, 55),
        (280, 55), (310, 72), (292, 110), (306, 124), (286, 180),
        (0, 180),
    ], np.int32)
    cv2.fillPoly(frame, [outline], (170, 151, 145))
    cv2.line(frame, (0, 180), (286, 180), (145, 137, 139), 2)

    recognizer = Alg2AutoRecognizer(
        particle_detector=_FixedParticleDetector([((80.0, 100.0), 12.0, 0.96)]),
        config=AutoRecognitionConfig(substrate_min_area_ratio=0.05))
    result = recognizer.process(frame, frame_id=0)

    assert result.substrate_mask[100, 80] == 255
    assert result.substrate_mask[65, 120] == 0  # Top notch.
    assert result.substrate_mask[95, 300] == 0  # Right-side damage.
    assert result.substrate_mask[210, 200] == 0
    assert len(result.substrate_polygon) > 4


def test_result_converts_to_existing_alg2_snapshot_contract():
    detector = _FixedParticleDetector([((80.0, 100.0), 12.0, 0.96)])
    recognizer = Alg2AutoRecognizer(particle_detector=detector)
    result = recognizer.process(_microscope_frame(), frame_id=7,
                                timestamp_s=1.25)
    snapshot = result.to_snapshot(px_per_mm=120.0)

    assert snapshot.frame_id == 7
    assert snapshot.timestamp == 1.25
    assert snapshot.particle(1).position_px == (80.0, 100.0)
    assert snapshot.substrate.contains((80.0, 100.0))
    assert snapshot.transform.px_per_mm == 120.0


def test_tracker_keeps_ball_identity_between_frames():
    detector = _FixedParticleDetector([((80.0, 100.0), 12.0, 0.96)])
    recognizer = Alg2AutoRecognizer(particle_detector=detector)
    first = recognizer.process(_microscope_frame(), frame_id=0)
    detector.detections = [((86.0, 103.0), 12.0, 0.95)]
    second = recognizer.process(_microscope_frame(), frame_id=1)

    assert first.particles[0].track_id == second.particles[0].track_id == 1


def test_manual_substrate_is_propagated_without_auto_refresh():
    detector = _FixedParticleDetector([((80.0, 100.0), 12.0, 0.96)])
    polygon = [(20.0, 40.0), (300.0, 40.0), (300.0, 200.0), (20.0, 200.0)]
    recognizer = Alg2AutoRecognizer(
        particle_detector=detector,
        manual_substrate_polygon=polygon,
        config=AutoRecognitionConfig(auto_substrate=False))
    result = recognizer.process(_microscope_frame(), frame_id=0)
    assert result.substrate_confidence == 1.0
    assert set(result.substrate_polygon) == set(polygon)
    assert "substrate_not_found" not in result.uncertain_reason


def test_ui_overlay_suppression_masks_annotation_colors():
    frame = np.zeros((80, 100, 3), np.uint8)
    cv2.rectangle(frame, (10, 10), (90, 70), (255, 0, 0), 2)
    cleaned, mask = suppress_ui_overlays(frame)
    assert mask.sum() > 0
    assert cleaned.shape == frame.shape


def test_beam_detector_excludes_green_ball():
    frame = np.full((160, 220, 3), 150, np.uint8)
    cv2.circle(frame, (60, 80), 15, (20, 230, 140), -1)
    cv2.circle(frame, (170, 80), 10, (230, 220, 20), -1)
    particle = __import__("obstacle_avoidance.models", fromlist=["Particle"]) \
        .Particle(1, (60.0, 80.0), 16.0, 0.95, 0)

    center, confidence, _mask = BeamSpotDetector().detect(frame, [particle])

    assert center is not None
    assert abs(center[0] - 170.0) <= 2.0
    assert abs(center[1] - 80.0) <= 2.0
    assert confidence > 0.4


_SUBSTRATE_TEXTURE = np.random.default_rng(3).integers(
    120, 190, size=(240, 320, 3), dtype=np.uint8)


def _textured_frame(shift_x: float, ball_x: float):
    """有衬底宽谱纹理的帧；球（亮盘）位置可独立于纹理平移设置。"""
    frame = np.roll(_SUBSTRATE_TEXTURE, int(shift_x), axis=1).copy()
    cv2.circle(frame, (int(ball_x), 120), 14, (255, 255, 255), -1)
    return frame


def _ball_mask(*centers):
    mask = np.zeros((240, 320), np.uint8)
    for x, y in centers:
        cv2.circle(mask, (int(x), int(y)), 20, 255, -1)
    return mask


def test_registration_ignores_moving_ball_and_tracks_substrate():
    """球相对样品运动时，配准必须给出样品（衬底）位移，而不是跟着球走。"""
    from obstacle_avoidance.auto_recognition import GlobalMotionEstimator

    estimator = GlobalMotionEstimator(analysis_width=320)
    estimator.update(_textured_frame(0.0, 80.0), _ball_mask((80.0, 120.0)))
    # 纹理左移 12px（样品移动），球反而右移 30px（光镊相对样品拖动）
    shift, confidence = estimator.update(_textured_frame(-12.0, 110.0),
                                         _ball_mask((110.0, 120.0)))
    assert confidence > 0.0
    assert shift[0] == pytest.approx(-12.0, abs=3.0)
    assert shift[1] == pytest.approx(0.0, abs=2.0)


def test_registration_stays_still_when_only_ball_moves():
    """样品不动、只有球动（光镊拖动）：衬底框不得跟着球跑。"""
    from obstacle_avoidance.auto_recognition import GlobalMotionEstimator

    faint = np.random.default_rng(11).integers(140, 161, size=(240, 320, 3),
                                              dtype=np.uint8)

    def frame(ball_x):
        img = faint.copy()
        cv2.circle(img, (int(ball_x), 120), 14, (255, 255, 255), -1)
        return img

    estimator = GlobalMotionEstimator(analysis_width=320)
    estimator.update(frame(80.0), _ball_mask((80.0, 120.0)))
    shift, confidence = estimator.update(frame(110.0), _ball_mask((110.0, 120.0)))
    assert confidence > 0.0
    assert abs(shift[0]) <= 5.0
    assert abs(shift[1]) <= 5.0


def test_manual_substrate_polygon_wins_over_stale_seed():
    """画框后换用手动多边形：框落在用户画的位置（按累计位移反算参考帧）。"""
    recognizer = Alg2AutoRecognizer(
        particle_detector=_FixedParticleDetector([((80.0, 100.0), 12.0, 0.96)]),
        config=AutoRecognitionConfig(auto_substrate=False))
    recognizer.process(_microscope_frame(), frame_id=0)
    recognizer.process(_textured_frame(-12.0, 80.0), frame_id=1)
    accumulated = recognizer.motion.cumulative
    assert accumulated[0] != 0.0

    drawn = [(20.0, 40.0), (200.0, 40.0), (200.0, 160.0), (20.0, 160.0)]
    recognizer.set_manual_substrate(drawn)
    result = recognizer.process(_textured_frame(-12.0, 80.0), frame_id=2)
    polygon = result.substrate_polygon
    assert len(polygon) == 4
    # 顶点可能有 1px 光栅化误差，按中心/范围核对：框必须落在用户画的位置
    assert (sum(p[0] for p in polygon) / 4,
            sum(p[1] for p in polygon) / 4) == pytest.approx((110.0, 100.0),
                                                            abs=1.5)
    assert result.substrate_confidence == pytest.approx(1.0)


def test_controller_pipeline_exposes_workspace_contract():
    detector = _FixedParticleDetector([((80.0, 100.0), 12.0, 0.96)])
    pipeline = AutoRecognitionPipeline(particle_detector=detector)

    result = pipeline.process(_microscope_frame(), frame_id=3)

    assert result.substrate is not None
    assert result.particles[0].track_id == 1
    assert pipeline.latest_result is not None
    assert pipeline.tracker.active_particles()[0].track_id == 1


def test_controller_pipeline_marks_missing_ball_uncertain():
    pipeline = AutoRecognitionPipeline(
        particle_detector=_FixedParticleDetector([]))

    result = pipeline.process(_microscope_frame(), frame_id=0)

    assert result.uncertain
    assert "no_fresh_particle_detection" in result.uncertain_reason
