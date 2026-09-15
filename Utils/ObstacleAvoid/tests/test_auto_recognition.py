"""Offline Alg2 automatic-recognition tests without loading YOLO weights."""
import cv2
import numpy as np

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
