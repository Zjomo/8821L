"""视觉层测试：合成帧检测精度、衬底 IoU、Unicode 路径、不确定性。"""
import math
import os

import cv2
import numpy as np
import pytest

from obstacle_avoidance.simulator import SimWorld
from obstacle_avoidance.vision import (ClassicDetector, VisionPipeline,
                                       load_image_unicode)

R = 12.0


def _world(obstacles=None, particles=None):
    return SimWorld(
        substrate_polygon=[(40, 40), (600, 40), (600, 440), (40, 440)],
        obstacles=obstacles or [],
        particles=particles or [((150, 240), R)],
        frame_size=(640, 480))


def test_particle_center_error_le_2px():
    world = _world()
    vis = VisionPipeline(expected_radius_px=R)
    res = vis.process(world.render(), frame_id=1)
    assert not res.uncertain
    p = res.particles[0]
    truth = world.particle_position(1)
    assert math.hypot(p.position_px[0] - truth[0],
                      p.position_px[1] - truth[1]) <= 2.0


def test_substrate_iou_ge_095():
    world = _world()
    det = ClassicDetector()
    sub = det.detect_substrate(world.render())
    true_poly = np.array(world.substrate.polygon, dtype=np.int32)
    det_poly = np.array(sub.polygon, dtype=np.int32)
    m1 = np.zeros((480, 640), np.uint8)
    m2 = np.zeros((480, 640), np.uint8)
    cv2.fillPoly(m1, [true_poly], 1)
    cv2.fillPoly(m2, [det_poly], 1)
    iou = np.logical_and(m1, m2).sum() / np.logical_or(m1, m2).sum()
    assert iou >= 0.95


def test_obstacle_detection():
    from obstacle_avoidance.models import Obstacle
    obs = [Obstacle(kind="circle", center=(320, 240), radius=45)]
    world = _world(obstacles=obs)
    vis = VisionPipeline(expected_radius_px=R)
    res = vis.process(world.render(), frame_id=1)
    assert len(res.obstacles) == 1
    ob = res.obstacles[0]
    assert ob.kind == "circle"
    assert math.hypot(ob.center[0] - 320, ob.center[1] - 240) <= 3.0
    assert abs(ob.radius - 45) <= 3.0


def test_unicode_image_path_roundtrip(tmp_path):
    """中文路径：imencode->tofile 写入，np.fromfile+imdecode 读取。"""
    frame = _world().render()
    path = str(tmp_path / "中文路径_帧.png")
    ok, buf = cv2.imencode(".png", frame)
    assert ok
    buf.tofile(path)
    loaded = load_image_unicode(path)
    assert loaded.shape == frame.shape
    # 直接 cv2.imread 对 Unicode 路径会失败（审计 P1），此处确保我们的入口可用
    assert loaded.mean() > 0


def test_missing_particle_is_uncertain():
    world = _world()
    vis = VisionPipeline(expected_radius_px=R)
    world.set_occluded(1, True)
    res = vis.process(world.render(), frame_id=1)
    assert res.uncertain
    assert res.uncertain_reason == "no_particle_detected"


def test_merged_blob_flagged_uncertain():
    """AG-03 相关：两球粘连成一块 -> 显式 uncertain，不重复计数。"""
    # 渲染两个几乎重叠的球（间隔 < 半径）
    world = _world(particles=[((300, 240), R), ((312, 240), R)])
    vis = VisionPipeline(expected_radius_px=R)
    res = vis.process(world.render(), frame_id=1)
    # 间隔 12px：可能仍分开；此处只断言“要么分开两球、要么显式不确定”
    if res.uncertain:
        assert res.uncertain_reason == "merged_particle_blob"
    else:
        assert len(res.particles) == 2


def test_tracker_keeps_ids_on_overlap_then_separation():
    """AG-03: 重叠期间保持 track（粘连帧显式 uncertain），分开后仍两个独立轨迹。"""
    world = _world(particles=[((200, 240), R), ((260, 240), R)])
    vis = VisionPipeline(expected_radius_px=R)
    # frame1: 分离两球 -> tracks 1,2
    res0 = vis.process(world.render(), frame_id=1)
    assert len(res0.particles) == 2
    # frame2: 靠近至粘连（单步 <= max_jump 40px，重叠 ~12px）
    world.move_particle(1, 40, 0)    # 240
    world.move_particle(2, -8, 0)    # 252
    res1 = vis.process(world.render(), frame_id=2)
    assert res1.uncertain
    assert res1.uncertain_reason == "merged_particle_blob"
    ids1 = {p.track_id for p in res1.particles}   # coast 保 ID
    assert ids1 == {1, 2}
    # frame3: 分开 -> 仍匹配原 track，无新增
    world.move_particle(1, 35, 0)    # 275
    world.move_particle(2, -30, 0)   # 222
    res2 = vis.process(world.render(), frame_id=3)
    assert not res2.uncertain
    ids2 = {p.track_id for p in res2.particles}
    assert ids2 == {1, 2}
