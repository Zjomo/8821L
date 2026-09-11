"""模型/坐标契约（PLAN 第 1 节验收）测试。"""
import math

from obstacle_avoidance.models import (CoordinateTransform, GoalRegion,
                                       Obstacle, SubstrateRegion,
                                       WorkspaceSnapshot, point_in_polygon)


def test_coordinate_roundtrip_within_1px():
    t = CoordinateTransform(px_per_mm=137.5)  # 非整倍数更严格
    for p in [(0.0, 0.0), (123.456, 7.89), (639.5, 479.5)]:
        q = t.to_px(t.to_mm(p))
        assert math.hypot(q[0] - p[0], q[1] - p[1]) <= 1.0


def test_substrate_feasibility_and_margin():
    sub = SubstrateRegion(polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
                          safety_margin_px=10.0)
    assert sub.is_feasible((50, 50))
    assert not sub.is_feasible((5, 50))       # 边界内但 margin 不足
    assert not sub.is_feasible((110, 50))     # 越界
    assert not sub.is_feasible((-1, 50))


def test_obstacle_blocks_and_clearance():
    ob = Obstacle(kind="circle", center=(50, 50), radius=10)
    assert ob.blocks((55, 50), 4.0)
    assert not ob.blocks((70, 50), 4.0)
    assert ob.clearance((70, 50)) == 10.0
    poly = Obstacle(kind="polygon", polygon=[(0, 0), (10, 0), (10, 10), (0, 10)])
    assert poly.blocks((5, 5), 0.0)
    assert poly.blocks((12, 5), 3.0)
    assert not poly.blocks((15, 5), 3.0)


def test_snapshot_serialization_roundtrip():
    sub = SubstrateRegion(polygon=[(0, 0), (640, 0), (640, 480), (0, 480)])
    snap = WorkspaceSnapshot(
        frame_id=7, timestamp=123.4, substrate=sub,
        obstacles=[Obstacle(kind="circle", center=(100, 100), radius=20)],
        particles=[__import__("obstacle_avoidance.models", fromlist=["Particle"])
                   .Particle(track_id=1, position_px=(50, 60), radius_px=12,
                             frame_id=7)],
        transform=CoordinateTransform(px_per_mm=100.0))
    snap2 = WorkspaceSnapshot.from_dict(snap.to_dict())
    assert snap2.frame_id == 7
    assert snap2.obstacles[0].radius == 20
    assert snap2.particles[0].position_px == (50.0, 60.0)
    assert snap2.transform.px_per_mm == 100.0
    # 障碍移动后签名变化（动态重规划依据）
    sig1 = snap.obstacle_signature()
    snap.obstacles[0].center = (110, 100)
    assert snap.obstacle_signature() != sig1


def test_goal_region_contains():
    g = GoalRegion(center=(100, 100), radius_px=20)
    assert g.contains_center((110, 100), ball_radius_px=5)
    assert not g.contains_center((130, 100), ball_radius_px=5)
    assert point_in_polygon((50, 50), [(0, 0), (100, 0), (100, 100), (0, 100)])


def test_target_lock_requires_stable_identity():
    from obstacle_avoidance.models import (Particle, TargetLock, TargetLockState,
                                           TargetSelection)
    selected = Particle(2, (100, 100), 10, 0.95, 1)
    lock = TargetLock(TargetSelection.from_particle(selected),
                      stable_frames_required=2)
    assert lock.update([selected]).track_id == 2
    assert lock.state == TargetLockState.SELECTING
    current = Particle(2, (102, 101), 10, 0.95, 2)
    assert lock.update([current]).track_id == 2
    assert lock.state == TargetLockState.LOCKED


def test_target_lock_rejects_large_identity_jump():
    from obstacle_avoidance.models import (Particle, TargetLock, TargetLockState,
                                           TargetSelection)
    selected = Particle(2, (100, 100), 10, 0.95, 1)
    lock = TargetLock(TargetSelection.from_particle(selected), max_jump_px=20)
    assert lock.update([Particle(2, (140, 100), 10, 0.95, 2)]) is None
    assert lock.state == TargetLockState.AMBIGUOUS
