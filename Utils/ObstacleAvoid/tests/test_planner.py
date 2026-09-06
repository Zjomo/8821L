"""规划层测试：直线、绕行、窄通道拒绝、NO_SAFE_PATH、越界拒绝。"""
import math

import pytest

from obstacle_avoidance.models import (FailureReason, Obstacle,
                                       SubstrateRegion, WorkspaceSnapshot,
                                       CoordinateTransform)
from obstacle_avoidance.planner import CollisionModel, GridPlanner, PlanConfig

INFL = CollisionModel().inflation_px  # 12+6+4 = 22


def _snap(obstacles=None):
    sub = SubstrateRegion(polygon=[(40, 40), (600, 40), (600, 440), (40, 440)],
                          safety_margin_px=6.0)
    return WorkspaceSnapshot(frame_id=1, timestamp=0.0, substrate=sub,
                             obstacles=obstacles or [], particles=[],
                             transform=CoordinateTransform(px_per_mm=100.0),
                             frame_size=(640, 480))


def test_straight_line_without_obstacles():
    p = GridPlanner(PlanConfig(grid_res_px=2.0)).plan(_snap(), (150, 240), (480, 240))
    assert p.success
    assert len(p.waypoints_px) == 2  # 直线简化为起终点
    assert abs(p.length_px - 330) < 3.0
    # 起终点距边界 110/120px，减 margin 6 -> ~104
    assert p.min_clearance_px >= 100


def test_detour_around_static_obstacle():
    ob = Obstacle(kind="circle", center=(315, 240), radius=45)
    p = GridPlanner().plan(_snap([ob]), (150, 240), (480, 240))
    assert p.success
    assert p.min_clearance_px >= INFL - 0.5
    # 绕行：路径中存在明显偏离 y=240 的 waypoint
    assert any(abs(y - 240) > 20 for _, y in p.waypoints_px)


def test_narrow_corridor_rejected_by_clearance():
    """通道几何存在但宽度 < 2*inflation 时应被膨胀模型封死。"""
    gap_a, gap_b = 231.0, 257.0            # 26px 通道 < 2*22=44px
    top = Obstacle(kind="polygon",
                   polygon=[(20, 20), (620, 20), (620, gap_a), (20, gap_a)])
    bottom = Obstacle(kind="polygon",
                      polygon=[(20, gap_b), (620, gap_b), (620, 460), (20, 460)])
    p = GridPlanner().plan(_snap([top, bottom]), (150, 244), (500, 244))
    assert not p.success
    assert p.failure_reason in (FailureReason.NO_SAFE_PATH,
                                FailureReason.LOW_CLEARANCE)


def test_no_safe_path_when_goal_sealed():
    obs = [Obstacle(kind="circle", center=(460, y), radius=40,
                    obstacle_id=f"w{i}")
           for i, y in enumerate(range(60, 421, 55))]
    p = GridPlanner().plan(_snap(obs), (150, 240), (540, 240))
    assert not p.success
    assert p.failure_reason == FailureReason.NO_SAFE_PATH


def test_goal_out_of_substrate_rejected():
    p = GridPlanner().plan(_snap(), (150, 240), (650, 240))
    assert not p.success
    assert p.failure_reason == FailureReason.OUT_OF_BOUNDS
    assert p.waypoints_px == []


def test_goal_too_close_to_boundary_low_clearance():
    p = GridPlanner().plan(_snap(), (150, 240), (592, 240))
    assert not p.success
    assert p.failure_reason == FailureReason.LOW_CLEARANCE


def test_plan_version_increments():
    planner = GridPlanner()
    r1 = planner.plan(_snap(), (100, 100), (200, 100))
    r2 = planner.plan(_snap(), (100, 100), (200, 100))
    assert r2.plan_version == r1.plan_version + 1
