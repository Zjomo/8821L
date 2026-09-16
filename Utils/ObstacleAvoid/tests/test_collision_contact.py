"""需求验收：接触体积（防误触）+ 防卡壳软重规划。

覆盖：
  ① contact_penetrations：球心侵入圆球/多边形障碍时返回重叠深度，未侵入返回空；
  ② resolve_contact：球心侵入多个障碍时被推回表面（矫正后不再侵入）；
  ③ 其他圆球作为圆形障碍：规划路径会避让 peer 球，路径最小 clearance 不低于阈值；
  ④ ControllerConfig 防卡壳参数字段存在且默认值合理。
"""
import math

from obstacle_avoidance.models import (Obstacle, SubstrateRegion,
                                       WorkspaceSnapshot, CoordinateTransform)
from obstacle_avoidance.planner import (CollisionModel, GridPlanner, PlanConfig)


def _snap(obstacles=None, particles=()):
    sub = SubstrateRegion(polygon=[(40, 40), (600, 40), (600, 440), (40, 440)],
                          safety_margin_px=6.0)
    return WorkspaceSnapshot(frame_id=1, timestamp=0.0, substrate=sub,
                             obstacles=list(obstacles or []),
                             particles=list(particles),
                             transform=CoordinateTransform(px_per_mm=100.0),
                             frame_size=(640, 480))


# ---------------------------------------------------------------- ① 接触体积
def test_contact_penetration_reports_overlap_depth():
    ob_circle = Obstacle(kind="circle", center=(200, 200), radius=40)
    p = GridPlanner(PlanConfig(grid_res_px=2.0))
    # 球心侵入圆形障碍内部 15px（dist 25 - radius 40 = -15）
    hits = p.contact_penetrations(_snap([ob_circle]), (225, 200))
    assert len(hits) == 1
    _ob, pen = hits[0]
    assert abs(pen - 15.0) < 1e-6  # clearance=-15 -> penetration=15


def test_contact_penetration_empty_when_outside():
    ob = Obstacle(kind="circle", center=(200, 200), radius=40)
    p = GridPlanner(PlanConfig(grid_res_px=2.0))
    hits = p.contact_penetrations(_snap([ob]), (260, 200))   # 距表面 20，未侵入
    assert hits == []


def test_contact_penetration_reports_multiple_objects():
    a = Obstacle(kind="circle", center=(200, 200), radius=40)
    b = Obstacle(kind="circle", center=(260, 200), radius=40)
    p = GridPlanner(PlanConfig(grid_res_px=2.0))
    # 球心处于两球交叠区（圆心距 60，两边 40 半径交叠在 x∈(220,240)）
    hits = p.contact_penetrations(_snap([a, b]), (228, 200))
    assert len(hits) == 2, f"实际命中: {hits}"
    assert sum(pen for _, pen in hits) > 0


# ---------------------------------------------------------------- ② 防误触矫正
def test_resolve_contact_pushes_out_of_obstacles():
    a = Obstacle(kind="circle", center=(200, 200), radius=40)
    b = Obstacle(kind="circle", center=(300, 200), radius=40)
    p = GridPlanner(PlanConfig(grid_res_px=2.0))
    # 球心钻进两球交叠区（圆心距 100，两半 40 无交叠只单边侵入：x∈(160,240)
    # 时侵入 a；这里放 230 -> dist=30<40 侵入 a）
    fixed = p.resolve_contact(_snap([a, b]), (230, 200), radius_px=12.0)
    # 矫正后不再侵入 a（被推离）
    assert not a.blocks(fixed, 0.0)


def test_resolve_contact_unchanged_when_no_contact():
    p = GridPlanner(PlanConfig(grid_res_px=2.0))
    assert p.resolve_contact(_snap(), (300, 300), 12.0) == (300, 300)


# ---------------------------------------------------------------- ③ 其他球避让
def test_planner_avoids_peer_balls_as_circle_obstacles():
    # peer 球当作圆形障碍：路径须绕过，最小 clearance 不低于阈值
    peer = Obstacle(kind="circle", center=(300, 240), radius=45)
    p = GridPlanner(PlanConfig(grid_res_px=2.0))
    res = p.plan(_snap([peer]), (150, 240), (460, 240))
    assert res.success
    assert res.min_clearance_px >= CollisionModel().inflation_px - 0.5
    assert any(abs(y - 240) > 20 for _, y in res.waypoints_px)


# ---------------------------------------------------------------- ④ 防卡壳参数
def test_controller_config_has_stall_soft_replan_fields():
    from obstacle_avoidance.controller import ControllerConfig
    cfg = ControllerConfig()
    assert cfg.stall_frames >= 1
    assert cfg.stall_progress_px >= 0
    assert cfg.soft_replan_limit >= 1