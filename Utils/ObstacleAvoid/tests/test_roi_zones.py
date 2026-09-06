"""ROI/区域划分（roi_zones）+ video03 场景 + UI 实时检测测试。"""
import os

import pytest

from obstacle_avoidance.roi_zones import RoiConfig, Zone
from obstacle_avoidance import video_sim

VIDEO_SIZE = (1112, 888)


# ---------------- 纯逻辑（无需权重）
def test_roi_config_roundtrip(tmp_path):
    cfg = RoiConfig(roi=(60, 300, 480, 420), video="x.mp4")
    cfg.zones = [Zone(name="goal_1", kind="goal", rect=(320, 620, 100, 80)),
                 Zone(name="free_1", kind="free", rect=(200, 400, 80, 60))]
    p = str(tmp_path / "roi.json")
    cfg.save(p)
    back = RoiConfig.load(p)
    assert back.roi == cfg.roi
    assert [(z.name, z.kind, z.rect) for z in back.zones] == \
        [(z.name, z.kind, z.rect) for z in cfg.zones]
    assert back.goal_zone().center() == (370.0, 660.0)


def test_roi_config_validation():
    cfg = RoiConfig(roi=(0, 0, 480, 420))
    with pytest.raises(ValueError):          # 区域超出 ROI
        cfg.zones = [Zone(name="g", kind="goal", rect=(400, 350, 200, 100))]
        cfg.validate()
    with pytest.raises(ValueError):          # 目标区多于一个
        cfg.zones = [Zone(name="g1", kind="goal", rect=(10, 10, 50, 50)),
                     Zone(name="g2", kind="goal", rect=(100, 100, 50, 50))]
        cfg.validate()
    with pytest.raises(ValueError):          # 类型非法
        RoiConfig(roi=(0, 0, 10, 10)).zones.append(
            Zone(name="b", kind="bad", rect=(0, 0, 1, 1)))
        cfg.validate()
    with pytest.raises(ValueError):          # ROI 超出视频
        RoiConfig(roi=(0, 0, 9999, 888)).validate(VIDEO_SIZE)


def test_planner_edge_clearance_configurable():
    """边界间隙独立可调：edge=39 拒绝贴边点，edge=1 接受；障碍膨胀不受影响。"""
    from obstacle_avoidance.models import (CoordinateTransform, FailureReason,
                                           Obstacle, SubstrateRegion,
                                           WorkspaceSnapshot)
    from obstacle_avoidance.planner import GridPlanner, PlanConfig

    sub = SubstrateRegion(
        polygon=[(10, 10), (410, 10), (410, 310), (10, 310)],
        safety_margin_px=4.0)
    snap = WorkspaceSnapshot(
        frame_id=0, timestamp=0.0, substrate=sub,
        obstacles=[Obstacle(kind="circle", center=(200, 160), radius=30)],
        particles=[], transform=CoordinateTransform(px_per_mm=100.0),
        frame_size=(420, 320))
    near_edge = (404, 160)   # 距右边界 6px、距障碍 170px

    strict = GridPlanner(PlanConfig())  # 默认：全膨胀 35+margin
    assert strict.check_point(snap, near_edge) == FailureReason.LOW_CLEARANCE
    loose = GridPlanner(PlanConfig(edge_clearance_px=1.0))
    assert loose.check_point(snap, near_edge) is None
    # 障碍膨胀不受 edge 参数影响：贴障碍点两种配置都拒绝
    near_obs = (200, 130)    # 距障碍边界 0px
    assert loose.check_point(snap, near_obs) == FailureReason.LOW_CLEARANCE
    # 占据栅格：边界带随 edge 收窄（障碍膨胀一致）
    b_strict, _ = strict.build_occupancy(snap)
    b_loose, _ = loose.build_occupancy(snap)
    assert b_loose.sum() < b_strict.sum()


# ---------------- video03 闭环（需权重+视频）
_needs_yolo = pytest.mark.skipif(
    not (os.path.isfile(video_sim.WEIGHTS) and os.path.isfile(video_sim.VIDEO)),
    reason="需要 weight/Duan_best.pt 与 Dataset/测试视频.mp4")


@_needs_yolo
def test_video03_closed_loop_with_goal_zone():
    """目标区中心作为终点：无障碍区配置下闭环 COMPLETE。"""
    cfg = RoiConfig(roi=(60, 300, 520, 420), video=video_sim.VIDEO)
    cfg.zones = [Zone(name="goal_1", kind="goal", rect=(320, 620, 100, 80))]
    cfg.validate(VIDEO_SIZE)
    world, run = video_sim.build_video_scenario("video03", config=cfg)
    assert world.window == (520, 420) and world.offset == [60.0, 300.0]
    assert world.static_obstacles == []
    result = run()
    assert result.final_state.value == "COMPLETE"
    assert result.final_error_px <= 8.0


@_needs_yolo
def test_video03_obstacle_zone_blocks_path():
    """障碍区多边形进入碰撞模型：规划必须绕行或拒绝，不得穿越。"""
    cfg = RoiConfig(roi=(60, 300, 520, 420), video=video_sim.VIDEO)
    # 障碍区横跨视野中部（起点->终点直线必经）
    cfg.zones = [Zone(name="goal_1", kind="goal", rect=(320, 620, 100, 80)),
                 Zone(name="obs_1", kind="obstacle", rect=(280, 440, 120, 140))]
    cfg.validate(VIDEO_SIZE)
    world, run = video_sim.build_video_scenario("video03", config=cfg)
    assert len(world.static_obstacles) == 1
    ob = world.static_obstacles[0]
    poly = [(x - 60, y - 300) for x, y in ob.polygon]   # 窗口坐标
    result = run()
    assert result.final_state.value in ("COMPLETE", "ABORTED")
    if result.plan and result.plan.success:             # 有路则不得穿越
        from obstacle_avoidance.models import point_in_polygon
        infl = 25.0 + 6.0 + 4.0
        for wp in result.plan.waypoints_px:
            if point_in_polygon(wp, poly):
                pytest.fail(f"路径穿越障碍区: {wp}")


@_needs_yolo
def test_video03_goal_low_clearance_nudged():
    """目标区贴边导致 clearance 不足 -> 自动微调到最近可行点并 COMPLETE。"""
    from obstacle_avoidance.reporter import RunReporter, replay
    cfg = RoiConfig(roi=(60, 300, 520, 420), video=video_sim.VIDEO)
    cfg.zones = [Zone(name="goal_1", kind="goal", rect=(320, 650, 100, 68))]
    cfg.validate(VIDEO_SIZE)     # 中心距下边界 26px < 所需 39px
    world, run = video_sim.build_video_scenario("video03", config=cfg)
    rep = RunReporter(None)
    result = run(rep=rep)
    adj = [e for e in rep.events if e.get("event") == "goal_adjusted"]
    rep.close()
    assert result.final_state.value == "COMPLETE"
    assert adj and adj[0]["from_px"] != adj[0]["to_px"]


# ---------------- UI 实时检测冒烟（需 PyQt5+视频）
def test_ui_live_roi_zones(tmp_path):
    qt = pytest.importorskip("PyQt5.QtWidgets")
    from obstacle_avoidance.qt_compat import QtCore
    from obstacle_avoidance.app import MainWindow

    app = qt.QApplication.instance() or qt.QApplication([])
    win = MainWindow()
    # 隔离用户配置（UI 仅保留 sim01；ROI 布局自动保存）
    win.SIM_ROI_CONFIG = str(tmp_path / "sim_roi_config.json")
    win._sim_cfg = {"balls": [], "grounds": [], "obstacles": []}
    win._sim_order = []
    win._ensure_live()   # sim01 -> SimMicroscopeWorld 仿真镜头
    win.on_live_toggled(True)
    for _ in range(3):
        win._live_tick()
    assert win.canvas.pixmap() is not None
    assert win._view_map is not None

    # sim01 ROI 布局画框：先画 ground（包含校验要求），再画 ball/obstacle/goal
    win.mode_combo.setCurrentText("衬底(ground)")
    win.on_rect_drawn(40, 40, 500, 400)
    assert len(win._sim_cfg["grounds"]) == 1
    win.mode_combo.setCurrentText("圆球(mask)")
    win.on_rect_drawn(100, 200, 30, 30)   # 中心(115,215) 在 ground(40,40,500,400) 内
    assert win._sim_cfg["balls"] == [[100, 200, 30, 30]]
    win.mode_combo.setCurrentText("障碍物(obstacle)")
    win.on_rect_drawn(300, 150, 60, 60)
    assert len(win._sim_cfg["obstacles"]) == 1
    win.mode_combo.setCurrentText("目标点(避障)")
    win.on_rect_drawn(500, 300, 10, 10)   # 中心(505,305) 在 ground 内
    assert win._sim_cfg["grounds"][0]["goal"] == [505, 305]

    # 越界球/障碍必须被拒绝
    win.mode_combo.setCurrentText("圆球(mask)")
    win.on_rect_drawn(600, 500, 30, 30)   # 中心(615,515) 在 ground 外
    assert len(win._sim_cfg["balls"]) == 1   # 仍只有之前那一个

    # 自动保存
    assert os.path.isfile(win.SIM_ROI_CONFIG)

    # 撤销（后画先撤：目标点 -> 障碍 -> 球 -> 衬底）
    win.on_undo_zone()
    assert win._sim_cfg["grounds"][0]["goal"] is None
    win.on_undo_zone()
    assert win._sim_cfg["obstacles"] == []
    win.on_undo_zone()
    assert win._sim_cfg["balls"] == []

    # ROI 布局注入场景构建：球/衬底映射到样本坐标/可行域
    from obstacle_avoidance.cli import build_scenario
    win._sim_cfg = {
        "grounds": [{"rect": [40, 40, 500, 400], "goal": None, "goal_range": None}],
        "balls": [[100, 200, 30, 30]],
        "obstacles": []}
    layout = {
        "balls": [list(b) for b in win._sim_cfg["balls"]],
        "obstacles": [list(o) for o in win._sim_cfg["obstacles"]],
        "grounds": [list(g["rect"]) for g in win._sim_cfg["grounds"]],
        "ground_goals": [g.get("goal") for g in win._sim_cfg["grounds"]],
        "ground_goal_ranges": [g.get("goal_range") for g in win._sim_cfg["grounds"]],
    }
    world, run = build_scenario("sim01", sim_layout=layout)
    bx, by, br = world._balls[0]
    assert (bx, by) == (100 + 15 + 1100, 200 + 15 + 700)  # 窗口->样本
    assert br == 15.0
    poly = world.substrate.polygon
    assert poly[0] == (40, 40) and poly[2] == (540, 440)  # ground 包围盒
    world.close()
    win.on_live_toggled(False)
    QtCore.QThread.msleep(10)
    win.close()
