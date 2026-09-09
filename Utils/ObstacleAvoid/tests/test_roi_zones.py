"""ROI/区域划分（roi_zones）+ video03 场景 + UI 实时检测测试。"""
import json
import os

import pytest

from obstacle_avoidance.roi_zones import (LayoutValidationError, RoiConfig, Zone,
                                          validate_sim_layout)
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
    """边界间隙独立可调：edge=39 拒绝贴边点，edge=1 接受（但有球半径
    碰撞体积下限）；障碍膨胀不受影响。"""
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
    loose_edge = (393, 160)  # 距右边界 17px（> 球半径 12 的下限）

    strict = GridPlanner(PlanConfig())  # 默认：全膨胀 35+margin
    assert strict.check_point(snap, near_edge) == FailureReason.LOW_CLEARANCE
    loose = GridPlanner(PlanConfig(edge_clearance_px=1.0))
    # 碰撞体积下限：球心距边界 6px < 球半径 12px，球体会越过边界 -> 拒绝
    assert loose.check_point(snap, near_edge) == FailureReason.LOW_CLEARANCE
    # 高于球半径下限的贴边点被接受
    assert loose.check_point(snap, loose_edge) is None
    # 障碍膨胀不受 edge 参数影响：贴障碍点两种配置都拒绝
    near_obs = (200, 130)    # 距障碍边界 0px
    assert loose.check_point(snap, near_obs) == FailureReason.LOW_CLEARANCE
    # 占据栅格：边界带随 edge 收窄（障碍膨胀一致）
    b_strict, _ = strict.build_occupancy(snap)
    b_loose, _ = loose.build_occupancy(snap)
    assert b_loose.sum() < b_strict.sum()


def test_sim_layout_requires_complete_single_ground_ownership():
    base = {
        "grounds": [[0, 0, 100, 100]],
        "balls": [[10, 10, 20, 20]],
        "obstacles": [[40, 40, 20, 20]],
        "ground_goals": [[80, 80]],
        "ground_goal_ranges": [None],
    }
    assert validate_sim_layout(base, mode="oa") == [[0]]

    partly_out = {**base, "balls": [[90, 90, 20, 20]]}
    with pytest.raises(LayoutValidationError, match="fully inside"):
        validate_sim_layout(partly_out, mode="oa")

    overlap = {**base, "grounds": [[0, 0, 80, 100], [20, 0, 80, 100]]}
    with pytest.raises(LayoutValidationError, match="exactly one substrate"):
        validate_sim_layout(overlap, mode="oa")


def test_sim_layout_task_target_contracts():
    common = {
        "grounds": [[0, 0, 100, 100]],
        "balls": [[10, 10, 20, 20]],
        "obstacles": [],
    }
    with pytest.raises(LayoutValidationError, match="target point"):
        validate_sim_layout({**common, "ground_goals": [None]}, mode="oa")
    with pytest.raises(LayoutValidationError, match="target range"):
        validate_sim_layout({**common, "ground_goal_ranges": [None]}, mode="ag")
    with pytest.raises(LayoutValidationError, match="only one target point"):
        validate_sim_layout({**common, "ground_goals": [[50, 50]],
                             "ground_goal_ranges": [[20, 20, 20, 20]]}, mode="oa")
    with pytest.raises(LayoutValidationError, match="only one target range"):
        validate_sim_layout({**common, "ground_goals": [[50, 50]],
                             "ground_goal_ranges": [[20, 20, 20, 20]]}, mode="ag")


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
    # 复位台位到样本中心（共享 sim_microscope.db 可能残留其它用例的拖动位置，
    # recenter=False 会恢复 -> 视窗原点漂移导致样本坐标断言失败）
    win._sim_live.motion_stage.move_to({"x": 750.0, "y": 500.0, "z": 0.0})
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
    # _sim_cfg 存样本绝对坐标：窗口(100,200) + 初始原点(1100,700)
    assert win._sim_cfg["balls"] == [[1200, 900, 30, 30]]
    win.mode_combo.setCurrentText("障碍物(obstacle)")
    win.on_rect_drawn(300, 150, 60, 60)
    assert len(win._sim_cfg["obstacles"]) == 1
    win.mode_combo.setCurrentText("目标点(避障)")
    win.on_rect_drawn(500, 300, 10, 10)   # 中心(505,305) 在 ground 内
    assert win._sim_cfg["grounds"][0]["goal"] == [505 + 1100, 305 + 700]

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


def test_ui_draw_gate_defaults_locked(tmp_path):
    """画框闸门：默认锁定防误触，开启后拖拽画框才生效。"""
    qt = pytest.importorskip("PyQt5.QtWidgets")
    from obstacle_avoidance.qt_compat import QtCore
    from obstacle_avoidance.app import MainWindow

    app = qt.QApplication.instance() or qt.QApplication([])
    win = MainWindow()
    win.SIM_ROI_CONFIG = str(tmp_path / "sim_roi_config.json")
    win._sim_cfg = {"balls": [], "grounds": [], "obstacles": []}
    win._sim_order = []
    # 默认锁定：按钮未勾选、画布闸门关闭
    assert win.draw_gate_btn.isChecked() is False
    assert win.canvas.drawing_enabled is False
    assert win.draw_gate_btn.text() == "画框:关"
    # 开启 -> 画布可画；关闭 -> 重新锁定
    win.draw_gate_btn.setChecked(True)
    assert win.canvas.drawing_enabled is True
    assert win.draw_gate_btn.text() == "画框:开"
    win.draw_gate_btn.setChecked(False)
    assert win.canvas.drawing_enabled is False
    # 程序化调用（on_rect_drawn 直调）不受闸门影响（保持兼容）
    win._ensure_live()
    win.on_live_toggled(True)
    for _ in range(3):
        win._live_tick()
    win.mode_combo.setCurrentText("衬底(ground)")
    win.on_rect_drawn(40, 40, 500, 400)
    assert len(win._sim_cfg["grounds"]) == 1
    # 锁定视角：勾选后禁用鼠标拖拽移动台
    assert win.view_lock_chk.isChecked() is False
    pos0 = (win._sim_live.micro_stage.position["x"],
            win._sim_live.micro_stage.position["y"])
    win.view_lock_chk.setChecked(True)
    win._on_sim_drag_start(100, 100)
    win._on_sim_drag_move(50, 30)          # 锁定 -> 台位不动
    assert (win._sim_live.micro_stage.position["x"],
            win._sim_live.micro_stage.position["y"]) == pos0
    win.view_lock_chk.setChecked(False)
    win._on_sim_drag_start(100, 100)
    win._on_sim_drag_move(50, 30)          # 解锁 -> 恢复拖拽
    assert (win._sim_live.micro_stage.position["x"],
            win._sim_live.micro_stage.position["y"]) != pos0
    win._on_sim_drag_end()
    win.on_live_toggled(False)
    QtCore.QThread.msleep(10)
    # 框定形状：圆/正方形取拖拽框中心+短边；长方形原样
    win.shape_combo.setCurrentText("圆形")
    assert win._apply_draw_shape(100, 50, 120, 60) == (130, 50, 60, 60, "circle")
    win.shape_combo.setCurrentText("正方形")
    assert win._apply_draw_shape(100, 50, 120, 60) == (130, 50, 60, 60, "rect")
    win.shape_combo.setCurrentText("长方形")
    assert win._apply_draw_shape(100, 50, 120, 60) == (100, 50, 120, 60, "rect")
    win.close()


def test_zone_shape_persistence(tmp_path):
    """Zone.shape 序列化往返 + 旧版配置（无 shape 字段）兼容。"""
    cfg = RoiConfig(roi=(50, 50, 300, 300))
    cfg.zones = [Zone(name="obs_1", kind="obstacle", rect=(100, 100, 80, 80),
                      shape="circle")]
    cfg.validate()
    p = cfg.save(str(tmp_path / "roi.json"))
    cfg2 = RoiConfig.load(p)
    assert cfg2.zones[0].shape == "circle"
    assert cfg2.zones[0].radius() == 40.0
    d = json.load(open(p, encoding="utf-8"))
    del d["zones"][0]["shape"]
    json.dump(d, open(p, "w", encoding="utf-8"))
    cfg3 = RoiConfig.load(p)
    assert cfg3.zones[0].shape == "rect"


def test_extract_trajectory_oa():
    """轨迹提取：task_config/plan/detection -> 目标/路径/逐帧粒子/轨迹链。"""
    from obstacle_avoidance.reporter import extract_trajectory
    evs = [
        {"event": "run_start", "run_id": "r1"},
        {"event": "task_config", "task_id": "oa", "track_id": 3,
         "goal": {"center": [200, 100]}},
        {"event": "plan", "task_id": "oa", "success": True,
         "waypoints_px": [[100, 100], [150, 100], [200, 100]]},
        {"event": "detection", "task_id": "oa", "uncertain": True,
         "particles": []},
        {"event": "detection", "task_id": "oa", "uncertain": False,
         "particles": [{"track_id": 3, "position_px": [100, 100],
                        "radius_px": 25}]},
        {"event": "detection", "task_id": "oa", "uncertain": False,
         "particles": [{"track_id": 3, "position_px": [130, 100],
                        "radius_px": 25},
                       {"track_id": 4, "position_px": [400, 300],
                        "radius_px": 20}]},
        {"event": "run_end", "final_state": "COMPLETE"},
    ]
    t = extract_trajectory(evs)
    assert t["run_id"] == "r1" and t["final_state"] == "COMPLETE"
    assert t["tasks"]["oa"]["goal"] == [200, 100]
    assert t["tasks"]["oa"]["track_id"] == 3
    assert t["tasks"]["oa"]["waypoints"] == [[100, 100], [150, 100], [200, 100]]
    assert len(t["frames"]) == 2            # uncertain 帧被剔除
    assert t["trails"]["oa"] == [[100.0, 100.0], [130.0, 100.0]]


def test_extract_trajectory_nn_chain():
    """无 track_id 时按最近邻链匹配（同帧多球不串迹）。"""
    from obstacle_avoidance.reporter import extract_trajectory
    evs = [
        {"event": "task_config", "task_id": "ag", "goal": {"center": [10, 10]}},
        {"event": "detection", "task_id": "ag", "uncertain": False,
         "particles": [{"track_id": 1, "position_px": [5, 5], "radius_px": 8},
                       {"track_id": 2, "position_px": [50, 50],
                        "radius_px": 8}]},
        {"event": "detection", "task_id": "ag", "uncertain": False,
         "particles": [{"track_id": 1, "position_px": [60, 60],
                        "radius_px": 8},
                       {"track_id": 2, "position_px": [52, 52],
                        "radius_px": 8}]},
    ]
    t = extract_trajectory(evs)
    assert t["trails"]["ag"] == [[5.0, 5.0], [52.0, 52.0]]


def test_ui_replay_last_round(tmp_path):
    """回放上一轮：加载 JSONL -> 摘要+动画数据，单步/播放联动。"""
    qt = pytest.importorskip("PyQt5.QtWidgets")
    from obstacle_avoidance.app import MainWindow

    app = qt.QApplication.instance() or qt.QApplication([])
    win = MainWindow()
    rep = str(tmp_path / "run.jsonl")
    with open(rep, "w", encoding="utf-8") as f:
        f.write(json.dumps({"seq": 1, "run_id": "r", "ts": "t",
                            "event": "run_start"}) + "\n")
        f.write(json.dumps({
            "seq": 2, "run_id": "r", "ts": "t", "event": "task_config",
            "task_id": "oa", "track_id": 3,
            "goal": {"center": [200, 100]}}) + "\n")
        for i, pos in enumerate([(100, 100), (130, 100), (160, 100),
                                 (198, 100)]):
            f.write(json.dumps({
                "seq": 3 + i, "run_id": "r", "ts": "t",
                "event": "detection", "task_id": "oa", "uncertain": False,
                "particles": [{"track_id": 3, "position_px": list(pos),
                               "radius_px": 25}]}) + "\n")
        f.write(json.dumps({"seq": 9, "run_id": "r", "ts": "t",
                            "event": "run_end",
                            "final_state": "COMPLETE"}) + "\n")
    win.report_path.setText(rep)
    win.on_replay()
    assert win.rp_play_btn.isEnabled()
    assert win.rp_slider.maximum() == 3
    assert len(win._rp_traj["frames"]) == 4
    assert win._rp_traj["trails"]["oa"] == [
        [100.0, 100.0], [130.0, 100.0], [160.0, 100.0], [198.0, 100.0]]
    # 单步推进
    win.rp_next_btn.click()
    assert win.rp_slider.value() == 1
    assert win.rp_label.text() == "检测帧 2/4"
    # 播放 -> tick 推进 -> 末尾自动停止
    win.on_replay_play()
    assert win.rp_play_btn.text() == "暂停"
    win._rp_tick()
    assert win.rp_slider.value() == 2
    win._rp_tick()
    win._rp_tick()
    assert win.rp_play_btn.text() == "播放"
    assert win.rp_slider.value() == 3
    win.close()


def test_zone_free_polygon_validation_and_contains():
    """Free 多边形：顶点校验 + 射线法 contains。"""
    pts = [(100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0)]
    z = Zone(name="obs_1", kind="obstacle", rect=(100, 100, 100, 100),
             shape="free", points=pts)
    RoiConfig(roi=(50, 50, 300, 300), zones=[z]).validate()
    assert z.contains((150, 150))         # 多边形内部
    assert not z.contains((50, 50))       # 多边形外部（外接矩形之外）
    # 缺顶点 / 非 free 带顶点 -> 非法
    with pytest.raises(ValueError):
        RoiConfig(roi=(50, 50, 300, 300), zones=[
            Zone(name="bad", kind="obstacle", rect=(100, 100, 50, 50),
                 shape="free")]).validate()
    with pytest.raises(ValueError):
        RoiConfig(roi=(50, 50, 300, 300), zones=[
            Zone(name="bad2", kind="obstacle", rect=(100, 100, 50, 50),
                 shape="rect", points=pts)]).validate()


def test_zone_free_polygon_persistence(tmp_path):
    """Free 多边形随配置 JSON 往返保存。"""
    pts = [(100.0, 100.0), (180.0, 120.0), (200.0, 200.0), (110.0, 180.0)]
    cfg = RoiConfig(roi=(50, 50, 300, 300))
    cfg.zones = [Zone(name="obs_1", kind="obstacle",
                      rect=(100, 100, 100, 100), shape="free", points=pts)]
    cfg.validate()
    p = cfg.save(str(tmp_path / "roi.json"))
    cfg2 = RoiConfig.load(p)
    assert cfg2.zones[0].shape == "free"
    assert cfg2.zones[0].points == pts


def test_ui_free_polygon_mode_and_draw(tmp_path):
    """Free 模式：画布进入多边形模式；顶点闭合按模式落库（虚拟=外接矩形）。"""
    qt = pytest.importorskip("PyQt5.QtWidgets")
    from obstacle_avoidance.app import MainWindow

    app = qt.QApplication.instance() or qt.QApplication([])
    win = MainWindow()
    # Free 形状 -> 画布多边形模式；其他形状退出
    win.shape_combo.setCurrentText("Free")
    assert win.canvas.polygon_mode
    win.shape_combo.setCurrentText("长方形")
    assert not win.canvas.polygon_mode
    # 闸门开启后 Free 提示
    win.shape_combo.setCurrentText("Free")
    win.draw_gate_btn.setChecked(True)
    assert win.canvas.polygon_mode and win.canvas.drawing_enabled
    # 虚拟模式：多边形闭合 -> 外接矩形落入 sim 配置（障碍须在衬底内）
    win.mode_combo.setCurrentText("障碍物(obstacle)")
    assert win._sim_cfg["grounds"], "需已有衬底（sim 配置载入）"
    gx, gy, gw, gh = win._rect_s2w(win._sim_cfg["grounds"][0]["rect"],
                                   win._sim_initial_origin())
    x0, y0 = gx + 5, gy + 5
    n0 = len(win._sim_cfg["obstacles"])
    win.on_points_drawn([(x0, y0), (x0 + 40, y0), (x0 + 40, y0 + 30),
                         (x0, y0 + 30)])
    assert len(win._sim_cfg["obstacles"]) == n0 + 1
    # 外接矩形尺寸（存储为样本坐标，位置含视窗原点偏移）
    assert list(win._sim_cfg["obstacles"][-1][2:]) == [40, 30]
    # <3 顶点忽略
    win.on_points_drawn([(10, 10), (20, 20)])
    assert len(win._sim_cfg["obstacles"]) == n0 + 1
    win.close()


def test_ui_free_polygon_obstacle_and_undo(tmp_path):
    """Free 障碍多边形：平行列表落库 + 撤销同步弹出 + 持久化往返。"""
    qt = pytest.importorskip("PyQt5.QtWidgets")
    from obstacle_avoidance.app import MainWindow

    app = qt.QApplication.instance() or qt.QApplication([])
    win = MainWindow()
    win.SIM_ROI_CONFIG = str(tmp_path / "sim_roi_config.json")
    win._sim_cfg = {
        "grounds": [], "balls": [], "obstacles": [], "obstacle_polys": []}
    win._sim_order = []
    win._ensure_live()
    win.mode_combo.setCurrentText("衬底(ground)")
    win.on_rect_drawn(40, 40, 500, 400)
    win.mode_combo.setCurrentText("障碍物(obstacle)")
    # 五边形障碍（边数=顶点数）
    pts = [(100.0, 150.0), (160.0, 130.0), (190.0, 180.0),
           (150.0, 220.0), (105.0, 200.0)]
    origin = win._sim_origin()
    win.on_points_drawn(pts)
    assert len(win._sim_cfg["obstacles"]) == 1
    poly_s = win._sim_cfg["obstacle_polys"][0]
    assert len(poly_s) == 5                       # 边数按点位确定
    assert list(poly_s[0]) == [100 + origin[0], 150 + origin[1]]  # 窗口->样本
    # 撤销：障碍与多边形同步弹出
    win.on_undo_zone()
    assert win._sim_cfg["obstacles"] == []
    assert win._sim_cfg["obstacle_polys"] == []
    # 重新画一个四边形 + 一个矩形障碍（混合），保存/载入往返
    win.on_points_drawn(pts[:4])
    win.on_rect_drawn(300, 200, 50, 40)           # 矩形障碍 -> poly=None
    assert win._sim_cfg["obstacle_polys"][1] is None
    win._save_sim_config()
    win2 = MainWindow()
    win2.SIM_ROI_CONFIG = win.SIM_ROI_CONFIG
    win2._load_sim_config()
    assert len(win2._sim_cfg["obstacle_polys"]) == 2
    assert [list(p) for p in win2._sim_cfg["obstacle_polys"][0]] == \
        [list(p) for p in win._sim_cfg["obstacle_polys"][0]]  # 多边形往返不丢
    assert win2._sim_cfg["obstacle_polys"][1] is None
    win.on_live_toggled(False)
    win.close()
    win2.close()


def test_oa_multi_ball_single_goal_ring_placement():
    """OA 多球共用一个目标点：目标点=组装中心，环形驻点逐球移动不重叠。"""
    import math

    from obstacle_avoidance import sim_microscope as _sm
    _sm._ensure_paths()               # 挂 simulator_app 到 sys.path
    pytest.importorskip("simulator_app")
    qt = pytest.importorskip("PyQt5.QtWidgets")
    from obstacle_avoidance.app import WorkerThread
    from obstacle_avoidance.cli import build_scenario
    from obstacle_avoidance.models import RunState
    from obstacle_avoidance.reporter import RunReporter

    qt.QApplication.instance() or qt.QApplication([])
    layout = {
        "grounds": [[40, 40, 500, 400]],
        "balls": [[100, 200, 30, 30], [180, 220, 30, 30]],
        "obstacles": [[300, 150, 60, 60]],
        "ground_goals": [[400, 340]],
        "ground_goal_ranges": [None],
    }
    world, _ = build_scenario("sim01", sim_layout=layout)
    try:
        w = WorkerThread("sim01", {})
        rep = RunReporter(None)
        final_balls = [list(b) for b in layout["balls"]]

        def set_ball_at(bi, cx, cy):
            _, _, bw, bh = final_balls[bi]
            final_balls[bi] = [int(cx - bw / 2), int(cy - bh / 2), bw, bh]

        state, failed = w._run_sim_assembly(
            world, rep, 0, [0, 1], layout["balls"], (400.0, 340.0),
            40.0 * math.sqrt(2), lambda: None, final_balls, set_ball_at)
        rep.close()
        assert state == RunState.COMPLETE and not failed
        c0 = (final_balls[0][0] + 15, final_balls[0][1] + 15)
        c1 = (final_balls[1][0] + 15, final_balls[1][1] + 15)
        # 两球驻点都在目标中心附近（环上），且互不重叠（不叠在同一点）
        assert math.dist(c0, (400, 340)) <= 70
        assert math.dist(c1, (400, 340)) <= 70
        assert math.dist(c0, c1) >= 2 * 12 * 1.2 - 1
    finally:
        world.close()


def test_build_sim_scenario_free_polygons():
    """布局多边形注入仿真世界：衬底=Free 顶点，障碍=多边形（窗口->样本）。"""
    from obstacle_avoidance.sim_microscope import build_sim_scenario

    ground_poly = [[40.0, 40.0], [540.0, 60.0], [520.0, 440.0], [60.0, 420.0]]
    obs_poly = [[200.0, 150.0], [280.0, 130.0], [300.0, 200.0],
                [240.0, 240.0], [190.0, 210.0]]
    layout = {
        "balls": [[100, 300, 30, 30]],
        "obstacles": [[400, 250, 40, 40], [500, 300, 30, 30]],
        "grounds": [[40, 40, 500, 400]],
        "ground_polys": [ground_poly],            # 单衬底 Free -> 真实可行域
        "obstacle_polys": [obs_poly, None],       # 平行列表；None=矩形
        "ground_goals": [[300, 380]],
        "ground_goal_ranges": [None],
    }
    world, run = build_sim_scenario(layout=layout)
    try:
        # 衬底可行域 = Free 多边形顶点（窗口坐标，运行期固定）
        assert [tuple(p) for p in world.substrate.polygon] == \
            [tuple(p) for p in ground_poly]
        # 多边形障碍转样本坐标（+初始视窗原点 1100,700）
        assert len(world._poly_obstacles) == 1
        shifted = [(px + 1100, py + 700) for px, py in obs_poly]
        assert [tuple(p) for p in world._poly_obstacles[0]] == shifted
        # 规划快照中多边形障碍为真实边界（kind=polygon，窗口坐标）
        snap = world.snapshot()
        poly_obs = [o for o in snap.obstacles if o.kind == "polygon"]
        assert len(poly_obs) == 1
        wp = poly_obs[0].polygon
        assert all(0 <= px <= 800 and 0 <= py <= 600 for px, py in wp)
    finally:
        world.close()
