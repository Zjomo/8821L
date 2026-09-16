"""需求1 验收：程序关闭/重启后框定元素（衬底/球/障碍/目标点/目标范围）
的位置与对象不丢失。

覆盖：
  ① 画 衬底+球+障碍+目标点 -> 落盘 -> 新窗口载入 -> 数量与样本坐标逐一相等；
  ② 运行更新球位（on_layout_updated）-> 重启后球位为新值（不回退到运行前）；
  ③ 撤销后重启，被撤销的对象不复活；
  ④ 视窗停在别处时重启 -> 自动把视窗对齐到元素包围盒中心（元素保持可见）。
"""
import json
import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="session")
def qapp():
    from obstacle_avoidance.qt_compat import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _new_window(qapp, cfg_path: str, load: bool = False):
    """新建 MainWindow 并隔离 sim 配置路径（避免污染真实 artifacts）。"""
    from obstacle_avoidance.app import MainWindow
    win = MainWindow()
    win.SIM_ROI_CONFIG = cfg_path
    win._sim_cfg = {"grounds": [], "balls": [], "obstacles": [],
                    "obstacle_polys": []}
    win._sim_ids = {"balls": [], "obstacles": []}
    win._sim_order = []
    win._pending_view_center = None
    if load:
        win._load_sim_config()
    return win


def _draw_layout(win):
    """在虚拟模式画一整套框定元素（窗口坐标）。"""
    win._ensure_live()
    win.mode_combo.setCurrentText("衬底(ground)")
    win.on_rect_drawn(40, 40, 500, 400)
    win.mode_combo.setCurrentText("圆球(mask)")
    win.on_rect_drawn(100, 200, 30, 30)
    win.mode_combo.setCurrentText("障碍物(obstacle)")
    win.on_rect_drawn(300, 150, 60, 60)
    win.mode_combo.setCurrentText("目标点(避障)")
    win.on_rect_drawn(500, 300, 10, 10)
    win.mode_combo.setCurrentText("目标范围(组装)")
    win.on_rect_drawn(460, 360, 60, 60)


def test_sim_config_roundtrip_after_restart(qapp, tmp_path):
    """① 衬底/球/障碍/目标点/目标范围：数量与样本坐标重启后逐一相等。"""
    cfg = str(tmp_path / "sim_roi_config.json")
    win = _new_window(qapp, cfg)
    try:
        _draw_layout(win)
        assert len(win._sim_cfg["grounds"]) == 1
        assert len(win._sim_cfg["balls"]) == 1
        assert len(win._sim_cfg["obstacles"]) == 1
        assert win._sim_cfg["grounds"][0]["goal"] is not None
        assert win._sim_cfg["grounds"][0]["goal_range"] is not None
        assert os.path.isfile(cfg), "画框应实时落盘"
        saved = json.load(open(cfg, encoding="utf-8"))
        assert saved["coords"] == "sample"
        assert "view" in saved          # 视窗一并落盘
    finally:
        win.close()

    # 重启：新窗口载入同一配置
    win2 = _new_window(qapp, cfg, load=True)
    try:
        assert len(win2._sim_cfg["grounds"]) == 1
        assert len(win2._sim_cfg["balls"]) == 1
        assert len(win2._sim_cfg["obstacles"]) == 1
        assert win2._sim_cfg["grounds"] == saved["grounds"]
        assert win2._sim_cfg["balls"] == saved["balls"]
        assert win2._sim_cfg["obstacles"] == saved["obstacles"]
        # 对象 ID 保持唯一（载入后不重复/不丢失）
        gid = win2._sim_cfg["grounds"][0]["id"]
        assert gid and gid.startswith("G")
    finally:
        win2.close()


def test_run_result_survives_restart(qapp, tmp_path):
    """② 运行写回的球位必须即时落盘，重启后是新值。"""
    cfg = str(tmp_path / "sim_roi_config.json")
    win = _new_window(qapp, cfg)
    try:
        _draw_layout(win)
        ball_before = list(win._sim_cfg["balls"][0])
        # 模拟 worker 运行结束回传最终球位（运行视窗坐标）
        o = win._sim_origin()
        moved_w = [ball_before[0] - o[0] + 120, ball_before[1] - o[1],
                   ball_before[2], ball_before[3]]
        win.on_layout_updated([moved_w])
        assert win._sim_cfg["balls"][0] == win._rect_w2s(moved_w, o)
        assert win._sim_cfg["balls"][0] != ball_before
        saved_ball = list(win._sim_cfg["balls"][0])
        on_disk = json.load(open(cfg, encoding="utf-8"))["balls"][0]
        assert on_disk == saved_ball, "运行结果必须即时落盘"
    finally:
        win.close()

    win2 = _new_window(qapp, cfg, load=True)
    try:
        assert win2._sim_cfg["balls"][0] == saved_ball
    finally:
        win2.close()


def test_undo_then_restart_object_stays_removed(qapp, tmp_path):
    """③ 撤销（按当前画框模式）后重启，被撤销对象不复活。

    新需求：撤销只作用于"当前画框模式"对应的对象类型，因此逐类切换到对应
    模式后分别撤销：目标范围 -> 目标点 -> 障碍 -> 球 -> 衬底。
    """
    cfg = str(tmp_path / "sim_roi_config.json")
    win = _new_window(qapp, cfg)
    try:
        _draw_layout(win)
        # 按画框模式逐类撤销（撤销仅生效于当前模式对应的对象）
        win.mode_combo.setCurrentText("目标范围(组装)"); win.on_undo_zone()
        win.mode_combo.setCurrentText("目标点(避障)"); win.on_undo_zone()
        win.mode_combo.setCurrentText("障碍物(obstacle)"); win.on_undo_zone()
        win.mode_combo.setCurrentText("圆球(mask)"); win.on_undo_zone()
        win.mode_combo.setCurrentText("衬底(ground)"); win.on_undo_zone()
        assert win._sim_cfg["balls"] == []
        assert win._sim_cfg["obstacles"] == []
        assert win._sim_cfg["grounds"] == []
        assert win._sim_cfg["grounds"][:] == []
    finally:
        win.close()

    win2 = _new_window(qapp, cfg, load=True)
    try:
        assert win2._sim_cfg["balls"] == []
        assert win2._sim_cfg["obstacles"] == []
        assert win2._sim_cfg["grounds"] == []
    finally:
        win2.close()


def test_view_realigns_to_elements_on_restart(qapp, tmp_path):
    """④ 台位停在别处（视窗离开元素）时，重启后视窗自动对齐到元素。"""
    from obstacle_avoidance import sim_microscope

    cfg = str(tmp_path / "sim_roi_config.json")
    win = _new_window(qapp, cfg)
    try:
        _draw_layout(win)
        center = win._element_bbox_center()
        assert center is not None
        # 把台位挪到另一侧极值（软限位 ±1000µm 内），元素被带出视窗；
        # closeEvent 会把视窗一并落盘。
        pos = win._sim_live.motion_stage.position
        win._sim_live.motion_stage.move_to(
            {"x": -900.0 if pos["x"] > 0 else 900.0,
             "y": -900.0 if pos["y"] > 0 else 900.0})
        ox, oy = win._sim_origin()
        w, h = sim_microscope.WINDOW
        assert not (0 <= center[0] - ox < w and 0 <= center[1] - oy < h), \
            "构造失败：元素应已在视窗之外"
    finally:
        win.close()

    # 重启：载入时挂起"元素不在视窗内"，建立实时世界后立即对齐
    win2 = _new_window(qapp, cfg, load=True)
    try:
        assert win2._pending_view_center is not None, "应记录待对齐的视窗中心"
        win2._ensure_live()
        wx, wy = win2._sim_live._view_origin()
        w, h = sim_microscope.WINDOW
        c = win2._element_bbox_center()
        assert 0 <= c[0] - wx < w and 0 <= c[1] - wy < h, \
            f"元素 {c} 仍不在视窗 origin=({wx},{wy}) 内"
        assert win2._pending_view_center is None
    finally:
        win2.close()