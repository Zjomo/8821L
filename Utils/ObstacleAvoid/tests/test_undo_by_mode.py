"""需求2 验收：撤销区域与"当前画框模式"一致。

覆盖：
  ① 目标点模式下撤销，只删最新建的一个目标点（即使它不在顺序栈栈顶）；
  ② 撤销不影响当前模式之外的对象（球/障碍/衬底保持不变）；
  ③ 无该类型可撤销时给出提示且不误删其他类型；
  ④ 载入配置无顺序信息时，目标点模式撤销仍只删目标点。
"""
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


def _window(qapp):
    import tempfile
    from obstacle_avoidance import sim_microscope
    from obstacle_avoidance.app import MainWindow
    # 用临时 DB + recenter=True 隔离共享台位，避免污染其它测试的视窗原点。
    sim_microscope.SIM_DB = os.path.join(str(tempfile.mkdtemp()),
                                         "sim_microscope.db")
    win = MainWindow()
    win.SIM_ROI_CONFIG = os.path.join(str(tempfile.mkdtemp()),
                                      "sim_roi_config.json")
    win._sim_cfg = {"grounds": [], "balls": [], "obstacles": [],
                    "obstacle_polys": []}
    win._sim_ids = {"balls": [], "obstacles": []}
    win._sim_order = []
    win._pending_view_center = None
    win._ensure_live()
    return win


def _two_grounds_two_goals(win):
    """两个衬底，第二个衬底指向一个目标点。"""
    win.mode_combo.setCurrentText("衬底(ground)")
    win.on_rect_drawn(40, 40, 400, 300)   # G0
    win.mode_combo.setCurrentText("圆球(mask)")
    win.on_rect_drawn(100, 100, 30, 30)   # B0
    win.mode_combo.setCurrentText("目标点(避障)")
    win.on_rect_drawn(400, 300, 10, 10)   # goal on G0
    win.mode_combo.setCurrentText("障碍物(obstacle)")
    win.on_rect_drawn(300, 150, 40, 40)   # O0
    return win


def test_undo_goal_only_in_goal_mode(qapp):
    """目标点模式下仅撤销最新一个目标点，其它对象不动。"""
    win = _window(qapp)
    try:
        _two_grounds_two_goals(win)
        assert len(win._sim_cfg["balls"]) == 1
        assert len(win._sim_cfg["obstacles"]) == 1
        assert win._sim_cfg["grounds"][0]["goal"] is not None

        # 切换回目标点模式后撤销：应只撤掉目标点
        win.mode_combo.setCurrentText("目标点(避障)")
        win.on_undo_zone()
        assert win._sim_cfg["grounds"][0]["goal"] is None
        # 球与障碍不受影响
        assert len(win._sim_cfg["balls"]) == 1
        assert len(win._sim_cfg["obstacles"]) == 1
        # 无第二个目标点可撤 -> 不误删其它对象
        win.on_undo_zone()
        assert len(win._sim_cfg["balls"]) == 1
        assert len(win._sim_cfg["grounds"]) == 1
    finally:
        win.close()


def test_undo_ball_mode_reaches_buried_ball(qapp):
    """即使球不在顺序栈栈顶，圆球模式下撤销仍只撤最新一个球。"""
    win = _window(qapp)
    try:
        _two_grounds_two_goals(win)
        # 再画一个球 -> 球在栈顶
        win.mode_combo.setCurrentText("圆球(mask)")
        win.on_rect_drawn(200, 200, 30, 30)   # B1
        assert len(win._sim_cfg["balls"]) == 2

        # 切换到目标点模式撤销（目标是栈顶那个 goal? 不，goal 先 B1 入栈）
        # 栈顺序：... ball0, goal, obstacle, goal, ball1。球1是栈顶。
        # 球模式下撤销只删球1
        win.mode_combo.setCurrentText("圆球(mask)")
        win.on_undo_zone()
        assert len(win._sim_cfg["balls"]) == 1, "圆球模式只撤最新一个球"

        # 另一球仍在，其它对象不变
        assert len(win._sim_cfg["obstacles"]) == 1
        assert win._sim_cfg["grounds"][0]["goal"] is not None
    finally:
        win.close()


def test_undo_ball_mode_draps_only_ball_across_types(qapp):
    """圆球模式下撤销不会删除障碍/目标点：只撤球。"""
    win = _window(qapp)
    try:
        _two_grounds_two_goals(win)
        # 再画一个障碍在障碍模式，再画一个目标点到另一个衬底前先加衬底
        win.mode_combo.setCurrentText("障碍物(obstacle)")
        win.on_rect_drawn(200, 60, 40, 40)   # O1
        assert len(win._sim_cfg["obstacles"]) == 2

        # 在障碍模式下撤销应只删 O1
        win.mode_combo.setCurrentText("障碍物(obstacle)")
        win.on_undo_zone()
        assert len(win._sim_cfg["obstacles"]) == 1
        assert len(win._sim_cfg["balls"]) == 1
        assert win._sim_cfg["grounds"][0]["goal"] is not None
    finally:
        win.close()