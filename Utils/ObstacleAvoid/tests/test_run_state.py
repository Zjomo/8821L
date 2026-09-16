"""需求3 验收：激光/避障把圆球送到目标点后，画面停在最终达到的窗口状态与
位置，不回运行起点。

覆盖：
  ① WorkerThread：虚拟模式运行结束回传最终台位（含 Alg2 激光运行）；
     电机模式不回传（不移动/不改动真实台位）；
  ② MainWindow.on_stage_ready + on_done：实时世界视窗同步到最终台位，
     元素（目标点）仍能看到，预览刷新；视窗不回运行起点；
  ③ 未收到最终台位时 on_done 不改动视窗；
  ④ 真实虚拟模式运行（避障）：结束后画面停在 worker 最终窗口，且最终视窗
     与球位一并落盘。
"""
import json
import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from obstacle_avoidance import sim_microscope  # noqa: E402


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
    """画 衬底+球+目标点（窗口坐标）。"""
    win._ensure_live()
    win.mode_combo.setCurrentText("衬底(ground)")
    win.on_rect_drawn(40, 40, 500, 400)
    win.mode_combo.setCurrentText("圆球(mask)")
    win.on_rect_drawn(100, 200, 30, 30)
    win.mode_combo.setCurrentText("目标点(避障)")
    win.on_rect_drawn(500, 300, 10, 10)


class _FakeStage:
    position = {"x": 640.0, "y": 700.0, "z": 0.0}


class _FakeWorld:
    motion_stage = _FakeStage()


# ---------------------------------------------------------------- ① 信号
@pytest.mark.parametrize("algorithm", ["Alg1", "Alg2"])
def test_worker_emits_final_stage_in_virtual_mode(qapp, algorithm):
    """虚拟模式（含 Alg2 激光）运行结束回传最终台位。"""
    from obstacle_avoidance.app import WorkerThread
    got = []
    worker = WorkerThread("sim01", {})
    worker.stage_ready.connect(got.append)
    worker._execution_mode = "virtual"
    worker._algorithm = algorithm
    worker._emit_final_stage(_FakeWorld())
    assert got and got[-1]["x"] == 640.0 and got[-1]["y"] == 700.0


def test_worker_skips_final_stage_in_motor_mode(qapp):
    """电机模式不回传（真实台位由避障/组装控制器驱动，不被 UI 覆盖）。"""
    from obstacle_avoidance.app import WorkerThread
    got = []
    worker = WorkerThread("sim01", {})
    worker.stage_ready.connect(got.append)
    worker._execution_mode = "motor"
    worker._emit_final_stage(_FakeWorld())
    assert got == []


# ---------------------------------------------------------------- ② 视窗停留
def test_view_stays_at_final_stage_after_run(qapp, tmp_path):
    """运行结束：视窗停在最终台位（目标点在画面内），不回运行起点。"""
    cfg = str(tmp_path / "sim_roi_config.json")
    win = _new_window(qapp, cfg)
    try:
        _draw_layout(win)
        start = dict(win._sim_live.motion_stage.position)
        origin_before = win._sim_origin()
        goal = win._sim_cfg["grounds"][0]["goal"]      # 样本坐标
        # worker 回传的最终台位：目标点进入视窗中心（µm）
        final = {"x": goal[0] * sim_microscope.SIM_PIXEL_SIZE,
                 "y": goal[1] * sim_microscope.SIM_PIXEL_SIZE, "z": 0.0}
        assert abs(final["x"]) < 1000.0 and abs(final["y"]) < 1000.0
        win.on_stage_ready(final)
        win.on_done(0)
        # 实时世界台位 = 最终台位（不停在起点）
        pos = win._sim_live.motion_stage.position
        assert abs(pos["x"] - final["x"]) < 1e-6
        assert abs(pos["y"] - final["y"]) < 1e-6
        assert (pos["x"], pos["y"]) != (start["x"], start["y"])
        assert win._sim_origin() != origin_before
        # 最终窗口状态：目标点仍在画面内（且居视窗中心）
        w, h = sim_microscope.WINDOW
        ox, oy = win._sim_origin()
        assert 0 <= goal[0] - ox < w and 0 <= goal[1] - oy < h, \
            f"目标点 {goal} 不在最终视窗 origin=({ox},{oy}) 内"
        assert abs((goal[0] - ox) - w / 2.0) < 1.0
        assert abs((goal[1] - oy) - h / 2.0) < 1.0
        # 需求1：最终视窗与元素一并落盘
        saved = json.load(open(cfg, encoding="utf-8"))
        assert saved["view"]["stage_um"][:2] == [pytest.approx(final["x"]),
                                                 pytest.approx(final["y"])]
        assert saved["balls"] == win._sim_cfg["balls"]
    finally:
        win.close()


def test_on_done_without_final_stage_keeps_view(qapp, tmp_path):
    """未收到最终台位（电机模式/中止）时 on_done 不改动视窗。"""
    cfg = str(tmp_path / "sim_roi_config.json")
    win = _new_window(qapp, cfg)
    try:
        _draw_layout(win)
        before = dict(win._sim_live.motion_stage.position)
        win._final_stage_position = None
        win.on_done(1)
        after = win._sim_live.motion_stage.position
        assert (before["x"], before["y"]) == (after["x"], after["y"])
    finally:
        win.close()


# ---------------------------------------------------------------- ④ 端到端
def test_real_virtual_run_ends_at_final_view(qapp, tmp_path):
    """真实虚拟运行：结束画面停在 worker 最终窗口（非运行起点）。"""
    cfg = str(tmp_path / "sim_roi_config.json")
    win = _new_window(qapp, cfg)
    captured = []
    try:
        _draw_layout(win)
        # Alg2（标定激光）在仿真里移动台位/样品，窗口位置随运行改变；
        # 目标点画在标定光斑处（光镊把球吸到光斑 = 到达目标点）。
        win.alg_combo.setCurrentText("Alg2")
        win._beam_calibrated = True
        win.ball_step_spin.setValue(0.02)   # 单步 40px（光镊对准需大步）
        w, h = sim_microscope.WINDOW
        win.beam_x_spin.setValue(w / 2.0)
        win.beam_y_spin.setValue(h / 2.0)
        win.mode_combo.setCurrentText("目标点(避障)")
        win.on_rect_drawn(w / 2.0 - 5, h / 2.0 - 5, 10, 10)
        start = dict(win._sim_live.motion_stage.position)
        win.on_run(mode="oa")
        qapp.processEvents()
        worker = win.worker
        assert worker is not None
        worker.stage_ready.connect(captured.append)   # 捕获最终台位
        assert worker.wait(180000)
        qapp.processEvents()
        assert "COMPLETE" in win.state_label.text(), win.state_label.text()
        assert captured, "虚拟运行结束应回传最终台位"
        final = captured[-1]
        pos = win._sim_live.motion_stage.position
        assert abs(pos["x"] - final["x"]) < 1e-6 and \
            abs(pos["y"] - final["y"]) < 1e-6, "实时世界未同步到最终台位"
        assert (pos["x"], pos["y"]) != (start["x"], start["y"]), \
            "画面被带回运行起点"
        # 目标点仍在最终画面内（停在最终窗口状态）
        goal = win._sim_cfg["grounds"][0]["goal"]
        ox, oy = win._sim_origin()
        assert 0 <= goal[0] - ox < w and 0 <= goal[1] - oy < h
        saved = json.load(open(cfg, encoding="utf-8"))
        assert saved["view"]["stage_um"][:2] == [pytest.approx(final["x"]),
                                                 pytest.approx(final["y"])]
    finally:
        if win.worker is not None and win.worker.isRunning():
            win.worker.request_stop()
            win.worker.wait(10000)
        win.close()