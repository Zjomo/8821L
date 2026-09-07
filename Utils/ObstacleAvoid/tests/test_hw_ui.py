"""硬件安全 HW-01/HW-02 + UI-01/UI-02 测试。"""
import json
import os

import pytest


# ---------------------------------------------------------------- HW
def test_hw01_dryrun_default_regression(tmp_path):
    """HW-01: dryrun 默认回归——命令全部落报告、可追溯、确定性完成。"""
    from obstacle_avoidance.cli import main
    prefix = str(tmp_path / "hw01")
    rc = main(["--scenario", "oa01", "--report-prefix", prefix])
    assert rc == 0
    report = prefix + ".jsonl"
    assert os.path.exists(report)
    events = [json.loads(l) for l in open(report, encoding="utf-8") if l.strip()]
    cmds = [e for e in events if e["event"] == "stage_command"]
    assert len(cmds) > 0
    # 每条命令可追溯 waypoint 与帧
    for c in cmds:
        assert "waypoint_index" in c and "frame_id" in c
        assert c["source_plan_version"] >= 1
    # 拒绝真实驱动
    rc2 = main(["--scenario", "oa01", "--xy-driver", "newport"])
    assert rc2 == 2


def test_hw02_comm_timeout_fault_no_more_commands(tmp_path):
    """HW-02: 通信超时 -> FAULT，后续无命令。"""
    from obstacle_avoidance.cli import build_scenario
    from obstacle_avoidance.controller import (ControllerConfig,
                                                ObstacleAvoidController)
    from obstacle_avoidance.models import FailureReason, GoalRegion, RunState
    from obstacle_avoidance.planner import GridPlanner
    from obstacle_avoidance.reporter import RunReporter
    from obstacle_avoidance.simulator import DryRunStage
    from obstacle_avoidance.vision import VisionPipeline

    world, _ = build_scenario("oa01")
    snap = world.snapshot()
    stage = world.make_stage(1)
    orig_move = stage.move_by

    class FlakyStage:
        """前 2 条成功，之后超时。"""

        def move_by(self, dx, dy, **kw):
            if len(stage.commands) >= 2:
                raise TimeoutError("simulated stage timeout")
            return orig_move(dx, dy, **kw)

        @property
        def last_command(self):
            return stage.last_command

    ctl = ObstacleAvoidController(FlakyStage(),
                                  VisionPipeline(expected_radius_px=12.0),
                                  GridPlanner(), ControllerConfig(),
                                  RunReporter(None))
    result = ctl.run(snap, 1, GoalRegion(center=(480, 240), radius_px=20),
                     task_id="hw02", get_frame=world.render,
                     get_snapshot=world.snapshot)
    assert result.final_state == RunState.FAULT
    assert result.failure_reason == FailureReason.COMM_TIMEOUT
    assert len(stage.commands) == 2  # 超时后无后续命令


# ---------------------------------------------------------------- UI
@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from obstacle_avoidance.qt_compat import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def test_ui01_offscreen_launch(qapp, tmp_path):
    """UI-01: offscreen 启动 + 打开样例帧。"""
    from obstacle_avoidance.app import MainWindow, frame_to_pix
    from obstacle_avoidance.cli import build_scenario
    win = MainWindow()
    assert win.scenario_combo.count() == 1   # UI 仅保留 sim01
    assert win.scenario_combo.currentText() == "sim01"
    # 样例帧渲染 -> QImage 转换
    world, _ = build_scenario("oa01")
    img = frame_to_pix(world.render())
    assert img.width() == 640 and img.height() == 480
    win.close()


def test_ui02_report_replay(qapp, tmp_path):
    """UI-02: 加载 JSONL 报告 -> 指标/状态/命令可复现。"""
    from obstacle_avoidance.cli import main
    from obstacle_avoidance.app import MainWindow
    prefix = str(tmp_path / "ui02")
    assert main(["--scenario", "oa01", "--report-prefix", prefix]) == 0
    win = MainWindow()
    win.report_path.setText(prefix + ".jsonl")
    win.on_replay()
    text = win.replay_out.toPlainText()
    assert "event_count" in text and "stage_command_count" in text
    assert "COMPLETE" in text


def test_ui03_worker_streams_frames(qapp):
    """UI 仿真模拟：运行场景时工作线程推送帧（预览+过程+收尾）。"""
    from obstacle_avoidance.app import WorkerThread
    frames = []
    w = WorkerThread("oa01", {})
    w.frame_ready.connect(lambda img: frames.append(img))
    w.start()
    assert w.wait(60000)
    qapp.processEvents()             # 投递跨线程排队的帧信号
    assert len(frames) >= 5          # 初始帧 + 每次渲染 + 收尾帧
    assert frames[0].shape == (480, 640, 3)
    # 过程帧与初始帧不同（球已移动 / 路径叠加）
    assert any((f != frames[0]).any() for f in frames[1:])

    # 场景预览：切换场景即显示初始帧（sim01 走后台预览线程 -> 轮询等待）
    from obstacle_avoidance.qt_compat import QtCore
    from obstacle_avoidance.app import MainWindow
    win = MainWindow()
    win.on_scenario_changed()
    for _ in range(150):
        if win.canvas.pixmap() is not None and not win.canvas.pixmap().isNull():
            break
        QtCore.QThread.msleep(100)
        qapp.processEvents()
    assert win.canvas.pixmap() is not None and not win.canvas.pixmap().isNull()


def test_ui04_xyz_jog_and_estop(qapp):
    from obstacle_avoidance.app import MainWindow
    from obstacle_avoidance.motion import MotionStateError
    win = MainWindow()
    try:
        win._ensure_live()
        stage = win._sim_live.motion_stage
        start = stage.position["z"]
        win.xyz_axis_combo.setCurrentText("z")
        win.xyz_step_spin.setValue(5.0)
        win.xyz_steps_spin.setValue(10.0)
        win.xyz_speed_spin.setValue(8.0)
        win.xyz_accel_spin.setValue(16.0)
        win.xyz_apply_btn.click()
        win.xyz_jog_plus.click()
        assert stage.position["z"] == pytest.approx(start + 5.0)
        assert "steps" in win.xyz_telemetry_out.toPlainText()
        assert stage.last_telemetry.steps["z"] == 50
        assert stage.last_telemetry.peak_speed["z"] <= 8.0 + 1e-9
        win.xyz_stop_btn.click()
        with pytest.raises(MotionStateError):
            stage.move_by({"z": 1.0})
        win.xyz_enable_btn.click()
        stage.move_by({"z": -1.0})
    finally:
        win.close()


def test_ui05_xyz_panel_compact_layout(qapp):
    """XYZ 面板保持紧凑，控件不依赖横向滚动也不会被裁切。"""
    from obstacle_avoidance.app import MainWindow

    win = MainWindow()
    try:
        win.resize(1280, 800)
        win.show()
        qapp.processEvents()
        assert win.xyz_box.height() <= 270
        assert win.xyz_telemetry_out.height() <= 100
        # Jog controls share one row; maintenance controls share another.
        assert win.xyz_jog_minus.geometry().center().y() == \
            win.xyz_jog_plus.geometry().center().y()
        centers = [getattr(win, name).geometry().center().x() for name in
                   ("xyz_home_btn", "xyz_zero_btn", "xyz_enable_btn",
                    "xyz_stop_btn")]
        assert centers == sorted(centers)
        assert win.xyz_position_label.objectName() == "xyzPositionLabel"
    finally:
        win.close()
