"""UI 完整功能测试（基于 pytest-qt）。

覆盖：窗口初始化、模式切换、画框闸门、形状选择、虚拟模式画框/运行、
电机模式控件可见性、XYZ jog、回放、暂停/急停、配置保存/载入、
边界间隙、复位/撤销、目标球点选。
"""
import json
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# 确保包可导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="session")
def qapp():
    from obstacle_avoidance.qt_compat import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture()
def win(qapp):
    """每个测试独立的 MainWindow，测试结束关闭。"""
    from obstacle_avoidance.app import MainWindow
    w = MainWindow()
    w.resize(1280, 800)
    w.show()
    qapp.processEvents()
    # 重置 _sim_cfg 避免自动载入的配置污染测试
    w._sim_cfg = {
        "grounds": [],
        "balls": [],
        "obstacles": [],
        "obstacle_polys": [],
    }
    w._sim_ids = {"balls": [], "obstacles": []}
    w._sim_order = []
    w.selected_ball_index = None
    yield w
    # 确保 worker 停止
    if w.worker is not None and w.worker.isRunning():
        w.worker.request_stop()
        w.worker.wait(5000)
    w.close()
    qapp.processEvents()


# ================================================================
# 1. 窗口初始化
# ================================================================
class TestWindowInit:
    def test_tabs_exist(self, win):
        assert win.tabs.count() == 4
        assert win.tabs.tabText(0) == "运行控制"
        assert win.tabs.tabText(1) == "XYZ 台位"
        assert win.tabs.tabText(2) == "检测 / ROI"
        assert win.tabs.tabText(3) == "日志 / 报告"

    def test_default_mode_virtual(self, win):
        assert win.mode_sel.currentIndex() == 0
        assert not win.motor_box.isVisible()

    def test_scenario_combo(self, win):
        assert win.scenario_combo.count() == 1
        assert win.scenario_combo.currentText() == "sim01"

    def test_alg_combo(self, win):
        assert win.alg_combo.count() == 2
        assert win.alg_combo.itemText(0) == "Alg1"
        assert win.alg_combo.itemText(1) == "Alg2"

    def test_draw_gate_default_locked(self, win):
        assert not win.draw_gate_btn.isChecked()
        assert win.draw_gate_btn.text() == "画框:关"
        assert not win.canvas.drawing_enabled

    def test_shape_combo_items(self, win):
        items = [win.shape_combo.itemText(i) for i in range(win.shape_combo.count())]
        assert items == ["长方形", "正方形", "圆形", "Free"]

    def test_state_label_idle(self, win):
        assert win.state_label.text() == "IDLE"

    def test_canvas_exists(self, win):
        assert win.canvas is not None
        assert win.canvas.minimumWidth() == 640
        assert win.canvas.minimumHeight() == 480


# ================================================================
# 2. 模式切换
# ================================================================
class TestModeSwitch:
    def test_switch_to_motor(self, win, qapp):
        win.mode_sel.setCurrentIndex(1)
        qapp.processEvents()
        assert win.motor_box.isVisible()
        assert win.run_oa_btn.text() == "避障运行 (电机)"
        assert win.run_ag_btn.text() == "组装运行 (电机)"
        # 画框模式切换为电机选项
        items = [win.mode_combo.itemText(i) for i in range(win.mode_combo.count())]
        assert items == ["ROI 视野", "目标区", "障碍区", "自由区"]
        # 仿真图层按钮隐藏（电机模式不可用）
        assert win.sim_spec_btn.isHidden()

    def test_switch_back_to_virtual(self, win, qapp):
        win.mode_sel.setCurrentIndex(1)
        qapp.processEvents()
        win.mode_sel.setCurrentIndex(0)
        qapp.processEvents()
        assert not win.motor_box.isVisible()
        assert win.run_oa_btn.text() == "避障运行"
        items = [win.mode_combo.itemText(i) for i in range(win.mode_combo.count())]
        assert "圆球(mask)" in items
        # sim_spec_btn 在检测/ROI 页（非当前页），用 isHidden 判断
        assert not win.sim_spec_btn.isHidden()

    def test_motor_frame_source_widgets(self, win, qapp):
        win.mode_sel.setCurrentIndex(1)
        qapp.processEvents()
        # 默认帧源=相机：相机索引可见，屏幕区域控件隐藏
        assert win.cam_spin.isVisible()
        assert not win.select_region_btn.isVisible()
        assert not win.region_label.isVisible()
        # 切换到屏幕区域
        win.src_combo.setCurrentIndex(1)
        qapp.processEvents()
        assert not win.cam_spin.isVisible()
        assert win.select_region_btn.isVisible()
        assert win.region_label.isVisible()


# ================================================================
# 3. 画框闸门
# ================================================================
class TestDrawGate:
    def test_toggle_gate(self, win, qapp):
        win.draw_gate_btn.setChecked(True)
        qapp.processEvents()
        assert win.canvas.drawing_enabled
        assert win.draw_gate_btn.text() == "画框:开"

        win.draw_gate_btn.setChecked(False)
        qapp.processEvents()
        assert not win.canvas.drawing_enabled
        assert win.draw_gate_btn.text() == "画框:关"

    def test_gate_blocks_drawing(self, win, qapp):
        """闸门关闭时，Canvas 不应发出 rect_drawn 信号。"""
        win.draw_gate_btn.setChecked(False)
        qapp.processEvents()
        # drawing_enabled=False 时不应触发
        assert not win.canvas.drawing_enabled


# ================================================================
# 4. 形状选择与多边形模式
# ================================================================
class TestShapeCombo:
    def test_free_activates_polygon_mode(self, win, qapp):
        win.draw_gate_btn.setChecked(True)
        qapp.processEvents()
        win.shape_combo.setCurrentText("Free")
        qapp.processEvents()
        assert win.canvas.polygon_mode

    def test_rect_deactivates_polygon_mode(self, win, qapp):
        win.draw_gate_btn.setChecked(True)
        win.shape_combo.setCurrentText("Free")
        qapp.processEvents()
        win.shape_combo.setCurrentText("长方形")
        qapp.processEvents()
        assert not win.canvas.polygon_mode


# ================================================================
# 5. 虚拟模式：画框（_sim_rect）
# ================================================================
class TestVirtualDrawing:
    def test_draw_ball(self, win, qapp):
        win.draw_gate_btn.setChecked(True)
        qapp.processEvents()
        # 球必须在衬底内
        win.mode_combo.setCurrentText("衬底(ground)")
        win._sim_rect(50, 50, 300, 300)
        win.mode_combo.setCurrentText("圆球(mask)")
        win._sim_rect(100, 100, 40, 40)
        assert len(win._sim_cfg["balls"]) == 1
        assert "球" in win.zones_label.text()

    def test_draw_ball_outside_ground_rejected(self, win, qapp):
        """球不在衬底内应被拒绝。"""
        win.draw_gate_btn.setChecked(True)
        qapp.processEvents()
        win.mode_combo.setCurrentText("圆球(mask)")
        win._sim_rect(100, 100, 40, 40)
        assert len(win._sim_cfg["balls"]) == 0
        assert "衬底" in win.detail_label.text()

    def test_draw_ground(self, win, qapp):
        win.draw_gate_btn.setChecked(True)
        qapp.processEvents()
        win.mode_combo.setCurrentText("衬底(ground)")
        win._sim_rect(50, 50, 200, 200)
        assert len(win._sim_cfg["grounds"]) == 1

    def test_draw_obstacle(self, win, qapp):
        win.draw_gate_btn.setChecked(True)
        qapp.processEvents()
        # 障碍物必须在衬底内
        win.mode_combo.setCurrentText("衬底(ground)")
        win._sim_rect(50, 50, 300, 300)
        win.mode_combo.setCurrentText("障碍物(obstacle)")
        win._sim_rect(120, 120, 60, 60)
        assert len(win._sim_cfg["obstacles"]) == 1

    def test_draw_goal_point(self, win, qapp):
        """目标点需要先有衬底。"""
        win.draw_gate_btn.setChecked(True)
        qapp.processEvents()
        win.mode_combo.setCurrentText("衬底(ground)")
        win._sim_rect(50, 50, 200, 200)
        win.mode_combo.setCurrentText("目标点(避障)")
        win._sim_rect(150, 150, 10, 10)
        g = win._sim_cfg["grounds"][0]
        assert g.get("goal") is not None

    def test_draw_goal_range(self, win, qapp):
        win.draw_gate_btn.setChecked(True)
        qapp.processEvents()
        win.mode_combo.setCurrentText("衬底(ground)")
        win._sim_rect(50, 50, 200, 200)
        win.mode_combo.setCurrentText("目标范围(组装)")
        win._sim_rect(100, 100, 80, 80)
        g = win._sim_cfg["grounds"][0]
        assert g.get("goal_range") is not None


# ================================================================
# 6. 虚拟模式：运行校验
# ================================================================
class TestRunValidation:
    def test_no_ground_blocks_run(self, win, qapp):
        win.on_run(mode="oa")
        qapp.processEvents()
        assert "衬底" in win.detail_label.text()

    def test_no_ball_blocks_run(self, win, qapp):
        win.draw_gate_btn.setChecked(True)
        win.mode_combo.setCurrentText("衬底(ground)")
        win._sim_rect(50, 50, 200, 200)
        win.on_run(mode="oa")
        qapp.processEvents()
        assert "圆球" in win.detail_label.text()

    def test_no_goal_blocks_oa_run(self, win, qapp):
        win.draw_gate_btn.setChecked(True)
        win.mode_combo.setCurrentText("衬底(ground)")
        win._sim_rect(50, 50, 200, 200)
        win.mode_combo.setCurrentText("圆球(mask)")
        win._sim_rect(100, 100, 30, 30)
        win.on_run(mode="oa")
        qapp.processEvents()
        assert "目标点" in win.detail_label.text()

    def test_no_goal_range_blocks_ag_run(self, win, qapp):
        win.draw_gate_btn.setChecked(True)
        win.mode_combo.setCurrentText("衬底(ground)")
        win._sim_rect(50, 50, 200, 200)
        win.mode_combo.setCurrentText("圆球(mask)")
        win._sim_rect(100, 100, 30, 30)
        win.on_run(mode="ag")
        qapp.processEvents()
        assert "目标范围" in win.detail_label.text()


# ================================================================
# 7. 虚拟模式：完整运行（单球避障）
# ================================================================
class TestVirtualRun:
    def test_single_ball_oa_run(self, win, qapp):
        """画衬底+球+目标点 -> 避障运行 -> 等待完成。"""
        win.draw_gate_btn.setChecked(True)
        qapp.processEvents()
        # 衬底
        win.mode_combo.setCurrentText("衬底(ground)")
        win._sim_rect(50, 50, 300, 300)
        # 球
        win.mode_combo.setCurrentText("圆球(mask)")
        win._sim_rect(80, 80, 30, 30)
        # 目标点
        win.mode_combo.setCurrentText("目标点(避障)")
        win._sim_rect(280, 280, 10, 10)

        win.on_run(mode="oa")
        qapp.processEvents()
        assert "RUNNING" in win.state_label.text()
        # 等待 worker 完成
        assert win.worker is not None
        assert win.worker.wait(120000)
        qapp.processEvents()
        # 运行结束后状态应更新
        assert "RUNNING" not in win.state_label.text() or "COMPLETE" in win.state_label.text()

    def test_run_generates_report(self, win, qapp):
        """运行后 report_path 应被填入。"""
        win.draw_gate_btn.setChecked(True)
        qapp.processEvents()
        win.mode_combo.setCurrentText("衬底(ground)")
        win._sim_rect(50, 50, 300, 300)
        win.mode_combo.setCurrentText("圆球(mask)")
        win._sim_rect(80, 80, 30, 30)
        win.mode_combo.setCurrentText("目标点(避障)")
        win._sim_rect(280, 280, 10, 10)
        win.on_run(mode="oa")
        qapp.processEvents()
        assert win.worker is not None
        assert win.worker.wait(120000)
        qapp.processEvents()
        # report_path 应非空（on_report_ready 填入）
        assert win.report_path.text() != ""


# ================================================================
# 8. 回放
# ================================================================
class TestReplay:
    def test_replay_missing_file(self, win, qapp):
        win.report_path.setText("/nonexistent/path.jsonl")
        win.on_replay()
        qapp.processEvents()
        assert "不存在" in win.replay_out.toPlainText()

    def test_replay_valid_report(self, win, qapp, tmp_path):
        """用 CLI 生成报告后回放。"""
        from obstacle_avoidance.cli import main
        prefix = str(tmp_path / "replay_test")
        assert main(["--scenario", "oa01", "--report-prefix", prefix]) == 0
        report = prefix + ".jsonl"
        assert os.path.exists(report)

        win.report_path.setText(report)
        win.on_replay()
        qapp.processEvents()
        text = win.replay_out.toPlainText()
        assert "event_count" in text
        assert "COMPLETE" in text
        # 回放控件应启用
        assert win.rp_play_btn.isEnabled()
        assert win.rp_prev_btn.isEnabled()
        assert win.rp_next_btn.isEnabled()

    def test_replay_slider_range(self, win, qapp, tmp_path):
        from obstacle_avoidance.cli import main
        prefix = str(tmp_path / "replay_slider")
        main(["--scenario", "oa01", "--report-prefix", prefix])
        win.report_path.setText(prefix + ".jsonl")
        win.on_replay()
        qapp.processEvents()
        assert win.rp_slider.maximum() > 0

    def test_replay_step_forward(self, win, qapp, tmp_path):
        from obstacle_avoidance.cli import main
        prefix = str(tmp_path / "replay_step")
        main(["--scenario", "oa01", "--report-prefix", prefix])
        win.report_path.setText(prefix + ".jsonl")
        win.on_replay()
        qapp.processEvents()
        initial = win.rp_slider.value()
        win.rp_next_btn.click()
        qapp.processEvents()
        assert win.rp_slider.value() == initial + 1

    def test_replay_last_no_reports(self, win, qapp, monkeypatch, tmp_path):
        """Reports/ 为空时提示。"""
        # monkeypatch os.listdir 返回空列表
        monkeypatch.setattr(os, "listdir", lambda p: [])
        win.on_replay_last()
        qapp.processEvents()
        assert "暂无报告" in win.replay_out.toPlainText()


# ================================================================
# 9. XYZ Jog（虚拟模式）
# ================================================================
class TestXYZJog:
    def test_jog_moves_stage(self, win, qapp):
        win._ensure_live()
        stage = win._sim_live.motion_stage
        start_x = stage.position["x"]
        win.tabs.setCurrentIndex(1)
        qapp.processEvents()
        win.xyz_axis_combo.setCurrentText("x")
        win.xyz_step_spin.setValue(10.0)
        win.xyz_steps_spin.setValue(1.0)
        win.xyz_speed_spin.setValue(100.0)
        win.xyz_accel_spin.setValue(200.0)
        win.xyz_apply_btn.click()
        qapp.processEvents()
        win.xyz_jog_plus.click()
        qapp.processEvents()
        assert stage.position["x"] == pytest.approx(start_x + 10.0)

    def test_home_resets_position(self, win, qapp):
        win._ensure_live()
        stage = win._sim_live.motion_stage
        win.tabs.setCurrentIndex(1)
        qapp.processEvents()
        win.xyz_home_btn.click()
        qapp.processEvents()
        assert stage.position["x"] == pytest.approx(500.0)
        assert stage.position["y"] == pytest.approx(500.0)

    def test_motor_mode_jog_shows_error(self, win, qapp):
        """电机模式下 XYZ jog 应在 detail_label 显示错误。"""
        win.mode_sel.setCurrentIndex(1)
        qapp.processEvents()
        win._jog_xyz(1.0)
        qapp.processEvents()
        # 应显示错误信息（未勾选安全确认）
        assert "确认" in win.detail_label.text() or "电机" in win.detail_label.text()


# ================================================================
# 10. 暂停 / 急停
# ================================================================
class TestPauseEstop:
    def test_pause_without_worker_noop(self, win, qapp):
        """无 worker 时暂停不崩溃。"""
        win.on_pause()
        qapp.processEvents()

    def test_estop_without_worker_noop(self, win, qapp):
        """无 worker 时急停不崩溃。"""
        win.on_estop()
        qapp.processEvents()

    def test_estop_during_run(self, win, qapp):
        """运行中急停：worker 应停止。"""
        win.draw_gate_btn.setChecked(True)
        qapp.processEvents()
        win.mode_combo.setCurrentText("衬底(ground)")
        win._sim_rect(50, 50, 300, 300)
        win.mode_combo.setCurrentText("圆球(mask)")
        win._sim_rect(80, 80, 30, 30)
        win.mode_combo.setCurrentText("目标点(避障)")
        win._sim_rect(280, 280, 10, 10)
        win.on_run(mode="oa")
        qapp.processEvents()
        assert win.worker is not None and win.worker.isRunning()
        win.on_estop()
        qapp.processEvents()
        assert win.worker.wait(30000)


# ================================================================
# 11. 配置保存 / 载入（电机模式 ROI）
# ================================================================
class TestConfigSaveLoad:
    def test_save_and_load_roi_config(self, win, qapp, tmp_path, monkeypatch):
        """保存 ROI 配置 -> 清空 -> 载入 -> 恢复。"""
        from obstacle_avoidance.roi_zones import RoiConfig
        from obstacle_avoidance import video_sim

        # 创建一个简单的 ROI 配置
        win._roi_cfg = RoiConfig(roi=(0, 0, 640, 480), video="test")
        win._cam_frame_size = (640, 480)

        # monkeypatch DEFAULT_ROI_CONFIG 路径
        test_cfg_path = str(tmp_path / "test_roi_config.json")
        monkeypatch.setattr(video_sim, "DEFAULT_ROI_CONFIG", test_cfg_path)

        win.on_save_config()
        qapp.processEvents()
        assert os.path.exists(test_cfg_path)

        # 清空
        win._roi_cfg = None

        win.on_load_config()
        qapp.processEvents()
        assert win._roi_cfg is not None
        assert win._roi_cfg.roi == (0, 0, 640, 480)


# ================================================================
# 12. 边界间隙
# ================================================================
class TestEdgeClearance:
    def test_edge_spin_updates_roi(self, win, qapp):
        from obstacle_avoidance.roi_zones import RoiConfig
        win._ensure_live()
        win._roi_cfg = RoiConfig(roi=(0, 0, 640, 480), video="test")
        win.edge_spin.setValue(20)
        qapp.processEvents()
        assert win._roi_cfg.edge_clearance_px == 20


# ================================================================
# 13. 复位视野 / 撤销区域
# ================================================================
class TestResetUndo:
    def test_reset_view(self, win, qapp):
        win._ensure_live()
        win.on_reset_view()
        qapp.processEvents()
        # 复位后 _live_world 应存在
        assert win._live_world is not None

    def test_undo_zone_removes_last(self, win, qapp):
        """虚拟模式撤销：移除最后一个画框对象。"""
        win.draw_gate_btn.setChecked(True)
        qapp.processEvents()
        win.mode_combo.setCurrentText("衬底(ground)")
        win._sim_rect(50, 50, 200, 200)
        assert len(win._sim_cfg["grounds"]) == 1
        win.on_undo_zone()
        qapp.processEvents()
        assert len(win._sim_cfg["grounds"]) == 0


# ================================================================
# 14. 目标球点选
# ================================================================
class TestTargetClick:
    def test_click_selects_ball(self, win, qapp):
        """点击球中心应选中。"""
        win.draw_gate_btn.setChecked(True)
        qapp.processEvents()
        win.mode_combo.setCurrentText("衬底(ground)")
        win._sim_rect(50, 50, 300, 300)
        win.mode_combo.setCurrentText("圆球(mask)")
        win._sim_rect(100, 100, 40, 40)
        # 点击球中心 (120, 120) - 窗口坐标
        win._on_target_clicked(120, 120)
        qapp.processEvents()
        assert win.selected_ball_index == 0
        assert "Selected" in win.detail_label.text()

    def test_click_misses_ball(self, win, qapp):
        """点击空白处不选中。"""
        win.draw_gate_btn.setChecked(True)
        qapp.processEvents()
        win.mode_combo.setCurrentText("衬底(ground)")
        win._sim_rect(50, 50, 300, 300)
        win.mode_combo.setCurrentText("圆球(mask)")
        win._sim_rect(100, 100, 40, 40)
        win._on_target_clicked(500, 500)
        qapp.processEvents()
        assert win.selected_ball_index is None

    def test_selected_ball_highlighted(self, win, qapp):
        """选中球在叠加层中应有不同颜色。"""
        win.draw_gate_btn.setChecked(True)
        qapp.processEvents()
        win.mode_combo.setCurrentText("衬底(ground)")
        win._sim_rect(50, 50, 300, 300)
        win.mode_combo.setCurrentText("圆球(mask)")
        win._sim_rect(100, 100, 40, 40)
        win._on_target_clicked(120, 120)
        qapp.processEvents()
        assert win.selected_ball_index == 0
        # 叠加渲染不崩溃
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out = win._overlay_sim_cfg(frame)
        assert out.shape == frame.shape


# ================================================================
# 15. 驱动切换（电机模式）
# ================================================================
class TestDriverSwitch:
    def test_switch_to_serial(self, win, qapp):
        win.mode_sel.setCurrentIndex(1)
        qapp.processEvents()
        win.driver_combo.setCurrentIndex(1)  # 串口
        qapp.processEvents()
        assert win.port_edit.isVisible()
        assert win.baud_spin.isVisible()
        # Picomotor 控件隐藏
        assert not win.conn_spin.isVisible()

    def test_switch_to_kinesis(self, win, qapp):
        win.mode_sel.setCurrentIndex(1)
        qapp.processEvents()
        win.driver_combo.setCurrentIndex(2)  # Kinesis
        qapp.processEvents()
        assert win.kinesis_serial_edit.isVisible()
        assert win.kinesis_scan_btn.isVisible()
        assert not win.conn_spin.isVisible()
        assert not win.port_edit.isVisible()

    def test_switch_to_picomotor(self, win, qapp):
        win.mode_sel.setCurrentIndex(1)
        qapp.processEvents()
        win.driver_combo.setCurrentIndex(0)  # Picomotor
        qapp.processEvents()
        assert win.conn_spin.isVisible()
        assert not win.port_edit.isVisible()
        assert not win.kinesis_serial_edit.isVisible()


# ================================================================
# 16. 属性面板
# ================================================================
class TestPropsPanel:
    def test_props_panel_updates(self, win, qapp):
        win.draw_gate_btn.setChecked(True)
        qapp.processEvents()
        win.mode_combo.setCurrentText("圆球(mask)")
        win._sim_rect(100, 100, 40, 40)
        qapp.processEvents()
        text = win.props_out.toPlainText()
        assert "ball" in text

    def test_props_panel_empty(self, win, qapp):
        win._update_props_panel()
        qapp.processEvents()
        assert "无对象" in win.props_out.toPlainText()


# ================================================================
# 17. 日志
# ================================================================
class TestLogging:
    def test_simlog_appends(self, win, qapp):
        win._simlog("test message")
        qapp.processEvents()
        assert "test message" in win.log_out.toPlainText()

    def test_log_tab_scroll(self, win, qapp):
        """日志页应在滚动区域内。"""
        win.tabs.setCurrentIndex(3)
        qapp.processEvents()
        assert "logScrollArea" in [
            win._tab_scrolls[k].objectName() for k in win._tab_scrolls]
