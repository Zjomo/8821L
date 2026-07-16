"""PyQt 主窗口：PI 光谱仪测试 UI。"""

from __future__ import annotations

import csv
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from pi_spectrometer.ui.qt_compat import (
    QtCore, QtGui, QtWidgets, Signal, Slot, AlignCenter, QAction
)

from pi_spectrometer.core.base import SpectrometerBackend
from pi_spectrometer.core.exceptions import SpectrometerError
from pi_spectrometer.core.recipe import (
    AcquisitionRecipe,
    RecipeAction,
    RecipeRunner,
    RecipeStep,
)
from pi_spectrometer.core.types import ROI, SpectrometerResult
from pi_spectrometer.picam.camera import PICamCamera
from pi_spectrometer.picam.demo import DemoCamera, MockSpectrometerBackend
from pi_spectrometer.processing.background import (
    BackgroundFrame,
    BackgroundFrameLibrary,
    correct_spectrometer_result,
)
from pi_spectrometer.processing.stats import compute_spectrum_stats
from pi_spectrometer.ui.plot_widget import PlotWidget

# 延迟导入 IsoPlane，避免在 64-bit Python 中立即加载 32-bit DLL
try:
    from pi_spectrometer.picam.isoplane import IsoPlaneBackend
except Exception:
    IsoPlaneBackend = None


class WorkerSignals(QtCore.QObject):
    """工作线程信号。"""

    result_ready = Signal(SpectrometerResult)
    error = Signal(str)
    log = Signal(str)


class AcquireWorker(QtCore.QRunnable):
    """在后台线程执行光谱采集。"""

    def __init__(self, backend: SpectrometerBackend, cycle_index: int = 0):
        super().__init__()
        self.backend = backend
        self.cycle_index = cycle_index
        self.signals = WorkerSignals()

    def run(self) -> None:
        try:
            self.signals.log.emit(f"开始采集 #{self.cycle_index}")
            result = self.backend.acquire(num_frames=1)
            result.index = self.cycle_index
            self.signals.log.emit(
                f"采集完成 #{self.cycle_index}: {result.num_points} 点, "
                f"raw_peak={result.raw_peak:.2f}, fit_peak={result.fit_peak:.2f}"
                if result.raw_peak is not None and result.fit_peak is not None
                else f"采集完成 #{self.cycle_index}: {result.num_points} 点"
            )
            self.signals.result_ready.emit(result)
        except Exception as e:
            self.signals.error.emit(str(e))


class MainWindow(QtWidgets.QMainWindow):
    """PI 光谱仪测试主窗口。"""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Princeton Instruments 光谱仪测试 UI")
        self.resize(1200, 800)

        self.backend: Optional[SpectrometerBackend] = None
        self.continuous_timer: Optional[QtCore.QTimer] = None
        self.cycle_index = 0
        self.last_result: Optional[SpectrometerResult] = None
        self.pool = QtCore.QThreadPool.globalInstance()

        # LightField 风格增强功能
        self.bg_library = BackgroundFrameLibrary()
        self.current_dark_name: Optional[str] = None
        self.bg_correction_enabled = False
        self.history_enabled = False
        self.annotate_enabled = True
        self.auto_save_enabled = False
        self.auto_save_dir = Path(".")
        self.auto_save_template = "spectrum_{index:04d}_{timestamp}.csv"

        self.recipe: AcquisitionRecipe = AcquisitionRecipe(name="default")
        self.recipe_runner: Optional[RecipeRunner] = None
        self.recipe_thread: Optional[threading.Thread] = None

        self._setup_ui()
        self._log("UI 初始化完成。请选择后端并点击“连接”。")

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        h_layout = QtWidgets.QHBoxLayout(central)

        # 左侧控制面板
        self.control_panel = self._build_control_panel()
        h_layout.addWidget(self.control_panel, 1)

        # 右侧绘图 + 日志
        right = QtWidgets.QVBoxLayout()
        self.plot = PlotWidget()
        right.addWidget(self.plot, 3)

        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        if hasattr(self.log_text, "setMaximumBlockCount"):
            self.log_text.setMaximumBlockCount(500)
        right.addWidget(self.log_text, 1)

        right_widget = QtWidgets.QWidget()
        right_widget.setLayout(right)
        h_layout.addWidget(right_widget, 3)

        self._build_menu()

    def _build_control_panel(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("控制面板")
        layout = QtWidgets.QFormLayout(group)

        # 后端选择
        self.backend_combo = QtWidgets.QComboBox()
        self.backend_combo.addItem("PICam Demo 相机", "demo")
        self.backend_combo.addItem("PICam 真实相机", "picam")
        self.backend_combo.addItem("无 SDK 模拟", "mock")
        layout.addRow("后端类型:", self.backend_combo)

        # 曝光
        self.exposure_spin = QtWidgets.QDoubleSpinBox()
        self.exposure_spin.setRange(0.001, 3600)
        self.exposure_spin.setDecimals(4)
        self.exposure_spin.setValue(0.1)
        self.exposure_spin.setSuffix(" s")
        layout.addRow("曝光时间:", self.exposure_spin)

        # 温度
        self.temp_spin = QtWidgets.QDoubleSpinBox()
        self.temp_spin.setRange(-100, 50)
        self.temp_spin.setDecimals(1)
        self.temp_spin.setValue(-25)
        self.temp_spin.setSuffix(" °C")
        layout.addRow("目标温度:", self.temp_spin)

        # ROI
        roi_group = QtWidgets.QGroupBox("ROI")
        roi_layout = QtWidgets.QFormLayout(roi_group)
        self.roi_x = QtWidgets.QSpinBox()
        self.roi_x.setRange(0, 4096)
        self.roi_x.setValue(0)
        self.roi_w = QtWidgets.QSpinBox()
        self.roi_w.setRange(1, 4096)
        self.roi_w.setValue(1024)
        self.roi_y = QtWidgets.QSpinBox()
        self.roi_y.setRange(0, 4096)
        self.roi_y.setValue(0)
        self.roi_h = QtWidgets.QSpinBox()
        self.roi_h.setRange(1, 4096)
        self.roi_h.setValue(256)
        roi_layout.addRow("X / 宽度:", self._hbox(self.roi_x, self.roi_w))
        roi_layout.addRow("Y / 高度:", self._hbox(self.roi_y, self.roi_h))
        layout.addRow(roi_group)

        # 按钮
        self.connect_btn = QtWidgets.QPushButton("连接")
        self.connect_btn.clicked.connect(self.on_connect)
        layout.addRow(self.connect_btn)

        self.apply_btn = QtWidgets.QPushButton("应用参数")
        self.apply_btn.clicked.connect(self.on_apply_parameters)
        self.apply_btn.setEnabled(False)
        layout.addRow(self.apply_btn)

        self.single_btn = QtWidgets.QPushButton("单帧采集")
        self.single_btn.clicked.connect(self.on_single_acquire)
        self.single_btn.setEnabled(False)
        layout.addRow(self.single_btn)

        self.continuous_btn = QtWidgets.QPushButton("连续采集")
        self.continuous_btn.setCheckable(True)
        self.continuous_btn.clicked.connect(self.on_continuous_toggle)
        self.continuous_btn.setEnabled(False)
        layout.addRow(self.continuous_btn)

        self.save_btn = QtWidgets.QPushButton("保存当前光谱 (CSV)")
        self.save_btn.clicked.connect(self.on_save_csv)
        self.save_btn.setEnabled(False)
        layout.addRow(self.save_btn)

        # IsoPlane 单色仪
        self.isoplane_btn = QtWidgets.QPushButton("连接 IsoPlane 单色仪")
        self.isoplane_btn.clicked.connect(self.on_connect_isoplane)
        layout.addRow(self.isoplane_btn)

        self.isoplane_info_label = QtWidgets.QLabel("-")
        self.isoplane_info_label.setWordWrap(True)
        layout.addRow("单色仪信息:", self.isoplane_info_label)

        # 诊断
        self.diagnose_btn = QtWidgets.QPushButton("运行环境诊断")
        self.diagnose_btn.clicked.connect(self.on_diagnose)
        layout.addRow(self.diagnose_btn)

        # LightField 风格功能页签
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._build_background_panel(), "背景")
        self.tabs.addTab(self._build_stats_panel(), "统计")
        self.tabs.addTab(self._build_recipe_panel(), "Recipe")
        self.tabs.addTab(self._build_autosave_panel(), "自动保存")
        layout.addRow(self.tabs)

        # 状态
        self.status_label = QtWidgets.QLabel("未连接")
        layout.addRow("状态:", self.status_label)

        self.info_label = QtWidgets.QLabel("-")
        self.info_label.setWordWrap(True)
        layout.addRow("设备信息:", self.info_label)

        return group

    def _hbox(self, *widgets) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        for w in widgets:
            layout.addWidget(w)
        return container

    def _build_menu(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("文件")
        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    # ------------------------------------------------------------------
    # LightField 风格功能面板
    # ------------------------------------------------------------------
    def _build_background_panel(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(widget)

        self.bg_capture_btn = QtWidgets.QPushButton("采集暗背景")
        self.bg_capture_btn.clicked.connect(self.on_capture_dark)
        self.bg_capture_btn.setEnabled(False)
        layout.addRow(self.bg_capture_btn)

        self.bg_load_btn = QtWidgets.QPushButton("加载暗背景 (CSV)")
        self.bg_load_btn.clicked.connect(self.on_load_dark)
        layout.addRow(self.bg_load_btn)

        self.bg_list_combo = QtWidgets.QComboBox()
        self.bg_list_combo.currentTextChanged.connect(self.on_dark_selected)
        layout.addRow("背景帧:", self.bg_list_combo)

        self.bg_enable_check = QtWidgets.QCheckBox("启用暗背景扣除")
        self.bg_enable_check.stateChanged.connect(self.on_bg_enable_changed)
        layout.addRow(self.bg_enable_check)

        self.bg_info_label = QtWidgets.QLabel("无背景帧")
        self.bg_info_label.setWordWrap(True)
        layout.addRow("信息:", self.bg_info_label)

        return widget

    def _build_stats_panel(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(widget)

        self.stats_peak_label = QtWidgets.QLabel("-")
        self.stats_fwhm_label = QtWidgets.QLabel("-")
        self.stats_centroid_label = QtWidgets.QLabel("-")
        self.stats_integral_label = QtWidgets.QLabel("-")
        self.stats_snr_label = QtWidgets.QLabel("-")
        self.stats_num_peaks_label = QtWidgets.QLabel("-")

        layout.addRow("峰位 / 峰强:", self.stats_peak_label)
        layout.addRow("FWHM:", self.stats_fwhm_label)
        layout.addRow("质心:", self.stats_centroid_label)
        layout.addRow("积分:", self.stats_integral_label)
        layout.addRow("SNR:", self.stats_snr_label)
        layout.addRow("检测峰数:", self.stats_num_peaks_label)

        self.stats_multipeak_check = QtWidgets.QCheckBox("多峰检测")
        layout.addRow(self.stats_multipeak_check)

        self.annotate_check = QtWidgets.QCheckBox("图上标注峰值")
        self.annotate_check.setChecked(True)
        self.annotate_check.stateChanged.connect(self.on_annotate_changed)
        layout.addRow(self.annotate_check)

        self.history_check = QtWidgets.QCheckBox("叠加历史轨迹")
        self.history_check.stateChanged.connect(self.on_history_changed)
        layout.addRow(self.history_check)

        self.log_y_check = QtWidgets.QCheckBox("Y 轴对数")
        self.log_y_check.stateChanged.connect(self.on_log_y_changed)
        layout.addRow(self.log_y_check)

        return widget

    def _build_recipe_panel(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        self.recipe_list = QtWidgets.QListWidget()
        layout.addWidget(self.recipe_list)

        btn_layout = QtWidgets.QHBoxLayout()
        self.recipe_add_acquire_btn = QtWidgets.QPushButton("+ 采集")
        self.recipe_add_acquire_btn.clicked.connect(lambda: self.on_recipe_add(RecipeAction.ACQUIRE))
        self.recipe_add_wait_btn = QtWidgets.QPushButton("+ 等待")
        self.recipe_add_wait_btn.clicked.connect(lambda: self.on_recipe_add(RecipeAction.WAIT))
        self.recipe_remove_btn = QtWidgets.QPushButton("- 删除")
        self.recipe_remove_btn.clicked.connect(self.on_recipe_remove)
        btn_layout.addWidget(self.recipe_add_acquire_btn)
        btn_layout.addWidget(self.recipe_add_wait_btn)
        btn_layout.addWidget(self.recipe_remove_btn)
        layout.addLayout(btn_layout)

        run_layout = QtWidgets.QHBoxLayout()
        self.recipe_run_btn = QtWidgets.QPushButton("运行 Recipe")
        self.recipe_run_btn.clicked.connect(self.on_recipe_run)
        self.recipe_pause_btn = QtWidgets.QPushButton("暂停")
        self.recipe_pause_btn.clicked.connect(self.on_recipe_pause)
        self.recipe_stop_btn = QtWidgets.QPushButton("停止")
        self.recipe_stop_btn.clicked.connect(self.on_recipe_stop)
        run_layout.addWidget(self.recipe_run_btn)
        run_layout.addWidget(self.recipe_pause_btn)
        run_layout.addWidget(self.recipe_stop_btn)
        layout.addLayout(run_layout)

        self.recipe_status_label = QtWidgets.QLabel("就绪")
        layout.addWidget(self.recipe_status_label)

        return widget

    def _build_autosave_panel(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(widget)

        self.autosave_check = QtWidgets.QCheckBox("启用自动保存")
        self.autosave_check.stateChanged.connect(self.on_autosave_changed)
        layout.addRow(self.autosave_check)

        self.autosave_dir_btn = QtWidgets.QPushButton("选择目录")
        self.autosave_dir_btn.clicked.connect(self.on_select_autosave_dir)
        self.autosave_dir_label = QtWidgets.QLabel(str(self.auto_save_dir))
        layout.addRow("目录:", self._hbox(self.autosave_dir_btn, self.autosave_dir_label))

        self.autosave_template_edit = QtWidgets.QLineEdit(self.auto_save_template)
        self.autosave_template_edit.textChanged.connect(self.on_autosave_template_changed)
        layout.addRow("文件名模板:", self.autosave_template_edit)

        return widget

    # ------------------------------------------------------------------
    # 日志
    # ------------------------------------------------------------------
    def _log(self, message: str) -> None:
        timestamp = QtCore.QDateTime.currentDateTime().toString("hh:mm:ss.zzz")
        self.log_text.append(f"[{timestamp}] {message}")

    # ------------------------------------------------------------------
    # 事件处理
    # ------------------------------------------------------------------
    def on_connect(self) -> None:
        if self.backend is not None:
            self._disconnect()
            return

        backend_type = self.backend_combo.currentData()
        try:
            if backend_type == "demo":
                self.backend = DemoCamera()
            elif backend_type == "picam":
                self.backend = PICamCamera()
            else:
                self.backend = MockSpectrometerBackend()

            ok = self.backend.connect()
            if not ok:
                raise SpectrometerError("连接失败")

            self._log(f"已连接: {self.backend.name}")
            self.status_label.setText("已连接")
            self.connect_btn.setText("断开")
            self.apply_btn.setEnabled(True)
            self.single_btn.setEnabled(True)
            self.continuous_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            self.bg_capture_btn.setEnabled(True)

            if isinstance(self.backend, PICamCamera):
                info = self.backend.get_camera_info()
                self.info_label.setText(
                    f"Model: {info.get('model', '-')}\n"
                    f"Serial: {info.get('serial_number', '-')}\n"
                    f"Sensor: {info.get('sensor_name', '-')}\n"
                    f"Demo: {info.get('demo', False)}"
                )
            else:
                self.info_label.setText("模拟后端")

            self.on_apply_parameters()
        except Exception as e:
            self._log(f"连接错误: {e}")
            self.backend = None

    def _disconnect(self) -> None:
        self._stop_continuous()
        if self.backend:
            try:
                self.backend.disconnect()
            except Exception as e:
                self._log(f"断开时出错: {e}")
            self.backend = None

        self.status_label.setText("未连接")
        self.connect_btn.setText("连接")
        self.apply_btn.setEnabled(False)
        self.single_btn.setEnabled(False)
        self.continuous_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.bg_capture_btn.setEnabled(False)
        self.info_label.setText("-")
        self._log("已断开")

    def on_apply_parameters(self) -> None:
        if self.backend is None:
            return
        try:
            roi = ROI(
                x=self.roi_x.value(),
                width=self.roi_w.value(),
                y=self.roi_y.value(),
                height=self.roi_h.value(),
            )
            self.backend.set_roi(roi)
            self.backend.set_exposure(self.exposure_spin.value())
            if self.backend.supports_temperature():
                self.backend.set_sensor_temperature(self.temp_spin.value())
            self._log(
                f"参数已应用: exposure={self.backend.get_exposure():.4f}s, "
                f"ROI=({roi.x},{roi.y},{roi.width},{roi.height})"
            )
        except Exception as e:
            self._log(f"应用参数失败: {e}")

    def on_single_acquire(self) -> None:
        if self.backend is None:
            return
        self.cycle_index += 1
        worker = AcquireWorker(self.backend, self.cycle_index)
        worker.signals.result_ready.connect(self._on_result_ready)
        worker.signals.error.connect(self._on_error)
        worker.signals.log.connect(self._log)
        self.pool.start(worker)

    def on_continuous_toggle(self) -> None:
        if self.continuous_btn.isChecked():
            self.continuous_btn.setText("停止连续采集")
            self.continuous_timer = QtCore.QTimer(self)
            self.continuous_timer.timeout.connect(self.on_single_acquire)
            self.continuous_timer.start(200)  # 200 ms 间隔，实际受曝光限制
        else:
            self._stop_continuous()

    def _stop_continuous(self) -> None:
        if self.continuous_timer is not None:
            self.continuous_timer.stop()
            self.continuous_timer = None
        if self.continuous_btn is not None:
            self.continuous_btn.setChecked(False)
            self.continuous_btn.setText("连续采集")

    def _on_result_ready(self, result: SpectrometerResult) -> None:
        # 背景校正
        if self.bg_correction_enabled and self.current_dark_name is not None:
            try:
                result = correct_spectrometer_result(
                    result, library=self.bg_library, dark_name=self.current_dark_name
                )
                self._log(f"已应用暗背景扣除: {self.current_dark_name}")
            except Exception as e:
                self._log(f"背景扣除失败: {e}")

        self.last_result = result

        # 历史轨迹
        if self.history_enabled:
            self.plot.add_history_trace(result.raw_y, result.wavelength)

        title = f"Spectrum #{result.index}"
        self.plot.update_plot(result.raw_y, result.wavelength, title=title)

        # 统计
        self._update_stats(result)

        # 峰值标注
        self.plot.clear_peak_annotations()
        if self.annotate_enabled:
            self._annotate_peak(result)

        # 自动保存
        if self.auto_save_enabled:
            self._auto_save_result(result)

    def _on_error(self, message: str) -> None:
        self._log(f"采集错误: {message}")

    def _update_stats(self, result: SpectrometerResult) -> None:
        try:
            stats = compute_spectrum_stats(
                result.wavelength,
                result.raw_y,
                find_peaks=self.stats_multipeak_check.isChecked(),
            )
            self.stats_peak_label.setText(
                f"{stats.peak_x:.3f} nm / {stats.peak_y:.2f}"
                if stats.peak_x is not None and stats.peak_y is not None
                else "-"
            )
            self.stats_fwhm_label.setText(
                f"{stats.fwhm_x:.3f} nm"
                if stats.fwhm_x is not None
                else "-"
            )
            self.stats_centroid_label.setText(
                f"{stats.centroid_x:.3f} nm"
                if stats.centroid_x is not None
                else "-"
            )
            self.stats_integral_label.setText(
                f"{stats.integral:.2f}"
                if stats.integral is not None
                else "-"
            )
            self.stats_snr_label.setText(
                f"{stats.snr:.2f}"
                if stats.snr is not None
                else "-"
            )
            self.stats_num_peaks_label.setText(str(stats.num_peaks))
        except Exception as e:
            self._log(f"统计计算失败: {e}")

    def _annotate_peak(self, result: SpectrometerResult) -> None:
        try:
            stats = compute_spectrum_stats(result.wavelength, result.raw_y)
            if stats.peak_x is not None and stats.peak_y is not None:
                self.plot.annotate_peak(
                    stats.peak_x,
                    stats.peak_y,
                    f"Peak\n{stats.peak_x:.2f} nm",
                )
            if (
                stats.fwhm_left_x is not None
                and stats.fwhm_right_x is not None
                and stats.fwhm_y is not None
            ):
                self.plot.annotate_fwhm(
                    stats.fwhm_left_x,
                    stats.fwhm_right_x,
                    stats.fwhm_y,
                )
        except Exception as e:
            self._log(f"峰值标注失败: {e}")

    def _auto_save_result(self, result: SpectrometerResult) -> None:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.auto_save_template.format(
                index=result.index,
                timestamp=timestamp,
            )
            path = self.auto_save_dir / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            x = result.wavelength
            y = result.raw_y
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["wavelength_nm", "intensity"])
                for i in range(len(y)):
                    writer.writerow(
                        [float(x[i]) if x is not None else i, float(y[i])]
                    )
            self._log(f"自动保存: {path}")
        except Exception as e:
            self._log(f"自动保存失败: {e}")

    def on_save_csv(self) -> None:
        if self.last_result is None:
            self._log("没有可保存的数据")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "保存光谱", f"spectrum_{self.last_result.index:04d}.csv", "CSV (*.csv)"
        )
        if not path:
            return
        try:
            x = self.last_result.wavelength
            y = self.last_result.raw_y
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["wavelength_nm", "intensity"])
                for i in range(len(y)):
                    writer.writerow(
                        [float(x[i]) if x is not None else i, float(y[i])]
                    )
            self._log(f"已保存: {path}")
        except Exception as e:
            self._log(f"保存失败: {e}")

    def on_connect_isoplane(self) -> None:
        """测试连接 IsoPlane 单色仪。"""
        if IsoPlaneBackend is None:
            self._log("IsoPlane 后端不可用：可能是 32-bit DLL 与 64-bit Python 不匹配")
            return
        try:
            mono = IsoPlaneBackend()
            mono.connect()
            info = mono.get_info()
            self.isoplane_info_label.setText(
                f"Model: {info.get('model', '-')}\n"
                f"Serial: {info.get('serial', '-')}\n"
                f"Wavelength: {info.get('wavelength_nm', '-')} nm\n"
                f"Grating: {info.get('grating', '-')}"
            )
            self._log(
                f"IsoPlane 已连接: {info.get('model', '-')}, "
                f"波长={info.get('wavelength_nm', '-')} nm"
            )
            mono.disconnect()
        except Exception as e:
            self._log(f"IsoPlane 连接错误: {e}")
            self.isoplane_info_label.setText(f"连接失败: {e}")

    def on_diagnose(self) -> None:
        """运行环境诊断。"""
        import subprocess
        import sys

        script = Path(__file__).resolve().parent.parent / "diagnose_pi_environment.py"
        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            for line in result.stdout.splitlines():
                self._log(line)
            if result.returncode != 0 and result.stderr:
                self._log(f"诊断脚本错误: {result.stderr}")
        except Exception as e:
            self._log(f"运行诊断失败: {e}")

    # ------------------------------------------------------------------
    # 背景/暗场处理事件
    # ------------------------------------------------------------------
    def on_capture_dark(self) -> None:
        if self.backend is None:
            self._log("未连接设备，无法采集暗背景")
            return
        try:
            self._log("正在采集暗背景...")
            result = self.backend.acquire(num_frames=1)
            name = f"dark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            frame = BackgroundFrame(name=name, y=result.raw_y, kind="dark")
            self.bg_library.add(frame)
            self.current_dark_name = name
            self._refresh_bg_list()
            self._log(f"暗背景已采集并保存为: {name}")
        except Exception as e:
            self._log(f"采集暗背景失败: {e}")

    def on_load_dark(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "加载暗背景 CSV", "", "CSV (*.csv)"
        )
        if not path:
            return
        try:
            y_values = []
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                intensity_col = 1 if header and len(header) > 1 else 0
                for row in reader:
                    if len(row) > intensity_col:
                        y_values.append(float(row[intensity_col]))
            name = f"dark_loaded_{Path(path).stem}"
            frame = BackgroundFrame(name=name, y=np.array(y_values), kind="dark")
            self.bg_library.add(frame)
            self.current_dark_name = name
            self._refresh_bg_list()
            self._log(f"已加载暗背景: {name} ({len(y_values)} 点)")
        except Exception as e:
            self._log(f"加载暗背景失败: {e}")

    def on_dark_selected(self, name: str) -> None:
        if name:
            self.current_dark_name = name
            frame = self.bg_library.get(name)
            info = f"{frame.kind}: {len(frame.y)} 点" if frame else ""
            self.bg_info_label.setText(info)

    def on_bg_enable_changed(self, state: int) -> None:
        self.bg_correction_enabled = state == QtCore.Qt.Checked
        self._log(f"暗背景扣除: {'启用' if self.bg_correction_enabled else '禁用'}")

    def _refresh_bg_list(self) -> None:
        self.bg_list_combo.clear()
        for name in self.bg_library.list():
            self.bg_list_combo.addItem(name)
        if self.current_dark_name:
            self.bg_list_combo.setCurrentText(self.current_dark_name)

    # ------------------------------------------------------------------
    # 统计与绘图选项事件
    # ------------------------------------------------------------------
    def on_annotate_changed(self, state: int) -> None:
        self.annotate_enabled = state == QtCore.Qt.Checked
        if not self.annotate_enabled:
            self.plot.clear_peak_annotations()
        elif self.last_result is not None:
            self._annotate_peak(self.last_result)

    def on_history_changed(self, state: int) -> None:
        self.history_enabled = state == QtCore.Qt.Checked
        if not self.history_enabled:
            self.plot.clear_history()

    def on_log_y_changed(self, state: int) -> None:
        self.plot.set_log_mode(state == QtCore.Qt.Checked)

    # ------------------------------------------------------------------
    # 自动保存事件
    # ------------------------------------------------------------------
    def on_autosave_changed(self, state: int) -> None:
        self.auto_save_enabled = state == QtCore.Qt.Checked
        self._log(f"自动保存: {'启用' if self.auto_save_enabled else '禁用'}")

    def on_select_autosave_dir(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "选择自动保存目录", str(self.auto_save_dir)
        )
        if path:
            self.auto_save_dir = Path(path)
            self.autosave_dir_label.setText(path)

    def on_autosave_template_changed(self, text: str) -> None:
        self.auto_save_template = text

    # ------------------------------------------------------------------
    # Recipe 事件
    # ------------------------------------------------------------------
    def on_recipe_add(self, action: RecipeAction) -> None:
        if action == RecipeAction.ACQUIRE:
            step = RecipeStep(
                action=RecipeAction.ACQUIRE,
                params={"num_frames": 1},
                repeats=1,
                delay_s=0.0,
            )
        elif action == RecipeAction.WAIT:
            step = RecipeStep(
                action=RecipeAction.WAIT,
                params={"seconds": 1.0},
                repeats=1,
                delay_s=0.0,
            )
        else:
            return
        self.recipe.add_step(step)
        self._refresh_recipe_list()

    def on_recipe_remove(self) -> None:
        row = self.recipe_list.currentRow()
        if row >= 0:
            self.recipe.remove_step(row)
            self._refresh_recipe_list()

    def _refresh_recipe_list(self) -> None:
        self.recipe_list.clear()
        for i, step in enumerate(self.recipe.steps):
            text = f"{i}: {step.action.value} {step.params} x{step.repeats}"
            self.recipe_list.addItem(text)

    def on_recipe_run(self) -> None:
        if self.backend is None:
            self._log("未连接设备，无法运行 Recipe")
            return
        if self.recipe_thread is not None and self.recipe_thread.is_alive():
            self._log("Recipe 已在运行")
            return

        try:
            self.recipe_runner = self.recipe.create_runner(
                backend=self.backend,
                on_step_start=self._on_recipe_step_start,
                on_step_done=self._on_recipe_step_done,
                on_log=self._log,
                on_error=self._on_recipe_error,
            )
        except ValueError as e:
            self._log(f"Recipe 验证失败: {e}")
            return

        self.recipe_thread = threading.Thread(target=self._run_recipe, daemon=True)
        self.recipe_thread.start()
        self.recipe_status_label.setText("运行中...")

    def _run_recipe(self) -> None:
        if self.recipe_runner is None:
            return
        results = self.recipe_runner.run()
        self.recipe_status_label.setText(f"完成，共采集 {len(results)} 帧")
        self._log("Recipe 执行结束")

    def _on_recipe_step_start(self, index: int, step: RecipeStep) -> None:
        self.recipe_status_label.setText(f"步骤 {index}: {step.action.value}")

    def _on_recipe_step_done(self, index: int, step: RecipeStep, result) -> None:
        if isinstance(result, SpectrometerResult):
            # 复用主界面的结果处理流程
            self._on_result_ready(result)

    def _on_recipe_error(self, message: str) -> None:
        self.recipe_status_label.setText(f"错误: {message}")

    def on_recipe_pause(self) -> None:
        if self.recipe_runner is None:
            return
        if self.recipe_runner.is_paused():
            self.recipe_runner.resume()
            self.recipe_pause_btn.setText("暂停")
        else:
            self.recipe_runner.pause()
            self.recipe_pause_btn.setText("继续")

    def on_recipe_stop(self) -> None:
        if self.recipe_runner is not None:
            self.recipe_runner.stop()
        self.recipe_status_label.setText("已停止")

    # ------------------------------------------------------------------
    # 窗口关闭
    # ------------------------------------------------------------------
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._stop_continuous()
        if self.recipe_runner is not None:
            self.recipe_runner.stop()
        if self.backend is not None:
            self.backend.disconnect()
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
