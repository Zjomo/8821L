"""PyQt 主窗口：PI 光谱仪测试 UI。"""

from __future__ import annotations

import csv
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from pi_spectrometer.ui.qt_compat import (
    QtCore, QtGui, QtWidgets, Signal, Slot, AlignCenter, QAction
)

from pi_spectrometer.core.base import SpectrometerBackend
from pi_spectrometer.core.exceptions import SpectrometerError
from pi_spectrometer.core.types import ROI, SpectrometerResult
from pi_spectrometer.picam.camera import PICamCamera
from pi_spectrometer.picam.demo import DemoCamera, MockSpectrometerBackend
from pi_spectrometer.ui.plot_widget import PlotWidget


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
        self.last_result = result
        title = f"Spectrum #{result.index}"
        self.plot.update_plot(result.raw_y, result.wavelength, title=title)

    def _on_error(self, message: str) -> None:
        self._log(f"采集错误: {message}")

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

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._stop_continuous()
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
