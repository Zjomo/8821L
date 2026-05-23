from __future__ import annotations

import time
from typing import Callable, List, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import SpotZoom


class PicomotorAxisWidget(QWidget):
    def __init__(
        self,
        axis: int,
        controller_getter: Callable[[], Optional[SpotZoom.PicoMotor8742Controller]],
        log_callback: Callable[[str], None],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.axis = int(axis)
        self.controller_getter = controller_getter
        self.log_callback = log_callback
        self._build_ui()

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(500)
        self._poll_timer.timeout.connect(self.refresh_position)
        self._poll_timer.start()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        title = QLabel(f"Axis {self.axis}")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        root.addWidget(title)

        pos_group = QGroupBox("Position")
        pos_layout = QHBoxLayout(pos_group)
        self.position_label = QLabel("-")
        self.position_label.setAlignment(Qt.AlignCenter)
        self.position_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        btn_refresh_pos = QPushButton("Refresh")
        btn_refresh_pos.clicked.connect(self.refresh_position)
        pos_layout.addWidget(self.position_label, 1)
        pos_layout.addWidget(btn_refresh_pos)
        root.addWidget(pos_group)

        vel_group = QGroupBox("Velocity")
        vel_layout = QGridLayout(vel_group)
        self.speed_spin = QSpinBox()
        self.speed_spin.setRange(1, 50000)
        self.speed_spin.setValue(200)
        self.accel_spin = QSpinBox()
        self.accel_spin.setRange(1, 100000)
        self.accel_spin.setValue(2000)
        btn_read_vel = QPushButton("Read")
        btn_read_vel.clicked.connect(self.read_velocity)
        btn_set_vel = QPushButton("Apply")
        btn_set_vel.clicked.connect(self.set_velocity)
        vel_layout.addWidget(QLabel("Speed"), 0, 0)
        vel_layout.addWidget(self.speed_spin, 0, 1)
        vel_layout.addWidget(QLabel("Accel"), 1, 0)
        vel_layout.addWidget(self.accel_spin, 1, 1)
        vel_layout.addWidget(btn_read_vel, 2, 0)
        vel_layout.addWidget(btn_set_vel, 2, 1)
        root.addWidget(vel_group)

        rel_group = QGroupBox("Relative Move")
        rel_layout = QGridLayout(rel_group)
        self.rel_steps_spin = QSpinBox()
        self.rel_steps_spin.setRange(1, 1000000)
        self.rel_steps_spin.setValue(50)
        btn_rel_pos = QPushButton("+")
        btn_rel_pos.clicked.connect(lambda: self.move_relative(+self.rel_steps_spin.value()))
        btn_rel_neg = QPushButton("-")
        btn_rel_neg.clicked.connect(lambda: self.move_relative(-self.rel_steps_spin.value()))
        rel_layout.addWidget(QLabel("Steps"), 0, 0)
        rel_layout.addWidget(self.rel_steps_spin, 0, 1)
        rel_layout.addWidget(btn_rel_pos, 1, 0)
        rel_layout.addWidget(btn_rel_neg, 1, 1)
        root.addWidget(rel_group)

        abs_group = QGroupBox("Absolute Move")
        abs_layout = QGridLayout(abs_group)
        self.abs_pos_spin = QSpinBox()
        self.abs_pos_spin.setRange(-2000000000, 2000000000)
        self.abs_pos_spin.setValue(0)
        btn_abs = QPushButton("Move To")
        btn_abs.clicked.connect(self.move_absolute)
        abs_layout.addWidget(QLabel("Target"), 0, 0)
        abs_layout.addWidget(self.abs_pos_spin, 0, 1)
        abs_layout.addWidget(btn_abs, 1, 0, 1, 2)
        root.addWidget(abs_group)

        jog_group = QGroupBox("Jog")
        jog_layout = QGridLayout(jog_group)
        self.jog_duration_spin = QDoubleSpinBox()
        self.jog_duration_spin.setRange(0.1, 10.0)
        self.jog_duration_spin.setDecimals(2)
        self.jog_duration_spin.setValue(1.0)
        btn_jog_plus_hold = QPushButton("Jog + (hold)")
        btn_jog_plus_hold.pressed.connect(lambda: self.start_jog("+"))
        btn_jog_plus_hold.released.connect(self.stop_jog)
        btn_jog_minus_hold = QPushButton("Jog - (hold)")
        btn_jog_minus_hold.pressed.connect(lambda: self.start_jog("-"))
        btn_jog_minus_hold.released.connect(self.stop_jog)
        btn_jog_plus_1s = QPushButton("Jog + for t")
        btn_jog_plus_1s.clicked.connect(lambda: self.jog_for("+"))
        btn_jog_minus_1s = QPushButton("Jog - for t")
        btn_jog_minus_1s.clicked.connect(lambda: self.jog_for("-"))
        btn_stop = QPushButton("Stop")
        btn_stop.clicked.connect(self.stop_axis)
        jog_layout.addWidget(QLabel("Duration(s)"), 0, 0)
        jog_layout.addWidget(self.jog_duration_spin, 0, 1)
        jog_layout.addWidget(btn_jog_plus_hold, 1, 0)
        jog_layout.addWidget(btn_jog_minus_hold, 1, 1)
        jog_layout.addWidget(btn_jog_plus_1s, 2, 0)
        jog_layout.addWidget(btn_jog_minus_1s, 2, 1)
        jog_layout.addWidget(btn_stop, 3, 0, 1, 2)
        root.addWidget(jog_group)

        op_row = QHBoxLayout()
        btn_zero = QPushButton("Set Zero Here")
        btn_zero.clicked.connect(self.set_zero)
        btn_estop = QPushButton("Emergency Stop")
        btn_estop.clicked.connect(self.emergency_stop_axis)
        btn_estop.setStyleSheet("background:#b91c1c;color:#fff;")
        op_row.addWidget(btn_zero)
        op_row.addWidget(btn_estop)
        root.addLayout(op_row)
        root.addStretch(1)

    def _controller(self) -> SpotZoom.PicoMotor8742Controller:
        controller = self.controller_getter()
        if controller is None:
            raise RuntimeError("Controller not connected")
        return controller

    def _run(self, op_name: str, fn) -> None:
        try:
            fn()
            self.refresh_position()
        except Exception as exc:
            self.log_callback(f"[Axis {self.axis}] {op_name} failed: {exc}")
            QMessageBox.critical(self, "Axis Error", f"{op_name} failed: {exc}")

    def refresh_position(self) -> None:
        try:
            pos = self._controller().get_pos(self.axis)
            self.position_label.setText(str(pos))
        except Exception:
            self.position_label.setText("-")

    def read_velocity(self) -> None:
        def _impl() -> None:
            speed, accel = self._controller().get_vel(self.axis)
            self.speed_spin.setValue(int(speed))
            self.accel_spin.setValue(int(accel))
            self.log_callback(f"[Axis {self.axis}] velocity: speed={speed}, accel={accel}")

        self._run("read velocity", _impl)

    def set_velocity(self) -> None:
        self._run(
            "set velocity",
            lambda: self._controller().set_vel(
                axis=self.axis,
                speed=self.speed_spin.value(),
                accel=self.accel_spin.value(),
            ),
        )

    def move_relative(self, steps: int) -> None:
        self._run(
            f"move rel {steps}",
            lambda: self._controller().move_rel(axis=self.axis, steps=int(steps), wait=True),
        )

    def move_absolute(self) -> None:
        target = self.abs_pos_spin.value()
        self._run(
            f"move abs {target}",
            lambda: self._controller().move_abs(axis=self.axis, position=int(target), wait=True),
        )

    def start_jog(self, direction: str) -> None:
        self._run(
            f"start jog {direction}",
            lambda: self._controller().jog(axis=self.axis, direction=direction),
        )

    def stop_jog(self) -> None:
        self.stop_axis()

    def jog_for(self, direction: str) -> None:
        duration = float(self.jog_duration_spin.value())
        self._run(
            f"jog_for {direction} {duration:.2f}s",
            lambda: self._controller().jog_for(axis=self.axis, direction=direction, duration_s=duration),
        )

    def stop_axis(self) -> None:
        self._run("stop axis", lambda: self._controller().stop(axis=self.axis, immediate=False))

    def emergency_stop_axis(self) -> None:
        self._run("emergency stop axis", lambda: self._controller().stop(axis=self.axis, immediate=True))

    def set_zero(self) -> None:
        self._run("set zero", lambda: self._controller().set_zero_here(axis=self.axis, position=0))


class PicomotorDriverPanel(QWidget):
    def __init__(self, log_callback: Optional[Callable[[str], None]] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.controller: Optional[SpotZoom.PicoMotor8742Controller] = None
        self.axis_widgets: List[PicomotorAxisWidget] = []
        self.log_callback = log_callback
        self._build_ui()
        self.refresh_usb_count()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        conn_group = QGroupBox("Picomotor 8742/8743 Driver Console")
        conn_layout = QFormLayout(conn_group)

        self.usb_count_label = QLabel("-")
        btn_refresh_usb = QPushButton("Refresh USB Count")
        btn_refresh_usb.clicked.connect(self.refresh_usb_count)
        usb_row = QHBoxLayout()
        usb_row.addWidget(self.usb_count_label)
        usb_row.addWidget(btn_refresh_usb)
        usb_row.addStretch(1)
        usb_row_wrap = QWidget()
        usb_row_wrap.setLayout(usb_row)
        conn_layout.addRow("USB devices", usb_row_wrap)

        self.conn_spin = QSpinBox()
        self.conn_spin.setRange(0, 32)
        self.conn_spin.setValue(0)
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["auto", "pyusb", "serial"])
        self.timeout_spin = QDoubleSpinBox()
        self.timeout_spin.setRange(0.1, 60.0)
        self.timeout_spin.setDecimals(1)
        self.timeout_spin.setValue(5.0)
        self.multiaddr_check = QCheckBox("multiaddr")
        self.scan_check = QCheckBox("scan")
        self.scan_check.setChecked(True)
        conn_layout.addRow("Controller index", self.conn_spin)
        conn_layout.addRow("Backend", self.backend_combo)
        conn_layout.addRow("Timeout(s)", self.timeout_spin)
        conn_layout.addRow(self.multiaddr_check, self.scan_check)

        btn_row = QHBoxLayout()
        self.btn_connect = QPushButton("Connect")
        self.btn_connect.clicked.connect(self.connect_device)
        self.btn_disconnect = QPushButton("Disconnect")
        self.btn_disconnect.clicked.connect(self.disconnect_device)
        self.btn_disconnect.setEnabled(False)
        self.btn_stop_all = QPushButton("Stop All")
        self.btn_stop_all.clicked.connect(self.stop_all_axes)
        self.btn_stop_all.setEnabled(False)
        self.btn_emergency_stop = QPushButton("Emergency Stop All")
        self.btn_emergency_stop.clicked.connect(self.emergency_stop_all_axes)
        self.btn_emergency_stop.setEnabled(False)
        self.btn_emergency_stop.setStyleSheet("background:#b91c1c;color:#fff;")
        btn_row.addWidget(self.btn_connect)
        btn_row.addWidget(self.btn_disconnect)
        btn_row.addWidget(self.btn_stop_all)
        btn_row.addWidget(self.btn_emergency_stop)
        btn_row.addStretch(1)
        btn_row_wrap = QWidget()
        btn_row_wrap.setLayout(btn_row)
        conn_layout.addRow(btn_row_wrap)

        self.status_label = QLabel("Disconnected")
        conn_layout.addRow("Status", self.status_label)

        self.device_id_label = QLabel("-")
        self.axes_label = QLabel("-")
        conn_layout.addRow("Device ID", self.device_id_label)
        conn_layout.addRow("Axes", self.axes_label)
        root.addWidget(conn_group)

        self.axis_tabs = QTabWidget()
        root.addWidget(self.axis_tabs, 1)

        log_group = QGroupBox("Driver Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(170)
        btn_clear_log = QPushButton("Clear Log")
        btn_clear_log.clicked.connect(self.log_text.clear)
        log_layout.addWidget(self.log_text)
        log_layout.addWidget(btn_clear_log)
        root.addWidget(log_group)

    def _log(self, message: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {message}"
        self.log_text.append(line)
        if callable(self.log_callback):
            self.log_callback(f"[Picomotor] {message}")

    def refresh_usb_count(self) -> None:
        try:
            count = SpotZoom.PicoMotor8742Controller.usb_device_count()
            self.usb_count_label.setText(str(count))
        except Exception as exc:
            self.usb_count_label.setText("error")
            self._log(f"refresh usb count failed: {exc}")

    def _controller_getter(self) -> Optional[SpotZoom.PicoMotor8742Controller]:
        return self.controller

    def _set_connected_state(self, connected: bool) -> None:
        self.btn_connect.setEnabled(not connected)
        self.btn_disconnect.setEnabled(connected)
        self.btn_stop_all.setEnabled(connected)
        self.btn_emergency_stop.setEnabled(connected)
        self.conn_spin.setEnabled(not connected)
        self.backend_combo.setEnabled(not connected)
        self.timeout_spin.setEnabled(not connected)
        self.multiaddr_check.setEnabled(not connected)
        self.scan_check.setEnabled(not connected)

    def connect_device(self) -> None:
        if self.controller is not None:
            return
        try:
            controller = SpotZoom.PicoMotor8742Controller(
                conn=int(self.conn_spin.value()),
                backend=self.backend_combo.currentText().strip(),
                timeout=float(self.timeout_spin.value()),
                multiaddr=bool(self.multiaddr_check.isChecked()),
                scan=bool(self.scan_check.isChecked()),
            ).open()
            axes = controller.axes()
            device_id = controller.get_id()
        except Exception as exc:
            QMessageBox.critical(self, "Connect Error", f"Failed to connect: {exc}")
            self._log(f"connect failed: {exc}")
            return

        self.controller = controller
        self.axis_tabs.clear()
        self.axis_widgets = []
        for axis in axes:
            axis_widget = PicomotorAxisWidget(
                axis=axis,
                controller_getter=self._controller_getter,
                log_callback=self._log,
                parent=self.axis_tabs,
            )
            self.axis_tabs.addTab(axis_widget, f"Axis {axis}")
            self.axis_widgets.append(axis_widget)

        self.device_id_label.setText(str(device_id))
        self.axes_label.setText(", ".join(str(axis) for axis in axes))
        self.status_label.setText("Connected")
        self._set_connected_state(True)
        self._log(f"connected: id={device_id}, axes={axes}")

    def disconnect_device(self) -> None:
        if self.controller is None:
            return
        try:
            self.controller.close()
        except Exception as exc:
            self._log(f"disconnect warning: {exc}")
        self.controller = None
        self.axis_tabs.clear()
        self.axis_widgets = []
        self.status_label.setText("Disconnected")
        self.device_id_label.setText("-")
        self.axes_label.setText("-")
        self._set_connected_state(False)
        self._log("disconnected")

    def stop_all_axes(self) -> None:
        if self.controller is None:
            return
        try:
            self.controller.stop(axis="all", immediate=False)
            self._log("stop all axes")
        except Exception as exc:
            QMessageBox.critical(self, "Stop Error", f"Stop all failed: {exc}")
            self._log(f"stop all failed: {exc}")

    def emergency_stop_all_axes(self) -> None:
        if self.controller is None:
            return
        try:
            self.controller.stop(axis="all", immediate=True)
            self._log("emergency stop all axes")
        except Exception as exc:
            QMessageBox.critical(self, "Emergency Stop Error", f"Emergency stop failed: {exc}")
            self._log(f"emergency stop failed: {exc}")

    def shutdown(self) -> None:
        self.disconnect_device()
