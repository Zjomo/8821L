#!/usr/bin/env python3

"""PyQt5 显微镜模拟界面。

布局：
* 左侧 —— 相机视野（十字准星 + 比例尺 + 台位叠加显示）
* 右侧 —— 位移台控制（绝对/相对/点动）、相机参数（曝光/增益/采集）、预设位置
* 底部 —— 操作日志（对应 SQLite 中的 move_log）

多线程：
* :class:`MotionWorker`  以有限速度插值移动位移台（模拟真实电机）
* :class:`CameraWorker`  触发相机并取帧（曝光 sleep 放在线程里，避免卡 UI）
"""

import time
from typing import Dict, Mapping

import numpy as np
from PyQt5.QtCore import QPoint, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


# ----------------------------------------------------------------------
# 样本图层形状选项（与 devices.generate_sample 的形状键对应）
# ----------------------------------------------------------------------
_SHAPE_NAMES = ["椭圆", "矩形", "三角形", "星形", "随机斑块", "自定义"]
_SHAPE_KEYS = {
    "椭圆": "ellipse", "矩形": "rect", "三角形": "triangle",
    "星形": "star", "随机斑块": "blob", "自定义": "custom",
}
_LAYER_ROWS = (("掩码", "mask"), ("衬底", "ground"), ("障碍物", "obstacle"))


# ----------------------------------------------------------------------
# 工作线程
# ----------------------------------------------------------------------
class MotionWorker(QThread):
    """按有限速度把位移台插值移动到目标位置（模拟真实电机运动）。"""

    progressed = pyqtSignal(dict)
    finishedMove = pyqtSignal(bool, str)  # (是否完成未被中断, 移动来源)

    def __init__(self, stage, target: Mapping[str, float], source: str, velocity: float = 2000.0):
        super().__init__()
        self._stage = stage
        self._target = dict(target)
        self._source = source
        self._velocity = velocity  # µm/s
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        start = dict(self._stage.position)
        axes = [a for a in self._target if a in self._stage.axes]
        dist = max((abs(self._target[a] - start[a]) for a in axes), default=0.0)
        total_ms = int(dist / self._velocity * 1000)
        steps = max(2, total_ms // 20)  # 约 50 Hz 刷新
        for i in range(1, steps + 1):
            if self._stop:
                break
            t = i / steps
            interp = {a: start[a] + (self._target[a] - start[a]) * t for a in axes}
            self._stage.move_to(interp, persist=False)
            self.progressed.emit(dict(self._stage.position))
            self.msleep(max(1, total_ms // steps))
        # 无论是否被中断，都把最终真实位置持久化
        self._stage.move_to(dict(self._stage.position), persist=True)
        self.finishedMove.emit(not self._stop, self._source)


class CameraWorker(QThread):
    """相机采集线程：触发 → 取帧，支持连续采集、单张快照与单帧刷新。

    「快照」会把图像写入 SQLite；「刷新」只更新视野显示。
    """

    frameReady = pyqtSignal(object, bool, object)  # (frame, 是否为快照, 帧对应台位)

    def __init__(self, camera):
        super().__init__()
        self._cam = camera
        self._live = False
        self._snap = False
        self._refresh = False
        self._stop = False

    def set_live(self, enabled: bool) -> None:
        self._live = enabled

    def request_snap(self) -> None:
        self._snap = True

    def request_refresh(self) -> None:
        self._refresh = True

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        while not self._stop:
            if self._live or self._snap or self._refresh:
                self._cam.trigger()
                frame = None
                deadline = time.time() + 5.0  # 超时保护
                while frame is None and not self._stop and time.time() < deadline:
                    frame = self._cam._fetch_data()  # noqa: SLF001
                    if frame is None:
                        self.msleep(2)
                if frame is not None:
                    self.frameReady.emit(frame, self._snap, self._cam.last_frame_pos())
                self._snap = False
                self._refresh = False
            else:
                self.msleep(50)


# ----------------------------------------------------------------------
# 视野显示
# ----------------------------------------------------------------------
class ViewWidget(QLabel):
    """相机视野：灰度帧 + 十字准星 + 比例尺 + 台位叠加 + 鼠标拖动移动。

    支持平移预补偿：台位已动而新帧未到时，按「当前台位 − 帧对应台位」
    立即平移缓存帧绘制，拖动/点动时画面以事件节奏顺滑跟随，不跳变。
    """

    dragStarted = pyqtSignal()
    dragMoved = pyqtSignal(dict)  # 拖动后的台位实际位置（已被限位钳制）
    dragFinished = pyqtSignal()

    def __init__(self, pixel_size: float, parent=None):
        super().__init__(parent)
        self._pixel_size = pixel_size
        self.setFixedSize(800, 600)
        self.setStyleSheet("background: #101014; border: 1px solid #333;")
        self._pos = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._stage = None
        self._drag_px0 = None
        self._drag_stage0 = None
        self._frame_pm = None  # 最近一帧（无叠加层）
        self._frame_pos = None  # 最近一帧对应的台位
        self.setCursor(Qt.OpenHandCursor)

    def set_stage(self, stage) -> None:
        self._stage = stage
        self._pos = dict(stage.position)  # 初始化读数，避免首帧前预补偿偏移错误

    # ------------------------------------------------------------------
    # 鼠标拖动：按住左键拖动图像 = 抓取样本移动台位
    # ------------------------------------------------------------------
    def mousePressEvent(self, ev) -> None:
        if ev.button() == Qt.LeftButton and self._stage is not None:
            self._drag_px0 = ev.pos()
            self._drag_stage0 = dict(self._stage.position)
            self.setCursor(Qt.ClosedHandCursor)
            self.dragStarted.emit()
            ev.accept()
        else:
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev) -> None:
        if self._drag_stage0 is not None and self._drag_px0 is not None:
            delta = ev.pos() - self._drag_px0
            # 台位 µm 目标 = 起点 − 像素差 × 像素尺寸（内容跟随光标）
            # 超限量程时由 StageAxis 自动钳制
            self._stage.move_to(
                {
                    "x": self._drag_stage0["x"] - delta.x() * self._pixel_size,
                    "y": self._drag_stage0["y"] - delta.y() * self._pixel_size,
                },
                persist=False,  # 拖动过程不逐事件写库
            )
            pos = dict(self._stage.position)
            self.set_stage_position(pos)
            self.dragMoved.emit(pos)
            ev.accept()
        else:
            super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev) -> None:
        if self._drag_stage0 is not None:
            self._drag_stage0 = None
            self._drag_px0 = None
            self.setCursor(Qt.OpenHandCursor)
            self.dragFinished.emit()
            ev.accept()
        else:
            super().mouseReleaseEvent(ev)

    def set_stage_position(self, pos: Mapping[str, float]) -> None:
        self._pos = dict(pos)
        self.update()  # 立即重绘 → 平移预补偿以事件节奏跟随

    def set_frame(self, frame: np.ndarray, frame_pos: Mapping[str, float] | None = None) -> None:
        h, w = frame.shape
        img = QImage(frame.tobytes(), w, h, w, QImage.Format_Grayscale8)
        self._frame_pm = QPixmap.fromImage(img.copy())
        if frame_pos is not None:
            self._frame_pos = dict(frame_pos)
        self.update()

    def paintEvent(self, ev) -> None:
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor(16, 16, 20))
        if self._frame_pm is not None:
            # 平移预补偿：台位已动而新帧未到时即时平移缓存帧
            dx = dy = 0.0
            if self._frame_pos is not None:
                dx = (self._pos["x"] - self._frame_pos["x"]) / self._pixel_size
                dy = (self._pos["y"] - self._frame_pos["y"]) / self._pixel_size
                # 限制预补偿幅度，帧长时间未更新时不显示误导性内容
                dx = max(-160.0, min(160.0, dx))
                dy = max(-120.0, min(120.0, dy))
            p.drawPixmap(QPoint(int(round(-dx)), int(round(-dy))), self._frame_pm)
        # ---- 屏幕固定叠加层（不随平移移动）----
        p.setRenderHint(QPainter.Antialiasing)
        cx, cy = w // 2, h // 2
        # 十字准星
        p.setPen(QPen(QColor(0, 255, 120, 180), 1))
        p.drawLine(cx - 16, cy, cx - 4, cy)
        p.drawLine(cx + 4, cy, cx + 16, cy)
        p.drawLine(cx, cy - 16, cx, cy - 4)
        p.drawLine(cx, cy + 4, cx, cy + 16)
        # 比例尺：20 µm
        bar = int(round(20.0 / self._pixel_size))
        x0, y0 = 12, h - 18
        p.setPen(QPen(Qt.white, 3))
        p.drawLine(x0, y0, x0 + bar, y0)
        p.drawLine(x0, y0 - 4, x0, y0 + 4)
        p.drawLine(x0 + bar, y0 - 4, x0 + bar, y0 + 4)
        font = QFont()
        font.setPointSize(10)
        p.setFont(font)
        p.drawText(x0 + bar + 8, y0 + 5, "20 µm")
        # 台位读数
        p.setPen(QPen(QColor(0, 255, 120)))
        p.drawText(
            12, 22,
            "X %8.1f   Y %8.1f   Z %6.1f  µm" % (self._pos["x"], self._pos["y"], self._pos["z"]),
        )
        # 拖动提示
        p.setPen(QPen(QColor(255, 255, 255, 130)))
        p.drawText(w - 208, 22, "按住左键拖动图像移动台位")
        p.end()


# ----------------------------------------------------------------------
# 主窗口
# ----------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self, db, stage, camera, parent=None):
        super().__init__(parent)
        self._db = db
        self._stage = stage
        self._camera = camera
        self._motion: MotionWorker | None = None

        # 采集线程先创建（UI 构建时会连接其信号）
        self._cam_worker = CameraWorker(camera)

        self.setWindowTitle("显微镜模拟器 —— PyQt + SQLite（python-microscope 架构）")
        self._build_ui()

        self._cam_worker.frameReady.connect(self._on_frame)
        self._cam_worker.start()
        self._cam_worker.request_refresh()  # 启动即采一帧，避免黑屏

        # 相机启用后才能触发
        self._camera.enable()
        self._stage.enable()

        # 初始状态
        self._refresh_positions()
        self._refresh_log()
        self._load_camera_params()
        self._update_status()

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)

        top = QHBoxLayout()
        self._view = ViewWidget(pixel_size=self._camera._pixel_size)
        self._view.set_stage(self._stage)
        self._view.dragStarted.connect(self._on_drag_started)
        self._view.dragMoved.connect(self._on_drag_moved)
        self._view.dragFinished.connect(self._on_drag_finished)
        top.addWidget(self._view)
        # 控制面板放入滚动容器，防止小屏被裁剪
        scroll = QScrollArea()
        scroll.setWidget(self._build_side_panel())
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(380)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        top.addWidget(scroll)
        root.addLayout(top)

        log_group = QGroupBox("操作日志（SQLite: move_log）")
        log_layout = QVBoxLayout(log_group)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(120)
        self._log.setFont(QFont("Consolas", 8))
        log_layout.addWidget(self._log)
        root.addWidget(log_group)
        self.setCentralWidget(central)

    def _build_side_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(360)
        layout = QVBoxLayout(panel)

        # ---- 位移台 ----
        stage_box = QGroupBox("位移台（µm）")
        sv = QVBoxLayout(stage_box)

        pos_row = QHBoxLayout()
        self._pos_labels = {}
        for axis in ("x", "y", "z"):
            lab = QLabel("0.0")
            lab.setFont(QFont("Consolas", 12, QFont.Bold))
            lab.setStyleSheet("color: #0a0;")
            col = QVBoxLayout()
            head = QLabel(axis.upper())
            head.setAlignment(Qt.AlignCenter)
            col.addWidget(head)
            col.addWidget(lab)
            self._pos_labels[axis] = lab
            pos_row.addLayout(col)
        sv.addLayout(pos_row)

        # 绝对移动
        abs_row = QHBoxLayout()
        self._abs_spins = {}
        for axis in ("x", "y", "z"):
            lo, hi = self._stage.limits[axis]
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(1)
            sp.setSingleStep(10.0)
            sp.setSuffix(" " + axis.upper())
            sp.setValue(self._stage.position[axis])
            self._abs_spins[axis] = sp
            abs_row.addWidget(sp)
        self._btn_abs = QPushButton("移动到")
        self._btn_abs.clicked.connect(self._on_move_absolute)
        abs_row.addWidget(self._btn_abs)
        sv.addLayout(abs_row)

        # 点动
        jog_row1 = QHBoxLayout()
        jog_row2 = QHBoxLayout()
        self._step = QComboBox()
        self._step.addItems(["0.1", "1", "10", "50", "200"])
        self._step.setCurrentText("10")
        jog_row1.addWidget(QLabel("步距"))
        jog_row1.addWidget(self._step)
        jog_row1.addStretch(1)
        self._jog_buttons = {}
        for axis, sign, text in (
            ("x", -1, "X−"), ("y", -1, "Y−"), ("z", -1, "Z−"),
            ("x", +1, "X+"), ("y", +1, "Y+"), ("z", +1, "Z+"),
        ):
            btn = QPushButton(text)
            btn.setFixedWidth(50)
            btn.clicked.connect(lambda _, a=axis, s=sign: self._on_jog(a, s))
            self._jog_buttons[(axis, sign)] = btn
            if sign < 0:
                jog_row2.addWidget(btn)
            else:
                jog_row1.addWidget(btn)
        sv.addLayout(jog_row1)
        sv.addLayout(jog_row2)

        stop_row = QHBoxLayout()
        self._btn_stop = QPushButton("急停")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_stop)
        self._btn_focus0 = QPushButton("对焦归零 (Z→0)")
        self._btn_focus0.clicked.connect(lambda: self._start_move({"z": 0.0}, "focus"))
        stop_row.addWidget(self._btn_focus0)
        stop_row.addWidget(self._btn_stop)
        sv.addLayout(stop_row)
        layout.addWidget(stage_box)

        # ---- 相机 ----
        cam_box = QGroupBox("相机")
        cv = QVBoxLayout(cam_box)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("曝光 (ms)"))
        self._exposure = QDoubleSpinBox()
        self._exposure.setRange(1.0, 1000.0)
        self._exposure.setDecimals(0)
        row1.addWidget(self._exposure)
        row1.addWidget(QLabel("增益"))
        self._gain = QDoubleSpinBox()
        self._gain.setRange(0.0, 64.0)
        self._gain.setDecimals(1)
        self._gain.setSingleStep(0.5)
        row1.addWidget(self._gain)
        cv.addLayout(row1)
        row2 = QHBoxLayout()
        self._live = QCheckBox("连续采集")
        self._live.toggled.connect(self._cam_worker.set_live)
        row2.addWidget(self._live)
        self._btn_snap = QPushButton("拍快照")
        self._btn_snap.clicked.connect(self._cam_worker.request_snap)
        row2.addWidget(self._btn_snap)
        cv.addLayout(row2)
        layout.addWidget(cam_box)

        # ---- 样本图层（掩码 / 衬底 / 障碍物） ----
        sample_box = QGroupBox("样本图层（掩码 / 衬底 / 障碍物）")
        la = QVBoxLayout(sample_box)
        self._layer_table = QTableWidget(len(_LAYER_ROWS), 6)
        self._layer_table.setHorizontalHeaderLabels(
            ["图层", "数量", "形状", "标签", "尺寸(px)", "自定义顶点"]
        )
        self._layer_table.verticalHeader().setVisible(False)
        self._layer_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self._layer_table.setColumnWidth(0, 46)
        self._layer_table.setColumnWidth(1, 40)
        self._layer_table.setColumnWidth(2, 66)
        self._layer_table.setColumnWidth(3, 56)
        self._layer_table.setColumnWidth(4, 52)
        self._layer_table.horizontalHeader().setStretchLastSection(True)
        self._layer_table.setFixedHeight(120)
        spec = self._camera.get_sample_spec()
        for row, (name, cat) in enumerate(_LAYER_ROWS):
            cfg = spec[cat]
            item = QTableWidgetItem(name)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self._layer_table.setItem(row, 0, item)
            sp = QSpinBox()
            sp.setRange(0, 500)
            sp.setValue(int(cfg["count"]))
            self._layer_table.setCellWidget(row, 1, sp)
            combo = QComboBox()
            combo.addItems(_SHAPE_NAMES)
            combo.setCurrentText(
                next((n for n, k in _SHAPE_KEYS.items() if k == cfg["shape"]), "椭圆")
            )
            self._layer_table.setCellWidget(row, 2, combo)
            self._layer_table.setCellWidget(row, 3, QLineEdit(str(cfg["label"])))
            sz = QDoubleSpinBox()
            sz.setRange(4.0, 400.0)
            sz.setDecimals(0)
            sz.setValue(float(cfg["size"]))
            self._layer_table.setCellWidget(row, 4, sz)
            cust = QLineEdit(str(cfg.get("custom", "")))
            cust.setPlaceholderText("x,y x,y x,y ...")
            self._layer_table.setCellWidget(row, 5, cust)
        la.addWidget(self._layer_table)
        btn_regen = QPushButton("重新生成样本")
        btn_regen.clicked.connect(self._on_regenerate_sample)
        la.addWidget(btn_regen)
        layout.addWidget(sample_box)

        # ---- 预设位置 ----
        pos_box = QGroupBox("预设位置（SQLite: saved_positions）")
        pv = QVBoxLayout(pos_box)
        self._pos_list = QListWidget()
        self._pos_list.itemDoubleClicked.connect(lambda item: self._on_goto_saved())
        pv.addWidget(self._pos_list)
        name_row = QHBoxLayout()
        self._pos_name = QLineEdit()
        self._pos_name.setPlaceholderText("位置名称")
        name_row.addWidget(self._pos_name)
        btn_save = QPushButton("保存")
        btn_save.clicked.connect(self._on_save_position)
        name_row.addWidget(btn_save)
        pv.addLayout(name_row)
        btn_row = QHBoxLayout()
        btn_go = QPushButton("前往")
        btn_go.clicked.connect(self._on_goto_saved)
        btn_del = QPushButton("删除")
        btn_del.clicked.connect(self._on_delete_position)
        btn_row.addWidget(btn_go)
        btn_row.addWidget(btn_del)
        pv.addLayout(btn_row)
        layout.addWidget(pos_box)

        layout.addStretch(1)
        return panel

    # ------------------------------------------------------------------
    # 位移台操作
    # ------------------------------------------------------------------
    def _set_moving(self, moving: bool) -> None:
        self._btn_abs.setEnabled(not moving)
        self._btn_focus0.setEnabled(not moving)
        self._btn_stop.setEnabled(moving)
        for btn in self._jog_buttons.values():
            btn.setEnabled(not moving)
        for sp in self._abs_spins.values():
            sp.setEnabled(not moving)

    def _start_move(self, target: Dict[str, float], source: str) -> None:
        if self._motion is not None and self._motion.isRunning():
            return
        self._motion = MotionWorker(self._stage, target, source)
        self._motion.progressed.connect(self._on_progress)
        self._motion.finishedMove.connect(self._on_move_finished)
        self._set_moving(True)
        self._motion.start()

    def _on_jog(self, axis: str, sign: int) -> None:
        step = float(self._step.currentText()) * sign
        self._start_move({axis: self._stage.position[axis] + step}, "jog")

    def _on_move_absolute(self) -> None:
        target = {a: sp.value() for a, sp in self._abs_spins.items()}
        self._start_move(target, "absolute")

    def _on_stop(self) -> None:
        if self._motion is not None:
            self._motion.stop()

    def _on_progress(self, pos: Dict[str, float]) -> None:
        self._refresh_pos_labels(pos)
        self._view.set_stage_position(pos)
        self._sync_abs_spins(pos)
        # 移动过程中让视野跟随台位（非连续采集模式下每帧单独刷新）
        self._cam_worker.request_refresh()

    # ------------------------------------------------------------------
    # 鼠标拖动移动
    # ------------------------------------------------------------------
    def _on_drag_started(self) -> None:
        # 快速预览：跳过曝光拟真等待，帧率不受曝光时间限制
        self._camera.set_fast_preview(True)
        # 若有移动任务在跑，先急停并等它收尾（其最后的插值/落库步骤
        # 否则可能与拖动写位置竞争），之后由拖动独占台位
        if self._motion is not None and self._motion.isRunning():
            self._motion.stop()
            self._motion.wait(200)

    def _on_drag_moved(self, pos: Dict[str, float]) -> None:
        self._refresh_pos_labels(pos)
        self._sync_abs_spins(pos)
        self._cam_worker.request_refresh()  # 视野实时跟随拖动

    def _on_drag_finished(self) -> None:
        self._camera.set_fast_preview(False)  # 恢复曝光拟真节奏
        if self._motion is not None and self._motion.isRunning():
            # 拖动开始前的旧任务刚被急停，等它收尾（最长 200ms）
            self._motion.wait(200)
        if self._motion is not None and self._motion.isRunning():
            # 仍未收尾则直接持久化当前位置，不做平滑停靠
            self._stage.move_to(dict(self._stage.position), persist=True)
            return
        # 从当前位置平滑停靠到钳位后的目标（含持久化与日志）
        self._start_move(dict(self._stage.position), "drag")

    def _on_move_finished(self, completed: bool, source: str) -> None:
        pos = dict(self._stage.position)
        self._refresh_pos_labels(pos)
        self._view.set_stage_position(pos)
        self._sync_abs_spins(pos)
        self._set_moving(False)
        self._cam_worker.request_refresh()  # 停稳后再刷新最终帧
        note = "完成" if completed else "被急停中断"
        self._db.log_move(source, pos["x"], pos["y"], pos["z"], note)
        self._append_log(f"[移动/{source}] {note} → X={pos['x']:.1f} Y={pos['y']:.1f} Z={pos['z']:.1f}")
        self._refresh_log()

    def _refresh_pos_labels(self, pos) -> None:
        for axis, lab in self._pos_labels.items():
            lab.setText("%.1f" % pos[axis])

    def _sync_abs_spins(self, pos) -> None:
        for axis, sp in self._abs_spins.items():
            sp.blockSignals(True)
            sp.setValue(pos[axis])
            sp.blockSignals(False)

    def _refresh_positions(self) -> None:
        self._pos_list.clear()
        for row in self._db.list_positions():
            text = "%s  (X %.1f, Y %.1f, Z %.1f)" % (row["name"], row["x"], row["y"], row["z"])
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, row["id"])
            self._pos_list.addItem(item)

    def _on_save_position(self) -> None:
        name = self._pos_name.text().strip()
        if not name:
            QMessageBox.information(self, "保存位置", "请输入位置名称。")
            return
        pos = self._stage.position
        if not self._db.add_position(name, pos["x"], pos["y"], pos["z"]):
            QMessageBox.warning(self, "保存位置", f"名称「{name}」已存在。")
            return
        self._pos_name.clear()
        self._refresh_positions()
        self._append_log(f"[预设] 已保存位置「{name}」")

    def _on_goto_saved(self) -> None:
        item = self._pos_list.currentItem()
        if item is None:
            return
        row = self._db.get_position(item.data(Qt.UserRole))
        if row is not None:
            self._start_move({"x": row["x"], "y": row["y"], "z": row["z"]}, "preset")

    def _on_delete_position(self) -> None:
        item = self._pos_list.currentItem()
        if item is not None:
            self._db.delete_position(item.data(Qt.UserRole))
            self._refresh_positions()
            self._append_log("[预设] 已删除位置")

    # ------------------------------------------------------------------
    # 相机
    # ------------------------------------------------------------------
    def _load_camera_params(self) -> None:
        self._exposure.setValue(self._camera.get_exposure_time() * 1000.0)
        self._gain.setValue(self._camera.get_gain())
        self._exposure.valueChanged.connect(
            lambda v: self._camera.set_exposure_time(v / 1000.0)
        )
        self._gain.valueChanged.connect(self._camera.set_gain)

    def _on_frame(self, frame, is_snap: bool, frame_pos=None) -> None:
        self._view.set_frame(frame, frame_pos)
        if is_snap:
            pos = frame_pos if frame_pos is not None else self._stage.position
            png = self._db.encode_png(frame)
            acq_id = self._db.save_acquisition(
                png,
                exposure_ms=self._camera.get_exposure_time() * 1000.0,
                gain=self._camera.get_gain(),
                x=pos["x"], y=pos["y"], z=pos["z"],
            )
            self._append_log(f"[快照] 已保存 acquisitions.id={acq_id}")

    # ------------------------------------------------------------------
    # 样本图层
    # ------------------------------------------------------------------
    def _on_regenerate_sample(self) -> None:
        t = self._layer_table
        spec = {}
        for row, (name, cat) in enumerate(_LAYER_ROWS):
            spec[cat] = {
                "count": t.cellWidget(row, 1).value(),
                "shape": _SHAPE_KEYS[t.cellWidget(row, 2).currentText()],
                "label": t.cellWidget(row, 3).text().strip() or name,
                "size": t.cellWidget(row, 4).value(),
                "custom": t.cellWidget(row, 5).text().strip(),
            }
        self._camera.set_sample_spec(spec)
        self._cam_worker.request_refresh()
        summary = ", ".join(
            "%s×%d(%s)" % (name, spec[cat]["count"], spec[cat]["shape"])
            for name, cat in _LAYER_ROWS
        )
        self._append_log("[样本] 已重新生成：%s" % summary)

    # ------------------------------------------------------------------
    # 日志 / 状态栏
    # ------------------------------------------------------------------
    def _refresh_log(self) -> None:
        self._log.setPlainText(
            "\n".join(
                "{} [{:6s}] X {:8.1f} Y {:8.1f} Z {:6.1f}  {}".format(
                    r["ts"], r["source"], r["x"], r["y"], r["z"], r["note"] or ""
                )
                for r in self._db.recent_moves()
            )
        )
        self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())

    def _append_log(self, text: str) -> None:
        self._log.appendPlainText(time.strftime("[%H:%M:%S] ") + text)

    def _update_status(self) -> None:
        self.statusBar().showMessage(
            "位移台: %s | 相机: %s | 数据库: %s"
            % (
                "已启用" if self._stage.get_is_enabled() else "停用",
                "已启用" if self._camera.get_is_enabled() else "停用",
                self._db._conn.execute("PRAGMA database_list").fetchone()[2],
            )
        )

    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:
        self._cam_worker.stop()
        self._cam_worker.wait(2000)
        if self._motion is not None and self._motion.isRunning():
            self._motion.stop()
            self._motion.wait(3000)
        self._camera.shutdown()
        self._stage.shutdown()
        self._db.close()
        super().closeEvent(event)
