"""
视频前后景分离 UI 模块。

功能：
  1. 固定 UI 框架作为 3 个播放窗口的容器
  2. 左：原视频；中：前后景分离（保留前景原色）；右：二值掩码（前景白/背景黑）
  3. 参数配置区：视频选择、分割方法、最小面积、播放控制等

使用方法：
  python -m Utils.ForegroundAndBackground.video_seg_ui
  python -m Utils.ForegroundAndBackground.video_seg_ui path/to/video.mp4
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Qt 兼容
try:
    from PySide6.QtCore import Qt, QTimer, Signal, QThread, QMutex
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSlider,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    from PySide2.QtCore import Qt, QTimer, Signal, QThread, QMutex
    from PySide2.QtGui import QImage, QPixmap
    from PySide2.QtWidgets import (
        QApplication,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSlider,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )


# ============================================================
# 前后景分离算法（复用 video_seg.py）
# ============================================================

def remove_small_components(binary: np.ndarray, min_area: int = 100) -> np.ndarray:
    """删除小连通域噪声。"""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    out = np.zeros_like(binary)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 255
    return out


def clean_binary(binary: np.ndarray, min_area: int = 100) -> np.ndarray:
    """二值图后处理：开运算 → 闭运算 → 去小连通域。"""
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = remove_small_components(binary, min_area=min_area)
    return binary


def _make_mask(frame: np.ndarray, method: str) -> np.ndarray:
    """根据方法生成二值掩码。"""
    if method == "bright":
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 35, 55])
        upper = np.array([179, 255, 255])
        return cv2.inRange(hsv, lower, upper)
    elif method == "otsu":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    elif method == "blue":
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, np.array([85, 40, 45]), np.array([130, 255, 255]))
    elif method == "yellow":
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, np.array([10, 45, 55]), np.array([45, 255, 255]))
    else:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, np.array([0, 35, 55]), np.array([179, 255, 255]))


def segment_frame(frame: np.ndarray, method: str, min_area: int):
    """
    对单帧进行前后景分离，返回 (前景提取图, 二值掩码图)。

    前景提取图：前景保留原色，背景黑色。
    二值掩码图：前景白色，背景黑色。
    """
    mask = _make_mask(frame, method)
    mask_clean = clean_binary(mask, min_area=min_area)

    # 前景提取
    mask_3ch = cv2.merge([mask_clean, mask_clean, mask_clean])
    fg = cv2.bitwise_and(frame, mask_3ch)

    # 二值掩码（白底黑底）
    binary = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR)

    return fg, binary


# ============================================================
# 视频播放线程
# ============================================================

class VideoWorker(QThread):
    """后台线程：逐帧读取视频并处理。"""

    frame_ready = Signal(object, object, object, int, int)
    finished = Signal()

    def __init__(self, video_path: str, method: str = "bright", min_area: int = 100):
        super().__init__()
        self.video_path = video_path
        self.method = method
        self.min_area = min_area
        self._mutex = QMutex()
        self._paused = False
        self._stop = False
        self._seek_frame = -1

    def set_paused(self, paused: bool):
        self._mutex.lock()
        self._paused = paused
        self._mutex.unlock()

    def request_stop(self):
        self._mutex.lock()
        self._stop = True
        self._mutex.unlock()

    def request_seek(self, frame_idx: int):
        self._mutex.lock()
        self._seek_frame = frame_idx
        self._mutex.unlock()

    def update_params(self, method: str, min_area: int):
        self._mutex.lock()
        self.method = method
        self.min_area = min_area
        self._mutex.unlock()

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.finished.emit()
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval_ms = max(1, int(1000 / fps))

        while True:
            self._mutex.lock()
            stop = self._stop
            paused = self._paused
            seek = self._seek_frame
            self._seek_frame = -1
            self._mutex.unlock()

            if stop:
                break

            if seek >= 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, seek)

            if paused:
                self.msleep(50)
                continue

            ret, frame = cap.read()
            if not ret:
                # 循环播放
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # 获取当前参数
            self._mutex.lock()
            method = self.method
            min_area = self.min_area
            self._mutex.unlock()

            fg, binary = segment_frame(frame, method, min_area)

            self.frame_ready.emit(frame, fg, binary, frame_idx, total)
            self.msleep(interval_ms)

        cap.release()
        self.finished.emit()


# ============================================================
# 视频显示标签
# ============================================================

class VideoLabel(QLabel):
    """用于显示视频帧的 QLabel，自动缩放。"""

    def __init__(self, title: str = ""):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setStyleSheet(
            "QLabel { background-color: #1a1a2e; border: 2px solid #444; "
            "border-radius: 4px; color: #ccc; font-size: 14px; }"
        )
        if title:
            self.setText(title)

    def update_frame(self, frame_bgr: np.ndarray):
        """将 BGR numpy 帧显示到标签上。"""
        if frame_bgr is None or frame_bgr.size == 0:
            return
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.setPixmap(
            pix.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )


# ============================================================
# 主窗口
# ============================================================

class VideoSegMainWindow(QMainWindow):
    """视频前后景分离主窗口。"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("视频前后景分离工具")
        self.setMinimumSize(1200, 700)
        self.resize(1400, 800)

        self._worker: Optional[VideoWorker] = None
        self._video_path: str = ""

        self._build_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(8)

        # ---- 顶部：参数配置区 ----
        config_group = QGroupBox("参数配置")
        config_layout = QHBoxLayout(config_group)
        config_layout.setSpacing(12)

        # 视频文件
        file_form = QFormLayout()
        file_form.setContentsMargins(4, 4, 4, 4)
        self.btn_open = QPushButton("选择视频")
        self.btn_open.setMinimumWidth(90)
        self.lbl_file = QLabel("未选择")
        self.lbl_file.setStyleSheet("color: #888;")
        file_form.addRow("视频文件", self.btn_open)
        file_form.addRow("", self.lbl_file)
        config_layout.addLayout(file_form)

        # 分割方法
        method_form = QFormLayout()
        method_form.setContentsMargins(4, 4, 4, 4)
        self.combo_method = QComboBox()
        self.combo_method.addItems(["bright", "otsu", "blue", "yellow"])
        self.combo_method.setCurrentText("bright")
        self.combo_method.setToolTip(
            "bright: 提取所有亮色前景\n"
            "otsu: Otsu 自动二值化\n"
            "blue: 提取蓝色区域\n"
            "yellow: 提取黄色/橙色区域"
        )
        method_form.addRow("分割方法", self.combo_method)
        config_layout.addLayout(method_form)

        # 最小面积
        area_form = QFormLayout()
        area_form.setContentsMargins(4, 4, 4, 4)
        self.spin_min_area = QSpinBox()
        self.spin_min_area.setRange(10, 5000)
        self.spin_min_area.setValue(100)
        self.spin_min_area.setSingleStep(10)
        self.spin_min_area.setToolTip("小于此面积的连通域将被过滤")
        area_form.addRow("最小面积", self.spin_min_area)
        config_layout.addLayout(area_form)

        # 播放控制按钮
        btn_form = QFormLayout()
        btn_form.setContentsMargins(4, 4, 4, 4)
        btn_row = QHBoxLayout()
        self.btn_play = QPushButton("▶ 播放")
        self.btn_play.setEnabled(False)
        self.btn_pause = QPushButton("⏸ 暂停")
        self.btn_pause.setEnabled(False)
        self.btn_stop = QPushButton("⏹ 停止")
        self.btn_stop.setEnabled(False)
        btn_row.addWidget(self.btn_play)
        btn_row.addWidget(self.btn_pause)
        btn_row.addWidget(self.btn_stop)
        btn_form.addRow("播放控制", btn_row)
        config_layout.addLayout(btn_form)

        root_layout.addWidget(config_group)

        # ---- 中部：三视频显示区 ----
        video_layout = QHBoxLayout()
        video_layout.setSpacing(8)

        self.lbl_original = VideoLabel("原视频")
        self.lbl_segmented = VideoLabel("前景提取（背景黑色）")
        self.lbl_binary = VideoLabel("二值掩码（白/黑）")

        video_layout.addWidget(self.lbl_original, 1)
        video_layout.addWidget(self.lbl_segmented, 1)
        video_layout.addWidget(self.lbl_binary, 1)

        root_layout.addLayout(video_layout, 1)

        # ---- 底部：进度条 + 状态 ----
        bottom_layout = QHBoxLayout()

        self.slider_frame = QSlider(Qt.Horizontal)
        self.slider_frame.setRange(0, 100)
        self.slider_frame.setEnabled(False)
        bottom_layout.addWidget(self.slider_frame, 1)

        self.lbl_status = QLabel("就绪")
        self.lbl_status.setMinimumWidth(200)
        self.lbl_status.setStyleSheet("color: #888; padding: 4px;")
        bottom_layout.addWidget(self.lbl_status)

        root_layout.addLayout(bottom_layout)

    # ------------------------------------------------------------------
    # 信号连接
    # ------------------------------------------------------------------

    def _connect_signals(self):
        self.btn_open.clicked.connect(self._on_open_video)
        self.btn_play.clicked.connect(self._on_play)
        self.btn_pause.clicked.connect(self._on_pause)
        self.btn_stop.clicked.connect(self._on_stop)

        self.combo_method.currentTextChanged.connect(self._on_params_changed)
        self.spin_min_area.valueChanged.connect(self._on_params_changed)

        self.slider_frame.sliderMoved.connect(self._on_seek)

    # ------------------------------------------------------------------
    # 槽函数
    # ------------------------------------------------------------------

    def _on_open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频文件",
            str(Path(__file__).parent / "Video"),
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv);;所有文件 (*.*)",
        )
        if not path:
            return

        self._video_path = path
        self.lbl_file.setText(Path(path).name)
        self.lbl_file.setStyleSheet("color: #ddd;")
        self.lbl_status.setText(f"已加载: {Path(path).name}")

        # 启用播放按钮
        self.btn_play.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)

        # 自动开始播放
        self._start_worker()

    def _start_worker(self):
        """启动视频处理线程。"""
        self._stop_worker()

        method = self.combo_method.currentText()
        min_area = self.spin_min_area.value()

        self._worker = VideoWorker(self._video_path, method, min_area)
        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

        self.btn_play.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.slider_frame.setEnabled(True)
        self.lbl_status.setText("播放中...")

    def _stop_worker(self):
        """停止当前工作线程。"""
        if self._worker is not None:
            self._worker.request_stop()
            self._worker.wait(3000)
            self._worker = None

    def _on_play(self):
        if self._worker is None:
            self._start_worker()
        else:
            self._worker.set_paused(False)
            self.btn_pause.setEnabled(True)
            self.btn_play.setEnabled(False)
            self.lbl_status.setText("播放中...")

    def _on_pause(self):
        if self._worker is not None:
            self._worker.set_paused(True)
            self.btn_play.setEnabled(True)
            self.btn_pause.setEnabled(False)
            self.lbl_status.setText("已暂停")

    def _on_stop(self):
        self._stop_worker()
        self.btn_play.setEnabled(bool(self._video_path))
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.slider_frame.setEnabled(False)
        self.lbl_status.setText("已停止")

    def _on_params_changed(self):
        """参数变化时实时更新到工作线程。"""
        if self._worker is not None:
            method = self.combo_method.currentText()
            min_area = self.spin_min_area.value()
            self._worker.update_params(method, min_area)

    def _on_seek(self, value: int):
        """拖动进度条跳转到指定帧。"""
        if self._worker is not None:
            self._worker.request_seek(value)

    def _on_frame_ready(self, original, segmented, binary, frame_idx, total):
        """收到新帧，更新三个显示区域。"""
        self.lbl_original.update_frame(original)
        self.lbl_segmented.update_frame(segmented)
        self.lbl_binary.update_frame(binary)

        # 更新进度条
        if total > 0:
            if self.slider_frame.maximum() != total:
                self.slider_frame.setRange(0, total)
            self.slider_frame.setValue(frame_idx)
            self.lbl_status.setText(
                f"帧 {frame_idx}/{total}  |  "
                f"方法: {self.combo_method.currentText()}  |  "
                f"最小面积: {self.spin_min_area.value()}"
            )

    def _on_worker_finished(self):
        """工作线程结束。"""
        self.lbl_status.setText("播放结束")

    def closeEvent(self, event):
        """关闭窗口时释放资源。"""
        self._stop_worker()
        super().closeEvent(event)


# ============================================================
# 入口
# ============================================================

def main():
    """主入口。"""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # 深色主题
    from PySide6.QtGui import QPalette, QColor
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 30, 46))
    palette.setColor(QPalette.WindowText, QColor(205, 214, 244))
    palette.setColor(QPalette.Base, QColor(49, 50, 68))
    palette.setColor(QPalette.AlternateBase, QColor(49, 50, 68))
    palette.setColor(QPalette.ToolTipBase, QColor(205, 214, 244))
    palette.setColor(QPalette.ToolTipText, QColor(205, 214, 244))
    palette.setColor(QPalette.Text, QColor(205, 214, 244))
    palette.setColor(QPalette.Button, QColor(49, 50, 68))
    palette.setColor(QPalette.ButtonText, QColor(205, 214, 244))
    palette.setColor(QPalette.Highlight, QColor(137, 180, 250))
    palette.setColor(QPalette.HighlightedText, QColor(30, 30, 46))
    app.setPalette(palette)

    win = VideoSegMainWindow()
    win.show()

    # 如果命令行指定了视频路径，自动加载
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        if Path(video_path).exists():
            win._video_path = video_path
            win.lbl_file.setText(Path(video_path).name)
            win.lbl_file.setStyleSheet("color: #ddd;")
            win._start_worker()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
