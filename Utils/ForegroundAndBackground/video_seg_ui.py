"""
视频前后景分离 UI 模块。

功能：
  1. 固定 UI 框架作为 3 个播放窗口的容器
  2. 支持两种模式：视频模式 / 屏幕ROI模式
  3. 视频模式：播放本地视频，进行前后景分离
  4. 屏幕ROI模式：截取屏幕指定区域，实时前后景分离
  5. 检测最长直线并计算与水平线夹角
  6. 参数配置区：模式选择、视频选择、分割方法、最小面积、播放控制等

使用方法：
  python -m Utils.ForegroundAndBackground.video_seg_ui
  python -m Utils.ForegroundAndBackground.video_seg_ui path/to/video.mp4
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# Qt 兼容
try:
    from PySide6.QtCore import Qt, QTimer, Signal, QThread, QMutex, QPoint, QRect
    from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QPalette, QColor
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
        QDialog,
        QStackedWidget,
    )
except ImportError:
    from PySide2.QtCore import Qt, QTimer, Signal, QThread, QMutex, QPoint, QRect
    from PySide2.QtGui import QImage, QPixmap, QPainter, QPen, QPalette, QColor
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
        QDialog,
        QStackedWidget,
    )


# ============================================================
# 前后景分离算法
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
    """对单帧进行前后景分离，返回 (前景提取图, 二值掩码图)。"""
    mask = _make_mask(frame, method)
    mask_clean = clean_binary(mask, min_area=min_area)

    # 前景提取
    mask_3ch = cv2.merge([mask_clean, mask_clean, mask_clean])
    fg = cv2.bitwise_and(frame, mask_3ch)

    # 二值掩码
    binary = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR)

    return fg, binary


# ============================================================
# 直线检测（稳定化版本）
# ============================================================

class StableLineDetector:
    """
    稳定化直线检测器。
    
    通过多帧累积、EMA平滑和跳变过滤，确保检测到的最长直线在视频序列中保持稳定。
    """
    
    def __init__(
        self,
        ema_alpha: float = 0.3,
        jump_threshold: float = 5.0,
        min_confidence: int = 3,
        history_size: int = 10,
    ):
        """
        Args:
            ema_alpha: EMA平滑系数（0-1），越小越平滑
            jump_threshold: 角度跳变阈值（度），超过此值认为是噪声
            min_confidence: 最小置信帧数，至少检测到N次才认为稳定
            history_size: 历史记录窗口大小
        """
        self.ema_alpha = ema_alpha
        self.jump_threshold = jump_threshold
        self.min_confidence = min_confidence
        self.history_size = history_size
        
        # 稳定状态
        self._stable_line: Optional[Tuple] = None
        self._stable_angle: float = 0.0
        self._confidence: int = 0
        
        # 历史记录
        self._angle_history: List[float] = []
        self._line_history: List[Tuple] = []
        
    def detect(self, frame: np.ndarray) -> Tuple[Optional[Tuple], float]:
        """
        检测稳定化的最长直线。
        
        Returns:
            (line, angle_deg): 稳定的直线和角度，如果未达到置信度则返回None
        """
        # 单帧检测
        current_line, current_angle = detect_longest_line_single(frame)
        
        if current_line is None:
            # 当前帧未检测到直线，保持历史状态
            return self._stable_line, self._stable_angle
        
        # 更新历史
        self._angle_history.append(current_angle)
        self._line_history.append(current_line)
        
        # 限制历史窗口
        if len(self._angle_history) > self.history_size:
            self._angle_history.pop(0)
            self._line_history.pop(0)
        
        # 首次检测或置信度不足
        if self._stable_line is None:
            self._stable_line = current_line
            self._stable_angle = current_angle
            self._confidence = 1
            return self._stable_line, self._stable_angle
        
        # 计算角度变化
        angle_diff = abs(current_angle - self._stable_angle)
        # 处理角度环绕（0度和180度接近）
        if angle_diff > 90:
            angle_diff = 180 - angle_diff
        
        # 跳变过滤
        if angle_diff > self.jump_threshold:
            # 认为是噪声，保持当前稳定值
            return self._stable_line, self._stable_angle
        
        # EMA平滑
        smoothed_angle = (
            self.ema_alpha * current_angle + 
            (1 - self.ema_alpha) * self._stable_angle
        )
        
        # 平滑直线坐标
        x1, y1, x2, y2 = current_line
        sx1, sy1, sx2, sy2 = self._stable_line
        
        smoothed_line = (
            int(self.ema_alpha * x1 + (1 - self.ema_alpha) * sx1),
            int(self.ema_alpha * y1 + (1 - self.ema_alpha) * sy1),
            int(self.ema_alpha * x2 + (1 - self.ema_alpha) * sx2),
            int(self.ema_alpha * y2 + (1 - self.ema_alpha) * sy2),
        )
        
        # 更新稳定状态
        self._stable_line = smoothed_line
        self._stable_angle = smoothed_angle
        self._confidence = min(self._confidence + 1, self.min_confidence)
        
        return self._stable_line, self._stable_angle
    
    def reset(self):
        """重置稳定状态。"""
        self._stable_line = None
        self._stable_angle = 0.0
        self._confidence = 0
        self._angle_history.clear()
        self._line_history.clear()


def detect_longest_line_single(frame: np.ndarray) -> Tuple[Optional[Tuple], float]:
    """单帧直线检测（未稳定化）。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=50,
        minLineLength=30, maxLineGap=10,
    )

    if lines is None or len(lines) == 0:
        return None, 0.0

    longest_line = None
    max_length = 0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length > max_length:
            max_length = length
            longest_line = (int(x1), int(y1), int(x2), int(y2))

    if longest_line is None:
        return None, 0.0

    x1, y1, x2, y2 = longest_line
    dx = x2 - x1
    dy = y2 - y1
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)
    angle_deg = abs(angle_deg)
    if angle_deg > 90:
        angle_deg = 180 - angle_deg

    return longest_line, float(angle_deg)


def draw_line_on_frame(frame: np.ndarray, line: Tuple, angle_deg: float) -> np.ndarray:
    """在图像上绘制检测到的直线和角度信息。"""
    result = frame.copy()
    if line is not None:
        x1, y1, x2, y2 = line
        cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(result, (x1, y1), 5, (0, 0, 255), -1)
        cv2.circle(result, (x2, y2), 5, (0, 0, 255), -1)
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        cv2.putText(
            result, f"Angle: {angle_deg:.1f} deg",
            (mid_x + 10, mid_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
        )
    return result


# ============================================================
# 屏幕ROI选择对话框
# ============================================================

class ScreenRoiSelectDialog(QDialog):
    """屏幕ROI选择对话框：截图后让用户划取ROI区域。"""

    def __init__(self, screenshot: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("划取屏幕ROI区域")
        self.setModal(True)
        self.screenshot = screenshot
        self.roi_rect: Optional[Tuple[int, int, int, int]] = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        hint = QLabel("在图像上拖拽鼠标划取ROI区域，然后点击确定")
        hint.setStyleSheet("color: #ccc; padding: 8px;")
        layout.addWidget(hint)

        self.lbl_image = _RoiSelectLabel(self.screenshot)
        self.lbl_image.roi_selected.connect(self._on_roi_selected)
        layout.addWidget(self.lbl_image, 1)

        btn_layout = QHBoxLayout()
        self.btn_ok = QPushButton("确定")
        self.btn_ok.setEnabled(False)
        self.btn_cancel = QPushButton("取消")
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_ok)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)

    def _on_roi_selected(self, rect: Tuple[int, int, int, int]):
        self.roi_rect = rect
        self.btn_ok.setEnabled(True)

    def get_roi(self) -> Optional[Tuple[int, int, int, int]]:
        return self.roi_rect


class _RoiSelectLabel(QLabel):
    """用于划取ROI的图像标签。"""

    roi_selected = Signal(tuple)

    def __init__(self, image_bgr: np.ndarray):
        super().__init__()
        self.image_bgr = image_bgr
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setStyleSheet("border: 2px solid #666;")
        self._start_pos: Optional[QPoint] = None
        self._end_pos: Optional[QPoint] = None
        self._drawing = False
        self._update_display()

    def _update_display(self):
        h, w = self.image_bgr.shape[:2]
        rgb = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        scaled_pix = pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        if self._start_pos and self._end_pos:
            painter = QPainter(scaled_pix)
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            rect = QRect(self._start_pos, self._end_pos).normalized()
            painter.drawRect(rect)
            painter.end()

        self.setPixmap(scaled_pix)

    def _get_image_offset(self) -> Tuple[int, int]:
        pix = self.pixmap()
        if pix is None:
            return 0, 0
        offset_x = (self.width() - pix.width()) // 2
        offset_y = (self.height() - pix.height()) // 2
        return offset_x, offset_y

    def _pos_to_image_coords(self, pos: QPoint) -> Tuple[int, int]:
        pix = self.pixmap()
        if pix is None:
            return 0, 0
        offset_x, offset_y = self._get_image_offset()
        lx = pos.x() - offset_x
        ly = pos.y() - offset_y
        h, w = self.image_bgr.shape[:2]
        scale_x = w / pix.width()
        scale_y = h / pix.height()
        ix = max(0, min(int(lx * scale_x), w - 1))
        iy = max(0, min(int(ly * scale_y), h - 1))
        return ix, iy

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._start_pos = event.pos()
            self._end_pos = event.pos()
            self._drawing = True

    def mouseMoveEvent(self, event):
        if self._drawing:
            self._end_pos = event.pos()
            self._update_display()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._drawing:
            self._end_pos = event.pos()
            self._drawing = False
            x1, y1 = self._pos_to_image_coords(self._start_pos)
            x2, y2 = self._pos_to_image_coords(self._end_pos)
            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            if w > 10 and h > 10:
                self.roi_selected.emit((x, y, w, h))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()


# ============================================================
# 视频播放线程
# ============================================================

class VideoWorker(QThread):
    """后台线程：逐帧读取视频并处理。"""

    frame_ready = Signal(object, object, object, int, int, float)
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
        
        # 创建稳定化直线检测器
        line_detector = StableLineDetector(
            ema_alpha=0.3,
            jump_threshold=5.0,
            min_confidence=3,
        )

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
                # 跳转时重置检测器状态
                line_detector.reset()

            if paused:
                self.msleep(50)
                continue

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                line_detector.reset()
                continue

            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            self._mutex.lock()
            method = self.method
            min_area = self.min_area
            self._mutex.unlock()

            # 前后景分离
            fg, binary = segment_frame(frame, method, min_area)

            # 稳定化直线检测
            line, angle_deg = line_detector.detect(fg)
            if line is not None:
                fg = draw_line_on_frame(fg, line, angle_deg)

            self.frame_ready.emit(frame, fg, binary, frame_idx, total, angle_deg)
            self.msleep(interval_ms)

        cap.release()
        self.finished.emit()


# ============================================================
# 屏幕ROI播放线程
# ============================================================

class ScreenRoiWorker(QThread):
    """后台线程：实时截取屏幕ROI区域并处理。"""

    frame_ready = Signal(object, object, object, float)
    finished = Signal()

    def __init__(
        self,
        screen_roi: Tuple[int, int, int, int],
        method: str = "bright",
        min_area: int = 100,
        fps: float = 10.0,
    ):
        super().__init__()
        self.screen_roi = screen_roi  # (x, y, w, h)
        self.method = method
        self.min_area = min_area
        self.fps = fps
        self._mutex = QMutex()
        self._paused = False
        self._stop = False

    def set_paused(self, paused: bool):
        self._mutex.lock()
        self._paused = paused
        self._mutex.unlock()

    def request_stop(self):
        self._mutex.lock()
        self._stop = True
        self._mutex.unlock()

    def update_params(self, method: str, min_area: int):
        self._mutex.lock()
        self.method = method
        self.min_area = min_area
        self._mutex.unlock()

    def update_screen_roi(self, roi: Tuple[int, int, int, int]):
        self._mutex.lock()
        self.screen_roi = roi
        self._mutex.unlock()

    def run(self):
        try:
            import pyautogui
        except ImportError:
            self.finished.emit()
            return

        interval_ms = max(50, int(1000 / self.fps))
        
        # 创建稳定化直线检测器
        line_detector = StableLineDetector(
            ema_alpha=0.3,
            jump_threshold=5.0,
            min_confidence=3,
        )

        while True:
            self._mutex.lock()
            stop = self._stop
            paused = self._paused
            screen_roi = self.screen_roi
            method = self.method
            min_area = self.min_area
            self._mutex.unlock()

            if stop:
                break

            if paused:
                self.msleep(50)
                continue

            # 截取屏幕ROI区域
            x, y, w, h = screen_roi
            try:
                screenshot = pyautogui.screenshot(region=(x, y, w, h))
                screenshot_rgb = np.array(screenshot)
                frame = cv2.cvtColor(screenshot_rgb, cv2.COLOR_RGB2BGR)
            except Exception:
                self.msleep(interval_ms)
                continue

            # 前后景分离
            fg, binary = segment_frame(frame, method, min_area)

            # 稳定化直线检测
            line, angle_deg = line_detector.detect(fg)
            if line is not None:
                fg = draw_line_on_frame(fg, line, angle_deg)

            self.frame_ready.emit(frame, fg, binary, angle_deg)
            self.msleep(interval_ms)

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
            "border-radius: 4px; color: #ccc; font-size: 12px; }"
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

        # 当前模式: "video" 或 "screen_roi"
        self._current_mode: str = "video"

        # 视频模式相关
        self._video_worker: Optional[VideoWorker] = None
        self._video_path: str = ""

        # 屏幕ROI模式相关
        self._screen_roi_worker: Optional[ScreenRoiWorker] = None
        self._screen_roi: Optional[Tuple[int, int, int, int]] = None

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

        # 模式选择
        mode_form = QFormLayout()
        mode_form.setContentsMargins(4, 4, 4, 4)
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["video", "screen_roi"])
        self.combo_mode.setCurrentText("video")
        self.combo_mode.setToolTip(
            "video: 播放本地视频\n"
            "screen_roi: 截取屏幕指定区域实时处理"
        )
        mode_form.addRow("工作模式", self.combo_mode)
        config_layout.addLayout(mode_form)

        # 视频文件（视频模式专用）
        file_form = QFormLayout()
        file_form.setContentsMargins(4, 4, 4, 4)
        self.btn_open = QPushButton("选择视频")
        self.btn_open.setMinimumWidth(90)
        self.lbl_file = QLabel("未选择")
        self.lbl_file.setStyleSheet("color: #888;")
        file_form.addRow("视频文件", self.btn_open)
        file_form.addRow("", self.lbl_file)
        config_layout.addLayout(file_form)

        # 屏幕ROI（屏幕ROI模式专用）
        roi_form = QFormLayout()
        roi_form.setContentsMargins(4, 4, 4, 4)
        self.btn_select_roi = QPushButton("选择屏幕ROI")
        self.btn_select_roi.setToolTip("截取屏幕并划取ROI区域")
        self.lbl_roi = QLabel("未设置")
        self.lbl_roi.setStyleSheet("color: #888;")
        roi_form.addRow("屏幕ROI", self.btn_select_roi)
        roi_form.addRow("", self.lbl_roi)
        config_layout.addLayout(roi_form)

        # 分割方法
        method_form = QFormLayout()
        method_form.setContentsMargins(4, 4, 4, 4)
        self.combo_method = QComboBox()
        self.combo_method.addItems(["bright", "otsu", "blue", "yellow"])
        self.combo_method.setCurrentText("bright")
        method_form.addRow("分割方法", self.combo_method)
        config_layout.addLayout(method_form)

        # 最小面积
        area_form = QFormLayout()
        area_form.setContentsMargins(4, 4, 4, 4)
        self.spin_min_area = QSpinBox()
        self.spin_min_area.setRange(10, 5000)
        self.spin_min_area.setValue(100)
        self.spin_min_area.setSingleStep(10)
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

        self.lbl_original = VideoLabel("原图")
        self.lbl_segmented = VideoLabel("前景提取")
        self.lbl_binary = VideoLabel("二值掩码")

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
        self.lbl_status.setMinimumWidth(300)
        self.lbl_status.setStyleSheet("color: #888; padding: 4px;")
        bottom_layout.addWidget(self.lbl_status)

        root_layout.addLayout(bottom_layout)

        # 初始状态：视频模式控件可见，屏幕ROI控件隐藏
        self._update_mode_visibility()

    # ------------------------------------------------------------------
    # 信号连接
    # ------------------------------------------------------------------

    def _connect_signals(self):
        self.combo_mode.currentTextChanged.connect(self._on_mode_changed)
        self.btn_open.clicked.connect(self._on_open_video)
        self.btn_play.clicked.connect(self._on_play)
        self.btn_pause.clicked.connect(self._on_pause)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_select_roi.clicked.connect(self._on_select_roi)

        self.combo_method.currentTextChanged.connect(self._on_params_changed)
        self.spin_min_area.valueChanged.connect(self._on_params_changed)

        self.slider_frame.sliderMoved.connect(self._on_seek)

    # ------------------------------------------------------------------
    # 模式切换
    # ------------------------------------------------------------------

    def _update_mode_visibility(self):
        """根据当前模式更新控件可见性。"""
        is_video = self._current_mode == "video"
        is_screen_roi = self._current_mode == "screen_roi"

        # 视频模式控件
        self.btn_open.setVisible(is_video)
        self.lbl_file.setVisible(is_video)
        self.slider_frame.setVisible(is_video)

        # 屏幕ROI模式控件
        self.btn_select_roi.setVisible(is_screen_roi)
        self.lbl_roi.setVisible(is_screen_roi)

        # 播放控制
        if is_video:
            self.btn_play.setEnabled(bool(self._video_path))
        elif is_screen_roi:
            self.btn_play.setEnabled(self._screen_roi is not None)

    def _on_mode_changed(self, mode: str):
        """模式切换。"""
        # 停止当前工作
        self._stop_all_workers()

        self._current_mode = mode
        self._update_mode_visibility()

        # 更新窗口标题
        if mode == "video":
            self.setWindowTitle("视频前后景分离工具 - 视频模式")
            self.lbl_status.setText("已切换到视频模式")
        else:
            self.setWindowTitle("视频前后景分离工具 - 屏幕ROI模式")
            self.lbl_status.setText("已切换到屏幕ROI模式，请先选择屏幕ROI区域")

        # 清空显示
        self.lbl_original.setText("原图")
        self.lbl_segmented.setText("前景提取")
        self.lbl_binary.setText("二值掩码")

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

        self.btn_play.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)

        self._start_video_worker()

    def _on_select_roi(self):
        """选择屏幕ROI区域。"""
        try:
            import pyautogui
        except ImportError:
            QMessageBox.warning(
                self, "依赖缺失", "需要安装 pyautogui：pip install pyautogui"
            )
            return

        QMessageBox.information(
            self, "准备截图",
            "点击确定后，将在 3 秒后截取全屏。\n请切换到目标窗口并准备好划取ROI。",
        )

        for i in range(3, 0, -1):
            self.lbl_status.setText(f"截图倒计时: {i}...")
            QApplication.processEvents()
            time.sleep(1)

        screenshot = pyautogui.screenshot()
        screenshot_rgb = np.array(screenshot)
        screenshot_bgr = cv2.cvtColor(screenshot_rgb, cv2.COLOR_RGB2BGR)

        dialog = ScreenRoiSelectDialog(screenshot_bgr, self)
        dialog.resize(800, 600)

        if dialog.exec_() == QDialog.Accepted:
            roi = dialog.get_roi()
            if roi is not None:
                x, y, w, h = roi
                self._screen_roi = roi
                self.lbl_roi.setText(f"ROI: ({x}, {y}, {w}, {h})")
                self.lbl_roi.setStyleSheet("color: #ddd;")
                self.lbl_status.setText(f"已设置屏幕ROI: ({x}, {y}, {w}, {h})")

                self.btn_play.setEnabled(True)

    def _start_video_worker(self):
        """启动视频处理线程。"""
        self._stop_all_workers()

        method = self.combo_method.currentText()
        min_area = self.spin_min_area.value()

        self._video_worker = VideoWorker(self._video_path, method, min_area)
        self._video_worker.frame_ready.connect(self._on_video_frame_ready)
        self._video_worker.finished.connect(self._on_video_worker_finished)
        self._video_worker.start()

        self.btn_play.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.slider_frame.setEnabled(True)
        self.lbl_status.setText("视频播放中...")

    def _start_screen_roi_worker(self):
        """启动屏幕ROI处理线程。"""
        self._stop_all_workers()

        if self._screen_roi is None:
            QMessageBox.warning(self, "错误", "请先选择屏幕ROI区域")
            return

        method = self.combo_method.currentText()
        min_area = self.spin_min_area.value()

        self._screen_roi_worker = ScreenRoiWorker(
            self._screen_roi, method, min_area, fps=10.0
        )
        self._screen_roi_worker.frame_ready.connect(self._on_screen_roi_frame_ready)
        self._screen_roi_worker.finished.connect(self._on_screen_roi_worker_finished)
        self._screen_roi_worker.start()

        self.btn_play.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.lbl_status.setText("屏幕ROI实时处理中...")

    def _stop_all_workers(self):
        """停止所有工作线程。"""
        if self._video_worker is not None:
            self._video_worker.request_stop()
            self._video_worker.wait(3000)
            self._video_worker = None

        if self._screen_roi_worker is not None:
            self._screen_roi_worker.request_stop()
            self._screen_roi_worker.wait(3000)
            self._screen_roi_worker = None

    def _on_play(self):
        if self._current_mode == "video":
            if self._video_worker is None:
                self._start_video_worker()
            else:
                self._video_worker.set_paused(False)
                self.btn_pause.setEnabled(True)
                self.btn_play.setEnabled(False)
                self.lbl_status.setText("视频播放中...")
        else:
            if self._screen_roi_worker is None:
                self._start_screen_roi_worker()
            else:
                self._screen_roi_worker.set_paused(False)
                self.btn_pause.setEnabled(True)
                self.btn_play.setEnabled(False)
                self.lbl_status.setText("屏幕ROI实时处理中...")

    def _on_pause(self):
        if self._current_mode == "video":
            if self._video_worker is not None:
                self._video_worker.set_paused(True)
                self.btn_play.setEnabled(True)
                self.btn_pause.setEnabled(False)
                self.lbl_status.setText("视频已暂停")
        else:
            if self._screen_roi_worker is not None:
                self._screen_roi_worker.set_paused(True)
                self.btn_play.setEnabled(True)
                self.btn_pause.setEnabled(False)
                self.lbl_status.setText("屏幕ROI处理已暂停")

    def _on_stop(self):
        self._stop_all_workers()

        if self._current_mode == "video":
            self.btn_play.setEnabled(bool(self._video_path))
            self.slider_frame.setEnabled(False)
        else:
            self.btn_play.setEnabled(self._screen_roi is not None)

        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText("已停止")

    def _on_params_changed(self):
        """参数变化时实时更新到工作线程。"""
        method = self.combo_method.currentText()
        min_area = self.spin_min_area.value()

        if self._video_worker is not None:
            self._video_worker.update_params(method, min_area)
        if self._screen_roi_worker is not None:
            self._screen_roi_worker.update_params(method, min_area)

    def _on_seek(self, value: int):
        """拖动进度条跳转到指定帧（仅视频模式）。"""
        if self._video_worker is not None:
            self._video_worker.request_seek(value)

    def _on_video_frame_ready(self, original, segmented, binary, frame_idx, total, angle_deg):
        """视频模式：收到新帧，更新显示区域。"""
        self.lbl_original.update_frame(original)
        self.lbl_segmented.update_frame(segmented)
        self.lbl_binary.update_frame(binary)

        if total > 0:
            if self.slider_frame.maximum() != total:
                self.slider_frame.setRange(0, total)
            self.slider_frame.setValue(frame_idx)

            status = f"[视频模式] 帧 {frame_idx}/{total}"
            if angle_deg > 0:
                status += f"  |  直线夹角: {angle_deg:.1f}°"
            status += f"  |  方法: {self.combo_method.currentText()}"
            self.lbl_status.setText(status)

    def _on_video_worker_finished(self):
        """视频模式工作线程结束。"""
        self.lbl_status.setText("视频播放结束")

    def _on_screen_roi_frame_ready(self, original, segmented, binary, angle_deg):
        """屏幕ROI模式：收到新帧，更新显示区域。"""
        self.lbl_original.update_frame(original)
        self.lbl_segmented.update_frame(segmented)
        self.lbl_binary.update_frame(binary)

        status = "[屏幕ROI模式] 实时处理中"
        if angle_deg > 0:
            status += f"  |  直线夹角: {angle_deg:.1f}°"
        status += f"  |  方法: {self.combo_method.currentText()}"
        self.lbl_status.setText(status)

    def _on_screen_roi_worker_finished(self):
        """屏幕ROI模式工作线程结束。"""
        self.lbl_status.setText("屏幕ROI处理已停止")

    def closeEvent(self, event):
        """关闭窗口时释放资源。"""
        self._stop_all_workers()
        super().closeEvent(event)


# ============================================================
# 入口
# ============================================================

def main():
    """主入口。"""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # 深色主题
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
            win._start_video_worker()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
