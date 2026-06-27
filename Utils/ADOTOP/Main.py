from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QPoint, QRect
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QPalette
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "YoloUI") not in sys.path:
    sys.path.insert(0, str(ROOT / "YoloUI"))

from YoloUI.yolo_seg_gui import MainWindow as YoloMainWindow
from ForegroundAndBackground.video_seg import segment_foreground_background, clean_binary


# ============================================================
# 通用图像工具
# ============================================================

def cv2_to_qpixmap(cv_img):
    if cv_img is None:
        return QPixmap()
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qt_img)


def detect_longest_line_single(frame: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=30, maxLineGap=10)
    if lines is None or len(lines) == 0:
        return None, 0.0

    best = None
    best_len = 0.0
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        length = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
        if length > best_len:
            best_len = length
            best = (x1, y1, x2, y2)

    if best is None:
        return None, 0.0

    x1, y1, x2, y2 = best
    angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
    if angle > 90:
        angle = 180 - angle
    return best, angle


def draw_line_on_frame(frame: np.ndarray, line: Tuple[int, int, int, int], angle: float) -> np.ndarray:
    out = frame.copy()
    x1, y1, x2, y2 = line
    cv2.line(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(out, f"{angle:.1f} deg", (max(0, x1), max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return out


# ============================================================
# 前后景分离 + 直线检测
# ============================================================

class StableLineDetector:
    def __init__(self, ema_alpha: float = 0.3, jump_threshold: float = 5.0):
        self.ema_alpha = ema_alpha
        self.jump_threshold = jump_threshold
        self._stable_line: Optional[Tuple[int, int, int, int]] = None
        self._stable_angle = 0.0

    def reset(self):
        self._stable_line = None
        self._stable_angle = 0.0

    def detect(self, frame: np.ndarray):
        current_line, current_angle = detect_longest_line_single(frame)
        if current_line is None:
            return self._stable_line, self._stable_angle

        if self._stable_line is None:
            self._stable_line = current_line
            self._stable_angle = current_angle
            return self._stable_line, self._stable_angle

        diff = abs(current_angle - self._stable_angle)
        if diff > 90:
            diff = 180 - diff
        if diff > self.jump_threshold:
            return self._stable_line, self._stable_angle

        x1, y1, x2, y2 = current_line
        sx1, sy1, sx2, sy2 = self._stable_line
        self._stable_line = (
            int(self.ema_alpha * x1 + (1 - self.ema_alpha) * sx1),
            int(self.ema_alpha * y1 + (1 - self.ema_alpha) * sy1),
            int(self.ema_alpha * x2 + (1 - self.ema_alpha) * sx2),
            int(self.ema_alpha * y2 + (1 - self.ema_alpha) * sy2),
        )
        self._stable_angle = self.ema_alpha * current_angle + (1 - self.ema_alpha) * self._stable_angle
        return self._stable_line, self._stable_angle


def segment_frame(frame: np.ndarray, method: str, min_area: int):
    fg, binary = segment_foreground_background(frame, method=method, min_area=min_area), None
    if fg is None or fg.size == 0:
        return frame, frame

    if method == "bright":
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 35, 55]), np.array([179, 255, 255]))
    elif method == "otsu":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "blue":
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([85, 40, 45]), np.array([130, 255, 255]))
    elif method == "yellow":
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([10, 45, 55]), np.array([45, 255, 255]))
    else:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 35, 55]), np.array([179, 255, 255]))

    mask = clean_binary(mask, min_area=min_area)
    mask_3ch = cv2.merge([mask, mask, mask])
    fg = cv2.bitwise_and(frame, mask_3ch)
    binary = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return fg, binary


class ForegroundWorker(QThread):
    frame_ready = pyqtSignal(object, object, object, float)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, mode: str, video_path: str = "", roi: Optional[Tuple[int, int, int, int]] = None, method: str = "bright", min_area: int = 100):
        super().__init__()
        self.mode = mode
        self.video_path = video_path
        self.roi = roi
        self.method = method
        self.min_area = min_area
        self._running = True
        self._paused = False

    def stop(self):
        self._running = False

    def pause(self, paused: bool):
        self._paused = paused

    def run(self):
        detector = StableLineDetector()
        try:
            if self.mode == "video":
                cap = cv2.VideoCapture(self.video_path)
                if not cap.isOpened():
                    self.error.emit("无法打开视频文件")
                    return
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                interval_ms = max(1, int(1000 / fps))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                while self._running:
                    if self._paused:
                        self.msleep(50)
                        continue
                    ret, frame = cap.read()
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        detector.reset()
                        continue
                    fg, binary = segment_frame(frame, self.method, self.min_area)
                    line, angle = detector.detect(fg)
                    if line is not None:
                        fg = draw_line_on_frame(fg, line, angle)
                    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    self.frame_ready.emit(frame, fg, binary, angle)
                    self.msleep(interval_ms)
                cap.release()
            else:
                try:
                    import pyautogui
                except Exception:
                    self.error.emit("屏幕ROI模式需要安装 pyautogui")
                    return
                if self.roi is None:
                    self.error.emit("请先选择屏幕ROI")
                    return
                x, y, w, h = self.roi
                interval_ms = 100
                while self._running:
                    if self._paused:
                        self.msleep(50)
                        continue
                    try:
                        shot = pyautogui.screenshot(region=(x, y, w, h))
                        frame = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)
                    except Exception:
                        self.msleep(interval_ms)
                        continue
                    fg, binary = segment_frame(frame, self.method, self.min_area)
                    line, angle = detector.detect(fg)
                    if line is not None:
                        fg = draw_line_on_frame(fg, line, angle)
                    self.frame_ready.emit(frame, fg, binary, angle)
                    self.msleep(interval_ms)
        except Exception as e:
            self.error.emit(str(e))
        self.finished.emit()


# ============================================================
# ROI 选择
# ============================================================

class RoiSelectLabel(QLabel):
    roi_selected = pyqtSignal(tuple)

    def __init__(self, image_bgr: np.ndarray):
        super().__init__()
        self.image_bgr = image_bgr
        self._start_pos: Optional[QPoint] = None
        self._end_pos: Optional[QPoint] = None
        self._drawing = False
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setStyleSheet("border: 2px solid #666;")
        self._update_display()

    def _update_display(self):
        h, w = self.image_bgr.shape[:2]
        rgb = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        if self._start_pos and self._end_pos:
            painter = QPainter(scaled)
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            rect = QRect(self._start_pos, self._end_pos).normalized()
            painter.drawRect(rect)
            painter.end()
        self.setPixmap(scaled)

    def _pos_to_img(self, pos: QPoint):
        pix = self.pixmap()
        if pix is None:
            return 0, 0
        offset_x = (self.width() - pix.width()) // 2
        offset_y = (self.height() - pix.height()) // 2
        lx = pos.x() - offset_x
        ly = pos.y() - offset_y
        h, w = self.image_bgr.shape[:2]
        ix = max(0, min(int(lx * w / max(1, pix.width())), w - 1))
        iy = max(0, min(int(ly * h / max(1, pix.height())), h - 1))
        return ix, iy

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._start_pos = event.pos()
            self._end_pos = event.pos()
            self._drawing = True

    def mouseMoveEvent(self, event):
        if self._drawing:
            self._end_pos = event.pos()
            self._update_display()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._drawing:
            self._end_pos = event.pos()
            self._drawing = False
            x1, y1 = self._pos_to_img(self._start_pos)
            x2, y2 = self._pos_to_img(self._end_pos)
            x, y = min(x1, x2), min(y1, y2)
            w, h = abs(x2 - x1), abs(y2 - y1)
            if w > 10 and h > 10:
                self.roi_selected.emit((x, y, w, h))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()


class ScreenRoiDialog(QWidget):
    def __init__(self, screenshot: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("划取屏幕ROI区域")
        self.screenshot = screenshot
        self.roi_rect: Optional[Tuple[int, int, int, int]] = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("拖拽鼠标划取ROI区域，然后点击确定"))
        self.image_label = RoiSelectLabel(self.screenshot)
        self.image_label.roi_selected.connect(self._on_roi_selected)
        layout.addWidget(self.image_label, 1)
        btns = QHBoxLayout()
        self.btn_ok = QPushButton("确定")
        self.btn_ok.setEnabled(False)
        self.btn_cancel = QPushButton("取消")
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)
        btns.addStretch()
        btns.addWidget(self.btn_ok)
        btns.addWidget(self.btn_cancel)
        layout.addLayout(btns)

    def _on_roi_selected(self, rect):
        self.roi_rect = rect
        self.btn_ok.setEnabled(True)

    def accept(self):
        super().accept()

    def reject(self):
        super().reject()


# ============================================================
# Foreground/Edge 通用显示
# ============================================================

class ImageView(QLabel):
    def __init__(self, title: str = ""):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setStyleSheet("QLabel { background-color: #1a1a2e; border: 2px solid #444; border-radius: 4px; color: #ccc; }")
        if title:
            self.setText(title)

    def set_frame(self, frame_bgr: np.ndarray):
        if frame_bgr is None or frame_bgr.size == 0:
            return
        self.setPixmap(cv2_to_qpixmap(frame_bgr).scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))


# ============================================================
# 前后景融合页
# ============================================================

class ForegroundPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._video_path = ""
        self._roi = None
        self._worker: Optional[ForegroundWorker] = None
        self._build_ui()
        self._connect()

    def _build_ui(self):
        root = QVBoxLayout(self)

        cfg = QGroupBox("参数配置")
        fl = QHBoxLayout(cfg)

        self.cb_mode = QComboBox()
        self.cb_mode.addItems(["video", "screen_roi"])

        self.btn_open = QPushButton("选择视频")
        self.lbl_file = QLabel("未选择")
        self.btn_select_roi = QPushButton("选择屏幕ROI")
        self.lbl_roi = QLabel("未设置")
        self.cb_method = QComboBox()
        self.cb_method.addItems(["bright", "otsu", "blue", "yellow"])
        self.sb_min_area = QSpinBox()
        self.sb_min_area.setRange(10, 5000)
        self.sb_min_area.setValue(100)

        self.btn_start = QPushButton("▶ 播放")
        self.btn_pause = QPushButton("⏸ 暂停")
        self.btn_stop = QPushButton("⏹ 停止")
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)

        for widget, label in [
            (self.cb_mode, "模式"),
            (self.btn_open, "视频"),
            (self.btn_select_roi, "ROI"),
            (self.cb_method, "分割"),
            (self.sb_min_area, "最小面积"),
            (self.btn_start, "开始"),
            (self.btn_pause, "暂停"),
            (self.btn_stop, "停止"),
        ]:
            box = QVBoxLayout()
            box.addWidget(QLabel(label))
            box.addWidget(widget)
            fl.addLayout(box)
        file_box = QVBoxLayout(); file_box.addWidget(QLabel("视频文件")); file_box.addWidget(self.lbl_file); fl.addLayout(file_box)
        roi_box = QVBoxLayout(); roi_box.addWidget(QLabel("屏幕ROI")); roi_box.addWidget(self.lbl_roi); fl.addLayout(roi_box)
        root.addWidget(cfg)

        views = QHBoxLayout()
        self.lbl_original = ImageView("原图")
        self.lbl_segmented = ImageView("前景提取")
        self.lbl_binary = ImageView("二值掩码")
        views.addWidget(self.lbl_original)
        views.addWidget(self.lbl_segmented)
        views.addWidget(self.lbl_binary)
        root.addLayout(views, 1)

        self.lbl_status = QLabel("就绪")
        root.addWidget(self.lbl_status)

    def _connect(self):
        self.btn_open.clicked.connect(self._open_video)
        self.btn_select_roi.clicked.connect(self._select_roi)
        self.btn_start.clicked.connect(self._start)
        self.btn_pause.clicked.connect(self._pause)
        self.btn_stop.clicked.connect(self._stop)
        self.cb_mode.currentTextChanged.connect(self._update_mode_ui)

    def _update_mode_ui(self):
        is_video = self.cb_mode.currentText() == "video"
        self.btn_open.setVisible(is_video)
        self.lbl_file.setVisible(is_video)
        self.btn_select_roi.setVisible(not is_video)
        self.lbl_roi.setVisible(not is_video)
        self.btn_start.setEnabled(bool(self._video_path) if is_video else self._roi is not None)

    def _open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", str(ROOT / "ForegroundAndBackground" / "Video"), "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*.*)")
        if path:
            self._video_path = path
            self.lbl_file.setText(Path(path).name)
            self.btn_start.setEnabled(True)
            self.lbl_status.setText(f"已加载视频: {Path(path).name}")

    def _select_roi(self):
        try:
            import pyautogui
        except Exception:
            QMessageBox.warning(self, "依赖缺失", "需要安装 pyautogui")
            return
        QMessageBox.information(self, "提示", "点击确定后将截图并进入 ROI 划取")
        QApplication.processEvents()
        time.sleep(1)
        shot = pyautogui.screenshot()
        img = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)
        dlg = ScreenRoiDialog(img, self)
        dlg.resize(1000, 700)
        if dlg.exec() == QWidget.DialogCode.Accepted and dlg.roi_rect is not None:
            self._roi = dlg.roi_rect
            x, y, w, h = self._roi
            self.lbl_roi.setText(f"({x}, {y}, {w}, {h})")
            self.btn_start.setEnabled(True)
            self.lbl_status.setText("ROI 已设置")

    def _start(self):
        self._stop()
        mode = self.cb_mode.currentText()
        self._worker = ForegroundWorker(
            mode=mode,
            video_path=self._video_path,
            roi=self._roi,
            method=self.cb_method.currentText(),
            min_area=self.sb_min_area.value(),
        )
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.lbl_status.setText("处理中...")

    def _pause(self):
        if self._worker is None:
            return
        paused = self.btn_pause.text() == "⏸ 暂停"
        self._worker.pause(paused)
        self.btn_pause.setText("▶ 继续" if paused else "⏸ 暂停")
        self.lbl_status.setText("已暂停" if paused else "处理中...")

    def _stop(self):
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait(2000)
            self._worker = None
        self.btn_start.setEnabled(bool(self._video_path) if self.cb_mode.currentText() == "video" else self._roi is not None)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("⏸ 暂停")
        self.btn_stop.setEnabled(False)

    def _on_frame(self, original, segmented, binary, angle_deg):
        self.lbl_original.set_frame(original)
        self.lbl_segmented.set_frame(segmented)
        self.lbl_binary.set_frame(binary)
        self.lbl_status.setText(f"处理中 | 直线角度 {angle_deg:.1f}° | 方法 {self.cb_method.currentText()}")

    def _on_error(self, msg):
        self._stop()
        QMessageBox.critical(self, "处理错误", msg)

    def _on_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText("已停止")


# ============================================================
# 边缘检测页
# ============================================================

class EdgeDetectPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._path = ""
        self._build_ui()
        self._connect()

    def _build_ui(self):
        root = QVBoxLayout(self)
        cfg = QGroupBox("边缘检测")
        fl = QHBoxLayout(cfg)
        self.le_path = QLineEdit()
        self.btn_open = QPushButton("选择图片/视频")
        self.btn_run = QPushButton("开始检测")
        self.cb_source = QComboBox()
        self.cb_source.addItems(["image", "video_first_frame"])
        self.lbl_angle = QLabel("角度: -")
        fl.addWidget(self.cb_source)
        fl.addWidget(self.le_path)
        fl.addWidget(self.btn_open)
        fl.addWidget(self.btn_run)
        fl.addWidget(self.lbl_angle)
        root.addWidget(cfg)

        views = QHBoxLayout()
        self.lbl_orig = ImageView("原图")
        self.lbl_edge = ImageView("边缘/最长直线")
        views.addWidget(self.lbl_orig)
        views.addWidget(self.lbl_edge)
        root.addLayout(views, 1)

        self.tb_log = QTextBrowser()
        self.tb_log.setMinimumHeight(120)
        root.addWidget(self.tb_log)

    def _connect(self):
        self.btn_open.clicked.connect(self._open)
        self.btn_run.clicked.connect(self._run)

    def _open(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片或视频", str(ROOT), "媒体文件 (*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mkv);;所有文件 (*.*)")
        if path:
            self._path = path
            self.le_path.setText(path)

    def _run(self):
        path = self.le_path.text().strip()
        if not path or not Path(path).exists():
            QMessageBox.warning(self, "提示", "请选择有效文件")
            return
        if self.cb_source.currentText() == "image" or Path(path).suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
            frame = cv2.imread(path)
            if frame is None:
                QMessageBox.warning(self, "提示", "无法读取图片")
                return
        else:
            cap = cv2.VideoCapture(path)
            ok, frame = cap.read()
            cap.release()
            if not ok:
                QMessageBox.warning(self, "提示", "无法读取视频首帧")
                return
        line, angle = detect_longest_line_single(frame)
        out = frame.copy()
        if line is not None:
            out = draw_line_on_frame(out, line, angle)
        self.lbl_orig.set_frame(frame)
        self.lbl_edge.set_frame(out)
        self.lbl_angle.setText(f"角度: {angle:.1f}°")
        self.tb_log.append(f"检测完成: {Path(path).name} | 角度 {angle:.1f}°")


# ============================================================
# 统一主窗口
# ============================================================

class UnifiedMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("统一视觉系统 - YOLO / 前后景分离 / 边缘检测")
        self.resize(1600, 1000)

        self._yolo_host = YoloMainWindow()
        yolo_widget = self._yolo_host.centralWidget()
        if yolo_widget is not None:
            yolo_widget.setParent(None)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.tabs.addTab(yolo_widget, "YOLO 分割")
        self.tabs.addTab(ForegroundPage(), "前后景分离")
        self.tabs.addTab(EdgeDetectPage(), "边缘检测")


# ============================================================
# 入口
# ============================================================

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 46))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(205, 214, 244))
    palette.setColor(QPalette.ColorRole.Base, QColor(49, 50, 68))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(49, 50, 68))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(205, 214, 244))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(205, 214, 244))
    palette.setColor(QPalette.ColorRole.Text, QColor(205, 214, 244))
    palette.setColor(QPalette.ColorRole.Button, QColor(49, 50, 68))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(205, 214, 244))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(137, 180, 250))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(30, 30, 46))
    app.setPalette(palette)

    win = UnifiedMainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
