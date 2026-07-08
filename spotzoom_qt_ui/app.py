from __future__ import annotations

import json
import random
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .qt_compat import (
    QAbstractTableModel,
    QModelIndex,
    QObject,
    Qt,
    QProcess,
    QTimer,
    QAction,
    QImage,
    QPixmap,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSizePolicy,
    Signal,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    app_exec,
)

from dataclasses import dataclass, field


@dataclass
class UiPreferences:
    theme_dark: bool = True
    autozoom_window_titles: List[str] = field(default_factory=list)


from .models import AlignmentStrategy, EventRecord, RunMode, RuntimeProfile, TestCaseSpec, UiStatus
from .services import (
    DeviceRegistryService,
    DeviceTestService,
    ModuleCatalogService,
    RuntimeControlService,
)
from .picomotor_driver_panel import PicomotorDriverPanel


import numpy as np
import cv2
from SpotZoom import UCCFrameSource
from collections import deque

try:
    from PySide6.QtGui import QPainter, QPen, QColor, QFont, QCursor
    from PySide6.QtCore import QPointF, QRect
except ImportError:
    from PySide2.QtGui import QPainter, QPen, QColor, QFont, QCursor
    from PySide2.QtCore import QPointF, QRect



class AutoZoomWindowComboBox(QComboBox):
    """点击下拉时自动刷新当前可见窗口列表的下拉框。"""

    popupRequested = Signal()

    def showPopup(self) -> None:
        self.popupRequested.emit()
        super().showPopup()


class AutoZoomRoiPreviewLabel(QLabel):
    """
    支持在参考帧上鼠标拖拽框选 ROI 的预览标签。

    - 显示抓拍的参考帧或实时 ROI 画面；
    - 只有在 reference_mode=True 时才允许框选；
    - 框选完成后发出 roiSelected(rect)，rect 为 (x, y, w, h)。
    """

    roiSelected = Signal(int, int, int, int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setStyleSheet(
            "background-color: #1a1a2e; color: #888; border: 1px solid #333; "
            "border-radius: 4px; font-size: 14px;"
        )
        self.setMouseTracking(True)
        self._pixmap: Optional[QPixmap] = None
        self._reference_mode = False
        self._drawing = False
        self._start_pos: Optional[Tuple[int, int]] = None
        self._roi: Optional[QRect] = None
        self._scale = 1.0
        self._offset: Tuple[int, int] = (0, 0)

    def set_reference_mode(self, enabled: bool) -> None:
        """切换为参考帧模式（允许框选）或实时预览模式。"""
        self._reference_mode = enabled
        if enabled:
            self.setCursor(QCursor(Qt.CrossCursor))
        else:
            self.setCursor(QCursor(Qt.ArrowCursor))
        self.update()

    def set_preview_pixmap(self, pixmap: QPixmap) -> None:
        """设置要显示的 QPixmap（已按标签大小缩放）。"""
        self._pixmap = pixmap
        self.update()

    def clear_preview(self) -> None:
        self._pixmap = None
        self._roi = None
        self.update()

    def set_roi(self, x: int, y: int, w: int, h: int) -> None:
        self._roi = QRect(int(x), int(y), int(w), int(h))
        self.update()

    def get_roi(self) -> Optional[Tuple[int, int, int, int]]:
        if self._roi is None:
            return None
        return (self._roi.x(), self._roi.y(), self._roi.width(), self._roi.height())

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#1a1a2e"))

        if self._pixmap is not None and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
            self._scale = scaled.width() / max(1, self._pixmap.width())
            self._offset = (x, y)
        else:
            self._scale = 1.0
            self._offset = (0, 0)

        if self._reference_mode and self._roi is not None:
            pen = QPen(QColor("#22D3EE"))
            pen.setWidth(2)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            ox, oy = self._offset
            roi_display = QRect(
                int(ox + self._roi.x() * self._scale),
                int(oy + self._roi.y() * self._scale),
                int(self._roi.width() * self._scale),
                int(self._roi.height() * self._scale),
            )
            painter.drawRect(roi_display)

    def mousePressEvent(self, event) -> None:
        if not self._reference_mode or self._pixmap is None:
            super().mousePressEvent(event)
            return
        if event.button() == Qt.LeftButton:
            self._drawing = True
            ox, oy = self._offset
            x = int((event.pos().x() - ox) / self._scale)
            y = int((event.pos().y() - oy) / self._scale)
            x = max(0, min(x, self._pixmap.width() - 1))
            y = max(0, min(y, self._pixmap.height() - 1))
            self._roi = QRect(x, y, 0, 0)
            self._start_pos = (x, y)
            self.update()

    def mouseMoveEvent(self, event) -> None:
        if not self._reference_mode or not self._drawing or self._pixmap is None:
            super().mouseMoveEvent(event)
            return
        ox, oy = self._offset
        x = int((event.pos().x() - ox) / self._scale)
        y = int((event.pos().y() - oy) / self._scale)
        x = max(0, min(x, self._pixmap.width() - 1))
        y = max(0, min(y, self._pixmap.height() - 1))
        sx, sy = self._start_pos
        self._roi = QRect(min(sx, x), min(sy, y), abs(x - sx), abs(y - sy))
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.LeftButton or not self._reference_mode or not self._drawing:
            super().mouseReleaseEvent(event)
            return
        self._drawing = False
        if self._roi is not None and self._roi.width() > 2 and self._roi.height() > 2:
            self.roiSelected.emit(
                self._roi.x(), self._roi.y(), self._roi.width(), self._roi.height()
            )
        else:
            self._roi = None
            self.update()


STATUS_STYLE = {
    UiStatus.NORMAL: "color: #9FB3C8;",
    UiStatus.RUNNING: "color: #60A5FA;",
    UiStatus.SUCCESS: "color: #34D399;",
    UiStatus.FAILED: "color: #F87171;",
    UiStatus.DISCONNECTED: "color: #F59E0B;",
    UiStatus.DISABLED: "color: #6B7280;",
    UiStatus.WARNING: "color: #FBBF24;",
    UiStatus.ARMED: "color: #22D3EE;",
}


def status_text(status: UiStatus) -> str:
    return status.value.upper()


def analyze_spot(frame: np.ndarray) -> Optional[dict]:
    """分析单帧光斑参数（轻量级，用于实时预览）。

    Returns
    -------
    dict or None
        {
            "min_val": float,
            "peak": float,
            "peak_loc": (px, py),
            "centroid": (cx, cy),
            "width": (wx, wy),   # 二阶矩标准差
        }
    """
    if frame is None or frame.size == 0:
        return None

    # 转灰度
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        gray = frame.astype(np.float64)

    min_val = float(gray.min())
    peak = float(gray.max())

    # 如果画面太暗，无法检测
    if peak < 10:
        return None

    # 峰值位置
    py, px = np.unravel_index(np.argmax(gray), gray.shape)
    peak_loc = (int(px), int(py))

    # 背景扣除（底部10%百分位）
    bg = np.percentile(gray, 10.0)
    signal = np.maximum(gray - bg, 0.0)
    total = signal.sum()
    if total < 1e-6:
        return None

    h, w = gray.shape
    yy, xx = np.mgrid[:h, :w]

    # 灰度加权质心
    cx = float(np.sum(xx * signal) / total)
    cy = float(np.sum(yy * signal) / total)

    # 二阶矩宽度 (X/Y 方向标准差)
    dx = xx - cx
    dy = yy - cy
    var_x = float(np.sum(dx * dx * signal) / total)
    var_y = float(np.sum(dy * dy * signal) / total)
    wx = float(np.sqrt(max(var_x, 0.0)))
    wy = float(np.sqrt(max(var_y, 0.0)))

    return {
        "min_val": min_val,
        "peak": peak,
        "peak_loc": peak_loc,
        "centroid": (cx, cy),
        "width": (wx, wy),
    }


def compute_spot_profile(frame: np.ndarray, cx: float, cy: float) -> Optional[Dict]:
    """计算光斑在质心位置的水平/垂直 RGB 剖面图。

    Parameters
    ----------
    frame : np.ndarray
        BGR 格式图像。
    cx, cy : float
        光斑质心坐标。

    Returns
    -------
    dict or None
        {
            "horizontal": {"x": np.ndarray, "r": ..., "g": ..., "b": ...},
            "vertical":   {"y": np.ndarray, "r": ..., "g": ..., "b": ...},
        }
    """
    if frame is None or frame.size == 0:
        return None
    if len(frame.shape) != 3 or frame.shape[2] < 3:
        return None

    h, w = frame.shape[:2]
    cx_int = int(round(cx))
    cy_int = int(round(cy))
    cx_int = max(0, min(cx_int, w - 1))
    cy_int = max(0, min(cy_int, h - 1))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 水平线：y = cy_int，x 从 0 到 w-1
    horizontal = rgb[cy_int, :].astype(np.float64)
    # 垂直线：x = cx_int，y 从 0 到 h-1
    vertical = rgb[:, cx_int].astype(np.float64)

    return {
        "horizontal": {
            "x": np.arange(w),
            "r": horizontal[:, 0],
            "g": horizontal[:, 1],
            "b": horizontal[:, 2],
        },
        "vertical": {
            "y": np.arange(h),
            "r": vertical[:, 0],
            "g": vertical[:, 1],
            "b": vertical[:, 2],
        },
    }


class SpotCurveWidget(QWidget):
    """光斑位置历史曲线绘制控件（XYCurve）。"""

    def __init__(self, max_points: int = 200, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.max_points = max_points
        self.x_history: deque = deque(maxlen=max_points)
        self.y_history: deque = deque(maxlen=max_points)
        self.setMinimumSize(200, 120)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

    def append(self, x: float, y: float) -> None:
        self.x_history.append(x)
        self.y_history.append(y)
        self.update()

    def clear_history(self) -> None:
        self.x_history.clear()
        self.y_history.clear()
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        mid_y = h // 2

        # 背景
        painter.fillRect(self.rect(), QColor("#1a1a2e"))

        # 网格
        pen_grid = QPen(QColor("#333"))
        pen_grid.setWidth(1)
        painter.setPen(pen_grid)
        for i in range(0, w, 40):
            painter.drawLine(i, 0, i, h)
        for i in range(0, h, 20):
            painter.drawLine(0, i, w, i)

        # 中线
        pen_axis = QPen(QColor("#555"))
        pen_axis.setWidth(1)
        painter.setPen(pen_axis)
        painter.drawLine(0, mid_y, w, mid_y)

        if len(self.x_history) < 2:
            painter.end()
            return

        # 计算显示范围
        all_vals = list(self.x_history) + list(self.y_history)
        if not all_vals:
            painter.end()
            return
        max_val = max(abs(v) for v in all_vals) if all_vals else 1.0
        margin = max(max_val * 0.2, 1.0)
        y_max = max_val + margin
        y_min = -(max_val + margin)
        y_range = y_max - y_min if y_max != y_min else 1.0

        # 绘制 X 曲线 (青色)
        pen_x = QPen(QColor("#22D3EE"))
        pen_x.setWidth(2)
        painter.setPen(pen_x)
        pts_x = []
        n = len(self.x_history)
        for i, val in enumerate(self.x_history):
            px = int((i / max(n - 1, 1)) * (w - 2)) + 1
            py = int(h - 1 - ((val - y_min) / y_range) * (h - 2))
            pts_x.append(QPointF(px, py))
        for i in range(len(pts_x) - 1):
            painter.drawLine(pts_x[i], pts_x[i + 1])

        # 绘制 Y 曲线 (品红)
        pen_y = QPen(QColor("#F472B6"))
        pen_y.setWidth(2)
        painter.setPen(pen_y)
        pts_y = []
        for i, val in enumerate(self.y_history):
            px = int((i / max(n - 1, 1)) * (w - 2)) + 1
            py = int(h - 1 - ((val - y_min) / y_range) * (h - 2))
            pts_y.append(QPointF(px, py))
        for i in range(len(pts_y) - 1):
            painter.drawLine(pts_y[i], pts_y[i + 1])

        # 图例
        font = QFont("Microsoft YaHei", 8)
        painter.setFont(font)
        painter.setPen(QColor("#22D3EE"))
        painter.drawText(8, 16, "X 位置")
        painter.setPen(QColor("#F472B6"))
        painter.drawText(60, 16, "Y 位置")

        painter.end()


class SpotProfileWidget(QWidget):
    """光斑 RGB 剖面图绘制控件（水平线 y=cy 与垂直线 x=cx）。"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.profile: Optional[Dict] = None
        self.setMinimumSize(360, 240)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

    def set_profile(self, profile: Optional[Dict]) -> None:
        self.profile = profile
        self.update()

    def clear_profile(self) -> None:
        self.profile = None
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        painter.fillRect(self.rect(), QColor("#1a1a2e"))

        if self.profile is None:
            painter.setPen(QColor("#888"))
            font = QFont("Microsoft YaHei", 10)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignCenter, "无剖面数据")
            painter.end()
            return

        margin = 28
        gap = 12
        plot_h = max(40, (h - margin * 2 - gap) // 2)
        plot_w = max(80, w - margin * 2)

        self._draw_sub_profile(
            painter,
            margin,
            margin,
            plot_w,
            plot_h,
            self.profile.get("horizontal", {}),
            "水平剖面 (y=质心)",
            "x (像素)",
        )
        self._draw_sub_profile(
            painter,
            margin,
            margin + plot_h + gap,
            plot_w,
            plot_h,
            self.profile.get("vertical", {}),
            "垂直剖面 (x=质心)",
            "y (像素)",
        )

        painter.end()

    def _draw_sub_profile(
        self,
        painter: QPainter,
        px: int,
        py: int,
        pw: int,
        ph: int,
        data: Dict,
        title: str,
        x_label: str,
    ) -> None:
        painter.fillRect(px, py, pw, ph, QColor("#111118"))
        pen_border = QPen(QColor("#333"))
        pen_border.setWidth(1)
        painter.setPen(pen_border)
        painter.drawRect(px, py, pw, ph)

        pen_grid = QPen(QColor("#222"))
        pen_grid.setWidth(1)
        painter.setPen(pen_grid)
        for i in range(px, px + pw, max(1, pw // 8)):
            painter.drawLine(i, py, i, py + ph)
        for i in range(py, py + ph, max(1, ph // 4)):
            painter.drawLine(px, i, px + pw, i)

        axis_key = "x" if "x" in data else "y"
        xs = data.get(axis_key)
        if xs is None or len(xs) < 2:
            painter.setPen(QColor("#888"))
            font = QFont("Microsoft YaHei", 9)
            painter.setFont(font)
            painter.drawText(px, py, pw, ph, Qt.AlignCenter, "数据不足")
            return

        r = data.get("r")
        g = data.get("g")
        b = data.get("b")
        if r is None or g is None or b is None:
            return

        max_val = max(
            float(np.max(r)), float(np.max(g)), float(np.max(b)), 1.0
        )
        margin_y = max_val * 0.1
        y_max = max_val + margin_y
        y_min = -margin_y
        y_range = y_max - y_min if y_max != y_min else 1.0
        n = len(xs)

        colors = {
            "r": QColor("#FF5252"),
            "g": QColor("#69F0AE"),
            "b": QColor("#448AFF"),
        }
        labels = {"r": "R", "g": "G", "b": "B"}

        for ch in ("r", "g", "b"):
            vals = data.get(ch)
            if vals is None or len(vals) != n:
                continue
            pen = QPen(colors[ch])
            pen.setWidth(1)
            painter.setPen(pen)
            pts = []
            for i in range(n):
                x = px + int((i / max(n - 1, 1)) * (pw - 1))
                y = py + ph - 1 - int(
                    ((float(vals[i]) - y_min) / y_range) * (ph - 1)
                )
                pts.append(QPointF(x, y))
            for i in range(len(pts) - 1):
                painter.drawLine(pts[i], pts[i + 1])

        painter.setPen(QColor("#CCC"))
        font = QFont("Microsoft YaHei", 8)
        painter.setFont(font)
        painter.drawText(px + 4, py + 13, title)
        painter.drawText(px + pw - 90, py + ph - 4, x_label)
        painter.drawText(px + 4, py + ph - 4, f"max={max_val:.0f}")

        legend_x = px + pw - 60
        legend_y = py + 13
        for idx, ch in enumerate(("r", "g", "b")):
            painter.setPen(colors[ch])
            painter.drawText(legend_x + idx * 18, legend_y, labels[ch])


class ErrorDistributionWidget(QWidget):
    """收敛误差分布可视化控件：时间序列曲线 + 直方图。"""

    def __init__(self, max_points: int = 500, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.max_points = max_points
        # (timestamp, dist) tuples
        self.error_history: deque = deque(maxlen=max_points)
        self.setMinimumSize(200, 150)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

    def append_dist(self, dist: float) -> None:
        self.error_history.append((time.time(), dist))
        self.update()

    def clear_history(self) -> None:
        self.error_history.clear()
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # 背景
        painter.fillRect(self.rect(), QColor("#1a1a2e"))

        # 网格
        pen_grid = QPen(QColor("#333"))
        pen_grid.setWidth(1)
        painter.setPen(pen_grid)
        for i in range(0, w, 40):
            painter.drawLine(i, 0, i, h)
        for i in range(0, h, 20):
            painter.drawLine(0, i, w, i)

        n = len(self.error_history)
        if n < 2:
            font = QFont("Microsoft YaHei", 10)
            painter.setFont(font)
            painter.setPen(QColor("#9FB3C8"))
            painter.drawText(self.rect(), Qt.AlignCenter, "等待记录数据...")
            painter.end()
            return

        dists = [d for _, d in self.error_history]
        min_d = min(dists)
        max_d = max(dists)
        d_range = max_d - min_d if max_d != min_d else 1.0
        margin = d_range * 0.15
        y_min = max(0.0, min_d - margin)
        y_max = max_d + margin
        y_range = y_max - y_min if y_max != y_min else 1.0

        chart_top = 20
        chart_bottom = h - 2
        chart_left = 2
        chart_right = w - 2
        chart_h = chart_bottom - chart_top
        chart_w = chart_right - chart_left

        # ========== 上部分：时间序列曲线 ==========
        curve_bottom = chart_top + chart_h // 2 - 4
        curve_h = curve_bottom - chart_top

        pen_curve = QPen(QColor("#FBBF24"))
        pen_curve.setWidth(2)
        painter.setPen(pen_curve)
        pts = []
        for i, d in enumerate(dists):
            px = chart_left + int((i / max(n - 1, 1)) * chart_w)
            py = curve_bottom - int(((d - y_min) / y_range) * curve_h)
            pts.append(QPointF(px, py))
        for i in range(len(pts) - 1):
            painter.drawLine(pts[i], pts[i + 1])

        # 标注文字
        font = QFont("Microsoft YaHei", 8)
        painter.setFont(font)
        painter.setPen(QColor("#FBBF24"))
        painter.drawText(chart_left + 4, chart_top + 12, f"总误差 ─ 当前 {dists[-1]:.3f}px")

        # ========== 下部分：直方图 ==========
        hist_top = curve_bottom + 4
        hist_bottom = chart_bottom
        hist_h = hist_bottom - hist_top

        # 自动分 bin（最多 15 个）
        num_bins = min(15, max(5, n // 3))
        bin_w = chart_w / num_bins
        bin_edges = [min_d + i * d_range / num_bins for i in range(num_bins + 1)]
        bin_counts = [0] * num_bins
        for d in dists:
            idx = min(num_bins - 1, int((d - min_d) / d_range * num_bins))
            bin_counts[idx] += 1
        max_count = max(bin_counts) if max(bin_counts) > 0 else 1

        pen_bar = QPen(QColor("#34D399"))
        pen_bar.setWidth(1)
        brush_bar = QColor(52, 211, 153, 80)
        for i, count in enumerate(bin_counts):
            bar_h = int((count / max_count) * (hist_h - 2))
            bar_x = chart_left + int(i * bin_w) + 1
            bar_w = max(int(bin_w) - 2, 2)
            bar_y = hist_bottom - bar_h
            painter.fillRect(bar_x, bar_y, bar_w, bar_h, brush_bar)
            painter.setPen(pen_bar)
            painter.drawRect(bar_x, bar_y, bar_w, bar_h)

        # 直方图标注
        painter.setPen(QColor("#34D399"))
        painter.drawText(chart_left + 4, hist_top + 12,
                         f"分布 (n={n} μ={sum(dists)/n:.3f} σ={self._std(dists):.3f})")

        painter.end()

    @staticmethod
    def _std(vals: List[float]) -> float:
        n = len(vals)
        if n < 2:
            return 0.0
        mean = sum(vals) / n
        return (sum((x - mean) ** 2 for x in vals) / (n - 1)) ** 0.5


class RunEventStreamModel(QAbstractTableModel):
    HEADERS = ["时间", "事件", "摘要"]

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._rows: List[EventRecord] = []

    def set_rows(self, rows: List[EventRecord]) -> None:
        self.beginResetModel()
        self._rows = rows
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self.HEADERS)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal and 0 <= section < len(self.HEADERS):
            return self.HEADERS[section]
        return str(section + 1)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole:
            return None
        row = self._rows[index.row()]
        col = index.column()
        if col == 0:
            return row.timestamp
        if col == 1:
            return row.event_name
        return row.payload_summary


class SpotZoomQtMainWindow(QMainWindow):
    NAV_ITEMS = [
        "仪表盘",
        "准直工作台",
        "模块中心",
        "AutoZoom",
        "设备中心",
        "设备测试",
        "运行模式",
        "仿真实验室",
        "日志与报告",
        "设置",
    ]

    def __init__(self, repo_root: Optional[Path] = None):
        super().__init__()
        self.runtime = RuntimeControlService(repo_root=repo_root)
        self.device_registry = DeviceRegistryService()
        self.device_test = DeviceTestService(self.runtime)
        self.module_catalog = ModuleCatalogService()
        self.profile = self.runtime.default_profile()
        self.ui_prefs = self._load_ui_preferences()

        self.env_status = UiStatus.NORMAL
        self.startup_status = UiStatus.NORMAL
        self.run_status = UiStatus.NORMAL

        self.run_process: Optional[QProcess] = None
        self.module_rows = []

        self.controls: Dict[str, QWidget] = {}
        self.status_fields: Dict[str, QLabel] = {}
        self.dashboard_fields: Dict[str, QLabel] = {}
        self.runtime_fields: Dict[str, QLabel] = {}

        # UCC 实时预览
        self._ucc_preview_source: Optional[UCCFrameSource] = None
        self._ucc_preview_timer: Optional[QTimer] = None
        self._ucc_preview_running: bool = False
        self._ucc_preview_fps_counter: int = 0
        self._ucc_preview_fps_time: float = 0.0
        self._ucc_preview_actual_fps: float = 0.0

        # 准直工作台 UCC 预览 + 闭环控制
        self._alignment_ucc_source: Optional[UCCFrameSource] = None
        self._alignment_ucc_timer: Optional[QTimer] = None
        self._alignment_ucc_running: bool = False
        self._alignment_ucc_fps_counter: int = 0
        self._alignment_ucc_fps_time: float = 0.0
        self._alignment_ucc_actual_fps: float = 0.0
        self._alignment_target_point: Optional[Tuple[float, float]] = None
        self._alignment_last_centroid: Optional[Tuple[float, float]] = None
        self._alignment_last_centroid_ts: float = 0.0
        self._alignment_centroid_history: List[Tuple[float, float]] = []
        self._alignment_stabilizing: bool = False
        self._alignment_stabilize_timer: Optional[QTimer] = None
        self._alignment_jitter_timer: Optional[QTimer] = None
        self._alignment_jitter_running: bool = False
        self._alignment_precision_samples: List[Tuple[float, float, float]] = []
        self._alignment_convergence_record_timer: Optional[QTimer] = None
        self._alignment_convergence_record_rows: List[List[object]] = []
        self._alignment_convergence_record_start_ts: float = 0.0
        self._alignment_convergence_record_end_ts: float = 0.0
        self._alignment_converged_centroid: Optional[Tuple[float, float]] = None
        # 4轴→探测器 映射标定
        self._alignment_calibrated: bool = False
        self._alignment_calib_active: bool = False  # 正在标定中
        self._alignment_calib_steps_per_px_x: float = 1.0  # X方向 步/像素
        self._alignment_calib_steps_per_px_y: float = 1.0  # Y方向 步/像素
        self._alignment_calib_direction_x: int = 1
        self._alignment_calib_direction_y: int = 1

        self.setWindowTitle("SpotZoom 主动激光束稳定控制台")
        self.resize(1680, 980)
        self._build_ui()
        self._apply_profile_to_controls(self.profile)
        self.btn_axis4_enter.setEnabled(True)
        self.axis4_status_label.setText(
            f"状态：4轴闭环已激活 | 策略={self.profile.alignment_strategy.value} | 探测器模式={self.profile.detector_mode}"
        )
        self.statusBar().showMessage("就绪")
        self._update_status_bar()

        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(1500)
        self.poll_timer.timeout.connect(self._refresh_runtime_streams)
        self.poll_timer.start()

    def _build_ui(self) -> None:
        root = QWidget(self)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)
        root_layout.addWidget(self._build_top_status_strip())

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_nav_panel())
        splitter.addWidget(self._build_pages_stack())
        splitter.addWidget(self._build_side_console())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([240, 1040, 360])
        root_layout.addWidget(splitter)
        self.setCentralWidget(root)
        self.nav_list.setCurrentRow(0)

        self._build_menu_actions()
        self._theme_dark = True
        self._apply_app_style()
        self._apply_theme_style()

    def _build_menu_actions(self) -> None:
        toolbar = self.addToolBar("main")
        toolbar.setMovable(False)
        act_refresh = QAction("刷新", self)
        act_refresh.triggered.connect(self._refresh_all_panels)
        toolbar.addAction(act_refresh)
        act_export = QAction("导出报告", self)
        act_export.triggered.connect(self._export_report)
        toolbar.addAction(act_export)
        act_theme = QAction("切换主题", self)
        act_theme.triggered.connect(self._toggle_theme)
        toolbar.addAction(act_theme)

    def _apply_app_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget { background: #11161C; color: #D9E2EC; font-size: 11px; }
            QGroupBox { border: 1px solid #2B3642; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #9FB3C8; }
            QPushButton { background: #1B2530; border: 1px solid #334155; padding: 4px 10px; border-radius: 3px; }
            QPushButton:hover { background: #243244; }
            QPushButton:disabled { color: #6B7280; border-color: #374151; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit, QTableWidget, QListWidget, QTreeWidget {
              background: #0F141A; border: 1px solid #2B3642; color: #D9E2EC;
            }
            QHeaderView::section { background: #18212B; color: #9FB3C8; border: 1px solid #2B3642; padding: 3px; }
            """
        )

    def _apply_light_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget { background: #F6F8FB; color: #1F2937; font-size: 11px; }
            QGroupBox { border: 1px solid #CBD5E1; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #334155; }
            QPushButton { background: #FFFFFF; border: 1px solid #94A3B8; padding: 4px 10px; border-radius: 3px; }
            QPushButton:hover { background: #E2E8F0; }
            QPushButton:disabled { color: #94A3B8; border-color: #CBD5E1; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit, QTableWidget, QListWidget, QTreeWidget {
              background: #FFFFFF; border: 1px solid #CBD5E1; color: #1F2937;
            }
            QHeaderView::section { background: #E2E8F0; color: #334155; border: 1px solid #CBD5E1; padding: 3px; }
            """
        )

    def _apply_theme_style(self) -> None:
        if self._theme_dark:
            self._apply_app_style()
        else:
            self._apply_light_style()

    def _toggle_theme(self) -> None:
        self._theme_dark = not getattr(self, "_theme_dark", True)
        self._apply_theme_style()

    def _build_top_status_strip(self) -> QWidget:
        bar = QFrame()
        layout = QGridLayout(bar)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setHorizontalSpacing(8)
        specs = [
            ("运行模式", "mode"),
            ("检测后端", "backend"),
            ("XY 驱动", "xy"),
            ("Z 驱动", "z"),
            ("设备总状态", "devices"),
            ("运行中", "run_lock"),
            ("环境检查", "env"),
            ("启动自检", "startup"),
            ("运行状态", "run"),
        ]
        for idx, (label, key) in enumerate(specs):
            left = QLabel(f"{label}:")
            right = QLabel("-")
            right.setObjectName(f"status_{key}")
            self.status_fields[key] = right
            layout.addWidget(left, 0, idx * 2)
            layout.addWidget(right, 0, idx * 2 + 1)
        return bar

    def _build_nav_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        self.nav_list = QListWidget()
        for item in self.NAV_ITEMS:
            self.nav_list.addItem(QListWidgetItem(item))
        self.nav_list.currentRowChanged.connect(self._switch_page)
        layout.addWidget(self.nav_list)
        return panel

    def _build_pages_stack(self) -> QWidget:
        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_dashboard_page())
        self.pages.addWidget(self._build_alignment_page())
        self.pages.addWidget(self._build_module_page())
        self.pages.addWidget(self._build_autozoom_page())
        self.pages.addWidget(self._build_device_center_page())
        self.pages.addWidget(self._build_device_test_page())
        self.pages.addWidget(self._build_run_modes_page())
        self.pages.addWidget(self._build_simulation_lab_page())
        self.pages.addWidget(self._build_logs_report_page())
        self.pages.addWidget(self._build_settings_page())
        return self.pages

    def _build_side_console(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.runtime_status_text = QTextEdit()
        self.runtime_status_text.setReadOnly(True)
        self.runtime_status_text.setMinimumHeight(220)
        layout.addWidget(QLabel("运行事件面板"))
        layout.addWidget(self.runtime_status_text, 2)

        self.quick_event_table = QTableWidget(0, 3)
        self.quick_event_table.setHorizontalHeaderLabels(["时间", "事件", "摘要"])
        self.quick_event_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(QLabel("最近关键事件"))
        layout.addWidget(self.quick_event_table, 3)
        return panel

    def _build_dashboard_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        top = QGridLayout()
        fields = [
            ("运行模式", "mode"),
            ("请求后端", "requested_backend"),
            ("实际后端", "resolved_backend"),
            ("XY 驱动", "xy_driver"),
            ("Z 驱动", "z_driver"),
            ("ROI 状态", "roi"),
            ("目标状态", "target"),
            ("设备总览", "device"),
            ("环境检查", "env"),
            ("启动自检", "startup"),
            ("最近运行结果", "run"),
            ("风险提示", "risk"),
        ]
        for idx, (label, key) in enumerate(fields):
            top.addWidget(QLabel(f"{label}"), idx // 4, (idx % 4) * 2)
            v = QLabel("-")
            self.dashboard_fields[key] = v
            top.addWidget(v, idx // 4, (idx % 4) * 2 + 1)
        layout.addLayout(top)

        btn_row = QHBoxLayout()
        btn_start = QPushButton("开始准直")
        btn_start.clicked.connect(self._start_alignment)
        btn_test = QPushButton("设备测试")
        btn_test.clicked.connect(lambda: self.nav_list.setCurrentRow(5))
        btn_module = QPushButton("模块配置")
        btn_module.clicked.connect(lambda: self.nav_list.setCurrentRow(2))
        btn_row.addWidget(btn_start)
        btn_row.addWidget(btn_test)
        btn_row.addWidget(btn_module)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self.dashboard_event_table = QTableWidget(0, 3)
        self.dashboard_event_table.setHorizontalHeaderLabels(["时间", "事件", "摘要"])
        self.dashboard_event_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(QLabel("最近20 条关键事件"))
        layout.addWidget(self.dashboard_event_table, 1)
        return page

    def _build_alignment_page(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setSpacing(8)

        # ==================================================================
        # 左侧：实时图像区 — UCC 探测器 + 显微镜画面
        # ==================================================================
        image_group = QGroupBox("实时图像区")
        image_layout = QVBoxLayout(image_group)

        dual_view_layout = QHBoxLayout()
        dual_view_layout.setSpacing(10)

        # UCC 探测器画面
        ucc_view = QVBoxLayout()
        self._alignment_ucc_label = QLabel("UCC 探测器\n点击「启动UCC预览」")
        self._alignment_ucc_label.setAlignment(Qt.AlignCenter)
        self._alignment_ucc_label.setMinimumSize(360, 260)
        self._alignment_ucc_label.setStyleSheet(
            "background-color: #1a1a2e; color: #888; border: 1px solid #333; "
            "border-radius: 4px; font-size: 13px;"
        )
        ucc_view.addWidget(QLabel("UCC 探测器"))
        ucc_view.addWidget(self._alignment_ucc_label, 1)
        dual_view_layout.addLayout(ucc_view, 1)

        # 显微镜画面（预留）
        micro_view = QVBoxLayout()
        self._alignment_micro_label = QLabel("显微镜图像\n待通信接入")
        self._alignment_micro_label.setAlignment(Qt.AlignCenter)
        self._alignment_micro_label.setMinimumSize(280, 260)
        self._alignment_micro_label.setStyleSheet(
            "background-color: #1a1a2e; color: #666; border: 1px solid #333; "
            "border-radius: 4px; font-size: 13px;"
        )
        micro_view.addWidget(QLabel("显微镜 (ToupView/NIS)"))
        micro_view.addWidget(self._alignment_micro_label, 1)
        dual_view_layout.addLayout(micro_view, 1)

        image_layout.addLayout(dual_view_layout)

        # 图像叠加信息栏
        self.image_overlay_label = QLabel("质心: - | 目标点: - | 偏差: - | 状态: 未启动")
        self.image_overlay_label.setStyleSheet("color: #9FB3C8; font-size: 12px;")
        image_layout.addWidget(self.image_overlay_label)

        # UCC 控制行
        ucc_ctrl_row = QHBoxLayout()
        self._alignment_ucc_device = self._spin(0, 9, 1)
        self._alignment_ucc_device.setPrefix("设备 ")
        self._alignment_ucc_resolution = self._combo(["PAL", "NTSC", "AUTO"])
        self._alignment_ucc_resolution.setCurrentText("AUTO")
        self._alignment_ucc_format = self._combo(["AUTO", "MJPG", "YUY2", "YUYV", "UYVY"])
        self._alignment_ucc_format.setCurrentText("AUTO")

        self._alignment_ucc_btn_start = QPushButton("▶ 启动UCC预览")
        self._alignment_ucc_btn_start.clicked.connect(self._start_alignment_ucc_preview)
        self._alignment_ucc_btn_stop = QPushButton("■ 停止UCC预览")
        self._alignment_ucc_btn_stop.clicked.connect(self._stop_alignment_ucc_preview)
        self._alignment_ucc_btn_stop.setEnabled(False)

        ucc_ctrl_row.addWidget(QLabel("设备:"))
        ucc_ctrl_row.addWidget(self._alignment_ucc_device)
        ucc_ctrl_row.addWidget(QLabel("分辨率:"))
        ucc_ctrl_row.addWidget(self._alignment_ucc_resolution)
        ucc_ctrl_row.addWidget(QLabel("格式:"))
        ucc_ctrl_row.addWidget(self._alignment_ucc_format)
        ucc_ctrl_row.addWidget(self._alignment_ucc_btn_start)
        ucc_ctrl_row.addWidget(self._alignment_ucc_btn_stop)
        ucc_ctrl_row.addStretch(1)
        image_layout.addLayout(ucc_ctrl_row)
        self._alignment_ucc_path_info = QLabel("调试帧目录: Tmp_Frames / FrameTmp | 预览状态: 未启动")
        self._alignment_ucc_path_info.setStyleSheet("color: #8BA3B8; font-size: 12px;")
        image_layout.addWidget(self._alignment_ucc_path_info)

        layout.addWidget(image_group, 3)

        # ==================================================================
        # 右侧：控制区 + 参数区 + 曲线 + 参数表 + 状态区
        # ==================================================================
        right = QWidget()
        right_layout = QVBoxLayout(right)

        # --- 4轴闭环入口与状态显示 ---
        axis4_group = QGroupBox("4轴闭环入口")
        axis4_layout = QGridLayout(axis4_group)
        axis4_layout.setHorizontalSpacing(10)
        axis4_layout.setVerticalSpacing(8)
        self.btn_axis4_enter = QPushButton("进入4轴闭环")
        self.btn_axis4_enter.clicked.connect(self._enter_alignment_4axis_mode)
        self.btn_axis4_enter.setToolTip("将闭环策略切换为 dual_detector_4axis，并联动探测器模式")
        self.btn_set_target = QPushButton("🎯 设定目标点")
        self.btn_set_target.clicked.connect(self._set_alignment_target_point)
        self.btn_set_target.setEnabled(False)
        self.btn_calibrate = QPushButton("📏 标定映射")
        self.btn_calibrate.clicked.connect(self._calibrate_axis_mapping)
        self.btn_calibrate.setEnabled(False)
        self.btn_calibrate.setToolTip("逐轴移动+测量质心偏移，标定4轴步长与像素的映射关系")
        self.btn_stop_calibrate = QPushButton("⏸ 停止标定")
        self.btn_stop_calibrate.clicked.connect(self._stop_alignment_calibration)
        self.btn_stop_calibrate.setEnabled(False)
        self.btn_jitter = QPushButton("🌀 模拟抖动")
        self.btn_jitter.clicked.connect(self._simulate_alignment_jitter)
        self.btn_jitter.setEnabled(False)
        self.btn_record_convergence = QPushButton("📝 收敛误差记录")
        self.btn_record_convergence.clicked.connect(self._start_convergence_error_recording)
        self.btn_record_convergence.setEnabled(True)
        self.btn_stop_record_convergence = QPushButton("⏹ 停止记录")
        self.btn_stop_record_convergence.clicked.connect(lambda: self._stop_convergence_error_recording(auto_export=True))
        self.btn_stop_record_convergence.setEnabled(False)
        self.btn_stabilize = QPushButton("🔄 开始稳定闭环")
        self.btn_stabilize.clicked.connect(self._start_alignment_stabilization)
        self.btn_stabilize.setEnabled(False)
        self.btn_stop_stabilize = QPushButton("⏹ 停止闭环")
        self.btn_stop_stabilize.clicked.connect(self._stop_alignment_stabilization)
        self.btn_stop_stabilize.setEnabled(False)
        axis4_layout.addWidget(self.btn_axis4_enter, 0, 0, 1, 2)
        axis4_layout.addWidget(self.btn_set_target, 1, 0)
        axis4_layout.addWidget(self.btn_calibrate, 1, 1)
        axis4_layout.addWidget(self.btn_stop_calibrate, 2, 0)
        axis4_layout.addWidget(self.btn_stabilize, 2, 1)
        axis4_layout.addWidget(self.btn_jitter, 3, 0)
        axis4_layout.addWidget(self.btn_record_convergence, 3, 1)
        axis4_layout.addWidget(self.btn_stop_record_convergence, 4, 0)
        axis4_layout.addWidget(self.btn_stop_stabilize, 4, 1)
        axis4_layout.addWidget(QLabel("  "), 5, 0, 1, 2)  # spacing
        self._alignment_calib_info_label = QLabel()
        self._alignment_calib_info_label.setStyleSheet("color: #93C5FD; font-size: 11px;")
        self._alignment_calib_info_label.setWordWrap(True)
        axis4_layout.addWidget(self._alignment_calib_info_label, 1, 2, 2, 1)
        self.axis4_status_label = QLabel("状态：未进入4轴闭环")
        self.axis4_status_label.setStyleSheet("color: #9FB3C8; font-size: 12px;")
        axis4_layout.addWidget(self.axis4_status_label, 4, 0, 1, 2)
        self.axis4_calib_label = QLabel("标定：未标定")
        self.axis4_calib_label.setStyleSheet("color: #F59E0B; font-size: 11px;")
        axis4_layout.addWidget(self.axis4_calib_label, 5, 0, 1, 2)
        # 校正镜选择行
        axis4_layout.addWidget(QLabel("校正镜组:"), 6, 0)
        self._alignment_correction_mirror = QComboBox()
        self._alignment_correction_mirror.addItems(["mirror2", "mirror1", "both"])
        self._alignment_correction_mirror.setToolTip(
            "选择本探测器对应的校正镜:\n"
            "  mirror2 — Mirror2 校正本探测器（默认）\n"
            "  mirror1 — Mirror1 校正本探测器\n"
            "  both    — 两个镜子同时参与"
        )
        axis4_layout.addWidget(self._alignment_correction_mirror, 6, 1)
        axis4_layout.addWidget(QLabel("  "), 7, 0, 1, 2)  # spacing
        right_layout.addWidget(axis4_group)

        # --- 抖动参数 ---
        jitter_group = QGroupBox("抖动与闭环参数")
        jitter_form = QFormLayout(jitter_group)
        self._alignment_jitter_amplitude = self._spin(5, 2000, 50)
        self._alignment_jitter_interval = self._dspin(0.1, 5.0, 0.5, 2)
        self._alignment_stabilize_kp = self._dspin(0.01, 2.0, 0.3, 2)
        self._alignment_stabilize_tolerance = self._spin(1, 50, 5)
        self._alignment_stabilize_interval = self._spin(50, 5000, 150)
        self._alignment_stabilize_interval.setToolTip("稳定闭环每次纠偏的间隔，单位 ms")
        self._alignment_spot_loss_timeout = self._dspin(0.5, 60.0, 5.0, 1)
        self._alignment_spot_loss_timeout.setToolTip("光斑消失超过此时间(秒)自动停止闭环，保护电机")
        self._alignment_pixel_size_um = self._dspin(0.01, 1000.0, 2.5, 2)
        self._alignment_detector_distance_mm = self._dspin(1.0, 10000.0, 100.0, 1)
        self._alignment_pixel_size_um.setToolTip("用于将 px 误差换算为 μm 位置精度")
        self._alignment_detector_distance_mm.setToolTip("两个探测器间距，用于将位置差估算为 μrad 指向精度")
        self._alignment_record_duration_min = self._dspin(0.1, 2880.0, 60.0, 1)
        self._alignment_record_export_path = QLineEdit(str(self.runtime.repo_root / "artifacts" / "convergence_error_record.xlsx"))
        self._alignment_record_browse_btn = QPushButton("选择路径")
        self._alignment_record_browse_btn.clicked.connect(self._select_convergence_record_export_path)
        self._alignment_record_status_label = QLabel("记录状态：未开始")
        self._alignment_record_status_label.setStyleSheet("color: #9FB3C8; font-size: 11px;")
        self._alignment_record_exported_path: str = ""
        self._alignment_calib_step = self._spin(1, 5000, 500)
        self._alignment_calib_interval = self._spin(100, 10000, 2000)
        self._alignment_calib_step.setToolTip("标定单次移动步数")
        self._alignment_calib_interval.setToolTip("标定状态机 tick 间隔，单位 ms")
        jitter_form.addRow("标定步长 (步)", self._alignment_calib_step)
        jitter_form.addRow("标定间隔 (ms)", self._alignment_calib_interval)
        self._alignment_calib_step.valueChanged.connect(self._update_alignment_calib_info_label)
        self._alignment_calib_interval.valueChanged.connect(self._update_alignment_calib_info_label)
        self._update_alignment_calib_info_label()
        jitter_form.addRow("闭环 Kp 增益", self._alignment_stabilize_kp)
        jitter_form.addRow("收敛容差 (px)", self._alignment_stabilize_tolerance)
        jitter_form.addRow("闭环间隔 (ms)", self._alignment_stabilize_interval)
        jitter_form.addRow("光斑丢失超时 (s)", self._alignment_spot_loss_timeout)
        jitter_form.addRow("像素尺寸 (μm/px)", self._alignment_pixel_size_um)
        jitter_form.addRow("探测器间距 (mm)", self._alignment_detector_distance_mm)
        jitter_form.addRow("记录时长 (min)", self._alignment_record_duration_min)
        record_path_wrap = QWidget()
        record_path_layout = QHBoxLayout(record_path_wrap)
        record_path_layout.setContentsMargins(0, 0, 0, 0)
        record_path_layout.addWidget(self._alignment_record_export_path)
        record_path_layout.addWidget(self._alignment_record_browse_btn)
        jitter_form.addRow("导出路径 (.xlsx)", record_path_wrap)
        jitter_form.addRow("记录状态", self._alignment_record_status_label)
        right_layout.addWidget(jitter_group)

        # --- 稳定精度统计 ---
        precision_group = QGroupBox("稳定精度统计")
        precision_layout = QGridLayout(precision_group)
        precision_layout.setHorizontalSpacing(10)
        precision_layout.setVerticalSpacing(6)
        self._alignment_precision_fields: Dict[str, QLabel] = {}
        precision_items = [
            ("samples", "采样数"),
            ("rms", "RMS(px)"),
            ("p95", "P95(px)"),
            ("max", "Max(px)"),
            ("x_std", "X_std(px)"),
            ("y_std", "Y_std(px)"),
            ("rms_um", "RMS(μm)"),
            ("p95_um", "P95(μm)"),
            ("max_um", "Max(μm)"),
            ("pointing_p95", "指向P95(μrad)"),
            ("linear", "线性区"),
            ("range", "范围状态"),
        ]
        for idx, (key, label) in enumerate(precision_items):
            row = idx // 2
            col = (idx % 2) * 2
            precision_layout.addWidget(QLabel(label), row, col)
            value_label = QLabel("--")
            value_label.setStyleSheet("color: #E5E7EB; font-weight: 600;")
            self._alignment_precision_fields[key] = value_label
            precision_layout.addWidget(value_label, row, col + 1)
        right_layout.addWidget(precision_group)

        # --- 收敛误差分布可视化 ---
        error_dist_group = QGroupBox("收敛误差分布")
        error_dist_layout = QVBoxLayout(error_dist_group)
        self._alignment_error_dist_widget = ErrorDistributionWidget(max_points=500)
        error_dist_layout.addWidget(self._alignment_error_dist_widget)
        right_layout.addWidget(error_dist_group)

        # --- XY 曲线 ---
        curve_group = QGroupBox("XY 位置曲线 (XYCurve)")
        curve_layout = QVBoxLayout(curve_group)
        self._alignment_curve_widget = SpotCurveWidget(max_points=200)
        curve_layout.addWidget(self._alignment_curve_widget)
        right_layout.addWidget(curve_group, 1)

        # --- 光斑参数表 ---
        param_group = QGroupBox("光斑参数 (Calcult)")
        param_layout = QVBoxLayout(param_group)
        self._alignment_param_table = QTableWidget()
        self._alignment_param_table.setColumnCount(2)
        self._alignment_param_table.setHorizontalHeaderLabels(["参数", "数值"])
        self._alignment_param_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._alignment_param_table.setSelectionMode(QTableWidget.NoSelection)
        self._alignment_param_table.verticalHeader().setVisible(False)
        self._alignment_param_table.horizontalHeader().setStretchLastSection(True)
        self._alignment_param_table.setMaximumHeight(120)
        param_names = ["Min, Peak", "Peak Loc (X, Y)", "Centroid (X, Y)", "Width (X, Y)"]
        self._alignment_param_table.setRowCount(len(param_names))
        for i, name in enumerate(param_names):
            self._alignment_param_table.setItem(i, 0, QTableWidgetItem(name))
            self._alignment_param_table.setItem(i, 1, QTableWidgetItem("--"))
        param_layout.addWidget(self._alignment_param_table)
        right_layout.addWidget(param_group)

        # --- 关键参数区 ---
        key_param_group = QGroupBox("关键参数区")
        key_param_layout = QGridLayout(key_param_group)
        key_param_layout.setHorizontalSpacing(10)
        key_param_layout.setVerticalSpacing(6)
        key_param_layout.setContentsMargins(6, 8, 6, 8)
        self.controls["tolerance_px"] = self._spin(1, 300, 6)
        self.controls["detect_retry"] = self._spin(1, 20, 6)
        self.controls["detect_retry_interval"] = self._dspin(0.01, 10.0, 0.25, 2)
        self.controls["settle_time"] = self._dspin(0.0, 10.0, 0.35, 2)
        self.controls["max_align_rounds"] = self._spin(1, 500, 60)
        self.controls["max_iterations"] = self._spin(1, 1000, 8)
        self.controls["x_move_step"] = self._spin(1, 10000, 500)
        self.controls["y_move_step"] = self._spin(1, 10000, 500)
        self.controls["z_step"] = self._dspin(0.1, 100.0, 1.0, 2)
        self.controls["min_focus_score"] = self._dspin(0.0, 10000.0, 20.0, 2)
        self.controls["adaptive_step"] = QCheckBox()
        self.controls["enable_recovery_scan"] = QCheckBox()
        self.controls["startup_motion_check_enabled"] = QCheckBox()
        self.controls["startup_motion_check_timeout"] = self._dspin(0.1, 20.0, 1.0, 2)
        self.controls["startup_motion_check_xy_steps"] = self._spin(1, 50, 1)
        self.controls["startup_motion_check_z_step"] = self._dspin(0.1, 20.0, 1.0, 2)
        self.controls["frame_cache_enabled"] = QCheckBox()
        self.controls["disable_z_axis"] = QCheckBox()
        pairs = [
            ("tolerance_px", "容差 (px)"),
            ("detect_retry", "检测重试次数"),
            ("detect_retry_interval", "检测重试间隔 (s)"),
            ("settle_time", "稳定时间 (s)"),
            ("max_align_rounds", "最大对准轮次"),
            ("max_iterations", "最大迭代次数"),
            ("x_move_step", "X 移动步长"),
            ("y_move_step", "Y 移动步长"),
            ("z_step", "Z 步长"),
            ("min_focus_score", "最小焦点评分"),
            ("adaptive_step", "启用自适应步长"),
            ("enable_recovery_scan", "启用恢复扫描"),
            ("startup_motion_check_enabled", "启用启动自检"),
            ("startup_motion_check_timeout", "启动自检超时 (s)"),
            ("startup_motion_check_xy_steps", "启动自检 XY 步数"),
            ("startup_motion_check_z_step", "启动自检 Z 步长"),
            ("frame_cache_enabled", "启用帧缓存 (./Tmp_Frames)"),
            ("disable_z_axis", "不启用 Z 轴 (4轴模式)"),
        ]
        for idx, (key, label) in enumerate(pairs):
            widget = self.controls[key]
            row = idx % 9
            col = (idx // 9) * 2
            key_param_layout.addWidget(QLabel(label), row, col)
            key_param_layout.addWidget(widget, row, col + 1)
        right_layout.addWidget(key_param_group)

        # --- 过程状态区 ---
        runtime_group = QGroupBox("过程状态区")
        runtime_layout = QFormLayout(runtime_group)
        for key, label in [
            ("round", "当前轮次"),
            ("phase", "当前阶段"),
            ("dxdy", "当前 dx,dy"),
            ("actions", "本轮动作"),
            ("steps", "累计步数"),
            ("latency", "设备响应"),
            ("detect_cost", "检测耗时"),
            ("converged", "收敛状态"),
            ("risk", "风险告警"),
        ]:
            v = QLabel("-")
            self.runtime_fields[key] = v
            runtime_layout.addRow(label, v)
        right_layout.addWidget(runtime_group, 1)

        # 右侧滚动区，避免控件过多时超出屏幕
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.NoFrame)
        right_scroll.setWidget(right)
        layout.addWidget(right_scroll, 3)
        return page

    def _build_module_page(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.module_type_filter = QListWidget()
        self.module_type_filter.addItem("全部类型")
        self.module_type_filter.currentRowChanged.connect(self._refresh_module_table)
        left_layout.addWidget(QLabel("类型筛选"))
        left_layout.addWidget(self.module_type_filter)
        layout.addWidget(left, 1)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        filter_row = QHBoxLayout()
        self.module_search = QLineEdit()
        self.module_search.setPlaceholderText("搜索模块名标题")
        self.module_search.textChanged.connect(self._refresh_module_table)
        self.module_version_filter = QComboBox()
        self.module_version_filter.addItems(["全部版本", "v2", "v3", "v4", "v5", "v6", "v7", "v8"])
        self.module_version_filter.currentIndexChanged.connect(self._refresh_module_table)
        self.module_integrated_only = QCheckBox("只看已接入")
        self.module_integrated_only.stateChanged.connect(self._refresh_module_table)
        btn_reload = QPushButton("刷新目录")
        btn_reload.clicked.connect(self._refresh_module_catalog)
        filter_row.addWidget(self.module_search, 1)
        filter_row.addWidget(self.module_version_filter)
        filter_row.addWidget(self.module_integrated_only)
        filter_row.addWidget(btn_reload)
        center_layout.addLayout(filter_row)

        self.module_table = QTableWidget(0, 8)
        self.module_table.setHorizontalHeaderLabels(
            ["模块名", "版本", "类型", "状态", "接入位置", "简介", "配置入口", "文档入口"]
        )
        self.module_table.horizontalHeader().setStretchLastSection(True)
        self.module_table.itemSelectionChanged.connect(self._show_selected_module_detail)
        center_layout.addWidget(self.module_table, 1)
        layout.addWidget(center, 3)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel("模块详情"))
        self.module_detail_text = QPlainTextEdit()
        self.module_detail_text.setReadOnly(True)
        right_layout.addWidget(self.module_detail_text, 1)
        layout.addWidget(right, 2)
        return page

    def _build_autozoom_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        title = QLabel("AutoZoom 独立页面")
        title.setStyleSheet("font-size: 18px; font-weight: 700;")
        layout.addWidget(title)

        summary = QLabel(
            "这里集中展示 AutoZoom 显微镜自动聚焦相关模块，按 控制 / 参考 / Z轴 / 日志 四个子区管理。"
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)

        tabs = QTabWidget()
        tabs.addTab(self._build_autozoom_console_tab(), "控制台")
        tabs.addTab(self._build_autozoom_reference_tab(), "聚焦参考")
        tabs.addTab(self._build_autozoom_zaxis_tab(), "Z轴控制")
        tabs.addTab(self._build_autozoom_module_tab(), "模块清单")
        layout.addWidget(tabs, 1)

        self.autozoom_log = QPlainTextEdit()
        self.autozoom_log.setReadOnly(True)
        self.autozoom_log.setPlaceholderText("AutoZoom 运行日志 / 参考建立日志 / 补焦状态")
        layout.addWidget(self.autozoom_log, 1)

        return page

    def _build_autozoom_console_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        btn_row = QHBoxLayout()
        self.autozoom_start_button = QPushButton("启动 AutoZoom 演示")
        self.autozoom_start_button.clicked.connect(self._run_autozoom_demo)
        self.autozoom_ref_button = QPushButton("建立聚焦参考")
        self.autozoom_ref_button.clicked.connect(self._build_autozoom_reference)
        self.autozoom_check_button = QPushButton("检查补焦状态")
        self.autozoom_check_button.clicked.connect(self._check_autozoom_focus)
        self.autozoom_open_folder_button = QPushButton("打开 AutoZoom 目录")
        self.autozoom_open_folder_button.clicked.connect(self._open_autozoom_folder)
        btn_row.addWidget(self.autozoom_start_button)
        btn_row.addWidget(self.autozoom_ref_button)
        btn_row.addWidget(self.autozoom_check_button)
        btn_row.addWidget(self.autozoom_open_folder_button)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self.autozoom_status_label = QLabel("状态：未就绪")
        layout.addWidget(self.autozoom_status_label)

        self.autozoom_console_detail = QPlainTextEdit()
        self.autozoom_console_detail.setReadOnly(True)
        self.autozoom_console_detail.setPlaceholderText("AutoZoom 当前状态 / 提示 / 操作结果")
        layout.addWidget(self.autozoom_console_detail, 1)
        return page

    def _build_autozoom_reference_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        splitter = QSplitter(Qt.Horizontal)

        # ---------- 左侧：配置 ----------
        left = QWidget()
        left_layout = QVBoxLayout(left)

        capture_group = QGroupBox("采集配置")
        form = QFormLayout(capture_group)

        self.autozoom_capture_mode = QComboBox()
        self.autozoom_capture_mode.addItem("屏幕区域", "screen_region")
        self.autozoom_capture_mode.addItem("窗口 ROI", "window_roi")
        self.autozoom_capture_mode.addItem("USB 相机", "usb_camera")
        self.autozoom_capture_mode.addItem("导入视频", "video_file")
        self.autozoom_capture_mode.currentIndexChanged.connect(self._on_autozoom_capture_mode_changed)

        self.autozoom_window_title = AutoZoomWindowComboBox()
        self.autozoom_window_title.setEditable(True)
        self.autozoom_window_title.setInsertPolicy(QComboBox.NoInsert)
        self.autozoom_window_title.setPlaceholderText("选择或输入窗口标题")
        self.autozoom_window_title.popupRequested.connect(self._refresh_autozoom_window_list)
        self.autozoom_window_title.currentTextChanged.connect(self._on_autozoom_window_selected)
        if self.autozoom_window_title.lineEdit() is not None:
            self.autozoom_window_title.lineEdit().editingFinished.connect(self._sync_autozoom_window_title_history)

        self.autozoom_window_match_mode = QComboBox()
        self.autozoom_window_match_mode.addItems(["contains", "exact"])

        self.autozoom_window_padding_left = self._spin(0, 500, 0)
        self.autozoom_window_padding_top = self._spin(0, 500, 0)
        self.autozoom_window_padding_right = self._spin(0, 500, 0)
        self.autozoom_window_padding_bottom = self._spin(0, 500, 0)
        pad_row = QHBoxLayout()
        for label_text, widget in [
            ("左", self.autozoom_window_padding_left),
            ("上", self.autozoom_window_padding_top),
            ("右", self.autozoom_window_padding_right),
            ("下", self.autozoom_window_padding_bottom),
        ]:
            pad_row.addWidget(QLabel(label_text))
            pad_row.addWidget(widget)
        pad_row.addStretch(1)
        pad_wrap = QWidget()
        pad_wrap.setLayout(pad_row)

        # 视频文件配置
        self.autozoom_video_group = QGroupBox("视频文件配置")
        video_form = QFormLayout(self.autozoom_video_group)
        self.autozoom_video_path = QLineEdit()
        self.autozoom_video_path.setPlaceholderText("选择视频文件路径")
        self.autozoom_video_path.setReadOnly(True)
        self.autozoom_video_browse = QPushButton("浏览...")
        self.autozoom_video_browse.clicked.connect(self._select_autozoom_video_file)
        video_path_row = QHBoxLayout()
        video_path_row.addWidget(self.autozoom_video_path, 1)
        video_path_row.addWidget(self.autozoom_video_browse)
        video_form.addRow("视频路径", video_path_row)
        self.autozoom_video_group.setVisible(False)

        # USB 相机配置
        self.autozoom_usb_group = QGroupBox("USB 相机配置")
        usb_form = QFormLayout(self.autozoom_usb_group)
        self.autozoom_usb_device = self._spin(0, 20, 0)
        self.autozoom_usb_device.setPrefix("设备 ")
        self.autozoom_usb_resolution = self._combo(["PAL", "NTSC", "AUTO"])
        self.autozoom_usb_resolution.setCurrentText("AUTO")
        self.autozoom_usb_format = self._combo(["AUTO", "MJPG", "YUY2", "YUYV", "UYVY"])
        self.autozoom_usb_format.setCurrentText("AUTO")
        usb_form.addRow("设备索引", self.autozoom_usb_device)
        usb_form.addRow("分辨率", self.autozoom_usb_resolution)
        usb_form.addRow("像素格式", self.autozoom_usb_format)
        self.autozoom_usb_group.setVisible(False)

        self.autozoom_ref_count = QSpinBox()
        self.autozoom_ref_count.setRange(1, 100)
        self.autozoom_ref_count.setValue(5)
        self.autozoom_ref_sleep = QDoubleSpinBox()
        self.autozoom_ref_sleep.setRange(0.0, 10.0)
        self.autozoom_ref_sleep.setDecimals(2)
        self.autozoom_ref_sleep.setSingleStep(0.1)
        self.autozoom_ref_sleep.setValue(0.15)

        form.addRow("采集模式", self.autozoom_capture_mode)
        form.addRow("窗口标题", self.autozoom_window_title)
        form.addRow("匹配方式", self.autozoom_window_match_mode)
        form.addRow("窗口边缘补偿", pad_wrap)
        form.addRow("采集次数", self.autozoom_ref_count)
        form.addRow("采集间隔(s)", self.autozoom_ref_sleep)
        left_layout.addWidget(capture_group)
        left_layout.addWidget(self.autozoom_video_group)
        left_layout.addWidget(self.autozoom_usb_group)

        roi_group = QGroupBox("ROI 区域（在右侧预览图中拖拽框选）")
        roi_form = QFormLayout(roi_group)
        self.autozoom_roi_x = self._spin(0, 99999, 0)
        self.autozoom_roi_y = self._spin(0, 99999, 0)
        self.autozoom_roi_w = self._spin(1, 99999, 300)
        self.autozoom_roi_h = self._spin(1, 99999, 300)
        self.autozoom_roi_x.setReadOnly(True)
        self.autozoom_roi_y.setReadOnly(True)
        self.autozoom_roi_w.setReadOnly(True)
        self.autozoom_roi_h.setReadOnly(True)
        roi_row = QHBoxLayout()
        for label_text, widget in [
            ("X", self.autozoom_roi_x),
            ("Y", self.autozoom_roi_y),
            ("W", self.autozoom_roi_w),
            ("H", self.autozoom_roi_h),
        ]:
            roi_row.addWidget(QLabel(label_text))
            roi_row.addWidget(widget)
        roi_row.addStretch(1)
        roi_wrap = QWidget()
        roi_wrap.setLayout(roi_row)
        roi_form.addRow("坐标", roi_wrap)
        left_layout.addWidget(roi_group)

        btn_row = QHBoxLayout()
        self.autozoom_capture_frame_button = QPushButton("抓拍参考帧")
        self.autozoom_capture_frame_button.setToolTip("捕获当前窗口/USB 一帧完整图像，用于在右侧框选 ROI")
        self.autozoom_capture_frame_button.clicked.connect(self._capture_reference_frame)

        self.autozoom_start_preview_button = QPushButton("开始 ROI 预览")
        self.autozoom_start_preview_button.clicked.connect(self._start_autozoom_roi_preview)
        self.autozoom_stop_preview_button = QPushButton("停止 ROI 预览")
        self.autozoom_stop_preview_button.clicked.connect(self._stop_autozoom_roi_preview)
        self.autozoom_stop_preview_button.setEnabled(False)

        self.autozoom_test_capture_button = QPushButton("测试窗口采集")
        self.autozoom_test_capture_button.clicked.connect(self._test_autozoom_capture)

        btn_row.addWidget(self.autozoom_capture_frame_button)
        btn_row.addWidget(self.autozoom_start_preview_button)
        btn_row.addWidget(self.autozoom_stop_preview_button)
        btn_row.addWidget(self.autozoom_test_capture_button)
        btn_row.addStretch(1)
        left_layout.addLayout(btn_row)
        left_layout.addStretch(1)

        # ---------- 右侧：预览 ----------
        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.autozoom_preview_label = AutoZoomRoiPreviewLabel()
        self.autozoom_preview_label.roiSelected.connect(self._on_autozoom_roi_selected)
        right_layout.addWidget(self.autozoom_preview_label, 1)
        self.autozoom_preview_info = QLabel("状态: 未启动")
        self.autozoom_preview_info.setStyleSheet("color: #888;")
        right_layout.addWidget(self.autozoom_preview_info)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 2)

        # ---------- 底部：参考信息 ----------
        self.autozoom_reference_text = QPlainTextEdit()
        self.autozoom_reference_text.setReadOnly(True)
        self.autozoom_reference_text.setPlaceholderText("聚焦参考信息、基线结果和参考目录")
        self.autozoom_reference_text.setMaximumHeight(160)
        layout.addWidget(self.autozoom_reference_text, 1)

        # 初始化历史记录与窗口列表
        self._autozoom_window_title_history = []
        self._load_autozoom_window_title_history()
        self._refresh_autozoom_window_list()
        return page

    def _build_autozoom_zaxis_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        form = QFormLayout()
        self.autozoom_z_axis = QSpinBox()
        self.autozoom_z_axis.setRange(1, 16)
        self.autozoom_z_axis.setValue(1)
        self.autozoom_z_speed = QSpinBox()
        self.autozoom_z_speed.setRange(1, 10000)
        self.autozoom_z_speed.setValue(100)
        self.autozoom_z_accel = QSpinBox()
        self.autozoom_z_accel.setRange(1, 10000)
        self.autozoom_z_accel.setValue(100)
        self.autozoom_z_steps = QSpinBox()
        self.autozoom_z_steps.setRange(1, 100000)
        self.autozoom_z_steps.setValue(10)
        form.addRow("Z 轴编号", self.autozoom_z_axis)
        form.addRow("速度", self.autozoom_z_speed)
        form.addRow("加速度", self.autozoom_z_accel)
        form.addRow("移动步数", self.autozoom_z_steps)
        layout.addLayout(form)

        move_row = QHBoxLayout()
        btn_pos = QPushButton("正向移动")
        btn_neg = QPushButton("反向移动")
        btn_pos.clicked.connect(lambda: self._move_autozoom_zaxis(+1))
        btn_neg.clicked.connect(lambda: self._move_autozoom_zaxis(-1))
        move_row.addWidget(btn_pos)
        move_row.addWidget(btn_neg)
        move_row.addStretch(1)
        layout.addLayout(move_row)

        self.autozoom_z_text = QPlainTextEdit()
        self.autozoom_z_text.setReadOnly(True)
        self.autozoom_z_text.setPlaceholderText("Z 轴连接状态、移动结果、错误信息")
        layout.addWidget(self.autozoom_z_text, 1)
        return page

    def _build_autozoom_module_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        self.autozoom_table = QTableWidget(0, 6)
        self.autozoom_table.setHorizontalHeaderLabels(["模块名", "版本", "类型", "状态", "接入位置", "简介"])
        self.autozoom_table.horizontalHeader().setStretchLastSection(True)
        self.autozoom_table.itemSelectionChanged.connect(self._show_selected_autozoom_detail)
        layout.addWidget(self.autozoom_table, 2)

        btn_row = QHBoxLayout()
        btn_refresh = QPushButton("刷新 AutoZoom 列表")
        btn_refresh.clicked.connect(self._refresh_autozoom_table)
        btn_module_center = QPushButton("转到模块中心")
        btn_module_center.clicked.connect(lambda: self.nav_list.setCurrentRow(2))
        btn_row.addWidget(btn_refresh)
        btn_row.addWidget(btn_module_center)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self.autozoom_detail = QPlainTextEdit()
        self.autozoom_detail.setReadOnly(True)
        layout.addWidget(self.autozoom_detail, 1)

        self._refresh_autozoom_table()
        return page

    def _refresh_autozoom_table(self) -> None:
        if not hasattr(self, "autozoom_table"):
            return
        rows = [row for row in self.module_rows if row.version == 8 or "AutoZoom" in row.title or "focus_" in row.name]
        self.autozoom_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            values = [row.name, f"v{row.version}", row.type_label, status_text(row.status), row.placement, row.summary]
            for c, value in enumerate(values):
                self.autozoom_table.setItem(i, c, QTableWidgetItem(str(value)))
            self.autozoom_table.item(i, 0).setData(Qt.UserRole, row)
        if rows:
            self.autozoom_table.selectRow(0)
            self._show_selected_autozoom_detail()

    def _load_autozoom_window_title_history(self) -> None:
        if not hasattr(self, "autozoom_window_title"):
            return
        history = list(getattr(self, "_autozoom_window_title_history", []))
        current = self._autozoom_window_title_value() if hasattr(self, "_autozoom_window_title_value") else ""
        if current and current not in history:
            history.insert(0, current)
        self._autozoom_window_title_history = history[:10]
        self.autozoom_window_title.blockSignals(True)
        try:
            self.autozoom_window_title.clear()
            for title in self._autozoom_window_title_history:
                if title:
                    self.autozoom_window_title.addItem(title)
            if current:
                idx = self.autozoom_window_title.findText(current)
                if idx >= 0:
                    self.autozoom_window_title.setCurrentIndex(idx)
        finally:
            self.autozoom_window_title.blockSignals(False)

    def _sync_autozoom_window_title_history(self) -> None:
        if not hasattr(self, "autozoom_window_title"):
            return
        title = self._autozoom_window_title_value()
        if not title:
            return
        history = list(getattr(self, "_autozoom_window_title_history", []))
        if title in history:
            history.remove(title)
        history.insert(0, title)
        self._autozoom_window_title_history = history[:10]
        self._load_autozoom_window_title_history()

    def _autozoom_window_title_value(self) -> str:
        if not hasattr(self, "autozoom_window_title"):
            return ""
        return self.autozoom_window_title.currentText().strip()

    def _run_autozoom_demo(self) -> None:
        self._append_log("[AutoZoom] 启动演示入口")
        if hasattr(self, "autozoom_status_label"):
            self.autozoom_status_label.setText("状态：正在运行演示")
        if hasattr(self, "autozoom_console_detail"):
            self.autozoom_console_detail.setPlainText("运行 demo 模式，用于验证 AutoZoom 入口和截图/聚焦链路。")
        try:
            from Utils.AutoZoom.Focus.run_autofocus import run_demo
            run_demo()
            self._append_log("[AutoZoom] 演示执行完成")
            if hasattr(self, "autozoom_status_label"):
                self.autozoom_status_label.setText("状态：演示完成")
        except Exception as exc:
            self._append_log(f"[AutoZoom] 演示执行失败: {exc}")
            if hasattr(self, "autozoom_status_label"):
                self.autozoom_status_label.setText("状态：演示失败")
            if hasattr(self, "autozoom_console_detail"):
                self.autozoom_console_detail.setPlainText(str(exc))
            QMessageBox.warning(self, "AutoZoom", f"演示执行失败: {exc}")

    def _build_autozoom_reference(self) -> None:
        self._append_log("[AutoZoom] 开始建立聚焦参考")
        if hasattr(self, "autozoom_reference_text"):
            self.autozoom_reference_text.setPlainText("正在建立聚焦参考，请稍候...")
        try:
            from Utils.AutoZoom.Focus.config import AutofocusConfig
            from Utils.AutoZoom.Focus.metrics import FocusMetricsCalculator
            from Utils.AutoZoom.Focus.scorer import FocusScorer
            from Utils.AutoZoom.Focus.z_axis import ZAxisController
            from Utils.AutoZoom.Focus.controller import AutofocusController

            cfg = AutofocusConfig(
                capture_mode=self.autozoom_capture_mode.currentData() if hasattr(self, "autozoom_capture_mode") else "screen_region",
                window_title=self._autozoom_window_title_value(),
                window_match_mode=self.autozoom_window_match_mode.currentText() if hasattr(self, "autozoom_window_match_mode") else "contains",
                window_padding=(
                    int(self.autozoom_window_padding_left.value()) if hasattr(self, "autozoom_window_padding_left") else 0,
                    int(self.autozoom_window_padding_top.value()) if hasattr(self, "autozoom_window_padding_top") else 0,
                    int(self.autozoom_window_padding_right.value()) if hasattr(self, "autozoom_window_padding_right") else 0,
                    int(self.autozoom_window_padding_bottom.value()) if hasattr(self, "autozoom_window_padding_bottom") else 0,
                ),
                focus_reference_capture_count=int(self.autozoom_ref_count.value()) if hasattr(self, "autozoom_ref_count") else 5,
                focus_roi=self._autozoom_roi_value(),
                usb_device_index=int(self.autozoom_usb_device.value()) if hasattr(self, "autozoom_usb_device") else 0,
                usb_resolution=self.autozoom_usb_resolution.currentText() if hasattr(self, "autozoom_usb_resolution") else "AUTO",
                usb_pixel_format=self._autozoom_usb_format_value(),
            )
            self._sync_autozoom_window_title_history()
            metrics_calc = FocusMetricsCalculator(cfg)
            scorer = FocusScorer(cfg, metrics_calc)
            z_axis = ZAxisController(cfg)
            controller = AutofocusController(cfg, scorer, metrics_calc, z_axis)
            controller.on_log = self._append_log
            ref = controller.build_reference(
                output_root=self.runtime.artifacts_dir,
            )
            text = json.dumps(ref, ensure_ascii=False, indent=2)
            if hasattr(self, "autozoom_reference_text"):
                self.autozoom_reference_text.setPlainText(text)
            self._append_log("[AutoZoom] 聚焦参考已建立")
        except Exception as exc:
            self._append_log(f"[AutoZoom] 建立参考失败: {exc}")
            if hasattr(self, "autozoom_reference_text"):
                self.autozoom_reference_text.setPlainText(str(exc))
            QMessageBox.warning(self, "AutoZoom", f"建立参考失败: {exc}")

    def _check_autozoom_focus(self) -> None:
        self._append_log("[AutoZoom] 检查补焦状态")
        try:
            from Utils.AutoZoom.Focus.config import AutofocusConfig
            from Utils.AutoZoom.Focus.metrics import FocusMetricsCalculator
            from Utils.AutoZoom.Focus.scorer import FocusScorer
            from Utils.AutoZoom.Focus.z_axis import ZAxisController
            from Utils.AutoZoom.Focus.controller import AutofocusController

            cfg = AutofocusConfig(
                capture_mode=self.autozoom_capture_mode.currentData() if hasattr(self, "autozoom_capture_mode") else "screen_region",
                window_title=self._autozoom_window_title_value(),
                window_match_mode=self.autozoom_window_match_mode.currentText() if hasattr(self, "autozoom_window_match_mode") else "contains",
                window_padding=(
                    int(self.autozoom_window_padding_left.value()) if hasattr(self, "autozoom_window_padding_left") else 0,
                    int(self.autozoom_window_padding_top.value()) if hasattr(self, "autozoom_window_padding_top") else 0,
                    int(self.autozoom_window_padding_right.value()) if hasattr(self, "autozoom_window_padding_right") else 0,
                    int(self.autozoom_window_padding_bottom.value()) if hasattr(self, "autozoom_window_padding_bottom") else 0,
                ),
                focus_roi=self._autozoom_roi_value(),
                usb_device_index=int(self.autozoom_usb_device.value()) if hasattr(self, "autozoom_usb_device") else 0,
                usb_resolution=self.autozoom_usb_resolution.currentText() if hasattr(self, "autozoom_usb_resolution") else "AUTO",
                usb_pixel_format=self._autozoom_usb_format_value(),
            )
            metrics_calc = FocusMetricsCalculator(cfg)
            scorer = FocusScorer(cfg, metrics_calc)
            z_axis = ZAxisController(cfg)
            controller = AutofocusController(cfg, scorer, metrics_calc, z_axis)
            controller.on_log = self._append_log
            if hasattr(self, "autozoom_reference_text"):
                self.autozoom_reference_text.appendPlainText("\n[AutoZoom] 检查当前补焦状态...")
            live = metrics_calc.capture_live()
            focus_score, ratios = scorer.score_ratio(live.get("roi_metrics"))
            need, reasons = controller.evaluate_trigger(focus_score)
            msg = {
                "focus_score_ratio": focus_score,
                "component_ratios": ratios,
                "need_autofocus": need,
                "reasons": reasons,
            }
            if hasattr(self, "autozoom_console_detail"):
                self.autozoom_console_detail.setPlainText(json.dumps(msg, ensure_ascii=False, indent=2))
            self._append_log(f"[AutoZoom] 当前补焦触发={need}, 原因={reasons}")
        except Exception as exc:
            self._append_log(f"[AutoZoom] 检查失败: {exc}")
            if hasattr(self, "autozoom_console_detail"):
                self.autozoom_console_detail.setPlainText(str(exc))
            QMessageBox.warning(self, "AutoZoom", f"检查失败: {exc}")

    def _test_autozoom_capture(self) -> None:
        try:
            from Utils.AutoZoom.Focus.config import AutofocusConfig
            from Utils.AutoZoom.Focus.metrics import FocusMetricsCalculator

            cfg = AutofocusConfig(
                capture_mode=self.autozoom_capture_mode.currentData() if hasattr(self, "autozoom_capture_mode") else "screen_region",
                window_title=self._autozoom_window_title_value(),
                window_match_mode=self.autozoom_window_match_mode.currentText() if hasattr(self, "autozoom_window_match_mode") else "contains",
                window_padding=(
                    int(self.autozoom_window_padding_left.value()) if hasattr(self, "autozoom_window_padding_left") else 0,
                    int(self.autozoom_window_padding_top.value()) if hasattr(self, "autozoom_window_padding_top") else 0,
                    int(self.autozoom_window_padding_right.value()) if hasattr(self, "autozoom_window_padding_right") else 0,
                    int(self.autozoom_window_padding_bottom.value()) if hasattr(self, "autozoom_window_padding_bottom") else 0,
                ),
                focus_roi=self._autozoom_roi_value(),
                usb_device_index=int(self.autozoom_usb_device.value()) if hasattr(self, "autozoom_usb_device") else 0,
                usb_resolution=self.autozoom_usb_resolution.currentText() if hasattr(self, "autozoom_usb_resolution") else "AUTO",
                usb_pixel_format=self._autozoom_usb_format_value(),
            )
            metrics_calc = FocusMetricsCalculator(cfg)
            result = metrics_calc.capture_live()
            preview = {
                "capture_area": result.get("capture_area"),
                "roi": result.get("roi"),
                "full_tenengrad": result.get("full", {}).get("tenengrad"),
                "roi_tenengrad": result.get("roi_metrics", {}).get("tenengrad"),
            }
            if hasattr(self, "autozoom_reference_text"):
                self.autozoom_reference_text.setPlainText(json.dumps(preview, ensure_ascii=False, indent=2))
            self._append_log(f"[AutoZoom] 测试采集成功: {preview}")
        except Exception as exc:
            self._append_log(f"[AutoZoom] 测试采集失败: {exc}")
            QMessageBox.warning(self, "AutoZoom", f"测试采集失败: {exc}")

    def _autozoom_roi_value(self) -> Tuple[int, int, int, int]:
        """读取界面 ROI 坐标控件，返回 (x, y, w, h)。"""
        x = int(self.autozoom_roi_x.value()) if hasattr(self, "autozoom_roi_x") else 0
        y = int(self.autozoom_roi_y.value()) if hasattr(self, "autozoom_roi_y") else 0
        w = int(self.autozoom_roi_w.value()) if hasattr(self, "autozoom_roi_w") else 300
        h = int(self.autozoom_roi_h.value()) if hasattr(self, "autozoom_roi_h") else 300
        return (x, y, w, h)

    def _set_autozoom_roi_value(self, x: int, y: int, w: int, h: int) -> None:
        """设置界面 ROI 坐标控件。"""
        if hasattr(self, "autozoom_roi_x"):
            self.autozoom_roi_x.setValue(max(0, int(x)))
        if hasattr(self, "autozoom_roi_y"):
            self.autozoom_roi_y.setValue(max(0, int(y)))
        if hasattr(self, "autozoom_roi_w"):
            self.autozoom_roi_w.setValue(max(1, int(w)))
        if hasattr(self, "autozoom_roi_h"):
            self.autozoom_roi_h.setValue(max(1, int(h)))
        if hasattr(self, "autozoom_preview_label"):
            self.autozoom_preview_label.set_roi(x, y, w, h)

    def _autozoom_usb_format_value(self) -> Optional[str]:
        """获取 USB 像素格式，AUTO 返回 None。"""
        if not hasattr(self, "autozoom_usb_format"):
            return None
        fmt = self.autozoom_usb_format.currentText()
        return None if fmt == "AUTO" else fmt

    def _open_autozoom_usb_source(self) -> Optional[UCCFrameSource]:
        """根据当前 USB 配置打开 UCCFrameSource，失败时弹窗并返回 None。"""
        if not hasattr(self, "autozoom_usb_device"):
            return None
        device_idx = int(self.autozoom_usb_device.value())
        resolution = self.autozoom_usb_resolution.currentText() if hasattr(self, "autozoom_usb_resolution") else "AUTO"
        pixel_format = self._autozoom_usb_format_value()
        try:
            source = UCCFrameSource(
                device_index=device_idx,
                resolution=resolution,
                pixel_format=pixel_format,
            )
            self._append_log(f"[AutoZoom] USB 相机已打开: device={device_idx}, resolution={resolution}, format={pixel_format or 'AUTO'}")
            return source
        except Exception as exc:
            self._append_log(f"[AutoZoom] USB 相机打开失败 (device={device_idx}): {exc}")
            QMessageBox.warning(self, "AutoZoom", f"USB 相机打开失败:\n{exc}")
            return None

    def _close_autozoom_usb_source(self, source: Optional[UCCFrameSource]) -> None:
        """安全释放 USB 相机资源。"""
        if source is not None:
            try:
                source.release()
                self._append_log("[AutoZoom] USB 相机已释放")
            except Exception as exc:
                self._append_log(f"[AutoZoom] USB 相机释放异常: {exc}")

    def _select_autozoom_video_file(self) -> None:
        """弹出文件对话框选择视频文件。"""
        if not hasattr(self, "autozoom_video_path"):
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频文件",
            "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv);;所有文件 (*.*)",
        )
        if path:
            self.autozoom_video_path.setText(path)
            self._append_log(f"[AutoZoom] 已选择视频文件: {path}")

    def _open_autozoom_video_source(self) -> Optional[cv2.VideoCapture]:
        """根据当前视频文件路径打开 cv2.VideoCapture，失败时弹窗并返回 None。"""
        if not hasattr(self, "autozoom_video_path"):
            QMessageBox.warning(self, "AutoZoom", "未配置视频文件路径")
            return None
        path = self.autozoom_video_path.text().strip()
        if not path:
            QMessageBox.warning(self, "AutoZoom", "请先选择视频文件")
            return None
        try:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise RuntimeError(f"无法打开视频文件: {path}")
            self._append_log(f"[AutoZoom] 视频文件已打开: {path}")
            return cap
        except Exception as exc:
            self._append_log(f"[AutoZoom] 视频文件打开失败: {exc}")
            QMessageBox.warning(self, "AutoZoom", f"视频文件打开失败:\n{exc}")
            return None

    def _close_autozoom_video_source(self, cap: Optional[cv2.VideoCapture]) -> None:
        """安全释放视频文件资源。"""
        if cap is not None:
            try:
                cap.release()
                self._append_log("[AutoZoom] 视频文件已释放")
            except Exception as exc:
                self._append_log(f"[AutoZoom] 视频文件释放异常: {exc}")

    def _grab_autozoom_video_frame(
        self,
        cap: Optional[cv2.VideoCapture] = None,
        auto_release: bool = True,
    ) -> Tuple[Optional[np.ndarray], Optional[cv2.VideoCapture]]:
        """从视频文件读取下一帧 RGB 图像；到末尾时自动循环。

        返回 (rgb_image, cap)。如果传入了 cap 则复用，否则临时打开并在
        auto_release=True 时自动释放。
        """
        opened_here = False
        if cap is None:
            cap = self._open_autozoom_video_source()
            opened_here = True
        if cap is None:
            return None, None
        try:
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None or frame_bgr.size == 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None or frame_bgr.size == 0:
                    raise RuntimeError("视频文件读取帧为空")
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            return rgb, cap
        except Exception as exc:
            self._append_log(f"[AutoZoom] 视频文件读帧失败: {exc}")
            if opened_here or auto_release:
                self._close_autozoom_video_source(cap)
            return None, None
        finally:
            if (opened_here or auto_release) and cap is not None:
                self._close_autozoom_video_source(cap)

    def _grab_autozoom_usb_frame(
        self,
        source: Optional[UCCFrameSource] = None,
        auto_release: bool = True,
    ) -> Tuple[Optional[np.ndarray], Optional[UCCFrameSource]]:
        """从 USB 相机读取一帧 RGB 图像。

        返回 (rgb_image, source)。如果传入了 source 则复用，否则临时打开并在
        auto_release=True 时自动释放。
        """
        opened_here = False
        if source is None:
            source = self._open_autozoom_usb_source()
            opened_here = True
        if source is None:
            return None, None
        try:
            frame_bgr = source.grab_frame()
            if frame_bgr is None or frame_bgr.size == 0:
                raise RuntimeError("USB 相机读取帧为空")
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            return rgb, source
        except Exception as exc:
            self._append_log(f"[AutoZoom] USB 相机读帧失败: {exc}")
            if opened_here or auto_release:
                self._close_autozoom_usb_source(source)
            return None, None
        finally:
            if (opened_here or auto_release) and source is not None:
                self._close_autozoom_usb_source(source)

    def _on_autozoom_capture_mode_changed(self, _index: int) -> None:
        """根据采集模式切换相关控件的可见性。"""
        mode = self.autozoom_capture_mode.currentData() if hasattr(self, "autozoom_capture_mode") else "screen_region"
        is_window = mode == "window_roi"
        is_usb = mode == "usb_camera"
        is_video = mode == "video_file"
        if hasattr(self, "autozoom_window_title"):
            self.autozoom_window_title.setVisible(is_window)
        if hasattr(self, "autozoom_window_match_mode"):
            self.autozoom_window_match_mode.setVisible(is_window)
        if hasattr(self, "autozoom_window_padding_left"):
            self.autozoom_window_padding_left.parentWidget().setVisible(not is_usb and not is_video)
        if hasattr(self, "autozoom_usb_group"):
            self.autozoom_usb_group.setVisible(is_usb)
        if hasattr(self, "autozoom_video_group"):
            self.autozoom_video_group.setVisible(is_video)

    def _refresh_autozoom_window_list(self) -> None:
        """枚举当前可见窗口并刷新下拉框。"""
        if not hasattr(self, "autozoom_window_title"):
            return
        try:
            import win32gui
            titles = set()

            def _enum(hwnd, _):
                if not win32gui.IsWindowVisible(hwnd):
                    return
                text = (win32gui.GetWindowText(hwnd) or "").strip()
                if text:
                    titles.add(text)

            win32gui.EnumWindows(_enum, None)
            current = self._autozoom_window_title_value()
            history = list(getattr(self, "_autozoom_window_title_history", []))
            self.autozoom_window_title.blockSignals(True)
            try:
                self.autozoom_window_title.clear()
                for title in history:
                    if title:
                        self.autozoom_window_title.addItem(title)
                for title in sorted(titles):
                    if title and title not in history:
                        self.autozoom_window_title.addItem(title)
                if current:
                    idx = self.autozoom_window_title.findText(current)
                    if idx >= 0:
                        self.autozoom_window_title.setCurrentIndex(idx)
            finally:
                self.autozoom_window_title.blockSignals(False)
        except Exception as exc:
            self._append_log(f"[AutoZoom] 刷新窗口列表失败: {exc}")

    def _on_autozoom_window_selected(self, text: str) -> None:
        """用户从下拉框选择窗口后，自动切到 window_roi 模式。"""
        if not text.strip():
            return
        if hasattr(self, "autozoom_capture_mode"):
            idx = self.autozoom_capture_mode.findData("window_roi")
            if idx >= 0:
                self.autozoom_capture_mode.setCurrentIndex(idx)
        self._sync_autozoom_window_title_history()

    def _display_autozoom_reference_frame(self, image_rgb: np.ndarray, source_desc: str) -> None:
        """将 RGB 参考帧显示到预览区并设置默认 ROI。"""
        self._autozoom_reference_frame_shape = image_rgb.shape[:2]
        h, w = image_rgb.shape[:2]
        qt_image = QImage(image_rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image.copy())

        self.autozoom_preview_label.clear_preview()
        self.autozoom_preview_label.set_preview_pixmap(pixmap)
        self.autozoom_preview_label.set_reference_mode(True)

        default_w, default_h = min(300, w), min(300, h)
        default_x, default_y = (w - default_w) // 2, (h - default_h) // 2
        self._set_autozoom_roi_value(default_x, default_y, default_w, default_h)

        self.autozoom_preview_info.setText(f"状态: {source_desc} 参考帧已捕获 {w}x{h}，请拖拽框选 ROI")
        self._append_log(f"[AutoZoom] {source_desc} 参考帧已捕获: {w}x{h}")

    def _capture_reference_frame(self) -> None:
        """抓拍当前数据源的一帧完整图像，供用户框选 ROI。"""
        mode = self.autozoom_capture_mode.currentData() if hasattr(self, "autozoom_capture_mode") else "screen_region"
        self._append_log(f"[AutoZoom] 抓拍参考帧 (mode={mode})...")
        try:
            if mode == "usb_camera":
                rgb, _ = self._grab_autozoom_usb_frame()
                if rgb is None or rgb.size == 0:
                    raise RuntimeError("未能从 USB 相机获取图像")
                self._display_autozoom_reference_frame(rgb, "USB")
            else:
                cfg = self._get_autozoom_cfg_for_preview()
                from Utils.AutoZoom.Focus.metrics import FocusMetricsCalculator
                metrics_calc = FocusMetricsCalculator(cfg)
                result = metrics_calc.capture_live()
                image_rgb = result.get("full_rgb")
                if image_rgb is None or image_rgb.size == 0:
                    raise RuntimeError("未能获取到窗口/屏幕图像")
                self._display_autozoom_reference_frame(image_rgb, "窗口")
        except Exception as exc:
            self._append_log(f"[AutoZoom] 抓拍参考帧失败: {exc}")
            QMessageBox.warning(self, "AutoZoom", f"抓拍参考帧失败: {exc}")

    def _on_autozoom_roi_selected(self, x: int, y: int, w: int, h: int) -> None:
        """用户在参考帧上完成 ROI 框选。"""
        self._set_autozoom_roi_value(x, y, w, h)
        self.autozoom_preview_info.setText(f"状态: ROI 已选定 ({x}, {y}, {w}, {h})，可点击「开始 ROI 预览」")
        self._append_log(f"[AutoZoom] ROI 已选定: ({x}, {y}, {w}, {h})")

    def _get_autozoom_cfg_for_preview(self) -> "AutofocusConfig":
        """根据当前界面控件构造 AutofocusConfig，用于预览/抓拍。"""
        from Utils.AutoZoom.Focus.config import AutofocusConfig
        return AutofocusConfig(
            capture_mode=self.autozoom_capture_mode.currentData() if hasattr(self, "autozoom_capture_mode") else "screen_region",
            window_title=self._autozoom_window_title_value(),
            window_match_mode=self.autozoom_window_match_mode.currentText() if hasattr(self, "autozoom_window_match_mode") else "contains",
            window_padding=(
                int(self.autozoom_window_padding_left.value()) if hasattr(self, "autozoom_window_padding_left") else 0,
                int(self.autozoom_window_padding_top.value()) if hasattr(self, "autozoom_window_padding_top") else 0,
                int(self.autozoom_window_padding_right.value()) if hasattr(self, "autozoom_window_padding_right") else 0,
                int(self.autozoom_window_padding_bottom.value()) if hasattr(self, "autozoom_window_padding_bottom") else 0,
            ),
            focus_roi=self._autozoom_roi_value(),
            usb_device_index=int(self.autozoom_usb_device.value()) if hasattr(self, "autozoom_usb_device") else 0,
            usb_resolution=self.autozoom_usb_resolution.currentText() if hasattr(self, "autozoom_usb_resolution") else "AUTO",
            usb_pixel_format=self._autozoom_usb_format_value(),
        )

    def _start_autozoom_roi_preview(self) -> None:
        """启动 ROI 实时预览定时器。"""
        if getattr(self, "_autozoom_roi_preview_running", False):
            return
        try:
            mode = self.autozoom_capture_mode.currentData() if hasattr(self, "autozoom_capture_mode") else "screen_region"
            if mode == "usb_camera":
                self._autozoom_roi_preview_usb_source = self._open_autozoom_usb_source()
                if self._autozoom_roi_preview_usb_source is None:
                    return
            elif mode == "video_file":
                self._autozoom_roi_preview_video_source = self._open_autozoom_video_source()
                if self._autozoom_roi_preview_video_source is None:
                    return

            self.autozoom_preview_label.set_reference_mode(False)
            self.autozoom_start_preview_button.setEnabled(False)
            self.autozoom_stop_preview_button.setEnabled(True)
            self.autozoom_capture_frame_button.setEnabled(False)
            self._autozoom_roi_preview_running = True

            if not hasattr(self, "_autozoom_roi_preview_timer"):
                self._autozoom_roi_preview_timer = QTimer(self)
                self._autozoom_roi_preview_timer.timeout.connect(self._update_autozoom_roi_preview)
            self._autozoom_roi_preview_timer.start(200)
            self.autozoom_preview_info.setText("状态: ROI 实时预览中...")
            self._append_log(f"[AutoZoom] 开始 ROI 实时预览 (mode={mode})")
        except Exception as exc:
            self._append_log(f"[AutoZoom] 启动 ROI 预览失败: {exc}")
            self._stop_autozoom_roi_preview()

    def _stop_autozoom_roi_preview(self) -> None:
        """停止 ROI 实时预览定时器。"""
        self._autozoom_roi_preview_running = False
        if hasattr(self, "_autozoom_roi_preview_timer"):
            self._autozoom_roi_preview_timer.stop()
        if hasattr(self, "autozoom_start_preview_button"):
            self.autozoom_start_preview_button.setEnabled(True)
        if hasattr(self, "autozoom_stop_preview_button"):
            self.autozoom_stop_preview_button.setEnabled(False)
        if hasattr(self, "autozoom_capture_frame_button"):
            self.autozoom_capture_frame_button.setEnabled(True)
        if hasattr(self, "autozoom_preview_info"):
            self.autozoom_preview_info.setText("状态: 预览已停止")
        usb_source = getattr(self, "_autozoom_roi_preview_usb_source", None)
        if usb_source is not None:
            self._close_autozoom_usb_source(usb_source)
            self._autozoom_roi_preview_usb_source = None
        video_source = getattr(self, "_autozoom_roi_preview_video_source", None)
        if video_source is not None:
            self._close_autozoom_video_source(video_source)
            self._autozoom_roi_preview_video_source = None
        self._append_log("[AutoZoom] 停止 ROI 实时预览")

    def _update_autozoom_roi_preview(self) -> None:
        """定时捕获窗口/USB 帧并显示 ROI 区域。"""
        if not getattr(self, "_autozoom_roi_preview_running", False):
            return
        try:
            mode = self.autozoom_capture_mode.currentData() if hasattr(self, "autozoom_capture_mode") else "screen_region"
            roi_rgb = None
            tenengrad = 0.0

            if mode == "usb_camera":
                source = getattr(self, "_autozoom_roi_preview_usb_source", None)
                if source is None or not source.is_healthy():
                    self.autozoom_preview_info.setText("状态: USB 相机连接异常，尝试重连...")
                    if source is not None:
                        self._close_autozoom_usb_source(source)
                    self._autozoom_roi_preview_usb_source = self._open_autozoom_usb_source()
                    source = self._autozoom_roi_preview_usb_source
                    if source is None:
                        return
                frame_bgr = source.grab_frame()
                if frame_bgr is None or frame_bgr.size == 0:
                    return
                full_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                x, y, w, h = self._autozoom_roi_value()
                fh, fw = full_rgb.shape[:2]
                x = max(0, min(x, fw - 1))
                y = max(0, min(y, fh - 1))
                w = max(1, min(w, fw - x))
                h = max(1, min(h, fh - y))
                roi_rgb = full_rgb[y : y + h, x : x + w].copy()
                gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
                gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                tenengrad = float((gx ** 2 + gy ** 2).sum())
            elif mode == "video_file":
                cap = getattr(self, "_autozoom_roi_preview_video_source", None)
                if cap is None or not cap.isOpened():
                    self.autozoom_preview_info.setText("状态: 视频文件异常，尝试重开...")
                    if cap is not None:
                        self._close_autozoom_video_source(cap)
                    self._autozoom_roi_preview_video_source = self._open_autozoom_video_source()
                    cap = self._autozoom_roi_preview_video_source
                    if cap is None:
                        return
                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None or frame_bgr.size == 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame_bgr = cap.read()
                    if not ret or frame_bgr is None or frame_bgr.size == 0:
                        return
                full_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                x, y, w, h = self._autozoom_roi_value()
                fh, fw = full_rgb.shape[:2]
                x = max(0, min(x, fw - 1))
                y = max(0, min(y, fh - 1))
                w = max(1, min(w, fw - x))
                h = max(1, min(h, fh - y))
                roi_rgb = full_rgb[y : y + h, x : x + w].copy()
                gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
                gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                tenengrad = float((gx ** 2 + gy ** 2).sum())
            else:
                cfg = self._get_autozoom_cfg_for_preview()
                from Utils.AutoZoom.Focus.metrics import FocusMetricsCalculator
                metrics_calc = FocusMetricsCalculator(cfg)
                result = metrics_calc.capture_live()
                roi_rgb = result.get("roi_rgb")
                tenengrad = result.get("roi_metrics", {}).get("tenengrad", 0.0)

            if roi_rgb is None or roi_rgb.size == 0:
                return
            h, w = roi_rgb.shape[:2]
            qt_image = QImage(roi_rgb.data, w, h, w * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image.copy())
            self.autozoom_preview_label.set_preview_pixmap(pixmap)
            if mode == "usb_camera":
                source_name = "USB"
            elif mode == "video_file":
                source_name = "视频"
            else:
                source_name = "窗口"
            info_text = f"状态: {source_name} 实时预览 {w}x{h} | tenengrad={tenengrad:.2f}"
            self.autozoom_preview_info.setText(info_text)
        except Exception as exc:
            self._append_log(f"[AutoZoom] ROI 预览更新失败: {exc}")

    def _move_autozoom_zaxis(self, direction: int) -> None:
        self._append_log(f"[AutoZoom] Z 轴移动请求: direction={direction}")
        try:
            from Utils.AutoZoom.Focus.config import AutofocusConfig
            from Utils.AutoZoom.Focus.z_axis import ZAxisController

            cfg = AutofocusConfig(
                z_axis=int(self.autozoom_z_axis.value()) if hasattr(self, "autozoom_z_axis") else 1,
                z_speed=int(self.autozoom_z_speed.value()) if hasattr(self, "autozoom_z_speed") else 100,
                z_accel=int(self.autozoom_z_accel.value()) if hasattr(self, "autozoom_z_accel") else 100,
            )
            controller = ZAxisController(cfg)
            controller.connect()
            steps = int(self.autozoom_z_steps.value()) if hasattr(self, "autozoom_z_steps") else 10
            controller.move_relative(direction * steps)
            text = f"Z 轴移动完成: axis={cfg.z_axis}, steps={direction * steps}"
            if hasattr(self, "autozoom_z_text"):
                self.autozoom_z_text.setPlainText(text)
            self._append_log(f"[AutoZoom] {text}")
        except Exception as exc:
            if hasattr(self, "autozoom_z_text"):
                self.autozoom_z_text.setPlainText(str(exc))
            self._append_log(f"[AutoZoom] Z 轴移动失败: {exc}")
            QMessageBox.warning(self, "AutoZoom", f"Z 轴移动失败: {exc}")

    def _open_autozoom_folder(self) -> None:
        try:
            import os
            os.startfile(str(Path("Utils") / "AutoZoom"))
        except Exception as exc:
            QMessageBox.warning(self, "AutoZoom", f"打开目录失败: {exc}")

    def _build_device_center_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        top = QHBoxLayout()
        self.device_topology_tree = QTreeWidget()
        self.device_topology_tree.setHeaderLabels(["设备拓扑", "状态"])
        top.addWidget(self.device_topology_tree, 2)

        self.device_panel_table = QTableWidget(0, 3)
        self.device_panel_table.setHorizontalHeaderLabels(["子系统", "状态", "摘要"])
        self.device_panel_table.horizontalHeader().setStretchLastSection(True)
        top.addWidget(self.device_panel_table, 3)

        self.device_detail_text = QPlainTextEdit()
        self.device_detail_text.setReadOnly(True)
        top.addWidget(self.device_detail_text, 2)
        layout.addLayout(top, 2)

        cards = QHBoxLayout()
        mrc_card = QGroupBox("Newport MRC 4轴")
        mrc_form = QFormLayout(mrc_card)
        self.controls["newport_conn"] = self._spin(0, 16, 0)
        self.controls["mrc_mirror1_x_axis"] = self._spin(1, 8, 1)
        self.controls["mrc_mirror1_y_axis"] = self._spin(1, 8, 2)
        self.controls["mrc_mirror2_x_axis"] = self._spin(1, 8, 3)
        self.controls["mrc_mirror2_y_axis"] = self._spin(1, 8, 4)
        self.controls["mrc_mirror1_x_sign"] = self._combo(["1", "-1"])
        self.controls["mrc_mirror1_y_sign"] = self._combo(["1", "-1"])
        self.controls["mrc_mirror2_x_sign"] = self._combo(["1", "-1"])
        self.controls["mrc_mirror2_y_sign"] = self._combo(["1", "-1"])
        self.controls["mrc_virtual_axis_mode"] = self._combo(["共享", "反射镜1", "反射镜2"])
        mrc_form.addRow("8742 连接", self.controls["newport_conn"])
        mrc_form.addRow("反射镜1 X 轴", self.controls["mrc_mirror1_x_axis"])
        mrc_form.addRow("反射镜1 Y 轴", self.controls["mrc_mirror1_y_axis"])
        mrc_form.addRow("反射镜2 X 轴", self.controls["mrc_mirror2_x_axis"])
        mrc_form.addRow("反射镜2 Y 轴", self.controls["mrc_mirror2_y_axis"])
        mrc_form.addRow("反射镜1 X 方向", self.controls["mrc_mirror1_x_sign"])
        mrc_form.addRow("反射镜1 Y 方向", self.controls["mrc_mirror1_y_sign"])
        mrc_form.addRow("反射镜2 X 方向", self.controls["mrc_mirror2_x_sign"])
        mrc_form.addRow("反射镜2 Y 方向", self.controls["mrc_mirror2_y_sign"])
        mrc_form.addRow("虚拟轴模式", self.controls["mrc_virtual_axis_mode"])
        self.mrc_alloc_label = QLabel("-")
        mrc_form.addRow("分配预览", self.mrc_alloc_label)
        mrc_refresh = QPushButton("刷新分配预览")
        mrc_refresh.clicked.connect(self._refresh_mrc_allocation)
        mrc_form.addRow(mrc_refresh)
        cards.addWidget(mrc_card, 3)

        z_card = QGroupBox("Picomotor Z 子系统")
        z_form = QFormLayout(z_card)
        self.controls["z_picomotor_conn"] = self._spin(0, 16, 1)
        self.controls["z_picomotor_axis"] = self._spin(1, 8, 1)
        self.controls["z_picomotor_sign"] = self._combo(["1", "-1"])
        self.controls["z_picomotor_velocity"] = self._spin(0, 50000, 0)
        self.controls["z_picomotor_acceleration"] = self._spin(0, 50000, 0)
        z_form.addRow("Z 连接", self.controls["z_picomotor_conn"])
        z_form.addRow("Z 轴", self.controls["z_picomotor_axis"])
        z_form.addRow("Z 方向", self.controls["z_picomotor_sign"])
        z_form.addRow("速度", self.controls["z_picomotor_velocity"])
        z_form.addRow("加速度", self.controls["z_picomotor_acceleration"])
        z_btn_row = QHBoxLayout()
        z_test_btn = QPushButton("方向测试")
        z_up_btn = QPushButton("单步上移")
        z_down_btn = QPushButton("单步下移")
        z_test_btn.clicked.connect(lambda: self._run_device_test_by_id("startup_check"))
        z_up_btn.clicked.connect(lambda: self._run_device_test_by_id("picomotor_z_up"))
        z_down_btn.clicked.connect(lambda: self._run_device_test_by_id("picomotor_z_down"))
        z_btn_row.addWidget(z_test_btn)
        z_btn_row.addWidget(z_up_btn)
        z_btn_row.addWidget(z_down_btn)
        z_btn_wrap = QWidget()
        z_btn_wrap.setLayout(z_btn_row)
        z_form.addRow(z_btn_wrap)
        cards.addWidget(z_card, 2)
        layout.addLayout(cards)
        return page

    def _build_device_test_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        tabs = QTabWidget()
        tabs.addTab(self._build_device_test_automated_tab(), "自动化测试")
        self.picomotor_driver_panel = PicomotorDriverPanel(log_callback=self._append_log, parent=tabs)
        tabs.addTab(self.picomotor_driver_panel, "Picomotor 8742/8743 驱动调试")
        layout.addWidget(tabs, 1)
        return page

    def _build_device_test_automated_tab(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)

        self.test_tree = QTreeWidget()
        self.test_tree.setHeaderLabels(["测试项", "分类"])
        self.test_tree.itemSelectionChanged.connect(self._on_test_selected)
        layout.addWidget(self.test_tree, 2)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        self.test_desc = QLabel("选择一个测试项")
        self.test_desc.setWordWrap(True)
        center_layout.addWidget(self.test_desc)
        btn_row = QHBoxLayout()
        btn_selected = QPushButton("执行选中测试")
        btn_selected.clicked.connect(self._run_selected_test)
        btn_startup = QPushButton("手动触发启动自检")
        btn_startup.clicked.connect(lambda: self._run_device_test_by_id("startup_check"))
        btn_row.addWidget(btn_selected)
        btn_row.addWidget(btn_startup)
        btn_row.addStretch(1)
        center_layout.addLayout(btn_row)
        self.test_raw_output = QPlainTextEdit()
        self.test_raw_output.setReadOnly(True)
        center_layout.addWidget(self.test_raw_output, 1)
        layout.addWidget(center, 3)

        self.test_result_table = QTableWidget(0, 5)
        self.test_result_table.setHorizontalHeaderLabels(["测试项", "状态", "耗时(ms)", "错误/信息", "原始返回"])
        self.test_result_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.test_result_table, 4)
        return page

    def _build_run_modes_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        mode_row = QHBoxLayout()
        sim_card = QGroupBox("模拟模式")
        sim_layout = QFormLayout(sim_card)
        self.mode_sim_radio = QCheckBox("启用模拟模式")
        sim_layout.addRow(self.mode_sim_radio)
        self.controls["frame_source_image"] = QLineEdit()
        btn_browse = QPushButton("选择图像")
        btn_browse.clicked.connect(self._select_frame_image)
        browse_row = QHBoxLayout()
        browse_row.addWidget(self.controls["frame_source_image"])
        browse_row.addWidget(btn_browse)
        browse_wrap = QWidget()
        browse_wrap.setLayout(browse_row)
        sim_layout.addRow("帧源图像", browse_wrap)
        self.controls["frame_source_image_2"] = QLineEdit()
        btn_browse_2 = QPushButton("选择图像")
        btn_browse_2.clicked.connect(self._select_frame_image_2)
        browse_row_2 = QHBoxLayout()
        browse_row_2.addWidget(self.controls["frame_source_image_2"])
        browse_row_2.addWidget(btn_browse_2)
        browse_wrap_2 = QWidget()
        browse_wrap_2.setLayout(browse_row_2)
        sim_layout.addRow("探测器2 帧源图像", browse_wrap_2)
        self.controls["sim_jitter_px"] = self._spin(0, 200, 0)
        self.controls["sim_noise_std"] = self._dspin(0.0, 100.0, 0.0, 2)
        sim_layout.addRow("模拟抖动 (px)", self.controls["sim_jitter_px"])
        sim_layout.addRow("模拟噪声标准差", self.controls["sim_noise_std"])
        sim_layout.addRow(QLabel("提示：不会驱动真实设备"))
        mode_row.addWidget(sim_card, 1)

        real_card = QGroupBox("真实设备模式")
        real_layout = QVBoxLayout(real_card)
        self.mode_real_radio = QCheckBox("启用真实设备模式")
        self.real_mode_arm = QCheckBox("风险确认：允许驱动真实设备(armed)")
        real_layout.addWidget(self.mode_real_radio)
        real_layout.addWidget(self.real_mode_arm)
        self.preflight_list = QListWidget()
        for item in [
            "已确认激光安全防护",
            "已确认设备连接",
            "已确认ROI 与目标区确认",
            "已确认启动自检参数",
            "已确认运行锁配置",
        ]:
            self.preflight_list.addItem(item)
        real_layout.addWidget(QLabel("运行前检查清单"))
        real_layout.addWidget(self.preflight_list, 1)
        mode_row.addWidget(real_card, 1)
        layout.addLayout(mode_row, 2)

        cfg_group = QGroupBox("闭环架构配置")
        cfg_form = QFormLayout(cfg_group)
        # 4轴/5轴 策略选择
        self.controls["alignment_strategy"] = self._combo(["dual_detector_4axis", "z_scan_legacy"])
        # 探测器模式
        self.controls["detector_mode"] = self._combo(["single_detector", "dual_detector"])
        self.controls["detector_backend"] = self._combo(["auto", "yolo", "classic"])
        self.controls["xy_driver"] = self._combo(["dryrun", "thorlabs", "newport", "newport-mrc4"])
        self.controls["z_driver"] = self._combo(["dryrun", "wheel", "xps", "picomotor"])
        self.controls["select_roi"] = QCheckBox()
        # 策略说明标签
        self.strategy_desc = QLabel("4轴模式：双镜闭环，禁用Z轴")
        self.strategy_desc.setStyleSheet("color: #888; font-style: italic;")
        # 策略联动：4轴模式禁用Z轴，5轴模式启用Z轴
        self.controls["alignment_strategy"].currentTextChanged.connect(self._on_alignment_strategy_changed)
        self.controls["detector_mode"].currentTextChanged.connect(self._on_detector_mode_changed)
        cfg_form.addRow("闭环策略", self.controls["alignment_strategy"])
        cfg_form.addRow(self.strategy_desc)
        cfg_form.addRow("探测器模式", self.controls["detector_mode"])
        cfg_form.addRow("检测后端", self.controls["detector_backend"])
        cfg_form.addRow("XY 驱动", self.controls["xy_driver"])
        cfg_form.addRow("Z 驱动", self.controls["z_driver"])
        cfg_form.addRow("启用 ROI", self.controls["select_roi"])
        layout.addWidget(cfg_group, 1)

        # 持续监控控制
        monitor_group = QGroupBox("持续监控（对准后保持运行）")
        monitor_layout = QHBoxLayout(monitor_group)
        btn_start_monitor = QPushButton("▶ 开始持续监控")
        btn_start_monitor.clicked.connect(self._start_continuous_monitoring)
        btn_stop_monitor = QPushButton("■ 停止监控")
        btn_stop_monitor.clicked.connect(self._stop_continuous_monitoring)
        self.monitor_status_label = QLabel("监控状态：未启动")
        self.monitor_status_label.setStyleSheet("color: #888;")
        monitor_layout.addWidget(btn_start_monitor)
        monitor_layout.addWidget(btn_stop_monitor)
        monitor_layout.addWidget(self.monitor_status_label)
        monitor_layout.addStretch(1)
        layout.addWidget(monitor_group)

        # UCC 相机实时预览
        ucc_preview_group = QGroupBox("UCC 相机实时预览")
        ucc_preview_layout = QVBoxLayout(ucc_preview_group)

        # 画面上方区域：左=预览画面，右=XY曲线+参数表
        ucc_top_layout = QHBoxLayout()

        # 左侧：预览画面
        self._ucc_preview_label = QLabel("点击「启动预览」打开 UCC 相机画面")
        self._ucc_preview_label.setAlignment(Qt.AlignCenter)
        self._ucc_preview_label.setMinimumSize(488, 360)
        self._ucc_preview_label.setStyleSheet(
            "background-color: #1a1a2e; color: #888; border: 1px solid #333; "
            "border-radius: 4px; font-size: 14px;"
        )
        ucc_top_layout.addWidget(self._ucc_preview_label, 3)

        # 右侧：XY 曲线 + 参数表 + RGB 剖面图
        ucc_right_layout = QVBoxLayout()

        # XY 曲线
        ucc_curve_group = QGroupBox("XY 位置曲线 (XYCurve)")
        ucc_curve_layout = QVBoxLayout(ucc_curve_group)
        self._ucc_curve_widget = SpotCurveWidget(max_points=200)
        ucc_curve_layout.addWidget(self._ucc_curve_widget)
        ucc_right_layout.addWidget(ucc_curve_group, 1)

        # 参数表 (Calcult)
        ucc_param_group = QGroupBox("光斑参数 (Calcult)")
        ucc_param_layout = QVBoxLayout(ucc_param_group)
        self._ucc_param_table = QTableWidget()
        self._ucc_param_table.setColumnCount(2)
        self._ucc_param_table.setHorizontalHeaderLabels(["参数", "数值"])
        self._ucc_param_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._ucc_param_table.setSelectionMode(QTableWidget.NoSelection)
        self._ucc_param_table.verticalHeader().setVisible(False)
        self._ucc_param_table.horizontalHeader().setStretchLastSection(True)
        self._ucc_param_table.setMaximumHeight(140)
        # 初始化空行
        param_names = ["Min, Peak", "Peak Loc (X, Y)", "Centroid (X, Y)", "Width (X, Y)"]
        self._ucc_param_table.setRowCount(len(param_names))
        for i, name in enumerate(param_names):
            self._ucc_param_table.setItem(i, 0, QTableWidgetItem(name))
            self._ucc_param_table.setItem(i, 1, QTableWidgetItem("--"))
        ucc_param_layout.addWidget(self._ucc_param_table)
        ucc_right_layout.addWidget(ucc_param_group, 1)

        # RGB 剖面图 (Profile)
        ucc_profile_group = QGroupBox("光斑 RGB 剖面图 (Profile)")
        ucc_profile_layout = QVBoxLayout(ucc_profile_group)
        self._ucc_profile_widget = SpotProfileWidget()
        ucc_profile_layout.addWidget(self._ucc_profile_widget)
        ucc_right_layout.addWidget(ucc_profile_group, 2)

        ucc_top_layout.addLayout(ucc_right_layout, 2)
        ucc_preview_layout.addLayout(ucc_top_layout)

        # 控制行
        ucc_control_row = QHBoxLayout()
        self._ucc_preview_device = self._spin(0, 9, 1)
        self._ucc_preview_device.setPrefix("设备 ")
        self._ucc_preview_resolution = self._combo(["PAL", "NTSC", "AUTO"])
        self._ucc_preview_resolution.setCurrentText("AUTO")

        self._ucc_preview_btn_start = QPushButton("▶ 启动预览")
        self._ucc_preview_btn_start.clicked.connect(self._start_ucc_preview)
        self._ucc_preview_btn_stop = QPushButton("■ 停止预览")
        self._ucc_preview_btn_stop.clicked.connect(self._stop_ucc_preview)
        self._ucc_preview_btn_stop.setEnabled(False)

        self._ucc_preview_info = QLabel("状态: 未启动")
        self._ucc_preview_info.setStyleSheet("color: #888;")

        self._ucc_preview_format = self._combo(["AUTO", "MJPG", "YUY2", "YUYV", "UYVY"])
        self._ucc_preview_format.setCurrentText("AUTO")

        self._ucc_preview_btn_save = QPushButton("💾 保存调试帧")
        self._ucc_preview_btn_save.clicked.connect(self._save_ucc_debug_frame)
        self._ucc_preview_btn_save.setEnabled(False)

        ucc_control_row.addWidget(QLabel("设备索引:"))
        ucc_control_row.addWidget(self._ucc_preview_device)
        ucc_control_row.addWidget(QLabel("分辨率:"))
        ucc_control_row.addWidget(self._ucc_preview_resolution)
        ucc_control_row.addWidget(QLabel("像素格式:"))
        ucc_control_row.addWidget(self._ucc_preview_format)
        ucc_control_row.addWidget(self._ucc_preview_btn_start)
        ucc_control_row.addWidget(self._ucc_preview_btn_stop)
        ucc_control_row.addWidget(self._ucc_preview_btn_save)
        ucc_control_row.addWidget(self._ucc_preview_info)
        ucc_control_row.addStretch(1)
        ucc_preview_layout.addLayout(ucc_control_row)

        layout.addWidget(ucc_preview_group)

        self.run_mode_snapshot = QPlainTextEdit()
        self.run_mode_snapshot.setReadOnly(True)
        layout.addWidget(QLabel("配置快照 / 最近一次真实运行记录"))
        layout.addWidget(self.run_mode_snapshot, 1)
        return page

    def _build_simulation_lab_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        top = QHBoxLayout()
        btn_sample = QPushButton("生成示例图像")
        btn_sample.clicked.connect(self._use_sample_frame)
        btn_sim_test = QPushButton("运行模拟图像测试")
        btn_sim_test.clicked.connect(lambda: self._run_device_test_by_id("simulated_frame"))
        btn_classic = QPushButton("Classic 检测测试")
        btn_classic.clicked.connect(lambda: self._run_device_test_by_id("classic_backend"))
        btn_yolo = QPushButton("YOLO 检测测试")
        btn_yolo.clicked.connect(lambda: self._run_device_test_by_id("yolo_backend"))
        for btn in [btn_sample, btn_sim_test, btn_classic, btn_yolo]:
            top.addWidget(btn)
        top.addStretch(1)
        layout.addLayout(top)
        self.sim_preview = QLabel("仿真预览")
        self.sim_preview.setMinimumSize(760, 500)
        self.sim_preview.setAlignment(Qt.AlignCenter)
        self.sim_preview.setStyleSheet("background:#060A10; border:1px solid #2B3642;")
        layout.addWidget(self.sim_preview, 1)
        return page

    def _build_logs_report_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        btn_row = QHBoxLayout()
        btn_refresh = QPushButton("刷新日志/事件/统计")
        btn_refresh.clicked.connect(self._refresh_runtime_streams)
        btn_export = QPushButton("导出运行报告")
        btn_export.clicked.connect(self._export_report)
        btn_row.addWidget(btn_refresh)
        btn_row.addWidget(btn_export)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self.logs_text = QPlainTextEdit()
        self.logs_text.setReadOnly(True)
        layout.addWidget(QLabel("实时日志"), 0)
        layout.addWidget(self.logs_text, 2)

        self.report_text = QPlainTextEdit()
        self.report_text.setReadOnly(True)
        layout.addWidget(QLabel("运行报告 / 统计"), 0)
        layout.addWidget(self.report_text, 2)
        return page

    def _build_settings_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        body_layout = QVBoxLayout(body)

        for title, rows in [
            ("基础参数", [("window_title", "窗口标题"), ("window_wait_seconds", "窗口等待秒数"), ("log_level", "日志级别")]),
            ("运动参数", [("newport_timeout", "Newport 超时"), ("z_step", "Z 步长"), ("disable_run_lock", "禁用运行锁")]),
            ("ROI / 图像参数", [("sim_jitter_px", "模拟抖动"), ("sim_noise_std", "模拟噪声"), ("select_roi", "启用 ROI"), ("select_roi_2", "探测器2 启用 ROI")]),
            ("安全参数", [("startup_motion_check_timeout", "启动自检超时"), ("startup_motion_check_xy_steps", "启动自检 XY 步数")]),
            ("双探测器参数", [("detector2_focal_length", "探测器2 焦距 (mm)"), ("comparison_mode", "对比模式"), ("detector_weight", "主探测器权重"), ("touview_weight", "ToupView 权重"), ("disagreement_threshold_px", "不一致阈值 (px)")]),
            ("UCC 探测器参数", [("ucc_device", "探测器1 设备索引"), ("ucc_resolution", "探测器1 分辨率"), ("ucc_exposure", "探测器1 曝光"), ("ucc_gain", "探测器1 增益"), ("ucc_brightness", "探测器1 亮度"), ("ucc_contrast", "探测器1 对比度"), ("ucc_device_2", "探测器2 设备索引"), ("ucc_resolution_2", "探测器2 分辨率")]),
            ("中间帧保存", [("save_intermediate_frames", "保存ROI中间帧"), ("frame_cache_enabled", "启用帧缓存 (./Tmp_Frames)")]),
        ]:
            group = QGroupBox(title)
            form = QFormLayout(group)
            for key, label in rows:
                if key not in self.controls:
                    if key == "window_title":
                        self.controls[key] = QLineEdit()
                    elif key == "window_wait_seconds":
                        self.controls[key] = self._dspin(1.0, 120.0, 10.0, 1)
                    elif key == "log_level":
                        self.controls[key] = self._combo(["DEBUG", "INFO", "WARNING", "ERROR"])
                    elif key == "newport_timeout":
                        self.controls[key] = self._dspin(0.1, 60.0, 5.0, 1)
                    elif key == "disable_run_lock":
                        self.controls[key] = QCheckBox()
                    elif key == "select_roi_2":
                        self.controls[key] = QCheckBox()
                    elif key == "detector2_focal_length":
                        self.controls[key] = self._dspin(50.0, 500.0, 200.0, 1)
                    elif key == "comparison_mode":
                        self.controls[key] = self._combo(["detector_primary", "touview_primary", "average", "weighted"])
                    elif key == "detector_weight":
                        self.controls[key] = self._dspin(0.0, 1.0, 0.7, 2)
                    elif key == "touview_weight":
                        self.controls[key] = self._dspin(0.0, 1.0, 0.3, 2)
                    elif key == "disagreement_threshold_px":
                        self.controls[key] = self._dspin(0.0, 50.0, 10.0, 1)
                    elif key == "ucc_device":
                        self.controls[key] = self._spin(0, 10, 0)
                    elif key == "ucc_device_2":
                        self.controls[key] = self._spin(0, 10, 0)
                    elif key == "ucc_resolution":
                        self.controls[key] = self._combo(["PAL", "NTSC", "AUTO"])
                    elif key == "ucc_resolution_2":
                        self.controls[key] = self._combo(["PAL", "NTSC", "AUTO"])
                    elif key == "save_intermediate_frames":
                        self.controls[key] = QCheckBox()
                    elif key == "frame_cache_enabled":
                        self.controls[key] = QCheckBox()
                    elif key in ("ucc_exposure", "ucc_gain", "ucc_brightness", "ucc_contrast"):
                        self.controls[key] = self._dspin(-100.0, 100.0, 0.0, 1)
                    else:
                        continue
                form.addRow(label, self.controls[key])
            body_layout.addWidget(group)

        # frame_tmp_dir 带浏览按钮的行
        tmp_dir_group = QGroupBox("中间帧保存目录")
        tmp_dir_layout = QHBoxLayout(tmp_dir_group)
        self.controls["frame_tmp_dir"] = QLineEdit("FrameTmp")
        btn_browse_tmp = QPushButton("选择目录")
        btn_browse_tmp.clicked.connect(lambda: self._select_frame_tmp_dir())
        tmp_dir_layout.addWidget(self.controls["frame_tmp_dir"])
        tmp_dir_layout.addWidget(btn_browse_tmp)
        body_layout.addWidget(tmp_dir_group)

        io_group = QGroupBox("配置导入/导出")
        io_layout = QHBoxLayout(io_group)
        btn_export_cfg = QPushButton("导出配置")
        btn_export_cfg.clicked.connect(self._export_config_json)
        btn_import_cfg = QPushButton("导入配置")
        btn_import_cfg.clicked.connect(self._import_config_json)
        btn_default = QPushButton("恢复默认值")
        btn_default.clicked.connect(self._reset_to_default_profile)
        io_layout.addWidget(btn_export_cfg)
        io_layout.addWidget(btn_import_cfg)
        io_layout.addWidget(btn_default)
        io_layout.addStretch(1)
        body_layout.addWidget(io_group)
        body_layout.addStretch(1)

        scroll.setWidget(body)
        layout.addWidget(scroll, 1)
        return page

    def _load_ui_preferences(self) -> UiPreferences:
        prefs = UiPreferences()
        try:
            raw = self.runtime.read_run_report(self.runtime.repo_root / "artifacts" / "ui_prefs.json")
            if isinstance(raw, dict):
                prefs.theme_dark = bool(raw.get("theme_dark", prefs.theme_dark))
                titles = raw.get("autozoom_window_titles", [])
                if isinstance(titles, list):
                    prefs.autozoom_window_titles = [str(v) for v in titles if str(v).strip()]
        except Exception:
            pass
        return prefs

    def _save_ui_preferences(self) -> None:
        try:
            payload = {
                "theme_dark": bool(getattr(self, "_theme_dark", True)),
                "autozoom_window_titles": list(getattr(self, "_autozoom_window_title_history", [])),
            }
            path = self.runtime.repo_root / "artifacts" / "ui_prefs.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            self._append_log(f"[UI] 保存偏好失败: {exc}")

    def _update_status_bar(self, page_name: Optional[str] = None, message: Optional[str] = None) -> None:
        parts = [
            f"页面={page_name or self.current_page_name()}",
            f"主题={'深色' if getattr(self, '_theme_dark', True) else '浅色'}",
            f"运行模式={self.profile.run_mode.value}",
            f"运行状态={status_text(self.run_status)}",
        ]
        if message:
            parts.append(message)
        self.statusBar().showMessage(" | ".join(parts))

    def current_page_name(self) -> str:
        idx = self.nav_list.currentRow() if hasattr(self, "nav_list") else -1
        if 0 <= idx < len(self.NAV_ITEMS):
            return self.NAV_ITEMS[idx]
        return "-"

    def _switch_page(self, idx: int) -> None:
        if hasattr(self, "pages") and 0 <= idx < self.pages.count():
            self.pages.setCurrentIndex(idx)
            self._update_status_bar(page_name=self.NAV_ITEMS[idx] if idx < len(self.NAV_ITEMS) else None)

    def _spin(self, min_v: int, max_v: int, val: int) -> QSpinBox:
        w = QSpinBox()
        w.setRange(min_v, max_v)
        w.setValue(val)
        return w

    def _dspin(self, min_v: float, max_v: float, val: float, decimals: int) -> QDoubleSpinBox:
        w = QDoubleSpinBox()
        w.setRange(min_v, max_v)
        w.setDecimals(decimals)
        w.setValue(val)
        return w

    def _combo(self, values: Iterable[str]) -> QComboBox:
        w = QComboBox()
        w.addItems(list(values))
        return w

    def _reset_to_default_profile(self) -> None:
        self.profile = self.runtime.default_profile()
        self._apply_profile_to_controls(self.profile)
        self._refresh_all_panels()
        self._save_ui_preferences()
        self._append_log("已恢复默认配置值")

    def _apply_profile_to_controls(self, p: RuntimeProfile) -> None:
        self.mode_sim_radio.setChecked(p.run_mode == RunMode.SIMULATION)
        self.mode_real_radio.setChecked(p.run_mode == RunMode.REAL)

        self._set_combo("detector_backend", p.detector_backend)
        self._set_combo("detector_mode", p.detector_mode)
        # 校正镜组（准直工作台用）
        idx = self._alignment_correction_mirror.findText(p.correction_mirror)
        if idx >= 0:
            self._alignment_correction_mirror.setCurrentIndex(idx)
        self._set_combo("alignment_strategy", p.alignment_strategy.value)
        self._set_combo("xy_driver", p.xy_driver)
        self._set_combo("z_driver", p.z_driver)
        self._set_text("frame_source_image", p.frame_source_image or "")
        self._set_text("frame_source_image_2", p.frame_source_image_2 or "")
        self._set_spin("sim_jitter_px", p.sim_jitter_px)
        self._set_dspin("sim_noise_std", p.sim_noise_std)
        self._set_check("select_roi", p.select_roi)
        self._set_check("select_roi_2", p.select_roi_2)
        self._set_spin("tolerance_px", p.tolerance_px)
        self._set_spin("detect_retry", p.detect_retry)
        self._set_dspin("detect_retry_interval", p.detect_retry_interval)
        self._set_dspin("settle_time", p.settle_time)
        self._set_spin("max_align_rounds", p.max_align_rounds)
        self._set_spin("max_iterations", p.max_iterations)
        self._set_spin("x_move_step", p.x_move_step)
        self._set_spin("y_move_step", p.y_move_step)
        self._set_dspin("z_step", p.z_step)
        self._set_dspin("min_focus_score", p.min_focus_score)
        self._set_check("adaptive_step", p.adaptive_step)
        self._set_check("enable_recovery_scan", p.enable_recovery_scan)
        self._set_check("startup_motion_check_enabled", p.startup_motion_check_enabled)
        self._set_check("frame_cache_enabled", p.frame_cache_enabled)
        self._set_check("disable_z_axis", p.disable_z_axis)
        self._set_dspin("startup_motion_check_timeout", p.startup_motion_check_timeout)
        self._set_spin("startup_motion_check_xy_steps", p.startup_motion_check_xy_steps)
        self._set_dspin("startup_motion_check_z_step", p.startup_motion_check_z_step)
        self._set_spin("newport_conn", p.newport_conn)
        self._set_spin("mrc_mirror1_x_axis", p.mrc_mirror1_x_axis)
        self._set_spin("mrc_mirror1_y_axis", p.mrc_mirror1_y_axis)
        self._set_spin("mrc_mirror2_x_axis", p.mrc_mirror2_x_axis)
        self._set_spin("mrc_mirror2_y_axis", p.mrc_mirror2_y_axis)
        self._set_combo("mrc_mirror1_x_sign", str(p.mrc_mirror1_x_sign))
        self._set_combo("mrc_mirror1_y_sign", str(p.mrc_mirror1_y_sign))
        self._set_combo("mrc_mirror2_x_sign", str(p.mrc_mirror2_x_sign))
        self._set_combo("mrc_mirror2_y_sign", str(p.mrc_mirror2_y_sign))
        self._set_combo("mrc_virtual_axis_mode", p.mrc_virtual_axis_mode)
        self._set_spin("z_picomotor_conn", p.z_picomotor_conn)
        self._set_spin("z_picomotor_axis", p.z_picomotor_axis)
        self._set_combo("z_picomotor_sign", str(p.z_picomotor_sign))
        self._set_spin("z_picomotor_velocity", int(p.z_picomotor_velocity or 0))
        self._set_spin("z_picomotor_acceleration", int(p.z_picomotor_acceleration or 0))
        self._set_text("window_title", p.window_title)
        self._set_dspin("window_wait_seconds", p.window_wait_seconds)
        self._set_combo("log_level", p.log_level)
        self._set_dspin("newport_timeout", p.newport_timeout)
        self._set_check("disable_run_lock", p.disable_run_lock)
        self._set_dspin("detector2_focal_length", p.detector2_focal_length)
        self._set_combo("comparison_mode", p.comparison_mode)
        self._set_dspin("detector_weight", p.detector_weight)
        self._set_dspin("touview_weight", p.touview_weight)
        self._set_dspin("disagreement_threshold_px", p.disagreement_threshold_px)
        # UCC 探测器参数
        self._set_spin("ucc_device", p.ucc_device if p.ucc_device is not None else 0)
        self._set_combo("ucc_resolution", p.ucc_resolution)
        self._set_dspin("ucc_exposure", p.ucc_exposure if p.ucc_exposure is not None else 0.0)
        self._set_dspin("ucc_gain", p.ucc_gain if p.ucc_gain is not None else 0.0)
        self._set_dspin("ucc_brightness", p.ucc_brightness if p.ucc_brightness is not None else 0.0)
        self._set_dspin("ucc_contrast", p.ucc_contrast if p.ucc_contrast is not None else 0.0)
        self._set_spin("ucc_device_2", p.ucc_device_2 if p.ucc_device_2 is not None else 0)
        self._set_combo("ucc_resolution_2", p.ucc_resolution_2)
        # 中间帧保存
        self._set_check("save_intermediate_frames", p.save_intermediate_frames)
        self._set_text("frame_tmp_dir", p.frame_tmp_dir)

    def _collect_profile_from_controls(self) -> RuntimeProfile:
        p = self.profile
        p.run_mode = RunMode.REAL if self.mode_real_radio.isChecked() else RunMode.SIMULATION
        p.detector_backend = self._combo_value("detector_backend", p.detector_backend)
        p.detector_mode = self._combo_value("detector_mode", p.detector_mode)
        strategy_text = self._combo_value("alignment_strategy", p.alignment_strategy.value)
        p.alignment_strategy = AlignmentStrategy(strategy_text)
        p.xy_driver = self._combo_value("xy_driver", p.xy_driver)
        p.z_driver = self._combo_value("z_driver", p.z_driver)
        p.frame_source_image = self._text_value("frame_source_image", p.frame_source_image or "")
        p.frame_source_image_2 = self._text_value("frame_source_image_2", p.frame_source_image_2 or "")
        p.sim_jitter_px = self._spin_value("sim_jitter_px", p.sim_jitter_px)
        p.sim_noise_std = self._dspin_value("sim_noise_std", p.sim_noise_std)
        p.select_roi = self._check_value("select_roi", p.select_roi)
        p.select_roi_2 = self._check_value("select_roi_2", p.select_roi_2)
        p.tolerance_px = self._spin_value("tolerance_px", p.tolerance_px)
        p.detect_retry = self._spin_value("detect_retry", p.detect_retry)
        p.detect_retry_interval = self._dspin_value("detect_retry_interval", p.detect_retry_interval)
        p.settle_time = self._dspin_value("settle_time", p.settle_time)
        p.max_align_rounds = self._spin_value("max_align_rounds", p.max_align_rounds)
        p.max_iterations = self._spin_value("max_iterations", p.max_iterations)
        p.x_move_step = self._spin_value("x_move_step", p.x_move_step)
        p.y_move_step = self._spin_value("y_move_step", p.y_move_step)
        p.z_step = self._dspin_value("z_step", p.z_step)
        p.min_focus_score = self._dspin_value("min_focus_score", p.min_focus_score)
        p.adaptive_step = self._check_value("adaptive_step", p.adaptive_step)
        p.enable_recovery_scan = self._check_value("enable_recovery_scan", p.enable_recovery_scan)
        p.startup_motion_check_enabled = self._check_value("startup_motion_check_enabled", p.startup_motion_check_enabled)
        p.frame_cache_enabled = self._check_value("frame_cache_enabled", p.frame_cache_enabled)
        p.disable_z_axis = self._check_value("disable_z_axis", p.disable_z_axis)
        p.startup_motion_check_timeout = self._dspin_value("startup_motion_check_timeout", p.startup_motion_check_timeout)
        p.startup_motion_check_xy_steps = self._spin_value("startup_motion_check_xy_steps", p.startup_motion_check_xy_steps)
        p.startup_motion_check_z_step = self._dspin_value("startup_motion_check_z_step", p.startup_motion_check_z_step)
        p.newport_conn = self._spin_value("newport_conn", p.newport_conn)
        p.mrc_mirror1_x_axis = self._spin_value("mrc_mirror1_x_axis", p.mrc_mirror1_x_axis)
        p.mrc_mirror1_y_axis = self._spin_value("mrc_mirror1_y_axis", p.mrc_mirror1_y_axis)
        p.mrc_mirror2_x_axis = self._spin_value("mrc_mirror2_x_axis", p.mrc_mirror2_x_axis)
        p.mrc_mirror2_y_axis = self._spin_value("mrc_mirror2_y_axis", p.mrc_mirror2_y_axis)
        p.mrc_mirror1_x_sign = int(self._combo_value("mrc_mirror1_x_sign", str(p.mrc_mirror1_x_sign)))
        p.mrc_mirror1_y_sign = int(self._combo_value("mrc_mirror1_y_sign", str(p.mrc_mirror1_y_sign)))
        p.mrc_mirror2_x_sign = int(self._combo_value("mrc_mirror2_x_sign", str(p.mrc_mirror2_x_sign)))
        p.mrc_mirror2_y_sign = int(self._combo_value("mrc_mirror2_y_sign", str(p.mrc_mirror2_y_sign)))
        p.mrc_virtual_axis_mode = self._combo_value("mrc_virtual_axis_mode", p.mrc_virtual_axis_mode)
        p.z_picomotor_conn = self._spin_value("z_picomotor_conn", p.z_picomotor_conn)
        p.z_picomotor_axis = self._spin_value("z_picomotor_axis", p.z_picomotor_axis)
        p.z_picomotor_sign = int(self._combo_value("z_picomotor_sign", str(p.z_picomotor_sign)))
        z_vel = self._spin_value("z_picomotor_velocity", int(p.z_picomotor_velocity or 0))
        z_acc = self._spin_value("z_picomotor_acceleration", int(p.z_picomotor_acceleration or 0))
        p.z_picomotor_velocity = None if z_vel <= 0 else z_vel
        p.z_picomotor_acceleration = None if z_acc <= 0 else z_acc
        p.window_title = self._text_value("window_title", p.window_title)
        p.window_wait_seconds = self._dspin_value("window_wait_seconds", p.window_wait_seconds)
        p.log_level = self._combo_value("log_level", p.log_level)
        p.newport_timeout = self._dspin_value("newport_timeout", p.newport_timeout)
        p.disable_run_lock = self._check_value("disable_run_lock", p.disable_run_lock)
        p.detector2_focal_length = self._dspin_value("detector2_focal_length", p.detector2_focal_length)
        p.comparison_mode = self._combo_value("comparison_mode", p.comparison_mode)
        p.detector_weight = self._dspin_value("detector_weight", p.detector_weight)
        p.touview_weight = self._dspin_value("touview_weight", p.touview_weight)
        p.disagreement_threshold_px = self._dspin_value("disagreement_threshold_px", p.disagreement_threshold_px)
        # 校正镜组（准直工作台对齐用）
        p.correction_mirror = self._alignment_correction_mirror.currentText()
        # UCC 探测器参数
        p.ucc_device = self._spin_value("ucc_device", p.ucc_device or 0)
        p.ucc_resolution = self._combo_value("ucc_resolution", p.ucc_resolution)
        p.ucc_exposure = self._dspin_value("ucc_exposure", p.ucc_exposure or 0.0)
        p.ucc_gain = self._dspin_value("ucc_gain", p.ucc_gain or 0.0)
        p.ucc_brightness = self._dspin_value("ucc_brightness", p.ucc_brightness or 0.0)
        p.ucc_contrast = self._dspin_value("ucc_contrast", p.ucc_contrast or 0.0)
        p.ucc_device_2 = self._spin_value("ucc_device_2", p.ucc_device_2 or 0)
        p.ucc_resolution_2 = self._combo_value("ucc_resolution_2", p.ucc_resolution_2)
        # 中间帧保存
        p.save_intermediate_frames = self._check_value("save_intermediate_frames", p.save_intermediate_frames)
        p.frame_tmp_dir = self._text_value("frame_tmp_dir", p.frame_tmp_dir)
        return p

    def _set_combo(self, key: str, value: str) -> None:
        widget = self.controls.get(key)
        if isinstance(widget, QComboBox):
            idx = widget.findText(str(value))
            if idx >= 0:
                widget.setCurrentIndex(idx)

    def _set_text(self, key: str, value: str) -> None:
        widget = self.controls.get(key)
        if isinstance(widget, QLineEdit):
            widget.setText(str(value))

    def _set_spin(self, key: str, value: int) -> None:
        widget = self.controls.get(key)
        if isinstance(widget, QSpinBox):
            widget.setValue(int(value))

    def _set_dspin(self, key: str, value: float) -> None:
        widget = self.controls.get(key)
        if isinstance(widget, QDoubleSpinBox):
            widget.setValue(float(value))

    def _set_check(self, key: str, value: bool) -> None:
        widget = self.controls.get(key)
        if isinstance(widget, QCheckBox):
            widget.setChecked(bool(value))

    def _combo_value(self, key: str, default: str) -> str:
        widget = self.controls.get(key)
        if isinstance(widget, QComboBox):
            return widget.currentText().strip() or default
        return default

    def _text_value(self, key: str, default: str) -> str:
        widget = self.controls.get(key)
        if isinstance(widget, QLineEdit):
            return widget.text().strip() or default
        return default

    def _spin_value(self, key: str, default: int) -> int:
        widget = self.controls.get(key)
        if isinstance(widget, QSpinBox):
            return int(widget.value())
        return default

    def _dspin_value(self, key: str, default: float) -> float:
        widget = self.controls.get(key)
        if isinstance(widget, QDoubleSpinBox):
            return float(widget.value())
        return default

    def _check_value(self, key: str, default: bool) -> bool:
        widget = self.controls.get(key)
        if isinstance(widget, QCheckBox):
            return bool(widget.isChecked())
        return default

    def _refresh_all_panels(self) -> None:
        self.profile = self._collect_profile_from_controls()
        self._refresh_status_headers()
        self._refresh_dashboard()
        self._refresh_device_center()
        self._refresh_module_catalog()
        self._refresh_mrc_allocation()
        self._refresh_preview_images()

    def _refresh_status_headers(self) -> None:
        snap = self.runtime.build_snapshot(
            self.profile,
            env_status=self.env_status,
            startup_status=self.startup_status,
            run_status=self.run_status,
        )
        status_map = {
            "mode": snap.mode_label,
            "backend": f"{snap.requested_backend}/{snap.resolved_backend}",
            "xy": snap.xy_driver,
            "z": snap.z_driver,
            "devices": snap.device_status_label,
            "run_lock": status_text(snap.run_lock_status),
            "env": status_text(self.env_status),
            "startup": status_text(self.startup_status),
            "run": status_text(self.run_status),
        }
        for key, val in status_map.items():
            w = self.status_fields.get(key)
            if w is None:
                continue
            w.setText(str(val))
        self.status_fields["env"].setStyleSheet(STATUS_STYLE[self.env_status])
        self.status_fields["startup"].setStyleSheet(STATUS_STYLE[self.startup_status])
        self.status_fields["run"].setStyleSheet(STATUS_STYLE[self.run_status])

    def _refresh_dashboard(self) -> None:
        snap = self.runtime.build_snapshot(
            self.profile,
            env_status=self.env_status,
            startup_status=self.startup_status,
            run_status=self.run_status,
        )
        mapping = {
            "mode": snap.mode_label,
            "requested_backend": snap.requested_backend,
            "resolved_backend": snap.resolved_backend,
            "xy_driver": snap.xy_driver,
            "z_driver": snap.z_driver,
            "roi": snap.roi_label,
            "target": snap.target_label,
            "device": snap.device_status_label,
            "env": status_text(self.env_status),
            "startup": status_text(self.startup_status),
            "run": status_text(self.run_status),
            "risk": snap.risk_label,
        }
        for key, val in mapping.items():
            if key in self.dashboard_fields:
                self.dashboard_fields[key].setText(str(val))

        events = self.runtime.read_recent_events(self.profile.event_stream_jsonl, limit=20)
        self._fill_event_table(self.dashboard_event_table, events)
        self._fill_event_table(self.quick_event_table, events[:10])

    def _refresh_device_center(self) -> None:
        self.device_topology_tree.clear()
        root_specs = [
            ("图像源", ["NIS", "Simulated Frame Source"]),             # 默认为ToupView
            ("XY 子系统", ["Thorlabs", "Newport", "Newport MRC 4-axis"]),
            ("Z 子系统", ["Wheel", "XPS", "Picomotor Z"]),
            ("检测后端", ["YOLO", "Classic"]),
            ("检查子系统", ["环境检查", "启动运动自检"]),
        ]
        for title, children in root_specs:
            root = QTreeWidgetItem([title, "normal"])
            for child in children:
                root.addChild(QTreeWidgetItem([child, "standby"]))
            self.device_topology_tree.addTopLevelItem(root)

        panels = self.device_registry.build_panels(self.profile, self.runtime)
        self.device_panel_table.setRowCount(len(panels))
        detail_rows = []
        for row, panel in enumerate(panels):
            self.device_panel_table.setItem(row, 0, QTableWidgetItem(panel.title))
            self.device_panel_table.setItem(row, 1, QTableWidgetItem(status_text(panel.status)))
            self.device_panel_table.setItem(row, 2, QTableWidgetItem(panel.summary))
            for k, v in panel.details.items():
                detail_rows.append(f"{panel.title}.{k}: {v}")
        self.device_detail_text.setPlainText("\n".join(detail_rows))

    def _refresh_module_catalog(self) -> None:
        rows = self.module_catalog.list_modules(self.profile)
        self.module_rows = rows
        if self.module_type_filter.count() <= 1:
            seen = sorted({row.type_label for row in rows})
            for item in seen:
                self.module_type_filter.addItem(item)
            self.module_type_filter.setCurrentRow(0)
        self._refresh_module_table()

    def _refresh_module_table(self) -> None:
        if not hasattr(self, "module_table"):
            return
        search = self.module_search.text().strip().lower() if hasattr(self, "module_search") else ""
        type_filter = self.module_type_filter.currentItem().text() if self.module_type_filter.currentItem() else "全部类型"
        ver_text = self.module_version_filter.currentText() if hasattr(self, "module_version_filter") else "全部版本"
        integrated_only = self.module_integrated_only.isChecked() if hasattr(self, "module_integrated_only") else False

        filtered = []
        for row in self.module_rows:
            if search and search not in row.name.lower() and search not in row.title.lower():
                continue
            if type_filter != "全部类型" and row.type_label != type_filter:
                continue
            if ver_text != "全部版本" and row.version != int(ver_text.replace("v", "")):
                continue
            if integrated_only and row.status != UiStatus.RUNNING:
                continue
            filtered.append(row)

        self.module_table.setRowCount(len(filtered))
        for i, row in enumerate(filtered):
            values = [
                row.name,
                f"v{row.version}",
                row.type_label,
                status_text(row.status),
                row.placement,
                row.summary,
                row.config_symbol or "-",
                row.docs_path,
            ]
            for c, value in enumerate(values):
                self.module_table.setItem(i, c, QTableWidgetItem(str(value)))
            self.module_table.item(i, 0).setData(Qt.UserRole, row)

    def _show_selected_module_detail(self) -> None:
        row = self.module_table.currentRow()
        if row < 0:
            return
        item = self.module_table.item(row, 0)
        if item is None:
            return
        module = item.data(Qt.UserRole)
        if module is None:
            return
        detail = {
            "模块名": module.name,
            "版本": f"v{module.version}",
            "类型": module.type_label,
            "状态": status_text(module.status),
            "接入位置": module.placement,
            "import_path": module.import_path,
            "source_path": module.source_path,
            "primary_symbol": module.primary_symbol,
            "config_symbol": module.config_symbol,
            "docs_path": module.docs_path,
            "依赖缺失原因": module.dependency_reason or "-",
            "说明": module.summary,
        }
        self.module_detail_text.setPlainText("\n".join(f"{k}: {v}" for k, v in detail.items()))

    def _refresh_mrc_allocation(self) -> None:
        self.profile = self._collect_profile_from_controls()
        preview = self.device_registry.mrc_allocation_preview(self.profile)
        self.mrc_alloc_label.setText(preview)

    def _fill_event_table(self, table: QTableWidget, rows: List[EventRecord]) -> None:
        table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            table.setItem(i, 0, QTableWidgetItem(row.timestamp))
            table.setItem(i, 1, QTableWidgetItem(row.event_name))
            table.setItem(i, 2, QTableWidgetItem(row.payload_summary))

    def _show_selected_autozoom_detail(self) -> None:
        row = self.autozoom_table.currentRow()
        if row < 0:
            return
        item = self.autozoom_table.item(row, 0)
        if item is None:
            return
        module = item.data(Qt.UserRole)
        if module is None:
            return
        detail = {
            "模块名": module.name,
            "版本": f"v{module.version}",
            "类型": module.type_label,
            "状态": status_text(module.status),
            "接入位置": module.placement,
            "import_path": module.import_path,
            "source_path": module.source_path,
            "primary_symbol": module.primary_symbol,
            "config_symbol": module.config_symbol,
            "docs_path": module.docs_path,
            "依赖缺失原因": module.dependency_reason or "-",
            "说明": module.summary,
        }
        self.autozoom_detail.setPlainText("\n".join(f"{k}: {v}" for k, v in detail.items()))
        if hasattr(self, "autozoom_log"):
            self.autozoom_log.appendPlainText(f"[AutoZoom] 当前选中: {module.title} ({module.name})")

    def _refresh_runtime_streams(self) -> None:
        self._refresh_dashboard()
        report = self.runtime.read_run_report(self.profile.run_report_json)
        self.report_text.setPlainText(json.dumps(report, ensure_ascii=False, indent=2) if report else "{}")
        self.run_mode_snapshot.setPlainText(json.dumps(report, ensure_ascii=False, indent=2) if report else "{}")

    def _append_log(self, text: str) -> None:
        logs = getattr(self, "logs_text", None)
        if isinstance(logs, QPlainTextEdit):
            logs.appendPlainText(text)
        runtime_text = getattr(self, "runtime_status_text", None)
        if isinstance(runtime_text, QTextEdit):
            runtime_text.append(text)

    def _decode_bytes(self, raw: bytes) -> str:
        if not raw:
            return ""
        for enc in ("utf-8", "gbk", "latin-1"):
            try:
                return raw.decode(enc)
            except Exception:
                continue
        return raw.decode("utf-8", errors="ignore")

    def _run_env_check(self) -> None:
        self.profile = self._collect_profile_from_controls()
        result = self.runtime.run_command(self.profile, check_env=True, timeout=180.0)
        self.env_status = result.status
        self._append_log(result.command_preview)
        self._append_log(result.stdout or result.stderr or result.message)
        self._refresh_all_panels()

    def _run_startup_check(self) -> None:
        self.profile = self._collect_profile_from_controls()
        result = self.runtime.run_command(self.profile, startup_check_only=True, timeout=180.0)
        self.startup_status = result.status
        self._append_log(result.command_preview)
        self._append_log(result.stdout or result.stderr or result.message)
        self._refresh_all_panels()

    def _single_step_run(self) -> None:
        self.profile = self._collect_profile_from_controls()
        result = self.runtime.run_command(self.profile, single_step=True, timeout=180.0)
        self.run_status = result.status
        self._append_log(result.command_preview)
        self._append_log(result.stdout or result.stderr or result.message)
        self._refresh_all_panels()

    def _start_alignment(self) -> None:
        if self.run_process is not None and self.run_process.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "运行中", "当前已有运行中的任务")
            return
        self.profile = self._collect_profile_from_controls()
        if self.profile.run_mode == RunMode.REAL and not self.real_mode_arm.isChecked():
            QMessageBox.warning(self, "风险确认", "真实设备模式需要先勾选armed 风险确认")
            return

        cmd = self.runtime.build_command(self.profile)
        self.run_process = QProcess(self)
        self.run_process.setProgram(cmd[0])
        self.run_process.setArguments(cmd[1:])
        self.run_process.setWorkingDirectory(str(self.runtime.repo_root))
        self.run_process.readyReadStandardOutput.connect(self._on_proc_stdout)
        self.run_process.readyReadStandardError.connect(self._on_proc_stderr)
        self.run_process.finished.connect(self._on_proc_finished)
        self.run_process.started.connect(lambda: self._on_proc_started(cmd))
        self.run_process.start()

    def _on_proc_started(self, cmd: List[str]) -> None:
        self.run_status = UiStatus.RUNNING
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._append_log(self.runtime.command_preview(cmd))
        self._refresh_all_panels()

    def _on_proc_stdout(self) -> None:
        if not self.run_process:
            return
        text = self._decode_bytes(bytes(self.run_process.readAllStandardOutput()))
        if text:
            self._append_log(text.rstrip())

    def _on_proc_stderr(self) -> None:
        if not self.run_process:
            return
        text = self._decode_bytes(bytes(self.run_process.readAllStandardError()))
        if text:
            self._append_log(text.rstrip())

    def _on_proc_finished(self, code: int, _status: QProcess.ExitStatus) -> None:
        self.run_status = UiStatus.SUCCESS if code == 0 else UiStatus.FAILED
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._append_log(f"任务结束，返回码 {code}")
        self._refresh_all_panels()

    def _pause_alignment(self) -> None:
        # 目前 SpotZoom 主流程没有原生pause 命令，v1 用"温和停止"替代。
        self._stop_alignment(tag="pause")

    def _stop_alignment(self, tag: str = "stop") -> None:
        if not self.run_process or self.run_process.state() == QProcess.NotRunning:
            return
        self._append_log(f"请求{tag}运行流程。")
        self.run_process.terminate()
        if not self.run_process.waitForFinished(2000):
            self.run_process.kill()
        self.run_status = UiStatus.WARNING
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._refresh_all_panels()

    # ==================================================================
    # 准直工作台 — UCC 预览 + 闭环控制
    # ==================================================================

    def _start_alignment_ucc_preview(self) -> None:
        if self._alignment_ucc_running:
            self._append_log("[准直工作台] UCC 预览已在运行中")
            return
        try:
            device = self._alignment_ucc_device.value()
            resolution = self._alignment_ucc_resolution.currentText()
            fmt = self._alignment_ucc_format.currentText()

            self._alignment_ucc_source = UCCFrameSource(
                device_index=device,
                resolution=resolution or "AUTO",
                pixel_format=fmt if fmt != "AUTO" else None,
            )
            self._append_log(f"[准直工作台] UCC 预览已启动: 设备={device} 分辨率={resolution} 格式={fmt}")
        except Exception as e:
            self._append_log(f"[准直工作台] UCC 预览启动失败: {e}")
            QMessageBox.warning(self, "UCC 错误", f"无法打开 UCC 相机:\n{e}")
            self._alignment_ucc_source = None
            return

        self._alignment_ucc_timer = QTimer(self)
        self._alignment_ucc_timer.setInterval(50)
        self._alignment_ucc_timer.timeout.connect(self._update_alignment_ucc_preview)
        self._alignment_ucc_timer.start()

        self._alignment_ucc_running = True
        self._alignment_ucc_btn_start.setEnabled(False)
        self._alignment_ucc_btn_stop.setEnabled(True)
        self._alignment_ucc_path_info.setText(f"调试帧目录: Tmp_Frames / FrameTmp | 预览状态: 运行中 | 设备={device} | 分辨率={resolution}")
        self.btn_set_target.setEnabled(True)
        self.btn_jitter.setEnabled(True)
        self.btn_stabilize.setEnabled(True)

    def _stop_alignment_ucc_preview(self) -> None:
        self._stop_alignment_stabilization()
        self._stop_alignment_jitter()

        self._alignment_ucc_running = False

        if self._alignment_ucc_timer is not None:
            self._alignment_ucc_timer.stop()
            self._alignment_ucc_timer = None

        if self._alignment_ucc_source is not None:
            try:
                self._alignment_ucc_source.release()
            except Exception as e:
                self._append_log(f"[准直工作台] UCC 释放异常: {e}")
            self._alignment_ucc_source = None

        self._alignment_ucc_label.setText("UCC 探测器\n已停止预览")
        self._alignment_ucc_btn_start.setEnabled(True)
        self._alignment_ucc_btn_stop.setEnabled(False)
        self.btn_set_target.setEnabled(False)
        self.btn_jitter.setEnabled(False)
        self.btn_stabilize.setEnabled(False)
        self._append_log("[准直工作台] UCC 预览已停止")

    def _update_alignment_ucc_preview(self) -> None:
        source = self._alignment_ucc_source
        if source is None or not self._alignment_ucc_running:
            return

        try:
            bgr = source.grab_frame()
            if bgr is None:
                return
        except Exception as e:
            self._append_log(f"[准直工作台] UCC 帧读取失败: {e}")
            return

        self._alignment_ucc_fps_counter += 1
        now = time.time()
        if self._alignment_ucc_fps_time == 0.0:
            self._alignment_ucc_fps_time = now
        elapsed = now - self._alignment_ucc_fps_time
        if elapsed >= 1.0:
            self._alignment_ucc_actual_fps = self._alignment_ucc_fps_counter / elapsed
            self._alignment_ucc_fps_counter = 0
            self._alignment_ucc_fps_time = now

        frame = bgr.copy()
        height, width = frame.shape[:2]

        # 光斑分析
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)
        _, thresh = cv2.threshold(gray, int(max_val * 0.5), 255, cv2.THRESH_BINARY)
        moments = cv2.moments(thresh)
        if moments["m00"] > 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            self._alignment_last_centroid = (cx, cy)
            self._alignment_last_centroid_ts = time.time()
            self._alignment_centroid_history.append((cx, cy))
            if len(self._alignment_centroid_history) > 200:
                self._alignment_centroid_history = self._alignment_centroid_history[-200:]

            # 叠加标注
            cv2.drawMarker(frame, (int(cx), int(cy)), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.circle(frame, (int(cx), int(cy)), 8, (0, 255, 255), 1)

            if self._alignment_target_point is not None:
                tx, ty = self._alignment_target_point
                cv2.drawMarker(frame, (int(tx), int(ty)), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                cv2.circle(frame, (int(tx), int(ty)), 10, (0, 255, 0), 2)
                cv2.line(frame, (int(cx), int(cy)), (int(tx), int(ty)), (255, 0, 255), 1)

            # 质心宽高
            if moments["mu20"] + moments["mu02"] > 0:
                sigma_x = (moments["mu20"] / moments["m00"]) ** 0.5
                sigma_y = (moments["mu02"] / moments["m00"]) ** 0.5
            else:
                sigma_x = sigma_y = 0.0

            self._alignment_param_table.item(0, 1).setText(f"{min_val:.1f}, {max_val:.1f}")
            self._alignment_param_table.item(1, 1).setText(f"({max_loc[0]}, {max_loc[1]})")
            self._alignment_param_table.item(2, 1).setText(f"({cx:.1f}, {cy:.1f})")
            self._alignment_param_table.item(3, 1).setText(f"({sigma_x:.1f}, {sigma_y:.1f})")

            # 曲线
            if self._alignment_centroid_history:
                xs = [p[0] for p in self._alignment_centroid_history]
                ys = [p[1] for p in self._alignment_centroid_history]
                self._alignment_curve_widget.append(xs[-1], ys[-1])

            # 偏差信息
            if self._alignment_target_point is not None:
                tx, ty = self._alignment_target_point
                dx = cx - tx
                dy = cy - ty
                dist = (dx * dx + dy * dy) ** 0.5
                status = f"稳定闭环中" if self._alignment_stabilizing else "目标已锁定"
                status += f" | 偏差 Δ=({dx:.1f}, {dy:.1f})px d={dist:.1f}px"
            else:
                status = "未设定目标点"
            self.image_overlay_label.setText(
                f"质心: ({cx:.1f}, {cy:.1f}) | 目标点: {self._alignment_target_point or '未设定'}\n"
                f"FPS: {self._alignment_ucc_actual_fps:.1f} | 状态: {status}"
            )
        else:
            self._alignment_param_table.item(2, 1).setText("--")
            self.image_overlay_label.setText("检测中... 未检测到光斑")

        # 显示
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self._alignment_ucc_label.setPixmap(
            pix.scaled(self._alignment_ucc_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def _enter_alignment_4axis_mode(self) -> None:
        self._set_combo("alignment_strategy", AlignmentStrategy.DUAL_DETECTOR_4AXIS.value)
        self._set_combo("detector_mode", "dual_detector")
        self.profile = self._collect_profile_from_controls()
        self.axis4_status_label.setText(
            f"状态：4轴闭环已激活 | 策略={self.profile.alignment_strategy.value} | 探测器模式={self.profile.detector_mode}"
        )
        self.image_overlay_label.setText("目标点: -- | 当前点: -- | 偏差: -- | 状态: 已进入4轴闭环")
        self.btn_set_target.setEnabled(True)
        self.btn_calibrate.setEnabled(True)
        self.btn_jitter.setEnabled(True)
        self.btn_stabilize.setEnabled(True)
        self.btn_axis4_enter.setEnabled(False)
        self._append_log(
            f"[准直工作台] 已切换到4轴闭环入口: strategy={self.profile.alignment_strategy.value}, detector_mode={self.profile.detector_mode}; 4轴映射={self._get_alignment_mirror_axes()}"
        )

    def _set_alignment_target_point(self) -> None:
        if self._alignment_last_centroid is None:
            QMessageBox.warning(self, "无光斑", "请先确保 UCC 预览中检测到光斑质心")
            return
        self._alignment_target_point = self._alignment_last_centroid
        cx, cy = self._alignment_target_point
        self._alignment_centroid_history.clear()
        self._append_log(f"[准直工作台] 已设定目标点: ({cx:.1f}, {cy:.1f})")
        self._refresh_alignment_status_label("目标点已设定，等待闭环启动")
        self.btn_stabilize.setEnabled(True)

    def _get_alignment_mirror_axes(self):
        self.profile = self._collect_profile_from_controls()
        return {
            "mirror1_x": self.profile.mrc_mirror1_x_axis,
            "mirror1_y": self.profile.mrc_mirror1_y_axis,
            "mirror2_x": self.profile.mrc_mirror2_x_axis,
            "mirror2_y": self.profile.mrc_mirror2_y_axis,
        }

    def _get_alignment_axis_widget(self, axis_num: int):
        panel = getattr(self, "picomotor_driver_panel", None)
        if panel is None:
            return None
        for aw in panel.axis_widgets:
            if aw.axis == axis_num:
                return aw
        return None

    def _refresh_alignment_status_label(self, status_text: str = "-") -> None:
        cx, cy = self._alignment_last_centroid if self._alignment_last_centroid is not None else (None, None)
        tx, ty = self._alignment_target_point if self._alignment_target_point is not None else (None, None)
        if cx is not None and tx is not None:
            dx = cx - tx
            dy = cy - ty
            status = f"目标点: ({tx:.1f}, {ty:.1f}) | 当前点: ({cx:.1f}, {cy:.1f})\n偏差: ({dx:.1f}, {dy:.1f}) | 状态: {status_text}"
        else:
            status = f"目标点: -- | 当前点: --\n偏差: -- | 状态: {status_text}"
        self.image_overlay_label.setText(status)
        if hasattr(self, "axis4_status_label"):
            mode = self.profile.alignment_strategy.value if hasattr(self, "profile") else "-"
            det = self.profile.detector_mode if hasattr(self, "profile") else "-"
            self.axis4_status_label.setText(f"状态：{status_text} | 策略={mode} | 探测器模式={det}")

    def _apply_alignment_delta_to_all_axes(self, dx: int, dy: int, label: str) -> None:
        axes = self._get_alignment_mirror_axes()
        # 根据用户选择的校正镜组决定参与运动的轴
        #   mirror2 — 仅 Mirror2 参与校正（默认）
        #   mirror1 — 仅 Mirror1 参与校正
        #   both    — 两个镜子同时参与
        correction = self._alignment_correction_mirror.currentText()
        assignments = [
            (axes["mirror1_x"], int(dx) * int(self.profile.mrc_mirror1_x_sign), "mirror1_x"),
            (axes["mirror1_y"], int(dy) * int(self.profile.mrc_mirror1_y_sign), "mirror1_y"),
            (axes["mirror2_x"], int(dx) * int(self.profile.mrc_mirror2_x_sign), "mirror2_x"),
            (axes["mirror2_y"], int(dy) * int(self.profile.mrc_mirror2_y_sign), "mirror2_y"),
        ]
        moved = []
        for axis_num, steps, name in assignments:
            widget = self._get_alignment_axis_widget(axis_num)
            if widget is None:
                continue
            if steps == 0:
                continue
            # 仅当该镜组被选中时才移动
            if name.startswith("mirror1_") and correction not in ("mirror1", "both"):
                continue
            if name.startswith("mirror2_") and correction not in ("mirror2", "both"):
                continue
            widget.move_relative(steps)
            moved.append(f"{name}={steps}")
        if moved:
            self._append_log(f"{label}: " + ", ".join(moved))
        else:
            self._append_log(f"{label}: 未找到可用轴")

        # 虚拟模式下同步偏移质心，使标定/抖动/闭环均有真实偏差可测
        if self._is_virtual_mode() and self._alignment_last_centroid is not None:
            scale = 0.05
            cx, cy = self._alignment_last_centroid
            self._alignment_last_centroid = (cx + dx * scale, cy + dy * scale)

    def _is_virtual_mode(self) -> bool:
        """检测当前是否为虚拟控制器模式"""
        panel = getattr(self, "picomotor_driver_panel", None)
        if panel is not None and getattr(panel, "status_label", None) is not None:
            return panel.status_label.text() == "虚拟模式"
        return False

    def _calibrate_axis_mapping(self) -> None:
        """基于探测器坐标与4轴移动，标定方向与步长/像素比"""
        if self._alignment_calib_active:
            return
        if self._alignment_last_centroid is None:
            QMessageBox.warning(self, "无光斑", "请先确保 UCC 预览中检测到光斑质心")
            return
        if not self._alignment_axes_ready():
            return

        self.btn_calibrate.setEnabled(False)
        self.btn_stop_calibrate.setEnabled(True)
        self.btn_calibrate.setText("标定中...")
        self._alignment_calib_active = True

        calib_step = int(self._alignment_calib_step.value())
        correction = self._alignment_correction_mirror.currentText()
        self._append_log(f"[标定] 校正镜组={correction}，本次标定将移动 {correction} 轴并观察质心变化")
        self._calib_origin = self._alignment_last_centroid
        self._calib_phase = 0
        self._calib_step_size = calib_step
        self._calib_dx_px = 0.0
        self._calib_dy_px = 0.0

        self._append_log(f"[标定] 开始4轴→探测器映射标定，步长={calib_step}")
        self._append_log(f"[标定] 起始质心: ({self._calib_origin[0]:.1f}, {self._calib_origin[1]:.1f})")

        self._calib_timer = QTimer(self)
        self._calib_timer.setInterval(int(self._alignment_calib_interval.value()))
        self._calib_timer.timeout.connect(self._calibrate_tick)
        self._calib_timer.start()

    def _calibrate_cleanup(self) -> None:
        """清理标定状态"""
        self._alignment_calib_active = False
        timer = getattr(self, "_calib_timer", None)
        if timer is not None:
            timer.stop()
            self._calib_timer = None  # type: ignore[assignment]
        self.btn_calibrate.setEnabled(True)
        self.btn_stop_calibrate.setEnabled(False)
        self.btn_calibrate.setText("📏 标定映射")

    def _stop_alignment_calibration(self) -> None:
        if not self._alignment_calib_active:
            return
        self._append_log("[标定] 用户手动停止标定")
        self._calibrate_cleanup()

    def _update_alignment_calib_info_label(self) -> None:
        self._alignment_calib_info_label.setText(
            f"当前标定参数\n"
            f"步长: {self._alignment_calib_step.value()} 步\n"
            f"间隔: {self._alignment_calib_interval.value()} ms"
        )

    def _calibrate_tick(self) -> None:
        """标定状态机——每个 tick 执行一步"""
        try:
            self._calibrate_tick_impl()
        except Exception as e:
            self._append_log(f"[标定] 错误: {e}")
            import traceback
            self._append_log(traceback.format_exc())
            self._calibrate_cleanup()

    def _calibrate_tick_impl(self) -> None:
        centroid = self._alignment_last_centroid
        phase = self._calib_phase
        step = self._calib_step_size

        if phase == 0:
            # Phase 0：记录原点 → X轴正向移动 step 步
            self._apply_alignment_delta_to_all_axes(step, 0, f"[标定] X轴正向移动 step={step}")
            self._calib_phase = 1

        elif phase == 1:
            # Phase 1：读取新质心 → 计算探测器Y变化（mirror2_X实际影响探测器Y）→ X轴归位
            if centroid is None:
                return
            dx = centroid[1] - self._calib_origin[1]
            if abs(dx) < 0.5:
                self._append_log(f"[标定] X轴(mirror2_X)移动后探测器Y未明显变化(ΔY={dx:.1f})，等待下一帧...")
                return
            self._calib_dx_px = dx
            self._append_log(f"[标定] X轴移动后: 质心=({centroid[0]:.1f}, {centroid[1]:.1f}), 探测器ΔY={dx:.2f}px")
            self._apply_alignment_delta_to_all_axes(-step, 0, "[标定] X轴归位")
            self._calib_phase = 2

        elif phase == 2:
            # Phase 2：等待归位稳定 → Y轴正向移动
            if centroid is None:
                return
            self._apply_alignment_delta_to_all_axes(0, step, f"[标定] Y轴正向移动 step={step}")
            self._calib_phase = 3

        elif phase == 3:
            # Phase 3：读取新质心 → 计算探测器X变化（mirror2_Y实际影响探测器X）→ Y轴归位
            if centroid is None:
                return
            dy = centroid[0] - self._calib_origin[0]
            if abs(dy) < 0.5:
                self._append_log(f"[标定] Y轴(mirror2_Y)移动后探测器X未明显变化(ΔX={dy:.1f})，等待下一帧...")
                return
            self._calib_dy_px = dy
            self._append_log(f"[标定] Y轴移动后: 质心=({centroid[0]:.1f}, {centroid[1]:.1f}), 探测器ΔX={dy:.2f}px")
            self._apply_alignment_delta_to_all_axes(0, -step, "[标定] Y轴归位")
            self._calib_phase = 4

        elif phase == 4:
            # Phase 4：计算结果 → 写入标定参数
            if abs(self._calib_dx_px) < 0.1 or abs(self._calib_dy_px) < 0.1:
                self._append_log("[标定] 质心变化太小，标定失败")
                self._calibrate_cleanup()
                return

            self._alignment_calib_direction_x = -1 if self._calib_dx_px < 0 else 1
            self._alignment_calib_direction_y = -1 if self._calib_dy_px < 0 else 1
            self._alignment_calib_steps_per_px_x = step / abs(self._calib_dx_px)
            self._alignment_calib_steps_per_px_y = step / abs(self._calib_dy_px)
            self._alignment_calibrated = True

            self._append_log("[标定] ✅ 标定完成 (轴交叉映射):")
            self._append_log(
                f"[标定]    mirror2_X → 探测器Y: {step}步 → {self._calib_dx_px:.2f}px, "
                f"即 {self._alignment_calib_steps_per_px_x:.1f} 步/像素"
            )
            self._append_log(
                f"[标定]    mirror2_Y → 探测器X: {step}步 → {self._calib_dy_px:.2f}px, "
                f"即 {self._alignment_calib_steps_per_px_y:.1f} 步/像素"
            )
            self.axis4_calib_label.setText(
                f"标定: X→Y {self._alignment_calib_steps_per_px_x:.1f}步/px | "
                f"Y→X {self._alignment_calib_steps_per_px_y:.1f}步/px"
            )
            self.axis4_calib_label.setStyleSheet("color: #22C55E; font-size: 11px;")
            self._calibrate_cleanup()

    def _simulate_alignment_jitter(self) -> None:
        if self._alignment_jitter_running:
            self._stop_alignment_jitter()
            self.btn_jitter.setText("🌀 模拟抖动")
            self._append_log("[准直工作台] 模拟抖动已停止")
            return

        panel = getattr(self, "picomotor_driver_panel", None)
        if panel is None:
            QMessageBox.warning(self, "未连接", "请先在「Picomotor 8742/8743 驱动调试」页面连接设备")
            return
        if not self._alignment_axes_ready():
            return

        self._alignment_jitter_running = True
        self.btn_jitter.setText("⏹ 停止抖动")

        amplitude = self._alignment_jitter_amplitude.value()
        interval_ms = int(self._alignment_jitter_interval.value() * 1000)

        def _do_jitter() -> None:
            if not self._alignment_jitter_running:
                return
            try:
                amplitude = self._alignment_jitter_amplitude.value()
                dx = random.randint(-amplitude, amplitude)
                dy = random.randint(-amplitude, amplitude)
                self._apply_alignment_delta_to_all_axes(dx, dy, f"[抖动] ΔX={dx}, ΔY={dy}")
            except Exception as e:
                self._append_log(f"[抖动] 电机控制异常: {e}")

        self._alignment_jitter_timer = QTimer(self)
        self._alignment_jitter_timer.setInterval(interval_ms)
        self._alignment_jitter_timer.timeout.connect(_do_jitter)
        self._alignment_jitter_timer.start()
        self._append_log(f"[准直工作台] 模拟抖动已启动: 幅值={amplitude}步 间隔={self._alignment_jitter_interval.value()}s")

    def _stop_alignment_jitter(self) -> None:
        self._alignment_jitter_running = False
        if self._alignment_jitter_timer is not None:
            self._alignment_jitter_timer.stop()
            self._alignment_jitter_timer = None
        self.btn_jitter.setText("🌀 模拟抖动")

    def _select_convergence_record_export_path(self) -> None:
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "选择收敛误差记录导出路径",
            self._alignment_record_export_path.text().strip(),
            "Excel Files (*.xlsx)",
        )
        if out_path:
            if not out_path.lower().endswith(".xlsx"):
                out_path += ".xlsx"
            self._alignment_record_export_path.setText(out_path)

    def _update_convergence_record_status_label(self) -> None:
        timer = self._alignment_convergence_record_timer
        if timer is None:
            if self._alignment_convergence_record_rows:
                exported_path = self._alignment_record_exported_path or self._alignment_record_export_path.text().strip()
                self._alignment_record_status_label.setText(
                    f"记录状态：已结束 / 共记录 {len(self._alignment_convergence_record_rows)} 条 / 已导出路径 {exported_path}"
                )
            else:
                self._alignment_record_status_label.setText("记录状态：未开始")
            return
        remain_minutes = max(0.0, (self._alignment_convergence_record_end_ts - time.time()) / 60.0)
        exported_path = self._alignment_record_exported_path or self._alignment_record_export_path.text().strip()
        self._alignment_record_status_label.setText(
            f"记录中：剩余 {remain_minutes:.1f} min / 已记录 {len(self._alignment_convergence_record_rows)} 条 / 已导出路径 {exported_path}"
        )

    def _write_simple_xlsx(self, out_path: Path, rows: List[List[object]]) -> None:
        from xml.sax.saxutils import escape as xml_escape

        shared_strings: List[str] = []
        shared_index: Dict[str, int] = {}

        def _ss_idx(text: str) -> int:
            if text not in shared_index:
                shared_index[text] = len(shared_strings)
                shared_strings.append(text)
            return shared_index[text]

        sheet_rows: List[str] = []
        for row_idx, row in enumerate(rows, start=1):
            cells: List[str] = []
            for col_idx, value in enumerate(row, start=1):
                col_name = ""
                x = col_idx
                while x > 0:
                    x, rem = divmod(x - 1, 26)
                    col_name = chr(65 + rem) + col_name
                cell_ref = f"{col_name}{row_idx}"
                text = "" if value is None else str(value)
                ss_id = _ss_idx(text)
                cells.append(f'<c r="{cell_ref}" t="s"><v>{ss_id}</v></c>')
            sheet_rows.append(f'<row r="{row_idx}">' + "".join(cells) + "</row>")

        shared_xml = "".join(f"<si><t>{xml_escape(s)}</t></si>" for s in shared_strings)
        worksheet_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<sheetData>' + "".join(sheet_rows) + '</sheetData></worksheet>'
        )
        workbook_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="convergence_record" sheetId="1" r:id="rId1"/></sheets></workbook>'
        )
        workbook_rels_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/>'
            '</Relationships>'
        )
        rels_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            '</Relationships>'
        )
        content_types_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            '<Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>'
            '</Types>'
        )
        shared_strings_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="{len(shared_strings)}" uniqueCount="{len(shared_strings)}">{shared_xml}</sst>'
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", content_types_xml)
            zf.writestr("_rels/.rels", rels_xml)
            zf.writestr("xl/workbook.xml", workbook_xml)
            zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
            zf.writestr("xl/worksheets/sheet1.xml", worksheet_xml)
            zf.writestr("xl/sharedStrings.xml", shared_strings_xml)

    def _export_convergence_record_xlsx(self) -> None:
        out_path = Path(self._alignment_record_export_path.text().strip())
        rows = [[
            "记录时间",
            "收敛坐标质心X",
            "收敛坐标质心Y",
            "目标点X",
            "目标点Y",
            "当前质心X",
            "当前质心Y",
            "X坐标误差(px)",
            "Y坐标误差(px)",
            "总体误差(px)",
        ]] + self._alignment_convergence_record_rows
        self._write_simple_xlsx(out_path, rows)
        self._append_log(f"[收敛误差记录] 已导出: {out_path}")

    def _record_convergence_error_sample(self) -> None:
        if self._alignment_last_centroid is None or self._alignment_target_point is None or self._alignment_converged_centroid is None:
            return
        cx, cy = self._alignment_last_centroid
        tx, ty = self._alignment_target_point
        ccx, ccy = self._alignment_converged_centroid
        dx = cx - tx
        dy = cy - ty
        dist = (dx * dx + dy * dy) ** 0.5
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self._alignment_convergence_record_rows.append([
            stamp, f"{ccx:.3f}", f"{ccy:.3f}", f"{tx:.3f}", f"{ty:.3f}", f"{cx:.3f}", f"{cy:.3f}", f"{dx:.3f}", f"{dy:.3f}", f"{dist:.3f}"
        ])
        self._update_convergence_record_status_label()
        remain = self._alignment_convergence_record_end_ts - time.time()
        total_min = (self._alignment_convergence_record_end_ts - self._alignment_convergence_record_start_ts) / 60.0
        elapsed_min = (time.time() - self._alignment_convergence_record_start_ts) / 60.0
        self._append_log(
            f"[收敛误差记录] 采样 #{len(self._alignment_convergence_record_rows)}: "
            f"Δ=({dx:.2f}, {dy:.2f}) dist={dist:.3f}px "
            f"丨已用 {elapsed_min:.1f}/{total_min:.1f} min"
        )
        # 更新分布可视化
        self._alignment_error_dist_widget.append_dist(dist)
        if time.time() >= self._alignment_convergence_record_end_ts:
            self._stop_convergence_error_recording(auto_export=True)

    def _start_convergence_error_recording(self) -> None:
        if self._alignment_target_point is None or self._alignment_last_centroid is None:
            QMessageBox.information(self, "未准备好", "请先确保已经检测到光斑并设定目标点")
            return
        out_path = self._alignment_record_export_path.text().strip()
        if not out_path:
            QMessageBox.warning(self, "路径为空", "请先设置 .xlsx 导出路径")
            return
        if not out_path.lower().endswith(".xlsx"):
            QMessageBox.warning(self, "格式错误", "导出路径必须是 .xlsx 文件")
            return
        duration_minutes = self._alignment_record_duration_min.value()
        self._alignment_convergence_record_rows = []
        self._alignment_error_dist_widget.clear_history()
        self._alignment_convergence_record_start_ts = time.time()
        self._alignment_convergence_record_end_ts = self._alignment_convergence_record_start_ts + duration_minutes * 60.0
        self._alignment_converged_centroid = self._alignment_last_centroid
        if self._alignment_convergence_record_timer is not None:
            self._alignment_convergence_record_timer.stop()
        self._alignment_convergence_record_timer = QTimer(self)
        self._alignment_convergence_record_timer.setInterval(60_000)
        self._alignment_convergence_record_timer.timeout.connect(self._record_convergence_error_sample)
        self._alignment_convergence_record_timer.start()
        self._record_convergence_error_sample()
        self.btn_record_convergence.setEnabled(False)
        self.btn_stop_record_convergence.setEnabled(True)
        self._alignment_record_status_label.setText("记录状态：记录中")
        self._alignment_record_exported_path = out_path
        self._update_convergence_record_status_label()
        self._append_log(f"[收敛误差记录] 已启动: 时长={duration_minutes:.1f}min, 间隔=1min, 导出={out_path}")

    def _stop_convergence_error_recording(self, auto_export: bool = False) -> None:
        if self._alignment_convergence_record_timer is not None:
            self._alignment_convergence_record_timer.stop()
            self._alignment_convergence_record_timer = None
        if auto_export and self._alignment_convergence_record_rows:
            self._export_convergence_record_xlsx()
        self.btn_record_convergence.setEnabled(True)
        self.btn_stop_record_convergence.setEnabled(False)
        if self._alignment_convergence_record_rows:
            self._alignment_record_status_label.setText(
                f"记录状态：已结束 / 共记录 {len(self._alignment_convergence_record_rows)} 条"
            )
        else:
            self._alignment_record_status_label.setText("记录状态：未开始")

    def _reset_alignment_precision_stats(self) -> None:
        self._alignment_precision_samples.clear()
        fields = getattr(self, "_alignment_precision_fields", {})
        for label in fields.values():
            label.setText("--")

    def _record_alignment_precision_sample(self, dx: float, dy: float, dist: float) -> None:
        self._alignment_precision_samples.append((dx, dy, dist))
        samples = self._alignment_precision_samples
        n = len(samples)
        xs = [item[0] for item in samples]
        ys = [item[1] for item in samples]
        rs = sorted(item[2] for item in samples)
        rms = (sum(r * r for r in rs) / n) ** 0.5
        p95_index = min(n - 1, int(0.95 * (n - 1)))
        p95 = rs[p95_index]
        max_r = rs[-1]
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        x_std = (sum((x - mean_x) ** 2 for x in xs) / n) ** 0.5
        y_std = (sum((y - mean_y) ** 2 for y in ys) / n) ** 0.5
        pixel_um = self._alignment_pixel_size_um.value()
        detector_distance_mm = self._alignment_detector_distance_mm.value()
        pointing_p95_urad = p95 * pixel_um / detector_distance_mm * 1000.0
        linear_ok = p95 < self._alignment_stabilize_tolerance.value() * 2
        range_ok = max_r < self._alignment_stabilize_tolerance.value() * 4
        fields = self._alignment_precision_fields
        fields["samples"].setText(str(n))
        fields["rms"].setText(f"{rms:.2f}")
        fields["p95"].setText(f"{p95:.2f}")
        fields["max"].setText(f"{max_r:.2f}")
        fields["x_std"].setText(f"{x_std:.2f}")
        fields["y_std"].setText(f"{y_std:.2f}")
        fields["rms_um"].setText(f"{rms * pixel_um:.2f}")
        fields["p95_um"].setText(f"{p95 * pixel_um:.2f}")
        fields["max_um"].setText(f"{max_r * pixel_um:.2f}")
        fields["pointing_p95"].setText(f"{pointing_p95_urad:.2f}")
        fields["linear"].setText("OK" if linear_ok else "超出")
        fields["range"].setText("OK" if range_ok else "超限")

    def _start_alignment_stabilization(self) -> None:
        if self._alignment_stabilizing:
            return
        # 启动闭环前自动停止抖动，避免两者对抗
        if self._alignment_jitter_running:
            self._stop_alignment_jitter()
            self._append_log("[准直工作台] 闭环启动，已自动停止抖动")
        if self._alignment_target_point is None:
            QMessageBox.warning(self, "无目标点", "请先点击「设定目标点」锁定当前光斑位置")
            return

        panel = getattr(self, "picomotor_driver_panel", None)
        if panel is None:
            QMessageBox.warning(self, "未连接", "请先在「Picomotor 8742/8743 驱动调试」页面连接设备")
            return
        if not self._alignment_axes_ready():
            return

        self._alignment_stabilizing = True
        self._alignment_convergence_record_unlocked = False
        self._alignment_converged_centroid = None
        self.btn_record_convergence.setEnabled(False)
        self._alignment_record_status_label.setText("记录状态：未开始")
        self._stop_convergence_error_recording(auto_export=False)
        self._reset_alignment_precision_stats()
        self.btn_stabilize.setEnabled(False)
        self.btn_stop_stabilize.setEnabled(True)
        self._append_log("[准直工作台] 稳定闭环已启动")

        self._alignment_stabilize_timer = QTimer(self)
        self._alignment_stabilize_timer.setInterval(int(self._alignment_stabilize_interval.value()))
        self._alignment_stabilize_timer.timeout.connect(self._alignment_stabilize_step)
        self._alignment_stabilize_timer.start()
        self._append_log(f"[准直工作台] 稳定闭环已启动: 间隔={int(self._alignment_stabilize_interval.value())}ms")

    def _stop_alignment_stabilization(self) -> None:
        self._alignment_stabilizing = False
        if self._alignment_stabilize_timer is not None:
            self._alignment_stabilize_timer.stop()
            self._alignment_stabilize_timer = None
        self.btn_stabilize.setEnabled(True)
        self.btn_stop_stabilize.setEnabled(False)
        self._append_log("[准直工作台] 稳定闭环已停止")

    def _alignment_axes_ready(self) -> bool:
        panel = getattr(self, "picomotor_driver_panel", None)
        if panel is None:
            QMessageBox.warning(self, "面板未就绪", "请先在「Picomotor 8742/8743 驱动调试」页面初始化")
            return False
        # 如果面板没有轴控件，先尝试自动连接真实控制器
        if not panel.axis_widgets:
            self.profile = self._collect_profile_from_controls()
            auto_connected = panel.try_auto_connect(conn=self.profile.newport_conn)
            if auto_connected:
                self._append_log(f"[准直工作台] 已自动连接控制器: axes={[aw.axis for aw in panel.axis_widgets]}")
            else:
                # 自动连接失败，创建虚拟轴用于测试
                self._append_log("[准直工作台] 未检测到真实控制器，自动创建虚拟轴...")
                panel.ensure_virtual_axes()
        axes = self._get_alignment_mirror_axes()
        missing = [name for name, axis in axes.items() if self._get_alignment_axis_widget(axis) is None]
        if missing:
            QMessageBox.warning(self, "轴未就绪", f"以下轴未连接或未初始化：{', '.join(missing)}")
            self._append_log(f"[准直工作台] 4轴未就绪: {', '.join(missing)}")
            return False
        return True

    def _alignment_stabilize_step(self) -> None:
        if not self._alignment_stabilizing:
            return
        if self._alignment_last_centroid is None or self._alignment_target_point is None:
            self._refresh_alignment_status_label("等待探测到目标点")
            return

        # 光斑丢失保护：如果最后一次检测到光斑超过超时阈值，自动停止闭环
        timeout_s = self._alignment_spot_loss_timeout.value()
        if self._alignment_last_centroid_ts > 0 and time.time() - self._alignment_last_centroid_ts > timeout_s:
            self._append_log(
                f"[闭环] ⚠️ 光斑已丢失 {time.time() - self._alignment_last_centroid_ts:.0f}s"
                f"（超时 {timeout_s}s），自动停止闭环保护电机"
            )
            self._stop_alignment_stabilization()
            return

        cx, cy = self._alignment_last_centroid
        tx, ty = self._alignment_target_point
        dx = cx - tx
        dy = cy - ty
        dist = (dx * dx + dy * dy) ** 0.5
        self._record_alignment_precision_sample(dx, dy, dist)

        tolerance = self._alignment_stabilize_tolerance.value()
        if dist < tolerance:
            self._alignment_converged_centroid = (cx, cy)
            # 仅在未进行记录时才启用记录按钮，避免记录过程中被重新启用导致误操作
            if self._alignment_convergence_record_timer is None:
                self.btn_record_convergence.setEnabled(True)
                self._alignment_record_status_label.setText("记录状态：已收敛，可直接记录")
            self._refresh_alignment_status_label("已收敛")
            self._append_log(f"[闭环] 已收敛: 偏差=({dx:.1f}, {dy:.1f}) 容差={tolerance}")
            return

        kp = self._alignment_stabilize_kp.value()
        if self._alignment_calibrated:
            # 轴交叉修正: mirror2_X ↔ 探测器Y, mirror2_Y ↔ 探测器X
            steps_x = int(dy * self._alignment_calib_steps_per_px_x * kp)  # 探测器Y偏差驱动mirror2_X
            steps_y = int(dx * self._alignment_calib_steps_per_px_y * kp)  # 探测器X偏差驱动mirror2_Y
        else:
            # 未标定，也按轴交叉处理
            steps_x = int(dy * kp * self._alignment_calib_direction_x)
            steps_y = int(dx * kp * self._alignment_calib_direction_y)

        if abs(steps_x) < 1 and abs(steps_y) < 1:
            self._append_log(f"[闭环] 步长不足: 偏差=({dx:.1f}, {dy:.1f}) steps=({steps_x}, {steps_y}) 增益={kp}")
            return

        self._refresh_alignment_status_label(f"纠偏中 Δ=({dx:.1f}, {dy:.1f})")

        try:
            calib_tag = "标定" if self._alignment_calibrated else "无标定"
            self._apply_alignment_delta_to_all_axes(steps_x, steps_y, f"[闭环] [{calib_tag}] 纠偏 Δ=({dx:.1f}, {dy:.1f})→steps=({steps_x}, {steps_y})")
        except Exception as e:
            self._append_log(f"[闭环] 电机控制异常: {e}")

    # ==================================================================

    def _toggle_roi(self) -> None:
        current = self._check_value("select_roi", False)
        self._set_check("select_roi", not current)
        self._append_log(f"ROI 开关已切换: {'启用' if not current else '禁用'}")
        self._refresh_all_panels()

    def _run_selected_test(self) -> None:
        item = self.test_tree.currentItem()
        if item is None:
            return
        spec = item.data(0, Qt.UserRole)
        if isinstance(spec, TestCaseSpec):
            self._run_device_test_by_id(spec.test_id)
        elif isinstance(spec, str):
            self._run_device_test_by_id(spec)

    def _run_device_test_by_id(self, test_id: str) -> None:
        self.profile = self._collect_profile_from_controls()
        result = self.device_test.run_test(test_id, self.profile)
        row = self.test_result_table.rowCount()
        self.test_result_table.insertRow(row)
        self.test_result_table.setItem(row, 0, QTableWidgetItem(result.title))
        self.test_result_table.setItem(row, 1, QTableWidgetItem(status_text(result.status)))
        self.test_result_table.setItem(row, 2, QTableWidgetItem(str(result.duration_ms)))
        self.test_result_table.setItem(row, 3, QTableWidgetItem(result.message))
        self.test_result_table.setItem(row, 4, QTableWidgetItem(result.raw_return))
        self.test_raw_output.setPlainText(result.raw_return)
        self._append_log(f"[DeviceTest] {result.title}: {result.status.value} / {result.message}")

    def _start_continuous_monitoring(self) -> None:
        """启动持续监控模式。"""
        profile = self._collect_profile_from_controls()
        controller = getattr(self, '_controller', None)
        if controller is None:
            self._append_log("错误：控制器未初始化，请先运行对准")
            return
        
        if not hasattr(controller, 'run_continuous_monitoring'):
            self._append_log("错误：当前控制器不支持持续监控模式")
            return
        
        def _run_monitor():
            try:
                controller.run_continuous_monitoring()
            except Exception as exc:
                self._append_log(f"持续监控异常: {exc}")
        
        self._monitor_thread = threading.Thread(target=_run_monitor, daemon=True)
        self._monitor_thread.start()
        self.monitor_status_label.setText("监控状态：运行中")
        self.monitor_status_label.setStyleSheet("color: green; font-weight: bold;")
        self._append_log("持续监控已启动")

    def _stop_continuous_monitoring(self) -> None:
        """停止持续监控模式。"""
        controller = getattr(self, '_controller', None)
        if controller is not None and hasattr(controller, 'stop_continuous_monitoring'):
            controller.stop_continuous_monitoring()
        
        self.monitor_status_label.setText("监控状态：已停止")
        self.monitor_status_label.setStyleSheet("color: #888;")
        self._append_log("持续监控已停止")

    # ------------------------------------------------------------------ #
    #  UCC 相机实时预览
    # ------------------------------------------------------------------ #
    def _start_ucc_preview(self) -> None:
        """启动 UCC 相机实时预览。"""
        if self._ucc_preview_running:
            self._append_log("UCC 预览已在运行中")
            return

        device_idx = self._ucc_preview_device.value()
        resolution = self._ucc_preview_resolution.currentText()
        pixel_format = self._ucc_preview_format.currentText()
        if pixel_format == "AUTO":
            pixel_format = None

        try:
            self._ucc_preview_source = UCCFrameSource(
                device_index=device_idx,
                resolution=resolution,
                pixel_format=pixel_format,
            )
        except Exception as exc:
            self._append_log(f"UCC 相机打开失败 (device={device_idx}): {exc}")
            QMessageBox.warning(self, "预览失败", f"无法打开 UCC 相机:\n{exc}")
            return

        # 启动定时器，每 50ms 采集一帧（约 20 FPS）
        self._ucc_preview_timer = QTimer(self)
        self._ucc_preview_timer.setInterval(50)
        self._ucc_preview_timer.timeout.connect(self._update_ucc_preview)
        self._ucc_preview_timer.start()

        self._ucc_preview_running = True
        self._ucc_preview_fps_counter = 0
        self._ucc_preview_fps_time = time.time()
        self._ucc_preview_actual_fps = 0.0
        self._ucc_preview_last_frame: Optional[np.ndarray] = None

        # 清空分析数据
        self._ucc_curve_widget.clear_history()
        self._ucc_profile_widget.clear_profile()
        for i in range(self._ucc_param_table.rowCount()):
            self._ucc_param_table.setItem(i, 1, QTableWidgetItem("--"))

        self._ucc_preview_btn_start.setEnabled(False)
        self._ucc_preview_btn_stop.setEnabled(True)
        self._ucc_preview_btn_save.setEnabled(True)
        self._ucc_preview_info.setText(f"状态: 运行中 | {resolution}")
        self._ucc_preview_info.setStyleSheet("color: #34D399; font-weight: bold;")
        self._append_log(
            f"UCC 实时预览已启动 (device={device_idx}, resolution={resolution}, "
            f"format={pixel_format or 'AUTO'})"
        )

    def _stop_ucc_preview(self) -> None:
        """停止 UCC 相机实时预览。"""
        self._ucc_preview_running = False

        if self._ucc_preview_timer is not None:
            self._ucc_preview_timer.stop()
            self._ucc_preview_timer = None

        if self._ucc_preview_source is not None:
            try:
                self._ucc_preview_source.release()
            except Exception:
                pass
            self._ucc_preview_source = None

        self._ucc_preview_btn_start.setEnabled(True)
        self._ucc_preview_btn_stop.setEnabled(False)
        self._ucc_preview_btn_save.setEnabled(False)
        self._ucc_preview_label.setText("点击「启动预览」打开 UCC 相机画面")
        self._ucc_preview_info.setText("状态: 已停止")
        self._ucc_preview_info.setStyleSheet("color: #888;")
        # 清空分析数据
        self._ucc_curve_widget.clear_history()
        self._ucc_profile_widget.clear_profile()
        for i in range(self._ucc_param_table.rowCount()):
            self._ucc_param_table.setItem(i, 1, QTableWidgetItem("--"))
        self._append_log("UCC 实时预览已停止")

    def _save_ucc_debug_frame(self) -> None:
        """保存当前原始帧和显示帧到调试目录。"""
        if self._ucc_preview_last_frame is None:
            QMessageBox.information(self, "保存调试帧", "暂无可用帧")
            return
        
        debug_dir = Path("UCC_Debug")
        debug_dir.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        
        # 保存原始帧（BGR）
        raw_path = debug_dir / f"ucc_raw_{ts}.png"
        cv2.imwrite(str(raw_path), self._ucc_preview_last_frame)
        
        # 保存 RGB 转换后的帧
        rgb = cv2.cvtColor(self._ucc_preview_last_frame, cv2.COLOR_BGR2RGB)
        rgb_path = debug_dir / f"ucc_rgb_{ts}.png"
        cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        
        # 保存元数据
        meta_path = debug_dir / f"ucc_meta_{ts}.txt"
        h, w = self._ucc_preview_last_frame.shape[:2]
        ch = self._ucc_preview_last_frame.shape[2] if len(self._ucc_preview_last_frame.shape) > 2 else 1
        source = self._ucc_preview_source
        fourcc = source._actual_fourcc_str if source else "N/A"
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(f"分辨率: {w}x{h}\n")
            f.write(f"通道: {ch}\n")
            f.write(f"FourCC: {fourcc}\n")
            f.write(f"dtype: {self._ucc_preview_last_frame.dtype}\n")
        
        self._append_log(f"调试帧已保存到 {debug_dir}/ucc_*_{ts}.*")
        QMessageBox.information(self, "保存成功", f"调试帧已保存到:\n{debug_dir}")

    def _update_ucc_preview(self) -> None:
        """定时器回调：抓帧并显示到预览标签。"""
        source = self._ucc_preview_source
        if source is None or not self._ucc_preview_running:
            return

        try:
            if not source.is_healthy():
                self._ucc_preview_label.setText("⚠️ 相机连接异常，尝试重连...")
                if not source.reconnect():
                    self._stop_ucc_preview()
                    self._append_log("UCC 预览：重连失败")
                    return
                self._append_log("UCC 预览：重连成功")

            frame = source.grab_frame()
            if frame is None:
                return
            self._ucc_preview_last_frame = frame.copy()

            # 统计实际 FPS
            self._ucc_preview_fps_counter += 1
            now = time.time()
            elapsed = now - self._ucc_preview_fps_time
            if elapsed >= 1.0:
                self._ucc_preview_actual_fps = self._ucc_preview_fps_counter / elapsed
                self._ucc_preview_fps_counter = 0
                self._ucc_preview_fps_time = now

            h, w = frame.shape[:2]

            # 光斑分析
            spot = analyze_spot(frame)
            if spot is not None:
                cx, cy = spot["centroid"]
                px, py = spot["peak_loc"]
                wx, wy = spot["width"]

                # 在 BGR 帧上叠加十字准线和标记
                # 十字准线（画面中心）
                cx_int, cy_int = int(round(cx)), int(round(cy))
                color_cross = (0, 255, 255)   # 青色
                color_centroid = (0, 255, 0)  # 绿色
                color_peak = (0, 0, 255)      # 红色

                # 水平线（贯穿画面）
                cv2.line(frame, (0, cy_int), (w, cy_int), color_cross, 1, cv2.LINE_AA)
                # 垂直线（贯穿画面）
                cv2.line(frame, (cx_int, 0), (cx_int, h), color_cross, 1, cv2.LINE_AA)
                # 质心圆点
                cv2.circle(frame, (cx_int, cy_int), 4, color_centroid, -1, cv2.LINE_AA)
                cv2.circle(frame, (cx_int, cy_int), 8, color_centroid, 1, cv2.LINE_AA)
                # 峰值位置叉号
                cv2.drawMarker(frame, (px, py), color_peak, cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA)

                # 更新 XY 曲线（使用相对于画面中心的偏移，单位：像素）
                offset_x = cx - w / 2.0
                offset_y = cy - h / 2.0
                self._ucc_curve_widget.append(offset_x, offset_y)

                # 计算并更新 RGB 剖面图（基于原始 BGR 帧）
                profile = compute_spot_profile(self._ucc_preview_last_frame, cx, cy)
                if profile is not None:
                    self._ucc_profile_widget.set_profile(profile)

                # 更新参数表
                self._ucc_param_table.setItem(0, 1, QTableWidgetItem(f"{spot['min_val']:.1f}, {spot['peak']:.1f}"))
                self._ucc_param_table.setItem(1, 1, QTableWidgetItem(f"{px}, {py}"))
                self._ucc_param_table.setItem(2, 1, QTableWidgetItem(f"{cx:.2f}, {cy:.2f}"))
                self._ucc_param_table.setItem(3, 1, QTableWidgetItem(f"{wx:.2f}, {wy:.2f}"))
            else:
                # 画面太暗，没有检测到光斑
                for i in range(self._ucc_param_table.rowCount()):
                    self._ucc_param_table.setItem(i, 1, QTableWidgetItem("--"))

            # BGR → RGB → QPixmap
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            ch = rgb.shape[2]
            bytes_per_line = int(rgb.strides[0])
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # 按比例缩放至预览区域
            pix = QPixmap.fromImage(qimg)
            scaled = pix.scaled(
                self._ucc_preview_label.width(),
                self._ucc_preview_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self._ucc_preview_label.setPixmap(scaled)

            # 更新信息栏
            info_text = (
                f"状态: 运行中 | {w}×{h} @ {self._ucc_preview_actual_fps:.1f} FPS"
                f" | {source.resolution}"
            )
            self._ucc_preview_info.setText(info_text)
            self._ucc_preview_info.setStyleSheet("color: #34D399; font-weight: bold;")

        except Exception as exc:
            self._append_log(f"UCC 预览帧采集异常: {exc}")

    def _on_test_selected(self) -> None:
        item = self.test_tree.currentItem()
        if item is None:
            self.test_desc.setText("选择一个测试项")
            return
        spec = item.data(0, Qt.UserRole)
        if isinstance(spec, TestCaseSpec):
            self.test_desc.setText(f"{spec.title}\n{spec.description}")

    def _load_test_tree(self) -> None:
        self.test_tree.clear()
        groups: Dict[str, QTreeWidgetItem] = {}
        for spec in self.device_test.list_tests():
            if spec.category not in groups:
                groups[spec.category] = QTreeWidgetItem([spec.category, spec.category])
                self.test_tree.addTopLevelItem(groups[spec.category])
            item = QTreeWidgetItem([spec.title, spec.category])
            item.setData(0, Qt.UserRole, spec)
            groups[spec.category].addChild(item)
        self.test_tree.expandAll()

    def _select_frame_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择模拟图像",
            str(self.runtime.repo_root),
            "Images (*.png *.jpg *.jpeg *.bmp)",
        )
        if path:
            self._set_text("frame_source_image", path)
            self._refresh_preview_images()

    def _select_frame_image_2(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择探测器2模拟图像",
            str(self.runtime.repo_root),
            "Images (*.png *.jpg *.jpeg *.bmp)",
        )
        if path:
            self._set_text("frame_source_image_2", path)
            self._refresh_preview_images()

    def _select_frame_tmp_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "选择中间帧保存目录",
            str(Path.cwd() / "FrameTmp"),
        )
        if path:
            self._set_text("frame_tmp_dir", path)

    def _on_alignment_strategy_changed(self, strategy: str) -> None:
        if strategy == "dual_detector_4axis":
            self._set_check("disable_z_axis", True)
            self.strategy_desc.setText("4轴模式：双镜闭环，禁用Z轴")
            self.strategy_desc.setStyleSheet("color: #4a9; font-style: italic;")
            self._append_log("4轴双镜闭环模式已启用，自动禁用Z轴")
        else:
            self.strategy_desc.setText("5轴模式：Z扫描 + XY对准")
            self.strategy_desc.setStyleSheet("color: #94a; font-style: italic;")
            self._set_check("disable_z_axis", False)
            self._append_log("5轴Z扫描模式已启用")

    def _on_detector_mode_changed(self, mode: str) -> None:
        if mode == "dual_detector":
            self._set_check("disable_z_axis", True)
            self._append_log("双探测器模式已启用，自动禁用Z轴")
        else:
            pass

    def _use_sample_frame(self) -> None:
        sample = str(self.runtime.ensure_sample_frame())
        self._set_text("frame_source_image", sample)
        self._append_log(f"已切换到示例图像: {sample}")
        self._refresh_preview_images()

    def _refresh_preview_images(self) -> None:
        path = self._text_value("frame_source_image", "")
        if not path:
            path = str(self.runtime.ensure_sample_frame())
        qimg = QImage(path)
        if qimg.isNull():
            sim_preview = getattr(self, "sim_preview", None)
            if sim_preview:
                sim_preview.setText("图像加载失败")
            return
        pix = QPixmap.fromImage(qimg)
        sim_preview = getattr(self, "sim_preview", None)
        if sim_preview:
            sim_preview.setPixmap(pix.scaled(sim_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _export_report(self) -> None:
        self.profile = self._collect_profile_from_controls()
        report = self.runtime.read_run_report(self.profile.run_report_json)
        if not report:
            QMessageBox.information(self, "导出报告", "当前没有可导出的运行报告")
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出运行报告",
            str(self.runtime.repo_root / "artifacts" / "spotzoom_ui_export_report.json"),
            "JSON (*.json)",
        )
        if not out_path:
            return
        Path(out_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        self._append_log(f"已导出报告 {out_path}")

    def _export_config_json(self) -> None:
        self.profile = self._collect_profile_from_controls()
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出配置",
            str(self.runtime.repo_root / "artifacts" / "spotzoom_ui_profile.json"),
            "JSON (*.json)",
        )
        if not out_path:
            return
        payload = self.profile.__dict__.copy()
        payload["run_mode"] = self.profile.run_mode.value
        Path(out_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self._append_log(f"已导出配置 {out_path}")

    def _import_config_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "导入配置",
            str(self.runtime.repo_root / "artifacts"),
            "JSON (*.json)",
        )
        if not path:
            return
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.profile = self.runtime.default_profile()
        for key, value in payload.items():
            if key == "run_mode":
                value = RunMode(value)
            if hasattr(self.profile, key):
                setattr(self.profile, key, value)
        self._apply_profile_to_controls(self.profile)
        self._refresh_all_panels()
        self._append_log(f"已导入配置 {path}")

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._load_test_tree()
        self._refresh_all_panels()

    def closeEvent(self, event) -> None:
        # 停止 UCC 预览
        self._stop_ucc_preview()
        self._stop_alignment_ucc_preview()
        self._save_ui_preferences()

        panel = getattr(self, "picomotor_driver_panel", None)
        if panel is not None:
            try:
                panel.shutdown()
            except Exception:
                pass
        super().closeEvent(event)


def run_qt_ui(repo_root: Optional[Path] = None) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = SpotZoomQtMainWindow(repo_root=repo_root)
    window.show()
    return app_exec(app)









