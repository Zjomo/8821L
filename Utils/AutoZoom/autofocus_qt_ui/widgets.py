"""
autofocus_qt_ui 专用 UI 控件：
  - 支持 ROI 框选的预览标签
  - 聚焦分数趋势图（轻量自绘，不依赖 matplotlib）
  - 指标表格
"""

from __future__ import annotations

from typing import Deque, Optional, Tuple
from collections import deque

import numpy as np

from .qt_compat import (
    Qt,
    Signal,
    QColor,
    QCursor,
    QFont,
    QPainter,
    QPen,
    QImage,
    QPixmap,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
    QVBoxLayout,
)

from .themes import Theme, get_theme


class RoiPreviewLabel(QLabel):
    """
    支持鼠标拖拽框选 ROI 的图像预览标签。

    - reference_mode=True 时允许框选；
    - 框选完成后发出 roiSelected(x, y, w, h)。
    """

    roiSelected = Signal(int, int, int, int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._theme = get_theme("dark")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 320)
        self.setMouseTracking(True)
        self._pixmap: Optional[QPixmap] = None
        self._reference_mode = False
        self._drawing = False
        self._start_pos: Optional[Tuple[int, int]] = None
        self._current_pos: Optional[Tuple[int, int]] = None
        self._roi: Optional[Tuple[int, int, int, int]] = None
        self.apply_theme(self._theme)

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        self.setStyleSheet(
            f"background-color: {theme.bg_secondary}; color: {theme.text_disabled}; "
            f"border: 1px solid {theme.border}; border-radius: 4px; font-size: 14px;"
        )
        self.update()

    def set_reference_mode(self, enabled: bool) -> None:
        self._reference_mode = enabled
        self.setCursor(QCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor))
        self.update()

    def set_preview_image(self, image: np.ndarray) -> None:
        """从 numpy BGR/RGB 数组设置预览图，自动等比缩放。"""
        if image is None or image.size == 0:
            self._pixmap = None
            self.update()
            return

        h, w = image.shape[:2]
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        # OpenCV 默认 BGR，转换为 RGB 用于 QImage
        rgb = image[:, :, ::-1] if image.shape[2] == 3 else image
        # QImage 要求底层 buffer C-contiguous；切片可能产生非连续视图
        rgb = np.ascontiguousarray(rgb)
        bytes_per_line = 3 * w
        qimage = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage.copy())

        scaled = pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._pixmap = scaled
        self.update()

    def clear_preview(self) -> None:
        self._pixmap = None
        self._roi = None
        self.update()

    def set_roi(self, x: int, y: int, w: int, h: int) -> None:
        self._roi = (int(x), int(y), int(w), int(h))
        self.update()

    def get_roi(self) -> Optional[Tuple[int, int, int, int]]:
        return self._roi

    def paintEvent(self, event) -> None:
        theme = self._theme
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(theme.bg_secondary))

        if self._pixmap is not None:
            x = (self.width() - self._pixmap.width()) // 2
            y = (self.height() - self._pixmap.height()) // 2
            painter.drawPixmap(x, y, self._pixmap)
        else:
            painter.setPen(QColor(theme.text_disabled))
            painter.setFont(QFont("Microsoft YaHei", 12))
            painter.drawText(self.rect(), Qt.AlignCenter, "暂无预览")

        roi = self._get_display_roi()
        if roi is not None:
            rx, ry, rw, rh = roi
            pen = QPen(QColor(theme.roi_border))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(rx, ry, rw, rh)
            painter.setPen(QColor(theme.roi_text))
            painter.setFont(QFont("Microsoft YaHei", 9))
            painter.drawText(rx, ry - 6, f"ROI {rw}x{rh}")

        if self._drawing and self._start_pos and self._current_pos:
            x1, y1 = self._start_pos
            x2, y2 = self._current_pos
            rx, ry, rw, rh = min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)
            pen = QPen(QColor(theme.warning))
            pen.setWidth(2)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(rx, ry, rw, rh)

        painter.end()

    def _get_display_roi(self) -> Optional[Tuple[int, int, int, int]]:
        """将 ROI（相对于原始图像坐标）映射到当前显示坐标。"""
        if self._roi is None or self._pixmap is None:
            return None
        px = (self.width() - self._pixmap.width()) // 2
        py = (self.height() - self._pixmap.height()) // 2
        rx, ry, rw, rh = self._roi
        return (px + rx, py + ry, rw, rh)

    def mousePressEvent(self, event) -> None:
        if not self._reference_mode:
            return
        self._drawing = True
        self._start_pos = (event.x(), event.y())
        self._current_pos = self._start_pos
        self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._drawing:
            self._current_pos = (event.x(), event.y())
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if not self._drawing:
            return
        self._drawing = False
        self._current_pos = (event.x(), event.y())
        if self._start_pos is None:
            return

        x1, y1 = self._start_pos
        x2, y2 = self._current_pos
        rx, ry, rw, rh = min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)

        if self._pixmap is not None:
            px = (self.width() - self._pixmap.width()) // 2
            py = (self.height() - self._pixmap.height()) // 2
            rx -= px
            ry -= py
            rx = max(0, rx)
            ry = max(0, ry)
            rw = min(rw, self._pixmap.width() - rx)
            rh = min(rh, self._pixmap.height() - ry)

        if rw > 10 and rh > 10:
            self._roi = (rx, ry, rw, rh)
            self.roiSelected.emit(rx, ry, rw, rh)
        self.update()


class FocusScorePlot(QWidget):
    """自绘 FocusScore_ratio 趋势图。"""

    def __init__(self, parent: Optional[QWidget] = None, max_points: int = 200) -> None:
        super().__init__(parent)
        self._theme = get_theme("dark")
        self.setMinimumSize(300, 180)
        self._scores: Deque[Tuple[int, float]] = deque(maxlen=max_points)
        self._trigger_ratio: float = 0.90
        self._stop_ratio: float = 0.95

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        self.update()

    def append(self, cycle: int, score: float) -> None:
        self._scores.append((cycle, float(score)))
        self.update()

    def clear(self) -> None:
        self._scores.clear()
        self.update()

    def set_thresholds(self, trigger: float, stop: float) -> None:
        self._trigger_ratio = trigger
        self._stop_ratio = stop
        self.update()

    def paintEvent(self, event) -> None:
        theme = self._theme
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        painter.fillRect(self.rect(), QColor(theme.plot_bg))

        if not self._scores:
            painter.setPen(QColor(theme.text_disabled))
            painter.setFont(QFont("Microsoft YaHei", 11))
            painter.drawText(self.rect(), Qt.AlignCenter, "FocusScore 曲线")
            return

        margin = 40
        plot_w = w - 2 * margin
        plot_h = h - 2 * margin
        left, top = margin, margin

        painter.setPen(QPen(QColor(theme.plot_grid), 1))
        painter.drawLine(left, top + plot_h, left + plot_w, top + plot_h)
        painter.drawLine(left, top, left, top + plot_h)

        for val, color, label in [
            (self._stop_ratio, theme.success, "stop"),
            (self._trigger_ratio, theme.warning, "trigger"),
        ]:
            y = top + plot_h - int(val * plot_h)
            painter.setPen(QPen(QColor(color), 1, Qt.DashLine))
            painter.drawLine(left, y, left + plot_w, y)
            painter.setPen(QColor(color))
            painter.setFont(QFont("Microsoft YaHei", 8))
            painter.drawText(left + plot_w + 4, y + 4, label)

        cycles = [c for c, _ in self._scores]
        scores = [s for _, s in self._scores]
        min_c, max_c = cycles[0], cycles[-1]
        if max_c == min_c:
            max_c += 1

        points = []
        for c, s in self._scores:
            x = left + int((c - min_c) / (max_c - min_c) * plot_w)
            y = top + plot_h - int(s * plot_h)
            points.append((x, y))

        if len(points) > 1:
            painter.setPen(QPen(QColor(theme.plot_line), 2))
            for i in range(len(points) - 1):
                painter.drawLine(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1])

        painter.setBrush(QColor(theme.plot_point))
        painter.setPen(Qt.NoPen)
        for x, y in points:
            painter.drawEllipse(x - 2, y - 2, 4, 4)

        painter.setPen(QColor(theme.plot_text))
        painter.setFont(QFont("Microsoft YaHei", 8))
        painter.drawText(left, top + plot_h + 18, str(min_c))
        painter.drawText(left + plot_w - 20, top + plot_h + 18, str(max_c))
        painter.drawText(4, top + 12, "1.0")
        painter.drawText(4, top + plot_h, "0.0")

        painter.end()


class MetricsTable(QTableWidget):
    """显示 14 项聚焦指标的表格。"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._theme = get_theme("dark")
        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(["指标", "整图", "ROI"])
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setAlternatingRowColors(True)
        self.setMinimumHeight(200)
        self.apply_theme(self._theme)

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        self.setStyleSheet(
            f"QHeaderView::section {{ background-color: {theme.table_header_bg}; color: {theme.text_primary}; }}"
            f"QTableWidget {{ background-color: {theme.bg_secondary}; color: {theme.text_primary}; gridline-color: {theme.border}; }}"
            f"QTableWidget::item:alternate {{ background-color: {theme.table_row_alt}; }}"
        )

    def update_metrics(self, full: dict, roi: dict) -> None:
        names = [
            "tenengrad", "laplacian_var", "brenner", "highfreq_ratio",
            "local_contrast", "edge_width", "halo_width", "brightness_mean",
            "brightness_std", "red_blue_ratio", "modified_laplacian",
            "dct_energy", "smd", "entropy",
        ]
        self.setRowCount(len(names))
        for i, name in enumerate(names):
            self.setItem(i, 0, QTableWidgetItem(name))
            self.setItem(i, 1, QTableWidgetItem(self._fmt(full.get(name))))
            self.setItem(i, 2, QTableWidgetItem(self._fmt(roi.get(name))))
        self.resizeColumnsToContents()

    @staticmethod
    def _fmt(value) -> str:
        if value is None:
            return "-"
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return str(value)
