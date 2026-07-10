"""
autofocus_qt_ui 专用 UI 控件：
  - 支持 ROI 框选的预览标签（含交互式剖面线）
  - 聚焦分数趋势图（轻量自绘，不依赖 matplotlib）
  - 指标表格
  - RGB 三通道剖面图
"""

from __future__ import annotations

from typing import Deque, List, Optional, Tuple
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
    QHBoxLayout,
    QDialog,
    QGridLayout,
    QPushButton,
    QScrollArea,
    QSizePolicy,
)

from .themes import Theme, get_theme


class RoiPreviewLabel(QLabel):
    """
    支持鼠标拖拽框选 ROI 和交互式剖面线的图像预览标签。

    - reference_mode=True 时允许框选 ROI；
    - 框选完成后发出 roiSelected(x, y, w, h)；
    - 支持横线/竖线剖面线，可点击选中并拖动；
    - 剖面线位置变化时发出 crossHairChanged(x, y)。
    """

    roiSelected = Signal(int, int, int, int)
    crossHairChanged = Signal(int, int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._theme = get_theme("light")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 320)
        self.setMouseTracking(True)
        self._pixmap: Optional[QPixmap] = None
        self._original_image: Optional[np.ndarray] = None
        self._reference_image: Optional[np.ndarray] = None
        self._reference_pixmap: Optional[QPixmap] = None
        self._comparison_mode: bool = False
        self._reference_mode = False
        self._drawing = False
        self._start_pos: Optional[Tuple[int, int]] = None
        self._current_pos: Optional[Tuple[int, int]] = None
        self._roi: Optional[Tuple[int, int, int, int]] = None

        # 剖面线：相对于显示区域的坐标 (px)
        self._cross_hair_x: Optional[int] = None
        self._cross_hair_y: Optional[int] = None
        self._cross_hair_enabled: bool = True
        self._show_v_line: bool = True
        self._show_h_line: bool = True
        self._dragging_line: Optional[str] = None  # "h" / "v" / None
        self._zoom_ratio: float = 1.0

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

    def set_cross_hair_enabled(self, enabled: bool) -> None:
        self._cross_hair_enabled = enabled
        self.update()

    def set_show_v_line(self, show: bool) -> None:
        self._show_v_line = show
        self.update()

    def set_show_h_line(self, show: bool) -> None:
        self._show_h_line = show
        self.update()

    def set_zoom_ratio(self, ratio: float) -> None:
        self._zoom_ratio = ratio
        self.update()

    def set_cross_hair_position(self, x: int, y: int) -> None:
        """设置剖面线交叉点位置（相对于原始图像坐标）。"""
        self._cross_hair_x = int(x)
        self._cross_hair_y = int(y)
        self.update()

    def get_cross_hair_position(self) -> Optional[Tuple[int, int]]:
        if self._cross_hair_x is not None and self._cross_hair_y is not None:
            return (self._cross_hair_x, self._cross_hair_y)
        return None

    def set_preview_image(self, image: np.ndarray) -> None:
        """从 numpy BGR/RGB 数组设置预览图，自动等比缩放。"""
        if image is None or image.size == 0:
            self._pixmap = None
            self._original_image = None
            self.update()
            return

        self._original_image = image.copy()
        self._pixmap = self._array_to_pixmap(image, self._available_single_size())

        # 自动初始化剖面线位置到图像中心
        h, w = image.shape[:2]
        if self._cross_hair_x is None or self._cross_hair_y is None:
            self._cross_hair_x = w // 2
            self._cross_hair_y = h // 2

        self.update()

    def set_reference_image(self, image: Optional[np.ndarray]) -> None:
        """设置基准图；传入 None 则清除。"""
        if image is None or image.size == 0:
            self._reference_image = None
            self._reference_pixmap = None
            self.update()
            return
        self._reference_image = image.copy()
        self._reference_pixmap = self._array_to_pixmap(image, self._available_single_size())
        self.update()

    def set_comparison_mode(self, enabled: bool) -> None:
        """启用/禁用基准图与实时图左右对比模式。"""
        self._comparison_mode = bool(enabled)
        if self._original_image is not None:
            self._pixmap = self._array_to_pixmap(self._original_image, self._available_single_size())
        if self._reference_image is not None:
            self._reference_pixmap = self._array_to_pixmap(self._reference_image, self._available_single_size())
        self.update()

    def _available_single_size(self) -> Tuple[int, int]:
        """对比模式下单张图可用的尺寸；非对比模式返回整个控件尺寸。"""
        if self._comparison_mode:
            return (max(1, self.width() // 2 - 8), max(1, self.height() - 16))
        return (self.width(), self.height())

    @staticmethod
    def _array_to_pixmap(image: np.ndarray, target_size: Tuple[int, int]) -> QPixmap:
        """将 numpy 图像转换为按目标尺寸等比缩放后的 QPixmap。"""
        h, w = image.shape[:2]
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        rgb = image[:, :, ::-1] if image.shape[2] == 3 else image
        rgb = np.ascontiguousarray(rgb)
        bytes_per_line = 3 * w
        qimage = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage.copy())

        target_w, target_h = target_size
        scaled = pixmap.scaled(
            target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        return scaled

    def clear_preview(self) -> None:
        self._pixmap = None
        self._original_image = None
        self._roi = None
        self.update()

    def set_roi(self, x: int, y: int, w: int, h: int) -> None:
        self._roi = (int(x), int(y), int(w), int(h))
        self.update()

    def get_roi(self) -> Optional[Tuple[int, int, int, int]]:
        return self._roi

    def get_original_image(self) -> Optional[np.ndarray]:
        return self._original_image

    def _display_to_image_coord(self, dx: int, dy: int) -> Tuple[int, int]:
        """将显示坐标转换为原始图像坐标。"""
        if self._pixmap is None:
            return dx, dy
        px = (self.width() - self._pixmap.width()) // 2
        py = (self.height() - self._pixmap.height()) // 2
        scale_x = self._original_image.shape[1] / self._pixmap.width() if self._original_image is not None else 1.0
        scale_y = self._original_image.shape[0] / self._pixmap.height() if self._original_image is not None else 1.0
        ix = int((dx - px) * scale_x)
        iy = int((dy - py) * scale_y)
        return ix, iy

    def _image_to_display_coord(self, ix: int, iy: int) -> Tuple[int, int]:
        """将原始图像坐标转换为显示坐标。"""
        if self._pixmap is None:
            return ix, iy
        px = (self.width() - self._pixmap.width()) // 2
        py = (self.height() - self._pixmap.height()) // 2
        scale_x = self._pixmap.width() / self._original_image.shape[1] if self._original_image is not None else 1.0
        scale_y = self._pixmap.height() / self._original_image.shape[0] if self._original_image is not None else 1.0
        dx = int(ix * scale_x) + px
        dy = int(iy * scale_y) + py
        return dx, dy

    def paintEvent(self, event) -> None:
        theme = self._theme
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(theme.bg_secondary))

        if self._comparison_mode and self._reference_pixmap is not None:
            # 左右对比：左侧基准图，右侧实时图
            ref_pix = self._reference_pixmap
            live_pix = self._pixmap

            gap = 8
            total_w = ref_pix.width() + (live_pix.width() if live_pix is not None else 0) + gap
            x = (self.width() - total_w) // 2
            y = (self.height() - ref_pix.height()) // 2

            painter.drawPixmap(x, y, ref_pix)
            self._draw_image_label(painter, x, y, ref_pix.width(), "基准图", theme)

            if live_pix is not None:
                live_x = x + ref_pix.width() + gap
                live_y = (self.height() - live_pix.height()) // 2
                painter.drawPixmap(live_x, live_y, live_pix)
                self._draw_image_label(painter, live_x, live_y, live_pix.width(), "实时图", theme)

            # 绘制 ROI（仅针对实时图区域）
            if live_pix is not None:
                self._draw_roi_on_comparison(painter, live_x, live_y, live_pix)
        elif self._pixmap is not None:
            x = (self.width() - self._pixmap.width()) // 2
            y = (self.height() - self._pixmap.height()) // 2
            painter.drawPixmap(x, y, self._pixmap)
        else:
            painter.setPen(QColor(theme.text_disabled))
            painter.setFont(QFont("Microsoft YaHei", 12))
            painter.drawText(self.rect(), Qt.AlignCenter, "暂无预览")
            painter.end()
            return

        # 绘制 ROI
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

        # 绘制交互式剖面线（横线和竖线）
        if self._cross_hair_enabled and self._cross_hair_x is not None and self._cross_hair_y is not None:
            dcx, dcy = self._image_to_display_coord(self._cross_hair_x, self._cross_hair_y)
            pix_x = (self.width() - self._pixmap.width()) // 2
            pix_y = (self.height() - self._pixmap.height()) // 2
            pix_w = self._pixmap.width()
            pix_h = self._pixmap.height()

            # 竖线：红色
            if self._show_v_line:
                pen = QPen(QColor(255, 60, 60))
                pen.setWidth(2)
                pen.setStyle(Qt.DashLine)
                painter.setPen(pen)
                painter.drawLine(dcx, pix_y, dcx, pix_y + pix_h)

            # 横线：蓝色
            if self._show_h_line:
                pen = QPen(QColor(60, 120, 255))
                pen.setWidth(2)
                pen.setStyle(Qt.DashLine)
                painter.setPen(pen)
                painter.drawLine(pix_x, dcy, pix_x + pix_w, dcy)

            # 交叉点圆圈
            if self._show_v_line or self._show_h_line:
                painter.setPen(QPen(QColor(255, 255, 60), 2))
                painter.setBrush(QColor(255, 255, 60, 80))
                painter.drawEllipse(dcx - 6, dcy - 6, 12, 12)

                # 坐标标签
                painter.setPen(QColor(255, 255, 60))
                painter.setFont(QFont("Consolas", 8))
                painter.drawText(dcx + 10, dcy - 8, f"({self._cross_hair_x}, {self._cross_hair_y})")

        # 绘制正在拖拽的 ROI 框
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
        """将 ROI（原始图像坐标）映射到当前显示坐标。"""
        if self._roi is None or self._pixmap is None or self._original_image is None:
            return None
        px = (self.width() - self._pixmap.width()) // 2
        py = (self.height() - self._pixmap.height()) // 2
        scale_x = self._pixmap.width() / self._original_image.shape[1]
        scale_y = self._pixmap.height() / self._original_image.shape[0]
        rx, ry, rw, rh = self._roi
        return (
            int(rx * scale_x) + px,
            int(ry * scale_y) + py,
            int(rw * scale_x),
            int(rh * scale_y),
        )

    def _draw_image_label(
        self,
        painter: QPainter,
        x: int,
        y: int,
        width: int,
        label: str,
        theme: Theme,
    ) -> None:
        """在图像左上角绘制半透明标签。"""
        painter.setPen(QColor(theme.text_primary))
        painter.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        text_rect = painter.boundingRect(x + 4, y + 4, width - 8, 20, Qt.AlignLeft, label)
        painter.fillRect(text_rect.adjusted(-2, -2, 2, 2), QColor(0, 0, 0, 160))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(text_rect, Qt.AlignLeft, label)

    def _draw_roi_on_comparison(
        self,
        painter: QPainter,
        pix_x: int,
        pix_y: int,
        pixmap: QPixmap,
    ) -> None:
        """在对比模式的实时图区域绘制 ROI。"""
        if self._roi is None or self._original_image is None:
            return
        scale_x = pixmap.width() / self._original_image.shape[1]
        scale_y = pixmap.height() / self._original_image.shape[0]
        rx, ry, rw, rh = self._roi
        dx = int(rx * scale_x) + pix_x
        dy = int(ry * scale_y) + pix_y
        dw = int(rw * scale_x)
        dh = int(rh * scale_y)
        pen = QPen(QColor(self._theme.roi_border))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawRect(dx, dy, dw, dh)

    def _is_near_line(self, ex: int, ey: int) -> Optional[str]:
        """判断鼠标是否靠近某条剖面线，返回 'h' / 'v' / None。"""
        if not self._cross_hair_enabled or self._pixmap is None:
            return None
        if self._cross_hair_x is None or self._cross_hair_y is None:
            return None
        dcx, dcy = self._image_to_display_coord(self._cross_hair_x, self._cross_hair_y)
        threshold = 8
        if abs(ex - dcx) <= threshold:
            return "v"
        if abs(ey - dcy) <= threshold:
            return "h"
        return None

    def mousePressEvent(self, event) -> None:
        if self._pixmap is None:
            return

        ex, ey = event.x(), event.y()

        # 检查是否靠近剖面线
        line = self._is_near_line(ex, ey)
        if line is not None:
            self._dragging_line = line
            self.setCursor(QCursor(Qt.ClosedHandCursor))
            return

        if not self._reference_mode:
            return
        self._drawing = True
        self._start_pos = (ex, ey)
        self._current_pos = self._start_pos
        self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._dragging_line is not None:
            # 拖动剖面线
            ix, iy = self._display_to_image_coord(event.x(), event.y())
            if self._original_image is not None:
                h, w = self._original_image.shape[:2]
                ix = max(0, min(w - 1, ix))
                iy = max(0, min(h - 1, iy))
            self._cross_hair_x = ix
            self._cross_hair_y = iy
            self.crossHairChanged.emit(ix, iy)
            self.update()
            return

        if self._drawing:
            self._current_pos = (event.x(), event.y())
            self.update()
            return

        # 更新鼠标样式：靠近剖面线时显示手型
        line = self._is_near_line(event.x(), event.y())
        if line is not None:
            self.setCursor(QCursor(Qt.OpenHandCursor))
        else:
            self.setCursor(QCursor(Qt.CrossCursor if self._reference_mode else Qt.ArrowCursor))

    def mouseReleaseEvent(self, event) -> None:
        if self._dragging_line is not None:
            self._dragging_line = None
            self.setCursor(QCursor(Qt.OpenHandCursor))
            return

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
            # 转为相对于 pixmap 的坐标
            rx -= px
            ry -= py
            rx = max(0, rx)
            ry = max(0, ry)
            rw = min(rw, self._pixmap.width() - rx)
            rh = min(rh, self._pixmap.height() - ry)

            # 缩放为原始图像坐标
            if self._original_image is not None:
                scale_x = self._original_image.shape[1] / self._pixmap.width()
                scale_y = self._original_image.shape[0] / self._pixmap.height()
                rx = int(rx * scale_x)
                ry = int(ry * scale_y)
                rw = int(rw * scale_x)
                rh = int(rh * scale_y)

        if rw > 10 and rh > 10:
            self._roi = (rx, ry, rw, rh)
            self.roiSelected.emit(rx, ry, rw, rh)
        self.update()


class FocusScorePlot(QWidget):
    """自绘 FocusScore_ratio 趋势图，支持鼠标悬停查看数据点。"""

    def __init__(self, parent: Optional[QWidget] = None, max_points: int = 200) -> None:
        super().__init__(parent)
        self._theme = get_theme("light")
        self.setMinimumSize(300, 200)
        self.setMouseTracking(True)
        self._scores: Deque[Tuple[int, float]] = deque(maxlen=max_points)
        self._trigger_ratio: float = 0.90
        self._stop_ratio: float = 0.95
        self._hover_cycle: Optional[int] = None
        self._hover_score: Optional[float] = None
        self._hover_pos: Optional[Tuple[int, int]] = None

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        self.update()

    def append(self, cycle: int, score: float) -> None:
        self._scores.append((cycle, float(score)))
        self.update()

    def clear(self) -> None:
        self._scores.clear()
        self._hover_cycle = None
        self._hover_score = None
        self._hover_pos = None
        self.update()

    def set_thresholds(self, trigger: float, stop: float) -> None:
        self._trigger_ratio = trigger
        self._stop_ratio = stop
        self.update()

    def mouseMoveEvent(self, event) -> None:
        self._hover_cycle = None
        self._hover_score = None
        self._hover_pos = (event.x(), event.y())
        if not self._scores:
            self.update()
            return
        w, h = self.width(), self.height()
        margin = 45
        plot_w = w - 2 * margin
        plot_h = h - 2 * margin
        left, top = margin, margin
        mx, my = event.x(), event.y()
        if left <= mx <= left + plot_w and top <= my <= top + plot_h:
            cycles = [c for c, _ in self._scores]
            scores = [s for _, s in self._scores]
            min_c, max_c = cycles[0], cycles[-1]
            if max_c == min_c:
                max_c += 1
            # 找到最近的 x 坐标对应的数据点
            rel_x = (mx - left) / plot_w
            idx = int(rel_x * (len(cycles) - 1))
            idx = max(0, min(len(cycles) - 1, idx))
            self._hover_cycle = cycles[idx]
            self._hover_score = scores[idx]
        self.update()

    def leaveEvent(self, event) -> None:
        self._hover_cycle = None
        self._hover_score = None
        self._hover_pos = None
        self.update()

    def paintEvent(self, event) -> None:
        theme = self._theme
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        painter.fillRect(self.rect(), QColor(theme.plot_bg))

        # 标题
        painter.setPen(QColor(theme.text_primary))
        painter.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        painter.drawText(0, 4, w, 20, Qt.AlignCenter, "FocusScore 趋势图")

        if not self._scores:
            painter.setPen(QColor(theme.text_disabled))
            painter.setFont(QFont("Microsoft YaHei", 11))
            painter.drawText(self.rect(), Qt.AlignCenter, "暂无数据")
            painter.end()
            return

        margin = 45
        plot_w = w - 2 * margin
        plot_h = h - 2 * margin
        left, top = margin, margin

        # Y 轴网格线 + 标签
        y_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
        for yt in y_ticks:
            y = top + plot_h - int(yt * plot_h)
            painter.setPen(QPen(QColor(theme.plot_grid), 1, Qt.DotLine))
            painter.drawLine(left, y, left + plot_w, y)
            painter.setPen(QColor(theme.plot_text))
            painter.setFont(QFont("Consolas", 7))
            painter.drawText(2, y + 4, f"{yt:.2f}")

        # X 轴
        painter.setPen(QPen(QColor(theme.plot_grid), 1))
        painter.drawLine(left, top + plot_h, left + plot_w, top + plot_h)
        painter.drawLine(left, top, left, top + plot_h)

        # 阈值线
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

        # 曲线
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

        # 数据点
        painter.setBrush(QColor(theme.plot_point))
        painter.setPen(Qt.NoPen)
        for x, y in points:
            painter.drawEllipse(x - 2, y - 2, 4, 4)

        # X 轴刻度标签
        painter.setPen(QColor(theme.plot_text))
        painter.setFont(QFont("Microsoft YaHei", 8))
        painter.drawText(left, top + plot_h + 18, str(min_c))
        right_label = str(max_c)
        painter.drawText(left + plot_w - painter.fontMetrics().horizontalAdvance(right_label) - 2,
                         top + plot_h + 18, right_label)

        # 最新值显示（右上角）
        if scores:
            latest_s = scores[-1]
            latest_c = cycles[-1]
            painter.setPen(QColor(theme.text_primary))
            painter.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
            value_text = f"cycle={latest_c}  score={latest_s:.4f}"
            tw = painter.fontMetrics().horizontalAdvance(value_text)
            painter.drawText(left + plot_w - tw - 6, top + 4, value_text)

        # 鼠标悬停 tooltip
        if self._hover_cycle is not None and self._hover_score is not None and self._hover_pos is not None:
            mx, my = self._hover_pos
            hover_text = f"cycle={self._hover_cycle}  score={self._hover_score:.4f}"
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(0, 0, 0, 190))
            fm = painter.fontMetrics()
            text_w = fm.horizontalAdvance(hover_text) + 12
            tx, ty = mx + 12, my - 24
            if tx + text_w > w:
                tx = mx - text_w - 12
            if ty < 0:
                ty = my + 4
            painter.drawRoundedRect(tx, ty, text_w, 22, 4, 4)
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Consolas", 8))
            painter.drawText(tx + 6, ty + 15, hover_text)

        painter.end()


class MetricsTable(QTableWidget):
    """显示 14 项聚焦指标的表格。"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._theme = get_theme("light")
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


class RgbProfilePlot(QWidget):
    """RGB 三通道剖面图，显示横线和竖线上的像素值曲线。

    提供：
      - 上方：竖线（垂直方向）上 RGB 三通道像素值曲线
      - 下方：横线（水平方向）上 RGB 三通道像素值曲线
      - 每条曲线用对应颜色绘制（R=红, G=绿, B=蓝）
      - 显示峰值及 FWHM（半高宽）量化聚焦程度
      - 鼠标悬停时显示当前点像素值 (R, G, B)
    """

    pixelHovered = Signal(int, int, int)  # R, G, B

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._theme = get_theme("light")
        self.setMinimumSize(300, 300)
        self.setMouseTracking(True)
        self._profile_h: Optional[List[Tuple[float, float, float]]] = None
        self._profile_v: Optional[List[Tuple[float, float, float]]] = None
        self._fwhm_h: Optional[Tuple[float, float, float]] = None
        self._fwhm_v: Optional[Tuple[float, float, float]] = None
        self._hover_rgb: Optional[Tuple[int, int, int]] = None
        self._hover_pos: Optional[Tuple[int, int]] = None
        self._hover_label: str = ""
        self._cross_hair_x: int = 0
        self._cross_hair_y: int = 0

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        self.update()

    def set_cross_hair_position(self, x: int, y: int) -> None:
        """设置当前十字线交叉点位置，用于悬停 tooltip 中显示实际 (x,y) 坐标。"""
        self._cross_hair_x = x
        self._cross_hair_y = y

    def set_profile(self, profile_h: np.ndarray, profile_v: np.ndarray) -> None:
        """设置剖面数据。

        profile_h: (N, 3) 横线（水平方向）上的 RGB 值
        profile_v: (N, 3) 竖线（垂直方向）上的 RGB 值
        """
        if profile_h is not None and len(profile_h) > 0:
            self._profile_h = [(float(r), float(g), float(b)) for r, g, b in profile_h]
            self._fwhm_h = self._compute_fwhm(profile_h)
        else:
            self._profile_h = None
            self._fwhm_h = None

        if profile_v is not None and len(profile_v) > 0:
            self._profile_v = [(float(r), float(g), float(b)) for r, g, b in profile_v]
            self._fwhm_v = self._compute_fwhm(profile_v)
        else:
            self._profile_v = None
            self._fwhm_v = None

        self.update()

    @staticmethod
    def _compute_fwhm(profile: np.ndarray) -> Tuple[float, float, float]:
        """计算 RGB 三通道各自的半高宽 (FWHM)。"""
        fwhms = []
        for ch in range(3):
            arr = profile[:, ch].astype(np.float64)
            if len(arr) < 3:
                fwhms.append(0.0)
                continue
            peak = float(np.max(arr))
            baseline = float(np.min(arr))
            half_max = (peak + baseline) / 2.0
            if peak - baseline < 1e-6:
                fwhms.append(0.0)
                continue
            above = arr >= half_max
            idxs = np.where(above)[0]
            if len(idxs) < 2:
                fwhms.append(0.0)
            else:
                fwhms.append(float(idxs[-1] - idxs[0] + 1))
        return tuple(fwhms)

    def clear(self) -> None:
        self._profile_h = None
        self._profile_v = None
        self._fwhm_h = None
        self._fwhm_v = None
        self._hover_rgb = None
        self._hover_pos = None
        self.update()

    def mouseMoveEvent(self, event) -> None:
        self._hover_pos = (event.x(), event.y())
        self._hover_rgb = None
        self._hover_label = ""

        if self._profile_h is None and self._profile_v is None:
            self.update()
            return

        w, h = self.width(), self.height()
        half_h = h // 2
        mx, my = event.x(), event.y()

        # 检查是否在剖面图区域内
        for profile, top, title, is_vertical in [
            (self._profile_v, 0, "竖线", True),
            (self._profile_h, half_h, "横线", False),
        ]:
            if profile is None or len(profile) < 2:
                continue
            margin = 35
            plot_w = w - 2 * margin
            plot_h = half_h - 2 * margin
            px, py = margin, top + margin

            if px <= mx <= px + plot_w and py <= my <= py + plot_h:
                n = len(profile)
                idx = int((mx - px) / plot_w * max(1, n - 1))
                idx = max(0, min(n - 1, idx))
                r, g, b = profile[idx]
                self._hover_rgb = (int(r), int(g), int(b))
                # 计算实际像素坐标：(x, y) 基于十字线交叉点
                if is_vertical:
                    px_coord = self._cross_hair_x  # 竖线：x 固定，y 变化
                    py_coord = idx
                else:
                    px_coord = idx  # 横线：y 固定，x 变化
                    py_coord = self._cross_hair_y
                self._hover_label = f"{title} x={px_coord} y={py_coord} R={int(r)} G={int(g)} B={int(b)}"
                break

        self.update()

    def leaveEvent(self, event) -> None:
        self._hover_rgb = None
        self._hover_pos = None
        self._hover_label = ""
        self.update()

    def paintEvent(self, event) -> None:
        theme = self._theme
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        painter.fillRect(self.rect(), QColor(theme.plot_bg))

        if self._profile_h is None and self._profile_v is None:
            painter.setPen(QColor(theme.text_disabled))
            painter.setFont(QFont("Microsoft YaHei", 11))
            painter.drawText(self.rect(), Qt.AlignCenter, "RGB 剖面图\n拖动预览中的十字线查看剖面")
            painter.end()
            return

        half_h = h // 2
        self._draw_profile(painter, self._profile_h, self._fwhm_h, 0, half_h, w, half_h, "横线 (水平) 剖面")
        self._draw_profile(painter, self._profile_v, self._fwhm_v, 0, 0, w, half_h, "竖线 (垂直) 剖面")

        # 鼠标悬停 tooltip
        if self._hover_rgb is not None and self._hover_pos is not None:
            mx, my = self._hover_pos
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(0, 0, 0, 180))
            painter.setFont(QFont("Consolas", 8))
            fm = painter.fontMetrics()
            text_w = fm.horizontalAdvance(self._hover_label) + 12
            tx, ty = mx + 12, my - 24
            if tx + text_w > w:
                tx = mx - text_w - 12
            if ty < 0:
                ty = my + 4
            painter.drawRoundedRect(tx, ty, text_w, 22, 4, 4)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(tx + 6, ty + 15, self._hover_label)

        painter.end()

    def _draw_profile(
        self,
        painter: QPainter,
        profile: Optional[List[Tuple[float, float, float]]],
        fwhm: Optional[Tuple[float, float, float]],
        left: int, top: int, total_w: int, total_h: int,
        title: str,
    ) -> None:
        if profile is None or len(profile) < 2:
            return

        theme = self._theme
        margin = 35
        plot_w = total_w - 2 * margin
        plot_h = total_h - 2 * margin
        px, py = left + margin, top + margin

        # 背景和网格
        painter.setPen(QPen(QColor(theme.plot_grid), 1, Qt.DotLine))
        for i in range(1, 5):
            y = py + int(plot_h * i / 4)
            painter.drawLine(px, y, px + plot_w, y)

        # 坐标轴
        painter.setPen(QPen(QColor(theme.plot_text), 1))
        painter.drawLine(px, py + plot_h, px + plot_w, py + plot_h)
        painter.drawLine(px, py, px, py + plot_h)

        # 标题
        painter.setFont(QFont("Microsoft YaHei", 8, QFont.Bold))
        painter.drawText(px, py - 10, title)

        n = len(profile)
        r_vals = [p[0] for p in profile]
        g_vals = [p[1] for p in profile]
        b_vals = [p[2] for p in profile]
        all_vals = r_vals + g_vals + b_vals
        v_min = float(np.min(all_vals))
        v_max = float(np.max(all_vals))
        if v_max - v_min < 1e-6:
            v_max = v_min + 1.0

        def _to_y(v: float) -> int:
            return py + plot_h - int((v - v_min) / (v_max - v_min) * plot_h)

        def _to_x(i: int) -> int:
            return px + int(i / max(1, n - 1) * plot_w)

        # 绘制 R/G/B 三条曲线
        colors = [
            (r_vals, QColor(255, 60, 60), "R"),
            (g_vals, QColor(60, 180, 60), "G"),
            (b_vals, QColor(60, 120, 255), "B"),
        ]
        for vals, color, label in colors:
            if len(vals) < 2:
                continue
            pen = QPen(color, 1.5)
            painter.setPen(pen)
            for i in range(len(vals) - 1):
                x1, y1 = _to_x(i), _to_y(vals[i])
                x2, y2 = _to_x(i + 1), _to_y(vals[i + 1])
                painter.drawLine(x1, y1, x2, y2)

            # 图例
            painter.setPen(color)
            painter.setFont(QFont("Consolas", 7))
            painter.drawText(px + plot_w - 60, py + 10 + colors.index((vals, color, label)) * 14, label)

        # FWHM 标注
        if fwhm is not None:
            painter.setPen(QColor(theme.plot_text))
            painter.setFont(QFont("Consolas", 7))
            fwhm_text = f"FWHM: R={fwhm[0]:.1f} G={fwhm[1]:.1f} B={fwhm[2]:.1f} px"
            painter.drawText(px + 4, py + 12, fwhm_text)

        # 刻度标签
        painter.setPen(QColor(theme.plot_text))
        painter.setFont(QFont("Consolas", 7))
        painter.drawText(px - 30, py + 7, f"{v_max:.0f}")
        painter.drawText(px - 30, py + plot_h, f"{v_min:.0f}")
        painter.drawText(px, py + plot_h + 16, "0")
        painter.drawText(px + plot_w - 16, py + plot_h + 16, f"{n}")


class LoopSummaryDialog(QDialog):
    """
    闭环结束后弹出的 2x2 图像变化回顾弹窗。

    从采集到的快照中自动挑选：
      - 基准图（若有）
      - 循环早期
      - 循环中期
      - 循环末期
    以 2x2 网格展示，帮助人眼快速回顾聚焦变化过程。
    """

    def __init__(
        self,
        snapshots: List[Tuple[int, np.ndarray]],
        reference_image: Optional[np.ndarray] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("闭环聚焦变化回顾")
        self.resize(900, 700)
        self._theme = get_theme("light")
        self._build_ui(snapshots, reference_image)

    def _build_ui(
        self,
        snapshots: List[Tuple[int, np.ndarray]],
        reference_image: Optional[np.ndarray],
    ) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # 标题
        title = QLabel("2x2 聚焦变化过程")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            f"font-size: 16px; font-weight: bold; color: {self._theme.text_primary};"
        )
        layout.addWidget(title)

        # 2x2 图像网格
        grid = QGridLayout()
        grid.setSpacing(12)
        images = self._select_images(snapshots, reference_image)
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for (cycle, img, label), (row, col) in zip(images, positions):
            card = self._build_image_card(img, label, cycle)
            grid.addWidget(card, row, col)
        layout.addLayout(grid, 1)

        # 关闭按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("关闭")
        close_btn.setMinimumWidth(100)
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def _select_images(
        self,
        snapshots: List[Tuple[int, np.ndarray]],
        reference_image: Optional[np.ndarray],
    ) -> List[Tuple[int, np.ndarray, str]]:
        """挑选 4 张最具代表性的图像：基准、早期、中期、末期。"""
        result: List[Tuple[int, np.ndarray, str]] = []
        has_ref = reference_image is not None
        if has_ref:
            result.append((0, reference_image, "基准图"))

        if not snapshots:
            # 没有快照时，用基准图占位
            while len(result) < 4 and has_ref:
                result.append((0, reference_image, "基准图"))
            return result

        # 按轮次排序并去重（保留每个轮次第一次出现的图像）
        seen_cycles: set = set()
        ordered: List[Tuple[int, np.ndarray]] = []
        for cycle, img in snapshots:
            if cycle not in seen_cycles:
                ordered.append((cycle, img))
                seen_cycles.add(cycle)
        ordered.sort(key=lambda x: x[0])

        need = 4 - len(result)
        n = len(ordered)
        if need <= 0:
            return result[:4]

        if n == 1:
            picks = [ordered[0]]
        elif n == 2:
            picks = [ordered[0], ordered[-1]]
        elif n == 3:
            if need == 1:
                picks = [ordered[-1]]
            elif need == 2:
                picks = [ordered[0], ordered[-1]]
            else:
                picks = [ordered[0], ordered[1], ordered[-1]]
        else:
            if need == 1:
                picks = [ordered[-1]]
            elif need == 2:
                picks = [ordered[0], ordered[-1]]
            elif need == 3:
                picks = [ordered[0], ordered[n // 2], ordered[-1]]
            else:
                # 无基准图时：早期、1/3、2/3、末期
                picks = [
                    ordered[0],
                    ordered[n // 3],
                    ordered[2 * n // 3],
                    ordered[-1],
                ]

        labels = ["早期", "过程 1", "过程 2", "末期"]
        for i, (cycle, img) in enumerate(picks[:need]):
            result.append((cycle, img, labels[i]))

        # 兜底补齐 4 张
        while len(result) < 4 and ordered:
            cycle, img = ordered[-1]
            result.append((cycle, img, "末期"))
        return result[:4]

    def _build_image_card(
        self,
        image: np.ndarray,
        label: str,
        cycle: int,
    ) -> QWidget:
        """构建单张图像卡片（含标签）。"""
        card = QWidget()
        card.setStyleSheet(
            f"background-color: {self._theme.bg_secondary}; "
            f"border: 1px solid {self._theme.border}; border-radius: 4px;"
        )
        layout = QVBoxLayout(card)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title = QLabel(f"{label}  第 {cycle} 轮" if cycle > 0 else label)
        title.setStyleSheet(
            f"color: {self._theme.text_primary}; font-weight: bold; border: none;"
        )
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none; background-color: transparent;")
        preview = RoiPreviewLabel()
        preview.setAlignment(Qt.AlignCenter)
        preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        preview.setMinimumSize(280, 220)
        preview.set_preview_image(image)
        scroll.setWidget(preview)
        layout.addWidget(scroll, 1)
        return card
