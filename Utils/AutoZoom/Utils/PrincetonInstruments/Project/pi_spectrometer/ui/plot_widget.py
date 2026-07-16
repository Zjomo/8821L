"""光谱实时绘图控件。

在 pyqtgraph 实时曲线基础上增加：
- 双光标（可拖拽）
- 峰值标注
- 对数/线性 Y 轴切换
- 历史轨迹叠加
"""

from typing import List, Optional, Tuple

import numpy as np

from pi_spectrometer.ui.qt_compat import QtCore, QtWidgets, AlignCenter


try:
    import pyqtgraph as pg
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False


class PlotWidget(QtWidgets.QWidget):
    """封装 pyqtgraph 的光谱绘图控件。"""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._setup_ui()
        self._x_data: Optional[np.ndarray] = None
        self._y_data: Optional[np.ndarray] = None
        self._history_items: List[pg.PlotDataItem] = []
        self._cursor_lines: List[pg.InfiniteLine] = []
        self._peak_items: List = []

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if PYQTGRAPH_AVAILABLE:
            self.plot_widget = pg.PlotWidget()
            self.plot_widget.setLabel("bottom", "Wavelength", units="nm")
            self.plot_widget.setLabel("left", "Intensity")
            self.plot_widget.showGrid(x=True, y=True)
            self.plot_item = self.plot_widget.plot(pen="y")
            layout.addWidget(self.plot_widget)
        else:
            self.plot_widget = QtWidgets.QLabel("pyqtgraph 未安装，无法显示实时曲线")
            self.plot_widget.setAlignment(AlignCenter)
            layout.addWidget(self.plot_widget)
            self.plot_item = None

        self.setLayout(layout)

    def update_plot(
        self,
        y: np.ndarray,
        x: Optional[np.ndarray] = None,
        title: Optional[str] = None,
    ) -> None:
        """更新光谱曲线。"""
        if self.plot_item is None:
            return

        if x is None:
            x = np.arange(len(y), dtype=np.float64)

        self._x_data = np.asarray(x, dtype=np.float64)
        self._y_data = np.asarray(y, dtype=np.float64)

        self.plot_item.setData(self._x_data, self._y_data)

        if title:
            self.plot_widget.setTitle(title)

    def clear_plot(self) -> None:
        """清空曲线。"""
        if self.plot_item is not None:
            self.plot_item.setData([], [])

    # ------------------------------------------------------------------
    # 光标
    # ------------------------------------------------------------------
    def add_cursor(self, x_pos: Optional[float] = None, color: str = "c") -> Optional[pg.InfiniteLine]:
        """添加一条可拖拽的垂直光标。"""
        if not PYQTGRAPH_AVAILABLE or self.plot_widget is None:
            return None
        if x_pos is None and self._x_data is not None and self._x_data.size:
            x_pos = float(np.mean(self._x_data))
        elif x_pos is None:
            x_pos = 0.0
        line = pg.InfiniteLine(pos=x_pos, angle=90, movable=True, pen=color)
        self.plot_widget.addItem(line)
        self._cursor_lines.append(line)
        return line

    def clear_cursors(self) -> None:
        """移除所有光标。"""
        if not PYQTGRAPH_AVAILABLE:
            return
        for line in self._cursor_lines:
            self.plot_widget.removeItem(line)
        self._cursor_lines.clear()

    def get_cursor_positions(self) -> List[float]:
        """返回所有光标的当前 x 位置。"""
        return [float(line.value()) for line in self._cursor_lines]

    # ------------------------------------------------------------------
    # 峰值标注
    # ------------------------------------------------------------------
    def annotate_peak(
        self,
        x: float,
        y: float,
        text: str,
        color: str = "g",
    ) -> Optional:
        """在谱线上添加峰值标注。"""
        if not PYQTGRAPH_AVAILABLE or self.plot_widget is None:
            return None
        scatter = pg.ScatterPlotItem(
            x=[x],
            y=[y],
            symbol="o",
            size=10,
            brush=color,
            pen=color,
        )
        label = pg.TextItem(text, anchor=(0.5, 1), color=color)
        label.setPos(x, y)
        self.plot_widget.addItem(scatter)
        self.plot_widget.addItem(label)
        self._peak_items.extend([scatter, label])
        return scatter

    def annotate_fwhm(
        self,
        left_x: float,
        right_x: float,
        y_level: float,
        color: str = "r",
    ) -> Optional:
        """标注 FWHM 区间。"""
        if not PYQTGRAPH_AVAILABLE or self.plot_widget is None:
            return None
        line = pg.PlotDataItem(
            x=[left_x, right_x],
            y=[y_level, y_level],
            pen=pg.mkPen(color=color, width=2, style=QtCore.Qt.DotLine),
        )
        self.plot_widget.addItem(line)
        self._peak_items.append(line)
        return line

    def clear_peak_annotations(self) -> None:
        """清除所有峰值标注。"""
        if not PYQTGRAPH_AVAILABLE:
            return
        for item in self._peak_items:
            self.plot_widget.removeItem(item)
        self._peak_items.clear()

    # ------------------------------------------------------------------
    # 对数/线性 Y 轴
    # ------------------------------------------------------------------
    def set_log_mode(self, log_y: bool = True) -> None:
        """切换 Y 轴对数/线性模式。"""
        if not PYQTGRAPH_AVAILABLE or self.plot_widget is None:
            return
        self.plot_widget.setLogMode(y=log_y)

    def set_linear_mode(self) -> None:
        """Y 轴切换为线性模式。"""
        self.set_log_mode(log_y=False)

    # ------------------------------------------------------------------
    # 历史轨迹
    # ------------------------------------------------------------------
    def add_history_trace(
        self,
        y: np.ndarray,
        x: Optional[np.ndarray] = None,
        max_history: int = 10,
        alpha: float = 0.4,
    ) -> None:
        """将当前曲线以半透明方式加入历史叠加。"""
        if not PYQTGRAPH_AVAILABLE or self.plot_widget is None:
            return
        if x is None:
            x = np.arange(len(y), dtype=np.float64)
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

        pen = pg.mkPen(color=(200, 200, 200, int(255 * alpha)), width=1)
        item = pg.PlotDataItem(x_arr, y_arr, pen=pen)
        self.plot_widget.addItem(item)
        self._history_items.append(item)

        while len(self._history_items) > max_history:
            old = self._history_items.pop(0)
            self.plot_widget.removeItem(old)

    def clear_history(self) -> None:
        """清空历史轨迹。"""
        if not PYQTGRAPH_AVAILABLE:
            return
        for item in self._history_items:
            self.plot_widget.removeItem(item)
        self._history_items.clear()

    def get_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """返回当前绘制的 (x, y) 数据。"""
        return self._x_data, self._y_data
