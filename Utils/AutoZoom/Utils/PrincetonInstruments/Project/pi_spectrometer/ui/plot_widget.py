"""光谱实时绘图控件。"""

from typing import Optional

import numpy as np

from pi_spectrometer.ui.qt_compat import QtWidgets, AlignCenter


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
