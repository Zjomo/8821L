"""
autofocus_qt_ui 主程序。

基于 PyQt 的 AutoZoom/Focus 自动聚焦界面：
  - 采集源选择（屏幕/窗口/USB/模拟）
  - ROI 实时预览与框选
  - 14 项聚焦指标显示
  - FocusScore_ratio 趋势图
  - 参考基线建立
  - 自动补焦闭环运行
  - 运行日志
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# ============================================================
# 预加载 llvmlite：必须在 PyQt6 之前加载，避免 Qt 修改 DLL 搜索路径
# 后导致 llvmlite.dll 加载失败 (WinError 1114)。
# 即使加载失败也不影响程序启动（仅 Z 轴电机控制不可用）。
# ============================================================
try:
    import llvmlite.binding  # noqa: F401
except Exception:
    pass

import numpy as np

from .qt_compat import (
    Qt,
    Signal,
    Slot,
    QAction,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTimer,
    QVBoxLayout,
    QWidget,
    app_exec,
)

# QtCore / QtGui 用于图像操作
from .qt_compat import QtCore, QtGui

# 将 AutoZoom 目录加入 sys.path，确保能导入 Focus 模块
AUTOZOOM_ROOT = Path(__file__).resolve().parent.parent
if str(AUTOZOOM_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOZOOM_ROOT))

from Focus.config import AutofocusConfig
from Focus.metrics import FocusMetricsCalculator
from Focus.scorer import FocusScorer
from Focus.z_axis import ZAxisController
from Focus.controller import AutofocusController
from Focus.simulator import FocusSimulator, create_demo_environment

from .config import (
    DEFAULT_CAPTURE_AREA,
    DEFAULT_CAPTURE_MODE,
    DEFAULT_CYCLES,
    DEFAULT_FOCUS_ROI,
    DEFAULT_INTERVAL_S,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEARCH_STRATEGY,
    DEFAULT_TRIGGER_COUNT,
    DEFAULT_TRIGGER_RATIO,
    DEFAULT_WINDOW_TITLE,
    DEFAULT_Z_AXIS,
    DEFAULT_Z_ENABLED,
)
from .detectors import detect_windows, detect_usb_cameras
from .widgets import FocusScorePlot, MetricsTable, RoiPreviewLabel, RgbProfilePlot
from .worker import AutofocusThread
from .themes import Theme, get_theme, available_themes

logger = logging.getLogger("autofocus_qt_ui.app")


@dataclass
class UiRuntimeArgs:
    """从 UI 收集的运行参数。"""

    output: str = DEFAULT_OUTPUT_DIR
    capture_mode: str = DEFAULT_CAPTURE_MODE
    window_title: str = DEFAULT_WINDOW_TITLE
    usb_device: int = 0
    capture_area: tuple = DEFAULT_CAPTURE_AREA
    focus_roi: tuple = DEFAULT_FOCUS_ROI
    cycles: int = DEFAULT_CYCLES
    interval: float = DEFAULT_INTERVAL_S
    z_enabled: bool = DEFAULT_Z_ENABLED
    z_axis: int = DEFAULT_Z_AXIS
    z_picomotor_conn: int = 0
    z_picomotor_backend: str = "auto"
    autofocus_enabled: bool = True
    trigger_ratio: float = DEFAULT_TRIGGER_RATIO
    stop_ratio: float = 0.95
    trigger_count: int = DEFAULT_TRIGGER_COUNT
    search_strategy: str = DEFAULT_SEARCH_STRATEGY
    demo_sim: bool = False


class AutofocusMainWindow(QMainWindow):
    """自动聚焦主窗口。"""

    def __init__(self) -> None:
        super().__init__()
        self._theme = get_theme("light")
        self.setWindowTitle("AutoZoom 自动聚焦控制台")
        self.resize(1400, 900)

        self._thread: Optional[AutofocusThread] = None
        self._controller: Optional[AutofocusController] = None
        self._simulator: Optional[FocusSimulator] = None
        self._reference_data: Optional[Dict[str, Any]] = None  # 建立参考后保存，供闭环使用

        # 视频播放器状态
        self._video_capture: Optional[Any] = None  # cv2.VideoCapture
        self._video_timer: Optional[QTimer] = None
        self._screen_timer: Optional[QTimer] = None  # 屏幕区域实时采集定时器
        self._video_playing: bool = False
        self._video_fps: float = 30.0
        self._video_frame_count: int = 0
        self._video_current_frame: int = 0
        self._video_slider_dragging: bool = False

        # 光斑跟踪状态
        self._tracked_spot: Optional[tuple] = None  # 正在跟踪的光斑中心 (x, y)

        self._build_menu()
        self._build_central_widget()
        self._connect_signals()
        self._apply_default_values()
        self.apply_theme(self._theme)

    def apply_theme(self, theme: Theme) -> None:
        """应用主题到主窗口及所有自定义控件。"""
        self._theme = theme
        self.setStyleSheet(theme.qss())
        if hasattr(self, "preview_label"):
            self.preview_label.apply_theme(theme)
        if hasattr(self, "metrics_table"):
            self.metrics_table.apply_theme(theme)
        if hasattr(self, "score_plot"):
            self.score_plot.apply_theme(theme)
        if hasattr(self, "profile_plot"):
            self.profile_plot.apply_theme(theme)
        self.log(f"已切换主题：{theme.display_name}")

    def _build_menu(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("文件")
        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        theme_menu = menubar.addMenu("主题")
        for name, display_name in available_themes().items():
            action = QAction(display_name, self)
            action.setCheckable(True)
            action.setChecked(name == self._theme.name)
            action.triggered.connect(lambda checked=False, n=name: self._switch_theme(n))
            theme_menu.addAction(action)

    def _switch_theme(self, name: str) -> None:
        """切换主题并更新菜单勾选状态。"""
        theme = get_theme(name)
        self.apply_theme(theme)
        # 更新菜单勾选
        for action in self.menuBar().actions():
            if action.text() == "主题":
                for sub in action.menu().actions():
                    sub.setChecked(sub.text() == theme.display_name)
                break

    def _build_central_widget(self) -> QWidget:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(12, 12, 12, 12)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        left_widget = self._build_left_panel()
        splitter.addWidget(left_widget)

        right_widget = self._build_right_panel()
        # 右侧容器也添加滚动条，防止内容超出屏幕
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setWidget(right_widget)
        splitter.addWidget(right_scroll)

        splitter.setSizes([420, 980])
        return central

    def _build_left_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(10)
        layout.setContentsMargins(6, 6, 6, 6)

        title = QLabel("AutoZoom 自动聚焦控制台")
        title.setObjectName("title")
        layout.addWidget(title)

        self.mode_group = QGroupBox("运行模式")
        mode_layout = QFormLayout(self.mode_group)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["屏幕区域", "窗口 ROI", "USB 相机", "本地视频文件"])
        mode_layout.addRow("采集源:", self.mode_combo)

        # 动态资源选择下拉框（窗口 / USB 相机）
        self.resource_combo = QComboBox()
        self.resource_combo.setMinimumWidth(200)
        mode_layout.addRow("设备选择:", self.resource_combo)

        self.window_title_edit = QLineEdit(DEFAULT_WINDOW_TITLE)
        mode_layout.addRow("窗口标题:", self.window_title_edit)

        self.usb_device_spin = QSpinBox()
        self.usb_device_spin.setRange(0, 9)
        mode_layout.addRow("USB 设备索引:", self.usb_device_spin)

        self.video_path_edit = QLineEdit()
        self.video_path_edit.setPlaceholderText("选择本地视频文件...")
        self.video_browse_btn = QPushButton("浏览...")
        video_h = QHBoxLayout()
        video_h.addWidget(self.video_path_edit)
        video_h.addWidget(self.video_browse_btn)
        mode_layout.addRow("视频文件:", video_h)

        self.sim_params_btn = QPushButton("模拟参数...")
        self.sim_params_btn.setToolTip("配置峰值位置、模糊系数、漂移等")
        mode_layout.addRow("", self.sim_params_btn)

        layout.addWidget(self.mode_group)

        self.roi_group = QGroupBox("采集区域")
        roi_layout = QFormLayout(self.roi_group)
        self.capture_area_edit = QLineEdit(
            ", ".join(str(v) for v in DEFAULT_CAPTURE_AREA)
        )
        roi_layout.addRow("截图区域 (L,T,W,H):", self.capture_area_edit)
        self.focus_roi_edit = QLineEdit(
            ", ".join(str(v) for v in DEFAULT_FOCUS_ROI)
        )
        roi_layout.addRow("ROI (x,y,w,h):", self.focus_roi_edit)
        self.load_preview_btn = QPushButton("加载预览")
        roi_layout.addRow("", self.load_preview_btn)

        # 剖面线显示开关
        self.show_v_line_check = QCheckBox("显示竖线剖面")
        self.show_v_line_check.setChecked(True)
        self.show_h_line_check = QCheckBox("显示横线剖面")
        self.show_h_line_check.setChecked(True)
        roi_layout.addRow("", self.show_v_line_check)
        roi_layout.addRow("", self.show_h_line_check)

        # 光斑检测按钮
        self.detect_spots_btn = QPushButton("自动检测光斑")
        roi_layout.addRow("", self.detect_spots_btn)

        layout.addWidget(self.roi_group)

        # 光斑检测配置模块
        self.spot_group = QGroupBox("光斑检测")
        spot_layout = QFormLayout(self.spot_group)

        self.spot_algo_combo = QComboBox()
        self.spot_algo_combo.addItems([
            "bright_blob (亮斑检测)",
            "otsu_threshold (大津阈值)",
            "gradient_peak (梯度峰值)",
            "contour (轮廓检测)",
        ])
        spot_layout.addRow("检测算法:", self.spot_algo_combo)

        self.spot_threshold_spin = QSpinBox()
        self.spot_threshold_spin.setRange(10, 255)
        self.spot_threshold_spin.setValue(30)
        self.spot_threshold_spin.setToolTip("最小阈值（用于二值化）")
        spot_layout.addRow("最小阈值:", self.spot_threshold_spin)

        self.spot_min_area_spin = QSpinBox()
        self.spot_min_area_spin.setRange(5, 10000)
        self.spot_min_area_spin.setValue(20)
        self.spot_min_area_spin.setToolTip("光斑最小面积 (px)")
        spot_layout.addRow("最小面积:", self.spot_min_area_spin)

        self.spot_max_area_spin = QSpinBox()
        self.spot_max_area_spin.setRange(100, 1000000)
        self.spot_max_area_spin.setValue(100000)
        self.spot_max_area_spin.setToolTip("光斑最大面积 (px)")
        spot_layout.addRow("最大面积:", self.spot_max_area_spin)

        self.spot_blur_spin = QSpinBox()
        self.spot_blur_spin.setRange(1, 31)
        self.spot_blur_spin.setValue(5)
        self.spot_blur_spin.setSingleStep(2)
        self.spot_blur_spin.setToolTip("高斯模糊内核大小（奇数）")
        spot_layout.addRow("模糊核:", self.spot_blur_spin)

        layout.addWidget(self.spot_group)

        self.focus_group = QGroupBox("补焦参数")
        focus_layout = QFormLayout(self.focus_group)

        self.autofocus_check = QCheckBox("启用自动补焦")
        self.autofocus_check.setChecked(True)
        focus_layout.addRow("", self.autofocus_check)

        self.z_enabled_check = QCheckBox("启用 Z 轴硬件")
        self.z_enabled_check.setChecked(DEFAULT_Z_ENABLED)
        focus_layout.addRow("", self.z_enabled_check)

        self.trigger_ratio_spin = QDoubleSpinBox()
        self.trigger_ratio_spin.setRange(0.1, 1.0)
        self.trigger_ratio_spin.setSingleStep(0.05)
        self.trigger_ratio_spin.setValue(DEFAULT_TRIGGER_RATIO)
        focus_layout.addRow("触发阈值:", self.trigger_ratio_spin)

        self.stop_ratio_spin = QDoubleSpinBox()
        self.stop_ratio_spin.setRange(0.1, 1.0)
        self.stop_ratio_spin.setSingleStep(0.05)
        self.stop_ratio_spin.setValue(0.95)
        focus_layout.addRow("目标阈值:", self.stop_ratio_spin)

        self.trigger_count_spin = QSpinBox()
        self.trigger_count_spin.setRange(1, 20)
        self.trigger_count_spin.setValue(DEFAULT_TRIGGER_COUNT)
        focus_layout.addRow("连续触发次数:", self.trigger_count_spin)

        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["hill_climb", "full_sweep", "curve_fit", "golden_section"])
        focus_layout.addRow("搜索策略:", self.strategy_combo)

        self.z_axis_spin = QSpinBox()
        self.z_axis_spin.setRange(1, 4)
        self.z_axis_spin.setValue(DEFAULT_Z_AXIS)
        focus_layout.addRow("Z 轴号:", self.z_axis_spin)

        self.z_picomotor_conn_spin = QSpinBox()
        self.z_picomotor_conn_spin.setRange(0, 31)
        self.z_picomotor_conn_spin.setValue(0)
        self.z_picomotor_conn_spin.setToolTip(
            "Newport 8742 控制器索引（conn），0 表示第一个 USB 设备。"
            "若连接报错请尝试切换此值。"
        )
        focus_layout.addRow("Z 控制器索引:", self.z_picomotor_conn_spin)

        self.z_picomotor_backend_combo = QComboBox()
        self.z_picomotor_backend_combo.addItems(["auto", "pyusb", "serial"])
        self.z_picomotor_backend_combo.setToolTip(
            "pylablib 连接后端，auto 自动选择。"
        )
        focus_layout.addRow("Z 控制器后端:", self.z_picomotor_backend_combo)

        layout.addWidget(self.focus_group)

        self.loop_group = QGroupBox("循环参数")
        loop_layout = QFormLayout(self.loop_group)
        self.cycles_spin = QSpinBox()
        self.cycles_spin.setRange(0, 10000)
        self.cycles_spin.setValue(DEFAULT_CYCLES)
        self.cycles_spin.setToolTip("0 表示无限循环")
        loop_layout.addRow("循环轮数:", self.cycles_spin)

        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setRange(0.0, 3600.0)
        self.interval_spin.setSingleStep(0.5)
        self.interval_spin.setValue(DEFAULT_INTERVAL_S)
        loop_layout.addRow("间隔 (s):", self.interval_spin)

        self.output_edit = QLineEdit(DEFAULT_OUTPUT_DIR)
        self.output_btn = QPushButton("浏览...")
        h = QHBoxLayout()
        h.addWidget(self.output_edit)
        h.addWidget(self.output_btn)
        loop_layout.addRow("输出目录:", h)

        layout.addWidget(self.loop_group)

        btn_layout = QHBoxLayout()
        self.build_ref_btn = QPushButton("建立参考")
        self.build_ref_btn.setObjectName("primary")
        self.build_ref_btn.setToolTip(
            "抓取当前画面作为清晰聚焦的参考基准。\n"
            "程序会多次采集并计算最优聚焦指标，\n"
            "作为后续自动聚焦的对比目标。"
        )
        self.start_btn = QPushButton("启动闭环")
        self.start_btn.setObjectName("primary")
        self.start_btn.setToolTip(
            "开始自动聚焦循环：\n"
            "每轮自动截图 → 对比参考基准 →\n"
            "若偏离则自动调节 Z 轴恢复清晰聚焦。"
        )
        self.pause_btn = QPushButton("暂停")
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setObjectName("danger")
        btn_layout.addWidget(self.build_ref_btn)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.pause_btn)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)

        self.status_label = QLabel("就绪")
        self.status_label.setObjectName("status")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        layout.addStretch()
        scroll.setWidget(container)
        return scroll

    def _build_right_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # 使用 QSplitter 允许用户拖动边界调整各区域大小
        right_splitter = QSplitter(Qt.Vertical)

        # ===== 1. 实时预览（独立于 Tab 之上） =====
        preview_group = QGroupBox("实时预览")
        preview_group_layout = QVBoxLayout(preview_group)
        preview_group_layout.setContentsMargins(4, 8, 4, 4)

        self.preview_label = RoiPreviewLabel()
        self.preview_label.set_reference_mode(True)
        preview_group_layout.addWidget(self.preview_label)

        # 预览工具按钮行
        preview_btn_layout = QHBoxLayout()
        self.set_roi_btn = QPushButton("应用 ROI")
        self.clear_roi_btn = QPushButton("清除 ROI")
        self.save_frame_btn = QPushButton("保存当前帧")
        preview_btn_layout.addWidget(self.set_roi_btn)
        preview_btn_layout.addWidget(self.clear_roi_btn)
        preview_btn_layout.addStretch()
        # 预览缩放比例
        preview_btn_layout.addWidget(QLabel("缩放:"))
        self.zoom_combo = QComboBox()
        self.zoom_combo.addItems(["25%", "50%", "75%", "100%", "150%", "200%"])
        self.zoom_combo.setCurrentText("100%")
        self.zoom_combo.setFixedWidth(70)
        preview_btn_layout.addWidget(self.zoom_combo)
        preview_btn_layout.addWidget(self.save_frame_btn)
        preview_group_layout.addLayout(preview_btn_layout)

        # 视频播放器控件
        video_controls = QHBoxLayout()
        self.video_play_btn = QPushButton("▶ 播放")
        self.video_play_btn.setFixedWidth(70)
        self.video_pause_btn = QPushButton("⏸ 暂停")
        self.video_pause_btn.setFixedWidth(70)
        self.video_stop_btn = QPushButton("⏹ 停止")
        self.video_stop_btn.setFixedWidth(70)
        self.video_slider = QSlider(Qt.Horizontal)
        self.video_slider.setRange(0, 1000)
        self.video_slider.setValue(0)
        self.video_slider.setToolTip("视频进度")
        self.video_time_label = QLabel("00:00 / 00:00")
        self.video_time_label.setFixedWidth(100)
        self.video_time_label.setAlignment(Qt.AlignCenter)

        video_controls.addWidget(self.video_play_btn)
        video_controls.addWidget(self.video_pause_btn)
        video_controls.addWidget(self.video_stop_btn)
        video_controls.addWidget(self.video_slider)
        video_controls.addWidget(self.video_time_label)
        preview_group_layout.addLayout(video_controls)

        # 视频信息标签
        self.video_info_label = QLabel("")
        self.video_info_label.setAlignment(Qt.AlignCenter)
        self.video_info_label.setStyleSheet("font-size: 11px; color: #888;")
        preview_group_layout.addWidget(self.video_info_label)

        right_splitter.addWidget(preview_group)

        # ===== 2. 分析 Tab（聚焦指标 / FocusScore 曲线 / 剖面图） =====
        self.tabs = QTabWidget()

        metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(metrics_tab)
        self.metrics_table = MetricsTable()
        metrics_layout.addWidget(self.metrics_table)
        self.tabs.addTab(metrics_tab, "聚焦指标")

        plot_tab = QWidget()
        plot_layout = QVBoxLayout(plot_tab)
        self.score_plot = FocusScorePlot()
        plot_layout.addWidget(self.score_plot)
        self.tabs.addTab(plot_tab, "FocusScore 曲线")

        profile_tab = QWidget()
        profile_layout = QVBoxLayout(profile_tab)
        self.profile_plot = RgbProfilePlot()
        profile_layout.addWidget(self.profile_plot)
        self.tabs.addTab(profile_tab, "剖面图")

        right_splitter.addWidget(self.tabs)

        # ===== 3. 日志（可折叠 GroupBox） =====
        self.log_group = QGroupBox("运行日志")
        self.log_group.setCheckable(True)
        self.log_group.setChecked(True)
        log_group_layout = QVBoxLayout(self.log_group)
        log_group_layout.setContentsMargins(4, 8, 4, 4)

        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumBlockCount(500)
        log_group_layout.addWidget(self.log_edit)

        right_splitter.addWidget(self.log_group)

        # 设置初始比例：预览 40%, 分析 40%, 日志 20%
        right_splitter.setSizes([360, 360, 180])
        right_splitter.setChildrenCollapsible(False)

        layout.addWidget(right_splitter)

        return widget

    def _connect_signals(self) -> None:
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.load_preview_btn.clicked.connect(self._load_preview)
        self.set_roi_btn.clicked.connect(self._apply_roi_from_preview)
        self.clear_roi_btn.clicked.connect(self._clear_roi)
        self.save_frame_btn.clicked.connect(self._save_current_frame)
        self.output_btn.clicked.connect(self._choose_output_dir)
        self.video_browse_btn.clicked.connect(self._choose_video_file)

        # 视频播放器
        self.video_play_btn.clicked.connect(self._on_video_play)
        self.video_pause_btn.clicked.connect(self._on_video_pause)
        self.video_stop_btn.clicked.connect(self._on_video_stop)
        self.video_slider.sliderPressed.connect(self._on_video_slider_pressed)
        self.video_slider.sliderReleased.connect(self._on_video_slider_released)
        self.video_slider.valueChanged.connect(self._on_video_slider_changed)

        self.build_ref_btn.clicked.connect(self._build_reference)
        self.start_btn.clicked.connect(self._start_loop)
        self.pause_btn.clicked.connect(self._pause_loop)
        self.stop_btn.clicked.connect(self._stop_loop)

        self.preview_label.roiSelected.connect(self._on_roi_selected)
        self.preview_label.crossHairChanged.connect(self._on_cross_hair_changed)

        self.show_v_line_check.toggled.connect(self._on_show_v_line_toggled)
        self.show_h_line_check.toggled.connect(self._on_show_h_line_toggled)
        self.zoom_combo.currentTextChanged.connect(self._on_zoom_changed)
        self.detect_spots_btn.clicked.connect(self._on_detect_spots)

    def _apply_default_values(self) -> None:
        self._on_mode_changed(0)
        self.video_path_edit.setEnabled(False)
        self.video_browse_btn.setEnabled(False)
        self._set_video_controls_visible(False)

        # 屏幕模式：默认截图区域和 ROI 设为全屏分辨率
        try:
            screen = QApplication.primaryScreen()
            if screen is not None:
                size = screen.size()
                sw, sh = size.width(), size.height()
                self.capture_area_edit.setText(f"0, 0, {sw}, {sh}")
                self.focus_roi_edit.setText(f"0, 0, {sw}, {sh}")
            else:
                self.capture_area_edit.setText(", ".join(str(v) for v in DEFAULT_CAPTURE_AREA))
                self.focus_roi_edit.setText(", ".join(str(v) for v in DEFAULT_FOCUS_ROI))
        except Exception:
            self.capture_area_edit.setText(", ".join(str(v) for v in DEFAULT_CAPTURE_AREA))
            self.focus_roi_edit.setText(", ".join(str(v) for v in DEFAULT_FOCUS_ROI))

    def _collect_args(self) -> UiRuntimeArgs:
        args = UiRuntimeArgs()
        args.output = self.output_edit.text().strip() or DEFAULT_OUTPUT_DIR
        # UI 显示模式 -> Focus 模块内部 capture_mode 的映射
        ui_mode = ["screen", "window", "usb", "video"][self.mode_combo.currentIndex()]
        args.capture_mode = {
            "screen": "screen_region",
            "window": "window_roi",
            "usb": "usb_camera",
            "video": "local_video_file",
        }[ui_mode]
        args.window_title = self.window_title_edit.text().strip()
        args.usb_device = self.usb_device_spin.value()
        args.capture_area = self._parse_rect(self.capture_area_edit.text(), DEFAULT_CAPTURE_AREA)
        args.focus_roi = self._parse_rect(self.focus_roi_edit.text(), DEFAULT_FOCUS_ROI)
        args.cycles = self.cycles_spin.value()
        args.interval = self.interval_spin.value()
        args.z_enabled = self.z_enabled_check.isChecked()
        args.z_axis = self.z_axis_spin.value()
        args.z_picomotor_conn = self.z_picomotor_conn_spin.value()
        args.z_picomotor_backend = self.z_picomotor_backend_combo.currentText().strip()
        args.autofocus_enabled = self.autofocus_check.isChecked()
        args.trigger_ratio = self.trigger_ratio_spin.value()
        args.stop_ratio = self.stop_ratio_spin.value()
        args.trigger_count = self.trigger_count_spin.value()
        args.search_strategy = self.strategy_combo.currentText()
        args.demo_sim = False
        return args

    @staticmethod
    def _parse_rect(text: str, default: tuple) -> tuple:
        try:
            parts = [int(x.strip()) for x in text.split(",")]
            if len(parts) == 4:
                return tuple(parts)
        except Exception:
            pass
        return default

    def _on_mode_changed(self, index: int) -> None:
        is_screen = index == 0
        is_window = index == 1
        is_usb = index == 2
        is_video = index == 3

        # 更新设备选择下拉框
        self.resource_combo.clear()
        self.resource_combo.setEnabled(not is_screen and not is_video)

        if is_window:
            self.resource_combo.addItem("正在检测窗口...", None)
            self._refresh_windows()
        elif is_usb:
            self.resource_combo.addItem("正在检测相机...", None)
            self._refresh_cameras()
        elif is_video:
            self.resource_combo.addItem("本地视频模式", None)
        else:
            self.resource_combo.addItem("当前模式无需选择", None)

        self.window_title_edit.setEnabled(is_window)
        self.usb_device_spin.setEnabled(is_usb)
        self.video_path_edit.setEnabled(is_video)
        self.video_browse_btn.setEnabled(is_video)
        self.sim_params_btn.setVisible(False)
        self.z_enabled_check.setEnabled(not is_video)

        # 视频模式下显示播放控件，隐藏"加载预览"按钮
        self._set_video_controls_visible(is_video)
        self.load_preview_btn.setVisible(not is_video)

        # 切换非屏幕模式时，停止屏幕实时预览
        if not is_screen:
            self._stop_screen_preview()

        # 资源下拉框变更时同步到对应编辑框
        self.resource_combo.currentIndexChanged.connect(
            self._on_resource_selected, type=Qt.UniqueConnection
        )

    def _refresh_windows(self) -> None:
        try:
            windows = detect_windows()
            self.resource_combo.clear()
            if not windows:
                self.resource_combo.addItem("未检测到可见窗口", None)
                return
            for title, hwnd in windows:
                # 下拉框显示标题，数据保存 hwnd / 标题
                self.resource_combo.addItem(title, title)
            # 默认选中之前编辑框里的标题（如果存在）
            current = self.window_title_edit.text().strip()
            idx = self.resource_combo.findData(current)
            if idx >= 0:
                self.resource_combo.setCurrentIndex(idx)
            else:
                self.window_title_edit.setText(windows[0][0])
        except Exception as exc:
            self.log(f"检测窗口失败：{exc}")
            self.resource_combo.clear()
            self.resource_combo.addItem("检测失败", None)

    def _refresh_cameras(self) -> None:
        try:
            cameras = detect_usb_cameras()
            self.resource_combo.clear()
            if not cameras:
                self.resource_combo.addItem("未检测到 USB 相机", None)
                return
            for idx, desc in cameras:
                self.resource_combo.addItem(desc, idx)
            self.usb_device_spin.setValue(cameras[0][0])
        except Exception as exc:
            self.log(f"检测相机失败：{exc}")
            self.resource_combo.clear()
            self.resource_combo.addItem("检测失败", None)

    def _on_resource_selected(self, index: int) -> None:
        mode = self.mode_combo.currentIndex()
        data = self.resource_combo.itemData(index)
        if mode == 1 and isinstance(data, str):
            self.window_title_edit.setText(data)
        elif mode == 2 and isinstance(data, int):
            self.usb_device_spin.setValue(data)

    def _load_preview(self) -> None:
        """加载预览：屏幕模式启动实时采集，其他模式单帧捕获。"""
        args = self._collect_args()

        # 屏幕模式：启动/停止实时采集
        if args.capture_mode == "screen_region":
            self._toggle_screen_preview()
            return

        # 其他模式：单帧捕获
        try:
            cfg = self._build_config(args)
            metrics_calc = FocusMetricsCalculator(cfg)

            if args.capture_mode == "local_video_file":
                metrics_calc.cfg.local_video_path = self.video_path_edit.text().strip()
                image = metrics_calc._capture_local_video_frame()
            else:
                live = metrics_calc.capture_live()
                image = live.get("roi_rgb")
                if image is None:
                    image = live.get("full_rgb")

            if image is not None:
                self.preview_label.set_preview_image(image)
                self._last_preview_image = image
                self._compute_and_update_profile()
                self.log("已加载预览")
            else:
                self.log("无法获取预览图像")
        except Exception as exc:
            self.log(f"加载预览失败：{exc}")
            QMessageBox.warning(self, "预览失败", str(exc))

    def _toggle_screen_preview(self) -> None:
        """启动/停止屏幕区域实时采集。"""
        if self._screen_timer is not None and self._screen_timer.isActive():
            self._stop_screen_preview()
            return
        self._start_screen_preview()

    def _start_screen_preview(self) -> None:
        """启动屏幕区域实时采集定时器。"""
        try:
            import pyautogui  # noqa: F811
        except ImportError:
            self.log("屏幕采集需要 pyautogui：pip install pyautogui")
            return

        self._screen_timer = QTimer(self)
        self._screen_timer.timeout.connect(self._capture_screen_frame)
        self._screen_timer.start(100)  # 10 FPS
        self.load_preview_btn.setText("停止预览")
        self.log("屏幕实时预览已启动")

    def _stop_screen_preview(self) -> None:
        """停止屏幕区域实时采集。"""
        if self._screen_timer is not None:
            self._screen_timer.stop()
            self._screen_timer = None
        self.load_preview_btn.setText("加载预览")
        self.log("屏幕实时预览已停止")

    def _capture_screen_frame(self) -> None:
        """捕获一帧屏幕画面并更新预览。"""
        try:
            import pyautogui  # noqa: F811
            args = self._collect_args()
            left, top, width, height = [int(v) for v in args.capture_area]
            screenshot = pyautogui.screenshot(region=(left, top, width, height))
            image = np.asarray(screenshot.convert("RGB"))
            image = self._apply_roi_crop(image)
            self.preview_label.set_preview_image(image)
            self._last_preview_image = image
            self._track_spot_on_frame(image)
            self._compute_and_update_profile()
            self._update_video_metrics(image)
        except Exception:
            pass  # 静默处理，避免定时器中断

    def _on_roi_selected(self, x: int, y: int, w: int, h: int) -> None:
        self.focus_roi_edit.setText(f"{x}, {y}, {w}, {h}")
        self.log(f"框选 ROI：({x}, {y}, {w}, {h})")

        # 视频模式下自动更新截图区域
        if self.mode_combo.currentIndex() == 3:
            image = self.preview_label.get_original_image()
            if image is not None:
                ih, iw = image.shape[:2]
                self.capture_area_edit.setText(f"0, 0, {iw}, {ih}")
                self.log(f"截图区域已自动更新: 0, 0, {iw}, {ih}")

    def _on_cross_hair_changed(self, x: int, y: int) -> None:
        self._compute_and_update_profile()

    def _on_show_v_line_toggled(self, checked: bool) -> None:
        self.preview_label.set_show_v_line(checked)

    def _on_show_h_line_toggled(self, checked: bool) -> None:
        self.preview_label.set_show_h_line(checked)

    def _on_zoom_changed(self, text: str) -> None:
        try:
            ratio = float(text.strip("%")) / 100.0
        except ValueError:
            ratio = 1.0
        self.preview_label.set_zoom_ratio(ratio)
        # 根据缩放调整预览区域最小尺寸
        base_w, base_h = 400, 320
        self.preview_label.setMinimumSize(int(base_w * ratio), int(base_h * ratio))

    def _on_detect_spots(self) -> None:
        """自动检测光斑并将十字线移焦点中心。"""
        image = self.preview_label.get_original_image()
        if image is None:
            self.log("请先加载预览图像")
            return

        try:
            import cv2
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image

            algo = self.spot_algo_combo.currentIndex()
            threshold = self.spot_threshold_spin.value()
            min_area = self.spot_min_area_spin.value()
            max_area = self.spot_max_area_spin.value()
            blur_kernel = self.spot_blur_spin.value()

            # 确保模糊核为奇数
            if blur_kernel % 2 == 0:
                blur_kernel += 1

            blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
            keypoints = []

            if algo == 0:  # bright_blob
                params = cv2.SimpleBlobDetector_Params()
                params.filterByArea = True
                params.minArea = min_area
                params.maxArea = max_area
                params.filterByCircularity = False
                params.filterByConvexity = False
                params.filterByInertia = False
                params.minThreshold = threshold
                params.maxThreshold = 220
                detector = cv2.SimpleBlobDetector_create(params)
                keypoints = detector.detect(blurred)

            elif algo == 1:  # otsu_threshold
                _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                keypoints = self._find_contour_keypoints(thresh, min_area, max_area)

            elif algo == 2:  # gradient_peak
                sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
                grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
                grad_mag = (grad_mag / grad_mag.max() * 255).astype(np.uint8)
                _, thresh = cv2.threshold(grad_mag, threshold, 255, cv2.THRESH_BINARY)
                keypoints = self._find_contour_keypoints(thresh, min_area, max_area)

            elif algo == 3:  # contour
                _, thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
                keypoints = self._find_contour_keypoints(thresh, min_area, max_area)

            # 退化方案：如果没检测到，用 OTSU 重试
            if not keypoints:
                _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                keypoints = self._find_contour_keypoints(thresh, min_area, max_area)

            if not keypoints:
                self.log("未检测到光斑")
                return

            self.log(f"检测到 {len(keypoints)} 个光斑 (算法: {self.spot_algo_combo.currentText().split()[0]})")

            if len(keypoints) == 1:
                kp = keypoints[0]
                cx, cy = int(kp.pt[0]), int(kp.pt[1])
                self.preview_label.set_cross_hair_position(cx, cy)
                self._compute_and_update_profile()
                # 存储原始图像坐标（用于跨 ROI 裁剪的跟踪）
                roi = self.preview_label.get_roi()
                if roi is not None:
                    rx, ry, rw, rh = roi
                    self._tracked_spot = (cx + rx, cy + ry)
                else:
                    self._tracked_spot = (cx, cy)
                self.log(f"光斑定位并开始跟踪: ({cx}, {cy})")
            else:
                self._spot_selection_dialog(keypoints)

        except Exception as exc:
            self.log(f"光斑检测失败: {exc}")

    @staticmethod
    def _find_contour_keypoints(thresh: np.ndarray, min_area: int, max_area: int) -> list:
        """从二值图像中通过轮廓检测提取光斑关键点。"""
        import cv2
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        keypoints = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area <= area <= max_area:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    size = np.sqrt(area / np.pi) * 2
                    keypoints.append(cv2.KeyPoint(cx, cy, size))
        return keypoints

    def _spot_selection_dialog(self, keypoints) -> None:
        """弹出光斑选择对话框，每个光斑附带缩略图。"""
        from .qt_compat import QDialog, QListWidget, QListWidgetItem, QDialogButtonBox

        image = self.preview_label.get_original_image()
        if image is None:
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("选择光斑 - 附带缩略图")
        dialog.resize(500, 420)
        layout = QVBoxLayout(dialog)

        label = QLabel(f"检测到 {len(keypoints)} 个光斑，请选择一个：")
        layout.addWidget(label)

        list_widget = QListWidget()
        list_widget.setIconSize(QtCore.QSize(80, 80))
        list_widget.setSpacing(4)

        for i, kp in enumerate(keypoints):
            cx, cy = int(kp.pt[0]), int(kp.pt[1])
            size = int(kp.size) + 10
            h, w = image.shape[:2]
            x1 = max(0, cx - size)
            y1 = max(0, cy - size)
            x2 = min(w, cx + size)
            y2 = min(h, cy + size)
            if x2 > x1 and y2 > y1:
                patch = image[y1:y2, x1:x2]
                patch_rgb = patch[:, :, ::-1] if patch.ndim == 3 else patch
                patch_rgb = np.ascontiguousarray(patch_rgb)
                qimg = QtGui.QImage(patch_rgb.data, patch_rgb.shape[1], patch_rgb.shape[0],
                                     patch_rgb.shape[1] * 3, QtGui.QImage.Format_RGB888)
                icon = QtGui.QIcon(QtGui.QPixmap.fromImage(qimg).scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                icon = QtGui.QIcon()

            item = QListWidgetItem(icon, f"光斑 {i + 1}: ({cx}, {cy}), 大小: {kp.size:.0f}")
            item.setData(Qt.UserRole, i)
            list_widget.addItem(item)

        layout.addWidget(list_widget)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        layout.addWidget(btn_box)

        if dialog.exec_() == QDialog.Accepted and list_widget.currentItem():
            idx = list_widget.currentItem().data(Qt.UserRole)
            kp = keypoints[idx]
            cx, cy = int(kp.pt[0]), int(kp.pt[1])
            self.preview_label.set_cross_hair_position(cx, cy)
            self._compute_and_update_profile()
            # 存储原始图像坐标（用于跨 ROI 裁剪的跟踪）
            roi = self.preview_label.get_roi()
            if roi is not None:
                rx, ry, rw, rh = roi
                self._tracked_spot = (cx + rx, cy + ry)
            else:
                self._tracked_spot = (cx, cy)
            self.log(f"光斑 {idx + 1} 已选中并开始跟踪: ({cx}, {cy})")

    def _compute_and_update_profile(self) -> None:
        """根据当前预览图像和剖面线位置计算 RGB 剖面图。"""
        image = self.preview_label.get_original_image()
        if image is None:
            return

        cross = self.preview_label.get_cross_hair_position()
        if cross is None:
            return

        cx, cy = cross
        h, w = image.shape[:2]
        cx = max(0, min(w - 1, cx))
        cy = max(0, min(h - 1, cy))

        try:
            # 提取横线（水平方向）上的 RGB 像素值
            profile_h = image[cy, :, :3].astype(np.float64) if image.ndim >= 3 else image[cy, :].astype(np.float64)
            if profile_h.ndim == 1:
                profile_h = np.stack([profile_h] * 3, axis=-1)

            # 提取竖线（垂直方向）上的 RGB 像素值
            profile_v = image[:, cx, :3].astype(np.float64) if image.ndim >= 3 else image[:, cx].astype(np.float64)
            if profile_v.ndim == 1:
                profile_v = np.stack([profile_v] * 3, axis=-1)

            self.profile_plot.set_profile(profile_h, profile_v)
            self.profile_plot.set_cross_hair_position(cx, cy)
        except Exception:
            pass

    def _apply_roi_from_preview(self) -> None:
        roi = self.preview_label.get_roi()
        if roi is None:
            QMessageBox.information(self, "提示", "请先在预览图中框选 ROI")
            return
        self.focus_roi_edit.setText(", ".join(str(v) for v in roi))
        self.log(f"应用 ROI：{roi}")
        # 即时调整十字线坐标到裁剪后坐标系，避免切换瞬间错位
        if self._tracked_spot is not None:
            rx, ry, rw, rh = roi
            tx, ty = self._tracked_spot
            cx, cy = tx - rx, ty - ry
            if 0 <= cx < rw and 0 <= cy < rh:
                self.preview_label.set_cross_hair_position(cx, cy)
        # 视频模式下立即刷新应用裁剪
        if self._video_capture is not None:
            self._show_video_frame(self._video_current_frame)

    def _clear_roi(self) -> None:
        # 恢复十字线到原始图像坐标，确保消除 ROI 后坐标正确
        if self._tracked_spot is not None:
            tx, ty = self._tracked_spot
            self.preview_label.set_cross_hair_position(tx, ty)
        self.preview_label.set_roi(0, 0, 0, 0)
        self.preview_label._roi = None
        self.focus_roi_edit.setText(", ".join(str(v) for v in DEFAULT_FOCUS_ROI))
        # 刷新当前帧以恢复完整画面
        if self._video_capture is not None:
            self._show_video_frame(self._video_current_frame)
        self.log("已清除 ROI")

    def _save_current_frame(self) -> None:
        if not hasattr(self, "_last_preview_image") or self._last_preview_image is None:
            QMessageBox.information(self, "提示", "请先加载预览")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存当前帧", "preview_frame.png", "PNG 图片 (*.png)"
        )
        if path:
            import cv2
            cv2.imwrite(path, self._last_preview_image)
            self.log(f"已保存：{path}")

    def _build_config(self, args: UiRuntimeArgs) -> AutofocusConfig:
        kwargs: Dict[str, Any] = {
            "capture_mode": args.capture_mode,
            "window_title": args.window_title,
            "capture_area": args.capture_area,
            "focus_roi": args.focus_roi,
            "z_enabled": args.z_enabled,
            "z_axis": args.z_axis,
            "z_picomotor_conn": args.z_picomotor_conn,
            "z_picomotor_backend": args.z_picomotor_backend,
            "autofocus_enabled": args.autofocus_enabled,
            "autofocus_focus_trigger_ratio": args.trigger_ratio,
            "autofocus_stop_ratio": args.stop_ratio,
            "autofocus_focus_trigger_count": args.trigger_count,
            "z_search_strategy": args.search_strategy,
            "usb_device_index": args.usb_device,
        }
        if args.capture_mode == "local_video_file":
            kwargs["local_video_path"] = self.video_path_edit.text().strip()
        return AutofocusConfig(**kwargs)

    def _build_controller(
        self, args: UiRuntimeArgs
    ) -> tuple[AutofocusController, Optional[FocusSimulator]]:
        cfg = self._build_config(args)
        metrics_calc = FocusMetricsCalculator(cfg)
        scorer = FocusScorer(cfg, metrics_calc)
        # 注入已保存的参考数据（建立参考 → 启动闭环之间共享）
        if self._reference_data is not None:
            scorer.focus_reference = self._reference_data
            scorer.focus_reference_ready = True
        z_axis = ZAxisController(cfg)
        controller = AutofocusController(cfg, scorer, metrics_calc, z_axis)
        return controller, None

    def _build_reference(self) -> None:
        self._stop_screen_preview()
        try:
            args = self._collect_args()
            controller, _ = self._build_controller(args)

            ref_dir = Path(args.output) / "reference"
            ref_dir.mkdir(parents=True, exist_ok=True)
            controller.on_log = self.log
            ref = controller.build_reference(output_root=ref_dir)
            count = ref.get("capture_count", 0)
            # 保存参考数据，供后续闭环使用（controller 在 _start_loop 中会新建）
            self._reference_data = controller.scorer.focus_reference
            self.log(f"参考基线建立完成：{count} 次采集")
            QMessageBox.information(self, "完成", f"参考基线已建立\n采集次数：{count}")
        except Exception as exc:
            logger.exception("build_reference failed")
            self.log(f"建立参考失败：{exc}")
            QMessageBox.critical(self, "错误", f"建立参考失败：{exc}")

    def _start_loop(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(self, "提示", "循环已在运行中")
            return

        # 清理上一次已完成但未释放的线程对象，确保可以重新启动
        if self._thread is not None:
            try:
                self._thread.wait(500)
            except Exception:
                pass
            self._disconnect_worker_signals()
            self._thread = None

        # 停止屏幕实时预览，避免与后台线程争抢截图资源导致闪退
        self._stop_screen_preview()

        args = self._collect_args()
        try:
            controller, simulator = self._build_controller(args)
        except Exception as exc:
            self.log(f"初始化失败：{exc}")
            QMessageBox.critical(self, "错误", f"初始化失败：{exc}")
            return

        # 检测 Z 轴电机连接状态，无电机时提前警告
        if args.z_enabled:
            ok, err_msg = controller.z_axis.check_available()
            if not ok:
                self.log(f"[Z轴] {err_msg}")
                reply = QMessageBox.warning(
                    self,
                    "Z 轴电机未连接",
                    f"{err_msg}\n\n闭环将继续运行并监控 FocusScore，"
                    "但触发补焦时 Z 轴移动将失败。\n\n是否继续？",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )
                if reply == QMessageBox.No:
                    self.log("用户取消启动闭环")
                    return
                self.log("用户选择在无电机状态下继续运行闭环")
            else:
                self.log("[Z轴] 电机连接检测通过")

        runtime_args = {
            "output": args.output,
            "cycles": args.cycles,
            "interval": args.interval,
        }

        self._thread = AutofocusThread(self)
        self._thread.configure(
            cfg=self._build_config(args),
            args=runtime_args,
            controller=controller,
            simulator=simulator,
        )

        self._thread.worker.preview_updated.connect(self._on_preview_updated)
        self._thread.worker.metrics_updated.connect(self._on_metrics_updated)
        self._thread.worker.score_updated.connect(self._on_score_updated)
        self._thread.worker.log.connect(self.log)
        self._thread.worker.status_changed.connect(self._on_status_changed)
        self._thread.worker.finished.connect(self._on_loop_finished)
        self._thread.worker.error.connect(self._on_loop_error)
        self._thread.worker.capture_requested.connect(self._on_worker_capture)

        self._set_controls_running(True)
        self.score_plot.set_thresholds(args.trigger_ratio, args.stop_ratio)
        self.score_plot.clear()
        self._thread.start()
        self.log("已启动自动聚焦闭环")

    def _build_and_start_thread(
        self,
        controller: AutofocusController,
        cfg: AutofocusConfig,
        runtime_args: Dict[str, Any],
    ) -> Any:
        """构建并启动 AutofocusThread（供测试调用）。"""
        from .worker import AutofocusThread

        thread = AutofocusThread(self)
        thread.configure(
            cfg=cfg,
            args=runtime_args,
            controller=controller,
            simulator=None,
        )
        thread.worker.preview_updated.connect(self._on_preview_updated)
        thread.worker.metrics_updated.connect(self._on_metrics_updated)
        thread.worker.score_updated.connect(self._on_score_updated)
        thread.worker.log.connect(self.log)
        thread.worker.status_changed.connect(self._on_status_changed)
        thread.worker.finished.connect(self._on_loop_finished)
        thread.worker.error.connect(self._on_loop_error)
        thread.worker.capture_requested.connect(self._on_worker_capture)

        self._set_controls_running(True)
        thread.start()
        self.log("已启动自动聚焦闭环")
        return thread

    def _pause_loop(self) -> None:
        if self._thread is None or not self._thread.isRunning():
            return
        if self._thread.worker._state.paused:
            self._thread.worker.resume_loop()
            self.pause_btn.setText("暂停")
            self.log("继续运行")
        else:
            self._thread.worker.pause_loop()
            self.pause_btn.setText("继续")
            self.log("已暂停")

    def _stop_loop(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            self._thread.worker.stop_loop()
            self.log("正在停止...")
            # 最多等待 2 秒让线程自然结束；若仍在运行则强制终止并清理 UI
            if self._thread.wait(2000):
                self._disconnect_worker_signals()
                self._set_controls_running(False)
                self.status_label.setText("已停止")
                self.log("闭环已停止")
            else:
                self._disconnect_worker_signals()
                self._set_controls_running(False)
                self.status_label.setText("停止超时")
                self.log("警告：停止请求超时，线程可能仍在收尾")

    @Slot(np.ndarray)
    def _on_preview_updated(self, image: np.ndarray) -> None:
        self.preview_label.set_preview_image(image)
        self._last_preview_image = image
        self._compute_and_update_profile()

    @Slot(dict, dict)
    def _on_metrics_updated(self, full: dict, roi: dict) -> None:
        self.metrics_table.update_metrics(full, roi)

    @Slot(int, float)
    def _on_score_updated(self, cycle: int, score: float) -> None:
        self.score_plot.append(cycle, score)

    @Slot(str)
    def log(self, message: str) -> None:
        self.log_edit.appendPlainText(message)

    @Slot(bool, int, int)
    def _on_status_changed(self, running: bool, cycle: int, total: int) -> None:
        if running:
            self.status_label.setText(f"运行中 - 第 {cycle} 轮")
            if total > 0:
                self.progress_bar.setValue(int(cycle / total * 100))
        else:
            self.status_label.setText("已停止")
            self.progress_bar.setValue(0)

    @Slot()
    def _on_loop_finished(self) -> None:
        self._disconnect_worker_signals()
        self._set_controls_running(False)
        # 等待线程彻底结束并释放引用，避免影响下一次启动
        if self._thread is not None:
            try:
                self._thread.wait(1000)
            except Exception:
                pass
            self._thread = None
        self.status_label.setText("完成")
        self.log("闭环运行结束")
        self.log("循环结束，可再次建立参考或启动闭环")

    def _disconnect_worker_signals(self) -> None:
        """断开当前 worker 的所有信号连接，防止旧信号残留。"""
        if self._thread is None or self._thread.worker is None:
            return
        try:
            w = self._thread.worker
            w.preview_updated.disconnect(self._on_preview_updated)
            w.metrics_updated.disconnect(self._on_metrics_updated)
            w.score_updated.disconnect(self._on_score_updated)
            w.log.disconnect(self.log)
            w.status_changed.disconnect(self._on_status_changed)
            w.finished.disconnect(self._on_loop_finished)
            w.error.disconnect(self._on_loop_error)
            w.capture_requested.disconnect(self._on_worker_capture)
        except Exception:
            pass  # 可能已经断开

    def _on_worker_capture(self) -> None:
        """响应后台线程的截图请求（在主线程安全执行 pyautogui）。"""
        if self._thread is None or self._thread.worker is None:
            return
        try:
            import pyautogui
            args = self._collect_args()
            left, top, width, height = [int(v) for v in args.capture_area]
            if width <= 0 or height <= 0:
                self.log("[错误] 截图区域无效，使用默认区域")
                self._thread.worker.set_captured_image(None)
                return
            screenshot = pyautogui.screenshot(region=(left, top, width, height))
            image = np.asarray(screenshot.convert("RGB"))
            self._thread.worker.set_captured_image(image)
        except Exception as e:
            logger.warning(f"主线程截图失败: {e}")
            try:
                self._thread.worker.set_captured_image(None)
            except Exception:
                pass

    @Slot(str)
    def _on_loop_error(self, message: str) -> None:
        self._disconnect_worker_signals()
        self._set_controls_running(False)
        self.status_label.setText("错误")
        self.log(f"错误：{message}")
        QMessageBox.critical(self, "运行错误", message)

    def _set_controls_running(self, running: bool) -> None:
        self.start_btn.setEnabled(not running)
        self.build_ref_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.pause_btn.setEnabled(running)
        self.pause_btn.setText("暂停")
        self.mode_combo.setEnabled(not running)
        self.cycles_spin.setEnabled(not running)

    def _choose_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择输出目录", self.output_edit.text())
        if path:
            self.output_edit.setText(path)

    def _choose_video_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv);;所有文件 (*.*)"
        )
        if path:
            self.video_path_edit.setText(path)
            self.log(f"已选择视频文件: {path}")
            self._detect_video_params(path)

    def _detect_video_params(self, video_path: str) -> None:
        """检测视频参数并自动填入采集区域。"""
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.log("无法打开视频文件检测参数")
                return

            self._video_fps = cap.get(cv2.CAP_PROP_FPS)
            if self._video_fps <= 0:
                self._video_fps = 30.0
            self._video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._video_current_frame = 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = self._video_frame_count / self._video_fps if self._video_fps > 0 else 0

            cap.release()

            # 自动填入截图区域
            self.capture_area_edit.setText(f"0, 0, {width}, {height}")

            # 显示视频信息
            mins, secs = divmod(int(duration), 60)
            self.video_info_label.setText(
                f"分辨率: {width}×{height} | 帧率: {self._video_fps:.1f} fps | "
                f"总帧数: {self._video_frame_count} | 时长: {mins}:{secs:02d}"
            )
            self.log(
                f"视频参数: {width}×{height}, {self._video_fps:.1f} fps, "
                f"{self._video_frame_count} 帧, {mins}:{secs:02d}"
            )

            # 更新进度条范围
            self.video_slider.setRange(0, self._video_frame_count - 1)
            self.video_slider.setValue(0)
            self._update_video_time_label()

            # 自动加载第一帧作为预览
            self._open_video_player(video_path)
            self._show_video_frame(0)

        except Exception as exc:
            self.log(f"检测视频参数失败: {exc}")

    # ============================================================
    #  视频播放器方法
    # ============================================================

    def _set_video_controls_visible(self, visible: bool) -> None:
        self.video_play_btn.setVisible(visible)
        self.video_pause_btn.setVisible(visible)
        self.video_stop_btn.setVisible(visible)
        self.video_slider.setVisible(visible)
        self.video_time_label.setVisible(visible)
        self.video_info_label.setVisible(visible)

    def _open_video_player(self, video_path: str) -> None:
        """打开视频文件用于播放器。"""
        self._close_video_player()
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.log("无法打开视频文件")
            return
        self._video_capture = cap
        self._video_playing = False
        self._video_current_frame = 0

    def _close_video_player(self) -> None:
        """释放视频播放器资源。"""
        self._video_playing = False
        if self._video_timer is not None:
            self._video_timer.stop()
            self._video_timer = None
        if self._video_capture is not None:
            try:
                self._video_capture.release()
            except Exception:
                pass
            self._video_capture = None

    def _show_video_frame(self, frame_idx: int) -> None:
        """跳转到指定帧并显示。"""
        if self._video_capture is None:
            return
        try:
            import cv2
            frame_idx = max(0, min(self._video_frame_count - 1, frame_idx))
            self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame_bgr = self._video_capture.read()
            if ret and frame_bgr is not None:
                image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                image = self._apply_roi_crop(image)
                self.preview_label.set_preview_image(image)
                self._last_preview_image = image
                self._video_current_frame = frame_idx

                if not self._video_slider_dragging:
                    self.video_slider.setValue(frame_idx)
                self._update_video_time_label()

                # 先跟踪光斑（更新十字线位置），再计算剖面和指标
                self._track_spot_on_frame(image)
                self._compute_and_update_profile()
                self._update_video_metrics(image)
        except Exception as exc:
            self.log(f"显示视频帧失败: {exc}")

    def _on_video_play(self) -> None:
        """播放视频。"""
        if self._video_capture is None:
            video_path = self.video_path_edit.text().strip()
            if not video_path:
                self.log("请先选择视频文件")
                return
            self._detect_video_params(video_path)

        if self._video_playing:
            return

        self._video_playing = True
        if self._video_timer is None:
            self._video_timer = QTimer(self)
            self._video_timer.timeout.connect(self._play_next_frame)

        interval = int(1000.0 / self._video_fps) if self._video_fps > 0 else 33
        self._video_timer.start(interval)
        self.video_play_btn.setEnabled(False)
        self.video_pause_btn.setEnabled(True)
        self.log("视频播放中...")

    def _on_video_pause(self) -> None:
        """暂停视频。"""
        if not self._video_playing:
            return
        self._video_playing = False
        if self._video_timer is not None:
            self._video_timer.stop()
        self.video_play_btn.setEnabled(True)
        self.video_pause_btn.setEnabled(False)
        self.log("视频已暂停")

    def _on_video_stop(self) -> None:
        """停止视频。"""
        self._video_playing = False
        if self._video_timer is not None:
            self._video_timer.stop()
        self.video_play_btn.setEnabled(True)
        self.video_pause_btn.setEnabled(False)
        self._video_current_frame = 0
        self.video_slider.setValue(0)
        self._update_video_time_label()
        self._show_video_frame(0)
        self._tracked_spot = None  # 停止跟踪
        self.log("视频已停止")

    def _on_video_slider_pressed(self) -> None:
        """用户开始拖动进度条。"""
        self._video_slider_dragging = True

    def _on_video_slider_released(self) -> None:
        """用户释放进度条，跳转到指定帧。"""
        self._video_slider_dragging = False
        target = self.video_slider.value()
        self._show_video_frame(target)

    def _on_video_slider_changed(self, value: int) -> None:
        """进度条值变化时更新时间标签。"""
        if self._video_slider_dragging:
            self._video_current_frame = value
            self._update_video_time_label()

    def _play_next_frame(self) -> None:
        """播放下一帧（由 QTimer 触发）。"""
        if not self._video_playing or self._video_capture is None:
            return

        try:
            import cv2
            # 如果播放到末尾，循环回到开头
            if self._video_current_frame >= self._video_frame_count - 1:
                self._video_current_frame = 0
                self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

            ret, frame_bgr = self._video_capture.read()
            if ret and frame_bgr is not None:
                image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                image = self._apply_roi_crop(image)
                self.preview_label.set_preview_image(image)
                self._last_preview_image = image
                self._video_current_frame += 1
                self.video_slider.setValue(self._video_current_frame)
                self._update_video_time_label()

                # 先跟踪光斑，再计算剖面和指标
                self._track_spot_on_frame(image)
                self._compute_and_update_profile()
                self._update_video_metrics(image)
            else:
                # 文件末尾，循环
                self._video_current_frame = 0
                self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame_bgr = self._video_capture.read()
                if ret and frame_bgr is not None:
                    image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    image = self._apply_roi_crop(image)
                    self.preview_label.set_preview_image(image)
                    self._last_preview_image = image
                    self._video_current_frame += 1
                    self.video_slider.setValue(self._video_current_frame)
                    self._update_video_time_label()

                    # 先跟踪光斑，再计算剖面和指标
                    self._track_spot_on_frame(image)
                    self._compute_and_update_profile()
                    self._update_video_metrics(image)
        except Exception as exc:
            self.log(f"播放出错: {exc}")
            self._on_video_pause()

    def _update_video_time_label(self) -> None:
        """更新视频时间标签。"""
        total_sec = self._video_frame_count / self._video_fps if self._video_fps > 0 else 0
        cur_sec = self._video_current_frame / self._video_fps if self._video_fps > 0 else 0
        cur_m, cur_s = divmod(int(cur_sec), 60)
        total_m, total_s = divmod(int(total_sec), 60)
        self.video_time_label.setText(f"{cur_m:02d}:{cur_s:02d} / {total_m:02d}:{total_s:02d}")

    def _apply_roi_crop(self, image: np.ndarray) -> np.ndarray:
        """如果设置了 ROI，对图像进行裁剪。"""
        roi = self.preview_label.get_roi()
        if roi is None:
            return image
        rx, ry, rw, rh = roi
        h, w = image.shape[:2]
        rx = max(0, min(w - 1, rx))
        ry = max(0, min(h - 1, ry))
        rw = min(rw, w - rx)
        rh = min(rh, h - ry)
        if rw <= 0 or rh <= 0:
            return image
        return image[ry:ry + rh, rx:rx + rw]

    def _update_video_metrics(self, image: np.ndarray) -> None:
        """更新聚焦指标和 FocusScore 曲线（跳帧计算以降低 CPU 负载）。"""
        try:
            # 跳帧优化：每 3 帧计算一次指标，大幅降低 CPU 负载
            if not hasattr(self, "_video_metrics_skip_counter"):
                self._video_metrics_skip_counter = 0
            self._video_metrics_skip_counter += 1
            if self._video_metrics_skip_counter % 3 != 0:
                return

            if not hasattr(self, "_video_metrics_calc"):
                self._video_metrics_calc = FocusMetricsCalculator(
                    self._build_config(self._collect_args())
                )
            if not hasattr(self, "_video_score_cycle"):
                self._video_score_cycle = 0

            metrics = self._video_metrics_calc.compute_for_image(image)
            if metrics:
                self._video_score_cycle += 1
                self.metrics_table.update_metrics(metrics, metrics)
                score = metrics.get("laplacian_variance", 0)
                self.score_plot.append(self._video_score_cycle, score)
        except Exception:
            pass

    def _track_spot_on_frame(self, image: np.ndarray) -> None:
        """在当前帧上快速跟踪光斑（强度加权质心，高效且精准）。"""
        if self._tracked_spot is None:
            return
        try:
            import cv2
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image

            # 如果启用了 ROI 裁剪，将原始坐标转为裁剪后坐标
            roi = self.preview_label.get_roi()
            tx, ty = self._tracked_spot
            if roi is not None:
                rx, ry, rw, rh = roi
                tx -= rx
                ty -= ry
                if tx < 0 or ty < 0 or tx >= rw or ty >= rh:
                    return

            h, w = gray.shape
            # 局部窗口：在上一帧光斑位置周围裁剪
            window_size = 160
            half_win = window_size // 2
            x1 = max(0, tx - half_win)
            y1 = max(0, ty - half_win)
            x2 = min(w, tx + half_win)
            y2 = min(h, ty + half_win)
            if x2 <= x1 or y2 <= y1:
                return

            patch = gray[y1:y2, x1:x2].astype(np.float32)

            # 强度加权质心法：计算局部窗口内亮区的"重心"
            # 比阈值+轮廓法更鲁棒，不受阈值选择影响
            bg = np.percentile(patch, 30)  # 估计背景亮度
            patch_fg = np.maximum(patch - bg, 0)
            total = patch_fg.sum()
            if total < 1e-6:
                # 局部无亮区，回退到全帧搜索
                self._full_frame_track_spot(gray, tx, ty, roi)
                return

            yy, xx = np.mgrid[0:patch_fg.shape[0], 0:patch_fg.shape[1]]
            cx = int(np.round((xx * patch_fg).sum() / total))
            cy = int(np.round((yy * patch_fg).sum() / total))

            # 验证：质心不应偏离窗口中心太远（防止跟踪到无关亮区）
            patch_cx = tx - x1
            patch_cy = ty - y1
            dist = np.sqrt((cx - patch_cx) ** 2 + (cy - patch_cy) ** 2)
            if dist > half_win * 0.8:
                # 质心偏离太远，可能不是同一个光斑，回退到全帧搜索
                self._full_frame_track_spot(gray, tx, ty, roi)
                return

            new_x = x1 + cx
            new_y = y1 + cy

            # 更新：存储原始图像坐标，十字线用当前图像坐标
            if roi is not None:
                rx, ry, rw, rh = roi
                self._tracked_spot = (new_x + rx, new_y + ry)
            else:
                self._tracked_spot = (new_x, new_y)
            self.preview_label.set_cross_hair_position(new_x, new_y)
        except Exception:
            pass

    def _full_frame_track_spot(self, gray: np.ndarray, tx: int, ty: int, roi) -> None:
        """全帧快速搜索回退（仅阈值+轮廓，避免昂贵 blob detector）。"""
        try:
            import cv2
            threshold = self.spot_threshold_spin.value()
            _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            if not np.any(thresh):
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            min_area = self.spot_min_area_spin.value()
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            best_pt = None
            best_dist = float('inf')
            for cnt in contours:
                if cv2.contourArea(cnt) < min_area:
                    continue
                M = cv2.moments(cnt)
                if M["m00"] <= 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                d = (cx - tx) ** 2 + (cy - ty) ** 2
                if d < best_dist:
                    best_dist = d
                    best_pt = (cx, cy)

            if best_pt is not None:
                new_x, new_y = best_pt
                max_dist = max(gray.shape[0], gray.shape[1]) * 0.3
                if np.sqrt(best_dist) <= max_dist:
                    if roi is not None:
                        rx, ry, rw, rh = roi
                        self._tracked_spot = (new_x + rx, new_y + ry)
                    else:
                        self._tracked_spot = (new_x, new_y)
                    self.preview_label.set_cross_hair_position(new_x, new_y)
        except Exception:
            pass

    def closeEvent(self, event) -> None:
        if self._thread is not None and self._thread.isRunning():
            self._thread.worker.stop_loop()
            self._thread.wait(2000)
        self._stop_screen_preview()
        self._close_video_player()
        event.accept()


def run_autofocus_ui() -> int:
    app = QApplication(sys.argv)
    window = AutofocusMainWindow()
    window.show()
    return app_exec(app)
