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
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    app_exec,
)

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
from .widgets import FocusScorePlot, MetricsTable, RoiPreviewLabel
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
    autofocus_enabled: bool = True
    trigger_ratio: float = DEFAULT_TRIGGER_RATIO
    stop_ratio: float = 0.95
    trigger_count: int = DEFAULT_TRIGGER_COUNT
    search_strategy: str = DEFAULT_SEARCH_STRATEGY
    demo_sim: bool = True


class AutofocusMainWindow(QMainWindow):
    """自动聚焦主窗口。"""

    def __init__(self) -> None:
        super().__init__()
        self._theme = get_theme("dark")
        self.setWindowTitle("AutoZoom 自动聚焦控制台")
        self.resize(1400, 900)

        self._thread: Optional[AutofocusThread] = None
        self._controller: Optional[AutofocusController] = None
        self._simulator: Optional[FocusSimulator] = None

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
        splitter.addWidget(right_widget)

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
        self.mode_combo.addItems(["模拟模式 (无硬件)", "屏幕区域", "窗口 ROI", "USB 相机"])
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
        layout.addWidget(self.roi_group)

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
        self.start_btn = QPushButton("启动闭环")
        self.start_btn.setObjectName("primary")
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
        layout.setSpacing(10)
        layout.setContentsMargins(6, 6, 6, 6)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        preview_tab = QWidget()
        preview_layout = QVBoxLayout(preview_tab)
        self.preview_label = RoiPreviewLabel()
        self.preview_label.set_reference_mode(True)
        preview_layout.addWidget(self.preview_label)

        preview_btn_layout = QHBoxLayout()
        self.set_roi_btn = QPushButton("应用 ROI")
        self.clear_roi_btn = QPushButton("清除 ROI")
        self.save_frame_btn = QPushButton("保存当前帧")
        preview_btn_layout.addWidget(self.set_roi_btn)
        preview_btn_layout.addWidget(self.clear_roi_btn)
        preview_btn_layout.addStretch()
        preview_btn_layout.addWidget(self.save_frame_btn)
        preview_layout.addLayout(preview_btn_layout)
        self.tabs.addTab(preview_tab, "实时预览")

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

        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumBlockCount(500)
        layout.addWidget(self.log_edit)

        return widget

    def _connect_signals(self) -> None:
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.load_preview_btn.clicked.connect(self._load_preview)
        self.set_roi_btn.clicked.connect(self._apply_roi_from_preview)
        self.clear_roi_btn.clicked.connect(self._clear_roi)
        self.save_frame_btn.clicked.connect(self._save_current_frame)
        self.output_btn.clicked.connect(self._choose_output_dir)
        self.sim_params_btn.clicked.connect(self._show_sim_params_dialog)

        self.build_ref_btn.clicked.connect(self._build_reference)
        self.start_btn.clicked.connect(self._start_loop)
        self.pause_btn.clicked.connect(self._pause_loop)
        self.stop_btn.clicked.connect(self._stop_loop)

        self.preview_label.roiSelected.connect(self._on_roi_selected)

    def _apply_default_values(self) -> None:
        self._on_mode_changed(0)
        self._sim_params: Dict[str, Any] = {
            "peak_z": 50,
            "blur_scale": 0.3,
            "initial_z": 0,
            "drift_rate": 0.0,
            "focus_degrade": 0.0,
            "pattern": "cells",
            "image_size": (400, 400),
            "noise_level": 0.02,
        }

    def _collect_args(self) -> UiRuntimeArgs:
        args = UiRuntimeArgs()
        args.output = self.output_edit.text().strip() or DEFAULT_OUTPUT_DIR
        # UI 显示模式 -> Focus 模块内部 capture_mode 的映射
        ui_mode = ["sim", "screen", "window", "usb"][self.mode_combo.currentIndex()]
        args.capture_mode = {
            "sim": "screen_region",
            "screen": "screen_region",
            "window": "window_roi",
            "usb": "usb_camera",
        }[ui_mode]
        args.window_title = self.window_title_edit.text().strip()
        args.usb_device = self.usb_device_spin.value()
        args.capture_area = self._parse_rect(self.capture_area_edit.text(), DEFAULT_CAPTURE_AREA)
        args.focus_roi = self._parse_rect(self.focus_roi_edit.text(), DEFAULT_FOCUS_ROI)
        args.cycles = self.cycles_spin.value()
        args.interval = self.interval_spin.value()
        args.z_enabled = self.z_enabled_check.isChecked()
        args.z_axis = self.z_axis_spin.value()
        args.autofocus_enabled = self.autofocus_check.isChecked()
        args.trigger_ratio = self.trigger_ratio_spin.value()
        args.stop_ratio = self.stop_ratio_spin.value()
        args.trigger_count = self.trigger_count_spin.value()
        args.search_strategy = self.strategy_combo.currentText()
        args.demo_sim = args.capture_mode == "sim"
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
        is_sim = index == 0
        is_screen = index == 1
        is_window = index == 2
        is_usb = index == 3

        # 更新设备选择下拉框
        self.resource_combo.clear()
        self.resource_combo.setEnabled(not is_sim and not is_screen)

        if is_window:
            self.resource_combo.addItem("正在检测窗口...", None)
            self._refresh_windows()
        elif is_usb:
            self.resource_combo.addItem("正在检测相机...", None)
            self._refresh_cameras()
        else:
            self.resource_combo.addItem("当前模式无需选择", None)

        self.window_title_edit.setEnabled(is_window)
        self.usb_device_spin.setEnabled(is_usb)
        self.sim_params_btn.setEnabled(is_sim)
        self.z_enabled_check.setEnabled(not is_sim)

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
        if mode == 2 and isinstance(data, str):
            self.window_title_edit.setText(data)
        elif mode == 3 and isinstance(data, int):
            self.usb_device_spin.setValue(data)

    def _load_preview(self) -> None:
        try:
            args = self._collect_args()
            cfg = self._build_config(args)
            metrics_calc = FocusMetricsCalculator(cfg)

            if args.demo_sim:
                from Focus.simulator import ImageGenerator
                gen = ImageGenerator(
                    cfg,
                    peak_z=self._sim_params.get("peak_z", 50),
                    blur_scale=self._sim_params.get("blur_scale", 0.3),
                    image_size=tuple(self._sim_params.get("image_size", (400, 400))),
                    pattern=self._sim_params.get("pattern", "cells"),
                    noise_level=self._sim_params.get("noise_level", 0.02),
                )
                image = gen.generate(z=0)
            else:
                live = metrics_calc.capture_live()
                # capture_live 返回的图像字段为 roi_rgb / full_rgb
                # 注意：不能用 or 直接判断 numpy 数组，会触发 ambiguous truth value 错误
                image = live.get("roi_rgb")
                if image is None:
                    image = live.get("full_rgb")

            if image is not None:
                self.preview_label.set_preview_image(image)
                self._last_preview_image = image
                self.log("已加载预览")
            else:
                self.log("无法获取预览图像")
        except Exception as exc:
            self.log(f"加载预览失败：{exc}")
            QMessageBox.warning(self, "预览失败", str(exc))

    def _on_roi_selected(self, x: int, y: int, w: int, h: int) -> None:
        self.focus_roi_edit.setText(f"{x}, {y}, {w}, {h}")
        self.log(f"框选 ROI：({x}, {y}, {w}, {h})")

    def _apply_roi_from_preview(self) -> None:
        roi = self.preview_label.get_roi()
        if roi is None:
            QMessageBox.information(self, "提示", "请先在预览图中框选 ROI")
            return
        self.focus_roi_edit.setText(", ".join(str(v) for v in roi))
        self.log(f"应用 ROI：{roi}")

    def _clear_roi(self) -> None:
        self.preview_label.set_roi(0, 0, 0, 0)
        self.focus_roi_edit.setText(", ".join(str(v) for v in DEFAULT_FOCUS_ROI))

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
            "autofocus_enabled": args.autofocus_enabled,
            "autofocus_focus_trigger_ratio": args.trigger_ratio,
            "autofocus_stop_ratio": args.stop_ratio,
            "autofocus_focus_trigger_count": args.trigger_count,
            "z_search_strategy": args.search_strategy,
            "usb_device_index": args.usb_device,
        }
        return AutofocusConfig(**kwargs)

    def _build_controller(
        self, args: UiRuntimeArgs
    ) -> tuple[AutofocusController, Optional[FocusSimulator]]:
        cfg = self._build_config(args)
        simulator = None

        if args.demo_sim:
            simulator, metrics_calc, scorer, controller = create_demo_environment(
                cfg,
                peak_z=self._sim_params.get("peak_z", 50),
                blur_scale=self._sim_params.get("blur_scale", 0.3),
                image_size=tuple(self._sim_params.get("image_size", (400, 400))),
                noise_level=self._sim_params.get("noise_level", 0.02),
                pattern=self._sim_params.get("pattern", "cells"),
                initial_z=self._sim_params.get("initial_z", 0),
                drift_rate=self._sim_params.get("drift_rate", 0.0),
                focus_degrade_per_cycle=self._sim_params.get("focus_degrade", 0.0),
                autofocus_enabled=True,
            )
            return controller, simulator

        metrics_calc = FocusMetricsCalculator(cfg)
        scorer = FocusScorer(cfg, metrics_calc)
        z_axis = ZAxisController(cfg)
        controller = AutofocusController(cfg, scorer, metrics_calc, z_axis)
        return controller, None

    def _build_reference(self) -> None:
        try:
            args = self._collect_args()
            controller, simulator = self._build_controller(args)
            if simulator is not None:
                simulator.virtual_z_axis.move_absolute(simulator.peak_z)

            ref_dir = Path(args.output) / "reference"
            ref_dir.mkdir(parents=True, exist_ok=True)
            controller.on_log = self.log
            ref = controller.build_reference(output_root=ref_dir)
            count = ref.get("capture_count", 0)
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

        args = self._collect_args()
        try:
            controller, simulator = self._build_controller(args)
        except Exception as exc:
            self.log(f"初始化失败：{exc}")
            QMessageBox.critical(self, "错误", f"初始化失败：{exc}")
            return

        if args.demo_sim and not controller.focus_reference_ready:
            simulator.virtual_z_axis.move_absolute(simulator.peak_z)
            ref_dir = Path(args.output) / "reference"
            ref_dir.mkdir(parents=True, exist_ok=True)
            controller.build_reference(output_root=ref_dir)

        runtime_args = {
            "output": args.output,
            "cycles": args.cycles,
            "interval": args.interval,
        }

        controller.on_log = self.log

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

        self._set_controls_running(True)
        self.score_plot.set_thresholds(args.trigger_ratio, args.stop_ratio)
        self.score_plot.clear()
        self._thread.start()
        self.log("已启动自动聚焦闭环")

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

    @Slot(np.ndarray)
    def _on_preview_updated(self, image: np.ndarray) -> None:
        self.preview_label.set_preview_image(image)
        self._last_preview_image = image

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
        self._set_controls_running(False)
        self.status_label.setText("完成")
        self.log("闭环运行结束")

    @Slot(str)
    def _on_loop_error(self, message: str) -> None:
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

    def _show_sim_params_dialog(self) -> None:
        QMessageBox.information(
            self,
            "模拟参数",
            f"当前模拟参数：\n"
            f"  peak_z: {self._sim_params.get('peak_z')}\n"
            f"  blur_scale: {self._sim_params.get('blur_scale')}\n"
            f"  initial_z: {self._sim_params.get('initial_z')}\n"
            f"  drift_rate: {self._sim_params.get('drift_rate')}\n"
            f"  focus_degrade: {self._sim_params.get('focus_degrade')}\n"
            f"  pattern: {self._sim_params.get('pattern')}\n"
            f"  image_size: {self._sim_params.get('image_size')}\n"
            f"  noise_level: {self._sim_params.get('noise_level')}\n\n"
            "可在 app.py 中 self._sim_params 修改默认值。",
        )

    def closeEvent(self, event) -> None:
        if self._thread is not None and self._thread.isRunning():
            self._thread.worker.stop_loop()
            self._thread.wait(2000)
        event.accept()


def run_autofocus_ui() -> int:
    app = QApplication(sys.argv)
    window = AutofocusMainWindow()
    window.show()
    return app_exec(app)
