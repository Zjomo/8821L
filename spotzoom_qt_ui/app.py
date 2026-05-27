from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

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

from .picomotor_driver_panel import PicomotorDriverPanel
from .models import EventRecord, RunMode, RuntimeProfile, TestCaseSpec, UiStatus
from .services import (
    DeviceRegistryService,
    DeviceTestService,
    ModuleCatalogService,
    RuntimeControlService,
)


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
        "Dashboard",
        "Alignment Workspace",
        "Module Center",
        "Device Center",
        "Device Test",
        "Run Modes",
        "Simulation Lab",
        "Logs & Reports",
        "Settings",
    ]

    def __init__(self, repo_root: Optional[Path] = None):
        super().__init__()
        self.runtime = RuntimeControlService(repo_root=repo_root)
        self.device_registry = DeviceRegistryService()
        self.device_test = DeviceTestService(self.runtime)
        self.module_catalog = ModuleCatalogService()
        self.profile = self.runtime.default_profile()

        self.env_status = UiStatus.NORMAL
        self.startup_status = UiStatus.NORMAL
        self.run_status = UiStatus.NORMAL

        self.run_process: Optional[QProcess] = None
        self.module_rows = []

        self.controls: Dict[str, QWidget] = {}
        self.status_fields: Dict[str, QLabel] = {}
        self.dashboard_fields: Dict[str, QLabel] = {}
        self.runtime_fields: Dict[str, QLabel] = {}

        self.setWindowTitle("SpotZoom 主动激光束稳定控制台")
        self.resize(1680, 980)
        self._build_ui()
        self._apply_profile_to_controls(self.profile)
        self._refresh_all_panels()
        self._append_log("UI 启动完成。")

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
        self._apply_app_style()

    def _build_menu_actions(self) -> None:
        toolbar = self.addToolBar("main")
        toolbar.setMovable(False)
        act_refresh = QAction("刷新", self)
        act_refresh.triggered.connect(self._refresh_all_panels)
        toolbar.addAction(act_refresh)
        act_export = QAction("导出报告", self)
        act_export.triggered.connect(self._export_report)
        toolbar.addAction(act_export)

    def _apply_app_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget { background: #11161C; color: #D9E2EC; font-size: 12px; }
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

    def _build_top_status_strip(self) -> QWidget:
        bar = QFrame()
        layout = QGridLayout(bar)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setHorizontalSpacing(12)
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
        btn_test.clicked.connect(lambda: self.nav_list.setCurrentRow(4))
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

        image_group = QGroupBox("实时图像区")
        image_layout = QVBoxLayout(image_group)
        self.image_mode_combo = QComboBox()
        self.image_mode_combo.addItems(["原图", "检测叠加", "阈值图", "候选点图", "轨迹图"])
        image_layout.addWidget(self.image_mode_combo)
        self.image_label = QLabel("No Frame")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(760, 540)
        self.image_label.setStyleSheet("background:#060A10; border:1px solid #2B3642;")
        image_layout.addWidget(self.image_label, 1)
        self.image_overlay_label = QLabel("ROI: - | P1/P2/P3: - | 置信度: - | 焦点评分: - | 偏差向量: -")
        image_layout.addWidget(self.image_overlay_label)
        layout.addWidget(image_group, 3)

        right = QWidget()
        right_layout = QVBoxLayout(right)

        ctrl = QGroupBox("准直控制区")
        ctrl_layout = QGridLayout(ctrl)
        self.btn_run = QPushButton("开始")
        self.btn_pause = QPushButton("暂停")
        self.btn_stop = QPushButton("停止")
        self.btn_step = QPushButton("单步一轮")
        self.btn_roi = QPushButton("重新选择 ROI")
        self.btn_env = QPushButton("环境检查")
        self.btn_startup = QPushButton("启动自检")
        self.btn_export = QPushButton("导出报告")
        actions = [
            self.btn_run,
            self.btn_pause,
            self.btn_stop,
            self.btn_step,
            self.btn_roi,
            self.btn_env,
            self.btn_startup,
            self.btn_export,
        ]
        for i, btn in enumerate(actions):
            ctrl_layout.addWidget(btn, i // 2, i % 2)
        self.btn_run.clicked.connect(self._start_alignment)
        self.btn_pause.clicked.connect(self._pause_alignment)
        self.btn_stop.clicked.connect(self._stop_alignment)
        self.btn_step.clicked.connect(self._single_step_run)
        self.btn_roi.clicked.connect(self._toggle_roi)
        self.btn_env.clicked.connect(self._run_env_check)
        self.btn_startup.clicked.connect(self._run_startup_check)
        self.btn_export.clicked.connect(self._export_report)
        right_layout.addWidget(ctrl)

        param_group = QGroupBox("关键参数区")
        param_form = QFormLayout(param_group)
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
        pairs = [
            ("tolerance_px", "tolerance_px"),
            ("detect_retry", "detect_retry"),
            ("detect_retry_interval", "detect_retry_interval"),
            ("settle_time", "settle_time"),
            ("max_align_rounds", "max_align_rounds"),
            ("max_iterations", "max_iterations"),
            ("x_move_step", "x_move_step"),
            ("y_move_step", "y_move_step"),
            ("z_step", "z_step"),
            ("min_focus_score", "min_focus_score"),
            ("adaptive_step", "启用 adaptive step"),
            ("enable_recovery_scan", "启用 recovery scan"),
            ("startup_motion_check_enabled", "启用 startup check"),
            ("startup_motion_check_timeout", "startup timeout"),
            ("startup_motion_check_xy_steps", "startup xy steps"),
            ("startup_motion_check_z_step", "startup z step"),
        ]
        for key, label in pairs:
            widget = self.controls[key]
            text = label if isinstance(label, str) else key
            if text == key:
                text = key
            param_form.addRow(text, widget)
        right_layout.addWidget(param_group)

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

        layout.addWidget(right, 2)
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
        self.module_version_filter.addItems(["全部版本", "v2", "v3", "v4", "v5", "v6", "v7"])
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
        mrc_card = QGroupBox("Newport MRC 4-axis")
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
        self.controls["mrc_virtual_axis_mode"] = self._combo(["shared", "mirror1", "mirror2"])
        mrc_form.addRow("8742 conn", self.controls["newport_conn"])
        mrc_form.addRow("mirror1_x_axis", self.controls["mrc_mirror1_x_axis"])
        mrc_form.addRow("mirror1_y_axis", self.controls["mrc_mirror1_y_axis"])
        mrc_form.addRow("mirror2_x_axis", self.controls["mrc_mirror2_x_axis"])
        mrc_form.addRow("mirror2_y_axis", self.controls["mrc_mirror2_y_axis"])
        mrc_form.addRow("mirror1_x_sign", self.controls["mrc_mirror1_x_sign"])
        mrc_form.addRow("mirror1_y_sign", self.controls["mrc_mirror1_y_sign"])
        mrc_form.addRow("mirror2_x_sign", self.controls["mrc_mirror2_x_sign"])
        mrc_form.addRow("mirror2_y_sign", self.controls["mrc_mirror2_y_sign"])
        mrc_form.addRow("virtual_axis_mode", self.controls["mrc_virtual_axis_mode"])
        self.mrc_alloc_label = QLabel("-")
        mrc_form.addRow("allocation preview", self.mrc_alloc_label)
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
        z_form.addRow("z_picomotor_conn", self.controls["z_picomotor_conn"])
        z_form.addRow("z_picomotor_axis", self.controls["z_picomotor_axis"])
        z_form.addRow("z_picomotor_sign", self.controls["z_picomotor_sign"])
        z_form.addRow("velocity", self.controls["z_picomotor_velocity"])
        z_form.addRow("acceleration", self.controls["z_picomotor_acceleration"])
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
        btn_startup = QPushButton("手动触发 startup check")
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
        sim_layout.addRow("frame_source_image", browse_wrap)
        self.controls["sim_jitter_px"] = self._spin(0, 200, 0)
        self.controls["sim_noise_std"] = self._dspin(0.0, 100.0, 0.0, 2)
        sim_layout.addRow("sim_jitter_px", self.controls["sim_jitter_px"])
        sim_layout.addRow("sim_noise_std", self.controls["sim_noise_std"])
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

        cfg_group = QGroupBox("运行模式配置")
        cfg_form = QFormLayout(cfg_group)
        self.controls["detector_backend"] = self._combo(["auto", "yolo", "classic"])
        self.controls["xy_driver"] = self._combo(["dryrun", "thorlabs", "newport", "newport-mrc4"])
        self.controls["z_driver"] = self._combo(["dryrun", "wheel", "xps", "picomotor"])
        self.controls["select_roi"] = QCheckBox()
        cfg_form.addRow("detector backend", self.controls["detector_backend"])
        cfg_form.addRow("xy driver", self.controls["xy_driver"])
        cfg_form.addRow("z driver", self.controls["z_driver"])
        cfg_form.addRow("启用 ROI", self.controls["select_roi"])
        layout.addWidget(cfg_group, 1)

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
        self.sim_preview = QLabel("Simulation Preview")
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
            ("运动参数", [("newport_timeout", "newport timeout"), ("z_step", "z_step"), ("disable_run_lock", "禁用运行锁")]),
            ("ROI / 图像参数", [("sim_jitter_px", "sim_jitter_px"), ("sim_noise_std", "sim_noise_std"), ("select_roi", "启用 ROI")]),
            ("安全参数", [("startup_motion_check_timeout", "startup timeout"), ("startup_motion_check_xy_steps", "startup xy steps")]),
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
                    else:
                        continue
                form.addRow(label, self.controls[key])
            body_layout.addWidget(group)

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

    def _switch_page(self, idx: int) -> None:
        if hasattr(self, "pages") and 0 <= idx < self.pages.count():
            self.pages.setCurrentIndex(idx)

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
        self._append_log("已恢复默认配置值")

    def _apply_profile_to_controls(self, p: RuntimeProfile) -> None:
        self.mode_sim_radio.setChecked(p.run_mode == RunMode.SIMULATION)
        self.mode_real_radio.setChecked(p.run_mode == RunMode.REAL)

        self._set_combo("detector_backend", p.detector_backend)
        self._set_combo("xy_driver", p.xy_driver)
        self._set_combo("z_driver", p.z_driver)
        self._set_text("frame_source_image", p.frame_source_image or "")
        self._set_spin("sim_jitter_px", p.sim_jitter_px)
        self._set_dspin("sim_noise_std", p.sim_noise_std)
        self._set_check("select_roi", p.select_roi)
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

    def _collect_profile_from_controls(self) -> RuntimeProfile:
        p = self.profile
        p.run_mode = RunMode.REAL if self.mode_real_radio.isChecked() else RunMode.SIMULATION
        p.detector_backend = self._combo_value("detector_backend", p.detector_backend)
        p.xy_driver = self._combo_value("xy_driver", p.xy_driver)
        p.z_driver = self._combo_value("z_driver", p.z_driver)
        p.frame_source_image = self._text_value("frame_source_image", p.frame_source_image or "")
        p.sim_jitter_px = self._spin_value("sim_jitter_px", p.sim_jitter_px)
        p.sim_noise_std = self._dspin_value("sim_noise_std", p.sim_noise_std)
        p.select_roi = self._check_value("select_roi", p.select_roi)
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

    def _toggle_roi(self) -> None:
        current = self._check_value("select_roi", False)
        self._set_check("select_roi", not current)
        self._append_log(f"ROI 开关已切换: {'启用' if not current else '禁用'}")
        self._refresh_all_panels()

    def _run_selected_test(self) -> None:
        item = self.test_tree.currentItem()
        if item is None:
            return
        test_id = item.data(0, Qt.UserRole)
        if not test_id:
            return
        self._run_device_test_by_id(str(test_id))

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
            self.image_label.setText("图像加载失败")
            self.sim_preview.setText("图像加载失败")
            return
        pix = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pix.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.sim_preview.setPixmap(pix.scaled(self.sim_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

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




