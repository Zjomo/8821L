import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QDockWidget,
    QFormLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from annotation_module import ManualAnnotator
from recognition_module import (
    FrameResult,
    build_segmentation_view,
    classify_contours,
    contour_metrics,
    extract_contours,
    load_templates,
    preprocess_frame,
)

try:
    from PIL import ImageGrab  # type: ignore
except Exception:  # pragma: no cover
    ImageGrab = None


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def cv_to_qpixmap(frame: np.ndarray) -> QPixmap:
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(image.copy())


class SegmentationWindow(QMainWindow):
    def __init__(self, input_path: Optional[str] = None, output_dir: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("三对象识别 / 分割 / 标注")
        self.resize(1700, 980)

        self.input_path = Path(input_path) if input_path else None
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.template_path = self.output_dir / "templates.json"
        self.templates = load_templates(self.template_path)
        self.annotator = ManualAnnotator(self.template_path)

        self.capture = None
        self.source_type: Optional[str] = None
        self.frame_count = 0
        self.current_idx = 0
        self.paused = True
        self.recording = False
        self.roi: Optional[Tuple[int, int, int, int]] = None
        self.screen_roi: Optional[Tuple[int, int, int, int]] = None
        self.last_frame: Optional[np.ndarray] = None
        self.last_binary: Optional[np.ndarray] = None
        self.last_result: Optional[FrameResult] = None
        self.current_fps = 25.0
        self.seek_guard = False
        self.pending_seek = None
        self.last_update_ms = time.time()
        self.screen_cache: Optional[np.ndarray] = None
        self.screen_cache_ts = 0.0
        self.screen_cache_interval = 0.03
        self.auto_play = False

        self._build_ui()
        self._build_actions()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        if self.input_path is not None:
            self._load_source(self.input_path)

    def _icon(self, style_enum):
        return self.style().standardIcon(style_enum)

    def _build_actions(self):
        self.menu_bar = self.menuBar()
        file_menu = self.menu_bar.addMenu("文件")
        view_menu = self.menu_bar.addMenu("视图")
        help_menu = self.menu_bar.addMenu("帮助")

        self.open_action = QAction(self._icon(QStyle.SP_DialogOpenButton), "打开媒体", self)
        self.open_action.triggered.connect(self.open_file)
        file_menu.addAction(self.open_action)

        self.screen_action = QAction(self._icon(QStyle.SP_ComputerIcon), "屏幕模式", self)
        self.screen_action.triggered.connect(self.open_screen_mode)
        file_menu.addAction(self.screen_action)

        self.save_action = QAction(self._icon(QStyle.SP_DialogSaveButton), "保存快照", self)
        self.save_action.triggered.connect(self.save_snapshot)
        file_menu.addAction(self.save_action)

        file_menu.addSeparator()
        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        self.roi_action = QAction(self._icon(QStyle.SP_TitleBarMenuButton), "选择ROI", self)
        self.roi_action.triggered.connect(self.select_roi)
        view_menu.addAction(self.roi_action)

        self.annotate_action = QAction(self._icon(QStyle.SP_FileDialogDetailedView), "手动标注", self)
        self.annotate_action.triggered.connect(self.manual_annotate)
        view_menu.addAction(self.annotate_action)

        self.record_action = QAction(self._icon(QStyle.SP_DialogApplyButton), "开始记录", self)
        self.record_action.triggered.connect(self.toggle_record)
        view_menu.addAction(self.record_action)

        self.play_action = QAction(self._icon(QStyle.SP_MediaPlay), "播放/暂停", self)
        self.play_action.triggered.connect(self.toggle_pause)
        view_menu.addAction(self.play_action)

        help_action = QAction("使用说明", self)
        help_action.triggered.connect(self.show_help)
        help_menu.addAction(help_action)

        toolbar = QToolBar("主工具栏", self)
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.addToolBar(toolbar)

        for action in [self.open_action, self.screen_action, self.roi_action, self.annotate_action, self.record_action, self.play_action, self.save_action]:
            toolbar.addAction(action)


    def _build_ui(self):
        root = QWidget(self)
        self.setCentralWidget(root)

        self.original_label = QLabel("原图")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet("background:#111;color:#ccc;border:1px solid #444;")

        self.seg_label = QLabel("分割图")
        self.seg_label.setAlignment(Qt.AlignCenter)
        self.seg_label.setStyleSheet("background:#111;color:#ccc;border:1px solid #444;")

        center_row = QHBoxLayout()
        center_row.addWidget(self.original_label, 1)
        center_row.addWidget(self.seg_label, 1)

        layout = QVBoxLayout(root)
        layout.addLayout(center_row, 1)

        self._build_side_docks()

    def _build_side_docks(self):
        ctrl = QWidget(self)
        ctrl_layout = QVBoxLayout(ctrl)

        self.status_label = QLabel("未加载媒体")
        self.status_label.setWordWrap(True)
        ctrl_layout.addWidget(self.status_label)

        form = QFormLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderPressed.connect(self._on_slider_pressed)
        self.slider.sliderReleased.connect(self._on_slider_released)
        self.slider.sliderMoved.connect(self.on_seek)
        form.addRow("视频进度", self.slider)

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 120)
        self.fps_spin.setValue(25)
        self.fps_spin.valueChanged.connect(self._on_fps_changed)
        form.addRow("目标FPS", self.fps_spin)

        ctrl_layout.addLayout(form)

        button_row = QHBoxLayout()
        open_btn = QPushButton("打开")
        open_btn.clicked.connect(self.open_file)
        button_row.addWidget(open_btn)
        screen_btn = QPushButton("屏幕")
        screen_btn.clicked.connect(self.open_screen_mode)
        button_row.addWidget(screen_btn)
        ctrl_layout.addLayout(button_row)

        button_row2 = QHBoxLayout()
        roi_btn = QPushButton("ROI")
        roi_btn.clicked.connect(self.select_roi)
        button_row2.addWidget(roi_btn)
        annot_btn = QPushButton("标注")
        annot_btn.clicked.connect(self.manual_annotate)
        button_row2.addWidget(annot_btn)
        ctrl_layout.addLayout(button_row2)

        button_row3 = QHBoxLayout()
        play_btn = QPushButton("播放/暂停")
        play_btn.clicked.connect(self.toggle_pause)
        button_row3.addWidget(play_btn)
        record_btn = QPushButton("记录")
        record_btn.clicked.connect(self.toggle_record)
        button_row3.addWidget(record_btn)
        save_btn = QPushButton("快照")
        save_btn.clicked.connect(self.save_snapshot)
        button_row3.addWidget(save_btn)
        ctrl_layout.addLayout(button_row3)

        ctrl_layout.addStretch(1)
        ctrl_dock = QDockWidget("控制面板", self)
        ctrl_dock.setWidget(ctrl)
        ctrl_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, ctrl_dock)

        info = QWidget(self)
        info_layout = QVBoxLayout(info)
        self.info_table = QTableWidget(3, 8)
        self.info_table.setHorizontalHeaderLabels(["对象", "状态", "中心X", "中心Y", "角度", "置信度", "面积", "周长"])
        self.info_table.verticalHeader().setVisible(False)
        self.info_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.info_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.info_table.setSelectionMode(QTableWidget.NoSelection)
        self.info_table.setAlternatingRowColors(True)
        for row, name in enumerate(["圆底", "长条", "旋转体"]):
            self.info_table.setItem(row, 0, QTableWidgetItem(name))
            for col in range(1, 8):
                self.info_table.setItem(row, col, QTableWidgetItem("-"))

        info_layout.addWidget(self.info_table)
        info_layout.addStretch(1)
        info_dock = QDockWidget("识别结果", self)
        info_dock.setWidget(info)
        info_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, info_dock)

    def _on_fps_changed(self, value: int):
        self.current_fps = float(value)
        if self.source_type == "video" and not self.paused:
            self.timer.setInterval(max(10, int(1000 / max(1.0, self.current_fps))))

    def _on_slider_pressed(self):
        self.seek_guard = True

    def _on_slider_released(self):
        self.seek_guard = False
        if self.pending_seek is not None and self.capture is not None:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.pending_seek)
            self.current_idx = self.pending_seek
            self.paused = True
            self.pending_seek = None

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片/视频",
            str(self.input_path.parent if self.input_path else Path.cwd()),
            "媒体文件 (*.mp4 *.avi *.mov *.mkv *.wmv *.m4v *.jpg *.jpeg *.png *.bmp *.tif *.tiff)",
        )
        if not file_path:
            return
        self._load_source(Path(file_path))

    def open_screen_mode(self):
        self._load_source(Path("screen"))

    def _capture_screen(self) -> np.ndarray:
        if ImageGrab is None:
            raise RuntimeError("当前环境缺少 Pillow/ImageGrab，无法截屏。")
        img = ImageGrab.grab()
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def _load_source(self, path: Path):
        self.release_capture()
        self.source_type = None
        self.roi = None
        self.screen_roi = None
        self.current_idx = 0
        self.paused = True
        self.frame_count = 0
        self.pending_seek = None

        if path.name.lower() == "screen":
            self.source_type = "screen"
            self.last_frame = self._capture_screen()
            self.select_roi()
            self.status_label.setText("已进入屏幕模式")
            self.info_label.setText("屏幕模式已就绪")
            return

        suffix = path.suffix.lower()
        if suffix in VIDEO_EXTS:
            self.source_type = "video"
            self.capture = cv2.VideoCapture(str(path))
            if not self.capture.isOpened():
                QMessageBox.critical(self, "错误", f"无法打开视频: {path}")
                return
            self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            self.current_fps = float(self.capture.get(cv2.CAP_PROP_FPS) or self.fps_spin.value())
            self.fps_spin.setValue(max(1, int(round(self.current_fps))))
            self.timer.setInterval(max(10, int(1000 / max(1.0, self.current_fps))))
            ok, frame = self.capture.read()
            if not ok:
                QMessageBox.critical(self, "错误", "无法读取视频首帧")
                return
            self.last_frame = frame
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.slider.setEnabled(self.frame_count > 1)
            self.slider.setMaximum(max(0, self.frame_count - 1))
            self.select_roi()
            self.paused = False
            self.status_label.setText(f"已加载视频: {path.name}")
            self.info_label.setText("视频已加载，等待识别")
            return

        if suffix in IMG_EXTS:
            self.source_type = "image"
            frame = cv2.imread(str(path))
            if frame is None:
                QMessageBox.critical(self, "错误", f"无法读取图片: {path}")
                return
            self.last_frame = frame
            self.slider.setEnabled(False)
            self.select_roi()
            self.paused = True
            self.status_label.setText(f"已加载图片: {path.name}")
            self.info_label.setText("图片模式已就绪")
            return

        QMessageBox.warning(self, "提示", "不支持的输入类型")

    def release_capture(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def select_roi(self):
        if self.last_frame is None:
            return
        title = "选择屏幕ROI" if self.source_type == "screen" else "选择ROI"
        roi = cv2.selectROI(title, self.last_frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(title)
        if roi[2] <= 0 or roi[3] <= 0:
            self.roi = (0, 0, self.last_frame.shape[1], self.last_frame.shape[0])
        else:
            self.roi = tuple(map(int, roi))
        if self.source_type == "screen":
            self.screen_roi = self.roi
        self.status_label.setText(f"ROI={self.roi}")

    def current_frame(self) -> Optional[np.ndarray]:
        if self.source_type == "video":
            if self.capture is None:
                return None
            if self.paused:
                return self.last_frame
            if self.frame_count > 1 and self.seek_guard:
                return self.last_frame
            ok, frame = self.capture.read()
            if not ok:
                self.paused = True
                return self.last_frame
            self.current_idx = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            self.last_frame = frame
            return frame
        if self.source_type == "image":
            return self.last_frame
        if self.source_type == "screen":
            now = time.time()
            if now - self.screen_cache_ts >= self.screen_cache_interval:
                self.screen_cache = self._capture_screen()
                self.screen_cache_ts = now
            return self.screen_cache
        return None

    def crop_roi(self, frame: np.ndarray) -> np.ndarray:
        roi = self.screen_roi if self.source_type == "screen" else self.roi
        if roi is None:
            return frame
        x, y, w, h = roi
        return frame[max(0, y): max(0, y) + h, max(0, x): max(0, x) + w].copy()

    def analyze_frame(self, frame: np.ndarray) -> FrameResult:
        binary = preprocess_frame(frame)
        contours = extract_contours(frame, binary)
        detections = classify_contours(contours, frame.shape, self.templates)
        return FrameResult(detections=detections, binary=binary, contours=contours)

    def update_frame(self):
        if self.source_type is None:
            return
        frame = self.current_frame()
        if frame is None:
            return

        crop = self.crop_roi(frame)
        result = self.analyze_frame(crop)
        self.last_binary = result.binary
        self.last_result = result

        seg = build_segmentation_view(result.binary, result.detections)
        rod_angle_text = ""
        rod_det = result.detections.get("rod")
        if rod_det is not None and rod_det.angle is not None:
            cv2.line(seg, (0, seg.shape[0] // 2), (seg.shape[1], seg.shape[0] // 2), (255, 255, 255), 1)
            rod_angle_text = f"长条角度={rod_det.angle:.1f}°"
            cv2.putText(seg, f"rod angle={rod_det.angle:.1f} deg", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        status = [
            f"模式={self.source_type}",
            f"帧号={self.current_idx}",
            f"播放={'是' if not self.paused else '否'}",
            f"记录={'是' if self.recording else '否'}",
            f"ROI={self.roi}",
        ]
        if rod_angle_text:
            status.append(rod_angle_text)
        self.status_label.setText(" | ".join(status))

        rows = {
            "circle": {"name": "圆底", "det": result.detections.get("circle")},
            "rod": {"name": "长条", "det": result.detections.get("rod")},
            "rotator": {"name": "旋转体", "det": result.detections.get("rotator")},
        }
        for row, key in enumerate(["circle", "rod", "rotator"]):
            det = rows[key]["det"]
            self.info_table.item(row, 0).setText(rows[key]["name"])
            if det is None:
                for col in range(1, 5):
                    self.info_table.item(row, col).setText("-")
                continue
            self.info_table.item(row, 1).setText(f"{det.center[0]:.1f}")
            self.info_table.item(row, 2).setText(f"{det.center[1]:.1f}")
            self.info_table.item(row, 3).setText("-" if det.angle is None else f"{det.angle:.2f}")
            self.info_table.item(row, 4).setText(f"{det.score:.3f}")


        self.original_label.setPixmap(cv_to_qpixmap(crop).scaled(self.original_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.seg_label.setPixmap(cv_to_qpixmap(seg).scaled(self.seg_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        if self.source_type == "video" and self.frame_count > 1:
            self.seek_guard = True
            self.slider.setValue(max(0, min(self.current_idx, self.frame_count - 1)))
            self.seek_guard = False

    def show_help(self):
        QMessageBox.information(
            self,
            "使用说明",
            "打开媒体后先选 ROI，再点击手动标注。视频可拖动进度条定位，播放/暂停后仍可逐帧查看。",
        )

    def on_seek(self, value: int):

        if self.source_type != "video" or self.capture is None:
            return
        self.pending_seek = value
        if self.seek_guard:
            return
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, value)
        ok, frame = self.capture.read()
        if ok:
            self.current_idx = value
            self.last_frame = frame
            self.paused = True
            self.slider.setValue(value)
            self.update_frame()

    def toggle_pause(self):
        if self.source_type == "video":
            self.paused = not self.paused
            if not self.paused and self.capture is not None and self.pending_seek is not None:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.pending_seek)
                self.pending_seek = None

    def toggle_record(self):
        self.recording = not self.recording
        self.record_action.setText("停止记录" if self.recording else "开始记录")

    def save_snapshot(self):
        if self.last_frame is None or self.last_binary is None or self.last_result is None:
            return
        ts = int(time.time() * 1000)
        crop = self.crop_roi(self.last_frame)
        seg = build_segmentation_view(self.last_binary, self.last_result.detections)
        out_img = np.hstack([crop, seg])
        cv2.imwrite(str(self.output_dir / f"snapshot_{ts}.png"), out_img)
        with open(self.output_dir / f"snapshot_{ts}.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "roi": self.roi,
                    "source_type": self.source_type,
                    "detections": {k: asdict(v) if v is not None else None for k, v in self.last_result.detections.items()},
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        self.status_label.setText(f"已保存快照：{ts}")

    def manual_annotate(self):
        if self.last_frame is None or self.last_binary is None:
            QMessageBox.information(self, "提示", "请先加载媒体并完成 ROI 选择")
            return
        crop = self.crop_roi(self.last_frame)
        self.annotator.run(crop, self.last_binary)
        self.templates = load_templates(self.template_path)
        self.status_label.setText("模板已更新")

    def closeEvent(self, event):
        self.release_capture()
        super().closeEvent(event)
