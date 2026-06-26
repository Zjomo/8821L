import sys
import os
import glob
import time
import traceback
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QComboBox, QSpinBox,
    QTextBrowser, QProgressBar, QTabWidget, QFileDialog,
    QMessageBox, QGroupBox, QGridLayout, QListWidget, QScrollArea,
    QSizePolicy, QSplitter, QSlider, QCheckBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QPoint, QRect, QTimer
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QScreen, QGuiApplication

from ultralytics import YOLO
import torch

import dataset_utils

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def cv2_to_qpixmap(cv_img):
    if cv_img is None:
        return QPixmap()
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qt_img)

# ------------------------------------------------------------------
# 重定向 stdout 到 UI
# ------------------------------------------------------------------
class StreamRedirector(QObject):
    text_written = pyqtSignal(str)
    def write(self, text):
        if text:
            self.text_written.emit(text)
    def flush(self):
        pass

# ------------------------------------------------------------------
# 训练线程
# ------------------------------------------------------------------
class TrainThread(QThread):
    progress = pyqtSignal(int)
    finished_train = pyqtSignal(bool, str)

    def __init__(self, model_path, data_yaml, epochs, imgsz, batch, device, parent=None):
        super().__init__(parent)
        self.model_path = model_path
        self.data_yaml = data_yaml
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch = batch
        self.device = device

    def run(self):
        try:
            import torch
            dev = self.device
            if dev == "auto":
                dev = "0" if torch.cuda.is_available() else "cpu"
            model = YOLO(self.model_path)
            model.train(
                data=self.data_yaml,
                epochs=self.epochs,
                imgsz=self.imgsz,
                batch=self.batch,
                device=dev,
                project="runs/segment",
                name="gui_train",
                exist_ok=True,
                verbose=True,
            )
            best = os.path.join("runs", "segment", "gui_train", "weights", "best.pt")
            self.finished_train.emit(True, os.path.abspath(best))
        except Exception as e:
            self.finished_train.emit(False, str(e))

    def stop(self):
        self.terminate()

# ------------------------------------------------------------------
# 图片预测线程
# ------------------------------------------------------------------
class PredictThread(QThread):
    result_ready = pyqtSignal(list)  # [(orig_path, result_path), ...]
    debug_info = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, weights, sources, parent=None):
        super().__init__(parent)
        self.weights = weights
        self.sources = [os.path.abspath(s) for s in sources]

    def _cleanup_old_projects(self, base_dir, keep=3):
        try:
            import shutil
            dirs = sorted([d for d in os.listdir(base_dir) if d.startswith("predict_gui_") and os.path.isdir(os.path.join(base_dir, d))])
            for d in dirs[:-keep]:
                shutil.rmtree(os.path.join(base_dir, d))
        except Exception as e:
            self.debug_info.emit(f"清理旧目录失败: {e}")

    def run(self):
        try:
            model = YOLO(self.weights)
            import shutil
            base_tmp = r"E:\jupyter file\2_Optics\8821L\Utils\YoloUI\tmp_predict_gui"
            os.makedirs(base_tmp, exist_ok=True)
            self._cleanup_old_projects(base_tmp, keep=3)

            # 每次预测使用唯一时间戳目录，彻底避免命名冲突
            tmp_project = os.path.join(base_tmp, f"predict_gui_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time()*1000)%1000}")
            os.makedirs(tmp_project, exist_ok=True)

            self.debug_info.emit(f"预测临时目录: {tmp_project}")
            self.debug_info.emit(f"输入源: {self.sources}")

            results = model.predict(
                source=self.sources,
                save=True,
                project=tmp_project,
                name="predict",
                exist_ok=True,
                verbose=False,
                show_boxes=False,
                show_labels=False,
                show_conf=False,
            )

            # 取第一个结果对象里的实际保存目录
            if results and hasattr(results[0], 'save_dir'):
                save_dir = str(results[0].save_dir)
            else:
                save_dir = os.path.join(tmp_project, "predict")

            self.debug_info.emit(f"YOLO save_dir: {save_dir}")

            # 确保文件已写入
            if not os.path.exists(save_dir):
                time.sleep(0.2)

            out = []
            for s in self.sources:
                base = os.path.basename(s)
                cand = os.path.join(save_dir, base)
                if os.path.exists(cand):
                    out.append((s, cand))
                else:
                    # 回退：在 save_dir 下找同名（忽略大小写）或最近修改的图像文件
                    fallback = ""
                    try:
                        files = [f for f in os.listdir(save_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                        if files:
                            # 优先同名
                            matches = [f for f in files if f.lower() == base.lower()]
                            if matches:
                                fallback = os.path.join(save_dir, matches[0])
                            else:
                                fallback = os.path.join(save_dir, files[0])
                    except Exception as e:
                        self.debug_info.emit(f"回退查找失败: {e}")
                    out.append((s, fallback))
                    self.debug_info.emit(f"回退查找: {s} -> {fallback}")

            self.result_ready.emit(out)
        except Exception as e:
            import traceback
            self.error.emit(str(e) + "\n" + traceback.format_exc())

# ------------------------------------------------------------------
# 视频推理线程
# ------------------------------------------------------------------
class VideoThread(QThread):
    frame_ready = pyqtSignal(object, object, int, int)  # orig, result, frame_idx, total_frames
    finished_video = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, weights, video_path, device="cpu", start_frame=0, parent=None):
        super().__init__(parent)
        self.weights = weights
        self.video_path = video_path
        self.device = device
        self.start_frame = start_frame
        self._running = True
        self._paused = False
        self._target_frame = None
        self._lock = False

    def run(self):
        try:
            model = YOLO(self.weights)
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error.emit("无法打开视频文件")
                return
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            frame_idx = self.start_frame
            while self._running and frame_idx < total:
                while self._paused and self._running:
                    time.sleep(0.05)
                    if self._target_frame is not None and not self._lock:
                        frame_idx = self._target_frame
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        self._target_frame = None
                if not self._running:
                    break
                if self._target_frame is not None and not self._lock:
                    frame_idx = self._target_frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    self._target_frame = None
                ret, frame = cap.read()
                if not ret:
                    break
                orig = frame.copy()
                result = model.predict(source=frame, device=self.device, verbose=False, show_boxes=False, show_labels=False, show_conf=False)[0]
                plotted = result.plot(boxes=False, labels=False, probs=False)
                self.frame_ready.emit(orig, plotted, frame_idx, total)
                frame_idx += 1
            cap.release()
            self.finished_video.emit()
        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        self._running = False

    def pause(self, paused):
        self._paused = paused

    def seek(self, frame_idx):
        self._target_frame = frame_idx

# ------------------------------------------------------------------
# ROI 实时屏幕分割线程
# ------------------------------------------------------------------
class ROIScreenThread(QThread):
    frame_ready = pyqtSignal(object, object)  # original_cv_img, result_cv_img
    finished_screen = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, weights, roi_rect, device="cpu", fps=10.0, parent=None):
        super().__init__(parent)
        self.weights = weights
        self.roi_rect = roi_rect  # (x, y, w, h)
        self.device = device
        self.fps = fps
        self._running = True
        self._paused = False

    def run(self):
        try:
            import pyautogui
            model = YOLO(self.weights)
            interval_ms = max(50, int(1000 / self.fps))
            while self._running:
                while self._paused and self._running:
                    time.sleep(0.05)
                if not self._running:
                    break
                x, y, w, h = self.roi_rect
                try:
                    screenshot = pyautogui.screenshot(region=(x, y, w, h))
                    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    self.error.emit(f"截屏失败: {e}")
                    time.sleep(interval_ms / 1000.0)
                    continue
                orig = frame.copy()
                result = model.predict(source=frame, device=self.device, verbose=False, show_boxes=False, show_labels=False, show_conf=False)[0]
                plotted = result.plot(boxes=False, labels=False, probs=False)
                self.frame_ready.emit(orig, plotted)
                time.sleep(interval_ms / 1000.0)
        except Exception as e:
            import traceback
            self.error.emit(str(e) + "\n" + traceback.format_exc())
        self.finished_screen.emit()

    def stop(self):
        self._running = False

    def pause(self, paused):
        self._paused = paused

# ------------------------------------------------------------------
# ROI 选择标签
# ------------------------------------------------------------------
class ROILabel(QLabel):
    roi_selected = pyqtSignal(QRect)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._roi = QRect()
        self._drawing = False
        self._start = QPoint()
        self._pixmap = QPixmap()

    def setPixmap(self, pm: QPixmap):
        self._pixmap = pm
        super().setPixmap(pm)
        self._roi = QRect()

    def mousePressEvent(self, event):
        if self._pixmap.isNull():
            return
        self._drawing = True
        self._start = event.pos()
        self._roi = QRect()

    def mouseMoveEvent(self, event):
        if self._drawing:
            self._roi = QRect(self._start, event.pos()).normalized()
            self.update()

    def mouseReleaseEvent(self, event):
        if self._drawing:
            self._drawing = False
            self.roi_selected.emit(self._roi)

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._roi.isNull() and self._roi.width() > 2 and self._roi.height() > 2:
            painter = QPainter(self)
            pen = QPen(Qt.GlobalColor.red, 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(self._roi)

    def get_roi(self):
        return self._roi

    def clear_roi(self):
        self._roi = QRect()
        self.update()

# ------------------------------------------------------------------
# 主窗口
# ------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv8 语义分割工具")
        self.setMinimumSize(1400, 900)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self._build_dataset_tab()
        self._build_train_tab()
        self._build_predict_tab()
        self._build_eval_tab()

        self.redirector = StreamRedirector()
        self.redirector.text_written.connect(self._append_log)

        completed_best = r"E:\jupyter file\2_Optics\8821L\Utils\YoloUI\runs\segment\fore_back_70\weights\best.pt"
        for le in [self.le_predict_weights, self.le_video_weights, self.le_roi_weights]:
            if os.path.exists(completed_best):
                le.setText(completed_best)

    # ======================== 通用 ========================
    def _browse_file(self, line_edit, caption, filter_):
        path, _ = QFileDialog.getOpenFileName(self, caption, "", filter_)
        if path:
            line_edit.setText(path)

    def _browse_dir(self, line_edit, caption):
        path = QFileDialog.getExistingDirectory(self, caption)
        if path:
            line_edit.setText(path)

    def _append_log(self, text):
        self.tb_log.moveCursor(self.tb_log.textCursor().MoveOperation.End)
        self.tb_log.insertPlainText(text)
        self.tb_log.ensureCursorVisible()

    def _make_image_label(self, title=""):
        lbl = QLabel()
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setMinimumSize(400, 300)
        lbl.setStyleSheet("background-color: #e0e0e0; border: 1px solid #aaa;")
        if title:
            lbl.setText(title)
        return lbl

    # ======================== 训练页 ========================
    def _build_dataset_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        gb = QGroupBox("数据集准备")
        gl = QGridLayout(gb)

        gl.addWidget(QLabel("图片目录:"), 0, 0)
        self.le_ds_img = QLineEdit()
        gl.addWidget(self.le_ds_img, 0, 1)
        btn = QPushButton("浏览...")
        btn.clicked.connect(lambda: self._browse_dir(self.le_ds_img, "选择图片目录"))
        gl.addWidget(btn, 0, 2)

        gl.addWidget(QLabel("标注目录:"), 1, 0)
        self.le_ds_lbl = QLineEdit()
        gl.addWidget(self.le_ds_lbl, 1, 1)
        btn2 = QPushButton("浏览...")
        btn2.clicked.connect(lambda: self._browse_dir(self.le_ds_lbl, "选择标注目录"))
        gl.addWidget(btn2, 1, 2)

        gl.addWidget(QLabel("数据格式:"), 2, 0)
        self.cb_ds_fmt = QComboBox()
        self.cb_ds_fmt.addItems(["auto", "yolo_seg", "yolo_det", "yolo_11", "coco"])
        gl.addWidget(self.cb_ds_fmt, 2, 1)

        gl.addWidget(QLabel("输出目录:"), 3, 0)
        self.le_ds_out = QLineEdit()
        self.le_ds_out.setText(r"E:\jupyter file\2_Optics\8821L\Utils\YoloUI\Dataset\my_dataset")
        gl.addWidget(self.le_ds_out, 3, 1)
        btn3 = QPushButton("浏览...")
        btn3.clicked.connect(lambda: self._browse_dir(self.le_ds_out, "选择输出目录"))
        gl.addWidget(btn3, 3, 2)

        gl.addWidget(QLabel("Train 比例:"), 4, 0)
        self.sb_ds_train = QSpinBox(); self.sb_ds_train.setRange(1, 99); self.sb_ds_train.setValue(70); gl.addWidget(self.sb_ds_train, 4, 1)
        gl.addWidget(QLabel("Val 比例:"), 4, 2)
        self.sb_ds_val = QSpinBox(); self.sb_ds_val.setRange(0, 99); self.sb_ds_val.setValue(20); gl.addWidget(self.sb_ds_val, 4, 3)
        gl.addWidget(QLabel("Test 比例:"), 5, 0)
        self.sb_ds_test = QSpinBox(); self.sb_ds_test.setRange(0, 99); self.sb_ds_test.setValue(10); gl.addWidget(self.sb_ds_test, 5, 1)

        layout.addWidget(gb)

        hbtn = QHBoxLayout()
        self.btn_ds_prepare = QPushButton("开始准备数据集")
        self.btn_ds_prepare.setStyleSheet("font-size: 16px; padding: 8px;")
        self.btn_ds_prepare.clicked.connect(self._prepare_dataset)
        hbtn.addWidget(self.btn_ds_prepare)
        self.btn_ds_use = QPushButton("应用到训练")
        self.btn_ds_use.setEnabled(False)
        self.btn_ds_use.clicked.connect(self._apply_dataset_to_train)
        hbtn.addWidget(self.btn_ds_use)
        layout.addLayout(hbtn)

        self.lbl_ds_result = QLabel("结果: 未开始")
        layout.addWidget(self.lbl_ds_result)

        self.tb_ds_log = QTextBrowser()
        self.tb_ds_log.setMinimumHeight(250)
        layout.addWidget(self.tb_ds_log)
        layout.addStretch()
        self.tabs.addTab(w, "数据集准备")

    def _prepare_dataset(self):
        img_dir = self.le_ds_img.text()
        lbl_dir = self.le_ds_lbl.text()
        out_dir = self.le_ds_out.text()
        fmt = self.cb_ds_fmt.currentText()
        train_r = self.sb_ds_train.value() / 100.0
        val_r = self.sb_ds_val.value() / 100.0
        test_r = self.sb_ds_test.value() / 100.0

        if not img_dir or not os.path.isdir(img_dir):
            QMessageBox.warning(self, "提示", "请选择有效的图片目录"); return
        if not lbl_dir or not os.path.isdir(lbl_dir):
            QMessageBox.warning(self, "提示", "请选择有效的标注目录"); return
        if abs(train_r + val_r + test_r - 1.0) > 1e-6:
            QMessageBox.warning(self, "提示", "Train + Val + Test 比例之和必须等于 100"); return

        try:
            self.tb_ds_log.clear()
            self.tb_ds_log.append(f"开始解析: {img_dir}")
            self.tb_ds_log.append(f"格式: {fmt}")
            yaml_path, info = dataset_utils.build_yolo_dataset(
                img_dir, lbl_dir, out_dir, fmt=fmt,
                train_ratio=train_r, val_ratio=val_r, test_ratio=test_r,
                force_val_from_train=True
            )
            self.tb_ds_log.append(f"准备完成: {yaml_path}")
            self.tb_ds_log.append(f"划分: {info}")
            self.lbl_ds_result.setText(f"准备完成: {yaml_path}  |  {info}")
            self.btn_ds_use.setEnabled(True)
            self._last_prepared_dataset = out_dir
        except Exception as e:
            self.tb_ds_log.append(f"错误: {str(e)}")
            QMessageBox.critical(self, "数据集准备错误", str(e))

    def _apply_dataset_to_train(self):
        if hasattr(self, '_last_prepared_dataset') and self._last_prepared_dataset:
            self.le_data.setText(self._last_prepared_dataset)
            self.tabs.setCurrentIndex(1)  # 切换到模型训练页

    def _build_train_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        gb_model = QGroupBox("模型选择")
        gl = QGridLayout(gb_model)
        gl.addWidget(QLabel("预训练模型:"), 0, 0)
        self.cb_model = QComboBox()
        self.cb_model.addItems(["yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt", "yolov8l-seg.pt", "自定义"])
        self.cb_model.currentTextChanged.connect(self._on_model_change)
        gl.addWidget(self.cb_model, 0, 1)
        gl.addWidget(QLabel("自定义路径:"), 1, 0)
        self.le_custom_model = QLineEdit()
        self.le_custom_model.setEnabled(False)
        gl.addWidget(self.le_custom_model, 1, 1)
        btn = QPushButton("浏览...")
        btn.clicked.connect(lambda: self._browse_file(self.le_custom_model, "选择权重", "*.pt"))
        gl.addWidget(btn, 1, 2)
        layout.addWidget(gb_model)

        gb_data = QGroupBox("数据集")
        gl2 = QGridLayout(gb_data)
        gl2.addWidget(QLabel("数据集目录 (含 data.yaml):"), 0, 0)
        self.le_data = QLineEdit()
        default = r"E:\jupyter file\2_Optics\8821L\Utils\YoloUI\Fore_BackGround_70_export"
        self.le_data.setText(default)
        gl2.addWidget(self.le_data, 0, 1)
        btn2 = QPushButton("浏览...")
        btn2.clicked.connect(lambda: self._browse_dir(self.le_data, "选择数据集根目录"))
        gl2.addWidget(btn2, 0, 2)
        layout.addWidget(gb_data)

        gb_hyp = QGroupBox("训练参数")
        gl3 = QGridLayout(gb_hyp)
        gl3.addWidget(QLabel("Epochs:"), 0, 0)
        self.sb_epochs = QSpinBox(); self.sb_epochs.setRange(1, 1000); self.sb_epochs.setValue(100); gl3.addWidget(self.sb_epochs, 0, 1)
        gl3.addWidget(QLabel("Image Size:"), 0, 2)
        self.sb_imgsz = QSpinBox(); self.sb_imgsz.setRange(32, 2048); self.sb_imgsz.setValue(640); gl3.addWidget(self.sb_imgsz, 0, 3)
        gl3.addWidget(QLabel("Batch:"), 1, 0)
        self.sb_batch = QSpinBox(); self.sb_batch.setRange(1, 64); self.sb_batch.setValue(8); gl3.addWidget(self.sb_batch, 1, 1)
        gl3.addWidget(QLabel("Device:"), 1, 2)
        self.cb_device = QComboBox(); self.cb_device.addItems(["auto", "cpu", "0"]); gl3.addWidget(self.cb_device, 1, 3)
        gl3.addWidget(QLabel("无 val 自动拆分:"), 2, 0)
        self.chk_auto_val = QCheckBox("从 train 划分 15% 作为 val")
        self.chk_auto_val.setChecked(True)
        gl3.addWidget(self.chk_auto_val, 2, 1)
        layout.addWidget(gb_hyp)

        hbtn = QHBoxLayout()
        self.btn_start = QPushButton("开始训练")
        self.btn_start.setStyleSheet("font-size: 16px; padding: 8px;")
        self.btn_start.clicked.connect(self._start_train)
        hbtn.addWidget(self.btn_start)
        self.btn_stop = QPushButton("停止训练")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_train)
        hbtn.addWidget(self.btn_stop)
        layout.addLayout(hbtn)

        self.pb_train = QProgressBar()
        self.pb_train.setRange(0, 100)
        self.pb_train.setValue(0)
        layout.addWidget(self.pb_train)

        self.tb_log = QTextBrowser()
        self.tb_log.setMinimumHeight(250)
        layout.addWidget(self.tb_log)
        layout.addStretch()
        self.tabs.addTab(w, "模型训练")

    def _on_model_change(self, text):
        self.le_custom_model.setEnabled(text == "自定义")

    def _start_train(self):
        model_text = self.cb_model.currentText()
        if model_text == "自定义":
            model_path = self.le_custom_model.text()
            if not model_path:
                QMessageBox.warning(self, "提示", "请选择自定义模型权重路径"); return
        else:
            model_path = model_text
        data_yaml = os.path.join(self.le_data.text(), "data.yaml")
        if not os.path.exists(data_yaml):
            QMessageBox.warning(self, "提示", f"未找到 data.yaml:\n{data_yaml}"); return

        # 检查 val 并自动拆分
        if not dataset_utils.has_val_in_yaml(self.le_data.text()):
            if self.chk_auto_val.isChecked():
                ok = dataset_utils.ensure_val_split(self.le_data.text(), val_ratio=0.15)
                if not ok:
                    QMessageBox.critical(self, "错误", "数据集中没有 val，且无法从 train 自动拆分"); return
                QMessageBox.information(self, "提示", "已自动从 train 拆分 15% 作为 val")
            else:
                QMessageBox.warning(self, "提示", "数据集中没有 val 验证集，请勾选\"无 val 自动拆分\"或先在\"数据集准备\"页生成数据集"); return

        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True); self.pb_train.setValue(0); self.tb_log.clear()
        sys.stdout = self.redirector
        self.train_thread = TrainThread(model_path, data_yaml, self.sb_epochs.value(), self.sb_imgsz.value(), self.sb_batch.value(), self.cb_device.currentText())
        self.train_thread.finished_train.connect(self._on_train_finished)
        self.train_thread.start()

    def _stop_train(self):
        if self.train_thread: self.train_thread.stop()
        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False); sys.stdout = sys.__stdout__

    def _on_train_finished(self, success, msg):
        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False); sys.stdout = sys.__stdout__
        if success:
            self.pb_train.setValue(100)
            QMessageBox.information(self, "完成", f"训练完成！\nBest: {msg}")
            for le in [self.le_predict_weights, self.le_video_weights, self.le_roi_weights]:
                le.setText(msg)
        else:
            QMessageBox.critical(self, "错误", f"训练失败:\n{msg}")

    # ======================== 预测页 ========================
    def _build_predict_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        sub_tabs = QTabWidget()
        sub_tabs.addTab(self._build_img_predict_widget(), "图片预测")
        sub_tabs.addTab(self._build_video_predict_widget(), "视频预测")
        sub_tabs.addTab(self._build_roi_predict_widget(), "ROI 分割")
        layout.addWidget(sub_tabs)
        self.tabs.addTab(w, "模型预测")

    # ---- 图片预测 ----
    def _build_img_predict_widget(self):
        w = QWidget()
        vl = QVBoxLayout(w)

        gb = QGroupBox("配置")
        gl = QGridLayout(gb)
        gl.addWidget(QLabel("权重路径:"), 0, 0)
        self.le_predict_weights = QLineEdit()
        gl.addWidget(self.le_predict_weights, 0, 1)
        btn_w = QPushButton("浏览...")
        btn_w.clicked.connect(lambda: self._browse_file(self.le_predict_weights, "选择权重", "*.pt"))
        gl.addWidget(btn_w, 0, 2)

        gl.addWidget(QLabel("图片/文件夹:"), 1, 0)
        self.le_predict_src = QLineEdit()
        gl.addWidget(self.le_predict_src, 1, 1)
        btn_src = QPushButton("浏览...")
        btn_src.clicked.connect(self._browse_predict_src)
        gl.addWidget(btn_src, 1, 2)
        vl.addWidget(gb)

        hbtn = QHBoxLayout()
        self.btn_predict = QPushButton("开始预测")
        self.btn_predict.setStyleSheet("font-size: 16px; padding: 8px;")
        self.btn_predict.clicked.connect(self._start_predict)
        hbtn.addWidget(self.btn_predict)
        vl.addLayout(hbtn)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        scroll_orig = QScrollArea(); scroll_orig.setWidgetResizable(True)
        self.img_orig_container = QWidget()
        self.img_orig_layout = QVBoxLayout(self.img_orig_container)
        scroll_orig.setWidget(self.img_orig_container)
        splitter.addWidget(scroll_orig)

        scroll_result = QScrollArea(); scroll_result.setWidgetResizable(True)
        self.img_result_container = QWidget()
        self.img_result_layout = QVBoxLayout(self.img_result_container)
        scroll_result.setWidget(self.img_result_container)
        splitter.addWidget(scroll_result)
        splitter.setSizes([600, 600])
        vl.addWidget(splitter)
        return w

    def _browse_predict_src(self):
        files, _ = QFileDialog.getOpenFileNames(self, "选择图片", "", "Images (*.jpg *.jpeg *.png *.bmp)")
        if files:
            self.le_predict_src.setText(";".join(files))
        else:
            folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
            if folder:
                self.le_predict_src.setText(folder)

    def _clear_img_results(self):
        for layout in [self.img_orig_layout, self.img_result_layout]:
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

    def _start_predict(self):
        weights = self.le_predict_weights.text()
        src = self.le_predict_src.text()
        if not weights or not os.path.exists(weights):
            QMessageBox.warning(self, "提示", "请选择有效的权重文件"); return
        if not src:
            QMessageBox.warning(self, "提示", "请选择图片或文件夹"); return
        if ";" in src:
            sources = src.split(";")
        elif os.path.isdir(src):
            sources = sorted(glob.glob(os.path.join(src, "*.jpg")) + glob.glob(os.path.join(src, "*.png")))
        else:
            sources = [src]
        if not sources:
            QMessageBox.warning(self, "提示", "未找到可预测的图片"); return

        if getattr(self, '_predict_running', False):
            QMessageBox.warning(self, "提示", "预测正在进行中，请等待完成")
            return
        self._predict_running = True
        self.btn_predict.setEnabled(False)
        self._clear_img_results()
        self.predict_thread = PredictThread(weights, sources)
        self.predict_thread.result_ready.connect(self._on_predict_result)
        self.predict_thread.debug_info.connect(self._append_log)
        self.predict_thread.error.connect(self._on_predict_error)
        self.predict_thread.start()

    def _on_predict_result(self, pairs):
        self._predict_running = False
        self.btn_predict.setEnabled(True)
        for orig, result in pairs:
            lbl_o = QLabel()
            pm_o = QPixmap(orig)
            if not pm_o.isNull():
                pm_o = pm_o.scaledToWidth(500, Qt.TransformationMode.SmoothTransformation)
                lbl_o.setPixmap(pm_o)
            lbl_o.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.img_orig_layout.addWidget(QLabel(os.path.basename(orig)))
            self.img_orig_layout.addWidget(lbl_o)

            lbl_r = QLabel()
            if result and os.path.exists(result):
                pm_r = QPixmap(result)
                if not pm_r.isNull():
                    pm_r = pm_r.scaledToWidth(500, Qt.TransformationMode.SmoothTransformation)
                    lbl_r.setPixmap(pm_r)
                else:
                    lbl_r.setText("分割图加载失败")
            else:
                lbl_r.setText(f"分割图未生成\n(result={result})")
            lbl_r.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.img_result_layout.addWidget(QLabel("分割结果"))
            self.img_result_layout.addWidget(lbl_r)
        self.img_orig_layout.addStretch()
        self.img_result_layout.addStretch()

    def _on_predict_error(self, msg):
        self._predict_running = False
        self.btn_predict.setEnabled(True)
        QMessageBox.critical(self, "预测错误", msg)

    # ---- 视频预测 ----
    def _build_video_predict_widget(self):
        w = QWidget()
        vl = QVBoxLayout(w)

        gb = QGroupBox("配置")
        gl = QGridLayout(gb)
        gl.addWidget(QLabel("权重路径:"), 0, 0)
        self.le_video_weights = QLineEdit()
        gl.addWidget(self.le_video_weights, 0, 1)
        btn_w = QPushButton("浏览...")
        btn_w.clicked.connect(lambda: self._browse_file(self.le_video_weights, "选择权重", "*.pt"))
        gl.addWidget(btn_w, 0, 2)

        gl.addWidget(QLabel("视频文件:"), 1, 0)
        self.le_video_src = QLineEdit()
        gl.addWidget(self.le_video_src, 1, 1)
        btn_src = QPushButton("浏览...")
        btn_src.clicked.connect(lambda: self._browse_file(self.le_video_src, "选择视频", "*.mp4 *.avi *.mkv"))
        gl.addWidget(btn_src, 1, 2)
        vl.addWidget(gb)

        hbtn = QHBoxLayout()
        self.btn_video_start = QPushButton("开始视频分割")
        self.btn_video_start.setStyleSheet("font-size: 16px; padding: 8px;")
        self.btn_video_start.clicked.connect(self._start_video)
        hbtn.addWidget(self.btn_video_start)
        self.btn_video_pause = QPushButton("暂停")
        self.btn_video_pause.setEnabled(False)
        self.btn_video_pause.clicked.connect(self._toggle_video_pause)
        hbtn.addWidget(self.btn_video_pause)
        self.btn_video_stop = QPushButton("停止")
        self.btn_video_stop.setEnabled(False)
        self.btn_video_stop.clicked.connect(self._stop_video)
        hbtn.addWidget(self.btn_video_stop)
        vl.addLayout(hbtn)

        self.slider_video = QSlider(Qt.Orientation.Horizontal)
        self.slider_video.setRange(0, 0)
        self.slider_video.setEnabled(False)
        self.slider_video.sliderPressed.connect(self._video_slider_pressed)
        self.slider_video.sliderReleased.connect(self._video_slider_released)
        self.slider_video.valueChanged.connect(self._video_slider_moved)
        vl.addWidget(self.slider_video)

        self.lbl_video_progress = QLabel("帧: 0 / 0")
        vl.addWidget(self.lbl_video_progress)

        h = QHBoxLayout()
        h.addWidget(QLabel("原视频帧"))
        h.addWidget(QLabel("分割结果"))
        vl.addLayout(h)

        h2 = QHBoxLayout()
        self.lbl_video_orig = self._make_image_label("原视频帧")
        self.lbl_video_result = self._make_image_label("分割结果")
        h2.addWidget(self.lbl_video_orig)
        h2.addWidget(self.lbl_video_result)
        vl.addLayout(h2)
        vl.addStretch()
        return w

    def _start_video(self):
        weights = self.le_video_weights.text()
        video = self.le_video_src.text()
        if not weights or not os.path.exists(weights):
            QMessageBox.warning(self, "提示", "请选择有效的权重文件"); return
        if not video or not os.path.exists(video):
            QMessageBox.warning(self, "提示", "请选择有效的视频文件"); return

        cap = cv2.VideoCapture(video)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        self.slider_video.setRange(0, max(0, total - 1))
        self.slider_video.setValue(0)
        self.slider_video.setEnabled(total > 0)
        self.lbl_video_progress.setText(f"帧: 0 / {total}")

        self.btn_video_start.setEnabled(False)
        self.btn_video_pause.setEnabled(True)
        self.btn_video_pause.setText("暂停")
        self.btn_video_stop.setEnabled(True)
        self.video_thread = VideoThread(weights, video, self.cb_device.currentText())
        self.video_thread.frame_ready.connect(self._on_video_frame)
        self.video_thread.finished_video.connect(self._on_video_finished)
        self.video_thread.error.connect(self._on_video_error)
        self.video_thread.start()

    def _toggle_video_pause(self):
        if not self.video_thread:
            return
        if self.video_thread._paused:
            self.video_thread.pause(False)
            self.btn_video_pause.setText("暂停")
        else:
            self.video_thread.pause(True)
            self.btn_video_pause.setText("继续")

    def _stop_video(self):
        if self.video_thread:
            self.video_thread.stop()
        self.btn_video_start.setEnabled(True)
        self.btn_video_pause.setEnabled(False)
        self.btn_video_stop.setEnabled(False)
        self.slider_video.setEnabled(False)

    def _video_slider_pressed(self):
        if self.video_thread:
            self.video_thread._lock = True

    def _video_slider_released(self):
        if self.video_thread:
            self.video_thread._lock = False
            self.video_thread.seek(self.slider_video.value())

    def _video_slider_moved(self, value):
        self.lbl_video_progress.setText(f"帧: {value} / {self.slider_video.maximum() + 1}")

    def _on_video_frame(self, orig_cv, result_cv, frame_idx, total):
        self.lbl_video_orig.setPixmap(cv2_to_qpixmap(orig_cv).scaledToWidth(600, Qt.TransformationMode.SmoothTransformation))
        self.lbl_video_result.setPixmap(cv2_to_qpixmap(result_cv).scaledToWidth(600, Qt.TransformationMode.SmoothTransformation))
        self.slider_video.blockSignals(True)
        self.slider_video.setValue(frame_idx)
        self.slider_video.blockSignals(False)
        self.lbl_video_progress.setText(f"帧: {frame_idx + 1} / {total}")

    def _on_video_finished(self):
        self.btn_video_start.setEnabled(True)
        self.btn_video_pause.setEnabled(False)
        self.btn_video_stop.setEnabled(False)

    def _on_video_error(self, msg):
        self.btn_video_start.setEnabled(True)
        self.btn_video_pause.setEnabled(False)
        self.btn_video_stop.setEnabled(False)
        self.slider_video.setEnabled(False)
        QMessageBox.critical(self, "视频处理错误", msg)

    # ---- ROI 分割 ----
    def _build_roi_predict_widget(self):
        w = QWidget()
        vl = QVBoxLayout(w)

        gb = QGroupBox("配置")
        gl = QGridLayout(gb)
        gl.addWidget(QLabel("权重路径:"), 0, 0)
        self.le_roi_weights = QLineEdit()
        gl.addWidget(self.le_roi_weights, 0, 1)
        btn_w = QPushButton("浏览...")
        btn_w.clicked.connect(lambda: self._browse_file(self.le_roi_weights, "选择权重", "*.pt"))
        gl.addWidget(btn_w, 0, 2)
        vl.addWidget(gb)

        hbtn = QHBoxLayout()
        self.btn_roi_capture = QPushButton("截取当前屏幕")
        self.btn_roi_capture.setStyleSheet("font-size: 16px; padding: 8px;")
        self.btn_roi_capture.clicked.connect(self._capture_screen)
        hbtn.addWidget(self.btn_roi_capture)
        self.btn_roi_clear = QPushButton("清除 ROI")
        self.btn_roi_clear.clicked.connect(self._clear_roi)
        hbtn.addWidget(self.btn_roi_clear)
        self.btn_roi_run = QPushButton("对 ROI 进行分割")
        self.btn_roi_run.setStyleSheet("font-size: 16px; padding: 8px;")
        self.btn_roi_run.clicked.connect(self._run_roi_seg)
        hbtn.addWidget(self.btn_roi_run)
        vl.addLayout(hbtn)

        hbtn2 = QHBoxLayout()
        self.btn_roi_screen_start = QPushButton("开始实时屏幕分割")
        self.btn_roi_screen_start.setStyleSheet("font-size: 16px; padding: 8px;")
        self.btn_roi_screen_start.clicked.connect(self._start_roi_screen)
        hbtn2.addWidget(self.btn_roi_screen_start)
        self.btn_roi_screen_pause = QPushButton("暂停")
        self.btn_roi_screen_pause.setEnabled(False)
        self.btn_roi_screen_pause.clicked.connect(self._toggle_roi_screen_pause)
        hbtn2.addWidget(self.btn_roi_screen_pause)
        self.btn_roi_screen_stop = QPushButton("停止")
        self.btn_roi_screen_stop.setEnabled(False)
        self.btn_roi_screen_stop.clicked.connect(self._stop_roi_screen)
        hbtn2.addWidget(self.btn_roi_screen_stop)
        vl.addLayout(hbtn2)

        self.lbl_roi_info = QLabel('点击"截取当前屏幕"，然后在左侧图片上拖拽选择 ROI 区域')
        vl.addWidget(self.lbl_roi_info)

        h = QHBoxLayout()
        self.lbl_roi_select = ROILabel()
        self.lbl_roi_select.setMinimumSize(500, 400)
        self.lbl_roi_select.setStyleSheet("background-color: #e0e0e0; border: 1px solid #aaa;")
        self.lbl_roi_select.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_roi_select.roi_selected.connect(self._on_roi_selected)
        h.addWidget(self.lbl_roi_select)

        right = QVBoxLayout()
        self.lbl_roi_result = self._make_image_label("ROI 单张分割结果")
        right.addWidget(self.lbl_roi_result)
        self.lbl_roi_screen_result = self._make_image_label("ROI 实时屏幕分割结果")
        right.addWidget(self.lbl_roi_screen_result)
        h.addLayout(right)
        vl.addLayout(h)
        vl.addStretch()
        return w

    def _capture_screen(self):
        self.hide()
        QApplication.processEvents()
        time.sleep(0.5)
        screen = QGuiApplication.primaryScreen()
        pix = screen.grabWindow(0)
        self.show()
        self._roi_pixmap = pix
        self._roi_image_cv = self._qpixmap_to_cv(pix)
        scaled = pix.scaled(self.lbl_roi_select.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.lbl_roi_select.setPixmap(scaled)
        self.lbl_roi_select.clear_roi()
        self.lbl_roi_result.setText("ROI 单张分割结果")
        self.lbl_roi_screen_result.setText("ROI 实时屏幕分割结果")
        self.lbl_roi_info.setText(f"已截取屏幕: {pix.width()}x{pix.height()}  请拖拽鼠标选择 ROI")

    def _qpixmap_to_cv(self, pixmap):
        qimg = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
        w, h = qimg.width(), qimg.height()
        ptr = qimg.bits()
        ptr.setsize(h * w * 3)
        arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 3))
        return cv2.cvtColor(arr.copy(), cv2.COLOR_RGB2BGR)

    def _clear_roi(self):
        self.lbl_roi_select.clear_roi()
        self.lbl_roi_info.setText("ROI 已清除，请重新拖拽选择")

    def _on_roi_selected(self, rect):
        self.lbl_roi_info.setText(f'已选 ROI: x={rect.x()}, y={rect.y()}, w={rect.width()}, h={rect.height()}  点击"对 ROI 进行分割"或"开始实时屏幕分割"')

    def _get_roi_image_coords(self):
        """将 QLabel 上的 ROI 矩形映射到原始截图坐标，返回 (x1,y1,x2,y2)。"""
        roi_rect = self.lbl_roi_select.get_roi()
        if roi_rect.isNull() or roi_rect.width() < 3 or roi_rect.height() < 3:
            return None
        h_img, w_img = self._roi_image_cv.shape[:2]
        pm = self._roi_pixmap
        scaled = pm.scaled(self.lbl_roi_select.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        scale_x = w_img / scaled.width()
        scale_y = h_img / scaled.height()
        x1 = max(0, int(roi_rect.x() * scale_x))
        y1 = max(0, int(roi_rect.y() * scale_y))
        x2 = min(w_img, int((roi_rect.x() + roi_rect.width()) * scale_x))
        y2 = min(h_img, int((roi_rect.y() + roi_rect.height()) * scale_y))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def _run_roi_seg(self):
        weights = self.le_roi_weights.text()
        if not weights or not os.path.exists(weights):
            QMessageBox.warning(self, "提示", "请选择有效的权重文件"); return
        if not hasattr(self, '_roi_image_cv') or self._roi_image_cv is None:
            QMessageBox.warning(self, "提示", "请先截取屏幕"); return

        coords = self._get_roi_image_coords()
        if coords is None:
            QMessageBox.warning(self, "提示", "请先拖拽选择 ROI 区域"); return
        x1, y1, x2, y2 = coords

        try:
            model = YOLO(weights)
            crop = self._roi_image_cv[y1:y2, x1:x2]
            result = model.predict(source=crop, device=self.cb_device.currentText(), verbose=False, show_boxes=False, show_labels=False, show_conf=False)[0]
            plotted = result.plot(boxes=False, labels=False, probs=False)
            self.lbl_roi_result.setPixmap(cv2_to_qpixmap(plotted).scaledToWidth(500, Qt.TransformationMode.SmoothTransformation))
            self.lbl_roi_info.setText(f"ROI 分割完成: [{x1},{y1},{x2},{y2}]")
        except Exception as e:
            QMessageBox.critical(self, "ROI 分割错误", str(e))

    def _start_roi_screen(self):
        weights = self.le_roi_weights.text()
        if not weights or not os.path.exists(weights):
            QMessageBox.warning(self, "提示", "请选择有效的权重文件"); return
        if not hasattr(self, '_roi_image_cv') or self._roi_image_cv is None:
            QMessageBox.warning(self, "提示", "请先截取屏幕"); return

        coords = self._get_roi_image_coords()
        if coords is None:
            QMessageBox.warning(self, "提示", "请先拖拽选择 ROI 区域"); return
        x1, y1, x2, y2 = coords

        self.btn_roi_screen_start.setEnabled(False)
        self.btn_roi_screen_pause.setEnabled(True)
        self.btn_roi_screen_pause.setText("暂停")
        self.btn_roi_screen_stop.setEnabled(True)

        self.roi_screen_thread = ROIScreenThread(weights, (x1, y1, x2 - x1, y2 - y1), self.cb_device.currentText())
        self.roi_screen_thread.frame_ready.connect(self._on_roi_screen_frame)
        self.roi_screen_thread.finished_screen.connect(self._on_roi_screen_finished)
        self.roi_screen_thread.error.connect(self._on_roi_screen_error)
        self.roi_screen_thread.start()
        self.lbl_roi_info.setText(f"实时屏幕分割中: ROI=({x1},{y1},{x2-x1},{y2-y1})")

    def _toggle_roi_screen_pause(self):
        if not self.roi_screen_thread:
            return
        if self.roi_screen_thread._paused:
            self.roi_screen_thread.pause(False)
            self.btn_roi_screen_pause.setText("暂停")
        else:
            self.roi_screen_thread.pause(True)
            self.btn_roi_screen_pause.setText("继续")

    def _stop_roi_screen(self):
        if self.roi_screen_thread:
            self.roi_screen_thread.stop()
        self.btn_roi_screen_start.setEnabled(True)
        self.btn_roi_screen_pause.setEnabled(False)
        self.btn_roi_screen_stop.setEnabled(False)

    def _on_roi_screen_frame(self, orig_cv, result_cv):
        self.lbl_roi_screen_result.setPixmap(cv2_to_qpixmap(result_cv).scaledToWidth(500, Qt.TransformationMode.SmoothTransformation))

    def _on_roi_screen_finished(self):
        self.btn_roi_screen_start.setEnabled(True)
        self.btn_roi_screen_pause.setEnabled(False)
        self.btn_roi_screen_stop.setEnabled(False)
        self.lbl_roi_info.setText("实时屏幕分割已停止")

    def _on_roi_screen_error(self, msg):
        self.btn_roi_screen_start.setEnabled(True)
        self.btn_roi_screen_pause.setEnabled(False)
        self.btn_roi_screen_stop.setEnabled(False)
        QMessageBox.critical(self, "实时屏幕分割错误", msg)

    # ======================== 评估页 ========================
    def _build_eval_tab(self):
        w = QWidget()
        layout = QHBoxLayout(w)

        left = QVBoxLayout()
        left.addWidget(QLabel("训练结果目录:"))
        self.le_eval_dir = QLineEdit()
        left.addWidget(self.le_eval_dir)
        btn_eval = QPushButton("浏览...")
        btn_eval.clicked.connect(self._browse_eval_dir)
        left.addWidget(btn_eval)

        btn_scan = QPushButton("扫描 runs/segment")
        btn_scan.clicked.connect(self._scan_runs)
        left.addWidget(btn_scan)

        self.lw_eval = QListWidget()
        self.lw_eval.currentTextChanged.connect(self._on_eval_dir_changed)
        left.addWidget(self.lw_eval)

        left.addWidget(QLabel("可用图表:"))
        self.lw_images = QListWidget()
        self.lw_images.itemClicked.connect(self._show_eval_image)
        left.addWidget(self.lw_images)

        left.addStretch()
        layout.addLayout(left, 1)

        right = QVBoxLayout()
        self.lbl_eval_img = QLabel("请选择图表")
        self.lbl_eval_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_eval_img.setMinimumSize(600, 400)
        self.lbl_eval_img.setStyleSheet("background-color: #f0f0f0;")
        right.addWidget(self.lbl_eval_img)
        layout.addLayout(right, 3)

        self.tabs.addTab(w, "评估可视化")

    def _browse_eval_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择训练结果目录")
        if d:
            self.le_eval_dir.setText(d)
            self._load_eval_dir(d)

    def _scan_runs(self):
        base = os.path.join(os.getcwd(), "runs", "segment")
        if not os.path.exists(base):
            QMessageBox.information(self, "提示", f"未找到 {base}"); return
        dirs = [os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
        self.lw_eval.clear()
        for d in sorted(dirs, key=os.path.getmtime, reverse=True):
            self.lw_eval.addItem(d)

    def _on_eval_dir_changed(self, path):
        if path:
            self._load_eval_dir(path)

    def _load_eval_dir(self, path):
        self.le_eval_dir.setText(path)
        self.lw_images.clear()
        if not os.path.isdir(path):
            return
        candidates = [
            "results.png", "confusion_matrix.png", "confusion_matrix_normalized.png",
            "F1_curve.png", "PR_curve.png", "P_curve.png", "R_curve.png",
            "labels.jpg", "labels_correlogram.jpg",
        ]
        for c in candidates:
            p = os.path.join(path, c)
            if os.path.exists(p):
                self.lw_images.addItem(p)

    def _show_eval_image(self, item):
        path = item.text()
        pm = QPixmap(path)
        if not pm.isNull():
            pm = pm.scaled(self.lbl_eval_img.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.lbl_eval_img.setPixmap(pm)
        else:
            self.lbl_eval_img.setText("无法加载图片")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.lw_images.currentItem():
            self._show_eval_image(self.lw_images.currentItem())

# ------------------------------------------------------------------
# 入口
# ------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
