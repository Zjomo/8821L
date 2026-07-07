"""
自动聚焦后台工作线程，封装 Focus 模块的闭环流程，避免阻塞 UI。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .qt_compat import QObject, QThread, Signal, Slot

logger = logging.getLogger("autofocus_qt_ui.worker")


@dataclass
class AutofocusRuntimeState:
    """运行时的共享状态，用于线程间传递。"""

    running: bool = False
    paused: bool = False
    cycle_index: int = 0
    total_cycles: int = 0
    focus_score: Optional[float] = None
    virtual_z: Optional[int] = None
    roi_image: Optional[np.ndarray] = None
    full_metrics: Dict[str, float] = field(default_factory=dict)
    roi_metrics: Dict[str, float] = field(default_factory=dict)
    log_messages: List[str] = field(default_factory=list)
    error_message: Optional[str] = None


class AutofocusWorker(QObject):
    """
    后台自动聚焦工作线程。

    信号：
      - preview_updated(image): 最新 ROI 图像
      - metrics_updated(full, roi): 聚焦指标
      - score_updated(cycle, score): FocusScore_ratio
      - log(message): 日志
      - status_changed(running, cycle, total): 状态更新
      - finished(): 正常结束
      - error(message): 错误
    """

    preview_updated = Signal(np.ndarray)
    metrics_updated = Signal(dict, dict)
    score_updated = Signal(int, float)
    log = Signal(str)
    status_changed = Signal(bool, int, int)
    finished = Signal()
    error = Signal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._state = AutofocusRuntimeState()
        self._controller: Optional[Any] = None
        self._simulator: Optional[Any] = None
        self._cfg: Optional[Any] = None
        self._args: Optional[Dict[str, Any]] = None

    def configure(
        self,
        cfg: Any,
        args: Dict[str, Any],
        controller: Any,
        simulator: Optional[Any] = None,
    ) -> None:
        self._cfg = cfg
        self._args = args
        self._controller = controller
        self._simulator = simulator

    def start_loop(self) -> None:
        self._state.running = True
        self._state.paused = False
        self._state.cycle_index = 0
        self._state.total_cycles = int(self._args.get("cycles", 0))
        self._run_loop()

    def stop_loop(self) -> None:
        self._state.running = False

    def pause_loop(self) -> None:
        self._state.paused = True

    def resume_loop(self) -> None:
        self._state.paused = False

    def build_reference(self) -> None:
        """建立参考基线（在后台线程中运行）。"""
        try:
            self.log.emit("开始建立聚焦参考基线...")
            ref_dir = Path(self._args.get("output", "focus_output")) / "reference"
            ref_dir.mkdir(parents=True, exist_ok=True)
            ref = self._controller.build_reference(output_root=ref_dir)
            count = ref.get("capture_count", 0)
            self.log.emit(f"参考基线建立完成：共 {count} 次采集")
        except Exception as exc:
            logger.exception("build_reference failed")
            self.error.emit(f"建立参考基线失败：{exc}")

    def _run_loop(self) -> None:
        if self._controller is None or self._cfg is None:
            self.error.emit("工作线程未配置")
            return

        output_root = Path(self._args.get("output", "focus_output"))
        output_root.mkdir(parents=True, exist_ok=True)
        save_dir = output_root / "cycles"
        cycles = int(self._args.get("cycles", 0))
        interval = max(0.0, float(self._args.get("interval", 5.0)))

        self.status_changed.emit(True, 0, cycles)
        self.log.emit(
            f"启动闭环：cycles={'无限' if cycles <= 0 else cycles}, interval={interval}s"
        )

        try:
            while self._state.running:
                while self._state.paused and self._state.running:
                    time.sleep(0.1)
                if not self._state.running:
                    break

                self._state.cycle_index += 1
                if cycles > 0 and self._state.cycle_index > cycles:
                    break

                cycle = self._state.cycle_index
                cycle_save_dir = save_dir / f"cycle_{cycle:04d}"
                cycle_save_dir.mkdir(parents=True, exist_ok=True)

                self.log.emit(f"========== 第 {cycle} 轮 ==========")
                self.status_changed.emit(True, cycle, cycles)

                result = self._controller.check_and_autofocus(
                    cycle_index=cycle,
                    save_dir=str(cycle_save_dir),
                )

                score = result.get("focus_score_ratio") if isinstance(result, dict) else None
                self._state.focus_score = score
                self._update_preview(result)

                if score is not None:
                    self.score_updated.emit(cycle, score)
                    self.log.emit(f"第 {cycle} 轮 FocusScore_ratio = {score:.4f}")

                if self._simulator is not None:
                    virtual_z = self._simulator.virtual_z_axis.get_position()
                    self._state.virtual_z = virtual_z
                    self.log.emit(f"虚拟 Z 位置 = {virtual_z}")
                    self._simulator.next_cycle()

                if cycles <= 0 or cycle < cycles:
                    self.log.emit(f"等待 {interval}s 后进行下一轮...")
                    slept = 0.0
                    while slept < interval and self._state.running and not self._state.paused:
                        time.sleep(0.1)
                        slept += 0.1

        except Exception as exc:
            logger.exception("autofocus loop failed")
            self.error.emit(f"闭环运行异常：{exc}")
        finally:
            self._cleanup()
            self.status_changed.emit(False, self._state.cycle_index, cycles)
            self.finished.emit()

    def _update_preview(self, result: Dict[str, Any]) -> None:
        """从结果中提取 ROI 图像和指标，发送给 UI。"""
        if not isinstance(result, dict):
            return

        # Focus 模块返回的图像字段为 roi_rgb / full_rgb，指标字段为 full / roi_metrics
        roi_image = result.get("roi_rgb")
        if roi_image is None:
            roi_image = result.get("full_rgb")
        full_metrics = result.get("full")
        if full_metrics is None:
            full_metrics = {}
        roi_metrics = result.get("roi_metrics")
        if roi_metrics is None:
            roi_metrics = {}

        if roi_image is not None and isinstance(roi_image, np.ndarray):
            self.preview_updated.emit(roi_image)

        self.metrics_updated.emit(full_metrics, roi_metrics)

    def _cleanup(self) -> None:
        try:
            if self._controller is not None:
                self._controller.z_axis.close()
        except Exception as exc:
            self.log.emit(f"关闭 Z 轴时出错：{exc}")


class AutofocusThread(QThread):
    """包装 AutofocusWorker 的 QThread，方便生命周期管理。"""

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.worker = AutofocusWorker()
        self.worker.moveToThread(self)
        self.started.connect(self.worker.start_loop)

    def configure(
        self,
        cfg: Any,
        args: Dict[str, Any],
        controller: Any,
        simulator: Optional[Any] = None,
    ) -> None:
        self.worker.configure(cfg, args, controller, simulator)
