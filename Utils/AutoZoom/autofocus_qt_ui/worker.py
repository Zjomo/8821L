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
    capture_requested = Signal()  # 请求主线程截图（避免后台线程调用 GUI API 崩溃）

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._state = AutofocusRuntimeState()
        self._controller: Optional[Any] = None
        self._simulator: Optional[Any] = None
        self._cfg: Optional[Any] = None
        self._args: Optional[Dict[str, Any]] = None
        self._captured_image: Optional[np.ndarray] = None

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

    def set_captured_image(self, image: Optional[np.ndarray]) -> None:
        """接收主线程捕获的截图。"""
        self._captured_image = image

    def _request_capture(self) -> Optional[np.ndarray]:
        """请求主线程截图并等待结果（避免后台线程直接调用 GUI API）。"""
        self._captured_image = None
        self.capture_requested.emit()
        timeout = 3.0
        start = time.time()
        while self._captured_image is None and time.time() - start < timeout:
            if not self._state.running:
                return None
            time.sleep(0.02)
        return self._captured_image

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
        interval = max(0.0, float(self._args.get("interval", 1.5)))

        self.status_changed.emit(True, 0, cycles)
        self.log.emit(
            f"启动闭环：cycles={'无限' if cycles <= 0 else cycles}, interval={interval}s"
        )

        # 用布尔标志跟踪 monkey-patch 状态，避免 dir() 的不确定性
        monkey_patched = False
        original_capture_live = None
        original_capture_and_save = None

        try:
            # 将 controller 的日志回调改为 worker 信号，杜绝后台线程直接调用 Qt GUI
            self._controller.on_log = self.log.emit

            # 注册停止回调，使用户点击“停止”时能立即中断补焦搜索
            self._controller.should_stop = lambda: not self._state.running

            # 安全措施：屏幕区域模式下，将所有截图操作委托给主线程
            if self._controller.metrics_calc.cfg.capture_mode == "screen_region":
                original_capture_live = self._controller.metrics_calc.capture_live
                original_capture_and_save = self._controller.metrics_calc.capture_and_save
                worker_self = self

                def safe_capture_live():
                    image = worker_self._request_capture()
                    if image is not None:
                        roi = worker_self._controller.metrics_calc.clamp_roi(
                            tuple(worker_self._controller.metrics_calc.cfg.focus_roi), image.shape
                        )
                        x, y, rw, rh = roi
                        roi_rgb = image[y : y + rh, x : x + rw].copy()
                        full_metrics = worker_self._controller.metrics_calc.compute_for_image(image)
                        roi_metrics = worker_self._controller.metrics_calc.compute_for_image(roi_rgb)
                        return {
                            "ok": True,
                            "full_rgb": image,
                            "roi_rgb": roi_rgb,
                            "full": full_metrics,
                            "roi_metrics": roi_metrics,
                        }
                    # 主线程截图超时，返回错误结果（绝不 fallback 到后台线程调用 pyautogui）
                    worker_self.log.emit("[警告] 主线程截图超时，返回空结果")
                    return {"ok": False, "full_rgb": None, "roi_rgb": None, "full": {}, "roi_metrics": {}}

                def safe_capture_and_save(cycle_index, save_dir, on_log=None):
                    image = worker_self._request_capture()
                    if image is None:
                        # 主线程截图超时，返回错误结果（绝不 fallback 到后台线程调用 pyautogui）
                        worker_self.log.emit("[警告] 主线程截图超时，返回空结果")
                        return {"ok": False, "cycle_index": cycle_index, "full": {}, "roi_metrics": {}}
                    left, top, width, height = [int(v) for v in worker_self._controller.metrics_calc.cfg.capture_area]
                    roi = worker_self._controller.metrics_calc.clamp_roi(
                        tuple(worker_self._controller.metrics_calc.cfg.focus_roi), image.shape
                    )
                    x, y, rw, rh = roi
                    roi_rgb = image[y : y + rh, x : x + rw].copy()
                    full_metrics = worker_self._controller.metrics_calc.compute_for_image(image)
                    roi_metrics = worker_self._controller.metrics_calc.compute_for_image(roi_rgb)
                    return {
                        "ok": True,
                        "cycle_index": cycle_index,
                        "capture_area": [left, top, width, height],
                        "roi": roi,
                        "full": full_metrics,
                        "roi_metrics": roi_metrics,
                        "full_rgb": image,
                        "roi_rgb": roi_rgb,
                    }

                self._controller.metrics_calc.capture_live = safe_capture_live
                self._controller.metrics_calc.capture_and_save = safe_capture_and_save
                monkey_patched = True

            # 被动模式仅在自动补焦启用时生效；否则退化为主动模式，受 cycles 限制
            passive_mode = (
                getattr(self._cfg, "autofocus_passive_mode", False)
                and self._cfg.autofocus_enabled
            )
            trigger_ratio = float(self._cfg.autofocus_focus_trigger_ratio)
            passive_consecutive_good = max(
                1, int(getattr(self._cfg, "autofocus_passive_consecutive_good", 3))
            )
            consecutive_good_count = 0

            while self._state.running:
                while self._state.paused and self._state.running:
                    time.sleep(0.1)
                if not self._state.running:
                    break

                self._state.cycle_index += 1
                cycle = self._state.cycle_index

                # 主动模式才受循环轮数限制；被动模式忽略 cycles
                if not passive_mode and cycles > 0 and cycle > cycles:
                    break

                cycle_save_dir = save_dir / f"cycle_{cycle:04d}"
                cycle_save_dir.mkdir(parents=True, exist_ok=True)

                self.log.emit(f"========== 第 {cycle} 轮 ==========")
                self.status_changed.emit(True, cycle, 0 if passive_mode else cycles)

                # save_dir=None 强制走 capture_live（已被 monkey-patch 到主线程安全采集）
                result = self._controller.check_and_autofocus(
                    cycle_index=cycle,
                    save_dir=None,
                )

                score = result.get("focus_score_ratio") if isinstance(result, dict) else None
                self._state.focus_score = score
                self._update_preview(result)

                if score is not None:
                    self.score_updated.emit(cycle, score)
                    self.log.emit(f"第 {cycle} 轮 FocusScore_ratio = {score:.4f}")

                # 被动补焦：统计连续达标轮数，达标 n 轮后自动停止
                if passive_mode and self._cfg.autofocus_enabled and score is not None:
                    if score > trigger_ratio:
                        consecutive_good_count += 1
                        self.log.emit(
                            f"[被动补焦] 分数达标 {score:.4f} > {trigger_ratio}, "
                            f"连续达标 {consecutive_good_count}/{passive_consecutive_good}"
                        )
                        if consecutive_good_count >= passive_consecutive_good:
                            self.log.emit(
                                f"[被动补焦] 连续达标 {passive_consecutive_good} 轮，停止循环"
                            )
                            break
                    else:
                        consecutive_good_count = 0
                        self.log.emit(
                            f"[被动补焦] 分数未达标 {score:.4f} <= {trigger_ratio}，"
                            "本轮已触发补焦，进入常规等待"
                        )

                if self._simulator is not None:
                    virtual_z = self._simulator.virtual_z_axis.get_position()
                    self._state.virtual_z = virtual_z
                    self.log.emit(f"虚拟 Z 位置 = {virtual_z}")
                    self._simulator.next_cycle()

                # 被动模式或主动模式非最后一轮时等待间隔
                should_wait = passive_mode or cycles <= 0 or cycle < cycles
                if self._state.running and should_wait:
                    self.log.emit(f"等待 {interval}s 后进行下一轮...")
                    slept = 0.0
                    while slept < interval and self._state.running and not self._state.paused:
                        time.sleep(0.1)
                        slept += 0.1

        except Exception as exc:
            logger.exception("autofocus loop failed")
            self.error.emit(f"闭环运行异常：{exc}")
        finally:
            # 恢复原始方法（使用布尔标志，避免 dir() 跨实现差异）
            if monkey_patched and self._controller is not None:
                mc = self._controller.metrics_calc
                if original_capture_live is not None:
                    try:
                        mc.capture_live = original_capture_live
                    except Exception:
                        pass
                if original_capture_and_save is not None:
                    try:
                        mc.capture_and_save = original_capture_and_save
                    except Exception:
                        pass
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
        # worker 结束后退出线程事件循环，确保线程正常终止并被安全销毁
        self.worker.finished.connect(self.quit)

    def configure(
        self,
        cfg: Any,
        args: Dict[str, Any],
        controller: Any,
        simulator: Optional[Any] = None,
    ) -> None:
        self.worker.configure(cfg, args, controller, simulator)
