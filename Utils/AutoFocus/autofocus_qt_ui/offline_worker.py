"""
离线数据集检测工作线程。

将 offline_dataset_detection.OfflineDatasetDetector 包装到 QThread 中，
避免在 UI 主线程执行视频抽帧、FocusScore 计算等耗时操作。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .qt_compat import QObject, QThread, Signal, Slot

logger = logging.getLogger("autofocus_qt_ui.offline_worker")


@dataclass
class OfflineDatasetRuntimeState:
    """离线数据集检测运行时状态。"""

    running: bool = False
    total: int = 0
    current: int = 0
    output_dir: Optional[Path] = None
    results: List[Any] = field(default_factory=list)
    error_message: Optional[str] = None


class OfflineDatasetWorker(QObject):
    """离线数据集检测后台工作对象。"""

    log = Signal(str)
    progress = Signal(int, int)  # current, total
    finished = Signal(bool, str)  # ok, output_dir or error message
    image_processed = Signal(str, float)  # image_name, score

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._detector: Optional[Any] = None
        self._state = OfflineDatasetRuntimeState()

    def configure(
        self,
        detector: Any,
        input_path: Any,
        output_dir: Optional[Any] = None,
        reference_path: Optional[Any] = None,
        interval_seconds: float = 3.0,
    ) -> None:
        """
        配置离线检测任务。

        参数
        ----------
        detector : OfflineDatasetDetector
            已配置好的离线数据集检测器。
        input_path : str | Path
            输入视频或图片文件夹路径。
        output_dir : str | Path, optional
            输出目录路径。
        reference_path : str | Path, optional
            手动指定的基准图路径。
        interval_seconds : float, default 3.0
            视频帧提取间隔（秒）。
        """
        self._detector = detector
        self._input_path = Path(input_path) if input_path else None
        self._output_dir = Path(output_dir) if output_dir else None
        self._reference_path = Path(reference_path) if reference_path else None
        self._interval_seconds = interval_seconds
        self._state = OfflineDatasetRuntimeState()

    @Slot()
    def run(self) -> None:
        """在后台线程中执行离线数据集检测。"""
        if self._detector is None or self._input_path is None:
            self.finished.emit(False, "离线检测任务未正确配置")
            return

        self._state.running = True
        ok = False
        output_dir_str = ""

        try:
            detector = self._detector

            # 使用 detector 的日志回调同时发射 progress
            original_on_log = detector.on_log

            def _log_with_progress(msg: str) -> None:
                original_on_log(msg)
                self.log.emit(msg)
                # 尝试从日志中解析进度
                if "第 " in msg and " 轮" in msg:
                    try:
                        parts = msg.split("第 ")[1].split(" 轮")[0]
                        current = int(parts)
                        if self._state.total > 0:
                            self.progress.emit(current, self._state.total)
                    except Exception:
                        pass

            detector.on_log = _log_with_progress

            # 预估总数用于进度条
            if detector.is_video(self._input_path):
                self._state.total = self._estimate_video_frames()
            elif detector.is_image_folder(self._input_path):
                self._state.total = len(detector.list_image_paths(self._input_path))
            else:
                self._state.total = 0

            output_dir, results = detector.run(
                input_path=self._input_path,
                output_dir=self._output_dir,
                reference_path=self._reference_path,
                interval_seconds=self._interval_seconds,
                delete_zero_score=True,
            )

            self._state.output_dir = output_dir
            self._state.results = results
            output_dir_str = str(output_dir)

            for result in results:
                if not result.deleted and result.annotated_path is not None:
                    self.image_processed.emit(
                        result.annotated_path.name,
                        result.score if result.score is not None else 0.0,
                    )

            kept = sum(1 for r in results if not r.deleted)
            deleted = sum(1 for r in results if r.deleted)
            self.log.emit(
                f"[离线检测] 完成：总计 {len(results)} 张，"
                f"保留 {kept} 张，删除 {deleted} 张"
            )
            ok = True

        except Exception as exc:
            logger.exception("离线数据集检测失败")
            self._state.error_message = str(exc)
            self.log.emit(f"[离线检测] 失败：{exc}")
            output_dir_str = str(exc)
        finally:
            self._state.running = False
            # 恢复原始日志回调
            if self._detector is not None:
                self._detector.on_log = original_on_log
            self.finished.emit(ok, output_dir_str)

    def _estimate_video_frames(self) -> int:
        """估算视频将提取的帧数，用于进度条。"""
        try:
            import cv2

            cap = cv2.VideoCapture(str(self._input_path))
            if not cap.isOpened():
                return 0
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            interval_frames = max(1, int(fps * self._interval_seconds))
            return max(1, total_frames // interval_frames)
        except Exception:
            return 0

    def stop(self) -> None:
        """请求停止（离线检测为一次性任务，目前仅标记状态）。"""
        self._state.running = False
        self.log.emit("[离线检测] 收到停止请求")


class OfflineDatasetThread(QThread):
    """包装 OfflineDatasetWorker 的 QThread。"""

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.worker = OfflineDatasetWorker()
        self.worker.moveToThread(self)
        self.started.connect(self.worker.run)

    def configure(self, **kwargs: Any) -> None:
        self.worker.configure(**kwargs)

    def run(self) -> None:
        self.worker.run()
