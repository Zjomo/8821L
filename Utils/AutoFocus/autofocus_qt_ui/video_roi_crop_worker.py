"""
离线视频 ROI 裁剪后台工作线程。

将 video_roi_crop.VideoRoiCropper 包装到 QThread 中，避免在主线程执行
视频逐帧读取与写入，防止 UI 卡顿。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .qt_compat import QObject, QThread, Signal, Slot
from .video_roi_crop import VideoRoiCropper

logger = logging.getLogger("autofocus_qt_ui.video_roi_crop_worker")


@dataclass
class VideoRoiCropRuntimeState:
    """视频 ROI 裁剪运行时状态。"""

    running: bool = False
    total: int = 0
    current: int = 0
    output_path: Optional[Path] = None
    error_message: Optional[str] = None


class VideoRoiCropWorker(QObject):
    """视频 ROI 裁剪后台工作对象。"""

    log = Signal(str)
    progress = Signal(int, int)  # current, total
    finished = Signal(bool, str)  # ok, output_path or error message

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._cropper: Optional[VideoRoiCropper] = None
        self._input_path: Optional[Path] = None
        self._output_path: Optional[Path] = None
        self._roi: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._state = VideoRoiCropRuntimeState()

    def configure(
        self,
        input_path: Any,
        output_path: Optional[Any] = None,
        roi: tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> None:
        """配置裁剪任务。

        参数
        ----------
        input_path : str | Path
            输入视频路径。
        output_path : str | Path, optional
            输出视频路径。
        roi : tuple[int, int, int, int]
            ROI (x, y, w, h)。
        """
        self._input_path = Path(input_path) if input_path else None
        self._output_path = Path(output_path) if output_path else None
        self._roi = roi
        self._state = VideoRoiCropRuntimeState()

        self._cropper = VideoRoiCropper(
            on_log=self.log.emit,
            on_progress=self._on_progress,
        )

    def _on_progress(self, current: int, total: int) -> None:
        self._state.current = current
        self._state.total = total
        self.progress.emit(current, total)

    @Slot()
    def run(self) -> None:
        """在后台线程中执行视频 ROI 裁剪。"""
        if self._cropper is None or self._input_path is None:
            self.finished.emit(False, "裁剪任务未正确配置")
            return

        self._state.running = True
        ok = False
        result_str = ""

        try:
            output = self._cropper.crop_video(
                input_path=self._input_path,
                output_path=self._output_path,
                roi=self._roi,
            )
            self._state.output_path = output
            result_str = str(output)
            ok = True
        except Exception as exc:
            logger.exception("视频 ROI 裁剪失败")
            self._state.error_message = str(exc)
            self.log.emit(f"[ROI裁剪] 失败：{exc}")
            result_str = str(exc)
        finally:
            self._state.running = False
            self.finished.emit(ok, result_str)

    def stop(self) -> None:
        """请求停止（当前实现为标记状态；实际写入为一次性流程）。"""
        self._state.running = False
        self.log.emit("[ROI裁剪] 收到停止请求")


class VideoRoiCropThread(QThread):
    """包装 VideoRoiCropWorker 的 QThread。"""

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.worker = VideoRoiCropWorker()
        self.worker.moveToThread(self)
        self.started.connect(self.worker.run)

    def configure(
        self,
        input_path: Any,
        output_path: Optional[Any] = None,
        roi: tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> None:
        self.worker.configure(input_path, output_path, roi)

    def run(self) -> None:
        self.worker.run()
