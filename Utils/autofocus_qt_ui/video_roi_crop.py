"""
离线视频 ROI 裁剪核心模块。

提供将本地视频按指定 ROI 区域逐帧裁剪并输出为新视频的能力。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

# 常见视频编码，按偏好顺序尝试
_VIDEO_CODECS = ["mp4v", "XVID", "MJPG"]


def _clamp_roi(roi: Tuple[int, int, int, int], frame_shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
    """将 ROI 限制在帧范围内，确保宽/高为正。"""
    rx, ry, rw, rh = roi
    h, w = frame_shape[:2]

    rx = max(0, min(w - 1, rx))
    ry = max(0, min(h - 1, ry))
    rw = max(1, min(rw, w - rx))
    rh = max(1, min(rh, h - ry))
    return rx, ry, rw, rh


def _guess_output_path(input_path: Union[str, Path], roi: Tuple[int, int, int, int]) -> Path:
    """根据输入视频路径和 ROI 自动生成输出视频路径。"""
    input_path = Path(input_path)
    rx, ry, rw, rh = roi
    return input_path.parent / f"{input_path.stem}_roi_{rx}_{ry}_{rw}_{rh}{input_path.suffix}"


class VideoRoiCropper:
    """视频 ROI 裁剪器：读取视频、裁剪 ROI、写入新视频。"""

    def __init__(
        self,
        on_log: Optional[Callable[[str], None]] = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        self.on_log = on_log or logger.info
        self.on_progress = on_progress

    @staticmethod
    def is_video(path: Union[str, Path]) -> bool:
        """判断路径是否为支持的视频文件。"""
        return Path(path).suffix.lower() in VIDEO_EXTENSIONS

    @staticmethod
    def crop_frame(frame: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
        """对单帧进行 ROI 裁剪。

        参数
        ----------
        frame : np.ndarray
            BGR/RGB 图像数组。
        roi : tuple[int, int, int, int]
            (x, y, w, h)。

        返回
        -------
        np.ndarray
            裁剪后的图像。
        """
        rx, ry, rw, rh = _clamp_roi(roi, frame.shape)
        return frame[ry : ry + rh, rx : rx + rw].copy()

    def get_video_info(self, input_path: Union[str, Path]) -> dict:
        """读取视频基本信息。

        返回
        -------
        dict
            包含 fps、frame_count、width、height、duration。
        """
        if cv2 is None:
            raise ImportError("需要安装 opencv-python：pip install opencv-python")

        input_path = Path(input_path)
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频：{input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        return {
            "fps": fps,
            "frame_count": max(0, frame_count),
            "width": width,
            "height": height,
            "duration": frame_count / fps if fps > 0 else 0.0,
        }

    def crop_video(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        roi: Tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> Path:
        """执行视频 ROI 裁剪。

        参数
        ----------
        input_path : str | Path
            输入视频路径。
        output_path : str | Path, optional
            输出视频路径。为 None 时自动生成。
        roi : tuple[int, int, int, int]
            ROI (x, y, w, h)。

        返回
        -------
        Path
            输出视频路径。
        """
        if cv2 is None:
            raise ImportError("需要安装 opencv-python：pip install opencv-python")

        input_path = Path(input_path)
        if not input_path.is_file():
            raise ValueError(f"输入视频不存在：{input_path}")

        if not self.is_video(input_path):
            raise ValueError(f"不支持的输入文件格式：{input_path.suffix}")

        # 打开视频读取第一帧以确定实际尺寸并校验 ROI
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频：{input_path}")

        ret, first_frame = cap.read()
        if not ret or first_frame is None:
            cap.release()
            raise ValueError(f"无法读取视频首帧：{input_path}")

        # 校验原始 ROI 尺寸
        rx, ry, rw, rh = roi
        if rw <= 0 or rh <= 0:
            cap.release()
            raise ValueError(f"ROI 尺寸无效：{roi}")

        # 修正 ROI 到帧范围内
        roi = _clamp_roi(roi, first_frame.shape)
        rx, ry, rw, rh = roi

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = max(1, frame_count)

        if output_path is None:
            output_path = _guess_output_path(input_path, roi)
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 尝试创建 VideoWriter
        writer = None
        fourcc = None
        for codec in _VIDEO_CODECS:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (rw, rh))
            if writer.isOpened():
                self.on_log(f"[ROI裁剪] 使用编码：{codec}")
                break
            writer.release()
            writer = None

        if writer is None or not writer.isOpened():
            cap.release()
            raise RuntimeError("无法创建视频写入器，请检查 OpenCV 视频编码支持")

        self.on_log(
            f"[ROI裁剪] 开始裁剪：{input_path.name} -> {output_path.name}, "
            f"ROI=({rx},{ry},{rw},{rh}), fps={fps:.2f}, 总帧数={frame_count}"
        )

        processed = 0
        # 写入首帧
        cropped = self.crop_frame(first_frame, roi)
        writer.write(cropped)
        processed += 1
        self._emit_progress(processed, frame_count)

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            cropped = self.crop_frame(frame, roi)
            writer.write(cropped)
            processed += 1
            self._emit_progress(processed, frame_count)

        cap.release()
        writer.release()

        self.on_log(
            f"[ROI裁剪] 完成：共写入 {processed} 帧，输出 {output_path}"
        )
        return output_path

    def _emit_progress(self, current: int, total: int) -> None:
        if self.on_progress is not None:
            try:
                self.on_progress(current, total)
            except Exception:
                pass


def main() -> None:
    """命令行入口示例。"""
    import argparse

    parser = argparse.ArgumentParser(description="离线视频 ROI 裁剪")
    parser.add_argument("input", help="输入视频文件路径")
    parser.add_argument("--output", "-o", help="输出视频文件路径（可选）")
    parser.add_argument(
        "--roi", required=True, help="ROI，格式 x,y,w,h"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parts = [int(p.strip()) for p in args.roi.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI 格式应为 x,y,w,h")
    roi = tuple(parts)

    cropper = VideoRoiCropper()
    output = cropper.crop_video(args.input, args.output, roi)
    print(f"输出视频：{output}")


if __name__ == "__main__":
    main()
