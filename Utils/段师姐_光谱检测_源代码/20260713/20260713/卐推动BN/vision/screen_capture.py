# vision/capture.py
from __future__ import annotations

import time
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pyautogui
from PIL import Image

logger = logging.getLogger(__name__)

CaptureArea = Tuple[int, int, int, int]  # (left, top, width, height)


class FixedRegionScreenCapture:
    """
    固定屏幕区域截图模块。

    用途：
        1. 被 run_sample_task.py 直接调用；
        2. 被 sample_cycle.py 作为 capture 对象调用；
        3. 每次调用 capture() 只截取一张图，不启动线程。

    capture_area:
        (left, top, width, height)
    """

    def __init__(
        self,
        capture_area: CaptureArea,
        output_dir: str | Path = "outputs/captured_frames",
        save_image: bool = True,
    ):
        self.capture_area = capture_area
        self.output_dir = Path(output_dir)
        self.save_image = save_image

        if self.save_image:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def capture(self) -> np.ndarray:
        """
        截取固定区域，返回 numpy RGB 图像。

        返回：
            image_np: np.ndarray, shape = (H, W, 3), RGB
        """
        frame = pyautogui.screenshot(region=self.capture_area)
        image_np = np.array(frame)

        return image_np

    def capture_pil(self) -> Image.Image:
        """
        截取固定区域，返回 PIL.Image。
        """
        frame = pyautogui.screenshot(region=self.capture_area)
        return frame

    def capture_and_save(
        self,
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        截取固定区域并保存图片。

        返回：
            {
                "ok": True,
                "image": image_np,
                "image_path": "...",
                "capture_area": (...),
                "timestamp": ...
            }
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        if filename is None:
            filename = f"capture_{timestamp}.png"

        save_path = self.output_dir / filename

        try:
            frame = pyautogui.screenshot(region=self.capture_area)
            image_np = np.array(frame)

            if self.save_image:
                frame.save(save_path)
                image_path = str(save_path)
            else:
                image_path = None

            logger.info(f"截图成功: {image_path}")

            return {
                "ok": True,
                "image": image_np,
                "image_path": image_path,
                "capture_area": self.capture_area,
                "timestamp": timestamp,
                "shape": image_np.shape,
            }

        except Exception as e:
            logger.error(f"固定区域截图失败: {e}")

            return {
                "ok": False,
                "image": None,
                "image_path": None,
                "capture_area": self.capture_area,
                "timestamp": timestamp,
                "error": str(e),
            }


def capture_fixed_region_once(
    capture_area: CaptureArea,
    output_dir: str | Path = "outputs/captured_frames",
    filename: Optional[str] = None,
) -> Dict[str, Any]:
    """
    便捷函数：直接截取一次固定区域并保存。

    适合简单脚本调用。
    """
    capturer = FixedRegionScreenCapture(
        capture_area=capture_area,
        output_dir=output_dir,
        save_image=True,
    )

    return capturer.capture_and_save(filename=filename)