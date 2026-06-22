# vision/angle_detect.py
from __future__ import annotations

import time
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union
import numpy as np

import sys
CURRENT_FILE = Path(__file__).resolve()
CURRENT_DIR = CURRENT_FILE.parent
PROJECT_ROOT = CURRENT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision.screen_capture import FixedRegionScreenCapture, CaptureArea
from vision.newpro_1 import AngleDetector
logger = logging.getLogger(__name__)


class ScreenAngleDetector:
    """
    屏幕捕获 + 角度检测封装模块。

    功能：
        1. 初始化时加载一次 YOLO 角度检测模型；
        2. 每次 detect_once() 捕获一张屏幕图像；
        3. 把截图传入 AngleDetector.detect_angle()；
        4. 输出角度值 angle_deg。

    输入：
        capture_area: (left, top, width, height)

    输出：
        angle_deg: Optional[float]
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        capture_area: CaptureArea = (116, 98, 1112, 886),
        output_dir: Union[str, Path] = "outputs/captured_frames",
        save_image: bool = True,
        device: Optional[str] = None,
        num: int = 0,
        cw: int = 0,
        use_second_detect: bool = True,
    ):
        self.model_path = str(model_path)
        self.capture_area = capture_area
        self.output_dir = Path(output_dir)
        self.save_image = bool(save_image)

        self.device = device
        self.num = int(num)
        self.cw = int(cw)
        self.use_second_detect = bool(use_second_detect)

        self.capturer = FixedRegionScreenCapture(
            capture_area=self.capture_area,
            output_dir=self.output_dir,
            save_image=self.save_image,
        )

        self.angle_detector = AngleDetector(
            model_path=self.model_path,
            device=self.device,
        )

        self.last_capture_result: Optional[Dict[str, Any]] = None
        self.last_image_np: Optional[np.ndarray] = None
        self.last_image_path: Optional[str] = None
        self.last_angle_deg: Optional[float] = None
        self.last_result: Optional[Dict[str, Any]] = None

    def capture_once(self) -> Dict[str, Any]:
        """
        只捕获一张屏幕图像。

        返回：
            {
                "ok": True/False,
                "image": image_np,
                "image_path": "...",
                "capture_area": (...),
                "timestamp": "...",
                "shape": ...
            }
        """
        result = self.capturer.capture_and_save()

        self.last_capture_result = result

        if result.get("ok", False):
            self.last_image_np = result.get("image")
            self.last_image_path = result.get("image_path")
        else:
            self.last_image_np = None
            self.last_image_path = None

        return result

    def detect_angle_from_image(
        self,
        image_np: np.ndarray,
    ) -> Optional[float]:
        """
        对已经捕获到的图像做角度检测。

        输入：
            image_np: numpy RGB 图像

        输出：
            angle_deg
        """
        angle_deg = self.angle_detector.detect_angle(
            image=image_np,
            num=self.num,
            cw=self.cw,
            use_second_detect=self.use_second_detect,
        )

        self.last_angle_deg = angle_deg
        return angle_deg

    def detect_once(self) -> Optional[float]:
        """
        捕获一张屏幕图像，并检测角度。

        这是最常用的调用接口。

        返回：
            angle_deg: Optional[float]
        """
        result = self.detect_once_detail()

        if result.get("ok", False):
            return result.get("angle_deg")

        return None

    def detect_once_detail(self) -> Dict[str, Any]:
        """
        捕获一张屏幕图像，并检测角度。

        返回详细信息：
            {
                "ok": True/False,
                "angle_deg": ...,
                "image_path": ...,
                "capture_area": ...,
                "timestamp": ...,
                "reason": ...
            }
        """
        capture_result = self.capture_once()

        if not capture_result.get("ok", False):
            result = {
                "ok": False,
                "angle_deg": None,
                "reason": "capture_failed",
                "error": capture_result.get("error"),
                "image_path": None,
                "capture_area": self.capture_area,
                "timestamp": capture_result.get("timestamp"),
                "capture_result": capture_result,
            }

            self.last_result = result
            return result

        image_np = capture_result.get("image")

        if image_np is None:
            result = {
                "ok": False,
                "angle_deg": None,
                "reason": "captured_image_is_none",
                "image_path": capture_result.get("image_path"),
                "capture_area": self.capture_area,
                "timestamp": capture_result.get("timestamp"),
                "capture_result": capture_result,
            }

            self.last_result = result
            return result

        try:
            angle_deg = self.detect_angle_from_image(image_np)

            if angle_deg is None:
                result = {
                    "ok": False,
                    "angle_deg": None,
                    "reason": "angle_detect_failed",
                    "image_path": capture_result.get("image_path"),
                    "capture_area": self.capture_area,
                    "timestamp": capture_result.get("timestamp"),
                    "shape": capture_result.get("shape"),
                    "capture_result": capture_result,
                }

                self.last_result = result
                return result

            result = {
                "ok": True,
                "angle_deg": float(angle_deg),
                "reason": "ok",
                "image_path": capture_result.get("image_path"),
                "capture_area": self.capture_area,
                "timestamp": capture_result.get("timestamp"),
                "shape": capture_result.get("shape"),
                "capture_result": capture_result,
            }

            self.last_result = result
            return result

        except Exception as e:
            logger.exception("屏幕捕获后的角度检测失败")

            result = {
                "ok": False,
                "angle_deg": None,
                "reason": "angle_detect_exception",
                "error": str(e),
                "image_path": capture_result.get("image_path"),
                "capture_area": self.capture_area,
                "timestamp": capture_result.get("timestamp"),
                "shape": capture_result.get("shape"),
                "capture_result": capture_result,
            }

            self.last_result = result
            return result


def get_angle_from_screen_once(
    model_path: Union[str, Path],
    capture_area: CaptureArea = (116, 98, 1112, 886),
    output_dir: Union[str, Path] = "outputs/captured_frames",
    save_image: bool = True,
    device: Optional[str] = None,
    num: int = 0,
    cw: int = 0,
    use_second_detect: bool = True,
) -> Optional[float]:
 
    detector = ScreenAngleDetector(
        model_path=model_path,
        capture_area=capture_area,
        output_dir=output_dir,
        save_image=save_image,
        device=device,
        num=num,
        cw=cw,
        use_second_detect=use_second_detect,
    )

    return detector.detect_once()


def get_angle_from_screen_once_detail(
    model_path: Union[str, Path],
    capture_area: CaptureArea = (116, 98, 1112, 886),
    output_dir: Union[str, Path] = "outputs/captured_frames",
    save_image: bool = True,
    device: Optional[str] = None,
    num: int = 0,
    cw: int = 0,
    use_second_detect: bool = True,
) -> Dict[str, Any]:
    """
    便捷函数：屏幕捕获一次，然后检测一次角度，返回详细结果。

    注意：
        每调用一次都会重新加载模型。
        正式循环测量时，不建议用这个函数。
    """
    detector = ScreenAngleDetector(
        model_path=model_path,
        capture_area=capture_area,
        output_dir=output_dir,
        save_image=save_image,
        device=device,
        num=num,
        cw=cw,
        use_second_detect=use_second_detect,
    )

    return detector.detect_once_detail()

'''
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    model_path = r"D:/desktop/train/model/best_wan12.2.pt"

    capture_area = (116, 98, 1112, 886)

    # 推荐用法：
    # 先创建 detector，这里模型只加载一次。
    detector = ScreenAngleDetector(
        model_path=model_path,
        capture_area=capture_area,
        output_dir="outputs/captured_frames",
        save_image=True,
        device=None,
        num=0,
        cw=0,
        use_second_detect=True,
    )

    # 检测一次
    result = detector.detect_once_detail()

    print("========== 检测结果 ==========")
    print(f"ok          = {result.get('ok')}")
    print(f"angle_deg   = {result.get('angle_deg')}")
    print(f"reason      = {result.get('reason')}")
    print(f"image_path  = {result.get('image_path')}")

    # 如果你想连续检测两次，也不要重新创建 detector
    time.sleep(1.0)

    angle2 = detector.detect_once()
    print(f"第二次检测 angle_deg = {angle2}")
    '''