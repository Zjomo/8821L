# angle_detector_once.py
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Union, Any, Dict, List, Tuple

import numpy as np
from PIL import Image
from ultralytics import YOLO


ImageInput = Union[str, Path, Image.Image, np.ndarray]


def least_squares_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if len(x) < 2 or len(y) < 2:
        return float("nan"), float("nan")

    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_xx = np.sum(x * x)

    denominator = n * sum_xx - sum_x * sum_x

    if abs(denominator) < 1e-12:
        return float("nan"), float("nan")

    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = np.mean(y) - slope * np.mean(x)

    return float(slope), float(intercept)


def angle_change(angle: float) -> float:
    angle = float(angle)
    if angle < 0:
        angle += 180.0
    return angle


def normalize_angle_0_180(angle: float) -> float:
    angle = float(angle) % 180.0
    if angle < 0:
        angle += 180.0
    return angle


def delete_noise(arr: np.ndarray, axis: str, jump_thresh: float = 100.0) -> np.ndarray:
    arr = np.asarray(arr)

    if arr.ndim != 2 or len(arr) <= 2:
        return arr

    i = 0
    while i < len(arr) - 1:
        if axis == "y":
            if abs(arr[i + 1][1] - arr[i][1]) > jump_thresh:
                arr = np.delete(arr, i + 1, axis=0)
            else:
                i += 1
        elif axis == "x":
            if abs(arr[i + 1][0] - arr[i][0]) > jump_thresh:
                arr = np.delete(arr, i + 1, axis=0)
            else:
                i += 1
        else:
            i += 1

    return arr


def find_duplicates(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)

    if arr1.ndim != 2 or arr2.ndim != 2:
        return np.empty((0, 2), dtype=np.float64)

    if len(arr1) == 0 or len(arr2) == 0:
        return np.empty((0, 2), dtype=np.float64)

    set1 = set(tuple(x) for x in arr1)
    set2 = set(tuple(x) for x in arr2)

    duplicated = list(set1.intersection(set2))

    if len(duplicated) == 0:
        return np.empty((0, 2), dtype=np.float64)

    return np.asarray(duplicated, dtype=np.float64)


def swap_positions(values: List[float], pos1: int, pos2: int) -> List[float]:
    values = list(values)
    values[pos1], values[pos2] = values[pos2], values[pos1]
    return values


def load_image(image: ImageInput) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")

    image_path = Path(image)
    if not image_path.exists():
        raise FileNotFoundError(f"图片不存在：{image_path}")

    return Image.open(image_path).convert("RGB")


class AngleDetector:
    """
    输入：一张图片
    输出：角度值，单位 degree

    检测逻辑：
        1. 先检测一次；
        2. 如果第一次角度处于可靠区间，直接返回；
        3. 如果第一次角度不可靠，把图片旋转 45° 或 -45°；
        4. 再检测一次；
        5. 第二次角度减去旋转角度，得到最终角度。
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: Optional[str] = None,
        max_det: int = 1,
        mask_thresh: float = 0.5,
    ):
        self.model_path = str(model_path)
        self.device = device
        self.max_det = int(max_det)
        self.mask_thresh = float(mask_thresh)

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"YOLO 模型不存在：{self.model_path}")

        print(f"[AngleDetector] 正在加载 YOLO 模型：{self.model_path}")
        self.model = YOLO(self.model_path)
        print("[AngleDetector] YOLO 模型加载完成，后续不会重复加载")

    def detect_angle(
        self,
        image: ImageInput,
        num: int = 0,
        cw: int = 0,
        use_second_detect: bool = True,
    ) -> Optional[float]:
        """
        对外调用函数。

        输入：
            image：一张图片，可以是：
                1. 图片路径
                2. PIL.Image
                3. numpy.ndarray

        输出：
            angle_deg：角度值，单位 degree
        """
        pil_img = load_image(image)

        angle1 = self._detect_once(pil_img, num=num, cw=cw)

        if angle1 is None:
            return None

        angle1 = float(angle1)

        if not use_second_detect:
            return normalize_angle_0_180(angle1)

        # 第一次检测结果可靠，直接返回
        if 20 < angle1 < 70 or 110 < angle1 < 160:
            return normalize_angle_0_180(angle1)

        # 第一次检测结果不可靠，旋转后第二次检测
        if 0 <= angle1 <= 20 or 90 <= angle1 <= 110:
            rotate_angle = 45.0
        else:
            rotate_angle = -45.0

        rotated_img = pil_img.rotate(rotate_angle, expand=False)

        angle2 = self._detect_once(rotated_img, num=num, cw=cw)

        if angle2 is None:
            return normalize_angle_0_180(angle1)

        final_angle = float(angle2) - rotate_angle
        return normalize_angle_0_180(final_angle)

    def _detect_once(
        self,
        image: Image.Image,
        num: int = 0,
        cw: int = 0,
    ) -> Optional[float]:
        """
        单次检测，相当于你原来的 fun()。
        """
        num = int(num) % 4

        results = self.model(
            image,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
        )

        if results is None or len(results) == 0:
            return None

        result = results[0]

        if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
            return None

        if result.boxes is None or len(result.boxes) == 0:
            return None

        mask_np = result.masks.data[0].detach().cpu().numpy()
        mask_bin = (mask_np > self.mask_thresh).astype(np.uint8)

        mask_h, mask_w = mask_bin.shape

        box = result.boxes.xywhn[0].detach().cpu().numpy().astype(float).tolist()
        cxn, cyn, wn, hn = box

        x = cxn * mask_w
        y = cyn * mask_h
        w = wn * mask_w
        h = hn * mask_h

        num0 = w * 0.12

        up = y - h / 2.0 + num0
        down = y + h / 2.0 - num0
        left = x - w / 2.0 + num0
        right = x + w / 2.0 - num0

        lst1 = []
        lst2 = []
        lst3 = []
        lst4 = []

        # 横向扫描边界
        for i in range(mask_h):
            for j in range(mask_w - 1):
                if mask_bin[i, j] != mask_bin[i, j + 1] and mask_bin[i, j] == 0 and left < j < right:
                    lst1.append([j + 1, i])

                if mask_bin[i, j] != mask_bin[i, j + 1] and mask_bin[i, j] == 1 and left < j < right:
                    lst2.append([j, i])

        # 纵向扫描边界
        for i in range(mask_w):
            for j in range(mask_h - 1):
                if mask_bin[j, i] != mask_bin[j + 1, i] and mask_bin[j, i] == 0 and up < j < down:
                    lst3.append([i, j + 1])

                if mask_bin[j, i] != mask_bin[j + 1, i] and mask_bin[j, i] == 1 and up < j < down:
                    lst4.append([i, j])

        arr1 = delete_noise(np.asarray(lst1, dtype=np.float64), "x")
        arr2 = delete_noise(np.asarray(lst2, dtype=np.float64), "x")
        arr3 = delete_noise(np.asarray(lst3, dtype=np.float64), "y")
        arr4 = delete_noise(np.asarray(lst4, dtype=np.float64), "y")

        if len(arr1) < 2 or len(arr2) < 2 or len(arr3) < 2 or len(arr4) < 2:
            return 0.0 if num in (0, 2) else 90.0

        idx1 = int(np.argmin(arr1[:, 0]))
        idx2 = int(np.argmax(arr2[:, 0]))
        idx3 = int(np.argmin(arr3[:, 1]))
        idx4 = int(np.argmax(arr4[:, 1]))

        my_array1 = arr1[:idx1]       # 左上
        my_array2 = arr1[idx1:]       # 左下
        my_array3 = arr2[:idx2]       # 右上
        my_array4 = arr2[idx2:]       # 右下

        my_array5 = arr3[:idx3]       # 左上
        my_array6 = arr3[idx3:]       # 右上
        my_array7 = arr4[:idx4]       # 左下
        my_array8 = arr4[idx4:]       # 右下

        line1 = find_duplicates(my_array4, my_array8)  # 右下
        line2 = find_duplicates(my_array2, my_array7)  # 左下
        line3 = find_duplicates(my_array1, my_array5)  # 左上
        line4 = find_duplicates(my_array3, my_array6)  # 右上

        lines = [line1, line2, line3, line4]

        if not all(len(line) > 2 for line in lines):
            return 0.0 if num in (0, 2) else 90.0

        angles = []

        for line in lines:
            slope, _ = least_squares_fit(line[:, 0], line[:, 1])
            if math.isnan(slope):
                angles.append(90.0)
            else:
                angles.append(angle_change(-math.atan(slope) * 180.0 / math.pi))

        if abs(abs(angles[2] - angles[0])) < 10 and abs(abs(angles[3] - angles[1])) < 10:
            if int(cw) == 0:
                return float(angles[num])
            elif int(cw) == 1:
                return float(swap_positions(angles, 1, 3)[num])
            else:
                return float(angles[num])

        return 0.0 if num in (0, 2) else 90.0