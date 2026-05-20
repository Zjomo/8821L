"""
经典机器视觉光斑检测器 (ClassicSpotDetector)

基于经典图像处理与机器视觉算法的光斑检测与定位模块。

算法原理:
- Otsu Threshold — 大津法自适应阈值分割 (Otsu 1979)
- Morphological Operations — 形态学开运算去噪、闭运算填充
- Contour Analysis — 轮廓提取与形状筛选 (Hu 矩、面积比)
- Connected Components — 连通域分析 (Two-Pass 算法)
- Weighted Centroid — 灰度加权质心定位 (ISO 11146)
- Gaussian Mixture Model — 高斯混合模型背景建模 (GMM)
- Mean Shift — 均值漂移聚类 (Fukunaga & Hostetler 1975)

功能:
- 完全替代深度学习 (YOLO) 的光斑检测功能
- 自适应阈值分割 + 形态学后处理 + 轮廓筛选
- 支持多光斑场景 (返回最亮/最大光斑)
- 亚像素级质心定位
- 背景建模与自适应更新

依赖: numpy, opencv-python (cv2)
不依赖: PyTorch, TensorFlow, ultralytics 或任何深度学习框架
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger("SpotZoom.ClassicDetector")


class DetectionMethod(Enum):
    """光斑检测方法枚举。"""
    OTSU = "otsu"                     # 大津法阈值 + 轮廓筛选
    ADAPTIVE_THRESHOLD = "adaptive"   # 自适应局部阈值
    TOPHAT = "tophat"                 # 顶帽变换 + 阈值
    GMM_BACKGROUND = "gmm"            # 高斯混合模型背景减除
    MEAN_SHIFT = "meanshift"          # 均值漂移密度估计


class SelectionStrategy(Enum):
    """多光斑选择策略。"""
    BRIGHTEST = "brightest"    # 选最亮的
    LARGEST = "largest"        # 选面积最大的
    CENTRAL = "central"        # 选最靠近图像中心的


@dataclass
class SpotDetection:
    """光斑检测结果 (与 SpotZoom.py 中的 SpotDetection 兼容)。"""
    center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    label: str = "lightspot"
    class_id: int = 0


class ClassicSpotDetector:
    """经典机器视觉光斑检测器。

    使用 OpenCV 传统图像处理算法实现光斑检测与定位，
    完全不依赖深度学习框架。

    Parameters
    ----------
    method : DetectionMethod or str
        检测方法。默认 'otsu' (大津法)。
    selection : SelectionStrategy or str
        多光斑选择策略。默认 'brightest'。
    min_area : int
        最小光斑面积 (像素²)。
    max_area : int
        最大光斑面积 (像素²)。0 表示不限制。
    min_circularity : float
        最小圆度 (0~1)。低于此值的候选被过滤。
    min_intensity_ratio : float
        最小强度比 (光斑峰值 / 背景均值)。
    morph_kernel_size : int
        形态学核大小 (奇数)。
    use_subpixel : bool
        是否使用亚像素质心定位。
    gmm_history : int
        GMM 背景模型历史帧数 (仅 GMM 方法)。
    gmm_var_threshold : float
        GMM 方差阈值 (仅 GMM 方法)。
    """

    def __init__(
        self,
        method: DetectionMethod | str = DetectionMethod.OTSU,
        selection: SelectionStrategy | str = SelectionStrategy.BRIGHTEST,
        min_area: int = 20,
        max_area: int = 0,
        min_circularity: float = 0.2,
        min_intensity_ratio: float = 1.5,
        morph_kernel_size: int = 5,
        use_subpixel: bool = True,
        gmm_history: int = 200,
        gmm_var_threshold: float = 16.0,
    ):
        if isinstance(method, str):
            method = DetectionMethod(method)
        if isinstance(selection, str):
            selection = SelectionStrategy(selection)
        self.method = method
        self.selection = selection
        self.min_area = int(min_area)
        self.max_area = int(max_area)
        self.min_circularity = float(min_circularity)
        self.min_intensity_ratio = float(min_intensity_ratio)
        self.morph_kernel_size = int(morph_kernel_size)
        self.use_subpixel = bool(use_subpixel)
        self.gmm_history = int(gmm_history)
        self.gmm_var_threshold = float(gmm_var_threshold)

        # 形态学核
        k = self.morph_kernel_size
        self._kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        self._kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        # GMM 背景减除器 (仅 GMM 方法使用)
        self._bg_subtractor: Optional[cv2.BackgroundSubtractorMOG2] = None
        if self.method == DetectionMethod.GMM_BACKGROUND:
            self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=self.gmm_history,
                varThreshold=self.gmm_var_threshold,
                detectShadows=False,
            )

        self._frame_count = 0

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> Optional[SpotDetection]:
        """检测图像中的光斑。

        Parameters
        ----------
        frame : np.ndarray
            BGR 格式图像。

        Returns
        -------
        Optional[SpotDetection]
            检测结果。未检测到光斑时返回 None。
        """
        if frame is None or frame.size == 0:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.uint8)

        # 根据方法获取候选光斑
        candidates = self._detect_candidates(gray, frame)

        if not candidates:
            return None

        # 多光斑选择
        selected = self._select_candidate(candidates, gray.shape)

        if selected is None:
            return None

        cx, cy, bbox, confidence = selected

        # 亚像素质心精化
        if self.use_subpixel:
            cx, cy = self._refine_subpixel(gray, bbox, cx, cy)

        self._frame_count += 1
        LOGGER.debug(
            "ClassicDetector frame #%d: spot at (%d, %d), conf=%.3f, method=%s",
            self._frame_count, cx, cy, confidence, self.method.value,
        )

        return SpotDetection(
            center=(int(cx), int(cy)),
            bbox=bbox,
            confidence=confidence,
            label="lightspot",
            class_id=0,
        )

    def reset(self) -> None:
        """重置检测器内部状态 (如背景模型)。"""
        self._frame_count = 0
        if self._bg_subtractor is not None:
            self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=self.gmm_history,
                varThreshold=self.gmm_var_threshold,
                detectShadows=False,
            )

    # ------------------------------------------------------------------
    # 检测方法
    # ------------------------------------------------------------------

    def _detect_candidates(
        self, gray: np.ndarray, bgr: np.ndarray
    ) -> List[Tuple[int, int, Tuple[int, int, int, int], float]]:
        """根据所选方法检测候选光斑。

        Returns
        -------
        List of (cx, cy, bbox, confidence)
        """
        if self.method == DetectionMethod.OTSU:
            return self._method_otsu(gray)
        elif self.method == DetectionMethod.ADAPTIVE_THRESHOLD:
            return self._method_adaptive(gray)
        elif self.method == DetectionMethod.TOPHAT:
            return self._method_tophat(gray)
        elif self.method == DetectionMethod.GMM_BACKGROUND:
            return self._method_gmm(gray)
        elif self.method == DetectionMethod.MEAN_SHIFT:
            return self._method_meanshift(gray, bgr)
        else:
            LOGGER.warning("Unknown method %s, falling back to OTSU", self.method)
            return self._method_otsu(gray)

    def _method_otsu(
        self, gray: np.ndarray
    ) -> List[Tuple[int, int, Tuple[int, int, int, int], float]]:
        """大津法自适应阈值 + 形态学后处理 + 轮廓筛选。"""
        # Otsu 阈值
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 形态学: 开运算去噪 → 闭运算填充
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self._kernel_open)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self._kernel_close)

        return self._find_contour_spots(binary, gray)

    def _method_adaptive(
        self, gray: np.ndarray
    ) -> List[Tuple[int, int, Tuple[int, int, int, int], float]]:
        """自适应局部阈值 (Gaussian-weighted)。"""
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, blockSize=31, C=5,
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self._kernel_open)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self._kernel_close)
        return self._find_contour_spots(binary, gray)

    def _method_tophat(
        self, gray: np.ndarray
    ) -> List[Tuple[int, int, Tuple[int, int, int, int], float]]:
        """顶帽变换 (提取亮小目标) + Otsu 阈值。"""
        kernel_size = max(self.morph_kernel_size * 3, 15)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

        _, binary = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self._kernel_open)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self._kernel_close)
        return self._find_contour_spots(binary, gray)

    def _method_gmm(
        self, gray: np.ndarray
    ) -> List[Tuple[int, int, Tuple[int, int, int, int], float]]:
        """高斯混合模型背景减除。"""
        if self._bg_subtractor is None:
            return []

        mask = self._bg_subtractor.apply(gray)
        # GMM 前景掩码
        _, binary = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self._kernel_open)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self._kernel_close)
        return self._find_contour_spots(binary, gray)

    def _method_meanshift(
        self, gray: np.ndarray, bgr: np.ndarray
    ) -> List[Tuple[int, int, Tuple[int, int, int, int], float]]:
        """均值漂移密度估计定位光斑。

        在灰度图上以最亮点为起点进行均值漂移，
        收敛到灰度密度极大值点。
        """
        h, w = gray.shape

        # 找到全局最亮点作为起点
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)
        if max_val < 10:
            return []

        # 均值漂移参数
        start = np.array([float(max_loc[0]), float(max_loc[1])])
        bandwidth = 30.0
        max_iter = 20
        converge_thresh = 0.5

        for _ in range(max_iter):
            # 计算窗口内加权质心
            x0 = max(0, int(start[0] - bandwidth))
            y0 = max(0, int(start[1] - bandwidth))
            x1 = min(w, int(start[0] + bandwidth))
            y1 = min(h, int(start[1] + bandwidth))

            if x1 <= x0 or y1 <= y0:
                break

            window = gray[y0:y1, x0:x1].astype(np.float64)
            total = window.sum()
            if total < 1e-6:
                break

            yy, xx = np.mgrid[y0:y1, x0:x1]
            new_x = float(np.sum(xx * window) / total)
            new_y = float(np.sum(yy * window) / total)

            shift = np.sqrt((new_x - start[0]) ** 2 + (new_y - start[1]) ** 2)
            start = np.array([new_x, new_y])

            if shift < converge_thresh:
                break

        cx, cy = int(round(start[0])), int(round(start[1]))

        # 估计 bbox (带宽为半径)
        bx1 = max(0, cx - int(bandwidth))
        by1 = max(0, cy - int(bandwidth))
        bx2 = min(w, cx + int(bandwidth))
        by2 = min(h, cy + int(bandwidth))

        # 计算置信度 (基于峰值/背景比)
        bg_region = gray[max(0, by1 - 10):by1, bx1:bx2] if by1 > 10 else gray[0:max(1, by1), bx1:bx2]
        bg_mean = float(np.mean(bg_region)) if bg_region.size > 0 else 1.0
        confidence = min(max_val / max(bg_mean, 1.0), 1.0)

        if confidence < 0.1:
            return []

        return [(cx, cy, (bx1, by1, bx2, by2), confidence)]

    # ------------------------------------------------------------------
    # 轮廓分析
    # ------------------------------------------------------------------

    def _find_contour_spots(
        self, binary: np.ndarray, gray: np.ndarray
    ) -> List[Tuple[int, int, Tuple[int, int, int, int], float]]:
        """从二值图中提取轮廓并筛选光斑候选。"""
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        h, w = gray.shape
        bg_mean = float(np.mean(gray)) + 1e-6

        for contour in contours:
            area = cv2.contourArea(contour)

            # 面积过滤
            if area < self.min_area:
                continue
            if self.max_area > 0 and area > self.max_area:
                continue

            # 圆度过滤
            perimeter = cv2.arcLength(contour, True)
            if perimeter < 1e-6:
                continue
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < self.min_circularity:
                continue

            # 外接矩形
            x, y, bw, bh = cv2.boundingRect(contour)
            bbox = (x, y, x + bw, y + bh)

            # 质心
            M = cv2.moments(contour)
            if M["m00"] < 1e-6:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # 强度比 (峰值 / 背景)
            roi = gray[y:y + bh, x:x + bw]
            peak = float(roi.max()) if roi.size > 0 else 0
            intensity_ratio = peak / bg_mean
            if intensity_ratio < self.min_intensity_ratio:
                continue

            # 置信度 = 归一化的综合评分
            conf = self._compute_confidence(
                circularity, intensity_ratio, area, peak
            )

            candidates.append((cx, cy, bbox, conf))

        return candidates

    @staticmethod
    def _compute_confidence(
        circularity: float,
        intensity_ratio: float,
        area: float,
        peak: float,
    ) -> float:
        """计算候选光斑的综合置信度 [0, 1]。"""
        # 圆度分量 (权重 0.3)
        circ_score = min(circularity, 1.0) * 0.3

        # 强度比分量 (权重 0.4)
        intensity_score = min(intensity_ratio / 10.0, 1.0) * 0.4

        # 峰值分量 (权重 0.3)
        peak_score = min(peak / 255.0, 1.0) * 0.3

        return min(circ_score + intensity_score + peak_score, 1.0)

    # ------------------------------------------------------------------
    # 候选选择
    # ------------------------------------------------------------------

    def _select_candidate(
        self,
        candidates: List[Tuple[int, int, Tuple[int, int, int, int], float]],
        image_shape: Tuple[int, int],
    ) -> Optional[Tuple[int, int, Tuple[int, int, int, int], float]]:
        """从候选列表中选择最佳光斑。"""
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        h, w = image_shape
        center_x, center_y = w / 2.0, h / 2.0

        if self.selection == SelectionStrategy.BRIGHTEST:
            # 按置信度 (综合了亮度) 降序
            return max(candidates, key=lambda c: c[3])

        elif self.selection == SelectionStrategy.LARGEST:
            # 按 bbox 面积降序
            return max(
                candidates,
                key=lambda c: (c[2][2] - c[2][0]) * (c[2][3] - c[2][1]),
            )

        elif self.selection == SelectionStrategy.CENTRAL:
            # 按到图像中心的距离升序
            return min(
                candidates,
                key=lambda c: (c[0] - center_x) ** 2 + (c[1] - center_y) ** 2,
            )

        return candidates[0]

    # ------------------------------------------------------------------
    # 亚像素精化
    # ------------------------------------------------------------------

    @staticmethod
    def _refine_subpixel(
        gray: np.ndarray,
        bbox: Tuple[int, int, int, int],
        cx: int,
        cy: int,
    ) -> Tuple[float, float]:
        """在 bbox 内进行灰度加权亚像素精化。"""
        x1, y1, x2, y2 = bbox
        h, w = gray.shape
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(x2, w)
        y2 = min(y2, h)
        if x2 - x1 < 3 or y2 - y1 < 3:
            return (float(cx), float(cy))

        crop = gray[y1:y2, x1:x2].astype(np.float64)

        # 背景估计
        bg = np.percentile(crop, 10)
        signal = np.maximum(crop - bg, 0.0)
        total = signal.sum()
        if total < 1e-6:
            return (float(cx), float(cy))

        yy, xx = np.mgrid[:crop.shape[0], :crop.shape[1]]
        refined_cx = float(np.sum(xx * signal) / total) + x1
        refined_cy = float(np.sum(yy * signal) / total) + y1

        return (refined_cx, refined_cy)
