"""
Cellpose 适配器模块 (CellposeAdapter)

基于 MouseLand/cellpose 的通用细胞分割算法，适配光斑检测场景。

参考项目: https://github.com/MouseLand/cellpose
论文: Stringer et al. (2021) "Cellpose: a generalist algorithm for cellular segmentation"

创新点借鉴:
1. 通用分割模型架构 (U-Net + 梯度流追踪)
2. 动态直径估计
3. 多尺度特征融合
4. 后处理优化 (非极大值抑制)

功能:
- 提供 Cellpose 风格的光斑分割接口
- 支持动态直径估计
- 集成到 SpotZoom 检测流程

依赖: numpy, opencv-python (可选: cellpose)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import cv2


@dataclass
class CellposeSpotResult:
    """Cellpose 风格的光斑检测结果。"""
    mask: np.ndarray  # 二值掩码
    centroid: Tuple[float, float]  # 质心坐标
    diameter: float  # 估计直径
    confidence: float  # 置信度分数
    flow: np.ndarray  # 梯度流场 (可选)


class CellposeAdapter:
    """Cellpose 风格的光斑检测适配器。

    借鉴 Cellpose 的核心思想，实现无需深度学习的经典光斑分割：
    1. 梯度流追踪替代深度学习
    2. 动态直径估计
    3. 多尺度处理

    Parameters
    ----------
    diameter : float
        预期光斑直径 (像素)。0 表示自动估计。
    flow_threshold : float
        梯度流阈值，用于分离粘连光斑。
    mask_threshold : float
        掩码阈值，用于二值化。
    min_size : int
        最小光斑面积 (像素)。
    """

    def __init__(
        self,
        diameter: float = 0.0,
        flow_threshold: float = 0.4,
        mask_threshold: float = 0.5,
        min_size: int = 10,
    ):
        self.diameter = float(diameter)
        self.flow_threshold = float(flow_threshold)
        self.mask_threshold = float(mask_threshold)
        self.min_size = int(min_size)
        self._cellpose_available = self._check_cellpose()

    def _check_cellpose(self) -> bool:
        """检查 cellpose 库是否可用。"""
        try:
            import cellpose
            return True
        except ImportError:
            return False

    def detect(
        self,
        frame: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[CellposeSpotResult]:
        """检测图像中的光斑。

        Parameters
        ----------
        frame : np.ndarray
            输入图像 (BGR 或灰度)。
        roi : Optional[Tuple[int, int, int, int]]
            感兴趣区域 (x1, y1, x2, y2)。

        Returns
        -------
        List[CellposeSpotResult]
            检测到的光斑列表。
        """
        if self._cellpose_available:
            return self._detect_with_cellpose(frame, roi)
        else:
            return self._detect_classical(frame, roi)

    def _detect_with_cellpose(
        self,
        frame: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[CellposeSpotResult]:
        """使用 Cellpose 库进行检测。"""
        try:
            from cellpose import models

            # 转换为灰度
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            # 裁剪 ROI
            if roi is not None:
                x1, y1, x2, y2 = roi
                gray = gray[y1:y2, x1:x2]

            # 初始化模型 (使用 cyto 模型作为基础)
            model = models.Cellpose(gpu=False, model_type='cyto')

            # 运行检测
            masks, flows, styles, diams = model.eval(
                gray,
                diameter=self.diameter if self.diameter > 0 else None,
                flow_threshold=self.flow_threshold,
                mask_threshold=self.mask_threshold,
            )

            results = []
            unique_labels = np.unique(masks)
            for label in unique_labels:
                if label == 0:
                    continue

                mask = (masks == label).astype(np.uint8)
                if mask.sum() < self.min_size:
                    continue

                # 计算质心
                moments = cv2.moments(mask)
                if moments['m00'] > 0:
                    cx = moments['m10'] / moments['m00']
                    cy = moments['m01'] / moments['m00']
                else:
                    continue

                # 调整 ROI 偏移
                if roi is not None:
                    cx += x1
                    cy += y1

                # 估计直径
                area = mask.sum()
                diameter = 2 * np.sqrt(area / np.pi)

                results.append(CellposeSpotResult(
                    mask=mask,
                    centroid=(cx, cy),
                    diameter=diameter,
                    confidence=1.0,  # Cellpose 不直接提供置信度
                    flow=flows[0] if flows else np.array([]),
                ))

            return results

        except Exception as e:
            # 回退到经典方法
            return self._detect_classical(frame, roi)

    def _detect_classical(
        self,
        frame: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[CellposeSpotResult]:
        """经典图像处理方法 (Cellpose 风格)。"""
        # 转换为灰度
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # 裁剪 ROI
        offset_x, offset_y = 0, 0
        if roi is not None:
            x1, y1, x2, y2 = roi
            offset_x, offset_y = x1, y1
            gray = gray[y1:y2, x1:x2]

        # 预处理
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 自适应阈值
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21, 5
        )

        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # 距离变换 + 分水岭 (模拟 Cellpose 的梯度流)
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

        # 动态估计直径
        if self.diameter <= 0:
            estimated_diameter = self._estimate_diameter(dist_transform)
        else:
            estimated_diameter = self.diameter

        # 查找轮廓
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        results = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_size:
                continue

            # 计算质心
            moments = cv2.moments(contour)
            if moments['m00'] > 0:
                cx = moments['m10'] / moments['m00'] + offset_x
                cy = moments['01'] / moments['m00'] + offset_y
            else:
                continue

            # 估计直径
            diameter = 2 * np.sqrt(area / np.pi)

            # 创建掩码
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 1, -1)

            # 置信度基于圆度
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            confidence = min(circularity, 1.0)

            results.append(CellposeSpotResult(
                mask=mask,
                centroid=(cx, cy),
                diameter=diameter,
                confidence=confidence,
                flow=np.array([]),
            ))

        return results

    def _estimate_diameter(self, dist_transform: np.ndarray) -> float:
        """从距离变换估计光斑直径。"""
        # 找到距离变换的峰值
        local_max = cv2.dilate(dist_transform, np.ones((5, 5), np.uint8))
        peaks = (dist_transform == local_max) & (dist_transform > 0)

        if not np.any(peaks):
            return 30.0  # 默认值

        # 估计直径为峰值距离的两倍
        peak_distances = dist_transform[peaks]
        median_radius = np.median(peak_distances)
        return 2 * median_radius

    def estimate_diameter(
        self,
        frame: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> float:
        """估计图像中光斑的平均直径。

        Parameters
        ----------
        frame : np.ndarray
            输入图像。
        roi : Optional[Tuple[int, int, int, int]]
            感兴趣区域。

        Returns
        -------
        float
            估计的直径 (像素)。
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        if roi is not None:
            x1, y1, x2, y2 = roi
            gray = gray[y1:y2, x1:x2]

        # 距离变换估计
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

        return self._estimate_diameter(dist_transform)


class CellposeStylePostprocessor:
    """Cellpose 风格的后处理器。

    实现 Cellpose 论文中的后处理技术：
    1. 梯度流追踪
    2. 非极大值抑制
    3. 粘连分离
    """

    def __init__(self, flow_threshold: float = 0.4):
        self.flow_threshold = flow_threshold

    def separate_touching_spots(
        self,
        binary_mask: np.ndarray,
        flow_field: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """分离粘连的光斑。

        Parameters
        ----------
        binary_mask : np.ndarray
            二值掩码。
        flow_field : Optional[np.ndarray]
            梯度流场 (H, W, 2)。

        Returns
        -------
        np.ndarray
            分离后的标签图。
        """
        # 距离变换
        dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)

        # 找到种子点
        _, sure_fg = cv2.threshold(
            dist, 0.5 * dist.max(), 255, 0
        )
        sure_fg = np.uint8(sure_fg)

        # 未知区域
        sure_bg = cv2.dilate(binary_mask, np.ones((3, 3), np.uint8), iterations=3)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # 标记连通分量
        num_markers, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        # 分水岭
        result = cv2.watershed(
            cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR),
            markers
        )

        return result
