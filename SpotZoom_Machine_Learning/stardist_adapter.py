"""
StarDist 适配器模块 (StarDistAdapter)

基于 StarDist 的星凸形对象检测算法，适配光斑检测场景。

参考项目: https://github.com/stardist/stardist
论文:
- Schmidt et al. (2018) "Cell Detection with Star-convex Polygons" (MICCAI)
- Weigert et al. (2020) "Star-convex Polyhedra for 3D Object Detection" (WACV)

创新点借鉴:
1. 星凸多边形表示 (Star-convex Polygons)
2. 径向距离回归
3. 非极大值抑制 (NMS)
4. 多类别支持

功能:
- 提供 StarDist 风格的光斑检测
- 星凸形状建模
- 径向距离预测

依赖: numpy, opencv-python
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import cv2
import math


@dataclass
class StarConvexPolygon:
    """星凸多边形表示。"""
    center: Tuple[float, float]  # 中心点
    radii: np.ndarray  # 径向距离 (n_rays,)
    angles: np.ndarray  # 角度 (n_rays,)
    confidence: float  # 置信度

    def to_contour(self) -> np.ndarray:
        """转换为 OpenCV 轮廓格式。"""
        points = []
        for angle, radius in zip(self.angles, self.radii):
            x = self.center[0] + radius * math.cos(angle)
            y = self.center[1] + radius * math.sin(angle)
            points.append([x, y])
        return np.array(points, dtype=np.float32)

    def to_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """转换为二值掩码。"""
        mask = np.zeros(shape, dtype=np.uint8)
        contour = self.to_contour().reshape(-1, 1, 2).astype(np.int32)
        cv2.fillPoly(mask, [contour], 1)
        return mask

    @property
    def area(self) -> float:
        """计算多边形面积 (Shoelace 公式)。"""
        contour = self.to_contour()
        return cv2.contourArea(contour)

    @property
    def diameter(self) -> float:
        """估计直径 (平均径向距离 * 2)。"""
        return 2 * np.mean(self.radii)


@dataclass
class StarDistSpotResult:
    """StarDist 风格的光斑检测结果。"""
    polygon: StarConvexPolygon
    centroid: Tuple[float, float]
    class_id: int  # 类别 ID (支持多类)
    class_name: str


class StarDistAdapter:
    """StarDist 风格的光斑检测适配器。

    借鉴 StarDist 的核心思想，实现经典星凸形状检测：
    1. 径向距离采样
    2. 星凸约束
    3. NMS 后处理

    Parameters
    ----------
    n_rays : int
        径向射线数量。
    prob_threshold : float
        概率阈值。
    nms_threshold : float
        NMS IoU 阈值。
    min_distance : int
        峰值检测最小距离。
    """

    def __init__(
        self,
        n_rays: int = 32,
        prob_threshold: float = 0.5,
        nms_threshold: float = 0.3,
        min_distance: int = 10,
    ):
        self.n_rays = int(n_rays)
        self.prob_threshold = float(prob_threshold)
        self.nms_threshold = float(nms_threshold)
        self.min_distance = int(min_distance)
        self._angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)

    def detect(
        self,
        frame: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[StarDistSpotResult]:
        """检测图像中的光斑。

        Parameters
        ----------
        frame : np.ndarray
            输入图像 (BGR 或灰度)。
        roi : Optional[Tuple[int, int, int, int]]
            感兴趣区域 (x1, y1, x2, y2)。

        Returns
        -------
        List[StarDistSpotResult]
            检测到的光斑列表。
        """
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

        # 检测峰值 (概率图)
        prob_map = self._compute_probability_map(blurred)

        # 峰值检测
        peaks = self._find_peaks(prob_map)

        # 为每个峰值计算星凸多边形
        polygons = []
        for peak in peaks:
            cx, cy = peak
            radii = self._compute_radii(blurred, cx, cy)
            polygon = StarConvexPolygon(
                center=(cx + offset_x, cy + offset_y),
                radii=radii,
                angles=self._angles.copy(),
                confidence=prob_map[cy, cx],
            )
            polygons.append(polygon)

        # NMS
        polygons = self._nms(polygons)

        # 构建结果
        results = []
        for polygon in polygons:
            results.append(StarDistSpotResult(
                polygon=polygon,
                centroid=polygon.center,
                class_id=0,
                class_name="spot",
            ))

        return results

    def _compute_probability_map(self, gray: np.ndarray) -> np.ndarray:
        """计算概率图 (模拟 StarDist 的概率输出)。"""
        # 归一化
        normalized = cv2.normalize(gray, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # 高斯滤波平滑
        prob = cv2.GaussianBlur(normalized, (5, 5), 0)

        # 增强对比度
        prob = np.power(prob, 0.5)

        return prob

    def _find_peaks(self, prob_map: np.ndarray) -> List[Tuple[int, int]]:
        """在概率图中检测峰值。"""
        # 局部最大值
        kernel_size = 2 * self.min_distance + 1
        local_max = cv2.dilate(prob_map, np.ones((kernel_size, kernel_size), np.uint8))

        # 峰值位置
        peaks_mask = (prob_map == local_max) & (prob_map > self.prob_threshold)
        peaks_y, peaks_x = np.where(peaks_mask)

        # 按概率排序
        peak_probs = prob_map[peaks_y, peaks_x]
        sorted_indices = np.argsort(peak_probs)[::-1]

        peaks = []
        for idx in sorted_indices:
            peaks.append((int(peaks_x[idx]), int(peaks_y[idx])))

        return peaks

    def _compute_radii(
        self,
        gray: np.ndarray,
        cx: int,
        cy: int,
    ) -> np.ndarray:
        """计算径向距离。"""
        h, w = gray.shape
        radii = np.zeros(self.n_rays)

        # 归一化灰度
        normalized = cv2.normalize(gray, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        max_radius = min(cx, cy, w - cx, h - cy, 100)

        for i, angle in enumerate(self._angles):
            # 沿射线采样
            for r in range(1, int(max_radius)):
                x = int(cx + r * math.cos(angle))
                y = int(cy + r * math.sin(angle))

                if x < 0 or x >= w or y < 0 or y >= h:
                    radii[i] = r - 1
                    break

                # 检测边缘 (灰度梯度)
                if r > 1:
                    prev_x = int(cx + (r - 1) * math.cos(angle))
                    prev_y = int(cy + (r - 1) * math.sin(angle))
                    grad = abs(float(normalized[y, x]) - float(normalized[prev_y, prev_x]))
                    if grad > 0.1 or normalized[y, x] < 0.3:
                        radii[i] = r - 1
                        break
            else:
                radii[i] = max_radius - 1

        return radii

    def _nms(self, polygons: List[StarConvexPolygon]) -> List[StarConvexPolygon]:
        """非极大值抑制。"""
        if not polygons:
            return []

        # 按置信度排序
        sorted_polygons = sorted(polygons, key=lambda p: p.confidence, reverse=True)

        keep = []
        suppressed = set()

        for i, poly in enumerate(sorted_polygons):
            if i in suppressed:
                continue

            keep.append(poly)

            for j in range(i + 1, len(sorted_polygons)):
                if j in suppressed:
                    continue

                other = sorted_polygons[j]
                iou = self._compute_iou(poly, other)

                if iou > self.nms_threshold:
                    suppressed.add(j)

        return keep

    def _compute_iou(
        self,
        poly1: StarConvexPolygon,
        poly2: StarConvexPolygon,
    ) -> float:
        """计算两个星凸多边形的 IoU。"""
        # 简化的 IoU 计算 (基于中心距离和半径)
        dx = poly1.center[0] - poly2.center[0]
        dy = poly1.center[1] - poly2.center[1]
        center_dist = math.sqrt(dx * dx + dy * dy)

        r1 = np.mean(poly1.radii)
        r2 = np.mean(poly2.radii)

        if center_dist > r1 + r2:
            return 0.0

        # 近似 IoU
        intersection = max(0, min(r1, r2) - center_dist / 2)
        union = r1 + r2 - intersection

        return intersection / union if union > 0 else 0.0


class StarDistMetrics:
    """StarDist 风格的评估指标。"""

    @staticmethod
    def matching(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        thresh: float = 0.5,
    ) -> Dict[str, float]:
        """计算匹配指标。

        Parameters
        ----------
        y_true : np.ndarray
            真实标签图。
        y_pred : np.ndarray
            预测标签图。
        thresh : float
            IoU 阈值。

        Returns
        -------
        Dict[str, float]
            包含 tp, fp, fn, precision, recall, f1, accuracy 的字典。
        """
        # 获取实例
        true_instances = np.unique(y_true)
        true_instances = true_instances[true_instances != 0]

        pred_instances = np.unique(y_pred)
        pred_instances = pred_instances[pred_instances != 0]

        n_true = len(true_instances)
        n_pred = len(pred_instances)

        if n_true == 0 and n_pred == 0:
            return {
                'tp': 0, 'fp': 0, 'fn': 0,
                'precision': 1.0, 'recall': 1.0,
                'f1': 1.0, 'accuracy': 1.0,
            }

        # 计算 IoU 矩阵
        iou_matrix = np.zeros((n_true, n_pred))
        for i, t in enumerate(true_instances):
            true_mask = (y_true == t)
            for j, p in enumerate(pred_instances):
                pred_mask = (y_pred == p)
                intersection = np.logical_and(true_mask, pred_mask).sum()
                union = np.logical_or(true_mask, pred_mask).sum()
                if union > 0:
                    iou_matrix[i, j] = intersection / union

        # 匹配
        tp = 0
        matched_true = set()
        matched_pred = set()

        for i in range(n_true):
            for j in range(n_pred):
                if iou_matrix[i, j] > thresh and i not in matched_true and j not in matched_pred:
                    tp += 1
                    matched_true.add(i)
                    matched_pred.add(j)

        fp = n_pred - len(matched_pred)
        fn = n_true - len(matched_true)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        return {
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
        }
