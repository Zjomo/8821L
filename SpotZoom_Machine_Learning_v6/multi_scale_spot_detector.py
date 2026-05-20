"""
Multi-Scale Spot Detector - 多尺度光斑检测器

Inspired by:
- DeepTrack2 (DeepTrackAI): Multi-scale feature pyramid for particle detection
- YOLOv11 (Ultralytics): Multi-scale detection heads with PANet
- EfficientDet (Google): BiFPN cross-scale feature fusion

Core Innovation:
- 多尺度特征金字塔检测，适应不同大小的光斑
- 跨尺度特征融合 (BiFPN-like)，提升小光斑检测率
- 自适应尺度选择，根据历史检测自动选择最优尺度
- 纯 numpy+cv2 实现，零外部依赖
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class MultiScaleDetectorConfig:
    """多尺度检测器配置"""
    # 检测尺度列表 (高斯模糊核大小)
    scales: Tuple[int, ...] = (3, 7, 15, 25)
    # 每个尺度的最小斑点面积
    min_areas: Tuple[int, ...] = (10, 30, 100, 300)
    # 每个尺度的最大斑点面积
    max_areas: Tuple[int, ...] = (500, 2000, 8000, 30000)
    # 跨尺度融合权重 (越大越信任大尺度)
    scale_fusion_weights: Tuple[float, ...] = (0.2, 0.3, 0.3, 0.2)
    # 自适应尺度选择的历史窗口
    adaptive_history_length: int = 20
    # 置信度阈值
    confidence_threshold: float = 0.3
    # 非极大值抑制 IoU 阈值
    nms_iou_threshold: float = 0.5
    # 是否启用自适应尺度
    adaptive_scale_enabled: bool = True


@dataclass
class MultiScaleDetectionResult:
    """多尺度检测结果"""
    center: Optional[Tuple[float, float]]
    confidence: float
    best_scale: int  # 最优尺度索引
    scale_scores: List[float]  # 各尺度置信度
    all_detections: List[Tuple[float, float, float]]  # (x, y, conf)
    fusion_map: Optional[np.ndarray] = None  # 融合响应图


class MultiScaleSpotDetector:
    """多尺度光斑检测器

    使用多尺度特征金字塔检测不同大小的光斑，
    通过跨尺度融合提升检测鲁棒性。

    Inspired by DeepTrack2's multi-scale particle detection and
    YOLOv11's feature pyramid network.
    """

    def __init__(self, config: Optional[MultiScaleDetectorConfig] = None):
        self.config = config or MultiScaleDetectorConfig()
        self._scale_history: Deque[int] = deque(maxlen=self.config.adaptive_history_length)
        self._prev_detections: Deque[MultiScaleDetectionResult] = deque(maxlen=5)

    def reset(self) -> None:
        """重置检测器状态"""
        self._scale_history.clear()
        self._prev_detections.clear()

    def _detect_at_scale(
        self, gray: np.ndarray, scale_idx: int
    ) -> List[Tuple[float, float, float]]:
        """在指定尺度上检测光斑

        Args:
            gray: 灰度图像
            scale_idx: 尺度索引

        Returns:
            检测结果列表 [(x, y, confidence), ...]
        """
        ksize = self.config.scales[scale_idx]
        min_area = self.config.min_areas[scale_idx]
        max_area = self.config.max_areas[scale_idx]

        # 多尺度预处理
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)

        # 自适应阈值检测
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, ksize * 2 + 1, -5
        )

        # 形态学操作去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 连通域分析
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            M = cv2.moments(cnt)
            if M["m00"] < 1e-6:
                continue

            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            # 计算置信度: 基于圆度和对比度
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0.0

            # 局部对比度
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mean_val = cv2.mean(gray, mask=mask)[0]
            bg_mask = cv2.bitwise_not(mask)
            if np.count_nonzero(bg_mask) > 0:
                bg_mean = cv2.mean(gray, mask=bg_mask)[0]
                contrast = max(0, (mean_val - bg_mean) / (bg_mean + 1e-6))
            else:
                contrast = 0.0

            confidence = 0.4 * min(1.0, circularity) + 0.6 * min(1.0, contrast / 2.0)
            detections.append((cx, cy, confidence))

        return detections

    def _build_fusion_map(
        self, gray: np.ndarray, scale_detections: List[List[Tuple[float, float, float]]]
    ) -> np.ndarray:
        """构建跨尺度融合响应图 (BiFPN-like)

        Args:
            gray: 灰度图像
            scale_detections: 各尺度检测结果

        Returns:
            融合响应图
        """
        h, w = gray.shape
        fusion = np.zeros((h, w), dtype=np.float32)

        for scale_idx, dets in enumerate(scale_detections):
            weight = self.config.scale_fusion_weights[scale_idx]
            for cx, cy, conf in dets:
                # 高斯响应核
                radius = int(self.config.scales[scale_idx] * 1.5)
                y_min = max(0, int(cy) - radius)
                y_max = min(h, int(cy) + radius + 1)
                x_min = max(0, int(cx) - radius)
                x_max = min(w, int(cx) + radius + 1)

                yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
                gaussian = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (radius / 2) ** 2))
                fusion[y_min:y_max, x_min:x_max] += weight * conf * gaussian

        return fusion

    def _non_max_suppression(
        self, detections: List[Tuple[float, float, float]], iou_threshold: float
    ) -> List[Tuple[float, float, float]]:
        """非极大值抑制"""
        if not detections:
            return []

        # 按置信度排序
        dets = sorted(detections, key=lambda x: x[2], reverse=True)
        kept = []

        while dets:
            best = dets.pop(0)
            kept.append(best)
            dets = [
                d for d in dets
                if np.sqrt((d[0] - best[0]) ** 2 + (d[1] - best[1]) ** 2) > 10
            ]

        return kept

    def _select_adaptive_scale(self) -> Optional[int]:
        """根据历史检测自适应选择最优尺度"""
        if len(self._scale_history) < 3:
            return None

        # 统计最近检测中最常出现的尺度
        scale_counts = [0] * len(self.config.scales)
        for s in self._scale_history:
            scale_counts[s] += 1

        return int(np.argmax(scale_counts))

    def detect(self, frame: np.ndarray) -> MultiScaleDetectionResult:
        """执行多尺度光斑检测

        Args:
            frame: BGR 或灰度图像

        Returns:
            MultiScaleDetectionResult 检测结果
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # 自适应尺度选择
        preferred_scale = None
        if self.config.adaptive_scale_enabled:
            preferred_scale = self._select_adaptive_scale()

        # 多尺度检测
        scale_detections = []
        scale_scores = []
        for i in range(len(self.config.scales)):
            dets = self._detect_at_scale(gray, i)
            scale_detections.append(dets)
            max_conf = max((d[2] for d in dets), default=0.0)
            scale_scores.append(max_conf)

        # 构建融合响应图
        fusion_map = self._build_fusion_map(gray, scale_detections)

        # 跨尺度融合：合并所有检测结果
        all_dets = []
        for i, dets in enumerate(scale_detections):
            weight = self.config.scale_fusion_weights[i]
            if preferred_scale is not None and i != preferred_scale:
                weight *= 0.7  # 降低非偏好尺度的权重
            for cx, cy, conf in dets:
                all_dets.append((cx, cy, conf * weight))

        # NMS
        all_dets = self._non_max_suppression(all_dets, self.config.nms_iou_threshold)

        # 选择最佳检测
        if all_dets and all_dets[0][2] >= self.config.confidence_threshold:
            best = all_dets[0]
            best_scale = int(np.argmax(scale_scores))
            self._scale_history.append(best_scale)

            result = MultiScaleDetectionResult(
                center=(best[0], best[1]),
                confidence=best[2],
                best_scale=best_scale,
                scale_scores=scale_scores,
                all_detections=all_dets,
                fusion_map=fusion_map,
            )
        else:
            result = MultiScaleDetectionResult(
                center=None,
                confidence=0.0,
                best_scale=-1,
                scale_scores=scale_scores,
                all_detections=all_dets,
                fusion_map=fusion_map,
            )

        self._prev_detections.append(result)
        return result

    def get_diagnostics(self) -> dict:
        """获取检测器诊断信息"""
        return {
            "scale_history": list(self._scale_history),
            "preferred_scale": self._select_adaptive_scale(),
            "num_prev_detections": len(self._prev_detections),
            "scales": self.config.scales,
        }
