"""
SAM 2 风格光斑分割器 (SAM2SpotSegmenter)

灵感来源:
- Meta SAM 2 (Segment Anything Model 2) — Meta 的通用图像分割模型，
  支持提示引导分割 (点/框/掩码) 和视频流式记忆追踪。
  项目地址: https://github.com/facebookresearch/segment-anything-2
- GrabCut 算法 — Rother et al. (2004) "GrabCut: Interactive Foreground
  Extraction using Iterated Graph Cuts"，基于图割的交互式前景分割。
- Canny 边缘检测 — Canny (1986) 边缘检测算子，用于边缘感知的掩码精修。

算法原理:
- GrabCut Segmentation — 以边界框为提示，通过迭代图割算法实现
  前景/背景分割，生成像素级掩码。
- Edge-Aware Refinement — 利用 Canny 边缘检测对齐掩码边界与图像
  真实边缘，消除分割伪影。
- Template Matching Tracking — 基于模板匹配的时序追踪，维护记忆库
  实现跨帧光斑追踪 (灵感来自 SAM2 的流式记忆机制)。
- Morphological Analysis — 基于图像矩和轮廓分析的形态学度量。

功能:
- 像素级光斑分割 (超越 YOLO 边界框检测)
- 支持边界框提示和中心点提示
- 多光斑批量分割 (与 YOLO 检测结果联动)
- 边缘感知掩码精修
- 基于记忆的时序光斑追踪
- 光斑形态学度量 (面积、周长、圆度、离心率、质心)
- IoU 计算与轮廓提取

依赖: numpy, opencv-python (cv2)
不依赖: PyTorch, Meta SAM2 或任何深度学习框架
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger("SpotZoom.SAM2SpotSegmenter")


# ---------------------------------------------------------------------------
# 配置与结果数据类
# ---------------------------------------------------------------------------


@dataclass
class SAM2SpotConfig:
    """SAM2 风格光斑分割器的配置参数。

    Attributes
    ----------
    grabcut_iters : int
        GrabCut 算法迭代次数。值越大分割越精细，但耗时更长。
    edge_threshold_low : int
        Canny 边缘检测低阈值。
    edge_threshold_high : int
        Canny 边缘检测高阈值。
    edge_dilate_kernel : int
        边缘膨胀核大小，用于扩大边缘影响区域。
    memory_length : int
        记忆库最大长度 (帧数)，用于时序追踪。
    min_area : int
        最小光斑面积 (像素²)，低于此值的分割结果将被丢弃。
    max_area : int
        最大光斑面积 (像素²)，超过此值的分割结果将被丢弃。0 表示不限制。
    refine_iterations : int
        边缘感知精修迭代次数。
    template_match_method : int
        OpenCV 模板匹配方法，默认 cv2.TM_CCOEFF_NORMED。
    track_confidence_threshold : float
        追踪置信度阈值，低于此值认为追踪失败。
    morph_open_kernel : int
        形态学开运算核大小，用于去噪。
    morph_close_kernel : int
        形态学闭运算核大小，用于填充孔洞。
    """

    grabcut_iters: int = 5
    edge_threshold_low: int = 30
    edge_threshold_high: int = 100
    edge_dilate_kernel: int = 3
    memory_length: int = 10
    min_area: int = 20
    max_area: int = 0
    refine_iterations: int = 3
    template_match_method: int = cv2.TM_CCOEFF_NORMED
    track_confidence_threshold: float = 0.3
    morph_open_kernel: int = 3
    morph_close_kernel: int = 3


@dataclass
class SAM2SegmentResult:
    """SAM2 风格分割结果。

    Attributes
    ----------
    mask : np.ndarray
        二值分割掩码 (H, W)，前景为 255，背景为 0。
    contour : np.ndarray
        轮廓点数组，形状为 (N, 1, 2) 或空数组。
    confidence : float
        分割置信度 [0, 1]。
    morphology : dict
        形态学度量字典，包含 area, perimeter, circularity,
        eccentricity, centroid 等字段。
    bbox : Tuple[int, int, int, int]
        边界框 (x1, y1, x2, y2)，由掩码计算得出。
    """

    mask: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.uint8))
    contour: np.ndarray = field(default_factory=lambda: np.zeros((0, 1, 2), dtype=np.int32))
    confidence: float = 0.0
    morphology: Dict[str, Any] = field(default_factory=dict)
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)


# ---------------------------------------------------------------------------
# SAM2 风格光斑分割器
# ---------------------------------------------------------------------------


class SAM2SpotSegmenter:
    """SAM 2 风格光斑分割器。

    借鉴 Meta SAM 2 的提示引导分割和流式记忆追踪思想，
    使用 GrabCut 算法实现纯 OpenCV + NumPy 的像素级光斑分割。

    核心流程:
    1. 接收边界框或中心点提示
    2. 使用 GrabCut 生成分割掩码
    3. 边缘感知精修 (Canny + 形态学)
    4. 提取轮廓与形态学度量
    5. 可选: 基于记忆库的时序追踪

    Parameters
    ----------
    config : SAM2SpotConfig or None
        配置参数。为 None 时使用默认配置。
    """

    def __init__(self, config: Optional[SAM2SpotConfig] = None) -> None:
        """初始化分割器。

        Parameters
        ----------
        config : SAM2SpotConfig or None
            配置参数实例。为 None 时使用默认值。
        """
        self.config = config if config is not None else SAM2SpotConfig()

        # 记忆库: 存储历史掩码和对应图像块，用于时序追踪
        self._memory_bank: Deque[Tuple[np.ndarray, np.ndarray]] = deque(
            maxlen=self.config.memory_length
        )

        # 运行统计
        self._segment_count: int = 0
        self._track_count: int = 0
        self._refine_count: int = 0

        LOGGER.info(
            "SAM2SpotSegmenter: 初始化完成 "
            "(grabcut_iters=%d, edge=[%d,%d], memory_len=%d, min_area=%d)",
            self.config.grabcut_iters,
            self.config.edge_threshold_low,
            self.config.edge_threshold_high,
            self.config.memory_length,
            self.config.min_area,
        )

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def segment(
        self,
        image: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        center_point: Optional[Tuple[int, int]] = None,
    ) -> SAM2SegmentResult:
        """从图像中分割光斑。

        支持两种提示模式:
        - 边界框提示 (bbox): 直接用于 GrabCut 初始化。
        - 中心点提示 (center_point): 先估计边界框再进行分割。

        Parameters
        ----------
        image : np.ndarray
            输入图像 (灰度或 BGR)。
        bbox : Tuple[int, int, int, int] or None
            边界框提示 (x1, y1, x2, y2)。
        center_point : Tuple[int, int] or None
            中心点提示 (cx, cy)。

        Returns
        -------
        SAM2SegmentResult
            分割结果，包含掩码、轮廓、置信度和形态学度量。

        Raises
        ------
        ValueError
            当 bbox 和 center_point 均为 None 时。
        """
        self._validate_image(image)

        if bbox is None and center_point is None:
            raise ValueError("必须提供 bbox 或 center_point 中的至少一个提示")

        # 转为 BGR (GrabCut 需要 3 通道)
        bgr = self._ensure_bgr(image)

        # 确定边界框
        if bbox is not None:
            x1, y1, x2, y2 = self._clamp_bbox(bbox, bgr.shape)
        else:
            x1, y1, x2, y2 = self._point_to_bbox(center_point, bgr.shape)

        # GrabCut 分割
        mask = self._grabcut_segment(bgr, (x1, y1, x2, y2))

        # 面积过滤
        area = int(np.sum(mask > 0))
        if area < self.config.min_area:
            LOGGER.debug(
                "segment: 掩码面积 %d < min_area %d，返回空结果",
                area, self.config.min_area,
            )
            self._segment_count += 1
            return SAM2SegmentResult(
                mask=np.zeros(bgr.shape[:2], dtype=np.uint8),
                confidence=0.0,
            )

        if self.config.max_area > 0 and area > self.config.max_area:
            LOGGER.debug(
                "segment: 掩码面积 %d > max_area %d，返回空结果",
                area, self.config.max_area,
            )
            self._segment_count += 1
            return SAM2SegmentResult(
                mask=np.zeros(bgr.shape[:2], dtype=np.uint8),
                confidence=0.0,
            )

        # 边缘感知精修
        refined_mask = self.refine_mask(mask, bgr)
        self._refine_count += 1

        # 构建结果
        contour = self.extract_contour(refined_mask)
        morphology = self.compute_spot_morphology(refined_mask, bgr)
        confidence = self._compute_confidence(refined_mask, bgr, (x1, y1, x2, y2))
        result_bbox = self._mask_to_bbox(refined_mask)

        # 更新记忆库
        self._memory_bank.append((refined_mask.copy(), bgr.copy()))

        self._segment_count += 1
        LOGGER.debug(
            "segment: 完成 (area=%d, confidence=%.3f, bbox=%s)",
            morphology.get("area", 0), confidence, result_bbox,
        )

        return SAM2SegmentResult(
            mask=refined_mask,
            contour=contour,
            confidence=confidence,
            morphology=morphology,
            bbox=result_bbox,
        )

    def segment_multi(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
    ) -> List[SAM2SegmentResult]:
        """批量分割多个光斑。

        接收 YOLO 风格的检测结果列表，对每个检测执行像素级分割。

        Parameters
        ----------
        image : np.ndarray
            输入图像 (灰度或 BGR)。
        detections : list of dict
            检测结果列表，每个元素为字典，需包含 'bbox' 键
            (格式为 [x1, y1, x2, y2]) 或 'center' 键 (格式为 [cx, cy])。

        Returns
        -------
        list of SAM2SegmentResult
            分割结果列表，与 detections 一一对应。
        """
        self._validate_image(image)
        results: List[SAM2SegmentResult] = []

        for idx, det in enumerate(detections):
            bbox = det.get("bbox")
            center = det.get("center")

            # 转换 bbox 格式: list/tuple -> tuple
            if bbox is not None:
                bbox = tuple(int(v) for v in bbox)  # type: ignore[assignment]
            if center is not None:
                center = (int(center[0]), int(center[1]))  # type: ignore[assignment]

            try:
                result = self.segment(image, bbox=bbox, center_point=center)
                results.append(result)
            except Exception as exc:
                LOGGER.warning(
                    "segment_multi: 第 %d 个检测分割失败: %s", idx, exc
                )
                results.append(SAM2SegmentResult())

        LOGGER.info(
            "segment_multi: 完成 %d/%d 个光斑分割",
            sum(1 for r in results if r.confidence > 0),
            len(detections),
        )
        return results

    def compute_iou(self, mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        """计算两个掩码之间的交并比 (IoU)。

        Parameters
        ----------
        mask_a : np.ndarray
            第一个二值掩码。
        mask_b : np.ndarray
            第二个二值掩码。

        Returns
        -------
        float
            IoU 值，范围 [0, 1]。掩码为空时返回 0.0。
        """
        if mask_a is None or mask_b is None:
            return 0.0
        if mask_a.size == 0 or mask_b.size == 0:
            return 0.0

        # 确保二值化
        a = (mask_a > 0).astype(np.uint8)
        b = (mask_b > 0).astype(np.uint8)

        # 尺寸不一致时取交集区域
        if a.shape != b.shape:
            h = min(a.shape[0], b.shape[0])
            w = min(a.shape[1], b.shape[1])
            a = a[:h, :w]
            b = b[:h, :w]

        intersection = int(np.sum(a & b))
        union = int(np.sum(a | b))

        if union == 0:
            return 0.0

        return float(intersection) / float(union)

    def refine_mask(
        self,
        mask: np.ndarray,
        image: np.ndarray,
    ) -> np.ndarray:
        """边缘感知掩码精修。

        使用 Canny 边缘检测对齐掩码边界与图像真实边缘，
        并通过形态学操作去除噪声和填充孔洞。

        Parameters
        ----------
        mask : np.ndarray
            初始二值掩码。
        image : np.ndarray
            原始图像 (灰度或 BGR)。

        Returns
        -------
        np.ndarray
            精修后的二值掩码。
        """
        if mask is None or mask.size == 0:
            return mask

        refined = mask.copy()

        # 转为灰度
        gray = self._ensure_gray(image)

        for iteration in range(self.config.refine_iterations):
            # 1. Canny 边缘检测
            edges = cv2.Canny(
                gray,
                self.config.edge_threshold_low,
                self.config.edge_threshold_high,
            )

            # 2. 膨胀边缘以扩大影响区域
            if self.config.edge_dilate_kernel > 0:
                ksize = self.config.edge_dilate_kernel
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (ksize, ksize)
                )
                edges = cv2.dilate(edges, kernel, iterations=1)

            # 3. 在边缘附近修正掩码:
            #    - 如果掩码边界外的像素有强边缘，将其纳入掩码
            #    - 如果掩码边界内的像素有强边缘且灰度值低，将其移除
            edge_region = (edges > 0) & (refined == 0)
            if np.any(edge_region):
                # 扩展掩码到边缘区域 (如果边缘区域灰度足够高)
                mean_val = float(np.mean(gray[refined > 0])) if np.any(refined > 0) else 128.0
                threshold = mean_val * 0.3
                expand_condition = edge_region & (gray >= threshold)
                refined[expand_condition] = 255

            # 4. 形态学开运算 (去噪)
            if self.config.morph_open_kernel > 0:
                ksize = self.config.morph_open_kernel
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (ksize, ksize)
                )
                refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)

            # 5. 形态学闭运算 (填充孔洞)
            if self.config.morph_close_kernel > 0:
                ksize = self.config.morph_close_kernel
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (ksize, ksize)
                )
                refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)

        # 最终面积验证
        area = int(np.sum(refined > 0))
        if area < self.config.min_area:
            LOGGER.debug(
                "refine_mask: 精修后面积 %d < min_area %d，返回原始掩码",
                area, self.config.min_area,
            )
            return mask

        return refined

    def track_spot(
        self,
        image: np.ndarray,
        prev_mask: np.ndarray,
    ) -> SAM2SegmentResult:
        """基于记忆的时序光斑追踪。

        受 SAM 2 流式记忆机制启发，维护历史掩码记忆库，
        使用模板匹配在新帧中定位光斑并更新分割。

        Parameters
        ----------
        image : np.ndarray
            当前帧图像 (灰度或 BGR)。
        prev_mask : np.ndarray
            上一帧的分割掩码。

        Returns
        -------
        SAM2SegmentResult
            追踪分割结果。追踪失败时返回空结果。
        """
        self._validate_image(image)

        if prev_mask is None or prev_mask.size == 0:
            LOGGER.warning("track_spot: prev_mask 为空，无法追踪")
            return SAM2SegmentResult()

        bgr = self._ensure_bgr(image)
        gray = self._ensure_gray(image)
        h, w = bgr.shape[:2]

        # 从上一帧掩码中提取模板区域
        prev_bbox = self._mask_to_bbox(prev_mask)
        if prev_bbox == (0, 0, 0, 0):
            LOGGER.warning("track_spot: prev_mask 无有效区域")
            return SAM2SegmentResult()

        px1, py1, px2, py2 = prev_bbox
        pw, ph = px2 - px1, py2 - py1

        # 扩大搜索范围 (1.5 倍)
        margin_x = int(pw * 0.25)
        margin_y = int(ph * 0.25)
        sx1 = max(0, px1 - margin_x)
        sy1 = max(0, py1 - margin_y)
        sx2 = min(w, px2 + margin_x)
        sy2 = min(h, py2 + margin_y)

        if sx2 <= sx1 or sy2 <= sy1:
            return SAM2SegmentResult()

        # 提取模板 (上一帧掩码区域的灰度图像)
        template_gray = self._ensure_gray(
            self._memory_bank[-1][1] if self._memory_bank else gray
        )
        tpl_x1 = max(0, px1)
        tpl_y1 = max(0, py1)
        tpl_x2 = min(template_gray.shape[1], px2)
        tpl_y2 = min(template_gray.shape[0], py2)
        template = template_gray[tpl_y1:tpl_y2, tpl_x1:tpl_x2]

        if template.size == 0 or template.shape[0] < 2 or template.shape[1] < 2:
            return SAM2SegmentResult()

        # 在当前帧搜索区域进行模板匹配
        search_region = gray[sy1:sy2, sx1:sx2]
        if (
            search_region.shape[0] <= template.shape[0]
            or search_region.shape[1] <= template.shape[1]
        ):
            return SAM2SegmentResult()

        match_result = cv2.matchTemplate(
            search_region,
            template,
            self.config.template_match_method,
        )
        _, max_val, _, max_loc = cv2.minMaxLoc(match_result)

        if max_val < self.config.track_confidence_threshold:
            LOGGER.debug(
                "track_spot: 模板匹配置信度 %.3f < 阈值 %.3f，追踪失败",
                max_val, self.config.track_confidence_threshold,
            )
            self._track_count += 1
            return SAM2SegmentResult(confidence=float(max_val))

        # 计算新边界框
        match_x, match_y = max_loc
        new_x1 = sx1 + match_x
        new_y1 = sy1 + match_y
        new_x2 = new_x1 + template.shape[1]
        new_y2 = new_y1 + template.shape[0]

        # 限制在图像范围内
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)
        new_x2 = min(w, new_x2)
        new_y2 = min(h, new_y2)

        # 使用新边界框进行 GrabCut 分割
        new_mask = self._grabcut_segment(bgr, (new_x1, new_y1, new_x2, new_y2))

        # 使用上一帧掩码作为先验进行融合
        if prev_mask.shape == new_mask.shape:
            # 与上一帧掩码取交集，增强时序一致性
            intersection = (new_mask > 0) & (prev_mask > 0)
            if np.sum(intersection) > self.config.min_area * 0.5:
                # 保留新掩码中与旧掩码重叠的部分，同时保留新掩码的扩展区域
                new_mask[intersection] = 255

        # 精修
        refined_mask = self.refine_mask(new_mask, bgr)
        self._refine_count += 1

        # 面积过滤
        area = int(np.sum(refined_mask > 0))
        if area < self.config.min_area:
            self._track_count += 1
            return SAM2SegmentResult(confidence=float(max_val))

        # 构建结果
        contour = self.extract_contour(refined_mask)
        morphology = self.compute_spot_morphology(refined_mask, bgr)
        result_bbox = self._mask_to_bbox(refined_mask)

        # 更新记忆库
        self._memory_bank.append((refined_mask.copy(), bgr.copy()))

        self._track_count += 1
        LOGGER.debug(
            "track_spot: 追踪成功 (confidence=%.3f, area=%d, bbox=%s)",
            max_val, morphology.get("area", 0), result_bbox,
        )

        return SAM2SegmentResult(
            mask=refined_mask,
            contour=contour,
            confidence=float(max_val),
            morphology=morphology,
            bbox=result_bbox,
        )

    def extract_contour(self, mask: np.ndarray) -> np.ndarray:
        """从掩码中提取轮廓点。

        Parameters
        ----------
        mask : np.ndarray
            二值掩码。

        Returns
        -------
        np.ndarray
            轮廓点数组，形状为 (N, 1, 2)。无轮廓时返回空数组。
        """
        if mask is None or mask.size == 0:
            return np.zeros((0, 1, 2), dtype=np.int32)

        binary = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return np.zeros((0, 1, 2), dtype=np.int32)

        # 返回最大轮廓
        max_contour = max(contours, key=cv2.contourArea)
        return max_contour

    def compute_spot_morphology(
        self,
        mask: np.ndarray,
        image: np.ndarray,
    ) -> Dict[str, Any]:
        """计算光斑形态学度量。

        Parameters
        ----------
        mask : np.ndarray
            二值分割掩码。
        image : np.ndarray
            原始图像 (用于计算强度相关度量)。

        Returns
        -------
        dict
            形态学度量字典，包含以下字段:
            - area: 面积 (像素²)
            - perimeter: 周长 (像素)
            - circularity: 圆度 [0, 1]，1 为完美圆形
            - eccentricity: 离心率 [0, 1]，0 为圆形
            - centroid: 质心坐标 (cx, cy)
            - bbox_area: 边界框面积
            - extent: 面积占边界框比例
            - mean_intensity: 掩码区域平均灰度
            - max_intensity: 掩码区域最大灰度
            - solidity: 轮廓实心度 [0, 1]
        """
        result: Dict[str, Any] = {
            "area": 0,
            "perimeter": 0.0,
            "circularity": 0.0,
            "eccentricity": 0.0,
            "centroid": (0.0, 0.0),
            "bbox_area": 0,
            "extent": 0.0,
            "mean_intensity": 0.0,
            "max_intensity": 0.0,
            "solidity": 0.0,
        }

        if mask is None or mask.size == 0:
            return result

        binary = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return result

        # 使用最大轮廓
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)

        if area < 1:
            return result

        perimeter = cv2.arcLength(contour, True)

        # 圆度: 4 * pi * area / perimeter^2
        if perimeter > 0:
            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        else:
            circularity = 0.0

        # 拟合椭圆计算离心率
        eccentricity = 0.0
        if len(contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(contour)
                (cx, cy), (ma, mi), angle = ellipse
                if ma > 0:
                    eccentricity = math.sqrt(1.0 - (min(ma, mi) / max(ma, mi)) ** 2)
            except cv2.error:
                eccentricity = 0.0

        # 图像矩计算质心
        moments = cv2.moments(binary)
        if moments["m00"] > 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            centroid = (float(cx), float(cy))
        else:
            centroid = (0.0, 0.0)

        # 边界框面积和 extent
        x, y, bw, bh = cv2.boundingRect(contour)
        bbox_area = bw * bh
        extent = float(area) / float(bbox_area) if bbox_area > 0 else 0.0

        # 实心度
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / float(hull_area) if hull_area > 0 else 0.0

        # 强度度量
        gray = self._ensure_gray(image)
        mean_intensity = 0.0
        max_intensity = 0.0
        if np.any(binary > 0):
            mean_intensity = float(np.mean(gray[binary > 0]))
            max_intensity = float(np.max(gray[binary > 0]))

        result.update({
            "area": int(area),
            "perimeter": float(perimeter),
            "circularity": min(float(circularity), 1.0),
            "eccentricity": float(eccentricity),
            "centroid": centroid,
            "bbox_area": int(bbox_area),
            "extent": float(extent),
            "mean_intensity": float(mean_intensity),
            "max_intensity": float(max_intensity),
            "solidity": float(solidity),
        })

        return result

    def get_health_report(self) -> Dict[str, Any]:
        """返回分割器状态报告。

        Returns
        -------
        dict
            状态报告字典，包含:
            - status: 状态字符串
            - segment_count: 累计分割次数
            - track_count: 累计追踪次数
            - refine_count: 累计精修次数
            - memory_usage: 记忆库使用量
            - memory_capacity: 记忆库容量
            - config: 当前配置参数
        """
        return {
            "status": "healthy",
            "segment_count": self._segment_count,
            "track_count": self._track_count,
            "refine_count": self._refine_count,
            "memory_usage": len(self._memory_bank),
            "memory_capacity": self.config.memory_length,
            "config": {
                "grabcut_iters": self.config.grabcut_iters,
                "edge_threshold_low": self.config.edge_threshold_low,
                "edge_threshold_high": self.config.edge_threshold_high,
                "edge_dilate_kernel": self.config.edge_dilate_kernel,
                "memory_length": self.config.memory_length,
                "min_area": self.config.min_area,
                "max_area": self.config.max_area,
                "refine_iterations": self.config.refine_iterations,
                "track_confidence_threshold": self.config.track_confidence_threshold,
                "morph_open_kernel": self.config.morph_open_kernel,
                "morph_close_kernel": self.config.morph_close_kernel,
            },
        }

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _validate_image(self, image: np.ndarray) -> None:
        """验证输入图像有效性。"""
        if image is None:
            raise ValueError("输入图像不能为 None")
        if not isinstance(image, np.ndarray):
            raise ValueError(f"输入图像必须为 np.ndarray，实际类型为 {type(image)}")
        if image.size == 0:
            raise ValueError("输入图像不能为空")
        if image.ndim not in (2, 3):
            raise ValueError(f"输入图像维度必须为 2 或 3，实际为 {image.ndim}")

    def _ensure_bgr(self, image: np.ndarray) -> np.ndarray:
        """确保图像为 BGR 三通道格式。"""
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if image.shape[2] == 3:
            return image.copy()
        # 单通道在第三维
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)

    def _ensure_gray(self, image: np.ndarray) -> np.ndarray:
        """确保图像为灰度单通道格式。"""
        if len(image.shape) == 2:
            return image.copy()
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        return image[:, :, 0].copy()

    def _clamp_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        shape: Tuple[int, ...],
    ) -> Tuple[int, int, int, int]:
        """将边界框限制在图像范围内。"""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        return (x1, y1, x2, y2)

    def _point_to_bbox(
        self,
        center: Optional[Tuple[int, int]],
        shape: Tuple[int, ...],
    ) -> Tuple[int, int, int, int]:
        """从中心点估计边界框。

        使用自适应窗口大小: 基于图像尺寸的固定比例。
        """
        if center is None:
            return (0, 0, shape[1], shape[0])

        cx, cy = int(center[0]), int(center[1])
        h, w = shape[:2]

        # 自适应半径: 图像短边的 10%
        radius = max(10, int(min(h, w) * 0.1))

        x1 = max(0, cx - radius)
        y1 = max(0, cy - radius)
        x2 = min(w, cx + radius)
        y2 = min(h, cy + radius)

        return (x1, y1, x2, y2)

    def _grabcut_segment(
        self,
        bgr: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """使用 GrabCut 算法进行前景分割。

        Parameters
        ----------
        bgr : np.ndarray
            BGR 三通道图像。
        bbox : Tuple[int, int, int, int]
            边界框 (x1, y1, x2, y2)。

        Returns
        -------
        np.ndarray
            二值掩码，前景为 255，背景为 0。
        """
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1

        if bw < 2 or bh < 2:
            LOGGER.debug("grabcut_segment: 边界框太小 (%d x %d)", bw, bh)
            return np.zeros(bgr.shape[:2], dtype=np.uint8)

        mask = np.zeros(bgr.shape[:2], dtype=np.uint8)

        # 初始化 GrabCut: 边界框内标记为可能前景 (GC_PR_FGD=3)
        # 边界框外标记为确定背景 (GC_BGD=0)
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)

        rect = (x1, y1, bw, bh)

        try:
            cv2.grabCut(
                bgr,
                mask,
                rect,
                bgd_model,
                fgd_model,
                self.config.grabcut_iters,
                cv2.GC_INIT_WITH_RECT,
            )
        except cv2.error as exc:
            LOGGER.warning("grabcut_segment: GrabCut 执行失败: %s", exc)
            return np.zeros(bgr.shape[:2], dtype=np.uint8)

        # 生成二值掩码: 前景 (GC_FGD=1) 和可能前景 (GC_PR_FGD=3) 设为 255
        binary_mask = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
            255,
            0,
        ).astype(np.uint8)

        return binary_mask

    def _compute_confidence(
        self,
        mask: np.ndarray,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> float:
        """计算分割置信度。

        基于以下因素综合评估:
        1. 掩码区域与边界框的重叠比例
        2. 掩码区域的对比度 (前景 vs 背景灰度差异)
        3. 掩码的紧凑度 (面积/边界框面积)

        Parameters
        ----------
        mask : np.ndarray
            分割掩码。
        image : np.ndarray
            原始图像。
        bbox : Tuple[int, int, int, int]
            提示边界框。

        Returns
        -------
        float
            置信度 [0, 1]。
        """
        if mask is None or mask.size == 0:
            return 0.0

        gray = self._ensure_gray(image)
        h, w = gray.shape
        x1, y1, x2, y2 = bbox

        binary = (mask > 0).astype(np.uint8)
        mask_area = int(np.sum(binary))

        if mask_area == 0:
            return 0.0

        # 因素 1: 掩码与边界框的重叠比例
        bbox_mask = np.zeros((h, w), dtype=np.uint8)
        bbox_mask[y1:y2, x1:x2] = 1
        overlap = int(np.sum(binary & bbox_mask))
        overlap_ratio = float(overlap) / float(mask_area) if mask_area > 0 else 0.0

        # 因素 2: 对比度 (前景均值 vs 边界框内背景均值)
        fg_mean = float(np.mean(gray[binary > 0])) if np.any(binary > 0) else 0.0
        bg_region = (bbox_mask > 0) & (binary == 0)
        bg_mean = float(np.mean(gray[bg_region])) if np.any(bg_region) else 0.0

        if bg_mean > 0:
            contrast = abs(fg_mean - bg_mean) / max(fg_mean, bg_mean)
        else:
            contrast = 1.0 if fg_mean > 0 else 0.0

        # 因素 3: 紧凑度
        bbox_area = (x2 - x1) * (y2 - y1)
        compactness = float(mask_area) / float(bbox_area) if bbox_area > 0 else 0.0
        # 理想紧凑度在 0.3~0.9 之间
        compactness_score = 1.0 - abs(compactness - 0.6) / 0.6
        compactness_score = max(0.0, min(1.0, compactness_score))

        # 加权融合
        confidence = (
            0.3 * overlap_ratio
            + 0.4 * min(contrast, 1.0)
            + 0.3 * compactness_score
        )

        return max(0.0, min(1.0, confidence))

    def _mask_to_bbox(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """从掩码计算边界框。

        Parameters
        ----------
        mask : np.ndarray
            二值掩码。

        Returns
        -------
        Tuple[int, int, int, int]
            边界框 (x1, y1, x2, y2)。掩码为空时返回 (0, 0, 0, 0)。
        """
        if mask is None or mask.size == 0:
            return (0, 0, 0, 0)

        binary = (mask > 0).astype(np.uint8)
        cols = np.any(binary, axis=0)
        rows = np.any(binary, axis=1)

        if not np.any(cols) or not np.any(rows):
            return (0, 0, 0, 0)

        x1 = int(np.argmax(cols))
        x2 = int(len(cols) - np.argmax(cols[::-1]))
        y1 = int(np.argmax(rows))
        y2 = int(len(rows) - np.argmax(rows[::-1]))

        return (x1, y1, x2, y2)
