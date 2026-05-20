"""
LodeSTAR 灵感无监督光斑检测器 (LodeSTARDetector)

灵感来源:
- DeepTrack 2.0 LodeSTAR 算法 — 基于梯度特征的无监督粒子追踪
- Phase Correlation — 傅里叶域亚像素配准 (Kuglin & Hines 1975)
- Multi-scale Template Matching — 金字塔多尺度模板匹配
- Adaptive Local Thresholding — 自适应局部统计阈值

算法原理:
- Reference Learning — 从参考图像中提取光斑特征模板 (无需标注)
- Multi-scale Template Matching — 高斯金字塔多尺度归一化互相关匹配
- Phase Correlation — 傅里叶变换相位相关实现亚像素位移估计
- Gradient-based Feature Analysis — 梯度幅值与方向特征 (模拟 LodeSTAR 梯度追踪)
- Adaptive Thresholding — 基于局部均值/标准差的自适应阈值分割
- Confidence Scoring — 综合多维度特征的置信度评分机制

功能:
- 完全无监督: 仅需一张参考图像即可学习光斑外观
- 无需任何标注数据或深度学习框架
- 多尺度搜索适应不同光斑尺寸
- 亚像素级定位精度 (相位相关 + 质心精化)
- 自适应阈值适应不同光照条件
- 置信度评分用于多光斑选择与异常过滤

依赖: numpy, opencv-python (cv2)
不依赖: PyTorch, TensorFlow 或任何深度学习框架
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger("SpotZoom.LodeSTARDetector")


# ======================================================================
# 数据类
# ======================================================================


@dataclass
class LodeSTARConfig:
    """LodeSTAR 检测器配置参数。

    Attributes
    ----------
    template_size : int
        参考模板尺寸 (像素)。从参考图像中裁剪的正方形区域边长。
    pyramid_levels : int
        高斯金字塔层数。用于多尺度搜索。
    scale_range : Tuple[float, float]
        搜索缩放范围 (最小倍率, 最大倍率)。
    scale_steps : int
        缩放搜索步数。
    match_threshold : float
        归一化互相关匹配阈值 [0, 1]。低于此值的候选被过滤。
    nms_overlap : float
        非极大值抑制 (NMS) 的 IoU 重叠阈值。
    adaptive_block_size : int
        自适应阈值局部块大小 (奇数)。
    adaptive_c : int
        自适应阈值常数偏移。
    gradient_weight : float
        梯度特征在置信度评分中的权重。
    phase_corr_weight : float
        相位相关特征在置信度评分中的权重。
    template_weight : float
        模板匹配特征在置信度评分中的权重。
    min_spot_area : int
        最小光斑面积 (像素^2)。
    max_spots : int
        最大返回光斑数。0 表示不限制。
    subpixel_refine : bool
        是否进行亚像素精化。
    bg_percentile : float
        背景估计百分位数 (0-100)。
    """

    template_size: int = 64
    pyramid_levels: int = 4
    scale_range: Tuple[float, float] = (0.5, 2.0)
    scale_steps: int = 8
    match_threshold: float = 0.5
    nms_overlap: float = 0.3
    adaptive_block_size: int = 31
    adaptive_c: int = 5
    gradient_weight: float = 0.35
    phase_corr_weight: float = 0.35
    template_weight: float = 0.30
    min_spot_area: int = 16
    max_spots: int = 0
    subpixel_refine: bool = True
    bg_percentile: float = 15.0


@dataclass
class LodeSTARDetection:
    """单个光斑检测结果。

    Attributes
    ----------
    center : Tuple[float, float]
        光斑中心坐标 (x, y)，亚像素精度。
    bbox : Tuple[int, int, int, int]
        边界框 (x1, y1, x2, y2)，整数像素。
    confidence : float
        综合置信度分数 [0, 1]。
    ncc_score : float
        归一化互相关匹配分数。
    phase_score : float
        相位相关匹配分数。
    gradient_score : float
        梯度特征匹配分数。
    scale : float
        检测到的最佳缩放因子。
    snr : float
        信噪比估计。
    """

    center: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    confidence: float
    ncc_score: float = 0.0
    phase_score: float = 0.0
    gradient_score: float = 0.0
    scale: float = 1.0
    snr: float = 0.0


@dataclass
class LodeSTARReport:
    """LodeSTAR 检测分析报告。

    Attributes
    ----------
    num_detections : int
        检测到的光斑数量。
    mean_confidence : float
        平均置信度。
    max_confidence : float
        最大置信度。
    processing_time_ms : float
        处理耗时 (毫秒)。
    template_learned : bool
        是否已成功学习参考模板。
    template_size : Tuple[int, int]
        模板尺寸 (高, 宽)。
    scales_searched : int
        搜索的缩放层数。
    candidates_before_nms : int
        NMS 前的候选数量。
    detections : List[LodeSTARDetection]
        所有检测结果列表。
    gradient_magnitude_mean : float
        参考模板的平均梯度幅值。
    gradient_direction_std : float
        参考模板的梯度方向标准差。
    """

    num_detections: int = 0
    mean_confidence: float = 0.0
    max_confidence: float = 0.0
    processing_time_ms: float = 0.0
    template_learned: bool = False
    template_size: Tuple[int, int] = (0, 0)
    scales_searched: int = 0
    candidates_before_nms: int = 0
    detections: List[LodeSTARDetection] = field(default_factory=list)
    gradient_magnitude_mean: float = 0.0
    gradient_direction_std: float = 0.0


# ======================================================================
# 核心检测器
# ======================================================================


class LodeSTARDetector:
    """LodeSTAR 灵感无监督光斑检测器。

    受 DeepTrack 2.0 LodeSTAR 算法启发，使用经典计算机视觉方法
    实现无监督光斑检测。仅需一张参考图像即可学习光斑外观特征，
    后续在新图像中通过多尺度模板匹配、相位相关和梯度特征分析
    实现鲁棒的光斑定位。

    工作流程:
    1. learn_reference(image) — 从参考图像中自动提取光斑特征模板
    2. detect(image) — 在新图像中检测光斑
    3. analyze(image) — 检测并返回详细分析报告

    Parameters
    ----------
    config : LodeSTARConfig or None
        配置参数。为 None 时使用默认配置。
    """

    def __init__(self, config: Optional[LodeSTARConfig] = None):
        self.config = config if config is not None else LodeSTARConfig()

        # 参考特征 (由 learn_reference 填充)
        self._template: Optional[np.ndarray] = None
        self._template_gradient_mag: Optional[np.ndarray] = None
        self._template_gradient_dir: Optional[np.ndarray] = None
        self._template_padded: Optional[np.ndarray] = None  # 用于相位相关
        self._reference_learned: bool = False
        self._template_center: Optional[Tuple[float, float]] = None

        # 内部状态
        self._frame_count: int = 0
        self._last_report: Optional[LodeSTARReport] = None

        LOGGER.debug("LodeSTARDetector 初始化完成, config=%s", self.config)

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def learn_reference(self, image: np.ndarray) -> bool:
        """从参考图像中学习光斑外观特征 (无监督)。

        自动定位参考图像中最显著的光斑，提取其特征模板，
        包括灰度模板、梯度幅值图和梯度方向图。

        Parameters
        ----------
        image : np.ndarray
            参考图像 (BGR 或灰度格式)。

        Returns
        -------
        bool
            是否成功学习参考特征。
        """
        if image is None or image.size == 0:
            LOGGER.warning("learn_reference: 输入图像为空")
            return False

        gray = self._to_gray(image)
        h, w = gray.shape

        LOGGER.debug(
            "learn_reference: 图像尺寸 %dx%d, 模板尺寸 %d",
            w, h, self.config.template_size,
        )

        # 1. 自动定位参考图像中最亮/最显著的光斑
        spot_center = self._find_reference_spot(gray)
        if spot_center is None:
            LOGGER.warning("learn_reference: 未能定位参考光斑")
            return False

        cx, cy = spot_center
        LOGGER.debug("learn_reference: 参考光斑位于 (%.1f, %.1f)", cx, cy)

        # 2. 裁剪模板区域
        ts = self.config.template_size
        half = ts // 2
        x1 = max(0, int(cx) - half)
        y1 = max(0, int(cy) - half)
        x2 = min(w, int(cx) + half)
        y2 = min(h, int(cy) + half)

        # 边界保护: 如果裁剪区域不够大，调整起点
        if x2 - x1 < ts // 2 or y2 - y1 < ts // 2:
            LOGGER.warning("learn_reference: 光斑过于靠近图像边缘")
            return False

        template_crop = gray[y1:y2, x1:x2].astype(np.float64)

        # 3. 背景归一化
        bg = np.percentile(template_crop, self.config.bg_percentile)
        template_norm = template_crop - bg
        template_norm = np.maximum(template_norm, 0.0)

        # 归一化到 [0, 1]
        t_max = template_norm.max()
        if t_max > 1e-6:
            template_norm /= t_max

        self._template = template_norm
        self._template_center = (float(cx - x1), float(cy - y1))

        # 4. 计算梯度特征 (模拟 LodeSTAR 的梯度追踪)
        self._compute_template_gradients()

        # 5. 准备相位相关用的填充模板
        self._prepare_phase_template()

        self._reference_learned = True
        LOGGER.info(
            "learn_reference: 成功学习参考模板, 尺寸=%s, "
            "梯度幅值均值=%.3f, 梯度方向标准差=%.3f",
            self._template.shape,
            self._template_gradient_mag.mean() if self._template_gradient_mag is not None else 0.0,
            self._template_gradient_dir.std() if self._template_gradient_dir is not None else 0.0,
        )
        return True

    def detect(self, image: np.ndarray) -> Optional[LodeSTARDetection]:
        """在输入图像中检测光斑。

        使用学习到的参考特征，通过多尺度模板匹配、
        相位相关和梯度特征分析检测最显著的光斑。

        Parameters
        ----------
        image : np.ndarray
            输入图像 (BGR 或灰度格式)。

        Returns
        -------
        Optional[LodeSTARDetection]
            检测结果。未检测到光斑时返回 None。
        """
        if not self._reference_learned:
            LOGGER.warning("detect: 尚未学习参考特征，请先调用 learn_reference()")
            return None

        if image is None or image.size == 0:
            return None

        detections = self._detect_all(image)
        if not detections:
            return None

        # 返回置信度最高的检测结果
        best = max(detections, key=lambda d: d.confidence)
        self._frame_count += 1
        LOGGER.debug(
            "detect frame #%d: spot at (%.2f, %.2f), conf=%.3f, ncc=%.3f, "
            "phase=%.3f, grad=%.3f, scale=%.2f",
            self._frame_count, best.center[0], best.center[1],
            best.confidence, best.ncc_score, best.phase_score,
            best.gradient_score, best.scale,
        )
        return best

    def analyze(self, image: np.ndarray) -> LodeSTARReport:
        """检测光斑并返回详细分析报告。

        Parameters
        ----------
        image : np.ndarray
            输入图像 (BGR 或灰度格式)。

        Returns
        -------
        LodeSTARReport
            详细分析报告。
        """
        t0 = time.perf_counter()

        detections: List[LodeSTARDetection] = []
        candidates_before_nms = 0

        if self._reference_learned and image is not None and image.size > 0:
            detections = self._detect_all(image)
            candidates_before_nms = self._last_candidates_count

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        confidences = [d.confidence for d in detections] if detections else [0.0]

        report = LodeSTARReport(
            num_detections=len(detections),
            mean_confidence=float(np.mean(confidences)),
            max_confidence=float(np.max(confidences)),
            processing_time_ms=elapsed_ms,
            template_learned=self._reference_learned,
            template_size=(
                self._template.shape[:2]
                if self._template is not None else (0, 0)
            ),
            scales_searched=self.config.scale_steps,
            candidates_before_nms=candidates_before_nms,
            detections=detections,
            gradient_magnitude_mean=(
                float(self._template_gradient_mag.mean())
                if self._template_gradient_mag is not None else 0.0
            ),
            gradient_direction_std=(
                float(self._template_gradient_dir.std())
                if self._template_gradient_dir is not None else 0.0
            ),
        )

        self._last_report = report
        LOGGER.debug(
            "analyze: %d detections, mean_conf=%.3f, time=%.1fms",
            report.num_detections, report.mean_confidence, report.processing_time_ms,
        )
        return report

    def reset(self) -> None:
        """重置检测器到初始状态。

        清除学习到的参考特征和所有内部状态。
        """
        self._template = None
        self._template_gradient_mag = None
        self._template_gradient_dir = None
        self._template_padded = None
        self._reference_learned = False
        self._template_center = None
        self._frame_count = 0
        self._last_report = None
        self._last_candidates_count = 0
        LOGGER.debug("LodeSTARDetector 已重置")

    # ------------------------------------------------------------------
    # 参考学习 (内部方法)
    # ------------------------------------------------------------------

    def _find_reference_spot(
        self, gray: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """在参考图像中自动定位最显著的光斑。

        使用顶帽变换增强亮目标，然后通过自适应阈值和
        加水质心定位光斑中心。

        Parameters
        ----------
        gray : np.ndarray
            灰度图像 (uint8)。

        Returns
        -------
        Optional[Tuple[float, float]]
            光斑中心 (x, y)。未找到时返回 None。
        """
        h, w = gray.shape

        # 顶帽变换增强小亮目标
        kernel_size = max(self.config.template_size // 2, 15)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

        # 自适应阈值
        block = self.config.adaptive_block_size
        binary = cv2.adaptiveThreshold(
            tophat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, blockSize=block, C=self.config.adaptive_c,
        )

        # 形态学清理
        morph_k = 3
        mk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, mk)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, mk)

        # 查找轮廓
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            # 回退: 使用最亮点
            _, max_val, _, max_loc = cv2.minMaxLoc(gray)
            if max_val < 10:
                return None
            return (float(max_loc[0]), float(max_loc[1]))

        # 选择面积最大的候选 (假设参考图像中光斑是主要亮目标)
        best_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(best_contour)

        if area < self.config.min_spot_area:
            # 回退到最亮点
            _, max_val, _, max_loc = cv2.minMaxLoc(gray)
            if max_val < 10:
                return None
            return (float(max_loc[0]), float(max_loc[1]))

        # 灰度加权质心
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [best_contour], -1, 255, -1)

        gray_f = gray.astype(np.float64)
        masked = gray_f * (mask > 0).astype(np.float64)
        total = masked.sum()
        if total < 1e-6:
            M = cv2.moments(best_contour)
            if M["m00"] < 1e-6:
                return None
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            yy, xx = np.indices(gray.shape)
            cx = float(np.sum(xx * masked) / total)
            cy = float(np.sum(yy * masked) / total)

        return (cx, cy)

    def _compute_template_gradients(self) -> None:
        """计算参考模板的梯度特征。

        计算 Sobel 梯度幅值和方向，模拟 LodeSTAR 算法中
        基于梯度特征的无监督追踪机制。
        """
        if self._template is None:
            return

        tmpl = self._template.astype(np.float32)

        # Sobel 梯度
        grad_x = cv2.Sobel(tmpl, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(tmpl, cv2.CV_32F, 0, 1, ksize=3)

        # 梯度幅值
        mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        mag_max = mag.max()
        if mag_max > 1e-6:
            mag /= mag_max
        self._template_gradient_mag = mag

        # 梯度方向 (弧度)
        direction = np.arctan2(grad_y, grad_x)
        self._template_gradient_dir = direction

    def _prepare_phase_template(self) -> None:
        """准备用于相位相关的填充模板。

        将模板填充到目标图像尺寸，以便进行傅里叶域相位相关。
        实际使用时会在 detect 中根据目标图像尺寸重新填充。
        """
        # 此处仅标记已准备，实际填充在 _phase_correlate 中完成
        self._template_padded = self._template

    # ------------------------------------------------------------------
    # 检测核心 (内部方法)
    # ------------------------------------------------------------------

    def _detect_all(self, image: np.ndarray) -> List[LodeSTARDetection]:
        """在图像中检测所有光斑。

        Parameters
        ----------
        image : np.ndarray
            输入图像。

        Returns
        -------
        List[LodeSTARDetection]
            检测结果列表 (按置信度降序排列)。
        """
        gray = self._to_gray(image)
        h, w = gray.shape

        # 背景归一化
        gray_f = self._normalize_image(gray)

        # 多尺度模板匹配
        candidates = self._multi_scale_match(gray_f)

        self._last_candidates_count = len(candidates)

        if not candidates:
            return []

        # 非极大值抑制
        detections = self._nms(candidates)

        # 限制最大数量
        if self.config.max_spots > 0:
            detections = detections[: self.config.max_spots]

        # 按置信度降序排列
        detections.sort(key=lambda d: d.confidence, reverse=True)

        return detections

    def _multi_scale_match(
        self, gray: np.ndarray
    ) -> List[LodeSTARDetection]:
        """多尺度模板匹配。

        在不同缩放因子下进行归一化互相关匹配，
        收集所有超过阈值的候选位置。

        Parameters
        ----------
        gray : np.ndarray
            归一化后的灰度图像 (float64, [0, 1])。

        Returns
        -------
        List[LodeSTARDetection]
            候选检测结果列表。
        """
        if self._template is None:
            return []

        candidates: List[LodeSTARDetection] = []
        th, tw = self._template.shape[:2]

        # 生成缩放因子序列
        scales = np.linspace(
            self.config.scale_range[0],
            self.config.scale_range[1],
            self.config.scale_steps,
        )

        for scale in scales:
            # 缩放模板
            new_w = max(3, int(tw * scale))
            new_h = max(3, int(th * scale))

            scaled_tmpl = cv2.resize(
                self._template, (new_w, new_h),
                interpolation=cv2.INTER_LINEAR,
            )

            # 归一化缩放后的模板
            tmpl_max = scaled_tmpl.max()
            if tmpl_max > 1e-6:
                scaled_tmpl_norm = scaled_tmpl / tmpl_max
            else:
                continue

            # 归一化互相关匹配
            result = cv2.matchTemplate(
                gray.astype(np.float32),
                scaled_tmpl_norm.astype(np.float32),
                cv2.TM_CCOEFF_NORMED,
            )

            # 找到超过阈值的候选位置
            locs = np.where(result >= self.config.match_threshold)

            for pt_y, pt_x in zip(locs[0], locs[1]):
                ncc_val = float(result[pt_y, pt_x])

                # 候选中心
                cx = pt_x + new_w / 2.0
                cy = pt_y + new_h / 2.0

                # 边界框
                bbox = (
                    int(pt_x),
                    int(pt_y),
                    int(pt_x + new_w),
                    int(pt_y + new_h),
                )

                # 计算梯度特征匹配分数
                grad_score = self._compute_gradient_score(gray, pt_x, pt_y, new_w, new_h, scale)

                # 计算相位相关分数
                phase_score = self._compute_phase_score(
                    gray, pt_x, pt_y, new_w, new_h
                )

                # 计算信噪比
                snr = self._compute_snr(gray, pt_x, pt_y, new_w, new_h)

                # 综合置信度
                confidence = (
                    self.config.template_weight * ncc_val
                    + self.config.gradient_weight * grad_score
                    + self.config.phase_corr_weight * phase_score
                )

                # 亚像素精化
                if self.config.subpixel_refine:
                    cx, cy = self._subpixel_refine(gray, pt_x, pt_y, new_w, new_h)

                det = LodeSTARDetection(
                    center=(cx, cy),
                    bbox=bbox,
                    confidence=float(confidence),
                    ncc_score=ncc_val,
                    phase_score=float(phase_score),
                    gradient_score=float(grad_score),
                    scale=float(scale),
                    snr=float(snr),
                )
                candidates.append(det)

        return candidates

    def _compute_gradient_score(
        self,
        gray: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        scale: float,
    ) -> float:
        """计算梯度特征匹配分数。

        比较目标区域与参考模板的梯度幅值分布和方向一致性，
        模拟 LodeSTAR 算法的梯度追踪机制。

        Parameters
        ----------
        gray : np.ndarray
            归一化灰度图像。
        x, y : int
            区域左上角坐标。
        w, h : int
            区域宽高。
        scale : float
            当前缩放因子。

        Returns
        -------
        float
            梯度匹配分数 [0, 1]。
        """
        if self._template_gradient_mag is None or self._template_gradient_dir is None:
            return 0.0

        img_h, img_w = gray.shape

        # 边界检查
        if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
            return 0.0

        crop = gray[y: y + h, x: x + w].astype(np.float32)

        # 计算目标区域梯度
        grad_x = cv2.Sobel(crop, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=3)

        target_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        target_dir = np.arctan2(grad_y, grad_x)

        # 归一化梯度幅值
        t_mag_max = target_mag.max()
        if t_mag_max > 1e-6:
            target_mag_norm = target_mag / t_mag_max
        else:
            return 0.0

        # 缩放参考模板梯度到目标尺寸
        ref_mag_scaled = cv2.resize(
            self._template_gradient_mag.astype(np.float32),
            (w, h),
            interpolation=cv2.INTER_LINEAR,
        )

        # 梯度幅值相似度 (归一化点积)
        ref_flat = ref_mag_scaled.flatten()
        tgt_flat = target_mag_norm.flatten()
        ref_norm = np.linalg.norm(ref_flat)
        tgt_norm = np.linalg.norm(tgt_flat)
        if ref_norm < 1e-6 or tgt_norm < 1e-6:
            mag_sim = 0.0
        else:
            mag_sim = float(np.dot(ref_flat, tgt_flat) / (ref_norm * tgt_norm))

        # 梯度方向一致性 (仅在梯度幅值较大的区域)
        ref_dir_scaled = cv2.resize(
            self._template_gradient_dir.astype(np.float32),
            (w, h),
            interpolation=cv2.INTER_LINEAR,
        )

        # 使用梯度幅值作为权重
        weight = (ref_mag_scaled + target_mag_norm) / 2.0
        weight_sum = weight.sum()
        if weight_sum < 1e-6:
            dir_consistency = 0.0
        else:
            # 方向差 (处理周期性)
            diff = np.abs(ref_dir_scaled - target_dir)
            diff = np.minimum(diff, 2 * np.pi - diff)  # 周期性修正
            dir_consistency = float(
                1.0 - np.average(diff, weights=weight) / np.pi
            )

        # 综合梯度分数
        gradient_score = 0.5 * mag_sim + 0.5 * dir_consistency
        return float(np.clip(gradient_score, 0.0, 1.0))

    def _compute_phase_score(
        self,
        gray: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> float:
        """计算相位相关匹配分数。

        使用傅里叶变换相位相关估计目标区域与参考模板之间的
        亚像素位移，相位相关峰值越高表示匹配越好。

        Parameters
        ----------
        gray : np.ndarray
            归一化灰度图像。
        x, y : int
            区域左上角坐标。
        w, h : int
            区域宽高。

        Returns
        -------
        float
            相位相关分数 [0, 1]。
        """
        if self._template is None:
            return 0.0

        img_h, img_w = gray.shape

        if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
            return 0.0

        if w < 4 or h < 4:
            return 0.0

        crop = gray[y: y + h, x: x + w].astype(np.float32)

        # 缩放模板到目标尺寸
        tmpl_resized = cv2.resize(
            self._template.astype(np.float32), (w, h),
            interpolation=cv2.INTER_LINEAR,
        )

        # 归一化
        crop_norm = crop - crop.mean()
        tmpl_norm = tmpl_resized - tmpl_resized.mean()

        crop_std = crop_norm.std()
        tmpl_std = tmpl_norm.std()
        if crop_std < 1e-6 or tmpl_std < 1e-6:
            return 0.0

        # 傅里叶变换
        fft_crop = np.fft.fft2(crop_norm)
        fft_tmpl = np.fft.fft2(tmpl_norm)

        # 互功率谱
        cross_power = fft_crop * np.conj(fft_tmpl)
        magnitude = np.abs(cross_power)
        magnitude = np.maximum(magnitude, 1e-10)  # 避免除零
        cross_spectrum = cross_power / magnitude

        # 逆傅里叶变换得到相位相关面
        phase_corr = np.real(np.fft.ifft2(cross_spectrum))

        # 相位相关峰值
        peak_val = float(phase_corr.max())
        # 理想匹配时峰值为 1.0 (归一化后)
        phase_corr_size = phase_corr.size
        if phase_corr_size > 0:
            normalized_peak = peak_val / phase_corr_size
        else:
            normalized_peak = 0.0

        return float(np.clip(normalized_peak, 0.0, 1.0))

    def _compute_snr(
        self,
        gray: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> float:
        """计算候选区域的信噪比。

        Parameters
        ----------
        gray : np.ndarray
            归一化灰度图像。
        x, y : int
            区域左上角坐标。
        w, h : int
            区域宽高。

        Returns
        -------
        float
            信噪比估计值。
        """
        img_h, img_w = gray.shape

        if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
            return 0.0

        crop = gray[y: y + h, x: x + w]
        peak = float(crop.max())

        # 使用区域边缘像素估计背景噪声
        border_pixels = []
        if h > 2 and w > 2:
            border_pixels.extend(crop[0, :].tolist())       # 上边
            border_pixels.extend(crop[-1, :].tolist())      # 下边
            border_pixels.extend(crop[:, 0].tolist())       # 左边
            border_pixels.extend(crop[:, -1].tolist())      # 右边

        if border_pixels:
            noise_std = float(np.std(border_pixels))
        else:
            noise_std = 1e-6

        return peak / max(noise_std, 1e-6)

    def _subpixel_refine(
        self,
        gray: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> Tuple[float, float]:
        """亚像素级中心精化。

        使用灰度加权质心法在候选区域内进行亚像素精化，
        并结合相位相关进行亚像素位移修正。

        Parameters
        ----------
        gray : np.ndarray
            归一化灰度图像。
        x, y : int
            区域左上角坐标。
        w, h : int
            区域宽高。

        Returns
        -------
        Tuple[float, float]
            精化后的中心坐标 (cx, cy)。
        """
        img_h, img_w = gray.shape

        # 扩展搜索区域以提高质心精度
        margin = max(2, w // 4)
        rx1 = max(0, x - margin)
        ry1 = max(0, y - margin)
        rx2 = min(img_w, x + w + margin)
        ry2 = min(img_h, y + h + margin)

        region = gray[ry1:ry2, rx1:rx2].astype(np.float64)

        # 背景估计
        bg = np.percentile(region, self.config.bg_percentile)
        signal = region - bg
        signal = np.maximum(signal, 0.0)

        total = signal.sum()
        if total < 1e-6:
            return (x + w / 2.0, y + h / 2.0)

        # 加权质心
        yy, xx = np.indices(signal.shape)
        cx_local = float(np.sum(xx * signal) / total)
        cy_local = float(np.sum(yy * signal) / total)

        # 转换回图像坐标
        cx = rx1 + cx_local
        cy = ry1 + cy_local

        # 相位相关亚像素修正 (如果模板可用)
        if self._template is not None and w >= 8 and h >= 8:
            try:
                dx, dy = self._phase_subpixel(gray, x, y, w, h)
                cx += dx
                cy += dy
            except Exception:
                LOGGER.debug("subpixel_refine: 相位相关修正失败，使用质心结果")

        return (cx, cy)

    def _phase_subpixel(
        self,
        gray: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> Tuple[float, float]:
        """使用相位相关进行亚像素位移估计。

        Parameters
        ----------
        gray : np.ndarray
            归一化灰度图像。
        x, y : int
            区域左上角坐标。
        w, h : int
            区域宽高。

        Returns
        -------
        Tuple[float, float]
            亚像素位移修正量 (dx, dy)。
        """
        if self._template is None:
            return (0.0, 0.0)

        img_h, img_w = gray.shape

        if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
            return (0.0, 0.0)

        crop = gray[y: y + h, x: x + w].astype(np.float32)
        tmpl_resized = cv2.resize(
            self._template.astype(np.float32), (w, h),
            interpolation=cv2.INTER_LINEAR,
        )

        # 归一化
        crop_norm = crop - crop.mean()
        tmpl_norm = tmpl_resized - tmpl_resized.mean()

        # 傅里叶变换
        fft_crop = np.fft.fft2(crop_norm)
        fft_tmpl = np.fft.fft2(tmpl_norm)

        # 互功率谱
        cross_power = fft_crop * np.conj(fft_tmpl)
        magnitude = np.abs(cross_power)
        magnitude = np.maximum(magnitude, 1e-10)
        cross_spectrum = cross_power / magnitude

        # 逆傅里叶变换
        phase_corr = np.real(np.fft.ifft2(cross_spectrum))

        # 在 3x3 邻域内拟合抛物面进行亚像素定位
        peak_idx = np.unravel_index(np.argmax(phase_corr), phase_corr.shape)
        py_peak, px_peak = peak_idx

        # 边界检查
        if (
            py_peak < 1 or py_peak >= phase_corr.shape[0] - 1
            or px_peak < 1 or px_peak >= phase_corr.shape[1] - 1
        ):
            return (0.0, 0.0)

        # 二维抛物面拟合
        p00 = phase_corr[py_peak, px_peak]
        p10 = phase_corr[py_peak, px_peak + 1]
        pm10 = phase_corr[py_peak, px_peak - 1]
        p01 = phase_corr[py_peak + 1, px_peak]
        pm01 = phase_corr[py_peak - 1, px_peak]

        dx = 0.5 * (p10 - pm10) / (2 * p00 - p10 - pm10 + 1e-10)
        dy = 0.5 * (p01 - pm01) / (2 * p00 - p01 - pm01 + 1e-10)

        # 限制修正范围 (避免异常值)
        dx = float(np.clip(dx, -1.0, 1.0))
        dy = float(np.clip(dy, -1.0, 1.0))

        return (dx, dy)

    # ------------------------------------------------------------------
    # 非极大值抑制
    # ------------------------------------------------------------------

    def _nms(
        self, candidates: List[LodeSTARDetection]
    ) -> List[LodeSTARDetection]:
        """非极大值抑制 (NMS)。

        按置信度降序排列，移除与高置信度检测框重叠过高的候选。

        Parameters
        ----------
        candidates : List[LodeSTARDetection]
            候选检测结果列表。

        Returns
        -------
        List[LodeSTARDetection]
            NMS 后的检测结果列表。
        """
        if not candidates:
            return []

        # 按置信度降序排列
        sorted_cands = sorted(candidates, key=lambda d: d.confidence, reverse=True)

        keep: List[LodeSTARDetection] = []
        suppressed = set()

        for i, det_i in enumerate(sorted_cands):
            if i in suppressed:
                continue

            keep.append(det_i)

            for j in range(i + 1, len(sorted_cands)):
                if j in suppressed:
                    continue

                det_j = sorted_cands[j]
                iou = self._compute_iou(det_i.bbox, det_j.bbox)

                if iou >= self.config.nms_overlap:
                    suppressed.add(j)

        return keep

    @staticmethod
    def _compute_iou(
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int],
    ) -> float:
        """计算两个边界框的 IoU (交并比)。

        Parameters
        ----------
        box1, box2 : Tuple[int, int, int, int]
            (x1, y1, x2, y2) 边界框。

        Returns
        -------
        float
            IoU 值 [0, 1]。
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter <= 0:
            return 0.0

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / max(union, 1e-6)

    # ------------------------------------------------------------------
    # 图像预处理工具
    # ------------------------------------------------------------------

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        """将输入图像转换为灰度格式 (uint8)。

        Parameters
        ----------
        image : np.ndarray
            BGR 或灰度图像。

        Returns
        -------
        np.ndarray
            灰度图像 (uint8)。
        """
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _normalize_image(self, gray: np.ndarray) -> np.ndarray:
        """归一化灰度图像。

        执行背景扣除和归一化，使图像适合模板匹配。

        Parameters
        ----------
        gray : np.ndarray
            灰度图像 (uint8)。

        Returns
        -------
        np.ndarray
            归一化后的灰度图像 (float64, [0, 1])。
        """
        gray_f = gray.astype(np.float64)

        # 背景估计与扣除
        bg = np.percentile(gray_f, self.config.bg_percentile)
        signal = gray_f - bg
        signal = np.maximum(signal, 0.0)

        # 归一化到 [0, 1]
        s_max = signal.max()
        if s_max > 1e-6:
            signal /= s_max

        return signal

    # ------------------------------------------------------------------
    # 属性访问
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """检测器是否已就绪 (已学习参考特征)。"""
        return self._reference_learned

    @property
    def frame_count(self) -> int:
        """已处理的帧数。"""
        return self._frame_count

    @property
    def last_report(self) -> Optional[LodeSTARReport]:
        """最近一次分析报告。"""
        return self._last_report

    @property
    def template_shape(self) -> Optional[Tuple[int, int]]:
        """参考模板尺寸 (高, 宽)。未学习时返回 None。"""
        if self._template is not None:
            return self._template.shape[:2]
        return None


# ======================================================================
# 便捷函数
# ======================================================================


def detect_spots(
    image: np.ndarray,
    reference: np.ndarray,
    config: Optional[LodeSTARConfig] = None,
) -> List[LodeSTARDetection]:
    """便捷函数: 使用参考图像检测光斑。

    Parameters
    ----------
    image : np.ndarray
        待检测图像 (BGR 或灰度)。
    reference : np.ndarray
        参考图像 (BGR 或灰度)，用于学习光斑外观。
    config : LodeSTARConfig or None
        配置参数。

    Returns
    -------
    List[LodeSTARDetection]
        检测结果列表。
    """
    detector = LodeSTARDetector(config=config)
    if not detector.learn_reference(reference):
        return []
    return detector._detect_all(image)
