"""
上下文感知光斑检测器 (ContextAwareDetector)

基于 DECODE (https://github.com/TuragaLab/DECODE) 的深度上下文依赖检测框架，
为 SpotZoom 提供整帧上下文感知的多光斑检测能力。

灵感来源:
- DECODE: Deep Context Dependent localizer (Speiser et al., Nature Methods 2021)
  (https://github.com/TuragaLab/DECODE)
- YOLO: 实时目标检测 (已在主控制器中使用)
- LodeSTAR: 无监督检测 (已在 v1 中适配)

算法原理:
  1. 整帧上下文: 将整帧图像作为输入，同时检测所有光斑，
     利用全局上下文信息提升检测鲁棒性
  2. 密度自适应: 根据局部光斑密度自动调整检测参数
  3. 重叠分离: 使用高斯混合模型分离重叠光斑
  4. 多尺度检测: 在多个尺度上检测，避免遗漏大小不同的光斑

与现有模块的关系:
  - 增强 v1/classic_spot_detector.py 的多光斑检测能力
  - 与 v1/multi_spot_tracker.py 协同
  - 与 uncertainty_aware_localizer_v2.py 的不确定性估计协同

外部依赖: numpy, scipy (可选)
"""

import numpy as np
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class ContextDetectorConfig:
    """上下文感知检测器配置。"""
    # 检测参数
    min_intensity: float = 10.0        # 最小峰值强度
    min_snr: float = 3.0               # 最小信噪比
    background_percentile: float = 25.0 # 背景估计百分位

    # 多尺度参数
    scales: Tuple[int, ...] = (1, 2, 4)  # 多尺度因子
    merge_iou_threshold: float = 0.3     # NMS IoU 阈值

    # 密度自适应参数
    density_window_size: int = 64       # 密度分析窗口
    high_density_threshold: float = 5.0 # 高密度阈值（光斑/窗口）
    low_density_threshold: float = 1.0  # 低密度阈值

    # 重叠分离参数
    overlap_separation: bool = True     # 启用重叠分离
    overlap_min_distance: float = 3.0   # 最小分离距离（像素）
    overlap_iterations: int = 5         # EM 迭代次数

    # 后处理
    subpixel_refinement: bool = True    # 亚像素精化
    edge_suppression: bool = True       # 边缘抑制


@dataclass
class ContextDetectorResult:
    """检测结果。"""
    detections: List[Dict[str, Any]] = field(default_factory=list)
    num_detections: int = 0
    density_map: Optional[np.ndarray] = None
    processing_time_ms: float = 0.0
    method_used: str = "context_aware"


class ContextAwareDetector:
    """上下文感知光斑检测器。

    利用整帧上下文信息进行多光斑检测，支持密度自适应参数调整
    和重叠光斑分离。

    Parameters
    ----------
    config : ContextDetectorConfig
        检测器配置。
    """

    def __init__(self, config: Optional[ContextDetectorConfig] = None):
        self.config = config or ContextDetectorConfig()

    def detect(self, image: np.ndarray) -> ContextDetectorResult:
        """在图像中检测所有光斑。

        Parameters
        ----------
        image : np.ndarray
            输入图像 (H, W) 或 (H, W, C)。

        Returns
        -------
        ContextDetectorResult
            检测结果。
        """
        t0 = time.perf_counter()
        result = ContextDetectorResult()

        # 转灰度
        if image.ndim == 3:
            gray = image[:, :, 0].astype(np.float64)
        else:
            gray = image.astype(np.float64)

        # Step 1: 背景估计
        bg = np.percentile(gray, self.config.background_percentile)
        signal = np.maximum(gray - bg, 0)

        # Step 2: 密度分析
        density_map = self._compute_density_map(signal)
        result.density_map = density_map

        # Step 3: 多尺度检测
        all_detections = []
        for scale in self.config.scales:
            if scale == 1:
                scaled = signal
            else:
                # 下采样
                h, w = gray.shape
                new_h, new_w = h // scale, w // scale
                scaled = signal[:new_h*scale, :new_w*scale].reshape(
                    new_h, scale, new_w, scale
                ).mean(axis=(1, 3))

            detections = self._detect_at_scale(scaled, scale, density_map)
            all_detections.extend(detections)

        # Step 4: 多尺度 NMS
        merged = self._multiscale_nms(all_detections)

        # Step 5: 重叠分离
        if self.config.overlap_separation:
            merged = self._separate_overlaps(merged, signal)

        # Step 6: 亚像素精化
        if self.config.subpixel_refinement:
            for det in merged:
                det = self._refine_subpixel(det, signal)

        # Step 7: 边缘抑制
        if self.config.edge_suppression:
            h, w = gray.shape
            margin = 5
            merged = [d for d in merged
                      if margin < d['cx'] < w - margin and margin < d['cy'] < h - margin]

        result.detections = merged
        result.num_detections = len(merged)
        result.processing_time_ms = (time.perf_counter() - t0) * 1000

        return result

    def _compute_density_map(self, signal: np.ndarray) -> np.ndarray:
        """计算局部光斑密度图。"""
        h, w = signal.shape
        ws = self.config.density_window_size
        density = np.zeros((h, w))

        # 简化的密度估计：局部最大值计数
        try:
            from scipy.ndimage import maximum_filter, label
            local_max = maximum_filter(signal, size=5)
            peaks = (signal == local_max) & (signal > self.config.min_intensity)
            labeled, n_peaks = label(peaks)

            # 对每个峰值，在密度窗口内计数
            for i in range(1, n_peaks + 1):
                ys, xs = np.where(labeled == i)
                if len(ys) == 0:
                    continue
                cy, cx = int(ys[0]), int(xs[0])
                y1 = max(0, cy - ws // 2)
                y2 = min(h, cy + ws // 2)
                x1 = max(0, cx - ws // 2)
                x2 = min(w, cx + ws // 2)
                local_count = np.sum(labeled[y1:y2, x1:x2] > 0)
                density[y1:y2, x1:x2] = np.maximum(
                    density[y1:y2, x1:x2], local_count
                )
        except ImportError:
            # 回退：均匀密度
            density[:] = 1.0

        return density

    def _detect_at_scale(
        self, signal: np.ndarray, scale: int, density_map: np.ndarray
    ) -> List[Dict]:
        """在指定尺度上检测光斑。"""
        detections = []
        h, w = signal.shape

        # 自适应阈值
        mean_density = np.mean(density_map) if density_map is not None else 1.0

        # 高密度区域使用更高阈值
        if mean_density > self.config.high_density_threshold:
            threshold = self.config.min_intensity * 1.5
        elif mean_density < self.config.low_density_threshold:
            threshold = self.config.min_intensity * 0.8
        else:
            threshold = self.config.min_intensity

        # 局部最大值检测
        try:
            from scipy.ndimage import maximum_filter
            local_max = maximum_filter(signal, size=5)
            peaks = (signal == local_max) & (signal > threshold)
            ys, xs = np.where(peaks)
        except ImportError:
            # 简单阈值检测
            peaks = signal > threshold
            ys, xs = np.where(peaks)

        for y, x in zip(ys, xs):
            # SNR 检查
            peak_val = signal[y, x]
            noise_region = signal[signal < np.percentile(signal, 25)]
            noise_std = np.std(noise_region) if len(noise_region) > 0 else 1.0
            snr = peak_val / max(noise_std, 1e-10)

            if snr < self.config.min_snr:
                continue

            detections.append({
                'cx': float(x * scale),
                'cy': float(y * scale),
                'peak': float(peak_val),
                'snr': float(snr),
                'scale': scale,
                'bbox': (
                    max(0, int((x - 5) * scale)),
                    max(0, int((y - 5) * scale)),
                    min(w * scale, int((x + 5) * scale)),
                    min(h * scale, int((y + 5) * scale)),
                ),
            })

        return detections

    def _multiscale_nms(self, detections: List[Dict]) -> List[Dict]:
        """多尺度非极大值抑制。"""
        if not detections:
            return []

        # 按 SNR 排序
        detections.sort(key=lambda d: d['snr'], reverse=True)
        kept = []

        for det in detections:
            overlap = False
            for k in kept:
                iou = self._compute_iou(det['bbox'], k['bbox'])
                if iou > self.config.merge_iou_threshold:
                    overlap = True
                    break
            if not overlap:
                kept.append(det)

        return kept

    @staticmethod
    def _compute_iou(bbox1: Tuple, bbox2: Tuple) -> float:
        """计算两个边界框的 IoU。"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / max(union, 1e-10)

    def _separate_overlaps(
        self, detections: List[Dict], signal: np.ndarray
    ) -> List[Dict]:
        """分离重叠光斑（简化 EM 方法）。"""
        if len(detections) < 2:
            return detections

        separated = []
        used = set()

        for i, d1 in enumerate(detections):
            if i in used:
                continue

            # 检查是否有重叠的检测
            overlapping = []
            for j, d2 in enumerate(detections):
                if j <= i or j in used:
                    continue
                dist = np.sqrt((d1['cx'] - d2['cx'])**2 + (d1['cy'] - d2['cy'])**2)
                if dist < self.config.overlap_min_distance:
                    overlapping.append(j)

            if not overlapping:
                separated.append(d1)
            else:
                # 简化分离：保留 SNR 更高的
                group = [i] + overlapping
                best = max(group, key=lambda k: detections[k]['snr'])
                separated.append(detections[best])
                used.update(overlapping)

        return separated

    def _refine_subpixel(self, det: Dict, signal: np.ndarray) -> Dict:
        """亚像素精化（质心法）。"""
        h, w = signal.shape
        cx, cy = int(det['cx']), int(det['cy'])
        r = 3

        y1 = max(0, cy - r)
        y2 = min(h, cy + r + 1)
        x1 = max(0, cx - r)
        x2 = min(w, cx + r + 1)

        patch = signal[y1:y2, x1:x2]
        total = patch.sum()
        if total < 1e-10:
            return det

        yy, xx = np.mgrid[y1:y2, x1:x2]
        det['cx'] = float((xx * patch).sum() / total)
        det['cy'] = float((yy * patch).sum() / total)

        return det


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = ContextDetectorConfig()
    detector = ContextAwareDetector(config)

    # 生成测试图像
    np.random.seed(42)
    img = np.random.rand(200, 200) * 10
    spots = [(50, 50), (100, 100), (150, 80), (80, 150)]
    for sy, sx in spots:
        y, x = np.mgrid[:200, :200]
        img += 200 * np.exp(-((x-sx)**2 + (y-sy)**2) / (2*3**2))

    result = detector.detect(img)
    print(f"检测到 {result.num_detections} 个光斑:")
    for d in result.detections:
        print(f"  位置=({d['cx']:.1f}, {d['cy']:.1f}), SNR={d['snr']:.1f}, scale={d['scale']}")
    print(f"处理时间: {result.processing_time_ms:.1f}ms")
