"""
图像质量感知器 (v7.0)

基于无参考图像质量评估 (NR-IQA) 的光斑图像质量实时评分模块。
不依赖参考图像，仅从单帧光斑图像中提取多维特征进行质量评估。

灵感来源:
- scikit-image (https://scikit-image.org/) — 图像质量评估算法
- BRISQUE (IEEE TIP 2012) — 自然场景无参考质量评估
- NIQE (IEEE TIP 2013) — 自然图像质量评估器
- ISO 12233 — 分辨率测量标准

算法原理:
  综合以下 5 个维度的特征进行质量评分:
  1. 清晰度 (Laplacian 方差)
  2. 对比度 (RMS 对比度 + Michelson 对比度)
  3. 信噪比 (峰值/背景噪声)
  4. 信息熵 (Shannon 熵)
  5. 频域能量比 (高频/总能量)

外部依赖: numpy, cv2
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ImageQualityConfig:
    """图像质量评估配置。"""
    # 各维度权重 (总和应为 1.0)
    w_sharpness: float = 0.30
    w_contrast: float = 0.20
    w_snr: float = 0.25
    w_entropy: float = 0.10
    w_frequency: float = 0.15

    # ROI 配置
    roi_padding: int = 10  # ROI 内边距 (像素)

    # 频域分析参数
    freq_cutoff_ratio: float = 0.5  # 高频截止比例

    # 历史窗口
    history_length: int = 50


@dataclass
class ImageQualityReport:
    """图像质量评估报告。"""
    overall_score: float = 0.0          # 综合质量分数 (0-100)
    sharpness_score: float = 0.0        # 清晰度分数
    contrast_score: float = 0.0         # 对比度分数
    snr_score: float = 0.0              # 信噪比分数
    entropy_score: float = 0.0          # 信息熵分数
    frequency_score: float = 0.0        # 频域能量分数

    # 原始特征值
    laplacian_var: float = 0.0          # Laplacian 方差
    rms_contrast: float = 0.0           # RMS 对比度
    michelson_contrast: float = 0.0     # Michelson 对比度
    peak_snr: float = 0.0               # 峰值信噪比
    shannon_entropy: float = 0.0        # Shannon 熵
    freq_energy_ratio: float = 0.0      # 高频能量比

    # 趋势
    trend: str = "stable"               # "improving", "stable", "degrading"
    score_history: list = field(default_factory=list)

    # 诊断
    diagnosis: str = ""                 # 质量诊断描述
    suggestions: list = field(default_factory=list)  # 改善建议


class ImageQualityAssessor:
    """无参考图像质量评估器。

    对光斑图像进行实时质量评估，输出 0-100 的综合质量分数。
    支持趋势分析和质量诊断。

    使用示例:
        assessor = ImageQualityAssessor()
        report = assessor.assess(image)
        print(f"Quality: {report.overall_score:.1f}/100")
        print(f"Diagnosis: {report.diagnosis}")
    """

    def __init__(self, config: Optional[ImageQualityConfig] = None):
        self._config = config or ImageQualityConfig()
        self._score_history: list = []
        self._frame_count = 0

    @property
    def config(self) -> ImageQualityConfig:
        return self._config

    def assess(self, image: np.ndarray) -> ImageQualityReport:
        """评估光斑图像质量。

        Args:
            image: 灰度或彩色图像 (H, W) 或 (H, W, 3)

        Returns:
            ImageQualityReport: 完整的质量评估报告
        """
        self._frame_count += 1

        # 预处理
        gray = self._to_gray(image)
        gray = self._normalize(gray)

        # 计算各维度特征
        lap_var = self._compute_laplacian_variance(gray)
        rms_c, michelson_c = self._compute_contrast(gray)
        peak_snr = self._compute_snr(gray)
        entropy = self._compute_entropy(gray)
        freq_ratio = self._compute_frequency_energy(gray)

        # 归一化到 0-100
        sharpness = self._normalize_sharpness(lap_var)
        contrast = self._normalize_contrast(rms_c)
        snr = self._normalize_snr(peak_snr)
        ent = self._normalize_entropy(entropy)
        freq = self._normalize_frequency(freq_ratio)

        # 加权综合
        cfg = self._config
        overall = (
            cfg.w_sharpness * sharpness +
            cfg.w_contrast * contrast +
            cfg.w_snr * snr +
            cfg.w_entropy * ent +
            cfg.w_frequency * freq
        )
        overall = min(max(overall, 0.0), 100.0)

        # 趋势分析
        self._score_history.append(overall)
        if len(self._score_history) > self._config.history_length:
            self._score_history.pop(0)

        trend = self._compute_trend()

        # 诊断
        diagnosis, suggestions = self._diagnose(
            sharpness, contrast, snr, ent, freq
        )

        return ImageQualityReport(
            overall_score=overall,
            sharpness_score=sharpness,
            contrast_score=contrast,
            snr_score=snr,
            entropy_score=entropy,
            frequency_score=freq,
            laplacian_var=lap_var,
            rms_contrast=rms_c,
            michelson_contrast=michelson_c,
            peak_snr=peak_snr,
            shannon_entropy=entropy,
            freq_energy_ratio=freq_ratio,
            trend=trend,
            score_history=list(self._score_history),
            diagnosis=diagnosis,
            suggestions=suggestions,
        )

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        """转换为灰度图。"""
        if image.ndim == 2:
            return image.copy()
        if image.ndim == 3:
            # 尝试延迟导入 cv2
            try:
                import cv2
                return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            except ImportError:
                # 简单加权平均
                return np.mean(image[:, :, :3], axis=2).astype(np.float64)
        raise ValueError(f"Unsupported image shape: {image.shape}")

    def _normalize(self, gray: np.ndarray) -> np.ndarray:
        """归一化到 [0, 1]。"""
        img = gray.astype(np.float64)
        mn, mx = img.min(), img.max()
        if mx - mn < 1e-10:
            return np.zeros_like(img)
        return (img - mn) / (mx - mn)

    def _compute_laplacian_variance(self, img: np.ndarray) -> float:
        """计算 Laplacian 方差 (清晰度指标)。"""
        try:
            import cv2
            lap = cv2.Laplacian(img, cv2.CV_64F)
            return float(np.var(lap))
        except ImportError:
            # 简单差分近似
            dx = np.diff(img, axis=1)
            dy = np.diff(img, axis=0)
            grad_var = float(np.var(dx[:, :-1]) + np.var(dy[:-1, :]))
            return grad_var * 0.5

    def _compute_contrast(self, img: np.ndarray) -> Tuple[float, float]:
        """计算 RMS 对比度和 Michelson 对比度。"""
        mean = np.mean(img)
        std = np.std(img)
        rms = float(std) if mean > 1e-10 else 0.0

        min_val = np.min(img)
        max_val = np.max(img)
        denom = max_val + min_val
        michelson = float((max_val - min_val) / denom) if denom > 1e-10 else 0.0

        return rms, michelson

    def _compute_snr(self, img: np.ndarray) -> float:
        """计算峰值信噪比 (峰值/背景标准差)。"""
        peak = np.max(img)
        # 使用图像边缘区域估计背景噪声
        h, w = img.shape
        border = max(1, min(10, h // 10, w // 10))
        border_pixels = np.concatenate([
            img[:border, :].flatten(),
            img[-border:, :].flatten(),
            img[:, :border].flatten(),
            img[:, -border:].flatten(),
        ])
        bg_std = np.std(border_pixels)
        if bg_std < 1e-10:
            return 100.0  # 完美信噪比
        return float(peak / bg_std)

    def _compute_entropy(self, img: np.ndarray) -> float:
        """计算 Shannon 信息熵。"""
        # 量化为 256 级
        quantized = np.clip((img * 255).astype(np.int32), 0, 255)
        hist, _ = np.histogram(quantized, bins=256, range=(0, 256))
        hist = hist.astype(np.float64)
        total = hist.sum()
        if total < 1:
            return 0.0
        prob = hist / total
        prob = prob[prob > 0]
        return float(-np.sum(prob * np.log2(prob)))

    def _compute_frequency_energy(self, img: np.ndarray) -> float:
        """计算频域高频能量比。"""
        # 2D FFT
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        # 高频区域 (外圈)
        cutoff = int(min(h, w) * self._config.freq_cutoff_ratio)
        y, x = np.ogrid[-cy:h - cy, -cx:w - cx]
        high_freq_mask = (x**2 + y**2) >= cutoff**2

        total_energy = np.sum(magnitude**2)
        if total_energy < 1e-10:
            return 0.0
        high_energy = np.sum(magnitude[high_freq_mask]**2)
        return float(high_energy / total_energy)

    def _normalize_sharpness(self, lap_var: float) -> float:
        """归一化清晰度分数。"""
        # 典型光斑图像 Laplacian 方差范围: 0.001 ~ 0.1
        score = min(lap_var / 0.05, 1.0) * 100
        return score

    def _normalize_contrast(self, rms_contrast: float) -> float:
        """归一化对比度分数。"""
        # RMS 对比度范围: 0 ~ 0.5
        score = min(rms_contrast / 0.3, 1.0) * 100
        return score

    def _normalize_snr(self, peak_snr: float) -> float:
        """归一化信噪比分数。"""
        # SNR 范围: 1 ~ 100
        score = min(np.log10(max(peak_snr, 1.0)) / 2.0, 1.0) * 100
        return score

    def _normalize_entropy(self, entropy: float) -> float:
        """归一化信息熵分数。"""
        # 熵范围: 0 ~ 8 (log2(256))
        score = min(entropy / 6.0, 1.0) * 100
        return score

    def _normalize_frequency(self, freq_ratio: float) -> float:
        """归一化频域能量分数。"""
        # 高频能量比范围: 0 ~ 0.5
        score = min(freq_ratio / 0.3, 1.0) * 100
        return score

    def _compute_trend(self) -> str:
        """计算质量分数趋势。"""
        if len(self._score_history) < 10:
            return "stable"

        recent = self._score_history[-10:]
        slope = (recent[-1] - recent[0]) / len(recent)

        if slope > 0.5:
            return "improving"
        elif slope < -0.5:
            return "degrading"
        return "stable"

    def _diagnose(
        self,
        sharpness: float,
        contrast: float,
        snr: float,
        entropy: float,
        frequency: float
    ) -> Tuple[str, list]:
        """生成质量诊断和建议。"""
        issues = []
        suggestions = []

        if sharpness < 30:
            issues.append("模糊")
            suggestions.append("检查焦距是否正确对焦")
        if contrast < 25:
            issues.append("低对比度")
            suggestions.append("调整光源强度或相机曝光")
        if snr < 30:
            issues.append("低信噪比")
            suggestions.append("增加光源亮度或降低相机增益")
        if entropy < 20:
            issues.append("信息量不足")
            suggestions.append("检查图像是否过度饱和或全黑")
        if frequency < 20:
            issues.append("高频缺失")
            suggestions.append("检查光学系统分辨率和相机对焦")

        if not issues:
            diagnosis = "图像质量良好"
        else:
            diagnosis = "检测到问题: " + ", ".join(issues)

        return diagnosis, suggestions

    def reset(self):
        """重置评估器状态。"""
        self._score_history.clear()
        self._frame_count = 0
        logger.info("ImageQualityAssessor: Reset")

    def get_score_history(self) -> list:
        """获取历史质量分数。"""
        return list(self._score_history)
