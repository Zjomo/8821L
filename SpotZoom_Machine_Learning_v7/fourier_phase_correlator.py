"""
傅里叶相位相关器 (Fourier Phase Correlator)

基于傅里叶相位相关的高精度亚像素图像配准模块。
通过频域互功率谱分析实现亚像素级 (1/100 像素) 的位移检测，
支持多分辨率粗到精策略和漂移估计。

灵感来源:
- Sub-pixel phase correlation registration (Guizar-Sicairos et al. 2008)
- Fourier-based image registration (Kuglin & Hines 1975)
- OpenCV phaseCorrelate: OpenCV 相位相关实现
- scikit-image register_translation: 科学图像配准

算法原理:
  1. 互功率谱: R = (F1 * conj(F2)) / |F1 * conj(F2)|
     其中 F1, F2 为两幅图像的傅里叶变换
  2. 逆傅里叶变换: r = IFFT(R)
     峰值位置即为整数像素位移
  3. 亚像素精度: 对峰值邻域进行上采样 (DFT 插值)
     精度可达 1/upsampling_factor 像素
  4. 多分辨率: 从低分辨率开始，逐步细化到全分辨率
  5. 窗函数: 使用 Hanning/Blackman 窗减少边缘效应

外部依赖: numpy, cv2 (可选)
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from enum import Enum

logger = logging.getLogger(__name__)


class WindowType(Enum):
    """窗函数类型枚举。"""
    NONE = "none"           # 无窗函数
    HANNING = "hanning"     # Hanning 窗
    HAMMING = "hamming"     # Hamming 窗
    BLACKMAN = "blackman"   # Blackman 窗
    TUKEY = "tukey"         # Tukey 窗


@dataclass
class PhaseCorrelationResult:
    """相位相关配准结果。"""
    shift_x: float = 0.0                 # x 方向位移 (像素)
    shift_y: float = 0.0                 # y 方向位移 (像素)
    confidence: float = 0.0              # 置信度 (0-1, 1=最高)
    peak_value: float = 0.0              # 相关峰归一化值
    peak_sharpness: float = 0.0          # 峰值锐度 (能量集中度)
    subpixel_accuracy: float = 0.0       # 亚像素精度估计
    coarse_shift_x: float = 0.0          # 粗位移 x (整数像素)
    coarse_shift_y: float = 0.0          # 粗位移 y (整数像素)
    cross_power_spectrum: np.ndarray = None  # 互功率谱 (用于诊断)
    correlation_map: np.ndarray = None   # 相关图 (用于诊断)
    processing_time_ms: float = 0.0      # 处理耗时 (ms)


@dataclass
class PhaseCorrelatorConfig:
    """傅里叶相位相关器配置。"""
    # 亚像素精度
    upsampling_factor: int = 100          # 上采样因子 (精度 = 1/factor 像素)

    # 窗函数
    window_type: WindowType = WindowType.HANNING

    # 多分辨率
    coarse_levels: int = 3                # 粗到精细层数

    # 置信度
    min_confidence: float = 0.1           # 最小置信度阈值

    # 漂移估计
    drift_estimation_window: int = 10     # 漂移估计窗口 (帧数)

    # 数值参数
    epsilon: float = 1e-10                # 数值稳定性常数
    max_image_size: int = 2048            # 最大图像尺寸 (性能限制)


class FourierPhaseCorrelator:
    """傅里叶相位相关器。

    高精度亚像素图像配准，基于频域互功率谱分析。
    支持多分辨率策略和漂移估计。

    使用示例:
        correlator = FourierPhaseCorrelator()
        result = correlator.register(reference_image, target_image)
        print(f"Shift: ({result.shift_x:.3f}, {result.shift_y:.3f}) pixels")
        print(f"Confidence: {result.confidence:.2%}")
        print(f"Sub-pixel accuracy: 1/{correlator.config.upsampling_factor} px")
    """

    def __init__(self, config: Optional[PhaseCorrelatorConfig] = None):
        self._config = config or PhaseCorrelatorConfig()
        self._shift_history: List[Tuple[float, float]] = []
        self._confidence_history: List[float] = []

    @property
    def config(self) -> PhaseCorrelatorConfig:
        return self._config

    def register(
        self,
        reference: np.ndarray,
        target: np.ndarray,
    ) -> PhaseCorrelationResult:
        """计算两幅图像之间的亚像素位移。

        Args:
            reference: 参考图像 (H, W), float64
            target: 目标图像 (H, W), float64

        Returns:
            PhaseCorrelationResult: 配准结果
        """
        import time
        t0 = time.perf_counter()

        # 输入预处理
        ref = self._preprocess(reference)
        tgt = self._preprocess(target)

        # 多分辨率粗到精
        coarse_shift = np.array([0.0, 0.0])

        for level in range(self._config.coarse_levels, 0, -1):
            scale = 2 ** level

            # 下采样
            ref_small = self._downsample(ref, scale)
            tgt_small = self._downsample(tgt, scale)

            if ref_small.shape[0] < 8 or ref_small.shape[1] < 8:
                continue

            # 粗位移估计
            level_shift = self._compute_integer_shift(ref_small, tgt_small)

            # 转换到原始分辨率
            coarse_shift = level_shift * scale

        # 精细亚像素估计 (使用原始分辨率 + 粗位移补偿)
        result = self._compute_subpixel_shift(ref, tgt, coarse_shift)

        # 保存历史
        self._shift_history.append((result.shift_x, result.shift_y))
        self._confidence_history.append(result.confidence)
        if len(self._shift_history) > 200:
            self._shift_history.pop(0)
            self._confidence_history.pop(0)

        result.processing_time_ms = (time.perf_counter() - t0) * 1000

        logger.debug(
            f"FourierPhaseCorrelator: shift=({result.shift_x:.4f}, {result.shift_y:.4f}), "
            f"confidence={result.confidence:.3f}, peak={result.peak_value:.3f}, "
            f"time={result.processing_time_ms:.1f}ms"
        )

        return result

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """图像预处理。

        - 转换为灰度 float64
        - 限制最大尺寸
        - 应用窗函数

        Args:
            image: 输入图像

        Returns:
            预处理后图像
        """
        # 转灰度
        if image.ndim == 3:
            gray = np.mean(image[:, :, :min(3, image.shape[2])], axis=2)
        else:
            gray = image.copy()

        gray = gray.astype(np.float64)

        # 限制尺寸
        cfg = self._config
        h, w = gray.shape
        if max(h, w) > cfg.max_image_size:
            scale = cfg.max_image_size / max(h, w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            try:
                import cv2
                gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
            except ImportError:
                step_h = max(1, h // new_h)
                step_w = max(1, w // new_w)
                gray = gray[:new_h * step_h:step_h, :new_w * step_w:step_w]

        return gray

    def _apply_window(self, image: np.ndarray) -> np.ndarray:
        """应用窗函数。

        Args:
            image: 输入图像 (H, W)

        Returns:
            加窗后图像
        """
        cfg = self._config
        if cfg.window_type == WindowType.NONE:
            return image

        h, w = image.shape

        if cfg.window_type == WindowType.HANNING:
            win_h = np.hanning(h)
            win_w = np.hanning(w)
        elif cfg.window_type == WindowType.HAMMING:
            win_h = np.hamming(h)
            win_w = np.hamming(w)
        elif cfg.window_type == WindowType.BLACKMAN:
            win_h = np.blackman(h)
            win_w = np.blackman(w)
        elif cfg.window_type == WindowType.TUKEY:
            alpha = 0.5
            win_h = self._tukey_window(h, alpha)
            win_w = self._tukey_window(w, alpha)
        else:
            return image

        window_2d = np.outer(win_h, win_w)
        return image * window_2d

    def _tukey_window(self, length: int, alpha: float) -> np.ndarray:
        """Tukey 窗函数。

        Args:
            length: 窗长度
            alpha: 锥度参数 (0=矩形, 1=Hanning)

        Returns:
            窗函数
        """
        x = np.arange(length, dtype=np.float64) / (length - 1)
        window = np.ones(length, dtype=np.float64)

        # 左侧锥度
        left = x < alpha / 2
        window[left] = 0.5 * (1 + np.cos(2 * np.pi * (x[left] / alpha - 0.5)))

        # 右侧锥度
        right = x > 1 - alpha / 2
        window[right] = 0.5 * (1 + np.cos(2 * np.pi * (x[right] / alpha - 0.5 + 1)))

        return window

    def _downsample(self, image: np.ndarray, factor: int) -> np.ndarray:
        """下采样图像。

        Args:
            image: 输入图像
            factor: 下采样因子

        Returns:
            下采样后图像
        """
        h, w = image.shape
        new_h = max(4, h // factor)
        new_w = max(4, w // factor)

        try:
            import cv2
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        except ImportError:
            step_h = max(1, h // new_h)
            step_w = max(1, w // new_w)
            return image[:new_h * step_h:step_h, :new_w * step_w:step_w]

    def _compute_integer_shift(
        self, ref: np.ndarray, tgt: np.ndarray
    ) -> np.ndarray:
        """计算整数像素位移。

        Args:
            ref: 参考图像
            tgt: 目标图像

        Returns:
            位移 [shift_y, shift_x]
        """
        # 应用窗函数
        ref_win = self._apply_window(ref)
        tgt_win = self._apply_window(tgt)

        # 傅里叶变换
        F1 = np.fft.fft2(ref_win)
        F2 = np.fft.fft2(tgt_win)

        # 互功率谱
        cross_power = F1 * np.conj(F2)
        magnitude = np.abs(cross_power) + self._config.epsilon
        normalized = cross_power / magnitude

        # 逆傅里叶变换
        correlation = np.real(np.fft.ifft2(normalized))

        # 找峰值
        max_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
        shift_y, shift_x = max_idx

        # 处理周期性边界 (如果位移超过一半，取反方向)
        h, w = correlation.shape
        if shift_y > h // 2:
            shift_y -= h
        if shift_x > w // 2:
            shift_x -= w

        return np.array([float(shift_y), float(shift_x)])

    def _compute_subpixel_shift(
        self,
        ref: np.ndarray,
        tgt: np.ndarray,
        coarse_shift: np.ndarray,
    ) -> PhaseCorrelationResult:
        """计算亚像素位移。

        使用 DFT 上采样方法实现亚像素精度。

        Args:
            ref: 参考图像
            tgt: 目标图像
            coarse_shift: 粗位移 [shift_y, shift_x]

        Returns:
            PhaseCorrelationResult
        """
        cfg = self._config

        # 应用窗函数
        ref_win = self._apply_window(ref)
        tgt_win = self._apply_window(tgt)

        # 傅里叶变换
        F1 = np.fft.fft2(ref_win)
        F2 = np.fft.fft2(tgt_win)

        # 互功率谱
        cross_power = F1 * np.conj(F2)
        magnitude = np.abs(cross_power) + cfg.epsilon
        normalized = cross_power / magnitude

        # 整数位移 (用于诊断)
        correlation = np.real(np.fft.ifft2(normalized))
        max_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
        int_shift_y, int_shift_x = max_idx
        h, w = correlation.shape
        if int_shift_y > h // 2:
            int_shift_y -= h
        if int_shift_x > w // 2:
            int_shift_x -= w

        # 亚像素精化: DFT 上采样
        sub_shift = self._dft_upsample(
            normalized, coarse_shift, cfg.upsampling_factor
        )

        # 置信度评估
        peak_value = float(np.max(correlation))
        confidence = self._compute_confidence(correlation, peak_value)

        # 峰值锐度
        sharpness = self._compute_peak_sharpness(correlation)

        # 亚像素精度估计
        accuracy = 1.0 / cfg.upsampling_factor

        result = PhaseCorrelationResult(
            shift_x=sub_shift[1],
            shift_y=sub_shift[0],
            confidence=confidence,
            peak_value=peak_value,
            peak_sharpness=sharpness,
            subpixel_accuracy=accuracy,
            coarse_shift_x=float(int_shift_x),
            coarse_shift_y=float(int_shift_y),
            cross_power_spectrum=np.abs(normalized),
            correlation_map=correlation,
        )

        return result

    def _dft_upsample(
        self,
        cross_power: np.ndarray,
        coarse_shift: np.ndarray,
        upsampling_factor: int,
    ) -> np.ndarray:
        """DFT 上采样实现亚像素位移。

        在互功率谱的峰值邻域进行高分辨率 DFT，
        实现亚像素精度的位移检测。

        Args:
            cross_power: 归一化互功率谱
            coarse_shift: 粗位移 [shift_y, shift_x]
            upsampling_factor: 上采样因子

        Returns:
            精细位移 [shift_y, shift_x]
        """
        h, w = cross_power.shape
        cfg = self._config

        # 粗位移对应的频域中心
        center_y = int(round(coarse_shift[0])) % h
        center_x = int(round(coarse_shift[1])) % w

        # 上采样区域大小 (2x2 像素邻域)
        radius = 1
        upsampled_size = upsampling_factor * 2 * radius

        # 频率坐标
        freq_y = np.arange(upsampled_size, dtype=np.float64) / upsampled_size - 0.5
        freq_x = np.arange(upsampled_size, dtype=np.float64) / upsampled_size - 0.5
        freq_y_grid, freq_x_grid = np.meshgrid(freq_y, freq_x, indexing='ij')

        # 在互功率谱的峰值邻域进行 DFT
        # 收集邻域像素
        upsampled = np.zeros((upsampled_size, upsampled_size), dtype=np.complex128)

        for iy in range(-radius, radius + 1):
            for ix in range(-radius, radius + 1):
                src_y = (center_y + iy) % h
                src_x = (center_x + ix) % w
                # 计算该像素对上采样 DFT 的贡献
                phase = 2 * np.pi * (
                    iy * freq_y_grid + ix * freq_x_grid
                )
                upsampled += cross_power[src_y, src_x] * np.exp(1j * phase)

        # 找到上采样后的峰值
        upsampled_real = np.real(upsampled)
        max_idx = np.unravel_index(np.argmax(upsampled_real), upsampled_real.shape)
        sub_y = max_idx[0] / upsampling_factor - radius
        sub_x = max_idx[1] / upsampling_factor - radius

        # 组合粗位移和精细位移
        total_shift_y = coarse_shift[0] + sub_y
        total_shift_x = coarse_shift[1] + sub_x

        return np.array([total_shift_y, total_shift_x])

    def _compute_confidence(
        self, correlation: np.ndarray, peak_value: float
    ) -> float:
        """计算配准置信度。

        基于相关峰与次峰的比值。

        Args:
            correlation: 相关图
            peak_value: 峰值

        Returns:
            置信度 (0-1)
        """
        # 将峰值置零后找次峰
        masked = correlation.copy()
        h, w = masked.shape
        cy, cx = np.unravel_index(np.argmax(masked), masked.shape)
        # 掩盖峰值邻域 (5x5)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                yy = (cy + dy) % h
                xx = (cx + dx) % w
                masked[yy, xx] = 0

        second_peak = np.max(np.abs(masked))

        if second_peak < self._config.epsilon:
            return 1.0

        ratio = peak_value / (second_peak + self._config.epsilon)
        confidence = 1.0 - 1.0 / ratio
        return float(np.clip(confidence, 0, 1))

    def _compute_peak_sharpness(self, correlation: np.ndarray) -> float:
        """计算相关峰锐度。

        锐度 = 峰值能量 / 总能量

        Args:
            correlation: 相关图

        Returns:
            峰值锐度 (0-1)
        """
        total_energy = np.sum(correlation ** 2)
        if total_energy < self._config.epsilon:
            return 0.0

        # 峰值邻域能量 (3x3)
        h, w = correlation.shape
        cy, cx = np.unravel_index(np.argmax(correlation), correlation.shape)
        peak_energy = 0.0
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                yy = (cy + dy) % h
                xx = (cx + dx) % w
                peak_energy += correlation[yy, xx] ** 2

        return float(peak_energy / total_energy)

    def estimate_drift(
        self, n_frames: int = 10
    ) -> Tuple[float, float, float]:
        """估计累积漂移。

        Args:
            n_frames: 使用最近 N 帧

        Returns:
            (drift_x, drift_y, drift_rate)
            drift_rate: 漂移速率 (像素/帧)
        """
        if len(self._shift_history) < 2:
            return 0.0, 0.0, 0.0

        recent = self._shift_history[-n_frames:]
        shifts = np.array(recent)

        # 累积漂移
        drift_x = float(np.sum(shifts[:, 0]))
        drift_y = float(np.sum(shifts[:, 1]))

        # 漂移速率
        drift_rate = float(np.mean(np.sqrt(shifts[:, 0] ** 2 + shifts[:, 1] ** 2)))

        return drift_x, drift_y, drift_rate

    def compensate_drift(
        self, image: np.ndarray, n_frames: int = 10
    ) -> np.ndarray:
        """补偿累积漂移。

        Args:
            image: 输入图像
            n_frames: 使用最近 N 帧估计漂移

        Returns:
            漂移补偿后图像
        """
        drift_x, drift_y, _ = self.estimate_drift(n_frames)

        if abs(drift_x) < 0.01 and abs(drift_y) < 0.01:
            return image.copy()

        try:
            import cv2
            h, w = image.shape[:2]
            M = np.float32([[1, 0, -drift_x], [0, 1, -drift_y]])
            compensated = cv2.warpAffine(image, M, (w, h))
            return compensated
        except ImportError:
            # 简单平移
            result = image.copy()
            dx = int(round(drift_x))
            dy = int(round(drift_y))
            if dx > 0:
                result[:, dx:] = image[:, :-dx]
            elif dx < 0:
                result[:, :dx] = image[:, -dx:]
            if dy > 0:
                result[dy:, :] = image[:-dy, :]
            elif dy < 0:
                result[:dy, :] = image[-dy:, :]
            return result

    def reset(self):
        """重置相关器状态。"""
        self._shift_history.clear()
        self._confidence_history.clear()
        logger.info("FourierPhaseCorrelator: Reset")

    def get_shift_history(self) -> List[Tuple[float, float]]:
        """获取位移历史。"""
        return list(self._shift_history)

    def get_average_confidence(self) -> float:
        """获取平均置信度。"""
        if not self._confidence_history:
            return 0.0
        return float(np.mean(self._confidence_history))
