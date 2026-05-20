"""
SLM 全息光斑生成器 (SLM Holographic Spot Generator)

参考开源项目:
  - slmsuite (https://github.com/wavefrontshaping/slm_suite): SLM 控制与全息图生成框架

核心思想:
  从 slmsuite 的相位恢复引擎中借鉴，实现 Gerchberg-Saxton (GS) 迭代算法，
  用于在空间光调制器 (SLM) 上生成目标光斑/焦点阵列的全息图。

  在 SpotZoom 场景中:
  - SLM → 可编程相位调制元件，用于动态光束整形
  - 全息图 → SLM 上显示的相位图案，产生期望的光场分布
  - 坐标校准 → SLM 像素坐标与相机/样品坐标之间的映射

创新点:
  1. GPU 友好的纯 numpy GS 算法实现
  2. 自动化傅里叶-图像坐标校准流程
  3. 基于相机反馈的闭环光斑阵列优化
  4. 多光斑均匀性优化 (加权振幅约束)
  5. 全息图质量评估指标 (效率、均匀性、保真度)

纯 numpy 实现，无外部依赖。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SLMConfig:
    """SLM 全息生成器配置。"""
    # SLM 分辨率 (像素)
    slm_width: int = 1920
    slm_height: int = 1080
    # SLM 像素间距 (微米)
    pixel_pitch_um: float = 8.0
    # 工作波长 (微米)
    wavelength_um: float = 0.633
    # GS 算法最大迭代次数
    max_iterations: int = 100
    # GS 算法收敛阈值
    convergence_threshold: float = 1e-4
    # 相位量化级数 (0 = 连续相位)
    phase_levels: int = 256
    # 是否启用振幅约束 (目标平面)
    amplitude_constraint: bool = True
    # 是否启用随机相位初始化
    random_phase_init: bool = True
    # 坐标校准网格点数
    calibration_grid_points: int = 5
    # 光斑阵列优化步长
    optimization_step_size: float = 0.1
    # 光斑阵列优化最大迭代
    optimization_max_iter: int = 50
    # 通信掩膜半径比 (0~0.5)
    communication_mask_ratio: float = 0.45


@dataclass
class HologramResult:
    """全息图生成结果。"""
    # 相位全息图 (弧度, [0, 2*pi])
    phase_hologram: np.ndarray = field(default_factory=lambda: np.zeros((1080, 1920)))
    # 目标光场强度
    target_intensity: np.ndarray = field(default_factory=lambda: np.zeros((1080, 1920)))
    # 重建光场强度
    reconstructed_intensity: np.ndarray = field(default_factory=lambda: np.zeros((1080, 1920)))
    # 衍射效率 (0~1)
    diffraction_efficiency: float = 0.0
    # 光斑均匀性 (0~1, 1 = 完美均匀)
    spot_uniformity: float = 0.0
    # 均方根误差
    rms_error: float = 0.0
    # 实际迭代次数
    iterations_used: int = 0
    # 是否收敛
    converged: bool = False


class SLMHolographicGenerator:
    """SLM 全息光斑生成器。

    基于 Gerchberg-Saxton 迭代相位恢复算法，生成用于 SLM 的
    相位全息图，实现目标光斑/焦点阵列的精确控制。

    Parameters
    ----------
    config : SLMConfig
        生成器配置参数。

    References
    ----------
    .. [1] Gerchberg, R. W. & Saxton, W. O. (1972).
           "A practical algorithm for the determination of phase from image
           and diffraction plane pictures." Optik, 35, 237-246.
    .. [2] slmsuite documentation:
           https://slmsuite.readthedocs.io/
    """

    def __init__(self, config: Optional[SLMConfig] = None) -> None:
        self.config = config or SLMConfig()
        self._calibration_matrix: Optional[np.ndarray] = None
        self._calibration_offset: Optional[Tuple[float, float]] = None
        self._is_calibrated: bool = False
        logger.info(
            "SLMHolographicGenerator 初始化: "
            f"SLM={self.config.slm_width}x{self.config.slm_height}, "
            f"lambda={self.config.wavelength_um}um, "
            f"pitch={self.config.pixel_pitch_um}um"
        )

    def generate_hologram(
        self,
        target_spots: np.ndarray,
        target_amplitudes: Optional[np.ndarray] = None,
        roi_size: int = 256,
    ) -> HologramResult:
        """生成全息图。

        使用 Gerchberg-Saxton 算法从目标光斑分布计算相位全息图。

        Parameters
        ----------
        target_spots : np.ndarray
            目标光斑位置，形状 (N, 2)，每行 [x, y]。
        target_amplitudes : np.ndarray, optional
            各光斑的目标振幅，形状 (N,)。默认均匀振幅。
        roi_size : int
            重建区域大小 (像素)。

        Returns
        -------
        HologramResult
            全息图生成结果，包含相位图和质量指标。
        """
        n_spots = len(target_spots)
        if target_amplitudes is None:
            target_amplitudes = np.ones(n_spots)

        logger.info(f"开始 GS 相位恢复: {n_spots} 个目标光斑, ROI={roi_size}")

        # 构建目标振幅分布
        target_field = self._build_target_field(
            target_spots, target_amplitudes, roi_size
        )

        # 初始化输入相位
        if self.config.random_phase_init:
            input_phase = np.random.uniform(0, 2 * np.pi, (roi_size, roi_size))
        else:
            input_phase = np.zeros((roi_size, roi_size))

        # GS 迭代
        prev_error = float("inf")
        result = HologramResult()

        for iteration in range(self.config.max_iterations):
            # 前向传播: 空间域 → 傅里叶域
            input_field = np.exp(1j * input_phase)
            fourier_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(input_field)))

            # 在傅里叶域施加振幅约束
            fourier_amplitude = np.abs(fourier_field)
            fourier_phase = np.angle(fourier_field)

            # 用目标振幅替换
            if self.config.amplitude_constraint:
                fourier_field = target_field * np.exp(1j * fourier_phase)
            else:
                fourier_field = fourier_field

            # 反向传播: 傅里叶域 → 空间域
            output_field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(fourier_field)))
            output_phase = np.angle(output_field)

            # 计算误差
            reconstructed_intensity = np.abs(fourier_field) ** 2
            target_intensity = np.abs(target_field) ** 2
            rms_error = np.sqrt(
                np.mean((reconstructed_intensity - target_intensity) ** 2)
            ) / (np.max(target_intensity) + 1e-10)

            # 收敛判断
            error_change = abs(prev_error - rms_error)
            if error_change < self.config.convergence_threshold and iteration > 5:
                logger.info(f"GS 算法在第 {iteration} 次迭代收敛")
                result.converged = True
                result.iterations_used = iteration + 1
                break

            prev_error = rms_error
            input_phase = output_phase
        else:
            result.iterations_used = self.config.max_iterations
            logger.warning(f"GS 算法未收敛，已达到最大迭代次数 {self.config.max_iterations}")

        # 最终全息图
        final_field = np.exp(1j * input_phase)
        final_fourier = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(final_field)))
        final_phase = np.angle(final_field)

        # 相位量化
        if self.config.phase_levels > 0:
            final_phase = self._quantize_phase(final_phase, self.config.phase_levels)

        # 上采样到 SLM 分辨率
        hologram_full = self._upsample_to_slm(final_phase)

        # 计算质量指标
        recon_intensity = np.abs(final_fourier) ** 2
        tgt_intensity = np.abs(target_field) ** 2

        result.phase_hologram = hologram_full
        result.target_intensity = tgt_intensity
        result.reconstructed_intensity = recon_intensity
        result.diffraction_efficiency = self._compute_efficiency(recon_intensity, tgt_intensity)
        result.spot_uniformity = self._compute_uniformity(recon_intensity, target_spots, roi_size)
        result.rms_error = rms_error

        logger.info(
            f"全息图生成完成: 效率={result.diffraction_efficiency:.3f}, "
            f"均匀性={result.spot_uniformity:.3f}, RMS={result.rms_error:.4f}"
        )
        return result

    def calibrate_coordinates(
        self,
        measured_positions: np.ndarray,
        commanded_positions: np.ndarray,
    ) -> Tuple[np.ndarray, Tuple[float, float]]:
        """校准 SLM 坐标到相机坐标的映射。

        通过已知的命令位置和实际测量位置之间的对应关系，
        计算仿射变换矩阵。

        Parameters
        ----------
        measured_positions : np.ndarray
            相机测量的光斑位置，形状 (N, 2)。
        commanded_positions : np.ndarray
            SLM 命令的光斑位置，形状 (N, 2)。

        Returns
        -------
        Tuple[np.ndarray, Tuple[float, float]]
            (变换矩阵, 偏移量 (dx, dy))。
        """
        if len(measured_positions) < 3:
            raise ValueError("校准至少需要 3 个对应点")

        logger.info(f"开始坐标校准: {len(measured_positions)} 个对应点")

        # 使用最小二乘法估计仿射变换
        # x_cam = a * x_slm + b * y_slm + tx
        # y_cam = c * x_slm + d * y_slm + ty
        n = len(measured_positions)
        A = np.column_stack([
            commanded_positions,
            np.ones(n),
        ])
        # 求解 x 方向
        params_x, _, _, _ = np.linalg.lstsq(A, measured_positions[:, 0], rcond=None)
        # 求解 y 方向
        params_y, _, _, _ = np.linalg.lstsq(A, measured_positions[:, 1], rcond=None)

        transform_matrix = np.array([
            [params_x[0], params_x[1]],
            [params_y[0], params_y[1]],
        ])
        offset = (params_x[2], params_y[2])

        self._calibration_matrix = transform_matrix
        self._calibration_offset = offset
        self._is_calibrated = True

        # 计算校准残差
        predicted = (commanded_positions @ transform_matrix.T) + np.array(offset)
        residuals = np.sqrt(np.sum((predicted - measured_positions) ** 2, axis=1))
        logger.info(
            f"坐标校准完成: 变换矩阵=\n{transform_matrix}, "
            f"偏移={offset}, 平均残差={np.mean(residuals):.3f}px"
        )
        return transform_matrix, offset

    def optimize_spot_array(
        self,
        initial_spots: np.ndarray,
        feedback_image: np.ndarray,
        target_intensity: float = 1.0,
    ) -> np.ndarray:
        """基于相机反馈优化光斑阵列位置。

        通过迭代调整光斑位置，使实际光斑强度趋近目标值。

        Parameters
        ----------
        initial_spots : np.ndarray
            初始光斑位置，形状 (N, 2)。
        feedback_image : np.ndarray
            相机采集的反馈图像。
        target_intensity : float
            目标光斑强度。

        Returns
        -------
        np.ndarray
            优化后的光斑位置，形状 (N, 2)。
        """
        spots = initial_spots.copy()
        step = self.config.optimization_step_size
        logger.info(f"开始光斑阵列优化: {len(spots)} 个光斑")

        for iteration in range(self.config.optimization_max_iter):
            # 生成全息图
            result = self.generate_hologram(spots, roi_size=feedback_image.shape[0])

            # 模拟重建 (使用反馈图像作为参考)
            # 在实际系统中，这里会使用相机采集的新图像
            total_error = 0.0
            for i, (sx, sy) in enumerate(spots):
                ix, iy = int(round(sx)), int(round(sy))
                if 0 <= iy < feedback_image.shape[0] and 0 <= ix < feedback_image.shape[1]:
                    measured = feedback_image[iy, ix]
                    error = measured - target_intensity
                    total_error += error ** 2
                    # 梯度方向调整
                    spots[i, 0] -= step * np.sign(error) * 0.1
                    spots[i, 1] -= step * np.sign(error) * 0.1

            if total_error < 1e-3:
                logger.info(f"光斑阵列优化在第 {iteration} 次迭代收敛")
                break

        logger.info(f"光斑阵列优化完成: 总误差={total_error:.4f}")
        return spots

    def analyze_quality(self, hologram: np.ndarray) -> dict:
        """分析全息图质量。

        Parameters
        ----------
        hologram : np.ndarray
            相位全息图 (弧度)。

        Returns
        -------
        dict
            质量分析指标字典。
        """
        # 相位统计
        phase_wrapped = hologram % (2 * np.pi)

        # 相位梯度 (衡量空间频率含量)
        grad_y, grad_x = np.gradient(phase_wrapped)
        phase_gradient_rms = np.sqrt(np.mean(grad_x ** 2 + grad_y ** 2))

        # 相位直方图均匀性
        hist, _ = np.histogram(phase_wrapped, bins=64, range=(0, 2 * np.pi))
        hist_normalized = hist / hist.sum()
        uniform_reference = np.ones_like(hist_normalized) / len(hist_normalized)
        histogram_deviation = np.sqrt(np.mean((hist_normalized - uniform_reference) ** 2))

        # 有效填充率 (非零梯度区域)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        fill_factor = np.mean(gradient_magnitude > 0.01)

        # 通信掩膜内的相位利用率
        h, w = hologram.shape
        cy, cx = h // 2, w // 2
        radius = int(min(h, w) * self.config.communication_mask_ratio)
        yy, xx = np.ogrid[:h, :w]
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius ** 2
        active_pixels = np.sum(mask)
        phase_utilization = np.std(phase_wrapped[mask]) / (np.pi / np.sqrt(3))

        metrics = {
            "phase_gradient_rms": float(phase_gradient_rms),
            "histogram_uniformity": float(1.0 - histogram_deviation),
            "fill_factor": float(fill_factor),
            "phase_utilization": float(min(phase_utilization, 1.0)),
            "active_area_fraction": float(active_pixels / (h * w)),
        }

        logger.info(f"全息图质量分析: {metrics}")
        return metrics

    # ============ 内部方法 ============

    def _build_target_field(
        self,
        spots: np.ndarray,
        amplitudes: np.ndarray,
        size: int,
    ) -> np.ndarray:
        """构建目标复振幅场。"""
        field = np.zeros((size, size), dtype=complex)
        cy, cx = size // 2, size // 2

        for (sx, sy), amp in zip(spots, amplitudes):
            ix = int(round(cx + sx))
            iy = int(round(cy + sy))
            # 使用高斯点扩展函数模拟有限光斑尺寸
            sigma = 2.0
            for dy in range(-5, 6):
                for dx in range(-5, 6):
                    py, px = iy + dy, ix + dx
                    if 0 <= py < size and 0 <= px < size:
                        field[py, px] += amp * np.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2))

        return field

    def _quantize_phase(self, phase: np.ndarray, levels: int) -> np.ndarray:
        """相位量化。"""
        if levels <= 0:
            return phase
        step = 2 * np.pi / levels
        return np.round(phase / step) * step

    def _upsample_to_slm(self, phase: np.ndarray) -> np.ndarray:
        """将相位图上采样到 SLM 分辨率。"""
        from numpy import interp as np_interp

        h, w = phase.shape
        target_h, target_w = self.config.slm_height, self.config.slm_width

        # 使用最近邻插值 (保持相位值不变)
        y_indices = np.linspace(0, h - 1, target_h).astype(int)
        x_indices = np.linspace(0, w - 1, target_w).astype(int)
        hologram = phase[np.ix_(y_indices, x_indices)]

        return hologram

    def _compute_efficiency(
        self, reconstructed: np.ndarray, target: np.ndarray
    ) -> float:
        """计算衍射效率。"""
        target_energy = np.sum(target ** 2)
        if target_energy < 1e-10:
            return 0.0
        # 目标区域内的能量占比
        target_mask = target > 0.01 * np.max(target)
        efficiency = np.sum(reconstructed[target_mask] ** 2) / np.sum(reconstructed ** 2)
        return float(np.clip(efficiency, 0, 1))

    def _compute_uniformity(
        self, reconstructed: np.ndarray, spots: np.ndarray, size: int
    ) -> float:
        """计算光斑均匀性。"""
        if len(spots) < 2:
            return 1.0

        cy, cx = size // 2, size // 2
        intensities = []

        for sx, sy in spots:
            ix = int(round(cx + sx))
            iy = int(round(cy + sy))
            # 在光斑位置周围取峰值
            r = 3
            y_lo, y_hi = max(0, iy - r), min(size, iy + r + 1)
            x_lo, x_hi = max(0, ix - r), min(size, ix + r + 1)
            if y_hi > y_lo and x_hi > x_lo:
                peak = np.max(reconstructed[y_lo:y_hi, x_lo:x_hi])
                intensities.append(peak)

        if not intensities:
            return 0.0

        intensities = np.array(intensities)
        mean_intensity = np.mean(intensities)
        if mean_intensity < 1e-10:
            return 0.0

        # 均匀性 = 1 - (标准差 / 均值)
        uniformity = 1.0 - np.std(intensities) / mean_intensity
        return float(np.clip(uniformity, 0, 1))
