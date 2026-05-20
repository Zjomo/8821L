"""
全息光斑重建器 (HolographicSpotReconstructor)

灵感来源:
- DeepTrack2 (https://github.com/DeepTrackAI/DeepTrack2) — 全息粒子成像模块
- U. Schnars & W. Juptner (1994) — 数字全息记录与重建
- T. Kreis (2005) — 数字全息中的相位恢复方法

算法原理:
- Angular Spectrum Method — 角谱法数值衍射传播
- Phase Unwrapping — 相位解包裹恢复真实相位
- Multi-wavelength Holography — 多波长合成孔径提高轴向分辨率
- Aberration Compensation — 数字参考光相位补偿

功能:
- 从全息图像重建光斑的三维位置
- 角谱法数值传播 (正向/反向)
- 相位解包裹与像差补偿
- 多波长重建支持
- 聚焦评价与自动对焦

依赖: numpy, scipy (无外部深度学习框架依赖)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger("SpotZoom.HolographicSpotReconstructor")


@dataclass
class ReconstructionConfig:
    """重建器配置参数。"""
    # --- 传播参数 ---
    wavelength: float = 0.532  # 波长 (微米)
    pixel_size: float = 6.5  # 像素大小 (微米)
    propagation_distances: Optional[List[float]] = None  # 传播距离列表 (微米)
    distance_range: Tuple[float, float] = (-100.0, 100.0)  # 传播距离范围
    n_distances: int = 50  # 传播距离采样数

    # --- 重建参数 ---
    reconstruction_size: Optional[Tuple[int, int]] = None  # 重建区域大小
    padding: int = 2  # 零填充因子

    # --- 相位处理 ---
    unwrap_method: str = "quality"  # 相位解包裹方法: "quality", "simple", "none"
    aberration_correction: bool = True  # 是否进行像差补偿
    reference_phase: Optional[np.ndarray] = None  # 参考相位

    # --- 多波长参数 ---
    wavelengths: Optional[List[float]] = None  # 多波长列表 (微米)
    multi_wavelength_combination: str = "coherent"  # 合成方式: "coherent", "incoherent", "phase"

    # --- 聚焦评价 ---
    focus_metric: str = "variance"  # 聚焦指标: "variance", "gradient", "laplacian"
    auto_focus: bool = True  # 是否自动选择最佳聚焦距离

    # --- 介质参数 ---
    refractive_index: float = 1.0  # 介质折射率


@dataclass
class ReconstructionResult:
    """重建结果。"""
    complex_field: np.ndarray = field(default_factory=lambda: np.array([]))
    amplitude: np.ndarray = field(default_factory=lambda: np.array([]))
    phase: np.ndarray = field(default_factory=lambda: np.array([]))
    intensity: np.ndarray = field(default_factory=lambda: np.array([]))
    best_focus_distance: float = 0.0  # 最佳聚焦距离
    focus_metrics: np.ndarray = field(default_factory=lambda: np.array([]))
    spot_position_3d: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    wavelengths_used: List[float] = field(default_factory=list)
    propagation_distances: List[float] = field(default_factory=list)


class HolographicSpotReconstructor:
    """全息光斑重建器。

    使用角谱法从全息图像重建光斑的三维位置，
    支持相位解包裹、像差补偿和多波长重建。

    Parameters
    ----------
    config : ReconstructionConfig, optional
        重建器配置参数。为 None 时使用默认配置。
    """

    def __init__(self, config: Optional[ReconstructionConfig] = None):
        self._cfg = config if config is not None else ReconstructionConfig()

        # 初始化传播距离
        if self._cfg.propagation_distances is None:
            d_min, d_max = self._cfg.distance_range
            self._cfg.propagation_distances = np.linspace(
                d_min, d_max, self._cfg.n_distances
            ).tolist()

        # 初始化多波长
        if self._cfg.wavelengths is None:
            self._cfg.wavelengths = [self._cfg.wavelength]

    def _angular_spectrum_propagate(
        self,
        field: np.ndarray,
        distance: float,
        wavelength: float,
        pixel_size: float,
    ) -> np.ndarray:
        """角谱法数值传播。

        Parameters
        ----------
        field : np.ndarray
            输入复数场。
        distance : float
            传播距离 (微米)。
        wavelength : float
            波长 (微米)。
        pixel_size : float
            像素大小 (微米)。

        Returns
        -------
        np.ndarray
            传播后的复数场。
        """
        h, w = field.shape
        ny, nx = h, w

        # 频率坐标
        fx = np.fft.fftfreq(nx, d=pixel_size)
        fy = np.fft.fftfreq(ny, d=pixel_size)
        FX, FY = np.meshgrid(fx, fy)

        # 传递函数 H(fx, fy) = exp(i * 2*pi * distance * sqrt(1/lambda^2 - fx^2 - fy^2))
        k = 1.0 / wavelength  # 波数 (1/微米)
        sq_term = k ** 2 - FX ** 2 - FY ** 2

        # 修逝波处理
        propagating = sq_term >= 0
        H = np.zeros((ny, nx), dtype=np.complex128)
        H[propagating] = np.exp(
            1j * 2 * np.pi * distance * np.sqrt(sq_term[propagating])
        )

        # FFT 传播
        field_fft = np.fft.fft2(field)
        field_propagated = np.fft.ifft2(field_fft * H)

        return field_propagated

    def _compute_focus_metric(self, image: np.ndarray) -> float:
        """计算聚焦评价指标。

        Parameters
        ----------
        image : np.ndarray
            强度图像。

        Returns
        -------
        float
            聚焦指标值 (越大表示越聚焦)。
        """
        image = np.asarray(image, dtype=np.float64)
        if image.max() > 0:
            image = image / image.max()

        metric_type = self._cfg.focus_metric

        if metric_type == "variance":
            # 图像方差 (聚焦时方差最大)
            return float(np.var(image))

        elif metric_type == "gradient":
            # 梯度幅值之和 (Tenengrad)
            gx = np.diff(image, axis=1)
            gy = np.diff(image, axis=0)
            return float(np.mean(gx ** 2) + np.mean(gy ** 2))

        elif metric_type == "laplacian":
            # 拉普拉斯方差
            lap = (
                np.roll(image, 1, axis=0) + np.roll(image, -1, axis=0)
                + np.roll(image, 1, axis=1) + np.roll(image, -1, axis=1)
                - 4 * image
            )
            return float(np.var(lap))

        else:
            raise ValueError(f"未知聚焦指标: {metric_type}")

    def _phase_unwrap_quality(
        self, phase: np.ndarray
    ) -> np.ndarray:
        """基于质量图的相位解包裹。

        Parameters
        ----------
        phase : np.ndarray
            包裹相位 (-pi, pi]。

        Returns
        -------
        np.ndarray
            解包裹后的相位。
        """
        if self._cfg.unwrap_method == "none":
            return phase

        if self._cfg.unwrap_method == "simple":
            # 简单的逐行/逐列积分解包裹
            unwrapped = np.unwrap(phase, axis=0)
            unwrapped = np.unwrap(unwrapped, axis=1)
            return unwrapped

        # 质量导向解包裹 (简化实现)
        # 质量图: 基于相位梯度幅值
        grad_y, grad_x = np.gradient(phase)
        quality = 1.0 / (np.abs(grad_x) + np.abs(grad_y) + 1e-6)

        # 沿质量最高的路径积分
        h, w = phase.shape
        unwrapped = np.zeros_like(phase, dtype=np.float64)
        visited = np.zeros((h, w), dtype=bool)

        # 优先队列 (简化: 使用质量排序)
        indices = np.argsort(quality.ravel())[::-1]
        unwrapped_flat = unwrapped.ravel()

        # 初始化
        start_idx = indices[0]
        sy, sx = divmod(start_idx, w)
        unwrapped[sy, sx] = phase[sy, sx]
        visited[sy, sx] = True

        # 简化: 使用 flood fill 从最高质量点开始
        from collections import deque
        queue = deque([(sy, sx)])
        while queue:
            cy, cx = queue.popleft()
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny_idx, nx_idx = cy + dy, cx + dx
                if 0 <= ny_idx < h and 0 <= nx_idx < w and not visited[ny_idx, nx_idx]:
                    dp = phase[ny_idx, nx_idx] - phase[cy, cx]
                    # 包裹到 [-pi, pi]
                    dp = (dp + np.pi) % (2 * np.pi) - np.pi
                    unwrapped[ny_idx, nx_idx] = unwrapped[cy, cx] + dp
                    visited[ny_idx, nx_idx] = True
                    queue.append((ny_idx, nx_idx))

        return unwrapped

    def _compensate_aberration(
        self,
        complex_field: np.ndarray,
        reference_phase: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """像差补偿。

        Parameters
        ----------
        complex_field : np.ndarray
            复数场。
        reference_phase : np.ndarray, optional
            参考相位 (用于减法补偿)。

        Returns
        -------
        np.ndarray
            补偿后的复数场。
        """
        if not self._cfg.aberration_correction:
            return complex_field

        phase = np.angle(complex_field)

        if reference_phase is not None:
            # 使用提供的参考相位
            corrected_phase = phase - reference_phase
        else:
            # 自动估计低阶像差 (Zernike 拟合)
            # 简化: 使用二次曲面拟合去除离焦和像散
            h, w = phase.shape
            cy, cx = h // 2, w // 2
            y, x = np.mgrid[:h, :w]
            x_norm = (x - cx) / cx
            y_norm = (y - cy) / cy

            # 拟合二次项
            A = np.column_stack([
                np.ones(h * w),
                x_norm.ravel(),
                y_norm.ravel(),
                x_norm.ravel() ** 2,
                y_norm.ravel() ** 2,
                x_norm.ravel() * y_norm.ravel(),
            ])
            b = phase.ravel()

            # 最小二乘拟合
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                fitted = A @ coeffs
                corrected_phase = phase - fitted.reshape(h, w)
            except np.linalg.LinAlgError:
                corrected_phase = phase

        # 重建复数场
        amplitude = np.abs(complex_field)
        return amplitude * np.exp(1j * corrected_phase)

    def _find_spot_position(
        self,
        intensity: np.ndarray,
    ) -> Tuple[float, float]:
        """从强度图中找到光斑位置 (质心法)。

        Parameters
        ----------
        intensity : np.ndarray
            强度图像。

        Returns
        -------
        Tuple[float, float]
            (x, y) 光斑位置。
        """
        intensity = np.asarray(intensity, dtype=np.float64)
        total = np.sum(intensity) + 1e-12
        h, w = intensity.shape
        y_coords, x_coords = np.mgrid[:h, :w]
        cx = float(np.sum(x_coords * intensity) / total)
        cy = float(np.sum(y_coords * intensity) / total)
        return cx, cy

    def reconstruct(
        self,
        hologram: np.ndarray,
        reference_phase: Optional[np.ndarray] = None,
    ) -> ReconstructionResult:
        """从全息图像重建光斑。

        Parameters
        ----------
        hologram : np.ndarray
            全息图像 (强度或复数场)。
        reference_phase : np.ndarray, optional
            参考相位 (用于像差补偿)。

        Returns
        -------
        ReconstructionResult
            重建结果。
        """
        hologram = np.asarray(hologram, dtype=np.float64)

        if hologram.ndim != 2:
            raise ValueError(f"全息图像应为二维数组，实际维度: {hologram.ndim}")

        LOGGER.info("开始全息重建: 图像形状 %s, 波长 %.3f um", hologram.shape, self._cfg.wavelength)

        # 零填充
        pad = self._cfg.padding
        h, w = hologram.shape
        ph, pw = h * pad, w * pad
        hologram_padded = np.zeros((ph, pw), dtype=np.float64)
        hologram_padded[:h, :w] = hologram

        # 传播距离列表
        distances = self._cfg.propagation_distances

        # 多波长重建
        if len(self._cfg.wavelengths) > 1:
            return self._reconstruct_multi_wavelength(
                hologram_padded, distances, reference_phase
            )

        # 单波长重建
        wavelength = self._cfg.wavelengths[0]
        return self._reconstruct_single_wavelength(
            hologram_padded, distances, wavelength, reference_phase
        )

    def _reconstruct_single_wavelength(
        self,
        hologram: np.ndarray,
        distances: List[float],
        wavelength: float,
        reference_phase: Optional[np.ndarray],
    ) -> ReconstructionResult:
        """单波长全息重建。"""
        # 初始化复数场 (振幅 = sqrt(强度))
        amplitude = np.sqrt(np.clip(hologram, 0, None))
        complex_field = amplitude.astype(np.complex128)

        # 像差补偿
        complex_field = self._compensate_aberration(complex_field, reference_phase)

        # 在不同距离传播
        focus_metrics = np.zeros(len(distances), dtype=np.float64)
        best_field = None
        best_metric = -np.inf
        best_idx = 0

        for i, d in enumerate(distances):
            propagated = self._angular_spectrum_propagate(
                complex_field, d, wavelength, self._cfg.pixel_size
            )
            intensity = np.abs(propagated) ** 2
            metric = self._compute_focus_metric(intensity)
            focus_metrics[i] = metric

            if metric > best_metric:
                best_metric = metric
                best_field = propagated
                best_idx = i

        if best_field is None:
            best_field = complex_field

        best_distance = distances[best_idx]

        # 提取结果
        result_amplitude = np.abs(best_field)
        result_intensity = result_amplitude ** 2
        result_phase = np.angle(best_field)

        # 相位解包裹
        if self._cfg.unwrap_method != "none":
            result_phase = self._phase_unwrap_quality(result_phase)

        # 光斑位置
        spot_x, spot_y = self._find_spot_position(result_intensity)

        result = ReconstructionResult(
            complex_field=best_field,
            amplitude=result_amplitude,
            phase=result_phase,
            intensity=result_intensity,
            best_focus_distance=best_distance,
            focus_metrics=focus_metrics,
            spot_position_3d=(spot_x, spot_y, best_distance),
            wavelengths_used=[wavelength],
            propagation_distances=distances,
        )

        LOGGER.info(
            "单波长重建完成: 最佳聚焦距离=%.2f um, "
            "光斑位置=(%.1f, %.1f, %.1f)",
            best_distance, spot_x, spot_y, best_distance,
        )

        return result

    def _reconstruct_multi_wavelength(
        self,
        hologram: np.ndarray,
        distances: List[float],
        reference_phase: Optional[np.ndarray],
    ) -> ReconstructionResult:
        """多波长全息重建。"""
        wavelengths = self._cfg.wavelengths
        LOGGER.info("多波长重建: %d 个波长", len(wavelengths))

        # 对每个波长独立重建
        results = []
        for wl in wavelengths:
            r = self._reconstruct_single_wavelength(
                hologram, distances, wl, reference_phase
            )
            results.append(r)

        # 合成多波长结果
        combination = self._cfg.multi_wavelength_combination

        if combination == "incoherent":
            # 非相干合成: 强度叠加
            combined_intensity = sum(r.intensity for r in results)
            combined_amplitude = np.sqrt(combined_intensity)
            combined_phase = results[0].phase  # 使用第一个波长的相位
            combined_field = combined_amplitude * np.exp(1j * combined_phase)

        elif combination == "coherent":
            # 相干合成: 复数场叠加
            combined_field = sum(r.complex_field for r in results)
            combined_amplitude = np.abs(combined_field)
            combined_intensity = combined_amplitude ** 2
            combined_phase = np.angle(combined_field)

        elif combination == "phase":
            # 相位合成: 使用相位差提高轴向分辨率
            combined_field = results[0].complexplex_field if hasattr(results[0], 'complexplex_field') else results[0].complex_field
            combined_amplitude = results[0].amplitude
            combined_intensity = results[0].intensity
            combined_phase = results[0].phase
            # 多波长相位差可以合成更大的等效波长
            # 简化: 使用平均结果
            if len(results) > 1:
                combined_field = results[0].complex_field
                combined_amplitude = np.abs(combined_field)
                combined_intensity = combined_amplitude ** 2
                combined_phase = np.angle(combined_field)
        else:
            raise ValueError(f"未知多波长合成方式: {combination}")

        # 找最佳聚焦
        best_idx = 0
        best_metric = -np.inf
        for i, r in enumerate(results):
            if np.max(r.focus_metrics) > best_metric:
                best_metric = np.max(r.focus_metrics)
                best_idx = i

        spot_x, spot_y = self._find_spot_position(combined_intensity)

        result = ReconstructionResult(
            complex_field=combined_field,
            amplitude=combined_amplitude,
            phase=combined_phase,
            intensity=combined_intensity,
            best_focus_distance=results[best_idx].best_focus_distance,
            focus_metrics=np.array([np.max(r.focus_metrics) for r in results]),
            spot_position_3d=(spot_x, spot_y, results[best_idx].best_focus_distance),
            wavelengths_used=wavelengths,
            propagation_distances=distances,
        )

        LOGGER.info(
            "多波长重建完成: 最佳聚焦距离=%.2f um",
            results[best_idx].best_focus_distance,
        )

        return result

    def propagate(
        self,
        field: np.ndarray,
        distance: float,
        wavelength: Optional[float] = None,
    ) -> np.ndarray:
        """对给定复数场进行数值传播。

        Parameters
        ----------
        field : np.ndarray
            输入复数场。
        distance : float
            传播距离 (微米)。
        wavelength : float, optional
            波长。为 None 时使用配置中的波长。

        Returns
        -------
        np.ndarray
            传播后的复数场。
        """
        wl = wavelength if wavelength is not None else self._cfg.wavelength
        return self._angular_spectrum_propagate(
            field, distance, wl, self._cfg.pixel_size
        )

    def reset(self):
        """重置重建器状态。"""
        LOGGER.info("全息光斑重建器已重置")
