"""
相位恢复分析器 (PhaseRetrievalAnalyzer)

灵感来源:
- HCIPy (https://github.com/ehpor/hcipy) — 高对比度成像仿真中的相位恢复
- AOtools (https://github.com/AOtools/aotools) — 自适应光学相位恢复工具
- Gerchberg-Saxton 算法 — 经典迭代傅里叶相位恢复方法 (Gerchberg & Saxton, 1972)

算法原理:
- Gerchberg-Saxton (GS) 迭代相位恢复 — 在光瞳面和像面之间交替约束，
  通过 FFT/IFFT 迭代逼近真实相位分布
- Error Reduction (ER) — 经典 GS 算法，在光瞳面施加振幅约束
- Hybrid Input-Output (HIO) — Fienup 改进算法，引入反馈参数加速收敛
- Phase Unwrapping — 1D 相位解包裹，消除 2π 跳变
- Wavefront RMS — 从恢复的相位计算波前均方根误差
- Strehl Estimation — Maréchal 近似估计 Strehl 比

功能:
- 从光斑图像估计波前相位分布 (无需物理波前传感器)
- 支持 GS-ER 和 GS-HIO 两种迭代变体
- 计算波前 RMS 和 Strehl 比估计
- 提供像差模式分解 (类 Zernike 模式振幅)
- 收敛性监控与残差能量跟踪

依赖: numpy, opencv-python (仅用于图像预处理)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger("SpotZoom.PhaseRetrievalAnalyzer")

# 模块默认禁用标志
phase_retrieval_enabled: bool = False


@dataclass
class PhaseRetrievalReport:
    """相位恢复分析报告。"""
    phase_map: np.ndarray  # 恢复的相位图 (2D, 弧度)
    wavefront_rms: float  # 波前 RMS 误差 (弧度)
    strehl_estimate: float  # Strehl 比估计 (Maréchal 近似)
    convergence_history: List[float]  # 每次迭代的残差能量
    iteration_count: int  # 实际迭代次数
    is_converged: bool  # 是否收敛
    aberration_breakdown: Dict[str, float]  # 类 Zernike 模式振幅分解


class PhaseRetrievalAnalyzer:
    """相位恢复分析器。

    使用 Gerchberg-Saxton (GS) 迭代算法从光斑图像中恢复波前相位分布，
    实现无波前传感器的波前感知。

    GS 算法原理:
        1. 初始化随机相位 (或零相位)
        2. 前向传播: 相位 -> FFT -> 复振幅像面
        3. 像面约束: 用实测振幅替换计算振幅
        4. 逆向传播: IFFT -> 光瞳面复振幅
        5. 光瞳约束: 施加光瞳掩模 (孔径约束)
        6. 重复 2-5 直到收敛

    Parameters
    ----------
    grid_size : int
        内部计算网格尺寸 (像素)。建议使用 2 的幂次。
    max_iterations : int
        最大 GS 迭代次数。
    convergence_threshold : float
        收敛判定阈值 (残差能量变化率)。
    algorithm : str
        算法类型: "er" (Error Reduction) 或 "hio" (Hybrid Input-Output)。
    hio_beta : float
        HIO 算法的反馈参数 beta (仅 algorithm="hio" 时有效)。
    wavelength_m : float
        波长 (米)，用于物理单位换算。
    pixel_pitch_m : float
        像素间距 (米)，用于物理单位换算。
    """

    def __init__(
        self,
        grid_size: int = 128,
        max_iterations: int = 50,
        convergence_threshold: float = 1e-4,
        algorithm: str = "er",
        hio_beta: float = 0.9,
        wavelength_m: float = 632.8e-9,
        pixel_pitch_m: float = 5.0e-6,
    ):
        self.grid_size = int(grid_size)
        self.max_iterations = int(max_iterations)
        self.convergence_threshold = float(convergence_threshold)
        self.algorithm = algorithm.lower().strip()
        self.hio_beta = float(hio_beta)
        self.wavelength_m = float(wavelength_m)
        self.pixel_pitch_m = float(pixel_pitch_m)

        if self.algorithm not in ("er", "hio"):
            raise ValueError(f"algorithm 必须为 'er' 或 'hio'，当前值: {self.algorithm}")
        if self.grid_size < 16:
            raise ValueError(f"grid_size 必须 >= 16，当前值: {self.grid_size}")

        # 预计算光瞳掩模 (圆形孔径)
        self._pupil_mask = self._create_pupil_mask(self.grid_size)
        self._pupil_mask_float = self._pupil_mask.astype(np.float64)

        # 内部状态
        self._last_phase: Optional[np.ndarray] = None
        self._convergence_history: List[float] = []

        LOGGER.info(
            "PhaseRetrievalAnalyzer: 初始化完成 (grid=%d, max_iter=%d, algo=%s)",
            self.grid_size, self.max_iterations, self.algorithm,
        )

    def analyze(
        self,
        image: np.ndarray,
        target_position: Optional[Tuple[float, float]] = None,
    ) -> PhaseRetrievalReport:
        """主分析入口: 从光斑图像恢复波前相位。

        Parameters
        ----------
        image : np.ndarray
            输入光斑图像 (灰度或 BGR)。
        target_position : Tuple[float, float] or None
            目标位置 (cx, cy)。为 None 时使用图像中心。

        Returns
        -------
        PhaseRetrievalReport
            相位恢复分析报告。
        """
        try:
            # 预处理图像
            processed = self._preprocess_image(image, target_position)

            # 执行 GS 相位恢复
            phase_map, residual = self.retrieve_phase(processed, self.max_iterations)

            # 计算波前 RMS
            wavefront_rms = self._estimate_wavefront_rms(phase_map)

            # Strehl 比估计 (Maréchal 近似)
            strehl = float(np.exp(-(wavefront_rms) ** 2))

            # 像差模式分解
            aberration_breakdown = self._decompose_aberrations(phase_map)

            # 收敛判定
            is_converged = False
            if len(self._convergence_history) >= 10:
                recent = self._convergence_history[-10:]
                rate = abs(recent[-1] - recent[0]) / max(abs(recent[0]), 1e-12)
                is_converged = rate < self.convergence_threshold

            self._last_phase = phase_map

            LOGGER.info(
                "PhaseRetrievalAnalyzer: 分析完成 (RMS=%.4f rad, Strehl=%.4f, "
                "迭代=%d, 收敛=%s)",
                wavefront_rms, strehl, len(self._convergence_history),
                "是" if is_converged else "否",
            )

            return PhaseRetrievalReport(
                phase_map=phase_map,
                wavefront_rms=round(wavefront_rms, 6),
                strehl_estimate=round(strehl, 6),
                convergence_history=list(self._convergence_history),
                iteration_count=len(self._convergence_history),
                is_converged=is_converged,
                aberration_breakdown=aberration_breakdown,
            )

        except Exception as e:
            LOGGER.error("PhaseRetrievalAnalyzer: 分析失败: %s", e)
            return self._empty_report()

    def retrieve_phase(
        self,
        image: np.ndarray,
        max_iterations: int = 50,
    ) -> Tuple[np.ndarray, float]:
        """核心 GS 迭代相位恢复算法。

        Parameters
        ----------
        image : np.ndarray
            预处理后的光斑图像 (2D, 已归一化)。
        max_iterations : int
            最大迭代次数。

        Returns
        -------
        Tuple[np.ndarray, float]
            (恢复的相位图, 最终残差能量)。
        """
        max_iterations = min(int(max_iterations), self.max_iterations)
        size = self.grid_size

        # 确保图像尺寸匹配
        if image.shape[0] != size or image.shape[1] != size:
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)

        image = image.astype(np.float64)
        total = image.sum()
        if total < 1e-12:
            return np.zeros((size, size), dtype=np.float64), 1.0

        # 目标振幅 (实测光斑的平方根)
        target_amplitude = np.sqrt(image / total)

        # 初始化: 随机相位 + 光瞳约束
        np.random.seed(42)  # 可重复性
        phase = np.random.uniform(-np.pi, np.pi, (size, size)).astype(np.float64)
        phase *= self._pupil_mask_float

        # HIO 需要保存上一次光瞳面相位
        prev_pupil_phase = phase.copy()

        self._convergence_history = []

        for iteration in range(max_iterations):
            # 前向传播: 光瞳面 -> 像面
            pupil_field = self._pupil_mask_float * np.exp(1j * phase)
            image_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_field)))

            # 像面约束: 替换振幅，保留相位
            computed_amplitude = np.abs(image_field)
            new_image_field = target_amplitude * np.exp(1j * np.angle(image_field))

            # 逆向传播: 像面 -> 光瞳面
            new_pupil_field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(new_image_field)))

            # 计算残差能量
            residual = float(np.sum(
                (np.abs(new_pupil_field) - self._pupil_mask_float) ** 2
            ))
            self._convergence_history.append(residual)

            # 光瞳约束
            new_amplitude = np.abs(new_pupil_field)
            new_phase = np.angle(new_pupil_field)

            if self.algorithm == "hio":
                # Hybrid Input-Output
                outside_mask = self._pupil_mask_float < 0.5
                phase = np.where(
                    outside_mask,
                    prev_pupil_phase - self.hio_beta * new_amplitude * np.exp(1j * new_phase),
                    new_phase,
                ).real
                prev_pupil_phase = phase.copy()
            else:
                # Error Reduction
                phase = new_phase * self._pupil_mask_float

            # 提前收敛检查
            if iteration >= 10 and len(self._convergence_history) >= 10:
                recent = self._convergence_history[-10:]
                rate = abs(recent[-1] - recent[0]) / max(abs(recent[0]), 1e-12)
                if rate < self.convergence_threshold:
                    LOGGER.debug(
                        "PhaseRetrievalAnalyzer: 在第 %d 次迭代收敛", iteration + 1
                    )
                    break

        final_residual = self._convergence_history[-1] if self._convergence_history else 1.0
        return phase, final_residual

    def _compute_psf(self, phase: np.ndarray, pupil_mask: np.ndarray) -> np.ndarray:
        """从相位计算 PSF (点扩散函数)。

        Parameters
        ----------
        phase : np.ndarray
            波前相位 (2D, 弧度)。
        pupil_mask : np.ndarray
            光瞳掩模。

        Returns
        -------
        np.ndarray
            PSF 强度分布。
        """
        pupil_field = pupil_mask.astype(np.float64) * np.exp(1j * phase)
        image_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_field)))
        psf = np.abs(image_field) ** 2
        return psf

    def _estimate_wavefront_rms(self, phase: np.ndarray) -> float:
        """从恢复的相位计算波前 RMS 误差。

        仅在光瞳孔径内计算，排除活塞项 (均值)。

        Parameters
        ----------
        phase : np.ndarray
            恢复的相位图。

        Returns
        -------
        float
            波前 RMS 误差 (弧度)。
        """
        mask = self._pupil_mask > 0
        if not np.any(mask):
            return 0.0

        phase_values = phase[mask]
        # 去除活塞 (均值)
        phase_values = phase_values - np.mean(phase_values)
        rms = float(np.sqrt(np.mean(phase_values ** 2)))
        return rms

    def _unwrap_phase_1d(self, phase_profile: np.ndarray) -> np.ndarray:
        """简单 1D 相位解包裹。

        通过检测相邻像素间的 2π 跳变来消除包裹效应。

        Parameters
        ----------
        phase_profile : np.ndarray
            1D 包裹相位序列。

        Returns
        -------
        np.ndarray
            解包裹后的相位序列。
        """
        if phase_profile.ndim != 1 or len(phase_profile) < 2:
            return phase_profile.copy()

        unwrapped = phase_profile.copy().astype(np.float64)
        for i in range(1, len(unwrapped)):
            diff = unwrapped[i] - unwrapped[i - 1]
            # 检测大于 π 的跳变
            while diff > np.pi:
                unwrapped[i] -= 2.0 * np.pi
                diff = unwrapped[i] - unwrapped[i - 1]
            while diff < -np.pi:
                unwrapped[i] += 2.0 * np.pi
                diff = unwrapped[i] - unwrapped[i - 1]

        return unwrapped

    def _decompose_aberrations(self, phase: np.ndarray) -> Dict[str, float]:
        """将恢复的相位分解为类 Zernike 模式振幅。

        使用简单的矩方法估计低阶像差模式的振幅:
        - Tip/Tilt: 一阶矩 (质心偏移)
        - Defocus: 径向二阶矩
        - Astigmatism: 椭圆度
        - Coma: 非对称性

        Parameters
        ----------
        phase : np.ndarray
            恢复的相位图。

        Returns
        -------
        Dict[str, float]
            模式名称到振幅的映射。
        """
        mask = self._pupil_mask > 0
        if not np.any(mask) or phase.size == 0:
            return {}

        phase_values = phase.copy().astype(np.float64)
        phase_values[~mask] = 0.0

        # 去除活塞
        mean_phase = np.mean(phase_values[mask])
        phase_centered = phase_values - mean_phase

        size = phase.shape[0]
        cy, cx = size / 2.0, size / 2.0
        yy, xx = np.mgrid[:size, :size]
        dx = (xx - cx) / (size / 2.0)
        dy = (yy - cy) / (size / 2.0)

        # 归一化坐标
        dx_masked = dx[mask]
        dy_masked = dy[mask]
        phase_masked = phase_centered[mask]

        # Tip (X 倾斜): <phase * x>
        tip = float(np.mean(phase_masked * dx_masked))

        # Tilt (Y 倾斜): <phase * y>
        tilt = float(np.mean(phase_masked * dy_masked))

        # Defocus (离焦): <phase * r^2>
        r2 = dx_masked ** 2 + dy_masked ** 2
        defocus = float(np.mean(phase_masked * r2))

        # Astigmatism (像散): <phase * (x^2 - y^2)>
        astig_0 = float(np.mean(phase_masked * (dx_masked ** 2 - dy_masked ** 2)))

        # Astigmatism 45: <phase * 2*x*y>
        astig_45 = float(np.mean(phase_masked * 2.0 * dx_masked * dy_masked))

        # Coma X: <phase * (3r^2 - 2) * x>
        coma_x = float(np.mean(phase_masked * (3.0 * r2 - 2.0) * dx_masked))

        # Coma Y: <phase * (3r^2 - 2) * y>
        coma_y = float(np.mean(phase_masked * (3.0 * r2 - 2.0) * dy_masked))

        # Spherical (球差): <phase * (6r^4 - 6r^2 + 1)>
        spherical = float(np.mean(phase_masked * (6.0 * r2 ** 2 - 6.0 * r2 + 1.0)))

        breakdown = {
            "tip_x": round(tip, 6),
            "tilt_y": round(tilt, 6),
            "defocus": round(defocus, 6),
            "astigmatism_0": round(astig_0, 6),
            "astigmatism_45": round(astig_45, 6),
            "coma_x": round(coma_x, 6),
            "coma_y": round(coma_y, 6),
            "spherical": round(spherical, 6),
        }

        return breakdown

    def _create_pupil_mask(self, size: int) -> np.ndarray:
        """创建圆形光瞳掩模。

        Parameters
        ----------
        size : int
            网格尺寸。

        Returns
        -------
        np.ndarray
            二值光瞳掩模 (uint8)。
        """
        yy, xx = np.mgrid[:size, :size]
        cy, cx = size / 2.0, size / 2.0
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (size / 2.0)
        mask = (r <= 1.0).astype(np.uint8)
        return mask

    def _preprocess_image(
        self,
        image: np.ndarray,
        target_position: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """预处理输入图像。

        Parameters
        ----------
        image : np.ndarray
            输入图像。
        target_position : Tuple[float, float] or None
            目标位置。

        Returns
        -------
        np.ndarray
            预处理后的图像 (2D float64, grid_size x grid_size)。
        """
        # 转灰度
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
        else:
            gray = image.astype(np.float64)

        h, w = gray.shape

        # 裁剪中心区域
        crop_size = min(h, w)
        if crop_size < 8:
            raise ValueError(f"图像尺寸过小: {h}x{w}")

        y_start = (h - crop_size) // 2
        x_start = (w - crop_size) // 2
        crop = gray[y_start:y_start + crop_size, x_start:x_start + crop_size]

        # 调整到网格尺寸
        resized = cv2.resize(crop, (self.grid_size, self.grid_size),
                             interpolation=cv2.INTER_AREA)

        # 背景扣除
        bg = np.percentile(resized, 10)
        signal = np.maximum(resized - bg, 0.0)

        # 归一化
        total = signal.sum()
        if total > 1e-12:
            signal = signal / total

        return signal

    def _empty_report(self) -> PhaseRetrievalReport:
        """返回空报告。"""
        return PhaseRetrievalReport(
            phase_map=np.zeros((self.grid_size, self.grid_size), dtype=np.float64),
            wavefront_rms=0.0,
            strehl_estimate=0.0,
            convergence_history=[],
            iteration_count=0,
            is_converged=False,
            aberration_breakdown={},
        )

    def reset(self) -> None:
        """重置分析器内部状态。"""
        self._last_phase = None
        self._convergence_history = []

        LOGGER.info("PhaseRetrievalAnalyzer: 分析器已重置")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    analyzer = PhaseRetrievalAnalyzer(
        grid_size=128,
        max_iterations=50,
        algorithm="er",
    )

    print("=== 相位恢复分析器测试 ===\n")

    # 生成模拟光斑图像 (含像差的 PSF)
    size = 128
    yy, xx = np.mgrid[:size, :size]
    cy, cx = size / 2.0, size / 2.0
    dx = (xx - cx) / (size / 2.0)
    dy = (yy - cy) / (size / 2.0)
    r2 = dx ** 2 + dy ** 2
    mask = r2 <= 1.0

    # 模拟含离焦和像散的波前
    phase = 0.5 * (2.0 * r2 - 1.0) + 0.3 * (dx ** 2 - dy ** 2)
    phase *= mask

    # 计算 PSF
    pupil_field = mask.astype(np.float64) * np.exp(1j * phase)
    image_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_field)))
    psf = np.abs(image_field) ** 2
    psf = psf / psf.max() * 255.0
    psf_uint8 = np.clip(psf, 0, 255).astype(np.uint8)

    print(f"模拟 PSF 图像尺寸: {psf_uint8.shape}")
    print(f"PSF 峰值: {psf.max():.2f}")

    # 执行相位恢复
    report = analyzer.analyze(psf_uint8)

    print(f"\n--- 相位恢复结果 ---")
    print(f"波前 RMS: {report.wavefront_rms:.4f} rad")
    print(f"Strehl 比估计: {report.strehl_estimate:.4f}")
    print(f"迭代次数: {report.iteration_count}")
    print(f"是否收敛: {'是' if report.is_converged else '否'}")
    print(f"最终残差: {report.convergence_history[-1]:.6f}" if report.convergence_history else "无收敛历史")

    print(f"\n--- 像差分解 ---")
    for mode, amplitude in report.aberration_breakdown.items():
        print(f"  {mode}: {amplitude:+.4f}")

    # 测试 HIO 算法
    print(f"\n--- HIO 算法测试 ---")
    analyzer_hio = PhaseRetrievalAnalyzer(
        grid_size=128,
        max_iterations=50,
        algorithm="hio",
        hio_beta=0.9,
    )
    report_hio = analyzer_hio.analyze(psf_uint8)
    print(f"HIO 波前 RMS: {report_hio.wavefront_rms:.4f} rad")
    print(f"HIO Strehl: {report_hio.strehl_estimate:.4f}")
    print(f"HIO 迭代次数: {report_hio.iteration_count}")
    print(f"HIO 收敛: {'是' if report_hio.is_converged else '否'}")

    # 测试 1D 相位解包裹
    print(f"\n--- 1D 相位解包裹测试 ---")
    wrapped = np.array([0.1, 0.5, 1.0, 1.5, -2.5, -2.0, -1.5, 2.0, 2.5, 3.0])
    unwrapped = analyzer._unwrap_phase_1d(wrapped)
    print(f"包裹相位: {wrapped}")
    print(f"解包裹:   {unwrapped}")

    print("\n测试完成")
