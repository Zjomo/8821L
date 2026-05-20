"""
大气湍流模拟器 (AtmosphericTurbulenceSimulator)

灵感来源:
- HCIPy (https://github.com/ehpor/hcipy) — 大气湍流相位屏生成
- AOtools (https://github.com/AOtools/aotools) — Kolmogorov/von Kármán 湍流模型
- Roddier (1999) — "Adaptive Optics in Astronomy" 湍流理论

算法原理:
- Kolmogorov Turbulence — 大气湍流的统计模型，相位功率谱密度正比于 f^(-11/3)
- Von Kármán Model — 改进模型，引入外尺度 L0，避免低频发散:
    PSD(f) = 0.023 * r0^(-5/3) * (f^2 + L0^(-2))^(-11/3)
- FFT Phase Screen Generation — 通过对功率谱密度采样并做 IFFT 生成相位屏
- Taylor Frozen Flow — 泰勒冻结流假设: 湍流在短时间内随风向平移而不演化
- Beam Wander & Spread — 光束漂移 (整体偏移) 和展宽 (PSF 扩大) 效应
- Fried Parameter (r0) — 大气相干长度，描述湍流强度:
    D/r0 越大，湍流越强

功能:
- 生成 Kolmogorov/von Kármán 相位屏
- 时间演化 (Taylor 冻结流)
- 模拟湍流对光斑图像的退化效果
- 光束漂移模拟
- 从图像序列估计湍流强度
- 生成完整测试序列

依赖: numpy, opencv-python (仅用于图像处理)
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger("SpotZoom.AtmosphericTurbulenceSimulator")

# 模块默认禁用标志
turbulence_sim_enabled: bool = False


@dataclass
class TurbulenceProfile:
    """湍流参数配置。"""
    fried_parameter: float  # Fried 参数 r0 (米)
    outer_scale: float  # 外尺度 L0 (米), 0 表示纯 Kolmogorov
    wind_speed: float  # 风速 (米/秒)
    cn_squared: float  # 大气折射率结构常数 Cn² (m^(-2/3))
    wavelength: float  # 波长 (米)


@dataclass
class TurbulenceStatistics:
    """湍流统计信息。"""
    mean_strehl: float  # 平均 Strehl 比
    mean_rms: float  # 平均波前 RMS (弧度)
    wander_rms: float  # 光束漂移 RMS (像素)
    scintillation_index: float  # 闪烁指数


class AtmosphericTurbulenceSimulator:
    """大气湍流模拟器。

    使用 Kolmogorov/von Kármán 模型生成大气湍流相位屏，
    模拟湍流对光斑图像的退化效果，用于系统测试和鲁棒性验证。

    Parameters
    ----------
    fried_parameter : float
        Fried 参数 r0 (米)。典型值 0.05~0.20m。
        r0 越小表示湍流越强。
    outer_scale : float or None
        外尺度 L0 (米)。为 None 或 0 时使用纯 Kolmogorov 模型。
        典型值 10~100m。
    wind_speed : float
        风速 (米/秒)。用于 Taylor 冻结流时间演化。
    wavelength : float
        波长 (米)。默认 632.8nm (HeNe 激光)。
    cn_squared : float or None
        大气折射率结构常数 Cn² (m^(-2/3))。
        为 None 时从 r0 和波长自动计算。
    screen_size : int
        相位屏尺寸 (像素)。建议使用 2 的幂次。
    subharmonics : bool
        是否添加子谐波以改善低频成分。
    seed : int or None
        随机种子。为 None 时使用系统随机数。
    """

    def __init__(
        self,
        fried_parameter: float = 0.10,
        outer_scale: Optional[float] = None,
        wind_speed: float = 10.0,
        wavelength: float = 632.8e-9,
        cn_squared: Optional[float] = None,
        screen_size: int = 256,
        subharmonics: bool = True,
        seed: Optional[int] = None,
    ):
        self.fried_parameter = float(fried_parameter)
        self.outer_scale = float(outer_scale) if outer_scale else 0.0
        self.wind_speed = float(wind_speed)
        self.wavelength = float(wavelength)
        self.screen_size = int(screen_size)
        self.subharmonics = subharmonics

        if self.fried_parameter <= 0:
            raise ValueError(f"fried_parameter 必须 > 0，当前值: {self.fried_parameter}")

        # 从 r0 计算 Cn² (如果未提供)
        if cn_squared is not None:
            self.cn_squared = float(cn_squared)
        else:
            # Cn² = 0.023 * r0^(-5/3) * (2*pi/lambda)^(2) * lambda^(-1/3)
            # 简化: 使用标准关系
            self.cn_squared = 0.023 * self.fried_parameter ** (-5.0 / 3.0)

        # 随机数生成器
        self._rng = np.random.RandomState(seed)

        # 当前相位屏状态
        self._current_phase_screen: Optional[np.ndarray] = None
        self._phase_screen_offset_x: float = 0.0
        self._phase_screen_offset_y: float = 0.0

        # 统计信息
        self._strehl_history: List[float] = []
        self._rms_history: List[float] = []
        self._wander_history: List[Tuple[float, float]] = []

        LOGGER.info(
            "AtmosphericTurbulenceSimulator: 初始化完成 "
            "(r0=%.3fm, L0=%.1fm, wind=%.1fm/s, screen=%d)",
            self.fried_parameter, self.outer_scale, self.wind_speed,
            self.screen_size,
        )

    @property
    def profile(self) -> TurbulenceProfile:
        """获取当前湍流参数配置。"""
        return TurbulenceProfile(
            fried_parameter=self.fried_parameter,
            outer_scale=self.outer_scale,
            wind_speed=self.wind_speed,
            cn_squared=self.cn_squared,
            wavelength=self.wavelength,
        )

    def generate_phase_screen(
        self,
        size: int,
        fried_parameter: Optional[float] = None,
        outer_scale: Optional[float] = None,
    ) -> np.ndarray:
        """生成 Kolmogorov/von Kármán 相位屏。

        使用 FFT 方法: 对功率谱密度采样并做 IFFT 生成随机相位屏。

        Parameters
        ----------
        size : int
            相位屏尺寸 (像素)。
        fried_parameter : float or None
            覆盖默认 r0 值。
        outer_scale : float or None
            覆盖默认外尺度值。

        Returns
        -------
        np.ndarray
            相位屏 (2D, 弧度)。
        """
        r0 = fried_parameter if fried_parameter is not None else self.fried_parameter
        L0 = outer_scale if outer_scale is not None else self.outer_scale
        size = int(size)

        if size < 4:
            raise ValueError(f"size 必须 >= 4，当前值: {size}")

        # 频率网格
        df = 1.0 / size  # 频率间隔
        fx = np.fft.fftfreq(size, d=df)
        fy = np.fft.fftfreq(size, d=df)
        FX, FY = np.meshgrid(fx, fy)
        f_sq = FX ** 2 + FY ** 2

        # 避免零频
        f_sq[0, 0] = 1e-10

        # 功率谱密度
        if L0 > 0:
            # Von Kármán 模型
            # PSD(f) = 0.023 * r0^(-5/3) * exp(-f²/L0²) / (f² + 1/L0²)^(11/6)
            L0_inv_sq = 1.0 / (L0 ** 2)
            psd = (0.023 * r0 ** (-5.0 / 3.0) *
                   np.exp(-f_sq * L0 ** 2) /
                   (f_sq + L0_inv_sq) ** (11.0 / 6.0))
        else:
            # 纯 Kolmogorov 模型
            # PSD(f) = 0.023 * r0^(-5/3) * f^(-11/3)
            psd = 0.023 * r0 ** (-5.0 / 3.0) * f_sq ** (-11.0 / 6.0)

        # 零频设为 0 (去除 piston)
        psd[0, 0] = 0.0

        # 生成随机相位屏
        # amplitude = sqrt(PSD * df^2)
        amplitude = np.sqrt(np.maximum(psd, 0.0)) * df
        random_phase = self._rng.uniform(0, 2.0 * np.pi, (size, size))
        cn = amplitude * np.exp(1j * random_phase)

        # IFFT 得到相位屏
        phase_screen = np.real(np.fft.ifft2(np.fft.ifftshift(cn))) * size * size

        # 添加子谐波 (改善低频)
        if self.subharmonics:
            phase_screen = self._add_subharmonics(phase_screen, r0, L0, size)

        # 归一化到弧度
        phase_screen = phase_screen * (2.0 * np.pi / self.wavelength)

        self._current_phase_screen = phase_screen
        self._phase_screen_offset_x = 0.0
        self._phase_screen_offset_y = 0.0

        return phase_screen

    def evolve_phase_screen(
        self,
        dt: float,
        wind_speed: Optional[float] = None,
    ) -> np.ndarray:
        """使用 Taylor 冻结流假设演化相位屏。

        湍流图案以风速平移，不发生形变。

        Parameters
        ----------
        dt : float
            时间步长 (秒)。
        wind_speed : float or None
            覆盖默认风速。风向默认为水平 (+x 方向)。

        Returns
        -------
        np.ndarray
            演化后的相位屏。
        """
        if self._current_phase_screen is None:
            raise RuntimeError("请先调用 generate_phase_screen() 生成初始相位屏")

        ws = wind_speed if wind_speed is not None else self.wind_speed

        # 计算平移量 (像素)
        # 假设相位屏的物理尺寸与屏幕尺寸成正比
        # 简化: 1 像素 = 1 单位距离
        shift_pixels = ws * dt

        self._phase_screen_offset_x += shift_pixels

        # 使用循环平移
        shift_int = int(round(self._phase_screen_offset_x))
        if shift_int != 0:
            self._current_phase_screen = np.roll(
                self._current_phase_screen, -shift_int, axis=1
            )
            self._phase_screen_offset_x -= shift_int

        return self._current_phase_screen

    def apply_to_image(
        self,
        image: np.ndarray,
        strength: float = 1.0,
    ) -> np.ndarray:
        """将湍流效果应用到光斑图像。

        模拟光束漂移 (整体偏移) 和光束展宽 (模糊)。

        Parameters
        ----------
        image : np.ndarray
            输入光斑图像。
        strength : float
            湍流强度因子 (0~2)。1.0=正常强度。

        Returns
        -------
        np.ndarray
            退化后的图像。
        """
        image = image.astype(np.float64)
        h, w = image.shape[:2]

        # 1. 光束漂移 (随机偏移)
        wander_sigma = strength * self.fried_parameter * 50.0  # 缩放到像素
        wander_sigma = max(wander_sigma, 0.1)
        dx = self._rng.normal(0, wander_sigma)
        dy = self._rng.normal(0, wander_sigma)

        # 记录漂移
        self._wander_history.append((dx, dy))

        # 2. 光束展宽 (高斯模糊)
        blur_sigma = strength * self.fried_parameter * 20.0
        blur_sigma = max(blur_sigma, 0.1)
        blur_sigma = min(blur_sigma, min(h, w) * 0.2)

        # 应用模糊
        degraded = cv2.GaussianBlur(image, (0, 0), blur_sigma)

        # 应用漂移 (仿射变换)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        degraded = cv2.warpAffine(degraded, M, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)

        # 3. 闪烁效应 (强度波动)
        scintillation = 1.0 + self._rng.normal(0, 0.05 * strength)
        degraded = np.clip(degraded * scintillation, 0, None)

        return degraded

    def simulate_beam_wander(
        self,
        num_frames: int,
        r0: Optional[float] = None,
        exposure_time: float = 0.01,
    ) -> List[Tuple[float, float]]:
        """模拟光束漂移轨迹。

        Parameters
        ----------
        num_frames : int
            帧数。
        r0 : float or None
            覆盖默认 r0。
        exposure_time : float
            曝光时间 (秒)。

        Returns
        -------
        List[Tuple[float, float]]
            每帧的漂移量 (dx, dy) 列表 (像素)。
        """
        r0 = r0 if r0 is not None else self.fried_parameter

        # 光束漂移方差正比于 (D/r0)^(5/3) * (lambda/D)
        # 简化模型: 漂移标准差与 r0 成反比
        wander_sigma = 1.0 / max(r0, 0.01) * 5.0  # 像素

        wander: List[Tuple[float, float]] = []
        x, y = 0.0, 0.0

        for _ in range(num_frames):
            # 随机游走 + 均值回归
            x += self._rng.normal(0, wander_sigma * np.sqrt(exposure_time))
            y += self._rng.normal(0, wander_sigma * np.sqrt(exposure_time))
            # 均值回归 (防止漂移过大)
            x *= 0.99
            y *= 0.99
            wander.append((round(x, 4), round(y, 4)))

        return wander

    def estimate_turbulence_strength(
        self,
        image_sequence: List[np.ndarray],
    ) -> Dict[str, float]:
        """从图像序列估计湍流强度。

        通过分析 PSF 展宽和光束漂移来估计 r0。

        Parameters
        ----------
        image_sequence : List[np.ndarray]
            光斑图像序列。

        Returns
        -------
        Dict[str, float]
            估计结果字典:
            - 'estimated_r0': 估计的 Fried 参数 (米)
            - 'wander_rms': 漂移 RMS (像素)
            - 'width_ratio': PSF 宽度比 (相对于第一帧)
            - 'strehl_mean': 平均 Strehl 比
        """
        if len(image_sequence) < 3:
            return {
                'estimated_r0': 0.0,
                'wander_rms': 0.0,
                'width_ratio': 1.0,
                'strehl_mean': 0.0,
            }

        # 计算每帧的质心和宽度
        centroids: List[Tuple[float, float]] = []
        widths: List[float] = []

        for img in image_sequence:
            gray = img.astype(np.float64) if img.ndim == 2 else \
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)

            total = gray.sum()
            if total < 1e-12:
                continue

            yy, xx = np.mgrid[:gray.shape[0], :gray.shape[1]]
            cx = float(np.sum(xx * gray) / total)
            cy = float(np.sum(yy * gray) / total)
            centroids.append((cx, cy))

            # 二阶矩宽度
            dx = xx - cx
            dy = yy - cy
            sigma = float(np.sqrt(
                (np.sum(dx ** 2 * gray) + np.sum(dy ** 2 * gray)) / total
            ))
            widths.append(sigma)

        if len(centroids) < 3:
            return {
                'estimated_r0': 0.0,
                'wander_rms': 0.0,
                'width_ratio': 1.0,
                'strehl_mean': 0.0,
            }

        # 漂移 RMS
        cx_arr = np.array([c[0] for c in centroids])
        cy_arr = np.array([c[1] for c in centroids])
        wander_rms = float(np.sqrt(
            np.var(cx_arr) + np.var(cy_arr)
        ))

        # PSF 宽度比
        if len(widths) >= 2 and widths[0] > 1e-6:
            width_ratio = float(np.mean(widths[1:])) / widths[0]
        else:
            width_ratio = 1.0

        # 估计 r0 (简化: r0 ∝ 1/wander_rms)
        if wander_rms > 1e-6:
            estimated_r0 = 5.0 / wander_rms  # 经验关系
        else:
            estimated_r0 = 1.0

        # 平均 Strehl (基于宽度比)
        strehl_mean = 1.0 / max(width_ratio ** 2, 1.0)

        return {
            'estimated_r0': round(estimated_r0, 4),
            'wander_rms': round(wander_rms, 4),
            'width_ratio': round(width_ratio, 4),
            'strehl_mean': round(strehl_mean, 4),
        }

    def generate_test_sequence(
        self,
        num_frames: int,
        base_image: np.ndarray,
        strength: float = 1.0,
        exposure_time: float = 0.01,
    ) -> List[np.ndarray]:
        """生成完整的湍流退化测试序列。

        Parameters
        ----------
        num_frames : int
            帧数。
        base_image : np.ndarray
            基础光斑图像 (无湍流)。
        strength : float
            湍流强度因子。
        exposure_time : float
            曝光时间 (秒)。

        Returns
        -------
        List[np.ndarray]
            退化后的图像序列。
        """
        sequence: List[np.ndarray] = []
        self._strehl_history = []
        self._rms_history = []

        for i in range(num_frames):
            degraded = self.apply_to_image(base_image, strength)

            # 计算统计
            gray = degraded.astype(np.float64) if degraded.ndim == 2 else \
                cv2.cvtColor(degraded, cv2.COLOR_BGR2GRAY).astype(np.float64)
            total = gray.sum()
            if total > 1e-12:
                peak = gray.max()
                strehl = peak / max(base_image.astype(np.float64).max(), 1e-12)
                self._strehl_history.append(min(strehl, 1.0))

            sequence.append(degraded)

        return sequence

    def get_statistics(self) -> TurbulenceStatistics:
        """获取湍流模拟统计信息。

        Returns
        -------
        TurbulenceStatistics
            统计信息。
        """
        mean_strehl = float(np.mean(self._strehl_history)) if self._strehl_history else 0.0
        mean_rms = float(np.mean(self._rms_history)) if self._rms_history else 0.0

        if self._wander_history:
            wx = np.array([w[0] for w in self._wander_history])
            wy = np.array([w[1] for w in self._wander_history])
            wander_rms = float(np.sqrt(np.var(wx) + np.var(wy)))
        else:
            wander_rms = 0.0

        # 闪烁指数
        if len(self._strehl_history) > 1:
            arr = np.array(self._strehl_history)
            scintillation = float(np.var(arr) / max(np.mean(arr) ** 2, 1e-12))
        else:
            scintillation = 0.0

        return TurbulenceStatistics(
            mean_strehl=round(mean_strehl, 6),
            mean_rms=round(mean_rms, 6),
            wander_rms=round(wander_rms, 6),
            scintillation_index=round(scintillation, 6),
        )

    def reset(self) -> None:
        """重置模拟器状态。"""
        self._current_phase_screen = None
        self._phase_screen_offset_x = 0.0
        self._phase_screen_offset_y = 0.0
        self._strehl_history = []
        self._rms_history = []
        self._wander_history = []

        LOGGER.info("AtmosphericTurbulenceSimulator: 模拟器已重置")

    # ======================== 内部方法 ========================

    def _add_subharmonics(
        self,
        phase_screen: np.ndarray,
        r0: float,
        L0: float,
        size: int,
    ) -> np.ndarray:
        """添加子谐波以改善相位屏的低频成分。

        使用 3 层子谐波，每层将最低频率扩展 3 倍。

        Parameters
        ----------
        phase_screen : np.ndarray
            原始相位屏。
        r0 : float
            Fried 参数。
        L0 : float
            外尺度。
        size : int
            相位屏尺寸。

        Returns
        -------
        np.ndarray
            添加子谐波后的相位屏。
        """
        result = phase_screen.copy()
        df = 1.0 / size

        for p in range(1, 4):
            # 子谐波频率
            sub_df = df / (3.0 ** p)
            sub_fx = np.array([-sub_df, 0.0, sub_df])
            sub_fy = np.array([-sub_df, 0.0, sub_df])
            sub_FX, sub_FY = np.meshgrid(sub_fx, sub_fy)
            sub_f_sq = sub_FX ** 2 + sub_FY ** 2
            sub_f_sq[1, 1] = 1e-10  # 避免除零

            # 子谐波 PSD
            if L0 > 0:
                L0_inv_sq = 1.0 / (L0 ** 2)
                sub_psd = (0.023 * r0 ** (-5.0 / 3.0) *
                           np.exp(-sub_f_sq * L0 ** 2) /
                           (sub_f_sq + L0_inv_sq) ** (11.0 / 6.0))
            else:
                sub_psd = 0.023 * r0 ** (-5.0 / 3.0) * sub_f_sq ** (-11.0 / 6.0)

            sub_psd[1, 1] = 0.0  # 去除零频

            # 生成子谐波
            sub_amplitude = np.sqrt(np.maximum(sub_psd, 0.0)) * sub_df
            sub_random = self._rng.uniform(0, 2.0 * np.pi, (3, 3))
            sub_cn = sub_amplitude * np.exp(1j * sub_random)

            # 将子谐波添加到相位屏
            for i in range(3):
                for j in range(3):
                    if i == 1 and j == 1:
                        continue
                    fx = sub_fx[j]
                    fy = sub_fy[i]
                    yy, xx = np.mgrid[:size, :size]
                    harmonic = sub_cn[i, j] * np.exp(
                        1j * 2.0 * np.pi * (fx * xx + fy * yy)
                    )
                    result += np.real(harmonic)

        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    simulator = AtmosphericTurbulenceSimulator(
        fried_parameter=0.10,
        outer_scale=20.0,
        wind_speed=10.0,
        wavelength=632.8e-9,
        screen_size=256,
        seed=42,
    )

    print("=== 大气湍流模拟器测试 ===\n")

    # 1. 生成相位屏
    print("--- 相位屏生成 ---")
    phase_screen = simulator.generate_phase_screen(256)
    print(f"相位屏尺寸: {phase_screen.shape}")
    print(f"相位范围: [{phase_screen.min():.2f}, {phase_screen.max():.2f}] rad")
    print(f"相位 RMS: {np.std(phase_screen):.4f} rad")

    # 2. 时间演化
    print(f"\n--- Taylor 冻结流演化 ---")
    for i in range(5):
        evolved = simulator.evolve_phase_screen(dt=0.01)
        print(f"  [t={i*0.01:.3f}s] RMS={np.std(evolved):.4f} rad")

    # 3. 图像退化
    print(f"\n--- 图像退化测试 ---")
    size = 64
    yy, xx = np.mgrid[:size, :size]
    cx, cy = size / 2.0, size / 2.0
    sigma = 2.0
    base_psf = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))
    base_psf = (base_psf / base_psf.max() * 255.0).astype(np.uint8)

    for strength in [0.5, 1.0, 2.0]:
        degraded = simulator.apply_to_image(base_psf, strength=strength)
        print(f"  strength={strength:.1f}: "
              f"峰值={degraded.max():.1f}, "
              f"均值={degraded.mean():.2f}")

    # 4. 光束漂移模拟
    print(f"\n--- 光束漂移模拟 ---")
    wander = simulator.simulate_beam_wander(num_frames=100, exposure_time=0.01)
    wander_x = [w[0] for w in wander]
    wander_y = [w[1] for w in wander]
    print(f"  漂移 X RMS: {np.std(wander_x):.4f} px")
    print(f"  漂移 Y RMS: {np.std(wander_y):.4f} px")
    print(f"  最大漂移: {max(np.max(np.abs(wander_x)), np.max(np.abs(wander_y))):.4f} px")

    # 5. 湍流强度估计
    print(f"\n--- 湍流强度估计 ---")
    simulator.reset()
    sequence = simulator.generate_test_sequence(
        num_frames=50, base_image=base_psf, strength=1.0
    )
    estimate = simulator.estimate_turbulence_strength(sequence)
    print(f"  估计 r0: {estimate['estimated_r0']:.4f} m")
    print(f"  漂移 RMS: {estimate['wander_rms']:.4f} px")
    print(f"  宽度比: {estimate['width_ratio']:.4f}")
    print(f"  平均 Strehl: {estimate['strehl_mean']:.4f}")

    # 6. 统计信息
    print(f"\n--- 湍流统计 ---")
    stats = simulator.get_statistics()
    print(f"  平均 Strehl: {stats.mean_strehl:.4f}")
    print(f"  漂移 RMS: {stats.wander_rms:.4f} px")
    print(f"  闪烁指数: {stats.scintillation_index:.4f}")

    print("\n测试完成")
