"""
多层大气湍流仿真器 (MultiLayerTurbulenceSimulator)

灵感来源:
- HCIPy (https://github.com/ehpor/hcipy) — 多层相位屏 + Fresnel 层间传播
- AOtools (https://github.com/AOtools/aotools) — 多层湍流模型与闪烁仿真
- Roddier (1999) — "Adaptive Optics in Astronomy" 多层湍流理论
- Sasiela (1994) — "Electromagnetic Wave Propagation in Turbulence" 层间传播

算法原理:
- Kolmogorov Turbulence — 大气湍流统计模型，相位功率谱密度正比于 f^(-11/3)
- Multi-Layer Phase Screen — 将大气分为多个高度层，每层独立生成相位屏
- Fresnel Propagation — 层间 Fresnel 衍射传播，模拟闪烁效应 (scintillation)
- Von Kármán Model — 改进模型，引入外尺度 L0，避免低频发散:
    PSD(f) = 0.023 * r0^(-5/3) * (f^2 + L0^(-2))^(-11/3)
- FFT Phase Screen Generation — 通过对功率谱密度采样并做 IFFT 生成相位屏
- Taylor Frozen Flow — 泰勒冻结流假设: 湍流在短时间内随风向平移而不演化
- Cn² Profile — 折射率结构常数随高度分布，描述各层湍流强度
- Scintillation — 闪烁效应: 光强起伏，由多层相位屏 Fresnel 传播产生

功能:
- 多层大气湍流仿真 (支持自定义层数、高度、风速、r0、Cn²)
- 单层相位屏生成 (FFT 法 + Kolmogorov/Von Kármán 谱)
- Fresnel 层间传播 (模拟闪烁效应)
- 多层合成波前计算
- 闪烁图样计算
- 湍流统计信息输出

依赖: numpy, opencv-python (仅用于图像处理辅助)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger("SpotZoom.MultiLayerTurbulenceSimulator")

# 模块默认禁用标志
multi_layer_turbulence_enabled: bool = False


@dataclass
class TurbulenceLayerConfig:
    """单层湍流参数配置。"""
    name: str  # 层名称 (如 "ground", "high_alt")
    altitude: float  # 高度 (米)
    wind_speed: float  # 风速 (米/秒)
    wind_direction: float  # 风向角度 (弧度, 0=+x方向)
    r0: float  # Fried 参数 (米), 该层贡献的相干长度
    cn_squared: float  # 折射率结构常数 Cn² (m^(-2/3))
    outer_scale: float = 25.0  # 外尺度 L0 (米)
    weight: float = 1.0  # 层权重 (用于合成)


@dataclass
class MultiLayerTurbulenceReport:
    """多层湍流仿真报告。"""
    num_layers: int  # 湍流层数
    combined_rms: float  # 合成波前 RMS (弧度)
    combined_strehl: float  # 合成 Strehl 比
    isoplanatic_angle: float  # 等晕角 (弧度)
    coherence_time: float  # 相干时间 (秒)
    r0_total: float  # 等效总 Fried 参数 (米)
    layer_rms: Dict[str, float] = field(default_factory=dict)  # 各层波前 RMS
    layer_strehl: Dict[str, float] = field(default_factory=dict)  # 各层 Strehl 比


@dataclass
class ScintillationStats:
    """闪烁统计信息。"""
    scintillation_index: float  # 闪烁指数 σ²_I / <I>²
    mean_intensity: float  # 平均强度
    intensity_std: float  # 强度标准差
    log_amplitude_variance: float  # 对数振幅方差 σ²_χ
    normalized_variance: float  # 归一化强度方差


class TurbulenceLayer:
    """单层大气湍流参数容器。

    Parameters
    ----------
    config : TurbulenceLayerConfig
        湍流层配置参数。
    seed : int or None
        随机种子。
    """

    def __init__(
        self,
        config: TurbulenceLayerConfig,
        seed: Optional[int] = None,
    ):
        self.config = config
        self._rng = np.random.RandomState(seed)

        # 当前相位屏状态
        self._phase_screen: Optional[np.ndarray] = None
        self._offset_x: float = 0.0
        self._offset_y: float = 0.0

        # 统计信息
        self._rms_history: List[float] = []

        LOGGER.debug(
            "TurbulenceLayer '%s': 高度=%.0fm, r0=%.3fm, wind=%.1fm/s, Cn²=%.2e",
            config.name, config.altitude, config.r0,
            config.wind_speed, config.cn_squared,
        )

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def altitude(self) -> float:
        return self.config.altitude

    @property
    def r0(self) -> float:
        return self.config.r0

    @property
    def wind_speed(self) -> float:
        return self.config.wind_speed

    def evolve(self, dt: float) -> None:
        """使用 Taylor 冻结流假设演化相位屏。

        Parameters
        ----------
        dt : float
            时间步长 (秒)。
        """
        if self._phase_screen is None:
            return

        ws = self.config.wind_speed
        direction = self.config.wind_direction

        # 计算平移量 (像素)
        shift_x = ws * np.cos(direction) * dt
        shift_y = ws * np.sin(direction) * dt

        self._offset_x += shift_x
        self._offset_y += shift_y

        # 使用循环平移
        shift_x_int = int(round(self._offset_x))
        shift_y_int = int(round(self._offset_y))

        if shift_x_int != 0:
            self._phase_screen = np.roll(
                self._phase_screen, -shift_x_int, axis=1
            )
            self._offset_x -= shift_x_int

        if shift_y_int != 0:
            self._phase_screen = np.roll(
                self._phase_screen, -shift_y_int, axis=0
            )
            self._offset_y -= shift_y_int

    def get_phase_screen(self) -> Optional[np.ndarray]:
        """获取当前相位屏。"""
        return self._phase_screen

    def set_phase_screen(self, screen: np.ndarray) -> None:
        """设置相位屏。"""
        self._phase_screen = screen.copy()
        self._offset_x = 0.0
        self._offset_y = 0.0

    def record_rms(self) -> None:
        """记录当前相位屏的 RMS。"""
        if self._phase_screen is not None:
            rms = float(np.std(self._phase_screen))
            self._rms_history.append(rms)

    def reset(self) -> None:
        """重置层状态。"""
        self._phase_screen = None
        self._offset_x = 0.0
        self._offset_y = 0.0
        self._rms_history.clear()


class MultiLayerTurbulenceSimulator:
    """多层大气湍流仿真器。

    模拟多层大气湍流引起的波前畸变，支持闪烁效应 (scintillation)。
    使用 FFT 方法生成各层相位屏，通过 Fresnel 传播模拟层间耦合。

    Parameters
    ----------
    wavelength : float
        波长 (米)。默认 632.8nm (HeNe 激光)。
    grid_size : int
        相位屏网格尺寸 (像素)。建议 2 的幂次。
    pixel_scale : float
        像素尺度 (米/像素)。
    seed : int or None
        全局随机种子。各层使用 seed + layer_index 作为独立种子。
    subharmonics : bool
        是否添加子谐波以改善低频成分。
    """

    def __init__(
        self,
        wavelength: float = 632.8e-9,
    grid_size: int = 256,
    pixel_scale: float = 0.005,
        seed: Optional[int] = None,
        subharmonics: bool = True,
    ):
        self.wavelength = float(wavelength)
        self.grid_size = int(grid_size)
        self.pixel_scale = float(pixel_scale)
        self.subharmonics = subharmonics
        self._global_seed = seed

        # 湍流层列表
        self._layers: List[TurbulenceLayer] = []

        # 缓存
        self._last_combined_wavefront: Optional[np.ndarray] = None
        self._last_scintillation: Optional[np.ndarray] = None

        LOGGER.info(
            "MultiLayerTurbulenceSimulator: 初始化完成 "
            "(lambda=%.1fnm, grid=%d, pixel_scale=%.4fm)",
            self.wavelength * 1e9, self.grid_size, self.pixel_scale,
        )

    @property
    def num_layers(self) -> int:
        return len(self._layers)

    @property
    def layers(self) -> List[TurbulenceLayer]:
        return list(self._layers)

    def add_layer(self, config: TurbulenceLayerConfig) -> None:
        """添加湍流层。

        Parameters
        ----------
        config : TurbulenceLayerConfig
            湍流层配置。
        """
        layer_seed = (
            self._global_seed + len(self._layers)
            if self._global_seed is not None
            else None
        )
        layer = TurbulenceLayer(config, seed=layer_seed)
        self._layers.append(layer)

        LOGGER.info(
            "MultiLayerTurbulenceSimulator: 添加层 '%s' (高度=%.0fm, r0=%.3fm)",
            config.name, config.altitude, config.r0,
        )

    def remove_layer(self, name: str) -> None:
        """移除指定名称的湍流层。

        Parameters
        ----------
        name : str
            层名称。
        """
        original_len = len(self._layers)
        self._layers = [l for l in self._layers if l.name != name]
        if len(self._layers) < original_len:
            LOGGER.info("MultiLayerTurbulenceSimulator: 移除层 '%s'", name)
        else:
            LOGGER.warning(
                "MultiLayerTurbulenceSimulator: 未找到层 '%s'", name
            )

    def get_layer(self, name: str) -> Optional[TurbulenceLayer]:
        """获取指定名称的湍流层。"""
        for layer in self._layers:
            if layer.name == name:
                return layer
        return None

    def setup_default_layers(self) -> None:
        """设置默认的多层湍流配置。

        使用典型夜间天文台条件:
        - 地面层: 强湍流, 低风速
        - 中间层: 中等湍流
        - 高空层: 弱湍流, 高风速 (急流)
        """
        self._layers.clear()

        default_configs = [
            TurbulenceLayerConfig(
                name="ground",
                altitude=0.0,
                wind_speed=5.0,
                wind_direction=0.0,
                r0=0.15,
                cn_squared=3.6e-14,
                outer_scale=25.0,
                weight=0.6,
            ),
            TurbulenceLayerConfig(
                name="mid_alt",
                altitude=5000.0,
                wind_speed=15.0,
                wind_direction=np.pi / 6,
                r0=0.25,
                cn_squared=8.0e-15,
                outer_scale=30.0,
                weight=0.3,
            ),
            TurbulenceLayerConfig(
                name="high_alt",
                altitude=12000.0,
                wind_speed=30.0,
                wind_direction=np.pi / 3,
                r0=0.40,
                cn_squared=2.0e-15,
                outer_scale=40.0,
                weight=0.1,
            ),
        ]

        for config in default_configs:
            self.add_layer(config)

        LOGGER.info(
            "MultiLayerTurbulenceSimulator: 已设置 %d 层默认湍流配置",
            len(default_configs),
        )

    def generate_phase_screen(
        self,
        layer: TurbulenceLayer,
        grid_size: Optional[int] = None,
        pixel_scale: Optional[float] = None,
    ) -> np.ndarray:
        """生成单层相位屏 (FFT 法 + Kolmogorov/Von Kármán 谱)。

        Parameters
        ----------
        layer : TurbulenceLayer
            目标湍流层。
        grid_size : int or None
            相位屏尺寸。为 None 时使用仿真器默认值。
        pixel_scale : float or None
            像素尺度 (米/像素)。为 None 时使用仿真器默认值。

        Returns
        -------
        np.ndarray
            相位屏 (2D, 弧度)。
        """
        size = int(grid_size) if grid_size is not None else self.grid_size
        ps = float(pixel_scale) if pixel_scale is not None else self.pixel_scale

        if size < 4:
            raise ValueError(f"grid_size 必须 >= 4, 当前值: {size}")

        r0 = layer.r0
        L0 = layer.config.outer_scale
        rng = layer._rng

        # 频率网格 (空间频率, cycles/meter)
        # 使用无量纲归一化频率以保持数值稳定性
        # 物理频率 f_phys = f_norm / D, 其中 D = size * ps
        D = size * ps  # 总物理尺寸 (米)
        df = 1.0 / D  # 物理频率间隔
        fx = np.fft.fftfreq(size, d=ps)
        fy = np.fft.fftfreq(size, d=ps)
        FX, FY = np.meshgrid(fx, fy)
        f_sq = FX ** 2 + FY ** 2

        # 避免零频
        f_sq_safe = f_sq.copy()
        f_sq_safe[0, 0] = 1e-20

        # 功率谱密度 (相位结构函数, 单位: rad² m²)
        if L0 > 0:
            # Von Kármán 模型
            # Φ_φ(f) = 0.023 * r0^(-5/3) * exp(-f²/L0²) / (f² + 1/L0²)^(11/6)
            L0_inv_sq = 1.0 / (L0 ** 2)
            psd = (
                0.023 * r0 ** (-5.0 / 3.0)
                * np.exp(-f_sq * L0 ** 2)
                / (f_sq_safe + L0_inv_sq) ** (11.0 / 6.0)
            )
        else:
            # 纯 Kolmogorov 模型
            # Φ_φ(f) = 0.023 * r0^(-5/3) * f^(-11/3)
            psd = 0.023 * r0 ** (-5.0 / 3.0) * f_sq_safe ** (-11.0 / 6.0)

        # 零频设为 0 (去除 piston)
        psd[0, 0] = 0.0

        # 生成随机相位屏
        # 振幅 = sqrt(PSD * df²), 其中 df² 是频率单元面积
        amplitude = np.sqrt(np.maximum(psd, 0.0) * df * df)
        random_phase = rng.uniform(0, 2.0 * np.pi, (size, size))
        cn = amplitude * np.exp(1j * random_phase)

        # IFFT 得到相位屏
        phase_screen = np.real(np.fft.ifft2(np.fft.ifftshift(cn))) * (size * size)

        # 添加子谐波 (改善低频)
        if self.subharmonics:
            phase_screen = self._add_subharmonics(
                phase_screen, r0, L0, size, ps, rng
            )

        # PSD 公式 0.023 * r0^(-5/3) 已包含波长归一化，
        # 输出即为弧度单位的相位屏，无需额外转换
        layer.set_phase_screen(phase_screen)

        LOGGER.debug(
            "generate_phase_screen: 层 '%s', RMS=%.4f rad, 范围=[%.2f, %.2f]",
            layer.name,
            np.std(phase_screen),
            phase_screen.min(),
            phase_screen.max(),
        )

        return phase_screen

    def propagate_between_layers(
        self,
        phase: np.ndarray,
        distance: float,
        wavelength: Optional[float] = None,
        pixel_scale: Optional[float] = None,
    ) -> np.ndarray:
        """Fresnel 层间传播。

        使用角谱方法 (Angular Spectrum Method) 实现 Fresnel 传播，
        模拟光波在两层湍流之间的衍射效应。

        Parameters
        ----------
        phase : np.ndarray
            输入相位屏 (弧度)。
        distance : float
            传播距离 (米)。
        wavelength : float or None
            波长 (米)。为 None 时使用仿真器默认值。
        pixel_scale : float or None
            像素尺度 (米/像素)。为 None 时使用仿真器默认值。

        Returns
        -------
        np.ndarray
            传播后的复振幅场。
        """
        wl = float(wavelength) if wavelength is not None else self.wavelength
        ps = float(pixel_scale) if pixel_scale is not None else self.pixel_scale

        size = phase.shape[0]

        # 构建复振幅场 E = exp(i * phase)
        E = np.exp(1j * phase)

        # 角谱方法
        # 1. FFT
        E_fft = np.fft.fft2(E)

        # 2. 频率网格
        fx = np.fft.fftfreq(size, d=ps)
        fy = np.fft.fftfreq(size, d=ps)
        FX, FY = np.meshgrid(fx, fy)

        # 3. Fresnel 传递函数
        # H(fx, fy) = exp(i * 2π * d * sqrt(1/λ² - fx² - fy²))
        # 对于 Fresnel 近似: H ≈ exp(i * π * λ * d * (fx² + fy²))
        f_sq = FX ** 2 + FY ** 2

        # 判断是否在传播带宽内
        k = 2.0 * np.pi / wl
        prop_mask = f_sq < (1.0 / wl) ** 2

        # Fresnel 传递函数 (近轴近似)
        H = np.zeros((size, size), dtype=np.complex128)
        H[prop_mask] = np.exp(1j * np.pi * wl * distance * f_sq[prop_mask])

        # 4. 传播
        E_propagated = np.fft.ifft2(E_fft * H)

        LOGGER.debug(
            "propagate_between_layers: 距离=%.1fm, 波长=%.1fnm, "
            "振幅范围=[%.4f, %.4f]",
            distance, wl * 1e9,
            np.abs(E_propagated).min(),
            np.abs(E_propagated).max(),
        )

        return E_propagated

    def compute_combined_wavefront(
        self,
        grid_size: Optional[int] = None,
        wavelength: Optional[float] = None,
        time: float = 0.0,
        dt: Optional[float] = None,
    ) -> np.ndarray:
        """计算多层合成波前。

        将各层相位屏叠加，考虑 Fresnel 层间传播效应，
        生成到达孔径平面的合成波前。

        Parameters
        ----------
        grid_size : int or None
            网格尺寸。为 None 时使用默认值。
        wavelength : float or None
            波长 (米)。为 None 时使用默认值。
        time : float
            当前时间 (秒)，用于时间演化。
        dt : float or None
            时间步长 (秒)。为 None 时不进行演化。

        Returns
        -------
        np.ndarray
            合成波前相位 (2D, 弧度)。
        """
        size = int(grid_size) if grid_size is not None else self.grid_size
        wl = float(wavelength) if wavelength is not None else self.wavelength

        if not self._layers:
            raise RuntimeError("未添加任何湍流层，请先调用 add_layer() 或 setup_default_layers()")

        # 确保所有层都有相位屏
        for layer in self._layers:
            if layer.get_phase_screen() is None:
                self.generate_phase_screen(layer, grid_size=size)

        # 时间演化
        if dt is not None and dt > 0:
            for layer in self._layers:
                layer.evolve(dt)

        # 按高度排序 (从低到高)
        sorted_layers = sorted(self._layers, key=lambda l: l.altitude)

        # 从最高层开始向下传播
        # 初始化复振幅场 (平面波)
        E = np.ones((size, size), dtype=np.complex128)

        for i, layer in enumerate(sorted_layers):
            phase = layer.get_phase_screen()
            if phase is None:
                continue

            # 确保尺寸匹配
            if phase.shape[0] != size:
                phase = cv2.resize(
                    phase, (size, size),
                    interpolation=cv2.INTER_LINEAR,
                )

            # 应用该层相位
            E = E * np.exp(1j * phase)

            # Fresnel 传播到下一层 (如果有下一层)
            if i < len(sorted_layers) - 1:
                next_layer = sorted_layers[i + 1]
                distance = abs(next_layer.altitude - layer.altitude)
                if distance > 0:
                    E = self.propagate_between_layers(
                        np.angle(E), distance, wl
                    )
                    # 取振幅和相位
                    amplitude = np.abs(E)
                    phase_propagated = np.angle(E)
                    E = amplitude * np.exp(1j * phase_propagated)

            # 记录 RMS
            layer.record_rms()

        # 提取合成波前相位
        combined_phase = np.angle(E)

        self._last_combined_wavefront = combined_phase

        LOGGER.debug(
            "compute_combined_wavefront: %d 层, RMS=%.4f rad, "
            "时间=%.4fs",
            len(sorted_layers),
            np.std(combined_phase),
            time,
        )

        return combined_phase

    def compute_scintillation(
        self,
        wavefront: np.ndarray,
        wavelength: Optional[float] = None,
        pixel_scale: Optional[float] = None,
    ) -> np.ndarray:
        """计算闪烁图样。

        闪烁效应由波前振幅变化引起，表现为光强起伏。
        通过对波前做 Fresnel 传播提取振幅分量。

        Parameters
        ----------
        wavefront : np.ndarray
            波前相位 (2D, 弧度)。
        wavelength : float or None
            波长 (米)。为 None 时使用默认值。
        pixel_scale : float or None
            像素尺度 (米/像素)。为 None 时使用默认值。

        Returns
        -------
        np.ndarray
            闪烁图样 (归一化强度, 2D)。
        """
        wl = float(wavelength) if wavelength is not None else self.wavelength
        ps = float(pixel_scale) if pixel_scale is not None else self.pixel_scale

        size = wavefront.shape[0]

        # 构建复振幅场
        E = np.exp(1j * wavefront)

        # 对波前做短距离 Fresnel 传播以提取闪烁分量
        # 传播距离取为湍流层间平均距离的一部分
        if self._layers:
            altitudes = [l.altitude for l in self._layers]
            if len(altitudes) > 1:
                mean_sep = np.mean(np.diff(sorted(altitudes)))
            else:
                mean_sep = 1000.0
        else:
            mean_sep = 1000.0

        # 使用一小段传播距离来提取闪烁
        prop_distance = mean_sep * 0.1

        E_propagated = self.propagate_between_layers(
            wavefront, prop_distance, wl, ps
        )

        # 闪烁图样 = 传播后强度
        intensity = np.abs(E_propagated) ** 2

        # 归一化
        mean_intensity = intensity.mean()
        if mean_intensity > 1e-12:
            intensity_normalized = intensity / mean_intensity
        else:
            intensity_normalized = np.ones_like(intensity)

        self._last_scintillation = intensity_normalized

        LOGGER.debug(
            "compute_scintillation: 均值=%.4f, 标准差=%.4f, "
            "范围=[%.4f, %.4f]",
            mean_intensity,
            np.std(intensity_normalized),
            intensity_normalized.min(),
            intensity_normalized.max(),
        )

        return intensity_normalized

    def get_turbulence_statistics(self) -> MultiLayerTurbulenceReport:
        """获取湍流统计信息。

        Returns
        -------
        MultiLayerTurbulenceReport
            多层湍流统计报告。
        """
        if not self._layers:
            return MultiLayerTurbulenceReport(
                num_layers=0,
                combined_rms=0.0,
                combined_strehl=0.0,
                isoplanatic_angle=0.0,
                coherence_time=0.0,
                r0_total=0.0,
            )

        # 各层统计
        layer_rms: Dict[str, float] = {}
        layer_strehl: Dict[str, float] = {}

        for layer in self._layers:
            phase = layer.get_phase_screen()
            if phase is not None:
                rms = float(np.std(phase))
                # Maréchal 近似 Strehl 比
                strehl = float(np.exp(-(2.0 * np.pi * rms) ** 2))
            else:
                rms = 0.0
                strehl = 1.0

            layer_rms[layer.name] = round(rms, 6)
            layer_strehl[layer.name] = round(strehl, 6)

        # 合成波前统计
        if self._last_combined_wavefront is not None:
            combined_rms = float(np.std(self._last_combined_wavefront))
            combined_strehl = float(
                np.exp(-(2.0 * np.pi * combined_rms) ** 2)
            )
        else:
            combined_rms = 0.0
            combined_strehl = 1.0

        # 等效总 r0 (从各层 Cn² 积分计算)
        # r0_total^(-5/3) = Σ (Cn²_i * Δh_i) * (2π/λ)² * 0.423
        total_cn2_dh = 0.0
        sorted_layers = sorted(self._layers, key=lambda l: l.altitude)
        for i, layer in enumerate(sorted_layers):
            if i < len(sorted_layers) - 1:
                dh = abs(sorted_layers[i + 1].altitude - layer.altitude)
            else:
                dh = abs(layer.altitude - (
                    sorted_layers[i - 1].altitude if i > 0 else 0.0
                ))
            total_cn2_dh += layer.config.cn_squared * dh

        k = 2.0 * np.pi / self.wavelength
        if total_cn2_dh > 0:
            r0_total = (0.423 * k ** 2 * total_cn2_dh) ** (-3.0 / 5.0)
        else:
            r0_total = float('inf')

        # 等晕角 θ₀ ≈ 0.314 * r0 / h_eff (简化)
        if sorted_layers:
            # 加权平均高度
            total_weight = sum(l.config.weight for l in self._layers)
            if total_weight > 0:
                h_eff = sum(
                    l.altitude * l.config.weight / total_weight
                    for l in self._layers
                )
            else:
                h_eff = sorted_layers[-1].altitude / 2.0
            isoplanatic_angle = 0.314 * r0_total / max(h_eff, 1.0)
        else:
            isoplanatic_angle = 0.0

        # 相干时间 τ₀ ≈ 0.314 * r0 / v_eff
        total_v = sum(
            l.wind_speed * l.config.weight for l in self._layers
        )
        total_weight = sum(l.config.weight for l in self._layers)
        v_eff = total_v / max(total_weight, 1e-12)
        coherence_time = 0.314 * r0_total / max(v_eff, 1e-12)

        report = MultiLayerTurbulenceReport(
            num_layers=len(self._layers),
            combined_rms=round(combined_rms, 6),
            combined_strehl=round(min(combined_strehl, 1.0), 6),
            isoplanatic_angle=round(isoplanatic_angle, 6),
            coherence_time=round(coherence_time, 6),
            r0_total=round(r0_total, 6),
            layer_rms=layer_rms,
            layer_strehl=layer_strehl,
        )

        LOGGER.info(
            "get_turbulence_statistics: %d 层, r0_total=%.4fm, "
            "Strehl=%.4f, θ₀=%.4f rad, τ₀=%.4fs",
            report.num_layers, report.r0_total,
            report.combined_strehl, report.isoplanatic_angle,
            report.coherence_time,
        )

        return report

    def get_scintillation_stats(
        self,
        scintillation_map: Optional[np.ndarray] = None,
    ) -> ScintillationStats:
        """计算闪烁统计信息。

        Parameters
        ----------
        scintillation_map : np.ndarray or None
            闪烁图样。为 None 时使用最近计算的图样。

        Returns
        -------
        ScintillationStats
            闪烁统计信息。
        """
        if scintillation_map is None:
            scintillation_map = self._last_scintillation

        if scintillation_map is None:
            return ScintillationStats(
                scintillation_index=0.0,
                mean_intensity=0.0,
                intensity_std=0.0,
                log_amplitude_variance=0.0,
                normalized_variance=0.0,
            )

        mean_i = float(np.mean(scintillation_map))
        std_i = float(np.std(scintillation_map))

        # 闪烁指数 σ²_I / <I>²
        if mean_i > 1e-12:
            scint_index = (std_i / mean_i) ** 2
        else:
            scint_index = 0.0

        # 对数振幅方差 σ²_χ
        # χ = ln(I / <I>) / 2
        if mean_i > 1e-12:
            log_amplitude = 0.5 * np.log(
                np.maximum(scintillation_map / mean_i, 1e-12)
            )
            log_amp_var = float(np.var(log_amplitude))
        else:
            log_amp_var = 0.0

        # 归一化强度方差
        norm_var = float(np.var(scintillation_map))

        return ScintillationStats(
            scintillation_index=round(scint_index, 6),
            mean_intensity=round(mean_i, 6),
            intensity_std=round(std_i, 6),
            log_amplitude_variance=round(log_amp_var, 6),
            normalized_variance=round(norm_var, 6),
        )

    def reset(self) -> None:
        """重置所有层状态和缓存。"""
        for layer in self._layers:
            layer.reset()

        self._last_combined_wavefront = None
        self._last_scintillation = None

        LOGGER.info("MultiLayerTurbulenceSimulator: 仿真器已重置")

    # ======================== 内部方法 ========================

    def _add_subharmonics(
        self,
        phase_screen: np.ndarray,
        r0: float,
        L0: float,
        size: int,
        pixel_scale: float,
        rng: np.random.RandomState,
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
        pixel_scale : float
            像素尺度 (米/像素)。
        rng : np.random.RandomState
            随机数生成器。

        Returns
        -------
        np.ndarray
            添加子谐波后的相位屏。
        """
        result = phase_screen.copy()
        df = 1.0 / (size * pixel_scale)

        for p in range(1, 4):
            sub_df = df / (3.0 ** p)
            sub_fx = np.array([-sub_df, 0.0, sub_df])
            sub_fy = np.array([-sub_df, 0.0, sub_df])
            sub_FX, sub_FY = np.meshgrid(sub_fx, sub_fy)
            sub_f_sq = sub_FX ** 2 + sub_FY ** 2
            sub_f_sq[1, 1] = 1e-20

            if L0 > 0:
                L0_inv_sq = 1.0 / (L0 ** 2)
                sub_psd = (
                    0.023 * r0 ** (-5.0 / 3.0)
                    * np.exp(-sub_f_sq * L0 ** 2)
                    / (sub_f_sq + L0_inv_sq) ** (11.0 / 6.0)
                )
            else:
                sub_psd = 0.023 * r0 ** (-5.0 / 3.0) * sub_f_sq ** (-11.0 / 6.0)

            sub_psd[1, 1] = 0.0

            sub_amplitude = np.sqrt(np.maximum(sub_psd, 0.0)) * sub_df
            sub_random = rng.uniform(0, 2.0 * np.pi, (3, 3))
            sub_cn = sub_amplitude * np.exp(1j * sub_random)

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

    print("=== 多层大气湍流仿真器测试 ===\n")

    # 1. 创建仿真器并设置默认层
    simulator = MultiLayerTurbulenceSimulator(
        wavelength=632.8e-9,
        grid_size=256,
        pixel_scale=0.01,
        seed=42,
        subharmonics=True,
    )
    simulator.setup_default_layers()

    print(f"湍流层数: {simulator.num_layers}")
    for layer in simulator.layers:
        print(f"  层 '{layer.name}': 高度={layer.altitude:.0f}m, "
              f"r0={layer.r0:.3f}m, 风速={layer.wind_speed:.1f}m/s")

    # 2. 生成各层相位屏
    print(f"\n--- 相位屏生成 ---")
    for layer in simulator.layers:
        phase = simulator.generate_phase_screen(layer)
        print(f"  层 '{layer.name}': RMS={np.std(phase):.4f} rad, "
              f"范围=[{phase.min():.2f}, {phase.max():.2f}]")

    # 3. 计算合成波前
    print(f"\n--- 合成波前计算 ---")
    combined = simulator.compute_combined_wavefront(time=0.0)
    print(f"  合成波前 RMS: {np.std(combined):.4f} rad")
    print(f"  合成波前范围: [{combined.min():.2f}, {combined.max():.2f}]")

    # 4. 时间演化
    print(f"\n--- 时间演化 (Taylor 冻结流) ---")
    for i in range(5):
        evolved = simulator.compute_combined_wavefront(
            time=(i + 1) * 0.01, dt=0.01
        )
        print(f"  [t={(i+1)*0.01:.3f}s] RMS={np.std(evolved):.4f} rad")

    # 5. 闪烁计算
    print(f"\n--- 闪烁效应 ---")
    scintillation = simulator.compute_scintillation(combined)
    print(f"  闪烁图样均值: {scintillation.mean():.4f}")
    print(f"  闪烁图样标准差: {scintillation.std():.4f}")
    print(f"  闪烁图样范围: [{scintillation.min():.4f}, {scintillation.max():.4f}]")

    scint_stats = simulator.get_scintillation_stats()
    print(f"  闪烁指数: {scint_stats.scintillation_index:.6f}")
    print(f"  对数振幅方差: {scint_stats.log_amplitude_variance:.6f}")

    # 6. 湍流统计
    print(f"\n--- 湍流统计报告 ---")
    report = simulator.get_turbulence_statistics()
    print(f"  层数: {report.num_layers}")
    print(f"  等效 r0: {report.r0_total:.4f} m")
    print(f"  合成 Strehl: {report.combined_strehl:.4f}")
    print(f"  等晕角: {report.isoplanatic_angle:.6f} rad")
    print(f"  相干时间: {report.coherence_time:.4f} s")
    print(f"  各层 RMS: {report.layer_rms}")
    print(f"  各层 Strehl: {report.layer_strehl}")

    # 7. 自定义层测试
    print(f"\n--- 自定义层测试 ---")
    simulator.reset()
    simulator.add_layer(TurbulenceLayerConfig(
        name="strong",
        altitude=0.0,
        wind_speed=8.0,
        wind_direction=0.0,
        r0=0.05,
        cn_squared=1.0e-13,
        outer_scale=20.0,
        weight=1.0,
    ))
    custom_combined = simulator.compute_combined_wavefront()
    print(f"  强湍流层 RMS: {np.std(custom_combined):.4f} rad")

    print("\n测试完成")
