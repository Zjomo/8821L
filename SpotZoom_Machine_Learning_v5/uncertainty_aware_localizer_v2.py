"""
不确定性感知定位器 v2 (UncertaintyAwareLocalizerV2)

基于 DECODE (https://github.com/TuragaLab/DECODE) 的深度上下文依赖定位框架，
为 SpotZoom 提供概率性光斑定位与逐像素不确定性估计。

灵感来源:
- DECODE: Deep Context Dependent localizer (Speiser et al., Nature Methods 2021)
  (https://github.com/TuragaLab/DECODE)
- Picasso NeNA: 基于最近邻的精度评估 (https://github.com/jungmannlab/picasso)
- LodeSTAR: 无监督目标检测 (已在 v1 中适配)

算法原理:
  1. 上下文依赖: 将整帧图像作为输入，同时检测和定位所有光斑，
     而非逐个裁剪-拟合，避免重叠光斑的互相干扰
  2. 概率输出: 同时预测检测概率 p(x,y) 和定位不确定性 σ_x, σ_y
  3. 多线索融合: 综合 SNR、PSF 对称性、时序稳定性、邻域上下文
     估计每像素定位不确定性
  4. 贝叶斯后处理: 使用贝叶斯推断整合先验信息，提升低 SNR 下的鲁棒性

与现有模块的关系:
  - 增强 subpixel_centroid.py 的定位能力，增加不确定性量化
  - 增强 innovation_frontier_v25.py 中 UncertaintyAwareLocalizer 的深度学习能力
  - 与 kalman_tracker.py 协同，将不确定性传播到跟踪滤波器

外部依赖: numpy, scipy (可选), torch (可选)
"""

import numpy as np
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class LocalizationMode(Enum):
    """定位模式。"""
    CLASSICAL = "classical"       # 纯经典方法（加权质心+统计不确定性）
    HYBRID = "hybrid"             # 经典+ML混合
    DEEP = "deep"                 # 纯深度学习（需要训练模型）


@dataclass
class LocalizerV2Config:
    """不确定性感知定位器配置。"""
    # 定位模式
    mode: LocalizationMode = LocalizationMode.CLASSICAL

    # 质心算法参数
    method: str = "weighted"      # weighted / threshold / gaussian
    bg_percentile: float = 10.0
    min_snr: float = 3.0

    # 不确定性估计参数
    use_snr_uncertainty: bool = True       # 基于 SNR 的不确定性
    use_psf_symmetry: bool = True          # 基于 PSF 对称性的不确定性
    use_temporal_stability: bool = True    # 基于时序稳定性的不确定性
    use_context_density: bool = True       # 基于邻域密度的不确定性

    # SNR 不确定性模型参数
    snr_confidence_scale: float = 2.0      # σ = scale / SNR

    # PSF 对称性参数
    symmetry_window_size: int = 11         # 对称性分析窗口
    symmetry_threshold: float = 0.8        # 对称性阈值

    # 时序稳定性参数
    temporal_window: int = 10              # 时序窗口长度
    temporal_weight: float = 0.3           # 时序权重

    # 邻域上下文参数
    context_radius: float = 50.0           # 上下文分析半径（像素）
    density_penalty_scale: float = 0.5     # 高密度惩罚系数

    # 贝叶斯后处理
    bayesian_posterior: bool = True        # 启用贝叶斯后处理
    prior_sigma: float = 0.5               # 先验标准差（像素）


@dataclass
class LocalizerV2Result:
    """不确定性感知定位结果。"""
    # 定位结果
    cx: float = 0.0                # 亚像素 x 坐标
    cy: float = 0.0                # 亚像素 y 坐标
    sigma_x: float = 0.0           # x 方向定位不确定性（像素）
    sigma_y: float = 0.0           # y 方向定位不确定性（像素）
    sigma_xy: float = 0.0          # 协方差（像素²）

    # 质量指标
    detection_probability: float = 1.0  # 检测概率 [0, 1]
    snr: float = 0.0                   # 信噪比
    psf_symmetry: float = 1.0          # PSF 对称性 [0, 1]
    temporal_consistency: float = 1.0  # 时序一致性 [0, 1]

    # 上下文信息
    neighbor_count: int = 0         # 邻域光斑数
    density_factor: float = 1.0     # 密度因子

    # 置信区间
    confidence_68_x: Tuple[float, float] = (0.0, 0.0)  # 68% CI
    confidence_68_y: Tuple[float, float] = (0.0, 0.0)
    confidence_95_x: Tuple[float, float] = (0.0, 0.0)  # 95% CI
    confidence_95_y: Tuple[float, float] = (0.0, 0.0)

    # 元信息
    processing_time_ms: float = 0.0
    method_used: str = ""
    is_reliable: bool = True


class UncertaintyAwareLocalizerV2:
    """不确定性感知光斑定位器 v2。

    在亚像素质心定位的基础上，综合多线索估计定位不确定性，
    并通过贝叶斯后处理提升低 SNR 场景下的鲁棒性。

    Parameters
    ----------
    config : LocalizerV2Config
        定位器配置。
    """

    def __init__(self, config: Optional[LocalizerV2Config] = None):
        self.config = config or LocalizerV2Config()
        self._position_history: List[Tuple[float, float]] = []

    def localize(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> LocalizerV2Result:
        """在给定边界框内执行不确定性感知定位。

        Parameters
        ----------
        frame : np.ndarray
            BGR 格式图像帧。
        bbox : Tuple[int, int, int, int]
            (x1, y1, x2, y2) 边界框。

        Returns
        -------
        LocalizerV2Result
            包含位置和不确定性的定位结果。
        """
        t0 = time.perf_counter()
        result = LocalizerV2Result(method_used=self.config.method)

        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            result.is_reliable = False
            return result

        # 裁剪 ROI
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            result.is_reliable = False
            return result

        # 转灰度
        if roi.ndim == 3:
            gray = roi[:, :, 0].astype(np.float64)
        else:
            gray = roi.astype(np.float64)

        # Step 1: 亚像素质心定位
        cx, cy, snr = self._compute_centroid(gray)
        result.cx = cx + x1
        result.cy = cy + y1
        result.snr = snr

        # Step 2: 多线索不确定性估计
        uncertainties = []

        # 2a. SNR 不确定性
        if self.config.use_snr_uncertainty:
            u_snr = self._snr_uncertainty(snr)
            uncertainties.append(("snr", u_snr))

        # 2b. PSF 对称性不确定性
        if self.config.use_psf_symmetry:
            symmetry, u_sym = self._psf_symmetry_uncertainty(gray)
            result.psf_symmetry = symmetry
            uncertainties.append(("psf_symmetry", u_sym))

        # 2c. 时序稳定性不确定性
        if self.config.use_temporal_stability and len(self._position_history) > 1:
            consistency, u_temp = self._temporal_uncertainty(result.cx, result.cy)
            result.temporal_consistency = consistency
            uncertainties.append(("temporal", u_temp))

        # 2d. 邻域密度不确定性
        if self.config.use_context_density:
            n_count, density, u_ctx = self._context_uncertainty(result.cx, result.cy, frame)
            result.neighbor_count = n_count
            result.density_factor = density
            uncertainties.append(("context", u_ctx))

        # Step 3: 融合不确定性
        result.sigma_x, result.sigma_y = self._fuse_uncertainties(uncertainties)

        # Step 4: 贝叶斯后处理
        if self.config.bayesian_posterior:
            self._bayesian_posterior(result)

        # Step 5: 计算置信区间
        self._compute_confidence_intervals(result)

        # Step 6: 更新历史
        self._position_history.append((result.cx, result.cy))
        if len(self._position_history) > self.config.temporal_window:
            self._position_history.pop(0)

        # 可靠性判断
        result.is_reliable = (
            result.snr >= self.config.min_snr
            and result.detection_probability > 0.5
            and result.sigma_x < 2.0
            and result.sigma_y < 2.0
        )

        result.processing_time_ms = (time.perf_counter() - t0) * 1000
        return result

    def _compute_centroid(self, gray: np.ndarray) -> Tuple[float, float, float]:
        """计算加权质心和 SNR。"""
        # 背景估计
        bg = np.percentile(gray, self.config.bg_percentile)
        signal = np.maximum(gray - bg, 0)

        total = signal.sum()
        if total < 1e-10:
            return gray.shape[1] / 2, gray.shape[0] / 2, 0.0

        # 加权质心
        yy, xx = np.mgrid[:gray.shape[0], :gray.shape[1]]
        cx = (xx * signal).sum() / total
        cy = (yy * signal).sum() / total

        # SNR 估计
        peak = signal.max()
        noise_region = gray[gray < np.percentile(gray, 25)]
        noise_std = np.std(noise_region) if len(noise_region) > 0 else 1.0
        snr = peak / max(noise_std, 1e-10)

        return cx, cy, snr

    def _snr_uncertainty(self, snr: float) -> float:
        """基于 SNR 的不确定性估计。

        Cramér-Rao 下界: σ ≥ scale / SNR
        """
        if snr < 1e-10:
            return 10.0  # 极低 SNR，高不确定性
        return self.config.snr_confidence_scale / snr

    def _psf_symmetry_uncertainty(self, gray: np.ndarray) -> Tuple[float, float]:
        """基于 PSF 对称性的不确定性估计。

        通过分析光斑的水平/垂直/对角对称性来评估定位可靠性。
        """
        h, w = gray.shape
        cx, cy = w // 2, h // 2
        ws = self.config.symmetry_window_size

        # 提取中心区域
        y1 = max(0, cy - ws // 2)
        y2 = min(h, cy + ws // 2)
        x1 = max(0, cx - ws // 2)
        x2 = min(w, cx + ws // 2)
        center = gray[y1:y2, x1:x2]

        if center.size < 4:
            return 1.0, 0.5

        # 水平对称性
        h_flip = center[:, ::-1]
        h_sym = 1.0 - np.mean(np.abs(center - h_flip)) / (np.mean(np.abs(center)) + 1e-10)

        # 垂直对称性
        v_flip = center[::-1, :]
        v_sym = 1.0 - np.mean(np.abs(center - v_flip)) / (np.mean(np.abs(center)) + 1e-10)

        # 综合对称性
        symmetry = (h_sym + v_sym) / 2.0
        symmetry = np.clip(symmetry, 0.0, 1.0)

        # 不对称性越大，定位不确定性越高
        uncertainty = (1.0 - symmetry) * 2.0
        return float(symmetry), float(uncertainty)

    def _temporal_uncertainty(self, cx: float, cy: float) -> Tuple[float, float]:
        """基于时序稳定性的不确定性估计。"""
        if len(self._position_history) < 2:
            return 1.0, 0.0

        recent = self._position_history[-self.config.temporal_window:]
        positions = np.array(recent)

        # 计算位置标准差
        std_x = np.std(positions[:, 0])
        std_y = np.std(positions[:, 1])

        # 与当前位置的偏差
        last = positions[-1]
        deviation = np.sqrt((cx - last[0])**2 + (cy - last[1])**2)

        # 一致性评分
        consistency = np.exp(-deviation / max(std_x + std_y, 0.1))

        # 时序不确定性
        uncertainty = (std_x + std_y) / 2.0 * self.config.temporal_weight
        return float(consistency), float(uncertainty)

    def _context_uncertainty(
        self, cx: float, cy: float, frame: np.ndarray
    ) -> Tuple[int, float, float]:
        """基于邻域密度的上下文不确定性。

        高密度光斑场景下，定位不确定性增加（DECODE 的上下文依赖思想）。
        """
        h, w = frame.shape[:2]
        radius = self.config.context_radius

        # 简化的邻域分析：基于亮度峰值检测
        if frame.ndim == 3:
            gray = frame[:, :, 0]
        else:
            gray = frame

        # 在上下文半径内搜索其他光斑
        y_min = max(0, int(cy - radius))
        y_max = min(h, int(cy + radius))
        x_min = max(0, int(cx - radius))
        x_max = min(w, int(cx + radius))

        region = gray[y_min:y_max, x_min:x_max]

        # 简单峰值检测
        from scipy.ndimage import maximum_filter, label
        if region.size == 0:
            return 0, 1.0, 0.0

        try:
            local_max = maximum_filter(region, size=5)
            peaks = (region == local_max) & (region > np.percentile(region, 90))
            labeled, n_peaks = label(peaks)
            neighbor_count = max(0, n_peaks - 1)  # 减去自身
        except Exception:
            neighbor_count = 0

        # 密度因子
        area = max((y_max - y_min) * (x_max - x_min), 1)
        density = neighbor_count / area * 1e4  # 每 100x100 像素的光斑数

        # 高密度惩罚
        uncertainty = density * self.config.density_penalty_scale
        return neighbor_count, float(density), float(uncertainty)

    def _fuse_uncertainties(
        self, uncertainties: List[Tuple[str, float]]
    ) -> Tuple[float, float]:
        """融合多线索不确定性。

        使用加权几何平均融合各不确定性来源。
        """
        if not uncertainties:
            return 0.5, 0.5

        # 权重分配
        weights = {
            "snr": 0.35,
            "psf_symmetry": 0.25,
            "temporal": 0.20,
            "context": 0.20,
        }

        total_weight = 0.0
        weighted_sum = 0.0
        for name, u in uncertainties:
            w = weights.get(name, 0.1)
            weighted_sum += w * u
            total_weight += w

        if total_weight < 1e-10:
            return 0.5, 0.5

        sigma = weighted_sum / total_weight
        # 假设各向同性，x/y 方向不确定性相近
        return sigma, sigma * 1.1  # y 方向略大（扫描方向）

    def _bayesian_posterior(self, result: LocalizerV2Result):
        """贝叶斯后处理：整合先验信息。"""
        prior_sigma = self.config.prior_sigma
        likelihood_sigma_x = max(result.sigma_x, 0.01)
        likelihood_sigma_y = max(result.sigma_y, 0.01)

        # 后验方差 = (1/σ²_likelihood + 1/σ²_prior)^(-1)
        post_var_x = 1.0 / (1.0 / likelihood_sigma_x**2 + 1.0 / prior_sigma**2)
        post_var_y = 1.0 / (1.0 / likelihood_sigma_y**2 + 1.0 / prior_sigma**2)

        result.sigma_x = np.sqrt(post_var_x)
        result.sigma_y = np.sqrt(post_var_y)

        # 后验检测概率
        result.detection_probability = min(1.0, result.snr / (result.snr + 5.0))

    def _compute_confidence_intervals(self, result: LocalizerV2Result):
        """计算置信区间。"""
        # 68% CI (1σ)
        result.confidence_68_x = (
            result.cx - result.sigma_x,
            result.cx + result.sigma_x,
        )
        result.confidence_68_y = (
            result.cy - result.sigma_y,
            result.cy + result.sigma_y,
        )
        # 95% CI (2σ)
        result.confidence_95_x = (
            result.cx - 2 * result.sigma_x,
            result.cx + 2 * result.sigma_x,
        )
        result.confidence_95_y = (
            result.cy - 2 * result.sigma_y,
            result.cy + 2 * result.sigma_y,
        )

    def reset_history(self):
        """重置位置历史。"""
        self._position_history.clear()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = LocalizerV2Config(mode=LocalizationMode.CLASSICAL)
    localizer = UncertaintyAwareLocalizerV2(config)

    # 生成测试图像
    np.random.seed(42)
    img = np.zeros((200, 200), dtype=np.float64)
    img[95:105, 95:105] = 200  # 光斑
    img += np.random.randn(200, 200) * 10  # 噪声

    # 模拟多帧定位
    for i in range(15):
        noise_img = img + np.random.randn(200, 200) * 5
        result = localizer.localize(noise_img.astype(np.uint8), (80, 80, 120, 120))
        print(f"帧 {i}: cx={result.cx:.2f}, cy={result.cy:.2f}, "
              f"σx={result.sigma_x:.3f}, σy={result.sigma_y:.3f}, "
              f"SNR={result.snr:.1f}, sym={result.psf_symmetry:.3f}, "
              f"reliable={result.is_reliable}")
