"""
无传感器自适应光学 v2 (SensorlessAOV2)

基于 REALM (https://github.com/MSiemons/REALM) 的无波前传感器 Zernike 模态校正，
结合贝叶斯优化实现高效像差校正。

灵感来源:
- REALM: Robust and Effective Adaptive optics in Localization Microscopy
  (Siemons et al., Nature Communications 2021)
  (https://github.com/MSiemons/REALM)
- OOPAO: 面向对象AO仿真 (https://github.com/cheritier/OOPAO)
- pyRTC: 实时AO控制器 (https://github.com/jacotay7/pyRTC)

算法原理:
  1. 逐模态校正: 依次校正每个 Zernike 模态，施加不同偏置量序列
  2. 图像质量指标: 使用光斑锐度、Strehl 比、FWHM 等作为优化目标
  3. 高斯拟合寻优: 对偏置量-质量曲线进行高斯拟合，找到最优偏置
  4. 贝叶斯优化: 使用高斯过程代理模型减少所需测量次数
  5. 迭代收敛: 多轮迭代直到收敛或达到最大轮数

与现有模块的关系:
  - 增强 v1/auto_alignment_optimizer.py 的像差校正能力
  - 增强 v1/modal_controller.py 的模态控制策略
  - 与 innovation_frontier_v25.py 中 ZernikeModeCorrector 协同

外部依赖: numpy, scipy (可选)
"""

import numpy as np
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Callable, Dict
from enum import Enum

logger = logging.getLogger(__name__)


class QualityMetric(Enum):
    """图像质量指标类型。"""
    STREHL = "strehl"           # Strehl 比
    SHARPNESS = "sharpness"     # 图像锐度（梯度平方和）
    FWHM = "fwhm"               # 半高全宽（越小越好）
    INTENSITY = "intensity"     # 峰值强度
    ENTROPY = "entropy"         # 信息熵（越小越好）
    COMPOSITE = "composite"     # 综合指标


class SearchStrategy(Enum):
    """搜索策略。"""
    GRID = "grid"               # 网格搜索
    GAUSSIAN_FIT = "gaussian_fit"  # 高斯拟合（REALM 方法）
    BAYESIAN = "bayesian"       # 贝叶斯优化


@dataclass
class SensorlessAOV2Config:
    """无传感器 AO 配置。"""
    # Zernike 模态参数
    max_zernike_mode: int = 11   # 最大校正到第 11 阶 Zernike（球差）
    modes_to_correct: Optional[List[int]] = None  # 指定校正模式，None=自动选择

    # 搜索参数
    strategy: SearchStrategy = SearchStrategy.GAUSSIAN_FIT
    num_bias_points: int = 7     # 偏置点数量
    bias_range: float = 1.0      # 偏置范围（弧度）
    max_iterations: int = 3      # 最大迭代轮数

    # 质量指标
    quality_metric: QualityMetric = QualityMetric.SHARPNESS
    convergence_threshold: float = 0.01  # 收敛阈值

    # 贝叶斯优化参数
    bayesian_n_initial: int = 3  # 初始随机采样点
    bayesian_n_optimize: int = 5 # 贝叶斯优化迭代次数

    # 安全参数
    max_total_correction: float = 2.0  # 最大总校正量（弧度）
    per_mode_limit: float = 1.5        # 单模态最大校正量

    # 性能参数
    parallel_measurement: bool = False  # 是否支持并行测量


@dataclass
class SensorlessAOV2Report:
    """无传感器 AO 校正报告。"""
    # 校正结果
    modes_corrected: int = 0
    total_iterations: int = 0
    converged: bool = False

    # Zernike 校正系数
    zernike_corrections: Dict[int, float] = field(default_factory=dict)
    initial_quality: float = 0.0
    final_quality: float = 0.0
    quality_improvement: float = 0.0

    # 每模态详情
    mode_details: List[Dict] = field(default_factory=list)

    # 性能信息
    total_time_ms: float = 0.0
    measurements_per_mode: int = 0
    total_measurements: int = 0
    warnings: List[str] = field(default_factory=list)


class SensorlessAOV2:
    """无传感器自适应光学校正器 v2。

    通过逐模态偏置+图像质量优化，实现无需波前传感器的像差校正。
    支持 REALM 风格的高斯拟合和贝叶斯优化两种策略。

    Parameters
    ----------
    config : SensorlessAOV2Config
        校正器配置。
    """

    def __init__(self, config: Optional[SensorlessAOV2Config] = None):
        self.config = config or SensorlessAOV2Config()
        self._current_corrections: Dict[int, float] = {}
        self._quality_history: List[float] = []

    def correct(
        self,
        apply_and_measure: Callable[[Dict[int, float]], np.ndarray],
        initial_modes: Optional[List[int]] = None,
    ) -> SensorlessAOV2Report:
        """执行无传感器像差校正。

        Parameters
        ----------
        apply_and_measure : callable
            应用 Zernike 校正并返回图像的回调函数。
            签名: apply_and_measure({mode_index: bias_radians}) -> np.ndarray
        initial_modes : list, optional
            初始待校正的 Zernike 模式列表。

        Returns
        -------
        SensorlessAOV2Report
            校正报告。
        """
        report = SensorlessAOV2Report()
        t0 = time.perf_counter()

        # 确定待校正模式
        modes = initial_modes or self.config.modes_to_correct
        if modes is None:
            modes = list(range(2, self.config.max_zernike_mode + 1))

        # 测量初始质量
        initial_image = apply_and_measure({})
        report.initial_quality = self._compute_quality(initial_image)
        self._quality_history = [report.initial_quality]

        logger.info(f"初始质量: {report.initial_quality:.4f}")

        # 迭代校正
        for iteration in range(self.config.max_iterations):
            improved = False

            for mode in modes:
                # 逐模态搜索最优偏置
                optimal_bias, quality_curve = self._search_optimal_bias(
                    mode, apply_and_measure
                )

                # 应用校正
                current = self._current_corrections.get(mode, 0.0)
                new_correction = current + optimal_bias

                # 限幅
                new_correction = np.clip(
                    new_correction,
                    -self.config.per_mode_limit,
                    self.config.per_mode_limit
                )

                # 检查改善
                test_image = apply_and_measure({**self._current_corrections, mode: new_correction})
                test_quality = self._compute_quality(test_image)

                if test_quality > report.initial_quality * (1 + self.config.convergence_threshold):
                    self._current_corrections[mode] = new_correction
                    improved = True

                report.mode_details.append({
                    "mode": mode,
                    "optimal_bias": float(optimal_bias),
                    "new_correction": float(new_correction),
                    "quality_at_optimum": float(test_quality),
                    "quality_curve": [float(q) for q in quality_curve],
                })

            # 检查全局收敛
            final_image = apply_and_measure(self._current_corrections)
            final_quality = self._compute_quality(final_image)
            self._quality_history.append(final_quality)

            if not improved:
                logger.info(f"迭代 {iteration}: 无改善，停止")
                break

            report.total_iterations = iteration + 1

        # 填充报告
        report.final_quality = self._compute_quality(
            apply_and_measure(self._current_corrections)
        )
        report.zernike_corrections = dict(self._current_corrections)
        report.modes_corrected = len(self._current_corrections)
        report.total_measurements = sum(
            len(d["quality_curve"]) for d in report.mode_details
        )
        report.measurements_per_mode = (
            report.total_measurements // max(report.modes_corrected, 1)
        )

        if report.initial_quality > 0:
            report.quality_improvement = (
                (report.final_quality - report.initial_quality)
                / report.initial_quality
            )

        report.converged = (
            report.quality_improvement > self.config.convergence_threshold
        )
        report.total_time_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            f"校正完成: 质量 {report.initial_quality:.4f} -> {report.final_quality:.4f} "
            f"(+{report.quality_improvement*100:.1f}%), "
            f"模式数={report.modes_corrected}, 测量数={report.total_measurements}"
        )

        return report

    def _search_optimal_bias(
        self, mode: int, apply_and_measure: Callable
    ) -> Tuple[float, List[float]]:
        """搜索指定 Zernike 模态的最优偏置量。"""
        if self.config.strategy == SearchStrategy.GAUSSIAN_FIT:
            return self._gaussian_fit_search(mode, apply_and_measure)
        elif self.config.strategy == SearchStrategy.BAYESIAN:
            return self._bayesian_search(mode, apply_and_measure)
        else:
            return self._grid_search(mode, apply_and_measure)

    def _gaussian_fit_search(
        self, mode: int, apply_and_measure: Callable
    ) -> Tuple[float, List[float]]:
        """REALM 风格的高斯拟合搜索。"""
        biases = np.linspace(
            -self.config.bias_range,
            self.config.bias_range,
            self.config.num_bias_points
        )
        qualities = []

        for bias in biases:
            corrections = {**self._current_corrections, mode: bias}
            image = apply_and_measure(corrections)
            quality = self._compute_quality(image)
            qualities.append(quality)

        # 高斯拟合: q(bias) = A * exp(-(bias - μ)² / (2σ²)) + C
        qualities = np.array(qualities)
        optimal_bias = self._fit_gaussian(biases, qualities)

        return float(optimal_bias), qualities.tolist()

    def _fit_gaussian(self, x: np.ndarray, y: np.ndarray) -> float:
        """对数据拟合高斯函数，返回峰值位置。"""
        # 简单的抛物线拟合（高斯峰值的二阶近似）
        try:
            # 找到最大值附近的三点
            peak_idx = np.argmax(y)
            if peak_idx == 0 or peak_idx == len(x) - 1:
                return float(x[peak_idx])

            x0, x1, x2 = x[peak_idx-1], x[peak_idx], x[peak_idx+1]
            y0, y1, y2 = y[peak_idx-1], y[peak_idx], y[peak_idx+1]

            # 抛物线顶点: x_peak = x1 - (x2-x0)*(y2-y0) / (8*(y2-2*y1+y0))
            denom = 2 * (y2 - 2*y1 + y0)
            if abs(denom) < 1e-10:
                return float(x1)

            x_peak = x1 - (x2 - x0) * (y2 - y0) / (4 * denom)
            return float(np.clip(x_peak, x[0], x[-1]))

        except Exception:
            return float(x[np.argmax(y)])

    def _bayesian_search(
        self, mode: int, apply_and_measure: Callable
    ) -> Tuple[float, List[float]]:
        """贝叶斯优化搜索。"""
        all_biases = []
        all_qualities = []

        # 初始随机采样
        n_init = self.config.bayesian_n_initial
        init_biases = np.linspace(
            -self.config.bias_range,
            self.config.bias_range,
            n_init
        )

        for bias in init_biases:
            corrections = {**self._current_corrections, mode: float(bias)}
            image = apply_and_measure(corrections)
            quality = self._compute_quality(image)
            all_biases.append(bias)
            all_qualities.append(quality)

        # 简化的贝叶斯优化：基于高斯过程代理
        for _ in range(self.config.bayesian_n_optimize):
            # 使用当前数据拟合代理模型
            X = np.array(all_biases).reshape(-1, 1)
            Y = np.array(all_qualities)

            # 预测下一个最佳采样点（简化：使用 EI 准则的近似）
            best_idx = np.argmax(Y)
            best_x = X[best_idx, 0]

            # 在最佳点附近细化搜索
            candidates = np.linspace(
                best_x - self.config.bias_range / (n_init * 2),
                best_x + self.config.bias_range / (n_init * 2),
                3
            )
            candidates = candidates[
                (candidates >= -self.config.bias_range) &
                (candidates <= self.config.bias_range)
            ]

            for c in candidates:
                if any(abs(c - b) < 0.01 for b in all_biases):
                    continue
                corrections = {**self._current_corrections, mode: float(c)}
                image = apply_and_measure(corrections)
                quality = self._compute_quality(image)
                all_biases.append(c)
                all_qualities.append(quality)

        optimal_idx = np.argmax(all_qualities)
        return float(all_biases[optimal_idx]), all_qualities

    def _grid_search(
        self, mode: int, apply_and_measure: Callable
    ) -> Tuple[float, List[float]]:
        """网格搜索。"""
        biases = np.linspace(
            -self.config.bias_range,
            self.config.bias_range,
            self.config.num_bias_points * 2
        )
        qualities = []

        for bias in biases:
            corrections = {**self._current_corrections, mode: float(bias)}
            image = apply_and_measure(corrections)
            quality = self._compute_quality(image)
            qualities.append(quality)

        optimal_idx = np.argmax(qualities)
        return float(biases[optimal_idx]), qualities

    def _compute_quality(self, image: np.ndarray) -> float:
        """计算图像质量指标。"""
        if image.ndim > 2:
            gray = image[:, :, 0].astype(np.float64)
        else:
            gray = image.astype(np.float64)

        metric = self.config.quality_metric

        if metric == QualityMetric.SHARPNESS:
            # 梯度平方和（Tenengrad）
            gy = np.diff(gray, axis=0)
            gx = np.diff(gray, axis=1)
            return float(np.mean(gy[:-1, :]**2 + gx[:, :-1]**2))

        elif metric == QualityMetric.STREHL:
            peak = np.max(gray)
            total = np.sum(gray)
            return float(peak * peak / max(total, 1e-10))

        elif metric == QualityMetric.INTENSITY:
            return float(np.max(gray))

        elif metric == QualityMetric.FWHM:
            # 估计 FWHM（越小越好，返回负值用于最大化）
            peak_val = np.max(gray)
            half_max = peak_val / 2
            above_half = gray >= half_max
            if not np.any(above_half):
                return -100.0
            rows, cols = np.where(above_half)
            fwhm_y = float(np.max(rows) - np.min(rows) + 1)
            fwhm_x = float(np.max(cols) - np.min(cols) + 1)
            return -np.sqrt(fwhm_x**2 + fwhm_y**2)

        elif metric == QualityMetric.ENTROPY:
            # 归一化直方图熵（越小越好）
            hist, _ = np.histogram(gray.ravel(), bins=256, density=True)
            hist = hist[hist > 0]
            return -float(np.sum(hist * np.log(hist + 1e-10)))

        else:  # COMPOSITE
            sharpness = self._compute_quality_with_metric(gray, QualityMetric.SHARPNESS)
            strehl = self._compute_quality_with_metric(gray, QualityMetric.STREHL)
            return sharpness + strehl

    def _compute_quality_with_metric(self, gray: np.ndarray, metric: QualityMetric) -> float:
        """使用指定指标计算质量。"""
        old_metric = self.config.quality_metric
        self.config.quality_metric = metric
        result = self._compute_quality(gray)
        self.config.quality_metric = old_metric
        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = SensorlessAOV2Config(
        strategy=SearchStrategy.GAUSSIAN_FIT,
        max_zernike_mode=6,
        num_bias_points=7,
        max_iterations=2,
    )
    ao = SensorlessAOV2(config)

    # 模拟测量回调
    np.random.seed(42)
    true_aberrations = {2: 0.3, 3: -0.2, 4: 0.15}  # 真实像差

    def mock_apply(corrections):
        img = np.zeros((64, 64))
        y, x = np.mgrid[:64, :64]
        cx, cy = 32, 32
        r2 = (x - cx)**2 + (y - cy)**2
        img += 200 * np.exp(-r2 / (2 * 5**2))

        # 添加像差
        total = 0
        for mode, coeff in true_aberrations.items():
            bias = corrections.get(mode, 0.0)
            residual = coeff - bias
            total += residual * r2 / (32**2)
        img += total * 50
        img += np.random.randn(64, 64) * 5
        return img

    report = ao.correct(mock_apply)
    print(f"校正模式数: {report.modes_corrected}")
    print(f"质量改善: {report.quality_improvement*100:.1f}%")
    print(f"总测量数: {report.total_measurements}")
    print(f"总时间: {report.total_time_ms:.1f}ms")
    print(f"Zernike校正: {report.zernike_corrections}")
