"""
NeNA 精度评估器 (NeNAPrecisionAssessor)

基于 Picasso (https://github.com/jungmannlab/picasso) 的 NeNA 方法，
为 SpotZoom 提供实验级光斑定位精度评估。

灵感来源:
- Picasso NeNA: Nearest Neighbor based precision estimation
  (Schnitzbauer et al., Nature Protocols 2017)
  (https://github.com/jungmannlab/picasso)
- DECODE: 概率定位不确定性 (Speiser et al., Nature Methods 2021)

算法原理:
  1. 最近邻分析 (NeNA): 利用多次重复定位的最近邻距离统计
     来估计实际定位精度，无需已知真实位置
  2. Cramér-Rao 下界 (CRB): 基于噪声模型的理论最优精度估计
  3. 交叉验证: 将定位结果分组进行交叉精度评估
  4. 时序精度: 基于位置时间序列的 Allan 方差分析

与现有模块的关系:
  - 增强 v1/spot_quality.py 的精度评估维度
  - 与 uncertainty_aware_localizer_v2.py 的不确定性估计互补
  - 与 v1/spectral_analyzer.py 的 Allan 方差分析协同

外部依赖: numpy, scipy (可选)
"""

import numpy as np
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from enum import Enum

logger = logging.getLogger(__name__)


class PrecisionMethod(Enum):
    """精度评估方法。"""
    NENA = "nena"               # 最近邻分析
    CRB = "crb"                 # Cramér-Rao 下界
    ALLAN = "allan"             # Allan 方差
    REPEATABILITY = "repeatability"  # 重复性分析
    COMPREHENSIVE = "comprehensive"  # 综合评估


@dataclass
class NeNAConfig:
    """NeNA 精度评估配置。"""
    # NeNA 参数
    nena_min_samples: int = 30       # 最小样本数
    nena_percentile: float = 0.25    # 最近邻百分位数
    nena_bootstrap_iterations: int = 1000  # Bootstrap 迭代次数

    # CRB 参数
    crb_background_method: str = "median"  # median / percentile
    crb_background_percentile: float = 25.0

    # Allan 方差参数
    allan_min_tau: float = 1.0       # 最小时间常数
    allan_max_tau: float = 100.0     # 最大时间常数
    allan_num_points: int = 50       # 采样点数

    # 重复性参数
    repeatability_groups: int = 5    # 重复组数
    repeatability_min_per_group: int = 10

    # 输出
    compute_confidence: bool = True  # 计算置信区间
    confidence_level: float = 0.95   # 置信水平


@dataclass
class NeNAReport:
    """精度评估报告。"""
    method_used: str = ""

    # NeNA 结果
    nena_precision_x: float = 0.0    # x 方向精度 (nm 或 px)
    nena_precision_y: float = 0.0    # y 方向精度
    nena_precision_2d: float = 0.0   # 2D 精度
    nena_confidence_interval: Optional[Tuple[float, float]] = None

    # CRB 结果
    crb_precision_x: float = 0.0
    crb_precision_y: float = 0.0
    crb_snr: float = 0.0

    # Allan 方差结果
    allan_min_precision: float = 0.0  # 最优时间常数处的精度
    allan_optimal_tau: float = 0.0    # 最优时间常数
    allan_noise_floor: float = 0.0    # 噪声基底

    # 重复性结果
    repeatability_x: float = 0.0
    repeatability_y: float = 0.0

    # 综合评估
    overall_precision: float = 0.0
    precision_grade: str = ""        # A/B/C/D/F

    # 元信息
    num_samples: int = 0
    processing_time_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)


class NeNAPrecisionAssessor:
    """NeNA 精度评估器。

    使用多种方法评估光斑定位精度，包括最近邻分析、
    Cramér-Rao 下界、Allan 方差和重复性分析。

    Parameters
    ----------
    config : NeNAConfig
        评估器配置。
    """

    def __init__(self, config: Optional[NeNAConfig] = None):
        self.config = config or NeNAConfig()

    def assess(
        self,
        positions: np.ndarray,
        pixel_size: float = 1.0,
        timestamps: Optional[np.ndarray] = None,
        method: PrecisionMethod = PrecisionMethod.COMPREHENSIVE,
    ) -> NeNAReport:
        """评估定位精度。

        Parameters
        ----------
        positions : np.ndarray
            定位位置数组，形状 (N, 2) 或 (N, 3)，列为 [x, y] 或 [x, y, t]。
        pixel_size : float
            像素尺寸（用于转换为物理单位）。
        timestamps : np.ndarray, optional
            时间戳数组，形状 (N,)。
        method : PrecisionMethod
            评估方法。

        Returns
        -------
        NeNAReport
            精度评估报告。
        """
        t0 = time.perf_counter()
        report = NeNAReport(method_used=method.value)
        report.num_samples = len(positions)

        if len(positions) < self.config.nena_min_samples:
            report.warnings.append(
                f"样本数不足 ({len(positions)} < {self.config.nena_min_samples})"
            )
            report.precision_grade = "F"
            report.processing_time_ms = (time.perf_counter() - t0) * 1000
            return report

        pos = positions[:, :2] * pixel_size  # 转换为物理单位

        if method in (PrecisionMethod.NENA, PrecisionMethod.COMPREHENSIVE):
            self._nena_analysis(pos, report)

        if method in (PrecisionMethod.CRB, PrecisionMethod.COMPREHENSIVE):
            # CRB 需要图像数据，此处使用位置统计近似
            self._crb_approximation(pos, report)

        if method in (PrecisionMethod.ALLAN, PrecisionMethod.COMPREHENSIVE):
            if timestamps is not None:
                self._allan_analysis(pos, timestamps, report)
            else:
                report.warnings.append("Allan 方差需要时间戳数据")

        if method in (PrecisionMethod.REPEATABILITY, PrecisionMethod.COMPREHENSIVE):
            self._repeatability_analysis(pos, report)

        # 综合评估
        self._compute_overall(report)

        report.processing_time_ms = (time.perf_counter() - t0) * 1000
        return report

    def _nena_analysis(self, positions: np.ndarray, report: NeNAReport):
        """NeNA 最近邻精度分析。"""
        n = len(positions)

        # 计算每个点到其最近邻的距离
        nn_distances = np.zeros(n)
        nn_dx = np.zeros(n)
        nn_dy = np.zeros(n)

        for i in range(n):
            diffs = positions - positions[i]
            dists = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)
            dists[i] = np.inf  # 排除自身
            nn_idx = np.argmin(dists)
            nn_distances[i] = dists[nn_idx]
            nn_dx[i] = abs(diffs[nn_idx, 0])
            nn_dy[i] = abs(diffs[nn_idx, 1])

        # 使用百分位数估计精度
        percentile = self.config.nena_percentile
        nn_x = np.percentile(nn_dx, percentile * 100)
        nn_y = np.percentile(nn_dy, percentile * 100)
        nn_2d = np.percentile(nn_distances, percentile * 100)

        # NeNA 精度 = 最近邻距离 / sqrt(2)
        report.nena_precision_x = nn_x / np.sqrt(2)
        report.nena_precision_y = nn_y / np.sqrt(2)
        report.nena_precision_2d = nn_2d / np.sqrt(2)

        # Bootstrap 置信区间
        if self.config.compute_confidence:
            bootstrap_precisions = []
            for _ in range(self.config.bootstrap_iterations):
                idx = np.random.choice(n, n, replace=True)
                boot_pos = positions[idx]
                boot_dists = np.zeros(n)
                for i in range(n):
                    diffs = boot_pos - boot_pos[i]
                    dists = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)
                    dists[i] = np.inf
                    boot_dists[i] = np.min(dists)
                bootstrap_precisions.append(np.percentile(boot_dists, percentile * 100) / np.sqrt(2))

            alpha = 1 - self.config.confidence_level
            report.nena_confidence_interval = (
                float(np.percentile(bootstrap_precisions, alpha/2 * 100)),
                float(np.percentile(bootstrap_precisions, (1-alpha/2) * 100)),
            )

    def _crb_approximation(self, positions: np.ndarray, report: NeNAReport):
        """Cramér-Rao 下界近似估计。"""
        # 基于位置方差的 CRB 近似
        # 对于静态光斑: CRB ≈ σ_position / sqrt(N)
        std_x = np.std(positions[:, 0])
        std_y = np.std(positions[:, 1])
        n = len(positions)

        report.crb_precision_x = std_x / np.sqrt(n) * np.sqrt(2)
        report.crb_precision_y = std_y / np.sqrt(n) * np.sqrt(2)

        # SNR 近似
        mean_pos = np.mean(positions, axis=0)
        spread = np.sqrt(np.mean((positions - mean_pos)**2))
        report.crb_snr = 1.0 / max(spread, 1e-10)

    def _allan_analysis(
        self, positions: np.ndarray, timestamps: np.ndarray, report: NeNAReport
    ):
        """Allan 方差分析。"""
        dt = np.median(np.diff(timestamps))
        if dt <= 0:
            report.warnings.append("时间戳无效")
            return

        taus = np.logspace(
            np.log10(self.config.allan_min_tau * dt),
            np.log10(min(self.config.allan_max_tau * dt, timestamps[-1] - timestamps[0])),
            self.config.allan_num_points
        )

        allan_var_x = []
        allan_var_y = []

        for tau in taus:
            m = max(1, int(tau / dt))
            if m < 1:
                continue

            # 计算 Allan 方差
            clusters = np.arange(0, len(positions) - 2*m, m)
            if len(clusters) < 2:
                continue

            var_x, var_y = 0.0, 0.0
            for k in clusters:
                diff_x = positions[k+2*m, 0] - 2*positions[k+m, 0] + positions[k, 0]
                diff_y = positions[k+2*m, 1] - 2*positions[k+m, 1] + positions[k, 1]
                var_x += diff_x**2
                var_y += diff_y**2

            var_x /= 2 * len(clusters)
            var_y /= 2 * len(clusters)
            allan_var_x.append(var_x)
            allan_var_y.append(var_y)

        if allan_var_x:
            allan_var_x = np.array(allan_var_x)
            allan_var_y = np.array(allan_var_y)
            allan_dev = np.sqrt((allan_var_x + allan_var_y) / 2)

            min_idx = np.argmin(allan_dev)
            report.allan_min_precision = float(allan_dev[min_idx])
            report.allan_optimal_tau = float(taus[min_idx])
            report.allan_noise_floor = float(np.min(allan_dev[-3:]) if len(allan_dev) > 3 else allan_dev[-1])

    def _repeatability_analysis(self, positions: np.ndarray, report: NeNAReport):
        """重复性分析。"""
        n = len(positions)
        group_size = n // self.config.repeatability_groups

        if group_size < self.config.repeatability_min_per_group:
            report.warnings.append("每组样本不足，跳过重复性分析")
            return

        group_stds_x = []
        group_stds_y = []

        for i in range(self.config.repeatability_groups):
            start = i * group_size
            end = start + group_size
            group = positions[start:end]
            group_stds_x.append(np.std(group[:, 0]))
            group_stds_y.append(np.std(group[:, 1]))

        report.repeatability_x = float(np.mean(group_stds_x))
        report.repeatability_y = float(np.mean(group_stds_y))

    def _compute_overall(self, report: NeNAReport):
        """综合评估。"""
        precisions = []

        if report.nena_precision_2d > 0:
            precisions.append(report.nena_precision_2d)
        if report.crb_precision_x > 0:
            precisions.append(np.sqrt(report.crb_precision_x**2 + report.crb_precision_y**2))
        if report.allan_min_precision > 0:
            precisions.append(report.allan_min_precision)
        if report.repeatability_x > 0:
            precisions.append(np.sqrt(report.repeatability_x**2 + report.repeatability_y**2))

        if precisions:
            report.overall_precision = float(np.mean(precisions))

        # 精度等级
        p = report.overall_precision
        if p < 0.1:
            report.precision_grade = "A"  # 亚像素级
        elif p < 0.3:
            report.precision_grade = "B"  # 高精度
        elif p < 0.5:
            report.precision_grade = "C"  # 中等精度
        elif p < 1.0:
            report.precision_grade = "D"  # 低精度
        else:
            report.precision_grade = "F"  # 不可接受


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = NeNAConfig()
    assessor = NeNAPrecisionAssessor(config)

    # 模拟定位数据
    np.random.seed(42)
    n_samples = 200
    true_x, true_y = 100.0, 100.0
    positions = np.column_stack([
        true_x + np.random.randn(n_samples) * 0.15,
        true_y + np.random.randn(n_samples) * 0.18,
    ])
    timestamps = np.arange(n_samples) * 0.01  # 10ms 间隔

    report = assessor.assess(positions, pixel_size=100.0, timestamps=timestamps)
    print(f"NeNA 精度: x={report.nena_precision_x:.2f}nm, y={report.nena_precision_y:.2f}nm")
    print(f"CRB 精度: x={report.crb_precision_x:.2f}nm, y={report.crb_precision_y:.2f}nm")
    print(f"Allan 最优精度: {report.allan_min_precision:.2f}nm (τ={report.allan_optimal_tau:.3f}s)")
    print(f"重复性: x={report.repeatability_x:.2f}nm, y={report.repeatability_y:.2f}nm")
    print(f"综合精度: {report.overall_precision:.2f}nm, 等级: {report.precision_grade}")
