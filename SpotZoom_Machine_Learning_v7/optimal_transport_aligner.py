"""
最优传输对齐器 (Optimal Transport Aligner)

基于最优传输理论的光斑分布对齐质量评估与最优步长计算模块。
使用 Sinkhorn 算法计算当前光斑分布与目标分布之间的最优传输距离，
用于对齐质量评估和收敛预测。

灵感来源:
- POT / PythonOT (PythonOT/POT): Python 最优传输库
  (https://github.com/PythonOT/POT)
- Sinkhorn 算法 (Cuturi 2013): 熵正则化最优传输
- Optimal Transport for Machine Learning (Peyre & Cuturi 2019)

算法原理:
  给定两个离散分布 (当前光斑分布 μ 和目标分布 ν):
  1. 构建传输代价矩阵 C (基于光斑位置距离)
  2. 求解熵正则化最优传输问题:
     min <T, C> - ε * H(T)
     其中 H(T) 为传输计划的熵
  3. 使用 Sinkhorn 迭代求解对偶变量
  4. 传输距离用于评估对齐质量
  5. 传输计划用于诊断对齐偏差方向

外部依赖: numpy
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from enum import Enum

logger = logging.getLogger(__name__)


class TransportCostMetric(Enum):
    """传输代价度量枚举。"""
    EUCLIDEAN = "euclidean"       # 欧氏距离
    SQUARED = "squared"           # 平方距离
    ABSOLUTE = "absolute"         # 绝对距离
    ANGULAR = "angular"           # 角度距离 (适用于角度参数)


@dataclass
class OTAlignmentResult:
    """最优传输对齐结果。"""
    ot_distance: float = 0.0              # 最优传输距离 (Wasserstein)
    convergence_iterations: int = 0       # 收敛所需迭代次数
    converged: bool = False               # 是否收敛
    alignment_quality: float = 0.0        # 对齐质量评分 (0-1, 1=完美)
    transport_plan: np.ndarray = None     # 传输计划矩阵
    optimal_step: np.ndarray = None       # 建议的最优步长
    step_confidence: float = 0.0          # 步长置信度 (0-1)
    entropy_regularization: float = 0.0   # 实际使用的熵正则化系数
    marginal_error_source: float = 0.0    # 源分布边际误差
    marginal_error_target: float = 0.0    # 目标分布边际误差
    processing_time_ms: float = 0.0       # 处理耗时 (ms)


@dataclass
class OTAlignerConfig:
    """最优传输对齐器配置。"""
    # Sinkhorn 算法参数
    entropy_regularization: float = 0.1    # 熵正则化系数 ε
    max_iterations: int = 500              # 最大迭代次数
    convergence_threshold: float = 1e-6    # 收敛阈值

    # 代价度量
    transport_cost_metric: TransportCostMetric = TransportCostMetric.SQUARED

    # 自适应参数
    adaptive_regularization: bool = True   # 根据噪声自适应调整 ε
    min_regularization: float = 0.01       # 最小正则化系数
    max_regularization: float = 1.0        # 最大正则化系数

    # 对齐评估
    quality_threshold_excellent: float = 0.95  # 优秀对齐阈值
    quality_threshold_good: float = 0.8        # 良好对齐阈值
    quality_threshold_poor: float = 0.5        # 较差对齐阈值

    # 步长计算
    step_scale_factor: float = 0.5         # 步长缩放因子
    max_step_size: float = 10.0            # 最大步长

    # 数值稳定性
    log_domain: bool = True                # 使用对数域计算 (防止数值溢出)
    numerical_stability: float = 1e-16     # 数值稳定性常数


class OptimalTransportAligner:
    """最优传输对齐器。

    使用 Sinkhorn 算法计算光斑分布之间的最优传输距离，
    用于对齐质量评估和最优步长计算。

    使用示例:
        aligner = OptimalTransportAligner()
        result = aligner.compute_alignment(
            current_positions, target_positions,
            current_weights, target_weights
        )
        print(f"OT distance: {result.ot_distance:.4f}")
        print(f"Alignment quality: {result.alignment_quality:.2%}")
        if result.optimal_step is not None:
            print(f"Suggested step: {result.optimal_step}")
    """

    def __init__(self, config: Optional[OTAlignerConfig] = None):
        self._config = config or OTAlignerConfig()
        self._distance_history: List[float] = []

    @property
    def config(self) -> OTAlignerConfig:
        return self._config

    def compute_alignment(
        self,
        current_positions: np.ndarray,
        target_positions: np.ndarray,
        current_weights: Optional[np.ndarray] = None,
        target_weights: Optional[np.ndarray] = None,
        noise_level: float = 0.0,
    ) -> OTAlignmentResult:
        """计算最优传输对齐。

        Args:
            current_positions: 当前光斑位置 (N, D), D=2 或 3
            target_positions: 目标光斑位置 (M, D)
            current_weights: 当前分布权重 (N,), None=均匀分布
            target_weights: 目标分布权重 (M,), None=均匀分布
            noise_level: 噪声水平 (用于自适应正则化)

        Returns:
            OTAlignmentResult: 对齐结果
        """
        import time
        t0 = time.perf_counter()

        # 输入验证
        current_positions = np.asarray(current_positions, dtype=np.float64)
        target_positions = np.asarray(target_positions, dtype=np.float64)

        if current_positions.ndim == 1:
            current_positions = current_positions.reshape(-1, 1)
        if target_positions.ndim == 1:
            target_positions = target_positions.reshape(-1, 1)

        n_source = current_positions.shape[0]
        n_target = target_positions.shape[0]

        # 默认均匀权重
        if current_weights is None:
            current_weights = np.ones(n_source, dtype=np.float64) / n_source
        else:
            current_weights = np.asarray(current_weights, dtype=np.float64)
            current_weights /= current_weights.sum()

        if target_weights is None:
            target_weights = np.ones(n_target, dtype=np.float64) / n_target
        else:
            target_weights = np.asarray(target_weights, dtype=np.float64)
            target_weights /= target_weights.sum()

        # 自适应正则化
        epsilon = self._compute_adaptive_regularization(noise_level)

        # 构建代价矩阵
        cost_matrix = self._build_cost_matrix(current_positions, target_positions)

        # Sinkhorn 算法
        transport_plan, iterations, converged = self._sinkhorn(
            cost_matrix, current_weights, target_weights, epsilon
        )

        # 计算传输距离
        ot_distance = float(np.sum(transport_plan * cost_matrix))

        # 评估对齐质量
        quality = self._compute_alignment_quality(ot_distance, cost_matrix)

        # 计算最优步长
        optimal_step, step_conf = self._compute_optimal_step(
            transport_plan, current_positions, target_positions
        )

        # 边际误差
        marginal_source = np.sum(transport_plan, axis=1)
        marginal_target = np.sum(transport_plan, axis=0)
        marginal_error_source = float(
            np.max(np.abs(marginal_source - current_weights))
        )
        marginal_error_target = float(
            np.max(np.abs(marginal_target - target_weights))
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        self._distance_history.append(ot_distance)
        if len(self._distance_history) > 200:
            self._distance_history.pop(0)

        result = OTAlignmentResult(
            ot_distance=ot_distance,
            convergence_iterations=iterations,
            converged=converged,
            alignment_quality=quality,
            transport_plan=transport_plan,
            optimal_step=optimal_step,
            step_confidence=step_conf,
            entropy_regularization=epsilon,
            marginal_error_source=marginal_error_source,
            marginal_error_target=marginal_error_target,
            processing_time_ms=elapsed_ms,
        )

        logger.debug(
            f"OptimalTransportAligner: OT_dist={ot_distance:.4f}, "
            f"quality={quality:.2%}, converged={converged} "
            f"({iterations} iters, eps={epsilon:.4f}), time={elapsed_ms:.1f}ms"
        )

        return result

    def _build_cost_matrix(
        self, source: np.ndarray, target: np.ndarray
    ) -> np.ndarray:
        """构建传输代价矩阵。

        Args:
            source: 源分布位置 (N, D)
            target: 目标分布位置 (M, D)

        Returns:
            代价矩阵 (N, M)
        """
        cfg = self._config
        metric = cfg.transport_cost_metric

        # 欧氏距离矩阵
        diff = source[:, np.newaxis, :] - target[np.newaxis, :, :]
        euclidean_dist = np.sqrt(np.sum(diff ** 2, axis=2) + cfg.numerical_stability)

        if metric == TransportCostMetric.EUCLIDEAN:
            return euclidean_dist
        elif metric == TransportCostMetric.SQUARED:
            return euclidean_dist ** 2
        elif metric == TransportCostMetric.ABSOLUTE:
            return np.sum(np.abs(diff), axis=2)
        elif metric == TransportCostMetric.ANGULAR:
            # 角度距离 (假设最后一维是角度)
            angle_diff = diff[:, :, -1]
            return np.minimum(np.abs(angle_diff), 2 * np.pi - np.abs(angle_diff))
        else:
            return euclidean_dist ** 2

    def _compute_adaptive_regularization(self, noise_level: float) -> float:
        """根据噪声水平自适应调整熵正则化系数。

        噪声越高 -> 正则化越大 (更平滑的传输计划)
        噪声越低 -> 正则化越小 (更精确的传输距离)

        Args:
            noise_level: 噪声水平

        Returns:
            熵正则化系数 ε
        """
        cfg = self._config

        if not cfg.adaptive_regularization:
            return cfg.entropy_regularization

        # 线性映射: noise_level 0 -> min_eps, noise_level 1 -> max_eps
        noise_norm = np.clip(noise_level, 0, 1)
        epsilon = cfg.min_regularization + noise_norm * (
            cfg.max_regularization - cfg.min_regularization
        )
        return float(epsilon)

    def _sinkhorn(
        self,
        cost_matrix: np.ndarray,
        source_weights: np.ndarray,
        target_weights: np.ndarray,
        epsilon: float,
    ) -> Tuple[np.ndarray, int, bool]:
        """Sinkhorn 算法求解熵正则化最优传输。

        求解: min <T, C> - ε * H(T)
        s.t. T * 1 = a, T^T * 1 = b, T >= 0

        使用对偶变量 u, v 迭代更新:
            u = a / (K @ v)
            v = b / (K^T @ u)
        其中 K = exp(-C / ε)

        Args:
            cost_matrix: 代价矩阵 (N, M)
            source_weights: 源分布权重 (N,)
            target_weights: 目标分布权重 (M,)
            epsilon: 熵正则化系数

        Returns:
            (transport_plan, iterations, converged)
        """
        cfg = self._config
        K = np.exp(-cost_matrix / epsilon)

        # 初始化对偶变量
        u = np.ones_like(source_weights)
        v = np.ones_like(target_weights)

        converged = False
        iterations = 0

        for i in range(cfg.max_iterations):
            u_prev = u.copy()

            # Sinkhorn 迭代
            u = source_weights / (K @ v + cfg.numerical_stability)
            v = target_weights / (K.T @ u + cfg.numerical_stability)

            # 检查收敛
            if i % 10 == 0:
                change = np.max(np.abs(u - u_prev))
                if change < cfg.convergence_threshold:
                    converged = True
                    iterations = i + 1
                    break

        if not converged:
            iterations = cfg.max_iterations

        # 计算传输计划
        transport_plan = np.diag(u) @ K @ np.diag(v)

        return transport_plan, iterations, converged

    def _compute_alignment_quality(
        self, ot_distance: float, cost_matrix: np.ndarray
    ) -> float:
        """计算对齐质量评分。

        将 OT 距离归一化到 [0, 1] 区间:
        - 0 = 完全不对齐
        - 1 = 完美对齐

        Args:
            ot_distance: 最优传输距离
            cost_matrix: 代价矩阵

        Returns:
            对齐质量 (0-1)
        """
        cfg = self._config

        # 归一化: 使用代价矩阵的中位数作为参考尺度
        median_cost = np.median(cost_matrix)
        if median_cost < cfg.numerical_stability:
            return 1.0

        normalized_distance = ot_distance / median_cost

        # 转换为质量评分 (指数衰减)
        quality = np.exp(-normalized_distance)
        return float(np.clip(quality, 0, 1))

    def _compute_optimal_step(
        self,
        transport_plan: np.ndarray,
        current_positions: np.ndarray,
        target_positions: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], float]:
        """根据传输计划计算最优步长。

        步长 = 加权平均位移方向
        权重 = 传输计划中对应的最大耦合强度

        Args:
            transport_plan: 传输计划 (N, M)
            current_positions: 当前位置 (N, D)
            target_positions: 目标位置 (M, D)

        Returns:
            (optimal_step, confidence)
        """
        cfg = self._config

        # 对每个源点，找到其主要传输目标
        source_assignments = np.argmax(transport_plan, axis=1)
        assignment_weights = np.max(transport_plan, axis=1)

        # 计算位移
        displacements = (
            target_positions[source_assignments] - current_positions
        )

        # 加权平均位移
        total_weight = np.sum(assignment_weights)
        if total_weight < cfg.numerical_stability:
            return None, 0.0

        weighted_displacement = np.sum(
            displacements * assignment_weights[:, np.newaxis], axis=0
        ) / total_weight

        # 缩放步长
        step_magnitude = np.linalg.norm(weighted_displacement)
        if step_magnitude > cfg.max_step_size:
            weighted_displacement = (
                weighted_displacement / step_magnitude * cfg.max_step_size
            )
            step_magnitude = cfg.max_step_size

        optimal_step = weighted_displacement * cfg.step_scale_factor

        # 置信度: 基于传输计划的集中程度
        max_plan = np.max(transport_plan)
        mean_plan = np.mean(transport_plan)
        concentration = max_plan / (mean_plan + cfg.numerical_stability)
        confidence = float(np.clip(1.0 - 1.0 / concentration, 0, 1))

        return optimal_step, confidence

    def visualize_transport_plan(
        self,
        transport_plan: np.ndarray,
        current_positions: np.ndarray,
        target_positions: np.ndarray,
    ) -> np.ndarray:
        """可视化传输计划 (生成热力图)。

        将传输计划可视化为图像，用于诊断对齐偏差。

        Args:
            transport_plan: 传输计划矩阵 (N, M)
            current_positions: 当前位置 (N, 2)
            target_positions: 目标位置 (M, 2)

        Returns:
            可视化图像 (H, W, 3), uint8
        """
        try:
            import cv2
        except ImportError:
            logger.warning("cv2 not available, returning empty visualization")
            return np.zeros((100, 100, 3), dtype=np.uint8)

        # 创建画布
        all_pos = np.vstack([current_positions, target_positions])
        margin = 20
        x_min, y_min = np.min(all_pos, axis=0) - margin
        x_max, y_max = np.max(all_pos, axis=0) + margin
        canvas_w = int(x_max - x_min) + 1
        canvas_h = int(y_max - y_min) + 1
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 240

        # 绘制传输连线
        max_val = np.max(transport_plan)
        if max_val > 1e-10:
            normalized_plan = transport_plan / max_val

            # 只绘制主要的传输连线 (top 20%)
            threshold = np.percentile(normalized_plan, 80)
            for i in range(transport_plan.shape[0]):
                for j in range(transport_plan.shape[1]):
                    if normalized_plan[i, j] >= threshold:
                        intensity = int(normalized_plan[i, j] * 255)
                        color = (0, intensity, 255 - intensity)
                        pt1 = (int(current_positions[i, 0] - x_min),
                               int(current_positions[i, 1] - y_min))
                        pt2 = (int(target_positions[j, 0] - x_min),
                               int(target_positions[j, 1] - y_min))
                        cv2.line(canvas, pt1, pt2, color, 1)

        # 绘制源点 (蓝色)
        for pos in current_positions:
            cx = int(pos[0] - x_min)
            cy = int(pos[1] - y_min)
            cv2.circle(canvas, (cx, cy), 4, (255, 0, 0), -1)

        # 绘制目标点 (绿色)
        for pos in target_positions:
            cx = int(pos[0] - x_min)
            cy = int(pos[1] - y_min)
            cv2.circle(canvas, (cx, cy), 4, (0, 200, 0), -1)

        return canvas

    def predict_convergence(
        self, n_history: int = 10
    ) -> Tuple[float, float]:
        """基于 OT 距离历史预测收敛趋势。

        Args:
            n_history: 使用最近 N 个历史数据点

        Returns:
            (trend_slope, predicted_remaining_distance)
            trend_slope < 0 表示距离在减小 (正在收敛)
        """
        if len(self._distance_history) < n_history:
            return 0.0, float(self._distance_history[-1]) if self._distance_history else 0.0

        recent = np.array(self._distance_history[-n_history:])
        x = np.arange(len(recent), dtype=np.float64)

        # 线性拟合
        A = np.vstack([np.ones(len(x)), x]).T
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, recent, rcond=None)
            slope = coeffs[1]
            intercept = coeffs[0]
            predicted = intercept + slope * (len(recent) + 5)
            return float(slope), max(0.0, float(predicted))
        except np.linalg.LinAlgError:
            return 0.0, float(recent[-1])

    def reset(self):
        """重置对齐器状态。"""
        self._distance_history.clear()
        logger.info("OptimalTransportAligner: Reset")

    def get_distance_history(self) -> List[float]:
        """获取 OT 距离历史。"""
        return list(self._distance_history)
