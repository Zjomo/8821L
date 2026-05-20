"""
拓扑感知优化器 (Topology Aware Optimizer)

基于 CMA-ES (协方差矩阵自适应进化策略) 的全局优化模块。
用于多参数光学对齐的全局搜索，支持多目标 Pareto 前沿估计、
约束处理和自适应重启策略。

灵感来源:
- CMA-ES (CMA-ES/pycma): 协方差矩阵自适应进化策略
  (https://github.com/CMA-ES/pycma)
- Bayesian Optimization (fmfn/BayesianOptimization): 贝叶斯优化
- DEAP (DEAP): 进化算法框架 (https://github.com/DEAP/deap)
- Platypus: 多目标优化库

算法原理:
  CMA-ES 是一种二阶黑箱优化算法:
  1. 维护一个搜索分布 N(m, C*σ²)
  2. 每代采样 λ 个候选解
  3. 根据适应度排序，更新均值 m (加权重组)
  4. 自适应更新协方差矩阵 C (搜索步形状)
  5. 自适应更新步长 σ (搜索步大小)

  多目标扩展:
  - Pareto 非支配排序
  - 拥挤距离计算
  - 约束违反惩罚

外部依赖: numpy
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Callable, Dict
from enum import Enum

logger = logging.getLogger(__name__)


class RestartStrategy(Enum):
    """重启策略枚举。"""
    NONE = "none"               # 不重启
    INCREASE = "increase"       # 增加种群大小重启
    IPOP = "ipop"               # IPOP 策略 (双倍种群)
    BIPOP = "bipop"             # BIPOP 策略 (交替大小)


class ConstraintMethod(Enum):
    """约束处理方法枚举。"""
    PENALTY = "penalty"         # 惩罚函数法
    BARRIER = "barrier"         # 屏障函数法
    DEATH = "death"             # 死亡惩罚 (不可行解适应度=inf)


@dataclass
class ParetoPoint:
    """Pareto 前沿上的点。"""
    parameters: np.ndarray = None       # 参数向量
    objectives: np.ndarray = None       # 目标值
    constraint_violation: float = 0.0   # 约束违反量
    rank: int = 0                       # Pareto 排名
    crowding_distance: float = 0.0      # 拥挤距离


@dataclass
class TopologyOptResult:
    """拓扑优化结果。"""
    best_parameters: np.ndarray = None      # 最优参数
    best_objective: float = np.inf          # 最优目标值
    best_objectives: np.ndarray = None      # 多目标最优值
    pareto_front: List[ParetoPoint] = field(default_factory=list)  # Pareto 前沿
    convergence_history: List[float] = field(default_factory=list)  # 收敛历史
    generations_used: int = 0               # 使用的代数
    evaluations_used: int = 0               # 使用的评估次数
    final_step_size: float = 0.0            # 最终步长
    final_condition: float = 0.0            # 最终停止条件值
    restarts_performed: int = 0             # 重启次数
    converged: bool = False                 # 是否收敛
    processing_time_ms: float = 0.0         # 处理耗时 (ms)


@dataclass
class TopologyOptimizerConfig:
    """拓扑感知优化器配置。"""
    # CMA-ES 参数
    population_size: Optional[int] = None   # 种群大小 (None=自动: 4+3*ln(n))
    max_generations: int = 500              # 最大代数
    initial_step_size: float = 0.5          # 初始步长 σ

    # 重启策略
    restart_strategy: RestartStrategy = RestartStrategy.IPOP
    max_restarts: int = 5                   # 最大重启次数

    # 约束处理
    constraint_method: ConstraintMethod = ConstraintMethod.PENALTY
    penalty_factor: float = 100.0           # 惩罚因子

    # 多目标
    objective_weights: Optional[np.ndarray] = None  # 目标权重 (None=等权)

    # 收敛判定
    tolerance: float = 1e-8                 # 收敛容限
    stagnation_generations: int = 50        # 停滞代数阈值

    # 边界约束
    lower_bounds: Optional[np.ndarray] = None   # 下界
    upper_bounds: Optional[np.ndarray] = None   # 上界

    # 数值参数
    epsilon: float = 1e-10                  # 数值稳定性常数


class TopologyAwareOptimizer:
    """拓扑感知优化器。

    基于 CMA-ES 的全局优化，支持多目标 Pareto 优化和约束处理。
    纯 numpy 实现，适用于多参数光学对齐优化。

    使用示例:
        def objective(x):
            return (x[0] - 1.0)**2 + (x[1] - 2.0)**2

        optimizer = TopologyAwareOptimizer(
            lower_bounds=np.array([-5, -5]),
            upper_bounds=np.array([5, 5])
        )
        result = optimizer.optimize(objective, dim=2)
        print(f"Best: {result.best_parameters}, obj={result.best_objective:.6f}")
    """

    def __init__(self, config: Optional[TopologyOptimizerConfig] = None):
        self._config = config or TopologyOptimizerConfig()
        self._generation: int = 0
        self._evaluations: int = 0
        self._restart_count: int = 0

    @property
    def config(self) -> TopologyOptimizerConfig:
        return self._config

    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        dim: int,
        x0: Optional[np.ndarray] = None,
        multi_objective: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> TopologyOptResult:
        """执行优化。

        Args:
            objective: 单目标函数 f(x) -> float
            dim: 参数维度
            x0: 初始点 (None=随机)
            multi_objective: 多目标函数 f(x) -> (n_obj,) (可选)

        Returns:
            TopologyOptResult: 优化结果
        """
        import time
        t0 = time.perf_counter()

        cfg = self._config

        # 设置边界
        if cfg.lower_bounds is not None:
            lb = np.asarray(cfg.lower_bounds, dtype=np.float64)
        else:
            lb = np.full(dim, -10.0, dtype=np.float64)
        if cfg.upper_bounds is not None:
            ub = np.asarray(cfg.upper_bounds, dtype=np.float64)
        else:
            ub = np.full(dim, 10.0, dtype=np.float64)

        # 初始点
        if x0 is not None:
            mean = np.asarray(x0, dtype=np.float64).ravel()
        else:
            mean = (lb + ub) / 2.0 + np.random.uniform(-0.1, 0.1, dim)

        # 种群大小
        pop_size = cfg.population_size or int(4 + 3 * np.log(dim))
        pop_size = max(10, pop_size)

        # 初始步长
        sigma = cfg.initial_step_size

        # CMA-ES 状态
        C = np.eye(dim, dtype=np.float64)  # 协方差矩阵
        p_sigma = np.zeros(dim, dtype=np.float64)  # 进化路径 (σ)
        p_c = np.zeros(dim, dtype=np.float64)      # 进化路径 (C)

        # CMA-ES 参数
        mu = pop_size // 2  # 父代数量
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1, dtype=np.float64))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2.0) / (dim + mu_eff + 5.0)
        d_sigma = 1.0 + 2.0 * max(0, np.sqrt((mu_eff - 1.0) / (dim + 1.0)) - 1.0) + c_sigma
        c_c = (4.0 + mu_eff / dim) / (dim + 4.0 + 2.0 * mu_eff / dim)
        c_1 = 2.0 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1.0 - c_1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((dim + 2.0) ** 2 + mu_eff))

        # 优化循环
        best_obj = np.inf
        best_params = mean.copy()
        convergence_history = []
        stagnation_count = 0
        prev_best = np.inf
        restart = 0

        # Pareto 前沿 (多目标)
        pareto_front: List[ParetoPoint] = []

        for gen in range(cfg.max_generations):
            self._generation = gen + 1

            # 采样
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # 协方差矩阵不正定，重置
                C = np.eye(dim, dtype=np.float64)
                L = np.eye(dim, dtype=np.float64)

            z = np.random.randn(pop_size, dim)
            population = mean + sigma * (z @ L.T)

            # 边界处理
            population = np.clip(population, lb, ub)

            # 评估
            fitness = np.full(pop_size, np.inf)
            for i in range(pop_size):
                x = population[i]
                # 约束违反
                violation = self._compute_constraint_violation(x, lb, ub)
                if violation > 0:
                    if cfg.constraint_method == ConstraintMethod.DEATH:
                        fitness[i] = np.inf
                        self._evaluations += 1
                        continue
                    elif cfg.constraint_method == ConstraintMethod.PENALTY:
                        raw_obj = objective(x)
                        fitness[i] = raw_obj + cfg.penalty_factor * violation
                    elif cfg.constraint_method == ConstraintMethod.BARRIER:
                        fitness[i] = np.inf
                        self._evaluations += 1
                        continue
                else:
                    fitness[i] = objective(x)

                # 多目标评估
                if multi_objective is not None:
                    obj_values = multi_objective(x)
                    point = ParetoPoint(
                        parameters=x.copy(),
                        objectives=obj_values,
                        constraint_violation=violation,
                    )
                    pareto_front = self._update_pareto_front(pareto_front, point)

                self._evaluations += 1

            # 排序
            sorted_indices = np.argsort(fitness)
            sorted_population = population[sorted_indices]

            # 更新最优
            if fitness[sorted_indices[0]] < best_obj:
                best_obj = fitness[sorted_indices[0]]
                best_params = sorted_population[0].copy()
                stagnation_count = 0
            else:
                stagnation_count += 1

            convergence_history.append(best_obj)

            # 选择父代
            selected = sorted_population[:mu]

            # 更新均值
            old_mean = mean.copy()
            mean = np.sum(weights[:, np.newaxis] * selected, axis=0)

            # 更新进化路径
            y_w = (mean - old_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(
                c_sigma * (2 - c_sigma) * mu_eff
            ) * (C @ y_w)

            h_sigma = 1.0 if (
                np.linalg.norm(p_sigma) / np.sqrt(
                    1 - (1 - c_sigma) ** (2 * (gen + 1))
                ) < (1.4 + 2.0 / (dim + 1)) * np.sqrt(dim)
            ) else 0.0

            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(
                c_c * (2 - c_c) * mu_eff
            ) * y_w

            # 更新协方差矩阵
            rank_one_update = np.outer(p_c, p_c)
            rank_mu_update = np.zeros((dim, dim), dtype=np.float64)
            for i in range(mu):
                y_i = (selected[i] - old_mean) / sigma
                rank_mu_update += weights[i] * np.outer(y_i, y_i)

            C = ((1 - c_1 - c_mu) * C +
                 c_1 * rank_one_update +
                 c_mu * rank_mu_update)

            # 更新步长
            sigma *= np.exp(
                (c_sigma / 2) * (
                    np.linalg.norm(p_sigma) ** 2 / dim - 1
                ) - (c_sigma / d_sigma)
            )

            # 收敛检查
            if best_obj < cfg.tolerance:
                logger.info(f"Converged at generation {gen + 1}")
                break

            # 停滞检查 -> 重启
            if stagnation_count >= cfg.stagnation_generations:
                if (cfg.restart_strategy != RestartStrategy.NONE and
                        restart < cfg.max_restarts):
                    restart += 1
                    self._restart_count += 1
                    logger.info(
                        f"Restart #{restart} at generation {gen + 1}, "
                        f"best={best_obj:.6f}"
                    )

                    if cfg.restart_strategy == RestartStrategy.IPOP:
                        pop_size *= 2
                    elif cfg.restart_strategy == RestartStrategy.BIPOP:
                        if restart % 2 == 0:
                            pop_size *= 2

                    # 重新计算权重
                    mu = pop_size // 2
                    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1, dtype=np.float64))
                    weights /= np.sum(weights)
                    mu_eff = 1.0 / np.sum(weights ** 2)

                    # 重置部分状态
                    C = np.eye(dim, dtype=np.float64)
                    p_sigma = np.zeros(dim, dtype=np.float64)
                    p_c = np.zeros(dim, dtype=np.float64)
                    sigma = cfg.initial_step_size
                    stagnation_count = 0

                    # 在最优解附近重新开始
                    mean = best_params + np.random.normal(0, sigma, dim)
                    mean = np.clip(mean, lb, ub)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # 多目标加权聚合
        if multi_objective is not None and pareto_front:
            best_pareto = self._select_best_pareto(pareto_front, cfg.objective_weights)
            best_obj = float(np.sum(best_pareto.objectives))
            best_params = best_pareto.parameters
        elif cfg.objective_weights is not None and multi_objective is not None:
            best_obj = float(best_obj)

        result = TopologyOptResult(
            best_parameters=best_params,
            best_objective=float(best_obj),
            pareto_front=pareto_front,
            convergence_history=convergence_history,
            generations_used=self._generation,
            evaluations_used=self._evaluations,
            final_step_size=float(sigma),
            restarts_performed=self._restart_count,
            converged=best_obj < cfg.tolerance,
            processing_time_ms=elapsed_ms,
        )

        logger.info(
            f"TopologyAwareOptimizer: best_obj={best_obj:.6f}, "
            f"generations={self._generation}, evaluations={self._evaluations}, "
            f"restarts={self._restart_count}, time={elapsed_ms:.1f}ms"
        )

        return result

    def _compute_constraint_violation(
        self, x: np.ndarray, lb: np.ndarray, ub: np.ndarray
    ) -> float:
        """计算约束违反量。

        Args:
            x: 参数向量
            lb: 下界
            ub: 上界

        Returns:
            约束违反量 (0=无违反)
        """
        violation = 0.0
        below = lb - x
        above = x - ub
        violation += np.sum(np.maximum(0, below) ** 2)
        violation += np.sum(np.maximum(0, above) ** 2)
        return float(np.sqrt(violation))

    def _update_pareto_front(
        self,
        front: List[ParetoPoint],
        new_point: ParetoPoint,
    ) -> List[ParetoPoint]:
        """更新 Pareto 前沿。

        Args:
            front: 当前 Pareto 前沿
            new_point: 新点

        Returns:
            更新后的 Pareto 前沿
        """
        # 检查新点是否被支配
        is_dominated = False
        for point in front:
            if self._dominates(point, new_point):
                is_dominated = True
                break

        if is_dominated:
            return front

        # 移除被新点支配的点
        front = [p for p in front if not self._dominates(new_point, p)]

        # 添加新点
        front.append(new_point)

        # 限制前沿大小
        if len(front) > 100:
            # 按拥挤距离排序，保留最分散的点
            front = self._prune_pareto(front, 100)

        return front

    def _dominates(self, a: ParetoPoint, b: ParetoPoint) -> bool:
        """检查 a 是否支配 b。

        a 支配 b 当且仅当:
        - a 在所有目标上不差于 b
        - a 在至少一个目标上严格优于 b

        Args:
            a: Pareto 点 a
            b: Pareto 点 b

        Returns:
            True 如果 a 支配 b
        """
        if a.objectives is None or b.objectives is None:
            return False

        # 约束违反处理
        if a.constraint_violation > 0 and b.constraint_violation == 0:
            return False
        if a.constraint_violation == 0 and b.constraint_violation > 0:
            return True
        if a.constraint_violation > 0 and b.constraint_violation > 0:
            return a.constraint_violation < b.constraint_violation

        n_obj = min(a.objectives.shape[0], b.objectives.shape[0])
        at_least_one_better = False
        for i in range(n_obj):
            if a.objectives[i] > b.objectives[i]:
                return False
            if a.objectives[i] < b.objectives[i]:
                at_least_one_better = True

        return at_least_one_better

    def _prune_pareto(
        self, front: List[ParetoPoint], max_size: int
    ) -> List[ParetoPoint]:
        """修剪 Pareto 前沿 (按拥挤距离)。

        Args:
            front: Pareto 前沿
            max_size: 最大大小

        Returns:
            修剪后的前沿
        """
        if len(front) <= max_size:
            return front

        # 计算拥挤距离
        n_obj = front[0].objectives.shape[0] if front[0].objectives is not None else 1
        n = len(front)

        for point in front:
            point.crowding_distance = 0.0

        for m in range(n_obj):
            # 按第 m 个目标排序
            sorted_front = sorted(front, key=lambda p: p.objectives[m] if p.objectives is not None else 0)
            sorted_front[0].crowding_distance = np.inf
            sorted_front[-1].crowding_distance = np.inf

            obj_range = sorted_front[-1].objectives[m] - sorted_front[0].objectives[m]
            if obj_range < self._config.epsilon:
                continue

            for i in range(1, n - 1):
                sorted_front[i].crowding_distance += (
                    sorted_front[i + 1].objectives[m] - sorted_front[i - 1].objectives[m]
                ) / obj_range

        # 按拥挤距离排序，保留最分散的
        front.sort(key=lambda p: p.crowding_distance, reverse=True)
        return front[:max_size]

    def _select_best_pareto(
        self,
        front: List[ParetoPoint],
        weights: Optional[np.ndarray] = None,
    ) -> ParetoPoint:
        """从 Pareto 前沿中选择最佳点。

        使用加权和或膝点法。

        Args:
            front: Pareto 前沿
            weights: 目标权重

        Returns:
            最佳 Pareto 点
        """
        if not front:
            return ParetoPoint()

        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64)
            weights = weights / (np.sum(weights) + 1e-10)

            best_point = front[0]
            best_score = np.inf

            for point in front:
                if point.objectives is None:
                    continue
                score = np.sum(weights * point.objectives)
                if score < best_score:
                    best_score = score
                    best_point = point

            return best_point
        else:
            # 膝点法: 选择到理想点最近的点
            ideal = np.min([p.objectives for p in front if p.objectives is not None], axis=0)
            best_point = front[0]
            best_dist = np.inf

            for point in front:
                if point.objectives is None:
                    continue
                dist = np.linalg.norm(point.objectives - ideal)
                if dist < best_dist:
                    best_dist = dist
                    best_point = point

            return best_point

    def expected_improvement(
        self,
        x: np.ndarray,
        best_y: float,
        mean_pred: float,
        std_pred: float,
    ) -> float:
        """计算期望改进 (Expected Improvement)。

        用于进度监控和停止判据。

        EI(x) = E[max(f_best - f(x), 0)]

        Args:
            x: 候选点
            best_y: 当前最优值
            mean_pred: 预测均值
            std_pred: 预测标准差

        Returns:
            期望改进值
        """
        if std_pred < self._config.epsilon:
            return 0.0

        from scipy.stats import norm
        try:
            z = (best_y - mean_pred) / std_pred
            ei = (best_y - mean_pred) * norm.cdf(z) + std_pred * norm.pdf(z)
            return float(max(0, ei))
        except (ImportError, Exception):
            # 简化 EI 估计
            z = (best_y - mean_pred) / (std_pred + self._config.epsilon)
            if z <= 0:
                return 0.0
            return float(max(0, (best_y - mean_pred) * min(1.0, z)))

    def reset(self):
        """重置优化器状态。"""
        self._generation = 0
        self._evaluations = 0
        self._restart_count = 0
        logger.info("TopologyAwareOptimizer: Reset")

    def get_progress(self) -> Dict[str, float]:
        """获取优化进度。"""
        return {
            "generation": float(self._generation),
            "evaluations": float(self._evaluations),
            "restarts": float(self._restart_count),
        }
