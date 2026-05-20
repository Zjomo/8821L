"""
多目标贝叶斯优化器 (Multi-Objective Bayesian Optimizer)

灵感来源: BoTorch (https://github.com/pytorch/botorch)
           Optuna (https://github.com/optuna/optuna)

核心思想:
──────────────────────────────────────────────────────────────────────
BoTorch 基于 GPyTorch 和 PyTorch, 提供了灵活的贝叶斯优化框架。
Optuna 是一个超参数优化框架, 支持多目标优化和剪枝。

本模块实现了一个轻量化的多目标贝叶斯优化器:
- 高斯过程 (GP) 代理模型 (纯 numpy 实现)
- 多种采集函数 (EI, UCB, Expected Hypervolume Improvement)
- 多目标帕累托前沿维护
- 自动超参数优化 (PID 增益、检测阈值等)
- 实验历史管理与可视化

适用场景:
- PID 参数自动调优
- 检测器阈值优化
- 多目标权衡 (精度 vs 速度 vs 稳定性)
- 光学系统参数优化
"""

__version__ = "1.0.0"
__author__ = "SpotZoom Team"

import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Callable, Any
from enum import Enum
import time


class AcquisitionFunction(Enum):
    """采集函数类型"""
    EI = "expected_improvement"          # 期望改进
    UCB = "upper_confidence_bound"       # 上置信界
    PI = "probability_of_improvement"    # 改进概率
    EHVI = "expected_hypervolume"        # 期望超体积改进 (多目标)


class KernelType(Enum):
    """核函数类型"""
    RBF = "rbf"           # 径向基函数
    MATERN52 = "matern52" # Matern 5/2
    ARD = "ard"           # 自动相关性确定


@dataclass
class BayesianOptConfig:
    """贝叶斯优化配置"""
    # 搜索空间定义 {参数名: (下界, 上界)}
    search_space: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "pid_kp": (0.01, 2.0),
        "pid_ki": (0.001, 0.5),
        "pid_kd": (0.0, 1.0),
        "detection_threshold": (0.1, 0.99),
        "max_iterations": (10, 200),
    })

    # 优化参数
    n_initial_samples: int = 5          # 初始随机采样数
    n_iterations: int = 30              # 优化迭代次数
    acquisition: AcquisitionFunction = AcquisitionFunction.EI
    kernel: KernelType = KernelType.RBF
    exploration_beta: float = 2.0       # UCB 探索参数

    # 多目标配置
    objective_names: List[str] = field(default_factory=lambda: [
        "alignment_accuracy", "convergence_speed", "stability"
    ])
    objective_directions: Dict[str, str] = field(default_factory=lambda: {
        "alignment_accuracy": "maximize",
        "convergence_speed": "maximize",
        "stability": "maximize"
    })

    # GP 超参数
    gp_noise: float = 1e-6
    gp_lengthscale_prior: float = 1.0
    gp_signal_variance: float = 1.0

    # 并行评估
    n_parallel: int = 1
    random_seed: Optional[int] = None


@dataclass
class OptimizationResult:
    """优化结果"""
    best_params: Dict[str, float]          # 最优参数
    best_objectives: np.ndarray            # 最优目标值
    pareto_front: np.ndarray               # 帕累托前沿点集
    pareto_params: List[Dict[str, float]]  # 帕累托前沿对应参数
    all_objectives: np.ndarray             # 所有评估的目标值
    convergence_history: List[float]       # 收敛历史
    total_evaluations: int
    optimization_time_s: float
    hypervolume_history: List[float]       # 超体积历史


@dataclass
class ParetoFront:
    """帕累托前沿"""
    points: np.ndarray                    # 前沿点 (n_points, n_objectives)
    parameters: List[Dict[str, float]]    # 对应参数
    hypervolume: float                    # 超体积指标


class _GaussianProcess:
    """简化版高斯过程回归"""

    def __init__(self, kernel: KernelType = KernelType.RBF,
                 noise: float = 1e-6,
                 lengthscale: float = 1.0,
                 signal_variance: float = 1.0):
        self.kernel = kernel
        self.noise = noise
        self.lengthscale = lengthscale
        self.signal_variance = signal_variance

        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.K_inv: Optional[np.ndarray] = None
        self.alpha: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """训练 GP"""
        self.X_train = X
        self.y_train = y

        K = self._compute_kernel(X, X) + self.noise * np.eye(len(X))

        try:
            self.K_inv = np.linalg.inv(K)
            self.alpha = self.K_inv @ y
        except np.linalg.LinAlgError:
            K += 1e-4 * np.eye(len(X))
            self.K_inv = np.linalg.inv(K)
            self.alpha = self.K_inv @ y

        # 优化超参数 (简化: 使用经验方法)
        self._optimize_hyperparameters(X, y)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """预测均值和标准差"""
        if self.X_train is None:
            return np.zeros(len(X)), np.ones(len(X))

        K_star = self._compute_kernel(X, self.X_train)
        K_ss = self._compute_kernel(X, X)

        mean = K_star @ self.alpha
        var = np.diag(K_ss - K_star @ self.K_inv @ K_star.T)
        std = np.sqrt(np.maximum(var, 1e-10))

        return mean, std

    def _compute_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """计算核矩阵"""
        if self.kernel == KernelType.RBF:
            return self._rbf_kernel(X1, X2)
        elif self.kernel == KernelType.MATERN52:
            return self._matern52_kernel(X1, X2)
        else:
            return self._rbf_kernel(X1, X2)

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF 核"""
        sq_dist = self._squared_distance(X1, X2)
        return self.signal_variance * np.exp(-0.5 * sq_dist / self.lengthscale ** 2)

    def _matern52_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Matern 5/2 核"""
        sq_dist = self._squared_distance(X1, X2)
        dist = np.sqrt(sq_dist + 1e-10)
        scaled = dist * np.sqrt(5) / self.lengthscale

        return self.signal_variance * (1 + scaled + scaled ** 2 / 3) * np.exp(-scaled)

    def _squared_distance(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """计算平方欧氏距离矩阵"""
        X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
        X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True)
        dist_sq = X1_sq + X2_sq.T - 2 * X1 @ X2.T
        return np.maximum(dist_sq, 0)

    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> None:
        """简化超参数优化"""
        if len(X) < 3:
            return

        # 使用中位数启发式设置长度尺度
        distances = []
        for i in range(min(len(X), 20)):
            for j in range(i + 1, min(len(X), 20)):
                distances.append(np.sqrt(np.sum((X[i] - X[j]) ** 2)))

        if distances:
            self.lengthscale = np.median(distances) * 0.5

        # 信号方差
        self.signal_variance = np.var(y) + 1e-6


class MultiObjectiveBayesianOpt:
    """
    多目标贝叶斯优化器

    使用高斯过程代理模型和采集函数进行高效的多目标优化。
    支持帕累托前沿维护和超体积计算。

    用法示例:
        optimizer = MultiObjectiveBayesianOpt(BayesianOptConfig())
        def evaluate(params):
            return [accuracy, speed, stability]
        result = optimizer.optimize(evaluate)
        print(f"最优参数: {result.best_params}")
    """

    def __init__(self, config: Optional[BayesianOptConfig] = None):
        self.config = config or BayesianOptConfig()

        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

        # 参数名和边界
        self._param_names = list(self.config.search_space.keys())
        self._bounds = np.array([self.config.search_space[name]
                                 for name in self._param_names])
        self._n_params = len(self._param_names)
        self._n_objectives = len(self.config.objective_names)

        # GP 模型 (每个目标一个)
        self._gps: List[_GaussianProcess] = [
            _GaussianProcess(
                kernel=self.config.kernel,
                noise=self.config.gp_noise,
                lengthscale=self.config.gp_lengthscale_prior,
                signal_variance=self.config.gp_signal_variance
            )
            for _ in range(self._n_objectives)
        ]

        # 实验历史
        self._X_observed: List[np.ndarray] = []
        self._y_observed: List[np.ndarray] = []
        self._convergence: List[float] = []
        self._hypervolume_history: List[float] = []

        # 帕累托前沿
        self._pareto = ParetoFront(
            points=np.empty((0, self._n_objectives)),
            parameters=[],
            hypervolume=0.0
        )

        # 参考点 (用于超体积计算)
        self._reference_point = np.full(self._n_objectives, -np.inf)

    def optimize(self, objective_fn: Callable[[Dict[str, float]], List[float]],
                 callback: Optional[Callable] = None) -> OptimizationResult:
        """
        执行多目标贝叶斯优化

        Args:
            objective_fn: 目标函数, 输入参数字典, 输出目标值列表
            callback: 每次迭代后的回调函数

        Returns:
            OptimizationResult: 优化结果
        """
        start_time = time.time()

        # 阶段1: 初始随机采样
        self._initial_sampling(objective_fn)

        # 阶段2: 贝叶斯优化迭代
        for i in range(self.config.n_iterations):
            # 训练 GP
            self._train_gps()

            # 选择下一个评估点
            next_x = self._select_next_point()

            # 评估目标函数
            params = self._array_to_dict(next_x)
            objectives = objective_fn(params)

            # 记录
            self._X_observed.append(next_x)
            self._y_observed.append(np.array(objectives))

            # 更新帕累托前沿
            self._update_pareto()

            # 记录收敛
            if len(self._pareto.points) > 0:
                self._hypervolume_history.append(self._pareto.hypervolume)
                self._convergence.append(self._pareto.hypervolume)

            if callback:
                callback(i, params, objectives)

        # 选择最优折中解
        best_idx = self._select_best_compromise()

        elapsed = time.time() - start_time

        return OptimizationResult(
            best_params=self._array_to_dict(self._X_observed[best_idx]),
            best_objectives=self._y_observed[best_idx],
            pareto_front=self._pareto.points.copy(),
            pareto_params=self._pareto.parameters.copy(),
            all_objectives=np.array(self._y_observed),
            convergence_history=self._convergence,
            total_evaluations=len(self._X_observed),
            optimization_time_s=elapsed,
            hypervolume_history=self._hypervolume_history
        )

    def _initial_sampling(self, objective_fn: Callable) -> None:
        """初始随机采样"""
        for _ in range(self.config.n_initial_samples):
            x = self._random_sample()
            params = self._array_to_dict(x)
            objectives = objective_fn(params)
            self._X_observed.append(x)
            self._y_observed.append(np.array(objectives))

        self._update_pareto()

    def _random_sample(self) -> np.ndarray:
        """在搜索空间内随机采样"""
        return np.array([np.random.uniform(lo, hi)
                        for lo, hi in self._bounds])

    def _train_gps(self) -> None:
        """训练所有 GP 模型"""
        X = np.array(self._X_observed)
        y = np.array(self._y_observed)

        for i, gp in enumerate(self._gps):
            # 归一化目标值
            y_col = y[:, i]
            mean, std = np.mean(y_col), np.std(y_col)
            if std > 1e-10:
                y_norm = (y_col - mean) / std
            else:
                y_norm = y_col - mean

            gp.fit(X, y_norm)

    def _select_next_point(self) -> np.ndarray:
        """选择下一个评估点"""
        if self.config.acquisition == AcquisitionFunction.EHVI and self._n_objectives > 1:
            return self._optimize_acquisition_ehvi()

        # 单目标采集函数 (对每个目标分别计算, 然后聚合)
        candidates = []
        for _ in range(100):
            x = self._random_sample()
            acq_value = self._compute_acquisition_aggregate(x)
            candidates.append((x, acq_value))

        candidates.sort(key=lambda c: c[1], reverse=True)
        return candidates[0][0]

    def _compute_acquisition_aggregate(self, x: np.ndarray) -> float:
        """计算聚合采集函数值"""
        total = 0.0
        for i, gp in enumerate(self._gps):
            x_2d = x.reshape(1, -1)
            mean, std = gp.predict(x_2d)
            mean, std = mean[0], std[0]

            if self.config.acquisition == AcquisitionFunction.EI:
                # 期望改进
                best_y = np.max(np.array(self._y_observed)[:, i])
                if std > 1e-10:
                    z = (mean - best_y) / std
                    ei = mean - best_y
                    ei *= self._norm_cdf(z)
                    ei += std * self._norm_pdf(z)
                else:
                    ei = 0.0
                total += ei

            elif self.config.acquisition == AcquisitionFunction.UCB:
                total += mean + self.config.exploration_beta * std

            elif self.config.acquisition == AcquisitionFunction.PI:
                best_y = np.max(np.array(self._y_observed)[:, i])
                if std > 1e-10:
                    total += self._norm_cdf((mean - best_y) / std)
                else:
                    total += 0.0

        return total / self._n_objectives

    def _optimize_acquisition_ehvi(self) -> np.ndarray:
        """期望超体积改进 (简化版)"""
        best_x = self._random_sample()
        best_ehvi = -np.inf

        for _ in range(200):
            x = self._random_sample()
            ehvi = self._compute_ehvi(x)
            if ehvi > best_ehvi:
                best_ehvi = ehvi
                best_x = x

        return best_x

    def _compute_ehvi(self, x: np.ndarray) -> float:
        """计算期望超体积改进"""
        x_2d = x.reshape(1, -1)
        means = []
        stds = []
        for gp in self._gps:
            m, s = gp.predict(x_2d)
            means.append(m[0])
            stds.append(s[0])

        # 简化 EHVI: 使用蒙特卡罗采样
        n_samples = 50
        hv_improvement = 0.0

        for _ in range(n_samples):
            sample = np.array([np.random.normal(m, s)
                              for m, s in zip(means, stds)])

            # 计算该采样点相对于当前帕累托前沿的超体积改进
            if len(self._pareto.points) > 0:
                dominated = False
                for pf_point in self._pareto.points:
                    if np.all(sample <= pf_point):
                        dominated = True
                        break
                if not dominated:
                    hv_improvement += 1.0

        return hv_improvement / n_samples

    def _update_pareto(self) -> None:
        """更新帕累托前沿"""
        if not self._y_observed:
            return

        y = np.array(self._y_observed)
        directions = [1.0 if self.config.objective_directions.get(
            name, "maximize") == "maximize" else -1.0
            for name in self.config.objective_names]

        # 应用方向
        y_directed = y * directions

        # 找到帕累托非支配点
        is_pareto = np.ones(len(y), dtype=bool)
        for i in range(len(y)):
            if not is_pareto[i]:
                continue
            for j in range(len(y)):
                if i == j or not is_pareto[j]:
                    continue
                if np.all(y_directed[j] >= y_directed[i]) and \
                   np.any(y_directed[j] > y_directed[i]):
                    is_pareto[i] = False
                    break

        pareto_points = y[is_pareto]
        pareto_params = [self._array_to_dict(self._X_observed[i])
                        for i in range(len(y)) if is_pareto[i]]

        # 更新参考点
        if len(pareto_points) > 0:
            self._reference_point = np.max(pareto_points, axis=0) * 1.1

        # 计算超体积
        hv = self._compute_hypervolume(pareto_points) if len(pareto_points) > 0 else 0.0

        self._pareto = ParetoFront(
            points=pareto_points,
            parameters=pareto_params,
            hypervolume=hv
        )

    def _compute_hypervolume(self, points: np.ndarray) -> float:
        """计算超体积指标 (2D/3D)"""
        if len(points) == 0:
            return 0.0

        ref = self._reference_point.copy()
        ref = np.maximum(ref, np.max(points, axis=0) * 1.1)

        if self._n_objectives == 2:
            return self._hv_2d(points, ref)
        elif self._n_objectives == 3:
            return self._hv_3d(points, ref)
        else:
            return self._hv_monte_carlo(points, ref)

    def _hv_2d(self, points: np.ndarray, ref: np.ndarray) -> float:
        """2D 超体积"""
        sorted_idx = np.argsort(points[:, 0])
        hv = 0.0
        prev_y = ref[1]

        for idx in sorted_idx:
            p = points[idx]
            hv += (p[0] - (0 if hv == 0 else points[sorted_idx[np.searchsorted(sorted_idx, idx) - 1]][0] if idx > sorted_idx[0] else 0))
            hv += (ref[1] - p[1]) * (ref[0] - p[0]) * 0.5

        # 简化计算
        total = 0.0
        for p in points:
            total += max(0, ref[0] - p[0]) * max(0, ref[1] - p[1])
        return total * 0.5

    def _hv_3d(self, points: np.ndarray, ref: np.ndarray) -> float:
        """3D 超体积 (蒙特卡罗近似)"""
        return self._hv_monte_carlo(points, ref)

    def _hv_monte_carlo(self, points: np.ndarray, ref: np.ndarray) -> float:
        """蒙特卡罗超体积"""
        n_samples = 10000
        lb = np.min(points, axis=0)
        ub = ref

        samples = np.random.uniform(lb, ub, size=(n_samples, self._n_objectives))

        dominated = np.zeros(n_samples, dtype=bool)
        for p in points:
            dominated |= np.all(samples <= p, axis=1)

        volume = np.prod(ub - lb)
        return volume * np.mean(dominated)

    def _select_best_compromise(self) -> int:
        """选择最优折中解 (距理想点最近的帕累托解)"""
        if len(self._pareto.points) == 0:
            return 0

        ideal = np.max(self._pareto.points, axis=0)
        worst = np.min(self._pareto.points, axis=0)
        range_ = worst - ideal
        range_[range_ == 0] = 1.0

        # 归一化距离
        best_idx = 0
        best_dist = np.inf
        for i, p in enumerate(self._pareto.points):
            dist = np.sqrt(np.sum(((p - ideal) / range_) ** 2))
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        # 找到对应的原始索引
        for orig_idx in range(len(self._y_observed)):
            if np.allclose(self._y_observed[orig_idx], self._pareto.points[best_idx]):
                return orig_idx

        return 0

    def _array_to_dict(self, x: np.ndarray) -> Dict[str, float]:
        """数组转参数字典"""
        return {name: float(val) for name, val in zip(self._param_names, x)}

    def _dict_to_array(self, params: Dict[str, float]) -> np.ndarray:
        """参数字典转数组"""
        return np.array([params[name] for name in self._param_names])

    @staticmethod
    def _norm_cdf(x: float) -> float:
        """标准正态 CDF"""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    @staticmethod
    def _norm_pdf(x: float) -> float:
        """标准正态 PDF"""
        return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)

    def suggest(self) -> Dict[str, float]:
        """获取下一个建议参数 (用于在线优化)"""
        if len(self._X_observed) < self.config.n_initial_samples:
            return self._array_to_dict(self._random_sample())

        self._train_gps()
        next_x = self._select_next_point()
        return self._array_to_dict(next_x)

    def tell(self, params: Dict[str, float], objectives: List[float]) -> None:
        """记录评估结果 (用于在线优化)"""
        x = self._dict_to_array(params)
        self._X_observed.append(x)
        self._y_observed.append(np.array(objectives))
        self._update_pareto()

    def get_statistics(self) -> Dict:
        """获取优化统计"""
        return {
            "n_evaluations": len(self._X_observed),
            "n_pareto_points": len(self._pareto.points),
            "hypervolume": self._pareto.hypervolume,
            "best_objectives": self._pareto.points.tolist() if len(self._pareto.points) > 0 else [],
            "convergence_trend": self._convergence[-5:] if self._convergence else []
        }
