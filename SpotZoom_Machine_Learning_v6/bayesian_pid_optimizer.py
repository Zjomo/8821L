"""
Bayesian PID Optimizer - 贝叶斯 PID 参数优化器

Inspired by:
- BoTorch (pytorch): Bayesian optimization with PyTorch
- Optuna (optuna): Hyperparameter optimization framework
- Stable-Baselines3: Auto-tuning RL hyperparameters with Optuna

Core Innovation:
- 基于高斯过程的贝叶斯优化自动调优 PID 参数
- 多目标优化: 同时优化收敛速度和稳定性
- 在线自适应: 运行过程中持续优化 PID 参数
- 采集函数支持: EI, UCB, PI
- 纯 numpy 实现，零外部依赖
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, List, Optional, Tuple

import numpy as np


@dataclass
class BayesianPIDConfig:
    """贝叶斯 PID 优化器配置"""
    # PID 参数搜索范围
    kp_range: Tuple[float, float] = (0.1, 10.0)
    ki_range: Tuple[float, float] = (0.0, 2.0)
    kd_range: Tuple[float, float] = (0.0, 5.0)
    # 贝叶斯优化参数
    initial_samples: int = 5  # 初始随机采样数
    max_iterations: int = 30  # 最大优化迭代
    exploration_weight: float = 2.0  # UCB 探索权重
    # 在线优化参数
    online_optimization_enabled: bool = True
    online_window_size: int = 20  # 在线评估窗口
    online_improvement_threshold: float = 0.05  # 触发在线优化的改进阈值
    # 目标权重
    convergence_weight: float = 0.6  # 收敛速度权重
    stability_weight: float = 0.4  # 稳定性权重
    # 采集函数类型: 'ucb', 'ei', 'pi'
    acquisition: str = 'ucb'


@dataclass
class BayesianPIDResult:
    """贝叶斯优化结果"""
    best_kp: float
    best_ki: float
    best_kd: float
    best_score: float
    convergence_speed: float
    stability_score: float
    iteration: int
    improvement_pct: float


class BayesianPIDOptimizer:
    """贝叶斯 PID 参数优化器

    使用高斯过程贝叶斯优化自动搜索最优 PID 参数，
    支持离线优化和在线自适应调优。

    Inspired by BoTorch's Bayesian optimization and Optuna's
    hyperparameter search framework.
    """

    def __init__(self, config: Optional[BayesianPIDConfig] = None):
        self.config = config or BayesianPIDConfig()
        self._observation_history: List[Tuple[np.ndarray, float]] = []
        self._online_scores: Deque[float] = deque(maxlen=self.config.online_window_size)
        self._current_best = BayesianPIDResult(
            best_kp=1.0, best_ki=0.1, best_kd=0.5,
            best_score=-np.inf, convergence_speed=0.0,
            stability_score=0.0, iteration=0, improvement_pct=0.0,
        )
        self._rng = np.random.RandomState(42)

    def _normalize_params(self, kp: float, ki: float, kd: float) -> np.ndarray:
        """将 PID 参数归一化到 [0, 1]"""
        x0 = (kp - self.config.kp_range[0]) / (self.config.kp_range[1] - self.config.kp_range[0])
        x1 = (ki - self.config.ki_range[0]) / (self.config.ki_range[1] - self.config.ki_range[0])
        x2 = (kd - self.config.kd_range[0]) / (self.config.kd_range[1] - self.config.kd_range[0])
        return np.array([np.clip(x0, 0, 1), np.clip(x1, 0, 1), np.clip(x2, 0, 1)])

    def _denormalize_params(self, x: np.ndarray) -> Tuple[float, float, float]:
        """将归一化参数还原为 PID 参数"""
        kp = x[0] * (self.config.kp_range[1] - self.config.kp_range[0]) + self.config.kp_range[0]
        ki = x[1] * (self.config.ki_range[1] - self.config.ki_range[0]) + self.config.ki_range[0]
        kd = x[2] * (self.config.kd_range[1] - self.config.kd_range[0]) + self.config.kd_range[0]
        return float(kp), float(ki), float(kd)

    def _squared_exponential_kernel(
        self, x1: np.ndarray, x2: np.ndarray, length_scale: float = 0.3
    ) -> float:
        """平方指数核函数"""
        diff = x1 - x2
        return float(np.exp(-0.5 * np.dot(diff, diff) / (length_scale ** 2)))

    def _gaussian_process_predict(
        self, x_query: np.ndarray
    ) -> Tuple[float, float]:
        """高斯过程预测

        Args:
            x_query: 查询点

        Returns:
            (均值, 标准差)
        """
        n = len(self._observation_history)
        if n == 0:
            return 0.0, 1.0

        X = np.array([obs[0] for obs in self._observation_history])
        y = np.array([obs[1] for obs in self._observation_history])

        # 构建核矩阵
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self._squared_exponential_kernel(X[i], X[j])

        # 查询点与训练点的核
        k = np.array([self._squared_exponential_kernel(x_query, X[i]) for i in range(n)])

        # 添加噪声项
        noise = 1e-4 * np.eye(n)
        K_noise = K + noise

        try:
            L = np.linalg.cholesky(K_noise)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            v = np.linalg.solve(L, k)
            mean = float(np.dot(k, alpha))
            var = float(1.0 - np.dot(v, v))
            std = float(np.sqrt(max(0, var)))
        except np.linalg.LinAlgError:
            mean = float(np.mean(y))
            std = float(np.std(y)) + 0.1

        return mean, std

    def _acquisition_function(self, mean: float, std: float) -> float:
        """采集函数"""
        if self.config.acquisition == 'ucb':
            return mean + self.config.exploration_weight * std
        elif self.config.acquisition == 'ei':
            # Expected Improvement
            best = self._current_best.best_score
            if std < 1e-6:
                return 0.0
            z = (mean - best) / std
            ei = (mean - best) * self._norm_cdf(z) + std * self._norm_pdf(z)
            return float(ei)
        elif self.config.acquisition == 'pi':
            # Probability of Improvement
            best = self._current_best.best_score
            if std < 1e-6:
                return 0.0
            z = (mean - best) / std
            return float(self._norm_cdf(z))
        return mean

    @staticmethod
    def _norm_pdf(x: float) -> float:
        """标准正态分布 PDF"""
        return float(np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi))

    @staticmethod
    def _norm_cdf(x: float) -> float:
        """标准正态分布 CDF (近似)"""
        return float(0.5 * (1 + np.tanh(x * 0.7978845608)))  # Abramowitz & Stegun 近似

    def _evaluate_objective(
        self,
        error_history: List[float],
        overshoot_count: int = 0,
    ) -> float:
        """评估 PID 参数的目标函数

        综合考虑收敛速度和稳定性。

        Args:
            error_history: 误差历史
            overshoot_count: 过冲次数

        Returns:
            综合得分 (越高越好)
        """
        if len(error_history) < 2:
            return 0.0

        errors = np.array(error_history)

        # 收敛速度: 误差衰减率
        if errors[0] > 1e-6:
            decay_rate = -np.log(errors[-1] / errors[0] + 1e-10) / len(errors)
        else:
            decay_rate = 10.0
        convergence_score = min(1.0, decay_rate / 5.0)

        # 稳定性: 误差方差和过冲
        error_variance = np.var(errors[-min(10, len(errors)):])
        stability_score = max(0, 1.0 - error_variance / (np.mean(np.abs(errors)) + 1e-6))
        stability_score *= max(0, 1.0 - overshoot_count / 10.0)

        # 综合得分
        score = (
            self.config.convergence_weight * convergence_score +
            self.config.stability_weight * stability_score
        )
        return float(score)

    def suggest_next(self) -> Tuple[float, float, float]:
        """建议下一组 PID 参数

        Returns:
            (kp, ki, kd)
        """
        n = len(self._observation_history)

        if n < self.config.initial_samples:
            # 随机采样阶段
            x = self._rng.rand(3)
            return self._denormalize_params(x)

        # 贝叶斯优化: 在候选点中选择采集函数值最大的
        n_candidates = 100
        candidates = self._rng.rand(n_candidates, 3)

        best_acq = -np.inf
        best_x = candidates[0]

        for x in candidates:
            mean, std = self._gaussian_process_predict(x)
            acq = self._acquisition_function(mean, std)
            if acq > best_acq:
                best_acq = acq
                best_x = x

        return self._denormalize_params(best_x)

    def record_observation(
        self,
        kp: float, ki: float, kd: float,
        error_history: List[float],
        overshoot_count: int = 0,
    ) -> BayesianPIDResult:
        """记录观测结果

        Args:
            kp, ki, kd: PID 参数
            error_history: 误差历史
            overshoot_count: 过冲次数

        Returns:
            当前最优结果
        """
        x = self._normalize_params(kp, ki, kd)
        score = self._evaluate_objective(error_history, overshoot_count)

        self._observation_history.append((x, score))

        # 更新最优结果
        if score > self._current_best.best_score:
            improvement = (score - self._current_best.best_score) / (
                abs(self._current_best.best_score) + 1e-6
            )
            self._current_best = BayesianPIDResult(
                best_kp=kp, best_ki=ki, best_kd=kd,
                best_score=score,
                convergence_speed=score * self.config.convergence_weight,
                stability_score=score * self.config.stability_weight,
                iteration=len(self._observation_history),
                improvement_pct=improvement * 100,
            )

        return self._current_best

    def update_online_score(self, error: float) -> Optional[BayesianPIDResult]:
        """更新在线评估分数

        Args:
            error: 当前误差

        Returns:
            如果触发在线优化则返回建议参数，否则返回 None
        """
        self._online_scores.append(error)

        if not self.config.online_optimization_enabled:
            return None

        if len(self._online_scores) < self.config.online_window_size:
            return None

        # 检查是否需要重新优化
        recent_errors = list(self._online_scores)
        current_score = self._evaluate_objective(recent_errors)

        if current_score < self._current_best.best_score * (1 - self.config.online_improvement_threshold):
            # 触发在线优化
            kp, ki, kd = self.suggest_next()
            return BayesianPIDResult(
                best_kp=kp, best_ki=ki, best_kd=kd,
                best_score=self._current_best.best_score,
                convergence_speed=0.0,
                stability_score=0.0,
                iteration=len(self._observation_history),
                improvement_pct=0.0,
            )

        return None

    def get_best_params(self) -> Tuple[float, float, float]:
        """获取当前最优 PID 参数"""
        return (
            self._current_best.best_kp,
            self._current_best.best_ki,
            self._current_best.best_kd,
        )

    def get_diagnostics(self) -> dict:
        """获取优化器诊断信息"""
        return {
            "num_observations": len(self._observation_history),
            "best_score": self._current_best.best_score,
            "best_params": {
                "kp": self._current_best.best_kp,
                "ki": self._current_best.best_ki,
                "kd": self._current_best.best_kd,
            },
            "improvement_pct": self._current_best.improvement_pct,
            "online_window_full": len(self._online_scores) >= self.config.online_window_size,
        }
