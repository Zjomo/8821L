"""
智能实验调度器 (IntelligentExperimentScheduler)

灵感来源:
- DeepTrack2 (https://github.com/DeepTrackAI/DeepTrack2) — 序列化实验管道
- Bayesian Optimization — 贝叶斯优化用于超参数搜索
- Spearmint (https://github.com/HIPS/Spearmint) — 高斯过程贝叶斯优化

算法原理:
- Bayesian Optimization — 基于高斯代理模型的黑箱优化
- Gaussian Process — 高斯过程回归用于建模目标函数
- Acquisition Function — 采集函数 (EI, UCB) 平衡探索与利用
- Pareto Front — 多目标优化的帕累托前沿

功能:
- 多目标贝叶斯优化用于对准参数搜索
- 自适应采样策略 (探索 vs 利用平衡)
- 实验历史管理与帕累托前沿追踪
- 支持自定义目标函数和参数空间
- 实验结果可视化数据输出

依赖: numpy, scipy (无外部深度学习框架依赖)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm

LOGGER = logging.getLogger("SpotZoom.IntelligentExperimentScheduler")


@dataclass
class SchedulerConfig:
    """调度器配置参数。"""
    # --- 参数空间 ---
    param_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    # 例如: {"x_offset": (-10, 10), "y_offset": (-10, 10), "focus": (-5, 5)}

    # --- 贝叶斯优化参数 ---
    n_initial_samples: int = 5  # 初始随机采样数
    n_iterations: int = 50  # 优化迭代次数
    acquisition_function: str = "ei"  # "ei" (期望改进), "ucb" (上置信界)
    ucb_beta: float = 2.0  # UCB 探索参数
    gp_noise: float = 1e-6  # 高斯过程噪声方差
    gp_length_scale: float = 1.0  # 高斯过程长度尺度

    # --- 多目标参数 ---
    n_objectives: int = 1  # 目标函数数量
    objective_weights: Optional[List[float]] = None  # 目标权重

    # --- 自适应采样 ---
    exploration_fraction: float = 0.3  # 探索比例 (0~1)
    min_exploration_samples: int = 3  # 最小探索样本数

    # --- 实验管理 ---
    max_history_size: int = 1000  # 最大历史记录数
    random_seed: Optional[int] = None  # 随机种子


@dataclass
class ExperimentRecord:
    """单次实验记录。"""
    parameters: Dict[str, float]  # 参数值
    objectives: List[float]  # 目标函数值
    iteration: int  # 迭代次数
    timestamp: float = 0.0  # 时间戳
    is_pareto: bool = False  # 是否在帕累托前沿上


@dataclass
class ScheduleResult:
    """调度结果。"""
    best_parameters: Dict[str, float] = field(default_factory=dict)
    best_objectives: List[float] = field(default_factory=list)
    pareto_front: List[ExperimentRecord] = field(default_factory=list)
    all_records: List[ExperimentRecord] = field(default_factory=list)
    convergence_history: List[float] = field(default_factory=list)
    total_iterations: int = 0
    improvement_ratio: float = 0.0


class _GaussianProcess:
    """简化高斯过程回归模型。"""

    def __init__(self, noise: float = 1e-6, length_scale: float = 1.0):
        self._noise = noise
        self._length_scale = length_scale
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._K_inv: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None

    def _rbf_kernel(
        self, X1: np.ndarray, X2: np.ndarray
    ) -> np.ndarray:
        """RBF (径向基函数) 核。"""
        sq_dist = np.sum(X1 ** 2, axis=1, keepdims=True) + \
                  np.sum(X2 ** 2, axis=1) - 2 * X1 @ X2.T
        return np.exp(-0.5 * sq_dist / (self._length_scale ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray):
        """拟合高斯过程。"""
        self._X_train = X.copy()
        self._y_train = y.copy()

        K = self._rbf_kernel(X, X) + self._noise * np.eye(len(X))
        self._K_inv = np.linalg.inv(K)
        self._alpha = self._K_inv @ y

    def predict(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """预测均值和标准差。

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (均值, 标准差)
        """
        if self._X_train is None:
            raise RuntimeError("模型未拟合，请先调用 fit()")

        K_s = self._rbf_kernel(X, self._X_train)
        K_ss = self._rbf_kernel(X, X)

        mu = K_s @ self._alpha
        cov = K_ss - K_s @ self._K_inv @ K_s.T

        std = np.sqrt(np.diag(cov))
        std = np.maximum(std, 1e-12)

        return mu, std

    def is_fitted(self) -> bool:
        """是否已拟合。"""
        return self._X_train is not None


class IntelligentExperimentScheduler:
    """智能实验调度器。

    使用贝叶斯优化自动搜索最优对准参数，
    支持多目标优化和帕累托前沿追踪。

    Parameters
    ----------
    config : SchedulerConfig, optional
        调度器配置参数。为 None 时使用默认配置。
    """

    def __init__(self, config: Optional[SchedulerConfig] = None):
        self._cfg = config if config is not None else SchedulerConfig()
        self._rng = np.random.RandomState(self._cfg.random_seed)
        self._history: List[ExperimentRecord] = []
        self._gp: Optional[_GaussianProcess] = None
        self._param_names: List[str] = []
        self._iteration: int = 0

    def _parameters_to_vector(
        self, params: Dict[str, float]
    ) -> np.ndarray:
        """将参数字典转换为归一化向量。"""
        vec = np.zeros(len(self._param_names), dtype=np.float64)
        for i, name in enumerate(self._param_names):
            lo, hi = self._cfg.param_ranges[name]
            vec[i] = (params[name] - lo) / (hi - lo + 1e-12)
        return vec

    def _vector_to_parameters(
        self, vec: np.ndarray
    ) -> Dict[str, float]:
        """将归一化向量转换为参数字典。"""
        params = {}
        for i, name in enumerate(self._param_names):
            lo, hi = self._cfg.param_ranges[name]
            params[name] = float(lo + vec[i] * (hi - lo))
        return params

    def _random_sample(self) -> Dict[str, float]:
        """在参数空间中随机采样。"""
        params = {}
        for name, (lo, hi) in self._cfg.param_ranges.items():
            params[name] = float(self._rng.uniform(lo, hi))
        return params

    def _compute_acquisition(
        self,
        X: np.ndarray,
        gp_mean: np.ndarray,
        gp_std: np.ndarray,
        best_y: float,
    ) -> np.ndarray:
        """计算采集函数值。

        Parameters
        ----------
        X : np.ndarray
            候选点。
        gp_mean : np.ndarray
            GP 预测均值。
        gp_std : np.ndarray
            GP 预测标准差。
        best_y : float
            当前最优目标值。

        Returns
        -------
        np.ndarray
            采集函数值。
        """
        if self._cfg.acquisition_function == "ei":
            # 期望改进 (Expected Improvement)
            improvement = gp_mean - best_y
            Z = improvement / (gp_std + 1e-12)
            ei = improvement * norm.cdf(Z) + gp_std * norm.pdf(Z)
            return ei

        elif self._cfg.acquisition_function == "ucb":
            # 上置信界 (Upper Confidence Bound)
            return gp_mean + self._cfg.ucb_beta * gp_std

        else:
            raise ValueError(f"未知采集函数: {self._cfg.acquisition_function}")

    def _update_pareto_front(self):
        """更新帕累托前沿。"""
        if self._cfg.n_objectives < 2:
            # 单目标: 最优记录即为帕累托
            if self._history:
                best = min(self._history, key=lambda r: r.objectives[0])
                for r in self._history:
                    r.is_pareto = (r is best)
            return

        # 多目标帕累托前沿
        for r in self._history:
            r.is_pareto = True

        for i, ri in enumerate(self._history):
            for j, rj in enumerate(self._history):
                if i == j:
                    continue
                # 检查 rj 是否支配 ri
                dominated = True
                for k in range(self._cfg.n_objectives):
                    if ri.objectives[k] < rj.objectives[k]:
                        dominated = False
                        break
                if dominated:
                    ri.is_pareto = False
                    break

    def _weighted_objective(self, objectives: List[float]) -> float:
        """计算加权目标值。"""
        if self._cfg.objective_weights is None:
            return sum(objectives)
        weights = self._cfg.objective_weights
        return sum(w * o for w, o in zip(weights, objectives))

    def set_objective(
        self,
        objective_func: Callable[[Dict[str, float]], List[float]],
    ):
        """设置目标函数。

        Parameters
        ----------
        objective_func : callable
            接受参数字典，返回目标值列表的函数。
            对于最小化问题，值越小越好。
        """
        self._objective_func = objective_func
        LOGGER.info("目标函数已设置")

    def optimize(
        self,
        objective_func: Optional[Callable[[Dict[str, float]], List[float]]] = None,
    ) -> ScheduleResult:
        """执行贝叶斯优化搜索最优参数。

        Parameters
        ----------
        objective_func : callable, optional
            目标函数。如果之前已设置可省略。

        Returns
        -------
        ScheduleResult
            优化结果。
        """
        if objective_func is not None:
            self.set_objective(objective_func)

        if not hasattr(self, '_objective_func'):
            raise RuntimeError("请先设置目标函数 (set_objective 或传入 objective_func)")

        if not self._cfg.param_ranges:
            raise ValueError("请设置参数范围 (param_ranges)")

        self._param_names = list(self._cfg.param_ranges.keys())
        self._iteration = 0

        LOGGER.info(
            "开始贝叶斯优化: %d 参数, %d 初始样本, %d 迭代",
            len(self._param_names),
            self._cfg.n_initial_samples,
            self._cfg.n_iterations,
        )

        convergence: List[float] = []

        # --- 阶段 1: 初始随机采样 ---
        for i in range(self._cfg.n_initial_samples):
            params = self._random_sample()
            objectives = self._objective_func(params)

            record = ExperimentRecord(
                parameters=params,
                objectives=objectives,
                iteration=self._iteration,
            )
            self._history.append(record)
            convergence.append(self._weighted_objective(objectives))
            self._iteration += 1

            LOGGER.debug(
                "初始采样 %d: 参数=%s, 目标=%s",
                i + 1, params, objectives,
            )

        # --- 阶段 2: 贝叶斯优化 ---
        for i in range(self._cfg.n_iterations):
            # 构建训练数据
            X_train = np.array([
                self._parameters_to_vector(r.parameters) for r in self._history
            ])
            y_train = np.array([
                self._weighted_objective(r.objectives) for r in self._history
            ])

            # 拟合高斯过程
            self._gp = _GaussianProcess(
                noise=self._cfg.gp_noise,
                length_scale=self._cfg.gp_length_scale,
            )
            self._gp.fit(X_train, y_train)

            # 生成候选点
            n_candidates = 1000
            X_candidates = self._rng.uniform(
                0, 1, size=(n_candidates, len(self._param_names))
            )

            # GP 预测
            gp_mean, gp_std = self._gp.predict(X_candidates)

            # 当前最优值
            best_y = np.min(y_train)

            # 计算采集函数
            acq_values = self._compute_acquisition(
                X_candidates, gp_mean, gp_std, best_y
            )

            # 自适应采样: 以一定概率随机探索
            if (self._rng.random() < self._cfg.exploration_fraction
                    and i >= self._cfg.min_exploration_samples):
                # 探索: 随机采样
                best_idx = self._rng.randint(n_candidates)
            else:
                # 利用: 选择采集函数最大的候选点
                best_idx = int(np.argmax(acq_values))

            # 转换为参数
            params = self._vector_to_parameters(X_candidates[best_idx])
            objectives = self._objective_func(params)

            record = ExperimentRecord(
                parameters=params,
                objectives=objectives,
                iteration=self._iteration,
            )
            self._history.append(record)
            convergence.append(self._weighted_objective(objectives))
            self._iteration += 1

            # 限制历史大小
            if len(self._history) > self._cfg.max_history_size:
                self._history = self._history[-self._cfg.max_history_size:]

            if (i + 1) % 10 == 0:
                LOGGER.info(
                    "迭代 %d/%d: 当前最优=%.6f",
                    i + 1, self._cfg.n_iterations, np.min(convergence),
                )

        # --- 更新帕累托前沿 ---
        self._update_pareto_front()

        # --- 构建结果 ---
        best_record = min(self._history, key=lambda r: self._weighted_objective(r.objectives))
        pareto_records = [r for r in self._history if r.is_pareto]

        initial_best = convergence[0] if convergence else 0
        final_best = min(convergence) if convergence else 0
        improvement = (
            (initial_best - final_best) / (abs(initial_best) + 1e-12)
        )

        result = ScheduleResult(
            best_parameters=best_record.parameters,
            best_objectives=best_record.objectives,
            pareto_front=pareto_records,
            all_records=list(self._history),
            convergence_history=convergence,
            total_iterations=self._iteration,
            improvement_ratio=improvement,
        )

        LOGGER.info(
            "优化完成: 最优参数=%s, 最优目标=%s, 改善=%.2f%%",
            best_record.parameters,
            best_record.objectives,
            improvement * 100,
        )

        return result

    def suggest_next(self) -> Dict[str, float]:
        """建议下一个实验参数。

        Returns
        -------
        Dict[str, float]
            建议的参数值。
        """
        if not self._history:
            return self._random_sample()

        if len(self._history) < self._cfg.n_initial_samples:
            return self._random_sample()

        # 使用当前历史构建 GP 并建议
        X_train = np.array([
            self._parameters_to_vector(r.parameters) for r in self._history
        ])
        y_train = np.array([
            self._weighted_objective(r.objectives) for r in self._history
        ])

        self._gp = _GaussianProcess(
            noise=self._cfg.gp_noise,
            length_scale=self._cfg.gp_length_scale,
        )
        self._gp.fit(X_train, y_train)

        n_candidates = 500
        X_candidates = self._rng.uniform(
            0, 1, size=(n_candidates, len(self._param_names))
        )
        gp_mean, gp_std = self._gp.predict(X_candidates)
        best_y = np.min(y_train)
        acq_values = self._compute_acquisition(
            X_candidates, gp_mean, gp_std, best_y
        )
        best_idx = int(np.argmax(acq_values))

        return self._vector_to_parameters(X_candidates[best_idx])

    def get_pareto_front(self) -> List[ExperimentRecord]:
        """获取帕累托前沿。"""
        self._update_pareto_front()
        return [r for r in self._history if r.is_pareto]

    def get_history(self) -> List[ExperimentRecord]:
        """获取实验历史。"""
        return list(self._history)

    def reset(self):
        """重置调度器状态。"""
        self._history.clear()
        self._gp = None
        self._iteration = 0
        LOGGER.info("智能实验调度器已重置")
