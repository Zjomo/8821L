"""
收敛预测器 (v7.0)

基于指数衰减模型的对准过程收敛预测模块。
实时预测剩余迭代次数和最终对准精度，支持早期终止判断。

灵感来源:
- python-control (https://github.com/python-control/python-control) — 控制系统分析
- Bluesky (https://github.com/bluesky/bluesky) — 实验计划终止条件
- SCIKIT-LEARN 曲线拟合 — 指数衰减拟合

算法原理:
  对准误差通常遵循指数衰减模型:
    e(t) = e_0 * exp(-λt) + e_∞

  其中:
    e_0: 初始误差
    λ: 衰减速率
    e_∞: 稳态误差 (非零表示系统存在偏差)

  通过在线拟合此模型，可以:
  1. 预测达到目标精度所需的剩余迭代次数
  2. 检测收敛停滞 (λ → 0)
  3. 估计最终可达精度 (e_∞)
  4. 判断是否需要干预 (如重新对焦)

外部依赖: numpy
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class ConvergenceStatus(Enum):
    """收敛状态枚举。"""
    CONVERGED = "converged"           # 已收敛
    CONVERGING = "converging"         # 正在收敛
    STALLED = "stalled"               # 停滞 (衰减速率接近 0)
    DIVERGING = "diverging"           # 发散
    INSUFFICIENT_DATA = "insufficient"  # 数据不足


@dataclass
class ConvergencePrediction:
    """收敛预测结果。"""
    status: ConvergenceStatus = ConvergenceStatus.INSUFFICIENT_DATA
    current_error: float = 0.0         # 当前误差 (像素)
    predicted_final_error: float = 0.0 # 预测最终误差
    decay_rate: float = 0.0            # 衰减速率 λ
    initial_error: float = 0.0         # 初始误差 e_0
    steady_state_error: float = 0.0    # 稳态误差 e_∞
    remaining_iterations: int = -1     # 预测剩余迭代次数 (-1=无法预测)
    confidence: float = 0.0            # 预测置信度 (0-1)
    fit_r_squared: float = 0.0         # 拟合 R²
    estimated_time_to_converge: float = -1.0  # 预计收敛时间 (秒, -1=无法预测)
    should_terminate: bool = False     # 是否建议提前终止
    should_intervene: bool = False     # 是否建议人工干预


@dataclass
class ConvergenceConfig:
    """收敛预测器配置。"""
    # 拟合参数
    min_samples_for_fit: int = 8       # 最少样本数
    max_samples: int = 200             # 最大历史样本数

    # 收敛判定
    convergence_threshold: float = 2.0 # 收敛阈值 (像素)
    stall_threshold: float = 0.01      # 停滞阈值 (λ < 此值判定为停滞)
    divergence_threshold: float = -0.005  # 发散阈值 (λ < 此值判定为发散)

    # 提前终止
    early_terminate_confidence: float = 0.8  # 提前终止所需置信度
    early_terminate_iterations: int = 50     # 最少迭代次数后才允许提前终止

    # 干预建议
    intervention_stall_iterations: int = 30  # 停滞多少次后建议干预
    intervention_divergence_count: int = 3   # 连续发散多少次后建议干预


class ConvergencePredictor:
    """对准过程收敛预测器。

    基于指数衰减模型实时预测对准过程的收敛行为。

    使用示例:
        predictor = ConvergencePredictor()
        for error in error_sequence:
            prediction = predictor.update(error)
            if prediction.should_intervene:
                print("Warning: convergence stalled!")
            if prediction.should_terminate:
                print("Converged! Terminating early.")
                break
    """

    def __init__(self, config: Optional[ConvergenceConfig] = None):
        self._config = config or ConvergenceConfig()
        self._errors: list = []
        self._timestamps: list = []
        self._iteration = 0
        self._stall_count = 0
        self._divergence_count = 0
        self._start_time: Optional[float] = None

    @property
    def config(self) -> ConvergenceConfig:
        return self._config

    def update(self, error: float, timestamp: Optional[float] = None) -> ConvergencePrediction:
        """更新误差记录并生成收敛预测。

        Args:
            error: 当前对准误差 (像素)
            timestamp: 时间戳 (秒，可选)

        Returns:
            ConvergencePrediction: 收敛预测结果
        """
        import time
        if timestamp is None:
            timestamp = time.perf_counter()
        if self._start_time is None:
            self._start_time = timestamp

        self._iteration += 1
        self._errors.append(error)
        self._timestamps.append(timestamp)

        # 截断历史
        if len(self._errors) > self._config.max_samples:
            self._errors.pop(0)
            self._timestamps.pop(0)

        # 数据不足
        if len(self._errors) < self._config.min_samples_for_fit:
            return ConvergencePrediction(
                status=ConvergenceStatus.INSUFFICIENT_DATA,
                current_error=error,
            )

        # 拟合指数衰减模型
        params, r_squared = self._fit_exponential_decay()

        if params is None:
            return ConvergencePrediction(
                status=ConvergenceStatus.INSUFFICIENT_DATA,
                current_error=error,
            )

        e0, lam, e_inf = params

        # 判断收敛状态
        status = self._classify_status(lam, error, e_inf)

        # 计算预测
        remaining = self._predict_remaining_iterations(e0, lam, e_inf)
        confidence = self._compute_confidence(r_squared, len(self._errors))
        time_to_converge = self._estimate_time_to_converge(remaining)

        # 判断是否应终止
        should_terminate = self._check_early_terminate(
            error, status, confidence, remaining
        )

        # 判断是否应干预
        should_intervene = self._check_intervention(status)

        prediction = ConvergencePrediction(
            status=status,
            current_error=error,
            predicted_final_error=e_inf,
            decay_rate=lam,
            initial_error=e0,
            steady_state_error=e_inf,
            remaining_iterations=remaining,
            confidence=confidence,
            fit_r_squared=r_squared,
            estimated_time_to_converge=time_to_converge,
            should_terminate=should_terminate,
            should_intervene=should_intervene,
        )

        logger.debug(
            f"ConvergencePredictor: iter={self._iteration}, error={error:.2f}, "
            f"status={status.value}, λ={lam:.4f}, e∞={e_inf:.2f}, "
            f"remaining={remaining}, R²={r_squared:.3f}"
        )

        return prediction

    def _fit_exponential_decay(self) -> Tuple[Optional[Tuple[float, float, float]], float]:
        """拟合指数衰减模型 e(t) = e0 * exp(-λt) + e_inf。

        使用线性化方法: ln(e - e_inf) = ln(e0) - λt
        通过网格搜索 e_inf 找到最佳拟合。

        Returns:
            (params, r_squared): ((e0, λ, e_inf), R²) 或 (None, 0)
        """
        n = len(self._errors)
        t = np.arange(n, dtype=np.float64)
        e = np.array(self._errors, dtype=np.float64)

        # 网格搜索 e_inf
        best_r2 = -np.inf
        best_params = None

        e_min = np.min(e)
        e_max = np.max(e)

        # e_inf 搜索范围: [0, e_min * 0.9]
        if e_min < 0:
            e_inf_candidates = np.linspace(0, 0, 5)
        else:
            e_inf_candidates = np.linspace(0, max(e_min * 0.9, 0.1), 20)

        for e_inf in e_inf_candidates:
            shifted = e - e_inf
            valid = shifted > 1e-10

            if np.sum(valid) < self._config.min_samples_for_fit // 2:
                continue

            log_shifted = np.log(shifted[valid])
            t_valid = t[valid]

            # 线性拟合: log(e - e_inf) = log(e0) - λt
            try:
                A = np.vstack([np.ones(len(t_valid)), -t_valid]).T
                result = np.linalg.lstsq(A, log_shifted, rcond=None)
                coeffs = result[0]
                log_e0 = coeffs[0]
                lam = coeffs[1]

                if lam <= 0:
                    continue

                e0 = np.exp(log_e0)

                # 计算 R²
                predicted = e0 * np.exp(-lam * t) + e_inf
                ss_res = np.sum((e - predicted) ** 2)
                ss_tot = np.sum((e - np.mean(e)) ** 2)
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0

                if r2 > best_r2:
                    best_r2 = r2
                    best_params = (e0, lam, e_inf)
            except (np.linalg.LinAlgError, ValueError):
                continue

        return best_params, max(best_r2, 0.0)

    def _classify_status(
        self, lam: float, current_error: float, e_inf: float
    ) -> ConvergenceStatus:
        """分类收敛状态。"""
        cfg = self._config

        if current_error <= cfg.convergence_threshold:
            return ConvergenceStatus.CONVERGED
        elif lam < cfg.divergence_threshold:
            return ConvergenceStatus.DIVERGING
        elif lam < cfg.stall_threshold:
            return ConvergenceStatus.STALLED
        else:
            return ConvergenceStatus.CONVERGING

    def _predict_remaining_iterations(
        self, e0: float, lam: float, e_inf: float
    ) -> int:
        """预测达到收敛阈值所需的剩余迭代次数。

        求解: e0 * exp(-λ * n) + e_inf = threshold
        => n = -ln((threshold - e_inf) / e0) / λ
        """
        threshold = self._config.convergence_threshold

        if e_inf >= threshold:
            return -1  # 无法达到阈值

        ratio = (threshold - e_inf) / e0
        if ratio <= 0 or ratio >= 1:
            return -1

        try:
            n = -np.log(ratio) / lam
            return max(0, int(np.ceil(n)) - len(self._errors))
        except (ValueError, ZeroDivisionError):
            return -1

    def _compute_confidence(self, r_squared: float, n_samples: int) -> float:
        """计算预测置信度。"""
        # R² 贡献
        r2_conf = min(r_squared, 1.0)

        # 样本数贡献
        min_n = self._config.min_samples_for_fit
        sample_conf = min((n_samples - min_n) / (min_n * 2), 1.0)

        return float(0.6 * r2_conf + 0.4 * sample_conf)

    def _estimate_time_to_converge(self, remaining: int) -> float:
        """估计收敛所需时间。"""
        if remaining < 0:
            return -1.0

        if len(self._timestamps) < 2:
            return -1.0

        avg_interval = (
            self._timestamps[-1] - self._timestamps[0]
        ) / (len(self._timestamps) - 1)

        return remaining * avg_interval

    def _check_early_terminate(
        self,
        error: float,
        status: ConvergenceStatus,
        confidence: float,
        remaining: int
    ) -> bool:
        """检查是否应提前终止。"""
        cfg = self._config

        if self._iteration < cfg.early_terminate_iterations:
            return False

        if status == ConvergenceStatus.CONVERGED:
            return True

        if (status == ConvergenceStatus.CONVERGING and
                confidence > cfg.early_terminate_confidence and
                0 <= remaining <= 3):
            return True

        return False

    def _check_intervention(self, status: ConvergenceStatus) -> bool:
        """检查是否需要人工干预。"""
        cfg = self._config

        if status == ConvergenceStatus.STALLED:
            self._stall_count += 1
            self._divergence_count = 0
        elif status == ConvergenceStatus.DIVERGING:
            self._divergence_count += 1
            self._stall_count = 0
        else:
            self._stall_count = max(0, self._stall_count - 1)
            self._divergence_count = max(0, self._divergence_count - 1)

        if self._stall_count >= cfg.intervention_stall_iterations:
            return True
        if self._divergence_count >= cfg.intervention_divergence_count:
            return True

        return False

    def reset(self):
        """重置预测器状态。"""
        self._errors.clear()
        self._timestamps.clear()
        self._iteration = 0
        self._stall_count = 0
        self._divergence_count = 0
        self._start_time = None
        logger.info("ConvergencePredictor: Reset")

    def get_error_history(self) -> list:
        """获取误差历史。"""
        return list(self._errors)

    def get_current_iteration(self) -> int:
        """获取当前迭代次数。"""
        return self._iteration
