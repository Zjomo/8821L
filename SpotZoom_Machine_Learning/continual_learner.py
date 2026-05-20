"""
持续学习适配器 (ContinualLearner)

灵感来源:
- Elastic Weight Consolidation (EWC, Kirkpatrick et al. 2017) — 弹性权重巩固，
  通过 Fisher 信息矩阵估计参数重要性，防止灾难性遗忘
- Experience Replay (Rolnick et al. 2019) — 经验回放缓冲区，
  混合新旧样本进行训练以保持对历史分布的记忆
- Page-Hinkley Change Detection (Page 1954) — Page-Hinkley 变点检测，
  一种累积和 (CUSUM) 类型的在线变点检测方法，适用于工业过程监控
- Online Statistics (Welford 1962) — Welford 在线算法，
  单遍扫描计算均值和方差，适合流式数据处理

算法原理:
- Elastic Weight Consolidation (EWC) — 利用 Fisher 信息矩阵对角线作为
  参数重要性权重，在参数更新时对重要参数施加更大的正则化惩罚，
  从而在适应新数据的同时保护已学到的关键知识
- Experience Replay Buffer — 维护一个固定大小的环形缓冲区，存储历史
  代表性样本，在每次更新时从缓冲区中采样旧样本与新样本混合计算统计量
- Page-Hinkley Test — 监控观测值的累积偏差，当累积偏差超过自适应阈值时
  触发漂移告警；适用于检测光学系统参数的渐变和突变漂移
- Welford Online Algorithm — 在线递推更新均值和方差，无需存储全部历史数据，
  时间复杂度 O(1) 每样本
- Adaptive Threshold — 基于在线统计量自动调整检测阈值，适应不同工况

功能:
- 在线持续学习：随着光学系统运行，自动适应检测参数
- 漂移检测：通过 Page-Hinkley 检测光学系统参数漂移
- 经验回放：维护代表性样本缓冲区，防止灾难性遗忘
- 参数重要性估计：通过 EWC 机制保护关键参数
- 自适应阈值：根据在线统计自动调整检测灵敏度
- 生成持续学习分析报告

依赖: numpy, logging, dataclasses (无深度学习框架依赖)
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger("SpotZoom.ContinualLearner")


# ---------------------------------------------------------------------------
# 数据类定义
# ---------------------------------------------------------------------------


@dataclass
class ContinualConfig:
    """持续学习配置。

    Attributes
    ----------
    feature_dim : int
        特征维度 (默认 6: [cx, cy, sigma_x, sigma_y, intensity, ellipticity])。
    replay_buffer_size : int
        经验回放缓冲区最大容量。
    ewc_lambda : float
        EWC 正则化系数 (越大越保护旧参数)。
    ewc_fisher_decay : float
        Fisher 信息矩阵的指数衰减率 (0~1)。
    ph_threshold : float
        Page-Hinkley 检测阈值 (越小越敏感)。
    ph_alpha : float
        Page-Hinkley 累积偏差的遗忘因子 (0~1)。
    adaptation_lr : float
        参数自适应学习率。
    min_samples_for_drift : int
        触发漂移检测所需的最小样本数。
    adaptive_threshold_lr : float
        自适应阈值调整的学习率。
    warmup_samples : int
        预热阶段样本数 (预热期间不触发漂移告警)。
    """
    feature_dim: int = 6
    replay_buffer_size: int = 200
    ewc_lambda: float = 100.0
    ewc_fisher_decay: float = 0.99
    ph_threshold: float = 5.0
    ph_alpha: float = 0.005
    adaptation_lr: float = 0.01
    min_samples_for_drift: int = 30
    adaptive_threshold_lr: float = 0.001
    warmup_samples: int = 50


@dataclass
class ReplaySample:
    """经验回放样本。

    Attributes
    ----------
    features : np.ndarray
        光斑特征向量，形状 (feature_dim,)。
    timestamp : float
        采样时间戳 (秒)。
    confidence : float
        检测置信度 [0, 1]。
    label : int
        样本标签 (0=正常, 1=异常)。
    metadata : Dict[str, float]
        附加元数据 (如帧号、温度等)。
    """
    features: np.ndarray = field(default_factory=lambda: np.zeros(6))
    timestamp: float = 0.0
    confidence: float = 1.0
    label: int = 0
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class DriftEvent:
    """漂移事件。

    Attributes
    ----------
    detected : bool
        是否检测到漂移。
    timestamp : float
        漂移检测时间戳。
    drift_magnitude : float
        漂移幅度 (标准化)。
    drift_direction : np.ndarray
        漂移方向向量，形状 (feature_dim,)。
    ph_statistic : float
        Page-Hinkley 检验统计量。
    ph_threshold_used : float
        使用的检测阈值。
    affected_features : List[int]
        受影响最大的特征索引列表。
    severity : str
        漂移严重程度 ('mild', 'moderate', 'severe')。
    description : str
        漂移描述信息。
    """
    detected: bool = False
    timestamp: float = 0.0
    drift_magnitude: float = 0.0
    drift_direction: np.ndarray = field(default_factory=lambda: np.zeros(6))
    ph_statistic: float = 0.0
    ph_threshold_used: float = 0.0
    affected_features: List[int] = field(default_factory=list)
    severity: str = "none"
    description: str = ""


@dataclass
class ContinualState:
    """持续学习器当前状态。

    Attributes
    ----------
    current_parameters : np.ndarray
        当前检测参数向量，形状 (feature_dim,)。
    reference_parameters : np.ndarray
        参考参数向量 (EWC 锚点)，形状 (feature_dim,)。
    fisher_information : np.ndarray
        Fisher 信息对角线，形状 (feature_dim,)。
    online_mean : np.ndarray
        在线均值，形状 (feature_dim,)。
    online_var : np.ndarray
        在线方差，形状 (feature_dim,)。
    online_count : int
        在线统计累计样本数。
    replay_buffer_size : int
        当前回放缓冲区中的样本数。
    total_updates : int
        累计更新次数。
    total_drift_events : int
        累计漂移事件数。
    current_threshold : float
        当前自适应阈值。
    is_adapting : bool
        是否正在执行自适应。
    adaptation_progress : float
        自适应进度 [0, 1]。
    """
    current_parameters: np.ndarray = field(default_factory=lambda: np.zeros(6))
    reference_parameters: np.ndarray = field(default_factory=lambda: np.zeros(6))
    fisher_information: np.ndarray = field(default_factory=lambda: np.zeros(6))
    online_mean: np.ndarray = field(default_factory=lambda: np.zeros(6))
    online_var: np.ndarray = field(default_factory=lambda: np.ones(6))
    online_count: int = 0
    replay_buffer_size: int = 0
    total_updates: int = 0
    total_drift_events: int = 0
    current_threshold: float = 5.0
    is_adapting: bool = False
    adaptation_progress: float = 0.0


@dataclass
class ContinualReport:
    """持续学习分析报告。

    Attributes
    ----------
    total_samples_processed : int
        总处理样本数。
    total_updates : int
        总更新次数。
    total_drift_events : int
        总漂移事件数。
    current_mean : np.ndarray
        当前在线均值。
    current_variance : np.ndarray
        当前在线方差。
    current_parameters : np.ndarray
        当前参数。
    reference_parameters : np.ndarray
        参考参数。
    parameter_drift : np.ndarray
        参数漂移量 (current - reference)。
    fisher_information : np.ndarray
        Fisher 信息矩阵对角线。
    replay_buffer_utilization : float
        回放缓冲区利用率 [0, 1]。
    current_threshold : float
        当前自适应阈值。
    last_drift_event : Optional[DriftEvent]
        最近一次漂移事件。
    drift_history : List[DriftEvent]
        漂移事件历史。
    adaptation_count : int
        自适应执行次数。
    mean_update_magnitude : float
        平均参数更新幅度。
    feature_stability_scores : np.ndarray
        各特征的稳定性分数 (0~1, 越大越稳定)。
    """
    total_samples_processed: int = 0
    total_updates: int = 0
    total_drift_events: int = 0
    current_mean: np.ndarray = field(default_factory=lambda: np.zeros(6))
    current_variance: np.ndarray = field(default_factory=lambda: np.ones(6))
    current_parameters: np.ndarray = field(default_factory=lambda: np.zeros(6))
    reference_parameters: np.ndarray = field(default_factory=lambda: np.zeros(6))
    parameter_drift: np.ndarray = field(default_factory=lambda: np.zeros(6))
    fisher_information: np.ndarray = field(default_factory=lambda: np.zeros(6))
    replay_buffer_utilization: float = 0.0
    current_threshold: float = 5.0
    last_drift_event: Optional[DriftEvent] = None
    drift_history: List[DriftEvent] = field(default_factory=list)
    adaptation_count: int = 0
    mean_update_magnitude: float = 0.0
    feature_stability_scores: np.ndarray = field(default_factory=lambda: np.ones(6))


# ---------------------------------------------------------------------------
# 持续学习器
# ---------------------------------------------------------------------------


class ContinualLearner:
    """持续/在线学习适配器 — 适应光学系统的在线参数漂移。

    通过 Elastic Weight Consolidation (EWC) 保护关键参数、经验回放缓冲区
    维持历史记忆、Page-Hinkley 变点检测发现系统漂移、Welford 在线统计算法
    实时跟踪特征分布，实现光学对准系统的在线自适应。

    Parameters
    ----------
    config : ContinualConfig, optional
        持续学习配置。为 None 时使用默认配置。
    """

    def __init__(self, config: Optional[ContinualConfig] = None):
        self._config = config if config is not None else ContinualConfig()
        self._dim = self._config.feature_dim

        # --- 内部状态 ---
        self._reset_internal_state()

        LOGGER.info(
            "ContinualLearner 初始化完成: feature_dim=%d, "
            "replay_buffer_size=%d, ewc_lambda=%.2f, ph_threshold=%.2f",
            self._dim,
            self._config.replay_buffer_size,
            self._config.ewc_lambda,
            self._config.ph_threshold,
        )

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """重置持续学习器到初始状态。

        清空所有内部状态，包括回放缓冲区、在线统计量、Fisher 信息矩阵、
        Page-Hinkley 检测器状态和漂移历史。参数恢复为零向量。
        """
        self._reset_internal_state()
        LOGGER.info("ContinualLearner 已重置")

    def update(self, sample: ReplaySample) -> None:
        """喂入新观测样本，执行在线学习更新。

        执行以下步骤:
        1. 验证输入特征维度
        2. 更新在线统计量 (Welford 算法)
        3. 将样本存入经验回放缓冲区
        4. 更新 Fisher 信息矩阵 (EWC 参数重要性)
        5. 更新 Page-Hinkley 检测器状态
        6. 执行参数自适应微调

        Parameters
        ----------
        sample : ReplaySample
            新的观测样本，包含特征向量、时间戳和置信度。
        """
        features = np.asarray(sample.features, dtype=np.float64)
        if features.shape != (self._dim,):
            LOGGER.warning(
                "特征维度不匹配: 期望 (%d,), 实际 %s, 跳过此样本",
                self._dim,
                features.shape,
            )
            return

        self._total_updates += 1

        # 1. Welford 在线统计更新
        self._update_online_stats(features)

        # 2. 经验回放缓冲区写入
        self._add_to_replay_buffer(sample)

        # 3. Fisher 信息矩阵更新 (EWC)
        self._update_fisher_information(features)

        # 4. Page-Hinkley 检测器更新
        self._update_page_hinkley(features)

        # 5. 参数自适应
        self._adaptive_parameter_update(features)

        if self._total_updates % 100 == 0:
            LOGGER.debug(
                "持续学习进度: updates=%d, buffer=%d/%d, "
                "mean_norm=%.4f, threshold=%.4f",
                self._total_updates,
                len(self._replay_buffer),
                self._config.replay_buffer_size,
                np.linalg.norm(self._online_mean),
                self._current_threshold,
            )

    def detect_drift(self) -> Optional[DriftEvent]:
        """检测光学系统是否发生参数漂移。

        基于 Page-Hinkley 变点检测算法，结合自适应阈值，
        判断当前观测分布是否相对于参考分布发生了显著偏移。

        Returns
        -------
        DriftEvent or None
            若检测到漂移则返回 DriftEvent，否则返回 None。
        """
        if self._online_count < self._config.min_samples_for_drift:
            return None

        if self._online_count < self._config.warmup_samples:
            LOGGER.debug(
                "预热阶段 (count=%d < warmup=%d), 跳过漂移检测",
                self._online_count,
                self._config.warmup_samples,
            )
            return None

        # 计算各特征的 PH 统计量
        ph_stats = self._ph_cumsum  # 形状 (feature_dim,)
        ph_min = self._ph_min_cumsum  # 形状 (feature_dim,)

        # 最大偏差: max(T_n - min_{k<=n} T_k)
        deviations = ph_stats - ph_min

        # 找到偏差最大的特征
        max_idx = int(np.argmax(deviations))
        max_deviation = float(deviations[max_idx])

        # 自适应阈值
        effective_threshold = self._current_threshold

        if max_deviation > effective_threshold:
            # 检测到漂移
            drift_magnitude = float(max_deviation / max(effective_threshold, 1e-10))

            # 漂移方向: 当前均值与参考参数的差
            drift_dir = self._online_mean - self._reference_params
            drift_dir_norm = np.linalg.norm(drift_dir)
            if drift_dir_norm > 1e-10:
                drift_dir = drift_dir / drift_dir_norm

            # 受影响特征: 偏差超过阈值 50% 的特征
            affected = [
                int(i)
                for i in range(self._dim)
                if deviations[i] > effective_threshold * 0.5
            ]

            # 严重程度
            if drift_magnitude < 2.0:
                severity = "mild"
            elif drift_magnitude < 5.0:
                severity = "moderate"
            else:
                severity = "severe"

            event = DriftEvent(
                detected=True,
                timestamp=time.time(),
                drift_magnitude=drift_magnitude,
                drift_direction=drift_dir.copy(),
                ph_statistic=max_deviation,
                ph_threshold_used=effective_threshold,
                affected_features=affected,
                severity=severity,
                description=(
                    f"特征 {max_idx} 检测到{severity}漂移, "
                    f"幅度={drift_magnitude:.2f}, "
                    f"PH统计量={max_deviation:.4f}"
                ),
            )

            self._total_drift_events += 1
            self._drift_history.append(event)

            # 保留最近 50 个漂移事件
            if len(self._drift_history) > 50:
                self._drift_history = self._drift_history[-50:]

            # 漂移后重置 PH 检测器
            self._reset_page_hinkley()

            LOGGER.warning(
                "检测到光学系统漂移: severity=%s, magnitude=%.2f, "
                "affected_features=%s",
                severity,
                drift_magnitude,
                affected,
            )

            return event

        return None

    def adapt(self) -> ContinualState:
        """执行参数自适应调整。

        当检测到漂移或定期维护时调用。利用经验回放缓冲区中的历史样本
        和 EWC 正则化约束，计算新的检测参数。

        自适应策略:
        1. 从回放缓冲区采样计算历史分布统计量
        2. 结合当前在线统计量计算目标参数
        3. 通过 EWC 正则化约束更新步长 (重要参数更新更小)
        4. 更新参考参数锚点

        Returns
        -------
        ContinualState
            自适应后的持续学习器状态。
        """
        self._is_adapting = True
        self._adaptation_progress = 0.0

        try:
            if len(self._replay_buffer) < 5:
                LOGGER.debug(
                    "回放缓冲区样本不足 (%d < 5), 跳过自适应",
                    len(self._replay_buffer),
                )
                return self.get_state()

            # 1. 从回放缓冲区采样
            replay_features = self._sample_replay_features()
            replay_mean = np.mean(replay_features, axis=0)
            replay_var = np.var(replay_features, axis=0)

            # 2. 计算目标参数: 在线均值与回放均值的加权混合
            # 权重取决于回放缓冲区利用率
            buffer_ratio = len(self._replay_buffer) / max(
                self._config.replay_buffer_size, 1
            )
            alpha = self._config.adaptation_lr * (1.0 + buffer_ratio)

            # 目标参数: 向在线均值方向移动，但受 EWC 约束
            target = self._online_mean.copy()

            # 3. EWC 正则化: 参数更新步长受 Fisher 信息约束
            # 重要参数 (高 Fisher) 更新更慢
            fisher_normalized = self._fisher_diag.copy()
            fisher_max = np.max(fisher_normalized)
            if fisher_max > 1e-10:
                fisher_normalized = fisher_normalized / fisher_max

            # EWC 惩罚因子: fisher 越大，步长越小
            ewc_penalty = 1.0 / (1.0 + self._config.ewc_lambda * fisher_normalized)

            # 参数更新
            delta = target - self._current_params
            update_step = alpha * delta * ewc_penalty

            self._current_params = self._current_params + update_step
            self._adaptation_progress = 1.0
            self._adaptation_count += 1

            # 4. 更新参考参数锚点 (缓慢移动)
            anchor_lr = 0.01
            self._reference_params = (
                (1.0 - anchor_lr) * self._reference_params
                + anchor_lr * self._current_params
            )

            # 5. 自适应阈值调整
            self._adapt_threshold(replay_var)

            LOGGER.info(
                "参数自适应完成: update_norm=%.6f, "
                "buffer_util=%.2f, threshold=%.4f",
                np.linalg.norm(update_step),
                buffer_ratio,
                self._current_threshold,
            )

        finally:
            self._is_adapting = False

        return self.get_state()

    def analyze(self) -> ContinualReport:
        """生成持续学习分析报告。

        汇总当前学习器状态、在线统计量、漂移历史、缓冲区利用率等
        信息，生成完整的分析报告。

        Returns
        -------
        ContinualReport
            持续学习分析报告。
        """
        param_drift = self._current_params - self._reference_params

        # 特征稳定性分数: 基于在线变异系数的倒数 (归一化到 [0, 1])
        stability_scores = np.ones(self._dim)
        for i in range(self._dim):
            if self._online_var[i] > 1e-10 and abs(self._online_mean[i]) > 1e-10:
                cv = np.sqrt(self._online_var[i]) / abs(self._online_mean[i])
                # CV 越小越稳定; 映射到 [0, 1]
                stability_scores[i] = float(np.exp(-cv))
            elif self._online_var[i] <= 1e-10:
                stability_scores[i] = 1.0  # 零方差 = 完全稳定

        # 平均参数更新幅度
        mean_update_mag = 0.0
        if self._adaptation_count > 0 and len(self._update_history) > 0:
            mean_update_mag = float(np.mean(self._update_history))

        buffer_util = (
            len(self._replay_buffer) / max(self._config.replay_buffer_size, 1)
        )

        report = ContinualReport(
            total_samples_processed=self._online_count,
            total_updates=self._total_updates,
            total_drift_events=self._total_drift_events,
            current_mean=self._online_mean.copy(),
            current_variance=self._online_var.copy(),
            current_parameters=self._current_params.copy(),
            reference_parameters=self._reference_params.copy(),
            parameter_drift=param_drift.copy(),
            fisher_information=self._fisher_diag.copy(),
            replay_buffer_utilization=buffer_util,
            current_threshold=self._current_threshold,
            last_drift_event=(
                self._drift_history[-1] if self._drift_history else None
            ),
            drift_history=list(self._drift_history),
            adaptation_count=self._adaptation_count,
            mean_update_magnitude=mean_update_mag,
            feature_stability_scores=stability_scores.copy(),
        )

        LOGGER.debug(
            "分析报告: samples=%d, drifts=%d, buffer_util=%.2f, "
            "param_drift_norm=%.4f",
            report.total_samples_processed,
            report.total_drift_events,
            report.replay_buffer_utilization,
            np.linalg.norm(param_drift),
        )

        return report

    def get_state(self) -> ContinualState:
        """获取当前持续学习器状态。

        Returns
        -------
        ContinualState
            当前状态快照。
        """
        return ContinualState(
            current_parameters=self._current_params.copy(),
            reference_parameters=self._reference_params.copy(),
            fisher_information=self._fisher_diag.copy(),
            online_mean=self._online_mean.copy(),
            online_var=self._online_var.copy(),
            online_count=self._online_count,
            replay_buffer_size=len(self._replay_buffer),
            total_updates=self._total_updates,
            total_drift_events=self._total_drift_events,
            current_threshold=self._current_threshold,
            is_adapting=self._is_adapting,
            adaptation_progress=self._adaptation_progress,
        )

    # ------------------------------------------------------------------
    # 内部方法: 状态管理
    # ------------------------------------------------------------------

    def _reset_internal_state(self) -> None:
        """重置所有内部状态变量。"""
        dim = self._dim

        # 检测参数
        self._current_params: np.ndarray = np.zeros(dim, dtype=np.float64)
        self._reference_params: np.ndarray = np.zeros(dim, dtype=np.float64)

        # Fisher 信息矩阵对角线 (EWC)
        self._fisher_diag: np.ndarray = np.zeros(dim, dtype=np.float64)

        # Welford 在线统计
        self._online_mean: np.ndarray = np.zeros(dim, dtype=np.float64)
        self._online_m2: np.ndarray = np.zeros(dim, dtype=np.float64)  # Welford M2
        self._online_var: np.ndarray = np.ones(dim, dtype=np.float64)
        self._online_count: int = 0

        # 经验回放缓冲区 (环形缓冲区)
        self._replay_buffer: Deque[ReplaySample] = deque(
            maxlen=self._config.replay_buffer_size
        )

        # Page-Hinkley 检测器状态
        self._ph_cumsum: np.ndarray = np.zeros(dim, dtype=np.float64)
        self._ph_min_cumsum: np.ndarray = np.zeros(dim, dtype=np.float64)

        # 自适应阈值
        self._current_threshold: float = self._config.ph_threshold

        # 计数器
        self._total_updates: int = 0
        self._total_drift_events: int = 0
        self._adaptation_count: int = 0

        # 漂移历史
        self._drift_history: List[DriftEvent] = []

        # 自适应状态
        self._is_adapting: bool = False
        self._adaptation_progress: float = 0.0

        # 更新历史 (用于统计)
        self._update_history: List[float] = []

    # ------------------------------------------------------------------
    # 内部方法: Welford 在线统计
    # ------------------------------------------------------------------

    def _update_online_stats(self, features: np.ndarray) -> None:
        """使用 Welford 在线算法更新均值和方差。

        Welford 算法在数值上比朴素两遍扫描更稳定，
        且只需单遍扫描，适合流式数据。

        Parameters
        ----------
        features : np.ndarray
            新样本特征向量，形状 (feature_dim,)。
        """
        self._online_count += 1
        n = self._online_count

        # Welford 递推公式
        delta = features - self._online_mean
        self._online_mean = self._online_mean + delta / n
        delta2 = features - self._online_mean
        self._online_m2 = self._online_m2 + delta * delta2

        # 无偏方差估计
        if n > 1:
            self._online_var = self._online_m2 / (n - 1)
        else:
            self._online_var = np.zeros(self._dim, dtype=np.float64)

    # ------------------------------------------------------------------
    # 内部方法: 经验回放缓冲区
    # ------------------------------------------------------------------

    def _add_to_replay_buffer(self, sample: ReplaySample) -> None:
        """将样本添加到经验回放缓冲区。

        当缓冲区已满时，自动淘汰最旧的样本 (FIFO 策略)。

        Parameters
        ----------
        sample : ReplaySample
            待存储的样本。
        """
        self._replay_buffer.append(sample)

    def _sample_replay_features(
        self, n_samples: Optional[int] = None
    ) -> np.ndarray:
        """从回放缓冲区中采样特征矩阵。

        Parameters
        ----------
        n_samples : int, optional
            采样数量。为 None 时使用全部缓冲区样本。

        Returns
        -------
        np.ndarray
            采样特征矩阵，形状 (n_samples, feature_dim)。
        """
        buffer_list = list(self._replay_buffer)
        n = len(buffer_list)

        if n == 0:
            return np.zeros((0, self._dim), dtype=np.float64)

        if n_samples is not None and n_samples < n:
            # 随机采样 (无放回)
            indices = np.random.choice(n, size=n_samples, replace=False)
            buffer_list = [buffer_list[i] for i in indices]

        features = np.array(
            [s.features for s in buffer_list], dtype=np.float64
        )
        return features

    # ------------------------------------------------------------------
    # 内部方法: Fisher 信息矩阵 (EWC)
    # ------------------------------------------------------------------

    def _update_fisher_information(self, features: np.ndarray) -> None:
        """更新 Fisher 信息矩阵对角线估计。

        在 EWC 中，Fisher 信息矩阵的对角线元素衡量每个参数对
        观测分布的敏感度。这里使用梯度的平方作为经验 Fisher 估计:
            F_i ≈ (x_i - mu_i)^2 / var_i

        使用指数移动平均 (EMA) 进行平滑更新，旧信息按
        ewc_fisher_decay 衰减。

        Parameters
        ----------
        features : np.ndarray
            新样本特征向量，形状 (feature_dim,)。
        """
        decay = self._config.ewc_fisher_decay

        # 经验 Fisher 估计: 残差平方 / 方差
        residuals = features - self._online_mean
        safe_var = np.maximum(self._online_var, 1e-10)
        empirical_fisher = (residuals ** 2) / safe_var

        # EMA 平滑更新
        self._fisher_diag = decay * self._fisher_diag + (1.0 - decay) * empirical_fisher

    # ------------------------------------------------------------------
    # 内部方法: Page-Hinkley 变点检测
    # ------------------------------------------------------------------

    def _update_page_hinkley(self, features: np.ndarray) -> None:
        """更新 Page-Hinkley 检测器状态。

        Page-Hinkley 检测器监控观测值的累积偏差:
            T_n = sum_{k=1}^{n} (x_k - x_bar - delta)
            U_n = min_{k<=n} T_k

        当 T_n - U_n > threshold 时触发告警。

        使用遗忘因子 alpha 实现对历史累积偏差的指数衰减，
        使检测器对近期变化更敏感。

        Parameters
        ----------
        features : np.ndarray
            新样本特征向量，形状 (feature_dim,)。
        """
        alpha = self._config.ph_alpha

        # 偏差: 当前特征与在线均值的差
        deviation = features - self._online_mean

        # 累积和 (带遗忘)
        self._ph_cumsum = (1.0 - alpha) * self._ph_cumsum + deviation

        # 更新最小累积和
        self._ph_min_cumsum = np.minimum(self._ph_min_cumsum, self._ph_cumsum)

    def _reset_page_hinkley(self) -> None:
        """重置 Page-Hinkley 检测器状态。

        在检测到漂移后调用，重新开始累积偏差。
        """
        self._ph_cumsum = np.zeros(self._dim, dtype=np.float64)
        self._ph_min_cumsum = np.zeros(self._dim, dtype=np.float64)

    # ------------------------------------------------------------------
    # 内部方法: 参数自适应
    # ------------------------------------------------------------------

    def _adaptive_parameter_update(self, features: np.ndarray) -> None:
        """执行单步参数自适应更新。

        在每次收到新样本时，对当前参数进行小幅调整。
        调整幅度受 EWC 正则化约束。

        Parameters
        ----------
        features : np.ndarray
            新样本特征向量，形状 (feature_dim,)。
        """
        if self._online_count < self._config.warmup_samples:
            # 预热阶段: 直接跟踪均值
            self._current_params = self._online_mean.copy()
            self._reference_params = self._online_mean.copy()
            return

        # 计算目标方向
        delta = features - self._current_params

        # EWC 正则化: Fisher 越大，更新越慢
        fisher_max = np.max(self._fisher_diag)
        if fisher_max > 1e-10:
            fisher_norm = self._fisher_diag / fisher_max
        else:
            fisher_norm = np.ones(self._dim, dtype=np.float64)

        ewc_weight = 1.0 / (1.0 + self._config.ewc_lambda * fisher_norm)

        # 自适应步长: 随着样本增多逐渐减小
        decay_lr = self._config.adaptation_lr / (
            1.0 + self._online_count * 1e-4
        )

        # 参数更新
        update_step = decay_lr * delta * ewc_weight
        self._current_params = self._current_params + update_step

        # 记录更新幅度
        step_norm = float(np.linalg.norm(update_step))
        self._update_history.append(step_norm)
        if len(self._update_history) > 1000:
            self._update_history = self._update_history[-1000:]

    def _adapt_threshold(self, replay_var: np.ndarray) -> None:
        """自适应调整 Page-Hinkley 检测阈值。

        基于回放缓冲区中的方差信息调整阈值:
        - 方差大时提高阈值 (减少误报)
        - 方差小时降低阈值 (提高灵敏度)

        Parameters
        ----------
        replay_var : np.ndarray
            回放缓冲区中的特征方差，形状 (feature_dim,)。
        """
        # 平均变异系数
        safe_mean = np.maximum(np.abs(self._online_mean), 1e-10)
        safe_std = np.sqrt(np.maximum(replay_var, 1e-10))
        mean_cv = float(np.mean(safe_std / safe_mean))

        # 阈值调整: CV 越大，阈值越高
        base_threshold = self._config.ph_threshold
        target_threshold = base_threshold * (1.0 + mean_cv)

        # 平滑过渡
        lr = self._config.adaptive_threshold_lr
        self._current_threshold = (
            (1.0 - lr) * self._current_threshold + lr * target_threshold
        )

        # 阈值下限保护
        self._current_threshold = max(self._current_threshold, base_threshold * 0.5)
        self._current_threshold = min(self._current_threshold, base_threshold * 5.0)
