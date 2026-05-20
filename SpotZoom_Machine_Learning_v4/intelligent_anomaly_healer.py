"""
智能异常诊断与自愈系统 (Intelligent Anomaly Diagnosis & Self-Healing System)

灵感来源: PyOD (https://github.com/yzhao062/pyod)
           sktime (https://github.com/sktime/sktime)
           工业预测性维护框架

核心思想:
──────────────────────────────────────────────────────────────────────
PyOD 是最全面的 Python 异常检测库，提供 30+ 种检测算法。
sktime 提供了统一的时间序列分析框架。

本模块将这些思想适配到光斑对准系统:
- 孤立森林 (Isolation Forest) 检测光斑位置/质量异常
- 时间序列分解 (STL) 识别趋势和季节性异常
- 自适应阈值动态调整告警灵敏度
- 自动诊断根因并推荐自愈策略
- 异常模式分类 (漂移/跳动/丢失/退化)

适用场景:
- 长时间运行中的异常检测
- 硬件故障预警
- 环境变化自适应
- 自动恢复策略推荐
"""

__version__ = "1.0.0"
__author__ = "SpotZoom Team"

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any, Callable
from enum import Enum
from collections import deque
import time


class AnomalyType(Enum):
    """异常类型"""
    POSITION_DRIFT = "position_drift"       # 位置漂移
    POSITION_JUMP = "position_jump"         # 位置跳变
    SPOT_LOSS = "spot_loss"                 # 光斑丢失
    QUALITY_DEGRADATION = "quality_degrade" # 质量退化
    INTENSITY_FLICKER = "intensity_flicker" # 强度闪烁
    VIBRATION_ANOMALY = "vibration"         # 振动异常
    HARDWARE_STALL = "hardware_stall"       # 硬件卡死
    COMMUNICATION_ERROR = "comm_error"      # 通信错误
    UNKNOWN = "unknown"


class SeverityLevel(Enum):
    """严重程度"""
    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class HealingActionType(Enum):
    """自愈动作类型"""
    RECALIBRATE = "recalibrate"             # 重新校准
    RESET_CONTROLLER = "reset_controller"   # 重置控制器
    SWITCH_DETECTOR = "switch_detector"     # 切换检测器
    ADJUST_GAINS = "adjust_gains"           # 调整增益
    PAUSE_AND_RESUME = "pause_resume"       # 暂停后恢复
    FULL_RESTART = "full_restart"           # 完全重启
    LOG_AND_CONTINUE = "log_continue"       # 仅记录
    ESCALATE = "escalate"                   # 上报


@dataclass
class HealingAction:
    """自愈动作"""
    action_type: HealingActionType
    description: str
    confidence: float = 0.0
    estimated_recovery_time_ms: float = 0.0
    parameters: Dict = field(default_factory=dict)


@dataclass
class AnomalyReport:
    """异常报告"""
    anomaly_type: AnomalyType
    severity: SeverityLevel
    description: str
    timestamp: float
    detected_value: float
    expected_range: Tuple[float, float]
    anomaly_score: float          # [0, 1], 越大越异常
    recommended_actions: List[HealingAction]
    context: Dict = field(default_factory=dict)


@dataclass
class AnomalyHealerConfig:
    """异常诊断与自愈配置"""
    # 孤立森林参数
    isolation_n_estimators: int = 100
    isolation_contamination: float = 0.05
    isolation_max_samples: int = 64

    # 时间序列参数
    stl_window_size: int = 20
    stl_trend_degree: int = 1
    stl_seasonal_period: int = 10

    # 自适应阈值参数
    threshold_initial_sigma: float = 3.0
    threshold_min_sigma: float = 1.5
    threshold_max_sigma: float = 5.0
    threshold_adaptation_rate: float = 0.01

    # 历史缓冲区
    history_length: int = 500
    feature_window: int = 50

    # 自愈参数
    auto_heal_enabled: bool = True
    max_auto_heals_per_hour: int = 10
    consecutive_anomaly_threshold: int = 3

    # 特征工程
    feature_names: List[str] = field(default_factory=lambda: [
        "position_x", "position_y", "intensity", "quality_score",
        "velocity_x", "velocity_y", "size_x", "size_y"
    ])


class _IsolationTree:
    """简化版孤立树节点"""

    def __init__(self, max_depth: int = 8):
        self.max_depth = max_depth
        self.split_feature: Optional[int] = None
        self.split_value: Optional[float] = None
        self.left: Optional['_IsolationTree'] = None
        self.right: Optional['_IsolationTree'] = None
        self.is_leaf = False
        self.size = 0

    def fit(self, X: np.ndarray, depth: int = 0) -> None:
        n_samples, n_features = X.shape

        if depth >= self.max_depth or n_samples <= 1:
            self.is_leaf = True
            self.size = n_samples
            return

        # 随机选择分裂特征和值
        self.split_feature = np.random.randint(0, n_features)
        min_val = X[:, self.split_feature].min()
        max_val = X[:, self.split_feature].max()

        if min_val == max_val:
            self.is_leaf = True
            self.size = n_samples
            return

        self.split_value = np.random.uniform(min_val, max_val)

        left_mask = X[:, self.split_feature] < self.split_value
        right_mask = ~left_mask

        self.left = _IsolationTree(self.max_depth)
        self.right = _IsolationTree(self.max_depth)
        self.left.fit(X[left_mask], depth + 1)
        self.right.fit(X[right_mask], depth + 1)
        self.size = n_samples

    def path_length(self, x: np.ndarray, depth: int = 0) -> float:
        if self.is_leaf:
            return depth + _c(n=self.size)
        if x[self.split_feature] < self.split_value:
            return self.left.path_length(x, depth + 1)
        else:
            return self.right.path_length(x, depth + 1)


def _c(n: int) -> float:
    """孤立树平均路径长度 (失败函数)"""
    if n <= 1:
        return 0
    return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n


class _IsolationForest:
    """简化版孤立森林"""

    def __init__(self, n_estimators: int = 100, max_samples: int = 64,
                 max_depth: int = 8):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.trees: List[_IsolationTree] = []
        self._mean_path_length: float = 0.0

    def fit(self, X: np.ndarray) -> None:
        n = min(self.max_samples, X.shape[0])
        self.trees = []
        total_path = 0

        for _ in range(self.n_estimators):
            indices = np.random.choice(X.shape[0], n, replace=True)
            tree = _IsolationTree(self.max_depth)
            tree.fit(X[indices])
            self.trees.append(tree)
            total_path += _c(n)

        self._mean_path_length = total_path / self.n_estimators

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """计算异常分数 (越小越异常)"""
        scores = np.zeros(X.shape[0])
        for tree in self.trees:
            for i in range(X.shape[0]):
                scores[i] += tree.path_length(X[i])

        scores /= self.n_estimators
        # 归一化到 [0, 1], 0=正常, 1=异常
        n = min(self.max_samples, X.shape[0])
        c_n = _c(n)
        anomaly = 2 ** (-scores / c_n) if c_n > 0 else np.zeros_like(scores)
        return anomaly


class _STLDecomposer:
    """简化版 STL 时间序列分解"""

    def __init__(self, period: int = 10, trend_degree: int = 1):
        self.period = period
        self.trend_degree = trend_degree

    def decompose(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """分解为趋势、季节性和残差"""
        n = len(series)
        if n < self.period * 2:
            return (np.full(n, np.mean(series)),
                    np.zeros(n),
                    series - np.mean(series))

        # 趋势提取 (移动平均)
        trend = self._moving_average(series, self.period)

        # 去趋势
        detrended = series - trend

        # 季节性提取
        seasonal = np.zeros(n)
        for i in range(n):
            idx = i % self.period
            values = detrended[idx::self.period]
            seasonal[i] = np.median(values) if len(values) > 0 else 0

        # 残差
        residual = series - trend - seasonal

        return trend, seasonal, residual

    def _moving_average(self, series: np.ndarray, window: int) -> np.ndarray:
        """移动平均"""
        kernel = np.ones(window) / window
        return np.convolve(series, kernel, mode='same')


class IntelligentAnomalyHealer:
    """
    智能异常诊断与自愈系统

    功能:
    1. 多维特征异常检测 (孤立森林)
    2. 时间序列异常识别 (STL 分解)
    3. 自适应阈值告警
    4. 异常根因诊断
    5. 自动自愈策略推荐

    用法示例:
        healer = IntelligentAnomalyHealer(AnomalyHealerConfig())
        healer.update_features({
            "position_x": 320.5, "position_y": 240.3,
            "intensity": 1500.0, "quality_score": 0.85
        })
        report = healer.diagnose()
        if report:
            for action in report.recommended_actions:
                print(f"建议: {action.description}")
    """

    def __init__(self, config: Optional[AnomalyHealerConfig] = None):
        self.config = config or AnomalyHealerConfig()

        # 特征历史
        self._feature_buffer: Dict[str, deque] = {
            name: deque(maxlen=self.config.history_length)
            for name in self.config.feature_names
        }

        # 异常检测器
        self._isolation_forest: Optional[_IsolationForest] = None
        self._stl = _STLDecomposer(
            period=self.config.stl_seasonal_period,
            trend_degree=self.config.stl_trend_degree
        )

        # 自适应阈值
        self._thresholds: Dict[str, Dict] = {
            name: {"mean": 0.0, "std": 1.0, "sigma": self.config.threshold_initial_sigma}
            for name in self.config.feature_names
        }

        # 异常历史
        self._anomaly_history: List[AnomalyReport] = []
        self._consecutive_counts: Dict[AnomalyType, int] = {
            at: 0 for at in AnomalyType
        }

        # 自愈计数
        self._heal_counts: List[float] = []  # 时间戳列表
        self._total_anomalies = 0
        self._total_heals = 0

        # 训练状态
        self._is_trained = False
        self._samples_since_train = 0

    def update_features(self, features: Dict[str, float]) -> None:
        """
        更新特征值

        Args:
            features: 特征字典 {特征名: 值}
        """
        for name, value in features.items():
            if name in self._feature_buffer:
                self._feature_buffer[name].append(value)
                self._update_threshold(name, value)

        self._samples_since_train += 1

        # 自动训练
        if self._samples_since_train >= 50 and not self._is_trained:
            self._train_detector()

    def diagnose(self) -> Optional[AnomalyReport]:
        """
        执行异常诊断

        Returns:
            AnomalyReport 或 None (无异常时)
        """
        if not self._is_trained:
            return None

        # 收集当前特征向量
        feature_vector = self._get_current_features()
        if feature_vector is None:
            return None

        # 孤立森林检测
        anomaly_score = self._detect_anomaly_iforest(feature_vector)

        # 时间序列异常检测
        ts_anomalies = self._detect_ts_anomalies()

        # 自适应阈值检测
        threshold_anomalies = self._detect_threshold_violations()

        # 综合判断
        if anomaly_score > self.config.isolation_contamination:
            anomaly_type = self._classify_anomaly(ts_anomalies, threshold_anomalies)
            severity = self._assess_severity(anomaly_score, ts_anomalies, threshold_anomalies)
            actions = self._recommend_healing(anomaly_type, severity)

            report = AnomalyReport(
                anomaly_type=anomaly_type,
                severity=severity,
                description=self._generate_description(anomaly_type, anomaly_score),
                timestamp=time.time(),
                detected_value=anomaly_score,
                expected_range=(0, self.config.isolation_contamination),
                anomaly_score=anomaly_score,
                recommended_actions=actions,
                context={
                    "ts_anomalies": ts_anomalies,
                    "threshold_violations": threshold_anomalies,
                    "feature_vector": feature_vector.tolist()
                }
            )

            self._anomaly_history.append(report)
            self._consecutive_counts[anomaly_type] += 1
            self._total_anomalies += 1

            # 自动执行自愈
            if (self.config.auto_heal_enabled and
                    severity.value >= SeverityLevel.HIGH.value and
                    self._can_auto_heal()):
                self._execute_healing(actions[0] if actions else None)

            return report

        # 重置连续计数
        for at in AnomalyType:
            self._consecutive_counts[at] = max(0, self._consecutive_counts[at] - 1)

        return None

    def _train_detector(self) -> None:
        """训练孤立森林检测器"""
        X = self._get_feature_matrix()
        if X is None or X.shape[0] < 20:
            return

        self._isolation_forest = _IsolationForest(
            n_estimators=self.config.isolation_n_estimators,
            max_samples=min(self.config.isolation_max_samples, X.shape[0])
        )
        self._isolation_forest.fit(X)
        self._is_trained = True
        self._samples_since_train = 0

    def _get_feature_matrix(self) -> Optional[np.ndarray]:
        """获取特征矩阵"""
        min_len = min(len(buf) for buf in self._feature_buffer.values())
        if min_len < 10:
            return None

        matrix = np.column_stack([
            np.array(list(self._feature_buffer[name])[-min_len:])
            for name in self.config.feature_names
        ])
        return matrix

    def _get_current_features(self) -> Optional[np.ndarray]:
        """获取当前特征向量"""
        features = []
        for name in self.config.feature_names:
            buf = self._feature_buffer[name]
            if len(buf) == 0:
                return None
            features.append(buf[-1])
        return np.array(features)

    def _detect_anomaly_iforest(self, x: np.ndarray) -> float:
        """孤立森林异常检测"""
        if self._isolation_forest is None:
            return 0.0
        return float(self._isolation_forest.score_samples(x.reshape(1, -1))[0])

    def _detect_ts_anomalies(self) -> Dict[str, float]:
        """时间序列异常检测"""
        anomalies = {}
        for name in self.config.feature_names:
            series = np.array(self._feature_buffer[name])
            if len(series) < self.config.stl_window_size * 2:
                continue

            recent = series[-self.config.stl_window_size:]
            trend, seasonal, residual = self._stl.decompose(recent)

            if len(residual) > 0:
                std = np.std(residual)
                if std > 1e-10:
                    last_residual = abs(residual[-1]) / std
                    anomalies[name] = float(last_residual)

        return anomalies

    def _detect_threshold_violations(self) -> Dict[str, Dict]:
        """自适应阈值违规检测"""
        violations = {}
        for name in self.config.feature_names:
            buf = self._feature_buffer[name]
            if len(buf) < 5:
                continue

            value = buf[-1]
            t = self._thresholds[name]
            lower = t["mean"] - t["sigma"] * t["std"]
            upper = t["mean"] + t["sigma"] * t["std"]

            if value < lower or value > upper:
                violations[name] = {
                    "value": value,
                    "range": (lower, upper),
                    "deviation_sigma": abs(value - t["mean"]) / max(t["std"], 1e-10)
                }

        return violations

    def _classify_anomaly(self, ts_anomalies: Dict,
                          threshold_violations: Dict) -> AnomalyType:
        """分类异常类型"""
        # 检查位置相关异常
        pos_features = ["position_x", "position_y"]
        vel_features = ["velocity_x", "velocity_y"]

        pos_violations = {k: v for k, v in threshold_violations.items()
                         if k in pos_features}
        vel_violations = {k: v for k, v in threshold_violations.items()
                         if k in vel_features}

        if "intensity" in threshold_violations:
            if threshold_violations["intensity"]["value"] < 1e-3:
                return AnomalyType.SPOT_LOSS
            return AnomalyType.INTENSITY_FLICKER

        if "quality_score" in threshold_violations:
            return AnomalyType.QUALITY_DEGRADATION

        if pos_violations:
            max_dev = max(v["deviation_sigma"] for v in pos_violations.values())
            if max_dev > 5.0:
                return AnomalyType.POSITION_JUMP
            return AnomalyType.POSITION_DRIFT

        if vel_violations:
            return AnomalyType.VIBRATION_ANOMALY

        return AnomalyType.UNKNOWN

    def _assess_severity(self, anomaly_score: float,
                         ts_anomalies: Dict,
                         threshold_violations: Dict) -> SeverityLevel:
        """评估严重程度"""
        max_ts = max(ts_anomalies.values()) if ts_anomalies else 0
        max_tv = max(v["deviation_sigma"] for v in threshold_violations.values()) \
            if threshold_violations else 0

        combined = anomaly_score * 0.3 + max_ts * 0.3 + max_tv * 0.4

        if combined > 0.9:
            return SeverityLevel.CRITICAL
        elif combined > 0.7:
            return SeverityLevel.HIGH
        elif combined > 0.5:
            return SeverityLevel.MEDIUM
        elif combined > 0.3:
            return SeverityLevel.LOW
        return SeverityLevel.INFO

    def _recommend_healing(self, anomaly_type: AnomalyType,
                           severity: SeverityLevel) -> List[HealingAction]:
        """推荐自愈策略"""
        actions = []

        if anomaly_type == AnomalyType.POSITION_DRIFT:
            actions.append(HealingAction(
                action_type=HealingActionType.RECALIBRATE,
                description="重新校准光斑位置参考点",
                confidence=0.8,
                estimated_recovery_time_ms=500
            ))
            actions.append(HealingAction(
                action_type=HealingActionType.ADJUST_GAINS,
                description="增大 PID 增益以跟踪漂移",
                confidence=0.6,
                estimated_recovery_time_ms=100
            ))

        elif anomaly_type == AnomalyType.POSITION_JUMP:
            actions.append(HealingAction(
                action_type=HealingActionType.RESET_CONTROLLER,
                description="重置控制器积分项，防止发散",
                confidence=0.9,
                estimated_recovery_time_ms=200
            ))
            actions.append(HealingAction(
                action_type=HealingActionType.SWITCH_DETECTOR,
                description="切换到经典检测器验证",
                confidence=0.7,
                estimated_recovery_time_ms=300
            ))

        elif anomaly_type == AnomalyType.SPOT_LOSS:
            actions.append(HealingAction(
                action_type=HealingActionType.PAUSE_AND_RESUME,
                description="暂停对准，扩大搜索范围后恢复",
                confidence=0.7,
                estimated_recovery_time_ms=2000
            ))
            actions.append(HealingAction(
                action_type=HealingActionType.FULL_RESTART,
                description="完全重启检测流程",
                confidence=0.5,
                estimated_recovery_time_ms=5000
            ))

        elif anomaly_type == AnomalyType.QUALITY_DEGRADATION:
            actions.append(HealingAction(
                action_type=HealingActionType.RECALIBRATE,
                description="触发自动校准流程",
                confidence=0.75,
                estimated_recovery_time_ms=1000
            ))

        elif anomaly_type == AnomalyType.VIBRATION_ANOMALY:
            actions.append(HealingAction(
                action_type=HealingActionType.ADJUST_GAINS,
                description="降低控制器带宽以抑制振动",
                confidence=0.8,
                estimated_recovery_time_ms=100
            ))

        elif anomaly_type == AnomalyType.HARDWARE_STALL:
            actions.append(HealingAction(
                action_type=HealingActionType.ESCALATE,
                description="硬件可能卡死，需要人工检查",
                confidence=0.9,
                estimated_recovery_time_ms=0
            ))

        if not actions:
            actions.append(HealingAction(
                action_type=HealingActionType.LOG_AND_CONTINUE,
                description="记录异常，继续运行",
                confidence=1.0,
                estimated_recovery_time_ms=0
            ))

        return actions

    def _generate_description(self, anomaly_type: AnomalyType,
                              score: float) -> str:
        """生成异常描述"""
        descriptions = {
            AnomalyType.POSITION_DRIFT: f"光斑位置持续漂移 (异常分数: {score:.3f})",
            AnomalyType.POSITION_JUMP: f"光斑位置突然跳变 (异常分数: {score:.3f})",
            AnomalyType.SPOT_LOSS: f"光斑信号丢失 (异常分数: {score:.3f})",
            AnomalyType.QUALITY_DEGRADATION: f"光斑质量退化 (异常分数: {score:.3f})",
            AnomalyType.INTENSITY_FLICKER: f"光强异常波动 (异常分数: {score:.3f})",
            AnomalyType.VIBRATION_ANOMALY: f"检测到异常振动 (异常分数: {score:.3f})",
            AnomalyType.HARDWARE_STALL: f"硬件响应异常 (异常分数: {score:.3f})",
            AnomalyType.COMMUNICATION_ERROR: f"通信异常 (异常分数: {score:.3f})",
            AnomalyType.UNKNOWN: f"未知异常类型 (异常分数: {score:.3f})",
        }
        return descriptions.get(anomaly_type, f"检测到异常 (分数: {score:.3f})")

    def _update_threshold(self, name: str, value: float) -> None:
        """自适应更新阈值"""
        t = self._thresholds[name]
        buf = self._feature_buffer[name]

        if len(buf) < 5:
            return

        recent = np.array(list(buf)[-self.config.stl_window_size:])
        new_mean = np.mean(recent)
        new_std = np.std(recent)

        alpha = self.config.threshold_adaptation_rate
        t["mean"] = t["mean"] * (1 - alpha) + new_mean * alpha
        t["std"] = t["std"] * (1 - alpha) + new_std * alpha

        # 动态调整 sigma
        if new_std > 0:
            cv = new_std / max(abs(new_mean), 1e-10)
            if cv > 0.5:
                t["sigma"] = min(t["sigma"] + 0.1, self.config.threshold_max_sigma)
            elif cv < 0.1:
                t["sigma"] = max(t["sigma"] - 0.05, self.config.threshold_min_sigma)

    def _can_auto_heal(self) -> bool:
        """检查是否可以执行自动自愈"""
        now = time.time()
        # 清理过期记录
        self._heal_counts = [t for t in self._heal_counts if now - t < 3600]
        return len(self._heal_counts) < self.config.max_auto_heals_per_hour

    def _execute_healing(self, action: Optional[HealingAction]) -> None:
        """执行自愈动作 (记录)"""
        if action is None:
            return
        self._heal_counts.append(time.time())
        self._total_heals += 1

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            "total_anomalies_detected": self._total_anomalies,
            "total_auto_heals": self._total_heals,
            "is_trained": self._is_trained,
            "anomaly_type_counts": {
                at.value: sum(1 for r in self._anomaly_history if r.anomaly_type == at)
                for at in AnomalyType
            },
            "recent_anomaly_scores": [r.anomaly_score for r in self._anomaly_history[-10:]],
            "current_thresholds": {
                name: {"mean": round(t["mean"], 4), "std": round(t["std"], 4),
                       "sigma": round(t["sigma"], 2)}
                for name, t in self._thresholds.items()
            }
        }

    def reset(self) -> None:
        """重置"""
        self._feature_buffer = {name: deque(maxlen=self.config.history_length)
                               for name in self.config.feature_names}
        self._isolation_forest = None
        self._anomaly_history = []
        self._is_trained = False
        self._samples_since_train = 0
        self._total_anomalies = 0
        self._total_heals = 0
