"""
时序融合预测器 (TemporalFusionPredictor)

灵感来源:
- 多传感器数据融合 (Multi-Sensor Data Fusion) — 将多个异构传感器的
  时序信号融合为统一的状态估计，广泛应用于自动驾驶和工业控制领域
- Transformer Attention Mechanism — 注意力加权融合机制，对不同信号源
  分配自适应权重，实现信息的选择性聚合
- Bluesky 流式数据采集 (NSLS-II) — 实验物理设施中的流式时序数据采集
  与在线分析框架，支持多通道信号的同步与融合

算法原理:
- Exponential Moving Average (EMA) — 指数移动平均，带自适应遗忘因子
- Attention-Weighted Fusion — 注意力加权融合，基于信号质量动态分配权重
- Signal Normalization — 信号归一化，消除不同量纲的影响
- Adaptive Forgetting Factor — 自适应遗忘因子，根据信号变化率自动调节

功能:
- 融合多个时序信号 (光斑位置、焦距评分、质量指标、电机步数、波前)
- 使用注意力加权融合生成统一的状态预测
- 预测未来光斑位置和质量趋势
- 提供融合状态和注意力权重可视化

依赖: numpy, logging, dataclasses (无 PyTorch/TensorFlow)
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger("SpotZoom.TemporalFusionPredictor")


class FusionSignal(Enum):
    """融合信号类型。"""
    POSITION = "position"       # 光斑位置信号
    FOCUS = "focus"             # 焦距评分信号
    QUALITY = "quality"         # 质量指标信号
    MOTOR = "motor"             # 电机步数信号
    WAVEFRONT = "wavefront"     # 波前误差信号


@dataclass
class FusionState:
    """融合状态快照。"""
    fused_position: Tuple[float, float]       # 融合后的位置 (x, y)
    predicted_trend: Tuple[float, float]      # 预测趋势 (dx, dy)
    confidence: float                         # 融合置信度 [0, 1]
    attention_weights: Dict[str, float]       # 各信号注意力权重
    timestamp: float                          # 时间戳


@dataclass
class _SignalBuffer:
    """单信号缓冲区，存储归一化后的时序数据。"""
    values: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    timestamps: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    ema: float = 0.0               # 指数移动平均值
    ema_var: float = 1.0            # EMA 方差 (用于归一化)
    forgetting_factor: float = 0.1  # 自适应遗忘因子
    is_active: bool = False         # 信号是否活跃
    raw_min: float = 0.0            # 原始值最小值 (用于归一化)
    raw_max: float = 1.0            # 原始值最大值 (用于归一化)


class TemporalFusionPredictor:
    """时序融合预测器。

    将多个异构时序信号通过注意力加权融合为统一的状态估计，
    并预测未来光斑位置和质量趋势。

    融合流程:
    1. 各信号独立归一化 (Z-score)
    2. 计算各信号的 EMA 及其自适应遗忘因子
    3. 基于信号质量和一致性计算注意力权重
    4. 加权融合生成统一状态估计
    5. 基于融合状态进行趋势预测

    Parameters
    ----------
    buffer_size : int
        各信号缓冲区长度。
    base_forgetting_factor : float
        基础遗忘因子 (EMA alpha)。值越大响应越快，平滑越弱。
    min_samples : int
        最少样本数，低于此数不进行融合预测。
    prediction_horizon : int
        默认预测步长。
    """

    def __init__(
        self,
        buffer_size: int = 100,
        base_forgetting_factor: float = 0.1,
        min_samples: int = 5,
        prediction_horizon: int = 5,
    ):
        self.buffer_size = int(buffer_size)
        self.base_forgetting_factor = float(base_forgetting_factor)
        self.min_samples = int(min_samples)
        self.prediction_horizon = int(prediction_horizon)

        # 各信号的缓冲区
        self._buffers: Dict[FusionSignal, _SignalBuffer] = {}
        for signal in FusionSignal:
            self._buffers[signal] = _SignalBuffer(
                values=deque(maxlen=self.buffer_size),
                timestamps=deque(maxlen=self.buffer_size),
            )

        # 注意力权重
        self._attention_weights: Dict[str, float] = {
            s.value: 1.0 / len(FusionSignal) for s in FusionSignal
        }

        # 融合状态
        self._fused_position: Tuple[float, float] = (0.0, 0.0)
        self._fused_velocity: Tuple[float, float] = (0.0, 0.0)
        self._last_timestamp: Optional[float] = None
        self._sample_count: int = 0

        LOGGER.info(
            "TemporalFusionPredictor: 初始化完成 (buffer=%d, alpha=%.3f, "
            "min_samples=%d, horizon=%d)",
            self.buffer_size, self.base_forgetting_factor,
            self.min_samples, self.prediction_horizon,
        )

    def update(self, signal_dict: Dict[str, float], timestamp: Optional[float] = None) -> FusionState:
        """更新融合器，接收多个信号的新采样值。

        Parameters
        ----------
        signal_dict : dict
            信号字典，键为信号名称 (对应 FusionSignal.value)，
            值为浮点数采样值。例如:
            {'position': 12.5, 'focus': 0.85, 'quality': 0.92}
            对于位置信号，传入标量距离 (如到目标距离的范数)。
        timestamp : float or None
            时间戳。为 None 时使用系统当前时间。

        Returns
        -------
        FusionState
            当前融合状态快照。
        """
        if timestamp is None:
            timestamp = time.time()

        # 1. 更新各信号缓冲区
        for signal_name, value in signal_dict.items():
            buf = self._get_buffer_by_name(signal_name)
            if buf is None:
                LOGGER.debug("未知信号类型: %s，忽略", signal_name)
                continue

            buf.is_active = True
            buf.values.append(float(value))
            buf.timestamps.append(float(timestamp))

            # 更新原始值范围 (用于归一化)
            if len(buf.values) == 1:
                buf.raw_min = float(value)
                buf.raw_max = float(value)
            else:
                buf.raw_min = min(buf.raw_min, float(value))
                buf.raw_max = max(buf.raw_max, float(value))

            # 自适应遗忘因子: 信号变化越大，遗忘因子越大 (响应更快)
            if len(buf.values) >= 2:
                recent_change = abs(float(value) - buf.ema)
                buf.forgetting_factor = min(
                    self.base_forgetting_factor + 0.3 * recent_change / max(buf.ema_var, 1e-6),
                    0.5,
                )

            # 更新 EMA
            if not buf.is_active or len(buf.values) == 1:
                buf.ema = float(value)
            else:
                alpha = buf.forgetting_factor
                buf.ema = alpha * float(value) + (1.0 - alpha) * buf.ema

            # 更新 EMA 方差
            diff = float(value) - buf.ema
            buf.ema_var = 0.95 * buf.ema_var + 0.05 * diff * diff
            buf.ema_var = max(buf.ema_var, 1e-12)

        # 2. 计算注意力权重
        self._compute_attention_weights()

        # 3. 执行融合
        self._fuse(timestamp)

        self._last_timestamp = timestamp
        self._sample_count += 1

        return self.get_fusion_state()

    def predict(self, horizon: Optional[int] = None) -> Tuple[float, float]:
        """预测未来光斑位置趋势。

        基于融合速度进行线性外推预测。

        Parameters
        ----------
        horizon : int or None
            预测步长。为 None 时使用默认值。

        Returns
        -------
        Tuple[float, float]
            预测的位置偏移量 (dx, dy)。
        """
        if horizon is None:
            horizon = self.prediction_horizon

        if self._sample_count < self.min_samples:
            return (0.0, 0.0)

        # 基于融合速度的线性外推
        pred_dx = self._fused_velocity[0] * horizon
        pred_dy = self._fused_velocity[1] * horizon

        # 置信度衰减: 预测越远越不确定
        decay = max(0.0, 1.0 - 0.05 * horizon)

        LOGGER.debug(
            "TemporalFusionPredictor: 预测 horizon=%d -> (%.4f, %.4f), decay=%.3f",
            horizon, pred_dx, pred_dy, decay,
        )

        return (pred_dx * decay, pred_dy * decay)

    def get_attention_weights(self) -> Dict[str, float]:
        """获取当前各信号的注意力权重。

        Returns
        -------
        Dict[str, float]
            信号名称到权重的映射。
        """
        return dict(self._attention_weights)

    def get_fusion_state(self) -> FusionState:
        """获取当前融合状态快照。

        Returns
        -------
        FusionState
            融合状态。
        """
        pred_trend = self._fused_velocity
        confidence = self._compute_confidence()

        return FusionState(
            fused_position=self._fused_position,
            predicted_trend=pred_trend,
            confidence=round(confidence, 4),
            attention_weights=dict(self._attention_weights),
            timestamp=self._last_timestamp or 0.0,
        )

    def reset(self) -> None:
        """重置融合器，清除所有缓冲区和状态。"""
        for signal in FusionSignal:
            buf = self._buffers[signal]
            buf.values.clear()
            buf.timestamps.clear()
            buf.ema = 0.0
            buf.ema_var = 1.0
            buf.forgetting_factor = self.base_forgetting_factor
            buf.is_active = False
            buf.raw_min = 0.0
            buf.raw_max = 1.0

        self._attention_weights = {
            s.value: 1.0 / len(FusionSignal) for s in FusionSignal
        }
        self._fused_position = (0.0, 0.0)
        self._fused_velocity = (0.0, 0.0)
        self._last_timestamp = None
        self._sample_count = 0

        LOGGER.info("TemporalFusionPredictor: 融合器已重置")

    # ======================== 内部方法 ========================

    def _get_buffer_by_name(self, name: str) -> Optional[_SignalBuffer]:
        """根据信号名称获取缓冲区。"""
        for signal in FusionSignal:
            if signal.value == name:
                return self._buffers[signal]
        return None

    def _normalize(self, value: float, buf: _SignalBuffer) -> float:
        """将原始值归一化到 [0, 1] 范围。

        使用 min-max 归一化。如果范围为零，返回 0.5。

        Parameters
        ----------
        value : float
            原始值。
        buf : _SignalBuffer
            信号缓冲区 (含 raw_min, raw_max)。

        Returns
        -------
        float
            归一化值 [0, 1]。
        """
        range_val = buf.raw_max - buf.raw_min
        if range_val < 1e-12:
            return 0.5
        return (value - buf.raw_min) / range_val

    def _compute_attention_weights(self) -> None:
        """计算各活跃信号的注意力权重。

        权重基于:
        - 信号一致性 (EMA 方差越小，权重越高)
        - 数据充分性 (样本越多，权重越高)
        - 信号新鲜度 (最近更新时间越近，权重越高)
        """
        active_signals = [
            s for s in FusionSignal if self._buffers[s].is_active
        ]

        if not active_signals:
            return

        scores: Dict[str, float] = {}
        now = self._last_timestamp or time.time()

        for signal in active_signals:
            buf = self._buffers[signal]
            name = signal.value

            # 一致性分数: 方差越小越好
            consistency = 1.0 / (1.0 + buf.ema_var)

            # 数据充分性分数
            data_score = min(len(buf.values) / max(self.min_samples, 1), 1.0)

            # 新鲜度分数
            if buf.timestamps:
                age = now - buf.timestamps[-1]
                freshness = max(0.0, 1.0 - age / 5.0)  # 5秒内衰减
            else:
                freshness = 0.0

            scores[name] = consistency * data_score * freshness

        # Softmax 归一化
        total = sum(scores.values())
        if total > 1e-12:
            for name in scores:
                self._attention_weights[name] = round(scores[name] / total, 4)
        else:
            # 均匀分配
            n = len(active_signals)
            for signal in active_signals:
                self._attention_weights[signal.value] = round(1.0 / n, 4)

        # 非活跃信号权重置零
        for signal in FusionSignal:
            if not self._buffers[signal].is_active:
                self._attention_weights[signal.value] = 0.0

    def _fuse(self, timestamp: float) -> None:
        """执行注意力加权融合。

        将各信号的 EMA 值加权融合，生成统一的位置和速度估计。
        """
        # 融合位置: 使用各信号归一化 EMA 的加权和
        weighted_sum = 0.0
        total_weight = 0.0

        for signal in FusionSignal:
            buf = self._buffers[signal]
            if not buf.is_active or len(buf.values) < 2:
                continue

            w = self._attention_weights.get(signal.value, 0.0)
            if w < 1e-12:
                continue

            norm_val = self._normalize(buf.ema, buf)
            weighted_sum += w * norm_val
            total_weight += w

        if total_weight > 1e-12:
            fused_val = weighted_sum / total_weight
        else:
            fused_val = 0.0

        # 将融合值映射到位置空间 (简化: 使用位置信号的原始值)
        pos_buf = self._buffers[FusionSignal.POSITION]
        if pos_buf.is_active and len(pos_buf.values) >= 2:
            # 位置信号直接使用其 EMA 作为位置估计
            self._fused_position = (pos_buf.ema, pos_buf.ema)
            # 速度估计: 最近两个 EMA 的差分
            if len(pos_buf.values) >= 2:
                recent = list(pos_buf.values)
                vel = recent[-1] - recent[-2]
                self._fused_velocity = (vel, vel)
        else:
            # 无位置信号时，使用融合值作为位置代理
            old_pos = self._fused_position
            self._fused_position = (fused_val, fused_val)
            self._fused_velocity = (
                self._fused_position[0] - old_pos[0],
                self._fused_position[1] - old_pos[1],
            )

    def _compute_confidence(self) -> float:
        """计算融合置信度。

        基于活跃信号数量、数据充分性和权重集中度。

        Returns
        -------
        float
            置信度 [0, 1]。
        """
        active_count = sum(
            1 for s in FusionSignal if self._buffers[s].is_active
        )

        if active_count == 0:
            return 0.0

        # 活跃信号比例
        signal_ratio = active_count / len(FusionSignal)

        # 数据充分性
        min_buf_len = min(
            len(self._buffers[s].values)
            for s in FusionSignal if self._buffers[s].is_active
        )
        data_ratio = min(min_buf_len / max(self.min_samples, 1), 1.0)

        # 权重集中度 (熵的逆): 权重越集中，置信度越高
        weights = np.array([
            self._attention_weights[s.value]
            for s in FusionSignal
        ])
        weights = weights[weights > 1e-12]
        if len(weights) > 0:
            weights_norm = weights / weights.sum()
            entropy = -np.sum(weights_norm * np.log(weights_norm + 1e-12))
            max_entropy = np.log(len(weights))
            concentration = 1.0 - (entropy / max(max_entropy, 1e-12))
        else:
            concentration = 0.0

        confidence = signal_ratio * 0.3 + data_ratio * 0.4 + concentration * 0.3
        return round(float(np.clip(confidence, 0.0, 1.0)), 4)

    @property
    def sample_count(self) -> int:
        """已处理的样本总数。"""
        return self._sample_count


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    predictor = TemporalFusionPredictor(
        buffer_size=50,
        base_forgetting_factor=0.1,
        min_samples=3,
        prediction_horizon=5,
    )

    # 模拟多传感器数据流
    np.random.seed(42)
    print("=== 时序融合预测器测试 ===\n")

    for i in range(30):
        t = i * 0.1
        signal_dict = {
            "position": 10.0 + 0.5 * np.sin(t) + np.random.normal(0, 0.1),
            "focus": 0.8 + 0.1 * np.cos(t * 2) + np.random.normal(0, 0.02),
            "quality": 0.9 + 0.05 * np.sin(t * 1.5) + np.random.normal(0, 0.01),
        }
        state = predictor.update(signal_dict, timestamp=t)

        if i % 10 == 9:
            pred = predictor.predict(horizon=3)
            print(f"[t={t:.1f}s] 融合位置: ({state.fused_position[0]:.4f}, "
                  f"{state.fused_position[1]:.4f})")
            print(f"  预测趋势: ({state.predicted_trend[0]:.4f}, "
                  f"{state.predicted_trend[1]:.4f})")
            print(f"  置信度: {state.confidence:.4f}")
            print(f"  注意力权重: {state.attention_weights}")
            print(f"  3步预测: ({pred[0]:.4f}, {pred[1]:.4f})")
            print()

    # 测试重置
    predictor.reset()
    print("重置后样本数:", predictor.sample_count)
    print("测试完成")
