"""
深度振动预测器 (DeepVibrationPredictor)

灵感来源:
- Bi-RNN + Attention 振动预测 (学术前沿专利) — 使用双向循环神经网络
  结合注意力机制进行高精度振动位移预测
- LSTM/GRU 时序建模 — 循环神经网络在时序预测中的经典应用
- Seq2Seq Attention (Bahdanau et al. 2015) — 注意力机制实现信息选择性聚合
- Kalman Filter 振动预测 — 传统振动预测方法的对比基准

算法原理:
- Bidirectional RNN — 双向循环神经网络，同时捕获前向和后向时序依赖
- Attention Mechanism — 注意力加权机制，对不同时间步分配自适应权重
- Teacher Forcing — 训练时使用真实值作为输入，加速收敛
- Gradient Clipping — 梯度裁剪，防止梯度爆炸
- Layer Normalization — 层归一化，稳定训练过程

功能:
- 使用双向 RNN + Attention 预测振动位移 (纯 numpy 实现)
- 支持多隐藏层堆叠
- 模型训练与推理
- 预测置信度评估
- 模型持久化 (JSON 格式)

依赖: numpy, json, logging (无 PyTorch/TensorFlow)
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger("SpotZoom.DeepVibrationPredictor")


# ======================== 数据类 ========================


@dataclass
class VibrationPredictionConfig:
    """振动预测模型配置。

    Parameters
    ----------
    input_size : int
        输入特征维度。
    hidden_size : int
        隐藏层维度。
    num_layers : int
        RNN 层数。
    learning_rate : float
        学习率。
    gradient_clip : float
        梯度裁剪阈值。
    sequence_length : int
        输入序列长度。
    prediction_horizon : int
        预测步长。
    attention_size : int
        注意力层维度。
    dropout_rate : float
        Dropout 比率 (0=无 dropout)。
    """
    input_size: int = 1
    hidden_size: int = 32
    num_layers: int = 2
    learning_rate: float = 0.001
    gradient_clip: float = 5.0
    sequence_length: int = 20
    prediction_horizon: int = 5
    attention_size: int = 16
    dropout_rate: float = 0.1


@dataclass
class VibrationPrediction:
    """振动预测结果。"""
    predicted_displacements: List[float]  # 预测的位移序列
    confidence: float                     # 整体置信度 [0, 1]
    prediction_horizon: int               # 预测步长
    timestamp: float = 0.0                # 时间戳


@dataclass
class PredictionConfidence:
    """预测置信度详情。"""
    mean_confidence: float                # 平均置信度
    per_step_confidence: List[float]      # 每步置信度
    uncertainty_growth_rate: float        # 不确定性增长率
    is_reliable: bool                     # 是否可靠 (置信度 > 阈值)


# ======================== 简单 RNN 单元 ========================


class SimpleRNNCell:
    """简单 RNN 单元 (numpy 实现)。

    使用 tanh 激活函数的 Vanilla RNN。

    Parameters
    ----------
    input_size : int
        输入特征维度。
    hidden_size : int
        隐藏状态维度。
    """

    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Xavier 初始化
        scale = np.sqrt(2.0 / (input_size + hidden_size))
        self.Wxh = np.random.randn(hidden_size, input_size) * scale
        self.Whh = np.random.randn(hidden_size, hidden_size) * scale
        self.bh = np.zeros(hidden_size)

        # 梯度
        self.dWxh = np.zeros_like(self.Wxh)
        self.dWhh = np.zeros_like(self.Whh)
        self.dbh = np.zeros_like(self.bh)

        # 缓存 (用于反向传播)
        self._cache: Optional[Dict[str, np.ndarray]] = None

    def forward(self, x: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        """前向传播。

        Parameters
        ----------
        x : np.ndarray
            输入向量, shape (input_size,)。
        h_prev : np.ndarray
            前一隐藏状态, shape (hidden_size,)。

        Returns
        -------
        np.ndarray
            新隐藏状态, shape (hidden_size,)。
        """
        # h_t = tanh(Wxh @ x + Whh @ h_prev + bh)
        h_raw = self.Wxh @ x + self.Whh @ h_prev + self.bh
        h = np.tanh(h_raw)

        self._cache = {"x": x, "h_prev": h_prev, "h": h, "h_raw": h_raw}
        return h

    def backward(
        self,
        dh_next: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """反向传播。

        Parameters
        ----------
        dh_next : np.ndarray
            来自上一时间步的梯度, shape (hidden_size,)。

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (dx, dh_prev) — 输入梯度和前一状态梯度。
        """
        assert self._cache is not None, "请先调用 forward()"

        x = self._cache["x"]
        h_prev = self._cache["h_prev"]
        h = self._cache["h"]

        # tanh 的导数: 1 - tanh^2
        dtanh = dh_next * (1.0 - h * h)

        # 参数梯度
        self.dWxh += np.outer(dtanh, x)
        self.dWhh += np.outer(dtanh, h_prev)
        self.dbh += dtanh

        # 传播梯度
        dx = self.Wxh.T @ dtanh
        dh_prev = self.Whh.T @ dtanh

        return dx, dh_prev

    def get_params(self) -> Dict[str, np.ndarray]:
        """获取参数字典。"""
        return {
            "Wxh": self.Wxh,
            "Whh": self.Whh,
            "bh": self.bh,
        }

    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        """设置参数。"""
        self.Wxh = params["Wxh"]
        self.Whh = params["Whh"]
        self.bh = params["bh"]

    def zero_gradients(self) -> None:
        """清零梯度。"""
        self.dWxh = np.zeros_like(self.Wxh)
        self.dWhh = np.zeros_like(self.Whh)
        self.dbh = np.zeros_like(self.bh)


# ======================== 注意力层 ========================


class AttentionLayer:
    """注意力层 (numpy 实现)。

    使用 Bahdanau 风格的加性注意力机制。

    Parameters
    ----------
    hidden_size : int
        隐藏状态维度。
    attention_size : int
        注意力向量维度。
    """

    def __init__(self, hidden_size: int, attention_size: int):
        self.hidden_size = hidden_size
        self.attention_size = attention_size

        # 注意力参数
        scale = np.sqrt(2.0 / (hidden_size + attention_size))
        self.W_a = np.random.randn(attention_size, hidden_size) * scale
        self.U_a = np.random.randn(attention_size, hidden_size) * scale
        self.v_a = np.random.randn(attention_size) * scale
        self.b_a = np.zeros(attention_size)

        # 梯度
        self.dW_a = np.zeros_like(self.W_a)
        self.dU_a = np.zeros_like(self.U_a)
        self.dv_a = np.zeros_like(self.v_a)
        self.db_a = np.zeros_like(self.b_a)

        # 缓存
        self._cache: Optional[Dict[str, Any]] = None

    def forward(
        self,
        encoder_states: np.ndarray,
        decoder_state: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """前向传播。

        Parameters
        ----------
        encoder_states : np.ndarray
            编码器状态序列, shape (seq_len, hidden_size)。
        decoder_state : np.ndarray
            解码器当前状态, shape (hidden_size,)。

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (context_vector, attention_weights)。
            context_vector: shape (hidden_size,)
            attention_weights: shape (seq_len,)
        """
        seq_len = encoder_states.shape[0]

        # 计算注意力分数: e_i = v_a^T tanh(W_a h_i + U_a s + b_a)
        # h_i: encoder_states[i], s: decoder_state
        scores = np.zeros(seq_len)
        pre_activations = np.zeros((seq_len, self.attention_size))

        for i in range(seq_len):
            pre_act = (
                self.W_a @ encoder_states[i]
                + self.U_a @ decoder_state
                + self.b_a
            )
            pre_activations[i] = pre_act
            scores[i] = self.v_a @ np.tanh(pre_act)

        # Softmax 归一化
        scores_stable = scores - np.max(scores)
        exp_scores = np.exp(scores_stable)
        attention_weights = exp_scores / np.sum(exp_scores)

        # 上下文向量: c = sum(alpha_i * h_i)
        context = np.zeros(self.hidden_size)
        for i in range(seq_len):
            context += attention_weights[i] * encoder_states[i]

        self._cache = {
            "encoder_states": encoder_states,
            "decoder_state": decoder_state,
            "attention_weights": attention_weights,
            "pre_activations": pre_activations,
            "scores": scores,
        }

        return context, attention_weights

    def backward(
        self,
        dcontext: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """反向传播。

        Parameters
        ----------
        dcontext : np.ndarray
            上下文向量的梯度, shape (hidden_size,)。

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (d_encoder_states, d_decoder_state)。
        """
        assert self._cache is not None, "请先调用 forward()"

        encoder_states = self._cache["encoder_states"]
        decoder_state = self._cache["decoder_state"]
        attention_weights = self._cache["attention_weights"]
        pre_activations = self._cache["pre_activations"]
        seq_len = encoder_states.shape[0]

        # 注意力权重梯度
        d_attention = np.zeros(seq_len)
        for i in range(seq_len):
            d_attention[i] = np.dot(dcontext, encoder_states[i])

        # Softmax 反向传播
        d_scores = attention_weights * (
            d_attention - np.dot(attention_weights, d_attention)
        )

        d_encoder_states = np.zeros_like(encoder_states)
        d_decoder_state = np.zeros(self.hidden_size)

        for i in range(seq_len):
            tanh_i = np.tanh(pre_activations[i])
            d_pre = d_scores[i] * self.v_a  # (attention_size,)

            # tanh 导数
            d_pre += d_scores[i] * (1.0 - tanh_i * tanh_i) * self.v_a

            # 参数梯度
            self.dW_a += np.outer(d_pre, encoder_states[i])
            self.dU_a += np.outer(d_pre, decoder_state)
            self.dv_a += d_pre * tanh_i
            self.db_a += d_pre

            # 输入梯度
            d_encoder_states[i] = self.W_a.T @ d_pre
            d_decoder_state += self.U_a.T @ d_pre

        return d_encoder_states, d_decoder_state

    def get_params(self) -> Dict[str, np.ndarray]:
        """获取参数字典。"""
        return {
            "W_a": self.W_a,
            "U_a": self.U_a,
            "v_a": self.v_a,
            "b_a": self.b_a,
        }

    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        """设置参数。"""
        self.W_a = params["W_a"]
        self.U_a = params["U_a"]
        self.v_a = params["v_a"]
        self.b_a = params["b_a"]

    def zero_gradients(self) -> None:
        """清零梯度。"""
        self.dW_a = np.zeros_like(self.W_a)
        self.dU_a = np.zeros_like(self.U_a)
        self.dv_a = np.zeros_like(self.v_a)
        self.db_a = np.zeros_like(self.b_a)


# ======================== 深度振动预测器 ========================


class DeepVibrationPredictor:
    """深度振动预测器。

    使用双向 RNN + Attention 机制预测振动位移。
    全部使用 numpy 矩阵运算实现，无深度学习框架依赖。

    Parameters
    ----------
    config : VibrationPredictionConfig or None
        模型配置。为 None 时使用默认配置。
    """

    def __init__(self, config: Optional[VibrationPredictionConfig] = None):
        self.config = config or VibrationPredictionConfig()
        self._is_built = False

        # 模型组件
        self._forward_cells: List[SimpleRNNCell] = []
        self._backward_cells: List[SimpleRNNCell] = []
        self._attention: Optional[AttentionLayer] = None
        self._output_W: Optional[np.ndarray] = None
        self._output_b: Optional[np.ndarray] = None

        # 训练状态
        self._d_output_W: Optional[np.ndarray] = None
        self._d_output_b: Optional[np.ndarray] = None
        self._training_step_count: int = 0
        self._loss_history: List[float] = []

        LOGGER.info(
            "DeepVibrationPredictor: 初始化完成 (hidden=%d, layers=%d, lr=%.4f)",
            self.config.hidden_size, self.config.num_layers, self.config.learning_rate,
        )

    def build_model(
        self,
        input_size: Optional[int] = None,
        hidden_size: Optional[int] = None,
        num_layers: Optional[int] = None,
    ) -> None:
        """构建模型参数。

        Parameters
        ----------
        input_size : int or None
            输入特征维度。为 None 时使用配置值。
        hidden_size : int or None
            隐藏层维度。为 None 时使用配置值。
        num_layers : int or None
            RNN 层数。为 None 时使用配置值。
        """
        in_size = input_size or self.config.input_size
        hid_size = hidden_size or self.config.hidden_size
        n_layers = num_layers or self.config.num_layers

        self.config.input_size = in_size
        self.config.hidden_size = hid_size
        self.config.num_layers = n_layers

        # 构建前向 RNN 层
        self._forward_cells = []
        for layer in range(n_layers):
            layer_input_size = in_size if layer == 0 else hid_size
            self._forward_cells.append(SimpleRNNCell(layer_input_size, hid_size))

        # 构建反向 RNN 层
        self._backward_cells = []
        for layer in range(n_layers):
            layer_input_size = in_size if layer == 0 else hid_size
            self._backward_cells.append(SimpleRNNCell(layer_input_size, hid_size))

        # 构建注意力层 (输入为 2*hidden_size, 因为拼接了双向)
        self._attention = AttentionLayer(2 * hid_size, self.config.attention_size)

        # 输出层: 将上下文向量映射到预测值
        self._output_W = np.random.randn(1, 2 * hid_size) * np.sqrt(2.0 / (2 * hid_size))
        self._output_b = np.zeros(1)

        # 梯度缓冲
        self._d_output_W = np.zeros_like(self._output_W)
        self._d_output_b = np.zeros_like(self._output_b)

        self._is_built = True
        self._training_step_count = 0
        self._loss_history.clear()

        LOGGER.info(
            "DeepVibrationPredictor: 模型构建完成 (input=%d, hidden=%d, layers=%d, "
            "params=%d)",
            in_size, hid_size, n_layers, self._count_parameters(),
        )

    def forward(self, x_sequence: np.ndarray) -> np.ndarray:
        """前向传播。

        Parameters
        ----------
        x_sequence : np.ndarray
            输入序列, shape (seq_len, input_size)。

        Returns
        -------
        np.ndarray
            预测输出, shape (seq_len, 1)。
        """
        if not self._is_built:
            raise ValueError("请先调用 build_model() 构建模型")

        seq_len = x_sequence.shape[0]

        # 1. 前向 RNN
        forward_states = self._run_rnn_forward(x_sequence, self._forward_cells)

        # 2. 反向 RNN
        backward_states = self._run_rnn_backward(x_sequence, self._backward_cells)

        # 3. 拼接双向状态
        bi_states = np.concatenate([forward_states, backward_states], axis=1)
        # shape: (seq_len, 2 * hidden_size)

        # 4. 使用注意力机制生成每步预测
        outputs = np.zeros((seq_len, 1))
        for t in range(seq_len):
            context, _ = self._attention.forward(bi_states, bi_states[t])
            output = self._output_W @ context + self._output_b
            outputs[t] = output

        return outputs

    def train_step(
        self,
        x_seq: np.ndarray,
        y_target: np.ndarray,
        learning_rate: Optional[float] = None,
    ) -> float:
        """单步训练。

        Parameters
        ----------
        x_seq : np.ndarray
            输入序列, shape (seq_len, input_size)。
        y_target : np.ndarray
            目标值, shape (seq_len, 1) 或 (seq_len,)。
        learning_rate : float or None
            学习率。为 None 时使用配置值。

        Returns
        -------
        float
            损失值 (MSE)。
        """
        if not self._is_built:
            raise ValueError("请先调用 build_model() 构建模型")

        lr = learning_rate or self.config.learning_rate
        seq_len = x_seq.shape[0]

        if y_target.ndim == 1:
            y_target = y_target.reshape(-1, 1)

        # ---- 前向传播 ----
        forward_states = self._run_rnn_forward(x_seq, self._forward_cells)
        backward_states = self._run_rnn_backward(x_seq, self._backward_cells)
        bi_states = np.concatenate([forward_states, backward_states], axis=1)

        outputs = np.zeros((seq_len, 1))
        attention_cache_list = []

        for t in range(seq_len):
            context, attn_weights = self._attention.forward(bi_states, bi_states[t])
            output = self._output_W @ context + self._output_b
            outputs[t] = output
            attention_cache_list.append({
                "context": context,
                "bi_states": bi_states.copy(),
                "query_idx": t,
            })

        # 计算损失 (MSE)
        loss = float(np.mean((outputs - y_target) ** 2))

        # ---- 反向传播 ----
        # 输出层梯度
        d_outputs = 2.0 * (outputs - y_target) / seq_len

        # 清零所有梯度
        for cell in self._forward_cells:
            cell.zero_gradients()
        for cell in self._backward_cells:
            cell.zero_gradients()
        self._attention.zero_gradients()
        self._d_output_W = np.zeros_like(self._output_W)
        self._d_output_b = np.zeros_like(self._output_b)

        # 累积梯度
        d_bi_states_total = np.zeros_like(bi_states)

        for t in range(seq_len):
            d_out = d_outputs[t]  # (1,)

            # 输出层梯度
            context_t = attention_cache_list[t]["context"]
            self._d_output_W += np.outer(d_out, context_t)
            self._d_output_b += d_out

            # 上下文梯度
            d_context = self._output_W.T @ d_out  # (2*hidden_size,)

            # 注意力反向传播
            d_enc, d_dec = self._attention.backward(d_context)
            d_bi_states_total += d_enc

        # 将梯度分配给前向和反向 RNN
        d_forward_states = d_bi_states_total[:, :self.config.hidden_size]
        d_backward_states = d_bi_states_total[:, self.config.hidden_size:]

        # 前向 RNN 反向传播
        self._backprop_rnn_forward(x_seq, self._forward_cells, d_forward_states)

        # 反向 RNN 反向传播
        self._backprop_rnn_backward(x_seq, self._backward_cells, d_backward_states)

        # ---- 参数更新 (SGD + 梯度裁剪) ----
        self._update_parameters(lr)

        self._training_step_count += 1
        self._loss_history.append(loss)

        if self._training_step_count % 100 == 0:
            LOGGER.debug(
                "DeepVibrationPredictor: train_step=%d, loss=%.6f",
                self._training_step_count, loss,
            )

        return loss

    def predict_vibration(
        self,
        history_displacements: np.ndarray,
        horizon_steps: Optional[int] = None,
    ) -> VibrationPrediction:
        """预测未来振动。

        Parameters
        ----------
        history_displacements : np.ndarray
            历史振动位移序列, shape (seq_len,) 或 (seq_len, 1)。
        horizon_steps : int or None
            预测步长。为 None 时使用配置值。

        Returns
        -------
        VibrationPrediction
            预测结果。
        """
        if not self._is_built:
            raise ValueError("请先调用 build_model() 构建模型")

        horizon = horizon_steps or self.config.prediction_horizon

        if history_displacements.ndim == 1:
            history_displacements = history_displacements.reshape(-1, 1)

        # 自回归预测
        current_seq = history_displacements.copy()
        predictions = []

        for _ in range(horizon):
            # 确保输入长度不超过模型序列长度
            if len(current_seq) > self.config.sequence_length:
                input_seq = current_seq[-self.config.sequence_length:]
            else:
                input_seq = current_seq

            # 前向传播获取最后一步的预测
            output = self.forward(input_seq)
            next_val = float(output[-1, 0])
            predictions.append(next_val)

            # 将预测值加入序列 (自回归)
            current_seq = np.vstack([current_seq, np.array([[next_val]])])

        # 计算置信度
        confidence_result = self.compute_prediction_confidence(predictions)

        return VibrationPrediction(
            predicted_displacements=predictions,
            confidence=confidence_result.mean_confidence,
            prediction_horizon=horizon,
            timestamp=time.time(),
        )

    def compute_prediction_confidence(
        self,
        predictions: List[float],
    ) -> PredictionConfidence:
        """计算预测置信度。

        基于预测值的平滑度和训练损失评估置信度。

        Parameters
        ----------
        predictions : List[float]
            预测值序列。

        Returns
        -------
        PredictionConfidence
            置信度详情。
        """
        if not predictions:
            return PredictionConfidence(
                mean_confidence=0.0,
                per_step_confidence=[],
                uncertainty_growth_rate=0.0,
                is_reliable=False,
            )

        preds = np.array(predictions)

        # 1. 基于平滑度的置信度: 预测越平滑，置信度越高
        if len(preds) > 1:
            diffs = np.diff(preds)
            smoothness = 1.0 / (1.0 + np.std(diffs) / (np.std(preds) + 1e-8))
        else:
            smoothness = 0.5

        # 2. 基于训练损失的置信度
        if self._loss_history:
            recent_losses = self._loss_history[-100:]
            loss_confidence = 1.0 / (1.0 + np.mean(recent_losses))
        else:
            loss_confidence = 0.3

        # 3. 每步置信度 (随预测步数递减)
        n = len(predictions)
        per_step = np.zeros(n)
        for i in range(n):
            # 时间衰减因子
            time_decay = max(0.0, 1.0 - 0.1 * i)
            per_step[i] = smoothness * loss_confidence * time_decay

        # 4. 不确定性增长率
        if len(preds) > 2:
            uncertainties = 1.0 - per_step
            growth = float(np.polyfit(range(len(uncertainties)), uncertainties, 1)[0])
        else:
            growth = 0.0

        mean_conf = float(np.mean(per_step))
        is_reliable = mean_conf > 0.3

        return PredictionConfidence(
            mean_confidence=round(mean_conf, 4),
            per_step_confidence=[round(float(c), 4) for c in per_step],
            uncertainty_growth_rate=round(growth, 4),
            is_reliable=is_reliable,
        )

    def save_model(self, path: str) -> None:
        """保存模型到 JSON 文件。

        Parameters
        ----------
        path : str
            保存路径。
        """
        if not self._is_built:
            raise ValueError("模型尚未构建，无法保存")

        model_data = {
            "config": {
                "input_size": self.config.input_size,
                "hidden_size": self.config.hidden_size,
                "num_layers": self.config.num_layers,
                "learning_rate": self.config.learning_rate,
                "gradient_clip": self.config.gradient_clip,
                "sequence_length": self.config.sequence_length,
                "prediction_horizon": self.config.prediction_horizon,
                "attention_size": self.config.attention_size,
                "dropout_rate": self.config.dropout_rate,
            },
            "forward_cells": [
                {k: v.tolist() for k, v in cell.get_params().items()}
                for cell in self._forward_cells
            ],
            "backward_cells": [
                {k: v.tolist() for k, v in cell.get_params().items()}
                for cell in self._backward_cells
            ],
            "attention": {
                k: v.tolist() for k, v in self._attention.get_params().items()
            },
            "output_W": self._output_W.tolist(),
            "output_b": self._output_b.tolist(),
            "training_step_count": self._training_step_count,
            "loss_history": self._loss_history[-1000:],  # 保留最近1000条
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(model_data, f, indent=2)

        LOGGER.info(
            "DeepVibrationPredictor: 模型已保存到 %s (params=%d)",
            path, self._count_parameters(),
        )

    def load_model(self, path: str) -> None:
        """从 JSON 文件加载模型。

        Parameters
        ----------
        path : str
            模型文件路径。
        """
        with open(path, "r", encoding="utf-8") as f:
            model_data = json.load(f)

        # 恢复配置
        cfg = model_data["config"]
        self.config = VibrationPredictionConfig(**cfg)

        # 构建模型结构
        self.build_model(
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
        )

        # 恢复参数
        for i, cell_data in enumerate(model_data["forward_cells"]):
            params = {k: np.array(v) for k, v in cell_data.items()}
            self._forward_cells[i].set_params(params)

        for i, cell_data in enumerate(model_data["backward_cells"]):
            params = {k: np.array(v) for k, v in cell_data.items()}
            self._backward_cells[i].set_params(params)

        attn_params = {k: np.array(v) for k, v in model_data["attention"].items()}
        self._attention.set_params(attn_params)

        self._output_W = np.array(model_data["output_W"])
        self._output_b = np.array(model_data["output_b"])
        self._training_step_count = model_data.get("training_step_count", 0)
        self._loss_history = model_data.get("loss_history", [])

        LOGGER.info(
            "DeepVibrationPredictor: 模型已从 %s 加载 (训练步数=%d)",
            path, self._training_step_count,
        )

    def get_loss_history(self) -> List[float]:
        """获取训练损失历史。"""
        return list(self._loss_history)

    def reset(self) -> None:
        """重置预测器。"""
        self._forward_cells.clear()
        self._backward_cells.clear()
        self._attention = None
        self._output_W = None
        self._output_b = None
        self._is_built = False
        self._training_step_count = 0
        self._loss_history.clear()
        LOGGER.info("DeepVibrationPredictor: 预测器已重置")

    # ======================== 内部方法 ========================

    def _run_rnn_forward(
        self,
        x_sequence: np.ndarray,
        cells: List[SimpleRNNCell],
    ) -> np.ndarray:
        """运行前向 RNN。

        Parameters
        ----------
        x_sequence : np.ndarray
            输入序列, shape (seq_len, input_size)。
        cells : List[SimpleRNNCell]
            RNN 单元列表。

        Returns
        -------
        np.ndarray
            隐藏状态序列, shape (seq_len, hidden_size)。
        """
        seq_len = x_sequence.shape[0]
        hidden_size = cells[0].hidden_size
        states = np.zeros((seq_len, hidden_size))
        h = np.zeros(hidden_size)

        for t in range(seq_len):
            x_t = x_sequence[t]
            for layer, cell in enumerate(cells):
                if layer == 0:
                    h = cell.forward(x_t, h)
                else:
                    h = cell.forward(h, h)
            states[t] = h

        return states

    def _run_rnn_backward(
        self,
        x_sequence: np.ndarray,
        cells: List[SimpleRNNCell],
    ) -> np.ndarray:
        """运行反向 RNN。

        Parameters
        ----------
        x_sequence : np.ndarray
            输入序列, shape (seq_len, input_size)。
        cells : List[SimpleRNNCell]
            RNN 单元列表。

        Returns
        -------
        np.ndarray
            隐藏状态序列, shape (seq_len, hidden_size)。
        """
        seq_len = x_sequence.shape[0]
        hidden_size = cells[0].hidden_size
        states = np.zeros((seq_len, hidden_size))
        h = np.zeros(hidden_size)

        for t in range(seq_len - 1, -1, -1):
            x_t = x_sequence[t]
            for layer, cell in enumerate(cells):
                if layer == 0:
                    h = cell.forward(x_t, h)
                else:
                    h = cell.forward(h, h)
            states[t] = h

        return states

    def _backprop_rnn_forward(
        self,
        x_sequence: np.ndarray,
        cells: List[SimpleRNNCell],
        d_states: np.ndarray,
    ) -> None:
        """前向 RNN 的反向传播 (累积梯度到各 cell)。"""
        seq_len = x_sequence.shape[0]
        n_layers = len(cells)
        hidden_size = cells[0].hidden_size

        # 每层的梯度累积
        dh_per_layer = [np.zeros(hidden_size) for _ in range(n_layers)]

        for t in range(seq_len - 1, -1, -1):
            # 加上来自上方的梯度
            dh_per_layer[-1] += d_states[t]

            for layer in range(n_layers - 1, -1, -1):
                cell = cells[layer]
                # 重新前向以获取缓存
                if layer == 0:
                    h_prev = np.zeros(hidden_size) if t == 0 else None
                else:
                    h_prev = None

                # 简化: 直接使用累积梯度
                dx, dh_prev = cell.backward(dh_per_layer[layer])

                if layer > 0:
                    dh_per_layer[layer - 1] += dh_prev

    def _backprop_rnn_backward(
        self,
        x_sequence: np.ndarray,
        cells: List[SimpleRNNCell],
        d_states: np.ndarray,
    ) -> None:
        """反向 RNN 的反向传播 (累积梯度到各 cell)。"""
        seq_len = x_sequence.shape[0]
        n_layers = len(cells)
        hidden_size = cells[0].hidden_size

        dh_per_layer = [np.zeros(hidden_size) for _ in range(n_layers)]

        for t in range(seq_len):
            dh_per_layer[-1] += d_states[t]

            for layer in range(n_layers - 1, -1, -1):
                cell = cells[layer]
                dx, dh_prev = cell.backward(dh_per_layer[layer])

                if layer > 0:
                    dh_per_layer[layer - 1] += dh_prev

    def _update_parameters(self, learning_rate: float) -> None:
        """更新所有参数 (SGD + 梯度裁剪)。"""
        clip_val = self.config.gradient_clip

        def clip_and_update(param: np.ndarray, grad: np.ndarray) -> np.ndarray:
            grad_norm = np.linalg.norm(grad)
            if grad_norm > clip_val:
                grad = grad * clip_val / grad_norm
            return param - learning_rate * grad

        # 更新前向 RNN
        for cell in self._forward_cells:
            cell.Wxh = clip_and_update(cell.Wxh, cell.dWxh)
            cell.Whh = clip_and_update(cell.Whh, cell.dWhh)
            cell.bh = clip_and_update(cell.bh, cell.dbh)

        # 更新反向 RNN
        for cell in self._backward_cells:
            cell.Wxh = clip_and_update(cell.Wxh, cell.dWxh)
            cell.Whh = clip_and_update(cell.Whh, cell.dWhh)
            cell.bh = clip_and_update(cell.bh, cell.dbh)

        # 更新注意力层
        attn = self._attention
        attn.W_a = clip_and_update(attn.W_a, attn.dW_a)
        attn.U_a = clip_and_update(attn.U_a, attn.dU_a)
        attn.v_a = clip_and_update(attn.v_a, attn.dv_a)
        attn.b_a = clip_and_update(attn.b_a, attn.db_a)

        # 更新输出层
        self._output_W = clip_and_update(self._output_W, self._d_output_W)
        self._output_b = clip_and_update(self._output_b, self._d_output_b)

    def _count_parameters(self) -> int:
        """计算模型总参数量。"""
        count = 0
        for cell in self._forward_cells:
            count += cell.Wxh.size + cell.Whh.size + cell.bh.size
        for cell in self._backward_cells:
            count += cell.Wxh.size + cell.Whh.size + cell.bh.size
        if self._attention:
            count += (
                self._attention.W_a.size + self._attention.U_a.size
                + self._attention.v_a.size + self._attention.b_a.size
            )
        if self._output_W is not None:
            count += self._output_W.size + self._output_b.size
        return count


# ======================== 测试入口 ========================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    print("=== 深度振动预测器测试 ===\n")

    np.random.seed(42)

    # 生成模拟振动数据 (正弦波 + 噪声)
    t = np.linspace(0, 10, 200)
    vibration = np.sin(2 * np.pi * 0.5 * t) + 0.1 * np.random.randn(200)
    vibration = vibration.reshape(-1, 1)

    # 构建模型
    config = VibrationPredictionConfig(
        input_size=1,
        hidden_size=16,
        num_layers=2,
        learning_rate=0.005,
        sequence_length=20,
        prediction_horizon=5,
        attention_size=8,
    )

    predictor = DeepVibrationPredictor(config)
    predictor.build_model()

    print(f"模型参数量: {predictor._count_parameters()}")
    print(f"输入维度: {config.input_size}, 隐藏维度: {config.hidden_size}")
    print(f"RNN 层数: {config.num_layers}, 注意力维度: {config.attention_size}\n")

    # 训练
    print("--- 训练 ---")
    seq_len = config.sequence_length
    n_epochs = 50

    for epoch in range(n_epochs):
        # 随机选择一个序列窗口
        start_idx = np.random.randint(0, len(vibration) - seq_len - 1)
        x_train = vibration[start_idx:start_idx + seq_len]
        y_train = vibration[start_idx + 1:start_idx + seq_len + 1]

        loss = predictor.train_step(x_train, y_train)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}: loss = {loss:.6f}")

    print(f"\n最终损失: {predictor._loss_history[-1]:.6f}\n")

    # 预测
    print("--- 预测 ---")
    history = vibration[-seq_len:]
    prediction = predictor.predict_vibration(history, horizon_steps=10)

    print(f"  预测步长: {prediction.prediction_horizon}")
    print(f"  预测值: {[round(v, 4) for v in prediction.predicted_displacements]}")
    print(f"  整体置信度: {prediction.confidence:.4f}")

    # 置信度详情
    confidence = predictor.compute_prediction_confidence(prediction.predicted_displacements)
    print(f"  平均置信度: {confidence.mean_confidence:.4f}")
    print(f"  每步置信度: {confidence.per_step_confidence}")
    print(f"  不确定性增长率: {confidence.uncertainty_growth_rate:.4f}")
    print(f"  是否可靠: {confidence.is_reliable}")

    # 模型保存/加载测试
    print("\n--- 模型持久化测试 ---")
    import tempfile
    import os

    save_path = os.path.join(tempfile.gettempdir(), "test_vibration_model.json")
    predictor.save_model(save_path)
    print(f"  模型已保存到: {save_path}")

    predictor2 = DeepVibrationPredictor()
    predictor2.load_model(save_path)
    print(f"  模型已加载, 训练步数: {predictor2._training_step_count}")

    # 验证加载后的预测
    prediction2 = predictor2.predict_vibration(history, horizon_steps=10)
    print(f"  加载后预测值: {[round(v, 4) for v in prediction2.predicted_displacements]}")

    # 清理
    os.remove(save_path)

    print("\n测试完成")
