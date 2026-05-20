"""
因果状态预测器 (Causal State Predictor)

基于因果 (单向) 建模的实时光斑状态预测模块。使用因果 1D 卷积和
状态空间模型进行时序特征提取与未来状态预测，确保在线推理时
不会泄露未来信息。

灵感来源:
- Mamba / S6 (state-spaces/mamba): 选择性状态空间模型
  (https://github.com/state-spaces/mamba)
- CausalConv1D: 因果 1D 卷积 (Mamba 架构组件)
- S4 (HazyResearch/state-spaces): 结构化状态空间序列模型
  (https://github.com/HazyResearch/state-spaces)

算法原理:
  1. 因果 1D 卷积: 仅使用当前及过去的信息提取时序特征
  2. 状态空间模型 (SSM):
     dx/dt = Ax + Bu
     y = Cx + Du
     通过零阶保持 (ZOH) 离散化:
     x[k+1] = A_bar * x[k] + B_bar * u[k]
  3. 选择性扫描机制 (简化 Mamba):
     根据输入内容动态调整状态更新速率
  4. 在线推理: 逐帧更新状态，无未来信息依赖

外部依赖: numpy
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from enum import Enum

logger = logging.getLogger(__name__)


class ODESolverType(Enum):
    """ODE 离散化方法枚举。"""
    ZOH = "zoh"       # 零阶保持 (Zero-Order Hold)
    EULER = "euler"    # 前向欧拉
    BILINEAR = "bilinear"  # 双线性变换


@dataclass
class CausalPredictionResult:
    """因果状态预测结果。"""
    predicted_state: np.ndarray = None       # 预测的状态向量
    predicted_position: np.ndarray = None    # 预测的光斑位置 (x, y)
    state_uncertainty: float = 0.0           # 状态不确定性
    prediction_horizon: float = 0.0          # 预测时间范围 (秒)
    causal_features: np.ndarray = None       # 因果特征向量
    scan_output: np.ndarray = None           # 扫描输出 (选择性扫描)
    processing_time_ms: float = 0.0          # 处理耗时 (ms)
    is_online: bool = True                   # 是否在线模式


@dataclass
class CausalPredictorConfig:
    """因果状态预测器配置。"""
    # 状态空间模型参数
    state_dim: int = 16                      # 状态维度
    conv_kernel_size: int = 3                # 因果卷积核大小
    scan_window: int = 32                    # 扫描窗口长度

    # 离散化
    dt_discretization: float = 0.01          # 离散化时间步长 (秒)
    discretization_method: ODESolverType = ODESolverType.ZOH

    # 选择性扫描
    selective_scan_enabled: bool = True      # 启用选择性扫描
    selection_gate_threshold: float = 0.5    # 选择门阈值

    # 自适应状态维度
    adaptive_state_dim: bool = True          # 根据跟踪难度自适应调整
    min_state_dim: int = 8                   # 最小状态维度
    max_state_dim: int = 32                  # 最大状态维度
    difficulty_window: int = 20              # 难度评估窗口

    # 输入输出
    input_dim: int = 4                       # 输入维度 (x, y, dx, dy)
    output_dim: int = 2                      # 输出维度 (x, y)

    # 正则化
    state_decay: float = 0.99                # 状态衰减因子 (防止发散)
    input_noise_std: float = 0.01            # 输入噪声标准差


class CausalStatePredictor:
    """因果状态预测器。

    使用因果 1D 卷积和状态空间模型进行实时光斑状态预测。
    确保在线推理时不泄露未来信息。

    使用示例:
        predictor = CausalStatePredictor()
        # 在线模式: 逐帧更新
        for observation in observations:
            result = predictor.update(observation)
            print(f"Predicted position: {result.predicted_position}")
        # 批量模式: 一次性处理序列
        result = predictor.predict_sequence(observation_sequence)
    """

    def __init__(self, config: Optional[CausalPredictorConfig] = None):
        self._config = config or CausalPredictorConfig()
        self._state: Optional[np.ndarray] = None
        self._observation_buffer: List[np.ndarray] = []
        self._feature_buffer: List[np.ndarray] = []
        self._difficulty_history: List[float] = []
        self._current_state_dim: int = self._config.state_dim
        self._ssm_params: Optional[dict] = None
        self._initialized = False

    @property
    def config(self) -> CausalPredictorConfig:
        return self._config

    def update(self, observation: np.ndarray) -> CausalPredictionResult:
        """在线更新状态并预测。

        Args:
            observation: 当前观测 (input_dim,) 或 (input_dim,)

        Returns:
            CausalPredictionResult: 预测结果
        """
        import time
        t0 = time.perf_counter()

        observation = np.asarray(observation, dtype=np.float64).ravel()

        if observation.shape[0] != self._config.input_dim:
            logger.warning(
                f"Input dim mismatch: expected {self._config.input_dim}, "
                f"got {observation.shape[0]}. Padding/truncating."
            )
            padded = np.zeros(self._config.input_dim, dtype=np.float64)
            n = min(observation.shape[0], self._config.input_dim)
            padded[:n] = observation[:n]
            observation = padded

        # 添加噪声 (正则化)
        noise = np.random.normal(0, self._config.input_noise_std, observation.shape)
        noisy_obs = observation + noise

        # 更新观测缓冲
        self._observation_buffer.append(noisy_obs)
        max_buffer = self._config.scan_window * 2
        if len(self._observation_buffer) > max_buffer:
            self._observation_buffer.pop(0)

        # 初始化 SSM 参数
        if not self._initialized:
            self._initialize_ssm()
            self._initialized = True

        # 自适应状态维度
        if self._config.adaptive_state_dim:
            self._adapt_state_dimension()

        # 因果 1D 卷积特征提取
        causal_features = self._causal_conv1d(noisy_obs)

        # 状态空间模型更新
        self._ssm_step(causal_features)

        # 选择性扫描
        if self._config.selective_scan_enabled:
            scan_output = self._selective_scan(causal_features)
        else:
            scan_output = self._state.copy()

        # 预测输出
        predicted_state = self._state.copy()
        predicted_position = self._state_to_position(predicted_state)

        # 计算不确定性
        uncertainty = self._compute_uncertainty()

        # 更新难度历史
        difficulty = self._estimate_tracking_difficulty()
        self._difficulty_history.append(difficulty)
        if len(self._difficulty_history) > self._config.difficulty_window:
            self._difficulty_history.pop(0)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        result = CausalPredictionResult(
            predicted_state=predicted_state,
            predicted_position=predicted_position,
            state_uncertainty=uncertainty,
            prediction_horizon=self._config.dt_discretization,
            causal_features=causal_features,
            scan_output=scan_output,
            processing_time_ms=elapsed_ms,
            is_online=True,
        )

        logger.debug(
            f"CausalStatePredictor: pos={predicted_position}, "
            f"uncertainty={uncertainty:.4f}, state_dim={self._current_state_dim}, "
            f"time={elapsed_ms:.2f}ms"
        )

        return result

    def predict_sequence(
        self, observations: np.ndarray
    ) -> List[CausalPredictionResult]:
        """批量预测序列。

        Args:
            observations: 观测序列 (T, input_dim)

        Returns:
            预测结果列表
        """
        results = []
        for i in range(observations.shape[0]):
            result = self.update(observations[i])
            results.append(result)
        return results

    def _initialize_ssm(self):
        """初始化状态空间模型参数。

        SSM: dx/dt = Ax + Bu, y = Cx + Du

        A: 状态转移矩阵 (HiPPO 初始化)
        B: 输入矩阵
        C: 输出矩阵
        D: 直通矩阵
        """
        cfg = self._config
        n = self._current_state_dim
        m = cfg.input_dim
        p = cfg.output_dim

        # HiPPO 矩阵初始化 (简化版)
        # A: 对角占优矩阵，特征值为负实部 (稳定系统)
        eigenvalues = -np.exp(np.linspace(-3, 0, n)).astype(np.float64)
        A = np.diag(eigenvalues)

        # 添加弱耦合 (使系统具有动态行为)
        off_diag = np.random.normal(0, 0.01, (n, n)).astype(np.float64)
        A = A + off_diag - off_diag.T  # 保持反对称部分

        # B: 随机输入映射
        B = np.random.normal(0, 0.1, (n, m)).astype(np.float64)

        # C: 输出映射 (前 output_dim 个状态)
        C = np.zeros((p, n), dtype=np.float64)
        C[0, 0] = 1.0
        if p > 1 and n > 1:
            C[1, 1] = 1.0

        # D: 直通项
        D = np.zeros((p, m), dtype=np.float64)

        # 离散化
        A_bar, B_bar = self._discretize(A, B, cfg.dt_discretization)

        self._ssm_params = {
            'A': A, 'B': B, 'C': C, 'D': D,
            'A_bar': A_bar, 'B_bar': B_bar,
        }

        # 初始化状态
        self._state = np.zeros(n, dtype=np.float64)

        logger.info(
            f"CausalStatePredictor: SSM initialized, "
            f"state_dim={n}, dt={cfg.dt_discretization}"
        )

    def _discretize(
        self, A: np.ndarray, B: np.ndarray, dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """离散化连续时间 SSM。

        支持三种方法:
        - ZOH: A_bar = exp(A*dt), B_bar = A^{-1}(exp(A*dt) - I)B
        - EULER: A_bar = I + A*dt, B_bar = B*dt
        - BILINEAR: A_bar = (I - A*dt/2)^{-1}(I + A*dt/2)

        Args:
            A: 连续状态矩阵 (n, n)
            B: 连续输入矩阵 (n, m)
            dt: 时间步长

        Returns:
            (A_bar, B_bar): 离散化矩阵
        """
        cfg = self._config
        method = cfg.discretization_method
        n = A.shape[0]

        if method == ODESolverType.EULER:
            A_bar = np.eye(n) + A * dt
            B_bar = B * dt

        elif method == ODESolverType.BILINEAR:
            I = np.eye(n)
            lhs = I - A * dt / 2
            rhs = I + A * dt / 2
            try:
                A_bar = np.linalg.solve(lhs, rhs)
                B_bar = np.linalg.solve(lhs, B * dt)
            except np.linalg.LinAlgError:
                # 回退到欧拉
                A_bar = np.eye(n) + A * dt
                B_bar = B * dt

        else:  # ZOH (默认)
            # 矩阵指数 (Taylor 展开)
            A_bar = np.eye(n)
            power = np.eye(n)
            for k in range(1, 20):
                power = power @ A * dt / k
                A_bar += power
                if np.max(np.abs(power)) < 1e-12:
                    break

            # B_bar = A^{-1}(A_bar - I)B
            try:
                B_bar = np.linalg.solve(A, (A_bar - np.eye(n)) @ B)
            except np.linalg.LinAlgError:
                B_bar = B * dt

        return A_bar.astype(np.float64), B_bar.astype(np.float64)

    def _causal_conv1d(self, input_vec: np.ndarray) -> np.ndarray:
        """因果 1D 卷积特征提取。

        使用历史观测缓冲进行因果卷积，仅使用当前和过去的信息。

        Args:
            input_vec: 当前输入 (input_dim,)

        Returns:
            特征向量 (state_dim,)
        """
        cfg = self._config
        k = cfg.conv_kernel_size
        n = self._current_state_dim

        # 构建输入序列 (最近 k 个观测)
        buffer = self._observation_buffer
        if len(buffer) < k:
            # 填充零
            padded = [np.zeros_like(input_vec)] * (k - len(buffer)) + list(buffer)
        else:
            padded = list(buffer[-k:])

        # 因果卷积: 对每个输入维度独立卷积
        # 简化: 使用加权求和作为卷积
        weights = np.array([0.1, 0.3, 0.6])[:k]
        weights = weights / weights.sum()

        stacked = np.array(padded)  # (k, input_dim)
        convolved = np.sum(stacked * weights[:, np.newaxis], axis=0)

        # 投影到状态维度
        if not self._initialized:
            self._initialize_ssm()

        # 使用 SSM 的 B 矩阵作为投影
        B = self._ssm_params['B']
        # B: (state_dim, input_dim), convolved: (input_dim,) -> features: (state_dim,)
        features = B @ convolved

        # 截断或填充到当前状态维度
        if features.shape[0] > n:
            features = features[:n]
        elif features.shape[0] < n:
            padded_feat = np.zeros(n, dtype=np.float64)
            padded_feat[:features.shape[0]] = features
            features = padded_feat

        return features

    def _ssm_step(self, input_features: np.ndarray):
        """状态空间模型单步更新。

        x[k+1] = A_bar * x[k] + B_bar * u[k]

        Args:
            input_features: 输入特征 (state_dim,)
        """
        if self._ssm_params is None:
            return

        cfg = self._config
        A_bar = self._ssm_params['A_bar']
        B_bar = self._ssm_params['B_bar']

        # 确保维度匹配
        n = self._current_state_dim
        if A_bar.shape[0] != n:
            # 重新初始化
            self._initialize_ssm()
            A_bar = self._ssm_params['A_bar']
            B_bar = self._ssm_params['B_bar']

        # 截断输入到状态维度
        u = input_features[:n]

        # 状态更新: A_bar @ state + input_features (已投影到状态空间)
        # causal_features 已经通过 B 矩阵投影到 state_dim
        new_state = A_bar @ self._state + u

        # 状态衰减 (防止发散)
        new_state *= cfg.state_decay

        # 裁剪异常值
        max_val = 100.0
        new_state = np.clip(new_state, -max_val, max_val)

        self._state = new_state

    def _selective_scan(self, input_features: np.ndarray) -> np.ndarray:
        """选择性扫描机制 (简化 Mamba)。

        根据输入内容动态调整状态更新:
        - 重要输入 -> 更新更多状态
        - 不重要输入 -> 保持状态不变

        Args:
            input_features: 输入特征 (state_dim,)

        Returns:
            扫描输出 (state_dim,)
        """
        cfg = self._config
        n = self._current_state_dim

        # 计算输入重要性 (基于输入幅度)
        input_magnitude = np.abs(input_features)
        if np.max(input_magnitude) < 1e-10:
            return self._state.copy()

        # 归一化
        normalized_mag = input_magnitude / (np.max(input_magnitude) + 1e-10)

        # 选择门: sigmoid 函数
        gate = 1.0 / (1.0 + np.exp(
            -10.0 * (normalized_mag - cfg.selection_gate_threshold)
        ))

        # 门控更新
        output = gate * self._state + (1 - gate) * input_features[:n]

        return output

    def _state_to_position(self, state: np.ndarray) -> np.ndarray:
        """将状态向量转换为预测位置。

        Args:
            state: 状态向量 (state_dim,)

        Returns:
            预测位置 (output_dim,)
        """
        if self._ssm_params is None:
            return np.zeros(self._config.output_dim, dtype=np.float64)

        C = self._ssm_params['C']
        n = self._current_state_dim

        if C.shape[1] != n:
            C = C[:, :n]

        position = C @ state
        return position

    def _compute_uncertainty(self) -> float:
        """计算状态不确定性。

        基于状态向量的范数和最近状态变化率。

        Returns:
            不确定性 (0-1)
        """
        if self._state is None:
            return 1.0

        # 状态范数 (归一化)
        state_norm = np.linalg.norm(self._state)

        # 如果有历史特征，计算变化率
        if len(self._feature_buffer) >= 2:
            recent_changes = []
            for i in range(1, min(5, len(self._feature_buffer))):
                change = np.linalg.norm(
                    self._feature_buffer[-i] - self._feature_buffer[-i - 1]
                )
                recent_changes.append(change)
            avg_change = np.mean(recent_changes)
            uncertainty = min(1.0, avg_change / (state_norm + 1e-10))
        else:
            uncertainty = 0.5

        return float(np.clip(uncertainty, 0, 1))

    def _estimate_tracking_difficulty(self) -> float:
        """估计当前跟踪难度。

        基于观测变化率和状态不确定性。

        Returns:
            难度评分 (0-1, 1=最难)
        """
        if len(self._observation_buffer) < 2:
            return 0.5

        # 观测变化率
        recent_obs = np.array(self._observation_buffer[-10:])
        if recent_obs.shape[0] < 2:
            return 0.5

        diffs = np.diff(recent_obs, axis=0)
        avg_diff = np.mean(np.linalg.norm(diffs, axis=1))

        # 归一化难度
        difficulty = min(1.0, avg_diff / 10.0)
        return float(difficulty)

    def _adapt_state_dimension(self):
        """根据跟踪难度自适应调整状态维度。

        难度高 -> 增大状态维度 (更多建模能力)
        难度低 -> 减小状态维度 (更高效)
        """
        cfg = self._config

        if len(self._difficulty_history) < cfg.difficulty_window:
            return

        avg_difficulty = np.mean(self._difficulty_history[-cfg.difficulty_window:])

        # 目标状态维度
        target_dim = int(
            cfg.min_state_dim + avg_difficulty * (cfg.max_state_dim - cfg.min_state_dim)
        )
        target_dim = max(cfg.min_state_dim, min(cfg.max_state_dim, target_dim))

        # 只在变化超过阈值时调整
        if abs(target_dim - self._current_state_dim) >= 4:
            old_dim = self._current_state_dim
            self._current_state_dim = target_dim

            # 调整状态向量
            if self._state is not None:
                new_state = np.zeros(target_dim, dtype=np.float64)
                copy_n = min(old_dim, target_dim)
                new_state[:copy_n] = self._state[:copy_n]
                self._state = new_state

            # 重新初始化 SSM
            self._initialized = False

            logger.info(
                f"CausalStatePredictor: Adapted state dim "
                f"{old_dim} -> {target_dim} (difficulty={avg_difficulty:.2f})"
            )

    def reset(self):
        """重置预测器状态。"""
        self._state = None
        self._observation_buffer.clear()
        self._feature_buffer.clear()
        self._difficulty_history.clear()
        self._initialized = False
        self._current_state_dim = self._config.state_dim
        logger.info("CausalStatePredictor: Reset")

    def get_state(self) -> Optional[np.ndarray]:
        """获取当前状态向量。"""
        return self._state.copy() if self._state is not None else None

    def get_tracking_difficulty(self) -> float:
        """获取当前跟踪难度。"""
        if not self._difficulty_history:
            return 0.5
        return float(self._difficulty_history[-1])
