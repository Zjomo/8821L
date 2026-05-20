"""
Mamba / 状态空间模型 (SSM) 时序预测器 (MambaPredictor)

灵感来源:
- Mamba: Linear-Time Sequence Modeling with Selective State Spaces
  (Gu & Dao, 2024) — 基于选择性状态空间模型的线性时间序列建模架构，
  在语言建模和时序预测任务中展现出卓越性能
- Structured State Space Models (S4, Gu et al. 2022) — 结构化状态空间模型，
  利用 HiPPO 矩阵初始化实现对长程依赖的高效建模
- Bilinear Transform (Tustin Method) — 双线性变换，将连续时间 SSM
  离散化为离散时间递推关系
- Selective Scan Mechanism — 选择性扫描机制，使模型能够根据输入内容
  动态调整状态更新策略，实现选择性记忆与遗忘

算法原理:
  连续时间状态空间模型:
    dx/dt = Ax + Bu
    y = Cx + Du

  其中:
    A: 状态转移矩阵 (N x N)，控制状态的内在动态演化
    B: 输入矩阵 (N x 1)，将外部输入映射到状态空间
    C: 输出矩阵 (1 x N)，从状态空间提取观测输出
    D: 直通矩阵 (标量)，输入到输出的直接通路

  离散化 (零阶保持 / 双线性变换):
    A_bar = (I - dt/2 * A)^{-1} (I + dt/2 * A)
    B_bar = (I - dt/2 * A)^{-1} * dt * B

  离散递推:
    h[t] = A_bar * h[t-1] + B_bar * u[t]
    y[t] = C * h[t] + D * u[t]

  选择性扫描 (简化版):
    B 和 C 矩阵根据输入进行参数化，实现输入依赖的状态更新:
    B_t = B_proj * sigmoid(B_linear @ u_t)
    C_t = C_proj * sigmoid(C_linear @ u_t)

  输入门控 (简化版):
    gate_t = sigmoid(W_gate @ u_t + b_gate)
    h[t] = gate_t * (A_bar * h[t-1] + B_bar_t * u[t]) + (1 - gate_t) * h[t-1]

功能:
- 使用结构化状态空间模型预测未来光斑位置
- 支持选择性扫描机制 (输入依赖的 B、C 矩阵)
- 支持输入门控 (控制信息流动)
- O(n) 线性复杂度推理 (递推形式)
- 提供模型状态检查与性能分析报告
- 支持模型重置与在线更新

依赖: numpy, logging, dataclasses (无 PyTorch/TensorFlow)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger("SpotZoom.MambaPredictor")


# ======================== 数据类 ========================


@dataclass
class MambaConfig:
    """Mamba 预测器配置。

    Parameters
    ----------
    state_dim : int
        SSM 隐状态维度 N。值越大模型容量越高，但计算量也越大。
        典型值: 8, 16, 32, 64。
    input_dim : int
        输入特征维度。对于 2D 光斑位置，设为 2 (x, y)。
    output_dim : int
        输出预测维度。通常与 input_dim 相同。
    dt : float
        离散化时间步长。影响 A_bar, B_bar 的缩放。
        值越小，离散系统越接近连续系统。
    dt_min : float
        可学习 dt 的下界 (用于参数化 dt 时)。
    dt_max : float
        可学习 dt 的上界。
    d_state : int
        选择性扫描中 B、C 投影的中间维度。
    use_selective_scan : bool
        是否启用选择性扫描 (输入依赖的 B、C 矩阵)。
    use_input_gating : bool
        是否启用输入门控机制。
    use_bilinear : bool
        是否使用双线性变换离散化 (True) 或零阶保持 (False)。
    A_init_scheme : str
        A 矩阵初始化方案。可选:
        - "hippo": HiPPO 矩阵 (对角化后的特征值)
        - "random": 随机初始化 (负实数特征值保证稳定性)
        - "diagonal": 对角矩阵，元素为负实数
    buffer_size : int
        历史观测缓冲区大小。
    prediction_horizon : int
        默认预测步长。
    min_samples : int
        最少样本数，低于此数不进行预测。
    """

    state_dim: int = 16
    input_dim: int = 2
    output_dim: int = 2
    dt: float = 0.1
    dt_min: float = 0.001
    dt_max: float = 0.1
    d_state: int = 16
    use_selective_scan: bool = True
    use_input_gating: bool = True
    use_bilinear: bool = True
    A_init_scheme: str = "hippo"
    buffer_size: int = 256
    prediction_horizon: int = 10
    min_samples: int = 3


@dataclass
class MambaPrediction:
    """Mamba 预测结果。

    Attributes
    ----------
    predicted_positions : np.ndarray
        预测的未来光斑位置序列，shape (horizon, output_dim)。
    predicted_velocities : np.ndarray
        预测的速度 (一阶差分)，shape (horizon, output_dim)。
    confidence : float
        整体预测置信度 [0, 1]。
    per_step_confidence : np.ndarray
        每步预测置信度，shape (horizon,)。
    horizon : int
        预测步长。
    timestamp : float
        预测生成时间戳。
    state_norm : float
        预测时刻的隐状态范数 (用于诊断)。
    """

    predicted_positions: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2))
    )
    predicted_velocities: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2))
    )
    confidence: float = 0.0
    per_step_confidence: np.ndarray = field(
        default_factory=lambda: np.zeros(0)
    )
    horizon: int = 0
    timestamp: float = 0.0
    state_norm: float = 0.0


@dataclass
class MambaState:
    """Mamba 模型内部状态快照。

    Attributes
    ----------
    hidden_state : np.ndarray
        SSM 隐状态向量 h，shape (state_dim,)。
    last_observation : np.ndarray
        最近一次观测值，shape (input_dim,)。
    last_output : np.ndarray
        最近一次输出值，shape (output_dim,)。
    observation_count : int
        已处理的观测总数。
    is_initialized : bool
        模型是否已初始化。
    dt_current : float
        当前离散化时间步长。
    """

    hidden_state: np.ndarray = field(
        default_factory=lambda: np.zeros(0)
    )
    last_observation: np.ndarray = field(
        default_factory=lambda: np.zeros(0)
    )
    last_output: np.ndarray = field(
        default_factory=lambda: np.zeros(0)
    )
    observation_count: int = 0
    is_initialized: bool = False
    dt_current: float = 0.0


@dataclass
class MambaReport:
    """Mamba 模型分析报告。

    Attributes
    ----------
    observation_count : int
        已处理的观测总数。
    state_norm : float
        当前隐状态 L2 范数。
    state_energy_distribution : np.ndarray
        隐状态各维度的能量分布 (归一化)。
    a_matrix_stability : float
        A_bar 矩阵的谱半径 (最大特征值绝对值)。
        值 < 1 表示离散系统稳定。
    a_eigenvalues : np.ndarray
        A_bar 矩阵的特征值。
    prediction_history_rmse : float
        历史预测误差 RMSE (如果有)。
    effective_memory_length : float
        估计的有效记忆长度 (基于 A_bar 衰减)。
    input_gate_activity : float
        输入门控的平均激活值 (如果启用)。
    selective_scan_activity : Dict[str, float]
        选择性扫描的 B、C 投影激活统计。
    is_healthy : bool
        模型是否处于健康状态 (数值稳定)。
    diagnostics : Dict[str, str]
        诊断信息字典。
    """

    observation_count: int = 0
    state_norm: float = 0.0
    state_energy_distribution: np.ndarray = field(
        default_factory=lambda: np.zeros(0)
    )
    a_matrix_stability: float = 0.0
    a_eigenvalues: np.ndarray = field(
        default_factory=lambda: np.zeros(0)
    )
    prediction_history_rmse: float = 0.0
    effective_memory_length: float = 0.0
    input_gate_activity: float = 0.0
    selective_scan_activity: Dict[str, float] = field(
        default_factory=dict
    )
    is_healthy: bool = True
    diagnostics: Dict[str, str] = field(
        default_factory=dict
    )


# ======================== 辅助函数 ========================


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid 函数。

    Parameters
    ----------
    x : np.ndarray
        输入数组。

    Returns
    -------
    np.ndarray
        sigmoid(x)，值域 (0, 1)。
    """
    # 数值稳定实现：分别处理正负区域
    pos_mask = x >= 0
    neg_mask = ~pos_mask
    result = np.zeros_like(x, dtype=np.float64)
    result[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
    exp_x = np.exp(x[neg_mask])
    result[neg_mask] = exp_x / (1.0 + exp_x)
    return result


def _softplus(x: np.ndarray) -> np.ndarray:
    """数值稳定的 softplus 函数: log(1 + exp(x))。

    Parameters
    ----------
    x : np.ndarray
        输入数组。

    Returns
    -------
    np.ndarray
        softplus(x)，值域 (0, +inf)。
    """
    return np.where(x > 20.0, x, np.log1p(np.exp(np.clip(x, -50, 20))))


def _init_hippo_matrix(n: int) -> np.ndarray:
    """初始化 HiPPO 矩阵 (对角化形式)。

    基于 HiPPO-LegT (Legendre Tensorized) 方法，
    生成具有良好长程记忆特性的状态转移矩阵。

    HiPPO 矩阵的对角化特征值为:
      lambda_k = -(2k + 1) / 2,  k = 0, 1, ..., N-1

    这些负实数特征值保证系统稳定，且不同的衰减速率
    使模型能同时捕获快速变化和缓慢变化的信号成分。

    Parameters
    ----------
    n : int
        状态维度。

    Returns
    -------
    np.ndarray
        对角化的 A 矩阵，shape (n, n)。
    """
    # HiPPO-LegT 特征值: -(2k+1)/2
    eigenvalues = -np.array([(2 * k + 1) / 2.0 for k in range(n)],
                            dtype=np.float64)
    return np.diag(eigenvalues)


def _init_random_stable_matrix(n: int, scale: float = 1.0) -> np.ndarray:
    """初始化随机稳定的 A 矩阵。

    生成一个具有负实数特征值的矩阵，保证连续时间系统稳定。
    通过 Schur 分解构造: A = Q * diag(eigenvalues) * Q^T

    Parameters
    ----------
    n : int
        状态维度。
    scale : float
        特征值缩放因子。

    Returns
    -------
    np.ndarray
        稳定的 A 矩阵，shape (n, n)。
    """
    # 随机正交矩阵 (通过 QR 分解)
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    # 负实数特征值
    eigenvalues = -np.sort(np.random.uniform(0.1, 2.0, n)) * scale
    A = Q @ np.diag(eigenvalues) @ Q.T
    return A


def _init_diagonal_matrix(n: int, scale: float = 1.0) -> np.ndarray:
    """初始化对角 A 矩阵。

    对角元素为负实数，保证稳定性。

    Parameters
    ----------
    n : int
        状态维度。
    scale : float
        特征值缩放因子。

    Returns
    -------
    np.ndarray
        对角 A 矩阵，shape (n, n)。
    """
    eigenvalues = -np.sort(np.random.uniform(0.1, 2.0, n)) * scale
    return np.diag(eigenvalues)


# ======================== 主类 ========================


class MambaPredictor:
    """Mamba / 状态空间模型时序预测器。

    基于结构化状态空间模型 (SSM) 和选择性扫描机制的时间序列预测器，
    用于 SpotZoom 光学对准系统中的光斑位置预测。

    核心特性:
    1. 结构化状态空间模型: dx/dt = Ax + Bu, y = Cx + Du
    2. 双线性变换离散化: 将连续 SSM 转化为离散递推
    3. 选择性扫描: B、C 矩阵根据输入动态调整
    4. 输入门控: 控制信息流入隐状态的比例
    5. O(n) 线性复杂度推理: 递推形式，无需注意力计算

    使用示例::

        predictor = MambaPredictor()
        for obs in observations:
            predictor.update(obs)
        prediction = predictor.predict(horizon=10)
        report = predictor.analyze()

    Parameters
    ----------
    config : MambaConfig, optional
        预测器配置。为 None 时使用默认配置。
    """

    def __init__(self, config: Optional[MambaConfig] = None):
        self._config = config or MambaConfig()
        self._initialize_matrices()
        self._initialize_state()
        self._initialize_buffers()

        LOGGER.debug(
            "MambaPredictor 初始化完成: state_dim=%d, input_dim=%d, "
            "selective_scan=%s, input_gating=%s, discretization=%s",
            self._config.state_dim,
            self._config.input_dim,
            self._config.use_selective_scan,
            self._config.use_input_gating,
            "bilinear" if self._config.use_bilinear else "zoh",
        )

    # -------------------- 初始化方法 --------------------

    def _initialize_matrices(self) -> None:
        """初始化 SSM 矩阵 (A, B, C, D) 及选择性扫描投影矩阵。

        根据 config.A_init_scheme 选择不同的 A 矩阵初始化方案:
        - "hippo": HiPPO-LegT 对角矩阵 (推荐，具有理论保证的长程记忆)
        - "random": 随机稳定矩阵 (通过 Schur 分解构造)
        - "diagonal": 随机对角矩阵
        """
        cfg = self._config
        N = cfg.state_dim
        D_in = cfg.input_dim
        D_out = cfg.output_dim

        # ---- A 矩阵 (状态转移) ----
        if cfg.A_init_scheme == "hippo":
            self._A = _init_hippo_matrix(N)
        elif cfg.A_init_scheme == "random":
            self._A = _init_random_stable_matrix(N)
        elif cfg.A_init_scheme == "diagonal":
            self._A = _init_diagonal_matrix(N)
        else:
            LOGGER.warning("未知 A_init_scheme='%s'，回退到 hippo",
                           cfg.A_init_scheme)
            self._A = _init_hippo_matrix(N)

        # ---- B 矩阵 (输入映射, N x D_in) ----
        scale_b = np.sqrt(2.0 / (N + D_in))
        self._B = np.random.randn(N, D_in) * scale_b

        # ---- C 矩阵 (输出映射, D_out x N) ----
        scale_c = np.sqrt(2.0 / (N + D_out))
        self._C = np.random.randn(D_out, N) * scale_c

        # ---- D 矩阵 (直通, D_out x D_in) ----
        self._D = np.zeros((D_out, D_in), dtype=np.float64)

        # ---- 离散化 ----
        self._discretize()

        # ---- 选择性扫描投影矩阵 ----
        if cfg.use_selective_scan:
            ds = cfg.d_state
            # B 投影: 从输入空间到选择性 B 空间
            self._B_linear = np.random.randn(ds, D_in) * scale_b
            self._B_proj = np.random.randn(N, ds) * np.sqrt(2.0 / (N + ds))
            # C 投影: 从输入空间到选择性 C 空间
            self._C_linear = np.random.randn(ds, D_in) * scale_c
            self._C_proj = np.random.randn(D_out, ds) * np.sqrt(2.0 / (D_out + ds))
            # dt 投影: 从输入空间到标量 dt
            self._dt_linear = np.random.randn(ds, D_in) * scale_b
            self._dt_proj = np.random.randn(1, ds) * np.sqrt(2.0 / (1 + ds))
            # 选择性扫描激活记录
            self._scan_B_activation: List[float] = []
            self._scan_C_activation: List[float] = []
            self._scan_dt_activation: List[float] = []
        else:
            self._B_linear = None
            self._B_proj = None
            self._C_linear = None
            self._C_proj = None
            self._dt_linear = None
            self._dt_proj = None
            self._scan_B_activation = []
            self._scan_C_activation = []
            self._scan_dt_activation = []

        # ---- 输入门控矩阵 ----
        if cfg.use_input_gating:
            self._W_gate = np.random.randn(N, D_in) * np.sqrt(2.0 / (N + D_in))
            self._b_gate = np.zeros(N, dtype=np.float64)
            self._gate_activations: List[float] = []
        else:
            self._W_gate = None
            self._b_gate = None
            self._gate_activations = []

    def _discretize(self) -> None:
        """将连续时间 SSM 离散化。

        双线性变换 (Tustin 方法):
          A_bar = (I - dt/2 * A)^{-1} (I + dt/2 * A)
          B_bar = (I - dt/2 * A)^{-1} * dt * B

        零阶保持 (ZOH):
          A_bar = exp(A * dt)
          B_bar = A^{-1} (exp(A * dt) - I) * B

        离散化后，递推关系为:
          h[t] = A_bar * h[t-1] + B_bar * u[t]
          y[t] = C * h[t] + D * u[t]
        """
        cfg = self._config
        dt = cfg.dt
        N = cfg.state_dim

        if cfg.use_bilinear:
            # 双线性变换
            I = np.eye(N, dtype=np.float64)
            half_dt_A = (dt / 2.0) * self._A
            # (I - dt/2 * A)^{-1}
            try:
                inv_term = np.linalg.inv(I - half_dt_A)
            except np.linalg.LinAlgError:
                LOGGER.warning("双线性变换矩阵奇异，添加正则化")
                inv_term = np.linalg.inv(I - half_dt_A + 1e-6 * I)
            self._A_bar = inv_term @ (I + half_dt_A)
            self._B_bar = inv_term @ (dt * self._B)
        else:
            # 零阶保持 (ZOH)
            try:
                self._A_bar = np.linalg.matrix_exp(self._A * dt)
                # B_bar = A^{-1} (A_bar - I) B
                A_inv = np.linalg.inv(self._A)
                self._B_bar = A_inv @ (self._A_bar - np.eye(N)) @ self._B
            except np.linalg.LinAlgError:
                LOGGER.warning("ZOH 离散化失败，回退到双线性变换")
                I = np.eye(N, dtype=np.float64)
                half_dt_A = (dt / 2.0) * self._A
                inv_term = np.linalg.inv(I - half_dt_A + 1e-6 * I)
                self._A_bar = inv_term @ (I + half_dt_A)
                self._B_bar = inv_term @ (dt * self._B)

        LOGGER.debug(
            "SSM 离散化完成: dt=%.4f, A_bar 谱半径=%.4f",
            dt, self._spectral_radius(),
        )

    def _initialize_state(self) -> None:
        """初始化模型运行时状态。"""
        N = self._config.state_dim
        D_in = self._config.input_dim
        D_out = self._config.output_dim

        # 隐状态
        self._h = np.zeros(N, dtype=np.float64)
        # 最后一次观测
        self._last_obs: Optional[np.ndarray] = None
        # 最后一次输出
        self._last_output: Optional[np.ndarray] = None
        # 观测计数
        self._obs_count: int = 0
        # 是否已初始化
        self._initialized: bool = False
        # 当前 dt (选择性扫描时动态调整)
        self._dt_current: float = self._config.dt

    def _initialize_buffers(self) -> None:
        """初始化历史数据缓冲区。"""
        self._obs_buffer: List[np.ndarray] = []
        self._output_buffer: List[np.ndarray] = []
        self._prediction_errors: List[float] = []

    # -------------------- 公共属性 --------------------

    @property
    def config(self) -> MambaConfig:
        """当前配置。"""
        return self._config

    @property
    def is_initialized(self) -> bool:
        """模型是否已用观测数据初始化。"""
        return self._initialized

    @property
    def observation_count(self) -> int:
        """已处理的观测总数。"""
        return self._obs_count

    # -------------------- 核心方法 --------------------

    def reset(self) -> None:
        """重置模型状态，清空所有缓冲区和隐状态。

        保留 SSM 矩阵 (A, B, C, D) 和投影矩阵不变，
        仅重置运行时状态。适用于切换跟踪目标或异常恢复。
        """
        self._initialize_state()
        self._initialize_buffers()
        self._scan_B_activation = []
        self._scan_C_activation = []
        self._scan_dt_activation = []
        if self._gate_activations is not None:
            self._gate_activations = []
        LOGGER.info("MambaPredictor 已重置")

    def update(self, observation: np.ndarray) -> np.ndarray:
        """用新的观测值更新模型状态并返回当前输出。

        执行完整的 SSM 前向传播:
        1. (可选) 选择性扫描: 根据输入计算动态 B_t, C_t, dt_t
        2. 离散递推: h[t] = A_bar * h[t-1] + B_bar_t * u[t]
        3. (可选) 输入门控: h[t] = gate * h_new + (1 - gate) * h[t-1]
        4. 输出计算: y[t] = C_t * h[t] + D * u[t]

        Parameters
        ----------
        observation : np.ndarray
            新的观测值，shape (input_dim,) 或 (input_dim,) 的兼容数组。
            对于 2D 光斑位置，为 [x, y]。

        Returns
        -------
        np.ndarray
            当前输出，shape (output_dim,)。
            通常为滤波后的位置估计。

        Raises
        ------
        ValueError
            如果 observation 的维度与 input_dim 不匹配。
        """
        obs = np.asarray(observation, dtype=np.float64).ravel()

        if obs.shape[0] != self._config.input_dim:
            raise ValueError(
                f"观测维度不匹配: 期望 {self._config.input_dim}, "
                f"得到 {obs.shape[0]}"
            )

        # 首次观测初始化
        if not self._initialized:
            self._h = np.zeros(self._config.state_dim, dtype=np.float64)
            self._initialized = True
            self._last_obs = obs.copy()
            self._last_output = obs.copy()
            self._obs_count = 1
            self._obs_buffer.append(obs.copy())
            self._output_buffer.append(obs.copy())
            LOGGER.debug("MambaPredictor 首次初始化: obs=%s", obs)
            return obs.copy()

        u = obs

        # ---- 选择性扫描: 计算输入依赖的 B_t, C_t, dt_t ----
        if self._config.use_selective_scan and self._B_proj is not None:
            # 动态 dt: softplus 保证正值，并裁剪到 [dt_min, dt_max]
            dt_logit = (self._dt_proj @ _sigmoid(self._dt_linear @ u)).item()
            dt_dynamic = float(np.clip(
                _softplus(np.array([dt_logit]))[0],
                self._config.dt_min,
                self._config.dt_max,
            ))
            self._dt_current = dt_dynamic

            # 使用动态 dt 重新离散化
            self._discretize_with_dt(dt_dynamic)

            # 动态 B_t: B_proj @ sigmoid(B_linear @ u)
            B_input = _sigmoid(self._B_linear @ u)
            B_t = self._B_proj @ B_input  # (N,)
            self._scan_B_activation.append(float(np.mean(B_input)))

            # 动态 C_t: C_proj @ sigmoid(C_linear @ u)
            C_input = _sigmoid(self._C_linear @ u)
            C_t = self._C_proj @ C_input  # (D_out,)
            self._scan_C_activation.append(float(np.mean(C_input)))
            self._scan_dt_activation.append(dt_dynamic)

            # 限制激活记录长度
            max_act = self._config.buffer_size
            if len(self._scan_B_activation) > max_act:
                self._scan_B_activation = self._scan_B_activation[-max_act:]
                self._scan_C_activation = self._scan_C_activation[-max_act:]
                self._scan_dt_activation = self._scan_dt_activation[-max_act:]
        else:
            B_t = self._B_bar  # (N, D_in) @ u -> (N,) 在下面处理
            C_t = self._C      # (D_out, N)

        # ---- SSM 递推 ----
        h_prev = self._h.copy()

        if self._config.use_selective_scan and self._B_proj is not None:
            # B_t 是 (N,) 向量，u 是 (D_in,) 向量
            h_new = self._A_bar @ h_prev + B_t * u.sum() / self._config.input_dim
        else:
            # B_bar 是 (N, D_in) 矩阵
            h_new = self._A_bar @ h_prev + self._B_bar @ u

        # ---- 输入门控 ----
        if self._config.use_input_gating and self._W_gate is not None:
            gate = _sigmoid(self._W_gate @ u + self._b_gate)
            self._gate_activations.append(float(np.mean(gate)))
            if len(self._gate_activations) > self._config.buffer_size:
                self._gate_activations = (
                    self._gate_activations[-self._config.buffer_size:]
                )
            h_new = gate * h_new + (1.0 - gate) * h_prev

        # ---- 输出计算 ----
        if self._config.use_selective_scan and self._C_proj is not None:
            # C_t 是 (D_out,)，需要与 h_new (N,) 做内积
            # 这里 C_t = C_proj @ sigmoid(C_linear @ u) 已经是 (D_out,)
            # 但我们需要 C_t @ h_new，所以 C_t 应该是 (D_out, N)
            # 修正: 使用完整的 C_t 矩阵 = C_proj @ diag(sigmoid(C_linear @ u)) @ ...
            # 简化方案: C_t 作为 (D_out, N) = C @ h_new + D @ u
            # 其中 C 使用选择性加权
            C_weighted = self._C * C_input[np.newaxis, :]  # (D_out, N) * (1, N)
            y = C_weighted @ h_new + self._D @ u
        else:
            y = self._C @ h_new + self._D @ u

        # ---- 更新状态 ----
        self._h = h_new
        self._last_obs = obs.copy()
        self._last_output = y.copy()
        self._obs_count += 1

        # ---- 缓冲区管理 ----
        self._obs_buffer.append(obs.copy())
        self._output_buffer.append(y.copy())
        if len(self._obs_buffer) > self._config.buffer_size:
            self._obs_buffer = self._obs_buffer[-self._config.buffer_size:]
            self._output_buffer = self._output_buffer[-self._config.buffer_size:]

        # ---- 记录预测误差 (如果有历史输出) ----
        if len(self._output_buffer) >= 2:
            prev_pred = self._output_buffer[-2]
            error = float(np.linalg.norm(obs - prev_pred))
            self._prediction_errors.append(error)
            if len(self._prediction_errors) > self._config.buffer_size:
                self._prediction_errors = (
                    self._prediction_errors[-self._config.buffer_size:]
                )

        LOGGER.debug(
            "update: obs=%s, output=%s, h_norm=%.4f, dt=%.4f",
            obs, y, np.linalg.norm(self._h), self._dt_current,
        )
        return y

    def predict(self, horizon: Optional[int] = None) -> MambaPrediction:
        """预测未来光斑位置。

        基于当前隐状态，通过自回归递推预测未来 horizon 步的输出。
        每一步使用上一步的预测输出作为输入，递推更新隐状态。

        预测过程:
        1. 保存当前隐状态
        2. 对每一步 t = 1, ..., horizon:
           a. u_t = y_{t-1} (自回归)
           b. h_t = A_bar * h_{t-1} + B_bar * u_t
           c. y_t = C * h_t + D * u_t
        3. 恢复隐状态 (预测不影响模型状态)

        Parameters
        ----------
        horizon : int, optional
            预测步长。为 None 时使用 config.prediction_horizon。

        Returns
        -------
        MambaPrediction
            预测结果，包含预测位置、速度、置信度等。
        """
        if not self._initialized:
            LOGGER.warning("模型未初始化，返回空预测")
            return MambaPrediction()

        if self._obs_count < self._config.min_samples:
            LOGGER.warning(
                "观测数 %d < min_samples %d，返回空预测",
                self._obs_count, self._config.min_samples,
            )
            return MambaPrediction()

        h_horizon = horizon if horizon is not None else self._config.prediction_horizon
        h_horizon = max(1, int(h_horizon))

        # 保存当前状态
        h_backup = self._h.copy()

        # 如果没有上次输出，使用上次观测
        if self._last_output is not None:
            u = self._last_output.copy()
        elif self._last_obs is not None:
            u = self._last_obs.copy()
        else:
            LOGGER.warning("无历史数据，返回空预测")
            return MambaPrediction()

        predicted_positions = []
        per_step_confidence = []

        for step in range(h_horizon):
            # 自回归: 使用上一步输出作为输入
            h_new = self._A_bar @ self._h + self._B_bar @ u
            y = self._C @ h_new + self._D @ u
            self._h = h_new

            predicted_positions.append(y.copy())

            # 置信度随步数指数衰减
            decay = np.exp(-0.1 * (step + 1))
            state_stability = max(0.0, 1.0 - self._spectral_radius())
            confidence = decay * state_stability
            per_step_confidence.append(confidence)

            u = y.copy()

        # 恢复隐状态 (预测不修改模型状态)
        self._h = h_backup

        positions = np.array(predicted_positions)  # (horizon, output_dim)
        velocities = np.diff(positions, axis=0, prepend=positions[:1])

        overall_confidence = float(np.mean(per_step_confidence))

        prediction = MambaPrediction(
            predicted_positions=positions,
            predicted_velocities=velocities,
            confidence=overall_confidence,
            per_step_confidence=np.array(per_step_confidence),
            horizon=h_horizon,
            timestamp=time.time(),
            state_norm=float(np.linalg.norm(h_backup)),
        )

        LOGGER.debug(
            "predict: horizon=%d, confidence=%.4f, state_norm=%.4f",
            h_horizon, overall_confidence, prediction.state_norm,
        )
        return prediction

    def analyze(self) -> MambaReport:
        """生成模型分析报告。

        分析内容包括:
        1. 隐状态能量分布 (各维度的归一化能量)
        2. A_bar 矩阵的谱半径和特征值 (稳定性分析)
        3. 有效记忆长度估计
        4. 历史预测误差 RMSE
        5. 选择性扫描和输入门控的激活统计
        6. 综合健康诊断

        Returns
        -------
        MambaReport
            完整的分析报告。
        """
        diagnostics: Dict[str, str] = {}

        # ---- 隐状态分析 ----
        h_norm = float(np.linalg.norm(self._h))
        if h_norm > 1e-10:
            energy = (self._h ** 2) / (h_norm ** 2)
        else:
            energy = np.zeros(self._config.state_dim)
        energy = np.sort(energy)[::-1]  # 降序排列

        # ---- A_bar 稳定性分析 ----
        eigenvalues = np.linalg.eigvals(self._A_bar)
        spectral_radius = float(np.max(np.abs(eigenvalues)))
        is_stable = spectral_radius < 1.0

        if not is_stable:
            diagnostics["stability"] = (
                f"警告: A_bar 谱半径 {spectral_radius:.4f} >= 1.0，"
                f"离散系统可能不稳定！建议减小 dt 或检查 A 矩阵。"
            )
        else:
            diagnostics["stability"] = (
                f"A_bar 谱半径 {spectral_radius:.4f} < 1.0，系统稳定。"
            )

        # ---- 有效记忆长度估计 ----
        # 基于最大特征值模的衰减: memory ~ -1 / log(|lambda_max|)
        if spectral_radius > 0 and spectral_radius < 1.0:
            memory_length = -1.0 / np.log(spectral_radius)
        elif spectral_radius >= 1.0:
            memory_length = float("inf")
        else:
            memory_length = 0.0
        diagnostics["memory"] = (
            f"有效记忆长度约 {memory_length:.1f} 步"
            if memory_length != float("inf")
            else "有效记忆长度: 无穷大 (系统不稳定)"
        )

        # ---- 预测误差 RMSE ----
        if self._prediction_errors:
            rmse = float(np.sqrt(np.mean(np.array(self._prediction_errors) ** 2)))
        else:
            rmse = 0.0
        diagnostics["prediction_rmse"] = f"历史预测 RMSE: {rmse:.4f} 像素"

        # ---- 选择性扫描激活统计 ----
        scan_activity: Dict[str, float] = {}
        if self._scan_B_activation:
            scan_activity["B_mean"] = float(np.mean(self._scan_B_activation))
            scan_activity["B_std"] = float(np.std(self._scan_B_activation))
        if self._scan_C_activation:
            scan_activity["C_mean"] = float(np.mean(self._scan_C_activation))
            scan_activity["C_std"] = float(np.std(self._scan_C_activation))
        if self._scan_dt_activation:
            scan_activity["dt_mean"] = float(np.mean(self._scan_dt_activation))
            scan_activity["dt_std"] = float(np.std(self._scan_dt_activation))

        # ---- 输入门控激活统计 ----
        if self._gate_activations:
            gate_activity = float(np.mean(self._gate_activations))
        else:
            gate_activity = 0.0

        # ---- 健康诊断 ----
        is_healthy = True
        if not is_stable:
            is_healthy = False
        if h_norm > 1e6:
            is_healthy = False
            diagnostics["health"] = "隐状态范数过大，可能存在数值溢出"
        if np.any(np.isnan(self._h)) or np.any(np.isinf(self._h)):
            is_healthy = False
            diagnostics["health"] = "隐状态包含 NaN 或 Inf"
        if is_healthy and "health" not in diagnostics:
            diagnostics["health"] = "模型状态健康"

        # ---- 能量集中度 ----
        top5_energy = float(np.sum(energy[:5])) if len(energy) >= 5 else float(np.sum(energy))
        diagnostics["energy"] = f"前5维能量占比: {top5_energy:.2%}"

        report = MambaReport(
            observation_count=self._obs_count,
            state_norm=h_norm,
            state_energy_distribution=energy,
            a_matrix_stability=spectral_radius,
            a_eigenvalues=eigenvalues,
            prediction_history_rmse=rmse,
            effective_memory_length=memory_length,
            input_gate_activity=gate_activity,
            selective_scan_activity=scan_activity,
            is_healthy=is_healthy,
            diagnostics=diagnostics,
        )

        LOGGER.info(
            "分析报告: obs_count=%d, state_norm=%.4f, spectral_radius=%.4f, "
            "healthy=%s, rmse=%.4f",
            self._obs_count, h_norm, spectral_radius,
            is_healthy, rmse,
        )
        return report

    # -------------------- 辅助方法 --------------------

    def _discretize_with_dt(self, dt: float) -> None:
        """使用指定 dt 重新离散化 SSM。

        与 _discretize() 相同的算法，但使用自定义 dt 值。
        用于选择性扫描中动态调整时间步长。

        Parameters
        ----------
        dt : float
            离散化时间步长。
        """
        N = self._config.state_dim

        if self._config.use_bilinear:
            I = np.eye(N, dtype=np.float64)
            half_dt_A = (dt / 2.0) * self._A
            try:
                inv_term = np.linalg.inv(I - half_dt_A)
            except np.linalg.LinAlgError:
                inv_term = np.linalg.inv(I - half_dt_A + 1e-6 * I)
            self._A_bar = inv_term @ (I + half_dt_A)
            self._B_bar = inv_term @ (dt * self._B)
        else:
            try:
                self._A_bar = np.linalg.matrix_exp(self._A * dt)
                A_inv = np.linalg.inv(self._A)
                self._B_bar = A_inv @ (self._A_bar - np.eye(N)) @ self._B
            except np.linalg.LinAlgError:
                I = np.eye(N, dtype=np.float64)
                half_dt_A = (dt / 2.0) * self._A
                inv_term = np.linalg.inv(I - half_dt_A + 1e-6 * I)
                self._A_bar = inv_term @ (I + half_dt_A)
                self._B_bar = inv_term @ (dt * self._B)

    def _spectral_radius(self) -> float:
        """计算 A_bar 矩阵的谱半径 (最大特征值绝对值)。

        Returns
        -------
        float
            谱半径。值 < 1 表示离散系统稳定。
        """
        eigenvalues = np.linalg.eigvals(self._A_bar)
        return float(np.max(np.abs(eigenvalues)))

    def get_state(self) -> MambaState:
        """获取当前模型状态快照。

        Returns
        -------
        MambaState
            当前状态的可复制快照。
        """
        return MambaState(
            hidden_state=self._h.copy(),
            last_observation=self._last_obs.copy() if self._last_obs is not None else np.zeros(0),
            last_output=self._last_output.copy() if self._last_output is not None else np.zeros(0),
            observation_count=self._obs_count,
            is_initialized=self._initialized,
            dt_current=self._dt_current,
        )

    def set_state(self, state: MambaState) -> None:
        """从快照恢复模型状态。

        Parameters
        ----------
        state : MambaState
            要恢复的状态快照。
        """
        self._h = state.hidden_state.copy()
        self._last_obs = state.last_observation.copy()
        self._last_output = state.last_output.copy()
        self._obs_count = state.observation_count
        self._initialized = state.is_initialized
        self._dt_current = state.dt_current
        LOGGER.debug("模型状态已从快照恢复: obs_count=%d", self._obs_count)

    def get_matrices(self) -> Dict[str, np.ndarray]:
        """获取当前 SSM 矩阵 (用于调试或可视化)。

        Returns
        -------
        Dict[str, np.ndarray]
            包含 'A', 'B', 'C', 'D', 'A_bar', 'B_bar' 的字典。
        """
        return {
            "A": self._A.copy(),
            "B": self._B.copy(),
            "C": self._C.copy(),
            "D": self._D.copy(),
            "A_bar": self._A_bar.copy(),
            "B_bar": self._B_bar.copy(),
        }

    def get_history(self) -> Dict[str, np.ndarray]:
        """获取历史观测和输出数据。

        Returns
        -------
        Dict[str, np.ndarray]
            包含 'observations' 和 'outputs' 的字典。
        """
        obs = np.array(self._obs_buffer) if self._obs_buffer else np.zeros((0, self._config.input_dim))
        outs = np.array(self._output_buffer) if self._output_buffer else np.zeros((0, self._config.output_dim))
        return {
            "observations": obs,
            "outputs": outs,
        }

    def __repr__(self) -> str:
        return (
            f"MambaPredictor("
            f"state_dim={self._config.state_dim}, "
            f"input_dim={self._config.input_dim}, "
            f"output_dim={self._config.output_dim}, "
            f"selective_scan={self._config.use_selective_scan}, "
            f"input_gating={self._config.use_input_gating}, "
            f"obs_count={self._obs_count}, "
            f"initialized={self._initialized})"
        )
