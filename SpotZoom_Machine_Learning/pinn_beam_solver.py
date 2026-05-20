"""
物理信息神经网络光束传播求解器 (PINNBeamSolver)

灵感来源:
- Raissi, Perdikaris, Karniadakis (2019) — Physics-Informed Neural Networks,
  将物理方程 (PDE) 作为损失函数约束嵌入神经网络训练
- Chen et al. (2024, Nature) — PINN 在光学衍射与波传播中的应用,
  展示了 PINN 在求解 Helmholtz 方程和 Fresnel 衍射方面的有效性
- Wang et al. (2025, Nature Photonics) — 深度学习驱动的光束整形与传播预测,
  将 Fresnel 积分作为物理约束实现数据高效的场分布重建
- prysm (https://github.com/brandondube/prysm) — 纯 Python 光学传播,
  提供标量衍射理论的参考实现
- numpy-based autograd — 数值梯度 (中心差分法) 实现自动微分

算法原理:
- Fully Connected Neural Network — 全连接前馈神经网络,
  输入 (x, z) 空间坐标, 输出复振幅场 (Re[U], Im[U])
- ReLU Activation — 修正线性单元激活函数: f(x) = max(0, x)
- Fresnel Diffraction Integral — Fresnel 衍射积分作为物理损失:
    U(x, z) = (1/sqrt(i*lambda*z)) * integral[U(x', 0) * exp(i*pi*(x-x')^2/(lambda*z)) dx']
    网络预测场应满足此传播方程
- Helmholtz Equation — 亥姆霍兹方程残差作为正则化:
    d^2U/dx^2 + d^2U/dz^2 + k^2 * U = 0
    其中 k = 2*pi/lambda 为波数
- Adam Optimizer (numpy) — 纯 numpy 实现的自适应矩估计优化器,
  一阶矩 m_t = beta1*m_{t-1} + (1-beta1)*g_t
  二阶矩 v_t = beta2*v_{t-1} + (1-beta2)*g_t^2
  参数更新: theta -= lr * m_hat / (sqrt(v_hat) + eps)
- Numerical Differentiation — 中心差分法自动微分:
    df/dx ≈ (f(x+h) - f(x-h)) / (2h)
    d^2f/dx^2 ≈ (f(x+h) - 2*f(x) + f(x-h)) / h^2
- MSE Loss — 均方误差数据损失:
    L_data = (1/N) * sum(|I_pred - I_meas|^2)
- Physics-Informed Loss — 物理信息总损失:
    L_total = lambda_data * L_data + lambda_fresnel * L_fresnel + lambda_helmholtz * L_helmholtz

功能:
- 求解光束传播逆问题: 从测量强度分布推断初始场分布
- 预测任意传播距离处的光束场分布
- Fresnel 衍射积分物理约束
- Helmholtz 方程正则化
- 纯 numpy 实现, 无 PyTorch/TensorFlow 依赖
- 可配置网络结构和损失权重
- 完整的训练历史与收敛监控

依赖: numpy (无其他外部依赖)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger("SpotZoom.PINNBeamSolver")

# 模块默认禁用标志
pinn_beam_solver_enabled: bool = False


# ======================== 物理常数 ========================

SPEED_OF_LIGHT = 2.998e8       # 光速 (m/s)
PLANCK_CONSTANT = 6.626e-34    # 普朗克常数 (J*s)


# ======================== 数据类 ========================


@dataclass
class PINNConfig:
    """PINN 光束求解器配置。

    Attributes
    ----------
    wavelength_m : float
        光束波长 (米)，默认 632.8nm (HeNe 激光)。
    grid_size : int
        内部计算网格尺寸 (采样点数)，用于横向坐标离散化。
    z_max_m : float
        最大传播距离 (米)。
    n_layers : List[int]
        神经网络各层神经元数量。默认 [2, 64, 64, 64, 2]，
        输入层 2 (x, z)，输出层 2 (Re[U], Im[U])。
    learning_rate : float
        Adam 优化器初始学习率。
    adam_beta1 : float
        Adam 一阶矩衰减率。
    adam_beta2 : float
        Adam 二阶矩衰减率。
    adam_epsilon : float
        Adam 数值稳定项。
    max_epochs : int
        最大训练轮数。
    convergence_threshold : float
        收敛判定阈值 (损失变化小于此值时提前停止)。
    patience : int
        早停耐心值 (连续多少轮损失无改善后停止)。
    lambda_data : float
        数据损失权重 (MSE against measured intensity)。
    lambda_fresnel : float
        Fresnel 衍射物理损失权重。
    lambda_helmholtz : float
        Helmholtz 方程正则化权重。
    grad_epsilon : float
        数值微分步长 (米)。
    n_collocation : int
        配点数量 (物理约束采样点数)。
    n_physics_samples : int
        每轮用于计算物理损失的采样点数。
    beam_waist_m : float
        初始高斯光束束腰半径 (米)，用于 Xavier 初始化参考。
    random_seed : Optional[int]
        随机种子，为 None 时不固定。
    """

    wavelength_m: float = 632.8e-9
    grid_size: int = 128
    z_max_m: float = 0.1
    n_layers: List[int] = field(default_factory=lambda: [2, 64, 64, 64, 2])
    learning_rate: float = 1e-3
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_epochs: int = 500
    convergence_threshold: float = 1e-6
    patience: int = 50
    lambda_data: float = 1.0
    lambda_fresnel: float = 0.5
    lambda_helmholtz: float = 0.1
    grad_epsilon: float = 1e-7
    n_collocation: int = 256
    n_physics_samples: int = 128
    beam_waist_m: float = 50e-6
    random_seed: Optional[int] = None


@dataclass
class PINNBeamProfile:
    """PINN 预测的光束场分布。

    Attributes
    ----------
    x_coords : np.ndarray
        横向坐标数组 (米)。
    z_distance : float
        传播距离 (米)。
    amplitude : np.ndarray
        场振幅分布 |U(x)|。
    phase : np.ndarray
        场相位分布 arg(U(x)) (弧度)。
    intensity : np.ndarray
        场强度分布 |U(x)|^2 (归一化)。
    complex_field : np.ndarray
        复振幅场 U(x) = amplitude * exp(i*phase)。
    beam_width_m : float
        估算的光束宽度 (1/e^2 半径, 米)。
    peak_intensity : float
        峰值强度。
    """

    x_coords: np.ndarray
    z_distance: float
    amplitude: np.ndarray
    phase: np.ndarray
    intensity: np.ndarray
    complex_field: np.ndarray
    beam_width_m: float
    peak_intensity: float


@dataclass
class PINNResult:
    """PINN 求解结果。

    Attributes
    ----------
    is_converged : bool
        是否成功收敛。
    total_epochs : int
        实际训练轮数。
    final_loss : float
        最终总损失值。
    final_data_loss : float
        最终数据损失值。
    final_physics_loss : float
        最终物理损失值 (Fresnel + Helmholtz)。
    loss_history : List[float]
        每轮总损失历史记录。
    data_loss_history : List[float]
        每轮数据损失历史记录。
    physics_loss_history : List[float]
        每轮物理损失历史记录。
    initial_field : np.ndarray
        求解得到的初始场 (z=0) 复振幅分布。
    training_time_s : float
        训练耗时 (秒)。
    """

    is_converged: bool
    total_epochs: int
    final_loss: float
    final_data_loss: float
    final_physics_loss: float
    loss_history: List[float]
    data_loss_history: List[float]
    physics_loss_history: List[float]
    initial_field: np.ndarray
    training_time_s: float


@dataclass
class PINNReport:
    """PINN 分析报告。

    Attributes
    ----------
    wavelength_m : float
        使用的工作波长 (米)。
    network_architecture : str
        网络结构描述。
    total_parameters : int
        网络可训练参数总数。
    is_converged : bool
        是否收敛。
    convergence_epoch : int
        收敛时的轮数 (未收敛则为 -1)。
    final_loss : float
        最终总损失。
    final_data_loss : float
        最终数据损失。
    final_fresnel_loss : float
        最终 Fresnel 物理损失。
    final_helmholtz_loss : float
        最终 Helmholtz 正则化损失。
    training_time_s : float
        训练耗时 (秒)。
    initial_beam_width_m : float
        初始场估算光束宽度 (米)。
    initial_peak_strehl : float
        初始场峰值 Strehl 比 (归一化峰值)。
    loss_reduction_ratio : float
        损失下降比率 (初始损失 / 最终损失)。
    gradient_norm_final : float
        最终轮次梯度范数。
    recommendations : List[str]
        诊断建议列表。
    """

    wavelength_m: float
    network_architecture: str
    total_parameters: int
    is_converged: bool
    convergence_epoch: int
    final_loss: float
    final_data_loss: float
    final_fresnel_loss: float
    final_helmholtz_loss: float
    training_time_s: float
    initial_beam_width_m: float
    initial_peak_strehl: float
    loss_reduction_ratio: float
    gradient_norm_final: float
    recommendations: List[str]


# ======================== 神经网络核心 ========================


class _SimpleNeuralNetwork:
    """纯 numpy 实现的全连接前馈神经网络。

    支持任意层数的全连接结构，使用 ReLU 激活函数。
    输入维度和输出维度由 n_layers 的首尾元素决定。

    Parameters
    ----------
    n_layers : List[int]
        各层神经元数量列表，例如 [2, 64, 64, 64, 2]。
    random_seed : Optional[int]
        随机种子。
    """

    def __init__(self, n_layers: List[int], random_seed: Optional[int] = None):
        self.n_layers = list(n_layers)
        self.n_layer_count = len(n_layers)
        self.random_seed = random_seed

        # 权重和偏置列表
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        # Xavier/Glorot 初始化
        rng = np.random.RandomState(random_seed)
        for i in range(self.n_layer_count - 1):
            fan_in = n_layers[i]
            fan_out = n_layers[i + 1]
            # Xavier 初始化: std = sqrt(2 / (fan_in + fan_out))
            std = np.sqrt(2.0 / (fan_in + fan_out))
            W = rng.randn(fan_in, fan_out) * std
            b = np.zeros(fan_out)
            self.weights.append(W)
            self.biases.append(b)

        LOGGER.debug(
            "_SimpleNeuralNetwork: 初始化完成, 结构=%s, 参数量=%d",
            n_layers, self.count_parameters(),
        )

    def count_parameters(self) -> int:
        """计算网络可训练参数总数。"""
        total = 0
        for W, b in zip(self.weights, self.biases):
            total += W.size + b.size
        return total

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        """ReLU 激活函数: max(0, x)。"""
        return np.maximum(0.0, x)

    @staticmethod
    def _relu_derivative(x: np.ndarray) -> np.ndarray:
        """ReLU 导数: x > 0 时为 1，否则为 0。"""
        return (x > 0.0).astype(np.float64)

    def forward(self, X: np.ndarray) -> List[np.ndarray]:
        """前向传播，返回所有层的预激活值和输出。

        Parameters
        ----------
        X : np.ndarray
            输入数据，形状 (N, input_dim)。

        Returns
        -------
        List[np.ndarray]
            各层输出列表，长度为 n_layer_count。
            最后一项为网络最终输出。
        """
        activations = [X]
        current = X
        for i in range(self.n_layer_count - 1):
            z = current @ self.weights[i] + self.biases[i]
            if i < self.n_layer_count - 2:
                # 隐藏层使用 ReLU
                current = self._relu(z)
            else:
                # 输出层不使用激活函数 (线性输出)
                current = z
            activations.append(current)
        return activations

    def predict(self, X: np.ndarray) -> np.ndarray:
        """前向传播，仅返回最终输出。

        Parameters
        ----------
        X : np.ndarray
            输入数据，形状 (N, input_dim)。

        Returns
        -------
        np.ndarray
            网络输出，形状 (N, output_dim)。
        """
        activations = self.forward(X)
        return activations[-1]

    def get_flat_parameters(self) -> np.ndarray:
        """将所有参数展平为一维向量。"""
        parts = []
        for W, b in zip(self.weights, self.biases):
            parts.append(W.ravel())
            parts.append(b.ravel())
        return np.concatenate(parts)

    def set_flat_parameters(self, flat: np.ndarray) -> None:
        """从一维向量恢复所有参数。"""
        idx = 0
        for i in range(self.n_layer_count - 1):
            w_size = self.weights[i].size
            self.weights[i] = flat[idx:idx + w_size].reshape(self.weights[i].shape)
            idx += w_size
            b_size = self.biases[i].size
            self.biases[i] = flat[idx:idx + b_size]
            idx += b_size


class _AdamOptimizer:
    """纯 numpy 实现的 Adam 优化器。

    Parameters
    ----------
    lr : float
        学习率。
    beta1 : float
        一阶矩衰减率。
    beta2 : float
        二阶矩衰减率。
    epsilon : float
        数值稳定项。
    """

    def __init__(self, lr: float = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m: Optional[np.ndarray] = None
        self.v: Optional[np.ndarray] = None

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """执行一步 Adam 参数更新。

        Parameters
        ----------
        params : np.ndarray
            当前参数向量。
        grads : np.ndarray
            梯度向量。

        Returns
        -------
        np.ndarray
            更新后的参数向量。
        """
        self.t += 1
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        # 更新一阶矩和二阶矩
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grads ** 2)

        # 偏差校正
        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)

        # 参数更新
        params = params - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return params

    def reset(self) -> None:
        """重置优化器内部状态。"""
        self.t = 0
        self.m = None
        self.v = None


# ======================== 主求解器类 ========================


class PINNBeamSolver:
    """物理信息神经网络 (PINN) 光束传播求解器。

    使用物理信息神经网络求解光束传播问题。网络输入为空间坐标 (x, z)，
    输出为复振幅场的实部和虚部 (Re[U], Im[U])。训练过程中同时最小化:
      1. 数据损失: 预测强度与实测强度的均方误差
      2. Fresnel 衍射物理损失: 场分布应满足 Fresnel 传播积分
      3. Helmholtz 方程正则化: 场分布应满足标量亥姆霍兹方程

    该实现完全基于 numpy，不依赖 PyTorch 或 TensorFlow。
    自动微分通过中心差分法数值梯度实现。

    Parameters
    ----------
    config : Optional[PINNConfig]
        求解器配置。为 None 时使用默认配置。

    Examples
    --------
    >>> config = PINNConfig(wavelength_m=632.8e-9, max_epochs=200)
    >>> solver = PINNBeamSolver(config)
    >>> result = solver.solve(measured_data=intensity_profile)
    >>> profile = solver.predict_field(z_distance=0.05)
    >>> report = solver.analyze()
    """

    def __init__(self, config: Optional[PINNConfig] = None):
        self.config = config if config is not None else PINNConfig()

        # 验证配置
        self._validate_config()

        # 物理参数
        self._wavelength = self.config.wavelength_m
        self._k = 2.0 * np.pi / self._wavelength  # 波数

        # 坐标网格
        self._x_max = 5.0 * self.config.beam_waist_m  # 横向范围
        self._x_coords = np.linspace(-self._x_max, self._x_max,
                                     self.config.grid_size)
        self._z_max = self.config.z_max_m

        # 初始化神经网络
        self._nn = _SimpleNeuralNetwork(
            self.config.n_layers,
            random_seed=self.config.random_seed,
        )

        # 初始化 Adam 优化器
        self._optimizer = _AdamOptimizer(
            lr=self.config.learning_rate,
            beta1=self.config.adam_beta1,
            beta2=self.config.adam_beta2,
            epsilon=self.config.adam_epsilon,
        )

        # 训练状态
        self._is_trained: bool = False
        self._last_result: Optional[PINNResult] = None
        self._measured_data: Optional[np.ndarray] = None
        self._loss_history: List[float] = []
        self._data_loss_history: List[float] = []
        self._physics_loss_history: List[float] = []
        self._fresnel_loss_history: List[float] = []
        self._helmholtz_loss_history: List[float] = []
        self._grad_norm_history: List[float] = []

        LOGGER.info(
            "PINNBeamSolver: 初始化完成 (lambda=%.1fnm, layers=%s, params=%d)",
            self._wavelength * 1e9,
            self.config.n_layers,
            self._nn.count_parameters(),
        )

    def _validate_config(self) -> None:
        """验证配置参数的合法性。"""
        cfg = self.config
        if cfg.wavelength_m <= 0:
            raise ValueError(f"wavelength_m 必须 > 0，当前值: {cfg.wavelength_m}")
        if cfg.grid_size < 16:
            raise ValueError(f"grid_size 必须 >= 16，当前值: {cfg.grid_size}")
        if len(cfg.n_layers) < 3:
            raise ValueError(
                f"n_layers 至少需要 3 层 (输入-隐藏-输出)，当前: {cfg.n_layers}"
            )
        if cfg.n_layers[0] != 2:
            raise ValueError(
                f"n_layers 首元素必须为 2 (输入 x, z)，当前: {cfg.n_layers[0]}"
            )
        if cfg.n_layers[-1] != 2:
            raise ValueError(
                f"n_layers 末元素必须为 2 (输出 Re, Im)，当前: {cfg.n_layers[-1]}"
            )
        if cfg.learning_rate <= 0:
            raise ValueError(f"learning_rate 必须 > 0，当前值: {cfg.learning_rate}")
        if cfg.max_epochs < 1:
            raise ValueError(f"max_epochs 必须 >= 1，当前值: {cfg.max_epochs}")
        if cfg.lambda_data < 0 or cfg.lambda_fresnel < 0 or cfg.lambda_helmholtz < 0:
            raise ValueError("损失权重 lambda_data/fresnel/helmholtz 必须 >= 0")
        if cfg.beam_waist_m <= 0:
            raise ValueError(f"beam_waist_m 必须 > 0，当前值: {cfg.beam_waist_m}")

    def reset(self) -> None:
        """重置求解器到初始状态。

        重新初始化神经网络权重、优化器状态和训练历史。
        配置参数保持不变。
        """
        self._nn = _SimpleNeuralNetwork(
            self.config.n_layers,
            random_seed=self.config.random_seed,
        )
        self._optimizer = _AdamOptimizer(
            lr=self.config.learning_rate,
            beta1=self.config.adam_beta1,
            beta2=self.config.adam_beta2,
            epsilon=self.config.adam_epsilon,
        )
        self._is_trained = False
        self._last_result = None
        self._measured_data = None
        self._loss_history.clear()
        self._data_loss_history.clear()
        self._physics_loss_history.clear()
        self._fresnel_loss_history.clear()
        self._helmholtz_loss_history.clear()
        self._grad_norm_history.clear()

        LOGGER.info("PINNBeamSolver: 已重置")

    def _normalize_input(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """将物理坐标归一化到 [-1, 1] 范围，提高数值稳定性。

        Parameters
        ----------
        x : np.ndarray
            横向坐标 (米)。
        z : np.ndarray
            传播距离坐标 (米)。

        Returns
        -------
        np.ndarray
            归一化后的输入矩阵，形状 (N, 2)。
        """
        x_norm = x / self._x_max
        z_norm = z / self._z_max
        return np.column_stack([x_norm, z_norm])

    def _compute_data_loss(self, params: np.ndarray) -> float:
        """计算数据损失: 预测强度与实测强度的 MSE。

        Parameters
        ----------
        params : np.ndarray
            展平的网络参数向量。

        Returns
        -------
        float
            数据损失值。
        """
        if self._measured_data is None:
            return 0.0

        self._nn.set_flat_parameters(params)

        # 在 z=0 平面预测场
        z_zeros = np.zeros(self.config.grid_size)
        X_input = self._normalize_input(self._x_coords, z_zeros)
        output = self._nn.predict(X_input)  # (N, 2): [Re, Im]

        # 计算预测强度
        predicted_intensity = output[:, 0] ** 2 + output[:, 1] ** 2

        # 归一化
        pred_max = np.max(predicted_intensity)
        if pred_max > 1e-30:
            predicted_intensity = predicted_intensity / pred_max

        meas_max = np.max(self._measured_data)
        if meas_max > 1e-30:
            measured_normalized = self._measured_data / meas_max
        else:
            measured_normalized = self._measured_data

        # MSE
        mse = np.mean((predicted_intensity - measured_normalized) ** 2)
        return float(mse)

    def _compute_fresnel_loss(self, params: np.ndarray) -> float:
        """计算 Fresnel 衍射积分物理损失。

        在多个 z 平面上验证网络预测场是否满足 Fresnel 传播关系。
        从 z=0 的场通过 Fresnel 积分计算 z=z_i 处的场，
        与网络直接预测的场进行比较。

        Parameters
        ----------
        params : np.ndarray
            展平的网络参数向量。

        Returns
        -------
        float
            Fresnel 物理损失值。
        """
        self._nn.set_flat_parameters(params)

        # 获取 z=0 处的场 (源场)
        z_zeros = np.zeros(self.config.grid_size)
        X_src = self._normalize_input(self._x_coords, z_zeros)
        src_output = self._nn.predict(X_src)
        U_src = src_output[:, 0] + 1j * src_output[:, 1]

        # 在若干 z 采样点验证 Fresnel 传播
        n_z_samples = min(8, max(2, self.config.n_physics_samples // 16))
        z_samples = np.linspace(
            self._wavelength * 100,
            self._z_max * 0.8,
            n_z_samples,
        )

        total_residual = 0.0
        dx = self._x_coords[1] - self._x_coords[0]

        for z_val in z_samples:
            # 网络直接预测 z=z_val 处的场
            X_tgt = self._normalize_input(self._x_coords,
                                          np.full(self.config.grid_size, z_val))
            tgt_output = self._nn.predict(X_tgt)
            U_pinn = tgt_output[:, 0] + 1j * tgt_output[:, 1]

            # Fresnel 积分计算传播场 (离散近似)
            U_fresnel = self._fresnel_propagate(U_src, z_val, dx)

            # 归一化后比较
            pinn_norm = np.max(np.abs(U_pinn))
            fresnel_norm = np.max(np.abs(U_fresnel))

            if pinn_norm > 1e-30 and fresnel_norm > 1e-30:
                U_pinn_norm = U_pinn / pinn_norm
                U_fresnel_norm = U_fresnel / fresnel_norm
                residual = np.mean(np.abs(U_pinn_norm - U_fresnel_norm) ** 2)
            else:
                residual = 0.0

            total_residual += residual

        avg_residual = total_residual / n_z_samples
        return float(avg_residual)

    def _fresnel_propagate(self, U_in: np.ndarray, z: float,
                           dx: float) -> np.ndarray:
        """Fresnel 衍射积分传播 (离散近似)。

        使用传递函数方法计算 Fresnel 传播:
            U_out = IFFT[ FFT[U_in] * H(f) ]
        其中 H(f) = exp(i*k*z) * exp(-i*pi*lambda*z*f^2)

        Parameters
        ----------
        U_in : np.ndarray
            输入场复振幅。
        z : float
            传播距离 (米)。
        dx : float
            空间采样间隔 (米)。

        Returns
        -------
        np.ndarray
            传播后的场复振幅。
        """
        N = len(U_in)
        # 空间频率坐标
        fx = np.fft.fftfreq(N, d=dx)

        # Fresnel 传递函数
        H = np.exp(1j * self._k * z) * np.exp(
            -1j * np.pi * self._wavelength * z * fx ** 2
        )

        # 传播
        U_in_fft = np.fft.fft(U_in)
        U_out_fft = U_in_fft * H
        U_out = np.fft.ifft(U_out_fft)

        return U_out

    def _compute_helmholtz_loss(self, params: np.ndarray) -> float:
        """计算 Helmholtz 方程残差正则化损失。

        亥姆霍兹方程: d^2U/dx^2 + d^2U/dz^2 + k^2 * U = 0
        使用中心差分法计算二阶偏导数。

        Parameters
        ----------
        params : np.ndarray
            展平的网络参数向量。

        Returns
        -------
        float
            Helmholtz 残差损失值。
        """
        self._nn.set_flat_parameters(params)

        eps = self.config.grad_epsilon

        # 在配点处采样
        rng = np.random.RandomState(42)
        x_col = rng.uniform(-self._x_max * 0.8, self._x_max * 0.8,
                            self.config.n_physics_samples)
        z_col = rng.uniform(eps * 10, self._z_max * 0.8,
                            self.config.n_physics_samples)

        X_base = self._normalize_input(x_col, z_col)
        U_base = self._nn.predict(X_base)  # (N, 2)

        # d^2U/dx^2: 中心差分
        X_xp = self._normalize_input(x_col + eps, z_col)
        X_xm = self._normalize_input(x_col - eps, z_col)
        U_xp = self._nn.predict(X_xp)
        U_xm = self._nn.predict(X_xm)
        d2U_dx2 = (U_xp - 2.0 * U_base + U_xm) / (eps ** 2)

        # d^2U/dz^2: 中心差分
        X_zp = self._normalize_input(x_col, z_col + eps)
        X_zm = self._normalize_input(x_col, z_col - eps)
        U_zp = self._nn.predict(X_zp)
        U_zm = self._nn.predict(X_zm)
        d2U_dz2 = (U_zp - 2.0 * U_base + U_zm) / (eps ** 2)

        # Helmholtz 残差: d2U/dx2 + d2U/dz2 + k^2 * U
        # 注意: k^2 需要使用归一化坐标的缩放因子
        k_normalized = self._k * self._x_max  # 归一化坐标下的波数
        residual = d2U_dx2 + d2U_dz2 + (k_normalized ** 2) * U_base

        # MSE of residual
        loss = np.mean(residual ** 2)
        return float(loss)

    def _compute_total_loss(self, params: np.ndarray) -> Tuple[float, float, float, float, float]:
        """计算总损失及其各分量。

        Parameters
        ----------
        params : np.ndarray
            展平的网络参数向量。

        Returns
        -------
        Tuple[float, float, float, float, float]
            (total_loss, data_loss, fresnel_loss, helmholtz_loss, physics_loss)
        """
        data_loss = self._compute_data_loss(params)
        fresnel_loss = self._compute_fresnel_loss(params)
        helmholtz_loss = self._compute_helmholtz_loss(params)
        physics_loss = fresnel_loss + helmholtz_loss

        total = (self.config.lambda_data * data_loss
                 + self.config.lambda_fresnel * fresnel_loss
                 + self.config.lambda_helmholtz * helmholtz_loss)

        return total, data_loss, fresnel_loss, helmholtz_loss, physics_loss

    def _compute_numerical_gradient(self, params: np.ndarray) -> np.ndarray:
        """使用中心差分法计算损失函数对参数的数值梯度。

        Parameters
        ----------
        params : np.ndarray
            展平的网络参数向量。

        Returns
        -------
        np.ndarray
            梯度向量，形状与 params 相同。
        """
        eps = self.config.grad_epsilon
        n_params = len(params)
        grads = np.zeros(n_params)

        # 为提高效率，每次前向传播计算两个方向
        base_loss, _, _, _, _ = self._compute_total_loss(params)

        for i in range(n_params):
            params_plus = params.copy()
            params_plus[i] += eps
            loss_plus, _, _, _, _ = self._compute_total_loss(params_plus)

            params_minus = params.copy()
            params_minus[i] -= eps
            loss_minus, _, _, _, _ = self._compute_total_loss(params_minus)

            grads[i] = (loss_plus - loss_minus) / (2.0 * eps)

        return grads

    def _compute_gradient_fast(self, params: np.ndarray) -> np.ndarray:
        """使用优化的中心差分法计算梯度 (向量化实现)。

        对每个参数扰动后仅计算一次前向传播，利用基线损失值。
        相比 _compute_numerical_gradient 更高效。

        Parameters
        ----------
        params : np.ndarray
            展平的网络参数向量。

        Returns
        -------
        np.ndarray
            梯度向量。
        """
        eps = self.config.grad_epsilon
        n_params = len(params)
        grads = np.zeros(n_params)

        base_loss, _, _, _, _ = self._compute_total_loss(params)

        for i in range(n_params):
            params_perturbed = params.copy()
            params_perturbed[i] += eps
            loss_p, _, _, _, _ = self._compute_total_loss(params_perturbed)
            grads[i] = (loss_p - base_loss) / eps

        return grads

    def solve(self, measured_data: Optional[np.ndarray] = None) -> PINNResult:
        """求解光束传播逆问题。

        使用 PINN 方法，结合数据损失和物理约束损失训练神经网络，
        从测量强度分布推断初始场分布。

        Parameters
        ----------
        measured_data : Optional[np.ndarray]
            测量的光束强度分布 (1D 数组)。
            为 None 时仅使用物理约束 (正问题)。
            应与 grid_size 长度匹配，或将被插值到网格上。

        Returns
        -------
        PINNResult
            求解结果，包含收敛状态、损失历史和初始场分布。
        """
        # 处理测量数据
        if measured_data is not None:
            measured_data = np.asarray(measured_data, dtype=np.float64)
            if measured_data.ndim != 1:
                raise ValueError(
                    f"measured_data 必须为 1D 数组，当前维度: {measured_data.ndim}"
                )
            # 插值到网格
            if len(measured_data) != self.config.grid_size:
                from numpy import interp as _np_interp
                x_meas = np.linspace(-self._x_max, self._x_max,
                                     len(measured_data))
                measured_data = _np_interp(self._x_coords, x_meas, measured_data)
            self._measured_data = measured_data
        else:
            self._measured_data = None

        LOGGER.info(
            "PINNBeamSolver: 开始求解 (epochs=%d, lr=%.1e, "
            "lambda_d=%.1f, lambda_f=%.1f, lambda_h=%.1f)",
            self.config.max_epochs, self.config.learning_rate,
            self.config.lambda_data, self.config.lambda_fresnel,
            self.config.lambda_helmholtz,
        )

        # 清空历史
        self._loss_history.clear()
        self._data_loss_history.clear()
        self._physics_loss_history.clear()
        self._fresnel_loss_history.clear()
        self._helmholtz_loss_history.clear()
        self._grad_norm_history.clear()

        start_time = time.time()

        # 获取初始参数
        params = self._nn.get_flat_parameters()

        # 训练循环
        best_loss = float('inf')
        best_params = params.copy()
        no_improve_count = 0

        for epoch in range(1, self.config.max_epochs + 1):
            # 计算损失
            total_loss, data_loss, fresnel_loss, helmholtz_loss, physics_loss = \
                self._compute_total_loss(params)

            # 记录历史
            self._loss_history.append(total_loss)
            self._data_loss_history.append(data_loss)
            self._physics_loss_history.append(physics_loss)
            self._fresnel_loss_history.append(fresnel_loss)
            self._helmholtz_loss_history.append(helmholtz_loss)

            # 计算梯度 (使用快速方法)
            grads = self._compute_gradient_fast(params)
            grad_norm = float(np.linalg.norm(grads))
            self._grad_norm_history.append(grad_norm)

            # Adam 更新
            params = self._optimizer.step(params, grads)

            # 早停检查
            if total_loss < best_loss - self.config.convergence_threshold:
                best_loss = total_loss
                best_params = params.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            # 日志输出 (每 50 轮或首尾)
            if epoch == 1 or epoch % 50 == 0 or epoch == self.config.max_epochs:
                LOGGER.debug(
                    "PINNBeamSolver: Epoch %d/%d, loss=%.6e "
                    "(data=%.2e, fresnel=%.2e, helm=%.2e, |g|=%.2e)",
                    epoch, self.config.max_epochs, total_loss,
                    data_loss, fresnel_loss, helmholtz_loss, grad_norm,
                )

            # 早停
            if no_improve_count >= self.config.patience:
                LOGGER.info(
                    "PINNBeamSolver: 早停触发于 Epoch %d (patience=%d)",
                    epoch, self.config.patience,
                )
                break

        # 恢复最优参数
        self._nn.set_flat_parameters(best_params)

        # 获取初始场
        z_zeros = np.zeros(self.config.grid_size)
        X_init = self._normalize_input(self._x_coords, z_zeros)
        init_output = self._nn.predict(X_init)
        initial_field = init_output[:, 0] + 1j * init_output[:, 1]

        training_time = time.time() - start_time
        is_converged = no_improve_count < self.config.patience

        self._is_trained = True
        self._last_result = PINNResult(
            is_converged=is_converged,
            total_epochs=epoch,
            final_loss=best_loss,
            final_data_loss=self._data_loss_history[-1] if self._data_loss_history else 0.0,
            final_physics_loss=self._physics_loss_history[-1] if self._physics_loss_history else 0.0,
            loss_history=list(self._loss_history),
            data_loss_history=list(self._data_loss_history),
            physics_loss_history=list(self._physics_loss_history),
            initial_field=initial_field,
            training_time_s=training_time,
        )

        LOGGER.info(
            "PINNBeamSolver: 求解完成 (converged=%s, epochs=%d, "
            "loss=%.6e, time=%.2fs)",
            is_converged, epoch, best_loss, training_time,
        )

        return self._last_result

    def predict_field(self, z_distance: float) -> PINNBeamProfile:
        """预测给定传播距离处的光束场分布。

        Parameters
        ----------
        z_distance : float
            传播距离 (米)。z=0 为初始平面。

        Returns
        -------
        PINNBeamProfile
            预测的光束场分布，包含振幅、相位、强度等信息。
        """
        if not self._is_trained:
            LOGGER.warning("PINNBeamSolver: 尚未训练，预测结果可能不准确")

        z_distance = float(z_distance)
        z_arr = np.full(self.config.grid_size, z_distance)
        X_input = self._normalize_input(self._x_coords, z_arr)
        output = self._nn.predict(X_input)

        # 复振幅场
        complex_field = output[:, 0] + 1j * output[:, 1]
        amplitude = np.abs(complex_field)
        phase = np.angle(complex_field)
        intensity = amplitude ** 2

        # 归一化强度
        peak_intensity = float(np.max(intensity))
        if peak_intensity > 1e-30:
            intensity_norm = intensity / peak_intensity
        else:
            intensity_norm = intensity

        # 估算光束宽度 (1/e^2 半径)
        beam_width = self._estimate_beam_width(self._x_coords, intensity_norm)

        profile = PINNBeamProfile(
            x_coords=self._x_coords.copy(),
            z_distance=z_distance,
            amplitude=amplitude,
            phase=phase,
            intensity=intensity_norm,
            complex_field=complex_field,
            beam_width_m=beam_width,
            peak_intensity=peak_intensity,
        )

        LOGGER.debug(
            "PINNBeamSolver: 预测 z=%.4fm 处场分布 (beam_width=%.2fum)",
            z_distance, beam_width * 1e6,
        )

        return profile

    def _estimate_beam_width(self, x: np.ndarray, intensity: np.ndarray) -> float:
        """估算光束 1/e^2 半径。

        Parameters
        ----------
        x : np.ndarray
            横向坐标 (米)。
        intensity : np.ndarray
            归一化强度分布。

        Returns
        -------
        float
            光束 1/e^2 半径 (米)。
        """
        threshold = np.exp(-2.0)  # ~0.1353
        above = intensity >= threshold
        if not np.any(above):
            # 回退到二阶矩方法
            total = np.sum(intensity)
            if total < 1e-30:
                return self._x_max
            x_mean = np.sum(x * intensity) / total
            x2_mean = np.sum(x ** 2 * intensity) / total
            return float(np.sqrt(x2_mean - x_mean ** 2)) * 2.0

        indices = np.where(above)[0]
        x_left = x[indices[0]]
        x_right = x[indices[-1]]
        return float((x_right - x_left) / 2.0)

    def analyze(self) -> PINNReport:
        """生成 PINN 求解分析报告。

        Returns
        -------
        PINNReport
            包含训练统计、收敛分析和诊断建议的报告。
        """
        if not self._is_trained or self._last_result is None:
            LOGGER.warning("PINNBeamSolver: 尚未训练，分析结果可能不完整")
            # 返回默认报告
            return PINNReport(
                wavelength_m=self._wavelength,
                network_architecture=str(self.config.n_layers),
                total_parameters=self._nn.count_parameters(),
                is_converged=False,
                convergence_epoch=-1,
                final_loss=0.0,
                final_data_loss=0.0,
                final_fresnel_loss=0.0,
                final_helmholtz_loss=0.0,
                training_time_s=0.0,
                initial_beam_width_m=0.0,
                initial_peak_strehl=0.0,
                loss_reduction_ratio=1.0,
                gradient_norm_final=0.0,
                recommendations=["请先调用 solve() 方法进行训练"],
            )

        result = self._last_result

        # 网络信息
        arch_str = " -> ".join(str(n) for n in self.config.n_layers)
        n_params = self._nn.count_parameters()

        # 收敛信息
        is_converged = result.is_converged
        conv_epoch = result.total_epochs if is_converged else -1

        # 损失信息
        final_loss = result.final_loss
        final_data = result.final_data_loss
        final_fresnel = self._fresnel_loss_history[-1] if self._fresnel_loss_history else 0.0
        final_helmholtz = self._helmholtz_loss_history[-1] if self._helmholtz_loss_history else 0.0

        # 损失下降比率
        if self._loss_history and self._loss_history[0] > 1e-30:
            loss_ratio = self._loss_history[0] / max(final_loss, 1e-30)
        else:
            loss_ratio = 1.0

        # 梯度范数
        grad_norm = self._grad_norm_history[-1] if self._grad_norm_history else 0.0

        # 初始场分析
        initial_field = result.initial_field
        initial_intensity = np.abs(initial_field) ** 2
        peak_strehl = float(np.max(initial_intensity)) if np.max(initial_intensity) > 0 else 0.0

        # 归一化后估算光束宽度
        if np.max(initial_intensity) > 1e-30:
            norm_intensity = initial_intensity / np.max(initial_intensity)
        else:
            norm_intensity = initial_intensity
        beam_width = self._estimate_beam_width(self._x_coords, norm_intensity)

        # 生成诊断建议
        recommendations = self._generate_recommendations(
            is_converged, final_loss, final_data, final_fresnel,
            final_helmholtz, loss_ratio, grad_norm, result.total_epochs,
        )

        report = PINNReport(
            wavelength_m=self._wavelength,
            network_architecture=arch_str,
            total_parameters=n_params,
            is_converged=is_converged,
            convergence_epoch=conv_epoch,
            final_loss=final_loss,
            final_data_loss=final_data,
            final_fresnel_loss=final_fresnel,
            final_helmholtz_loss=final_helmholtz_loss,
            training_time_s=result.training_time_s,
            initial_beam_width_m=beam_width,
            initial_peak_strehl=peak_strehl,
            loss_reduction_ratio=loss_ratio,
            gradient_norm_final=grad_norm,
            recommendations=recommendations,
        )

        LOGGER.info(
            "PINNBeamSolver: 分析报告生成完成 (converged=%s, loss=%.2e, "
            "reduction=%.1fx, params=%d)",
            is_converged, final_loss, loss_ratio, n_params,
        )

        return report

    def _generate_recommendations(
        self,
        is_converged: bool,
        final_loss: float,
        data_loss: float,
        fresnel_loss: float,
        helmholtz_loss: float,
        loss_ratio: float,
        grad_norm: float,
        epochs: int,
    ) -> List[str]:
        """根据训练结果生成诊断建议。

        Parameters
        ----------
        is_converged : bool
            是否收敛。
        final_loss : float
            最终总损失。
        data_loss : float
            最终数据损失。
        fresnel_loss : float
            最终 Fresnel 损失。
        helmholtz_loss : float
            最终 Helmholtz 损失。
        loss_ratio : float
            损失下降比率。
        grad_norm : float
            最终梯度范数。
        epochs : int
            训练轮数。

        Returns
        -------
        List[str]
            诊断建议列表。
        """
        recs: List[str] = []

        if not is_converged:
            recs.append(
                "训练未收敛。建议增加 max_epochs 或调整学习率。"
                "可尝试使用余弦退火学习率调度。"
            )

        if loss_ratio < 2.0:
            recs.append(
                "损失下降不显著 (ratio < 2x)。可能需要增大学习率或"
                "检查测量数据质量。"
            )

        if grad_norm > 1e2:
            recs.append(
                "梯度范数较大，可能存在梯度爆炸风险。建议降低学习率"
                "或启用梯度裁剪。"
            )
        elif grad_norm < 1e-8:
            recs.append(
                "梯度范数极小，可能已陷入局部极小值或鞍点。"
                "建议增大学习率或使用不同的随机种子重新初始化。"
            )

        if fresnel_loss > data_loss * 5:
            recs.append(
                "Fresnel 物理损失远大于数据损失。建议增大 lambda_fresnel 权重"
                "或增加配点数量 n_collocation。"
            )

        if helmholtz_loss > data_loss * 10:
            recs.append(
                "Helmholtz 正则化损失较大。场分布可能不满足波动方程。"
                "建议增大 lambda_helmholtz 或检查坐标归一化。"
            )

        if epochs >= self.config.max_epochs * 0.95:
            recs.append(
                "训练使用了接近最大轮数。建议增大 max_epochs"
                "或降低 convergence_threshold。"
            )

        if len(recs) == 0:
            recs.append("训练状态良好，各项损失指标正常。")

        return recs

    def get_loss_history(self) -> Dict[str, List[float]]:
        """获取训练损失历史记录。

        Returns
        -------
        Dict[str, List[float]
            包含 'total', 'data', 'physics', 'fresnel', 'helmholtz',
            'grad_norm' 各项历史记录的字典。
        """
        return {
            "total": list(self._loss_history),
            "data": list(self._data_loss_history),
            "physics": list(self._physics_loss_history),
            "fresnel": list(self._fresnel_loss_history),
            "helmholtz": list(self._helmholtz_loss_history),
            "grad_norm": list(self._grad_norm_history),
        }

    @property
    def is_trained(self) -> bool:
        """bool: 求解器是否已完成训练。"""
        return self._is_trained

    @property
    def network(self) -> _SimpleNeuralNetwork:
        """_SimpleNeuralNetwork: 内部神经网络实例 (只读)。"""
        return self._nn

    @property
    def config_ref(self) -> PINNConfig:
        """PINNConfig: 当前配置 (只读引用)。"""
        return self.config
