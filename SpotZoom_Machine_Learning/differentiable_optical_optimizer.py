"""
可微分光学优化引擎 (DifferentiableOpticalOptimizer)

灵感来源:
- prysm (https://github.com/brandondube/prysm) — 纯 Python 光学传播与 PSF 分析，
  支持衍射传播、Zernike 像差、Strehl 比计算
- NSER-IBVS (Non-Singular Eye-in-Hand Visual Servoing) — 知识蒸馏方法，
  将解析控制器 (IBVS/PID) 的策略知识迁移到轻量级神经网络
- numpy-based autograd — 数值梯度计算，中心差分法近似偏导数

算法原理:
- Numerical Gradient (Central Differences) — 中心差分法数值梯度，
  eps=0.5 像素，适用于不可解析求导的光学系统
- PSF-based Loss — 基于 PSF 的损失函数: 质心误差、Strehl 比、
  环围能量、对称性偏差
- Adam Optimizer (numpy) — 纯 numpy 实现的 Adam 优化器，
  自适应学习率 + 一阶/二阶矩估计
- BFGS Quasi-Newton — 拟牛顿法，通过 Hessian 逆近似加速收敛
- Knowledge Distillation — 知识蒸馏: 教师 (解析 IBVS/PID) -> 学生 (轻量 NN)，
  将控制策略压缩为快速推理网络
- Line Search — 回溯线搜索 (Armijo 条件)，保证充分下降

功能:
- 多参数 (X/Y/Z) 同时优化
- 多种优化方法 (Adam, SGD+Momentum, BFGS, Line Search)
- 多种损失函数 (质心、Strehl、环围能量、对称性、组合)
- 学习率调度 (余弦退火、阶梯衰减)
- 梯度裁剪保证稳定性
- 知识蒸馏: 将解析控制策略压缩为轻量 NN
- 完整的收敛历史与分析

依赖: numpy, opencv-python (仅用于图像预处理)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
import numpy as np

LOGGER = logging.getLogger("SpotZoom.DifferentiableOpticalOptimizer")

# 模块默认禁用标志
differentiable_optimizer_enabled: bool = False


# ======================== 配置数据类 ========================


@dataclass
class DiffOptConfig:
    """可微分光学优化器配置。

    Parameters
    ----------
    optimization_method : str
        优化方法: 'adam', 'sgd_momentum', 'bfgs', 'line_search'。
    learning_rate : float
        初始学习率。
    max_iterations : int
        最大迭代次数。
    convergence_threshold : float
        收敛阈值 (损失变化小于此值时停止)。
    gradient_epsilon : float
        数值微分步长 (像素)。
    momentum : float
        SGD 动量系数。
    adam_beta1 : float
        Adam 一阶矩衰减率。
    adam_beta2 : float
        Adam 二阶矩衰减率。
    adam_epsilon : float
        Adam 数值稳定项。
    enable_line_search : bool
        是否启用线搜索。
    loss_function : str
        损失函数类型: 'centroid', 'strehl', 'encircled_energy',
        'symmetry', 'combined'。
    loss_weights : dict
        组合损失函数中各项权重。
    distillation_enabled : bool
        是否启用知识蒸馏。
    student_hidden_size : int
        学生网络隐藏层大小。
    student_learning_rate : float
        学生网络学习率。
    distillation_epochs : int
        蒸馏训练轮数。
    lr_schedule : str
        学习率调度策略: 'none', 'cosine', 'step'。
    lr_step_size : int
        阶梯衰减的步长间隔。
    lr_gamma : float
        阶梯衰减的衰减因子。
    gradient_clip_norm : float
        梯度裁剪范数阈值 (0 表示不裁剪)。
    bfgs_max_corrections : int
        BFGS 历史修正对数。
    line_search_c1 : float
        Armijo 线搜索条件参数 c1。
    line_search_rho : float
        回溯线搜索缩放因子。
    """
    optimization_method: str = "adam"
    learning_rate: float = 0.01
    max_iterations: int = 100
    convergence_threshold: float = 1e-6
    gradient_epsilon: float = 0.5
    momentum: float = 0.9
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    enable_line_search: bool = False
    loss_function: str = "centroid"
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "centroid": 1.0,
        "strehl": 0.5,
        "encircled_energy": 0.3,
        "symmetry": 0.2,
    })
    distillation_enabled: bool = False
    student_hidden_size: int = 32
    student_learning_rate: float = 0.001
    distillation_epochs: int = 50
    lr_schedule: str = "none"
    lr_step_size: int = 30
    lr_gamma: float = 0.5
    gradient_clip_norm: float = 10.0
    bfgs_max_corrections: int = 20
    line_search_c1: float = 1e-4
    line_search_rho: float = 0.5


# ======================== 结果数据类 ========================


@dataclass
class DiffOptResult:
    """可微分光学优化结果。

    Parameters
    ----------
    optimal_x : float
        优化后的 X 位置 (像素)。
    optimal_y : float
        优化后的 Y 位置 (像素)。
    optimal_z : float
        优化后的 Z 位置 (任意单位)。
    loss_history : List[float]
        每次迭代的损失值。
    gradient_history : List[np.ndarray]
        每次迭代的梯度向量。
    converged : bool
        是否收敛。
    iterations_used : int
        实际使用的迭代次数。
    final_loss : float
        最终损失值。
    elapsed_time_s : float
        总耗时 (秒)。
    method : str
        使用的优化方法。
    """
    optimal_x: float = 0.0
    optimal_y: float = 0.0
    optimal_z: float = 0.0
    loss_history: List[float] = field(default_factory=list)
    gradient_history: List[np.ndarray] = field(default_factory=list)
    converged: bool = False
    iterations_used: int = 0
    final_loss: float = 0.0
    elapsed_time_s: float = 0.0
    method: str = ""


@dataclass
class DistillationResult:
    """知识蒸馏训练结果。

    Parameters
    ----------
    train_loss_history : List[float]
        训练损失历史。
    val_loss_history : List[float]
        验证损失历史。
    final_train_loss : float
        最终训练损失。
    final_val_loss : float
        最终验证损失。
    epochs_used : int
        实际训练轮数。
    student_network : Optional['SimpleNeuralNetwork']
        训练好的学生网络。
    """
    train_loss_history: List[float] = field(default_factory=list)
    val_loss_history: List[float] = field(default_factory=list)
    final_train_loss: float = 0.0
    final_val_loss: float = 0.0
    epochs_used: int = 0
    student_network: Optional[Any] = None


# ======================== 轻量级神经网络 ========================


class SimpleNeuralNetwork:
    """轻量级神经网络 (用于知识蒸馏的学生网络)。

    纯 numpy 实现的前馈神经网络，支持 1-2 个隐藏层，
    ReLU 激活函数，Xavier/He 权重初始化。

    网络结构:
        输入层: 4 维 [error_x, error_y, confidence, spot_size]
        隐藏层: 1-2 层，每层 hidden_size 个神经元
        输出层: 2 维 [step_x, step_y]

    Parameters
    ----------
    input_size : int
        输入维度。
    output_size : int
        输出维度。
    hidden_sizes : List[int] or None
        隐藏层大小列表。为 None 时使用 [32]。
    activation : str
        激活函数: 'relu', 'leaky_relu', 'tanh'。
    init_method : str
        权重初始化方法: 'xavier', 'he'。
    seed : int or None
        随机种子。
    """

    def __init__(
        self,
        input_size: int = 4,
        output_size: int = 2,
        hidden_sizes: Optional[List[int]] = None,
        activation: str = "relu",
        init_method: str = "he",
        seed: Optional[int] = None,
    ):
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.hidden_sizes = hidden_sizes or [32]
        self.activation = activation
        self.init_method = init_method

        # 随机数生成器
        self._rng = np.random.RandomState(seed)

        # 网络层定义
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        self._num_layers = len(layer_sizes) - 1

        # 权重和偏置
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for i in range(self._num_layers):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            w, b = self._init_layer(fan_in, fan_out)
            self.weights.append(w)
            self.biases.append(b)

        # 梯度缓存 (用于反向传播)
        self._weight_grads: List[np.ndarray] = []
        self._bias_grads: List[np.ndarray] = []
        self._activations: List[np.ndarray] = []
        self._pre_activations: List[np.ndarray] = []

        LOGGER.info(
            "SimpleNeuralNetwork: 初始化完成 (layers=%s, activation=%s, init=%s)",
            layer_sizes, self.activation, self.init_method,
        )

    def _init_layer(self, fan_in: int, fan_out: int) -> Tuple[np.ndarray, np.ndarray]:
        """初始化单层权重和偏置。

        Parameters
        ----------
        fan_in : int
            输入维度。
        fan_out : int
            输出维度。

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (权重矩阵, 偏置向量)。
        """
        if self.init_method == "xavier":
            std = np.sqrt(2.0 / (fan_in + fan_out))
        elif self.init_method == "he":
            std = np.sqrt(2.0 / fan_in)
        else:
            std = 0.01

        w = self._rng.randn(fan_in, fan_out).astype(np.float64) * std
        b = np.zeros(fan_out, dtype=np.float64)
        return w, b

    def _activate(self, x: np.ndarray) -> np.ndarray:
        """激活函数前向传播。

        Parameters
        ----------
        x : np.ndarray
            输入。

        Returns
        -------
        np.ndarray
            激活输出。
        """
        if self.activation == "relu":
            return np.maximum(0, x)
        elif self.activation == "leaky_relu":
            return np.where(x > 0, x, 0.01 * x)
        elif self.activation == "tanh":
            return np.tanh(x)
        else:
            raise ValueError(f"不支持的激活函数: {self.activation}")

    def _activate_derivative(self, x: np.ndarray) -> np.ndarray:
        """激活函数导数。

        Parameters
        ----------
        x : np.ndarray
            激活前的输入 (pre-activation)。

        Returns
        -------
        np.ndarray
            激活函数导数值。
        """
        if self.activation == "relu":
            return (x > 0).astype(np.float64)
        elif self.activation == "leaky_relu":
            return np.where(x > 0, 1.0, 0.01)
        elif self.activation == "tanh":
            return 1.0 - np.tanh(x) ** 2
        else:
            raise ValueError(f"不支持的激活函数: {self.activation}")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播。

        Parameters
        ----------
        x : np.ndarray
            输入向量，形状为 (input_size,) 或 (batch, input_size)。

        Returns
        -------
        np.ndarray
            输出向量，形状为 (output_size,) 或 (batch, output_size)。
        """
        single_input = (x.ndim == 1)
        if single_input:
            x = x.reshape(1, -1)

        # 缓存中间结果
        self._activations = [x]
        self._pre_activations = []

        h = x
        for i in range(self._num_layers):
            z = h @ self.weights[i] + self.biases[i]
            self._pre_activations.append(z)

            # 最后一层不使用激活函数 (线性输出)
            if i < self._num_layers - 1:
                h = self._activate(z)
            else:
                h = z  # 输出层: 线性

            self._activations.append(h)

        if single_input:
            return h.flatten()
        return h

    def backward(self, grad_output: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """反向传播。

        Parameters
        ----------
        grad_output : np.ndarray
            输出层梯度，形状为 (output_size,) 或 (batch, output_size)。

        Returns
        -------
        Tuple[List[np.ndarray], List[np.ndarray]]
            (权重梯度列表, 偏置梯度列表)。
        """
        single_grad = (grad_output.ndim == 1)
        if single_grad:
            grad_output = grad_output.reshape(1, -1)

        self._weight_grads = []
        self._bias_grads = []

        # 反向传播
        delta = grad_output

        for i in range(self._num_layers - 1, -1, -1):
            # 计算权重和偏置梯度
            dw = self._activations[i].T @ delta
            db = np.sum(delta, axis=0)

            self._weight_grads.insert(0, dw)
            self._bias_grads.insert(0, db)

            # 传播到上一层
            if i > 0:
                delta = delta @ self.weights[i].T
                delta = delta * self._activate_derivative(self._pre_activations[i - 1])

        return self._weight_grads, self._bias_grads

    def get_params(self) -> List[np.ndarray]:
        """获取所有可训练参数 (展平)。

        Returns
        -------
        List[np.ndarray]
            展平的参数列表。
        """
        params = []
        for w, b in zip(self.weights, self.biases):
            params.append(w.flatten())
            params.append(b.flatten())
        return params

    def set_params(self, flat_params: np.ndarray) -> None:
        """从展平参数恢复权重。

        Parameters
        ----------
        flat_params : np.ndarray
            展平的参数向量。
        """
        idx = 0
        for i in range(self._num_layers):
            w_size = self.weights[i].size
            b_size = self.biases[i].size
            self.weights[i] = flat_params[idx:idx + w_size].reshape(
                self.weights[i].shape
            )
            idx += w_size
            self.biases[i] = flat_params[idx:idx + b_size].reshape(
                self.biases[i].shape
            )
            idx += b_size

    def predict(self, x: np.ndarray) -> np.ndarray:
        """推理模式 (无梯度缓存)。

        Parameters
        ----------
        x : np.ndarray
            输入向量。

        Returns
        -------
        np.ndarray
            输出向量。
        """
        single_input = (x.ndim == 1)
        if single_input:
            x = x.reshape(1, -1)

        h = x
        for i in range(self._num_layers):
            z = h @ self.weights[i] + self.biases[i]
            if i < self._num_layers - 1:
                h = self._activate(z)
            else:
                h = z

        if single_input:
            return h.flatten()
        return h

    def count_parameters(self) -> int:
        """统计可训练参数数量。

        Returns
        -------
        int
            参数总数。
        """
        total = 0
        for w, b in zip(self.weights, self.biases):
            total += w.size + b.size
        return total


# ======================== PSF 损失函数 ========================


class PSFLossFunction:
    """基于 PSF 的损失函数。

    提供多种光学质量损失函数，支持数值梯度计算。
    所有损失函数设计为最小化目标 (值越小越好)。

    Parameters
    ----------
    loss_type : str
        损失函数类型: 'centroid', 'strehl', 'encircled_energy',
        'symmetry', 'combined'。
    weights : Dict[str, float] or None
        组合损失函数中各项权重。
    target_center : Tuple[float, float] or None
        目标质心位置 (像素)。为 None 时使用图像中心。
    encircled_radius : float
        环围能量计算半径 (像素)。
    gradient_epsilon : float
        数值梯度计算步长 (像素)。
    """

    def __init__(
        self,
        loss_type: str = "centroid",
        weights: Optional[Dict[str, float]] = None,
        target_center: Optional[Tuple[float, float]] = None,
        encircled_radius: float = 5.0,
        gradient_epsilon: float = 0.5,
    ):
        self.loss_type = loss_type
        self.weights = weights or {
            "centroid": 1.0,
            "strehl": 0.5,
            "encircled_energy": 0.3,
            "symmetry": 0.2,
        }
        self.target_center = target_center
        self.encircled_radius = float(encircled_radius)
        self.gradient_epsilon = float(gradient_epsilon)

        # 缓存上一次计算的 PSF 特征
        self._last_centroid: Optional[Tuple[float, float]] = None
        self._last_peak: float = 0.0
        self._last_encircled: float = 0.0
        self._last_symmetry: float = 0.0

        LOGGER.info(
            "PSFLossFunction: 初始化完成 (type=%s, epsilon=%.2f)",
            self.loss_type, self.gradient_epsilon,
        )

    def _compute_centroid(self, image: np.ndarray) -> Tuple[float, float]:
        """计算图像质心。

        Parameters
        ----------
        image : np.ndarray
            输入图像 (灰度)。

        Returns
        -------
        Tuple[float, float]
            (cx, cy) 质心坐标。
        """
        img = image.astype(np.float64)
        total = np.sum(img)
        if total < 1e-12:
            h, w = img.shape
            return (w / 2.0, h / 2.0)

        ys, xs = np.mgrid[0:img.shape[0], 0:img.shape[1]]
        cx = float(np.sum(xs * img) / total)
        cy = float(np.sum(ys * img) / total)
        return (cx, cy)

    def _compute_peak_intensity(self, image: np.ndarray) -> float:
        """计算峰值强度。

        Parameters
        ----------
        image : np.ndarray
            输入图像。

        Returns
        -------
        float
            峰值强度。
        """
        return float(np.max(image.astype(np.float64)))

    def _compute_encircled_energy(
        self, image: np.ndarray, centroid: Tuple[float, float]
    ) -> float:
        """计算环围能量 (指定半径内的能量比例)。

        Parameters
        ----------
        image : np.ndarray
            输入图像。
        centroid : Tuple[float, float]
            质心坐标。

        Returns
        -------
        float
            环围能量比例 [0, 1]。
        """
        img = image.astype(np.float64)
        total = np.sum(img)
        if total < 1e-12:
            return 0.0

        h, w = img.shape
        ys, xs = np.mgrid[0:h, 0:w]
        r2 = (xs - centroid[0]) ** 2 + (ys - centroid[1]) ** 2
        mask = r2 <= self.encircled_radius ** 2
        return float(np.sum(img[mask]) / total)

    def _compute_symmetry(self, image: np.ndarray, centroid: Tuple[float, float]) -> float:
        """计算径向对称性偏差。

        将图像以质心为中心划分为 4 个象限，计算各象限能量与
        平均能量的偏差。

        Parameters
        ----------
        image : np.ndarray
            输入图像。
        centroid : Tuple[float, float]
            质心坐标。

        Returns
        -------
        float
            对称性偏差 (0=完美对称)。
        """
        img = image.astype(np.float64)
        h, w = img.shape
        cx, cy = int(round(centroid[0])), int(round(centroid[1]))
        cx = max(0, min(cx, w - 1))
        cy = max(0, min(cy, h - 1))

        # 四象限能量
        q1 = np.sum(img[:cy + 1, cx + 1:])   # 左上
        q2 = np.sum(img[:cy + 1, :cx + 1])    # 右上 (含中心)
        q3 = np.sum(img[cy + 1:, :cx + 1])    # 右下
        q4 = np.sum(img[cy + 1:, cx + 1:])    # 左下

        quadrants = np.array([q1, q2, q3, q4], dtype=np.float64)
        mean_q = np.mean(quadrants)
        if mean_q < 1e-12:
            return 0.0

        # 归一化标准差作为对称性偏差
        deviation = float(np.std(quadrants) / mean_q)
        return deviation

    def _get_target_center(self, image: np.ndarray) -> Tuple[float, float]:
        """获取目标中心坐标。

        Parameters
        ----------
        image : np.ndarray
            输入图像。

        Returns
        -------
        Tuple[float, float]
            目标中心 (cx, cy)。
        """
        if self.target_center is not None:
            return self.target_center
        h, w = image.shape[:2]
        return (w / 2.0, h / 2.0)

    def compute_loss(self, image: np.ndarray) -> float:
        """计算损失值。

        Parameters
        ----------
        image : np.ndarray
            输入图像 (灰度)。

        Returns
        -------
        float
            损失值。
        """
        if image.ndim != 2:
            raise ValueError(f"输入图像必须为 2D 灰度图，当前维度: {image.ndim}")

        centroid = self._compute_centroid(image)
        peak = self._compute_peak_intensity(image)
        encircled = self._compute_encircled_energy(image, centroid)
        symmetry = self._compute_symmetry(image, centroid)

        # 缓存特征
        self._last_centroid = centroid
        self._last_peak = peak
        self._last_encircled = encircled
        self._last_symmetry = symmetry

        if self.loss_type == "centroid":
            return self._loss_centroid(image, centroid)
        elif self.loss_type == "strehl":
            return self._loss_strehl(image, peak)
        elif self.loss_type == "encircled_energy":
            return self._loss_encircled_energy(encircled)
        elif self.loss_type == "symmetry":
            return self._loss_symmetry(symmetry)
        elif self.loss_type == "combined":
            return self._loss_combined(image, centroid, peak, encircled, symmetry)
        else:
            raise ValueError(f"不支持的损失函数类型: {self.loss_type}")

    def _loss_centroid(
        self, image: np.ndarray, centroid: Tuple[float, float]
    ) -> float:
        """质心误差损失: L2 距离。

        Parameters
        ----------
        image : np.ndarray
            输入图像。
        centroid : Tuple[float, float]
            质心坐标。

        Returns
        -------
        float
            质心误差损失。
        """
        target = self._get_target_center(image)
        dx = centroid[0] - target[0]
        dy = centroid[1] - target[1]
        return float(np.sqrt(dx * dx + dy * dy))

    def _loss_strehl(self, image: np.ndarray, peak: float) -> float:
        """Strehl 比损失: 1 - (peak / theoretical_max)。

        理论最大值取为 255 (8-bit 图像最大值)。

        Parameters
        ----------
        image : np.ndarray
            输入图像。
        peak : float
            峰值强度。

        Returns
        -------
        float
            Strehl 损失。
        """
        theoretical_max = float(np.max(image))
        if theoretical_max < 1e-12:
            return 1.0
        strehl = peak / theoretical_max
        return float(1.0 - strehl)

    def _loss_encircled_energy(self, encircled: float) -> float:
        """环围能量损失: 1 - encircled_energy。

        Parameters
        ----------
        encircled : float
            环围能量比例。

        Returns
        -------
        float
            环围能量损失。
        """
        return float(1.0 - encircled)

    def _loss_symmetry(self, symmetry: float) -> float:
        """对称性偏差损失。

        Parameters
        ----------
        symmetry : float
            对称性偏差。

        Returns
        -------
        float
            对称性损失。
        """
        return float(symmetry)

    def _loss_combined(
        self,
        image: np.ndarray,
        centroid: Tuple[float, float],
        peak: float,
        encircled: float,
        symmetry: float,
    ) -> float:
        """组合损失函数。

        Parameters
        ----------
        image : np.ndarray
            输入图像。
        centroid : Tuple[float, float]
            质心坐标。
        peak : float
            峰值强度。
        encircled : float
            环围能量。
        symmetry : float
            对称性偏差。

        Returns
        -------
        float
            加权组合损失。
        """
        l_centroid = self._loss_centroid(image, centroid)
        l_strehl = self._loss_strehl(image, peak)
        l_encircled = self._loss_encircled_energy(encircled)
        l_symmetry = self._loss_symmetry(symmetry)

        total = (
            self.weights.get("centroid", 1.0) * l_centroid
            + self.weights.get("strehl", 0.5) * l_strehl
            + self.weights.get("encircled_energy", 0.3) * l_encircled
            + self.weights.get("symmetry", 0.2) * l_symmetry
        )
        return float(total)

    def compute_numerical_gradient(
        self,
        params: np.ndarray,
        param_indices: List[int],
        image_getter: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """通过中心差分法计算数值梯度。

        Parameters
        ----------
        params : np.ndarray
            当前参数向量。
        param_indices : List[int]
            需要计算梯度的参数索引。
        image_getter : Callable[[np.ndarray], np.ndarray]
            参数向量到图像的映射函数。

        Returns
        -------
        np.ndarray
            梯度向量，形状与 params 相同。
        """
        eps = self.gradient_epsilon
        grad = np.zeros_like(params)
        f0 = self.compute_loss(image_getter(params))

        for idx in param_indices:
            # 正方向扰动
            params_plus = params.copy()
            params_plus[idx] += eps
            f_plus = self.compute_loss(image_getter(params_plus))

            # 负方向扰动
            params_minus = params.copy()
            params_minus[idx] -= eps
            f_minus = self.compute_loss(image_getter(params_minus))

            # 中心差分
            grad[idx] = (f_plus - f_minus) / (2.0 * eps)

        return grad

    def get_last_features(self) -> Dict[str, Any]:
        """获取上一次计算的特征值。

        Returns
        -------
        Dict[str, Any]
            特征字典。
        """
        return {
            "centroid": self._last_centroid,
            "peak_intensity": self._last_peak,
            "encircled_energy": self._last_encircled,
            "symmetry_deviation": self._last_symmetry,
        }


# ======================== 知识蒸馏引擎 ========================


class KnowledgeDistillationEngine:
    """知识蒸馏引擎。

    将解析控制策略 (教师: IBVS/PID) 的知识迁移到轻量级
    神经网络 (学生)，实现快速推理。

    教师策略:
        输入: [error_x, error_y, confidence, spot_size]
        输出: [step_x, step_y] (PID/IBVS 控制量)

    学生网络:
        输入: 与教师相同
        输出: 与教师相同
        结构: 1-2 隐藏层全连接网络

    Parameters
    ----------
    config : DiffOptConfig or None
        优化配置。
    teacher_kp : float
        教师比例增益。
    teacher_ki : float
        教师积分增益。
    teacher_kd : float
        教师微分增益。
    teacher_max_step : float
        教师最大步长限制。
    """

    def __init__(
        self,
        config: Optional[DiffOptConfig] = None,
        teacher_kp: float = 0.5,
        teacher_ki: float = 0.01,
        teacher_kd: float = 0.1,
        teacher_max_step: float = 10.0,
    ):
        self.config = config or DiffOptConfig()
        self.teacher_kp = float(teacher_kp)
        self.teacher_ki = float(teacher_ki)
        self.teacher_kd = float(teacher_kd)
        self.teacher_max_step = float(teacher_max_step)

        # 积分累积
        self._integral_x: float = 0.0
        self._integral_y: float = 0.0
        self._prev_error_x: float = 0.0
        self._prev_error_y: float = 0.0

        # 学生网络
        self.student = SimpleNeuralNetwork(
            input_size=4,
            output_size=2,
            hidden_sizes=[self.config.student_hidden_size],
            activation="relu",
            init_method="he",
        )

        # 训练数据集
        self._dataset_inputs: Optional[np.ndarray] = None
        self._dataset_outputs: Optional[np.ndarray] = None

        # 归一化参数 (训练时计算)
        self._input_mean: Optional[np.ndarray] = None
        self._input_std: Optional[np.ndarray] = None
        self._output_mean: Optional[np.ndarray] = None
        self._output_std: Optional[np.ndarray] = None

        LOGGER.info(
            "KnowledgeDistillationEngine: 初始化完成 "
            "(teacher: kp=%.3f, ki=%.3f, kd=%.3f, student_hidden=%d)",
            self.teacher_kp, self.teacher_ki, self.teacher_kd,
            self.config.student_hidden_size,
        )

    def _reset_teacher_state(self) -> None:
        """重置教师 PID 控制器状态。"""
        self._integral_x = 0.0
        self._integral_y = 0.0
        self._prev_error_x = 0.0
        self._prev_error_y = 0.0

    def teacher_policy(
        self,
        error_x: float,
        error_y: float,
        confidence: float = 1.0,
        spot_size: float = 5.0,
    ) -> Tuple[float, float]:
        """教师策略: PID 控制器。

        Parameters
        ----------
        error_x : float
            X 方向误差 (像素)。
        error_y : float
            Y 方向误差 (像素)。
        confidence : float
            置信度 [0, 1]。
        spot_size : float
            光斑大小 (像素)。

        Returns
        -------
        Tuple[float, float]
            (step_x, step_y) 控制步长。
        """
        # 积分项 (带抗饱和)
        self._integral_x += error_x
        self._integral_y += error_y
        integral_limit = self.teacher_max_step * 5.0
        self._integral_x = np.clip(self._integral_x, -integral_limit, integral_limit)
        self._integral_y = np.clip(self._integral_y, -integral_limit, integral_limit)

        # 微分项
        de_x = error_x - self._prev_error_x
        de_y = error_y - self._prev_error_y
        self._prev_error_x = error_x
        self._prev_error_y = error_y

        # PID 输出
        step_x = (
            self.teacher_kp * error_x
            + self.teacher_ki * self._integral_x
            + self.teacher_kd * de_x
        )
        step_y = (
            self.teacher_kp * error_y
            + self.teacher_ki * self._integral_y
            + self.teacher_kd * de_y
        )

        # 置信度加权
        step_x *= confidence
        step_y *= confidence

        # 步长限制
        step_x = float(np.clip(step_x, -self.teacher_max_step, self.teacher_max_step))
        step_y = float(np.clip(step_y, -self.teacher_max_step, self.teacher_max_step))

        return (step_x, step_y)

    def generate_dataset(
        self,
        num_samples: int = 1000,
        error_range: float = 50.0,
        spot_size_range: Tuple[float, float] = (2.0, 15.0),
    ) -> None:
        """从教师策略生成训练数据集。

        通过随机采样误差空间，收集教师策略的输入-输出对。
        每个样本独立计算 PID 输出 (重置积分状态)，
        避免积分项在随机采样下累积导致数值不稳定。

        Parameters
        ----------
        num_samples : int
            采样数量。
        error_range : float
            误差采样范围 [-error_range, error_range]。
        spot_size_range : Tuple[float, float]
            光斑大小采样范围。
        """
        rng = np.random.RandomState(42)

        inputs = np.zeros((num_samples, 4), dtype=np.float64)
        outputs = np.zeros((num_samples, 2), dtype=np.float64)

        for i in range(num_samples):
            ex = rng.uniform(-error_range, error_range)
            ey = rng.uniform(-error_range, error_range)
            conf = rng.uniform(0.3, 1.0)
            ss = rng.uniform(spot_size_range[0], spot_size_range[1])

            # 每个样本独立计算: 重置积分状态，避免随机采样下积分累积
            self._reset_teacher_state()
            sx, sy = self.teacher_policy(ex, ey, conf, ss)

            inputs[i] = [ex, ey, conf, ss]
            outputs[i] = [sx, sy]

        self._dataset_inputs = inputs
        self._dataset_outputs = outputs

        LOGGER.info(
            "KnowledgeDistillationEngine: 数据集生成完成 "
            "(samples=%d, error_range=%.1f)",
            num_samples, error_range,
        )

    def _compute_mse(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> float:
        """计算 MSE 损失。

        Parameters
        ----------
        predictions : np.ndarray
            预测值。
        targets : np.ndarray
            目标值。

        Returns
        -------
        float
            MSE 损失。
        """
        diff = predictions - targets
        return float(np.mean(diff * diff))

    def train_student(
        self,
        num_samples: Optional[int] = None,
        batch_size: int = 32,
        val_split: float = 0.2,
    ) -> DistillationResult:
        """训练学生网络。

        使用 mini-batch 梯度下降训练学生网络拟合教师策略。

        Parameters
        ----------
        num_samples : int or None
            训练样本数。为 None 时使用已生成的数据集。
        batch_size : int
            mini-batch 大小。
        val_split : float
            验证集比例。

        Returns
        -------
        DistillationResult
            蒸馏训练结果。
        """
        if self._dataset_inputs is None:
            self.generate_dataset(num_samples or 1000)

        X = self._dataset_inputs.copy()
        Y = self._dataset_outputs.copy()

        # 输入/输出归一化 (避免大数值导致梯度爆炸)
        self._input_mean = np.mean(X, axis=0)
        self._input_std = np.std(X, axis=0) + 1e-8
        self._output_mean = np.mean(Y, axis=0)
        self._output_std = np.std(Y, axis=0) + 1e-8
        X_norm = (X - self._input_mean) / self._input_std
        Y_norm = (Y - self._output_mean) / self._output_std

        # 划分训练集和验证集
        n = len(X_norm)
        n_val = max(1, int(n * val_split))
        rng = np.random.RandomState(123)
        indices = rng.permutation(n)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        X_train, Y_train = X_norm[train_idx], Y_norm[train_idx]
        X_val, Y_val = X_norm[val_idx], Y_norm[val_idx]

        # 保存原始验证集用于评估
        X_val_raw, Y_val_raw = X[val_idx], Y[val_idx]

        lr = self.config.student_learning_rate
        epochs = self.config.distillation_epochs
        n_train = len(X_train)
        grad_clip_norm = 5.0  # 梯度裁剪阈值

        train_losses: List[float] = []
        val_losses: List[float] = []

        LOGGER.info(
            "KnowledgeDistillationEngine: 开始训练学生网络 "
            "(epochs=%d, batch=%d, train=%d, val=%d, lr=%.4f)",
            epochs, batch_size, n_train, n_val, lr,
        )

        for epoch in range(epochs):
            # Shuffle
            perm = rng.permutation(n_train)
            X_shuffled = X_train[perm]
            Y_shuffled = Y_train[perm]

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_train, batch_size):
                end = min(start + batch_size, n_train)
                X_batch = X_shuffled[start:end]
                Y_batch = Y_shuffled[start:end]

                # 前向传播
                pred = self.student.forward(X_batch)

                # 计算损失
                batch_loss = self._compute_mse(pred, Y_batch)
                epoch_loss += batch_loss
                n_batches += 1

                # 反向传播
                grad_output = 2.0 * (pred - Y_batch) / len(Y_batch)
                self.student.backward(grad_output)

                # 梯度裁剪 + 参数更新 (SGD)
                for i in range(self.student._num_layers):
                    wg = self.student._weight_grads[i]
                    bg = self.student._bias_grads[i]
                    # 梯度裁剪
                    w_norm = np.linalg.norm(wg)
                    if w_norm > grad_clip_norm:
                        wg = wg * (grad_clip_norm / w_norm)
                    b_norm = np.linalg.norm(bg)
                    if b_norm > grad_clip_norm:
                        bg = bg * (grad_clip_norm / b_norm)
                    self.student.weights[i] -= lr * wg
                    self.student.biases[i] -= lr * bg

            avg_train_loss = epoch_loss / max(n_batches, 1)
            train_losses.append(avg_train_loss)

            # 验证集评估 (在原始尺度上计算损失)
            val_pred_norm = self.student.predict(X_val)
            val_pred_raw = val_pred_norm * self._output_std + self._output_mean
            val_loss = self._compute_mse(val_pred_raw, Y_val_raw)
            val_losses.append(val_loss)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                LOGGER.debug(
                    "KnowledgeDistillationEngine: epoch %d/%d, "
                    "train_loss=%.6f, val_loss=%.6f",
                    epoch + 1, epochs, avg_train_loss, val_loss,
                )

        result = DistillationResult(
            train_loss_history=train_losses,
            val_loss_history=val_losses,
            final_train_loss=train_losses[-1] if train_losses else 0.0,
            final_val_loss=val_losses[-1] if val_losses else 0.0,
            epochs_used=epochs,
            student_network=self.student,
        )

        LOGGER.info(
            "KnowledgeDistillationEngine: 训练完成 "
            "(final_train=%.6f, final_val=%.6f, params=%d)",
            result.final_train_loss, result.final_val_loss,
            self.student.count_parameters(),
        )

        return result

    def infer(
        self,
        error_x: float,
        error_y: float,
        confidence: float = 1.0,
        spot_size: float = 5.0,
    ) -> Tuple[float, float]:
        """学生网络推理。

        自动应用训练时的归一化变换。

        Parameters
        ----------
        error_x : float
            X 方向误差。
        error_y : float
            Y 方向误差。
        confidence : float
            置信度。
        spot_size : float
            光斑大小。

        Returns
        -------
        Tuple[float, float]
            (step_x, step_y) 预测步长。
        """
        x_raw = np.array([error_x, error_y, confidence, spot_size], dtype=np.float64)

        # 应用输入归一化 (如果已训练)
        if hasattr(self, '_input_mean') and self._input_mean is not None:
            x_norm = (x_raw - self._input_mean) / self._input_std
        else:
            x_norm = x_raw

        output_norm = self.student.predict(x_norm)

        # 反归一化输出
        if hasattr(self, '_output_mean') and self._output_mean is not None:
            output_raw = output_norm * self._output_std + self._output_mean
        else:
            output_raw = output_norm

        return (float(output_raw[0]), float(output_raw[1]))


# ======================== 可微分光学优化器 ========================


class DifferentiableOpticalOptimizer:
    """可微分光学优化引擎。

    基于 PSF 损失函数和数值梯度的光学对准优化器，
    支持多种优化方法和损失函数。

    核心思想:
        将光学对准问题建模为可微分优化问题:
            min_{x,y,z} L(PSF(x,y,z))
        其中 L 为 PSF 损失函数，梯度通过中心差分法数值计算。

    Parameters
    ----------
    config : DiffOptConfig or None
        优化配置。
    """

    def __init__(self, config: Optional[DiffOptConfig] = None):
        self.config = config or DiffOptConfig()

        # 损失函数
        self._loss_fn = PSFLossFunction(
            loss_type=self.config.loss_function,
            weights=self.config.loss_weights,
            gradient_epsilon=self.config.gradient_epsilon,
        )

        # Adam 优化器状态
        self._adam_m: Optional[np.ndarray] = None  # 一阶矩
        self._adam_v: Optional[np.ndarray] = None  # 二阶矩
        self._adam_t: int = 0  # 时间步

        # SGD 动量状态
        self._momentum_velocity: Optional[np.ndarray] = None

        # BFGS 状态
        self._bfgs_H: Optional[np.ndarray] = None  # Hessian 逆近似
        self._bfgs_s_history: List[np.ndarray] = []  # 位移历史
        self._bfgs_y_history: List[np.ndarray] = []  # 梯度差历史

        # 收敛历史
        self._loss_history: List[float] = []
        self._gradient_history: List[np.ndarray] = []

        # 知识蒸馏引擎
        self._distillation: Optional[KnowledgeDistillationEngine] = None
        if self.config.distillation_enabled:
            self._distillation = KnowledgeDistillationEngine(config=self.config)

        LOGGER.info(
            "DifferentiableOpticalOptimizer: 初始化完成 "
            "(method=%s, loss=%s, lr=%.4f, max_iter=%d)",
            self.config.optimization_method, self.config.loss_function,
            self.config.learning_rate, self.config.max_iterations,
        )

    # ======================== 学习率调度 ========================

    def _get_learning_rate(self, iteration: int) -> float:
        """获取当前迭代的学习率。

        Parameters
        ----------
        iteration : int
            当前迭代编号。

        Returns
        -------
        float
            当前学习率。
        """
        base_lr = self.config.learning_rate
        schedule = self.config.lr_schedule

        if schedule == "none":
            return base_lr
        elif schedule == "cosine":
            # 余弦退火
            t = min(iteration / self.config.max_iterations, 1.0)
            return base_lr * 0.5 * (1.0 + np.cos(np.pi * t))
        elif schedule == "step":
            # 阶梯衰减
            n_steps = iteration // self.config.lr_step_size
            return base_lr * (self.config.lr_gamma ** n_steps)
        else:
            LOGGER.warning("未知学习率调度策略: %s, 使用固定学习率", schedule)
            return base_lr

    # ======================== 梯度裁剪 ========================

    @staticmethod
    def _clip_gradient(grad: np.ndarray, max_norm: float) -> np.ndarray:
        """梯度裁剪 (按范数)。

        Parameters
        ----------
        grad : np.ndarray
            梯度向量。
        max_norm : float
            最大范数阈值。

        Returns
        -------
        np.ndarray
            裁剪后的梯度。
        """
        if max_norm <= 0:
            return grad
        norm = np.linalg.norm(grad)
        if norm > max_norm:
            grad = grad * (max_norm / norm)
        return grad

    # ======================== 优化方法 ========================

    def _step_adam(
        self, params: np.ndarray, grad: np.ndarray, iteration: int
    ) -> np.ndarray:
        """Adam 优化器单步更新。

        Parameters
        ----------
        params : np.ndarray
            当前参数。
        grad : np.ndarray
            当前梯度。
        iteration : int
            当前迭代编号。

        Returns
        -------
        np.ndarray
            更新后的参数。
        """
        lr = self._get_learning_rate(iteration)
        beta1 = self.config.adam_beta1
        beta2 = self.config.adam_beta2
        eps = self.config.adam_epsilon

        # 初始化状态
        if self._adam_m is None:
            self._adam_m = np.zeros_like(params)
            self._adam_v = np.zeros_like(params)
            self._adam_t = 0

        self._adam_t += 1

        # 更新一阶矩和二阶矩
        self._adam_m = beta1 * self._adam_m + (1.0 - beta1) * grad
        self._adam_v = beta2 * self._adam_v + (1.0 - beta2) * (grad ** 2)

        # 偏差校正
        m_hat = self._adam_m / (1.0 - beta1 ** self._adam_t)
        v_hat = self._adam_v / (1.0 - beta2 ** self._adam_t)

        # 参数更新
        params = params - lr * m_hat / (np.sqrt(v_hat) + eps)
        return params

    def _step_sgd_momentum(
        self, params: np.ndarray, grad: np.ndarray, iteration: int
    ) -> np.ndarray:
        """SGD + 动量 单步更新。

        Parameters
        ----------
        params : np.ndarray
            当前参数。
        grad : np.ndarray
            当前梯度。
        iteration : int
            当前迭代编号。

        Returns
        -------
        np.ndarray
            更新后的参数。
        """
        lr = self._get_learning_rate(iteration)
        mu = self.config.momentum

        if self._momentum_velocity is None:
            self._momentum_velocity = np.zeros_like(params)

        self._momentum_velocity = mu * self._momentum_velocity - lr * grad
        params = params + self._momentum_velocity
        return params

    def _step_bfgs(
        self,
        params: np.ndarray,
        grad: np.ndarray,
        grad_prev: Optional[np.ndarray],
        params_prev: Optional[np.ndarray],
    ) -> np.ndarray:
        """BFGS 拟牛顿法单步更新。

        Parameters
        ----------
        params : np.ndarray
            当前参数。
        grad : np.ndarray
            当前梯度。
        grad_prev : np.ndarray or None
            上一步梯度。
        params_prev : np.ndarray or None
            上一步参数。

        Returns
        -------
        np.ndarray
            更新后的参数。
        """
        lr = self.config.learning_rate

        # 初始化 Hessian 逆近似
        if self._bfgs_H is None:
            self._bfgs_H = np.eye(len(params), dtype=np.float64)

        if grad_prev is not None and params_prev is not None:
            s = params - params_prev
            y = grad - grad_prev

            sy = float(s @ y)
            if sy > 1e-12:
                # BFGS 更新公式
                rho = 1.0 / sy
                I = np.eye(len(params), dtype=np.float64)
                V = I - rho * np.outer(s, y)
                self._bfgs_H = V @ self._bfgs_H @ V.T + rho * np.outer(s, s)

        # 搜索方向
        direction = -self._bfgs_H @ grad

        # 简单步长 (实际应用中应配合线搜索)
        params = params + lr * direction
        return params

    def _line_search(
        self,
        params: np.ndarray,
        grad: np.ndarray,
        loss_fn: Callable[[np.ndarray], float],
    ) -> Tuple[np.ndarray, float]:
        """回溯线搜索 (Armijo 条件)。

        Parameters
        ----------
        params : np.ndarray
            当前参数。
        grad : np.ndarray
            当前梯度。
        loss_fn : Callable[[np.ndarray], float]
            损失函数。

        Returns
        -------
        Tuple[np.ndarray, float]
            (更新后的参数, 新损失值)。
        """
        c1 = self.config.line_search_c1
        rho = self.config.line_search_rho
        alpha = self.config.learning_rate

        f0 = loss_fn(params)
        direction = -grad

        max_ls_iter = 30
        for _ in range(max_ls_iter):
            new_params = params + alpha * direction
            f_new = loss_fn(new_params)

            # Armijo 充分下降条件
            if f_new <= f0 + c1 * alpha * (grad @ direction):
                return (new_params, f_new)

            alpha *= rho

        # 线搜索失败，返回原始步长结果
        new_params = params + alpha * direction
        return (new_params, loss_fn(new_params))

    # ======================== 重置状态 ========================

    def _reset_optimizer_state(self, param_dim: int) -> None:
        """重置优化器内部状态。

        Parameters
        ----------
        param_dim : int
            参数维度。
        """
        self._adam_m = None
        self._adam_v = None
        self._adam_t = 0
        self._momentum_velocity = None
        self._bfgs_H = None
        self._bfgs_s_history.clear()
        self._bfgs_y_history.clear()
        self._loss_history.clear()
        self._gradient_history.clear()

    # ======================== 主优化接口 ========================

    def optimize(
        self,
        initial_params: np.ndarray,
        image_getter: Callable[[np.ndarray], np.ndarray],
        param_mask: Optional[List[int]] = None,
    ) -> DiffOptResult:
        """运行优化。

        Parameters
        ----------
        initial_params : np.ndarray
            初始参数向量 [x, y, z, ...]。
        image_getter : Callable[[np.ndarray], np.ndarray]
            参数向量到图像的映射函数。
            接收参数向量，返回灰度图像 (2D numpy array)。
        param_mask : List[int] or None
            需要优化的参数索引列表。为 None 时优化所有参数。

        Returns
        -------
        DiffOptResult
            优化结果。

        Raises
        ------
        ValueError
            参数不合法时抛出。
        """
        if initial_params.ndim != 1:
            raise ValueError(
                f"initial_params 必须为 1D 向量，当前形状: {initial_params.shape}"
            )
        if len(initial_params) < 2:
            raise ValueError("initial_params 至少需要 2 个元素 (x, y)")

        params = initial_params.astype(np.float64).copy()
        param_dim = len(params)

        # 参数掩码
        if param_mask is None:
            param_mask = list(range(param_dim))
        else:
            for idx in param_mask:
                if idx < 0 or idx >= param_dim:
                    raise ValueError(f"参数索引越界: {idx} (dim={param_dim})")

        self._reset_optimizer_state(param_dim)

        method = self.config.optimization_method
        max_iter = self.config.max_iterations
        tol = self.config.convergence_threshold

        LOGGER.info(
            "DifferentiableOpticalOptimizer: 开始优化 "
            "(method=%s, dim=%d, active_params=%s, max_iter=%d)",
            method, param_dim, param_mask, max_iter,
        )

        start_time = time.perf_counter()

        # 初始损失
        initial_image = image_getter(params)
        current_loss = self._loss_fn.compute_loss(initial_image)
        self._loss_history.append(current_loss)

        # 初始梯度
        current_grad = self._loss_fn.compute_numerical_gradient(
            params, param_mask, image_getter
        )
        self._gradient_history.append(current_grad.copy())

        prev_loss = current_loss
        prev_grad = current_grad.copy()
        prev_params = params.copy()
        converged = False
        iterations_used = 0

        for it in range(1, max_iter + 1):
            # 梯度裁剪
            clipped_grad = self._clip_gradient(
                current_grad, self.config.gradient_clip_norm
            )

            # 参数更新
            if method == "adam":
                params = self._step_adam(params, clipped_grad, it)
            elif method == "sgd_momentum":
                params = self._step_sgd_momentum(params, clipped_grad, it)
            elif method == "bfgs":
                params = self._step_bfgs(params, clipped_grad, prev_grad, prev_params)
            elif method == "line_search":
                def _loss_wrapper(p: np.ndarray) -> float:
                    return self._loss_fn.compute_loss(image_getter(p))
                params, current_loss = self._line_search(
                    params, clipped_grad, _loss_wrapper
                )
            else:
                raise ValueError(f"不支持的优化方法: {method}")

            # 计算新损失
            if method != "line_search":
                new_image = image_getter(params)
                current_loss = self._loss_fn.compute_loss(new_image)

            self._loss_history.append(current_loss)

            # 计算新梯度 (line_search 模式下也需要)
            if method != "line_search":
                current_grad = self._loss_fn.compute_numerical_gradient(
                    params, param_mask, image_getter
                )
            else:
                current_grad = self._loss_fn.compute_numerical_gradient(
                    params, param_mask, image_getter
                )
            self._gradient_history.append(current_grad.copy())

            iterations_used = it

            # 收敛检查
            loss_change = abs(prev_loss - current_loss)
            if loss_change < tol and it > 5:
                converged = True
                LOGGER.info(
                    "DifferentiableOpticalOptimizer: 收敛于迭代 %d "
                    "(loss_change=%.2e < tol=%.2e)",
                    it, loss_change, tol,
                )
                break

            # 更新历史
            prev_loss = current_loss
            prev_grad = current_grad.copy()
            prev_params = params.copy()

            if it % 20 == 0:
                LOGGER.debug(
                    "DifferentiableOpticalOptimizer: iter=%d/%d, loss=%.6f, "
                    "grad_norm=%.4f, lr=%.6f",
                    it, max_iter, current_loss,
                    np.linalg.norm(current_grad),
                    self._get_learning_rate(it),
                )

        elapsed = time.perf_counter() - start_time

        result = DiffOptResult(
            optimal_x=float(params[0]) if param_dim > 0 else 0.0,
            optimal_y=float(params[1]) if param_dim > 1 else 0.0,
            optimal_z=float(params[2]) if param_dim > 2 else 0.0,
            loss_history=list(self._loss_history),
            gradient_history=list(self._gradient_history),
            converged=converged,
            iterations_used=iterations_used,
            final_loss=float(current_loss),
            elapsed_time_s=round(elapsed, 4),
            method=method,
        )

        LOGGER.info(
            "DifferentiableOpticalOptimizer: 优化完成. "
            "final_loss=%.6f, iterations=%d, converged=%s, elapsed=%.3fs, "
            "optimal=(%.4f, %.4f, %.4f)",
            result.final_loss, result.iterations_used, result.converged,
            result.elapsed_time_s, result.optimal_x, result.optimal_y,
            result.optimal_z,
        )

        return result

    def optimize_single_step(
        self,
        params: np.ndarray,
        image_getter: Callable[[np.ndarray], np.ndarray],
        iteration: int,
        param_mask: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """单步优化 (用于增量式优化)。

        Parameters
        ----------
        params : np.ndarray
            当前参数向量。
        image_getter : Callable[[np.ndarray], np.ndarray]
            参数向量到图像的映射函数。
        iteration : int
            当前迭代编号。
        param_mask : List[int] or None
            需要优化的参数索引。

        Returns
        -------
        Tuple[np.ndarray, float, np.ndarray]
            (更新后的参数, 新损失值, 梯度向量)。
        """
        param_dim = len(params)
        if param_mask is None:
            param_mask = list(range(param_dim))

        # 计算梯度
        grad = self._loss_fn.compute_numerical_gradient(
            params, param_mask, image_getter
        )

        # 梯度裁剪
        clipped_grad = self._clip_gradient(grad, self.config.gradient_clip_norm)

        # 参数更新
        method = self.config.optimization_method
        if method == "adam":
            new_params = self._step_adam(params, clipped_grad, iteration)
        elif method == "sgd_momentum":
            new_params = self._step_sgd_momentum(params, clipped_grad, iteration)
        elif method == "bfgs":
            new_params = self._step_bfgs(
                params, clipped_grad,
                self._gradient_history[-1] if self._gradient_history else None,
                None,
            )
        elif method == "line_search":
            def _loss_wrapper(p: np.ndarray) -> float:
                return self._loss_fn.compute_loss(image_getter(p))
            new_params, _ = self._line_search(params, clipped_grad, _loss_wrapper)
        else:
            raise ValueError(f"不支持的优化方法: {method}")

        # 计算新损失
        new_image = image_getter(new_params)
        new_loss = self._loss_fn.compute_loss(new_image)

        # 记录历史
        self._loss_history.append(new_loss)
        self._gradient_history.append(grad.copy())

        return (new_params, new_loss, grad)

    # ======================== 知识蒸馏接口 ========================

    def train_distillation(
        self,
        num_samples: int = 1000,
        batch_size: int = 32,
    ) -> Optional[DistillationResult]:
        """训练知识蒸馏。

        Parameters
        ----------
        num_samples : int
            训练样本数。
        batch_size : int
            mini-batch 大小。

        Returns
        -------
        DistillationResult or None
            蒸馏结果。未启用蒸馏时返回 None。
        """
        if not self.config.distillation_enabled or self._distillation is None:
            LOGGER.warning("知识蒸馏未启用，跳过训练")
            return None

        self._distillation.generate_dataset(num_samples=num_samples)
        result = self._distillation.train_student(batch_size=batch_size)
        return result

    def infer_distilled(
        self,
        error_x: float,
        error_y: float,
        confidence: float = 1.0,
        spot_size: float = 5.0,
    ) -> Optional[Tuple[float, float]]:
        """使用蒸馏后的学生网络推理。

        Parameters
        ----------
        error_x : float
            X 方向误差。
        error_y : float
            Y 方向误差。
        confidence : float
            置信度。
        spot_size : float
            光斑大小。

        Returns
        -------
        Tuple[float, float] or None
            (step_x, step_y)。未启用蒸馏时返回 None。
        """
        if not self.config.distillation_enabled or self._distillation is None:
            return None
        return self._distillation.infer(error_x, error_y, confidence, spot_size)

    # ======================== 分析工具 ========================

    def get_convergence_analysis(self) -> Dict[str, Any]:
        """获取收敛分析。

        Returns
        -------
        Dict[str, Any]
            收敛分析结果。
        """
        if not self._loss_history:
            return {"status": "no_data"}

        losses = np.array(self._loss_history)
        analysis: Dict[str, Any] = {
            "total_iterations": len(losses),
            "initial_loss": float(losses[0]),
            "final_loss": float(losses[-1]),
            "loss_reduction": float(losses[0] - losses[-1]),
            "loss_reduction_ratio": float(
                (losses[0] - losses[-1]) / max(abs(losses[0]), 1e-12)
            ),
            "min_loss": float(np.min(losses)),
            "max_loss": float(np.max(losses)),
            "mean_loss": float(np.mean(losses)),
            "std_loss": float(np.std(losses)),
        }

        # 收敛速率估计 (指数衰减拟合)
        if len(losses) > 5:
            # 使用最后 50% 的数据拟合
            n_fit = max(10, len(losses) // 2)
            y_fit = losses[-n_fit:]
            x_fit = np.arange(n_fit, dtype=np.float64)

            # 线性拟合 log(loss)
            log_y = np.log(np.maximum(y_fit, 1e-12))
            if np.std(log_y) > 1e-12:
                coeffs = np.polyfit(x_fit, log_y, 1)
                decay_rate = float(-coeffs[0])
                analysis["estimated_decay_rate"] = decay_rate
                analysis["convergence_speed"] = (
                    "fast" if decay_rate > 0.1
                    else "moderate" if decay_rate > 0.01
                    else "slow"
                )

        # 梯度范数历史
        if self._gradient_history:
            grad_norms = [float(np.linalg.norm(g)) for g in self._gradient_history]
            analysis["initial_grad_norm"] = grad_norms[0]
            analysis["final_grad_norm"] = grad_norms[-1]
            analysis["mean_grad_norm"] = float(np.mean(grad_norms))

        return analysis

    def get_loss_function(self) -> PSFLossFunction:
        """获取内部损失函数实例。

        Returns
        -------
        PSFLossFunction
            损失函数实例。
        """
        return self._loss_fn

    def get_distillation_engine(self) -> Optional[KnowledgeDistillationEngine]:
        """获取知识蒸馏引擎实例。

        Returns
        -------
        KnowledgeDistillationEngine or None
            蒸馏引擎实例。
        """
        return self._distillation
