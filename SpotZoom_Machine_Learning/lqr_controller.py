"""
LQR 线性二次型调节器 (v7.0)

基于现代控制理论的 LQR 最优控制器，用于光斑对准系统的电机伺服回路。
相比传统 PID，LQR 能自动处理多变量耦合，提供全局最优控制律。

灵感来源:
- python-control (https://github.com/python-control/python-control) — BSD-3
- MATLAB lqr() 函数
- Anderson & Moore, "Optimal Control: Linear Quadratic Methods" (1990)

算法原理:
  离散 LQR 通过求解离散代数 Riccati 方程 (DARE):
    P = A^T P A - A^T P B (R + B^T P B)^{-1} B^T P A + Q
  得到最优反馈增益:
    K = (R + B^T P B)^{-1} B^T P A
  控制律: u[k] = -K * x[k]

外部依赖: numpy, scipy (可选，用于 DARE 求解)
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class LQRConfig:
    """LQR 控制器配置。"""
    # 状态权重矩阵 Q (4x4): [x, y, vx, vy]
    q_position: float = 10.0       # 位置误差权重
    q_velocity: float = 1.0        # 速度权重
    q_cross_xy: float = 0.0        # X-Y 交叉权重

    # 控制权重矩阵 R (2x2): [ux, uy]
    r_control: float = 1.0         # 控制 effort 权重
    r_cross: float = 0.0           # UX-UY 交叉权重

    # 系统模型参数
    dt: float = 0.1                # 采样周期 (秒)
    damping: float = 0.8           # 速度衰减系数 (0-1)
    max_control: float = 500.0     # 最大控制输出 (步数)

    # 安全限制
    max_position_error: float = 500.0  # 最大位置误差 (像素)
    max_velocity: float = 100.0        # 最大速度 (像素/帧)


@dataclass
class LQRState:
    """LQR 控制器状态。"""
    x: np.ndarray = field(default_factory=lambda: np.zeros(4))
    K: Optional[np.ndarray] = None    # 反馈增益矩阵 (2x4)
    P: Optional[np.ndarray] = None    # Riccati 解 (4x4)
    converged: bool = False
    iteration_count: int = 0
    total_steps: int = 0


@dataclass
class LQRMetrics:
    """LQR 性能指标。"""
    cost_history: list = field(default_factory=list)
    position_error_history: list = field(default_factory=list)
    control_effort_history: list = field(default_factory=list)
    max_cost: float = 0.0
    avg_cost: float = 0.0
    total_control_effort: float = 0.0


class LQRController:
    """离散 LQR 最优控制器。

    将光斑对准问题建模为 LQR 问题:
    - 状态: x = [error_x, error_y, velocity_x, velocity_y]
    - 控制: u = [step_x, step_y]
    - 目标: 最小化 J = Σ (x^T Q x + u^T R u)

    使用示例:
        controller = LQRController()
        controller.compute_gain()  # 计算 K
        steps = controller.update(error_x=50.0, error_y=-30.0)
        # steps = (step_x, step_y)
    """

    def __init__(self, config: Optional[LQRConfig] = None):
        self._config = config or LQRConfig()
        self._state = LQRState()
        self._metrics = LQRMetrics()
        self._prev_error = np.zeros(2)

    @property
    def config(self) -> LQRConfig:
        return self._config

    @property
    def state(self) -> LQRState:
        return self._state

    @property
    def metrics(self) -> LQRMetrics:
        return self._metrics

    def _build_system_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """构建离散状态空间矩阵 A, B。

        状态模型:
            x[k+1] = A x[k] + B u[k]

        其中:
            x = [error_x, error_y, vel_x, vel_y]
            u = [step_x, step_y]

        Returns:
            (A, B): 状态转移矩阵和控制矩阵
        """
        dt = self._config.dt
        d = self._config.damping

        A = np.array([
            [1.0, 0.0, dt,   0.0],
            [0.0, 1.0, 0.0, dt  ],
            [0.0, 0.0, d,    0.0],
            [0.0, 0.0, 0.0, d   ],
        ], dtype=np.float64)

        B = np.array([
            [-1.0,  0.0],
            [ 0.0, -1.0],
            [ 0.0,  0.0],
            [ 0.0,  0.0],
        ], dtype=np.float64)

        return A, B

    def _build_weight_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """构建权重矩阵 Q 和 R。

        Returns:
            (Q, R): 状态权重和控制权重
        """
        qp = self._config.q_position
        qv = self._config.q_velocity
        qc = self._config.q_cross_xy

        Q = np.diag([qp, qp, qv, qv])
        if qc != 0.0:
            Q[0, 1] = qc
            Q[1, 0] = qc

        rc = self._config.r_control
        rx = self._config.r_cross

        R = np.diag([rc, rc])
        if rx != 0.0:
            R[0, 1] = rx
            R[1, 0] = rx

        return Q, R

    def compute_gain(
        self,
        max_iterations: int = 200,
        tolerance: float = 1e-10
    ) -> np.ndarray:
        """求解离散代数 Riccati 方程 (DARE)，计算最优反馈增益 K。

        使用迭代法求解:
            P_{k+1} = A^T P_k A - A^T P_k B (R + B^T P_k B)^{-1} B^T P_k A + Q

        Args:
            max_iterations: 最大迭代次数
            tolerance: 收敛容差

        Returns:
            K: 反馈增益矩阵 (2x4)
        """
        A, B = self._build_system_matrices()
        Q, R = self._build_weight_matrices()

        n = A.shape[0]
        P = Q.copy()  # 初始猜测

        for i in range(max_iterations):
            # 计算 Schur 补
            BtPB = B.T @ P @ B
            try:
                S = np.linalg.inv(R + BtPB)
            except np.linalg.LinAlgError:
                logger.warning("LQR: Riccati iteration singular matrix, using regularized inverse")
                S = np.linalg.inv(R + BtPB + 1e-8 * np.eye(R.shape[0]))

            APB = A.T @ P @ B
            P_new = A.T @ P @ A - APB @ S @ B.T @ P @ A + Q

            # 检查收敛
            diff = np.max(np.abs(P_new - P))
            P = P_new
            self._state.iteration_count = i + 1

            if diff < tolerance:
                logger.info(f"LQR: Riccati converged in {i + 1} iterations (diff={diff:.2e})")
                break
        else:
            logger.warning(f"LQR: Riccati did not converge in {max_iterations} iterations (diff={diff:.2e})")

        # 计算最优增益
        K = S @ B.T @ P @ A

        self._state.K = K
        self._state.P = P
        self._state.converged = (diff < tolerance)

        logger.info(f"LQR: Optimal gain K =\n{K}")
        return K

    def update(
        self,
        error_x: float,
        error_y: float,
        measured_vel_x: Optional[float] = None,
        measured_vel_y: Optional[float] = None,
        confidence: float = 1.0
    ) -> Tuple[float, float]:
        """计算 LQR 控制输出。

        Args:
            error_x: X 方向像素误差
            error_y: Y 方向像素误差
            measured_vel_x: 测量的 X 速度 (可选，否则用差分估计)
            measured_vel_y: 测量的 Y 速度 (可选)
            confidence: 检测置信度 (0-1)，低置信度时降低控制输出

        Returns:
            (step_x, step_y): 电机步数指令
        """
        # 估计速度
        if measured_vel_x is not None and measured_vel_y is not None:
            vel_x = measured_vel_x
            vel_y = measured_vel_y
        else:
            vel_x = error_x - self._prev_error[0]
            vel_y = error_y - self._prev_error[1]

        self._prev_error = np.array([error_x, error_y])

        # 构建状态向量
        x = np.array([error_x, error_y, vel_x, vel_y], dtype=np.float64)

        # 安全限幅
        x[0] = np.clip(x[0], -self._config.max_position_error, self._config.max_position_error)
        x[1] = np.clip(x[1], -self._config.max_position_error, self._config.max_position_error)
        x[2] = np.clip(x[2], -self._config.max_velocity, self._config.max_velocity)
        x[3] = np.clip(x[3], -self._config.max_velocity, self._config.max_velocity)

        self._state.x = x

        # 计算控制输出
        if self._state.K is None:
            self.compute_gain()

        u = -self._state.K @ x  # (2,)

        # 置信度缩放
        if confidence < 0.3:
            scale = confidence / 0.3
            u *= scale
            logger.debug(f"LQR: Low confidence ({confidence:.2f}), scaling control by {scale:.2f}")

        # 限幅
        u[0] = np.clip(u[0], -self._config.max_control, self._config.max_control)
        u[1] = np.clip(u[1], -self._config.max_control, self._config.max_control)

        # 记录指标
        cost = float(x @ self._build_weight_matrices()[0] @ x + u @ self._build_weight_matrices()[1] @ u)
        self._metrics.cost_history.append(cost)
        self._metrics.position_error_history.append(np.sqrt(error_x**2 + error_y**2))
        self._metrics.control_effort_history.append(np.sqrt(u[0]**2 + u[1]**2))
        self._metrics.total_control_effort += np.sqrt(u[0]**2 + u[1]**2)

        if self._metrics.cost_history:
            self._metrics.max_cost = max(self._metrics.max_cost, cost)
            self._metrics.avg_cost = np.mean(self._metrics.cost_history[-100:])

        self._state.total_steps += 1

        return float(u[0]), float(u[1])

    def reset(self):
        """重置控制器状态。"""
        self._state = LQRState()
        self._metrics = LQRMetrics()
        self._prev_error = np.zeros(2)
        logger.info("LQR: Controller reset")

    def get_gain_matrix(self) -> Optional[np.ndarray]:
        """获取当前反馈增益矩阵。"""
        return self._state.K

    def get_cost_summary(self) -> dict:
        """获取控制成本摘要。"""
        return {
            "total_steps": self._state.total_steps,
            "max_cost": self._metrics.max_cost,
            "avg_cost": self._metrics.avg_cost,
            "total_effort": self._metrics.total_control_effort,
            "converged": self._state.converged,
            "riccati_iterations": self._state.iteration_count,
        }
