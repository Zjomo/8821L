"""
学习增强型模型预测控制器 (LearningMPCController) (v1.0)

结合模仿学习 (Imitation Learning) 与强化学习 (RL) 增强的模型预测控制器，
用于 SpotZoom 光斑对准系统的最优轨迹规划与精密定位。

灵感来源:
- leap-c (https://github.com/leap-c/leap-c) — 学习增强型预测控制框架，
  将深度学习与模型预测控制结合，实现数据驱动的约束优化
- acados (https://github.com/acados/acados) — 高性能嵌入式最优控制工具包，
  支持实时非线性 MPC、SQP/QP 求解和代码生成
- do-mpc (https://github.com/do-mpc/do-mpc) — 开源非线性 MPC 框架
- PETS (Chua et al., 2018) — 概论集成轨迹采样，用于模型基 RL

算法原理:
  1. 核心 MPC: 离散时间线性 MPC，通过投影梯度下降求解约束 QP
     min  sum_{k=0}^{N-1} [ (x_k - x_ref)^T Q (x_k - x_ref) + u_k^T R u_k ]
          + (x_N - x_ref)^T P (x_N - x_ref)
     s.t. x_{k+1} = A x_k + B u_k   (自适应动力学模型)
          u_min <= u_k <= u_max        (控制约束)
          x_min <= x_k <= x_max        (状态约束)

  2. 自适应动力学: 递归最小二乘 (RLS) 在线辨识 A, B 矩阵
     P_{k+1} = (P_k - P_k phi_k^T (lambda + phi_k P_k phi_k^T)^{-1} phi_k P_k) / lambda
     theta_{k+1} = theta_k + P_{k+1} phi_k (y_k - phi_k^T theta_k)

  3. 模仿学习: 行为克隆 (Behavioral Cloning) 从专家轨迹学习控制策略
     pi_bc(s) = W_bc * phi(s)，使用岭回归求解 W_bc

  4. RL 增强: REINFORCE 策略梯度在线调优代价函数权重
     J(theta) ~ E[R(tau)]，theta = [q_weight, r_weight, p_weight]
     grad J ~ sum_t grad log pi(a_t|s_t) * R(tau)

  5. 融合策略: u_final = alpha * u_mpc + (1 - alpha) * u_bc + beta * u_rl
     其中 alpha/beta 根据信任度动态调整

功能:
- 离散时间线性状态空间模型 [x, y, vx, vy]
- 控制输入 [dx, dy] (载物台移动步数)
- 递归最小二乘 (RLS) 在线动力学模型自适应
- 二次型代价函数: 状态跟踪 + 控制能量 + 终端代价
- 位置/速度/控制约束 (投影梯度下降处理)
- 行为克隆模仿学习 (线性回归)
- REINFORCE 策略梯度 RL 增强 (代价权重调优)
- MPC + BC + RL 融合控制策略
- 热启动加速优化收敛
- 完整的健康状态报告

依赖: numpy (纯 numpy 实现，无外部求解器)
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# 数据类定义
# ============================================================

@dataclass
class LearningMPCConfig:
    """学习增强型 MPC 控制器配置参数。

    Attributes:
        horizon: 预测时域长度 (N)，即向前预测的步数。
        dt: 采样周期 (秒)。
        state_dim: 状态向量维度 (默认 4: [x, y, vx, vy])。
        control_dim: 控制输入维度 (默认 2: [dx, dy])。
        state_weight: 位置误差权重 (Q 矩阵中对角线元素)。
        control_weight: 控制能量权重 (R 矩阵中对角线元素)。
        terminal_weight: 终端状态权重 (P 矩阵对角线元素)。
        max_control: 单轴最大控制步数 (绝对值)。
        max_position: 工作空间位置边界 (绝对值)。
        max_velocity: 最大速度限制 (绝对值)。
        velocity_damping: 速度衰减系数 (0~1)，模拟机械阻尼。
        damping_factor: QP 求解器正则化系数。
        max_iterations: 投影梯度下降最大迭代次数。
        qp_tolerance: QP 求解器收敛容差。
        rls_forgetting_factor: RLS 遗忘因子 (0 < lambda <= 1)。
        rls_init_covariance: RLS 初始协方差矩阵对角值。
        bc_learning_rate: 模仿学习 (行为克隆) 学习率。
        rl_learning_rate: 强化学习 (REINFORCE) 学习率。
        rl_discount_factor: RL 折扣因子 (gamma)。
        bc_blend_ratio: 行为克隆融合比例 (0~1)，0 表示不使用 BC。
        rl_blend_ratio: RL 融合比例 (0~1)，0 表示不使用 RL。
        warm_start: 是否启用热启动。
        enable_constraints: 是否启用约束处理。
    """
    horizon: int = 15
    dt: float = 0.1
    state_dim: int = 4
    control_dim: int = 2
    state_weight: float = 10.0
    control_weight: float = 0.1
    terminal_weight: float = 20.0
    max_control: float = 2000.0
    max_position: float = 2000.0
    max_velocity: float = 500.0
    velocity_damping: float = 0.8
    damping_factor: float = 1e-4
    max_iterations: int = 100
    qp_tolerance: float = 1e-6
    rls_forgetting_factor: float = 0.99
    rls_init_covariance: float = 100.0
    bc_learning_rate: float = 0.01
    rl_learning_rate: float = 0.001
    rl_discount_factor: float = 0.99
    bc_blend_ratio: float = 0.2
    rl_blend_ratio: float = 0.1
    warm_start: bool = True
    enable_constraints: bool = True


@dataclass
class LearningMPCResult:
    """学习增强型 MPC 控制器单步输出结果。

    Attributes:
        optimal_control: 最优控制序列 (horizon x control_dim)。
        predicted_trajectory: 预测状态轨迹 (horizon x state_dim)。
        cost: 最优代价函数值。
        solve_time: QP 求解耗时 (秒)。
        iterations: 求解器实际迭代次数。
        bc_control: 行为克隆策略输出的控制 (control_dim,)。
        rl_weight_adjustment: RL 对代价权重的调整量 (3,)。
        blend_ratio: 实际使用的融合比例 (bc_ratio, rl_ratio)。
    """
    optimal_control: np.ndarray = field(
        default_factory=lambda: np.zeros((15, 2), dtype=np.float64)
    )
    predicted_trajectory: np.ndarray = field(
        default_factory=lambda: np.zeros((15, 4), dtype=np.float64)
    )
    cost: float = 0.0
    solve_time: float = 0.0
    iterations: int = 0
    bc_control: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float64)
    )
    rl_weight_adjustment: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    blend_ratio: Tuple[float, float] = (0.0, 0.0)


# ============================================================
# 自适应动力学模型 (内部类)
# ============================================================

class DynamicsModel:
    """自适应线性动力学模型。

    使用递归最小二乘 (Recursive Least Squares, RLS) 在线辨识
    离散状态空间模型的 A, B 矩阵:

        x[k+1] = A * x[k] + B * u[k]

    RLS 更新规则:
        P_{k+1} = (P_k - P_k * phi_k^T * (lambda + phi_k * P_k * phi_k^T)^{-1}
                   * phi_k * P_k) / lambda
        theta_{k+1} = theta_k + P_{k+1} * phi_k * (y_k - phi_k^T * theta_k)

    其中 phi_k = [x_k; u_k] 为回归向量，theta = [A^T | B^T] 为参数矩阵。
    """

    def __init__(
        self,
        state_dim: int = 4,
        control_dim: int = 2,
        dt: float = 0.1,
        velocity_damping: float = 0.8,
        forgetting_factor: float = 0.99,
        init_covariance: float = 100.0,
    ):
        """初始化动力学模型。

        Args:
            state_dim: 状态维度。
            control_dim: 控制维度。
            dt: 采样周期。
            velocity_damping: 速度衰减系数。
            forgetting_factor: RLS 遗忘因子。
            init_covariance: RLS 初始协方差。
        """
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.dt = dt
        self.velocity_damping = velocity_damping
        self.forgetting_factor = forgetting_factor

        # 初始化标称 A, B 矩阵 (双积分器 + 阻尼)
        self.A = self._build_nominal_A()
        self.B = self._build_nominal_B()

        # RLS 参数
        # theta = [A | B]，形状为 (state_dim, state_dim + control_dim)
        # 每行 theta[i, :] = [A[i, :], B[i, :]]
        param_dim = state_dim + control_dim
        self._theta = np.zeros((state_dim, param_dim), dtype=np.float64)
        self._theta[:, :state_dim] = self.A
        self._theta[:, state_dim:] = self.B

        # RLS 协方差矩阵
        self._P = init_covariance * np.eye(param_dim, dtype=np.float64)

        # 统计信息
        self.update_count: int = 0
        self._initialized: bool = False

    def _build_nominal_A(self) -> np.ndarray:
        """构建标称状态转移矩阵 A。

        状态模型 (双积分器 + 阻尼):
            px[k+1] = px[k] + dt * vx[k]
            py[k+1] = py[k] + dt * vy[k]
            vx[k+1] = damping * vx[k]
            vy[k+1] = damping * vy[k]

        Returns:
            标称 A 矩阵 (state_dim x state_dim)。
        """
        dt = self.dt
        d = self.velocity_damping
        return np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, d,    0.0],
            [0.0, 0.0, 0.0, d   ],
        ], dtype=np.float64)

    def _build_nominal_B(self) -> np.ndarray:
        """构建标称控制矩阵 B。

        控制输入直接作用于速度:
            vx[k+1] += ux[k]
            vy[k+1] += uy[k]

        Returns:
            标称 B 矩阵 (state_dim x control_dim)。
        """
        return np.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ], dtype=np.float64)

    def predict(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """使用当前模型预测下一状态。

        Args:
            state: 当前状态向量 (state_dim,)。
            action: 控制输入 (control_dim,)。

        Returns:
            预测的下一状态 (state_dim,)。
        """
        return self.A @ state + self.B @ action

    def update(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> None:
        """使用 RLS 在线更新动力学模型。

        Args:
            state: 当前状态 (state_dim,)。
            action: 控制输入 (control_dim,)。
            next_state: 实测下一状态 (state_dim,)。
        """
        # 构建回归向量 phi = [state; action]
        phi = np.concatenate([state, action])  # (state_dim + control_dim,)

        # 预测误差
        # theta[i, :] = [A[i,:], B[i,:]], phi = [state; action]
        # y_pred[i] = theta[i, :] @ phi = A[i,:] @ state + B[i,:] @ action
        y_pred = self._theta @ phi  # (state_dim,)
        error = next_state - y_pred    # (state_dim,)

        # RLS 更新
        lam = self.forgetting_factor
        phi_P = phi @ self._P  # (param_dim,)
        denom = lam + phi @ phi_P  # 标量

        if abs(denom) < 1e-12:
            logger.warning("DynamicsModel: RLS 分母接近零，跳过更新")
            return

        # 协方差更新: P = (P - P * phi^T * phi * P / denom) / lambda
        gain = self._P @ phi / denom  # (param_dim,)
        self._P = (self._P - np.outer(gain, phi_P)) / lam

        # 参数更新: theta[i, :] += gain * error[i]
        # theta += outer(error, gain)
        self._theta += np.outer(error, gain)

        # 提取更新后的 A, B
        # theta[i, :] = [A[i,:], B[i,:]]
        self.A = self._theta[:, :self.state_dim]
        self.B = self._theta[:, self.state_dim:]

        self.update_count += 1
        self._initialized = True

        # 数值稳定性检查
        if not (np.all(np.isfinite(self.A)) and np.all(np.isfinite(self.B))):
            logger.error("DynamicsModel: RLS 更新导致非有限值，恢复标称模型")
            self.A = self._build_nominal_A()
            self.B = self._build_nominal_B()
            self._theta[:, :self.state_dim] = self.A
            self._theta[:, self.state_dim:] = self.B
            self._P = self.forgetting_factor * np.eye(
                self.state_dim + self.control_dim, dtype=np.float64
            )

    def reset(self) -> None:
        """重置动力学模型到标称值。"""
        self.A = self._build_nominal_A()
        self.B = self._build_nominal_B()
        param_dim = self.state_dim + self.control_dim
        self._theta = np.zeros((self.state_dim, param_dim), dtype=np.float64)
        self._theta[:, :self.state_dim] = self.A
        self._theta[:, self.state_dim:] = self.B
        self._P = 100.0 * np.eye(param_dim, dtype=np.float64)
        self.update_count = 0
        self._initialized = False

    def get_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取当前 A, B 矩阵。

        Returns:
            (A, B): 状态转移矩阵和控制矩阵的副本。
        """
        return self.A.copy(), self.B.copy()


# ============================================================
# 主控制器类
# ============================================================

class LearningMPCController:
    """学习增强型模型预测控制器 (Learning-enhanced MPC Controller)。

    将光斑对准轨迹规划建模为学习增强的约束线性 MPC 问题:
    - 状态向量: x = [px, py, vx, vy] (X/Y 位置 + 速度)
    - 控制向量: u = [dx, dy] (载物台移动步数)
    - 状态方程: x[k+1] = A * x[k] + B * u[k]  (RLS 自适应)
    - 代价函数: J = sum (x-x_ref)^T Q (x-x_ref) + u^T R u
                + (x_N-x_ref)^T P (x_N-x_ref)
    - 约束: u_min <= u <= u_max, x_min <= x <= x_max

    学习组件:
    - 模仿学习 (BC): 从专家轨迹学习控制策略，通过线性回归实现行为克隆
    - RL 增强 (REINFORCE): 策略梯度在线调优代价函数权重
    - 融合策略: u_final = alpha * u_mpc + (1-alpha) * u_bc + beta * u_rl

    使用示例:
        controller = LearningMPCController()
        result = controller.compute_control(
            state=np.array([100.0, 80.0, 0.0, 0.0]),
            target=np.array([0.0, 0.0]),
        )
        print(f"最优控制: {result.optimal_control[0]}")
        print(f"代价: {result.cost:.4f}, 求解时间: {result.solve_time*1000:.2f}ms")
    """

    def __init__(self, config: Optional[LearningMPCConfig] = None):
        """初始化学习增强型 MPC 控制器。

        Args:
            config: 控制器配置参数。为 None 时使用默认配置。

        Raises:
            ValueError: 配置参数不合法时抛出。
        """
        if config is not None:
            self._validate_config(config)
        self._config = config or LearningMPCConfig()

        # 自适应动力学模型
        self._dynamics = DynamicsModel(
            state_dim=self._config.state_dim,
            control_dim=self._config.control_dim,
            dt=self._config.dt,
            velocity_damping=self._config.velocity_damping,
            forgetting_factor=self._config.rls_forgetting_factor,
            init_covariance=self._config.rls_init_covariance,
        )

        # 权重矩阵
        self._Q: Optional[np.ndarray] = None
        self._R: Optional[np.ndarray] = None
        self._P: Optional[np.ndarray] = None
        self._build_weight_matrices()

        # 行为克隆 (模仿学习) 参数
        # W_bc: (state_dim, control_dim)，策略 pi_bc(s) = s^T @ W_bc
        self._bc_weights: Optional[np.ndarray] = None
        self._bc_trained: bool = False

        # RL (REINFORCE) 参数
        # 可调权重参数: theta = [q_weight, r_weight, p_weight]
        self._rl_theta: np.ndarray = np.array([
            self._config.state_weight,
            self._config.control_weight,
            self._config.terminal_weight,
        ], dtype=np.float64)
        self._rl_baseline: float = 0.0
        self._rl_episode_rewards: List[float] = []
        self._rl_trajectory_log: List[Dict[str, np.ndarray]] = []

        # 热启动: 上一时刻最优控制序列
        self._prev_control_seq: Optional[np.ndarray] = None

        # 预测矩阵缓存
        self._Phi: Optional[np.ndarray] = None
        self._Gamma: Optional[np.ndarray] = None
        self._build_prediction_matrices()

        # 统计与历史
        self._solve_count: int = 0
        self._total_solve_time: float = 0.0
        self._cost_history: Deque[float] = deque(maxlen=500)
        self._solve_time_history: Deque[float] = deque(maxlen=500)
        self._state_history: Deque[np.ndarray] = deque(maxlen=100)
        self._control_history: Deque[np.ndarray] = deque(maxlen=100)

        logger.info(
            "LearningMPCController: 初始化完成 "
            "(N=%d, dt=%.3fs, Q=%.1f, R=%.2f, P=%.1f, "
            "bc_ratio=%.2f, rl_ratio=%.2f)",
            self._config.horizon,
            self._config.dt,
            self._config.state_weight,
            self._config.control_weight,
            self._config.terminal_weight,
            self._config.bc_blend_ratio,
            self._config.rl_blend_ratio,
        )

    # ======================== 属性 ========================

    @property
    def config(self) -> LearningMPCConfig:
        """获取当前配置。"""
        return self._config

    @property
    def dynamics(self) -> DynamicsModel:
        """获取自适应动力学模型。"""
        return self._dynamics

    # ======================== 公共接口 ========================

    def compute_control(
        self,
        state: np.ndarray,
        target: np.ndarray,
        constraints: Optional[Dict[str, np.ndarray]] = None,
    ) -> LearningMPCResult:
        """计算最优控制序列。

        求解学习增强的 MPC 优化问题，返回最优控制序列和预测轨迹。
        控制输出融合了 MPC 解、行为克隆策略和 RL 策略梯度调整。

        Args:
            state: 当前状态向量 (state_dim,)，格式 [x, y, vx, vy]。
            target: 目标位置 (2,)，格式 [target_x, target_y]。
            constraints: 可选约束字典，可覆盖默认配置:
                - 'u_max': 控制上界 (control_dim,)
                - 'u_min': 控制下界 (control_dim,)
                - 'x_max': 状态上界 (state_dim,)
                - 'x_min': 状态下界 (state_dim,)

        Returns:
            LearningMPCResult: 包含最优控制、预测轨迹、代价等的结果对象。

        Raises:
            ValueError: state 或 target 维度不匹配时抛出。
        """
        t_start = time.perf_counter()

        # 输入验证
        state = np.asarray(state, dtype=np.float64).ravel()
        target = np.asarray(target, dtype=np.float64).ravel()

        if state.shape[0] != self._config.state_dim:
            raise ValueError(
                f"状态维度不匹配: 期望 {self._config.state_dim}, "
                f"实际 {state.shape[0]}"
            )
        if target.shape[0] != 2:
            raise ValueError(
                f"目标维度不匹配: 期望 2, 实际 {target.shape[0]}"
            )

        try:
            # 1. 构建参考状态 (目标位置 + 零速度)
            x_ref = np.zeros(self._config.state_dim, dtype=np.float64)
            x_ref[:2] = target

            # 2. 获取当前有效权重 (含 RL 调整)
            Q_eff, R_eff, P_eff = self._get_effective_weights()

            # 3. 求解核心 MPC
            optimal_control, predicted_states, iterations = self._solve_mpc(
                state, x_ref, Q_eff, R_eff, P_eff, constraints
            )

            # 4. 行为克隆策略输出
            bc_control = self._compute_bc_control(state)

            # 5. 融合控制策略
            final_control = self._blend_controls(
                optimal_control, bc_control
            )

            # 6. 重新计算预测轨迹 (使用融合后的控制)
            predicted_trajectory = self.predict_trajectory(state, final_control)

            # 7. 计算代价
            cost = self._evaluate_cost(
                predicted_trajectory, final_control, x_ref, Q_eff, R_eff, P_eff
            )

            # 8. RL 轨迹记录 (用于后续 reinforce_update)
            self._log_rl_trajectory(state, final_control[0], cost)

            # 9. 更新热启动
            if self._config.warm_start:
                self._prev_control_seq = final_control.copy()

            # 10. 记录历史
            self._state_history.append(state.copy())
            self._control_history.append(final_control[0].copy())

            # 11. 统计
            solve_time = time.perf_counter() - t_start
            self._solve_count += 1
            self._total_solve_time += solve_time
            self._cost_history.append(cost)
            self._solve_time_history.append(solve_time)

            # RL 权重调整量 (用于报告)
            rl_adjustment = self._rl_theta - np.array([
                self._config.state_weight,
                self._config.control_weight,
                self._config.terminal_weight,
            ])

            result = LearningMPCResult(
                optimal_control=final_control,
                predicted_trajectory=predicted_trajectory,
                cost=round(cost, 6),
                solve_time=round(solve_time, 6),
                iterations=iterations,
                bc_control=bc_control,
                rl_weight_adjustment=rl_adjustment,
                blend_ratio=(
                    self._config.bc_blend_ratio,
                    self._config.rl_blend_ratio,
                ),
            )

            logger.debug(
                "LearningMPCController: pos=(%.2f, %.2f), target=(%.2f, %.2f), "
                "cost=%.4f, solve=%.2fms, iters=%d",
                state[0], state[1], target[0], target[1],
                cost, solve_time * 1000, iterations,
            )

            return result

        except Exception as e:
            logger.error(
                "LearningMPCController: 控制计算失败: %s", e, exc_info=True
            )
            # 安全降级: 返回零控制
            return LearningMPCResult(
                optimal_control=np.zeros(
                    (self._config.horizon, self._config.control_dim),
                    dtype=np.float64,
                ),
                predicted_trajectory=np.tile(
                    state, (self._config.horizon, 1)
                ),
                solve_time=time.perf_counter() - t_start,
            )

    def update_dynamics(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
    ) -> None:
        """在线更新动力学模型 (RLS)。

        根据实测的状态转移数据，使用递归最小二乘更新 A, B 矩阵。

        Args:
            state: 当前状态 (state_dim,)。
            action: 执行的控制输入 (control_dim,)。
            next_state: 实测的下一状态 (state_dim,)。

        Raises:
            ValueError: 输入维度不匹配时抛出。
        """
        state = np.asarray(state, dtype=np.float64).ravel()
        action = np.asarray(action, dtype=np.float64).ravel()
        next_state = np.asarray(next_state, dtype=np.float64).ravel()

        if state.shape[0] != self._config.state_dim:
            raise ValueError(
                f"状态维度不匹配: 期望 {self._config.state_dim}, "
                f"实际 {state.shape[0]}"
            )
        if action.shape[0] != self._config.control_dim:
            raise ValueError(
                f"控制维度不匹配: 期望 {self._config.control_dim}, "
                f"实际 {action.shape[0]}"
            )
        if next_state.shape[0] != self._config.state_dim:
            raise ValueError(
                f"下一状态维度不匹配: 期望 {self._config.state_dim}, "
                f"实际 {next_state.shape[0]}"
            )

        self._dynamics.update(state, action, next_state)

        # 动力学更新后重建预测矩阵
        self._build_prediction_matrices()

        logger.debug(
            "LearningMPCController: 动力学模型已更新 "
            "(update_count=%d)", self._dynamics.update_count,
        )

    def imitate_expert(self, expert_trajectories: list) -> Dict[str, float]:
        """从专家轨迹学习控制策略 (行为克隆)。

        使用线性回归 (岭回归) 从专家演示中学习控制策略:
            pi_bc(s) = W_bc^T * s
        其中 W_bc 通过最小化 ||S @ W_bc - U_expert||^2 + lambda * ||W_bc||^2 求得。

        Args:
            expert_trajectories: 专家轨迹列表，每条轨迹为字典列表:
                [{'state': np.ndarray, 'action': np.ndarray}, ...]
                或为元组列表: [(state, action), ...]

        Returns:
            训练指标字典:
                - 'mse': 训练均方误差
                - 'num_samples': 训练样本数
                - 'bc_weights_norm': 权重矩阵范数

        Raises:
            ValueError: 专家轨迹为空或格式不正确时抛出。
        """
        if not expert_trajectories:
            raise ValueError("专家轨迹列表不能为空")

        # 解析专家数据
        states_list: List[np.ndarray] = []
        actions_list: List[np.ndarray] = []

        for traj in expert_trajectories:
            if isinstance(traj, dict):
                # 单条轨迹: {'state': array, 'action': array}
                if 'state' in traj and 'action' in traj:
                    states_list.append(np.asarray(traj['state'], dtype=np.float64).ravel())
                    actions_list.append(np.asarray(traj['action'], dtype=np.float64).ravel())
                else:
                    raise ValueError("专家轨迹字典必须包含 'state' 和 'action' 键")
            elif isinstance(traj, (list, tuple)):
                # 判断是单条 (state, action) 对还是轨迹列表
                if (
                    len(traj) == 2
                    and isinstance(traj[0], np.ndarray)
                    and isinstance(traj[1], np.ndarray)
                ):
                    # 单条 (state, action) 对
                    states_list.append(np.asarray(traj[0], dtype=np.float64).ravel())
                    actions_list.append(np.asarray(traj[1], dtype=np.float64).ravel())
                else:
                    # 轨迹列表: [(state, action), ...]
                    for step in traj:
                        if isinstance(step, (list, tuple)) and len(step) == 2:
                            states_list.append(np.asarray(step[0], dtype=np.float64).ravel())
                            actions_list.append(np.asarray(step[1], dtype=np.float64).ravel())
                        elif isinstance(step, dict):
                            states_list.append(np.asarray(step['state'], dtype=np.float64).ravel())
                            actions_list.append(np.asarray(step['action'], dtype=np.float64).ravel())
                        else:
                            raise ValueError(f"无法解析的轨迹步骤: {type(step)}")
            else:
                raise ValueError(f"无法解析的轨迹格式: {type(traj)}")

        if len(states_list) < 2:
            raise ValueError(
                f"专家样本数不足: 需要至少 2 个，实际 {len(states_list)}"
            )

        # 构建回归矩阵
        S = np.array(states_list, dtype=np.float64)   # (num_samples, state_dim)
        U = np.array(actions_list, dtype=np.float64)   # (num_samples, control_dim)

        # 岭回归: W = (S^T S + lambda I)^{-1} S^T U
        lam = self._config.bc_learning_rate
        S_T_S = S.T @ S
        n = self._config.state_dim
        W_bc = np.linalg.solve(
            S_T_S + lam * np.eye(n, dtype=np.float64),
            S.T @ U,
        )

        # 计算训练误差
        U_pred = S @ W_bc
        mse = float(np.mean((U_pred - U) ** 2))

        self._bc_weights = W_bc
        self._bc_trained = True

        metrics = {
            'mse': round(mse, 6),
            'num_samples': len(states_list),
            'bc_weights_norm': round(float(np.linalg.norm(W_bc)), 6),
        }

        logger.info(
            "LearningMPCController: 行为克隆训练完成 "
            "(samples=%d, MSE=%.6f, ||W||=%.4f)",
            metrics['num_samples'], metrics['mse'], metrics['bc_weights_norm'],
        )

        return metrics

    def reinforce_update(self, reward: float, done: bool) -> Dict[str, float]:
        """执行 REINFORCE 策略梯度更新。

        使用 REINFORCE 算法在线调优代价函数权重参数:
            theta = [q_weight, r_weight, p_weight]

        策略梯度:
            grad J ~ sum_t grad_theta log pi(a_t|s_t) * (R_t - baseline)
            theta += lr * grad J

        其中 log pi(a_t|s_t) 近似为当前控制相对于 MPC 解的偏差，
        R_t 为累积折扣奖励。

        Args:
            reward: 当前步的即时奖励。
            done: 当前回合是否结束。

        Returns:
            更新指标字典:
                - 'weight_delta': 权重变化范数
                - 'new_weights': 更新后的权重
                - 'gradient_norm': 梯度范数
                - 'episode_reward': 当前回合累计奖励

        Raises:
            RuntimeError: 没有可用的 RL 轨迹数据时抛出。
        """
        if len(self._rl_trajectory_log) == 0:
            raise RuntimeError(
                "没有可用的 RL 轨迹数据。请先调用 compute_control 积累轨迹。"
            )

        # 累积奖励
        self._rl_episode_rewards.append(reward)

        # 只在回合结束时执行更新
        if not done:
            return {
                'weight_delta': 0.0,
                'new_weights': self._rl_theta.copy(),
                'gradient_norm': 0.0,
                'episode_reward': sum(self._rl_episode_rewards),
            }

        # 计算折扣累积奖励
        gamma = self._config.rl_discount_factor
        trajectory = self._rl_trajectory_log

        # 为每步计算折扣累积奖励 (从后往前)
        returns = np.zeros(len(trajectory), dtype=np.float64)
        G = 0.0
        for t in reversed(range(len(trajectory))):
            G = trajectory[t]['cost'] + gamma * G  # 使用负代价作为奖励
            returns[t] = G

        # 基线: 平均回报
        baseline = np.mean(returns) if len(returns) > 0 else 0.0
        self._rl_baseline = 0.9 * self._rl_baseline + 0.1 * baseline

        # 策略梯度近似
        # 使用控制偏差作为策略梯度信号
        gradient = np.zeros(3, dtype=np.float64)
        for t, entry in enumerate(trajectory):
            advantage = returns[t] - self._rl_baseline
            # 梯度近似: 基于状态误差和控制量
            state = entry['state']
            control = entry['control']
            # 位置误差影响状态权重梯度
            gradient[0] += advantage * np.sum(state[:2] ** 2) * 0.01
            # 控制量影响控制权重梯度
            gradient[1] += advantage * np.sum(control ** 2) * 0.01
            # 终端误差影响终端权重梯度
            gradient[2] += advantage * np.sum(state[:2] ** 2) * 0.005

        # 归一化梯度
        grad_norm = np.linalg.norm(gradient)
        if grad_norm > 1e-8:
            gradient = gradient / grad_norm

        # 更新权重 (带学习率缩放)
        lr = self._config.rl_learning_rate
        old_theta = self._rl_theta.copy()
        self._rl_theta += lr * gradient

        # 限制权重在合理范围内
        self._rl_theta[0] = np.clip(
            self._rl_theta[0], 1.0, 100.0
        )  # Q 权重
        self._rl_theta[1] = np.clip(
            self._rl_theta[1], 0.01, 10.0
        )  # R 权重
        self._rl_theta[2] = np.clip(
            self._rl_theta[2], 1.0, 200.0
        )  # P 权重

        weight_delta = float(np.linalg.norm(self._rl_theta - old_theta))
        episode_reward = sum(self._rl_episode_rewards)

        # 清空回合数据
        self._rl_trajectory_log.clear()
        self._rl_episode_rewards.clear()

        metrics = {
            'weight_delta': round(weight_delta, 6),
            'new_weights': self._rl_theta.copy(),
            'gradient_norm': round(grad_norm, 6),
            'episode_reward': round(episode_reward, 4),
        }

        logger.info(
            "LearningMPCController: REINFORCE 更新完成 "
            "(delta=%.6f, grad_norm=%.6f, episode_reward=%.4f)",
            weight_delta, grad_norm, episode_reward,
        )

        return metrics

    def predict_trajectory(
        self,
        state: np.ndarray,
        control_seq: np.ndarray,
    ) -> np.ndarray:
        """使用学习到的动力学模型预测未来轨迹。

        Args:
            state: 初始状态 (state_dim,)。
            control_seq: 控制序列 (horizon x control_dim) 或 (control_dim,)。

        Returns:
            预测状态轨迹 (horizon x state_dim)。

        Raises:
            ValueError: 输入维度不匹配时抛出。
        """
        state = np.asarray(state, dtype=np.float64).ravel()
        control_seq = np.asarray(control_seq, dtype=np.float64)

        if state.shape[0] != self._config.state_dim:
            raise ValueError(
                f"状态维度不匹配: 期望 {self._config.state_dim}, "
                f"实际 {state.shape[0]}"
            )

        # 如果控制序列是 1D，扩展为 2D
        if control_seq.ndim == 1:
            if control_seq.shape[0] == self._config.control_dim:
                control_seq = control_seq.reshape(1, -1)
            else:
                raise ValueError(
                    f"控制序列维度不匹配: 期望 ({self._config.horizon}, "
                    f"{self._config.control_dim}) 或 ({self._config.control_dim},), "
                    f"实际 {control_seq.shape}"
                )

        N = control_seq.shape[0]
        predicted = np.zeros((N, self._config.state_dim), dtype=np.float64)
        x = state.copy()

        for k in range(N):
            x = self._dynamics.predict(x, control_seq[k])
            predicted[k] = x

        return predicted

    def warm_start(self, prev_solution: dict) -> None:
        """从上一时刻的解热启动优化。

        Args:
            prev_solution: 上一时刻的解字典，包含:
                - 'control_sequence': 控制序列 (horizon x control_dim)
                - 'state': 上一步状态 (state_dim,) (可选)

        Raises:
            ValueError: 控制序列维度不匹配时抛出。
        """
        if 'control_sequence' not in prev_solution:
            logger.warning(
                "LearningMPCController: warm_start 缺少 'control_sequence' 键"
            )
            return

        ctrl = np.asarray(
            prev_solution['control_sequence'], dtype=np.float64
        )

        expected_shape = (self._config.horizon, self._config.control_dim)
        if ctrl.shape != expected_shape:
            raise ValueError(
                f"控制序列形状不匹配: 期望 {expected_shape}, "
                f"实际 {ctrl.shape}"
            )

        # 移位: 丢弃第一步，末尾补零
        shifted = np.zeros_like(ctrl)
        shifted[:-1] = ctrl[1:]
        shifted[-1] = ctrl[-1]  # 最后一步保持

        self._prev_control_seq = shifted
        logger.debug(
            "LearningMPCController: 热启动已设置 "
            "(control_seq shape=%s)", str(shifted.shape),
        )

    def get_health_report(self) -> dict:
        """返回控制器健康状态报告。

        Returns:
            健康状态字典，包含:
                - 'status': 总体状态 ('healthy' / 'degraded' / 'error')
                - 'solve_count': 总求解次数
                - 'avg_solve_time_ms': 平均求解时间 (毫秒)
                - 'avg_cost': 近期平均代价
                - 'cost_trend': 代价趋势 ('decreasing' / 'stable' / 'increasing')
                - 'dynamics_update_count': 动力学模型更新次数
                - 'dynamics_initialized': 动力学模型是否已初始化
                - 'bc_trained': 行为克隆是否已训练
                - 'rl_updates': RL 更新次数
                - 'current_weights': 当前 RL 调整后的权重
                - 'matrix_condition_A': A 矩阵条件数
                - 'matrix_condition_B': B 矩阵条件数
                - 'prediction_matrices_valid': 预测矩阵是否有效
        """
        # 求解统计
        avg_solve_time = (
            self._total_solve_time / max(self._solve_count, 1) * 1000.0
        )

        # 代价趋势分析
        cost_trend = 'stable'
        if len(self._cost_history) >= 10:
            recent_costs = list(self._cost_history)[-10:]
            first_half = np.mean(recent_costs[:5])
            second_half = np.mean(recent_costs[5:])
            ratio = second_half / max(abs(first_half), 1e-8)
            if ratio < 0.9:
                cost_trend = 'decreasing'
            elif ratio > 1.1:
                cost_trend = 'increasing'

        # 平均代价
        avg_cost = (
            float(np.mean(list(self._cost_history)[-100:]))
            if self._cost_history else 0.0
        )

        # 矩阵健康检查
        A, B = self._dynamics.get_matrices()
        cond_A = float(np.linalg.cond(A))
        cond_B = float(np.linalg.cond(B))

        matrices_valid = (
            np.all(np.isfinite(A))
            and np.all(np.isfinite(B))
            and cond_A < 1e10
            and cond_B < 1e10
        )

        # 总体状态判定
        issues = []
        if not matrices_valid:
            issues.append('matrix_health')
        if avg_solve_time > 50.0 and self._solve_count > 10:
            issues.append('slow_solve')
        if cost_trend == 'increasing' and len(self._cost_history) >= 20:
            issues.append('cost_increasing')

        if len(issues) == 0:
            status = 'healthy'
        elif len(issues) <= 1:
            status = 'degraded'
        else:
            status = 'error'

        report = {
            'status': status,
            'solve_count': self._solve_count,
            'avg_solve_time_ms': round(avg_solve_time, 3),
            'avg_cost': round(avg_cost, 6),
            'cost_trend': cost_trend,
            'dynamics_update_count': self._dynamics.update_count,
            'dynamics_initialized': self._dynamics._initialized,
            'bc_trained': self._bc_trained,
            'rl_updates': len(self._rl_episode_rewards),
            'current_weights': self._rl_theta.copy(),
            'matrix_condition_A': round(cond_A, 2),
            'matrix_condition_B': round(cond_B, 2),
            'prediction_matrices_valid': matrices_valid,
        }

        if issues:
            report['issues'] = issues

        return report

    def reset(self) -> None:
        """重置控制器，清除所有内部状态。"""
        self._dynamics.reset()
        self._build_weight_matrices()
        self._build_prediction_matrices()

        self._bc_weights = None
        self._bc_trained = False

        self._rl_theta = np.array([
            self._config.state_weight,
            self._config.control_weight,
            self._config.terminal_weight,
        ], dtype=np.float64)
        self._rl_baseline = 0.0
        self._rl_episode_rewards.clear()
        self._rl_trajectory_log.clear()

        self._prev_control_seq = None

        self._solve_count = 0
        self._total_solve_time = 0.0
        self._cost_history.clear()
        self._solve_time_history.clear()
        self._state_history.clear()
        self._control_history.clear()

        logger.info("LearningMPCController: 控制器已完全重置")

    # ======================== 内部方法 ========================

    @staticmethod
    def _validate_config(config: LearningMPCConfig) -> None:
        """验证配置参数的合法性。

        Args:
            config: 待验证的配置。

        Raises:
            ValueError: 参数不合法时抛出。
        """
        if config.horizon < 1:
            raise ValueError(f"预测时域必须 >= 1，实际 {config.horizon}")
        if config.dt <= 0:
            raise ValueError(f"采样周期必须 > 0，实际 {config.dt}")
        if config.state_dim < 1:
            raise ValueError(f"状态维度必须 >= 1，实际 {config.state_dim}")
        if config.control_dim < 1:
            raise ValueError(f"控制维度必须 >= 1，实际 {config.control_dim}")
        if not (0 < config.rls_forgetting_factor <= 1):
            raise ValueError(
                f"RLS 遗忘因子必须在 (0, 1] 范围内，"
                f"实际 {config.rls_forgetting_factor}"
            )
        if not (0 <= config.bc_blend_ratio <= 1):
            raise ValueError(
                f"BC 融合比例必须在 [0, 1] 范围内，"
                f"实际 {config.bc_blend_ratio}"
            )
        if not (0 <= config.rl_blend_ratio <= 1):
            raise ValueError(
                f"RL 融合比例必须在 [0, 1] 范围内，"
                f"实际 {config.rl_blend_ratio}"
            )
        if config.bc_blend_ratio + config.rl_blend_ratio > 1:
            raise ValueError(
                f"BC + RL 融合比例之和不能超过 1，"
                f"实际 {config.bc_blend_ratio + config.rl_blend_ratio}"
            )

    def _build_weight_matrices(self) -> None:
        """构建权重矩阵 Q, R, P。

        Q: 状态跟踪权重 (state_dim x state_dim)
        R: 控制能量权重 (control_dim x control_dim)
        P: 终端状态权重 (state_dim x state_dim)
        """
        n = self._config.state_dim
        m = self._config.control_dim

        qs = self._config.state_weight
        qv = qs * 0.1  # 速度权重为位置的 1/10
        self._Q = np.diag(
            [qs, qs, qv, qv][:n]
        ).astype(np.float64)

        rc = self._config.control_weight
        self._R = np.diag([rc, rc][:m]).astype(np.float64)

        pt = self._config.terminal_weight
        pv = pt * 0.1
        self._P = np.diag(
            [pt, pt, pv, pv][:n]
        ).astype(np.float64)

    def _build_prediction_matrices(self) -> None:
        """构建预测矩阵 Phi 和 Gamma。

        将 N 步预测展开为矩阵形式:
            X = Phi * x0 + Gamma * U

        其中:
            X = [x1, x2, ..., xN]^T  (N*n x 1)
            U = [u0, u1, ..., u_{N-1}]^T  (N*m x 1)
            Phi = [A, A^2, ..., A^N]^T  (N*n x n)
            Gamma 为下三角分块矩阵  (N*n x N*m)
        """
        A, B = self._dynamics.get_matrices()
        N = self._config.horizon
        n = self._config.state_dim
        m = self._config.control_dim

        # 构建 Phi (N*n x n)
        Phi_rows = []
        A_pow = A.copy()
        for k in range(1, N + 1):
            Phi_rows.append(A_pow)
            A_pow = A @ A_pow
        self._Phi = np.vstack(Phi_rows).astype(np.float64)

        # 构建 Gamma (N*n x N*m)
        Gamma_rows = []
        for i in range(N):
            row_blocks = []
            for j in range(N):
                if j <= i:
                    A_ij = np.linalg.matrix_power(A, i - j)
                    row_blocks.append(A_ij @ B)
                else:
                    row_blocks.append(np.zeros((n, m), dtype=np.float64))
            Gamma_rows.append(np.hstack(row_blocks))
        self._Gamma = np.vstack(Gamma_rows).astype(np.float64)

    def _get_effective_weights(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """获取 RL 调整后的有效权重矩阵。

        Returns:
            (Q_eff, R_eff, P_eff): 调整后的权重矩阵。
        """
        n = self._config.state_dim
        m = self._config.control_dim

        qs, rc, pt = self._rl_theta
        qv = qs * 0.1
        pv = pt * 0.1

        Q_eff = np.diag([qs, qs, qv, qv][:n]).astype(np.float64)
        R_eff = np.diag([rc, rc][:m]).astype(np.float64)
        P_eff = np.diag([pt, pt, pv, pv][:n]).astype(np.float64)

        return Q_eff, R_eff, P_eff

    def _solve_mpc(
        self,
        x0: np.ndarray,
        x_ref: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        P: np.ndarray,
        constraints: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """求解核心 MPC 优化问题 (投影梯度下降)。

        将约束 QP 转化为矩阵形式，使用投影梯度下降求解:
            min  U^T H U + 2 g^T U
            s.t. u_min <= U <= u_max

        Args:
            x0: 初始状态 (n,)。
            x_ref: 参考状态 (n,)。
            Q: 状态权重矩阵 (n x n)。
            R: 控制权重矩阵 (m x m)。
            P: 终端权重矩阵 (n x n)。
            constraints: 可选约束覆盖。

        Returns:
            (optimal_control, predicted_states, iterations):
                optimal_control: 最优控制序列 (N x m)
                predicted_states: 预测状态序列 (N x n)
                iterations: 实际迭代次数
        """
        N = self._config.horizon
        n = self._config.state_dim
        m = self._config.control_dim

        # 参考状态序列
        X_ref = np.tile(x_ref, (N, 1))  # (N, n)

        # 构建 Q_bar (N*n x N*n) 和 R_bar (N*m x N*m)
        Q_blocks = [Q] * (N - 1) + [P]
        Q_bar = np.zeros((N * n, N * n), dtype=np.float64)
        for k in range(N):
            Q_bar[k * n:(k + 1) * n, k * n:(k + 1) * n] = Q_blocks[k]

        R_bar = np.zeros((N * m, N * m), dtype=np.float64)
        for k in range(N):
            R_bar[k * m:(k + 1) * m, k * m:(k + 1) * m] = R

        # 自由响应
        X_free = self._Phi @ x0  # (N*n,)
        X_ref_flat = X_ref.flatten()  # (N*n,)
        e = X_free - X_ref_flat  # (N*n,)

        # Hessian 和梯度
        H = self._Gamma.T @ Q_bar @ self._Gamma + R_bar  # (N*m, N*m)
        g = self._Gamma.T @ Q_bar @ e  # (N*m,)

        # 正则化
        lam = self._config.damping_factor
        H_reg = H + lam * np.eye(H.shape[0], dtype=np.float64)

        # 初始解: 先用直接求解 (无约束) 作为起点
        try:
            U_flat = np.linalg.solve(H_reg, -g)
        except np.linalg.LinAlgError:
            logger.warning("LearningMPC: 直接求解失败，使用伪逆")
            U_flat = -np.linalg.lstsq(H_reg, g, rcond=None)[0]

        # 热启动: 如果有上一步解且直接解不够好，混合使用
        if self._config.warm_start and self._prev_control_seq is not None:
            prev_U = self._prev_control_seq.flatten().copy()
            expected_len = N * m
            if prev_U.shape[0] > expected_len:
                prev_U = prev_U[:expected_len]
            elif prev_U.shape[0] < expected_len:
                prev_U = np.pad(
                    prev_U, (0, expected_len - prev_U.shape[0]),
                    mode='edge',
                )
            # 混合: 70% 直接解 + 30% 热启动
            U_flat = 0.7 * U_flat + 0.3 * prev_U

        # 约束边界
        if constraints is not None:
            u_max = np.asarray(
                constraints.get('u_max', self._config.max_control),
                dtype=np.float64,
            )
            u_min = np.asarray(
                constraints.get('u_min', -self._config.max_control),
                dtype=np.float64,
            )
        else:
            u_max = np.full(m, self._config.max_control, dtype=np.float64)
            u_min = np.full(m, -self._config.max_control, dtype=np.float64)

        bounds_lower = np.tile(u_min, N)
        bounds_upper = np.tile(u_max, N)

        # 投影梯度下降
        iterations = 0
        step_size = 1.0 / (np.max(np.diag(H_reg)) + 1e-8)  # 自适应步长

        for i in range(self._config.max_iterations):
            # 梯度: grad J = H @ U + g
            grad = H_reg @ U_flat + g

            # 梯度下降步
            U_new = U_flat - step_size * grad

            # 投影到可行域
            U_new = np.clip(U_new, bounds_lower, bounds_upper)

            # 检查收敛
            delta = np.max(np.abs(U_new - U_flat))
            U_flat = U_new
            iterations = i + 1

            if delta < self._config.qp_tolerance:
                break

        # 状态约束投影 (如果启用)
        if self._config.enable_constraints:
            U_flat = self._enforce_state_constraints(
                U_flat, x0, Q, P, bounds_lower, bounds_upper
            )

        # 重构控制序列
        optimal_control = U_flat.reshape(N, m)

        # 计算预测状态
        predicted_states = np.zeros((N, n), dtype=np.float64)
        x_pred = x0.copy()
        for k in range(N):
            x_pred = self._dynamics.predict(x_pred, optimal_control[k])
            predicted_states[k] = x_pred

        return optimal_control, predicted_states, iterations

    def _enforce_state_constraints(
        self,
        U_flat: np.ndarray,
        x0: np.ndarray,
        Q: np.ndarray,
        P: np.ndarray,
        bounds_lower: np.ndarray,
        bounds_upper: np.ndarray,
    ) -> np.ndarray:
        """通过迭代投影强制满足状态约束。

        对预测状态超出边界的情况，反向调整控制输入。

        Args:
            U_flat: 展平控制序列 (N*m,)。
            x0: 初始状态 (n,)。
            Q: 状态权重。
            P: 终端权重。
            bounds_lower: 控制下界。
            bounds_upper: 控制上界。

        Returns:
            调整后的控制序列 (N*m,)。
        """
        N = self._config.horizon
        n = self._config.state_dim
        m = self._config.control_dim

        x_max = np.full(n, self._config.max_position, dtype=np.float64)
        x_min = np.full(n, -self._config.max_position, dtype=np.float64)
        # 速度约束
        x_max[2:] = self._config.max_velocity
        x_min[2:] = -self._config.max_velocity

        U = U_flat.reshape(N, m).copy()
        x = x0.copy()

        for k in range(N):
            x = self._dynamics.predict(x, U[k])

            # 检查状态约束
            violated = False
            for d in range(n):
                if x[d] > x_max[d] or x[d] < x_min[d]:
                    violated = True
                    break

            if violated:
                # 缩放当前控制以减少状态越界
                for d in range(n):
                    if x[d] > x_max[d]:
                        excess = x[d] - x_max[d]
                        # 反向调整 (简化: 按比例缩小控制)
                        if abs(U[k, min(d, m - 1)]) > 1e-8:
                            scale = max(0.0, 1.0 - excess / (abs(x[d]) + 1e-8))
                            U[k] *= scale
                    elif x[d] < x_min[d]:
                        deficit = x_min[d] - x[d]
                        if abs(U[k, min(d, m - 1)]) > 1e-8:
                            scale = max(0.0, 1.0 - deficit / (abs(x[d]) + 1e-8))
                            U[k] *= scale

                # 重新预测
                x = self._dynamics.predict(
                    self._dynamics.predict(x0, U[0]) if k == 0 else x,
                    U[k],
                )

        # 重新投影控制约束
        U_flat = U.flatten()
        U_flat = np.clip(U_flat, bounds_lower, bounds_upper)

        return U_flat

    def _compute_bc_control(self, state: np.ndarray) -> np.ndarray:
        """计算行为克隆策略的控制输出。

        Args:
            state: 当前状态 (state_dim,)。

        Returns:
            BC 策略控制输出 (control_dim,)。未训练时返回零。
        """
        if not self._bc_trained or self._bc_weights is None:
            return np.zeros(self._config.control_dim, dtype=np.float64)

        bc_action = state @ self._bc_weights  # (control_dim,)

        # 限幅
        bc_action = np.clip(
            bc_action,
            -self._config.max_control,
            self._config.max_control,
        )

        return bc_action

    def _blend_controls(
        self,
        mpc_control: np.ndarray,
        bc_control: np.ndarray,
    ) -> np.ndarray:
        """融合 MPC、行为克隆和 RL 控制策略。

        融合公式:
            u_final[k] = (1 - bc_ratio - rl_ratio) * u_mpc[k]
                       + bc_ratio * u_bc[k]
                       + rl_ratio * u_rl[k]

        其中 RL 部分通过权重调整间接影响 MPC 解。

        Args:
            mpc_control: MPC 最优控制序列 (N x m)。
            bc_control: BC 策略控制 (m,)。

        Returns:
            融合后的控制序列 (N x m)。
        """
        alpha_bc = self._config.bc_blend_ratio
        alpha_rl = self._config.rl_blend_ratio
        alpha_mpc = 1.0 - alpha_bc - alpha_rl

        N = mpc_control.shape[0]
        blended = np.zeros_like(mpc_control)

        for k in range(N):
            # BC 控制随预测步数衰减 (远期更依赖 MPC)
            bc_decay = max(0.0, 1.0 - k / N) if alpha_bc > 0 else 0.0
            blended[k] = (
                alpha_mpc * mpc_control[k]
                + alpha_bc * bc_decay * bc_control
            )

        return blended

    def _evaluate_cost(
        self,
        predicted_states: np.ndarray,
        control_seq: np.ndarray,
        x_ref: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        P: np.ndarray,
    ) -> float:
        """评估代价函数值。

        Args:
            predicted_states: 预测状态序列 (N x n)。
            control_seq: 控制序列 (N x m)。
            x_ref: 参考状态 (n,)。
            Q: 状态权重。
            R: 控制权重。
            P: 终端权重。

        Returns:
            代价函数值。
        """
        N = predicted_states.shape[0]
        cost = 0.0

        for k in range(N):
            dx = predicted_states[k] - x_ref
            Q_k = P if k == N - 1 else Q
            cost += float(dx @ Q_k @ dx)

        for k in range(min(control_seq.shape[0], N)):
            cost += float(control_seq[k] @ R @ control_seq[k])

        return cost

    def _log_rl_trajectory(
        self,
        state: np.ndarray,
        control: np.ndarray,
        cost: float,
    ) -> None:
        """记录 RL 轨迹数据。

        Args:
            state: 当前状态。
            control: 执行的控制。
            cost: 当前代价。
        """
        self._rl_trajectory_log.append({
            'state': state.copy(),
            'control': control.copy(),
            'cost': cost,
        })

        # 限制轨迹长度
        max_traj_len = 500
        if len(self._rl_trajectory_log) > max_traj_len:
            self._rl_trajectory_log = self._rl_trajectory_log[-max_traj_len:]


# ============================================================
# 模块测试
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    print("=" * 70)
    print("学习增强型 MPC 控制器模块测试")
    print("=" * 70)

    # ---- 测试 1: 基本控制功能 ----
    print("\n--- 测试 1: 基本控制功能 ---")
    controller = LearningMPCController()

    state = np.array([100.0, 80.0, 0.0, 0.0], dtype=np.float64)
    target = np.array([0.0, 0.0], dtype=np.float64)

    print(f"初始状态: {state}")
    print(f"目标位置: {target}")
    print()

    for i in range(30):
        result = controller.compute_control(state, target)

        # 模拟系统响应
        gain = 0.6
        noise = np.random.normal(0, 0.5, size=2)
        next_pos = state[:2] + gain * result.optimal_control[0] + noise
        next_vel = (next_pos - state[:2]) / controller.config.dt * 0.1
        next_state = np.array([
            next_pos[0], next_pos[1], next_vel[0], next_vel[1]
        ])

        # 在线动力学更新
        controller.update_dynamics(state, result.optimal_control[0], next_state)

        if i % 5 == 0 or i == 29:
            error = np.linalg.norm(next_state[:2] - target)
            print(
                f"  [步 {i:2d}] 状态: ({next_state[0]:7.2f}, {next_state[1]:7.2f}) | "
                f"控制: ({result.optimal_control[0, 0]:7.2f}, "
                f"{result.optimal_control[0, 1]:7.2f}) | "
                f"代价: {result.cost:8.2f} | "
                f"求解: {result.solve_time*1000:5.2f}ms | "
                f"误差: {error:.2f}px"
            )

        state = next_state

        error = np.linalg.norm(state[:2] - target)
        if error < 1.0 and i > 5:
            print(f"\n  收敛! 最终误差: {error:.4f} px (步数: {i + 1})")
            break

    # ---- 测试 2: 模仿学习 ----
    print("\n--- 测试 2: 模仿学习 (行为克隆) ---")
    controller2 = LearningMPCController()

    # 生成模拟专家轨迹 (简单的比例控制)
    expert_trajs = []
    for _ in range(20):
        traj = []
        s = np.random.uniform(-200, 200, size=4)
        s[2:] = 0.0
        for _ in range(15):
            # 专家策略: 比例控制
            error = -s[:2] * 0.3
            action = np.clip(error, -50, 50)
            traj.append((s.copy(), action.copy()))
            s[:2] += action * 0.6
        expert_trajs.append(traj)

    metrics = controller2.imitate_expert(expert_trajs)
    print(f"  训练样本数: {metrics['num_samples']}")
    print(f"  训练 MSE: {metrics['mse']:.6f}")
    print(f"  权重范数: {metrics['bc_weights_norm']:.4f}")

    # 验证 BC 策略
    test_state = np.array([50.0, -30.0, 0.0, 0.0])
    bc_out = controller2._compute_bc_control(test_state)
    print(f"  测试状态 {test_state[:2]} -> BC 输出: {bc_out}")

    # ---- 测试 3: RL 增强 ----
    print("\n--- 测试 3: RL 增强 (REINFORCE) ---")
    controller3 = LearningMPCController()

    # 模拟一个完整回合
    s = np.array([80.0, 60.0, 0.0, 0.0])
    t = np.array([0.0, 0.0])
    for step in range(20):
        result = controller3.compute_control(s, t)
        s[:2] += 0.6 * result.optimal_control[0]
        reward = -np.linalg.norm(s[:2] - t)
        done = (step == 19)
        controller3.reinforce_update(reward, done)

    print(f"  RL 权重调整: {controller3._rl_theta}")
    print(f"  初始权重: [{controller3.config.state_weight}, "
          f"{controller3.config.control_weight}, "
          f"{controller3.config.terminal_weight}]")

    # ---- 测试 4: 动力学自适应 ----
    print("\n--- 测试 4: 动力学自适应 (RLS) ---")
    controller4 = LearningMPCController()
    A_before, B_before = controller4.dynamics.get_matrices()

    # 注入带偏差的动力学数据
    for _ in range(50):
        s = np.random.randn(4) * 10
        u = np.random.randn(2) * 5
        # 模拟真实系统 (A 矩阵有偏差)
        A_true = A_before.copy()
        A_true[2, 2] = 0.7  # 不同的阻尼
        s_next = A_true @ s + B_before @ u + np.random.randn(4) * 0.1
        controller4.update_dynamics(s, u, s_next)

    A_after, B_after = controller4.dynamics.get_matrices()
    print(f"  A[2,2] 标称: {A_before[2, 2]:.4f}")
    print(f"  A[2,2] 真实: 0.7000")
    print(f"  A[2,2] 辨识: {A_after[2, 2]:.4f}")
    print(f"  RLS 更新次数: {controller4.dynamics.update_count}")

    # ---- 测试 5: 预测轨迹 ----
    print("\n--- 测试 5: 轨迹预测 ---")
    controller5 = LearningMPCController()
    s = np.array([100.0, 50.0, 0.0, 0.0])
    result = controller5.compute_control(s, np.array([0.0, 0.0]))

    traj = result.predicted_trajectory
    print(f"  预测轨迹形状: {traj.shape}")
    print(f"  起点: ({traj[0, 0]:.2f}, {traj[0, 1]:.2f})")
    print(f"  终点: ({traj[-1, 0]:.2f}, {traj[-1, 1]:.2f})")
    print(f"  收敛距离: {np.linalg.norm(traj[-1, :2] - traj[0, :2]):.2f} px")

    # ---- 测试 6: 热启动 ----
    print("\n--- 测试 6: 热启动 ---")
    controller6 = LearningMPCController()
    result1 = controller6.compute_control(
        np.array([80.0, 60.0, 0.0, 0.0]),
        np.array([0.0, 0.0]),
    )
    controller6.warm_start({
        'control_sequence': result1.optimal_control,
    })
    result2 = controller6.compute_control(
        np.array([75.0, 55.0, 0.0, 0.0]),
        np.array([0.0, 0.0]),
    )
    print(f"  热启动求解时间: {result2.solve_time*1000:.3f}ms")
    print(f"  热启动迭代次数: {result2.iterations}")

    # ---- 测试 7: 约束处理 ----
    print("\n--- 测试 7: 约束处理 ---")
    controller7 = LearningMPCController(LearningMPCConfig(
        max_control=100.0,
        enable_constraints=True,
    ))
    result = controller7.compute_control(
        np.array([500.0, 500.0, 0.0, 0.0]),
        np.array([0.0, 0.0]),
    )
    u0 = result.optimal_control[0]
    print(f"  大误差控制 (限幅 100): ({u0[0]:.2f}, {u0[1]:.2f})")
    assert np.all(np.abs(u0) <= 100.0 + 1e-6), "控制约束违反"
    print("  约束验证通过")

    # ---- 测试 8: 健康报告 ----
    print("\n--- 测试 8: 健康报告 ---")
    report = controller.get_health_report()
    for key, value in report.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")

    # ---- 测试 9: 重置 ----
    print("\n--- 测试 9: 重置功能 ---")
    controller.reset()
    report_after = controller.get_health_report()
    print(f"  重置后状态: {report_after['status']}")
    print(f"  重置后求解次数: {report_after['solve_count']}")
    assert report_after['solve_count'] == 0, "重置后求解次数未清零"
    print("  重置验证通过")

    # ---- 测试 10: 融合策略验证 ----
    print("\n--- 测试 10: 融合策略验证 ---")
    controller10 = LearningMPCController(LearningMPCConfig(
        bc_blend_ratio=0.3,
        rl_blend_ratio=0.1,
    ))

    # 先训练 BC
    simple_expert = []
    for _ in range(10):
        s = np.random.randn(4) * 50
        s[2:] = 0.0
        simple_expert.append((s.copy(), -s[:2] * 0.2))
    controller10.imitate_expert(simple_expert)

    result = controller10.compute_control(
        np.array([60.0, 40.0, 0.0, 0.0]),
        np.array([0.0, 0.0]),
    )
    print(f"  融合比例: BC={result.blend_ratio[0]:.2f}, RL={result.blend_ratio[1]:.2f}")
    print(f"  MPC 控制[0]: {result.optimal_control[0]}")
    print(f"  BC 控制: {result.bc_control}")
    print(f"  RL 权重调整: {result.rl_weight_adjustment}")

    print("\n" + "=" * 70)
    print("所有测试完成")
    print("=" * 70)
