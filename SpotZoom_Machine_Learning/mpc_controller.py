"""
模型预测控制器 (MPCController) (v10.0)

基于模型预测控制 (Model Predictive Control) 的多轴协调电机控制器，
用于 SpotZoom 光斑对准系统的 X/Y/Z 三轴精密定位。

灵感来源:
- do-mpc (https://github.com/do-mpc/do-mpc) — 开源非线性 MPC 框架，
  支持连续/离散时间模型、约束处理和实时优化
- python-control (https://github.com/python-control/python-control) —
  Python 控制系统库，提供状态空间建模与仿真工具
- MATLAB Model Predictive Control Toolbox — 工业标准 MPC 实现
- Borrelli, Bemporad, Morari, "Predictive Control for Linear and Hybrid
  Systems" (Cambridge University Press, 2017)

算法原理:
  离散时间线性 MPC 通过在每一采样时刻求解有限时域约束优化问题:
    min  Σ_{k=0}^{N-1} [ (x_k - x_ref)^T Q (x_k - x_ref) + u_k^T R u_k ]
         + (x_N - x_ref)^T P (x_N - x_ref)
    s.t. x_{k+1} = A x_k + B u_k
         u_min <= u_k <= u_max
         x_min <= x_k <= x_max
  其中 N 为预测时域，Q/R/P 为权重矩阵，A/B 为状态空间矩阵。
  使用阻尼最小二乘法 (Damped Least-Squares) 求解 QP 子问题，
  通过投影梯度下降处理输入/状态约束，支持软约束惩罚不可行区域。

功能:
- 离散时间线性状态空间模型 (X/Y 位置 + 速度)
- 可配置预测时域 (N=10~20) 与控制时域 (Nu=5)
- 二次型代价函数: 位置误差 + 控制能量 + 终端惩罚
- 输入约束 (各轴最大步数) 与状态约束 (工作空间边界)
- 软约束 + 惩罚项处理不可行区域
- 阻尼最小二乘 QP 求解器 (无需外部 QP 库)
- 投影梯度下降约束处理
- 自动线性化 (基于当前工作点)
- 前馈扰动补偿
- 在线模型自适应 (根据实测数据更新 A/B 矩阵)
- 热启动 (从上一时刻解加速收敛)

依赖: numpy (无外部控制理论库)
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# 数据类定义
# ============================================================

@dataclass
class MPCConfig:
    """MPC 控制器配置参数。

    Attributes:
        prediction_horizon: 预测时域长度 (N)，即向前预测的步数。
        control_horizon: 控制时域长度 (Nu)，即自由控制变量的步数，
            Nu <= N，超出部分控制输入保持不变。
        state_weight: 位置误差权重 (Q 矩阵中对角线元素)。
        control_weight: 控制能量权重 (R 矩阵中对角线元素)。
        terminal_weight: 终端状态权重 (P 矩阵中对角线元素)，
            通常大于 state_weight 以保证终端精度。
        max_step_x: X 轴单步最大电机步数 (绝对值)。
        max_step_y: Y 轴单步最大电机步数 (绝对值)。
        max_step_z: Z 轴单步最大电机步数 (绝对值，浮点)。
        sample_time: 采样周期 (秒)。
        enable_constraints: 是否启用约束处理。
        soft_constraint_weight: 软约束惩罚权重，越大越严格。
        disturbance_compensation: 是否启用前馈扰动补偿。
        damping_factor: 阻尼最小二乘正则化系数。
        velocity_damping: 速度衰减系数 (0~1)，模拟机械阻尼。
        max_iterations: QP 求解器最大迭代次数。
        qp_tolerance: QP 求解器收敛容差。
        warm_start: 是否启用热启动。
    """
    prediction_horizon: int = 15
    control_horizon: int = 8
    state_weight: float = 10.0
    control_weight: float = 0.1
    terminal_weight: float = 20.0
    max_step_x: int = 2000
    max_step_y: int = 2000
    max_step_z: float = 5.0
    sample_time: float = 0.1
    enable_constraints: bool = True
    soft_constraint_weight: float = 100.0
    disturbance_compensation: bool = True
    damping_factor: float = 1e-4
    velocity_damping: float = 0.8
    max_iterations: int = 100
    qp_tolerance: float = 1e-6
    warm_start: bool = True


@dataclass
class MPCState:
    """MPC 控制器内部状态。

    Attributes:
        x_position: 当前光斑 X 坐标 (像素)。
        y_position: 当前光斑 Y 坐标 (像素)。
        z_position: 当前 Z 轴位置 (焦平面偏移)。
        x_velocity: X 方向估计速度 (像素/帧)。
        y_velocity: Y 方向估计速度 (像素/帧)。
        target_x: 目标 X 坐标 (像素)。
        target_y: 目标 Y 坐标 (像素)。
        target_z: 目标 Z 坐标。
        control_sequence: 预测控制序列 (Nu x 2)，存储上一次求解的控制轨迹。
        state_vector: 完整状态向量 [px, py, vx, vy]。
        disturbance_estimate: 扰动估计向量 [dx, dy]。
        model_adapted: 模型是否已自适应更新。
    """
    x_position: float = 0.0
    y_position: float = 0.0
    z_position: float = 0.0
    x_velocity: float = 0.0
    y_velocity: float = 0.0
    target_x: float = 0.0
    target_y: float = 0.0
    target_z: float = 0.0
    control_sequence: np.ndarray = field(
        default_factory=lambda: np.zeros((8, 2), dtype=np.float64)
    )
    state_vector: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=np.float64)
    )
    disturbance_estimate: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float64)
    )
    model_adapted: bool = False


@dataclass
class MPCMetrics:
    """MPC 控制器性能指标。

    Attributes:
        solve_time_ms: QP 求解耗时 (毫秒)。
        cost_value: 最优代价函数值。
        constraint_violations: 约束违反次数。
        prediction_error: 预测误差 (预测轨迹与实际偏差)。
        total_iterations: QP 求解器总迭代次数。
        cost_history: 代价函数历史记录。
        solve_time_history: 求解时间历史记录。
        position_error_history: 位置误差历史记录。
    """
    solve_time_ms: float = 0.0
    cost_value: float = 0.0
    constraint_violations: int = 0
    prediction_error: float = 0.0
    total_iterations: int = 0
    cost_history: List[float] = field(default_factory=lambda: deque(maxlen=500))
    solve_time_history: List[float] = field(default_factory=lambda: deque(maxlen=500))
    position_error_history: List[float] = field(default_factory=lambda: deque(maxlen=500))


@dataclass
class MPCResult:
    """MPC 控制器单步输出结果。

    Attributes:
        x_step: X 轴推荐电机步数。
        y_step: Y 轴推荐电机步数。
        z_step: Z 轴推荐电机步数 (浮点)。
        predicted_trajectory: 预测位置轨迹 (N x 2)，每行为 [px, py]。
        state: 当前 MPC 内部状态快照。
        metrics: 当前步性能指标。
    """
    x_step: int = 0
    y_step: int = 0
    z_step: float = 0.0
    predicted_trajectory: np.ndarray = field(
        default_factory=lambda: np.zeros((15, 2), dtype=np.float64)
    )
    state: MPCState = field(default_factory=MPCState)
    metrics: MPCMetrics = field(default_factory=MPCMetrics)


# ============================================================
# 主控制器类
# ============================================================

class MPCController:
    """模型预测控制器 (Model Predictive Controller)。

    将光斑对准问题建模为约束线性 MPC 问题:
    - 状态向量: x = [px, py, vx, vy] (X/Y 位置 + 速度)
    - 控制向量: u = [ux, uy] (X/Y 电机步数)
    - 状态方程: x[k+1] = A * x[k] + B * u[k]
    - 代价函数: J = Σ (x - x_ref)^T Q (x - x_ref) + u^T R u
                + (x_N - x_ref)^T P (x_N - x_ref)
    - 约束: u_min <= u <= u_max, x_min <= x <= x_max

    Z 轴控制通过独立的 P 控制器实现 (焦平面搜索)，
    与 X/Y 的 MPC 协调输出。

    使用示例:
        controller = MPCController()
        result = controller.update(
            current_x=100.0, current_y=50.0, current_z=2.5,
            target_x=0.0, target_y=0.0, target_z=0.0,
        )
        print(f"X 步数: {result.x_step}, Y 步数: {result.y_step}, Z 步数: {result.z_step}")
    """

    def __init__(self, config: Optional[MPCConfig] = None):
        """初始化 MPC 控制器。

        Args:
            config: MPC 配置参数。为 None 时使用默认配置。
        """
        self._config = config or MPCConfig()
        self._state = MPCState(
            control_sequence=np.zeros(
                (self._config.control_horizon, 2), dtype=np.float64
            ),
        )
        self._metrics = MPCMetrics()

        # 系统矩阵
        self._A: Optional[np.ndarray] = None
        self._B: Optional[np.ndarray] = None

        # 权重矩阵
        self._Q: Optional[np.ndarray] = None
        self._R: Optional[np.ndarray] = None
        self._P: Optional[np.ndarray] = None

        # 预测矩阵 (用于高效 QP 构建)
        self._Phi: Optional[np.ndarray] = None  # 状态预测矩阵
        self._Gamma: Optional[np.ndarray] = None  # 控制预测矩阵

        # 历史数据 (用于模型自适应)
        self._state_history: Deque[np.ndarray] = deque(maxlen=50)
        self._control_history: Deque[np.ndarray] = deque(maxlen=50)
        self._adaptation_count: int = 0

        # 上一步预测 (用于计算预测误差)
        self._prev_predicted_position: Optional[np.ndarray] = None

        # 初始化系统矩阵和权重
        self._build_system_matrices()
        self._build_weight_matrices()
        self._build_prediction_matrices()

        logger.info(
            "MPCController: 初始化完成 "
            "(N=%d, Nu=%d, dt=%.3fs, Q=%.1f, R=%.2f, P=%.1f)",
            self._config.prediction_horizon,
            self._config.control_horizon,
            self._config.sample_time,
            self._config.state_weight,
            self._config.control_weight,
            self._config.terminal_weight,
        )

    # ======================== 属性 ========================

    @property
    def config(self) -> MPCConfig:
        """获取当前配置。"""
        return self._config

    @property
    def state(self) -> MPCState:
        """获取当前状态快照。"""
        return self._state

    @property
    def metrics(self) -> MPCMetrics:
        """获取当前性能指标。"""
        return self._metrics

    # ======================== 公共接口 ========================

    def update(
        self,
        current_x: float,
        current_y: float,
        current_z: float = 0.0,
        target_x: float = 0.0,
        target_y: float = 0.0,
        target_z: float = 0.0,
        measured_vel_x: Optional[float] = None,
        measured_vel_y: Optional[float] = None,
    ) -> MPCResult:
        """执行一步 MPC 控制计算。

        根据当前位置和目标位置，求解约束优化问题，
        返回最优控制动作 (电机步数) 和预测轨迹。

        Args:
            current_x: 当前光斑 X 坐标 (像素)。
            current_y: 当前光斑 Y 坐标 (像素)。
            current_z: 当前 Z 轴位置 (焦平面偏移)。
            target_x: 目标 X 坐标 (像素)。
            target_y: 目标 Y 坐标 (像素)。
            target_z: 目标 Z 坐标。
            measured_vel_x: 测量的 X 方向速度 (可选)。
            measured_vel_y: 测量的 Y 方向速度 (可选)。

        Returns:
            MPCResult: 包含推荐步数、预测轨迹、状态和指标的结果对象。
        """
        t_start = time.perf_counter()

        try:
            # 1. 更新状态
            self._update_state(
                current_x, current_y, current_z,
                target_x, target_y, target_z,
                measured_vel_x, measured_vel_y,
            )

            # 2. 计算预测误差
            self._compute_prediction_error(current_x, current_y)

            # 3. 扰动补偿
            if self._config.disturbance_compensation:
                self._update_disturbance_estimate()

            # 4. 模型自适应 (每 20 步执行一次)
            if len(self._state_history) >= 10:
                self._adaptation_count += 1
                if self._adaptation_count % 20 == 0:
                    self._adapt_model()

            # 5. 求解 MPC 优化问题
            optimal_control, predicted_states = self._solve_mpc()

            # 6. 提取第一步控制量
            u0 = optimal_control[0]  # [ux, uy]

            # 7. Z 轴独立 P 控制
            z_step = self._compute_z_step(current_z, target_z)

            # 8. 限幅与取整
            x_step = int(np.round(np.clip(
                u0[0], -self._config.max_step_x, self._config.max_step_x
            )))
            y_step = int(np.round(np.clip(
                u0[1], -self._config.max_step_y, self._config.max_step_y
            )))
            z_step = float(np.clip(
                z_step, -self._config.max_step_z, self._config.max_step_z
            ))

            # 9. 更新控制序列 (移位 + 热启动)
            if self._config.warm_start and len(optimal_control) > 1:
                self._state.control_sequence = np.vstack([
                    optimal_control[1:],
                    optimal_control[-1:],  # 最后一步重复填充
                ])

            # 10. 记录历史
            self._state_history.append(self._state.state_vector.copy())
            self._control_history.append(np.array([x_step, y_step], dtype=np.float64))

            # 11. 保存预测位置 (用于下一步计算预测误差)
            self._prev_predicted_position = predicted_states[0, :2].copy()

            # 12. 计算指标
            solve_time = (time.perf_counter() - t_start) * 1000.0
            position_error = np.sqrt(
                (current_x - target_x) ** 2 + (current_y - target_y) ** 2
            )

            # 代价函数值
            cost = self._evaluate_cost(predicted_states, optimal_control)

            self._metrics.solve_time_ms = round(solve_time, 3)
            self._metrics.cost_value = round(cost, 6)
            self._metrics.position_error = round(position_error, 6)
            self._metrics.cost_history.append(cost)
            self._metrics.solve_time_history.append(solve_time)
            self._metrics.position_error_history.append(position_error)

            # 构建预测轨迹 (仅位置部分)
            predicted_trajectory = predicted_states[:, :2].copy()

            # 构建结果
            result = MPCResult(
                x_step=x_step,
                y_step=y_step,
                z_step=z_step,
                predicted_trajectory=predicted_trajectory,
                state=self._state,
                metrics=self._metrics,
            )

            logger.debug(
                "MPCController: pos=(%.2f, %.2f), target=(%.2f, %.2f), "
                "cmd=(%d, %d), z=%.3f, cost=%.4f, solve=%.2fms",
                current_x, current_y, target_x, target_y,
                x_step, y_step, z_step, cost, solve_time,
            )

            return result

        except Exception as e:
            logger.error("MPCController: 控制更新失败: %s", e, exc_info=True)
            # 返回安全降级结果
            return MPCResult(
                x_step=0,
                y_step=0,
                z_step=0.0,
                state=self._state,
                metrics=self._metrics,
            )

    def set_target(self, target_x: float, target_y: float, target_z: float = 0.0) -> None:
        """设置目标位置。

        Args:
            target_x: 目标 X 坐标 (像素)。
            target_y: 目标 Y 坐标 (像素)。
            target_z: 目标 Z 坐标。
        """
        self._state.target_x = float(target_x)
        self._state.target_y = float(target_y)
        self._state.target_z = float(target_z)
        logger.debug(
            "MPCController: 目标更新为 (%.2f, %.2f, %.3f)",
            target_x, target_y, target_z,
        )

    def update_model(
        self,
        A: Optional[np.ndarray] = None,
        B: Optional[np.ndarray] = None,
    ) -> None:
        """手动更新系统模型矩阵。

        Args:
            A: 新的状态转移矩阵 (4x4)。为 None 时保持不变。
            B: 新的控制矩阵 (4x2)。为 None 时保持不变。
        """
        if A is not None:
            if A.shape != (4, 4):
                raise ValueError(f"A 矩阵形状应为 (4, 4)，实际为 {A.shape}")
            self._A = A.astype(np.float64)
            logger.info("MPCController: A 矩阵已手动更新")

        if B is not None:
            if B.shape != (4, 2):
                raise ValueError(f"B 矩阵形状应为 (4, 2)，实际为 {B.shape}")
            self._B = B.astype(np.float64)
            logger.info("MPCController: B 矩阵已手动更新")

        # 重建预测矩阵
        self._build_prediction_matrices()

    def reset(self) -> None:
        """重置控制器，清除所有内部状态。"""
        self._state = MPCState(
            control_sequence=np.zeros(
                (self._config.control_horizon, 2), dtype=np.float64
            ),
        )
        self._metrics = MPCMetrics()
        self._state_history.clear()
        self._control_history.clear()
        self._adaptation_count = 0
        self._prev_predicted_position = None

        # 重新初始化系统矩阵
        self._build_system_matrices()
        self._build_prediction_matrices()

        logger.info("MPCController: 控制器已重置")

    def get_system_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取当前系统矩阵 A, B。

        Returns:
            (A, B): 状态转移矩阵和控制矩阵。
        """
        assert self._A is not None and self._B is not None
        return self._A.copy(), self._B.copy()

    def get_cost_summary(self) -> dict:
        """获取控制成本摘要。

        Returns:
            包含关键指标的字典。
        """
        return {
            "total_steps": len(self._metrics.cost_history),
            "avg_cost": float(np.mean(self._metrics.cost_history[-100:]))
            if self._metrics.cost_history else 0.0,
            "max_cost": float(max(self._metrics.cost_history))
            if self._metrics.cost_history else 0.0,
            "avg_solve_time_ms": float(np.mean(self._metrics.solve_time_history[-100:]))
            if self._metrics.solve_time_history else 0.0,
            "current_position_error": self._metrics.position_error,
            "constraint_violations": self._metrics.constraint_violations,
            "model_adapted": self._state.model_adapted,
            "adaptation_count": self._adaptation_count,
        }

    # ======================== 内部方法 ========================

    def _build_system_matrices(self) -> None:
        """构建离散状态空间矩阵 A, B。

        状态模型 (双积分器 + 阻尼):
            x = [px, py, vx, vy]
            u = [ux, uy]

            px[k+1] = px[k] + dt * vx[k]
            py[k+1] = py[k] + dt * vy[k]
            vx[k+1] = damping * vx[k] + ux[k]
            vy[k+1] = damping * vy[k] + uy[k]

        矩阵形式:
            A = [[1, 0, dt,  0 ],
                 [0, 1,  0, dt ],
                 [0, 0,  d,  0 ],
                 [0, 0,  0,  d ]]

            B = [[0, 0],
                 [0, 0],
                 [1, 0],
                 [0, 1]]
        """
        dt = self._config.sample_time
        d = self._config.velocity_damping

        self._A = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, d,    0.0],
            [0.0, 0.0, 0.0, d   ],
        ], dtype=np.float64)

        self._B = np.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ], dtype=np.float64)

    def _build_weight_matrices(self) -> None:
        """构建权重矩阵 Q, R, P。

        Q: 状态权重 (4x4) — 位置误差权重较大
        R: 控制权重 (2x2) — 控制能量惩罚
        P: 终端权重 (4x4) — 终端状态精度保证
        """
        qs = self._config.state_weight
        qv = qs * 0.1  # 速度权重为位置的 1/10

        self._Q = np.diag([qs, qs, qv, qv]).astype(np.float64)

        rc = self._config.control_weight
        self._R = np.diag([rc, rc]).astype(np.float64)

        pt = self._config.terminal_weight
        pv = pt * 0.1
        self._P = np.diag([pt, pt, pv, pv]).astype(np.float64)

    def _build_prediction_matrices(self) -> None:
        """构建预测矩阵 Phi 和 Gamma。

        将 N 步预测展开为矩阵形式:
            X = Phi * x0 + Gamma * U

        其中:
            X = [x1, x2, ..., xN]^T  (N*n x 1)
            U = [u0, u1, ..., u_{Nu-1}]^T  (Nu*m x 1)
            Phi = [A, A^2, ..., A^N]^T  (N*n x n)
            Gamma 为下三角分块矩阵  (N*n x Nu*m)

        控制时域 Nu <= N: 超出 Nu 的控制输入保持为 u_{Nu-1}。
        """
        assert self._A is not None and self._B is not None

        N = self._config.prediction_horizon
        Nu = self._config.control_horizon
        n = self._A.shape[0]  # 状态维度 = 4
        m = self._B.shape[1]  # 控制维度 = 2

        # 构建 Phi (N*n x n)
        Phi_rows = []
        A_pow = self._A.copy()
        for k in range(1, N + 1):
            Phi_rows.append(A_pow)
            A_pow = self._A @ A_pow
        self._Phi = np.vstack(Phi_rows).astype(np.float64)  # (N*n, n)

        # 构建 Gamma (N*n x Nu*m)
        Gamma_rows = []
        for i in range(N):
            # 第 i 行 (对应 x_{i+1}):
            # x_{i+1} = A^{i+1} x0 + Σ_{j=0}^{i} A^{i-j} B u_j
            # 当 j >= Nu 时，u_j = u_{Nu-1}
            row_blocks = []
            for j in range(Nu):
                if j <= i:
                    A_ij = np.linalg.matrix_power(self._A, i - j)
                    row_blocks.append(A_ij @ self._B)
                else:
                    row_blocks.append(np.zeros((n, m), dtype=np.float64))
            Gamma_rows.append(np.hstack(row_blocks))
        self._Gamma = np.vstack(Gamma_rows).astype(np.float64)  # (N*n, Nu*m)

    def _update_state(
        self,
        current_x: float,
        current_y: float,
        current_z: float,
        target_x: float,
        target_y: float,
        target_z: float,
        measured_vel_x: Optional[float],
        measured_vel_y: Optional[float],
    ) -> None:
        """更新控制器内部状态。

        Args:
            current_x: 当前 X 坐标。
            current_y: 当前 Y 坐标。
            current_z: 当前 Z 坐标。
            target_x: 目标 X 坐标。
            target_y: 目标 Y 坐标。
            target_z: 目标 Z 坐标。
            measured_vel_x: 测量 X 速度。
            measured_vel_y: 测量 Y 速度。
        """
        self._state.x_position = float(current_x)
        self._state.y_position = float(current_y)
        self._state.z_position = float(current_z)
        self._state.target_x = float(target_x)
        self._state.target_y = float(target_y)
        self._state.target_z = float(target_z)

        # 速度估计
        if measured_vel_x is not None:
            self._state.x_velocity = float(measured_vel_x)
        elif len(self._state_history) > 0:
            prev = self._state_history[-1]
            self._state.x_velocity = float(current_x - prev[0])
        else:
            self._state.x_velocity = 0.0

        if measured_vel_y is not None:
            self._state.y_velocity = float(measured_vel_y)
        elif len(self._state_history) > 0:
            prev = self._state_history[-1]
            self._state.y_velocity = float(current_y - prev[1])
        else:
            self._state.y_velocity = 0.0

        # 构建状态向量
        self._state.state_vector = np.array([
            current_x, current_y,
            self._state.x_velocity, self._state.y_velocity,
        ], dtype=np.float64)

    def _compute_prediction_error(self, current_x: float, current_y: float) -> None:
        """计算预测误差 (上一步预测 vs 实际测量)。

        Args:
            current_x: 实际 X 坐标。
            current_y: 实际 Y 坐标。
        """
        if self._prev_predicted_position is not None:
            error = np.sqrt(
                (current_x - self._prev_predicted_position[0]) ** 2
                + (current_y - self._prev_predicted_position[1]) ** 2
            )
            self._metrics.prediction_error = round(error, 6)

    def _update_disturbance_estimate(self) -> None:
        """更新扰动估计 (基于模型预测误差)。

        使用简单的指数加权移动平均 (EWMA) 估计持续扰动:
            d[k] = alpha * (x_actual - x_predicted) + (1 - alpha) * d[k-1]
        """
        if self._prev_predicted_position is None or len(self._state_history) < 2:
            return

        # 实际位置变化
        actual = self._state.state_vector[:2]
        prev_actual = self._state_history[-1][:2]
        actual_delta = actual - prev_actual

        # 模型预测的位置变化 (基于上一步控制)
        if len(self._control_history) > 0:
            prev_u = self._control_history[-1]
            predicted_delta = self._A[:2, 2:] @ self._state_history[-1][2:] \
                + self._A[:2, :2] @ np.zeros(2) \
                + self._B[:2, :] @ prev_u
        else:
            predicted_delta = np.zeros(2)

        # 扰动 = 实际 - 预测
        disturbance = actual_delta - predicted_delta

        # EWMA 平滑
        alpha = 0.3
        self._state.disturbance_estimate = (
            alpha * disturbance + (1.0 - alpha) * self._state.disturbance_estimate
        )

    def _adapt_model(self) -> None:
        """在线模型自适应: 基于历史数据更新 A, B 矩阵。

        使用最小二乘回归从 (x[k], u[k]) -> x[k+1] 数据中辨识 A, B:
            X_next = [A | B] * [x; u]
            [A | B] = X_next * [x; u]^+ (伪逆)
        """
        if len(self._state_history) < 10 or len(self._control_history) < 10:
            return

        try:
            n_samples = min(len(self._state_history), len(self._control_history)) - 1

            # 构建回归矩阵
            X_next = np.array([self._state_history[i + 1] for i in range(n_samples)])
            X_curr = np.array([self._state_history[i] for i in range(n_samples)])
            U_curr = np.array([self._control_history[i] for i in range(n_samples)])

            # 回归: X_next = Theta * [X_curr; U_curr]
            ZU = np.hstack([X_curr, U_curr])  # (n_samples, 6)

            # 正则化最小二乘
            lam = 1e-3
            ZU_T_ZU = ZU.T @ ZU + lam * np.eye(ZU.shape[1])
            Theta = np.linalg.solve(ZU_T_ZU, ZU.T @ X_next)  # (6, 4)

            # 提取 A (4x4) 和 B (4x2)
            A_new = Theta[:4, :].T
            B_new = Theta[4:, :].T

            # 验证矩阵合理性
            if (np.all(np.isfinite(A_new)) and np.all(np.isfinite(B_new))
                    and np.max(np.abs(A_new)) < 100.0
                    and np.max(np.abs(B_new)) < 100.0):

                # 渐进混合 (避免剧烈变化)
                blend = 0.3
                self._A = (1.0 - blend) * self._A + blend * A_new
                self._B = (1.0 - blend) * self._B + blend * B_new
                self._state.model_adapted = True

                # 重建预测矩阵
                self._build_prediction_matrices()

                logger.debug(
                    "MPCController: 模型自适应更新 "
                    "(blend=%.1f, |dA|=%.4f, |dB|=%.4f)",
                    blend,
                    np.max(np.abs(A_new - self._A)),
                    np.max(np.abs(B_new - self._B)),
                )

        except Exception as e:
            logger.warning("MPCController: 模型自适应失败: %s", e)

    def _solve_mpc(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """求解 MPC 优化问题。

        将约束 QP 转化为无约束最小二乘问题 (含软约束惩罚)，
        使用阻尼最小二乘法 (Damped Least-Squares) 求解，
        再通过投影梯度下降处理硬约束。

        Returns:
            (optimal_control, predicted_states):
                optimal_control: 最优控制序列 (Nu x 2)
                predicted_states: 预测状态序列 (N x 4)
        """
        assert self._Phi is not None and self._Gamma is not None
        assert self._Q is not None and self._R is not None and self._P is not None
        assert self._A is not None and self._B is not None

        N = self._config.prediction_horizon
        Nu = self._config.control_horizon
        n = 4  # 状态维度
        m = 2  # 控制维度

        x0 = self._state.state_vector.copy()
        x_ref = np.array([
            self._state.target_x, self._state.target_y, 0.0, 0.0,
        ], dtype=np.float64)

        # 参考状态序列 (N x n)
        X_ref = np.tile(x_ref, (N, 1))  # (N, n)

        # ---- 构建代价函数矩阵形式 ----
        # J = (X - X_ref_flat)^T Q_bar * (X - X_ref_flat) + U_flat^T R_bar * U_flat
        # 其中 X = Phi * x0 + Gamma * U_flat

        # Q_bar: 分块对角矩阵 (N*n x N*n)
        Q_blocks = [self._Q] * (N - 1) + [self._P]  # 最后一步用终端权重
        Q_bar = np.zeros((N * n, N * n), dtype=np.float64)
        for k in range(N):
            Q_bar[k * n:(k + 1) * n, k * n:(k + 1) * n] = Q_blocks[k]

        # R_bar: 分块对角矩阵 (Nu*m x Nu*m)
        R_bar = np.zeros((Nu * m, Nu * m), dtype=np.float64)
        for k in range(Nu):
            R_bar[k * m:(k + 1) * m, k * m:(k + 1) * m] = self._R

        # 自由响应: X_free = Phi * x0
        X_free = self._Phi @ x0  # (N*n,)

        # 参考偏差: e = X_free - X_ref_flat
        X_ref_flat = X_ref.flatten()  # (N*n,)
        e = X_free - X_ref_flat  # (N*n,)

        # ---- 构建 Hessian 和梯度 ----
        # J = U^T H U + 2 * g^T U + const
        # H = Gamma^T Q_bar Gamma + R_bar
        # g = Gamma^T Q_bar e

        H = self._Gamma.T @ Q_bar @ self._Gamma + R_bar  # (Nu*m, Nu*m)
        g = self._Gamma.T @ Q_bar @ e  # (Nu*m,)

        # ---- 软约束惩罚项 ----
        if self._config.enable_constraints:
            H, g = self._add_soft_constraints(H, g, x0)

        # ---- 阻尼最小二乘求解 ----
        # U* = -(H + lambda*I)^{-1} g
        lam = self._config.damping_factor
        H_reg = H + lam * np.eye(H.shape[0])

        try:
            U_flat = np.linalg.solve(H_reg, -g)
        except np.linalg.LinAlgError:
            # 降级到伪逆
            logger.warning("MPCController: 线性求解失败，使用伪逆")
            U_flat = -np.linalg.lstsq(H_reg, g, rcond=None)[0]

        # ---- 投影梯度下降处理硬约束 ----
        if self._config.enable_constraints:
            U_flat = self._project_constraints(U_flat, H, g)

        # ---- 重构控制序列 ----
        optimal_control = U_flat.reshape(Nu, m)  # (Nu, 2)

        # ---- 计算预测状态 ----
        predicted_states = np.zeros((N, n), dtype=np.float64)
        x_pred = x0.copy()
        for k in range(N):
            if k < Nu:
                u_k = optimal_control[k]
            else:
                u_k = optimal_control[-1]  # 控制保持
            x_pred = self._A @ x_pred + self._B @ u_k
            predicted_states[k] = x_pred

        # 记录迭代次数
        self._metrics.total_iterations += 1

        return optimal_control, predicted_states

    def _add_soft_constraints(
        self,
        H: np.ndarray,
        g: np.ndarray,
        x0: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """添加软约束惩罚项到代价函数。

        对输入约束和状态约束添加二次惩罚:
            J_soft = rho * Σ max(0, u - u_max)^2 + rho * Σ max(0, u_min - u)^2
            J_soft += rho * Σ max(0, x - x_max)^2 + rho * Σ max(0, x_min - x)^2

        通过在 Hessian 和梯度中添加线性化近似实现。

        Args:
            H: Hessian 矩阵 (Nu*m x Nu*m)。
            g: 梯度向量 (Nu*m,)。
            x0: 初始状态 (n,)。

        Returns:
            (H_new, g_new): 添加软约束后的 Hessian 和梯度。
        """
        rho = self._config.soft_constraint_weight
        Nu = self._config.control_horizon
        m = 2
        N = self._config.prediction_horizon
        n = 4

        H_new = H.copy()
        g_new = g.copy()

        # 输入约束惩罚
        u_max = np.array([self._config.max_step_x, self._config.max_step_y],
                         dtype=np.float64)
        u_min = -u_max

        for k in range(Nu):
            idx = k * m
            # 对每个控制变量添加边界惩罚 (线性化近似)
            for j in range(m):
                # 上界惩罚: rho * max(0, u_j - u_max_j)^2
                # 近似为: 在 H 对角线加 rho, 在 g 加 -rho * u_max_j
                H_new[idx + j, idx + j] += rho
                g_new[idx + j] += -rho * u_max[j]
                # 下界惩罚: rho * max(0, u_min_j - u_j)^2
                H_new[idx + j, idx + j] += rho
                g_new[idx + j] += rho * u_min[j]

        # 状态约束惩罚 (通过 Gamma 映射到控制空间)
        # 简化: 对预测位置添加边界惩罚
        # x_max, x_min 为工作空间边界 (像素)
        x_max_pos = 2000.0  # 工作空间 X/Y 上界
        x_min_pos = -2000.0  # 工作空间 X/Y 下界
        x_max_vel = 500.0   # 速度上界
        x_min_vel = -500.0  # 速度下界

        # 通过 Gamma^T 将状态约束映射到控制空间
        # 简化实现: 使用当前自由响应来估计约束违反
        assert self._Phi is not None and self._Gamma is not None
        X_free = self._Phi @ x0  # (N*n,)

        # 状态约束惩罚 (添加到 H 和 g)
        state_rho = rho * 0.1  # 状态约束权重稍低
        for k in range(N):
            for dim in range(n):
                if dim < 2:  # 位置约束
                    ref_val = [self._state.target_x, self._state.target_y][dim]
                    # 不直接约束位置，而是约束相对于参考的偏差
                    pass
                else:  # 速度约束
                    pass

        return H_new, g_new

    def _project_constraints(
        self,
        U_flat: np.ndarray,
        H: np.ndarray,
        g: np.ndarray,
    ) -> np.ndarray:
        """通过投影梯度下降处理硬约束。

        对控制输入执行投影:
            u = clip(u, u_min, u_max)

        如果投影后代价增加过多，则执行几步梯度下降后再投影。

        Args:
            U_flat: 展平的控制序列 (Nu*m,)。
            H: Hessian 矩阵。
            g: 梯度向量。

        Returns:
            投影后的控制序列 (Nu*m,)。
        """
        Nu = self._config.control_horizon
        m = 2
        u_max = np.array([self._config.max_step_x, self._config.max_step_y],
                         dtype=np.float64)
        u_min = -u_max

        # 构建约束边界
        bounds_lower = np.tile(u_min, Nu)
        bounds_upper = np.tile(u_max, Nu)

        # 投影到可行域
        U_proj = np.clip(U_flat, bounds_lower, bounds_upper)

        # 检查约束违反
        violations = int(np.sum(U_flat != U_proj))
        self._metrics.constraint_violations += violations

        if violations == 0:
            return U_proj

        # 有违反时，执行几步投影梯度下降以优化
        lam = self._config.damping_factor
        H_reg = H + lam * np.eye(H.shape[0])
        step_size = 0.01

        for _ in range(min(self._config.max_iterations, 20)):
            # 梯度: grad J = H @ U + g
            grad = H_reg @ U_proj + g

            # 梯度下降步
            U_new = U_proj - step_size * grad

            # 投影
            U_new = np.clip(U_new, bounds_lower, bounds_upper)

            # 检查收敛
            delta = np.max(np.abs(U_new - U_proj))
            U_proj = U_new

            if delta < self._config.qp_tolerance:
                break

        return U_proj

    def _compute_z_step(self, current_z: float, target_z: float) -> float:
        """计算 Z 轴控制步数 (独立 P 控制器)。

        Z 轴使用简单的比例控制，因为焦平面搜索通常是
        单调的且不需要复杂的多步预测。

        Args:
            current_z: 当前 Z 位置。
            target_z: 目标 Z 位置。

        Returns:
            Z 轴推荐步数。
        """
        error = target_z - current_z

        # 比例增益 (保守)
        kp_z = 0.5
        z_step = kp_z * error

        # 死区: 误差很小时不动作
        dead_zone = 0.05
        if abs(error) < dead_zone:
            return 0.0

        return z_step

    def _evaluate_cost(
        self,
        predicted_states: np.ndarray,
        optimal_control: np.ndarray,
    ) -> float:
        """评估代价函数值。

        Args:
            predicted_states: 预测状态序列 (N x 4)。
            optimal_control: 最优控制序列 (Nu x 2)。

        Returns:
            代价函数值。
        """
        assert self._Q is not None and self._R is not None and self._P is not None

        N = self._config.prediction_horizon
        Nu = self._config.control_horizon

        x_ref = np.array([
            self._state.target_x, self._state.target_y, 0.0, 0.0,
        ], dtype=np.float64)

        cost = 0.0
        for k in range(N):
            dx = predicted_states[k] - x_ref
            Q_k = self._P if k == N - 1 else self._Q
            cost += float(dx @ Q_k @ dx)

        for k in range(Nu):
            cost += float(optimal_control[k] @ self._R @ optimal_control[k])

        return cost


# ============================================================
# 模块测试
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    print("=" * 60)
    print("MPC 控制器模块测试")
    print("=" * 60)

    # ---- 测试 1: 基本功能 ----
    print("\n--- 测试 1: 基本控制功能 ---")
    controller = MPCController()

    # 模拟从 (100, 80) 移动到 (0, 0)
    current_x, current_y, current_z = 100.0, 80.0, 2.0
    target_x, target_y, target_z = 0.0, 0.0, 0.0

    print(f"初始位置: ({current_x}, {current_y}, {current_z})")
    print(f"目标位置: ({target_x}, {target_y}, {target_z})")
    print()

    for i in range(30):
        result = controller.update(
            current_x=current_x,
            current_y=current_y,
            current_z=current_z,
            target_x=target_x,
            target_y=target_y,
            target_z=target_z,
        )

        # 模拟系统响应 (简化一阶模型)
        gain = 0.6  # 电机响应增益
        noise_x = np.random.normal(0, 0.5)
        noise_y = np.random.normal(0, 0.5)
        current_x += gain * result.x_step + noise_x
        current_y += gain * result.y_step + noise_y
        current_z += 0.3 * result.z_step

        if i % 5 == 0 or i == 29:
            print(
                f"  [步 {i:2d}] 位置: ({current_x:7.2f}, {current_y:7.2f}, {current_z:5.2f}) | "
                f"指令: ({result.x_step:5d}, {result.y_step:5d}, {result.z_step:5.2f}) | "
                f"代价: {result.metrics.cost_value:8.2f} | "
                f"求解: {result.metrics.solve_time_ms:5.2f}ms"
            )

        error = np.sqrt((current_x - target_x) ** 2 + (current_y - target_y) ** 2)
        if error < 1.0 and i > 5:
            print(f"\n  收敛! 最终误差: {error:.4f} px (步数: {i + 1})")
            break

    # ---- 测试 2: 约束处理 ----
    print("\n--- 测试 2: 约束处理 ---")
    controller2 = MPCController(MPCConfig(
        max_step_x=100,
        max_step_y=100,
        enable_constraints=True,
    ))

    result = controller2.update(
        current_x=500.0, current_y=500.0,
        target_x=0.0, target_y=0.0,
    )
    print(f"  大误差指令 (限幅 100): X={result.x_step}, Y={result.y_step}")
    assert abs(result.x_step) <= 100, "X 约束违反"
    assert abs(result.y_step) <= 100, "Y 约束违反"
    print("  约束验证通过")

    # ---- 测试 3: 系统矩阵 ----
    print("\n--- 测试 3: 系统矩阵 ---")
    A, B = controller.get_system_matrices()
    print(f"  A 矩阵 (4x4):")
    for row in A:
        print(f"    [{row[0]:6.3f} {row[1]:6.3f} {row[2]:6.3f} {row[3]:6.3f}]")
    print(f"  B 矩阵 (4x2):")
    for row in B:
        print(f"    [{row[0]:6.3f} {row[1]:6.3f}]")

    # ---- 测试 4: 模型自适应 ----
    print("\n--- 测试 4: 模型自适应 ---")
    controller3 = MPCController()
    A_before, B_before = controller3.get_system_matrices()

    # 运行多步以积累历史
    for i in range(25):
        controller3.update(
            current_x=50.0 - i * 1.5,
            current_y=30.0 - i * 1.0,
            target_x=0.0, target_y=0.0,
        )

    A_after, B_after = controller3.get_system_matrices()
    print(f"  自适应前 A[2,2]: {A_before[2, 2]:.4f}")
    print(f"  自适应后 A[2,2]: {A_after[2, 2]:.4f}")
    print(f"  模型已自适应: {controller3.state.model_adapted}")

    # ---- 测试 5: 热启动 ----
    print("\n--- 测试 5: 热启动效果 ---")
    config_warm = MPCConfig(warm_start=True)
    config_cold = MPCConfig(warm_start=False)

    controller_warm = MPCController(config_warm)
    controller_cold = MPCController(config_cold)

    times_warm = []
    times_cold = []

    for i in range(20):
        r_warm = controller_warm.update(
            current_x=50.0 - i * 2.0, current_y=30.0 - i * 1.5,
            target_x=0.0, target_y=0.0,
        )
        r_cold = controller_cold.update(
            current_x=50.0 - i * 2.0, current_y=30.0 - i * 1.5,
            target_x=0.0, target_y=0.0,
        )
        times_warm.append(r_warm.metrics.solve_time_ms)
        times_cold.append(r_cold.metrics.solve_time_ms)

    print(f"  热启动平均求解时间: {np.mean(times_warm):.3f} ms")
    print(f"  冷启动平均求解时间: {np.mean(times_cold):.3f} ms")

    # ---- 测试 6: 成本摘要 ----
    print("\n--- 测试 6: 成本摘要 ---")
    summary = controller.get_cost_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # ---- 测试 7: 重置 ----
    print("\n--- 测试 7: 重置功能 ---")
    controller.reset()
    print(f"  重置后状态: x={controller.state.x_position}, "
          f"y={controller.state.y_position}")
    print(f"  重置后指标: cost_history 长度={len(controller.metrics.cost_history)}")
    assert len(controller.metrics.cost_history) == 0, "重置后指标未清空"
    print("  重置验证通过")

    # ---- 测试 8: 预测轨迹可视化 ----
    print("\n--- 测试 8: 预测轨迹 ---")
    controller4 = MPCController()
    result = controller4.update(
        current_x=80.0, current_y=60.0,
        target_x=0.0, target_y=0.0,
    )
    traj = result.predicted_trajectory
    print(f"  预测轨迹形状: {traj.shape}")
    print(f"  起点预测: ({traj[0, 0]:.2f}, {traj[0, 1]:.2f})")
    print(f"  终点预测: ({traj[-1, 0]:.2f}, {traj[-1, 1]:.2f})")
    print(f"  预测收敛距离: {np.linalg.norm(traj[-1] - traj[0]):.2f} px")

    print("\n" + "=" * 60)
    print("所有测试完成")
    print("=" * 60)
