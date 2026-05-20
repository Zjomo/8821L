"""
神经 ODE 控制器 (Neural ODE Controller)

基于神经常微分方程的连续时间控制信号生成模块。
使用 Euler/RK4 求解器实现纯 numpy 的 ODE 求解，
用于光斑运动预测和连续时间轨迹规划。

灵感来源:
- torchdiffeq (rtqichen/torchdiffeq): 神经常微分方程库
  (https://github.com/rtqichen/torchdiffeq)
- Latent ODE for time series (Rubanova et al. 2019)
- Neural ODE (Chen et al. 2018): 神经网络作为 ODE 右端项

算法原理:
  神经 ODE 将连续时间动态建模为:
    dz/dt = f_θ(z, t, u)
  其中 f_θ 是一个参数化的向量场 (动力学模型)。

  1. 动力学模型: 使用线性/非线性映射近似光斑运动动力学
  2. ODE 求解: Euler 方法或 RK4 方法数值积分
  3. 连续轨迹: 在任意时间点查询状态
  4. 自适应步长: 根据局部误差估计调整步长
  5. 稳定性监控: 类 Lyapunov 检查确保控制信号稳定

外部依赖: numpy
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class ODESolver(Enum):
    """ODE 求解器枚举。"""
    EULER = "euler"       # 前向欧拉
    RK4 = "rk4"           # 四阶 Runge-Kutta
    MIDPOINT = "midpoint" # 中点法


@dataclass
class NeuralODEResult:
    """神经 ODE 控制结果。"""
    trajectory: np.ndarray = None          # 预测轨迹 (T, state_dim)
    time_points: np.ndarray = None         # 时间点 (T,)
    control_signals: np.ndarray = None     # 控制信号 (T, control_dim)
    final_state: np.ndarray = None         # 最终状态
    is_stable: bool = True                 # 是否稳定
    stability_metric: float = 0.0          # 稳定性指标
    integration_steps: int = 0             # 积分步数
    adaptive_steps_used: int = 0           # 自适应步数
    max_local_error: float = 0.0           # 最大局部误差
    processing_time_ms: float = 0.0        # 处理耗时 (ms)


@dataclass
class NeuralODEConfig:
    """神经 ODE 控制器配置。"""
    # ODE 求解器
    ode_solver: ODESolver = ODESolver.RK4
    step_size: float = 0.01                # 基础步长 (秒)
    max_steps: int = 1000                  # 最大积分步数

    # 稳定性
    stability_threshold: float = 100.0     # 稳定性阈值 (状态范数上限)
    lyapunov_check_interval: int = 10      # Lyapunov 检查间隔

    # 预测
    prediction_horizon: float = 1.0        # 预测时间范围 (秒)

    # 自适应步长
    adaptive_step_size: bool = True        # 启用自适应步长
    min_step_size: float = 1e-6            # 最小步长
    max_step_size: float = 0.1             # 最大步长
    error_tolerance: float = 1e-4          # 局部误差容限

    # 动力学模型
    state_dim: int = 4                     # 状态维度 (x, y, vx, vy)
    control_dim: int = 2                   # 控制维度 (fx, fy)
    dynamics_nonlinearity: bool = True     # 启用非线性动力学
    damping_factor: float = 0.1            # 阻尼因子

    # 数值参数
    numerical_stability: float = 1e-10     # 数值稳定性常数


class NeuralODEController:
    """神经 ODE 控制器。

    使用神经常微分方程生成连续时间控制信号。
    纯 numpy 实现，支持 Euler 和 RK4 求解器。

    使用示例:
        controller = NeuralODEController()
        # 设置初始状态
        controller.set_state(np.array([100.0, 200.0, 0.0, 0.0]))  # x, y, vx, vy
        # 预测未来轨迹
        result = controller.predict_trajectory()
        print(f"Trajectory shape: {result.trajectory.shape}")
        print(f"Stable: {result.is_stable}")
        # 生成控制信号
        result = controller.compute_control(target=np.array([0.0, 0.0]))
    """

    def __init__(self, config: Optional[NeuralODEConfig] = None):
        self._config = config or NeuralODEConfig()
        self._state: Optional[np.ndarray] = None
        self._dynamics_params: Optional[dict] = None
        self._time: float = 0.0
        self._initialized = False

    @property
    def config(self) -> NeuralODEConfig:
        return self._config

    def set_state(self, state: np.ndarray):
        """设置当前状态。

        Args:
            state: 状态向量 (state_dim,)
        """
        state = np.asarray(state, dtype=np.float64).ravel()
        cfg = self._config

        if state.shape[0] != cfg.state_dim:
            logger.warning(
                f"State dim mismatch: expected {cfg.state_dim}, got {state.shape[0]}"
            )
            padded = np.zeros(cfg.state_dim, dtype=np.float64)
            n = min(state.shape[0], cfg.state_dim)
            padded[:n] = state[:n]
            state = padded

        self._state = state.copy()
        self._time = 0.0

        if not self._initialized:
            self._initialize_dynamics()
            self._initialized = True

        logger.info(f"NeuralODEController: State set to {state}")

    def predict_trajectory(
        self,
        initial_state: Optional[np.ndarray] = None,
        horizon: Optional[float] = None,
    ) -> NeuralODEResult:
        """预测未来轨迹。

        Args:
            initial_state: 初始状态 (None=使用当前状态)
            horizon: 预测时间范围 (None=使用配置默认值)

        Returns:
            NeuralODEResult: 预测结果
        """
        import time
        t0 = time.perf_counter()

        cfg = self._config

        if not self._initialized:
            self._initialize_dynamics()
            self._initialized = True

        if initial_state is not None:
            self.set_state(initial_state)
        elif self._state is None:
            self._state = np.zeros(cfg.state_dim, dtype=np.float64)

        if horizon is None:
            horizon = cfg.prediction_horizon

        # ODE 积分
        trajectory, time_points, control_signals, steps = self._integrate(
            self._state.copy(), 0.0, horizon, np.zeros(cfg.control_dim)
        )

        # 稳定性检查
        is_stable, stability_metric = self._check_stability(trajectory)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        result = NeuralODEResult(
            trajectory=trajectory,
            time_points=time_points,
            control_signals=control_signals,
            final_state=trajectory[-1],
            is_stable=is_stable,
            stability_metric=stability_metric,
            integration_steps=steps,
            processing_time_ms=elapsed_ms,
        )

        logger.debug(
            f"NeuralODEController: predicted {steps} steps, "
            f"stable={is_stable}, stability={stability_metric:.4f}, "
            f"time={elapsed_ms:.1f}ms"
        )

        return result

    def compute_control(
        self,
        target: np.ndarray,
        initial_state: Optional[np.ndarray] = None,
    ) -> NeuralODEResult:
        """计算控制信号 (驱使状态趋向目标)。

        使用 PD 控制器作为 ODE 右端项的一部分，
        生成驱使光斑从当前位置移向目标位置的控制信号。

        Args:
            target: 目标状态 (至少 2 维: x, y)
            initial_state: 初始状态 (None=使用当前状态)

        Returns:
            NeuralODEResult: 控制结果
        """
        import time
        t0 = time.perf_counter()

        cfg = self._config

        if not self._initialized:
            self._initialize_dynamics()
            self._initialized = True

        if initial_state is not None:
            self.set_state(initial_state)
        elif self._state is None:
            self._state = np.zeros(cfg.state_dim, dtype=np.float64)

        target = np.asarray(target, dtype=np.float64).ravel()
        target_state = np.zeros(cfg.state_dim, dtype=np.float64)
        n = min(target.shape[0], cfg.state_dim)
        target_state[:n] = target[:n]

        # 带目标的 ODE 积分
        trajectory, time_points, control_signals, steps = self._integrate_with_target(
            self._state.copy(), target_state, 0.0, cfg.prediction_horizon
        )

        is_stable, stability_metric = self._check_stability(trajectory)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        result = NeuralODEResult(
            trajectory=trajectory,
            time_points=time_points,
            control_signals=control_signals,
            final_state=trajectory[-1],
            is_stable=is_stable,
            stability_metric=stability_metric,
            integration_steps=steps,
            processing_time_ms=elapsed_ms,
        )

        return result

    def _initialize_dynamics(self):
        """初始化动力学模型参数。

        动力学模型: dz/dt = A*z + B*u + g(z)
        其中 g(z) 是非线性项。
        """
        cfg = self._config
        n = cfg.state_dim
        m = cfg.control_dim

        # 状态矩阵 A (阻尼谐振子模型)
        # [0, 0, 1, 0]     位置导数 = 速度
        # [0, 0, 0, 1]     位置导数 = 速度
        # [-k, 0, -c, 0]   加速度 = -k*x - c*vx
        # [0, -k, 0, -c]   加速度 = -k*y - c*vy
        A = np.zeros((n, n), dtype=np.float64)
        stiffness = 1.0
        damping = cfg.damping_factor

        for i in range(n // 2):
            A[i, n // 2 + i] = 1.0  # 位置 -> 速度
            A[n // 2 + i, i] = -stiffness  # 位置 -> 加速度 (回复力)
            A[n // 2 + i, n // 2 + i] = -damping  # 速度 -> 加速度 (阻尼)

        # 输入矩阵 B
        B = np.zeros((n, m), dtype=np.float64)
        for i in range(m):
            B[n // 2 + i, i] = 1.0  # 控制力影响加速度

        # 非线性参数
        if cfg.dynamics_nonlinearity:
            # 非线性刚度 (Duffing 振子)
            nonlin_coeff = 0.01
        else:
            nonlin_coeff = 0.0

        self._dynamics_params = {
            'A': A, 'B': B,
            'nonlin_coeff': nonlin_coeff,
            'stiffness': stiffness,
            'damping': damping,
        }

        logger.info(
            f"NeuralODEController: Dynamics initialized, "
            f"state_dim={n}, control_dim={m}"
        )

    def _dynamics(
        self, state: np.ndarray, t: float, control: np.ndarray
    ) -> np.ndarray:
        """计算状态导数 (ODE 右端项)。

        dz/dt = A*z + B*u + g(z)

        Args:
            state: 当前状态 (state_dim,)
            t: 当前时间
            control: 控制输入 (control_dim,)

        Returns:
            状态导数 (state_dim,)
        """
        params = self._dynamics_params
        A = params['A']
        B = params['B']

        # 线性部分
        dz = A @ state + B @ control

        # 非线性部分 (Duffing 型)
        if params['nonlin_coeff'] > 0:
            n = state.shape[0]
            for i in range(n // 2):
                dz[n // 2 + i] -= params['nonlin_coeff'] * state[i] ** 3

        return dz

    def _dynamics_with_target(
        self, state: np.ndarray, t: float, target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """带目标状态的动力学 (PD 控制)。

        Args:
            state: 当前状态
            t: 当前时间
            target: 目标状态

        Returns:
            (state_derivative, control_signal)
        """
        cfg = self._config

        # PD 控制器
        Kp = 10.0  # 比例增益
        Kd = 5.0   # 微分增益

        # 位置误差
        error = target - state
        pos_error = error[:cfg.state_dim // 2]
        vel_error = error[cfg.state_dim // 2:] if cfg.state_dim >= 4 else np.zeros(cfg.control_dim)

        # 控制信号
        control = Kp * pos_error + Kd * vel_error

        # 限制控制幅度
        max_control = 50.0
        control = np.clip(control, -max_control, max_control)

        # 状态导数
        dz = self._dynamics(state, t, control)

        return dz, control

    def _integrate(
        self,
        initial_state: np.ndarray,
        t_start: float,
        t_end: float,
        control: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """ODE 积分 (无目标)。

        Args:
            initial_state: 初始状态
            t_start: 起始时间
            t_end: 结束时间
            control: 控制信号

        Returns:
            (trajectory, time_points, control_signals, steps)
        """
        cfg = self._config
        solver = cfg.ode_solver

        trajectory = [initial_state.copy()]
        time_points = [t_start]
        control_signals = [control.copy()]

        state = initial_state.copy()
        t = t_start
        steps = 0
        adaptive_count = 0

        while t < t_end and steps < cfg.max_steps:
            if cfg.adaptive_step_size:
                dt = self._compute_adaptive_step(state, t, control)
                adaptive_count += 1
            else:
                dt = cfg.step_size

            dt = min(dt, t_end - t)
            if dt < cfg.numerical_stability:
                break

            # ODE 求解步
            if solver == ODESolver.EULER:
                new_state = self._euler_step(state, t, dt, control)
            elif solver == ODESolver.RK4:
                new_state = self._rk4_step(state, t, dt, control)
            elif solver == ODESolver.MIDPOINT:
                new_state = self._midpoint_step(state, t, dt, control)
            else:
                new_state = self._euler_step(state, t, dt, control)

            state = new_state
            t += dt
            steps += 1

            trajectory.append(state.copy())
            time_points.append(t)
            control_signals.append(control.copy())

            # 稳定性检查
            if steps % cfg.lyapunov_check_interval == 0:
                if np.linalg.norm(state) > cfg.stability_threshold:
                    logger.warning(
                        f"NeuralODEController: State diverged at step {steps}, "
                        f"norm={np.linalg.norm(state):.2f}"
                    )
                    break

        return (
            np.array(trajectory),
            np.array(time_points),
            np.array(control_signals),
            steps,
        )

    def _integrate_with_target(
        self,
        initial_state: np.ndarray,
        target: np.ndarray,
        t_start: float,
        t_end: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """带目标的 ODE 积分。

        Args:
            initial_state: 初始状态
            target: 目标状态
            t_start: 起始时间
            t_end: 结束时间

        Returns:
            (trajectory, time_points, control_signals, steps)
        """
        cfg = self._config

        trajectory = [initial_state.copy()]
        time_points = [t_start]
        _, initial_control = self._dynamics_with_target(initial_state, t_start, target)
        control_signals = [initial_control.copy()]

        state = initial_state.copy()
        t = t_start
        steps = 0

        while t < t_end and steps < cfg.max_steps:
            dt = cfg.step_size
            dt = min(dt, t_end - t)

            # RK4 步 (带目标)
            k1, u1 = self._dynamics_with_target(state, t, target)
            k2, _ = self._dynamics_with_target(state + 0.5 * dt * k1, t + 0.5 * dt, target)
            k3, _ = self._dynamics_with_target(state + 0.5 * dt * k2, t + 0.5 * dt, target)
            k4, u4 = self._dynamics_with_target(state + dt * k3, t + dt, target)

            new_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            control = u1  # 使用当前步的控制信号

            state = new_state
            t += dt
            steps += 1

            trajectory.append(state.copy())
            time_points.append(t)
            control_signals.append(control.copy())

            # 稳定性检查
            if steps % cfg.lyapunov_check_interval == 0:
                if np.linalg.norm(state) > cfg.stability_threshold:
                    logger.warning(f"NeuralODEController: Diverged at step {steps}")
                    break

        return (
            np.array(trajectory),
            np.array(time_points),
            np.array(control_signals),
            steps,
        )

    def _euler_step(
        self, state: np.ndarray, t: float, dt: float, control: np.ndarray
    ) -> np.ndarray:
        """前向欧拉步。"""
        return state + dt * self._dynamics(state, t, control)

    def _rk4_step(
        self, state: np.ndarray, t: float, dt: float, control: np.ndarray
    ) -> np.ndarray:
        """四阶 Runge-Kutta 步。"""
        k1 = self._dynamics(state, t, control)
        k2 = self._dynamics(state + 0.5 * dt * k1, t + 0.5 * dt, control)
        k3 = self._dynamics(state + 0.5 * dt * k2, t + 0.5 * dt, control)
        k4 = self._dynamics(state + dt * k3, t + dt, control)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _midpoint_step(
        self, state: np.ndarray, t: float, dt: float, control: np.ndarray
    ) -> np.ndarray:
        """中点法步。"""
        k1 = self._dynamics(state, t, control)
        k2 = self._dynamics(state + 0.5 * dt * k1, t + 0.5 * dt, control)
        return state + dt * k2

    def _compute_adaptive_step(
        self, state: np.ndarray, t: float, control: np.ndarray
    ) -> float:
        """计算自适应步长。

        基于局部误差估计 (Euler vs RK2 差异)。

        Args:
            state: 当前状态
            t: 当前时间
            control: 控制输入

        Returns:
            自适应步长
        """
        cfg = self._config

        # Euler 估计
        euler_next = self._euler_step(state, t, cfg.step_size, control)

        # RK2 (中点法) 估计
        midpoint_next = self._midpoint_step(state, t, cfg.step_size, control)

        # 局部误差
        local_error = np.linalg.norm(euler_next - midpoint_next)

        # 步长调整
        if local_error < cfg.numerical_stability:
            new_dt = cfg.max_step_size
        else:
            desired_error = cfg.error_tolerance
            ratio = desired_error / (local_error + cfg.numerical_stability)
            # 保守调整 (幂律)
            new_dt = cfg.step_size * min(2.0, max(0.1, ratio ** 0.5))

        return float(np.clip(new_dt, cfg.min_step_size, cfg.max_step_size))

    def _check_stability(
        self, trajectory: np.ndarray
    ) -> Tuple[bool, float]:
        """检查轨迹稳定性 (类 Lyapunov 方法)。

        计算轨迹的"能量"是否递减:
        V(z) = 0.5 * z^T P z
        其中 P = I (简化版)

        Args:
            trajectory: 轨迹 (T, state_dim)

        Returns:
            (is_stable, stability_metric)
        """
        cfg = self._config

        # 计算每个时间步的"能量"
        energies = 0.5 * np.sum(trajectory ** 2, axis=1)

        # 检查能量是否发散
        max_energy = np.max(energies)
        is_stable = max_energy < cfg.stability_threshold

        # 稳定性指标: 能量变化率
        if len(energies) > 1:
            energy_diffs = np.diff(energies)
            # 正变化比例 (越小越稳定)
            positive_ratio = np.sum(energy_diffs > 0) / len(energy_diffs)
            stability_metric = 1.0 - positive_ratio
        else:
            stability_metric = 1.0

        return is_stable, float(stability_metric)

    def reset(self):
        """重置控制器状态。"""
        self._state = None
        self._time = 0.0
        self._initialized = False
        logger.info("NeuralODEController: Reset")

    def get_state(self) -> Optional[np.ndarray]:
        """获取当前状态。"""
        return self._state.copy() if self._state is not None else None
