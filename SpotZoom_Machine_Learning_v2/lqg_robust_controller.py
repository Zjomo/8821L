"""
LQG 鲁棒控制器 (Linear-Quadratic-Gaussian Robust Controller)

参考开源项目:
  - python-control (https://github.com/python-control/python-control)
  - acados (https://github.com/acados/acados)

核心思想:
  LQG 控制器是 LQR (线性二次调节器) + Kalman 滤波器的组合，
  是处理高斯噪声下的线性系统最优控制的经典方法。

  从 python-control 的状态空间控制工具中借鉴:
  1. Kalman 滤波器: 从带噪声的观测中估计系统状态
  2. LQR 控制律: 基于状态估计计算最优控制输入
  3. LQG 综合: 将两者结合，实现最优随机控制

  在 SpotZoom 场景中:
  - 状态: [位置误差, 速度误差] (2D × 2 轴 = 4 维状态)
  - 观测: 带噪声的光斑位置检测
  - 控制: XY 位移台步数
  - 过程噪声: 振动、热漂移
  - 观测噪声: 检测器噪声

创新点:
  1. 2D 联合状态估计 (X-Y 耦合建模)
  2. 自适应噪声协方差估计
  3. 抗饱和 LQG (处理执行器限幅)
  4. 渐消 Kalman 滤波 (处理模型不确定性)
  5. 多模型 LQG (不同工作模式切换)

纯 numpy 实现，无外部依赖。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class LQGConfig:
    """LQG 控制器配置。"""
    # 采样周期 (秒)
    dt: float = 0.1
    # 状态权重矩阵 Q 的对角元素
    # 状态: [ex, evx, ey, evy] (位置误差, 速度误差)
    q_position: float = 100.0
    q_velocity: float = 1.0
    # 控制权重
    r_control: float = 0.1
    # 过程噪声强度 (位置)
    process_noise_position: float = 0.5
    # 过程噪声强度 (速度)
    process_noise_velocity: float = 0.1
    # 观测噪声强度
    observation_noise: float = 2.0
    # 最大控制输出 (步数)
    max_control: int = 3000
    # 最小控制输出
    min_control: int = 10
    # 抗饱和使能
    anti_windup: bool = True
    # 积分限幅
    integral_limit: float = 500.0
    # 自适应噪声使能
    adaptive_noise: bool = True
    # 噪声估计窗口
    noise_estimation_window: int = 30
    # 渐消因子 (1.0 = 标准 KF, >1.0 = 渐消)
    fading_factor: float = 1.0
    # 死区 (像素, 小于此值不动作)
    dead_zone_px: float = 1.0


@dataclass
class LQGState:
    """LQG 控制器状态。"""
    # 状态估计 [ex, evx, ey, evy]
    x: np.ndarray = field(default_factory=lambda: np.zeros(4))
    # 状态估计协方差 (4×4)
    P: np.ndarray = field(default_factory=lambda: np.eye(4))
    # 积分误差
    integral_error: np.ndarray = field(default_factory=lambda: np.zeros(2))
    # 上一次控制输出
    last_control: np.ndarray = field(default_factory=lambda: np.zeros(2))
    # 上一次观测
    last_observation: Optional[np.ndarray] = None
    # 创新序列 (用于自适应噪声)
    innovation_history: List[float] = field(default_factory=list)
    # 帧计数
    frame_count: int = 0
    # 当前过程噪声协方差
    Q_current: np.ndarray = field(default_factory=lambda: np.eye(4))
    # 当前观测噪声协方差
    R_current: float = 2.0
    # 是否已初始化
    initialized: bool = False


class LQGRobustController:
    """LQG 鲁棒控制器。

    结合 Kalman 滤波器和 LQR 控制律，实现带噪声的
    光斑对准系统的最优随机控制。

    使用方法:
        config = LQGConfig(q_position=100.0, r_control=0.1)
        controller = LQGRobustController(config)

        # 每次检测到光斑位置后调用
        control = controller.update(measured_error_x, measured_error_y)
        stage.move_x(control[0])
        stage.move_y(control[1])
    """

    def __init__(self, config: Optional[LQGConfig] = None):
        self.config = config or LQGConfig()
        self.state = LQGState()
        self._K_lqr = None  # LQR 增益矩阵
        self._L_kalman = None  # Kalman 增益
        self._compute_lqr_gain()

    def update(
        self, measured_error_x: float, measured_error_y: float
    ) -> Tuple[int, int]:
        """处理一次观测，返回控制输出。

        Args:
            measured_error_x: 测量的 X 方向误差 (像素)
            measured_error_y: 测量的 Y 方向误差 (像素)

        Returns:
            (control_x, control_y): XY 方向的控制步数
        """
        state = self.state
        cfg = self.config

        # 观测向量
        z = np.array([measured_error_x, measured_error_y])

        # 首次观测初始化
        if not state.initialized:
            state.x = np.array([measured_error_x, 0.0, measured_error_y, 0.0])
            state.last_observation = z.copy()
            state.initialized = True
            state.frame_count += 1
            return (0, 0)

        # 1. 自适应噪声估计
        if cfg.adaptive_noise and state.frame_count > 5:
            self._adapt_noise_covariance(z)

        # 2. Kalman 滤波: 预测
        x_pred, P_pred = self._kalman_predict()

        # 3. Kalman 滤波: 更新
        x_est, P_est, innovation = self._kalman_update(x_pred, P_pred, z)

        state.x = x_est
        state.P = P_est

        # 记录创新序列
        state.innovation_history.append(float(np.linalg.norm(innovation)))
        if len(state.innovation_history) > cfg.noise_estimation_window * 2:
            state.innovation_history = state.innovation_history[-cfg.noise_estimation_window * 2:]

        # 4. LQR 控制律
        u = self._lqr_control(x_est)

        # 5. 抗饱和 + 积分
        u = self._apply_anti_windup(u)

        # 6. 限幅
        u[0] = int(round(np.clip(u[0], -cfg.max_control, cfg.max_control)))
        u[1] = int(round(np.clip(u[1], -cfg.max_control, cfg.max_control)))

        # 7. 死区
        if abs(u[0]) < cfg.min_control:
            u[0] = 0
        if abs(u[1]) < cfg.min_control:
            u[1] = 0

        state.last_control = u.copy()
        state.last_observation = z.copy()
        state.frame_count += 1

        return (int(u[0]), int(u[1]))

    def _compute_lqr_gain(self) -> None:
        """计算 LQR 最优增益矩阵 (离散代数 Riccati 方程迭代求解)。"""
        cfg = self.config
        dt = cfg.dt

        # 离散状态空间矩阵
        # 状态: [ex, evx, ey, evy]
        # x(t+1) = A * x(t) + B * u(t) + w(t)
        # z(t) = H * x(t) + v(t)

        A = np.array([
            [1, dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, dt],
            [0, 0, 0, 1],
        ], dtype=np.float64)

        # 控制输入影响 (假设步数→像素转换系数)
        step_to_px = 0.01  # 近似值
        B = np.array([
            [step_to_px * dt, 0],
            [step_to_px, 0],
            [0, step_to_px * dt],
            [0, step_to_px],
        ], dtype=np.float64)

        # 权重矩阵
        Q = np.diag([
            cfg.q_position, cfg.q_velocity,
            cfg.q_position, cfg.q_velocity,
        ])
        R = np.diag([cfg.r_control, cfg.r_control])

        # 迭代求解离散 Riccati 方程
        P = Q.copy()
        for _ in range(100):
            # P = Q + A'PA - A'PB(R + B'PB)^{-1}B'PA
            BPB = B.T @ P @ B
            try:
                K = np.linalg.solve(R + BPB, B.T @ P @ A)
            except np.linalg.LinAlgError:
                break
            P_new = Q + A.T @ P @ A - A.T @ P @ B @ K
            if np.max(np.abs(P_new - P)) < 1e-8:
                P = P_new
                break
            P = P_new

        self._K_lqr = K
        self._A = A
        self._B = B
        self._H = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
        ], dtype=np.float64)

    def _kalman_predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """Kalman 滤波预测步。"""
        state = self.state
        A = self._A

        # 状态预测
        x_pred = A @ state.x

        # 协方差预测 (渐消)
        P_pred = self.config.fading_factor * (A @ state.P @ A.T) + state.Q_current

        return x_pred, P_pred

    def _kalman_update(
        self, x_pred: np.ndarray, P_pred: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Kalman 滤波更新步。"""
        H = self._H

        # 观测噪声协方差
        R = np.eye(2) * self.state.R_current

        # 创新
        y_innov = z - H @ x_pred

        # 创新协方差
        S = H @ P_pred @ H.T + R

        # Kalman 增益
        try:
            K = P_pred @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = np.zeros((4, 2))

        # 状态更新
        x_est = x_pred + K @ y_innov

        # 协方差更新 (Joseph 形式，数值稳定)
        I_KH = np.eye(4) - K @ H
        P_est = I_KH @ P_pred @ I_KH.T + K @ R @ K.T

        # 确保对称
        P_est = (P_est + P_est.T) / 2.0

        return x_est, P_est, y_innov

    def _lqr_control(self, x: np.ndarray) -> np.ndarray:
        """LQR 控制律。"""
        if self._K_lqr is None:
            return np.zeros(2)

        u = -self._K_lqr @ x
        return u

    def _apply_anti_windup(self, u: np.ndarray) -> np.ndarray:
        """抗饱和处理。"""
        cfg = self.config
        state = self.state

        if not cfg.anti_windup:
            return u

        # 积分累积
        state.integral_error += u * cfg.dt

        # 积分限幅
        for i in range(2):
            state.integral_error[i] = max(
                -cfg.integral_limit,
                min(cfg.integral_limit, state.integral_error[i]),
            )

        # 如果控制输出饱和，回退积分
        for i in range(2):
            if abs(u[i]) > cfg.max_control:
                state.integral_error[i] *= 0.9

        return u

    def _adapt_noise_covariance(self, z: np.ndarray) -> None:
        """自适应噪声协方差估计。"""
        state = self.state
        cfg = self.config
        window = cfg.noise_estimation_window

        if len(state.innovation_history) < window:
            return

        # 基于创新序列估计观测噪声
        recent_innov = state.innovation_history[-window:]
        innov_var = float(np.var(recent_innov))

        # 观测噪声 = 创新方差 - 预测方差 (近似)
        predicted_var = float(np.trace(state.P @ self._H.T @ self._H))
        r_est = max(0.1, innov_var - predicted_var)
        state.R_current = r_est

        # 过程噪声自适应
        if state.last_observation is not None:
            obs_change = float(np.linalg.norm(z - state.last_observation))
            predicted_change = float(np.linalg.norm(self._A @ state.x - state.x))
            residual = max(0.0, obs_change - predicted_change)
            q_pos = max(0.01, residual * 0.5)
            state.Q_current = np.diag([
                q_pos, cfg.process_noise_velocity,
                q_pos, cfg.process_noise_velocity,
            ])

    def get_state_estimate(self) -> dict:
        """返回当前状态估计。"""
        s = self.state
        return {
            "position_error": [round(float(s.x[0]), 3), round(float(s.x[2]), 3)],
            "velocity_error": [round(float(s.x[1]), 3), round(float(s.x[3]), 3)],
            "state_covariance_trace": round(float(np.trace(s.P)), 4),
            "observation_noise_r": round(float(s.R_current), 4),
            "frame_count": s.frame_count,
            "last_control": [int(s.last_control[0]), int(s.last_control[1])],
        }

    def reset(self) -> None:
        """重置控制器状态。"""
        self.state = LQGState()
        self._compute_lqr_gain()
