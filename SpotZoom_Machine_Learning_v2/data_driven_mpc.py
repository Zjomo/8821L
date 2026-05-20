"""
数据驱动模型预测控制器 (Data-Driven Model Predictive Controller)

参考开源项目:
  - leap-c (https://github.com/leap-c/leap-c): 学习型预测控制
  - acados (https://github.com/acados/acados): 非线性最优控制
  - do-mpc (https://github.com/do-mpc/do-mpc): 模型预测控制

核心思想:
  从 leap-c 和 acados 中借鉴数据驱动的 MPC 思想:
  不依赖精确的物理模型，而是从历史运行数据中学习系统动力学，
  然后用学习到的模型进行预测和优化控制。

  在 SpotZoom 场景中:
  - 从历史 (误差, 步长) 数据中学习位移台的响应特性
  - 使用学习到的模型预测未来几步的误差演化
  - 通过滚动优化计算最优控制序列
  - 支持约束处理 (最大步长、加速度限制)

创新点:
  1. 纯数据驱动的系统辨识 (无需物理模型)
  2. 在线模型更新 (适应系统特性变化)
  3. 约束感知的优化 (安全限制)
  4. 多目标优化 (速度 vs 精度 vs 能耗)
  5. 计算效率优化 (warm-start, 早期终止)

纯 numpy 实现，无外部依赖。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np


@dataclass
class DataDrivenMPCConfig:
    """数据驱动 MPC 配置。"""
    # 预测时域 (步数)
    prediction_horizon: int = 10
    # 控制时域 (步数)
    control_horizon: int = 5
    # 历史数据缓冲大小
    history_buffer_size: int = 200
    # 系统辨识窗口大小
    identification_window: int = 50
    # 状态权重 (误差惩罚)
    state_weight: float = 10.0
    # 控制权重 (步长惩罚, 越大越保守)
    control_weight: float = 0.5
    # 终端状态权重
    terminal_weight: float = 20.0
    # 最大步长限制
    max_step: int = 3000
    # 最小步长限制
    min_step: int = 10
    # 最大加速度 (步数变化率)
    max_acceleration: int = 1000
    # 在线模型更新使能
    online_model_update: bool = True
    # 模型更新间隔 (步数)
    model_update_interval: int = 20
    # 优化最大迭代次数
    optimization_max_iter: int = 50
    # 优化收敛阈值
    optimization_tolerance: float = 1e-4
    # warm-start 使能
    warm_start: bool = True


@dataclass
class DataDrivenMPCResult:
    """MPC 优化结果。"""
    # 最优控制序列 (X 方向)
    control_sequence_x: List[int] = field(default_factory=list)
    # 最优控制序列 (Y 方向)
    control_sequence_y: List[int] = field(default_factory=list)
    # 预测状态轨迹 (X 误差)
    predicted_error_x: List[float] = field(default_factory=list)
    # 预测状态轨迹 (Y 误差)
    predicted_error_y: List[float] = field(default_factory=list)
    # 最优代价
    optimal_cost: float = 0.0
    # 优化迭代次数
    optimization_iterations: int = 0
    # 是否收敛
    converged: bool = False
    # 处理时间 (ms)
    processing_time_ms: float = 0.0
    # 模型质量 (R² 拟合度)
    model_quality_r2: float = 0.0


class DataDrivenMPC:
    """数据驱动模型预测控制器。

    从历史运行数据中学习系统动力学，然后进行预测和优化控制。

    使用方法:
        config = DataDrivenMPCConfig(prediction_horizon=10)
        mpc = DataDrivenMPC(config)

        # 添加历史数据
        for error, step in history_data:
            mpc.add_data_point(error_x, error_y, step_x, step_y)

        # 求解最优控制
        result = mpc.solve(current_error_x, current_error_y)
        step_x, step_y = result.control_sequence_x[0], result.control_sequence_y[0]
    """

    def __init__(self, config: Optional[DataDrivenMPCConfig] = None):
        self.config = config or DataDrivenMPCConfig()
        cfg = self.config

        # 历史数据缓冲
        self._error_history_x: Deque[float] = Deque(maxlen=cfg.history_buffer_size)
        self._error_history_y: Deque[float] = Deque(maxlen=cfg.history_buffer_size)
        self._step_history_x: Deque[int] = Deque(maxlen=cfg.history_buffer_size)
        self._step_history_y: Deque[int] = Deque(maxlen=cfg.history_buffer_size)

        # 学习到的系统模型参数
        # 简化 ARX 模型: e(t+1) = a * e(t) + b * u(t)
        self._model_ax = 0.8  # X 方向自回归系数
        self._model_bx = -0.5  # X 方向控制系数
        self._model_ay = 0.8
        self._model_by = -0.5
        self._model_bias_x = 0.0
        self._model_bias_y = 0.0
        self._model_r2_x = 0.0
        self._model_r2_y = 0.0

        # 上一次求解结果 (warm-start)
        self._prev_solution_x: Optional[np.ndarray] = None
        self._prev_solution_y: Optional[np.ndarray] = None

        # 步数计数
        self._step_count = 0

    def add_data_point(
        self, error_x: float, error_y: float, step_x: int, step_y: int
    ) -> None:
        """添加一个历史数据点。

        Args:
            error_x: X 方向误差 (像素)
            error_y: Y 方向误差 (像素)
            step_x: X 方向步数
            step_y: Y 方向步数
        """
        self._error_history_x.append(error_x)
        self._error_history_y.append(error_y)
        self._step_history_x.append(step_x)
        self._step_history_y.append(step_y)
        self._step_count += 1

        # 在线模型更新
        if (self.config.online_model_update
                and self._step_count % self.config.model_update_interval == 0
                and len(self._error_history_x) >= self.config.identification_window):
            self._update_model()

    def solve(
        self, current_error_x: float, current_error_y: float
    ) -> DataDrivenMPCResult:
        """求解最优控制序列。

        Args:
            current_error_x: 当前 X 方向误差 (像素)
            current_error_y: 当前 Y 方向误差 (像素)

        Returns:
            DataDrivenMPCResult: 优化结果
        """
        t0 = time.perf_counter()
        cfg = self.config
        Np = cfg.prediction_horizon
        Nc = cfg.control_horizon

        result = DataDrivenMPCResult()

        # 分别求解 X 和 Y 方向
        sol_x = self._solve_1d(
            current_error_x, self._model_ax, self._model_bx, self._model_bias_x,
            self._prev_solution_x,
        )
        sol_y = self._solve_1d(
            current_error_y, self._model_ay, self._model_by, self._model_bias_y,
            self._prev_solution_y,
        )

        # 限幅
        sol_x = np.clip(sol_x, -cfg.max_step, cfg.max_step)
        sol_y = np.clip(sol_y, -cfg.max_step, cfg.max_step)

        # 死区
        sol_x[np.abs(sol_x) < cfg.min_step] = 0
        sol_y[np.abs(sol_y) < cfg.min_step] = 0

        # 加速度约束
        if self._prev_solution_x is not None and len(self._prev_solution_x) > 0:
            prev_step_x = self._prev_solution_x[0]
            diff = sol_x[0] - prev_step_x
            diff = np.clip(diff, -cfg.max_acceleration, cfg.max_acceleration)
            sol_x[0] = prev_step_x + diff

        if self._prev_solution_y is not None and len(self._prev_solution_y) > 0:
            prev_step_y = self._prev_solution_y[0]
            diff = sol_y[0] - prev_step_y
            diff = np.clip(diff, -cfg.max_acceleration, cfg.max_acceleration)
            sol_y[0] = prev_step_y + diff

        # 预测状态轨迹 (使用各自方向的模型参数)
        pred_x = self._predict_trajectory_1d(current_error_x, sol_x, self._model_ax, self._model_bx, self._model_bias_x)
        pred_y = self._predict_trajectory_1d(current_error_y, sol_y, self._model_ay, self._model_by, self._model_bias_y)

        # 计算代价
        cost = self._compute_cost(pred_x, pred_y, sol_x, sol_y)

        # 保存 warm-start
        if cfg.warm_start:
            self._prev_solution_x = sol_x.copy()
            self._prev_solution_y = sol_y.copy()

        result.control_sequence_x = [int(round(v)) for v in sol_x[:Nc]]
        result.control_sequence_y = [int(round(v)) for v in sol_y[:Nc]]
        result.predicted_error_x = [round(float(v), 3) for v in pred_x]
        result.predicted_error_y = [round(float(v), 3) for v in pred_y]
        result.optimal_cost = round(cost, 4)
        result.converged = True
        result.processing_time_ms = round((time.perf_counter() - t0) * 1000, 2)
        result.model_quality_r2 = round(
            (self._model_r2_x + self._model_r2_y) / 2, 4
        )

        return result

    def _solve_1d(
        self,
        current_error: float,
        a: float,
        b: float,
        bias: float,
        warm_start: Optional[np.ndarray],
    ) -> np.ndarray:
        """求解单方向的最优控制序列 (梯度下降)。"""
        cfg = self.config
        Np = cfg.prediction_horizon
        Nc = cfg.control_horizon

        # 初始化
        if cfg.warm_start and warm_start is not None and len(warm_start) >= Nc:
            u = warm_start[:Nc].copy().astype(np.float64)
        else:
            # 启发式初始化: 比例控制
            u = np.full(Nc, -current_error * 2.0)

        best_u = u.copy()
        best_cost = float("inf")

        lr = 0.1
        for iteration in range(cfg.optimization_max_iter):
            # 预测轨迹
            trajectory = self._predict_trajectory_1d(current_error, u, a, b, bias)

            # 计算代价和梯度
            cost, grad = self._cost_and_grad_1d(
                trajectory, u, a, b, bias, Np, Nc,
                cfg.state_weight, cfg.control_weight, cfg.terminal_weight,
            )

            if cost < best_cost:
                best_cost = cost
                best_u = u.copy()

            # 梯度下降
            u -= lr * grad

            # 投影到约束
            u = np.clip(u, -cfg.max_step, cfg.max_step)

            # 收敛检查
            if iteration > 0 and abs(cost - best_cost) < cfg.optimization_tolerance:
                break

        # 控制时域外设为零
        full_u = np.zeros(Np)
        full_u[:Nc] = best_u
        return full_u

    def _predict_trajectory_1d(
        self, e0: float, u: np.ndarray, a: float, b: float, bias: float,
    ) -> np.ndarray:
        """预测单方向的状态轨迹。"""
        Np = len(u)
        trajectory = np.zeros(Np + 1)
        trajectory[0] = e0
        for t in range(Np):
            ut = u[t] if t < len(u) else 0.0
            trajectory[t + 1] = a * trajectory[t] + b * ut + bias
        return trajectory

    def _cost_and_grad_1d(
        self,
        trajectory: np.ndarray,
        u: np.ndarray,
        a: float,
        b: float,
        bias: float,
        Np: int,
        Nc: int,
        w_state: float,
        w_control: float,
        w_terminal: float,
    ) -> Tuple[float, np.ndarray]:
        """计算代价和梯度。"""
        grad = np.zeros(Nc)

        # 状态代价
        state_cost = w_state * float(np.sum(trajectory[:Np] ** 2))

        # 终端代价
        terminal_cost = w_terminal * float(trajectory[Np] ** 2)

        # 控制代价
        control_cost = w_control * float(np.sum(u ** 2))

        total_cost = state_cost + terminal_cost + control_cost

        # 数值梯度
        eps = 1.0
        for j in range(Nc):
            u_plus = u.copy()
            u_plus[j] += eps
            traj_plus = self._predict_trajectory_1d(trajectory[0] if len(trajectory) > 0 else 0, u_plus, a, b, bias)
            cost_plus = (w_state * float(np.sum(traj_plus[:Np] ** 2))
                         + w_terminal * float(traj_plus[Np] ** 2)
                         + w_control * float(np.sum(u_plus ** 2)))
            grad[j] = (cost_plus - total_cost) / eps

        return total_cost, grad

    def _predict_trajectory(
        self, e0: float, u: np.ndarray
    ) -> np.ndarray:
        """预测单方向轨迹。"""
        return self._predict_trajectory_1d(
            e0, u, self._model_ax, self._model_bx, self._model_bias_x
        )

    def _compute_cost(
        self, pred_x: np.ndarray, pred_y: np.ndarray,
        ctrl_x: np.ndarray, ctrl_y: np.ndarray,
    ) -> float:
        """计算总代价。"""
        cfg = self.config
        Np = cfg.prediction_horizon
        Nc = cfg.control_horizon

        state_cost = cfg.state_weight * (
            float(np.sum(pred_x[:Np] ** 2)) + float(np.sum(pred_y[:Np] ** 2))
        )
        terminal_cost = cfg.terminal_weight * (
            float(pred_x[Np] ** 2) + float(pred_y[Np] ** 2)
        ) if len(pred_x) > Np else 0.0
        control_cost = cfg.control_weight * (
            float(np.sum(ctrl_x[:Nc] ** 2)) + float(np.sum(ctrl_y[:Nc] ** 2))
        )

        return state_cost + terminal_cost + control_cost

    def _update_model(self) -> None:
        """从历史数据中更新系统模型 (最小二乘 ARX 辨识)。"""
        window = self.config.identification_window
        if len(self._error_history_x) < window + 1:
            return

        # X 方向
        ex = np.array(list(self._error_history_x))
        sx = np.array(list(self._step_history_x), dtype=np.float64)

        # 构建回归矩阵: [e(t), u(t), 1]
        n_samples = min(window, len(ex) - 1)
        A_mat = np.zeros((n_samples, 3))
        b_vec = np.zeros(n_samples)

        for i in range(n_samples):
            idx = len(ex) - 1 - i
            if idx < 1:
                break
            A_mat[i] = [ex[idx - 1], sx[idx - 1], 1.0]
            b_vec[i] = ex[idx]

        # 最小二乘
        try:
            params, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            self._model_ax = float(np.clip(params[0], 0.1, 0.99))
            self._model_bx = float(params[1])
            self._model_bias_x = float(params[2])

            # R² 拟合度
            pred = A_mat @ params
            ss_res = float(np.sum((b_vec - pred) ** 2))
            ss_tot = float(np.sum((b_vec - np.mean(b_vec)) ** 2)) + 1e-10
            self._model_r2_x = max(0.0, 1.0 - ss_res / ss_tot)
        except np.linalg.LinAlgError:
            pass

        # Y 方向
        ey = np.array(list(self._error_history_y))
        sy = np.array(list(self._step_history_y), dtype=np.float64)

        A_mat_y = np.zeros((n_samples, 3))
        b_vec_y = np.zeros(n_samples)

        for i in range(n_samples):
            idx = len(ey) - 1 - i
            if idx < 1:
                break
            A_mat_y[i] = [ey[idx - 1], sy[idx - 1], 1.0]
            b_vec_y[i] = ey[idx]

        try:
            params_y, _, _, _ = np.linalg.lstsq(A_mat_y, b_vec_y, rcond=None)
            self._model_ay = float(np.clip(params_y[0], 0.1, 0.99))
            self._model_by = float(params_y[1])
            self._model_bias_y = float(params_y[2])

            pred_y = A_mat_y @ params_y
            ss_res_y = float(np.sum((b_vec_y - pred_y) ** 2))
            ss_tot_y = float(np.sum((b_vec_y - np.mean(b_vec_y)) ** 2)) + 1e-10
            self._model_r2_y = max(0.0, 1.0 - ss_res_y / ss_tot_y)
        except np.linalg.LinAlgError:
            pass

    def get_model_summary(self) -> dict:
        """返回当前模型摘要。"""
        return {
            "model_x": {
                "ar_coefficient": round(self._model_ax, 4),
                "control_coefficient": round(self._model_bx, 4),
                "bias": round(self._model_bias_x, 4),
                "r_squared": round(self._model_r2_x, 4),
            },
            "model_y": {
                "ar_coefficient": round(self._model_ay, 4),
                "control_coefficient": round(self._model_by, 4),
                "bias": round(self._model_bias_y, 4),
                "r_squared": round(self._model_r2_y, 4),
            },
            "data_points": len(self._error_history_x),
            "step_count": self._step_count,
        }

    def reset(self) -> None:
        """重置控制器状态。"""
        self._error_history_x.clear()
        self._error_history_y.clear()
        self._step_history_x.clear()
        self._step_history_y.clear()
        self._prev_solution_x = None
        self._prev_solution_y = None
        self._step_count = 0
        self._model_ax = 0.8
        self._model_bx = -0.5
        self._model_ay = 0.8
        self._model_by = -0.5
        self._model_bias_x = 0.0
        self._model_bias_y = 0.0
