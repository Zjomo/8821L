"""
子空间系统辨识器 (SubspaceSystemIdentifier)

基于 ML for AO (https://github.com/AleksandarHaber/Machine-Learning-and-System-Identification-for-Adaptive-Optics)
的子空间辨识方法，为 SpotZoom 提供数据驱动的光学系统动力学建模能力。

灵感来源:
- Haber ML for AO: 子空间辨识用于变形镜系统辨识
  (https://github.com/AleksandarHaber/Machine-Learning-and-System-Identification-for-Adaptive-Optics)
- PINA: 方程学习 (https://github.com/mathLab/PINA)
- OOPAO: 系统辨识 (https://github.com/cheritier/OOPAO)

算法原理:
  1. 子空间辨识 (N4SID): 从输入-输出数据中辨识状态空间模型
     A*x(k+1) + B*u(k) = x(k+1), C*x(k) = y(k)
  2. 模型阶次选择: 使用奇异值分解确定最优模型阶次
  3. 模型验证: 使用验证数据评估模型精度
  4. 模型简化: 必要时降低模型阶次

与现有模块的关系:
  - 增强 v1/optical_system_identifier.py 的系统辨识能力
  - 增强 v3/system_identifier.py 的辨识精度
  - 与 v1/pinn_beam_solver.py 的物理模型协同

外部依赖: numpy, scipy
"""

import numpy as np
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class SubspaceConfig:
    """子空间辨识配置。"""
    # 模型参数
    max_order: int = 10              # 最大模型阶次
    min_order: int = 2               # 最小模型阶次
    auto_select_order: bool = True   # 自动选择阶次
    order_selection_threshold: float = 0.99  # 奇异值累积能量阈值

    # 数据参数
    min_data_points: int = 200       # 最小数据点数
    train_ratio: float = 0.8         # 训练集比例

    # 正则化
    regularization: float = 1e-6     # Tikhonov 正则化

    # 输出参数
    compute_frequency_response: bool = True  # 计算频率响应
    compute_step_response: bool = True       # 计算阶跃响应
    step_response_duration: float = 5.0      # 阶跃响应持续时间


@dataclass
class SubspaceReport:
    """子空间辨识报告。"""
    # 模型信息
    model_order: int = 0
    num_inputs: int = 0
    num_outputs: int = 0
    num_data_points: int = 0

    # 模型质量
    train_fit: float = 0.0           # 训练集拟合度 (R²)
    validation_fit: float = 0.0      # 验证集拟合度
    singular_values: List[float] = field(default_factory=list)
    order_selection_energy: float = 0.0

    # 频域特性
    bandwidth_hz: float = 0.0        # 带宽
    phase_margin_deg: float = 0.0    # 相位裕度
    gain_margin_db: float = 0.0      # 增益裕度

    # 性能信息
    processing_time_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)


class SubspaceSystemIdentifier:
    """子空间系统辨识器。

    从输入-输出数据中辨识光学系统的状态空间模型，
    用于控制器设计和系统性能预测。

    Parameters
    ----------
    config : SubspaceConfig
        辨识器配置。
    """

    def __init__(self, config: Optional[SubspaceConfig] = None):
        self.config = config or SubspaceConfig()
        self._A: Optional[np.ndarray] = None  # 状态转移矩阵
        self._B: Optional[np.ndarray] = None  # 输入矩阵
        self._C: Optional[np.ndarray] = None  # 输出矩阵
        self._D: Optional[np.ndarray] = None  # 直馈矩阵
        self._dt: float = 1.0  # 采样时间

    @property
    def is_identified(self) -> bool:
        return self._A is not None

    def identify(
        self,
        inputs: np.ndarray,
        outputs: np.ndarray,
        dt: float = 1.0,
    ) -> SubspaceReport:
        """执行系统辨识。

        Parameters
        ----------
        inputs : np.ndarray
            输入数据，形状 (N, n_inputs)。
        outputs : np.ndarray
            输出数据，形状 (N, n_outputs)。
        dt : float
            采样时间间隔。

        Returns
        -------
        SubspaceReport
            辨识报告。
        """
        t0 = time.perf_counter()
        report = SubspaceReport()
        self._dt = dt

        N, n_u = inputs.shape
        n_y = outputs.shape[1]
        report.num_inputs = n_u
        report.num_outputs = n_y
        report.num_data_points = N

        if N < self.config.min_data_points:
            report.warnings.append(
                f"数据点不足 ({N} < {self.config.min_data_points})"
            )
            report.processing_time_ms = (time.perf_counter() - t0) * 1000
            return report

        # 划分训练/验证集
        n_train = int(N * self.config.train_ratio)
        u_train = inputs[:n_train]
        y_train = outputs[:n_train]
        u_val = inputs[n_train:]
        y_val = outputs[n_train:]

        # 子空间辨识
        try:
            A, B, C, D, singular_values = self._n4sid(u_train, y_train)
            self._A, self._B, self._C, self._D = A, B, C, D
            report.singular_values = [float(s) for s in singular_values]

            # 自动选择阶次
            if self.config.auto_select_order:
                order = self._select_order(singular_values)
            else:
                order = self.config.max_order

            # 截断到选定阶次
            self._A = A[:order, :order]
            self._B = B[:order, :]
            self._C = C[:, :order]
            report.model_order = order

            # 计算拟合度
            report.train_fit = self._compute_fit(u_train, y_train)
            report.validation_fit = self._compute_fit(u_val, y_val)

            # 计算频率响应
            if self.config.compute_frequency_response:
                self._compute_frequency_response(report, dt)

        except Exception as e:
            logger.error(f"系统辨识失败: {e}")
            report.warnings.append(f"辨识异常: {str(e)}")

        report.processing_time_ms = (time.perf_counter() - t0) * 1000
        return report

    def _n4sid(
        self, u: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """N4SID 子空间辨识算法。"""
        N = len(u)
        n_u = u.shape[1]
        n_y = y.shape[1]

        # 构建 Hankel 矩阵
        # 简化实现：使用 past 输入/输出预测未来输出
        L = min(20, N // 3)  # 窗口长度

        # Past 和 Future 数据
        Y_past = np.zeros((L * n_y, N - 2 * L + 1))
        U_past = np.zeros((L * n_u, N - 2 * L + 1))
        Y_future = np.zeros((L * n_y, N - 2 * L + 1))

        for i in range(N - 2 * L + 1):
            for j in range(L):
                Y_past[j*n_y:(j+1)*n_y, i] = y[i+j]
                U_past[j*n_u:(j+1)*n_u, i] = u[i+j]
                Y_future[j*n_y:(j+1)*n_y, i] = y[i+L+j]

        # 构建观测矩阵
        W = np.vstack([Y_past, U_past])

        # SVD
        try:
            U_w, S, Vt = np.linalg.svd(W @ Y_future.T, full_matrices=False)
        except np.linalg.LinAlgError:
            # 回退：使用小规模 SVD
            U_w, S, Vt = np.linalg.svd(
                W @ Y_future.T + self.config.regularization * np.eye(W.shape[0]),
                full_matrices=False
            )

        # 状态估计
        n_states = min(self.config.max_order, len(S))
        S = S[:n_states]

        # 从 SVD 结果提取状态空间模型
        # 简化：使用最小二乘拟合
        n = n_states
        X = (U_w[:, :n] * np.sqrt(S[:n])).T  # 状态序列 (n, N-2L+1)

        # 最小二乘拟合 A, B, C, D
        # x(k+1) = A*x(k) + B*u(k)
        # y(k) = C*x(k) + D*u(k)
        n_cols = X.shape[1] - 1
        if n_cols < n + n_u:
            n = max(2, n_cols // 2)
            X = X[:n, :]
            n_states = n

        X1 = X[:, :-1]  # x(k)
        X2 = X[:, 1:]   # x(k+1)

        # 回归矩阵 [X1; U]
        U_reg = u[L:L+n_cols].T
        reg = np.vstack([X1, U_reg])

        # A, B
        try:
            params_x = np.linalg.lstsq(reg.T, X2.T, rcond=None)[0]
            A = params_x[:, :n].T
            B = params_x[:, n:].T
        except np.linalg.LinAlgError:
            A = np.eye(n) * 0.9
            B = np.zeros((n, n_u))

        # C, D
        Y_reg = y[L:L+n_cols].T
        reg_y = np.vstack([X1, U_reg])
        try:
            params_y = np.linalg.lstsq(reg_y.T, Y_reg.T, rcond=None)[0]
            C = params_y[:, :n].T
            D = params_y[:, n:].T
        except np.linalg.LinAlgError:
            C = np.zeros((n_y, n))
            D = np.zeros((n_y, n_u))

        return A, B, C, D, S

    def _select_order(self, singular_values: np.ndarray) -> int:
        """基于奇异值能量选择模型阶次。"""
        total_energy = np.sum(singular_values ** 2)
        cumulative = np.cumsum(singular_values ** 2) / total_energy

        for i, e in enumerate(cumulative):
            if e >= self.config.order_selection_threshold:
                return max(self.config.min_order, i + 1)

        return self.config.min_order

    def _compute_fit(self, u: np.ndarray, y: np.ndarray) -> float:
        """计算模型拟合度 (R²)。"""
        if self._A is None:
            return 0.0

        y_pred = self._simulate(u)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)

        if ss_tot < 1e-10:
            return 1.0
        return float(1.0 - ss_res / ss_tot)

    def _simulate(self, u: np.ndarray) -> np.ndarray:
        """使用辨识模型仿真输出。"""
        n = self._A.shape[0]
        x = np.zeros(n)
        y_sim = np.zeros((len(u), self._C.shape[0]))

        for k in range(len(u)):
            y_sim[k] = self._C @ x + self._D @ u[k]
            x = self._A @ x + self._B @ u[k]

        return y_sim

    def _compute_frequency_response(self, report: SubspaceReport, dt: float):
        """计算频率响应特性。"""
        if self._A is None:
            return

        try:
            # 离散时间频率响应
            freqs = np.logspace(-2, np.log10(0.5 / dt), 100)
            gains = []
            phases = []

            for f in freqs:
                z = np.exp(1j * 2 * np.pi * f * dt)
                # H(z) = C(zI - A)^(-1)B + D
                try:
                    H = self._C @ np.linalg.solve(z * np.eye(self._A.shape[0]) - self._A, self._B) + self._D
                    gains.append(20 * np.log10(np.max(np.abs(H)) + 1e-10))
                    phases.append(np.angle(H[0, 0], deg=True) if H.size > 0 else 0)
                except np.linalg.LinAlgError:
                    gains.append(-100)
                    phases.append(0)

            # 带宽估计（-3dB）
            gains = np.array(gains)
            max_gain = np.max(gains)
            threshold = max_gain - 3
            above = gains > threshold
            if np.any(above):
                report.bandwidth_hz = float(freqs[np.where(above)[0][-1]])

            # 简化的裕度估计
            report.gain_margin_db = float(max_gain - np.min(gains))
            report.phase_margin_deg = float(np.min(phases) + 180 if len(phases) > 0 else 0)

        except Exception as e:
            report.warnings.append(f"频率响应计算失败: {e}")

    def predict(self, u: np.ndarray, x0: Optional[np.ndarray] = None) -> np.ndarray:
        """使用辨识模型预测输出。"""
        if not self.is_identified:
            raise RuntimeError("模型未辨识")

        return self._simulate(u)

    def get_state_space(self) -> Optional[Dict[str, np.ndarray]]:
        """获取状态空间模型矩阵。"""
        if not self.is_identified:
            return None
        return {
            'A': self._A.copy(),
            'B': self._B.copy(),
            'C': self._C.copy(),
            'D': self._D.copy(),
            'dt': self._dt,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = SubspaceConfig(max_order=6)
    identifier = SubspaceSystemIdentifier(config)

    # 生成模拟数据（二阶系统）
    np.random.seed(42)
    N = 500
    dt = 0.01

    # 真实系统: x(k+1) = A*x(k) + B*u(k)
    A_true = np.array([[0.95, 0.1], [-0.1, 0.9]])
    B_true = np.array([[0.5], [0.1]])
    C_true = np.array([[1.0, 0.0]])
    D_true = np.array([[0.0]])

    u = np.random.randn(N, 1) * 0.5
    x = np.zeros(2)
    y = np.zeros((N, 1))

    for k in range(N):
        y[k] = C_true @ x + D_true @ u[k] + np.random.randn() * 0.05
        x = A_true @ x + B_true @ u[k]

    # 辨识
    report = identifier.identify(u, y, dt)
    print(f"模型阶次: {report.model_order}")
    print(f"训练拟合度: {report.train_fit:.4f}")
    print(f"验证拟合度: {report.validation_fit:.4f}")
    print(f"带宽: {report.bandwidth_hz:.1f} Hz")
    print(f"奇异值: {[f'{s:.3f}' for s in report.singular_values[:6]]}")
    print(f"处理时间: {report.processing_time_ms:.1f}ms")
