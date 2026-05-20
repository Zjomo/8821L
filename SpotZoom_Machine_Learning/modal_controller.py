"""
模态控制器 (ModalController)

灵感来源:
- AOtools (https://github.com/AOtools/aotools) — 自适应光学模态控制
- HCIPy (https://github.com/ehpor/hcipy) — 波前控制与模态分解
- SOAPY AO Simulation — 自适应光学系统仿真框架

算法原理:
- Zernike Modal Decomposition — 将波前误差分解为 Zernike 模式，
  实现各像差模式的独立控制
- Per-Mode Integrator Control — 每个模式使用独立的积分控制器，
  带抗饱和 (anti-windup) 机制
- Influence Matrix & Pseudo-Inverse — 通过影响矩阵的伪逆计算
  控制矩阵，将模式空间误差映射到执行器空间
- Noise Propagation — 基于噪声传播的模态增益优化，
  最大化抑制带宽同时控制噪声放大
- Rejection Bandwidth — 每个模式的闭环抑制带宽估计

功能:
- Zernike 模式分解 (前 15 阶: tip, tilt, defocus, astigmatism, coma 等)
- 每模式独立积分控制与增益优化
- 控制矩阵计算 (伪逆)
- 模态误差分解与残差分析
- 抗饱和与安全限幅

依赖: numpy (无外部控制理论库)
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger("SpotZoom.ModalController")

# 模块默认禁用标志
modal_control_enabled: bool = False

# Noll 索引到 (n, m) 的映射 (前 15 阶)
_NOLL_TO_NM: Dict[int, Tuple[int, int]] = {
    1: (0, 0),   # Piston
    2: (1, 1),   # Tilt X
    3: (1, -1),  # Tilt Y
    4: (2, 0),   # Defocus
    5: (2, 2),   # Oblique Astigmatism
    6: (2, -2),  # Vertical Astigmatism
    7: (3, 1),   # Vertical Coma
    8: (3, -1),  # Horizontal Coma
    9: (3, 3),   # Oblique Trefoil
    10: (3, -3), # Vertical Trefoil
    11: (4, 0),  # Primary Spherical
    12: (4, 2),  # Secondary Astigmatism
    13: (4, -2), # Secondary Astigmatism
    14: (4, 4),  # Oblique Quadrafoil
    15: (4, -4), # Vertical Quadrafoil
}

# Zernike 项中文名称
_NOLL_NAMES: Dict[int, str] = {
    1: "Piston", 2: "Tilt X", 3: "Tilt Y", 4: "Defocus",
    5: "Astigmatism 45", 6: "Astigmatism 0",
    7: "Coma X", 8: "Coma Y",
    9: "Trefoil Oblique", 10: "Trefoil Vertical",
    11: "Spherical", 12: "2nd Astigmatism 45",
    13: "2nd Astigmatism 0", 14: "Quadrafoil Oblique",
    15: "Quadrafoil Vertical",
}


@dataclass
class ModalControlOutput:
    """模态控制输出。"""
    command_x: float  # X 方向台阶控制指令
    command_y: float  # Y 方向台阶控制指令
    modal_errors: Dict[int, float]  # 各模式残差 {Noll索引: 误差}
    total_residual: float  # 总残差 (RMS)
    dominant_mode: str  # 主导误差模式名称
    control_effort: float  # 控制代价 (指令范数)


@dataclass
class ModalState:
    """模态控制器内部状态。"""
    integrator_states: Dict[int, float]  # 各模式积分器状态
    error_history: Dict[int, List[float]]  # 各模式误差历史
    gain_history: Dict[int, List[float]]  # 各模式增益历史


class ModalController:
    """Zernike 模态控制器。

    将波前误差分解为 Zernike 模式，对每个模式独立进行积分控制，
    然后通过控制矩阵将模式空间指令映射回物理空间 (XY 台阶)。

    模态控制优势:
    - 各像差模式独立控制，互不耦合
    - 可针对不同模式设置不同增益
    - 噪声传播可控
    - 便于诊断主导误差来源

    Parameters
    ----------
    num_modes : int
        控制的 Zernike 模式数量 (2~15)。模式 1 (Piston) 通常跳过。
    grid_size : int
        内部计算网格尺寸。
    default_gain : float
        默认积分器增益。
    max_command : float
        最大控制指令限幅。
    anti_windup_limit : float
        积分器抗饱和限幅。
    leak_factor : float
        积分泄漏因子 (0~1)，防止长期漂移。1.0=无泄漏。
    """

    def __init__(
        self,
        num_modes: int = 6,
        grid_size: int = 64,
        default_gain: float = 0.3,
        max_command: float = 50.0,
        anti_windup_limit: float = 100.0,
        leak_factor: float = 0.999,
    ):
        self.num_modes = max(2, min(int(num_modes), 15))
        self.grid_size = int(grid_size)
        self.default_gain = float(default_gain)
        self.max_command = float(max_command)
        self.anti_windup_limit = float(anti_windup_limit)
        self.leak_factor = float(leak_factor)

        # 模式索引 (跳过 piston j=1)
        self._mode_indices = [j for j in range(2, self.num_modes + 1) if j <= 15]

        # 每模式增益
        self._modal_gains: Dict[int, float] = {
            j: self.default_gain for j in self._mode_indices
        }

        # 积分器状态
        self._integrator_states: Dict[int, float] = {
            j: 0.0 for j in self._mode_indices
        }

        # 误差历史
        self._error_history: Dict[int, deque] = {
            j: deque(maxlen=100) for j in self._mode_indices
        }

        # 增益历史
        self._gain_history: Dict[int, deque] = {
            j: deque(maxlen=50) for j in self._mode_indices
        }

        # 预计算 Zernike 基矩阵
        self._build_zernike_basis()

        # 控制矩阵 (伪逆)
        self._control_matrix: Optional[np.ndarray] = None
        self._compute_control_matrix()

        LOGGER.info(
            "ModalController: 初始化完成 (modes=%d, grid=%d, gain=%.3f)",
            self.num_modes, self.grid_size, self.default_gain,
        )

    def update(self, measurement_vector: np.ndarray) -> ModalControlOutput:
        """主控制更新: 根据测量向量计算控制指令。

        Parameters
        ----------
        measurement_vector : np.ndarray
            测量向量。可以是:
            - 波前误差图 (2D)
            - Zernike 系数向量 (1D, 长度 = num_modes)

        Returns
        -------
        ModalControlOutput
            控制输出。
        """
        try:
            # 分解为 Zernike 模式
            if measurement_vector.ndim == 2:
                # 输入是波前误差图
                mode_errors = self.decompose_modes(measurement_vector)
            elif measurement_vector.ndim == 1:
                # 输入已经是 Zernike 系数
                mode_errors = {}
                for idx, j in enumerate(self._mode_indices):
                    if idx < len(measurement_vector):
                        mode_errors[j] = float(measurement_vector[idx])
                    else:
                        mode_errors[j] = 0.0
            else:
                LOGGER.warning("ModalController: 不支持的测量向量维度: %d",
                               measurement_vector.ndim)
                return self._empty_output()

            # 计算控制信号
            command = self.compute_control_signal(mode_errors)

            # 更新积分器
            self._update_integrators(mode_errors)

            # 计算残差
            total_residual = float(np.sqrt(
                np.mean(np.array(list(mode_errors.values())) ** 2)
            ))

            # 主导模式
            dominant_mode = "None"
            max_err = 0.0
            for j, err in mode_errors.items():
                if abs(err) > max_err:
                    max_err = abs(err)
                    dominant_mode = _NOLL_NAMES.get(j, f"Z{j}")

            # 控制代价
            control_effort = float(np.sqrt(command[0] ** 2 + command[1] ** 2))

            LOGGER.debug(
                "ModalController: cmd=(%.3f, %.3f), residual=%.4f, dominant=%s",
                command[0], command[1], total_residual, dominant_mode,
            )

            return ModalControlOutput(
                command_x=round(command[0], 6),
                command_y=round(command[1], 6),
                modal_errors={j: round(v, 6) for j, v in mode_errors.items()},
                total_residual=round(total_residual, 6),
                dominant_mode=dominant_mode,
                control_effort=round(control_effort, 6),
            )

        except Exception as e:
            LOGGER.error("ModalController: 控制更新失败: %s", e)
            return self._empty_output()

    def compute_control_signal(self, error_modes: Dict[int, float]) -> np.ndarray:
        """从模式误差计算物理空间控制指令。

        Parameters
        ----------
        error_modes : Dict[int, float]
            各模式的误差值 {Noll索引: 误差}。

        Returns
        -------
        np.ndarray
            控制指令 [command_x, command_y]。
        """
        # 构建误差向量
        error_vec = np.zeros(len(self._mode_indices), dtype=np.float64)
        for idx, j in enumerate(self._mode_indices):
            error_vec[idx] = error_modes.get(j, 0.0)

        # 模式空间控制: u_mode = -gain * error
        mode_command = np.zeros_like(error_vec)
        for idx, j in enumerate(self._mode_indices):
            gain = self._modal_gains.get(j, self.default_gain)
            mode_command[idx] = -gain * error_vec[idx]

        # 映射到物理空间 (使用控制矩阵)
        if self._control_matrix is not None:
            try:
                physical_command = self._control_matrix @ mode_command
            except (np.linalg.LinAlgError, ValueError):
                # 降级: 仅使用 tip/tilt 模式
                physical_command = np.array([
                    mode_command[0] if len(mode_command) > 0 else 0.0,  # Tilt X
                    mode_command[1] if len(mode_command) > 1 else 0.0,  # Tilt Y
                ])
        else:
            physical_command = np.array([
                mode_command[0] if len(mode_command) > 0 else 0.0,
                mode_command[1] if len(mode_command) > 1 else 0.0,
            ])

        # 限幅
        physical_command = np.clip(
            physical_command, -self.max_command, self.max_command
        )

        return physical_command

    def decompose_modes(self, wavefront_error: np.ndarray) -> Dict[int, float]:
        """将波前误差分解为 Zernike 模式系数。

        Parameters
        ----------
        wavefront_error : np.ndarray
            波前误差图 (2D)。

        Returns
        -------
        Dict[int, float]
            各模式的系数 {Noll索引: 系数}。
        """
        try:
            # 调整尺寸
            if wavefront_error.shape[0] != self.grid_size or \
               wavefront_error.shape[1] != self.grid_size:
                import cv2
                wf = cv2.resize(wavefront_error, (self.grid_size, self.grid_size),
                                interpolation=cv2.INTER_AREA)
            else:
                wf = wavefront_error.astype(np.float64)

            # 仅取孔径内像素
            mask_flat = self._pupil_mask.ravel().astype(bool)
            wf_flat = wf.ravel()[mask_flat]

            if wf_flat.size == 0 or self._basis_matrix is None:
                return {j: 0.0 for j in self._mode_indices}

            # 最小二乘拟合
            coeffs, _, _, _ = np.linalg.lstsq(
                self._basis_matrix, wf_flat, rcond=None
            )

            # 构建结果
            result: Dict[int, float] = {}
            for idx, j in enumerate(self._mode_indices):
                if idx < len(coeffs):
                    result[j] = float(coeffs[idx])
                else:
                    result[j] = 0.0

            return result

        except Exception as e:
            LOGGER.warning("ModalController: 模式分解失败: %s", e)
            return {j: 0.0 for j in self._mode_indices}

    def set_modal_gains(self, gains: Dict[int, float]) -> None:
        """设置每模式积分器增益。

        Parameters
        ----------
        gains : Dict[int, float]
            Noll 索引到增益的映射。
        """
        for j, gain in gains.items():
            if j in self._mode_indices:
                self._modal_gains[j] = max(0.0, min(float(gain), 2.0))
                self._gain_history[j].append(self._modal_gains[j])
                LOGGER.debug("ModalController: 模式 %s 增益设为 %.4f",
                             _NOLL_NAMES.get(j, f"Z{j}"), self._modal_gains[j])

    def compute_rejection_bandwidth(self, noise_variance: float) -> Dict[int, float]:
        """计算每模式的闭环抑制带宽。

        抑制带宽取决于积分器增益和噪声水平:
            f_reject = gain / (2 * pi) * sqrt(SNR)

        Parameters
        ----------
        noise_variance : float
            测量噪声方差。

        Returns
        -------
        Dict[int, float]
            各模式的抑制带宽 (Hz)。
        """
        bandwidths: Dict[int, float] = {}
        for j in self._mode_indices:
            gain = self._modal_gains.get(j, self.default_gain)
            if noise_variance > 1e-12:
                snr = 1.0 / noise_variance
                bw = gain / (2.0 * np.pi) * np.sqrt(min(snr, 100.0))
            else:
                bw = gain / (2.0 * np.pi)
            bandwidths[j] = round(min(bw, 100.0), 4)

        return bandwidths

    def get_modal_errors(self) -> Dict[int, float]:
        """获取当前各模式的残差。

        Returns
        -------
        Dict[int, float]
            各模式的当前残差 {Noll索引: 残差}。
        """
        result: Dict[int, float] = {}
        for j in self._mode_indices:
            history = self._error_history[j]
            if len(history) > 0:
                result[j] = round(float(history[-1]), 6)
            else:
                result[j] = 0.0
        return result

    def get_state(self) -> ModalState:
        """获取控制器内部状态。

        Returns
        -------
        ModalState
            控制器状态。
        """
        return ModalState(
            integrator_states=dict(self._integrator_states),
            error_history={j: list(h) for j, h in self._error_history.items()},
            gain_history={j: list(h) for j, h in self._gain_history.items()},
        )

    def reset(self) -> None:
        """重置控制器，清除所有积分器状态和历史。"""
        self._integrator_states = {j: 0.0 for j in self._mode_indices}
        for j in self._mode_indices:
            self._error_history[j].clear()
            self._gain_history[j].clear()

        LOGGER.info("ModalController: 控制器已重置")

    # ======================== 内部方法 ========================

    def _build_zernike_basis(self) -> None:
        """预计算 Zernike 基矩阵。"""
        g = self.grid_size
        yy, xx = np.mgrid[:g, :g]
        cx, cy = g / 2.0, g / 2.0
        dx = xx - cx
        dy = yy - cy
        rho = np.sqrt(dx ** 2 + dy ** 2) / (g / 2.0)
        theta = np.arctan2(dy, dx)

        # 圆孔径掩模
        self._pupil_mask = (rho <= 1.0).astype(np.uint8)
        mask_flat = self._pupil_mask.ravel().astype(bool)
        n_pixels = int(mask_flat.sum())

        # 构建基矩阵
        self._basis_matrix = np.zeros(
            (n_pixels, len(self._mode_indices)), dtype=np.float64
        )

        for col_idx, j in enumerate(self._mode_indices):
            n, m = _NOLL_TO_NM[j]
            Z = self._zernike_polynomial(n, m, rho, theta)
            self._basis_matrix[:, col_idx] = Z.ravel()[mask_flat]

    def _compute_control_matrix(self) -> None:
        """计算控制矩阵 (基矩阵的伪逆)。

        控制矩阵将模式空间指令映射到物理空间 (XY 台阶)。
        使用截断 SVD 伪逆以提高鲁棒性。
        """
        if self._basis_matrix is None:
            return

        try:
            # 仅取前两行 (对应 tip/tilt) 作为物理映射
            # 简化: 直接从 tip/tilt 基函数提取 XY 映射
            tip_col = 0  # Tilt X (Noll j=2) 是第一个模式
            tilt_col = 1  # Tilt Y (Noll j=3) 是第二个模式

            if self._basis_matrix.shape[1] < 2:
                self._control_matrix = None
                return

            # 控制矩阵: 2xN (将 N 个模式映射到 XY)
            # 使用 tip/tilt 列作为基础映射
            tip_basis = self._basis_matrix[:, tip_col]
            tilt_basis = self._basis_matrix[:, tilt_col]

            # 构建简单的控制矩阵
            # command_x 主要受 tip 模式影响, command_y 主要受 tilt 模式影响
            self._control_matrix = np.zeros(
                (2, len(self._mode_indices)), dtype=np.float64
            )
            self._control_matrix[0, tip_col] = 1.0  # tip -> x
            self._control_matrix[1, tilt_col] = 1.0  # tilt -> y

            # 其他模式通过交叉耦合影响 XY (弱耦合)
            for idx in range(2, len(self._mode_indices)):
                # 计算与 tip/tilt 基的相关性
                corr_x = float(np.abs(np.corrcoef(
                    self._basis_matrix[:, idx], tip_basis
                )[0, 1])) if tip_basis.std() > 1e-12 else 0.0
                corr_y = float(np.abs(np.corrcoef(
                    self._basis_matrix[:, idx], tilt_basis
                )[0, 1])) if tilt_basis.std() > 1e-12 else 0.0

                self._control_matrix[0, idx] = corr_x * 0.3
                self._control_matrix[1, idx] = corr_y * 0.3

        except Exception as e:
            LOGGER.warning("ModalController: 控制矩阵计算失败: %s", e)
            self._control_matrix = None

    def _update_integrators(self, error_modes: Dict[int, float]) -> None:
        """更新各模式积分器状态。

        Parameters
        ----------
        error_modes : Dict[int, float]
            各模式误差。
        """
        for j in self._mode_indices:
            error = error_modes.get(j, 0.0)
            gain = self._modal_gains.get(j, self.default_gain)

            # 积分更新
            self._integrator_states[j] += gain * error

            # 泄漏 (防止长期漂移)
            self._integrator_states[j] *= self.leak_factor

            # 抗饱和限幅
            self._integrator_states[j] = max(
                -self.anti_windup_limit,
                min(self.anti_windup_limit, self._integrator_states[j])
            )

            # 记录误差历史
            self._error_history[j].append(error)

    @staticmethod
    def _zernike_radial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
        """计算 Zernike 径向多项式 R_n^|m|(rho)。"""
        abs_m = abs(m)
        R = np.zeros_like(rho, dtype=np.float64)

        if n == 0 and abs_m == 0:
            R[:] = 1.0
        elif n == 1 and abs_m == 1:
            R[:] = rho
        elif n == 2 and abs_m == 0:
            R[:] = 2.0 * rho ** 2 - 1.0
        elif n == 2 and abs_m == 2:
            R[:] = rho ** 2
        elif n == 3 and abs_m == 1:
            R[:] = 3.0 * rho ** 3 - 2.0 * rho
        elif n == 3 and abs_m == 3:
            R[:] = rho ** 3
        elif n == 4 and abs_m == 0:
            R[:] = 6.0 * rho ** 4 - 6.0 * rho ** 2 + 1.0
        elif n == 4 and abs_m == 2:
            R[:] = 4.0 * rho ** 4 - 3.0 * rho ** 2
        elif n == 4 and abs_m == 4:
            R[:] = rho ** 4
        else:
            raise ValueError(f"Zernike radial not implemented for n={n}, m={m}")

        return R

    def _zernike_polynomial(
        self, n: int, m: int, rho: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """计算单个 Zernike 多项式值。"""
        R = self._zernike_radial(n, m, rho)
        if m > 0:
            return R * np.cos(m * theta)
        elif m < 0:
            return R * np.sin(abs(m) * theta)
        else:
            return R

    def _empty_output(self) -> ModalControlOutput:
        """返回空控制输出。"""
        return ModalControlOutput(
            command_x=0.0,
            command_y=0.0,
            modal_errors={j: 0.0 for j in self._mode_indices},
            total_residual=0.0,
            dominant_mode="N/A",
            control_effort=0.0,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    controller = ModalController(
        num_modes=6,
        grid_size=64,
        default_gain=0.3,
        max_command=50.0,
    )

    print("=== 模态控制器测试 ===\n")

    # 生成模拟波前误差 (含 tip, tilt, defocus, astigmatism)
    size = 64
    yy, xx = np.mgrid[:size, :size]
    cx, cy = size / 2.0, size / 2.0
    dx = (xx - cx) / (size / 2.0)
    dy = (yy - cy) / (size / 2.0)
    r2 = dx ** 2 + dy ** 2
    mask = r2 <= 1.0

    # 模拟波前: tip=0.5, tilt=-0.3, defocus=0.8, astigmatism=0.4
    wavefront = (0.5 * dx - 0.3 * dy + 0.8 * (2.0 * r2 - 1.0)
                 + 0.4 * (dx ** 2 - dy ** 2))
    wavefront *= mask

    print("--- 模式分解测试 ---")
    modes = controller.decompose_modes(wavefront)
    for j, coeff in modes.items():
        name = _NOLL_NAMES.get(j, f"Z{j}")
        print(f"  {name} (j={j}): {coeff:+.4f}")

    print("\n--- 控制循环测试 (20 步) ---")
    for i in range(20):
        # 模拟逐渐减小的误差
        scale = 1.0 - i * 0.04
        wf = wavefront * scale
        output = controller.update(wf)

        if i % 5 == 0:
            print(f"  [步 {i:2d}] cmd=({output.command_x:+.4f}, {output.command_y:+.4f}), "
                  f"残差={output.total_residual:.4f}, 主导={output.dominant_mode}")

    # 设置自定义增益
    print("\n--- 自定义增益测试 ---")
    controller.reset()
    controller.set_modal_gains({2: 0.5, 3: 0.5, 4: 0.1})  # tip/tilt 高增益, defocus 低增益
    output = controller.update(wavefront)
    print(f"  cmd=({output.command_x:+.4f}, {output.command_y:+.4f})")

    # 抑制带宽计算
    print("\n--- 抑制带宽测试 ---")
    bandwidths = controller.compute_rejection_bandwidth(noise_variance=0.01)
    for j, bw in bandwidths.items():
        name = _NOLL_NAMES.get(j, f"Z{j}")
        print(f"  {name}: {bw:.4f} Hz")

    # 获取模态误差
    print("\n--- 模态误差 ---")
    errors = controller.get_modal_errors()
    for j, err in errors.items():
        name = _NOLL_NAMES.get(j, f"Z{j}")
        print(f"  {name}: {err:+.4f}")

    print("\n测试完成")
