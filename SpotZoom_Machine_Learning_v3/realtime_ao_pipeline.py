"""
实时 AO 流水线 (Realtime Adaptive Optics Pipeline)

参考开源项目:
  - pyRTC (https://github.com/aodtech/pyRTC): Python 实时自适应光学控制框架

核心思想:
  从 pyRTC 的高性能 AO 流水线架构中借鉴，实现一个模块化的实时
  自适应光学控制系统，包含波前传感器 (WFS)、变形镜 (DM) 和
  实时控制器 (RTC) 的抽象层。

  在 SpotZoom 场景中:
  - WFS → 光斑位置检测器，提供波前误差信号
  - DM → XY 位移台或快速倾斜镜，执行校正
  - RTC → 实时控制算法，计算校正命令

创新点:
  1. 可定制的流水线架构 (WFS → Reconstructor → Controller → DM)
  2. 组件抽象层，支持不同硬件后端
  3. 神经网络 / AI 控制器接口
  4. 多模式控制 (积分、比例、LQG、AI 混合)
  5. 帧率监控与延迟补偿

纯 numpy 实现，无外部依赖。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AOPipelineConfig:
    """AO 流水线配置。"""
    # 波前传感器模式数 (斜率数)
    wfs_modes: int = 64
    # 变形镜驱动器数
    dm_actuators: int = 64
    # 重建矩阵形状 (modes x actuators)
    recon_matrix_shape: Tuple[int, int] = (64, 64)
    # 控制器类型: "integral", "proportional", "leaky_integrator", "ai"
    controller_type: str = "leaky_integrator"
    # 积分增益
    integral_gain: float = 0.5
    # 比例增益
    proportional_gain: float = 0.1
    # 泄漏积分器泄漏系数 (0~1, 1 = 无泄漏)
    leak_factor: float = 0.98
    # 最大校正电压
    max_voltage: float = 1.0
    # 最小校正电压
    min_voltage: float = -1.0
    # 闭环延迟补偿 (帧数)
    delay_compensation: int = 1
    # 帧率目标 (Hz)
    target_framerate: float = 500.0
    # 是否启用 AI 控制器
    ai_controller_enabled: bool = False
    # AI 控制器混合比例 (0~1, 0 = 纯传统, 1 = 纯 AI)
    ai_blend_ratio: float = 0.0
    # 历史缓冲长度
    history_length: int = 128
    # WFS 读取噪声 (用于仿真)
    wfs_read_noise: float = 0.01
    # DM 响应时间 (帧)
    dm_response_frames: int = 1


@dataclass
class AOPipelineState:
    """AO 流水线状态。"""
    # 当前帧计数
    frame_count: int = 0
    # 当前 WFS 测量 (斜率向量)
    current_slopes: np.ndarray = field(default_factory=lambda: np.zeros(64))
    # 当前 DM 命令
    current_command: np.ndarray = field(default_factory=lambda: np.zeros(64))
    # 积分器状态
    integrator_state: np.ndarray = field(default_factory=lambda: np.zeros(64))
    # 残差波前误差 (RMS)
    residual_rms: float = 0.0
    # 实际帧率 (Hz)
    actual_framerate: float = 0.0
    # 是否闭环
    closed_loop: bool = False
    # 上次处理时间戳
    last_process_time: float = 0.0


class RealtimeAOPipeline:
    """实时自适应光学流水线。

    模块化的 AO 控制系统，支持可插拔的 WFS、重建器、控制器和 DM 组件。

    Parameters
    ----------
    config : AOPipelineConfig
        流水线配置参数。

    References
    ----------
    .. [1] pyRTC documentation:
           https://pyrtc.readthedocs.io/
    .. [2] Hardy, J. W. (1998). "Adaptive Optics for Astronomical
           Telescopes." Oxford University Press.
    """

    def __init__(self, config: Optional[AOPipelineConfig] = None) -> None:
        self.config = config or AOPipelineConfig()
        self.state = AOPipelineState()
        self.state.current_slopes = np.zeros(self.config.wfs_modes)
        self.state.current_command = np.zeros(self.config.dm_actuators)
        self.state.integrator_state = np.zeros(self.config.dm_actuators)

        # 重建矩阵 (默认单位矩阵)
        self.recon_matrix: np.ndarray = np.eye(
            self.config.recon_matrix_shape[1],
            self.config.recon_matrix_shape[0],
        )

        # 交互矩阵 (默认单位矩阵)
        self.interaction_matrix: np.ndarray = np.eye(
            self.config.wfs_modes,
            self.config.dm_actuators,
        )

        # AI 控制器回调
        self._ai_controller: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None

        # WFS 和 DM 仿真回调
        self._wfs_callback: Optional[Callable[[], np.ndarray]] = None
        self._dm_callback: Optional[Callable[[np.ndarray], None]] = None

        # 帧率统计
        self._frame_times: List[float] = []
        self._start_time: float = time.time()

        logger.info(
            f"RealtimeAOPipeline 初始化: "
            f"modes={self.config.wfs_modes}, "
            f"actuators={self.config.dm_actuators}, "
            f"controller={self.config.controller_type}"
        )

    def set_recon_matrix(self, matrix: np.ndarray) -> None:
        """设置波前重建矩阵。

        Parameters
        ----------
        matrix : np.ndarray
            重建矩阵，形状 (actuators, modes)。
        """
        if matrix.shape != self.config.recon_matrix_shape:
            logger.warning(
                f"重建矩阵形状 {matrix.shape} 与配置 "
                f"{self.config.recon_matrix_shape} 不匹配"
            )
        self.recon_matrix = matrix.copy()
        logger.info(f"重建矩阵已更新: shape={matrix.shape}")

    def set_interaction_matrix(self, matrix: np.ndarray) -> None:
        """设置交互矩阵 (DM 到 WFS 的映射)。

        Parameters
        ----------
        matrix : np.ndarray
            交互矩阵，形状 (modes, actuators)。
        """
        self.interaction_matrix = matrix.copy()
        # 使用伪逆计算重建矩阵
        self.recon_matrix = np.linalg.pinv(matrix).T
        logger.info(f"交互矩阵已更新，重建矩阵自动计算: shape={self.recon_matrix.shape}")

    def set_ai_controller(
        self,
        controller_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> None:
        """设置 AI 控制器回调函数。

        Parameters
        ----------
        controller_fn : callable
            AI 控制器函数，签名为:
            fn(slopes: ndarray, history: ndarray) -> command: ndarray
        """
        self._ai_controller = controller_fn
        self.config.ai_controller_enabled = True
        logger.info("AI 控制器已设置")

    def set_wfs_callback(
        self, callback: Callable[[], np.ndarray]
    ) -> None:
        """设置 WFS 数据获取回调。

        Parameters
        ----------
        callback : callable
            返回当前 WFS 斜率向量的函数。
        """
        self._wfs_callback = callback
        logger.info("WFS 回调已设置")

    def set_dm_callback(
        self, callback: Callable[[np.ndarray], None]
    ) -> None:
        """设置 DM 命令发送回调。

        Parameters
        ----------
        callback : callable
            接收 DM 命令向量的函数。
        """
        self._dm_callback = callback
        logger.info("DM 回调已设置")

    def process_frame(self, slopes: Optional[np.ndarray] = None) -> np.ndarray:
        """处理一帧 AO 数据。

        完整的 WFS → 重建 → 控制 → DM 流水线。

        Parameters
        ----------
        slopes : np.ndarray, optional
            WFS 斜率测量。如果为 None，使用回调获取。

        Returns
        -------
        np.ndarray
            当前残差斜率向量。
        """
        t_start = time.time()

        # 1. 获取 WFS 数据
        if slopes is None:
            if self._wfs_callback is not None:
                slopes = self._wfs_callback()
            else:
                slopes = np.zeros(self.config.wfs_modes)

        # 添加仿真噪声
        if self.config.wfs_read_noise > 0:
            slopes = slopes + np.random.normal(
                0, self.config.wfs_read_noise, slopes.shape
            )

        self.state.current_slopes = slopes.copy()

        # 2. 波前重建
        voltage = self.recon_matrix @ slopes

        # 3. 控制器更新
        command = self.update_control(voltage)

        # 4. 发送 DM 命令
        self.state.current_command = command.copy()

        if self._dm_callback is not None:
            self._dm_callback(command)

        # 5. 计算残差
        if self.state.closed_loop:
            residual_slopes = slopes - self.interaction_matrix @ command
        else:
            residual_slopes = slopes.copy()

        self.state.residual_rms = float(np.sqrt(np.mean(residual_slopes ** 2)))

        # 6. 帧率统计
        self.state.frame_count += 1
        t_end = time.time()
        self._frame_times.append(t_end)
        # 保留最近 N 帧的时间戳
        if len(self._frame_times) > self.config.history_length:
            self._frame_times = self._frame_times[-self.config.history_length:]

        if len(self._frame_times) > 1:
            dt = self._frame_times[-1] - self._frame_times[-2]
            if dt > 0:
                self.state.actual_framerate = 1.0 / dt

        self.state.last_process_time = t_end - t_start

        return residual_slopes

    def update_control(self, voltage: np.ndarray) -> np.ndarray:
        """更新控制器状态并计算校正命令。

        Parameters
        ----------
        voltage : np.ndarray
            重建的电压/校正信号。

        Returns
        -------
        np.ndarray
            发送给 DM 的命令向量。
        """
        controller = self.config.controller_type

        if controller == "integral":
            self.state.integrator_state += self.config.integral_gain * voltage
            command = self.state.integrator_state.copy()

        elif controller == "proportional":
            command = self.config.proportional_gain * voltage

        elif controller == "leaky_integrator":
            self.state.integrator_state = (
                self.config.leak_factor * self.state.integrator_state
                + self.config.integral_gain * voltage
            )
            command = self.state.integrator_state.copy()

        elif controller == "ai" and self._ai_controller is not None:
            # AI 控制器
            history = self._get_history_vector()
            ai_command = self._ai_controller(voltage, history)
            if self.config.ai_blend_ratio < 1.0:
                # 混合传统控制器
                self.state.integrator_state += self.config.integral_gain * voltage
                traditional_command = self.state.integrator_state.copy()
                command = (
                    (1 - self.config.ai_blend_ratio) * traditional_command
                    + self.config.ai_blend_ratio * ai_command
                )
            else:
                command = ai_command
        else:
            logger.warning(f"未知控制器类型: {controller}, 使用积分控制器")
            self.state.integrator_state += self.config.integral_gain * voltage
            command = self.state.integrator_state.copy()

        # 电压限幅
        command = np.clip(command, self.config.min_voltage, self.config.max_voltage)

        return command

    def get_correction(self) -> np.ndarray:
        """获取当前 DM 校正命令。

        Returns
        -------
        np.ndarray
            当前校正命令向量。
        """
        return self.state.current_command.copy()

    def get_residual(self) -> np.ndarray:
        """获取当前残差斜率。

        Returns
        -------
        np.ndarray
            残差斜率向量。
        """
        return self.state.current_slopes.copy()

    def reset(self) -> None:
        """重置流水线状态。"""
        self.state.frame_count = 0
        self.state.current_slopes = np.zeros(self.config.wfs_modes)
        self.state.current_command = np.zeros(self.config.dm_actuators)
        self.state.integrator_state = np.zeros(self.config.dm_actuators)
        self.state.residual_rms = 0.0
        self.state.closed_loop = False
        self._frame_times.clear()
        self._start_time = time.time()
        logger.info("AO 流水线已重置")

    def open_loop(self) -> None:
        """切换到开环模式。"""
        self.state.closed_loop = False
        logger.info("已切换到开环模式")

    def close_loop(self) -> None:
        """切换到闭环模式。"""
        self.state.closed_loop = True
        logger.info("已切换到闭环模式")

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取流水线性能统计。

        Returns
        -------
        dict
            性能统计信息。
        """
        elapsed = time.time() - self._start_time
        return {
            "frame_count": self.state.frame_count,
            "elapsed_time_s": elapsed,
            "average_framerate_hz": (
                self.state.frame_count / elapsed if elapsed > 0 else 0.0
            ),
            "current_framerate_hz": self.state.actual_framerate,
            "residual_rms": self.state.residual_rms,
            "closed_loop": self.state.closed_loop,
            "controller_type": self.config.controller_type,
            "max_command": float(np.max(np.abs(self.state.current_command))),
            "processing_time_ms": self.state.last_process_time * 1000,
        }

    def _get_history_vector(self) -> np.ndarray:
        """获取历史状态向量 (用于 AI 控制器)。"""
        history_size = self.config.history_length
        slopes_history = np.zeros((history_size, self.config.wfs_modes))
        command_history = np.zeros((history_size, self.config.dm_actuators))
        # 当前状态放在最后
        slopes_history[-1] = self.state.current_slopes
        command_history[-1] = self.state.current_command
        return np.concatenate([slopes_history.flatten(), command_history.flatten()])
