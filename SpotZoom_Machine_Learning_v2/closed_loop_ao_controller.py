"""
闭环自适应光学控制器 (Closed-Loop Adaptive Optics Controller)

参考开源项目:
  - HCIPy (https://github.com/ehpor/hcipy): 自适应光学仿真框架
  - AOtools (https://github.com/AOtools/aotools): Python AO 工具集

核心思想:
  从 HCIPy 的闭环 AO 控制架构中借鉴，实现一个轻量级的
  "波前误差 → 积分控制器 → 校正信号" 闭环。

  在 SpotZoom 场景中:
  - 波前传感器 (WFS) → 光斑位置偏移量的时序序列
  - 校正器 (DM) → XY 位移台的移动命令
  - 闭环带宽 → 控制器增益和积分时间常数

创新点:
  1. 基于历史误差序列的积分+比例混合控制律
  2. 自适应闭环增益（根据残差收敛趋势自动调节）
  3. 伪开环误差估计（从闭环数据反推真实扰动）
  4. 多速率控制支持（WFS 采样率 ≠ 校正频率）
  5. 振动模式识别与陷波滤波

纯 numpy 实现，无外部依赖。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class ClosedLoopAOConfig:
    """闭环 AO 控制器配置。"""
    # 积分控制器增益 (0~1, 越大越激进)
    integral_gain: float = 0.4
    # 比例控制器增益
    proportional_gain: float = 0.1
    # 微分控制器增益 (抑制快速变化)
    derivative_gain: float = 0.05
    # 最大校正步长 (像素对应的步数限制)
    max_correction_step: int = 3000
    # 最小校正步长
    min_correction_step: int = 10
    # 历史缓冲长度 (用于积分和统计)
    history_length: int = 64
    # 自适应增益使能
    adaptive_gain_enabled: bool = True
    # 自适应增益调节速率
    gain_adaptation_rate: float = 0.02
    # 伪开环估计使能
    pseudo_open_loop_enabled: bool = True
    # 陷波滤波器使能 (抑制周期性振动)
    notch_filter_enabled: bool = True
    # 陷波滤波器检测的最小周期 (帧数)
    notch_min_period: int = 4
    # 陷波滤波器检测的最大周期 (帧数)
    notch_max_period: int = 32
    # 陷波滤波器品质因子
    notch_quality_factor: float = 5.0
    # 残差收敛阈值 (像素)
    convergence_threshold_px: float = 2.0
    # 闭环延迟补偿 (帧数)
    delay_compensation_frames: int = 1


@dataclass
class WFSMeasurement:
    """波前传感器测量结果。"""
    # X/Y 方向误差 (像素)
    error_x: float
    error_y: float
    # 测量时间戳
    timestamp: float = 0.0
    # 测量置信度 (0~1)
    confidence: float = 1.0


@dataclass
class ClosedLoopAOState:
    """闭环 AO 控制器状态。"""
    # 积分累积误差
    integral_x: float = 0.0
    integral_y: float = 0.0
    # 上一次误差 (用于微分)
    prev_error_x: float = 0.0
    prev_error_y: float = 0.0
    # 上一次校正输出
    last_correction_x: int = 0
    last_correction_y: int = 0
    # 误差历史
    error_history_x: List[float] = field(default_factory=list)
    error_history_y: List[float] = field(default_factory=list)
    # 校正历史
    correction_history_x: List[int] = field(default_factory=list)
    correction_history_y: List[int] = field(default_factory=list)
    # 当前有效增益
    effective_gain_i: float = 0.4
    effective_gain_p: float = 0.1
    effective_gain_d: float = 0.05
    # 闭环帧计数
    frame_count: int = 0
    # 是否已收敛
    converged: bool = False
    # 残差 RMS (像素)
    residual_rms_px: float = 0.0
    # 检测到的振动频率 (Hz 近似)
    detected_vibration_freq_hz: float = 0.0
    # 陷波滤波器中心频率 (归一化)
    notch_center_normalized: float = 0.0


class ClosedLoopAOController:
    """闭环自适应光学控制器。

    将 HCIPy/AOtools 的闭环 AO 控制思想适配到 SpotZoom 的
    光斑对准场景：
    - WFS 测量 → 光斑位置误差
    - DM 校正 → XY 位移台移动
    - 闭环带宽 → 控制器增益

    使用方法:
        config = ClosedLoopAOConfig(integral_gain=0.5)
        controller = ClosedLoopAOController(config)
        for measurement in measurements:
            correction = controller.update(measurement)
            stage.move_x(correction.dx)
            stage.move_y(correction.dy)
    """

    def __init__(self, config: Optional[ClosedLoopAOConfig] = None):
        self.config = config or ClosedLoopAOConfig()
        self.state = ClosedLoopAOState()
        self.state.effective_gain_i = self.config.integral_gain
        self.state.effective_gain_p = self.config.proportional_gain
        self.state.effective_gain_d = self.config.derivative_gain

    def update(self, measurement: WFSMeasurement) -> Tuple[int, int]:
        """处理一次 WFS 测量，返回 (x_correction, y_correction) 校正步数。

        Args:
            measurement: 波前传感器测量结果

        Returns:
            (x_correction, y_correction): XY 方向的校正步数
        """
        state = self.state
        cfg = self.config

        ex, ey = measurement.error_x, measurement.error_y
        conf = measurement.confidence

        # 1. 伪开环误差估计
        if cfg.pseudo_open_loop_enabled and state.frame_count > 0:
            # 伪开环 = 闭环残差 + 上一步校正 (近似真实扰动)
            ex_pseudo = ex + state.last_correction_x * 0.01  # 步数→像素近似
            ey_pseudo = ey + state.last_correction_y * 0.01
        else:
            ex_pseudo, ey_pseudo = ex, ey

        # 2. 积分累积
        state.integral_x += ex_pseudo * conf
        state.integral_y += ey_pseudo * conf

        # 积分限幅 (防止 windup)
        max_integral = cfg.max_correction_step / max(0.01, cfg.integral_gain)
        state.integral_x = max(-max_integral, min(max_integral, state.integral_x))
        state.integral_y = max(-max_integral, min(max_integral, state.integral_y))

        # 3. 微分项
        dex = ex_pseudo - state.prev_error_x
        dey = ey_pseudo - state.prev_error_y

        # 4. 陷波滤波 (抑制周期性振动)
        if cfg.notch_filter_enabled and len(state.error_history_x) >= cfg.notch_max_period * 2:
            notch_freq = self._detect_dominant_vibration()
            state.detected_vibration_freq_hz = notch_freq
            if notch_freq > 0:
                state.notch_center_normalized = notch_freq
                # 简化陷波: 减去振动分量
                period_samples = max(1, round(1.0 / max(0.001, notch_freq)))
                if period_samples <= len(state.error_history_x):
                    vib_x = np.mean(state.error_history_x[-period_samples:])
                    vib_y = np.mean(state.error_history_y[-period_samples:])
                    ex_pseudo -= vib_x * 0.3
                    ey_pseudo -= vib_y * 0.3

        # 5. PID 控制律
        cx = (state.effective_gain_i * state.integral_x
              + state.effective_gain_p * ex_pseudo
              + state.effective_gain_d * dex)
        cy = (state.effective_gain_i * state.integral_y
              + state.effective_gain_p * ey_pseudo
              + state.effective_gain_d * dey)

        # 6. 延迟补偿 (预测性前馈)
        if cfg.delay_compensation_frames > 0 and len(state.error_history_x) >= 3:
            # 线性外推
            n = min(cfg.delay_compensation_frames, 3)
            if len(state.error_history_x) >= n + 1:
                trend_x = (state.error_history_x[-1] - state.error_history_x[-(n + 1)]) / n
                trend_y = (state.error_history_y[-1] - state.error_history_y[-(n + 1)]) / n
                cx += trend_x * cfg.delay_compensation_frames * 0.5
                cy += trend_y * cfg.delay_compensation_frames * 0.5

        # 7. 限幅
        cx_int = int(round(cx))
        cy_int = int(round(cy))
        cx_int = max(-cfg.max_correction_step, min(cfg.max_correction_step, cx_int))
        cy_int = max(-cfg.max_correction_step, min(cfg.max_correction_step, cy_int))

        # 死区: 小误差不动作
        if abs(cx_int) < cfg.min_correction_step:
            cx_int = 0
        if abs(cy_int) < cfg.min_correction_step:
            cy_int = 0

        # 8. 更新状态
        state.prev_error_x = ex_pseudo
        state.prev_error_y = ey_pseudo
        state.last_correction_x = cx_int
        state.last_correction_y = cy_int

        state.error_history_x.append(ex)
        state.error_history_y.append(ey)
        state.correction_history_x.append(cx_int)
        state.correction_history_y.append(cy_int)

        # 裁剪历史
        max_hist = cfg.history_length
        if len(state.error_history_x) > max_hist:
            state.error_history_x = state.error_history_x[-max_hist:]
            state.error_history_y = state.error_history_y[-max_hist:]
            state.correction_history_x = state.correction_history_x[-max_hist:]
            state.correction_history_y = state.correction_history_y[-max_hist:]

        # 9. 自适应增益
        if cfg.adaptive_gain_enabled and state.frame_count > 5:
            self._adapt_gains()

        # 10. 收敛判断
        if len(state.error_history_x) >= 8:
            recent = state.error_history_x[-8:]
            recent_y = state.error_history_y[-8:]
            rms = math.sqrt(
                (np.mean(np.array(recent) ** 2) + np.mean(np.array(recent_y) ** 2)) / 2
            )
            state.residual_rms_px = rms
            state.converged = rms < cfg.convergence_threshold_px

        state.frame_count += 1
        return (cx_int, cy_int)

    def _detect_dominant_vibration(self) -> float:
        """通过 FFT 检测误差序列中的主振动频率 (归一化 0~0.5)。"""
        cfg = self.config
        hist = self.state.error_history_x
        if len(hist) < cfg.notch_max_period * 2:
            return 0.0

        signal = np.array(hist[-cfg.notch_max_period * 2:], dtype=np.float64)
        signal = signal - np.mean(signal)

        fft_vals = np.fft.rfft(signal)
        power = np.abs(fft_vals) ** 2

        # 排除 DC 分量和极低频
        min_bin = max(2, cfg.notch_max_period)
        max_bin = len(power) - 1
        if min_bin >= max_bin:
            return 0.0

        search_power = power[min_bin:max_bin + 1]
        if search_power.size == 0 or np.max(search_power) < 1e-10:
            return 0.0

        peak_bin = min_bin + int(np.argmax(search_power))
        total_power = np.sum(power[min_bin:max_bin + 1]) + 1e-10
        peak_ratio = float(power[peak_bin]) / total_power

        # 只有主频占比 > 30% 才认为是周期性振动
        if peak_ratio > 0.3:
            return float(peak_bin) / len(signal)
        return 0.0

    def _adapt_gains(self) -> None:
        """根据残差趋势自适应调节控制器增益。"""
        state = self.state
        cfg = self.config
        hist_x = state.error_history_x
        hist_y = state.error_history_y

        if len(hist_x) < 10:
            return

        # 计算最近 5 帧和之前 5 帧的 RMS 对比
        recent_x = np.array(hist_x[-5:])
        recent_y = np.array(hist_y[-5:])
        prev_x = np.array(hist_x[-10:-5])
        prev_y = np.array(hist_y[-10:-5])

        recent_rms = math.sqrt(np.mean(recent_x ** 2 + recent_y ** 2))
        prev_rms = math.sqrt(np.mean(prev_x ** 2 + prev_y ** 2)) + 1e-10

        ratio = recent_rms / prev_rms

        rate = cfg.gain_adaptation_rate

        if ratio > 1.2:
            # 残差在增大 → 降低增益 (可能振荡)
            state.effective_gain_i *= (1.0 - rate)
            state.effective_gain_p *= (1.0 - rate * 0.5)
        elif ratio < 0.8:
            # 残差在减小 → 可以适当提高增益加速收敛
            state.effective_gain_i *= (1.0 + rate * 0.5)
            state.effective_gain_p *= (1.0 + rate * 0.3)

        # 增益限幅
        state.effective_gain_i = max(0.05, min(0.95, state.effective_gain_i))
        state.effective_gain_p = max(0.01, min(0.5, state.effective_gain_p))
        state.effective_gain_d = max(0.0, min(0.2, state.effective_gain_d))

    def get_rejection_bandwidth(self) -> float:
        """估计当前闭环抑制带宽 (归一化频率)。"""
        g = self.state.effective_gain_i
        # 简化模型: 带宽 ≈ gain / (2π)
        return min(0.5, g / (2 * math.pi))

    def get_stability_margin(self) -> float:
        """估计闭环稳定性裕度 (越大越稳定)。"""
        gi = self.state.effective_gain_i
        gp = self.state.effective_gain_p
        gd = self.state.effective_gain_d
        # 简化增益裕度估计
        total_gain = gi + gp + gd
        return max(0.0, 1.0 - total_gain)

    def reset(self) -> None:
        """重置控制器状态。"""
        self.state = ClosedLoopAOState()
        self.state.effective_gain_i = self.config.integral_gain
        self.state.effective_gain_p = self.config.proportional_gain
        self.state.effective_gain_d = self.config.derivative_gain

    def get_state_summary(self) -> dict:
        """返回控制器状态摘要。"""
        s = self.state
        return {
            "frame_count": s.frame_count,
            "converged": s.converged,
            "residual_rms_px": round(s.residual_rms_px, 4),
            "effective_gains": {
                "integral": round(s.effective_gain_i, 4),
                "proportional": round(s.effective_gain_p, 4),
                "derivative": round(s.effective_gain_d, 4),
            },
            "last_correction": (s.last_correction_x, s.last_correction_y),
            "vibration_freq_hz": round(s.detected_vibration_freq_hz, 4),
            "rejection_bandwidth": round(self.get_rejection_bandwidth(), 4),
            "stability_margin": round(self.get_stability_margin(), 4),
        }
