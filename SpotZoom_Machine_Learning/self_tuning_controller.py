"""
自整定控制器 (SelfTuningController)

灵感来源:
- python-control LQR 自动整定 — 基于线性二次调节器的最优增益计算，
  通过求解 Riccati 方程自动确定状态反馈增益矩阵
- ARTIQ PID 自动整定 — 量子实验控制框架中的 PID 参数在线自动优化，
  使用继电反馈法 (Relay Feedback) 识别系统临界增益和周期
- Optuna 贝叶斯优化 — 超参数自动搜索框架，使用 TPE (Tree-structured
  Parzen Estimator) 进行高效的参数空间探索

算法原理:
- Relay Feedback Auto-Tuning (Astrom-Hagglund Method) — 继电反馈自动
  整定法，通过在系统中注入继电 (开关) 信号激发持续振荡，从振荡中
  识别临界增益 (Ku) 和临界周期 (Tu)
- Ziegler-Nichols Rules — 基于 Ku 和 Tu 的 PID 参数经验整定规则
- Critical Damping — 临界阻尼整定，追求最快无超调响应
- Performance Metrics — 实时计算超调量、调节时间、上升时间等性能指标

功能:
- 实时 PID 控制输出计算
- 继电反馈自动整定 (Astrom-Hagglund 方法)
- Ziegler-Nichols 规则整定
- 临界阻尼整定
- 控制性能指标实时评估

依赖: numpy, logging, dataclasses (无外部控制理论库)
"""

import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger("SpotZoom.SelfTuningController")


class TuningMethod(Enum):
    """整定方法。"""
    RELAY_FEEDBACK = "relay_feedback"       # 继电反馈法 (Astrom-Hagglund)
    ZIEGLER_NICHOLS = "ziegler_nichols"     # Ziegler-Nichols 规则
    CRITICAL_DAMPING = "critical_damping"   # 临界阻尼整定


@dataclass
class TuningState:
    """整定过程状态。"""
    is_tuning: bool                         # 是否正在整定
    progress: float                         # 整定进度 [0, 1]
    current_test_amplitude: float           # 当前测试信号幅值
    estimated_ultimate_gain: float          # 估计的临界增益 Ku
    estimated_ultimate_period: float        # 估计的临界周期 Tu (秒)
    method: str                             # 当前使用的整定方法
    cycles_completed: int                   # 已完成的振荡周期数
    target_cycles: int                      # 目标振荡周期数


@dataclass
class ControllerMetrics:
    """控制器性能指标。"""
    overshoot_pct: float            # 超调量百分比
    settling_time_s: float          # 调节时间 (秒, 2% 误差带)
    rise_time_s: float              # 上升时间 (秒, 10%~90%)
    steady_state_error: float       # 稳态误差
    bandwidth_hz: float             # 估计带宽 (Hz)
    samples: int                    # 参与统计的样本数


class SelfTuningController:
    """自整定 PID 控制器。

    实现带自动整定功能的 PID 控制器，支持继电反馈法、Ziegler-Nichols
    规则和临界阻尼三种整定策略。

    PID 控制律:
        u(t) = Kp * e(t) + Ki * integral(e) + Kd * d(e)/dt

    Parameters
    ----------
    kp, ki, kd : float
        初始 PID 增益。
    output_limits : Tuple[float, float]
        控制输出限幅范围。
    integral_limit : float
        积分项限幅 (防止积分饱和)。
    relay_amplitude : float
        继电反馈测试信号幅值。
    relay_target_cycles : int
        继电反馈法需要观察的振荡周期数。
    derivative_filter_alpha : float
        微分项低通滤波系数 (0~1)，越小滤波越强。
    """

    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.0,
        kd: float = 0.0,
        output_limits: Tuple[float, float] = (-100.0, 100.0),
        integral_limit: float = 50.0,
        relay_amplitude: float = 10.0,
        relay_target_cycles: int = 4,
        derivative_filter_alpha: float = 0.1,
    ):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.output_limits = (float(output_limits[0]), float(output_limits[1]))
        self.integral_limit = float(integral_limit)
        self.relay_amplitude = float(relay_amplitude)
        self.relay_target_cycles = int(relay_target_cycles)
        self.derivative_filter_alpha = float(derivative_filter_alpha)

        # PID 状态
        self._integral: float = 0.0
        self._prev_error: Optional[float] = None
        self._prev_output: Optional[float] = None
        self._filtered_derivative: float = 0.0

        # 整定状态
        self._tuning_state = TuningState(
            is_tuning=False,
            progress=0.0,
            current_test_amplitude=0.0,
            estimated_ultimate_gain=0.0,
            estimated_ultimate_period=0.0,
            method="",
            cycles_completed=0,
            target_cycles=relay_target_cycles,
        )

        # 继电反馈法内部状态
        self._relay_output: float = 0.0
        self._relay_peaks: Deque[float] = deque(maxlen=20)
        self._relay_troughs: Deque[float] = deque(maxlen=20)
        self._relay_crossing_times: Deque[float] = deque(maxlen=20)
        self._relay_last_sign: int = 0
        self._relay_start_time: Optional[float] = None

        # 性能指标跟踪
        self._response_history: Deque[Tuple[float, float]] = deque(maxlen=500)
        self._setpoint: float = 0.0
        self._response_start_time: Optional[float] = None
        self._initial_value: float = 0.0
        self._max_value: float = 0.0
        self._metrics: Optional[ControllerMetrics] = None

        LOGGER.info(
            "SelfTuningController: 初始化完成 (Kp=%.3f, Ki=%.3f, Kd=%.3f)",
            self.kp, self.ki, self.kd,
        )

    def update(self, error: float, dt: float) -> float:
        """执行一步 PID 控制计算。

        Parameters
        ----------
        error : float
            当前误差 (设定值 - 测量值)。
        dt : float
            时间步长 (秒)。

        Returns
        -------
        float
            控制输出值。
        """
        if self._tuning_state.is_tuning:
            return self._relay_step(error, dt)

        # 比例项
        p_term = self.kp * error

        # 积分项 (梯形积分 + 抗饱和)
        self._integral += 0.5 * (error + (self._prev_error or 0.0)) * dt
        self._integral = max(-self.integral_limit,
                             min(self.integral_limit, self._integral))
        i_term = self.ki * self._integral

        # 微分项 (低通滤波)
        if self._prev_error is not None and dt > 1e-9:
            raw_derivative = (error - self._prev_error) / dt
            alpha = self.derivative_filter_alpha
            self._filtered_derivative = (
                alpha * raw_derivative
                + (1.0 - alpha) * self._filtered_derivative
            )
        d_term = self.kd * self._filtered_derivative

        # 控制输出
        output = p_term + i_term + d_term
        output = max(self.output_limits[0],
                     min(self.output_limits[1], output))

        self._prev_error = error
        self._prev_output = output

        # 跟踪响应用于性能评估
        self._track_response(error, dt)

        LOGGER.debug(
            "SelfTuningController: error=%.4f, output=%.4f (P=%.4f, I=%.4f, D=%.4f)",
            error, output, p_term, i_term, d_term,
        )

        return output

    def start_auto_tune(
        self,
        method: TuningMethod = TuningMethod.RELAY_FEEDBACK,
        ku: Optional[float] = None,
        tu: Optional[float] = None,
    ) -> None:
        """启动自动整定。

        Parameters
        ----------
        method : TuningMethod
            整定方法。
        ku : float or None
            手动指定临界增益 (仅 ZIEGLER_NICHOLS 和 CRITICAL_DAMPING 需要)。
            为 None 时使用继电反馈法自动估计。
        tu : float or None
            手动指定临界周期 (秒)。为 None 时自动估计。
        """
        if self._tuning_state.is_tuning:
            LOGGER.warning("SelfTuningController: 整定已在进行中")
            return

        self._tuning_state = TuningState(
            is_tuning=True,
            progress=0.0,
            current_test_amplitude=self.relay_amplitude,
            estimated_ultimate_gain=ku or 0.0,
            estimated_ultimate_period=tu or 0.0,
            method=method.value,
            cycles_completed=0,
            target_cycles=self.relay_target_cycles,
        )

        # 重置继电反馈状态
        self._relay_peaks.clear()
        self._relay_troughs.clear()
        self._relay_crossing_times.clear()
        self._relay_last_sign = 0
        self._relay_output = self.relay_amplitude
        self._relay_start_time = time.time()

        # 重置 PID 状态
        self._integral = 0.0
        self._prev_error = None
        self._filtered_derivative = 0.0

        # 如果已有 Ku 和 Tu，直接应用规则
        if method == TuningMethod.ZIEGLER_NICHOLS and ku is not None and tu is not None:
            self._apply_ziegler_nichols(ku, tu)
            self._tuning_state.is_tuning = False
            self._tuning_state.progress = 1.0
            LOGGER.info(
                "SelfTuningController: Ziegler-Nichols 整定完成 "
                "(Ku=%.3f, Tu=%.3fs) -> Kp=%.3f, Ki=%.3f, Kd=%.3f",
                ku, tu, self.kp, self.ki, self.kd,
            )
        elif method == TuningMethod.CRITICAL_DAMPING and ku is not None and tu is not None:
            self._apply_critical_damping(ku, tu)
            self._tuning_state.is_tuning = False
            self._tuning_state.progress = 1.0
            LOGGER.info(
                "SelfTuningController: 临界阻尼整定完成 "
                "(Ku=%.3f, Tu=%.3fs) -> Kp=%.3f, Ki=%.3f, Kd=%.3f",
                ku, tu, self.kp, self.ki, self.kd,
            )
        else:
            LOGGER.info(
                "SelfTuningController: 启动 %s 自动整定",
                method.value,
            )

    def get_tuned_gains(self) -> Tuple[float, float, float]:
        """获取当前 PID 增益。

        Returns
        -------
        Tuple[float, float, float]
            (Kp, Ki, Kd)。
        """
        return (self.kp, self.ki, self.kd)

    def get_performance_metrics(self) -> ControllerMetrics:
        """获取控制器性能指标。

        Returns
        -------
        ControllerMetrics
            性能指标。如果数据不足，返回默认值。
        """
        if self._metrics is not None:
            return self._metrics

        return ControllerMetrics(
            overshoot_pct=0.0,
            settling_time_s=0.0,
            rise_time_s=0.0,
            steady_state_error=0.0,
            bandwidth_hz=0.0,
            samples=0,
        )

    def get_tuning_state(self) -> TuningState:
        """获取当前整定状态。

        Returns
        -------
        TuningState
            整定状态。
        """
        return self._tuning_state

    def reset(self) -> None:
        """重置控制器，清除所有状态。"""
        self._integral = 0.0
        self._prev_error = None
        self._prev_output = None
        self._filtered_derivative = 0.0
        self._tuning_state = TuningState(
            is_tuning=False, progress=0.0, current_test_amplitude=0.0,
            estimated_ultimate_gain=0.0, estimated_ultimate_period=0.0,
            method="", cycles_completed=0, target_cycles=self.relay_target_cycles,
        )
        self._relay_peaks.clear()
        self._relay_troughs.clear()
        self._relay_crossing_times.clear()
        self._relay_last_sign = 0
        self._response_history.clear()
        self._metrics = None

        LOGGER.info("SelfTuningController: 控制器已重置")

    # ======================== 内部方法 ========================

    def _relay_step(self, error: float, dt: float) -> float:
        """继电反馈法单步执行。

        在系统中注入开关信号，观察误差响应的振荡特性。
        """
        now = time.time()

        # 继电切换: 误差过零时翻转输出
        current_sign = 1 if error >= 0 else -1
        if self._relay_last_sign != 0 and current_sign != self._relay_last_sign:
            self._relay_crossing_times.append(now)
            self._relay_output = -self._relay_output

            # 记录峰值/谷值
            if current_sign > 0:
                self._relay_troughs.append(error)
            else:
                self._relay_peaks.append(error)

        self._relay_last_sign = current_sign

        # 检查是否收集到足够的振荡周期
        n_crossings = len(self._relay_crossing_times)
        n_cycles = n_crossings // 2

        self._tuning_state.cycles_completed = n_cycles
        self._tuning_state.progress = min(
            n_cycles / max(self.relay_target_cycles, 1), 1.0
        )

        if n_cycles >= self.relay_target_cycles:
            self._finish_relay_tuning()

        return self._relay_output

    def _finish_relay_tuning(self) -> None:
        """完成继电反馈整定，计算 Ku 和 Tu 并应用整定规则。"""
        # 计算临界周期 Tu: 两个相邻过零点间隔的两倍
        if len(self._relay_crossing_times) >= 4:
            periods = []
            for i in range(2, len(self._relay_crossing_times)):
                period = self._relay_crossing_times[i] - self._relay_crossing_times[i - 2]
                if period > 1e-9:
                    periods.append(period)

            tu = float(np.median(periods)) if periods else 1.0
        else:
            tu = 1.0

        # 计算临界增益 Ku: Ku = 4d / (pi * a)
        # d = 继电幅值, a = 振荡幅值
        if self._relay_peaks and self._relay_troughs:
            peak_amp = float(np.mean(list(self._relay_peaks)[-5:]))
            trough_amp = float(np.mean(list(self._relay_troughs)[-5:]))
            oscillation_amplitude = abs(peak_amp - trough_amp) / 2.0
        else:
            oscillation_amplitude = 1.0

        if oscillation_amplitude > 1e-9:
            ku = (4.0 * self.relay_amplitude) / (math.pi * oscillation_amplitude)
        else:
            ku = 1.0

        self._tuning_state.estimated_ultimate_gain = round(ku, 4)
        self._tuning_state.estimated_ultimate_period = round(tu, 4)

        # 根据选择的整定方法应用规则
        method = TuningMethod(self._tuning_state.method)
        if method == TuningMethod.ZIEGLER_NICHOLS:
            self._apply_ziegler_nichols(ku, tu)
        elif method == TuningMethod.CRITICAL_DAMPING:
            self._apply_critical_damping(ku, tu)
        else:
            # 继电反馈法默认使用 Ziegler-Nichols
            self._apply_ziegler_nichols(ku, tu)

        self._tuning_state.is_tuning = False
        self._tuning_state.progress = 1.0

        LOGGER.info(
            "SelfTuningController: 继电反馈整定完成 "
            "(Ku=%.3f, Tu=%.3fs) -> Kp=%.3f, Ki=%.3f, Kd=%.3f",
            ku, tu, self.kp, self.ki, self.kd,
        )

    def _apply_ziegler_nichols(self, ku: float, tu: float) -> None:
        """应用 Ziegler-Nichols PID 整定规则。

        经典 Z-N 规则:
            Kp = 0.6 * Ku
            Ki = 1.2 * Ku / Tu
            Kd = 0.075 * Ku * Tu

        Parameters
        ----------
        ku : float
            临界增益。
        tu : float
            临界周期 (秒)。
        """
        self.kp = round(0.6 * ku, 6)
        self.ki = round(1.2 * ku / max(tu, 1e-9), 6)
        self.kd = round(0.075 * ku * tu, 6)

        LOGGER.debug(
            "SelfTuningController: Z-N 规则 -> Kp=%.4f, Ki=%.4f, Kd=%.4f",
            self.kp, self.ki, self.kd,
        )

    def _apply_critical_damping(self, ku: float, tu: float) -> None:
        """应用临界阻尼整定规则。

        临界阻尼规则 (Tyreus-Luyben 改进版):
            Kp = 0.45 * Ku
            Ki = 0.54 * Ku / Tu
            Kd = 0.137 * Ku * Tu

        相比经典 Z-N，增益更保守，超调更小。

        Parameters
        ----------
        ku : float
            临界增益。
        tu : float
            临界周期 (秒)。
        """
        self.kp = round(0.45 * ku, 6)
        self.ki = round(0.54 * ku / max(tu, 1e-9), 6)
        self.kd = round(0.137 * ku * tu, 6)

        LOGGER.debug(
            "SelfTuningController: 临界阻尼规则 -> Kp=%.4f, Ki=%.4f, Kd=%.4f",
            self.kp, self.ki, self.kd,
        )

    def _track_response(self, error: float, dt: float) -> None:
        """跟踪阶跃响应用于性能指标计算。"""
        now = time.time()
        self._response_history.append((now, -error))  # 负误差 = 实际值

        if len(self._response_history) < 10:
            return

        # 检测阶跃变化 (误差突然增大)
        values = [v for _, v in self._response_history]
        recent_errors = [e for _, e in list(self._response_history)[-10:]]

        # 计算性能指标
        if len(values) >= 20:
            self._compute_metrics(values, now)


    def _compute_metrics(self, values: list, now: float) -> None:
        """从响应历史计算性能指标。

        Parameters
        ----------
        values : list
            响应值序列。
        now : float
            当前时间。
        """
        arr = np.array(values, dtype=np.float64)
        n = len(arr)

        if n < 20:
            return

        # 使用最后 100 个样本
        recent = arr[-min(100, n):]

        # 稳态值 (最后 20 个样本的均值)
        steady_state = float(np.mean(recent[-20:]))

        # 初始值 (前 5 个样本的均值)
        initial_value = float(np.mean(recent[:5]))

        # 最大值 (用于计算超调)
        max_value = float(np.max(recent))

        # 超调量
        step_size = abs(steady_state - initial_value)
        if step_size > 1e-9:
            overshoot = max(0.0, (max_value - steady_state) / step_size * 100.0)
        else:
            overshoot = 0.0

        # 上升时间 (10%~90%)
        target_10 = initial_value + 0.1 * (steady_state - initial_value)
        target_90 = initial_value + 0.9 * (steady_state - initial_value)
        t_10 = None
        t_90 = None
        for i, v in enumerate(recent):
            if t_10 is None and v >= target_10:
                t_10 = i
            if t_90 is None and v >= target_90:
                t_90 = i
                break

        rise_time = float(t_90 - t_10) if (t_10 is not None and t_90 is not None) else 0.0

        # 调节时间 (进入 2% 误差带)
        settling_time = 0.0
        band = 0.02 * step_size
        for i in range(len(recent) - 1, -1, -1):
            if abs(recent[i] - steady_state) > band:
                settling_time = float(len(recent) - 1 - i)
                break

        # 稳态误差
        steady_state_error = abs(float(np.mean(recent[-20:])) - steady_state)

        # 估计带宽 (基于上升时间)
        if rise_time > 1e-9:
            bandwidth = 0.35 / rise_time  # 经验公式
        else:
            bandwidth = 0.0

        self._metrics = ControllerMetrics(
            overshoot_pct=round(overshoot, 2),
            settling_time_s=round(settling_time, 4),
            rise_time_s=round(rise_time, 4),
            steady_state_error=round(steady_state_error, 6),
            bandwidth_hz=round(bandwidth, 2),
            samples=n,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    controller = SelfTuningController(
        kp=1.0, ki=0.1, kd=0.01,
        output_limits=(-50.0, 50.0),
        relay_amplitude=5.0,
        relay_target_cycles=4,
    )

    print("=== 自整定控制器测试 ===\n")

    # 模拟一阶系统响应: G(s) = 1 / (s + 1)
    # 离散化: x[k+1] = x[k] + dt * (-x[k] + u[k])
    dt = 0.05
    plant_state = 0.0
    setpoint = 10.0

    print("--- 阶跃响应 (整定前) ---")
    for i in range(200):
        error = setpoint - plant_state
        output = controller.update(error, dt)
        plant_state += dt * (-plant_state + output)
        plant_state = max(-100, min(100, plant_state))

        if i % 50 == 49:
            metrics = controller.get_performance_metrics()
            print(f"  [t={i*dt:.1f}s] 设定值={setpoint:.1f}, 实际值={plant_state:.3f}, "
                  f"误差={error:.3f}, 输出={output:.3f}")
            if metrics.samples > 0:
                print(f"    超调={metrics.overshoot_pct:.1f}%, "
                      f"调节时间={metrics.settling_time_s:.2f}s, "
                      f"上升时间={metrics.rise_time_s:.2f}s")

    # 自动整定
    print("\n--- 启动继电反馈自动整定 ---")
    controller.reset()
    plant_state = 0.0

    controller.start_auto_tune(method=TuningMethod.RELAY_FEEDBACK)

    for i in range(500):
        error = setpoint - plant_state
        output = controller.update(error, dt)
        plant_state += dt * (-plant_state + output)
        plant_state = max(-100, min(100, plant_state))

        ts = controller.get_tuning_state()
        if ts.is_tuning and i % 100 == 99:
            print(f"  [t={i*dt:.1f}s] 整定进度={ts.progress:.1%}, "
                  f"周期={ts.cycles_completed}/{ts.target_cycles}")

        if not ts.is_tuning and ts.progress >= 1.0:
            kp, ki, kd = controller.get_tuned_gains()
            print(f"\n  整定完成!")
            print(f"  Ku={ts.estimated_ultimate_gain:.4f}, "
                  f"Tu={ts.estimated_ultimate_period:.4f}s")
            print(f"  整定后增益: Kp={kp:.4f}, Ki={ki:.4f}, Kd={kd:.4f}")
            break

    # 整定后阶跃响应
    print("\n--- 阶跃响应 (整定后) ---")
    controller.reset()
    plant_state = 0.0

    for i in range(200):
        error = setpoint - plant_state
        output = controller.update(error, dt)
        plant_state += dt * (-plant_state + output)
        plant_state = max(-100, min(100, plant_state))

        if i % 50 == 49:
            metrics = controller.get_performance_metrics()
            print(f"  [t={i*dt:.1f}s] 设定值={setpoint:.1f}, 实际值={plant_state:.3f}, "
                  f"误差={error:.3f}, 输出={output:.3f}")
            if metrics.samples > 0:
                print(f"    超调={metrics.overshoot_pct:.1f}%, "
                      f"调节时间={metrics.settling_time_s:.2f}s, "
                      f"上升时间={metrics.rise_time_s:.2f}s")

    print("\n测试完成")
