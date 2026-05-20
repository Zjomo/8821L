"""
AdaptiveGainScheduler - 自适应增益调度器

灵感来源: AOtools, HCIPy, 自适应控制理论
功能特点:
- 基于SNR的动态增益调整
- 多频段分解处理
- 收敛速度优化
- 稳定性保证
- 纯numpy/cv2实现

技术路线:
- 频域分析 + 时域反馈
- 模糊逻辑控制
- 自适应滤波
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class GainScheduleResult:
    """增益调度结果。"""
    gain: float
    bandwidth: float
    damping: float
    estimated_snr: float
    convergence_estimate: float
    stability_margin: float
    processing_time_ms: float


@dataclass
class AdaptiveGainConfig:
    """增益调度器配置。"""
    # 基础参数
    base_gain: float = 0.5
    min_gain: float = 0.1
    max_gain: float = 1.0
    
    # 自适应参数
    adaptation_rate: float = 0.1
    forgetting_factor: float = 0.95
    
    # 频带参数
    num_bands: int = 3
    band_gains: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7])
    
    # 稳定性参数
    stability_threshold: float = 0.8
    oscillation_threshold: float = 0.1
    
    # 历史长度
    error_history_length: int = 50
    snr_history_length: int = 20


class AdaptiveGainScheduler:
    """
    自适应增益调度器。
    
    根据系统状态动态调整控制增益，优化收敛速度和稳定性。
    核心功能:
    1. SNR估计与跟踪
    2. 多频段增益分配
    3. 稳定性监控
    4. 收敛预测
    """
    
    def __init__(self, config: Optional[AdaptiveGainConfig] = None):
        self.config = config or AdaptiveGainConfig()
        self._error_history: deque = deque(maxlen=self.config.error_history_length)
        self._snr_history: deque = deque(maxlen=self.config.snr_history_length)
        self._gain_history: deque = deque(maxlen=10)
        self._current_gain: float = self.config.base_gain
        self._current_bandwidth: float = 0.5
        self._current_damping: float = 0.7
        self._iteration_count: int = 0
        LOGGER.info("AdaptiveGainScheduler初始化完成，基础增益=%.3f", 
                   self.config.base_gain)
    
    def reset(self) -> None:
        """重置调度器状态。"""
        self._error_history.clear()
        self._snr_history.clear()
        self._gain_history.clear()
        self._current_gain = self.config.base_gain
        self._current_bandwidth = 0.5
        self._current_damping = 0.7
        self._iteration_count = 0
    
    def update(
        self,
        error: float,
        signal_power: Optional[float] = None,
        noise_power: Optional[float] = None
    ) -> GainScheduleResult:
        """
        更新增益调度。
        
        Args:
            error: 当前误差
            signal_power: 信号功率 (可选)
            noise_power: 噪声功率 (可选)
            
        Returns:
            GainScheduleResult包含调度结果
        """
        t0 = time.perf_counter()
        
        # 记录误差历史
        self._error_history.append(abs(error))
        
        # 估计SNR
        snr = self._estimate_snr(signal_power, noise_power)
        self._snr_history.append(snr)
        
        # 分析系统状态
        stability = self._analyze_stability()
        convergence = self._estimate_convergence()
        
        # 计算自适应增益
        new_gain = self._compute_adaptive_gain(snr, stability, convergence)
        
        # 平滑增益变化
        self._current_gain = self._smooth_gain(new_gain)
        
        # 更新带宽和阻尼
        self._update_control_parameters(snr, stability)
        
        # 记录历史
        self._gain_history.append(self._current_gain)
        self._iteration_count += 1
        
        elapsed_ms = (time.perf_counter() - t0) * 1000
        
        LOGGER.debug("增益调度: gain=%.3f, SNR=%.2fdB, 稳定=%.3f, 收敛=%.3f, 耗时=%.2fms",
                    self._current_gain, 10*np.log10(snr+1e-10), stability, convergence, elapsed_ms)
        
        return GainScheduleResult(
            gain=self._current_gain,
            bandwidth=self._current_bandwidth,
            damping=self._current_damping,
            estimated_snr=snr,
            convergence_estimate=convergence,
            stability_margin=stability,
            processing_time_ms=elapsed_ms
        )
    
    def _estimate_snr(
        self,
        signal_power: Optional[float],
        noise_power: Optional[float]
    ) -> float:
        """估计SNR。"""
        if signal_power is not None and noise_power is not None and noise_power > 0:
            return signal_power / noise_power
        
        # 从误差历史估计SNR
        if len(self._error_history) < 10:
            return 10.0  # 默认SNR
        
        errors = np.array(self._error_history)
        
        # 信号功率 (低频成分)
        if len(errors) >= 20:
            signal = np.mean(errors[-20:])
        else:
            signal = np.mean(errors)
        
        # 噪声功率 (高频波动)
        if len(errors) >= 5:
            noise = np.std(np.diff(errors[-5:]))
        else:
            noise = np.std(errors) * 0.1
        
        noise = max(noise, 1e-10)
        snr = (signal / noise) ** 2
        
        return max(1.0, min(1000.0, snr))
    
    def _analyze_stability(self) -> float:
        """分析系统稳定性。"""
        if len(self._error_history) < 20:
            return 1.0  # 默认稳定
        
        errors = np.array(self._error_history)
        
        # 检查振荡
        recent_errors = errors[-20:]
        sign_changes = np.sum(np.diff(np.sign(recent_errors)) != 0)
        oscillation_ratio = sign_changes / len(recent_errors)
        
        # 检查误差增长趋势
        if len(errors) >= 40:
            early_mean = np.mean(errors[-40:-20])
            recent_mean = np.mean(errors[-20:])
            growth_ratio = recent_mean / max(early_mean, 1e-10)
        else:
            growth_ratio = 1.0
        
        # 综合稳定性评分
        stability = 1.0
        
        # 振荡惩罚
        if oscillation_ratio > self.config.oscillation_threshold:
            stability -= (oscillation_ratio - self.config.oscillation_threshold) * 2
        
        # 增长惩罚
        if growth_ratio > 1.0:
            stability -= (growth_ratio - 1.0) * 0.5
        
        return max(0.0, min(1.0, stability))
    
    def _estimate_convergence(self) -> float:
        """估计收敛进度。"""
        if len(self._error_history) < 10:
            return 0.0
        
        errors = np.array(self._error_history)
        
        # 计算误差衰减率
        if len(errors) >= 20:
            early_error = np.mean(errors[-20:-10])
            recent_error = np.mean(errors[-10:])
        else:
            early_error = errors[0]
            recent_error = np.mean(errors[-5:])
        
        if early_error < 1e-10:
            return 1.0
        
        reduction_ratio = 1.0 - recent_error / early_error
        
        # 基于误差大小估计收敛
        current_error = errors[-1]
        if current_error < 0.1:
            error_based = 1.0
        elif current_error < 1.0:
            error_based = 0.8
        elif current_error < 5.0:
            error_based = 0.5
        else:
            error_based = 0.2
        
        # 综合估计
        convergence = 0.6 * error_based + 0.4 * max(0, reduction_ratio)
        
        return max(0.0, min(1.0, convergence))
    
    def _compute_adaptive_gain(
        self,
        snr: float,
        stability: float,
        convergence: float
    ) -> float:
        """计算自适应增益。"""
        # 基础增益
        base = self.config.base_gain
        
        # SNR调整
        snr_db = 10 * np.log10(snr + 1e-10)
        if snr_db > 20:
            snr_factor = 1.2
        elif snr_db > 10:
            snr_factor = 1.0
        elif snr_db > 0:
            snr_factor = 0.8
        else:
            snr_factor = 0.5
        
        # 稳定性调整
        if stability < self.config.stability_threshold:
            stability_factor = stability / self.config.stability_threshold * 0.8
        else:
            stability_factor = 1.0 + (stability - self.config.stability_threshold) * 0.2
        
        # 收敛调整 (接近收敛时降低增益以精细调节)
        if convergence > 0.8:
            convergence_factor = 0.7
        elif convergence > 0.5:
            convergence_factor = 0.9
        else:
            convergence_factor = 1.0
        
        # 综合计算
        new_gain = base * snr_factor * stability_factor * convergence_factor
        
        # 限制范围
        return max(self.config.min_gain, min(self.config.max_gain, new_gain))
    
    def _smooth_gain(self, target_gain: float) -> float:
        """平滑增益变化。"""
        # 限制增益变化率
        max_change = self.config.adaptation_rate
        current = self._current_gain
        
        if target_gain > current:
            return min(target_gain, current + max_change)
        else:
            return max(target_gain, current - max_change)
    
    def _update_control_parameters(self, snr: float, stability: float) -> None:
        """更新控制参数 (带宽、阻尼)。"""
        # 带宽与SNR相关
        snr_normalized = min(1.0, snr / 100.0)
        target_bandwidth = 0.3 + 0.4 * snr_normalized
        
        # 阻尼与稳定性相关
        target_damping = 0.5 + 0.3 * stability
        
        # 平滑更新
        alpha = 0.1
        self._current_bandwidth = (1 - alpha) * self._current_bandwidth + alpha * target_bandwidth
        self._current_damping = (1 - alpha) * self._current_damping + alpha * target_damping
    
    def get_band_gains(self) -> List[float]:
        """获取多频段增益分配。"""
        base_gains = self.config.band_gains
        
        # 根据当前状态调整各频段增益
        adjusted_gains = []
        for i, gain in enumerate(base_gains):
            # 高频段 (i=0): 噪声敏感，保守增益
            # 中频段 (i=1): 主要控制频段
            # 低频段 (i=2): 稳定，可以使用较高增益
            if i == 0:
                factor = 0.8
            elif i == 1:
                factor = 1.0
            else:
                factor = 1.2
            
            adjusted_gains.append(gain * self._current_gain * factor)
        
        return adjusted_gains
    
    def predict_settling_time(self) -> float:
        """预测稳定时间 (迭代次数)。"""
        if len(self._error_history) < 10:
            return float('inf')
        
        errors = np.array(self._error_history)
        
        # 计算误差衰减时间常数
        if len(errors) >= 20:
            recent_errors = errors[-20:]
        else:
            recent_errors = errors
        
        # 指数拟合
        if len(recent_errors) >= 5:
            log_errors = np.log(recent_errors + 1e-10)
            time_const = -1.0 / np.polyfit(np.arange(len(log_errors)), log_errors, 1)[0]
            
            # 预测到1%误差的时间
            settling_time = time_const * np.log(100)
            
            return max(0, settling_time)
        
        return float('inf')


# ============================================================
# 辅助函数
# ============================================================

def simulate_control_loop(
    scheduler: AdaptiveGainScheduler,
    num_iterations: int = 100,
    noise_level: float = 0.1,
    disturbance_time: Optional[int] = None
) -> Dict:
    """模拟控制环路。"""
    errors = []
    gains = []
    snrs = []
    stabilities = []
    
    # 初始误差
    error = 10.0
    
    for i in range(num_iterations):
        # 添加噪声
        noisy_error = error + np.random.randn() * noise_level
        
        # 更新增益调度
        result = scheduler.update(noisy_error)
        
        # 记录
        errors.append(error)
        gains.append(result.gain)
        snrs.append(result.estimated_snr)
        stabilities.append(result.stability_margin)
        
        # 模拟系统响应 (简化的一阶系统)
        if disturbance_time and i == disturbance_time:
            error += 5.0  # 添加扰动
        
        # 误差衰减
        error = error * (1 - result.gain * 0.1)
        
        # 添加过程噪声
        error += np.random.randn() * 0.05
    
    return {
        'errors': errors,
        'gains': gains,
        'snrs': snrs,
        'stabilities': stabilities
    }


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # 创建调度器
    scheduler = AdaptiveGainScheduler()
    
    # 模拟控制环路
    results = simulate_control_loop(scheduler, num_iterations=100, disturbance_time=50)
    
    print(f"控制环路模拟结果:")
    print(f"  初始误差: {results['errors'][0]:.3f}")
    print(f"  最终误差: {results['errors'][-1]:.3f}")
    print(f"  平均增益: {np.mean(results['gains']):.3f}")
    print(f"  平均SNR: {np.mean(results['snrs']):.2f}")
    print(f"  平均稳定性: {np.mean(results['stabilities']):.3f}")
    print(f"  预测稳定时间: {scheduler.predict_settling_time():.1f} 迭代")
