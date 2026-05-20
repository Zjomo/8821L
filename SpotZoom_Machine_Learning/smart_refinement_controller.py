"""
智能精修控制器 (SmartRefinementController)

灵感来源:
- ViSP (Visual Servoing Platform) — 视觉伺服中的多阶段收敛策略
- python-control LQR — 线性二次调节器的自适应收敛控制
- ARTIQ Adaptive Sequencing — 量子实验中的自适应精修序列

算法原理:
- Multi-Stage Convergence — 多阶段收敛策略: 粗 -> 中 -> 细 -> 超细，
  每阶段使用不同的步长和增益参数
- Convergence Rate Detection — 收敛速率检测，通过指数衰减拟合判断
  当前收敛状态，自动切换阶段
- Adaptive Step Size with Momentum — 自适应步长 + 动量加速，
  类 Nesterov 加速梯度法提高收敛速度
- Convergence Prediction — 基于历史误差的收敛预测，估计剩余迭代次数
  和最终误差
- Local Minima Escape — 局部极小值逃逸策略，当检测到收敛停滞时
  施加随机扰动
- Confidence Estimation — 置信度估计，综合收敛速率、误差水平和
  历史一致性给出终止置信度

功能:
- 自动多阶段精修 (粗 -> 中 -> 细 -> 超细)
- 基于收敛速率的自动阶段切换
- 自适应步长与动量加速
- 收敛预测 (剩余迭代、最终误差)
- 局部极小值逃逸
- 自动终止与置信度评估

依赖: numpy (无外部控制理论库)
"""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger("SpotZoom.SmartRefinementController")

# 模块默认禁用标志
smart_refinement_enabled: bool = False


@dataclass
class RefinementStage:
    """精修阶段配置。"""
    name: str  # 阶段名称
    max_step: float  # 最大步长 (像素)
    min_step: float  # 最小步长 (像素)
    gain: float  # 控制增益
    momentum: float  # 动量系数 [0, 1)
    convergence_threshold: float  # 该阶段的收敛阈值 (误差)


@dataclass
class RefinementCommand:
    """精修控制指令。"""
    delta_x: float  # X 方向增量 (像素)
    delta_y: float  # Y 方向增量 (像素)
    stage: int  # 当前阶段索引
    step_size: float  # 当前步长
    confidence: float  # 置信度 [0, 1]
    should_terminate: bool  # 是否应终止
    reason: str  # 终止原因 (空字符串表示不终止)


@dataclass
class RefinementProgress:
    """精修进度报告。"""
    current_stage: int  # 当前阶段索引
    current_stage_name: str  # 当前阶段名称
    total_iterations: int  # 总迭代次数
    error_history: List[float]  # 误差历史
    convergence_rate: float  # 当前收敛速率
    estimated_remaining: int  # 估计剩余迭代次数
    confidence: float  # 终止置信度


class SmartRefinementController:
    """智能精修控制器。

    实现多阶段粗到细收敛策略，自动在探索 (大步长、快速收敛) 和
    开发 (小步长、高精度) 之间切换。

    四个阶段:
    0. 粗调 (Coarse): 大步长，快速逼近目标
    1. 中调 (Medium): 中等步长，平衡速度与精度
    2. 精调 (Fine): 小步长，高精度定位
    3. 超精调 (Ultra-fine): 极小步长，极限精度

    Parameters
    ----------
    target_error : float
        目标误差 (像素)。达到此误差时终止。
    max_iterations : int
        最大迭代次数。
    initial_position : Optional[np.ndarray]
        初始位置 [x, y]。为 None 时使用 [0, 0]。
    stuck_threshold : int
        判定收敛停滞的迭代次数阈值。
    perturbation_magnitude : float
        局部极小值逃逸扰动幅度 (像素)。
    nesterov_factor : float
        Nesterov 加速因子 [0, 1)。
    """

    def __init__(
        self,
        target_error: float = 0.1,
        max_iterations: int = 500,
        initial_position: Optional[np.ndarray] = None,
        stuck_threshold: int = 15,
        perturbation_magnitude: float = 2.0,
        nesterov_factor: float = 0.9,
    ):
        self.target_error = float(target_error)
        self.max_iterations = int(max_iterations)
        self.initial_position = (
            initial_position.copy().astype(np.float64)
            if initial_position is not None
            else np.array([0.0, 0.0], dtype=np.float64)
        )
        self.stuck_threshold = int(stuck_threshold)
        self.perturbation_magnitude = float(perturbation_magnitude)
        self.nesterov_factor = float(nesterov_factor)

        # 定义四个精修阶段
        self._stages: List[RefinementStage] = [
            RefinementStage(
                name="粗调 (Coarse)",
                max_step=20.0,
                min_step=2.0,
                gain=0.8,
                momentum=0.3,
                convergence_threshold=5.0,
            ),
            RefinementStage(
                name="中调 (Medium)",
                max_step=5.0,
                min_step=0.5,
                gain=0.5,
                momentum=0.5,
                convergence_threshold=1.0,
            ),
            RefinementStage(
                name="精调 (Fine)",
                max_step=1.0,
                min_step=0.05,
                gain=0.3,
                momentum=0.7,
                convergence_threshold=0.3,
            ),
            RefinementStage(
                name="超精调 (Ultra-fine)",
                max_step=0.2,
                min_step=0.005,
                gain=0.15,
                momentum=0.85,
                convergence_threshold=self.target_error,
            ),
        ]

        # 内部状态
        self._current_stage: int = 0
        self._iteration: int = 0
        self._position: np.ndarray = self.initial_position.copy()
        self._velocity: np.ndarray = np.zeros(2, dtype=np.float64)  # 动量
        self._error_history: Deque[float] = deque(maxlen=200)
        self._position_history: Deque[np.ndarray] = deque(maxlen=200)
        self._stuck_counter: int = 0
        self._prev_error: Optional[float] = None
        self._convergence_rate: float = 0.0
        self._best_error: float = float('inf')
        self._best_position: np.ndarray = self.initial_position.copy()

        LOGGER.info(
            "SmartRefinementController: 初始化完成 "
            "(target=%.3f px, max_iter=%d, stages=%d)",
            self.target_error, self.max_iterations, len(self._stages),
        )

    def update(
        self,
        current_error: float,
        current_position: np.ndarray,
    ) -> RefinementCommand:
        """主控制更新: 根据当前误差和位置计算精修指令。

        Parameters
        ----------
        current_error : float
            当前误差 (像素)。
        current_position : np.ndarray
            当前位置 [x, y]。

        Returns
        -------
        RefinementCommand
            精修控制指令。
        """
        try:
            current_error = float(current_error)
            current_position = np.array(current_position, dtype=np.float64).flatten()
            if len(current_position) < 2:
                current_position = np.array([current_position[0], 0.0], dtype=np.float64)

            self._iteration += 1
            self._position = current_position.copy()
            self._error_history.append(current_error)
            self._position_history.append(current_position.copy())

            # 更新最佳记录
            if current_error < self._best_error:
                self._best_error = current_error
                self._best_position = current_position.copy()

            # 选择阶段
            stage_idx = self.select_stage(current_error)
            self._current_stage = stage_idx
            stage = self._stages[stage_idx]

            # 计算收敛速率
            self._convergence_rate = self._compute_convergence_rate()

            # 检测停滞
            self._update_stuck_counter(current_error)

            # 计算控制指令
            delta = self._compute_delta(current_error, current_position, stage)

            # 更新动量
            self._velocity = (
                stage.momentum * self._velocity + (1.0 - stage.momentum) * delta
            )

            # Nesterov 前瞻
            lookahead_position = current_position + self.nesterov_factor * self._velocity

            # 计算步长 (自适应)
            step_size = self._compute_adaptive_step(current_error, stage)

            # 方向归一化
            vel_norm = np.linalg.norm(self._velocity)
            if vel_norm > 1e-12:
                direction = self._velocity / vel_norm
            else:
                direction = np.zeros(2, dtype=np.float64)

            command = direction * step_size

            # 限幅
            command = np.clip(command, -stage.max_step, stage.max_step)

            # 置信度
            confidence = self.compute_confidence(
                list(self._error_history), self._convergence_rate
            )

            # 终止判定
            should_terminate, reason = self._check_termination(
                current_error, confidence
            )

            # 局部极小值逃逸
            if self._stuck_counter >= self.stuck_threshold and not should_terminate:
                perturbation = self.apply_perturbation(
                    current_position, self._stuck_counter
                )
                command = command + perturbation
                self._stuck_counter = 0
                LOGGER.info(
                    "SmartRefinementController: 施加扰动逃逸局部极小值 "
                    "(stuck=%d)", self.stuck_counter,
                )

            self._prev_error = current_error

            LOGGER.debug(
                "SmartRefinementController: iter=%d, stage=%s, error=%.4f, "
                "step=%.4f, cmd=(%.4f, %.4f), conf=%.3f",
                self._iteration, stage.name, current_error, step_size,
                command[0], command[1], confidence,
            )

            return RefinementCommand(
                delta_x=round(command[0], 6),
                delta_y=round(command[1], 6),
                stage=stage_idx,
                step_size=round(step_size, 6),
                confidence=round(confidence, 4),
                should_terminate=should_terminate,
                reason=reason,
            )

        except Exception as e:
            LOGGER.error("SmartRefinementController: 控制更新失败: %s", e)
            return RefinementCommand(
                delta_x=0.0, delta_y=0.0, stage=0, step_size=0.0,
                confidence=0.0, should_terminate=True,
                reason=f"错误: {e}",
            )

    def select_stage(self, error_magnitude: float) -> int:
        """根据当前误差大小选择精修阶段。

        Parameters
        ----------
        error_magnitude : float
            当前误差 (像素)。

        Returns
        -------
        int
            阶段索引 (0~3)。
        """
        for idx, stage in enumerate(self._stages):
            if error_magnitude > stage.convergence_threshold:
                return idx

        # 误差小于所有阶段阈值，使用最后阶段
        return len(self._stages) - 1

    def predict_convergence(
        self,
        error_history: List[float],
    ) -> Tuple[int, float]:
        """预测收敛所需的剩余迭代次数和最终误差。

        使用指数衰减拟合: error(t) = A * exp(-lambda * t) + C

        Parameters
        ----------
        error_history : List[float]
            误差历史。

        Returns
        -------
        Tuple[int, float]
            (估计剩余迭代次数, 估计最终误差)。
        """
        if len(error_history) < 5:
            return (self.max_iterations - self._iteration, float(error_history[-1])
                    if error_history else 0.0)

        errors = np.array(error_history[-50:], dtype=np.float64)  # 最近 50 个
        n = len(errors)
        t = np.arange(n, dtype=np.float64)

        # 指数衰减拟合 (线性化: log(error) = log(A) - lambda * t)
        log_errors = np.log(np.maximum(errors, 1e-12))

        try:
            # 线性回归
            t_mean = np.mean(t)
            le_mean = np.mean(log_errors)
            numerator = float(np.sum((t - t_mean) * (log_errors - le_mean)))
            denominator = float(np.sum((t - t_mean) ** 2))

            if denominator < 1e-12:
                return (self.max_iterations - self._iteration, float(errors[-1]))

            slope = numerator / denominator  # -lambda
            intercept = le_mean - slope * t_mean  # log(A)

            lambda_rate = -slope  # 衰减率

            if lambda_rate <= 1e-6:
                # 不收敛
                return (self.max_iterations - self._iteration, float(errors[-1]))

            # 估计达到 target_error 所需的迭代数
            A = np.exp(intercept)
            if A <= self.target_error:
                return (0, self.target_error)

            # error(t) = A * exp(-lambda * t) = target
            # t = log(A / target) / lambda
            remaining = int(np.ceil(
                np.log(max(A / self.target_error, 1.0)) / lambda_rate
            ))

            # 估计最终误差 (外推到 max_iterations)
            final_error = A * np.exp(-lambda_rate * self.max_iterations)
            final_error = max(final_error, 0.0)

            return (max(remaining, 0), round(final_error, 6))

        except Exception:
            return (self.max_iterations - self._iteration, float(errors[-1])
                    if len(errors) > 0 else 0.0)

    def apply_perturbation(
        self,
        current_position: np.ndarray,
        stuck_counter: int,
    ) -> np.ndarray:
        """施加扰动以逃逸局部极小值。

        扰动策略:
        - 随机方向
        - 幅度随停滞次数递增
        - 优先尝试远离最近访问位置的方向

        Parameters
        ----------
        current_position : np.ndarray
            当前位置 [x, y]。
        stuck_counter : int
            停滞计数器。

        Returns
        -------
        np.ndarray
            扰动向量 [dx, dy]。
        """
        # 扰动幅度随停滞次数递增
        magnitude = self.perturbation_magnitude * (1.0 + 0.5 * min(stuck_counter, 10))

        # 随机方向
        angle = np.random.uniform(0, 2.0 * np.pi)
        perturbation = np.array([
            magnitude * np.cos(angle),
            magnitude * np.sin(angle),
        ])

        # 如果有历史位置，尝试远离最近的位置
        if len(self._position_history) > 5:
            recent = np.array(list(self._position_history)[-5:])
            center = np.mean(recent, axis=0)
            away = current_position - center
            away_norm = np.linalg.norm(away)
            if away_norm > 1e-6:
                # 混合随机方向和远离方向
                away_direction = away / away_norm
                perturbation = 0.5 * perturbation + 0.5 * away_direction * magnitude

        return perturbation

    def compute_confidence(
        self,
        error_history: List[float],
        convergence_rate: float,
    ) -> float:
        """计算终止置信度。

        综合以下因素:
        - 当前误差是否接近目标
        - 收敛速率是否足够快
        - 误差历史的一致性 (单调递减)

        Parameters
        ----------
        error_history : List[float]
            误差历史。
        convergence_rate : float
            收敛速率。

        Returns
        -------
        float
            置信度 [0, 1]。
        """
        if len(error_history) < 3:
            return 0.0

        current_error = error_history[-1]

        # 因素 1: 误差接近度
        if self.target_error > 1e-12:
            error_proximity = max(0.0, 1.0 - current_error / (self.target_error * 10.0))
        else:
            error_proximity = 1.0 if current_error < 0.01 else 0.0

        # 因素 2: 收敛速率
        rate_score = min(abs(convergence_rate) * 10.0, 1.0)

        # 因素 3: 单调性 (最近 10 步中误差下降的比例)
        recent = error_history[-10:] if len(error_history) >= 10 else error_history
        decreasing_count = sum(
            1 for i in range(1, len(recent)) if recent[i] <= recent[i - 1]
        )
        monotonicity = decreasing_count / max(len(recent) - 1, 1)

        # 因素 4: 误差稳定性 (最近误差的变异系数)
        recent_arr = np.array(recent, dtype=np.float64)
        if np.mean(recent_arr) > 1e-12:
            cv = float(np.std(recent_arr) / np.mean(recent_arr))
            stability = 1.0 / (1.0 + cv * 5.0)
        else:
            stability = 1.0

        # 综合置信度
        confidence = (
            0.35 * error_proximity +
            0.25 * rate_score +
            0.20 * monotonicity +
            0.20 * stability
        )

        return float(np.clip(confidence, 0.0, 1.0))

    def get_progress_report(self) -> RefinementProgress:
        """获取精修进度报告。

        Returns
        -------
        RefinementProgress
            进度报告。
        """
        remaining, final_error = self.predict_convergence(list(self._error_history))

        stage_idx = self._current_stage
        stage_name = (self._stages[stage_idx].name
                      if stage_idx < len(self._stages) else "Unknown")

        return RefinementProgress(
            current_stage=stage_idx,
            current_stage_name=stage_name,
            total_iterations=self._iteration,
            error_history=list(self._error_history),
            convergence_rate=round(self._convergence_rate, 6),
            estimated_remaining=remaining,
            confidence=round(self.compute_confidence(
                list(self._error_history), self._convergence_rate
            ), 4),
        )

    def reset(self) -> None:
        """重置控制器，清除所有状态。"""
        self._current_stage = 0
        self._iteration = 0
        self._position = self.initial_position.copy()
        self._velocity = np.zeros(2, dtype=np.float64)
        self._error_history.clear()
        self._position_history.clear()
        self._stuck_counter = 0
        self._prev_error = None
        self._convergence_rate = 0.0
        self._best_error = float('inf')
        self._best_position = self.initial_position.copy()

        LOGGER.info("SmartRefinementController: 控制器已重置")

    # ======================== 内部方法 ========================

    def _compute_delta(
        self,
        error: float,
        position: np.ndarray,
        stage: RefinementStage,
    ) -> np.ndarray:
        """计算梯度方向增量。

        使用数值差分估计梯度方向 (简化版)。
        在实际应用中，应使用图像雅可比或光流来估计。

        Parameters
        ----------
        error : float
            当前误差。
        position : np.ndarray
            当前位置。
        stage : RefinementStage
            当前阶段。

        Returns
        -------
        np.ndarray
            增量向量 [dx, dy]。
        """
        # 简化: 使用误差梯度方向
        # 在实际系统中，这里应该用图像雅可比计算
        # 这里使用一个简单的启发式方法

        if self._prev_error is not None and len(self._position_history) >= 2:
            # 使用最近两步的位置差和误差差来估计梯度
            prev_pos = self._position_history[-2] if len(self._position_history) >= 2 \
                else position
            delta_pos = position - prev_pos
            delta_error = error - self._prev_error

            delta_norm = np.linalg.norm(delta_pos)
            if delta_norm > 1e-12:
                # 梯度方向 = -delta_error / delta_pos (沿误差减小方向)
                gradient = -delta_error / (delta_norm ** 2) * delta_pos
                gradient_norm = np.linalg.norm(gradient)
                if gradient_norm > 1e-12:
                    return gradient / gradient_norm * stage.gain * error

        # 降级: 返回零向量
        return np.zeros(2, dtype=np.float64)

    def _compute_adaptive_step(
        self,
        error: float,
        stage: RefinementStage,
    ) -> float:
        """计算自适应步长。

        步长随误差线性缩放，限制在 [min_step, max_step] 范围内。

        Parameters
        ----------
        error : float
            当前误差。
        stage : RefinementStage
            当前阶段。

        Returns
        -------
        float
            步长。
        """
        # 线性缩放
        ratio = min(error / max(stage.convergence_threshold * 2.0, 1e-6), 1.0)
        step = stage.min_step + ratio * (stage.max_step - stage.min_step)

        # 收敛加速: 如果收敛速率好，可以适当增大步长
        if self._convergence_rate < -0.05:
            step *= 1.2

        return max(stage.min_step, min(step, stage.max_step))

    def _compute_convergence_rate(self) -> float:
        """计算当前收敛速率。

        使用最近 N 步的线性回归斜率。

        Returns
        -------
        float
            收敛速率 (负值表示收敛)。
        """
        if len(self._error_history) < 5:
            return 0.0

        recent = list(self._error_history)[-20:]
        n = len(recent)
        t = np.arange(n, dtype=np.float64)
        y = np.array(recent, dtype=np.float64)

        t_mean = np.mean(t)
        y_mean = np.mean(y)

        numerator = float(np.sum((t - t_mean) * (y - y_mean)))
        denominator = float(np.sum((t - t_mean) ** 2))

        if denominator < 1e-12:
            return 0.0

        return numerator / denominator

    def _update_stuck_counter(self, current_error: float) -> None:
        """更新停滞计数器。

        Parameters
        ----------
        current_error : float
            当前误差。
        """
        if self._prev_error is not None:
            improvement = abs(self._prev_error - current_error)
            if improvement < 1e-6:
                self._stuck_counter += 1
            else:
                self._stuck_counter = max(0, self._stuck_counter - 1)

    def _check_termination(
        self,
        current_error: float,
        confidence: float,
    ) -> Tuple[bool, str]:
        """检查是否应终止精修。

        Parameters
        ----------
        current_error : float
            当前误差。
        confidence : float
            置信度。

        Returns
        -------
        Tuple[bool, str]
            (是否终止, 原因)。
        """
        # 达到目标误差
        if current_error <= self.target_error:
            return (True, f"达到目标误差 ({current_error:.4f} <= {self.target_error:.4f})")

        # 超过最大迭代次数
        if self._iteration >= self.max_iterations:
            return (True, f"达到最大迭代次数 ({self._iteration})")

        # 高置信度终止
        if confidence > 0.95 and current_error < self.target_error * 3.0:
            return (True, f"高置信度终止 (conf={confidence:.3f}, error={current_error:.4f})")

        # 严重停滞
        if self._stuck_counter >= self.stuck_threshold * 3:
            return (True, f"收敛停滞 (stuck={self._stuck_counter})")

        return (False, "")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    controller = SmartRefinementController(
        target_error=0.1,
        max_iterations=300,
        initial_position=np.array([50.0, 30.0]),
        stuck_threshold=15,
        perturbation_magnitude=2.0,
    )

    print("=== 智能精修控制器测试 ===\n")

    # 模拟精修过程
    # 目标位置: (0, 0)，初始位置: (50, 30)
    target = np.array([0.0, 0.0])
    position = np.array([50.0, 30.0])

    print("--- 精修过程 ---")
    for i in range(300):
        # 计算误差 (到目标的距离)
        error = float(np.linalg.norm(position - target))

        # 添加少量噪声模拟测量不确定性
        error_noisy = error + np.random.normal(0, 0.05)
        error_noisy = max(error_noisy, 0.0)

        # 获取控制指令
        cmd = controller.update(error_noisy, position)

        # 应用指令 (模拟一阶系统)
        position = position + np.array([cmd.delta_x, cmd.delta_y])

        # 打印进度
        if i % 30 == 0 or cmd.should_terminate:
            progress = controller.get_progress_report()
            print(f"  [iter {i:3d}] 阶段: {progress.current_stage_name}, "
                  f"误差: {error:.4f} px, 步长: {cmd.step_size:.4f}, "
                  f"置信度: {cmd.confidence:.3f}, "
                  f"位置: ({position[0]:.2f}, {position[1]:.2f})")
            if progress.estimated_remaining > 0:
                print(f"           预计剩余: {progress.estimated_remaining} 步, "
                      f"收敛速率: {progress.convergence_rate:.4f}")

        if cmd.should_terminate:
            print(f"\n  终止原因: {cmd.reason}")
            break

    # 最终报告
    final_error = float(np.linalg.norm(position - target))
    print(f"\n--- 最终结果 ---")
    print(f"最终位置: ({position[0]:.4f}, {position[1]:.4f})")
    print(f"最终误差: {final_error:.4f} px")
    print(f"总迭代次数: {controller._iteration}")

    progress = controller.get_progress_report()
    print(f"\n--- 进度报告 ---")
    print(f"当前阶段: {progress.current_stage_name}")
    print(f"收敛速率: {progress.convergence_rate:.6f}")
    print(f"置信度: {progress.confidence:.4f}")

    # 收敛预测测试
    print(f"\n--- 收敛预测测试 ---")
    remaining, final_est = controller.predict_convergence(progress.error_history)
    print(f"估计剩余迭代: {remaining}")
    print(f"估计最终误差: {final_est:.6f} px")

    # 多阶段测试
    print(f"\n--- 阶段选择测试 ---")
    for test_error in [100.0, 8.0, 0.5, 0.05]:
        stage_idx = controller.select_stage(test_error)
        stage = controller._stages[stage_idx]
        print(f"  误差={test_error:6.2f} px -> 阶段 {stage_idx}: {stage.name}")

    print("\n测试完成")
