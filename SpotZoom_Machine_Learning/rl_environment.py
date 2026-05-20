"""
强化学习对准环境 (RLAlignmentEnvironment)

灵感来源:
- Stable-Baselines3 — 强化学习算法库 (PPO/SAC/TD3)
- OpenAI Gym — RL 环境接口标准
- DeepSeek 推理系统优化 — RL 用于资源调度
- adaptive_optics_gym — RL + 自适应光学仿真

算法原理:
- Markov Decision Process (MDP) — 马尔可夫决策过程
- Proximal Policy Optimization (PPO) — 近端策略优化
- Reward Shaping — 奖励塑形加速学习
- Curriculum Learning — 课程学习逐步增加难度

功能:
- 将光斑对准问题建模为强化学习环境
- 支持多种 RL 算法 (PPO/SAC/DQN)
- 提供奖励函数设计和状态空间定义
- 支持模拟训练和真实环境迁移

依赖: numpy, logging
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np

LOGGER = logging.getLogger("SpotZoom.RLEnvironment")


class ActionType(Enum):
    """动作类型。"""
    DISCRETE = "discrete"       # 离散动作空间
    CONTINUOUS = "continuous"   # 连续动作空间


@dataclass
class EnvironmentConfig:
    """环境配置。"""
    # 状态空间
    state_dim: int = 8          # 状态向量维度
    max_error_px: float = 100.0  # 最大误差 (像素)

    # 动作空间
    action_type: ActionType = ActionType.CONTINUOUS
    action_dim: int = 2          # 动作维度 (X, Y)
    max_step_px: float = 20.0    # 单步最大移动 (像素)

    # 奖励函数
    reward_alignment: float = 100.0      # 对准成功奖励
    reward_per_px_reduction: float = 1.0  # 每像素误差减少奖励
    penalty_per_step: float = -0.1        # 每步惩罚 (鼓励快速收敛)
    penalty_overshoot: float = -5.0       # 超调惩罚

    # 终止条件
    max_steps: int = 100         # 最大步数
    success_threshold_px: float = 3.0  # 成功阈值 (像素)

    # 模拟参数
    noise_level: float = 0.5     # 噪声水平
    latency_steps: int = 1       # 延迟步数


@dataclass
class EnvironmentState:
    """环境状态。"""
    position_x: float            # 当前 X 位置
    position_y: float            # 当前 Y 位置
    target_x: float              # 目标 X 位置
    target_y: float              # 目标 Y 位置
    error_x: float               # X 方向误差
    error_y: float               # Y 方向误差
    step_count: int              # 当前步数
    total_reward: float          # 累计奖励
    is_done: bool                # 是否结束
    is_success: bool             # 是否成功


@dataclass
class StepResult:
    """单步执行结果。"""
    state: np.ndarray
    reward: float
    is_done: bool
    info: Dict[str, Any]


class RLAlignmentEnvironment:
    """强化学习光斑对准环境。

    将光斑对准问题建模为强化学习环境，支持多种 RL 算法训练。

    Parameters
    ----------
    config : EnvironmentConfig or None
        环境配置。
    seed : int or None
        随机种子。
    """

    def __init__(
        self,
        config: Optional[EnvironmentConfig] = None,
        seed: Optional[int] = None,
    ):
        self.config = config or EnvironmentConfig()
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

        # 环境状态
        self._state = EnvironmentState(
            position_x=0.0,
            position_y=0.0,
            target_x=0.0,
            target_y=0.0,
            error_x=0.0,
            error_y=0.0,
            step_count=0,
            total_reward=0.0,
            is_done=False,
            is_success=False,
        )

        # 历史记录
        self._trajectory: List[Tuple[float, float]] = []
        self._reward_history: List[float] = []

        # 动作缓冲 (模拟延迟)
        self._action_buffer: List[Tuple[float, float]] = []

    @property
    def observation_space(self) -> Tuple[int, ...]:
        """观测空间形状。"""
        return (self.config.state_dim,)

    @property
    def action_space(self) -> Tuple[int, ...]:
        """动作空间形状。"""
        return (self.config.action_dim,)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """重置环境。

        Parameters
        ----------
        seed : int or None
            随机种子。
        options : dict or None
            可选参数。

        Returns
        -------
        Tuple[np.ndarray, Dict]
            (初始状态, 信息字典)
        """
        if seed is not None:
            np.random.seed(seed)

        # 随机初始化位置和目标
        max_err = self.config.max_error_px
        self._state.position_x = np.random.uniform(-max_err / 2, max_err / 2)
        self._state.position_y = np.random.uniform(-max_err / 2, max_err / 2)
        self._state.target_x = 0.0  # 目标在原点
        self._state.target_y = 0.0

        self._state.error_x = self._state.target_x - self._state.position_x
        self._state.error_y = self._state.target_y - self._state.position_y
        self._state.step_count = 0
        self._state.total_reward = 0.0
        self._state.is_done = False
        self._state.is_success = False

        self._trajectory = [(self._state.position_x, self._state.position_y)]
        self._reward_history = []
        self._action_buffer = []

        return self._get_observation(), {}

    def step(self, action: Union[np.ndarray, Tuple[float, float]]) -> StepResult:
        """执行一步动作。

        Parameters
        ----------
        action : np.ndarray or Tuple[float, float]
            动作向量 (dx, dy)。

        Returns
        -------
        StepResult
            步骤结果。
        """
        if self._state.is_done:
            raise RuntimeError("Environment is done. Call reset() first.")

        # 解析动作
        if isinstance(action, np.ndarray):
            action = (float(action[0]), float(action[1]))

        dx, dy = action

        # 限制动作范围
        max_step = self.config.max_step_px
        dx = np.clip(dx, -max_step, max_step)
        dy = np.clip(dy, -max_step, max_step)

        # 添加到动作缓冲 (模拟延迟)
        self._action_buffer.append((dx, dy))
        if len(self._action_buffer) > self.config.latency_steps:
            executed_dx, executed_dy = self._action_buffer.pop(0)
        else:
            executed_dx, executed_dy = 0.0, 0.0

        # 执行移动 (含噪声)
        noise_x = np.random.normal(0, self.config.noise_level)
        noise_y = np.random.normal(0, self.config.noise_level)

        self._state.position_x += executed_dx + noise_x
        self._state.position_y += executed_dy + noise_y

        # 更新误差
        prev_error = np.sqrt(self._state.error_x ** 2 + self._state.error_y ** 2)
        self._state.error_x = self._state.target_x - self._state.position_x
        self._state.error_y = self._state.target_y - self._state.position_y
        curr_error = np.sqrt(self._state.error_x ** 2 + self._state.error_y ** 2)

        # 计算奖励
        reward = self._compute_reward(prev_error, curr_error, dx, dy)

        # 更新状态
        self._state.step_count += 1
        self._state.total_reward += reward
        self._trajectory.append((self._state.position_x, self._state.position_y))
        self._reward_history.append(reward)

        # 检查终止条件
        is_success = curr_error < self.config.success_threshold_px
        is_done = (
            is_success or
            self._state.step_count >= self.config.max_steps
        )

        self._state.is_done = is_done
        self._state.is_success = is_success

        info = {
            "error": curr_error,
            "error_x": self._state.error_x,
            "error_y": self._state.error_y,
            "step_count": self._state.step_count,
            "is_success": is_success,
        }

        return StepResult(
            state=self._get_observation(),
            reward=reward,
            is_done=is_done,
            info=info,
        )

    def _compute_reward(
        self,
        prev_error: float,
        curr_error: float,
        dx: float,
        dy: float,
    ) -> float:
        """计算奖励。

        Parameters
        ----------
        prev_error : float
            移动前误差。
        curr_error : float
            移动后误差。
        dx, dy : float
            执行的动作。

        Returns
        -------
        float
            奖励值。
        """
        reward = 0.0

        # 误差减少奖励
        error_reduction = prev_error - curr_error
        reward += error_reduction * self.config.reward_per_px_reduction

        # 对准成功奖励
        if curr_error < self.config.success_threshold_px:
            reward += self.config.reward_alignment

        # 步数惩罚
        reward += self.config.penalty_per_step

        # 超调惩罚
        if error_reduction < 0:
            reward += self.config.penalty_overshoot * abs(error_reduction)

        return reward

    def _get_observation(self) -> np.ndarray:
        """获取观测向量。

        Returns
        -------
        np.ndarray
            观测向量。
        """
        # 状态向量: [error_x, error_y, error_norm, position_x, position_y, velocity_x, velocity_y, step_norm]
        error_norm = np.sqrt(self._state.error_x ** 2 + self._state.error_y ** 2)

        # 速度估计 (基于轨迹)
        if len(self._trajectory) >= 2:
            prev_x, prev_y = self._trajectory[-2]
            vx = self._state.position_x - prev_x
            vy = self._state.position_y - prev_y
        else:
            vx, vy = 0.0, 0.0

        step_norm = self._state.step_count / self.config.max_steps

        obs = np.array([
            self._state.error_x / self.config.max_error_px,
            self._state.error_y / self.config.max_error_px,
            error_norm / self.config.max_error_px,
            self._state.position_x / self.config.max_error_px,
            self._state.position_y / self.config.max_error_px,
            vx / self.config.max_step_px,
            vy / self.config.max_step_px,
            step_norm,
        ], dtype=np.float32)

        return obs

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """渲染环境。

        Parameters
        ----------
        mode : str
            渲染模式。

        Returns
        -------
        np.ndarray or None
            图像数组 (如果 mode='rgb_array')。
        """
        # 简单文本渲染
        error = np.sqrt(self._state.error_x ** 2 + self._state.error_y ** 2)
        status = "SUCCESS" if self._state.is_success else "RUNNING" if not self._state.is_done else "FAILED"

        LOGGER.info(
            "Step %d: pos=(%.1f, %.1f), error=%.1f px, reward=%.2f, status=%s",
            self._state.step_count,
            self._state.position_x,
            self._state.position_y,
            error,
            self._state.total_reward,
            status,
        )

        return None

    def close(self) -> None:
        """关闭环境。"""
        pass

    def get_trajectory(self) -> List[Tuple[float, float]]:
        """获取轨迹历史。"""
        return list(self._trajectory)

    def get_reward_history(self) -> List[float]:
        """获取奖励历史。"""
        return list(self._reward_history)

    def seed(self, seed: int) -> None:
        """设置随机种子。"""
        self.seed = seed
        np.random.seed(seed)

    @property
    def state(self) -> EnvironmentState:
        """当前环境状态。"""
        return self._state


class RLPolicyWrapper:
    """RL 策略包装器。

    将训练好的 RL 策略包装为可直接用于实际控制的接口。

    Parameters
    ----------
    model : Any
        训练好的 RL 模型 (如 Stable-Baselines3 模型)。
    config : EnvironmentConfig or None
        环境配置。
    """

    def __init__(
        self,
        model: Any,
        config: Optional[EnvironmentConfig] = None,
    ):
        self.model = model
        self.config = config or EnvironmentConfig()

        # 状态跟踪
        self._last_error: Optional[Tuple[float, float]] = None
        self._action_history: List[Tuple[float, float]] = []

    def predict_action(self, error_x: float, error_y: float) -> Tuple[float, float]:
        """根据误差预测动作。

        Parameters
        ----------
        error_x : float
            X 方向误差 (像素)。
        error_y : float
            Y 方向误差 (像素)。

        Returns
        -------
        Tuple[float, float]
            预测的动作 (dx, dy)。
        """
        # 构建观测向量
        error_norm = np.sqrt(error_x ** 2 + error_y ** 2)

        # 速度估计
        if self._last_error is not None:
            vx = error_x - self._last_error[0]
            vy = error_y - self._last_error[1]
        else:
            vx, vy = 0.0, 0.0

        obs = np.array([
            error_x / self.config.max_error_px,
            error_y / self.config.max_error_px,
            error_norm / self.config.max_error_px,
            0.0,  # position_x (未知)
            0.0,  # position_y (未知)
            vx / self.config.max_step_px,
            vy / self.config.max_step_px,
            0.0,  # step_norm (未知)
        ], dtype=np.float32).reshape(1, -1)

        # 使用模型预测
        try:
            action, _ = self.model.predict(obs, deterministic=True)
            dx, dy = float(action[0]), float(action[1])
        except Exception as e:
            LOGGER.warning("RL prediction failed: %s, using fallback", e)
            # 回退到简单比例控制
            dx = error_x * 0.5
            dy = error_y * 0.5

        # 限制动作范围
        max_step = self.config.max_step_px
        dx = np.clip(dx, -max_step, max_step)
        dy = np.clip(dy, -max_step, max_step)

        self._last_error = (error_x, error_y)
        self._action_history.append((dx, dy))

        return (dx, dy)

    def reset(self) -> None:
        """重置策略状态。"""
        self._last_error = None
        self._action_history.clear()
