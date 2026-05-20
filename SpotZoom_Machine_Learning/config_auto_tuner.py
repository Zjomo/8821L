"""
参数自动调优器 (ConfigAutoTuner)

灵感来源:
- Optuna — 超参数自动优化框架 (基于 TPE 贝叶斯优化)
- Bayesian Optimization — 贝叶斯优化 (Snoek et al., 2012)
- Grid Search + Hill Climbing — 网格搜索 + 山地攀登法
- Nelder-Mead Simplex — 单纯形法 (无导数优化)
- Ziegler-Nichols Method — PID 参数整定经典方法

算法原理:
- Grid Search — 网格搜索 (在参数空间均匀采样)
- Hill Climbing — 山地攀登法 (局部搜索，沿梯度方向改进)
- Convergence Score — 收敛评分 (综合对准精度与速度)
- Parameter Space Exploration — 参数空间探索与利用平衡

功能:
- 自动调优 PID 增益和对准参数
- 基于网格搜索 + 山地攀登法的混合优化策略
- 根据收敛历史评估参数质量
- 输出最优参数和调优报告

依赖: numpy, logging
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger("SpotZoom.ConfigAutoTuner")


@dataclass
class ParameterSpace:
    """参数搜索空间定义。"""
    pid_kp: Tuple[float, float] = (0.01, 2.0)       # PID 比例增益范围
    pid_ki: Tuple[float, float] = (0.0, 0.5)         # PID 积分增益范围
    pid_kd: Tuple[float, float] = (0.0, 1.0)         # PID 微分增益范围
    tolerance_px: Tuple[float, float] = (0.1, 5.0)   # 对准容差范围 (像素)
    settle_time: Tuple[float, float] = (0.5, 10.0)   # 稳定时间范围 (秒)


@dataclass
class TuningResult:
    """调优结果。"""
    parameters: Dict[str, float]    # 参数字典
    score: float                     # 收敛质量评分
    iteration: int                   # 迭代次数
    is_improved: bool                # 是否相比之前有改进


class ConfigAutoTuner:
    """参数自动调优器。

    使用网格搜索 + 山地攀登法混合策略自动搜索最优 PID 增益和对准参数。

    Parameters
    ----------
    parameter_space : ParameterSpace or None
        参数搜索空间定义。默认使用标准范围。
    grid_resolution : int
        网格搜索每维度的采样点数。
    hill_climb_steps : int
        山地攀登法的最大步数。
    hill_climb_step_size : float
        山地攀登法的步长 (相对于参数范围的比率)。
    improvement_threshold : float
        最小改进阈值 (评分提升低于此值停止搜索)。
    max_iterations : int
        最大总迭代次数。
    random_seed : int or None
        随机种子 (可复现)。
    """

    def __init__(
        self,
        parameter_space: Optional[ParameterSpace] = None,
        grid_resolution: int = 5,
        hill_climb_steps: int = 20,
        hill_climb_step_size: float = 0.1,
        improvement_threshold: float = 0.5,
        max_iterations: int = 100,
        random_seed: Optional[int] = None,
    ):
        self.parameter_space = parameter_space or ParameterSpace()
        self.grid_resolution = int(grid_resolution)
        self.hill_climb_steps = int(hill_climb_steps)
        self.hill_climb_step_size = float(hill_climb_step_size)
        self.improvement_threshold = float(improvement_threshold)
        self.max_iterations = int(max_iterations)

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        # 参数名称和对应范围
        self._param_names = ["pid_kp", "pid_ki", "pid_kd", "tolerance_px", "settle_time"]
        self._ranges = {
            "pid_kp": self.parameter_space.pid_kp,
            "pid_ki": self.parameter_space.pid_ki,
            "pid_kd": self.parameter_space.pid_kd,
            "tolerance_px": self.parameter_space.tolerance_px,
            "settle_time": self.parameter_space.settle_time,
        }

        # 搜索状态
        self._current_params: Optional[Dict[str, float]] = None
        self._best_params: Optional[Dict[str, float]] = None
        self._best_score: float = -np.inf
        self._iteration: int = 0
        self._phase: str = "grid"  # "grid" | "hill_climb" | "done"
        self._grid_points: List[Dict[str, float]] = []
        self._grid_index: int = 0
        self._score_history: List[Tuple[Dict[str, float], float]] = []

        # 初始化网格搜索
        self._init_grid_search()

    def _init_grid_search(self) -> None:
        """初始化网格搜索点。"""
        self._grid_points = self._generate_grid_points()
        self._grid_index = 0
        self._phase = "grid"
        LOGGER.info("ConfigAutoTuner: 网格搜索初始化, %d 个采样点", len(self._grid_points))

    def _generate_grid_points(self) -> List[Dict[str, float]]:
        """生成网格搜索采样点。

        Returns
        -------
        List[Dict[str, float]]
            参数字典列表。
        """
        # 为每个参数生成采样值
        param_values = {}
        for name in self._param_names:
            low, high = self._ranges[name]
            values = np.linspace(low, high, self.grid_resolution).tolist()
            param_values[name] = values

        # 生成所有组合 (笛卡尔积)
        points = []
        # 使用迭代方式生成组合，避免过多参数时内存问题
        self._generate_combinations(param_values, 0, {}, points)

        # 如果组合数过多，随机采样
        max_points = self.max_iterations // 2
        if len(points) > max_points:
            random.shuffle(points)
            points = points[:max_points]
            LOGGER.info("ConfigAutoTuner: 网格点过多，随机采样 %d 个", max_points)

        return points

    def _generate_combinations(
        self,
        param_values: Dict[str, List[float]],
        param_idx: int,
        current: Dict[str, float],
        result: List[Dict[str, float]],
    ) -> None:
        """递归生成参数组合。"""
        if param_idx >= len(self._param_names):
            result.append(dict(current))
            return

        name = self._param_names[param_idx]
        for value in param_values[name]:
            current[name] = value
            self._generate_combinations(param_values, param_idx + 1, current, result)

    def suggest_parameters(self) -> Dict[str, float]:
        """建议下一组待评估的参数。

        Returns
        -------
        Dict[str, float]
            参数字典。
        """
        if self._iteration >= self.max_iterations:
            LOGGER.info("ConfigAutoTuner: 已达最大迭代次数")
            return self._best_params or self._get_default_params()

        if self._phase == "grid":
            # 网格搜索阶段
            if self._grid_index < len(self._grid_points):
                params = self._grid_points[self._grid_index]
                self._current_params = params
                self._grid_index += 1
                LOGGER.debug("ConfigAutoTuner: 网格搜索 %d/%d", self._grid_index, len(self._grid_points))
                return params
            else:
                # 网格搜索完成，切换到山地攀登
                self._phase = "hill_climb"
                LOGGER.info("ConfigAutoTuner: 网格搜索完成, 最佳评分=%.2f, 切换到山地攀登",
                            self._best_score)

        if self._phase == "hill_climb":
            # 山地攀登阶段
            if self._best_params is None:
                return self._get_default_params()

            params = self._hill_climb_step()
            self._current_params = params
            return params

        # done 阶段: 返回最佳参数
        return self._best_params or self._get_default_params()

    def _hill_climb_step(self) -> Dict[str, float]:
        """执行一步山地攀登搜索。

        Returns
        -------
        Dict[str, float]
            新的参数组合。
        """
        if self._best_params is None:
            return self._get_default_params()

        # 随机选择一个参数进行扰动
        param_name = random.choice(self._param_names)
        low, high = self._ranges[param_name]
        step = (high - low) * self.hill_climb_step_size

        # 生成扰动后的值
        current_val = self._best_params[param_name]
        perturbation = random.uniform(-step, step)
        new_val = current_val + perturbation

        # 限制在范围内
        new_val = max(low, min(high, new_val))

        # 复制最佳参数并更新被扰动的参数
        new_params = dict(self._best_params)
        new_params[param_name] = new_val

        LOGGER.debug("ConfigAutoTuner: 山地攀登, 扰动 %s: %.4f -> %.4f",
                     param_name, current_val, new_val)

        return new_params

    def report_convergence(self, quality_score: float) -> None:
        """报告当前参数的收敛质量评分。

        Parameters
        ----------
        quality_score : float
            收敛质量评分。越高越好。
            建议评分标准:
            - 对准精度 (权重 0.5): 100 * exp(-error_px / tolerance_px)
            - 收敛速度 (权重 0.3): 100 * exp(-settle_time / target_time)
            - 稳定性 (权重 0.2): 100 * (1 - overshoot_ratio)
        """
        self._iteration += 1
        score = float(quality_score)

        # 记录历史
        if self._current_params is not None:
            self._score_history.append((dict(self._current_params), score))

        is_improved = False

        if score > self._best_score:
            improvement = score - self._best_score
            self._best_score = score
            if self._current_params is not None:
                self._best_params = dict(self._current_params)
            is_improved = True
            LOGGER.info(
                "ConfigAutoTuner: 迭代 %d, 新最佳评分=%.2f (改进 +%.2f), 参数=%s",
                self._iteration, score, improvement,
                {k: round(v, 4) for k, v in (self._best_params or {}).items()},
            )
        else:
            LOGGER.debug("ConfigAutoTuner: 迭代 %d, 评分=%.2f (无改进, 最佳=%.2f)",
                         self._iteration, score, self._best_score)

        # 检查是否应该结束山地攀登
        if self._phase == "hill_climb":
            recent_scores = [s for _, s in self._score_history[-10:]]
            if len(recent_scores) >= 10:
                recent_improvement = max(recent_scores) - min(recent_scores)
                if recent_improvement < self.improvement_threshold:
                    self._phase = "done"
                    LOGGER.info(
                        "ConfigAutoTuner: 山地攀登收敛, 近10次改进=%.2f < 阈值=%.2f",
                        recent_improvement, self.improvement_threshold,
                    )

        # 检查最大迭代次数
        if self._iteration >= self.max_iterations:
            self._phase = "done"
            LOGGER.info("ConfigAutoTuner: 达到最大迭代次数 %d", self.max_iterations)

    def get_best_parameters(self) -> TuningResult:
        """获取当前最佳参数。

        Returns
        -------
        TuningResult
            最佳参数调优结果。
        """
        params = self._best_params or self._get_default_params()
        return TuningResult(
            parameters=params,
            score=self._best_score,
            iteration=self._iteration,
            is_improved=self._iteration > 0,
        )

    def _get_default_params(self) -> Dict[str, float]:
        """获取默认参数 (搜索空间中点)。

        Returns
        -------
        Dict[str, float]
            默认参数字典。
        """
        return {
            name: (low + high) / 2.0
            for name, (low, high) in self._ranges.items()
        }

    def get_score_history(self) -> List[Tuple[Dict[str, float], float]]:
        """获取完整的评分历史。

        Returns
        -------
        List[Tuple[Dict[str, float], float]]
            [(parameters, score), ...] 评分历史列表。
        """
        return list(self._score_history)

    def get_top_parameters(self, n: int = 5) -> List[TuningResult]:
        """获取评分最高的 N 组参数。

        Parameters
        ----------
        n : int
            返回的参数组数。

        Returns
        -------
        List[TuningResult]
            按评分降序排列的调优结果列表。
        """
        sorted_history = sorted(self._score_history, key=lambda x: x[1], reverse=True)
        top_results = []
        for i, (params, score) in enumerate(sorted_history[:n]):
            top_results.append(TuningResult(
                parameters=params,
                score=score,
                iteration=i + 1,
                is_improved=True,
            ))
        return top_results

    @property
    def iteration(self) -> int:
        """当前迭代次数。"""
        return self._iteration

    @property
    def phase(self) -> str:
        """当前搜索阶段 ("grid" / "hill_climb" / "done")。"""
        return self._phase

    @property
    def best_score(self) -> float:
        """当前最佳评分。"""
        return self._best_score

    def reset(self) -> None:
        """重置调优器，重新开始搜索。"""
        self._current_params = None
        self._best_params = None
        self._best_score = -np.inf
        self._iteration = 0
        self._score_history.clear()
        self._init_grid_search()
        LOGGER.info("ConfigAutoTuner: 调优器已重置")
