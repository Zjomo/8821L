"""
多自由度自动对准寻优器 (AutoAlignmentOptimizer)

灵感来源:
- Rayoptics (https://github.com/michaeljk15/rayoptics) — 光学自动优化框架，
  使用阻尼最小二乘法 (DLS) 进行多参数同时寻优
- scipy.optimize — Nelder-Mead 单纯形法、Powell 方向集法等经典优化算法
- 光学对准工程实践 — 多自由度 (X/Y/Z 偏移 + 焦距 + 倾斜) 联合寻优

算法原理:
- Nelder-Mead Simplex — 无梯度直接搜索法，适用于非光滑目标函数
- Powell Direction Set — 共轭方向法，沿各坐标轴方向轮流搜索
- Coordinate Descent — 坐标下降法，每次沿一个维度优化
- Damped Least Squares (Marquardt) — 阻尼最小二乘法，平衡梯度与步长

功能:
- 多自由度对准参数同时寻优
- 支持多种优化算法 (Nelder-Mead, Powell, 坐标下降)
- 收敛历史记录与分析
- 参数范围约束与步长控制
- 优化结果评估与导出

依赖: numpy, logging (可选: scipy.optimize)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger("SpotZoom.AutoAlignmentOptimizer")


# ======================== 数据类 ========================


@dataclass
class AlignmentParameter:
    """对准参数定义。

    Parameters
    ----------
    name : str
        参数名称 (如 'x_offset', 'focal_length')。
    current_value : float
        当前值。
    min_value : float
        允许的最小值。
    max_value : float
        允许的最大值。
    step_size : float
        初始步长。
    """
    name: str
    current_value: float
    min_value: float = -1e6
    max_value: float = 1e6
    step_size: float = 1.0


@dataclass
class OptimizationConfig:
    """优化配置。

    Parameters
    ----------
    method : str
        优化方法: 'nelder_mead', 'powell', 'coordinate_descent'。
    max_iterations : int
        最大迭代次数。
    tolerance : float
        收敛容差 (目标函数值变化小于此值时停止)。
    initial_step_factor : float
        初始步长缩放因子。
    adaptive_step : bool
        是否启用自适应步长。
    max_no_improvement : int
        连续无改善的最大迭代次数。
    """
    method: str = "nelder_mead"
    max_iterations: int = 200
    tolerance: float = 1e-6
    initial_step_factor: float = 0.1
    adaptive_step: bool = True
    max_no_improvement: int = 30


@dataclass
class ConvergenceRecord:
    """单次迭代的收敛记录。"""
    iteration: int              # 迭代编号
    objective_value: float      # 目标函数值
    parameter_values: List[float]  # 参数值快照
    step_size: float            # 当前步长
    improvement: float          # 相对上次的改善量
    elapsed_time_s: float       # 累计耗时 (秒)


@dataclass
class OptimizationResult:
    """优化结果。

    Parameters
    ----------
    best_parameters : Dict[str, float]
        最优参数值。
    best_objective : float
        最优目标函数值。
    converged : bool
        是否收敛。
    total_iterations : int
        总迭代次数。
    elapsed_time_s : float
        总耗时 (秒)。
    convergence_history : List[ConvergenceRecord]
        收敛历史。
    method : str
        使用的优化方法。
    """
    best_parameters: Dict[str, float]
    best_objective: float
    converged: bool
    total_iterations: int
    elapsed_time_s: float
    convergence_history: List[ConvergenceRecord]
    method: str


# ======================== 优化引擎 ========================


class AutoAlignmentOptimizer:
    """多自由度自动对准寻优器。

    支持多种优化算法对多个对准参数进行同时寻优，
    找到使目标函数 (如光斑质量评分) 最大化的参数组合。

    Parameters
    ----------
    config : OptimizationConfig or None
        优化配置。为 None 时使用默认配置。
    """

    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()

        self._parameters: List[AlignmentParameter] = []
        self._objective_fn: Optional[Callable[[np.ndarray], float]] = None
        self._convergence_history: List[ConvergenceRecord] = []
        self._best_value: Optional[np.ndarray] = None
        self._best_objective: float = -np.inf
        self._iteration_count: int = 0
        self._start_time: float = 0.0
        self._no_improvement_count: int = 0

        LOGGER.info(
            "AutoAlignmentOptimizer: 初始化完成 (method=%s, max_iter=%d, tol=%.2e)",
            self.config.method, self.config.max_iterations, self.config.tolerance,
        )

    # ======================== 公共接口 ========================

    def add_parameter(self, param: AlignmentParameter) -> None:
        """添加优化参数。

        Parameters
        ----------
        param : AlignmentParameter
            对准参数定义。
        """
        self._parameters.append(param)
        LOGGER.info(
            "AutoAlignmentOptimizer: 添加参数 '%s', 范围=[%.4f, %.4f], 步长=%.4f",
            param.name, param.min_value, param.max_value, param.step_size,
        )

    def set_objective_function(
        self,
        func: Callable[[np.ndarray], float],
    ) -> None:
        """设置目标函数。

        目标函数接收参数向量 (numpy array)，返回标量质量值。
        优化器将最大化此值。

        Parameters
        ----------
        func : Callable[[np.ndarray], float]
            目标函数。
        """
        self._objective_fn = func
        LOGGER.info("AutoAlignmentOptimizer: 目标函数已设置")

    def optimize(
        self,
        method: Optional[str] = None,
    ) -> OptimizationResult:
        """运行优化。

        Parameters
        ----------
        method : str or None
            优化方法。为 None 时使用配置中的方法。
            支持: 'nelder_mead', 'powell', 'coordinate_descent'。

        Returns
        -------
        OptimizationResult
            优化结果。
        """
        if self._objective_fn is None:
            raise ValueError("请先调用 set_objective_function() 设置目标函数")
        if not self._parameters:
            raise ValueError("请先调用 add_parameter() 添加至少一个参数")

        method = method or self.config.method
        self._convergence_history.clear()
        self._iteration_count = 0
        self._no_improvement_count = 0
        self._start_time = time.perf_counter()

        # 初始参数向量
        x0 = np.array([p.current_value for p in self._parameters], dtype=np.float64)
        self._best_value = x0.copy()
        self._best_objective = self._evaluate(x0)

        LOGGER.info(
            "AutoAlignmentOptimizer: 开始优化 (method=%s, dim=%d, initial_obj=%.6f)",
            method, len(x0), self._best_objective,
        )

        # 选择优化方法
        if method == "nelder_mead":
            self._run_nelder_mead()
        elif method == "powell":
            self._run_powell()
        elif method == "coordinate_descent":
            self._run_coordinate_descent()
        else:
            raise ValueError(f"不支持的优化方法: {method}")

        elapsed = time.perf_counter() - self._start_time
        converged = self._no_improvement_count < self.config.max_no_improvement

        # 构建结果
        best_params = {}
        for i, p in enumerate(self._parameters):
            best_params[p.name] = float(self._best_value[i])

        result = OptimizationResult(
            best_parameters=best_params,
            best_objective=float(self._best_objective),
            converged=converged,
            total_iterations=self._iteration_count,
            elapsed_time_s=round(elapsed, 4),
            convergence_history=list(self._convergence_history),
            method=method,
        )

        LOGGER.info(
            "AutoAlignmentOptimizer: 优化完成. best_obj=%.6f, iterations=%d, "
            "converged=%s, elapsed=%.2fs",
            result.best_objective, result.total_iterations,
            result.converged, result.elapsed_time_s,
        )

        return result

    def step_coordinate_descent(self) -> Tuple[np.ndarray, float]:
        """单步坐标下降。

        Returns
        -------
        Tuple[np.ndarray, float]
            (当前参数向量, 当前目标函数值)。
        """
        if self._objective_fn is None or not self._parameters:
            raise ValueError("优化器未正确初始化")

        if self._best_value is None:
            x0 = np.array([p.current_value for p in self._parameters], dtype=np.float64)
            self._best_value = x0.copy()
            self._best_objective = self._evaluate(x0)

        x = self._best_value.copy()
        n = len(x)

        for i in range(n):
            param = self._parameters[i]
            step = param.step_size

            # 正方向探测
            x_plus = x.copy()
            x_plus[i] = min(x[i] + step, param.max_value)
            obj_plus = self._evaluate(x_plus)

            # 负方向探测
            x_minus = x.copy()
            x_minus[i] = max(x[i] - step, param.min_value)
            obj_minus = self._evaluate(x_minus)

            # 选择最佳方向
            if obj_plus > self._best_objective:
                x[i] = x_plus[i]
                self._best_objective = obj_plus
            elif obj_minus > self._best_objective:
                x[i] = x_minus[i]
                self._best_objective = obj_minus

        self._best_value = x.copy()
        self._record_iteration(x, self._best_objective, step)

        return (self._best_value.copy(), self._best_objective)

    def step_nelder_mead(self) -> Tuple[np.ndarray, float]:
        """Nelder-Mead 单步迭代。

        执行一次完整的 Nelder-Mead 单纯形迭代 (反射/扩展/收缩)。

        Returns
        -------
        Tuple[np.ndarray, float]
            (当前最优参数向量, 当前最优目标函数值)。
        """
        if self._objective_fn is None or not self._parameters:
            raise ValueError("优化器未正确初始化")

        if self._best_value is None:
            x0 = np.array([p.current_value for p in self._parameters], dtype=np.float64)
            self._best_value = x0.copy()
            self._best_objective = self._evaluate(x0)

        if not hasattr(self, '_simplex') or self._simplex is None:
            self._init_simplex()

        self._nelder_mead_iteration()
        self._record_iteration(
            self._simplex_vertices[0],
            self._simplex_values[0],
            self._current_nm_step,
        )

        self._best_value = self._simplex_vertices[0].copy()
        self._best_objective = float(self._simplex_values[0])

        return (self._best_value.copy(), self._best_objective)

    def get_current_solution(self) -> Tuple[Dict[str, float], float]:
        """获取当前最优解。

        Returns
        -------
        Tuple[Dict[str, float], float]
            (参数字典, 目标函数值)。
        """
        params = {}
        if self._best_value is not None:
            for i, p in enumerate(self._parameters):
                params[p.name] = float(self._best_value[i])
        return (params, float(self._best_objective))

    def get_convergence_history(self) -> List[ConvergenceRecord]:
        """获取收敛历史。

        Returns
        -------
        List[ConvergenceRecord]
            收敛记录列表。
        """
        return list(self._convergence_history)

    def reset(self) -> None:
        """重置优化器。"""
        self._convergence_history.clear()
        self._best_value = None
        self._best_objective = -np.inf
        self._iteration_count = 0
        self._no_improvement_count = 0
        self._simplex = None
        self._simplex_vertices = None
        self._simplex_values = None
        LOGGER.info("AutoAlignmentOptimizer: 优化器已重置")

    # ======================== 内部方法: 评估 ========================

    def _evaluate(self, x: np.ndarray) -> float:
        """评估目标函数并约束参数范围。"""
        x_clamped = self._clamp(x)
        try:
            value = self._objective_fn(x_clamped)
        except Exception as e:
            LOGGER.warning("AutoAlignmentOptimizer: 目标函数评估失败: %s", e)
            value = -np.inf
        return float(value)

    def _clamp(self, x: np.ndarray) -> np.ndarray:
        """将参数约束到允许范围内。"""
        x_clamped = x.copy()
        for i, p in enumerate(self._parameters):
            x_clamped[i] = np.clip(x_clamped[i], p.min_value, p.max_value)
        return x_clamped

    # ======================== 内部方法: 收敛记录 ========================

    def _record_iteration(
        self,
        x: np.ndarray,
        obj: float,
        step: float,
    ) -> None:
        """记录一次迭代。"""
        improvement = 0.0
        if self._convergence_history:
            improvement = obj - self._convergence_history[-1].objective_value

        elapsed = time.perf_counter() - self._start_time
        self._iteration_count += 1

        if improvement < self.config.tolerance:
            self._no_improvement_count += 1
        else:
            self._no_improvement_count = 0

        record = ConvergenceRecord(
            iteration=self._iteration_count,
            objective_value=round(obj, 8),
            parameter_values=[round(float(v), 8) for v in x],
            step_size=round(step, 8),
            improvement=round(improvement, 8),
            elapsed_time_s=round(elapsed, 4),
        )
        self._convergence_history.append(record)

    # ======================== 内部方法: 坐标下降 ========================

    def _run_coordinate_descent(self) -> None:
        """运行完整的坐标下降优化。"""
        x = self._best_value.copy()

        for iteration in range(self.config.max_iterations):
            if self._no_improvement_count >= self.config.max_no_improvement:
                LOGGER.info(
                    "AutoAlignmentOptimizer: 坐标下降收敛 (无改善 %d 次)",
                    self._no_improvement_count,
                )
                break

            prev_obj = self._best_objective
            step = self._get_adaptive_step(iteration)

            for i in range(len(x)):
                param = self._parameters[i]

                # 正方向
                x_plus = x.copy()
                x_plus[i] = min(x[i] + step, param.max_value)
                obj_plus = self._evaluate(x_plus)

                # 负方向
                x_minus = x.copy()
                x_minus[i] = max(x[i] - step, param.min_value)
                obj_minus = self._evaluate(x_minus)

                # 选择最佳
                if obj_plus >= obj_minus and obj_plus > self._best_objective:
                    x[i] = x_plus[i]
                    self._best_objective = obj_plus
                elif obj_minus > self._best_objective:
                    x[i] = x_minus[i]
                    self._best_objective = obj_minus

            self._best_value = x.copy()
            self._record_iteration(x, self._best_objective, step)

            if iteration % 20 == 0:
                LOGGER.debug(
                    "AutoAlignmentOptimizer: 坐标下降 iter=%d, obj=%.6f, step=%.6f",
                    iteration, self._best_objective, step,
                )

    # ======================== 内部方法: Nelder-Mead ========================

    def _init_simplex(self) -> None:
        """初始化 Nelder-Mead 单纯形。"""
        n = len(self._parameters)
        x0 = self._best_value.copy()

        # 构建初始单纯形: n+1 个顶点
        vertices = [x0.copy()]
        for i in range(n):
            xi = x0.copy()
            step = self._parameters[i].step_size * self.config.initial_step_factor
            xi[i] = np.clip(
                xi[i] + step,
                self._parameters[i].min_value,
                self._parameters[i].max_value,
            )
            vertices.append(xi)

        # 评估所有顶点
        values = [self._evaluate(v) for v in vertices]

        # 按目标函数值排序 (降序，因为我们最大化)
        sorted_indices = np.argsort(values)[::-1]
        self._simplex_vertices = [vertices[j] for j in sorted_indices]
        self._simplex_values = [values[j] for j in sorted_indices]
        self._current_nm_step = self._parameters[0].step_size * self.config.initial_step_factor
        self._simplex = True

        LOGGER.debug(
            "AutoAlignmentOptimizer: 单纯形初始化完成, 顶点数=%d, "
            "最优值=%.6f, 最差值=%.6f",
            n + 1, self._simplex_values[0], self._simplex_values[-1],
        )

    def _nelder_mead_iteration(self) -> None:
        """执行一次 Nelder-Mead 迭代。"""
        n = len(self._parameters)
        vertices = self._simplex_vertices
        values = self._simplex_values

        # 1. 排序 (降序)
        order = np.argsort(values)[::-1]
        vertices = [vertices[j] for j in order]
        values = [values[j] for j in order]

        # best = vertices[0] (最大值), worst = vertices[-1]
        # 计算重心 (排除最差点)
        centroid = np.zeros(n, dtype=np.float64)
        for i in range(n):
            centroid += vertices[i]
        centroid /= n

        # 2. 反射
        alpha = 1.0
        reflected = centroid + alpha * (centroid - vertices[-1])
        reflected = self._clamp(reflected)
        f_reflected = self._evaluate(reflected)

        if values[0] >= f_reflected > values[-2]:
            # 反射点介于最优和次差之间 -> 接受反射
            vertices[-1] = reflected
            values[-1] = f_reflected
            self._current_nm_step = float(np.linalg.norm(reflected - centroid))

        elif f_reflected > values[0]:
            # 反射点优于最优点 -> 尝试扩展
            gamma = 2.0
            expanded = centroid + gamma * (reflected - centroid)
            expanded = self._clamp(expanded)
            f_expanded = self._evaluate(expanded)

            if f_expanded > f_reflected:
                vertices[-1] = expanded
                values[-1] = f_expanded
                self._current_nm_step = float(np.linalg.norm(expanded - centroid))
            else:
                vertices[-1] = reflected
                values[-1] = f_reflected
                self._current_nm_step = float(np.linalg.norm(reflected - centroid))
        else:
            # 反射点差于次差点 -> 收缩
            rho = 0.5
            if f_reflected > values[-1]:
                # 外收缩
                contracted = centroid + rho * (reflected - centroid)
            else:
                # 内收缩
                contracted = centroid + rho * (vertices[-1] - centroid)

            contracted = self._clamp(contracted)
            f_contracted = self._evaluate(contracted)

            if f_contracted > max(f_reflected, values[-1]):
                vertices[-1] = contracted
                values[-1] = f_contracted
                self._current_nm_step = float(np.linalg.norm(contracted - centroid))
            else:
                # 收缩失败 -> 缩小整个单纯形
                sigma = 0.5
                for i in range(1, n + 1):
                    vertices[i] = vertices[0] + sigma * (vertices[i] - vertices[0])
                    vertices[i] = self._clamp(vertices[i])
                    values[i] = self._evaluate(vertices[i])
                self._current_nm_step *= sigma

        self._simplex_vertices = vertices
        self._simplex_values = values

    def _run_nelder_mead(self) -> None:
        """运行完整的 Nelder-Mead 优化。"""
        self._init_simplex()

        for iteration in range(self.config.max_iterations):
            if self._no_improvement_count >= self.config.max_no_improvement:
                LOGGER.info(
                    "AutoAlignmentOptimizer: Nelder-Mead 收敛 (无改善 %d 次)",
                    self._no_improvement_count,
                )
                break

            self._nelder_mead_iteration()

            self._best_value = self._simplex_vertices[0].copy()
            self._best_objective = float(self._simplex_values[0])
            self._record_iteration(
                self._best_value, self._best_objective, self._current_nm_step,
            )

            # 检查单纯形大小
            if self._simplex_size() < self.config.tolerance:
                LOGGER.info(
                    "AutoAlignmentOptimizer: Nelder-Mead 收敛 (单纯形大小=%.2e)",
                    self._simplex_size(),
                )
                break

            if iteration % 20 == 0:
                LOGGER.debug(
                    "AutoAlignmentOptimizer: Nelder-Mead iter=%d, obj=%.6f, "
                    "simplex_size=%.6f",
                    iteration, self._best_objective, self._simplex_size(),
                )

    def _simplex_size(self) -> float:
        """计算单纯形大小 (顶点到重心的最大距离)。"""
        n = len(self._parameters)
        centroid = np.mean(self._simplex_vertices, axis=0)
        max_dist = 0.0
        for v in self._simplex_vertices:
            dist = float(np.linalg.norm(v - centroid))
            if dist > max_dist:
                max_dist = dist
        return max_dist

    # ======================== 内部方法: Powell ========================

    def _run_powell(self) -> None:
        """运行 Powell 方向集优化。"""
        n = len(self._parameters)
        x = self._best_value.copy()

        # 初始方向集: 坐标轴方向
        directions = [np.zeros(n, dtype=np.float64) for _ in range(n)]
        for i in range(n):
            directions[i][i] = 1.0

        for iteration in range(self.config.max_iterations):
            if self._no_improvement_count >= self.config.max_no_improvement:
                LOGGER.info(
                    "AutoAlignmentOptimizer: Powell 收敛 (无改善 %d 次)",
                    self._no_improvement_count,
                )
                break

            prev_obj = self._best_objective
            x_start = x.copy()

            # 沿每个方向进行一维搜索
            max_delta = 0.0
            max_delta_idx = 0

            for i in range(n):
                direction = directions[i]
                step = self._get_adaptive_step(iteration)

                # 沿方向探测
                x_plus = x + step * direction
                x_plus = self._clamp(x_plus)
                obj_plus = self._evaluate(x_plus)

                x_minus = x - step * direction
                x_minus = self._clamp(x_minus)
                obj_minus = self._evaluate(x_minus)

                # 选择最佳
                if obj_plus >= obj_minus and obj_plus > self._best_objective:
                    delta = obj_plus - self._best_objective
                    x = x_plus
                    self._best_objective = obj_plus
                elif obj_minus > self._best_objective:
                    delta = obj_minus - self._best_objective
                    x = x_minus
                    self._best_objective = obj_minus
                else:
                    delta = 0.0

                if delta > max_delta:
                    max_delta = delta
                    max_delta_idx = i

            self._best_value = x.copy()
            self._record_iteration(x, self._best_objective, self._get_adaptive_step(iteration))

            # 更新方向集: 用总位移方向替换改善最大的方向
            total_displacement = x - x_start
            if np.linalg.norm(total_displacement) > 1e-12:
                directions[max_delta_idx] = total_displacement / np.linalg.norm(total_displacement)

            if iteration % 20 == 0:
                LOGGER.debug(
                    "AutoAlignmentOptimizer: Powell iter=%d, obj=%.6f",
                    iteration, self._best_objective,
                )

    # ======================== 内部方法: 自适应步长 ========================

    def _get_adaptive_step(self, iteration: int) -> float:
        """计算自适应步长。

        随迭代次数逐渐减小步长。
        """
        if not self.config.adaptive_step:
            return self._parameters[0].step_size * self.config.initial_step_factor

        decay = 1.0 / (1.0 + 0.01 * iteration)
        base_step = np.mean([p.step_size for p in self._parameters])
        return base_step * self.config.initial_step_factor * decay


# ======================== 测试入口 ========================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    print("=== 自动对准寻优器测试 ===\n")

    # 定义测试目标函数: 多峰函数 (负Rastrigin, 最大化)
    def test_objective(x: np.ndarray) -> float:
        """测试目标函数: 负 Rastrigin 函数 (最大化)。"""
        n = len(x)
        result = 10.0 * n
        for i in range(n):
            result += x[i] ** 2 - 10.0 * np.cos(2.0 * np.pi * x[i])
        return -result  # 取负以转为最大化

    # 测试 Nelder-Mead
    print("--- Nelder-Mead 优化 ---")
    optimizer_nm = AutoAlignmentOptimizer(OptimizationConfig(
        method="nelder_mead",
        max_iterations=500,
        tolerance=1e-8,
    ))
    optimizer_nm.add_parameter(AlignmentParameter("x_offset", 3.0, -5.12, 5.12, 0.5))
    optimizer_nm.add_parameter(AlignmentParameter("y_offset", 4.0, -5.12, 5.12, 0.5))
    optimizer_nm.add_parameter(AlignmentParameter("z_offset", 1.0, -5.12, 5.12, 0.5))
    optimizer_nm.set_objective_function(test_objective)

    result_nm = optimizer_nm.optimize("nelder_mead")
    print(f"  最优参数: {result_nm.best_parameters}")
    print(f"  最优目标值: {result_nm.best_objective:.6f}")
    print(f"  迭代次数: {result_nm.total_iterations}")
    print(f"  收敛: {result_nm.converged}")
    print(f"  耗时: {result_nm.elapsed_time_s:.3f}s\n")

    # 测试坐标下降
    print("--- 坐标下降优化 ---")
    optimizer_cd = AutoAlignmentOptimizer(OptimizationConfig(
        method="coordinate_descent",
        max_iterations=500,
        tolerance=1e-8,
    ))
    optimizer_cd.add_parameter(AlignmentParameter("x_offset", 3.0, -5.12, 5.12, 0.5))
    optimizer_cd.add_parameter(AlignmentParameter("y_offset", 4.0, -5.12, 5.12, 0.5))
    optimizer_cd.add_parameter(AlignmentParameter("z_offset", 1.0, -5.12, 5.12, 0.5))
    optimizer_cd.set_objective_function(test_objective)

    result_cd = optimizer_cd.optimize("coordinate_descent")
    print(f"  最优参数: {result_cd.best_parameters}")
    print(f"  最优目标值: {result_cd.best_objective:.6f}")
    print(f"  迭代次数: {result_cd.total_iterations}")
    print(f"  收敛: {result_cd.converged}")
    print(f"  耗时: {result_cd.elapsed_time_s:.3f}s\n")

    # 测试 Powell
    print("--- Powell 优化 ---")
    optimizer_pw = AutoAlignmentOptimizer(OptimizationConfig(
        method="powell",
        max_iterations=500,
        tolerance=1e-8,
    ))
    optimizer_pw.add_parameter(AlignmentParameter("x_offset", 3.0, -5.12, 5.12, 0.5))
    optimizer_pw.add_parameter(AlignmentParameter("y_offset", 4.0, -5.12, 5.12, 0.5))
    optimizer_pw.add_parameter(AlignmentParameter("z_offset", 1.0, -5.12, 5.12, 0.5))
    optimizer_pw.set_objective_function(test_objective)

    result_pw = optimizer_pw.optimize("powell")
    print(f"  最优参数: {result_pw.best_parameters}")
    print(f"  最优目标值: {result_pw.best_objective:.6f}")
    print(f"  迭代次数: {result_pw.total_iterations}")
    print(f"  收敛: {result_pw.converged}")
    print(f"  耗时: {result_pw.elapsed_time_s:.3f}s\n")

    # 测试单步接口
    print("--- 单步接口测试 ---")
    optimizer_step = AutoAlignmentOptimizer()
    optimizer_step.add_parameter(AlignmentParameter("x", 2.0, -5, 5, 0.3))
    optimizer_step.add_parameter(AlignmentParameter("y", 2.0, -5, 5, 0.3))
    optimizer_step.set_objective_function(lambda x: -(x[0]**2 + x[1]**2))

    for i in range(10):
        params, obj = optimizer_step.step_coordinate_descent()
        print(f"  步骤 {i+1}: params={params}, obj={obj:.6f}")

    print("\n测试完成")
