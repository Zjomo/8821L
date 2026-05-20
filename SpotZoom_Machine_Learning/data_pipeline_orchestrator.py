"""
数据流水线编排器 (DataPipelineOrchestrator)

灵感来源:
- Bluesky Plan 模式 (NSLS-II) — 同步辐射光源实验编排框架，将复杂实验
  分解为可组合、可中断、可恢复的 Plan，支持条件分支和参数扫描
- ARTIQ 编译实验 — 量子实验控制框架，将实验流程编译为确定性的时间序列，
  支持精确的时序控制和中断恢复
- PyMeasure Procedure — 测量过程管理框架，将测量步骤封装为可重用、
  可序列化的 Procedure 对象，支持进度跟踪和错误处理

算法原理:
- Pipeline Pattern — 流水线模式，将复杂任务分解为有序步骤序列
- Thread-based Execution — 基于线程的异步执行，支持暂停/恢复/中止
- Event Synchronization — 事件同步机制，使用 threading.Event 实现
  步骤间的协调和控制
- Conditional Retry — 条件重试，根据失败类型和重试策略决定是否重试

功能:
- 定义可组合的多步骤流水线
- 支持条件分支、条件重试、参数扫描
- 支持暂停、恢复、中止操作
- 实时进度跟踪和结果收集
- 超时保护和错误处理

依赖: numpy, logging, dataclasses, threading, time (无外部框架)
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np

LOGGER = logging.getLogger("SpotZoom.DataPipelineOrchestrator")


class PipelineState(Enum):
    """流水线状态。"""
    IDLE = "idle"               # 空闲
    RUNNING = "running"         # 运行中
    PAUSED = "paused"           # 已暂停
    COMPLETED = "completed"     # 已完成
    FAILED = "failed"           # 已失败
    ABORTED = "aborted"         # 已中止


@dataclass
class PipelineStep:
    """流水线步骤定义。"""
    name: str                                        # 步骤名称
    action: Callable[[], Any]                        # 执行动作 (无参数 callable)
    condition: Optional[Callable[[], bool]] = None   # 执行条件 (返回 True 才执行)
    retry_count: int = 0                             # 失败重试次数
    timeout_s: float = 60.0                          # 超时时间 (秒)
    on_failure: Optional[Callable[[Exception], Any]] = None  # 失败回调


@dataclass
class PipelineProgress:
    """流水线进度。"""
    current_step: int                # 当前步骤索引 (0-based)
    total_steps: int                 # 总步骤数
    elapsed_s: float                 # 已用时间 (秒)
    status: PipelineState            # 当前状态
    step_results: List[Any] = field(default_factory=list)  # 各步骤结果
    current_step_name: str = ""      # 当前步骤名称
    retry_attempt: int = 0           # 当前重试次数


@dataclass
class PipelineResult:
    """流水线执行结果。"""
    success: bool                            # 是否成功完成
    total_time_s: float                      # 总耗时 (秒)
    steps_completed: int                     # 完成的步骤数
    step_results: List[Any]                  # 各步骤结果
    final_state: PipelineState               # 最终状态
    failed_step: Optional[str] = None        # 失败步骤名称
    error_message: Optional[str] = None      # 错误消息


class DataPipelineOrchestrator:
    """数据流水线编排器。

    将复杂的多步骤对准/测量流程编排为可组合、可中断、可恢复的流水线。

    Parameters
    ----------
    default_timeout_s : float
        默认步骤超时时间 (秒)。
    default_retry_count : int
        默认失败重试次数。
    poll_interval_s : float
        暂停/中止轮询间隔 (秒)。
    """

    def __init__(
        self,
        default_timeout_s: float = 60.0,
        default_retry_count: int = 0,
        poll_interval_s: float = 0.1,
    ):
        self.default_timeout_s = float(default_timeout_s)
        self.default_retry_count = int(default_retry_count)
        self.poll_interval_s = float(poll_interval_s)

        # 流水线定义
        self._steps: List[PipelineStep] = []

        # 执行状态
        self._state: PipelineState = PipelineState.IDLE
        self._progress: PipelineProgress = PipelineProgress(
            current_step=0, total_steps=0, elapsed_s=0.0,
            status=PipelineState.IDLE,
        )

        # 线程控制
        self._thread: Optional[threading.Thread] = None
        self._pause_event = threading.Event()
        self._pause_event.set()  # 初始不暂停
        self._abort_event = threading.Event()
        self._lock = threading.Lock()

        # 结果
        self._result: Optional[PipelineResult] = None
        self._step_results: List[Any] = []

        LOGGER.info(
            "DataPipelineOrchestrator: 初始化完成 "
            "(timeout=%.1fs, retry=%d, poll=%.3fs)",
            self.default_timeout_s, self.default_retry_count,
            self.poll_interval_s,
        )

    def define_pipeline(self, steps: List[PipelineStep]) -> None:
        """定义流水线步骤序列。

        Parameters
        ----------
        steps : List[PipelineStep]
            步骤列表。

        Raises
        ------
        RuntimeError
            如果流水线正在运行。
        """
        with self._lock:
            if self._state == PipelineState.RUNNING:
                raise RuntimeError("流水线正在运行，无法重新定义")

            self._steps = list(steps)
            self._state = PipelineState.IDLE
            self._result = None
            self._step_results = []

            self._progress = PipelineProgress(
                current_step=0,
                total_steps=len(self._steps),
                elapsed_s=0.0,
                status=PipelineState.IDLE,
            )

            LOGGER.info(
                "DataPipelineOrchestrator: 流水线已定义 (%d 步)",
                len(self._steps),
            )

    def run(self) -> None:
        """启动流水线执行 (非阻塞)。

        Raises
        ------
        RuntimeError
            如果流水线已在运行或未定义。
        """
        with self._lock:
            if self._state == PipelineState.RUNNING:
                raise RuntimeError("流水线已在运行")
            if not self._steps:
                raise RuntimeError("流水线未定义")

            # 重置状态
            self._state = PipelineState.RUNNING
            self._abort_event.clear()
            self._pause_event.set()
            self._step_results = []
            self._result = None

            self._progress = PipelineProgress(
                current_step=0,
                total_steps=len(self._steps),
                elapsed_s=0.0,
                status=PipelineState.RUNNING,
            )

            self._thread = threading.Thread(
                target=self._execute_pipeline,
                daemon=True,
            )
            self._thread.start()

            LOGGER.info("DataPipelineOrchestrator: 流水线启动")

    def pause(self) -> None:
        """暂停流水线执行。"""
        if self._state != PipelineState.RUNNING:
            LOGGER.warning("DataPipelineOrchestrator: 无法暂停，当前状态: %s", self._state.value)
            return

        self._pause_event.clear()
        self._state = PipelineState.PAUSED
        with self._lock:
            self._progress.status = PipelineState.PAUSED
        LOGGER.info("DataPipelineOrchestrator: 流水线已暂停")

    def resume(self) -> None:
        """恢复流水线执行。"""
        if self._state != PipelineState.PAUSED:
            LOGGER.warning("DataPipelineOrchestrator: 无法恢复，当前状态: %s", self._state.value)
            return

        self._pause_event.set()
        self._state = PipelineState.RUNNING
        with self._lock:
            self._progress.status = PipelineState.RUNNING
        LOGGER.info("DataPipelineOrchestrator: 流水线已恢复")

    def abort(self) -> None:
        """中止流水线执行。"""
        if self._state not in (PipelineState.RUNNING, PipelineState.PAUSED):
            LOGGER.warning("DataPipelineOrchestrator: 无法中止，当前状态: %s", self._state.value)
            return

        self._abort_event.set()
        self._pause_event.set()  # 解除暂停以允许线程退出
        self._state = PipelineState.ABORTED
        with self._lock:
            self._progress.status = PipelineState.ABORTED
        LOGGER.info("DataPipelineOrchestrator: 流水线已中止")

    def get_progress(self) -> PipelineProgress:
        """获取当前进度。

        Returns
        -------
        PipelineProgress
            进度快照。
        """
        with self._lock:
            return PipelineProgress(
                current_step=self._progress.current_step,
                total_steps=self._progress.total_steps,
                elapsed_s=self._progress.elapsed_s,
                status=self._progress.status,
                step_results=list(self._step_results),
                current_step_name=self._progress.current_step_name,
                retry_attempt=self._progress.retry_attempt,
            )

    def get_results(self) -> Optional[PipelineResult]:
        """获取执行结果。

        Returns
        -------
        PipelineResult or None
            执行结果。流水线未完成时返回 None。
        """
        return self._result

    def reset(self) -> None:
        """重置编排器到初始状态。

        如果流水线正在运行，先中止。
        """
        if self._state in (PipelineState.RUNNING, PipelineState.PAUSED):
            self.abort()
            if self._thread is not None:
                self._thread.join(timeout=5.0)

        with self._lock:
            self._steps = []
            self._state = PipelineState.IDLE
            self._progress = PipelineProgress(
                current_step=0, total_steps=0, elapsed_s=0.0,
                status=PipelineState.IDLE,
            )
            self._result = None
            self._step_results = []
            self._thread = None

        LOGGER.info("DataPipelineOrchestrator: 编排器已重置")

    # ======================== 内部方法 ========================

    def _execute_pipeline(self) -> None:
        """流水线执行主循环 (在独立线程中运行)。"""
        start_time = time.time()
        steps_completed = 0
        failed_step_name: Optional[str] = None
        error_message: Optional[str] = None

        for i, step in enumerate(self._steps):
            # 检查中止
            if self._abort_event.is_set():
                self._state = PipelineState.ABORTED
                break

            # 等待暂停解除
            self._pause_event.wait()

            # 暂停解除后再检查一次中止
            if self._abort_event.is_set():
                self._state = PipelineState.ABORTED
                break

            # 更新进度
            with self._lock:
                self._progress.current_step = i
                self._progress.current_step_name = step.name
                self._progress.elapsed_s = time.time() - start_time

            # 检查条件
            if step.condition is not None:
                try:
                    should_run = bool(step.condition())
                except Exception as exc:
                    LOGGER.warning(
                        "DataPipelineOrchestrator: 步骤 '%s' 条件检查失败: %s",
                        step.name, exc,
                    )
                    should_run = False

                if not should_run:
                    LOGGER.info(
                        "DataPipelineOrchestrator: 步骤 '%s' 条件不满足，跳过",
                        step.name,
                    )
                    self._step_results.append(None)
                    steps_completed += 1
                    continue

            # 执行步骤 (含重试)
            result = self._execute_step(step, start_time)

            if result is None:
                # 步骤失败且重试用尽
                failed_step_name = step.name
                error_message = f"步骤 '{step.name}' 执行失败 (重试耗尽)"
                self._state = PipelineState.FAILED
                break

            self._step_results.append(result)
            steps_completed += 1

        # 计算总耗时
        total_time = time.time() - start_time

        # 生成结果
        if self._state == PipelineState.ABORTED:
            final_state = PipelineState.ABORTED
            success = False
        elif self._state == PipelineState.FAILED:
            final_state = PipelineState.FAILED
            success = False
        else:
            final_state = PipelineState.COMPLETED
            self._state = PipelineState.COMPLETED
            success = True

        with self._lock:
            self._progress.status = final_state
            self._progress.elapsed_s = total_time

        self._result = PipelineResult(
            success=success,
            total_time_s=round(total_time, 4),
            steps_completed=steps_completed,
            step_results=list(self._step_results),
            final_state=final_state,
            failed_step=failed_step_name,
            error_message=error_message,
        )

        LOGGER.info(
            "DataPipelineOrchestrator: 流水线 %s (耗时=%.2fs, 完成=%d/%d)",
            final_state.value, total_time, steps_completed, len(self._steps),
        )

    def _execute_step(
        self,
        step: PipelineStep,
        pipeline_start_time: float,
    ) -> Optional[Any]:
        """执行单个步骤 (含超时和重试)。

        Parameters
        ----------
        step : PipelineStep
            步骤定义。
        pipeline_start_time : float
            流水线启动时间。

        Returns
        -------
        Any or None
            步骤执行结果。失败时返回 None。
        """
        max_retries = step.retry_count if step.retry_count > 0 else self.default_retry_count
        timeout = step.timeout_s if step.timeout_s > 0 else self.default_timeout_s

        for attempt in range(max_retries + 1):
            # 检查中止
            if self._abort_event.is_set():
                return None

            # 更新重试进度
            with self._lock:
                self._progress.retry_attempt = attempt
                self._progress.elapsed_s = time.time() - pipeline_start_time

            LOGGER.debug(
                "DataPipelineOrchestrator: 执行步骤 '%s' (尝试 %d/%d)",
                step.name, attempt + 1, max_retries + 1,
            )

            # 使用线程执行步骤以支持超时
            result_holder: List[Any] = []
            error_holder: List[Exception] = []
            done_event = threading.Event()

            def _run_action():
                try:
                    result_holder.append(step.action())
                except Exception as exc:
                    error_holder.append(exc)
                finally:
                    done_event.set()

            action_thread = threading.Thread(target=_run_action, daemon=True)
            action_thread.start()

            # 等待完成或超时
            finished = done_event.wait(timeout=timeout)

            if not finished:
                # 超时
                LOGGER.warning(
                    "DataPipelineOrchestrator: 步骤 '%s' 超时 (%.1fs)",
                    step.name, timeout,
                )
                if attempt < max_retries:
                    LOGGER.info("  将重试...")
                    continue
                return None

            if error_holder:
                exc = error_holder[0]
                LOGGER.warning(
                    "DataPipelineOrchestrator: 步骤 '%s' 失败: %s",
                    step.name, exc,
                )

                # 调用失败回调
                if step.on_failure is not None:
                    try:
                        step.on_failure(exc)
                    except Exception as cb_exc:
                        LOGGER.error(
                            "DataPipelineOrchestrator: 步骤 '%s' 失败回调异常: %s",
                            step.name, cb_exc,
                        )

                if attempt < max_retries:
                    LOGGER.info("  将重试...")
                    time.sleep(0.1)  # 重试前短暂等待
                    continue
                return None

            # 成功
            with self._lock:
                self._progress.retry_attempt = 0
            LOGGER.debug(
                "DataPipelineOrchestrator: 步骤 '%s' 完成", step.name,
            )
            return result_holder[0] if result_holder else None

        return None

    @property
    def state(self) -> PipelineState:
        """当前流水线状态。"""
        return self._state


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    orchestrator = DataPipelineOrchestrator(
        default_timeout_s=5.0,
        default_retry_count=2,
        poll_interval_s=0.05,
    )

    print("=== 数据流水线编排器测试 ===\n")

    # 定义流水线步骤
    call_log = []

    def step1():
        """初始化步骤"""
        call_log.append("step1")
        time.sleep(0.1)
        return "初始化完成"

    def step2():
        """扫描步骤"""
        call_log.append("step2")
        time.sleep(0.2)
        return {"positions": [10, 20, 30]}

    def step3():
        """分析步骤 (可能失败)"""
        call_log.append("step3")
        time.sleep(0.1)
        if len(call_log) < 5:  # 前几次模拟失败
            raise RuntimeError("模拟分析失败")
        return {"quality": 0.95}

    def step3_failure_handler(exc):
        """步骤3失败回调"""
        call_log.append(f"step3_failure: {exc}")

    def step4():
        """优化步骤"""
        call_log.append("step4")
        time.sleep(0.15)
        return {"optimized": True}

    def condition_check():
        """条件检查: 只在前面的步骤成功时执行"""
        return len(call_log) >= 3

    steps = [
        PipelineStep(name="初始化", action=step1, timeout_s=2.0),
        PipelineStep(name="扫描", action=step2, timeout_s=3.0),
        PipelineStep(
            name="分析", action=step3,
            retry_count=3, timeout_s=2.0,
            on_failure=step3_failure_handler,
        ),
        PipelineStep(
            name="优化", action=step4,
            condition=condition_check, timeout_s=2.0,
        ),
    ]

    # 定义并运行流水线
    orchestrator.define_pipeline(steps)

    print("--- 启动流水线 ---")
    orchestrator.run()

    # 等待完成
    while orchestrator.state in (PipelineState.RUNNING, PipelineState.PAUSED):
        progress = orchestrator.get_progress()
        print(f"  进度: {progress.current_step + 1}/{progress.total_steps} "
              f"[{progress.status.value}] "
              f"步骤: {progress.current_step_name} "
              f"耗时: {progress.elapsed_s:.2f}s")
        time.sleep(0.15)

    # 获取结果
    result = orchestrator.get_results()
    if result is not None:
        print(f"\n--- 流水线结果 ---")
        print(f"  成功: {result.success}")
        print(f"  状态: {result.final_state.value}")
        print(f"  总耗时: {result.total_time_s:.2f}s")
        print(f"  完成步骤: {result.steps_completed}/{len(steps)}")
        print(f"  步骤结果: {result.step_results}")
        if result.failed_step:
            print(f"  失败步骤: {result.failed_step}")
            print(f"  错误: {result.error_message}")

    print(f"\n  调用日志: {call_log}")

    # 测试暂停/恢复
    print("\n--- 测试暂停/恢复 ---")
    call_log.clear()

    def slow_step():
        call_log.append("slow_step_tick")
        for _ in range(10):
            time.sleep(0.1)
        return "slow_done"

    orchestrator.define_pipeline([
        PipelineStep(name="慢速步骤", action=slow_step, timeout_s=10.0),
    ])
    orchestrator.run()

    time.sleep(0.3)
    print("  暂停...")
    orchestrator.pause()
    time.sleep(0.5)
    print("  恢复...")
    orchestrator.resume()

    while orchestrator.state == PipelineState.RUNNING:
        time.sleep(0.1)

    result = orchestrator.get_results()
    if result:
        print(f"  结果: 成功={result.success}, 耗时={result.total_time_s:.2f}s")
    print(f"  调用日志: {call_log}")

    # 测试中止
    print("\n--- 测试中止 ---")
    call_log.clear()

    def long_step():
        call_log.append("long_step_tick")
        for _ in range(100):
            time.sleep(0.1)
        return "long_done"

    orchestrator.define_pipeline([
        PipelineStep(name="长步骤", action=long_step, timeout_s=30.0),
    ])
    orchestrator.run()

    time.sleep(0.3)
    print("  中止...")
    orchestrator.abort()

    while orchestrator.state == PipelineState.RUNNING:
        time.sleep(0.1)

    result = orchestrator.get_results()
    if result:
        print(f"  结果: 成功={result.success}, 状态={result.final_state.value}")

    orchestrator.reset()
    print("\n测试完成")
