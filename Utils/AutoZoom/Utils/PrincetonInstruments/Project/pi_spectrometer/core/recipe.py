"""实验序列（Recipe）模块。

支持编排多步骤采集流程，例如：
    1. 设置曝光 0.1s
    2. 设置 ROI
    3. 采集 5 帧，每帧间隔 1s
    4. 保存数据

Recipe 在后台线程运行，支持暂停/停止。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pi_spectrometer.core.base import SpectrometerBackend
from pi_spectrometer.core.types import ROI, SpectrometerResult


class RecipeAction(str, Enum):
    """Recipe 支持的动作类型。"""

    ACQUIRE = "acquire"
    SET_EXPOSURE = "set_exposure"
    SET_TEMPERATURE = "set_temperature"
    SET_ROI = "set_roi"
    WAIT = "wait"
    SAVE = "save"


@dataclass
class RecipeStep:
    """单步 Recipe。"""

    action: RecipeAction
    params: Dict[str, Any] = field(default_factory=dict)
    repeats: int = 1
    delay_s: float = 0.0
    enabled: bool = True

    def validate(self) -> None:
        """验证步骤参数合法性。"""
        if self.repeats < 1:
            raise ValueError(f"repeats 必须 >= 1: {self.repeats}")
        if self.delay_s < 0:
            raise ValueError(f"delay_s 必须 >= 0: {self.delay_s}")

        action = RecipeAction(self.action)
        if action == RecipeAction.SET_EXPOSURE:
            if "seconds" not in self.params:
                raise ValueError("set_exposure 步骤需要 seconds 参数")
        elif action == RecipeAction.SET_TEMPERATURE:
            if "celsius" not in self.params:
                raise ValueError("set_temperature 步骤需要 celsius 参数")
        elif action == RecipeAction.SET_ROI:
            for key in ("x", "y", "width", "height"):
                if key not in self.params:
                    raise ValueError(f"set_roi 步骤需要 {key} 参数")
        elif action == RecipeAction.WAIT:
            if "seconds" not in self.params:
                raise ValueError("wait 步骤需要 seconds 参数")


class RecipeRunner:
    """在后台执行 Recipe。"""

    def __init__(
        self,
        backend: SpectrometerBackend,
        steps: List[RecipeStep],
        on_step_start: Optional[Callable[[int, RecipeStep], None]] = None,
        on_step_done: Optional[Callable[[int, RecipeStep, Any], None]] = None,
        on_log: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ):
        self.backend = backend
        self.steps = steps
        self.on_step_start = on_step_start
        self.on_step_done = on_step_done
        self.on_log = on_log
        self.on_error = on_error

        self._running = False
        self._paused = False
        self._current_step_index = -1
        self.results: List[SpectrometerResult] = []

    def _log(self, msg: str) -> None:
        if self.on_log:
            self.on_log(msg)

    def validate(self) -> None:
        """验证整个 Recipe。"""
        if not self.steps:
            raise ValueError("Recipe 步骤为空")
        for i, step in enumerate(self.steps):
            try:
                step.validate()
            except ValueError as e:
                raise ValueError(f"步骤 {i} 验证失败: {e}")

    def is_running(self) -> bool:
        return self._running

    def is_paused(self) -> bool:
        return self._paused

    def pause(self) -> None:
        self._paused = True
        self._log("Recipe 已暂停")

    def resume(self) -> None:
        self._paused = False
        self._log("Recipe 已继续")

    def stop(self) -> None:
        self._running = False
        self._paused = False
        self._log("Recipe 停止请求已发送")

    def run(self) -> List[SpectrometerResult]:
        """同步运行 Recipe，应在后台线程中调用。"""
        self.validate()
        self.results = []
        self._running = True
        self._paused = False
        self._current_step_index = -1

        try:
            for idx, step in enumerate(self.steps):
                self._current_step_index = idx
                if not self._running:
                    self._log("Recipe 被中断")
                    break
                if not step.enabled:
                    continue

                for repeat in range(step.repeats):
                    if not self._running:
                        break
                    self._wait_while_paused()
                    if not self._running:
                        break

                    step_label = f"步骤 {idx} ({step.action.value})"
                    if step.repeats > 1:
                        step_label += f" 第 {repeat + 1}/{step.repeats} 次"
                    self._log(f"开始 {step_label}")
                    if self.on_step_start:
                        self.on_step_start(idx, step)

                    result = self._execute_step(step)
                    if isinstance(result, SpectrometerResult):
                        self.results.append(result)

                    if self.on_step_done:
                        self.on_step_done(idx, step, result)

                    if step.delay_s > 0 and (repeat < step.repeats - 1 or step.repeats == 1):
                        self._sleep_interruptible(step.delay_s)

                    if not self._running:
                        break

                if not self._running:
                    break
        except Exception as e:
            self._log(f"Recipe 执行错误: {e}")
            if self.on_error:
                self.on_error(str(e))
        finally:
            self._running = False
            self._paused = False
            self._current_step_index = -1

        return self.results

    def _execute_step(self, step: RecipeStep) -> Any:
        action = RecipeAction(step.action)
        params = step.params

        if action == RecipeAction.ACQUIRE:
            num_frames = int(params.get("num_frames", 1))
            result = self.backend.acquire(num_frames=num_frames)
            return result

        if action == RecipeAction.SET_EXPOSURE:
            seconds = float(params["seconds"])
            self.backend.set_exposure(seconds)
            return None

        if action == RecipeAction.SET_TEMPERATURE:
            celsius = float(params["celsius"])
            self.backend.set_sensor_temperature(celsius)
            return None

        if action == RecipeAction.SET_ROI:
            roi = ROI(
                x=int(params["x"]),
                y=int(params["y"]),
                width=int(params["width"]),
                height=int(params["height"]),
                x_bin=int(params.get("x_bin", 1)),
                y_bin=int(params.get("y_bin", 1)),
            )
            self.backend.set_roi(roi)
            return None

        if action == RecipeAction.WAIT:
            seconds = float(params["seconds"])
            self._sleep_interruptible(seconds)
            return None

        if action == RecipeAction.SAVE:
            # save 动作由上层 UI 在 on_step_done 回调中处理
            return {"action": "save", "params": params}

        raise ValueError(f"未支持的 action: {action}")

    def _wait_while_paused(self) -> None:
        while self._paused and self._running:
            time.sleep(0.1)

    def _sleep_interruptible(self, seconds: float, step_s: float = 0.1) -> None:
        waited = 0.0
        while waited < seconds and self._running:
            self._wait_while_paused()
            if not self._running:
                break
            sleep_time = min(step_s, seconds - waited)
            time.sleep(sleep_time)
            waited += sleep_time


class AcquisitionRecipe:
    """Recipe 容器，便于序列化/反序列化。"""

    def __init__(self, name: str = "recipe", steps: Optional[List[RecipeStep]] = None):
        self.name = name
        self.steps: List[RecipeStep] = steps or []

    def add_step(self, step: RecipeStep) -> None:
        self.steps.append(step)

    def remove_step(self, index: int) -> bool:
        if 0 <= index < len(self.steps):
            del self.steps[index]
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "steps": [
                {
                    "action": s.action.value,
                    "params": s.params,
                    "repeats": s.repeats,
                    "delay_s": s.delay_s,
                    "enabled": s.enabled,
                }
                for s in self.steps
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AcquisitionRecipe":
        recipe = cls(name=data.get("name", "recipe"))
        for s in data.get("steps", []):
            recipe.add_step(
                RecipeStep(
                    action=RecipeAction(s["action"]),
                    params=s.get("params", {}),
                    repeats=s.get("repeats", 1),
                    delay_s=s.get("delay_s", 0.0),
                    enabled=s.get("enabled", True),
                )
            )
        return recipe

    def create_runner(
        self,
        backend: SpectrometerBackend,
        on_step_start: Optional[Callable[[int, RecipeStep], None]] = None,
        on_step_done: Optional[Callable[[int, RecipeStep, Any], None]] = None,
        on_log: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> RecipeRunner:
        return RecipeRunner(
            backend=backend,
            steps=list(self.steps),
            on_step_start=on_step_start,
            on_step_done=on_step_done,
            on_log=on_log,
            on_error=on_error,
        )
