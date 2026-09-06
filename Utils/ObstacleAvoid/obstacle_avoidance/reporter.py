"""运行报告（PLAN 第 6 节）：JSONL 流式写入 + 回放摘要。

事件类型：run_start / task_config / state_change / detection / plan /
stage_command / estop / pause / error / run_end。
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Optional

from .models import new_run_id, now


def _jsonable(v: Any) -> Any:
    if is_dataclass(v):
        return asdict(v)
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if hasattr(v, "value") and hasattr(v, "name"):  # Enum
        return v.value
    return v


class RunReporter:
    """流式 JSONL 报告器。path=None 时不落盘（测试用）。"""

    def __init__(self, path: Optional[str] = None, run_id: Optional[str] = None) -> None:
        self.path = path
        self.run_id = run_id or new_run_id()
        self.events: List[Dict[str, Any]] = []
        self._fh = open(path, "a", encoding="utf-8") if path else None
        self._seq = 0

    def log(self, event_type: str, task_id: str = "", **data: Any) -> Dict[str, Any]:
        self._seq += 1
        rec = {"seq": self._seq, "run_id": self.run_id, "ts": now(),
               "event": event_type, "task_id": task_id}
        rec.update(_jsonable(data))
        line = json.dumps(rec, ensure_ascii=False)
        self.events.append(rec)
        if self._fh:
            self._fh.write(line + "\n")
            self._fh.flush()
        return rec

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "RunReporter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# ---------------------------------------------------------------- replay
def replay(path: str) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def summarize(events: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """UI-02 回放摘要：状态、命令数、错误、最终指标。"""
    evs = list(events)
    summary: Dict[str, Any] = {
        "run_id": evs[0].get("run_id") if evs else None,
        "event_count": len(evs),
        "state_changes": [],
        "stage_command_count": 0,
        "errors": [],
        "plans": 0,
        "final_metrics": None,
        "final_state": None,
    }
    for e in evs:
        et = e.get("event")
        if et == "state_change":
            summary["state_changes"].append(
                {"ts": e.get("ts"), "from": e.get("from"), "to": e.get("to")})
            summary["final_state"] = e.get("to")
        elif et == "stage_command":
            summary["stage_command_count"] += 1
        elif et == "error":
            summary["errors"].append({"ts": e.get("ts"), "reason": e.get("reason")})
        elif et == "plan":
            summary["plans"] += 1
        elif et == "run_end":
            summary["final_metrics"] = e.get("metrics")
            if e.get("final_state"):
                summary["final_state"] = e.get("final_state")
    return summary
