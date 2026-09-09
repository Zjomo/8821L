"""运行报告（PLAN 第 6 节）：JSONL 流式写入 + 回放摘要。

事件类型：run_start / task_config / state_change / detection / plan /
stage_command / estop / pause / error / run_end。
"""
from __future__ import annotations

import json
import math
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

    def finalize_experiment(self, **fields: Any) -> Dict[str, Any]:
        """Append a durable, self-contained experiment summary event."""
        metrics = experiment_metrics(self.events)
        metrics.update(_jsonable(fields))
        self.log("experiment_summary", metrics=metrics)
        return metrics

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
        "motor_step_count": 0,
        "beam_target_switches": 0,
        "experiment_metrics": None,
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
        elif et == "motor_step":
            summary["motor_step_count"] += 1
        elif et == "beam_target":
            summary["beam_target_switches"] += 1
        elif et == "experiment_summary":
            summary["experiment_metrics"] = e.get("metrics")
        elif et == "run_end":
            summary["final_metrics"] = e.get("metrics")
            if e.get("final_state"):
                summary["final_state"] = e.get("final_state")
    return summary


def experiment_metrics(events: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute comparable Alg1/Alg2 metrics from one JSONL experiment."""
    evs = list(events)
    starts = [e for e in evs if e.get("event") == "run_start"]
    start = starts[0] if starts else (evs[0] if evs else {})
    ends = [e for e in evs if e.get("event") == "run_end"]
    end = ends[-1] if ends else {}
    plans = [e for e in evs if e.get("event") == "plan" and e.get("success")]
    detections = [e for e in evs if e.get("event") == "detection"]
    uncertain = [e for e in detections if e.get("uncertain")]
    commands = [e for e in evs if e.get("event") == "stage_command"]
    motors = [e for e in evs if e.get("event") == "motor_step"]
    beam = [e for e in evs if e.get("event") == "beam_target"]
    task_configs = [e for e in evs if e.get("event") == "task_config"]
    ts = [float(e.get("ts")) for e in evs if e.get("ts") is not None]
    duration = max(0.0, (max(ts) - min(ts)) if ts else 0.0)
    actual_mm = sum(math.hypot(float(e.get("dx_mm", 0.0)),
                               float(e.get("dy_mm", 0.0))) for e in commands)
    planned_px = sum(float(e.get("length_px", 0.0) or 0.0) for e in plans)
    clearances = [float(e["min_clearance_px"]) for e in plans
                  if e.get("min_clearance_px") is not None]
    final_metrics = end.get("metrics") or {}
    final_state = end.get("final_state")
    if not final_state and ends:
        final_state = ends[-1].get("state")
    steps_by_axis = {axis: 0 for axis in ("X", "Y", "Z")}
    reversals = 0
    last_dir = {}
    for e in motors:
        axis = str(e.get("axis", "")).upper()
        steps_by_axis[axis] = steps_by_axis.get(axis, 0) + int(e.get("steps", 0) or 0)
        direction = e.get("direction")
        if axis in last_dir and direction != last_dir[axis]:
            reversals += 1
        last_dir[axis] = direction
    trails = extract_trajectory(evs).get("trails", {})
    trail_px = 0.0
    net_px = 0.0
    for chain in trails.values():
        for a, b in zip(chain, chain[1:]):
            trail_px += math.dist(a, b)
        if len(chain) >= 2:
            net_px += math.dist(chain[0], chain[-1])
    efficiency = (net_px / trail_px) if trail_px > 1e-9 else None
    agg = final_metrics.get("metrics") if isinstance(final_metrics, dict) else None
    return {
        "run_id": start.get("run_id") or (evs[0].get("run_id") if evs else None),
        "algorithm": start.get("algorithm", "Alg1"),
        "task_mode": start.get("task_mode", start.get("mode", "virtual")),
        "scenario": start.get("scenario", "sim01"),
        "final_state": final_state,
        "success": bool(final_state == "COMPLETE" or final_metrics.get("completed")),
        "duration_s": round(duration, 6),
        "stage_commands": len(commands),
        "motor_step_events": len(motors),
        "motor_steps_total": int(sum(steps_by_axis.values())),
        "motor_steps_by_axis": steps_by_axis,
        "motor_direction_reversals": reversals,
        "beam_target_switches": len(beam),
        "focus_moves": sum(1 for e in motors if e.get("source") in
                            ("alg2-z-safe", "alg2-z-focus")),
        "plans": len(plans),
        "replans": max(0, len(plans) - max(1, len(task_configs))),
        "planned_path_px": round(planned_px, 3),
        "actual_path_mm": round(actual_mm, 6),
        "path_efficiency": (round(efficiency, 6) if efficiency is not None else None),
        "min_clearance_px": (min(clearances) if clearances else None),
        "detections": len(detections),
        "uncertain_detections": len(uncertain),
        "uncertain_rate": (round(len(uncertain) / len(detections), 6)
                           if detections else 0.0),
        "final_error_px": final_metrics.get("final_error_px"),
        "aggregation": agg if isinstance(agg, dict) else None,
    }


def aggregate_experiments(reports: Iterable[str | Iterable[Dict[str, Any]]]
                          ) -> Dict[str, Any]:
    """Aggregate multiple JSONL reports, grouped by algorithm."""
    rows: List[Dict[str, Any]] = []
    for item in reports:
        if isinstance(item, (str, os.PathLike)):
            try:
                rows.append(experiment_metrics(replay(os.fspath(item))))
            except (OSError, ValueError, TypeError):
                continue
        else:
            rows.append(experiment_metrics(item))
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("algorithm", "Alg1")), []).append(row)
    result = {"experiment_count": len(rows), "experiments": rows,
              "by_algorithm": {}}
    numeric = ("duration_s", "stage_commands", "motor_steps_total",
               "plans", "replans", "planned_path_px", "actual_path_mm",
               "path_efficiency", "min_clearance_px", "uncertain_rate",
               "final_error_px")
    for alg, items in groups.items():
        stats: Dict[str, Any] = {"runs": len(items),
                                 "successes": sum(bool(x["success"]) for x in items),
                                 "success_rate": sum(bool(x["success"]) for x in items) / len(items)}
        for key in numeric:
            vals = [float(x[key]) for x in items if x.get(key) is not None]
            stats[f"mean_{key}"] = (sum(vals) / len(vals)) if vals else None
        result["by_algorithm"][alg] = stats
    return result


# ---------------------------------------------------------------- replay
def extract_trajectory(events: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """从事件流提取回放数据（圆球移动完整过程）。

    返回::

        {
          "run_id": str|None, "final_state": str|None,
          "tasks": {task_id: {"goal": [x,y]|None, "track_id": int|None,
                               "waypoints": [[x,y],...]}},
          "frames": [{"ts", "task_id",
                      "particles": [{"track_id", "position_px", "radius_px"}]}],
          "trails": {task_id: [[x, y], ...]},   # 逐任务球心轨迹（最近邻链）
        }

    frames 仅含 uncertain=False 的 detection 事件；轨迹优先按 track_id
    匹配，缺失时用最近邻链（与控制器 _match_particle 一致）。
    """
    tasks: Dict[str, Dict[str, Any]] = {}
    frames: List[Dict[str, Any]] = []
    run_id = None
    final_state = None
    for e in events:
        et = e.get("event")
        if run_id is None:
            run_id = e.get("run_id")
        if et == "task_config":
            tid = e.get("task_id") or "task"
            goal = e.get("goal")
            tasks[tid] = {
                "goal": (list(goal.get("center") or [])
                         if isinstance(goal, dict) else None),
                "track_id": e.get("track_id"),
                "waypoints": [],
            }
        elif et == "plan" and e.get("success"):
            tid = e.get("task_id") or "task"
            tasks.setdefault(tid, {"goal": None, "track_id": None,
                                   "waypoints": []})
            tasks[tid]["waypoints"] = [list(p) for p in
                                       (e.get("waypoints_px") or [])]
        elif et == "detection" and not e.get("uncertain"):
            frames.append({
                "ts": e.get("ts"),
                "task_id": e.get("task_id") or "task",
                "particles": [
                    {"track_id": p.get("track_id"),
                     "position_px": [float(p["position_px"][0]),
                                     float(p["position_px"][1])],
                     "radius_px": float(p.get("radius_px", 0.0))}
                    for p in (e.get("particles") or [])
                    if p.get("position_px")],
            })
        elif et == "run_end":
            if e.get("final_state"):
                final_state = e.get("final_state")

    # 逐任务轨迹：track_id 优先，否则最近邻链
    trails: Dict[str, List[List[float]]] = {}
    for tid, info in tasks.items():
        track = info.get("track_id")
        chain: List[List[float]] = []
        last = None
        for fr in frames:
            if fr["task_id"] != tid or not fr["particles"]:
                continue
            best = None
            if track is not None:
                for p in fr["particles"]:
                    if p["track_id"] == track:
                        best = p
                        break
            if best is None and last is not None:
                best = min(fr["particles"],
                           key=lambda p: (p["position_px"][0] - last[0]) ** 2
                           + (p["position_px"][1] - last[1]) ** 2)
            if best is None:
                best = fr["particles"][0]
            pos = [round(best["position_px"][0], 1),
                   round(best["position_px"][1], 1)]
            if last is None or pos != last:
                chain.append(pos)
                last = pos
        if chain:
            trails[tid] = chain
    return {"run_id": run_id, "final_state": final_state,
            "tasks": tasks, "frames": frames, "trails": trails}
