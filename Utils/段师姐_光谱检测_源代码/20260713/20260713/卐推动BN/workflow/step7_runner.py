from __future__ import annotations

import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from modified_minimal_device_buttons import MeasurementWorkflow


class Step7Runner:
    """Owns Step7 A-pushes-B execution logic while sharing workflow state."""

    def __init__(self, workflow: "MeasurementWorkflow"):
        self.workflow = workflow
        self.stop_event = threading.Event()

    def __getattr__(self, name: str):
        return getattr(self.workflow, name)

    def __setattr__(self, name: str, value: Any):
        if name in {"workflow", "stop_event"} or "workflow" not in self.__dict__:
            object.__setattr__(self, name, value)
        else:
            setattr(self.workflow, name, value)

    def run_until_threshold(
        self,
        baseline_angle: Optional[float],
        min_delta_deg: Optional[float] = None,
        max_delta_deg: Optional[float] = None,
    ) -> Dict[str, Any]:
        return self.run_until_angle_delta(
            baseline_angle=baseline_angle,
            min_delta_deg=min_delta_deg,
            max_delta_deg=max_delta_deg,
        )

    def run_realtime_until_angle_delta(
        self,
        baseline_angle: Optional[float],
        min_delta_deg: Optional[float] = None,
        max_delta_deg: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Step7 实时双线程版本：控制线程运动，角度线程在 moving/pause 期间持续检测。"""
        self.log("========== RuleAB：Step7实时双线程，Bmask最长边监控角度 ==========")
        min_delta = float(self.cfg.rule_ab_angle_delta_min_deg if min_delta_deg is None else min_delta_deg)
        max_delta = None if max_delta_deg is None else float(max_delta_deg)
        follower = self.ensure_rule_ab_follower()
        self._ensure_rule_ab_feature_tracker_installed(follower, self._get_loaded_calibration_state())
        self.apply_runtime_rule_ab_params_to_follower(follower, reason="step7_realtime_start")
        self.stop_event.clear()
        stop_event = self.stop_event
        controller_done = threading.Event()
        records: List[Dict[str, Any]] = []
        records_lock = threading.Lock()
        phase_state: Dict[str, Any] = {"phase": "starting", "controller_action": "none", "step": 0}
        result_box: Dict[str, Any] = {"trigger_type": None, "stop_reason": "not_started", "final_angle": None, "final_delta": None, "controller_error": None, "angle_error": None}
        # 上一帧角度仅用于 CSV 诊断；不做角度修复或候选边替换。
        last_confirmed_angle_box: Dict[str, Any] = {"value": float(baseline_angle) if baseline_angle is not None else None}
        step_counter = {"value": 0}
        start_t = time.time()
        max_duration_s = float(getattr(self.cfg, "rule_ab_realtime_max_duration_s", 0.0) or 0.0)
        save_every = bool(getattr(self.cfg, "rule_ab_realtime_save_every_angle_frame", True))
        out_base = self.run_session_dir if self.run_session_dir is not None else self.output_root
        realtime_dir = Path(out_base) / str(getattr(self.cfg, "rule_ab_realtime_overlay_dir_name", "ab_angle_realtime_bmask_longest_edge"))
        realtime_dir.mkdir(parents=True, exist_ok=True)
        csv_path = realtime_dir / "step7_realtime_angle_records.csv"
        self.context.pop("step7_c_edge_route_state", None)

        def _append_record(rec: Dict[str, Any]) -> None:
            with records_lock:
                records.append(self._json_safe(rec))

        def angle_monitor_thread() -> None:
            frame_idx = 0
            old_override = self.context.get("bmask_longest_edge_overlay_dir_override")
            self.context["bmask_longest_edge_overlay_dir_override"] = str(getattr(self.cfg, "rule_ab_realtime_overlay_dir_name", "ab_angle_realtime_bmask_longest_edge"))
            try:
                while not stop_event.is_set() and not self.stop_requested and not self._is_midrun_recalibration_requested():
                    frame_idx += 1
                    try:
                        with self.rule_ab_vision_lock:
                            angle_result = self.detect_step7_yolo_obb_angle_once(
                                follower=follower,
                                label=f"step7_realtime_frame_{frame_idx:06d}",
                                baseline_angle=baseline_angle,
                                allow_fail=True,
                                allow_close_from_relocation=False,
                                save_overlay=save_every,
                            )
                        angle_ok = bool(angle_result.get("ok", False) and angle_result.get("angle_deg") is not None)
                        angle_deg = float(angle_result["angle_deg"]) if angle_ok else None
                        raw_angle = angle_deg
                        last_before = last_confirmed_angle_box.get("value")
                        delta = self.angle_diff_deg(float(angle_deg), float(baseline_angle)) if angle_deg is not None and baseline_angle is not None else None
                        raw_delta_to_last = self.angle_diff_deg(float(angle_deg), float(last_before)) if angle_deg is not None and last_before is not None else None
                        rec = {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                            "frame_index": int(frame_idx),
                            "step": int(phase_state.get("step", step_counter.get("value", 0)) or 0),
                            "phase": str(phase_state.get("phase", "unknown")),
                            "raw_angle": raw_angle,
                            "final_angle": angle_deg,
                            "angle_deg": angle_deg,
                            "baseline_angle": baseline_angle,
                            "last_confirmed_angle_before": last_before,
                            "raw_delta_to_last": raw_delta_to_last,
                            "delta_from_baseline": delta,
                            "final_delta_to_baseline": delta,
                            "angle_source": angle_result.get("angle_source"),
                            "edge_length_px": angle_result.get("edge_length_px"),
                            "endpoints": angle_result.get("endpoints"),
                            "trigger_type": "",
                            "controller_action": phase_state.get("controller_action", ""),
                            "raw_image_path": angle_result.get("raw_image_path") or angle_result.get("image_path"),
                            "overlay_image_path": angle_result.get("overlay_image_path") or angle_result.get("overlay_path"),
                            "reason": angle_result.get("reason"),
                        }
                        if not angle_ok:
                            rec["trigger_type"] = "angle_invalid"
                            _append_record(rec)
                            self.log("[Step7实时角度] 本帧 YOLO-OBB 角度无效；继续下一帧。")
                            self._interruptible_pause(float(getattr(self.cfg, "rule_ab_angle_watch_interval_s", 0.5)), stop_event, phase_state=None)
                            continue

                        # 成功得到 YOLO-OBB 原始角度后，更新上一帧角度，仅用于日志诊断，不参与修复。
                        last_confirmed_angle_box["value"] = float(angle_deg)

                        if angle_ok and delta is not None:
                            # Step7 当前逻辑：只使用当前轮 Step1 baseline 作为基准；
                            # 只判断 delta >= min_delta。max_delta / 6° 不再参与停止判定，
                            # 也不再把 delta >= max_delta 单独标记为 overshoot。
                            if float(delta) >= float(min_delta):
                                rec["trigger_type"] = "target_reached"
                                result_box.update({
                                    "trigger_type": "target_reached",
                                    "stop_reason": "realtime_bmask_longest_edge_delta_ge_target_min_continue_measurement",
                                    "final_angle": float(angle_deg),
                                    "final_delta": float(delta),
                                })
                                _append_record(rec)
                                self.log(
                                    f"[Step7实时角度] 角度变化已达到阈值：final_angle={angle_deg:.6f}°, "
                                    f"baseline={baseline_angle}, delta={delta:.3f}° >= {min_delta:.3f}°；"
                                    "停止/暂停Stage34当前推动，不再使用6°上限/overshoot判定；"
                                    "不在Step7角度线程关闭激光，后续由完整测量主流程正常关激光，并继续完整测量。"
                                )
                                stop_event.set()
                                self._stop_rule_ab_stage34_if_possible(follower, reason="angle_delta_ge_target_min_no_laser_off")
                                break
                        _append_record(rec)
                    except Exception as e:
                        result_box["angle_error"] = str(e)
                        _append_record({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], "frame_index": int(frame_idx), "step": int(phase_state.get("step", 0) or 0), "phase": str(phase_state.get("phase", "unknown")), "angle_deg": None, "baseline_angle": baseline_angle, "delta_from_baseline": None, "angle_source": "bmask_longest_edge_failed", "edge_length_px": None, "endpoints": None, "trigger_type": "", "controller_action": phase_state.get("controller_action", ""), "raw_image_path": "", "overlay_image_path": "", "reason": f"angle_monitor_exception:{e}"})
                    self._interruptible_pause(float(getattr(self.cfg, "rule_ab_angle_watch_interval_s", 0.5)), stop_event, phase_state=None)
            finally:
                if old_override is None:
                    self.context.pop("bmask_longest_edge_overlay_dir_override", None)
                else:
                    self.context["bmask_longest_edge_overlay_dir_override"] = old_override

        def controller_thread() -> None:
            try:
                step_idx = 0
                while not stop_event.is_set() and not self.stop_requested and not self._is_midrun_recalibration_requested():
                    if max_duration_s > 0 and (time.time() - start_t) >= max_duration_s:
                        result_box["stop_reason"] = "realtime_max_duration_reached"
                        stop_event.set()
                        break
                    if callable(getattr(self, "rule_ab_config_sync_callback", None)):
                        try:
                            self.rule_ab_config_sync_callback()
                        except Exception as e:
                            self.log(f"[Step7实时] 运行中同步 GUI 参数失败，本步继续使用旧参数：{e}")
                    self.apply_runtime_rule_ab_params_to_follower(follower, reason=f"step7_realtime_controller_step_{step_idx + 1}")
                    step_idx += 1
                    step_counter["value"] = int(step_idx)
                    phase_state.update({"step": int(step_idx), "phase": "planning"})
                    route_points = self._load_or_build_step7_c_edge_route(follower)
                    self.log(f"[Step7实时控制] step={step_idx}: 执行一次 C边沿路线运动；route_points={len(route_points)}；pause保留但可被stop_event打断。")
                    ok, route_info = self._run_one_c_edge_route_cycle(follower=follower, step_idx=step_idx, route_points=route_points, stop_event=stop_event, phase_state=phase_state)
                    _append_record({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], "frame_index": "", "step": int(step_idx), "phase": str(phase_state.get("phase", "unknown")), "angle_deg": None, "baseline_angle": baseline_angle, "delta_from_baseline": None, "angle_source": "controller", "edge_length_px": None, "endpoints": None, "trigger_type": "", "controller_action": phase_state.get("controller_action", ""), "raw_image_path": "", "overlay_image_path": route_info.get("route_overlay_path") if isinstance(route_info, dict) else "", "reason": route_info.get("reason", "controller_step_ok") if isinstance(route_info, dict) else "controller_step_ok"})
                    if stop_event.is_set():
                        break
                    if not ok:
                        result_box["stop_reason"] = "route_or_ab_seg_failed_retry_next_frame"
                        self.log(f"[Step7实时控制] 路线运动/A-B分割失败，继续等待下一帧：{route_info}")
                    phase_state["phase"] = "loop_interval"
                    self._interruptible_pause(float(getattr(self.cfg, "rule_ab_loop_interval_s", 0.15)), stop_event, phase_state=phase_state)
            except Exception as e:
                result_box["controller_error"] = str(e)
                result_box["stop_reason"] = f"controller_thread_exception:{e}"
                self.log(f"[Step7实时控制] controller_thread 异常：{e}")
                self.log(traceback.format_exc())
                stop_event.set()
            finally:
                phase_state["phase"] = "stopped"
                controller_done.set()

        t_angle = threading.Thread(target=angle_monitor_thread, name="step7_angle_monitor_thread", daemon=True)
        t_controller = threading.Thread(target=controller_thread, name="step7_controller_thread", daemon=True)
        t_angle.start()
        t_controller.start()
        while not controller_done.is_set() and not self.stop_requested:
            if self._is_midrun_recalibration_requested():
                result_box["stop_reason"] = "midrun_recalibration_requested"
                stop_event.set()
                self._stop_rule_ab_stage34_if_possible(follower, reason="midrun_recalibration_requested")
                break
            if stop_event.is_set():
                break
            time.sleep(0.02)
        if self.stop_requested:
            result_box["stop_reason"] = "user_stop_requested"
            stop_event.set()
            self._stop_rule_ab_stage34_if_possible(follower, reason="user_stop_requested")
        elif self._is_midrun_recalibration_requested():
            result_box["stop_reason"] = "midrun_recalibration_requested"
            stop_event.set()
            self._stop_rule_ab_stage34_if_possible(follower, reason="midrun_recalibration_requested")
        t_controller.join(timeout=3.0)
        stop_event.set()
        t_angle.join(timeout=3.0)
        try:
            self._write_step7_realtime_records_csv(records, csv_path)
        except Exception as e:
            self.log(f"[Step7实时] 保存实时角度 CSV 失败：{e}")
        trigger_type = result_box.get("trigger_type")
        ok = bool(trigger_type == "target_reached")
        overshoot = False
        allow_continue = bool(ok)
        result = {"ok": bool(allow_continue), "step7_success_in_target_range": bool(ok), "allow_continue_measurement": bool(allow_continue), "overshoot": bool(overshoot), "reason": str(result_box.get("stop_reason") or "stopped"), "baseline_angle": baseline_angle, "final_angle": result_box.get("final_angle"), "final_delta": result_box.get("final_delta"), "target_min": min_delta, "target_max": None, "angle_source": "bmask_longest_edge_realtime_thread", "records": self._json_safe(records), "realtime_csv_path": str(csv_path), "realtime_dir": str(realtime_dir), "controller_error": result_box.get("controller_error"), "angle_error": result_box.get("angle_error")}
        self.context["last_rule_ab_result"] = result
        self.notify_update()
        self.log(f"[Step7实时] 结束：ok={ok}, overshoot={overshoot}, reason={result['reason']}, final_angle={result['final_angle']}, final_delta={result['final_delta']}, records={len(records)}, csv={csv_path}")
        return result

    def run_until_angle_delta(
        self,
        baseline_angle: Optional[float],
        min_delta_deg: Optional[float] = None,
        max_delta_deg: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Step7 主循环：当前版本只使用“当前帧 Bmask 最长边”作为角度来源。

        不再使用 KLT、Profile 动态灰度扫描、上一帧 hold_last 或人工指定边。
        """
        if bool(getattr(self.cfg, "rule_ab_realtime_step7_enable", True)):
            return self.run_realtime_until_angle_delta(
                baseline_angle=baseline_angle,
                min_delta_deg=min_delta_deg,
                max_delta_deg=max_delta_deg,
            )
        self.stop_event.clear()

        self.log("========== RuleAB：A推动B，Bmask最长边检测B角度（旧串行兼容模式） ==========")

        min_delta = float(self.cfg.rule_ab_angle_delta_min_deg if min_delta_deg is None else min_delta_deg)
        max_delta = None if max_delta_deg is None else float(max_delta_deg)

        follower = self.ensure_rule_ab_follower()
        self._ensure_rule_ab_feature_tracker_installed(follower, self._get_loaded_calibration_state())
        self.apply_runtime_rule_ab_params_to_follower(follower, reason="step7_start")
        try:
            st = getattr(follower, "stage", None)
            self.log(
                f"[RuleAB] Stage34状态：enable_stage={getattr(follower.cfg, 'enable_stage', None)}, "
                f"stage_is_none={st is None}, x_channel={getattr(follower.cfg, 'stage_x_channel', None)}, "
                f"y_channel={getattr(follower.cfg, 'stage_y_channel', None)}, "
                f"step_x={getattr(follower.cfg, 'stage_step_x', None)}, step_y={getattr(follower.cfg, 'stage_step_y', None)}"
            )
        except Exception:
            pass

        records: List[Dict[str, Any]] = []
        reached = False
        allow_continue_measurement = False
        stop_reason = "not_started"
        final_angle: Optional[float] = None
        final_delta: Optional[float] = None

        # Step7 每次进入都重新用当前 A center 找最近 route index 作为起点；
        # 后续帧只按 current_route_index + direction 推进，不再每帧跳到最近线段。
        self.context.pop("step7_c_edge_route_state", None)

        # 串行兼容模式中上一帧角度也仅用于诊断；不做角度修复。
        last_confirmed_angle: Optional[float] = float(baseline_angle) if baseline_angle is not None else None

        step_idx = 0
        while True:
            if self.stop_event.is_set():
                stop_reason = "step7_runner_stop_requested"
                self._stop_rule_ab_stage34_if_possible(follower, reason="step7_runner_stop_requested")
                break
            if self.stop_requested:
                stop_reason = "user_stop_requested"
                break
            if self._is_midrun_recalibration_requested():
                stop_reason = "midrun_recalibration_requested"
                self._stop_rule_ab_stage34_if_possible(follower, reason="midrun_recalibration_requested")
                break

            if callable(getattr(self, "rule_ab_config_sync_callback", None)):
                try:
                    self.rule_ab_config_sync_callback()
                except Exception as e:
                    self.log(f"[RuleAB] 运行中同步 GUI 参数失败，本步继续使用旧参数：{e}")

            if min_delta_deg is None:
                min_delta = float(getattr(self.cfg, "rule_ab_angle_delta_min_deg", min_delta))
            if max_delta_deg is None:
                max_delta = None

            follower = self.ensure_rule_ab_follower()
            self._ensure_rule_ab_feature_tracker_installed(follower, self._get_loaded_calibration_state())
            self.apply_runtime_rule_ab_params_to_follower(follower, reason=f"step7_step_{step_idx + 1}")

            step_idx += 1
            use_route = bool(getattr(self.cfg, "rule_ab_use_c_edge_route", True))
            route_info: Dict[str, Any] = {"route_enabled": use_route}

            if use_route:
                route_points = self._load_or_build_step7_c_edge_route(follower)
                self.log(
                    f"[RuleAB] step={step_idx}: 执行一次 C边沿路线运动；route_points={len(route_points)}；"
                    f"目标角度范围=[{min_delta}, {max_delta}]；角度来源=当前帧Bmask最长边。"
                )
                try:
                    ok, route_info = self._run_one_c_edge_route_cycle(follower, step_idx, route_points)
                except Exception as e:
                    ok = False
                    route_info = {
                        "route_enabled": True,
                        "reason": "route_cycle_exception_retry_next_frame",
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }
                if not ok:
                    self.log(f"[RuleAB-Route] 路线运动/A-B分割失败：{route_info}；不停止完整测量，重新取下一帧继续。")
                    records.append({
                        "step": step_idx,
                        "rule_ab_ok": False,
                        "angle_deg": None,
                        "angle_delta_from_baseline": None,
                        "angle_source": "skip_angle_after_route_ab_seg_fail",
                        "angle_result": {"ok": False, "reason": "skip_angle_after_route_ab_seg_fail"},
                        "c_edge_route": self._json_safe(route_info),
                        "retry_next_frame": True,
                    })
                    stop_reason = "route_or_ab_seg_failed_retry_next_frame"
                    time.sleep(float(self.cfg.rule_ab_loop_interval_s))
                    continue
            else:
                self.log(
                    f"[RuleAB] step={step_idx}: 执行一次 rule_ab.run_one_cycle()；"
                    f"目标角度范围=[{min_delta}, {max_delta}]；角度来源=当前帧Bmask最长边。"
                )
                try:
                    ok = follower.run_one_cycle()
                except Exception as e:
                    ok = False
                    route_info = {
                        "route_enabled": False,
                        "reason": "run_one_cycle_exception_retry_next_frame",
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }
                if not ok:
                    self.log(f"[RuleAB] run_one_cycle/A-B分割失败：{route_info}；重新取下一帧继续。")
                    records.append({
                        "step": step_idx,
                        "rule_ab_ok": False,
                        "angle_deg": None,
                        "angle_delta_from_baseline": None,
                        "angle_source": "skip_angle_after_run_one_cycle_fail",
                        "angle_result": {"ok": False, "reason": "skip_angle_after_run_one_cycle_fail"},
                        "c_edge_route": self._json_safe(route_info),
                        "retry_next_frame": True,
                    })
                    stop_reason = "run_one_cycle_failed_retry_next_frame"
                    time.sleep(float(self.cfg.rule_ab_loop_interval_s))
                    continue

            # 每一步运动后，重新取当前帧 Bmask，并直接使用最大外轮廓最长边计算角度；不使用上一帧 hold_last。
            angle_result = self.detect_step7_yolo_obb_angle_once(
                follower=follower,
                label=f"rule_ab_bmask_longest_edge_step_{step_idx}",
                baseline_angle=baseline_angle,
                allow_fail=True,
                allow_close_from_relocation=False,
            )

            angle_ok = bool(angle_result.get("ok", False) and angle_result.get("angle_deg") is not None)
            raw_angle = float(angle_result["angle_deg"]) if angle_ok else None
            current_angle = raw_angle
            angle_source = str(angle_result.get("angle_source", "yolo_obb_failed"))
            allow_close_by_angle = bool(angle_result.get("allow_close", False))
            final_angle = current_angle
            final_delta = None
            if current_angle is not None and baseline_angle is not None:
                final_delta = self.angle_diff_deg(float(current_angle), float(baseline_angle))
            if angle_ok:
                last_confirmed_angle = float(current_angle)

            record = {
                "step": step_idx,
                "rule_ab_ok": bool(ok),
                "raw_angle": raw_angle,
                "final_angle": current_angle,
                "angle_deg": current_angle,
                "angle_delta_from_baseline": final_delta,
                "final_delta_to_baseline": final_delta,
                "baseline_angle": baseline_angle,
                "last_confirmed_angle_before": last_confirmed_angle,
                "raw_delta_to_last": self.angle_diff_deg(float(raw_angle), float(last_confirmed_angle)) if raw_angle is not None and last_confirmed_angle is not None else None,
                "angle_source": angle_source,
                "edge_length_px": angle_result.get("edge_length_px"),
                "endpoints": angle_result.get("endpoints"),
                "bmask_longest_edge_allow_close": allow_close_by_angle,
                "angle_result": self._json_safe(angle_result),
                "c_edge_route": self._json_safe(route_info),
                "target_min": min_delta,
                "target_max": max_delta,
            }
            records.append(record)

            self.log(
                f"[RuleAB-Bmask最长边] step={step_idx}, angle={current_angle}, baseline={baseline_angle}, "
                f"delta={final_delta}, source={angle_source}, length={angle_result.get('edge_length_px')}, "
                f"allow_close={allow_close_by_angle}"
            )

            if not angle_ok:
                stop_reason = "bmask_longest_edge_angle_failed_retry_next_frame_no_close"
                self.log("[RuleAB-Bmask最长边] 本帧 Bmask 最长边角度无效，不使用上一帧角度；继续下一步推动/重新取帧。")
                time.sleep(float(self.cfg.rule_ab_loop_interval_s))
                continue

            # 串行兼容模式同样只判断 delta >= min_delta；不再使用 max_delta / 6° 上限。
            close_hit = bool(
                allow_close_by_angle
                and final_delta is not None
                and float(final_delta) >= float(min_delta)
            )
            if close_hit:
                reached = True
                allow_continue_measurement = True
                stop_reason = "bmask_longest_edge_delta_ge_target_min_continue_measurement"
                self.log(
                    f"[RuleAB-Bmask最长边] 角度变化已达到阈值：delta={float(final_delta):.3f}° >= "
                    f"{float(min_delta):.3f}°；停止本次 Stage34 推动；不再使用6°上限/overshoot判定；"
                    "不在Step7分支直接关闭激光，后续由完整测量主流程正常关激光并继续完整测量。"
                )
                break

            if final_delta is not None and float(final_delta) < float(min_delta):
                stop_reason = "bmask_longest_edge_delta_below_target_continue"
                self.log(
                    f"[RuleAB-Bmask最长边] 当前角度变化 {float(final_delta):.3f}° 未达到关闭阈值 "
                    f"{float(min_delta):.3f}°；继续下一步推动。"
                )
            else:
                stop_reason = "bmask_longest_edge_angle_valid_continue"
                self.log("[RuleAB-Bmask最长边] 本帧角度有效但未满足关闭条件；继续下一步。")

            time.sleep(float(self.cfg.rule_ab_loop_interval_s))

        result = {
            "ok": bool(reached or allow_continue_measurement),
            "step7_success_in_target_range": bool(reached),
            "allow_continue_measurement": bool(allow_continue_measurement),
            "reason": stop_reason,
            "baseline_angle": baseline_angle,
            "final_angle": final_angle,
            "final_delta": final_delta,
            "target_min": min_delta,
            "target_max": None,
            "angle_source": "bmask_longest_edge",
            "records": records,
        }
        self.context["last_rule_ab_result"] = result
        self.notify_update()
        return result

    # --------------------------------------------------------
    # Step7：沿 C 四边形边沿预生成路线运动
    # --------------------------------------------------------

    @staticmethod

    def stop(self, reason: str = "Step7Runner.stop") -> bool:
        self.stop_event.set()
        follower = getattr(self.workflow, "rule_ab_follower", None)
        if follower is None:
            return False
        return bool(self.workflow._stop_rule_ab_stage34_if_possible(follower, reason=reason))

    def stop_stage34(self, reason: str = "Step7Runner.stop_stage34") -> bool:
        return self.stop(reason=reason)

    def interruptible_pause(self, seconds: float, stop_event=None, phase_state=None) -> bool:
        return self.workflow._interruptible_pause(
            seconds,
            stop_event=stop_event if stop_event is not None else self.stop_event,
            phase_state=phase_state,
        )

    def execute_rule_ab_action_with_watchable_pause(self, follower: Any, action_id: int, stop_event=None, phase_state=None) -> Dict[str, Any]:
        return self.workflow._execute_rule_ab_action_with_watchable_pause(
            follower=follower,
            action_id=action_id,
            stop_event=stop_event if stop_event is not None else self.stop_event,
            phase_state=phase_state,
        )
