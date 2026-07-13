from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from modified_minimal_device_buttons import MeasurementWorkflow


class Step9Runner:
    """Owns Step9 color-region-center alignment logic while sharing workflow state."""

    def __init__(self, workflow: "MeasurementWorkflow"):
        self.workflow = workflow

    def __getattr__(self, name: str):
        return getattr(self.workflow, name)

    def __setattr__(self, name: str, value: Any):
        if name == "workflow" or "workflow" not in self.__dict__:
            object.__setattr__(self, name, value)
        else:
            setattr(self.workflow, name, value)

    def begin_run_session(self) -> Path:
        """
        每次点击“单独测试Step9对齐”时新建一个独立文件夹。

        保存位置：
            <save_root>/step9_color_center_align/run_YYYYMMDD_HHMMSS_mmm/

        该文件夹用于保存：
            1. 每轮截图；
            2. 每轮颜色区域 mask；
            3. 每轮可选保存的 A/B mask；
            4. 每轮 overlay 检查图；
            5. step9_history.csv；
            6. step9_meta.json。
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        run_dir = self.output_root / "step9_color_center_align" / f"run_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)

        for sub in ("frames", "masks", "overlays", "color_initial", "ab_masks", "c_masks"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)

        self.step9_current_run_dir = run_dir
        self.step9_history_csv_path = run_dir / "step9_history.csv"

        meta = {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "capture_area": list(self.cfg.capture_area),
            "target_x_px": float(getattr(self.cfg, "rule_ac_target_x_px", -1.0)),
            "target_y_px": float(getattr(self.cfg, "rule_ac_target_y_px", -1.0)),
            "tolerance_px": float(getattr(self.cfg, "rule_ac_center_tolerance_px", 5.0)),
            "stage_step_size": int(getattr(self.cfg, "rule_ac_stage_step_size", 20)),
            "stage12_velocity": int(getattr(self.cfg, "rule_ac_stage12_velocity", 10)),
            "stage12_acceleration": int(getattr(self.cfg, "rule_ac_stage12_acceleration", 10)),
            "stage12_max_voltage": int(getattr(self.cfg, "rule_ac_stage12_max_voltage", 50)),
            "enable_stage": bool(getattr(self.cfg, "rule_ac_enable_stage", False)),
            "step9_detection_mode": "color_region_center",
            "color_mode": str(getattr(self.cfg, "rule_ac_color_mode", "include")),
            "color_hsv": [
                int(getattr(self.cfg, "rule_ac_color_h", -1)),
                int(getattr(self.cfg, "rule_ac_color_s", -1)),
                int(getattr(self.cfg, "rule_ac_color_v", -1)),
            ],
            "color_tol_hsv": [
                int(getattr(self.cfg, "rule_ac_color_h_tol", 12)),
                int(getattr(self.cfg, "rule_ac_color_s_tol", 70)),
                int(getattr(self.cfg, "rule_ac_color_v_tol", 70)),
            ],
            "color_min_area_px": int(getattr(self.cfg, "rule_ac_color_min_area_px", 50)),
        }
        try:
            with (run_dir / "step9_meta.json").open("w", encoding="utf-8") as f:
                json.dump(self._json_safe(meta), f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.log(f"[Step9-颜色中心] 保存 Step9 meta 失败：{e}")

        self.log(f"[Step9-颜色中心] 新建本次 Step9 保存文件夹：{run_dir}")
        return run_dir

    def get_output_dir(self, subdir: Optional[str] = None) -> Path:
        """
        获取 Step9 当前运行文件夹；如果还没有创建，则自动创建。
        """
        if self.step9_current_run_dir is None:
            self.begin_run_session()

        assert self.step9_current_run_dir is not None
        if subdir:
            p = self.step9_current_run_dir / str(subdir)
            p.mkdir(parents=True, exist_ok=True)
            return p
        return self.step9_current_run_dir

    def append_history_csv(self, row: Dict[str, Any]):
        """
        追加 Step9 单独测试/完整流程的每轮记录。
        """
        if self.step9_history_csv_path is None:
            self.get_output_dir()

        assert self.step9_history_csv_path is not None
        fieldnames = [
            "cycle",
            "time",
            "ok",
            "aligned",
            "reason",
            "center_x",
            "center_y",
            "target_x",
            "target_y",
            "dx",
            "dy",
            "tolerance_px",
            "color_area_px",
            "a_mask_area_px",
            "b_mask_area_px",
            "step9_c_source",
            "step9_c_area_px",
            "step9_c_center_x",
            "step9_c_center_y",
            "step9_c_dirty_before_update",
            "step9_c_update_ok",
            "step9_c_update_reason",
            "action_code",
            "action_name",
            "moved",
            "dry_run",
            "ch1_distance",
            "ch2_distance",
            "overlay_path",
            "frame_path",
            "b_mask_path",
            "color_mask_path",
            "a_mask_path",
            "b_mask_path",
            "c_mask_path",
            "c_raw_mask_path",
        ]
        file_exists = self.step9_history_csv_path.exists()
        with self.step9_history_csv_path.open("a", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    def stop_stage12(self, reason: str = "legacy_stop_step9_stage12"):
        """
        外部按钮调用：请求停止 Step9，并尝试立即停止 1/2 通道运动。
        """
        self.step9_stop_requested = True
        self.log(f"[Step9-颜色中心] 已请求停止 1/2 通道运动；reason={reason}")
        try:
            stage_obj = None
            if self.stage12_device is not None:
                stage_obj = self.stage12_device
            elif self.rule_ab_follower is not None:
                stage_obj = getattr(self.rule_ab_follower, "stage", None)
            self._stop_stage12_if_possible(stage_obj)
            self.log("[Step9-颜色中心] 已尝试发送 1/2 通道 stop 命令")
        except Exception as e:
            self.log(f"[Step9-颜色中心] 停止 1/2 通道失败：{e}")

    def run_until_aligned(self) -> Dict[str, Any]:
        """
        Step9 新逻辑：颜色检测区域中心对齐。

        本版本支持：
            1. 每次单独测试 Step9 生成独立保存文件夹；
            2. 每轮保存原始截图、颜色区域 mask、A/B mask、overlay 和 CSV；
            3. 运动过程中可修改 RuleAC/Step9 输入，下一轮立即读取新值；
            4. 停止按钮可请求停止 1/2 通道运动。
        """
        self.log("========== Step9：HSV颜色区域中心对齐，必要时移动1/2通道 ==========")
        self.step9_stop_requested = False
        # 当前 Step9 不再在循环中主动更新动态 C；该标志仅保留兼容旧字段。
        self.step9_c_mask_dirty = False

        # 如果是完整测量流程中直接调用 Step9，可能尚未创建 Step9 run 文件夹。
        self.get_output_dir()

        records: List[Dict[str, Any]] = []
        final_ok = False
        last_record: Dict[str, Any] = {}

        cycle_idx = 1
        while True:
            # 运行过程中同步 GUI 最新 RuleAC 输入。
            # 单独测试时 GUI 会设置 self.step9_config_sync_callback；
            # 完整测量流程中没有该回调时则使用当前 self.cfg。
            if callable(getattr(self, "step9_config_sync_callback", None)):
                try:
                    self.step9_config_sync_callback()
                except Exception as e:
                    self.log(f"[Step9-颜色中心] 运行中同步 GUI 参数失败，本轮继续使用旧参数：{e}")

            target_x = float(getattr(self.cfg, "rule_ac_target_x_px", -1.0))
            target_y = float(getattr(self.cfg, "rule_ac_target_y_px", -1.0))
            tol = float(getattr(self.cfg, "rule_ac_center_tolerance_px", 5.0))
            loop_interval_s = float(getattr(self.cfg, "rule_ac_loop_interval_s", 0.10))

            if target_x < 0 or target_y < 0:
                self.log(
                    "[Step9-颜色中心] 未设置目标点 rule_ac_target_x/y。"
                    "请先在 GUI 中点击“选定Step9目标”，或在目标x/y输入框中填写截图区域内坐标。"
                )
                result = {
                    "ok": False,
                    "reason": "target_not_set",
                    "target": None,
                    "records": records,
                    "run_dir": str(self.step9_current_run_dir) if self.step9_current_run_dir else "",
                }
                self.context["last_rule_ac_result"] = result
                self.notify_update()
                return result

            # Step9 exits only in these cases:
            #   1) 检测区域中心与选定目标点在容差内重合；
            #   2) 用户点击“停止1/2轴运动”或停止测量；
            #   3) 检测失败。

            if self.stop_requested or self.step9_stop_requested or self._is_midrun_recalibration_requested():
                if self._is_midrun_recalibration_requested():
                    self.log("[Step9-颜色中心] 收到中途重标定请求，停止 1/2 通道并退出 Step9。")
                else:
                    self.log("[Step9-颜色中心] 收到停止请求，停止 1/2 通道并退出 Step9。")
                try:
                    stage_obj, _ = self._get_stage12_device() if self.stage12_device is not None else (None, False)
                    self._stop_stage12_if_possible(stage_obj)
                except Exception:
                    pass
                break

            det = self._capture_step9_color_center_once()
            if not det.get("ok", False):
                action_code = 0
                save_info = self._save_step9_color_overlay(
                    det.get("image_rgb"),
                    det.get("color_mask"),
                    None,
                    (target_x, target_y),
                    cycle_idx,
                    action_code,
                    a_mask=det.get("a_mask"),
                    b_mask=det.get("b_mask"),
                    c_mask=det.get("c_mask"),
                    c_raw_mask=det.get("c_raw_mask"),
                    c_source=str(det.get("step9_c_source", "")),
                    force_save=True,
                )
                last_record = {
                    "cycle": cycle_idx,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "ok": False,
                    "aligned": False,
                    "reason": det.get("reason"),
                    "center_x": "",
                    "center_y": "",
                    "target_x": target_x,
                    "target_y": target_y,
                    "dx": "",
                    "dy": "",
                    "tolerance_px": tol,
                    "color_area_px": det.get("color_area_px"),
                    "a_mask_area_px": det.get("a_mask_area_px"),
                    "b_mask_area_px": det.get("b_mask_area_px"),
                    "step9_c_source": det.get("step9_c_source"),
                    "step9_c_area_px": det.get("step9_c_area_px"),
                    "step9_c_center_x": det.get("step9_c_center_x"),
                    "step9_c_center_y": det.get("step9_c_center_y"),
                    "step9_c_dirty_before_update": det.get("step9_c_dirty_before_update"),
                    "step9_c_update_ok": det.get("step9_c_update_ok"),
                    "step9_c_update_reason": det.get("step9_c_update_reason"),
                    "action_code": action_code,
                    "action_name": "",
                    "move_info": {"moved": False, "reason": det.get("reason")},
                    **save_info,
                }
                records.append(last_record)
                self.append_history_csv({
                    **last_record,
                    "moved": False,
                    "dry_run": "",
                    "ch1_distance": "",
                    "ch2_distance": "",
                })
                self.log(f"[Step9-颜色中心] cycle={cycle_idx}, 检测失败：{last_record}")
                break

            cx, cy = det["color_center"]
            dx = target_x - float(cx)
            dy = target_y - float(cy)
            abs_dx = abs(dx)
            abs_dy = abs(dy)

            if abs_dx <= tol and abs_dy <= tol:
                action_code = 0
                final_ok = True
                move_info = {"moved": False, "reason": "center_aligned"}
                save_info = self._save_step9_color_overlay(
                    det.get("image_rgb"),
                    det.get("color_mask"),
                    (cx, cy),
                    (target_x, target_y),
                    cycle_idx,
                    action_code,
                    a_mask=det.get("a_mask"),
                    b_mask=det.get("b_mask"),
                    c_mask=det.get("c_mask"),
                    c_raw_mask=det.get("c_raw_mask"),
                    c_source=str(det.get("step9_c_source", "")),
                    force_save=True,
                )
                last_record = {
                    "cycle": cycle_idx,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "ok": True,
                    "aligned": True,
                    "reason": "center_aligned",
                    "center_x": cx,
                    "center_y": cy,
                    "target_x": target_x,
                    "target_y": target_y,
                    "dx": dx,
                    "dy": dy,
                    "tolerance_px": tol,
                    "color_area_px": det.get("color_area_px"),
                    "a_mask_area_px": det.get("a_mask_area_px"),
                    "b_mask_area_px": det.get("b_mask_area_px"),
                    "step9_c_source": det.get("step9_c_source"),
                    "step9_c_area_px": det.get("step9_c_area_px"),
                    "step9_c_center_x": det.get("step9_c_center_x"),
                    "step9_c_center_y": det.get("step9_c_center_y"),
                    "step9_c_dirty_before_update": det.get("step9_c_dirty_before_update"),
                    "step9_c_update_ok": det.get("step9_c_update_ok"),
                    "step9_c_update_reason": det.get("step9_c_update_reason"),
                    "action_code": action_code,
                    "action_name": "",
                    "move_info": move_info,
                    **save_info,
                }
                records.append(last_record)
                self.append_history_csv({
                    **last_record,
                    "moved": False,
                    "dry_run": "",
                    "ch1_distance": "",
                    "ch2_distance": "",
                })
                self.log(
                    f"[Step9-颜色中心] cycle={cycle_idx}, 已对齐："
                    f"center=({cx:.2f},{cy:.2f}), target=({target_x:.2f},{target_y:.2f}), "
                    f"dx={dx:.2f}, dy={dy:.2f}, tol={tol}"
                )
                break

            action_code = self.direction_from_error(dx=dx, dy=dy, tol=tol)
            save_info = self._save_step9_color_overlay(
                det.get("image_rgb"),
                det.get("color_mask"),
                (cx, cy),
                (target_x, target_y),
                cycle_idx,
                action_code,
                a_mask=det.get("a_mask"),
                b_mask=det.get("b_mask"),
                c_mask=det.get("c_mask"),
                c_raw_mask=det.get("c_raw_mask"),
                c_source=str(det.get("step9_c_source", "")),
            )

            if self.step9_stop_requested:
                self.log("[Step9-颜色中心] 运动前收到停止请求，跳过本次 1/2 通道动作。")
                move_info = {"moved": False, "reason": "step9_stop_requested_before_move"}
            else:
                # _execute_stage12_diagonal_action 会在执行前读取当前 self.cfg，
                # 所以 step_size / velocity / acceleration / enable_stage 的 GUI 修改会在本轮生效。
                # 当前 Step9 不再在 CH1/CH2 运动后标记/更新动态 C。
                move_info = self.execute_stage12_action(action_code)

            action_name = move_info.get("action_name")
            last_record = {
                "cycle": cycle_idx,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ok": True,
                "aligned": False,
                "reason": "moving",
                "center_x": cx,
                "center_y": cy,
                "target_x": target_x,
                "target_y": target_y,
                "dx": dx,
                "dy": dy,
                "tolerance_px": tol,
                "color_area_px": det.get("color_area_px"),
                "a_mask_area_px": det.get("a_mask_area_px"),
                "b_mask_area_px": det.get("b_mask_area_px"),
                "step9_c_source": det.get("step9_c_source"),
                "step9_c_area_px": det.get("step9_c_area_px"),
                "step9_c_center_x": det.get("step9_c_center_x"),
                "step9_c_center_y": det.get("step9_c_center_y"),
                "step9_c_dirty_before_update": det.get("step9_c_dirty_before_update"),
                "step9_c_update_ok": det.get("step9_c_update_ok"),
                "step9_c_update_reason": det.get("step9_c_update_reason"),
                "action_code": action_code,
                "action_name": action_name,
                "move_info": move_info,
                **save_info,
            }
            records.append(last_record)
            self.append_history_csv({
                **last_record,
                "moved": move_info.get("moved"),
                "dry_run": move_info.get("dry_run"),
                "ch1_distance": move_info.get("ch1_distance"),
                "ch2_distance": move_info.get("ch2_distance"),
            })

            self.log(
                f"[Step9-颜色中心] cycle={cycle_idx}, "
                f"center=({cx:.2f},{cy:.2f}), target=({target_x:.2f},{target_y:.2f}), "
                f"dx={dx:.2f}, dy={dy:.2f}, tol={tol}, "
                f"color_area={det.get('color_area_px')}, "
                f"step={getattr(self.cfg, 'rule_ac_stage_step_size', None)}, "
                f"enable_stage={getattr(self.cfg, 'rule_ac_enable_stage', None)}, "
                f"action={action_code}({action_name}), moved={move_info.get('moved')}"
            )

            cycle_idx += 1
            time.sleep(loop_interval_s)

        result = {
            "ok": bool(final_ok),
            "reason": ("aligned" if final_ok else ("midrun_recalibration_requested" if self._is_midrun_recalibration_requested() else "not_aligned_or_stopped_or_detect_failed")),
            "target": (
                float(getattr(self.cfg, "rule_ac_target_x_px", -1.0)),
                float(getattr(self.cfg, "rule_ac_target_y_px", -1.0)),
            ),
            "tolerance_px": float(getattr(self.cfg, "rule_ac_center_tolerance_px", 5.0)),
            "history_len": len(records),
            "last_row": last_record,
            "records": records,
            "csv_path": str(self.step9_history_csv_path) if self.step9_history_csv_path else "",
            "run_dir": str(self.step9_current_run_dir) if self.step9_current_run_dir else "",
        }
        self.context["last_rule_ac_result"] = result
        self.notify_update()
        return result

    # --------------------------------------------------------
    # LabVIEW 光谱与保存
    # --------------------------------------------------------

    @staticmethod

    def run_until_threshold(self) -> Dict[str, Any]:
        return self.run_until_aligned()

    def capture_color_center_once(self) -> Dict[str, Any]:
        return self.workflow._capture_step9_color_center_once()

    def save_color_overlay(self, *args, **kwargs) -> Dict[str, Any]:
        return self.workflow._save_step9_color_overlay(*args, **kwargs)

    def direction_from_error(self, dx: float, dy: float, tol: float) -> int:
        return self.workflow._step9_direction_from_error(dx=dx, dy=dy, tol=tol)

    def execute_stage12_action(self, action_code: int) -> Dict[str, Any]:
        return self.workflow._execute_stage12_diagonal_action(action_code)

    def stop(self, reason: str = "Step9Runner.stop"):
        return self.stop_stage12(reason=reason)
