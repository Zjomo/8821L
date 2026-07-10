"""
自动补焦闭环控制器。

从 measurement_autofocus_shg_closed_loop.py 提取：
  - evaluate_autofocus_trigger()
  - step45_focus_check_and_autofocus()
  - run_closed_loop_autofocus()
  - update_shg_reference_and_ratio() (SHG 补焦部分)
"""

from __future__ import annotations

import time
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .config import AutofocusConfig
from .metrics import FocusMetricsCalculator
from .scorer import FocusScorer
from .search import create_search, FocusSearchStopped
from .z_axis import ZAxisController

logger = logging.getLogger(__name__)

LogCallback = Optional[Callable[[str], None]]


class AutofocusController:
    """
    自动补焦闭环控制器。

    职责：
      1. 判断是否触发补焦（基于 FocusScore_ratio 和可选的 SHG_ratio）；
      2. 执行闭环 Z 轴搜索（试探 → 搜索 → 回退）；
      3. 提供 step45 一站式接口（截图 → 判断 → 补焦 → 最终截图）。

    用法：
        controller = AutofocusController(cfg, scorer, metrics_calc, z_axis)
        controller.on_log = print

        # 建立参考
        controller.build_reference(output_root=some_path)

        # 每轮调用一次
        result = controller.check_and_autofocus(cycle_index=1, save_dir=some_path)
    """

    def __init__(
        self,
        cfg: AutofocusConfig,
        scorer: FocusScorer,
        metrics_calc: FocusMetricsCalculator,
        z_axis: ZAxisController,
    ):
        self.cfg = cfg
        self.scorer = scorer
        self.metrics_calc = metrics_calc
        self.z_axis = z_axis

        self.on_log: LogCallback = None
        self.should_stop: Optional[Callable[[], bool]] = None

        # 补焦状态
        self.autofocus_event_counter: int = 0
        self.consecutive_focus_low_count: int = 0

        # SHG 补焦状态（可选，由外部更新）
        self.shg_reference_values: List[float] = []
        self.shg_ref: Optional[float] = None
        self.last_shg_ratio: Optional[float] = None
        self.consecutive_shg_low_count: int = 0

    def _log(self, msg: str):
        if self.on_log:
            self.on_log(msg)
        else:
            logger.info(msg)

    # ================================================================
    # 补焦触发判断
    # ================================================================

    def evaluate_trigger(
        self,
        focus_score_ratio: Optional[float],
    ) -> Tuple[bool, List[str]]:
        """
        判断是否需要触发自动补焦。

        触发条件：
          1. FocusScore_ratio 连续 N 轮低于阈值；
          2. SHG_ratio 单轮硬阈值或连续 N 轮低于阈值（可选）。

        参数
        ----------
        focus_score_ratio : float or None

        返回
        -------
        (need_autofocus, reasons)
        """
        reasons: List[str] = []

        if focus_score_ratio is not None:
            if focus_score_ratio < float(self.cfg.autofocus_focus_trigger_ratio):
                self.consecutive_focus_low_count += 1
            else:
                self.consecutive_focus_low_count = 0

        if self.consecutive_focus_low_count >= int(
            self.cfg.autofocus_focus_trigger_count
        ):
            reasons.append(
                f"连续{self.consecutive_focus_low_count}轮 FocusScore_ratio "
                f"< {self.cfg.autofocus_focus_trigger_ratio}"
            )

        if self.last_shg_ratio is not None:
            if self.last_shg_ratio < float(self.cfg.autofocus_shg_hard_ratio):
                reasons.append(
                    f"单轮 SHG_ratio={self.last_shg_ratio:.4f} "
                    f"< {self.cfg.autofocus_shg_hard_ratio}"
                )
            elif self.consecutive_shg_low_count >= int(
                self.cfg.autofocus_shg_trigger_count
            ):
                reasons.append(
                    f"连续{self.consecutive_shg_low_count}轮 SHG_ratio "
                    f"< {self.cfg.autofocus_shg_trigger_ratio}"
                )

        return bool(reasons), reasons

    # ================================================================
    # 闭环补焦搜索
    # ================================================================

    def run_closed_loop(
        self,
        initial_focus_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        开环 Z 轴 + 图像反馈的闭环补焦。

        根据 cfg.z_search_strategy 选择搜索策略：
          - hill_climb    : 试探方向 + 沿方向递进 + 回退最佳位置（原有算法，新增自适应步长）
          - full_sweep    : 在 [-range, range] 全扫描并回到最佳位置
          - curve_fit     : 等间距采样 + 抛物线拟合峰值 + 局部微调
          - golden_section: 粗扫 bracket + 黄金分割细化

        参数
        ----------
        initial_focus_score : float, optional
            初始 FocusScore_ratio。为 None 时自动采集。

        返回
        -------
        {
            "ok": bool,
            "reason": str,
            "initial_score": float,
            "best_score": float,
            "best_relative_z_steps": int,
            "final_relative_z_steps": int,
            "history": [...]
        }
        """
        self.autofocus_event_counter += 1
        event_id = self.autofocus_event_counter
        target = float(self.cfg.autofocus_stop_ratio)
        min_improve = max(0.0, float(self.cfg.z_min_improve_ratio))
        max_iter = max(1, int(self.cfg.z_max_iter))
        max_total_steps = max(1, int(self.cfg.z_max_total_steps))
        patience = max(1, int(self.cfg.z_patience))
        strategy = str(self.cfg.z_search_strategy or "hill_climb")

        self._log(
            f"========== 闭环补焦 #{event_id} 开始: strategy={strategy}, target={target} =========="
        )

        self.z_axis.connect()

        initial_score = FocusMetricsCalculator.safe_float(initial_focus_score)

        def move_fn(delta: int) -> None:
            self.z_axis.move_relative(delta)

        def measure_fn(_phase: str, _iteration: int) -> Tuple[Optional[float], Optional[Dict[str, float]]]:
            live = self.metrics_calc.capture_live()
            score, comp = self.scorer.score_ratio(live.get("roi_metrics"))
            return score, comp

        search = create_search(
            strategy, self.cfg, move_fn, measure_fn, self._log, self.should_stop
        )
        try:
            result = search.search(
                initial_score=initial_score,
                target=target,
                max_total_steps=max_total_steps,
                max_iter=max_iter,
                patience=patience,
                min_improve=min_improve,
            )
        except FocusSearchStopped:
            self._log(f"========== 闭环补焦 #{event_id} 被用户停止 ==========")
            return {
                "ok": False,
                "reason": "stopped_by_user",
                "initial_score": FocusMetricsCalculator.safe_float(initial_score),
                "best_score": FocusMetricsCalculator.safe_float(initial_score),
                "best_relative_z_steps": 0,
                "final_relative_z_steps": 0,
                "history": search.history,
            }

        ok = bool(result.get("ok"))
        best_score = result.get("best_score")
        best_pos = result.get("best_relative_z_steps")
        self._log(
            f"========== 闭环补焦 #{event_id} 结束: ok={ok}, "
            f"best_score={best_score:.4f}, best_z={best_pos} =========="
        )
        return result

    # ================================================================
    # 一站式接口：Step 4.5 截图 → 判断 → 补焦 → 最终截图
    # ================================================================

    def check_and_autofocus(
        self,
        cycle_index: int = 0,
        save_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        一站式自动补焦流程，适合嵌入循环测量的 Step 4.5。

        流程：
          1. 截图并计算聚焦指标；
          2. 计算 FocusScore_ratio；
          3. 判断是否触发补焦；
          4. 如触发则执行闭环 Z 轴搜索；
          5. 补焦结束后重新截图，返回最终指标。

        参数
        ----------
        cycle_index : int
            当前轮次编号，用于文件名。
        save_dir : str | Path, optional
            截图保存目录。为 None 时不保存。

        返回
        -------
        focus_metrics dict
        """
        self._log("========== 自动补焦检查 ==========")

        # 1. 截图
        if save_dir is not None:
            focus_metrics = self.metrics_calc.capture_and_save(
                cycle_index=cycle_index, save_dir=save_dir, on_log=self._log
            )
        else:
            live = self.metrics_calc.capture_live()
            focus_metrics = live

        # 2. 计算评分
        roi_metrics = focus_metrics.get("roi_metrics") if isinstance(focus_metrics, dict) else None
        focus_score, comp_ratios = self.scorer.score_ratio(roi_metrics)

        if isinstance(focus_metrics, dict):
            focus_metrics["focus_score_ratio"] = focus_score
            focus_metrics["focus_component_ratios"] = comp_ratios

        # 3. 触发判断
        if focus_score is None:
            self._log("[补焦] 参考未建立，仅保存指标")
            return focus_metrics

        need, reasons = self.evaluate_trigger(focus_score)
        self._log(
            f"[补焦] FocusScore={focus_score:.4f}, "
            f"low_count={self.consecutive_focus_low_count}, "
            f"need={need}, reasons={reasons}"
        )

        if not need:
            return focus_metrics

        if not self.cfg.autofocus_enabled:
            self._log("[补焦] 已触发但 autofocus_enabled=False，不移动")
            return focus_metrics

        # 4. 闭环补焦
        self._log("[补焦] 触发: " + "; ".join(reasons))
        autofocus_result = self.run_closed_loop(
            initial_focus_score=focus_score
        )

        # 5. 补焦后重新截图
        self._log("[补焦] 结束，重新截图")
        if save_dir is not None:
            final_metrics = self.metrics_calc.capture_and_save(
                cycle_index=cycle_index, save_dir=save_dir, on_log=self._log
            )
        else:
            final_metrics = self.metrics_calc.capture_live()

        final_roi = final_metrics.get("roi_metrics") if isinstance(final_metrics, dict) else None
        final_score, _ = self.scorer.score_ratio(final_roi)
        if isinstance(final_metrics, dict):
            final_metrics["focus_score_ratio"] = final_score
        if final_score is not None and final_score >= float(
            self.cfg.autofocus_focus_trigger_ratio
        ):
            self.consecutive_focus_low_count = 0

        return final_metrics

    # ================================================================
    # SHG 参考管理（外部调用）
    # ================================================================

    def update_shg_reference(
        self, fit_peak: Optional[float]
    ) -> Optional[float]:
        """
        更新 SHG 参考和 ratio。

        前 shg_reference_capture_count 次有效值建立 SHG_ref，
        之后每轮计算 SHG_ratio 并更新连续低计数。

        参数
        ----------
        fit_peak : float or None
            本轮拟合峰值。

        返回
        -------
        SHG_ratio or None
        """
        if fit_peak is None:
            return None

        ref_n = max(1, int(self.cfg.autofocus_shg_trigger_count) + 1)
        if len(self.shg_reference_values) < ref_n:
            self.shg_reference_values.append(float(fit_peak))
            self.shg_ref = float(np.mean(self.shg_reference_values))
            self._log(
                f"[SHG] 累计 {len(self.shg_reference_values)}/{ref_n}: "
                f"ref={self.shg_ref}"
            )
        elif self.shg_ref is None:
            self.shg_ref = float(
                np.mean(self.shg_reference_values[:ref_n])
            )

        if self.shg_ref is not None and abs(self.shg_ref) > 1e-12:
            self.last_shg_ratio = float(fit_peak) / float(self.shg_ref)

            if self.last_shg_ratio < float(
                self.cfg.autofocus_shg_trigger_ratio
            ):
                self.consecutive_shg_low_count += 1
            else:
                self.consecutive_shg_low_count = 0

            self._log(
                f"[SHG] ratio={self.last_shg_ratio:.4f}, "
                f"low_count={self.consecutive_shg_low_count}"
            )

        return self.last_shg_ratio

    def build_reference(self, output_root=None):
        """委托给 FocusScorer.build_reference()。"""
        return self.scorer.build_reference(output_root=output_root, on_log=self._log)

    @property
    def focus_reference_ready(self) -> bool:
        return self.scorer.focus_reference_ready