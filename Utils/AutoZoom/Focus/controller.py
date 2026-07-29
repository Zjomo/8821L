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


def focus_score_ratio_in_tolerance(
    focus_score_ratio: Optional[float],
    trigger_ratio: float,
    absolute: bool = True,
) -> bool:
    """
    判断 FocusScore_ratio 是否在允许区间内。

    参数
    ----------
    absolute : bool, 默认 True
        True  时使用以 1.0 为中心的对称区间
              [trigger_ratio, 2 - trigger_ratio]。
              例如 trigger_ratio=0.95 时允许区间为 [0.95, 1.05]。
        False 时使用单边下限，score >= trigger_ratio 即认为达标。
    """
    if focus_score_ratio is None:
        return False
    if not absolute:
        return focus_score_ratio >= trigger_ratio
    lower = min(trigger_ratio, 2.0 - trigger_ratio)
    upper = max(trigger_ratio, 2.0 - trigger_ratio)
    return lower <= focus_score_ratio <= upper


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

    def _focus_ratio_in_tolerance(self, focus_score_ratio: Optional[float]) -> bool:
        """
        判断 FocusScore_ratio 是否在允许范围内。

        根据 cfg.autofocus_trigger_absolute 选择：
          - True（默认）：以 1.0 为中心的对称区间
            [trigger_ratio, 2 - trigger_ratio]
          - False：仅单边下限 score >= trigger_ratio
        """
        return focus_score_ratio_in_tolerance(
            focus_score_ratio,
            float(self.cfg.autofocus_focus_trigger_ratio),
            bool(getattr(self.cfg, "autofocus_trigger_absolute", True)),
        )

    def _tolerance_bounds(self) -> Tuple[float, float]:
        """返回允许区间 [lower, upper]。

        absolute=True 时为以 1.0 为中心的对称区间
        [trigger_ratio, 2 - trigger_ratio]；否则 upper 为 inf。
        """
        ratio = float(self.cfg.autofocus_focus_trigger_ratio)
        absolute = bool(getattr(self.cfg, "autofocus_trigger_absolute", True))
        lower = min(ratio, 2.0 - ratio) if absolute else ratio
        upper = max(ratio, 2.0 - ratio) if absolute else float("inf")
        return lower, upper

    def _update_reference_to_current(
        self, roi_metrics: Optional[Dict[str, Any]]
    ) -> None:
        """用当前 ROI 指标更新聚焦参考，用于 ratio > upper 时归一化。

        当 FocusScore_ratio 持续偏高（> upper）时，说明当前焦点优于参考，
        搜索策略（最大化分数）无法把它降下来。此时用当前 ROI 指标
        替换参考基线，使后续 ratio 归一到 ~1.0。
        """
        if not isinstance(roi_metrics, dict) or not roi_metrics:
            self._log("[补焦] 当前 ROI 指标为空，跳过参考更新")
            return
        ref = self.scorer.focus_reference
        if not isinstance(ref, dict):
            ref = {}
        ref["roi_metric_ref"] = roi_metrics
        ref["focus_score_ref"] = 1.0
        ref["reference_updated_reason"] = "ratio_above_upper_tolerance"
        self.scorer.focus_reference = ref
        self.scorer.focus_reference_ready = True
        self._log(
            "[补焦] 聚焦参考已更新为当前 ROI 指标 "
            f"(highfreq={roi_metrics.get('highfreq_ratio')}, "
            f"tenengrad={roi_metrics.get('tenengrad')})"
        )

    def evaluate_trigger(
        self,
        focus_score_ratio: Optional[float],
    ) -> Tuple[bool, List[str]]:
        """
        判断是否需要触发自动补焦。

        触发条件：
          1. FocusScore_ratio 连续 N 轮超出以 1.0 为中心的对称区间
             [autofocus_focus_trigger_ratio, 2 - autofocus_focus_trigger_ratio]；
          2. SHG_ratio 单轮硬阈值或连续 N 轮低于阈值（可选）。

        参数
        ----------
        focus_score_ratio : float or None

        返回
        -------
        (need_autofocus, reasons)
        """
        reasons: List[str] = []

        ratio = float(self.cfg.autofocus_focus_trigger_ratio)
        absolute = bool(getattr(self.cfg, "autofocus_trigger_absolute", True))
        lower = min(ratio, 2.0 - ratio) if absolute else ratio
        upper = max(ratio, 2.0 - ratio) if absolute else float("inf")

        if focus_score_ratio is not None:
            if self._focus_ratio_in_tolerance(focus_score_ratio):
                self.consecutive_focus_low_count = 0
            else:
                self.consecutive_focus_low_count += 1
                if absolute:
                    direction = "偏低" if focus_score_ratio < lower else "偏高"
                    self._log(
                        f"[补焦] FocusScore_ratio={focus_score_ratio:.4f} {direction}，"
                        f"超出允许区间 [{lower:.4f}, {upper:.4f}]"
                    )
                else:
                    self._log(
                        f"[补焦] FocusScore_ratio={focus_score_ratio:.4f} "
                        f"低于阈值 {ratio:.4f}"
                    )

        if self.consecutive_focus_low_count >= int(
            self.cfg.autofocus_focus_trigger_count
        ):
            if absolute:
                reasons.append(
                    f"连续{self.consecutive_focus_low_count}轮 FocusScore_ratio "
                    f"超出 [{lower:.4f}, {upper:.4f}]"
                )
            else:
                reasons.append(
                    f"连续{self.consecutive_focus_low_count}轮 FocusScore_ratio "
                    f"低于 {ratio:.4f}"
                )

        # 偏高时立即触发（不等连续 N 轮）：
        # ratio > upper 意味着当前焦点优于参考，需尽快搜索峰值并更新参考以归一化
        if (
            focus_score_ratio is not None
            and absolute
            and focus_score_ratio > upper
        ):
            reasons.append(
                f"FocusScore_ratio={focus_score_ratio:.4f} > {upper:.4f}（偏高），"
                f"立即触发补焦"
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

        # 计算允许区间上限，传给搜索策略以避免 ratio > upper 时误判"已达目标"
        _lower, _upper = self._tolerance_bounds()
        upper_target = _upper if _upper != float("inf") else None

        self._log(
            f"========== 闭环补焦 #{event_id} 开始: strategy={strategy}, "
            f"target={target}, upper_target={upper_target} =========="
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
                upper_target=upper_target,
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

    def close(self) -> None:
        """关闭 Z 轴控制器，释放 USB 连接。"""
        if self.z_axis is not None:
            try:
                self.z_axis.close()
            except Exception as exc:
                logger.warning(f"[补焦] 关闭 Z 轴控制器失败：{exc}")

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

        # 4. 闭环补焦（或仅检测记录）
        detection_only = bool(
            getattr(self.cfg, "autofocus_detection_only", False)
        )
        if detection_only:
            self._log(
                "[FocusScore检测] 触发: " + "; ".join(reasons) + "，"
                "检测模式开启，不执行 Z 轴闭环补焦"
            )
            if isinstance(focus_metrics, dict):
                focus_metrics["triggered"] = True
                focus_metrics["detection_only"] = True
            return focus_metrics

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

        # 补焦后 ratio 仍 > upper 时，搜索策略（最大化分数）无法把它降下来，
        # 此时用当前 ROI 指标更新参考基线，使后续 ratio 归一到 ~1.0
        if final_score is not None:
            _lower, _upper = self._tolerance_bounds()
            if final_score > _upper:
                self._log(
                    f"[补焦] 补焦后 FocusScore_ratio={final_score:.4f} 仍 > "
                    f"{_upper:.4f}（偏高），更新聚焦参考以归一化"
                )
                self._update_reference_to_current(final_roi)
                # 参考更新后重新计算 ratio（应归一到 ~1.0）
                final_score, _ = self.scorer.score_ratio(final_roi)
                if isinstance(final_metrics, dict):
                    final_metrics["focus_score_ratio"] = final_score
                    final_metrics["reference_updated"] = True

        if final_score is not None and self._focus_ratio_in_tolerance(final_score):
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