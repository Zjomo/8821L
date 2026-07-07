"""
聚焦评分与参考基线建立模块。

从 measurement_autofocus_shg_closed_loop.py 提取：
  - _focus_component_weights()
  - calculate_focus_score_ratio()
  - update_focus_score_context()
  - build_focus_reference()
"""

from __future__ import annotations

import time
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import AutofocusConfig
from .metrics import FocusMetricsCalculator, FOCUS_METRIC_NAMES

logger = logging.getLogger(__name__)


class FocusScorer:
    """
    聚焦评分器。

    功能：
      1. 基于加权分量计算 FocusScore_ratio（当前/参考的加权均值）；
      2. 建立并管理聚焦参考基线 (focus_reference)。
    """

    def __init__(self, cfg: AutofocusConfig, metrics_calc: FocusMetricsCalculator):
        self.cfg = cfg
        self.metrics_calc = metrics_calc

        # 参考基线
        self.focus_reference: Dict[str, Any] = {}
        self.focus_reference_ready: bool = False

    # ================================================================
    # 权重
    # ================================================================

    def component_weights(self) -> Dict[str, float]:
        """获取归一化的分量权重。"""
        weights = {
            "highfreq_ratio": float(self.cfg.focus_weight_highfreq),
            "tenengrad": float(self.cfg.focus_weight_tenengrad),
            "brenner": float(self.cfg.focus_weight_brenner),
            "red_blue_ratio": float(self.cfg.focus_weight_red_blue),
            "modified_laplacian": float(self.cfg.focus_weight_modified_laplacian),
            "dct_energy": float(self.cfg.focus_weight_dct_energy),
            "smd": float(self.cfg.focus_weight_smd),
            "entropy": float(self.cfg.focus_weight_entropy),
        }
        total = sum(max(0.0, float(v)) for v in weights.values())
        if total <= 1e-12:
            return {
                "highfreq_ratio": 0.4,
                "tenengrad": 0.3,
                "brenner": 0.3,
                "red_blue_ratio": 0.0,
                "modified_laplacian": 0.0,
                "dct_energy": 0.0,
                "smd": 0.0,
                "entropy": 0.0,
            }
        return {
            k: max(0.0, float(v)) / total for k, v in weights.items()
        }

    # ================================================================
    # 评分计算
    # ================================================================

    def score_ratio(
        self,
        roi_metrics: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[float], Dict[str, Optional[float]]]:
        """
        计算当前 ROI 聚焦分数相对参考值的比例。

        FocusScore_ratio = weighted_mean(metric_current / metric_ref)

        使用 highfreq_ratio、tenengrad、brenner、red_blue_ratio 四个分量。
        """
        if not isinstance(roi_metrics, dict):
            return None, {}
        ref = (
            self.focus_reference.get("roi_metric_ref")
            if isinstance(self.focus_reference, dict)
            else None
        )
        if not isinstance(ref, dict):
            return None, {}

        weights = self.component_weights()
        ratios: Dict[str, Optional[float]] = {}
        vals: List[float] = []
        wts: List[float] = []

        for name, weight in weights.items():
            cur = FocusMetricsCalculator.safe_float(roi_metrics.get(name))
            base = FocusMetricsCalculator.safe_float(ref.get(name))
            if cur is None or base is None or abs(base) <= 1e-12 or weight <= 0:
                ratios[name] = None
                continue
            ratio = float(cur) / float(base)
            ratio = max(0.0, min(2.5, ratio))  # 限幅
            ratios[name] = ratio
            vals.append(ratio)
            wts.append(weight)

        if not vals or sum(wts) <= 1e-12:
            return None, ratios

        score = float(np.average(vals, weights=wts))
        return score, ratios

    # ================================================================
    # 参考基线建立
    # ================================================================

    def build_reference(
        self,
        capture_count: Optional[int] = None,
        sleep_s: float = 0.15,
        output_root: Optional[Path] = None,
        on_log: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        采集多次 ROI 指标，建立当前样品的聚焦参考。

        参数
        ----------
        capture_count : int, optional
            采集次数，默认使用 cfg.focus_reference_capture_count。
        sleep_s : float
            每次采集间隔秒数。
        output_root : Path, optional
            保存参考截图的根目录；为 None 时不保存。
        on_log : callable, optional
            日志回调。

        返回
        -------
        focus_reference dict
        """
        n = int(capture_count or self.cfg.focus_reference_capture_count)
        n = max(1, n)
        _log = on_log or logger.info
        _log(f"[聚焦参考] 开始采集 {n} 次 ROI 指标")

        samples: List[Dict[str, Any]] = []
        ref_dir = None
        if output_root is not None:
            ref_dir = (
                Path(output_root)
                / "focus_reference"
                / datetime.now().strftime("%Y%m%d_%H%M%S")
            )
            ref_dir.mkdir(parents=True, exist_ok=True)

        for i in range(1, n + 1):
            live = self.metrics_calc.capture_live()
            roi_metrics = live.get("roi_metrics") or {}
            samples.append(roi_metrics)

            if ref_dir is not None:
                try:
                    from PIL import Image as PILImage

                    PILImage.fromarray(live["full_rgb"]).save(
                        ref_dir / f"reference_{i:02d}_full.png"
                    )
                    PILImage.fromarray(live["roi_rgb"]).save(
                        ref_dir / f"reference_{i:02d}_roi.png"
                    )
                except Exception:
                    pass

            _log(
                f"[聚焦参考] {i}/{n}: "
                f"tenengrad={roi_metrics.get('tenengrad')}, "
                f"brenner={roi_metrics.get('brenner')}, "
                f"highfreq={roi_metrics.get('highfreq_ratio')}, "
                f"red_blue={roi_metrics.get('red_blue_ratio')}"
            )
            if i < n and sleep_s > 0:
                time.sleep(float(sleep_s))

        # 计算各指标均值作为参考
        ref_values: Dict[str, Optional[float]] = {}
        for name in FOCUS_METRIC_NAMES:
            vals = [
                FocusMetricsCalculator.safe_float(s.get(name))
                for s in samples
            ]
            vals = [v for v in vals if v is not None]
            ref_values[name] = float(np.mean(vals)) if vals else None

        self.focus_reference = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "capture_count": n,
            "roi": list(self.cfg.focus_roi),
            "roi_metric_ref": ref_values,
            "reference_dir": str(ref_dir) if ref_dir else None,
            "focus_score_ref": 1.0,
        }
        self.focus_reference_ready = True

        # 保存参考 CSV
        if ref_dir is not None:
            try:
                ref_csv = ref_dir / "focus_reference_metrics.csv"
                with ref_csv.open("w", encoding="utf-8-sig", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["metric", "reference_mean"])
                    for name in FOCUS_METRIC_NAMES:
                        writer.writerow([name, ref_values.get(name)])
                _log(f"[聚焦参考] 参考指标已保存：{ref_csv}")
            except Exception as e:
                _log(f"[聚焦参考] 保存 CSV 失败：{e}")

        _log(
            "[聚焦参考] 建立完成："
            f"highfreq_ref={ref_values.get('highfreq_ratio')}, "
            f"tenengrad_ref={ref_values.get('tenengrad')}, "
            f"brenner_ref={ref_values.get('brenner')}, "
            f"red_blue_ref={ref_values.get('red_blue_ratio')}"
        )
        return self.focus_reference