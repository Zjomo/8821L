#!/usr/bin/env python3
"""
自动对焦模块独立运行示例。

用法：
    # 在项目根目录下执行：
    cd AutoZoom

    # 纯计算模式（不连接硬件）
    python Focus/run_autofocus.py

    # 连接 Newport Z 轴
    python Focus/run_autofocus.py --z-enabled
"""

from __future__ import annotations

import sys
import time
import logging
from pathlib import Path

# 将项目根目录加入 sys.path，确保 from Focus.xxx import ... 能正常工作
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Focus.config import AutofocusConfig
from Focus.metrics import FocusMetricsCalculator
from Focus.scorer import FocusScorer
from Focus.z_axis import ZAxisController
from Focus.controller import AutofocusController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("run_autofocus")


def run_demo():
    """
    演示自动对焦模块的完整流程（无硬件模式）。
    依赖 pyautogui（截图）和 opencv-python（图像处理）。
    """
    logger.info("=" * 60)
    logger.info("自动对焦模块 演示开始")
    logger.info("=" * 60)

    # 1. 配置
    cfg = AutofocusConfig(
        capture_area=(116, 98, 1112, 886),
        focus_roi=(0, 0, 300, 300),
        autofocus_enabled=False,  # 演示模式不移动 Z 轴
    )
    logger.info(f"截图区域: {cfg.capture_area}")
    logger.info(f"ROI: {cfg.focus_roi}")

    # 2. 初始化各模块
    metrics_calc = FocusMetricsCalculator(cfg)
    scorer = FocusScorer(cfg, metrics_calc)
    z_axis = ZAxisController(cfg)
    controller = AutofocusController(cfg, scorer, metrics_calc, z_axis)
    controller.on_log = lambda msg: logger.info(msg)

    # 3. 建立聚焦参考
    logger.info("\n--- 建立聚焦参考 ---")
    output_dir = Path("focus_demo_output")
    ref = controller.build_reference(output_root=output_dir)
    logger.info(f"参考建立完成，共 {ref.get('capture_count')} 次采集")
    logger.info(f"参考指标: {ref.get('roi_metric_ref')}")

    # 4. 截图并计算指标
    logger.info("\n--- 单次截图与指标计算 ---")
    metrics = metrics_calc.capture_and_save(
        cycle_index=0, save_dir=output_dir / "samples", on_log=logger.info
    )
    for name in FocusMetricsCalculator.METRIC_NAMES:
        full_val = metrics.get("full", {}).get(name, "N/A")
        roi_val = metrics.get("roi_metrics", {}).get(name, "N/A")
        logger.info(f"  {name:20s}  full={full_val:>12}  roi={roi_val:>12}")

    # 5. 计算 FocusScore_ratio
    logger.info("\n--- FocusScore_ratio ---")
    score, comp = scorer.score_ratio(metrics.get("roi_metrics"))
    logger.info(f"FocusScore_ratio = {score:.4f}")
    for k, v in comp.items():
        logger.info(f"  {k}: {v:.4f}" if v is not None else f"  {k}: None")

    # 6. 补焦触发判断
    logger.info("\n--- 补焦触发判断 ---")
    need, reasons = controller.evaluate_trigger(score)
    logger.info(f"需要补焦: {need}")
    if reasons:
        for r in reasons:
            logger.info(f"  原因: {r}")

    # 7. 展平指标
    logger.info("\n--- 展平指标 (用于 CSV 保存) ---")
    flat = FocusMetricsCalculator.flatten_metrics(metrics)
    for k, v in flat.items():
        logger.info(f"  {k}: {v}")

    logger.info("=" * 60)
    logger.info("演示结束")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_demo()