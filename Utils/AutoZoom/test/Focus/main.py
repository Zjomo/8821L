#!/usr/bin/env python3
"""
Focus 模块闭环主程序。

支持：
  - 建立聚焦参考基线
  - 定时循环自动对焦（截图 → 评分 → 触发判断 → 闭环搜索 → 保存）
  - 无硬件完全模拟模式（虚拟 Z 轴 + 模拟显微图像）
  - 结果保存为 CSV

用法示例：
    # 仅建立参考
    python Focus/main.py --build-ref --output output/ref_session

    # 闭环运行 20 轮，每轮间隔 5 秒
    python Focus/main.py --cycles 20 --interval 5 --output output/run_20260704

    # 无硬件完全模拟模式（虚拟聚焦搜索）
    python Focus/main.py --demo-sim --cycles 10 --peak-z 50 --blur-scale 0.3

    # 模拟模式 + Z 轴漂移（每轮自动偏离 3 步）
    python Focus/main.py --demo-sim --cycles 20 --drift-rate 3

    # 使用曲线拟合策略
    python Focus/main.py --demo-sim --strategy curve_fit --cycles 10
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# 将项目根目录加入 sys.path，确保 from Focus.xxx import ... 能正常工作
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Focus.config import AutofocusConfig
from Focus.metrics import FocusMetricsCalculator
from Focus.scorer import FocusScorer
from Focus.z_axis import ZAxisController
from Focus.controller import AutofocusController
from Focus.simulator import FocusSimulator, create_demo_environment

logger = logging.getLogger("Focus.main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Focus 自动对焦闭环主程序",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="focus_output",
        help="输出根目录",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="JSON 配置文件路径（键名与 AutofocusConfig 字段一致）",
    )

    # 运行模式
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--build-ref",
        action="store_true",
        help="仅建立聚焦参考基线后退出",
    )
    mode.add_argument(
        "--demo-sim",
        action="store_true",
        help="无硬件完全模拟模式：虚拟 Z 轴 + 模拟显微图像",
    )
    mode.add_argument(
        "--demo",
        action="store_true",
        help="旧演示模式：不移动 Z 轴，仍需真实截图",
    )

    # 模拟参数（仅 --demo-sim 生效）
    sim_group = parser.add_argument_group("模拟模式参数")
    sim_group.add_argument(
        "--peak-z",
        type=int,
        default=50,
        help="模拟聚焦曲线峰值位置（虚拟 Z 步数）",
    )
    sim_group.add_argument(
        "--blur-scale",
        type=float,
        default=0.3,
        help="模拟模糊系数：blur_radius = abs(z - peak_z) * blur_scale",
    )
    sim_group.add_argument(
        "--initial-z",
        type=int,
        default=0,
        help="虚拟 Z 轴起始位置",
    )
    sim_group.add_argument(
        "--drift-rate",
        type=float,
        default=0.0,
        help="每轮 Z 轴自动漂移步数（模拟真实漂移，正值远离峰值）",
    )
    sim_group.add_argument(
        "--focus-degrade",
        type=float,
        default=0.0,
        help="每轮峰值位置移动步数（模拟样品沉降或退化）",
    )
    sim_group.add_argument(
        "--pattern",
        type=str,
        default="cells",
        choices=["cells", "grid", "dots", "random"],
        help="模拟图像图案类型",
    )
    sim_group.add_argument(
        "--sim-image-size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=[400, 400],
        help="模拟图像尺寸",
    )
    sim_group.add_argument(
        "--noise-level",
        type=float,
        default=0.02,
        help="模拟噪声水平（相对于 255）",
    )
    sim_group.add_argument(
        "--base-image",
        type=str,
        default=None,
        help="外部基准图片路径（替代内置图案）",
    )

    # 循环参数
    parser.add_argument(
        "--cycles",
        type=int,
        default=0,
        help="循环轮数（0 表示无限循环，直到 Ctrl+C）",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="每轮之间的间隔时间（秒）",
    )

    # 常用覆盖参数
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=["hill_climb", "full_sweep", "curve_fit", "golden_section"],
        help="覆盖搜索策略",
    )
    parser.add_argument(
        "--z-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否启用 Z 轴硬件",
    )
    parser.add_argument(
        "--autofocus-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否允许自动补焦移动 Z 轴",
    )
    parser.add_argument(
        "--capture-area",
        type=int,
        nargs=4,
        metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"),
        default=None,
        help="覆盖屏幕截图区域",
    )
    parser.add_argument(
        "--focus-roi",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        default=None,
        help="覆盖 ROI 区域",
    )

    # 日志
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )
    parser.add_argument(
        "--log-file",
        action="store_true",
        help="同时将日志写入 output/main.log",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> AutofocusConfig:
    """根据命令行参数和可选 JSON 配置文件构造 AutofocusConfig。"""
    cfg_kwargs: Dict[str, Any] = {}

    if args.config is not None:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在：{config_path}")
        with config_path.open("r", encoding="utf-8") as f:
            cfg_kwargs.update(json.load(f))

    # 命令行覆盖
    if args.strategy is not None:
        cfg_kwargs["z_search_strategy"] = args.strategy
    if args.z_enabled is not None:
        cfg_kwargs["z_enabled"] = args.z_enabled
    if args.autofocus_enabled is not None:
        cfg_kwargs["autofocus_enabled"] = args.autofocus_enabled
    if args.capture_area is not None:
        cfg_kwargs["capture_area"] = tuple(args.capture_area)
    if args.focus_roi is not None:
        cfg_kwargs["focus_roi"] = tuple(args.focus_roi)

    # 旧演示模式默认不移动 Z 轴
    if args.demo:
        cfg_kwargs.setdefault("z_enabled", False)
        cfg_kwargs.setdefault("autofocus_enabled", False)

    # 模拟模式：虚拟 Z 轴启用，自动补焦启用
    if args.demo_sim:
        cfg_kwargs["z_enabled"] = True
        cfg_kwargs["autofocus_enabled"] = True

    return AutofocusConfig(**cfg_kwargs)


def setup_logging(output_root: Path, log_level: str, log_file: bool) -> None:
    """配置控制台（及可选文件）日志。"""
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        output_root.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.FileHandler(
                output_root / "main.log",
                mode="a",
                encoding="utf-8",
            )
        )

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def make_csv_row(cycle_index: int, result: Dict[str, Any]) -> Dict[str, Any]:
    """把一轮结果整理成 CSV 行。"""
    row: Dict[str, Any] = {
        "cycle_index": cycle_index,
        "time": result.get("time") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "focus_score_ratio": result.get("focus_score_ratio"),
    }
    flat = FocusMetricsCalculator.flatten_metrics(result)
    for k, v in flat.items():
        row[k] = v
    return row


def write_csv_headers(csv_path: Path, fieldnames: List[str]) -> None:
    """若 CSV 不存在则写入表头。"""
    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()


def append_csv_row(csv_path: Path, row: Dict[str, Any], fieldnames: List[str]) -> None:
    """追加一行到 CSV。"""
    write_csv_headers(csv_path, fieldnames)
    with csv_path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def build_reference(controller: AutofocusController, output_root: Path) -> None:
    """建立聚焦参考基线。"""
    ref_dir = output_root / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    logger.info("开始建立聚焦参考基线...")
    ref = controller.build_reference(output_root=ref_dir)
    logger.info(f"参考建立完成：{ref.get('capture_count')} 次采集")
    logger.info(f"参考指标：{ref.get('roi_metric_ref')}")


def run_loop(
    controller: AutofocusController,
    cfg: AutofocusConfig,
    args: argparse.Namespace,
    simulator: Optional[FocusSimulator] = None,
) -> None:
    """执行闭环循环。"""
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    save_dir = output_root / "cycles"
    csv_path = output_root / "focus_summary.csv"

    cycles = args.cycles
    interval = max(0.0, float(args.interval))
    cycle_index = 0
    running = True

    logger.info(f"开始闭环循环：cycles={'无限' if cycles <= 0 else cycles}, interval={interval}s")
    logger.info(f"搜索策略：{cfg.z_search_strategy}, Z轴启用：{cfg.z_enabled}, 自动补焦：{cfg.autofocus_enabled}")

    if simulator is not None:
        logger.info(
            f"模拟模式：peak_z={simulator.peak_z}, initial_z={args.initial_z}, "
            f"drift_rate={args.drift_rate}, focus_degrade={args.focus_degrade}"
        )

    # 预定义 CSV 列：以第一行结果为准，后续若出现新列则追加
    fieldnames: List[str] = []

    try:
        while running:
            cycle_index += 1
            if cycles > 0 and cycle_index > cycles:
                break

            logger.info(f"========== 第 {cycle_index} 轮 ==========")
            cycle_save_dir = save_dir / f"cycle_{cycle_index:04d}"
            cycle_save_dir.mkdir(parents=True, exist_ok=True)

            result = controller.check_and_autofocus(
                cycle_index=cycle_index,
                save_dir=str(cycle_save_dir),
            )

            score = result.get("focus_score_ratio")
            logger.info(f"第 {cycle_index} 轮 FocusScore_ratio = {score}")

            # 记录虚拟 Z 位置（模拟模式）
            if simulator is not None:
                virtual_z = simulator.virtual_z_axis.get_position()
                logger.info(f"虚拟 Z 位置 = {virtual_z}, 峰值位置 = {simulator.peak_z}")
                if isinstance(result, dict):
                    result["virtual_z"] = virtual_z

            row = make_csv_row(cycle_index, result)
            if not fieldnames:
                fieldnames = list(row.keys())
                # 确保基础列在最前面
                for col in reversed(["cycle_index", "time", "focus_score_ratio", "virtual_z"]):
                    if col in fieldnames:
                        fieldnames.remove(col)
                        fieldnames.insert(0, col)
            append_csv_row(csv_path, row, fieldnames)

            # 模拟模式：应用漂移和退化
            if simulator is not None:
                simulator.next_cycle()

            if cycles <= 0 or cycle_index < cycles:
                logger.info(f"等待 {interval}s 后进行下一轮...")
                time.sleep(interval)

    except KeyboardInterrupt:
        logger.warning("收到 Ctrl+C，正在安全退出...")
    finally:
        logger.info(f"循环结束，共执行 {cycle_index} 轮，结果保存至：{csv_path}")
        try:
            controller.z_axis.close()
        except Exception as exc:
            logger.warning(f"关闭 Z 轴时出错：{exc}")


def main() -> int:
    args = parse_args()
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    setup_logging(output_root, args.log_level, args.log_file)

    try:
        cfg = build_config(args)
    except Exception as exc:
        logger.error(f"配置加载失败：{exc}")
        return 1

    logger.info(f"输出目录：{output_root.resolve()}")

    simulator: Optional[FocusSimulator] = None

    if args.demo_sim:
        # 模拟模式：使用虚拟 Z 轴和模拟图像
        logger.info("启用无硬件完全模拟模式")
        simulator, metrics_calc, scorer, controller = create_demo_environment(
            cfg,
            peak_z=args.peak_z,
            blur_scale=args.blur_scale,
            image_size=tuple(args.sim_image_size),
            base_image_path=args.base_image,
            noise_level=args.noise_level,
            pattern=args.pattern,
            initial_z=args.initial_z,
            drift_rate=args.drift_rate,
            focus_degrade_per_cycle=args.focus_degrade,
            autofocus_enabled=True,
        )
        controller.on_log = lambda msg: logger.info(msg)

        if args.build_ref:
            build_reference(controller, output_root)
            return 0

        # 建立参考基线（模拟模式下，在峰值位置建立参考）
        if not controller.focus_reference_ready:
            logger.info("模拟模式：在峰值位置建立参考基线...")
            # 先将虚拟 Z 移动到峰值
            simulator.virtual_z_axis.move_absolute(simulator.peak_z)
            build_reference(controller, output_root)

        run_loop(controller, cfg, args, simulator=simulator)
        return 0

    # 非模拟模式：真实硬件或旧演示模式
    metrics_calc = FocusMetricsCalculator(cfg)
    scorer = FocusScorer(cfg, metrics_calc)
    z_axis = ZAxisController(cfg)
    controller = AutofocusController(cfg, scorer, metrics_calc, z_axis)
    controller.on_log = lambda msg: logger.info(msg)

    if args.build_ref:
        build_reference(controller, output_root)
        return 0

    # 非纯参考模式：先尝试建立/加载参考
    if not controller.focus_reference_ready:
        ref_dir = output_root / "reference"
        if ref_dir.exists() and any(ref_dir.glob("focus_reference_metrics.csv")):
            logger.info("发现已有参考目录，将在下一轮自动重建参考...")
        build_reference(controller, output_root)

    run_loop(controller, cfg, args, simulator=None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
