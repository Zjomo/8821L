#!/usr/bin/env python3
"""
Focus 模拟器 —— 无硬件演示模式的核心组件。

提供：
  - VirtualZAxis：虚拟 Z 轴，记录位置、支持相对移动
  - ImageGenerator：根据虚拟 Z 位置生成模拟显微图像
  - FocusSimulator：整合虚拟 Z 轴和图像生成，替代真实截图

用法示例：
    from Focus.simulator import FocusSimulator
    from Focus.config import AutofocusConfig
    from Focus.metrics import FocusMetricsCalculator
    from Focus.scorer import FocusScorer
    from Focus.controller import AutofocusController

    cfg = AutofocusConfig()
    sim = FocusSimulator(cfg, peak_z=30, blur_scale=0.5)

    # 替代 metrics_calc.capture_live()
    metrics_calc = FocusMetricsCalculator(cfg)
    metrics_calc.capture_live = sim.capture_live

    # 替代真实 Z 轴
    z_axis = sim.virtual_z_axis

    controller = AutofocusController(cfg, scorer, metrics_calc, z_axis)
"""

from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .config import AutofocusConfig
from .metrics import FocusMetricsCalculator

logger = logging.getLogger(__name__)


class VirtualZAxis:
    """
    虚拟 Z 轴控制器。

    不连接真实硬件，只记录虚拟位置，支持：
      - connect() / close()：空操作
      - move_relative(delta)：更新虚拟位置
      - get_position()：返回当前位置
    """

    def __init__(self, cfg: AutofocusConfig, initial_z: int = 0):
        self.cfg = cfg
        self.z = initial_z
        self.connected = False

    def connect(self) -> VirtualZAxis:
        self.connected = True
        return self

    def close(self) -> None:
        self.connected = False

    def move_relative(self, signed_steps: int) -> None:
        self.z += int(signed_steps)

    def move_absolute(self, target_steps: int) -> None:
        self.z = int(target_steps)

    def get_position(self) -> int:
        return self.z

    def reset(self, initial_z: int = 0) -> None:
        self.z = initial_z


class ImageGenerator:
    """
    根据虚拟 Z 位置生成模拟显微图像。

    设计思路：
      - 生成一张清晰的基准图像（棋盘格、纹理、或加载外部图片）
      - 根据当前 Z 与峰值 Z 的距离计算模糊半径
      - 用 Gaussian blur 模拟离焦效果
      - 可选添加噪声、亮度变化等

    聚焦曲线为单峰：score 最大在 peak_z，越远离越模糊。
    """

    def __init__(
        self,
        cfg: AutofocusConfig,
        peak_z: int = 50,
        blur_scale: float = 0.3,
        image_size: Tuple[int, int] = (400, 400),
        base_image_path: Optional[Union[str, Path]] = None,
        noise_level: float = 0.02,
        pattern: str = "cells",
    ):
        """
        参数
        ----------
        cfg : AutofocusConfig
        peak_z : int
            最佳聚焦时的虚拟 Z 步数（聚焦曲线峰值位置）
        blur_scale : float
            模糊系数：blur_radius = abs(z - peak_z) * blur_scale
        image_size : tuple
            生成的图像尺寸 (H, W)
        base_image_path : str | Path, optional
            外部基准图片路径。若提供则优先使用，否则生成纹理。
        noise_level : float
            添加高斯噪声的标准差（相对于 255）
        pattern : str
            内置图案类型：cells / grid / dots / random
        """
        self.cfg = cfg
        self.peak_z = peak_z
        self.blur_scale = max(0.01, blur_scale)
        self.image_size = image_size
        self.noise_level = max(0.0, min(1.0, noise_level))
        self.pattern = pattern

        # 加载或生成基准图像
        if base_image_path is not None and Path(base_image_path).exists():
            self.base_image = cv2.imread(str(base_image_path))
            if self.base_image is not None:
                self.base_image = cv2.resize(self.base_image, (image_size[1], image_size[0]))
            else:
                logger.warning(f"无法加载基准图片 {base_image_path}, 使用内置图案")
                self.base_image = self._generate_pattern()
        else:
            self.base_image = self._generate_pattern()

    def _generate_pattern(self) -> np.ndarray:
        """生成内置纹理图案。"""
        h, w = self.image_size
        img = np.zeros((h, w, 3), dtype=np.uint8)

        if self.pattern == "cells":
            # 类细胞图案：随机圆形 + 边缘
            for _ in range(20):
                cx = random.randint(0, w)
                cy = random.randint(0, h)
                r = random.randint(20, 60)
                color = (random.randint(100, 200), random.randint(100, 200), random.randint(100, 200))
                cv2.circle(img, (cx, cy), r, color, -1)
                cv2.circle(img, (cx, cy), r, (50, 50, 50), 2)
            # 添加一些细线
            for _ in range(10):
                x1 = random.randint(0, w)
                y1 = random.randint(0, h)
                x2 = random.randint(0, w)
                y2 = random.randint(0, h)
                cv2.line(img, (x1, y1), (x2, y2), (200, 200, 200), 1)

        elif self.pattern == "grid":
            # 棋盘格
            cell_size = 40
            for i in range(0, h, cell_size):
                for j in range(0, w, cell_size):
                    if (i // cell_size + j // cell_size) % 2 == 0:
                        img[i:i+cell_size, j:j+cell_size] = (180, 180, 180)
                    else:
                        img[i:i+cell_size, j:j+cell_size] = (80, 80, 80)

        elif self.pattern == "dots":
            # 高对比度点阵
            for i in range(0, h, 20):
                for j in range(0, w, 20):
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    cv2.circle(img, (j + 10, i + 10), 5, color, -1)

        else:
            # 随机纹理
            img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            # 添加一些边缘结构
            for _ in range(5):
                x1 = random.randint(0, w)
                y1 = random.randint(0, h)
                x2 = random.randint(0, w)
                y2 = random.randint(0, h)
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)

        return img

    def generate(self, current_z: int) -> np.ndarray:
        """
        根据当前 Z 位置生成模拟图像。

        参数
        ----------
        current_z : int
            当前虚拟 Z 步数

        返回
        -------
        blurred_image : np.ndarray (H, W, 3)
        """
        # 计算模糊半径
        distance = abs(current_z - self.peak_z)
        blur_radius = max(0, int(round(distance * self.blur_scale)))

        # 模糊基准图像
        if blur_radius > 0:
            # 确保 ksize 为奇数且 >= 3
            ksize = max(3, 2 * blur_radius + 1)
            blurred = cv2.GaussianBlur(self.base_image, (ksize, ksize), 0)
        else:
            blurred = self.base_image.copy()

        # 添加噪声
        if self.noise_level > 0:
            noise = np.random.normal(
                0, self.noise_level * 255, blurred.shape
            ).astype(np.float32)
            blurred = np.clip(blurred.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return blurred


class FocusSimulator:
    """
    完整聚焦模拟器，整合虚拟 Z 轴和图像生成。

    替代真实硬件，提供：
      - virtual_z_axis：虚拟 Z 轴控制器
      - capture_live()：根据当前虚拟 Z 位置生成图像并计算指标
      - 手动注入到 AutofocusController

    用法：
        sim = FocusSimulator(cfg)
        metrics_calc.capture_live = sim.capture_live
        controller.z_axis = sim.virtual_z_axis
    """

    def __init__(
        self,
        cfg: AutofocusConfig,
        peak_z: int = 50,
        blur_scale: float = 0.3,
        image_size: Tuple[int, int] = (400, 400),
        base_image_path: Optional[Union[str, Path]] = None,
        noise_level: float = 0.02,
        pattern: str = "cells",
        initial_z: int = 0,
        drift_rate: float = 0.0,
        focus_degrade_per_cycle: float = 0.0,
    ):
        """
        参数
        ----------
        cfg : AutofocusConfig
        peak_z : int
            最佳聚焦峰值位置（虚拟 Z 步数）
        blur_scale : float
            模糊系数
        image_size : tuple
            图像尺寸
        base_image_path : str | Path, optional
            外部基准图片
        noise_level : float
            噪声水平
        pattern : str
            内置图案类型
        initial_z : int
            起始虚拟 Z 位置
        drift_rate : float
            每轮自动漂移步数（模拟真实 Z 轴漂移）
        focus_degrade_per_cycle : float
            每轮聚焦分数衰减比例（模拟样品退化）
        """
        self.cfg = cfg
        self.peak_z = peak_z
        self.initial_peak_z = peak_z
        self.drift_rate = drift_rate
        self.focus_degrade_per_cycle = focus_degrade_per_cycle

        self.virtual_z_axis = VirtualZAxis(cfg, initial_z=initial_z)
        self.image_generator = ImageGenerator(
            cfg,
            peak_z=peak_z,
            blur_scale=blur_scale,
            image_size=image_size,
            base_image_path=base_image_path,
            noise_level=noise_level,
            pattern=pattern,
        )

        self.cycle_count = 0
        self._metrics_calc: Optional[FocusMetricsCalculator] = None

    def attach_metrics_calc(self, metrics_calc: FocusMetricsCalculator) -> None:
        """绑定 FocusMetricsCalculator，用于计算指标。"""
        self._metrics_calc = metrics_calc

    def capture_live(self) -> Dict[str, Any]:
        """
        替代 FocusMetricsCalculator.capture_live()。

        根据 virtual_z_axis 当前位置生成图像，并计算指标。
        """
        if self._metrics_calc is None:
            # 若未绑定，创建临时计算器
            self._metrics_calc = FocusMetricsCalculator(self.cfg)

        current_z = self.virtual_z_axis.get_position()
        logger.debug(f"[Simulator] capture_live at virtual Z={current_z}")

        # 生成模拟图像
        img = self.image_generator.generate(current_z)

        # 计算指标（整图 + ROI）
        full_metrics = self._metrics_calc.compute_for_image(img)

        # 计算 ROI 指标：先裁剪 ROI 区域
        roi = self.cfg.focus_roi
        if roi is not None and isinstance(roi, (tuple, list)) and len(roi) >= 4:
            x, y, w, h = roi[:4]
            h_img, w_img = img.shape[:2]
            x0 = max(0, int(x))
            y0 = max(0, int(y))
            x1 = min(w_img, x0 + int(w))
            y1 = min(h_img, y0 + int(h))
            roi_img = img[y0:y1, x0:x1]
            if roi_img.size > 0:
                roi_metrics = self._metrics_calc.compute_for_image(roi_img)
            else:
                roi_metrics = full_metrics
        else:
            roi_metrics = full_metrics

        return {
            "ok": True,
            "full_metrics": full_metrics,
            "roi_metrics": roi_metrics,
            "image": img,
            "virtual_z": current_z,
        }

    def capture_and_save(
        self,
        cycle_index: int = 0,
        save_dir: Optional[Union[str, Path]] = None,
        on_log: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        替代 FocusMetricsCalculator.capture_and_save()。

        生成模拟图像、计算指标，并保存到指定目录。
        """
        import time
        from datetime import datetime

        result = self.capture_live()
        img = result.get("image")
        roi_metrics = result.get("roi_metrics")
        full_metrics = result.get("full_metrics")

        if save_dir is not None:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_filename = f"simulated_{cycle_index:04d}_{timestamp}.png"
            csv_filename = f"metrics_{cycle_index:04d}_{timestamp}.csv"

            # 保存图像
            if img is not None:
                cv2.imwrite(str(save_path / img_filename), img)
                if on_log:
                    on_log(f"[模拟] 图像已保存：{img_filename}")

            # 保存指标 CSV
            if roi_metrics is not None:
                metrics_csv_path = save_path / csv_filename
                with metrics_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
                    import csv
                    writer = csv.writer(f)
                    writer.writerow(["metric", "value"])
                    for k, v in roi_metrics.items():
                        writer.writerow([k, v if v is not None else ""])
                if on_log:
                    on_log(f"[模拟] 指标已保存：{csv_filename}")

        return {
            "ok": True,
            "full_metrics": full_metrics,
            "roi_metrics": roi_metrics,
            "image_path": str(save_path / img_filename) if save_dir and img is not None else None,
            "virtual_z": result.get("virtual_z"),
        }

    def simulate_drift(self) -> None:
        """模拟 Z 轴漂移：每轮自动移动一点。"""
        if self.drift_rate != 0:
            drift = int(round(self.drift_rate))
            self.virtual_z_axis.move_relative(drift)
            logger.debug(f"[Simulator] drift {drift} steps, Z={self.virtual_z_axis.get_position()}")

    def simulate_focus_degrade(self) -> None:
        """模拟聚焦退化：峰值位置移动或模糊系数增加。"""
        if self.focus_degrade_per_cycle != 0:
            # 移动峰值位置（模拟样品沉降）
            self.peak_z += int(round(self.focus_degrade_per_cycle))
            self.image_generator.peak_z = self.peak_z
            logger.debug(
                f"[Simulator] focus degrade, new peak_z={self.peak_z}"
            )

    def next_cycle(self) -> None:
        """进入下一轮：应用漂移和退化。"""
        self.cycle_count += 1
        self.simulate_drift()
        self.simulate_focus_degrade()

    def reset(self, initial_z: int = 0, peak_z: Optional[int] = None) -> None:
        """重置模拟器状态。"""
        self.virtual_z_axis.reset(initial_z)
        if peak_z is not None:
            self.peak_z = peak_z
            self.image_generator.peak_z = peak_z
        self.cycle_count = 0

    def save_image(
        self,
        save_dir: Union[str, Path],
        filename: str = "simulated.png",
    ) -> Path:
        """保存当前生成的图像。"""
        current_z = self.virtual_z_axis.get_position()
        img = self.image_generator.generate(current_z)
        save_path = Path(save_dir) / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), img)
        return save_path


def create_demo_environment(
    cfg: AutofocusConfig,
    peak_z: int = 50,
    blur_scale: float = 0.3,
    image_size: Tuple[int, int] = (400, 400),
    base_image_path: Optional[Union[str, Path]] = None,
    noise_level: float = 0.02,
    pattern: str = "cells",
    initial_z: int = 0,
    drift_rate: float = 0.0,
    focus_degrade_per_cycle: float = 0.0,
    autofocus_enabled: bool = True,
) -> Tuple[FocusSimulator, FocusMetricsCalculator, "FocusScorer", "AutofocusController"]:
    """
    快速创建演示环境，返回已注入模拟器的组件。

    返回
    -------
    (simulator, metrics_calc, scorer, controller)
    """
    from .scorer import FocusScorer
    from .controller import AutofocusController

    # 创建模拟器
    sim = FocusSimulator(
        cfg,
        peak_z=peak_z,
        blur_scale=blur_scale,
        image_size=image_size,
        base_image_path=base_image_path,
        noise_level=noise_level,
        pattern=pattern,
        initial_z=initial_z,
        drift_rate=drift_rate,
        focus_degrade_per_cycle=focus_degrade_per_cycle,
    )

    # 创建计算器和评分器
    metrics_calc = FocusMetricsCalculator(cfg)
    sim.attach_metrics_calc(metrics_calc)

    # 替代 capture_live 和 capture_and_save
    metrics_calc.capture_live = sim.capture_live
    metrics_calc.capture_and_save = sim.capture_and_save

    scorer = FocusScorer(cfg, metrics_calc)

    # 用虚拟 Z 轴替代真实硬件
    controller = AutofocusController(cfg, scorer, metrics_calc, sim.virtual_z_axis)

    # 强制启用自动补焦（演示模式）
    cfg.autofocus_enabled = autofocus_enabled
    cfg.z_enabled = True  # 虚拟 Z 轴始终"启用"

    return sim, metrics_calc, scorer, controller