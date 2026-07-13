from __future__ import annotations

import csv
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SAM2_REPO_ROOT = PROJECT_ROOT / "sam2-main"
if SAM2_REPO_ROOT.exists() and str(SAM2_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM2_REPO_ROOT))

from control.stage34 import ThorlabsPiezoDevice
from vision.overlap_detector import SAM2ColorOverlapDetector
from vision.screen_capture import CaptureArea


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class RuleACConfig:
    # --------------------------------------------------------
    # overlap area threshold
    # --------------------------------------------------------
    area_threshold_px: float = 100.0
    use_um2_threshold: bool = False
    area_threshold_um2: Optional[float] = None
    um_per_px: Optional[float] = None

    # --------------------------------------------------------
    # screen capture and SAM2 overlap detector
    # --------------------------------------------------------
    capture_area: CaptureArea = (116, 98, 1112, 886)
    output_dir: str = "outputs/rule_ac_overlap_stage12"
    sample_config_path: str = "sample_config_rule_ac.json"
    sam2_cfg: str = "configs/sam2.1/sam2.1_hiera_t.yaml"
    sam2_checkpoint: str = str(SAM2_REPO_ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt")
    sam2_device: str = "cuda"

    # HSV overlap color detection
    h_tol: int = 15
    s_tol: int = 60
    v_tol: int = 60
    patch_radius: int = 3
    roi_pad: int = 10
    min_area: int = 30
    open_k: int = 3
    close_k: int = 5
    click_scale: float = 0.85

    # --------------------------------------------------------
    # Stage12 motion
    # --------------------------------------------------------
    enable_stage: bool = False
    stage_device_id: str = "27267878"
    stage_axis: str = "horizontal"  # horizontal / vertical
    stage_direction: int = 1        # 1 or -1
    stage_step_size: int = 100
    stage_velocity: float = 5.0
    stage_acceleration: float = 5.0
    stage_max_voltage: float = 100.0
    wait_stage_move: bool = True
    stage_move_timeout_s: float = 2.0

    # --------------------------------------------------------
    # loop
    # --------------------------------------------------------
    max_cycles: int = 100
    loop_interval_s: float = 0.3
    stop_when_threshold_reached: bool = True
    save_csv: bool = True


class RuleACOverlapController:
    """
    Rule AC:
        - detect overlap area from fixed screen capture using SAM2 + color detection;
        - if area < threshold: move Stage12;
        - if area >= threshold: stay.
    """

    def __init__(self, cfg: RuleACConfig):
        self.cfg = cfg

        base_output_dir = Path(cfg.output_dir)
        run_name = time.strftime("run_%Y%m%d_%H%M%S")
        self.output_dir = base_output_dir / run_name
        suffix = 1
        while self.output_dir.exists():
            self.output_dir = base_output_dir / f"{run_name}_{suffix:02d}"
            suffix += 1
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.output_dir / "rule_ac_log.csv"
        self.history: List[Dict[str, Any]] = []
        self.cycle_index = 0

        self.detector = SAM2ColorOverlapDetector(
            sam2_cfg=cfg.sam2_cfg,
            sam2_checkpoint=cfg.sam2_checkpoint,
            sam2_device=cfg.sam2_device,
            sample_config_path=cfg.sample_config_path,
            output_dir=self.output_dir,
            capture_area=cfg.capture_area,
            save_capture_image=True,
            h_tol=cfg.h_tol,
            s_tol=cfg.s_tol,
            v_tol=cfg.v_tol,
            patch_radius=cfg.patch_radius,
            roi_pad=cfg.roi_pad,
            min_area=cfg.min_area,
            open_k=cfg.open_k,
            close_k=cfg.close_k,
            um_per_px=cfg.um_per_px,
            click_scale=cfg.click_scale,
        )

        self.stage: Optional[ThorlabsPiezoDevice] = None
        if cfg.enable_stage:
            self.stage = ThorlabsPiezoDevice()
            self.stage.connect(cfg.stage_device_id)
            self._setup_stage_drive()
            logger.info("Stage12 已连接。")
        else:
            logger.warning("enable_stage=False：当前只计算面积和动作，不实际移动 Stage12。")

    def _setup_stage_drive(self) -> None:
        if self.stage is None:
            return

        axis = self.cfg.stage_axis.strip().lower()
        if axis in ("horizontal", "h", "水平"):
            channel = 1
        elif axis in ("vertical", "v", "垂直"):
            channel = 2
        else:
            raise ValueError(f"未知 stage_axis: {self.cfg.stage_axis}")

        self.stage.setup_drive(
            channel=channel,
            velocity=self.cfg.stage_velocity,
            acceleration=self.cfg.stage_acceleration,
            max_voltage=self.cfg.stage_max_voltage,
        )

    def _threshold_value(self) -> Tuple[str, float]:
        if self.cfg.use_um2_threshold:
            if self.cfg.area_threshold_um2 is None:
                raise ValueError("use_um2_threshold=True 时必须设置 area_threshold_um2。")
            return "um2", float(self.cfg.area_threshold_um2)
        return "px", float(self.cfg.area_threshold_px)

    def _area_value(self, detect_result: Dict[str, Any]) -> Optional[float]:
        unit, _ = self._threshold_value()
        if unit == "um2":
            area = detect_result.get("area_um2")
        else:
            area = detect_result.get("area_px")
        return None if area is None else float(area)

    def should_move(self, detect_result: Dict[str, Any]) -> Tuple[bool, str, Optional[float], float]:
        unit, threshold = self._threshold_value()

        if not detect_result.get("success", False):
            return False, f"detect_failed_{detect_result.get('reason', '')}", None, threshold

        area = self._area_value(detect_result)
        if area is None:
            return False, f"area_{unit}_missing", None, threshold

        if area < threshold:
            return True, f"area_{unit}_below_threshold", area, threshold

        return False, f"area_{unit}_reached_threshold", area, threshold

    def execute_stage_move(self) -> None:
        direction = 1 if self.cfg.stage_direction >= 0 else -1

        if self.stage is None:
            logger.info(
                "[DRY-RUN] Stage12 move axis=%s direction=%d step=%d",
                self.cfg.stage_axis,
                direction,
                self.cfg.stage_step_size,
            )
            return

        logger.info(
            "Stage12 move axis=%s direction=%d step=%d",
            self.cfg.stage_axis,
            direction,
            self.cfg.stage_step_size,
        )
        self.stage.move_axis(
            axis=self.cfg.stage_axis,
            direction=direction,
            step_size=self.cfg.stage_step_size,
        )

        if self.cfg.wait_stage_move:
            self.stage.wait_until_stopped(
                axis=self.cfg.stage_axis,
                timeout_s=self.cfg.stage_move_timeout_s,
            )

    def run_one_cycle(self) -> bool:
        self.cycle_index += 1
        result = self.detector.detect_once_from_screen(force_reselect_sample=False)
        move, reason, area, threshold = self.should_move(result)
        unit, _ = self._threshold_value()

        action = "MOVE_STAGE12" if move else "STAY"
        logger.info(
            "[Cycle %04d] area_%s=%s threshold=%.3f action=%s reason=%s",
            self.cycle_index,
            unit,
            "None" if area is None else f"{area:.3f}",
            threshold,
            action,
            reason,
        )

        row = {
            "time": time.time(),
            "cycle": self.cycle_index,
            "success": result.get("success", False),
            "reason": reason,
            "raw_reason": result.get("reason", ""),
            "area_px": result.get("area_px"),
            "area_um2": result.get("area_um2"),
            "threshold_unit": unit,
            "threshold": threshold,
            "action": action,
            "stage_axis": self.cfg.stage_axis,
            "stage_direction": 1 if self.cfg.stage_direction >= 0 else -1,
            "stage_step_size": self.cfg.stage_step_size,
            "image_path": result.get("image_path"),
            "vis_path": result.get("vis_path"),
            "roi_rect": result.get("roi_rect"),
        }
        self.history.append(row)

        if self.cfg.save_csv:
            self.flush_csv()

        if move:
            self.execute_stage_move()
            return True

        if self.cfg.stop_when_threshold_reached and result.get("success", False):
            return False

        return True

    def flush_csv(self) -> None:
        if not self.history:
            return

        fieldnames = sorted(set(k for row in self.history for k in row.keys()))
        with open(self.csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.history)

    def run(self) -> None:
        try:
            for _ in range(self.cfg.max_cycles):
                should_continue = self.run_one_cycle()
                if not should_continue:
                    logger.info("面积达到阈值或无需继续移动，停止 Rule AC。")
                    break
                time.sleep(float(self.cfg.loop_interval_s))
        except KeyboardInterrupt:
            logger.warning("用户中断 Rule AC。")
        finally:
            self.close()

    def close(self) -> None:
        if self.cfg.save_csv:
            self.flush_csv()
            logger.info("CSV 已保存: %s", self.csv_path)

        if self.stage is not None:
            try:
                self.stage.stop_all()
            except Exception:
                pass
            try:
                self.stage.close()
            except Exception:
                pass
            logger.info("Stage12 已关闭。")


def main() -> None:
    cfg = RuleACConfig(
        # 面积阈值：默认使用 px^2
        area_threshold_px=100.0,
        use_um2_threshold=False,
        area_threshold_um2=None,
        um_per_px=None,

        capture_area=(116, 98, 1112, 886),
        output_dir="outputs/rule_ac_overlap_stage12",
        sample_config_path="sample_config_rule_ac.json",

        # Stage12：调试时保持 False；确认方向后再改 True。
        enable_stage=False,
        stage_device_id="27267878",
        stage_axis="horizontal",
        stage_direction=1,
        stage_step_size=100,

        max_cycles=100,
        loop_interval_s=0.3,
        stop_when_threshold_reached=True,
    )

    controller = RuleACOverlapController(cfg)
    controller.run()


if __name__ == "__main__":
    main()
