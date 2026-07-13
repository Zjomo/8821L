"""
光谱补焦循环模块。

基于参考源码 0_measurement_workflow_real_virtual_same_detection_6.25.py 的 GUI 流程，
把“测光谱 → 等待 2min → 被动补焦检测”封装为独立可测试类 SpectrumAutofocusLoop，
主 GUI 只做最小集成。

流程：
    0. 手动交互式框选 ROI，并以该截图建立 Focus 参考基线。
    1. 关照明 → LabVIEW 光谱采集 → 开照明 → 保存数据。
    2. 等待 2min。
    3. 启动被动补焦检测：
       默认参数：触发阈值 0.95；目标阈值 0.95；连续触发次数 3；
       搜索策略 hill_climb；Z 轴号 1；Z 控制器索引 0；Z 控制器后端 auto；
       间隔 1s；被动补焦；最大连续补焦次数 10；连续达标次数 5；
       输出目录 focus_output（同时保存 FocusScore 曲线）。
    4. 回到第 1 步继续测光谱。
"""

from __future__ import annotations

import csv
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

AUTOZOOM_ROOT = Path(__file__).resolve().parent
if str(AUTOZOOM_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOZOOM_ROOT))

from Focus.config import AutofocusConfig
from Focus.controller import AutofocusController
from Focus.metrics import FocusMetricsCalculator
from Focus.scorer import FocusScorer
from Focus.simulator import create_demo_environment
from Focus.z_axis import ZAxisController

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


LogCallback = Optional[Callable[[str], None]]
StopCallback = Optional[Callable[[], bool]]


class SpectrumAutofocusLoop:
    """
    光谱补焦循环。

    与 tkinter 解耦：通过 on_log 回调输出日志，通过 should_stop 接收外部停止信号。
    """

    # 用户要求的默认被动补焦参数
    DEFAULT_TRIGGER_RATIO: float = 0.95
    DEFAULT_STOP_RATIO: float = 0.95
    DEFAULT_TRIGGER_COUNT: int = 3
    DEFAULT_SEARCH_STRATEGY: str = "hill_climb"
    DEFAULT_Z_AXIS: int = 1
    DEFAULT_Z_CONN: int = 0
    DEFAULT_Z_BACKEND: str = "auto"
    DEFAULT_INTERVAL_S: float = 1.0
    DEFAULT_PASSIVE_MODE: bool = True
    DEFAULT_PASSIVE_MAX_ATTEMPTS: int = 10
    DEFAULT_PASSIVE_CONSECUTIVE_GOOD: int = 5
    DEFAULT_WAIT_BETWEEN_SPECTRUM_S: float = 120.0

    def __init__(
        self,
        workflow: Any,
        cfg: AutofocusConfig,
        output_dir: str = "focus_output",
        on_log: LogCallback = None,
        should_stop: StopCallback = None,
    ):
        self.workflow = workflow
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.on_log = on_log
        self.should_stop = should_stop

        self._reference_image: Optional[np.ndarray] = None
        self._reference_ready: bool = False
        self._roi_selected: bool = False

        # 被动补焦运行期组件，延迟初始化
        self._metrics_calc: Optional[FocusMetricsCalculator] = None
        self._scorer: Optional[FocusScorer] = None
        self._controller: Optional[AutofocusController] = None
        self._simulator: Optional[Any] = None

    # ------------------------------------------------------------------
    # 日志与工具
    # ------------------------------------------------------------------
    def _log(self, msg: str) -> None:
        if self.on_log:
            self.on_log(msg)
        else:
            print(msg)

    def _stopped(self) -> bool:
        if self.should_stop is not None:
            try:
                return bool(self.should_stop())
            except Exception as e:
                self._log(f"[光谱补焦] should_stop 回调异常：{e}")
        return False

    def _safe_sleep(self, seconds: float, step_s: float = 0.1) -> bool:
        """可中断的睡眠，返回是否被外部停止。"""
        waited = 0.0
        while waited < seconds:
            if self._stopped():
                return True
            time.sleep(min(step_s, seconds - waited))
            waited += step_s
        return False

    @staticmethod
    def _parse_roi_text(text: str) -> Tuple[int, int, int, int]:
        """把 'x,y,w,h' 字符串解析为整数元组。"""
        parts = [p.strip() for p in str(text).split(",")]
        if len(parts) != 4:
            raise ValueError(f"ROI 格式应为 x,y,w,h，实际为：{text}")
        return tuple(int(p) for p in parts)  # type: ignore

    @staticmethod
    def roi_to_screen_capture_area(
        screen_capture_area: Tuple[int, int, int, int],
        relative_roi: Tuple[int, int, int, int],
    ) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        """
        把在当前截图区域内框选的相对 ROI 转换为新的屏幕截图区域与 Focus ROI。

        参数
        ----------
        screen_capture_area : (L, T, W, H)
            选择 ROI 前当前的屏幕截图区域。
        relative_roi : (rx, ry, rw, rh)
            相对于 screen_capture_area 的 ROI。

        返回
        -------
        (new_capture_area, new_focus_roi)
            new_capture_area = (L+rx, T+ry, rw, rh)
            new_focus_roi  = (0, 0, rw, rh)
        """
        L, T, W, H = screen_capture_area
        rx, ry, rw, rh = relative_roi
        new_capture_area = (L + rx, T + ry, rw, rh)
        new_focus_roi = (0, 0, rw, rh)
        return new_capture_area, new_focus_roi

    # ------------------------------------------------------------------
    # ROI 选择与参考建立
    # ------------------------------------------------------------------
    def select_focus_roi_interactively(self) -> Tuple[int, int, int, int]:
        """
        交互式框选 Focus ROI。

        使用 workflow._capture_current_rule_ab_frame() 获取当前全屏截图，
        弹出 OpenCV 窗口用鼠标拖拽矩形，Enter/N 确认，R 重置，ESC/Q 取消。
        返回的 ROI 会写入 cfg.focus_roi 与 cfg.capture_area，
        同时以该截图建立 Focus 参考基线。
        """
        if cv2 is None:
            raise ImportError("需要安装 opencv-python 才能交互式选择 ROI")

        self._log("[光谱补焦] 准备交互式选择 Focus ROI")
        output_dir = self.output_dir / "roi_select"
        output_dir.mkdir(parents=True, exist_ok=True)

        image_rgb = self.workflow._capture_current_rule_ab_frame(output_dir=output_dir)
        if image_rgb is None or image_rgb.size == 0:
            raise RuntimeError("截图失败，无法选择 ROI")

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        h, w = image_bgr.shape[:2]
        scale = 0.85
        show_w = max(1, int(w * scale))
        show_h = max(1, int(h * scale))
        display = cv2.resize(image_bgr, (show_w, show_h), interpolation=cv2.INTER_AREA)

        rect: List[Tuple[int, int]] = []
        drawing = False

        def redraw() -> np.ndarray:
            canvas = display.copy()
            lines = [
                "Select Focus ROI: drag rectangle",
                "Enter/N: confirm | R: reset | ESC/Q: cancel",
            ]
            for i, s in enumerate(lines):
                cv2.putText(
                    canvas,
                    s,
                    (18, 28 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            if len(rect) == 2:
                x0, y0 = rect[0]
                x1, y1 = rect[1]
                cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 255, 0), 2)
            return canvas

        def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
            nonlocal drawing
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                rect.clear()
                rect.append((x, y))
                rect.append((x, y))
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                if rect:
                    rect[-1] = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                if rect:
                    rect[-1] = (x, y)

        window_name = "Select Focus ROI"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, show_w, show_h)
        cv2.setMouseCallback(window_name, on_mouse)

        try:
            while True:
                cv2.imshow(window_name, redraw())
                key = cv2.waitKey(30) & 0xFF

                if key in (27, ord("q"), ord("Q")):
                    raise RuntimeError("用户取消了 ROI 选择")
                if key in (ord("r"), ord("R")):
                    rect.clear()
                if key in (13, 10, ord("n"), ord("N")):
                    if len(rect) != 2:
                        self._log("[光谱补焦] 请先拖拽选择一个矩形区域")
                        continue
                    break
        finally:
            try:
                cv2.destroyWindow(window_name)
            except Exception:
                pass

        x0, y0 = rect[0]
        x1, y1 = rect[1]
        x_min, x_max = sorted((x0, x1))
        y_min, y_max = sorted((y0, y1))

        roi_x = int(x_min / scale)
        roi_y = int(y_min / scale)
        roi_w = int((x_max - x_min) / scale)
        roi_h = int((y_max - y_min) / scale)

        # 裁剪到图像边界
        roi_x = max(0, min(roi_x, w - 1))
        roi_y = max(0, min(roi_y, h - 1))
        roi_w = max(1, min(roi_w, w - roi_x))
        roi_h = max(1, min(roi_h, h - roi_y))

        self.cfg.focus_roi = (roi_x, roi_y, roi_w, roi_h)
        self.cfg.capture_area = (0, 0, w, h)
        self._reference_image = image_rgb
        self._roi_selected = True

        self._log(
            f"[光谱补焦] ROI 已选择：({roi_x}, {roi_y}, {roi_w}, {roi_h})"
        )

        # 以本次截图建立参考
        self.build_reference_from_single_capture()
        return (roi_x, roi_y, roi_w, roi_h)

    def build_reference_from_single_capture(self) -> Dict[str, Any]:
        """
        以单张截图建立 Focus 参考基线。

        优先使用 select_focus_roi_interactively() 中保存的截图；
        若未保存，则调用 metrics_calc.capture_live() 采集一次。
        """
        if PILImage is None:
            raise ImportError("需要安装 pillow 才能保存参考图")

        ref_dir = self.output_dir / "reference"
        ref_dir.mkdir(parents=True, exist_ok=True)

        if self._reference_image is not None:
            image_rgb = self._reference_image
            self._log("[光谱补焦] 使用 ROI 选择时的截图建立参考")
        else:
            self._log("[光谱补焦] 采集一次截图建立参考")
            metrics_calc = self._get_or_create_metrics_calc()
            live = metrics_calc.capture_live()
            image_rgb = live.get("full_rgb")
            if image_rgb is None or image_rgb.size == 0:
                raise RuntimeError("截图失败，无法建立参考")

        self._scorer = self._get_or_create_scorer()
        ref = self._scorer.build_reference_from_image(
            image_rgb=image_rgb,
            output_root=self.output_dir,
            on_log=self._log,
        )
        self._reference_ready = True
        self._log("[光谱补焦] Focus 参考基线建立完成")
        return ref

    # ------------------------------------------------------------------
    # 光谱测量
    # ------------------------------------------------------------------
    def measure_spectrum_once(self, cycle_index: int) -> Dict[str, Any]:
        """
        单轮光谱测量：关照明 → LabVIEW 测光谱 → 开照明 → 保存数据。
        """
        self._log(f"[光谱补焦] 第 {cycle_index} 轮光谱测量开始")

        self.workflow.light_off()
        self._log("[光谱补焦] 照明光已关闭")

        result = self.workflow.request_labview_spectrum(cycle_index)
        self._log(
            f"[光谱补焦] LabVIEW 光谱采集完成：num_points={result.get('num_points')}"
        )

        self.workflow.light_on()
        self._log("[光谱补焦] 照明光已打开")

        # 调用 workflow 的异步保存入口，复用已有保存逻辑
        paths = self.workflow.build_save_path(cycle_index)
        angle_placeholder = {
            "ok": False,
            "angle_deg": None,
            "reason": "spectrum_autofocus_loop_no_angle",
        }
        self.workflow.start_save_cycle_result_async(
            cycle_index=cycle_index,
            paths=paths,
            angle_before_result=angle_placeholder,
            angle_after_result=angle_placeholder,
            labview_result=result,
        )
        self._log(f"[光谱补焦] 数据已提交异步保存：{paths.get('spectrum_csv')}")

        # 同时保存一份轻量摘要到 focus_output
        summary_path = self.output_dir / f"spectrum_summary_cycle_{cycle_index:04d}.json"
        try:
            summary = {
                "cycle_index": cycle_index,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "num_points": result.get("num_points"),
                "raw_peak": result.get("raw_peak"),
                "fit_peak": result.get("fit_peak"),
                "csv_path": result.get("csv_path"),
            }
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self._log(f"[光谱补焦] 保存光谱摘要失败：{e}")

        return result

    # ------------------------------------------------------------------
    # 被动补焦检测
    # ------------------------------------------------------------------
    def _get_or_create_metrics_calc(self) -> FocusMetricsCalculator:
        if self._metrics_calc is None:
            self._metrics_calc = FocusMetricsCalculator(self.cfg)
        return self._metrics_calc

    def _get_or_create_scorer(self) -> FocusScorer:
        if self._scorer is None:
            metrics_calc = self._get_or_create_metrics_calc()
            self._scorer = FocusScorer(self.cfg, metrics_calc)
        return self._scorer

    def _create_controller(self) -> Tuple[AutofocusController, Optional[Any]]:
        """创建 AutofocusController；virtual 模式下返回注入模拟器的组件。"""
        if hasattr(self.workflow, "_is_virtual_hardware_mode") and self.workflow._is_virtual_hardware_mode():
            self._log("[光谱补焦] virtual 模式：使用 FocusSimulator 进行补焦")
            simulator, metrics_calc, scorer, controller = create_demo_environment(
                self.cfg,
                peak_z=50,
                blur_scale=0.3,
                image_size=(400, 400),
                initial_z=0,
            )
            self._metrics_calc = metrics_calc
            self._scorer = scorer
            self._simulator = simulator
            controller.on_log = self._log
            controller.should_stop = self._stopped
            return controller, simulator

        metrics_calc = self._get_or_create_metrics_calc()
        scorer = self._get_or_create_scorer()
        z_axis = ZAxisController(self.cfg)
        controller = AutofocusController(self.cfg, scorer, metrics_calc, z_axis)
        controller.on_log = self._log
        controller.should_stop = self._stopped
        return controller, None

    def _configure_passive_autofocus(self) -> None:
        """按需求设置被动补焦参数。"""
        self.cfg.autofocus_focus_trigger_ratio = self.DEFAULT_TRIGGER_RATIO
        self.cfg.autofocus_stop_ratio = self.DEFAULT_STOP_RATIO
        self.cfg.autofocus_focus_trigger_count = self.DEFAULT_TRIGGER_COUNT
        self.cfg.z_search_strategy = self.DEFAULT_SEARCH_STRATEGY
        self.cfg.z_axis = self.DEFAULT_Z_AXIS
        self.cfg.z_picomotor_conn = self.DEFAULT_Z_CONN
        self.cfg.z_picomotor_backend = self.DEFAULT_Z_BACKEND
        self.cfg.autofocus_passive_mode = self.DEFAULT_PASSIVE_MODE
        self.cfg.autofocus_passive_max_attempts = self.DEFAULT_PASSIVE_MAX_ATTEMPTS
        self.cfg.autofocus_passive_consecutive_good = self.DEFAULT_PASSIVE_CONSECUTIVE_GOOD

    def run_passive_autofocus_detection(self, cycle_index: int) -> Dict[str, Any]:
        """
        启动被动补焦检测。

        若当前 FocusScore 未连续低于触发阈值，则只监测不移动；
        若触发补焦，则执行闭环搜索，直到连续达标次数满足或外部停止。
        返回最后一轮结果，并保存 FocusScore 历史曲线。
        """
        self._log("[光谱补焦] 启动被动补焦检测")
        self._configure_passive_autofocus()

        if not self._reference_ready:
            self.build_reference_from_single_capture()

        controller, simulator = self._create_controller()
        self._controller = controller

        trigger_ratio = float(self.cfg.autofocus_focus_trigger_ratio)
        consecutive_good_target = max(
            1, int(self.cfg.autofocus_passive_consecutive_good)
        )
        interval_s = max(0.0, float(self.DEFAULT_INTERVAL_S))
        consecutive_good_count = 0
        history: List[Dict[str, Any]] = []

        last_result: Dict[str, Any] = {}

        while True:
            if self._stopped():
                self._log("[光谱补焦] 收到停止信号，退出被动补焦检测")
                break

            result = controller.check_and_autofocus(
                cycle_index=cycle_index, save_dir=None
            )
            last_result = result if isinstance(result, dict) else {}
            score = last_result.get("focus_score_ratio")

            record = {
                "cycle_index": cycle_index,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "focus_score_ratio": score,
            }
            history.append(record)

            if score is None:
                self._log("[光谱补焦] FocusScore 为空，等待下一轮")
            elif score > trigger_ratio:
                consecutive_good_count += 1
                self._log(
                    f"[光谱补焦] FocusScore={score:.4f} > {trigger_ratio}，"
                    f"连续达标 {consecutive_good_count}/{consecutive_good_target}"
                )
                if consecutive_good_count >= consecutive_good_target:
                    self._log(
                        f"[光谱补焦] 连续达标 {consecutive_good_target} 轮，停止补焦"
                    )
                    break
            else:
                if consecutive_good_count > 0:
                    self._log(
                        f"[光谱补焦] FocusScore={score:.4f} <= {trigger_ratio}，"
                        "连续达标计数重置"
                    )
                consecutive_good_count = 0

            if simulator is not None:
                simulator.next_cycle()

            # 等待间隔，期间可被停止
            if self._safe_sleep(interval_s):
                break

        self._save_focus_score_history(cycle_index, history)
        self._draw_focus_score_curve(cycle_index, history, trigger_ratio)
        return last_result

    # ------------------------------------------------------------------
    # FocusScore 曲线保存
    # ------------------------------------------------------------------
    def _save_focus_score_history(
        self, cycle_index: int, history: List[Dict[str, Any]]
    ) -> None:
        if not history:
            return
        csv_path = self.output_dir / f"focus_score_history_cycle_{cycle_index:04d}.csv"
        try:
            with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["cycle_index", "time", "focus_score_ratio"]
                )
                writer.writeheader()
                writer.writerows(history)
            self._log(f"[光谱补焦] FocusScore 历史已保存：{csv_path}")
        except Exception as e:
            self._log(f"[光谱补焦] 保存 FocusScore 历史失败：{e}")

    def _draw_focus_score_curve(
        self,
        cycle_index: int,
        history: List[Dict[str, Any]],
        trigger_ratio: float,
    ) -> None:
        if plt is None or not history:
            return
        try:
            scores = [
                r.get("focus_score_ratio") for r in history
            ]
            indices = list(range(1, len(scores) + 1))

            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(indices, scores, "b-o", label="FocusScore_ratio")
            ax.axhline(
                trigger_ratio,
                color="r",
                linestyle="--",
                label=f"trigger={trigger_ratio}",
            )
            ax.set_xlabel("Detection round")
            ax.set_ylabel("FocusScore_ratio")
            ax.set_title(f"FocusScore Curve - Cycle {cycle_index}")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.6)
            fig.tight_layout()

            png_path = self.output_dir / f"focus_score_curve_cycle_{cycle_index:04d}.png"
            fig.savefig(png_path, dpi=150)
            plt.close(fig)
            self._log(f"[光谱补焦] FocusScore 曲线已保存：{png_path}")
        except Exception as e:
            self._log(f"[光谱补焦] 绘制 FocusScore 曲线失败：{e}")

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def run(self, max_cycles: int = 0) -> None:
        """
        光谱补焦循环主入口。

        参数
        ----------
        max_cycles : int
            最大循环轮数，0 表示无限循环直到外部停止。
        """
        self._log("========== 光谱补焦循环启动 ==========")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self._roi_selected:
            self.select_focus_roi_interactively()

        if not self._reference_ready:
            self.build_reference_from_single_capture()

        cycle_index = 0
        while True:
            if self._stopped():
                self._log("[光谱补焦] 收到停止信号，退出循环")
                break

            cycle_index += 1

            if max_cycles > 0 and cycle_index > max_cycles:
                self._log(
                    f"[光谱补焦] 已达到最大循环次数 {max_cycles}，结束"
                )
                break

            self._log(f"========== 光谱补焦循环第 {cycle_index} 轮 ==========")
            self.measure_spectrum_once(cycle_index)

            if self._stopped():
                self._log("[光谱补焦] 收到停止信号，退出循环")
                break

            self._log(
                f"[光谱补焦] 等待 {self.DEFAULT_WAIT_BETWEEN_SPECTRUM_S} 秒"
            )
            if self._safe_sleep(self.DEFAULT_WAIT_BETWEEN_SPECTRUM_S):
                break

            self.run_passive_autofocus_detection(cycle_index)

        self._log("========== 光谱补焦循环结束 ==========")
