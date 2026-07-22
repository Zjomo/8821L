"""
补焦集成模块。

从 0_measurement_workflow_real_virtual_same_detection_7_16.py 提取的补焦逻辑，
封装为独立的 FocusIntegration 类，便于测试和维护。

主要功能：
  1. 建立聚焦参考基准图（capture_focus_reference）
  2. 计算当前 FocusScore_ratio（compute_current_focus_score）
  3. 触发闭环补焦（run_autofocus_if_needed）
  4. 交互式 ROI 选择（select_focus_roi_interactively）
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

from Focus.config import AutofocusConfig
from Focus.controller import AutofocusController, focus_score_ratio_in_tolerance
from Focus.metrics import FocusMetricsCalculator
from Focus.scorer import FocusScorer
from Focus.simulator import create_demo_environment
from Focus.z_axis import ZAxisController


@dataclass
class FocusConfig:
    """补焦配置参数，从 MeasurementConfig 中提取。"""

    saf_capture_area: Tuple[int, int, int, int] = (0, 0, 300, 300)
    saf_focus_roi: Tuple[int, int, int, int] = (0, 0, 300, 300)
    focus_trigger_ratio: float = 0.95
    focus_stop_ratio: float = 0.95
    focus_trigger_count: int = 3
    focus_trigger_absolute: bool = True
    focus_detection_only: bool = False
    focus_z_enabled: bool = True
    focus_z_axis: int = 1
    focus_z_speed: int = 100
    focus_z_accel: int = 100
    focus_search_strategy: str = "hill_climb"
    hardware_mode: str = "real"


@dataclass
class FocusState:
    """补焦运行期状态。"""

    reference_ready: bool = False
    reference_image: Optional[np.ndarray] = None
    consecutive_low_count: int = 0
    metrics_calc: Optional[FocusMetricsCalculator] = None
    scorer: Optional[FocusScorer] = None
    controller: Optional[AutofocusController] = None
    simulator: Optional[Any] = None


class FocusIntegration:
    """
    补焦集成类。

    封装补焦相关的所有逻辑，包括：
      - 建立聚焦参考基准图
      - 计算当前 FocusScore_ratio
      - 判断是否触发补焦
      - 执行闭环补焦搜索
      - 交互式 ROI 选择

    用法：
        focus = FocusIntegration(
            config=FocusConfig(...),
            output_root=Path("output"),
            is_virtual_mode=lambda: False,
            capture_frame_callback=lambda: np.zeros((100, 100, 3)),
            log_callback=print,
            stop_callback=lambda: False,
        )

        # 建立参考
        ok = focus.capture_focus_reference(cycle_index=1)

        # 循环中判断补焦
        result = focus.run_autofocus_if_needed(cycle_index=1)
    """

    def __init__(
        self,
        config: FocusConfig,
        output_root: Path,
        is_virtual_mode: Callable[[], bool],
        capture_frame_callback: Callable[[], Optional[np.ndarray]],
        log_callback: Optional[Callable[[str], None]] = None,
        stop_callback: Optional[Callable[[], bool]] = None,
    ):
        """
        初始化补焦集成实例。

        参数
        ----------
        config : FocusConfig
            补焦配置参数。
        output_root : Path
            输出根目录，用于保存参考图等。
        is_virtual_mode : Callable[[], bool]
            判断是否为虚拟硬件模式的回调。
        capture_frame_callback : Callable[[], Optional[np.ndarray]]
            截取当前 RuleAB 画面（用于 ROI 选择）的回调。
        log_callback : Callable[[str], None], optional
            日志输出回调。
        stop_callback : Callable[[], bool], optional
            判断是否请求停止的回调。
        """
        self.config = config
        self.output_root = output_root
        self._is_virtual_mode = is_virtual_mode
        self._capture_frame = capture_frame_callback
        self._log = log_callback or print
        self._should_stop = stop_callback or (lambda: False)

        self._state = FocusState()

    # ------------------------------------------------------------------
    # 日志与工具
    # ------------------------------------------------------------------

    def log(self, msg: str) -> None:
        """输出日志。"""
        if self._log:
            self._log(msg)

    def _stopped(self) -> bool:
        """判断是否请求停止。"""
        try:
            return bool(self._should_stop())
        except Exception:
            return False

    @staticmethod
    def parse_roi_text(value: Any) -> Tuple[int, int, int, int]:
        """把 'x,y,w,h' 字符串解析为整数元组；失败返回 (0,0,300,300)。"""
        try:
            parts = [int(v.strip()) for v in str(value).split(",")]
            if len(parts) == 4:
                return tuple(parts)  # type: ignore
        except Exception:
            pass
        return (0, 0, 300, 300)

    # ------------------------------------------------------------------
    # 配置与组件
    # ------------------------------------------------------------------

    def _make_focus_config(self) -> AutofocusConfig:
        """根据当前配置构造 AutofocusConfig。"""
        capture_area = self.parse_roi_text(self.config.saf_capture_area)
        focus_roi = self.parse_roi_text(self.config.saf_focus_roi)
        return AutofocusConfig(
            capture_mode="screen_region",
            capture_area=capture_area,
            focus_roi=focus_roi,
            autofocus_enabled=True,
            autofocus_focus_trigger_ratio=float(self.config.focus_trigger_ratio),
            autofocus_stop_ratio=float(self.config.focus_stop_ratio),
            autofocus_focus_trigger_count=int(self.config.focus_trigger_count),
            autofocus_trigger_absolute=bool(self.config.focus_trigger_absolute),
            autofocus_detection_only=bool(self.config.focus_detection_only),
            z_enabled=bool(self.config.focus_z_enabled),
            z_axis=int(self.config.focus_z_axis),
            z_speed=int(self.config.focus_z_speed),
            z_accel=int(self.config.focus_z_accel),
            z_search_strategy=str(self.config.focus_search_strategy),
        )

    def _ensure_focus_components(self) -> None:
        """延迟初始化 Focus 评分组件。"""
        if self._state.metrics_calc is None:
            cfg = self._make_focus_config()
            self._state.metrics_calc = FocusMetricsCalculator(cfg)
        if self._state.scorer is None:
            self._state.scorer = FocusScorer(
                self._state.metrics_calc.cfg, self._state.metrics_calc
            )

    def _capture_current_focus_frame(self) -> Optional[np.ndarray]:
        """截取当前画面（使用 SAF 独立截图区域）。"""
        try:
            self._ensure_focus_components()
            result = self._state.metrics_calc.capture_live()
            image_rgb = result.get("full_rgb")
            if image_rgb is not None and image_rgb.size > 0:
                return image_rgb
        except Exception as e:
            self.log(f"[聚焦] 截图失败：{e}")
        return None

    def _ensure_focus_controller(self) -> AutofocusController:
        """获取或创建 AutofocusController。"""
        if self._state.controller is not None:
            return self._state.controller

        self._ensure_focus_components()
        cfg = self._state.metrics_calc.cfg

        if self._is_virtual_mode():
            self.log("[聚焦补焦] virtual 模式：使用 FocusSimulator")
            simulator, metrics_calc, scorer, controller = create_demo_environment(
                cfg,
                peak_z=50,
                blur_scale=0.3,
                image_size=(400, 400),
                initial_z=0,
            )
            self._state.metrics_calc = metrics_calc
            self._state.scorer = scorer
            self._state.simulator = simulator
        else:
            z_axis = ZAxisController(cfg)
            controller = AutofocusController(
                cfg, self._state.scorer, self._state.metrics_calc, z_axis
            )

        controller.on_log = self.log
        controller.should_stop = self._stopped
        self._state.controller = controller
        return controller

    # ------------------------------------------------------------------
    # ROI 选择
    # ------------------------------------------------------------------

    def select_focus_roi_interactively(self) -> bool:
        """
        交互式框选补焦 ROI。

        使用 capture_frame_callback 获取当前画面，
        在 OpenCV 窗口中拖拽矩形，Enter/N 确认、R 重置、ESC/Q 取消。
        选择成功后更新 config.saf_focus_roi 与 config.saf_capture_area。
        """
        if cv2 is None:
            self.log("[聚焦ROI] OpenCV 不可用，无法交互式选择 ROI")
            return False

        try:
            self.log("[聚焦ROI] 即将弹出 ROI 选择窗口；请拖拽矩形框选补焦区域后按 Enter/N 确认")
            output_dir = self.output_root / "focus_roi_select"
            output_dir.mkdir(parents=True, exist_ok=True)

            image_rgb = self._capture_frame()
            if image_rgb is None or image_rgb.size == 0:
                self.log("[聚焦ROI] 截图失败，无法选择 ROI")
                return False

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
                        self.log("[聚焦ROI] 用户取消了 ROI 选择")
                        return False
                    if key in (ord("r"), ord("R")):
                        rect.clear()
                    if key in (13, 10, ord("n"), ord("N")):
                        if len(rect) != 2:
                            self.log("[聚焦ROI] 请先拖拽选择一个矩形区域")
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

            # 更新 SAF 专用区域
            self.config.saf_capture_area = (0, 0, w, h)
            self.config.saf_focus_roi = (roi_x, roi_y, roi_w, roi_h)

            # 重置 Focus 组件
            self._state.metrics_calc = None
            self._state.scorer = None
            self._state.controller = None
            self._state.simulator = None

            self.log(
                f"[聚焦ROI] ROI 已选择：({roi_x}, {roi_y}, {roi_w}, {roi_h})，"
                f"saf_capture_area 已同步为 (0, 0, {w}, {h})"
            )
            return True
        except Exception as e:
            self.log(f"[聚焦ROI] 交互式选择 ROI 失败：{e}")
            self.log(traceback.format_exc())
            return False

    # ------------------------------------------------------------------
    # 参考建立
    # ------------------------------------------------------------------

    def capture_focus_reference(self, cycle_index: int) -> bool:
        """
        建立聚焦参考基准图。

        在第一轮循环照明光 OFF 前调用。
        若截图失败，自动弹出 ROI 选择窗口让用户选择。
        采集后自动弹出实时窗口显示参考基准图。
        """
        self.log("========== Step 1.5：建立聚焦参考图 ==========")
        try:
            self._ensure_focus_components()
            image_rgb = self._capture_current_focus_frame()

            if image_rgb is None or image_rgb.size == 0:
                self.log("[聚焦参考] 当前 SAF 截图区域无法获取有效图像，即将弹出 ROI 选择窗口")
                if not self.select_focus_roi_interactively():
                    self.log("[聚焦参考] ROI 选择失败或用户取消，跳过参考建立")
                    return False
                image_rgb = self._capture_current_focus_frame()
                if image_rgb is None or image_rgb.size == 0:
                    self.log("[聚焦参考] ROI 选择后仍无法截图，跳过参考建立")
                    return False

            # 弹出实时窗口
            try:
                window_name = "Focus Reference Baseline Image"
                cv2.imshow(window_name, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
                self.log("[聚焦参考] 已弹出实时窗口显示参考基准图")
            except Exception as e:
                self.log(f"[聚焦参考] 实时窗口显示失败：{e}")

            ref_dir = self.output_root / "focus_reference"
            ref_dir.mkdir(parents=True, exist_ok=True)
            ref = self._state.scorer.build_reference_from_image(
                image_rgb=image_rgb,
                output_root=ref_dir,
                on_log=self.log,
            )
            self._state.reference_image = ref.get("full_rgb")
            self._state.reference_ready = True
            self._state.consecutive_low_count = 0
            self.log(
                f"[聚焦参考] 第 {cycle_index} 轮已建立，ROI="
                f"{self._state.metrics_calc.cfg.focus_roi}"
            )
            return True
        except Exception as e:
            self.log(f"[聚焦参考] 建立失败：{e}")
            self.log(traceback.format_exc())
            return False

    # ------------------------------------------------------------------
    # 评分与补焦
    # ------------------------------------------------------------------

    def compute_current_focus_score(self) -> Optional[float]:
        """截取当前画面并计算与参考图的 FocusScore_ratio。"""
        if not self._state.reference_ready or self._state.scorer is None:
            return None
        image_rgb = self._capture_current_focus_frame()
        if image_rgb is None or image_rgb.size == 0:
            return None
        roi = self._state.metrics_calc.clamp_roi(
            tuple(self._state.metrics_calc.cfg.focus_roi), image_rgb.shape
        )
        x, y, rw, rh = roi
        roi_rgb = image_rgb[y : y + rh, x : x + rw].copy()
        roi_metrics = self._state.metrics_calc.compute_for_image(roi_rgb)
        score, _ = self._state.scorer.score_ratio(roi_metrics)
        return score

    def run_autofocus_if_needed(self, cycle_index: int) -> Dict[str, Any]:
        """
        计算当前 FocusScore_ratio，若连续超出阈值则执行补焦。

        返回 {"score": float|None, "triggered": bool, "autofocus_ok": bool}。
        """
        result = {"score": None, "triggered": False, "autofocus_ok": False}
        score = self.compute_current_focus_score()
        result["score"] = score

        if score is None:
            self.log("[聚焦补焦] 当前 FocusScore 为空，跳过")
            return result

        trigger_ratio = float(self.config.focus_trigger_ratio)
        absolute = bool(self.config.focus_trigger_absolute)
        lower = min(trigger_ratio, 2.0 - trigger_ratio) if absolute else trigger_ratio
        upper = max(trigger_ratio, 2.0 - trigger_ratio) if absolute else float("inf")

        if focus_score_ratio_in_tolerance(score, trigger_ratio, absolute):
            self._state.consecutive_low_count = 0
            self.log(
                f"[聚焦补焦] FocusScore_ratio={score:.4f} 在允许区间"
                f"[{lower:.4f}, {upper:.4f}] 内，不触发补焦"
            )
            return result

        self._state.consecutive_low_count += 1
        self.log(
            f"[聚焦补焦] FocusScore_ratio={score:.4f} 超出允许区间"
            f"[{lower:.4f}, {upper:.4f}]，连续低分计数="
            f"{self._state.consecutive_low_count}/{self.config.focus_trigger_count}"
        )

        if self._state.consecutive_low_count < int(self.config.focus_trigger_count):
            return result

        result["triggered"] = True
        if not bool(self.config.focus_z_enabled):
            self.log("[聚焦补焦] 已触发但 Z 轴禁用，不执行补焦")
            return result

        if bool(self.config.focus_detection_only):
            self.log("[聚焦补焦] FocusScore检测模式：仅记录，不执行 Z 轴闭环")
            return result

        try:
            self.log("[聚焦补焦] 触发补焦，启动闭环搜索")
            controller = self._ensure_focus_controller()
            autofocus_result = controller.run_closed_loop(initial_focus_score=score)
            result["autofocus_ok"] = bool(autofocus_result.get("ok", False))
            final_score = autofocus_result.get("best_score")
            self.log(
                f"[聚焦补焦] 闭环结束：ok={result['autofocus_ok']}, "
                f"best_score={final_score}"
            )
            if final_score is not None and focus_score_ratio_in_tolerance(
                final_score, trigger_ratio, absolute
            ):
                self._state.consecutive_low_count = 0
        except Exception as e:
            self.log(f"[聚焦补焦] 闭环补焦异常：{e}")
            self.log(traceback.format_exc())

        return result

    # ------------------------------------------------------------------
    # 状态访问
    # ------------------------------------------------------------------

    @property
    def reference_ready(self) -> bool:
        """参考图是否已建立。"""
        return self._state.reference_ready

    @property
    def reference_image(self) -> Optional[np.ndarray]:
        """参考图图像。"""
        return self._state.reference_image

    @property
    def metrics_calc(self) -> Optional[FocusMetricsCalculator]:
        """FocusMetricsCalculator 实例。"""
        return self._state.metrics_calc

    @property
    def scorer(self) -> Optional[FocusScorer]:
        """FocusScorer 实例。"""
        return self._state.scorer

    @property
    def controller(self) -> Optional[AutofocusController]:
        """AutofocusController 实例。"""
        return self._state.controller

    def reset_components(self) -> None:
        """重置所有 Focus 组件（配置变更时调用）。"""
        self._state.metrics_calc = None
        self._state.scorer = None
        self._state.controller = None
        self._state.simulator = None