"""
聚焦指标计算模块。

从 measurement_autofocus_shg_closed_loop.py 的以下方法提取：
  - compute_focus_metrics_for_image()
  - capture_focus_metrics_live()
  - capture_and_compute_focus_metrics()
  - _profile_edge_and_halo_width()
  - _safe_float()
  - _clamp_roi()
  - _flatten_focus_metrics()
"""

from __future__ import annotations

import math
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import pyautogui
except ImportError:
    pyautogui = None

try:
    import pygetwindow as gw
except ImportError:
    gw = None

try:
    import win32gui
    import win32ui
    import win32con
    import win32api
    import win32process
except ImportError:
    win32gui = None
    win32ui = None
    win32con = None
    win32api = None
    win32process = None

try:
    from PIL import Image
except ImportError:
    Image = None

from .config import AutofocusConfig

logger = logging.getLogger(__name__)


# 聚焦指标名称列表（顺序固定）
FOCUS_METRIC_NAMES = [
    "tenengrad",
    "laplacian_var",
    "brenner",
    "highfreq_ratio",
    "local_contrast",
    "edge_width",
    "halo_width",
    "brightness_mean",
    "brightness_std",
    "red_blue_ratio",
    "modified_laplacian",
    "dct_energy",
    "smd",
    "entropy",
]

FOCUS_SUMMARY_KEYS = (
    [f"full_{name}" for name in FOCUS_METRIC_NAMES]
    + [f"roi_{name}" for name in FOCUS_METRIC_NAMES]
)


class FocusMetricsCalculator:
    """
    聚焦指标计算器。

    功能：
      1. 对单张 RGB 图像计算 10 个聚焦/亮度/颜色指标；
      2. 截屏并计算整图和 ROI 区域的指标；
      3. 保存截图。
    """

    METRIC_NAMES = FOCUS_METRIC_NAMES
    SUMMARY_KEYS = FOCUS_SUMMARY_KEYS

    def __init__(self, cfg: AutofocusConfig):
        self.cfg = cfg
        self._usb_source = None

    def close(self) -> None:
        """释放 USB 相机等资源。"""
        if self._usb_source is not None:
            try:
                self._usb_source.release()
            except Exception:
                pass
            self._usb_source = None

    @staticmethod
    def _find_window_rect(
        window_title: str,
        match_mode: str = "contains",
    ) -> Tuple[Optional[int], Optional[Tuple[int, int, int, int]]]:
        title = (window_title or "").strip()
        if not title:
            raise ValueError("window_title 不能为空")

        if win32gui is not None:
            found = []

            def _enum(hwnd, _):
                if not win32gui.IsWindowVisible(hwnd):
                    return
                text = win32gui.GetWindowText(hwnd) or ""
                ok = text == title if match_mode == "exact" else title.lower() in text.lower()
                if ok:
                    found.append(hwnd)

            win32gui.EnumWindows(_enum, None)
            if found:
                hwnd = found[0]
                return hwnd, tuple(int(v) for v in win32gui.GetWindowRect(hwnd))

        if gw is not None:
            windows = gw.getWindowsWithTitle(title)
            if match_mode == "contains":
                windows = [w for w in windows if title.lower() in (w.title or "").lower()]
            elif match_mode == "exact":
                windows = [w for w in windows if (w.title or "") == title]
            if windows:
                w = windows[0]
                return None, (int(w.left), int(w.top), int(w.right), int(w.bottom))

        return None, None

    @staticmethod
    def _capture_window_by_hwnd(
        hwnd: int,
        rect: Tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        if win32gui is None or win32ui is None or win32con is None:
            return None
        left, top, right, bottom = [int(v) for v in rect]
        width = max(1, right - left)
        height = max(1, bottom - top)
        hwnd_dc = mfc_dc = save_dc = bitmap = None
        try:
            hwnd_dc = win32gui.GetWindowDC(hwnd)
            mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
            save_dc = mfc_dc.CreateCompatibleDC()
            bitmap = win32ui.CreateBitmap()
            bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
            save_dc.SelectObject(bitmap)

            ok = win32gui.PrintWindow(hwnd, save_dc.GetSafeHdc(), 2)
            if not ok:
                ok = win32gui.PrintWindow(hwnd, save_dc.GetSafeHdc(), 1)
            if not ok:
                ok = win32gui.PrintWindow(hwnd, save_dc.GetSafeHdc(), 0)
            if not ok:
                ok = win32gui.BitBlt(
                    save_dc.GetSafeHdc(), 0, 0, width, height,
                    hwnd_dc, 0, 0, win32con.SRCCOPY,
                )
            if not ok:
                return None

            bmp_info = bitmap.GetInfo()
            bmp_bytes = bitmap.GetBitmapBits(True)
            image = np.frombuffer(bmp_bytes, dtype=np.uint8)
            image.shape = (int(bmp_info["bmHeight"]), int(bmp_info["bmWidth"]), 4)
            return image[:, :, :3][:, :, ::-1].copy()
        except Exception:
            return None
        finally:
            try:
                if bitmap is not None:
                    win32gui.DeleteObject(bitmap.GetHandle())
                if save_dc is not None:
                    save_dc.DeleteDC()
                if mfc_dc is not None:
                    mfc_dc.DeleteDC()
                if hwnd_dc is not None:
                    win32gui.ReleaseDC(hwnd, hwnd_dc)
            except Exception:
                pass

    @staticmethod
    def _capture_window_region(
        window_title: str,
        match_mode: str = "contains",
        padding: Tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        if pyautogui is None:
            raise ImportError("需要安装 pyautogui：pip install pyautogui")

        hwnd, rect = FocusMetricsCalculator._find_window_rect(window_title, match_mode=match_mode)
        if rect is None:
            raise RuntimeError(f"未找到窗口: {window_title}")

        left, top, right, bottom = [int(v) for v in rect]
        pad_l, pad_t, pad_r, pad_b = [int(v) for v in (padding or (0, 0, 0, 0))]
        left = max(0, left + pad_l)
        top = max(0, top + pad_t)
        right = max(left + 1, right - pad_r)
        bottom = max(top + 1, bottom - pad_b)
        width = right - left
        height = bottom - top
        if width < 8 or height < 8:
            raise RuntimeError(
                f"窗口有效区域过小: {width}x{height}，可能窗口已最小化或隐藏"
            )

        image_rgb = None
        if hwnd is not None:
            full_rgb = FocusMetricsCalculator._capture_window_by_hwnd(hwnd, rect)
            if full_rgb is not None:
                rel_left = max(0, left - int(rect[0]))
                rel_top = max(0, top - int(rect[1]))
                image_rgb = full_rgb[rel_top : rel_top + height, rel_left : rel_left + width].copy()

        if image_rgb is None or image_rgb.size == 0:
            if hwnd is not None:
                logger.warning(
                    f"[窗口捕获] 已定位窗口 '{window_title}' 但 Win32 API 捕获失败，"
                    f"尝试激活并置顶目标窗口后屏幕截图"
                )
                image_rgb = FocusMetricsCalculator._capture_by_topping_window(
                    hwnd, left, top, width, height
                )

            if image_rgb is None or image_rgb.size == 0:
                if hwnd is not None:
                    raise RuntimeError(
                        f"已找到窗口 '{window_title}'，但无法激活/捕获该窗口。"
                        f"请确认窗口未最小化、未被系统权限隔离，并尽量不要选择资源管理器/系统窗口作为显微镜图像源。"
                    )
                screenshot = pyautogui.screenshot(region=(left, top, width, height))
                image_rgb = np.asarray(screenshot.convert("RGB"))

        logger.info(
            f"[窗口捕获] 成功捕获指定窗口 '{window_title}', "
            f"hwnd={hwnd}, rect=({left},{top},{width},{height})"
        )
        return image_rgb, (left, top, width, height)

    @staticmethod
    def _capture_by_topping_window(
        hwnd: int,
        left: int,
        top: int,
        width: int,
        height: int,
        settle_s: float = 0.4,
    ) -> Optional[np.ndarray]:
        """临时激活并置顶目标窗口，截图后恢复置顶状态。"""
        if win32gui is None or win32con is None or pyautogui is None:
            logger.warning("[窗口捕获] 缺少 win32gui/win32con/pyautogui，无法置顶截图")
            return None
        was_topmost = bool(
            win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE) & win32con.WS_EX_TOPMOST
        )
        try:
            # 恢复/显示窗口
            if win32gui.IsIconic(hwnd):
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            else:
                win32gui.ShowWindow(hwnd, win32con.SW_SHOW)

            # 使用 mouse_event 技巧绕过 Windows 前台窗口锁定
            try:
                import ctypes
                ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)  # MOUSEEVENTF_RIGHTUP
            except Exception as exc:
                logger.debug(f"[窗口捕获] mouse_event 失败: {exc}")

            # 附加线程输入以允许前台切换
            if win32api is not None and win32process is not None:
                fg_hwnd = win32gui.GetForegroundWindow()
                current_tid = win32api.GetCurrentThreadId()
                target_tid = win32process.GetWindowThreadProcessId(hwnd)[0]
                fg_tid = win32process.GetWindowThreadProcessId(fg_hwnd)[0] if fg_hwnd else 0
                try:
                    if fg_tid:
                        win32process.AttachThreadInput(current_tid, fg_tid, True)
                    win32process.AttachThreadInput(current_tid, target_tid, True)
                    win32gui.BringWindowToTop(hwnd)
                    win32gui.SetForegroundWindow(hwnd)
                    win32gui.SetActiveWindow(hwnd)
                except Exception as exc:
                    logger.warning(f"[窗口捕获] 前台切换异常: {exc}")
                finally:
                    try:
                        win32process.AttachThreadInput(current_tid, target_tid, False)
                        if fg_tid:
                            win32process.AttachThreadInput(current_tid, fg_tid, False)
                    except Exception:
                        pass

            # 置顶窗口
            win32gui.SetWindowPos(
                hwnd,
                win32con.HWND_TOPMOST,
                0, 0, 0, 0,
                win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW,
            )
            win32gui.SetWindowPos(
                hwnd,
                win32con.HWND_TOP,
                0, 0, 0, 0,
                win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW,
            )

            # 等待窗口完全渲染（硬件加速渲染需要更长时间）
            if settle_s > 0:
                time.sleep(settle_s)

            # 验证窗口是否在前台
            fg = win32gui.GetForegroundWindow()
            if fg != hwnd:
                logger.warning(
                    f"[窗口捕获] 窗口 '{win32gui.GetWindowText(hwnd)}' (hwnd={hwnd}) "
                    f"未能成功置顶前台，当前前台 hwnd={fg}"
                )

            logger.info(
                f"[窗口捕获] 开始屏幕截图 region=({left},{top},{width},{height})"
            )
            screenshot = pyautogui.screenshot(region=(left, top, width, height))
            img = np.asarray(screenshot.convert("RGB"))
            logger.info(f"[窗口捕获] 屏幕截图完成, shape={img.shape}")
            return img
        except Exception as exc:
            logger.error(f"[窗口捕获] 置顶截图异常: {exc}", exc_info=True)
            return None
        finally:
            try:
                if not was_topmost:
                    win32gui.SetWindowPos(
                        hwnd,
                        win32con.HWND_NOTOPMOST,
                        0, 0, 0, 0,
                        win32con.SWP_NOMOVE | win32con.SWP_NOSIZE,
                    )
            except Exception:
                pass

    @staticmethod
    def safe_float(value: Any) -> Optional[float]:
        """安全转 float，非有限值返回 None。"""
        try:
            v = float(value)
            if math.isfinite(v):
                return v
        except Exception:
            pass
        return None

    @staticmethod
    def clamp_roi(
        roi: Tuple[int, int, int, int],
        image_shape: Tuple[int, int, int],
    ) -> Tuple[int, int, int, int]:
        """将 ROI 裁剪到图像边界内。"""
        h, w = int(image_shape[0]), int(image_shape[1])
        try:
            x, y, rw, rh = [int(v) for v in roi]
        except Exception:
            x, y, rw, rh = 0, 0, min(300, w), min(300, h)

        x = max(0, min(x, max(0, w - 1)))
        y = max(0, min(y, max(0, h - 1)))
        rw = max(1, min(rw, w - x))
        rh = max(1, min(rh, h - y))
        return x, y, rw, rh

    @staticmethod
    def _profile_edge_and_halo_width(
        gray: np.ndarray,
        half_width: int = 60,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        从灰度图中估计 edge_width 和 halo_width。

        edge_width：沿最强边缘法线方向的 10%→90% 灰度过渡宽度，单位 px。
        halo_width：最强边缘附近一阶梯度主峰宽度，使用 25% 峰值阈值估计，单位 px。
        """
        try:
            if gray is None or gray.size < 25:
                return None, None

            arr = gray.astype(np.float64)
            h, w = arr.shape[:2]
            if h < 5 or w < 5:
                return None, None

            prof_x = np.mean(arr, axis=0)
            prof_y = np.mean(arr, axis=1)

            grad_x = np.abs(np.gradient(prof_x)) if len(prof_x) >= 3 else np.array([])
            grad_y = np.abs(np.gradient(prof_y)) if len(prof_y) >= 3 else np.array([])

            if grad_x.size == 0 and grad_y.size == 0:
                return None, None

            max_x = float(np.max(grad_x)) if grad_x.size else -1.0
            max_y = float(np.max(grad_y)) if grad_y.size else -1.0

            if max_x >= max_y:
                prof = prof_x
                grad = grad_x
            else:
                prof = prof_y
                grad = grad_y

            if grad.size == 0 or float(np.max(grad)) <= 1e-12:
                return None, None

            edge_idx = int(np.argmax(grad))
            n = len(prof)
            hw = max(8, int(half_width))
            left = max(0, edge_idx - hw)
            right = min(n, edge_idx + hw + 1)
            local_prof = prof[left:right]
            local_grad = grad[left:right]
            local_edge = edge_idx - left

            if len(local_prof) < 5:
                return None, None

            pre = local_prof[:max(1, local_edge)]
            post = local_prof[min(len(local_prof), local_edge + 1):]
            if len(pre) < 2 or len(post) < 2:
                low_val = float(np.percentile(local_prof, 10))
                high_val = float(np.percentile(local_prof, 90))
            else:
                a = float(np.median(pre))
                b = float(np.median(post))
                low_val, high_val = (a, b) if a <= b else (b, a)

            if abs(high_val - low_val) <= 1e-9:
                edge_width = None
            else:
                p10 = low_val + 0.10 * (high_val - low_val)
                p90 = low_val + 0.90 * (high_val - low_val)
                lo, hi = sorted((p10, p90))
                idxs = np.where((local_prof >= lo) & (local_prof <= hi))[0]
                edge_width = float(idxs[-1] - idxs[0] + 1) if idxs.size >= 2 else None

            gmax = float(np.max(local_grad))
            if gmax <= 1e-12:
                halo_width = None
            else:
                mask = local_grad >= (0.25 * gmax)
                center = int(np.argmax(local_grad))
                l = center
                r = center
                while l - 1 >= 0 and mask[l - 1]:
                    l -= 1
                while r + 1 < len(mask) and mask[r + 1]:
                    r += 1
                halo_width = float(r - l + 1)

            return edge_width, halo_width
        except Exception:
            return None, None

    def compute_for_image(
        self,
        image_rgb: np.ndarray,
    ) -> Dict[str, Optional[float]]:
        """计算一张 RGB 图像的 10 个聚焦/亮度/颜色指标。"""
        if cv2 is None:
            raise ImportError("需要安装 opencv-python：pip install opencv-python")

        if image_rgb is None or image_rgb.size == 0:
            return {name: None for name in FOCUS_METRIC_NAMES}

        rgb = np.asarray(image_rgb)
        if rgb.ndim == 2:
            gray_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
            rgb = np.stack([gray_u8, gray_u8, gray_u8], axis=-1)
        else:
            rgb = np.clip(rgb[:, :, :3], 0, 255).astype(np.uint8)
            gray_u8 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        gray = gray_u8.astype(np.float64)
        eps = 1e-12

        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = float(np.mean(gx * gx + gy * gy))

        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        laplacian_var = float(np.var(lap))

        brenner_terms = []
        if gray.shape[1] > 2:
            brenner_terms.append((gray[:, 2:] - gray[:, :-2]) ** 2)
        if gray.shape[0] > 2:
            brenner_terms.append((gray[2:, :] - gray[:-2, :]) ** 2)
        brenner = (
            float(np.mean([np.mean(v) for v in brenner_terms]))
            if brenner_terms
            else None
        )

        try:
            f = np.fft.fftshift(np.fft.fft2(gray - np.mean(gray)))
            power = np.abs(f) ** 2
            h, w = gray.shape
            yy, xx = np.ogrid[:h, :w]
            cy, cx = h // 2, w // 2
            radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            low_radius = max(
                1.0,
                float(self.cfg.focus_fft_low_radius_ratio) * min(h, w),
            )
            high_mask = radius > low_radius
            total_energy = float(np.sum(power))
            highfreq_ratio = float(np.sum(power[high_mask]) / (total_energy + eps))
        except Exception:
            highfreq_ratio = None

        try:
            local_mean = cv2.blur(gray, (15, 15))
            local_mean_sq = cv2.blur(gray * gray, (15, 15))
            local_var = np.maximum(local_mean_sq - local_mean * local_mean, 0.0)
            local_std = np.sqrt(local_var)
            local_contrast = float(np.mean(local_std) / (np.mean(gray) + eps))
        except Exception:
            local_contrast = None

        edge_width, halo_width = self._profile_edge_and_halo_width(
            gray,
            half_width=int(self.cfg.focus_edge_profile_half_width),
        )

        brightness_mean = float(np.mean(gray))
        brightness_std = float(np.std(gray))

        r_mean = float(np.mean(rgb[:, :, 0]))
        b_mean = float(np.mean(rgb[:, :, 2]))
        red_blue_ratio = float(r_mean / (b_mean + eps))

        # ---- 新增常用聚焦指标 ----
        try:
            dxx = cv2.Sobel(gray, cv2.CV_64F, 2, 0, ksize=3)
            dyy = cv2.Sobel(gray, cv2.CV_64F, 0, 2, ksize=3)
            modified_laplacian = float(np.mean(np.abs(dxx) + np.abs(dyy)))
        except Exception:
            modified_laplacian = None

        try:
            h_dct, w_dct = gray.shape
            dct_input = gray.astype(np.float32)
            # cv2.dct 对任意尺寸均支持，但 2 的幂次更高效
            dct = cv2.dct(dct_input)
            power_dct = dct * dct
            yy_d, xx_d = np.ogrid[:h_dct, :w_dct]
            low_radius_dct = max(1.0, float(self.cfg.focus_fft_low_radius_ratio) * min(h_dct, w_dct))
            high_mask_dct = np.sqrt((yy_d) ** 2 + (xx_d) ** 2) > low_radius_dct
            total_energy_dct = float(np.sum(power_dct))
            dct_energy = float(np.sum(power_dct[high_mask_dct]) / (total_energy_dct + eps))
        except Exception:
            dct_energy = None

        try:
            smd = (
                float(np.mean(np.abs(gray[:, 1:] - gray[:, :-1])))
                + float(np.mean(np.abs(gray[1:, :] - gray[:-1, :])))
            ) / 2.0
        except Exception:
            smd = None

        try:
            hist, _ = np.histogram(gray_u8, bins=256, range=(0, 256), density=True)
            p = hist[hist > 0]
            entropy = float(-np.sum(p * np.log2(p))) if p.size else 0.0
        except Exception:
            entropy = None

        values = {
            "tenengrad": tenengrad,
            "laplacian_var": laplacian_var,
            "brenner": brenner,
            "highfreq_ratio": highfreq_ratio,
            "local_contrast": local_contrast,
            "edge_width": edge_width,
            "halo_width": halo_width,
            "brightness_mean": brightness_mean,
            "brightness_std": brightness_std,
            "red_blue_ratio": red_blue_ratio,
            "modified_laplacian": modified_laplacian,
            "dct_energy": dct_energy,
            "smd": smd,
            "entropy": entropy,
        }
        return {k: self.safe_float(v) for k, v in values.items()}

    def _capture_usb_frame(self) -> np.ndarray:
        """从 USB 相机读取一帧 RGB 图像，会复用已打开的 _usb_source。"""
        # 优先使用项目自定义的 UCCFrameSource；如果不可用，回退到 OpenCV
        try:
            from SpotZoom import UCCFrameSource
            has_ucc = True
        except Exception as exc:
            UCCFrameSource = None
            has_ucc = False

        if has_ucc:
            if self._usb_source is None or not self._usb_source.is_healthy():
                if self._usb_source is not None:
                    try:
                        self._usb_source.release()
                    except Exception:
                        pass
                pixel_format = self.cfg.usb_pixel_format
                if pixel_format is not None:
                    pixel_format = pixel_format.upper()
                    if pixel_format == "AUTO":
                        pixel_format = None
                self._usb_source = UCCFrameSource(
                    device_index=int(self.cfg.usb_device_index),
                    resolution=self.cfg.usb_resolution.upper() if self.cfg.usb_resolution else "AUTO",
                    pixel_format=pixel_format,
                )

            frame_bgr = self._usb_source.grab_frame()
            if frame_bgr is None or frame_bgr.size == 0:
                raise RuntimeError("USB 相机帧采集失败")
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # OpenCV 回退
        if cv2 is None:
            raise ImportError("需要安装 opencv-python：pip install opencv-python")
        if self._usb_source is None:
            self._usb_source = cv2.VideoCapture(int(self.cfg.usb_device_index), cv2.CAP_DSHOW)
            if not self._usb_source.isOpened():
                self._usb_source.release()
                self._usb_source = None
                raise RuntimeError(f"无法打开 USB 相机索引 {self.cfg.usb_device_index}")
        ret, frame_bgr = self._usb_source.read()
        if not ret or frame_bgr is None or frame_bgr.size == 0:
            raise RuntimeError("USB 相机帧采集失败")
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def capture_live(self) -> Dict[str, Any]:
        """截图并计算整图/ROI 指标，不保存图片。"""
        if self.cfg.capture_mode == "window_roi":
            image_rgb, capture_rect = self._capture_window_region(
                self.cfg.window_title,
                match_mode=getattr(self.cfg, "window_match_mode", "contains"),
                padding=getattr(self.cfg, "window_padding", (0, 0, 0, 0)),
            )
            left, top, width, height = capture_rect
        elif self.cfg.capture_mode == "usb_camera":
            image_rgb = self._capture_usb_frame()
            height, width = image_rgb.shape[:2]
            left, top = 0, 0
        else:
            if pyautogui is None:
                raise ImportError("需要安装 pyautogui：pip install pyautogui")
            left, top, width, height = [int(v) for v in self.cfg.capture_area]
            screenshot = pyautogui.screenshot(region=(left, top, width, height))
            image_rgb = np.asarray(screenshot.convert("RGB"))

        roi = self.clamp_roi(tuple(self.cfg.focus_roi), image_rgb.shape)
        x, y, rw, rh = roi
        roi_rgb = image_rgb[y : y + rh, x : x + rw].copy()

        full_metrics = self.compute_for_image(image_rgb)
        roi_metrics = self.compute_for_image(roi_rgb)

        return {
            "ok": True,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "capture_area": [left, top, width, height],
            "roi": [x, y, rw, rh],
            "full": full_metrics,
            "roi_metrics": roi_metrics,
            "full_rgb": image_rgb,
            "roi_rgb": roi_rgb,
        }

    def capture_and_save(
        self,
        cycle_index: int,
        save_dir: Union[str, Path],
        on_log: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """截图、计算指标、保存整图和 ROI 截图。"""
        if pyautogui is None:
            raise ImportError("需要安装 pyautogui：pip install pyautogui")
        if Image is None:
            raise ImportError("需要安装 pillow：pip install pillow")

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_filename = f"cycle_{cycle_index:04d}_{timestamp}_focus_full.png"
        roi_filename = f"cycle_{cycle_index:04d}_{timestamp}_focus_roi.png"
        full_path = save_dir / full_filename
        roi_path = save_dir / roi_filename

        _log = on_log or logger.info
        if self.cfg.capture_mode == "window_roi":
            image_rgb, capture_rect = self._capture_window_region(
                self.cfg.window_title,
                match_mode=getattr(self.cfg, "window_match_mode", "contains"),
                padding=getattr(self.cfg, "window_padding", (0, 0, 0, 0)),
            )
            left, top, width, height = capture_rect
            _log(
                f"[聚焦] 窗口ROI capture window_title={self.cfg.window_title}, "
                f"rect=({left}, {top}, {width}, {height})"
            )
        elif self.cfg.capture_mode == "usb_camera":
            image_rgb = self._capture_usb_frame()
            height, width = image_rgb.shape[:2]
            left, top = 0, 0
            _log(
                f"[聚焦] USB capture device={self.cfg.usb_device_index}, "
                f"resolution={self.cfg.usb_resolution}, rect=({left}, {top}, {width}, {height})"
            )
        else:
            left, top, width, height = [int(v) for v in self.cfg.capture_area]
            _log(
                f"[聚焦] 截图区域 capture_area=({left}, {top}, {width}, {height})"
            )
            screenshot = pyautogui.screenshot(region=(left, top, width, height))
            image_rgb = np.asarray(screenshot.convert("RGB"))

        roi = self.clamp_roi(tuple(self.cfg.focus_roi), image_rgb.shape)
        x, y, rw, rh = roi
        roi_rgb = image_rgb[y : y + rh, x : x + rw].copy()

        full_metrics = self.compute_for_image(image_rgb)
        roi_metrics = self.compute_for_image(roi_rgb)

        Image.fromarray(image_rgb).save(full_path)
        Image.fromarray(roi_rgb).save(roi_path)

        result = {
            "ok": True,
            "cycle_index": int(cycle_index),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "capture_area": [left, top, width, height],
            "roi": [x, y, rw, rh],
            "full_image_path": str(full_path),
            "roi_image_path": str(roi_path),
            "full": full_metrics,
            "roi_metrics": roi_metrics,
        }

        _log(
            f"[聚焦] 完成: full_tenengrad={full_metrics.get('tenengrad')}, "
            f"roi_tenengrad={roi_metrics.get('tenengrad')}, roi=({x},{y},{rw},{rh})"
        )
        return result

    @staticmethod
    def flatten_metrics(focus_metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """将 capture_and_save() 返回的嵌套指标展平为一层 dict。"""
        flat = {key: None for key in FOCUS_SUMMARY_KEYS}
        if not isinstance(focus_metrics, dict):
            return flat
        full = focus_metrics.get("full") or {}
        roi = focus_metrics.get("roi_metrics") or {}
        for name in FOCUS_METRIC_NAMES:
            if isinstance(full, dict):
                flat[f"full_{name}"] = full.get(name)
            if isinstance(roi, dict):
                flat[f"roi_{name}"] = roi.get(name)
        return flat
