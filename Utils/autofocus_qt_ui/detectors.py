"""
自动检测设备资源的辅助模块。

提供：
  - 当前可见窗口列表检测
  - USB 相机接口检测
"""

from __future__ import annotations

import logging
from typing import List, Tuple

logger = logging.getLogger("autofocus_qt_ui.detectors")


def detect_windows() -> List[Tuple[str, int]]:
    """
    检测当前可见窗口，返回 [(标题, hwnd), ...]。

    优先使用 win32gui，回退到 pygetwindow。
    """
    windows: List[Tuple[str, int]] = []

    try:
        import win32gui

        def _enum(hwnd, _):
            if not win32gui.IsWindowVisible(hwnd):
                return
            text = win32gui.GetWindowText(hwnd) or ""
            if text.strip():
                windows.append((text, int(hwnd)))

        win32gui.EnumWindows(_enum, None)
        return windows
    except Exception as exc:
        logger.debug(f"win32gui 检测窗口失败: {exc}")

    try:
        import pygetwindow as gw

        for w in gw.getAllWindows():
            if w.isVisible and (w.title or "").strip():
                windows.append((w.title, 0))
        return windows
    except Exception as exc:
        logger.debug(f"pygetwindow 检测窗口失败: {exc}")

    return windows


def detect_usb_cameras(max_index: int = 10) -> List[Tuple[int, str]]:
    """
    检测可用 USB 相机接口，返回 [(索引, 描述), ...]。

    通过尝试打开 cv2.VideoCapture(index) 来判断是否可用。
    如果项目内存在 UCCFrameSource，也可扩展使用。
    """
    cameras: List[Tuple[int, str]] = []

    try:
        import cv2
    except ImportError:
        logger.warning("未安装 opencv-python，无法检测 USB 相机")
        return cameras

    for idx in range(max_index):
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap is None:
                continue
            opened = cap.isOpened()
            # 尝试读取一帧，进一步确认可用
            if opened:
                ret, _ = cap.read()
                if ret:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    desc = f"相机 {idx}"
                    if width > 0 and height > 0:
                        desc += f" ({width}x{height})"
                    cameras.append((idx, desc))
            cap.release()
        except Exception as exc:
            logger.debug(f"检测相机 {idx} 失败: {exc}")

    return cameras
