"""
ROI 排除区域工具模块。

用途：
  1. 把用户手动绘制的多边形 ROI 转成二值 mask；
  2. 对 SAM2/其他分割得到的 mask 应用 ROI 排除；
  3. 在图像上绘制 ROI 叠加图，便于人工核对。

设计原则：
  - 不与 SAM2、GUI、硬件耦合，便于单元测试；
  - 多个 ROI 多边形取并集；
  - 顶点越界时自动裁剪到图像边界，不抛异常；
  - 空 ROI / 空 mask 时行为保持原样（向后兼容）。
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np


RoiPolygon = List[Tuple[float, float]]


def normalize_roi_polygons(raw: Any) -> List[RoiPolygon]:
    """
    把任意可遍历的 ROI 描述归一化为 List[RoiPolygon]。

    兼容格式：
      - None / [] -> []
      - [[(x1,y1), (x2,y2), ...]]
      - [[[x1,y1], [x2,y2], ...]]
      - [{'x':x1,'y':y1}, {'x':x2,'y':y2}, ...]
    """
    if raw is None:
        return []

    polygons: List[RoiPolygon] = []

    # 如果 raw 是 numpy 数组，先转 list
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()

    if not isinstance(raw, (list, tuple)):
        return polygons

    # 如果 raw 是一维点列表（例如 [(x,y), (x,y)]），包成单个多边形
    if len(raw) > 0 and not isinstance(raw[0], (list, tuple, dict)):
        try:
            poly = _normalize_single_polygon(raw)
            if poly:
                polygons.append(poly)
        except Exception:
            pass
        return polygons

    for item in raw:
        if item is None:
            continue
        poly = _normalize_single_polygon(item)
        if poly:
            polygons.append(poly)

    return polygons


def _normalize_single_polygon(item: Any) -> Optional[RoiPolygon]:
    """把单个多边形的描述归一化为 List[Tuple[float, float]]。"""
    if isinstance(item, np.ndarray):
        item = item.tolist()

    if not isinstance(item, (list, tuple)) or len(item) == 0:
        return None

    out: RoiPolygon = []
    for p in item:
        if p is None:
            continue
        try:
            if isinstance(p, dict):
                x = float(p.get("x", p.get("X", 0.0)))
                y = float(p.get("y", p.get("Y", 0.0)))
            else:
                x, y = float(p[0]), float(p[1])
            out.append((x, y))
        except Exception:
            continue

    return out if len(out) >= 3 else None


def polygons_to_mask(image_shape_hw: Tuple[int, int], polygons: Any) -> Optional[np.ndarray]:
    """
    把多个 ROI 多边形转成二值排除 mask。

    参数
    ----------
    image_shape_hw : tuple(int, int)
        图像 (height, width)。
    polygons : Any
        多边形列表，会先调用 normalize_roi_polygons 归一化。

    返回
    -------
    np.ndarray or None
        形状 (H, W) 的 bool 数组，True 表示需要排除的像素；
        如果 polygons 为空，返回 None，表示不应用 ROI 排除。
    """
    polys = normalize_roi_polygons(polygons)
    if not polys:
        return None

    h, w = int(image_shape_hw[0]), int(image_shape_hw[1])
    if h <= 0 or w <= 0:
        return None

    mask = np.zeros((h, w), dtype=np.uint8)
    pts_list = []
    for poly in polys:
        pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
        pts_list.append(pts)

    if pts_list:
        cv2.fillPoly(mask, pts_list, 1)

    return mask.astype(bool)


def apply_roi_exclusion(
    mask: np.ndarray,
    exclude_mask: Optional[np.ndarray],
    inplace: bool = False,
) -> np.ndarray:
    """
    对 mask 应用 ROI 排除。

    参数
    ----------
    mask : np.ndarray
        输入 mask，会被转成 bool。
    exclude_mask : np.ndarray or None
        True 表示排除区域。为 None 时直接返回原 mask。
    inplace : bool
        True 时直接修改输入 mask；False 时返回副本。

    返回
    -------
    np.ndarray
        排除 ROI 后的 bool mask。
    """
    if exclude_mask is None:
        return mask if inplace else mask.copy()

    m = mask.astype(bool) if not inplace else mask.astype(bool, copy=False)
    ex = exclude_mask.astype(bool, copy=False)

    if m.shape != ex.shape:
        # 形状不一致时，尝试广播；若无法广播则只取交集区域
        h = min(m.shape[0], ex.shape[0])
        w = min(m.shape[1], ex.shape[1])
        m_crop = m[:h, :w]
        ex_crop = ex[:h, :w]
        result = m_crop.copy()
        result[ex_crop] = False
        if not inplace:
            return result
        m[:h, :w] = result
        return m

    if not inplace:
        m = m.copy()
    m[ex] = False
    return m


def draw_roi_overlay(
    image_bgr: np.ndarray,
    polygons: Any,
    color: Tuple[int, int, int] = (0, 0, 255),
    alpha: float = 0.35,
    line_thickness: int = 2,
) -> np.ndarray:
    """
    在 BGR 图像上绘制半透明 ROI 叠加图。

    参数
    ----------
    image_bgr : np.ndarray
        BGR 格式图像。
    polygons : Any
        多边形列表。
    color : tuple
        ROI 填充颜色，默认红色。
    alpha : float
        填充透明度。
    line_thickness : int
        边框线宽。

    返回
    -------
    np.ndarray
        绘制后的 BGR 图像副本。
    """
    polys = normalize_roi_polygons(polygons)
    if not polys:
        return image_bgr.copy()

    overlay = image_bgr.copy()
    canvas = image_bgr.copy()

    pts_list = []
    for poly in polys:
        pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
        pts_list.append(pts)

    if pts_list:
        cv2.fillPoly(canvas, pts_list, color)
        cv2.polylines(canvas, pts_list, isClosed=True, color=color, thickness=line_thickness)

    result = cv2.addWeighted(canvas, alpha, overlay, 1.0 - alpha, 0)

    # 标注 ROI 序号
    for i, poly in enumerate(polys):
        if not poly:
            continue
        pts_arr = np.asarray(poly, dtype=np.float32)
        center = tuple(pts_arr.mean(axis=0).astype(int))
        cv2.putText(
            result,
            f"ROI {i + 1}",
            center,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return result


def roi_mask_from_state(image_shape_hw: Tuple[int, int], state: Any) -> Optional[np.ndarray]:
    """
    从 CalibrationState / RuntimeConfig 等对象中提取 ROI 并转成 mask。

    参数
    ----------
    image_shape_hw : tuple(int, int)
        目标图像尺寸 (H, W)。
    state : Any
        包含 exclude_roi_polygons 字段的对象，或 dict。

    返回
    -------
    np.ndarray or None
        排除 mask；没有 ROI 时返回 None。
    """
    if state is None:
        return None

    if isinstance(state, dict):
        polygons = state.get("exclude_roi_polygons")
    else:
        polygons = getattr(state, "exclude_roi_polygons", None)

    return polygons_to_mask(image_shape_hw, polygons)


def select_roi_polygons_interactively(
    image_bgr: np.ndarray,
    existing_polygons: Any = None,
    window_name: str = "Draw ROI polygons: E=toggle, Left=vertex, Right/Enter=close, X=del, C=clear, ESC=cancel",
    scale: float = 0.85,
) -> List[RoiPolygon]:
    """
    用 OpenCV 窗口交互式绘制多边形 ROI 排除区域。

    按键 / 鼠标：
      - E：进入 / 退出 ROI 绘制模式；
      - ROI 模式下左键：添加当前多边形顶点；
      - ROI 模式下右键 / Enter / N：闭合当前多边形，开始下一个；
      - X：删除最后一个已闭合的多边形；
      - C：清空所有 ROI 多边形；
      - ESC：取消并抛出 RuntimeError；
      - 非 ROI 模式下按 Enter / N：结束绘制并返回当前 ROI 列表。

    参数
    ----------
    image_bgr : np.ndarray
        BGR 图像。
    existing_polygons : Any, optional
        预加载的 ROI 多边形，绘制时先显示出来。
    window_name : str
        窗口标题。
    scale : float
        显示缩放比例。

    返回
    -------
    List[RoiPolygon]
        用户绘制的多边形列表（原始图像坐标）。
    """
    if scale <= 0:
        scale = 1.0

    h, w = image_bgr.shape[:2]
    show_w = max(1, int(w * scale))
    show_h = max(1, int(h * scale))
    display = cv2.resize(image_bgr, (show_w, show_h), interpolation=cv2.INTER_AREA)

    roi_mode = False
    closed_polygons: List[List[Tuple[int, int]]] = []
    current_poly: List[Tuple[int, int]] = []
    mouse_pos: Tuple[int, int] = (0, 0)

    # 加载已有 ROI（如果有）
    for poly in normalize_roi_polygons(existing_polygons):
        closed_polygons.append([(int(x * scale), int(y * scale)) for x, y in poly])

    def to_display(x: float, y: float) -> Tuple[int, int]:
        return (int(x * scale), int(y * scale))

    def to_original(pt: Tuple[int, int]) -> Tuple[float, float]:
        return (float(pt[0]) / scale, float(pt[1]) / scale)

    def redraw() -> np.ndarray:
        canvas = display.copy()
        mode_text = "ROI" if roi_mode else "POINT"
        lines = [
            f"Mode: {mode_text} | ROIs: {len(closed_polygons)}",
            "E=toggle mode | Left=vertex | Right/Enter/N=close | X=del last | C=clear | Enter/N=finish | ESC=cancel",
        ]
        for i, line in enumerate(lines):
            cv2.putText(
                canvas,
                line,
                (18, 28 + i * 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # 已闭合 ROI
        overlay = canvas.copy()
        for idx, poly in enumerate(closed_polygons):
            if not poly:
                continue
            pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(overlay, [pts], (0, 0, 255))
            cv2.polylines(canvas, [pts], True, (0, 0, 255), 2)
            center = tuple(np.asarray(poly, dtype=np.float32).mean(axis=0).astype(int))
            cv2.putText(
                canvas,
                f"ROI {idx + 1}",
                center,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)

        # 当前正在绘制的多边形
        if roi_mode and current_poly:
            pts = np.asarray(current_poly, dtype=np.int32)
            for i in range(len(pts) - 1):
                cv2.line(canvas, tuple(pts[i]), tuple(pts[i + 1]), (0, 0, 255), 2)
            if len(pts) > 0:
                cv2.line(canvas, tuple(pts[-1]), mouse_pos, (0, 0, 255), 1, cv2.LINE_AA)
            for p in pts:
                cv2.circle(canvas, tuple(p), 4, (0, 0, 255), -1)

        return canvas

    def on_mouse(event, x, y, flags, param):
        nonlocal mouse_pos
        mouse_pos = (x, y)
        if not roi_mode:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            current_poly.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(current_poly) >= 3:
                closed_polygons.append(current_poly[:])
                current_poly.clear()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, show_w, show_h)
    cv2.setMouseCallback(window_name, on_mouse)

    try:
        while True:
            cv2.imshow(window_name, redraw())
            key = cv2.waitKey(30) & 0xFF

            if key == 27:
                raise RuntimeError("用户取消了 ROI 绘制。")

            if key in (ord("e"), ord("E")):
                roi_mode = not roi_mode
                if not roi_mode and current_poly:
                    # 退出 ROI 模式时，若当前多边形未闭合则丢弃
                    current_poly.clear()

            if roi_mode:
                if key in (13, 10, ord("n"), ord("N")):
                    if len(current_poly) >= 3:
                        closed_polygons.append(current_poly[:])
                        current_poly.clear()
                    else:
                        # 当前多边形无法闭合时，若已有 ROI 则结束绘制
                        if closed_polygons:
                            break
                if key in (ord("x"), ord("X")):
                    if current_poly:
                        current_poly.clear()
                    elif closed_polygons:
                        closed_polygons.pop()
                if key in (ord("c"), ord("C")):
                    current_poly.clear()
                    closed_polygons.clear()
            else:
                if key in (13, 10, ord("n"), ord("N")):
                    break
    finally:
        try:
            cv2.destroyWindow(window_name)
        except Exception:
            pass

    result: List[RoiPolygon] = []
    for poly in closed_polygons:
        normalized = [to_original(p) for p in poly]
        if len(normalized) >= 3:
            result.append(normalized)
    return result
