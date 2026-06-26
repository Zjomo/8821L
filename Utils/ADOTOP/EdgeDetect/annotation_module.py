import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from recognition_module import COLOR_MAP, LABEL_KEYS, contour_to_list, load_templates, save_templates


@dataclass
class TemplateState:
    template_path: Path
    templates: Dict[str, List[np.ndarray]]


class ManualAnnotator:
    def __init__(self, template_path: Path):
        self.template_path = template_path
        self.templates = load_templates(template_path)
        self.current_label = "circle"
        self.current_frame: Optional[np.ndarray] = None
        self.current_binary: Optional[np.ndarray] = None
        self.drawing = False
        self.start_pt: Optional[Tuple[int, int]] = None
        self.end_pt: Optional[Tuple[int, int]] = None
        self.window_name = "手动标注"

    def set_frame(self, frame: np.ndarray, binary: np.ndarray):
        self.current_frame = frame.copy()
        self.current_binary = binary.copy()

    def on_mouse(self, event, x, y, flags, param):
        if self.current_frame is None or self.current_binary is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_pt = (x, y)
            self.end_pt = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end_pt = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            self.end_pt = (x, y)
            self._commit_annotation()

    def _commit_annotation(self):
        assert self.current_frame is not None and self.current_binary is not None
        assert self.start_pt is not None and self.end_pt is not None
        x1, y1 = self.start_pt
        x2, y2 = self.end_pt
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])
        if x_max - x_min < 8 or y_max - y_min < 8:
            return

        roi_bin = self.current_binary[y_min:y_max, x_min:x_max]
        contours, _ = cv2.findContours(roi_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return
        contour = max(contours, key=cv2.contourArea)
        contour = contour + np.array([[[x_min, y_min]]], dtype=np.int32)
        self.templates.setdefault(self.current_label, []).append(contour)
        save_templates(self.template_path, self.templates)

    def build_canvas(self) -> np.ndarray:
        assert self.current_frame is not None
        canvas = self.current_frame.copy()
        if self.current_binary is not None:
            overlay = cv2.cvtColor(self.current_binary, cv2.COLOR_GRAY2BGR)
            canvas = cv2.addWeighted(canvas, 0.75, overlay, 0.25, 0)
        if self.drawing and self.start_pt and self.end_pt:
            color = COLOR_MAP.get(self.current_label, COLOR_MAP["manual"])
            cv2.rectangle(canvas, self.start_pt, self.end_pt, color, 2)
        cv2.putText(
            canvas,
            f"标注模式={self.current_label} | 1圆底 2长条 3旋转体 | 回车保存模板 | c清空模板 | q退出",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        return canvas

    def run(self, frame: np.ndarray, binary: np.ndarray):
        self.set_frame(frame, binary)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        while True:
            cv2.imshow(self.window_name, self.build_canvas())
            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                break
            if key in LABEL_KEYS:
                self.current_label = LABEL_KEYS[key]
            elif key == ord("c"):
                self.templates = {"circle": [], "rod": [], "rotator": []}
                save_templates(self.template_path, self.templates)
            elif key in (13, 10):
                save_templates(self.template_path, self.templates)
        cv2.destroyWindow(self.window_name)
