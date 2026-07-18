# sam2_screen_auto_segment.py
from __future__ import annotations

import sys
import csv
import json
import time
import random
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import cv2
import pyautogui
from PIL import Image

import torch


# ============================================================
# 1. 路径配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

SAM2_REPO_ROOT = PROJECT_ROOT / "sam2-main"

if not SAM2_REPO_ROOT.exists():
    raise FileNotFoundError(
        f"没有找到 sam2-main 文件夹：{SAM2_REPO_ROOT}\n"
        f"请确认本文件放在 06.03 文件夹下，也就是和 sam2-main 同级。"
    )

if str(SAM2_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM2_REPO_ROOT))


# ============================================================
# 2. SAM2 配置
# ============================================================

# 你现在 sam2-main 里的默认 tiny 权重路径
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
SAM2_CHECKPOINT = SAM2_REPO_ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt"

# 固定屏幕截图区域：left, top, width, height
# 这是你之前一直用的区域
CAPTURE_AREA: Tuple[int, int, int, int] = (116, 98, 1112, 886)

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "sam2_screen_auto_segment"


# ============================================================
# 3. 导入 SAM2
# ============================================================

try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except Exception as e:
    raise ImportError(
        "SAM2 导入失败。请确认你已经在当前 Python 环境中安装了 sam2：\n"
        "    pip install -e ./sam2-main\n\n"
        "并确认可以执行：\n"
        "    python -c \"import sam2; print('sam2 ok')\""
    ) from e


# ============================================================
# 4. 截图函数
# ============================================================

def capture_screen_region(
    capture_area: Tuple[int, int, int, int],
) -> np.ndarray:
    """
    截取屏幕固定区域。

    参数：
        capture_area = (left, top, width, height)

    返回：
        image_rgb: RGB 图像，np.ndarray, shape=(H, W, 3)
    """
    left, top, width, height = capture_area

    screenshot = pyautogui.screenshot(region=(left, top, width, height))
    if not isinstance(screenshot, Image.Image):
        screenshot = Image.fromarray(np.asarray(screenshot))

    image_rgb = np.asarray(screenshot.convert("RGB"))
    return image_rgb


# ============================================================
# 5. SAM2 整图自动分割
# ============================================================

def build_sam2_auto_mask_generator(
    sam2_cfg: str,
    sam2_checkpoint: Path,
    device: str = "cuda",
) -> SAM2AutomaticMaskGenerator:
    """
    创建 SAM2 自动分割器。
    """
    if not sam2_checkpoint.exists():
        raise FileNotFoundError(
            f"SAM2 checkpoint 不存在：{sam2_checkpoint}\n"
            f"请确认 checkpoints 文件夹里有 sam2.1_hiera_tiny.pt"
        )

    if device == "cuda" and not torch.cuda.is_available():
        print("[SAM2] CUDA 不可用，自动切换到 CPU。")
        device = "cpu"

    print(f"[SAM2] device={device}")
    print(f"[SAM2] cfg={sam2_cfg}")
    print(f"[SAM2] checkpoint={sam2_checkpoint}")

    sam2_model = build_sam2(
        config_file=sam2_cfg,
        ckpt_path=str(sam2_checkpoint),
        device=device,
    )

    # 自动 mask 生成器参数可以根据显存和速度调整
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,

        # 每边采样点数量。数值越大，分割越细，但速度越慢、显存占用越高。
        points_per_side=32,

        # 预测 IoU 阈值。越高越严格。
        pred_iou_thresh=0.86,

        # 稳定性阈值。越高越严格。
        stability_score_thresh=0.92,

        # 多尺度 crop。0 表示不裁剪；1 或 2 会更细，但更慢。
        crop_n_layers=1,

        # crop 后采样点下采样倍率。
        crop_n_points_downscale_factor=2,

        # 小区域去除阈值，单位 pixel。
        min_mask_region_area=50,
    )

    return mask_generator


def run_sam2_auto_segment(
    image_rgb: np.ndarray,
    mask_generator: SAM2AutomaticMaskGenerator,
) -> List[Dict[str, Any]]:
    """
    对整张图做 SAM2 自动分割。
    """
    print("[SAM2] 开始整图自动分割...")
    t0 = time.time()

    with torch.inference_mode():
        masks = mask_generator.generate(image_rgb)

    print(f"[SAM2] 分割完成：mask_count={len(masks)}, time={time.time() - t0:.3f} s")
    return masks


# ============================================================
# 6. 可视化与保存
# ============================================================

def make_random_color(seed: int) -> Tuple[int, int, int]:
    rng = random.Random(seed)
    return (
        rng.randint(40, 255),
        rng.randint(40, 255),
        rng.randint(40, 255),
    )


def overlay_masks(
    image_rgb: np.ndarray,
    masks: List[Dict[str, Any]],
    alpha: float = 0.45,
    draw_bbox: bool = True,
    draw_id: bool = True,
) -> np.ndarray:
    """
    生成彩色 mask 叠加图。
    """
    out = image_rgb.copy()
    overlay = image_rgb.copy()

    # 按面积从大到小画，避免小 mask 被大 mask 完全盖住
    masks_sorted = sorted(
        enumerate(masks),
        key=lambda x: float(x[1].get("area", 0)),
        reverse=True,
    )

    for idx, mask_info in masks_sorted:
        seg = mask_info["segmentation"].astype(bool)
        color = make_random_color(idx + 123)

        overlay[seg] = color

    out = cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0)

    if draw_bbox or draw_id:
        out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

        for idx, mask_info in enumerate(masks):
            bbox = mask_info.get("bbox", None)
            if bbox is None:
                continue

            x, y, w, h = bbox
            x, y, w, h = int(x), int(y), int(w), int(h)

            if draw_bbox:
                cv2.rectangle(
                    out_bgr,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 255),
                    1,
                )

            if draw_id:
                cv2.putText(
                    out_bgr,
                    str(idx),
                    (x, max(0, y - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        out = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

    return out


def make_label_image(
    image_shape_hw: Tuple[int, int],
    masks: List[Dict[str, Any]],
) -> np.ndarray:
    """
    生成 mask 编号图。
    背景为 0，第 1 个 mask 为 1，第 2 个 mask 为 2...
    """
    h, w = image_shape_hw
    label = np.zeros((h, w), dtype=np.uint16)

    # 大 mask 先写，小 mask 后写，保证小目标不会被覆盖
    masks_sorted = sorted(
        enumerate(masks),
        key=lambda x: float(x[1].get("area", 0)),
        reverse=True,
    )

    for idx, mask_info in masks_sorted:
        seg = mask_info["segmentation"].astype(bool)
        label[seg] = idx + 1

    return label


def save_results(
    image_rgb: np.ndarray,
    masks: List[Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, str]:
    """
    保存分割结果。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    original_path = output_dir / "screen_capture.png"
    cv2.imwrite(str(original_path), image_bgr)

    overlay_rgb = overlay_masks(image_rgb, masks)
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    overlay_path = output_dir / "sam2_overlay.png"
    cv2.imwrite(str(overlay_path), overlay_bgr)

    label_img = make_label_image(image_rgb.shape[:2], masks)
    label_path = output_dir / "sam2_label_uint16.png"
    cv2.imwrite(str(label_path), label_img)

    csv_path = output_dir / "sam2_masks_summary.csv"
    json_path = output_dir / "sam2_masks_summary.json"

    summary_rows: List[Dict[str, Any]] = []

    for idx, mask_info in enumerate(masks):
        seg = mask_info["segmentation"].astype(bool)
        mask_u8 = (seg.astype(np.uint8) * 255)

        mask_path = masks_dir / f"mask_{idx:04d}.png"
        cv2.imwrite(str(mask_path), mask_u8)

        bbox = mask_info.get("bbox", [None, None, None, None])
        point_coords = mask_info.get("point_coords", None)

        row = {
            "id": idx,
            "area": float(mask_info.get("area", 0)),
            "bbox_x": bbox[0] if bbox is not None else None,
            "bbox_y": bbox[1] if bbox is not None else None,
            "bbox_w": bbox[2] if bbox is not None else None,
            "bbox_h": bbox[3] if bbox is not None else None,
            "predicted_iou": float(mask_info.get("predicted_iou", 0)),
            "stability_score": float(mask_info.get("stability_score", 0)),
            "crop_box": mask_info.get("crop_box", None),
            "point_coords": point_coords,
            "mask_path": str(mask_path),
        }
        summary_rows.append(row)

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "id",
            "area",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
            "predicted_iou",
            "stability_score",
            "crop_box",
            "point_coords",
            "mask_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    print(f"[保存] 原始截图: {original_path}")
    print(f"[保存] 彩色叠加图: {overlay_path}")
    print(f"[保存] mask编号图: {label_path}")
    print(f"[保存] mask单图目录: {masks_dir}")
    print(f"[保存] CSV统计: {csv_path}")
    print(f"[保存] JSON统计: {json_path}")

    return {
        "original_path": str(original_path),
        "overlay_path": str(overlay_path),
        "label_path": str(label_path),
        "masks_dir": str(masks_dir),
        "csv_path": str(csv_path),
        "json_path": str(json_path),
    }


# ============================================================
# 7. 主函数
# ============================================================

def main():
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / run_name

    print("========== SAM2 屏幕固定区域整图分割 ==========")
    print(f"[路径] PROJECT_ROOT={PROJECT_ROOT}")
    print(f"[路径] SAM2_REPO_ROOT={SAM2_REPO_ROOT}")
    print(f"[截图] capture_area={CAPTURE_AREA}")
    print(f"[输出] output_dir={output_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    mask_generator = build_sam2_auto_mask_generator(
        sam2_cfg=SAM2_CFG,
        sam2_checkpoint=SAM2_CHECKPOINT,
        device=device,
    )

    image_rgb = capture_screen_region(CAPTURE_AREA)

    print(f"[截图] image shape={image_rgb.shape}")

    masks = run_sam2_auto_segment(
        image_rgb=image_rgb,
        mask_generator=mask_generator,
    )

    if len(masks) == 0:
        print("[结果] 没有生成任何 mask。")
    else:
        print(f"[结果] 共生成 {len(masks)} 个 mask。")

    save_results(
        image_rgb=image_rgb,
        masks=masks,
        output_dir=output_dir,
    )

    print("========== 完成 ==========")


if __name__ == "__main__":
    main()