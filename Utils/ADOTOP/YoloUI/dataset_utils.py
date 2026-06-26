"""
数据集准备工具模块。

支持格式：
  - Ultralytics YOLO Segmentation 1.0 (YOLOv8 分割)
  - Ultralytics YOLO Detection 1.0 (YOLOv8 检测)
  - YOLO 1.1 (CVAT 导出格式：images/ + labels/ + classes.txt)
  - COCO 1.0 (CVAT 导出：annotations/instances_default.json + images/)

输出：
  - YOLO 标准目录结构 images/train val test, labels/train val test
  - data.yaml
  - train.txt / val.txt / test.txt
"""

import os
import shutil
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Optional

random.seed(42)


class DatasetSample:
    """统一的数据样本表示。"""
    def __init__(self, image_path: str, label_path: Optional[str] = None,
                 annotations: Optional[List[dict]] = None, is_segment: bool = True):
        self.image_path = image_path
        self.label_path = label_path  # YOLO txt 路径或 COCO 标注
        self.annotations = annotations or []
        self.is_segment = is_segment


def _list_images(folder: str) -> List[str]:
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')
    return sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(exts)
    ])


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _copy_files(src_list: List[str], dst_dir: str):
    _ensure_dir(dst_dir)
    for src in src_list:
        if src and os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, os.path.basename(src)))


def detect_format(image_dir: str, label_dir: str) -> str:
    """根据目录内容猜测格式。"""
    if not os.path.isdir(label_dir):
        return "unknown"
    files = os.listdir(label_dir)
    if any(f.endswith('.json') for f in files):
        return "coco"
    if any(f.endswith('.xml') for f in files):
        return "pascal_voc"
    if 'classes.txt' in files or any(f.endswith('.txt') for f in files):
        # 检查 txt 内容判断 seg 或 det
        txt_files = [f for f in files if f.endswith('.txt')]
        if txt_files:
            sample = os.path.join(label_dir, txt_files[0])
            with open(sample, 'r') as f:
                line = f.readline().strip()
                if line:
                    parts = line.split()
                    if len(parts) > 5:
                        return "yolo_seg"
                    elif len(parts) == 5:
                        return "yolo_det"
        return "yolo_11"
    return "unknown"


def parse_yolo_seg(image_dir: str, label_dir: str) -> Tuple[List[DatasetSample], List[str]]:
    """解析 Ultralytics YOLO Segmentation / YOLO 1.1 分割格式。"""
    samples = []
    names = []
    classes_file = os.path.join(label_dir, 'classes.txt')
    if os.path.exists(classes_file):
        with open(classes_file, 'r', encoding='utf-8') as f:
            names = [line.strip() for line in f if line.strip()]

    for img_path in _list_images(image_dir):
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, base + '.txt')
        samples.append(DatasetSample(img_path, label_path if os.path.exists(label_path) else None, is_segment=True))
    return samples, names


def parse_yolo_det(image_dir: str, label_dir: str) -> Tuple[List[DatasetSample], List[str]]:
    """解析 Ultralytics YOLO Detection 格式。"""
    samples = []
    names = []
    classes_file = os.path.join(label_dir, 'classes.txt')
    if os.path.exists(classes_file):
        with open(classes_file, 'r', encoding='utf-8') as f:
            names = [line.strip() for line in f if line.strip()]

    for img_path in _list_images(image_dir):
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, base + '.txt')
        samples.append(DatasetSample(img_path, label_path if os.path.exists(label_path) else None, is_segment=False))
    return samples, names


def parse_coco(image_dir: str, annotation_file: str) -> Tuple[List[DatasetSample], List[str]]:
    """解析 COCO 1.0 格式。"""
    with open(annotation_file, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    categories = {cat['id']: cat['name'] for cat in coco.get('categories', [])}
    names = [categories.get(i, f'class_{i}') for i in sorted(categories.keys())]

    img_id_to_path = {img['id']: img['file_name'] for img in coco.get('images', [])}
    img_id_to_anns = {}
    for ann in coco.get('annotations', []):
        img_id_to_anns.setdefault(ann['image_id'], []).append(ann)

    samples = []
    for img_id, file_name in img_id_to_path.items():
        img_path = os.path.join(image_dir, file_name)
        if not os.path.exists(img_path):
            continue
        anns = img_id_to_anns.get(img_id, [])
        is_segment = any('segmentation' in ann and ann['segmentation'] for ann in anns)
        samples.append(DatasetSample(img_path, annotations=anns, is_segment=is_segment))
    return samples, names


def coco_to_yolo(samples: List[DatasetSample], output_label_dir: str, image_wh_map: dict, names: List[str]):
    """将 COCO 标注转换为 YOLO txt 格式。"""
    _ensure_dir(output_label_dir)
    for s in samples:
        base = os.path.splitext(os.path.basename(s.image_path))[0]
        out_path = os.path.join(output_label_dir, base + '.txt')
        w, h = image_wh_map.get(s.image_path, (1, 1))
        lines = []
        for ann in s.annotations:
            cat_id = ann['category_id']
            class_idx = list(names).index(cat_id) if isinstance(names, dict) else cat_id
            if s.is_segment and 'segmentation' in ann and ann['segmentation']:
                seg = ann['segmentation']
                if isinstance(seg, list):
                    poly = seg[0]
                elif isinstance(seg, dict):
                    # RLE 格式暂不支持
                    continue
                else:
                    continue
                norm = [str(class_idx)]
                for i in range(0, len(poly), 2):
                    norm.append(f"{poly[i] / w:.6f}")
                    norm.append(f"{poly[i+1] / h:.6f}")
                lines.append(' '.join(norm))
            else:
                x, y, bw, bh = ann['bbox']
                cx = (x + bw / 2) / w
                cy = (y + bh / 2) / h
                nw = bw / w
                nh = bh / h
                lines.append(f"{class_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))


def split_dataset(samples: List[DatasetSample], train_ratio: float, val_ratio: float, test_ratio: float) -> dict:
    """按比例划分数据集。"""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train + val + test 比例之和必须等于 1")
    n = len(samples)
    if n == 0:
        raise ValueError("没有有效样本")

    indices = list(range(n))
    random.shuffle(indices)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return {
        'train': [samples[i] for i in indices[:train_end]],
        'val': [samples[i] for i in indices[train_end:val_end]],
        'test': [samples[i] for i in indices[val_end:]],
    }


def build_yolo_dataset(image_dir: str, label_dir: str, output_dir: str,
                       fmt: str = "auto", train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
                       force_val_from_train=True) -> Tuple[str, dict]:
    """
    将输入数据集整理为 YOLO 标准结构。

    Returns:
        data_yaml_path, split_info
    """
    _ensure_dir(output_dir)

    if fmt == "auto":
        fmt = detect_format(image_dir, label_dir)
        if fmt == "unknown":
            raise ValueError(f"无法识别 {label_dir} 的标注格式")

    # 解析
    if fmt in ("yolo_seg", "yolo_11"):
        samples, names = parse_yolo_seg(image_dir, label_dir)
    elif fmt == "yolo_det":
        samples, names = parse_yolo_det(image_dir, label_dir)
    elif fmt == "coco":
        ann_file = os.path.join(label_dir, 'instances_default.json')
        if not os.path.exists(ann_file):
            candidates = [f for f in os.listdir(label_dir) if f.endswith('.json')]
            if not candidates:
                raise FileNotFoundError("未找到 COCO 标注 JSON 文件")
            ann_file = os.path.join(label_dir, candidates[0])
        samples, names = parse_coco(image_dir, ann_file)
    else:
        raise ValueError(f"不支持的格式: {fmt}")

    if not samples:
        raise ValueError("未解析到任何样本")

    # 划分
    splits = split_dataset(samples, train_ratio, val_ratio, test_ratio)

    # 如果没有 val 且 force_val_from_train=True，从 train 拆出一部分作为 val
    if force_val_from_train and len(splits['val']) == 0 and len(splits['train']) > 1:
        n_val = max(1, int(len(splits['train']) * 0.15))
        splits['val'] = splits['train'][-n_val:]
        splits['train'] = splits['train'][:-n_val]

    split_info = {k: len(v) for k, v in splits.items()}

    # 创建目录并复制文件
    missing_labels = 0
    for split_name, split_samples in splits.items():
        if not split_samples:
            continue
        img_out = os.path.join(output_dir, 'images', split_name)
        lbl_out = os.path.join(output_dir, 'labels', split_name)
        _ensure_dir(img_out)
        _ensure_dir(lbl_out)

        for s in split_samples:
            shutil.copy2(s.image_path, os.path.join(img_out, os.path.basename(s.image_path)))
            if s.label_path and os.path.exists(s.label_path):
                shutil.copy2(s.label_path, os.path.join(lbl_out, os.path.basename(s.label_path)))
            else:
                missing_labels += 1
                if missing_labels <= 5:  # 只打印前 5 个
                    print(f"[警告] 标签文件不存在：{s.label_path}")
    
    if missing_labels > 0:
        print(f"[警告] 共 {missing_labels} 个样本缺少标签文件")

    # COCO 格式需要转换标注
    if fmt == "coco":
        import cv2
        for split_name, split_samples in splits.items():
            if not split_samples:
                continue
            lbl_out = os.path.join(output_dir, 'labels', split_name)
            image_wh_map = {}
            for s in split_samples:
                img = cv2.imread(s.image_path)
                if img is not None:
                    image_wh_map[s.image_path] = (img.shape[1], img.shape[0])
            coco_to_yolo(split_samples, lbl_out, image_wh_map, names)

    # 生成 txt 列表
    for split_name in ['train', 'val', 'test']:
        split_samples = splits.get(split_name, [])
        if not split_samples:
            continue
        with open(os.path.join(output_dir, f'{split_name}.txt'), 'w', encoding='utf-8') as f:
            for s in split_samples:
                f.write(f"images/{split_name}/{os.path.basename(s.image_path)}\n")

    # 生成 data.yaml
    if not names:
        names = ['Other', 'Round']

    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(f'path: {output_dir}\n')
        f.write('train: images/train\n')
        f.write('val: images/val\n')
        if splits.get('test'):
            f.write('test: images/test\n')
        f.write('names:\n')
        for i, name in enumerate(names):
            f.write(f'  {i}: {name}\n')

    return yaml_path, split_info


def ensure_val_split(dataset_dir: str, val_ratio: float = 0.15) -> bool:
    """
    如果 YOLO 数据集中没有 val，从 train 中拆分一部分作为 val。

    Returns:
        True 表示执行了拆分或已经有 val；False 表示无法拆分。
    """
    yaml_path = os.path.join(dataset_dir, 'data.yaml')
    train_img_dir = os.path.join(dataset_dir, 'images', 'train')
    train_lbl_dir = os.path.join(dataset_dir, 'labels', 'train')
    val_img_dir = os.path.join(dataset_dir, 'images', 'val')
    val_lbl_dir = os.path.join(dataset_dir, 'labels', 'val')

    if not os.path.isdir(train_img_dir) or not os.path.isdir(train_lbl_dir):
        return False

    # 检查是否已有 val
    if os.path.isdir(val_img_dir) and os.listdir(val_img_dir):
        return True

    train_imgs = _list_images(train_img_dir)
    if len(train_imgs) < 2:
        return False

    n_val = max(1, int(len(train_imgs) * val_ratio))
    random.shuffle(train_imgs)
    val_imgs = train_imgs[:n_val]
    train_imgs = train_imgs[n_val:]

    _ensure_dir(val_img_dir)
    _ensure_dir(val_lbl_dir)

    for img_path in val_imgs:
        base = os.path.splitext(os.path.basename(img_path))[0]
        lbl_src = os.path.join(train_lbl_dir, base + '.txt')
        lbl_dst = os.path.join(val_lbl_dir, base + '.txt')
        if os.path.exists(lbl_src):
            shutil.copy2(lbl_src, lbl_dst)
        # 移动图片（或复制）
        dst_img = os.path.join(val_img_dir, os.path.basename(img_path))
        shutil.move(img_path, dst_img)

    # 更新 data.yaml
    lines = []
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

    has_val = any(line.strip().startswith('val:') for line in lines)
    if not has_val:
        with open(yaml_path, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(line)
            f.write('val: images/val\n')

    return True


def has_val_in_yaml(dataset_dir: str) -> bool:
    """检查 data.yaml 中是否包含 val 字段。"""
    yaml_path = os.path.join(dataset_dir, 'data.yaml')
    if not os.path.exists(yaml_path):
        return False
    with open(yaml_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip().startswith('val:'):
                val = line.split(':', 1)[1].strip()
                return bool(val)
    return False
