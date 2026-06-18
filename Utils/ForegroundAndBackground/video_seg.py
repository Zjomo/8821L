"""
视频前后景分离模块。

功能：
  1. 读取视频文件
  2. 对每帧进行前后景分离（背景为偏黑色，前景为各种材料物品）
  3. 双窗口同步播放：左边原视频，右边分割结果

使用方法：
  python video_seg.py                  # 弹出文件选择对话框
  python video_seg.py path/to/video.mp4  # 直接指定视频路径
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


# ============================================================
# 前后景分离算法
# ============================================================

def remove_small_components(binary: np.ndarray, min_area: int = 100) -> np.ndarray:
    """
    删除小连通域噪声。
    
    参数
    ----------
    binary : np.ndarray
        0/255 二值图
    min_area : int
        最小连通域面积阈值
    
    返回
    -------
    np.ndarray
        清理后的二值图
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    out = np.zeros_like(binary)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 255
    
    return out


def clean_binary(binary: np.ndarray, min_area: int = 100) -> np.ndarray:
    """
    二值图后处理：
      1. 开运算去小噪声
      2. 闭运算补小孔
      3. 删除小连通域
    
    参数
    ----------
    binary : np.ndarray
        原始二值图
    min_area : int
        最小连通域面积
    
    返回
    -------
    np.ndarray
        清理后的二值图
    """
    kernel = np.ones((3, 3), np.uint8)
    
    # 开运算去小噪声
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    # 闭运算补小孔
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    # 删除小连通域
    binary = remove_small_components(binary, min_area=min_area)
    
    return binary


def segment_foreground_background(
    frame: np.ndarray,
    method: str = "bright",
    min_area: int = 100,
) -> np.ndarray:
    """
    对单帧图像进行前后景分离。
    
    背景假设为偏黑色，前景为各种材料物品（亮色区域）。
    
    参数
    ----------
    frame : np.ndarray
        BGR 格式图像
    method : str
        分割方法：
          - "bright": 提取所有亮色目标（推荐）
          - "otsu": Otsu 自动二值化
          - "blue": 提取蓝色区域
          - "yellow": 提取黄色/橙色区域
    min_area : int
        最小连通域面积，用于去噪
    
    返回
    -------
    np.ndarray
        分割结果（BGR 格式），前景为原色，背景为黑色
    """
    if frame is None or frame.size == 0:
        return frame
    
    # 生成二值掩码
    if method == "bright":
        mask = _binarize_bright_color(frame)
    elif method == "otsu":
        mask = _binarize_gray_otsu(frame)
    elif method == "blue":
        mask = _binarize_blue(frame)
    elif method == "yellow":
        mask = _binarize_yellow_orange(frame)
    else:
        mask = _binarize_bright_color(frame)
    
    # 清理二值图
    mask = clean_binary(mask, min_area=min_area)
    
    # 将掩码应用到原图
    mask_3ch = cv2.merge([mask, mask, mask])
    result = cv2.bitwise_and(frame, mask_3ch)
    
    return result


def _binarize_gray_otsu(img: np.ndarray) -> np.ndarray:
    """灰度 Otsu 自动二值化，适合提取高亮区域。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _binarize_bright_color(img: np.ndarray) -> np.ndarray:
    """提取所有亮色目标（包括各种颜色的前景物体）。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 提取饱和度 > 35 且亮度 > 55 的区域（排除黑色背景）
    lower = np.array([0, 35, 55])
    upper = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    return mask


def _binarize_blue(img: np.ndarray) -> np.ndarray:
    """提取蓝色区域。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([85, 40, 45])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    return mask


def _binarize_yellow_orange(img: np.ndarray) -> np.ndarray:
    """提取黄色/橙黄色区域。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([10, 45, 55])
    upper_yellow = np.array([45, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    return mask


# ============================================================
# 视频播放
# ============================================================

def play_video_with_segmentation(
    video_path: str,
    method: str = "bright",
    min_area: int = 100,
    show_mask: bool = False,
) -> None:
    """
    播放视频并实时显示前后景分离结果。
    
    左窗口：原视频
    右窗口：分割结果（或二值掩码）
    
    参数
    ----------
    video_path : str
        视频文件路径
    method : str
        分割方法：bright / otsu / blue / yellow
    min_area : int
        最小连通域面积
    show_mask : bool
        True: 右窗口显示二值掩码
        False: 右窗口显示前景提取结果（背景为黑色）
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频信息: {width}x{height}, {fps:.2f}fps, {total_frames} 帧")
    print(f"分割方法: {method}, 最小面积: {min_area}")
    print("按 'q' 或 ESC 退出，按空格暂停/继续")
    
    # 创建窗口
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Segmented", cv2.WINDOW_NORMAL)
    
    # 调整窗口位置和大小
    cv2.moveWindow("Original", 50, 50)
    cv2.moveWindow("Segmented", width + 70, 50)
    
    paused = False
    frame_idx = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                # 视频结束，从头播放
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                continue
            
            frame_idx += 1
            
            # 前后景分离
            if show_mask:
                # 显示二值掩码
                if method == "bright":
                    mask = _binarize_bright_color(frame)
                elif method == "otsu":
                    mask = _binarize_gray_otsu(frame)
                elif method == "blue":
                    mask = _binarize_blue(frame)
                elif method == "yellow":
                    mask = _binarize_yellow_orange(frame)
                else:
                    mask = _binarize_bright_color(frame)
                result = clean_binary(mask, min_area=min_area)
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            else:
                # 显示前景提取结果
                result = segment_foreground_background(
                    frame, method=method, min_area=min_area
                )
            
            # 添加帧号
            cv2.putText(
                frame, f"Frame: {frame_idx}/{total_frames}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
            cv2.putText(
                result, f"Method: {method}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
        
        # 显示
        cv2.imshow("Original", frame)
        cv2.imshow("Segmented", result)
        
        # 键盘控制
        key = cv2.waitKey(int(1000 / fps)) & 0xFF
        if key == 27 or key == ord('q'):  # ESC 或 q 退出
            break
        elif key == ord(' '):  # 空格暂停/继续
            paused = not paused
            if paused:
                print(f"暂停于帧 {frame_idx}")
            else:
                print("继续播放")
        elif key == ord('m'):  # m 切换掩码/前景显示
            show_mask = not show_mask
            print(f"显示模式: {'掩码' if show_mask else '前景提取'}")
        elif key == ord('1'):  # 切换到 bright 方法
            method = "bright"
            print(f"切换方法: {method}")
        elif key == ord('2'):  # 切换到 otsu 方法
            method = "otsu"
            print(f"切换方法: {method}")
        elif key == ord('3'):  # 切换到 blue 方法
            method = "blue"
            print(f"切换方法: {method}")
        elif key == ord('4'):  # 切换到 yellow 方法
            method = "yellow"
            print(f"切换方法: {method}")
    
    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# 文件选择
# ============================================================

def select_video_file() -> Optional[str]:
    """弹出文件选择对话框，返回选择的视频路径。"""
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        
        file_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[
                ("视频文件", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("所有文件", "*.*")
            ]
        )
        
        root.destroy()
        
        if file_path:
            return file_path
        return None
        
    except Exception as e:
        print(f"文件选择对话框出错: {e}")
        return None


# ============================================================
# 主函数
# ============================================================

def main():
    """主入口函数。"""
    # 解析命令行参数
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        print("请选择视频文件...")
        video_path = select_video_file()
    
    if not video_path:
        print("未选择视频文件，退出")
        return
    
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"视频文件不存在: {video_path}")
        return
    
    print(f"视频路径: {video_path}")
    print()
    print("快捷键说明:")
    print("  空格    - 暂停/继续")
    print("  q/ESC   - 退出")
    print("  m       - 切换显示模式（掩码/前景提取）")
    print("  1       - bright 方法（提取亮色）")
    print("  2       - otsu 方法（Otsu 二值化）")
    print("  3       - blue 方法（提取蓝色）")
    print("  4       - yellow 方法（提取黄色）")
    print()
    
    play_video_with_segmentation(
        str(video_path),
        method="bright",
        min_area=100,
        show_mask=False,
    )


if __name__ == "__main__":
    main()
