import cv2
import numpy as np
from pathlib import Path


def remove_small_components(binary, min_area=100):
    """
    删除小连通域噪声
    binary: 0/255 二值图
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


def clean_binary(binary, min_area=100):
    """
    二值图后处理：
    1. 开运算去小噪声
    2. 闭运算补小孔
    3. 删除小连通域
    """
    kernel = np.ones((3, 3), np.uint8)

    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    binary = remove_small_components(binary, min_area=min_area)

    return binary


def binarize_gray_otsu(img):
    """
    灰度 Otsu 自动二值化
    适合提取高亮区域，但对颜色区分不强
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return binary


def binarize_bright_color(img):
    """
    提取所有亮色目标：
    包括蓝色圆、黄色杆、亮黄色块
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # H: 颜色 0~179
    # S: 饱和度，越大颜色越鲜艳
    # V: 亮度，越大越亮
    lower = np.array([0, 35, 55])
    upper = np.array([179, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    return mask


def binarize_blue(img):
    """
    提取蓝色圆形区域
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([85, 40, 45])
    upper_blue = np.array([130, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    return mask


def binarize_yellow_orange(img):
    """
    提取黄色/橙黄色区域：
    包括黄色杆和右侧亮黄色块
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([10, 45, 55])
    upper_yellow = np.array([45, 255, 255])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    return mask


def main():
    image_path = r"frames1/frame_020000.jpg"
    save_dir = Path("binary_results")
    save_dir.mkdir(exist_ok=True)

    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    binary_gray = binarize_gray_otsu(img)
    binary_gray = clean_binary(binary_gray, min_area=80)

    binary_bright = binarize_bright_color(img)
    binary_bright = clean_binary(binary_bright, min_area=80)

    binary_blue = binarize_blue(img)
    binary_blue = clean_binary(binary_blue, min_area=80)

    binary_yellow = binarize_yellow_orange(img)
    binary_yellow = clean_binary(binary_yellow, min_area=80)

    binary_blue_yellow = cv2.bitwise_or(binary_blue, binary_yellow)
    binary_blue_yellow = clean_binary(binary_blue_yellow, min_area=80)

    #cv2.imwrite(str(save_dir / "01_gray_otsu.png"), binary_gray)
    cv2.imwrite(str(save_dir / "02_bright_color.png"), binary_bright)
    #cv2.imwrite(str(save_dir / "03_blue.png"), binary_blue)
    #cv2.imwrite(str(save_dir / "04_yellow_orange.png"), binary_yellow)
    cv2.imwrite(str(save_dir / "05_blue_yellow_combined.png"), binary_blue_yellow)

    cv2.imshow("original", img)
    #cv2.imshow("gray otsu", binary_gray)
    cv2.imshow("bright color", binary_bright)
    #cv2.imshow("blue", binary_blue)
    #cv2.imshow("yellow orange", binary_yellow)
    cv2.imshow("blue + yellow", binary_blue_yellow)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()