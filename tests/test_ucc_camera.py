"""
UCC CCD 相机 USB 直连测试工具

用于验证：
  1. USB 设备是否被系统识别
  2. OpenCV 能否通过 UVC 协议打开相机
  3. 实时帧采集是否正常
  4. 曝光、增益等参数是否可配置
  5. PAL/NTSC 制式切换是否生效

使用方法:
  python test_ucc_camera.py                          # 自动扫描 camera 0~5
  python test_ucc_camera.py --device 0               # 只测试 camera 0
  python test_ucc_camera.py --device 0 --resolution NTSC
  python test_ucc_camera.py --scan-range 0 10        # 扫描 camera 0~9
  python test_ucc_camera.py --live                   # 连接成功后预览实时画面（ESC退出）
  python test_ucc_camera.py --device 0 --save test.jpg  # 保存一张测试帧

依赖:
  pip install opencv-python numpy
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


RESOLUTIONS: Dict[str, Dict[str, int]] = {
    "PAL":  {"width": 976, "height": 582, "fps": 25},
    "NTSC": {"width": 976, "height": 494, "fps": 30},
    "AUTO": {"width": 0,   "height": 0,   "fps": 0},
}

CAMERA_PROP_NAMES: Dict[int, str] = {
    cv2.CAP_PROP_POS_MSEC:       "POS_MSEC (当前位置ms)",
    cv2.CAP_PROP_FRAME_WIDTH:    "FRAME_WIDTH (帧宽)",
    cv2.CAP_PROP_FRAME_HEIGHT:   "FRAME_HEIGHT (帧高)",
    cv2.CAP_PROP_FPS:            "FPS (帧率)",
    cv2.CAP_PROP_FOURCC:         "FOURCC (编码格式)",
    cv2.CAP_PROP_FRAME_COUNT:    "FRAME_COUNT (总帧数)",
    cv2.CAP_PROP_FORMAT:         "FORMAT (格式)",
    cv2.CAP_PROP_MODE:           "MODE (模式)",
    cv2.CAP_PROP_BRIGHTNESS:     "BRIGHTNESS (亮度)",
    cv2.CAP_PROP_CONTRAST:       "CONTRAST (对比度)",
    cv2.CAP_PROP_SATURATION:     "SATURATION (饱和度)",
    cv2.CAP_PROP_HUE:            "HUE (色调)",
    cv2.CAP_PROP_GAIN:           "GAIN (增益)",
    cv2.CAP_PROP_EXPOSURE:       "EXPOSURE (曝光)",
    cv2.CAP_PROP_CONVERT_RGB:    "CONVERT_RGB (RGB转换)",
    cv2.CAP_PROP_WHITE_BALANCE_BLUE_U:  "WHITE_BALANCE_BLUE_U",
    cv2.CAP_PROP_WHITE_BALANCE_RED_V:   "WHITE_BALANCE_RED_V",
    cv2.CAP_PROP_AUTO_EXPOSURE:  "AUTO_EXPOSURE (自动曝光)",
    cv2.CAP_PROP_AUTOFOCUS:      "AUTOFOCUS (自动对焦)",
}


def fourcc_to_str(fourcc: float) -> str:
    try:
        code = int(fourcc)
        return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
    except Exception:
        return "N/A"


def _open_camera(device_index: int) -> cv2.VideoCapture:
    """在Windows上强制使用DSHOW后端打开相机，避免MSMF的兼容性问题。"""
    if sys.platform == "win32":
        cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(device_index)
    return cap


def scan_usb_cameras(scan_range: range) -> List[Tuple[int, cv2.VideoCapture]]:
    """扫描所有可用的 USB 相机设备。

    在 Windows 上，OpenCV 通过 DirectShow 后端枚举 UVC 设备。
    每个物理相机会对应一个 device_index (0, 1, 2...)。

    Returns:
        List of (device_index, VideoCapture) tuples for successfully opened devices.
    """
    found: List[Tuple[int, cv2.VideoCapture]] = []
    print(f"\n{'='*60}")
    print(f"  正在扫描 camera {scan_range.start} ~ {scan_range.stop - 1} ...")
    print(f"{'='*60}")

    for idx in scan_range:
        print(f"\n  [camera {idx}] 尝试打开 ...", end=" ", flush=True)
        cap = _open_camera(idx)
        if cap.isOpened():
            # 尝试读取一帧验证设备是否真正可用
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                print("✅ 已连接")
                # 放回第一帧（cap.read 已消耗）
                found.append((idx, cap))
            else:
                print("⚠️ 已连接但无法采集（可能是虚拟设备或兼容性问题）")
                cap.release()
        else:
            print("❌ 未检测到设备")
            cap.release()

    print(f"\n  扫描完成：共发现 {len(found)} 个设备")
    return found


def dump_device_properties(cap: cv2.VideoCapture, device_index: int) -> None:
    """打印相机所有可读属性。"""
    print(f"\n  --- camera {device_index} 属性 ---")
    for prop_id, prop_name in sorted(CAMERA_PROP_NAMES.items()):
        try:
            value = cap.get(prop_id)
            if prop_id == cv2.CAP_PROP_FOURCC:
                display = f"{value:.0f} ({fourcc_to_str(value)})"
            elif prop_id == cv2.CAP_PROP_FPS:
                display = f"{value:.1f}"
            else:
                display = f"{value:.1f}" if isinstance(value, float) else f"{int(value)}"
            print(f"    {prop_name:40s} = {display}")
        except Exception:
            print(f"    {prop_name:40s} = (不可读)")


def configure_device(
    cap: cv2.VideoCapture,
    resolution: str,
    exposure: Optional[float] = None,
    gain: Optional[float] = None,
    brightness: Optional[float] = None,
    contrast: Optional[float] = None,
) -> None:
    """配置相机参数。"""
    print(f"\n  --- 配置 camera ---")
    res_cfg = RESOLUTIONS[resolution.upper()]
    print(f"  制式: {resolution.upper()}  (目标 {res_cfg['width']}x{res_cfg['height']} @ {res_cfg['fps']}fps)")

    if res_cfg["width"] > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, res_cfg["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res_cfg["height"])
    if res_cfg["fps"] > 0:
        cap.set(cv2.CAP_PROP_FPS, res_cfg["fps"])

    params = {
        "EXPOSURE": (cv2.CAP_PROP_EXPOSURE, exposure),
        "GAIN": (cv2.CAP_PROP_GAIN, gain),
        "BRIGHTNESS": (cv2.CAP_PROP_BRIGHTNESS, brightness),
        "CONTRAST": (cv2.CAP_PROP_CONTRAST, contrast),
    }

    for name, (prop_id, value) in params.items():
        if value is not None:
            cap.set(prop_id, value)
            print(f"  {name}: 设置为 {value}")

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  实际: {actual_w}x{actual_h} @ {actual_fps:.1f}fps")

    # 检测分辨率不兼容并给出建议
    target_w = res_cfg["width"]
    target_h = res_cfg["height"]
    if target_w > 0 and (actual_w != target_w or actual_h != target_h):
        print(f"  ⚠️ 分辨率不匹配：目标 {target_w}x{target_h} → 实际 {actual_w}x{actual_h}")
        if actual_w <= 800 and actual_h <= 600:
            print(f"  💡 提示：当前设备可能是内置摄像头（{actual_w}x{actual_h}），不支持 PAL 制式")
            print(f"     建议使用 --resolution AUTO 或指定正确的 UCC 设备索引")
        print()


def grab_test_frames(
    cap: cv2.VideoCapture,
    device_index: int,
    num_frames: int = 5,
) -> bool:
    """连续采集若干帧测试稳定性。"""
    print(f"\n  --- 连续采集 {num_frames} 帧测试 ---")
    ok_count = 0
    for i in range(1, num_frames + 1):
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"    帧 {i}: ❌ 读取失败")
            continue
        h, w = frame.shape[:2]
        ch = frame.shape[2] if len(frame.shape) > 2 else 1
        mean_val = float(np.mean(frame))
        min_val = float(np.min(frame))
        max_val = float(np.max(frame))
        print(f"    帧 {i}: ✅ {w}x{h} ch={ch}  mean={mean_val:6.1f}  min={min_val}  max={max_val}")
        ok_count += 1

    ratio = ok_count / num_frames * 100
    status = "✅ 稳定" if ratio == 100 else ("⚠️ 部分丢帧" if ratio >= 50 else "❌ 连接不稳定")
    print(f"  成功率: {ok_count}/{num_frames} ({ratio:.0f}%)  {status}")
    return ok_count > 0


def show_live_preview(cap: cv2.VideoCapture, device_index: int) -> None:
    """实时预览，按 ESC 退出。"""
    print(f"\n  --- 实时预览 camera {device_index} (按 ESC 退出) ---")
    win_name = f"UCC Camera {device_index} - Live Preview"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    fps_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("    帧读取失败，预览中断")
            break

        frame_count += 1
        elapsed = time.time() - fps_start
        if elapsed >= 2.0 and frame_count > 0:
            fps_val = frame_count / elapsed
            frame_count = 0
            fps_start = time.time()
            print(f"    FPS: {fps_val:.1f}", end="\r", flush=True)

        cv2.imshow(win_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cv2.destroyWindow(win_name)
    print(f"\n  预览结束")


def save_frame(cap: cv2.VideoCapture, save_path: str) -> None:
    """保存一帧图像。"""
    ret, frame = cap.read()
    if not ret or frame is None:
        print("  ❌ 读取帧失败，无法保存")
        return
    path = Path(save_path)
    cv2.imwrite(str(path), frame)
    file_size_kb = path.stat().st_size / 1024
    print(f"  ✅ 已保存: {path}  ({frame.shape[1]}x{frame.shape[0]}, {file_size_kb:.1f} KB)")


def quick_check_connection(device_index: int) -> bool:
    """快速检查指定设备是否可用。无需打印，适合被外部调用。"""
    cap = cv2.VideoCapture(device_index)
    ok = cap.isOpened()
    cap.release()
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="UCC CCD 相机 USB 直连测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--device", type=int, default=None,
        help="指定要测试的 camera index（不指定则自动扫描 0~5）",
    )
    parser.add_argument(
        "--scan-range", type=int, nargs=2, default=None,
        metavar=("START", "END"),
        help="扫描范围，如 --scan-range 0 10",
    )
    parser.add_argument(
        "--resolution", type=str, default="PAL",
        choices=["PAL", "NTSC", "AUTO"],
        help="分辨率制式 (default: PAL)",
    )
    parser.add_argument(
        "--exposure", type=float, default=None,
        help="手动设置曝光值",
    )
    parser.add_argument(
        "--gain", type=float, default=None,
        help="手动设置增益值",
    )
    parser.add_argument(
        "--brightness", type=float, default=None,
        help="手动设置亮度",
    )
    parser.add_argument(
        "--contrast", type=float, default=None,
        help="手动设置对比度",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="连接成功后进入实时预览模式",
    )
    parser.add_argument(
        "--save", type=str, default=None, metavar="PATH",
        help="保存一张测试帧到指定路径",
    )
    parser.add_argument(
        "--frames", type=int, default=5,
        help="连续采集帧数 (default: 5)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="安静模式，只输出关键结果",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  UCC CCD 相机 USB 直连测试")
    print("=" * 60)
    print(f"  OpenCV 版本: {cv2.__version__}")
    print(f"  后端: {cv2.videoio_registry.getBackendName(cv2.CAP_DSHOW)}")

    devices: List[Tuple[int, cv2.VideoCapture]] = []

    if args.device is not None:
        cap = _open_camera(args.device)
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                devices.append((args.device, cap))
                print(f"\n  camera {args.device}: ✅ 已连接")
            else:
                print(f"\n  camera {args.device}: ⚠️ 已连接但无法采集帧")
                cap.release()
        else:
            print(f"\n  camera {args.device}: ❌ 无法打开")
            cap.release()
    else:
        if args.scan_range is not None:
            scan_range = range(args.scan_range[0], args.scan_range[1])
        else:
            scan_range = range(0, 6)
        devices = scan_usb_cameras(scan_range)

    if not devices:
        print("\n" + "=" * 60)
        print("  ❌ 未检测到任何 UCC 相机设备")
        print("=" * 60)
        print("\n  请检查：")
        print("  1. CCD 相机是否已通过 USB 连接")
        print("  2. 设备管理器中是否有 'USB Composite Device' 或类似设备")
        print("  3. 相机 12V 电源是否已接通")
        print("  4. 尝试更换 USB 端口或重新插拔")
        return 1

    all_ok = True
    for device_index, cap in devices:
        print(f"\n{'='*60}")
        print(f"  测试 camera {device_index}")
        print(f"{'='*60}")

        if not args.quiet:
            dump_device_properties(cap, device_index)

        configure_device(
            cap, args.resolution,
            exposure=args.exposure,
            gain=args.gain,
            brightness=args.brightness,
            contrast=args.contrast,
        )

        frame_ok = grab_test_frames(cap, device_index, num_frames=args.frames)

        if args.save:
            save_frame(cap, args.save)

        if args.live and frame_ok:
            show_live_preview(cap, device_index)

        if not frame_ok:
            all_ok = False

    for _, cap in devices:
        cap.release()

    print(f"\n{'='*60}")
    if all_ok:
        print("  ✅ 全部测试通过")
    else:
        print("  ⚠️ 部分测试未通过，请检查上述输出")
    print(f"{'='*60}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())