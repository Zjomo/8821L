#!/usr/bin/env python3

"""模拟显微镜设备：SQLite 持久化的位移台 + 相机。

遵循 python-microscope 的 ABC 体系（:mod:`microscope.abc`）：

* :class:`SQLiteStage`  实现 :class:`microscope.abc.Stage`，由 X/Y/Z 三个
  :class:`SQLiteStageAxis` 组成，移动逻辑参考 ``microscope.simulators.SimulatedStage``
  （超限自动截断，不抛异常），并把实时位置写入 SQLite。
* :class:`SampleAwareCamera` 继承 ``microscope.simulators.SimulatedCamera``，
  参考 ``microscope.simulators.stage_aware_camera.StageAwareCamera``：按位移台
  X/Y 在一张大样本图上裁剪视野，Z 轴模拟离焦模糊，曝光/增益影响亮度与噪声。

单位约定：位移台坐标为微米（µm），样本像素大小 ``pixel_size`` µm/px。
"""

import logging
import math
import os
import sys
import time
from typing import Mapping, Tuple

import numpy as np

# 让脚本可以直接运行（无需 pip install microscope）：
# 从本文件位置向上找到包含 microscope 包的仓库根目录。
def _ensure_repo_on_path() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    candidates = (
        root,                                      # simulator_app 与 microscope/ 同级
        os.path.join(root, "microscope-master"),   # 仓库根在嵌套目录中
    )
    for cand in candidates:
        if os.path.isfile(os.path.join(cand, "microscope", "__init__.py")):
            if cand not in sys.path:
                sys.path.insert(0, cand)
            return
    raise ImportError(
        "未找到 microscope 包：请把 simulator_app 放在 python-microscope 仓库根目录内运行"
    )


_ensure_repo_on_path()

import microscope
import microscope.abc
from microscope.simulators import SimulatedCamera

_logger = logging.getLogger(__name__)

# 曝光参考值：曝光时间为此值的多少倍，亮度就乘多少
_REFERENCE_EXPOSURE_S = 0.05


class SQLiteStageAxis(microscope.abc.StageAxis):
    """单轴：位置可持久化到 SQLite。

    Args:
        name: 轴名，如 "x"。
        limits: :class:`microscope.AxisLimits`（µm）。
        db: :class:`db.Database` 实例。
        default: 无持久化记录时的初始位置（默认取量程中点）。
    """

    def __init__(
        self,
        name: str,
        limits: microscope.AxisLimits,
        db,
        default: float = None,
    ) -> None:
        super().__init__()
        self._name = name
        self._limits = limits
        self._db = db
        if default is None:
            default = limits.lower + (limits.upper - limits.lower) / 2.0
        # 重启后恢复上一次的位置
        self._position = db.get_state_float("stage." + name, default)

    @property
    def position(self) -> float:
        return self._position

    @property
    def limits(self) -> microscope.AxisLimits:
        return self._limits

    def move_by(self, delta: float, persist: bool = True) -> None:
        self.move_to(self._position + delta, persist=persist)

    def move_to(self, pos: float, persist: bool = True) -> None:
        # 与 SimulatedStage 一致：超限截断，不抛异常
        pos = min(max(pos, self._limits.lower), self._limits.upper)
        self._position = pos
        if persist:
            self._db.set_state("stage." + self._name, pos)


class SQLiteStage(microscope.abc.Stage):
    """XYZ 三轴电动位移台（模拟），位置持久化到 SQLite。"""

    def __init__(self, db, limits: Mapping[str, Tuple[float, float]]) -> None:
        super().__init__()
        self._axes = {
            name: SQLiteStageAxis(name, microscope.AxisLimits(lo, hi), db)
            for name, (lo, hi) in limits.items()
        }

    def _do_shutdown(self) -> None:
        pass

    def may_move_on_enable(self) -> bool:
        # 模拟"上电需要归零"的硬件行为
        return False

    @property
    def axes(self) -> Mapping[str, microscope.abc.StageAxis]:
        return self._axes

    def move_by(self, delta: Mapping[str, float], persist: bool = True) -> None:
        unknown = set(delta) - set(self.axes)
        if unknown:
            raise KeyError(f"unknown stage axes: {sorted(unknown)}")
        # Resolve every target before mutating an axis.  This gives callers a
        # predictable all-or-nothing operation even though the microscope ABC
        # permits stages to move axes sequentially.
        targets = {name: self.axes[name].position + float(d)
                   for name, d in delta.items()}
        self.move_to(targets, persist=persist)

    def move_to(self, position: Mapping[str, float], persist: bool = True) -> None:
        unknown = set(position) - set(self.axes)
        if unknown:
            raise KeyError(f"unknown stage axes: {sorted(unknown)}")
        targets = {}
        for name, pos in position.items():
            value = float(pos)
            if not math.isfinite(value):
                raise ValueError(f"non-finite position for axis {name}")
            axis = self.axes[name]
            targets[name] = min(max(value, axis.limits.lower), axis.limits.upper)
        for name, value in targets.items():
            self.axes[name].move_to(value, persist=persist)

    def describe(self) -> dict:
        """Return a JSON-friendly stage capability and state snapshot."""
        return {
            "enabled": bool(self.get_is_enabled()),
            "axes": {
                name: {
                    "position": axis.position,
                    "lower": axis.limits.lower,
                    "upper": axis.limits.upper,
                }
                for name, axis in self.axes.items()
            },
        }


# ----------------------------------------------------------------------
# 样本图层（掩码 / 衬底 / 障碍物）
# ----------------------------------------------------------------------
DEFAULT_SAMPLE_SPEC = {
    "ground":   {"count": 4, "shape": "blob",    "label": "衬底",   "size": 90.0, "custom": ""},
    "mask":     {"count": 3, "shape": "ellipse", "label": "掩码",   "size": 30.0, "custom": ""},
    "obstacle": {"count": 2, "shape": "star",    "label": "障碍物", "size": 28.0, "custom": ""},
}

# 各图层的绘制样式（RGBA）：衬底在最底层，障碍物在最上层
_LAYER_STYLE = {
    "ground":   {"fill": (70, 85, 115, 170),  "outline": (200, 215, 240, 210)},
    "mask":     {"fill": (0, 190, 210, 110),  "outline": (180, 255, 255, 230)},
    "obstacle": {"fill": (235, 70, 45, 235),  "outline": (255, 235, 120, 255)},
}


def _rotate(points, deg: float):
    a = math.radians(deg)
    ca, sa = math.cos(a), math.sin(a)
    return [(x * ca - y * sa, x * sa + y * ca) for x, y in points]


def _parse_custom_vertices(text: str):
    """解析自定义顶点，如 ``"0,0 40,0 40,25 10,30"``，归一化到 [-0.5, 0.5]。"""
    try:
        vals = [float(t) for t in str(text).replace(";", " ").replace(",", " ").split()]
    except ValueError:
        return None
    if len(vals) < 6 or len(vals) % 2:
        return None
    pts = list(zip(vals[0::2], vals[1::2]))
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    w, h = (max(xs) - min(xs)) or 1.0, (max(ys) - min(ys)) or 1.0
    scale = 1.0 / max(w, h)
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    return [((x - cx) * scale, (y - cy) * scale) for x, y in pts]


def _unit_shape(shape: str, rng, custom: str = ""):
    """返回单位形状的顶点列表（外接半径 ~1，中心在原点）。"""
    if shape == "rect":
        return [(-1, -0.7), (1, -0.7), (1, 0.7), (-1, 0.7)]
    if shape == "triangle":
        return [(0, -1), (0.95, 0.75), (-0.95, 0.75)]
    if shape == "star":
        return [
            (
                (1.0 if i % 2 == 0 else 0.45) * math.cos(i * math.pi / 5 - math.pi / 2),
                (1.0 if i % 2 == 0 else 0.45) * math.sin(i * math.pi / 5 - math.pi / 2),
            )
            for i in range(10)
        ]
    if shape == "custom":
        pts = _parse_custom_vertices(custom)
        if pts is not None:
            return pts
        # 顶点不合法时回退为椭圆
    if shape == "blob":
        k = int(rng.integers(6, 10))
        ang = np.sort(rng.uniform(0, 2 * np.pi, k))
        rad = rng.uniform(0.5, 1.0, k)
        return [(r * math.cos(a), 0.9 * r * math.sin(a)) for a, r in zip(ang, rad)]
    # ellipse（默认）
    ry = float(rng.uniform(0.55, 0.85))
    n = 40
    return [
        (math.cos(2 * math.pi * i / n), ry * math.sin(2 * math.pi * i / n))
        for i in range(n)
    ]


# 支持中文的字体（load_default 只有 ASCII，画中文标签会乱码）
_CJK_FONT = None


def _get_cjk_font(size: int = 16):
    """返回支持中文的 PIL 字体；Windows 依次尝试雅黑/黑体/宋体。"""
    global _CJK_FONT
    if _CJK_FONT is None:
        from PIL import ImageFont

        candidates = [
            "C:/Windows/Fonts/msyh.ttc",   # 微软雅黑
            "C:/Windows/Fonts/msyh.ttf",
            "C:/Windows/Fonts/simhei.ttf",  # 黑体
            "C:/Windows/Fonts/simsun.ttc",  # 宋体
            "msyh.ttc", "simhei.ttf", "simsun.ttc", "arial.ttf",
        ]
        for path in candidates:
            try:
                _CJK_FONT = ImageFont.truetype(path, size)
                break
            except OSError:
                continue
        if _CJK_FONT is None:
            _CJK_FONT = ImageFont.load_default()
    return _CJK_FONT


_SAMPLE_CACHE: dict = {}
_SAMPLE_CACHE_MAX = 3   # 复用相同 (spec, seed) 样本：worker 每次运行
                        # 构建新世界时省去 ~1s 的 PIL 重绘


def generate_sample(spec: Mapping[str, Mapping], seed: int, height: int = 2000, width: int = 3000):
    """按图层配置生成 H&E 组织风格样本，返回 (灰度数组, 对象清单)。

    只保留灰度（相机逐帧输出灰度），照明不均用稀疏网格叠加，
    不加静态噪声（读出噪声在每帧 _fetch_data 中添加），保证生成速度。
    相同 (spec, seed, 尺寸) 直接命中进程级缓存（数组只读共享，
    相机仅做切片裁剪不就地修改）。
    """
    cache_key = (
        int(seed), int(height), int(width),
        tuple(sorted(
            (str(cat), tuple(sorted((str(k), str(v)) for k, v in cfg.items())))
            for cat, cfg in (spec or {}).items())),
    )
    cached = _SAMPLE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (width, height), (58, 34, 52))
    draw = ImageDraw.Draw(img, "RGBA")
    # 低频背景：柔和的组织斑块
    for _ in range(500):
        cx, cy = rng.uniform(0, width), rng.uniform(0, height)
        rx, ry = rng.uniform(40, 260), rng.uniform(40, 260)
        alpha = int(rng.uniform(6, 26))
        color = (int(rng.uniform(120, 210)), int(rng.uniform(90, 150)), 40)
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=color + (alpha,))
    # 细胞核：紫色椭圆
    for _ in range(1500):
        cx, cy = rng.uniform(0, width), rng.uniform(0, height)
        rx, ry = rng.uniform(5, 22), rng.uniform(4, 18)
        v = int(rng.uniform(60, 130))
        color = (150, v, 200)
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=color + (150,))
        # 核仁：中心亮点
        nv = int(rng.uniform(170, 230))
        draw.ellipse(
            (cx - rx / 3, cy - ry / 3, cx + rx / 3, cy + ry / 3),
            fill=(nv, nv - 60, nv) + (160,),
        )
    # 荧光颗粒：绿色亮点
    for _ in range(600):
        cx, cy = rng.uniform(0, width), rng.uniform(0, height)
        r = rng.uniform(1, 4)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(80, 255, 120, 220))
    # 标记十字：每 500 px（= 250 µm）一个视野定位标记
    for gx in range(0, width, 500):
        for gy in range(0, height, 500):
            draw.line((gx - 12, gy, gx + 12, gy), fill=(255, 255, 255, 130))
            draw.line((gx, gy - 12, gx, gy + 12), fill=(255, 255, 255, 130))

    # 图层对象：衬底 → 掩码 → 障碍物（上层覆盖下层）
    objects = []
    font = _get_cjk_font(16)
    for cat in ("ground", "mask", "obstacle"):
        cfg = dict(spec.get(cat) or {})
        count = int(cfg.get("count", 0))
        shape = str(cfg.get("shape", "ellipse"))
        prefix = str(cfg.get("label") or cat)
        base_size = float(cfg.get("size", 30))
        custom = str(cfg.get("custom", ""))
        style = _LAYER_STYLE[cat]
        for i in range(count):
            size = base_size * float(rng.uniform(0.6, 1.4))
            cx = float(rng.uniform(size, width - size))
            cy = float(rng.uniform(size, height - size))
            angle = float(rng.uniform(0, 360))
            pts = _unit_shape(shape, rng, custom)
            world = [(cx + x * size, cy + y * size) for x, y in _rotate(pts, angle)]
            draw.polygon(world, fill=style["fill"], outline=style["outline"])
            label = "%s-%02d" % (prefix, i + 1)
            draw.text(
                (cx, cy - size - 6), label,
                fill=(255, 255, 255, 255), font=font,
                stroke_width=2, stroke_fill=(0, 0, 0, 210),
            )
            objects.append(
                {"category": cat, "label": label, "shape": shape,
                 "cx": round(cx, 2), "cy": round(cy, 2),
                 "size": round(size, 2), "angle": round(angle, 2),
                 "params": custom}
            )

    arr = np.asarray(img).astype(np.float32)
    # RGB → 灰度（相机逐帧输出灰度），照明不均在灰度上叠加；
    # 静态噪声省略（读出噪声每帧添加），稀疏 ogrid 避免 48MB 临时数组
    gray = arr @ np.array([0.299, 0.587, 0.114], np.float32)
    yy, xx = np.ogrid[0:height, 0:width]
    gray *= (
        1.0
        + 0.12 * np.sin(xx / 700.0) * np.cos(yy / 500.0)
        + 0.05 * np.cos(xx / 250.0 + yy / 300.0)
    ).astype(np.float32)
    result = (np.clip(gray, 0, 255).astype(np.uint8), objects)
    if len(_SAMPLE_CACHE) >= _SAMPLE_CACHE_MAX:
        _SAMPLE_CACHE.pop(next(iter(_SAMPLE_CACHE)))
    _SAMPLE_CACHE[cache_key] = result
    return result


class SampleAwareCamera(SimulatedCamera):
    """模拟相机：视野随位移台位置变化，Z 轴模拟离焦。

    与 :class:`StageAwareCamera` 的区别：
    * 样本为程序合成（组织切片风格），无需外部图片；
    * 样本含三个可配置图层（掩码/衬底/障碍物）：每类可指定数量、
      标签与形状（含自定义任意多边形顶点），配置持久化到 SQLite；
    * 曝光/增益持久化到 SQLite，重启后恢复；
    * 用亮度因子 + 泊松噪声模拟曝光增益效果。
    """

    def __init__(self, stage: microscope.abc.Stage, db, **kwargs) -> None:
        super().__init__(sensor_shape=(800, 600), **kwargs)
        # 清空 SimulatedCamera 的测试用 settings（同 StageAwareCamera 做法）
        self._settings = {}
        self._stage = stage
        self._db = db
        self._pixel_size = 0.5  # µm/px，视野 = 400 x 300 µm

        # 曝光/增益从 SQLite 恢复
        self._exposure_time = db.get_state_float("camera.exposure_s", _REFERENCE_EXPOSURE_S)
        self._gain = db.get_state_float("camera.gain", 0.0)
        # 快速预览（拖动时跳过曝光等待）与最近一帧对应的台位
        self._fast_preview = False
        self._last_frame_pos = None

        # 样本图层配置从 SQLite 恢复（无记录用默认值），
        # 合成 3000 x 2000 px = 1500 x 1000 µm 样本
        self._seed = int(db.get_state("sample.seed", "20260904"))
        self._spec = self._load_spec()
        self._sample, objects = generate_sample(self._spec, self._seed)
        self._objects = objects
        self._db.replace_sample_objects(objects)

    # ------------------------------------------------------------------
    # 样本图层配置（掩码 / 衬底 / 障碍物）
    # ------------------------------------------------------------------
    def _load_spec(self) -> dict:
        spec = {}
        for cat, default in DEFAULT_SAMPLE_SPEC.items():
            spec[cat] = {
                "count": int(self._db.get_state("sample.%s.count" % cat, str(default["count"]))),
                "shape": self._db.get_state("sample.%s.shape" % cat, default["shape"]),
                "label": self._db.get_state("sample.%s.label" % cat, default["label"]),
                "size": float(self._db.get_state("sample.%s.size" % cat, str(default["size"]))),
                "custom": self._db.get_state("sample.%s.custom" % cat, default["custom"]) or "",
            }
        return spec

    def get_sample_spec(self) -> dict:
        return {cat: dict(cfg) for cat, cfg in self._spec.items()}

    def get_sample_objects(self) -> list:
        return list(self._objects)

    def set_sample_spec(self, spec: Mapping[str, Mapping]) -> None:
        """按新配置重建样本（整体换引用，采集线程读到旧/新样本均一致）。

        配置与当前完全一致时直接返回（不重播种、不重绘）——worker 每次
        运行构建新世界都会重放同一 spec，跳过后命中 generate_sample
        进程缓存，省去 ~1-16s 的样本重绘。
        """
        new_spec = {
            cat: {
                "count": int(spec.get(cat, {}).get("count", d["count"])),
                "shape": str(spec.get(cat, {}).get("shape", d["shape"])),
                "label": str(spec.get(cat, {}).get("label", d["label"])),
                "size": float(spec.get(cat, {}).get("size", d["size"])),
                "custom": str(spec.get(cat, {}).get("custom", d["custom"])),
            }
            for cat, d in DEFAULT_SAMPLE_SPEC.items()
        }
        if new_spec == self._spec:
            return
        self._spec = new_spec
        seed = int(np.random.default_rng().integers(1, 2**31 - 1))
        new_sample, objects = generate_sample(self._spec, seed)
        self._seed = seed
        self._objects = objects
        self._sample = new_sample
        for cat, cfg in self._spec.items():
            self._db.set_state("sample.%s.count" % cat, cfg["count"])
            self._db.set_state("sample.%s.shape" % cat, cfg["shape"])
            self._db.set_state("sample.%s.label" % cat, cfg["label"])
            self._db.set_state("sample.%s.size" % cat, cfg["size"])
            self._db.set_state("sample.%s.custom" % cat, cfg["custom"])
        self._db.set_state("sample.seed", seed)
        self._db.replace_sample_objects(objects)

    # ------------------------------------------------------------------
    # 曝光 / 增益（持久化）
    # ------------------------------------------------------------------
    def set_exposure_time(self, value: float) -> None:
        self._exposure_time = value
        self._db.set_state("camera.exposure_s", value)

    def get_exposure_time(self) -> float:
        return self._exposure_time

    def set_gain(self, value: float) -> None:
        self._gain = value
        self._db.set_state("camera.gain", value)

    def get_gain(self) -> float:
        return self._gain

    def set_fast_preview(self, enabled: bool) -> None:
        """交互（拖动）时跳过曝光等待，提高帧率；不改变图像亮度。"""
        self._fast_preview = bool(enabled)

    def last_frame_pos(self):
        """返回最近一帧对应的台位 {x, y, z}（无帧时为 None）。"""
        return self._last_frame_pos

    def capture(self, timeout_s: float = 5.0):
        """Public synchronous frame API; avoids leaking ``_fetch_data``."""
        if not self.get_is_enabled():
            self.enable()
        self.trigger()
        deadline = time.monotonic() + float(timeout_s)
        frame = None
        while frame is None and time.monotonic() < deadline:
            frame = self._fetch_data()
            if frame is None:
                time.sleep(0.002)
        if frame is None:
            raise TimeoutError(f"camera frame timeout after {timeout_s}s")
        return frame

    # ------------------------------------------------------------------
    # 帧生成
    # ------------------------------------------------------------------
    def _fetch_loop(self) -> None:
        # 基类 DataDevice.enable() 会启动后台轮询线程不断调用 _fetch_data，
        # 与 GUI 的 CameraWorker 竞争消费 _triggered（双方都 sleep 曝光时间
        # 后扣减计数），造成丢帧与高延迟。禁用基类轮询：帧仅由
        # CameraWorker 手动触发/获取。
        self._fetch_thread_run = False

    def _fetch_data(self):
        if not self._acquiring or self._triggered == 0:
            return None

        # 曝光拟真等待；拖动等交互场景可开启快速预览跳过，保证画面流畅
        if not self._fast_preview:
            time.sleep(self._exposure_time)
        self._triggered -= 1

        width = self._roi.width // self._binning.h
        height = self._roi.height // self._binning.v

        # 位移台位置（µm）→ 样本像素坐标，视野中心对准台面位置
        xstart = int(self._stage.position["x"] / self._pixel_size - width / 2)
        ystart = int(self._stage.position["y"] / self._pixel_size - height / 2)
        xend, yend = xstart + width, ystart + height
        # 记录本帧对应的台位（供 GUI 做平移预补偿，消除拖动跳变）
        self._last_frame_pos = dict(self._stage.position)

        # 视野越界部分填充黑色（同 StageAwareCamera 的处理）；样本已是灰度图
        if (
            xstart < 0
            or ystart < 0
            or xend > self._sample.shape[1]
            or yend > self._sample.shape[0]
        ):
            gray = np.zeros((height, width), dtype=np.uint8)
            sx0, sy0 = max(0, xstart), max(0, ystart)
            sx1, sy1 = min(xend, self._sample.shape[1]), min(yend, self._sample.shape[0])
            gray[sy0 - ystart : sy1 - ystart, sx0 - xstart : sx1 - xstart] = self._sample[
                sy0:sy1, sx0:sx1
            ]
        else:
            gray = self._sample[ystart:yend, xstart:xend]

        # Z 轴离焦：|z| 越大模糊越强（scipy 延迟导入）
        blur = abs(self._stage.position["z"]) / 2.0
        if blur > 0.1:
            import scipy.ndimage

            gray = scipy.ndimage.gaussian_filter(gray, blur)

        # 曝光/增益 → 亮度因子 + 噪声（单次 float32 高斯复用为
        # 散粒/读出两项，比逐像素泊松快约 10 倍）
        factor = (self._exposure_time / _REFERENCE_EXPOSURE_S) * (1.0 + self._gain / 8.0)
        if abs(factor - 1.0) < 1e-6:
            # 快速路径（默认曝光/增益）：uint8 域加性高斯噪声
            # （cv2.randn SIMD，比 float32 逐像素路径快约 8 倍）。
            # cv2.add 不带 dst → 新数组，绝不就地写缓存样本
            import cv2

            noise = np.empty(gray.shape, np.uint8)
            cv2.randn(noise, 0, 6)
            return cv2.add(gray, noise)
        rng = np.random.default_rng()
        n = rng.standard_normal(gray.shape, dtype=np.float32)
        signal = gray.astype(np.float32) * factor
        signal *= 1.0 + 0.04 * n
        signal += 2.5 * n
        return np.clip(signal, 0, 255).astype(np.uint8)
