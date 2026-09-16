"""视觉层（PLAN 第 2 节）：Unicode 安全读图、classic OpenCV 检测、多球跟踪。

- 分割后端可插拔（当前 classic；接口保留 Cellpose/micro-sam/YOLO 扩展点）。
- 连续丢帧 / 遮挡 / 低置信度 -> ``VisionResult.uncertain=True``（DETECTION_UNCERTAIN）。
- 合成帧验收：球心误差 <= 2 px，衬底边界 IoU >= 0.95。
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .models import Obstacle, Particle, Point, SubstrateRegion


# ---------------------------------------------------------------- io
def load_image_unicode(path: str) -> np.ndarray:
    """中文/Unicode 路径安全读取：np.fromfile + cv2.imdecode。"""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot decode image: {path}")
    return img


# ---------------------------------------------------------------- results
@dataclass
class VisionResult:
    frame_id: int
    substrate: Optional[SubstrateRegion] = None
    obstacles: List[Obstacle] = field(default_factory=list)
    particles: List[Particle] = field(default_factory=list)
    uncertain: bool = False
    uncertain_reason: str = ""
    confidence: float = 0.0
    frame: Optional[np.ndarray] = None  # 叠加调试用
    offscreen_ids: List[int] = field(default_factory=list)  # 需求2：外推元素


# ---------------------------------------------------------------- classic
class ClassicDetector:
    """基于阈值的 classic 检测：亮斑=球、暗区=障碍、大亮区=衬底。

    合成仿真帧约定：背景 0，衬底 ~200，障碍 ~40，粒子 ~255（软边）。
    """

    def __init__(self,
                 min_particle_radius_px: float = 4.0,
                 min_obstacle_area_px2: float = 120.0,
                 min_confidence: float = 0.45) -> None:
        self.min_particle_radius_px = min_particle_radius_px
        self.min_obstacle_area_px2 = min_obstacle_area_px2
        self.min_confidence = min_confidence

    # -- substrate
    def detect_substrate(self, frame: np.ndarray) -> Optional[SubstrateRegion]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 500:
            return None
        eps = 0.005 * cv2.arcLength(c, True)
        poly = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(float)
        # 轮廓点向内收缩 ~1.5px 抵消阈值膨胀带来的边界外扩
        poly = self._shrink_polygon(poly.tolist(), 1.5)
        return SubstrateRegion(polygon=[(p[0], p[1]) for p in poly], safety_margin_px=0.0)

    @staticmethod
    def _shrink_polygon(poly: Sequence[Sequence[float]], d: float) -> List[List[float]]:
        cx = sum(p[0] for p in poly) / len(poly)
        cy = sum(p[1] for p in poly) / len(poly)
        out = []
        for p in poly:
            vx, vy = p[0] - cx, p[1] - cy
            n = math.hypot(vx, vy) or 1.0
            out.append([p[0] - vx / n * d, p[1] - vy / n * d])
        return out

    # -- particles
    def detect_particles(self, frame: np.ndarray,
                         expected_radius_px: Optional[float] = None
                         ) -> Tuple[List[Tuple[Point, float, float]], List[dict]]:
        """返回 ([(center, radius, confidence)], ambiguous_blobs)。

        粘连判定：面积超限（>1.6x 单球）或二阶矩伸长比 > 1.35 的细长 blob。
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found: List[Tuple[Point, float, float]] = []
        ambiguous: List[dict] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area <= 0:
                continue
            perim = cv2.arcLength(c, True)
            r_eq = math.sqrt(area / math.pi)
            # 多边形近似后的 circularity（抑制像素阶梯导致的周长虚高）
            approx = cv2.approxPolyDP(c, 0.02 * perim, True)
            pa = cv2.contourArea(approx)
            pp = cv2.arcLength(approx, True)
            circularity = min(1.0, 4 * math.pi * pa / (pp * pp + 1e-9)) \
                if pp > 0 else 0.0
            m = cv2.moments(c)
            mu20, mu02 = abs(m["mu20"]), abs(m["mu02"])
            ratio = math.sqrt(max(mu20, mu02) / max(min(mu20, mu02), 1e-6))
            merged = (expected_radius_px and
                      (r_eq > expected_radius_px * 1.6 or
                       (ratio > 1.35 and r_eq > expected_radius_px * 0.8)))
            if merged:
                ambiguous.append({"area": float(area), "ratio": float(ratio)})
                continue
            if r_eq < self.min_particle_radius_px:
                continue
            # 小 blob 的周长受像素阶梯/高光分裂主导，圆度阈值按半径放宽：
            # r_eq>=6 用 0.5（正常圆判定），更小的球放宽到 0.15（真实小球可以很小）
            circ_min = 0.5 if r_eq >= 6.0 else 0.15
            if circularity < circ_min:
                continue
            if m["m00"] == 0:
                continue
            cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
            # circularity 阈值归一（0.5 以上视为圆），叠加半径一致性
            conf = min(1.0, circularity / 0.5)
            if expected_radius_px:
                conf *= math.exp(-((r_eq - expected_radius_px) / expected_radius_px) ** 2)
            found.append(((cx, cy), r_eq, conf))
        return found, ambiguous

    # -- obstacles
    def detect_obstacles(self, frame: np.ndarray,
                         substrate: Optional[SubstrateRegion]) -> List[Obstacle]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, dark = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
        if substrate is not None:
            smask = np.zeros_like(dark)
            cv2.fillPoly(smask, [np.array(substrate.polygon, dtype=np.int32)], 255)
            dark = cv2.bitwise_and(dark, smask)
        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        obstacles: List[Obstacle] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_obstacle_area_px2:
                continue
            # 保守近似：凸包覆盖粘连凹口，避免规划模型出现假缝隙
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            hull_perim = cv2.arcLength(hull, True)
            m = cv2.moments(hull)
            if m["m00"] == 0:
                continue
            cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
            circularity = min(1.0, 4 * math.pi * hull_area /
                              (hull_perim * hull_perim + 1e-9))
            if circularity >= 0.75:
                obstacles.append(Obstacle(kind="circle", center=(cx, cy),
                                          radius=math.sqrt(hull_area / math.pi)))
            else:
                eps = 0.005 * hull_perim
                poly = cv2.approxPolyDP(hull, eps, True).reshape(-1, 2).astype(float)
                obstacles.append(Obstacle(kind="polygon",
                                          polygon=[(p[0], p[1]) for p in poly]))
        return obstacles


# ---------------------------------------------------------------- yolo
class YoloDetector:
    """YOLO 检测后端（PLAN 第 2 节可插拔扩展点）。

    与 ClassicDetector 同协议：detect_substrate / detect_particles /
    detect_obstacles，可直接注入 VisionPipeline。
    - 衬底：模型只检测球，衬底取全帧内缩矩形（工作域=视野）。
    - 障碍：返回空（动态障碍由快照层从其他球构造）。
    - ultralytics 为懒加载可选依赖。
    """

    def __init__(self, weights: str, imgsz: int = 640, conf: float = 0.25,
                 device: Optional[str] = None,
                 substrate_inset_px: float = 10.0) -> None:
        from ultralytics import YOLO  # 懒加载
        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.conf = conf
        self.device = device
        self.substrate_inset_px = substrate_inset_px
        self.min_confidence = conf
        self.last_raw: List[Tuple[Point, float, float]] = []
        self._infer_lock = threading.Lock()   # 共享实例多线程推理串行化

    def detect_substrate(self, frame: np.ndarray) -> SubstrateRegion:
        h, w = frame.shape[:2]
        d = self.substrate_inset_px
        return SubstrateRegion(
            polygon=[(d, d), (w - d, d), (w - d, h - d), (d, h - d)],
            safety_margin_px=4.0)

    def detect_particles(self, frame: np.ndarray,
                         expected_radius_px: Optional[float] = None
                         ) -> Tuple[List[Tuple[Point, float, float]], List[dict]]:
        with self._infer_lock:   # predict 非线程安全，串行化
            res = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf,
                                     verbose=False,
                                     device=self.device)[0]
        found: List[Tuple[Point, float, float]] = []
        for b in res.boxes:
            x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            r = max((x2 - x1), (y2 - y1)) / 2
            c = float(b.conf)
            if expected_radius_px:
                c *= math.exp(-((r - expected_radius_px) / expected_radius_px)
                              ** 2)
            found.append(((cx, cy), r, c))
        self.last_raw = found
        return found, []

    def detect_obstacles(self, frame: np.ndarray,
                         substrate: Optional[SubstrateRegion]) -> List[Obstacle]:
        return []


# ---------------------------------------------------------------- tracker
@dataclass
class _Track:
    track_id: int
    position: Point
    radius: float
    confidence: float
    lost_count: int = 0
    last_frame: int = -1        # 最后一次被轮询的帧
    last_seen_frame: int = -1   # 最后一次真实检测帧（coast 不更新）
    velocity: Point = (0.0, 0.0)
    history: List[Point] = field(default_factory=list)


class ParticleTracker:
    """最近邻多球跟踪：速度预测 + 贪心匹配；重叠时 coast 保 ID（AG-03）。"""

    def __init__(self, max_jump_px: float = 40.0, max_lost_frames: int = 3,
                 velocity_damp: float = 0.5,
                 keep_lost_frames: Optional[int] = None) -> None:
        self.max_jump_px = max_jump_px
        self.max_lost_frames = max_lost_frames
        # 需求2：丢检后仍保留 track 的帧数上限（离屏/遮挡不立刻丢 ID）。
        # None = 沿用旧行为（max_lost_frames）。
        self.keep_lost_frames = (
            int(max_lost_frames) if keep_lost_frames is None
            else max(int(max_lost_frames), int(keep_lost_frames)))
        self.velocity_damp = velocity_damp
        self._tracks: Dict[int, _Track] = {}
        self._next_id = 1
        self._shift: Point = (0.0, 0.0)   # 需求2：本帧台位命令位移（像素）

    def add_shift(self, dx_px: float, dy_px: float) -> None:
        """登记一次台位命令位移（图像像素，已含方向符号）。

        需求2：离屏/丢检期间球随载物台运动，按命令位移外推比开环速度外推
        更贴近真实位置（控制器据此继续定位运动）。
        """
        self._shift = (self._shift[0] + dx_px, self._shift[1] + dy_px)

    def update(self, detections: Sequence[Tuple[Point, float, float]],
               frame_id: int) -> List[Particle]:
        shift, self._shift = self._shift, (0.0, 0.0)
        has_shift = abs(shift[0]) > 1e-9 or abs(shift[1]) > 1e-9
        # 预测位置（有台位命令 -> 按命令位移；否则速度阻尼外推）
        preds: Dict[int, Point] = {}
        for tid, t in self._tracks.items():
            if has_shift:
                preds[tid] = (t.position[0] + shift[0],
                              t.position[1] + shift[1])
                continue
            dt = max(1, frame_id - t.last_frame)
            preds[tid] = (t.position[0] + t.velocity[0] * dt * self.velocity_damp,
                          t.position[1] + t.velocity[1] * dt * self.velocity_damp)
        # 候选对（track id 与检测索引分开放置，避免命名空间冲突）
        cand: List[Tuple[float, int, int]] = []
        for tid, pp in preds.items():
            for i, (pos, _r, _c) in enumerate(detections):
                d = math.hypot(pos[0] - pp[0], pos[1] - pp[1])
                # Offline/video sources may deliver frames with a larger
                # frame-id gap than the live camera. Scale the gate by the
                # elapsed frame count while keeping it bounded.
                last_frame = self._tracks[tid].last_frame
                dt = max(1, frame_id - last_frame)
                gate = self.max_jump_px * min(4.0, math.sqrt(float(dt)))
                if d <= gate:
                    cand.append((d, tid, i))
        cand.sort()
        used_t, used_d, matches = set(), set(), {}
        for _d, tid, i in cand:
            if tid in used_t or i in used_d:
                continue
            matches[tid] = i
            used_t.add(tid)
            used_d.add(i)
        # 更新已匹配 / coast 未匹配
        alive: List[Particle] = []
        dropped: List[int] = []
        for tid, t in self._tracks.items():
            if tid in matches:
                pos, r, conf = detections[matches[tid]]
                dt = max(1, frame_id - t.last_frame)
                t.velocity = ((pos[0] - t.position[0]) / dt,
                              (pos[1] - t.position[1]) / dt)
                t.position, t.radius, t.confidence = pos, r, conf
                t.lost_count = 0
                t.last_frame = t.last_seen_frame = frame_id
                pframe = frame_id
            else:
                t.lost_count += 1
                pframe = t.last_seen_frame   # coast：保留最后真实观测帧
                t.last_frame = frame_id
                # 需求2：位置继续外推（有台位命令时按命令位移 = 球随载物台
                # 运动；否则按速度），否则丢检期间位置冻结，控制器会误判
                # "已到位"；外推位置让定位运动继续。
                t.position = preds[tid]
                if has_shift:
                    t.velocity = shift
            t.history.append(t.position)
            if t.lost_count > self.keep_lost_frames:
                dropped.append(tid)
                continue
            alive.append(Particle(track_id=tid, position_px=t.position,
                                  radius_px=t.radius,
                                  confidence=t.confidence * (0.5 ** t.lost_count),
                                  frame_id=pframe,
                                  velocity_px_s=t.velocity,
                                  history_px=list(t.history)))
        for tid in dropped:
            del self._tracks[tid]
        # 未匹配检测 -> 新 track
        for i, (pos, r, conf) in enumerate(detections):
            if i in used_d:
                continue
            t = _Track(self._next_id, pos, r, conf, last_frame=frame_id,
                       history=[pos])
            self._tracks[t.track_id] = t
            alive.append(Particle(track_id=t.track_id, position_px=pos,
                                  radius_px=r, confidence=conf,
                                  frame_id=frame_id, velocity_px_s=(0.0, 0.0),
                                  history_px=[pos]))
            self._next_id += 1
        return alive

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self._shift = (0.0, 0.0)

    def active_particles(self) -> List[Particle]:
        """当前所有 track 的快照（供快照层读取，不推进跟踪状态）。"""
        out = []
        for t in self._tracks.values():
            out.append(Particle(track_id=t.track_id, position_px=t.position,
                                radius_px=t.radius,
                                confidence=t.confidence * (0.5 ** t.lost_count),
                                frame_id=t.last_seen_frame,
                                velocity_px_s=t.velocity,
                                history_px=list(t.history)))
        return out


# ---------------------------------------------------------------- ledger
@dataclass
class LedgerEntry:
    """台账条目：元素最后已知位置/速度（窗口像素）。"""
    track_id: int
    position: Point
    radius: float
    frame_id: int
    velocity: Point = (0.0, 0.0)
    in_frame: bool = True
    shift: Point = (0.0, 0.0)   # 自上次真实观测以来的台位命令位移（像素）


class PositionLedger:
    """元素位置台账（需求2）：始终记录各元素（圆球/障碍）的位置。

    跟踪器丢检、元素离开画面范围后，台账仍按速度外推给出位置，控制器据此
    继续定位运动，而不是直接 ABORTED【FAIL】。仅当台账也无记录（从未见过该
    元素）或外推超过 ``max_age_frames`` 时才允许回到旧的中止路径。
    """

    def __init__(self, max_age_frames: int = 90) -> None:
        self.max_age_frames = max(1, int(max_age_frames))
        self._entries: Dict[int, LedgerEntry] = {}
        self._last_frame: int = -1

    @staticmethod
    def _in_frame(pos: Point, frame_size: Optional[Tuple[int, int]]) -> bool:
        if frame_size is None:
            return True
        w, h = frame_size
        return 0.0 <= pos[0] < w and 0.0 <= pos[1] < h

    def add_shift(self, dx_px: float, dy_px: float) -> None:
        """登记一次台位命令位移（图像像素，已含方向符号）。

        需求2：元素离屏/丢检期间仍随载物台运动，记录命令位移后可按
        "最后已知位置 + 命令位移"给出位置（死推算），持续定位运动。
        """
        if abs(dx_px) <= 1e-9 and abs(dy_px) <= 1e-9:
            return
        for e in self._entries.values():
            e.shift = (e.shift[0] + dx_px, e.shift[1] + dy_px)

    def record(self, particles: Sequence[Particle], frame_id: int,
               frame_size: Optional[Tuple[int, int]] = None) -> None:
        """记录本帧可信检测（通常只记新鲜检测）。"""
        self._last_frame = max(self._last_frame, int(frame_id))
        for p in particles:
            self._entries[p.track_id] = LedgerEntry(
                track_id=p.track_id, position=p.position_px,
                radius=float(p.radius_px), frame_id=int(frame_id),
                velocity=tuple(p.velocity_px_s or (0.0, 0.0)),
                in_frame=self._in_frame(p.position_px, frame_size))

    def predict(self, frame_id: int, known_ids: Sequence[int] = (),
                frame_size: Optional[Tuple[int, int]] = None) -> List[Particle]:
        """台账中"本帧未跟踪到"的元素外推位置。"""
        known = set(known_ids)
        out: List[Particle] = []
        for tid, e in self._entries.items():
            if tid in known:
                continue
            age = int(frame_id) - e.frame_id
            if age <= 0 or age > self.max_age_frames:
                continue
            out.append(self._extrapolated(e, age, frame_id, frame_size))
        return out

    def extrapolate(self, track_id: int, frame_id: Optional[int] = None,
                    frame_size: Optional[Tuple[int, int]] = None
                    ) -> Optional[Particle]:
        """该元素当前位置（含本帧已跟踪的）；台账无记录/超时返回 None。

        需求2：控制器在跟踪器丢检（元素离屏）时用它继续定位，
        而不是直接中止。
        """
        e = self._entries.get(int(track_id))
        if e is None:
            return None
        age = 0 if frame_id is None else max(0, int(frame_id) - e.frame_id)
        if age > self.max_age_frames:
            return None
        return self._extrapolated(e, age, e.frame_id + age, frame_size)

    def _extrapolated(self, e: "LedgerEntry", age: int, frame_id: int,
                      frame_size: Optional[Tuple[int, int]]) -> Particle:
        if abs(e.shift[0]) > 1e-9 or abs(e.shift[1]) > 1e-9:
            # 需求2：期间有台位命令 -> 死推算（元素随载物台运动），
            # 比开环速度外推更接近真实位置。
            pos = (e.position[0] + e.shift[0], e.position[1] + e.shift[1])
        else:
            pos = (e.position[0] + e.velocity[0] * age,
                   e.position[1] + e.velocity[1] * age)
        return Particle(track_id=e.track_id, position_px=pos,
                        radius_px=e.radius,
                        confidence=max(0.05, 0.5 ** min(age, 4)),
                        frame_id=int(frame_id), velocity_px_s=e.velocity,
                        in_frame=self._in_frame(pos, frame_size))

    def entry(self, track_id: int) -> Optional[LedgerEntry]:
        return self._entries.get(int(track_id))

    def elements(self) -> List[LedgerEntry]:
        return list(self._entries.values())

    def prune(self, frame_id: int) -> None:
        stale = [tid for tid, e in self._entries.items()
                 if int(frame_id) - e.frame_id > self.max_age_frames]
        for tid in stale:
            del self._entries[tid]

    def clear(self) -> None:
        self._entries.clear()
        self._last_frame = -1


# ---------------------------------------------------------------- pipeline
class VisionPipeline:
    """检测 + 跟踪 + 不确定性判定。"""

    def __init__(self, detector: Optional[ClassicDetector] = None,
                 tracker: Optional[ParticleTracker] = None,
                 expected_radius_px: Optional[float] = None,
                 ledger: Optional[PositionLedger] = None) -> None:
        self.detector = detector or ClassicDetector()
        self.tracker = tracker or ParticleTracker()
        self.expected_radius_px = expected_radius_px
        self.ledger = ledger if ledger is not None else PositionLedger()

    def apply_stage_shift(self, dx_px: float, dy_px: float) -> None:
        """告知跟踪器/台账本帧台位命令位移（图像像素，已含方向符号）。

        需求2：元素离屏/丢检时按"元素随载物台运动"外推位置，控制器据此
        继续定位运动，而不是直接 ABORTED。
        """
        self.ledger.add_shift(dx_px, dy_px)
        self.tracker.add_shift(dx_px, dy_px)

    def process(self, frame: np.ndarray, frame_id: int,
                expect_particle: bool = True,
                allow_offscreen: bool = False) -> VisionResult:
        """检测一帧。

        ``allow_offscreen=True``（需求2）：元素已离开画面范围（位置在外）时，
        用跟踪器/台账的外推位置补齐元素列表，且不判 ``uncertain``——离屏不是
        检测失败，控制器据外推位置继续定位运动。画内遮挡/丢检仍按检测不确定
        处理（不盲动），保证原有安全语义。
        """
        res = VisionResult(frame_id=frame_id, frame=frame)
        res.substrate = self.detector.detect_substrate(frame)
        if res.substrate is None:
            res.uncertain = True
            res.uncertain_reason = "substrate_not_found"
            return res
        res.obstacles = self.detector.detect_obstacles(frame, res.substrate)
        dets, ambiguous = self.detector.detect_particles(
            frame, expected_radius_px=self.expected_radius_px)
        res.particles = self.tracker.update(dets, frame_id)
        fresh = [p for p in res.particles if p.frame_id == frame_id]
        frame_size = (int(frame.shape[1]), int(frame.shape[0]))
        for p in res.particles:
            p.in_frame = PositionLedger._in_frame(p.position_px, frame_size)
        self.ledger.record(fresh, frame_id, frame_size)
        self.ledger.prune(frame_id)
        extrapolated = False
        if allow_offscreen:
            extra = self.ledger.predict(
                frame_id, [p.track_id for p in res.particles], frame_size)
            if extra:
                res.particles = list(res.particles) + extra
                extrapolated = True
        # 需求2：元素位置已在画面外 -> 台账记录继续有效，用于定位运动
        offscreen = [p for p in res.particles if not p.in_frame]
        if allow_offscreen and offscreen:
            res.offscreen_ids = [p.track_id for p in offscreen]
        if ambiguous:
            res.uncertain = True
            res.uncertain_reason = "merged_particle_blob"
        elif expect_particle and not fresh:
            if res.offscreen_ids:
                # 元素离屏（外推位置）：继续定位，不算检测失败
                res.uncertain_reason = ("offscreen_extrapolated"
                                        if extrapolated else "offscreen_coasted")
            else:
                # 画内无新鲜检测（遮挡/完全丢失）-> 显式不确定，不盲动
                res.uncertain = True
                res.uncertain_reason = ("no_particle_detected" if not dets
                                        else "low_confidence")
        else:
            worst = min((p.confidence for p in fresh), default=1.0)
            if worst < self.detector.min_confidence:
                res.uncertain = True
                res.uncertain_reason = "low_confidence"
            res.confidence = worst
        return res


# ---------------------------------------------------------------- shared detector
# 进程级 YOLO 单例：权重加载 3-5s，避免场景切换/实时检测反复重建。
_shared_detector: Optional[YoloDetector] = None
_shared_lock = threading.Lock()


def get_shared_detector(weights: str) -> YoloDetector:
    """返回共享 YoloDetector（双检锁）。首次调用同步加载权重——
    UI 场景请在启动时调用 warmup_shared_detector_async 预热。"""
    global _shared_detector
    if _shared_detector is None:
        with _shared_lock:
            if _shared_detector is None:
                _shared_detector = YoloDetector(weights)
    return _shared_detector


def try_shared_detector(weights: str) -> Optional[YoloDetector]:
    """非阻塞获取：已加载则返回实例，否则 None（不触发加载）。"""
    return _shared_detector


def warmup_shared_detector_async(weights: str,
                                 on_error=None) -> threading.Thread:
    """后台线程预热权重，UI 启动时调用；完成后 try_shared_detector 命中。"""
    def _w() -> None:
        try:
            get_shared_detector(weights)
        except Exception as exc:  # noqa: BLE001 - 预热失败由调用方提示
            if on_error is not None:
                try:
                    on_error(exc)
                except Exception:  # noqa: BLE001
                    pass
    t = threading.Thread(target=_w, daemon=True, name="yolo-warmup")
    t.start()
    return t
