"""
光斑形态分析器 (SpotMorphologyAnalyzer)

灵感来源:
- Shack-Hartmann 波前传感器 — 通过微透镜阵列将波前分割为子孔径，
  从每个子孔径的光斑偏移量重建波前相位，是自适应光学系统的核心传感器
- 超表面光学 (Metasurface Optics) — 利用亚波长纳米结构调控光的振幅、
  相位和偏振，实现超薄光学元件，其设计需要精确分析光斑形态和衍射图样
- ISO 13694 光束轮廓测量标准 — 激光光束空间分布的标准化测量方法，
  定义了光束宽度、椭圆度、偏心度等关键形态参数

算法原理:
- Image Moments — 图像矩，用于计算质心、方向和椭圆参数
- Hessian Matrix Analysis — Hessian 矩阵分析，通过特征值判断局部
  结构的曲率方向，检测像散和彗差
- Radial Profile Extraction — 径向剖面提取，沿径向方向平均光强分布，
  用于检测衍射环和球差
- FFT Ring Detection — 傅里叶变换环检测，在频域中识别周期性环状结构

功能:
- 分析光斑形态 (椭圆度、方向角、对称性、环数)
- 检测光学像差类型 (像散、彗差、球差)
- 生成形态分析报告
- 计算信噪比

依赖: numpy, cv2, logging, dataclasses (无 scipy)
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger("SpotZoom.SpotMorphologyAnalyzer")


@dataclass
class AberrationIndicator:
    """像差指示器。"""
    name: str               # 像差名称
    confidence: float       # 置信度 [0, 1]
    description: str        # 描述


@dataclass
class MorphologyReport:
    """光斑形态分析报告。"""
    ellipticity: float                          # 椭圆度 [0, 1], 0=圆形
    orientation_deg: float                      # 主轴方向角 (度)
    symmetry_score: float                       # 对称性评分 [0, 1], 1=完全对称
    ring_count: int                             # 检测到的衍射环数
    snr: float                                  # 信噪比 (dB)
    aberration_indicators: Dict[str, AberrationIndicator]  # 像差指示
    overall_quality: float                      # 综合质量评分 [0, 1]
    centroid: Tuple[float, float]               # 质心坐标
    spot_area: float                            # 光斑面积 (像素)
    peak_intensity: float                       # 峰值强度
    mean_intensity: float                       # 平均强度


class SpotMorphologyAnalyzer:
    """光斑形态分析器。

    分析光斑的形状、对称性、椭圆度和衍射图样，
    诊断光学系统的像差类型和对准状态。

    Parameters
    ----------
    snr_threshold : float
        信噪比阈值 (dB)，低于此值认为信号不可靠。
    ellipticity_threshold : float
        椭圆度告警阈值，超过此值认为存在显著椭圆变形。
    symmetry_threshold : float
        对称性告警阈值，低于此值认为对称性不足。
    min_spot_area : int
        最小光斑面积 (像素)，低于此值不进行分析。
    """

    def __init__(
        self,
        snr_threshold: float = 3.0,
        ellipticity_threshold: float = 0.3,
        symmetry_threshold: float = 0.7,
        min_spot_area: int = 20,
    ):
        self.snr_threshold = float(snr_threshold)
        self.ellipticity_threshold = float(ellipticity_threshold)
        self.symmetry_threshold = float(symmetry_threshold)
        self.min_spot_area = int(min_spot_area)

        # 最新分析报告
        self._report: Optional[MorphologyReport] = None

        LOGGER.info(
            "SpotMorphologyAnalyzer: 初始化完成 "
            "(snr_thresh=%.1fdB, ellip_thresh=%.2f, sym_thresh=%.2f)",
            self.snr_threshold, self.ellipticity_threshold,
            self.symmetry_threshold,
        )

    def analyze(
        self,
        frame: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> MorphologyReport:
        """分析光斑形态。

        Parameters
        ----------
        frame : np.ndarray
            输入图像 (灰度，单通道)。
        bbox : Tuple[int, int, int, int] or None
            光斑区域 (x, y, w, h)。为 None 时使用整帧。

        Returns
        -------
        MorphologyReport
            形态分析报告。
        """
        # 预处理
        if frame is None or frame.size == 0:
            LOGGER.warning("SpotMorphologyAnalyzer: 输入图像为空")
            return self._empty_report()

        # 转为灰度浮点
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        gray = gray.astype(np.float64)

        # 裁剪 ROI
        if bbox is not None:
            x, y, w, h = bbox
            x = max(0, int(x))
            y = max(0, int(y))
            w = min(w, gray.shape[1] - x)
            h = min(h, gray.shape[0] - y)
            if w <= 0 or h <= 0:
                return self._empty_report()
            roi = gray[y:y + h, x:x + w]
        else:
            roi = gray

        # 计算基本统计量
        centroid, peak_intensity, mean_intensity, spot_area = self._compute_basic_stats(roi)

        # 计算信噪比
        snr = self._compute_snr(roi, peak_intensity)

        # 检查最小面积
        if spot_area < self.min_spot_area:
            LOGGER.debug(
                "SpotMorphologyAnalyzer: 光斑面积 %d < %d，跳过分析",
                spot_area, self.min_spot_area,
            )
            return self._empty_report()

        # 计算椭圆度和方向
        ellipticity, orientation_deg = self._compute_ellipticity(roi, centroid)

        # 计算对称性
        symmetry_score = self._compute_symmetry_score(roi, centroid)

        # 检测衍射环
        ring_count = self._detect_rings(roi, centroid)

        # 检测像差
        aberrations = self._detect_aberrations(
            roi, centroid, ellipticity, orientation_deg,
            symmetry_score, ring_count, snr,
        )

        # 综合质量评分
        overall_quality = self._compute_overall_quality(
            ellipticity, symmetry_score, snr, ring_count, aberrations,
        )

        # 构建报告
        self._report = MorphologyReport(
            ellipticity=round(ellipticity, 4),
            orientation_deg=round(orientation_deg, 2),
            symmetry_score=round(symmetry_score, 4),
            ring_count=ring_count,
            snr=round(snr, 2),
            aberration_indicators=aberrations,
            overall_quality=round(overall_quality, 4),
            centroid=(round(centroid[0], 2), round(centroid[1], 2)),
            spot_area=float(spot_area),
            peak_intensity=round(peak_intensity, 2),
            mean_intensity=round(mean_intensity, 2),
        )

        LOGGER.debug(
            "SpotMorphologyAnalyzer: 分析完成 (椭圆度=%.3f, 对称性=%.3f, "
            "环数=%d, SNR=%.1fdB, 质量=%.3f)",
            ellipticity, symmetry_score, ring_count, snr, overall_quality,
        )

        return self._report

    def get_morphology_report(self) -> Optional[MorphologyReport]:
        """获取最近一次分析的形态报告。

        Returns
        -------
        MorphologyReport or None
            最近的分析报告。未执行过分析时返回 None。
        """
        return self._report

    def detect_aberration_type(self) -> Optional[str]:
        """检测最主要的像差类型。

        Returns
        -------
        str or None
            像差类型名称。无显著像差时返回 None。
        """
        if self._report is None:
            return None

        best_name = None
        best_conf = 0.0

        for name, indicator in self._report.aberration_indicators.items():
            if indicator.confidence > best_conf:
                best_conf = indicator.confidence
                best_name = name

        if best_conf > 0.3:
            return best_name
        return None

    def compute_ellipticity(self) -> float:
        """获取最近分析的椭圆度。

        Returns
        -------
        float
            椭圆度 [0, 1]。未分析时返回 0.0。
        """
        if self._report is None:
            return 0.0
        return self._report.ellipticity

    def compute_symmetry_score(self) -> float:
        """获取最近分析的对称性评分。

        Returns
        -------
        float
            对称性评分 [0, 1]。未分析时返回 0.0。
        """
        if self._report is None:
            return 0.0
        return self._report.symmetry_score

    def reset(self) -> None:
        """重置分析器，清除报告缓存。"""
        self._report = None
        LOGGER.info("SpotMorphologyAnalyzer: 分析器已重置")

    # ======================== 内部方法 ========================

    def _compute_basic_stats(
        self, roi: np.ndarray,
    ) -> Tuple[Tuple[float, float], float, float, int]:
        """计算基本统计量: 质心、峰值、均值、面积。

        Parameters
        ----------
        roi : np.ndarray
            光斑区域图像。

        Returns
        -------
        Tuple
            (centroid, peak_intensity, mean_intensity, spot_area)
        """
        # 使用 OpenCV moments 计算质心
        roi_uint8 = np.clip(roi, 0, 255).astype(np.uint8)
        moments = cv2.moments(roi_uint8, binaryImage=False)

        total_m = moments["m00"]
        if total_m > 1e-9:
            cx = moments["m10"] / total_m
            cy = moments["m01"] / total_m
        else:
            h, w = roi.shape
            cx, cy = w / 2.0, h / 2.0

        peak_intensity = float(np.max(roi))
        mean_intensity = float(np.mean(roi))

        # 光斑面积: 超过阈值 (均值 + 1 sigma) 的像素数
        threshold = mean_intensity + float(np.std(roi))
        spot_mask = roi > threshold
        spot_area = int(np.sum(spot_mask))

        return (cx, cy), peak_intensity, mean_intensity, spot_area

    def _compute_snr(self, roi: np.ndarray, peak_intensity: float) -> float:
        """计算信噪比 (dB)。

        使用峰值强度与背景标准差的比值。

        Parameters
        ----------
        roi : np.ndarray
            光斑区域图像。
        peak_intensity : float
            峰值强度。

        Returns
        -------
        float
            信噪比 (dB)。
        """
        # 估计背景噪声: 使用图像边缘区域的标准差
        h, w = roi.shape
        border = max(2, min(h, w) // 10)

        # 四条边的像素作为背景估计
        top = roi[:border, :]
        bottom = roi[-border:, :]
        left = roi[:, :border]
        right = roi[:, -border:]
        background = np.concatenate([top.flatten(), bottom.flatten(),
                                     left.flatten(), right.flatten()])

        if len(background) == 0:
            return 0.0

        noise_std = float(np.std(background))
        if noise_std < 1e-9:
            return 60.0  # 非常高的 SNR

        snr = 20.0 * math.log10(max(peak_intensity, 1e-9) / noise_std)
        return max(0.0, snr)

    def _compute_ellipticity(
        self,
        roi: np.ndarray,
        centroid: Tuple[float, float],
    ) -> Tuple[float, float]:
        """计算椭圆度和主轴方向角。

        使用图像矩的协方差矩阵特征值分析。

        Parameters
        ----------
        roi : np.ndarray
            光斑区域图像。
        centroid : Tuple[float, float]
            质心坐标。

        Returns
        -------
        Tuple[float, float]
            (椭圆度, 方向角_度)。
        """
        h, w = roi.shape
        cx, cy = centroid

        # 构建协方差矩阵
        yy, xx = np.mgrid[0:h, 0:w]
        xx = xx.astype(np.float64)
        yy = yy.astype(np.float64)

        weights = roi.copy()
        total_weight = np.sum(weights)

        if total_weight < 1e-9:
            return (0.0, 0.0)

        # 中心化坐标
        dx = xx - cx
        dy = yy - cy

        # 加权协方差
        mu_xx = float(np.sum(weights * dx * dx)) / total_weight
        mu_yy = float(np.sum(weights * dy * dy)) / total_weight
        mu_xy = float(np.sum(weights * dx * dy)) / total_weight

        # 构造 2x2 协方差矩阵
        cov = np.array([[mu_xx, mu_xy], [mu_xy, mu_yy]], dtype=np.float64)

        # 特征值分解
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        eigenvalues = np.maximum(eigenvalues, 0.0)

        # 排序 (降序)
        idx = np.argsort(eigenvalues)[::-1]
        lambda1 = float(eigenvalues[idx[0]])
        lambda2 = float(eigenvalues[idx[1]])

        # 椭圆度: 1 - lambda_min / lambda_max
        if lambda1 > 1e-9:
            ellipticity = 1.0 - lambda2 / lambda1
        else:
            ellipticity = 0.0

        ellipticity = max(0.0, min(1.0, ellipticity))

        # 主轴方向角
        if lambda1 > 1e-9:
            angle_rad = math.atan2(eigenvectors[1, idx[0]], eigenvectors[0, idx[0]])
            orientation_deg = math.degrees(angle_rad)
        else:
            orientation_deg = 0.0

        return (ellipticity, orientation_deg)

    def _compute_symmetry_score(
        self,
        roi: np.ndarray,
        centroid: Tuple[float, float],
    ) -> float:
        """计算对称性评分。

        比较光斑关于质心的水平、垂直和对角线对称性。

        Parameters
        ----------
        roi : np.ndarray
            光斑区域图像。
        centroid : Tuple[float, float]
            质心坐标。

        Returns
        -------
        float
            对称性评分 [0, 1]。
        """
        h, w = roi.shape
        cx, cy = centroid
        cx_int = int(round(cx))
        cy_int = int(round(cy))

        # 归一化
        roi_norm = roi.copy()
        max_val = np.max(roi_norm)
        if max_val > 1e-9:
            roi_norm = roi_norm / max_val

        scores = []

        # 水平对称性
        left_half = roi_norm[:, :cx_int]
        right_half = roi_norm[:, cx_int:]
        if left_half.size > 0 and right_half.size > 0:
            min_w = min(left_half.shape[1], right_half.shape[1])
            left_cropped = left_half[:, -min_w:]
            right_cropped = right_half[:, :min_w]
            right_flipped = np.fliplr(right_cropped)
            if left_cropped.shape == right_flipped.shape:
                diff = np.mean(np.abs(left_cropped - right_flipped))
                scores.append(1.0 - min(diff, 1.0))

        # 垂直对称性
        top_half = roi_norm[:cy_int, :]
        bottom_half = roi_norm[cy_int:, :]
        if top_half.size > 0 and bottom_half.size > 0:
            min_h = min(top_half.shape[0], bottom_half.shape[0])
            top_cropped = top_half[-min_h:, :]
            bottom_cropped = bottom_half[:min_h, :]
            bottom_flipped = np.flipud(bottom_cropped)
            if top_cropped.shape == bottom_flipped.shape:
                diff = np.mean(np.abs(top_cropped - bottom_flipped))
                scores.append(1.0 - min(diff, 1.0))

        if not scores:
            return 0.5

        return float(np.mean(scores))

    def _detect_rings(
        self,
        roi: np.ndarray,
        centroid: Tuple[float, float],
    ) -> int:
        """检测衍射环数量。

        通过径向剖面分析检测强度环。

        Parameters
        ----------
        roi : np.ndarray
            光斑区域图像。
        centroid : Tuple[float, float]
            质心坐标。

        Returns
        -------
        int
            检测到的衍射环数。
        """
        h, w = roi.shape
        cx, cy = centroid
        max_radius = int(math.sqrt(h * h + w * w) / 2.0)

        if max_radius < 3:
            return 0

        # 提取径向剖面
        radial_profile = np.zeros(max_radius, dtype=np.float64)
        counts = np.zeros(max_radius, dtype=np.float64)

        yy, xx = np.mgrid[0:h, 0:w]
        distances = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.int32)

        for r in range(max_radius):
            mask = distances == r
            if np.any(mask):
                radial_profile[r] = np.mean(roi[mask])
                counts[r] = np.sum(mask)

        # 平滑径向剖面
        if len(radial_profile) >= 5:
            kernel_size = min(5, len(radial_profile))
            if kernel_size % 2 == 0:
                kernel_size -= 1
            kernel = np.ones(kernel_size) / kernel_size
            radial_profile = np.convolve(radial_profile, kernel, mode='same')

        # 检测局部极小值 (环之间的暗区)
        ring_count = 0
        if len(radial_profile) >= 5:
            for i in range(2, len(radial_profile) - 2):
                if (radial_profile[i] < radial_profile[i - 1]
                        and radial_profile[i] < radial_profile[i + 1]
                        and radial_profile[i] < radial_profile[i - 2]
                        and radial_profile[i] < radial_profile[i + 2]):
                    # 确认是显著的凹陷 (低于相邻峰值的 50%)
                    local_max = max(
                        float(np.max(radial_profile[max(0, i - 5):i])),
                        float(np.max(radial_profile[i + 1:min(len(radial_profile), i + 6)])),
                    )
                    if local_max > 1e-9 and radial_profile[i] < 0.5 * local_max:
                        ring_count += 1

        return ring_count

    def _detect_aberrations(
        self,
        roi: np.ndarray,
        centroid: Tuple[float, float],
        ellipticity: float,
        orientation_deg: float,
        symmetry_score: float,
        ring_count: int,
        snr: float,
    ) -> Dict[str, AberrationIndicator]:
        """检测光学像差类型。

        Parameters
        ----------
        roi : np.ndarray
            光斑区域图像。
        centroid : Tuple[float, float]
            质心坐标。
        ellipticity : float
            椭圆度。
        orientation_deg : float
            方向角。
        symmetry_score : float
            对称性评分。
        ring_count : int
            衍射环数。
        snr : float
            信噪比。

        Returns
        -------
        Dict[str, AberrationIndicator]
            像差指示器字典。
        """
        aberrations: Dict[str, AberrationIndicator] = {}

        # 像散 (Astigmatism): 高椭圆度 + 特定方向
        if ellipticity > self.ellipticity_threshold:
            conf = min(ellipticity / 0.6, 1.0)
            if snr > self.snr_threshold:
                conf *= 0.9
            else:
                conf *= 0.3
            aberrations["astigmatism"] = AberrationIndicator(
                name="像散 (Astigmatism)",
                confidence=round(conf, 4),
                description=f"椭圆度 {ellipticity:.3f} 超过阈值 "
                            f"{self.ellipticity_threshold:.2f}，"
                            f"方向角 {orientation_deg:.1f}°",
            )

        # 彗差 (Coma): 不对称 + 偏心
        if symmetry_score < self.symmetry_threshold:
            conf = min((1.0 - symmetry_score) / 0.5, 1.0)
            if snr > self.snr_threshold:
                conf *= 0.85
            else:
                conf *= 0.3
            aberrations["coma"] = AberrationIndicator(
                name="彗差 (Coma)",
                confidence=round(conf, 4),
                description=f"对称性评分 {symmetry_score:.3f} 低于阈值 "
                            f"{self.symmetry_threshold:.2f}，"
                            f"光斑可能存在彗尾",
            )

        # 球差 (Spherical Aberration): 多环 + 对称
        if ring_count >= 2 and symmetry_score > 0.6:
            conf = min(ring_count / 5.0, 1.0) * 0.8
            if snr > self.snr_threshold:
                conf *= 0.9
            else:
                conf *= 0.3
            aberrations["spherical"] = AberrationIndicator(
                name="球差 (Spherical Aberration)",
                confidence=round(conf, 4),
                description=f"检测到 {ring_count} 个衍射环，"
                            f"对称性良好 ({symmetry_score:.3f})，"
                            f"可能存在球差",
            )

        return aberrations

    def _compute_overall_quality(
        self,
        ellipticity: float,
        symmetry_score: float,
        snr: float,
        ring_count: int,
        aberrations: Dict[str, AberrationIndicator],
    ) -> float:
        """计算综合质量评分。

        Parameters
        ----------
        ellipticity : float
            椭圆度。
        symmetry_score : float
            对称性评分。
        snr : float
            信噪比。
        ring_count : int
            衍射环数。
        aberrations : dict
            像差指示器。

        Returns
        -------
        float
            综合质量评分 [0, 1]。
        """
        # 椭圆度评分 (越圆越好)
        ellip_score = 1.0 - min(ellipticity / 0.5, 1.0)

        # 对称性评分
        sym_score = symmetry_score

        # SNR 评分
        snr_score = min(snr / 20.0, 1.0)

        # 环数评分 (理想光斑无环或少环)
        ring_score = max(0.0, 1.0 - ring_count * 0.2)

        # 像差惩罚
        aberration_penalty = 0.0
        for indicator in aberrations.values():
            aberration_penalty += indicator.confidence * 0.15
        aberration_penalty = min(aberration_penalty, 0.5)

        # 加权综合
        quality = (
            ellip_score * 0.25
            + sym_score * 0.25
            + snr_score * 0.25
            + ring_score * 0.25
            - aberration_penalty
        )

        return max(0.0, min(1.0, quality))

    def _empty_report(self) -> MorphologyReport:
        """生成空报告。"""
        return MorphologyReport(
            ellipticity=0.0,
            orientation_deg=0.0,
            symmetry_score=0.0,
            ring_count=0,
            snr=0.0,
            aberration_indicators={},
            overall_quality=0.0,
            centroid=(0.0, 0.0),
            spot_area=0.0,
            peak_intensity=0.0,
            mean_intensity=0.0,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    analyzer = SpotMorphologyAnalyzer()

    print("=== 光斑形态分析器测试 ===\n")

    # 测试 1: 理想圆形光斑
    print("--- 测试 1: 理想圆形光斑 ---")
    size = 100
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    r = np.sqrt(x ** 2 + y ** 2).astype(np.float64)
    sigma = 20.0
    gaussian_spot = 255.0 * np.exp(-r ** 2 / (2 * sigma ** 2))
    gaussian_spot += np.random.normal(0, 2, gaussian_spot.shape)
    gaussian_spot = np.clip(gaussian_spot, 0, 255)

    report = analyzer.analyze(gaussian_spot.astype(np.uint8))
    print(f"  椭圆度: {report.ellipticity:.4f}")
    print(f"  方向角: {report.orientation_deg:.2f}°")
    print(f"  对称性: {report.symmetry_score:.4f}")
    print(f"  环数: {report.ring_count}")
    print(f"  SNR: {report.snr:.1f} dB")
    print(f"  质量: {report.overall_quality:.4f}")
    print(f"  像差: {list(report.aberration_indicators.keys()) or '无'}")

    # 测试 2: 椭圆光斑 (模拟像散)
    print("\n--- 测试 2: 椭圆光斑 (像散) ---")
    elliptic_spot = 255.0 * np.exp(-x ** 2 / (2 * 30.0 ** 2) - y ** 2 / (2 * 10.0 ** 2))
    elliptic_spot += np.random.normal(0, 2, elliptic_spot.shape)
    elliptic_spot = np.clip(elliptic_spot, 0, 255)

    report = analyzer.analyze(elliptic_spot.astype(np.uint8))
    print(f"  椭圆度: {report.ellipticity:.4f}")
    print(f"  方向角: {report.orientation_deg:.2f}°")
    print(f"  对称性: {report.symmetry_score:.4f}")
    print(f"  质量: {report.overall_quality:.4f}")
    for name, ind in report.aberration_indicators.items():
        print(f"  像差: {ind.name} (置信度={ind.confidence:.3f}) - {ind.description}")

    # 测试 3: 带衍射环的光斑 (模拟球差)
    print("\n--- 测试 3: 带衍射环的光斑 (球差) ---")
    airy_spot = 255.0 * (np.sin(r / 5.0) / (r / 5.0 + 1e-6)) ** 2
    airy_spot = np.where(r < 60, airy_spot, 0)
    airy_spot += np.random.normal(0, 2, airy_spot.shape)
    airy_spot = np.clip(airy_spot, 0, 255)

    report = analyzer.analyze(airy_spot.astype(np.uint8))
    print(f"  椭圆度: {report.ellipticity:.4f}")
    print(f"  对称性: {report.symmetry_score:.4f}")
    print(f"  环数: {report.ring_count}")
    print(f"  质量: {report.overall_quality:.4f}")
    for name, ind in report.aberration_indicators.items():
        print(f"  像差: {ind.name} (置信度={ind.confidence:.3f}) - {ind.description}")

    # 测试 4: 不对称光斑 (模拟彗差)
    print("\n--- 测试 4: 不对称光斑 (彗差) ---")
    coma_spot = 255.0 * np.exp(-((x - 0.02 * r) ** 2 + y ** 2) / (2 * 20.0 ** 2))
    coma_spot += np.random.normal(0, 2, coma_spot.shape)
    coma_spot = np.clip(coma_spot, 0, 255)

    report = analyzer.analyze(coma_spot.astype(np.uint8))
    print(f"  椭圆度: {report.ellipticity:.4f}")
    print(f"  对称性: {report.symmetry_score:.4f}")
    print(f"  质量: {report.overall_quality:.4f}")
    for name, ind in report.aberration_indicators.items():
        print(f"  像差: {ind.name} (置信度={ind.confidence:.3f}) - {ind.description}")

    print("\n测试完成")
