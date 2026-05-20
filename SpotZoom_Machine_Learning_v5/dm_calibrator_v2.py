"""
变形镜校准器 v2 (DMCalibratorV2)

基于 dmlib (https://github.com/jacopoantonello/dmlib) 的变形镜校准方法，
为 SpotZoom 提供高精度变形镜标定与 Zernike 模态控制能力。

灵感来源:
- dmlib: 变形镜校准与控制库 (https://github.com/jacopoantonello/dmlib)
- slmsuite: SLM/DM 硬件控制 (https://github.com/holodyne/slmsuite)
- python-microscope: 显微镜设备抽象 (https://github.com/python-microscope/microscope)

算法原理:
  1. 影响矩阵标定: 逐驱动器施加电压，测量波前响应，构建影响矩阵
  2. Zernike 模态分解: 将影响矩阵分解为 Zernike 模态基底，
     实现模态空间控制
  3. 闭环校正: 基于影响矩阵的伪逆计算校正电压
  4. 校准持久化: 导出/导入校准文件，避免重复标定

与现有模块的关系:
  - 增强 v3/deformable_mirror_calibrator.py 的校准精度
  - 与 v1/zernike_analyzer.py 的 Zernike 基底协同
  - 与 v3/hardware_abstraction_layer.py 的硬件抽象协同

外部依赖: numpy, scipy, json (标准库)
"""

import numpy as np
import logging
import time
import json
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class CalibrationState(Enum):
    """校准状态。"""
    IDLE = "idle"
    MEASURING_INFLUENCE = "measuring_influence"
    COMPUTING_MATRIX = "computing_matrix"
    ZERNIKE_DECOMPOSITION = "zernike_decomposition"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class DMCalibratorV2Config:
    """变形镜校准器配置。"""
    # 硬件参数
    num_actuators: int = 32           # 驱动器数量
    actuator_voltage_range: Tuple[float, float] = (0.0, 1.0)  # 电压范围
    max_actuator_voltage: float = 1.0  # 最大安全电压

    # 校准参数
    calibration_voltage: float = 0.5   # 标定电压（占最大电压的比例）
    settle_time_ms: float = 50.0       # 驱动器响应稳定时间
    num_averaging_frames: int = 5      # 每次测量的平均帧数

    # Zernike 模态参数
    max_zernike_order: int = 4         # 最大 Zernike 阶数
    zernike_modes: int = 15            # Zernike 模态数量
    regularization_lambda: float = 1e-3  # Tikhonov 正则化参数

    # 质量控制
    min_influence_snr: float = 3.0     # 影响函数最小 SNR
    max_condition_number: float = 1e6  # 最大条件数
    validation_threshold: float = 0.9  # 校准验证阈值

    # 持久化
    auto_save: bool = True             # 自动保存校准结果
    calibration_dir: str = "./calibration_data"


@dataclass
class DMCalibratorV2Report:
    """校准执行报告。"""
    state: CalibrationState = CalibrationState.IDLE
    num_actuators: int = 0
    zernike_modes: int = 0

    # 影响矩阵质量
    influence_matrix_condition: float = 0.0
    influence_matrix_rank: int = 0
    mean_influence_snr: float = 0.0
    min_influence_snr: float = 0.0

    # Zernike 分解质量
    zernike_coverage: float = 0.0      # Zernike 模态覆盖率
    dominant_modes: List[int] = field(default_factory=list)  # 主导模态

    # 校准精度
    reconstruction_error: float = 0.0  # 重建误差
    validation_score: float = 0.0      # 验证得分

    # 性能信息
    total_time_ms: float = 0.0
    calibration_file: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class DMCalibratorV2:
    """变形镜校准器 v2。

    提供高精度变形镜标定，包括影响矩阵测量、Zernike 模态分解、
    校准验证和持久化存储。

    Parameters
    ----------
    config : DMCalibratorV2Config
        校准器配置。
    """

    def __init__(self, config: Optional[DMCalibratorV2Config] = None):
        self.config = config or DMCalibratorV2Config()
        self._influence_matrix: Optional[np.ndarray] = None
        self._zernike_basis: Optional[np.ndarray] = None
        self._control_matrix: Optional[np.ndarray] = None
        self._state = CalibrationState.IDLE
        self._zernike_cache: Dict[int, np.ndarray] = {}

    @property
    def state(self) -> CalibrationState:
        return self._state

    @property
    def is_calibrated(self) -> bool:
        return self._state == CalibrationState.COMPLETED

    def calibrate(
        self,
        measure_callback: callable,
    ) -> DMCalibratorV2Report:
        """执行完整校准流程。

        Parameters
        ----------
        measure_callback : callable
            测量回调函数，签名: measure_callback(actuator_index, voltage) -> np.ndarray
            返回波前测量数据（展平的一维数组）。

        Returns
        -------
        DMCalibratorV2Report
            校准报告。
        """
        report = DMCalibratorV2Report(
            num_actuators=self.config.num_actuators,
            zernike_modes=self.config.zernike_modes,
        )
        t0 = time.perf_counter()

        try:
            # Step 1: 测量影响矩阵
            self._state = CalibrationState.MEASURING_INFLUENCE
            logger.info(f"开始影响矩阵测量 ({self.config.num_actuators} 驱动器)")
            self._measure_influence_matrix(measure_callback, report)

            # Step 2: 计算控制矩阵
            self._state = CalibrationState.COMPUTING_MATRIX
            logger.info("计算控制矩阵")
            self._compute_control_matrix(report)

            # Step 3: Zernike 模态分解
            self._state = CalibrationState.ZERNIKE_DECOMPOSITION
            logger.info("Zernike 模态分解")
            self._zernike_decomposition(report)

            # Step 4: 验证
            self._state = CalibrationState.VALIDATING
            self._validate_calibration(measure_callback, report)

            self._state = CalibrationState.COMPLETED
            logger.info("校准完成")

        except Exception as e:
            self._state = CalibrationState.FAILED
            logger.error(f"校准失败: {e}")
            report.warnings.append(f"校准异常: {str(e)}")

        report.state = self._state
        report.total_time_ms = (time.perf_counter() - t0) * 1000

        # 自动保存
        if self.config.auto_save and self._state == CalibrationState.COMPLETED:
            report.calibration_file = self.save_calibration()

        return report

    def _measure_influence_matrix(
        self, measure_callback: callable, report: DMCalibratorV2Report
    ):
        """测量影响矩阵。"""
        n_act = self.config.num_actuators
        voltage = self.config.calibration_voltage

        # 获取测量维度
        try:
            flat_response = measure_callback(0, 0.0)
            response_shape = flat_response.shape
            n_pixels = flat_response.size
        except Exception as e:
            raise RuntimeError(f"测量回调失败: {e}")

        # 影响矩阵: (n_act, n_pixels)
        influence_matrix = np.zeros((n_act, n_pixels))
        snr_list = []

        for i in range(n_act):
            # 测量基线
            baseline_frames = []
            for _ in range(self.config.num_averaging_frames):
                baseline_frames.append(measure_callback(i, 0.0))
            baseline = np.mean([f.flatten() for f in baseline_frames], axis=0)

            # 施加电压并测量
            active_frames = []
            for _ in range(self.config.num_averaging_frames):
                active_frames.append(measure_callback(i, voltage))
            active = np.mean([f.flatten() for f in active_frames], axis=0)

            # 影响函数 = 有电压 - 无电压
            influence = active - baseline
            influence_matrix[i] = influence

            # 计算 SNR
            noise = np.std(baseline)
            signal = np.max(np.abs(influence))
            snr = signal / max(noise, 1e-10)
            snr_list.append(snr)

            if (i + 1) % 10 == 0:
                logger.info(f"  已测量 {i+1}/{n_act} 驱动器")

        self._influence_matrix = influence_matrix
        report.mean_influence_snr = float(np.mean(snr_list))
        report.min_influence_snr = float(np.min(snr_list))

    def _compute_control_matrix(self, report: DMCalibratorV2Report):
        """计算控制矩阵（伪逆 + 正则化）。"""
        M = self._influence_matrix
        if M is None:
            raise RuntimeError("影响矩阵未测量")

        # Tikhonov 正则化伪逆: (M^T M + λI)^(-1) M^T
        MtM = M @ M.T
        n = MtM.shape[0]
        regularized = MtM + self.config.regularization_lambda * np.eye(n)

        try:
            self._control_matrix = np.linalg.solve(regularized, M)
        except np.linalg.LinAlgError:
            self._control_matrix = np.linalg.pinv(M)
            report.warnings.append("伪逆计算使用 SVD 回退")

        # 质量指标
        report.influence_matrix_condition = float(np.linalg.cond(MtM))
        report.influence_matrix_rank = int(np.linalg.matrix_rank(M, tol=1e-6))

        if report.influence_matrix_condition > self.config.max_condition_number:
            report.warnings.append(
                f"条件数过高 ({report.influence_matrix_condition:.1e})，"
                f"建议增加正则化或减少驱动器数量"
            )

    def _zernike_decomposition(self, report: DMCalibratorV2Report):
        """Zernike 模态分解。"""
        M = self._influence_matrix
        if M is None:
            raise RuntimeError("影响矩阵未测量")

        n_modes = min(self.config.zernike_modes, M.shape[0])
        response_shape = M.shape[1]

        # 生成 Zernike 基底
        side = int(np.sqrt(response_shape))
        zernike_basis = np.zeros((n_modes, response_shape))

        for j in range(n_modes):
            zernike_basis[j] = self._get_zernike_mode(j + 1, side).flatten()

        # 正交化
        zernike_basis = self._orthonormalize(zernike_basis)

        # 投影影响矩阵到 Zernike 空间
        projection = M @ zernike_basis.T  # (n_act, n_modes)
        mode_energy = np.sum(projection ** 2, axis=0)
        total_energy = np.sum(mode_energy)

        if total_energy > 0:
            coverage = np.sum(mode_energy > 0.01 * total_energy) / n_modes
            report.zernike_coverage = float(coverage)

        # 主导模态
        sorted_modes = np.argsort(mode_energy)[::-1]
        report.dominant_modes = [int(m + 1) for m in sorted_modes[:5]]

        self._zernike_basis = zernike_basis

    def _validate_calibration(
        self, measure_callback: callable, report: DMCalibratorV2Report
    ):
        """验证校准质量。"""
        if self._control_matrix is None:
            return

        # 使用前几个驱动器进行验证
        n_test = min(5, self.config.num_actuators)
        errors = []

        for i in range(n_test):
            voltage = self.config.calibration_voltage

            # 测量实际响应
            active = measure_callback(i, voltage).flatten()
            baseline = measure_callback(i, 0.0).flatten()
            actual_response = active - baseline

            # 使用控制矩阵预测
            command = np.zeros(self.config.num_actuators)
            command[i] = voltage
            predicted_response = self._control_matrix.T @ command

            # 计算重建误差
            error = np.sqrt(np.mean((actual_response - predicted_response) ** 2))
            norm = np.sqrt(np.mean(actual_response ** 2))
            if norm > 1e-10:
                errors.append(error / norm)

        if errors:
            report.reconstruction_error = float(np.mean(errors))
            report.validation_score = float(1.0 - np.mean(errors))

            if report.validation_score < self.config.validation_threshold:
                report.warnings.append(
                    f"验证得分偏低 ({report.validation_score:.3f})，"
                    f"建议重新校准"
                )

    def compute_correction(self, wavefront_error: np.ndarray) -> np.ndarray:
        """基于波前误差计算校正电压。

        Parameters
        ----------
        wavefront_error : np.ndarray
            波前误差（展平一维数组）。

        Returns
        -------
        np.ndarray
            校正电压向量。
        """
        if self._control_matrix is None:
            raise RuntimeError("校准未完成，无法计算校正")

        voltages = self._control_matrix @ wavefront_error.flatten()

        # 限幅
        v_min, v_max = self.config.actuator_voltage_range
        voltages = np.clip(voltages, v_min, self.config.max_actuator_voltage)

        return voltages

    def compute_zernike_correction(
        self, zernike_coefficients: np.ndarray
    ) -> np.ndarray:
        """基于 Zernike 系数计算校正电压。

        Parameters
        ----------
        zernike_coefficients : np.ndarray
            Zernike 系数向量。

        Returns
        -------
        np.ndarray
            校正电压向量。
        """
        if self._zernike_basis is None or self._control_matrix is None:
            raise RuntimeError("Zernike 分解未完成")

        # Zernike 系数 -> 波前误差
        wavefront = self._zernike_basis.T @ zernike_coefficients

        # 波前误差 -> 校正电压
        return self.compute_correction(wavefront)

    def save_calibration(self, filepath: Optional[str] = None) -> str:
        """保存校准数据到文件。"""
        if filepath is None:
            Path(self.config.calibration_dir).mkdir(parents=True, exist_ok=True)
            filepath = str(
                Path(self.config.calibration_dir) / "dm_calibration.json"
            )

        data = {
            "version": "2.0",
            "num_actuators": self.config.num_actuators,
            "zernike_modes": self.config.zernike_modes,
            "influence_matrix": self._influence_matrix.tolist() if self._influence_matrix is not None else None,
            "control_matrix": self._control_matrix.tolist() if self._control_matrix is not None else None,
            "zernike_basis": self._zernike_basis.tolist() if self._zernike_basis is not None else None,
            "config": {
                "max_zernike_order": self.config.max_zernike_order,
                "regularization_lambda": self.config.regularization_lambda,
            },
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"校准数据已保存到 {filepath}")
        return filepath

    def load_calibration(self, filepath: str) -> bool:
        """从文件加载校准数据。"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            if data.get("influence_matrix") is not None:
                self._influence_matrix = np.array(data["influence_matrix"])
            if data.get("control_matrix") is not None:
                self._control_matrix = np.array(data["control_matrix"])
            if data.get("zernike_basis") is not None:
                self._zernike_basis = np.array(data["zernike_basis"])

            self._state = CalibrationState.COMPLETED
            logger.info(f"校准数据已从 {filepath} 加载")
            return True

        except Exception as e:
            logger.error(f"加载校准数据失败: {e}")
            return False

    def _get_zernike_mode(self, n: int, size: int) -> np.ndarray:
        """生成第 n 个 Zernike 模式（Noll 编号）。"""
        if n in self._zernike_cache:
            cached = self._zernike_cache[n]
            if cached.shape == (size, size):
                return cached

        from .zernike_common import noll_to_zernike
        mode = noll_to_zernike(n, size)
        self._zernike_cache[n] = mode
        return mode

    @staticmethod
    def _orthonormalize(basis: np.ndarray) -> np.ndarray:
        """Gram-Schmidt 正交归一化。"""
        ortho = np.zeros_like(basis)
        for i in range(len(basis)):
            v = basis[i].copy()
            for j in range(i):
                v -= np.dot(v, ortho[j]) * ortho[j]
            norm = np.linalg.norm(v)
            if norm > 1e-10:
                ortho[i] = v / norm
        return ortho


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = DMCalibratorV2Config(
        num_actuators=12,
        calibration_voltage=0.5,
        num_averaging_frames=3,
    )
    calibrator = DMCalibratorV2(config)

    # 模拟测量回调
    np.random.seed(42)
    n_pixels = 64 * 64

    def mock_measure(actuator_idx, voltage):
        response = np.random.randn(64, 64) * 0.1
        y, x = np.mgrid[:64, :64]
        cx, cy = 32 + (actuator_idx % 4 - 1.5) * 8, 32 + (actuator_idx // 4 - 1.5) * 8
        r2 = (x - cx)**2 + (y - cy)**2
        response += voltage * 5.0 * np.exp(-r2 / (2 * 4**2))
        return response

    report = calibrator.calibrate(mock_measure)
    print(f"校准状态: {report.state.value}")
    print(f"影响矩阵条件数: {report.influence_matrix_condition:.1f}")
    print(f"平均影响SNR: {report.mean_influence_snr:.1f}")
    print(f"Zernike覆盖率: {report.zernike_coverage:.3f}")
    print(f"验证得分: {report.validation_score:.3f}")
    print(f"总时间: {report.total_time_ms:.1f}ms")
    print(f"警告: {report.warnings}")
