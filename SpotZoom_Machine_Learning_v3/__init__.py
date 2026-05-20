"""
SpotZoom Machine Learning v3 — 基于科研前沿开源项目的创新模块集合

本包从以下知名开源项目中提取核心算法思想，针对 SpotZoom 光斑对准系统
进行适配和轻量化实现（纯 numpy + cv2，无外部深度学习框架依赖）。

参考开源项目与论文:
──────────────────────────────────────────────────────────────────────
1. slmsuite (https://github.com/wavefrontshaping/slm_suite)
   - 空间光调制器控制与全息图生成
   - 借鉴: Gerchberg-Saxton 相位恢复、SLM 坐标校准、光斑阵列优化

2. pyRTC (https://github.com/aodtech/pyRTC)
   - Python 实时自适应光学控制框架
   - 借鉴: 高性能 AO 流水线架构、WFS/DM/RTC 抽象层、AI 控制器接口

3. prysm (https://github.com/brandondube/prysm)
   - 光学衍射与像差分析库
   - 借鉴: Strehl 比、MTF、包围能量、Zernike 多项式拟合

4. python-microscope (https://github.com/python-microscope/microscope)
   - 显微镜硬件抽象层
   - 借鉴: 统一设备控制接口、硬件触发同步、网络分布式架构

5. IRIS Autofocus by bunnie studios (https://github.com/bunnie/iris-audio)
   - 基于拉普拉斯方差的自适应对焦算法
   - 借鉴: 焦点评分、曲线拟合、机械稳定性处理

6. DeepTrack2 (https://github.com/DeepTrack/DeepTrack2)
   - 深度学习光学粒子追踪
   - 借鉴: 合成光学数据生成、域随机化、像差模拟

7. dmlib (https://github.com/spacetelescope/dmlib)
   - 变形镜校准库
   - 借鉴: 干涉校准仿真、Zernike 模式控制、影响函数矩阵

8. ML for AO System Identification
   - 机器学习驱动的自适应光学系统辨识
   - 借鉴: 数据驱动系统辨识、频率响应估计、PID 参数优化

模块列表:
──────────────────────────────────────────────────────────────────────
1. slm_holographic_spot_generator.py   — SLM 全息光斑生成器 (← slmsuite)
2. realtime_ao_pipeline.py             — 实时 AO 流水线 (← pyRTC)
3. strehl_quality_assessor.py          — Strehl 质量评估器 (← prysm)
4. hardware_abstraction_layer.py       — 硬件抽象层 (← python-microscope)
5. laplacian_autofocus.py              — 拉普拉斯自动对焦 (← IRIS autofocus)
6. synthetic_data_generator.py         — 合成数据生成器 (← DeepTrack2)
7. deformable_mirror_calibrator.py     — 变形镜校准器 (← dmlib)
8. system_identifier.py                — 系统辨识器 (← ML for AO)
"""

__version__ = "3.0.0"
__author__ = "SpotZoom Team"


def _try_import(module_name: str):
    """尝试导入子模块，失败返回 None。"""
    try:
        import importlib
        return importlib.import_module(f".{module_name}", __package__)
    except (ImportError, ModuleNotFoundError, AttributeError, OSError, ValueError):
        return None


# ============ SLM 全息光斑生成器 ============
_slm = _try_import("slm_holographic_spot_generator")
if _slm:
    SLMHolographicGenerator = _slm.SLMHolographicGenerator
    SLMConfig = _slm.SLMConfig
    HologramResult = _slm.HologramResult

# ============ 实时 AO 流水线 ============
_rtc = _try_import("realtime_ao_pipeline")
if _rtc:
    RealtimeAOPipeline = _rtc.RealtimeAOPipeline
    AOPipelineConfig = _rtc.AOPipelineConfig
    AOPipelineState = _rtc.AOPipelineState

# ============ Strehl 质量评估器 ============
_strehl = _try_import("strehl_quality_assessor")
if _strehl:
    StrehlQualityAssessor = _strehl.StrehlQualityAssessor
    StrehlConfig = _strehl.StrehlConfig
    StrehlAssessment = _strehl.StrehlAssessment

# ============ 硬件抽象层 ============
_hal = _try_import("hardware_abstraction_layer")
if _hal:
    HardwareAbstractionLayer = _hal.HardwareAbstractionLayer
    HALConfig = _hal.HALConfig
    DeviceInfo = _hal.DeviceInfo

# ============ 拉普拉斯自动对焦 ============
_af = _try_import("laplacian_autofocus")
if _af:
    LaplacianAutofocus = _af.LaplacianAutofocus
    AutofocusConfig = _af.AutofocusConfig
    FocusResult = _af.FocusResult

# ============ 合成数据生成器 ============
_synth = _try_import("synthetic_data_generator")
if _synth:
    SyntheticDataGenerator = _synth.SyntheticDataGenerator
    SyntheticDataConfig = _synth.SyntheticDataConfig
    SyntheticSample = _synth.SyntheticSample

# ============ 变形镜校准器 ============
_dm = _try_import("deformable_mirror_calibrator")
if _dm:
    DMCalibrator = _dm.DMCalibrator
    DMCalibConfig = _dm.DMCalibConfig
    DMCalibResult = _dm.DMCalibResult

# ============ 系统辨识器 ============
_sid = _try_import("system_identifier")
if _sid:
    SystemIdentifier = _sid.SystemIdentifier
    SysIdConfig = _sid.SysIdConfig
    SysIdResult = _sid.SysIdResult
