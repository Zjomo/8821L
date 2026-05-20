"""
SpotZoom Machine Learning v2 — 基于科研前沿开源项目的创新模块集合

本包从以下知名开源项目中提取核心算法思想，针对 SpotZoom 光斑对准系统
进行适配和轻量化实现（纯 numpy，无外部深度学习框架依赖）。

参考开源项目与论文:
──────────────────────────────────────────────────────────────────────
1. HCIPy (https://github.com/ehpor/hcipy)
   - High Contrast Imaging for Python
   - 借鉴: 自适应光学仿真、波前传感与校正、Zernike 分解

2. AOtools (https://github.com/AOtools/aotools)
   - Python 自适应光学工具集
   - 借鉴: 大气湍流相位屏生成、多层传播、闭环 AO 控制

3. Deep Image Prior (https://github.com/DmitryUlyanov/deep-image-prior)
   - 无训练图像恢复
   - 借鉴: 零样本光斑图像增强、去噪先验

4. OpenCLAW (https://github.com/openclaw)
   - 自适应波模拟计算库
   - 借鉴: 光束传播数值模拟、自适应网格方法

5. leap-c (https://github.com/leap-c/leap-c)
   - 学习型预测控制
   - 借鉴: 数据驱动的闭环控制策略优化

6. python-control (https://github.com/python-control/python-control)
   - 控制系统工具箱
   - 借鉴: LQG/H∞ 鲁棒控制器设计、频域分析

7. acados (https://github.com/acados/acados)
   - 非线性最优控制
   - 借鉴: 实时非线性 MPC、约束处理

8. torchdiffeq (https://github.com/rtqichen/torchdiffeq)
   - 神经常微分方程
   - 借鉴: 连续时间动态系统建模

模块列表:
──────────────────────────────────────────────────────────────────────
1. closed_loop_ao_controller.py  — 闭环自适应光学控制器 (← HCIPy/AOtools)
2. fourier_psf_analyzer.py       — 傅里叶 PSF 分析与波前重建 (← HCIPy)
3. deep_image_prior_enhancer.py  — 零样本光斑图像增强器 (← Deep Image Prior)
4. adaptive_beam_propagator.py   — 自适应光束传播模拟器 (← OpenCLAW)
5. data_driven_mpc.py            — 数据驱动模型预测控制器 (← leap-c/acados)
6. lqg_robust_controller.py      — LQG 鲁棒控制器 (← python-control)
"""

__version__ = "1.0.0"
__author__ = "SpotZoom Team"


def _try_import(module_name: str):
    """尝试导入子模块，失败返回 None。"""
    try:
        import importlib
        return importlib.import_module(f".{module_name}", __package__)
    except (ImportError, ModuleNotFoundError, AttributeError, OSError, ValueError):
        return None


# ============ 闭环自适应光学控制器 ============
_cl_ao = _try_import("closed_loop_ao_controller")
if _cl_ao:
    ClosedLoopAOController = _cl_ao.ClosedLoopAOController
    ClosedLoopAOConfig = _cl_ao.ClosedLoopAOConfig
    ClosedLoopAOState = _cl_ao.ClosedLoopAOState
    WFSMeasurement = _cl_ao.WFSMeasurement

# ============ 傅里叶 PSF 分析器 ============
_fourier_psf = _try_import("fourier_psf_analyzer")
if _fourier_psf:
    FourierPSFAnalyzer = _fourier_psf.FourierPSFAnalyzer
    PSFAnalysisResult = _fourier_psf.PSFAnalysisResult
    WavefrontReconstruction = _fourier_psf.WavefrontReconstruction

# ============ 零样本光斑图像增强器 ============
_dip = _try_import("deep_image_prior_enhancer")
if _dip:
    DeepImagePriorEnhancer = _dip.DeepImagePriorEnhancer
    DIPEnhancementResult = _dip.DIPEnhancementResult

# ============ 自适应光束传播模拟器 ============
_beam_prop = _try_import("adaptive_beam_propagator")
if _beam_prop:
    AdaptiveBeamPropagator = _beam_prop.AdaptiveBeamPropagator
    BeamPropagationResult = _beam_prop.BeamPropagationResult
    PropagationConfig = _beam_prop.PropagationConfig

# ============ 数据驱动 MPC 控制器 ============
_dd_mpc = _try_import("data_driven_mpc")
if _dd_mpc:
    DataDrivenMPC = _dd_mpc.DataDrivenMPC
    DataDrivenMPCConfig = _dd_mpc.DataDrivenMPCConfig
    DataDrivenMPCResult = _dd_mpc.DataDrivenMPCResult

# ============ LQG 鲁棒控制器 ============
_lqg = _try_import("lqg_robust_controller")
if _lqg:
    LQGRobustController = _lqg.LQGRobustController
    LQGConfig = _lqg.LQGConfig
    LQGState = _lqg.LQGState
