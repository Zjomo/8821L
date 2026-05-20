# SpotZoom 前沿开源项目调研报告

**调研日期**: 2026-05-12
**项目**: SpotZoom — 光斑闭环对准系统 (8821L)
**版本**: v9.0 (第九轮扩展版)

---

## 一、调研背景

SpotZoom 是一个基于 YOLO 深度学习 + 电机平台控制的光学实验室自动化系统，实现光斑自动检测与对准。为提升系统性能和鲁棒性，调研了科研前沿中与本项目相关的开源项目和技术方向。

---

## 二、相关开源项目与技术方向

### 2.1 自适应光学 (Adaptive Optics) 方向

| 项目/技术 | GitHub Stars | 许可证 | 核心能力 | 可借鉴模块 |
|-----------|-------------|--------|----------|-----------|
| **HCIPy** | 138 | MIT | 高对比度成像仿真 | Zernike 像差分解、PSF 仿真、波前传播 |
| **AOtools** | 147 | LGPL-3.0 | 自适应光学工具集 | Zernike 多项式、波前重建、闭环控制 |
| **SOAPY** | 109 | GPL-3.0 | AO 仿真系统 | 模块化 AO 架构、LQG 控制器 |
| **AOloop** | - | - | 实时波前校正 | 闭环反馈架构、实时性能监控 |
| **adaptive_optics_gym** | 18 | - | RL + AO 仿真 | RL 环境设计、Sensorless 控制 |

**可借鉴创新点**:
- **Zernike 像差分解**: 分析光斑形状诊断对准偏差来源 (离焦、像散、彗差等)
- **实时波前质量评估**: 波前误差 RMS 评估方法，改造为光斑对准质量评分
- **增益调度控制**: AO 系统中根据信噪比动态调整积分增益的策略
- **RL 环境设计**: 将光斑对准建模为强化学习问题

### 2.2 视觉伺服 (Visual Servoing) 方向

| 项目/技术 | GitHub Stars | 许可证 | 核心能力 | 可借鉴模块 |
|-----------|-------------|--------|----------|-----------|
| **ViSP** | 878 | GPL-2.0 | 视觉伺服框架 | IBVS/PBVS 控制律、交互矩阵、特征跟踪 |
| **OpenCV contrib** | 4000+ | Apache-2.0 | 计算机视觉 | 光流法、亚像素角点、轮廓矩 |
| **BoxMOT** | 3800+ commits | AGPL-3.0 | 多目标跟踪 | 9 种跟踪算法统一接口、C++ 后端 |
| **ByteTrack** | 5600 | MIT | 轻量跟踪 | 低分框恢复、极轻量模型 |

**可借鉴创新点**:
- **图像雅可比 (Image Jacobian)**: 建立像素偏差到物理位移的映射模型，替代固定步长
- **Broyden 在线估计**: 在线学习雅可比矩阵，自动适应不同光学系统
- **光流法跟踪**: 跟踪光斑在连续帧间的运动
- **低分框恢复**: ByteTrack 的策略恢复低置信度检测

### 2.3 光束分析/光斑追踪方向

| 项目/技术 | GitHub Stars | 许可证 | 核心能力 | 可借鉴模块 |
|-----------|-------------|--------|----------|-----------|
| **BoT-SORT** | 1300 | MIT | 多目标跟踪 | 相机运动补偿 (CMC)、ReID 外观特征 |
| **StrongSORT++** | 854 | GPL-3.0 | 跟踪升级 | GSI 高斯插值、AFLink 轨迹连接 |
| **Laser-beam-profiler-camera** | 15 | GPL-3.0 | 光束分析 | 低成本相机集成、实时光束剖面 |
| **Automatic-Laser-Beam-Alignment** | - | - | 激光对准仿真 | 自动质心检测、反馈控制 |

**可借鉴创新点**:
- **GSI 高斯插值**: 补偿漏检帧，提供平滑轨迹
- **相机运动补偿**: 补偿相机自身运动，提取目标真实位移
- **2D 高斯拟合**: 精确提取光斑中心、束腰半径、椭圆度
- **指向稳定性分析**: 评估光斑位置随时间的抖动

### 2.4 硬件控制/仪器自动化方向

| 项目/技术 | GitHub Stars | 许可证 | 核心能力 | 可借鉴模块 |
|-----------|-------------|--------|----------|-----------|
| **PyMeasure** | 694 | MIT | 仪器控制框架 | Adapter 模式、Procedure 实验管理 |
| **ARTIQ** | 479 | LGPLv3+ | 实时控制 | FPGA RTIO、编译-执行分离 |
| **python-control** | 1900 | BSD-3 | 控制系统设计 | LQR/PID 设计、卡尔曼滤波器、频域分析 |
| **Bluesky** | 300+ | BSD-3 | 实验编排 | Plan 模式、中断恢复、流式数据 |
| **ophyd-async** | 20 | BSD-3 | 异步硬件抽象 | 异步并行控制、协议适配器 |
| **dodal** | - | Apache-2.0 | 光束线设备 | Composite Device、配置服务器 |
| **PyLECO** | 13 | MIT | 分布式控制 | Actor 模式、Coordinator 路由 |

**可借鉴创新点**:
- **Device-Signal 架构**: 电机控制器 HAL 的最佳参考
- **PseudoPositioner**: 将物理电机映射为光学坐标
- **Plan 模式**: Python 生成器描述自适应对准策略
- **LQR 控制器设计**: 用于电机伺服回路的参数整定

### 2.5 深度学习/模型优化方向

| 项目/技术 | GitHub Stars | 许可证 | 核心能力 | 可借鉴模块 |
|-----------|-------------|--------|----------|-----------|
| **Ultralytics YOLO** | 49600 | AGPL-3.0 | YOLO 全系列 | 一键导出 TensorRT/ONNX、内置跟踪 |
| **Stable-Baselines3** | 10000+ | MIT | RL 算法库 | PPO/SAC/TD3、Optuna 调参 |
| **PPAL** | 98 | Apache-2.0 | 主动学习 | 即插即用、不确定性采样 |
| **AL-MDN** | 176 | NVIDIA | 概率建模 AL | 混合密度网络、单次推理不确定性 |
| **Label Studio** | 24500 | Apache-2.0 | 数据标注 | ML 后端、主动学习闭环 |

**可借鉴创新点**:
- **TensorRT 加速**: YOLO11n + TensorRT = 1.5ms/帧 (660+ FPS)
- **PPO/SAC RL**: 自动调优 PID 参数，收敛速度提升 30-50%
- **PPAL 主动学习**: 用 30-50% 标注量达到 90%+ 全量性能
- **Label Studio 集成**: 预标注 + 人工校正的标注流水线

### 2.6 模型预测控制/优化方向 (第三轮新增)

| 项目/技术 | GitHub Stars | 许可证 | 核心能力 | 可借鉴模块 |
|-----------|-------------|--------|----------|-----------|
| **do-mpc** | 1200 | LGPL-3.0 | 非线性 MPC/MHE | 多轴约束协调控制、鲁棒多阶段 MPC |
| **prysm** | 328 | MIT | 物理光学计算 | GPU 加速 PSF、PyTorch 可微后端、Zernike/MTF |
| **pyRTC** | 13 | GPL-3.0 | 实时 AO 控制 | 组件化流水线、共享内存 IPC、Soft/Hard 双模式 |
| **OOPAO** | 49 | 学术开源 | 端到端 AO 仿真 | OOP 光学建模、闭环性能评估 |
| **slmsuite** | 159 | MIT | SLM 控制/全息术 | 自动 Fourier 域校准、GPU 相位检索 |
| **NSER-IBVS** | 15 | AFL-3.0 | 自监督视觉伺服 | Teacher-Student 知识蒸馏、数字孪生训练 |
| **python-microscope** | 84 | GPL-3.0 | 显微镜控制 | 设备聚合模式、硬件触发链 |
| **lowfssim** | 9 | Apache-2.0 | NASA JPL LOWFS | 高保真光学数字孪生、35pm 精度验证 |
| **DeepTrack 2.0** | 200+ | MIT | 深度学习粒子跟踪 | 神经网络光斑跟踪、合成数据生成 |
| **optsim** | 24 | MIT | 光学传播仿真 | SHWFS 图像处理、散斑场仿真 |

**可借鉴创新点**:
- **MPC 多轴协调**: 同时处理 X/Y/Z 约束和耦合，实现最优对准轨迹
- **可微光学**: PyTorch 后端自动微分，梯度下降优化光学系统参数
- **组件化实时控制**: Sensor -> Processor -> Controller -> Corrector 流水线
- **自动化系统标定**: 自动建立相机像素与光学坐标的映射关系
- **知识蒸馏**: 解析控制器作为教师，训练轻量学生网络 (10x+ 加速)
- **数字孪生**: 高保真光学模型 + 实时仿真作为预测控制器

---

## 三、创新模块补充方案

### 第一批 (v1.0, 已实现)

| # | 模块 | 来源启发 | 功能 | 收益 |
|---|------|---------|------|------|
| 1 | KalmanSpotTracker | Centroid Tracking + Kalman Filter | 卡尔曼滤波时序平滑 | 减少检测抖动 |
| 2 | SubPixelCentroid | pyBeamProfiling | 亚像素质心定位 | 精度提升到 0.1px |
| 3 | SpotQualityAnalyzer | AO 波前质量评估 | 多维质量评估 | 智能对准判断 |
| 4 | AdaptiveGainScheduler | AO 增益调度 | 自适应 PID 增益 | 快速稳定收敛 |
| 5 | TrajectoryRecorder | 实验自动化框架 | 轨迹记录分析 | 实验可追溯性 |
| 6 | SafetyManager | 工业控制安全标准 | 安全限位保护 | 防止设备损坏 |
| 7 | ClassicSpotDetector | OpenCV 经典算法 | 无需 YOLO 的检测 | 降低依赖 |

### 第二批 (v2.0, 已实现)

| # | 模块 | 来源启发 | 功能 | 收益 |
|---|------|---------|------|------|
| 8 | **ZernikeAberrationAnalyzer** | AOtools/HCIPy | Zernike 像差分解 | 诊断对准偏差来源 |
| 9 | **ImageJacobianController** | ViSP | 图像雅可比伺服控制 | 自适应步长映射 |
| 10 | **ModelInferenceOptimizer** | Ultralytics/BoxMOT | 推理后端优化 | 推理加速 2-5x |
| 11 | **ActiveLearningCollector** | PPAL/AL-MDN | 主动学习数据采集 | 自动收集训练数据 |
| 12 | **AdaptiveFocusSearcher** | AO 闭环控制 | 黄金分割焦搜索 | 智能焦平面定位 |

### 第三批 (v3.0, 本轮新增)

| # | 模块 | 来源启发 | 功能 | 收益 |
|---|------|---------|------|------|
| 13 | **DeepLearningBackendAccelerator** | DeepSeek/TensorRT | 推理后端自动选择与加速 | 2-3x 推理加速 |
| 14 | **IntelligentAnomalyDetector** | Prometheus/OpenTelemetry | 实时异常检测与告警 | 早期问题发现 |
| 15 | **RLAlignmentEnvironment** | Stable-Baselines3/Gym | 强化学习对准环境 | 智能策略学习 |

### 第四批 (v4.0, 已实现)

| # | 模块 | 来源启发 | 功能 | 收益 |
|---|------|---------|------|------|
| 16 | **WavefrontPredictor** | AOtools/HCIPy | AR(p) 波前误差预测 | 前馈补偿 |
| 17 | **VibrationCompensator** | BoT-SORT/ISO 11670 | FFT 振动检测与补偿 | 抗环境振动 |
| 18 | **GaussianBeamFitter** | pyBeamProfiling/ISO 13694 | 2D 高斯光束拟合 | 光束参数提取 |
| 19 | **EventBus** | Bluesky/Prometheus | 发布-订阅事件总线 | 模块解耦通信 |
| 20 | **BeamStabilityAnalyzer** | ISO 11670 | Allan 方差稳定性分析 | 指向稳定性评估 |

### 第五批 (v5.0, 已实现)

| # | 模块 | 来源启发 | 功能 | 收益 |
|---|------|---------|------|------|
| 21 | **TemporalFusionPredictor** | 多传感器融合/Transformer | EMA+注意力时序融合 | 多源信号融合预测 |
| 22 | **SelfTuningController** | python-control/ARTIQ | 继电反馈 PID 自整定 | 自动参数优化 |
| 23 | **SpotMorphologyAnalyzer** | Shack-Hartmann/ISO 13694 | 光斑形态与像差分析 | 形态诊断 |
| 24 | **DataPipelineOrchestrator** | Bluesky/ARTIQ | 流水线任务编排 | 多步骤自动化 |
| 25 | **DiagnosticHealthMonitor** | Prometheus/OpenTelemetry | 系统健康度综合评估 | 全面监控 |

### 第六批 (v6.0, 本轮新增)

| # | 模块 | 来源启发 | 功能 | 收益 |
|---|------|---------|------|------|
| 26 | **PhaseRetrievalAnalyzer** | HCIPy/AOtools GS 算法 | Gerchberg-Saxton 相位恢复 | 无需波前传感器的波前估计 |
| 27 | **PSFEstimator** | HCIPy/AOtools PSF 仿真 | PSF 质量 Strehl/FWHM 估计 | 实时像质评估 |
| 28 | **ModalController** | AOtools/SOAPY 模态控制 | Zernike 模态分解+积分控制 | AO 标准控制策略 |
| 29 | **AtmosphericTurbulenceSimulator** | HCIPy/AOtools 相位屏 | Kolmogorov 湍流仿真 | 系统鲁棒性测试 |
| 30 | **SmartRefinementController** | ViSP/python-control | 四阶段粗到精收敛策略 | 高精度自动对准 |

### 第八批 (v8.0, 本轮新增)

| # | 模块 | 来源启发 | 功能 | 收益 |
|---|------|---------|------|------|
| 31 | **MultiLayerTurbulenceSimulator** | HCIPy 多层相位屏 | 多层大气湍流+闪烁仿真 | 动态环境仿真真实性大幅提升 |
| 32 | **ComposableOpticalPipeline** | POPPY/NASA STScI | 光学元件级联管线引擎 | 灵活光路配置与PSF评估 |
| 33 | **AutoAlignmentOptimizer** | Rayoptics 自动优化 | 多自由度同时寻优 | 智能自动对准参数优化 |
| 34 | **DeepVibrationPredictor** | Bi-RNN+Attention | 纯numpy深度振动预测 | 非平稳振动环境稳定性 |
| 35 | **DomainRandomizer** | Sim-to-Real 迁移 | 仿真域随机化增强 | 模型鲁棒性与泛化能力 |

### 第九批 (v9.0, 已实现)

| # | 模块 | 来源启发 | 功能 | 收益 |
|---|------|---------|------|------|
| 36 | **TransferLearningAdapter** | Stable-Baselines3 预训练迁移 | 迁移学习适配器 | 跨场景策略复用 |
| 37 | **FederatedLearningCoordinator** | PyMeasure 分布式控制 | 联邦学习协调器 | 多节点协同训练 |
| 38 | **OpticalSystemIdentifier** | python-control 系统辨识 | 光学系统自动辨识 | 模型参数自动估计 |
| 39 | **RobustSpotEstimator** | BoT-SORT RANSAC | 鲁棒估计器 | 抗异常值干扰 |
| 40 | **SpectralAnalyzer** | AOtools 功率谱密度 | 光谱/频率分析器 | 振动频率识别 |

### 第十批 (v10.0, 本轮新增 — 第三轮前沿调研)

| # | 模块 | 来源启发 | 功能 | 收益 |
|---|------|---------|------|------|
| 41 | **MPCController** | do-mpc 非线性 MPC | 模型预测多轴协调控制 | 约束优化+前馈补偿 |
| 42 | **DifferentiableOpticalOptimizer** | prysm 可微后端/NSER-IBVS | 可微光学优化+知识蒸馏 | 梯度优化+策略轻量化 |
| 43 | **RealtimeControlPipeline** | pyRTC 组件化 AO 控制 | 实时控制管线引擎 | 软/硬实时双模式架构 |
| 44 | **AutoCalibrationEngine** | slmsuite Fourier 校准/python-microscope | 自动化系统标定 | 坐标映射+迟滞补偿 |
| 45 | **DigitalTwinSimulator** | OOPAO 端到端仿真/lowfssim NASA JPL | 光学数字孪生仿真器 | 高保真闭环仿真验证 |

### 第十一批 (v16.0, 本轮新增 — 第四轮前沿调研)

| # | 模块 | 来源启发 | 功能 | 收益 |
|---|------|---------|------|------|
| 46 | **NeuralOperatorProxy** | NeuralOperator (FNO) / torchdiffeq | 神经算子快速光束传播代理模型 | 替代数值仿真，100x+ 加速 |
| 47 | **SAM2SpotSegmenter** | SAM 2 (Meta) / GrabCut | 像素级光斑分割+时序跟踪 | 超越边界框的精确分割 |
| 48 | **LearningMPCController** | leap-c / acados / CasADi | 学习增强型模型预测控制 | IL+RL+MPC 融合最优策略 |
| 49 | **DifferentiableRayTracer** | Optiland / LightPipes / Ray-Optics | 可微分光线追踪+系统优化 | 端到端光学系统参数优化 |
| 50 | **ContinuousStateEstimator** | torchdiffeq (Neural ODE) | 连续时间状态估计器 | 帧间漂移预测+异常检测 |

---

## 四、新增模块详细说明

### 模块 8: Zernike 像差分析器 (ZernikeAberrationAnalyzer)
- **算法**: Zernike 多项式最小二乘拟合 (Noll 1976 索引)
- **输出**: 15 项 Zernike 系数、波前 RMS、Strehl 比估计
- **诊断能力**: 离焦、像散、彗差、球差等像差识别
- **配置**: `zernike_enabled`, `zernike_max_order`, `zernike_alignment_threshold`

### 模块 9: 图像雅可比视觉伺服控制器 (ImageJacobianController)
- **算法**: Broyden 在线雅可比估计 + 阻尼最小二乘逆
- **核心**: 自动学习像素偏差 → 电机步数的映射矩阵
- **特性**: 在线学习 (越用越准)、阻尼正则化、自适应步长
- **配置**: `image_jacobian_enabled`, `image_jacobian_initial_step_per_px`, `image_jacobian_damping`

### 模块 10: YOLO 模型推理优化器 (ModelInferenceOptimizer)
- **功能**: 自动检测最优推理后端 (TensorRT > ONNX > PyTorch)
- **分析**: 延迟基准测试 (avg/P50/P95/P99)、吞吐量统计
- **建议**: 自动生成优化建议 (安装 TensorRT/ONNX 等)
- **配置**: `model_optimizer_enabled`, `model_optimizer_target_latency_ms`

### 模块 11: 主动学习数据采集器 (ActiveLearningCollector)
- **算法**: 不确定性采样 + 多样性采样 + 难度校准
- **采集策略**: 检测失败、低置信度、高偏差、低焦点评分
- **输出**: YOLO 格式标注文件 + JSON 元数据
- **配置**: `active_learning_enabled`, `active_learning_output_dir`, `active_learning_max_samples`

### 模块 12: 自适应焦平面搜索器 (AdaptiveFocusSearcher)
- **算法**: 粗搜索 (等间距扫描) + 精搜索 (黄金分割法)
- **特性**: 由粗到精、FWHM 估计、迟滞补偿
- **收益**: 替代固定步长 Z 搜索，更快找到最佳焦点
- **配置**: `focus_search_enabled`, `focus_search_coarse_range`, `focus_search_coarse_step`

---

## 五、实施优先级

| 优先级 | 模块 | 实施难度 | 预期收益 | 状态 |
|--------|------|----------|----------|------|
| P0 | 卡尔曼滤波光斑追踪器 | 中 | 高 | ✅ 已实现 |
| P0 | 安全保护与限位管理器 | 低 | 高 | ✅ 已实现 |
| P1 | 亚像素质心定位器 | 中 | 高 | ✅ 已实现 |
| P1 | 光斑质量评估器 | 中 | 中 | ✅ 已实现 |
| P1 | **图像雅可比控制器** | 中 | 高 | ✅ 已实现 |
| P1 | **Zernike 像差分析器** | 中 | 中 | ✅ 已实现 |
| P2 | 自适应增益调度控制器 | 高 | 中 | ✅ 已实现 |
| P2 | 运动轨迹记录与分析器 | 中 | 中 | ✅ 已实现 |
| P2 | **模型推理优化器** | 低 | 中 | ✅ 已实现 |
| P2 | **主动学习数据采集器** | 中 | 中 | ✅ 已实现 |
| P3 | **自适应焦平面搜索器** | 高 | 中 | ✅ 已实现 |
| P3 | 经典光斑检测器 | 中 | 中 | ✅ 已实现 |
| P2 | **推理后端加速器** | 中 | 高 | ✅ 本轮新增 |
| P2 | **智能异常检测器** | 中 | 中 | ✅ 本轮新增 |
| P3 | **RL 对准环境** | 高 | 高 | ✅ 本轮新增 |
| P1 | **相位恢复波前分析器** | 中 | 高 | ✅ v6.0 新增 |
| P1 | **PSF 质量估计器** | 低 | 高 | ✅ v6.0 新增 |
| P1 | **Zernike 模态控制器** | 高 | 高 | ✅ v6.0 新增 |
| P2 | **大气湍流仿真器** | 中 | 中 | ✅ v6.0 新增 |
| P1 | **智能精修控制器** | 中 | 高 | ✅ v6.0 新增 |
| P1 | **MPC 模型预测控制器** | 高 | 高 | ✅ v10.0 新增 |
| P1 | **可微光学优化器** | 高 | 高 | ✅ v10.0 新增 |
| P1 | **实时控制管线** | 中 | 高 | ✅ v10.0 新增 |
| P2 | **自动化系统标定** | 中 | 高 | ✅ v10.0 新增 |
| P2 | **数字孪生仿真器** | 高 | 高 | ✅ v10.0 新增 |
| P1 | **神经算子代理模型** | 中 | 高 | ✅ v16.0 新增 |
| P1 | **学习增强型 MPC** | 高 | 高 | ✅ v16.0 新增 |
| P1 | **可微分光线追踪器** | 中 | 高 | ✅ v16.0 新增 |
| P2 | **SAM2 光斑分割器** | 中 | 高 | ✅ v16.0 新增 |
| P2 | **连续时间状态估计器** | 中 | 高 | ✅ v16.0 新增 |

---

## 六、推荐技术路线

```
[数据层]  ActiveLearningCollector + Label Studio --> 最小标注成本训练 YOLO
    |
[检测层]  Ultralytics YOLO --> ModelInferenceOptimizer (TensorRT/ONNX) --> 亚毫秒级推理
    |         + SAM2SpotSegmenter (像素级分割 + 时序跟踪)
    |
[分析层]  SubPixelCentroid + SpotQualityAnalyzer + ZernikeAberrationAnalyzer --> 精密定位+像差诊断
    |         + PhaseRetrievalAnalyzer (无传感器波前估计) + PSFEstimator (实时像质评估)
    |
[跟踪层]  KalmanSpotTracker + ContinuousStateEstimator --> 离散+连续时间状态估计
    |
[控制层]  LearningMPCController (IL+RL+MPC 融合) + MPCController (多轴约束优化)
    |         + ImageJacobianController (自适应步长)
    |         + AdaptiveGainScheduler + ModalController + SmartRefinementController
    |         + DifferentiableOpticalOptimizer (梯度优化) + DifferentiableRayTracer (光线追踪优化)
    |
[管线层]  RealtimeControlPipeline (Sensor->Processor->Controller->Corrector)
    |         + DataPipelineOrchestrator (多步骤自动化)
    |
[执行层]  SafetyManager + TrajectoryRecorder --> 安全保护+数据记录
    |
[标定层]  AutoCalibrationEngine (坐标映射+迟滞补偿+响应分析)
    |
[搜索层]  AdaptiveFocusSearcher --> 智能焦平面定位
    |
[仿真层]  DigitalTwinSimulator (高保真闭环仿真) + AtmosphericTurbulenceSimulator
    |         + NeuralOperatorProxy (快速代理模型) + DifferentiableRayTracer (可微光线追踪)
```

---

## 七、参考文献

1. HCIPy - High-Contrast Imaging in Python: https://github.com/ehpor/hcipy
2. ViSP - Visual Servoing Platform: https://github.com/lagadic/visp
3. PyMeasure - Scientific Instrument Control: https://github.com/pymeasure/pymeasure
4. ARTIQ - Advanced Real-Time Infrastructure for Quantum: https://github.com/m-labs/artiq
5. python-control - Control Systems Library: https://github.com/python-control/python-control
6. Bluesky - Data Acquisition: https://github.com/bluesky/bluesky
7. BoxMOT - Multi-Object Tracker: https://github.com/mikel-brostrom/boxmot
8. ByteTrack - Multi-Object Tracking: https://github.com/ifzhang/ByteTrack
9. BoT-SORT - Robust Multi-Object Tracking: https://github.com/NirAharon/BoT-SORT
10. Stable-Baselines3 - RL Algorithms: https://github.com/DLR-RM/stable-baselines3
11. PPAL - Active Learning for Detection: https://github.com/ChenhongyiYang/PPAL
12. AL-MDN - Probabilistic Active Learning: https://github.com/NVlabs/AL-MDN
13. Ultralytics YOLO: https://github.com/ultralytics/ultralytics
14. Noll, R.J. (1976). "Zernike polynomials and atmospheric turbulence", JOSA
15. Chaumette & Hutchinson (2006). "Visual Servo Control", IEEE R&A
16. ISO 24157:2023 - Ophthalmic optics — Zernike representation
17. Gerchberg, R.W. & Saxton, W.O. (1972). "A practical algorithm for the determination of phase from image and diffraction plane pictures", Optik
18. Fienup, J.R. (1982). "Phase retrieval algorithms: a comparison", Applied Optics
19. Southwell, W.H. (1980). "Wave-front estimation from wave-front slope measurements", JOSA
20. Roddier, F. (1999). "Adaptive Optics in Astronomy", Cambridge University Press
21. Noll, R.J. (1976). "Zernike polynomials and atmospheric turbulence", JOSA
22. Fried, D.L. (1966). "Optical resolution through a randomly inhomogeneous medium", JOSA
23. Taylor, G.I. (1938). "The spectrum of turbulence", Proc. Royal Society
24. do-mpc - Nonlinear MPC Toolbox: https://github.com/do-mpc/do-mpc
25. prysm - Physical Optics: https://github.com/brandondube/prysm
26. pyRTC - Real-Time AO Control: https://github.com/jacotay7/pyRTC
27. OOPAO - Object Oriented Python AO: https://github.com/cheritier/OOPAO
28. slmsuite - SLM Control and Holography: https://github.com/holodyne/slmsuite
29. NSER-IBVS - Neural-Analytical Visual Servoing (ICCV 2025): https://github.com/SpaceTime-Vision-Robotics-Laboratory/nser-ibvs-drone
30. python-microscope - Microscope Control: https://github.com/python-microscope/microscope
31. lowfssim - NASA JPL LOWFS Model: https://github.com/nasa-jpl/lowfssim
32. DeepTrack 2.0 - Deep Learning Particle Tracking: https://github.com/DeepTrackAI/DeepTrack
33. optsim - Optical Propagation Simulation: https://github.com/cbasedlf/optsim
34. Diehl, M. et al. (2005). "A survey of numerical methods for nonlinear optimal control", IFAC Proceedings
35. NeuralOperator - Neural Operators: https://github.com/neuraloperator/neuraloperator
36. SAM 2 - Segment Anything Model 2: https://github.com/facebookresearch/segment-anything-2
37. leap-c - Learning-Enhanced Predictive Control: https://github.com/leap-c/leap-c
38. acados - Fast Nonlinear MPC: https://github.com/acados/acados
39. CasADi - Symbolic Optimization: https://github.com/casadi/casadi
40. Optiland - Differentiable Optical Design: https://github.com/optiland/optiland
41. torchdiffeq - Neural ODE Solver: https://github.com/rtqichen/torchdiffeq
42. LightPipes - Beam Propagation Simulation: https://github.com/opticspy/LightPipes
43. PYNQ - Python for Zynq FPGA: https://github.com/Xilinx/PYNQ
44. Rahimi, A. & Recht, B. (2007). "Random Features for Large-Scale Kernel Machines", NeurIPS
