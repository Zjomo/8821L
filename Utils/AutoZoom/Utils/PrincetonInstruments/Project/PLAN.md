# Princeton Instruments 光谱仪 Python 直接控制开发计划

## 1. 目标与交付物

### 1.1 总体目标
在 `e:\jupyter file\2_Optics\8821L\Utils\AutoZoom\Utils\PrincetonInstruments\Project` 目录下，基于 **PICam SDK** 和 **PyQt6/PySide6**，实现一个可直接控制 Princeton Instruments（PI）光谱仪/相机的测试 UI，并分析现有测量工作流与光谱仪的数据耦合关系，给出可替换其他光谱仪的接口改造方案。

### 1.2 交付物清单
| 编号 | 交付物 | 路径 | 说明 |
|------|--------|------|------|
| D1 | PI 光谱仪抽象接口层 | `pi_spectrometer/` | 封装 PICam SDK，提供统一光谱仪接口 |
| D2 | PICam ctypes 绑定 | `pi_spectrometer/picam_binding.py` | 基于 ctypes 加载 Picam.dll 的基础绑定 |
| D3 | PyQt 测试 UI | `pi_spectrometer/ui/main_window.py` | 设备连接、参数配置、采集、实时绘图 |
| D4 | 测试脚本 | `tests/test_picam_*` | 单元测试、集成测试、虚拟相机测试 |
| D5 | 接口迁移分析报告 | 本文档第 4 节 | 现有工作流的光谱仪耦合点分析 |
| D6 | 现有工作流最小改动补丁 | `patches/measurement_workflow_adapter.py` | 让现有 GUI 可选使用 PI 直接控制 |

---

## 2. 现状分析

### 2.1 当前光谱仪通信架构
当前 `0_measurement_workflow_real_virtual_same_detection_6.25.py` 通过 **LabVIEW TCP 中间层** 获取光谱数据：

```
0_measurement_workflow_real_virtual_same_detection_6.25.py
    ↓ 调用
control.labview_tcp_server.LabVIEWTCPServer
    ↓ TCP Socket (127.0.0.1:65432)
LabVIEW 前端
    ↓ Princeton Instruments LightField / PICam
PI-MAX4 / IsoPlane 光谱仪硬件
```

### 2.2 关键数据流字段
工作流内部对光谱数据的约定如下：

| 字段名 | 类型 | 来源 | 用途 |
|--------|------|------|------|
| `raw_values` | `List[float]` | LabVIEW 原始 y / CSV | 原始强度谱 |
| `threshold_values` | `List[float]` | Python 后处理 | 阈值过滤后强度 |
| `median_values` | `List[float]` | Python 后处理 | 中值滤波后强度 |
| `fit_values` | `List[float]` | LabVIEW / Python 兜底 | 拟合/显示曲线 |
| `x_axis_values` | `List[float]` | `516中心波长.xlsx` | 波长横坐标 |
| `raw_original_peak` | `float` | Python 计算 | 原始峰值 |
| `raw_filtered_peak` | `float` | Python 计算 | 阈值过滤后峰值 |
| `raw_median_peak` | `float` | Python 计算 | 中值滤波后峰值 |
| `fit_peak` | `float` | LabVIEW / Python 计算 | 拟合峰值 |
| `num_points` | `int` | 数据长度 | 校验有效点数 |
| `csv_path` | `str` | TCP 模块 | 本地保存路径 |

### 2.3 数据入口函数
- `MeasurementWorkflow.request_labview_spectrum(cycle_index)`：触发一次光谱采集
- `MeasurementWorkflow._parse_labview_result(result)`：解析 LabVIEW/TCP 返回
- `MeasurementWorkflow.load_x_axis_values_from_xlsx()`：加载波长横坐标
- `MeasurementWorkflow._remove_above_threshold()`、`_median_filter_1d()`：后处理
- `SpectrumAutofocusLoop.measure_spectrum_once()`：光谱补焦循环中调用光谱

### 2.4 配置参数
在 `config_angle_repair_fixed.py` 的 `DEFAULT_CONFIG` 中，与光谱仪直接相关的配置：

```python
'tcp_host': '127.0.0.1'
'tcp_port': 65432
'tcp_output_dir': 'labview_csv_output'
'tcp_command': 'MEASURE'
'raw_remove_above': 3000.0
'median_filter_window': 5
'x_axis_xlsx_path': '516中心波长.xlsx'
```

---

## 3. 新模块设计：PICam SDK 直接控制

### 3.1 目录结构
```
e:\jupyter file\2_Optics\8821L\Utils\AutoZoom\Utils\PrincetonInstruments\Project\
├── PLAN.md                              # 本计划文档
├── README.md                            # 使用说明
├── requirements.txt                     # Python 依赖
├── pi_spectrometer/                     # 核心包
│   ├── __init__.py
│   ├── core/                            # 抽象层
│   │   ├── __init__.py
│   │   ├── base.py                      # SpectrometerBackend 抽象基类
│   │   ├── types.py                     # 数据类型/枚举
│   │   └── exceptions.py                # 自定义异常
│   ├── picam/                           # PICam 后端
│   │   ├── __init__.py
│   │   ├── binding.py                   # ctypes 加载 Picam.dll
│   │   ├── constants.py                 # PICam 枚举/常量
│   │   ├── camera.py                    # PICamCamera 类
│   │   └── demo.py                      # 软件模拟相机（无需硬件）
│   ├── processing/                      # 数据处理
│   │   ├── __init__.py
│   │   ├── filters.py                   # 阈值/中值滤波
│   │   └── peaks.py                     # 峰值/拟合
│   └── ui/                              # PyQt 测试 UI
│       ├── __init__.py
│       ├── main_window.py               # 主窗口
│       ├── plot_widget.py               # 光谱实时绘图
│       └── controls.py                  # 参数面板
├── tests/                               # 测试
│   ├── __init__.py
│   ├── test_picam_binding.py
│   ├── test_picam_camera.py
│   ├── test_processing.py
│   ├── test_ui_smoke.py
│   └── conftest.py
└── scripts/                             # 独立脚本
    ├── picam_demo.py                    # 无硬件演示
    └── migrate_from_labview.py          # 迁移辅助示例
```

### 3.2 抽象接口层
定义 `SpectrometerBackend` 抽象基类，隔离硬件细节：

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

class SpectrometerBackend(ABC):
    @abstractmethod
    def connect(self) -> bool: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def set_exposure(self, seconds: float) -> None: ...

    @abstractmethod
    def get_exposure(self) -> float: ...

    @abstractmethod
    def set_sensor_temperature(self, celsius: float) -> None: ...

    @abstractmethod
    def get_sensor_temperature(self) -> Optional[float]: ...

    @abstractmethod
    def set_roi(self, x: int, width: int, xbin: int,
                y: int, height: int, ybin: int) -> None: ...

    @abstractmethod
    def acquire(self, num_frames: int = 1) -> np.ndarray: ...

    @abstractmethod
    def get_wavelength_axis(self) -> Optional[np.ndarray]: ...
```

### 3.3 PICam 后端实现要点
1. **加载 DLL**：通过 `ctypes.CDLL` 加载 `Picam.dll`，支持从 `PicamRoot` 环境变量定位
2. **枚举相机**：调用 `Picam_GetAvailableCameraIDs`
3. **连接相机**：`Picam_ConnectDemoCamera`（模拟）或 `Picam_OpenCamera`
4. **参数配置**：
   - `ExposureTime`
   - `SensorTemperatureSetPoint`
   - `ReadoutControlMode`
   - `AdcSpeed`、`AdcAnalogGain`
   - `ROI`（通过 `PicamRois` 结构体）
5. **采集**：`Picam_Acquire` 同步采集或 `Picam_StartAcquisition` + 事件回调异步采集
6. **门控支持（PI-MAX4）**：
   - `GatingMode`、`GateWidth`、`GateDelay`
   - `IntensifierStatus`、`IntensifierGain`

### 3.4 PyQt 测试 UI 功能
| 功能模块 | 说明 |
|----------|------|
| 设备发现 | 扫描并列出可用 PI 相机 |
| 连接/断开 | 连接选中设备 |
| 参数面板 | 曝光时间、制冷温度、ADC 速度、增益、ROI、门控参数 |
| 采集控制 | 单帧/连续采集、停止 |
| 实时绘图 | 使用 pyqtgraph 显示光谱曲线 |
| 数据处理 | 阈值过滤、中值滤波、峰值检测 |
| 文件保存 | SPE/CSV/PNG |
| 日志窗口 | 实时显示运行日志 |

---

## 4. 更换其他光谱仪的接口更新分析

若将现有 `0_measurement_workflow_real_virtual_same_detection_6.25.py` 中的光谱仪从 **LabVIEW TCP** 更换为 **其他光谱仪**（例如 PI 直接控制、Ocean Optics、Thorlabs CCS 等），需要更新以下接口：

### 4.1 必须更新的接口（数据入口层）

| 位置 | 当前实现 | 更新内容 |
|------|----------|----------|
| `control/labview_tcp_server.py` | TCP Server 等待 LabVIEW | 替换为目标光谱仪的 Python 驱动/后端 |
| `MeasurementWorkflow._create_labview_tcp_server()` | 创建 `LabVIEWTCPServer` | 创建新后端的实例（如 `PISpectrometerBackend`） |
| `MeasurementWorkflow.request_labview_spectrum()` | 调用 TCP `request_measure` | 调用新后端的 `acquire()` 并封装为统一结果字典 |

### 4.2 必须保持兼容的数据结构

新光谱仪后端返回的字典必须包含以下键，才能被 `_parse_labview_result()` 正确消费：

```python
{
    "ok": True,
    "index": int,
    "num_points": int,
    "raw_y": List[float],          # 原始强度谱，必须
    "fit_y": List[float],          # 拟合曲线，可选
    "csv_path": str,               # 本地保存路径，可选
    "wavelength": List[float],     # 波长横坐标，可选
    "fit_peak": float,             # 拟合峰值，可选
}
```

### 4.3 配置参数层更新

| 配置项 | 当前值 | 更新建议 |
|--------|--------|----------|
| `tcp_host` / `tcp_port` | LabVIEW 地址 | 替换为光谱仪连接参数（如 USB 索引、串口、GigE IP） |
| `tcp_command` | `MEASURE` | 删除或改为新后端的触发命令 |
| `x_axis_xlsx_path` | `516中心波长.xlsx` | 保留，横坐标映射逻辑通用 |
| `raw_remove_above` / `median_filter_window` | 后处理参数 | 保留，后处理逻辑与硬件无关 |

### 4.4 调用链梳理

现有工作流中依赖光谱仪的调用链：

```
run_full_measurement()
    → start_tcp_server() / wait_labview_ready()
    → request_labview_spectrum(cycle_index)
        → _request_labview_spectrum_compat()
        → _parse_labview_result()
        → load_x_axis_values_from_xlsx()
        → _remove_above_threshold()
        → _median_filter_1d()
    → start_save_cycle_result_async()

spectrum_autofocus_loop()
    → SpectrumAutofocusLoop.measure_spectrum_once()
        → workflow.request_labview_spectrum()
```

**最小改动策略**：
1. 新增一个 `SpectrometerBackend` 抽象层；
2. 实现对应硬件后端（如 `PISpectrometerBackend`）；
3. 在 `MeasurementWorkflow` 中用工厂模式创建后端；
4. 保持 `request_labview_spectrum()` 的返回结构不变；
5. 将新后端采集的数据封装成与当前 LabVIEW TCP 返回一致的字典。

### 4.5 不同光谱仪的特有关注点

| 光谱仪类型 | 特殊关注点 |
|------------|------------|
| PI PICam 直接控制 | 需要安装 PICam SDK；注意制冷、门控、ROI；32/64 位一致性 |
| PI LightField Python API | 依赖 LightField 软件运行；通过 COM/Automation 调用 |
| Ocean Optics | 通常使用 `seabreeze` 库；横坐标由设备校准系数生成 |
| Thorlabs CCS | 使用 `TLCCS` DLL；波长由设备内部校准生成 |
| 通用 SCPI 光谱仪 | 通过 VISA/串口发送 SCPI 命令；需自定义命令集 |

---

## 5. 执行计划

### 阶段 1：环境与依赖准备（1 天） ✅
- [x] 确认已安装 PICam SDK 或获取安装包
- [x] 创建 `requirements.txt`（PyQt6/PySide6、pyqtgraph、numpy、pytest）
- [x] 创建目录结构与空文件骨架
- [x] 验证 `ctypes` 能加载 `Picam.dll`（代码已就绪，真实硬件环境待验证）

### 阶段 2：PICam 绑定与抽象层（2 天） ✅
- [x] 实现 `picam_binding.py`：加载 DLL、错误处理、版本查询
- [x] 实现 `constants.py`：常用 PICam 枚举常量
- [x] 实现 `SpectrometerBackend` 抽象基类
- [x] 实现 `PICamCamera`：连接、参数设置、同步采集、ROI、温度
- [x] 实现 `DemoCamera`：软件模拟，支持无硬件开发测试

### 阶段 3：数据处理层（1 天） ✅
- [x] 迁移 `_remove_above_threshold`、`_median_filter_1d` 到 `filters.py`
- [x] 实现峰值检测与简单高斯拟合 `peaks.py`
- [x] 确保输出字段与现有 `_parse_labview_result` 兼容

### 阶段 4：PyQt 测试 UI（2 天） ✅
- [x] 实现主窗口布局与设备发现面板
- [x] 实现参数配置控件（曝光、温度、ROI、门控）
- [x] 集成 pyqtgraph 实时绘图
- [x] 实现采集控制与日志窗口
- [x] 实现 CSV 保存

### 阶段 5：测试（2 天） ✅
- [x] 编写 `test_core.py`：抽象层、ROI、结果类型、Mock 后端
- [x] 编写 `test_adapter.py`：适配器生命周期、工作流补丁
- [x] 编写 `test_processing.py`：滤波、峰值计算
- [x] 编写 `test_ui_smoke.py`：UI 导入/启动冒烟测试
- [x] 运行 pytest：`19 passed, 1 skipped`
- [ ] 在目标机器上连接真实 PI 设备进行集成测试（待真实硬件验证）

### 阶段 6：现有工作流适配（1 天） ✅
- [x] 实现 `patches/measurement_workflow_adapter.py`
- [x] 在 `MeasurementWorkflow` 中增加后端工厂（不破坏现有 LabVIEW TCP 逻辑）
- [x] 通过配置切换 `labview_tcp` / `picam_direct` 模式
- [x] 验证 `SpectrumAutofocusLoop` 可直接复用新后端（通过统一 `request_measure` 接口）

### 阶段 7：文档与验收（1 天） ✅
- [x] 编写 `README.md`（安装、运行、测试）
- [x] 更新本 `PLAN.md` 中的进度
- [x] 运行 demo 脚本验证闭环：`python scripts/picam_demo.py --backend mock`

---

## 6. 风险与回滚策略

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| PICam SDK 未安装或版本不匹配 | 无法加载 DLL | 优先实现 DemoCamera；在文档中明确 SDK 版本要求 |
| 64/32 位 Python 与 DLL 不匹配 | ctypes 加载失败 | 要求 64 位 Python + 64 位 PICam |
| 真实 PI 设备不可用 | 集成测试受阻 | 使用软件模拟相机完成 90% 开发；硬件测试作为最后一步 |
| PyQt 与现有 tkinter GUI 冲突 | 同进程内两套 UI 框架可能冲突 | 新 UI 作为独立进程/工具运行；不直接嵌入现有 tkinter |
| 现有工作流依赖 LabVIEW 状态机 | 直接替换可能改变时序 | 保持 `request_labview_spectrum` 返回结构不变，仅内部实现替换 |

---

## 7. 验收标准

- [ ] `pytest tests/` 全部通过（含无硬件模拟测试）
- [ ] PyQt UI 能在无硬件模式下启动并显示模拟光谱
- [ ] 在真实 PI 设备上能完成：连接、设参、采集、绘图、保存
- [ ] `measurement_workflow_adapter.py` 能让现有工作流在 `picam_direct` 模式下跑通一次完整采集
- [ ] 文档完整：安装、运行、接口说明、故障排查

---

*计划创建时间：2026-07-15*
*负责人：待分配*
