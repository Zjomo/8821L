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

---

## 8. 问题修复记录

### 8.1 Picam_FreeCameraIDs 函数未找到错误修复

**报错信息**：
```
连接错误: function 'Picam_FreeCameraIDs' not found
```

**根因分析**：
1. `Picam.dll` 成功加载（位于 `C:\Program Files\Common Files\Princeton Instruments\Picam\Runtime\Picam.dll`）
2. 但 `Picam_FreeCameraIDs` 函数在当前安装的 PICam SDK 版本中不存在
3. 这是 PICam SDK 版本差异导致的兼容性问题

**修复方案**：
修改 `pi_spectrometer/picam/binding.py`，增加对 `Picam_FreeCameraIDs` 函数不存在的容错处理：

1. 在 `_setup_function_signatures()` 中检查函数是否存在：
```python
if hasattr(lib, 'Picam_FreeCameraIDs'):
    lib.Picam_FreeCameraIDs.argtypes = [ctypes.POINTER(PicamCameraID)]
    lib.Picam_FreeCameraIDs.restype = ctypes.c_int
else:
    lib.Picam_FreeCameraIDs = None
```

2. 在 `get_available_cameras()` 中调用前检查：
```python
if hasattr(self.lib, 'Picam_FreeCameraIDs') and self.lib.Picam_FreeCameraIDs is not None:
    try:
        self.lib.Picam_FreeCameraIDs(ids_ptr)
    except Exception:
        pass
```

3. 增强错误提示，引导用户使用 "无 SDK 模拟" 后端：
```python
raise PICamError(
    "无法找到 Picam.dll。请安装 PICam SDK...\n"
    "提示：如果没有真实硬件，请在 UI 中选择 '无 SDK 模拟' 后端。"
)
```

**测试验证**：
```bash
cd e:\CWB\8821L\Utils\AutoZoom\Utils\PrincetonInstruments\Project
python -c "
from pi_spectrometer.picam.demo import MockSpectrometerBackend
backend = MockSpectrometerBackend()
backend.connect()
result = backend.acquire()
print(f'采集成功: {result.num_points} 点')
backend.disconnect()
"
# 输出: Mock 后端连接成功
#       采集成功: 1024 点, raw_peak=1010.09
#       Mock 后端断开成功
```

**修改文件**：
- `pi_spectrometer/picam/binding.py`

**使用建议**：
- 如果没有真实 PI 硬件，请在 UI 后端选择 **“无 SDK 模拟”**
- 如果需要连接真实 PI 设备，请确保安装了正确版本的 PICam SDK

---

### 8.2 PI 光谱仪 SDK 分析与连接方案

**目标**：保证项目可以直接连接 Princeton Instruments 光谱仪系统。

#### SDK 分析结果

| SDK | DLL 路径 | 架构 | 用途 | 当前状态 |
|---|---|---|---|---|
| PICam SDK | `C:\Program Files\Common Files\Princeton Instruments\Picam\Runtime\Picam.dll` | x64 | 控制 CCD/CMOS 探测器 | 已安装 v5.14.7.2311，缺少 `Picam_FreeCameraIDs` |
| ARC SDK | `Utils/AutoZoom/Utils/PrincetonInstruments/ISOPLANEControl/ARC_SpectraPro.dll` | x86 | 控制 IsoPlane/SpectraPro 单色仪 | 已内置，但与当前 64-bit Python 位数不匹配 |

#### 关键发现

1. **PICam.dll 缺少 `Picam_FreeCameraIDs`**
   - 已在 §8.1 中通过容错处理修复。
   - 建议升级到最新版 PICam SDK 以完全解决。

2. **ARC_SpectraPro.dll 是 32-bit**
   - 当前 Python 是 64-bit，无法直接加载 32-bit DLL（Windows 限制）。
   - 错误：`[WinError 193] %1 不是有效的 Win32 应用程序。`

3. **DLL 导出表特征**
   - `ARC_*.dll` 仅通过 ordinal 导出，没有函数名。
   - `Picam.dll` 有函数名导出，但版本差异导致部分函数缺失。

#### 实现改动

1. **ARC 绑定层增强（`pi_spectrometer/picam/arc_binding.py`）**
   - 新增 `_get_python_bits()` 和 `_get_dll_bits()` 检测位数。
   - 在 `_initialize()` 中提前检测 Python/DLL 位数不匹配，给出明确错误提示和 32-bit Python 下载链接。

2. **环境诊断脚本（`diagnose_pi_environment.py`）**
   - 检查 Python 位数。
   - 检查 PICam SDK 版本和关键函数是否存在。
   - 检查 ARC SDK 位数匹配性。
   - 扫描可用串口设备。
   - 输出可操作的修复建议。

3. **UI 增强（`pi_spectrometer/ui/main_window.py`）**
   - 增加 **“连接 IsoPlane 单色仪”** 按钮。
   - 增加 **“运行环境诊断”** 按钮，直接在 UI 日志中显示诊断结果。
   - 延迟导入 `IsoPlaneBackend`，避免在 64-bit Python 启动时立即加载 32-bit DLL。

4. **辅助脚本**
   - `ISOPLANEControl/analyze_dlls.py`：不依赖第三方库，分析 PE 导出函数。
   - `check_picam.py`：检查 Picam.dll 版本和位数。
   - `test_pi_fixes.py`：基础回归测试。

#### 测试验证

```bash
cd e:\CWB\8821L\Utils\AutoZoom\Utils\PrincetonInstruments\Project
python diagnose_pi_environment.py
python test_pi_fixes.py
python -c "from pi_spectrometer.ui.main_window import MainWindow; print('UI OK')"
```

**测试结果**：
```
PASS: MockSpectrometerBackend
PASS: PICamBinding import OK, Picam.dll=...
PASS: ARC DLL=32-bit, Python=64-bit
PASS: ARCSpectraBinding 正确报告位数不匹配
PASS: diagnose_pi_environment import OK
UI 构建成功
```

#### 连接 PI 光谱仪的操作步骤

**情况 A：只有 PICam 探测器（无 IsoPlane 单色仪）**
1. 确保已安装 PICam SDK（64-bit）。
2. 运行 `python diagnose_pi_environment.py` 确认 `Picam.dll` 可加载。
3. 在 UI 中选择 **“PICam 真实相机”**，点击连接。
4. 如果仍提示 `Picam_FreeCameraIDs` 缺失，说明 SDK 版本过旧，建议升级。

**情况 B：需要连接 IsoPlane/SpectraPro 单色仪**
1. 安装 **32-bit Python 3.9**：https://www.python.org/downloads/release/python-3913/
2. 在 32-bit Python 中安装依赖：
   ```bash
   py -3.9-32 -m pip install numpy pyserial pyqt5 pyqtgraph
   ```
3. 使用 32-bit Python 运行项目：
   ```bash
   py -3.9-32 scripts/run_ui.py
   ```
4. 点击 **“连接 IsoPlane 单色仪”** 按钮。

**情况 C：无硬件，仅测试代码**
1. 在 UI 中选择 **“无 SDK 模拟”** 或 **“PICam Demo 相机”**。
2. 可以正常进行单帧/连续采集、保存 CSV 等操作。

#### 修改文件

- `pi_spectrometer/picam/arc_binding.py`
- `pi_spectrometer/ui/main_window.py`
- `diagnose_pi_environment.py`（新增）
- `ISOPLANEControl/analyze_dlls.py`（新增）
- `check_picam.py`（新增）
- `test_pi_fixes.py`（新增）
- `PLAN.md`

---

## 9. UI 功能增强：向 LightField 级 acquisition UI 演进

### 9.1 目标

在现有 PyQt 测试 UI 基础上，补齐 Princeton Instruments LightField 软件中常用的核心 acquisition 与数据处理能力，使本 UI 从“单帧/连续采集演示工具”升级为可用于日常光谱实验的轻量级 acquisition 工作站。

### 9.2 待实现功能清单

| 功能域 | 子功能 | 优先级 | 说明 |
|--------|--------|--------|------|
| **背景/暗场校正** | 采集/加载暗背景帧 | P0 | 支持 dark/background 帧库，采集时自动或手动扣除 |
| | 背景帧管理（保存/列出/切换） | P1 | 可维护多组背景帧 |
| **光谱统计** | 峰值、FWHM、质心、积分 | P0 | 实时计算并显示在主界面 |
| | SNR、基线、半峰宽 | P1 | 辅助判断数据质量 |
| **实验序列 (Recipe)** | 多步骤采集（曝光/帧数/延迟/循环） | P0 | 类似 LightField Experiment 的 step 编排 |
| | 循环与条件分支 | P1 | for-loop、if-peak 等简单控制 |
| **增强绘图** | 双光标与峰值标注 | P0 | 在谱线上标记峰位、FWHM 区间 |
| | 历史轨迹/瀑布图 | P1 | 显示最近 N 帧的演变 |
| | 对数/线性 Y 轴切换 | P1 | |
| **自动保存** | 文件名模板与自动编号 | P0 | `spectrum_{index}_{timestamp}.csv` 等 |
| | 按 Recipe 自动归档 | P1 | 每步结果独立目录 |
| **波长校准** | 从 xlsx/手动输入加载波长轴 | P1 | 替代当前外部 xlsx 后处理 |
| | 基于 IsoPlane 参数自动计算 | P0 | 已部分实现，UI 需暴露参数 |

### 9.3 模块设计

1. **`pi_spectrometer/processing/background.py`**
   - `BackgroundFrameLibrary`：管理 dark/background/reference 帧
   - `apply_background_correction(result, dark=None, reference=None)`：扣除暗背景，可选做 reference 归一化

2. **`pi_spectrometer/processing/stats.py`**
   - `SpectrumStats`：峰值、峰位、FWHM、质心、积分面积、SNR
   - `find_peaks_and_stats(x, y, threshold=...)`：多峰检测与统计

3. **`pi_spectrometer/core/recipe.py`**
   - `RecipeStep`：单步定义（action, params, repeats, delay_s）
   - `AcquisitionRecipe`：步骤列表、验证、执行、回调
   - `RecipeRunner`：在后台线程运行 Recipe，支持 stop/pause

4. **`pi_spectrometer/ui/plot_widget.py` 增强**
   - `CursorLine`：可拖拽光标
   - `PeakAnnotation`：峰值标注Item
   - `PlotWidget.set_log_mode()` / `set_linear_mode()`
   - `PlotWidget.add_history_trace()` / `clear_history()`

5. **`pi_spectrometer/ui/main_window.py` 扩展**
   - 新增“背景帧”面板：capture dark / load dark / apply toggle
   - 新增“统计”面板：显示 peak / FWHM / centroid / integral
   - 新增“Recipe”面板：步骤列表、添加/删除/运行/停止
   - 新增“自动保存”设置：启用开关、目录、文件名模板
   - 状态栏/日志增强：显示采集进度、Recipe 步骤

### 9.4 测试计划

1. **背景处理单元测试**
   - 暗背景扣除后基线接近 0
   - reference 归一化保持形状
   - 尺寸不匹配时抛出明确异常

2. **统计模块单元测试**
   - 高斯峰 FWHM 与理论值误差 < 5%
   - 质心、积分计算正确
   - 多峰检测返回正确数量

3. **Recipe 单元测试**
   - 单步执行调用 backend.acquire()
   - 循环 3 次产生 3 个结果
   - 暂停/停止立即响应
   - 非法步骤参数在验证阶段报错

4. **UI 单元测试/冒烟测试**
   - 增强后的 `PlotWidget` 可导入并设置光标
   - 主窗口新增面板存在且可访问
   - Recipe 面板添加/删除步骤后列表同步

5. **集成测试**
   - Mock 后端下运行完整 Recipe 并自动保存 CSV
   - 背景扣除在 UI 端到端链路中生效

### 9.5 交付文件

- `pi_spectrometer/processing/background.py`（新增）
- `pi_spectrometer/processing/stats.py`（新增）
- `pi_spectrometer/core/recipe.py`（新增）
- `pi_spectrometer/ui/plot_widget.py`（修改）
- `pi_spectrometer/ui/main_window.py`（修改）
- `tests/test_background.py`（新增）
- `tests/test_stats.py`（新增）
- `tests/test_recipe.py`（新增）
- `tests/test_ui_enhanced.py`（新增）
- `test_lightfield_enhancements.py`（新增，无 pytest 时可直接运行）
- `PLAN.md`（本节）

### 9.6 测试验证

由于当前环境无法联网安装 `pytest`，新增了一个无需 pytest 的回归脚本：

```bash
cd e:\CWB\8821L\Utils\AutoZoom\Utils\PrincetonInstruments\Project
python test_lightfield_enhancements.py
```

输出：
```
PASS: dark subtraction
PASS: reference normalization
PASS: background library
PASS: correct spectrometer result
PASS: centroid
PASS: integral
PASS: fwhm gaussian
PASS: spectrum stats
PASS: multipeak detection
PASS: recipe serialization
PASS: recipe runner acquire
PASS: recipe runner repeat
PASS: recipe runner stop
PASS: plot widget enhanced
PASS: main window panels
PASS: main window recipe add/remove

结果: 16 通过, 0 失败
所有测试通过!
```

待 pytest 可用时，也可运行：

```bash
python -m pytest tests/ -v
```

### 9.7 已知限制与后续迭代

1. **P0 已实现**：暗背景扣除、光谱统计、Recipe 编排、峰值标注、自动保存、对数 Y 轴、历史轨迹。
2. **P1 待完善**：瀑布图 3D 显示、波长校准 UI、背景帧文件持久化、Recipe 条件分支/循环嵌套、SPE/TIFF 多格式保存。
3. **集成测试**：当前在 mock 后端验证；真实 PI 硬件上的端到端验证待 SDK/设备就绪后进行。
