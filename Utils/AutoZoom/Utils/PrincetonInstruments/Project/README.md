# Princeton Instruments 光谱仪 Python 直接控制

基于 PICam SDK 和 PyQt6 的 PI 光谱仪测试 UI 与控制库，支持跳过 LabVIEW TCP 中间层直接采集光谱。

## 目录结构

```
Project/
├── pi_spectrometer/         # 核心包
│   ├── core/                # 抽象基类、数据类型、异常
│   ├── picam/               # PICam SDK 绑定与相机后端
│   ├── processing/          # 阈值/中值滤波、峰值拟合
│   └── ui/                  # PyQt6 测试 UI
├── patches/                 # 现有工作流适配器
├── scripts/                 # 演示脚本
├── tests/                   # pytest 测试
├── PLAN.md                  # 开发计划
├── README.md                # 本文件
└── requirements.txt         # 依赖
```

## 安装

```powershell
cd "e:\jupyter file\2_Optics\8821L\Utils\AutoZoom\Utils\PrincetonInstruments\Project"
pip install -r requirements.txt
```

> 真实硬件控制需要安装 Princeton Instruments PICam SDK（64 位），并确保 `Picam.dll` 可被找到。

> **Qt 绑定说明**：UI 代码同时兼容 PyQt6 和 PyQt5。Anaconda 环境若 PyQt6 出现 `DLL load failed`，请改用 PyQt5：
> ```powershell
> pip uninstall PyQt6 PyQt6-Qt6 PyQt6_sip -y
> pip install PyQt5 pyqtgraph
> ```

## 快速开始

### 1. 命令行演示（无需硬件）

```powershell
python scripts/picam_demo.py --backend mock --count 3 --exposure 0.05
```

后端选项：
- `mock`：完全脱离 PICam DLL 的模拟
- `demo`：PICam 软件模拟相机（需要 SDK）
- `picam`：真实 PI 相机（需要 SDK + 硬件）

### 2. 启动 PyQt 测试 UI

```powershell
python scripts/run_ui.py
```

UI 功能：
- 后端选择（mock / demo / picam）
- 曝光、温度、ROI 参数配置
- 单帧 / 连续采集
- 实时光谱曲线绘制，支持光标、峰值/FWHM 标注、历史轨迹叠加、对数 Y 轴
- 暗背景/背景扣除与帧库管理
- 实时光谱统计：峰位、峰强、FWHM、质心、积分、SNR、多峰检测
- 实验序列（Recipe）：多步骤采集编排、循环、暂停/停止
- 自动保存：文件名模板与自动编号
- CSV 保存

### 3. 在现有工作流中使用

在 `0_measurement_workflow_real_virtual_same_detection_6.25.py` 中：

```python
from pi_spectrometer.patches.measurement_workflow_adapter import patch_measurement_workflow

# 在 MeasurementWorkflow.__init__ 末尾调用
patch_measurement_workflow(self)
```

并在配置中新增：

```python
DEFAULT_CONFIG = {
    ...,
    "spectrometer_backend": "picam",        # labview_tcp / picam / picam_demo / mock
    "picam_dll_path": None,
    "picam_camera_index": 0,
    "picam_exposure": 0.1,
    "picam_temperature": -25.0,
}
```

工作流将根据 `spectrometer_backend` 自动选择 LabVIEW TCP 或 PI 直接控制。

## 运行测试

```powershell
python -m pytest tests/ -v
```

若 pytest 未安装，可运行新增的独立回归脚本：

```powershell
python test_lightfield_enhancements.py
```

当前状态：核心测试 16/16 通过（无 pytest 独立脚本）；完整 pytest 套件待环境就绪后运行。

## 更换其他光谱仪的接口说明

若需将工作流迁移到其他光谱仪（Ocean Optics、Thorlabs CCS 等），只需：

1. 实现 `SpectrometerBackend` 抽象基类；
2. 返回包含 `raw_y` 的 `SpectrometerResult`；
3. 在配置中选择新后端。

上层的数据解析、阈值过滤、中值滤波、峰值计算、保存逻辑均无需改动。

## 注意事项

- 64 位 Python + 64 位 PICam SDK 才能加载 `Picam.dll`。
- PI-MAX4 门控操作涉及高压，请遵循硬件安全手册。
- 首次使用真实设备前，建议先用 `mock` 或 `demo` 后端验证流程。

## 作者

基于 PLAN.md 自动化开发完成。
