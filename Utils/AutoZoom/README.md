# AutoZoom — 自动光学变焦闭环测量系统

自动光学变焦（AutoZoom）实验的自动化闭环测量系统，通过 Python GUI 集成控制信号发生器、照明继电器、屏幕视觉角度检测、光谱数据采集与自动补焦，实现完整的循环测量流程。



## 系统架构

```
AutoZoom/
├── config_conditional_second_detect.py   # 默认参数配置
├── measurement_autofocus_shg_closed_loop.py  # 主程序：含自动补焦 + SHG 闭环的测量 GUI
├── measurement_with_focus_roi_metrics.py     # 简化版：含聚焦 ROI 指标的测量 GUI
│
├── control/                        # 硬件控制层
│   ├── illumination_relay.py       # 串口继电器 → 照明光开关
│   ├── signal_generator_rigol.py   # Rigol DG4062 信号发生器 (PyVISA)
│   ├── labview_tcp_server.py       # TCP Server ↔ LabVIEW 通信
│   ├── labview_tcp_server_1.py     # TCP Server 变体
│   ├── labview_tcp_server_binary.py# TCP Server 二进制协议版
│   └── labview_tcp.py              # TCP 客户端
│
├── vision/                         # 视觉检测层
│   ├── screen_capture.py           # 固定区域屏幕截图 (pyautogui)
│   ├── newpro_1.py                 # YOLO 角度检测 (ultralytics)
│   └── newpro_wan.py               # YOLO 角度检测变体
│
├── logic/                          # 业务逻辑层
│   └── angle_detect.py             # ScreenAngleDetector (截图+角度检测封装)
│
└── outputs/                        # 输出目录
    └── captured_frames/            # 循环测量截图保存
```

## 核心工作流程

每轮测量包含 12 个步骤的闭环流程：

| 步骤 | 操作 | 说明 |
|------|------|------|
| **Step 01** | 关闭照明光 → 等待稳定 | 消除照明光对基线采集的干扰 |
| **Step 02** | 发送 TCP `MEASURE` 命令 | LabVIEW 采集基线光谱，返回原始数据 |
| **Step 03** | 截屏 + YOLO 角度检测 | 检测"测量前角度"(angle_before) |
| **Step 04** | 打开照明光 | 为后续视觉检测提供照明 |
| **Step 05** | 打开信号发生器 CH1 (脉冲光) | 脉冲时长可调，控制光学变焦量 |
| **Step 06** | 关闭照明光 | 准备光谱采集 |
| **Step 07** | 等待 LabVIEW 就绪 | 等待 LabVIEW 准备采集激发后光谱 |
| **Step 08** | 发送 TCP `MEASURE` 命令 | LabVIEW 采集激发后光谱 |
| **Step 09** | 截屏 + YOLO 角度检测 | 检测"测量后角度"(angle_after) |
| **Step 10** | 光谱数据处理 | 高点滤除 → 中值滤波 → 峰值拟合 → Δw 计算 |
| **Step 11** | 自适应调节脉冲时长 | 根据 Δangle 调整下一轮 CH1 ON 时间 |
| **Step 12** | 保存数据 | 每轮 CSV + Excel 汇总 |

## 关键特性

### 1. 角度自适应闭环调节
- 根据当前角度与上一轮角度的差值（Δangle），自动调整信号发生器 CH1 的脉冲时长
- 目标 Δangle 范围可配置（默认 1° ~ 4°）
- Δangle 偏小 → 增大脉冲时长；Δangle 偏大 → 减小脉冲时长

### 2. 条件二次角度检测
- 当 YOLO 检测角度落在特定范围（如 0°~20° 或 90°~110°）时，自动启用二次检测
- 避免角度边界附近因噪声产生错误读数

### 3. 自动补焦 (Autofocus)
- 基于多种清晰度指标（Tenengrad、Laplacian、Brenner、高频比等）加权计算 FocusScore
- 低于阈值时自动触发 Newport Picomotor Z 轴闭环寻焦
- 支持 SHG（二次谐波）信号强度作为补焦触发条件

### 4. 光谱数据处理
- 从 LabVIEW 经 TCP 接收原始光谱数据
- 高点滤除（剔除高于阈值的数据点）
- 中值滤波平滑
- 峰值拟合与 Δw（半高宽变化量）计算

## 硬件配置

| 设备 | 型号 | 通信方式 | 用途 |
|------|------|----------|------|
| 照明继电器 | - | 串口 (RS232, 9600bps) | 照明光开关控制 |
| 信号发生器 | Rigol DG4062 | USB (PyVISA) | 双通道脉冲信号输出 |
| 光谱仪 | - | LabVIEW → TCP | 光谱数据采集 |
| 运动控制器 | Newport 8742 Picomotor | USB (pylablib) | Z 轴自动对焦 |
| 屏幕 | - | pyautogui 截图 | 角度读数视觉检测 |

## 配置参数

所有默认参数集中在 [`config_conditional_second_detect.py`](config_conditional_second_detect.py) 中，包括：

- **测量参数**：最大循环次数、脉冲时长、角度目标范围
- **照明光**：串口号
- **信号发生器**：VISA 地址、通道电压/频率/占空比/延时
- **角度检测**：YOLO 模型路径、截图区域、二次检测范围
- **TCP 通信**：IP 地址、端口号
- **光谱处理**：滤波参数、波长横坐标来源
- **自动补焦**：Z 轴参数、触发阈值、权重系数

## 快速开始

### 环境要求

- Python 3.9+
- Windows（硬件驱动依赖）

### 安装依赖

```bash
pip install ultralytics pyautogui pyvisa pylablib pyserial numpy matplotlib pillow openpyxl opencv-python
```

### 运行

```bash
# 完整版（含自动补焦 + SHG 闭环）
python measurement_autofocus_shg_closed_loop.py

# 简化版（含聚焦 ROI 指标）
python measurement_with_focus_roi_metrics.py
```

### 使用步骤

1. 启动程序，在 GUI 中配置参数（或直接使用默认参数）
2. 点击 **连接设备** — 自动连接照明继电器、信号发生器、加载 YOLO 模型
3. 点击 **启动 TCP Server** — 等待 LabVIEW 连接就绪
4. （可选）点击 **建立聚焦参考** — 采集多次 ROI 指标建立基线
5. 点击 **运行完整循环测量** — 开始自动闭环测量
6. 实时观察 GUI 日志、角度变化曲线和峰值变化曲线
7. 测量完成后，数据自动保存至 `measurement_output/save/MM.DD/` 目录