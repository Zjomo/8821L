# autofocus_qt_ui — AutoZoom 自动聚焦 PyQt 界面

基于 PyQt6/PySide2 的 AutoZoom/Focus 自动聚焦图形界面，与命令行主程序 `Focus/main.py` 功能对齐。

## 位置

本模块位于 AutoZoom 项目目录内：

```
Utils/AutoZoom/
├── Focus/                # 核心自动对焦模块
├── autofocus_qt_ui/      # PyQt UI 界面（新增）
│   ├── app.py
│   ├── config.py
│   ├── qt_compat.py
│   ├── widgets.py
│   ├── worker.py
│   └── README.md
```

## 功能

- **采集源选择**：模拟模式（无硬件）、屏幕区域、窗口 ROI、USB 相机
- **实时预览**：支持 ROI 鼠标框选
- **14 项聚焦指标**：整图 + ROI 双列显示
- **FocusScore_ratio 曲线**：实时自绘趋势图 + 触发/目标阈值线
- **参考基线建立**：一键建立聚焦参考
- **自动补焦闭环**：真实 Z 轴 / 模拟模式均可运行
- **被动补焦模式**：在“循环参数”中开启后，忽略循环轮数并按间隔运行；FocusScore 连续达标指定轮数后自动停止
- **运行日志**：实时显示后台工作线程日志

## 离线数据集检测（offline_dataset_detection.py）

位于 `Utils/AutoZoom/offline_dataset_detection.py`，可对已有视频或图片文件夹做离线 FocusScore 评分与标注。

### 功能

- **输入**：视频文件（mp4/avi/mov 等）或包含图片的文件夹
- **视频处理**：按每 3 秒一帧提取图片（间隔可调）
- **基准图选择**：通过 `reference_path` 手动指定；未指定则默认使用第一张图片
- **FocusScore 计算**：以基准图建立参考，对后续图片计算 `FocusScore_ratio`
- **图像标注**：将得分标注在每张图片的**正上方**
- **自动过滤**：`FocusScore_ratio == 0` 的图片会被删除，不进入输出文件夹
- **输出**：新建文件夹，仅包含标注后的非零分图片

### 使用方式

```python
from offline_dataset_detection import OfflineDatasetDetector
from Focus.config import AutofocusConfig

cfg = AutofocusConfig(focus_roi=(0, 0, 300, 300))
detector = OfflineDatasetDetector(cfg)
output_dir, results = detector.run(
    input_path=r"path/to/video.mp4",      # 或图片文件夹路径
    output_dir=r"path/to/output",         # 可选，默认自动生成
    reference_path=None,                  # 可选，手动指定基准图
    interval_seconds=3.0,                 # 视频帧提取间隔
)
```

命令行：

```bash
cd Utils/AutoZoom
python offline_dataset_detection.py path/to/video.mp4 -o path/to/output -r path/to/ref.jpg --roi 0,0,300,300
```

## 运行方式

```bash
# 方式 1：进入 AutoZoom 目录后运行
cd Utils/AutoZoom
python -m autofocus_qt_ui

# 方式 2：从项目根目录指定模块路径
python -m Utils.AutoZoom.autofocus_qt_ui
```

## 使用流程

1. 选择运行模式（建议先用“模拟模式”测试）
2. 点击 **加载预览**，确认图像正常
3. 在预览图中框选 ROI，点击 **应用 ROI**
4. 设置补焦参数与循环参数
   - 默认触发阈值 **0.95**，目标阈值默认 **0.95**
   - **使用绝对值区间（以 1.0 对称）**：默认勾选。勾选后允许区间为
     `[触发阈值, 2 - 触发阈值]`，例如阈值 0.95 时允许区间为 **[0.95, 1.05]**，
     FocusScore_ratio 低于 0.95 **或** 高于 1.05 都会触发补焦；取消勾选则仅
     在分数低于触发阈值时触发。
   - **FocusScore检测**：点击后按钮变为“退出FocusScore检测”。开启后，当触发补焦条件时
     仅保存当前图像（`cycles/cycle_xxxx/trigger_detection.jpg`）和分数到文本文件，
     **不执行 Z 轴闭环搜索**，避免硬件操作或设备报错。
   - **离线数据集检测**：在左侧面板底部填写输入路径、输出目录、基准图（可选）和抽帧间隔，
     点击 **运行离线检测** 即可对视频或图片文件夹进行离线 FocusScore 评分与标注。
     ROI 沿用上方“ROI 设置”中的值。
5. 可在“循环参数”中勾选 **启用被动补焦**，此时“循环轮数”会被禁用
   - **连续达标次数**：FocusScore 连续多少轮在允许区间内后自动停止
   - **关闭达标阈值**：点击后该按钮变为“开启达标阈值”，被动补焦不再因连续达标而自动停止，只有手动点击 **停止** 才会结束循环
   - **最大连续补焦次数**：单轮分数未达标时，连续补焦的最大尝试次数/安全上限
6. 点击 **建立参考** 建立 FocusScore 基线
7. 点击 **启动闭环** 开始自动聚焦
   - 主动模式：按设置的循环轮数运行，每轮检测一次，超出允许区间触发一次补焦后进入等待
   - 被动模式：忽略循环轮数，按间隔持续运行；连续达标指定轮数后自动停止，也可随时点击 **停止** 中断

## 依赖

```bash
pip install numpy opencv-python PySide6
# 或 Windows 7 兼容
pip install numpy opencv-python PySide2
```

如需真实 Z 轴控制，还需安装 `pylablib`。
