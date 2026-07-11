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
   - 可在“循环参数”中勾选 **启用被动补焦**，此时“循环轮数”会被禁用
     - **连续达标次数**：FocusScore 连续多少轮大于触发阈值后自动停止
     - **最大连续补焦次数**：单轮分数未达标时，连续补焦的最大尝试次数/安全上限
5. 点击 **建立参考** 建立 FocusScore 基线
6. 点击 **启动闭环** 开始自动聚焦
   - 主动模式：按设置的循环轮数运行，每轮检测一次，低于阈值触发一次补焦后进入等待
   - 被动模式：忽略循环轮数，按间隔持续运行；连续达标指定轮数后自动停止，也可随时点击 **停止** 中断

## 依赖

```bash
pip install numpy opencv-python PySide6
# 或 Windows 7 兼容
pip install numpy opencv-python PySide2
```

如需真实 Z 轴控制，还需安装 `pylablib`。
