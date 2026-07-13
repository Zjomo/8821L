# 硬件配置与 ROI 同步更新计划

## 1. 摘要

基于参考源码 `Utils/AutoZoom/0_measurement_workflow_real_virtual_same_detection_6.25.py`，完成以下需求：

1. 照明光控制串口：从 `COM17` 更新为 `COM20`；
2. 默认硬件模式：从 `"virtual"` 改为 `"real"`；
3. YOLO-OBB 长边角度检测默认模型路径：改为 `.\vision\best_wan12.2.pt`；
4. 截图区域与“光谱补焦循环”中手动选择的 ROI 区域同步更新。

用户确认范围：仅修改主程序与相关配置；不改动 `Utils/AutoZoom/` 下其他仍含 `COM17` 的脚本；不新建 README。

## 2. 当前状态分析

通过直接读取关键文件，确认主代码已具备以下改动：

| 文件 | 当前状态 |
|------|---------|
| `Utils/AutoZoom/config_angle_repair_fixed.py` | `light_port='COM20'`、`hardware_mode='real'`、`angle_model_path=r'.\vision\best_wan12.2.pt'` 已设置 |
| `Utils/AutoZoom/Utils/config_angle_repair_fixed.py` | 同上，已同步 |
| `Utils/AutoZoom/0_measurement_workflow_real_virtual_same_detection_6.25.py` | `MeasurementConfig` fallback 已更新为 `COM20`、`real`、`.\vision\best_wan12.2.pt`；GUI 默认值同步；`select_saf_roi` 已实现 ROI→截图区域同步 |
| `Utils/AutoZoom/spectrum_autofocus_loop.py` | 已新增 `roi_to_screen_capture_area` 静态方法 |
| `Utils/AutoZoom/test_measurement_defaults.py` | 已覆盖主 config 的默认值校验 |
| `Utils/AutoZoom/test_spectrum_autofocus_loop.py` | 已覆盖 ROI 坐标转换与源码集成检查 |

剩余工作：
- `test_measurement_defaults.py` 未覆盖 `Utils/AutoZoom/Utils/config_angle_repair_fixed.py` 这份重复配置；
- 需要运行测试确认无回归；
- 需要确认模型文件真实存在。

## 3. 拟议改动

### 3.1 验证并补全配置同步测试

**文件**: `Utils/AutoZoom/test_measurement_defaults.py`

**改动**: 新增对 `Utils/AutoZoom/Utils/config_angle_repair_fixed.py` 的默认值校验，确保两份 config 一致。

**原因**: 项目内存中存在两份同名配置，已同步修改，但测试只验证了主目录版本，存在遗漏。

**实现方式**: 在 `test_measurement_defaults.py` 中动态导入 `Utils/config_angle_repair_fixed.py`，并复用同一组断言检查 `light_port`、`hardware_mode`、`angle_model_path`。

### 3.2 运行测试并修复回归

**命令**（在 `Utils/AutoZoom` 目录下执行）：

```powershell
python test_measurement_defaults.py
python test_spectrum_autofocus_loop.py
python test_passive_mode.py
```

**预期结果**: 所有测试通过；若失败，根据错误信息修复对应源码或测试。

### 3.3 验证模型文件存在

**路径**: `Utils/AutoZoom/vision/best_wan12.2.pt`

**方式**: 测试 `test_default_angle_model_path` 中已包含存在性检查，运行后会输出 `INFO` 或 `WARN`。若文件缺失，仅提示用户，不阻塞通过。

## 4. 假设与决策

- **范围限定**: 仅修改主程序 `0_measurement_workflow_real_virtual_same_detection_6.25.py` 与两份 `config_angle_repair_fixed.py`；不碰 `measurement_with_focus_roi_metrics.py`、`config_conditional_second_detect.py`、`measurement_autofocus_shg_closed_loop.py` 等其他仍含 `COM17` 的脚本，也不修改 `段师姐_光谱检测_源代码/` 下的历史备份。
- **不新建 README**: 按用户选择，仅通过测试与代码完成交付，不额外撰写文档。
- **重复配置**: 两份 `config_angle_repair_fixed.py` 内容完全相同，新增测试确保未来不会漂移。

## 5. 验证步骤

1. 读取 `Utils/AutoZoom/config_angle_repair_fixed.py` 与 `Utils/AutoZoom/Utils/config_angle_repair_fixed.py`，确认 `COM20`、`real`、`.\vision\best_wan12.2.pt` 正确。
2. 读取 `Utils/AutoZoom/0_measurement_workflow_real_virtual_same_detection_6.25.py` 的 `MeasurementConfig` 与 `build_config_from_ui` 相关 fallback，确认已同步。
3. 确认 `spectrum_autofocus_loop.py` 中 `roi_to_screen_capture_area` 方法存在且坐标转换正确。
4. 运行 `test_measurement_defaults.py`，确保 5 项检查全部通过（含新增的 Utils 目录 config 检查）。
5. 运行 `test_spectrum_autofocus_loop.py`，确保 11 项测试全部通过。
6. 运行 `test_passive_mode.py`，确保被动补焦相关测试无回归。
7. 检查测试输出中模型文件存在性提示，确认 `vision/best_wan12.2.pt` 已就位。
