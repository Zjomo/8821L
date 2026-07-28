# 7_25 移除“单次光谱采集”自动保存光谱数据复选框实施计划

## 1. 目标

在 `0_measurement_workflow_real_virtual_same_detection_7_25.py` 的“6. 光谱仪通信”模块中，移除“单次光谱采集”区域的“自动保存光谱数据”复选框，并清理所有相关的变量、配置字段和自动保存逻辑。

## 2. 涉及文件

| 文件 | 变更内容 |
|------|----------|
| `0_measurement_workflow_real_virtual_same_detection_7_25.py` | 移除 Checkbutton 控件、`save_single_spectrum_var` 变量、`measure_spectrum_once` 中的自动保存逻辑、`MeasurementConfig.save_single_spectrum_enabled` 字段 |
| `test_remove_auto_save_spectrum_checkbox_7_25.py` | 新增 4 项验证测试 |
| `PLAN_remove_auto_save_spectrum_checkbox_7_25.md` | 本计划文档 |

## 3. 实施步骤

### 3.1 移除 UI 复选框

删除“6. 光谱仪通信”→“单次光谱采集”区域的 `ttk.Checkbutton`：

```python
# 已删除
# ttk.Checkbutton(
#     tcp_buttons, text="自动保存光谱数据", variable=self.save_single_spectrum_var
# ).grid(row=2, column=0, columnspan=2, padx=4, pady=3, sticky="w")
```

### 3.2 移除 GUI 变量

删除 `MeasurementWorkflowGUI.__init__` 中的 `save_single_spectrum_var` 声明：

```python
# 已删除
# self.save_single_spectrum_var = tk.BooleanVar(value=cfg0.save_single_spectrum_enabled)
```

### 3.3 移除自动保存逻辑

删除 `MeasurementWorkflowGUI.measure_spectrum_once` 中的自动保存分支：

```python
# 已删除
# if self.save_single_spectrum_var.get():
#     try:
#         saved_path = wf.save_single_spectrum_to_xlsx()
#         if saved_path is not None:
#             self.log(f"[TCP] 已自动保存光谱数据：{saved_path}")
#     except Exception as e:
#         self.log(f"[TCP] 自动保存光谱数据失败：{e}")
#         self.log(traceback.format_exc())
```

### 3.4 移除配置字段

删除 `MeasurementConfig` 中的 `save_single_spectrum_enabled` 字段：

```python
# 已删除
# # 单次光谱采集后是否自动保存完整光谱数据到 xlsx
# save_single_spectrum_enabled: bool = bool(_cfg("save_single_spectrum_enabled", True))
```

## 4. 测试计划

| 测试项 | 方法 |
|--------|------|
| `MeasurementConfig` 不再包含 `save_single_spectrum_enabled` | `test_config_has_no_save_single_spectrum_enabled` |
| GUI 实例不再包含 `save_single_spectrum_var` | `test_gui_has_no_save_single_spectrum_var` |
| `measure_spectrum_once` 不再引用已删除变量/文本 | `test_measure_spectrum_once_does_not_reference_checkbox` |
| 主文件源码不再出现复选框文本及变量 | `test_ui_does_not_contain_auto_save_checkbox` |

## 5. 测试结果

```text
test_remove_auto_save_spectrum_checkbox_7_25.py::test_config_has_no_save_single_spectrum_enabled PASSED
test_remove_auto_save_spectrum_checkbox_7_25.py::test_gui_has_no_save_single_spectrum_var PASSED
test_remove_auto_save_spectrum_checkbox_7_25.py::test_measure_spectrum_once_does_not_reference_checkbox PASSED
test_remove_auto_save_spectrum_checkbox_7_25.py::test_ui_does_not_contain_auto_save_checkbox PASSED

============================= 28 passed in 14.66s =============================
```

## 6. 风险与回退

- 该复选框删除后，用户仍可通过“完整循环测量”流程或手动调用 `save_single_spectrum_to_xlsx()` 保存光谱数据；
- 若后续需要恢复自动保存功能，可在 `measure_spectrum_once` 中重新加入保存调用，并决定是否通过新的配置项控制；
- 本次清理仅影响 GUI 控件和自动保存触发逻辑，不删除 `MeasurementWorkflow.save_single_spectrum_to_xlsx()` 本身的保存能力。
