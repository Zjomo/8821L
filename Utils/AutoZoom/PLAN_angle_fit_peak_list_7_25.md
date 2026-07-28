# 角度-拟合峰值列表显示初始化点（序号 0）

## 1. 需求

UI 右侧的"角度-拟合峰值列表"中，要求将"第一轮初始化"的"角度"与"拟合峰值"也补充上（序号为 0）。

## 2. 涉及文件与函数

- `0_measurement_workflow_real_virtual_same_detection_7_25.py`
  - `MeasurementWorkflow.run_one_cycle()` —— 单轮完整测量流程
  - `MeasurementWorkflow._append_plot_point()` —— 记录 GUI 绘图/列表数据点
  - `MeasurementWorkflowGUI.update_angle_fit_peak_plot()` —— 更新右侧列表与中间图像
  - `MeasurementWorkflowGUI.update_angle_fit_xy_list()` —— 把数据点刷新到右侧 Treeview

## 3. 根因

原代码在 `run_one_cycle()` 的初始光谱采集阶段（Step 3-6，序号 0）使用条件判断：

```python
initial_append_plot = sub_loop_count == 0
```

只有当不存在实际主循环（`sub_loop_count == 0`）时，才会把初始化点追加到 `plot_points`。当存在子循环时，序号 0 的初始化点被丢弃，右侧列表直接从序号 1 开始显示。

此外，`update_angle_fit_peak_plot()` 中右侧列表的显示序号从 1 开始：

```python
record_index = len(list_points) + 1
```

即使追加了序号 0 的点，也会显示为序号 1。

## 4. 实现方案

### 4.1 始终追加初始化绘图点

在 `run_one_cycle()` 的 Step 3-6 阶段：

```python
# Step 3-6：初始光谱采集（基准测量，序号0）
# 这是所有后续测量的基准参考。
# 无论是否存在实际主循环，都向角度-拟合峰值列表追加序号 0 的初始化点，
# 确保右侧列表显示"第一轮初始化"的角度与拟合峰值。
sub_loop_count = max(0, int(getattr(self.cfg, "sub_loop_iterations_per_cycle", 1)))
initial_append_plot = True
```

### 4.2 右侧列表序号从 0 开始

在 `MeasurementWorkflowGUI.update_angle_fit_peak_plot()` 中：

```python
record_index = len(list_points)
```

这样：
- 初始化点（cycle_index=0）显示为序号 0
- 后续有效记录依次显示为 1, 2, 3...

真实 cycle_index 仍保存在 `point["cycle_index"]`、CSV/XLSX 和日志中。

## 5. 测试项

| 测试项 | 方法 |
|--------|------|
| cycle_index=0 的角度与拟合峰值被记录 | `test_append_plot_point_zero_recorded` |
| 右侧列表中初始化点显示为序号 0 | `test_update_angle_fit_peak_plot_starts_at_zero` |
| 有子循环时仍追加序号 0 点 | `test_initial_append_plot_is_always_true_in_run_one_cycle` |

## 6. 测试结果

```text
3 passed in 11.82s
```

## 7. 风险与回退

- 右侧列表序号从 0 开始后，如果用户期望从 1 开始，可能需要调整习惯；
- 若后续又想隐藏序号 0 的初始化点，只需把 `initial_append_plot` 改回条件判断即可；
- 该改动不影响 CSV/XLSX 保存的真实 cycle_index。
