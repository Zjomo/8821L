# 自动补焦参数集成与光谱数据保存实施计划

## 1. 目标

在 `0_measurement_workflow_real_virtual_same_detection_7_16.py` 系统中集成 `autofocus_qt_ui` 已有的补焦参数与闭环能力，实现：

1. **光谱补焦循环 UI 可配置全部补焦参数**；
2. **完整循环测量在颜色区域对齐后自动进行 FocusScore 判断与补焦**，并将光照/光谱采集步骤移到补焦之后；
3. **单次光谱采集后自动导出完整光谱数据到 xlsx**。

## 2. 涉及文件

| 文件 | 变更内容 |
|------|----------|
| `Utils/AutoZoom/0_measurement_workflow_real_virtual_same_detection_7_16.py` | 光谱补焦循环 UI 增加参数；MeasurementWorkflow 增加聚焦参考图管理与补焦方法；`run_one_cycle` 调整步骤顺序；`measure_spectrum_once` 增加 xlsx 保存 |
| `Utils/AutoZoom/spectrum_autofocus_loop.py` | `_configure_passive_autofocus()` 改为优先使用 `cfg` 已传入值，不再全部硬覆盖；暴露 `build_reference_from_image` 的复用入口 |
| `Utils/AutoZoom/Focus/scorer.py` | 已在 `build_reference`/`score_ratio` 提供能力，可能需要增加 `build_reference_from_image` 公共方法（若尚未存在） |
| `Utils/AutoZoom/tests/test_autofocus_integration.py` | 新增测试 |

## 3. 实施步骤

### 3.1 需求 1：光谱补焦循环 UI 增加补焦参数

在 `0_measurement_workflow_real_virtual_same_detection_7_16.py` 的 **9. 光谱补焦循环** 区域增加以下控件：

- 触发阈值 `saf_trigger_ratio_var`（默认 0.95）
- 目标阈值 `saf_stop_ratio_var`（默认 0.95）
- 连续触发次数 `saf_trigger_count_var`（默认 3）
- 使用绝对值区间 `saf_trigger_absolute_var`（默认 True）
- FocusScore 检测模式 `saf_detection_only_var`（默认 False）
- 被动补焦模式 `saf_passive_mode_var`（默认 True）
- 最大连续补焦次数 `saf_passive_attempts_var`（默认 10）
- 连续达标次数 `saf_passive_good_var`（默认 5）
- 关闭达标阈值 `saf_disable_auto_stop_var`（默认 False）
- Z 轴启用 `saf_z_enabled_var`（默认 True）
- Z 轴号 `saf_z_axis_var`（默认 1）
- Z 速度 `saf_z_speed_var`（默认 100）
- Z 加速度 `saf_z_accel_var`（默认 100）
- 搜索策略 `saf_search_strategy_var`（默认 hill_climb）
- 补焦间隔 `saf_interval_var`（默认 1.0）
- 光谱间等待 `saf_wait_between_spectrum_var`（默认 120.0）

修改 `_make_saf_config()` 将上述变量写入 `AutofocusConfig`。

修改 `spectrum_autofocus_loop.py` 的 `_configure_passive_autofocus()`：只在对应字段未显式配置（或为默认值）时才使用类默认，避免覆盖 GUI 传入值。

### 3.2 需求 2：完整循环测量嵌入补焦逻辑

#### 3.2.1 MeasurementWorkflow 新增状态与方法

新增实例属性：

- `_focus_reference_image`: `np.ndarray | None`
- `_focus_reference_metrics`: `dict | None`
- `_focus_scorer`: `FocusScorer | None`
- `_focus_metrics_calc`: `FocusMetricsCalculator | None`
- `_focus_controller`: `AutofocusController | None`
- `_focus_reference_roi`: 与 `saf_roi_var` 一致

新增方法：

- `_ensure_focus_components()`: 延迟初始化 FocusScorer/FocusMetricsCalculator/AutofocusController。
- `capture_focus_reference(cycle_index)`: 第一轮回环在 light_off 前调用。使用当前 ROI（`saf_roi_var`）截取画面，建立参考基线并保存参考图。
- `compute_current_focus_score()`: 截取当前 ROI，计算 `FocusScore_ratio`。
- `run_autofocus_if_needed(cycle_index)`: 根据当前 `FocusScore_ratio`、触发阈值、绝对值区间、连续触发次数判断是否触发；触发后调用 `AutofocusController.run_closed_loop()` 执行补焦，返回最终分数与是否触发。

#### 3.2.2 调整 `run_one_cycle` 步骤顺序

新的单轮流程：

1. **Step 1**：Bmask 最长边角度检测（不变）。
2. **Step 2**：生成保存路径（不变）。
3. **Step 1.5（仅第一轮）**：在照明光 OFF 前，截取当前 ROI 画面建立聚焦参考图。
4. **Step 6**：打开激光（不变）。
5. **Step 7**：A 推动 B（不变）。
6. **Step 8**：关闭激光（不变）。
7. **Step 9**：检测颜色区域中心，与选定位置对齐（不变）。
8. **Step 9.5**：补焦判断。计算当前 ROI 与参考图的 `FocusScore_ratio`；若触发阈值，执行补焦。
9. **Step 3**：照明光 OFF，等待稳定。
10. **Step 4**：LabVIEW 光谱采集。
11. **Step 5**：照明光 ON。
12. **Step 5.5**：保存本轮数据。

> 注意：Step 7/8/9 依赖激光，因此光照/光谱采集必须放在颜色对齐和补焦之后；参考图建立必须在第一轮 light_off 之前。

#### 3.2.3 补焦触发判断

复用 `Focus.controller.focus_score_ratio_in_tolerance(score, trigger_ratio, absolute)`：

- 未达标时累计 `consecutive_focus_low_count`。
- 当连续次数 >= `trigger_count` 时触发补焦。
- 补焦完成后重置计数。

#### 3.2.4 降级与容错

- 若参考图未建立（例如第一轮截图失败），跳过补焦，按原流程继续。
- 若 `FocusScore_ratio` 为 None（未建立参考），记录日志并跳过。
- virtual 模式下补焦使用 `Focus.simulator.create_demo_environment` 避免硬件报错。

### 3.3 需求 3：单次光谱采集保存 xlsx

#### 3.3.1 导出内容

在点击 **单次光谱采集** 后，将当前 context 中的光谱数据保存为：

```
{output_root}/save/{MM.DD}/measurement_summary_YYYYMMDD_HHMMSS.xlsx
```

xlsx 包含一个工作表 `光谱数据`，列：

- 波长 / index
- 原始强度 raw_values
- 阈值滤波 intensity threshold_values
- 中值滤波 intensity median_values
- 拟合曲线 fit_values

若 `x_axis_values` 可用，第一列为波长；否则使用 `0..N-1` index。

#### 3.3.2 实现

新增方法：

- `save_single_spectrum_to_xlsx()`: 读取当前 `self.context` 光谱数组，生成带时间戳的 xlsx。

在 `measure_spectrum_once()` 成功采集后调用，并在 `tcp_result_var` 中显示保存路径。

## 4. 测试计划

| 测试项 | 方法 |
|--------|------|
| UI 控件存在 | 启动 `MeasurementWorkflowGUI`（不进入 mainloop），断言光谱补焦循环区域存在阈值、绝对值、被动模式等输入控件 |
| `_make_saf_config` 参数传递 | 设置 GUI 变量，调用 `_make_saf_config()`，断言返回的 `AutofocusConfig` 各字段与 GUI 一致 |
| 聚焦参考图建立 | 构造 synthetic ROI 图像，调用 `capture_focus_reference()`，断言 `_focus_reference_image` 非空且 scorer 已就绪 |
| FocusScore_ratio 计算 | 使用清晰/模糊合成图像，调用 `compute_current_focus_score()`，断言清晰图 ratio ~1.0，模糊图 ratio 明显低于阈值 |
| 补焦触发逻辑 | 构造 ratio=0.8、触发阈值 0.95、连续 1 次的场景，断言 `run_autofocus_if_needed()` 返回 triggered=True；ratio=1.0 时不触发 |
| `run_one_cycle` 流程顺序 | 对 virtual workflow mock 激光/光谱/RuleAB/颜色对齐，断言第一轮调用 `capture_focus_reference`，Step9 后调用补焦与采光谱 |
| 单次光谱 xlsx 保存 | mock LabVIEW result 填入 context，调用 `save_single_spectrum_to_xlsx()`，断言文件存在、工作表正确、行数等于数据长度 |
| 边界：无参考图 | 在未建立参考时调用 `compute_current_focus_score()`，断言返回 None 且不抛异常 |

测试文件：`Utils/AutoZoom/test_autofocus_integration.py`

测试结果：`27 passed`（新测试 12 项 + 相关回归测试 15 项）：
- `test_autofocus_integration.py`: 12 passed
- `test_full_measurement_c_fix.py`: 4 passed
- `test_import_7_16.py`: 1 passed（脚本形式）
- `test_spectrum_autofocus_loop.py`: 11 passed（并修复了 `interval_s` 实例属性化后的测试适配）

> 注：其余历史测试文件（如 `test_final_crash_fix.py`、`test_pi_backend_7_16.py` 等）因模块级 QApplication 单例冲突或依赖真实 LabVIEW TCP 连接，无法在本次批量运行中通过，属于既有环境/架构问题，与本次改动无关。

## 5. 风险与回退

- `run_one_cycle` 步骤顺序调整较大，可能影响现有 Δw 判断、中途重标定等逻辑。改动时保持原有的停止/重标定检查点不变。
- 补焦闭环依赖 Z 轴硬件，virtual 模式下使用 simulator；真实环境需用户确认 Z 轴参数。
- xlsx 保存依赖 openpyxl，已在项目初始化中导入。
