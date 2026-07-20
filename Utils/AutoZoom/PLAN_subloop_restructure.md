# 完整循环测量子循环重构实施计划

## 1. 目标

优化 `0_measurement_workflow_real_virtual_same_detection_7_16.py` 中“运行完整循环测量”的循环逻辑，使单轮循环满足如下新流程：

```
Step 1：角度检测一次，得到 current_angle
Step 1.5（仅第一轮）：在照明光 OFF 前，截取当前 ROI 画面建立聚焦参考图
Step 2：生成保存路径
Step 3：照明光 OFF，并按 stable_wait_ms 等待稳定
Step 4：LabVIEW 光谱采集
Step 5：照明光 ON
Step 6：保存本轮数据；保存角度使用 Step1 的 YOLO-OBB baseline 原始角度

Step 7：打开激光
Step 8：A推动B（logic/rule_ab.py），每次运动后检测B对边角度；
        Step7 角度只用于关闭激光/是否继续推动，不再覆盖本轮保存角度。
Step 9：检测颜色区域中心，与提前选定位置对齐；偏离过大则移动1/2通道
Step 10：补焦判断。计算当前 ROI 与参考图的 FocusScore_ratio；
          若触发阈值，执行补焦。
Step 11：照明光 OFF，并按 stable_wait_ms 等待稳定
Step 12：LabVIEW 光谱采集
Step 13：照明光 ON
Step 14：保存本轮数据；保存角度使用 Step1 的 YOLO-OBB baseline 原始角度

Step 15：继续 Step 7 到 Step 14 的循环（固定次数，可配置）
```

## 2. 涉及文件

| 文件 | 变更内容 |
|------|----------|
| `Utils/AutoZoom/0_measurement_workflow_real_virtual_same_detection_7_16.py` | 新增配置字段 `sub_loop_iterations_per_cycle`；GUI 增加输入框；`run_one_cycle` 重构为先执行 Step 1–6，再进入 Step 7–14 子循环；`laser_on()` 增加重复调用保护 |
| `Utils/AutoZoom/test_subloop_restructure.py` | 新增单元/集成测试 |
| `Utils/AutoZoom/PLAN_subloop_restructure.md` | 本计划 |

## 3. 实施步骤

### 3.1 新增配置字段

在 `MeasurementConfig` 中新增：

```python
sub_loop_iterations_per_cycle: int = int(_cfg("sub_loop_iterations_per_cycle", 1))
```

语义：每个外循环周期内，Step 7–14 子循环重复执行的次数；默认 1，保持与当前行为最接近；0 表示跳过子循环。

### 3.2 GUI 增加输入框

在“基础参数”区域（`max_cycles`、`stable_wait_ms` 附近）增加：

- Label: “子循环次数”
- Entry: 绑定 `self.sub_loop_iterations_per_cycle_var`

并在 `apply_runtime_params`/`_build_config_from_gui` 中同步到 `MeasurementConfig`。

### 3.3 重构 `run_one_cycle`

将原方法拆分为清晰阶段：

1. **Step 1**：角度检测（保持不变）
2. **Step 1.5**：第一轮建立聚焦参考图（保持不变）
3. **Step 2**：生成保存路径（保持不变）
4. **Step 3–6**：照明光 OFF → LabVIEW 光谱采集 → 照明光 ON → 异步保存（使用 Step1 角度）
5. **Step 7–14 子循环**：重复 `sub_loop_iterations_per_cycle` 次：
   - Step 7：`laser_on()`（增加已开启则跳过）
   - Step 8：`run_rule_ab_until_angle_delta`（A 推动 B）
   - Step 9：`run_rule_ac_until_threshold`（颜色区域对齐）
   - Step 10：`run_autofocus_if_needed`（补焦判断）
   - Step 11–14：照明光 OFF → LabVIEW 光谱采集 → 照明光 ON → 异步保存（使用 Step1 角度）
6. 子循环结束后关闭激光（安全兜底），更新上下文并返回。

### 3.4 `laser_on()` 幂等保护

避免子循环多次调用导致激光轴过冲：

```python
def laser_on(self):
    if bool(self.context.get("laser_on", False)):
        self.log("[激光开关] 激光已经是 ON 状态，跳过重复打开")
        return
    ...
```

### 3.5 保存角度策略

所有 `start_save_cycle_result_async` 调用均使用 `angle_before_result=angle_result`（Step1 的 YOLO-OBB baseline），`angle_after_result` 标记为“使用 Step1 角度”。这与当前实现一致，需确保子循环内不覆盖 `save_angle_deg`。

### 3.6 停止/重标定检查点

在子循环每次迭代前后保留原有检查：
- `self.stop_requested`
- `self._is_midrun_recalibration_requested()`
- Step 8/Step 9 返回的重标定原因

## 4. 测试计划

| 测试项 | 方法 |
|--------|------|
| 配置字段默认值 | 断言 `MeasurementConfig().sub_loop_iterations_per_cycle == 1` |
| GUI 变量与配置同步 | 修改 GUI 输入框，调用配置构建方法，断言字段同步 |
| `laser_on()` 幂等 | 连续调用两次，验证第二次不触发真实/虚拟轴运动且 `context["laser_on"]` 仍为 True |
| `run_one_cycle` 子循环次数 | mock 各步骤方法，设置 `sub_loop_iterations_per_cycle=N`，断言 Step 7–14 相关调用恰好 N 次 |
| 保存角度来源 | mock 光谱数据，断言所有 `start_save_cycle_result_async` 的 `angle_before_result` 来自 Step1 |
| 聚焦参考图仅在第一轮建立 | 调用两轮 `run_one_cycle`，断言 `capture_focus_reference` 只在 cycle_index=1 调用一次 |
| 子循环 0 次 | 设置 `sub_loop_iterations_per_cycle=0`，断言子循环体不执行，仅执行 Step 1–6 |
| 安全兜底关激光 | 子循环结束后断言 `laser_off` 被调用 |

测试文件：`Utils/AutoZoom/test_subloop_restructure.py`

## 5. 测试结果

本次新增测试 `test_subloop_restructure.py`：6 passed

| 测试项 | 结果 |
|--------|------|
| `test_config_default_sub_loop_iterations` | PASS |
| `test_laser_on_idempotent` | PASS |
| `test_run_one_cycle_sub_loop_count` | PASS |
| `test_focus_reference_only_first_cycle` | PASS |
| `test_sub_loop_zero` | PASS |
| `test_save_angle_from_step1` | PASS |

相关回归测试：
- `test_autofocus_integration.py`: 12 passed
- `test_full_measurement_c_fix.py`: 4 passed
- `test_spectrum_autofocus_loop.py`: 11 passed

合计：33 passed。

> 注：其余历史测试文件（如 `test_final_crash_fix.py`、`test_pi_backend_7_16.py` 等）因模块级 QApplication 单例冲突或依赖真实 LabVIEW TCP 连接，无法在本次批量运行中通过，属于既有环境/架构问题，与本次改动无关。

## 6. 风险与回退

- `run_one_cycle` 是核心流程，重构步骤顺序可能影响现有 Δw 判断、中途重标定等逻辑。改动时保持原有检查点不变。
- 子循环次数默认 1，用户显式设置为 1 时行为与当前版本最接近（仅多了一次 Step 3–6 的初始光谱采集）。
- `laser_on()` 增加幂等保护后，多次调用不会导致 Newport 轴重复运动；真实环境需确认激光状态与 `context["laser_on"]` 一致。
- 若子循环内发生非 Bmask/角度类失败，仍按原逻辑停止完整循环测量。
