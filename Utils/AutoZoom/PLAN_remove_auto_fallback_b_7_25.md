# 7_25 去除 RuleAB-B 兜底生成逻辑实施计划

## 1. 需求

去除 `_call_initialize_abc_with_loaded_calibration` 中基于 A 点偏移自动生成默认 B 点的兜底逻辑。

## 2. 背景分析

### 2.1 原兜底逻辑位置

- **文件**：`0_measurement_workflow_real_virtual_same_detection_7_25.py`
- **方法**：`_call_initialize_abc_with_loaded_calibration`
- **原行号**：2060-2097

### 2.2 原兜底逻辑行为

当标定包缺少 B 正点时，系统会：

1. 截取当前 RuleAB 区域图像
2. 基于 A 点位置 + 固定偏移（默认 `+100, 0`）生成默认 B 点
3. 写入 `state.rule_ab_b_positive_points`
4. 将该点注入外部 RuleAB follower 模块

### 2.3 对 A 区域检测的潜在影响

虽然该兜底 B 点声称"仅用于兼容 RuleAB 初始化"，但它可能通过以下途径影响 A 区域检测：

| 影响点 | 说明 |
|--------|------|
| 标定状态污染 | `state.rule_ab_b_positive_points` 被修改，可能改变后续 B 点相关逻辑 |
| 额外截图 | 调用 `_capture_current_rule_ab_frame()` 可能改变内部状态或缓存 |
| 注入 follower | 通过 `_inject_calibration_into_rule_ab_follower` 将错误 B 点写入 prompt_points，可能影响 SAM2/A/B/C 首帧分割 |
| A/B mask 守卫 | A = A - dilate(B_current)，错误 B 点可能导致 A mask 被错误减去 |

## 3. 实施内容

### 3.1 代码变更

在 `_call_initialize_abc_with_loaded_calibration` 中删除兜底 B 点生成代码块：

```python
# 已删除：
# b_pos = self._calib_points_to_tuples(state.rule_ab_b_positive_points)
# if not b_pos:
#     try:
#         fallback_dir = self.output_root / "rule_ab_from_measurement" / "fallback_b"
#         frame = self._capture_current_rule_ab_frame(fallback_dir)
#         ...
```

同时更新了方法的 docstring，去除相关描述。

### 3.2 保留的行为

- `_inject_calibration_into_rule_ab_follower` 仍然会把标定状态中的 B 点注入 follower（如果标定包本身有 B 点）
- 角度检测 B 点（`angle_b_positive_points`）不受影响
- 若 RuleAB 模块强制需要 B 点但标定包未提供，则初始化会失败并抛出异常

## 4. 涉及文件

| 文件 | 变更内容 |
|------|----------|
| `0_measurement_workflow_real_virtual_same_detection_7_25.py` | 删除兜底 B 点生成逻辑，更新 docstring |
| `PLAN_remove_auto_fallback_b_7_25.md` | 本计划文档 |
| `test_remove_auto_fallback_b_7_25.py` | 新增测试文件 |

## 5. 测试计划

| 测试项 | 方法 |
|--------|------|
| 缺少 B 点时不再自动生成默认 B 点 | `test_no_auto_fallback_b_when_missing` |
| 注入 follower 的 B 点为空列表 | `test_injected_b_points_empty_when_missing` |
| 原有 B 点标定不受影响 | `test_existing_b_points_preserved` |
| 方法中不再调用 `_capture_current_rule_ab_frame` | `test_no_capture_called_for_fallback_b` |

## 6. 测试结果

```text
============================= 41 passed in 17.43s =============================
```

## 7. 风险与回退

- 去除兜底后，如果 RuleAB 模块仍强制需要 B 点且标定包未提供，初始化会失败
- 若后续需要恢复兜底，可重新添加该代码块
- 建议同时检查是否需要恢复 GUI 中的"标定 RuleAB-B"入口
