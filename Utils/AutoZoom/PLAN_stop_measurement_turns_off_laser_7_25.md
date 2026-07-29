# 7_25 手动停止测量后关闭激光功能验证与测试

## 1. 需求

在手动点击"停止测量"按钮后，系统应自动关闭激光，确保安全。

## 2. 现状分析

### 2.1 已有实现

经代码分析，该功能已在 `MeasurementWorkflow.request_stop()` 方法中实现（第14034-14054行）：

```python
def request_stop(self):
    self.stop_requested = True
    self.is_measuring = False
    self.log("[流程] 收到停止请求")

    # 停止测量后，兜底关闭激光；当前版本不再操作 Rigol。
    try:
        if self.context.get("laser_on", False):
            self.laser_off()
            self.log("[流程] 停止测量：激光已 OFF")
    except Exception as e:
        self.log(f"[流程] 停止测量时关闭激光失败：{e}")

    # 兜底：将当前未完成轮的已有数据刷入 measurement_summary.csv，防止数据丢失
    try:
        self._flush_unsaved_cycle_to_csv()
    except Exception as e:
        self.log(f"[流程] 停止测量时刷新未保存 CSV 失败：{e}")

    self.save_summary_xlsx()
    self.notify_update()
```

### 2.2 激光关闭触发点

代码中存在三处激光关闭逻辑：

| 场景 | 方法 | 行号 | 说明 |
|------|------|------|------|
| 手动停止测量 | `request_stop()` | 14039-14045 | 已实现 |
| 测量正常结束 | `finish()` | 14134-14140 | 已实现 |
| 关闭全部设备 | `close_all()` | 14190-14194 | 已实现 |

### 2.3 GUI 调用链

```
GUI.request_stop() [行17928]
    ↓
MeasurementWorkflow.request_stop() [行14034]
    ↓
laser_off() [行2575]
```

## 3. 实施内容

由于功能已实现，本次主要工作是：

1. 创建测试用例验证激光关闭行为
2. 确保测试覆盖以下场景：
   - 激光开启状态下手动停止测量 → 应关闭激光
   - 激光已关闭状态下手动停止测量 → 不重复调用 laser_off
   - laser_off 失败时应有日志记录，不阻塞流程

## 4. 涉及文件

| 文件 | 变更内容 |
|------|----------|
| `PLAN_stop_measurement_turns_off_laser_7_25.md` | 本计划文档 |
| `test_stop_measurement_turns_off_laser_7_25.py` | 新增测试文件 |

## 5. 测试计划

| 测试项 | 方法 |
|--------|------|
| 激光开启时停止测量应关闭激光 | `test_request_stop_turns_off_laser_when_on` |
| 激光已关闭时停止测量不重复调用 | `test_request_stop_does_not_call_laser_off_when_already_off` |
| laser_off 失败时有日志记录 | `test_request_stop_handles_laser_off_failure` |
| finish 方法正确关闭激光 | `test_finish_turns_off_laser` |
| close_all 方法正确关闭激光 | `test_close_all_turns_off_laser` |

## 6. 测试结果

```text
test_stop_measurement_turns_off_laser_7_25.py::test_request_stop_turns_off_laser_when_on PASSED
test_stop_measurement_turns_off_laser_7_25.py::test_request_stop_does_not_call_laser_off_when_already_off PASSED
test_stop_measurement_turns_off_laser_7_25.py::test_request_stop_handles_laser_off_failure PASSED
test_stop_measurement_turns_off_laser_7_25.py::test_finish_turns_off_laser PASSED
test_stop_measurement_turns_off_laser_7_25.py::test_close_all_turns_off_laser PASSED
test_stop_measurement_turns_off_laser_7_25.py::test_close_all_does_not_call_laser_off_when_already_off PASSED

============================= 39 passed in 14.04s =============================
```

## 7. 风险与回退

- 若 `laser_off()` 抛出异常，当前实现会捕获并记录日志，不阻塞其他清理流程
- 若需移除激光关闭逻辑，只需删除 `request_stop()` 中的 try-except 块
- 虚拟模式下使用 `VirtualNewportPicomotor8742`，行为与真实硬件一致