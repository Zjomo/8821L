# 运行完整循环测量 Step 日志实施计划

## 1. 目标

确保 `运行完整循环测量` 过程中，每一步（Step）都在日志中以统一格式体现：

```
[YYYY-MM-DD HH:MM:SS] ========== Step X：<描述> ==========
```

## 2. 当前状态

`run_one_cycle` 方法中已包含 Step 1-9 的全部日志，无需新增代码：

| 步骤 | 日志 | 描述 |
|------|------|------|
| Step 1 | `========== Step 1：Bmask最长边角度检测 current_angle ==========` | YOLO-OBB 检测 Bmask 最长边 |
| Step 2 | `========== Step 2：生成保存路径 ==========` | 构建本轮保存目录 |
| Step 3 | `========== Step 3：照明光 OFF，等待稳定 ==========` | 关闭照明光 + 等待 |
| Step 4 | `========== Step 4：LabVIEW 光谱采集 ==========` | 通过 TCP 请求 LabVIEW |
| Step 5 | `========== Step 5：照明光 ON ==========` | 打开照明光 |
| Step 5.5 | `========== Step 5.5：保存数据（异步线程，使用 Step1 最终角度） ==========` | 提交异步保存 |
| Step 6 | `========== Step 6：打开激光 ==========` | 移动激光轴开启 |
| Step 7 | `========== Step 7：A推动B，并用当前帧Bmask最长边检测B角度 ==========` | 推动 A 直到角度变化 |
| Step 8 | `========== Step 8：关闭激光 ==========` | 移动激光轴关闭 |
| Step 9 | `========== Step 9：检测颜色区域中心，必要时移动1/2通道 ==========` | 颜色区域对齐 |

## 3. 实施步骤

### 3.1 已完成

- `run_one_cycle` 方法中已包含完整的 Step 1-9 日志
- 日志格式统一为 `========== Step X：<描述> ==========`
- 通过 LogManager 实时写入 `./Log` 目录

### 3.2 测试

新增测试文件 [`test_step_logs.py`](file:///e:/jupyter%20file/2_Optics/8821L/Utils/AutoZoom/test_step_logs.py)：

- `test_run_one_cycle_has_step_logs`：源码扫描，验证 10 个 Step 日志都存在
- `test_step_log_format_consistent`：验证日志格式统一且 Step 编号顺序正确
- `test_step_logs_cover_full_cycle`：验证关键步骤都有对应日志
- `test_step_log_via_runtime`：mock 完整流程，运行 `run_one_cycle`，捕获每步 log 调用

## 4. 测试结果

```
PASS: run_one_cycle_has_step_logs（10 步）
PASS: step_log_format_consistent（10 个 Step 日志格式正确）
PASS: step_logs_cover_full_cycle（10 个关键步骤）
PASS: step_log_via_runtime（10 个 Step 日志全部输出）

All 4 tests passed!
```

## 5. 日志查看方式

### 5.1 文件查看

日志由 LogManager 实时写入 `./Log/log_YYYYMMDD_HHMMSS.txt`：

```powershell
Get-Content "E:\jupyter file\2_Optics\8821L\Log\log_20260724_*.txt" -Tail 50
```

### 5.2 GUI 实时查看

GUI 日志 Text 控件实时显示所有 `self.log()` 调用，无需额外配置。

## 6. 风险与回退

- 日志输出已统一，无需回退
- 若 Step 编号需要重新编排，只需修改 `self.log()` 中的字符串，无需调整其他逻辑