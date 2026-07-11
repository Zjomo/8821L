# AutoZoom/Focus 被动补焦模式 V2 实施计划

## Context
上一版已实现被动补焦开关与“最大连续补焦次数”，但行为是“每轮分数低于阈值时，在本轮内连续补焦最多 N 次”。新需求要求把被动模式改为：
- 勾选“启用被动补焦”后，**不再参考“循环轮数”**，仅按“间隔”无限运行；
- 新增“连续达标次数 n”：FocusScore 连续 n 轮大于触发阈值后自动停止；
- 保留原有“最大连续补焦次数”作为独立参数；
- 手动点击“停止”可随时中断。

## 设计决策
- **触发节奏**：采用“按间隔周期补焦”方案。每轮只调用一次 `check_and_autofocus()`，分数未达标时由控制器内部完成一次闭环搜索，然后等待间隔进入下一轮。
- **两个独立参数**：
  - `autofocus_passive_max_attempts` → UI 标签改为“最大连续补焦次数”：作为安全兜底，记录连续多轮补焦后仍未达标的次数上限（或仍保留为单轮内最大尝试次数，视实现而定；本计划按“单轮失败后累计安全上限”使用）。
  - `autofocus_passive_consecutive_good` → UI 标签“连续达标次数”：成功停止条件，FocusScore 连续 n 轮大于触发阈值后停止。
- **循环轮数 UI**：勾选被动模式后禁用并置灰 `cycles_spin`。
- **手动停止**：复用现有 `_stop_loop()`，设置 `worker._state.running = False`。

## 关键文件与改动点

### 1. 配置层
- `Utils/AutoZoom/Focus/config.py`
  - 在 `autofocus_passive_max_attempts` 后新增：
    ```python
    autofocus_passive_consecutive_good: int = 3
    """被动补焦模式下，FocusScore 连续多少轮大于触发阈值后自动停止循环"""
    ```
- `Utils/AutoZoom/autofocus_qt_ui/config.py`
  - 新增 `DEFAULT_PASSIVE_CONSECUTIVE_GOOD: int = 3`

### 2. UI 层（`Utils/AutoZoom/autofocus_qt_ui/app.py`）
- `UiRuntimeArgs` 新增 `passive_consecutive_good: int`。
- 循环参数组：
  - 将原有 `passive_attempts_spin` 标签从“最大连续次数:”改为“最大连续补焦次数:”，tooltip 明确为“单轮分数未达标时连续补焦的最大尝试次数/安全上限”。
  - 新增 `passive_good_spin`（范围 1~100，默认值 3），标签“连续达标次数:”，tooltip 说明“FocusScore 连续多少轮超过触发阈值后自动停止循环”。
- 新增槽函数 `_on_passive_changed(state)`：勾选/取消时启用或禁用 `cycles_spin`。
- `_connect_signals()` 中连接 `passive_check.stateChanged` 到 `_on_passive_changed`。
- `_apply_default_values()` 末尾调用一次 `_on_passive_changed` 初始化状态。
- `_set_controls_running()` 停止运行后，根据被动状态恢复 `cycles_spin` 可用性。
- `_collect_args()` 读取 `passive_consecutive_good`。
- `_build_config()` 写入 `autofocus_passive_consecutive_good`。

### 3. Worker 循环逻辑（`Utils/AutoZoom/autofocus_qt_ui/worker.py`）
- `_run_loop()` 开头读取：
  ```python
  passive_mode = getattr(self._cfg, "autofocus_passive_mode", False)
  trigger_ratio = float(self._cfg.autofocus_focus_trigger_ratio)
  passive_consecutive_good = max(1, int(getattr(self._cfg, "autofocus_passive_consecutive_good", 3)))
  consecutive_good_count = 0
  ```
- 循环控制：
  - 主动模式（`passive_mode=False`）：仍使用 `cycles` 作为上限。
  - 被动模式（`passive_mode=True`）：忽略 `cycles`，`cycle_index` 无限递增；`status_changed` 的 total 传 0 表示无限运行。
- 每轮调用 `check_and_autofocus(cycle_index=cycle, save_dir=None)` 一次。
- 被动模式下的分数处理：
  - `score > trigger_ratio`：`consecutive_good_count += 1`，日志输出当前计数；若 `>= passive_consecutive_good` 则日志“连续达标 n 轮，停止循环”并 `break`。
  - `score <= trigger_ratio`：重置 `consecutive_good_count = 0`；本轮已在 `check_and_autofocus` 内触发补焦，等待间隔后进入下一轮。
- 间隔等待对两种模式都保留。

### 4. 测试更新（`Utils/AutoZoom/test_passive_mode.py`）
- `_make_cfg()` 新增 `passive_consecutive_good` 参数。
- 新增 `test_default_passive_consecutive_good()`：验证默认值为 3。
- 更新 `test_ui_controls_reflect_defaults()`：
  - 断言 `passive_attempts_spin` 值为 10；
  - 断言 `passive_good_spin` 值为 3；
  - 勾选被动模式后断言 `cycles_spin` 被禁用。
- 更新被动模式单元测试：
  - `test_passive_mode_repeats_until_above_trigger`：改为 `passive_consecutive_good=1`，mock 序列 `[0.85, 0.88, 0.96]`，验证连续补焦后最终达标停止。
  - `test_passive_mode_respects_max_attempts`：验证未达标时按最大尝试次数/安全上限停止或持续运行（根据实现）。
  - 新增 `test_passive_mode_ignores_cycles`：`cycles=1` 但被动模式运行多轮，证明忽略循环轮数。
  - 新增 `test_passive_mode_consecutive_good_stop`：mock 连续返回高分，验证运行 `passive_consecutive_good` 轮后自动停止。
  - 新增 `test_passive_mode_manual_stop`：运行中调用 `worker.stop_loop()`，验证循环立即退出。
- `_run_worker()` 增加 `max_cycles` 参数，用于测试时手动限制运行轮数。

### 5. 文档更新
- `Utils/AutoZoom/Focus/README.md`
  - 在配置表中新增 `autofocus_passive_consecutive_good` 一行。
  - 补充被动模式行为说明：忽略循环轮数、以连续达标次数作为自动停止条件、手动停止始终有效。
- `Utils/AutoZoom/autofocus_qt_ui/README.md`
  - 更新使用流程，说明被动模式的两种结束条件：连续达标停止、手动停止。

## 实施顺序
1. 修改 `Focus/config.py` 与 `autofocus_qt_ui/config.py`。
2. 修改 `app.py`：新增 UI 控件、禁用/启用 `cycles_spin`、参数传递。
3. 修改 `worker.py`：重写被动循环逻辑。
4. 更新 `test_passive_mode.py` 并运行通过。
5. 更新两份 README。
6. 在 UI 模拟模式下验证被动行为。

## 验证方式
- 运行 `python test_passive_mode.py`。
- 启动 UI，验证：
  - 未勾选被动模式时“循环轮数”可用；
  - 勾选后“循环轮数”变灰禁用；
  - 被动模式下连续达标指定轮数后自动停止；
  - 点击“停止”立即中断。
