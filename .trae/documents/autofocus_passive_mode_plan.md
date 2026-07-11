# AutoZoom/Focus 被动补焦模式实施计划

## Context
当前 AutoZoom 的自动对焦以“主动技能”形式存在：用户点击“启动闭环”后，每轮只调用一次 `check_and_autofocus()`；若本轮分数低于触发阈值，则执行一次补焦搜索并进入下一轮等待。需求要求：
1. 把默认触发阈值提高到 **0.95**，并在 UI/文档中说明“目标阈值”的含义。
2. 在“循环参数”中增加一个**被动补焦**开关，开启后后台会在分数低于触发阈值时连续补焦，直到分数回升到触发阈值以上（或达到最大尝试次数），再恢复常规间隔监测。
3. 补充对 `FocusScore_ratio` 含义、>1 解释及潜在问题的讨论。

## 设计决策
- **被动模式**：采用“新增开关、默认关闭”方案，保留现有主动循环行为；开启后每轮监测后追加连续补焦逻辑。
- **阈值设置**：触发阈值默认固定为 **0.95**；目标阈值仍保持独立可配置，默认 **0.95**，闭环搜索以目标阈值作为停止条件，监测以触发阈值作为触发条件。
- **补焦上限**：被动模式下单次连续补焦设置最大尝试次数（默认 10），防止 Z 轴无限震荡。
- **兼容性**：被动模式仅在 `autofocus_enabled=True` 时生效；关闭时不改变任何现有流程。

## 关键文件与改动点

### 1. 配置层
- `Utils/AutoZoom/Focus/config.py`
  - `autofocus_focus_trigger_ratio` 默认值改为 `0.95`。
  - 新增 `autofocus_passive_mode: bool = False`
  - 新增 `autofocus_passive_max_attempts: int = 10`
  - 给 `autofocus_stop_ratio` 补充注释说明其为“闭环搜索停止目标值”。
- `Utils/AutoZoom/autofocus_qt_ui/config.py`
  - `DEFAULT_TRIGGER_RATIO` 改为 `0.95`
  - 新增 `DEFAULT_PASSIVE_MODE: bool = False`
  - 新增 `DEFAULT_PASSIVE_MAX_ATTEMPTS: int = 10`

### 2. UI 层
- `Utils/AutoZoom/autofocus_qt_ui/app.py`
  - `UiRuntimeArgs` 新增 `passive_mode`、`passive_max_attempts`。
  - “循环参数”组新增：
    - `passive_check`：复选框“启用被动补焦”
    - `passive_attempts_spin`：最大连续补焦次数（1~50，默认 10）
  - 给“目标阈值”和新增控件添加中文 tooltip。
  - `_collect_args()` 读取新控件值；`_build_config()` 写入 `AutofocusConfig`。

### 3. Worker 循环逻辑
- `Utils/AutoZoom/autofocus_qt_ui/worker.py`
  - 在 `_run_loop()` 常规调用 `check_and_autofocus()` 之后、进入下一轮等待之前，插入被动补焦循环：
    - 条件：`passive_mode=True`、`autofocus_enabled=True`、本轮分数 `< trigger_ratio`
    - 循环调用 `check_and_autofocus()`，直到 `score >= trigger_ratio` 或达到 `max_attempts`
    - 每次调用后通过 `score_updated`、`log` 信号同步 UI
    - 循环期间不调用 `simulator.next_cycle()`，避免漂移重复施加
    - 循环内高频检查 `self._state.running` / `self._state.paused`，保证停止/暂停响应

### 4. 文档层
- `Utils/AutoZoom/Focus/README.md`
  - 更新配置表默认值。
  - 新增“FocusScore_ratio 含义与注意事项”讨论节，回答：
    1. FocusScore 是否仅与清晰度相关、与物体形状关系不大；
    2. 大于 1 是否说明当前图像更清晰；
    3. 潜在 bug/注意事项（限幅、参考接近零、触发/目标阈值关系、被动模式连续移动风险）。
- `Utils/AutoZoom/autofocus_qt_ui/README.md`
  - 更新默认参数与使用流程说明。

### 5. 测试
- 新增 `Utils/AutoZoom/test_passive_mode.py`
  - `test_default_thresholds`：验证配置/UI 默认触发阈值 0.95、目标阈值 0.95、被动默认关闭/最大 10 次。
  - `test_passive_mode_repeats_until_above_trigger`：mock `check_and_autofocus` 返回 `[0.85, 0.88, 0.96]`，验证被动开启时连续调用 3 次后停止。
  - `test_passive_mode_respects_max_attempts`：mock 始终返回低分，验证达到最大尝试次数后退出被动循环。
  - `test_passive_mode_off_keeps_single_autofocus`：验证关闭被动时每轮仅调用一次。
  - `test_passive_disabled_when_autofocus_off`：验证自动补焦关闭时不进入被动循环。
  - `test_integration_simulator_passive_recovery`：使用 `FocusSimulator` 验证被动模式最终能把分数拉回 0.95 以上。

## 实施顺序
1. 修改 `Focus/config.py` 与 `autofocus_qt_ui/config.py`。
2. 修改 `app.py`：新增 UI 控件、读取、配置传递、tooltip。
3. 修改 `worker.py`：插入被动补焦循环。
4. 更新两份 README 文档。
5. 编写并运行 `test_passive_mode.py`。
6. 在模拟模式下启动 UI 验证被动行为。

## 验证方式
- 运行 `python -m pytest Utils/AutoZoom/test_passive_mode.py -v`（或等价测试命令）。
- 启动 UI：`python -m Utils.AutoZoom.autofocus_qt_ui`，选择模拟模式，建立参考后勾选“启用被动补焦”，启动闭环，观察日志中是否出现连续补焦并在分数 ≥ 0.95 后恢复等待。
