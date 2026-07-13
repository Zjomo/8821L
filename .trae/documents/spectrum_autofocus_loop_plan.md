# 光谱补焦循环功能实施计划

> **当前状态**：核心实现与 GUI 集成已完成；剩余工作为新增单元/集成测试、回归测试、静态检查与 README 更新。

## 1. 摘要

在现有 tkinter GUI `0_measurement_workflow_real_virtual_same_detection_6.25.py` 中新增“9. 光谱补焦循环”独立控制区与后台线程，实现：

1. 手动交互式框选 ROI，并以其截图建立 Focus 参考基线；
2. 循环执行：关照明 → LabVIEW 光谱采集 → 开照明 → 保存数据；
3. 等待 2 min；
4. 启动被动补焦检测（默认参数已固定）；
5. 补焦完成后回到第 2 步继续测光谱。

核心逻辑拆分到新文件 `Utils/AutoZoom/spectrum_autofocus_loop.py`，主 GUI 文件只做最小集成（UI 控件 + 启动/停止线程 + 日志转发）。新增单元/集成测试 `test_spectrum_autofocus_loop.py`，覆盖 ROI 解析、循环状态机、被动补焦触发与停止、FocusScore 曲线保存等关键路径。

## 2. 现状分析

- 主文件：`Utils/AutoZoom/0_measurement_workflow_real_virtual_same_detection_6.25.py`
  - 使用 tkinter，`MeasurementWorkflowGUI._build_ui()` 当前有 8 个左侧控制区。
  - `MeasurementWorkflow.run_one_cycle()` 负责单轮完整测量，但**未集成任何 Focus/自动对焦逻辑**。
  - 已有 `light_on()` / `light_off()` / `request_labview_spectrum()` / `start_save_cycle_result_async()` 等接口，可被新模块调用。
  - 已有 `_capture_current_rule_ab_frame()` 可用于截取当前固定屏幕区域。
- Focus 模块：`Utils/AutoZoom/Focus/`
  - `config.py::AutofocusConfig` 已包含被动补焦全部参数。
  - `controller.py::AutofocusController.check_and_autofocus()` 实现单轮“截图→评分→判断→补焦”。
  - `scorer.py::FocusScorer.build_reference()` 只能从实时采集建立参考，**不支持以单张已有图片建立参考**。
  - `simulator.py::create_demo_environment()` 提供无硬件模拟环境，便于测试。
- autofocus_qt_ui：`Utils/AutoZoom/autofocus_qt_ui/worker.py` 已实现被动模式状态机，但基于 PyQt 信号，不能直接复用于 tkinter；其逻辑可作为参考。

## 3. 拟修改内容

### 3.1 新增文件：`Utils/AutoZoom/spectrum_autofocus_loop.py`

封装 `SpectrumAutofocusLoop` 类，职责单一、与 tkinter 解耦：

- `__init__(self, workflow, cfg, output_dir, on_log=None, should_stop=None)`
  - `workflow`: `MeasurementWorkflow` 实例，用于调用照明、光谱、保存接口。
  - `cfg`: `AutofocusConfig` 实例。
  - `output_dir`: 补焦输出目录（默认 `"focus_output"`）。
  - `on_log`: 日志回调，接收字符串。
  - `should_stop`: 返回 `bool` 的可调用对象，用于外部停止。

- `select_focus_roi_interactively() -> Tuple[int, int, int, int]`
  - 调用 `workflow._capture_current_rule_ab_frame()` 获取当前全屏截图。
  - 使用 OpenCV 窗口显示，允许鼠标拖拽矩形框选 ROI。
  - 返回 ROI `(x, y, w, h)`，并同步写入 `cfg.focus_roi` 与 `cfg.capture_area`。
  - 快捷键：`Enter/N` 确认，`R` 重置，`ESC/Q` 取消。

- `build_reference_from_single_capture() -> Dict[str, Any]`
  - 调用 `metrics_calc.capture_live()` 一次，取其 `roi_metrics` 作为参考。
  - 直接写入 `scorer.focus_reference`，避免 `build_reference()` 的多次采集。
  - 保存参考截图到 `output_dir/reference/`。

- `measure_spectrum_once(cycle_index: int) -> Dict[str, Any]`
  - `workflow.light_off()`
  - `result = workflow.request_labview_spectrum(cycle_index)`
  - `workflow.light_on()`
  - 触发 `workflow.start_save_cycle_result_async(...)` 保存本轮数据。
  - 返回 `result`。

- `run_passive_autofocus_detection(cycle_index: int) -> Dict[str, Any]`
  - 配置 `cfg` 为被动模式：
    - `autofocus_focus_trigger_ratio = 0.95`
    - `autofocus_stop_ratio = 0.95`
    - `autofocus_focus_trigger_count = 3`
    - `z_search_strategy = "hill_climb"`
    - `z_axis = 1`
    - `z_picomotor_conn = 0`
    - `z_picomotor_backend = "auto"`
    - `autofocus_passive_mode = True`
    - `autofocus_passive_max_attempts = 10`
    - `autofocus_passive_consecutive_good = 5`
    - `interval = 1`（秒）
  - 实例化 `FocusMetricsCalculator`、`FocusScorer`、`ZAxisController`、`AutofocusController`。
  - 先执行一次 `check_and_autofocus()` 获取当前 FocusScore：
    - 若 `score >= trigger_ratio`：记录“已达标，跳过补焦”，直接返回。
    - 若 `score < trigger_ratio`：进入被动循环，模仿 `AutofocusWorker._run_loop()` 逻辑，但改为 tkinter 线程安全的直接调用（不依赖 Qt 信号）。
  - 循环停止条件：
    - 连续 `autofocus_passive_consecutive_good` 轮 FocusScore > trigger_ratio；
    - 或 `should_stop()` 返回 `True`；
    - 或补焦过程中发生不可恢复异常。
  - 保存 FocusScore 历史到 `output_dir/focus_score_history_cycle_{cycle_index}.csv`，并绘制曲线到 `output_dir/focus_score_curve_cycle_{cycle_index}.png`。

- `run(max_cycles: int = 0)`
  - 主循环：
    1. `select_focus_roi_interactively()`（若尚未选择）。
    2. `build_reference_from_single_capture()`（若尚未建立）。
    3. `cycle_index` 递增。
    4. `measure_spectrum_once(cycle_index)`。
    5. 若 `max_cycles > 0` 且 `cycle_index >= max_cycles`，停止；否则等待 2 min。
    6. `run_passive_autofocus_detection(cycle_index)`。
    7. 回到 3。
  - 每轮检查 `should_stop()`，为 `True` 时优雅退出。

### 3.2 修改主 GUI 文件：`Utils/AutoZoom/0_measurement_workflow_real_virtual_same_detection_6.25.py`

#### 3.2.1 顶部导入

在文件顶部新增：

```python
import sys
from pathlib import Path

AUTOZOOM_ROOT = Path(__file__).resolve().parent
if str(AUTOZOOM_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOZOOM_ROOT))

from Focus.config import AutofocusConfig
from spectrum_autofocus_loop import SpectrumAutofocusLoop
```

#### 3.2.2 `MeasurementWorkflowGUI.__init__` 新增状态

```python
self.spectrum_autofocus_loop: Optional[SpectrumAutofocusLoop] = None
self.spectrum_autofocus_thread: Optional[threading.Thread] = None
self.spectrum_autofocus_stop_requested = False
```

#### 3.2.3 `_build_ui()` 新增第 9 区

在“8. Rule AB / Rule AC 单独测试”区之后新增：

```python
saf_frame = ttk.LabelFrame(left_inner, text="9. 光谱补焦循环", padding=10, style="Panel.TLabelframe")
saf_frame.pack(fill=tk.X, padx=4, pady=(0, 8))

# 控件：
# - ROI 输入框 + “选择ROI”按钮
# - “开始光谱补焦循环”按钮（Primary）
# - “停止光谱补焦循环”按钮（Danger）
# - 状态标签
```

具体变量：
- `self.saf_roi_var`: `tk.StringVar`，默认值 `"0,0,300,300"`。
- `self.saf_output_dir_var`: `tk.StringVar`，默认值 `"focus_output"`。
- `self.saf_status_var`: `tk.StringVar`，默认值 `"光谱补焦循环：未启动"`。
- `self.saf_max_cycles_var`: `tk.IntVar`，默认值 `0`（0 表示无限循环）。

按钮命令：
- `self.select_saf_roi_thread`
- `self.start_spectrum_autofocus_loop_thread`
- `self.stop_spectrum_autofocus_loop`

#### 3.2.4 新增方法

- `select_saf_roi_thread()` / `select_saf_roi()`
  - 创建临时 `SpectrumAutofocusLoop`，调用 `select_focus_roi_interactively()`，将返回 ROI 写回 `self.saf_roi_var`。

- `start_spectrum_autofocus_loop_thread()` / `start_spectrum_autofocus_loop()`
  - 检查是否已在运行。
  - 根据当前 GUI 的 `capture_area_var` 和 `saf_roi_var` 构造 `AutofocusConfig`。
  - 创建 `SpectrumAutofocusLoop` 实例，传入：
    - `workflow = self.ensure_workflow()`
    - `cfg`：上述 `AutofocusConfig`
    - `output_dir = self.saf_output_dir_var.get()`
    - `on_log = self.log`
    - `should_stop = lambda: self.spectrum_autofocus_stop_requested`
  - 在后台线程运行 `loop.run(max_cycles=self.saf_max_cycles_var.get())`。
  - 线程结束时在 GUI 中更新状态。

- `stop_spectrum_autofocus_loop()`
  - 设置 `self.spectrum_autofocus_stop_requested = True`。
  - 更新状态标签。

- `_parse_focus_roi(text: str) -> Tuple[int, int, int, int]`
  - 解析 `saf_roi_var` 字符串为整数元组，异常时返回默认 `(0,0,300,300)` 并记录日志。

#### 3.2.5 主窗口关闭/停止联动

在 `request_stop()` 中追加：

```python
self.spectrum_autofocus_stop_requested = True
```

确保点击全局“停止测量”时，光谱补焦循环也立即退出。

### 3.3 修改 `Utils/AutoZoom/Focus/scorer.py`

新增方法 `build_reference_from_image(self, image_rgb: np.ndarray, output_root: Optional[Path] = None, on_log: Optional[Callable] = None) -> Dict[str, Any]`：

- 对输入的整张 RGB 图计算 `roi_metrics`（使用 `metrics_calc.compute_for_image` 对 ROI 区域）。
- 将结果直接设为 `focus_reference`，`focus_reference_ready = True`。
- 可选保存参考截图与 CSV。

这样 `SpectrumAutofocusLoop.build_reference_from_single_capture()` 可以直接调用，避免重复采集。

### 3.4 新增测试文件：`Utils/AutoZoom/test_spectrum_autofocus_loop.py`

测试目标：不依赖真实硬件，使用 `FocusSimulator` 和 Mock 覆盖核心逻辑。

测试用例：

1. `test_parse_focus_roi_valid()` / `test_parse_focus_roi_invalid()`
   - 验证 `_parse_focus_roi` 对合法/非法字符串的处理。

2. `test_roi_clamping()`
   - 验证 ROI 超出截图边界时被正确裁剪。

3. `test_build_reference_from_single_capture()`
   - 使用模拟图像验证参考建立后 `scorer.focus_reference_ready == True`。

4. `test_passive_autofocus_skips_when_score_above_trigger()`
   - 构造当前分数 `>= 0.95` 的模拟场景，验证 `run_passive_autofocus_detection()` 不移动 Z 轴并立即返回。

5. `test_passive_autofocus_triggers_recovery()`
   - 使用 `create_demo_environment()` 让 Z 轴从离焦位置开始，验证被动模式最终把分数拉回 `>= 0.95`。

6. `test_stop_requested_interrupts_loop()`
   - 验证 `should_stop()` 返回 `True` 时，循环立即优雅退出。

7. `test_measure_spectrum_once_light_sequence()`
   - Mock `workflow.light_on/off`、`request_labview_spectrum`、`start_save_cycle_result_async`，验证调用顺序为 OFF → spectrum → ON → save。

8. `test_focus_score_curve_saved()`
   - 验证被动补焦结束后，CSV 与 PNG 曲线文件被创建。

## 4. 关键设计决策

- **独立模块**：将光谱补焦循环核心拆出到 `spectrum_autofocus_loop.py`，主 GUI 文件只做最小集成，降低 854KB 大文件变更风险。
- **直接使用 Focus 模块**：不引入 PyQt UI 的 `AutofocusWorker`，避免 Qt 依赖与 tkinter 线程冲突；在 `SpectrumAutofocusLoop` 中显式调用 `AutofocusController`。
- **ROI 与参考**：用户手动框选的 ROI 同时作为操作区域和参考图。参考建立时只采集一次，避免多次截图期间样品/照明变化。
- **被动模式参数**：按需求硬编码为类常量，不在 GUI 中暴露（减少 UI 复杂度，保证可重复性）。若未来需要可再开放。
- **Z 轴连接**：`AutofocusController.run_closed_loop()` 内部会调用 `z_axis.connect()`；virtual 模式下使用 `FocusSimulator` 替换 Z 轴与图像源，无需真实硬件即可测试。
- **等待 2 min**：在主循环中通过 `time.sleep(120)` 实现，期间定期检查 `should_stop()`，保证停止按钮响应。

## 5. 实施步骤（Plan Checklist）

- [x] 1. 修改 `Focus/scorer.py`，新增 `build_reference_from_image()`。
- [x] 2. 创建 `Utils/AutoZoom/spectrum_autofocus_loop.py`，实现：
  - [x] `SpectrumAutofocusLoop.__init__()`
  - [x] `select_focus_roi_interactively()`
  - [x] `build_reference_from_single_capture()`
  - [x] `measure_spectrum_once()`
  - [x] `run_passive_autofocus_detection()`
  - [x] `run()`
  - [x] 辅助方法：`_log()`、`_draw_focus_score_curve()`、`_save_focus_score_history()`。
- [x] 3. 修改主 GUI 文件：
  - [x] 顶部导入 `AutofocusConfig` 与 `SpectrumAutofocusLoop`。
  - [x] `__init__` 中新增状态变量。
  - [x] `_build_ui()` 中新增“9. 光谱补焦循环”区。
  - [x] 新增 `select_saf_roi_thread/select_saf_roi()`。
  - [x] 新增 `start_spectrum_autofocus_loop_thread/start_spectrum_autofocus_loop()`。
  - [x] 新增 `stop_spectrum_autofocus_loop()`。
  - [x] 新增 `_parse_focus_roi()`。
  - [x] `request_stop()` 中追加停止光谱补焦循环。
- [x] 4. 创建 `Utils/AutoZoom/test_spectrum_autofocus_loop.py`，实现上述 8 个测试用例。
- [x] 5. 运行测试并修复问题：
  - [x] `python Utils/AutoZoom/test_spectrum_autofocus_loop.py`
  - [x] `python Utils/AutoZoom/test_passive_mode.py`（确保不破坏原有被动模式）
- [x] 6. 验证 GUI 无语法错误：
  - [x] `python -m py_compile "Utils/AutoZoom/0_measurement_workflow_real_virtual_same_detection_6.25.py"`
- [x] 7. 更新相关 README：
  - [x] 在 `Utils/AutoZoom/README.md` 中新增“光谱补焦循环”使用说明。
- [x] 8. 汇总变更，向用户报告。

## 6. 验证与测试

- **单元测试**：`test_spectrum_autofocus_loop.py` 全部通过。
- **回归测试**：`test_passive_mode.py` 全部通过。
- **静态检查**：主 GUI 文件 `py_compile` 通过。
- **手动验证建议**（需要真实硬件时执行）：
  1. 启动 GUI，切换到 virtual 模式。
  2. 点击“选择ROI”，在弹窗中框选区域并确认。
  3. 点击“开始光谱补焦循环”，观察日志：
     - 照明 OFF → LabVIEW/虚拟光谱采集 → 照明 ON → 保存 → 等待 2 min → 补焦检测 → 循环。
  4. 点击“停止光谱补焦循环”，确认当前步骤结束后退出。
  5. 检查 `focus_output/` 目录下是否生成参考图、FocusScore 曲线 CSV/PNG。

## 7. 风险与回退

- **风险 1**：主文件体积大，编辑时容易误改。回退：改动前不修改无关区域，使用小范围 `Edit` 替换。
- **风险 2**：屏幕截图在主线程与后台线程间调用可能触发 tkinter 跨线程问题。回退：`select_focus_roi_interactively()` 在主线程运行；`build_reference_from_single_capture()` 和 `measure_spectrum_once()` 在后台线程运行，但只调用 `MeasurementWorkflow` 已存在的线程安全方法。
- **风险 3**：Z 轴真实硬件连接失败。回退：在 virtual 模式下使用 `FocusSimulator`；真实模式失败时记录日志并停止循环，不崩溃 GUI。
