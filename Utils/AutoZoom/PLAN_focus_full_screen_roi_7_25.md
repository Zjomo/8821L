# 7_25 补焦参考图全屏 ROI 选择与隔离实施计划

## 1. 目标

在 `0_measurement_workflow_real_virtual_same_detection_7_25.py` 的「运行完整循环测量」过程中，调整补焦参考图（Focus Reference）的 ROI 逻辑：

1. **补焦基准图的 ROI 选择应在整个屏幕上进行**，而不是局限在标定/角度检测的 `capture_area` 区域内；
2. **补焦画面 ROI 范围独立参考**，修改 `saf_capture_area` / `saf_focus_roi` 不影响「标定 ABC」、「角度检测」等画面作用区域（`capture_area` / `focus_roi`）。

## 2. 涉及文件

| 文件 | 变更内容 |
|------|----------|
| `0_measurement_workflow_real_virtual_same_detection_7_25.py` | 新增 `_capture_full_screen_frame`；修改 `_select_focus_roi_interactively` 使用全屏底图；`_parse_focus_roi_text` 支持 tuple/list 输入；更新相关文档字符串 |
| `test_focus_full_screen_roi_7_25.py` | 新增 5 项测试 |
| `PLAN_focus_full_screen_roi_7_25.md` | 本计划文档 |

## 3. 实施步骤

### 3.1 新增全屏截图方法

在 `MeasurementWorkflow` 中新增：

```python
def _capture_full_screen_frame(self) -> Optional[np.ndarray]:
    """截取整个屏幕画面（RGB），用于补焦基准图 ROI 交互式选择。"""
    try:
        if FixedRegionScreenCapture is not None:
            frame = pyautogui.screenshot(region=None)
            image_rgb = np.array(frame)
            return image_rgb
    except Exception as e:
        self.log(f"[聚焦] 全屏截图失败（pyautogui）：{e}")

    try:
        from PIL import ImageGrab
        img = ImageGrab.grab().convert("RGB")
        return np.array(img)
    except Exception as e:
        self.log(f"[聚焦] 全屏截图失败（PIL）：{e}")

    return None
```

### 3.2 修改交互式 ROI 选择

`_select_focus_roi_interactively` 关键改动：

- 底图从 `_capture_current_rule_ab_frame()`（基于 `capture_area`）改为 `_capture_full_screen_frame()`；
- 窗口标题改为 `"Select Focus ROI (Full Screen)"`；
- ROI 坐标经反缩放后直接作为相对于全屏的坐标；
- 选择成功后：
  - `self.cfg.saf_capture_area = (0, 0, screen_w, screen_h)`
  - `self.cfg.saf_focus_roi = (roi_x, roi_y, roi_w, roi_h)`
- 日志明确说明 `capture_area / focus_roi 保持不变`。

### 3.3 增强 ROI 解析兼容性

`_parse_focus_roi_text` 支持字符串和四元组：

```python
@staticmethod
def _parse_focus_roi_text(value: Any) -> Tuple[int, int, int, int]:
    try:
        if isinstance(value, (tuple, list)) and len(value) == 4:
            return tuple(int(v) for v in value)
        parts = [int(v.strip().strip("()[]")) for v in str(value).split(",")]
        if len(parts) == 4:
            return tuple(parts)
    except Exception:
        pass
    return (0, 0, 300, 300)
```

### 3.4 保持区域隔离

- `MeasurementConfig.__post_init__` 已说明 `saf_capture_area` / `saf_focus_roi` 与 `capture_area` / `focus_roi` 独立；
- `_make_focus_config` 仅读取 `cfg.saf_capture_area` / `cfg.saf_focus_roi`；
- GUI 中「光谱补焦循环」面板的 `saf_capture_area_var` / `saf_roi_var` 只影响补焦配置；
- `_select_focus_roi_interactively` 不修改 `cfg.capture_area` 或 `cfg.focus_roi`。

## 4. 测试计划

| 测试项 | 方法 |
|--------|------|
| 全屏截图返回全屏尺寸 | `test_capture_full_screen_frame_returns_full_screen_size` |
| ROI 选择使用全屏底图 | `test_select_focus_roi_interactively_uses_full_screen` |
| ROI 选择窗口置顶、移动、彻底销毁 | `test_select_focus_roi_interactively_uses_full_screen` 中断言 |
| 过小选区被拒绝 | `test_select_focus_roi_rejects_tiny_selection` |
| stop_requested 可中断选择 | `test_select_focus_roi_respects_stop_request` |
| `_make_focus_config` 使用 saf 区域 | `test_make_focus_config_uses_saf_areas` |
| 补焦区域与标定区域隔离 | `test_focus_area_isolation_from_calibration` |
| 全屏 ROI 下建立参考并保存到文件 | `test_capture_focus_reference_with_full_screen_roi` |
| 每次运行重新选择 ROI | `test_begin_new_run_session_resets_focus_state` |

## 5. 测试结果

```text
11 passed in 11.42s
```

## 6. 强制 ROI 选择阻塞

### 6.1 需求

在「运行完整循环测量」过程中，只有用户完成「补焦基准图」的 ROI 选择后，才能进入下一步；否则流程阻塞并反复提示，防止未选 ROI 导致卡死。

### 6.2 实现

在 `MeasurementWorkflow` 中新增标志位 `self._focus_roi_selected: bool = False`：

- `_select_focus_roi_interactively` 选择成功后将其置为 `True`；
- `capture_focus_reference` 中使用 `while not self._focus_roi_selected` 循环强制要求选择：
  - 若用户取消/选择失败，调用 `_show_roi_selection_required_dialog` 阻塞提示；
  - 用户确认后再次弹出 ROI 选择窗口；
  - 收到 `stop_requested` 时返回 `False`，避免无法退出。
- `run_one_cycle` 中检查 `capture_focus_reference` 返回值，失败时设置 `stop_requested=True` 并返回 `False`，终止本轮及后续循环。

### 6.3 新增测试项

| 测试项 | 方法 |
|--------|------|
| 用户取消一次后完成选择仍能建立参考 | `test_capture_focus_reference_requires_roi_selection` |
| 停止请求下可退出阻塞循环 | `test_capture_focus_reference_returns_false_when_stopped` |
| 参考建立失败时 run_one_cycle 终止测量 | `test_run_one_cycle_stops_when_focus_reference_fails` |

## 7. ROI 选择窗口残留/卡死修复

### 7.1 需求

用户反馈：在 `_select_focus_roi_interactively` 中选定 ROI 并按 Enter/N 确认后，OpenCV ROI 选择窗口界面框仍然残留在屏幕上，看起来像是"卡住"没有继续。

### 7.2 根因

1. OpenCV 窗口在 `cv2.destroyWindow()` 后需要事件循环（`cv2.waitKey`）才能真正从屏幕移除；原代码缺少 `waitKey`，导致窗口残留。
2. ROI 选择窗口没有置顶，可能被 tkinter 提示框或其他窗口遮挡，用户按键无法被窗口接收。
3. 循环内未检查 `self.stop_requested`，外部停止按钮无法中断 ROI 选择。
4. 用户单击（几乎未拖动）会形成 1×1 的极小选区，可能让后续流程产生异常 ROI。

### 7.3 实现

在 `_select_focus_roi_interactively` 中：

- 创建窗口后尝试 `cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)` 置顶，并 `cv2.moveWindow(window_name, 0, 0)`；
- 循环顶部检查 `self.stop_requested`，为真则直接返回 `False`；
- 确认时增加选区大小检查：宽高均小于 2px 视为无效，要求重新拖拽；
- 关闭窗口时采用三步清理：
  ```python
  cv2.imshow(window_name, redraw())
  cv2.waitKey(100)
  cv2.destroyWindow(window_name)
  cv2.waitKey(300)
  cv2.destroyAllWindows()
  cv2.waitKey(100)
  ```

### 7.4 新增/更新测试项

| 测试项 | 方法 |
|--------|------|
| 成功选择时窗口置顶、移动、彻底销毁 | 在 `test_select_focus_roi_interactively_uses_full_screen` 中断言 |
| 过小选区被拒绝 | `test_select_focus_roi_rejects_tiny_selection` |
| stop_requested 可中断选择 | `test_select_focus_roi_respects_stop_request` |

### 7.5 不再弹出 "Focus Reference Baseline Image" 预览窗口

#### 现象

用户反馈：第一个 ROI 选择窗口关闭后，第二个窗口 "Focus Reference Baseline Image" 弹出并显示参考图，但随后标题栏变为"未响应"，流程卡住。

#### 根因

`capture_focus_reference` 中调用 `cv2.imshow()` 显示参考图后只执行了一次 `cv2.waitKey(1)`。OpenCV 窗口需要主线程持续调用 `cv2.waitKey()` 来处理事件循环；否则即使图像已经显示，窗口也会因无事件响应而显示"未响应"。

#### 决策

用户确认不需要该实时预览窗口。ROI 选择窗口已经足以确认补焦区域，参考图也会保存到文件供后续查看。因此直接移除 `"Focus Reference Baseline Image"` 预览窗口，从根本上避免窗口未响应问题。

#### 实现

在 `capture_focus_reference` 中：

- 删除 `cv2.imshow` 显示参考图的相关代码；
- 删除 `cv2.waitKey` 事件循环和 `cv2.destroyWindow` 关闭逻辑；
- 保留 `self._focus_scorer.build_reference_from_image(...)` 保存参考图到 `focus_reference/` 目录。

```python
self.log("[聚焦参考] 参考图已建立并保存，不再弹出实时预览窗口")
ref_dir = self.output_root / "focus_reference"
ref_dir.mkdir(parents=True, exist_ok=True)
ref = self._focus_scorer.build_reference_from_image(
    image_rgb,
    focus_roi=focus_roi,
    save_dir=str(ref_dir),
    tag=f"cycle_{cycle_index:04d}",
)
```

## 8. 风险与回退

- 全屏截图依赖 `pyautogui.screenshot(region=None)` 或 `PIL.ImageGrab.grab()`，在多显示器环境下可能只截取主屏；
- OpenCV 显示全屏图像时，0.85 缩放仍可能超出小屏显示范围，用户可通过窗口最大化查看；
- 强制阻塞循环在用户未选 ROI 时会反复弹出提示，若需临时绕过，可置 `self._focus_roi_selected = True`（仅用于调试）；
- 若需恢复旧行为（在 capture_area 内选择 ROI），只需把 `_capture_full_screen_frame()` 改回 `_capture_current_rule_ab_frame()`，并把 `saf_capture_area` 同步为 `capture_area` 尺寸。

## 9. 每次运行重新选择补焦基准图 ROI

### 9.1 需求

每次点击"运行完整循环测量"后，都要重新进行一次"补焦基准图"的 ROI 选取，并为后续补焦流程更新参考图。

### 9.2 根因

`_focus_roi_selected`、`_focus_reference_ready`、`_focus_reference_image` 这三个状态在 `MeasurementWorkflow` 实例生命周期内只会被设置一次。完成一次测量后再次点击"运行完整循环测量"，这些状态仍为 `True`，导致 `capture_focus_reference()` 跳过 ROI 选择和参考更新，直接使用上一轮旧数据。

### 9.3 实现

在 `begin_new_run_session()`（每次点击"运行完整循环测量"时调用）中增加重置逻辑：

```python
self._focus_roi_selected = False
self._focus_reference_ready = False
self._focus_reference_image = None
self.log(
    "[聚焦参考] 新运行会话开始，已重置补焦 ROI 与参考图状态，"
    "将在 Step 1.5 重新选择补焦基准图 ROI"
)
```

这样每次新运行都会强制 `capture_focus_reference()` 重新进入 ROI 选择流程，并基于新选定的 ROI 重新建立参考图。

### 9.4 新增测试项

| 测试项 | 方法 |
|--------|------|
| begin_new_run_session 重置聚焦状态 | `test_begin_new_run_session_resets_focus_state` |
