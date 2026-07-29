# Plan：手动划取 ROI 排除区域，消除"其他图案"对 A/B/C 识别的干扰

## Context

在显微镜标定期间，视野中可能引入额外图案（如左下角的表情包）。SAM2 基于点提示的分割会把这些图案误识别为 A（黄色方块）或 B（长条）的一部分，导致：
- A mask 混入干扰图案；
- B mask 错误覆盖 A 区域；
- 后续运动控制基于错误 mask 产生错误动作。

用户提出的方案是：**手动划取 ROI 排除区域，将干扰图案永久排除在 A/B/C 分割之外**。经确认，本计划采用：
- **多边形 ROI**（非矩形），支持任意形状；
- **持续生效**：首帧标定 + 后续帧 A/B 跟踪 + SAM2 video tracking 均排除 ROI；
- **交互方式**：在现有 OpenCV 点选窗口中按 `E` 切换 ROI 模式，左键逐点绘制多边形，右键/Enter 闭合，X 删除最后一个 ROI，C 清空，E 退出。

## Recommended Approach

### Phase 1：新增纯工具模块 `logic/roi_exclusion.py`

新增独立工具模块，负责 ROI 多边形与 mask 的转换及应用，不与 SAM2/GUI/硬件耦合，便于单元测试。

- `RoiPolygon = List[Tuple[float, float]]`
- `polygons_to_mask(image_shape_hw, polygons) -> np.ndarray[bool]`：多个多边形取并集生成排除 mask。
- `apply_roi_exclusion(mask, exclude_mask, inplace=False) -> np.ndarray[bool]`：对 mask 执行 `mask &= ~exclude_mask`。
- `draw_roi_overlay(image_bgr, polygons, color=(0,0,255), alpha=0.35) -> np.ndarray`：在图像上绘制半透明 ROI。
- `normalize_roi_polygons(raw) -> List[RoiPolygon]`：兼容 list/tuple/dict 等输入。

### Phase 2：扩展数据结构

#### `logic/actual_nano_boundary_following_sam2VideoAB_xyStep_detailed_pathfix_strict_c.py` 的 `RuntimeConfig`

新增字段：

```python
exclude_roi_polygons: List[List[List[float]]] = field(default_factory=list)
```

表示多个多边形 ROI，每个 ROI 是闭合顶点序列。默认空列表，完全向后兼容。

#### `0_measurement_workflow_real_virtual_same_detection_7_25.py` 的 `CalibrationState`

新增同名字段：

```python
exclude_roi_polygons: List[List[List[float]]] = field(default_factory=list)
```

使 ROI 能随完整标定 JSON 持久化。在 `from_dict` 中增加归一化，兼容旧标定文件（缺失字段默认为空）。

### Phase 3：改造 SAM2 分割器 `ABCSegmenter`

在 `logic/actual_nano_boundary_following_sam2VideoAB_xyStep_detailed_pathfix_strict_c.py` 中：

1. `__init__` 增加 `exclude_roi_mask: Optional[np.ndarray] = None` 参数或 `set_exclude_roi_mask()` 方法。
2. 所有返回 mask 的函数在返回前调用 `_apply_exclude_roi(mask)`：
   - `_predict_mask_by_points`（首帧 A/B/C 分割）
   - `_predict_a_mask_by_box_and_points_temporal`（后续帧 A）
   - 对应的 B 后续帧分割函数
   - SAM2 video tracking 输出转换后（`_infer_ab_by_sam2_video_two_frame`）
   - `initialize_with_points` / `initialize_ab_with_static_c_map` 生成的 A/B/C mask
3. 对于后续帧多候选 mask 评分，**先对每个候选应用 ROI 排除，再计算时序评分**，避免选中被 ROI 挖空前得分高、挖空后面积剧变的错误候选。

### Phase 4：改造 OpenCV 交互点选窗口

需要在以下 4 个交互函数中增加 ROI 多边形绘制能力：

1. `select_a_b_c_points_interactively`（logic 文件，A/B/C 首帧）
2. `select_a_b_points_interactively`（logic 文件，复用 static C 时只点 A/B）
3. `_select_object_points_interactively`（workflow 文件，RuleAB-A 标定）
4. `_select_c_points_interactively`（workflow 文件，全局 C 标定）

#### 交互细节

- 默认模式：保持现有行为（左键正点 / 右键负点 / R 重置 / Z 撤销 / Backspace 上一个目标 / Enter 完成 / Esc 取消）。
- 按 `E` 进入/退出 ROI 模式。
- ROI 模式下：
  - **左键点击**：添加当前 ROI 多边形顶点。
  - **右键点击 / Enter / N**：闭合当前多边形，开始下一个 ROI。
  - **X**：删除最后一个已闭合的多边形。
  - **C**：清空所有 ROI 多边形。
  - **E**：退出 ROI 模式，回到点选模式。
  - **Esc**：取消整个标定。
- 可视化：
  - 正在绘制的多边形用红色实线连接已有点，最后一个点与鼠标位置用虚线跟随。
  - 已闭合 ROI 用红色半透明填充，并标注 `ROI 1`、`ROI 2` 等。
  - 顶部文字增加一行：`Mode: ROI | ROIs: N | Left=vertex, Right/Enter=close, X=del, C=clear, E=exit`。

#### 返回值变更

- `select_a_b_c_points_interactively` 返回 `ABCInteractiveResult`（dataclass，含 a/b/c prompt 和 exclude_roi_polygons）。
- `select_a_b_points_interactively` 返回 `ABInteractiveResult`（含 a/b prompt 和 exclude_roi_polygons）。
- `_select_object_points_interactively` 返回 `Tuple[pos, neg, exclude_roi_polygons]`。
- `_select_c_points_interactively` 返回 `Tuple[pos, neg, exclude_roi_polygons]`。

所有调用方必须同步解包更新。

### Phase 5：改造标定流程

#### `0_measurement_workflow_real_virtual_same_detection_7_25.py`

1. `calibrate_rule_ab_a`：
   - 调用 `_select_object_points_interactively` 获取 ROI。
   - 将 ROI 写入 `CalibrationState.exclude_roi_polygons`。
   - 保存标定 JSON。

2. `calibrate_global_c`：
   - 调用 `_select_c_points_interactively` 获取 ROI。
   - 将 ROI 写入 `CalibrationState.exclude_roi_polygons`。
   - 在生成 static C map 目录时额外保存：
     - `exclude_roi_polygons.json`：ROI 多边形数据；
     - `static_c_roi_overlay.png`：C mask + ROI 叠加图，便于人工核对。

3. `_predict_c_mask_by_sam2_points`：
   - SAM2 得到 C mask 后，用标定 ROI 过滤一次。

4. `_call_initialize_abc_with_loaded_calibration` / `ensure_rule_ab_follower`：
   - 将 ROI 像点一样注入 follower / cfg / segmenter。

5. RuleAB-C 独立运行入口：
   - 构造 `RuleABRuntimeConfig` 时传入 `exclude_roi_polygons`。

### Phase 6：在 `ActualNanoBoundaryFollower` 中落地

在 `logic/actual_nano_boundary_following_sam2VideoAB_xyStep_detailed_pathfix_strict_c.py` 中：

1. `__init__` 构造完 `ABCSegmenter` 后，根据 `cfg.exclude_roi_polygons` 和当前图像尺寸生成 mask 并设置到 segmenter。
2. `initialize_abc_with_first_frame`：交互点选返回 ROI 后，同步设置到 segmenter 与 cfg。
3. `initialize_abc_with_calibration`：从 `calibration_state` 读取 ROI 并设置；同时把 ROI 写回 `self.cfg` 供后续帧使用。

### Phase 7：单元测试

新增测试文件 `test_roi_exclusion_7_25.py`，覆盖：

1. 多边形 ROI 转 mask 正确性（凸/凹多边形）。
2. ROI 排除基本功能（mask 被正确挖空）。
3. 空 ROI 不改变原 mask 行为。
4. 多个 ROI 多边形取并集排除。
5. 越界 ROI 顶点不抛异常且正确裁剪。
6. `CalibrationState` ROI 字段序列化 round-trip。
7. `RuntimeConfig` 默认空 ROI。
8. `ABCSegmenter` 对 mock predictor 返回的 mask 正确应用 ROI。
9. 后续帧多候选 mask 评分前已应用 ROI（错误候选被排除）。
10. 显示坐标到原始坐标转换（scale 还原）。

测试风格沿用现有项目惯例：动态导入模块，使用 `if __name__ == "__main__"` 直接运行。

## Critical Files to Modify

- `f:\0_My_jupyter\2_Optics\8821L\Utils\AutoZoom\logic\roi_exclusion.py`（新增）
- `f:\0_My_jupyter\2_Optics\8821L\Utils\AutoZoom\logic\actual_nano_boundary_following_sam2VideoAB_xyStep_detailed_pathfix_strict_c.py`
- `f:\0_My_jupyter\2_Optics\8821L\Utils\AutoZoom\logic\actual_nano_boundary_following_sam2VideoAB_xyStep_detailed_pathfix.py`（保持与 strict_c 一致，可选）
- `f:\0_My_jupyter\2_Optics\8821L\Utils\AutoZoom\0_measurement_workflow_real_virtual_same_detection_7_25.py`
- `f:\0_My_jupyter\2_Optics\8821L\Utils\AutoZoom\test_roi_exclusion_7_25.py`（新增）

## Verification Plan

1. 单元测试：运行 `python test_roi_exclusion_7_25.py`，所有断言通过。
2. 语法检查：对修改的 7_25 和 logic 文件运行 `python -m py_compile`。
3. 端到端验证（需真实 GUI 环境）：
   - 启动完整测量 GUI；
   - 在标定 A 或标定 C 时点选窗口按 `E` 进入 ROI 模式，绘制覆盖左下角干扰图案的多边形；
   - 完成标定后检查生成的 `static_c_map` 目录是否包含 `exclude_roi_polygons.json` 和 `static_c_roi_overlay.png`；
   - 运行完整测量，观察 A/B/C mask 是否不再包含干扰图案。
4. 向后兼容：加载旧的标定 JSON（无 ROI 字段），确认流程正常运行。

## Risks & Mitigations

| 风险点 | 影响 | 规避措施 |
|--------|------|----------|
| 返回类型变更导致旧调用方报错 | 运行时解包失败 | 全文搜索 4 个交互函数调用点并统一更新；使用 dataclass 明确返回值字段 |
| ROI 误覆盖 A/B/C 真实区域 | mask 为空、初始化失败 | ROI 用红色半透明高亮；确认窗口显示 ROI 叠加；日志输出应用 ROI 后的 A/B/C 面积 |
| 候选 mask 先评分再排除 | 选中错误候选 | 后续帧对每个候选先应用 ROI 再评分 |
| static C 被 ROI 永久挖空 | 后续修改 ROI 无法恢复旧 C | 这是符合预期的标定行为；修改 ROI 后提示重新标定 C；保存 ROI 叠加图备查 |
| capture_area 变化导致 ROI 错位 | ROI mask 与当前图像不匹配 | 用当前图像实际 shape 生成 mask；加载旧标定时若 capture_area 不一致则日志警告 |
| SAM2 video tracking 输出未过滤 ROI | video predictor 输出包含干扰区 | 在 video tracking 输出转换后统一应用 ROI |
| 多边形绘制与点选事件冲突 | 同一窗口下鼠标语义混淆 | 显式 `mode` 标志：ROI 模式忽略左右键点选，点选模式忽略多边形绘制 |

## User-Confirmed Decisions

- ROI 形状：**多边形**（左键逐点绘制，右键/Enter 闭合）。
- ROI 生效范围：**首帧标定 + 后续帧跟踪 + SAM2 video tracking 全程生效**。
- 交互按键：**E 切换 ROI/点选模式，X 删除最后一个 ROI，C 清空所有 ROI**。
