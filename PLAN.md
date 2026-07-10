## 4轴双镜闭环替代现有5轴(Z扫描)方案

### Summary
基于你选定的方向（**双段解耦 + 双探测器位角分离 + 直接改代码并做模拟/UI测试**），将 SpotZoom 从当前 `P1→Z上移(P2)→Z下移(P3)` 的 5轴时序闭环，扩展为**纯4轴实时闭环**：

- **Stage 1（镜1_x/y）**：主控“位置误差”（Detector 1）
- **Stage 2（镜2_x/y）**：主控“角度误差”（Detector 2，前置透镜做角度-位置映射）
- 去除对 Z 轴扫描的依赖，在 4 轴系统中实现“位置+方向”同时稳定。

> 设计依据来自你给的产品链路及其公开文档：4轴典型配置为“两执行器+两探测器”，且可通过 Detector2 前置透镜实现角度判别；控制上本质是两个控制段（stage1/stage2）协同。  
> 参考：  
> - https://www.surisetech.com/mrc-systems-active-laser-beam-stabilization/  
> - https://www.mrc-systems.de/downloads/en/laser-beam-stabilization/Description_MRC_Setup-configurations_v1_en.pdf  
> - https://mrc-systems.de/downloads/en/laser-beam-stabilization/Manual_MRC-BA-Software_vers2.3.pdf  

### Implementation Changes
1. **运行策略分离（保留兼容）**
- 新增 `alignment_strategy`：
  - `z_scan_legacy`（现有 P1/P2/P3 流程，默认保留）
  - `dual_detector_4axis`（新流程，禁用 Z 动作）
- `SpotZoomController.run()` 拆分为两条策略入口，避免旧逻辑被破坏。
- `z_driver` 在 `dual_detector_4axis` 下变为可选且默认不参与控制。

2. **双探测器采集抽象**
- 新增 `FrameSourcePair`（或等效抽象）：
  - `source_pos`（Detector 1）
  - `source_ang`（Detector 2）
- CLI/UI 增加第二路参数：
  - `--window-title-2`
  - `--frame-source-image-2`
  - `--select-roi-2 / --skip-roi-2`
- 模拟模式支持双图源或单图源双ROI（后者仅测试便捷，不作为真实部署推荐）。

3. **4轴控制律（双段解耦 + 小耦合补偿）**
- 主误差定义：
  - `e_pos = [dx1, dy1]`（Detector 1）
  - `e_ang = [dx2, dy2]`（Detector 2，经光学标定换算角度可选）
- 控制分配：
  - `u1 = Kp1*e_pos + Ki1*∫e_pos + C12*e_ang`
  - `u2 = Kp2*e_ang + Ki2*∫e_ang + C21*e_pos`
- 默认先启用解耦主项（`C12=C21=0`），标定后再打开小耦合补偿。

4. **4轴耦合标定与闭环收敛**
- 新增标定例程：逐轴注入 ±Δ，测 `[dx1,dy1,dx2,dy2]`，估计 4x4 灵敏度矩阵 `J`。
- 生成控制矩阵（伪逆 + 正则化），用于耦合补偿和方向自动判定（替代手工 sign 猜测）。
- 收敛判据改为双探测器同时达标：
  - `||e_pos|| < tol_pos`
  - `||e_ang|| < tol_ang`
  - 连续 `N` 帧成立后判定稳定。

5. **UI 扩展（spotzoom_qt_ui）**
- `Run Modes`：新增策略切换（legacy / 4axis）、Detector2 配置组。
- `Alignment Workspace`：双画面/双状态（位置误差、角度误差、4轴命令向量）。
- `Device Center`：明确 Stage1/Stage2 与 Detector1/Detector2 拓扑绑定。
- `Device Test`：新增测试项
  - 第二路采集连通性
  - 4轴逐轴扰动响应测试
  - 4x4 标定流程测试
  - 4轴闭环 dryrun / simulated 跑通测试
- `Logs & Reports`：增加 `e_pos/e_ang` 轨迹、矩阵 `J` 快照、饱和/限幅统计。

### Test Plan
1. **算法与控制单测**
- 4x4 标定矩阵求解（满秩/欠秩/噪声）正确性与退化保护。
- 控制分配限幅、积分抗饱和、符号自动判定测试。
- 收敛判据（双阈值 + 连续帧）测试。

2. **模拟流程测试（你要求重点）**
- 双模拟源 + 可控扰动（平移扰动、角度扰动、耦合扰动）回归：
  - 仅位置扰动时 Stage1 主动作
  - 仅角度扰动时 Stage2 主动作
  - 耦合扰动时两段协同且稳定收敛
- 与 legacy 流程并行保留，验证旧模式不回归。

3. **UI 测试**
- Qt 离屏启动 + 页面构建（含新增 Detector2 与 4axis 参数）。
- Device Test 新增测试项可执行并回填状态/耗时/错误。
- 运行中状态刷新（双误差、四轴命令、收敛状态）无布局抖动。

4. **端到端验收**
- `--alignment-strategy dual_detector_4axis` 全链路跑通（dryrun/simulated）。
- `--alignment-strategy z_scan_legacy` 继续可用。

### Assumptions & Defaults
- 真实 4轴部署按“两探测器+两执行器”标准结构；Detector2 前置透镜用于角度判别（默认建议焦距 ~200 mm，后续可参数化）。
- 默认不删除 legacy 代码路径，先并行新旧两策略，降低联机风险。
- 第一期仅做 **P/PI 控制 + 耦合矩阵补偿**；不在首版引入 MPC/LQG。
- Z 轴逻辑在 `dual_detector_4axis` 策略中完全旁路，但保留原驱动代码用于兼容其他场景。

---

## AutoZoom Focus：Picomotor 连接失败 & 停止按钮失效修复

### Summary
针对 AutoZoom `autofocus_qt_ui` 点击 **“启动闭环”** 后报 `error connecting to the Picomotor controller`，以及点击 **“停止”** 无响应两个问题，进行闭环补焦链路的健壮性修复。

### Root Cause
1. **Picomotor 连接失败**：`Utils/AutoZoom/Focus/z_axis.py` 中 `ZAxisController.connect()` 硬编码 `Newport.Picomotor8742(conn=0)`，无法适配多控制器或多 USB 索引场景；当实际设备不在索引 0 时直接连接失败。
2. **停止按钮无响应**：补焦搜索策略（`Focus/search.py`）长时间执行电机移动/等待，没有检查用户停止请求；`AutofocusWorker` 仅能在轮询间隔处退出，无法在搜索中途响应停止。

### Implementation Changes
1. **配置扩展（`Utils/AutoZoom/Focus/config.py`）**
   - 新增 `z_picomotor_conn`：控制器索引（默认 0）。
   - 新增 `z_picomotor_backend`：连接后端 `auto/pyusb/serial`。
   - 新增 `z_picomotor_timeout` / `z_picomotor_multiaddr` / `z_picomotor_scan`。
   - 新增 `z_picomotor_velocity` / `z_picomotor_acceleration`（可选覆盖）。

2. **Z 轴控制器修复（`Utils/AutoZoom/Focus/z_axis.py`）**
   - `connect()` 改用 `self.cfg.z_picomotor_conn` 等配置参数构造 `Picomotor8742`。
   - `check_available()` 校验 `conn` 是否超出检测到的设备数量，给出可操作的错误提示。
   - 连接失败时保留原始异常并提示检查 USB/索引/占用。

3. **UI 控件（`Utils/AutoZoom/autofocus_qt_ui/app.py`）**
   - 在“补焦参数”区域增加 **Z 控制器索引** 与 **Z 控制器后端** 输入。
   - `_collect_args()` / `_build_config()` 将新参数传入 `AutofocusConfig`。
   - `_stop_loop()` 等待工作线程最多 2 秒，超时后强制清理 UI 状态。

4. **可中断搜索（`Utils/AutoZoom/Focus/search.py`）**
   - `BaseFocusSearch` 新增 `should_stop` 回调与 `FocusSearchStopped` 异常。
   - 在 `_move`、`_settle`、`_measure`、`_goto` 及四大策略主循环中检查停止请求。
   - `create_search()` 透传 `should_stop`。

5. **Controller 与 Worker 联动（`Utils/AutoZoom/Focus/controller.py`、`Utils/AutoZoom/autofocus_qt_ui/worker.py`）**
   - `AutofocusController` 新增 `should_stop` 回调。
   - `run_closed_loop()` 捕获 `FocusSearchStopped`，返回 `reason="stopped_by_user"`。
   - `AutofocusWorker._run_loop()` 注册 `controller.should_stop = lambda: not self._state.running`。

### Test Plan
1. **配置单测**：验证新增 Picomotor 字段默认值与覆盖值。
2. **Z 轴连接单测**：验证 `_conn_kwargs()` 不再硬编码；`check_available()` 对越界索引返回明确错误。
3. **搜索停止单测**：四大策略在 `should_stop=True` 时立即抛出 `FocusSearchStopped`；`_settle` 长睡眠可被中断。
4. **Controller 停止单测**：`run_closed_loop()` 被中断后返回 `stopped_by_user` 而不崩溃。
5. **Worker 停止单测**：`stop_loop()` 将 `running` 置为 `False`。
6. **回归测试**：重新跑通 `Utils/AutoZoom/test_final_crash_fix.py` 确保主线程截图安全路径未被破坏。

### Test Results
- `python Utils/AutoZoom/test_picomotor_and_stop.py`：全部通过。
- `python Utils/AutoZoom/test_final_crash_fix.py`：全部通过。

### Files Modified
- `Utils/AutoZoom/Focus/config.py`
- `Utils/AutoZoom/Focus/z_axis.py`
- `Utils/AutoZoom/Focus/search.py`
- `Utils/AutoZoom/Focus/controller.py`
- `Utils/AutoZoom/autofocus_qt_ui/app.py`
- `Utils/AutoZoom/autofocus_qt_ui/worker.py`
- `Utils/AutoZoom/test_picomotor_and_stop.py`（新增）
- `PLAN.md`

---

## AutoZoom Focus：闭环结束后无法重启 & 默认参数调整

### Summary
针对 AutoZoom `autofocus_qt_ui` 的三个新需求进行修复：
1. 点击 **“启动闭环”** 循环结束后，无法再次启动闭环或建立参考；
2. 默认补焦间隔改为 **1.5 秒**；
3. 默认循环次数改为 **30 次**。

### Root Cause
1. **无法再次启动**：`AutofocusThread` 默认 `run()` 会启动事件循环且永不退出，worker 完成后底层 QThread 仍在运行；`_on_loop_finished()` 未显式等待线程结束并释放引用，导致下一次 `_start_loop()` 可能误判线程仍在运行，或 QThread 在销毁时仍运行而崩溃。
2. **默认参数不符合需求**：`Utils/AutoZoom/autofocus_qt_ui/config.py` 中 `DEFAULT_CYCLES=20`、`DEFAULT_INTERVAL_S=5.0`，与用户要求的 30 次、1.5 秒不一致。

### Implementation Changes
1. **默认参数调整**
   - `Utils/AutoZoom/autofocus_qt_ui/config.py`：`DEFAULT_CYCLES` 改为 `30`，`DEFAULT_INTERVAL_S` 改为 `1.5`。
   - `Utils/AutoZoom/autofocus_qt_ui/worker.py`：fallback 间隔默认值同步改为 `1.5`。

2. **线程生命周期修复（`Utils/AutoZoom/autofocus_qt_ui/worker.py`）**
   - `AutofocusThread` 中连接 `worker.finished -> self.quit`，worker 完成后立即退出线程事件循环，确保线程正常终止。

3. **UI 状态重置（`Utils/AutoZoom/autofocus_qt_ui/app.py`）**
   - `_start_loop()` 启动新线程前，若存在上次已完成但未释放的线程，则 `wait()` 并清空引用，避免影响重启。
   - `_on_loop_finished()` 中等待线程彻底结束并设置 `self._thread = None`；日志新增 `循环结束，可再次建立参考或启动闭环`。
   - 新增 `_build_and_start_thread()` 方法，统一封装线程构建与启动逻辑，便于 UI 和测试复用。

### Test Plan
1. **默认参数单测**：验证 `DEFAULT_CYCLES=30`、`DEFAULT_INTERVAL_S=1.5`，且 UI 控件初始值正确。
2. **循环结束后重启单测**：启动 1 轮闭环，等待自然结束，验证 `start_btn`、`build_ref_btn` 可用，`stop_btn` 禁用，日志包含“循环结束”。
3. **连续两次启动单测**：连续启动两次闭环，第二次也能正常结束且 UI 状态正确。
4. **回归测试**：
   - `python Utils/AutoZoom/test_picomotor_and_stop.py`
   - `python Utils/AutoZoom/test_final_crash_fix.py`

### Test Results
- `python Utils/AutoZoom/test_loop_restart_and_defaults.py`：全部通过。
- `python Utils/AutoZoom/test_picomotor_and_stop.py`：全部通过。
- `python Utils/AutoZoom/test_final_crash_fix.py`：全部通过。

### Files Modified
- `Utils/AutoZoom/autofocus_qt_ui/config.py`
- `Utils/AutoZoom/autofocus_qt_ui/worker.py`
- `Utils/AutoZoom/autofocus_qt_ui/app.py`
- `Utils/AutoZoom/test_loop_restart_and_defaults.py`（新增）
- `PLAN.md`

---

## AutoZoom Focus：实时预览基准图对比 & 2x2 闭环回顾弹窗

### Summary
为 AutoZoom `autofocus_qt_ui` 增加两项可视化能力：
1. **启动闭环后**，实时预览区域左右并排显示 **基准图** 与 **实时图**，方便人眼直观比较聚焦状态；
2. **闭环自然结束后**，自动弹出 **2x2 图像变化回顾弹窗**，展示基准图、早期、中期、末期四个阶段的 ROI 图像。

### Implementation Changes
1. **保存基准图像（`Utils/AutoZoom/Focus/scorer.py`）**
   - `build_reference()` 在建立参考基线时，把第一张采集到的 `roi_rgb` / `full_rgb` 存入 `focus_reference`，供 UI 调用。

2. **预览对比控件（`Utils/AutoZoom/autofocus_qt_ui/widgets.py`）**
   - `RoiPreviewLabel` 新增 `set_reference_image()`、`set_comparison_mode()`。
   - 对比模式下 `paintEvent` 左右并排绘制基准图与实时图，并分别标注“基准图”“实时图”。
   - 实时图区域继续绘制 ROI 框，保持原有交互能力。
   - 新增 `LoopSummaryDialog`：2x2 网格弹窗，自动从快照中挑选基准图、早期、中期、末期 4 张图像。

3. **主窗口集成（`Utils/AutoZoom/autofocus_qt_ui/app.py`）**
   - `_build_reference()` 完成后将基准 ROI 图像保存到 `self._reference_image` 并设置到预览控件。
   - `_start_loop()` 与 `_build_and_start_thread()` 启动时开启对比模式（若已有基准图）。
   - `_on_preview_updated()` 每轮采集时把 `(cycle, roi_rgb)` 追加到 `self._loop_snapshots`。
   - `_on_loop_finished()` 异步调用 `_show_loop_summary_dialog()`，以非阻塞方式展示 2x2 弹窗。
   - `_on_loop_error()` 退出对比模式。

### Test Plan
1. **基准图保存单测**：验证 `build_reference()` 返回的 `focus_reference` 包含 `roi_rgb`。
2. **对比模式单测**：验证 `RoiPreviewLabel` 进入对比模式后同时保存基准图和实时图的 pixmap。
3. **弹窗图像挑选单测**：验证 `LoopSummaryDialog._select_images()` 始终返回 4 张图像，且第一张为基准图、最后一张为末期。
4. **集成单测**：运行闭环后验证 `self._loop_snapshots` 已收集、`_summary_dialog` 已创建并显示。
5. **回归测试**：
   - `python Utils/AutoZoom/test_loop_restart_and_defaults.py`
   - `python Utils/AutoZoom/test_picomotor_and_stop.py`
   - `python Utils/AutoZoom/test_final_crash_fix.py`

### Test Results
- `python Utils/AutoZoom/test_preview_comparison_and_summary.py`：全部通过。
- `python Utils/AutoZoom/test_loop_restart_and_defaults.py`：全部通过。
- `python Utils/AutoZoom/test_picomotor_and_stop.py`：全部通过。
- `python Utils/AutoZoom/test_final_crash_fix.py`：全部通过。

### Files Modified
- `Utils/AutoZoom/Focus/scorer.py`
- `Utils/AutoZoom/autofocus_qt_ui/widgets.py`
- `Utils/AutoZoom/autofocus_qt_ui/app.py`
- `Utils/AutoZoom/test_preview_comparison_and_summary.py`（新增）
- `PLAN.md`
