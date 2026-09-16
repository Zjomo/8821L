# PLAN：框定元素状态保持 / 离屏续跑 / 运行结束停在最终窗口（2026-09-15）

## 0. 需求
1. 解决上一次程序关闭时，框定元素（球 / 衬底 / 障碍物 / 目标点 / 目标范围）的位置、对象丢失问题。
2. 始终记录画面中所有元素位置；元素离开画面范围时不得直接 `ABORTED【FAIL】`，始终保证定位运动。
3. Alg2 标定激光控制圆球移动到目标点后，画面停在最终达到的窗口状态与位置（不回起点）。

## 1. 现状与根因（已逐条核对代码）

### 需求 1
- **R1-1 运行结果不落盘**：`_sim_cfg` 只在画框 `_sim_rect`（app.py:2857）与撤销 `on_undo_zone`（app.py:3599）时 `_save_sim_config()`。
  运行结束后 `on_layout_updated`（app.py:4308-4321）改写了 `_sim_cfg["balls"]` 却**不落盘** → 关闭程序后球的位置回退到运行前（"元素位置丢失"）。
- **R1-2 视窗与元素脱节**：`closeEvent`（app.py:2180-2206）只把台位写回 SQLite；而 worker 运行期间台位变化也写同一 SQLite
  （sim_microscope.py:154-169，`motion_stage` 的 `on_move → micro_stage.move_to`）。重启后恢复的视窗是"运行终点"，
  元素存的是样本绝对坐标，若离该视窗较远则画面里看不到已画元素 → 观感上"对象丢失"。

### 需求 2
- **R2-1 不确定性即中止**：`VisionPipeline.process`（vision.py:365-373）`expect_particle and not fresh` → `uncertain`；
  Alg2（algorithm2.py:463-469）、Alg1 控制器重试超限 → `ABORTED(DETECTION_UNCERTAIN)`。
- **R2-2 找不到目标球即中止**：`_target_particle`（algorithm2.py:276-304）返回 None → `ABORTED(TARGET_LOST)`（algorithm2.py:474-478）；
  aggregation.py:248 与 Alg1 控制器同类路径。
- **R2-3 coast 太短**：`ParticleTracker(max_lost_frames=3)`（vision.py:244-300），3 帧后 track 被删除，位置信息彻底丢失。
- **R2-4 快照无历史**：`SimMicroscopeWorld.snapshot()`（sim_microscope.py:390-415）只含"可见"粒子，`WorkspaceSnapshot` 无跨帧位置记忆。

### 需求 3
- **R3-1 结束后改由 `_sim_live` 渲染**：运行帧来自 worker 世界（WYSIWYG，origin=运行起点）；结束后 `_refresh_sim_preview`（app.py:3068-3087）
  改从 `_sim_live` 取帧，而 `_sim_live.motion_stage` 仍停在**运行前**位置 → 画面跳回起点；
  `_overlay_sim_cfg`（app.py:3006-3020）非运行期同样用 `_sim_origin()`（= 旧台位）。
- worker 结束时未把最终台位回传 UI（`layout_updated` 只带球框，见 app.py:423/612/639）。

## 2. 实施方案

### W1 框定元素持久化（需求 1）
- [x] W1.1 `on_layout_updated` 末尾补 `_save_sim_config()`（app.py:4320），运行结果即时落盘。
- [x] W1.2 `_save_sim_config`（app.py:3096）增加 `"view": {"origin": [ox,oy], "stage_um": [x,y]}`（有 `_sim_live` 时写当前视窗）。
- [x] W1.3 `_load_sim_config`（app.py:3106）末尾：若已有元素且其包围盒中心不在当前视窗内 → `_sim_live.motion_stage.move_to(...)`
      把包围盒中心移到视窗中心（`_simlog` 记录）；无 `_sim_live` 时暂存，`_ensure_live` 后应用。
- [x] W1.4 核对载入后 `_sim_ids` / 属性面板映射（app.py:3148-3157）保证对象 ID 唯一且与列表一致。

### W2 离屏元素不中止（需求 2）
- [x] W2.1 `ParticleTracker` 增加可配置 coast 长度（`keep_lost_frames`，默认 30）与"上次真实位置 + 速度×dt"外推（vision.py:294-297, 328-338）。
- [x] W2.2 新增 `PositionLedger`（放 vision.py，避免新建文件）：按 track_id/元素角色记录
      `(样本坐标, 窗口坐标, 最后可见帧, 来源=检测|外推)`，对外 `record_detected / predict / offscreen`。
- [x] W2.3 `WorkspaceSnapshot` 元素加 `in_frame` 标记（models.py:314-333）；`SimMicroscopeWorld.snapshot()` 用台账补齐离屏元素
      （仿真世界有真值，可直接给出），障碍/衬底/目标点同理。
- [x] W2.4 `VisionPipeline.process`（vision.py:352-380）新增 `allow_offscreen=False`：无新鲜检测但台账有记录时不再判 `uncertain`。
- [x] W2.5 控制器兜底（algorithm2.py:462-478、controller.py 对应段、aggregation.py:236-254）：
      目标球/衬底/目标点缺失 → 用台账外推位置继续定位运动；仅当外推超时/超出样本范围才 `ABORTED`，并记 `element_offscreen` 事件。
      新增开关 `ControllerConfig.allow_offscreen_elements: bool = True`，保留旧中止为台账无数据时的兜底。
- [x] W2.6 `on_state`（app.py:4368-4372）对"外推中"显示 `TRACKING(离屏外推)`，避免误报 `ABORTED【FAIL】`。
- [x] W2.7 起点临界越界不中止：膨胀模型随实测球半径自适应（YOLO 半径抖动 ±1px），球可能停在"恰好贴着
      膨胀边界"处（实测 infl=36.256 vs 球位 clearance=36.18，差 0.08px）→ 起点被拒即 `ABORTED(LOW_CLEARANCE)`。
      `GridPlanner.nearest_feasible` + `plan()` 改为：起点仅因 `LOW_CLEARANCE` 被拒时就近（≤8px）挪到可行点后继续规划；
      附近无可行点（球深陷障碍）仍走原中止路径。这也修掉了改动前就失败的 `test_video02_static_obstacle_bypass`。

### W3 运行结束停在最终窗口（需求 3）
- [x] W3.1 `WorkerThread` 增加 `stage_ready = Signal(dict)`（app.py:125-160 附近），在 sim 路径 `world.close()` 之前 emit
      `dict(world.motion_stage.position)`（app.py:638 / 652 两条路径）。
- [x] W3.2 UI 新增 `on_stage_ready` 暂存 `self._final_stage_position`；`on_done`（app.py:4384-4388）在虚拟模式下
      `_sim_live.motion_stage.move_to(pos)`（走 `on_move` 落盘）后 `_refresh_sim_preview()` → 画面停在最终窗口。
- [x] W3.3 结束不重置 `_run_origin`（运行期 overlay 仍需要）；结束后 overlay 自然使用 `_sim_origin()`（= 最终台位）。
      电机模式（真实台位）不受影响，不做任何移动。

## 3. 执行顺序
W3.1 → W3.2 → W1.1 → W1.2/W1.3 → W2.1 → W2.2/W2.3 → W2.4 → W2.5 → W2.6
（W3 与 W1.1 改动点相邻，先做；W2 风险最高放最后。）

## 4. 验收标准（功能测试必须通过）
- 新增 `tests/test_sim_persistence.py`：
  ① 画 衬底+目标点+球+障碍 → 新窗口载入 → 数量与样本坐标逐一相等；
  ② 运行更新球位（模拟 `on_layout_updated`）→ 重启后球位为新值；
  ③ 撤销后重启，被撤销对象不复活。
- 新增 `tests/test_offscreen.py`：
  ④ 目标球移出视野 → 运行不进入 `ABORTED`，台账仍有位置记录且最终到达目标点；
  ⑤ 衬底/目标点离屏 → 同上（用外推位置）。
- 新增 `tests/test_run_state.py`：
  ⑥ sim01 Alg2 跑完 → `_sim_live.motion_stage.position == worker 最终台位`；
  ⑦ 结束后 `_sim_origin()` 与最后一帧窗口一致（画面不再跳回起点）；
  ⑧ `_sim_cfg["balls"]` 已更新为目标点位置且已落盘。
- 回归：`python -m pytest tests -q` **全绿**（194 passed, 1 skipped），含改动前即失败的
  `test_video_yolo.py::test_video02_static_obstacle_bypass`（已由 W2.7 修复）。

## 5. 风险与回滚
- **W2 风险最高**（改控制循环）：用 `allow_offscreen_elements` 开关隔离；外推设帧数/距离上限，超限仍 `ABORTED`，并写 `element_offscreen` 日志便于复盘。
- **W1.3 启动自动移视窗**会改变启动画面：仅在元素确实不在视窗内时才移动，并写操作日志；回滚只需去掉该调用。
- 每个工作流按"文件 + 函数"为最小改动单位，可独立回滚。