# PLAN：Alg2 光斑移动 abort 修复 + 多球报错与日志补全（2026-09-16）

## 0. 需求
1. 修复"标定激光，向圆球移动过程中出现 aborted"的 bug。
2. 修复"多个球时出现报错"，并补全/修正日志（让多球场景可诊断）。

## 1. 现状与根因（已读代码/运行复现确认）

所在文件：[algorithm2.py](obstacle_avoidance/algorithm2.py)（FixedBeamController）。

### 需求 1：向光斑移动时 abort
- 对齐阶段（`run` L407-466）唯一的中止出口是 `if not aligned -> ABORTED(SPOT_SLIP)`，
  **不区分失败原因**：
  - 若目标球从未被成功检测/解析到（`uncertain`/`_target_particle` 返回 None/
    离屏补不到），走 `continue` 空转，`align_limit` 耗尽后也落到 SPOT_SLIP——
    明明是"检测不到球"，却报"球没对齐（光斑滑移）"，是误报且原因误导。
- 对齐用 `_capture_limit_px`（`radius*0.8`，小球如 12px → 9.6px）作"光斑命中"判据，
  若球半径被检测偏差放小，判据被过度收紧，稍远即判"未命中"，叠加上面的空转导致超限 abort。

### 需求 2：多球报错 + 日志问题
- 跟踪阶段 `_target_particle`（L299-327）在**beam_lock 后仍可能静默切换目标**：
  若锁定球某一帧未被检出，而某个邻居恰好更靠近光斑（`margin >= 0.35*radius`），
  就会返回邻居，调用方随即 `set_alg2_target(新id)`，被锁定的真球变成障碍、激光保持——
  多球场景先"错认目标"再"报错中止"，且**日志完全看不出目标被切换**。
- Alg2 对比 Alg1（controller.py L264/L277）：跟踪循环**从不记录 `detection` 事件**，
  `extract_trajectory` 的回放帧因此是空的；也没有"目标切换/拒绝邻居"的任何记录
  —— 这就是"日志有问题"（多球时无从诊断）。

## 2. 实施方案

### W1 目标球锁定（钉住 beam_lock 身份）
- [x] W1.1 `FixedBeamController.__init__` 增 `self._locked_track_id = None`。
- [x] W1.2 `_target_particle(..., pinned)`：`pinned=True` 时只按钉住 track_id 解析，
      哪怕邻居离光斑更近也不切换；本帧缺失返回 None 交给离屏外推续追。
- [x] W1.3 `beam_lock` 成功后设置 `self._locked_track_id = track_id`；跟踪循环用 `pinned=True`。
- [x] W1.4 对齐阶段用 `ball_seen` 区分：全程未解析到球 → `ABORTED TARGET_LOST`
      （detail 含 "never resolved"）；已解析但超限未命中 → `ABORTED SPOT_SLIP`，不再误报。

### W2 Alg2 日志补全（多球可诊断）
- [x] W2.1 跟踪循环每帧记录 `detection`（镜像 controller.py）：含全部
      `particles=[p.to_dict()...]` 与 `target_track_id`。
- [x] W2.2 目标切换（pinned 回落）/拒绝邻居时记录 `beam_target`（当前锁定逻辑已抑制切换，
      因此切换事件即为异常信号，便于诊断）。
- [x] W2.3 各 abort 分支 detail 补充具体 track/原因（对齐 vs 丢失 vs 离屏）。
      （注：初始计划中的动态延长对齐步数已丢弃——条件用"px 与步数比较"，语义错误且有风险。）

## 3. 执行顺序
W1.4（对齐原因区分）→ W1.1/W1.2/W1.3（锁定）→ W2.1/W2.2/W2.3（日志）。

## 4. 验收标准（功能测试必须通过）
- 新增 `tests/test_algorithm2_multiball.py`：
  ① pinned 状态不因邻居更靠光斑切换目标（unpinned 才回落最近邻）；
  ② 多球端到端 COMPLETE，产生 `detection` 日志（含全部球），无 `beam_target` 切换；
  ③ 初始目标缺失 → `ABORTED(TARGET_LOST)`。
- 回归：`python -m pytest tests -q` **全绿**（197 passed, 1 skipped）。

## 5. 风险与回滚
- 锁定会收紧多球追踪：若真球彻底丢失（钉住 id 不在任何粒子中且离屏也无记录），
  当前会走离屏/外推续追，超限后在 abort（`TARGET_LOST`）前仍记录 `detection` 便于判断。
  回滚只需去掉 W1.2/W1.3 的钉住逻辑。
- 日志为新增事件，不改变既有事件含义，无兼容风险。