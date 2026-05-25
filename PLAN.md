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
