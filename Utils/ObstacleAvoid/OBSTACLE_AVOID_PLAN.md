# 避障与聚拢实施 PLAN

目标：在现有 `SpotZoomController`、视觉检测器、stage protocol 和 Qt 运行服务之上，增加“单球到终点避障”和“多球聚拢到区域”两条可仿真、可审计、可安全回退的流程。

## 计划清单

### 1. 建立领域模型与坐标契约

- 新增 `obstacle_avoidance/models.py`：`ParticleTrack`、`SubstrateRegion`、`Obstacle`、`GoalRegion`、`WorkspaceSnapshot`、`StageCommand`、`RunState`。
- 明确像素坐标、显微镜物理坐标、位移台步数三套坐标及方向/单位；所有对象带时间戳、置信度和来源帧。
- 定义可行域：衬底 polygon 内缩安全边界；球半径、光斑半径、障碍膨胀半径进入碰撞模型。
- 验收：坐标往返误差 <= 1 px（仿真）；越界命令在规划层被拒绝；JSON schema 可序列化。
- 回滚点：只启用模型和仿真，不接管现有准直默认流程。

### 2. 实现视觉层：衬底、障碍物和多球跟踪

- `obstacle_avoidance/vision.py`：复用 OpenCV/现有 classic detector；可插拔 `Cellpose`/`micro-sam`/YOLO 分割；Trackpy/DeepTrack2 作为多目标关联参考。
- 输出对象检测、障碍 mask、衬底 mask、球半径和置信度；连续丢帧、遮挡、低置信度必须显式返回 `DETECTION_UNCERTAIN`。
- 修复 Unicode 图片读取，统一使用 `np.fromfile + cv2.imdecode` 或 QFile/QImage 入口。
- 验收：合成帧中球中心误差 <= 2 px；边界 IoU >= 0.95；障碍漏检率和误检率分别记录。

### 3. 实现规划层：静态与动态避障

- `obstacle_avoidance/planner.py`：先做 occupancy grid + A*；需要动态障碍时增加 D* Lite 或速度障碍；对每个时间步做球/光斑半径膨胀碰撞检测。
- 规划输出带 waypoint、预计长度、最小 clearance、版本号和失败原因；无路可走时返回 `NO_SAFE_PATH`，禁止直接下发电机。
- 起点/终点在衬底外、障碍重叠或 clearance 不足时拒绝任务。
- 验收：无障碍直线路径；静态障碍绕行；窄通道拒绝；动态障碍重规划延迟和成功率有指标。

### 4. 实现闭环位移与安全状态机

- `obstacle_avoidance/controller.py`：复用 `XYStageProtocol` 和 `RunReporter`，按 waypoint 做“小步移动 -> 等待稳定 -> 重检 -> 更新轨迹”。
- 状态：`IDLE -> CALIBRATING -> TRACKING -> PLANNING -> MOVING -> VERIFYING -> COMPLETE/ABORTED/FAULT`。
- 任何检测不确定、越界、通信超时、急停或障碍突现都进入安全停止；恢复必须重新获取目标和规划，不能盲目续跑。
- 硬件策略：新增显式确认开关；CLI/UI 默认 dryrun，真实设备需二次确认和启动运动自检。
- 验收：急停到最后一次命令 <= 100 ms（仿真目标）；每条电机命令可追溯到 waypoint 和视觉帧。

### 5. 实现聚拢任务与分配策略

- `obstacle_avoidance/aggregation.py`：检测多个球，使用 Hungarian/最短路或 optimal transport 将球分配到聚拢区域入口；逐球规划，保留已到达球的安全占位。
- 支持开环定距和闭环视觉两模式；闭环模式以区域内数量、最大半径、质心偏差和速度阈值判定完成。
- 处理球重叠、暂时丢失、目标区域容量不足和任务取消；避免把一个球误关联为另一个球。
- 验收：N=1、N=2、N>=10；聚拢区域边界、满载、遮挡和单球丢失均有明确结果。

### 6. 接入 Qt、运行报告和数据持久化

- `spotzoom_qt_ui` 增加任务选择、衬底/障碍/终点标注、路径预览、暂停/急停、置信度和 clearance 面板。
- 修复 Qt 兼容层：按实际部署选择 PySide6/PySide2，或补充 PyQt5 适配；不要在 `app.py` 中绕过兼容层直接导入某一绑定。
- `RunReporter` 扩展 schema：任务配置、帧 ID、检测、规划、stage 命令、异常和最终指标；JSONL 保持流式，SQLite 仅在需要跨任务查询时引入。
- `RuntimeControlService.build_command` 使用 `self.python_exe`，单步模式同步设置稳定帧策略。
- 验收：离线仿真 UI 可启动；日志无未处理异常；报告可被 UI 重新加载；导出 JSON/CSV 与任务 ID 一致。

## 新增模块黑盒测试表

| ID | 场景 | 输入/操作 | 预期 |
|---|---|---|---|
| OA-01 | 单球无障碍 | 起点、终点均在衬底内 | 生成直线路径并闭环到达 |
| OA-02 | 单球静态障碍 | 障碍横跨直线路径 | 生成绕行 waypoint，clearance 始终达标 |
| OA-03 | 无可行路径 | 障碍封闭目标 | 返回 `NO_SAFE_PATH`，无电机命令 |
| OA-04 | 衬底边界 | 终点在 polygon 外 | 拒绝任务，报告越界原因 |
| OA-05 | 障碍突现 | 移动中新增/移动障碍 | 安全停机并重规划 |
| OA-06 | 低置信度 | 连续低置信度或丢帧 | 进入 `DETECTION_UNCERTAIN`，不盲动 |
| OA-07 | 终点判定 | 距离 <= tolerance 且连续稳定帧 | `COMPLETE`，记录最终误差 |
| AG-01 | 单球聚拢 | 1 个球、一个区域 | 进入区域并完成 |
| AG-02 | 多球聚拢 | N=2/10，路径有交叉 | 分配稳定、逐球避碰、区域内数量正确 |
| AG-03 | 球重叠 | 两球在检测框重叠 | 保持 track ID，不重复计数 |
| AG-04 | 聚拢区域非法 | 区域在衬底外/面积不足 | 拒绝任务并说明原因 |
| AG-05 | 任务暂停/急停 | 运行中点击暂停/急停 | 新命令停止，恢复需重新验证 |
| HW-01 | dryrun 回归 | 仿真图像 + dryrun stage | 现有准直流程结果不变 |
| HW-02 | 通信超时 | 模拟 stage/camera 超时 | 状态为 `FAULT`，无后续命令 |
| UI-01 | UI 启动 | offscreen 启动、打开样例帧 | 无导入错误、资源正常显示 |
| UI-02 | 报告回放 | 加载 JSON/JSONL 报告 | 路径、事件、错误和指标可复现 |

## 验证顺序与门禁

1. 先通过模型/规划纯单测，再通过合成图像视觉单测。
2. 使用 `dryrun + classic + --no-preview` 跑 OA-01 至 OA-07 和 AG-01 至 AG-04。
3. 使用噪声、抖动、丢帧和动态障碍做压力回归，保存 JSONL。
4. UI 使用 offscreen 启动和 QProcess 端到端验证；真实 Qt 绑定必须与 lock/requirements 一致。
5. 真实设备仅在硬件确认、启动运动自检、急停和通信超时测试通过后放行；先单轴、再双轴、最后多球任务。

## 完成定义

- P0 功能：单球终点避障和多球聚拢均有仿真通过证据。
- P1 安全：越界、无路、低置信度、通信异常均不下发危险命令。
- P1 可用：Qt UI 可启动，路径和状态可观察，报告可回放。
- P2 质量：核心模块职责清晰，异常有类型，闭环无无界循环，性能指标进入报告。
- 未满足以上门禁前，版本状态只能标为“设计/仿真验证中”，不得标记为真实设备完成。
