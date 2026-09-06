# 避障与聚拢需求：本轮项目审计报告

审计日期：2026-09-02（Asia/Shanghai）  
项目根目录：`E:\jupyter file\2_Optics\8821L`  
入口：`SpotZoom.py`；桌面 UI：`spotzoom_qt_ui`；视觉/控制扩展：`SpotZoom_Machine_Learning*`

## 结论摘要

当前仓库是显微镜光斑闭环准直/自动对焦工具，不是 Web 服务。现有主流程能在无硬件仿真下完成光斑检测和 XY/Z 闭环，但用户描述的“材料圆球、衬底边界、障碍识别、终点导航、聚拢区域”尚未在核心模块中实现。Qt UI 在本机无法导入，原因是代码只支持 PySide6/PySide2，而环境安装的是 PyQt5。

| 领域 | 结果 | 证据 |
|---|---|---|
| Python 构建/语法 | 通过 | `python -m compileall -q SpotZoom.py spotzoom_qt_ui tests SpotZoom_Machine_Learning`，退出码 0 |
| 核心 ML/电机单测 | 通过 | `test_spotzoom_ml_unified.py`：3 passed；`test_spotzoom_picomotor_drivers.py`：4 passed |
| Qt 服务测试 | 阻断 | `test_spotzoom_qt_ui_services.py` 收集失败：缺少 PySide6/PySide2 |
| 单探测器仿真闭环 | 通过 | `artifacts/current_smoke_report.json`：success，5/5 检测成功，0.724 s |
| 双探测器 4 轴仿真 | 条件通过 | `dual_detector_smoke_report3.json` 在 `converge_stable_frames=5`、6 次迭代下通过 |
| 单步双探测器流程 | 失败 | 默认稳定帧数为 5，但 `max_iterations=1`，即使误差为 0 也返回 `not_converged` |
| 静态图像资源 | 文件有效；Unicode 路径有风险 | 47 个图片经 `imdecode` 全部通过；中文路径直接 `cv2.imread` 失败 |
| Web 页面/REST API/数据库 | 不适用/未实现 | 核心代码无 Flask/FastAPI/HTTP/SQLite；UI 通过 `QProcess` 调 CLI、读写 JSON/JSONL |
| 真实相机/电机/网络设备 | 未验证 | 当前环境未连接 ToupView、UCC、Newport、Thorlabs 或 XPS |

## 已执行检查

```text
python -m compileall -q SpotZoom.py spotzoom_qt_ui tests SpotZoom_Machine_Learning
python -m pytest -q tests/test_spotzoom_ml_unified.py
python -m pytest -q tests/test_spotzoom_picomotor_drivers.py
python SpotZoom.py --check-env --frame-source-image artifacts\sim_spot.png --xy-driver dryrun --z-driver dryrun --detector-backend classic --skip-roi --disable-run-lock
python SpotZoom.py --frame-source-image artifacts\sim_spot.png --xy-driver dryrun --z-driver dryrun --detector-backend classic --skip-roi --disable-run-lock --max-iterations 1 --no-preview
python SpotZoom.py --frame-source-image artifacts\sim_spot.png --frame-source-image-2 artifacts\sim_spot.png --xy-driver dryrun --z-driver dryrun --detector-backend classic --alignment-strategy dual_detector_4axis --detector-mode dual_detector --skip-roi --skip-roi-2 --disable-run-lock --max-iterations 6 --settle-time 0
```

异常输入校验也符合预期：不存在的图像和不存在的 YOLO 模型分别返回退出码 1，并输出明确错误。全量 `pytest -q` 超过约 64 秒无收集摘要，已停止；很可能包含需要真实相机的 `tests/test_ucc_camera.py`，不能据此判为通过。

## 问题清单

### P0：需求功能缺口

核心目录中没有 `obstacle`、`avoidance`、`substrate`、`goal`、`path planning`、`aggregation` 等实现符号或跟踪状态。`SpotZoomController` 当前只实现 P1 -> Z up/P2 -> XY 对齐 -> Z down/P3 的准直闭环（`SpotZoom.py:8959` 起），不能表达多个材料球、可行域、障碍物、终点或聚拢区域。影响模块：主控制器、视觉检测、UI、运行报告和硬件安全层。详见 [OBSTACLE_AVOID_PLAN.md](OBSTACLE_AVOID_PLAN.md)。

### P1：Qt UI 无法启动

`spotzoom_qt_ui/qt_compat.py:9-104` 仅尝试 PySide6/PySide2；`spotzoom_qt_ui/app.py:81-86` 又直接导入 PySide6/PySide2。当前环境只有 PyQt5，导致 `from spotzoom_qt_ui import run_qt_ui` 和 Qt 服务测试在收集阶段失败。依赖声明 [requirements-win7.txt:5](requirements-win7.txt:5) 与实际代码/解释器组合也不匹配。影响页面访问、前端控制台、设备面板、QProcess 流程和新增模块 UI 验收。

### P1：默认启动可能触碰真实硬件

`SpotZoom.py` 的 CLI 默认 `--xy-driver=newport`、`--z-driver=wheel`。直接运行入口会尝试真实设备；虽然有启动运动自检，但缺少显式“硬件确认”门槛。建议把 dryrun 作为默认或要求 `--real-hardware-confirm`，并在 UI 与 CLI 统一安全策略。影响电机、激光/光斑和实验样品安全。

### P1：Unicode 图像路径兼容性

`SpotZoom.py:4167-4172` 使用 `cv2.imread(str(frame_path))`。本机验证 `Document/img/实际光路.jpg`：直接 `cv2.imread` 返回空，而 `np.fromfile + cv2.imdecode` 成功。中文数据集/用户选择文件会被误报为“无法解码”。

### P2：4 轴单步测试的收敛语义不一致

`FourAxisController` 需要连续稳定帧，默认 `converge_stable_frames=5`；UI 的单步命令在 `spotzoom_qt_ui/services.py:229` 只把 `max_iterations` 强制为 1。因而精确零误差的双探测器单步运行仍返回 `not_converged`（`artifacts/dual_detector_smoke_report.json`）。应在单步模式传入稳定帧数 1，或把“当前误差已在容差内”与“连续稳定帧数”分开呈现。

### P2：自定义 Python 解释器未生效

`RuntimeControlService.__init__` 保存了 `self.python_exe`，但 `build_command` 在 `spotzoom_qt_ui/services.py:229` 使用 `sys.executable`。选择专用 YOLO/虚拟环境时会启动错误解释器，影响依赖隔离和可复现性。

### P2：可维护性与性能风险

历史静态审计（`audit_results.json`，2026-05-13）记录：134 个 Python 文件、100,388 行、687 个类、2,408 个函数、58 个长函数、94 个重复签名、92 处 broad `except Exception`、45 个文件使用 `print`、`AlignmentConfig` 246 个字段、10 个线程创建点。`SpotZoom.py` 约 1.4 MB，ML 版本目录重复度高。建议按视觉、规划、控制、设备、持久化拆分，并限制异常捕获范围；不要在闭环中反复创建数组或同步写文件。

### P3：历史报告可信度

`project_health_report.json` 的 `overall_score=0` 与大量历史问题不能代表本轮状态；报告时间为 2026-05-11。新报告必须包含提交号、命令、环境和时间戳。当前生成的仿真证据位于 `artifacts/current_smoke_report.json`、`artifacts/dual_detector_smoke_report3.json` 和对应 JSONL 事件流。

## 日志、接口、数据和资源结论

- 控制台：CLI 日志正常，仿真运行事件以 JSONL 写入；Qt 控制台无法验证，因为 UI 导入被 Qt 依赖阻断。
- 接口：未发现 Web/REST 接口；`RuntimeControlService` 通过 `subprocess.run` 启动 `python -m SpotZoom`，参数和返回码可观测。
- 网络：XPS 使用 TCP，Newport/相机依赖本机驱动；无真实设备时只能做参数错误和超时策略审查，不能验证断线重连。
- 数据库：未发现 SQLite/SQL/ORM；运行报告、事件流、配置使用 JSON/JSONL，适合先做 schema 校验，后续再决定是否引入 SQLite。
- 静态资源：47 个 `artifacts`/`Document/img` 图片用 Unicode-safe 解码全部有效；OpenCV 直接读取中文路径失败，需修复后再做 UI 资源回归。

## 科研前沿开源项目检索（GitHub API，2026-09-02）

| 项目 | GitHub 元数据 | 对本需求的可复用部分 |
|---|---|---|
| [Trackpy](https://github.com/soft-matter/trackpy) | 508 stars，更新 2026-08-25 | 多粒子检测、链接、轨迹质量指标 |
| [DeepTrack2](https://github.com/DeepTrackAI/DeepTrack2) | 243 stars，MIT，更新 2026-08-23 | 显微成像仿真、训练数据和可组合视觉管线 |
| [micro-sam](https://github.com/computational-cell-analytics/micro-sam) | 716 stars，MIT，更新 2026-09-01 | 显微镜衬底/障碍物交互式分割 |
| [Cellpose](https://github.com/MouseLand/cellpose) | 2,337 stars，BSD-3-Clause，更新 2026-09-02 | 材料球/障碍物分割，支持人工校正 |
| [SAM 2](https://github.com/facebookresearch/sam2) | 19,793 stars，Apache-2.0，更新 2026-09-02 | 视频级目标分割与遮挡后重识别 |
| [python-microscope](https://github.com/python-microscope/microscope) | 85 stars，GPL-3.0，更新 2026-08-12 | 相机、位移台、硬件触发抽象；需评估 GPL 兼容性 |
| [pyRTC](https://github.com/jacotay7/pyRTC) | 13 stars，GPL-3.0，更新 2026-03-08 | 实时自适应光学控制循环；可参考线程/延迟监控 |
| [dmlib](https://github.com/jacopoantonello/dmlib) | 24 stars，更新 2026-05-11 | 变形镜标定和控制接口 |
| [Velocity Obstacle + RRT*](https://github.com/Abeilles14/Velocity-Obstacle-and-Motion-Planning) | 66 stars，MIT，更新 2026-04-26 | 动态障碍避碰、RRT* 路径规划基线 |
| [PSO Path Planning](https://github.com/smkalami/path-planning) | 54 stars，MIT，更新 2026-03-25 | 低速静态场景的全局路径基线，不建议直接用于安全闭环 |
| [SmartTrap（仓库内镜像）](Utils/SmartTrap-main/SmartTrap-main/Software/README.md) | 本地源码，PyQt 测试模式 | 光镊粒子追踪、`move_avoiding_particles` 业务语义、无硬件测试模式 |
| [X-AnyLabeling（仓库内镜像）](Utils/X-AnyLabeling-main/README.md) | 本地源码，GPL-3.0，上游约 10.3k stars | 标注/分割工作流和模型接入参考 |

推荐组合：`micro-sam/Cellpose` 负责衬底与障碍分割，`Trackpy/DeepTrack2` 负责多球跟踪，`A*/D* Lite + 速度障碍` 负责路径，沿用现有 stage protocol 和 `RunReporter` 做闭环执行与审计。GPL 项目只作为架构参考，集成前需完成许可证评估。

## 本轮风险结论

在修复 Qt 依赖、增加硬件确认和落地 P0 功能前，不应宣称“已完成避障/聚拢”。当前可交付状态是：光斑准直仿真通过；新增功能为设计阶段；真实设备、动态障碍和多球聚拢均未验证。
