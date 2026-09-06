# Microscope-master 仿真子模块维护说明

## 已完成

- `API/microscope-master/simulator_app` 已成为可导入 Python 包，并保留旧的直接脚本启动方式。
- 新增稳定无头入口：`SimulatorConfig`、`MicroscopeSimulator`、`create_simulator`。
- 默认使用 `:memory:` 数据库，库导入不会修改源码目录；需要重启恢复时显式传入 `db_path`。
- 数据库自动创建父目录、支持上下文管理、关闭幂等，并使用 SQLite `user_version` 作为迁移标记。
- `SQLiteStage` 对多轴目标先完整校验再写入，避免多轴操作中途留下部分状态；新增 `describe()` 能力快照。
- `SampleAwareCamera.capture()` 提供公开同步采集接口，调用方不再需要依赖 `_fetch_data()`。
- `obstacle_avoidance.sim_microscope` 改用 `simulator_app` 包导入，并在关闭时释放数据库连接。
- `main.py` 支持 `--db` 和 `--slow-preview`，可使用模块方式或脚本方式启动。

## 推荐使用方式

```python
from simulator_app import create_simulator

with create_simulator(db_path="artifacts/my-project/microscope.db") as sim:
    sim.move_to({"x": 750, "y": 500, "z": 0})
    frame = sim.capture()
```

项目只需要 `sim.stage`、`sim.camera` 或 `sim.db` 的高级能力时，可以直接访问这些对象；构造和资源回收仍由 `MicroscopeSimulator` 管理。

## 后续维护边界

1. `simulator_app/api.py` 只负责公共构造、生命周期和稳定的应用级接口。
2. `devices.py` 只实现 python-microscope 设备语义；项目特定场景放在外层，不继续堆入设备类。
3. `db.py` 的表结构变更必须增加 `PRAGMA user_version` 迁移步骤，并保留旧数据库可读性。
4. 新增公共方法先写无头测试，再补 GUI；GUI 不应成为模块可用性的前置条件。
5. 修改 vendored `microscope-master` 核心包前，先确认能否在 `simulator_app` 适配层解决，避免升级源代码时产生难以同步的分叉。

## 验证命令

```powershell
python -m pytest tests -q
python -c "import sys; sys.path.insert(0, 'API/microscope-master'); from simulator_app import create_simulator; s=create_simulator(); print(s.capture().shape); s.close()"
python -m simulator_app.main --db artifacts/microscope.db
```

当前项目测试结果：`67 passed`。完整 pytest 若递归收集 vendored 上游测试，Windows 环境下仍可能出现其 Pyro4 multiprocessing 测试失败；这不属于仿真子模块公共 API 回归。
