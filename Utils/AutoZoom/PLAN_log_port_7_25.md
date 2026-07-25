# 日志实时保存功能移植到 7_25 实施计划

## 1. 目标

将 `0_measurement_workflow_real_virtual_same_detection_7_23.py` 中的「运行日志实时保存到文件」功能完整移植到 `0_measurement_workflow_real_virtual_same_detection_7_25.py`，包括：

1. `MeasurementConfig` 中保留 `save_log_to_file` 和 `log_dir` 字段；
2. `MeasurementWorkflow` 在初始化时创建 `LogManager`，`log()` 方法同时写入文件；
3. GUI「6. 光谱仪通信」区域显示「保存运行日志到文件」复选框和「日志目录」输入框；
4. GUI 配置变化时自动重建 `LogManager`；
5. 程序关闭时正确关闭 `LogManager`。

## 2. 涉及文件

| 文件 | 变更内容 |
|------|----------|
| `0_measurement_workflow_real_virtual_same_detection_7_25.py` | 导入 `LogManager`；添加配置字段；`MeasurementWorkflow` 初始化/写入/关闭日志管理器；GUI 增加日志保存控件；配置同步 |
| `log_manager.py` | 复用（无修改） |
| `test_log_manager_7_25.py` | 新增：6 项测试 |
| `PLAN_log_port_7_25.md` | 本计划文档 |

## 3. 实施步骤

### 3.1 导入 LogManager

在 7_25.py 文件顶部添加：

```python
from log_manager import LogManager
```

### 3.2 MeasurementConfig 添加字段

```python
# 运行日志实时保存到文件
save_log_to_file: bool = bool(_cfg("save_log_to_file", True))
log_dir: str = str(_cfg("log_dir", "./Log"))
```

### 3.3 MeasurementWorkflow 集成日志管理器

关键点：**`_log_manager` 必须在 `_ensure_summary_xlsx()` 之前初始化**，因为后者会调用 `self.log()`。

初始化位置：

```python
self.output_root = Path(cfg.save_root)
self.output_root.mkdir(parents=True, exist_ok=True)

# 日志管理器（实时保存运行日志到文件）
self._log_manager: Optional[LogManager] = None
self._init_log_manager()
```

方法：

```python
def _init_log_manager(self) -> None:
    if getattr(self.cfg, "save_log_to_file", True):
        log_dir = str(getattr(self.cfg, "log_dir", "./Log"))
        self._log_manager = LogManager(log_dir=log_dir, enabled=True, on_log=None)
        self._log_manager.start()

def _shutdown_log_manager(self) -> None:
    if self._log_manager is not None:
        self._log_manager.stop()
        self._log_manager = None
```

`log()` 方法同时写入 `LogManager`：

```python
def log(self, msg: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now}] {msg}"

    if self._log_manager is not None:
        self._log_manager.log(line)

    if self.on_log is not None:
        self.on_log(line)
    else:
        print(line)
```

`close_all()` 末尾关闭日志管理器。

### 3.4 GUI 增加日志保存控件

在「6. 光谱仪通信」区域增加：

```python
self.save_log_to_file_var = tk.BooleanVar(value=getattr(cfg0, "save_log_to_file", True))
self.log_dir_var = tk.StringVar(value=getattr(cfg0, "log_dir", "./Log"))
```

布局：

```python
log_save_frame = ttk.Frame(tcp_frame)
log_save_frame.grid(row=5, column=0, columnspan=4, padx=2, pady=4, sticky="ew")
log_save_frame.columnconfigure(2, weight=1)
ttk.Checkbutton(log_save_frame, text="保存运行日志到文件", variable=self.save_log_to_file_var).grid(row=0, column=0, padx=(4, 6), pady=2, sticky="w")
ttk.Label(log_save_frame, text="日志目录:").grid(row=0, column=1, padx=(8, 2), pady=2, sticky="e")
ttk.Entry(log_save_frame, textvariable=self.log_dir_var).grid(row=0, column=2, padx=2, pady=2, sticky="ew")
```

注意：新增 `log_save_frame` 后，`tcp_status_var` 和 `tcp_result_var` 行号顺延为 row=6/7，避免重叠。

### 3.5 配置同步

- `build_config_from_ui()` 中传入 `save_log_to_file` 和 `log_dir`；
- `sync_config_from_ui_to_workflow()` 中检测日志配置变化，变化时调用 `_shutdown_log_manager()` + `_init_log_manager()`；
- trace 列表中加入 `save_log_to_file_var` 和 `log_dir_var`。

## 4. 测试计划

| 测试项 | 方法 |
|--------|------|
| LogManager 文件创建 | `test_log_manager_creates_file_7_25` |
| 禁用时不创建文件 | `test_log_manager_disabled_7_25` |
| 多线程安全 | `test_log_manager_thread_safety_7_25` |
| 7_25 MeasurementConfig 字段 | `test_measurement_config_log_params_7_25` |
| 7_25 MeasurementWorkflow 集成 | `test_workflow_log_manager_integration_7_25` |
| 配置变化重建 LogManager | `test_log_config_change_reinitializes_manager_7_25` |

## 5. 测试结果

```text
PASS: log_manager_creates_file_7_25
PASS: log_manager_disabled_7_25
PASS: log_manager_thread_safety_7_25
PASS: measurement_config_log_params_7_25
PASS: workflow_log_manager_integration_7_25
PASS: log_config_change_reinitializes_manager_7_25

All tests passed!
```

## 6. 风险与回退

- 7_25 的 `__init__` 在 `_ensure_summary_xlsx()` 之前就调用 `self.log()`，因此 `_log_manager` 初始化顺序非常关键。已验证当前顺序可正常工作。
- 切换 `log_dir` 时会关闭旧日志文件并新建，运行中切换会截断旧日志；若需保留旧日志，可改为不关闭旧 manager、仅对新目录创建第二个 manager。
