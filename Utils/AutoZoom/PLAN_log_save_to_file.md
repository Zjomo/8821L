# 运行日志实时保存功能实施计划

## 1. 目标

将项目运行日志实时保存至 `./Log` 文件夹，支持：

1. **实时写入**：每条日志立即写入文件，而非批量写入
2. **线程安全**：支持多线程并发写入
3. **GUI 可配置**：提供开关和目录配置
4. **自动创建目录**：日志目录不存在时自动创建
5. **统计信息**：提供已写入行数、字节数等统计

## 2. 涉及文件

| 文件 | 变更内容 |
|------|----------|
| `Utils/AutoZoom/log_manager.py` | 新增 LogManager 类 |
| `Utils/AutoZoom/0_measurement_workflow_real_virtual_same_detection_7_23.py` | MeasurementConfig 添加日志参数；MeasurementWorkflow 集成 LogManager |
| `Utils/AutoZoom/test_log_manager.py` | 新增测试用例 |

## 3. 实施步骤

### 3.1 LogManager 类设计

```python
class LogManager:
    """
    日志管理器：实时保存日志到 ./Log 目录。

    功能：
      1. 线程安全的日志写入
      2. 自动按日期分割日志文件
      3. 支持多订阅者（同时写入文件和回调）
      4. 日志文件自动创建与轮转
    """

    def __init__(self, log_dir="./Log", enabled=True, on_log=None):
        self.log_dir = Path(log_dir)
        self.enabled = enabled
        self.on_log = on_log
        self._lock = threading.Lock()
        self._file_handle = None
        self._current_log_path = None
        self._current_date = ""
        self._started = False

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def log(self, message: str) -> None: ...
    def get_log_path(self) -> Optional[Path]: ...
    def get_stats(self) -> dict: ...
```

### 3.2 MeasurementConfig 新增参数

```python
# 运行日志实时保存到文件
save_log_to_file: bool = bool(_cfg("save_log_to_file", True))
log_dir: str = str(_cfg("log_dir", "./Log"))
```

### 3.3 MeasurementWorkflow 集成

1. **初始化顺序调整**：LogManager 必须在调用 `log()` 之前初始化
2. **log 方法修改**：同时写入 LogManager 和原有回调
3. **关闭处理**：在 `close_all()` 中关闭 LogManager

```python
def log(self, msg: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now}] {msg}"

    # 写入日志管理器（实时保存到文件）
    if self._log_manager is not None:
        self._log_manager.log(line)

    # 原有回调或打印
    if self.on_log is not None:
        self.on_log(line)
    else:
        print(line)
```

### 3.4 GUI 配置控件

在「光谱仪通信」模块添加：

- `保存运行日志到文件` 复选框
- `日志目录` 输入框

配置同步：在 `start_tcp()` 中同步日志配置到 `workflow.cfg`

## 4. 测试计划

| 测试项 | 方法 |
|--------|------|
| LogManager 文件创建 | 创建 LogManager，写入日志，验证文件存在且内容正确 |
| LogManager 禁用模式 | enabled=False，验证不创建文件 |
| LogManager 统计信息 | 写入多条日志，验证 total_lines/total_bytes 正确 |
| 线程安全写入 | 多线程并发写入，验证无数据丢失 |
| 上下文管理器 | 验证 with 语句正确启动/关闭 |
| 回调函数 | 验证 on_log 回调被正确调用 |
| MeasurementConfig 参数 | 验证 save_log_to_file/log_dir 字段存在 |
| MeasurementWorkflow 集成 | 创建 workflow，调用 log()，验证文件内容 |

## 5. 测试结果

所有测试通过：

```text
PASS: log_manager_creates_file
PASS: log_manager_disabled
PASS: log_manager_stats
PASS: log_manager_thread_safety
PASS: log_manager_context_manager
PASS: log_manager_callback
PASS: measurement_config_log_params
PASS: workflow_log_manager_integration
PASS: log_manager_creates_directory

All tests passed!
```

## 6. 使用方式

### 6.1 默认行为

程序启动后，日志自动保存到 `./Log` 目录，文件名格式：

```
log_YYYYMMDD_HHMMSS.txt
```

### 6.2 GUI 配置

在「光谱仪通信」模块：

- 勾选/取消「保存运行日志到文件」开关
- 修改「日志目录」路径（如 `./measurement_output/Log`）

### 6.3 配置文件

在 `config_angle_repair_fixed.py` 中添加：

```python
DEFAULT_CONFIG["save_log_to_file"] = True
DEFAULT_CONFIG["log_dir"] = "./Log"
```

## 7. 风险与回退

- **初始化顺序**：LogManager 必须在 `_ensure_summary_xlsx()` 之前初始化，否则 `log()` 调用会失败。已修复。
- **磁盘空间**：长期运行可能产生大量日志文件，建议定期清理。
- **权限问题**：如果日志目录不可写，LogManager 会禁用自身并打印警告。