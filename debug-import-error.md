# [OPEN] debug-import-error

## 问题
`python -m spotzoom_qt_ui` 启动时报 `ImportError: cannot import name 'RuntimeControlService' from 'spotzoom_qt_ui.services'`

## 假设
1. `services.py` 中 `RuntimeControlService` 定义被意外删除或缩进错位，导致模块导入时找不到该类。
2. `services.py` 中出现了错误的重复插入/截断，造成模块级结构损坏，`app.py` 的导入链被中断。
3. `services.py` 在导入阶段抛出了其他异常，Python 最终表现为无法导入 `RuntimeControlService`。
4. 之前加入的 AutoZoom 模块接入代码误伤了 `services.py` 的原始结构，导致服务类不完整。

## 证据记录
- `spotzoom_qt_ui/services.py` 顶部存在 `RuntimeControlService` 相关方法被错误挂到 `ModuleCatalogService` 的结构损坏。
- `RuntimeControlService` 类头已恢复，`python -c "import spotzoom_qt_ui"` 现已成功。

## 处理记录
- 恢复 `RuntimeControlService` 类定义。
- 补回 `_shorten_payload()`。
- 保留 AutoZoom 模块目录集成。
