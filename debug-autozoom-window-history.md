# [OPEN] debug-autozoom-window-history

## 症状
`python -m spotzoom_qt_ui` 在构建 AutoZoom 聚焦参考页时崩溃：
`AttributeError: 'SpotZoomQtMainWindow' object has no attribute '_sync_autozoom_window_title_history'`

## 假设
1. `_sync_autozoom_window_title_history` 方法在多次编辑中被覆盖或删除，导致信号连接时找不到。
2. `_load_autozoom_window_title_history` / `_autozoom_window_title_value` 等相关辅助方法也可能被一并覆盖，后续还会继续报相邻缺失方法。
3. AutoZoom 参考页的窗口标题下拉框已创建，但其回调方法未在类作用域内，导致 UI 构建阶段直接失败。
4. 当前 `app.py` 中 AutoZoom 区域存在重复/错位的方法块，可能导致方法定义不完整或缩进不在类内。

## 状态
[OPEN]
