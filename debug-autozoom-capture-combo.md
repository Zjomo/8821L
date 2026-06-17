# [OPEN] debug-autozoom-capture-combo

## 症状
点击 AutoZoom 聚焦参考页中的“测试窗口采集”时，报错：
`[AutoZoom] 测试采集失败: 'PySide6.QtWidgets.QComboBox' object has no attribute 'text'`

## 假设
1. `测试窗口采集` 相关代码把 `QComboBox` 当作 `QLineEdit` 使用，错误调用了 `.text()`。
2. 该页其他位置（建立参考 / 检查补焦）也存在同样的 `QComboBox.text()` 误用，修复一个后还会在相邻函数继续报错。
3. 窗口标题输入框虽然是可编辑 `QComboBox`，但当前取值函数 `_autozoom_window_title_value()` 可能未被统一复用。
4. 该错误只影响 AutoZoom 聚焦参考页的窗口采集路径，不影响 UI 主体启动。

## 状态
[OPEN]
