# [OPEN] debug-autozoom-picomotor-reconnect

## 症状
AutoZoom Z 轴第一次移动成功，随后反向/再次移动失败：
`error connecting to the Picomotor controller`

## 假设
1. AutoZoom 每次移动都新建 `ZAxisController` 并重新连接，第一次连接后设备资源未充分释放，导致后续连接失败。
2. Newport/Picomotor 8742 同一时间只能被一个对象持有，Picomotor 调试面板或其他模块连接后未释放，AutoZoom 再连接冲突。
3. `ZAxisController.move_relative()` 完成后没有显式断开或清理句柄，导致下一次连接失败。
4. AutoZoom 应该复用一个持久化 Z 轴控制器，而不是每次按钮点击都重新 connect。
5. 连接失败需要在 UI 层提供“重新连接/断开”逻辑，避免重复创建驱动实例。

## 状态
[OPEN]
