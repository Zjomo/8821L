# 离线视频 ROI 裁剪模块实施计划

## 1. 目标

在 `autofocus_qt_ui` 中新增一个独立的“离线视频 ROI 裁剪”模块，支持：

- 输入本地视频文件；
- 在预览画面上手动框选 ROI；
- 拖动视频进度条，实时观察 ROI 区域的变化；
- 将 ROI 区域逐帧裁剪并输出为新的视频文件。

通过裁剪掉背景区域，减少后续聚焦/检测任务中的背景噪声。

## 2. 输入 / 处理 / 输出

| 阶段 | 说明 |
|------|------|
| 输入 | 本地视频文件（mp4/avi/mov/mkv 等 OpenCV 支持格式） |
| 处理 | 1. 加载视频并显示首帧；2. 用户在首帧/任意帧上框选 ROI；3. 拖动进度条时，实时显示当前帧 ROI 裁剪后的画面；4. 点击“生成 ROI 视频”后，后台逐帧读取、裁剪、写入新视频 |
| 输出 | 新的视频文件（默认与源视频同目录，文件名带 `_roi_<x>_<y>_<w>_<h>` 后缀） |

## 3. 模块设计

### 3.1 核心模块：`video_roi_crop.py`

位置：`Utils/AutoZoom/autofocus_qt_ui/video_roi_crop.py`

职责：

- `VideoRoiCropper` 类封装视频 ROI 裁剪逻辑；
- 提供 `crop_frame(frame, roi)` 静态方法；
- 提供 `crop_video(input_path, output_path, roi, progress_callback, log_callback)` 主流程；
- 自动读取视频编码、帧率、总帧数；
- 输出视频默认使用与输入相同的 fps，编码为 `mp4v`（失败时回退 `XVID`/`MJPG`）。

### 3.2 后台线程：`video_roi_crop_worker.py`

位置：`Utils/AutoZoom/autofocus_qt_ui/video_roi_crop_worker.py`

职责：

- `VideoRoiCropWorker` 继承 `QObject`，通过 Signal 发射进度、日志、完成状态；
- `VideoRoiCropThread` 包装 `QThread`；
- 避免在主线程执行长时间视频写入，防止 UI 卡顿。

### 3.3 UI 集成：`app.py`

在左侧面板新增分组框“离线视频 ROI 裁剪”：

- 输入视频路径 + 浏览按钮；
- 输出视频路径（默认自动生成）+ 浏览按钮；
- “加载视频”按钮；
- “生成 ROI 视频”按钮（主按钮）。

复用右侧的预览、ROI 框选、进度条、播放控件：

- 点击“加载视频”后，自动切换到“本地视频文件”模式，打开视频并显示首帧；
- 框选 ROI 后，调用 `应用 ROI` 即可将 ROI 用于裁剪；
- 拖动进度条时，预览区实时显示 ROI 裁剪后的画面；
- 点击“生成 ROI 视频”后，后台线程执行裁剪，进度条显示进度。

## 4. 测试计划

| 测试项 | 方法 |
|--------|------|
| 单元测试：帧裁剪 | 使用合成视频帧验证裁剪后的尺寸与像素正确性 |
| 单元测试：视频写入 | 生成临时视频，裁剪 ROI，验证输出文件存在、帧数一致、尺寸等于 ROI |
| 单元测试：后台线程 | 验证 Worker 信号发射顺序与进度值递增 |
| 集成测试：UI 状态 | 验证加载视频后控件可用性、ROI 应用后裁剪预览正常 |
| 边界测试 | ROI 越界时自动Clamp；空ROI时给出提示并不生成视频 |

测试文件：`tests/test_video_roi_crop.py`

## 5. 实施步骤

1. 创建 `video_roi_crop.py` 核心裁剪逻辑；
2. 创建 `video_roi_crop_worker.py` 后台工作线程；
3. 在 `app.py` 中新增 UI 分组、信号连接与状态管理；
4. 实现 ROI 裁剪实时预览；
5. 编写并运行 `tests/test_video_roi_crop.py`；
6. 更新 `README.md` 说明新模块用法；
7. 走查代码，确保线程安全与资源释放。
