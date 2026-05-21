## 课题7：基于2轴的MRC主动激光束稳定系统【2天】

基于拉普拉斯方差/目标检测的光斑自动准直系统 -- 将激光自动准直压电落地

1、背景

```
激光束自动对准（或自动准直）

是指利用光学传感器、电机驱动器和控制系统，使激光束的位置、角度或焦点自动保持在预定轨迹上的技术。它广泛应用于工业加工、科研实验与量子计算中。

核心工作原理系统主要通过“检测-反馈-调整”的闭环机制来运作：
	光束检测：使用位置敏感探测器（PSD）或光电二极管阵列实时捕捉激光束的位置和指向偏移。
	信号分析：将光斑偏差数据传输至控制器（如基于 FPGA 的系统或计算机），计算出所需的补偿量。
	闭环调整：系统驱动快速反射镜（FSM）或电动调整架进行微小角度补偿，将光束精确对准目标（如光纤耦合、光学腔或靶点）。

主要应用场景激光加工（切割/打标）：通过光束自动对准，确保聚焦头在加工复杂曲面时长焦距保持稳定，精度通常可达毫米或微米级。

```

2、原理

```
通俗来讲就是 
激光本来是打在一个固定的点上，但是因为热漂移、振动、机械松动等因素，
探测器会发现"光点的位置变了"，现在通过控制器驱动压电电机让转向镜轻微抓弄，将光点拨回原来的位置

```

3、硬件：

​	激光器是否能用	✔

​	准直探测器 -- 光束质量分析仪 -- 确定型号、说明书 + 软件 + 代码 		刘师兄帮忙解决

​	光路安装规划 -- 主要关注探测器的位置						尹师姐与朱师姐帮忙解决	

​	8742驱动器，还需要一根连接设备的USB线 -- 5个电机 -- 两个镜头分别有2个电机，Z轴一个

4、软件：

​	如何理解4轴的准直原理？	✔					两轴会实现光点在一条直线上某一点的准直，但是4轴能实现光点在整条直线上的准直

​	准备5个模拟的数据输入接口，分别给5个电机准备

​	探索后续可待优化的系统部分

​	数据集采集问题 -- 光斑检测算法优化

5、参考链接：

https://www.surisetech.com/mrc-systems-active-laser-beam-stabilization/

https://www.auniontech.com/index.php/details-307.html

https://www.lbtek.com/product/328?fid=184&sid=266&tid=

6、冲冲冲

（1）思路

```
采图 -> 检测光斑 -> 算偏差 -> 驱动 XY/Z -> 再检测 -> 直到收敛 的闭环控制程序，
SpotZoom_Machine_Learning 脚本中的大量模块主要是给这个主流程做增强、兜底和质量把关。

```

（2）数据流

```
run_spotzoom.bat
  -> SpotZoom.py main()
    -> parse_args / check_environment
    -> 选择检测后端(YOLO / classic / worker)
    -> 连接图像源(ToupView窗口 / 模拟图)
    -> 连接执行器(XY台 / Z轴)
    -> 组装 AlignmentConfig
    -> 创建 SpotZoomController
    -> run():
         1) 先找初始目标点 P1
         2) Z 上移，检测 P2，XY 对齐到 P1
         3) Z 下移，检测 P3，XY 对齐到 P1
         4) 判断 P1/P2/P3 是否收敛
         5) 不收敛就继续下一轮
    -> 记录报告、释放硬件、释放锁文件

```

（3） 启动与参数配置

```
调用SpotZoom.py，通过 main() 先解析参数 parse_args() ，再做环境检查check_environment() ，
之后组装 4 个核心对象：
图像源 window、检测器 detector、XY 执行器 xy_stage、Z 执行器 z_stage
相关参数集中与 `AlignmentConfig`，作为系统的配置中心。

```

（4）图像采集层【实机模式 & 仿真模式】

```
实机模式：`ToupViewWindow` 
找到 ToupView 窗口 > 截屏抓图 > 手动选 ROI > 鼠标滚轮控制 Z 方向缩放

仿真模式：`SimulatedFrameWindow` 
从本地图片生成测试帧 > 可加抖动、噪声 > 主要用于离线冒烟测试

```

（5）检测层（3种）

```
YOLO 微调：`SpotYOLODetector` 
OpenCV 检测：`SpotClassicDetector` 
外部 Python 子进程跑 YOLO：`SpotYOLOWorkerClient`

后端选择逻辑在 `build_detector()` 
auto 优先 YOLO
YOLO 不满足条件就回退 classic
yolo_python 时会走 worker 子进程模式 `worker_main()`

```

（6）控制主循环

```
核心控制器是 `SpotZoomController`，主循环在 `run()` 

1. 先检测一次，定义初始目标点 P1
2. 每轮迭代：
   - Z 上移 -> 得到 P2 
   - 调用 `_align_to_target()` 把 P2 拉回 P1
   - Z 下移 -> 得到 P3 
   - 再把 P3 拉回 P1
3. 调用 `_is_converged()`判断 P2、P3 是否都回到 P1 附近
4. 收敛则结束，否则继续下一轮

```

（7）单次检测与兜底链【对该项目中的分级处理进行筛选与优化 -- 主检测 + 多级补救】

```
`_detect_center_with_retry()`单次检测流程大致是：

1. 抓一帧
2. 预处理 `_prepare_detection_frame()`
   - 可选 v7 去噪
   - v26 图像增强
   - 多帧去噪
   - flow restoration

3. 主检测 `_detect_with_runtime_metrics()` 
4. 检测后再做一串筛选/增强：
   SAM2 精修、v26 像差门控、光学质量门控、焦点评分 _focus_score、置信度门控、不确定性门控、domain probe、平滑 _smooth_detection

若主检测失败，会按顺序走备用链，大致宝库是：
	LodeSTAR fallback、StarDist fallback、memory recover、v27 wavelet fallback、Kalman template recovery、KLT 光流 recovery、pyramid template recovery、log-polar recovery、phase correlation recovery、ECC recovery、CMC affine recovery、ORB homography recovery、emporal ensemble recovery、buffered reuse、recovery scan 等检测方法

```

（8）执行层：怎么把像素误差变成电机动作

```
- XY 步数计算在 `_calc_axis_steps()` 
  - 可以固定步长、可以自适应步长、也可以启用 PID
- 执行动作在 `_align_to_target()` 
  - 算 dx/dy、过安全检查、调 xy_stage.move_x/move_y、等待稳定，再重检

XY 驱动有 3 类：
- `DryRunStage` 、`ThorlabsXYStage` 、`NewportXYStage` 、构造入口在 `build_xy_stage()` 

Z 轴也有 3 类：
- `DryRunZAxis`、`ToupViewWheelZAxis` 、`XPSZAxis` 、构造入口在 `build_z_stage()` 

```

（9）保护与记录层 与 清理层关闭

```
- 防并发锁：`RunLockFile，防止两次程序同时抢硬件
- 运行记录：`RunReporter` ，记录事件流、检测耗时，并输出 JSON 报告
- 清理层 会统一释放：controller / detector，XY/Z 驱动，OpenCV 窗口，run lock 见main()末尾 
- 控制器内部还会批量 reset/close 各种可选 ML 模块，见 `SpotZoomController.close()`

```

（10）整体流程总结

````
窗口取图/模拟取图 -> 主检测器找光斑 -> 一串质量与稳定性筛选 -> 失败时多级恢复 -> 把像素偏差换成 XY/Z 动作 -> 重复闭环直到 P1/P2/P3 收敛 -> 记录报告并安全退出

````







### 🎯 TODO

复现项目，完成各种bug，直至跑通

将项目上传至github				✔			

恢复项目之前的创新优化模块		✔

找到控制驱动器位移移动的逻辑部分【方便后续调试】

更新日志非中文的问题				✔

更新中文乱码问题					✔

在文件内容补充一个虚拟环境文件，保证项目可以正常运行【当前使用的是cp11_torch22】

将所有设备的通信协议封装成Python函数，即可随时随地进行模拟









## 课题8：基于4轴的MRC主动激光束稳定系统【2天】


## 课题7：基于2轴的MRC主动激光束稳定系统【2天】

基于拉普拉斯方差/目标检测的光斑自动准直系统 -- 将激光自动准直压电落地

1、背景

```
激光束自动对准（或自动准直）

是指利用光学传感器、电机驱动器和控制系统，使激光束的位置、角度或焦点自动保持在预定轨迹上的技术。它广泛应用于工业加工、科研实验与量子计算中。

核心工作原理系统主要通过“检测-反馈-调整”的闭环机制来运作：
	光束检测：使用位置敏感探测器（PSD）或光电二极管阵列实时捕捉激光束的位置和指向偏移。
	信号分析：将光斑偏差数据传输至控制器（如基于 FPGA 的系统或计算机），计算出所需的补偿量。
	闭环调整：系统驱动快速反射镜（FSM）或电动调整架进行微小角度补偿，将光束精确对准目标（如光纤耦合、光学腔或靶点）。

主要应用场景激光加工（切割/打标）：通过光束自动对准，确保聚焦头在加工复杂曲面时长焦距保持稳定，精度通常可达毫米或微米级。

```

2、原理

```
通俗来讲就是 
激光本来是打在一个固定的点上，但是因为热漂移、振动、机械松动等因素，
探测器会发现"光点的位置变了"，现在通过控制器驱动压电电机让转向镜轻微抓弄，将光点拨回原来的位置

```

3、硬件：

​	激光器是否能用	✔

​	准直探测器 -- 光束质量分析仪 -- 确定型号、说明书 + 软件 + 代码 		刘师兄帮忙解决

​	光路安装规划 -- 主要关注探测器的位置						尹师姐与朱师姐帮忙解决	

​	8742驱动器，还需要一根连接设备的USB线 -- 5个电机 -- 两个镜头分别有2个电机，Z轴一个

4、软件：

​	如何理解4轴的准直原理？	✔					两轴会实现光点在一条直线上某一点的准直，但是4轴能实现光点在整条直线上的准直

​	准备5个模拟的数据输入接口，分别给5个电机准备

​	探索后续可待优化的系统部分

​	数据集采集问题 -- 光斑检测算法优化

5、参考链接：

https://www.surisetech.com/mrc-systems-active-laser-beam-stabilization/

https://www.auniontech.com/index.php/details-307.html

https://www.lbtek.com/product/328?fid=184&sid=266&tid=

6、冲冲冲

（1）思路

```
采图 -> 检测光斑 -> 算偏差 -> 驱动 XY/Z -> 再检测 -> 直到收敛 的闭环控制程序，
SpotZoom_Machine_Learning 脚本中的大量模块主要是给这个主流程做增强、兜底和质量把关。

```

（2）数据流

```
run_spotzoom.bat
  -> SpotZoom.py main()
    -> parse_args / check_environment
    -> 选择检测后端(YOLO / classic / worker)
    -> 连接图像源(ToupView窗口 / 模拟图)
    -> 连接执行器(XY台 / Z轴)
    -> 组装 AlignmentConfig
    -> 创建 SpotZoomController
    -> run():
         1) 先找初始目标点 P1
         2) Z 上移，检测 P2，XY 对齐到 P1
         3) Z 下移，检测 P3，XY 对齐到 P1
         4) 判断 P1/P2/P3 是否收敛
         5) 不收敛就继续下一轮
    -> 记录报告、释放硬件、释放锁文件

```

（3） 启动与参数配置

```
调用SpotZoom.py，通过 main() 先解析参数 parse_args() ，再做环境检查check_environment() ，
之后组装 4 个核心对象：
图像源 window、检测器 detector、XY 执行器 xy_stage、Z 执行器 z_stage
相关参数集中与 `AlignmentConfig`，作为系统的配置中心。

```

（4）图像采集层【实机模式 & 仿真模式】

```
实机模式：`ToupViewWindow` 
找到 ToupView 窗口 > 截屏抓图 > 手动选 ROI > 鼠标滚轮控制 Z 方向缩放

仿真模式：`SimulatedFrameWindow` 
从本地图片生成测试帧 > 可加抖动、噪声 > 主要用于离线冒烟测试

```

（5）检测层（3种）

```
YOLO 微调：`SpotYOLODetector` 
OpenCV 检测：`SpotClassicDetector` 
外部 Python 子进程跑 YOLO：`SpotYOLOWorkerClient`

后端选择逻辑在 `build_detector()` 
auto 优先 YOLO
YOLO 不满足条件就回退 classic
yolo_python 时会走 worker 子进程模式 `worker_main()`

```

（6）控制主循环

```
核心控制器是 `SpotZoomController`，主循环在 `run()` 

1. 先检测一次，定义初始目标点 P1
2. 每轮迭代：
   - Z 上移 -> 得到 P2 
   - 调用 `_align_to_target()` 把 P2 拉回 P1
   - Z 下移 -> 得到 P3 
   - 再把 P3 拉回 P1
3. 调用 `_is_converged()`判断 P2、P3 是否都回到 P1 附近
4. 收敛则结束，否则继续下一轮

```

（7）单次检测与兜底链【对该项目中的分级处理进行筛选与优化 -- 主检测 + 多级补救】

```
`_detect_center_with_retry()`单次检测流程大致是：

1. 抓一帧
2. 预处理 `_prepare_detection_frame()`
   - 可选 v7 去噪
   - v26 图像增强
   - 多帧去噪
   - flow restoration

3. 主检测 `_detect_with_runtime_metrics()` 
4. 检测后再做一串筛选/增强：
   SAM2 精修、v26 像差门控、光学质量门控、焦点评分 _focus_score、置信度门控、不确定性门控、domain probe、平滑 _smooth_detection

若主检测失败，会按顺序走备用链，大致宝库是：
	LodeSTAR fallback、StarDist fallback、memory recover、v27 wavelet fallback、Kalman template recovery、KLT 光流 recovery、pyramid template recovery、log-polar recovery、phase correlation recovery、ECC recovery、CMC affine recovery、ORB homography recovery、emporal ensemble recovery、buffered reuse、recovery scan 等检测方法

```

（8）执行层：怎么把像素误差变成电机动作

```
- XY 步数计算在 `_calc_axis_steps()` 
  - 可以固定步长、可以自适应步长、也可以启用 PID
- 执行动作在 `_align_to_target()` 
  - 算 dx/dy、过安全检查、调 xy_stage.move_x/move_y、等待稳定，再重检

XY 驱动有 3 类：
- `DryRunStage` 、`ThorlabsXYStage` 、`NewportXYStage` 、构造入口在 `build_xy_stage()` 

Z 轴也有 3 类：
- `DryRunZAxis`、`ToupViewWheelZAxis` 、`XPSZAxis` 、构造入口在 `build_z_stage()` 

```

（9）保护与记录层 与 清理层关闭

```
- 防并发锁：`RunLockFile，防止两次程序同时抢硬件
- 运行记录：`RunReporter` ，记录事件流、检测耗时，并输出 JSON 报告
- 清理层 会统一释放：controller / detector，XY/Z 驱动，OpenCV 窗口，run lock 见main()末尾 
- 控制器内部还会批量 reset/close 各种可选 ML 模块，见 `SpotZoomController.close()`

```

（10）整体流程总结

````
窗口取图/模拟取图 -> 主检测器找光斑 -> 一串质量与稳定性筛选 -> 失败时多级恢复 -> 把像素偏差换成 XY/Z 动作 -> 重复闭环直到 P1/P2/P3 收敛 -> 记录报告并安全退出

````

（11）模拟测试脚本

```
# 生成一张模拟光斑图
python -c "from pathlib import Path; import cv2, numpy as np; p=Path('__tmp_spot.jpg'); img=np.zeros((256,256,3), dtype=np.uint8); cv2.circle(img,(128,128),20,(255,255,255),-1); img=cv2.GaussianBlur(img,(0,0),4.0); cv2.imwrite(str(p), img)"

# 环境检查
python SpotZoom.py --check-env --frame-source-image __tmp_spot.jpg --xy-driver dryrun --z-driver dryrun --detector-backend classic --skip-roi --disable-run-lock

# 最小冒烟测试
python SpotZoom.py --frame-source-image __tmp_spot_offcenter.jpg --xy-driver dryrun --z-driver dryrun --detector-backend classic --skip-roi --disable-run-lock --max-iterations 1


# 标准模拟自动准直
python SpotZoom.py --frame-source-image __tmp_spot_offcenter.jpg --xy-driver dryrun --z-driver dryrun --detector-backend classic --skip-roi --disable-run-lock --max-iterations 50

# 启用当前这批统一模块的模拟流程
python SpotZoom.py --frame-source-image __tmp_spot.jpg --xy-driver dryrun --z-driver dryrun --detector-backend classic --skip-roi --disable-run-lock --max-iterations 1 --frontier-v6-diffusion-preprocess --frontier-v6-multiscale-fallback --frontier-v7-denoiser --frontier-v7-phase-refine --frontier-temporal-ensemble --cl-ao --fourier-psf --dip-enhancer --beam-propagator --dd-mpc --lqg --slm-generator-enabled --ao-pipeline-enabled --strehl-assessor-enabled --hal-enabled --laplacian-autofocus-enabled --synthetic-data-enabled --dm-calibrator-enabled --sys-identifier-enabled --frontier-v4-bundle

# 带扰动的压力测试
python SpotZoom.py --frame-source-image __tmp_spot.jpg --sim-jitter-px 6 --sim-noise-std 8 --xy-driver dryrun --z-driver dryrun --detector-backend classic --skip-roi --disable-run-lock --max-iterations 3 --frontier-v6-diffusion-preprocess --frontier-v6-multiscale-fallback --frontier-v7-denoiser --frontier-v7-phase-refine --frontier-temporal-ensemble

# 保存运行记录
python SpotZoom.py --frame-source-image __tmp_spot.jpg --xy-driver dryrun --z-driver dryrun --detector-backend classic --skip-roi --disable-run-lock --max-iterations 1 --run-report-json spotzoom_sim_report.json --event-stream-jsonl spotzoom_sim_events.jsonl --log-level DEBUG

```

（12）真实设备测试脚本

```
# 先只查环境，不动硬件【Otsu阈值法+brightest候选：自动把亮斑从背景里抠出来，再挑最亮的当作真正的光斑】
# --xy-driver thorlabs|newport 		【x,y轴 选用thorlabs|newport】
# --newport-conn --newport-x-axis --newport-y-axis 

# --z-driver xps|wheel				【z轴 选用xps|wheel】
# --xps-ip --xps-port --xps-group

python SpotZoom.py --check-env --xy-driver dryrun --z-driver dryrun --detector-backend classic --disable-run-lock


# 真实相机 + XY/Z 全 dryrun
python SpotZoom.py --xy-driver dryrun --z-driver dryrun --detector-backend classic --max-iterations 1 --log-level DEBUG --run-report-json real_cam_smoke.json --event-stream-jsonl real_cam_smoke.jsonl


# 真实 XY + Z 还 dryrun
python SpotZoom.py --xy-driver <thorlabs|newport> --z-driver dryrun --detector-backend classic --max-iterations 1 --log-level DEBUG


# 全真机
python SpotZoom.py --xy-driver <thorlabs|newport> --z-driver <wheel|xps> --detector-backend classic --max-iterations 1 --log-level DEBUG


# 如果要带上这批统一模块，再在第4步"全真机"后加
--frontier-v6-diffusion-preprocess --frontier-v6-multiscale-fallback --frontier-v7-denoiser --frontier-v7-phase-refine --frontier-temporal-ensemble


```

（13）电机设备驱动函数 更新

```
电机分配策略：
现在我准备设计一个4轴的MRC主动激光束稳定系统，其中：
8742cl控制器分别控制2个8821L电机的4个轴，分别控制两块镜片；
（2轴会实现光点在一条直线上某一点的准直，但是4轴能实现光点在整条直线上）
而Z轴再额外用一个8742cl控制器进行驱动，控制镜头的变焦。

参考"8742控制器封装函数.txt"，进行驱动函数的更新，并保证代码通过测试，进行一遍系统处理流程。



```







### 🎯 TODO

## 2026-5-21

复现项目，完成各种bug，直至跑通

找到控制驱动器位移移动的逻辑部分【方便后续调试】

在文件内容补充一个虚拟环境文件，保证项目可以正常运行【当前使用的是cp11_torch22】

将所有设备的通信协议封装成Python函数，即可随时随地进行模拟









## 2026-5-20

简单任务用trae，复杂优化任务用codex							✔

main(本地分支)  到 origin/main(github)，实现push与fetch			✔

将项目上传至github									   		✔			

恢复项目之前的创新优化模块							    		✔

更新日志非中文的问题，将英文统一更新为中文，更新中文乱码问题	   ✔		

完成每个项目文件/文件夹的理解									✔										

统一所有"综合检测报告"，整合为一个最终的统一文件				    ✔

各版本优化模块，按类型统一进行函数封装，并对每个优化模块，用一个.md文件进行说明，将SpotZoom_Machine_Learning_v*中的模块进行统一封装【将项目各创新模块整合，合并各个重复功能的文件夹】	✔			

统一各设备封装函数【XPS/8742CL/8743CL/XPS/信号发生器/脉冲激光器】	✔







## 课题8：基于4轴的MRC主动激光束稳定系统【2天】

