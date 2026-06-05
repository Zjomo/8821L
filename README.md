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

#### 2026-6-5

1、完成探测器与4轴电机，在距离和方向上的正确映射							✔

```
> mirror1 移动对单探测器的光斑位置没有反应，说明mirror1 仅作为半透半反，
	不参与校正（参与第二个探测器的校正，这在双探测器系统中生效）
> mirror2 移动两轴会有对应XY的反应，说明mirror2 参与该探测器的校正

```

2、引入标定步长的配置 + 暂停标定的按钮，实现动态标定（步长控制变量）		  ✔

3、当前设定的是像素变化>0.5px，才会进入标定校准的步骤						✔

```
> 故设定步长=2000，可稳定触发

```

4、标定过程中，X轴和Y轴在探测器中的表现为mirror2_x/u 对应的是y/x轴在移动 	  ✔

```
设置X轴的校准量 与 mirror2_y 映射；
设置y轴的校准量 与 mirror2_x 映射。

```



#### 2026-6-4

1、完成单探测器的模拟抖动和校准逻辑

2、统一4轴控制器与两个光镜（4轴）的对应关系

```
将控制器的轴1、轴2 指定为mirror1
将控制器的轴3、轴4 指定为mirror2

```

3、将"UCC实时预览图像" 补充道 "准直工作台"的"实时预览图"界面中，同时优化参数配置窗口



#### 2026-6-1

默认参数需要优化：Z轴默认不启用、模拟模式下，默认使用4轴模式、默认x、y的移动步长为5	✔

要求在ROI框定之后，程序一直保持运行的状态，期间不断检测探测器光斑、显微镜关班的准直情况	✔

在UI系统中，设置4轴单探测器模式，以及5轴（加Z轴）模式	✔

在系统进行校准的过程中，将ROI选中的中间帧，保存至”./FrameTmp“文件夹，同时将视频帧相关的配置，补充到UI系统上；	✔

将系统所有相关的默认配置，先新增一个config.py，在将配置文件补充到config.py中		✔

UCC 探测器通信：当前似乎还没有涉及到UCC探测器的通信 	✔

```
-- 通信成功，但是因为电压问题，屏幕全黑没有成像
-- 当前可以检测到UCC的分辨率和像素范围，继续优化，完成UCC探测器的实时预览画面
-- 成功完成了 UCC的相关通信，但是仅仅是“缩略版”的光斑显示
-- 补充calcult、preview、Curve三种嵌套模式（当前已实现preview模式）

```



#### 2026-5-30

四轴双探测器 -- 闭环思路	✔

【前提：光斑需要手动预准直，保持动态规划准直的状态，且符合所有市面系统的应用背景】

```
 首先，启动时拍一张照片image_0，分两种情况：
     （1）未检测到光斑（未手动粗调，暂时先不考虑，但后续可考虑自动找光斑）
     	记录miss，连续3次失败，重新捕获目标点
     （2）检测到光斑（已手动粗调，再自动化细调）
         对image_0 进行去噪、多尺度增强、SAM2分割精化、像差门控、光学质量门控等帧预处理，得：image_0_pro
         进行灰度化，得：image_0_pro_gray，
         基于classic 进行光斑检测，实现二值化（otsu）、形态学去噪、轮廓提取、轮廓筛选、多光斑选择（brighten），
         再完成质心定位（分为：粗定位、亚像素精化），
         进行质量门控（最多6次detect_retry=6），顺利满足质量评分阈值后，进行下一步
         输出光斑得位置目标点、角度目标点Target_0（作为后续所有 e_pos的起始点）
         
 随后，进入闭环迭代循环：
     双探测器同步拍照 → 根据classic+octu+btighten算法栈，检测当前光斑中心，
     计算误差 e_pos、e_ang，                               
     若误差范数 ≤ 4px 且在滑动窗口中 75% 以上帧满足则判定收敛退出；           
     否则通过 PI 控制律（含 dt 积分、振荡检测、饱和保护）计算 4轴电机的控制量 u1_x, u1_y, u2_x, u2_y，
     驱动电机完成移动（宏观位置准直、积分位置微调准直、角度准直），     
     先经 Jacobian 逆矩阵解耦消除轴间串扰，                             
     再限幅后下发到 Newport MRC-4 控制器，                           
     驱动两个反射镜的 4 个压电促动器微调角度，                              
     等待 0.35 秒机械稳定，                                          
     继续视觉成像，最多8次迭代实现"光斑准直损失函数"收敛，直到光束恢复准直。
     
 最后，光斑准直的"收敛判定条件"【项目采用市面上常见的第二种】：
     （1）5轴三点共轴判据
     （2）4轴双镜，根据：
         误差范式（光斑中心与目标点的“位置+角度的欧几里得范数”） & 
         滑动窗口比例阈值（如75%：10帧中至少有8帧满足条件）
     （3）探测器对比模式，根据：
         单一误差范数（光斑中心与目标点的欧式距离 < n个像素） & 
         连续稳定帧计数（如5frames：连续5帧满足，即收敛）
 
```

UCC光束检测器通信代码补充	✔



#### 2026-5-29

完成 "4轴双探测器的显微镜光斑准直系统" 的初步逻辑闭环	✔

```
 基于官方 MRC 主动光束稳定页的两级闭环思路 + dual_detector_4axis架构
 按“两级观测 + 两级执行 + 统一耦合解耦”来做
 
 
 实现逻辑
 1. 硬件分工
    - Detector1 负责“位置误差” e_pos
    - Detector2 负责“角度误差” e_pos
    - Mirror1 X/Y 主要吃 e_pos
    - Mirror2 X/Y 主要吃 e_pos
    - 4 个轴不是各干各的，而是通过标定矩阵解耦后协同工作
 
 2. 控制律【根据当前的误差，如何计算应该输出的控制量 -- 误差到执行器的映射函数】
    - e_pos = [dx1, dy1]
    - e_ang = [dx2, dy2]
    - u1 = Kp1e_pos + Ki1∫e_pos + C12e_ang
    - u2 = Kp2e_ang + Ki2∫e_ang + C21e_pos
    - 初期先把 C12/C21 = 0，只跑主通道，稳定后再补耦合项
 
 3. 启动流程
    - 先机械粗对准，让两个探测器都能看到光斑
    - 记录两路中心点作为零位
    - 小脉冲测试 4 个轴的正负方向
    - 测一次 4x4 灵敏度矩阵 J
    - 检查是否饱和是否跑偏是否存在明显串扰
 
 4. 闭环流程
    - 同步采集两路图像、检测两个中心点和置信度
    - 计算 e_pos / e_ang，滤波门控收敛判定，下发镜面修正，等待稳定后再采样
    - 连续 N 帧都满足阈值，判定收敛
 
 5. 异常处理
    - 检测不到光斑：走重捕获或 fallback
    - 信号饱和：降曝光/减增益/缩 ROI
    - 来回振荡：降 Kp，增大 settle time
    - 方向反了：翻转对应轴符号
    - 串扰明显：重新做 J 标定
 
 落到你这个项目里，直接对应的入口就是：
 - SpotZoom.py 里的 --alignment-strategy dual_detector_4axis
 - --detector-mode dual_detector
 - --frame-source-image / --frame-source-image-2
 - --stage1-kp/ki--stage2-kp/ki
 - --coupling-c12--coupling-c21
 - spotzoom_qt_ui 已经在 UI 层把这些参数编排好了
 
 实际操作建议
 - 先用 dryrun + classic 把闭环逻辑跑通
 - 再接真机
 - 先只开 stage1，再开 stage2
 - 最后再把耦合补偿打开
 
 
```



#### 2026-5-28

补充了双 CCD探测器架构	✔

更新了实际光路架构		✔

继续完成光路的准直测试	✔





#### 2025-5-27

项目UI兼容win7版本	✔

解决UI/代码字符乱码问题	✔

UI - 驱动器测试	✔

UI - NIS窗口测试	✔

UI - 补充准直缓存视频帧	✔

UI - 更新英文界面为中文（除补充专用名词）	✔

UI - 给Z轴补充了一个禁用选项	✔





#### 2026-5-26

网购了另一个光束分析仪，准备2个探测器 -- 咨询发票、质保、测试不行可退货、400-800nm波长、光斑大小<5mm	✔





#### 2026-5-25

1、在305设备，部署git，随时push 8821L项目的最新版本，指定：	✔

```
实现 github实现项目的版本控制，晚点直接删除slave的8821L项目，再从github上 pull下来：
    git_email:	1411xxxxxx@qq.com
    git_name:	jomxx
```

2、在现有的8742控制器基础上，再补充一个8742控制器，两者实现协同控制【Z轴优化掉】		✔

3、解决两个电机的问题，相互完成通讯【多个电机之间完成通信，同理，将Z轴优化掉】			✔

4、每次都需要进行数据集采集的问题，解决光斑数据不通用的问题 -- 用classic算法应对			✔

5、otsu、adaptive、tophat、gmm、meanshift、brightest、largest、central算法复习			✔

```
光斑二值化：
otsu：基于"背景"和“前景”两类像素的"类间方差最大化"，从0-255逐个测试阈值，进而实现二值化
adaptive：区别otsu整体作为一块，adaptive将整体切成若干 n*n的子区间，再进行重复的二值化
tophat：原图 - 腐蚀(原图)
gmm：假设图像像素符合几个"高斯分布--均值-方差"，将像素软分类几个高斯族群，然后迭代出各像素属于"前景"、"背景"两类的概率
meanshift：漂移聚类，利用聚类算法，计算像素之间的距离(亮度-空间坐标系)，进行聚类学习

最优光斑选取：
brightest：计算每个连通域的平局亮度或峰值亮度，选最亮的那个
latgest：计算每个亮斑的像素面积，选面积最大的连通域
central：计算每个光斑的centroid（质心），到图像中心的距离，选最近的那个

当前系统默认：
classic(区别与DP-YOLO) + otsu + brightest

```

6、备份V2版本（2026-5-25版本）	✔

7、基于官方网站系统架构，基于PLAN.MD优化5轴 至 4轴	✔

（1）优化1：保留 原来"5轴系统"("Z轴"的部分去除)，新增"4轴系统"，本质上就是两点确定一条直线【稳定控制逻辑闭环】

```
# 环境测试
	python SpotZoom.py --check-env


# 4轴双镜闭环策略测试
	python SpotZoom.py --alignment-strategy dual_detector_4axis --xy-driver dryrun --z-driver dryrun --detector-backend classic --frame-source-image artifacts/spotzoom_qt_sample.png --skip-roi --no-preview --max-iterations 5 --disable-startup-motion-check --disable-run-lock
	
	
# 传统Z扫描策略测试
	python SpotZoom.py --alignment-strategy z_scan_legacy --xy-driver dryrun --z-driver dryrun --detector-backend classic --frame-source-image artifacts/spotzoom_qt_sample.png --skip-roi --no-preview --max-iterations 3 --disable-startup-motion-check --disable-run-lock


# 4轴策略带耦合补偿测试
	python SpotZoom.py --alignment-strategy dual_detector_4axis --xy-driver dryrun --z-driver dryrun --detector-backend classic --frame-source-image artifacts/spotzoom_qt_sample.png --skip-roi --no-preview --max-iterations 5 --coupling-c12 0.1 --coupling-c21 0.1 --disable-startup-motion-check --disable-run-lock
	
	
# UI界面测试
	python -m spotzoom_qt_ui
	
	
# UI启动并加载4轴配置
	python -m spotzoom_qt_ui --alignment-strategy dual_detector_4axis --xy-driver dryrun --z-driver dryrun --detector-backend classic --frame-source-image artifacts/spotzoom_qt_sample.png --skip-roi --no-preview
	
	
# 4轴逐轴扰动响应测试（dryrun模式）
	python SpotZoom.py --alignment-strategy dual_detector_4axis --xy-driver dryrun --z-driver dryrun --detector-backend classic --frame-source-image artifacts/spotzoom_qt_sample.png --skip-roi --no-preview --max-iterations 1 --disable-startup-motion-check --disable-run-lock --stage1-kp 2.0 --stage2-kp 1.5
   
```

（2）优化2：保留 原来toupview 显微镜成效的自动准直，新增基于"探测器"效果对比的自动准直

```
# 传统ToupView显微镜成像自动准直
	python SpotZoom.py --alignment-strategy z_scan_legacy --xy-driver dryrun --z-driver dryrun --detector-backend classic --frame-source-image sample.png --skip-roi --no-preview --max-iterations 5 --disable-startup-motion-check --disable-run-lock


#  4轴双镜闭环（单探测器模式）
	python SpotZoom.py --alignment-strategy dual_detector_4axis --detector-mode single_detector --xy-driver dryrun --z-driver dryrun --detector-backend classic --frame-source-image sample.png --skip-roi --no-preview --max-iterations 5 --sequential-stage1-iterations 2 --stage1-gain-factor 2.5 --disable-startup-motion-check --disable-run-lock


# 探测器效果对比（探测器优先模式）
	python SpotZoom.py --alignment-strategy detector_comparison --comparison-mode detector_primary --detector-weight 0.7 --touview-weight 0.3 --xy-driver dryrun --z-driver dryrun --detector-backend classic --frame-source-image sample.png --skip-roi --no-preview --max-iterations 5 --disagreement-threshold-px 10.0 --disable-startup-motion-check --disable-run-lock
  
  
# 探测器效果对比（ToupView验证模式）
  	python SpotZoom.py --alignment-strategy detector_comparison --comparison-mode touview_verify --detector-weight 0.3 --touview-weight 0.7 --xy-driver dryrun --z-driver dryrun --detector-backend classic --frame-source-image sample.png --skip-roi --no-preview --max-iterations 5 --disable-startup-motion-check --disable-run-lock
  
  
# 探测器效果对比（自动切换模式）
  	python SpotZoom.py --alignment-strategy detector_comparison --comparison-mode auto_switch --xy-driver dryrun --z-driver dryrun --detector-backend classic --frame-source-image sample.png --skip-roi --no-preview --max-iterations 5 --disable-startup-motion-check --disable-run-lock
  
# 双探测器模式（后续扩展）
  	python SpotZoom.py --alignment-strategy dual_detector_4axis --detector-mode dual_detector --frame-source-image touview.png --frame-source-image-2 detector.png --xy-driver dryrun --z-driver dryrun --detector-backend classic --skip-roi --no-preview --max-iterations 5 --disable-startup-motion-check --disable-run-lock

```

8、通过usb，对CCD探测器完成通信控制的逻辑闭环	✔

```	
1、思路
	CCD相机 → USB采集卡 → UVC驱动 → OpenCV VideoCapture → Python

2、核心功能
    自动连接UCC相机（cv2.VideoCapture）
    支持PAL/NTSC/AUTO三种分辨率模式
    可配置曝光、增益、亮度、对比度
    ROI区域选择
    相机参数动态调整
    自动资源释放


3、单UCC相机（作为主帧源）
    python SpotZoom.py --ucc-device 0 --ucc-resolution PAL --ucc-exposure -5 --ucc-gain 10 --xy-driver dryrun --z-driver dryrun --detector-backend classic --skip-roi --no-preview --max-iterations 5 --disable-startup-motion-check --disable-run-lock
  
  
4、双UCC相机（4轴双镜闭环）
    python SpotZoom.py --alignment-strategy dual_detector_4axis --detector-mode single_detector --ucc-device 0 --ucc-resolution PAL --ucc-device-2 1 --ucc-resolution-2 PAL --xy-driver dryrun --z-driver dryrun --detector-backend classic --skip-roi --no-preview --max-iterations 5 --disable-startup-motion-check --disable-run-lock
  
  
```





#### 2026-5-24

休息



#### 2026-5-23

0、光路搭建：显微镜、驱动电机、探测器、光路、两块镜子闭环		✔

1、驱动电机 -- 可以将电机控制系统补充到现有的UI系统内			  ✔

2、将控制器安装电机至光路内									 ✔

3、总结当前涉及的"光斑检测算法"								✔

```    
0、相关算法
        YOLO、otsu、adaptive、tophat、gmm、meanshift、brightest、largest、central，还结合一些图片形态学的约束，
        然后找了一些文献，来优化每个模块的算子，保证尽可能的优化图片检测、电机驱动的效果

1. 主检测后端
    YOLO 检测（`SpotYOLODetector`）
    经典机器学习算法 
        二值化：otsu/adaptive/tophat/gmm/meanshift；
        候选筛选：brightest / largest / central；
        还会结合面积、圆度、强度比、形态学核等约束，并启用亚像素定位。
    YOLO Worker 子进程检测（SpotYOLOWorkerClient）
    【当前默认架构：classic后端+ otsu + brightest】

2. 丢检后的 fallback / recovery（按开关启用）
    - 多尺度 fallback
    - LodeSTAR fallback
    - StarDist fallback
    - Wavelet fallback
    - 模板类恢复（memory/pyramid/log-polar）
    - KLT 光流恢复
    - 相位相关（phase correlation）恢复
    - ECC 运动补偿恢复
    - CMC 仿射恢复
    - ORB+RANSAC 单应恢复
    - Kalman+模板恢复
    - temporal ensemble 恢复
    - recovery scan（XY 交叉扫描）

3. 检测后的时序稳态/关联门控（Frontier）
    一系列时间连续性、一致性、应力/滞回门控与相位修正，用于“检测结果修正与防抖”

```

4、下一步程序的测试目标规划	✔

```    
启动程序，基于UI内容界面/toupview 选定ROI区域，基于classic机器学习算法，检测光斑位置，确定中心坐标P0
    并将光斑范围进行框定，通过上下循环移动Z轴（如何解决两个电机的问题？）：
    (1) Z轴上升50单位：检测光斑位置，确定中心坐标P1，计算P0与P1的距离，基于曼哈顿距离（x1-x2）+（y1-y2），移动至P0位置
    (2) Z轴下降50单位：检测光斑位置，确定中心坐标P2，计算P0与P2的距离，基于曼哈顿距离（x1-x2）+（y1-y2），移动至P0位置
    (3) 循环(1)、(2)，直到满足结束条件：P0、P1、P2坐标的误差<0.01
    【上述思路，由于涉及了Z轴 引入了原有系统的多余轴，由此，需要被优化掉，具体的思路可以参照市面上的设备进行参考】

    0、数据集问题，光斑问题每次都采集数据不太通用，所以采用机器视觉来解决
    1、解决两个电机的问题，相互完成通讯【多个电机之间完成通信】
    2、如何基于规律控制两个轴？-- 找到一个稳定控制的逻辑 -- 由于是4轴系统，所以需要控制变量【多个轴之间完成逻辑闭环】
    3、研究是否能够通过usb，对CCD探测器完成通信控制的逻辑闭环
    4、github实现项目的版本控制，晚点直接删除slave的8821L项目，再从github上 pull下来【安装不了一点】
    5、优化逻辑，将原有的"5轴系统"中"Z轴"的部分去除，即上述的"准直思路需要进行修改" > 查询资料进行解决

```

5、程序补充通过测试		✔

```
1、完成libuse.dll 的环境配置
2、完成list > List 的 python 3.8语法习惯优化
3、完成toupview > NIS 窗口的优化

```

6、完成每台设备的github最新项目更新	✔



#### 2026-5-22

网卡设置与正常联网	✔

Python环境、pip环境、win7-vscode-python拓展插件、todesk安装		✔

everything、Snipaste、clash+网桥、安装探测器		✔

git环境部署、删除非git的旧项目 -- 完成项目的版本控制 -- 多个设备协同分工，更新对应模块合并实现版本更新	✔

解决了win7-64bit版本的兼容问题	✔

完成整体光路的搭建与使用测试 -- 光路闭环【高反透镜 + 银镜反射 + CCD光斑探测器 + 显微镜可视化】	✔



#### 2026-5-21

git 可以通过分支操作，实现版本号的控制，同时可通过"签出"，随时更新本地项目至对应的版本号(v1、v2等)	✔

通过plan模式，实现对应上下文的指令推荐，可以大幅度解放prompt的设计痛点	✔

将抽象封装函数，设计成UI产品化界面，同时方便后续的bug检测（基于UI段的debug难度显然 < 抽象的终端日志debug）✔

复现项目，完成各种bug，直至跑通

找到控制驱动器位移移动的逻辑部分【方便后续调试】

在文件内容补充一个虚拟环境文件，保证项目可以正常运行【当前使用的是cp11_torch22】

将所有设备的通信协议封装成Python函数，即可随时随地进行模拟



#### 2026-5-20

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

## SpotZoom Qt UI (v1)

Launch the native desktop console:

```bash
python -m spotzoom_qt_ui
```

The UI includes Dashboard, Alignment Workspace, Module Center, Device Center, Device Test, Run Modes, Simulation Lab, Logs & Reports, and Settings.
