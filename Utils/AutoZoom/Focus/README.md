# Focus — 自动对焦与聚焦指标模块

从 AutoZoom 闭环测量系统中提取的独立自动对焦模块。提供聚焦指标计算、FocusScore 评分、参考基线建立、Z 轴硬件控制以及闭环补焦搜索的完整解决方案。

---

## 目录结构

```
Focus/
├── __init__.py             # 包导出
├── config.py               # AutofocusConfig 数据类（全部聚焦相关参数）
├── metrics.py              # FocusMetricsCalculator（10 项聚焦指标计算 + 截图）
├── scorer.py               # FocusScorer（FocusScore 加权评分 + 参考基线）
├── z_axis.py               # ZAxisController（Newport 8742 Picomotor 控制）
├── controller.py           # AutofocusController（触发判断 + 闭环补焦搜索）
└── run_autofocus.py        # 独立运行示例脚本
```

---

## 快速开始

### 依赖

```bash
pip install numpy opencv-python pyautogui pillow pylablib
```

- `opencv-python` — Sobel / Laplacian 等图像梯度计算
- `pyautogui` — 屏幕截图
- `pillow` — 保存截图
- `pylablib` — Newport Picomotor Z 轴控制（可选，纯指标计算不需要）

### 运行示例

```bash
cd AutoZoom
python Focus/run_autofocus.py
```

示例脚本将：
1. 截取屏幕指定区域
2. 计算 10 项聚焦指标（整图 + ROI）
3. 建立聚焦参考基线
4. 计算 FocusScore_ratio
5. 评估补焦触发条件

---

## 模块说明

### 1. `config.py` — AutofocusConfig

所有聚焦与自动补焦参数的集中配置。

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `capture_area` | `(116, 98, 1112, 886)` | 屏幕截图区域 `(left, top, width, height)` |
| `focus_roi` | `(0, 0, 300, 300)` | ROI 相对于截图左上角 `(x, y, w, h)` |
| `focus_fft_low_radius_ratio` | `0.10` | FFT 高频能量低频圆盘半径比例 |
| `focus_edge_profile_half_width` | `60` | 边缘轮廓分析半宽度 (px) |
| `focus_weight_highfreq` | `0.35` | 高频比权重 |
| `focus_weight_tenengrad` | `0.25` | Tenengrad 权重 |
| `focus_weight_brenner` | `0.25` | Brenner 权重 |
| `focus_weight_red_blue` | `0.15` | 红蓝比权重 |
| `autofocus_enabled` | `True` | 启用自动补焦 |
| `autofocus_focus_trigger_ratio` | `0.90` | FocusScore 触发阈值 |
| `autofocus_focus_trigger_count` | `3` | 连续触发次数 |
| `autofocus_stop_ratio` | `0.95` | 补焦目标值 |
| `z_probe_steps` | `10` | 试探步数 |
| `z_search_steps` | `10` | 搜索步数 |
| `z_settle_time_s` | `0.20` | 移动后稳定等待时间 (s) |
| `z_max_iter` | `40` | 最大搜索迭代 |
| `z_patience` | `3` | 连续无进步耐心轮数 |

### 2. `metrics.py` — FocusMetricsCalculator

聚焦指标计算核心。对一张 RGB 图像同时计算以下 10 项指标：

| 指标 | 含义 | 计算方式 |
|------|------|----------|
| `tenengrad` | 梯度能量 | Sobel 梯度平方均值 |
| `laplacian_var` | 边缘方差 | Laplacian 响应方差 |
| `brenner` | 邻域差分 | 相隔 2px 的灰度差平方均值 |
| `highfreq_ratio` | 高频能量占比 | FFT 去低频圆盘后高频能量 / 总能量 |
| `local_contrast` | 局部对比度 | 15×15 局部标准差均值 / 全局亮度均值 |
| `edge_width` | 边缘宽度 | 最强边缘 10%→90% 过渡宽度 |
| `halo_width` | 晕轮宽度 | 最强边缘梯度主峰 25% 阈值宽度 |
| `brightness_mean` | 亮度均值 | 灰度均值 |
| `brightness_std` | 亮度标准差 | 灰度标准差 |
| `red_blue_ratio` | 红蓝比 | R 通道均值 / B 通道均值 |

**核心方法：**

```python
# 对单张图像计算指标
metrics = calc.compute_for_image(image_rgb)

# 截图并计算（不保存）
live = calc.capture_live()

# 截图、计算、保存
result = calc.capture_and_save(cycle_index=0, save_dir="output")

# 展平嵌套指标为平坦 dict（适合 CSV 保存）
flat = FocusMetricsCalculator.flatten_metrics(result)
```

### 3. `scorer.py` — FocusScorer

聚焦评分与参考基线管理。

- **`score_ratio(roi_metrics)`** → `(FocusScore_ratio, component_ratios)`
  - 加权公式：`FocusScore = Σ(w_i × metric_i_current / metric_i_ref)`
  - 每个分量 ratio 限幅到 `[0.0, 2.5]`
  - 权重归一化确保总和为 1

- **`build_reference(capture_count, output_root)`** → `focus_reference dict`
  - 多次截图取各指标均值作为参考
  - 可选保存参考截图和 CSV

### 4. `z_axis.py` — ZAxisController

Newport 8742 Picomotor 硬件控制封装。

```python
z = ZAxisController(cfg)
z.connect()               # 连接硬件
z.move_relative(100)      # 正向 100 步
z.move_relative(-50)      # 反向 50 步
```

### 5. `controller.py` — AutofocusController

自动补焦的高层编排器，封装完整的闭环搜索逻辑。

#### 触发判断 (`evaluate_trigger`)

触发自动补焦的条件（任一满足即可）：
1. **聚焦退化** — FocusScore_ratio 连续 N 轮低于 `autofocus_focus_trigger_ratio`
2. **SHG 骤降** — SHG_ratio 单轮低于 `autofocus_shg_hard_ratio`（可选）
3. **SHG 持续偏低** — SHG_ratio 连续 N 轮低于 `autofocus_shg_trigger_ratio`（可选）

#### 闭环搜索算法 (`run_closed_loop`)

```
1. 记录初始 FocusScore
2. 试探 +z 方向，若分数提升则继续；否则试探 -z
3. 沿提升方向递进搜索，每次移动后截图评估
4. 连续无进步达到 patience 或达到目标值则停止
5. 回退到搜索过程中的最佳位置
```

#### 一站式接口 (`check_and_autofocus`)

```python
result = controller.check_and_autofocus(cycle_index=1, save_dir="output")
# 内部完成: 截图 → 评分 → 判断 → 闭环搜索 → 最终截图
```

---

## 评分公式

```
FocusScore_ratio = weighted_mean(metric_current / metric_ref)

其中 metric ∈ {highfreq_ratio, tenengrad, brenner, red_blue_ratio}
权重: 0.35 / 0.25 / 0.25 / 0.15 (可配置)

每个分量的 ratio 限幅 [0.0, 2.5] 防止异常值
```

- `FocusScore_ratio ≈ 1.0` 表示当前聚焦状态与参考一致
- `FocusScore_ratio < 0.9` 提示聚焦退化，可能需要补焦
- `FocusScore_ratio ≥ 0.95` 视为聚焦已恢复

---

## 独立使用

```python
from Focus import (
    AutofocusConfig,
    FocusMetricsCalculator,
    FocusScorer,
    ZAxisController,
    AutofocusController,
)

# 1. 配置
cfg = AutofocusConfig()

# 2. 初始化
metrics_calc = FocusMetricsCalculator(cfg)
scorer = FocusScorer(cfg, metrics_calc)
z_axis = ZAxisController(cfg)
controller = AutofocusController(cfg, scorer, metrics_calc, z_axis)
controller.on_log = print

# 3. 建立参考
controller.build_reference(output_root="my_output")

# 4. 每轮测量时调用
result = controller.check_and_autofocus(cycle_index=1, save_dir="my_output/session_1")
focus_score = result.get("focus_score_ratio")
roi_metrics = result.get("roi_metrics")
```

---

## 原始项目来源

本模块从 [`measurement_autofocus_shg_closed_loop.py`](../measurement_autofocus_shg_closed_loop.py) 的 `MeasurementWorkflow` 类中提取，代码结构和算法与原始版本保持一致。