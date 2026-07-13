import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# 创建保存图片的文件夹
if not os.path.exists('results'):
    os.makedirs('results')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 从Excel文件读取数据
df = pd.read_excel('final_prediction45.xlsx', sheet_name='Sheet1')

# 提取数据
steps = df['step'].values
history_angles = df['history_angle'].values  # 历史角度数据
predict_angles = df['predict_angle'].values  # 预测角度数据
intensities = df['intensity'].values  # 强度数据
peak_probs = df['peak_prob'].values  # 峰值概率数据

# 自动确定数据长度
total_steps = len(steps)
print(f"Excel数据总长度: {total_steps} 个时间步")

# 自动确定历史数据和预测数据的边界
# 找到第一个非NaN的预测角度和峰值概率的位置
first_pred_idx = None
for i in range(total_steps):
    if not pd.isna(predict_angles[i]) and not pd.isna(peak_probs[i]):
        first_pred_idx = i
        break

if first_pred_idx is None:
    first_pred_idx = total_steps // 2  # 如果没有明确边界，使用中间点作为分隔

print(f"预测数据开始于时间步: {first_pred_idx}")

# 根据数据自动划分历史时间和未来时间
historical_time = steps[:first_pred_idx]  # 历史时间步
future_time = steps[first_pred_idx:]      # 未来时间步

# 提取对应的数据
historical_intensity = intensities[:first_pred_idx]
historical_angle = history_angles[:first_pred_idx]
predicted_angle = predict_angles[first_pred_idx:]
peak_probability = peak_probs[first_pred_idx:]

# 找到峰值时刻（概率最大值对应的时间步）
if len(peak_probability) > 0:
    peak_time_index = np.argmax(peak_probability)
    peak_time = future_time[peak_time_index]  # 峰值时间步
    peak_prob = peak_probability[peak_time_index]  # 峰值概率
    peak_angle = predicted_angle[peak_time_index]  # 峰值对应角度
else:
    peak_time = future_time[0] if len(future_time) > 0 else total_steps-1
    peak_prob = 0
    peak_angle = predicted_angle[0] if len(predicted_angle) > 0 else 0

print(f"预测峰值时刻: 时间步 {peak_time}")
print(f"峰值概率: {peak_prob:.4f}")
print(f"对应角度: {peak_angle:.2f}°")

# 创建第一个图表：双子图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# 自动设置横坐标范围
x_min = min(steps) - 0.5
x_max = max(steps) + 0.5

# 第一个子图：强度序列和峰值概率分布
ax1.plot(historical_time, historical_intensity, 'b-o', linewidth=2, markersize=6, 
         label='历史强度序列', markerfacecolor='white')
ax1.set_xlabel('时间步', fontsize=12)
ax1.set_ylabel('强度', color='b', fontsize=12)
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(x_min, x_max)

# 创建第二个y轴用于概率分布
ax1b = ax1.twinx()
if len(future_time) > 0:  # 只有有预测数据时才绘制
    ax1b.plot(future_time, peak_probability, 'r-s', linewidth=2, markersize=6, 
              label='峰值概率分布', markerfacecolor='white')
    ax1b.set_ylabel('峰值概率', color='r', fontsize=12)
    ax1b.tick_params(axis='y', labelcolor='r')
    if len(peak_probability) > 0:
        ax1b.set_ylim(0, max(peak_probability) * 1.1)
    
    # 标记峰值点
    ax1b.axvline(x=peak_time, color='g', linestyle='--', linewidth=2, 
                 label=f'预测峰值时刻 (t={peak_time})')
    ax1b.plot(peak_time, peak_prob, 'go', markersize=10, 
              label=f'峰值概率: {peak_prob:.3f}')

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines1b, labels1b = ax1b.get_legend_handles_labels()
ax1.legend(lines1 + lines1b, labels1 + labels1b, loc='upper left')

ax1.set_title('光谱信号强度序列与峰值概率分布预测', fontsize=14, fontweight='bold')

# 第二个子图：角度序列预测
ax2.plot(historical_time, historical_angle, 'g-o', linewidth=2, markersize=6, 
         label='历史角度序列', markerfacecolor='white')
if len(future_time) > 0:  # 只有有预测数据时才绘制
    ax2.plot(future_time, predicted_angle, 'g--s', linewidth=2, markersize=6, 
             label='预测角度序列', markerfacecolor='white')
ax2.set_xlabel('时间步', fontsize=12)
ax2.set_ylabel('角度 (°)', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(x_min, x_max)

# 标记峰值对应的角度
if len(future_time) > 0:  # 只有有预测数据时才标记
    ax2.axvline(x=peak_time, color='r', linestyle='--', linewidth=2, 
                label=f'峰值时刻 (t={peak_time})')
    ax2.plot(peak_time, peak_angle, 'ro', markersize=10, 
             label=f'临界角度: {peak_angle:.2f}°')

ax2.legend(loc='upper left')
ax2.set_title('角度序列轨迹预测与临界角度估计', fontsize=14, fontweight='bold')

plt.tight_layout()

# 保存第一个图表
plt.savefig('results/双子图_峰值与角度预测.png', dpi=300, bbox_inches='tight')
plt.savefig('results/双子图_峰值与角度预测.pdf', bbox_inches='tight')
plt.show()

# 创建第二个图表：组合图
fig, ax1 = plt.subplots(figsize=(12, 6))

# 绘制强度序列（只显示历史部分）
ax1.plot(historical_time, historical_intensity, 'b-o', linewidth=2, 
         label='强度序列', markerfacecolor='white')
ax1.set_xlabel('时间步', fontsize=12)
ax1.set_ylabel('强度', color='b', fontsize=12)
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(x_min, x_max)

# 绘制角度序列（历史和预测）
ax2 = ax1.twinx()
# 组合历史角度和预测角度
ax2.plot(historical_time, historical_angle, 'g-o', linewidth=2, 
         label='历史角度', markerfacecolor='white')
if len(future_time) > 0:  # 只有有预测数据时才绘制
    ax2.plot(future_time, predicted_angle, 'g--s', linewidth=2, 
             label='预测角度', markerfacecolor='white')
ax2.set_ylabel('角度 (°)', color='g', fontsize=12)
ax2.tick_params(axis='y', labelcolor='g')

# 标记峰值点
if len(future_time) > 0:  # 只有有预测数据时才标记
    ax1.axvline(x=peak_time, color='r', linestyle='--', linewidth=2, 
                label=f'峰值时刻 t={peak_time}')
    ax2.plot(peak_time, peak_angle, 'ro', markersize=10, 
             label=f'临界角度: {peak_angle:.2f}°')

# 添加概率分布作为背景色或点状图
ax1b = ax1.twinx()
ax1b.spines['right'].set_position(('outward', 60))
if len(future_time) > 0:  # 只有有预测数据时才绘制
    ax1b.plot(future_time, peak_probability, 'm-^', linewidth=1, markersize=5, 
              label='峰值概率', alpha=0.7)
    ax1b.set_ylabel('峰值概率', color='m', fontsize=10)
    ax1b.tick_params(axis='y', labelcolor='m')
    if len(peak_probability) > 0:
        ax1b.set_ylim(0, max(peak_probability) * 1.2)

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines1b, labels1b = ax1b.get_legend_handles_labels()
ax1.legend(lines1 + lines2 + lines1b, labels1 + labels2 + labels1b, 
           loc='upper left', bbox_to_anchor=(0, 1))

ax1.set_title('多目标并行预测框架结果展示\n（强度序列、角度轨迹与峰值概率）', 
             fontsize=14, fontweight='bold')

plt.tight_layout()

# 保存第二个图表
plt.savefig('results/组合图_多目标预测结果.png', dpi=300, bbox_inches='tight')
plt.savefig('results/组合图_多目标预测结果.pdf', bbox_inches='tight')
plt.show()

# 可选：创建第三个图表，专门展示峰值概率分布的形态
if len(future_time) > 0:  # 只有有预测数据时才创建这个图表
    plt.figure(figsize=(10, 6))
    plt.plot(future_time, peak_probability, 'r-s', linewidth=2, markersize=8, 
             label='峰值概率分布', markerfacecolor='white')
    plt.axvline(x=peak_time, color='g', linestyle='--', linewidth=2, 
                label=f'预测峰值时刻 (t={peak_time})')
    plt.plot(peak_time, peak_prob, 'go', markersize=12, 
             label=f'峰值概率: {peak_prob:.3f}')

    # 标记"先上升后下降"的形态特征
    # 找到上升段和下降段
    rise_start = min(future_time)  # 概率开始上升的位置
    rise_end = peak_time
    fall_end = max(future_time)  # 概率下降结束的位置

    # 用不同颜色标记上升段和下降段
    rise_indices = [i for i, t in enumerate(future_time) if t <= rise_end]
    fall_indices = [i for i, t in enumerate(future_time) if t > rise_end]

    if len(rise_indices) > 0:
        plt.plot([future_time[i] for i in rise_indices], 
                 [peak_probability[i] for i in rise_indices], 
                 'r-', linewidth=3, alpha=0.7, label='上升段')
    if len(fall_indices) > 0:
        plt.plot([future_time[i] for i in fall_indices], 
                 [peak_probability[i] for i in fall_indices], 
                 'r-', linewidth=3, alpha=0.3, label='下降段')

    plt.xlabel('时间步', fontsize=12)
    plt.ylabel('峰值概率', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(min(future_time)-0.5, max(future_time)+0.5)  # 自动设置横坐标范围
    plt.legend()
    plt.title('峰值概率分布形态分析\n（符合"先上升后下降"规律）', fontsize=14, fontweight='bold')

    # 保存第三个图表
    plt.tight_layout()
    plt.savefig('results/峰值概率分布形态分析.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/峰值概率分布形态分析.pdf', bbox_inches='tight')
    plt.show()

print("所有图表已保存到 'results' 文件夹中：")
print("- 双子图_峰值与角度预测.png/.pdf")
print("- 组合图_多目标预测结果.png/.pdf")
if len(future_time) > 0:
    print("- 峰值概率分布形态分析.png/.pdf")