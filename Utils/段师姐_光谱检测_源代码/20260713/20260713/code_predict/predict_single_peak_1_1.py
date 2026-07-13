# ============================================================
# predict_peaks.py
# 使用训练好的模型预测峰值概率
# ============================================================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
import joblib

# -----------------------
# 0. 配置参数
# -----------------------
MODEL_PATH = "peak_prediction_mse.pt"
INPUT_FILE = "BN_BN3_delete_29_rows.xlsx"  # 请替换为实际文件路径
OUTPUT_FILE = "peak_predictions29.csv"
SEQ_LEN_IN = 10
SEQ_LEN_OUT = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# 1. 加载模型和配置
# -----------------------
print("🚀 加载训练好的模型...")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

# 获取模型配置
input_size = checkpoint['input_size']
seq_len_in = checkpoint['seq_len_in']
seq_len_out = checkpoint['seq_len_out']
feature_columns = checkpoint['feature_columns']

print(f"模型配置: 输入维度={input_size}, 输入序列长度={seq_len_in}, 输出序列长度={seq_len_out}")

# 初始化模型
class PeakModel(nn.Module):
    def __init__(self, in_ch=input_size, hidden=128, out_len=seq_len_out):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_ch, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, out_len)
        )
        
    def forward(self, x):
        features = self.encoder(x)
        features = features.squeeze(-1)
        out = self.classifier(features)
        return torch.sigmoid(out)

model = PeakModel(input_size, 128, seq_len_out).to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("✅ 模型加载完成")

# -----------------------
# 2. 特征工程函数（与训练时相同）
# -----------------------
def calculate_window_features_fixed(window_intensities, current_index, window_size=10):
    """
    特征计算：确保不使用未来信息
    只使用当前时间点及之前的信息计算特征
    """
    n = len(window_intensities)
    if n < window_size:
        padding = window_size - n
        window_intensities = np.pad(window_intensities, (0, padding), mode='edge')
        n = window_size
    
    intensities_raw = window_intensities
    
    features_dict = {}
    
    # 1. 只使用到当前时间点的数据
    valid_length = current_index + 1  # 只使用到当前时间点
    valid_data = intensities_raw[:valid_length]
    
    # 2. 基础特征（只使用历史数据）
    features_dict['intensity_raw'] = np.full(n, valid_data[-1] if len(valid_data) > 0 else 0.0)
    
    # 3. 平滑特征（使用历史数据计算）
    spans = [3, 5]
    smooth_features = {}
    
    for span in spans:
        if span < valid_length:
            # 只使用历史数据计算平滑
            historical_data = valid_data
            intensity_smooth = pd.Series(historical_data).ewm(span=min(span, len(historical_data))).mean().values
            current_smooth = intensity_smooth[-1] if len(intensity_smooth) > 0 else historical_data[-1]
            features_dict[f'intensity_smooth_span{span}'] = np.full(n, current_smooth)
            smooth_features[span] = np.full(n, current_smooth)
        else:
            features_dict[f'intensity_smooth_span{span}'] = np.full(n, valid_data[-1] if len(valid_data) > 0 else 0.0)
            smooth_features[span] = np.full(n, valid_data[-1] if len(valid_data) > 0 else 0.0)
    
    # 4. 差分特征（基于历史数据）
    if valid_length >= 2:
        last_diff = valid_data[-1] - valid_data[-2]
    else:
        last_diff = 0.0
    
    features_dict['diff_raw'] = np.full(n, last_diff)
    
    for span in spans:
        if span in smooth_features:
            # 简化差分计算
            features_dict[f'diff_smooth_span{span}'] = np.full(n, last_diff)  # 使用相同的差分值
    
    # 5. 斜率特征（只使用历史数据）
    sequence_types = ['raw'] + [f'smooth_span{span}' for span in spans]
    
    for seq_type in sequence_types:
        if seq_type == 'raw':
            values = valid_data
        else:
            values = valid_data  # 简化处理
        
        # 长期斜率（使用所有可用历史数据）
        if len(values) > 1:
            x_long = np.arange(len(values))
            if np.std(values) > 1e-8:
                try:
                    slope_long = stats.linregress(x_long, values).slope
                except:
                    slope_long = 0.0
            else:
                slope_long = 0.0
        else:
            slope_long = 0.0
        
        features_dict[f'long_slope_{seq_type}'] = np.full(n, slope_long)
        
        # 动量特征（近期变化）
        if len(values) >= 2:
            short_momentum = values[-1] - values[-2]
        else:
            short_momentum = 0.0
        
        features_dict[f'short_momentum_{seq_type}'] = np.full(n, short_momentum)
    
    # 6. 确保特征维度一致
    for key in features_dict:
        if len(features_dict[key]) != n:
            features_dict[key] = np.resize(features_dict[key], n)
    
    return features_dict

# -----------------------
# 3. 归一化器类（与训练时相同）
# -----------------------
class TemporalNormalizer:
    def __init__(self, robust=True):
        self.robust = robust
        self.feature_columns = feature_columns
        self.fitted = True
    
    def normalize_window(self, window_intensities, current_index):
        """时序归一化：只使用历史信息"""
        n = len(window_intensities)
        
        # 1. 只使用到当前时间点的数据进行归一化
        historical_data = window_intensities[:current_index+1]
        
        if len(historical_data) == 0:
            intensities_normalized = np.zeros(n, dtype=np.float32)
        else:
            # 使用历史数据计算统计量
            valid_mask = np.isfinite(historical_data)
            valid_values = historical_data[valid_mask]
            
            if len(valid_values) == 0:
                intensities_normalized = np.zeros(n, dtype=np.float32)
            else:
                if self.robust:
                    median = np.median(valid_values)
                    mad = np.median(np.abs(valid_values - median))
                    std = mad * 1.4826 if mad > 0 else 1.0
                else:
                    median = np.mean(valid_values)
                    std = np.std(valid_values) if np.std(valid_values) > 0 else 1.0
                
                # 归一化整个窗口，但使用历史统计量
                normalized_values = (window_intensities - median) / std if std > 0 else (window_intensities - median)
                normalized_values[~np.isfinite(normalized_values)] = 0.0
                intensities_normalized = normalized_values.astype(np.float32)
        
        # 2. 时序特征计算
        features_dict = calculate_window_features_fixed(intensities_normalized, current_index, n)
        
        # 3. 组织特征矩阵
        feature_matrix = np.zeros((n, len(self.feature_columns)), dtype=np.float32)
        
        for i, col in enumerate(self.feature_columns):
            if col in features_dict:
                v = features_dict[col].astype(np.float32)
                v = np.nan_to_num(v, nan=0.0)
                v[~np.isfinite(v)] = 0.0
                if len(v) == n:
                    feature_matrix[:, i] = v
                else:
                    feature_matrix[:min(n, len(v)), i] = v[:n]
            else:
                feature_matrix[:, i] = 0.0
        
        return feature_matrix

# -----------------------
# 4. 数据加载和预处理
# -----------------------
print("📊 加载和预处理数据...")
# 读取Excel文件
df = pd.read_excel(INPUT_FILE)
intensities = df['intensity'].values

# 初始化归一化器
scaler = TemporalNormalizer(robust=True)

# 创建滑动窗口
predictions = []
window_indices = []

for i in range(0, len(intensities) - SEQ_LEN_IN + 1):
    # 获取当前窗口
    window_intensities = intensities[i:i+SEQ_LEN_IN]
    
    # 计算特征
    features = scaler.normalize_window(window_intensities, current_index=SEQ_LEN_IN-1)
    
    # 转换为模型输入格式
    features_tensor = torch.tensor(features).permute(1, 0).unsqueeze(0).float().to(DEVICE)  # (1, 45, 10)
    
    # 预测
    with torch.no_grad():
        output = model(features_tensor)
        pred_probs = output.cpu().numpy()[0]  # (10,)
    
    # 保存预测结果
    predictions.append(pred_probs)
    window_indices.append(i)

# -----------------------
# 5. 保存预测结果
# -----------------------
print("💾 保存预测结果...")
# 创建结果DataFrame
result_df = pd.DataFrame(predictions, columns=[f't{i}_pred' for i in range(SEQ_LEN_OUT)])

# 添加窗口起始索引
result_df['window_start'] = window_indices

# 保存到CSV文件
result_df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ 预测完成! 结果已保存到: {OUTPUT_FILE}")
print(f"总共预测了 {len(predictions)} 个窗口")
print(f"每个窗口预测了 {SEQ_LEN_OUT} 个时间点的峰值概率")

# 显示前几个预测结果
print("\n📋 前5个窗口的预测结果:")
print(result_df.head().round(4))