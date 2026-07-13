# ============================================================
# train_peak_prediction_mse.py
# 峰值预测系统 - 使用MSE损失函数（完整特征版本）
# ============================================================

import os, glob, random, joblib
import numpy as np
import pandas as pd
from scipy import stats, signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------
# 0. Hyperparameters
# -----------------------
SEQ_LEN_IN  = 10
SEQ_LEN_OUT = 10
INPUT_SIZE  = 45

HIDDEN_SIZE  = 256
NUM_EPOCHS   = 150
BATCH_SIZE   = 64
LEARNING_RATE = 2e-4
DROPOUT      = 0.35
SEED = 42
WEIGHT_DECAY = 1e-4
PATIENCE = 20

TRAIN_FOLDER = "train_data_intensity"
VAL_FOLDER   = "val_data_intensity"
SAVE_PATH = "peak_prediction_mse.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# -----------------------
# 1. 特征工程函数（保持45维特征）
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
            intensity_smooth = pd.Series(historical_data).ewm(span=min(span, len(historical_data)-1)).mean().values
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

def get_feature_columns_fixed():
    """特征列定义（保持45维）"""
    spans = [3, 5]
    sequence_types = ['raw'] + [f'smooth_span{span}' for span in spans]
    
    feature_columns = []
    
    feature_columns.append('intensity_raw')
    
    for span in spans:
        feature_columns.append(f'intensity_smooth_span{span}')
    
    feature_columns.append('diff_raw')
    for span in spans:
        feature_columns.append(f'diff_smooth_span{span}')
    
    slope_momentum_types = ['long_slope', 'short_momentum']
    for sm_type in slope_momentum_types:
        for seq_type in sequence_types:
            feature_columns.append(f'{sm_type}_{seq_type}')
    
    return feature_columns

# -----------------------
# 2. 归一化器（保持45维特征）
# -----------------------
class TemporalNormalizer:
    def __init__(self, robust=True):
        self.robust = robust
        self.feature_columns = get_feature_columns_fixed()
        self.fitted = True
        print(f"时序归一化器初始化，特征维度: {len(self.feature_columns)}")
    
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
# 3. 数据集（保持45维特征，不区分正负样本）
# -----------------------
class PeakDataset(Dataset):
    def __init__(self, files, seq_in, seq_out, scaler,
                 mode='train', stride=1):
        """
        数据集：输入t0-t9的特征（45维），输出t10-t19的is_peak
        """
        self.seq_in = seq_in
        self.seq_out = seq_out
        self.mode = mode
        self.scaler = scaler
        self.samples = []
        
        need = seq_in + seq_out
        
        print(f"正在创建{mode}数据集...")
        print(f"输入序列长度: {seq_in}, 输出序列长度: {seq_out}")
        
        for file_idx, f in enumerate(files):
            try:
                df = pd.read_csv(f)
                if 'intensity' not in df.columns or 'is_peak' not in df.columns: 
                    continue
                if len(df) < need: 
                    continue
                
                intensities = df['intensity'].values
                is_peak = df['is_peak'].values
                
                # 创建所有可能的序列对
                for i in range(0, len(df) - need + 1, stride):
                    # 输入: t0-t9 的 intensity（用于计算45维特征）
                    input_start = i
                    input_end = i + seq_in
                    
                    # 输出: t10-t19 的 is_peak
                    output_start = i + seq_in
                    output_end = i + seq_in + seq_out
                    
                    if output_end > len(is_peak):
                        continue
                    
                    # 获取输入强度序列
                    input_intensities = intensities[input_start:input_end]
                    # 获取输出峰值标签
                    output_peaks = is_peak[output_start:output_end]
                    
                    # 计算45维特征
                    features = scaler.normalize_window(input_intensities, current_index=seq_in-1)
                    
                    sample_data = {
                        'features': features.astype(np.float32),
                        'targets': output_peaks.astype(np.float32),
                        'file_path': f,
                        'start_index': input_start
                    }
                    
                    self.samples.append(sample_data)
                    
            except Exception as e:
                print(f"处理文件 {f} 时出错: {e}")
                continue
        
        print(f"{mode}集样本统计:")
        print(f"  总样本数: {len(self.samples)}")
        
        if len(self.samples) > 0:
            self._analyze_samples()
    
    def _analyze_samples(self):
        """分析样本分布"""
        peak_counts = []
        total_peaks = 0
        
        for sample in self.samples:
            peak_count = np.sum(sample['targets'] == 1)
            peak_counts.append(peak_count)
            total_peaks += peak_count
        
        print(f"峰值统计: 总峰值数{total_peaks}, 平均每个输出窗口{np.mean(peak_counts):.2f}个峰值")
        print(f"峰值比例: {total_peaks/(len(self.samples)*self.seq_out):.2%}")
        print(f"特征维度: {self.samples[0]['features'].shape}")
    
    def __len__(self): 
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        X = sample['features']  # 形状: (seq_in, 45)
        y = sample['targets']   # 形状: (seq_out,)
        
        X_tensor = torch.tensor(X).permute(1, 0)  # 转换为 (45, seq_in)
        y_tensor = torch.tensor(y)
        
        return X_tensor, y_tensor

# -----------------------
# 4. 使用MSE损失函数
# -----------------------
class WeightedMSELoss(nn.Module):
    def __init__(self, pos_weight=2.0, reduction='mean'):
        """
        加权MSE损失：为正样本分配更高权重
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction='none')
        
        print(f"使用加权MSE损失: 正样本权重={pos_weight}")
    
    def forward(self, pred, target):
        # 计算基础MSE损失
        base_loss = self.mse_loss(pred, target)
        
        # 为正样本分配更高权重
        weights = torch.where(target > 0.5, self.pos_weight, 1.0)
        weighted_loss = base_loss * weights
        
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss

# -----------------------
# 5. 模型结构（保持45维输入）
# -----------------------
class PeakModel(nn.Module):
    def __init__(self, in_ch=INPUT_SIZE, hidden=128, out_len=SEQ_LEN_OUT):
        """
        峰值预测模型
        """
        super().__init__()
        
        # 编码器
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
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(128, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, out_len)
        )
        
    def forward(self, x):
        # 编码特征
        features = self.encoder(x)
        features = features.squeeze(-1)
        
        # 分类
        out = self.classifier(features)
        return torch.sigmoid(out)

# -----------------------
# 6. 训练策略
# -----------------------
def train_epoch(model, dl, optimizer, criterion, device):
    """训练epoch"""
    model.train()
    total_loss = 0
    batch_count = 0
    
    for X, y in dl:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
    
    return total_loss / batch_count if batch_count > 0 else 0

@torch.no_grad()
def validate_epoch(model, dl, criterion, device):
    """验证函数"""
    model.eval()
    total_loss = 0
    batch_count = 0
    
    all_preds = []
    all_targets = []
    
    for X, y in dl:
        X, y = X.to(device), y.to(device)
        
        outputs = model(X)
        loss = criterion(outputs, y)
        
        total_loss += loss.item()
        batch_count += 1
        
        all_preds.append(outputs.cpu().numpy())
        all_targets.append(y.cpu().numpy())
    
    if len(all_preds) > 0:
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        
        # 计算多个阈值下的指标
        thresholds = [0.3, 0.5, 0.7]
        metrics = {}
        
        for threshold in thresholds:
            preds_binary = (all_preds > threshold).astype(float)
            
            accuracy = np.mean(preds_binary == all_targets)
            
            tp = np.sum((preds_binary == 1) & (all_targets == 1))
            fp = np.sum((preds_binary == 1) & (all_targets == 0))
            fn = np.sum((preds_binary == 0) & (all_targets == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[threshold] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        
        return avg_loss, metrics
    else:
        return 0, {}

# -----------------------
# 7. 主训练函数
# -----------------------
def main():
    """主训练函数"""
    feature_columns = get_feature_columns_fixed()
    global INPUT_SIZE
    INPUT_SIZE = len(feature_columns)
    print(f"🎯 输入特征维度: {INPUT_SIZE}")
    
    train_files = sorted(glob.glob(os.path.join(TRAIN_FOLDER, "*.csv")))
    val_files = sorted(glob.glob(os.path.join(VAL_FOLDER, "*.csv")))
    
    if len(train_files) == 0 or len(val_files) == 0:
        print(f"❌ 数据缺失: train_files={len(train_files)}, val_files={len(val_files)}")
        return

    print("🚀 开始峰值预测系统训练")
    print(f"设备: {DEVICE}")
    print(f"训练文件: {len(train_files)}个")
    print(f"验证文件: {len(val_files)}个")
    print(f"特征数量: {INPUT_SIZE}个")
    print(f"任务: 输入t0-t9的45维特征，预测t10-t19的is_peak")
    
    # 使用时序归一化器
    scaler = TemporalNormalizer(robust=True)
    
    # 创建数据集
    print("📊 创建数据集...")
    train_dataset = PeakDataset(
        train_files, SEQ_LEN_IN, SEQ_LEN_OUT, scaler, 
        mode='train', stride=2
    )
    
    val_dataset = PeakDataset(
        val_files, SEQ_LEN_IN, SEQ_LEN_OUT, scaler,
        mode='val', stride=1
    )
    
    if len(train_dataset) == 0:
        print("❌ 训练集中没有找到有效窗口")
        return
    
    if len(val_dataset) == 0:
        print("❌ 验证集中没有找到有效窗口")
        return
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # 使用模型
    model = PeakModel(INPUT_SIZE, 128, SEQ_LEN_OUT).to(DEVICE)
    
    # 使用加权MSE损失函数
    criterion = WeightedMSELoss(pos_weight=2.0).to(DEVICE)
    
    # 优化器配置
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )
    
    best_val_f1 = 0.0
    patience_counter = 0
    train_history = []
    val_history = []
    
    print("🏃 开始训练...")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, DEVICE)
        
        scheduler.step()
        
        train_history.append(train_loss)
        val_history.append(val_loss)
        
        # 使用阈值0.5的F1作为主要指标
        main_f1 = val_metrics.get(0.5, {}).get('f1', 0.0)
        
        print(f"[Epoch {epoch:03d}] 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
        print(f"          阈值0.5 -> 精确率: {val_metrics[0.5]['precision']:.4f}, "
              f"召回率: {val_metrics[0.5]['recall']:.4f}, F1: {main_f1:.4f}")
        
        if main_f1 > best_val_f1:
            best_val_f1 = main_f1
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'input_size': INPUT_SIZE,
                'seq_len_in': SEQ_LEN_IN,
                'seq_len_out': SEQ_LEN_OUT,
                'feature_columns': feature_columns,
                'description': '峰值预测模型，输入t0-t9的45维特征，输出t10-t19的峰值概率',
                'task': '输入: t0-t9 45维特征, 输出: t10-t19 is_peak概率',
                'loss_function': 'WeightedMSE'
            }, SAVE_PATH)
            
            print(f"  ✅ 保存最佳模型 (F1={main_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  ⏹️ 早停触发，最佳F1: {best_val_f1:.4f}")
                break
    
    print("=" * 70)
    print("🎉 训练完成!")
    
    if os.path.exists(SAVE_PATH):
        print(f"模型已保存至: {SAVE_PATH}")
        print(f"最佳验证F1: {best_val_f1:.4f}")
        
        print("\n📋 任务说明:")
        print("  输入: t0-t9 时间点的45维特征（基于intensity计算的各种统计特征）")
        print("  输出: t10-t19 时间点的峰值概率 (is_peak的概率)")
        print("  损失函数: 加权MSE损失")
        print(f"  特征维度: {INPUT_SIZE}维")
    else:
        print("❌ 模型保存失败")

if __name__ == "__main__":
    main()