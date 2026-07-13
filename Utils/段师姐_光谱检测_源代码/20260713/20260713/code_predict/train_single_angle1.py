import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')
from scipy import stats

# ============================================================
# 0. Configurations
# ============================================================
WINDOW_SIZE = 10
PRED_STEPS = 10
BATCH_SIZE = 32
HIDDEN_SIZE = 128
NUM_LAYERS = 2
LEARNING_RATE = 1e-4
EPOCHS = 100
DROPOUT_RATE = 0.3
MODEL_SAVE_PATH = "angle_seq2seq_final.pth"
TRAIN_DATA_DIR = "train_angle_enhance"
VAL_DATA_DIR = "val_angle_enhance"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# ============================================================
# 1. 最终数据集类（直接预测角度值，而非差分）
# ============================================================
class FinalAngleSeq2SeqDataset(Dataset):
    def __init__(self, file_path, window_size=WINDOW_SIZE, pred_steps=PRED_STEPS):
        """
        最终数据集类 - 直接预测角度值
        使用固定归一化范围[0, 360]，确保滑动窗口一致性
        """
        # 加载数据
        df = pd.read_excel(file_path)
        if 'angle' not in df.columns:
            raise ValueError(f"文件 {file_path} 中没有 'angle' 列")
        
        angles = df['angle'].values
        angles = angles[~np.isnan(angles)]  # 移除NaN
        angles = angles % 360  # 归一化到0-360度
        
        self.data = angles
        self.window_size = window_size
        self.pred_steps = pred_steps
        self.filename = os.path.basename(file_path)
        
        # 检查数据长度是否足够
        self.min_required = window_size + pred_steps
        if len(self.data) < self.min_required:
            raise ValueError(f"文件 {self.filename} 数据长度不足: {len(self.data)} < {self.min_required}")
        
        # 计算总样本数
        self.total_samples = len(self.data) - window_size - pred_steps + 1
        
        # 固定归一化参数 - 所有文件使用相同的范围[0, 360]
        self.angle_min = 0
        self.angle_max = 360
        
        print(f"文件 {self.filename}: {len(self.data)} 个点 -> {self.total_samples} 个样本")
        print(f"  使用固定角度归一化范围: [{self.angle_min}, {self.angle_max}]")

    def __len__(self):
        return self.total_samples

    def _normalize_angle(self, angle_data):
        """归一化角度到[0,1]范围（使用固定范围0-360）"""
        return (angle_data - self.angle_min) / (self.angle_max - self.angle_min)

    def _denormalize_angle(self, normalized_angle):
        """反归一化角度"""
        return normalized_angle * (self.angle_max - self.angle_min) + self.angle_min

    def _calculate_angle_diff(self, angles):
        """计算角度差分，处理360°边界（用于特征工程）"""
        diff = np.diff(angles)
        # 处理角度跳变（360°边界）
        diff = np.where(diff > 180, diff - 360, diff)
        diff = np.where(diff < -180, diff + 360, diff)
        return diff

    def _normalize_diff(self, diff_data):
        """归一化差分到[-1,1]范围（使用固定范围）"""
        # 使用固定差分范围[-180, 180]，覆盖所有可能的差分值
        diff_min, diff_max = -180, 180
        normalized = (diff_data - diff_min) / (diff_max - diff_min)
        return normalized * 2 - 1

    def _calculate_slope(self, window_data):
        """计算窗口数据的斜率特征"""
        if len(window_data) < 2:
            return 0.0
        
        x = np.arange(len(window_data))
        slope, _, _, _, _ = stats.linregress(x, window_data)
        return slope

    def _normalize_slope(self, slope):
        """归一化斜率特征"""
        max_slope = 10.0
        normalized = slope / max_slope
        return np.clip(normalized, -1, 1)

    def __getitem__(self, idx):
        # 输入窗口: t0-t9
        x_window_original = self.data[idx:idx + self.window_size].copy()
        
        # 归一化输入窗口（使用固定范围0-360）
        x_window_normalized = self._normalize_angle(x_window_original)
        
        # 输出窗口: t10-t19（直接预测角度值）
        output_start = idx + self.window_size
        output_end = output_start + self.pred_steps
        y_window_original = self.data[output_start:output_end].copy()
        
        # 确保输出长度正确
        if len(y_window_original) < self.pred_steps:
            padding = self.pred_steps - len(y_window_original)
            y_window_original = np.pad(y_window_original, (0, padding), mode='edge')
        
        # 归一化输出窗口（使用固定范围0-360）
        y_window_normalized = self._normalize_angle(y_window_original)
        
        # 构建增强特征
        features = []
        
        # 1. 归一化角度值（固定范围0-360）
        features.append(x_window_normalized)
        
        # 2. 差分特征（使用固定范围[-180, 180]）
        if len(x_window_original) > 1:
            diff_1 = self._calculate_angle_diff(x_window_original)
            diff_1 = np.concatenate([[0], diff_1])  # 第一个差分设为0
            diff_1_normalized = self._normalize_diff(diff_1)
            features.append(diff_1_normalized)
        else:
            features.append(np.zeros_like(x_window_normalized))
        
        # 3. 窗口斜率特征
        slope = self._calculate_slope(x_window_original)
        slope_normalized = self._normalize_slope(slope)
        slope_feature = np.ones_like(x_window_normalized) * slope_normalized
        features.append(slope_feature)
        
        # 4. 移动平均特征
        if len(x_window_original) >= 3:
            ma_3 = np.convolve(x_window_original, np.ones(3)/3, mode='valid')
            ma_3 = np.concatenate([x_window_original[:2], ma_3])
            ma_3_normalized = self._normalize_angle(ma_3)
            features.append(ma_3_normalized)
        else:
            features.append(x_window_normalized)
        
        # 组合所有特征
        x_features = np.column_stack(features)
        
        # 转换为tensor
        x_tensor = torch.FloatTensor(x_features)
        y_tensor = torch.FloatTensor(y_window_normalized)  # 直接预测角度值
        
        sample_info = {
            'angle_min': self.angle_min,
            'angle_max': self.angle_max,
            'x_original': x_window_original,
            'y_original': y_window_original,
            'last_angle': x_window_original[-1],
            'filename': self.filename,
            'data_index': idx
        }
        
        return x_tensor, y_tensor, sample_info

# ============================================================
# 2. 最终模型（直接预测角度值）
# ============================================================
class FinalBiGRU(nn.Module):
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, 
                 pred_steps=PRED_STEPS, dropout_rate=DROPOUT_RATE):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_steps = pred_steps
        self.input_size = input_size
        
        # 双向GRU
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # 输出层 - 直接预测角度值
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, pred_steps),
            nn.Sigmoid()  # 输出在[0,1]范围内，对应归一化角度
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # GRU处理
        gru_out, _ = self.gru(x)  # [batch, seq_len, hidden_size*2]
        
        # 注意力机制
        attention_weights = self.attention(gru_out)  # [batch, seq_len, 1]
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # 加权求和
        context = torch.sum(gru_out * attention_weights, dim=1)  # [batch, hidden_size*2]
        
        # 输出预测（归一化角度值）
        prediction = self.output_layer(context)
        
        return prediction

# ============================================================
# 3. 最终损失函数（直接处理角度预测）
# ============================================================
class FinalAngleLossFunction(nn.Module):
    def __init__(self, pred_steps=PRED_STEPS, device=DEVICE):
        super().__init__()
        self.pred_steps = pred_steps
        self.device = device
        
        # 时间衰减权重
        weights = np.linspace(1.0, 3.0, pred_steps)
        self.weights = torch.tensor(weights, dtype=torch.float32, device=device).unsqueeze(0)
        
        self.mse_loss = nn.MSELoss(reduction='none')
        self.mae_loss = nn.L1Loss(reduction='none')
    
    def _circular_loss(self, pred_deg, target_deg):
        """圆形角度损失，处理360°边界"""
        diff = torch.abs(pred_deg - target_deg)
        circular_diff = torch.minimum(diff, 360.0 - diff)
        return circular_diff
    
    def forward(self, preds_normalized, targets_normalized, sample_infos):
        batch_size = preds_normalized.size(0)
        
        # 反归一化角度值
        preds_angle = torch.zeros_like(preds_normalized)
        targets_angle = torch.zeros_like(targets_normalized)
        
        for i in range(batch_size):
            angle_min = sample_infos['angle_min'][i]
            angle_max = sample_infos['angle_max'][i]
            
            # 从[0,1]反归一化到原始角度范围
            preds_angle[i] = preds_normalized[i] * (angle_max - angle_min) + angle_min
            targets_angle[i] = targets_normalized[i] * (angle_max - angle_min) + angle_min
        
        # 1. 加权MSE损失（角度值的MSE）
        weighted_mse = (self.mse_loss(preds_normalized, targets_normalized) * self.weights).mean()
        
        # 2. MAE损失
        mae_loss = self.mae_loss(preds_normalized, targets_normalized).mean()
        
        # 3. 圆形角度损失
        circular_loss = self._circular_loss(preds_angle, targets_angle).mean() / 180.0
        
        # 4. 角度变化的平滑性损失
        if self.pred_steps > 1:
            pred_angle_changes = torch.diff(preds_angle, dim=1)
            target_angle_changes = torch.diff(targets_angle, dim=1)
            # 处理角度跳变
            pred_angle_changes = torch.remainder(pred_angle_changes + 180, 360) - 180
            target_angle_changes = torch.remainder(target_angle_changes + 180, 360) - 180
            smoothness_loss = self.mse_loss(pred_angle_changes, target_angle_changes).mean() / 100.0
        else:
            smoothness_loss = torch.tensor(0.0, device=self.device)
        
        # 组合损失
        total_loss = (weighted_mse * 0.4 + 
                     mae_loss * 0.3 + 
                     circular_loss * 0.2 + 
                     smoothness_loss * 0.1)
        
        loss_components = {
            'angle_mse': weighted_mse.item(),
            'angle_mae': mae_loss.item(),
            'angle_circular_loss': circular_loss.item(),
            'smoothness_loss': smoothness_loss.item()
        }
        
        return total_loss, loss_components

# ============================================================
# 4. 训练函数（适配最终模型）
# ============================================================
def train_final_model(model, train_loader, val_loader, epochs=EPOCHS):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    criterion = FinalAngleLossFunction(pred_steps=PRED_STEPS, device=DEVICE)
    
    best_val_loss = float('inf')
    train_history = {'total': [], 'angle_mse': [], 'angle_mae': [], 'angle_circular': [], 'smoothness': []}
    val_history = {'total': [], 'angle_mse': [], 'angle_mae': [], 'angle_circular': [], 'smoothness': []}
    
    patience = 10
    patience_counter = 0
    
    print("开始训练...")
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_losses = {'total': 0.0, 'angle_mse': 0.0, 'angle_mae': 0.0, 
                       'angle_circular': 0.0, 'smoothness': 0.0}
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [训练]')
        for batch_idx, (x, y, sample_infos) in enumerate(pbar):
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            preds = model(x)
            total_loss, loss_components = criterion(preds, y, sample_infos)
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses['total'] += total_loss.item()
            train_losses['angle_mse'] += loss_components['angle_mse']
            train_losses['angle_mae'] += loss_components['angle_mae']
            train_losses['angle_circular'] += loss_components['angle_circular_loss']
            train_losses['smoothness'] += loss_components['smoothness_loss']
            
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    '总损失': f'{total_loss.item():.4f}',
                    '角度MSE': f'{loss_components["angle_mse"]:.4f}',
                    '角度MAE': f'{loss_components["angle_mae"]:.4f}',
                })
        
        # 验证阶段
        model.eval()
        val_losses = {'total': 0.0, 'angle_mse': 0.0, 'angle_mae': 0.0, 
                     'angle_circular': 0.0, 'smoothness': 0.0}
        
        with torch.no_grad():
            for x, y, sample_infos in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                preds = model(x)
                total_loss, loss_components = criterion(preds, y, sample_infos)
                
                val_losses['total'] += total_loss.item()
                val_losses['angle_mse'] += loss_components['angle_mse']
                val_losses['angle_mae'] += loss_components['angle_mae']
                val_losses['angle_circular'] += loss_components['angle_circular_loss']
                val_losses['smoothness'] += loss_components['smoothness_loss']
        
        # 计算平均损失
        for key in train_losses:
            train_losses[key] /= len(train_loader)
            val_losses[key] /= len(val_loader)
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        for key in train_history:
            train_history[key].append(train_losses[key])
            val_history[key].append(val_losses[key])
        
        print(f'Epoch {epoch+1}:')
        print(f'  训练 - 总损失: {train_losses["total"]:.6f}, 角度MSE: {train_losses["angle_mse"]:.6f}')
        print(f'  验证 - 总损失: {val_losses["total"]:.6f}, 角度MSE: {val_losses["angle_mse"]:.6f}')
        print(f'  学习率: {optimizer.param_groups[0]["lr"]:.2e}')
        
        # 保存最佳模型
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': best_val_loss,
                'train_history': train_history,
                'val_history': val_history,
                'model_config': {
                    'hidden_size': HIDDEN_SIZE,
                    'num_layers': NUM_LAYERS,
                    'pred_steps': PRED_STEPS,
                    'input_size': model.input_size,
                    'window_size': WINDOW_SIZE
                },
                'normalization_params': {
                    'angle_min': 0,
                    'angle_max': 360
                }
            }, MODEL_SAVE_PATH)
            patience_counter = 0
            print(f'  ✓ 保存最佳模型 (损失: {best_val_loss:.6f})')
        else:
            patience_counter += 1
        
        # 早期停止
        if patience_counter >= patience:
            print(f'  🛑 早期停止 (连续 {patience} 轮未改进)')
            break
    
    return train_history, val_history

# ============================================================
# 5. 数据加载和工具函数
# ============================================================
def custom_collate_fn(batch):
    """自定义collate函数"""
    x_list, y_list, info_list = zip(*batch)
    
    x_batch = torch.stack(x_list)
    y_batch = torch.stack(y_list)
    
    keys = info_list[0].keys()
    info_batch = {key: [info[key] for info in info_list] for key in keys}
    
    return x_batch, y_batch, info_batch

def load_final_datasets(train_dir, val_dir):
    """加载最终数据集"""
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise ValueError("数据目录不存在")
    
    def load_dir_datasets(directory, dataset_type):
        files = [f for f in os.listdir(directory) if f.endswith('.xlsx')]
        datasets = []
        valid_files = 0
        
        for file in files:
            file_path = os.path.join(directory, file)
            try:
                dataset = FinalAngleSeq2SeqDataset(
                    file_path,
                    window_size=WINDOW_SIZE,
                    pred_steps=PRED_STEPS
                )
                datasets.append(dataset)
                valid_files += 1
                print(f"  ✓ 加载 {file}: {len(dataset)} 样本")
            except Exception as e:
                print(f"  ✗ 跳过 {file}: {e}")
                continue
        
        print(f"{dataset_type}集: 成功加载 {valid_files}/{len(files)} 个文件")
        
        if len(datasets) == 0:
            raise ValueError(f"没有有效的{dataset_type}数据文件")
        
        return ConcatDataset(datasets)
    
    print("加载训练数据...")
    train_dataset = load_dir_datasets(train_dir, "训练")
    print("加载验证数据...")
    val_dataset = load_dir_datasets(val_dir, "验证")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, collate_fn=custom_collate_fn)
    
    print(f"训练样本总数: {len(train_dataset)}")
    print(f"验证样本总数: {len(val_dataset)}")
    
    return train_loader, val_loader

def plot_training_history(train_history, val_history):
    """绘制训练历史"""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_history['total'], label='训练总损失')
    plt.plot(val_history['total'], label='验证总损失')
    plt.title('总损失')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(train_history['angle_mse'], label='训练角度MSE')
    plt.plot(val_history['angle_mse'], label='验证角度MSE')
    plt.title('角度MSE损失')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(train_history['angle_mae'], label='训练角度MAE')
    plt.plot(val_history['angle_mae'], label='验证角度MAE')
    plt.title('角度MAE损失')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history_final.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================
# 6. 主函数
# ============================================================
def main():
    print("🚀 启动最终角度序列预测训练")
    print("=" * 60)
    print(f"输入: t0-t9的角度值 + 增强特征")
    print("输出: t10-t19的角度值（直接预测）")
    print(f"归一化策略: 固定范围[0, 360]，确保滑动窗口一致性")
    
    try:
        # 加载数据
        train_loader, val_loader = load_final_datasets(TRAIN_DATA_DIR, VAL_DATA_DIR)
        
        # 确定输入维度
        sample_x, sample_y, sample_info = next(iter(train_loader))
        input_dim = sample_x.shape[2]
        
        print(f"输入特征维度: {input_dim}")
        
        # 创建模型
        model = FinalBiGRU(
            input_size=input_dim,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS
        ).to(DEVICE)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")
        
        # 训练模型
        train_history, val_history = train_final_model(model, train_loader, val_loader)
        
        # 绘制训练历史
        plot_training_history(train_history, val_history)
        
        print("🎉 训练完成！")
        print(f"最佳模型已保存至: {MODEL_SAVE_PATH}")
        
    except Exception as e:
        print(f"训练过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()