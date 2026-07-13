import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 配置参数（与训练代码保持一致）
# ============================================================
WINDOW_SIZE = 10
PRED_STEPS = 10
BATCH_SIZE = 1
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT_RATE = 0.3
MODEL_SAVE_PATH = "angle_seq2seq_final.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# ============================================================
# 1. 预测用的数据集类（基于训练代码修改）
# ============================================================
class PredictionDataset(Dataset):
    def __init__(self, file_path, window_size=WINDOW_SIZE, pred_steps=PRED_STEPS):
        """
        预测用的数据集类 - 直接预测角度值
        使用固定归一化范围[0, 360]
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
        
        # 预测模式下只需要窗口大小即可
        if len(self.data) < self.window_size:
            raise ValueError(f"文件 {self.filename} 数据长度不足: {len(self.data)} < {self.window_size}")
        
        # 计算总样本数（从第10个点开始预测）
        self.total_samples = max(0, len(self.data) - window_size + 1)
        
        # 固定归一化参数 - 所有文件使用相同的范围[0, 360]
        self.angle_min = 0
        self.angle_max = 360
        
        print(f"文件 {self.filename}: {len(self.data)} 个点 -> {self.total_samples} 个预测样本")
        print(f"  使用固定角度归一化范围: [{self.angle_min}, {self.angle_max}]")

    def __len__(self):
        return self.total_samples

    def _normalize_angle(self, angle_data):
        """归一化角度到[0,1]范围（使用固定范围0-360）"""
        return (angle_data - self.angle_min) / (self.angle_max - self.angle_min)

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
        slope = np.polyfit(x, window_data, 1)[0]  # 线性回归斜率
        return slope

    def _normalize_slope(self, slope):
        """归一化斜率特征"""
        max_slope = 10.0
        normalized = slope / max_slope
        return np.clip(normalized, -1, 1)

    def __getitem__(self, idx):
        # 输入窗口: 从idx开始的window_size个点
        x_window_original = self.data[idx:idx + self.window_size].copy()
        
        # 归一化输入窗口（使用固定范围0-360）
        x_window_normalized = self._normalize_angle(x_window_original)
        
        # 构建增强特征（与训练时相同）
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
        
        # 预测模式下没有真实的y，使用占位符
        y_tensor = torch.zeros(self.pred_steps)
        
        sample_info = {
            'angle_min': self.angle_min,
            'angle_max': self.angle_max,
            'x_original': x_window_original,
            'last_angle': x_window_original[-1],
            'filename': self.filename,
            'data_index': idx,
            'window_start': idx,
            'window_end': idx + self.window_size - 1
        }
        
        return x_tensor, y_tensor, sample_info

# ============================================================
# 2. 模型定义（与训练代码相同）
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
# 3. 预测函数
# ============================================================
def predict_single_file(model, file_path, output_csv_path):
    """
    对单个文件进行预测
    """
    print(f"\n开始预测文件: {file_path}")
    
    # 创建预测数据集
    try:
        dataset = PredictionDataset(file_path)
    except Exception as e:
        print(f"创建数据集失败: {e}")
        return None
    
    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, 
                           collate_fn=custom_collate_fn)
    
    # 存储预测结果
    all_predictions = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y, sample_infos) in enumerate(dataloader):
            x = x.to(DEVICE)
            
            # 预测
            preds_normalized = model(x)
            
            # 将预测结果转换到CPU
            preds_normalized = preds_normalized.cpu().numpy()
            
            # 处理每个样本
            for i in range(len(sample_infos['filename'])):
                # 获取样本信息
                angle_min = sample_infos['angle_min'][i]
                angle_max = sample_infos['angle_max'][i]
                window_start = sample_infos['window_start'][i]
                window_end = sample_infos['window_end'][i]
                x_original = sample_infos['x_original'][i]
                last_angle = sample_infos['last_angle'][i]
                
                # 反归一化角度预测（从[0,1]到[0,360]）
                pred_angles = preds_normalized[i] * (angle_max - angle_min) + angle_min
                
                # 确保角度在0-360范围内
                pred_angles = pred_angles % 360
                
                # 存储预测结果
                prediction_info = {
                    'window_index': batch_idx * BATCH_SIZE + i,
                    'window_start': window_start,
                    'window_end': window_end,
                    'last_angle': last_angle,  # 窗口最后一个角度（t9）
                    'predicted_angles': pred_angles,  # t10-t19的预测角度
                    'input_window': x_original,  # t0-t9的输入角度
                }
                all_predictions.append(prediction_info)
            
            # 打印进度
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
                print(f"  已处理 {batch_idx + 1}/{len(dataloader)} 个窗口")
    
    # 保存预测结果到CSV
    save_predictions_to_csv(all_predictions, output_csv_path, file_path)
    
    return all_predictions

def custom_collate_fn(batch):
    """自定义collate函数"""
    x_list, y_list, info_list = zip(*batch)
    
    x_batch = torch.stack(x_list)
    y_batch = torch.stack(y_list)
    
    keys = info_list[0].keys()
    info_batch = {key: [info[key] for info in info_list] for key in keys}
    
    return x_batch, y_batch, info_batch

def save_predictions_to_csv(predictions, output_path, original_file_path):
    """
    将预测结果保存到CSV文件，格式为：
    window_index, prediction_index, time_index, predicted_angle, actual_angle, window_start_index, last_window_angle
    """
    # 提取原始数据用于参考
    df_original = pd.read_excel(original_file_path)
    original_angles = df_original['angle'].values
    original_angles = original_angles[~np.isnan(original_angles)]  # 移除NaN
    
    # 准备CSV数据
    csv_data = []
    
    for pred in predictions:
        window_index = pred['window_index']
        window_start = pred['window_start']
        window_end = pred['window_end']
        last_window_angle = pred['last_angle']
        predicted_angles = pred['predicted_angles']
        
        # 为每个预测点创建一行记录
        for j in range(PRED_STEPS):
            # 计算预测的时间点索引
            time_index = window_end + 1 + j
            
            # 获取实际角度（如果存在）
            actual_angle = np.nan
            if time_index < len(original_angles):
                actual_angle = original_angles[time_index]
            
            # 创建数据行
            row = {
                'window_index': window_index,
                'prediction_index': j,  # 0表示t10，1表示t11，...，9表示t19
                'time_index': time_index,
                'predicted_angle': predicted_angles[j],
                'actual_angle': actual_angle,
                'window_start_index': window_start,
                'last_window_angle': last_window_angle
            }
            
            csv_data.append(row)
    
    # 创建DataFrame并保存
    df_output = pd.DataFrame(csv_data)
    
    # 确保列的顺序正确
    column_order = [
        'window_index', 
        'prediction_index', 
        'time_index', 
        'predicted_angle', 
        'actual_angle', 
        'window_start_index', 
        'last_window_angle'
    ]
    df_output = df_output[column_order]
    
    df_output.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"✓ 预测结果已保存至: {output_path}")
    print(f"  共 {len(predictions)} 个窗口的预测结果")
    print(f"  每个窗口预测未来 {PRED_STEPS} 步角度值")
    print(f"  总预测点数: {len(csv_data)}")
    
    # 计算并显示统计信息
    if len(csv_data) > 0:
        # 只计算有实际值的预测点的误差
        valid_predictions = [p for p in csv_data if not np.isnan(p['actual_angle'])]
        if valid_predictions:
            errors = []
            for p in valid_predictions:
                pred_angle = p['predicted_angle']
                actual_angle = p['actual_angle']
                # 计算圆形误差（处理360°边界）
                error = min(abs(pred_angle - actual_angle), 360 - abs(pred_angle - actual_angle))
                errors.append(error)
            
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            print(f"  平均预测误差: {mean_error:.2f}°")
            print(f"  预测误差标准差: {std_error:.2f}°")

# ============================================================
# 4. 主预测函数
# ============================================================
def main_predict(input_excel_file, output_csv_file):
    """
    主预测函数
    """
    print("🚀 开始角度预测")
    print("=" * 50)
    print(f"预测模式: 攒够{WINDOW_SIZE}个点开始预测，每个窗口预测{PRED_STEPS}个点")
    print(f"输出格式: window_index, prediction_index, time_index, predicted_angle, actual_angle, window_start_index, last_window_angle")
    
    # 检查模型文件是否存在
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"错误: 模型文件 {MODEL_SAVE_PATH} 不存在")
        print("请先运行训练代码训练模型")
        return
    
    # 检查输入文件是否存在
    if not os.path.exists(input_excel_file):
        print(f"错误: 输入文件 {input_excel_file} 不存在")
        return
    
    try:
        # 加载模型检查点
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
        model_config = checkpoint['model_config']
        
        print("加载模型配置:")
        print(f"  输入窗口大小: {model_config['window_size']}")
        print(f"  预测步长: {model_config['pred_steps']}")
        print(f"  隐藏层大小: {model_config['hidden_size']}")
        print(f"  网络层数: {model_config['num_layers']}")
        
        # 创建模型（需要先确定输入维度）
        # 由于输入维度取决于特征工程，我们需要先查看一个样本来确定
        temp_dataset = PredictionDataset(input_excel_file)
        if len(temp_dataset) == 0:
            print("错误: 数据长度不足，无法形成完整窗口")
            return
            
        sample_x, sample_y, sample_info = temp_dataset[0]
        input_dim = sample_x.shape[1]
        
        print(f"确定输入特征维度: {input_dim}")
        
        # 创建模型实例
        model = FinalBiGRU(
            input_size=input_dim,
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            pred_steps=model_config['pred_steps']
        ).to(DEVICE)
        
        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ 模型加载成功")
        
        # 进行预测
        predictions = predict_single_file(model, input_excel_file, output_csv_file)
        
        if predictions:
            print(f"\n🎉 预测完成!")
            print(f"输入文件: {input_excel_file}")
            print(f"输出文件: {output_csv_file}")
            print(f"总预测窗口数: {len(predictions)}")
            
            # 显示最后一个窗口的预测结果作为示例
            last_pred = predictions[-1]
            print(f"\n最后一个窗口预测示例 (窗口 {last_pred['window_index']}):")
            print(f"  输入窗口: t{last_pred['window_start']}-t{last_pred['window_end']}")
            print(f"  最后角度 t{last_pred['window_end']}: {last_pred['last_angle']:.2f}°")
            print(f"  预测角度 t{last_pred['window_end']+1}-t{last_pred['window_end']+PRED_STEPS}:")
            for j in range(min(5, PRED_STEPS)):  # 只显示前5个预测
                print(f"    t{last_pred['window_end']+1+j}: {last_pred['predicted_angles'][j]:.2f}°")
            if PRED_STEPS > 5:
                print(f"    ... (共{PRED_STEPS}个预测点)")
        
    except Exception as e:
        print(f"预测过程中出错: {e}")
        import traceback
        traceback.print_exc()

# ============================================================
# 5. 使用示例
# ============================================================
if __name__ == "__main__":
    # 配置输入输出文件路径
    INPUT_EXCEL_FILE = "BN_BN3_delete_1_rows_aug1_0.xlsx"  # 请替换为你的Excel文件路径
    OUTPUT_CSV_FILE = "prediction_results_final1.csv"
    
    # 执行预测
    main_predict(INPUT_EXCEL_FILE, OUTPUT_CSV_FILE)