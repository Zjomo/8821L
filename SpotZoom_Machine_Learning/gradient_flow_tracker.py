"""
梯度流追踪模块 - 基于CellPose创新理念
用于spot运动的连续追踪和预测
"""

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.interpolate import RectBivariateSpline
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class GradientFlowTracker:
    """
    基于CellPose风格的动态梯度流追踪器
    
    创新点：
    1. 使用流场表示spot运动的潜在方向
    2. 双线性插值优化梯度计算
    3. 支持多尺度追踪
    """
    
    def __init__(self, diameter: int = 30, flow_threshold: float = 0.4, 
                 use_bilinear: bool = True):
        """
        初始化梯度流追踪器
        
        Args:
            diameter: spot直径估计（像素）
            flow_threshold: 流场阈值
            use_bilinear: 是否使用双线性插值
        """
        self.diameter = diameter
        self.flow_threshold = flow_threshold
        self.use_bilinear = use_bilinear
        self._flow_x = None
        self._flow_y = None
        
    def compute_flow_field(self, image: np.ndarray, 
                          spots: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算图像的流场
        
        Args:
            image: 输入图像 (H, W)
            spots: 当前检测到的spot位置 (N, 2) [y, x]
            
        Returns:
            flow_x, flow_y: x和y方向的流场
        """
        # 1. 图像预处理 - 高斯平滑
        sigma = self.diameter / 4
        img_smooth = gaussian_filter(image.astype(float), sigma=sigma)
        
        # 2. 计算梯度
        dy, dx = np.gradient(img_smooth)
        
        # 3. 初始化流场
        flow_x = np.zeros_like(image, dtype=float)
        flow_y = np.zeros_like(image, dtype=float)
        
        if spots is not None and len(spots) > 0:
            # 4. 基于spot位置构建吸引流场
            y_grid, x_grid = np.mgrid[:image.shape[0], :image.shape[1]]
            
            for spot in spots:
                y, x = spot[0], spot[1]
                
                # 计算到spot的距离
                dist_y = y_grid - y
                dist_x = x_grid - x
                dist = np.sqrt(dist_x**2 + dist_y**2)
                
                # 高斯加权流向中心（吸引流）
                weight = np.exp(-dist**2 / (2 * (self.diameter/2)**2))
                flow_x += -dist_x * weight
                flow_y += -dist_y * weight
        else:
            # 如果没有spot，使用梯度流
            flow_x = -dx
            flow_y = -dy
        
        # 5. 归一化流场
        flow_mag = np.sqrt(flow_x**2 + flow_y**2)
        flow_mag[flow_mag < 1e-10] = 1  # 避免除零
        flow_x = flow_x / flow_mag
        flow_y = flow_y / flow_mag
        
        # 6. 应用阈值
        flow_mag = np.sqrt(flow_x**2 + flow_y**2)
        mask = flow_mag > self.flow_threshold
        flow_x = flow_x * mask
        flow_y = flow_y * mask
        
        self._flow_x = flow_x
        self._flow_y = flow_y
        
        return flow_x, flow_y
    
    def track_with_flow(self, spots_prev: np.ndarray, 
                       dt: float = 1.0) -> np.ndarray:
        """
        使用流场追踪spot
        
        Args:
            spots_prev: 上一帧的spot位置 (N, 2) [y, x]
            dt: 时间步长
            
        Returns:
            spots_tracked: 追踪后的spot位置
        """
        if self._flow_x is None or self._flow_y is None:
            logger.warning("流场未计算，请先调用compute_flow_field")
            return spots_prev
        
        spots_tracked = []
        
        for spot in spots_prev:
            y, x = spot[0], spot[1]
            
            if self.use_bilinear:
                # 使用双线性插值获取流场值
                dx = self._bilinear_interpolate(self._flow_x, y, x) * dt
                dy = self._bilinear_interpolate(self._flow_y, y, x) * dt
            else:
                # 最近邻插值
                yi, xi = int(round(y)), int(round(x))
                if (0 <= yi < self._flow_x.shape[0] and 
                    0 <= xi < self._flow_x.shape[1]):
                    dx = self._flow_x[yi, xi] * dt
                    dy = self._flow_y[yi, xi] * dt
                else:
                    dx, dy = 0, 0
            
            new_spot = np.array([y + dy, x + dx])
            spots_tracked.append(new_spot)
        
        return np.array(spots_tracked)
    
    def _bilinear_interpolate(self, field: np.ndarray, y: float, 
                             x: float) -> float:
        """
        双线性插值
        
        Args:
            field: 2D场
            y, x: 插值位置
            
        Returns:
            插值结果
        """
        y0, x0 = int(np.floor(y)), int(np.floor(x))
        y1, x1 = y0 + 1, x0 + 1
        
        # 边界检查
        if y0 < 0 or y1 >= field.shape[0] or x0 < 0 or x1 >= field.shape[1]:
            # 退化为最近邻
            yi, xi = int(round(y)), int(round(x))
            if 0 <= yi < field.shape[0] and 0 <= xi < field.shape[1]:
                return field[yi, xi]
            return 0.0
        
        # 双线性插值
        wy = y - y0
        wx = x - x0
        
        value = ((1 - wy) * (1 - wx) * field[y0, x0] +
                 wy * (1 - wx) * field[y1, x0] +
                 (1 - wy) * wx * field[y0, x1] +
                 wy * wx * field[y1, x1])
        
        return value
    
    def predict_trajectory(self, spot_init: np.ndarray, 
                          n_steps: int = 10) -> np.ndarray:
        """
        预测spot的轨迹
        
        Args:
            spot_init: 初始位置 [y, x]
            n_steps: 预测步数
            
        Returns:
            trajectory: 预测的轨迹 (n_steps+1, 2)
        """
        if self._flow_x is None or self._flow_y is None:
            logger.warning("流场未计算")
            return np.array([spot_init])
        
        trajectory = [spot_init.copy()]
        current_pos = spot_init.copy()
        
        for _ in range(n_steps):
            y, x = current_pos[0], current_pos[1]
            
            # 获取流场方向
            dx = self._bilinear_interpolate(self._flow_x, y, x)
            dy = self._bilinear_interpolate(self._flow_y, y, x)
            
            # 更新位置
            current_pos[0] += dy
            current_pos[1] += dx
            
            trajectory.append(current_pos.copy())
        
        return np.array(trajectory)
    
    def compute_flow_consistency(self, spots_current: np.ndarray,
                                spots_previous: np.ndarray) -> float:
        """
        计算流场一致性分数
        
        衡量实际spot运动与流场预测的一致性
        
        Args:
            spots_current: 当前帧spot位置
            spots_previous: 上一帧spot位置
            
        Returns:
            consistency_score: 一致性分数 (0-1)
        """
        if len(spots_current) == 0 or len(spots_previous) == 0:
            return 0.0
        
        # 简单的最近邻匹配
        total_consistency = 0.0
        matched_pairs = 0
        
        for prev_spot in spots_previous:
            # 找到最近的当前spot
            distances = np.sqrt(np.sum((spots_current - prev_spot)**2, axis=1))
            nearest_idx = np.argmin(distances)
            nearest_dist = distances[nearest_idx]
            
            if nearest_dist < self.diameter:  # 匹配成功
                curr_spot = spots_current[nearest_idx]
                
                # 实际运动向量
                actual_dy = curr_spot[0] - prev_spot[0]
                actual_dx = curr_spot[1] - prev_spot[1]
                
                # 流场预测
                pred_dx = self._bilinear_interpolate(self._flow_x, 
                                                     prev_spot[0], prev_spot[1])
                pred_dy = self._bilinear_interpolate(self._flow_y, 
                                                     prev_spot[0], prev_spot[1])
                
                # 计算一致性（余弦相似度）
                actual_mag = np.sqrt(actual_dx**2 + actual_dy**2)
                pred_mag = np.sqrt(pred_dx**2 + pred_dy**2)
                
                if actual_mag > 0 and pred_mag > 0:
                    cos_sim = ((actual_dx * pred_dx + actual_dy * pred_dy) / 
                              (actual_mag * pred_mag))
                    total_consistency += (cos_sim + 1) / 2  # 映射到0-1
                    matched_pairs += 1
        
        if matched_pairs > 0:
            return total_consistency / matched_pairs
        return 0.0


class MultiScaleFlowTracker:
    """
    多尺度梯度流追踪器
    
    在不同尺度上计算流场，提高追踪鲁棒性
    """
    
    def __init__(self, diameters: List[int] = [15, 30, 60]):
        """
        初始化多尺度追踪器
        
        Args:
            diameters: 不同尺度的直径列表
        """
        self.diameters = diameters
        self.trackers = {d: GradientFlowTracker(diameter=d) for d in diameters}
        
    def compute_multi_scale_flow(self, image: np.ndarray,
                                 spots: Optional[np.ndarray] = None
                                 ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        计算多尺度流场
        
        Args:
            image: 输入图像
            spots: spot位置
            
        Returns:
            flows: 各尺度的流场字典 {diameter: (flow_x, flow_y)}
        """
        flows = {}
        for diameter, tracker in self.trackers.items():
            flow_x, flow_y = tracker.compute_flow_field(image, spots)
            flows[diameter] = (flow_x, flow_y)
        return flows
    
    def track_multi_scale(self, spots_prev: np.ndarray,
                         image: np.ndarray) -> Dict[int, np.ndarray]:
        """
        多尺度追踪
        
        Args:
            spots_prev: 上一帧spot位置
            image: 当前帧图像
            
        Returns:
            tracked_spots: 各尺度的追踪结果
        """
        results = {}
        for diameter, tracker in self.trackers.items():
            tracker.compute_flow_field(image, spots_prev)
            tracked = tracker.track_with_flow(spots_prev)
            results[diameter] = tracked
        return results
    
    def ensemble_track(self, spots_prev: np.ndarray,
                      image: np.ndarray,
                      weights: Optional[List[float]] = None) -> np.ndarray:
        """
        集成多尺度追踪结果
        
        Args:
            spots_prev: 上一帧spot位置
            image: 当前帧图像
            weights: 各尺度的权重
            
        Returns:
            ensemble_result: 集成后的追踪结果
        """
        if weights is None:
            weights = [1.0] * len(self.diameters)
        
        multi_scale_results = self.track_multi_scale(spots_prev, image)
        
        # 加权平均
        ensemble = np.zeros_like(spots_prev, dtype=float)
        total_weight = sum(weights)
        
        for (diameter, tracked), weight in zip(multi_scale_results.items(), weights):
            if len(tracked) == len(spots_prev):
                ensemble += tracked * weight
        
        ensemble /= total_weight
        
        return ensemble


# 便捷函数
def create_flow_tracker(diameter: int = 30, **kwargs) -> GradientFlowTracker:
    """创建流场追踪器的便捷函数"""
    return GradientFlowTracker(diameter=diameter, **kwargs)


def create_multi_scale_tracker(diameters: List[int] = [15, 30, 60]
                               ) -> MultiScaleFlowTracker:
    """创建多尺度追踪器的便捷函数"""
    return MultiScaleFlowTracker(diameters=diameters)
