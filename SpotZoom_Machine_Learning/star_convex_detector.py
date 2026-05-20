"""
星形凸Spot检测模块 - 基于StarDist创新理念
用于不规则形状spot的检测和分割
"""

import numpy as np
from scipy.ndimage import maximum_filter, gaussian_filter, label
from scipy.spatial.distance import cdist
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class StarConvexSpotDetector:
    """
    基于StarDist的星形凸spot检测器
    
    创新点：
    1. 使用射线距离表示对象边界
    2. 星形凸形状约束
    3. 非极大值抑制优化
    """
    
    def __init__(self, n_rays: int = 32, nms_threshold: float = 0.5,
                 min_radius: float = 3.0, max_radius: float = 50.0):
        """
        初始化星形凸检测器
        
        Args:
            n_rays: 射线数量（决定形状分辨率）
            nms_threshold: 非极大值抑制阈值
            min_radius: 最小半径
            max_radius: 最大搜索半径
        """
        self.n_rays = n_rays
        self.nms_threshold = nms_threshold
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.angles = np.linspace(0, 2*np.pi, n_rays, endpoint=False)
        
    def compute_ray_distances(self, image: np.ndarray, 
                             center: Tuple[float, float]) -> np.ndarray:
        """
        计算从中心沿各射线的距离
        
        Args:
            image: 输入图像
            center: 中心点 (y, x)
            
        Returns:
            distances: 沿各射线的距离
        """
        cy, cx = center
        distances = np.zeros(self.n_rays)
        
        for i, angle in enumerate(self.angles):
            # 沿射线采样
            for r in np.arange(self.min_radius, self.max_radius, 0.5):
                y = cy + r * np.sin(angle)
                x = cx + r * np.cos(angle)
                
                # 边界检查
                if (y < 0 or y >= image.shape[0] - 1 or 
                    x < 0 or x >= image.shape[1] - 1):
                    distances[i] = r - 0.5
                    break
                
                # 双线性插值获取强度
                yi, xi = int(y), int(x)
                dy, dx = y - yi, x - xi
                
                intensity = ((1-dy) * (1-dx) * image[yi, xi] +
                            dy * (1-dx) * image[yi+1, xi] +
                            (1-dy) * dx * image[yi, xi+1] +
                            dy * dx * image[yi+1, xi+1])
                
                # 检测边缘（强度下降或梯度变化）
                if r > self.min_radius:
                    prev_y = cy + (r-0.5) * np.sin(angle)
                    prev_x = cx + (r-0.5) * np.cos(angle)
                    prev_intensity = self._interpolate(image, prev_y, prev_x)
                    
                    # 强度下降超过50%认为是边界
                    if intensity < prev_intensity * 0.5:
                        distances[i] = r - 0.5
                        break
                    
                    # 或者梯度很大
                    gradient = abs(intensity - prev_intensity)
                    if gradient > prev_intensity * 0.3:
                        distances[i] = r - 0.5
                        break
            else:
                distances[i] = self.max_radius
        
        return distances
    
    def _interpolate(self, image: np.ndarray, y: float, x: float) -> float:
        """双线性插值"""
        yi, xi = int(y), int(x)
        if yi < 0 or yi >= image.shape[0]-1 or xi < 0 or xi >= image.shape[1]-1:
            return 0.0
        
        dy, dx = y - yi, x - xi
        return ((1-dy) * (1-dx) * image[yi, xi] +
                dy * (1-dx) * image[yi+1, xi] +
                (1-dy) * dx * image[yi, xi+1] +
                dy * dx * image[yi+1, xi+1])
    
    def distances_to_polygon(self, center: Tuple[float, float], 
                            distances: np.ndarray) -> np.ndarray:
        """
        将射线距离转换为多边形顶点
        
        Args:
            center: 中心点
            distances: 射线距离
            
        Returns:
            polygon: 多边形顶点 (n_rays, 2) [y, x]
        """
        cy, cx = center
        polygon = np.zeros((self.n_rays, 2))
        
        for i, (angle, dist) in enumerate(zip(self.angles, distances)):
            polygon[i, 0] = cy + dist * np.sin(angle)
            polygon[i, 1] = cx + dist * np.cos(angle)
        
        return polygon
    
    def compute_shape_features(self, distances: np.ndarray) -> Dict:
        """
        计算形状特征
        
        Args:
            distances: 射线距离
            
        Returns:
            features: 形状特征字典
        """
        features = {}
        
        # 基本统计
        features['mean_radius'] = np.mean(distances)
        features['std_radius'] = np.std(distances)
        features['min_radius'] = np.min(distances)
        features['max_radius'] = np.max(distances)
        
        # 圆度（标准差与均值的比）
        features['circularity'] = 1.0 - features['std_radius'] / (features['mean_radius'] + 1e-10)
        
        # 面积估计（使用多边形面积公式）
        polygon = self.distances_to_polygon((0, 0), distances)
        features['area'] = self._polygon_area(polygon)
        
        # 周长
        features['perimeter'] = self._polygon_perimeter(polygon)
        
        # 紧凑度
        if features['perimeter'] > 0:
            features['compactness'] = (4 * np.pi * features['area'] / 
                                      features['perimeter']**2)
        else:
            features['compactness'] = 0
        
        return features
    
    def _polygon_area(self, polygon: np.ndarray) -> float:
        """计算多边形面积（鞋带公式）"""
        n = len(polygon)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += polygon[i, 0] * polygon[j, 1]
            area -= polygon[j, 0] * polygon[i, 1]
        return abs(area) / 2.0
    
    def _polygon_perimeter(self, polygon: np.ndarray) -> float:
        """计算多边形周长"""
        n = len(polygon)
        perimeter = 0.0
        for i in range(n):
            j = (i + 1) % n
            perimeter += np.sqrt(np.sum((polygon[i] - polygon[j])**2))
        return perimeter
    
    def detect(self, image: np.ndarray, 
               prob_threshold: float = 0.5,
               min_distance: int = 10) -> List[Dict]:
        """
        检测图像中的星形凸spot
        
        Args:
            image: 输入图像
            prob_threshold: 概率阈值
            min_distance: 候选点之间的最小距离
            
        Returns:
            spots: 检测到的spot列表
        """
        # 1. 计算概率图
        prob_map = self._compute_probability_map(image)
        
        # 2. 找到局部最大值作为候选中心
        local_max = maximum_filter(prob_map, size=min_distance) == prob_map
        candidates = np.argwhere((prob_map > prob_threshold) & local_max)
        
        logger.info(f"找到 {len(candidates)} 个候选点")
        
        # 3. 为每个候选计算星形凸形状
        spots = []
        for cy, cx in candidates:
            center = (float(cy), float(cx))
            
            # 计算射线距离
            distances = self.compute_ray_distances(image, center)
            
            # 过滤太小的spot
            mean_radius = np.mean(distances)
            if mean_radius < self.min_radius:
                continue
            
            # 计算形状特征
            features = self.compute_shape_features(distances)
            
            # 生成多边形
            polygon = self.distances_to_polygon(center, distances)
            
            spots.append({
                'center': center,
                'distances': distances,
                'polygon': polygon,
                'features': features,
                'probability': prob_map[cy, cx]
            })
        
        logger.info(f"计算形状后剩余 {len(spots)} 个spot")
        
        # 4. 非极大值抑制
        spots = self._nms(spots)
        
        logger.info(f"NMS后剩余 {len(spots)} 个spot")
        
        return spots
    
    def _compute_probability_map(self, image: np.ndarray) -> np.ndarray:
        """
        计算概率图
        
        使用高斯滤波和局部对比度增强
        """
        # 高斯平滑
        smoothed = gaussian_filter(image.astype(float), sigma=2)
        
        # 局部对比度归一化
        local_mean = gaussian_filter(smoothed, sigma=10)
        local_std = np.sqrt(gaussian_filter((smoothed - local_mean)**2, sigma=10))
        
        # 避免除零
        local_std[local_std < 1e-10] = 1e-10
        
        # 标准化
        normalized = (smoothed - local_mean) / local_std
        
        # 映射到0-1
        prob_map = (normalized - normalized.min()) / (normalized.max() - normalized.min() + 1e-10)
        
        return prob_map
    
    def _nms(self, spots: List[Dict]) -> List[Dict]:
        """
        非极大值抑制
        
        基于spot之间的重叠程度进行抑制
        """
        if len(spots) == 0:
            return spots
        
        # 按概率排序
        spots = sorted(spots, key=lambda x: x['probability'], reverse=True)
        keep = []
        
        for spot in spots:
            # 检查与已保留spot的重叠
            overlap = False
            
            for kept in keep:
                # 计算中心距离
                dist = np.sqrt((spot['center'][0] - kept['center'][0])**2 + 
                              (spot['center'][1] - kept['center'][1])**2)
                
                # 计算平均半径
                spot_radius = spot['features']['mean_radius']
                kept_radius = kept['features']['mean_radius']
                
                # 如果距离小于阈值，认为是重叠
                if dist < (spot_radius + kept_radius) * self.nms_threshold:
                    overlap = True
                    break
            
            if not overlap:
                keep.append(spot)
        
        return keep
    
    def create_mask(self, spot: Dict, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        从spot创建二值掩码
        
        Args:
            spot: spot字典
            image_shape: 图像形状
            
        Returns:
            mask: 二值掩码
        """
        mask = np.zeros(image_shape, dtype=bool)
        polygon = spot['polygon']
        
        # 使用扫描线算法填充多边形
        # 简化版本：使用圆形近似
        cy, cx = int(spot['center'][0]), int(spot['center'][1])
        radius = int(spot['features']['mean_radius'])
        
        y_grid, x_grid = np.ogrid[:image_shape[0], :image_shape[1]]
        dist = np.sqrt((y_grid - cy)**2 + (x_grid - cx)**2)
        mask[dist <= radius] = True
        
        return mask


class InstanceSegmentationMetrics:
    """
    实例分割评估指标
    
    基于StarDist的评估方法
    """
    
    @staticmethod
    def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
        """计算两个掩码的IoU"""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    @staticmethod
    def match_instances(pred_spots: List[Dict], 
                       true_spots: List[Dict],
                       iou_threshold: float = 0.5) -> Dict:
        """
        匹配预测和真实实例
        
        Args:
            pred_spots: 预测的spot列表
            true_spots: 真实的spot列表
            iou_threshold: IoU阈值
            
        Returns:
            metrics: 匹配指标
        """
        n_pred = len(pred_spots)
        n_true = len(true_spots)
        
        if n_pred == 0 or n_true == 0:
            return {
                'tp': 0, 'fp': n_pred, 'fn': n_true,
                'precision': 0.0, 'recall': 0.0, 'f1': 0.0
            }
        
        # 计算IoU矩阵
        iou_matrix = np.zeros((n_pred, n_true))
        for i, pred in enumerate(pred_spots):
            for j, true in enumerate(true_spots):
                # 简化的距离匹配（实际应该用掩码IoU）
                dist = np.sqrt((pred['center'][0] - true['center'][0])**2 + 
                              (pred['center'][1] - true['center'][1])**2)
                max_dist = max(pred['features']['mean_radius'], 
                              true['features']['mean_radius'])
                
                # 将距离转换为IoU近似
                if dist < max_dist:
                    iou_matrix[i, j] = 1.0 - dist / max_dist
        
        # 匈牙利算法匹配
        from scipy.optimize import linear_sum_assignment
        
        # 只考虑IoU大于阈值的匹配
        valid_matches = iou_matrix >= iou_threshold
        
        if not valid_matches.any():
            return {
                'tp': 0, 'fp': n_pred, 'fn': n_true,
                'precision': 0.0, 'recall': 0.0, 'f1': 0.0
            }
        
        # 使用线性分配找到最佳匹配
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        
        tp = sum(1 for r, c in zip(row_ind, col_ind) if iou_matrix[r, c] >= iou_threshold)
        fp = n_pred - tp
        fn = n_true - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


# 便捷函数
def create_star_detector(n_rays: int = 32, **kwargs) -> StarConvexSpotDetector:
    """创建星形凸检测器的便捷函数"""
    return StarConvexSpotDetector(n_rays=n_rays, **kwargs)


def evaluate_segmentation(pred_spots: List[Dict], 
                         true_spots: List[Dict]) -> Dict:
    """评估分割结果的便捷函数"""
    metrics = InstanceSegmentationMetrics()
    return metrics.match_instances(pred_spots, true_spots)
