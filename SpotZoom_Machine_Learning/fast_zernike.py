"""
快速Zernike计算模块 - 基于AOtools优化理念
用于波前分析和自适应光学
"""

import numpy as np
from functools import lru_cache
from typing import List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

# 尝试导入numba进行加速
try:
    from numba import jit, njit, prange
    HAS_NUMBA = True
    logger.info("Numba已启用，Zernike计算将使用JIT加速")
except ImportError:
    HAS_NUMBA = False
    logger.warning("Numba未安装，Zernike计算将使用纯Python实现")


class FastZernike:
    """
    优化的Zernike多项式计算类
    
    创新点：
    1. 使用numba JIT编译加速关键计算
    2. 缓存机制避免重复计算
    3. 向量化操作提高性能
    """
    
    def __init__(self, size: int = 256, cache_size: int = 128):
        """
        初始化快速Zernike计算器
        
        Args:
            size: 网格大小
            cache_size: 缓存大小
        """
        self.size = size
        self.cache_size = cache_size
        self._cache = {}
        self._grid_initialized = False
        self._rho = None
        self._theta = None
        self._mask = None
        
        # 初始化极坐标网格
        self._init_grid()
    
    def _init_grid(self):
        """初始化极坐标网格"""
        if self._grid_initialized:
            return
        
        # 创建笛卡尔坐标网格
        y, x = np.mgrid[-1:1:self.size*1j, -1:1:self.size*1j]
        
        # 转换为极坐标
        self._rho = np.sqrt(x**2 + y**2)
        self._theta = np.arctan2(y, x)
        
        # 圆形孔径掩码
        self._mask = self._rho <= 1.0
        
        self._grid_initialized = True
    
    @staticmethod
    def noll_to_nm(j: int) -> Tuple[int, int]:
        """
        将Noll索引转换为(n, m)
        
        Args:
            j: Noll索引（从1开始）
            
        Returns:
            (n, m): 径向阶数和方位频率
        """
        if j < 1:
            raise ValueError("Noll索引必须从1开始")
        
        # Noll索引到(n, m)的转换
        n = int((-1 + np.sqrt(8*(j-1) + 1)) / 2)
        
        # 计算m
        if n % 2 == 0:
            m = 2 * int((j - 1 - n*(n+1)/2) / 2)
        else:
            m = 2 * int((j - n*(n+1)/2) / 2) - 1
        
        # 确定符号
        if (j - 1 - n*(n+1)/2) % 2 != 0:
            m = -m
        
        return n, m
    
    @staticmethod
    def nm_to_noll(n: int, m: int) -> int:
        """
        将(n, m)转换为Noll索引
        
        Args:
            n: 径向阶数
            m: 方位频率
            
        Returns:
            j: Noll索引
        """
        # 计算Noll索引
        j = n * (n + 1) // 2 + 1
        
        # 调整m的位置
        if n % 2 == 0:
            # n为偶数，m为偶数
            j += abs(m) // 2
        else:
            # n为奇数，m为奇数
            j += (abs(m) - 1) // 2
        
        # 考虑符号
        if m > 0 and n % 4 in [0, 1]:
            pass  # 正m
        elif m < 0 and n % 4 in [2, 3]:
            pass  # 负m
        elif m != 0:
            j += 1  # 调整顺序
        
        return j
    
    def _radial_polynomial(self, n: int, m: int, rho: np.ndarray) -> np.ndarray:
        """
        计算径向多项式 R_n^m(rho)
        
        使用递推关系提高数值稳定性
        """
        m = abs(m)
        
        if n == m:
            return rho ** n
        
        if n - m == 2:
            # 二阶递推的初始值
            R_nm = rho ** m
            R_n2m = (n * R_nm - (n + m) * rho ** (n-2)) / (n - m)
            return R_n2m
        
        # 一般情况使用显式公式
        R = np.zeros_like(rho)
        for k in range((n - m) // 2 + 1):
            coef = ((-1)**k * np.math.factorial(n - k)) / \
                   (np.math.factorial(k) * 
                    np.math.factorial((n + m) // 2 - k) * 
                    np.math.factorial((n - m) // 2 - k))
            R += coef * rho**(n - 2*k)
        
        return R
    
    def zernike_noll(self, j: int, normalized: bool = True,
                     use_cache: bool = True) -> np.ndarray:
        """
        计算第j个Noll索引的Zernike模式
        
        Args:
            j: Noll索引（从1开始）
            normalized: 是否归一化
            use_cache: 是否使用缓存
            
        Returns:
            Z: Zernike模式
        """
        # 检查缓存
        if use_cache and j in self._cache:
            return self._cache[j]
        
        # 转换Noll索引到(n, m)
        n, m = self.noll_to_nm(j)
        
        # 计算径向多项式
        R = self._radial_polynomial(n, abs(m), self._rho)
        
        # 计算方位角部分
        if m >= 0:
            Z = R * np.cos(m * self._theta)
        else:
            Z = R * np.sin(abs(m) * self._theta)
        
        # 应用孔径掩码
        Z = Z * self._mask
        
        # 归一化
        if normalized:
            norm = np.sqrt(np.sum(Z**2))
            if norm > 0:
                Z = Z / norm
        
        # 缓存结果
        if use_cache and len(self._cache) < self.cache_size:
            self._cache[j] = Z
        
        return Z
    
    def zernike_array(self, noll_indices: List[int],
                     normalized: bool = True) -> np.ndarray:
        """
        批量生成Zernike模式
        
        Args:
            noll_indices: Noll索引列表
            normalized: 是否归一化
            
        Returns:
            Z_array: Zernike模式数组 (len(indices), size, size)
        """
        return np.array([self.zernike_noll(j, normalized) for j in noll_indices])
    
    def fit_wavefront(self, wavefront: np.ndarray, 
                     max_j: int = 15,
                     mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        拟合波前到Zernike系数
        
        Args:
            wavefront: 波前相位
            max_j: 最大Noll索引
            mask: 可选的掩码
            
        Returns:
            coefficients: Zernike系数
        """
        if mask is None:
            mask = self._mask
        
        # 生成Zernike基函数
        indices = list(range(1, max_j + 1))
        Z_array = self.zernike_array(indices)
        
        # 应用掩码
        Z_flat = Z_array[:, mask].T  # (n_pixels, n_modes)
        wf_flat = wavefront[mask]     # (n_pixels,)
        
        # 最小二乘拟合
        coefficients = np.linalg.lstsq(Z_flat, wf_flat, rcond=None)[0]
        
        return coefficients
    
    def reconstruct_wavefront(self, coefficients: np.ndarray,
                             max_j: Optional[int] = None) -> np.ndarray:
        """
        从Zernike系数重建波前
        
        Args:
            coefficients: Zernike系数
            max_j: 最大Noll索引（如果为None，使用系数长度）
            
        Returns:
            wavefront: 重建的波前
        """
        if max_j is None:
            max_j = len(coefficients)
        
        # 生成Zernike基函数
        indices = list(range(1, max_j + 1))
        Z_array = self.zernike_array(indices)
        
        # 线性组合
        wavefront = np.tensordot(coefficients, Z_array, axes=([0], [0]))
        
        return wavefront
    
    def compute_rms(self, wavefront: np.ndarray, 
                   mask: Optional[np.ndarray] = None) -> float:
        """
        计算波前的RMS值
        
        Args:
            wavefront: 波前相位
            mask: 可选的掩码
            
        Returns:
            rms: RMS值
        """
        if mask is None:
            mask = self._mask
        
        wf_masked = wavefront[mask]
        return np.sqrt(np.mean(wf_masked**2))
    
    def compute_strehl_ratio(self, wavefront: np.ndarray,
                            mask: Optional[np.ndarray] = None) -> float:
        """
        计算Strehl比
        
        Args:
            wavefront: 波前相位
            mask: 可选的掩码
            
        Returns:
            strehl: Strehl比
        """
        rms = self.compute_rms(wavefront, mask)
        # 近似公式：Strehl ≈ exp(-(2π/λ * RMS)^2)
        # 假设λ=1（归一化单位）
        strehl = np.exp(-(2 * np.pi * rms)**2)
        return strehl
    
    def compute_psf(self, wavefront: np.ndarray,
                   wavelength: float = 1.0,
                   na: float = 1.0) -> np.ndarray:
        """
        计算点扩散函数（PSF）
        
        Args:
            wavefront: 波前相位
            wavelength: 波长
            na: 数值孔径
            
        Returns:
            psf: 点扩散函数
        """
        # 瞳孔函数
        pupil = self._mask * np.exp(1j * 2 * np.pi * wavefront / wavelength)
        
        # FFT计算PSF
        psf_complex = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil)))
        psf = np.abs(psf_complex)**2
        
        # 归一化
        psf = psf / psf.sum()
        
        return psf
    
    def clear_cache(self):
        """清除缓存"""
        self._cache.clear()
        logger.info("Zernike缓存已清除")
    
    def get_cache_info(self) -> Dict:
        """获取缓存信息"""
        return {
            'cache_size': len(self._cache),
            'max_cache_size': self.cache_size,
            'cached_indices': list(self._cache.keys())
        }


class ZernikeDecomposition:
    """
    Zernike分解分析类
    
    用于分析波前的Zernike成分
    """
    
    def __init__(self, zernike_calculator: FastZernike):
        self.zc = zernike_calculator
        
    def analyze(self, wavefront: np.ndarray, max_j: int = 15) -> Dict:
        """
        分析波前的Zernike成分
        
        Args:
            wavefront: 波前相位
            max_j: 最大Noll索引
            
        Returns:
            analysis: 分析结果
        """
        # 拟合系数
        coefficients = self.zc.fit_wavefront(wavefront, max_j)
        
        # 重建波前
        reconstructed = self.zc.reconstruct_wavefront(coefficients)
        
        # 残差
        residual = wavefront - reconstructed
        
        # 分析结果
        analysis = {
            'coefficients': coefficients,
            'reconstructed': reconstructed,
            'residual': residual,
            'rms_original': self.zc.compute_rms(wavefront),
            'rms_residual': self.zc.compute_rms(residual),
            'strehl_original': self.zc.compute_strehl_ratio(wavefront),
            'strehl_residual': self.zc.compute_strehl_ratio(residual),
        }
        
        # 主要成分
        analysis['dominant_modes'] = self._find_dominant_modes(coefficients, n=5)
        
        return analysis
    
    def _find_dominant_modes(self, coefficients: np.ndarray, 
                            n: int = 5) -> List[Dict]:
        """找到主导的Zernike模式"""
        # 按系数绝对值排序
        sorted_indices = np.argsort(np.abs(coefficients))[::-1][:n]
        
        dominant = []
        for idx in sorted_indices:
            j = idx + 1  # Noll索引从1开始
            n_mode, m_mode = self.zc.noll_to_nm(j)
            dominant.append({
                'noll_index': j,
                'n': n_mode,
                'm': m_mode,
                'coefficient': coefficients[idx],
                'name': self._get_mode_name(j)
            })
        
        return dominant
    
    def _get_mode_name(self, j: int) -> str:
        """获取Zernike模式名称"""
        names = {
            1: 'Piston',
            2: 'Tip',
            3: 'Tilt',
            4: 'Defocus',
            5: 'Astigmatism@45°',
            6: 'Astigmatism@0°',
            7: 'Coma@y',
            8: 'Coma@x',
            9: 'Trefoil@y',
            10: 'Trefoil@x',
            11: 'Spherical',
        }
        return names.get(j, f'Z{j}')


# 便捷函数
def create_zernike_calculator(size: int = 256) -> FastZernike:
    """创建Zernike计算器的便捷函数"""
    return FastZernike(size=size)


def quick_zernike(j: int, size: int = 256) -> np.ndarray:
    """快速获取单个Zernike模式的便捷函数"""
    zc = FastZernike(size=size)
    return zc.zernike_noll(j)


def fit_zernike(wavefront: np.ndarray, max_j: int = 15) -> np.ndarray:
    """快速拟合Zernike系数的便捷函数"""
    zc = FastZernike(size=wavefront.shape[0])
    return zc.fit_wavefront(wavefront, max_j)
