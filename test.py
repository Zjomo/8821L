import numpy as np
from PIL import Image
from scipy.signal import fftconvolve

# 1. 硬件与目标参数
target_size = 256
iterations = 80
lut_2pi = 213  # 根据芯片出厂报告，532nm下的 2pi 对应灰度级为 213

# 2. 构建目标分布 (256x256)
target_I = np.zeros((target_size, target_size), dtype=np.float64)
cy, cx = target_size // 2, target_size // 2
spacing = 10  # 三点像素间距

# 强度梯度设置: 1.0 : 0.65 : 0.35
target_I[cy, cx - spacing] = 1.00
target_I[cy, cx]           = 0.65
target_I[cy, cx + spacing] = 0.35

# 3. 高斯压窄核 (sigma = 0.6，实现光斑缩细)
sigma = 0.6
x = np.arange(-3, 4)
y = np.arange(-3, 4)
xx, yy = np.meshgrid(x, y)
kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
kernel /= np.max(kernel)

# 二维卷积实现高斯压窄
target_I_narrowed = fftconvolve(target_I, kernel, mode='same')
target_I_narrowed = np.maximum(target_I_narrowed, 0)
target_A = np.sqrt(target_I_narrowed)

# 4. 加权 GS 算法计算 CGH 相位
np.random.seed(0)
phase_cgh = 2 * np.pi * np.random.rand(target_size, target_size)
A_in = np.ones((target_size, target_size), dtype=np.float64)

for _ in range(iterations):
    # SLM -> 傅里叶面
    field_slm = A_in * np.exp(1j * phase_cgh)
    field_fourier = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field_slm)))
    
    # 傅里叶面约束
    fourier_phase = np.angle(field_fourier)
    field_fourier_constrained = target_A * np.exp(1j * fourier_phase)
    
    # 傅里叶面 -> SLM 面
    field_slm_new = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(field_fourier_constrained)))
    phase_cgh = np.angle(field_slm_new)

# 5. 映射至 2pi 灰度级 (213)，保持 256x256 尺寸
phase_normalized = (phase_cgh % (2 * np.pi)) / (2 * np.pi)
cgh_gray_256 = np.round(phase_normalized * lut_2pi).astype(np.uint8)

# 6. 直接保存为 256x256 像素的 BMP 灰度图
img = Image.fromarray(cgh_gray_256)
img.save("CGH_3points_256x256.bmp")

print("已成功为你直接生成 256x256 像素灰度图：CGH_3points_256x256.bmp！")