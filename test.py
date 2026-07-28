import numpy as np
from PIL import Image

# ==========================================
# 1. 硬件与相位校准参数
# ==========================================
slm_w, slm_h = 1272, 1024  # 滨松 SLM 物理像素
lut_2pi = 213             # 532nm 下 2pi 对应的灰度阶
iterations = 80           # GS 算法迭代次数

# ==========================================
# 2. 构造与 point-13_100x100 相同逻辑的目标阵列 (Target)
# ==========================================
# 创建 1272x1024 全黑画布
target_intensity = np.zeros((slm_h, slm_w), dtype=np.float64)

cy, cx = slm_h // 2, slm_w // 2
spot_spacing = 30  # 三点间距（像素），可根据视野大小调节

# 设置三个点的相对强度梯度 (1.0 : 0.7 : 0.4)
# 对应 3 个具有高斯包络的亮点
spot_sigma = 3.5  # 高斯包络腰束，控制每个光斑的饱满度
Y, X = np.ogrid[:slm_h, :slm_w]

centers_and_weights = [
    ((cy - spot_spacing, cx), 1.0),  # 顶点 (强度 1.0)
    ((cy, cx),                0.7),  # 中点 (强度 0.7)
    ((cy + spot_spacing, cx), 0.4)   # 底点 (强度 0.4)
]

for (spot_y, spot_x), weight in centers_and_weights:
    r_sq = (Y - spot_y)**2 + (X - spot_x)**2
    # 在目标面上直接写入高斯形貌的光斑
    target_intensity += weight * np.exp(-r_sq / (2 * spot_sigma**2))

target_amplitude = np.sqrt(target_intensity)

# ==========================================
# 3. 滨松官方同款 GS 相位检索算法
# ==========================================
np.random.seed(42)
# 初始随机相位 (0 ~ 2pi)
phase_cgh = 2 * np.pi * np.random.rand(slm_h, slm_w)
A_in = np.ones((slm_h, slm_w), dtype=np.float64)

print("正在运行滨松同款 GS 频域相位检索...")
for i in range(iterations):
    # 1. SLM 面 (频域) -> 焦平面 (空域)
    field_slm = A_in * np.exp(1j * phase_cgh)
    field_fourier = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field_slm)))
    
    # 2. 约束焦平面振幅，保留相位
    fourier_phase = np.angle(field_fourier)
    field_fourier_constrained = target_amplitude * np.exp(1j * fourier_phase)
    
    # 3. 焦平面 -> SLM 面
    field_slm_new = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(field_fourier_constrained)))
    
    # 4. 更新 SLM 相位
    phase_cgh = np.angle(field_slm_new)

print("计算完成！正在生成 BMP 相位图...")

# ==========================================
# 4. 映射至芯片 2pi 灰度并导出 BMP 图
# ==========================================
phase_normalized = (phase_cgh % (2 * np.pi)) / (2 * np.pi)
cgh_gray = np.round(phase_normalized * lut_2pi).astype(np.uint8)

output_filename = "Hamamatsu_Official_Logic_3Spots_1272x1024.bmp"
img = Image.fromarray(cgh_gray)
img.save(output_filename)

print(f"已成功导出：{output_filename}")