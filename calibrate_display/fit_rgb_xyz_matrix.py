import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator


# ==========================================================
# Load PCHIP Model (from 代码2)
# ==========================================================

with open("Measure_specbos/pixel_luminance_model_pchip_B100_C100_s800.json") as f:
    model = json.load(f)

fit_domain = model["fit_domain"]
p_samples = np.array(model["pixel_samples"])
Y_samples = np.array(model["luminance_samples"])

# 重建 p -> L 插值器
p2L_model = PchipInterpolator(p_samples, Y_samples)

def pixel_to_linear(rgb_normalized):
    """
    将 [0,1] 的 display RGB 转换到线性亮度域
    rgb_normalized: shape (N, 3), 值域 [0, 1]
    返回 shape (N, 3) 线性值
    """
    Y_fit = p2L_model(rgb_normalized)           # 插值到 fit 域
    if fit_domain == "log10":
        return 10 ** Y_fit                       # log10 → 线性
    else:
        return Y_fit                             # 已是线性


# ==========================================================
# Load Measurement Data
# ==========================================================

with open("Measure_specbos/rgb_xyz_measure_B100_C100_s800.json") as f:
    data = json.load(f)

RGB_display = np.array([d["RGB"] for d in data])   # display (gamma) 域, [0,1]
XYZ         = np.array([d["XYZ"] for d in data])


# ==========================================================
# Convert Display RGB → Linear RGB
# ==========================================================

RGB_linear = pixel_to_linear(RGB_display)   # shape (24, 3)

print("Display RGB (first 3):\n", RGB_display[:3])
print("Linear  RGB (first 3):\n", RGB_linear[:3])
print("Linear  XYZ (first 3):\n", XYZ[:3])


# ==========================================================
# Fit Matrix  (在线性域拟合)
# ==========================================================

M_rgb_xyz, _, _, _ = np.linalg.lstsq(RGB_linear, XYZ, rcond=None)
# M_xyz_rgb = np.linalg.inv(M_rgb_xyz)
M_xyz_rgb, _, _, _ = np.linalg.lstsq(XYZ, RGB_linear, rcond=None)

print("\nM_rgb_xyz (linear RGB → XYZ):\n", M_rgb_xyz)
print("\nM_xyz_rgb (XYZ → linear RGB):\n", M_xyz_rgb)

# 保存矩阵
matrices = {
    "fit_domain": "linear_rgb",
    "note": "RGB is linearized via PCHIP model before matrix fitting",
    "M_rgb_xyz": M_rgb_xyz.tolist(),
    "M_xyz_rgb": M_xyz_rgb.tolist()
}
with open("Measure_specbos/rgb_xyz_matrix_B100_C100_s800.json", "w") as f:
    json.dump(matrices, f, indent=4)
print("\nSaved rgb_xyz_matrix.json")


# ==========================================================
# Reconstruct RGB (linear → display)
# ==========================================================

# 用矩阵从 XYZ 反推线性 RGB
RGB_linear_reconstructed = XYZ @ M_xyz_rgb

# 线性 RGB → display RGB（用 L2p 插值器逆变换）
L2p_model = PchipInterpolator(Y_samples, p_samples)

def linear_to_pixel(rgb_linear):
    if fit_domain == "log10":
        Y_fit = np.log10(np.clip(rgb_linear, 1e-8, None))
    else:
        Y_fit = np.maximum(rgb_linear, 0)
    return np.clip(L2p_model(Y_fit), 0, 1)

RGB_display_reconstructed = linear_to_pixel(RGB_linear_reconstructed)


# ==========================================================
# Visualization
# ==========================================================

def show_patches(rgb_array, title):
    fig, axes = plt.subplots(4, 6, figsize=(6, 4))
    for i in range(24):
        r, c = i // 6, i % 6
        axes[r, c].imshow([[np.clip(rgb_array[i], 0, 1)]])
        axes[r, c].set_xticks([])
        axes[r, c].set_yticks([])
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

show_patches(RGB_display,               "Original Display RGB")
show_patches(RGB_display_reconstructed, "Reconstructed Display RGB (via Linear Domain)")

# ==========================================================
# Error Analysis
# ==========================================================

diff = RGB_display - RGB_display_reconstructed
print(f"\nReconstruction Error (display RGB):")
print(f"  Max  abs error: {np.abs(diff).max():.4f}")
print(f"  Mean abs error: {np.abs(diff).mean():.4f}")