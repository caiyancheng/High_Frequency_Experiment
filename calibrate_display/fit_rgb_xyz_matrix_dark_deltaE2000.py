import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator


# ==========================================================
# Load PCHIP Model (from 代码2)
# ==========================================================

with open("Measure_specbos/pixel_luminance_model_pchip_B100_C100_s800_dark.json") as f:
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

with open("Measure_specbos/rgb_xyz_measure_B100_C100_s800_dark.json") as f:
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
with open("Measure_specbos/rgb_xyz_matrix_B100_C100_s800_dark.json", "w") as f:
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

# 用矩阵从线性 RGB 正向推算 XYZ（用于 deltaE 计算）
XYZ_reconstructed = RGB_linear @ M_rgb_xyz


# ==========================================================
# ΔE 2000 计算模块
# ==========================================================

def xyz_to_lab(XYZ, XYZ_n=None):
    """
    将 XYZ 转换为 CIELAB (D65 白点)
    XYZ: shape (N, 3)
    XYZ_n: 白点 XYZ，默认 D65
    返回 shape (N, 3) 的 Lab
    """
    if XYZ_n is None:
        # CIE D65 标准白点
        XYZ_n = np.array([95.047, 100.000, 108.883])

    xyz_ratio = XYZ / XYZ_n

    def f(t):
        delta = 6.0 / 29.0
        return np.where(
            t > delta ** 3,
            np.cbrt(t),
            t / (3 * delta ** 2) + 4.0 / 29.0
        )

    fx = f(xyz_ratio[:, 0])
    fy = f(xyz_ratio[:, 1])
    fz = f(xyz_ratio[:, 2])

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    return np.stack([L, a, b], axis=1)


def delta_e_2000(Lab1, Lab2):
    """
    计算两组 Lab 色彩的 ΔE 2000
    Lab1, Lab2: shape (N, 3), 列顺序 [L, a, b]
    返回 shape (N,) 的 ΔE00 值
    参考: Sharma et al. (2005), "The CIEDE2000 Color-Difference Formula"
    """
    L1, a1, b1 = Lab1[:, 0], Lab1[:, 1], Lab1[:, 2]
    L2, a2, b2 = Lab2[:, 0], Lab2[:, 1], Lab2[:, 2]

    # Step 1: C*ab 和 h*ab
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    C_avg = (C1 + C2) / 2.0

    C_avg7 = C_avg**7
    G = 0.5 * (1 - np.sqrt(C_avg7 / (C_avg7 + 25**7)))

    a1p = a1 * (1 + G)
    a2p = a2 * (1 + G)

    C1p = np.sqrt(a1p**2 + b1**2)
    C2p = np.sqrt(a2p**2 + b2**2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    # Step 2: 差值 ΔL', ΔC', ΔH'
    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = np.where(
        np.abs(h2p - h1p) <= 180,
        h2p - h1p,
        np.where(h2p - h1p > 180, h2p - h1p - 360, h2p - h1p + 360)
    )
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2))

    # Step 3: 平均值
    Lp_avg = (L1 + L2) / 2.0
    Cp_avg = (C1p + C2p) / 2.0

    hp_avg = np.where(
        np.abs(h1p - h2p) <= 180,
        (h1p + h2p) / 2.0,
        np.where(
            h1p + h2p < 360,
            (h1p + h2p + 360) / 2.0,
            (h1p + h2p - 360) / 2.0
        )
    )

    # Step 4: 权重函数
    T = (1
         - 0.17 * np.cos(np.radians(hp_avg - 30))
         + 0.24 * np.cos(np.radians(2 * hp_avg))
         + 0.32 * np.cos(np.radians(3 * hp_avg + 6))
         - 0.20 * np.cos(np.radians(4 * hp_avg - 63)))

    SL = 1 + 0.015 * (Lp_avg - 50)**2 / np.sqrt(20 + (Lp_avg - 50)**2)
    SC = 1 + 0.045 * Cp_avg
    SH = 1 + 0.015 * Cp_avg * T

    Cp_avg7 = Cp_avg**7
    RC = 2 * np.sqrt(Cp_avg7 / (Cp_avg7 + 25**7))
    d_theta = 30 * np.exp(-((hp_avg - 275) / 25)**2)
    RT = -np.sin(np.radians(2 * d_theta)) * RC

    # Step 5: 最终 ΔE00
    dE = np.sqrt(
        (dLp / SL)**2 +
        (dCp / SC)**2 +
        (dHp / SH)**2 +
        RT * (dCp / SC) * (dHp / SH)
    )
    return dE


# ==========================================================
# 计算 ΔE 2000（真实 XYZ vs 重建 XYZ）
# ==========================================================

# 自动估算白点：取所有测量中亮度最高的点作为参考白点
XYZ_n = XYZ[np.argmax(XYZ[:, 1])]
print(f"\nWhite point estimate XYZ_n = {XYZ_n}")

Lab_measured     = xyz_to_lab(XYZ,               XYZ_n=XYZ_n)
Lab_reconstructed = xyz_to_lab(XYZ_reconstructed, XYZ_n=XYZ_n)

dE2000 = delta_e_2000(Lab_measured, Lab_reconstructed)

print(f"\n===== ΔE 2000 Results (Measured XYZ vs Reconstructed XYZ) =====")
print(f"  Num samples:      {len(dE2000)}")
print(f"  Max  ΔE00:        {dE2000.max():.4f}")
print(f"  Mean ΔE00:        {dE2000.mean():.4f}")
print(f"  Median ΔE00:      {np.median(dE2000):.4f}")
print(f"  ΔE00 < 1 ratio:   {(dE2000 < 1).mean() * 100:.1f}%")
print(f"  ΔE00 < 2 ratio:   {(dE2000 < 2).mean() * 100:.1f}%")
print(f"  ΔE00 < 3 ratio:   {(dE2000 < 3).mean() * 100:.1f}%")
print(f"\n  Per-sample ΔE00:")
for i, de in enumerate(dE2000):
    print(f"    Sample {i+1:2d}: ΔE00 = {de:.4f}")


# ==========================================================
# Visualization — 4 行 × 12 列（每列对为 GT | Pred），标注 ΔE00
# 布局：每行 6 对色块，每对占 2 列（左=GT，右=Pred）
#        对与对之间插入细分隔，行与行之间留 ΔE 数字空间
# ==========================================================

n_cols_per_row = 6          # 每行多少对
n_rows         = 4          # 行数（共 24 色块 → 4×6）
fig_w, fig_h   = 14, 8

fig = plt.figure(figsize=(fig_w, fig_h))
fig.patch.set_facecolor('#1a1a1a')

# 整体标题
fig.suptitle(
    "Color Patch Comparison: GT vs Pred   |   ΔE 2000 annotated above each pair",
    color='white', fontsize=12, fontweight='bold', y=0.98
)

# GridSpec：4 大行，每大行含 2 子行（上=ΔE标注占位，下=色块）
# 12 列（每对 2 列），列间距略大于对内间距
from matplotlib.gridspec import GridSpec

gs = GridSpec(
    nrows=8,          # 4 大行 × 2（标注行 + 色块行）
    ncols=12,
    figure=fig,
    hspace=0.08,      # 子行间距
    wspace=0.06,      # 列间距
    left=0.02, right=0.98,
    top=0.93, bottom=0.02
)

for patch_idx in range(24):
    row_block = patch_idx // n_cols_per_row   # 0-3 大行
    col_pair  = patch_idx  % n_cols_per_row   # 0-5 对

    gs_row_label = row_block * 2       # 标注子行
    gs_row_patch = row_block * 2 + 1   # 色块子行
    gs_col_gt    = col_pair  * 2       # GT 列
    gs_col_pred  = col_pair  * 2 + 1   # Pred 列

    gt_color   = np.clip(RGB_display[patch_idx],               0, 1)
    pred_color = np.clip(RGB_display_reconstructed[patch_idx], 0, 1)
    de_val     = dE2000[patch_idx]

    # ---- 标注行：跨 GT+Pred 两列，写 ΔE 数字 ----
    ax_label = fig.add_subplot(gs[gs_row_label, gs_col_gt:gs_col_pred + 1])
    ax_label.set_facecolor('#1a1a1a')
    ax_label.set_xlim(0, 1)
    ax_label.set_ylim(0, 1)
    ax_label.axis('off')

    # 颜色按阈值区分
    de_color = '#55dd55' if de_val < 1 else '#ffaa33' if de_val < 2 else '#ff4444'
    ax_label.text(
        0.5, 0.15,
        f"ΔE={de_val:.2f}",
        ha='center', va='bottom',
        color=de_color, fontsize=7.5, fontweight='bold',
        transform=ax_label.transAxes
    )

    # ---- 色块行：GT ----
    ax_gt = fig.add_subplot(gs[gs_row_patch, gs_col_gt])
    ax_gt.imshow([[gt_color]], aspect='auto')
    ax_gt.set_xticks([])
    ax_gt.set_yticks([])
    # 细边框标示 GT
    for spine in ax_gt.spines.values():
        spine.set_edgecolor('#888888')
        spine.set_linewidth(0.8)
    # 左下角小标
    ax_gt.text(0.04, 0.06, 'GT', transform=ax_gt.transAxes,
               color='white', fontsize=5.5, alpha=0.75,
               ha='left', va='bottom',
               bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.35, lw=0))

    # ---- 色块行：Pred ----
    ax_pred = fig.add_subplot(gs[gs_row_patch, gs_col_pred])
    ax_pred.imshow([[pred_color]], aspect='auto')
    ax_pred.set_xticks([])
    ax_pred.set_yticks([])
    for spine in ax_pred.spines.values():
        spine.set_edgecolor(de_color)
        spine.set_linewidth(1.2)
    ax_pred.text(0.04, 0.06, 'Pred', transform=ax_pred.transAxes,
                 color='white', fontsize=5.5, alpha=0.75,
                 ha='left', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.35, lw=0))

# 图例说明
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#55dd55', label='ΔE < 1  (Excellent)'),
    Patch(facecolor='#ffaa33', label='ΔE 1–2  (Good)'),
    Patch(facecolor='#ff4444', label='ΔE ≥ 2  (Attention)'),
]
fig.legend(
    handles=legend_elements,
    loc='lower center', ncol=3,
    fontsize=8, framealpha=0.3,
    labelcolor='white',
    facecolor='#333333',
    edgecolor='#555555',
    bbox_to_anchor=(0.5, -0.01)
)

plt.savefig("color_patch_comparison_deltaE.png", dpi=150,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()


# ==========================================================
# Error Analysis (原有)
# ==========================================================

diff = RGB_display - RGB_display_reconstructed
print(f"\nReconstruction Error (display RGB):")
print(f"  Max  abs error: {np.abs(diff).max():.4f}")
print(f"  Mean abs error: {np.abs(diff).mean():.4f}")