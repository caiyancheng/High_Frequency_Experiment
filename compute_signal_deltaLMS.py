"""
compute_lms.py
==============
从 gabor_render.py 的 get_color_matrices() 中提取数学逻辑，
计算：
  1. 背景 LMS（绝对值，cd/m²）
  2. 各颜色通道 (ach / rg / yv) 在 LMS 空间的 delta 方向（L2 归一化）

运行：
    python compute_lms.py
"""

import numpy as np


# ──────────────────────────────────────────────────────────────
# 0. 参数（与 gabor_render.py 保持完全一致）
# ──────────────────────────────────────────────────────────────

MEAN_LUMINANCE = 1.0   # cd/m²，按需修改

# 归一化 LMS 白点（代码中 get_color_matrices 里的 lms_gray）
lms_gray = np.array([0.739876529525622,
                     0.320136241543338,
                     0.020793708751515])

# D65 白点（XYZ，Y=1 归一化）
white_point_d65 = np.array([0.9505, 1.0000, 1.0888])

# XYZ → LMS（Hunt-Pointer-Estevez 类型，代码里的 M_xyz_lms）
M_xyz_lms = np.array([
    [ 0.187596268556126,  0.585168649077728, -0.026384263306304],
    [-0.133397430663221,  0.405505777260049,  0.034502127690364],
    [ 0.000244379021663, -0.000542995890619,  0.019406849066323]
])


# ──────────────────────────────────────────────────────────────
# 1. 构造 M_lms_dkl（LMS → DKL）及其逆（DKL → LMS）
#    公式来自 get_color_matrices()
# ──────────────────────────────────────────────────────────────

mc1 = lms_gray[0] / lms_gray[1]                     # L/M 比值
mc2 = (lms_gray[0] + lms_gray[1]) / lms_gray[2]     # (L+M)/S 比值

M_lms_dkl = np.array([
    [ 1,     1,    0   ],
    [ 1,    -mc1,  0   ],
    [-1,    -1,    mc2 ],
])

M_dkl_lms = np.linalg.inv(M_lms_dkl)   # DKL → LMS


# ──────────────────────────────────────────────────────────────
# 2. 背景 LMS
#    dkl_bg = mean_lum * (white_d65 @ M_xyz_lms.T) @ M_lms_dkl.T
#    lms_bg = M_dkl_lms @ dkl_bg  （等价于直接用 XYZ 路径）
# ──────────────────────────────────────────────────────────────

# D65 白点在 LMS 空间（归一化，Y=1 对应 1 cd/m²）
lms_white_normalized = white_point_d65 @ M_xyz_lms.T  # shape (3,)

# 绝对 LMS 背景（单位：cd/m²）
lms_bg = MEAN_LUMINANCE * lms_white_normalized

# 同样用 DKL 路径交叉验证
dkl_bg = MEAN_LUMINANCE * (lms_white_normalized @ M_lms_dkl.T)
lms_bg_check = M_dkl_lms @ dkl_bg   # 应与 lms_bg 完全相同


# ──────────────────────────────────────────────────────────────
# 3. 各颜色通道的 delta LMS 方向（L2 归一化）
#
#    Shader 公式：lin = M_dkl2rgb × (dkl_bg + mod_val × col_dir)
#    调制部分：   delta_dkl = mod_val × col_dir
#    对应 LMS：   delta_lms = M_dkl_lms × col_dir   （mod_val 是标量，不影响方向）
#    归一化：     delta_lms_hat = delta_lms / ||delta_lms||
# ──────────────────────────────────────────────────────────────

col_dirs_dkl = {
    'ach': np.array([1., 0., 0.]),   # 亮度轴
    'rg':  np.array([0., 1., 0.]),   # 红-绿轴
    'yv':  np.array([0., 0., 1.]),   # 黄-紫轴
}

delta_lms = {}
for name, col_dir in col_dirs_dkl.items():
    raw  = M_dkl_lms @ col_dir
    norm = np.linalg.norm(raw)
    delta_lms[name] = raw / norm


# ──────────────────────────────────────────────────────────────
# 4. 打印结果
# ──────────────────────────────────────────────────────────────

print("=" * 55)
print(f"  mean_luminance = {MEAN_LUMINANCE} cd/m²")
print("=" * 55)

print("\n[背景 LMS]  (单位: cd/m²)")
print(f"  L = {lms_bg[0]:.6f}")
print(f"  M = {lms_bg[1]:.6f}")
print(f"  S = {lms_bg[2]:.6f}")

print("\n[交叉验证: via dkl_bg → LMS，应与上方完全一致]")
print(f"  L = {lms_bg_check[0]:.6f}")
print(f"  M = {lms_bg_check[1]:.6f}")
print(f"  S = {lms_bg_check[2]:.6f}")

print("\n[各颜色通道 delta LMS 方向 (L2 归一化)]")
for name, d in delta_lms.items():
    print(f"  {name:3s}: ΔL={d[0]:+.6f}  ΔM={d[1]:+.6f}  ΔS={d[2]:+.6f}")

print("\n[参考值（论文/预期）]")
print("  ach: [+0.9182, +0.3953, +0.0260]")
print("  rg : [+0.7071, -0.7071,  0.0000]")
print("  yv : [ 0.0000,  0.0000, +1.0000]")

print("\n[辅助量]")
print(f"  mc1 = L/M         = {mc1:.8f}")
print(f"  mc2 = (L+M)/S     = {mc2:.8f}")
print(f"  dkl_bg = {dkl_bg}")
print(f"    D (luminance)   = {dkl_bg[0]:.6f}")
print(f"    K (R-G, ~0)     = {dkl_bg[1]:.6f}")
print(f"    L (Y-V, ~0)     = {dkl_bg[2]:.6f}")