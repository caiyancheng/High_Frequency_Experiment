"""
plot_moa_results.py

绘制 MOA_results.csv 中的人类实验结果图：
  - 3 个子图（ach / rg / yv）并排排列
  - X 轴：时间频率 (Hz)，对数刻度
  - Y 轴：视网膜空间频率 (cpd)，对数刻度
  - 同一子图内用不同颜色区分亮度等级
  - 多名观测者取几何平均值
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── 用户设置 ──────────────────────────────────────────────────────────────


CSV_IN = "MOA_results.csv"

# 指定要包含的观测者（注释掉的名字会被排除）
OBSERVERS = [
    "YanchengCai",
    # "Rafal Mantiuk",
    # "YifanDing",
    # "Shuqi Lou",
    # "Yaru Liu",
    # "LinShen",
]

# 各亮度等级的颜色（按亮度升序，每行对应一个亮度）
LUM_COLORS = {
    "low":  (0.1, 0.4, 0.8),   # 低亮度 → 蓝色
    "high": (0.9, 0.3, 0.1),   # 高亮度 → 红色
}

# 颜色方向
COLOR_DIRS   = ["ach", "rg", "yv"]
COLOR_TITLES = ["Achromatic", "Red-Green", "Yellow-Violet"]

# ── 读取数据 ──────────────────────────────────────────────────────────────

df = pd.read_csv(CSV_IN)

# 筛选指定观测者
df = df[df["name"].isin(OBSERVERS)].copy()
if df.empty:
    raise ValueError("未找到指定观测者的数据，请检查 OBSERVERS 列表。")

# 唯一亮度等级，升序排列
lum_levels = sorted(df["mean_luminance"].unique())
n_lum = len(lum_levels)
lum_color_list = list(LUM_COLORS.values())[:n_lum]

print(f"观测者   : {', '.join(OBSERVERS)}")
print(f"亮度等级 : {lum_levels} cd/m²")
print(f"颜色方向 : {COLOR_DIRS}\n")

# ── 绘图 ──────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(14, 4.8),
                         facecolor="white", constrained_layout=True)

for cc, (col_dir, col_title, ax) in enumerate(
        zip(COLOR_DIRS, COLOR_TITLES, axes)):

    df_c = df[df["color"] == col_dir]

    legend_handles = []
    legend_labels  = []

    for ll, (lum, lcolor) in enumerate(zip(lum_levels, lum_color_list)):

        df_l = df_c[np.isclose(df_c["mean_luminance"], lum)]
        if df_l.empty:
            continue

        # 对每个时间频率点，取所有观测者 cpd 的几何平均
        tf_vals = sorted(df_l["temporal_frequency_hz"].unique())
        mean_cpd = []

        for tf in tf_vals:
            rows = df_l[np.isclose(df_l["temporal_frequency_hz"], tf)]
            cpd_vals = rows["retinal_spatial_frequency_cpd"].dropna().values
            if len(cpd_vals) == 0:
                mean_cpd.append(np.nan)
            else:
                mean_cpd.append(10 ** np.mean(np.log10(cpd_vals)))

        tf_arr  = np.array(tf_vals)
        cpd_arr = np.array(mean_cpd)

        # 只保留有效点
        valid = ~np.isnan(cpd_arr)
        if not valid.any():
            continue

        h, = ax.plot(tf_arr[valid], cpd_arr[valid], "-o",
                     color=lcolor, markerfacecolor=lcolor,
                     markersize=7, linewidth=1.8,
                     label=f"{lum:.0f} cd/m²")
        legend_handles.append(h)
        legend_labels.append(f"{lum:.0f} cd/m²")

    # 坐标轴格式
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_xlabel("Temporal frequency (Hz)", fontsize=11)
    if cc == 0:
        ax.set_ylabel("Retinal spatial frequency (cpd)", fontsize=11)
    ax.set_title(col_title, fontsize=12, fontweight="bold")

    if legend_handles:
        ax.legend(legend_handles, legend_labels,
                  loc="best", fontsize=9)

obs_str = ", ".join(OBSERVERS)
fig.suptitle(f"MOA Results — {obs_str}", fontsize=11, fontweight="bold")

out_path = "MOA_results_plot.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"图已保存到 {out_path}")
plt.show()