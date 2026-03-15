"""
fix_moa_results.py
==================
交互式工具：修正 MOA_results.csv 中指定行的 distance_m，
并用与 gabor_render.py 完全一致的公式重新计算以下派生字段：

  - retinal_spatial_frequency_cpd   (rho_cpd)
  - temporal_frequency_hz           (omega)
  - theta_deg                       (屏幕水平视角，弧度→角度)
  - gabor_radius_px

用法
----
  python fix_moa_results.py                         # 默认文件 MOA_results.csv
  python fix_moa_results.py --csv /path/to/file.csv

操作流程
--------
  1. 显示全部数据（带行号）
  2. 输入要修改的行号（逗号分隔，或 all）
  3. 对每行依次输入新 distance_m（直接回车 = 保留原值）
  4. 显示修改预览（高亮变更）
  5. 输入 y 确认写回；原文件自动备份为 <n>.bak
"""

import argparse
import math
import os
import shutil
import sys

import pandas as pd


# ──────────────────────────────────────────────────────────────────────────
# 公式（与 gabor_render.py 逐行对应）
# ──────────────────────────────────────────────────────────────────────────

def compute_display_width(diagonal_inch: float, R_x: float, R_y: float) -> float:
    """屏幕物理宽度（米）。  gabor_render.compute_display_width"""
    D_m = diagonal_inch * 0.0254
    return D_m * (R_x / math.sqrt(R_x ** 2 + R_y ** 2))


def compute_spatiotemporal_frequency(R_x, W, d, f_p, v_p):
    """
    返回 (rho_cpd, omega_hz, theta_deg)。
    gabor_render.compute_spatiotemporal_frequency，theta 转为角度存储。
    """
    theta_rad = 2 * math.atan(W / (2 * d))          # 屏幕水平视角 (rad)
    rho_rad   = (R_x * f_p) / theta_rad              # cycles/rad
    rho_cpd   = rho_rad * (math.pi / 180)            # → cpd
    omega     = f_p * v_p                            # Hz
    theta_deg = math.degrees(theta_rad)
    return rho_cpd, omega, theta_deg


def visual_radius_deg_to_px(visual_radius_deg, d, W, R_x) -> float:
    """Gabor 半径（像素）。  gabor_render.visual_radius_deg_to_px"""
    phi_rad = math.radians(visual_radius_deg)
    R_phys  = d * math.tan(phi_rad)
    return (R_phys / W) * R_x


def recompute_row(row: pd.Series, new_distance_m: float) -> dict:
    """
    给定新 distance_m，返回需要更新的字段字典。
    其余字段（name, color, speed, contrast, lum, sf_cpp …）保持不变。
    """
    W = compute_display_width(
        float(row["diagonal_inch"]),
        float(row["resolution_x"]),
        float(row["resolution_y"]),
    )
    rho_cpd, omega, theta_deg = compute_spatiotemporal_frequency(
        R_x=float(row["resolution_x"]),
        W=W,
        d=new_distance_m,
        f_p=float(row["spatial_frequency_cpp"]),
        v_p=float(row["speed_px_per_sec"]),
    )
    radius_px = visual_radius_deg_to_px(
        visual_radius_deg=float(row["visual_radius_deg"]),
        d=new_distance_m,
        W=W,
        R_x=float(row["resolution_x"]),
    )
    return {
        "distance_m":                     round(new_distance_m, 4),
        "retinal_spatial_frequency_cpd":  round(rho_cpd, 4),
        "temporal_frequency_hz":          round(omega, 4),
        "theta_deg":                      round(theta_deg, 4),
        "gabor_radius_px":                round(radius_px, 2),
    }


# ──────────────────────────────────────────────────────────────────────────
# 终端辅助
# ──────────────────────────────────────────────────────────────────────────

CYAN   = "\033[96m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
RED    = "\033[91m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

CHANGED_COLS = [
    "distance_m",
    "retinal_spatial_frequency_cpd",
    "temporal_frequency_hz",
    "theta_deg",
    "gabor_radius_px",
]

SHOW_COLS = [
    "name", "color", "speed_px_per_sec", "mean_luminance",
    "spatial_frequency_cpp", "repeat_index",
    "distance_m", "retinal_spatial_frequency_cpd",
    "temporal_frequency_hz", "theta_deg", "gabor_radius_px",
]


def _fmt(val):
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def print_table(df: pd.DataFrame, highlight_rows=None):
    """打印 DataFrame，highlight_rows 为要高亮的原始索引列表。"""
    cols = [c for c in SHOW_COLS if c in df.columns]
    col_w = 20
    header = f"{'#':>4}  " + "  ".join(f"{c:>{col_w}}" for c in cols)
    print(BOLD + header + RESET)
    print("-" * len(header))
    for idx, row in df.iterrows():
        is_hl = highlight_rows is not None and idx in highlight_rows
        line  = f"{idx:>4}  " + "  ".join(f"{_fmt(row[c]):>{col_w}}" for c in cols)
        print((YELLOW + line + RESET) if is_hl else line)


def print_diff(old_row: pd.Series, new_vals: dict, row_idx: int):
    """并排显示修改前后的差异。"""
    print(f"\n  {BOLD}行 {row_idx}  差异预览{RESET}")
    print(f"  {'字段':<40}  {'原值':>12}  →  {'新值':>12}")
    print(f"  {'-'*40}  {'-'*12}     {'-'*12}")
    for col in CHANGED_COLS:
        old_v = _fmt(float(old_row[col]))
        new_v = _fmt(new_vals[col])
        if old_v != new_v:
            print(f"  {CYAN}{col:<40}{RESET}  {RED}{old_v:>12}{RESET}  →  {GREEN}{new_v:>12}{RESET}")
        else:
            print(f"  {col:<40}  {old_v:>12}     {new_v:>12}")


# ──────────────────────────────────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="交互式修正 MOA_results.csv 中的 distance_m 并重算派生字段"
    )
    parser.add_argument("--csv", default="MOA_results.csv",
                        help="CSV 文件路径（默认 MOA_results.csv）")
    args = parser.parse_args()

    csv_path = args.csv
    if not os.path.exists(csv_path):
        print(f"{RED}错误：找不到文件 {csv_path}{RESET}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    total = len(df)

    # ── 1. 显示全表 ──────────────────────────────────────────────────────
    print(f"\n{BOLD}=== MOA_results  共 {total} 行 ==={RESET}\n")
    print_table(df)

    # ── 2. 选择行号 ──────────────────────────────────────────────────────
    print(f"\n{BOLD}请输入要修改的行号{RESET}（逗号分隔，如 0,3,7；输入 all 修改全部）：")
    raw = input("  > ").strip()
    if not raw:
        print("未选择任何行，退出。")
        return

    if raw.lower() == "all":
        target_indices = list(df.index)
    else:
        try:
            target_indices = [int(x.strip()) for x in raw.split(",")]
        except ValueError:
            print(f"{RED}输入格式错误，请输入整数行号。{RESET}")
            sys.exit(1)

    invalid = [i for i in target_indices if i not in df.index]
    if invalid:
        print(f"{RED}以下行号不存在：{invalid}{RESET}")
        sys.exit(1)

    # ── 3. 逐行输入新 distance_m ─────────────────────────────────────────
    pending: dict = {}   # {row_idx: new_values_dict}

    print()
    for idx in target_indices:
        row = df.loc[idx]
        print(f"  行 {idx}  |  color={row['color']}  speed={row['speed_px_per_sec']}"
              f"  lum={row['mean_luminance']}  "
              f"当前 distance_m = {float(row['distance_m']):.4f} m")
        raw_d = input(f"    新 distance_m（直接回车保留原值）: ").strip()
        if not raw_d:
            print(f"    → 跳过（保留 {float(row['distance_m']):.4f} m）")
            continue
        try:
            new_d = float(raw_d)
        except ValueError:
            print(f"    {RED}无效输入，跳过此行。{RESET}")
            continue
        if new_d <= 0:
            print(f"    {RED}距离必须 > 0，跳过此行。{RESET}")
            continue

        new_vals = recompute_row(row, new_d)
        pending[idx] = new_vals

    if not pending:
        print("\n没有任何修改，退出。")
        return

    # ── 4. 预览差异 ──────────────────────────────────────────────────────
    print(f"\n{BOLD}=== 修改预览 ==={RESET}")
    for idx, new_vals in pending.items():
        print_diff(df.loc[idx], new_vals, idx)

    print(f"\n{BOLD}修改后全表（黄色行已更新）：{RESET}\n")
    df_preview = df.copy()
    for idx, new_vals in pending.items():
        for col, val in new_vals.items():
            df_preview.at[idx, col] = val
    print_table(df_preview, highlight_rows=list(pending.keys()))

    # ── 5. 确认写回 ──────────────────────────────────────────────────────
    print(f"\n{BOLD}确认写回？（y / n）{RESET}")
    confirm = input("  > ").strip().lower()
    if confirm != "y":
        print("已取消，原文件未修改。")
        return

    bak_path = csv_path + ".bak"
    shutil.copy2(csv_path, bak_path)
    print(f"  原文件已备份至 {bak_path}")

    for idx, new_vals in pending.items():
        for col, val in new_vals.items():
            df.at[idx, col] = val
    df.to_csv(csv_path, index=False)
    print(f"  {GREEN}✓ 已写回 {csv_path}（共修改 {len(pending)} 行）{RESET}\n")


if __name__ == "__main__":
    main()