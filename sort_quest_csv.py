"""
sort_quest_csv.py
=================
对 Quest_results.csv 进行排序整理：
  1. 先按被试姓名 (name) 聚合
  2. 名字相同时，按以下字段依次排序：
       color → mean_luminance → spatial_frequency_cpp
       → speed_px_per_sec → trial_index

用法：
  python sort_quest_csv.py                          # 默认输入/输出 Quest_results.csv
  python sort_quest_csv.py -i raw.csv -o sorted.csv
"""

import argparse
import csv
import os
import sys

# ---------- 排序键定义 ----------
COLOR_ORDER = {"ach": 0, "rg": 1, "yv": 2}   # 颜色通道自定义顺序

def sort_key(row: dict):
    return (
        row["name"],                                         # 1. 被试姓名
        COLOR_ORDER.get(row["color"], 99),                   # 2. 颜色通道
        float(row["mean_luminance"]),                        # 3. 平均亮度
        float(row["spatial_frequency_cpp"]),                 # 4. 空间频率
        float(row["speed_px_per_sec"]),                      # 5. 速度
        int(row["trial_index"]),                             # 6. 试次序号
    )


def sort_quest_csv(input_path: str, output_path: str):
    if not os.path.exists(input_path):
        print(f"[错误] 找不到输入文件：{input_path}")
        sys.exit(1)

    with open(input_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not rows:
        print("[警告] CSV 文件为空，无需排序。")
        return

    sorted_rows = sorted(rows, key=sort_key)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted_rows)

    # ---------- 打印摘要 ----------
    names = {}
    for r in sorted_rows:
        names.setdefault(r["name"], 0)
        names[r["name"]] += 1

    print(f"✓ 排序完成：{len(sorted_rows)} 行 → {output_path}")
    print(f"  {'被试':<25} {'试次数':>6}")
    print(f"  {'-'*32}")
    for name, count in names.items():
        print(f"  {name:<25} {count:>6}")


def main():
    parser = argparse.ArgumentParser(description="Sort Quest_results.csv")
    parser.add_argument("-i", "--input",  default="Quest_results.csv",
                        help="输入 CSV 路径（默认: Quest_results.csv）")
    parser.add_argument("-o", "--output", default=None,
                        help="输出 CSV 路径（默认: 覆盖原文件）")
    args = parser.parse_args()

    output = args.output if args.output else args.input
    sort_quest_csv(args.input, output)


if __name__ == "__main__":
    main()