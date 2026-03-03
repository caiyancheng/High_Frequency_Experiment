import argparse
import json
import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ==========================================================
# Load JSON results
# ==========================================================

def load_patch_results(data_dir):
    """
    Load all luminance_pixel_measure_patch*.json files from data_dir.

    Returns
    -------
    dict : { patch_size (int) -> {"pixel_values": [...], "Y_means": [...]} }
    """

    pattern = os.path.join(data_dir, "luminance_pixel_measure_patch*.json")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No matching JSON files found in: {data_dir}\n"
            f"Expected pattern: luminance_pixel_measure_patch<SIZE>.json"
        )

    data = {}

    for fpath in files:
        match = re.search(r"patch(\d+)\.json$", fpath)
        if not match:
            print(f"  Skipping unrecognised file: {fpath}")
            continue

        size = int(match.group(1))

        with open(fpath, "r") as f:
            raw = json.load(f)

        pixel_values = []
        Y_means = []

        for key in sorted(raw.keys(), key=lambda k: int(k.split("_")[1])):
            entry = raw[key]
            pixel_values.append(entry["pixel_value"])
            Y_means.append(entry["Y_mean"])

        data[size] = {
            "pixel_values": pixel_values,
            "Y_means": Y_means,
        }

        print(f"  Loaded patch {size:>5}px  ({len(pixel_values)} points)  [{fpath}]")

    return data


# ==========================================================
# Plot
# ==========================================================

def plot_luminance(data, output_path=None, log_scale=False):
    """
    Plot luminance (Y) vs pixel value for all patch sizes on one figure.
    """

    sizes = sorted(data.keys())
    n = len(sizes)

    cmap = cm.get_cmap("plasma", n)
    colors = [cmap(i) for i in range(n)]

    fig, ax = plt.subplots(figsize=(9, 6))

    for color, size in zip(colors, sizes):
        pv = np.array(data[size]["pixel_values"])
        ym = np.array(data[size]["Y_means"])

        ax.plot(
            pv, ym,
            marker="o",
            markersize=4,
            linewidth=1.5,
            color=color,
            label=f"{size} px",
        )

    ax.set_xlabel("Pixel Value (0–255)", fontsize=13)
    ax.set_ylabel("Luminance Y (cd/m²)", fontsize=13)
    ax.set_title("Luminance vs. Pixel Value — Multiple Patch Sizes", fontsize=14)

    ax.set_xlim(-5, 260)
    if log_scale:
        ax.set_yscale("log")

    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(title="Patch Size", fontsize=10, title_fontsize=10,
               loc="upper left", framealpha=0.8)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"\nFigure saved: {output_path}")
    else:
        plt.show()

    return fig


# ==========================================================
# Main
# ==========================================================

def main():

    parser = argparse.ArgumentParser(
        description="Plot luminance vs pixel value for multiple patch sizes."
    )
    parser.add_argument(
        "--data_dir", type=str,
        default="Measure_specbos",
        help="Directory containing luminance_pixel_measure_patch*.json files."
    )
    parser.add_argument(
        "--output", type=str,
        default=None,
        help="If given, save the figure to this path (e.g. plot.png). "
             "Otherwise the plot is shown interactively."
    )
    parser.add_argument(
        "--log_y", action="store_true",
        help="Use logarithmic scale on the Y (luminance) axis."
    )

    args = parser.parse_args()

    print(f"\nLoading data from: {args.data_dir}")
    data = load_patch_results(args.data_dir)

    print(f"\nPlotting {len(data)} patch size(s)...")
    plot_luminance(data, output_path=args.output, log_scale=args.log_y)


if __name__ == "__main__":
    main()