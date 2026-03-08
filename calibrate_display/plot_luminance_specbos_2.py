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

def load_patch_results(data_dir, sizes=None, dark=False):
    """
    Load luminance_pixel_measure_patch*.json files from data_dir.

    Parameters
    ----------
    data_dir : str
        Directory containing JSON files.
    sizes : list of int, optional
        If given, only load files matching these patch sizes.
        If None, load all sizes found in the directory.
    dark : bool
        If True, match files with the '_dark' suffix
        (e.g. luminance_pixel_measure_patch400_dark.json).
        If False, match files without the suffix
        (e.g. luminance_pixel_measure_patch400.json).

    Returns
    -------
    dict : { patch_size (int) -> {"pixel_values": [...], "Y_means": [...]} }
    """

    suffix = "_dark" if dark else ""

    if sizes:
        # Build explicit file list from the requested sizes
        files = []
        for s in sizes:
            fname = f"luminance_pixel_measure_patch{s}{suffix}.json"
            fpath = os.path.join(data_dir, fname)
            if os.path.isfile(fpath):
                files.append(fpath)
            else:
                print(f"  WARNING: file not found — {fpath}")
        files = sorted(files)
    else:
        # Fall back to glob: pick up every matching file in the directory
        pattern = os.path.join(data_dir, f"luminance_pixel_measure_patch*{suffix}.json")
        files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No matching JSON files found in: {data_dir}\n"
            f"Expected pattern: luminance_pixel_measure_patch<SIZE>{suffix}.json"
        )

    # Regex that handles both flavours
    regex = re.compile(r"patch(\d+)(_dark)?\.json$")

    data = {}

    for fpath in files:
        match = regex.search(fpath)
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

        tag = " (dark)" if dark else ""
        print(f"  Loaded patch {size:>5}px{tag}  ({len(pixel_values)} points)  [{fpath}]")

    return data


# ==========================================================
# Plot
# ==========================================================

def plot_luminance(data, output_path=None, log_scale=False, dark=False):
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

    dark_tag = " — Dark" if dark else ""
    ax.set_title(f"Luminance vs. Pixel Value — Multiple Patch Sizes{dark_tag}", fontsize=14)

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
        "--sizes", type=int, nargs="+", default=[400, 600, 800, 1000],
        metavar="SIZE",
        help="Patch sizes to load, e.g. --sizes 400 600 800 1000. "
             "If omitted, all sizes found in data_dir are loaded."
    )
    parser.add_argument(
        "--dark", action="store_true", default=True,
        help="Match files with the '_dark' suffix "
             "(e.g. luminance_pixel_measure_patch400_dark.json)."
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

    sizes_info = str(args.sizes) if args.sizes else "all"
    dark_info  = " (dark)" if args.dark else ""
    print(f"\nLoading data from : {args.data_dir}")
    print(f"Patch sizes       : {sizes_info}{dark_info}")

    data = load_patch_results(args.data_dir, sizes=args.sizes, dark=args.dark)

    print(f"\nPlotting {len(data)} patch size(s)...")
    plot_luminance(data, output_path=args.output, log_scale=args.log_y, dark=args.dark)


if __name__ == "__main__":
    main()