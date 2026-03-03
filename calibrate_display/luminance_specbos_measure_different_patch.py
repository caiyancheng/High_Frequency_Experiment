import argparse
import numpy as np
import json
import os
import time
import glfw
from OpenGL.GL import *
from gfxdisp.specbos import specbos_measure


# ==========================================================
# Generate 8-bit Pixel Values
# ==========================================================

def generate_pixel_values(num_points=20,
                          include_black=True,
                          scale="log"):
    """
    Generate pixel values between 0–255.

    Parameters
    ----------
    num_points : int
        Number of sampling points.
    include_black : bool
        Whether to force include 0.
    scale : str
        "log" or "linear"
    """

    if scale not in ["log", "linear"]:
        raise ValueError("scale must be 'log' or 'linear'")

    if scale == "log":
        values = np.logspace(
            np.log10(1),
            np.log10(255),
            num_points
        )
    else:  # linear
        values = np.linspace(
            0,
            255,
            num_points
        )

    pixel_values = np.unique(np.round(values).astype(int))

    if include_black and 0 not in pixel_values:
        pixel_values = np.concatenate(([0], pixel_values))

    return np.sort(pixel_values)


# ==========================================================
# OpenGL Setup
# ==========================================================

def create_second_monitor_window():

    if not glfw.init():
        raise RuntimeError("GLFW init failed")

    monitors = glfw.get_monitors()
    if len(monitors) < 2:
        raise RuntimeError("Second monitor not detected")

    second_monitor = monitors[1]
    mode = glfw.get_video_mode(second_monitor)
    width, height = mode.size
    x_pos, y_pos = glfw.get_monitor_pos(second_monitor)

    glfw.window_hint(glfw.DECORATED, glfw.FALSE)
    glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
    glfw.window_hint(glfw.FLOATING, glfw.TRUE)

    window = glfw.create_window(width, height, "Gamma Measure", None, None)
    glfw.set_window_pos(window, x_pos, y_pos)
    glfw.make_context_current(window)

    glfw.swap_interval(1)
    glViewport(0, 0, width, height)

    return window, width, height


# ==========================================================
# Measure one patch size
# ==========================================================

def measure_patch_size(window, screen_w, screen_h,
                       square_size, pixel_list,
                       repeat, output_path):
    """
    Sweep through all pixel values for a given patch size,
    measure luminance, and save results to output_path.

    Parameters
    ----------
    window       : GLFW window handle
    screen_w/h   : screen resolution
    square_size  : side length of the gray patch in pixels
    pixel_list   : array of integer pixel values (0–255)
    repeat       : number of repeated measurements per patch
    output_path  : JSON file path for saving results
    """

    results = {}

    for idx, pixel in enumerate(pixel_list):

        normalized = pixel / 255.0

        print(f"\n  [Patch {square_size}px] Pixel {pixel} ({idx+1}/{len(pixel_list)})")

        # Clear full screen to black
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        # Compute centered patch origin
        x0 = int((screen_w - square_size) / 2)
        y0 = int((screen_h - square_size) / 2)

        # Draw gray patch using scissor
        glEnable(GL_SCISSOR_TEST)
        glScissor(x0, y0, square_size, square_size)

        glClearColor(normalized, normalized, normalized, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glDisable(GL_SCISSOR_TEST)
        glfw.swap_buffers(window)
        glfw.poll_events()

        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            print("  ESC pressed, aborting.")
            break

        time.sleep(0.7)

        Y_list = []
        x_list = []
        y_list = []

        while len(Y_list) < repeat:

            glfw.poll_events()

            try:
                Y, x, y = specbos_measure()
            except Exception:
                continue

            if Y is None:
                continue

            Y_list.append(Y)
            x_list.append(x)
            y_list.append(y)

            print(f"    {len(Y_list)}/{repeat} Y={Y:.4f}")

        results[f"P_{idx}"] = {
            "pixel_value": int(pixel),
            "normalized": float(normalized),
            "Y_mean": float(np.mean(Y_list)),
            "Y_list": Y_list,
            "x_list": x_list,
            "y_list": y_list
        }

        time.sleep(0.5)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n  Saved: {output_path}")
    return results


# ==========================================================
# Main
# ==========================================================

def main():

    parser = argparse.ArgumentParser(
        description="Measure luminance vs. pixel value for multiple patch sizes."
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="Measure_specbos",
        help="Directory to save JSON result files."
    )
    parser.add_argument(
        "--patch_sizes", type=int, nargs="+",
        default=[800, 1000],
        help="List of square patch sizes in pixels, e.g. --patch_sizes 200 400 600 800 1000"
    )
    parser.add_argument(
        "--points", type=int, default=20,
        help="Number of pixel value sampling points."
    )
    parser.add_argument(
        "--repeat", type=int, default=3,
        help="Number of repeated measurements per pixel value."
    )
    parser.add_argument(
        "--scale", type=str, default="linear", choices=["linear", "log"],
        help="Sampling scale for pixel values."
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pixel_list = generate_pixel_values(
        args.points, include_black=True, scale=args.scale
    )

    print(f"Patch sizes to measure : {args.patch_sizes}")
    print(f"Pixel values           : {pixel_list.tolist()}")
    print(f"Repeats per point      : {args.repeat}")
    print(f"Output directory       : {args.output_dir}\n")

    window, screen_w, screen_h = create_second_monitor_window()

    try:
        for square_size in args.patch_sizes:

            print(f"\n{'='*50}")
            print(f"  Starting patch size: {square_size} x {square_size} px")
            print(f"{'='*50}")

            output_path = os.path.join(
                args.output_dir,
                f"luminance_pixel_measure_patch{square_size}.json"
            )

            measure_patch_size(
                window=window,
                screen_w=screen_w,
                screen_h=screen_h,
                square_size=square_size,
                pixel_list=pixel_list,
                repeat=args.repeat,
                output_path=output_path,
            )

    finally:
        glfw.destroy_window(window)
        glfw.terminate()

    print("\nAll patch sizes measured. Done.")


if __name__ == "__main__":
    main()