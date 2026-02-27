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
        # Avoid log(0), start from 1
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

    return window


# ==========================================================
# Main Measurement
# ==========================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str,
                        default="Measure_specbos/luminance_pixel_measure.json")
    parser.add_argument("--points", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=5)

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output)
                if os.path.dirname(args.output) else ".", exist_ok=True)

    pixel_list = generate_pixel_values(args.points, include_black=True, scale="linear")

    window = create_second_monitor_window()

    results = {}

    try:
        for idx, pixel in enumerate(pixel_list):

            normalized = pixel / 255.0

            print(f"\nDisplaying pixel value: {pixel}")

            glClearColor(normalized, normalized, normalized, 1.0)
            glClear(GL_COLOR_BUFFER_BIT)
            glfw.swap_buffers(window)
            glfw.poll_events()

            if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
                break

            time.sleep(0.7)

            Y_list = []
            x_list = []
            y_list = []

            while len(Y_list) < args.repeat:

                glfw.poll_events()

                try:
                    Y, x, y = specbos_measure()
                except:
                    continue

                if Y is None:
                    continue

                Y_list.append(Y)
                x_list.append(x)
                y_list.append(y)

                print(f"  {len(Y_list)}/{args.repeat} Y={Y:.4f}")

            results[f"P_{idx}"] = {
                "pixel_value": int(pixel),
                "normalized": float(normalized),
                "Y_mean": float(np.mean(Y_list)),
                "Y_list": Y_list,
                "x_list": x_list,
                "y_list": y_list
            }

            time.sleep(0.5)

    finally:
        glfw.destroy_window(window)
        glfw.terminate()

    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)

    print("\nMeasurement finished.")


if __name__ == "__main__":
    main()