import argparse
import numpy as np
import json
import os
import time
import threading
import glfw
from OpenGL.GL import *
from gfxdisp.specbos import specbos_measure


# ==========================================================
# 24 patches (Macbeth-like)
# ==========================================================

def generate_24_patch_rgb():
    patches = np.array([
        [115, 82, 68],[194,150,130],[98,122,157],[87,108,67],
        [133,128,177],[103,189,170],[214,126,44],[80,91,166],
        [193,90,99],[94,60,108],[157,188,64],[224,163,46],
        [56,61,150],[70,148,73],[175,54,60],[231,199,31],
        [187,86,149],[8,133,161],[243,243,242],[200,200,200],
        [160,160,160],[122,122,121],[85,85,85],[52,52,52]
    ])
    return patches / 255.0


# ==========================================================
# OpenGL
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

    window = glfw.create_window(width, height, "RGB XYZ Measure", None, None)
    glfw.set_window_pos(window, x_pos, y_pos)
    glfw.make_context_current(window)

    glfw.swap_interval(1)
    glViewport(0, 0, width, height)

    return window


# ==========================================================
# xyY → XYZ
# ==========================================================

def xyY_to_XYZ(x, y, Y):
    if y == 0:
        return np.array([0,0,0])
    X = (x/y)*Y
    Z = ((1-x-y)/y)*Y
    return np.array([X,Y,Z])


# ==========================================================
# Measurement Thread
# ==========================================================

def measure_patch(repeat, result):

    Y_list, x_list, y_list = [], [], []

    while len(Y_list) < repeat:
        try:
            Y, x, y = specbos_measure()
        except:
            continue

        if Y is None:
            continue

        Y_list.append(Y)
        x_list.append(x)
        y_list.append(y)

        print(f"  {len(Y_list)}/{repeat}  Y={Y:.3f}")

    result["XYZ"] = xyY_to_XYZ(
        np.mean(x_list),
        np.mean(y_list),
        np.mean(Y_list)
    )


# ==========================================================
# Main
# ==========================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--output", type=str,
        default="Measure_specbos/rgb_xyz_measure.json")

    args = parser.parse_args()

    os.makedirs("Measure_specbos", exist_ok=True)

    patches = generate_24_patch_rgb()
    window = create_second_monitor_window()

    measurements = []

    try:
        for idx, rgb in enumerate(patches):

            print(f"\nPatch {idx+1}/24  RGB={rgb}")

            glClearColor(rgb[0], rgb[1], rgb[2], 1.0)
            glClear(GL_COLOR_BUFFER_BIT)
            glfw.swap_buffers(window)
            glfw.poll_events()

            time.sleep(0.7)

            result = {}
            t = threading.Thread(
                target=measure_patch,
                args=(args.repeat, result)
            )
            t.start()
            t.join()

            measurements.append({
                "RGB": rgb.tolist(),
                "XYZ": result["XYZ"].tolist()
            })

            time.sleep(0.5)

    finally:
        glfw.destroy_window(window)
        glfw.terminate()

    with open(args.output, "w") as f:
        json.dump(measurements, f, indent=4)

    print("\nMeasurement finished.")
    print("Saved to:", args.output)


if __name__ == "__main__":
    main()