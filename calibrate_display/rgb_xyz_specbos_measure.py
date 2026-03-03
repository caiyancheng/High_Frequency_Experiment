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
    ], dtype=np.float32)

    patches /= 255.0

    # 🔥 找到全局最大值
    max_val = np.max(patches)

    # 🔥 归一化到 0.65
    patches = patches / max_val * 0.65

    print(f"\nMax RGB after scaling = {np.max(patches):.4f}")

    return patches

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

    framebuffer_w, framebuffer_h = glfw.get_framebuffer_size(window)
    glViewport(0, 0, framebuffer_w, framebuffer_h)

    return window, framebuffer_w, framebuffer_h


# ==========================================================
# xyY → XYZ
# ==========================================================

def xyY_to_XYZ(x, y, Y):
    if y == 0:
        return np.array([0, 0, 0])
    X = (x / y) * Y
    Z = ((1 - x - y) / y) * Y
    return np.array([X, Y, Z])


# ==========================================================
# Sleep while keeping GLFW alive
# ==========================================================

def sleep_with_events(seconds):
    deadline = time.time() + seconds
    while time.time() < deadline:
        glfw.poll_events()
        time.sleep(0.05)


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

        xyz = xyY_to_XYZ(x, y, Y)
        print(f"  [{len(Y_list)}/{repeat}]  Y={Y:.4f}  x={x:.4f}  y={y:.4f}  "
              f"→ XYZ=({xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f})")

    mean_XYZ = xyY_to_XYZ(
        np.mean(x_list),
        np.mean(y_list),
        np.mean(Y_list)
    )
    result["XYZ"] = mean_XYZ
    print(f"  ★ Mean XYZ = ({mean_XYZ[0]:.4f}, {mean_XYZ[1]:.4f}, {mean_XYZ[2]:.4f})")


# ==========================================================
# Main
# ==========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--output", type=str,
                        default="Measure_specbos/rgb_xyz_measure_B100_C100_s800.json")

    args = parser.parse_args()

    os.makedirs("Measure_specbos", exist_ok=True)

    patches = generate_24_patch_rgb()
    window, screen_w, screen_h = create_second_monitor_window()

    measurements = []

    try:
        for idx, rgb in enumerate(patches):
            print(f"\n{'='*50}")
            print(f"Patch {idx+1}/24  RGB=({rgb[0]:.4f}, {rgb[1]:.4f}, {rgb[2]:.4f})")
            print(f"{'='*50}")

            # 1️⃣ 全屏清黑
            glClearColor(0.0, 0.0, 0.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT)

            # 2️⃣ 计算中心 800x800
            square_size = min(800, screen_w, screen_h)

            x0 = int((screen_w - square_size) / 2)
            y0 = int((screen_h - square_size) / 2)

            # 3️⃣ 限定绘制区域
            glEnable(GL_SCISSOR_TEST)
            glScissor(x0, y0, square_size, square_size)

            # 4️⃣ 填充颜色
            glClearColor(rgb[0], rgb[1], rgb[2], 1.0)
            glClear(GL_COLOR_BUFFER_BIT)

            glDisable(GL_SCISSOR_TEST)
            glfw.swap_buffers(window)
            glfw.poll_events()

            # 等待显示稳定（保持事件循环）
            sleep_with_events(0.7)

            result = {}
            t = threading.Thread(
                target=measure_patch,
                args=(args.repeat, result)
            )
            t.start()

            # 主线程持续处理事件，直到测量完成
            while t.is_alive():
                glfw.poll_events()
                time.sleep(0.05)

            t.join()

            measurements.append({
                "RGB": rgb.tolist(),
                "XYZ": result["XYZ"].tolist()
            })

            # 测量间隔（保持事件循环）
            sleep_with_events(0.5)

    finally:
        glfw.destroy_window(window)
        glfw.terminate()

    with open(args.output, "w") as f:
        json.dump(measurements, f, indent=4)

    print("\n" + "="*50)
    print("Measurement finished.")
    print("Saved to:", args.output)

    # 打印汇总表格
    print("\n{:<6} {:>20} {:>40}".format("Patch", "RGB", "XYZ"))
    print("-" * 70)
    for i, m in enumerate(measurements):
        r, g, b = m["RGB"]
        x, y, z = m["XYZ"]
        print(f"  {i+1:<4} RGB=({r:.3f},{g:.3f},{b:.3f})   "
              f"XYZ=({x:.4f}, {y:.4f}, {z:.4f})")


if __name__ == "__main__":
    main()