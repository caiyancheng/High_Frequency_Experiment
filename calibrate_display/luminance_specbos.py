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
# Utility
# ==========================================================

def remove_outliers(data_list, threshold=2):
    data_array = np.array(data_list)
    mean = np.mean(data_array)
    std = np.std(data_array)
    filtered = [val for val in data_list if abs(val - mean) <= threshold * std]
    return filtered if len(filtered) > 0 else data_list


def generate_log_luminance_points(num_points=10):
    """
    Generate log scale values between 0 and 1 (avoid 0).
    """
    values = np.logspace(-3, 0, num_points)  # 0.001 → 1
    return values


# ==========================================================
# Display Thread
# ==========================================================

class DisplayThread(threading.Thread):
    def __init__(self, luminance_list, event_start_measure, event_measure_done):
        super().__init__()
        self.luminance_list = luminance_list
        self.event_start_measure = event_start_measure
        self.event_measure_done = event_measure_done
        self.current_index = -1

    def run(self):
        if not glfw.init():
            raise RuntimeError("GLFW init failed")

        monitors = glfw.get_monitors()
        if len(monitors) < 2:
            raise RuntimeError("Second monitor not detected")

        second_monitor = monitors[1]
        mode = glfw.get_video_mode(second_monitor)
        width, height = mode.size

        window = glfw.create_window(
            width, height,
            "Luminance Display",
            second_monitor,
            None
        )

        glfw.make_context_current(window)
        glfw.swap_interval(1)

        for idx, lum in enumerate(self.luminance_list):
            self.current_index = idx

            print(f"\nDisplaying luminance: {lum:.5f}")

            # Render full screen gray
            glClearColor(lum, lum, lum, 1.0)
            glClear(GL_COLOR_BUFFER_BIT)
            glfw.swap_buffers(window)

            # small stabilization delay
            time.sleep(0.5)

            # Signal measurement thread
            self.event_start_measure.set()

            # Wait for measurement to finish
            self.event_measure_done.wait()
            self.event_measure_done.clear()

            time.sleep(0.5)

        glfw.terminate()


# ==========================================================
# Measurement Thread
# ==========================================================

class MeasurementThread(threading.Thread):
    def __init__(self, luminance_list, event_start_measure, event_measure_done):
        super().__init__()
        self.luminance_list = luminance_list
        self.event_start_measure = event_start_measure
        self.event_measure_done = event_measure_done
        self.results = {}

    def run(self):
        repeat_times = 5

        for idx, lum in enumerate(self.luminance_list):

            # Wait until display ready
            self.event_start_measure.wait()
            self.event_start_measure.clear()

            print(f"Measuring luminance index {idx}")

            Y_list, x_list, y_list = [], [], []
            attempts = 0

            while len(Y_list) < repeat_times:
                try:
                    Y, x, y = specbos_measure()
                except Exception:
                    continue

                if Y is None:
                    continue

                Y_list.append(Y)
                x_list.append(x)
                y_list.append(y)
                attempts += 1

                print(f"  Measurement {len(Y_list)}/{repeat_times}: Y={Y}")

            # Outlier removal
            Y_list = remove_outliers(Y_list)
            x_list = remove_outliers(x_list)
            y_list = remove_outliers(y_list)

            self.results[f"L_{idx}"] = {
                "target_rgb": float(lum),
                "Y_list": Y_list,
                "x_list": x_list,
                "y_list": y_list,
                "repeat_times": repeat_times
            }

            self.event_measure_done.set()


# ==========================================================
# Main
# ==========================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default="luminance_measure_specbos.json"
    )
    parser.add_argument(
        "--points",
        type=int,
        default=10
    )

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    luminance_list = generate_log_luminance_points(args.points)

    event_start_measure = threading.Event()
    event_measure_done = threading.Event()

    display_thread = DisplayThread(
        luminance_list,
        event_start_measure,
        event_measure_done
    )

    measurement_thread = MeasurementThread(
        luminance_list,
        event_start_measure,
        event_measure_done
    )

    display_thread.start()
    measurement_thread.start()

    display_thread.join()
    measurement_thread.join()

    with open(args.output, "w") as f:
        json.dump(measurement_thread.results, f, indent=4)

    print("\nFinished. Results saved to:", args.output)


if __name__ == "__main__":
    main()