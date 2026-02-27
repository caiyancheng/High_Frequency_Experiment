import argparse
import numpy as np
from gfxdisp.specbos import specbos_measure, specbos_get_sprad
import json
import os


def remove_outliers(data_list, threshold=2):
    """
    Remove outliers based on mean ± threshold * std.
    If all values are filtered out, return the original list.
    """
    data_array = np.array(data_list)
    mean = np.mean(data_array)
    std = np.std(data_array)
    filtered = [val for val in data_list if abs(val - mean) <= threshold * std]
    return filtered if len(filtered) > 0 else data_list


def main():
    # ---------------------- Argument Parsing ---------------------- #
    parser = argparse.ArgumentParser(
        description="Measure color patches using specbos and save results to JSON."
    )
    parser.add_argument(
        "--output", type=str,
        default=r"Color_measure_specbos/Color_measure_Sony_a7R3_FE90_F20_Eizo_specbos_2025_5_5_filter.json",
        help="Path to save JSON results"
    )
    args = parser.parse_args()

    # Ensure directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # ---------------------- Measurement --------------------------- #
    repeat_times = 10
    json_data_dict = {}
    color_length = 9

    for color_index in range(color_length):
        print(f"Measuring color patch {color_index}")
        Y_list, x_list, y_list = [], [], []
        attempt_count = 0

        while len(Y_list) < repeat_times:
            Y = None
            while Y is None:
                try:
                    Y, x, y = specbos_measure()
                except Exception:
                    continue

            attempt_count += 1
            if Y is None or x is None or y is None:
                print(f"  Measurement invalid (None). Retrying... (Attempt {attempt_count})")
                continue

            print(Y)
            Y_list.append(Y)
            x_list.append(x)
            y_list.append(y)

        json_data = {
            "Y_list": Y_list,
            "x_list": x_list,
            "y_list": y_list,
            "repeat_times": repeat_times,
        }
        json_data_dict[f"C_{color_index}"] = json_data

    # ---------------------- Save Results -------------------------- #
    with open(args.output, "w") as f:
        json.dump(json_data_dict, f, indent=4)

    print(f"Measurement finished. Data saved to {args.output}")


if __name__ == "__main__":
    main()
