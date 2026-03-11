import pandas as pd

df = pd.read_csv("Quest_results.csv")

if "sf_cpp_eff" not in df.columns:
    df.insert(df.columns.get_loc("spatial_frequency_cpp") + 1, "sf_cpp_eff", df["spatial_frequency_cpp"])

if "speed_eff" not in df.columns:
    df.insert(df.columns.get_loc("sf_cpp_eff") + 1, "speed_eff", df["speed_px_per_sec"])

df.to_csv("Quest_results.csv", index=False)
print("Done.")