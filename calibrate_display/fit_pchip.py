import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.interpolate import PchipInterpolator
import os
print("Current Working Directory:", os.getcwd())

# =========================================================
# Argument Parser
# =========================================================
parser = argparse.ArgumentParser()
parser.add_argument("--fit_log", action="store_true",
                    help="Fit in log10 luminance domain")
parser.add_argument("--plot_log", action="store_true",
                    help="Use log scale in plots")
args = parser.parse_args()

FIT_LOG = args.fit_log
PLOT_LOG = args.plot_log


# =========================================================
# Load data
# =========================================================
# with open("Measure_specbos/luminance_pixel_measure_B100_C100_s800.json", "r") as f:
#     data = json.load(f)
with open("Measure_specbos/luminance_pixel_measure_patch800_dark.json", "r") as f:
    data = json.load(f)

pixels = []
Y_values = []

for k in data:
    pixels.append(data[k]["pixel_value"])
    Y_values.append(data[k]["Y_mean"])

pixels = np.array(pixels)
Y_values = np.array(Y_values)

# Sort (PCHIP 必须单调)
sort_idx = np.argsort(pixels)
pixels = pixels[sort_idx]
Y_values = Y_values[sort_idx]

p_norm = pixels / 255.0

epsilon = 1e-8
Y_safe = np.clip(Y_values, epsilon, None)


# =========================================================
# Select fitting domain
# =========================================================
if FIT_LOG:
    Y_fit = np.log10(Y_safe)
    fit_domain = "log10"
else:
    Y_fit = Y_safe
    fit_domain = "linear"


# =========================================================
# PCHIP Fit
# =========================================================
p2L_model = PchipInterpolator(p_norm, Y_fit)
L2p_model = PchipInterpolator(Y_fit, p_norm)


# =========================================================
# Save model
# =========================================================
model = {
    "model_name": "PixelLuminanceModel_PCHIP",
    "fit_domain": fit_domain,
    "method": "PCHIP",
    "pixel_samples": p_norm.tolist(),
    "luminance_samples": Y_fit.tolist()
}

with open("Measure_specbos/pixel_luminance_model_pchip_B100_C100_s800_dark.json", "w") as f:
    json.dump(model, f, indent=4)

print("Saved pixel_luminance_model_pchip.json")
print(f"Fit domain: {fit_domain}")


# =========================================================
# Generate smooth curves
# =========================================================
p_smooth = np.linspace(0, 1, 500)
Y_fit_smooth = p2L_model(p_smooth)

if FIT_LOG:
    L_fit = 10 ** Y_fit_smooth
else:
    L_fit = Y_fit_smooth


# Inverse smooth
if FIT_LOG:
    Y_smooth = np.linspace(np.log10(Y_safe.min()),
                           np.log10(Y_safe.max()), 500)
    L_smooth = 10 ** Y_smooth
else:
    Y_smooth = np.linspace(Y_safe.min(),
                           Y_safe.max(), 500)
    L_smooth = Y_smooth

p_fit_inv = L2p_model(Y_smooth)


# =========================================================
# Plot 1️⃣ : p -> L
# =========================================================
plt.figure(figsize=(8, 6))

plt.scatter(p_norm, Y_safe, label="Measured", zorder=3)
plt.plot(p_smooth, L_fit,
         label="PCHIP Fit", linewidth=2)

if PLOT_LOG:
    plt.yscale("log")

plt.xlabel("Normalized Pixel p")
plt.ylabel("Luminance Y (cd/m²)")
plt.title("Pixel → Luminance (PCHIP)")
plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()


# =========================================================
# Plot 2️⃣ : L -> p
# =========================================================
plt.figure(figsize=(8, 6))

plt.scatter(Y_safe, p_norm, label="Measured", zorder=3)
plt.plot(L_smooth, p_fit_inv,
         label="Inverse PCHIP", linewidth=2)

if PLOT_LOG:
    plt.xscale("log")

plt.xlabel("Luminance Y (cd/m²)")
plt.ylabel("Normalized Pixel p")
plt.title("Luminance → Pixel (PCHIP)")
plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()


# =========================================================
# Plot 3️⃣ : Round-trip Consistency
# =========================================================
p_test = np.linspace(0, 1, 200)
Y_test = p2L_model(p_test)

p_roundtrip = L2p_model(Y_test)

plt.figure(figsize=(6, 6))
plt.plot(p_test, p_test, 'k--', label="Ideal")
plt.plot(p_test, p_roundtrip, label="Round-trip")
plt.xlabel("Original p")
plt.ylabel("Recovered p")
plt.title("Round-trip Consistency Check (PCHIP)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()