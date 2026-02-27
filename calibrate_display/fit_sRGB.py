import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# =========================================================
# Load data
# =========================================================
with open("Measure_specbos/luminance_pixel_measure.json", "r") as f:
    data = json.load(f)

pixels = []
Y_values = []

for k in data:
    pixels.append(data[k]["pixel_value"])
    Y_values.append(data[k]["Y_mean"])

pixels = np.array(pixels)
Y_values = np.array(Y_values)

p_norm = pixels / 255.0
Y_safe = np.clip(Y_values, 1e-6, None)


# =========================================================
# sRGB EOTF
# =========================================================
def srgb_eotf(p):
    p = np.clip(p, 0.0, 1.0)
    return np.where(
        p <= 0.04045,
        p / 12.92,
        ((p + 0.055) / 1.055) ** 2.4
    )


# =========================================================
# Fit L_max
# =========================================================
L_min = np.min(Y_safe)

def srgb_model(p, L_max):
    return L_min + (L_max - L_min) * srgb_eotf(p)


popt, _ = curve_fit(srgb_model, p_norm, Y_safe, p0=[np.max(Y_safe)])
L_max_fit = popt[0]

print("Fitted L_min:", L_min)
print("Fitted L_max:", L_max_fit)


# =========================================================
# Save model
# =========================================================
model = {
    "model_name": "sRGB_Luminance_Model",
    "L_min": float(L_min),
    "L_max": float(L_max_fit)
}

with open("srgb_luminance_model.json", "w") as f:
    json.dump(model, f, indent=4)

print("Saved srgb_luminance_model.json")


# =========================================================
# Plot
# =========================================================
p_smooth = np.linspace(0, 1, 500)
L_fit = srgb_model(p_smooth, L_max_fit)

plt.figure(figsize=(8,6))
plt.scatter(p_norm, Y_safe, label="Measured", zorder=3)
plt.plot(p_smooth, L_fit, label="sRGB Fit", linewidth=2)

plt.xlabel("Normalized Pixel p")
plt.ylabel("Luminance Y (cd/m²)")
plt.title("sRGB Pixel → Luminance Fit")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()