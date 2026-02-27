import json
import numpy as np
import matplotlib.pyplot as plt


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

epsilon = 1e-6
Y_safe = np.clip(Y_values, epsilon, None)
logY = np.log10(Y_safe)


# =========================================================
# Fit p -> L   (log domain)
# =========================================================
degree_p2L = 6
coeffs_p2L = np.polyfit(p_norm, logY, degree_p2L)


# =========================================================
# Fit L -> p
# =========================================================
degree_L2p = 6
coeffs_L2p = np.polyfit(logY, p_norm, degree_L2p)


# =========================================================
# Save model
# =========================================================
model = {
    "model_name": "PixelLuminanceModel",
    "fit_domain": "log10",
    "degree_p2L": degree_p2L,
    "coeffs_p2L": coeffs_p2L.tolist(),
    "degree_L2p": degree_L2p,
    "coeffs_L2p": coeffs_L2p.tolist()
}

with open("pixel_luminance_model.json", "w") as f:
    json.dump(model, f, indent=4)

print("Saved pixel_luminance_model.json")


# =========================================================
# Generate smooth curves
# =========================================================
p_smooth = np.linspace(0, 1, 500)
logL_fit = np.polyval(coeffs_p2L, p_smooth)
L_fit = 10 ** logL_fit

# inverse check
logL_smooth = np.linspace(logY.min(), logY.max(), 500)
p_fit_inv = np.polyval(coeffs_L2p, logL_smooth)
L_smooth = 10 ** logL_smooth


# =========================================================
# Plot 1️⃣ : p -> L  (Y in log domain)
# =========================================================
plt.figure(figsize=(8,6))

plt.scatter(p_norm, Y_safe, label="Measured", zorder=3)
plt.plot(p_smooth, L_fit, label=f"Poly Fit (deg={degree_p2L})", linewidth=2)

plt.yscale("log")

plt.xlabel("Normalized Pixel p")
plt.ylabel("Luminance Y (cd/m²)")
plt.title("Pixel → Luminance (Log Y Domain)")
plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()


# =========================================================
# Plot 2️⃣ : L -> p  (log L on x-axis)
# =========================================================
plt.figure(figsize=(8,6))

plt.scatter(Y_safe, p_norm, label="Measured", zorder=3)
plt.plot(L_smooth, p_fit_inv, label=f"Inverse Fit (deg={degree_L2p})", linewidth=2)

plt.xscale("log")

plt.xlabel("Luminance Y (cd/m²)")
plt.ylabel("Normalized Pixel p")
plt.title("Luminance → Pixel")
plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()


# =========================================================
# Plot 3️⃣ : Consistency Check (p -> L -> p)
# =========================================================
p_test = np.linspace(0, 1, 200)
logL_test = np.polyval(coeffs_p2L, p_test)
p_roundtrip = np.polyval(coeffs_L2p, logL_test)

plt.figure(figsize=(6,6))
plt.plot(p_test, p_test, 'k--', label="Ideal")
plt.plot(p_test, p_roundtrip, label="Round-trip")
plt.xlabel("Original p")
plt.ylabel("Recovered p")
plt.title("Round-trip Consistency Check")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()