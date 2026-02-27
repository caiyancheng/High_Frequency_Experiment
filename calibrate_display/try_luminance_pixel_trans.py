from calibrate_display.display_calibrate import SRGBLuminanceModel
import numpy as np
import matplotlib.pyplot as plt


# ==========================================================
# Load Model
# ==========================================================
model = SRGBLuminanceModel()


# ==========================================================
# 1️⃣ 构造 pixel array (0~255)
# ==========================================================
pixels = np.arange(0, 256)
p_norm = pixels / 255.0


# ==========================================================
# 2️⃣ p -> L
# ==========================================================
L_values = model.p2L(p_norm)


# ==========================================================
# 3️⃣ L -> p
# ==========================================================
p_recovered = model.L2p(L_values)


# ==========================================================
# 4️⃣ 误差分析
# ==========================================================
error = p_recovered - p_norm

rmse = np.sqrt(np.mean(error ** 2))
mae = np.mean(np.abs(error))
max_err = np.max(np.abs(error))

print("===== Round-trip Error =====")
print("RMSE:", rmse)
print("MAE :", mae)
print("Max :", max_err)


# ==========================================================
# 5️⃣ Plot 1 : Pixel -> Luminance
# ==========================================================
plt.figure(figsize=(8,6))
plt.plot(pixels, L_values, linewidth=2)
plt.xlabel("Pixel (0-255)")
plt.ylabel("Luminance (cd/m²)")
plt.title("Pixel → Luminance (sRGB Model)")
plt.grid(True)
plt.tight_layout()
plt.show()


# ==========================================================
# 6️⃣ Plot 2 : Luminance -> Pixel
# ==========================================================
plt.figure(figsize=(8,6))
plt.plot(L_values, p_recovered * 255, linewidth=2)
plt.xlabel("Luminance (cd/m²)")
plt.ylabel("Recovered Pixel")
plt.title("Luminance → Pixel")
plt.grid(True)
plt.tight_layout()
plt.show()


# ==========================================================
# 7️⃣ Plot 3 : Round-trip Consistency
# ==========================================================
plt.figure(figsize=(6,6))
plt.plot(pixels, pixels, 'k--', label="Ideal")
plt.plot(pixels, p_recovered * 255, label="Round-trip")
plt.xlabel("Original Pixel")
plt.ylabel("Recovered Pixel")
plt.title("Round-trip Consistency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()