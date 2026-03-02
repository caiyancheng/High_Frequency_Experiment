import numpy as np
import json
import matplotlib.pyplot as plt


# ==========================================================
# Load Data
# ==========================================================

with open("Measure_specbos/rgb_xyz_measure.json") as f:
    data = json.load(f)

RGB = np.array([d["RGB"] for d in data])
XYZ = np.array([d["XYZ"] for d in data])


# ==========================================================
# Fit Matrix
# ==========================================================

M_rgb_xyz, _, _, _ = np.linalg.lstsq(RGB, XYZ, rcond=None)
M_xyz_rgb = np.linalg.inv(M_rgb_xyz)

print("\nM_rgb_xyz:\n", M_rgb_xyz)
print("\nM_xyz_rgb:\n", M_xyz_rgb)


# ==========================================================
# Reconstruct RGB
# ==========================================================

RGB_reconstructed = (XYZ @ M_xyz_rgb.T)

RGB_reconstructed = np.clip(RGB_reconstructed, 0, 1)


# ==========================================================
# Visualization
# ==========================================================

fig, axes = plt.subplots(4, 6, figsize=(12,6))

for i in range(24):

    r = i // 6
    c = i % 6

    axes[r,c].imshow([[RGB[i]]])
    axes[r,c].set_xticks([])
    axes[r,c].set_yticks([])

plt.suptitle("Original Display RGB")
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(4, 6, figsize=(12,6))

for i in range(24):

    r = i // 6
    c = i % 6

    axes[r,c].imshow([[RGB_reconstructed[i]]])
    axes[r,c].set_xticks([])
    axes[r,c].set_yticks([])

plt.suptitle("Reconstructed RGB from Measured XYZ")
plt.tight_layout()
plt.show()