import json
import numpy as np


class PixelLuminanceModel:
    """
    Data-driven pixel <-> luminance calibration model.

    p : normalized pixel in [0,1]
    L : luminance in cd/m^2
    """

    def __init__(self, json_path):
        with open(json_path, "r") as f:
            data = json.load(f)

        self.coeffs_p2L = np.array(data["coeffs_p2L"])
        self.coeffs_L2p = np.array(data["coeffs_L2p"])
        self.epsilon = 1e-6

    # --------------------------------------------------
    # pixel -> luminance
    # --------------------------------------------------
    def p2L(self, p):
        p = np.clip(p, 0.0, 1.0)
        logL = np.polyval(self.coeffs_p2L, p)
        return 10 ** logL

    # --------------------------------------------------
    # luminance -> pixel
    # --------------------------------------------------
    def L2p(self, L):
        L_safe = np.clip(L, self.epsilon, None)
        logL = np.log10(L_safe)
        p = np.polyval(self.coeffs_L2p, logL)
        return np.clip(p, 0.0, 1.0)


class SRGBLuminanceModel:
    """
    Physically-based sRGB pixel <-> luminance model.

    p : normalized pixel in [0,1]
    L : luminance in cd/m^2
    """

    def __init__(self, json_path='srgb_luminance_model.json'):
        with open(json_path, "r") as f:
            data = json.load(f)

        self.L_min = data["L_min"]
        self.L_max = data["L_max"]
        self.epsilon = 1e-8


    # ==================================================
    # sRGB EOTF
    # ==================================================
    def _srgb_eotf(self, p):
        p = np.clip(p, 0.0, 1.0)
        return np.where(
            p <= 0.04045,
            p / 12.92,
            ((p + 0.055) / 1.055) ** 2.4
        )


    # ==================================================
    # sRGB inverse EOTF
    # ==================================================
    def _srgb_inverse_eotf(self, x):
        x = np.clip(x, 0.0, 1.0)
        return np.where(
            x <= 0.0031308,
            12.92 * x,
            1.055 * (x ** (1/2.4)) - 0.055
        )


    # ==================================================
    # pixel -> luminance
    # ==================================================
    def p2L(self, p):
        p = np.clip(p, 0.0, 1.0)
        linear = self._srgb_eotf(p)
        return self.L_min + (self.L_max - self.L_min) * linear


    # ==================================================
    # luminance -> pixel
    # ==================================================
    def L2p(self, L):
        L = np.clip(L, self.L_min, self.L_max)

        linear = (L - self.L_min) / (self.L_max - self.L_min)
        p = self._srgb_inverse_eotf(linear)

        return np.clip(p, 0.0, 1.0)