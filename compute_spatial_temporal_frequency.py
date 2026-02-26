import numpy as np

def compute_display_width(diagonal_inch: float, R_x: int, R_y: int):
    """
    Compute physical display width (meters) from diagonal size and resolution.
    """
    D_m = diagonal_inch * 0.0254  # inch -> meter
    W = D_m * (R_x / np.sqrt(R_x**2 + R_y**2))
    return W

def compute_spatiotemporal_frequency(
    R_x: int,
    W: float,
    d: float,
    f_p: float,
    v_p: float,
    radian: bool = False
):
    """
    Compute retinal spatial frequency (cpd) and temporal frequency (Hz)
    from display-domain parameters (exact geometric form, no small-angle approximation).

    Parameters
    ----------
    R_x : int
        Horizontal resolution (pixels)
    W : float
        Display width (meters)
    d : float
        Viewing distance (meters)
    f_p : float
        Spatial frequency in cycles per pixel (cpp)
    v_p : float
        Drift speed in pixels per second
    radian : bool
        If True, output spatial frequency in cycles per radian.
        If False (default), output cycles per degree (cpd).

    Returns
    -------
    rho : float
        Spatial frequency (cpd or cycles/radian)
    omega : float
        Temporal frequency (Hz)
    """

    # total horizontal visual angle (radians)
    theta = 2 * np.arctan(W / (2 * d))

    # spatial frequency (cycles per radian)
    rho_rad = (R_x * f_p) / theta

    if radian:
        rho = rho_rad
    else:
        # convert to cycles per degree
        rho = rho_rad * (np.pi / 180)

    # temporal frequency
    omega = f_p * v_p

    return rho, omega


# Example usage
if __name__ == "__main__":
    R_x = 3840       # 4K horizontal resolution
    R_y = 2160
    # W = 0.70         # 70 cm display width
    W = compute_display_width(27, R_x, R_y)
    d = 1.0          # 1 meter viewing distance
    f_p = 0.1       # cycles per pixel
    v_p = 100        # pixels per second

    rho, omega = compute_spatiotemporal_frequency(R_x, W, d, f_p, v_p)

    print(f"Spatial frequency (cpd): {rho:.3f}")
    print(f"Temporal frequency (Hz): {omega:.3f}")
