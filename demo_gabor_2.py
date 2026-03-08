"""
demo_gabor.py
=============
Quick demo: import gabor_renderer and show a single drifting Gabor stimulus
with fully specified parameters.
"""
import glfw
from gabor_render import ExperimentRenderer
from gabor_render import (
    compute_display_width,
    compute_spatiotemporal_frequency,
    visual_radius_deg_to_px,
)
import numpy as np

# ── Parameters ────────────────────────────────────────────────────────────────
DIAGONAL_INCH         = 27.0
VISUAL_RADIUS_DEG     = 2.0
MONITOR_INDEX         = 1
COLOR_DIRECTION       = "yv"
CONTRAST              = 0.92
SPATIAL_FREQ_CPP      = 0.1
SPEED_PX_PER_SEC      = 100
MEAN_LUMINANCE        = 5
DISTANCE_M            = 2.6
STIMULUS_DURATION_SEC = 100.0
# ──────────────────────────────────────────────────────────────────────────────

def print_stimulus_params(renderer, spatial_freq_cpp, speed_px_s, distance_m,
                           contrast, mean_luminance, color_direction,
                           visual_radius_deg):
    """Derive and print all physical / perceptual stimulus parameters."""
    W = renderer.W
    width  = renderer.width
    height = renderer.height
    refresh = renderer.refresh

    # Spatial frequency: cycles per degree
    # f_cpd = f_cpp * pixels_per_degree
    # pixels_per_degree = (width / W) * (W / (2*distance_m*tan(0.5°))) ...
    # simpler: use compute_spatiotemporal_frequency with f_p=spatial_freq_cpp, v_p=speed_px_s
    rho_cpd, omega_hz, theta_rad = compute_spatiotemporal_frequency(
        R_x=width,
        W=W,
        d=distance_m,
        f_p=spatial_freq_cpp,   # cycles/pixel
        v_p=speed_px_s          # pixels/second  → omega = f_p * v_p = cycles/second
    )
    theta_deg = np.rad2deg(theta_rad)

    # Gabor radius in pixels
    radius_px = visual_radius_deg_to_px(visual_radius_deg, distance_m, W, width)

    # Pixels per degree (at this distance)
    px_per_deg = (width / W) * (distance_m * np.tan(np.deg2rad(1.0)))
    # simpler direct:
    px_per_deg = width / theta_deg  # theta_deg = full horizontal FOV

    # Temporal frequency (Hz) = spatial_freq_cpp * speed_px_s
    temporal_freq_hz = spatial_freq_cpp * speed_px_s

    print("\n" + "="*55)
    print("  STIMULUS PARAMETERS")
    print("="*55)
    print(f"  Display diagonal      : {DIAGONAL_INCH:.1f} inch")
    print(f"  Resolution            : {width} × {height} px")
    print(f"  Refresh rate          : {refresh} Hz")
    print(f"  Display width         : {W*100:.2f} cm")
    print(f"  Viewing distance      : {distance_m:.2f} m")
    print(f"  Horizontal FOV        : {theta_deg:.2f}°")
    print(f"  Pixels per degree     : {width/theta_deg:.2f} px/°")
    print("-"*55)
    print(f"  Colour direction      : {color_direction}")
    print(f"  Mean luminance        : {mean_luminance:.1f} cd/m²")
    print(f"  Contrast (Michelson)  : {contrast:.4f}")
    print("-"*55)
    print(f"  Spatial freq (cpp)    : {spatial_freq_cpp:.4f} cyc/px")
    print(f"  Spatial freq (cpd)    : {rho_cpd:.4f} cyc/°")
    print(f"  Speed                 : {speed_px_s:.1f} px/s")
    print(f"  Temporal freq         : {temporal_freq_hz:.2f} Hz")
    print(f"  Gabor radius          : {visual_radius_deg:.2f}°  ({radius_px:.1f} px)")
    print("="*55 + "\n")


def main():
    glfw.init()
    renderer = ExperimentRenderer(
        diagonal_inch=DIAGONAL_INCH,
        visual_radius_deg=VISUAL_RADIUS_DEG,
        monitor_index=MONITOR_INDEX,
        lut_json_path="calibrate_display/Measure_specbos/pixel_luminance_model_pchip_B100_C100_s800_dark.json"
    # lut_json_path = "calibrate_display/Measure_specbos/pixel_luminance_model_pchip_B100_C100_s800.json"
    )
    renderer.init_window()
    renderer.set_condition(COLOR_DIRECTION, MEAN_LUMINANCE)

    # ── Print all derived parameters ──────────────────────────────────────────
    print_stimulus_params(
        renderer       = renderer,
        spatial_freq_cpp = SPATIAL_FREQ_CPP,
        speed_px_s     = SPEED_PX_PER_SEC,
        distance_m     = DISTANCE_M,
        contrast       = CONTRAST,
        mean_luminance = MEAN_LUMINANCE,
        color_direction= COLOR_DIRECTION,
        visual_radius_deg = VISUAL_RADIUS_DEG,
    )

    # Brief grey screen before stimulus
    renderer.show_flat(DISTANCE_M, (0.2, 0.2, 0.2))
    print("Showing Gabor …  (ESC to quit early)")

    esc = renderer.show_gabor(
        contrast=CONTRAST,
        distance_m=DISTANCE_M,
        spatial_frequency_cpp=SPATIAL_FREQ_CPP,
        speed=SPEED_PX_PER_SEC,
        duration=STIMULUS_DURATION_SEC,
    )

    if esc:
        print("ESC pressed — exiting.")
    else:
        print("Stimulus finished.")

    renderer.destroy()
    glfw.terminate()

if __name__ == "__main__":
    main()