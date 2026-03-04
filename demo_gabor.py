"""
demo_gabor.py
=============
Quick demo: import gabor_renderer and show a single drifting Gabor stimulus
with fully specified parameters.

Parameters used
---------------
spatial_frequency_cpp : 0.1
speed (px/s)          : 120
viewing distance (m)  : 0.5
radius (deg)          : 2.0
mean luminance (cd/m²): 50.0
colour direction      : rg
contrast              : 0.15
display diagonal (in) : 27
"""

import glfw
from gabor_render import ExperimentRenderer
import time
# ── Parameters ────────────────────────────────────────────────────────────────
DIAGONAL_INCH         = 27.0
VISUAL_RADIUS_DEG     = 2.0
MONITOR_INDEX         = 1       # change to 0 if you only have one display

COLOR_DIRECTION       = "rg"
MEAN_LUMINANCE        = 50
CONTRAST              = 0.14

SPATIAL_FREQ_CPP      = 0.2
# SPEED_PX_PER_SEC      = 180.0
SPEED_PX_PER_SEC      = 50.0
DISTANCE_M            = 1 #2.1
STIMULUS_DURATION_SEC = 100.0     # how long to show the Gabor
# ──────────────────────────────────────────────────────────────────────────────


def main():
    glfw.init()

    renderer = ExperimentRenderer(
        diagonal_inch=DIAGONAL_INCH,
        visual_radius_deg=VISUAL_RADIUS_DEG,
        monitor_index=MONITOR_INDEX,
    )
    renderer.init_window()
    renderer.set_condition(COLOR_DIRECTION, MEAN_LUMINANCE)

    # Brief grey screen before stimulus
    renderer.show_flat(DISTANCE_M,(0.2, 0.2, 0.2))
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