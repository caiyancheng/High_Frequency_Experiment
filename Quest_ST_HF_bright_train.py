"""
Quest 2AFC Experiment — TRAINING SESSION
=========================================
Identical to the main experiment but:
  - No CSV output
  - Fewer trials per condition (TRAINING_TRIALS_PER_CONDITION)
  - Console feedback shows "TRAINING" prominently
  - Uses a fixed subset of conditions (one per color by default)
"""

import argparse
import random
import time

import glfw
import numpy as np
from psychopy.data import QuestHandler

from control_display.control_display_main import rsbg_init, rsbg_update, rsbg_cleanup
from gabor_render import (
    ExperimentRenderer,
    compute_spatiotemporal_frequency,
)

# ==========================================================
# Constants
# ==========================================================
INITIAL_DISTANCE             = 1
UNIT_PER_M                   = 100 / 1.6
MIN_DIST, MAX_DIST           = 1, 2.6
TRAINING_TRIALS_PER_CONDITION = 10        # fewer trials for training
REPEATS_PER_POSITION         = 3

# ==========================================================
# Geometry & cpd_inv conversion
# ==========================================================

def ppd_from_dist(dist_m, screen_width_px, screen_width_m):
    half_fov_deg = np.degrees(np.arctan(screen_width_m / (2.0 * dist_m)))
    return screen_width_px / (2.0 * half_fov_deg)


def dist_from_ppd(ppd, screen_width_px, screen_width_m):
    half_fov_deg = screen_width_px / (2.0 * ppd)
    return screen_width_m / (2.0 * np.tan(np.radians(half_fov_deg)))


def cpd_inv_from_dist(dist_m, sf_cpp, screen_width_px, screen_width_m):
    ppd     = ppd_from_dist(dist_m, screen_width_px, screen_width_m)
    cpd     = ppd * sf_cpp
    cpd_inv = -(cpd ** (1.0 / 3.0))
    return float(cpd_inv)


def dist_from_cpd_inv(cpd_inv, sf_cpp, screen_width_px, screen_width_m):
    cpd = (-cpd_inv) ** 3.0
    ppd = cpd / sf_cpp
    return float(dist_from_ppd(ppd, screen_width_px, screen_width_m))


def cpd_inv_range_from_dist_range(min_dist, max_dist, sf_cpp,
                                   screen_width_px, screen_width_m):
    cpd_inv_near = cpd_inv_from_dist(min_dist, sf_cpp, screen_width_px, screen_width_m)
    cpd_inv_far  = cpd_inv_from_dist(max_dist, sf_cpp, screen_width_px, screen_width_m)
    return (cpd_inv_near, cpd_inv_far)


# ==========================================================
# Platform controller
# ==========================================================

class PlatformController:
    def __init__(self, port):
        self.port = rsbg_init(port)
        rsbg_update(self.port, 'reset')
        rsbg_update(self.port, 'move_and_wait', 0)

    def move_to(self, distance_m: float):
        delta = distance_m - INITIAL_DISTANCE
        units = int(round(-delta * UNIT_PER_M))
        rsbg_update(self.port, 'move_and_wait', units)

    def cleanup(self):
        rsbg_update(self.port, 'move_and_wait', 0)
        rsbg_cleanup(self.port)


# ==========================================================
# Quest helpers
# ==========================================================

def make_quest(start_cpd_inv, cpd_inv_range):
    lo, hi = min(cpd_inv_range), max(cpd_inv_range)
    span   = hi - lo
    return QuestHandler(
        startVal   = start_cpd_inv,
        startValSd = span / 2.0,
        pThreshold = 0.75,
        nTrials    = TRAINING_TRIALS_PER_CONDITION,
        beta=3.5, delta=0.05, gamma=0.5,
        grain=0.01,
        range=span,
        minVal=lo,
        maxVal=hi,
        method='quantile',
    )


def quest_suggest_cpd_inv(q, cpd_inv_range):
    lo, hi = min(cpd_inv_range), max(cpd_inv_range)
    return float(np.clip(q._questNextIntensity, lo, hi))


def quest_mean_cpd_inv(q, cpd_inv_range):
    lo, hi = min(cpd_inv_range), max(cpd_inv_range)
    return float(np.clip(q.mean(), lo, hi))


def quest_sd(q):
    return float(q.sd())


# ==========================================================
# Main
# ==========================================================

def main():
    parser = argparse.ArgumentParser(description="Training session (no CSV output)")
    parser.add_argument("--name", default='YanchengCai')
    parser.add_argument("--colors", nargs="+", default=["ach", "rg", "yv"])
    # Use one representative speed per color for training
    parser.add_argument("--ach_speeds",  nargs="+", type=float, default=[200])
    parser.add_argument("--rg_speeds",   nargs="+", type=float, default=[200])
    parser.add_argument("--yv_speeds",   nargs="+", type=float, default=[200])
    parser.add_argument("--ach_luminance_list", nargs="+", type=float, default=[50])
    parser.add_argument("--rg_luminance_list",  nargs="+", type=float, default=[50])
    parser.add_argument("--yv_luminance_list",  nargs="+", type=float, default=[50])
    parser.add_argument("--ach_spatial_frequency_cpp", type=float, default=0.05)
    parser.add_argument("--rg_spatial_frequency_cpp",  type=float, default=0.05)
    parser.add_argument("--yv_spatial_frequency_cpp",  type=float, default=0.05)
    parser.add_argument("--ach_contrast", type=float, default=0.9)
    parser.add_argument("--rg_contrast",  type=float, default=0.14)
    parser.add_argument("--yv_contrast",  type=float, default=0.92)
    parser.add_argument("--diagonal_inch",     type=float, default=27)
    parser.add_argument("--visual_radius_deg", type=float, default=2.0)
    parser.add_argument("--port",          default="/dev/ttyACM0")
    parser.add_argument("--duration",      type=float, default=2.0)
    parser.add_argument("--monitor_index", type=int,   default=1)
    parser.add_argument("--trials",        type=int,   default=TRAINING_TRIALS_PER_CONDITION,
                        help="Trials per condition in training (default: %(default)s)")
    args = parser.parse_args()

    n_trials = args.trials

    # Build condition list (typically just a handful for training)
    conditions = []
    for color in args.colors:
        contrast = getattr(args, f"{color}_contrast")
        sf_cpp   = getattr(args, f"{color}_spatial_frequency_cpp")
        for spd in getattr(args, f"{color}_speeds"):
            for lum in getattr(args, f"{color}_luminance_list"):
                conditions.append({
                    "color": color, "speed": spd, "contrast": contrast,
                    "luminance": lum, "spatial_frequency": sf_cpp,
                })
    random.shuffle(conditions)

    total_conditions = len(conditions)

    print("\n" + "="*60)
    print("  *** TRAINING SESSION — no data will be saved ***")
    print(f"  Participant : {args.name}")
    print(f"  Conditions  : {total_conditions}")
    print(f"  Trials each : {n_trials}")
    print("="*60 + "\n")

    platform = PlatformController(args.port)

    glfw.init()
    renderer = ExperimentRenderer(
        diagonal_inch=args.diagonal_inch,
        visual_radius_deg=args.visual_radius_deg,
        monitor_index=args.monitor_index,
        lut_json_path="calibrate_display/Measure_specbos/pixel_luminance_model_pchip_B100_C100_s800.json"
    )
    renderer.init_window()

    screen_width_px = renderer.width
    screen_width_m  = renderer.W

    experiment_start = time.perf_counter()
    aborted          = False

    try:
        for cond_idx, cond in enumerate(conditions):
            color, speed  = cond['color'],  cond['speed']
            contrast, lum = cond['contrast'], cond['luminance']
            sf_cpp        = cond['spatial_frequency']

            cpd_inv_range = cpd_inv_range_from_dist_range(
                MIN_DIST, MAX_DIST, sf_cpp, screen_width_px, screen_width_m)

            # Start from middle of distance range
            start_dist    = 1.1
            #float(np.clip((MIN_DIST + MAX_DIST) / 2, MIN_DIST, MAX_DIST))
            start_cpd_inv = cpd_inv_from_dist(
                start_dist, sf_cpp, screen_width_px, screen_width_m)

            print(f"\n{'='*60}")
            print(f"[TRAINING] Condition {cond_idx+1}/{total_conditions}")
            print(f"  Color:{color}  Speed:{speed}px/s  Lum:{lum}cd/m²  sf_cpp:{sf_cpp}")
            print(f"  cpd_inv range: [{cpd_inv_range[0]:.4f}, {cpd_inv_range[1]:.4f}]")
            print(f"  start_dist:{start_dist:.3f}m  start_cpd_inv:{start_cpd_inv:.4f}")
            print('='*60)

            quest = make_quest(start_cpd_inv, cpd_inv_range)

            half         = n_trials // 2
            interval_seq = [1]*half + [2]*half
            if n_trials % 2:
                interval_seq.append(random.choice([1, 2]))
            random.shuffle(interval_seq)
            seq_idx = 0

            dist         = start_dist
            cpd_inv_test = start_cpd_inv
            sf_cpp_eff   = sf_cpp
            speed_eff    = speed

            renderer.set_condition(color, lum)

            for trial_idx in range(n_trials):
                if trial_idx % REPEATS_PER_POSITION == 0:
                    cpd_inv_test = quest_suggest_cpd_inv(quest, cpd_inv_range)

                    dist_raw = dist_from_cpd_inv(
                        cpd_inv_test, sf_cpp_eff, screen_width_px, screen_width_m)

                    if dist_raw >= MAX_DIST * 0.95:
                        sf_cpp_eff *= 2
                        speed_eff  /= 2
                        cpd_inv_range = cpd_inv_range_from_dist_range(
                            MIN_DIST, MAX_DIST, sf_cpp_eff, screen_width_px, screen_width_m)
                        current_est = quest_mean_cpd_inv(quest, cpd_inv_range)
                        quest = make_quest(current_est, cpd_inv_range)
                        cpd_inv_test = quest_suggest_cpd_inv(quest, cpd_inv_range)
                        print(f"  [NEW RANGE] {cpd_inv_range[0]:.4f} ~ {cpd_inv_range[1]:.4f}")

                    elif dist_raw <= MIN_DIST * 1.05:
                        sf_cpp_eff /= 2
                        speed_eff  *= 2
                        cpd_inv_range = cpd_inv_range_from_dist_range(
                            MIN_DIST, MAX_DIST, sf_cpp_eff, screen_width_px, screen_width_m)
                        current_est = quest_mean_cpd_inv(quest, cpd_inv_range)
                        quest = make_quest(current_est, cpd_inv_range)
                        cpd_inv_test = quest_suggest_cpd_inv(quest, cpd_inv_range)
                        print(f"  [NEW RANGE] {cpd_inv_range[0]:.4f} ~ {cpd_inv_range[1]:.4f}")

                    cpd     = (-cpd_inv_test) ** 3
                    ppd_eff = cpd / sf_cpp_eff
                    dist    = float(np.clip(
                        dist_from_ppd(ppd_eff, screen_width_px, screen_width_m),
                        MIN_DIST, MAX_DIST))
                    platform.move_to(dist)

                test_interval = interval_seq[seq_idx % len(interval_seq)]
                seq_idx += 1

                renderer.set_gabor_direction(random.choice(['right', 'left', 'up', 'down']))

                # 2AFC trial
                while True:
                    if renderer.show_interval_cue(1, dist): aborted = True; break
                    c1 = contrast if test_interval == 1 else 0.0
                    if renderer.show_gabor(c1, dist, sf_cpp_eff, speed_eff, args.duration):
                        aborted = True; break
                    renderer.show_flat(dist, (0.2, 0.2, 0.2))
                    time.sleep(0.3)
                    if renderer.show_interval_cue(2, dist): aborted = True; break
                    c2 = contrast if test_interval == 2 else 0.0
                    if renderer.show_gabor(c2, dist, sf_cpp_eff, speed_eff, args.duration):
                        aborted = True; break
                    renderer.show_flat(dist, (0.2, 0.2, 0.2))

                    chosen, esc = renderer.wait_for_response(dist)
                    if esc: aborted = True; break
                    if chosen == 'repeat':
                        continue
                    break

                if aborted:
                    break

                correct = int(chosen == test_interval)
                quest.addResponse(correct, intensity=cpd_inv_test)

                rho_cpd, omega, _ = compute_spatiotemporal_frequency(
                    renderer.width, renderer.W, dist, sf_cpp_eff, speed_eff)

                est_cpd_inv = quest_mean_cpd_inv(quest, cpd_inv_range)
                est_dist    = dist_from_cpd_inv(
                    est_cpd_inv, sf_cpp_eff, screen_width_px, screen_width_m)

                result_str  = "✓ CORRECT" if correct else "✗ WRONG"
                print(f"  [TRAINING] Trial {trial_idx+1:2d}/{n_trials}  "
                      f"{result_str}  "
                      f"dist={dist:.3f}m  cpd_inv={cpd_inv_test:.4f}  "
                      f"int={test_interval}  chosen={chosen}  "
                      f"est={est_cpd_inv:.4f}({est_dist:.3f}m)  "
                      f"sf={rho_cpd:.3f}cpd  tf={omega:.2f}hz")

            if aborted:
                print("\n[ESC] Training aborted.")
                break

            est_cpd_inv = quest_mean_cpd_inv(quest, cpd_inv_range)
            est_dist    = dist_from_cpd_inv(
                est_cpd_inv, sf_cpp_eff, screen_width_px, screen_width_m)
            print(f"\n  [TRAINING] Condition done — "
                  f"est cpd_inv={est_cpd_inv:.5f}  dist={est_dist:.4f}m  "
                  f"SD={quest_sd(quest):.5f}  (not saved)")

            elapsed = time.perf_counter() - experiment_start
            esc = renderer.show_progress_screen(
                conditions_done=cond_idx + 1,
                conditions_total=total_conditions,
                elapsed_sec=elapsed, distance_m=dist, duration_auto=3.0)
            if esc:
                aborted = True
                print("\n[ESC] Training aborted at progress screen.")
                break

    finally:
        platform.cleanup()
        renderer.destroy()
        glfw.terminate()
        print("\nTraining session finished.")
        if aborted:
            print("(Aborted early.)")
        else:
            print("All training conditions complete. No data was saved.")


if __name__ == "__main__":
    main()