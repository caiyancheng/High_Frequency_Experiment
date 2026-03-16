"""
Quest 2AFC Experiment — main logic
===================================
Quest operates in cpd_inv space:
    cpd     = ppd_at_dist * sf_cpp          (actual spatial frequency in cpd)
    cpd_inv = -(cpd ^ (1/3))               (cube-root compressed, sign-flipped)

Inverses:
    cpd     = (-cpd_inv) ^ 3
    ppd     = cpd / sf_cpp
    dist    = screen_width_m / (2 * tan(radians(screen_width_px / (2*ppd))))
"""

import argparse
import csv
import os
import random
import time
from collections import Counter

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
INITIAL_DISTANCE      = 1
UNIT_PER_M            = 100 / 1.6
MIN_DIST, MAX_DIST    = 1, 2.6
MOA_CSV               = "MOA_results.csv"
QUEST_CSV             = "Quest_results.csv"
TRIALS_PER_CONDITION  = 30
QUEST_CONVERGE_THRESH = 0.005   # in cpd_inv units
REPEATS_PER_POSITION  = 3       # trials collected at same position before new Quest suggestion

# ==========================================================
# Geometry & cpd_inv conversion  (sf_cpp is per-condition)
# ==========================================================

def ppd_from_dist(dist_m, screen_width_px, screen_width_m):
    """Pixels per degree at a given viewing distance."""
    half_fov_deg = np.degrees(np.arctan(screen_width_m / (2.0 * dist_m)))
    return screen_width_px / (2.0 * half_fov_deg)


def dist_from_ppd(ppd, screen_width_px, screen_width_m):
    """Viewing distance (m) that yields the given ppd."""
    half_fov_deg = screen_width_px / (2.0 * ppd)
    return screen_width_m / (2.0 * np.tan(np.radians(half_fov_deg)))


def cpd_inv_from_dist(dist_m, sf_cpp, screen_width_px, screen_width_m):
    """distance → cpd_inv for a given spatial frequency sf_cpp (cycles/pixel)."""
    ppd     = ppd_from_dist(dist_m, screen_width_px, screen_width_m)
    cpd     = ppd * sf_cpp
    cpd_inv = -(cpd ** (1.0 / 3.0))
    return float(cpd_inv)


def dist_from_cpd_inv(cpd_inv, sf_cpp, screen_width_px, screen_width_m):
    """cpd_inv → distance (m) for a given spatial frequency sf_cpp."""
    cpd  = (-cpd_inv) ** 3.0
    ppd  = cpd / sf_cpp
    return float(dist_from_ppd(ppd, screen_width_px, screen_width_m))


def cpd_inv_range_from_dist_range(min_dist, max_dist, sf_cpp,
                                   screen_width_px, screen_width_m):
    """
    Returns (cpd_inv_near, cpd_inv_far).
    Near (small dist) → high ppd → high cpd → less negative cpd_inv.
    Far  (large dist) → low  ppd → low  cpd → more negative cpd_inv.
    """
    cpd_inv_near = cpd_inv_from_dist(min_dist, sf_cpp, screen_width_px, screen_width_m)
    cpd_inv_far  = cpd_inv_from_dist(max_dist, sf_cpp, screen_width_px, screen_width_m)
    return (cpd_inv_near, cpd_inv_far)   # e.g. (-0.8, -2.1)


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
# Quest helpers  —  all in cpd_inv space
# ==========================================================

def make_quest(start_cpd_inv, cpd_inv_range):
    """
    QuestHandler in cpd_inv space.
    cpd_inv_range = (cpd_inv_near, cpd_inv_far),  near > far (less negative → more negative).
    """
    lo, hi = min(cpd_inv_range), max(cpd_inv_range)
    span   = hi - lo
    return QuestHandler(
        startVal   = start_cpd_inv,
        startValSd = span / 2.0,
        pThreshold = 0.75,
        nTrials    = TRIALS_PER_CONDITION,
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
# CSV helpers
# ==========================================================

def init_quest_csv(path):
    empty = not os.path.exists(path) or os.stat(path).st_size == 0
    fh = open(path, 'a', newline='')
    w  = csv.writer(fh)
    if empty:
        w.writerow([
            "name", "color", "speed_px_per_sec", "contrast", "mean_luminance",
            "spatial_frequency_cpp", "sf_cpp_eff", "speed_eff", "diagonal_inch", "visual_radius_deg",
            "trial_index", "distance_m_tested", "cpd_inv_tested",
            "test_interval", "observer_response", "correct",
            "quest_estimate_cpd_inv", "quest_estimate_m", "quest_sd_cpd_inv",
            "retinal_spatial_freq_cpd", "temporal_freq_hz",
        ])
    return fh, w


def load_moa_distance(name, color, speed, luminance):
    if not os.path.exists(MOA_CSV):
        return 1.0
    distances = []
    with open(MOA_CSV) as f:
        for row in csv.DictReader(f):
            if (row.get('name') == name and row.get('color') == color
                    and abs(float(row.get('speed_px_per_sec', -1)) - speed) < 1e-3
                    and abs(float(row.get('mean_luminance', -1)) - luminance) < 1e-3):
                distances.append(float(row['distance_m']))
    return float(np.mean(distances)) if distances else 1.0


def load_completed_conditions(name, csv_path=QUEST_CSV):
    completed, counts = set(), Counter()
    if not os.path.exists(csv_path):
        return completed
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get('name') != name:
                continue
            counts[(row['color'],
                    float(row['speed_px_per_sec']),
                    float(row['mean_luminance']))] += 1
    return {k for k, v in counts.items() if v >= TRIALS_PER_CONDITION}


# ==========================================================
# Main
# ==========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default='YanchengCai')
    # parser.add_argument("--name", default='Leyi Yao')
    # parser.add_argument("--name", default='Yaru Liu')
    # parser.add_argument("--name", default='Mengqing Huang')
    # parser.add_argument("--name", default='Shuqi Lou')
    # parser.add_argument("--name", default='Rafal Mantiuk')
    # parser.add_argument("--name", default='YifanDing')
    # parser.add_argument("--name", default='LinShen')
    parser.add_argument("--colors", nargs="+", default=["ach", "rg", "yv"])
    # 10, 15, 20, 30, 45
    parser.add_argument("--ach_speeds", nargs="+", type=float, default=[40, 60, 80, 120, 180])
    parser.add_argument("--rg_speeds", nargs="+", type=float, default=[50, 75, 100, 150])
    parser.add_argument("--yv_speeds", nargs="+", type=float, default=[400, 600])
    parser.add_argument("--ach_luminance_list", nargs="+", type=float, default=[50])
    parser.add_argument("--rg_luminance_list", nargs="+", type=float, default=[50])
    parser.add_argument("--yv_luminance_list", nargs="+", type=float, default=[50])
    parser.add_argument("--ach_spatial_frequency_cpp", type=float, default=0.25)
    parser.add_argument("--rg_spatial_frequency_cpp", type=float, default=0.2)
    parser.add_argument("--yv_spatial_frequency_cpp", type=float, default=0.05)

    parser.add_argument("--ach_contrast", type=float, default=0.9)
    parser.add_argument("--rg_contrast",  type=float, default=0.14)
    parser.add_argument("--yv_contrast",  type=float, default=0.92)

    parser.add_argument("--diagonal_inch",     type=float, default=27)
    parser.add_argument("--visual_radius_deg", type=float, default=2.0)
    parser.add_argument("--port",     default="/dev/ttyACM0")
    parser.add_argument("--duration", type=float, default=1.0)   # 1 s per interval
    parser.add_argument("--monitor_index", type=int, default=1)
    args = parser.parse_args()

    SKIP_CONDITIONS = {
        # ("yv", 600,  5),
        # ("yv", 200, 50),
        # ("rg", 225,  5),
    }

    conditions = []
    for color in args.colors:
        contrast = getattr(args, f"{color}_contrast")
        sf_cpp   = getattr(args, f"{color}_spatial_frequency_cpp")
        for spd in getattr(args, f"{color}_speeds"):
            for lum in getattr(args, f"{color}_luminance_list"):
                if (color, spd, lum) in SKIP_CONDITIONS:
                    print(f"  [SKIP] {color} speed={spd} lum={lum}")
                    continue
                conditions.append({
                    "color": color, "speed": spd, "contrast": contrast,
                    "luminance": lum, "spatial_frequency": sf_cpp,
                })
    random.shuffle(conditions)

    completed = load_completed_conditions(args.name)
    if completed:
        before     = len(conditions)
        conditions = [c for c in conditions
                      if (c['color'], c['speed'], c['luminance']) not in completed]
        print(f"[RESUME] Skipping {before - len(conditions)} completed condition(s).")

    total_conditions = len(conditions)
    fh, writer       = init_quest_csv(QUEST_CSV)
    platform         = PlatformController(args.port)

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
            color, speed   = cond['color'],  cond['speed']
            contrast, lum  = cond['contrast'], cond['luminance']
            sf_cpp         = cond['spatial_frequency']

            # cpd_inv range is condition-specific (depends on sf_cpp)
            cpd_inv_range = cpd_inv_range_from_dist_range(
                MIN_DIST, MAX_DIST, sf_cpp, screen_width_px, screen_width_m)

            # Starting point: MOA distance → cpd_inv
            # start_dist    = float(np.clip(
            #     load_moa_distance(args.name, color, speed, lum),
            #     MIN_DIST, MAX_DIST))
            start_dist = float(np.clip(
                load_moa_distance('YanchengCai', color, speed, lum),
                MIN_DIST, MAX_DIST))
            start_cpd_inv = cpd_inv_from_dist(
                start_dist, sf_cpp, screen_width_px, screen_width_m)

            print(f"\n{'='*60}")
            print(f"Condition {cond_idx+1}/{total_conditions}")
            print(f"  Color:{color}  Speed:{speed}px/s  Lum:{lum}cd/m²  sf_cpp:{sf_cpp}")
            print(f"  cpd_inv range: [{cpd_inv_range[0]:.4f}, {cpd_inv_range[1]:.4f}]")
            print(f"  MOA dist:{start_dist:.3f}m  start_cpd_inv:{start_cpd_inv:.4f}")
            print('='*60)

            quest = make_quest(start_cpd_inv, cpd_inv_range)

            half = (TRIALS_PER_CONDITION + 10) // 2
            interval_seq = [1] * half + [2] * half
            random.shuffle(interval_seq)
            seq_idx = 0

            # prev_cpd_inv = None
            last_step    = 0.0
            converged    = False
            dist         = start_dist   # fallback for progress screen
            cpd_inv_test = start_cpd_inv
            sf_cpp_eff = sf_cpp
            speed_eff = speed
            renderer.set_condition(color, lum)
            max_trials = TRIALS_PER_CONDITION
            trial_idx = 0
            while trial_idx < max_trials:
                if trial_idx % REPEATS_PER_POSITION == 0:
                    # ---- Quest suggestion in cpd_inv space ----
                    cpd_inv_test = quest_suggest_cpd_inv(quest, cpd_inv_range)

                    # Convergence check in cpd_inv space
                    # if prev_cpd_inv is not None:
                    #     step = abs(cpd_inv_test - prev_cpd_inv)
                    #     last_step = step
                    #     if step < QUEST_CONVERGE_THRESH:
                    #         print(f"  [CONVERGE] step={step:.5f} at trial {trial_idx + 1}")
                    #         converged = True
                    #         break
                    # prev_cpd_inv = cpd_inv_test

                    # # ---- cpd_inv → distance for platform ----
                    # dist = float(np.clip(
                    #     dist_from_cpd_inv(cpd_inv_test, sf_cpp, screen_width_px, screen_width_m),
                    #     MIN_DIST, MAX_DIST))
                    # --- boundary adaptation ---
                    dist_raw = dist_from_cpd_inv(cpd_inv_test, sf_cpp_eff, screen_width_px, screen_width_m)

                    if dist_raw >= MAX_DIST * 0.95:
                        sf_cpp_eff *= 2
                        speed_eff /= 2
                        cpd_inv_range = cpd_inv_range_from_dist_range(
                            MIN_DIST, MAX_DIST, sf_cpp_eff, screen_width_px, screen_width_m)
                        current_est = quest_mean_cpd_inv(quest, cpd_inv_range)
                        quest = make_quest(current_est, cpd_inv_range)
                        cpd_inv_test = quest_suggest_cpd_inv(quest, cpd_inv_range)  # ← 加这行
                        print(f"  [NEW RANGE] {cpd_inv_range[0]:.4f} ~ {cpd_inv_range[1]:.4f}")

                    elif dist_raw <= MIN_DIST * 1.05:
                        sf_cpp_eff /= 2
                        speed_eff *= 2
                        cpd_inv_range = cpd_inv_range_from_dist_range(
                            MIN_DIST, MAX_DIST, sf_cpp_eff, screen_width_px, screen_width_m)
                        current_est = quest_mean_cpd_inv(quest, cpd_inv_range)
                        quest = make_quest(current_est, cpd_inv_range)
                        cpd_inv_test = quest_suggest_cpd_inv(quest, cpd_inv_range)  # ← 加这行
                        print(f"  [NEW RANGE] {cpd_inv_range[0]:.4f} ~ {cpd_inv_range[1]:.4f}")

                    # 此时 cpd_inv_test 已是新范围内的合理值
                    cpd = (-cpd_inv_test) ** 3
                    ppd_eff = cpd / sf_cpp_eff
                    dist = float(np.clip(
                        dist_from_ppd(ppd_eff, screen_width_px, screen_width_m),
                        MIN_DIST, MAX_DIST))
                    platform.move_to(dist)

                test_interval = interval_seq[seq_idx % len(interval_seq)]
                seq_idx += 1


                renderer.set_gabor_direction(random.choice(['right', 'left', 'up', 'down']))
                # ---- 2AFC trial (1 s per interval) ----
                while True:
                    if renderer.show_interval_cue(1, dist): aborted = True; break
                    c1 = contrast if test_interval == 1 else 0.0
                    if renderer.show_gabor(c1, dist, sf_cpp_eff, speed_eff, args.duration):
                        aborted = True
                        break
                    renderer.show_flat(dist, (0.2, 0.2, 0.2))
                    time.sleep(0.3)
                    if renderer.show_interval_cue(2, dist): aborted = True; break
                    c2 = contrast if test_interval == 2 else 0.0
                    if renderer.show_gabor(c2, dist, sf_cpp_eff, speed_eff, args.duration):
                        aborted = True
                        break
                    renderer.show_flat(dist, (0.2, 0.2, 0.2))

                    chosen, esc = renderer.wait_for_response(dist)
                    if esc: aborted = True; break

                    if chosen == 'repeat':  # 上键返回 'repeat'
                        continue
                    break  # 左/右键正常退出循环
                if aborted:
                    break
                correct = int(chosen == test_interval)

                # ---- Update Quest in cpd_inv space ----
                quest.addResponse(correct, intensity=cpd_inv_test)

                # ---- 收敛检查（只在达到原定trials上限时触发一次扩展）----
                if trial_idx + 1 == TRIALS_PER_CONDITION and quest_sd(quest) > QUEST_CONVERGE_THRESH:
                    max_trials = TRIALS_PER_CONDITION + 5
                    print(f"  [EXTEND] SD={quest_sd(quest):.5f} > thresh, adding 10 trials → max={max_trials}")

                trial_idx += 1  # ← while循环手动递增

                # ---- Logging ----
                rho_cpd, omega, _ = compute_spatiotemporal_frequency(
                    renderer.width, renderer.W, dist, sf_cpp_eff, speed_eff)

                est_cpd_inv = quest_mean_cpd_inv(quest, cpd_inv_range)
                est_dist = dist_from_cpd_inv(est_cpd_inv, sf_cpp_eff, screen_width_px, screen_width_m)

                writer.writerow([
                    args.name, color, speed, contrast, lum, sf_cpp, sf_cpp_eff, speed_eff,
                    args.diagonal_inch, args.visual_radius_deg,
                    trial_idx,
                    round(dist, 4), round(cpd_inv_test, 5),
                    test_interval, chosen, correct,
                    round(est_cpd_inv, 5), round(est_dist, 4),
                    round(quest_sd(quest), 5),
                    round(rho_cpd, 4), round(omega, 4),
                ])
                fh.flush()

                print(f"  Trial {trial_idx:2d}: "
                      f"dist={dist:.3f}m  cpd_inv={cpd_inv_test:.4f}  "
                      f"int={test_interval}  chosen={chosen}  ok={bool(correct)}  "
                      f"est={est_cpd_inv:.4f}({est_dist:.3f}m)  "
                      f"step={last_step:.5f}  "
                      f"sf={rho_cpd:.3f}cpd  tf={omega:.2f}hz")

            if aborted:
                print("\n[ESC] Aborted.")
                break

            if converged:
                print(f"  Converged after {trial_idx} trials.")

            est_cpd_inv = quest_mean_cpd_inv(quest, cpd_inv_range)
            est_dist = dist_from_cpd_inv(est_cpd_inv, sf_cpp_eff, screen_width_px, screen_width_m)
            print(f"\n  Final estimate: cpd_inv={est_cpd_inv:.5f}  "
                  f"dist={est_dist:.4f}m  SD={quest_sd(quest):.5f}")

            elapsed = time.perf_counter() - experiment_start
            esc = renderer.show_progress_screen(
                conditions_done=cond_idx + 1,
                conditions_total=total_conditions,
                elapsed_sec=elapsed, distance_m=dist, duration_auto=3.0)
            if esc:
                aborted = True
                print("\n[ESC] Aborted at progress screen.")
                break

    finally:
        platform.cleanup()
        fh.close()
        renderer.destroy()
        glfw.terminate()
        print("\nExperiment finished safely.")
        if aborted:
            print("(Aborted early — partial data saved.)")


if __name__ == "__main__":
    main()