"""
Quest 2AFC Experiment — main logic
===================================
Imports ExperimentRenderer from gabor_renderer; contains only
Quest staircase, platform control, CSV I/O, and the trial loop.
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
INITIAL_DISTANCE      = 0.5
UNIT_PER_M            = 100 / 1.6
MIN_DIST, MAX_DIST    = 0.5, 2.1
MOA_CSV               = "MOA_results.csv"
QUEST_CSV             = "Quest_results.csv"
TRIALS_PER_CONDITION  = 20
QUEST_CONVERGE_THRESH = 0.005   # metres


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

def make_quest(start_val):
    return QuestHandler(
        startVal   = -start_val,
        startValSd = (MAX_DIST - MIN_DIST) / 4.0,
        pThreshold = 0.75,
        nTrials    = TRIALS_PER_CONDITION,
        beta=3.5, delta=0.01, gamma=0.5,
        grain=0.001, range=MAX_DIST - MIN_DIST,
        minVal=-MAX_DIST, maxVal=-MIN_DIST,
        method='quantile',
    )

def quest_suggest(q): return float(np.clip(-q._questNextIntensity, MIN_DIST, MAX_DIST))
def quest_mean(q):    return float(np.clip(-q.mean(),              MIN_DIST, MAX_DIST))
def quest_sd(q):      return float(q.sd())


# ==========================================================
# CSV helpers
# ==========================================================

def init_quest_csv(path):
    empty = not os.path.exists(path) or os.stat(path).st_size == 0
    fh = open(path, 'a', newline='')
    w  = csv.writer(fh)
    if empty:
        w.writerow([
            "name","color","speed_px_per_sec","contrast","mean_luminance",
            "spatial_frequency_cpp","diagonal_inch","visual_radius_deg",
            "trial_index","distance_m_tested","test_interval",
            "observer_response","correct","quest_estimate_m",
            "retinal_spatial_freq_cpd","temporal_freq_hz",
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
    # parser.add_argument("--name",                  default='YanchengCai')
    parser.add_argument("--name", default='Rafal Mantiuk')
    parser.add_argument("--colors",     nargs="+", default=["ach","rg","yv"])
    parser.add_argument("--ach_speeds", nargs="+", type=float, default=[80, 120, 180])
    parser.add_argument("--rg_speeds", nargs="+", type=float, default=[100, 150, 225])
    parser.add_argument("--yv_speeds", nargs="+", type=float, default=[400, 600, 900])
    parser.add_argument("--ach_luminance_list", nargs="+", type=float, default=[10, 50])
    parser.add_argument("--rg_luminance_list", nargs="+", type=float, default=[10, 50])
    parser.add_argument("--yv_luminance_list", nargs="+", type=float, default=[10, 50])
    parser.add_argument("--ach_spatial_frequency_cpp", type=float, default=0.25)
    parser.add_argument("--rg_spatial_frequency_cpp", type=float, default=0.2)
    parser.add_argument("--yv_spatial_frequency_cpp", type=float, default=0.05)
    parser.add_argument("--ach_contrast", type=float, default=0.2)
    parser.add_argument("--rg_contrast",  type=float, default=0.1) #0.14 max
    parser.add_argument("--yv_contrast",  type=float, default=0.92)
    parser.add_argument("--diagonal_inch",     type=float, default=27)
    parser.add_argument("--visual_radius_deg", type=float, default=2.0)
    parser.add_argument("--port",     default="/dev/ttyACM0")
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--monitor_index", type=int, default=1)
    args = parser.parse_args()

    # Build condition list
    conditions = [
        {"color": color, "speed": spd, "contrast": getattr(args, f"{color}_contrast"),
         "luminance": lum, "spatial_frequency": getattr(args, f"{color}_spatial_frequency_cpp")}
        for color in args.colors
        for spd in getattr(args, f"{color}_speeds")
        for lum in getattr(args, f"{color}_luminance_list")
    ]
    random.shuffle(conditions)

    # Resume: skip completed conditions
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
    )
    renderer.init_window()

    experiment_start = time.perf_counter()
    aborted          = False

    try:
        for cond_idx, cond in enumerate(conditions):
            color, speed   = cond['color'],  cond['speed']
            contrast, lum  = cond['contrast'], cond['luminance']
            sf_cpp         = cond['spatial_frequency']

            start_dist = float(np.clip(
                load_moa_distance('YanchengCai', color, speed, lum),
                MIN_DIST, MAX_DIST))

            print(f"\n{'='*60}")
            print(f"Condition {cond_idx+1}/{total_conditions}")
            print(f"  Color:{color}  Speed:{speed}px/s  Lum:{lum}cd/m²")
            print(f"  Contrast:{contrast}  Quest start:{start_dist:.3f}m")
            print('='*60)

            quest = make_quest(start_dist)

            # Pre-generate balanced interval sequence
            half = TRIALS_PER_CONDITION // 2
            interval_seq = [1]*half + [2]*half
            random.shuffle(interval_seq)
            seq_idx = 0

            prev_dist     = None
            last_step_mm  = 0.0
            converged     = False
            trial_idx     = 0

            for trial_idx in range(TRIALS_PER_CONDITION):
                dist = quest_suggest(quest)

                # Convergence early-stop
                if prev_dist is not None:
                    step = abs(dist - prev_dist)
                    last_step_mm = step * 1000
                    if step < QUEST_CONVERGE_THRESH:
                        print(f"  [CONVERGE] step {step*1000:.2f}mm at trial {trial_idx+1}")
                        converged = True
                        break
                prev_dist = dist

                test_interval = interval_seq[seq_idx % len(interval_seq)]
                seq_idx += 1

                # ---- Move platform & set colour ----
                dist = float(np.clip(dist, MIN_DIST, MAX_DIST))
                platform.move_to(dist)
                renderer.set_condition(color, lum)

                # ---- 2AFC trial ----
                if renderer.show_interval_cue(1, dist): aborted = True; break
                c1 = contrast if test_interval == 1 else 0.0
                if renderer.show_gabor(c1, dist, sf_cpp, speed, args.duration):
                    aborted = True
                    break
                renderer.show_flat(dist, (0.2, 0.2, 0.2))
                time.sleep(0.3)
                if renderer.show_interval_cue(2, dist): aborted = True; break
                c2 = contrast if test_interval == 2 else 0.0
                if renderer.show_gabor(c2, dist, sf_cpp, speed, args.duration):
                    aborted = True
                    break
                renderer.show_flat(dist, (0.2, 0.2, 0.2))
                chosen, esc = renderer.wait_for_response(dist)
                if esc: aborted = True; break

                correct = int(chosen == test_interval)
                clamped_dist = float(np.clip(dist, MIN_DIST, MAX_DIST))
                quest.addResponse(correct, intensity=-clamped_dist)

                rho_cpd, omega, _ = compute_spatiotemporal_frequency(
                    renderer.width, renderer.W, dist, sf_cpp, speed)

                writer.writerow([
                    args.name, color, speed, contrast, lum, sf_cpp,
                    args.diagonal_inch, args.visual_radius_deg,
                    trial_idx, round(dist,4), test_interval, chosen, correct,
                    round(quest_mean(quest),4), round(rho_cpd,4), round(omega,4),
                ])
                fh.flush()

                print(f"  Trial {trial_idx+1:2d}: dist={dist:.3f}m  "
                      f"int={test_interval}  chosen={chosen}  "
                      f"ok={bool(correct)}  "
                      f"est={quest_mean(quest):.3f}m  "
                      f"step={last_step_mm:.1f}mm"
                      f"sf={rho_cpd:.3f}cpd  tf={omega:.2f}hz")  # ← 加这行

            if aborted:
                print("\n[ESC] Aborted.")
                break

            if converged:
                print(f"  Converged after {trial_idx} trials "
                      f"(est={quest_mean(quest):.4f}m, SD={quest_sd(quest):.4f}m)")

            elapsed = time.perf_counter() - experiment_start
            esc = renderer.show_progress_screen(
                conditions_done=cond_idx + 1,
                conditions_total=total_conditions,
                elapsed_sec=elapsed, distance_m=dist, duration_auto=3.0)
            if esc:
                aborted = True
                print("\n[ESC] Aborted at progress screen.")
                break

            print(f"\n  Final estimate: {quest_mean(quest):.4f}m  "
                  f"(SD={quest_sd(quest):.4f}m)")

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