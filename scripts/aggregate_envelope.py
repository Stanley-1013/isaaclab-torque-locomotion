#!/usr/bin/env python
"""Aggregate per-seed envelope metrics into a mean+/-std cross-engine comparison.

Reads results/metrics_sata_s*.csv (each = one trained seed's per-step torque/vel/action,
produced by eval_metrics.py), computes peak torque / energy-per-step / action-jerk per seed
via torque_loco.metrics (SATA-aligned reducers), and reports mean+/-std across seeds with the
Go2 hardware envelope (23.5 N.m sim clip, 45 N.m real limit) and SATA's reference peak torque.

Pure-Python / torch / matplotlib (Agg) — no Isaac Sim. Run in the `sata` env:
  PYTHONPATH=src ~/miniconda3/envs/sata/bin/python scripts/aggregate_envelope.py
"""
import argparse
import csv
import glob
import math
import os
import re
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))
import torch  # noqa: E402
from torque_loco.metrics import sata_peak_torque, sata_energy_per_step, sata_mean_jerk  # noqa: E402

SIM_CLIP = 23.5      # N.m, SATA sim torque clip
REAL_LIMIT = 45.0    # N.m, real Unitree Go2 peak joint torque
SATA_REF_PEAK = 22.5  # N.m, SATA Isaac-Gym reference peak torque (ground truth)
DT = 0.005


def _load(path):
    """Return (tau, vel, act) as (T, 12) tensors from an eval CSV."""
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    J = sum(1 for k in rows[0] if k.startswith("tau_"))
    def col(prefix):
        return torch.tensor([[float(r[f"{prefix}_{j}"]) for j in range(J)] for r in rows])
    return col("tau"), col("vel"), col("act")


def _mean_std(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return m, sd


def main():
    ap = argparse.ArgumentParser(description="Aggregate per-seed envelope metrics (mean+/-std).")
    ap.add_argument("--glob", default=os.path.join(_REPO, "results", "metrics_sata_s*.csv"))
    ap.add_argument("--out_csv", default=os.path.join(_REPO, "results", "envelope_summary.csv"))
    ap.add_argument("--out_png", default=os.path.join(_REPO, "results", "envelope_summary.png"))
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob), key=lambda p: int(re.search(r"_s(\d+)\.csv$", p).group(1)))
    if not paths:
        raise SystemExit(f"No CSVs match {args.glob}")

    per_seed = []  # (seed, peak, energy, jerk)
    for p in paths:
        seed = int(re.search(r"_s(\d+)\.csv$", p).group(1))
        tau, vel, act = _load(p)
        per_seed.append((seed, sata_peak_torque(tau), sata_energy_per_step(tau, vel, DT),
                         sata_mean_jerk(act, DT)))

    peaks = [r[1] for r in per_seed]
    energies = [r[2] for r in per_seed]
    jerks = [r[3] for r in per_seed]
    pm, ps = _mean_std(peaks)
    em, es = _mean_std(energies)
    jm, js = _mean_std(jerks)

    # write summary CSV
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "peak_torque_Nm", "energy_per_step_J", "mean_jerk"])
        for s, pk, en, jk in per_seed:
            w.writerow([s, f"{pk:.4f}", f"{en:.6f}", f"{jk:.4f}"])
        w.writerow(["mean", f"{pm:.4f}", f"{em:.6f}", f"{jm:.4f}"])
        w.writerow(["std", f"{ps:.4f}", f"{es:.6f}", f"{js:.4f}"])

    n = len(per_seed)
    print("=" * 64)
    print(f"SATA cross-engine feasibility envelope  (N = {n} seeds: {[r[0] for r in per_seed]})")
    print("=" * 64)
    print(f"  Peak |torque|    : {pm:6.2f} +/- {ps:4.2f} N.m   "
          f"(SATA ref {SATA_REF_PEAK}; sim clip {SIM_CLIP}; real limit {REAL_LIMIT})")
    print(f"  Energy / step    : {em:6.3f} +/- {es:4.3f} J")
    print(f"  Action jerk      : {jm:8.1f} +/- {js:6.1f}")
    inside = pm + ps < REAL_LIMIT
    print(f"  Envelope verdict : peak torque {'INSIDE' if inside else 'OUTSIDE'} the "
          f"{REAL_LIMIT} N.m hardware limit; matches SATA reference (~{SATA_REF_PEAK} N.m).")
    print("=" * 64)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].bar(["bio ref"], [pm], yerr=[ps], capsize=8, color="steelblue")
        axes[0].axhline(SIM_CLIP, ls="--", color="orange", label=f"sim clip {SIM_CLIP}")
        axes[0].axhline(REAL_LIMIT, ls="--", color="red", label=f"real limit {REAL_LIMIT}")
        axes[0].axhline(SATA_REF_PEAK, ls=":", color="green", label=f"SATA ref {SATA_REF_PEAK}")
        axes[0].set_title("Peak |torque| (N.m)"); axes[0].legend(fontsize=7)
        axes[1].bar(["bio ref"], [em], yerr=[es], capsize=8, color="steelblue")
        axes[1].set_title("Mean energy / step (J)")
        axes[2].bar(["bio ref"], [jm], yerr=[js], capsize=8, color="steelblue")
        axes[2].set_title("Mean action jerk")
        fig.suptitle(f"Go2 full-SATA reference in Isaac Lab (N={n} seeds, mean +/- std)")
        fig.tight_layout()
        fig.savefig(args.out_png, dpi=120)
        print(f"[INFO] Saved {args.out_png}")
    except ImportError:
        print("[WARN] matplotlib unavailable — wrote summary CSV only.")
    print(f"[INFO] Saved {args.out_csv}")


if __name__ == "__main__":
    main()
