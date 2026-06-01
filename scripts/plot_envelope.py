#!/usr/bin/env python
"""Plot feasibility-envelope metrics from a torque/velocity/action CSV.

Pure-Python script (no Isaac Sim, no gymnasium). Reads the CSV produced by
eval_metrics.py, reconstructs tau/vel/act tensors, calls the three SATA metrics
from torque_loco.metrics, prints a summary, and saves a bar chart.

Usage (sata conda env):
    ~/miniconda3/envs/sata/bin/python scripts/plot_envelope.py \
        --csv results/metrics_sata.csv \
        --out results/envelope_sata.png
"""

import argparse
import csv
import os
import sys

# ---------------------------------------------------------------------------
# Add src/ to sys.path so torque_loco is importable without installation.
# ---------------------------------------------------------------------------
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

from torque_loco.metrics import sata_peak_torque, sata_energy_per_step, sata_mean_jerk  # noqa: E402

# ---------------------------------------------------------------------------
# Headless matplotlib (must be set before any other matplotlib import).
# ---------------------------------------------------------------------------
_MATPLOTLIB_OK = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MATPLOTLIB_OK = True
except ImportError:
    pass

import torch  # noqa: E402


# ---------------------------------------------------------------------------
# Reference thresholds.
# ---------------------------------------------------------------------------
SIM_CLIP_NM = 23.5   # DCMotor effort_limit in Go2SataEnvCfg (sim clip)
REAL_LIMIT_NM = 45.0  # Physical Unitree Go2 peak torque spec


def load_csv(path: str):
    """Read CSV produced by eval_metrics.py.

    Returns:
        tau  (T, J) float32 torch tensor
        vel  (T, J) float32 torch tensor
        act  (T, J) float32 torch tensor
    """
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    rows = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"CSV is empty: {path}")

    # Determine joint count from column names.
    tau_keys = sorted([k for k in rows[0] if k.startswith("tau_")], key=lambda k: int(k.split("_")[1]))
    vel_keys = sorted([k for k in rows[0] if k.startswith("vel_")], key=lambda k: int(k.split("_")[1]))
    act_keys = sorted([k for k in rows[0] if k.startswith("act_")], key=lambda k: int(k.split("_")[1]))

    T = len(rows)
    J = len(tau_keys)

    tau_list = [[float(r[k]) for k in tau_keys] for r in rows]
    vel_list = [[float(r[k]) for k in vel_keys] for r in rows]
    act_list = [[float(r[k]) for k in act_keys] for r in rows]

    tau = torch.tensor(tau_list, dtype=torch.float32)   # (T, J)
    vel = torch.tensor(vel_list, dtype=torch.float32)   # (T, J)
    act = torch.tensor(act_list, dtype=torch.float32)   # (T, J)

    print(f"[INFO] Loaded {T} steps, {J} joints from {path}")
    return tau, vel, act


def main():
    parser = argparse.ArgumentParser(description="Plot SATA feasibility envelope metrics.")
    parser.add_argument(
        "--csv",
        type=str,
        default=os.path.join(_REPO, "results", "metrics_sata.csv"),
        help="Input CSV file produced by eval_metrics.py.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(_REPO, "results", "envelope_sata.png"),
        help="Output PNG path.",
    )
    args = parser.parse_args()

    # --- load data ---
    tau, vel, act = load_csv(args.csv)

    # --- compute metrics (dt = physics dt = 0.005 s) ---
    dt = 0.005
    peak_torque = sata_peak_torque(tau)
    energy_per_step = sata_energy_per_step(tau, vel, dt)
    mean_jerk = sata_mean_jerk(act, dt)

    # --- print summary ---
    print()
    print("=" * 60)
    print("SATA Feasibility-Envelope Metrics")
    print("=" * 60)
    print(f"  Peak |torque|       : {peak_torque:.3f}  N·m")
    print(f"  Mean energy/step   : {energy_per_step:.6f} J")
    print(f"  Mean action jerk   : {mean_jerk:.4f}  (normalized/s)")
    print()
    # One-line comparison to envelope bounds.
    if peak_torque <= SIM_CLIP_NM:
        status = f"WITHIN sim clip ({SIM_CLIP_NM} N·m)"
    elif peak_torque <= REAL_LIMIT_NM:
        status = f"ABOVE sim clip ({SIM_CLIP_NM} N·m) but within real limit ({REAL_LIMIT_NM} N·m)"
    else:
        status = f"EXCEEDS real Go2 limit ({REAL_LIMIT_NM} N·m) — infeasible on hardware"
    print(f"  Envelope verdict   : {status}")
    print("=" * 60)
    print()

    # --- bar plot ---
    if not _MATPLOTLIB_OK:
        print("[WARN] matplotlib not available — skipping plot. Install with: pip install matplotlib")
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    fig.suptitle("SATA Feasibility-Envelope Metrics", fontsize=14, fontweight="bold")

    metric_values = [peak_torque, energy_per_step, mean_jerk]
    metric_labels = ["Peak |torque|\n(N·m)", "Mean energy/step\n(J)", "Mean action jerk\n(norm/s)"]
    bar_colors = ["steelblue", "seagreen", "darkorange"]

    for ax, val, label, color in zip(axes, metric_values, metric_labels, bar_colors):
        ax.bar([label], [val], color=color, width=0.4)
        ax.set_ylabel("Value")
        ax.set_title(label.replace("\n", " "))

    # Add reference lines on the peak-torque axis (axes[0]).
    ax_tau = axes[0]
    ax_tau.axhline(
        SIM_CLIP_NM,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Sim clip {SIM_CLIP_NM} N·m",
    )
    ax_tau.axhline(
        REAL_LIMIT_NM,
        color="darkred",
        linestyle="-.",
        linewidth=1.5,
        label=f"Real Go2 limit {REAL_LIMIT_NM} N·m",
    )
    ax_tau.legend(fontsize=8, loc="upper right")

    # Extend y-axis so reference lines are always visible even if bar is small.
    current_ylim = ax_tau.get_ylim()
    ax_tau.set_ylim(0, max(current_ylim[1], REAL_LIMIT_NM * 1.1))

    plt.tight_layout()

    out_path = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved plot: {out_path}")


if __name__ == "__main__":
    main()
