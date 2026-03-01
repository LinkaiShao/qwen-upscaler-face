#!/usr/bin/env python3
"""Plot loss distribution percentages over training steps.

Parses the training log and shows how much each loss term contributes
to the total (after flow-floor scaling) as a stacked area chart.

Usage:
    python -m qwen_upscaler_face_gan.plot_loss_distribution <log_file>
    python -m qwen_upscaler_face_gan.plot_loss_distribution <log_file> --window 50
"""

import argparse
import re
import sys

import matplotlib.pyplot as plt
import numpy as np


# Matches the per-step log line from train.py
# [P1] step 510/9318 flow=0.0076 id=0.0445(Y) anchor=0.0054 gan=5.4550 topk=0.0225 lgan=0.0001 rho=0.20 fscale=0.56 cos=0.9500 ...
STEP_RE = re.compile(
    r"\[P(\d)\] step (\d+)/\d+ "
    r"flow=([\d.]+) id=([\d.]+)\(([Yn])\) "
    r"anchor=([\d.]+) gan=([\d.]+) "
    r"topk=([\d.]+) "
    r"lgan=([\d.]+) rho=([\d.]+) "
    r"fscale=([\d.]+) "
    r"cos=([\d.]+)"
)


def parse_log(path):
    """Parse training log, return list of dicts per logged step."""
    rows = []
    with open(path) as f:
        for line in f:
            m = STEP_RE.search(line)
            if not m:
                continue
            rows.append({
                "phase": int(m.group(1)),
                "step": int(m.group(2)),
                "flow": float(m.group(3)),
                "id": float(m.group(4)),
                "id_active": m.group(5) == "Y",
                "anchor": float(m.group(6)),
                "gan_raw": float(m.group(7)),
                "topk": float(m.group(8)),
                "lgan": float(m.group(9)),
                "rho": float(m.group(10)),
                "fscale": float(m.group(11)),
                "cos": float(m.group(12)),
            })
    return rows


def compute_percentages(rows, lambda_id, lambda_anchor, lambda_topk):
    """Compute per-step weighted contribution percentages."""
    steps = []
    pcts = {"flow": [], "id": [], "anchor": [], "gan": [], "topk": []}

    for r in rows:
        fscale = r["fscale"]
        flow_c = r["flow"]
        id_c = lambda_id * r["id"] * fscale
        anchor_c = lambda_anchor * r["anchor"] * fscale
        gan_c = r["lgan"] * r["gan_raw"] * fscale
        topk_c = lambda_topk * r["topk"] * fscale

        total = flow_c + id_c + anchor_c + gan_c + topk_c
        if total < 1e-10:
            continue

        steps.append(r["step"])
        pcts["flow"].append(flow_c / total * 100)
        pcts["id"].append(id_c / total * 100)
        pcts["anchor"].append(anchor_c / total * 100)
        pcts["gan"].append(gan_c / total * 100)
        pcts["topk"].append(topk_c / total * 100)

    return steps, pcts


def smooth(arr, window):
    """Simple moving average."""
    if window <= 1:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def main():
    parser = argparse.ArgumentParser(description="Plot loss distribution over training")
    parser.add_argument("log_file", help="Path to training log file")
    parser.add_argument("--window", type=int, default=20, help="Smoothing window (default: 20)")
    parser.add_argument("--lambda_id", type=float, default=0.15)
    parser.add_argument("--lambda_anchor", type=float, default=0.1)
    parser.add_argument("--lambda_topk", type=float, default=1.0)
    parser.add_argument("--save", type=str, default=None, help="Save plot to file instead of showing")
    cli = parser.parse_args()

    rows = parse_log(cli.log_file)
    if not rows:
        print(f"No training steps found in {cli.log_file}")
        sys.exit(1)

    print(f"Parsed {len(rows)} logged steps (step {rows[0]['step']}–{rows[-1]['step']})")

    steps, pcts = compute_percentages(rows, cli.lambda_id, cli.lambda_anchor, cli.lambda_topk)
    steps = np.array(steps)

    # Smooth
    w = cli.window
    flow_s = smooth(np.array(pcts["flow"]), w)
    id_s = smooth(np.array(pcts["id"]), w)
    anchor_s = smooth(np.array(pcts["anchor"]), w)
    gan_s = smooth(np.array(pcts["gan"]), w)
    topk_s = smooth(np.array(pcts["topk"]), w)

    # --- Stacked area plot ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])
    fig.suptitle("Loss Distribution Over Training", fontsize=14, fontweight="bold")

    colors = {
        "flow": "#2196F3",
        "topk": "#FF9800",
        "gan": "#E91E63",
        "id": "#4CAF50",
        "anchor": "#9E9E9E",
    }

    ax1.stackplot(
        steps, flow_s, topk_s, gan_s, id_s, anchor_s,
        labels=["flow", "top-k MSE", "GAN", "face id", "anchor"],
        colors=[colors["flow"], colors["topk"], colors["gan"], colors["id"], colors["anchor"]],
        alpha=0.85,
    )
    ax1.axhline(30, color="white", linestyle="--", linewidth=1.5, alpha=0.7, label="30% floor")
    ax1.set_ylabel("% of total loss")
    ax1.set_ylim(0, 100)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # Phase boundaries
    phases_seen = set()
    for r in rows:
        if r["phase"] not in phases_seen:
            phases_seen.add(r["phase"])
            if r["phase"] > 0:
                ax1.axvline(r["step"], color="white", linestyle="-", linewidth=2, alpha=0.6)
                ax1.text(r["step"] + 5, 95, f"Phase {r['phase']}",
                         fontsize=10, color="white", fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5))

    # --- Bottom panel: flow_scale and cos_sim ---
    fscales = smooth(np.array([r["fscale"] for r in rows]), w)
    cos_sims = smooth(np.array([r["cos"] for r in rows]), w)

    ax2.plot(steps, fscales, color="#FF5722", linewidth=1.5, label="flow_scale", alpha=0.8)
    ax2.plot(steps, cos_sims, color="#4CAF50", linewidth=1.5, label="cos_sim", alpha=0.8)
    ax2.axhline(0.95, color="#4CAF50", linestyle=":", linewidth=1, alpha=0.5)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Value")
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    for r in rows:
        if r["phase"] > 0 and r["step"] == min(rr["step"] for rr in rows if rr["phase"] == r["phase"]):
            ax2.axvline(r["step"], color="gray", linestyle="-", linewidth=1, alpha=0.4)

    plt.tight_layout()

    if cli.save:
        plt.savefig(cli.save, dpi=150, bbox_inches="tight")
        print(f"Saved to {cli.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
