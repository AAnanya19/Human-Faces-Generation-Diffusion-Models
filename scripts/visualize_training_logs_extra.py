#!/usr/bin/env python3
"""
Generate supplementary publication-quality training log visualizations (Figures 6–10)
for IEEE conference report on DDPM diffusion models.

Figures produced:
  6. Training Stability (Rolling Loss Std)
  7. FID vs Training Loss Scatter
  8. Convergence Speed (Epochs to FID < 60) — horizontal bar chart
  9. FID Improvement Rate (Marginal Gains)
 10. Noise Schedule Comparison (Linear vs Cosine) — 2-panel

All outputs are 300 DPI PNGs sized for IEEE column widths.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_ROOT / "results" / "celebahq" / "eval" / "logs"
FIG_DIR = REPO_ROOT / "results" / "celebahq" / "eval" / "figures"

# Config metadata: (label, filename, color, linestyle, marker)
CONFIGS = [
    ("Baseline",           "training_log_baseline.csv",                   "#0072B2", "-",  "o"),
    ("Cosine LR",          "training_log_cosineLR.csv",                   "#E69F00", "--", "s"),
    ("EMA",                "training_log_ema.csv",                        "#009E73", "-.", "D"),
    ("Cosine β Schedule", "training_log_cosine_diffusion_scheduler.csv", "#CC79A7", ":",  "^"),
]

DPI = 300
SINGLE_COL_W = 3.5   # inches – IEEE single-column width
DOUBLE_COL_W = 7.16  # inches – IEEE double-column width


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_rc():
    """Set matplotlib rcParams for IEEE-friendly appearance."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "serif"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.3,
        "lines.markersize": 5,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "figure.dpi": 150,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.facecolor": "#FAFAFA",
        "figure.facecolor": "white",
        "mathtext.fontset": "stix",
    })


def load_csv(filename: str) -> pd.DataFrame:
    """Load a training log CSV and deduplicate by keeping last occurrence per epoch."""
    df = pd.read_csv(LOG_DIR / filename)
    df = df.drop_duplicates(subset="epoch", keep="last").sort_values("epoch").reset_index(drop=True)
    return df


def save_fig(fig, name: str):
    path = FIG_DIR / name
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 6 – Training Stability (Rolling Loss Std)
# ---------------------------------------------------------------------------

def fig_training_stability(data: dict[str, pd.DataFrame]):
    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 2.8))

    for label, fname, color, ls, _ in CONFIGS:
        df = data[fname]
        rolling_std = df["avg_loss"].rolling(window=50, center=True, min_periods=10).std()
        ax.plot(df["epoch"], rolling_std, color=color, ls=ls, label=label, linewidth=1.3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Rolling Std of Loss")
    ax.set_title("Training Stability (Rolling Loss Standard Deviation)")
    ax.set_xlim(0, None)
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="0.8")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
    fig.tight_layout()
    save_fig(fig, "training_stability_rolling_std.png")


# ---------------------------------------------------------------------------
# Figure 7 – FID vs Training Loss Scatter
# ---------------------------------------------------------------------------

def fig_fid_vs_loss_scatter(data: dict[str, pd.DataFrame]):
    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 3.0))

    points = []
    for label, fname, color, _, marker in CONFIGS:
        df = data[fname]
        final_loss = df["avg_loss"].tail(50).mean()
        fid_rows = df.dropna(subset=["fid"])
        best_fid = fid_rows["fid"].min() if not fid_rows.empty else float("nan")
        points.append((label, final_loss, best_fid, color, marker))

    for label, loss, fid, color, marker in points:
        ax.scatter(loss, fid, color=color, marker=marker, s=80, zorder=5,
                   edgecolors="black", linewidths=0.5)
        ax.annotate(label, xy=(loss, fid), xytext=(8, 4),
                    textcoords="offset points", fontsize=8.5, color=color,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.7))

    ax.set_xlabel("Final Average Loss (last 50 epochs)")
    ax.set_ylabel("Best FID (lower is better)")
    ax.set_title("Final Loss vs. Best FID")
    fig.tight_layout()
    save_fig(fig, "fid_vs_loss_scatter.png")


# ---------------------------------------------------------------------------
# Figure 8 – Convergence Speed (Epochs to FID < 60)
# ---------------------------------------------------------------------------

def fig_convergence_speed(data: dict[str, pd.DataFrame]):
    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 2.4))

    results = []
    for label, fname, color, _, _ in CONFIGS:
        df = data[fname]
        fid_rows = df.dropna(subset=["fid"])
        below_60 = fid_rows[fid_rows["fid"] < 60]
        if below_60.empty:
            epochs_to_60 = None
        else:
            epochs_to_60 = int(below_60["epoch"].iloc[0])
        results.append((label, epochs_to_60, color))

    results.sort(key=lambda x: x[1] if x[1] is not None else 9999)

    labels = [r[0] for r in results]
    values = [r[1] if r[1] is not None else 0 for r in results]
    colors = [r[2] for r in results]
    never_reached = [r[1] is None for r in results]

    bars = ax.barh(labels, values, color=colors, edgecolor="black", linewidth=0.5, height=0.55)

    for i, (bar, val, never) in enumerate(zip(bars, values, never_reached)):
        if never:
            ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                    "Never", va="center", ha="left", fontsize=8, fontstyle="italic")
        else:
            ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                    f"Epoch {val}", va="center", ha="left", fontsize=8, fontweight="bold")

    max_val = max(v for v in values if v > 0) if any(v > 0 for v in values) else 100
    ax.set_xlim(0, max_val * 1.35)
    ax.set_xlabel("Epochs to Reach FID < 60")
    ax.set_title("Convergence Speed: Epochs to Reach FID < 60")
    ax.grid(axis="y", alpha=0)
    fig.tight_layout()
    save_fig(fig, "convergence_speed_bar.png")


# ---------------------------------------------------------------------------
# Figure 9 – FID Improvement Rate (Marginal Gains)
# ---------------------------------------------------------------------------

def fig_fid_improvement_rate(data: dict[str, pd.DataFrame]):
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W, 3.2))

    for label, fname, color, ls, marker in CONFIGS:
        df = data[fname]
        fid_rows = df.dropna(subset=["fid"]).copy()
        if len(fid_rows) < 2:
            continue
        fid_rows = fid_rows.sort_values("epoch").reset_index(drop=True)
        delta_fid = fid_rows["fid"].shift(1) - fid_rows["fid"]
        ax.plot(fid_rows["epoch"].iloc[1:], delta_fid.iloc[1:],
                color=color, ls=ls, marker=marker, markersize=5,
                label=label, linewidth=1.3)

    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ΔFID (positive = improvement)")
    ax.set_title("FID Improvement per Evaluation Interval")
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="0.8")
    fig.tight_layout()
    save_fig(fig, "fid_improvement_rate.png")


# ---------------------------------------------------------------------------
# Figure 10 – Noise Schedule Comparison (Linear vs Cosine)
# ---------------------------------------------------------------------------

def fig_noise_schedule_comparison():
    T = 1000
    t = np.arange(T)

    # Linear schedule
    beta_start, beta_end = 1e-4, 0.02
    beta_linear = beta_start + t / (T - 1) * (beta_end - beta_start)
    alpha_linear = 1.0 - beta_linear
    alpha_bar_linear = np.cumprod(alpha_linear)

    # Cosine schedule (Nichol & Dhariwal)
    s = 0.008
    f_t = np.cos(((t / T) + s) / (1 + s) * (np.pi / 2)) ** 2
    f_0 = np.cos((s / (1 + s)) * (np.pi / 2)) ** 2
    alpha_bar_cosine = f_t / f_0
    alpha_bar_cosine = np.clip(alpha_bar_cosine, 1e-8, 1.0)

    # Derive beta from alpha_bar for cosine
    alpha_bar_cosine_shifted = np.concatenate([[1.0], alpha_bar_cosine[:-1]])
    beta_cosine = 1.0 - (alpha_bar_cosine / alpha_bar_cosine_shifted)
    beta_cosine = np.clip(beta_cosine, 0, 0.999)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL_W, 2.8))

    # Left panel: β_t
    ax1.plot(t, beta_linear, color="#0072B2", ls="-", linewidth=1.3, label="Linear")
    ax1.plot(t, beta_cosine, color="#CC79A7", ls=":", linewidth=1.3, label="Cosine")
    ax1.set_xlabel("Timestep $t$")
    ax1.set_ylabel(r"$\beta_t$")
    ax1.set_title(r"Per-Step Noise ($\beta_t$)")
    ax1.legend(loc="upper left", framealpha=0.9, edgecolor="0.8")
    ax1.set_xlim(0, T - 1)

    # Right panel: ᾱ_t
    ax2.plot(t, alpha_bar_linear, color="#0072B2", ls="-", linewidth=1.3, label="Linear")
    ax2.plot(t, alpha_bar_cosine, color="#CC79A7", ls=":", linewidth=1.3, label="Cosine")
    ax2.set_xlabel("Timestep $t$")
    ax2.set_ylabel(r"$\bar{\alpha}_t$")
    ax2.set_title(r"Cumulative Signal Retained ($\bar{\alpha}_t$)")
    ax2.legend(loc="upper right", framealpha=0.9, edgecolor="0.8")
    ax2.set_xlim(0, T - 1)
    ax2.set_ylim(0, 1.05)

    fig.tight_layout(w_pad=1.5)
    save_fig(fig, "noise_schedule_comparison.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    setup_rc()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and deduplicating CSVs …")
    data: dict[str, pd.DataFrame] = {}
    for label, fname, *_ in CONFIGS:
        df = load_csv(fname)
        data[fname] = df
        fid_rows = df.dropna(subset=["fid"])
        best = fid_rows["fid"].min() if not fid_rows.empty else float("nan")
        print(f"  {label:25s}  epochs {df['epoch'].min()}–{df['epoch'].max()}"
              f"  ({len(df)} rows)  best FID = {best:.2f}")

    print("\nGenerating supplementary figures (6–10) …")
    fig_training_stability(data)
    fig_fid_vs_loss_scatter(data)
    fig_convergence_speed(data)
    fig_fid_improvement_rate(data)
    fig_noise_schedule_comparison()

    print("\n✓ All 5 supplementary figures generated in:")
    print(f"  {FIG_DIR}")
    for f in sorted(FIG_DIR.glob("*.png")):
        print(f"    • {f.name}")


if __name__ == "__main__":
    main()
