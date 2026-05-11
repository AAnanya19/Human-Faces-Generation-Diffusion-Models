#!/usr/bin/env python3
"""
Generate publication-quality training log visualizations for IEEE conference report.

Produces 5 figures from DDPM training logs across 4 configurations:
  1. Combined loss curves (all configs overlaid)
  2. FID progression (all configs overlaid)
  3. Learning rate schedule comparison
  4. Multi-panel summary (2×2 hero figure)
  5. Individual loss curves (2×2 subplots)

All outputs are 300 DPI PNGs sized for a two-column IEEE paper.
"""

import shutil
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

WHATSAPP_SOURCES = {
    "training_log_baseline.csv": Path(
        "/Users/the1finix/Library/Containers/net.whatsapp.WhatsApp/Data/tmp/documents/"
        "A45A88DC-E95B-47AC-B0DE-470463F20489/training_log_baseline.csv"
    ),
    "training_log_cosine_diffusion_scheduler.csv": Path(
        "/Users/the1finix/Library/Containers/net.whatsapp.WhatsApp/Data/tmp/documents/"
        "A14FE515-F844-40F8-AA19-00547A73AD61/training_log_cosine_diffusion_scheduler.csv"
    ),
    "training_log_cosineLR.csv": Path(
        "/Users/the1finix/Library/Containers/net.whatsapp.WhatsApp/Data/tmp/documents/"
        "B1DD4885-FC4B-462B-A808-B7CE1A39BAA2/training_log_cosineLR.csv"
    ),
    "training_log_ema.csv": Path(
        "/Users/the1finix/Library/Containers/net.whatsapp.WhatsApp/Data/tmp/documents/"
        "9E290DE0-07F7-4923-BF28-0E4CBD9FAF80/training_log_ema.csv"
    ),
}

# Config metadata: (label, filename, color, linestyle, marker)
CONFIGS = [
    ("Baseline",           "training_log_baseline.csv",                   "#0072B2", "-",  "o"),
    ("Cosine LR",          "training_log_cosineLR.csv",                   "#E69F00", "--", "s"),
    ("EMA",                "training_log_ema.csv",                        "#009E73", "-.", "D"),
    ("Cosine β Schedule", "training_log_cosine_diffusion_scheduler.csv", "#CC79A7", ":",  "^"),
]

DPI = 300
SINGLE_COL_W = 3.5   # inches – IEEE single-column width
DOUBLE_COL_W = 7.16   # inches – IEEE double-column width
SMOOTH_WINDOW = 20


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


def copy_logs():
    """Copy CSV files from WhatsApp temp into repo (idempotent)."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for name, src in WHATSAPP_SOURCES.items():
        dst = LOG_DIR / name
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  Copied {name}")
        elif dst.exists():
            print(f"  Already present: {name}")
        else:
            raise FileNotFoundError(f"Source not found: {src}")


def load_csv(filename: str) -> pd.DataFrame:
    """Load a training log CSV and deduplicate by keeping last occurrence per epoch."""
    df = pd.read_csv(LOG_DIR / filename)
    df = df.drop_duplicates(subset="epoch", keep="last").sort_values("epoch").reset_index(drop=True)
    return df


def smooth(series: pd.Series, window: int = SMOOTH_WINDOW) -> np.ndarray:
    """Simple centred moving average with min_periods=1 so edges aren't clipped."""
    return series.rolling(window=window, center=True, min_periods=1).mean().values


def save_fig(fig, name: str):
    path = FIG_DIR / name
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 1 – Combined loss curves
# ---------------------------------------------------------------------------

def fig_loss_curves(data: dict[str, pd.DataFrame]):
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W, 3.2))

    for label, fname, color, ls, _ in CONFIGS:
        df = data[fname]
        ax.plot(df["epoch"], df["avg_loss"], color=color, alpha=0.12, linewidth=0.5)
        ax.plot(df["epoch"], smooth(df["avg_loss"]), color=color, ls=ls,
                label=label, linewidth=1.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average MSE Loss")
    ax.set_title("Training Loss Across Configurations")
    ax.set_xlim(0, 1000)
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="0.8")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    fig.tight_layout()
    save_fig(fig, "loss_curves_all_configs.png")


# ---------------------------------------------------------------------------
# Figure 2 – FID progression
# ---------------------------------------------------------------------------

def fig_fid_progression(data: dict[str, pd.DataFrame]):
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W, 3.2))

    for label, fname, color, ls, marker in CONFIGS:
        df = data[fname]
        fid_rows = df.dropna(subset=["fid"])
        if fid_rows.empty:
            continue
        ax.plot(fid_rows["epoch"], fid_rows["fid"],
                color=color, ls=ls, marker=marker, markersize=5,
                label=label, linewidth=1.3)

        best_idx = fid_rows["fid"].idxmin()
        best_row = fid_rows.loc[best_idx]
        ax.plot(best_row["epoch"], best_row["fid"],
                marker="*", color=color, markersize=14, zorder=5,
                markeredgecolor="black", markeredgewidth=0.5)
        offset_y = 8 if best_row["fid"] > 45 else -12
        ax.annotate(f'{best_row["fid"]:.2f}',
                    xy=(best_row["epoch"], best_row["fid"]),
                    xytext=(6, offset_y), textcoords="offset points",
                    fontsize=8, fontweight="bold", color=color,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.8))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("FID (lower is better)")
    ax.set_title("FID Score Progression")
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="0.8")
    fig.tight_layout()
    save_fig(fig, "fid_progression_all_configs.png")


# ---------------------------------------------------------------------------
# Figure 3 – LR schedule comparison
# ---------------------------------------------------------------------------

def fig_lr_schedule(data: dict[str, pd.DataFrame]):
    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 2.6))

    baseline_df = data["training_log_baseline.csv"]
    cosine_df = data["training_log_cosineLR.csv"]

    ax.plot(baseline_df["epoch"], baseline_df["lr"],
            color="#0072B2", ls="-", label="Baseline (Fixed)", linewidth=1.5)
    ax.plot(cosine_df["epoch"], cosine_df["lr"],
            color="#E69F00", ls="--", label="Cosine Annealing", linewidth=1.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.legend(loc="center right", framealpha=0.9, edgecolor="0.8")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1e"))
    fig.tight_layout()
    save_fig(fig, "lr_schedule_comparison.png")


# ---------------------------------------------------------------------------
# Figure 4 – Multi-panel summary (hero figure)
# ---------------------------------------------------------------------------

def fig_summary_4panel(data: dict[str, pd.DataFrame]):
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_W, 5.4))

    # --- Top-left: Loss curves (smoothed) ---
    ax = axes[0, 0]
    for label, fname, color, ls, _ in CONFIGS:
        df = data[fname]
        ax.plot(df["epoch"], smooth(df["avg_loss"]),
                color=color, ls=ls, label=label, linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg MSE Loss")
    ax.set_title("(a) Training Loss", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 1000)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.85, edgecolor="0.8")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    # --- Top-right: FID progression ---
    ax = axes[0, 1]
    for label, fname, color, ls, marker in CONFIGS:
        df = data[fname]
        fid_rows = df.dropna(subset=["fid"])
        if fid_rows.empty:
            continue
        ax.plot(fid_rows["epoch"], fid_rows["fid"],
                color=color, ls=ls, marker=marker, markersize=4,
                label=label, linewidth=1.0)
        best_idx = fid_rows["fid"].idxmin()
        best_row = fid_rows.loc[best_idx]
        ax.plot(best_row["epoch"], best_row["fid"],
                marker="*", color=color, markersize=11, zorder=5,
                markeredgecolor="black", markeredgewidth=0.4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("FID")
    ax.set_title("(b) FID Progression", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.85, edgecolor="0.8")

    # --- Bottom-left: LR schedules ---
    ax = axes[1, 0]
    baseline_df = data["training_log_baseline.csv"]
    cosine_df = data["training_log_cosineLR.csv"]
    ax.plot(baseline_df["epoch"], baseline_df["lr"],
            color="#0072B2", ls="-", label="Fixed LR", linewidth=1.2)
    ax.plot(cosine_df["epoch"], cosine_df["lr"],
            color="#E69F00", ls="--", label="Cosine Annealing", linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("(c) LR Schedule", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="center right", framealpha=0.85, edgecolor="0.8")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1e"))

    # --- Bottom-right: Bar chart of best FID ---
    ax = axes[1, 1]
    best_fids = []
    for label, fname, color, _, _ in CONFIGS:
        df = data[fname]
        fid_rows = df.dropna(subset=["fid"])
        if fid_rows.empty:
            best_fids.append((label, float("inf"), color))
        else:
            best_fids.append((label, fid_rows["fid"].min(), color))

    best_fids.sort(key=lambda x: x[1])
    labels_sorted = [b[0] for b in best_fids]
    vals_sorted = [b[1] for b in best_fids]
    colors_sorted = [b[2] for b in best_fids]

    bars = ax.bar(labels_sorted, vals_sorted, color=colors_sorted,
                  edgecolor="black", linewidth=0.5, width=0.55)
    for bar, val in zip(bars, vals_sorted):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_ylabel("Best FID")
    ax.set_title("(d) Best FID Comparison", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(vals_sorted) * 1.15)
    ax.tick_params(axis="x", labelsize=8)

    fig.tight_layout(h_pad=1.5, w_pad=1.2)
    save_fig(fig, "training_summary_4panel.png")


# ---------------------------------------------------------------------------
# Figure 5 – Individual loss curves (2×2 subplots)
# ---------------------------------------------------------------------------

def fig_loss_individual(data: dict[str, pd.DataFrame]):
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_W, 5.0))

    all_losses = pd.concat([data[fname]["avg_loss"] for _, fname, *_ in CONFIGS])
    y_min = all_losses.quantile(0.001)
    y_max = all_losses.quantile(0.995) * 1.05

    for ax, (label, fname, color, _, _) in zip(axes.flat, CONFIGS):
        df = data[fname]
        ax.plot(df["epoch"], df["avg_loss"],
                color=color, alpha=0.25, linewidth=0.5, label="Raw")
        ax.plot(df["epoch"], smooth(df["avg_loss"]),
                color=color, linewidth=1.5, label="Smoothed")
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Avg MSE Loss")
        ax.set_xlim(0, 1000)
        ax.set_ylim(y_min, y_max)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.85, edgecolor="0.8")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    fig.tight_layout(h_pad=1.2, w_pad=1.0)
    save_fig(fig, "loss_curves_individual.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    setup_rc()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Step 1 — Copying training logs …")
    copy_logs()

    print("\nStep 2 — Loading and deduplicating CSVs …")
    data: dict[str, pd.DataFrame] = {}
    for label, fname, *_ in CONFIGS:
        df = load_csv(fname)
        data[fname] = df
        fid_rows = df.dropna(subset=["fid"])
        best = fid_rows["fid"].min() if not fid_rows.empty else float("nan")
        print(f"  {label:25s}  epochs {df['epoch'].min()}–{df['epoch'].max()}"
              f"  ({len(df)} rows)  best FID = {best:.2f}")

    print("\nStep 3 — Generating figures …")
    fig_loss_curves(data)
    fig_fid_progression(data)
    fig_lr_schedule(data)
    fig_summary_4panel(data)
    fig_loss_individual(data)

    print("\n✓ All 5 figures generated in:")
    print(f"  {FIG_DIR}")
    for f in sorted(FIG_DIR.glob("*.png")):
        print(f"    • {f.name}")


if __name__ == "__main__":
    main()
