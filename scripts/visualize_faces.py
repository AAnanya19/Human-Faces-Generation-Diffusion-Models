"""Post-training visualization for DDPM face generation runs.

This script is intended to be run after training has already finished.
It loads a saved checkpoint, reconstructs the U-Net + DDPM scheduler,
and writes a small set of visual artifacts to disk:

- a loss curve from the checkpoint's `loss_history` or a CSV log
- optional evaluation-metric plots from a CSV file you provide
- a generated sample grid from the checkpoint
- an optional denoising trajectory montage

Examples:
    python3 scripts/visualize_faces.py \
        --checkpoint runs/ddpm_runs/celebahq_run_001/ddpm_final.pth

    python3 scripts/visualize_faces.py \
        --checkpoint runs/ddpm_runs/celebahq_run_001/ddpm_final.pth \
        --metrics_csv runs/ddpm_runs/celebahq_run_001/eval_metrics.csv \
        --loss_csv runs/ddpm_runs/celebahq_run_001/loss_log.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision.utils as vutils

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.diffusion.sample import sample, sample_with_trajectory  # noqa: E402
from src.diffusion.scheduler import DDPMScheduler  # noqa: E402
from src.models.unet import UNet  # noqa: E402


DEFAULT_IMAGE_SIZE = 256
DEFAULT_TIMESTEPS = 1000
DEFAULT_BASE_CHANNELS = 128
DEFAULT_TIME_DIM = 512
DEFAULT_BATCH_SIZE = 8
DEFAULT_PREVIEW_IMAGES = 16
DEFAULT_TRAJECTORY_SAVE_EVERY = 100


def resolve_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = _ROOT / resolved
    return resolved


def load_json_if_exists(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def infer_run_config(checkpoint_path: Path) -> dict:
    candidates = [
        checkpoint_path.parent / "run_config.json",
        checkpoint_path.parent.parent / "run_config.json",
    ]
    for candidate in candidates:
        payload = load_json_if_exists(candidate)
        if payload is not None:
            return payload
    return {}


def resolve_model_config(run_config: dict, args: argparse.Namespace) -> dict:
    model_cfg = run_config.get("model", {}) if isinstance(run_config, dict) else {}
    channel_mults_value = model_cfg.get("channel_mults", args.channel_mults)
    attention_resolutions_value = model_cfg.get(
        "attention_resolutions", args.attention_resolutions
    )

    if isinstance(channel_mults_value, (list, tuple)):
        channel_mults = tuple(int(x) for x in channel_mults_value)
    else:
        channel_mults = tuple(
            int(x) for x in str(channel_mults_value).split(",") if str(x).strip()
        )

    if isinstance(attention_resolutions_value, (list, tuple)):
        attention_resolutions = tuple(int(x) for x in attention_resolutions_value)
    else:
        attention_resolutions = tuple(
            int(x)
            for x in str(attention_resolutions_value).split(",")
            if str(x).strip()
        )

    return {
        "image_size": int(run_config.get("image_size", args.image_size) or args.image_size),
        "timesteps": int(run_config.get("timesteps", args.timesteps) or args.timesteps),
        "base_channels": int(model_cfg.get("base_channels", args.base_channels) or args.base_channels),
        "time_dim": int(model_cfg.get("time_dim", args.time_dim) or args.time_dim),
        "channel_mults": channel_mults,
        "num_res_blocks": int(model_cfg.get("num_res_blocks", args.num_res_blocks) or args.num_res_blocks),
        "dropout": float(model_cfg.get("dropout", args.dropout) or args.dropout),
        "attention_resolutions": attention_resolutions,
    }


def load_checkpoint_payload(checkpoint_path: Path, device: str) -> dict | dict[str, torch.Tensor]:
    return torch.load(checkpoint_path, map_location=device)


def extract_state_dict(payload: dict | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
        if payload and all(isinstance(key, str) for key in payload.keys()):
            tensor_like_values = sum(torch.is_tensor(value) for value in payload.values())
            if tensor_like_values > 0:
                return payload  # plain state dict
    raise ValueError(
        "Checkpoint format not recognized. Expected either a plain state_dict or "
        "a dict containing model_state_dict/state_dict/model."
    )


def build_model(cfg: dict, device: str) -> UNet:
    return UNet(
        in_channels=3,
        out_channels=3,
        base_channels=cfg["base_channels"],
        time_dim=cfg["time_dim"],
        channel_mults=cfg["channel_mults"],
        num_res_blocks=cfg["num_res_blocks"],
        dropout=cfg["dropout"],
        attention_resolutions=cfg["attention_resolutions"],
        image_size=cfg["image_size"],
    ).to(device)


def load_loss_history(payload: dict | dict[str, torch.Tensor], loss_csv: Path | None) -> list[float]:
    if isinstance(payload, dict) and isinstance(payload.get("loss_history"), list):
        return [float(v) for v in payload["loss_history"]]

    if loss_csv is not None and loss_csv.is_file():
        frame = pd.read_csv(loss_csv)
        for column in ("avg_loss", "loss", "train_loss"):
            if column in frame.columns:
                return [float(v) for v in frame[column].dropna().tolist()]

    return []


def plot_loss_curve(loss_history: list[float], out_path: Path) -> None:
    if not loss_history:
        print("No loss history found; skipping loss plot.")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker="o", linewidth=1.5)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved loss curve to {out_path.resolve()}")


def plot_metrics_csv(metrics_csv: Path, out_path: Path) -> None:
    if not metrics_csv.is_file():
        print("No metrics CSV provided; skipping evaluation-metric plot.")
        return

    frame = pd.read_csv(metrics_csv)
    numeric_columns = list(frame.select_dtypes(include="number").columns)
    if not numeric_columns:
        print(f"Metrics file has no numeric columns: {metrics_csv}")
        return

    x_column = None
    for candidate in ("epoch", "step", "iteration", "index"):
        if candidate in frame.columns:
            x_column = candidate
            break

    x_values = frame[x_column] if x_column is not None else range(1, len(frame) + 1)

    plt.figure(figsize=(9, 5))
    for column in numeric_columns:
        if column == x_column:
            continue
        plt.plot(x_values, frame[column], marker="o", linewidth=1.5, label=column)

    if len(numeric_columns) == 1 and x_column is None:
        plt.plot(x_values, frame[numeric_columns[0]], marker="o", linewidth=1.5)

    plt.title("Evaluation Metrics")
    plt.xlabel(x_column or "Index")
    plt.ylabel("Metric Value")
    plt.grid(True, alpha=0.25)
    if len(numeric_columns) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved metrics plot to {out_path.resolve()}")


def load_fid_json(candidate: Path) -> dict | None:
    if not candidate.is_file():
        return None
    try:
        return json.loads(candidate.read_text())
    except Exception:
        return None


def plot_fid_value(fid_payload: dict, out_path: Path) -> None:
    if not fid_payload or "fid" not in fid_payload:
        print("No FID value found in JSON payload; skipping FID plot.")
        return
    fid_val = float(fid_payload["fid"])
    plt.figure(figsize=(3, 4))
    plt.bar([0], [fid_val], color="#4C72B0")
    plt.ylabel("FID")
    plt.xticks([])
    plt.title(f"FID = {fid_val:.2f}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved FID plot to {out_path.resolve()}")


def plot_fid_summary_csv(summary_csv: Path, out_path: Path) -> None:
    if not summary_csv.is_file():
        print("No FID summary CSV found; skipping.")
        return
    df = pd.read_csv(summary_csv)
    # Prefer a column named 'fid' or 'FID'
    col = None
    for candidate in ("fid", "FID", "FID_score"):
        if candidate in df.columns:
            col = candidate
            break
    if col is None:
        print(f"FID column not found in {summary_csv}; columns: {df.columns.tolist()}")
        return
    plt.figure(figsize=(8, 4))
    if "run_name" in df.columns:
        plt.bar(df["run_name"].astype(str), df[col], color="#4C72B0")
        plt.xticks(rotation=45, ha="right")
    else:
        plt.plot(df.index + 1, df[col], marker="o")
        plt.xlabel("Run")
    plt.ylabel("FID")
    plt.title("FID across runs")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved FID summary plot to {out_path.resolve()}")


def generate_preview_grid(
    model: UNet,
    scheduler: DDPMScheduler,
    image_size: int,
    batch_size: int,
    device: str,
    out_path: Path,
    nrow: int,
) -> None:
    batches = []
    remaining = batch_size
    while remaining > 0:
        current = min(remaining, batch_size)
        imgs = sample(
            model,
            scheduler,
            image_size=image_size,
            batch_size=current,
            channels=3,
            device=device,
        )
        batches.append(((imgs.clamp(-1, 1) + 1) * 0.5).cpu())
        remaining -= current

    all_imgs = torch.cat(batches, dim=0)
    vutils.save_image(all_imgs, out_path, nrow=nrow)
    print(f"Saved preview grid to {out_path.resolve()}")


def save_trajectory_montage(
    model: UNet,
    scheduler: DDPMScheduler,
    image_size: int,
    device: str,
    out_path: Path,
    save_every: int,
) -> None:
    final_images, trajectory = sample_with_trajectory(
        model,
        scheduler,
        image_size=image_size,
        batch_size=1,
        channels=3,
        device=device,
        save_every=save_every,
    )
    _ = final_images
    frames = torch.cat([step[:1].cpu() for step in trajectory], dim=0)
    vutils.save_image(frames, out_path, nrow=len(trajectory))
    print(f"Saved denoising trajectory to {out_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize DDPM face-generation checkpoints and evaluation metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a saved DDPM checkpoint.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/visualizations",
        help="Directory where plots and generated images will be written.",
    )
    parser.add_argument("--metrics_csv", type=str, default=None, help="Optional CSV file with evaluation metrics.")
    parser.add_argument("--loss_csv", type=str, default=None, help="Optional CSV file with logged training losses.")
    parser.add_argument("--device", type=str, default=None, help="Force a device: cuda, mps, or cpu.")
    parser.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--base_channels", type=int, default=DEFAULT_BASE_CHANNELS)
    parser.add_argument("--time_dim", type=int, default=DEFAULT_TIME_DIM)
    parser.add_argument("--channel_mults", type=str, default="1,2,4,8")
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attention_resolutions", type=str, default="16,8")
    parser.add_argument("--num_preview_images", type=int, default=DEFAULT_PREVIEW_IMAGES)
    parser.add_argument("--preview_batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--grid_rows", type=int, default=4)
    parser.add_argument("--trajectory_save_every", type=int, default=DEFAULT_TRAJECTORY_SAVE_EVERY)
    parser.add_argument("--no_samples", action="store_true", help="Skip generating a sample grid.")
    parser.add_argument("--no_trajectory", action="store_true", help="Skip generating a denoising trajectory montage.")
    parser.add_argument("--no_loss_plot", action="store_true", help="Skip plotting loss history.")
    parser.add_argument("--no_metrics_plot", action="store_true", help="Skip plotting metrics CSV data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_path = resolve_project_path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    out_dir = resolve_project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"torch : {torch.__version__}")
    print(f"device: {device}")

    run_config = infer_run_config(checkpoint_path)
    cfg = resolve_model_config(run_config, args)
    print(
        "Model config:",
        f"image_size={cfg['image_size']}",
        f"timesteps={cfg['timesteps']}",
        f"base_channels={cfg['base_channels']}",
        f"time_dim={cfg['time_dim']}",
    )

    payload = load_checkpoint_payload(checkpoint_path, device)
    state_dict = extract_state_dict(payload)

    model = build_model(cfg, device)
    model.load_state_dict(state_dict)
    model.eval()
    print("Checkpoint loaded successfully.")

    scheduler = DDPMScheduler(timesteps=cfg["timesteps"], device=device)

    loss_history = load_loss_history(payload, resolve_project_path(args.loss_csv) if args.loss_csv else None)
    if not args.no_loss_plot:
        plot_loss_curve(loss_history, out_dir / "loss_curve.png")

    if not args.no_metrics_plot:
        metrics_csv = resolve_project_path(args.metrics_csv) if args.metrics_csv else None
        if metrics_csv is not None:
            plot_metrics_csv(metrics_csv, out_dir / "evaluation_metrics.png")
        else:
            print("No metrics CSV provided; skipping evaluation-metric plot.")

    # ---------------------------------------------------------------
    # FID JSON / aggregate CSV support
    # ---------------------------------------------------------------
    fid_candidates = [
        checkpoint_path.parent / "fid.json",
        checkpoint_path.parent / "eval" / "fid.json",
        checkpoint_path.parent.parent / "fid.json",
        checkpoint_path.parent.parent / "eval" / "fid.json",
    ]
    found = False
    for cand in fid_candidates:
        payload = load_fid_json(cand)
        if payload is not None:
            print(f"Found FID JSON at: {cand}")
            plot_fid_value(payload, out_dir / "fid_value.png")
            # also copy the JSON for reference
            try:
                (out_dir / "fid.json").write_text(json.dumps(payload, indent=2))
            except Exception:
                pass
            found = True
            break
    if not found:
        print("No fid.json found in run folder(s).")

    # look for aggregate fid summary CSVs
    summary_candidates = [
        _ROOT / "results" / "celebahq" / "eval" / "fid_summary.csv",
        checkpoint_path.parent / "fid_summary.csv",
        checkpoint_path.parent.parent / "fid_summary.csv",
    ]
    for s in summary_candidates:
        if s.is_file():
            print(f"Found FID summary CSV: {s}")
            plot_fid_summary_csv(s, out_dir / "fid_summary_plot.png")
            break

    # copy any eval grids produced by scripts/evaluate_fid.py
    eval_dir = checkpoint_path.parent / "eval"
    try:
        import shutil

        if eval_dir.is_dir():
            for name in ("generated_grid.png", "generated_grid.jpg", "generated_grid.jpeg", "generated_grid.webp"):
                candidate = eval_dir / name
                if candidate.is_file():
                    shutil.copy(candidate, out_dir / candidate.name)
            for name in ("real_grid.png",):
                candidate = eval_dir / name
                if candidate.is_file():
                    shutil.copy(candidate, out_dir / candidate.name)
    except Exception:
        pass

    if not args.no_samples:
        preview_count = max(1, args.num_preview_images)
        preview_grid_path = out_dir / "sample_grid.png"
        with torch.inference_mode():
            batches = []
            remaining = preview_count
            while remaining > 0:
                current = min(args.preview_batch_size, remaining)
                imgs = sample(
                    model,
                    scheduler,
                    image_size=cfg["image_size"],
                    batch_size=current,
                    channels=3,
                    device=device,
                )
                batches.append(((imgs.clamp(-1, 1) + 1) * 0.5).cpu())
                remaining -= current
            preview_imgs = torch.cat(batches, dim=0)[:preview_count]
        vutils.save_image(preview_imgs, preview_grid_path, nrow=args.grid_rows)
        print(f"Saved sample grid to {preview_grid_path.resolve()}")

    if not args.no_trajectory:
        with torch.inference_mode():
            save_trajectory_montage(
                model,
                scheduler,
                image_size=cfg["image_size"],
                device=device,
                out_path=out_dir / "trajectory_grid.png",
                save_every=args.trajectory_save_every,
            )

    print(f"Visualization artifacts saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()