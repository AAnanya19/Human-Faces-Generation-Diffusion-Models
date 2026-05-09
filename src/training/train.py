"""DDPM training for CelebA-HQ/folder datasets and the original HF toy dataset."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import csv
import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torchvision.utils as vutils  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.data.butterfly_dataset import create_dataloaders  # noqa: E402
from src.diffusion.evaluation.metrics import (  # noqa: E402
    InceptionFeatureExtractor,
    calculate_fid_from_features,
    collect_features,
)
from src.diffusion.sample import sample, sample_with_trajectory  # noqa: E402
from src.diffusion.scheduler import DDPMScheduler  # noqa: E402
from src.models.unet import UNet  # noqa: E402


def resolve_device(requested_device: str | None) -> str:
    if requested_device:
        return requested_device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_int_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError(f"Expected comma-separated integers, got: {raw!r}")
    return values


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def write_run_metadata(save_dir: str | Path, metadata: dict) -> None:
    metadata_path = Path(save_dir) / "run_config.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))


def write_split_files(save_dir: str | Path, split_info: dict | None) -> None:
    if not split_info:
        return
    split_dir = Path(save_dir)
    train_files = split_info.get("train_files")
    test_files = split_info.get("test_files")
    if train_files is not None:
        (split_dir / "train_files.txt").write_text("\n".join(train_files) + "\n")
    if test_files is not None:
        (split_dir / "test_files.txt").write_text("\n".join(test_files) + "\n")


def append_csv_row(path: str | Path, fieldnames: list[str], row: dict) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_loss_log(save_dir: str | Path, epoch: int, avg_loss: float) -> None:
    append_csv_row(
        Path(save_dir) / "loss_log.csv",
        ["epoch", "avg_loss"],
        {"epoch": epoch, "avg_loss": f"{avg_loss:.8f}"},
    )


def append_training_log(
    save_dir: str | Path,
    *,
    epoch: int,
    avg_loss: float,
    lr: float,
    fid: float | None,
    best_fid: float | None,
    is_best: bool,
    fid_no_improve_count: int,
) -> None:
    append_csv_row(
        Path(save_dir) / "training_log.csv",
        [
            "epoch",
            "avg_loss",
            "lr",
            "fid",
            "best_fid",
            "is_best",
            "fid_no_improve_count",
        ],
        {
            "epoch": epoch,
            "avg_loss": f"{avg_loss:.8f}",
            "lr": f"{lr:.10f}",
            "fid": "" if fid is None else f"{fid:.6f}",
            "best_fid": "" if best_fid is None else f"{best_fid:.6f}",
            "is_best": int(is_best),
            "fid_no_improve_count": fid_no_improve_count,
        },
    )


def append_fid_log(
    save_dir: str | Path,
    *,
    epoch: int,
    fid: float,
    best_fid: float,
    is_best: bool,
    used_ema: bool,
    num_generated: int,
    num_real: int,
) -> None:
    append_csv_row(
        Path(save_dir) / "fid_log.csv",
        [
            "epoch",
            "fid",
            "best_fid",
            "is_best",
            "used_ema",
            "num_generated",
            "num_real",
        ],
        {
            "epoch": epoch,
            "fid": f"{fid:.6f}",
            "best_fid": f"{best_fid:.6f}",
            "is_best": int(is_best),
            "used_ema": int(used_ema),
            "num_generated": num_generated,
            "num_real": num_real,
        },
    )


def write_full_loss_log(save_dir: str | Path, losses: list[float]) -> None:
    log_path = Path(save_dir) / "loss_log.csv"
    lines = ["epoch,avg_loss"] + [
        f"{epoch},{loss:.8f}" for epoch, loss in enumerate(losses, start=1)
    ]
    log_path.write_text("\n".join(lines) + "\n")


def make_seeded_noise(
    shape: tuple[int, ...],
    *,
    device: str,
    seed: int,
) -> torch.Tensor:
    device_type = torch.device(device).type
    generator_device = "cpu" if device_type == "mps" else device
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    noise = torch.randn(shape, generator=generator, device=generator_device)
    return noise.to(device)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    lr_scheduler: str,
    epochs: int,
    cosine_t_max: int | None,
    cosine_eta_min: float,
):
    if lr_scheduler == "fixed":
        return None
    if lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_t_max or epochs,
            eta_min=cosine_eta_min,
        )
    raise ValueError("lr_scheduler must be either 'fixed' or 'cosine'")


class ModelEMA:
    """
    Keeps an exponential moving average of trainable model weights.

    Diffusion samples are often cleaner from EMA weights because they smooth out
    noisy step-to-step optimizer updates. The raw model is still kept for
    resuming training; EMA weights are used only for sampling/FID/checkpoint
    evaluation when enabled.
    """

    def __init__(self, model: nn.Module, decay: float) -> None:
        if decay < 0.0 or decay >= 1.0:
            raise ValueError("ema_decay must be >= 0.0 and < 1.0")
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(
                param.detach(),
                alpha=1.0 - self.decay,
            )

    @torch.no_grad()
    def reset(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].copy_(param.detach())

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "shadow": {
                name: tensor.detach().clone()
                for name, tensor in self.shadow.items()
            },
        }

    @torch.no_grad()
    def load_state_dict(self, state: dict) -> None:
        self.decay = float(state.get("decay", self.decay))
        shadow = state.get("shadow", state)
        for name, tensor in shadow.items():
            if name in self.shadow:
                self.shadow[name].copy_(tensor.to(self.shadow[name].device))

    def model_state_dict(self, model: nn.Module) -> dict:
        state = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }
        for name, tensor in self.shadow.items():
            state[name] = tensor.detach().clone()
        return state

    @contextmanager
    def average_parameters(self, model: nn.Module):
        backup = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name not in self.shadow:
                    continue
                backup[name] = param.detach().clone()
                param.copy_(self.shadow[name])
        try:
            yield
        finally:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in backup:
                        param.copy_(backup[name])


def build_checkpoint_state(
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    ema: ModelEMA | None,
    losses: list[float],
    fid_history: list[dict],
    best_fid: float | None,
    fid_no_improve_count: int,
    run_metadata: dict,
) -> dict:
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "eval_model_state_dict": (
            ema.model_state_dict(model) if ema is not None else model.state_dict()
        ),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss_history": losses,
        "fid_history": fid_history,
        "best_fid": best_fid,
        "fid_no_improve_count": fid_no_improve_count,
        "run_config": run_metadata,
        "diffusion_config": run_metadata["diffusion"],
        "model_config": run_metadata["model"],
        "training_config": run_metadata["training"],
        "evaluation_config": run_metadata["evaluation"],
    }
    if lr_scheduler is not None:
        state["lr_scheduler_state_dict"] = lr_scheduler.state_dict()
    if ema is not None:
        state["ema_state_dict"] = ema.state_dict()
    return state


def save_checkpoint(
    save_dir: str | Path,
    *,
    filename: str,
    checkpoint_state: dict,
) -> Path:
    path = Path(save_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_state, path)
    return path


def load_resume_state(
    checkpoint_path: str | None,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    ema: ModelEMA | None,
    device: str,
) -> tuple[int, list[float], list[dict], float | None, int]:
    if checkpoint_path is None:
        return 0, [], [], None, 0

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if lr_scheduler is not None and "lr_scheduler_state_dict" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        if ema is not None:
            if "ema_state_dict" in checkpoint:
                ema.load_state_dict(checkpoint["ema_state_dict"])
            else:
                ema.reset(model)
        return (
            int(checkpoint.get("epoch", 0)),
            list(checkpoint.get("loss_history", [])),
            list(checkpoint.get("fid_history", [])),
            checkpoint.get("best_fid"),
            int(checkpoint.get("fid_no_improve_count", 0)),
        )

    model.load_state_dict(checkpoint)
    if ema is not None:
        ema.reset(model)
    return 0, [], [], None, 0


@torch.no_grad()
def save_generated_samples(
    model: nn.Module,
    scheduler: DDPMScheduler,
    *,
    image_size: int,
    device: str,
    save_path: Path,
    batch_size: int,
    seed: int | None = None,
) -> None:
    initial_noise = None
    if seed is not None:
        initial_noise = make_seeded_noise(
            (batch_size, 3, image_size, image_size),
            device=device,
            seed=seed,
        )
    images = sample(
        model,
        scheduler,
        image_size=image_size,
        batch_size=batch_size,
        channels=3,
        device=device,
        initial_noise=initial_noise,
    )
    images = (images.clamp(-1, 1) + 1) * 0.5
    save_path.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(images, save_path, nrow=min(4, batch_size))


@torch.no_grad()
def save_trajectory_samples(
    model: nn.Module,
    scheduler: DDPMScheduler,
    *,
    image_size: int,
    device: str,
    save_path: Path,
    channels: int = 3,
    seed: int = 0,
    save_every: int = 100,
) -> None:
    initial_noise = make_seeded_noise(
        (1, channels, image_size, image_size),
        device=device,
        seed=seed,
    )
    _, trajectory = sample_with_trajectory(
        model,
        scheduler,
        image_size=image_size,
        batch_size=1,
        channels=channels,
        device=device,
        save_every=save_every,
        initial_noise=initial_noise,
    )
    frames = [((frame.clamp(-1, 1) + 1) * 0.5).cpu() for frame in trajectory]
    grid = torch.cat(frames, dim=0)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(grid, save_path, nrow=len(frames))


@torch.no_grad()
def collect_real_fid_images(test_loader, *, num_images: int) -> torch.Tensor:
    images = []
    collected = 0
    for batch in test_loader:
        needed = num_images - collected
        if needed <= 0:
            break
        batch = batch[:needed]
        images.append(((batch.clamp(-1, 1) + 1) * 0.5).cpu())
        collected += batch.shape[0]

    if collected < num_images:
        raise RuntimeError(
            f"FID needs {num_images} real test images, but only found {collected}. "
            "Increase --folder_test_size or lower --fid_num_images."
        )
    return torch.cat(images, dim=0)


@torch.no_grad()
def generate_images_for_fid(
    model: nn.Module,
    scheduler: DDPMScheduler,
    *,
    image_size: int,
    device: str,
    num_images: int,
    batch_size: int,
    seed: int,
    save_dir: Path | None,
) -> torch.Tensor:
    generated_batches = []
    generated = 0
    batch_index = 0
    while generated < num_images:
        current_batch_size = min(batch_size, num_images - generated)
        initial_noise = make_seeded_noise(
            (current_batch_size, 3, image_size, image_size),
            device=device,
            seed=seed + batch_index,
        )
        images = sample(
            model,
            scheduler,
            image_size=image_size,
            batch_size=current_batch_size,
            channels=3,
            device=device,
            initial_noise=initial_noise,
        )
        images = ((images.clamp(-1, 1) + 1) * 0.5).cpu()
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            for offset, image in enumerate(images):
                image_index = generated + offset
                vutils.save_image(image, save_dir / f"generated_{image_index:05d}.png")
        generated_batches.append(images)
        generated += current_batch_size
        batch_index += 1
        print(f"  FID generation: {generated}/{num_images}")
    return torch.cat(generated_batches, dim=0)


@torch.no_grad()
def evaluate_fid(
    model: nn.Module,
    scheduler: DDPMScheduler,
    *,
    test_loader,
    real_features: torch.Tensor | None,
    feature_extractor: InceptionFeatureExtractor,
    image_size: int,
    device: str,
    fid_device: str,
    fid_num_images: int,
    fid_batch_size: int,
    fid_seed: int,
    save_generated_dir: Path | None,
) -> tuple[float, torch.Tensor]:
    if real_features is None:
        print(f"Collecting {fid_num_images} fixed real test images for FID.")
        real_images = collect_real_fid_images(test_loader, num_images=fid_num_images)
        real_features = collect_features(
            real_images,
            feature_extractor,
            device=fid_device,
            batch_size=fid_batch_size,
        )

    generated_images = generate_images_for_fid(
        model,
        scheduler,
        image_size=image_size,
        device=device,
        num_images=fid_num_images,
        batch_size=fid_batch_size,
        seed=fid_seed,
        save_dir=save_generated_dir,
    )
    generated_features = collect_features(
        generated_images,
        feature_extractor,
        device=fid_device,
        batch_size=fid_batch_size,
    )
    return calculate_fid_from_features(real_features, generated_features), real_features


def train(
    # Diffusion
    timesteps: int = 1000,
    noise_schedule: str = "linear",
    # Model
    image_size: int = 64,
    base_channels: int = 64,
    time_dim: int = 256,
    channel_mults: tuple[int, ...] = (1, 2, 4, 8),
    num_res_blocks: int = 2,
    dropout: float = 0.1,
    attention_resolutions: tuple[int, ...] = (16, 8),
    # Training
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
    lr_scheduler: str = "fixed",
    cosine_t_max: int | None = None,
    cosine_eta_min: float = 1e-6,
    device: str = "cpu",
    save_dir: str = "checkpoints",
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    checkpoint_every: int = 50,
    use_ema: bool = False,
    ema_decay: float = 0.9999,
    dataset_source: str = "hf",
    dataset_path: str | None = None,
    folder_subset_size: int | None = None,
    folder_test_size: int = 0,
    split_seed: int = 42,
    num_workers: int = 0,
    resume_checkpoint: str | None = None,
    # Evaluation
    sample_every: int = 10,
    num_sample_images: int = 8,
    fixed_sample_seed: int = 123,
    fixed_trajectory_seed: int = 321,
    trajectory_save_every: int = 100,
    enable_fid: bool = False,
    fid_every: int = 50,
    fid_num_images: int = 300,
    fid_batch_size: int = 16,
    fid_seed: int = 999,
    fid_patience: int = 0,
    fid_device: str | None = None,
    save_fid_images: bool = True,
):
    os.makedirs(save_dir, exist_ok=True)

    run_metadata = {
        "diffusion": {
            "timesteps": timesteps,
            "noise_schedule": noise_schedule,
        },
        "model": {
            "in_channels": 3,
            "out_channels": 3,
            "image_size": image_size,
            "base_channels": base_channels,
            "time_dim": time_dim,
            "channel_mults": list(channel_mults),
            "num_res_blocks": num_res_blocks,
            "dropout": dropout,
            "attention_resolutions": list(attention_resolutions),
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "lr_scheduler": lr_scheduler,
            "cosine_t_max": cosine_t_max or epochs,
            "cosine_eta_min": cosine_eta_min,
            "weight_decay": weight_decay,
            "grad_clip": grad_clip,
            "checkpoint_every": checkpoint_every,
            "use_ema": use_ema,
            "ema_decay": ema_decay,
            "device": device,
            "save_dir": save_dir,
            "resume_checkpoint": resume_checkpoint,
            "dataset": {
                "source": dataset_source,
                "name": "huggan/smithsonian_butterflies_subset" if dataset_source == "hf" else None,
                "path": dataset_path,
                "folder_subset_size": folder_subset_size,
                "folder_test_size": folder_test_size,
                "split_seed": split_seed,
                "num_workers": num_workers,
            },
        },
        "evaluation": {
            "sample_every": sample_every,
            "num_sample_images": num_sample_images,
            "fixed_sample_seed": fixed_sample_seed,
            "fixed_trajectory_seed": fixed_trajectory_seed,
            "trajectory_save_every": trajectory_save_every,
            "enable_fid": enable_fid,
            "fid_every": fid_every,
            "fid_num_images": fid_num_images,
            "fid_batch_size": fid_batch_size,
            "fid_seed": fid_seed,
            "fid_patience": fid_patience,
            "fid_device": fid_device or device,
            "save_fid_images": save_fid_images,
        },
        "outputs": {
            "epoch_checkpoint_pattern": "ddpm_epoch_{epoch}.pth",
            "latest_checkpoint": "latest_checkpoint.pth",
            "best_model": "best_model.pth",
            "final_checkpoint": "ddpm_final.pth",
            "sample_directory": "generated_samples",
            "trajectory_directory": "trajectories",
            "fid_directory": "fid",
            "loss_log": "loss_log.csv",
            "training_log": "training_log.csv",
            "fid_log": "fid_log.csv",
        },
    }
    write_run_metadata(save_dir, run_metadata)

    train_loader, _, test_loader, split_info = create_dataloaders(
        batch_size=batch_size,
        image_size=image_size,
        dataset_source=dataset_source,
        dataset_path=dataset_path,
        seed=split_seed,
        folder_subset_size=folder_subset_size,
        folder_test_size=folder_test_size,
        num_workers=num_workers,
        return_split_info=True,
    )
    write_split_files(save_dir, split_info)
    if split_info is not None:
        train_files = split_info.get("train_files")
        test_files = split_info.get("test_files")
        run_metadata["training"]["dataset"]["train_split_file"] = "train_files.txt"
        run_metadata["training"]["dataset"]["test_split_file"] = "test_files.txt"
        run_metadata["training"]["dataset"]["train_size"] = (
            len(train_files) if train_files is not None else None
        )
        run_metadata["training"]["dataset"]["test_size"] = (
            len(test_files) if test_files is not None else None
        )
    if enable_fid:
        if test_loader is None:
            raise ValueError("FID is enabled but no test loader exists. Set --folder_test_size 300.")
        test_size = run_metadata["training"]["dataset"].get("test_size")
        if test_size is not None and test_size < fid_num_images:
            raise ValueError(
                f"FID needs {fid_num_images} fixed test images, but test split has {test_size}. "
                "Set --folder_test_size >= --fid_num_images."
            )
    write_run_metadata(save_dir, run_metadata)

    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=base_channels,
        time_dim=time_dim,
        channel_mults=channel_mults,
        num_res_blocks=num_res_blocks,
        dropout=dropout,
        attention_resolutions=attention_resolutions,
        image_size=image_size,
    ).to(device)
    scheduler = DDPMScheduler(
        timesteps=timesteps,
        noise_schedule=noise_schedule,
        device=device,
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ema = ModelEMA(model, decay=ema_decay) if use_ema else None
    lr_scheduler_obj = build_lr_scheduler(
        optimizer,
        lr_scheduler=lr_scheduler,
        epochs=epochs,
        cosine_t_max=cosine_t_max,
        cosine_eta_min=cosine_eta_min,
    )

    start_epoch, losses, fid_history, best_fid, fid_no_improve_count = load_resume_state(
        resume_checkpoint,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler_obj,
        ema=ema,
        device=device,
    )
    if losses:
        write_full_loss_log(save_dir, losses)

    run_metadata["training"]["resume_from_epoch"] = start_epoch
    run_metadata["evaluation"]["best_fid"] = best_fid
    write_run_metadata(save_dir, run_metadata)

    sample_dir = Path(save_dir) / "generated_samples"
    trajectory_dir = Path(save_dir) / "trajectories"
    fid_root = Path(save_dir) / "fid"
    real_fid_features = None
    feature_extractor = None
    resolved_fid_device = resolve_device(fid_device) if fid_device else device

    stopped_early = False
    final_epoch = start_epoch

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch in progress_bar:
            x_start = batch.to(device)
            noise = torch.randn_like(x_start)
            batch_size_current = x_start.shape[0]
            timesteps_batch = torch.randint(
                0,
                timesteps,
                (batch_size_current,),
                device=device,
            ).long()
            x_t = scheduler.add_noise(x_start, noise, timesteps_batch)
            pred_noise = model(x_t, timesteps_batch)
            loss = criterion(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            if ema is not None:
                ema.update(model)

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item(), lr=get_current_lr(optimizer))

        if lr_scheduler_obj is not None:
            lr_scheduler_obj.step()

        current_epoch = epoch + 1
        final_epoch = current_epoch
        current_lr = get_current_lr(optimizer)
        avg_loss = running_loss / len(train_loader)
        losses.append(avg_loss)
        append_loss_log(save_dir, current_epoch, avg_loss)
        print(f"Epoch [{current_epoch}/{epochs}] - Avg Loss: {avg_loss:.6f} - LR: {current_lr:.8f}")

        fid_score = None
        is_best = False
        should_checkpoint = checkpoint_every > 0 and current_epoch % checkpoint_every == 0
        should_fid = enable_fid and fid_every > 0 and current_epoch % fid_every == 0

        eval_context = ema.average_parameters(model) if ema is not None else nullcontext()
        with eval_context:
            if sample_every > 0 and current_epoch % sample_every == 0:
                save_generated_samples(
                    model,
                    scheduler,
                    image_size=image_size,
                    device=device,
                    save_path=sample_dir / f"epoch_{current_epoch}.png",
                    batch_size=num_sample_images,
                    seed=fixed_sample_seed,
                )
                save_trajectory_samples(
                    model,
                    scheduler,
                    image_size=image_size,
                    device=device,
                    save_path=trajectory_dir / f"epoch_{current_epoch}.png",
                    seed=fixed_trajectory_seed,
                    save_every=trajectory_save_every,
                )

            if should_fid:
                if feature_extractor is None:
                    feature_extractor = InceptionFeatureExtractor().to(resolved_fid_device)
                generated_dir = (
                    fid_root / f"epoch_{current_epoch:05d}" / "generated"
                    if save_fid_images
                    else None
                )
                fid_score, real_fid_features = evaluate_fid(
                    model,
                    scheduler,
                    test_loader=test_loader,
                    real_features=real_fid_features,
                    feature_extractor=feature_extractor,
                    image_size=image_size,
                    device=device,
                    fid_device=resolved_fid_device,
                    fid_num_images=fid_num_images,
                    fid_batch_size=fid_batch_size,
                    fid_seed=fid_seed,
                    save_generated_dir=generated_dir,
                )
                is_best = best_fid is None or fid_score < best_fid
                if is_best:
                    best_fid = fid_score
                    fid_no_improve_count = 0
                else:
                    fid_no_improve_count += 1

                fid_record = {
                    "epoch": current_epoch,
                    "fid": fid_score,
                    "best_fid": best_fid,
                    "is_best": is_best,
                    "used_ema": ema is not None,
                }
                fid_history.append(fid_record)
                append_fid_log(
                    save_dir,
                    epoch=current_epoch,
                    fid=fid_score,
                    best_fid=best_fid,
                    is_best=is_best,
                    used_ema=ema is not None,
                    num_generated=fid_num_images,
                    num_real=fid_num_images,
                )
                print(
                    f"Epoch [{current_epoch}/{epochs}] - FID: {fid_score:.4f} "
                    f"- Best FID: {best_fid:.4f}"
                )

        checkpoint_state = build_checkpoint_state(
            epoch=current_epoch,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler_obj,
            ema=ema,
            losses=losses,
            fid_history=fid_history,
            best_fid=best_fid,
            fid_no_improve_count=fid_no_improve_count,
            run_metadata=run_metadata,
        )

        if should_checkpoint or should_fid:
            epoch_checkpoint = save_checkpoint(
                save_dir,
                filename=f"ddpm_epoch_{current_epoch}.pth",
                checkpoint_state=checkpoint_state,
            )
            latest_checkpoint = save_checkpoint(
                save_dir,
                filename="latest_checkpoint.pth",
                checkpoint_state=checkpoint_state,
            )
            run_metadata["outputs"]["latest_checkpoint_path"] = str(latest_checkpoint)
            run_metadata["outputs"]["last_epoch_checkpoint_path"] = str(epoch_checkpoint)

        if is_best:
            best_path = save_checkpoint(
                save_dir,
                filename="best_model.pth",
                checkpoint_state=checkpoint_state,
            )
            run_metadata["outputs"]["best_model_path"] = str(best_path)

        append_training_log(
            save_dir,
            epoch=current_epoch,
            avg_loss=avg_loss,
            lr=current_lr,
            fid=fid_score,
            best_fid=best_fid,
            is_best=is_best,
            fid_no_improve_count=fid_no_improve_count,
        )

        run_metadata["training"]["completed_epoch"] = current_epoch
        run_metadata["training"]["current_lr"] = current_lr
        run_metadata["evaluation"]["best_fid"] = best_fid
        run_metadata["evaluation"]["fid_history"] = fid_history
        run_metadata["evaluation"]["fid_no_improve_count"] = fid_no_improve_count
        write_run_metadata(save_dir, run_metadata)

        if enable_fid and fid_patience > 0 and fid_no_improve_count >= fid_patience:
            print(
                f"Stopping early: FID did not improve for {fid_no_improve_count} "
                f"evaluation rounds (patience={fid_patience})."
            )
            stopped_early = True
            break

    final_state = build_checkpoint_state(
        epoch=final_epoch,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler_obj,
        ema=ema,
        losses=losses,
        fid_history=fid_history,
        best_fid=best_fid,
        fid_no_improve_count=fid_no_improve_count,
        run_metadata=run_metadata,
    )
    final_path = save_checkpoint(save_dir, filename="ddpm_final.pth", checkpoint_state=final_state)

    run_metadata["training"]["stopped_early"] = stopped_early
    run_metadata["training"]["final_epoch"] = final_epoch
    run_metadata["training"]["loss_history"] = losses
    run_metadata["outputs"]["final_checkpoint_path"] = str(final_path)
    write_run_metadata(save_dir, run_metadata)

    return model, losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Diffusion
    diffusion = parser.add_argument_group("Diffusion")
    diffusion.add_argument("--timesteps", type=int, default=1000)
    diffusion.add_argument("--noise_schedule", choices=["linear", "cosine"], default="linear")

    # Model
    model_args = parser.add_argument_group("Model")
    model_args.add_argument("--image_size", type=int, default=64)
    model_args.add_argument("--base_channels", type=int, default=64)
    model_args.add_argument("--time_dim", type=int, default=256)
    model_args.add_argument("--channel_mults", type=str, default="1,2,4,8")
    model_args.add_argument("--num_res_blocks", type=int, default=2)
    model_args.add_argument("--dropout", type=float, default=0.1)
    model_args.add_argument("--attention_resolutions", type=str, default="16,8")

    # Training
    training = parser.add_argument_group("Training")
    training.add_argument("--epochs", type=int, default=10)
    training.add_argument("--batch_size", type=int, default=16)
    training.add_argument("--lr", type=float, default=1e-4)
    training.add_argument("--lr_scheduler", choices=["fixed", "cosine"], default="fixed")
    training.add_argument("--cosine_t_max", type=int, default=None)
    training.add_argument("--cosine_eta_min", type=float, default=1e-6)
    training.add_argument("--device", type=str, default=None)
    training.add_argument("--save_dir", type=str, default="checkpoints")
    training.add_argument("--weight_decay", type=float, default=1e-4)
    training.add_argument("--grad_clip", type=float, default=1.0)
    training.add_argument("--checkpoint_every", type=int, default=50)
    # EMA is opt-in so baseline runs stay directly comparable unless enabled.
    training.add_argument("--use_ema", action="store_true")
    training.add_argument("--ema_decay", type=float, default=0.9999)
    training.add_argument("--dataset_source", type=str, default="hf")
    training.add_argument("--dataset_path", type=str, default=None)
    training.add_argument("--folder_subset_size", type=int, default=None)
    training.add_argument("--folder_test_size", type=int, default=0)
    training.add_argument("--split_seed", type=int, default=42)
    training.add_argument("--num_workers", type=int, default=0)
    training.add_argument("--resume_checkpoint", type=str, default=None)

    # Evaluation
    evaluation = parser.add_argument_group("Evaluation")
    evaluation.add_argument("--sample_every", type=int, default=10)
    evaluation.add_argument("--num_sample_images", type=int, default=8)
    evaluation.add_argument("--fixed_sample_seed", type=int, default=123)
    evaluation.add_argument("--fixed_trajectory_seed", type=int, default=321)
    evaluation.add_argument("--trajectory_save_every", type=int, default=100)
    evaluation.add_argument("--enable_fid", action="store_true")
    evaluation.add_argument("--fid_every", type=int, default=50)
    evaluation.add_argument("--fid_num_images", type=int, default=300)
    evaluation.add_argument("--fid_batch_size", type=int, default=16)
    evaluation.add_argument("--fid_seed", type=int, default=999)
    evaluation.add_argument("--fid_patience", type=int, default=0)
    evaluation.add_argument("--fid_device", type=str, default=None)
    evaluation.add_argument("--no_save_fid_images", action="store_true")

    args = parser.parse_args()
    device = resolve_device(args.device)
    print("Device:", device)
    print("Saving checkpoints to:", args.save_dir)

    model, losses = train(
        timesteps=args.timesteps,
        noise_schedule=args.noise_schedule,
        image_size=args.image_size,
        base_channels=args.base_channels,
        time_dim=args.time_dim,
        channel_mults=parse_int_list(args.channel_mults),
        num_res_blocks=args.num_res_blocks,
        dropout=args.dropout,
        attention_resolutions=parse_int_list(args.attention_resolutions),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_scheduler=args.lr_scheduler,
        cosine_t_max=args.cosine_t_max,
        cosine_eta_min=args.cosine_eta_min,
        device=device,
        save_dir=args.save_dir,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        checkpoint_every=args.checkpoint_every,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        dataset_source=args.dataset_source,
        dataset_path=args.dataset_path,
        folder_subset_size=args.folder_subset_size,
        folder_test_size=args.folder_test_size,
        split_seed=args.split_seed,
        num_workers=args.num_workers,
        resume_checkpoint=args.resume_checkpoint,
        sample_every=args.sample_every,
        num_sample_images=args.num_sample_images,
        fixed_sample_seed=args.fixed_sample_seed,
        fixed_trajectory_seed=args.fixed_trajectory_seed,
        trajectory_save_every=args.trajectory_save_every,
        enable_fid=args.enable_fid,
        fid_every=args.fid_every,
        fid_num_images=args.fid_num_images,
        fid_batch_size=args.fid_batch_size,
        fid_seed=args.fid_seed,
        fid_patience=args.fid_patience,
        fid_device=args.fid_device,
        save_fid_images=not args.no_save_fid_images,
    )

    print("Training complete.")
    print("Loss history:", losses)
