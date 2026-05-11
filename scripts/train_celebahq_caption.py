"""Caption-conditioned CelebA-HQ training launcher."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

try:
    from datasets import load_dataset  # noqa: E402
    from PIL import Image, ImageDraw, ImageFont  # noqa: E402
    from torchvision import transforms  # noqa: E402
    from torchvision.transforms import functional as TF  # noqa: E402
    from tqdm import tqdm  # noqa: E402
    from transformers import CLIPTextModel, CLIPTokenizer  # noqa: E402
except ImportError as exc:
    raise ImportError(
        "Caption training needs datasets, Pillow, torchvision, tqdm, and transformers. "
        "Install the project requirements before running this script."
    ) from exc

from src.diffusion.evaluation.metrics import (  # noqa: E402
    InceptionFeatureExtractor,
    calculate_fid_from_features,
    collect_features,
)
from src.diffusion.sample import sample, sample_with_trajectory  # noqa: E402
from src.diffusion.scheduler import DDPMScheduler  # noqa: E402
from src.models.clip_conditioned_unet import CLIPConditionedUNet  # noqa: E402
from src.training.train import (  # noqa: E402
    ModelEMA,
    append_fid_log,
    append_loss_log,
    append_training_log,
    build_checkpoint_state,
    build_lr_scheduler,
    get_current_lr,
    load_resume_state,
    make_seeded_noise,
    resolve_device,
    save_checkpoint,
    write_full_loss_log,
    write_run_metadata,
    write_split_files,
)


TEXT_COLUMN_CANDIDATES = (
    "text",
    "caption",
    "captions",
    "prompt",
    "description",
    "blip_caption",
    "llava_caption",
)
IMAGE_COLUMN_CANDIDATES = ("image", "img", "jpg", "png")


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        "Expected a boolean value: true/false, yes/no, 1/0, or on/off."
    )


def parse_int_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError(f"Expected comma-separated integers, got: {raw!r}")
    return values


def resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = _ROOT / resolved
    return resolved


def _normalize_caption(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return " ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def infer_column(dataset, requested: str | None, candidates: tuple[str, ...], kind: str) -> str:
    if requested is not None:
        if requested not in dataset.column_names:
            raise ValueError(
                f"Requested {kind} column {requested!r} is not in {dataset.column_names}."
            )
        return requested

    for candidate in candidates:
        if candidate in dataset.column_names:
            return candidate

    example = dataset[0]
    if kind == "caption":
        for column in dataset.column_names:
            value = example[column]
            if isinstance(value, str) or (
                isinstance(value, list) and value and isinstance(value[0], str)
            ):
                return column
    else:
        for column in dataset.column_names:
            value = example[column]
            if hasattr(value, "convert"):
                return column

    raise ValueError(
        f"Could not infer {kind} column from columns {dataset.column_names}. "
        f"Pass --{kind}_column explicitly."
    )


class CaptionedHFDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        hf_dataset,
        *,
        image_column: str,
        caption_column: str,
        transform=None,
    ):
        self.dataset = hf_dataset
        self.image_column = image_column
        self.caption_column = caption_column
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]
        image = item[self.image_column]
        if not hasattr(image, "convert"):
            image = Image.open(image)
        image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        caption = _normalize_caption(item[self.caption_column])
        return {"image": image, "caption": caption}


def get_transforms(image_size: int, *, train: bool) -> transforms.Compose:
    steps = [transforms.Resize((image_size, image_size))]
    if train:
        steps.append(transforms.RandomHorizontalFlip())
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    return transforms.Compose(steps)


def make_caption_collate_fn(tokenizer: CLIPTokenizer, max_length: int):
    def collate(examples: list[dict]) -> dict:
        captions = [example["caption"] for example in examples]
        tokens = tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "image": torch.stack([example["image"] for example in examples]),
            "caption": captions,
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
        }

    return collate


def create_caption_dataloaders(
    *,
    tokenizer: CLIPTokenizer,
    dataset_name: str,
    hf_split: str,
    cache_dir: str | None,
    image_size: int,
    train_size: int,
    test_size: int,
    seed: int,
    batch_size: int,
    num_workers: int,
    clip_max_length: int,
    image_column: str | None,
    caption_column: str | None,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader, dict]:
    data = load_dataset(dataset_name, split=hf_split, cache_dir=cache_dir)
    if len(data) < train_size + test_size:
        raise ValueError(
            f"{dataset_name} split {hf_split!r} has {len(data)} rows, but "
            f"train_size + test_size = {train_size + test_size}."
        )

    resolved_image_column = infer_column(data, image_column, IMAGE_COLUMN_CANDIDATES, "image")
    resolved_caption_column = infer_column(data, caption_column, TEXT_COLUMN_CANDIDATES, "caption")

    shuffled = data.shuffle(seed=seed)
    train_range = range(0, train_size)
    test_range = range(train_size, train_size + test_size)
    train_data = shuffled.select(train_range)
    test_data = shuffled.select(test_range)

    train_dataset = CaptionedHFDataset(
        train_data,
        image_column=resolved_image_column,
        caption_column=resolved_caption_column,
        transform=get_transforms(image_size, train=True),
    )
    test_dataset = CaptionedHFDataset(
        test_data,
        image_column=resolved_image_column,
        caption_column=resolved_caption_column,
        transform=get_transforms(image_size, train=False),
    )
    collate_fn = make_caption_collate_fn(tokenizer, clip_max_length)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    split_info = {
        "dataset_source": "hf",
        "dataset_name": dataset_name,
        "hf_split": hf_split,
        "image_column": resolved_image_column,
        "caption_column": resolved_caption_column,
        "train_size": train_size,
        "test_size": test_size,
        "split_seed": seed,
        "train_files": [
            f"{dataset_name}:{hf_split}:shuffled_row_{idx}" for idx in train_range
        ],
        "test_files": [
            f"{dataset_name}:{hf_split}:shuffled_row_{idx}" for idx in test_range
        ],
    }
    return train_loader, test_loader, split_info


def _tokenizer_subfolder_kwargs(subfolder: str | None) -> dict:
    return {"subfolder": subfolder} if subfolder else {}


def load_clip_components(
    *,
    clip_model_name: str,
    tokenizer_subfolder: str | None,
    text_encoder_subfolder: str | None,
    device: str,
) -> tuple[CLIPTokenizer, CLIPTextModel]:
    tokenizer = CLIPTokenizer.from_pretrained(
        clip_model_name,
        **_tokenizer_subfolder_kwargs(tokenizer_subfolder),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    text_encoder = CLIPTextModel.from_pretrained(
        clip_model_name,
        **_tokenizer_subfolder_kwargs(text_encoder_subfolder),
    )
    text_encoder.to(device)
    text_encoder.eval()
    for param in text_encoder.parameters():
        param.requires_grad_(False)
    return tokenizer, text_encoder


@torch.no_grad()
def encode_text_tokens(
    text_encoder: CLIPTextModel,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    text_encoder_device: str,
    model_device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = input_ids.to(text_encoder_device)
    attention_mask_for_encoder = attention_mask.to(text_encoder_device)
    outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask_for_encoder)
    encoder_hidden_states = outputs.last_hidden_state.to(model_device)
    return encoder_hidden_states, attention_mask.to(model_device)


def collect_fixed_text_batch(loader: DataLoader, *, num_items: int) -> dict:
    input_ids = []
    attention_masks = []
    captions = []
    collected = 0

    for batch in loader:
        needed = num_items - collected
        if needed <= 0:
            break
        input_ids.append(batch["input_ids"][:needed])
        attention_masks.append(batch["attention_mask"][:needed])
        captions.extend(batch["caption"][:needed])
        collected += min(needed, len(batch["caption"]))

    if collected < num_items:
        raise RuntimeError(
            f"Requested {num_items} fixed sample captions, but only collected {collected}."
        )

    return {
        "input_ids": torch.cat(input_ids, dim=0),
        "attention_mask": torch.cat(attention_masks, dim=0),
        "captions": captions,
    }


def _load_font(font_size: int):
    for font_name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(font_name, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_width(draw: ImageDraw.ImageDraw, text: str, font) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def _line_height(draw: ImageDraw.ImageDraw, font) -> int:
    bbox = draw.textbbox((0, 0), "Ag", font=font)
    return bbox[3] - bbox[1] + 4


def _truncate_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    font,
    max_width: int,
) -> str:
    if _text_width(draw, text, font) <= max_width:
        return text
    suffix = "..."
    for end in range(max(len(text) - 1, 0), 0, -1):
        candidate = text[:end].rstrip() + suffix
        if _text_width(draw, candidate, font) <= max_width:
            return candidate
    return suffix


def _wrap_caption(
    draw: ImageDraw.ImageDraw,
    caption: str,
    *,
    font,
    max_width: int,
    max_lines: int,
) -> list[str]:
    words = str(caption).replace("\n", " ").split()
    if not words:
        return [""]

    lines = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if _text_width(draw, candidate, font) <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
            current = word
        else:
            lines.append(_truncate_to_width(draw, word, font=font, max_width=max_width))
            current = ""

    if current:
        lines.append(current)

    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = _truncate_to_width(
            draw,
            lines[-1] + "...",
            font=font,
            max_width=max_width,
        )
    return lines


def _tensor_to_pil(image: torch.Tensor, *, scale: int) -> Image.Image:
    pil_image = TF.to_pil_image(image.detach().cpu().clamp(0, 1))
    if scale != 1:
        resampling = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
        pil_image = pil_image.resize(
            (pil_image.width * scale, pil_image.height * scale),
            resample=resampling,
        )
    return pil_image.convert("RGB")


def save_captioned_grid(
    images: torch.Tensor,
    captions: list[str],
    save_path: Path,
    *,
    nrow: int,
    scale: int,
    font_size: int,
    max_caption_lines: int,
) -> None:
    if images.ndim != 4:
        raise ValueError("images must have shape [batch, channels, height, width]")
    if len(captions) < images.shape[0]:
        captions = captions + [""] * (images.shape[0] - len(captions))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    font = _load_font(font_size)
    scratch = Image.new("RGB", (1, 1), "white")
    draw = ImageDraw.Draw(scratch)
    line_height = _line_height(draw, font)

    image_h = images.shape[-2] * scale
    image_w = images.shape[-1] * scale
    caption_pad = 6
    caption_h = caption_pad * 2 + line_height * max_caption_lines
    cell_w = image_w
    cell_h = image_h + caption_h
    nrow = max(1, min(nrow, images.shape[0]))
    ncol = math.ceil(images.shape[0] / nrow)

    grid = Image.new("RGB", (cell_w * nrow, cell_h * ncol), "white")
    grid_draw = ImageDraw.Draw(grid)
    for idx, image in enumerate(images):
        row = idx // nrow
        col = idx % nrow
        x = col * cell_w
        y = row * cell_h
        pil_image = _tensor_to_pil(image, scale=scale)
        grid.paste(pil_image, (x, y))

        lines = _wrap_caption(
            grid_draw,
            captions[idx],
            font=font,
            max_width=cell_w - caption_pad * 2,
            max_lines=max_caption_lines,
        )
        text_y = y + image_h + caption_pad
        for line in lines:
            grid_draw.text((x + caption_pad, text_y), line, fill=(0, 0, 0), font=font)
            text_y += line_height

    grid.save(save_path)


def append_caption_csv(path: Path, rows: list[tuple[int, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "caption"])
        if write_header:
            writer.writeheader()
        for index, caption in rows:
            writer.writerow({"index": index, "caption": caption})


class ConditionedModel(nn.Module):
    def __init__(self, model: nn.Module, model_kwargs: dict):
        super().__init__()
        self.model = model
        self.model_kwargs = model_kwargs

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.model(x, timesteps, **self.model_kwargs)


@torch.no_grad()
def save_generated_samples(
    model: nn.Module,
    scheduler: DDPMScheduler,
    *,
    image_size: int,
    device: str,
    save_path: Path,
    batch_size: int,
    captions: list[str],
    seed: int,
    model_kwargs: dict,
    caption_scale: int,
    caption_font_size: int,
    max_caption_lines: int,
) -> None:
    initial_noise = make_seeded_noise(
        (batch_size, 3, image_size, image_size),
        device=device,
        seed=seed,
    )
    conditioned_model = ConditionedModel(model, model_kwargs)
    images = sample(
        conditioned_model,
        scheduler,
        image_size=image_size,
        batch_size=batch_size,
        channels=3,
        device=device,
        initial_noise=initial_noise,
    )
    images = (images.clamp(-1, 1) + 1) * 0.5
    save_captioned_grid(
        images,
        captions,
        save_path,
        nrow=min(4, batch_size),
        scale=caption_scale,
        font_size=caption_font_size,
        max_caption_lines=max_caption_lines,
    )


@torch.no_grad()
def save_trajectory_samples(
    model: nn.Module,
    scheduler: DDPMScheduler,
    *,
    image_size: int,
    device: str,
    save_path: Path,
    caption: str,
    seed: int,
    save_every: int,
    model_kwargs: dict,
    caption_scale: int,
    caption_font_size: int,
    max_caption_lines: int,
) -> None:
    initial_noise = make_seeded_noise(
        (1, 3, image_size, image_size),
        device=device,
        seed=seed,
    )
    conditioned_model = ConditionedModel(model, model_kwargs)
    _, trajectory = sample_with_trajectory(
        conditioned_model,
        scheduler,
        image_size=image_size,
        batch_size=1,
        channels=3,
        device=device,
        save_every=save_every,
        initial_noise=initial_noise,
    )
    frames = [((frame.clamp(-1, 1) + 1) * 0.5).cpu() for frame in trajectory]
    grid = torch.cat(frames, dim=0)
    save_captioned_grid(
        grid,
        [caption] * grid.shape[0],
        save_path,
        nrow=grid.shape[0],
        scale=caption_scale,
        font_size=caption_font_size,
        max_caption_lines=max_caption_lines,
    )


@torch.no_grad()
def collect_real_fid_images(test_loader: DataLoader, *, num_images: int) -> torch.Tensor:
    images = []
    collected = 0
    for batch in test_loader:
        needed = num_images - collected
        if needed <= 0:
            break
        current = batch["image"][:needed]
        images.append(((current.clamp(-1, 1) + 1) * 0.5).cpu())
        collected += current.shape[0]

    if collected < num_images:
        raise RuntimeError(
            f"FID needs {num_images} real test images, but only found {collected}."
        )
    return torch.cat(images, dim=0)


@torch.no_grad()
def generate_images_for_fid(
    model: nn.Module,
    text_encoder: CLIPTextModel,
    scheduler: DDPMScheduler,
    *,
    test_loader: DataLoader,
    image_size: int,
    device: str,
    text_encoder_device: str,
    num_images: int,
    seed: int,
    save_dir: Path | None,
    caption_scale: int,
    caption_font_size: int,
    max_caption_lines: int,
) -> torch.Tensor:
    generated_batches = []
    generated = 0
    batch_index = 0
    caption_rows = []

    raw_dir = save_dir / "raw" if save_dir is not None else None
    captioned_dir = save_dir / "captioned" if save_dir is not None else None

    for batch in test_loader:
        if generated >= num_images:
            break

        current_batch_size = min(batch["image"].shape[0], num_images - generated)
        input_ids = batch["input_ids"][:current_batch_size]
        attention_mask = batch["attention_mask"][:current_batch_size]
        captions = batch["caption"][:current_batch_size]
        encoder_hidden_states, text_attention_mask = encode_text_tokens(
            text_encoder,
            input_ids=input_ids,
            attention_mask=attention_mask,
            text_encoder_device=text_encoder_device,
            model_device=device,
        )
        initial_noise = make_seeded_noise(
            (current_batch_size, 3, image_size, image_size),
            device=device,
            seed=seed + batch_index,
        )
        conditioned_model = ConditionedModel(
            model,
            {
                "encoder_hidden_states": encoder_hidden_states,
                "text_attention_mask": text_attention_mask,
            },
        )
        images = sample(
            conditioned_model,
            scheduler,
            image_size=image_size,
            batch_size=current_batch_size,
            channels=3,
            device=device,
            initial_noise=initial_noise,
        )
        images = ((images.clamp(-1, 1) + 1) * 0.5).cpu()

        if save_dir is not None:
            raw_dir.mkdir(parents=True, exist_ok=True)
            captioned_dir.mkdir(parents=True, exist_ok=True)
            for offset, image in enumerate(images):
                image_index = generated + offset
                raw_path = raw_dir / f"generated_{image_index:05d}.png"
                TF.to_pil_image(image.clamp(0, 1)).save(raw_path)
                save_captioned_grid(
                    image.unsqueeze(0),
                    [captions[offset]],
                    captioned_dir / f"generated_{image_index:05d}.png",
                    nrow=1,
                    scale=caption_scale,
                    font_size=caption_font_size,
                    max_caption_lines=max_caption_lines,
                )
                caption_rows.append((image_index, captions[offset]))

        generated_batches.append(images)
        generated += current_batch_size
        batch_index += 1
        print(f"  FID generation: {generated}/{num_images}")

    if generated < num_images:
        raise RuntimeError(
            f"FID needs {num_images} generated images, but only produced {generated}."
        )
    if save_dir is not None:
        append_caption_csv(save_dir / "captions.csv", caption_rows)
    return torch.cat(generated_batches, dim=0)


@torch.no_grad()
def evaluate_fid(
    model: nn.Module,
    text_encoder: CLIPTextModel,
    scheduler: DDPMScheduler,
    *,
    test_loader: DataLoader,
    real_features: torch.Tensor | None,
    feature_extractor: InceptionFeatureExtractor,
    image_size: int,
    device: str,
    text_encoder_device: str,
    fid_device: str,
    fid_num_images: int,
    fid_batch_size: int,
    fid_seed: int,
    save_generated_dir: Path | None,
    caption_scale: int,
    caption_font_size: int,
    max_caption_lines: int,
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
        text_encoder,
        scheduler,
        test_loader=test_loader,
        image_size=image_size,
        device=device,
        text_encoder_device=text_encoder_device,
        num_images=fid_num_images,
        seed=fid_seed,
        save_dir=save_generated_dir,
        caption_scale=caption_scale,
        caption_font_size=caption_font_size,
        max_caption_lines=max_caption_lines,
    )
    generated_features = collect_features(
        generated_images,
        feature_extractor,
        device=fid_device,
        batch_size=fid_batch_size,
    )
    return calculate_fid_from_features(real_features, generated_features), real_features


def train(
    # Dataset
    dataset_name: str = "Ryan-sjtu/celebahq-caption",
    hf_split: str = "train",
    hf_cache_dir: str | None = None,
    train_size: int = 3000,
    test_size: int = 300,
    image_column: str | None = None,
    caption_column: str | None = None,
    # CLIP
    clip_model_name: str = "runwayml/stable-diffusion-v1-5",
    clip_tokenizer_subfolder: str | None = "tokenizer",
    clip_text_encoder_subfolder: str | None = "text_encoder",
    clip_max_length: int = 77,
    text_encoder_device: str | None = None,
    # Diffusion
    timesteps: int = 1000,
    noise_schedule: str = "linear",
    # Model
    image_size: int = 128,
    base_channels: int = 64,
    time_dim: int = 256,
    channel_mults: tuple[int, ...] = (1, 2, 4, 8),
    num_res_blocks: int = 2,
    dropout: float = 0.1,
    attention_resolutions: tuple[int, ...] = (16, 8),
    cross_attention_resolutions: tuple[int, ...] = (16, 8),
    attention_heads: int = 4,
    # Training
    epochs: int = 1000,
    batch_size: int = 16,
    lr: float = 1e-4,
    lr_scheduler: str = "cosine",
    cosine_t_max: int | None = None,
    cosine_eta_min: float = 1e-6,
    device: str = "cpu",
    save_dir: str = "runs/ddpm_runs/celebahq_caption_clip128",
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    checkpoint_every: int = 100,
    use_ema: bool = True,
    ema_decay: float = 0.9999,
    split_seed: int = 67,
    num_workers: int = 8,
    resume_checkpoint: str | None = None,
    # Evaluation
    sample_every: int = 100,
    num_sample_images: int = 8,
    fixed_sample_seed: int = 123,
    fixed_trajectory_seed: int = 321,
    trajectory_save_every: int = 100,
    enable_fid: bool = True,
    fid_every: int = 100,
    fid_num_images: int = 300,
    fid_batch_size: int = 16,
    fid_seed: int = 999,
    fid_patience: int = 8,
    fid_device: str | None = None,
    save_fid_images: bool = True,
    caption_scale: int = 3,
    caption_font_size: int = 12,
    max_caption_lines: int = 3,
) -> tuple[nn.Module, list[float]]:
    os.makedirs(save_dir, exist_ok=True)
    resolved_text_encoder_device = text_encoder_device or device
    pin_memory = torch.device(device).type == "cuda"

    print("Loading frozen CLIP text encoder:", clip_model_name)
    tokenizer, text_encoder = load_clip_components(
        clip_model_name=clip_model_name,
        tokenizer_subfolder=clip_tokenizer_subfolder,
        text_encoder_subfolder=clip_text_encoder_subfolder,
        device=resolved_text_encoder_device,
    )
    context_dim = int(text_encoder.config.hidden_size)

    train_loader, test_loader, split_info = create_caption_dataloaders(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        hf_split=hf_split,
        cache_dir=hf_cache_dir,
        image_size=image_size,
        train_size=train_size,
        test_size=test_size,
        seed=split_seed,
        batch_size=batch_size,
        num_workers=num_workers,
        clip_max_length=clip_max_length,
        image_column=image_column,
        caption_column=caption_column,
        pin_memory=pin_memory,
    )
    write_split_files(save_dir, split_info)

    if enable_fid and test_size < fid_num_images:
        raise ValueError(
            f"FID needs {fid_num_images} fixed test images, but test_size={test_size}."
        )

    run_metadata = {
        "diffusion": {
            "timesteps": timesteps,
            "noise_schedule": noise_schedule,
        },
        "model": {
            "name": "CLIPConditionedUNet",
            "in_channels": 3,
            "out_channels": 3,
            "image_size": image_size,
            "base_channels": base_channels,
            "time_dim": time_dim,
            "context_dim": context_dim,
            "channel_mults": list(channel_mults),
            "num_res_blocks": num_res_blocks,
            "dropout": dropout,
            "attention_resolutions": list(attention_resolutions),
            "cross_attention_resolutions": list(cross_attention_resolutions),
            "attention_heads": attention_heads,
            "conditioning": {
                "type": "frozen_clip_text_encoder",
                "clip_model_name": clip_model_name,
                "clip_tokenizer_subfolder": clip_tokenizer_subfolder,
                "clip_text_encoder_subfolder": clip_text_encoder_subfolder,
                "clip_max_length": clip_max_length,
                "text_encoder_device": resolved_text_encoder_device,
            },
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
            "dataset": split_info,
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
            "caption_scale": caption_scale,
            "caption_font_size": caption_font_size,
            "max_caption_lines": max_caption_lines,
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

    model = CLIPConditionedUNet(
        in_channels=3,
        out_channels=3,
        base_channels=base_channels,
        time_dim=time_dim,
        context_dim=context_dim,
        channel_mults=channel_mults,
        num_res_blocks=num_res_blocks,
        dropout=dropout,
        attention_resolutions=attention_resolutions,
        cross_attention_resolutions=cross_attention_resolutions,
        attention_heads=attention_heads,
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

    fixed_text_batch = collect_fixed_text_batch(test_loader, num_items=num_sample_images)
    fixed_encoder_hidden_states, fixed_text_attention_mask = encode_text_tokens(
        text_encoder,
        input_ids=fixed_text_batch["input_ids"],
        attention_mask=fixed_text_batch["attention_mask"],
        text_encoder_device=resolved_text_encoder_device,
        model_device=device,
    )
    fixed_model_kwargs = {
        "encoder_hidden_states": fixed_encoder_hidden_states,
        "text_attention_mask": fixed_text_attention_mask,
    }
    fixed_captions = fixed_text_batch["captions"]

    stopped_early = False
    final_epoch = start_epoch

    for epoch in range(start_epoch, epochs):
        model.train()
        text_encoder.eval()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch in progress_bar:
            x_start = batch["image"].to(device, non_blocking=pin_memory)
            encoder_hidden_states, text_attention_mask = encode_text_tokens(
                text_encoder,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                text_encoder_device=resolved_text_encoder_device,
                model_device=device,
            )
            noise = torch.randn_like(x_start)
            batch_size_current = x_start.shape[0]
            timesteps_batch = torch.randint(
                0,
                timesteps,
                (batch_size_current,),
                device=device,
            ).long()
            x_t = scheduler.add_noise(x_start, noise, timesteps_batch)
            pred_noise = model(
                x_t,
                timesteps_batch,
                encoder_hidden_states=encoder_hidden_states,
                text_attention_mask=text_attention_mask,
            )
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
                    captions=fixed_captions,
                    seed=fixed_sample_seed,
                    model_kwargs=fixed_model_kwargs,
                    caption_scale=caption_scale,
                    caption_font_size=caption_font_size,
                    max_caption_lines=max_caption_lines,
                )
                save_trajectory_samples(
                    model,
                    scheduler,
                    image_size=image_size,
                    device=device,
                    save_path=trajectory_dir / f"epoch_{current_epoch}.png",
                    caption=fixed_captions[0],
                    seed=fixed_trajectory_seed,
                    save_every=trajectory_save_every,
                    model_kwargs={
                        "encoder_hidden_states": fixed_encoder_hidden_states[:1],
                        "text_attention_mask": fixed_text_attention_mask[:1],
                    },
                    caption_scale=caption_scale,
                    caption_font_size=caption_font_size,
                    max_caption_lines=max_caption_lines,
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
                    text_encoder,
                    scheduler,
                    test_loader=test_loader,
                    real_features=real_fid_features,
                    feature_extractor=feature_extractor,
                    image_size=image_size,
                    device=device,
                    text_encoder_device=resolved_text_encoder_device,
                    fid_device=resolved_fid_device,
                    fid_num_images=fid_num_images,
                    fid_batch_size=fid_batch_size,
                    fid_seed=fid_seed,
                    save_generated_dir=generated_dir,
                    caption_scale=caption_scale,
                    caption_font_size=caption_font_size,
                    max_caption_lines=max_caption_lines,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CLIP-caption-conditioned CelebA-HQ DDPM training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    local = parser.add_argument_group("Local paths")
    local.add_argument("--run_name", type=str, default="celebahq_caption_clip128")
    local.add_argument("--runs_root", type=str, default="runs/ddpm_runs")
    local.add_argument("--save_dir", type=str, default=None)
    local.add_argument("--allow_cpu", action="store_true")

    data = parser.add_argument_group("Caption dataset")
    data.add_argument("--dataset_name", type=str, default="Ryan-sjtu/celebahq-caption")
    data.add_argument("--hf_split", type=str, default="train")
    data.add_argument("--hf_cache_dir", type=str, default=None)
    data.add_argument("--train_size", type=int, default=3000)
    data.add_argument("--test_size", type=int, default=300)
    data.add_argument("--image_column", type=str, default=None)
    data.add_argument("--caption_column", type=str, default=None)
    data.add_argument("--split_seed", type=int, default=67)
    data.add_argument("--num_workers", type=int, default=8)

    clip = parser.add_argument_group("Frozen CLIP text encoder")
    clip.add_argument("--clip_model_name", type=str, default="runwayml/stable-diffusion-v1-5")
    clip.add_argument("--clip_tokenizer_subfolder", type=str, default="tokenizer")
    clip.add_argument("--clip_text_encoder_subfolder", type=str, default="text_encoder")
    clip.add_argument("--clip_max_length", type=int, default=77)
    clip.add_argument("--text_encoder_device", type=str, default=None)

    diffusion = parser.add_argument_group("Diffusion params")
    diffusion.add_argument("--timesteps", type=int, default=1000)
    diffusion.add_argument("--noise_schedule", choices=["linear", "cosine"], default="linear")

    model = parser.add_argument_group("Model params")
    model.add_argument("--image_size", type=int, default=128)
    model.add_argument("--base_channels", type=int, default=64)
    model.add_argument("--time_dim", type=int, default=256)
    model.add_argument("--channel_mults", type=str, default="1,2,4,8")
    model.add_argument("--num_res_blocks", type=int, default=2)
    model.add_argument("--dropout", type=float, default=0.1)
    model.add_argument("--attention_resolutions", type=str, default="16,8")
    model.add_argument("--cross_attention_resolutions", type=str, default="16,8")
    model.add_argument("--attention_heads", type=int, default=4)

    training = parser.add_argument_group("Training params")
    training.add_argument("--epochs", type=int, default=1000)
    training.add_argument("--batch_size", type=int, default=16)
    training.add_argument("--lr", type=float, default=1e-4)
    training.add_argument("--lr_scheduler", choices=["fixed", "cosine"], default="cosine")
    training.add_argument("--cosine_t_max", type=int, default=None)
    training.add_argument("--cosine_eta_min", type=float, default=1e-6)
    training.add_argument("--weight_decay", type=float, default=1e-4)
    training.add_argument("--grad_clip", type=float, default=1.0)
    training.add_argument("--checkpoint_every", type=int, default=100)
    training.add_argument("--use_ema", type=parse_bool, default=True)
    training.add_argument("--ema_decay", type=float, default=0.9999)
    training.add_argument("--resume_checkpoint", type=str, default=None)
    training.add_argument("--device", type=str, default=None)

    evaluation = parser.add_argument_group("Evaluation params")
    evaluation.add_argument("--sample_every", type=int, default=100)
    evaluation.add_argument("--num_sample_images", type=int, default=8)
    evaluation.add_argument("--fixed_sample_seed", type=int, default=123)
    evaluation.add_argument("--fixed_trajectory_seed", type=int, default=321)
    evaluation.add_argument("--trajectory_save_every", type=int, default=100)
    evaluation.add_argument("--disable_fid", action="store_true")
    evaluation.add_argument("--fid_every", type=int, default=100)
    evaluation.add_argument("--fid_num_images", type=int, default=300)
    evaluation.add_argument("--fid_batch_size", type=int, default=16)
    evaluation.add_argument("--fid_seed", type=int, default=999)
    evaluation.add_argument("--fid_patience", type=int, default=8)
    evaluation.add_argument("--fid_device", type=str, default=None)
    evaluation.add_argument("--no_save_fid_images", action="store_true")
    evaluation.add_argument("--caption_scale", type=int, default=3)
    evaluation.add_argument("--caption_font_size", type=int, default=12)
    evaluation.add_argument("--max_caption_lines", type=int, default=3)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = (
        resolve_project_path(args.save_dir)
        if args.save_dir
        else resolve_project_path(args.runs_root) / args.run_name
    )
    hf_cache_dir = str(resolve_project_path(args.hf_cache_dir)) if args.hf_cache_dir else None
    device = resolve_device(args.device)
    if device == "cpu" and not args.allow_cpu:
        raise RuntimeError(
            "No CUDA or Apple MPS GPU was detected. Pass --device cuda or --device mps, "
            "or pass --allow_cpu if this is just a smoke test."
        )

    save_dir.mkdir(parents=True, exist_ok=True)

    print("Project root:", _ROOT)
    print("Dataset:", args.dataset_name)
    print("Train/test:", args.train_size, args.test_size)
    print("Image size:", args.image_size)
    print("Run dir:", save_dir)
    print("torch:", torch.__version__)
    print("device:", device)

    train(
        dataset_name=args.dataset_name,
        hf_split=args.hf_split,
        hf_cache_dir=hf_cache_dir,
        train_size=args.train_size,
        test_size=args.test_size,
        image_column=args.image_column,
        caption_column=args.caption_column,
        clip_model_name=args.clip_model_name,
        clip_tokenizer_subfolder=args.clip_tokenizer_subfolder,
        clip_text_encoder_subfolder=args.clip_text_encoder_subfolder,
        clip_max_length=args.clip_max_length,
        text_encoder_device=args.text_encoder_device,
        timesteps=args.timesteps,
        noise_schedule=args.noise_schedule,
        image_size=args.image_size,
        base_channels=args.base_channels,
        time_dim=args.time_dim,
        channel_mults=parse_int_list(args.channel_mults),
        num_res_blocks=args.num_res_blocks,
        dropout=args.dropout,
        attention_resolutions=parse_int_list(args.attention_resolutions),
        cross_attention_resolutions=parse_int_list(args.cross_attention_resolutions),
        attention_heads=args.attention_heads,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_scheduler=args.lr_scheduler,
        cosine_t_max=args.cosine_t_max,
        cosine_eta_min=args.cosine_eta_min,
        device=device,
        save_dir=str(save_dir),
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        checkpoint_every=args.checkpoint_every,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        split_seed=args.split_seed,
        num_workers=args.num_workers,
        resume_checkpoint=args.resume_checkpoint,
        sample_every=args.sample_every,
        num_sample_images=args.num_sample_images,
        fixed_sample_seed=args.fixed_sample_seed,
        fixed_trajectory_seed=args.fixed_trajectory_seed,
        trajectory_save_every=args.trajectory_save_every,
        enable_fid=not args.disable_fid,
        fid_every=args.fid_every,
        fid_num_images=args.fid_num_images,
        fid_batch_size=args.fid_batch_size,
        fid_seed=args.fid_seed,
        fid_patience=args.fid_patience,
        fid_device=args.fid_device,
        save_fid_images=not args.no_save_fid_images,
        caption_scale=args.caption_scale,
        caption_font_size=args.caption_font_size,
        max_caption_lines=args.max_caption_lines,
    )

    print("Training complete.")
    print("Checkpoints:", save_dir)
    print("Best model:", save_dir / "best_model.pth")
    print("Latest checkpoint:", save_dir / "latest_checkpoint.pth")
    print("Samples:", save_dir / "generated_samples")
    print("Trajectories:", save_dir / "trajectories")
    print("FID log:", save_dir / "fid_log.csv")
    print("Training log:", save_dir / "training_log.csv")
    print("Loss log:", save_dir / "loss_log.csv")
    print("Metadata:", save_dir / "run_config.json")


if __name__ == "__main__":
    main()
