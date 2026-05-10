import csv
import json
import re
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms


class ButterflyDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    # Convert each image to RGB and apply preprocessing
    def __getitem__(self, idx):
        image = self.dataset[idx]["image"].convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


class FolderImageDataset(torch.utils.data.Dataset):
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(self, root_dir, transform=None, image_paths=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        if image_paths is None:
            self.image_paths = sorted(
                p for p in self.root_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in self.IMAGE_EXTENSIONS
            )
        else:
            self.image_paths = [Path(path) for path in image_paths]
        if not self.image_paths:
            raise FileNotFoundError(f"No images found under: {self.root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


def load_conditioning_metadata(metadata_file):
    metadata_path = Path(metadata_file)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Conditioning metadata file not found: {metadata_path}")

    suffix = metadata_path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        payload = torch.load(metadata_path, map_location="cpu")
    elif suffix == ".json":
        payload = json.loads(metadata_path.read_text())
    elif suffix == ".jsonl":
        payload = [json.loads(line) for line in metadata_path.read_text().splitlines() if line.strip()]
    else:
        raise ValueError(
            "Conditioning metadata must be a .pt/.pth, .json, or .jsonl file."
        )

    conditioning_map = {}

    def add_entry(image_name, value):
        if image_name is None or value is None:
            return
        conditioning_map[Path(image_name).name] = torch.as_tensor(value, dtype=torch.float32).flatten()

    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, dict):
                image_name = value.get("filename") or value.get("image") or value.get("path") or key
                vector = value.get("embedding") or value.get("conditioning") or value.get("vector")
                add_entry(image_name, vector)
            else:
                add_entry(key, value)
    else:
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError(
                    "JSONL conditioning metadata must contain dictionaries per line."
                )
            image_name = item.get("filename") or item.get("image") or item.get("path")
            vector = item.get("embedding") or item.get("conditioning") or item.get("vector")
            add_entry(image_name, vector)

    if not conditioning_map:
        raise ValueError(f"No conditioning vectors found in {metadata_path}")

    conditioning_dim = next(iter(conditioning_map.values())).numel()
    return conditioning_map, conditioning_dim


class ConditionedFolderDataset(torch.utils.data.Dataset):
    IMAGE_EXTENSIONS = FolderImageDataset.IMAGE_EXTENSIONS

    def __init__(self, image_paths, conditioning_map, transform=None):
        self.image_paths = [Path(path) for path in image_paths]
        self.conditioning_map = conditioning_map
        self.transform = transform

        missing = [path.name for path in self.image_paths if path.name not in self.conditioning_map]
        if missing:
            preview = ", ".join(missing[:5])
            raise KeyError(
                f"Missing conditioning for {len(missing)} images. Example files: {preview}"
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        conditioning = self.conditioning_map[image_path.name].clone()
        return {"image": image, "conditioning": conditioning}


# Define preprocessing steps


def get_transforms(image_size=128, train=True):
    transform_steps = [
        transforms.Resize((image_size, image_size)),
    ]
    if train:
        transform_steps.append(transforms.RandomHorizontalFlip())
    transform_steps.extend([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    return transforms.Compose(transform_steps)

# Load the full butterfly dataset without splitting


def full_dataset():
    from datasets import load_dataset

    return load_dataset("huggan/smithsonian_butterflies_subset")["train"]


# Pytorch dataloader for the full dataset


def create_dataloaders(
    batch_size=32,
    image_size=128,
    dataset_source="hf",
    dataset_path=None,
    seed=42,
    num_workers=0,
    folder_subset_size=None,
    folder_test_size=0,
    conditioning_metadata_file=None,
    return_split_info=False,
):
    split_info = None
    train_transform = get_transforms(image_size=image_size, train=True)
    eval_transform = get_transforms(image_size=image_size, train=False)

    if dataset_source == "hf":
        data = full_dataset()
        train_dataset = ButterflyDataset(data, transform=train_transform)
        test_dataset = None
        if return_split_info:
            split_info = {
                "dataset_source": "hf",
                "dataset_name": "huggan/smithsonian_butterflies_subset",
                "train_files": None,
                "test_files": None,
                "conditioning_metadata_file": None,
                "conditioning_enabled": False,
                "conditioning_dim": None,
            }
    elif dataset_source == "folder":
        if dataset_path is None:
            raise ValueError("dataset_path must be provided when dataset_source='folder'")

        folder_dataset = FolderImageDataset(dataset_path, transform=train_transform)
        total_images = len(folder_dataset)

        if folder_subset_size is not None:
            if folder_subset_size <= 0:
                raise ValueError("folder_subset_size must be positive when provided")
            if folder_subset_size > total_images:
                raise ValueError(
                    f"folder_subset_size={folder_subset_size} exceeds available images={total_images}"
                )
            selected_indices = list(range(folder_subset_size))
        else:
            selected_indices = list(range(total_images))

        selected_image_paths = [folder_dataset.image_paths[idx] for idx in selected_indices]
        train_folder_dataset = FolderImageDataset(
            dataset_path,
            transform=train_transform,
            image_paths=selected_image_paths,
        )
        eval_folder_dataset = FolderImageDataset(
            dataset_path,
            transform=eval_transform,
            image_paths=selected_image_paths,
        )

        if folder_test_size < 0:
            raise ValueError("folder_test_size must be non-negative")
        if folder_test_size >= len(selected_image_paths):
            raise ValueError(
                "folder_test_size must be smaller than the selected dataset size"
            )

        conditioning_map = None
        conditioning_dim = None
        if conditioning_metadata_file is not None:
            conditioning_map, conditioning_dim = load_conditioning_metadata(conditioning_metadata_file)
            train_folder_dataset = ConditionedFolderDataset(
                selected_image_paths,
                conditioning_map,
                transform=train_transform,
            )
            eval_folder_dataset = ConditionedFolderDataset(
                selected_image_paths,
                conditioning_map,
                transform=eval_transform,
            )

        if folder_test_size > 0:
            generator = torch.Generator().manual_seed(seed)
            permutation = torch.randperm(len(selected_image_paths), generator=generator).tolist()
            test_indices = permutation[:folder_test_size]
            train_indices = permutation[folder_test_size:]
            test_dataset = Subset(eval_folder_dataset, test_indices)
        else:
            train_indices = list(range(len(selected_image_paths)))
            test_dataset = None

        train_dataset = Subset(train_folder_dataset, train_indices)
        if return_split_info:
            train_files = [str(selected_image_paths[idx]) for idx in train_indices]
            test_files = [str(selected_image_paths[idx]) for idx in test_indices] if folder_test_size > 0 else []
            split_info = {
                "dataset_source": "folder",
                "dataset_path": str(Path(dataset_path)),
                "train_files": train_files,
                "test_files": test_files,
                "conditioning_metadata_file": (
                    str(Path(conditioning_metadata_file)) if conditioning_metadata_file is not None else None
                ),
                "conditioning_enabled": conditioning_map is not None,
                "conditioning_dim": conditioning_dim,
            }
    else:
        raise ValueError("dataset_source must be either 'hf' or 'folder'")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    if return_split_info:
        return train_loader, None, test_loader, split_info
    return train_loader, None, test_loader


# UNCOMMENT TO CHECK THAT DATALOADERS ARE WORKING - CHECKS NORMALISATION AND IMAGES
# def show_images(batch, title=""):
#     fig, axes = plt.subplots(1, 6, figsize=(15, 3))
#     fig.suptitle(title)

#     for i in range(6):
#         img = batch[i].permute(1, 2, 0).cpu()
#         img = (img * 0.5) + 0.5  # unnormalize
#         axes[i].imshow(img.clamp(0, 1))
#         axes[i].axis("off")

#     plt.show()


# if __name__ == "__main__":
#     train_loader, _, _ = create_dataloaders()

#     train_batch = next(iter(train_loader))

#     print("Train min/max:", train_batch.min().item(), train_batch.max().item())

#     show_images(train_batch, "Train Batch")
