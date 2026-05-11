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


def get_transforms(image_size=128, train=True):
    # The same resize and normalization pipeline will be used for both train and eval,
    # Horizontal flip is only enabled during training.
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


def full_dataset():
    from datasets import load_dataset

    return load_dataset("huggan/smithsonian_butterflies_subset")["train"]

def create_dataloaders(
    batch_size=32,
    image_size=128,
    dataset_source="hf",
    dataset_path=None,
    seed=42,
    num_workers=0,
    folder_subset_size=None,
    folder_test_size=0,
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
            }
    elif dataset_source == "folder":
        if dataset_path is None:
            raise ValueError(
                "dataset_path must be provided when dataset_source='folder'")
        full_folder_dataset = FolderImageDataset(
            dataset_path, transform=train_transform)
        eval_folder_dataset = FolderImageDataset(
            dataset_path,
            transform=eval_transform,
            image_paths=full_folder_dataset.image_paths,
        )
        total_images = len(full_folder_dataset)

        if folder_subset_size is not None:
            if folder_subset_size <= 0:
                raise ValueError(
                    "folder_subset_size must be positive when provided")
            if folder_subset_size > total_images:
                raise ValueError(
                    f"folder_subset_size={folder_subset_size} exceeds available images={total_images}"
                )
            # Take the first sorted paths so the subset is deterministic.
            selected_indices = list(range(folder_subset_size))
        else:
            selected_indices = list(range(total_images))

        if folder_test_size < 0:
            raise ValueError("folder_test_size must be non-negative")
        if folder_test_size >= len(selected_indices):
            raise ValueError(
                "folder_test_size must be smaller than the selected dataset size")

        if folder_test_size > 0:
            # Use a seeded permutation so the split is reproducible.
            generator = torch.Generator().manual_seed(seed)
            permutation = torch.randperm(
                len(selected_indices), generator=generator).tolist()
            shuffled_indices = [selected_indices[idx] for idx in permutation]
            test_indices = shuffled_indices[:folder_test_size]
            train_indices = shuffled_indices[folder_test_size:]
            test_dataset = Subset(eval_folder_dataset, test_indices)
        else:
            train_indices = selected_indices
            test_dataset = None

        train_dataset = Subset(full_folder_dataset, train_indices)
        if return_split_info:
            train_files = [str(full_folder_dataset.image_paths[idx]) for idx in train_indices]
            test_files = (
                [str(full_folder_dataset.image_paths[idx]) for idx in test_indices]
                if folder_test_size > 0
                else []
            )
            split_info = {
                "dataset_source": "folder",
                "dataset_path": str(Path(dataset_path)),
                "train_files": train_files,
                "test_files": test_files,
            }
    else:
        raise ValueError("dataset_source must be either 'hf' or 'folder'")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if return_split_info:
        return train_loader, None, test_loader, split_info
    return train_loader, None, test_loader
