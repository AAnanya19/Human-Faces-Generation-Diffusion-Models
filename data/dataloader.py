from datasets import load_dataset
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import matplotlib.pyplot as plt


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

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = sorted(
            p for p in self.root_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in self.IMAGE_EXTENSIONS
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No images found under: {self.root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


# Define preprocessing steps


def get_transforms(image_size=128):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

# Load the full butterfly dataset without splitting


def full_dataset():
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
):
    transform = get_transforms(image_size=image_size)
    if dataset_source == "hf":
        data = full_dataset()
        train_dataset = ButterflyDataset(data, transform=transform)
        test_dataset = None
    elif dataset_source == "folder":
        if dataset_path is None:
            raise ValueError(
                "dataset_path must be provided when dataset_source='folder'")
        full_folder_dataset = FolderImageDataset(
            dataset_path, transform=transform)
        total_images = len(full_folder_dataset)

        if folder_subset_size is not None:
            if folder_subset_size <= 0:
                raise ValueError(
                    "folder_subset_size must be positive when provided")
            if folder_subset_size > total_images:
                raise ValueError(
                    f"folder_subset_size={folder_subset_size} exceeds available images={total_images}"
                )
            selected_indices = list(range(folder_subset_size))
        else:
            selected_indices = list(range(total_images))

        if folder_test_size < 0:
            raise ValueError("folder_test_size must be non-negative")
        if folder_test_size >= len(selected_indices):
            raise ValueError(
                "folder_test_size must be smaller than the selected dataset size")

        if folder_test_size > 0:
            generator = torch.Generator().manual_seed(seed)
            permutation = torch.randperm(
                len(selected_indices), generator=generator).tolist()
            shuffled_indices = [selected_indices[idx] for idx in permutation]
            test_indices = shuffled_indices[:folder_test_size]
            train_indices = shuffled_indices[folder_test_size:]
            test_dataset = Subset(full_folder_dataset, test_indices)
        else:
            train_indices = selected_indices
            test_dataset = None

        train_dataset = Subset(full_folder_dataset, train_indices)
    else:
        raise ValueError("dataset_source must be either 'hf' or 'folder'")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

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
