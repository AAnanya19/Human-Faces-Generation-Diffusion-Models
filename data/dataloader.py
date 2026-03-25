from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
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

# Define preprocessing steps


def get_transforms(image_size=128):  # Can try 256 if GPU is strong
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

# Split full dataset into train, test, validation


def splits(seed=42):
    dataset = load_dataset("huggan/smithsonian_butterflies_subset")["train"]
    split_1 = dataset.train_test_split(test_size=0.2, seed=seed)
    split_2 = split_1["test"].train_test_split(test_size=0.5, seed=seed)
    return split_1["train"], split_2["train"], split_2["test"]

# Pytorch dataloaders for each split


def create_dataloaders(batch_size=32, image_size=128, seed=42, num_workers=0):
    train_data, val_data, test_data = splits(seed=seed)
    transform = get_transforms(image_size=image_size)

    train_dataset = ButterflyDataset(train_data, transform=transform)
    val_dataset = ButterflyDataset(val_data, transform=transform)
    test_dataset = ButterflyDataset(test_data, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


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
#     train_loader, val_loader, test_loader = create_dataloaders()

#     train_batch = next(iter(train_loader))
#     val_batch = next(iter(val_loader))
#     test_batch = next(iter(test_loader))

#     print("Train min/max:", train_batch.min().item(), train_batch.max().item())
#     print("Val min/max:", val_batch.min().item(), val_batch.max().item())
#     print("Test min/max:", test_batch.min().item(), test_batch.max().item())

#     show_images(train_batch, "Train Batch")
#     show_images(val_batch, "Val Batch")
#     show_images(test_batch, "Test Batch")
