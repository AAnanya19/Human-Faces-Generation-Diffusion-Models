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


def create_dataloaders(batch_size=32, image_size=128, seed=42, num_workers=0):
    del seed  # kept for backward compatible function signature
    data = full_dataset()
    transform = get_transforms(image_size=image_size)

    train_dataset = ButterflyDataset(data, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, None, None


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
