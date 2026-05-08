from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import Inception_V3_Weights, inception_v3


class InceptionFeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        weights = Inception_V3_Weights.DEFAULT
        self.model = inception_v3(weights=weights, aux_logits=True)
        self.model.fc = nn.Identity()
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = images.clamp(0, 1)
        images = F.interpolate(
            images,
            size=(299, 299),
            mode="bilinear",
            align_corners=False,
        )
        images = (images - self.mean) / self.std
        features = self.model(images)
        if isinstance(features, tuple):
            features = features[0]
        return features.float()


@torch.no_grad()
def collect_features(
    images: torch.Tensor,
    feature_extractor: InceptionFeatureExtractor,
    *,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    features = []
    for start in range(0, images.size(0), batch_size):
        batch = images[start:start + batch_size].to(device)
        features.append(feature_extractor(batch).cpu())
    return torch.cat(features, dim=0)


def calculate_activation_statistics(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    features = features.double()
    mean = features.mean(dim=0)
    centered = features - mean
    covariance = centered.T.matmul(centered) / (features.shape[0] - 1)
    return mean, covariance


def symmetric_matrix_square_root(matrix: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    matrix = (matrix + matrix.T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    eigenvalues = eigenvalues.clamp_min(eps)
    return (eigenvectors * eigenvalues.sqrt().unsqueeze(0)).matmul(eigenvectors.T)


def calculate_fid_from_features(
    real_features: torch.Tensor,
    generated_features: torch.Tensor,
) -> float:
    if real_features.shape[0] < 2 or generated_features.shape[0] < 2:
        raise ValueError("FID requires at least two real and two generated images.")

    real_mean, real_cov = calculate_activation_statistics(real_features)
    gen_mean, gen_cov = calculate_activation_statistics(generated_features)

    mean_diff = real_mean - gen_mean
    real_cov_sqrt = symmetric_matrix_square_root(real_cov)
    covmean = symmetric_matrix_square_root(real_cov_sqrt.matmul(gen_cov).matmul(real_cov_sqrt))

    fid = mean_diff.dot(mean_diff) + torch.trace(real_cov + gen_cov - 2.0 * covmean)
    return float(fid.clamp_min(0).item())
