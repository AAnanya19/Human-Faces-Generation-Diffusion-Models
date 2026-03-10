# Human Faces Generation with Diffusion Models

A group project implementing a **Denoising Diffusion Probabilistic Model (DDPM)** to generate high-fidelity, realistic human faces using the **CelebA-HQ 256×256** dataset. The model iteratively transforms random noise into detailed facial images via a reverse Markov process, evaluated using the **Fréchet Inception Distance (FID)** score.

---

## Overview

Diffusion models are a novel class of generative models that gradually refine noise into structured images — offering unprecedented control and quality compared to traditional GANs. This project trains a U-Net based DDPM on 30,000 high-resolution celebrity face images, exploring the impact of hyperparameter choices on generation quality.

---

## Results

| Configuration | FID Score | Notes |
|---|---|---|
| Baseline | TBC | Default hyperparameters |
| Optimised | TBC | Best hyperparameter configuration |

> Results will be updated upon project completion.

---

## Features

- **DDPM Implementation:** U-Net architecture with configurable noise schedule and diffusion timesteps
- **Toy Model:** Initial training on butterfly dataset to validate the pipeline before scaling
- **Hyperparameter Ablation:** Systematic exploration of noise schedules, timesteps, and optimiser settings
- **FID Evaluation:** Generated images evaluated against a 300-image held-out test set
- **Visualisations:** Training loss curves, generated image grids, and FID progression
- **Extra Credit:** Text conditioning and advanced visualisation (if implemented)

---

## Dataset

**CelebA-HQ 256×256** — 30,000 high-resolution celebrity face images in JPG format, widely used for evaluating unconditional generative models.

| Split | Images |
|---|---|
| Train | 2,700 |
| Test | 300 |

---

## Project Structure

```
Human-Faces-Generation-Diffusion-Models/
├── data/
│   ├── celeba_hq_256/             # CelebA-HQ dataset images
│   └── butterfly/                 # Toy butterfly dataset
├── models/
│   ├── unet.py                    # U-Net architecture
│   ├── diffusion.py               # DDPM forward/reverse process
│   └── noise_schedule.py         # Noise schedule configurations
├── notebooks/
│   ├── butterfly_training.ipynb   # Toy model training
│   └── celeba_training.ipynb      # Main face generation training
├── utils/
│   ├── dataset.py                 # Dataset class and dataloader
│   ├── fid.py                     # FID score computation
│   └── visualise.py              # Image grid and loss plotting
├── results/
│   ├── generated_images/          # Sample generated faces
│   └── fid_scores.txt             # FID evaluation results
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Prepare the Dataset

Download the CelebA-HQ 256×256 dataset and place it in `data/celeba_hq_256/`.

### 3. Train on Butterfly Dataset (Toy Model)

Open and run `notebooks/butterfly_training.ipynb` in Google Colab to validate the pipeline before training on CelebA-HQ.

### 4. Train on CelebA-HQ

Open `notebooks/celeba_training.ipynb` in Google Colab:

```python
# Example configuration
num_timesteps = 1000
noise_schedule = "linear"   # or "cosine"
batch_size = 32
learning_rate = 1e-4
optimizer = "Adam"
loss = "MSE"
```

### 5. Generate Images

```python
python generate.py --checkpoint models/best_model.pth --num_images 300
```

### 6. Compute FID Score

```python
python utils/fid.py --real data/celeba_hq_256/test/ --generated results/generated_images/
```

---

## Methodology

### Model Architecture
A **U-Net** with skip connections forms the backbone of the denoising network. The model learns to predict the noise added at each timestep, enabling the reverse diffusion process to reconstruct clean images from pure Gaussian noise.

### Forward Process
Noise is progressively added to a clean image over `T` timesteps according to a predefined noise schedule (linear or cosine), producing a sequence from the original image to pure noise.

### Reverse Process
The trained U-Net iteratively denoises the noisy image step by step to recover a realistic face image from random noise.

### Evaluation
Output quality is measured using **FID score** — comparing the statistical distribution of 300 generated images against the held-out test set. Lower FID indicates higher image quality and diversity.

---

## Tech Stack

- **Language:** Python 3.8+
- **Framework:** PyTorch
- **Compute:** Google Colab / Surrey AI Supercomputer
- **Evaluation:** FID Score (`pytorch-fid`)
- **Dataset:** CelebA-HQ 256×256 (30,000 images)

