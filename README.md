# Human Faces Generation with Diffusion Models

A group project focused on implementing a Denoising Diffusion Probabilistic Model (DDPM) for high quality image generation. The project begins with a butterfly toy task to validate the pipeline and build understanding of diffusion models, before scaling to human face generation on the CelebA-HQ 256×256 dataset. The overall objective is to train a model that progressively transforms random noise into realistic images through a learned reverse diffusion process, and to evaluate image quality using qualitative analysis and quantitative metrics such as Fréchet Inception Distance (FID).

## Overview

Diffusion models are a powerful class of generative models that learn to recover structured images from noise through an iterative denoising process. Unlike traditional GAN-based approaches, diffusion models are typically more stable to train and can produce highly realistic results with strong diversity.

This project is structured in two stages:

- **Stage 1 — Toy Task:** train and validate the diffusion pipeline on a butterfly dataset
- **Stage 2 — Main Task:** apply the pipeline to human face generation using CelebA-HQ

The project also explores how architectural and training choices affect image quality, training stability, and generation performance.

## Results

| Configuration | FID Score | Notes |
|---|---:|---|
| Baseline | TBC | Default hyperparameters |
| Optimised | TBC | Best hyperparameter configuration |

Results will be updated as experiments are completed.

## Features

- **DDPM Implementation:** U-Net-based diffusion model with configurable timesteps and noise schedules  
- **Toy Model Stage:** butterfly dataset used to validate the pipeline before scaling  
- **Scalable Training Pipeline:** modular code structure for preprocessing, training, sampling, and evaluation  
- **Hyperparameter Exploration:** comparison of settings such as learning rate, diffusion steps, and scheduler choices  
- **Evaluation:** generated outputs assessed using qualitative inspection and quantitative metrics  
- **Visualisations:** training loss curves, generated image grids, and experiment comparisons  
- **Extensible Design:** repository structured to support future additions such as conditioning or advanced sampling methods  

## Dataset

### Toy Dataset
A butterfly image dataset will be used in the first stage of the project to test the full diffusion pipeline on a simpler image generation task.

### Main Dataset
**CelebA-HQ 256×256** — a high-resolution face dataset commonly used in image generation research.

| Split | Images |
|---|---:|
| Train | 2,700 |
| Test | 300 |

Additional dataset details and preprocessing choices will be documented as the project develops.

## Methodology

The methodology is divided into a sequence of stages, starting from a simpler toy problem and then extending to the main human face generation task.

### 1. Data Preparation
The selected dataset is loaded, cleaned if necessary, and preprocessed into a consistent format suitable for training. This includes resizing, normalization, and dataset splitting. Exploratory data analysis is also performed to understand the image distribution and verify data quality.

### 2. Diffusion Pipeline
A DDPM-based pipeline is implemented to model the forward and reverse diffusion processes. In the forward process, noise is gradually added to training images over a fixed number of timesteps. In the reverse process, a neural network learns to predict and remove this noise, reconstructing the image step by step.

### 3. Model Architecture
The denoising network is based on a U-Net architecture, which is commonly used in diffusion models due to its ability to capture both local and global image features. The exact architecture and hyperparameters may be refined during experimentation.

### 4. Training
The model is trained to predict the noise added at each diffusion step, typically using mean squared error (MSE) loss. Optimisation settings such as learning rate, batch size, noise schedule, and number of timesteps are explored during development.

### 5. Sampling and Generation
After training, the model is used to generate images by starting from pure Gaussian noise and iteratively applying the learned reverse diffusion process. Generated images are saved and compared across experiments.

### 6. Evaluation
Model performance is evaluated through a combination of:
- visual inspection of generated samples
- diversity and realism analysis
- comparison across hyperparameter settings
- quantitative metrics such as FID for the main task

This section will be updated with exact evaluation details as experiments progress.

## Project Structure

```text
.
├── configs/        # Experiment configuration files
├── data/           # Dataset instructions and local setup notes
├── notebooks/      # EDA, experimentation, training, and evaluation notebooks
├── src/            # Core implementation
│   ├── data/       # Dataset loading and preprocessing
│   ├── diffusion/  # Forward/reverse diffusion processes and schedulers
│   ├── models/     # U-Net architecture
│   ├── training/   # Training loop, losses, and engine
│   ├── evaluation/ # Metrics and qualitative analysis
│   └── utils/      # Helper utilities
├── results/        # Samples, logs, and evaluation outputs
├── reports/        # Report figures and notes
└── .github/        # Pull request templates and repo workflow support
```

## Setup

Clone the repository:

```bash
git clone https://github.com/AAnanya19/Human-Faces-Generation-Diffusion-Models.git
cd Human-Faces-Generation-Diffusion-Models
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Workflow

- Create a separate branch for each task or feature
- Keep commits small, clear, and focused
- Open a pull request before merging into the main branch
- Avoid committing datasets, checkpoints, or unnecessary large files
- Update documentation and configs whenever major changes are introduced

## Current Status

This repository is currently under active development.  
The structure, methodology, and experimental results will continue to be updated as the project progresses through the toy task and the main face generation stage.

## License

This project is intended for academic use.
