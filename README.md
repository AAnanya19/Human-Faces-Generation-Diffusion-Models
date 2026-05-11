# Human Faces Generation with Diffusion Models

A group project focused on implementing a Denoising Diffusion Probabilistic Model (DDPM) for high quality image generation. The project begins with a butterfly toy task to validate the pipeline and build understanding of diffusion models, before scaling to human face generation on the CelebA-HQ 256×256 dataset. The project also includes a caption-conditioned CelebA-HQ experiment using frozen CLIP text features. The overall objective is to train a model that progressively transforms random noise into realistic images through a learned reverse diffusion process, and to evaluate image quality using qualitative analysis and quantitative metrics such as Fréchet Inception Distance (FID).

## Overview

Diffusion models are a powerful class of generative models that learn to recover structured images from noise through an iterative denoising process. Unlike traditional GAN-based approaches, diffusion models are typically more stable to train and can produce highly realistic results with strong diversity.

This project is structured in two stages:

- **Stage 1 — Toy Task:** train and validate the diffusion pipeline on a butterfly dataset
- **Stage 2 — Main Task:** apply the pipeline to human face generation using CelebA-HQ

An additional caption-conditioned experiment extends the face model so samples can be guided by text captions.

The project also explores how architectural and training choices affect image quality, training stability, and generation performance.

## Results


| Configuration | FID Score | Notes                             |
| ------------- | --------- | --------------------------------- |
| Baseline      | 46.56     | Default hyperparameters           |
| Optimised     | 40.98     | Best hyperparameter configuration |


## Features

- **DDPM Implementation:** U-Net-based diffusion model with configurable timesteps and noise schedules  
- **Toy Model Stage:** butterfly dataset used to validate the pipeline before scaling  
- **Scalable Training Pipeline:** modular code structure for preprocessing, training, sampling, and evaluation  
- **Hyperparameter Exploration:** comparison of settings such as learning rate, diffusion steps, and scheduler choices  
- **Evaluation:** generated outputs assessed using qualitative inspection and quantitative metrics  
- **Visualisations:** training loss curves, generated image grids, and experiment comparisons  
- **Text Conditioning:** CLIP-caption-conditioned U-Net for CelebA-HQ caption experiments  
- **Extensible Design:** repository structured to support additions such as advanced sampling methods

## Dataset

### Toy Dataset

A butterfly image dataset will be used in the first stage of the project to test the full diffusion pipeline on a simpler image generation task.

### Main Dataset

**CelebA-HQ 256×256** — a high-resolution face dataset commonly used in image generation research.


| Split | Images |
| ----- | ------ |
| Train | 2,700  |
| Test  | 300    |


Additional dataset details and preprocessing choices will be documented as the project develops.

### Caption Dataset

The text-conditioned experiment uses a Hugging Face CelebA-HQ caption dataset by default:

```text
Ryan-sjtu/celebahq-caption
```

The caption training script infers the image and caption columns automatically, but they can also be passed explicitly when using another dataset with the same image-caption structure.

## Methodology

The methodology is divided into a sequence of stages, starting from a simpler toy problem and then extending to the main human face generation task.

### 1. Data Preparation

The selected dataset is loaded, cleaned if necessary, and preprocessed into a consistent format suitable for training. This includes resizing, normalization, and dataset splitting. Exploratory data analysis is also performed to understand the image distribution and verify data quality.

### 2. Diffusion Pipeline

A DDPM-based pipeline is implemented to model the forward and reverse diffusion processes. In the forward process, noise is gradually added to training images over a fixed number of timesteps. In the reverse process, a neural network learns to predict and remove this noise, reconstructing the image step by step.

### 3. Model Architecture

The denoising network is based on a U-Net architecture, which is commonly used in diffusion models due to its ability to capture both local and global image features. The caption-conditioned version keeps the same DDPM setup and adds cross-attention from image features to frozen CLIP text embeddings. The exact architecture and hyperparameters may be refined during experimentation.

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
├── data/           # Dataset instructions and local setup notes
├── notebooks/      # EDA, experimentation, training, and evaluation notebooks
├── scripts/        # Training, generation, visualisation, and comparison scripts
├── src/            # Core implementation
│   ├── data/       # Dataset loading and preprocessing
│   ├── diffusion/  # Forward/reverse diffusion processes and schedulers
│   │   └── evaluation/ # FID and feature metrics
│   ├── models/     # U-Net architecture
│   └── training/   # Training loop, losses, checkpoints, and logs
└── results/        # Samples and evaluation outputs
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

## Local CelebA-HQ Training

The Colab notebook still works as before. To run the same CelebA-HQ folder
training workflow locally, keep the dataset and run outputs inside the repo:

```text
data/celeba_hq_256/      # local images, ignored by Git
runs/ddpm_runs/<run>/    # local checkpoints, samples, logs, ignored by Git
```

You can also place `data/celeba_hq_256.zip` in the repo and let the launcher
extract it.

Run from the project root:

```bash
python3 scripts/train_celebahq_local.py
```

This is the local running script for the CelebA task. The Colab training
notebook exposes the same core switches, including EMA, FID, checkpointing, and
LR scheduling. Local outputs are kept under `runs/ddpm_runs/<run>/`.

Useful local options:

```bash
# Use a specific local folder
python3 scripts/train_celebahq_local.py --dataset_dir /path/to/celeba_hq_256

# Extract a local zip first
python3 scripts/train_celebahq_local.py --dataset_zip /path/to/celeba_hq_256.zip

# Pick a new run folder
python3 scripts/train_celebahq_local.py --run_name celebahq_mps_test

# Toggle LR scheduling
python3 scripts/train_celebahq_local.py --lr_scheduler cosine
python3 scripts/train_celebahq_local.py --lr_scheduler fixed

# Toggle EMA
python3 scripts/train_celebahq_local.py --use_ema false
python3 scripts/train_celebahq_local.py --use_ema true

# Change FID early-stopping patience
python3 scripts/train_celebahq_local.py --fid_patience 6

# Keep/change the deterministic train/test split
python3 scripts/train_celebahq_local.py --split_seed 42

# Disable FID for a fast smoke test
python3 scripts/train_celebahq_local.py --disable_fid --epochs 1

# Resume a local run
python3 scripts/train_celebahq_local.py \
  --resume_checkpoint runs/ddpm_runs/celebahq_run_001/latest_checkpoint.pth
```

The launcher auto-selects `cuda` first, then Apple `mps`, then CPU. CPU training
is blocked by default because it is extremely slow; pass `--allow_cpu` only for
debugging.

Important run outputs:

```text
latest_checkpoint.pth   # overwritten every checkpoint/FID interval
best_model.pth          # overwritten only when FID improves
ddpm_epoch_50.pth       # epoch checkpoint snapshots
fid_log.csv             # epoch,FID,best-FID records
training_log.csv        # loss, LR, FID, best-model flags
run_config.json         # diffusion/model/training/evaluation params
fid/epoch_00050/        # generated images used for that FID calculation
```

Generate from a trained checkpoint:

```bash
python3 scripts/generate_faces.py \
  --checkpoint runs/ddpm_runs/celebahq_run_001/best_model.pth \
  --out_dir results/generated_faces \
  --num_images 300
```

## Caption-Conditioned CelebA-HQ Training

The caption-conditioned launcher trains `CLIPConditionedUNet`, a pixel-space DDPM U-Net with cross-attention to frozen CLIP text embeddings. It uses the Hugging Face caption dataset by default and saves captioned sample grids alongside the normal checkpoints, logs, FID outputs, and run metadata.

The caption workflow needs `transformers` in addition to the base project dependencies:

```bash
pip install transformers
```

Run from the project root:

```bash
python3 scripts/train_celebahq_caption.py
```

Useful caption options:

```bash
# Fast smoke test without FID
python3 scripts/train_celebahq_caption.py --disable_fid --epochs 1 --allow_cpu

# Use a smaller caption subset
python3 scripts/train_celebahq_caption.py --train_size 1000 --test_size 100

# Pick another Hugging Face dataset or explicit columns
python3 scripts/train_celebahq_caption.py \
  --dataset_name your-org/your-caption-dataset \
  --image_column image \
  --caption_column caption

# Change where Hugging Face dataset files are cached
python3 scripts/train_celebahq_caption.py --hf_cache_dir data/hf_cache

# Change the attention resolutions used for text conditioning
python3 scripts/train_celebahq_caption.py --cross_attention_resolutions 16,8

# Resume a caption-conditioned run
python3 scripts/train_celebahq_caption.py \
  --resume_checkpoint runs/ddpm_runs/celebahq_caption_clip128/latest_checkpoint.pth
```

Caption run outputs follow the same layout as the unconditional CelebA-HQ runs:

```text
generated_samples/      # fixed-caption sample grids
trajectories/           # fixed-caption denoising trajectories
fid/epoch_00050/        # generated FID images, raw and captioned
captions.csv            # captions used for saved FID images
run_config.json         # includes dataset, CLIP, and cross-attention settings
```

## Workflow

- Create a separate branch for each task or feature
- Keep commits small, clear, and focused
- Open a pull request before merging into the main branch
- Avoid committing datasets, checkpoints, or unnecessary large files
- Update documentation whenever major changes are introduced



## License

This project is intended for academic use.