# Report plan

## Task Division


| Section                  | Marks | First pass     | Reviewers |
| ------------------------ | ----- | -------------- | --------- |
| Abstract                 | 5%    | Zannat         | All       |
| Introduction             | 10%   | Zannat         | All       |
| Literature review        | 15%   | Hannah         | All       |
| Methodology              | 20%   | Ananya         | All       |
| Experiments              | 25%   | Hannah + Talha | All       |
| Conclusion + future work | 5%    | Fiyin          | All       |


The remaining 20% of the marks is for **creativity** and is graded across the whole report rather than as a single section.

(cosine noise schedule, EMA, DDIM sampler, optional FiLM text conditioning, the hyperparameter sweep) counts. 

### Methodology (20%)

Subsection structure

- **A. Dataset.** CelebA-HQ, first sorted 3000 images, 2700/300 train/test split at seed 67, 128×128 resolution, random horizontal flip on train only.
- **B. Forward diffusion.** Closed-form noising, linear β schedule 1e-4 → 2e-2 over T = 1000, with a cosine schedule as the alternative we ablate.
- **C. Denoising network.** Residual U-Net, base channels 64, channel multipliers (1, 2, 2, 4, 4), self-attention at 16×16 and 8×8, sinusoidal time embeddings.
- **D. Training.** AdamW with lr 1e-4 and weight decay 1e-4, grad clip 1.0, optional cosine LR annealing, EMA decay 0.9999 (enabled by default; EMA weights used for sampling and FID).
- **E. Reverse sampling.** Full ancestral DDPM reverse process.
- **F. Optional FiLM text conditioning.** Loaded from sidecar metadata; same training pipeline supports unconditional and conditioned runs. This is a creativity callout.
- **G. Evaluation protocol.** FID in Inception-V3 feature space, fixed real test set from `test_files.txt`, 300 generated images per run, EMA weights when enabled.

### Experiments (25%)

- Setup paragraph: Colab A100, library versions, run naming convention, seeds.
- Butterfly toy task: loss curve, sample grid, sanity-check FID (1–2 sentences).
- CelebA-HQ main results: loss curve for the best run, hyperparameter sweep table, best generated grid, real reference grid, one paragraph of honest discussion.
- Sweep table format being used (config / change from baseline / FID), current numbers:
  - Baseline: 46.56
  - Cosine LR: 41.74
  - EMA enabled: 40.98
  - Cosine β schedule: 45.72
  - Dropout 0.05: 41.30
  - `base_channels` raised to 96: still running
  - `batch_size` raised to 16: 42.2 (rerun)
- Separate DDPM vs DDIM table comparing inference time and FID once the DDIM run lands.



