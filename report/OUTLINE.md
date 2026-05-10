# Report plan

We are writing the report straight into an Overleaf project (IEEE conference, two-column, 5 pages excluding references). This file is just so we all know who is taking first pass on each section and what that section needs to cover. The actual editing happens in Overleaf, and we all read and tweak every section before it goes in.

Overleaf link: _TBC, Talha to share._

## Who is leading each section

Per Hannah's message on WhatsApp:

| Section | Marks | First pass | Reviewer |
|---|---|---|---|
| Abstract | 5% | Zannat | Hannah |
| Introduction | 10% | Zannat | Ananya |
| Literature review | 15% | Hannah | Zannat |
| Methodology | 20% | Ananya | Talha |
| Experiments | 25% | Hannah + Talha | Ananya + Fiyin |
| Conclusion + future work | 5% | Fiyin | Zannat |

"First pass" means whoever drafts the section first. We all still review and edit every section, since the coursework is explicit that there is no sole owner of any component and that every member must demonstrate working knowledge of the whole project. Reviewing matters as much as drafting; please leave Overleaf comments on each other's sections.

The remaining 20% of the marks is for **creativity** and is graded across the whole report rather than as a single section. Anything non-obvious we did (cosine noise schedule, EMA, DDIM sampler, optional FiLM text conditioning, the hyperparameter sweep, our evaluation tooling) counts. Whoever owns the relevant paragraph should call those out explicitly.

## What each section needs to cover

### Abstract (5%)
- 150–250 words.
- One sentence on the problem, one on the approach, two or three on experiments and the headline FID, one on what we learned.
- Written last so we can quote the final number.
- Talha already has a strong draft in the Word doc; it just needs the final FID and a couple of small tweaks once experiments lock in.

### Introduction (10%)
- Why diffusion models for faces.
- The two-stage plan: butterflies first, then CelebA-HQ.
- A short contribution list (DDPM with cosine schedule, EMA-stabilised training, DDIM sampling, FID-based evaluation, optional FiLM conditioning).
- One figure: pipeline overview.
- Zannat's first pass is in the Word doc and reads well; mostly needs the contributions list updated to match the final experiments.

### Literature review (15%)
- At least five papers, thematic rather than paper-by-paper.
- Rough arc: GANs as the prior baseline → DDPM foundations → architectural and schedule improvements → DDIM and efficiency → evaluation (FID).
- Hannah's draft already covers Sohl-Dickstein, Ho et al., BigGAN, StyleGAN2, Dhariwal & Nichol, and Song et al. (DDIM). Still to add: a short FID / Heusel et al. paragraph, and a sentence reconciling the reference numbering with the rest of the report.

### Methodology (20%)
Subsection structure Talha is using in the Word doc (Ananya leads, Talha reviews):

- **A. Dataset.** CelebA-HQ, first sorted 3000 images, 2700/300 train/test split at seed 67, 128×128 resolution, random horizontal flip on train only.
- **B. Forward diffusion.** Closed-form noising, linear β schedule 1e-4 → 2e-2 over T = 1000, with a cosine schedule as the alternative we ablate.
- **C. Denoising network.** Residual U-Net, base channels 64, channel multipliers (1, 2, 2, 4, 4), self-attention at 16×16 and 8×8, sinusoidal time embeddings.
- **D. Training.** AdamW with lr 1e-4 and weight decay 1e-4, grad clip 1.0, optional cosine LR annealing, EMA decay 0.9999 (enabled by default; EMA weights used for sampling and FID).
- **E. Reverse sampling.** Full ancestral DDPM reverse process.
- **F. Optional FiLM text conditioning.** Loaded from sidecar metadata; same training pipeline supports unconditional and conditioned runs. This is a creativity callout.
- **G. Evaluation protocol.** FID in Inception-V3 feature space, fixed real test set from `test_files.txt`, 300 generated images per run, EMA weights when enabled.
- **H. Implementation summary.** The single-column table Talha already put together listing every parameter.

### Experiments (25%)
- Setup paragraph: Colab A100, library versions, run naming convention, seeds.
- Butterfly toy task: loss curve, sample grid, sanity-check FID (1–2 sentences).
- CelebA-HQ main results: loss curve for the best run, hyperparameter sweep table, best generated grid, real reference grid, one paragraph of honest discussion.
- Sweep table format Talha is already using (config / change from baseline / FID), current numbers from the Word doc:
  - Baseline: 46.56
  - Cosine LR: 41.74
  - EMA enabled: 40.98
  - Cosine β schedule: 45.72
  - Dropout 0.05: 41.30
  - `base_channels` raised to 96: still running
  - `batch_size` raised to 16: 42.2 (rerun)
- Separate DDPM vs DDIM table comparing inference time and FID once the DDIM run lands.

### Conclusion and future work (5%)
- One paragraph recapping the best result and what worked.
- One paragraph being honest about limitations: we trained on 3000 images out of 30000, pixel-space at 128×128 rather than 256×256, and FID has known biases.
- One paragraph on what we would do next: latent diffusion, classifier-free guidance, longer training, the text-conditioned variant via FiLM.

## Figures and tables we will need

| ID | Owner | What | Where it lives |
|---|---|---|---|
| F1 | Talha | Pipeline overview (forward → U-Net → reverse) | manual diagram |
| F2 | Zannat | EDA, a few faces from the training subset | EDA notebook |
| F3 | Ananya | Training loss curve for the best run | `<RUN_DIR>/eval/loss_curve.png` |
| F4 | Fiyin | Best generated grid | `<RUN_DIR>/eval/generated_grid.png` |
| F5 | Fiyin | Real reference grid | `<RUN_DIR>/eval/real_grid.png` |
| T1 | Hannah + Talha | Hyperparameter × FID sweep | `results/celebahq/eval/fid_summary.md` |
| T2 | Talha | DDPM vs DDIM (inference time and FID) | TBC |

## How we are working

- Talha sets up the Overleaf project and pastes the link above.
- Each lead drops their first pass into Overleaf as soon as they can; mark the section with `% [DRAFT — name, date]` at the top so it is obvious what is still in flight.
- Reviewers leave Overleaf comments inline; anything substantive gets resolved in WhatsApp and noted in the comment thread.
- The bibliography stays canonical in `report/references.bib` here and is mirrored into Overleaf. Add new entries here first, then copy across.
- Final pass for length and tone is Talha's.

## TODO before we submit

- [ ] Talha to share the Overleaf link.
- [ ] Confirm the real submission deadline.
- [ ] Lock in the final FID number from the ongoing `base_channels` sweep.
- [ ] Add the DDPM vs DDIM comparison once that run finishes.
- [ ] Consolidate the literature-review references with the methodology references so the numbering is consistent across the report.
