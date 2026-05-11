# Training Visualisations

All figures were generated from the four training log CSV files in `../logs/`.
The scripts that produced them are `scripts/visualize_training_logs.py` and
`scripts/visualize_training_logs_extra.py`.

## Configurations compared

| Label | Key change from baseline | Best FID |
|---|---|---|
| Baseline | Fixed LR 1e-4, no EMA, dropout 0.1 | 46.55 |
| Cosine LR | Cosine annealing learning-rate schedule | 41.74 |
| EMA | Exponential moving average on weights | 40.98 |
| Cosine β Schedule | Cosine noise (β) schedule | 45.72 |

---

## Figure-by-figure guide

### 1. `loss_curves_all_configs.png` — Training Loss Across Configurations

Overlays the average MSE noise-prediction loss for all four runs over 1000
epochs. Baseline, Cosine LR and EMA converge to roughly the same loss
(≈ 0.010), while the Cosine β Schedule plateaus higher (≈ 0.017) because
the cosine noise distribution shifts where the model spends its capacity.
**Takeaway:** similar training loss does not guarantee similar generation
quality — Baseline and EMA reach the same loss yet EMA achieves a 5.5-point
lower FID.

### 2. `fid_progression_all_configs.png` — FID Score Progression

FID evaluated every 50 epochs for each configuration. Star markers show
the best checkpoint per run.
**Takeaway:** Baseline has the most erratic FID trajectory (swings of ±40
even after epoch 500) because the constant LR keeps the weights
oscillating. EMA and Cosine LR converge smoothly and reach the lowest
FIDs. The Cosine β Schedule starts strong (lowest FID at epoch 100) but
plateaus around 45.

### 3. `lr_schedule_comparison.png` — Learning-Rate Schedule

Compares the fixed LR used by the Baseline with the cosine annealing
schedule. The cosine curve starts at 1e-4 and smoothly decays to near zero
by epoch 1000, allowing broad exploration early and fine-grained refinement
later. This explains why Cosine LR achieves a lower FID than Baseline
despite reaching the same training loss.

### 4. `training_summary_4panel.png` — Four-Panel Summary (Hero Figure)

Combines panels (a) loss curves, (b) FID progression, (c) LR schedule, and
(d) a bar chart of best FID per configuration. Intended as the main figure
for the Experiments section.

### 5. `loss_curves_individual.png` — Per-Configuration Loss Detail

2×2 subplots showing raw and smoothed loss for each run individually, with
matched y-axes for fair comparison. Useful for appendix detail.

### 6. `training_stability_rolling_std.png` — Training Stability

Rolling standard deviation of loss (window = 50 epochs). All
configurations settle to similarly low jitter after the initial transient.
The real stability difference shows up in FID, not loss.

### 7. `fid_vs_loss_scatter.png` — Final Loss vs Best FID

Plots each configuration as a single point (final average loss vs. best
FID). Three configs cluster at nearly identical loss (≈ 0.010) but span
FIDs from 41 to 47. This demonstrates that training loss is a poor proxy
for generation quality — what matters is how the model behaves during the
full 1000-step reverse sampling chain.

### 8. `convergence_speed_bar.png` — Epochs to Reach FID < 60

Horizontal bar chart showing how many epochs each run needed to first break
below FID 60.
- Cosine β Schedule: **epoch 200** (fastest)
- EMA: **epoch 300**
- Baseline and Cosine LR: **epoch 500**

The cosine noise schedule helps the model learn useful features faster
because it preserves more image signal at low timesteps. However, it
plateaus — it reaches FID ≈ 45 and never improves further. EMA and Cosine
LR are slower to start but ultimately achieve better final quality.

### 9. `fid_improvement_rate.png` — Marginal FID Gains per Interval

Shows how much FID improves (ΔFID) between consecutive 50-epoch evaluation
windows. Most gains happen in the first 200 epochs; after epoch 400, all
configurations hover near zero. Baseline is the most erratic, swinging
between +40 and −50, confirming it benefits most from checkpoint selection.

### 10. `noise_schedule_comparison.png` — Linear vs Cosine Noise Schedules

A methodology figure (computed mathematically, not from logs).
- Left panel: per-step noise β_t. The linear schedule increases steadily;
  the cosine schedule adds slightly more in the mid-range but stays gentler
  at the extremes.
- Right panel: cumulative signal retained (ᾱ_t). The linear schedule
  destroys the image aggressively — only ≈ 20 % signal remains by
  timestep 400. The cosine schedule retains ≈ 40 % at the same point,
  giving the model more useful training signal at intermediate noise
  levels.
