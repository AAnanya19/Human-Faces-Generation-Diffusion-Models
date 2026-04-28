# Evaluation (`src/evaluation`)

This package scores **unconditional generation** by comparing two sets of images on disk:

- **Real** — reference distribution (e.g. butterfly **test** split exports, or the
  300-image CelebA-HQ **test** split held out by training).
- **Fake** — model outputs after sampling (saved as PNG/JPEG).

It is consumed end-to-end by two Colab notebooks:
[`notebooks/colab_butterfly_evaluate.ipynb`](../../notebooks/colab_butterfly_evaluate.ipynb)
and [`notebooks/colab_faces_evaluate.ipynb`](../../notebooks/colab_faces_evaluate.ipynb).

## Modules

| Module | Role |
|--------|------|
| `fid.py` | FID via `torchmetrics` + `torch-fidelity`; loads PNG folders, batches them as `uint8`. |
| `visualize.py` | Grids / montages for reports (`save_image_grid`, `grid_from_paths`). |
| `metrics.py` / `qualitative.py` | Thin re-exports for clearer imports. |
| `run_eval.py` | Low-level CLI: `python -m src.evaluation.run_eval --real-dir ... --fake-dir ...`. |

## Repo-root scripts that drive this package

| Script | Purpose |
|--------|---------|
| `scripts/export_butterfly_reference.py` | Save the butterfly **test** split as PNGs (real folder for the toy task). |
| `scripts/export_celebahq_reference.py`  | Read `test_files.txt` from a training run and save the held-out 300 CelebA-HQ images as PNGs. |
| `scripts/generate_butterflies.py`       | Sample butterflies from a checkpoint; `--num_images` writes individual PNGs (fake folder). |
| `scripts/generate_faces.py`             | Sample 300 faces from a checkpoint into a fake folder. |
| `scripts/evaluate_fid.py`               | Wrapper over `compute_fid_from_directories` that emits a rich `fid.json` plus optional real/fake grids. |
| `scripts/aggregate_fid.py`              | Join every run's `run_config.json` and `fid.json` into a single CSV / Markdown table for the report. |

## End-to-end flow (faces task)

For each training run produced by `notebooks/colab_celebahq_train.ipynb`,
the run directory on Drive looks like:

```
MyDrive/aml/ddpm_runs/<RUN_NAME>/
├── ddpm_final.pth          # checkpoint (state_dict)
├── run_config.json         # training hyperparameters
├── test_files.txt          # 300 absolute paths held out from training
├── loss_log.csv            # per-epoch average MSE
└── samples/                # in-training sample grids
```

`notebooks/colab_faces_evaluate.ipynb` then:

1. Reads `run_config.json` so eval can never drift from training.
2. Calls `scripts/generate_faces.py` → writes 300 PNGs to `<RUN_DIR>/generated_faces/`.
3. Calls `scripts/export_celebahq_reference.py` → writes 300 resized PNGs to `<RUN_DIR>/real_test_set/`.
4. Calls `scripts/evaluate_fid.py` → writes `<RUN_DIR>/fid.json` and `<RUN_DIR>/eval/{generated_grid,real_grid}.png`.
5. Plots `loss_log.csv` to `<RUN_DIR>/eval/loss_curve.png` for the report.
6. Calls `scripts/aggregate_fid.py` over `MyDrive/aml/ddpm_runs/` → `results/celebahq/eval/fid_summary.{csv,md}`.

The same flow works for the butterfly stage via
`notebooks/colab_butterfly_evaluate.ipynb`, which uses
`scripts/export_butterfly_reference.py` instead of `export_celebahq_reference.py`.

## Outputs and `results/.../eval/`

`.gitignore` ignores most of `results/` (large artefacts) but **allows**
`results/butterflies/eval/` and `results/celebahq/eval/` so the team can
commit small evaluation artefacts (FID summaries, thumbnail grids) if useful.
Larger artefacts (300 PNGs per run, etc.) live on Drive only.

## Dependencies

See root `requirements.txt`: `torchmetrics`, `torch-fidelity`, `pillow`, …
First FID run downloads Inception weights to `~/.cache/torch/hub/checkpoints/`.
