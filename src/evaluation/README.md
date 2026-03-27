# Evaluation (`src/evaluation`)

This package scores **unconditional generation** by comparing two sets of images on disk:

- **Real** — reference distribution (e.g. butterfly **test** split exports, or CelebA-HQ **test** faces).
- **Fake** — model outputs after sampling (saved as PNG/JPEG).

## What lives here

| Module | Role |
|--------|------|
| `fid.py` | Fréchet Inception Distance (FID) via torchmetrics + torch-fidelity; images loaded from folders. |
| `visualize.py` | Grids / montages for reports (`save_image_grid`, `grid_from_paths`). |
| `metrics.py` / `qualitative.py` | Thin re-exports so imports read `metrics` vs `qualitative` clearly. |
| `run_eval.py` | CLI: `python -m src.evaluation.run_eval --real-dir … --fake-dir …` |

Export butterfly **test** images for the real folder: `scripts/export_butterfly_reference.py` (repo root).

## Outputs and `results/.../eval/`

The repo **`.gitignore`** ignores most of `results/` (large artefacts) but **allows** `results/butterflies/eval/` so the team can optionally commit small evaluation artefacts (e.g. `fid.json`, thumbnail grids) if you want them in git.

That path is only a **default** for optional grid output (`run_eval.py` when using `--grid-from` without `--grid-out`). You can pass `--out-json` / `--grid-out` to any directory; using `results/butterflies/eval/` keeps evaluation outputs next to `samples/` and `loss_curves/` without mixing them.

## Dependencies

See root `requirements.txt`: `torchmetrics`, `torch-fidelity`, `pillow`, etc. First FID run may download Inception weights to the PyTorch hub cache.
