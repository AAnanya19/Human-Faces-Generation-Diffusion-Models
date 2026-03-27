"""
Quantitative evaluation (FID and future metrics).

Implementation lives in ``fid.py``; this module exposes a stable import path
for notebooks and scripts: ``from src.evaluation.metrics import ...``.
"""

from .fid import compute_fid_from_directories, list_image_paths

__all__ = ["compute_fid_from_directories", "list_image_paths"]
