"""
Qualitative evaluation (grids and montages for reports).

Implementation lives in ``visualize.py``; this module exposes a stable import path:
``from src.evaluation.qualitative import ...``.
"""

from .visualize import grid_from_paths, save_image_grid

__all__ = ["grid_from_paths", "save_image_grid"]
