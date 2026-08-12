from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from utils.artifacts import prepare_png_output_dir


def prepare_eval_output_dirs(out_path: Path, preview_path: Optional[Path], *, clean: bool) -> None:
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        prepare_png_output_dir(out_path, clean=clean)
        if preview_path is not None:
            prepare_png_output_dir(preview_path, clean=clean)
    if dist.is_initialized():
        dist.barrier()


def prepare_eval_output_dirs_local(out_path: Path, preview_path: Optional[Path], *, clean: bool) -> None:
    prepare_png_output_dir(out_path, clean=clean)
    if preview_path is not None:
        prepare_png_output_dir(preview_path, clean=clean)


def save_rgb(out_path: str | Path, image: torch.Tensor) -> None:
    arr = (image.detach().cpu().clamp(0, 1).permute(1, 2, 0).float().numpy() * 255).round().astype(np.uint8)
    Image.fromarray(arr).save(str(out_path))


def save_pair_side_by_side(out_path: str | Path, left: torch.Tensor, right: torch.Tensor) -> None:
    grid = torch.cat([left.detach().cpu().clamp(0, 1), right.detach().cpu().clamp(0, 1)], dim=2)
    arr = (grid.permute(1, 2, 0).float().numpy() * 255).round().astype(np.uint8)
    Image.fromarray(arr).save(str(out_path))


def save_triplet_side_by_side(out_path: str | Path, left: torch.Tensor, mid: torch.Tensor, right: torch.Tensor) -> None:
    grid = torch.cat([left.detach().cpu().clamp(0, 1), mid.detach().cpu().clamp(0, 1), right.detach().cpu().clamp(0, 1)], dim=2)
    arr = (grid.permute(1, 2, 0).float().numpy() * 255).round().astype(np.uint8)
    Image.fromarray(arr).save(str(out_path))
