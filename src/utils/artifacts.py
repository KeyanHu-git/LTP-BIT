from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def copy_preview_images(image_dir: str | Path, out_dir: str | Path, max_images: int = 16) -> None:
    paths = sorted(Path(image_dir).glob("*.png"))[:max_images]
    if not paths:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copy2(path, out_dir / path.name)


def save_preview_grid(
    image_dir: str | Path,
    out_path: str | Path,
    *,
    max_images: int = 64,
    thumb_size: int = 128,
) -> None:
    paths = sorted(Path(image_dir).glob("*.png"))[:max_images]
    if not paths:
        return
    cols = int(math.sqrt(max_images))
    rows = math.ceil(len(paths) / cols)
    canvas = Image.new("RGB", (cols * thumb_size, rows * thumb_size), "white")
    resample = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
    for i, path in enumerate(paths):
        img = Image.open(path).convert("RGB").resize((thumb_size, thumb_size), resample)
        canvas.paste(img, ((i % cols) * thumb_size, (i // cols) * thumb_size))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def prepare_png_output_dir(path: str | Path, *, clean: bool = False) -> Path:
    path = Path(path)
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    existing = sorted(path.glob("*.png"))
    if existing:
        raise RuntimeError(f"Output directory already contains png files: {path} ({len(existing)} found).")
    return path


def create_npz_from_sample_folder(sample_dir: str | Path, num: int = 50_000) -> str:
    sample_dir = Path(sample_dir)
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample = Image.open(sample_dir / f"{i:06d}.png")
        samples.append(np.asarray(sample).astype(np.uint8))
    arr = np.stack(samples)
    if arr.ndim != 4 or arr.shape[0] != num or arr.shape[-1] != 3:
        raise RuntimeError(f"Unexpected sample array shape: {arr.shape}")
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=arr)
    print(f"Saved .npz file to {npz_path} [shape={arr.shape}].")
    return npz_path
