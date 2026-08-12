#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (str(ROOT), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

from Dataset.datasets.utils import build_dataset_from_config as build_dataset
from raev2.configs import load_config
from raev2.utils.dist_utils import cleanup_distributed, setup_distributed
from raev2.utils.model_utils import instantiate_from_config
from utils.artifacts import prepare_png_output_dir


def parse_args():
    parser = argparse.ArgumentParser(description="RAEv2 Stage-1 reconstruction sampler.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-dir", default=None)
    parser.add_argument("--save-folder", default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--per-proc-batch-size", type=int, default=None)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default=None)
    parser.add_argument("--clean-output", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def save_rgb(path: Path, img: torch.Tensor) -> None:
    arr = img.clamp(0, 1).mul(255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(arr).save(path)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("RAEv2 Stage-1 reconstruction requires CUDA.")

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.set_grad_enabled(False)

    rank, world_size, device = setup_distributed()
    cfg = load_config(args.config)
    inf_cfg = cfg.get("inference", {})

    precision = args.precision or inf_cfg.get("precision", "bf16")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("Requested bf16 precision, but this CUDA device does not support bfloat16.")
    autocast_kwargs = {"enabled": precision == "bf16", "dtype": torch.bfloat16}

    num_samples = int(args.num_samples if args.num_samples is not None else inf_cfg.get("num_samples", 5000))
    per_proc_batch = int(args.per_proc_batch_size if args.per_proc_batch_size is not None else inf_cfg.get("per_proc_batch_size", 8))
    sample_dir = Path(args.sample_dir or inf_cfg.get("sample_dir", "experiments/raev2/stage1/reconstruction"))
    save_folder = args.save_folder or inf_cfg.get("save_folder", "dinov3l_k7_recon_val5k")
    out_dir = sample_dir / save_folder
    clean_output = bool(args.clean_output if args.clean_output is not None else inf_cfg.get("clean_output", False))

    if rank == 0:
        prepare_png_output_dir(out_dir, clean=clean_output)
        print(f"[raev2_stage1] cfg={args.config}")
        print(f"[raev2_stage1] out={out_dir} n={num_samples} bsz/rank={per_proc_batch} world={world_size}")
    if dist.is_initialized():
        dist.barrier()

    dataset = build_dataset(cfg.dataset)
    n_eval = min(num_samples, len(dataset))
    rank_indices = list(range(rank, n_eval, world_size))
    loader = DataLoader(
        Subset(dataset, rank_indices),
        batch_size=per_proc_batch,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    rae = instantiate_from_config(cfg.stage_1).to(device).eval()
    rae.requires_grad_(False)

    saved = 0
    with torch.inference_mode():
        for batch in loader:
            imgs = batch["img"].to(device, non_blocking=True)
            paths = batch["img_path"]
            with torch.amp.autocast(device_type=device.type, **autocast_kwargs):
                recon = rae(imgs).clamp(0, 1)
            for img, src_path in zip(recon, paths):
                save_rgb(out_dir / Path(str(src_path)).name, img.detach())
            saved += imgs.shape[0]
            if rank == 0 and saved % 64 == 0:
                print(f"[raev2_stage1] rank0_saved={saved}/{len(rank_indices)}")

    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        count = len([p for p in out_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("_")])
        print(f"[raev2_stage1] done files={count}")
        if count != n_eval:
            raise RuntimeError(
                f"Stage1 reconstruction count mismatch: expected {n_eval}, found {count} in {out_dir}."
            )
    cleanup_distributed()


if __name__ == "__main__":
    main()
