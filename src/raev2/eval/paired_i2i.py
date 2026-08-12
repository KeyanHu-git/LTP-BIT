from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist

from raev2.utils.guidance_utils import duplicate_condition_kwargs

from ._image_io import prepare_eval_output_dirs, save_rgb, save_triplet_side_by_side


__eval_exports__ = ["paired_output_names", "save_paired_i2i_fakes_ddp"]
__all__ = __eval_exports__


def paired_output_names(dataset, count: int) -> list[str]:
    target_paths = getattr(dataset, "target_paths", None)
    n = min(int(count), len(dataset))
    if target_paths is None:
        return [f"{index:06d}.png" for index in range(n)]
    return [Path(target_paths[index]).name for index in range(n)]


def _load_pairs(val_dataset, indices: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    targets, sources = [], []
    for idx in indices:
        target, source, _ = val_dataset[idx]
        targets.append(target)
        sources.append(source)
    return (
        torch.stack(targets, dim=0).to(device, non_blocking=True).clamp(0.0, 1.0),
        torch.stack(sources, dim=0).to(device, non_blocking=True).clamp(0.0, 1.0),
    )


@torch.no_grad()
def save_paired_i2i_fakes_ddp(
    model_fn,
    rae,
    sampler_fn,
    val_dataset,
    num_eval_samples: int,
    gen_batch_size: int,
    device: torch.device,
    latent_size,
    out_dir: str | Path,
    *,
    fixed_label: int = 0,
    seed: int = 42,
    preview_dir: Optional[str | Path] = None,
    preview_max_images: int = 16,
    autocast_kwargs: Optional[dict] = None,
    clean_output: bool = True,
    sample_model_kwargs: Optional[dict] = None,
    null_label: Optional[int] = None,
    use_guidance: bool = False,
) -> int:
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    requested_samples = int(num_eval_samples)
    if requested_samples <= 0:
        raise ValueError("num_eval_samples must be > 0.")
    if gen_batch_size < 1:
        raise ValueError("gen_batch_size must be >= 1.")
    n = min(requested_samples, len(val_dataset))
    if n <= 0:
        raise ValueError("val_dataset must contain at least one sample.")

    out_path = Path(out_dir)
    preview_path = Path(preview_dir) if preview_dir is not None else None
    prepare_eval_output_dirs(out_path, preview_path, clean=clean_output)

    output_names = paired_output_names(val_dataset, n)
    latent_size = tuple(latent_size)
    autocast_kwargs = autocast_kwargs or {"enabled": True, "dtype": torch.bfloat16}
    sample_model_kwargs = dict(sample_model_kwargs or {})
    generator = torch.Generator(device=device).manual_seed(seed) if rank == 0 else None

    saved = 0
    while saved < n:
        this_bs = min(int(gen_batch_size), n - saved)
        per_rank = math.ceil(this_bs / world_size)
        padded_bs = per_rank * world_size
        local_z = torch.empty((per_rank,) + latent_size, device=device)
        if rank == 0:
            z_full = torch.randn((this_bs,) + latent_size, device=device, generator=generator)
            if padded_bs > this_bs:
                pad = torch.zeros((padded_bs - this_bs,) + latent_size, device=device)
                z_full = torch.cat([z_full, pad], dim=0)
            scatter_z = list(z_full.split(per_rank, dim=0))
        else:
            scatter_z = None

        if dist.is_initialized():
            dist.scatter(local_z, scatter_list=scatter_z, src=0)
        else:
            local_z = scatter_z[0]

        start = saved + rank * per_rank
        end = min(saved + (rank + 1) * per_rank, saved + this_bs)
        batch_indices = list(range(start, end))
        if batch_indices:
            target_imgs, source_imgs = _load_pairs(val_dataset, batch_indices, device)
            context = torch.full((len(batch_indices),), fixed_label, device=device, dtype=torch.long)
            kwargs = dict(sample_model_kwargs)
            kwargs["attn_mask"] = None
            kwargs["ref_z"] = rae.encode(source_imgs)
            z_start = local_z[:len(batch_indices)]
            if use_guidance:
                if null_label is None:
                    raise ValueError("null_label is required when use_guidance=True.")
                batch_size = len(batch_indices)
                z_start = torch.cat([z_start, z_start], dim=0)
                kwargs = duplicate_condition_kwargs(kwargs, batch_size)
                kwargs["context"] = torch.cat(
                    [context, torch.full_like(context, int(null_label))],
                    dim=0,
                )
            else:
                kwargs["context"] = context
            with torch.amp.autocast(device_type=device.type, **autocast_kwargs):
                z_pred = sampler_fn(z_start, model_fn, **kwargs)[-1]
                if use_guidance:
                    z_pred, _ = z_pred.chunk(2, dim=0)
                fake_imgs = rae.decode(z_pred.float()).clamp(0.0, 1.0)

            for global_idx, source, fake, target in zip(batch_indices, source_imgs.cpu(), fake_imgs.cpu(), target_imgs.cpu()):
                fname = output_names[global_idx]
                save_rgb(out_path / fname, fake)
                if preview_path is not None and global_idx < preview_max_images:
                    save_triplet_side_by_side(preview_path / f"sample_{global_idx:03d}.png", source, fake, target)
        if dist.is_initialized():
            dist.barrier()
        saved += this_bs

    if dist.is_initialized():
        dist.barrier()
    return n
