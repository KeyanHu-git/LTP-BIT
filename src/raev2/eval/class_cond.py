from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.distributed as dist

from ._image_io import prepare_eval_output_dirs, prepare_eval_output_dirs_local, save_rgb


__eval_exports__ = ["save_class_cond_fakes", "save_class_cond_fakes_ddp"]
__all__ = __eval_exports__


@torch.no_grad()
def save_class_cond_fakes(
    model_fn,
    sampler_fn,
    rae,
    latent_size,
    num_classes: int,
    null_label: int,
    sample_model_kwargs: Mapping[str, Any],
    use_guidance: bool,
    num_samples: int,
    gen_batch_size: int,
    device: torch.device,
    out_dir: str | Path,
    *,
    seed: int = 42,
    autocast_kwargs: Optional[dict] = None,
    clean_output: bool = True,
    fixed_label: Optional[int] = None,
    labels: Optional[torch.Tensor] = None,
) -> int:
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0.")
    if gen_batch_size < 1:
        raise ValueError("gen_batch_size must be >= 1.")
    if labels is not None and fixed_label is None:
        labels = labels.to(device=device, dtype=torch.long)
        if labels.ndim != 1:
            raise ValueError("labels must be a 1D tensor.")
        if labels.numel() != int(num_samples):
            raise ValueError("labels length must equal num_samples.")

    out_path = Path(out_dir)
    prepare_eval_output_dirs_local(out_path, None, clean=clean_output)

    latent_size = tuple(latent_size)
    autocast_kwargs = autocast_kwargs or {"enabled": True, "dtype": torch.bfloat16}
    generator = torch.Generator(device=device).manual_seed(int(seed))
    saved = 0
    while saved < num_samples:
        this_bs = min(int(gen_batch_size), num_samples - saved)
        z = torch.randn((this_bs,) + latent_size, device=device, generator=generator)
        if fixed_label is not None:
            y = torch.full((this_bs,), int(fixed_label), device=device, dtype=torch.long)
        elif labels is not None:
            y = labels[saved:saved + this_bs]
        else:
            y = torch.randint(0, num_classes, (this_bs,), device=device, generator=generator)
        if use_guidance:
            z = torch.cat([z, z], dim=0)
            y = torch.cat([y, torch.full((this_bs,), null_label, device=device, dtype=torch.long)], dim=0)
        model_kwargs = dict(sample_model_kwargs)
        model_kwargs["y"] = y

        with torch.amp.autocast(device_type=device.type, **autocast_kwargs):
            samples = sampler_fn(z, model_fn, **model_kwargs)[-1]
            if use_guidance:
                samples, _ = samples.chunk(2, dim=0)
            imgs = rae.decode(samples.float()).clamp(0, 1)

        for local_idx, img in enumerate(imgs.detach().cpu()):
            save_rgb(out_path / f"{saved + local_idx:06d}.png", img)
        saved += this_bs

    return int(num_samples)


@torch.no_grad()
def save_class_cond_fakes_ddp(
    model_fn,
    sampler_fn,
    rae,
    latent_size,
    num_classes: int,
    null_label: int,
    sample_model_kwargs: Mapping[str, Any],
    use_guidance: bool,
    num_samples: int,
    gen_batch_size: int,
    device: torch.device,
    out_dir: str | Path,
    *,
    seed: int = 42,
    autocast_kwargs: Optional[dict] = None,
    clean_output: bool = True,
    fixed_label: Optional[int] = None,
    labels: Optional[torch.Tensor] = None,
) -> int:
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0.")
    if gen_batch_size < 1:
        raise ValueError("gen_batch_size must be >= 1.")
    if labels is not None:
        if rank != 0:
            raise ValueError("labels should only be provided on rank 0.")
        labels = labels.to(device=device, dtype=torch.long)
        if labels.ndim != 1:
            raise ValueError("labels must be a 1D tensor.")
        if labels.numel() != int(num_samples):
            raise ValueError("labels length must equal num_samples.")

    out_path = Path(out_dir)
    prepare_eval_output_dirs(out_path, None, clean=clean_output)

    latent_size = tuple(latent_size)
    autocast_kwargs = autocast_kwargs or {"enabled": True, "dtype": torch.bfloat16}
    generator = torch.Generator(device=device).manual_seed(int(seed)) if rank == 0 else None
    saved = 0
    while saved < num_samples:
        this_bs = min(int(gen_batch_size), num_samples - saved)
        per_rank = math.ceil(this_bs / world_size)
        padded_bs = per_rank * world_size
        local_z = torch.empty((per_rank,) + latent_size, device=device, dtype=torch.float32)
        local_y = torch.empty((per_rank,), device=device, dtype=torch.long)

        if rank == 0:
            z_full = torch.randn((this_bs,) + latent_size, device=device, generator=generator)
            if labels is not None:
                y_full = labels[saved:saved + this_bs]
            elif fixed_label is None:
                y_full = torch.randint(0, num_classes, (this_bs,), device=device, generator=generator)
            else:
                y_full = torch.full((this_bs,), int(fixed_label), device=device, dtype=torch.long)
            if padded_bs > this_bs:
                z_full = torch.cat([z_full, torch.zeros((padded_bs - this_bs,) + latent_size, device=device)], dim=0)
                y_full = torch.cat([y_full, torch.zeros((padded_bs - this_bs,), device=device, dtype=torch.long)], dim=0)
            scatter_z = list(z_full.split(per_rank, dim=0))
            scatter_y = list(y_full.split(per_rank, dim=0))
        else:
            scatter_z = None
            scatter_y = None

        if dist.is_initialized():
            dist.scatter(local_z, scatter_list=scatter_z, src=0)
            dist.scatter(local_y, scatter_list=scatter_y, src=0)
        else:
            local_z = scatter_z[0]
            local_y = scatter_y[0]

        valid = max(0, min(per_rank, this_bs - rank * per_rank))
        if valid > 0:
            model_kwargs = dict(sample_model_kwargs)
            z_in, y_in = local_z, local_y
            if use_guidance:
                z_in = torch.cat([z_in, z_in], dim=0)
                y_in = torch.cat([y_in, torch.full((per_rank,), null_label, device=device, dtype=torch.long)], dim=0)
            model_kwargs["y"] = y_in

            with torch.amp.autocast(device_type=device.type, **autocast_kwargs):
                samples = sampler_fn(z_in, model_fn, **model_kwargs)[-1]
                if use_guidance:
                    samples, _ = samples.chunk(2, dim=0)
                imgs = rae.decode(samples.float()).clamp(0, 1)

            for local_idx, img in enumerate(imgs[:valid].detach().cpu()):
                global_idx = saved + rank * per_rank + local_idx
                save_rgb(out_path / f"{global_idx:06d}.png", img)
        if dist.is_initialized():
            dist.barrier()
        saved += this_bs

    if dist.is_initialized():
        dist.barrier()
    return int(num_samples)
