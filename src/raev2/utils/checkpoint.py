"""Checkpoint save/load utilities for RAEv2 training."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def align_rope_buffers(model: torch.nn.Module, state: dict) -> tuple[dict, list[str]]:
    target_state = unwrap_model(model).state_dict()
    rope_keys = {
        "enc_rope.freqs_cos",
        "enc_rope.freqs_sin",
        "dec_rope.freqs_cos",
        "dec_rope.freqs_sin",
    }
    mismatched = [
        key
        for key, value in state.items()
        if key in rope_keys
        and key in target_state
        and value.shape != target_state[key].shape
    ]
    if not mismatched:
        return state, []

    aligned = state.copy()
    if hasattr(state, "_metadata"):
        aligned._metadata = state._metadata
    for key in mismatched:
        aligned[key] = target_state[key]
    return aligned, mismatched


def _atomic_torch_save(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def save_stage1_checkpoint(
    path: str,
    step: int,
    epoch: int,
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    disc: Optional[torch.nn.Module],
    disc_optimizer: Optional[torch.optim.Optimizer],
    disc_scheduler: Optional[LambdaLR],
) -> None:
    """Save Stage 1 training checkpoint (model + discriminator)."""
    state = {
        "step": step,
        "epoch": epoch,
        "model": unwrap_model(model).state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    if disc is not None:
        state["disc"] = unwrap_model(disc).state_dict()
    if disc_optimizer is not None:
        state["disc_optimizer"] = disc_optimizer.state_dict()
    if disc_scheduler is not None:
        state["disc_scheduler"] = disc_scheduler.state_dict()
    _atomic_torch_save(state, path)


def load_stage1_checkpoint(
    path: str,
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    disc: Optional[torch.nn.Module],
    disc_optimizer: Optional[torch.optim.Optimizer],
    disc_scheduler: Optional[LambdaLR],
) -> Tuple[int, int]:
    """Load Stage 1 training checkpoint. Returns (epoch, step)."""
    checkpoint = torch.load(path, map_location="cpu")
    unwrap_model(model).load_state_dict(checkpoint["model"], strict=True)
    ema_model.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if disc is not None and checkpoint.get("disc") is not None:
        unwrap_model(disc).load_state_dict(checkpoint["disc"], strict=True)
    if disc_optimizer is not None and checkpoint.get("disc_optimizer") is not None:
        disc_optimizer.load_state_dict(checkpoint["disc_optimizer"])
    if disc_scheduler is not None and checkpoint.get("disc_scheduler") is not None:
        disc_scheduler.load_state_dict(checkpoint["disc_scheduler"])
    return int(checkpoint.get("epoch", 0)), int(checkpoint.get("step", 0))


def save_stage2_checkpoint(
    path: str,
    step: int,
    epoch: int,
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
) -> None:
    """Save Stage 2 training checkpoint."""
    state = {
        "step": step,
        "epoch": epoch,
        "model": unwrap_model(model).state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    _atomic_torch_save(state, path)


def load_stage2_checkpoint(
    path: str,
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
) -> Tuple[int, int]:
    """Load Stage 2 training checkpoint. Returns (epoch, step)."""
    checkpoint = torch.load(path, map_location="cpu")
    unwrap_model(model).load_state_dict(checkpoint["model"], strict=True)
    ema_model.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return int(checkpoint.get("epoch", 0)), int(checkpoint.get("step", 0))


STAGE3_CHECKPOINT_FORMAT = "raev2_stage3_v1"


def _is_stage3_adapter_key(name: str) -> bool:
    return name.startswith("condition_control.") or name.endswith(
        (".lora_A", ".lora_B", ".lora_magnitude")
    )


def _stage3_trainable_names(model: torch.nn.Module) -> list[str]:
    return [name for name, param in unwrap_model(model).named_parameters() if param.requires_grad]


def _stage3_weight_mode(model: torch.nn.Module, requested: str) -> str:
    requested = str(requested).lower()
    if requested not in {"auto", "adapter", "full"}:
        raise ValueError(f"Unknown Stage3 checkpoint mode: {requested}")
    adapter_only = all(_is_stage3_adapter_key(name) for name in _stage3_trainable_names(model))
    if requested == "adapter" and not adapter_only:
        raise ValueError("Stage3 adapter checkpoint cannot contain trainable trunk parameters.")
    return ("adapter" if adapter_only else "full") if requested == "auto" else requested


def _stage3_state_dict(model: torch.nn.Module, mode: str) -> dict:
    state = unwrap_model(model).state_dict()
    return state if mode == "full" else {key: value for key, value in state.items() if _is_stage3_adapter_key(key)}


def _load_stage3_state_dict(model: torch.nn.Module, state: dict, mode: str) -> None:
    target = unwrap_model(model)
    if mode not in {"adapter", "full"}:
        raise RuntimeError(f"Unknown Stage3 checkpoint weight mode: {mode}")
    if mode == "full":
        target.load_state_dict(state, strict=True)
        return
    expected = {key for key in target.state_dict() if _is_stage3_adapter_key(key)}
    if set(state) != expected:
        raise RuntimeError("Stage3 adapter checkpoint keys do not match the current model.")
    target.load_state_dict(state, strict=False)


def save_stage3_checkpoint(
    path: str,
    step: int,
    epoch: int,
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    *,
    init_ckpt: str | None,
    mode: str = "auto",
    epoch_complete: bool = True,
    next_batch_idx: int | None = None,
    batches_per_epoch: int | None = None,
    grad_accum_steps: int | None = None,
    micro_batch_size: int | None = None,
    dataset_size: int | None = None,
    world_size: int | None = None,
    total_steps: int | None = None,
    rng_states: list | None = None,
) -> str:
    mode = _stage3_weight_mode(model, mode)
    if mode == "adapter" and not init_ckpt:
        raise ValueError("Stage3 adapter checkpoint requires training.init_ckpt.")
    state = {
        "format": STAGE3_CHECKPOINT_FORMAT,
        "weight_mode": mode,
        "step": int(step),
        "epoch": int(epoch),
        "epoch_complete": bool(epoch_complete),
        "init_ckpt": str(init_ckpt) if init_ckpt else None,
        "trainable_names": _stage3_trainable_names(model),
        "model": _stage3_state_dict(model, mode),
        "ema": _stage3_state_dict(ema_model, mode),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    progress_values = (
        next_batch_idx,
        batches_per_epoch,
        grad_accum_steps,
        micro_batch_size,
        dataset_size,
        world_size,
        total_steps,
        rng_states,
    )
    if any(value is not None for value in progress_values) and not all(
        value is not None for value in progress_values
    ):
        raise ValueError("Stage3 data progress must be provided as a complete set.")
    if all(value is not None for value in progress_values):
        state["data_progress"] = {
            "next_batch_idx": int(next_batch_idx),
            "batches_per_epoch": int(batches_per_epoch),
            "grad_accum_steps": int(grad_accum_steps),
            "micro_batch_size": int(micro_batch_size),
            "dataset_size": int(dataset_size),
            "world_size": int(world_size),
            "total_steps": int(total_steps),
            "rng_states": rng_states,
        }
    _atomic_torch_save(state, path)
    return mode


def load_stage3_checkpoint(
    path: str,
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    *,
    init_ckpt: str | None,
    return_progress: bool = False,
):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != STAGE3_CHECKPOINT_FORMAT:
        unwrap_model(model).load_state_dict(checkpoint["model"], strict=True)
        ema_model.load_state_dict(checkpoint["ema"], strict=True)
        mode = "legacy_full"
    else:
        epoch_complete = bool(checkpoint.get("epoch_complete", False))
        data_progress = checkpoint.get("data_progress") or {}
        if not epoch_complete:
            next_batch_idx = int(data_progress.get("next_batch_idx", 0))
            batches_per_epoch = int(data_progress.get("batches_per_epoch", 0))
            grad_accum_steps = int(data_progress.get("grad_accum_steps", 0))
            if not 0 < next_batch_idx < batches_per_epoch or grad_accum_steps <= 0:
                raise RuntimeError("Cannot resume an incomplete Stage3 checkpoint without a data position.")
            if next_batch_idx % grad_accum_steps != 0:
                raise RuntimeError("Stage3 checkpoint was not saved at an optimizer-step boundary.")
        mode = str(checkpoint["weight_mode"])
        if mode == "adapter" and checkpoint.get("init_ckpt") != (str(init_ckpt) if init_ckpt else None):
            raise RuntimeError("Stage3 adapter checkpoint uses a different Stage2 init checkpoint.")
        if checkpoint.get("trainable_names") != _stage3_trainable_names(model):
            raise RuntimeError("Stage3 trainable parameter order changed; optimizer resume is unsafe.")
        _load_stage3_state_dict(model, checkpoint["model"], mode)
        _load_stage3_state_dict(ema_model, checkpoint["ema"], mode)
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    result = (int(checkpoint.get("epoch", 0)), int(checkpoint.get("step", 0)), mode)
    if not return_progress:
        return result
    progress = dict(checkpoint.get("data_progress") or {})
    progress["epoch_complete"] = bool(checkpoint.get("epoch_complete", True))
    return (*result, progress)


def load_stage3_inference_checkpoint(
    checkpoint_or_path,
    model: torch.nn.Module,
    *,
    prefer_ema: bool,
    init_ckpt: str | None,
) -> str:
    checkpoint = (
        checkpoint_or_path
        if isinstance(checkpoint_or_path, dict)
        else torch.load(checkpoint_or_path, map_location="cpu", weights_only=False)
    )
    if checkpoint.get("format") != STAGE3_CHECKPOINT_FORMAT:
        if prefer_ema and "ema" in checkpoint:
            state, source = checkpoint["ema"], "ema"
        elif "model" in checkpoint:
            state, source = checkpoint["model"], "model"
        elif "ema" in checkpoint:
            state, source = checkpoint["ema"], "ema"
        else:
            state, source = checkpoint, "raw"
        unwrap_model(model).load_state_dict(state, strict=True)
        return source
    mode = str(checkpoint["weight_mode"])
    if mode == "adapter" and checkpoint.get("init_ckpt") != (str(init_ckpt) if init_ckpt else None):
        raise RuntimeError("Stage3 adapter checkpoint uses a different Stage2 init checkpoint.")
    source = "ema" if prefer_ema and checkpoint.get("ema") is not None else "model"
    _load_stage3_state_dict(model, checkpoint[source], mode)
    return f"{source}_{mode}"


__all__ = [
    "align_rope_buffers",
    "save_stage1_checkpoint",
    "load_stage1_checkpoint",
    "save_stage2_checkpoint",
    "load_stage2_checkpoint",
    "save_stage3_checkpoint",
    "load_stage3_checkpoint",
    "load_stage3_inference_checkpoint",
    "unwrap_model",
]
