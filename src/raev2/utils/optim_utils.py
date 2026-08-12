"""Optimizer and scheduler utilities using typed configs."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from raev2.configs import OptimizerConfig, SchedulerConfig


def _get(config, key: str, default=None):
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


class MuonAdamW(Optimizer):
    """Composite optimizer: Muon for 2D params, AdamW for the rest."""

    def __init__(self, muon_opt: Optimizer, adamw_opt: Optimizer):
        self._muon = muon_opt
        self._adamw = adamw_opt
        self.param_groups = muon_opt.param_groups + adamw_opt.param_groups
        self.defaults: Dict[str, Any] = {}

    @property
    def state(self) -> Dict:
        merged: Dict = {}
        merged.update(self._muon.state)
        merged.update(self._adamw.state)
        return merged

    def zero_grad(self, set_to_none: bool = False) -> None:
        self._muon.zero_grad(set_to_none=set_to_none)
        self._adamw.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None) -> None:
        self._muon.step(closure=closure)
        self._adamw.step(closure=closure)

    def state_dict(self) -> Dict[str, Any]:
        return {"muon": self._muon.state_dict(), "adamw": self._adamw.state_dict()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._muon.load_state_dict(state_dict["muon"])
        self._adamw.load_state_dict(state_dict["adamw"])
        self.param_groups = self._muon.param_groups + self._adamw.param_groups


def build_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    config: OptimizerConfig,
) -> tuple[Optimizer, str]:
    """Build optimizer from typed OptimizerConfig."""
    opt_type = _get(config, "type", "adamw")
    lr = float(_get(config, "lr", 2.0e-4))
    betas = tuple(_get(config, "betas", (0.9, 0.95)))
    weight_decay = float(_get(config, "weight_decay", 0.0))
    eps = float(_get(config, "eps", 1e-8))

    if opt_type == "adamw":
        optimizer = torch.optim.AdamW(
            parameters,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            fused=True,
        )
        msg = f"AdamW(lr={lr}, betas={betas}, wd={weight_decay})"

    elif opt_type == "gmuon":
        from gram_newton_schulz import Muon as GMuon

        params_list = list(parameters)
        muon_params = [p for p in params_list if p.ndim == 2]
        fallback_params = [p for p in params_list if p.ndim != 2]
        adamw_lr = _get(config, "adamw_lr")
        momentum = float(_get(config, "momentum", 0.95))
        nesterov = bool(_get(config, "nesterov", True))
        ns_preset = _get(config, "ns_coefficients_preset", "POLAR_EXPRESS_COEFFICIENTS")
        ns_use_kernels = bool(_get(config, "ns_use_kernels", False))

        adamw_opt = torch.optim.AdamW(
            fallback_params if fallback_params else [torch.nn.Parameter(torch.empty(0))],
            lr=float(adamw_lr) if adamw_lr is not None else lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
        )

        gmuon_opt = GMuon(
            muon_params,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            ns_coefficients_preset=ns_preset,
            ns_use_kernels=ns_use_kernels,
            adjust_lr="rms_norm",
        )

        optimizer = MuonAdamW(gmuon_opt, adamw_opt)
        msg = (f"GMuon(lr={lr}, momentum={momentum}, "
               f"preset={ns_preset}, kernels={ns_use_kernels}, "
               f"{len(muon_params)} 2D params, {len(fallback_params)} fallback)")

    else:
        raise ValueError(f"Unsupported optimizer '{opt_type}'. Choose from ['adamw', 'gmuon'].")

    return optimizer, msg


def build_scheduler(
    optimizer: Optimizer,
    steps_per_epoch: int,
    config: SchedulerConfig,
    state_dict: Optional[Dict[str, Any]] = None,
) -> tuple[LambdaLR, str]:
    """Build LR scheduler from typed SchedulerConfig."""
    # Compute steps from epochs or use direct step values
    warmup_steps_cfg = _get(config, "warmup_steps")
    decay_start_steps_cfg = _get(config, "decay_start_steps")
    decay_end_steps_cfg = _get(config, "decay_end_steps")
    schedule_type = _get(config, "type", "linear")
    base_lr = float(_get(config, "base_lr", 2.0e-4))
    final_lr = float(_get(config, "final_lr", 2.0e-5))
    warmup_from_zero = bool(_get(config, "warmup_from_zero", True))

    if warmup_steps_cfg is not None:
        warmup_steps = int(warmup_steps_cfg)
    else:
        warmup_steps = int(float(_get(config, "warmup_epochs", 1.0)) * steps_per_epoch)

    if decay_start_steps_cfg is not None:
        decay_start_steps = int(decay_start_steps_cfg)
    else:
        decay_start_epoch = _get(config, "decay_start_epoch")
        decay_start_steps = (
            int(float(decay_start_epoch) * steps_per_epoch)
            if decay_start_epoch is not None
            else warmup_steps
        )

    if decay_end_steps_cfg is not None:
        decay_end_steps = int(decay_end_steps_cfg)
    else:
        decay_end_steps = int(float(_get(config, "decay_end_epoch", 16.0)) * steps_per_epoch)

    warmup_steps = max(warmup_steps, 0)
    decay_start_steps = max(decay_start_steps, warmup_steps)
    decay_end_steps = max(decay_end_steps, decay_start_steps)
    total_decay_steps = max(decay_end_steps - decay_start_steps, 1)

    final_ratio = final_lr / base_lr if base_lr > 0 else 1.0

    # Set optimizer LR to base_lr
    for group in optimizer.param_groups:
        if group.get('name') not in ('encoder', 'decoder'):
            group["lr"] = base_lr

    if schedule_type == "linear":
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps if warmup_from_zero else 1.0
            if step < decay_start_steps:
                return 1.0
            if step >= decay_end_steps:
                return final_ratio
            progress = (step - decay_start_steps) / total_decay_steps
            return 1.0 - (1.0 - final_ratio) * progress

    elif schedule_type == "cosine":
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps if warmup_from_zero else 1.0
            if step < decay_start_steps:
                return 1.0
            if step >= decay_end_steps:
                return final_ratio
            progress = (step - decay_start_steps) / total_decay_steps
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return final_ratio + (1.0 - final_ratio) * cosine

    else:
        raise ValueError(f"Unsupported scheduler '{schedule_type}'. Choose from ['linear', 'cosine'].")

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    if state_dict is not None:
        scheduler.load_state_dict(state_dict)

    msg = (
        f"{schedule_type}(warmup={warmup_steps}, decay_start={decay_start_steps}, "
        f"decay_end={decay_end_steps}, lr={base_lr}->{final_lr})"
    )
    return scheduler, msg
