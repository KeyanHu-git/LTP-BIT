from __future__ import annotations

import re

import torch
import torch.nn as nn

from raev2.utils.config_utils import cfg_get


def _enabled_control(control_cfg) -> bool:
    if control_cfg is None:
        return False
    target = cfg_get(control_cfg, "target")
    control_type = cfg_get(control_cfg, "type")
    return target not in (None, "", "none") or control_type not in (None, "", "none")


def validate_stage3_data_mix(cfg) -> dict[str, float]:
    stage3_cfg = cfg.get("stage3", {}) if hasattr(cfg, "get") else {}
    policy = cfg_get(stage3_cfg, "data_mix", {})
    ratios = {
        "paired_ratio": float(cfg_get(policy, "paired_ratio", 1.0)),
        "prior_ratio": float(cfg_get(policy, "prior_ratio", 0.0)),
        "source_dropout_ratio": float(cfg_get(policy, "source_dropout_ratio", 0.0)),
    }
    for name in ("paired_ratio", "prior_ratio"):
        value = ratios[name]
        if value < 0:
            raise ValueError(f"stage3.data_mix.{name} must be >= 0.")
    if not 0.0 <= ratios["source_dropout_ratio"] <= 1.0:
        raise ValueError("stage3.data_mix.source_dropout_ratio must be in [0, 1].")
    if ratios["paired_ratio"] + ratios["prior_ratio"] <= 0:
        raise ValueError("At least one stage3.data_mix ratio must be positive.")
    if ratios["paired_ratio"] != 1.0 or ratios["prior_ratio"] != 0.0:
        raise NotImplementedError(
            "Stage3 data_mix is currently paired-only; use paired_ratio=1.0, "
            "and prior_ratio=0.0 until grouped prior losses are implemented."
        )
    return ratios


def apply_freeze_policy(model: nn.Module, freeze_cfg, logger=None) -> None:
    if freeze_cfg is None:
        return
    base_model = bool(cfg_get(freeze_cfg, "base_model", False))
    trainable_patterns = list(cfg_get(freeze_cfg, "trainable", []) or [])
    frozen_patterns = list(cfg_get(freeze_cfg, "frozen", []) or [])
    trainable_allowlist = list(cfg_get(freeze_cfg, "trainable_allowlist", []) or [])
    assert_trainable_nonzero = bool(cfg_get(freeze_cfg, "assert_trainable_nonzero", False))

    if base_model:
        for param in model.parameters():
            param.requires_grad_(False)

    if trainable_patterns:
        compiled = [re.compile(pattern) for pattern in trainable_patterns]
        for name, param in model.named_parameters():
            if any(pattern.search(name) for pattern in compiled):
                param.requires_grad_(True)

    if frozen_patterns:
        compiled = [re.compile(pattern) for pattern in frozen_patterns]
        for name, param in model.named_parameters():
            if any(pattern.search(name) for pattern in compiled):
                param.requires_grad_(False)

    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    if assert_trainable_nonzero and not trainable_names:
        raise ValueError("Stage3 freeze policy left no trainable parameters.")
    if trainable_allowlist:
        compiled = [re.compile(pattern) for pattern in trainable_allowlist]
        unexpected = [name for name in trainable_names if not any(pattern.search(name) for pattern in compiled)]
        if unexpected:
            raise ValueError(
                "Stage3 freeze policy left unexpected trainable parameters outside allowlist: "
                + ", ".join(unexpected[:20])
            )

    if logger is not None:
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        logger.info(f"Stage3 freeze policy: trainable={n_trainable / 1e6:.3f}M / total={n_total / 1e6:.3f}M")


def inject_stage3_control_config(cfg) -> None:
    stage3_cfg = cfg.get("stage3", {}) if hasattr(cfg, "get") else {}
    control_cfg = cfg_get(stage3_cfg, "control")
    if not _enabled_control(control_cfg):
        return
    params = cfg.stage_2.params
    if "condition_control" in params:
        existing = params.condition_control
        if existing != control_cfg:
            raise ValueError("Both stage_2.params.condition_control and stage3.control are set with different values.")
        return
    params.condition_control = control_cfg


def stage3_paired_training_losses(
    transport,
    model,
    x1,
    *,
    model_kwargs,
    model_kwargs_null,
    source_kwargs,
    cfg_dropout_prob: float,
    source_dropout_ratio: float = 0.0,
    base_model_coeff: float = 1.0,
):
    """Stage3 paired loss with independent label and source dropout policies."""
    from raev2.stage2.utils import apply_cfg_dropout

    model_kwargs, _ = apply_cfg_dropout(model_kwargs, model_kwargs_null, cfg_dropout_prob)
    source_kwargs = dict(source_kwargs)
    t, x0, x1, pure_noise_mask = transport.sample(x1, return_pure_noise_mask=True)
    if source_dropout_ratio > 0.0:
        enabled = pure_noise_mask | (
            torch.rand(pure_noise_mask.shape, device=pure_noise_mask.device, dtype=torch.float32)
            >= source_dropout_ratio
        )
        source_kwargs["source_enabled"] = enabled
    else:
        enabled = None
    model_kwargs.update(source_kwargs)
    terms = transport.training_losses(
        model,
        x1,
        model_kwargs=model_kwargs,
        model_kwargs_null=model_kwargs,
        base_model_coeff=base_model_coeff,
        cfg_dropout_prob=0.0,
        sampled=(t, x0, x1),
    )
    if enabled is not None:
        terms["loss_source_dropout_ratio"] = (~enabled).float().detach().mean()
    return terms
