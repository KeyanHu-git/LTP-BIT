from __future__ import annotations

import math
import re
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_DEFAULT_TARGETS = ("attn.q", "attn.k", "attn.v", "attn.proj")
_TRUNK_BLOCK_RE = re.compile(r"(?:^|\.)blocks\.(\d+)\.")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        dev, dtype = base.weight.device, base.weight.dtype
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features, device=dev, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=dev, dtype=dtype))
        self.scaling = float(alpha) / float(rank)
        self.enabled = True
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        if not self.enabled:
            return base
        return base + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


def apply_lora(
    model: nn.Module,
    rank: int = 0,
    alpha: float | None = None,
    target_substrings: Iterable[str] = _DEFAULT_TARGETS,
    *,
    encoder_rank: int | None = None,
    encoder_alpha: float | None = None,
    decoder_rank: int | None = None,
    decoder_alpha: float | None = None,
) -> Tuple[int, int]:
    """Apply LoRA, optionally using width-normalized ranks for encoder/decoder.

    Legacy configs keep using ``rank``/``alpha`` for every matching layer.  When
    stage-specific values are provided, trunk blocks are routed by their index
    relative to ``model.num_enc_blocks``.  This keeps the policy explicit and
    avoids coupling rank selection to module traversal order.
    """
    rank = int(rank)
    alpha = float(rank if alpha is None else alpha)
    encoder_rank = rank if encoder_rank is None else int(encoder_rank)
    decoder_rank = rank if decoder_rank is None else int(decoder_rank)
    encoder_alpha = float(encoder_rank if encoder_alpha is None else encoder_alpha)
    decoder_alpha = float(decoder_rank if decoder_alpha is None else decoder_alpha)
    targets = tuple(target_substrings)
    num_enc_blocks = int(getattr(model, "num_enc_blocks", 0))
    num_dec_blocks = int(getattr(model, "num_dec_blocks", 0))

    for param in model.parameters():
        param.requires_grad_(False)

    n_replaced = 0
    for parent_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if not any(target in full_name for target in targets):
                continue
            layer_rank, layer_alpha = rank, alpha
            block_match = _TRUNK_BLOCK_RE.search(full_name)
            if block_match is not None:
                block_index = int(block_match.group(1))
                if block_index < num_enc_blocks:
                    layer_rank, layer_alpha = encoder_rank, encoder_alpha
                elif block_index < num_enc_blocks + num_dec_blocks:
                    layer_rank, layer_alpha = decoder_rank, decoder_alpha
            if layer_rank <= 0:
                continue
            setattr(module, child_name, LoRALinear(child, rank=layer_rank, alpha=layer_alpha))
            n_replaced += 1

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_replaced, n_trainable


def apply_lora_from_config(model: nn.Module, lora_cfg) -> Tuple[int, int]:
    """Apply either the legacy global-rank or staged-rank LoRA config."""
    lora_cfg = lora_cfg or {}
    rank = int(lora_cfg.get("rank", 0))
    encoder_rank = lora_cfg.get("encoder_rank")
    decoder_rank = lora_cfg.get("decoder_rank")
    enabled = rank > 0 or int(encoder_rank or 0) > 0 or int(decoder_rank or 0) > 0
    if not enabled:
        return 0, 0
    return apply_lora(
        model,
        rank=rank,
        alpha=lora_cfg.get("alpha"),
        target_substrings=lora_cfg.get("target_substrings", _DEFAULT_TARGETS),
        encoder_rank=encoder_rank,
        encoder_alpha=lora_cfg.get("encoder_alpha"),
        decoder_rank=decoder_rank,
        decoder_alpha=lora_cfg.get("decoder_alpha"),
    )


def set_lora_enabled(model: nn.Module, enabled: bool) -> int:
    count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.enabled = bool(enabled)
            count += 1
    return count
