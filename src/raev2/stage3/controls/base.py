from __future__ import annotations

from importlib import import_module

import torch
import torch.nn as nn

from raev2.utils.config_utils import cfg_get, cfg_to_dict


def get_obj_from_str(target: str):
    module_name, obj_name = target.rsplit(".", 1)
    module = import_module(module_name)
    return getattr(module, obj_name)


def source_is_enabled(source_enabled) -> bool:
    if isinstance(source_enabled, bool):
        return source_enabled
    if isinstance(source_enabled, torch.Tensor):
        return True
    if isinstance(source_enabled, (list, tuple)):
        return any(bool(item) for item in source_enabled)
    return bool(source_enabled)


def source_mask_like(source_enabled, ref_z: torch.Tensor, ndim: int, *, dtype=None) -> torch.Tensor | None:
    dtype = ref_z.dtype if dtype is None else dtype
    if isinstance(source_enabled, bool):
        return None if source_enabled else torch.zeros((ref_z.shape[0],) + (1,) * (ndim - 1), device=ref_z.device, dtype=dtype)
    if isinstance(source_enabled, torch.Tensor):
        mask = source_enabled.to(device=ref_z.device, dtype=dtype)
    elif isinstance(source_enabled, (list, tuple)):
        mask = torch.as_tensor(source_enabled, device=ref_z.device, dtype=dtype)
    else:
        return None
    return mask.reshape(mask.shape[0], *([1] * (ndim - 1)))


def source_scale_like(source_scale, default: float, target: torch.Tensor, ndim: int) -> torch.Tensor | float:
    if source_scale is None:
        return float(default)
    if not isinstance(source_scale, torch.Tensor):
        return float(source_scale)
    scale = source_scale.to(device=target.device, dtype=target.dtype)
    if scale.ndim == 0:
        return scale
    return scale.reshape(scale.shape[0], *([1] * (ndim - 1)))


def source_any_enabled(source_enabled) -> bool:
    if isinstance(source_enabled, torch.Tensor):
        return bool(source_enabled.detach().to(dtype=torch.bool).any().cpu().item())
    if isinstance(source_enabled, (list, tuple)):
        return any(bool(v) for v in source_enabled)
    return source_is_enabled(source_enabled)


def sanitize_ref_z(ref_z: torch.Tensor, source_enabled) -> torch.Tensor:
    mask = source_mask_like(source_enabled, ref_z, ref_z.ndim, dtype=ref_z.dtype)
    if mask is None:
        return ref_z
    return torch.where(mask.to(dtype=torch.bool), ref_z, torch.zeros_like(ref_z))


def attn_mask_dtype(seq: torch.Tensor, attn_mask) -> torch.dtype:
    if attn_mask is not None:
        return attn_mask.dtype
    if torch.is_autocast_enabled():
        try:
            return torch.get_autocast_dtype("cuda")
        except (AttributeError, TypeError):
            return torch.get_autocast_gpu_dtype()
    return seq.dtype


class ConditionControl(nn.Module):
    """Base interface for Stage3 source-condition control modules.

    The training and sampling scripts should not know which concrete control
    method is active. They pass `ref_z` and `source_enabled`; this module decides
    whether to replace encoder input, inject residual fields, or do nothing.
    """

    def bind_model(self, model: nn.Module) -> None:
        object.__setattr__(self, "_bound_model", model)

    def init_from_config(self, model: nn.Module, init_cfg) -> dict:
        raise NotImplementedError(f"{type(self).__name__} does not support stage3.control.init policies")

    def encoder_input(self, x: torch.Tensor, *, ref_z=None, source_enabled=True, **kwargs) -> torch.Tensor:
        return x

    def begin_forward(self, *, x, t, ref_z=None, source_enabled=True, condition_kwargs=None, **kwargs) -> dict:
        return {}

    def encoder_sequence(self, seq: torch.Tensor, *, ref_z=None, source_enabled=True, state=None, **kwargs) -> torch.Tensor:
        return seq

    def after_encoder_block(
        self,
        seq: torch.Tensor,
        *,
        block_index: int,
        ref_z=None,
        source_enabled=True,
        state=None,
        **kwargs,
    ) -> torch.Tensor:
        return seq

    def encoder_rope(self, rope, *, seq: torch.Tensor, ref_z=None, source_enabled=True, state=None, **kwargs):
        return rope

    def encoder_block_modulation(
        self,
        block_index: int,
        *,
        seq: torch.Tensor,
        ref_z=None,
        source_enabled=True,
        state=None,
        **kwargs,
    ) -> torch.Tensor | None:
        return None

    def encoder_block_extra_kv(
        self,
        block_index: int,
        *,
        seq: torch.Tensor,
        ref_z=None,
        source_enabled=True,
        state=None,
        **kwargs,
    ):
        return None

    def encoder_attn_mask(
        self,
        attn_mask: torch.Tensor | None,
        *,
        seq: torch.Tensor,
        ref_z=None,
        source_enabled=True,
        state=None,
        **kwargs,
    ) -> torch.Tensor | None:
        return attn_mask

    def apply_seq(self, seq: torch.Tensor, *, ref_z=None, source_enabled=True, **kwargs) -> torch.Tensor:
        return seq

    def before_decoder(
        self,
        x_tokens: torch.Tensor,
        *,
        seq: torch.Tensor,
        ref_z=None,
        source_enabled=True,
        state=None,
        **kwargs,
    ) -> torch.Tensor:
        return x_tokens

    def decoder_condition(
        self,
        seq: torch.Tensor,
        *,
        x_tokens: torch.Tensor,
        block_index: int,
        ref_z=None,
        source_enabled=True,
        state=None,
        **kwargs,
    ) -> torch.Tensor:
        return seq

    def decoder_block_extra_kv(
        self,
        block_index: int,
        *,
        x_tokens: torch.Tensor,
        seq: torch.Tensor,
        ref_z=None,
        source_enabled=True,
        state=None,
        **kwargs,
    ):
        return None

    def after_decoder_block(
        self,
        x_tokens: torch.Tensor,
        *,
        seq: torch.Tensor,
        block_index: int,
        ref_z=None,
        source_enabled=True,
        state=None,
        **kwargs,
    ) -> torch.Tensor:
        return x_tokens

    def final_condition(
        self,
        seq: torch.Tensor,
        *,
        x_tokens: torch.Tensor,
        ref_z=None,
        source_enabled=True,
        state=None,
        **kwargs,
    ) -> torch.Tensor:
        return seq

    def prepare_decoder_block_delta(
        self,
        delta: torch.Tensor | None,
        *,
        c: torch.Tensor,
        block: nn.Module,
        block_index: int,
        **kwargs,
    ) -> torch.Tensor | None:
        return delta

    def prepare_final_layer_delta(
        self,
        delta: torch.Tensor | None,
        *,
        c: torch.Tensor,
        final_layer: nn.Module,
        **kwargs,
    ) -> torch.Tensor | None:
        return delta

    def consume_diagnostics(self) -> dict[str, float]:
        return {}


class NoConditionControl(ConditionControl):
    pass


def build_condition_control(config, model: nn.Module) -> ConditionControl:
    target = cfg_get(config, "target")
    control_type = cfg_get(config, "type")
    if target in (None, "", "none") and control_type in (None, "", "none"):
        control = NoConditionControl()
    else:
        if target in (None, ""):
            raise ValueError("Stage3 condition control config requires a 'target' when type is not 'none'.")
        params = cfg_to_dict(cfg_get(config, "params", {}))
        control = get_obj_from_str(str(target))(**params)
        if not isinstance(control, ConditionControl):
            raise TypeError(f"{target} must inherit ConditionControl.")
    control.bind_model(model)
    return control
