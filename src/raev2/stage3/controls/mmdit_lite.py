"""MMDiT-lite: a reduced-width dedicated ref stream over the first N encoder blocks.

The SAR ref tokens live in their own narrow stream (width d, own attention+MLP,
interactive: they read the current patch hiddens each block). Per dual block the
stream supplies K/V columns - projected to the main width, main k_norm and patch
RoPE applied - into the frozen encoder attention via the extra_kv port. A
per-block (or pair-shared, see gate_per_block) logit gate suppresses the new
columns at init (config pins -4, scale-
reparameterized so Adam can travel the logit range within a 40-epoch budget).
Ref tokens never enter the sequence, so cond tokens keep their columns and
source-off is the exact frozen path by construction.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed

from raev2.stage2.models.model_utils import RMSNorm, RoPE, SwiGLUFFN, apply_rope_table

from .base import ConditionControl, sanitize_ref_z, source_any_enabled


class RefStreamBlock(nn.Module):
    """One dual-stream step: ref self+cross attention (over [ref, patch->d]) + MLP."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.ctx_norm = RMSNorm(dim)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        self.mlp = SwiGLUFFN(dim, int(2 / 3 * dim * mlp_ratio))

    def forward(self, ref: torch.Tensor, patch_d: torch.Tensor, rope_cos, rope_sin) -> torch.Tensor:
        B, N, _ = ref.shape
        normed = self.norm1(ref)
        ctx = torch.cat([normed, self.ctx_norm(patch_d)], dim=1)
        q = self.q(normed).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k, v = self.kv(ctx).reshape(B, 2 * N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q = self.q_norm(q)
        k = self.k_norm(k)
        # position-aligned: ref grid and patch grid share the same angle rows
        q = apply_rope_table(q, rope_cos, rope_sin)
        k = apply_rope_table(k, torch.cat([rope_cos, rope_cos], dim=-2), torch.cat([rope_sin, rope_sin], dim=-2))
        out = nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 2, 1, 3).reshape(B, N, -1)
        ref = ref + self.proj(out)
        ref = ref + self.mlp(self.norm2(ref))
        return ref


class DynamicKVGate(nn.Module):
    """Zero-initialized, token-wise logit correction for source K/V columns."""

    def __init__(self, dim: int, scale: float = 1.0):
        super().__init__()
        if float(scale) <= 0.0:
            raise ValueError("dynamic_gate_scale must be positive.")
        self.scale = float(scale)
        self.norm = RMSNorm(dim)
        self.proj = nn.Linear(dim, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, ref: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.tanh(self.proj(self.norm(ref))).squeeze(-1)


class MMDiTLiteStack(nn.Module):
    def __init__(
        self,
        stream_width: int = 480,
        stream_heads: int = 8,
        dual_blocks: int = 14,
        gate_init: float = -4.0,
        gate_scale: float = 50.0,
        use_source_time: bool = True,
        use_source_context: bool = True,
        block_stride: int = 1,
        blocks_per_ref: int = 1,
        gate_per_block: bool = True,
        dynamic_gate: bool = False,
        dynamic_gate_scale: float = 1.0,
    ):
        super().__init__()
        if int(block_stride) < 1 or int(blocks_per_ref) < 1:
            raise ValueError("block_stride and blocks_per_ref must be >= 1.")
        if int(stream_width) <= 0 or int(stream_heads) <= 0 or int(stream_width) % int(stream_heads) != 0:
            raise ValueError("stream_width must be positive and divisible by stream_heads.")
        if float(gate_scale) == 0.0:
            raise ValueError("gate_scale must be non-zero.")
        if block_stride > 1 and blocks_per_ref > 1:
            raise ValueError("block_stride and blocks_per_ref are mutually exclusive.")
        self.block_stride = int(block_stride)
        self.blocks_per_ref = int(blocks_per_ref)
        self.gate_per_block = bool(gate_per_block)
        self.stream_width = int(stream_width)
        self.stream_heads = int(stream_heads)
        self.dual_blocks = int(dual_blocks)
        self.gate_init = float(gate_init)
        # Same travel reparameterization as the inline ref gate: Adam moves a raw
        # scalar ~lr per step, so the fixed scale buys the logit range in-budget.
        self.gate_scale = float(gate_scale)
        self.use_source_time = bool(use_source_time)
        self.use_source_context = bool(use_source_context)
        self.dynamic_gate_enabled = bool(dynamic_gate)
        self.dynamic_gate_scale = float(dynamic_gate_scale)

        self.ref_embedder = None
        self.t_proj = None
        self.ctx_proj = None
        self.patch_proj = None
        self.blocks = None
        self.to_main_k = None
        self.to_main_v = None
        self.gates = None
        self.dynamic_gate = None
        self.ref_rope = None
        self.num_patches = None
        self._bound_model = None

    def bind_model(self, model: nn.Module) -> None:
        object.__setattr__(self, "_bound_model", model)
        input_size = getattr(model.s_embedder, "img_size", None)
        if input_size is None:
            patch_size = model.s_patch_size[0] if isinstance(model.s_patch_size, (tuple, list)) else model.s_patch_size
            input_size = int(int(model.s_embedder.num_patches) ** 0.5) * int(patch_size)
        main_hidden = int(model.enc_hidden_size)
        d = self.stream_width
        self.num_patches = int(model.s_embedder.num_patches)
        coverage = self.dual_blocks * max(self.block_stride, self.blocks_per_ref)
        if coverage > int(model.num_enc_blocks):
            raise ValueError("mmdit_lite placement span exceeds encoder depth.")
        self.ref_embedder = PatchEmbed(input_size, model.s_patch_size, model.in_channels, d)
        self.t_proj = nn.Linear(main_hidden, d) if self.use_source_time else None
        self.ctx_proj = nn.Linear(main_hidden, d) if self.use_source_context else None
        self.patch_proj = nn.Linear(main_hidden, d)
        self.blocks = nn.ModuleList(
            RefStreamBlock(d, self.stream_heads) for _ in range(self.dual_blocks - 1)
        )
        self.to_main_k = nn.ModuleList(nn.Linear(d, main_hidden) for _ in range(self.dual_blocks))
        self.to_main_v = nn.ModuleList(nn.Linear(d, main_hidden) for _ in range(self.dual_blocks))
        num_gates = self.dual_blocks * (self.blocks_per_ref if self.gate_per_block else 1)
        self.gates = nn.Parameter(torch.full((num_gates,), self.gate_init / self.gate_scale))
        if self.dynamic_gate_enabled:
            self.dynamic_gate = DynamicKVGate(d, self.dynamic_gate_scale)
        self.ref_rope = RoPE(d // self.stream_heads, self.num_patches, 0)

    def _serving_first(self, state: int) -> int:
        return state * (self.blocks_per_ref if self.blocks_per_ref > 1 else self.block_stride)

    def _source_pool(self, state: int) -> list:
        first = self._serving_first(state)
        return [first + j for j in range(self.blocks_per_ref)] if self.blocks_per_ref > 1 else [first]

    @torch.no_grad()
    def init_from_trunk(self, model: nn.Module, *, generator, positions) -> dict:
        """Whole-head-copy init from the frozen trunk. Policy (positions/seed) comes
        from the caller; structural mapping is fully introspected. Random draws
        are limited to each stream block's source pick and MLP neuron subset."""
        head_dim = int(model.blocks[0].attn.head_dim)
        n_heads = int(model.blocks[0].attn.num_heads)
        if self.stream_width != self.stream_heads * head_dim:
            raise ValueError(f"trunk_headcopy requires stream head_dim to match the trunk: {self.stream_width} != {self.stream_heads}x{head_dim}")
        positions, sel = _select_head_channels(n_heads, head_dim, positions)
        d = len(sel)
        embed = model.s_embedder.proj
        self.ref_embedder.proj.weight.copy_(embed.weight[sel])
        self.ref_embedder.proj.bias.copy_(embed.bias[sel])
        select = torch.zeros(d, embed.weight.shape[0], device=embed.weight.device)
        select[torch.arange(d), sel] = 1.0
        for proj in (self.patch_proj, self.t_proj, self.ctx_proj):
            if proj is not None:
                proj.weight.copy_(select)
                proj.bias.zero_()
        src_choices = []
        for i, sb in enumerate(self.blocks):
            pool = self._source_pool(i + 1)
            src = pool[int(torch.randint(len(pool), (1,), generator=generator))]
            src_choices.append(src)
            tb = model.blocks[src]
            _headcopy_ref_block(sb, tb, sel, generator)
        for k in range(self.dual_blocks):
            tb = model.blocks[self._serving_first(k)]
            self.to_main_k[k].weight.copy_(tb.attn.k.weight[:, sel])
            self.to_main_k[k].bias.copy_(tb.attn.k.bias)
            self.to_main_v[k].weight.copy_(tb.attn.v.weight[:, sel])
            self.to_main_v[k].bias.copy_(tb.attn.v.bias)
        return {"positions": positions, "src_choices": src_choices}

    def extra_kv(
        self,
        block_index: int,
        *,
        seq: torch.Tensor,
        ref_z=None,
        source_enabled=True,
        source_scale=None,
        t=None,
        condition_kwargs=None,
        state=None,
        **kwargs,
    ):
        if ref_z is None or not source_any_enabled(source_enabled):
            return None
        if self.block_stride > 1 and block_index % self.block_stride != 0:
            return None
        ref_index = block_index // (self.block_stride if self.block_stride > 1 else self.blocks_per_ref)
        if ref_index >= self.dual_blocks:
            return None
        scale = self._resolve_scale(source_scale, ref_z)
        if scale is None:
            return None
        ref = self._ref_state(ref_index, seq=seq, ref_z=ref_z, source_enabled=source_enabled,
                              t=t, condition_kwargs=condition_kwargs, state=state)
        model = self._require_bound()
        attn = model.blocks[block_index].attn
        B, N = ref.shape[0], self.num_patches
        ek = self.to_main_k[ref_index](ref).reshape(B, N, attn.num_heads, attn.head_dim).permute(0, 2, 1, 3)
        ev = self.to_main_v[ref_index](ref).reshape(B, N, attn.num_heads, attn.head_dim).permute(0, 2, 1, 3)
        ek = attn.k_norm(ek)
        ek = apply_rope_table(ek, model.enc_rope.freqs_cos[:N], model.enc_rope.freqs_sin[:N])
        if not isinstance(scale, float) or scale != 1.0:
            ev = ev * (scale if isinstance(scale, float) else scale.view(B, 1, 1, 1).to(dtype=ev.dtype))
        gate_index = block_index if (self.blocks_per_ref > 1 and self.gate_per_block) else ref_index
        ebias = (self.gates[gate_index] * self.gate_scale).to(dtype=torch.float32).expand(B, 1, 1, N)
        if self.dynamic_gate is not None:
            ebias = ebias + self.dynamic_gate(ref).to(dtype=torch.float32).view(B, 1, 1, N)
        disabled = self._disabled_rows(source_enabled, ref_z)
        if disabled is not None:
            neg = torch.finfo(torch.float32).min / 2
            ebias = ebias + disabled.view(B, 1, 1, 1).to(dtype=torch.float32) * neg
        return ek, ev, ebias

    def _ref_state(self, ref_index, *, seq, ref_z, source_enabled, t, condition_kwargs, state):
        if ref_index == 0 or "mmdit_ref_state" not in state:
            model = self._require_bound()
            ref_z = sanitize_ref_z(ref_z, source_enabled)
            ref = self.ref_embedder(ref_z)
            if self.t_proj is not None:
                if t is None:
                    raise ValueError("MMDiTLiteStack(use_source_time=True) requires timestep t.")
                t_emb_base, _ = model.t_embedder(t, return_base_embed=True)
                ref = ref + self.t_proj(t_emb_base.to(dtype=ref.dtype))
            if self.ctx_proj is not None:
                condition_kwargs = condition_kwargs or {}
                if "context" not in condition_kwargs:
                    raise ValueError("MMDiTLiteStack(use_source_context=True) requires condition_kwargs['context'].")
                ctx = model.ctx_embedder(condition_kwargs["context"]).mean(dim=1, keepdim=True)
                ref = ref + self.ctx_proj(ctx.to(dtype=ref.dtype))
            state["mmdit_ref_state"] = ref
            state["mmdit_ref_block"] = 0
        # blocks must be visited consecutively; a gap would evolve with stale seq
        if state["mmdit_ref_block"] < ref_index - 1:
            raise RuntimeError(
                f"mmdit extra_kv called for ref block {ref_index} after ref block {state['mmdit_ref_block']}"
            )
        ref = state["mmdit_ref_state"]
        while state["mmdit_ref_block"] < ref_index:
            i = state["mmdit_ref_block"]
            patch_d = self.patch_proj(seq[:, : self.num_patches, :].to(dtype=ref.dtype))
            cos = self.ref_rope.freqs_cos[: self.num_patches]
            sin = self.ref_rope.freqs_sin[: self.num_patches]
            ref = self.blocks[i](ref, patch_d, cos, sin)
            state["mmdit_ref_state"] = ref
            state["mmdit_ref_block"] = i + 1
        return ref

    @staticmethod
    def _resolve_scale(source_scale, ref_z):
        """None -> injection off; float/tensor -> multiplier on the V columns."""
        if source_scale is None:
            return 1.0
        if isinstance(source_scale, torch.Tensor):
            if bool((source_scale.detach() == 0).all().cpu().item()):
                return None
            return source_scale.reshape(ref_z.shape[0])
        value = float(source_scale)
        return None if value == 0.0 else value

    @staticmethod
    def _disabled_rows(source_enabled, ref_z) -> torch.Tensor | None:
        if isinstance(source_enabled, bool) or ref_z is None:
            return None
        if isinstance(source_enabled, torch.Tensor):
            disabled = ~source_enabled.to(device=ref_z.device, dtype=torch.bool).reshape(-1)
        elif isinstance(source_enabled, (list, tuple)):
            disabled = ~torch.as_tensor(source_enabled, device=ref_z.device, dtype=torch.bool).reshape(-1)
        else:
            return None
        return disabled if bool(disabled.any()) else None

    def _require_bound(self):
        if self._bound_model is None:
            raise RuntimeError("MMDiTLiteStack.bind_model() must be called before forward.")
        return self._bound_model


def _headcopy_ref_block(sb, tb, sel: torch.Tensor, generator) -> None:
    """Copy selected-head submatrices of a trunk block into a RefStreamBlock
    (q/kv/proj/norms + an MLP neuron subset)."""
    sb.q.weight.copy_(tb.attn.q.weight[sel][:, sel])
    sb.q.bias.copy_(tb.attn.q.bias[sel])
    sb.kv.weight.copy_(torch.cat([tb.attn.k.weight[sel][:, sel], tb.attn.v.weight[sel][:, sel]]))
    sb.kv.bias.copy_(torch.cat([tb.attn.k.bias[sel], tb.attn.v.bias[sel]]))
    sb.proj.weight.copy_(tb.attn.proj.weight[sel][:, sel])
    sb.proj.bias.copy_(tb.attn.proj.bias[sel])
    sb.q_norm.weight.copy_(tb.attn.q_norm.weight)
    sb.k_norm.weight.copy_(tb.attn.k_norm.weight)
    sb.norm1.weight.copy_(tb.norm1.weight[sel])
    sb.norm2.weight.copy_(tb.norm2.weight[sel])
    sb.ctx_norm.weight.copy_(tb.norm1.weight[sel])
    neurons = torch.randperm(tb.mlp.w1.out_features, generator=generator)[: sb.mlp.w1.out_features]
    sb.mlp.w1.weight.copy_(tb.mlp.w1.weight[neurons][:, sel])
    sb.mlp.w1.bias.copy_(tb.mlp.w1.bias[neurons])
    sb.mlp.w2.weight.copy_(tb.mlp.w2.weight[neurons][:, sel])
    sb.mlp.w2.bias.copy_(tb.mlp.w2.bias[neurons])
    sb.mlp.w3.weight.copy_(tb.mlp.w3.weight[sel][:, neurons])
    sb.mlp.w3.bias.copy_(tb.mlp.w3.bias[sel])


def _select_head_channels(n_heads: int, head_dim: int, positions):
    if positions is None:
        raise ValueError("trunk_headcopy requires explicit head positions")
    positions = [int(p) for p in positions]
    if len(set(positions)) != len(positions) or any(p < 0 or p >= n_heads for p in positions):
        raise ValueError(f"invalid head positions for {n_heads} heads: {positions}")
    channels = torch.cat([torch.arange(p * head_dim, (p + 1) * head_dim) for p in positions])
    return positions, channels


def _copy_projector_submatrix(dst: nn.Linear, src: nn.Module, out_sel: torch.Tensor, in_sel: torch.Tensor) -> None:
    if isinstance(src, nn.Linear):
        dst.weight.copy_(src.weight[out_sel][:, in_sel])
        if src.bias is None:
            dst.bias.zero_()
        else:
            dst.bias.copy_(src.bias[out_sel])
        return
    if isinstance(src, nn.Identity):
        matrix = (out_sel[:, None] == in_sel[None, :]).to(device=dst.weight.device, dtype=dst.weight.dtype)
        dst.weight.copy_(matrix)
        dst.bias.zero_()
        return
    raise TypeError(f"unsupported trunk projector for head copy: {type(src).__name__}")


class _MMDiTLiteMixin:
    def init_from_config(self, model: nn.Module, init_cfg) -> dict:
        if init_cfg.get("type") != "trunk_headcopy":
            raise ValueError(f"unknown control init type: {init_cfg.get('type')}")
        seed = int(init_cfg.get("seed", 0))
        enc_generator = torch.Generator().manual_seed(seed)
        summary = {"enc": self.mmdit_lite.init_from_trunk(
            model, generator=enc_generator,
            positions=init_cfg["enc_positions"])}
        dec = getattr(self, "mmdit_dec", None)
        if dec is not None:
            dec_generator = torch.Generator().manual_seed(seed)
            summary["dec"] = dec.init_from_trunk(
                model, generator=dec_generator, enc_positions=summary["enc"]["positions"],
                positions=init_cfg["dec_positions"])
        return summary

    def _init_mmdit_lite(
        self,
        mmdit_stream_width: int = 480,
        mmdit_stream_heads: int = 8,
        mmdit_dual_blocks: int = 14,
        mmdit_gate_init: float = -4.0,
        mmdit_gate_scale: float = 50.0,
        mmdit_use_source_time: bool = True,
        mmdit_use_source_context: bool = True,
        mmdit_block_stride: int = 1,
        mmdit_blocks_per_ref: int = 1,
        mmdit_gate_per_block: bool = True,
        mmdit_dynamic_gate: bool = False,
        mmdit_dynamic_gate_scale: float = 1.0,
    ) -> None:
        self.mmdit_lite = MMDiTLiteStack(
            stream_width=mmdit_stream_width,
            stream_heads=mmdit_stream_heads,
            dual_blocks=mmdit_dual_blocks,
            gate_init=mmdit_gate_init,
            gate_scale=mmdit_gate_scale,
            use_source_time=mmdit_use_source_time,
            use_source_context=mmdit_use_source_context,
            block_stride=mmdit_block_stride,
            blocks_per_ref=mmdit_blocks_per_ref,
            gate_per_block=mmdit_gate_per_block,
            dynamic_gate=mmdit_dynamic_gate,
            dynamic_gate_scale=mmdit_dynamic_gate_scale,
        )

    def bind_model(self, model: nn.Module) -> None:
        super().bind_model(model)
        self.mmdit_lite.bind_model(model)

    def encoder_block_extra_kv(self, block_index, *, seq, ref_z=None, source_enabled=True, state=None, **kwargs):
        base = super().encoder_block_extra_kv(
            block_index, seq=seq, ref_z=ref_z, source_enabled=source_enabled, state=state, **kwargs,
        )
        if base is not None:
            return base
        return self.mmdit_lite.extra_kv(
            block_index, seq=seq, ref_z=ref_z, source_enabled=source_enabled, state=state, **kwargs,
        )


class MMDiTLiteDecStack(nn.Module):
    """Decoder-side ref stream initialized directly or from an encoder bridge,
    with one live-x dual step and one K/V exit per decoder block."""

    def __init__(
        self,
        stream_width: int = 640,
        stream_heads: int = 5,
        gate_init: float = -4.0,
        gate_scale: float = 50.0,
        dynamic_gate: bool = False,
        dynamic_gate_scale: float = 1.0,
    ):
        super().__init__()
        if int(stream_width) <= 0 or int(stream_heads) <= 0 or int(stream_width) % int(stream_heads) != 0:
            raise ValueError("stream_width must be positive and divisible by stream_heads.")
        if float(gate_scale) == 0.0:
            raise ValueError("gate_scale must be non-zero.")
        self.stream_width = int(stream_width)
        self.stream_heads = int(stream_heads)
        self.gate_init = float(gate_init)
        self.gate_scale = float(gate_scale)
        self.dynamic_gate_enabled = bool(dynamic_gate)
        self.dynamic_gate_scale = float(dynamic_gate_scale)
        self.bridge = None
        self.patch_proj = None
        self.blocks = None
        self.to_dec_k = None
        self.to_dec_v = None
        self.gates = None
        self.dynamic_gate = None
        self.ref_rope = None
        self.num_patches = None
        self._bound_model = None

    def bind_model(self, model: nn.Module, enc_stream_width: int | None) -> None:
        object.__setattr__(self, "_bound_model", model)
        d = self.stream_width
        num_dec = int(model.num_dec_blocks)
        dec_hidden = int(model.blocks[int(model.num_enc_blocks)].attn.dim)
        self.num_patches = int(model.x_embedder.num_patches)
        self.bridge = nn.Linear(int(enc_stream_width), d) if enc_stream_width is not None else None
        self.patch_proj = nn.Linear(dec_hidden, d)
        self.blocks = nn.ModuleList(RefStreamBlock(d, self.stream_heads) for _ in range(num_dec - 1))
        self.to_dec_k = nn.ModuleList(nn.Linear(d, dec_hidden) for _ in range(num_dec))
        self.to_dec_v = nn.ModuleList(nn.Linear(d, dec_hidden) for _ in range(num_dec))
        self.gates = nn.Parameter(torch.full((num_dec,), self.gate_init / self.gate_scale))
        if self.dynamic_gate_enabled:
            self.dynamic_gate = DynamicKVGate(d, self.dynamic_gate_scale)
        self.ref_rope = RoPE(d // self.stream_heads, self.num_patches, 0)

    @torch.no_grad()
    def init_from_trunk(self, model: nn.Module, *, generator, positions, enc_positions=None) -> dict:
        """Initialize the decoder stream in the selected trunk coordinate subspaces."""
        n_enc = int(model.num_enc_blocks)
        dec0 = model.blocks[n_enc]
        head_dim = int(dec0.attn.head_dim)
        n_heads = int(dec0.attn.num_heads)
        if self.stream_width != self.stream_heads * head_dim:
            raise ValueError(f"trunk_headcopy requires stream head_dim to match the trunk: {self.stream_width} != {self.stream_heads}x{head_dim}")
        positions, sel = _select_head_channels(n_heads, head_dim, positions)
        if self.bridge is not None:
            enc_head_dim = int(model.blocks[0].attn.head_dim)
            _, enc_sel = _select_head_channels(
                int(model.blocks[0].attn.num_heads), enc_head_dim, enc_positions,
            )
            _copy_projector_submatrix(self.bridge, model.s_projector, sel, enc_sel)
        d = len(sel)
        select = torch.zeros(d, dec0.attn.dim, device=self.patch_proj.weight.device)
        select[torch.arange(d), sel] = 1.0
        self.patch_proj.weight.copy_(select)
        self.patch_proj.bias.zero_()
        for j, sb in enumerate(self.blocks):
            tb = model.blocks[n_enc + j + 1]
            _headcopy_ref_block(sb, tb, sel, generator)
        for k in range(len(self.to_dec_k)):
            tb = model.blocks[n_enc + k]
            self.to_dec_k[k].weight.copy_(tb.attn.k.weight[:, sel])
            self.to_dec_k[k].bias.copy_(tb.attn.k.bias)
            self.to_dec_v[k].weight.copy_(tb.attn.v.weight[:, sel])
            self.to_dec_v[k].bias.copy_(tb.attn.v.bias)
        return {"positions": positions, "block_sources": list(range(1, len(self.blocks) + 1))}

    def extra_kv(self, block_index: int, *, x_tokens, scale, disabled, state, initial_state=None):
        if block_index == 0 or "mmdit_dec_state" not in state:
            if initial_state is None:
                if "mmdit_ref_state" not in state:
                    return None
                initial_state = self.bridge(state["mmdit_ref_state"])
            state["mmdit_dec_state"] = initial_state
            state["mmdit_dec_block"] = 0
        if state["mmdit_dec_block"] < block_index - 1:
            raise RuntimeError(
                f"decoder extra_kv called for block {block_index} after block {state['mmdit_dec_block']}"
            )
        ref = state["mmdit_dec_state"]
        while state["mmdit_dec_block"] < block_index:
            i = state["mmdit_dec_block"]
            patch_d = self.patch_proj(x_tokens[:, : self.num_patches, :].to(dtype=ref.dtype))
            cos = self.ref_rope.freqs_cos[: self.num_patches]
            sin = self.ref_rope.freqs_sin[: self.num_patches]
            ref = self.blocks[i](ref, patch_d, cos, sin)
            state["mmdit_dec_state"] = ref
            state["mmdit_dec_block"] = i + 1
        model = self._bound_model
        attn = model.blocks[int(model.num_enc_blocks) + block_index].attn
        B, N = ref.shape[0], self.num_patches
        ek = self.to_dec_k[block_index](ref).reshape(B, N, attn.num_heads, attn.head_dim).permute(0, 2, 1, 3)
        ev = self.to_dec_v[block_index](ref).reshape(B, N, attn.num_heads, attn.head_dim).permute(0, 2, 1, 3)
        ek = attn.k_norm(ek)
        ek = apply_rope_table(ek, model.dec_rope.freqs_cos[:N], model.dec_rope.freqs_sin[:N])
        if not isinstance(scale, float) or scale != 1.0:
            ev = ev * (scale if isinstance(scale, float) else scale.view(B, 1, 1, 1).to(dtype=ev.dtype))
        ebias = (self.gates[block_index] * self.gate_scale).to(dtype=torch.float32).expand(B, 1, 1, N)
        if self.dynamic_gate is not None:
            ebias = ebias + self.dynamic_gate(ref).to(dtype=torch.float32).view(B, 1, 1, N)
        if disabled is not None:
            neg = torch.finfo(torch.float32).min / 2
            ebias = ebias + disabled.view(B, 1, 1, 1).to(dtype=torch.float32) * neg
        return ek, ev, ebias


class MMDiTLiteEncDecControl(_MMDiTLiteMixin, ConditionControl):
    """mmdit-only control: encoder K/V injection (MMDiTLiteStack) plus a decoder
    continuation stream (MMDiTLiteDecStack). No adaLN / x-residual decoder
    branch; source-off is the exact frozen path on both sides by construction."""

    def __init__(
        self,
        mmdit_stream_width: int = 480,
        mmdit_stream_heads: int = 8,
        mmdit_dual_blocks: int = 14,
        mmdit_gate_init: float = -4.0,
        mmdit_gate_scale: float = 50.0,
        mmdit_use_source_time: bool = True,
        mmdit_use_source_context: bool = True,
        mmdit_block_stride: int = 1,
        mmdit_blocks_per_ref: int = 1,
        mmdit_gate_per_block: bool = True,
        mmdit_dynamic_gate: bool = False,
        mmdit_dynamic_gate_scale: float = 1.0,
        dec_stream_width: int = 640,
        dec_stream_heads: int = 5,
        dec_gate_init: float = -4.0,
        dec_gate_scale: float = 50.0,
        dec_dynamic_gate: bool = False,
        dec_dynamic_gate_scale: float = 1.0,
    ):
        super().__init__()
        self._init_mmdit_lite(
            mmdit_stream_width=mmdit_stream_width,
            mmdit_stream_heads=mmdit_stream_heads,
            mmdit_dual_blocks=mmdit_dual_blocks,
            mmdit_gate_init=mmdit_gate_init,
            mmdit_gate_scale=mmdit_gate_scale,
            mmdit_use_source_time=mmdit_use_source_time,
            mmdit_use_source_context=mmdit_use_source_context,
            mmdit_block_stride=mmdit_block_stride,
            mmdit_blocks_per_ref=mmdit_blocks_per_ref,
            mmdit_gate_per_block=mmdit_gate_per_block,
            mmdit_dynamic_gate=mmdit_dynamic_gate,
            mmdit_dynamic_gate_scale=mmdit_dynamic_gate_scale,
        )
        self.mmdit_dec = MMDiTLiteDecStack(
            stream_width=dec_stream_width,
            stream_heads=dec_stream_heads,
            gate_init=dec_gate_init,
            gate_scale=dec_gate_scale,
            dynamic_gate=dec_dynamic_gate,
            dynamic_gate_scale=dec_dynamic_gate_scale,
        )

    def bind_model(self, model: nn.Module) -> None:
        super().bind_model(model)
        self.mmdit_dec.bind_model(model, self.mmdit_lite.stream_width)

    def decoder_block_extra_kv(
        self,
        block_index: int,
        *,
        x_tokens,
        seq=None,
        ref_z=None,
        source_enabled=True,
        source_scale=None,
        state=None,
        **kwargs,
    ):
        if ref_z is None or not source_any_enabled(source_enabled):
            return None
        scale = self.mmdit_lite._resolve_scale(source_scale, ref_z)
        if scale is None:
            return None
        disabled = self.mmdit_lite._disabled_rows(source_enabled, ref_z)
        return self.mmdit_dec.extra_kv(block_index, x_tokens=x_tokens, scale=scale, disabled=disabled, state=state)

