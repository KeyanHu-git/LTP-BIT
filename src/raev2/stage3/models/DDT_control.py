from __future__ import annotations

import torch.nn.functional as F

from raev2.stage2.models.DDT import DDTFinalLayer, DiTwDDTHead
from raev2.stage3.controls.base import build_condition_control


class DiTwDDTHeadControlled(DiTwDDTHead):
    """Config-driven Stage3 DDT with pluggable source-condition control."""

    def __init__(self, condition_control=None, **kwargs):
        super().__init__(**kwargs)
        self.condition_control = build_condition_control(condition_control, self)

    def forward(
        self,
        x,
        t,
        return_intermediate=False,
        ref_z=None,
        source_enabled=True,
        source_scale=None,
        return_base=False,
        **condition_kwargs,
    ):
        zt_intermediate = None
        base_tokens = None
        control_state = self.condition_control.begin_forward(
            x=x,
            t=t,
            ref_z=ref_z,
            source_enabled=source_enabled,
            source_scale=source_scale,
            condition_kwargs=condition_kwargs,
        )
        seq_input = self.condition_control.encoder_input(
            x,
            ref_z=ref_z,
            source_enabled=source_enabled,
            source_scale=source_scale,
            t=t,
            condition_kwargs=condition_kwargs,
            state=control_state,
        )
        seq, t_emb_base = self._build_sequence(seq_input, t, condition_kwargs)
        seq = self.condition_control.encoder_sequence(
            seq,
            ref_z=ref_z,
            source_enabled=source_enabled,
            source_scale=source_scale,
            t=t,
            condition_kwargs=condition_kwargs,
            state=control_state,
        )
        encoder_rope = self.condition_control.encoder_rope(
            self.enc_rope,
            seq=seq,
            ref_z=ref_z,
            source_enabled=source_enabled,
            source_scale=source_scale,
            t=t,
            condition_kwargs=condition_kwargs,
            state=control_state,
        )
        attn_mask = self._build_attn_mask(seq, condition_kwargs)
        attn_mask = self.condition_control.encoder_attn_mask(
            attn_mask,
            seq=seq,
            ref_z=ref_z,
            source_enabled=source_enabled,
            source_scale=source_scale,
            t=t,
            condition_kwargs=condition_kwargs,
            state=control_state,
        )
        for i in range(self.num_enc_blocks):
            enc_modulation = self.condition_control.encoder_block_modulation(
                i,
                seq=seq,
                ref_z=ref_z,
                source_enabled=source_enabled,
                source_scale=source_scale,
                t=t,
                condition_kwargs=condition_kwargs,
                state=control_state,
            )
            enc_extra_kv = self.condition_control.encoder_block_extra_kv(
                i,
                seq=seq,
                ref_z=ref_z,
                source_enabled=source_enabled,
                source_scale=source_scale,
                t=t,
                condition_kwargs=condition_kwargs,
                state=control_state,
            )
            seq = self.blocks[i](seq, encoder_rope, attn_mask, modulation=enc_modulation, extra_kv=enc_extra_kv)
            seq = self.condition_control.after_encoder_block(
                seq,
                block_index=i,
                ref_z=ref_z,
                source_enabled=source_enabled,
                source_scale=source_scale,
                t=t,
                condition_kwargs=condition_kwargs,
                state=control_state,
            )
            if return_base and (i + 1) == getattr(self, "base_model_depth", -1):
                base_tokens = seq[:, :self.s_embedder.num_patches, :]
            if return_intermediate and (i + 1) == self.repa_layer_depth:
                zt_intermediate = self.repa_projector(seq[:, :self.s_embedder.num_patches, :])
        seq = self.s_projector(F.silu(t_emb_base + seq[:, :self.s_embedder.num_patches, :]))
        seq = self.condition_control.apply_seq(
            seq,
            ref_z=ref_z,
            source_enabled=source_enabled,
            source_scale=source_scale,
            t=t,
            condition_kwargs=condition_kwargs,
            state=control_state,
        )

        x = self.x_embedder(x)
        x = self.condition_control.before_decoder(
            x,
            seq=seq,
            ref_z=ref_z,
            source_enabled=source_enabled,
            source_scale=source_scale,
            t=t,
            condition_kwargs=condition_kwargs,
            state=control_state,
        )
        for i in range(self.num_dec_blocks):
            block = self.blocks[self.num_enc_blocks + i]
            block_seq = self.condition_control.decoder_condition(
                seq,
                x_tokens=x,
                block_index=i,
                ref_z=ref_z,
                source_enabled=source_enabled,
                source_scale=source_scale,
                t=t,
                condition_kwargs=condition_kwargs,
                state=control_state,
            )
            block_delta = self._decoder_block_delta(
                i,
                block_seq,
                ref_z=ref_z,
                source_enabled=source_enabled,
                source_scale=source_scale,
                t=t,
                condition_kwargs=condition_kwargs,
                state=control_state,
            )
            block_delta = self.condition_control.prepare_decoder_block_delta(
                block_delta,
                c=block_seq,
                block=block,
                block_index=i,
                ref_z=ref_z,
                source_enabled=source_enabled,
                source_scale=source_scale,
                t=t,
                condition_kwargs=condition_kwargs,
                state=control_state,
            )
            dec_extra_kv = self.condition_control.decoder_block_extra_kv(
                i,
                x_tokens=x,
                seq=block_seq,
                ref_z=ref_z,
                source_enabled=source_enabled,
                source_scale=source_scale,
                t=t,
                condition_kwargs=condition_kwargs,
                state=control_state,
            )
            x = block(x, block_seq, self.dec_rope, modulation_delta=block_delta, extra_kv=dec_extra_kv)
            x = self.condition_control.after_decoder_block(
                x,
                seq=block_seq,
                block_index=i,
                ref_z=ref_z,
                source_enabled=source_enabled,
                source_scale=source_scale,
                t=t,
                condition_kwargs=condition_kwargs,
                state=control_state,
            )
        final_seq = self.condition_control.final_condition(
            seq,
            x_tokens=x,
            ref_z=ref_z,
            source_enabled=source_enabled,
            source_scale=source_scale,
            t=t,
            condition_kwargs=condition_kwargs,
            state=control_state,
        )
        final_delta = self._final_layer_delta(
            final_seq,
            ref_z=ref_z,
            source_enabled=source_enabled,
            source_scale=source_scale,
            t=t,
            condition_kwargs=condition_kwargs,
            state=control_state,
        )
        final_delta = self.condition_control.prepare_final_layer_delta(
            final_delta,
            c=final_seq,
            final_layer=self.final_layer,
            ref_z=ref_z,
            source_enabled=source_enabled,
            source_scale=source_scale,
            t=t,
            condition_kwargs=condition_kwargs,
            state=control_state,
        )
        x = self.final_layer(x, final_seq, modulation_delta=final_delta)
        x = self.unpatchify(x, self.x_patch_size)

        if return_base:
            if base_tokens is None or not hasattr(self, "base_final_layer"):
                raise RuntimeError("return_base requires a configured internal-guidance base head.")
            x_base = F.silu(t_emb_base + base_tokens)
            x_base = self.base_final_layer(x_base, x_base)
            x_base = self.unpatchify(x_base, self.s_patch_size)
            if return_intermediate:
                return (x, x_base), zt_intermediate
            return x, x_base
        if return_intermediate:
            return x, zt_intermediate
        return x

    def _decoder_block_delta(self, block_index, c, **kwargs):
        fn = getattr(self.condition_control, "decoder_block_delta", None)
        if fn is None:
            return None
        return fn(block_index, c, **kwargs)

    def _final_layer_delta(self, c, **kwargs):
        fn = getattr(self.condition_control, "final_layer_delta", None)
        if fn is None:
            return None
        return fn(c, **kwargs)


class DiTwDDTHeadIGControlled(DiTwDDTHeadControlled):
    """Controlled DDT that preserves the Stage2 IG dual-output contract."""

    def __init__(self, base_model_depth=8, **kwargs):
        super().__init__(**kwargs)
        self.base_model_depth = base_model_depth
        self.base_final_layer = DDTFinalLayer(self.enc_hidden_size, self.s_patch_size, self.in_channels)
        self._init_base_final_layer()

    def forward(self, x, t, return_intermediate=False, **condition_kwargs):
        return super().forward(
            x,
            t,
            return_intermediate=return_intermediate,
            return_base=True,
            **condition_kwargs,
        )

    def forward_with_base(self, x, t, **condition_kwargs):
        return self.forward(x, t, **condition_kwargs)

    def _init_base_final_layer(self):
        final = self.base_final_layer
        final.adaln_modulation[-1].weight.data.zero_()
        final.adaln_modulation[-1].bias.data.zero_()
        final.linear.weight.data.zero_()
        final.linear.bias.data.zero_()
