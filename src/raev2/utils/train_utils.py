import torch
from omegaconf import DictConfig
from raev2.stage3.config import inject_stage3_control_config
from utils.torch_utils import requires_grad, update_ema


def curriculum_probability(
    epoch: int,
    total_epochs: int,
    *,
    start_probability: float,
    end_probability: float,
    hold_fraction: float,
    decay_end_epoch: int | None = None,
) -> float:
    if total_epochs <= 0:
        raise ValueError("total_epochs must be positive")
    if not 0 <= epoch < total_epochs:
        raise ValueError("epoch must be in [0, total_epochs)")
    if not 0.0 <= hold_fraction <= 1.0:
        raise ValueError("hold_fraction must be in [0, 1]")
    if not 0.0 <= start_probability <= 1.0:
        raise ValueError("start_probability must be in [0, 1]")
    if not 0.0 <= end_probability <= 1.0:
        raise ValueError("end_probability must be in [0, 1]")

    hold_epochs = min(int(hold_fraction * total_epochs), total_epochs)
    decay_end_epoch = total_epochs if decay_end_epoch is None else int(decay_end_epoch)
    if not 1 <= decay_end_epoch <= total_epochs:
        raise ValueError("decay_end_epoch must be in [1, total_epochs]")
    if hold_epochs < total_epochs and decay_end_epoch <= hold_epochs:
        raise ValueError("decay_end_epoch must be greater than the hold duration")
    if epoch < hold_epochs or hold_epochs == total_epochs:
        return float(start_probability)
    if epoch >= decay_end_epoch:
        return float(end_probability)

    transition_epochs = decay_end_epoch - hold_epochs
    progress = (epoch - hold_epochs + 1) / transition_epochs
    return float(start_probability) + (float(end_probability) - float(start_probability)) * progress


def accumulation_loss_weight(
    batch_index: int,
    num_batches: int,
    micro_batch_size: int,
    final_batch_size: int,
    grad_accum_steps: int,
) -> float:
    group_start = (batch_index // grad_accum_steps) * grad_accum_steps
    group_end = min(group_start + grad_accum_steps, num_batches)
    group_samples = (group_end - group_start) * micro_batch_size
    if group_end == num_batches:
        group_samples -= micro_batch_size - final_batch_size
    batch_size = final_batch_size if batch_index == num_batches - 1 else micro_batch_size
    return batch_size / group_samples


def prepare_stage2_model_config(config: DictConfig) -> DictConfig:
    resolution = int(config.stage_1.params.resolution)
    latent_stride = int(config.stage_1.params.get("decoder_patch_size", 16))
    if resolution <= 0 or latent_stride <= 0 or resolution % latent_stride != 0:
        raise ValueError(
            "stage_1.params.resolution must be positive and divisible by "
            "stage_1.params.decoder_patch_size."
        )

    input_size = resolution // latent_stride
    params = config.stage_2.params
    latent_size = [int(params.in_channels), input_size, input_size]
    params.input_size = input_size
    config.misc.latent_size = latent_size
    config.misc.time_dist_shift_dim = latent_size[0] * input_size * input_size

    dataset = config.get("dataset")
    if config.conditioning.type == "label" and dataset is not None and dataset.get("condition_type") is not None:
        config.conditioning.type = dataset.condition_type

    if config.conditioning.type == "text":
        config.conditioning.arch.num_c_tokens = config.conditioning.text_encoder.max_length

    if "condition_type" not in params:
        params.condition_type = config.conditioning.type
    if "num_classes" not in params:
        params.num_classes = config.misc.num_classes
    if "context_dim" not in params:
        params.context_dim = config.conditioning.get("context_dim")
    if "cond_arch" not in params:
        params.cond_arch = config.conditioning.arch

    if config.get("repa") is not None and config.repa.get("use_repa", False):
        params.setdefault("enable_repa", True)
        params.setdefault("repa_layer_depth", config.repa.repa_layer_depth)
        if config.repa.get("z_dim") is not None:
            params.setdefault("z_dim", config.repa.z_dim)

    internal_guidance = config.get("internal_guidance")
    if internal_guidance is not None and internal_guidance.get("base_model_depth") is not None:
        params.setdefault("base_model_depth", internal_guidance.base_model_depth)

    inject_stage3_control_config(config)

    return config.stage_2


def get_autocast_kwargs(args) -> dict:
    """Get autocast kwargs for bf16 or fp32 precision."""
    if args.precision == "bf16":
        return dict(enabled=True, dtype=torch.bfloat16)
    return dict(enabled=False)


def autocast_kwargs_from_config(config: DictConfig) -> dict:
    precision = str(config.training.get("precision", "bf16"))
    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("Requested bf16 precision, but this CUDA device does not support bfloat16.")
        return dict(enabled=True, dtype=torch.bfloat16)
    if precision == "fp32":
        return dict(enabled=False)
    raise ValueError(f"Unsupported precision: {precision}")
