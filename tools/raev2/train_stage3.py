#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (str(ROOT), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from Dataset.datasets.utils import build_dataset_from_config as build_dataset
from Evaluation import evaluate_config
from raev2.eval.paired_i2i import save_paired_i2i_fakes_ddp
from raev2.stage2.transport import Sampler, Transport
from raev2.stage3.config import (
    apply_freeze_policy,
    stage3_paired_training_losses,
    validate_stage3_data_mix,
)
from raev2.stage3.lora import apply_lora_from_config
from raev2.utils.checkpoint import align_rope_buffers, load_stage3_checkpoint, save_stage3_checkpoint
from raev2.utils.checkpoint import unwrap_model as unwrap
from raev2.utils.config_utils import cfg_to_dict
from raev2.utils.dist_utils import cleanup_distributed, setup_distributed
from raev2.utils.guidance_utils import get_model_forward_fn, is_guidance_active, uses_internal_guidance
from raev2.utils.model_utils import instantiate_from_config
from raev2.utils.optim_utils import build_optimizer, build_scheduler
from raev2.utils.resume_utils import configure_experiment_dirs, experiment_name_from_config, find_resume_checkpoint
from raev2.utils.train_utils import accumulation_loss_weight, curriculum_probability
from raev2.utils.train_utils import autocast_kwargs_from_config as autocast_kwargs
from raev2.utils.train_utils import prepare_stage2_model_config, update_ema
from utils import ConfigLoader
from utils.logging import format_metrics
from utils.rng import capture_rng_state, restore_rng_state, seed_all as set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="RAEv2 Stage-3 paired i2i training entrypoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--results-dir", default="experiments/raev2/stage3")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--init-ckpt", default=None)
    parser.add_argument("--fresh-start", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--wandb", action="store_true")
    return parser.parse_args()


def build_loader(dataset, cfg, rank: int, world_size: int, micro_batch_size: int, *, step_based: bool):
    drop_last = bool(cfg.training.get("drop_last", True))
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=int(cfg.training.global_seed),
        drop_last=drop_last,
    )
    loader_kwargs = {}
    if step_based:
        data_generator = torch.Generator()
        data_generator.manual_seed(int(cfg.training.global_seed) + rank)
        loader_kwargs["generator"] = data_generator
    loader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        sampler=sampler,
        num_workers=int(cfg.training.num_workers),
        pin_memory=True,
        drop_last=drop_last,
        **loader_kwargs,
    )
    return loader, sampler


def load_init_weights(model, ckpt_path: str, logger, *, strict: bool = True, allowed_missing_prefixes=()) -> None:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "ema" in checkpoint:
        state, source = checkpoint["ema"], "ema"
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state, source = checkpoint["model"], "model"
    else:
        state, source = checkpoint, "raw"
    state, aligned_rope = align_rope_buffers(model, state)
    keys = model.load_state_dict(state, strict=strict)
    if not strict:
        allowed = tuple(allowed_missing_prefixes or ())
        bad_missing = [key for key in keys.missing_keys if not key.startswith(allowed)]
        if bad_missing:
            raise RuntimeError(f"Unexpected missing keys while loading {ckpt_path}: {bad_missing[:20]}")
        if keys.unexpected_keys:
            raise RuntimeError(f"Unexpected checkpoint keys while loading {ckpt_path}: {keys.unexpected_keys[:20]}")
    logger.info(
        f"Initialized Stage3 model from {ckpt_path} ({source}); "
        f"missing={len(keys.missing_keys)} unexpected={len(keys.unexpected_keys)} "
        f"aligned_rope={len(aligned_rope)}"
    )


def validate_stage3_scope(cfg) -> None:
    if cfg.get("repa") is not None and bool(cfg.repa.get("use_repa", False)):
        raise NotImplementedError("RAEv2 Stage3 paired i2i does not support REPA yet.")
    if cfg.get("internal_guidance") is not None and cfg.internal_guidance.get("base_model_depth") is not None:
        raise NotImplementedError("RAEv2 Stage3 paired i2i does not support internal-guidance base loss yet.")


def consume_control_diagnostics(model) -> dict[str, float]:
    control = getattr(unwrap(model), "condition_control", None)
    if control is None:
        return {}
    consume = getattr(control, "consume_diagnostics", None)
    if consume is None:
        return {}
    return consume()


@torch.no_grad()
def update_trainable_ema(ema_model, model, names: list[str], decay: float) -> None:
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(unwrap(model).named_parameters())
    for name in names:
        ema_params[name].mul_(decay).add_(model_params[name].data, alpha=1.0 - decay)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("RAEv2 Stage-3 training requires CUDA.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    rank, world_size, device = setup_distributed()
    cfg = OmegaConf.create(ConfigLoader.load_recursive(args.config, overrides=args.override))
    prepare_stage2_model_config(cfg)
    validate_stage3_scope(cfg)
    mix_ratios = validate_stage3_data_mix(cfg)
    stage3_loss_cfg = cfg.get("stage3", {}).get("loss", {})
    internal_guidance_cfg = cfg.get("internal_guidance", {})
    base_model_coeff = float(stage3_loss_cfg.get("base_model_coeff", internal_guidance_cfg.get("base_model_coeff", 1.0)))
    total_steps_cfg = cfg.training.get("total_steps")
    total_steps = int(total_steps_cfg) if total_steps_cfg is not None else None
    step_based = total_steps is not None
    global_seed = int(cfg.training.global_seed)
    seed = global_seed + rank if step_based else global_seed * world_size + rank
    set_seed(seed)

    init_ckpt = args.init_ckpt or cfg.training.get("init_ckpt")
    if init_ckpt:
        cfg.training.init_ckpt = str(init_ckpt)

    experiment_dir, checkpoint_dir, logger = configure_experiment_dirs(
        args,
        rank,
        cfg,
        fallback_name=experiment_name_from_config(args.config),
    )
    experiment_dir = Path(experiment_dir)
    checkpoint_dir = Path(checkpoint_dir)
    fresh_start = bool(args.fresh_start) or os.environ.get("FRESH_START", "0") == "1"
    resume_ckpt = find_resume_checkpoint(str(experiment_dir), args.resume) if (args.resume or not fresh_start) else None
    if rank == 0:
        OmegaConf.save(cfg, experiment_dir / "config.yaml")
        logger.info(f"Config: {args.config}")
        logger.info(f"Experiment: {experiment_dir}")
        logger.info(f"Resume: {resume_ckpt or '<none>'}")

    global_batch = int(cfg.training.global_batch_size)
    grad_accum = int(cfg.training.grad_accum_steps)
    if global_batch % (world_size * grad_accum) != 0:
        raise ValueError("training.global_batch_size must be divisible by world_size * grad_accum_steps.")
    micro_batch = global_batch // (world_size * grad_accum)
    if total_steps is not None and total_steps <= 0:
        raise ValueError("training.total_steps must be positive.")
    if step_based and args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive.")
    stop_step = total_steps if step_based else None
    if step_based and args.max_steps is not None:
        stop_step = min(stop_step, args.max_steps) if stop_step is not None else args.max_steps

    dataset = build_dataset(cfg.dataset)
    expected_dataset_size = cfg.training.get("expected_dataset_size")
    if step_based and expected_dataset_size is not None and len(dataset) != int(expected_dataset_size):
        raise ValueError(
            f"Dataset size mismatch: expected {int(expected_dataset_size)}, found {len(dataset)}."
        )
    if step_based and bool(cfg.training.get("require_even_shards", False)) and len(dataset) % world_size != 0:
        raise ValueError("Dataset size must be divisible by world_size when require_even_shards=true.")
    loader, sampler = build_loader(dataset, cfg, rank, world_size, micro_batch, step_based=step_based)
    steps_per_epoch = math.ceil(len(loader) / grad_accum)
    if steps_per_epoch <= 0:
        raise ValueError("No optimizer steps per epoch. Check batch size and dataset length.")
    training_epochs = (
        math.ceil(total_steps / steps_per_epoch)
        if total_steps is not None
        else int(cfg.training.epochs)
    )
    drop_last = bool(cfg.training.get("drop_last", True))
    final_micro_batch = micro_batch
    if not drop_last and len(sampler) % micro_batch:
        final_micro_batch = len(sampler) % micro_batch

    rae = instantiate_from_config(cfg.stage_1).to(device).eval()
    rae.requires_grad_(False)
    model = instantiate_from_config(cfg.stage_2).to(device)
    allow_cold_start = bool(cfg.training.get("allow_cold_start", False))
    if resume_ckpt is None and not init_ckpt and not allow_cold_start:
        raise ValueError(
            "RAEv2 Stage3 requires training.init_ckpt or CLI --init-ckpt, "
            "or set training.allow_cold_start=true."
        )
    if init_ckpt and rank == 0:
        ckpt_cfg = cfg.get("stage3", {}).get("checkpoint", {})
        strict_init = bool(ckpt_cfg.get("strict_init", True))
        allowed_missing_prefixes = tuple(ckpt_cfg.get("allowed_missing_prefixes", []) or ())
        load_init_weights(model, init_ckpt, logger, strict=strict_init, allowed_missing_prefixes=allowed_missing_prefixes)

    control_init_cfg = cfg.get("stage3", {}).get("control", {}).get("init")
    if control_init_cfg is not None and resume_ckpt is None:
        init_summary = model.condition_control.init_from_config(model, control_init_cfg)
        if rank == 0:
            logger.info(f"Control init policy: {control_init_cfg.get('type')} -> {init_summary}")

    lora_cfg = cfg.get("lora", {})
    n_replaced, n_trainable = apply_lora_from_config(model, lora_cfg)
    if n_replaced > 0:
        if rank == 0:
            logger.info(f"LoRA wrapped {n_replaced} linear layers; trainable={n_trainable / 1e6:.3f}M")

    apply_freeze_policy(model, cfg.get("stage3", {}).get("freeze"), logger if rank == 0 else None)
    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    ema_model = deepcopy(model).to(device).eval()
    ema_model.requires_grad_(False)
    ddp_model = DDP(model, device_ids=[device.index], broadcast_buffers=False) if dist.is_initialized() else model
    model = unwrap(ddp_model)
    # EMA must snapshot post-broadcast weights: ranks build with different seeds,
    # and DDP only syncs the training model. decay=0.0 copies model into ema.
    update_ema(ema_model, model, decay=0.0)

    optimizer, opt_msg = build_optimizer([p for p in model.parameters() if p.requires_grad], cfg.training.optimizer)
    scheduler, sched_msg = None, None
    if cfg.training.get("scheduler") is not None:
        scheduler, sched_msg = build_scheduler(optimizer, steps_per_epoch, cfg.training.scheduler)

    latent_size = tuple(int(v) for v in cfg.misc.latent_size)
    shift_dim = cfg.misc.time_dist_shift_dim if "time_dist_shift_dim" in cfg.misc else math.prod(latent_size)
    time_dist_shift = math.sqrt(float(shift_dim) / float(cfg.misc.time_dist_shift_base))
    if "time_dist_shift" in cfg.transport:
        time_dist_shift = float(cfg.transport.time_dist_shift)
    time_sampling_cfg = OmegaConf.select(cfg, "stage3.loss.time_sampling") or {}
    transport = Transport(
        prediction=str(cfg.transport.prediction),
        time_dist_type=str(cfg.transport.time_dist_type),
        time_dist_shift=time_dist_shift,
        t_eps=float(cfg.transport.get("t_eps", 0.05)),
        pure_noise_prob=float(time_sampling_cfg.get("pure_noise_prob", 0.0)),
        pure_noise_t=float(time_sampling_cfg.get("pure_noise_t", 1.0)),
    )
    pure_noise_start_prob = time_sampling_cfg.get("pure_noise_start_prob")
    pure_noise_hold_frac = float(time_sampling_cfg.get("pure_noise_hold_frac", 0.0))
    pure_noise_decay_end_epoch = time_sampling_cfg.get("pure_noise_decay_end_epoch")
    if pure_noise_decay_end_epoch is not None:
        pure_noise_decay_end_epoch = int(pure_noise_decay_end_epoch)
        if step_based:
            raise ValueError("pure_noise_decay_end_epoch is only valid for epoch-based training.")
    pure_noise_update_interval_steps = (
        int(time_sampling_cfg.get("pure_noise_update_interval_steps", 1)) if step_based else 1
    )
    if step_based and pure_noise_update_interval_steps <= 0:
        raise ValueError("stage3.loss.time_sampling.pure_noise_update_interval_steps must be positive.")
    if step_based and not 0.0 <= pure_noise_hold_frac <= 1.0:
        raise ValueError("Stage3 pure-noise hold must be within the training schedule.")
    pure_noise_total_updates = (
        math.ceil(total_steps / pure_noise_update_interval_steps) if step_based else None
    )
    auto = autocast_kwargs(cfg)

    start_epoch, global_step, resume_batch_idx = 0, 0, 0
    resume_rng_state = None
    if resume_ckpt:
        if not step_based:
            start_epoch, global_step, checkpoint_mode = load_stage3_checkpoint(
                resume_ckpt,
                ddp_model,
                ema_model,
                optimizer,
                scheduler,
                init_ckpt=init_ckpt,
            )
            start_epoch += 1
            logger.info(
                f"Resumed from {resume_ckpt} ({checkpoint_mode}) at epoch={start_epoch}, step={global_step}."
            )
        else:
            start_epoch, global_step, checkpoint_mode, resume_progress = load_stage3_checkpoint(
                resume_ckpt,
                ddp_model,
                ema_model,
                optimizer,
                scheduler,
                init_ckpt=init_ckpt,
                return_progress=True,
            )
            if resume_progress["epoch_complete"]:
                start_epoch += 1
            else:
                resume_batch_idx = int(resume_progress["next_batch_idx"])
            expected_progress = {
                "batches_per_epoch": len(loader),
                "grad_accum_steps": grad_accum,
                "micro_batch_size": micro_batch,
                "dataset_size": len(dataset),
                "world_size": world_size,
                "total_steps": total_steps,
            }
            for key, expected in expected_progress.items():
                if int(resume_progress.get(key, 0)) != expected:
                    raise RuntimeError(f"Stage3 resume {key} changed from the saved training topology.")
            rng_states = resume_progress.get("rng_states")
            if not isinstance(rng_states, list) or len(rng_states) != world_size:
                raise RuntimeError("Stage3 step checkpoint does not contain complete per-rank RNG state.")
            resume_rng_state = rng_states[rank]
            logger.info(
                f"Resumed from {resume_ckpt} ({checkpoint_mode}) at "
                f"epoch={start_epoch}, batch={resume_batch_idx}, step={global_step}."
            )

    if rank == 0:
        logger.info(f"Dataset length: {len(dataset)}; micro_batch/rank={micro_batch}; steps/epoch={steps_per_epoch}")
        if step_based:
            logger.info(f"Training steps: total={total_steps}, stop={stop_step}")
        logger.info(f"Seeds: global={int(cfg.training.global_seed)}, process={seed}, rank={rank}")
        logger.info(f"Optimizer: {opt_msg}")
        if sched_msg:
            logger.info(f"Scheduler: {sched_msg}")
        logger.info(f"Stage3 loss: base_model_coeff={base_model_coeff:.6g}")
        logger.info(f"time_dist_shift={time_dist_shift:.6g}")

    checkpoint_cfg = cfg.get("stage3", {}).get("checkpoint", {})
    checkpoint_save_mode = str(checkpoint_cfg.get("save_mode", "auto" if step_based else "full"))
    checkpoint_policy = cfg.training.get("checkpoint_policy", {})
    checkpoint_epoch_interval = int(
        checkpoint_policy.get("interval_epochs", cfg.training.get("checkpoint_interval", 0))
    )
    checkpoint_step_interval = int(checkpoint_policy.get("interval_steps", 0)) if step_based else 0
    checkpoint_save_numbered = bool(checkpoint_policy.get("save_numbered", False))
    checkpoint_save_last_every_epoch = bool(checkpoint_policy.get("save_last_every_epoch", True))
    if step_based and checkpoint_epoch_interval > 0 and checkpoint_step_interval > 0:
        raise ValueError("Configure checkpoint intervals in epochs or steps, not both.")

    eval_cfg = cfg.get("eval", {})
    do_eval = bool(eval_cfg.get("enabled", False))
    eval_epoch_interval = int(eval_cfg.get("epoch_interval", 0))
    eval_step_interval = int(eval_cfg.get("step_interval", 0)) if step_based and do_eval else 0
    if step_based and do_eval and eval_epoch_interval > 0 and eval_step_interval > 0:
        raise ValueError("Configure eval intervals in epochs or steps, not both.")
    if do_eval:
        for key in ("real_set", "eval_dataset", "dataloader", "metrics"):
            if key not in cfg:
                raise ValueError(f"Stage3 eval requires config key: {key}")
        eval_ds_cfg = dict(OmegaConf.to_container(cfg.dataset, resolve=True))
        eval_ds_cfg["is_train"] = False
        eval_ds_cfg["split"] = str(eval_cfg.get("split", "test"))
        eval_dataset = build_dataset(OmegaConf.create(eval_ds_cfg))
        eval_payload_base = {
            "real_set": OmegaConf.to_container(cfg.real_set, resolve=True),
            "dataset": OmegaConf.to_container(cfg.eval_dataset, resolve=True),
            "dataloader": OmegaConf.to_container(cfg.dataloader, resolve=True),
            "metrics": OmegaConf.to_container(cfg.metrics, resolve=True),
        }
        eval_target_dir = str(getattr(eval_dataset, "target_path", eval_payload_base["real_set"]["image_dir"]))
        eval_payload_base["real_set"]["image_dir"] = eval_target_dir
        eval_payload_base["dataset"]["gt_dir"] = eval_target_dir
        sample_fn = Sampler(transport, cfg.guidance).sample_ode(num_steps=int(cfg.sampler.num_steps))
        if uses_internal_guidance(cfg.guidance) and not hasattr(ema_model, "base_final_layer"):
            raise NotImplementedError("RAEv2 Stage3 IG eval requires a model with base_final_layer.")
        eval_model_fn, eval_guidance_kwargs = get_model_forward_fn(ema_model, cfg.guidance)
        eval_use_guidance = is_guidance_active(cfg.guidance)
        eval_sample_model_kwargs = {
            **cfg_to_dict(eval_cfg.get("sample_model_kwargs", {})),
            **eval_guidance_kwargs,
        }
        eval_null_label = int(cfg.misc.get("null_label", cfg.misc.num_classes))

        def run_eval(eval_tag: str) -> None:
            fake_dir = experiment_dir / str(eval_cfg.get("save_dir", "sample")) / eval_tag / "fake"
            preview_dir = experiment_dir / str(eval_cfg.get("preview_dir", "sample_preview")) / eval_tag
            n_eval = save_paired_i2i_fakes_ddp(
                model_fn=eval_model_fn,
                rae=rae,
                sampler_fn=sample_fn,
                val_dataset=eval_dataset,
                num_eval_samples=int(eval_cfg.num_samples),
                gen_batch_size=int(eval_cfg.gen_batch_size),
                device=device,
                latent_size=latent_size,
                out_dir=fake_dir,
                fixed_label=int(cfg.dataset.get("fixed_label", 0)),
                null_label=eval_null_label,
                use_guidance=eval_use_guidance,
                seed=int(eval_cfg.get("seed", 42)),
                preview_dir=preview_dir,
                preview_max_images=int(eval_cfg.get("preview_max_images", 16)),
                autocast_kwargs=auto,
                sample_model_kwargs=eval_sample_model_kwargs,
            )
            payload = dict(eval_payload_base)
            payload["dataset"] = dict(payload["dataset"])
            payload["dataset"]["pred_dir"] = str(fake_dir)
            metrics = evaluate_config(payload, device, use_tqdm=False, distributed=True)
            if rank == 0:
                log_metrics = {f"eval/{str(k).lower()}": float(v) for k, v in metrics.items()}
                log_metrics["eval/n"] = float(n_eval)
                logger.info(format_metrics(log_metrics))

    if resume_rng_state is not None:
        restore_rng_state(resume_rng_state)

    for epoch in range(start_epoch, training_epochs):
        if step_based and global_step >= stop_step:
            break
        if pure_noise_start_prob is not None and not step_based:
            total = int(cfg.training.epochs)
            transport.pure_noise_prob = curriculum_probability(
                epoch,
                total,
                start_probability=float(pure_noise_start_prob),
                end_probability=float(time_sampling_cfg.get("pure_noise_prob", 0.0)),
                hold_fraction=pure_noise_hold_frac,
                decay_end_epoch=pure_noise_decay_end_epoch,
            )
        model.train()
        sampler.set_epoch(epoch)
        if step_based:
            loader.generator.manual_seed(global_seed + epoch * world_size + rank)
        optimizer.zero_grad(set_to_none=True)
        epoch_metrics = defaultdict(float)
        epoch_steps = 0
        for step, (target_imgs, source_imgs, labels) in enumerate(loader):
            if epoch == start_epoch and step < resume_batch_idx:
                continue
            target_imgs = target_imgs.to(device, non_blocking=True)
            source_imgs = source_imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            with torch.no_grad():
                z_tgt = rae.encode(target_imgs)
                z_src = rae.encode(source_imgs)

            null_label = int(cfg.misc.get("null_label", cfg.misc.num_classes))
            model_kwargs = dict(context=labels, attn_mask=None)
            model_kwargs_null = dict(
                context=torch.full_like(labels, null_label),
                attn_mask=None,
            )
            source_kwargs = dict(ref_z=z_src, source_enabled=True)
            group_start = (step // grad_accum) * grad_accum
            group_end = min(group_start + grad_accum, len(loader))
            group_size = group_end - group_start
            sync_now = (step + 1) == group_end
            if pure_noise_start_prob is not None and step_based and step == group_start:
                update = min(global_step // pure_noise_update_interval_steps, pure_noise_total_updates - 1)
                transport.pure_noise_prob = curriculum_probability(
                    update,
                    pure_noise_total_updates,
                    start_probability=float(pure_noise_start_prob),
                    end_probability=float(time_sampling_cfg.get("pure_noise_prob", 0.0)),
                    hold_fraction=pure_noise_hold_frac,
                )
            sync_context = nullcontext() if sync_now or not isinstance(ddp_model, DDP) else ddp_model.no_sync()
            with sync_context:
                with torch.amp.autocast(device_type=device.type, **auto):
                    loss_dict = stage3_paired_training_losses(
                        transport,
                        ddp_model,
                        z_tgt,
                        model_kwargs=model_kwargs,
                        model_kwargs_null=model_kwargs_null,
                        source_kwargs=source_kwargs,
                        cfg_dropout_prob=float(cfg.conditioning.get("cfg_dropout_prob", 0.1)),
                        source_dropout_ratio=mix_ratios["source_dropout_ratio"],
                        base_model_coeff=base_model_coeff,
                    )
                    if drop_last:
                        loss = loss_dict["loss"].mean() / group_size
                    else:
                        loss = loss_dict["loss"].mean() * accumulation_loss_weight(
                            step,
                            len(loader),
                            micro_batch,
                            final_micro_batch,
                            grad_accum,
                        )
                loss.backward()

            for metric_name, metric_value in loss_dict.items():
                epoch_metrics[metric_name] += float(metric_value.mean().detach().item())
            for metric_name, metric_value in consume_control_diagnostics(ddp_model).items():
                epoch_metrics[metric_name] += float(metric_value)
            epoch_steps += 1
            if not sync_now:
                continue

            if cfg.training.get("clip_grad") is not None and float(cfg.training.clip_grad) > 0:
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), float(cfg.training.clip_grad))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            update_trainable_ema(ema_model, model, trainable_names, decay=float(cfg.training.ema_decay))
            global_step += 1

            if int(cfg.training.log_interval) > 0 and global_step % int(cfg.training.log_interval) == 0 and rank == 0:
                train_metrics = {
                    f"train/{key}": value / max(1, epoch_steps)
                    for key, value in epoch_metrics.items()
                }
                train_metrics["train/lr"] = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"[Epoch {epoch} | Step {global_step}] "
                    + format_metrics(train_metrics)
                )

            planned_complete = step_based and global_step >= stop_step
            saved_epoch_complete = step + 1 == len(loader)
            step_interval_due = (
                checkpoint_step_interval > 0
                and global_step % checkpoint_step_interval == 0
            )
            epoch_interval_due = (
                step_based
                and checkpoint_step_interval <= 0
                and saved_epoch_complete
                and checkpoint_epoch_interval > 0
                and (epoch + 1) % checkpoint_epoch_interval == 0
            )
            step_save_due = planned_complete or step_interval_due or epoch_interval_due
            if do_eval and eval_step_interval > 0 and global_step % eval_step_interval == 0:
                run_eval(f"step{global_step:07d}_ema")
            rng_states = None
            if step_save_due:
                local_rng_state = capture_rng_state()
                if dist.is_initialized():
                    rng_states = [None] * world_size
                    dist.all_gather_object(rng_states, local_rng_state)
                else:
                    rng_states = [local_rng_state]
            if step_save_due and rank == 0:
                next_batch_idx = step + 1
                save_kwargs = {
                    "init_ckpt": init_ckpt,
                    "mode": checkpoint_save_mode,
                    "epoch_complete": saved_epoch_complete,
                    "next_batch_idx": 0 if saved_epoch_complete else next_batch_idx,
                    "batches_per_epoch": len(loader),
                    "grad_accum_steps": grad_accum,
                    "micro_batch_size": micro_batch,
                    "dataset_size": len(dataset),
                    "world_size": world_size,
                    "total_steps": total_steps,
                    "rng_states": rng_states,
                }
                if checkpoint_save_numbered:
                    numbered_name = (
                        f"step-{global_step:07d}.pt"
                        if checkpoint_step_interval > 0 or not saved_epoch_complete
                        else f"ep-{epoch + 1:04d}.pt"
                    )
                    save_stage3_checkpoint(
                        str(checkpoint_dir / numbered_name),
                        global_step,
                        epoch,
                        ddp_model,
                        ema_model,
                        optimizer,
                        scheduler,
                        **save_kwargs,
                    )
                checkpoint_path = checkpoint_dir / "ep-last.pt"
                actual_mode = save_stage3_checkpoint(
                    str(checkpoint_path),
                    global_step,
                    epoch,
                    ddp_model,
                    ema_model,
                    optimizer,
                    scheduler,
                    **save_kwargs,
                )
                logger.info(f"Saved Stage3 checkpoint: {checkpoint_path} ({actual_mode})")
            if step_save_due and dist.is_initialized():
                dist.barrier()
            if step_based and global_step >= stop_step:
                break
            if not step_based and args.max_steps is not None and global_step >= args.max_steps:
                break

        if rank == 0 and epoch_steps > 0:
            epoch_log_metrics = {
                f"epoch/{key}": value / epoch_steps
                for key, value in epoch_metrics.items()
            }
            logger.info(f"[Epoch {epoch}] " + format_metrics(epoch_log_metrics))
            if not step_based and checkpoint_step_interval <= 0:
                interval_due = (
                    checkpoint_epoch_interval > 0
                    and (epoch + 1) % checkpoint_epoch_interval == 0
                )
                final_due = epoch + 1 == training_epochs
                if checkpoint_save_last_every_epoch or interval_due or final_due:
                    checkpoint_path = checkpoint_dir / "ep-last.pt"
                    actual_mode = save_stage3_checkpoint(
                        str(checkpoint_path),
                        global_step,
                        epoch,
                        ddp_model,
                        ema_model,
                        optimizer,
                        scheduler,
                        init_ckpt=init_ckpt,
                        mode=checkpoint_save_mode,
                        epoch_complete=True,
                    )
                    logger.info(f"Saved Stage3 checkpoint: {checkpoint_path} ({actual_mode})")
                if checkpoint_save_numbered and (interval_due or final_due):
                    save_stage3_checkpoint(
                        str(checkpoint_dir / f"ep-{epoch + 1:04d}.pt"),
                        global_step,
                        epoch,
                        ddp_model,
                        ema_model,
                        optimizer,
                        scheduler,
                        init_ckpt=init_ckpt,
                        mode=checkpoint_save_mode,
                        epoch_complete=True,
                    )

        if do_eval and eval_step_interval <= 0 and eval_epoch_interval > 0 and (epoch + 1) % eval_epoch_interval == 0:
            run_eval(f"ep{epoch + 1:03d}_ema")

        if step_based and global_step >= stop_step:
            break
        if not step_based and args.max_steps is not None and global_step >= args.max_steps:
            break

        resume_batch_idx = 0

    if dist.is_initialized():
        dist.barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
