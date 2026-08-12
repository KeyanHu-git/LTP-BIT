#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import shutil
import sys
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (str(ROOT), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from Dataset.datasets.utils import build_dataset_from_config as build_dataset
from Dataset.datasets.utils import get_class_labels
from Evaluation import evaluate_config
from raev2.eval.class_cond import save_class_cond_fakes_ddp
from raev2.eval.label_protocol import load_metadata_label_plan
from raev2.stage2.transport import Sampler, Transport
from raev2.stage2.utils import get_null_cond, setup_repa_target_encoder
from raev2.utils.checkpoint import load_stage2_checkpoint as load_checkpoint
from raev2.utils.checkpoint import save_stage2_checkpoint as save_checkpoint
from raev2.utils.checkpoint import unwrap_model as unwrap
from raev2.utils.dist_utils import cleanup_distributed, setup_distributed
from raev2.utils.guidance_utils import get_model_forward_fn, is_guidance_active
from raev2.utils.model_utils import instantiate_from_config
from raev2.utils.optim_utils import build_optimizer, build_scheduler
from raev2.utils.resume_utils import configure_experiment_dirs, experiment_name_from_config, find_resume_checkpoint
from raev2.utils.train_utils import autocast_kwargs_from_config as autocast_kwargs
from raev2.utils.train_utils import prepare_stage2_model_config, update_ema
from utils import ConfigLoader
from utils.artifacts import copy_preview_images, save_preview_grid
from utils.rng import seed_all as set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="RAEv2 Stage-2 training entrypoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--results-dir", default="experiments/raev2/stage2")
    parser.add_argument("--resume", default=None, help="Full training checkpoint to resume.")
    parser.add_argument("--init-ckpt", default=None, help="Weights-only initialization checkpoint.")
    parser.add_argument("--fresh-start", action="store_true", help="Do not auto-resume from the experiment directory.")
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after this many optimizer steps.")
    parser.add_argument("--wandb", action="store_true")
    return parser.parse_args()


def build_loader(dataset, cfg, rank: int, world_size: int, micro_batch_size: int):
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=int(cfg.training.global_seed),
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        sampler=sampler,
        num_workers=int(cfg.training.num_workers),
        pin_memory=True,
        drop_last=True,
    )
    return loader, sampler


def build_eval_input_dataset(cfg, eval_cfg):
    if "eval_input_dataset" in cfg:
        eval_ds_cfg = dict(OmegaConf.to_container(cfg.eval_input_dataset, resolve=True))
    else:
        eval_ds_cfg = dict(OmegaConf.to_container(cfg.dataset, resolve=True))
    if "is_train" in eval_ds_cfg:
        eval_ds_cfg["is_train"] = False
    return build_dataset(OmegaConf.create(eval_ds_cfg))


def ensure_optimizer_available(cfg) -> None:
    if str(cfg.training.optimizer.type).lower() == "gmuon" and importlib.util.find_spec("gram_newton_schulz") is None:
        raise RuntimeError(
            "RAEv2 optimizer is gmuon, but package 'gram_newton_schulz' is not installed in this Python env. "
            "Install the upstream dependency or change training.optimizer.type explicitly in YAML for a smoke run."
        )


def load_init_weights(model, ckpt_path: str, logger, *, strict: bool = True, allowed_missing_prefixes=()) -> None:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "ema" in checkpoint:
        state = checkpoint["ema"]
        source = "ema"
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
        source = "model"
    else:
        state = checkpoint
        source = "raw"
    if strict:
        model.load_state_dict(state, strict=True)
        logger.info(f"Initialized Stage2 model from {ckpt_path} ({source}, strict=True).")
        return

    load_msg = model.load_state_dict(state, strict=False)
    missing = list(getattr(load_msg, "missing_keys", []))
    unexpected = list(getattr(load_msg, "unexpected_keys", []))
    allowed_missing_prefixes = tuple(str(v) for v in allowed_missing_prefixes)
    disallowed_missing = [
        key for key in missing
        if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
    ]
    if disallowed_missing or unexpected:
        raise RuntimeError(
            "Non-strict Stage2 init found incompatible keys: "
            f"missing={disallowed_missing}, unexpected={unexpected}"
        )
    logger.info(
        f"Initialized Stage2 model from {ckpt_path} ({source}, strict=False); "
        f"allowed_missing={missing}"
    )


def build_eval_payload_base(cfg):
    missing = [k for k in ("real_set", "eval_dataset", "dataloader", "metrics") if k not in cfg]
    if missing:
        raise ValueError(f"RAEv2 Stage2 eval missing: {', '.join(missing)}")
    return {
        "real_set": OmegaConf.to_container(cfg.real_set, resolve=True),
        "dataset": OmegaConf.to_container(cfg.eval_dataset, resolve=True),
        "dataloader": OmegaConf.to_container(cfg.dataloader, resolve=True),
        "metrics": OmegaConf.to_container(cfg.metrics, resolve=True),
    }


def checkpoint_interval_epochs(cfg) -> int:
    policy = cfg.training.get("checkpoint_policy")
    if policy is not None and "interval_epochs" in policy:
        return int(policy.interval_epochs)
    return int(cfg.training.get("checkpoint_interval", 0))


def save_epoch_checkpoint(cfg, checkpoint_dir: Path, epoch: int, global_step: int, ddp_model, ema_model, optimizer, scheduler) -> None:
    epoch_number = epoch + 1
    interval = checkpoint_interval_epochs(cfg)
    policy = cfg.training.get("checkpoint_policy", {})
    save_last_every_epoch = bool(policy.get("save_last_every_epoch", False))
    if interval > 0 and epoch_number % interval == 0:
        save_checkpoint(checkpoint_dir / f"ep-{epoch_number:07d}.pt", global_step, epoch, ddp_model, ema_model, optimizer, scheduler)
        save_checkpoint(checkpoint_dir / "ep-last.pt", global_step, epoch, ddp_model, ema_model, optimizer, scheduler)
    elif save_last_every_epoch:
        save_checkpoint(checkpoint_dir / "ep-last.pt", global_step, epoch, ddp_model, ema_model, optimizer, scheduler)


def run_epoch_eval(
    cfg,
    eval_payload_base,
    eval_cfg,
    eval_input_dataset,
    *,
    epoch: int,
    global_step: int,
    ema_model,
    rae,
    transport,
    latent_size,
    auto,
    device,
    experiment_dir: Path,
    rank: int,
    logger,
) -> None:
    eval_tag = f"ep{epoch + 1:03d}_ema"
    fake_dir = experiment_dir / str(eval_cfg.get("save_dir", "sample")) / eval_tag
    preview_dir = experiment_dir / str(eval_cfg.get("preview_dir", "sample_preview")) / eval_tag
    model_fn, sample_kwargs = get_model_forward_fn(ema_model, cfg.guidance)
    sample_fn = Sampler(transport, cfg.guidance).sample_ode(num_steps=int(cfg.sampler.num_steps))
    use_guidance = is_guidance_active(cfg.guidance)
    gen_batch = int(eval_cfg.gen_batch_size)
    if gen_batch <= 0:
        raise ValueError("eval.gen_batch_size must be > 0.")

    def sample_fn_with_context(z, wrapped_model_fn, y, **kwargs):
        kwargs.update(context=y, attn_mask=None)
        return sample_fn(z, wrapped_model_fn, **kwargs)

    labels = None
    if rank == 0:
        num_classes = int(cfg.misc.num_classes)
        label_source = str(eval_cfg.get("label_source", "dataset")).lower()
        if label_source == "metadata":
            metadata_path = eval_cfg.get("label_metadata_path")
            if not metadata_path:
                raise ValueError("eval.label_source=metadata requires eval.label_metadata_path")
            label_values, label_manifest = load_metadata_label_plan(
                metadata_path,
                int(eval_cfg.num_samples),
                label_key=str(eval_cfg.get("label_key", "label")),
                num_classes=num_classes,
            )
            labels = torch.tensor(label_values, device=device, dtype=torch.long)
            logger.info(f"Eval label protocol: {label_manifest}")
        elif label_source == "dataset":
            if eval_input_dataset is None:
                raise ValueError("eval.label_source=dataset requires eval_input_dataset")
            labels = get_class_labels(eval_input_dataset, int(eval_cfg.num_samples), device=device)
        else:
            raise ValueError(f"Unsupported eval.label_source={label_source!r}; use metadata or dataset")
        if torch.any((labels < 0) | (labels >= num_classes)):
            raise ValueError(f"Evaluation labels fall outside [0, {num_classes}).")
    n_eval = save_class_cond_fakes_ddp(
        model_fn,
        sample_fn_with_context,
        rae,
        latent_size,
        int(cfg.misc.num_classes),
        int(cfg.misc.get("null_label", cfg.misc.num_classes)),
        sample_kwargs,
        use_guidance,
        int(eval_cfg.num_samples),
        gen_batch,
        device,
        fake_dir,
        seed=int(eval_cfg.get("seed", 42)),
        autocast_kwargs=auto,
        clean_output=True,
        labels=labels,
    )
    payload = dict(eval_payload_base)
    payload["dataset"] = dict(payload["dataset"])
    payload["dataset"]["pred_dir"] = str(fake_dir)
    metrics = evaluate_config(payload, device, use_tqdm=False, distributed=True)
    if rank == 0:
        if int(eval_cfg.get("preview_max_images", 0)) > 0:
            copy_preview_images(fake_dir, preview_dir, max_images=int(eval_cfg.preview_max_images))
            save_preview_grid(fake_dir, preview_dir / "preview.png", max_images=int(eval_cfg.preview_max_images))
        log_metrics = {f"eval/{str(k).lower()}": float(v) for k, v in metrics.items()}
        log_metrics["eval/n"] = float(n_eval)
        logger.info(", ".join(f"{k}: {v:.6g}" for k, v in log_metrics.items()))
        if not bool(eval_cfg.get("keep_pngs", True)):
            shutil.rmtree(fake_dir, ignore_errors=True)
    if dist.is_initialized():
        dist.barrier()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("RAEv2 Stage-2 training requires CUDA.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    rank, world_size, device = setup_distributed()
    cfg = OmegaConf.create(ConfigLoader.load_recursive(args.config))
    seed = int(cfg.training.global_seed) * world_size + rank
    set_seed(seed)
    ensure_optimizer_available(cfg)
    prepare_stage2_model_config(cfg)

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

    dataset = build_dataset(cfg.dataset)
    global_batch = int(cfg.training.global_batch_size)
    grad_accum = int(cfg.training.grad_accum_steps)
    if global_batch % (world_size * grad_accum) != 0:
        raise ValueError("training.global_batch_size must be divisible by world_size * grad_accum_steps.")
    micro_batch = global_batch // (world_size * grad_accum)
    loader, sampler = build_loader(dataset, cfg, rank, world_size, micro_batch)
    train_batches_per_epoch = (len(loader) // grad_accum) * grad_accum
    steps_per_epoch = train_batches_per_epoch // grad_accum
    if steps_per_epoch <= 0:
        raise ValueError("No optimizer steps per epoch. Check batch size and dataset length.")

    rae = instantiate_from_config(cfg.stage_1).to(device).eval()
    rae.requires_grad_(False)

    repa_target_encoder = setup_repa_target_encoder(cfg, rank, device, logger)
    init_ckpt = args.init_ckpt or cfg.stage_2.get("ckpt")
    cfg.stage_2.ckpt = None
    if str(cfg.conditioning.type) != "label":
        raise NotImplementedError("This project Stage2 entry currently supports label conditioning only.")
    model = instantiate_from_config(cfg.stage_2).to(device)
    if init_ckpt and resume_ckpt is None:
        load_init_weights(
            model,
            init_ckpt,
            logger,
            strict=bool(cfg.stage_2.get("init_strict", True)),
            allowed_missing_prefixes=cfg.stage_2.get("allowed_missing_prefixes", []),
        )
    ema_model = deepcopy(model).to(device).eval()
    ema_model.requires_grad_(False)

    if bool(cfg.training.get("compile", False)):
        transport_compile = True
    else:
        transport_compile = False

    ddp_model = DDP(model, device_ids=[device.index], broadcast_buffers=False) if dist.is_initialized() else model
    model = unwrap(ddp_model)

    optimizer, opt_msg = build_optimizer([p for p in model.parameters() if p.requires_grad], cfg.training.optimizer)
    scheduler = None
    sched_msg = None
    if cfg.training.get("scheduler") is not None:
        scheduler, sched_msg = build_scheduler(optimizer, steps_per_epoch, cfg.training.scheduler)

    latent_size = tuple(int(v) for v in cfg.misc.latent_size)
    shift_dim = cfg.misc.time_dist_shift_dim if "time_dist_shift_dim" in cfg.misc else math.prod(latent_size)
    time_dist_shift = math.sqrt(float(shift_dim) / float(cfg.misc.time_dist_shift_base))
    transport = Transport(
        prediction=str(cfg.transport.prediction),
        time_dist_type=str(cfg.transport.time_dist_type),
        time_dist_shift=time_dist_shift,
        t_eps=float(cfg.transport.get("t_eps", 0.05)),
    )
    if transport_compile:
        transport.training_losses = torch.compile(transport.training_losses)

    start_epoch = 0
    global_step = 0
    if resume_ckpt:
        start_epoch, global_step = load_checkpoint(resume_ckpt, ddp_model, ema_model, optimizer, scheduler)
        start_epoch += 1
        logger.info(f"Resumed from {resume_ckpt} at epoch={start_epoch}, step={global_step}.")

    if rank == 0:
        logger.info(f"Dataset length: {len(dataset)}; micro_batch/rank={micro_batch}; steps/epoch={steps_per_epoch}")
        if train_batches_per_epoch != len(loader):
            logger.info(
                f"Dropping {len(loader) - train_batches_per_epoch} trailing micro-batches "
                "to keep gradient accumulation within the current epoch."
            )
        logger.info(f"Optimizer: {opt_msg}")
        if sched_msg:
            logger.info(f"Scheduler: {sched_msg}")
        logger.info(f"time_dist_shift={time_dist_shift:.6g}")

    auto = autocast_kwargs(cfg)
    eval_cfg = cfg.get("eval", {})
    eval_epoch_interval = int(eval_cfg.get("epoch_interval", 0))
    do_eval = bool(eval_cfg.get("enabled", False)) and eval_epoch_interval > 0
    if do_eval:
        eval_payload_base = build_eval_payload_base(cfg)
        eval_input_dataset = (
            None
            if str(eval_cfg.get("label_source", "dataset")).lower() == "metadata"
            else build_eval_input_dataset(cfg, eval_cfg)
        )
    elif rank == 0 and bool(eval_cfg.get("enabled", False)) and int(eval_cfg.get("eval_interval", 0)) > 0:
        logger.info("eval.eval_interval is configured, but this entry only runs epoch eval via eval.epoch_interval.")
    total_steps = int(cfg.training.epochs) * steps_per_epoch
    progress = tqdm(total=total_steps, initial=global_step, disable=rank != 0, desc="RAEv2 Stage2")
    model_kwargs_null = None
    last_epoch = start_epoch - 1

    for epoch in range(start_epoch, int(cfg.training.epochs)):
        model.train()
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        for step, batch in enumerate(loader):
            if step >= train_batches_per_epoch:
                break
            images = batch["img"].to(device, non_blocking=True)
            labels = batch["y"].to(device, non_blocking=True).long()
            if model_kwargs_null is None or model_kwargs_null["context"].shape[0] != labels.shape[0]:
                model_kwargs_null = get_null_cond(None, cfg.conditioning.type, int(cfg.misc.num_classes), labels.shape[0], device)

            with torch.no_grad():
                z = rae.encode(images)
                if repa_target_encoder is not None:
                    raw_images = images.clone() * 255.0
                    with torch.amp.autocast(device_type=device.type, **auto):
                        z_clean = repa_target_encoder.forward_features(
                            repa_target_encoder.preprocess(raw_images)
                        )["x_norm_patchtokens"]
                else:
                    z_clean = None

            model_kwargs = dict(context=labels, attn_mask=None)
            sync_now = (step + 1) % grad_accum == 0
            sync_context = nullcontext() if sync_now or not isinstance(ddp_model, DDP) else ddp_model.no_sync()
            with sync_context:
                with torch.amp.autocast(device_type=device.type, **auto):
                    loss_dict = transport.training_losses(
                        ddp_model,
                        z,
                        model_kwargs=model_kwargs,
                        model_kwargs_null=model_kwargs_null,
                        z_clean=z_clean,
                        repa_coeff=float(cfg.repa.repa_coeff) if bool(cfg.repa.get("use_repa", False)) else None,
                        base_model_coeff=float(cfg.get("internal_guidance", {}).get("base_model_coeff", 1.0)),
                        cfg_dropout_prob=float(cfg.conditioning.get("cfg_dropout_prob", 0.1)),
                    )
                    loss_diff = loss_dict["loss"].mean()
                    loss_repa = loss_dict.get("loss_repa", loss_diff.new_zeros(())).mean()
                    loss = (loss_diff + loss_repa) / grad_accum
                loss.backward()

            if not sync_now:
                continue

            if cfg.training.get("clip_grad") is not None:
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), float(cfg.training.clip_grad))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            update_ema(ema_model, model, decay=float(cfg.training.ema_decay))
            global_step += 1
            progress.update(1)

            if rank == 0 and int(cfg.training.log_interval) > 0 and global_step % int(cfg.training.log_interval) == 0:
                stats = {"loss": loss_diff.item(), "lr": optimizer.param_groups[0]["lr"]}
                if "loss_repa" in loss_dict:
                    stats["loss_repa"] = loss_repa.item()
                if "loss_base" in loss_dict:
                    stats["loss_base"] = loss_dict["loss_base"].mean().item()
                logger.info(f"[Epoch {epoch} | Step {global_step}] " + ", ".join(f"{k}: {v:.6g}" for k, v in stats.items()))
            if args.max_steps is not None and global_step >= args.max_steps:
                break

        last_epoch = epoch
        if rank == 0:
            save_epoch_checkpoint(cfg, checkpoint_dir, epoch, global_step, ddp_model, ema_model, optimizer, scheduler)
        if dist.is_initialized():
            dist.barrier()
        if do_eval and eval_epoch_interval > 0 and (epoch + 1) % eval_epoch_interval == 0:
            run_epoch_eval(
                cfg,
                eval_payload_base,
                eval_cfg,
                eval_input_dataset,
                epoch=epoch,
                global_step=global_step,
                ema_model=ema_model,
                rae=rae,
                transport=transport,
                latent_size=latent_size,
                auto=auto,
                device=device,
                experiment_dir=experiment_dir,
                rank=rank,
                logger=logger,
            )
        if args.max_steps is not None and global_step >= args.max_steps:
            break

    progress.close()
    if rank == 0:
        save_checkpoint(checkpoint_dir / "ep-last.pt", global_step, last_epoch, ddp_model, ema_model, optimizer, scheduler)
        logger.info(f"Done. epoch={last_epoch}, step={global_step}")
    if dist.is_initialized():
        dist.barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
