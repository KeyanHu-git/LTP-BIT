#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
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
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm

try:
    import torch._functorch.config as _ft_cfg
except ModuleNotFoundError:
    _ft_cfg = None
else:
    _ft_cfg.donated_buffer = False

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (str(ROOT), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from Dataset.datasets.utils import build_dataset_from_config as build_dataset
from Dataset.samplers import OnlineDistributedBatchSampler
from Evaluation import evaluate_config
from raev2.configs import load_config
from raev2.stage1.disc import GramLoss, LPIPS, build_discriminator, select_gan_losses, calculate_adaptive_weight
from raev2.utils.checkpoint import load_stage1_checkpoint as load_checkpoint
from raev2.utils.checkpoint import save_stage1_checkpoint as save_checkpoint
from raev2.utils.checkpoint import unwrap_model as unwrap
from raev2.utils.dist_utils import cleanup_distributed, setup_distributed
from raev2.utils.model_utils import instantiate_from_config
from raev2.utils.optim_utils import build_optimizer, build_scheduler
from raev2.utils.resume_utils import configure_experiment_dirs, experiment_name_from_config, find_resume_checkpoint
from raev2.utils.train_utils import autocast_kwargs_from_config as autocast_kwargs
from raev2.utils.train_utils import update_ema
from utils.artifacts import prepare_png_output_dir, save_preview_grid
from utils.logging import format_metrics
from utils.rng import seed_all as set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="RAEv2 Stage-1 RAE training entrypoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--results-dir", default="experiments/raev2/stage1")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--fresh-start", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--wandb", action="store_true")
    return parser.parse_args()


def _cfg_get(container, key: str, default=None):
    return container.get(key, default) if container is not None else default


def _parse_string_list(value) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]


def _parse_float_list(value) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [float(item.strip()) for item in value.split(",") if item.strip()]
    return [float(item) for item in value]


def _resolve_noise_tau(cfg, global_step: int, steps_per_epoch: int, default_tau: float) -> float:
    schedule = cfg.training.get("noise_tau_schedule")
    if schedule is None or not bool(schedule.get("enabled", False)):
        return default_tau

    start_epoch = float(schedule.get("start_epoch", 0.0))
    end_epoch = float(schedule.get("end_epoch", start_epoch))
    start_value = float(schedule.get("start_value", 0.0))
    end_value = float(schedule.get("end_value", default_tau))

    epoch = float(global_step) / max(float(steps_per_epoch), 1.0)
    if epoch <= start_epoch:
        return start_value
    if epoch >= end_epoch:
        return end_value
    if end_epoch <= start_epoch:
        return end_value
    ratio = (epoch - start_epoch) / (end_epoch - start_epoch)
    return start_value + ratio * (end_value - start_value)


def build_loader(dataset, cfg, rank: int, world_size: int, batch_size: int, grad_accum: int, max_train_steps: int | None, clock_interval_steps: int | None):
    sampler_shuffle = not bool(getattr(dataset, "shuffle_shards", False))
    num_workers = int(cfg.training.num_workers)
    online_cfg = cfg.training.get("online_sampling", {})
    if bool(_cfg_get(online_cfg, "enabled", False)):
        if max_train_steps is None:
            raise ValueError("training.online_sampling.enabled requires training.max_train_steps or --max-steps.")
        if clock_interval_steps is None:
            raise ValueError("training.online_sampling.enabled requires training.clock_interval_steps.")
        sampler = OnlineDistributedBatchSampler(
            dataset,
            batch_size=batch_size,
            micro_batches_per_epoch=int(clock_interval_steps) * int(grad_accum),
            max_train_steps=int(max_train_steps),
            grad_accum_steps=int(grad_accum),
            num_replicas=world_size,
            rank=rank,
            seed=int(cfg.training.global_seed),
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
        return loader, sampler

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=sampler_shuffle, seed=int(cfg.training.global_seed), drop_last=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    return loader, sampler


def save_rgb(path: Path, img: torch.Tensor) -> None:
    arr = img.detach().cpu().clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    Image.fromarray(arr).save(path)


def _set_stage1_train_mode(ddp_model, model) -> None:
    ddp_model.train()
    model.encoder.eval()
    model.decoder.train()


def _set_requires_grad(module, enabled: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(enabled)


def _as_plain_config(value):
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def _eval_job_value(suite_cfg, eval_cfg, key: str, default=None):
    value = suite_cfg.get(key, None) if suite_cfg is not None else None
    if value is not None:
        return value
    return eval_cfg.get(key, default)


def _build_eval_jobs(cfg, eval_cfg):
    suites = eval_cfg.get("suites", None)
    raw_suites = list(suites) if suites is not None else [None]
    multi_eval = suites is not None
    jobs = []
    for suite_idx, suite_cfg in enumerate(raw_suites):
        suite_cfg = suite_cfg or {}
        required = ("real_set", "eval_dataset", "dataloader", "metrics")
        missing = [k for k in required if k not in suite_cfg and k not in cfg]
        if missing:
            raise ValueError(f"RAEv2 Stage1 eval missing: {', '.join(missing)}")

        real_set = _as_plain_config(suite_cfg.get("real_set", cfg.get("real_set")))
        eval_dataset_cfg = _as_plain_config(suite_cfg.get("eval_dataset", cfg.get("eval_dataset")))
        dataloader = _as_plain_config(suite_cfg.get("dataloader", cfg.get("dataloader")))
        metrics = _as_plain_config(suite_cfg.get("metrics", cfg.get("metrics")))
        payload_base = {
            "real_set": real_set,
            "dataset": eval_dataset_cfg,
            "dataloader": dataloader,
            "metrics": metrics,
        }

        if "eval_input_dataset" in suite_cfg:
            input_dataset = build_dataset(suite_cfg.eval_input_dataset)
        elif not multi_eval and "eval_input_dataset" in cfg:
            input_dataset = build_dataset(cfg.eval_input_dataset)
        else:
            eval_ds_cfg = dict(_as_plain_config(cfg.dataset))
            eval_ds_cfg["dataroot"] = str(real_set["image_dir"])
            eval_ds_cfg["is_train"] = False
            input_dataset = build_dataset(OmegaConf.create(eval_ds_cfg))

        name = str(suite_cfg.get("name", real_set.get("name", f"eval_{suite_idx}")))
        save_dir = str(_eval_job_value(suite_cfg, eval_cfg, "save_dir", "eval_recon"))
        preview_dir = str(_eval_job_value(suite_cfg, eval_cfg, "preview_dir", "eval_recon_preview"))
        if multi_eval:
            save_dir = f"{save_dir}/{name}"
            preview_dir = f"{preview_dir}/{name}"
        jobs.append(
            {
                "name": name,
                "payload_base": payload_base,
                "dataset": input_dataset,
                "num_samples": int(_eval_job_value(suite_cfg, eval_cfg, "num_samples")),
                "gen_batch_size": int(_eval_job_value(suite_cfg, eval_cfg, "gen_batch_size")),
                "save_dir": save_dir,
                "preview_dir": preview_dir,
                "preview_max_images": int(_eval_job_value(suite_cfg, eval_cfg, "preview_max_images", 0)),
                "eval_interval": int(_eval_job_value(suite_cfg, eval_cfg, "eval_interval", 0)),
                "log_prefix": f"eval_ema/{name}" if multi_eval else "eval_ema",
            }
        )
    return jobs


def _run_eval_jobs(eval_jobs, ema_model, device, experiment_dir: Path, autocast_kwargs: dict, step: int, logger, rank: int) -> None:
    for job in eval_jobs:
        fake_dir = experiment_dir / job["save_dir"] / f"step{step:08d}_ema"
        n_eval = save_recon_fakes_ddp(
            ema_model,
            job["dataset"],
            num_eval_samples=job["num_samples"],
            batch_size=job["gen_batch_size"],
            device=device,
            out_dir=fake_dir,
            autocast_kwargs=autocast_kwargs,
        )
        if rank == 0 and job["preview_max_images"] > 0:
            preview_dir = experiment_dir / job["preview_dir"] / f"step{step:08d}_ema"
            save_preview_grid(fake_dir, preview_dir / "preview.png", max_images=job["preview_max_images"])
        eval_payload = dict(job["payload_base"])
        eval_payload["dataset"] = dict(eval_payload["dataset"])
        eval_payload["dataset"]["pred_dir"] = str(fake_dir)
        raw = evaluate_config(eval_payload, device, use_tqdm=False)
        if rank == 0:
            prefix = job["log_prefix"]
            metrics = {f"{prefix}/{k.lower()}": float(v) for k, v in raw.items()}
            metrics[f"{prefix}/n"] = float(n_eval)
            logger.info(format_metrics(metrics))


@torch.no_grad()
def save_recon_fakes_ddp(model, val_dataset, num_eval_samples: int, batch_size: int, device, out_dir: Path, autocast_kwargs: dict) -> int:
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    n = min(int(num_eval_samples), len(val_dataset))
    if rank == 0:
        prepare_png_output_dir(out_dir, clean=True)
    if dist.is_initialized():
        dist.barrier()

    start = rank * (n // world_size)
    end = (rank + 1) * (n // world_size) if rank < world_size - 1 else n
    loader = DataLoader(Subset(val_dataset, list(range(start, end))), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    image_paths = getattr(val_dataset, "image_paths", None)
    idx = start
    for batch in loader:
        images = batch["img"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, **autocast_kwargs):
            recons = model(images).float().clamp(0, 1)
        for recon in recons.cpu():
            fname = Path(image_paths[idx]).name if image_paths is not None else f"{idx:06d}.png"
            save_rgb(out_dir / fname, recon)
            idx += 1
    if dist.is_initialized():
        dist.barrier()
    return n


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("RAEv2 Stage-1 training requires CUDA.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    rank, world_size, device = setup_distributed()
    cfg = load_config(args.config)
    seed = int(cfg.training.global_seed) * world_size + rank
    set_seed(seed)

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
    global_batch = int(cfg.training.global_batch_size or int(cfg.training.batch_size) * world_size)
    grad_accum = int(cfg.training.get("grad_accum_steps", 1))
    if grad_accum < 1:
        raise ValueError("training.grad_accum_steps must be >= 1.")
    if global_batch % (world_size * grad_accum) != 0:
        raise ValueError("training.global_batch_size must be divisible by world_size * grad_accum_steps.")
    batch_size = global_batch // (world_size * grad_accum)
    max_train_steps_cfg = cfg.training.get("max_train_steps")
    max_train_steps = int(args.max_steps) if args.max_steps is not None else (int(max_train_steps_cfg) if max_train_steps_cfg is not None else None)
    clock_interval_steps = cfg.training.get("clock_interval_steps")
    clock_interval_steps = int(clock_interval_steps) if clock_interval_steps is not None else None
    loader, sampler = build_loader(dataset, cfg, rank, world_size, batch_size, grad_accum, max_train_steps, clock_interval_steps)
    steps_per_epoch = clock_interval_steps or (len(loader) // grad_accum)
    if steps_per_epoch <= 0:
        raise ValueError("No optimizer steps per epoch. Check batch size and dataset length.")
    if max_train_steps is None:
        max_train_steps = int(cfg.training.epochs) * steps_per_epoch

    eval_cfg = cfg.get("eval")
    do_eval = eval_cfg is not None and bool(eval_cfg.get("enabled", False))
    eval_jobs = []
    if do_eval:
        eval_jobs = _build_eval_jobs(cfg, eval_cfg)

    rae = instantiate_from_config(cfg.stage_1).to(device)
    rae.encoder.eval()
    rae.decoder.train()
    rae.encoder.requires_grad_(False)
    rae.decoder.requires_grad_(True)
    ema_model = deepcopy(rae).to(device).eval()
    ema_model.requires_grad_(False)

    ddp_model = DDP(rae, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=False) if dist.is_initialized() else rae
    model = unwrap(ddp_model)
    decoder = model.decoder
    base_noise_tau = float(getattr(model, "noise_tau", 0.0))

    disc_weight = float(cfg.gan.loss.get("disc_weight", 0.0))
    freeze_discriminator = bool(cfg.gan.loss.get("freeze_discriminator", False))
    if disc_weight > 0:
        discriminator, disc_aug = build_discriminator(cfg.gan.arch, device, cfg.gan.get("augment"))
        ddp_disc = (
            discriminator
            if freeze_discriminator
            else DDP(discriminator, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=False) if dist.is_initialized() else discriminator
        )
        discriminator = unwrap(ddp_disc)
        disc_loss_fn, gen_loss_fn = select_gan_losses(str(cfg.gan.loss.disc_loss), str(cfg.gan.loss.gen_loss))
    else:
        discriminator = None
        disc_aug = None
        ddp_disc = None
        disc_loss_fn = None
        gen_loss_fn = None
    lpips = LPIPS().to(device).eval()
    gram_weight = float(cfg.gan.loss.get("gram_weight", 0.0))
    if gram_weight < 0:
        raise ValueError("gan.loss.gram_weight must be >= 0.")
    gram_loss_fn = None
    if gram_weight > 0:
        gram_loss_fn = GramLoss(
            layers=_parse_string_list(cfg.gan.loss.get("gram_layers", None)),
            mode=str(cfg.gan.loss.get("gram_mode", "legacy")),
            input_range=str(cfg.gan.loss.get("gram_input_range", "minus_one_one")),
            clamp_input=bool(cfg.gan.loss.get("gram_clamp_input", False)),
            weights_path=cfg.gan.loss.get("gram_weights_path", None),
            layer_weights=_parse_float_list(cfg.gan.loss.get("gram_layer_weights", None)),
            pooling=str(cfg.gan.loss.get("gram_pooling", "max")),
        ).to(device).eval()

    init_checkpoint = cfg.training.get("init_checkpoint")
    if init_checkpoint is not None:
        if resume_ckpt is not None:
            raise ValueError("training.init_checkpoint cannot be used together with resume.")
        init_state = torch.load(str(init_checkpoint), map_location="cpu")
        load_model_from_ema = bool(cfg.training.get("init_model_from_ema", False)) and "ema" in init_state
        model_key = "ema" if load_model_from_ema else "model"
        unwrap(ddp_model).load_state_dict(init_state[model_key], strict=True)
        if bool(cfg.training.get("init_load_ema", True)) and "ema" in init_state:
            ema_model.load_state_dict(init_state["ema"], strict=True)
        else:
            ema_model.load_state_dict(unwrap(ddp_model).state_dict(), strict=True)
        if ddp_disc is not None and bool(cfg.training.get("init_load_discriminator", True)) and "disc" in init_state:
            unwrap(ddp_disc).load_state_dict(init_state["disc"], strict=True)
        del init_state
        gc.collect()
        if rank == 0:
            logger.info(
                f"Initialized weights from {init_checkpoint} ({model_key} -> model); "
                "optimizer/scheduler start from config."
            )
    if discriminator is not None and freeze_discriminator:
        discriminator.requires_grad_(False)
        discriminator.eval()
        if rank == 0:
            logger.info("Discriminator is frozen; GAN loss uses fixed discriminator weights.")

    optimizer, optim_msg = build_optimizer(decoder.parameters(), cfg.training.optimizer)
    if discriminator is not None and not freeze_discriminator:
        disc_optimizer, disc_optim_msg = build_optimizer([p for p in discriminator.parameters() if p.requires_grad], cfg.gan.optimizer)
    else:
        disc_optimizer = None
        disc_optim_msg = (
            "disabled (gan.loss.freeze_discriminator=true)"
            if discriminator is not None and freeze_discriminator
            else "disabled (gan.loss.disc_weight <= 0)"
        )
    scheduler = build_scheduler(optimizer, steps_per_epoch, cfg.training.scheduler)[0] if cfg.training.get("scheduler") is not None else None
    disc_scheduler = build_scheduler(disc_optimizer, steps_per_epoch, cfg.gan.scheduler)[0] if disc_optimizer is not None and cfg.gan.get("scheduler") is not None else None

    start_epoch = 0
    global_step = 0
    if resume_ckpt:
        start_epoch, global_step = load_checkpoint(resume_ckpt, ddp_model, ema_model, optimizer, scheduler, ddp_disc, disc_optimizer, disc_scheduler)
        logger.info(f"Resumed from {resume_ckpt} at epoch={start_epoch}, step={global_step}.")

    if rank == 0:
        online_cfg = cfg.training.get("online_sampling", {})
        if bool(_cfg_get(online_cfg, "enabled", False)):
            logger.info(
                "Online metadata sampling: "
                f"max_steps={max_train_steps}; clock_interval_steps={steps_per_epoch}"
            )
        logger.info(f"Dataset length: {len(dataset)}; micro_batch/rank={batch_size}; grad_accum={grad_accum}; steps/clock={steps_per_epoch}")
        logger.info(f"Optimizer: {optim_msg}")
        logger.info(f"Disc optimizer: {disc_optim_msg}")

    auto = autocast_kwargs(cfg)
    if do_eval and bool(eval_cfg.get("eval_at_start", False)) and global_step == 0:
        if rank == 0:
            logger.info("Running initial Stage1 eval at step=0.")
        _run_eval_jobs(eval_jobs, ema_model, device, experiment_dir, auto, global_step, logger, rank)

    progress = tqdm(total=max_train_steps, initial=global_step, disable=rank != 0, desc="RAEv2 Stage1")
    discriminator_only_steps_cfg = cfg.training.get("discriminator_only_steps")
    discriminator_only_epochs = int(cfg.training.get("discriminator_only_epochs", 0))
    if discriminator_only_epochs < 0:
        raise ValueError("training.discriminator_only_epochs must be >= 0.")
    discriminator_only_steps = int(discriminator_only_steps_cfg) if discriminator_only_steps_cfg is not None else discriminator_only_epochs * steps_per_epoch
    last_layer = decoder.decoder_pred.weight
    gan_start_step = int(cfg.gan.loss.get("disc_start_step", int(cfg.gan.loss.disc_start) * steps_per_epoch))
    disc_update_step = int(cfg.gan.loss.get("disc_upd_start_step", int(cfg.gan.loss.disc_upd_start) * steps_per_epoch))
    lpips_start_step = int(cfg.gan.loss.get("lpips_start_step", int(cfg.gan.loss.lpips_start) * steps_per_epoch))
    gram_start_cfg = cfg.gan.loss.get("gram_start", 0)
    gram_start_epoch = int(0 if gram_start_cfg is None else gram_start_cfg)
    gram_start_step_cfg = cfg.gan.loss.get("gram_start_step", None)
    gram_start_step = int(gram_start_step_cfg) if gram_start_step_cfg is not None else gram_start_epoch * steps_per_epoch
    checkpoint_policy = cfg.training.get("checkpoint_policy", {})
    save_numbered = bool(checkpoint_policy.get("save_numbered", False))
    save_last_every_epoch = bool(checkpoint_policy.get("save_last_every_epoch", False))
    checkpoint_interval = int(cfg.training.get("checkpoint_interval", 0))
    last_completed_epoch = start_epoch
    last_saved_epoch = None

    total_epochs = int(math.ceil(max_train_steps / steps_per_epoch))
    for epoch in range(start_epoch, total_epochs):
        _set_stage1_train_mode(ddp_model, model)
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        epoch_metrics = defaultdict(lambda: torch.zeros(1, device=device))
        num_batches = 0
        remaining_steps = max(0, max_train_steps - global_step)
        usable_micro_steps = min(steps_per_epoch, remaining_steps) * grad_accum
        disc_batches = []
        disc_fake_batches = []
        optimizer.zero_grad(set_to_none=True)
        for micro_step, batch in enumerate(loader):
            if micro_step >= usable_micro_steps:
                break
            if global_step >= max_train_steps:
                break
            images = batch["img"].to(device, non_blocking=True)
            disc_batches.append(images.detach())
            real_normed = images * 2.0 - 1.0
            current_noise_tau = _resolve_noise_tau(cfg, global_step, steps_per_epoch, base_noise_tau)
            model.noise_tau = current_noise_tau
            use_gan = global_step >= gan_start_step and disc_weight > 0
            discriminator_only_phase = global_step < discriminator_only_steps
            train_disc = (
                (global_step >= disc_update_step or discriminator_only_phase)
                and disc_weight > 0
                and not freeze_discriminator
            )
            use_lpips = global_step >= lpips_start_step and float(cfg.gan.loss.perceptual_weight) > 0
            use_gram = gram_loss_fn is not None and global_step >= gram_start_step
            use_generator_updates = not discriminator_only_phase

            if discriminator is not None:
                discriminator.eval()
                if use_gan and not freeze_discriminator:
                    _set_requires_grad(discriminator, False)
            sync_now = (micro_step + 1) % grad_accum == 0
            sync_context = nullcontext() if sync_now or not isinstance(ddp_model, DDP) else ddp_model.no_sync()
            with sync_context:
                if use_generator_updates:
                    with torch.amp.autocast(device_type=device.type, **auto):
                        recon = ddp_model(images)
                        recon_normed = recon * 2.0 - 1.0
                        rec_loss = (recon - images).abs().mean()
                        if use_gram and getattr(gram_loss_fn, "uses_lpips_features", False):
                            feats_real, feats_recon = lpips.extract_features_pair(real_normed, recon_normed)
                            lpips_loss = lpips.forward_from_features(feats_real, feats_recon) if use_lpips else rec_loss.new_zeros(())
                            gram_loss = gram_loss_fn.forward_from_features(feats_real, feats_recon)
                        else:
                            lpips_loss = lpips(real_normed, recon_normed) if use_lpips else rec_loss.new_zeros(())
                            gram_loss = gram_loss_fn(real_normed, recon_normed) if use_gram else rec_loss.new_zeros(())
                        recon_total = rec_loss + float(cfg.gan.loss.perceptual_weight) * lpips_loss + gram_weight * gram_loss
                        if use_gan:
                            logits_fake, _ = ddp_disc(disc_aug.aug(recon_normed), None)
                            gan_loss = gen_loss_fn(logits_fake)
                        else:
                            gan_loss = recon_total.new_zeros(())
                    fake_for_disc = None
                    if use_gan:
                        adaptive_weight = calculate_adaptive_weight(recon_total, gan_loss, last_layer, float(cfg.gan.loss.max_d_weight))
                        total_loss = recon_total + disc_weight * adaptive_weight * gan_loss
                    else:
                        adaptive_weight = recon_total.new_zeros(())
                        total_loss = recon_total
                    (total_loss / grad_accum).backward()
                else:
                    with torch.no_grad():
                        with torch.amp.autocast(device_type=device.type, **auto):
                            recon = ddp_model(images)
                            recon_normed = recon * 2.0 - 1.0
                            rec_loss = (recon - images).abs().mean()
                            if use_gram and getattr(gram_loss_fn, "uses_lpips_features", False):
                                feats_real, feats_recon = lpips.extract_features_pair(real_normed, recon_normed)
                                lpips_loss = lpips.forward_from_features(feats_real, feats_recon) if use_lpips else recon.new_zeros(())
                                gram_loss = gram_loss_fn.forward_from_features(feats_real, feats_recon)
                            else:
                                lpips_loss = lpips(real_normed, recon_normed) if use_lpips else recon.new_zeros(())
                                gram_loss = gram_loss_fn(real_normed, recon_normed) if use_gram else recon.new_zeros(())
                    fake_for_disc = recon_normed.detach()
                    recon_total = rec_loss + float(cfg.gan.loss.perceptual_weight) * lpips_loss + gram_weight * gram_loss
                    gan_loss = recon_total.new_zeros(())
                    adaptive_weight = recon_total.new_zeros(())
                    total_loss = recon_total
            disc_fake_batches.append(fake_for_disc)

            epoch_metrics["recon"] += rec_loss.detach()
            epoch_metrics["lpips"] += lpips_loss.detach()
            if gram_loss_fn is not None:
                epoch_metrics["gram"] += gram_loss.detach()
                epoch_metrics["gram_weighted"] += (gram_weight * gram_loss).detach()
            epoch_metrics["gan"] += gan_loss.detach()
            epoch_metrics["total"] += total_loss.detach()
            num_batches += 1
            if not sync_now:
                continue

            if use_generator_updates and cfg.training.get("clip_grad") is not None and float(cfg.training.clip_grad) > 0:
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), float(cfg.training.clip_grad))
            if use_generator_updates:
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                update_ema(ema_model, model, decay=float(cfg.training.ema_decay))
                optimizer.zero_grad(set_to_none=True)
            else:
                optimizer.zero_grad(set_to_none=True)

            disc_metrics = {}
            if train_disc and ddp_disc is not None:
                ddp_model.eval()
                ddp_disc.train()
                if not freeze_discriminator:
                    _set_requires_grad(discriminator, True)
                for _ in range(int(cfg.gan.loss.disc_updates)):
                    disc_optimizer.zero_grad(set_to_none=True)
                    d_loss_total = real_normed.new_zeros(())
                    accuracy_total = real_normed.new_zeros(())
                    logits_real_total = real_normed.new_zeros(())
                    logits_fake_total = real_normed.new_zeros(())
                    for disc_idx, disc_images in enumerate(disc_batches):
                        disc_real_normed = disc_images * 2.0 - 1.0
                        sync_disc = (disc_idx + 1) == len(disc_batches) or not isinstance(ddp_disc, DDP)
                        disc_sync_context = nullcontext() if sync_disc else ddp_disc.no_sync()
                        with disc_sync_context:
                            with torch.amp.autocast(device_type=device.type, **auto):
                                recon_disc = disc_fake_batches[disc_idx]
                                if recon_disc is None:
                                    with torch.no_grad():
                                        recon_disc = ddp_model(disc_images) * 2.0 - 1.0
                                fake_detached = recon_disc.clamp(-1.0, 1.0)
                                fake_detached = torch.round((fake_detached + 1.0) * 127.5) / 127.5 - 1.0
                                logits_fake, logits_real = ddp_disc(disc_aug.aug(fake_detached), disc_aug.aug(disc_real_normed))
                                d_loss = disc_loss_fn(logits_real, logits_fake)
                                accuracy = (logits_real > logits_fake).float().mean()
                            (d_loss / len(disc_batches)).backward()
                        d_loss_total += d_loss.detach()
                        accuracy_total += accuracy.detach()
                        logits_real_total += logits_real.detach().mean()
                        logits_fake_total += logits_fake.detach().mean()
                    disc_optimizer.step()
                    if disc_scheduler is not None:
                        disc_scheduler.step()
                    disc_optimizer.zero_grad(set_to_none=True)
                    inv_disc_batches = 1.0 / len(disc_batches)
                    disc_metrics = {
                        "loss/disc": d_loss_total * inv_disc_batches,
                        "disc/logits_real": logits_real_total * inv_disc_batches,
                        "disc/logits_fake": logits_fake_total * inv_disc_batches,
                        "disc/accuracy": accuracy_total * inv_disc_batches,
                    }
                    epoch_metrics["disc"] += disc_metrics["loss/disc"]
                _set_stage1_train_mode(ddp_model, model)
                ddp_disc.eval()
            disc_batches = []
            disc_fake_batches = []

            if rank == 0 and int(cfg.training.log_interval) > 0 and global_step % int(cfg.training.log_interval) == 0:
                stats = {
                    "loss/total": total_loss.detach().item(),
                    "loss/recon": rec_loss.detach().item(),
                    "loss/lpips": lpips_loss.detach().item(),
                    "loss/gan": gan_loss.detach().item(),
                    "train/noise_tau": current_noise_tau,
                    "lr/generator": 0.0 if not use_generator_updates else optimizer.param_groups[0]["lr"],
                }
                if gram_loss_fn is not None:
                    stats["loss/gram"] = gram_loss.detach().item()
                    stats["loss/gram_weighted"] = (gram_weight * gram_loss).detach().item()
                if disc_metrics:
                    stats.update({k: v.item() for k, v in disc_metrics.items()})
                    stats["disc/weight"] = adaptive_weight.item()
                logger.info(f"[Epoch {epoch} | Step {global_step}] {format_metrics(stats)}")

            if rank == 0 and int(cfg.training.sample_every) > 0 and global_step % int(cfg.training.sample_every) == 0:
                with torch.no_grad():
                    samples = ema_model(images[:4]).clamp(0, 1)
                    grid = make_grid(torch.cat([images[:4].cpu(), samples.cpu()], dim=0), nrow=4)
                    out = experiment_dir / "sample_preview" / f"step{global_step:08d}.png"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    save_image(grid, out)

            completed_step = global_step + 1
            if do_eval:
                due_eval_jobs = [
                    job for job in eval_jobs
                    if job["eval_interval"] > 0 and completed_step % job["eval_interval"] == 0
                ]
                if due_eval_jobs:
                    _run_eval_jobs(due_eval_jobs, ema_model, device, experiment_dir, auto, completed_step, logger, rank)

            global_step += 1
            progress.update(1)
            if global_step >= max_train_steps:
                break

        if rank == 0 and num_batches > 0:
            epoch_to_save = epoch + 1
            stats = {
                "epoch/loss_total": (epoch_metrics["total"] / num_batches).item(),
                "epoch/loss_recon": (epoch_metrics["recon"] / num_batches).item(),
                "epoch/loss_lpips": (epoch_metrics["lpips"] / num_batches).item(),
                "epoch/loss_gan": (epoch_metrics["gan"] / num_batches).item(),
            }
            if gram_loss_fn is not None:
                stats["epoch/loss_gram"] = (epoch_metrics["gram"] / num_batches).item()
                stats["epoch/loss_gram_weighted"] = (epoch_metrics["gram_weighted"] / num_batches).item()
            logger.info(f"[Epoch {epoch}] {format_metrics(stats)}")
            if save_last_every_epoch:
                save_checkpoint(checkpoint_dir / "ep-last.pt", global_step, epoch_to_save, ddp_model, ema_model, optimizer, scheduler, ddp_disc, disc_optimizer, disc_scheduler)
                last_saved_epoch = epoch_to_save
            if save_numbered and checkpoint_interval > 0 and epoch_to_save % checkpoint_interval == 0:
                save_checkpoint(checkpoint_dir / f"ep-{epoch_to_save:07d}.pt", global_step, epoch_to_save, ddp_model, ema_model, optimizer, scheduler, ddp_disc, disc_optimizer, disc_scheduler)
        last_completed_epoch = epoch + 1
        if global_step >= max_train_steps:
            break

    progress.close()
    if rank == 0:
        if last_saved_epoch != last_completed_epoch:
            save_checkpoint(checkpoint_dir / "ep-last.pt", global_step, last_completed_epoch, ddp_model, ema_model, optimizer, scheduler, ddp_disc, disc_optimizer, disc_scheduler)
        logger.info(f"Done. epoch={last_completed_epoch}, step={global_step}")
    if dist.is_initialized():
        dist.barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
