#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (str(ROOT), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from Dataset.datasets.utils import build_dataset_from_config as build_dataset
from Evaluation import evaluate_config
from raev2.eval.paired_i2i import paired_output_names, save_paired_i2i_fakes_ddp
from raev2.stage2.transport import Sampler, Transport
from raev2.stage3.lora import apply_lora_from_config, set_lora_enabled
from raev2.utils.checkpoint import (
    STAGE3_CHECKPOINT_FORMAT,
    align_rope_buffers,
    load_stage3_inference_checkpoint,
)
from raev2.utils.config_utils import cfg_to_dict
from raev2.utils.dist_utils import cleanup_distributed, setup_distributed
from raev2.utils.guidance_utils import get_model_forward_fn, is_guidance_active, uses_internal_guidance
from raev2.utils.model_utils import instantiate_from_config
from raev2.utils.train_utils import prepare_stage2_model_config
from utils import ConfigLoader
from utils.rng import seed_all


def parse_args():
    parser = argparse.ArgumentParser(description="RAEv2 Stage-3 paired i2i evaluator.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-dir", default=None)
    parser.add_argument("--save-folder", default=None)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--init-ckpt", default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--per-proc-batch-size", type=int, default=None)
    parser.add_argument("--gen-batch-size", type=int, default=None)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default=None)
    parser.add_argument("--global-seed", type=int, default=None)
    parser.add_argument("--clean-output", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--prefer-ema", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_stage2_base(model, path: str, cfg) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    state = checkpoint.get("ema", checkpoint.get("model", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
    checkpoint_cfg = cfg.get("stage3", {}).get("checkpoint", {})
    strict = bool(checkpoint_cfg.get("strict_init", True))
    state, _ = align_rope_buffers(model, state)
    keys = model.load_state_dict(state, strict=strict)
    if not strict:
        allowed = tuple(checkpoint_cfg.get("allowed_missing_prefixes", []) or ())
        missing = [key for key in keys.missing_keys if not key.startswith(allowed)]
        if missing or keys.unexpected_keys:
            raise RuntimeError(f"Stage2 base mismatch: missing={missing[:20]} unexpected={keys.unexpected_keys[:20]}")


def validate_stage3_scope(cfg) -> None:
    if cfg.get("repa") is not None and bool(cfg.repa.get("use_repa", False)):
        raise NotImplementedError("RAEv2 Stage3 paired i2i does not support REPA yet.")
    if cfg.get("internal_guidance") is not None and cfg.internal_guidance.get("base_model_depth") is not None:
        raise NotImplementedError("RAEv2 Stage3 paired i2i does not support internal guidance yet.")


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("RAEv2 Stage-3 eval requires CUDA.")

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.set_grad_enabled(False)

    cfg = OmegaConf.create(ConfigLoader.load_recursive(args.config))
    if args.init_ckpt:
        cfg.training.init_ckpt = args.init_ckpt
    prepare_stage2_model_config(cfg)
    validate_stage3_scope(cfg)
    inference_cfg = cfg.get("inference", {})
    seed = int(args.global_seed if args.global_seed is not None else inference_cfg.get("seed", cfg.eval.get("seed", cfg.training.global_seed)))
    rank, world_size, device = setup_distributed()
    seed_all(seed)

    ckpt = args.ckpt or inference_cfg.get("ckpt")
    if not ckpt:
        raise ValueError("RAEv2 Stage3 eval requires --ckpt or inference.ckpt.")
    if not Path(ckpt).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    prefer_ema = bool(args.prefer_ema if args.prefer_ema is not None else inference_cfg.get("prefer_ema", True))

    precision = args.precision or inference_cfg.get("precision", cfg.training.get("precision", "bf16"))
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("Requested bf16 precision, but this CUDA device does not support bfloat16.")
    auto = {"enabled": precision == "bf16", "dtype": torch.bfloat16}
    eval_cfg = cfg.get("eval", {})
    num_samples = int(args.num_samples if args.num_samples is not None else inference_cfg.get("num_samples", eval_cfg.get("num_samples", 256)))
    per_proc_batch = int(args.per_proc_batch_size if args.per_proc_batch_size is not None else inference_cfg.get("per_proc_batch_size", 8))
    gen_batch = int(
        args.gen_batch_size
        if args.gen_batch_size is not None
        else inference_cfg.get("gen_batch_size", eval_cfg.get("gen_batch_size", per_proc_batch * world_size))
    )
    sample_dir = Path(args.sample_dir or inference_cfg.get("sample_dir", "experiments/raev2/stage3/QXSLAB_SAROPT"))
    save_folder = args.save_folder or inference_cfg.get("save_folder", "eval")
    clean_output = bool(args.clean_output if args.clean_output is not None else inference_cfg.get("clean_output", False))

    default_sample_model_kwargs = cfg.eval.get("sample_model_kwargs", {}) if cfg.get("eval") is not None else {}
    sample_model_kwargs_cfg = inference_cfg.get("sample_model_kwargs", default_sample_model_kwargs)
    sample_model_kwargs = cfg_to_dict(sample_model_kwargs_cfg)

    rae = instantiate_from_config(cfg.stage_1).to(device).eval()
    rae.requires_grad_(False)
    model = instantiate_from_config(cfg.stage_2).to(device)
    checkpoint_header = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    init_ckpt = cfg.training.get("init_ckpt")
    if isinstance(checkpoint_header, dict) and checkpoint_header.get("format") == STAGE3_CHECKPOINT_FORMAT and checkpoint_header.get("weight_mode") == "adapter":
        if not init_ckpt:
            raise ValueError("Stage3 adapter evaluation requires training.init_ckpt.")
        load_stage2_base(model, str(init_ckpt), cfg)
    lora_cfg = cfg.get("lora", {})
    apply_lora_from_config(model, lora_cfg)
    source = load_stage3_inference_checkpoint(
        checkpoint_header,
        model,
        prefer_ema=prefer_ema,
        init_ckpt=init_ckpt,
    )
    use_lora = bool(inference_cfg.get("use_lora", True))
    lora_layers = set_lora_enabled(model, use_lora)
    if rank == 0:
        print(f"[raev2_stage3] loaded {ckpt} ({source}); use_lora={use_lora} lora_layers={lora_layers}")
    model.eval()
    model.requires_grad_(False)
    if uses_internal_guidance(cfg.guidance) and not hasattr(model, "base_final_layer"):
        raise NotImplementedError("RAEv2 Stage3 IG guidance requires a model with base_final_layer.")
    model_fn, guidance_kwargs = get_model_forward_fn(model, cfg.guidance)
    use_guidance = is_guidance_active(cfg.guidance)
    sample_model_kwargs = {**sample_model_kwargs, **guidance_kwargs}
    null_label = int(cfg.misc.get("null_label", cfg.misc.num_classes))
    if rank == 0:
        print(f"[raev2_stage3] cfg={args.config}")
        print(
            f"[raev2_stage3] out={sample_dir / save_folder} n={num_samples} "
            f"bsz/rank={per_proc_batch} genbs={gen_batch} world={world_size}"
        )
        print(
            f"[raev2_stage3] steps={int(cfg.sampler.num_steps)} "
            f"guidance={OmegaConf.to_container(cfg.guidance, resolve=True)}"
        )
        print(f"[raev2_stage3] sampling protocol=paired_i2i seed={seed}")

    ds_cfg = dict(OmegaConf.to_container(cfg.dataset, resolve=True))
    ds_cfg["is_train"] = False
    ds_cfg["split"] = str(cfg.eval.get("split", "test"))
    val_dataset = build_dataset(OmegaConf.create(ds_cfg))
    expected_names = sorted(paired_output_names(val_dataset, num_samples))
    expected_count = len(expected_names)

    latent_size = tuple(int(v) for v in cfg.misc.latent_size)
    shift_dim = cfg.misc.time_dist_shift_dim if "time_dist_shift_dim" in cfg.misc else math.prod(latent_size)
    time_dist_shift = math.sqrt(float(shift_dim) / float(cfg.misc.time_dist_shift_base))
    if "time_dist_shift" in cfg.transport:
        time_dist_shift = float(cfg.transport.time_dist_shift)
    transport = Transport(
        prediction=str(cfg.transport.prediction),
        time_dist_type=str(cfg.transport.time_dist_type),
        time_dist_shift=time_dist_shift,
        t_eps=float(cfg.transport.get("t_eps", 0.05)),
    )
    sample_fn = Sampler(transport, cfg.guidance).sample_ode(num_steps=int(cfg.sampler.num_steps))
    fake_dir = sample_dir / str(save_folder) / "fake"
    preview_dir = sample_dir / str(save_folder) / "preview"
    # A complete, canonically named image set is a valid generation checkpoint.
    # This lets metric-only retries reuse an existing complete sample set after a metric
    # constructor fails, while partial/non-canonical sets are rejected so two
    # sampling runs cannot be mixed silently.
    resume_state = torch.zeros((), dtype=torch.int64, device=device)
    if rank == 0 and not clean_output:
        existing_names = sorted(
            path.name for path in fake_dir.glob("*.png") if not path.name.startswith("_")
        )
        if existing_names:
            resume_state.fill_(1 if existing_names == expected_names else -len(existing_names))
    if dist.is_initialized():
        dist.broadcast(resume_state, src=0)

    resume_value = int(resume_state.item())
    if resume_value < 0:
        raise RuntimeError(
            "Output directory contains an incomplete or non-canonical PNG set: "
            f"{fake_dir} ({-resume_value} files); expected exactly {expected_count}."
        )
    if resume_value == 1:
        n_eval = expected_count
        if rank == 0:
            print(f"[raev2_stage3] reuse complete generation: {fake_dir} ({expected_count} files)")
    else:
        n_eval = save_paired_i2i_fakes_ddp(
            model_fn=model_fn,
            rae=rae,
            sampler_fn=sample_fn,
            val_dataset=val_dataset,
            num_eval_samples=num_samples,
            gen_batch_size=gen_batch,
            device=device,
            latent_size=latent_size,
            out_dir=fake_dir,
            fixed_label=int(cfg.dataset.get("fixed_label", 0)),
            null_label=null_label,
            use_guidance=use_guidance,
            seed=seed,
            preview_dir=preview_dir,
            preview_max_images=int(cfg.eval.get("preview_max_images", 16)),
            autocast_kwargs=auto,
            clean_output=clean_output,
            sample_model_kwargs=sample_model_kwargs,
        )

    payload = {
        "real_set": OmegaConf.to_container(cfg.real_set, resolve=True),
        "dataset": OmegaConf.to_container(cfg.eval_dataset, resolve=True),
        "dataloader": OmegaConf.to_container(cfg.dataloader, resolve=True),
        "metrics": OmegaConf.to_container(cfg.metrics, resolve=True),
    }
    eval_target_dir = str(getattr(val_dataset, "target_path", payload["real_set"]["image_dir"]))
    payload["real_set"]["image_dir"] = eval_target_dir
    payload["dataset"]["gt_dir"] = eval_target_dir
    payload["dataset"]["pred_dir"] = str(fake_dir)
    metrics = evaluate_config(payload, device, use_tqdm=False, distributed=True)
    if rank == 0:
        record = {f"eval/{str(k).lower()}": float(v) for k, v in metrics.items()} | {"eval/n": float(n_eval)}
        print(record)
        out_dir = fake_dir.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(
            json.dumps(
                {
                    **record,
                    "ckpt": str(ckpt),
                    "config": str(args.config),
                    "num_steps": int(cfg.sampler.num_steps),
                    "seed": int(seed),
                    "guidance": OmegaConf.to_container(cfg.guidance, resolve=True),
                    "use_lora": use_lora,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[raev2_stage3] wrote {out_dir / 'metrics.json'}")

    if dist.is_initialized():
        dist.barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
