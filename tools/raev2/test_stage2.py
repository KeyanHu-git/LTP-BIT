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

from Evaluation import evaluate_config
from raev2.eval.class_cond import save_class_cond_fakes_ddp
from raev2.eval.label_protocol import load_metadata_label_plan
from raev2.stage2.transport import Sampler, Transport
from raev2.utils.dist_utils import cleanup_distributed, setup_distributed
from raev2.utils.guidance_utils import get_model_forward_fn, is_guidance_active
from raev2.utils.model_utils import instantiate_from_config
from raev2.utils.train_utils import prepare_stage2_model_config
from utils import ConfigLoader
from utils.rng import seed_all


def parse_args():
    parser = argparse.ArgumentParser(description="RAEv2 Stage-2 class-conditional sampler.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-dir", default=None)
    parser.add_argument("--save-folder", default=None)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--per-proc-batch-size", type=int, default=None)
    parser.add_argument("--gen-batch-size", type=int, default=None)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default=None)
    parser.add_argument("--global-seed", type=int, default=None)
    parser.add_argument("--clean-output", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("RAEv2 Stage-2 sampling requires CUDA.")

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.set_grad_enabled(False)

    cfg = OmegaConf.create(ConfigLoader.load_recursive(args.config))
    inf_cfg = cfg.get("inference", {})
    seed = int(args.global_seed if args.global_seed is not None else inf_cfg.get("seed", cfg.training.global_seed))
    rank, world_size, device = setup_distributed()
    seed_all(seed)

    precision = args.precision or inf_cfg.get("precision", cfg.training.get("precision", "bf16"))
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("Requested bf16 precision, but this CUDA device does not support bfloat16.")
    autocast_kwargs = {"enabled": precision == "bf16", "dtype": torch.bfloat16}
    if args.ckpt is not None:
        cfg.stage_2.ckpt = args.ckpt
    if not cfg.stage_2.get("ckpt"):
        raise ValueError("RAEv2 Stage2 sampling requires --ckpt or stage_2.ckpt.")

    num_samples = int(args.num_samples if args.num_samples is not None else inf_cfg.get("num_samples", 256))
    per_proc_batch = int(args.per_proc_batch_size if args.per_proc_batch_size is not None else inf_cfg.get("per_proc_batch_size", 8))
    label_source = str(inf_cfg.get("label_source", "random")).lower()
    sample_dir = Path(args.sample_dir or inf_cfg.get("sample_dir", "experiments/raev2/stage2/ImageNet1K"))
    save_folder = args.save_folder or inf_cfg.get("save_folder", "dinov3l_k7_official256_s100_config")
    out_dir = sample_dir / save_folder
    clean_output = bool(args.clean_output if args.clean_output is not None else inf_cfg.get("clean_output", False))
    eval_cfg = cfg.get("eval", {})
    global_gen_batch = int(
        args.gen_batch_size
        if args.gen_batch_size is not None
        else inf_cfg.get("gen_batch_size", eval_cfg.get("gen_batch_size", per_proc_batch * world_size))
    )

    if rank == 0:
        print(f"[raev2_stage2] cfg={args.config}")
        print(
            f"[raev2_stage2] out={out_dir} n={num_samples} bsz/rank={per_proc_batch} "
            f"genbs={global_gen_batch} world={world_size}"
        )
        print(f"[raev2_stage2] steps={int(cfg.sampler.num_steps)} guidance={OmegaConf.to_container(cfg.guidance, resolve=True)}")
        print(f"[raev2_stage2] sampling protocol=fixed_global seed={seed}")
    if dist.is_initialized():
        dist.barrier()

    rae = instantiate_from_config(cfg.stage_1).to(device).eval()
    rae.requires_grad_(False)
    prepare_stage2_model_config(cfg)
    model = instantiate_from_config(cfg.stage_2).to(device).eval()
    model.requires_grad_(False)

    model_fn, sample_kwargs = get_model_forward_fn(model, cfg.guidance)
    use_guidance = is_guidance_active(cfg.guidance)

    latent_size = tuple(int(v) for v in cfg.misc.latent_size)
    shift_dim = cfg.misc.time_dist_shift_dim if "time_dist_shift_dim" in cfg.misc else math.prod(latent_size)
    shift = math.sqrt(float(shift_dim) / float(cfg.misc.time_dist_shift_base))
    transport = Transport(
        prediction=str(cfg.transport.prediction),
        time_dist_type=str(cfg.transport.time_dist_type),
        time_dist_shift=shift,
        t_eps=float(cfg.transport.get("t_eps", 0.05)),
    )
    sampler = Sampler(transport, cfg.guidance)
    sample_fn = sampler.sample_ode(num_steps=int(cfg.sampler.num_steps))

    null_label = int(cfg.misc.get("null_label", cfg.misc.num_classes))
    num_classes = int(cfg.misc.num_classes)
    labels = None
    label_manifest = {
        "protocol": "uniform_random",
        "seed": seed,
        "num_samples": num_samples,
        "num_classes": num_classes,
    }
    if label_source == "metadata":
        metadata_path = inf_cfg.get("label_metadata_path")
        if not metadata_path:
            raise ValueError("inference.label_source=metadata requires label_metadata_path")
        if rank == 0:
            label_values, label_manifest = load_metadata_label_plan(
                metadata_path,
                num_samples,
                label_key=str(inf_cfg.get("label_key", "label")),
                num_classes=num_classes,
            )
            labels = torch.tensor(label_values, device=device, dtype=torch.long)
            print(f"[raev2_stage2] label protocol={label_manifest}")
    elif label_source != "random":
        raise ValueError(f"Unsupported inference.label_source={label_source!r}; use random or metadata")

    def sample_fn_with_context(z, wrapped_model_fn, y, **kwargs):
        kwargs.update(context=y, attn_mask=None)
        return sample_fn(z, wrapped_model_fn, **kwargs)

    # A complete image set is a valid generation checkpoint. Resume directly
    # at metric evaluation, but reject partial or non-canonical output sets so
    # results from different runs cannot be mixed accidentally.
    resume_state = torch.zeros((), dtype=torch.int64, device=device)
    if rank == 0 and not clean_output:
        existing_names = sorted(
            path.name for path in out_dir.glob("*.png") if not path.name.startswith("_")
        )
        if existing_names:
            expected_names = [f"{index:06d}.png" for index in range(num_samples)]
            resume_state.fill_(1 if existing_names == expected_names else -len(existing_names))
    if dist.is_initialized():
        dist.broadcast(resume_state, src=0)

    resume_value = int(resume_state.item())
    if resume_value < 0:
        raise RuntimeError(
            f"Output directory contains an incomplete or non-canonical PNG set: "
            f"{out_dir} ({-resume_value} files); expected exactly {num_samples}."
        )
    if resume_value == 1:
        if rank == 0:
            print(f"[raev2_stage2] reuse complete generation: {out_dir} ({num_samples} files)")
    else:
        save_class_cond_fakes_ddp(
            model_fn,
            sample_fn_with_context,
            rae,
            latent_size,
            num_classes,
            null_label,
            sample_kwargs,
            use_guidance,
            num_samples,
            global_gen_batch,
            device,
            out_dir,
            seed=seed,
            autocast_kwargs=autocast_kwargs,
            clean_output=clean_output,
            labels=labels,
        )

    for key in ("real_set", "eval_dataset", "dataloader", "metrics"):
        if key not in cfg:
            raise ValueError(f"RAEv2 Stage2 evaluation requires config key: {key}")
    payload = {
        "real_set": OmegaConf.to_container(cfg.real_set, resolve=True),
        "dataset": OmegaConf.to_container(cfg.eval_dataset, resolve=True),
        "dataloader": OmegaConf.to_container(cfg.dataloader, resolve=True),
        "metrics": OmegaConf.to_container(cfg.metrics, resolve=True),
    }
    payload["dataset"]["pred_dir"] = str(out_dir)
    metrics = evaluate_config(payload, device, use_tqdm=False, distributed=True)

    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        count = len([p for p in out_dir.glob("*.png") if not p.name.startswith("_")])
        result_dir = out_dir.parent
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "sampling_manifest.json").write_text(
            json.dumps(label_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        record = {f"eval/{str(key).lower()}": float(value) for key, value in metrics.items()}
        record["eval/n"] = float(count)
        (result_dir / "metrics.json").write_text(
            json.dumps(
                {
                    **record,
                    "ckpt": str(cfg.stage_2.ckpt),
                    "config": str(args.config),
                    "num_steps": int(cfg.sampler.num_steps),
                    "seed": seed,
                    "stage_1": {
                        "ckpt": cfg.stage_1.get("ckpt"),
                        "encoder_name": cfg.stage_1.params.get("encoder_name"),
                        "pretrained_decoder_path": cfg.stage_1.params.get("pretrained_decoder_path"),
                        "normalization_stat_path": cfg.stage_1.params.get("normalization_stat_path"),
                    },
                    "label_protocol": label_manifest,
                    "guidance": OmegaConf.to_container(cfg.guidance, resolve=True),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"[raev2_stage2] done files={count} metrics={record}")
        print(f"[raev2_stage2] wrote {result_dir / 'metrics.json'}")
    cleanup_distributed()


if __name__ == "__main__":
    main()
