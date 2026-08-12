import dataclasses
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Optional, Tuple

import torch.distributed as dist
from omegaconf import OmegaConf

from utils.logging import create_logger

from .wandb_utils import initialize


def experiment_name_from_config(config_path: str) -> str:
    parts = Path(config_path).with_suffix("").parts
    if "train" in parts:
        idx = parts.index("train")
        return str(Path(*parts[idx + 1:]))
    return Path(config_path).stem


def configure_experiment_dirs(args, rank, full_cfg=None, fallback_name: Optional[str] = None) -> Tuple[str, str, logging.Logger]:
    experiment_name = os.environ.get("EXPERIMENT_NAME") or fallback_name
    if not experiment_name:
        raise ValueError("Set EXPERIMENT_NAME or pass fallback_name to configure_experiment_dirs.")

    experiment_dir = os.path.join(args.results_dir, experiment_name)
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir, "raev2")
        logger.info(f"Experiment directory: {experiment_dir}")
        if getattr(args, "wandb", False):
            initialize(args, os.environ["WANDB_ENTITY"], experiment_name, os.environ["WANDB_PROJECT"])
    else:
        logger = create_logger(None, "raev2")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return experiment_dir, checkpoint_dir, logger


def find_resume_checkpoint(resume_dir: str, candidate_ckpt: Optional[str] = None) -> Optional[str]:
    if candidate_ckpt:
        if not os.path.isfile(candidate_ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {candidate_ckpt}")
        return candidate_ckpt

    checkpoint_dir = os.path.join(resume_dir, "checkpoints")
    if not os.path.isdir(checkpoint_dir):
        return None

    last = os.path.join(checkpoint_dir, "ep-last.pt")
    if os.path.isfile(last):
        return last

    candidates = []
    for filename in os.listdir(checkpoint_dir):
        match = re.match(r"^(?:ep|step)-(\d+)\.(?:pt|ckpt)$", filename)
        if match:
            candidates.append((int(match.group(1)), os.path.join(checkpoint_dir, filename)))
    return max(candidates, default=(None, None))[1]


def save_worktree(path: str, config, extra_metadata: dict = None) -> None:
    try:
        config_dict = dataclasses.asdict(config)
    except TypeError:
        config_dict = OmegaConf.to_container(config, resolve=True)
    if extra_metadata:
        config_dict.update(extra_metadata)

    OmegaConf.save(OmegaConf.create(config_dict), os.path.join(path, "config.yaml"))
    worktree_path = os.path.join(os.getcwd(), "src")
    shutil.copytree(worktree_path, os.path.join(path, "src/"), dirs_exist_ok=True)
    print(f"Worktree {worktree_path} saved to {os.path.join(path, 'src/')}")
