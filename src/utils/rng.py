"""Project-level RNG / reproducibility helpers.

Used by train + sample + eval entries across rae and jit packages.
Single source of truth for "how this codebase manages randomness".
"""
from __future__ import annotations

import os
import random as _random

import numpy as np
import torch


def seed_all(seed: int) -> None:
    """Seed every RNG used in this codebase: torch (cpu+cuda), numpy, python random."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    _random.seed(seed)


def capture_rng_state(*, include_cuda: bool = True) -> dict:
    return {
        "python": _random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if include_cuda and torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict) -> None:
    _random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None:
        torch.cuda.set_rng_state(state["cuda"])


def set_deterministic(seed: int) -> None:
    """Strict reproducibility: bit-identical across runs (~10-30%% slower)."""
    seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: re-seed numpy/python inside each worker process."""
    base = torch.initial_seed() % (2 ** 32)
    np.random.seed((base + worker_id) % (2 ** 32))
    _random.seed((base + worker_id) % (2 ** 32))


def setup_repro(args, rank: int, seed: int) -> None:
    """Single entry-point: dispatch on args.deterministic, force fp32 + skip compile if strict.
    Mutates args.precision and args.compile when args.deterministic is True."""
    if getattr(args, "deterministic", False):
        set_deterministic(seed)
        args.precision = "fp32"
        args.compile = False
        if rank == 0:
            print("[repro] strict deterministic: fp32 + cudnn deterministic + no TF32 + no compile")
    else:
        seed_all(seed)
