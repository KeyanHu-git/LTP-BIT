import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist

from .rng import setup_repro


def init_runtime(
    args,
    seed: Optional[int] = None,
    device_arg: str = "device",
    seed_attr: str = "seed",
    scale_seed_by_world: bool = False,
    require_distributed: bool = False,
) -> Tuple[bool, int, int, torch.device]:
    distributed = ("RANK" in os.environ and "WORLD_SIZE" in os.environ) if require_distributed else int(os.environ.get("WORLD_SIZE", "1")) > 1
    if require_distributed and not distributed:
        raise RuntimeError("This entry requires torchrun/distributed environment variables.")

    if distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        world_size = 1
        requested = getattr(args, device_arg, None)
        device = torch.device(requested) if requested is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_seed = getattr(args, seed_attr, 0) if seed is None else seed
    repro_seed = int(base_seed) * world_size + rank if scale_seed_by_world else int(base_seed) + rank
    setup_repro(args, rank, repro_seed)
    return distributed, rank, world_size, device
