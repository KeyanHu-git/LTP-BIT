from __future__ import annotations

import math
from typing import Optional, Sequence, Union

import torch
from torch.utils.data import Sampler


class DistributedWeightedSampler(Sampler[int]):
    """DDP-aware weighted sampler.

    torch ships WeightedRandomSampler (single-process) and DistributedSampler (uniform DDP),
    but no built-in for the intersection. This draws total_size global indices via
    multinomial(weights) once per epoch (deterministic seed = base_seed + epoch),
    then strides them per rank so every rank sees a disjoint shard.

    Single-rank degenerates to vanilla WeightedRandomSampler.
    Uniform weights degenerates to DistributedSampler-with-replacement.
    """

    def __init__(
        self,
        weights: Union[Sequence[float], torch.Tensor],
        num_samples_per_epoch: Optional[int] = None,
        replacement: bool = True,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
    ):
        if num_replicas is None or rank is None:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                if num_replicas is None:
                    num_replicas = dist.get_world_size()
                if rank is None:
                    rank = dist.get_rank()
            else:
                num_replicas = 1 if num_replicas is None else num_replicas
                rank = 0 if rank is None else rank

        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank {rank} out of range [0, {num_replicas})")

        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.replacement = bool(replacement)
        self.seed = int(seed)
        self.epoch = 0

        n_total = int(num_samples_per_epoch) if num_samples_per_epoch is not None else len(self.weights)
        self.total_size = int(math.ceil(n_total / self.num_replicas) * self.num_replicas)
        self.num_samples = self.total_size // self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        idx = torch.multinomial(self.weights, self.total_size, self.replacement, generator=g).tolist()
        idx = idx[self.rank:self.total_size:self.num_replicas]
        return iter(idx)

    def __len__(self) -> int:
        return self.num_samples
