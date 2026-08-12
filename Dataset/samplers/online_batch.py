from __future__ import annotations

import torch
from torch.utils.data import Sampler


class OnlineDistributedBatchSampler(Sampler[list[int]]):
    """DDP-aware online sampler for metadata-backed datasets.

    The sampler keeps the full metadata pool available, but samples indices on
    demand for each micro-batch. It avoids treating ``len(dataset)`` as a hard
    epoch sweep, which is important for large NFS-hosted image pools.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        micro_batches_per_epoch: int,
        max_train_steps: int,
        grad_accum_steps: int,
        num_replicas: int,
        rank: int,
        seed: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if micro_batches_per_epoch <= 0:
            raise ValueError("micro_batches_per_epoch must be positive.")
        if max_train_steps <= 0:
            raise ValueError("max_train_steps must be positive.")
        if grad_accum_steps <= 0:
            raise ValueError("grad_accum_steps must be positive.")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank {rank} out of range [0, {num_replicas})")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.micro_batches_per_epoch = int(micro_batches_per_epoch)
        self.max_micro_batches = int(max_train_steps) * int(grad_accum_steps)
        self.grad_accum_steps = int(grad_accum_steps)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples_total = len(dataset)
        if self.num_samples_total <= 0:
            raise ValueError("dataset must not be empty.")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        start_micro = self.epoch * self.micro_batches_per_epoch
        end_micro = min(start_micro + self.micro_batches_per_epoch, self.max_micro_batches)
        total_micro_batch = self.batch_size * self.num_replicas
        rank_start = self.rank * self.batch_size
        rank_end = rank_start + self.batch_size

        for micro_idx in range(start_micro, end_micro):
            generator = torch.Generator()
            generator.manual_seed(self.seed + micro_idx)

            global_indices = torch.randint(self.num_samples_total, (total_micro_batch,), generator=generator)
            yield global_indices[rank_start:rank_end].tolist()

    def __len__(self) -> int:
        start_micro = self.epoch * self.micro_batches_per_epoch
        end_micro = min(start_micro + self.micro_batches_per_epoch, self.max_micro_batches)
        return max(0, end_micro - start_micro)
