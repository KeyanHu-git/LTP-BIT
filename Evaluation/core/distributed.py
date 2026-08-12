import torch.distributed as dist
from torch.utils.data import Subset


def shard_dataset_for_eval(dataset):
    if not dist.is_available() or not dist.is_initialized():
        return dataset
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dataset_len = len(dataset)
    if dataset_len < world_size:
        raise ValueError(
            "Distributed eval requires len(dataset) >= world_size so every rank "
            "receives at least one sample; got "
            f"len(dataset)={dataset_len}, world_size={world_size}."
        )
    return Subset(dataset, range(rank, dataset_len, world_size))
