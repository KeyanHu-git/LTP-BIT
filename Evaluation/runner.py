import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from .builder import build_dataset
from .core import Evaluator
from .core.distributed import shard_dataset_for_eval
from .registration import register_all


@dataclass
class EvaluationRun:
    metrics: Dict[str, float]
    num_samples: int
    duration_s: float
    dataset: Any


def _all_reduce_int(value: int, device: str) -> int:
    if not dist.is_initialized():
        return value
    t = torch.tensor(value, device=device, dtype=torch.int64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t.item())


def resolve_eval_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = dict(cfg)
    if "dataset" not in cfg and "eval_dataset" in cfg:
        cfg["dataset"] = cfg["eval_dataset"]
    dataloader_cfg = dict(cfg.get("dataloader", {}))
    dataloader_cfg.setdefault("batch_size", 32)
    dataloader_cfg.setdefault("num_workers", 4)
    dataloader_cfg.setdefault("shuffle", False)
    cfg["dataloader"] = dataloader_cfg
    return cfg


def run_evaluation_config(
    cfg: Mapping[str, Any],
    device: str | torch.device,
    *,
    use_tqdm: bool = False,
    distributed: Optional[bool] = None,
    logger=None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> EvaluationRun:
    register_all()
    cfg = resolve_eval_config(cfg)
    device = str(device)
    distributed = dist.is_initialized() if distributed is None else distributed
    start = time.time()

    base_dataset = build_dataset(cfg["dataset"])
    dataset = base_dataset
    dataloader_cfg = dict(cfg["dataloader"])

    if distributed:
        if bool(dataloader_cfg.get("shuffle", False)):
            raise ValueError("Distributed eval requires dataloader.shuffle=false.")
        dataset = shard_dataset_for_eval(dataset)
        dataloader_cfg["shuffle"] = False
    loader = DataLoader(dataset, **dataloader_cfg)

    evaluator = Evaluator(
        metrics_cfg=cfg["metrics"],
        device=device,
        logger=logger,
        use_tqdm=use_tqdm,
        real_set=cfg.get("real_set"),
        sync_dist=distributed,
    )
    results = evaluator.evaluate(loader, progress_callback=progress_callback)
    local_samples = evaluator.num_samples
    duration_s = time.time() - start
    total_samples = _all_reduce_int(local_samples, device) if distributed else local_samples

    del evaluator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return EvaluationRun(
        metrics={str(k): float(v) for k, v in results.items()},
        num_samples=total_samples,
        duration_s=duration_s,
        dataset=base_dataset,
    )


def evaluate_config(
    cfg: Mapping[str, Any],
    device: str | torch.device,
    *,
    use_tqdm: bool = False,
    distributed: Optional[bool] = None,
) -> Dict[str, float]:
    return run_evaluation_config(
        cfg,
        device,
        use_tqdm=use_tqdm,
        distributed=distributed,
    ).metrics
