import logging
from typing import Callable, Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..builder import build_metric


class Evaluator:
    def __init__(
        self,
        metrics_cfg: List[Dict],
        device: str = "cuda",
        logger: Optional[logging.Logger] = None,
        use_tqdm: bool = True,
        real_set: Optional[Dict] = None,
        sync_dist: bool = True,
    ):
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        self.use_tqdm = use_tqdm
        self.metrics = []
        self.num_samples = 0

        self.logger.info("Initializing Evaluator on %s", device)
        build_kwargs = {"device": device, "sync_dist": sync_dist}
        if real_set is not None:
            build_kwargs["real_set"] = real_set
        for cfg in metrics_cfg:
            metric = build_metric(cfg, **build_kwargs)
            metric.set_sync_dist(sync_dist)
            self.metrics.append(metric)
            self.logger.info("Registered Metric: %s", metric.name)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, progress_callback: Optional[Callable[[int, int], None]] = None) -> Dict[str, float]:
        self.logger.info("Starting evaluation loop")

        for m in self.metrics:
            m.reset()

        num_samples = 0
        skipped: set = set()
        total_batches = len(dataloader)
        iterator = tqdm(dataloader, desc="Evaluating") if self.use_tqdm else dataloader
        for batch_idx, batch in enumerate(iterator, start=1):
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                preds, targets = batch
            else:
                preds, targets = batch, None
            preds = preds.to(self.device)
            if targets is not None:
                targets = targets.to(self.device)
            num_samples += int(preds.shape[0])

            for metric in self.metrics:
                if metric.requires_target:
                    if targets is None:
                        if metric.name not in skipped:
                            self.logger.warning(
                                "Skipping metric %s: requires target but dataset yields no GT. "
                                "Either remove this metric or use a paired dataset.",
                                metric.name,
                            )
                            skipped.add(metric.name)
                        continue
                    metric.update(preds, targets)
                else:
                    metric.update(preds)
            if progress_callback is not None:
                progress_callback(batch_idx, total_batches)

        results = {}
        for metric in self.metrics:
            if metric.name in skipped:
                continue
            results[metric.name] = metric.compute()
            for key, value in getattr(metric, "last_breakdown", {}).items():
                results[f"{metric.name}/{key}"] = float(value)

        self.num_samples = num_samples
        return results
