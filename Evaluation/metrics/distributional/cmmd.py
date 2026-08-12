from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ...registry import METRICS
from ._cache import load_or_compute
from .base import BaseDistributionalMetric


@METRICS.register_module()
class CMMD(BaseDistributionalMetric):
    """CMMD with CLIP-ViT-L/14-336 + Gaussian RBF kernel.

    If `real_set` is given, the real-side L2-normalized CLIP embeddings are loaded
    from Evaluation/cached_features/<real_set.name>/clip_embeds.pt; if the cache file
    is missing and real_set.auto_resume is true (default), the metric computes and
    saves it on first construction. After that, only fake batches are consumed.
    """

    _real_cache_filename = "clip_embeds.pt"

    def __init__(
        self,
        model_path: str = "weights/evaluation/clip-vit-large-patch14-336",
        sigma: float = 10.0,
        scale: float = 1000.0,
        kernel_block_size: int = 4096,
        device: str = "cuda",
        real_set: Optional[Dict] = None,
        sync_dist: bool = True,
    ):
        super().__init__(device, sync_dist=sync_dist)
        from transformers import CLIPModel, CLIPImageProcessor
        self.model = CLIPModel.from_pretrained(model_path, local_files_only=True).to(self.device).eval()
        self.processor = CLIPImageProcessor.from_pretrained(model_path, local_files_only=True)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.sigma = sigma
        self.scale = scale
        self.kernel_block_size = int(kernel_block_size)
        if self.kernel_block_size <= 0:
            raise ValueError("kernel_block_size must be positive.")
        self._real_embeds = []
        self._fake_embeds = []
        self._real_loaded = False
        if real_set is not None:
            embeds, _ = load_or_compute(
                real_set,
                filename=self._real_cache_filename,
                compute_fn=self._compute_real_cache,
                device="cpu",
                cache_identity={
                    "metric": "CMMD",
                    "model_path": str(Path(model_path).expanduser().resolve(strict=False)),
                },
                sync_dist=self.sync_dist,
            )
            self._real_embeds = [embeds]
            self._real_loaded = True

    def _compute_real_cache(self, loader: DataLoader) -> torch.Tensor:
        chunks = []
        for x in loader:
            chunks.append(self._embed(x))
        return torch.cat(chunks, dim=0)

    @property
    def requires_target(self) -> bool:
        return not self._real_loaded

    @torch.no_grad()
    def _embed(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device, dtype=torch.float32).clamp(0.0, 1.0)
        pixel_values = self.processor(images=list(images), return_tensors="pt", do_rescale=False)["pixel_values"]
        feats = self.model.get_image_features(pixel_values=pixel_values.to(self.device))
        return F.normalize(feats, dim=-1).cpu()

    def update(self, preds: torch.Tensor, target: torch.Tensor = None) -> None:
        self._fake_embeds.append(self._embed(preds))
        if not self._real_loaded and target is not None:
            self._real_embeds.append(self._embed(target))

    def _gather_embeds(self, embeds: torch.Tensor) -> torch.Tensor:
        if not (self.sync_dist and dist.is_available() and dist.is_initialized()):
            return embeds
        embeds = embeds.to(self.device)
        local_n = torch.tensor([embeds.shape[0]], device=self.device, dtype=torch.long)
        sizes = [torch.zeros_like(local_n) for _ in range(dist.get_world_size())]
        dist.all_gather(sizes, local_n)
        max_n = int(torch.stack(sizes).max().item())
        if embeds.shape[0] < max_n:
            pad = torch.zeros((max_n - embeds.shape[0], embeds.shape[1]), device=self.device, dtype=embeds.dtype)
            embeds = torch.cat([embeds, pad], dim=0)
        gathered = [torch.empty_like(embeds) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, embeds)
        return torch.cat([chunk[: int(size.item())] for chunk, size in zip(gathered, sizes)], dim=0)

    def _kernel_mean(self, x: torch.Tensor, y: torch.Tensor, *, symmetric: bool = False) -> torch.Tensor:
        gamma = 1.0 / (2.0 * self.sigma ** 2)
        block_size = self.kernel_block_size
        total = torch.zeros((), device=x.device, dtype=torch.float64)
        for i in range(0, x.shape[0], block_size):
            x_block = x[i : i + block_size]
            x_sq = (x_block * x_block).sum(dim=-1)
            j_start = i if symmetric else 0
            for j in range(j_start, y.shape[0], block_size):
                y_block = y[j : j + block_size]
                y_sq = (y_block * y_block).sum(dim=-1)
                distances = torch.addmm(
                    x_sq[:, None] + y_sq[None, :],
                    x_block,
                    y_block.T,
                    beta=1.0,
                    alpha=-2.0,
                )
                block_sum = distances.mul_(-gamma).exp_().sum(dtype=torch.float64)
                total += block_sum if not symmetric or i == j else 2.0 * block_sum
        return total / (x.shape[0] * y.shape[0])

    def compute(self) -> float:
        x = torch.cat(self._real_embeds, dim=0)
        y = torch.cat(self._fake_embeds, dim=0)
        if not self._real_loaded:
            x = self._gather_embeds(x)
        else:
            x = x.to(self.device)
        y = self._gather_embeds(y).to(self.device)
        k_xx = self._kernel_mean(x, x, symmetric=True)
        k_yy = self._kernel_mean(y, y, symmetric=True)
        k_xy = self._kernel_mean(x, y)
        return (self.scale * (k_xx + k_yy - 2 * k_xy)).item()

    def reset(self) -> None:
        self._fake_embeds = []
        if not self._real_loaded:
            self._real_embeds = []
