from typing import Any, Dict, Optional, Union

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from ...registry import METRICS
from ._cache import load_or_compute
from .base import BaseDistributionalMetric


def _feature_identity(feature: Union[int, torch.nn.Module]) -> Any:
    if isinstance(feature, int):
        return feature
    identity = getattr(feature, "cache_identity", None)
    if callable(identity):
        identity = identity()
    if identity is not None:
        return identity
    return {
        "module": f"{feature.__class__.__module__}.{feature.__class__.__qualname__}",
        "num_features": getattr(feature, "num_features", None),
    }


@METRICS.register_module()
class FID(BaseDistributionalMetric):
    """Frechet Inception Distance.

    If `real_set` is given, the real-side features are loaded from
    Evaluation/cached_features/<real_set.name>/<cache_filename>; if the cache
    file is missing and real_set.auto_resume is true (default), the metric computes
    and saves it on first construction. After that, only fake batches are consumed.

    `feature` accepts the int presets (64/192/768/2048 → InceptionV3) **or** any
    `nn.Module` exposing a `num_features` attribute (used by FDr6's per-backbone
    sub-metrics). When a Module is passed, `normalize` should be False (custom
    encoders preprocess [0,1] floats themselves) and `cache_filename` should be
    set to a backbone-specific name to avoid clobbering the InceptionV3 cache.
    """

    DEFAULT_CACHE_FILENAME = "inception_fid_state.pt"
    _real_cache_filename = DEFAULT_CACHE_FILENAME

    def __init__(
        self,
        feature: Union[int, torch.nn.Module] = 2048,
        normalize: bool = True,
        device: str = "cuda",
        real_set: Optional[Dict] = None,
        cache_filename: Optional[str] = None,
        sync_dist: bool = True,
        feature_extractor_weights_path: Optional[str] = None,
        antialias: bool = True,
    ):
        super().__init__(device, sync_dist=sync_dist)
        from torchmetrics.image.fid import FrechetInceptionDistance
        self._cache_filename = cache_filename or self.DEFAULT_CACHE_FILENAME
        self._cache_identity = {
            "metric": "FID",
            "cache_filename": self._cache_filename,
            "feature": _feature_identity(feature),
            "normalize": bool(normalize),
            "feature_extractor_weights_path": feature_extractor_weights_path,
        }
        if not antialias:
            self._cache_identity["antialias"] = False
        # When real_set is given, we hold cached features across resets — torchmetrics
        # would otherwise wipe them on every reset() (default reset_real_features=True).
        fid_kwargs = dict(feature=feature, normalize=normalize, antialias=antialias)
        if feature_extractor_weights_path is not None:
            fid_kwargs["feature_extractor_weights_path"] = feature_extractor_weights_path
        if real_set is not None:
            fid_kwargs["reset_real_features"] = False
        try:
            self.metric = FrechetInceptionDistance(sync_on_compute=False, **fid_kwargs).to(self.device)
        except (TypeError, ValueError) as exc:
            if "sync_on_compute" not in str(exc):
                raise
            self.metric = FrechetInceptionDistance(**fid_kwargs).to(self.device)
        if hasattr(self.metric, "sync_on_compute"):
            self.metric.sync_on_compute = False
        self._real_loaded = False
        if real_set is not None:
            state, _ = load_or_compute(
                real_set,
                filename=self._cache_filename,
                compute_fn=self._compute_real_cache,
                device=str(self.device),
                cache_identity=self._cache_identity,
                sync_dist=self.sync_dist,
            )
            self.metric.real_features_sum = state["real_features_sum"].to(self.device).double()
            self.metric.real_features_cov_sum = state["real_features_cov_sum"].to(self.device).double()
            self.metric.real_features_num_samples = state["real_features_num_samples"].to(self.device).long()
            self._real_loaded = True

    def _compute_real_cache(self, loader: DataLoader):
        with torch.no_grad():
            for x in loader:
                self.metric.update(x.to(self.device), real=True)
        # torchmetrics state_dict() omits add_state vars; persist them manually.
        return {
            "real_features_sum": self.metric.real_features_sum.detach().cpu(),
            "real_features_cov_sum": self.metric.real_features_cov_sum.detach().cpu(),
            "real_features_num_samples": self.metric.real_features_num_samples.detach().cpu(),
        }

    @property
    def requires_target(self) -> bool:
        return not self._real_loaded

    def update(self, preds: torch.Tensor, target: torch.Tensor = None) -> None:
        if not self._real_loaded:
            if target is None:
                raise ValueError("FID without `real_set` requires a `target` batch.")
            self.metric.update(target.to(self.device), real=True)
        self.metric.update(preds.to(self.device), real=False)

    def compute(self) -> float:
        if self.sync_dist and dist.is_available() and dist.is_initialized():
            state_names = ["fake_features_sum", "fake_features_cov_sum", "fake_features_num_samples"]
            if not self._real_loaded:
                state_names.extend([
                    "real_features_sum",
                    "real_features_cov_sum",
                    "real_features_num_samples",
                ])
            for name in state_names:
                state = getattr(self.metric, name, None)
                if state is not None:
                    dist.all_reduce(state, op=dist.ReduceOp.SUM)
        return self.metric.compute().item()

    def reset(self) -> None:
        self.metric.reset()


@METRICS.register_module()
class PriorFID(FID):
    """FID against a cached Stage2 prior sample distribution."""

    _reference_set_key = "prior_set"

    def __init__(
        self,
        prior_set: Dict,
        sync_dist: bool = True,
        **kwargs,
    ):
        kwargs.pop("real_set", None)
        super().__init__(real_set=prior_set, sync_dist=sync_dist, **kwargs)

    @property
    def name(self) -> str:
        return "PriorFID"
