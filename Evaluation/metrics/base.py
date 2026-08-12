from abc import ABC, abstractmethod
import torch
import torch.distributed as dist


class BaseMetric(ABC):
    def __init__(self, device: str = "cuda", **kwargs):
        self.device = torch.device(device)

    @abstractmethod
    def update(self, preds: torch.Tensor, target: torch.Tensor = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def requires_target(self) -> bool:
        return True

    def set_sync_dist(self, enabled: bool):
        return self


class TorchMetricsWrapper(BaseMetric):
    def __init__(self, device: str = "cuda", **kwargs):
        super().__init__(device, **kwargs)
        self.metric = self._build_metric(**kwargs).to(self.device)
        self.sync_dist = False
        if hasattr(self.metric, "sync_on_compute"):
            self.metric.sync_on_compute = False

    @abstractmethod
    def _build_metric(self, **kwargs):
        raise NotImplementedError

    def set_sync_dist(self, enabled: bool):
        self.sync_dist = bool(enabled)
        return self

    def _gather_state_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.to(self.device)
        if tensor.ndim == 0:
            tensor = tensor.reshape(1)
        local_n = torch.tensor([tensor.shape[0]], device=self.device, dtype=torch.long)
        sizes = [torch.zeros_like(local_n) for _ in range(dist.get_world_size())]
        dist.all_gather(sizes, local_n)
        max_n = int(torch.stack(sizes).max().item())
        if tensor.shape[0] < max_n:
            pad_shape = (max_n - tensor.shape[0],) + tuple(tensor.shape[1:])
            tensor = torch.cat([tensor, torch.zeros(pad_shape, device=self.device, dtype=tensor.dtype)], dim=0)
        gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor)
        return torch.cat([chunk[: int(size.item())] for chunk, size in zip(gathered, sizes)], dim=0)

    def _sync_metric_states(self) -> None:
        if not (self.sync_dist and dist.is_available() and dist.is_initialized()):
            return
        for name, reduction in getattr(self.metric, "_reductions", {}).items():
            state = getattr(self.metric, name)
            if reduction is None:
                chunks = state if isinstance(state, list) else [state]
                if chunks:
                    chunks = [chunk.reshape(1) if chunk.ndim == 0 else chunk for chunk in chunks]
                    setattr(self.metric, name, [self._gather_state_tensor(torch.cat(chunks, dim=0))])
                continue
            if not torch.is_tensor(state):
                continue
            dist.all_reduce(state, op=dist.ReduceOp.SUM)
            if getattr(reduction, "__name__", "") == "dim_zero_mean":
                state.div_(dist.get_world_size())

    def update(self, preds: torch.Tensor, target: torch.Tensor = None) -> None:
        self.metric.update(preds.to(self.device), target.to(self.device))

    def compute(self) -> float:
        self._sync_metric_states()
        return self.metric.compute().item()

    def reset(self) -> None:
        self.metric.reset()


class NoReferenceMetric(BaseMetric):
    def __init__(self, device: str = "cuda", **kwargs):
        super().__init__(device, **kwargs)
        self.scores = []
        self.sync_dist = False

    @property
    def requires_target(self) -> bool:
        return False

    def compute(self) -> float:
        scores = self.scores
        if self.sync_dist and dist.is_available() and dist.is_initialized():
            local = torch.tensor(scores, device=self.device, dtype=torch.float64)
            local_n = torch.tensor([local.numel()], device=self.device, dtype=torch.long)
            sizes = [torch.zeros_like(local_n) for _ in range(dist.get_world_size())]
            dist.all_gather(sizes, local_n)
            max_n = int(torch.stack(sizes).max().item())
            if local.numel() < max_n:
                local = torch.cat([local, torch.zeros(max_n - local.numel(), device=self.device, dtype=local.dtype)])
            gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, local)
            scores = [
                float(v)
                for chunk, size in zip(gathered, sizes)
                for v in chunk[: int(size.item())].cpu().tolist()
            ]
        return sum(scores) / len(scores) if scores else 0.0

    def reset(self) -> None:
        self.scores = []

    def set_sync_dist(self, enabled: bool):
        self.sync_dist = bool(enabled)
        return self
