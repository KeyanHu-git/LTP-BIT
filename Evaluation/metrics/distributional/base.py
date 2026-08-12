from ..base import BaseMetric


class BaseDistributionalMetric(BaseMetric):
    def __init__(self, device: str = "cuda", sync_dist: bool = True, **kwargs):
        super().__init__(device, **kwargs)
        self.sync_dist = bool(sync_dist)

    def set_sync_dist(self, enabled: bool):
        self.sync_dist = bool(enabled)
        return self
