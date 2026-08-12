from ...registry import METRICS
from ..base import TorchMetricsWrapper


@METRICS.register_module()
class SSIM(TorchMetricsWrapper):
    def _build_metric(self, data_range: float = 1.0, **kwargs):
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        return StructuralSimilarityIndexMeasure(data_range=data_range)
