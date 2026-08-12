from ...registry import METRICS
from ..base import TorchMetricsWrapper


@METRICS.register_module()
class LPIPS(TorchMetricsWrapper):
    def _build_metric(self, net_type: str = "vgg", normalize: bool = True, **kwargs):
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        return LearnedPerceptualImagePatchSimilarity(net_type=net_type, normalize=normalize)
