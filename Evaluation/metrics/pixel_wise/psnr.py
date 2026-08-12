import torch

from ...registry import METRICS
from ..base import NoReferenceMetric, TorchMetricsWrapper


@METRICS.register_module()
class GlobalPSNR(TorchMetricsWrapper):
    """全池化 MSE 的 PSNR(2026-07-09 前的 eval/psnr 即此口径, 历史可比)。"""

    @property
    def name(self) -> str:
        return "Global_PSNR"

    def _build_metric(self, data_range: float = 1.0, **kwargs):
        from torchmetrics.image import PeakSignalNoiseRatio
        return PeakSignalNoiseRatio(data_range=data_range)


@METRICS.register_module()
class PSNR(NoReferenceMetric):
    """逐图 PSNR 的样本均值(文献口径)。复用 NoReferenceMetric 的逐样本跨卡聚合。"""

    def __init__(self, device: str = "cuda", data_range: float = 1.0, **kwargs):
        super().__init__(device, **kwargs)
        self.data_range = float(data_range)

    @property
    def requires_target(self) -> bool:
        return True

    def update(self, preds: torch.Tensor, target: torch.Tensor = None) -> None:
        p = preds.to(self.device).float()
        t = target.to(self.device).float()
        mse = (p - t).pow(2).flatten(1).mean(dim=1)
        psnr = 10.0 * torch.log10(self.data_range ** 2 / mse.clamp_min(1e-12))
        self.scores.extend(psnr.cpu().tolist())
