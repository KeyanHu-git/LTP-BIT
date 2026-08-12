from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.distributed as dist


_STATE_KEYS = (
    "real_features_sum",
    "real_features_cov_sum",
    "real_features_num_samples",
)


def _fake_state(metric: Any, *, sync_dist: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fake_sum = metric.fake_features_sum.double().clone()
    fake_cov_sum = metric.fake_features_cov_sum.double().clone()
    fake_num = metric.fake_features_num_samples.long().clone()
    if sync_dist and dist.is_available() and dist.is_initialized():
        for state in (fake_sum, fake_cov_sum, fake_num):
            dist.all_reduce(state, op=dist.ReduceOp.SUM)
    return fake_sum, fake_cov_sum, fake_num


def frechet_from_feature_stats(
    real_sum: torch.Tensor,
    real_cov_sum: torch.Tensor,
    real_num: torch.Tensor,
    fake_sum: torch.Tensor,
    fake_cov_sum: torch.Tensor,
    fake_num: torch.Tensor,
) -> float:
    import numpy as np
    from scipy import linalg

    real_sum = real_sum.detach().double().cpu()
    real_cov_sum = real_cov_sum.detach().double().cpu()
    real_num = real_num.detach().long().cpu()
    fake_sum = fake_sum.detach().double().cpu()
    fake_cov_sum = fake_cov_sum.detach().double().cpu()
    fake_num = fake_num.detach().long().cpu()

    real_n = int(real_num.item())
    fake_n = int(fake_num.item())
    if real_n < 2 or fake_n < 2:
        raise RuntimeError("Frechet distance requires at least two real and fake samples.")
    if real_sum.ndim != 1 or fake_sum.ndim != 1 or real_sum.shape != fake_sum.shape:
        raise ValueError(
            f"Frechet feature sums must be matching 1D tensors, got {tuple(real_sum.shape)} "
            f"and {tuple(fake_sum.shape)}."
        )
    dim = real_sum.numel()
    if real_cov_sum.shape != (dim, dim) or fake_cov_sum.shape != (dim, dim):
        raise ValueError(
            f"Frechet cov_sum tensors must be ({dim}, {dim}), got "
            f"{tuple(real_cov_sum.shape)} and {tuple(fake_cov_sum.shape)}."
        )
    for label, tensor in (
        ("real_sum", real_sum),
        ("real_cov_sum", real_cov_sum),
        ("fake_sum", fake_sum),
        ("fake_cov_sum", fake_cov_sum),
    ):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Frechet input '{label}' contains non-finite values.")

    mu_r = real_sum / real_n
    mu_f = fake_sum / fake_n
    cov_r = (real_cov_sum - real_n * torch.outer(mu_r, mu_r)) / (real_n - 1)
    cov_f = (fake_cov_sum - fake_n * torch.outer(mu_f, mu_f)) / (fake_n - 1)

    mu_r_np = mu_r.numpy()
    mu_f_np = mu_f.numpy()
    cov_r_np = cov_r.numpy()
    cov_f_np = cov_f.numpy()

    diff = mu_r_np - mu_f_np
    covmean, _ = linalg.sqrtm(cov_r_np.dot(cov_f_np), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(cov_r_np.shape[0]) * 1e-6
        covmean = linalg.sqrtm((cov_r_np + offset).dot(cov_f_np + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"Frechet covmean imaginary component {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real

    return float(diff.dot(diff) + np.trace(cov_r_np) + np.trace(cov_f_np) - 2 * np.trace(covmean))


def frechet_from_feature_state(
    real_state: Mapping[str, torch.Tensor],
    fake_state: Mapping[str, torch.Tensor],
) -> float:
    """Frechet distance from two cached sufficient-stat dicts.

    Both states use the Evaluation cache schema:
    ``{real_features_sum, real_features_cov_sum, real_features_num_samples}``.
    The second argument is named ``fake_state`` for algebraic symmetry with
    FID, but it can be another real/validation set when computing FDr
    normalizers.
    """

    def unpack(state: Mapping[str, torch.Tensor], label: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        missing = [key for key in _STATE_KEYS if key not in state]
        if missing:
            raise KeyError(f"Frechet cache state '{label}' missing keys: {missing}")
        return (
            state["real_features_sum"],
            state["real_features_cov_sum"],
            state["real_features_num_samples"],
        )

    real_sum, real_cov_sum, real_num = unpack(real_state, "real")
    fake_sum, fake_cov_sum, fake_num = unpack(fake_state, "fake")
    return frechet_from_feature_stats(real_sum, real_cov_sum, real_num, fake_sum, fake_cov_sum, fake_num)


def frechet_from_fid_state(metric: Any, *, sync_dist: bool = True) -> float:
    fake_sum, fake_cov_sum, fake_num = _fake_state(metric, sync_dist=sync_dist)
    return frechet_from_feature_stats(
        metric.real_features_sum,
        metric.real_features_cov_sum,
        metric.real_features_num_samples,
        fake_sum,
        fake_cov_sum,
        fake_num,
    )
