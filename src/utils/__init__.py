"""Project-level utilities.

ConfigLoader requires omegaconf (RAE env) — lazy-imported so that JiT env
(which lacks omegaconf) can `from utils.rng import seed_all` without crashing.
"""
from .rng import seed_all, set_deterministic, seed_worker, setup_repro
from .runtime import init_runtime
from .logging import format_metric_value, format_metrics


def __getattr__(name):
    if name == "ConfigLoader":
        from .config_loader import ConfigLoader
        return ConfigLoader
    raise AttributeError(f"module 'utils' has no attribute {name!r}")


__all__ = [
    "ConfigLoader",
    "seed_all",
    "set_deterministic",
    "seed_worker",
    "setup_repro",
    "init_runtime",
    "format_metric_value",
    "format_metrics",
]
