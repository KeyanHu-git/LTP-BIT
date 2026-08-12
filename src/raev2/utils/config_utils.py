from __future__ import annotations

from collections.abc import Mapping

from omegaconf import OmegaConf


def cfg_get(config, key: str, default=None):
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def cfg_to_dict(config) -> dict:
    if config is None:
        return {}
    if OmegaConf.is_config(config):
        return dict(OmegaConf.to_container(config, resolve=True) or {})
    if isinstance(config, Mapping):
        return dict(config)
    if hasattr(config, "items"):
        return dict(config.items())
    raise TypeError(f"Expected mapping-like config, got {type(config)!r}")
