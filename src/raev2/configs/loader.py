from __future__ import annotations

from os import PathLike

from omegaconf import DictConfig, OmegaConf

from utils import ConfigLoader


def load_config(path: str | PathLike, overrides=None) -> DictConfig:
    """Load one fully resolved RAEv2 config."""
    resolved = ConfigLoader.load_recursive(str(path), overrides=overrides)
    return OmegaConf.create(resolved)
