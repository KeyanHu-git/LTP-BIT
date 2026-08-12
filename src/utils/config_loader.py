"""Load YAML through recursive ``_base_`` inheritance and deep merging."""

import os
from collections.abc import Mapping, Sequence


def _omegaconf():
    from omegaconf import OmegaConf

    return OmegaConf


def _resolve_relative_values(value, config_dir):
    if isinstance(value, dict):
        return {key: _resolve_relative_values(item, config_dir) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_relative_values(item, config_dir) for item in value]
    if isinstance(value, str) and (value.startswith("./") or value.startswith("../")):
        return os.path.normpath(os.path.join(config_dir, value))
    return value


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        base_value = out.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            out[key] = _deep_merge(base_value, value)
        else:
            out[key] = value
    return out


class ConfigLoader:
    @staticmethod
    def set_path(target: dict, path: str, value) -> None:
        cursor = target
        parts = path.split(".")
        for key in parts[:-1]:
            next_value = cursor.get(key)
            if not isinstance(next_value, dict):
                next_value = {}
                cursor[key] = next_value
            cursor = next_value
        cursor[parts[-1]] = value

    @staticmethod
    def overrides_to_dict(overrides) -> dict:
        if not overrides:
            return {}
        OmegaConf = _omegaconf()
        if isinstance(overrides, Mapping):
            return OmegaConf.to_container(OmegaConf.create(overrides), resolve=True) or {}
        if isinstance(overrides, str) or not isinstance(overrides, Sequence):
            raise TypeError("Config overrides must be a mapping or a sequence of KEY=VALUE strings.")
        return OmegaConf.to_container(OmegaConf.from_dotlist(list(overrides)), resolve=True) or {}

    @staticmethod
    def apply_overrides(config: dict, overrides) -> dict:
        return _deep_merge(config, ConfigLoader.overrides_to_dict(overrides))

    @staticmethod
    def load_recursive(file_path, overrides=None):
        OmegaConf = _omegaconf()
        file_path = os.path.abspath(file_path)
        config = OmegaConf.to_container(OmegaConf.load(file_path), resolve=True) or {}
        config = _resolve_relative_values(config, os.path.dirname(file_path))
        base_paths = config.pop("_base_", [])
        if isinstance(base_paths, str):
            base_paths = [base_paths]
        merged = {}
        for base_path in base_paths:
            if not os.path.isabs(base_path):
                base_path = os.path.join(os.path.dirname(file_path), base_path)
            merged = _deep_merge(merged, ConfigLoader.load_recursive(base_path))
        return ConfigLoader.apply_overrides(_deep_merge(merged, config), overrides)
