import inspect
from typing import Any, Dict

from Dataset.registry import DATASETS

from .registry import METRICS


def _ensure_registered():
    from .registration import register_all
    register_all()


def build_metric(cfg: Dict[str, Any], **default_args) -> Any:
    metric_cls = get_metric_class(cfg)
    args = cfg.copy()
    args.pop("type")
    sig_params = inspect.signature(metric_cls.__init__).parameters
    accepts_var_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig_params.values())
    for k, v in default_args.items():
        if k in sig_params or (accepts_var_kwargs and k != "sync_dist"):
            args[k] = v
    return metric_cls(**args)


def get_metric_class(cfg: Dict[str, Any]) -> type:
    if not isinstance(cfg, dict):
        raise TypeError(f"Config must be dict, got {type(cfg).__name__}")
    if "type" not in cfg:
        raise ValueError("Config must contain 'type' field")
    _ensure_registered()
    metric_cls = METRICS.get(cfg["type"])
    if metric_cls is None:
        available = list(METRICS._modules.keys())
        raise KeyError(f"Metric '{cfg['type']}' not registered. Available: {available}")
    return metric_cls


def build_dataset(cfg: Dict[str, Any], **kwargs) -> Any:
    if not isinstance(cfg, dict):
        raise TypeError(f"Config must be dict, got {type(cfg).__name__}")
    if "type" not in cfg:
        raise ValueError("Config must contain 'type' field")

    _ensure_registered()

    args = cfg.copy()
    obj_type = args.pop("type")

    dataset_cls = DATASETS.get(obj_type)
    if dataset_cls is None:
        available = list(DATASETS._modules.keys())
        raise KeyError(f"Dataset '{obj_type}' not registered. Available: {available}")

    args.update(kwargs)
    return dataset_cls(**args)
