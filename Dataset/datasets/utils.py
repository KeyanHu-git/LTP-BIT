import os
import inspect
from pathlib import Path
from typing import Any, List

import torch
import yaml
from omegaconf import OmegaConf

from ..registry import DATASETS


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(root: str) -> List[str]:
    paths: List[str] = []
    for base, _, files in os.walk(root, followlinks=True):
        for fname in files:
            if Path(fname).suffix.lower() in _IMG_EXTS:
                paths.append(os.path.join(base, fname))
    paths.sort()
    return paths


_DATASETS_ROOT = Path(__file__).resolve().parents[2] / "configs" / "datasets"


def load_pipelines(project_dir: str):
    cfg_root = _DATASETS_ROOT / project_dir
    with open(cfg_root / "train.yaml", "r", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)["train_pipeline"]
    with open(cfg_root / "test.yaml", "r", encoding="utf-8") as f:
        test_cfg = yaml.safe_load(f)["test_pipeline"]
    return train_cfg, test_cfg


def build_dataset_from_config(dataset_cfg):
    cfg = dict(OmegaConf.to_container(dataset_cfg, resolve=True))
    dataset_type = cfg.pop("type")
    dataset_cls = DATASETS.get(dataset_type)
    if dataset_cls is None:
        raise KeyError(f"Dataset '{dataset_type}' is not registered.")
    sig = inspect.signature(dataset_cls.__init__).parameters
    return dataset_cls(**{k: v for k, v in cfg.items() if k in sig})


def _extract_label(sample: Any) -> int:
    if isinstance(sample, dict) and "y" in sample:
        return int(sample["y"])
    if isinstance(sample, (tuple, list)) and sample:
        return int(sample[-1])
    raise TypeError(f"Cannot extract class label from sample type {type(sample).__name__}")


def get_class_labels(dataset, num_samples: int, device=None) -> torch.Tensor:
    n = int(num_samples)
    if n <= 0:
        raise ValueError("num_samples must be > 0.")
    if hasattr(dataset, "get_labels"):
        labels = dataset.get_labels(n)
    else:
        if len(dataset) < n:
            raise ValueError(f"Requested {n} labels from dataset of length {len(dataset)}.")
        if hasattr(dataset, "get_label"):
            labels = [dataset.get_label(i) for i in range(n)]
        else:
            labels = [_extract_label(dataset[i]) for i in range(n)]
    labels = torch.as_tensor(labels, dtype=torch.long)
    if labels.ndim != 1:
        raise ValueError("class labels must be 1D.")
    if labels.numel() != n:
        raise ValueError(f"class label count {labels.numel()} does not match num_samples {n}.")
    return labels.to(device=device) if device is not None else labels

