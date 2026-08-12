from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from PIL import Image

from ..registry import DATASETS
from ..transforms import build_pipeline
from .base_datasets import FixedLabelDataset, PairedImageDataset
from .utils import load_pipelines


@DATASETS.register_module()
class Dataset_FixedLabelCC(FixedLabelDataset):
    def __init__(self, dataroot: str, pipeline_dir: str, is_train: bool = True, fixed_label: int = 0):
        super().__init__(dataroot, fixed_label=fixed_label)
        train_cfg, test_cfg = load_pipelines(pipeline_dir)
        self.pipeline = build_pipeline(train_cfg if is_train else test_cfg)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        image = Image.open(path).convert("RGB")
        out = self.pipeline({"img": image})
        return {"img": out["img"], "y": self.labels[index], "img_path": path, "idx": index}


@DATASETS.register_module()
class Dataset_MetadataLabelCC:
    def __init__(
        self,
        dataroot: str,
        pipeline_dir: str,
        is_train: bool = True,
        metadata_path: Optional[str] = None,
        split: Optional[str] = None,
        rel_path_key: str = "rel_path",
        label_key: str = "label",
        fixed_label: Optional[int] = None,
        path_prefix: Optional[str] = None,
        filter_split: bool = True,
    ):
        self.dataroot = Path(dataroot)
        if metadata_path is None:
            candidate = self.dataroot / "metadata.jsonl"
            metadata_path = candidate if candidate.exists() else self.dataroot.parent / "metadata.jsonl"
        self.metadata_path = Path(metadata_path)
        self.base_dir = self.metadata_path.parent
        self.path_prefix = Path(path_prefix) if path_prefix else None
        self.image_paths = []
        self.labels = []
        target_split = (split if split is not None else ("train" if is_train else None)) if filter_split else None

        with self.metadata_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if target_split is not None and row.get("split") != target_split:
                    continue
                path = Path(row[rel_path_key])
                if self.path_prefix is not None and not path.is_absolute():
                    path = self.path_prefix / path
                self.image_paths.append(str(path if path.is_absolute() else self.base_dir / path))
                self.labels.append(int(fixed_label if fixed_label is not None else row.get(label_key, 0)))

        train_cfg, test_cfg = load_pipelines(pipeline_dir)
        self.pipeline = build_pipeline(train_cfg if is_train else test_cfg)

    def __len__(self) -> int:
        return len(self.image_paths)

    def get_label(self, index: int) -> int:
        return self.labels[index]

    def get_labels(self, num_samples: int):
        if num_samples > len(self):
            raise ValueError(f"Requested {num_samples} labels from dataset of length {len(self)}.")
        return self.labels[:num_samples]

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        image = Image.open(path).convert("RGB")
        out = self.pipeline({"img": image})
        return {"img": out["img"], "y": self.labels[index], "img_path": path, "idx": index}


@DATASETS.register_module()
class Dataset_RAEPairedI2I(PairedImageDataset):
    def __init__(
        self,
        dataroot: str,
        pipeline_dir: str,
        split: str = "train",
        source_dir: str = "sar",
        target_dir: str = "opt",
        is_train: bool = True,
        fixed_label: int = 0,
    ):
        super().__init__(dataroot, split, source_dir, target_dir)
        train_cfg, test_cfg = load_pipelines(pipeline_dir)
        self.pipeline = build_pipeline(train_cfg if is_train else test_cfg)
        self.fixed_label = int(fixed_label)

    def __getitem__(self, index: int):
        source = Image.open(self.source_paths[index]).convert("RGB")
        target = Image.open(self.target_paths[index]).convert("RGB")
        out = self.pipeline({"img": target, "img_ref": source})
        return out["img"], out["img_ref"], self.fixed_label
