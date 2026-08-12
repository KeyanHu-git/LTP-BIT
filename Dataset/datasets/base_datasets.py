import os
from pathlib import Path
from typing import List, Optional, Tuple

from torch.utils.data import Dataset

from .utils import list_images


class _BaseImageDataset(Dataset):
    def __init__(
        self,
        dataroot: str,
        split: str,
        source_dir: Optional[str] = None,
        target_dir: Optional[str] = None,
    ):
        self.dataroot = dataroot
        self.split = split
        if not source_dir or not target_dir:
            raise ValueError("source_dir and target_dir are required for paired datasets.")
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.source_path = Path(dataroot) / split / self.source_dir
        self.target_path = Path(dataroot) / split / self.target_dir


class PairedImageDataset(_BaseImageDataset):
    def __init__(
        self,
        dataroot: str,
        split: str,
        source_dir: Optional[str] = None,
        target_dir: Optional[str] = None,
    ):
        super().__init__(dataroot, split, source_dir, target_dir)
        self.source_paths, self.target_paths = self._build_pairs()

    def _build_pairs(self) -> Tuple[List[str], List[str]]:
        source_map = {
            os.path.relpath(path, self.source_path): path for path in list_images(str(self.source_path))
        }
        target_map = {
            os.path.relpath(path, self.target_path): path for path in list_images(str(self.target_path))
        }
        source_keys, target_keys = set(source_map), set(target_map)
        common_keys = sorted(source_keys & target_keys)
        only_source, only_target = source_keys - target_keys, target_keys - source_keys
        if only_source or only_target:
            examples = sorted(only_source)[:3] + sorted(only_target)[:3]
            raise ValueError(
                "PairedImageDataset requires exact relative-path matches under "
                f"{self.source_path} and {self.target_path}: {len(common_keys)} matched, "
                f"{len(only_source)} source-only, {len(only_target)} target-only. Examples: {examples}"
            )
        return [source_map[key] for key in common_keys], [target_map[key] for key in common_keys]

    def __len__(self) -> int:
        return len(self.source_paths)

    def __getitem__(self, index: int):
        raise NotImplementedError


class ClassCondImageDataset(Dataset):
    def __init__(self, dataroot: str):
        self.dataroot = dataroot
        self.image_paths: List[str] = []
        self.labels: List[int] = []

    def __len__(self) -> int:
        return len(self.image_paths)

    def get_label(self, index: int) -> int:
        return int(self.labels[index])

    def get_labels(self, num_samples: int) -> List[int]:
        count = int(num_samples)
        if count > len(self):
            raise ValueError(f"Requested {count} labels from dataset of length {len(self)}.")
        return [self.get_label(index) for index in range(count)]

    def __getitem__(self, index: int):
        raise NotImplementedError


class FixedLabelDataset(ClassCondImageDataset):
    """Assign one fixed label to every image in a folder."""

    def __init__(self, dataroot: str, fixed_label: int = 0):
        super().__init__(dataroot)
        self.image_paths = sorted(list_images(dataroot))
        self.labels = [int(fixed_label)] * len(self.image_paths)
