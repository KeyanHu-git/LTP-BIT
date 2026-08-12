from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from ..registry import DATASETS
from ..transforms import build_pipeline


_PAIRED_PREPROCESS: List[Dict] = [{"type": "ToTensor", "keys": ["pred", "gt"]}]
_SINGLE_PREPROCESS: List[Dict] = [{"type": "ToTensor", "keys": ["pred"]}]


def _build(preprocess: Optional[List[Dict]], default: List[Dict]):
    return build_pipeline(preprocess if preprocess else default)


def _manifest_paths(root: Path, manifest_path: str) -> List[Path]:
    manifest = Path(manifest_path)
    relpaths = [line.strip() for line in manifest.read_text(encoding="utf-8").splitlines()]
    paths = [root / relpath for relpath in relpaths if relpath]
    if not paths:
        raise RuntimeError(f"No image paths found in {manifest}")
    return paths


@DATASETS.register_module()
class EvalSEN12PairedDataset(Dataset):
    def __init__(
        self,
        pred_dir: str,
        gt_dir: str,
        manifest_path: str,
        preprocess: Optional[List[Dict]] = None,
    ):
        super().__init__()
        self.pred_dir = Path(pred_dir)
        self.gt_dir = Path(gt_dir)
        self.pipeline = _build(preprocess, _PAIRED_PREPROCESS)
        self.pred_paths = sorted(path for path in self.pred_dir.iterdir() if path.is_file())
        gt_paths = _manifest_paths(self.gt_dir, manifest_path)
        self.gt_paths = {path.name: path for path in gt_paths}
        if len(self.gt_paths) != len(gt_paths):
            raise RuntimeError(f"Duplicate GT basenames found in {self.gt_dir}")
        missing = [path.name for path in self.pred_paths if path.name not in self.gt_paths]
        if not self.pred_paths:
            raise RuntimeError(f"No images found in {self.pred_dir}")
        if missing:
            raise RuntimeError(
                f"Missing {len(missing)} GT files for predictions: {', '.join(missing[:5])}"
            )

    def __len__(self) -> int:
        return len(self.pred_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pred_path = self.pred_paths[idx]
        pred = Image.open(pred_path).convert("RGB")
        gt = Image.open(self.gt_paths[pred_path.name]).convert("RGB")
        results = self.pipeline({"pred": pred, "gt": gt})
        return results["pred"], results["gt"]

    def iter_manifest_rows(self):
        for pred_path in self.pred_paths:
            gt_path = self.gt_paths[pred_path.name]
            yield {
                "id": pred_path.name,
                "pred_rel": pred_path.name,
                "gt_rel": gt_path.relative_to(self.gt_dir).as_posix(),
            }

    def manifest_roots(self) -> Dict[str, str]:
        return {"pred": str(self.pred_dir), "gt": str(self.gt_dir)}


@DATASETS.register_module()
class EvalSEN12SingleDataset(Dataset):
    def __init__(
        self,
        pred_dir: str,
        manifest_path: str,
        preprocess: Optional[List[Dict]] = None,
    ):
        super().__init__()
        self.pred_dir = Path(pred_dir)
        self.pipeline = _build(preprocess, _SINGLE_PREPROCESS)
        self.pred_paths = _manifest_paths(self.pred_dir, manifest_path)

    def __len__(self) -> int:
        return len(self.pred_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image = Image.open(self.pred_paths[idx]).convert("RGB")
        return self.pipeline({"pred": image})["pred"]

    def iter_manifest_rows(self):
        for path in self.pred_paths:
            relpath = path.relative_to(self.pred_dir).as_posix()
            yield {"id": relpath, "pred_rel": relpath, "gt_rel": ""}

    def manifest_roots(self) -> Dict[str, str]:
        return {"pred": str(self.pred_dir)}
