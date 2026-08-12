from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from ..registry import DATASETS
from ..transforms import build_pipeline


_DEFAULT_PREPROCESS: List[Dict] = [{"type": "ToTensor", "keys": ["pred", "gt"]}]


def _build(preprocess: Optional[List[Dict]]):
    return build_pipeline(preprocess if preprocess else _DEFAULT_PREPROCESS)


@DATASETS.register_module()
class EvalPairedDataset(Dataset):
    """Inference-time paired dataset: returns (pred_tensor, gt_tensor).

    Pipeline operates on a dict {pred, gt} via TRANSFORMS registry — same machinery
    used by training datasets. Default preprocess is ToTensor only.
    """

    VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    def __init__(
        self,
        pred_dir: str,
        gt_dir: str,
        preprocess: Optional[List[Dict]] = None,
    ):
        super().__init__()
        self.pred_dir = Path(pred_dir)
        self.gt_dir = Path(gt_dir)
        self.pipeline = _build(preprocess)

        self.pred_files = sorted([
            f.name for f in self.pred_dir.iterdir()
            if f.suffix.lower() in self.VALID_EXTS
        ])
        self.gt_files = sorted([
            f.name for f in self.gt_dir.iterdir()
            if f.suffix.lower() in self.VALID_EXTS
        ])
        if not self.pred_files:
            raise RuntimeError(f"No images found in {self.pred_dir}")
        gt_set = set(self.gt_files)
        missing = [f for f in self.pred_files if f not in gt_set]
        if missing:
            preview = ", ".join(missing[:5])
            raise RuntimeError(f"Missing {len(missing)} GT files for predictions: {preview}")

    def __len__(self) -> int:
        return len(self.pred_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fname = self.pred_files[idx]
        pred_img = Image.open(self.pred_dir / fname).convert("RGB")
        gt_img = Image.open(self.gt_dir / fname).convert("RGB")
        results = self.pipeline({"pred": pred_img, "gt": gt_img})
        return results["pred"], results["gt"]

    def iter_manifest_rows(self):
        for f in self.pred_files:
            yield {"id": f, "pred_rel": f, "gt_rel": f}

    def manifest_roots(self) -> Dict[str, str]:
        return {"pred": str(self.pred_dir), "gt": str(self.gt_dir)}


@DATASETS.register_module()
class EvalSingleDataset(Dataset):
    """Single-folder eval dataset (no GT). Used for Stage 1 unconditional generation eval
    (FID/CMMD vs cached real_set), and for the real-side cache builder itself.
    """

    VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    def __init__(
        self,
        pred_dir: str,
        preprocess: Optional[List[Dict]] = None,
    ):
        super().__init__()
        self.pred_dir = Path(pred_dir)
        self.pipeline = _build(preprocess)
        self.pred_files = sorted([
            f.name for f in self.pred_dir.iterdir()
            if f.suffix.lower() in self.VALID_EXTS and not f.name.startswith("_")
        ])
        if not self.pred_files:
            raise RuntimeError(f"No images found in {self.pred_dir}")

    def __len__(self) -> int:
        return len(self.pred_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        fname = self.pred_files[idx]
        results = self.pipeline({"pred": Image.open(self.pred_dir / fname).convert("RGB")})
        return results["pred"]

    def iter_manifest_rows(self):
        for f in self.pred_files:
            yield {"id": f, "pred_rel": f, "gt_rel": ""}

    def manifest_roots(self) -> Dict[str, str]:
        return {"pred": str(self.pred_dir)}
