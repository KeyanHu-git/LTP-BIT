"""Real-side feature cache — convention-based load/compute orchestrator for distributional metrics.

Convention (default): Evaluation/cached_features/<refset_name>/<filename>
Each distributional metric declares its own filename (e.g. FID -> inception_fid_state.pt,
CMMD -> clip_embeds.pt) so multiple metrics can share one refset's cache directory.

The image loader for the real side reuses the registered evaluation dataset and
preprocessing pipeline, keeping real/fake inputs byte-consistent.

Usage (inside a metric __init__):
    state, path = load_or_compute(
        real_set_cfg, filename=self._real_cache_filename,
        compute_fn=self._compute_real_cache, device=self.device,
    )

real_set_cfg fields:
    name        (required)  refset identifier; becomes the cache subdir name
    image_dir   (required)  filesystem path to the real images
    preprocess  (optional)  list of TRANSFORMS op cfgs; defaults to EvalSingleDataset default
    type        (default EvalSingleDataset) registered dataset type for image discovery
    manifest_path (optional) dataset manifest consumed by the registered dataset type
    batch_size  (default 32)
    num_workers (default 4)
    auto_resume (default True)   if cache miss: True → compute+save; False → raise
    cache_root  (default Evaluation/cached_features)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader


_CACHE_META_SCHEMA = "distributional-cache-v1"
_DEFAULT_PREPROCESS = [{"type": "ToTensor", "keys": ["pred", "gt"]}]
_CACHE_POLL_INTERVAL_S = 1.0


def resolve_cache_path(real_set_cfg: Dict, filename: str) -> Path:
    cache_root = Path(real_set_cfg.get("cache_root", "Evaluation/cached_features"))
    name = real_set_cfg["name"]
    return cache_root / name / filename


def load_cache_file(path: str | Path, map_location: str | torch.device = "cpu"):
    try:
        return torch.load(path, weights_only=True, map_location=map_location)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _image_dir_identity(path: str | Path) -> str:
    if path is None:
        return ""
    return str(Path(path).expanduser().resolve(strict=False))


def _preprocess_identity(real_set_cfg: Dict) -> Any:
    return _jsonable(real_set_cfg.get("preprocess") or _DEFAULT_PREPROCESS)


def _cache_identity(filename: str, cache_identity: Any = None) -> Any:
    return _jsonable(cache_identity if cache_identity is not None else {"filename": filename})


def _read_meta(cache_dir: Path) -> Dict:
    meta_path = cache_dir / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return {}
    return meta if isinstance(meta, dict) else {}


def _base_meta(real_set_cfg: Dict) -> Dict[str, Any]:
    return {
        "schema": _CACHE_META_SCHEMA,
        "name": str(real_set_cfg["name"]),
        "image_dir": _image_dir_identity(real_set_cfg["image_dir"]),
        "preprocess": _preprocess_identity(real_set_cfg),
        "type": str(real_set_cfg.get("type", "EvalSingleDataset")),
        "manifest_path": _image_dir_identity(real_set_cfg.get("manifest_path")),
    }


def _base_meta_matches(meta: Dict, expected: Dict[str, Any]) -> bool:
    if meta.get("schema") != expected["schema"]:
        return False
    if str(meta.get("name")) != expected["name"]:
        return False
    if _image_dir_identity(meta.get("image_dir")) != expected["image_dir"]:
        return False
    if _jsonable(meta.get("preprocess")) != expected["preprocess"]:
        return False
    if str(meta.get("type", "EvalSingleDataset")) != expected["type"]:
        return False
    if _image_dir_identity(meta.get("manifest_path")) != expected["manifest_path"]:
        return False
    return True


def _cache_meta_mismatches(
    cache_dir: Path,
    real_set_cfg: Dict,
    filename: str,
    cache_identity: Any = None,
) -> list[str]:
    meta = _read_meta(cache_dir)
    expected = _base_meta(real_set_cfg)
    mismatches = []
    if not _base_meta_matches(meta, expected):
        for key, expected_value in expected.items():
            actual_value = meta.get(key)
            if key == "image_dir" and actual_value is not None:
                actual_value = _image_dir_identity(actual_value)
            else:
                actual_value = _jsonable(actual_value)
            if actual_value != expected_value:
                mismatches.append(key)
    identities = meta.get("cache_identities")
    expected_identity = _cache_identity(filename, cache_identity)
    if not isinstance(identities, dict):
        mismatches.append(f"cache_identities.{filename}")
    elif _jsonable(identities.get(filename)) != expected_identity:
        mismatches.append(f"cache_identities.{filename}")
    return mismatches


def _cache_is_ready(
    cache_path: Path,
    real_set_cfg: Dict,
    filename: str,
    cache_identity: Any = None,
) -> bool:
    return cache_path.exists() and not _cache_meta_mismatches(
        cache_path.parent,
        real_set_cfg,
        filename,
        cache_identity,
    )


def _build_loader(real_set_cfg: Dict) -> DataLoader:
    from Evaluation.builder import build_dataset
    dataset_cfg = {
        "type": real_set_cfg.get("type", "EvalSingleDataset"),
        "pred_dir": real_set_cfg["image_dir"],
        "preprocess": real_set_cfg.get("preprocess"),
    }
    if real_set_cfg.get("manifest_path"):
        dataset_cfg["manifest_path"] = real_set_cfg["manifest_path"]
    ds = build_dataset(dataset_cfg)
    return DataLoader(
        ds,
        batch_size=real_set_cfg.get("batch_size", 32),
        shuffle=False,
        num_workers=real_set_cfg.get("num_workers", 4),
        pin_memory=True,
    )


def _write_meta(
    cache_dir: Path,
    real_set_cfg: Dict,
    n_samples: int,
    filename: str,
    cache_identity: Any = None,
) -> None:
    meta_path = cache_dir / "meta.json"
    base_meta = _base_meta(real_set_cfg)
    meta = _read_meta(cache_dir)
    if not _base_meta_matches(meta, base_meta):
        meta = {}
    meta.update(base_meta)
    meta["n_samples"] = n_samples
    existing_files = meta.get("files", [])
    files = set(existing_files if isinstance(existing_files, list) else [])
    files.add(filename)
    meta["files"] = sorted(files)
    identities = meta.get("cache_identities", {})
    if not isinstance(identities, dict):
        identities = {}
    identities[filename] = _cache_identity(filename, cache_identity)
    meta["cache_identities"] = {key: identities[key] for key in sorted(identities)}
    meta["last_update_utc"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    tmp_path = meta_path.with_suffix(f"{meta_path.suffix}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(meta, indent=2))
    os.replace(tmp_path, meta_path)


def load_or_compute(
    real_set_cfg: Dict,
    filename: str,
    compute_fn: Callable[[DataLoader], object],
    device: str,
    cache_identity: Any = None,
    sync_dist: bool = True,
) -> Tuple[object, Path]:
    cache_path = resolve_cache_path(real_set_cfg, filename)
    if cache_path.exists():
        mismatches = _cache_meta_mismatches(cache_path.parent, real_set_cfg, filename, cache_identity)
        if not mismatches:
            state = load_cache_file(cache_path, map_location=device)
            return state, cache_path
        if not real_set_cfg.get("auto_resume", True):
            raise RuntimeError(
                f"Cache metadata mismatch at {cache_path} and auto_resume=false. "
                f"Mismatched fields: {', '.join(mismatches)}."
            )

    if not real_set_cfg.get("auto_resume", True):
        raise FileNotFoundError(
            f"Cache miss at {cache_path} and auto_resume=false. "
            f"Either flip auto_resume=true or pre-warm via eval run."
        )

    distributed = bool(sync_dist) and dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    if rank == 0:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        loader = _build_loader(real_set_cfg)
        state = compute_fn(loader)
        tmp_path = cache_path.with_suffix(f"{cache_path.suffix}.{os.getpid()}.rank0.tmp")
        torch.save(state, tmp_path)
        os.replace(tmp_path, cache_path)
        _write_meta(
            cache_path.parent,
            real_set_cfg,
            n_samples=len(loader.dataset),
            filename=filename,
            cache_identity=cache_identity,
        )
    elif distributed:
        while not _cache_is_ready(cache_path, real_set_cfg, filename, cache_identity):
            time.sleep(_CACHE_POLL_INTERVAL_S)

    if not _cache_is_ready(cache_path, real_set_cfg, filename, cache_identity):
        raise FileNotFoundError(f"Cache was not created at {cache_path}.")
    state = load_cache_file(cache_path, map_location=device)
    return state, cache_path
