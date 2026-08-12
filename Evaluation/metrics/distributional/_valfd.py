"""FDr/FDr-K validation normalizer helpers.

The project-level convention matches other distributional caches:
read cached artifacts when present, otherwise compute and persist them when the
caller supplies enough refset information.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

from ._cache import load_cache_file, resolve_cache_path
from .frechet import frechet_from_feature_state


EnsureRefsetFn = Callable[[Dict[str, Any]], None]


def safe_cache_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def default_valfd_path(cache_root: str | Path, val_name: str, real_name: str) -> Path:
    return Path(cache_root) / "_valfd" / f"{safe_cache_name(val_name)}__vs__{safe_cache_name(real_name)}.json"


def read_valfd_json(path: str | Path) -> Dict[str, float]:
    payload = _read_payload(path)
    values = payload.get("normalizers", payload)
    return {str(k): float(v) for k, v in values.items()}


def _read_payload(path: str | Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"FDr val_fd JSON must contain an object, got {type(payload).__name__}: {path}")
    return payload


def validate_valfd(values: Mapping[str, Any], names: Iterable[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name in names:
        if name not in values:
            raise KeyError(f"FDr val_fd missing '{name}'. Provided keys: {list(values)}")
        value = float(values[name])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"FDr val_fd for '{name}' must be finite and > 0, got {value}.")
        out[name] = value
    return out


def _refset_cfg(spec: Any, cache_root: str) -> Dict[str, Any]:
    if isinstance(spec, str):
        return {"name": spec, "cache_root": cache_root}
    if isinstance(spec, Mapping) and "name" in spec:
        out = dict(spec)
        out.setdefault("cache_root", cache_root)
        return out
    raise TypeError("val_set must be a cache name string or a mapping with `name`.")


def _is_auto_spec(val_fd: Any) -> bool:
    return isinstance(val_fd, Mapping) and ("val_set" in val_fd or "mode" in val_fd)


def _resolve_explicit_valfd(val_fd: Any, names: Iterable[str]) -> Dict[str, float]:
    if isinstance(val_fd, str):
        values = read_valfd_json(val_fd)
    elif isinstance(val_fd, Mapping) and "path" in val_fd:
        values = read_valfd_json(str(val_fd["path"]))
    elif isinstance(val_fd, Mapping):
        values = {str(k): float(v) for k, v in val_fd.items()}
    else:
        raise TypeError(
            "val_fd must be null, numeric dict, path string, {path: ...}, or {val_set: ...}."
        )
    return validate_valfd(values, names)


def _missing_cache(refset: Dict[str, Any], cache_filenames: Mapping[str, str]) -> bool:
    return any(not resolve_cache_path(refset, filename).exists() for filename in cache_filenames.values())


def _payload_matches_auto_spec(
    payload: Mapping[str, Any],
    *,
    names: Iterable[str],
    real_cfg: Mapping[str, Any],
    val_cfg: Mapping[str, Any],
    protocol: Any = None,
) -> bool:
    if payload.get("schema") != "fdr-valfd-v1":
        return False
    payload_real = payload.get("real_set", {})
    payload_val = payload.get("val_set", {})
    if payload_real.get("name") != real_cfg.get("name"):
        return False
    if payload_val.get("name") != val_cfg.get("name"):
        return False
    if protocol is not None and payload.get("protocol") != protocol:
        return False
    payload_backbones = set(payload.get("backbones", []))
    if not set(names).issubset(payload_backbones):
        return False
    return True


def _ensure_feature_cache(
    refset: Dict[str, Any],
    *,
    role: str,
    cache_filenames: Mapping[str, str],
    ensure_refset: Optional[EnsureRefsetFn],
) -> None:
    if not _missing_cache(refset, cache_filenames):
        return
    if ensure_refset is None:
        raise FileNotFoundError(f"Missing {role} feature cache for valFD: {refset['name']}")
    if "image_dir" not in refset:
        missing = [
            str(resolve_cache_path(refset, filename))
            for filename in cache_filenames.values()
            if not resolve_cache_path(refset, filename).exists()
        ]
        raise FileNotFoundError(
            f"Missing {role} feature cache for valFD cache '{refset['name']}', and no "
            f"`image_dir` was supplied to compute it. Missing examples: {missing[:3]}"
        )
    ensure_refset(refset)


def compute_valfd_from_caches(
    real_set: Dict[str, Any],
    val_set: Dict[str, Any],
    cache_filenames: Mapping[str, str],
) -> Dict[str, float]:
    normalizers: Dict[str, float] = {}
    for name, filename in cache_filenames.items():
        real_path = resolve_cache_path(real_set, filename)
        val_path = resolve_cache_path(val_set, filename)
        if not real_path.exists() or not val_path.exists():
            raise FileNotFoundError(
                f"Missing valFD cache for '{name}': "
                f"real={real_path} exists={real_path.exists()}, "
                f"val={val_path} exists={val_path.exists()}."
            )
        normalizers[name] = frechet_from_feature_state(
            load_cache_file(real_path, map_location="cpu"),
            load_cache_file(val_path, map_location="cpu"),
        )
    return validate_valfd(normalizers, cache_filenames)


def write_valfd_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def resolve_val_fd(
    val_fd: Optional[Any],
    *,
    names: Iterable[str],
    real_set: Optional[Dict[str, Any]] = None,
    cache_filenames: Optional[Mapping[str, str]] = None,
    ensure_refset: Optional[EnsureRefsetFn] = None,
    logger: Any = None,
) -> Optional[Dict[str, float]]:
    names = list(names)
    if val_fd is None:
        return None
    if not _is_auto_spec(val_fd):
        return _resolve_explicit_valfd(val_fd, names)

    mode = str(val_fd.get("mode", "cache_pair")).replace("-", "_").lower()
    if mode != "cache_pair":
        raise ValueError("val_fd.mode supports only `cache_pair`.")
    if real_set is None or cache_filenames is None:
        raise ValueError("cache_pair val_fd requires real_set and cache_filenames.")

    cache_root = str(val_fd.get("cache_root") or real_set.get("cache_root") or "Evaluation/cached_features")
    protocol = val_fd.get("protocol")
    real_cfg = dict(real_set)
    real_cfg.setdefault("cache_root", cache_root)
    val_cfg = _refset_cfg(val_fd.get("val_set"), cache_root)
    if real_cfg["name"] == val_cfg["name"] and Path(real_cfg["cache_root"]) == Path(val_cfg["cache_root"]):
        raise ValueError("cache_pair val_fd requires distinct real_set and val_set caches.")

    path = Path(val_fd.get("path") or val_fd.get("cache_path") or default_valfd_path(
        cache_root, str(val_cfg["name"]), str(real_cfg["name"])
    ))
    if path.exists():
        payload = _read_payload(path)
        values = payload.get("normalizers", payload)
        if (
            all(name in values for name in names)
            and _payload_matches_auto_spec(
                payload,
                names=names,
                real_cfg=real_cfg,
                val_cfg=val_cfg,
                protocol=protocol,
            )
        ):
            cached = validate_valfd(values, names)
            if logger is not None:
                logger.info("Using cached val_fd normalizers: %s", path)
            return cached
        if logger is not None:
            logger.warning("val_fd cache %s does not match current spec for %s; recomputing.", path, names)

    _ensure_feature_cache(
        real_cfg, role="real_set", cache_filenames=cache_filenames, ensure_refset=ensure_refset
    )
    _ensure_feature_cache(
        val_cfg, role="val_set", cache_filenames=cache_filenames, ensure_refset=ensure_refset
    )

    normalizers = compute_valfd_from_caches(real_cfg, val_cfg, cache_filenames)
    write_valfd_json(path, {
        "schema": "fdr-valfd-v1",
        "definition": "FD_k(val_set, real_set) computed from cached feature statistics",
        "real_set": {"name": real_cfg["name"], "cache_root": str(real_cfg["cache_root"])},
        "val_set": {"name": val_cfg["name"], "cache_root": str(val_cfg["cache_root"])},
        "backbones": names,
        "protocol": protocol,
        "normalizers": normalizers,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
    })
    if logger is not None:
        logger.info("Wrote val_fd normalizers: %s", path)
    return normalizers
