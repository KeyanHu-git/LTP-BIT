#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (str(ROOT), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from raev2.configs import load_config


WRAPPER_PREFIXES = ("module.", "_orig_mod.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_torch(path: Path):
    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def normalize_key(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in WRAPPER_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key


def extract_decoder_state(checkpoint: Mapping, source: str) -> dict[str, torch.Tensor]:
    state = checkpoint.get(source)
    if not isinstance(state, Mapping):
        raise KeyError(f"Stage1 checkpoint has no mapping named {source!r}.")

    decoder = {}
    for raw_key, tensor in state.items():
        key = normalize_key(raw_key)
        if not key.startswith("decoder."):
            continue
        key = key.removeprefix("decoder.")
        if key in decoder:
            raise ValueError(f"Duplicate decoder key after prefix normalization: {key}")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Decoder state contains a non-tensor value: {raw_key}")
        decoder[key] = tensor

    if not decoder:
        raise ValueError("No decoder.* tensors found in the selected Stage1 state.")
    return decoder


def validate_schema(
    decoder: Mapping[str, torch.Tensor], reference: Mapping[str, torch.Tensor]
) -> dict:
    decoder_keys = set(decoder)
    reference_keys = set(reference)
    missing = sorted(reference_keys - decoder_keys)
    unexpected = sorted(decoder_keys - reference_keys)
    shape_mismatches = {
        key: {
            "expected": list(reference[key].shape),
            "actual": list(decoder[key].shape),
        }
        for key in sorted(decoder_keys & reference_keys)
        if decoder[key].shape != reference[key].shape
    }
    report = {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatches,
    }
    if missing or unexpected or shape_mismatches:
        raise RuntimeError(json.dumps(report, indent=2))
    return report


def save_atomic(state: Mapping[str, torch.Tensor], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(state), temporary)
        restored = load_torch(temporary)
        validate_schema(restored, state)
        for key, tensor in state.items():
            if restored[key].dtype != tensor.dtype or not torch.equal(restored[key], tensor):
                raise RuntimeError(f"Round-trip verification failed for decoder tensor: {key}")
        os.replace(temporary, output)
        output.chmod(0o644)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a strictly validated decoder from a full RAEv2 Stage1 checkpoint."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", choices=("ema", "model"), default="ema")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {args.output}")

    config = load_config(args.config)
    reference_path = Path(config.stage_1.params.pretrained_decoder_path)
    if not reference_path.is_absolute():
        reference_path = Path(__file__).resolve().parents[2] / reference_path

    checkpoint = load_torch(args.checkpoint)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Stage1 checkpoint must be a mapping.")
    decoder = extract_decoder_state(checkpoint, args.source)

    reference = load_torch(reference_path)
    if not isinstance(reference, Mapping):
        raise TypeError("Reference decoder must be a state-dict mapping.")
    report = validate_schema(decoder, reference)

    save_atomic(decoder, args.output)

    manifest = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_step": checkpoint.get("step"),
        "source": args.source,
        "reference_decoder": str(reference_path),
        "reference_decoder_sha256": sha256_file(reference_path),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "tensor_count": len(decoder),
        **report,
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
