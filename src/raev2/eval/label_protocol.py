"""Deterministic class-label plans for class-conditional evaluation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_metadata_label_plan(
    metadata_path: str | Path,
    num_samples: int,
    *,
    label_key: str = "label",
    num_classes: int | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Load the first ``num_samples`` labels in metadata order.

    The metadata order is part of the evaluation protocol. Returning a
    manifest makes the exact class histogram and label sequence auditable.
    """
    path = Path(metadata_path)
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")
    if not path.is_file():
        raise FileNotFoundError(f"label metadata not found: {path}")

    labels: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if len(labels) >= num_samples:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            if label_key not in row:
                raise KeyError(f"{path}:{line_number} missing label key {label_key!r}")
            label = int(row[label_key])
            if label < 0 or (num_classes is not None and label >= num_classes):
                raise ValueError(
                    f"{path}:{line_number} label {label} outside "
                    f"[0, {num_classes})"
                )
            labels.append(label)

    if len(labels) != num_samples:
        raise ValueError(
            f"label metadata contains {len(labels)} usable rows, expected {num_samples}: {path}"
        )

    sequence_bytes = ",".join(map(str, labels)).encode("ascii")
    counts = Counter(labels)
    manifest = {
        "protocol": "metadata_order",
        "metadata_path": str(path),
        "metadata_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "label_key": label_key,
        "num_samples": num_samples,
        "num_classes": num_classes,
        "label_counts": {str(label): counts.get(label, 0) for label in range(num_classes or (max(labels) + 1))},
        "label_sequence_sha256": hashlib.sha256(sequence_bytes).hexdigest(),
    }
    return labels, manifest
