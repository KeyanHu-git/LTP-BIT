import argparse
import csv
import json
import os
import platform
import sys
from datetime import datetime
from typing import Any, Dict, Tuple

import torch
import torch.distributed as dist
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utils import ConfigLoader
from .. import register_all
from ..runner import resolve_eval_config, run_evaluation_config
from Logger import LoggerFactory, RunContext, SinkConfig, log_event


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluation Runner")
    parser.add_argument("config", help="Path to YAML config")
    parser.add_argument("--device", default="cuda", help="Device used for evaluation")
    parser.add_argument("--output-dir", default="experiments/eval/runs", help="Output directory root")
    parser.add_argument("--run-id", default=None, help="Run id (default: utc timestamp)")
    parser.add_argument("--no-console", action="store_true", help="Disable console logging")
    parser.add_argument("--jsonl-log", action="store_true", help="Write logs in jsonl format")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE", help="Override a resolved config value.")
    return parser.parse_args(argv)


def _utc_now_iso():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _distributed_context() -> Tuple[bool, int, int, int]:
    if not dist.is_available():
        return False, 0, 1, 0
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return world_size > 1, rank, world_size, local_rank


def _init_process_group_if_needed(enabled: bool, requested_device: str) -> None:
    if not enabled or dist.is_initialized():
        return
    backend = "gloo"
    if requested_device.startswith("cuda") and torch.cuda.is_available() and platform.system() != "Windows":
        backend = "nccl"
    dist.init_process_group(backend=backend, init_method="env://")


def _select_device(requested: str, distributed: bool, local_rank: int) -> str:
    if requested.startswith("cuda") and torch.cuda.is_available():
        if distributed:
            torch.cuda.set_device(local_rank)
            return f"cuda:{local_rank}"
        if ":" in requested:
            torch.cuda.set_device(int(requested.split(":", 1)[1]))
        return requested
    return "cpu" if requested == "cpu" else requested


def _write_yaml(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_manifest_csv(path: str, dataset: Any) -> None:
    if not hasattr(dataset, "iter_manifest_rows"):
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "pred_rel", "gt_rel"])
        writer.writeheader()
        for row in dataset.iter_manifest_rows():
            writer.writerow(row)


def main(argv=None):
    args = _parse_args(argv)

    register_all()
    distributed, rank, world_size, local_rank = _distributed_context()
    _init_process_group_if_needed(distributed, args.device)
    # IMPORTANT: set per-rank cuda device BEFORE any dist.broadcast_object_list.
    # NCCL serializes Python objects to a CUDA tensor on torch.cuda.current_device();
    # without set_device(local_rank), every rank would target GPU 0 → contention/hang.
    device = _select_device(args.device, distributed=distributed, local_rank=local_rank)

    # run_id MUST be identical across all ranks. datetime.utcnow() per-rank
    # would silently diverge if processes cross a second boundary (rank 0
    # at HH:MM:59.99, rank 1 at HH:MM:60.01 → different run_dir, split logs).
    # rank-0 generates, broadcast to all.
    if args.run_id:
        run_id = args.run_id
    elif distributed:
        bcast_device = torch.device(device) if str(device).startswith("cuda") else None
        rid_box = [datetime.utcnow().strftime("%Y%m%d_%H%M%S") if rank == 0 else None]
        if bcast_device is not None:
            dist.broadcast_object_list(rid_box, src=0, device=bcast_device)
        else:
            dist.broadcast_object_list(rid_box, src=0)
        run_id = rid_box[0]
    else:
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = _ensure_dir(os.path.join(args.output_dir, run_id))
    start_utc = _utc_now_iso()
    sink_cfg = SinkConfig(
        level="INFO",
        console=not args.no_console,
        file=True,
        jsonl=args.jsonl_log,
        rank0_only=True,
    )
    context = RunContext(run_id=run_id, stage="eval", rank=rank, world_size=world_size, output_dir=run_dir)
    logger_factory = LoggerFactory(context=context, sink=sink_cfg, root_name="Evaluation")
    logger = logger_factory.get_root()
    runner_logger = logger_factory.get("Evaluation.runner")

    runner_logger.info("Run dir: %s", run_dir)
    runner_logger.info("Loading config: %s", args.config)
    cfg = ConfigLoader.load_recursive(args.config, overrides=args.override)
    log_event(runner_logger, "config_loaded", {"path": args.config})
    cfg = resolve_eval_config(cfg)
    cfg["device"] = device

    if rank == 0:
        _write_yaml(os.path.join(run_dir, "config.resolved.yaml"), cfg)

    run = run_evaluation_config(
        cfg,
        device,
        logger=logger,
        use_tqdm=(rank == 0 and not args.no_console),
        distributed=distributed,
    )
    results = run.metrics
    log_event(runner_logger, "dataset_built", {"type": cfg["dataset"].get("type", "?"), "size": len(run.dataset)})
    log_event(runner_logger, "eval_finished", {"duration_s": run.duration_s, "samples": run.num_samples})

    logger.info("Dataset size: %s", len(run.dataset))
    logger.info("Device: %s", device)
    logger.info("Distributed: %s (rank %s/%s)", distributed, rank, world_size)
    end_utc = _utc_now_iso()

    meta = {
        "run_id": run_id,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "duration_s": run.duration_s,
        "samples": run.num_samples,
        "device": device,
        "rank": rank,
        "world_size": world_size,
        "data_roots": run.dataset.manifest_roots() if hasattr(run.dataset, "manifest_roots") else {},
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if rank == 0:
        _write_json(os.path.join(run_dir, "metrics.summary.json"), results)
        _write_manifest_csv(os.path.join(run_dir, "manifest.csv"), run.dataset)
        _write_json(os.path.join(run_dir, "meta.json"), meta)
        log_event(runner_logger, "artifacts_written", {"run_dir": run_dir})

    logger.info("Finished in %.2fs", run.duration_s)
    for k, v in results.items():
        logger.info("%s: %.4f", k, v)

    logger_factory.close()

    if dist.is_initialized():
        if str(device).startswith("cuda") and torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
