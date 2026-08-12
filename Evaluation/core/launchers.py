import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml


def run_eval_task_with_torchrun(
    *,
    task_id: str,
    pred_dir: str,
    gt_dir: Optional[str],
    metrics_cfg: List[Dict],
    dataloader_cfg: Dict,
    preprocess_cfg: Optional[List[Dict]],
    real_set: Optional[Dict],
    output_root: Path,
    gpu_ids: Sequence[int],
    master_port: int = 29680,
) -> Dict:
    """Run one eval folder through the project DDP eval runner.

    Batch evaluation normally parallelizes across folders. This helper covers the
    complementary case: one large folder should be sharded across all GPUs, while
    metric synchronization remains owned by Evaluation.tools.eval and the metric
    implementations.
    """
    project_root = Path(__file__).resolve().parents[2]
    output_root = output_root.resolve()
    ddp_root = output_root / "_distributed_runs"
    cfg_dir = output_root / "_distributed_eval_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    dataset_cfg: Dict = {
        "type": "EvalPairedDataset" if gt_dir is not None else "EvalSingleDataset",
        "pred_dir": pred_dir,
        "preprocess": preprocess_cfg,
    }
    if gt_dir is not None:
        dataset_cfg["gt_dir"] = gt_dir

    eval_cfg = {
        "dataset": dataset_cfg,
        "dataloader": {
            **dict(dataloader_cfg),
            "pin_memory": True,
            "drop_last": False,
        },
        "metrics": metrics_cfg,
    }
    if real_set is not None:
        eval_cfg["real_set"] = real_set

    cfg_path = (cfg_dir / f"{task_id}.yaml").resolve()
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(eval_cfg, f, sort_keys=False, allow_unicode=True)

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={len(gpu_ids)}",
        "--master_addr=127.0.0.1",
        f"--master_port={master_port}",
        "-m",
        "Evaluation.tools.eval",
        str(cfg_path),
        "--device",
        "cuda",
        "--output-dir",
        str(ddp_root),
        "--run-id",
        task_id,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    env["PYTHONPATH"] = f"{project_root}:{project_root / 'src'}:{env.get('PYTHONPATH', '')}"

    start = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(project_root),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output_tail: List[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        output_tail.append(line)
        if len(output_tail) > 200:
            output_tail = output_tail[-200:]
    returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError("".join(output_tail)[-8000:])

    run_dir = ddp_root / task_id
    metrics_path = run_dir / "metrics.summary.json"
    meta_path = run_dir / "meta.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Distributed eval did not write {metrics_path}")

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    num_samples = 0
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        num_samples = int(meta.get("samples", 0))

    return {
        "task_id": task_id,
        "metrics": metrics,
        "num_samples": num_samples,
        "elapsed_time": time.time() - start,
        "status": "success",
        "distributed": True,
        "world_size": len(gpu_ids),
        "run_dir": str(run_dir),
    }
