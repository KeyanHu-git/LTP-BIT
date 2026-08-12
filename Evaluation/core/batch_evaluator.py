import os
from dataclasses import dataclass, field
from multiprocessing import Process, Queue
from pathlib import Path
from queue import Empty
from typing import Dict, List, Optional

import torch

from ..builder import build_dataset
from ..runner import run_evaluation_config
from ..writers import JsonWriter, ExcelWriter, TxtWriter
from .launchers import run_eval_task_with_torchrun


@dataclass
class EvaluationTask:
    task_id: str
    pred_dir: str
    gt_dir: Optional[str]
    metrics_cfg: List[Dict]
    dataloader_cfg: Dict
    preprocess_cfg: Optional[List[Dict]] = None
    device: str = "cuda:0"
    real_set: Optional[Dict] = None


def auto_tune_workers(num_processes: int, total_cores: int) -> int:
    available_cores = max(1, total_cores - 2)
    workers_per_process = max(1, available_cores // num_processes)
    return min(workers_per_process, 8)


def _as_bool(value, *, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    if value in (0, 1):
        return bool(value)
    raise TypeError(f"{key} must be a boolean, got {value!r}")


def _validate_config(config: Dict) -> None:
    if "data" not in config:
        raise ValueError("Missing required config key: 'data'")
    if "pred_root" not in config["data"]:
        raise ValueError("Missing config key: 'data.pred_root'")
    pred_root = Path(config["data"]["pred_root"])
    if not pred_root.exists():
        raise FileNotFoundError(f"data.pred_root does not exist: {pred_root}")
    gt_root = config["data"].get("gt_root")
    if gt_root is not None and not Path(gt_root).exists():
        raise FileNotFoundError(f"data.gt_root does not exist: {gt_root}")
    for reference_key in ("real_set", "prior_set"):
        reference_set = config.get(reference_key)
        if (
            isinstance(reference_set, dict)
            and reference_set.get("image_dir")
            and not Path(reference_set["image_dir"]).exists()
        ):
            raise FileNotFoundError(
                f"{reference_key}.image_dir does not exist: {reference_set['image_dir']}"
            )
    # gt_root is optional: when null, eval runs in fake-only mode (Stage 1 c2t).
    if "metrics" not in config or not config["metrics"]:
        raise ValueError("At least one metric is required in 'metrics'")
    expected = config["data"].get("expected_num_samples")
    if expected is not None and int(expected) <= 0:
        raise ValueError(f"data.expected_num_samples must be positive, got {expected}")


def _dataset_config(task: EvaluationTask) -> Dict:
    config: Dict = {
        "type": "EvalPairedDataset" if task.gt_dir is not None else "EvalSingleDataset",
        "pred_dir": task.pred_dir,
        "preprocess": task.preprocess_cfg,
    }
    if task.gt_dir is not None:
        config["gt_dir"] = task.gt_dir
    return config


def eval_single_folder(task: EvaluationTask, progress_queue=None) -> Dict:
    dataloader_cfg = dict(task.dataloader_cfg)
    num_workers = int(dataloader_cfg.get("num_workers", 0) or 0)
    dataloader_cfg["pin_memory"] = task.device.startswith("cuda")
    dataloader_cfg["persistent_workers"] = num_workers > 0
    dataloader_cfg.setdefault("drop_last", False)
    progress_callback = None
    if progress_queue:
        progress_callback = lambda done, total: progress_queue.put({"task_id": task.task_id, "progress": done / total})
    run = run_evaluation_config(
        {
            "dataset": _dataset_config(task),
            "dataloader": dataloader_cfg,
            "metrics": task.metrics_cfg,
            "real_set": task.real_set,
        },
        task.device,
        use_tqdm=False,
        distributed=False,
        progress_callback=progress_callback,
    )

    return {
        "task_id": task.task_id,
        "metrics": run.metrics,
        "num_samples": run.num_samples,
        "elapsed_time": run.duration_s,
        "status": "success"
    }


class BatchEvaluator:
    def __init__(self, config: Dict):
        if "outputs" not in config and "output" in config:
            config["outputs"] = config.pop("output")
        _validate_config(config)
        
        config.setdefault("outputs", {})
        config["outputs"].setdefault("root_dir", "experiments/eval/batch_eval")
        config["outputs"].setdefault("formats", ["json"])
        config["outputs"].setdefault("save_per_task", True)
        config["outputs"].setdefault("save_summary", True)
        
        config.setdefault("execution", {})
        config["execution"].setdefault("gpu_ids", "0")
        config["execution"].setdefault("max_workers", 4)
        config["execution"].setdefault("dataloader", {})
        config["execution"]["dataloader"].setdefault("batch_size", 32)
        config["execution"]["dataloader"].setdefault("num_workers", 0)
        config["execution"]["dataloader"].setdefault("pin_memory", True)

        prior_set = config.get("prior_set")
        if prior_set is not None:
            for metric_cfg in config["metrics"]:
                if metric_cfg.get("type") == "PriorFID":
                    metric_cfg.setdefault("prior_set", prior_set)
        
        self.config = config
        self.gpu_ids = self._parse_gpu_ids(config["execution"]["gpu_ids"])
        self.max_workers = config["execution"]["max_workers"]
        self.distributed_per_task = _as_bool(
            config["execution"].get("distributed_per_task", False),
            key="execution.distributed_per_task",
        )
        self.mode = config.get("mode", "multi_folder")
        
        self.writers = {
            "json": JsonWriter(),
            "excel": ExcelWriter(),
            "txt": TxtWriter()
        }
    
    def _parse_gpu_ids(self, gpu_ids_str: str) -> List[int]:
        if gpu_ids_str == "cpu":
            return []
        return [int(x.strip()) for x in gpu_ids_str.split(",")]
    
    def scan_tasks(self) -> List[EvaluationTask]:
        tasks = []
        data_cfg = self.config["data"]
        pred_root = Path(data_cfg["pred_root"])
        gt_root_raw = data_cfg.get("gt_root")
        gt_root = Path(gt_root_raw) if gt_root_raw is not None else None
        skip_missing = _as_bool(data_cfg.get("skip_missing", False), key="data.skip_missing")
        skip_incomplete = _as_bool(data_cfg.get("skip_incomplete", False), key="data.skip_incomplete")

        if self.mode == "single_folder":
            task = self._create_task("single", pred_root, gt_root)
            self._validate_task_data(task)
            tasks.append(task)
        else:
            include = data_cfg.get("include")
            exclude = data_cfg.get("exclude", [])

            if include:
                # entries may be plain subdir names or relative paths ("<exp>/inference/<run>")
                requested = [pred_root / entry for entry in include]
                missing = [str(path) for path in requested if not path.is_dir()]
                if missing and not skip_missing:
                    raise FileNotFoundError(f"data.include dirs not found under {pred_root}: {missing}")
                if missing:
                    print(f"Skipping {len(missing)} missing evaluation method(s): {missing}")
                subdirs = [path for path in requested if path.is_dir()]
            else:
                subdirs = [d for d in pred_root.iterdir() if d.is_dir()]
            subdirs = [d for d in subdirs if d.name not in exclude]

            for subdir in sorted(subdirs):
                task_id = subdir.relative_to(pred_root).as_posix().replace("/", "__")
                task = self._create_task(task_id, subdir, gt_root)
                try:
                    self._validate_task_data(task)
                except (FileNotFoundError, RuntimeError, ValueError) as exc:
                    if not skip_incomplete:
                        raise
                    print(f"Skipping incomplete evaluation method '{task_id}': {exc}")
                    continue
                tasks.append(task)

        return tasks

    def _validate_task_data(self, task: EvaluationTask) -> None:
        data_cfg = self.config["data"]
        expected = data_cfg.get("expected_num_samples")
        validate_files = _as_bool(data_cfg.get("skip_incomplete", False), key="data.skip_incomplete")
        if expected is None and not validate_files:
            return
        dataset = build_dataset(_dataset_config(task))
        if expected is None:
            return
        expected = int(expected)
        if len(dataset) != expected:
            raise ValueError(f"prediction count is {len(dataset)}, expected {expected}")
        gt_files = getattr(dataset, "gt_files", None)
        if gt_files is not None and len(gt_files) != expected:
            raise ValueError(f"ground-truth count is {len(gt_files)}, expected {expected}")

    def _create_task(self, task_id: str, pred_dir: Path, gt_dir: Optional[Path]) -> EvaluationTask:
        total_processes = len(self.gpu_ids) * 1 if self.gpu_ids else self.max_workers
        num_workers = auto_tune_workers(min(total_processes, self.max_workers), os.cpu_count() or 4)
        
        dataloader_cfg = dict(self.config["execution"]["dataloader"])
        if "num_workers" not in dataloader_cfg or dataloader_cfg.get("num_workers") is None:
            dataloader_cfg["num_workers"] = num_workers
        
        return EvaluationTask(
            task_id=task_id,
            pred_dir=str(pred_dir),
            gt_dir=str(gt_dir) if gt_dir is not None else None,
            metrics_cfg=self.config["metrics"],
            dataloader_cfg=dataloader_cfg,
            preprocess_cfg=self.config["data"].get("preprocess"),
            device="cuda:0",
            real_set=self.config.get("real_set"),
        )
    
    def execute(self) -> List[Dict]:
        tasks = self.scan_tasks()

        if not tasks:
            print("Warning: No tasks found to evaluate!")
            return []

        self._prewarm_real_cache()
        if len(self.gpu_ids) == 0:
            return self._execute_cpu(tasks)
        if self.distributed_per_task and len(self.gpu_ids) > 1:
            return self._execute_gpu_distributed_per_task(tasks)
        else:
            return self._execute_gpu(tasks)

    def _prewarm_real_cache(self) -> None:
        """Build any missing real-side cache file once in the parent process before fork.
        Avoids N workers each running a full feature-extraction pass on the same refset.

        Device default is cpu (safe with multiprocessing fork on CUDA); set
        execution.prewarm_device='cuda:N' to opt into GPU-side cold compute when the user
        accepts the fork-after-CUDA caveat.
        """
        real_set = self.config.get("real_set")
        from ..registry import METRICS
        from ..builder import build_metric
        from ..metrics.distributional._cache import resolve_cache_path

        needed = []
        for cfg in self.config["metrics"]:
            cls = METRICS.get(cfg["type"])
            if cls is None or not hasattr(cls, "_real_cache_filename"):
                continue
            reference_key = getattr(cls, "_reference_set_key", "real_set")
            reference_set = cfg.get(reference_key, self.config.get(reference_key))
            if reference_set is None:
                continue
            cache_filename = cfg.get("cache_filename", cls._real_cache_filename)
            if not resolve_cache_path(reference_set, cache_filename).exists():
                needed.append((cfg, reference_set))
        if not needed:
            return
        prewarm_device = self.config.get("execution", {}).get("prewarm_device", "cpu")
        print(f"[BatchEvaluator] pre-warming {len(needed)} real-set cache file(s) on {prewarm_device}")
        for cfg, reference_set in needed:
            build_metric(cfg, device=prewarm_device, real_set=reference_set)
        if prewarm_device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _execute_cpu(self, tasks: List[EvaluationTask]) -> List[Dict]:
        results = []
        for task in tasks:
            task.device = "cpu"
            result = eval_single_folder(task)
            results.append(result)
            self._save_single_result(result)
        
        self._save_summary(results)
        return results
    
    def _execute_gpu(self, tasks: List[EvaluationTask]) -> List[Dict]:
        task_queue = Queue()
        result_queue = Queue()
        progress_queue = Queue()
        
        for task in tasks:
            task_queue.put(task)
        
        processes = []
        for i in range(self.max_workers):
            gpu_id = self.gpu_ids[i % len(self.gpu_ids)]
            p = Process(
                target=self._worker_process,
                args=(i, gpu_id, task_queue, result_queue, progress_queue)
            )
            p.start()
            processes.append(p)
        
        for _ in range(self.max_workers):
            task_queue.put(None)
        
        try:
            results = self._collect_results_with_progress(
                len(tasks), result_queue, progress_queue, processes
            )
        finally:
            self._terminate_processes(processes)
        
        self._save_summary(results)
        return results

    def _worker_process(self, worker_id, gpu_id, task_queue, result_queue, progress_queue):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        while True:
            task = task_queue.get()
            if task is None:
                break
            
            task.device = "cuda:0"
            result = eval_single_folder(task, progress_queue)
            result_queue.put(result)

    def _execute_gpu_distributed_per_task(self, tasks: List[EvaluationTask]) -> List[Dict]:
        """Evaluate each folder with all configured GPUs.

        The normal batch path parallelizes across folders: one task is assigned to
        one worker/GPU. For a single 5K folder this leaves the other GPUs idle.
        The project-level Evaluation.tools.eval runner already supports torchrun
        and the distributional metrics already synchronize their states, so this
        path reuses that implementation for per-folder data sharding.
        """
        results = []
        for idx, task in enumerate(tasks, start=1):
            print(f"[distributed_per_task] {idx}/{len(tasks)} {task.task_id} on GPUs {self.gpu_ids}")
            result = self._run_task_with_torchrun(task)
            results.append(result)
            self._save_single_result(result)
        self._save_summary(results)
        return results

    def _run_task_with_torchrun(self, task: EvaluationTask) -> Dict:
        return run_eval_task_with_torchrun(
            task_id=task.task_id,
            pred_dir=task.pred_dir,
            gt_dir=task.gt_dir,
            metrics_cfg=task.metrics_cfg,
            dataloader_cfg=task.dataloader_cfg,
            preprocess_cfg=task.preprocess_cfg,
            real_set=task.real_set,
            output_root=Path(self.config["outputs"]["root_dir"]),
            gpu_ids=self.gpu_ids,
            master_port=int(self.config["execution"].get("master_port", 29680)),
        )

    def _worker_failure_message(self, processes, completed, total_tasks):
        failed = [
            (worker_id, p.pid, p.exitcode)
            for worker_id, p in enumerate(processes)
            if p.exitcode not in (None, 0)
        ]
        if failed:
            details = ", ".join(
                f"worker {worker_id} pid={pid} exitcode={exitcode}"
                for worker_id, pid, exitcode in failed
            )
            return (
                f"Batch evaluator worker failed "
                f"({completed}/{total_tasks} results collected): {details}"
            )
        if completed < total_tasks and all(p.exitcode is not None for p in processes):
            return (
                f"All batch evaluator workers exited before all results were collected "
                f"({completed}/{total_tasks})."
            )
        return None

    def _raise_if_worker_failed(self, processes, completed, total_tasks):
        message = self._worker_failure_message(processes, completed, total_tasks)
        if message:
            raise RuntimeError(message)

    def _terminate_processes(self, processes):
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join(timeout=5)
    
    def _collect_results_with_progress(self, total_tasks, result_queue, progress_queue, processes):
        try:
            from rich.progress import Progress
            use_rich = True
        except ImportError:
            use_rich = False
        
        results = []
        
        if use_rich:
            task_progress = {}
            
            with Progress() as progress:
                overall = progress.add_task("[cyan]Overall Progress", total=total_tasks)
                
                completed = 0
                while completed < total_tasks:
                    while True:
                        try:
                            update = progress_queue.get_nowait()
                            task_id = update["task_id"]
                            
                            if task_id not in task_progress:
                                task_progress[task_id] = progress.add_task(
                                    f"[green]{task_id}",
                                    total=100
                                )
                            
                            progress.update(
                                task_progress[task_id],
                                completed=int(update["progress"] * 100)
                            )
                        except Empty:
                            break
                    
                    try:
                        result = result_queue.get(timeout=0.1)
                    except Empty:
                        self._raise_if_worker_failed(processes, completed, total_tasks)
                    else:
                        results.append(result)
                        completed += 1
                        progress.update(overall, advance=1)
                        
                        if result["task_id"] in task_progress:
                            progress.update(
                                task_progress[result["task_id"]],
                                completed=100,
                                description=f"[green]✓ {result['task_id']}"
                            )
                        
                        self._save_single_result(result)
        else:
            completed = 0
            while completed < total_tasks:
                try:
                    result = result_queue.get(timeout=0.1)
                except Empty:
                    self._raise_if_worker_failed(processes, completed, total_tasks)
                else:
                    results.append(result)
                    completed += 1
                    print(f"Completed: {completed}/{total_tasks} - {result['task_id']}")
                    self._save_single_result(result)
        
        for p in processes:
            p.join()
        self._raise_if_worker_failed(processes, len(results), total_tasks)
        
        return results
    
    def _save_single_result(self, result: Dict):
        if not self.config["outputs"].get("save_per_task", True):
            return
        
        output_root = Path(self.config["outputs"]["root_dir"])
        task_dir = output_root / "per_task" / result["task_id"]
        
        formats = self.config["outputs"].get("formats", ["json"])
        
        if "json" in formats:
            self.writers["json"].write_single(result, task_dir / "metrics.json")
        if "excel" in formats:
            self.writers["excel"].write_single(result, task_dir / "metrics.xlsx")
        if "txt" in formats:
            self.writers["txt"].write_single(result, task_dir / "metrics.txt")
    
    def _save_summary(self, results: List[Dict]):
        if not self.config["outputs"].get("save_summary", True):
            return
        
        output_root = Path(self.config["outputs"]["root_dir"])
        
        formats = self.config["outputs"].get("formats", ["json"])
        
        if "json" in formats:
            self.writers["json"].write_summary(results, output_root / "summary.json")
        if "excel" in formats:
            self.writers["excel"].write_summary(results, output_root / "summary.xlsx")
        if "txt" in formats:
            self.writers["txt"].write_summary(results, output_root / "summary.txt")
