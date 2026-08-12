from Dataset.registry import DATASETS

from .registry import METRICS
from .core import Evaluator
from .builder import build_metric, build_dataset
from .runner import evaluate_config
from .registration import register_all


__all__ = [
    "METRICS",
    "DATASETS",
    "Evaluator",
    "BatchEvaluator",
    "EvaluationTask",
    "build_metric",
    "build_dataset",
    "evaluate_config",
    "register_all",
]


def __getattr__(name):
    if name in {"BatchEvaluator", "EvaluationTask"}:
        from .core.batch_evaluator import BatchEvaluator, EvaluationTask
        return {"BatchEvaluator": BatchEvaluator, "EvaluationTask": EvaluationTask}[name]
    raise AttributeError(name)
