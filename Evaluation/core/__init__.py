from .evaluator import Evaluator

__all__ = ["Evaluator", "BatchEvaluator", "EvaluationTask"]


def __getattr__(name):
    if name in {"BatchEvaluator", "EvaluationTask"}:
        from .batch_evaluator import BatchEvaluator, EvaluationTask
        return {"BatchEvaluator": BatchEvaluator, "EvaluationTask": EvaluationTask}[name]
    raise AttributeError(name)
