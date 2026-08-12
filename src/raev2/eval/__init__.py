from .registry import EVAL_EXPORTS, register_all_modules


register_all_modules()

__all__ = ["EVAL_EXPORTS", "register_all_modules", *sorted(EVAL_EXPORTS.module_dict)]


def __getattr__(name):
    try:
        return EVAL_EXPORTS.get(name)
    except KeyError as exc:
        raise AttributeError(name) from exc
