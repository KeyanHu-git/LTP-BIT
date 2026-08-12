import logging
import os
import sys


def create_logger(logging_dir: str | None, logger_name: str) -> logging.Logger:
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_initialized() else 0
    logger = logging.getLogger(logger_name)

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.propagate = False

    if rank != 0 or logging_dir is None:
        logger.setLevel(logging.CRITICAL + 1)
        return logger

    os.makedirs(logging_dir, exist_ok=True)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(os.path.join(logging_dir, "log.txt"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def format_metric_value(key: str, value: float) -> str:
    if "lr" in key.lower():
        return f"{value:.8e}"
    return f"{value:.4f}"


def format_metrics(metrics: dict) -> str:
    return ", ".join(f"{key}: {format_metric_value(key, value)}" for key, value in metrics.items())
