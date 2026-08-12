"""Structured logging used by the evaluation runner."""
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class RunContext:
    run_id: str
    stage: str
    rank: int = 0
    world_size: int = 1
    output_dir: Optional[str] = None


@dataclass(frozen=True)
class SinkConfig:
    level: str = "INFO"
    console: bool = True
    file: bool = True
    file_name: str = "run.log"
    jsonl: bool = False
    rank0_only: bool = True


class _ContextFilter(logging.Filter):
    def __init__(self, context: RunContext):
        super().__init__()
        self._context = context

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "run_id"):
            record.run_id = self._context.run_id
        if not hasattr(record, "stage"):
            record.stage = self._context.stage
        if not hasattr(record, "rank"):
            record.rank = self._context.rank
        if not hasattr(record, "world_size"):
            record.world_size = self._context.world_size
        return True


class _JsonlFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": datetime.utcfromtimestamp(record.created).isoformat(timespec="milliseconds") + "Z",
            "level": record.levelname,
            "logger": record.name,
            "run_id": getattr(record, "run_id", "-"),
            "stage": getattr(record, "stage", "-"),
            "rank": getattr(record, "rank", 0),
            "world_size": getattr(record, "world_size", 1),
            "message": record.getMessage(),
        }
        if hasattr(record, "event"):
            payload["event"] = getattr(record, "event")
        if hasattr(record, "payload"):
            payload["payload"] = getattr(record, "payload")
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class LoggerFactory:
    def __init__(self, context: RunContext, sink: Optional[SinkConfig] = None, root_name: str = "Evaluation"):
        self._context = context
        self._sink = sink or SinkConfig()
        self._root_name = root_name
        self._root_logger: Optional[logging.Logger] = None

    def get_root(self) -> logging.Logger:
        if self._root_logger is not None:
            return self._root_logger

        logger = logging.getLogger(self._root_name)
        logger.setLevel(getattr(logging, self._sink.level.upper(), logging.INFO))

        if getattr(logger, "_evaluation_configured", False):
            self._root_logger = logger
            return logger

        context_filter = _ContextFilter(self._context)
        effective_console = self._sink.console and (self._context.rank == 0 or not self._sink.rank0_only)
        effective_file = self._sink.file and self._context.output_dir is not None and (
            self._context.rank == 0 or not self._sink.rank0_only
        )

        if self._sink.jsonl:
            formatter: logging.Formatter = _JsonlFormatter()
        else:
            formatter = logging.Formatter(
                fmt="%(asctime)s %(levelname)s [%(name)s] [stage=%(stage)s run=%(run_id)s rank=%(rank)s/%(world_size)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

        if effective_console:
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            sh.addFilter(context_filter)
            logger.addHandler(sh)

        if effective_file:
            os.makedirs(self._context.output_dir, exist_ok=True)
            fh = logging.FileHandler(os.path.join(self._context.output_dir, self._sink.file_name), encoding="utf-8")
            fh.setFormatter(formatter)
            fh.addFilter(context_filter)
            logger.addHandler(fh)

        logger.propagate = False
        logger._evaluation_configured = True
        logger.run_id = self._context.run_id
        self._root_logger = logger
        return logger

    def get(self, name: str) -> logging.Logger:
        root = self.get_root()
        logger = logging.getLogger(name)
        logger.setLevel(root.level)
        logger.propagate = True
        return logger

    def close(self) -> None:
        logger = self.get_root()
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
        if hasattr(logger, "_evaluation_configured"):
            delattr(logger, "_evaluation_configured")


LoggingConfig = SinkConfig


def create_logger(
    name: str,
    output_dir: Optional[str] = None,
    run_id: Optional[str] = None,
    rank: int = 0,
    world_size: int = 1,
    cfg: Optional[LoggingConfig] = None,
    stage: str = "eval",
) -> logging.Logger:
    run_id_value = run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    context = RunContext(run_id=run_id_value, stage=stage, rank=rank, world_size=world_size, output_dir=output_dir)
    factory = LoggerFactory(context=context, sink=cfg, root_name=name)
    return factory.get_root()


def log_event(logger: logging.Logger, event: str, payload: Dict[str, Any]) -> None:
    logger.info(event, extra={"event": event, "payload": payload})
