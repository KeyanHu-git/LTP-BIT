"""RAEv2 configuration loading and typed component definitions."""

from .loader import load_config
from .shared import (
    DatasetConfig,
    EvalConfig,
    MiscConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)
from .stage1 import (
    DiscAugmentConfig,
    DiscriminatorArchConfig,
    GanConfig,
    GanLossConfig,
    Stage1Config,
)
from .stage2 import (
    ConditioningConfig,
    GuidanceConfig,
    RepaConfig,
    SamplerConfig,
    Stage2Config,
    TransportConfig,
)

__all__ = [
    "load_config",
    # Shared
    "ModelConfig",
    "MiscConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "DatasetConfig",
    "EvalConfig",
    "TrainingConfig",
    # Stage 1
    "Stage1Config",
    "DiscriminatorArchConfig",
    "DiscAugmentConfig",
    "GanLossConfig",
    "GanConfig",
    # Stage 2
    "Stage2Config",
    "TransportConfig",
    "SamplerConfig",
    "GuidanceConfig",
    "ConditioningConfig",
    "RepaConfig",
]
