"""Encoder registry for stage 2 models."""
from typing import Any, Protocol, runtime_checkable

from .DDT import DiTwDDTHead, DiTwDDTHeadIG
from .lightningDiT import LightningDiT


@runtime_checkable
class Stage2ModelProtocol(Protocol):
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...
    def train(self, mode: bool = True) -> Any: ...
    def eval(self) -> Any: ...


__all__ = ["DiTwDDTHead", "DiTwDDTHeadIG", "LightningDiT", "Stage2ModelProtocol"]
