from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List


class BaseWriter(ABC):
    @abstractmethod
    def write_single(self, result: Dict, output_path: Path) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def write_summary(self, results: List[Dict], output_path: Path) -> None:
        raise NotImplementedError

