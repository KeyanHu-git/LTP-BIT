from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class EvalExportSpec:
    module: str
    name: str


class EvalExportRegistry:
    def __init__(self, name: str):
        self.name = name
        self._module_dict: dict[str, EvalExportSpec] = {}

    @property
    def module_dict(self) -> Mapping[str, EvalExportSpec]:
        return MappingProxyType(self._module_dict)

    def register_module(self, name: str, module: str) -> None:
        if name in self._module_dict:
            old = self._module_dict[name]
            raise KeyError(f"{name} is already registered in {self.name}: {old.module}.{old.name}")
        self._module_dict[name] = EvalExportSpec(module=module, name=name)

    def get(self, name: str) -> Any:
        spec = self._module_dict.get(name)
        if spec is None:
            raise KeyError(f"{name} is not registered in {self.name}")
        module = import_module(f"{__package__}.{spec.module}")
        return getattr(module, spec.name)


EVAL_EXPORTS = EvalExportRegistry("raev2_eval_exports")
_REGISTERED = False


def _read_module_exports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            is_target = any(isinstance(target, ast.Name) and target.id == "__eval_exports__" for target in node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            is_target = isinstance(node.target, ast.Name) and node.target.id == "__eval_exports__"
            value = node.value
        else:
            continue
        if not is_target:
            continue
        if value is None:
            return []
        value = ast.literal_eval(value)
        if not isinstance(value, (list, tuple)) or not all(isinstance(x, str) for x in value):
            raise TypeError(f"{path}: __eval_exports__ must be a list/tuple of strings.")
        return list(value)
    return []


def register_all_modules() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    root = Path(__file__).resolve().parent
    for path in sorted(root.glob("*.py")):
        if path.name in {"__init__.py", "registry.py"} or path.name.startswith("_"):
            continue
        for name in _read_module_exports(path):
            EVAL_EXPORTS.register_module(name, path.stem)
    _REGISTERED = True
