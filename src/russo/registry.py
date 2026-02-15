from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable


def import_symbol(path: str) -> Any:
    module_path, _, symbol_name = path.rpartition(".")
    if not module_path:
        raise ValueError(f"Invalid import path: {path}")
    module = importlib.import_module(module_path)
    return getattr(module, symbol_name)


@dataclass
class ComponentFactory:
    name: str
    factory: Callable[..., Any]

    def build(self, **kwargs: Any) -> Any:
        return self.factory(**kwargs)


class ComponentRegistry:
    def __init__(self) -> None:
        self._registry: dict[str, ComponentFactory] = {}

    def register(self, name: str, factory: Callable[..., Any]) -> None:
        if name in self._registry:
            raise ValueError(f"Component '{name}' already registered.")
        self._registry[name] = ComponentFactory(name=name, factory=factory)

    def register_path(self, name: str, import_path: str) -> None:
        factory = import_symbol(import_path)
        self.register(name, factory)

    def get(self, name: str) -> ComponentFactory:
        if name not in self._registry:
            raise KeyError(f"Component '{name}' not found.")
        return self._registry[name]

    def build(self, name: str, **kwargs: Any) -> Any:
        return self.get(name).build(**kwargs)
