"""Lightweight registry: @register(name, description) over bench functions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from .harness import Report


@dataclass
class BenchSpec:
    name: str
    description: str
    fn: Callable[..., Report]


_REGISTRY: Dict[str, BenchSpec] = {}


def register(name: str, description: str = ""):
    def deco(fn: Callable[..., Report]) -> Callable[..., Report]:
        _REGISTRY[name] = BenchSpec(name=name, description=description, fn=fn)
        return fn
    return deco


def all_benches() -> Dict[str, BenchSpec]:
    return dict(_REGISTRY)


def get(name: str):
    return _REGISTRY.get(name)
