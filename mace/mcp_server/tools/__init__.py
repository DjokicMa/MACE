"""MACE's own MCP tools, one module per area. Each module's ``register(reg)``
adds its tools; the handlers import MACE's modules only when they run."""
from __future__ import annotations

from importlib import import_module

AREAS = ("files", "structure", "setup", "jobs", "results")


def register_all(reg) -> None:
    for area in AREAS:
        import_module(f"{__name__}.{area}").register(reg)
