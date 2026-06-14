"""Plotter handlers for mace plotting.

Importing this package registers every handler into the registry (its module
runs ``register(...)`` at import time). Later phases add ``cube``, ``freq``,
and ``spectra`` modules here; each is imported below so a single
``import mace.plotting.handlers`` populates the full registry.
"""
from . import legacy  # noqa: F401  (import side-effect: registers band/DOS/structure)

__all__ = ['legacy']
