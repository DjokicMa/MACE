"""
MACE Database Module
====================
Material tracking database with workflow isolation support.
"""

from .materials import MaterialDatabase
from .materials_contextual import ContextualMaterialDatabase, get_contextual_database

__all__ = [
    'MaterialDatabase',
    'ContextualMaterialDatabase', 
    'get_contextual_database'
]