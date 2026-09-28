#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spglib dataset access that works on every spglib version MACE allows
--------------------------------------------------------------------
requirements.txt allows spglib >= 1.16. Before 2.5, get_symmetry_dataset
returns a plain dict; from 2.5 it returns a SpglibDataset whose dict-style
access (dataset["rotations"]) is deprecated and will be removed. Wrapping the
result here lets callers use attribute access (dataset.rotations) on both.

Author: Marcus Djokic
Institution: Michigan State University, Mendoza Group
"""

from types import SimpleNamespace


def attribute_dataset(dataset):
    """Return ``dataset`` with attribute access, or None if spglib failed.

    A dict from an older spglib becomes a SimpleNamespace holding the same
    values; anything else (a SpglibDataset, or None) is returned unchanged.
    """
    if isinstance(dataset, dict):
        return SimpleNamespace(**dataset)
    return dataset
