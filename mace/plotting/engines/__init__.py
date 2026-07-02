"""Plotting engines consumed by ``mace plotting`` handlers.

These modules are the heavy visualization engines for cube volumetrics and
FREQ vibrational modes. This package is their single source of truth — they
ship with mace so ``mace plotting --cube/--freq`` works from any clone.

Both engines are import-safe (all CLI work is under ``if __name__ == '__main__'``)
and depend only on numpy + plotly (scipy / imageio imported lazily; kaleido is
needed for static-image export). Install those into whichever interpreter runs
``mace plotting``.

Modules
-------
crystal_cubeviz_plotly
    ``CubeFile`` reader/classifier + isosurface / slice plotly renderers and the
    cube-arithmetic (difference) path.
vibmode_viewer
    ``Crystal23FreqParser`` + ``VibModeAnimator`` for FREQ normal-mode viewers.

Author: Marcus Djokic
Institution: Michigan State University, Mendoza Group
"""
