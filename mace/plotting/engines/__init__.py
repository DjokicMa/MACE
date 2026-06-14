"""Plotting engines consumed by ``mace plotting`` handlers.

These modules are the heavy visualization engines for cube volumetrics and
FREQ vibrational modes. They were developed and validated under
``test/AddedPlottingFunctionalty/`` (gitignored) and **relocated here** so they
ship with the package — a clone has no ``test/`` tree, so importing them from
there would break ``mace plotting --cube/--freq`` everywhere but this machine.

Relocation (not copy): this is the single source of truth. The standalone
diagnostic scripts (``subtract_cubes``, ``diagnose_cube_subtraction``,
``slice_browser_standalone``) were intentionally NOT relocated — the cube engine's
own ``perform_cube_arithmetic`` / ``interpolate_cube_at_vertices`` diff path is
already non-orthogonal-safe, so they are redundant and stay as gitignored
diagnostics.

Both engines are import-safe (all CLI work is under ``if __name__ == '__main__'``)
and depend only on numpy + plotly (scipy / imageio imported lazily). They require
the anaconda interpreter where plotly/scipy/kaleido are installed.

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
