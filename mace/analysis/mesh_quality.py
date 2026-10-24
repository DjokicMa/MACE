"""Mesh quality presets for every surface the 3D view draws (1.1.6).

One notion of "how fine" for the pore surfaces (Analysis › Porosity), the
cube isosurfaces (density, spin, potential, orbitals, differences) and the
other mesh-like overlays: Draft (screening many materials), Standard (the
look before presets existed, the default), High and Publication (dense
triangulation for a figure; seconds and more memory). The pore surfaces have
one tier below Draft, Preview (a coarse grid, smoothed more): they are the
heaviest surfaces the view draws (two walls each, repeated over every cell of
a supercell), so the GUI draws them at Preview by default (Settings ›
Viewport › Pore surface mesh quality). Each preset is a set of numeric knobs
the GUI shows under Advanced; a knob given by hand wins over the preset,
inside the hard limits below, so no request can hang the page.
The page's copy of these numbers is static/js/viewport/mesh_quality.js
(tests/test_mesh_quality.py keeps the two in step).
"""
from typing import Any, Dict, Mapping, Optional

PRESETS = ("draft", "standard", "high", "publication")
DEFAULT = "standard"
# the pore surfaces' presets: one tier below Draft (pores only)
PORE_PRESETS = ("preview",) + PRESETS

# The pore surfaces (porosity._render_fields / pore_meshes):
#   grid           render grid points per cell axis, at most
#   upsample       the render grid may be this many times finer than the analysis grid
#                  (the distance field is exact; only the regions' labels are the grid's)
#   smooth         Gaussian smoothing of the render field, in render-grid voxels
#   max_triangles  the surfaces' triangle budget (a coarser grid for the channels beyond it)
PORES: Dict[str, Dict[str, float]] = {
    "preview": {"grid": 24, "upsample": 1, "smooth": 1.4, "max_triangles": 60_000},
    "draft": {"grid": 40, "upsample": 1, "smooth": 0.7, "max_triangles": 150_000},
    "standard": {"grid": 72, "upsample": 1, "smooth": 0.7, "max_triangles": 600_000},
    "high": {"grid": 96, "upsample": 2, "smooth": 0.8, "max_triangles": 1_200_000},
    "publication": {"grid": 128, "upsample": 3, "smooth": 0.9, "max_triangles": 2_400_000},
}
PORE_LIMITS = {"grid": (16, 160), "upsample": (1, 4), "smooth": (0.0, 2.0), "max_triangles": (20_000, 3_000_000)}

# The cube grids sent to the 3D view (gui/volume.py decimate):
#   points    grid points, at most (a larger cube is strided down to it); None: the
#             Settings' Volume quality (low/medium/high), as before presets
#   upsample  a grid smaller than ``points`` is refined this many times by cubic
#             interpolation (a smoother, denser isosurface; the values in between are
#             interpolated, never new data)
VOLUME: Dict[str, Dict[str, Any]] = {
    "draft": {"points": 250_000, "upsample": 1},
    "standard": {"points": None, "upsample": 1},
    "high": {"points": 2_000_000, "upsample": 1},
    "publication": {"points": 8_000_000, "upsample": 2},
}
VOLUME_LIMITS = {"points": (50_000, 8_000_000), "upsample": (1, 3)}
# the Settings' Volume quality (the Standard preset's grid)
LEGACY_POINTS = {"low": 250_000, "medium": 1_000_000, "high": 2_000_000}

# A CRYSTAL23 ORBITALS grid (gui/orbitals.py): the reader's spacing per preset
ORBITAL = {"draft": "preview", "standard": "normal", "high": "fine", "publication": "fine"}


def preset(name: Optional[str], table: Optional[Mapping[str, Any]] = None) -> str:
    """A known preset name (of ``table``, default every surface's; an unknown or empty one is Standard)."""
    return name if name in (PRESETS if table is None else table) else DEFAULT


def _clamp(v, lo, hi, cast):
    try:
        x = cast(v)
    except (TypeError, ValueError):
        raise ValueError(f"not a number: {v!r}") from None
    if x != x:
        raise ValueError("not a number: NaN")
    return min(max(x, cast(lo)), cast(hi))


def _knobs(table, limits, spec: Optional[Mapping[str, Any]]):
    spec = spec or {}
    name = preset(spec.get("quality"), table)
    out = dict(table[name])
    for k, (lo, hi) in limits.items():
        v = spec.get(k)
        if v is None or v == "":
            continue
        cast = float if isinstance(lo, float) else int
        out[k] = _clamp(v, lo, hi, cast)
    custom = any(out[k] != table[name][k] for k in limits)
    return {"quality": name, "custom": custom, **out}


def pores(spec: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The pore surfaces' knobs for ``{"quality": preset, <knob>: value ...}`` (None: Standard)."""
    return _knobs(PORES, PORE_LIMITS, spec)


def volume(spec: Optional[Mapping[str, Any]] = None, legacy: str = "medium") -> Dict[str, Any]:
    """A cube grid's knobs; ``legacy``: the Settings' Volume quality, the Standard preset's grid."""
    out = _knobs(VOLUME, VOLUME_LIMITS, spec)
    if out["points"] is None:
        out["points"] = LEGACY_POINTS.get(legacy, LEGACY_POINTS["medium"])
    return out
