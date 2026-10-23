"""Band-edge effective masses of a BAND run: the automatic masses of
``mace tau`` and the GUI's Relaxation Time dialog (mace/transport/mass.py
auto_masses), stored with the run so they are in the database too.

These are the accurate masses, the ones to use. The extractor's
real_*_effective_mass is a quick estimate (the curvature of one band along
the path's direction at the edge), and the DOSS run's dos_effective_mass_*
a rough one (fitted to CRYSTAL's smoothed DOS).

For each carrier (electrons at the CBM, holes at the VBM): the conductivity
mass m_c (along the band path through the edge; no transport axis is
chosen here) and the density-of-states mass m_d (from the path's directions
through the edge, or from the DOSS.DAT of the same parent when the path
cannot give it), in units of the electron mass, with where each came from.
A metal (no gap) or a molecule stores nothing.
"""
from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Any, Dict, Optional

from mace.properties.registry import Output, Sibling, Skip, derived

CARRIERS = ("electron", "hole")


def _band_file(output_file: Path) -> Optional[Path]:
    """The run's BAND.DAT (the extractor's rule), else its own BAND fort.25."""
    from mace.utils.property_extractor import CrystalPropertyExtractor
    dat, _ = CrystalPropertyExtractor._dat_file_for(Path(output_file), "BAND")
    if dat is not None:
        return dat
    f25 = Path(output_file).with_suffix(".f25")
    if f25.is_file():
        from mace.transport import mass
        if mass._kind_of(f25) == "BAND":
            return f25
    return None


def _summary(am: Dict[str, Any]) -> Dict[str, Any]:
    """What is kept of mass.auto_masses' record: the masses, their sources and
    warnings, and the edge each fit used (file names, not paths)."""
    out: Dict[str, Any] = {"m_c": am["m_star"], "m_d": am["m_dos"], "sources": am["sources"],
                           "warnings": am["warnings"], "window_meV": am.get("window_meV")}
    b = am.get("band")
    if b:
        out["band"] = {"file": Path(b["file"]).name, "edge_eV": b.get("edge_eV"), "at": b.get("at"),
                       "bands": b.get("bands"), "valleys": b.get("valleys"),
                       "directions": [{k: d.get(k) for k in ("band", "spin", "direction", "mass", "r2", "points")}
                                      for d in b.get("directions", [])]}
    d = am.get("dos")
    if d:
        out["dos"] = {"file": Path(d["file"]).name, "m_d": d.get("m_d"), "r2": d.get("r2"), "points": d.get("points")}
    return out


def masses(band: Path, dos: Optional[Path], dim: int) -> Dict[str, Dict[str, Any]]:
    """{carrier: summary} of the carriers MACE can fit (the GUI's numbers)."""
    from mace.transport import mass
    out = {}
    for carrier in CARRIERS:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                am = mass.auto_masses(carrier, None, dim, band=band, dos=dos)
        except (mass.MassError, OSError, ValueError) as e:
            # mace tau's message names its own flags: keep what the files said
            msg = str(e)
            if msg.startswith("m* is required"):
                msg = (msg.split("; tried ", 1)[1] if "; tried " in msg else
                       "the band path gives no mass at the edge (flat bands, or too few k points near it)")
            out[carrier] = {"error": msg}
            continue
        out[carrier] = _summary(am)
    return out


@derived("effective_masses", calc_types=("BAND",), category="band_structure", group="Effective masses",
         siblings=(Sibling("BAND", find=_band_file, description="the run's BAND.DAT (or BAND fort.25)"),
                   Sibling("DOSS", required=False,
                           description="the DOSS.DAT of the same parent: the DOS mass when the path gives none")),
         version="1",
         outputs=(Output("effective_mass_electron_mc", "m_e",
                         label="Effective mass m<sub>c</sub>, electrons (accurate)", precision=4,
                         description="accurate conductivity mass of electrons at the CBM (mace tau's fit: every "
                                     "degenerate band and equivalent valley, several directions, an energy "
                                     "window, R² checked). Use this one; real_electron_effective_mass is a quick "
                                     "estimate"),
                  Output("effective_mass_electron_md", "m_e",
                         label="Effective mass m<sub>d</sub>, electrons (accurate)", precision=4,
                         description="accurate density-of-states mass of electrons at the CBM (per valley; every "
                                     "degenerate band, R² checked). Use this one; dos_effective_mass_electron is "
                                     "a rough value from the smoothed DOS"),
                  Output("effective_mass_hole_mc", "m_e", label="Effective mass m<sub>c</sub>, holes (accurate)",
                         precision=4,
                         description="accurate conductivity mass of holes at the VBM (mace tau's fit: every "
                                     "degenerate band and equivalent valley, several directions, an energy "
                                     "window, R² checked). Use this one; real_hole_effective_mass is a quick "
                                     "estimate"),
                  Output("effective_mass_hole_md", "m_e", label="Effective mass m<sub>d</sub>, holes (accurate)",
                         precision=4,
                         description="accurate density-of-states mass of holes at the VBM (per valley; every "
                                     "degenerate band, R² checked). Use this one; dos_effective_mass_hole is a "
                                     "rough value from the smoothed DOS"),
                  Output("effective_masses", "JSON", "json", blob=True, label="Effective masses, accurate (details)",
                         description="per carrier: m_c, m_d, where each came from, warnings, the band edge "
                                     "(k point, bands, valleys) and every direction fitted")),
         source="BAND.DAT (+ DOSS.DAT) via mace.transport.mass.auto_masses (mace tau --m-star auto)")
def _effective_masses(ctx):
    """Band-edge effective masses (mace tau's automatic masses)."""
    dim = ctx.dimensionality or 3
    if dim == 0:
        raise Skip("a molecule has no band edges")
    got = masses(ctx.sibling("BAND"), ctx.sibling("DOSS"), dim)
    vals: Dict[str, Any] = {}
    for carrier, r in got.items():
        if "error" in r:
            continue
        vals[f"effective_mass_{carrier}_mc"] = r["m_c"]
        vals[f"effective_mass_{carrier}_md"] = r["m_d"]
    if not vals:
        raise Skip("no band edge to fit: " + "; ".join(f"{c}: {r['error']}" for c, r in got.items()))
    vals["effective_masses"] = {"dimensionality": dim, **got}
    return vals
