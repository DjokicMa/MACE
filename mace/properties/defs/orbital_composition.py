"""Orbital character of the frontier orbitals of an ORBITALS run, from its
Molden files as the GUI's orbital viewer reads them (mace/orbitals:
OrbitalRun, the coefficients C and the analytic overlap S(k) of its
self-check).

For each k point MACE can use and each spin channel, the HOMO and the LUMO
(the extractor's: mace/orbitals KData.homo) are split by Mulliken analysis,
w_mu = Re(c_mu* (S c)_mu), summed by atom shell ("O p") and by element,
and averaged over the orbital's degenerate set (one member of a degenerate
set is any mix of them). The weights of an orbital add up to 1. The
scalars are of the HOMO and LUMO the extractor stores
(orbitals_homo_band: the highest HOMO over the k points and spins, the
LUMO at its k point and spin). Files the extractor does not check (over
its size or basis limits) are left out here too.
"""
from __future__ import annotations

from typing import Any, Dict, List

from mace.properties.registry import Output, Skip, derived

L_NAMES = "spdfghi"


def _labels(kd) -> List[str]:
    out = [""] * kd.basis.nbf
    for sh in kd.basis.shells:
        sym = kd.md.symbols[sh["atom"]].strip().capitalize()
        for i in range(sh["off"], sh["off"] + sh["n"]):
            out[i] = f"{sym} {L_NAMES[sh['l']]}"
    return out


def mulliken(kd, spin: str, n: int) -> Dict[str, Any]:
    """The Mulliken split of orbital ``n`` (1-based in its channel) of ``spin``, averaged over its
    degenerate set: {levels, energy_au, by_shell, by_element, total}."""
    import numpy as np
    ns = kd.degenerate_set(spin, n)
    lv = kd.levels(spin)
    cols = [lv[i - 1].column for i in ns]
    C = kd.C[:, cols]
    w = np.real(np.conj(C) * (kd.S @ C)).sum(axis=1) / len(cols)
    by_shell: Dict[str, float] = {}
    for lab, v in zip(_labels(kd), w):
        by_shell[lab] = by_shell.get(lab, 0.0) + float(v)
    by_el: Dict[str, float] = {}
    for lab, v in by_shell.items():
        el = lab.split()[0]
        by_el[el] = by_el.get(el, 0.0) + v
    rnd = lambda d: {k: round(v, 5) for k, v in sorted(d.items(), key=lambda t: -t[1])}   # noqa: E731
    return {"levels": ns, "energy_au": float(lv[n - 1].energy), "total": round(float(w.sum()), 5),
            "by_shell": rnd(by_shell), "by_element": rnd(by_el)}


def composition(output_file, max_bytes: int, max_nbf: int) -> Dict[str, Any]:
    """{kpoints: [{index, label, frac, spins: [{spin, homo, lumo}]}], best: (k index, spin)} of an ORBITALS
    run (the .out and the Molden files beside it)."""
    import contextlib
    import io
    from mace.orbitals.dataset import OrbitalRun
    with contextlib.redirect_stdout(io.StringIO()):
        run = OrbitalRun.from_path(output_file)
    if run.out is not None and (run.out.nao or 0) > max_nbf:
        raise Skip(f"{run.out.nao} basis functions: over the self-check's limit ({max_nbf}), so no overlap matrix")
    kps: List[Dict[str, Any]] = []
    best = None
    skipped = []
    for e in run.kpoints():
        if not e.usable:
            skipped.append(f"{e.label}: {e.reason}")
            continue
        if sum(p.stat().st_size for p in e.files.values()) > max_bytes:
            skipped.append(f"{e.label}: files over {max_bytes // (1024 * 1024)} MB")
            continue
        kd = run.load(e.index, check=True)
        if kd.S is None:
            skipped.append(f"{e.label}: no overlap matrix")
            continue
        rec = {"index": e.index, "label": e.label, "frac": [float(x) for x in e.frac], "status": kd.status,
               "spins": []}
        for sp in kd.spins:
            h = kd.homo(sp)
            if h is None:
                continue
            s = {"spin": sp, "homo": mulliken(kd, sp, h)}
            if h < len(kd.levels(sp)):
                s["lumo"] = mulliken(kd, sp, h + 1)
            rec["spins"].append(s)
            # the extractor's rule (orbitals_homo_band): the highest HOMO over k points and spins
            if best is None or s["homo"]["energy_au"] > best[0]:
                best = (s["homo"]["energy_au"], e.index, sp)
        kps.append(rec)
    if not kps:
        raise Skip("no k point with Molden files MACE can use" + (f" ({'; '.join(skipped[:3])})" if skipped else ""))
    out: Dict[str, Any] = {"method": "Mulliken, w = Re(c* S c), averaged over each degenerate set",
                           "kpoints": kps}
    if skipped:
        out["skipped"] = skipped
    if best is not None:
        out["frontier"] = {"k_index": best[1], "spin": best[2]}
    return out


@derived("orbital_composition", calc_types=("ORBITALS",), category="orbitals", group="Orbital character",
         requires=("orbitals_homo_band",), version="1",
         source="the .molden files via mace/orbitals (OrbitalRun: coefficients and the overlap S(k) of its "
                "self-check)",
         outputs=(Output("homo_character", "category", "text", label="HOMO character",
                         description="atom shell with the largest Mulliken share of the HOMO (orbitals_homo_band)"),
                  Output("homo_character_fraction", "dimensionless", label="HOMO character share", precision=3,
                         description="its Mulliken share of the HOMO"),
                  Output("lumo_character", "category", "text", label="LUMO character",
                         description="atom shell with the largest Mulliken share of the LUMO (at the HOMO's k "
                                     "point and spin)"),
                  Output("lumo_character_fraction", "dimensionless", label="LUMO character share", precision=3,
                         description="its Mulliken share of the LUMO"),
                  Output("orbital_composition", "JSON", "json", blob=True, label="HOMO/LUMO composition",
                         description="per k point and spin channel: the HOMO's and LUMO's Mulliken shares by atom "
                                     "shell and by element (degenerate sets averaged)")))
def _orbital_composition(ctx):
    """Mulliken composition of the HOMO and LUMO of an ORBITALS run."""
    from mace.utils.property_extractor import CrystalPropertyExtractor as X
    try:
        comp = composition(ctx.output_file, X.ORBITALS_INDEX_MAX_BYTES, X.ORBITALS_CHECK_MAX_NBF)
    except Skip:
        raise
    except (OSError, ValueError) as e:
        raise Skip(f"the Molden files could not be read: {e}")
    vals: Dict[str, Any] = {"orbital_composition": comp}
    fr = comp.get("frontier")
    if fr:
        k = next(k for k in comp["kpoints"] if k["index"] == fr["k_index"])
        s = next(s for s in k["spins"] if s["spin"] == fr["spin"])
        for which in ("homo", "lumo"):
            if which in s and s[which]["by_shell"]:
                name, share = next(iter(s[which]["by_shell"].items()))
                vals[f"{which}_character"] = name
                vals[f"{which}_character_fraction"] = share
    return vals
