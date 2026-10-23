#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mechanical-property decks for CRYSTAL23: elastic, piezoelectric and
photoelastic tensors, and equations of state
---------------------------------------------------------------------------
The geometry-block modules of CRYSTAL23 manual chapters 12 and 13:

  ELASTCON   second-order elastic constants (13.3, p. 290)
  ELAPIEZO   elastic + direct/converse piezoelectric constants (13.6, p. 298)
  PIEZOCON   direct piezoelectric constants, Berry phase (13.4, p. 296)
  PIEZOCP    direct piezoelectric constants, CPHF/KS (13.5, p. 298)
  PHOTOELA   photoelastic (Pockels) and piezo-optic tensors (13.7, p. 301)
  EOS        E(V) scan fitted to four equations of state (ch. 12, p. 282)
  QHA        quasi-harmonic thermal properties: frequencies at several
             volumes, V(T), K(T), thermal expansion (ch. 9, p. 253)
  CPHF       coupled-perturbed HF/KS: polarizability, dielectric tensor,
             hyperpolarizabilities (ch. 10.2, p. 270)

Each keyword is the last record of the geometry input and opens a block
closed by its own END (p. 288: "Each of these keywords opens a block which
must be terminated with an END keyword"). This module holds the settings
model (one dict per deck, the CRYSTAL sub-keywords as snake_case keys), the
validation against the manual, the interactive questions, the writer of
the block, and the parent refusals. ``mace opt2d12`` (choices 4-7,
``--calc-type ELASTIC|EOS|QHA|CPHF``) writes the decks through it.

Only sub-keywords that differ from CRYSTAL's own defaults are written, so
a deck with every default is the manual's minimal block
("ELASTCON" / "END"). The defaults written in the comments and in
``MODES`` are the manual's bracketed ones.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, TextIO, Tuple

# The calculation types opt2d12 knows for these modules: ELASTIC carries the
# keyword (ELASTCON, ELAPIEZO, PIEZOCON, PIEZOCP, PHOTOELA) in
# elastic_settings["mode"]; EOS, QHA and CPHF are their own types.
ELASTIC_CALC_TYPES = ("ELASTIC", "EOS", "QHA", "CPHF")
# calculation types whose block is their own keyword (not elastic_settings["mode"])
OWN_TYPES = ("EOS", "QHA", "CPHF")

# The SCF energy threshold these modules set (manual p. 291, 296, 299, 302,
# 284: "SCF energy convergence threshold is set to 10^-8"; table p. 127).
# An explicit TOLDEE in block 3 overrides it, so MACE never writes a looser
# one than this into such a deck.
MODULE_TOLDEE = 8

_DIM_NAMES = {3: "3D crystal", 2: "slab (2D)", 1: "polymer (1D)", 0: "molecule (0D)"}
_DIM_OF_KEYWORD = {"CRYSTAL": 3, "SLAB": 2, "POLYMER": 1, "HELIX": 1, "MOLECULE": 0}

#: the modules, their manual section and page, the strain step default,
#: the periodicities they take, and the sub-keywords each one reads
MODES: Dict[str, Dict[str, Any]] = {
    "ELASTCON": {
        "title": "Elastic constants (ELASTCON)",
        "what": "second-order elastic constants, compliance tensor, seismic velocities",
        "manual": "CRYSTAL23 manual 13.3, p. 290",
        "step": 0.01, "dims": (1, 2, 3), "pcrystal_only": False,
        "keys": ("numderiv", "stepsize", "deform", "clampion", "print", "restart", "preoptgeom",
                 "seismdir", "awesome", "pressure", "presseos", "nuchess", "hessnum2", "modepart",
                 "hessstep", "toldeg", "toldex", "toldee", "maxcycle", "notrustr", "trustradius",
                 "maxtradius"),
    },
    "ELAPIEZO": {
        "title": "Elastic + piezoelectric constants (ELAPIEZO)",
        "what": "elastic constants plus direct (e) and converse (d) piezoelectric tensors",
        "manual": "CRYSTAL23 manual 13.6, p. 298",
        "step": 0.015, "dims": (1, 2, 3), "pcrystal_only": True,
        "keys": ("numderiv", "stepsize", "deform", "clampion", "print", "restart", "preoptgeom",
                 "nuchess", "hessnum2", "modepart", "hessstep",
                 "toldeg", "toldex", "toldee", "maxcycle", "notrustr", "trustradius", "maxtradius"),
    },
    "PIEZOCON": {
        "title": "Piezoelectric constants, Berry phase (PIEZOCON)",
        "what": "direct piezoelectric tensor e from the Berry phase",
        "manual": "CRYSTAL23 manual 13.4, p. 296",
        "step": 0.015, "dims": (1, 2, 3), "pcrystal_only": True,
        "keys": ("numderiv", "stepsize", "deform", "clampion", "print", "restart", "preoptgeom",
                 "nuchess", "hessnum2", "modepart", "hessstep",
                 "toldeg", "toldex", "toldee", "maxcycle", "notrustr", "trustradius", "maxtradius"),
    },
    "PIEZOCP": {
        "title": "Piezoelectric constants, CPHF/KS (PIEZOCP)",
        "what": "direct piezoelectric tensor e, analytical (CPHF/KS + internal strain)",
        "manual": "CRYSTAL23 manual 13.5, p. 298",
        "step": None, "dims": (3,), "pcrystal_only": False,
        "keys": ("hessnum2", "tolalpha"),
    },
    "PHOTOELA": {
        "title": "Photoelastic constants (PHOTOELA)",
        "what": "elasto-optic (Pockels) and piezo-optic tensors from CPHF/KS dielectric tensors",
        "manual": "CRYSTAL23 manual 13.7, p. 301",
        "step": 0.015, "dims": (1, 2, 3), "pcrystal_only": False,
        "keys": ("numderiv", "stepsize", "deform", "clampion", "print", "restart", "preoptgeom",
                 "tolalpha", "fmixing", "dynamic", "anderson", "broyden",
                 "toldeg", "toldex", "toldee", "maxcycle", "notrustr", "trustradius", "maxtradius"),
    },
    "EOS": {
        "title": "Equation of state (EOS)",
        "what": "E(V) scan with V-constrained optimizations, fitted to Murnaghan, Birch-Murnaghan, "
                "Poirier-Tarantola and Vinet",
        "manual": "CRYSTAL23 manual ch. 12, p. 284",
        "step": None, "dims": (3,), "pcrystal_only": False,
        "keys": ("range", "vrange", "prange", "print", "restart", "preoptgeom",
                 "toldeg", "toldex", "toldee", "maxcycle", "notrustr", "trustradius", "maxtradius"),
    },
    "QHA": {
        "title": "Quasi-harmonic approximation (QHA)",
        "what": "phonons at several volumes: V(T), bulk modulus K(T), thermal expansion, Cp-Cv",
        "manual": "CRYSTAL23 manual 9.2, p. 253",
        # "QHA generalized to 1D and 2D systems" (manual p. 14); measured on CRYSTAL23 for a polymer
        # (trans-polyacetylene) and a slab (h-BN): the volume scan ran with CVOLOPT. A molecule stops:
        # "ELASTIC OPTION NOT ALLOWED FOR MOLECULES"
        "step": None, "dims": (1, 2, 3), "pcrystal_only": False,
        "keys": ("qstep", "points", "vrange", "temperat", "preoptgeom", "restart", "neglefre", "scelphono"),
    },
    "CPHF": {
        "title": "Dielectric response, CPHF/KS (CPHF)",
        "what": "polarizability and dielectric tensor (static or at wavelengths), first and second "
                "hyperpolarizabilities (THIRD, FOURTH)",
        "manual": "CRYSTAL23 manual 10.2, p. 270",
        "step": None, "dims": (0, 1, 2, 3), "pcrystal_only": False,
        "keys": ("tolalpha", "maxcycle", "fmixing", "nodiis", "diisallk", "histdiis", "prtdiis", "anderson",
                 "broyden", "seledir", "cphf_dynamic", "damping", "third", "fourth", "toludik"),
    },
}

ELASTIC_MODES = ("ELASTCON", "ELAPIEZO", "PIEZOCON", "PIEZOCP", "PHOTOELA")

#: CRYSTAL's defaults (written nowhere: a value equal to one is not written)
DEFAULTS: Dict[str, Any] = {
    "numderiv": 3,                 # all strain modules [3]
    "toldeg": 0.0003, "toldex": 0.00062, "toldee": 8,   # module optimizations
    "tolalpha": 4,                 # PHOTOELA CPHF/KS [4]
    "fmixing": 60,                 # PHOTOELA [IPMIX=60]
    "range": (0.92, 1.08, 6),      # EOS [0.92 1.08 6]
    "qstep": 3.0,                  # QHA STEP [3 %]
    "points": 4,                   # QHA POINTS [4]
    "temperat": (50, 10.0, 300.0), # QHA TEMPERAT [50 temperatures, 10-300 K]
}

_VOIGT = ("xx", "yy", "zz", "yz", "xz", "xy")
# the strains a slab / polymer can take: elements with a non-periodic
# direction "are null by definition" (p. 288)
_ACTIVE_STRAINS = {3: (0, 1, 2, 3, 4, 5), 2: (0, 1, 5), 1: (0,)}

_BOOL_KEYWORDS = (("clampion", "CLAMPION"), ("print", "PRINT"), ("preoptgeom", "PREOPTGEOM"),
                  ("awesome", "AWESOME"), ("nuchess", "NUCHESS"), ("hessnum2", "HESSNUM2"),
                  ("modepart", "MODEPART"), ("anderson", "ANDERSON"), ("notrustr", "NOTRUSTR"),
                  ("nodiis", "NODIIS"), ("diisallk", "DIISALLK"), ("prtdiis", "PRTDIIS"),
                  ("third", "THIRD"), ("fourth", "FOURTH"))


def functional_family(name: Optional[str]) -> Optional[str]:
    """LDA, GGA, HYBRID, ... of a MACE functional name (d12_constants.FUNCTIONAL_CATEGORIES)."""
    if not name:
        return None
    base = str(name).upper().replace("-D3", "").replace("-D2", "")
    try:
        from d12_constants import FUNCTIONAL_CATEGORIES
    except Exception:                           # pragma: no cover
        return None
    for fam, info in FUNCTIONAL_CATEGORIES.items():
        if base in [str(f).upper() for f in info.get("functionals", [])]:
            return fam
    return "HF" if base in ("RHF", "UHF", "HF", "HF3C", "HFSOL3C") else None


# The modules that solve the CPHF/CPKS equations: the dielectric response of
# CPHF, of every strained cell of PHOTOELA, and PIEZOCP's piezoelectric tensor.
CPKS_MODES = ("CPHF", "PHOTOELA", "PIEZOCP")

# The functionals CRYSTAL23 can use for CPKS (manual p. 140, "Availability of
# XC functionals"): none of the meta-GGAs or meta-GGA hybrids, and of the
# LDA/GGA ones only this subset, by MACE name and CRYSTAL23 keyword, upper
# case, any -D3 left out. The 3c composites are not in that table: HF3C and
# HFSOL3C are Hartree-Fock (CPHF); PBESOL03C and HSESOL3C are PBEsol0 and
# HSEsol with semi-classical corrections (pp. 162-163), both listed; PBEH3C,
# HSE3C and B973C re-parametrize the PBEh, HSE and B97 forms (pp. 159-161),
# none of which is listed.
CPKS_LDA_GGA = ("SVWN", "BLYP", "PBE", "PBEXC", "PBESOL", "PBESOLXC", "SOGGA", "SOGGAXC", "WCGGA")
CPKS_GLOBAL_HYBRIDS = ("B3LYP", "PBE0", "PBESOL0", "B1WC", "WC1LYP", "B97H")
CPKS_RANGE_SEPARATED = ("RSHXLDA", "WB97", "WB97X", "LC-BLYP", "SC-BLYP", "CAM-B3LYP", "HSE06", "HSESOL",
                        "HISS", "LC-WPBE", "LC-WPBESOL", "LC-WBLYP", "LC-PBE")
CPKS_COMPOSITES = ("PBESOL03C", "HSESOL3C")
CPKS_HF = ("RHF", "UHF", "HF", "HF3C", "HFSOL3C")
CPKS_FUNCTIONALS = frozenset(CPKS_LDA_GGA + CPKS_GLOBAL_HYBRIDS + CPKS_RANGE_SEPARATED
                             + CPKS_COMPOSITES + CPKS_HF)
# a functional of EXCHANGE/CORRELAT records: the LDA/GGA column of that table
CPKS_EXCHANGE = frozenset({"LDA", "BECKE", "PBE", "PBESOL", "SOGGA", "WCGGA"})
CPKS_CORRELATION = frozenset({"VWN", "LYP", "PBE", "PBESOL", "WCGGA"})
CPKS_SUPPORTED_TEXT = ("Hartree-Fock (RHF, UHF, HF3C, HFSOL3C); SVWN, BLYP, PBE, PBEsol, SOGGA, WCGGA; "
                       "B3LYP, PBE0, PBEsol0, B1WC, WC1LYP, B97H; RSHXLDA, wB97, wB97X, LC-BLYP, SC-BLYP, "
                       "CAM-B3LYP, HSE06, HSEsol, HISS, LC-wPBE, LC-wPBEsol, LC-wBLYP, LC-PBE; "
                       "PBESOL03C, HSESOL3C")


def _functional_key(name: Optional[str]) -> str:
    base = str(name or "").strip().upper()
    for suffix in ("-D3", "-D2"):
        if base.endswith(suffix):
            base = base[:-len(suffix)]
    return base


def cpks_problem(mode: str, functional: Optional[str], custom_records: Optional[Sequence[str]] = None,
                 method: Optional[str] = None) -> Optional[str]:
    """Why CRYSTAL23 cannot run ``mode``'s CPHF/CPKS with this functional, or None."""
    if mode not in CPKS_MODES:
        return None
    key = _functional_key(functional)
    if not key and str(method or "").upper() == "HF":
        return None
    if key == "CUSTOM-XC" and custom_records:
        # EXCHANGE/CORRELAT of that column, HYBRID, or a listed functional the
        # records modify (HYBRID on B3LYP); anything else (NONLOCAL, the
        # range-separation records) is not known to work
        recs = [str(r).strip().upper() for r in custom_records]
        ok, xc, i = True, False, 0
        while i < len(recs):
            r = recs[i]
            value = recs[i + 1].split()[0] if i + 1 < len(recs) and recs[i + 1].split() else ""
            if r in ("EXCHANGE", "CORRELAT"):
                ok = ok and value in (CPKS_EXCHANGE if r == "EXCHANGE" else CPKS_CORRELATION)
                xc, i = True, i + 2
            elif r == "HYBRID":
                i += 2
            else:
                ok, xc, i = ok and r in CPKS_FUNCTIONALS, True, i + 1
        if ok and xc:
            return None
        try:
            from d12_constants import describe_custom_functional
            shown = describe_custom_functional(custom_records)
        except Exception:                       # pragma: no cover
            shown = "the parent's own EXCHANGE/CORRELAT records"
        return (f"{mode} solves the CPHF/CPKS equations, which CRYSTAL23 implements only for "
                f"{CPKS_SUPPORTED_TEXT} and for EXCHANGE/CORRELAT records of LDA, BECKE, PBE, PBESOL, SOGGA, "
                f"WCGGA / VWN, LYP, PBE, PBESOL, WCGGA (manual p. 140); this deck's functional is {shown}")
    if key in CPKS_FUNCTIONALS:
        return None
    shown = functional or "not known to MACE"
    return (f"{mode} solves the CPHF/CPKS equations, which CRYSTAL23 implements only for {CPKS_SUPPORTED_TEXT} "
            f"(manual p. 140: no meta-GGA or meta-GGA hybrid, and only that subset of the LDA/GGA functionals "
            f"and hybrids); this deck's functional is {shown}")


class ElasticSettingsError(ValueError):
    """A setting the manual does not allow for this module (the message says why)."""


def dimensionality_of(keyword: Optional[str]) -> int:
    """3, 2, 1 or 0 for CRYSTAL / SLAB / POLYMER (HELIX) / MOLECULE."""
    return _DIM_OF_KEYWORD.get(str(keyword or "CRYSTAL").upper(), 3)


def mode_of(calc_type: str, settings: Optional[Dict[str, Any]]) -> str:
    """The CRYSTAL keyword of a deck: EOS, or elastic_settings' mode (ELASTCON by default)."""
    if str(calc_type).upper() in OWN_TYPES:
        return str(calc_type).upper()
    mode = str((settings or {}).get("mode") or "ELASTCON").upper()
    return mode


def default_settings(mode: str = "ELASTCON") -> Dict[str, Any]:
    """The settings of a deck with every CRYSTAL default: {"mode": mode}."""
    mode = mode.upper()
    if mode not in MODES:
        raise ElasticSettingsError(f"unknown module {mode}: one of {', '.join(MODES)}")
    return {"mode": mode}


def _num(v, kind, name):
    try:
        return kind(v)
    except (TypeError, ValueError):
        raise ElasticSettingsError(f"{name} must be a{'n integer' if kind is int else ' number'}, not {v!r}")


def normalize(settings: Optional[Dict[str, Any]], calc_type: str = "ELASTIC") -> Dict[str, Any]:
    """A copy of ``settings`` with keys lower-cased, values typed, and the mode set."""
    raw = {str(k).lower(): v for k, v in (settings or {}).items()}
    mode = mode_of(calc_type, raw)
    out: Dict[str, Any] = {"mode": mode}
    for k, v in raw.items():
        if k == "mode" or v is None or v is False:
            continue
        if k in ("numderiv", "toldee", "maxcycle", "tolalpha", "fmixing", "dynamic", "points", "histdiis",
                 "toludik"):
            out[k] = _num(v, int, k.upper())
        elif k in ("stepsize", "toldeg", "toldex", "trustradius", "maxtradius", "hessstep",
                   "pressure", "presseos", "qstep", "damping"):
            out[k] = _num(v, float, k.upper())
        elif k == "deform":
            vals = v.split() if isinstance(v, str) else list(v)
            out[k] = [_num(x, int, "DEFORM") for x in vals]
        elif k == "seismdir":
            out[k] = [[_num(x, float, "SEISMDIR") for x in (d.split() if isinstance(d, str) else d)] for d in v]
        elif k in ("range", "vrange", "prange"):
            vals = v.split() if isinstance(v, str) else list(v)
            if len(vals) != 3:
                raise ElasticSettingsError(f"{k.upper()} takes three values (min max points)")
            out[k] = (_num(vals[0], float, k.upper()), _num(vals[1], float, k.upper()),
                      _num(vals[2], int, k.upper()))
        elif k == "temperat":
            vals = v.split() if isinstance(v, str) else list(v)
            if len(vals) != 3:
                raise ElasticSettingsError("TEMPERAT takes N TMIN TMAX")
            out[k] = (_num(vals[0], int, "TEMPERAT"), _num(vals[1], float, "TEMPERAT"), _num(vals[2], float, "TEMPERAT"))
        elif k == "scelphono":
            vals = v.split() if isinstance(v, str) else [x for row in v for x in (row if isinstance(row, (list, tuple)) else [row])]
            ints = [_num(x, int, "SCELPHONO") for x in vals]
            if len(ints) == 3:
                ints = [ints[0], 0, 0, 0, ints[1], 0, 0, 0, ints[2]]
            if len(ints) != 9:
                raise ElasticSettingsError("SCELPHONO is 3 integers (a diagonal supercell) or a 3x3 matrix")
            out[k] = ints
        elif k in ("seledir", "cphf_dynamic"):
            vals = v.split() if isinstance(v, str) else list(v)
            if k == "seledir":
                if len(vals) != 3:
                    raise ElasticSettingsError("SELEDIR takes NX NY NZ")
                out[k] = [_num(x, int, "SELEDIR") for x in vals]
            else:
                if len(vals) not in (2, 3):
                    raise ElasticSettingsError("CPHF DYNAMIC takes NSTEPS and one wavelength (nm), or "
                                               "NSTEPS FIRST LAST")
                out[k] = [_num(vals[0], int, "NSTEPS")] + [_num(x, float, "DYNAMIC") for x in vals[1:]]
        elif k == "neglefre":
            vals = v.split() if isinstance(v, str) else list(v)
            out[k] = [_num(x, float, "NEGLEFRE") for x in vals]
        elif k == "broyden":
            vals = v.split() if isinstance(v, str) else list(v)
            if len(vals) != 3:
                raise ElasticSettingsError("BROYDEN takes W0 IMIX ISTART")
            out[k] = (_num(vals[0], float, "W0"), _num(vals[1], int, "IMIX"), _num(vals[2], int, "ISTART"))
        elif k == "restart":
            r = str(v).upper() if not isinstance(v, bool) else "RESTART"
            if r not in ("RESTART", "RESTART2"):
                raise ElasticSettingsError("restart is RESTART (partial) or RESTART2 (complete)")
            out[k] = r
        elif k in dict(_BOOL_KEYWORDS):
            out[k] = bool(v)
        else:
            out[k] = v
    return out


def validate(settings: Dict[str, Any], dimensionality: Any = "CRYSTAL", two_component: bool = False,
             scf_executable: Optional[str] = None) -> List[str]:
    """Check ``settings`` (normalized) against the manual for this system.

    Raises ElasticSettingsError for what CRYSTAL would reject or what the
    manual excludes; returns the warnings (strings) for what is allowed but
    worth a note. ``dimensionality``: the geometry keyword or 0-3.
    """
    mode = settings.get("mode", "ELASTCON")
    info = MODES.get(mode)
    if info is None:
        raise ElasticSettingsError(f"unknown module {mode}: one of {', '.join(MODES)}")
    dim = dimensionality if isinstance(dimensionality, int) else dimensionality_of(dimensionality)
    warnings: List[str] = []

    if two_component and mode == "CPHF":
        raise ElasticSettingsError("CPHF/CPKS is not in the keywords a two-component SCF accepts (CRYSTAL23 "
                                   "manual ch. 6, p. 166): compute the response of the one-component parent")
    if two_component:
        raise ElasticSettingsError(
            f"{mode} needs analytical energy gradients and geometry optimizations of strained cells "
            "(manual ch. 12-13), and CRYSTAL23 stops a two-component SCF that optimizes "
            "(measured on HPCC: \"TWO-COMPONENT SCF IS NOT COMPATIBLE WITH OPTGEOM\"). "
            "Compute it from the scalar-relativistic (one-component) parent")
    if dim not in info["dims"]:
        allowed = ", ".join(_DIM_NAMES[d] for d in info["dims"])
        why = {"EOS": "an E(V) curve and the bulk modulus are defined for a 3D volume",
               "QHA": "the quasi-harmonic volume scan needs a periodic cell (CRYSTAL23 stops a molecule: "
                      "ELASTIC OPTION NOT ALLOWED FOR MOLECULES)",
               "PIEZOCP": "the analytical CPHF/KS piezoelectric tensor is implemented for 3D crystals (manual 13.5)"}
        raise ElasticSettingsError(
            f"{mode} is not available for a {_DIM_NAMES.get(dim, dim)} (allowed: {allowed})"
            + (f": {why[mode]}" if mode in why else
               ": a molecule has no lattice to strain"))

    if mode == "QHA" and dim in (1, 2) and settings.get("scelphono") is not None:
        raise ElasticSettingsError(
            "QHA with SCELPHONO is written for a 3D crystal only: a slab or polymer supercell takes a 2x2 "
            "or 1x1 matrix (manual p. 73), and that QHA has not been run on CRYSTAL23")
    unknown = sorted(k for k in settings if k != "mode" and k not in info["keys"])
    if unknown:
        raise ElasticSettingsError(
            f"{mode} does not read {', '.join(k.upper() for k in unknown)} ({info['manual']})")

    nd = settings.get("numderiv")
    if nd is not None and nd < 2:
        raise ElasticSettingsError("NUMDERIV counts the points including the central one: at least 2")
    st = settings.get("stepsize")
    if st is not None:
        if st <= 0:
            raise ElasticSettingsError("STEPSIZE is a strain step: positive")
        if st > 0.05:
            warnings.append(f"STEPSIZE {st} is a large strain (default {info['step']}); "
                            "the numerical derivative leaves the linear regime")
    df = settings.get("deform")
    if df is not None:
        if len(df) != 6 or any(x not in (0, 1) for x in df):
            raise ElasticSettingsError("DEFORM is six 0/1 flags for xx yy zz yz xz xy")
        if not any(df):
            raise ElasticSettingsError("DEFORM with no active deformation computes nothing")
        bad = [_VOIGT[i] for i, x in enumerate(df) if x and i not in _ACTIVE_STRAINS.get(dim, ())]
        if bad:
            raise ElasticSettingsError(
                f"DEFORM activates {', '.join(bad)}, which involve a non-periodic direction of a "
                f"{_DIM_NAMES[dim]} (null by definition, manual p. 288)")
    if settings.get("pressure") is not None and settings.get("presseos") is not None:
        raise ElasticSettingsError("PRESSURE (pre-optimize at P) and PRESSEOS (the structure is at P) exclude each other")
    if (settings.get("pressure") is not None or settings.get("presseos") is not None) and dim != 3:
        raise ElasticSettingsError("elastic constants under pressure are for 3D crystals")
    if settings.get("pressure") is not None and not settings.get("preoptgeom"):
        warnings.append("PRESSURE: added PREOPTGEOM, the constant-pressure pre-optimization the manual pairs it with (p. 294)")
        settings["preoptgeom"] = True
    if settings.get("modepart") or settings.get("hessstep") is not None or settings.get("hessnum2"):
        if mode != "PIEZOCP" and not settings.get("nuchess"):
            raise ElasticSettingsError("HESSNUM2, MODEPART and HESSSTEP refine NUCHESS (manual p. 295): turn NUCHESS on")
    if settings.get("nuchess") and settings.get("clampion"):
        raise ElasticSettingsError("NUCHESS computes the nuclear-relaxation term that CLAMPION leaves out: choose one")
    for k in ("range", "vrange", "prange"):
        r = settings.get(k)
        if r is None:
            continue
        lo, hi, n = r
        if n < 1:
            raise ElasticSettingsError(f"{k.upper()}: at least one point")
        if lo >= hi:
            raise ElasticSettingsError(f"{k.upper()}: the minimum {lo} is not below the maximum {hi}")
        if k == "range":
            if lo <= 0:
                raise ElasticSettingsError("RANGE takes volume ratios to the equilibrium volume (positive)")
            if not lo < 1.0 < hi:
                warnings.append(f"RANGE {lo}-{hi} does not bracket the equilibrium volume (1.0)")
            if n < 4:
                warnings.append("an E(V) fit of a third-order EOS needs at least 4 points; use 6 or more")
    pts = settings.get("points")
    if pts is not None and pts not in (4, 7, 13):
        raise ElasticSettingsError("QHA POINTS is 4, 7 or 13 volumes (equidistant at s, s/2, s/4 %; manual p. 253)")
    qs = settings.get("qstep")
    if qs is not None and not 0 < qs <= 10:
        raise ElasticSettingsError("QHA STEP is the volume step in % (default 3)")
    tp = settings.get("temperat")
    if tp is not None:
        n, t1, t2 = tp
        if n < 50:
            raise ElasticSettingsError("QHA TEMPERAT needs at least 50 temperatures (thermal expansion is a "
                                       "numerical derivative; manual p. 254)")
        if not 0 <= t1 < t2:
            raise ElasticSettingsError("QHA TEMPERAT: TMIN below TMAX, both in K")
    sc = settings.get("scelphono")
    if sc is not None:
        m = [sc[0:3], sc[3:6], sc[6:9]]
        det = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
               + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
        if det <= 0:
            raise ElasticSettingsError("SCELPHONO: the supercell matrix must have a positive determinant")
    nf = settings.get("neglefre")
    if nf is not None and not nf:
        raise ElasticSettingsError("NEGLEFRE lists the frequencies to leave out")
    sd = settings.get("seismdir")
    if sd is not None:
        if not sd or any(len(d) != 3 for d in sd):
            raise ElasticSettingsError("SEISMDIR takes Cartesian directions of three components each")
        if any(all(abs(x) < 1e-12 for x in d) for d in sd):
            raise ElasticSettingsError("SEISMDIR: a zero vector is not a direction")
    cd = settings.get("cphf_dynamic")
    if cd is not None:
        n, *w = cd
        if n < 1 or (n == 1) != (len(w) == 1) or any(x <= 0 for x in w):
            raise ElasticSettingsError("CPHF DYNAMIC: NSTEPS 1 with one wavelength, or NSTEPS > 1 with the first "
                                       "and last wavelength (nm; manual p. 272)")
    dyn = settings.get("dynamic")
    if dyn is not None and dyn <= 0:
        raise ElasticSettingsError("DYNAMIC takes the wavelength in nm (positive)")
    if settings.get("anderson") and settings.get("broyden"):
        raise ElasticSettingsError("ANDERSON and BROYDEN are two accelerators: choose one")

    if info["pcrystal_only"] and scf_executable and "MPP" in str(scf_executable).upper():
        raise ElasticSettingsError(
            f"{mode} runs with Pcrystal only: CRYSTAL stops if MPPcrystal is selected "
            f"({info['manual']}); the site config's [crystal] scf is {scf_executable}")
    if settings.get("restart"):
        info_file = info_file_name(mode)
        warnings.append(f"{settings['restart']} reads {info_file} of the run it continues: the job script "
                        f"stages <JOB>.{info_file} next to the deck (a run copies its own back there; "
                        "opt2d12 puts the parent run's there under the new deck's name)")
    if mode in ("PIEZOCON", "ELAPIEZO"):
        warnings.append("the Berry-phase polarization is defined for insulators: a metallic parent has no piezoelectric tensor")
    return warnings


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return repr(v) if abs(v) >= 1e-4 or v == 0 else f"{v:.6g}"
    return str(v)


def block_lines(settings: Dict[str, Any]) -> List[str]:
    """The module's block, keyword to its END (settings normalized and valid)."""
    mode = settings.get("mode", "ELASTCON")
    info = MODES[mode]
    lines = [mode]
    s = settings

    def has(k):
        return k in s and s[k] is not None and s[k] is not False

    if has("range") and tuple(s["range"]) != DEFAULTS["range"]:
        lo, hi, n = s["range"]
        lines += ["RANGE", f"{_fmt(float(lo))} {_fmt(float(hi))} {int(n)}"]
    for k in ("vrange", "prange"):
        if has(k):
            lo, hi, n = s[k]
            lines += [k.upper(), f"{_fmt(float(lo))} {_fmt(float(hi))} {int(n)}"]
    if has("qstep") and abs(s["qstep"] - DEFAULTS["qstep"]) > 1e-12:
        lines += ["STEP", _fmt(float(s["qstep"]))]
    if has("points") and s["points"] != DEFAULTS["points"]:
        lines += ["POINTS", str(int(s["points"]))]
    if has("temperat") and tuple(s["temperat"]) != DEFAULTS["temperat"]:
        n, t1, t2 = s["temperat"]
        lines += ["TEMPERAT", f"{int(n)} {_fmt(float(t1))} {_fmt(float(t2))}"]
    if has("neglefre"):
        lines += ["NEGLEFRE", str(len(s["neglefre"])), " ".join(_fmt(float(x)) for x in s["neglefre"])]
    if has("numderiv") and s["numderiv"] != DEFAULTS["numderiv"]:
        lines += ["NUMDERIV", str(int(s["numderiv"]))]
    if has("stepsize") and (info["step"] is None or abs(s["stepsize"] - info["step"]) > 1e-12):
        lines += ["STEPSIZE", _fmt(float(s["stepsize"]))]
    if has("deform"):
        lines += ["DEFORM", " ".join(str(int(x)) for x in s["deform"])]
    if has("pressure"):
        lines += ["PRESSURE", _fmt(float(s["pressure"]))]
    if has("presseos"):
        lines += ["PRESSEOS", _fmt(float(s["presseos"]))]
    for k in ("clampion", "preoptgeom", "nuchess", "hessnum2", "modepart"):
        if has(k):
            lines.append(k.upper())
    if has("hessstep"):
        lines += ["HESSSTEP", _fmt(float(s["hessstep"]))]
    # 4 is PHOTOELA's default (p. 302); PIEZOCP prints none, so its value is always written
    if mode == "CPHF":
        for k in ("nodiis", "diisallk", "prtdiis"):
            if has(k):
                lines.append(k.upper())
        if has("histdiis"):
            lines += ["HISTDIIS", str(int(s["histdiis"]))]
        if has("seledir"):
            lines += ["SELEDIR", " ".join(str(int(x)) for x in s["seledir"])]
        if has("cphf_dynamic"):
            n, *w = s["cphf_dynamic"]
            lines += ["DYNAMIC", str(int(n)), " ".join(_fmt(float(x)) for x in w)]
        if has("damping"):
            lines += ["DAMPING", _fmt(float(s["damping"]))]
        for k in ("third", "fourth"):
            if has(k):
                lines.append(k.upper())
        if has("toludik") and s["toludik"] != 6:
            lines += ["TOLUDIK", str(int(s["toludik"]))]
        if has("tolalpha") and s["tolalpha"] != 3:                 # CPHF's default is 3 (p. 271)
            lines += ["TOLALPHA", str(int(s["tolalpha"]))]
        if has("fmixing"):
            lines += ["FMIXING", str(int(s["fmixing"]))]
        if has("maxcycle") and s["maxcycle"] != 100:
            lines += ["MAXCYCLE", str(int(s["maxcycle"]))]
        if has("anderson"):
            lines.append("ANDERSON")
        if has("broyden"):
            w0, imix, istart = s["broyden"]
            lines += ["BROYDEN", f"{_fmt(float(w0))} {int(imix)} {int(istart)}"]
        lines.append("END")
        return lines
    if has("tolalpha") and (mode == "PIEZOCP" or s["tolalpha"] != DEFAULTS["tolalpha"]):
        lines += ["TOLALPHA", str(int(s["tolalpha"]))]
    if has("fmixing") and s["fmixing"] != DEFAULTS["fmixing"]:
        lines += ["FMIXING", str(int(s["fmixing"]))]
    if has("dynamic"):
        lines += ["DYNAMIC", str(int(s["dynamic"]))]
    if has("anderson"):
        lines.append("ANDERSON")
    if has("broyden"):
        w0, imix, istart = s["broyden"]
        lines += ["BROYDEN", f"{_fmt(float(w0))} {int(imix)} {int(istart)}"]
    if has("seismdir"):
        lines += ["SEISMDIR", str(len(s["seismdir"]))]
        lines += [" ".join(_fmt(float(x)) for x in d) for d in s["seismdir"]]
    if has("awesome"):
        lines.append("AWESOME")
    if has("toldeg") and abs(s["toldeg"] - DEFAULTS["toldeg"]) > 1e-12:
        lines += ["TOLDEG", _fmt(float(s["toldeg"]))]
    if has("toldex") and abs(s["toldex"] - DEFAULTS["toldex"]) > 1e-12:
        lines += ["TOLDEX", _fmt(float(s["toldex"]))]
    if has("toldee") and s["toldee"] != DEFAULTS["toldee"]:
        lines += ["TOLDEE", str(int(s["toldee"]))]
    if has("maxcycle"):
        lines += ["MAXCYCLE", str(int(s["maxcycle"]))]
    if has("notrustr"):
        lines.append("NOTRUSTR")
    for k in ("trustradius", "maxtradius"):
        if has(k):
            lines += [k.upper(), _fmt(float(s[k]))]
    if has("print"):
        lines.append("PRINT")
    if has("restart"):
        lines.append(s["restart"])
    lines.append("END")
    return lines


def geometry_lines(settings: Dict[str, Any]) -> List[str]:
    """Geometry records the module needs before its block: SCELPHONO for a
    QHA with phonon dispersion (the manual's corundum example, p. 255)."""
    sc = settings.get("scelphono")
    if not sc:
        return []
    return ["SCELPHONO"] + [" ".join(str(int(x)) for x in sc[r * 3:r * 3 + 3]) for r in range(3)]


def write_elastic_section(f: TextIO, settings: Optional[Dict[str, Any]], calc_type: str = "ELASTIC") -> None:
    """Write the module block after the coordinates (it is the last record of the geometry input)."""
    s = normalize(settings, calc_type)
    for line in geometry_lines(s) + block_lines(s):
        f.write(line + "\n")


#: QHA computes frequencies at each volume: FREQCALC's own threshold (table p. 127)
QHA_TOLDEE = 10


def scf_tolerances_for(tolerances: Optional[Dict[str, Any]], log: Optional[Callable[[str], None]] = None,
                       floor: int = MODULE_TOLDEE) -> Dict[str, Any]:
    """The parent's SCF tolerances with TOLDEE no looser than the module's 10^-8.

    Follow-up decks keep the parent's values; the one exception is a TOLDEE
    below 8, which in block 3 would loosen the threshold these modules set
    for themselves (manual: "SCF energy convergence threshold is set to
    10^-8. To modify it, see keyword TOLDEE"). TOLINTEG and the rest stay.
    """
    out = dict(tolerances or {})
    cur = out.get("TOLDEE")
    try:
        cur_i = int(cur) if cur is not None else None
    except (TypeError, ValueError):
        cur_i = None
    if cur_i is None or cur_i < floor:
        if log is not None:
            what = ("the frequency threshold QHA runs at, manual p. 127" if floor == QHA_TOLDEE else
                    "the elastic/EOS modules' own threshold, manual p. 291")
            log(f"  SCF TOLDEE {floor} ({what}; the parent's {cur if cur is not None else 'default'} "
                "would loosen it)")
        out["TOLDEE"] = floor
    return out


def describe(settings: Dict[str, Any]) -> str:
    """One line: 'ELASTCON: NUMDERIV 5, STEPSIZE 0.02' (or 'ELASTCON: CRYSTAL defaults')."""
    lines = block_lines(settings)[1:-1]
    return f"{settings.get('mode')}: " + (" ".join(lines) if lines else "CRYSTAL defaults")


# --------------------------------------------------------------- interactive
def configure_elastic(calc_type: str = "ELASTIC", dimensionality: Any = "CRYSTAL",
                      current: Optional[Dict[str, Any]] = None,
                      ask: Optional[Callable[[str], str]] = None) -> Dict[str, Any]:
    """The interactive questions; blank answers keep CRYSTAL's defaults.

    ``ask(prompt)`` reads an answer (default: the back-aware reader of
    menu_nav, input() without it).
    """
    if ask is None:
        try:
            from menu_nav import nav_read as _nav_read
            ask = lambda p: _nav_read(p)            # noqa: E731
        except Exception:                           # pragma: no cover
            ask = input
    dim = dimensionality if isinstance(dimensionality, int) else dimensionality_of(dimensionality)
    cur = normalize(current, calc_type) if current else {}

    def q(prompt, default=""):
        a = (ask(f"{prompt}{f' [{default}]' if default != '' else ''}: ") or "").strip()
        return a if a else str(default)

    def yn(prompt, default=False):
        d = "y" if default else "n"
        a = q(f"{prompt} (y/n)", d).lower()
        return a.startswith("y")

    if str(calc_type).upper() in OWN_TYPES:
        mode = str(calc_type).upper()
    else:
        choices = [m for m in ELASTIC_MODES if dim in MODES[m]["dims"]]
        print("\nMechanical property module:")
        for i, m in enumerate(choices, 1):
            print(f"  {i} - {MODES[m]['title']}: {MODES[m]['what']}")
        default_i = choices.index(cur.get("mode")) + 1 if cur.get("mode") in choices else 1
        while True:
            a = q("Module", default_i)
            if a.isdigit() and 1 <= int(a) <= len(choices):
                mode = choices[int(a) - 1]
                break
            if a.upper() in choices:
                mode = a.upper()
                break
            print(f"  Choose 1-{len(choices)}")
    info = MODES[mode]
    s: Dict[str, Any] = {"mode": mode}
    print(f"\n{info['title']} ({info['manual']}). Blank keeps CRYSTAL's default.")
    if mode == "EOS":
        lo, hi, n = cur.get("range", DEFAULTS["range"])
        a = q("Volume range: min ratio, max ratio, points", f"{lo} {hi} {n}")
        s["range"] = a.split()
        a = q("Pressure range printed at the end: Pmin Pmax points in GPa (blank: none)")
        if a:
            s["prange"] = a.split()
    elif mode == "QHA":
        a = q("Volume step in % (STEP)", cur.get("qstep", DEFAULTS["qstep"]))
        s["qstep"] = a
        a = q("Volumes: 4, 7 or 13 (POINTS)", cur.get("points", DEFAULTS["points"]))
        s["points"] = a
        n, t1, t2 = cur.get("temperat", DEFAULTS["temperat"])
        a = q("Temperatures: N TMIN TMAX in K (TEMPERAT, N >= 50)", f"{n} {t1:g} {t2:g}")
        s["temperat"] = a.split()
        a = q("Phonon supercell for dispersion, e.g. 2 2 2 (SCELPHONO; blank: Gamma only)", "")
        if a:
            s["scelphono"] = a.split()
    elif mode == "CPHF":
        if yn("First hyperpolarizability (THIRD)", bool(cur.get("third"))):
            s["third"] = True
        if yn("Second hyperpolarizability (FOURTH)", bool(cur.get("fourth"))):
            s["fourth"] = True
        a = q("Wavelengths: NSTEPS and one wavelength, or NSTEPS FIRST LAST in nm (DYNAMIC; blank: static)", "")
        if a:
            s["cphf_dynamic"] = a.split()
    elif mode == "PIEZOCP":
        if yn("Two-point formula for the Hessian (HESSNUM2)", bool(cur.get("hessnum2"))):
            s["hessnum2"] = True
        a = q("CPHF/KS tolerance TOLALPHA", cur.get("tolalpha", ""))
        if a:
            s["tolalpha"] = a
    else:
        a = q("Points per strain incl. the central one (NUMDERIV)", cur.get("numderiv", DEFAULTS["numderiv"]))
        s["numderiv"] = a
        a = q("Strain step (STEPSIZE)", cur.get("stepsize", info["step"]))
        s["stepsize"] = a
        if mode in ("ELASTCON", "ELAPIEZO", "PIEZOCON"):
            print("  Nuclear relaxation: 1 - optimize atoms in each strained cell (default), "
                  "2 - internal-strain Hessian (NUCHESS, recommended, p. 295), 3 - clamped ions (CLAMPION)")
            a = q("  Choice", 2 if cur.get("nuchess") else 3 if cur.get("clampion") else 1)
            if a == "2":
                s["nuchess"] = True
                if yn("  Two-point Hessian formula (HESSNUM2)", True):
                    s["hessnum2"] = True
            elif a == "3":
                s["clampion"] = True
        elif yn("Clamped-ion constants only (CLAMPION)", bool(cur.get("clampion"))):
            s["clampion"] = True
        if mode == "ELASTCON" and dim == 3:
            a = q("Hydrostatic pressure in GPa (blank: none)", cur.get("pressure", ""))
            if a:
                s["pressure"] = a
                s["preoptgeom"] = True
        if mode == "PHOTOELA":
            a = q("Wavelength in nm for frequency-dependent dielectric tensors (DYNAMIC; blank: static)",
                  cur.get("dynamic", ""))
            if a:
                s["dynamic"] = a
    if yn("Pre-optimize the structure first (PREOPTGEOM)", bool(cur.get("preoptgeom") or s.get("preoptgeom"))):
        s["preoptgeom"] = True
    if mode not in ("PIEZOCP", "QHA", "CPHF") and yn("Extended printing (PRINT, recommended)", True if not cur else bool(cur.get("print"))):
        s["print"] = True
    s = normalize(s, calc_type)
    for w in validate(s, dim):
        print(f"  Note: {w}")
    print(f"  {describe(s)}")
    return s


# ---------------------------------------------------------------- opt2d12 hook
def _config_settings(config_file: Optional[str]) -> Optional[Dict[str, Any]]:
    if not config_file:
        return None
    try:
        with open(config_file) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    try:
        from d12_config import unwrap_d12_config
        data = unwrap_d12_config(data)
    except Exception:
        pass
    v = data.get("elastic_settings") if isinstance(data, dict) else None
    return v if isinstance(v, dict) else None


#: settings given on the command line (``mace opt2d12 --elastic-settings``),
#: set by CRYSTALOptToD12.main before the files are processed
REQUESTED: Optional[Dict[str, Any]] = None


def load_requested(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """--elastic-settings: a JSON object, or the path of a JSON file holding one."""
    if not text:
        return None
    s = str(text).strip()
    if not s.startswith("{"):
        with open(s) as fh:
            s = fh.read()
    v = json.loads(s)
    if not isinstance(v, dict):
        raise ElasticSettingsError("--elastic-settings is a JSON object")
    return v


def parent_refusals(parent_out: str) -> List[Tuple[str, str]]:
    """Why parent_out is not the finished OPT or SP run a module deck starts from (as for
    phonopy's displaced supercells: terminated normally, last SCF converged, an optimization
    converged, no FREQCALC run), as (code, message). A missing file is opt2d12's to report; a
    module run's own output is the parent of its next deck at its reference cell (opt2d12
    checks that one), and a RESTART continues such a run."""
    if not parent_out or not os.path.isfile(parent_out):
        return []
    import sys
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from mace.phonopy_bridge.parent import run_refusals
    from mace.utils.calc_detection import is_frequency_output, is_optimization_output, mechanics_output_type
    with open(parent_out, errors="replace") as fh:
        text = fh.read()
    if mechanics_output_type(text):
        return []
    return run_refusals(text, is_optimization_output(text), is_frequency_output(text),
                        why="strained cells of an unrelaxed parent give wrong constants", freq_hint="")


def parent_refusal(parent_out: str) -> Optional[List[str]]:
    """parent_refusals as the lines opt2d12 prints to stderr, else None."""
    reasons = parent_refusals(parent_out)
    if not reasons:
        return None
    return [f"Refusing {os.path.basename(parent_out)}:"] + [f"  [{code}] {msg}" for code, msg in reasons]


def scf_executable() -> Optional[str]:
    """The site's [executables] scf, when the site config names one."""
    try:
        from mace.submission import site as _site
        return (_site.load().crystal_exe or {}).get("scf")
    except Exception:
        return None


def info_file_name(mode: str) -> str:
    """The restart file of a module: EOSINFO.DAT (EOS, QHA; manual pp. 253, 284)
    or ELASINFO.DAT (the elastic modules, p. 290)."""
    return "EOSINFO.DAT" if str(mode).upper() in ("EOS", "QHA") else "ELASINFO.DAT"


def _parent_scratch(parent_out: str) -> Optional[str]:
    """The scratch folder the parent run's job script ran CRYSTAL in, when it
    can be told from here (mace.recovery.opt_restart), else None."""
    try:
        from mace.recovery.opt_restart import job_name, job_scratch_dir
    except Exception:                           # pragma: no cover - a flat copy without mace
        return None
    folder = os.path.dirname(os.path.abspath(parent_out))
    stem = os.path.splitext(os.path.basename(parent_out))[0]
    try:
        with open(os.path.join(folder, f"{stem}.sh"), errors="replace") as fh:
            script = fh.read()
    except OSError:
        script = ""
    try:
        d = job_scratch_dir(script, job_name(script) or stem, submit_dir=folder)
    except Exception:
        return None
    return str(d) if d else None


def restart_info_plan(settings: Optional[Dict[str, Any]], parent_out: Optional[str],
                      new_deck: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """(problem, source, destination) of the restart file a RESTART/RESTART2 deck reads.

    The job script stages <JOB>.ELASINFO.DAT / <JOB>.EOSINFO.DAT next to the
    deck into its scratch folder, and a new deck has a new job name. Its file
    is the parent run's: the copy a finished run's job script left next to
    its output, else the one in the parent job's scratch folder (a run killed
    at its walltime copies nothing back). (None, None, None) when there is
    nothing to stage or the new deck's file is already there.
    """
    if not settings or not settings.get("restart"):
        return None, None, None
    info = info_file_name(settings.get("mode", ""))
    dst = f"{os.path.splitext(new_deck)[0]}.{info}"
    if os.path.isfile(dst) and os.path.getsize(dst) > 0:
        return None, None, None
    tried = []
    if parent_out:
        stem = os.path.splitext(parent_out)[0]
        tried.append(f"{stem}.{info}")
        scratch = _parent_scratch(parent_out)
        if scratch:
            tried.append(os.path.join(scratch, info))
        for src in tried:
            if os.path.isfile(src) and os.path.getsize(src) > 0:
                return None, src, dst
    where = ", ".join(tried) if tried else "no parent run"
    return (f"{settings['restart']} reads {info} of the run it continues, and none was found ({where}; "
            f"the parent job's scratch folder {'is not known here' if len(tried) < 2 else 'has none'}). "
            f"Run this from the folder of the {settings.get('mode')} run being continued, or put its "
            f"{info} next to the new deck as {os.path.basename(dst)}", None, None)


def resolve_for_opt2d12(options: Dict[str, Any], parent_settings: Dict[str, Any],
                        config_file: Optional[str] = None, interactive: bool = False,
                        log: Optional[Callable[[str], None]] = None) -> Optional[str]:
    """Fill options["elastic_settings"] and the SCF TOLDEE for an ELASTIC/EOS deck.

    The settings come from --elastic-settings, else the config file's
    "elastic_settings", else the parent's own (an elastic parent), else the
    questions (interactive) or CRYSTAL's defaults. Returns None when the
    deck can be written, else the reason it cannot.
    """
    say = log or print
    calc_type = str(options.get("calculation_type", "ELASTIC")).upper()
    dim = options.get("dimensionality", parent_settings.get("dimensionality", "CRYSTAL"))
    given = REQUESTED or _config_settings(config_file) or options.get("elastic_settings")
    # the parent deck's own module block (CrystalInputParser, parse_block,
    # without RESTART/RESTART2): a follow-up of the same module keeps it
    parent_es = {k: v for k, v in (parent_settings.get("elastic_settings") or {}).items() if k != "restart"}
    try:
        if given and given.get("restart") and parent_es and \
                str(given.get("mode") or calc_type).upper() == str(parent_es.get("mode")).upper():
            # a restart continues the parent's run: its settings, plus what was asked
            s = normalize({**parent_es, **given}, calc_type)
        elif given:
            s = normalize(given, calc_type)
        elif interactive:
            s = configure_elastic(calc_type, dim, current=parent_es)
        elif parent_es:
            s = normalize(parent_es, calc_type)
        else:
            s = normalize({"mode": calc_type if calc_type in OWN_TYPES else "ELASTCON"}, calc_type)
        from d12_constants import a_calc
        # the mode asked for: normalize() gives an EOS/QHA/CPHF deck its own
        # mode, so a block of another module would otherwise be dropped unseen
        asked = str({str(k).lower(): v for k, v in (given or {}).items()}.get("mode") or s["mode"]).upper()
        if calc_type in OWN_TYPES and asked != calc_type:
            return (f"--calc-type {calc_type} with {a_calc(asked)} block (elastic_settings mode {asked}): use "
                    f"--calc-type {asked if asked in OWN_TYPES else 'ELASTIC'}, or drop the mode")
        if calc_type == "ELASTIC" and s["mode"] in OWN_TYPES:
            return f"{a_calc(s['mode'])} block is --calc-type {s['mode']}"
        why = cpks_problem(s["mode"], options.get("functional"), options.get("custom_functional"),
                           options.get("method"))
        if why:
            return why
        # frequency-dependent CPHF/KS: PHOTOELA DYNAMIC runs the same DYNAMIC CPHF
        if (s["mode"] == "CPHF" and s.get("cphf_dynamic")) or (s["mode"] == "PHOTOELA" and s.get("dynamic")):
            fam = functional_family(options.get("functional"))
            if fam not in ("LDA", "GGA"):
                return (f"{s['mode']} DYNAMIC needs a pure LDA/GGA functional: CRYSTAL stops with Hartree-Fock "
                        f"or a hybrid (manual p. 272-273); this deck's is {options.get('functional') or 'unknown'}")
        warnings = validate(s, dim, two_component=bool(options.get("soc") or options.get("two_component")),
                            scf_executable=scf_executable())
    except ElasticSettingsError as e:
        return str(e)
    for w in warnings:
        say(f"  Note: {w}")
    options["elastic_settings"] = s
    options["tolerances"] = scf_tolerances_for(options.get("tolerances"), log=say,
                                               floor=QHA_TOLDEE if s["mode"] == "QHA" else MODULE_TOLDEE)
    say(f"  {describe(s)}")
    return None


def parse_block(lines: Sequence[str], start: int) -> Tuple[Dict[str, Any], int]:
    """Read a module block from a deck's lines (``lines[start]`` is its keyword).

    Returns (normalized settings, index of the line after its END). Used to
    carry an elastic parent's settings to its child (RESTART2 analysis).
    """
    mode = lines[start].strip().upper()
    s: Dict[str, Any] = {"mode": mode}
    i = start + 1
    one_value = {"NUMDERIV", "STEPSIZE", "PRESSURE", "PRESSEOS", "HESSSTEP", "TOLALPHA", "FMIXING",
                 "DYNAMIC", "TOLDEG", "TOLDEX", "TOLDEE", "MAXCYCLE", "TRUSTRADIUS", "MAXTRADIUS"}
    while i < len(lines):
        k = lines[i].strip().upper()
        if not k:
            i += 1
            continue
        if k == "END":
            return normalize(s, mode if mode in OWN_TYPES else "ELASTIC"), i + 1
        if k == "DYNAMIC" and mode == "CPHF":           # NSTEPS, then the wavelength(s)
            s["cphf_dynamic"] = [lines[i + 1].split()[0], *lines[i + 2].split()]
            i += 3
        elif k in one_value:
            s[k.lower()] = lines[i + 1].split()[0]
            i += 2
        elif k in ("DEFORM", "RANGE", "VRANGE", "PRANGE", "BROYDEN", "TEMPERAT"):
            s[k.lower()] = lines[i + 1].split()
            i += 2
        elif k == "SELEDIR":
            s["seledir"] = lines[i + 1].split()
            i += 2
        elif k == "DAMPING":
            s["damping"] = lines[i + 1].split()[0]
            i += 2
        elif k in ("HISTDIIS", "TOLUDIK"):
            s[k.lower()] = lines[i + 1].split()[0]
            i += 2
        elif k in ("STEP", "POINTS"):
            s["qstep" if k == "STEP" else "points"] = lines[i + 1].split()[0]
            i += 2
        elif k == "NEGLEFRE":
            s["neglefre"] = lines[i + 2].split()
            i += 3
        elif k == "SEISMDIR":
            n = int(lines[i + 1].split()[0])
            s["seismdir"] = [lines[i + 2 + j].split()[:3] for j in range(n)]
            i += 2 + n
        elif k in ("RESTART", "RESTART2"):
            s["restart"] = k
            i += 1
        else:
            s[k.lower()] = True
            i += 1
    raise ElasticSettingsError(f"{mode} block without its END")
