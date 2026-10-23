#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finite-displacement phonons with phonopy: opt2d12's PHONOPY calculation type
---------------------------------------------------------------------------
Choice 8 of opt2d12's calculation menu (from a finished run) and
``mace opt2d12 --calc-type PHONOPY``. The parent's settings go through
opt2d12's own questions (method, basis, SHRINK, tolerances, SCF, spin, the
parent's values as the defaults); this module adds phonopy's questions
(supercell, displacement amplitude, plus/minus displacements, the symmetry
of each displaced cell, the supercell's SHRINK) and writes the set:

    <stem>/<stem>_disp-NNN.d12 + .ext, phonopy_disp.yaml, mace_phonopy.json

phonopy (mace.phonopy_bridge.structure) builds the displaced supercells,
their .ext files (phonopy's own CRYSTAL writer; the job script stages
<job>.ext as fort.34) and phonopy_disp.yaml. Every displaced deck is written
by opt2d12's deck writer, as any other deck of the parent, with an EXTERNAL
geometry and GRADCAL before the final END. <stem> is the name opt2d12 gives
a deck of the parent, with the supercell as its type
(mat_opt_ph222_PBE0-D3_optimized).

The parent must be a finished 3D OPT (OPT END - CONVERGED) or SP run with
its .d12 (mace.phonopy_bridge.parent.refusal_reasons). Then:

    mace submit <stem> --walltime <printed>    (never --track or --progress)
    mace analyze phonopy <stem> [--nac-from FREQ.out]

The completion callback of the last displaced job runs the analysis by
itself when phonopy is installed where the job ends (mace.queue.manager).
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

#: settings given on the command line (mace opt2d12 --dim, --amplitude, --pm,
#: --cry-sym, --keep-shrink, --force), set by CRYSTALOptToD12.main
REQUESTED: Optional[Dict[str, Any]] = None

#: TOLDEE the displaced supercells run at, at least (phonopy's own template: 10)
DEFAULT_TOLDEE = 10
#: phonopy's default displacement (Angstrom)
DEFAULT_AMPLITUDE = 0.01
PM_CHOICES = ("auto", "yes", "no")
#: the suggested supercell makes each lattice vector at least this long (Angstrom)
SUGGESTED_LENGTH = 10.0


class PhonopySettingsError(ValueError):
    """A phonopy setting opt2d12 cannot write a set for."""


from d12_constants import ParentRefused  # noqa: E402  (raised for a parent that cannot seed a set)


# ---------------------------------------------------------------------------
# The parent
# ---------------------------------------------------------------------------

def parent_d12_for(output_file: Optional[str], input_file: Optional[str]) -> Optional[str]:
    """The parent's deck: the one given, else the .d12 of the same stem."""
    if input_file:
        return input_file
    if output_file:
        return os.path.splitext(output_file)[0] + ".d12"
    return None


def parent_refusal(output_file: Optional[str], input_file: Optional[str]) -> Optional[List[str]]:
    """Why this parent cannot seed a displacement set (the lines printed to
    stderr), else None: no finished run's output and its deck, or
    phonopy_bridge.parent.refusal_reasons."""
    from mace.phonopy_bridge import parent as P

    if not output_file:
        return [f"Error: a phonopy set starts from a finished run: give its output with "
                f"--out-file (and its deck {input_file or 'beside it'})"]
    d12 = parent_d12_for(output_file, input_file)
    for f in (output_file, d12):
        if not os.path.isfile(f):
            return [f"Error: {f} not found"]
    try:
        deck, out_text = P.read_parent(Path(d12), Path(output_file))
    except P.DeckSplitError as exc:
        return [f"Error: {d12}: {exc}"]
    reasons = P.refusal_reasons(deck, out_text)
    if reasons:
        return [f"Refusing {os.path.basename(output_file)}:"] + [f"  [{code}] {msg}" for code, msg in reasons]
    return None


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def _cell_lengths(cell: Optional[Sequence[Any]]) -> Optional[List[float]]:
    try:
        return [float(x) for x in list(cell)[:3]]
    except (TypeError, ValueError):
        return None


def suggest_dim(cell: Optional[Sequence[Any]]) -> str:
    """The diagonal supercell whose lattice vectors are each at least
    SUGGESTED_LENGTH long (cell: a b c ... of the parent's primitive cell);
    "2 2 2" when the cell is unknown."""
    abc = _cell_lengths(cell)
    if not abc or min(abc) <= 0:
        return "2 2 2"
    return " ".join(str(max(1, math.ceil(SUGGESTED_LENGTH / x - 1e-9))) for x in abc)


def normalize(settings: Optional[Dict[str, Any]], suggestion: str = "2 2 2") -> Dict[str, Any]:
    """The settings of a set, checked: dim ("N N N" or 9 integers), amplitude
    (Angstrom), pm, cry_sym, keep_shrink, force."""
    s = {str(k).lower(): v for k, v in (settings or {}).items()}
    dim = s.get("dim")
    if dim is None or str(dim).strip() == "":
        dim = suggestion
    if isinstance(dim, (list, tuple)):
        dim = " ".join(str(v) for v in (sum(dim, []) if dim and isinstance(dim[0], list) else dim))
    parts = str(dim).replace(",", " ").split()
    try:
        vals = [int(v) for v in parts]
    except ValueError:
        raise PhonopySettingsError(f'supercell "{dim}": "N N N" or 9 integers')
    if len(vals) not in (3, 9):
        raise PhonopySettingsError(f'supercell "{dim}": "N N N" or 9 integers')
    if len(vals) == 3 and min(vals) < 1:
        raise PhonopySettingsError(f'supercell "{dim}": each N is at least 1')
    out: Dict[str, Any] = {"dim": " ".join(str(v) for v in vals)}
    amp = s.get("amplitude")
    if amp not in (None, ""):
        try:
            amp = float(amp)
        except (TypeError, ValueError):
            raise PhonopySettingsError(f"displacement amplitude {amp!r} is a number in Angstrom")
        if not 0 < amp < 0.2:
            raise PhonopySettingsError("displacement amplitude is in Angstrom, between 0 and 0.2 "
                                       f"(phonopy's default {DEFAULT_AMPLITUDE})")
        out["amplitude"] = amp
    pm = str(s.get("pm") or "auto").lower()
    if pm not in PM_CHOICES:
        raise PhonopySettingsError(f"plus/minus displacements: {', '.join(PM_CHOICES)}")
    out["pm"] = pm
    for k in ("cry_sym", "keep_shrink", "force"):
        if s.get(k):
            out[k] = True
    return out


def dim_tag(dim: str) -> str:
    """The supercell in file names: '222' for 2 2 2 (or its diagonal 3x3 matrix), else
    'm' and the 9 integers (mace.phonopy_bridge.structure.dim_tag without numpy)."""
    vals = [int(v) for v in str(dim).split()]
    if len(vals) == 9:
        if any(vals[i] for i in (1, 2, 3, 5, 6, 7)):
            return "m" + "".join(str(v) for v in vals)
        vals = [vals[0], vals[4], vals[8]]
    return "".join(str(v) for v in vals)


def describe(s: Dict[str, Any]) -> str:
    words = [f"supercell {s['dim']}",
             f"displacement {s.get('amplitude', DEFAULT_AMPLITUDE)} A",
             f"plus/minus {s.get('pm', 'auto')}"]
    if s.get("cry_sym"):
        words.append("symmetry of each displaced cell (CRY_SYM)")
    words.append("the parent's SHRINK" if s.get("keep_shrink") else "SHRINK divided by the supercell")
    return "phonopy: " + ", ".join(words)


def configure_phonopy(current: Optional[Dict[str, Any]] = None, suggestion: str = "2 2 2",
                      ask: Optional[Callable[[str], str]] = None) -> Dict[str, Any]:
    """The interactive questions; a blank answer keeps the default shown."""
    if ask is None:
        try:
            from menu_nav import nav_read as _nav_read
            ask = lambda p: _nav_read(p)            # noqa: E731
        except Exception:                           # pragma: no cover
            ask = input
    cur = dict(current or {})

    def q(prompt, default=""):
        a = (ask(f"{prompt}{f' [{default}]' if default != '' else ''}: ") or "").strip()
        return a if a else str(default)

    def yn(prompt, default=False):
        return q(f"{prompt} (y/n)", "y" if default else "n").lower().startswith("y")

    print("\nFinite-displacement phonons with phonopy: displaced supercells, then "
          "mace analyze phonopy. Blank keeps the default shown.")
    while True:
        try:
            dim = q(f'Supercell, "N N N" or 9 integers (suggested: each vector at least '
                    f'{SUGGESTED_LENGTH:g} A)', cur.get("dim") or suggestion)
            amp = q("Displacement amplitude in Angstrom", cur.get("amplitude", DEFAULT_AMPLITUDE))
            pm = q(f"Plus and minus displacements ({'/'.join(PM_CHOICES)})", cur.get("pm", "auto"))
            s = normalize({"dim": dim, "amplitude": amp, "pm": pm}, suggestion)
            break
        except PhonopySettingsError as e:
            print(f"  {e}")
    if yn("Write each displaced cell's symmetry into its .ext (CRY_SYM: fewer k-points; phonopy "
          "warns some space groups, e.g. R-3m, do not work)", bool(cur.get("cry_sym"))):
        s["cry_sym"] = True
    if yn("Keep the parent's SHRINK in the supercells (default: divide it by the supercell size)",
          bool(cur.get("keep_shrink"))):
        s["keep_shrink"] = True
    if s.get("amplitude") == DEFAULT_AMPLITUDE:
        s.pop("amplitude")
    print(f"  {describe(s)}")
    return s


def _config_settings(config_file: Optional[str]) -> Optional[Dict[str, Any]]:
    if not config_file:
        return None
    import json
    try:
        with open(config_file) as fh:
            data = json.load(fh)
        from d12_config import unwrap_d12_config
        data = unwrap_d12_config(data)
    except Exception:
        return None
    v = data.get("phonopy_settings") if isinstance(data, dict) else None
    return v if isinstance(v, dict) else None


def toldee_floor(tolerances: Optional[Dict[str, Any]], log: Optional[Callable[[str], None]] = None,
                 floor: int = DEFAULT_TOLDEE) -> Dict[str, Any]:
    """The deck's SCF tolerances with TOLDEE no looser than 10 (the forces of
    the displaced supercells; phonopy's own template writes 10); a tighter
    TOLDEE (the parent's, or an answer) is kept."""
    out = dict(tolerances or {})
    cur = out.get("TOLDEE")
    try:
        cur_i = int(cur) if cur is not None else None
    except (TypeError, ValueError):
        cur_i = None
    if cur_i is None or cur_i < floor:
        if log is not None:
            log(f"  SCF TOLDEE {floor} (the forces of the displaced supercells; "
                f"{cur if cur is not None else 'the default'} would loosen them)")
        out["TOLDEE"] = floor
    return out


def resolve_for_opt2d12(options: Dict[str, Any], parent_settings: Dict[str, Any],
                        config_file: Optional[str] = None, interactive: bool = False,
                        log: Optional[Callable[[str], None]] = None) -> Optional[str]:
    """Fill options["phonopy_settings"] and the SCF TOLDEE of a PHONOPY set.

    The settings come from the command line (REQUESTED), else the config
    file's "phonopy_settings", else the questions (interactive), else the
    defaults (the suggested supercell). Returns None when the set can be
    written, else the reason it cannot.
    """
    say = log or print
    if options.get("soc") or options.get("two_component"):
        return ("a two-component (SOC) deck is not written for phonopy's displaced supercells "
                "(a scalar parent is needed)")
    cell = parent_settings.get("primitive_cell") or parent_settings.get("conventional_cell")
    suggestion = suggest_dim(cell)
    given = REQUESTED or _config_settings(config_file) or options.get("phonopy_settings")
    try:
        if given:
            s = normalize(given, suggestion)
        elif interactive:
            s = configure_phonopy(None, suggestion)
        else:
            s = normalize({}, suggestion)
            say(f"  Supercell {s['dim']} (suggested: each lattice vector at least "
                f"{SUGGESTED_LENGTH:g} A; --dim sets another)")
    except PhonopySettingsError as e:
        return str(e)
    options["phonopy_settings"] = s
    options["tolerances"] = toldee_floor(options.get("tolerances"), log=say)
    say(f"  {describe(s)}")
    return None


# ---------------------------------------------------------------------------
# The set
# ---------------------------------------------------------------------------

def unit_mesh(k_points: Any) -> Optional[Tuple[List[int], Optional[int]]]:
    """([ka, kb, kc], ISP or None) of a k_points setting: "IS ISP", "ka kb kc",
    a 3-tuple, or one number."""
    if k_points is None:
        return None
    try:
        if isinstance(k_points, (tuple, list)):
            vals = [int(v) for v in k_points]
            return (vals[:3], None) if len(vals) >= 3 else None
        parts = [int(v) for v in str(k_points).split()]
    except (TypeError, ValueError):
        return None
    if len(parts) == 3:
        return parts, None
    if len(parts) == 2 and parts[0] > 0:
        return [parts[0]] * 3, parts[1]
    if len(parts) == 1 and parts[0] > 0:
        return [parts[0]] * 3, None
    return None


def supercell_k_points(k_points: Any, diag: Sequence[int]) -> Optional[Tuple[Any, str]]:
    """(the supercell's k_points setting, the note) for a diagonal supercell:
    the parent's mesh divided by the supercell size in each direction
    (phonopy_bridge.parent.scaled_mesh). None: no mesh to scale."""
    from mace.phonopy_bridge.parent import scaled_mesh
    unit = unit_mesh(k_points)
    if unit is None:
        return None
    old, isp = unit
    new, new_isp, note = scaled_mesh(old, isp if isp else 2 * max(old), diag)
    if len(set(new)) == 1:
        return f"{new[0]} {new_isp}", note
    return tuple(new), note


def write_set(new_filename: str, output_file: str, input_file: Optional[str], options: Dict[str, Any],
              write_deck: Callable[[str, Dict[str, Any], Dict[str, Any], Any], bool],
              deck_number: Callable[[Any], int], parent_k_points: Any = None,
              unattended: bool = False, confirm: Optional[Callable[[Path], bool]] = None,
              log: Callable[[str], None] = print, warn: Callable[[str], None] = print,
              fail: Callable[[str], None] = print) -> Optional[str]:
    """Write the displacement set; returns its folder, or None (``fail`` said why).

    ``new_filename``: the deck opt2d12 would write (its stem names the set);
    ``write_deck(path, geometry_data, settings, parent_k_points)``: opt2d12's
    deck writer; ``deck_number(z)``: the atomic number opt2d12 writes for an
    atom numbered z in the parent's output (the .ext numbers the same atoms).
    ``confirm(folder)``: asked whether to write into a set folder that is not
    empty (opt2d12's question; never asked when ``unattended``).
    """
    import numpy as np

    from mace import __version__ as mace_version
    from mace.phonopy_bridge import parent as P
    from mace.phonopy_bridge import structure as S
    from mace.phonopy_bridge import writer as W
    from mace.phonopy_bridge.cli import FALLBACK_WALLTIME, INSTALL_HINT, _kpoints, suggest_walltime

    lines = parent_refusal(output_file, input_file)
    if lines:
        raise ParentRefused(lines)
    try:
        import phonopy
    except ImportError:
        fail(INSTALL_HINT)
        return None
    s = options.get("phonopy_settings") or normalize({})
    d12 = parent_d12_for(output_file, input_file)
    deck, out_text = P.read_parent(Path(d12), Path(output_file))
    dim = S.normalize_dim(s["dim"].split())
    diagonal = P.is_diagonal(dim.tolist())
    scale = diagonal and not s.get("keep_shrink")

    setdir = Path(os.path.splitext(new_filename)[0])
    stem = setdir.name
    if "workflow_outputs" in setdir.resolve().parts:
        fail(f"{setdir} is inside a workflow_outputs tree; the workflow engine would pick these "
             "jobs up. Write the set elsewhere (--output-dir).")
        return None
    if setdir.exists() and any(setdir.iterdir()) and not s.get("force"):
        if unattended or confirm is None or not confirm(setdir):
            fail(f"{setdir} is not empty (add --force to write into it)")
            return None

    warns = P.warnings_for(deck, out_text)
    cell, conv = S.read_parent_cell(Path(output_file))
    fold = S.folded_warning(cell)
    if fold:
        warns.append(fold)
    phonon = S.make_phonopy(cell, dim)
    S.generate(phonon, s.get("amplitude"), s.get("pm", "auto"))
    n_super = len(phonon.supercell.numbers)
    conv_deck = [deck_number(c) for c in conv]
    conv_super = S.supercell_conv_numbers(phonon, conv_deck)
    geometry = {"external_geometry": True,
                "coordinates": [{"atom_number": str(c)} for c in S.supercell_conv_numbers(phonon, conv)]}

    settings = dict(options)
    settings["calculation_type"] = "PHONOPY"
    settings["k_points_set"] = False
    edits = ["geometry: EXTERNAL, each displaced cell in <job>.ext (staged as fort.34)"]
    k_parent = parent_k_points
    if scale:
        scaled = supercell_k_points(settings.get("k_points"), [int(v) for v in np.diag(dim)])
        if scaled:
            settings["k_points"], note = scaled
            k_parent = settings["k_points"]
            edits.append(note)
    elif not diagonal and not s.get("keep_shrink"):
        warns.append(("SHRINK_KEPT", "non-diagonal supercell: SHRINK kept as in the parent (it "
                      "cannot be divided per direction); set it in the decks by hand"))
    scf = dict(settings.get("scf_settings") or {})
    if scf.pop("guessp", None):
        edits.append("GUESSP dropped (no density matrix for a displaced supercell)")
    settings["scf_settings"] = scf
    toldee = (settings.get("tolerances") or {}).get("TOLDEE", DEFAULT_TOLDEE)
    edits += [f"TOLDEE {toldee}", "GRADCAL before the final END (the forces)"]

    phonopy_cells = phonon.supercells_with_displacements
    names = W.job_names(stem, len(phonopy_cells))
    setdir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    try:
        if s.get("cry_sym"):
            warns += [("CRY_SYM", m) for m in S.cry_sym_warnings(cell)]
        W.write_ext_files(setdir, names, phonon, conv_super, cry_sym=bool(s.get("cry_sym")))
        written += [setdir / f"{n}.ext" for n in names]
        for name in names:
            path = setdir / f"{name}.d12"
            if not write_deck(str(path), geometry, settings, k_parent):
                return None
            written.append(path)
        if s.get("cry_sym"):
            (setdir / W.CRY_SYM).write_text("")   # phonopy-init --crystal -d reads it too
            written.append(setdir / W.CRY_SYM)
        with W.working_directory(setdir):
            phonon.save(W.DISP_YAML)
        written.append(setdir / W.DISP_YAML)
    finally:
        if len(written) < len(names) * 2 + 1:
            for f in written:
                try:
                    f.unlink()
                except OSError:
                    pass

    first = (setdir / f"{names[0]}.d12").read_text()
    try:
        written_deck = P.split_parent_deck(first)
    except P.DeckSplitError:
        written_deck = deck
    n_cells = int(round(abs(np.linalg.det(dim))))
    kp = _kpoints(first.splitlines())
    nk = kp[0] * kp[1] * kp[2] if kp else None
    if nk:
        walltime, capped = suggest_walltime(nk, n_super)
    else:
        walltime, capped = FALLBACK_WALLTIME, False

    W.write_manifest(setdir, {
        "parent_out": str(Path(output_file).resolve()), "parent_out_sha256": W.sha256(Path(output_file)),
        "parent_d12": str(Path(d12).resolve()), "parent_d12_sha256": W.sha256(Path(d12)),
        "dim": dim.tolist(), "amplitude": s.get("amplitude", DEFAULT_AMPLITUDE),
        "pm": s.get("pm", "auto"), "toldee": int(toldee), "scale_shrink": bool(scale),
        "cry_sym": bool(s.get("cry_sym")), "walltime": walltime,
        "edits": edits, "dropped_block1_keywords": deck.dropped_keywords,
        "basis": written_deck.basis_kind, "method_lines": P.method_lines(written_deck),
        "conv_numbers": [int(c) for c in conv_deck], "conv_numbers_supercell": conv_super,
        "n_atoms_supercell": n_super, "jobs": names,
        "warnings": [w[1] for w in warns],
        "mace_version": mace_version, "phonopy_version": phonopy.__version__,
    })

    # ---- report
    for _, msg in warns:
        warn(f"  Warning: {msg}")
    if deck.dropped_keywords:
        log("Dropped with block 1 (geometry comes from the output): " + ", ".join(deck.dropped_keywords))
    log("The displaced decks, relative to a deck of the parent:")
    for e in edits:
        log(f"  {e}")
    log(f"Supercell {S.dim_tag(dim)}: {n_super} atoms ({n_cells} cells), {len(names)} displaced "
        "deck(s), " + ("symmetry of each displaced cell (CRY_SYM)" if s.get("cry_sym")
                       else "P1 (no symmetry)"))
    if nk:
        log(f"  cost per job ~ {nk} k-points (before time reversal) x {n_super} atoms"
            f" = {nk * n_super} k-point x atom")
        if not scale and n_cells > 1:
            log(f"  The supercell keeps the parent's SHRINK: {n_cells}x the k-point density "
                "the parent used.")
        log(f"Walltime per job ~ {walltime} at mace submit's 32 tasks (k-points x atoms^3, "
            "calibrated on a hybrid-DFT diamond supercell measured on CRYSTAL23, with a 1.5x "
            "margin; lighter methods and CRY_SYM run faster). A job that hits its walltime leaves "
            "no forces, and the analysis reports it.")
        if capped:
            warn("  WARNING: the estimate is beyond 7 days, the longest mace submit asks "
                 "for: use CRY_SYM, a smaller supercell or fewer k-points (SHRINK).")
    else:
        log(f"Walltime per job: no SHRINK found to estimate from; {walltime} suggested.")
    log(f"Wrote {len(names)} deck(s) + .ext to {setdir}/")
    log("Next:")
    log(f"  mace submit {setdir} --walltime {walltime}    (do NOT add --track)")
    log(f"  mace analyze phonopy {setdir}    (runs by itself when the last job finishes, "
        "where phonopy is installed)")
    return str(setdir)
