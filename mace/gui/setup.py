"""Calculation Setup: the CLI is the schema and the source of every default (plan B1-B3).

The GUI never writes calculation settings of its own. A deck is set up the
way a user sets it up at the prompt:

* ``mace convert`` (a CIF: cif2d12), ``mace opt2d12`` (a CRYSTAL .out, a
  follow-up SP/OPT/FREQ) or ``mace opt2d3`` (a .out, a properties run:
  BAND/DOSS/...) runs in a scratch folder with its "save the answers" switch;
* every field of the Setup form is one of the CLI's own questions (FIELDS):
  changing it re-runs the CLI with that question answered (answer_driver.py),
  so the settings the CLI works out from an answer (the grid, D3, the 3c
  basis, the FREQ block) are worked out by the CLI, never guessed here;
* the saved-answers JSON is the CLI's own file, kept as text on the server
  and handed to the browser verbatim (``json_text``): the browser never
  re-serializes it (0.0 stays 0.0, 1e-07 stays 1e-07);
* the deck preview runs the printed command itself (``--batch
  --options_file`` / ``--config-file --yes`` / ``--config-file``) on that
  file, in a mirror of the input's folder under the task's temp dir, so the
  preview is the deck the command writes (P == G).

A few settings no question sets (opt2d12's one-mesh-for-every-file key) are
written into the JSON by hand, as the CLI's documentation says: those are
``patch`` fields, applied in Python with the CLI's own JSON layout. The
cif2d12 k-point mesh is worked out from each cell by ``convert`` and is not
editable at all (a JSON ``k_points`` is ignored there).

Spike S6 (plan T0) classified each field by answering only its question on
the manual route: on cif2d12 and opt2d12 every question but the basis set
sits behind "use these default settings? / use the parent's settings?"
(answered No), and that route itself rewrites other keys (origin choice,
symmetry validation, the OPT block), so every asked field is derived by a
CLI run (``DERIVED_BY_CLI``); none is patched.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from . import clirun

REPO = clirun.REPO
import sys as _sys
for _d in ("Crystal_d12", "Crystal_d3"):
    if str(REPO / _d) not in _sys.path:
        _sys.path.insert(0, str(REPO / _d))
MODES = ("cif2d12", "opt2d12", "opt2d3")
CALC_TYPES = {"cif2d12": ("OPT", "SP", "FREQ"), "opt2d12": ("SP", "OPT", "FREQ"),
              "opt2d3": ("BAND", "DOSS", "TRANSPORT", "CHARGE", "POTENTIAL", "CHARGE+POTENTIAL", "ORBITALS")}
DEFAULT_CALC = {"cif2d12": "OPT", "opt2d12": "SP", "opt2d3": "BAND"}
D12_GLOB_MAX = 20 * 1024 * 1024      # the .d12 files staged for opt2d3 (small text files)
OUT_SUFFIXES = (".out", ".log", ".o")
DECK_SUFFIXES = (".d12",)             # a deck as its own parent: opt2d12 --d12-file (no .out)


class SetupError(ValueError):
    """Setup can't make a deck from this (shown to the user, with the CLI's words)."""


# ---------------------------------------------------------------- modes and names
def mode_for(path: str, calc_type: Optional[str] = None) -> str:
    low = (path or "").lower()
    if low.endswith(".cif"):
        return "cif2d12"
    if low.endswith(OUT_SUFFIXES):
        return "opt2d3" if (calc_type or "").upper() in CALC_TYPES["opt2d3"] else "opt2d12"
    if low.endswith(DECK_SUFFIXES):
        if (calc_type or "").upper() in CALC_TYPES["opt2d3"]:
            raise SetupError(deck_properties_refusal(path, calc_type))
        return "opt2d12"
    raise SetupError("A deck needs a CIF (a new calculation), a CRYSTAL .out (a follow-up) or a .d12 deck")


def is_deck(path) -> bool:
    """A .d12 set up as its own parent (opt2d12 --d12-file), not an .out."""
    return str(path or "").lower().endswith(DECK_SUFFIXES)


def deck_properties_refusal(path: str, calc_type: Optional[str]) -> str:
    name = Path(str(path)).name
    return (f"{str(calc_type).upper()} reads the wavefunction (.f9) of a finished SCF run, which a deck does not have: "
            f"set it up from the run's output ({Path(name).stem}.out)")


def check_calc(mode: str, calc_type: Optional[str]) -> str:
    if mode not in MODES:
        raise SetupError(f"unknown mode {mode!r}")
    calc = (calc_type or DEFAULT_CALC[mode]).upper()
    if calc not in CALC_TYPES[mode]:
        raise SetupError(f"{mode} makes {', '.join(CALC_TYPES[mode])} decks, not {calc}")
    return calc


def d3_base(stem: str) -> str:
    """opt2d3's base name: the stem without one _opt/_sp/_OPT/_SP (CRYSTALOptToD3.D3Generator)."""
    for suffix in ("_opt", "_sp", "_OPT", "_SP", ".out", ".log"):
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def json_name(mode: str, calc: str, stem: str) -> str:
    if mode == "cif2d12":
        return f"{stem}_options.json"
    if mode == "opt2d12":
        return f"{stem}_{calc}_options.json"
    return f"{stem}_{calc.replace('+', '_')}_d3.json"


# ---------------------------------------------------------------- the wavefunction (opt2d3)
def wavefunction_candidates(out_name: str, base: Optional[str] = None) -> List[str]:
    """The names opt2d3 looks for beside the .out, in its order
    (CRYSTALOptToD3.D3Generator._copy_wavefunction; mirrored here because the
    list is built inside that method, and a test checks it against the CLI).
    ``base``: the name opt2d3 gives the files (cli_base), d3_base by default."""
    stem = Path(out_name).stem
    if base is None:
        base = d3_base(stem)
    # the output's own wavefunction first, then as before
    names = [f"{stem}.f9", f"{stem}.f98", "fort.9", "fort.98", f"{base}.f9", f"{base}.f98"]
    for suffix in ("sp", "opt", "freq", "soc"):
        names += [f"{base}_{suffix}.f9", f"{base}_{suffix}.f98"]
        for i in range(2, 100):
            names += [f"{base}_{suffix}{i}.f9", f"{base}_{suffix}{i}.f98"]
    return names


class StillRunning(Exception):
    """409: the run is still going; its wavefunction is copied when it ends."""


JOB_STATE_PROVIDER = None        # PLAN_submit's jobs.py: (out_path) -> "RUNNING" | "PENDING" | ... | None


def _f9_live(out_path: Path, f9: Path, st) -> bool:
    """Whether the run that writes ``f9`` is still going (PLAN_analysis C18.7),
    first rule that applies: the paired job is known queued or running; the
    paired .out has no end marker and is live; the job's scratch folder
    (``.<job>.scratch``) holds a fort.9 larger than the copy here (the
    job-end copy is still running); only when none of these is known, the
    .f9 changed in the last 10 s (30 s on a network file system)."""
    from mace.utils import livefile
    if JOB_STATE_PROVIDER is not None:
        try:
            js = JOB_STATE_PROVIDER(out_path)
            if js:
                return str(js).upper() in ("RUNNING", "PENDING", "CONFIGURING", "COMPLETING")
        except Exception:
            pass
    try:
        if out_path.is_file():
            snap = livefile.snapshot(out_path)
            tail = livefile.tail_text(out_path, snap)
            if not livefile.has_terminal_marker(tail):
                if livefile.is_live(out_path, snap, tail):
                    return True
    except OSError:
        pass
    ptr = out_path.parent / f".{out_path.stem}.scratch"
    try:
        scratch = Path(ptr.read_text().strip())
        sf9 = scratch / "fort.9"
        if sf9.is_file() and sf9.stat().st_size > st.st_size:
            return True
        if sf9.is_file():
            return False
    except (OSError, ValueError):
        pass
    young = 10.0
    try:
        from .dbsafety_loader import dbsafety
        if dbsafety().is_network(dbsafety().mount_of(f9)):
            young = 30.0
    except Exception:
        pass
    return time.time() - st.st_mtime < young


def find_wavefunction(out_path: Path) -> Optional[Dict[str, Any]]:
    """The wavefunction the CLI would copy, found by name and os.stat only
    (never opened): {name, bytes, live}, or None. ``live``: the run that
    writes it is still going (``_f9_live``)."""
    folder = out_path.parent
    for name in wavefunction_candidates(out_path.name, cli_base(out_path)):
        try:
            st = os.stat(folder / name)
        except OSError:
            continue
        return {"name": name, "bytes": int(st.st_size), "live": _f9_live(out_path, folder / name, st)}
    return None


def _anatase_siblings(out_path: Path) -> List[str]:
    """Sibling .out/.log names that have or make the same base name: they
    change opt2d3's output names (CRYSTALOptToD3.D3Generator.__init__)."""
    base = d3_base(out_path.stem)
    if base == out_path.stem:
        return []
    out = []
    for other in out_path.parent.glob(f"{glob.escape(base)}*"):
        if (other.suffix in (".out", ".log") and other.name != out_path.name
                and base in (other.stem, d3_base(other.stem))):
            out.append(other.name)
    return sorted(out)


def cli_base(out_path: Path) -> str:
    """The name opt2d3 gives the files of ``out_path``: d3_base, or the whole
    stem when another output beside it has or makes that name."""
    return out_path.stem if _anatase_siblings(out_path) else d3_base(out_path.stem)


def _d12_globs(out_path: Path) -> List[Path]:
    """The .d12 files opt2d3 reads beside the .out: <base>*.d12 and <first stem token>*.d12."""
    base = d3_base(out_path.stem)
    first = out_path.stem.split("_")[0]
    seen, out = set(), []
    for pat in (f"{base}*.d12", f"{first}*.d12"):
        for p in sorted(out_path.parent.glob(pat)):
            if p.is_file() and p.name not in seen:
                seen.add(p.name)
                out.append(p)
    return out


# ---------------------------------------------------------------- staging
class Staged:
    """A mirror, under ``tree``, of the files the printed command reads, at
    the same relative paths as in --dir; the command runs there unchanged."""

    def __init__(self, mode: str, calc: str, tree: Path, input_rel: str, json_rel: str, out_rel: str,
                 f9: Optional[Dict[str, Any]] = None, notes: Optional[List[str]] = None):
        self.mode, self.calc, self.tree = mode, calc, tree
        self.input_rel, self.json_rel, self.out_rel = input_rel, json_rel, out_rel
        self.f9 = f9
        self.notes = notes or []
        self.env: Dict[str, str] = {}          # added to the CLI's environment (setup_env)


#: the most a staged BAND/DOSS run (for a spin-locked SCF's E_F) may be, data and output together
STAGE_FERMI_MAX = 64 * 2**20


def stage(mode: str, calc: str, work: Path, source_path: Optional[Path], rel: Optional[str],
          cif_bytes: Optional[bytes] = None, cif_name: Optional[str] = None) -> Staged:
    """Copy what the CLI reads into ``work/tree`` (plan B2):

    * cif2d12: the CIF alone, in its own folder setup/<stem>/ (the
      "Own folder" layout: ``convert`` reads a whole directory);
    * opt2d12: the .out and the .d12 of the same name (pair_out_files);
    * opt2d3: the .out, every .d12 the CLI globs for, zero-byte stand-ins
      for the sibling outputs whose names change its file names, and one
      zero-byte stand-in named as the wavefunction the CLI would copy:
      the deck never reads it, and opt2d3 copies the f9 even through a
      symlink (spike S10), so a preview of a 2 GB f9 costs nothing.
    Nothing outside ``work`` is written, and no symlink is made."""
    tree = work / "tree"
    tree.mkdir(parents=True, exist_ok=True)
    if mode == "cif2d12":
        name = cif_name or (source_path.name if source_path else "structure.cif")
        stem = Path(name).stem
        folder = f"setup/{stem}"
        (tree / folder).mkdir(parents=True, exist_ok=True)
        if cif_bytes is not None:
            (tree / folder / name).write_bytes(cif_bytes)
        else:
            shutil.copyfile(source_path, tree / folder / name)
        return Staged(mode, calc, tree, f"{folder}/{name}", f"{folder}/{json_name(mode, calc, stem)}", folder)
    if source_path is None or rel is None:
        raise SetupError("a follow-up needs the CRYSTAL .out on disk")
    rel_dir = str(Path(rel).parent).replace("\\", "/")
    rel_dir = "" if rel_dir == "." else rel_dir
    dst = tree / rel_dir
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, dst / source_path.name)
    stem = source_path.stem
    jrel = (f"{rel_dir}/" if rel_dir else "") + json_name(mode, calc, stem)
    if mode == "opt2d12":
        d12 = source_path.with_suffix(".d12")
        if d12.is_file() and not is_deck(source_path):        # an .out's parent deck (a deck is copied above)
            shutil.copyfile(d12, dst / d12.name)
        f9 = source_path.with_suffix(".f9")
        if not is_deck(source_path) and f9.is_file():
            # a SOC restart ("soc_guess") checks that <parent>.f9 is there and not empty, then
            # copies it as <deck>.f20: a one-byte stand-in (an empty one for an empty .f9), never the
            # real bytes; "Write decks" runs the command on the real files
            (dst / f9.name).write_bytes(b"\0" if f9.stat().st_size else b"")
        return Staged(mode, calc, tree, rel, jrel, rel_dir or ".")
    # opt2d3: name mirroring
    total = 0
    for p in _d12_globs(source_path):
        total += p.stat().st_size
        if total > D12_GLOB_MAX:
            raise SetupError(f"the .d12 files beside {source_path.name} are over {D12_GLOB_MAX // 2**20} MB")
        shutil.copyfile(p, dst / p.name)
    notes = []
    for sib in _anatase_siblings(source_path):
        (dst / sib).touch()
        notes.append(f"{sib} is beside {source_path.name}: the CLI names the files {stem}_*")
    f9 = find_wavefunction(source_path)
    if f9 is not None:
        (dst / f9["name"]).touch()          # an empty stand-in, never the real bytes
    if calc in ("DOSS", "TRANSPORT"):
        # E_F-relative ranges: a spin-locked SCF's E_F comes from a BAND or DOSS run of it
        # (mace.utils.fermi_level.scf_fermi), so the one it would use is staged beside it
        from mace.utils.fermi_level import sibling_property_runs
        for dat in sibling_property_runs(source_path)[:1]:
            out = dat.with_name(dat.name.rsplit(".", 2)[0] + ".out")
            if dat.stat().st_size + out.stat().st_size < STAGE_FERMI_MAX:
                for f in (dat, out):
                    if not (dst / f.name).exists():
                        shutil.copyfile(f, dst / f.name)
    return Staged(mode, calc, tree, rel, jrel, rel_dir or ".", f9, notes)


# ---------------------------------------------------------------- the commands
def batch_args(st: Staged) -> List[str]:
    """The printed command's arguments (paths relative to --dir)."""
    if st.mode == "cif2d12":
        return ["convert", "--batch", "--options_file", st.json_rel, "--cif_dir", st.out_rel,
                "--output_dir", st.out_rel]
    if st.mode == "opt2d12":
        return ["opt2d12", source_flag(st.input_rel), st.input_rel, "--config-file", st.json_rel, "--yes"]
    return ["opt2d3", "-i", st.input_rel, "-t", st.calc, "--config-file", st.json_rel]


def source_flag(input_rel: str) -> str:
    """opt2d12's switch for its parent: --out-file for a run, --d12-file for a deck on its own."""
    return "--d12-file" if is_deck(input_rel) else "--out-file"


def command_line(args: List[str]) -> str:
    import shlex
    return "mace " + " ".join(shlex.quote(a) for a in args)


def ask_args(st: Staged, json_out: Path, out_dir: Path) -> List[str]:
    """The interactive command with its "save the answers" switch."""
    if st.mode == "cif2d12":
        return ["convert", "--cif_dir", st.out_rel, "--output_dir", str(out_dir), "--save_options",
                "--options_file", str(json_out)]
    if st.mode == "opt2d12":
        return ["opt2d12", source_flag(st.input_rel), st.input_rel, "--calc-type", st.calc, "--save-options",
                "--options-file", str(json_out), "--output-dir", str(out_dir)]
    return ["opt2d3", "-i", st.input_rel, "-t", st.calc, "--save-config", "--options-file", str(json_out),
            "--output-dir", str(out_dir)]


# ---------------------------------------------------------------- the form: the CLI's questions
# The gates in front of the questions (answered No to reach them) and each field's question.
GATES = {
    "cif2d12": {"N": {"match": r"use these default settings\?", "answer": "n"},
                "ADV": {"match": r"use these default advanced settings\?", "answer": "n"},
                "SYM": {"match": r"use these default symmetry settings\?", "answer": "n"}},
    "opt2d12": {"N": {"match": r"use these exact settings for the new calculation", "answer": "n"}},
    "opt2d3": {},
}
# The answers a user gives to reach the defaults: opt2d12's follow-up keeps the parent's
# settings ("use these exact settings?" is [y/N]: the manual route answers y, spike S2).
DEFAULT_RULES = {
    "cif2d12": [],
    "opt2d12": [{"match": r"use these exact settings for the new calculation", "answer": "y"}],
    "opt2d3": [],
}


# The questions Setup leaves at the CLI's default (Enter) on purpose, beyond the gates and the
# fields' own questions: the CLI's other questions on the Setup routes (spike S6 transcripts).
# answer_driver answers Enter only to a question a rule or this list names; any other question
# stops the run and is named, so a reworded prompt never becomes a silent default
# (tests/gui/test_setup_prompts.py runs every route and fails on a question not matched).
_D12_COMMON = [
    r"Select functional category", r"Select Hartree-Fock method", r"SPINLOCK value", r"Use level shifting",
    r"SCF method:\s.*Select method \(1-2\)", r"Enable PPAN", r"Modify memory settings", r"Use HISTDIIS", r"HISTDIIS size",
    r"Set maximum step size \(MAXTRADIUS\)",
    # a custom optimization convergence, custom SCF tolerances
    r"Custom convergence criteria:.*Enter TOLDEG", r"Enter TOLDEX \(", r"Enter TOLDEE \(energy difference",
    r"Enter MAXCYCLE \(max optimization steps\)", r"Enter TOLINTEG values", r"Enter TOLDEE value \(integer\)",
    # FREQ: the type, the IR method and each template's own questions
    r"Frequency calculation type:.*Select type \(1-2\)", r"Configure advanced frequency settings\?",
    r"CPHF/CPKS.*Select method \(1-3\)", r"Generate (IR|Raman) spectrum plot\?", r"Customize spectral range\?",
    r"Customize temperature range\?", r"Customize supercell size\?", r"Band path definition:.*Select method \(1-4\)",
    r"Automatic path format:.*Select format \(1-4\)", r"Select DOS calculation type:.*Select type \(1-2\)",
    r"Maximum frequency for DOS", r"Number of DOS bins", r"Calculate atom-projected DOS\?",
    r"Frequency calculation modes:.*Select mode \(1-2\)", r"Calculate Raman intensities\?",
    r"Calculate thermodynamic properties\?", r"Calculate atom-projected vibrational modes\?",
    r"Calculate zero-point energy\?", r"Enable mode tracking", r"Configure restart capability\?",
    r"Enable parallel displacement calculations\?",
    # behind a non-default answer ("Other questions"): level shifting, memory, MAXTRADIUS, FREQ's advanced settings
    r"LEVSHIFT ISHIFT ILOCK \[", r"(^|\n)BIPOSIZE \[", r"(^|\n)EXCHSIZE \[", r"Enter MAXTRADIUS \(max displacement, default",
    r"Apply Eckart conditions", r"Calculate anharmonic corrections\?", r"Print level for frequency calculation:.*Select print level",
    r"Enable mode following", r"Symmetry handling:.*Select symmetry option \(1-3\)",
    r"Maintain symmetry \(move all equivalent H atoms\)\?", r"Use extended 26-point grid", r"Calculate isotope effects",
    # FREQ, behind a non-default answer: each template's customizations (spectral and temperature
    # range, supercell, k-point density, INS), the band path's crystal system and its templates
    r"(Minimum|Maximum) frequency \(cm⁻¹\) \[", r"Number of temperature points \[", r"(Minimum|Maximum) temperature \(K\) \[",
    r"Supercell N[₁₂₃] \[", r"Select system \(1-6\) \[", r"Select template \(1-[234]\) \[",
    r"Use mixed label/coordinate segments\?", r"Customize k-point density\?", r"Shrink factor for k-points \[",
    r"Points per band segment \[", r"Select neutron weighting \(0-2\)", r"Maximum frequency for INS",
    r"Number of (spectrum|INS) bins", r"Inelastic Neutron Scattering\) spectrum\?",
    r"Both coherent and incoherent.*Select type \(0-2\)",
    # FREQ custom settings: phonon dispersion (supercell, INTERPHESS, bands, DOS), IR by CPHF or Wannier, restart
    r"Select supercell method \(1-3\)", r"Expansion in [xyz] \[", r"Use Fourier interpolation \(INTERPHESS\)\?",
    r"L[123] expansion factor \[", r"(^|\n)Print level \(0-2\) \[", r"Is this a polar material",
    r"Calculate phonon band structure\?", r"High-symmetry points only.*Select type \(1-2\)", r"Select path method \(1-4\)",
    r"Calculate phonon density of states\?", r"Relocalize at each displaced geometry \(RELOCAL\)\?",
    r"Maximum CPHF iterations \[", r"CPHF convergence tolerance \(10\^-x\) \[", r"Checkpoint interval \(calculations\) \[",
    # custom settings' IR/Raman spectrum parameters, asked after a Yes to the spectrum plot (IRSPEC/
    # RAMSPEC): Enter keeps CRYSTAL's defaults (no LENSTEP, Lorentzian, DAMPFAC 8, no RAMANEXP)
    r"Spectrum step size \(cm⁻¹\) \[", r"Select broadening \(1-2\)", r"(Lorentzian|Gaussian) FWHM \(cm⁻¹\) \[",
    r"Use RAMANEXP for experimental conditions\?", r"Laser wavelength \(nm\) \[", r"(^|\n)Temperature \(K\) \[",
]
KNOWN_QUESTIONS = {
    "cif2d12": _D12_COMMON + [r"Save these options for future batch processing\?", r"Atom writing options",
                              r"Validate symmetry operations\?", r"Trigonal axes preference", r"Select origin choice",
                              # the CIF's space group and spglib's differ: keep the CIF's (the batch default)
                              r"Use original CIF space group.*Select option \(default",
                              # spglib finds no smaller asymmetric unit (a P1 CIF, a slab or molecule
                              # in a box): Enter keeps the CLI's default, the full set either way
                              r"Use the 'reduced' structure anyway\?",
                              # "Use original CIF space group" answered 1 (try other tolerances): Enter
                              # takes a tolerance that matches, else keeps the CIF's group
                              r"Match found with tolerance.*Use tolerance \S+\?", r"Final choice \(default"],
    # the parent's basis set not kept: the basis questions cif2d12 asks
    "opt2d12": _D12_COMMON + [r"Select basis set type", r"Select (internal|external) basis set",
                              # a parent functional MACE cannot identify (e.g. B2PLYP), the method
                              # asked again: Enter uses HSE06, as the CLI says
                              r"Enter another functional, 'm' to choose from the menu, or press Enter to use"],
    "opt2d3": [
        # BAND: the template route, a band window; DOSS
        r"Select crystal system:", r"Available templates:.*Select template \(", r"Use mixed label/coordinate segments\?",
        r"Segment format:.*Select format \(1-2\)", r"Number of bands (below|above) Fermi", r"First band \[", r"Last band \[",
        r"Print eigenvalues to output\?", r"Atom selection:.*Select \(1-2\)", r"(Lower|Upper) energy limit \(eV (below|above) Fermi\)",
        r"Output format:.*Select format \(1-2\)", r"Print integrated DOS to output\?",
        # TRANSPORT
        r"(Minimum|Maximum) temperature \(K\)", r"Temperature step \(K\)", r"Chemical potential options:",
        r"(Minimum|Maximum) μ \(eV\)",
        # μ relative to the output's Fermi energy (an output that has one)
        r"(Minimum|Maximum) μ \(eV (below|above) Fermi\)", r"μ step \(eV\)", r"(Minimum|Maximum) energy \(eV\)", r"Energy step \(eV\)",
        r"Relaxation time \(fs\)", r"Use smearing for metallic systems\?",
        # CHARGE, POTENTIAL
        r"Charge density type:.*Select type \(1-2\)", r"Potential calculation type:.*Select type \(1-2\)",
        r"Number of grid points per direction", r"define non-periodic directions:.*Select option \(1-2\)",
        r"Scale factor for non-periodic directions", r"Bipolar expansion tolerance",
        # behind a non-default answer: TRANSPORT smearing, a 2D charge map (ECHG), potential at points (POTC)
        r"Smearing width \(eV\)", r"Smearing type:.*Select type \(0-2\)", r"Derivative order:.*Select order \(0-1\)",
        r"Coordinate type \(F\)ractional or \(C\)artesian",
        r"Number of points along the B-A segment", r"Calculation mode:.*Select mode \(0, 2 or 3\)",
        r"Define points:.*Select option \(1-3\)", r"Calculate at all atoms\?", r"Z (minimum|maximum) \(bohr\)",
        r"Number of planes \[", r"Half thickness of the volume",
        # POTC at custom points or from POTC.INP: how many (the points themselves have no default)
        r"(^|\n)Number of points( in file)?: ",
        # no wavefunction next to the output: Enter is No, and the CLI's refusal is shown
        r"will fail without the wavefunction file.*Continue anyway\?",
        # BAND: Setup has the k-path editor for it (Enter is No, the CLI's default)
        r"Edit the band k-path \(reorder, reverse, remove or add points\)\?",
    ],
}
# Not on the list: questions with no default to keep (opt2d3's "Path:", "Number of path segments:",
# "Enter projection specification:", "Point A (x y z):", ANHARM's H atom); Enter there answers
# nothing, so the run stops and names them. tests/gui/prompt_coverage.py lists every CLI question
# not on the list, and why (test_setup_prompts.py checks the CLI sources against both).


class Field:
    def __init__(self, id: str, label: str, section: str, kind: str, match: str, keys=(), gates=(),
                 modes=("cif2d12", "opt2d12"), calcs=None, unit: str = "", hint: str = ""):
        self.id, self.label, self.section, self.kind, self.match = id, label, section, kind, match
        self.keys, self.gates, self.modes, self.calcs, self.unit, self.hint = tuple(keys), tuple(gates), modes, calcs, unit, hint
        self.re = re.compile(match, re.I | re.S)

    def gates_for(self, mode: str, value: Any) -> Tuple[str, ...]:
        return self.gates

    def rules(self, mode: str, value: Any) -> List[Dict[str, Any]]:
        if self.kind == "menu":
            r = {"match": self.match, "choose": str(value)}
        elif self.kind == "yesno":
            r = {"match": self.match, "answer": "y" if value in (True, "y", "yes", "Yes", "true") else "n"}
        else:
            r = {"match": self.match, "answer": str(value)}
        return [r]


METHOD_Q = r"Select method type"
HF_METHOD_LABEL = "Hartree-Fock (HF)"
DFT_METHOD_LABEL = "Density Functional Theory (DFT)"
KEEP_BASIS = "Keep the parent's basis set"


def hf_methods() -> List[str]:
    """The Hartree-Fock variants of the CLI's own table (RHF, UHF, HF3C, HFSOL3C)."""
    try:
        import d12_constants as C
        return list(C.FUNCTIONAL_CATEGORIES["HF"]["functionals"])
    except (ImportError, KeyError):
        return []


class FunctionalField(Field):
    """The functional. The CLI asks for the method (Hartree-Fock or DFT), then for the functional's
    category, then for the functional (or for the Hartree-Fock variant); the functional alone says
    all three, so the form has this one row and the other answers are worked out from it."""

    def rules(self, mode: str, value: Any) -> List[Dict[str, Any]]:
        name = str(value)
        if name in hf_methods():
            return [{"match": METHOD_Q, "choose": HF_METHOD_LABEL},
                    {"match": r"Select Hartree-Fock method", "choose": name}]
        cat = functional_category(name)
        out = [{"match": METHOD_Q, "choose": DFT_METHOD_LABEL}]
        if cat:
            out.append({"match": r"Select functional category", "choose": cat})
        out.append({"match": self.match, "choose": name})
        return out


class BasisField(Field):
    """The basis set. The CLI asks whether the parent's is kept (a follow-up), then for the type
    (internal or external), then for the set; the set alone says all three."""

    def gates_for(self, mode: str, value: Any) -> Tuple[str, ...]:
        # a follow-up reaches the basis questions only past "use these exact settings?" (No)
        return (N,) if mode == "opt2d12" else ()

    def rules(self, mode: str, value: Any) -> List[Dict[str, Any]]:
        name = str(value)
        keep = FIELD["basis_keep"].match
        if name == KEEP_BASIS:
            return [{"match": keep, "answer": "y"}] if mode == "opt2d12" else []
        out = [{"match": keep, "answer": "n"}] if mode == "opt2d12" else []
        external = name in external_basis_labels() or "/" in name or name.startswith("full.basis")
        out.append({"match": FIELD["basis_type"].match, "choose": "EXTERNAL" if external else "INTERNAL"})
        out.append({"match": r"Select external basis set" if external else r"Select internal basis set",
                    "choose": name})
        return out


def external_basis_labels() -> List[str]:
    """The names the CLI's external basis menu gives the bundled sets (DZVP-REV2, TZVP-REV2, ...)."""
    try:
        import basis_library as BL
        return [BL.FAMILY_LABELS[f] for f in ("full.basis.doublezeta", "full.basis.triplezeta",
                                              "full.basis.triplezeta.rev2022")]
    except (ImportError, KeyError):
        return []


class SmearField(Field):
    """The smearing width: "Custom value" in the width menu, then the number."""

    def rules(self, mode: str, value: Any) -> List[Dict[str, Any]]:
        return [{"match": r"Smearing width options", "choose": "Custom value"},
                {"match": r"Enter smearing width in Hartree", "answer": str(value)}]


N, ADV, SYM = "N", "ADV", "SYM"
# id, label, section, kind, the question (regex on its text), JSON keys (display), gates, modes, calc types
FIELDS: List[Field] = [
    # cif2d12's calculation type is a question too (the other modes take it as --calc-type / -t)
    # the deck's dimensionality (plan E10): CRYSTAL, SLAB, POLYMER or MOLECULE; a molecule, slab or polymer is
    # oriented first (structures.standardize_for_dim), so the CLI's coordinate writer gets the cell it handles
    Field("dimensionality", "Dimensionality", "Structure", "menu", r"Select system dimensionality", ["dimensionality"], [N],
          ("cif2d12",), hint="CRYSTAL (3D), SLAB (2D, vacuum along z), POLYMER (1D, chain along x) or MOLECULE (0D)"),
    Field("calc_type", "Calculation", "Calculation", "menu", r"Select calculation type", ["calculation_type"], [N], ("cif2d12",)),
    Field("symmetry", "Symmetry handling", "Structure", "menu", r"Select symmetry handling", ["symmetry_handling"], [N, SYM],
          ("cif2d12",), hint="CIF keeps the file's space group; SPGLIB finds it; P1 writes every atom"),
    Field("opt_type", "Optimization type", "Optimization", "menu", r"Select optimization type", ["optimization_type", "optimization_settings.type"], [N],
          calcs=("OPT",)),
    Field("opt_convergence", "Convergence", "Optimization", "menu", r"Select convergence level", ["optimization_settings.convergence"], [N],
          calcs=("OPT",)),
    Field("freq_template", "Frequency calculation", "Frequencies", "menu", r"Select template \([01]-8\)", ["freq_settings.template"], [N],
          calcs=("FREQ",)),
    Field("numderiv", "NUMDERIV", "Frequencies", "menu", r"Numerical derivative method", ["freq_settings.numderiv"], [N], calcs=("FREQ",)),
    Field("ir", "IR intensities", "Frequencies", "yesno", r"Calculate IR intensities\?", ["freq_settings.intensities"], [N], calcs=("FREQ",)),
    Field("method", "Method", "Method", "menu", r"Select method type", ["method", "method_type"], [N]),
    FunctionalField("functional", "Functional", "Method", "menu", r"Select [\w/ -]*functional \(default", ["dft_functional", "functional"], [N]),
    Field("dispersion", "D3 dispersion", "Method", "yesno", r"Add D3 dispersion correction", ["dispersion", "use_dispersion"], [N]),
    Field("basis_keep", "Parent's basis set", "Method", "yesno", r"Keep the basis set from the original D12\?", ["use_original_external_basis"], [N],
          ("opt2d12",)),
    Field("basis_type", "Basis set type", "Method", "menu", r"Select basis set type", ["basis_set_type"], [], ("cif2d12",)),
    BasisField("basis_set", "Basis set", "Method", "menu", r"Select (internal|external) basis set", ["basis_set"], [],
               ("cif2d12", "opt2d12"), hint="internal or external follows from the set; a follow-up keeps its parent's"),
    Field("spin", "Spin polarized", "Method", "yesno", r"Use spin-polarized calculation\?", ["spin_polarized", "is_spin_polarized"], [N, ADV]),
    Field("dft_grid", "DFT grid", "k-points and SCF", "menu", r"Select integration grid", ["dft_grid"], [N]),
    Field("tolerance", "SCF tolerances", "k-points and SCF", "menu", r"Select (tolerance level|SCF convergence level)", ["tolerances.TOLINTEG"], [N]),
    Field("maxcycle", "SCF max cycles", "k-points and SCF", "value", r"(^|\n)MAXCYCLE \[", ["maxcycle", "scf_maxcycle"], [N, ADV]),
    Field("fmixing", "FMIXING", "k-points and SCF", "value", r"(^|\n)FMIXING \[", ["fmixing"], [N, ADV], unit="%"),
    Field("smearing", "Fermi smearing", "k-points and SCF", "yesno", r"Use Fermi smearing\?", ["use_smearing"], [N, ADV]),
    SmearField("smear_width", "Smearing width", "k-points and SCF", "value", r"Smearing width options|Enter smearing width",
               ["smearing_width"], [N, ADV], unit="Ha"),
    # opt2d12's other SCF records (its "modify settings" questions): the mixing method, LEVSHIFT, SPINLOCK
    Field("scf_method", "SCF mixing", "k-points and SCF", "menu", r"SCF method:\s.*Select method \(1-2\)",
          ["scf_settings.method", "scf_method"], [N], ("opt2d12",), hint="DIIS, or Broyden for a difficult SCF"),
    Field("levshift", "Level shifting (LEVSHIFT)", "k-points and SCF", "yesno", r"Use level shifting",
          ["scf_settings.levshift"], [N], ("opt2d12",)),
    Field("levshift_values", "LEVSHIFT ISHIFT ILOCK", "k-points and SCF", "value", r"LEVSHIFT ISHIFT ILOCK \[",
          ["scf_settings.levshift"], [N], ("opt2d12",), hint="the shift in 0.1 Ha, then 1 to lock it (0: release)"),
    Field("spinlock", "SPINLOCK (nα-nβ)", "k-points and SCF", "value", r"SPINLOCK value",
          ["spinlock"], [N], ("opt2d12",), hint="0: the spin state is optimized; a number holds nα-nβ"),
    # opt2d3
    Field("band_path", "Band path", "BAND", "menu", r"Band path definition", ["path_method"], [], ("opt2d3",), ("BAND",)),
    Field("band_format", "Path format", "BAND", "menu", r"Band path format", ["path_method"], [], ("opt2d3",), ("BAND",)),
    Field("n_points", "k-points along the path (NSUB)", "BAND", "value", r"Total number of k-points along path", ["n_points"], [],
          ("opt2d3",), ("BAND",)),
    Field("band_range", "Bands", "BAND", "menu", r"Band range selection", ["bands"], [], ("opt2d3",), ("BAND",)),
    Field("doss_projection", "Projections", "DOSS", "menu", r"DOS projection type", ["projection_type"], [], ("opt2d3",), ("DOSS",)),
    Field("doss_range", "Energy range", "DOSS", "menu", r"Energy/band range", ["energy_range"], [], ("opt2d3",), ("DOSS",)),
    Field("doss_resolution", "Resolution", "DOSS", "menu", r"Select energy resolution", ["energy_resolution"], [],
          ("opt2d3",), ("DOSS",), hint="the energy step of the DOS grid; each choice shows the points and step it gives "
                                       "(all bands: the span estimated from the atoms)"),
    Field("doss_step", "Energy step", "DOSS", "value", r"DOS energy step \(eV\)", ["energy_step_ev"], [], ("opt2d3",), ("DOSS",),
          unit="eV"),
    Field("doss_points", "Energy points", "DOSS", "value", r"Number of energy points", ["n_points"], [], ("opt2d3",), ("DOSS",),
          hint="written as given, however coarse (a screening run)"),
    # offered when a step over every band needs more than 5000 points (and E_F is known): the one-click window
    Field("doss_window", "Window around E_F", "DOSS", "yesno", r"Use a window around E_F", ["energy_window"], [],
          ("opt2d3",), ("DOSS",), hint="the default above 5000 points: only the energies around E_F, without the core "
                                       "levels and the high virtual bands; No keeps every band"),
    Field("doss_window_lo", "Window below E_F", "DOSS", "value", r"Window below E_F \(eV\)", [], [], ("opt2d3",),
          ("DOSS",), unit="eV"),
    Field("doss_window_hi", "Window above E_F", "DOSS", "value", r"Window above E_F \(eV\)", [], [], ("opt2d3",),
          ("DOSS",), unit="eV"),
    # a 1D material's default instead of the window: its bands by index (CRYSTAL needs a window's bounds in gaps)
    Field("doss_band_index", "Bands by index", "DOSS", "yesno", r"by index instead\?", [], [],
          ("opt2d3",), ("DOSS",), hint="a 1D material's default above 5000 points: its valence and low conduction "
                                       "bands by index, never by energy; No keeps every band"),
    Field("doss_conduction_bands", "Conduction bands", "DOSS", "value",
          r"Conduction bands above the highest occupied band", ["conduction_bands"], [], ("opt2d3",), ("DOSS",),
          hint="how far the 1D bands by index reach above the highest occupied band (--doss-conduction-bands)"),
    Field("npol", "NPOL", "DOSS", "value", r"NPOL value", ["npol"], [], ("opt2d3",), ("DOSS",)),
    Field("transport_k", "k-point mesh", "TRANSPORT", "menu", r"Select k-point mesh", ["k_resolution"], [], ("opt2d3",),
          ("TRANSPORT",), hint="BOLTZTRA's NEWK mesh, by k spacing along the reciprocal vectors; never coarser than the "
                               "parent SCF's (whose mesh converges energies, not transport)"),
    Field("transport_k_spacing", "k spacing", "TRANSPORT", "value", r"k-point spacing \(1/Å\)", ["k_spacing"], [],
          ("opt2d3",), ("TRANSPORT",), unit="1/Å"),
    Field("transport_is", "NEWK IS", "TRANSPORT", "value", r"NEWK shrinking factor IS", ["newk_is"], [], ("opt2d3",),
          ("TRANSPORT",)),
    Field("orbitals_kmesh", "k points", "ORBITALS", "menu", r"Select k points \(1-4\)", ["kmesh"], [], ("opt2d3",),
          ("ORBITALS",), hint="every orbital is written at each k point of the NEWK mesh; complex k points "
                              "(IS 3 and up) are usable in 1D only"),
    Field("orbitals_is", "Shrinking factor IS", "ORBITALS", "value", r"Shrinking factor IS \[", ["is"], [], ("opt2d3",),
          ("ORBITALS",)),
]
FIELD = {f.id: f for f in FIELDS}
# How each field is set (spike S6): "cli" = by answering the CLI's question (a CLI run);
# "patch" = written into the JSON (no question sets it); "readonly" = the CLI ignores it.
DERIVED_BY_CLI: Dict[str, str] = {f.id: "cli" for f in FIELDS}
PATCH_FIELDS = {
    # opt2d12: one k-point mesh for the follow-up (the CLI's documented hand-written key)
    "k_points_for_all_files": {"modes": ("opt2d12",), "label": "k-point mesh", "section": "k-points and SCF",
                               "hint": "written as k_points_for_all_files, the one key opt2d12 reads for a mesh: "
                                       "IS ISP (8 16), IS1 IS2 IS3 per axis, or auto (from the cell)"},
    # opt2d3 BAND: the edited k-path (Crystal_d3/d3_band_path.py), drawn by the k-path editor, not a form row
    "band_path": {"modes": ("opt2d3",), "label": "k-path", "section": "BAND", "kind": "kpath",
                  "hint": "written as band_path, the edited path opt2d3 computes (its own prompt writes the same key)"},
}
# Spin-orbit coupling (1.1.6): keys of the CLI's own settings file, as a user writes them
# (docs/cli.md, "Spin-orbit coupling (SOC) decks"; no CLI question sets them). The Setup
# panel shows them in its own section (kind "soc"), not as form rows.
SOC_KEYS = ("soc", "soc_ecp", "soc_basis", "soc_guess", "smear")
SOC_MODES = ("cif2d12", "opt2d12")
SOC_CALCS = ("SP",)
for _k, _label in zip(SOC_KEYS, ("Spin-orbit coupling", "Spin-orbit ECP", "Official SOC basis",
                                 "Starting guess", "SMEAR")):
    PATCH_FIELDS[_k] = {"modes": SOC_MODES, "label": _label, "section": "Spin-orbit coupling", "kind": "soc",
                        "hint": f'written as "{_k}" in the settings file, the key the CLI reads'}
DERIVED_BY_CLI.update({k: "patch" for k in PATCH_FIELDS})
DERIVED_BY_CLI["k_points_cif2d12"] = "readonly"


def functional_category(name: str) -> Optional[str]:
    try:
        import d12_constants as C
    except ImportError:
        return None
    for cat in C.FUNCTIONAL_CATEGORIES.values():
        if name.upper() in (f.upper() for f in cat.get("functionals", [])):
            return cat["name"]
    return None


def calc_groups() -> Dict[str, Any]:
    """The calculation types of Setup and the Calculation menu, grouped and numbered as the CLI's
    menus, each with what the file in front needs (the workflow's prerequisites:
    mace.workflow.common.constants SETUP_CALC_GROUPS and setup_requirement)."""
    from mace.workflow.common import constants as WC
    groups = [{"key": key, "label": label, "cli": cli,
               "types": [{"calc": t, "n": n, "label": words, "needs": WC.setup_requirement(t)} for t, n, words in types]}
              for key, label, cli, types in WC.SETUP_CALC_GROUPS]
    run = " or ".join(WC.FINISHED_RUN_TYPES)
    return {"groups": groups, "needs": {"run": f"a finished {run} output (.out)",
                                        "structure": "a structure (CIF), a deck (.d12) or a CRYSTAL output (.out)"}}


def schema() -> Dict[str, Any]:
    """What the form offers beyond the questions: the functionals by
    category and the internal basis sets with the elements each covers,
    read from the CLI's modules (never GUI literals)."""
    import d12_constants as C
    cats = [{"key": k, "name": v["name"], "functionals": list(v.get("functionals", [])),
             "basis_requirements": dict(v.get("basis_requirements", {}))}
            for k, v in C.FUNCTIONAL_CATEGORIES.items()]
    cover = {k: list(v) for k, v in getattr(C, "VERIFIED_INTERNAL_BASIS_ELEMENTS", {}).items()}
    return {"modes": list(MODES), "calc_types": {k: list(v) for k, v in CALC_TYPES.items()},
            "calc_groups": calc_groups(), "default_calc": DEFAULT_CALC, "functional_categories": cats,
            "internal_basis_sets": list(C.INTERNAL_BASIS_SETS), "basis_coverage": cover,
            "fields": [{"id": f.id, "label": f.label, "section": f.section, "kind": f.kind, "modes": list(f.modes),
                        "calcs": list(f.calcs) if f.calcs else None, "unit": f.unit, "hint": f.hint,
                        "how": DERIVED_BY_CLI[f.id]} for f in FIELDS],
            "patch_fields": {k: {**v, "modes": list(v["modes"])} for k, v in PATCH_FIELDS.items()},
            "derived_by_cli": DERIVED_BY_CLI, "soc": soc_schema(), "cli_hash": clirun.cli_hash(),
            "charge_flags": charge_schema(), "charge_modes": list(CHARGE_MODES),
            "static_kpath": {"switch": STATIC_KPATH_SWITCH, "modes": list(STATIC_KPATH_MODES), "label": "Static k-path",
                             "hint": "SeeK-path's path is the default and needs the seekpath package; without it the CLI "
                                     "stops, and with this switch it writes the static table path of the space group"}}


def known_questions(mode: str) -> List[str]:
    """The questions Enter may answer on this mode's routes: the gates, every field's
    question (left at its default unless answered), the CLI's save-the-answers question
    and KNOWN_QUESTIONS. answer_driver stops at any question not on this list."""
    out = [g["match"] for g in GATES[mode].values()] + [r["match"] for r in DEFAULT_RULES[mode]]
    for f in FIELDS:
        if mode in f.modes:
            out.append(f.match)
    return out + KNOWN_QUESTIONS[mode]


# an "Other questions" row of a question asked more than once (CHARGE+POTENTIAL's charge and
# potential blocks): "q:<text>#<n>", the n-th time it is asked
_NTH_RE = re.compile(r"^q:(.+)#(\d+)$", re.S)


def _q_answer(value: Any) -> Dict[str, Any]:
    """An "Other questions" row's answer: a menu row sends its option's name ("POTC"), which the CLI
    reads as a number ("Select type (1-2)"), so a name is typed as its menu line's number where the
    question shows one (or_choose); a number or a value question's answer is typed as it is."""
    text = str(value)
    if text.strip() and not re.fullmatch(r"\s*[-+]?[\d.eE+-]*\s*", text):
        return {"answer": text, "or_choose": True}
    return {"answer": text}


def answer_rules(mode: str, answers: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The answer script for these answers: the gates the fields need, then each field's answers."""
    gates, rules = [], []
    # the n-th ask of a repeated question: one rule per ask up to the last one answered, each used
    # once; an ask not answered here keeps its default (Enter)
    nth: Dict[str, Dict[int, Any]] = OrderedDict()
    for fid, value in answers.items():
        m = _NTH_RE.match(fid)
        if m:
            nth.setdefault(m.group(1), {})[int(m.group(2))] = value
    for q, by_n in nth.items():
        if mode in ("cif2d12", "opt2d12") and N not in gates:
            gates.append(N)
        for n in range(1, max(by_n) + 1):
            v = by_n.get(n, "")
            v = v if isinstance(v, dict) else _q_answer(v)
            rules.append({"match": re.escape(q), **{k: v[k] for k in ("answer", "choose", "or_choose") if k in v},
                          "once": True})
    for fid, value in answers.items():
        if _NTH_RE.match(fid):
            continue
        if fid.startswith("q:"):                         # a question no field names ("Other questions")
            q = str(fid[2:])
            gates_needed = [N] if mode in ("cif2d12", "opt2d12") else []
            for g in gates_needed:
                if g not in gates:
                    gates.append(g)
            v = value if isinstance(value, dict) else _q_answer(value)
            rules.append({"match": re.escape(q), **{k: v[k] for k in ("answer", "choose", "or_choose") if k in v}})
            continue
        f = FIELD.get(fid)
        if f is None or mode not in f.modes:
            raise SetupError(f"{fid!r} is not a {mode} setting")
        for g in f.gates_for(mode, value):
            if g not in gates:
                gates.append(g)
        rules += f.rules(mode, value)
    # the gates first (a rule found first wins), then the answers, then the defaults' own answers
    return [GATES[mode][g] for g in gates if g in GATES[mode]] + rules + DEFAULT_RULES[mode]


# ---------------------------------------------------------------- the CLI's JSON, kept as text
def dumps_like_cli(obj: Any) -> str:
    """json.dump(obj, f, indent=2) as every CLI saves its answers: no trailing newline."""
    return json.dumps(obj, indent=2)


def loads_ordered(text: str) -> Any:
    return json.loads(text, object_pairs_hook=OrderedDict)


def _cast_like(old: Any, value: Any) -> Any:
    """``value`` with the type the CLI's own value has (30.0 stays a float: 40 -> 40.0)."""
    if isinstance(old, bool):
        return value in (True, "true", "True", "y", "yes", 1, "1")
    if isinstance(old, float):
        return float(value)
    if isinstance(old, int):
        f = float(value)
        return int(f) if f.is_integer() else f
    if isinstance(old, list) and isinstance(value, str):
        return [int(x) if x.lstrip("-").isdigit() else x for x in value.split()]
    return value


def apply_patch(json_text: str, patch: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Write ``patch`` ({key: value}, a dotted key for a nested one) into the
    CLI's JSON text, each value cast to the type the key already has, keys
    in the CLI's order; an empty patch gives the same bytes back."""
    if not patch:
        return json_text, []
    obj = loads_ordered(json_text)
    target = obj["configuration"] if isinstance(obj, dict) and isinstance(obj.get("configuration"), dict) \
        and obj.get("type") == "d3_configuration" else obj
    changed = []
    for key, value in patch.items():
        parts = key.split(".")
        d = target
        for p in parts[:-1]:
            d = d.setdefault(p, OrderedDict())
        last = parts[-1]
        new = _cast_like(d[last], value) if last in d and d[last] is not None else value
        if value is None:
            if last in d:
                del d[last]
                changed.append(key)
            continue
        if d.get(last, object()) != new or type(d.get(last)) is not type(new):
            d[last] = new
            changed.append(key)
    return dumps_like_cli(obj), changed


def json_value(obj: Any, key: str) -> Any:
    d = obj.get("configuration", obj) if isinstance(obj, dict) and obj.get("type") == "d3_configuration" else obj
    for p in key.split("."):
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


def validate(mode: str, json_text: str) -> List[str]:
    """Problems that stop the CLI from reading this JSON (empty = fine)."""
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        return [f"not valid JSON: {exc}"]
    if not isinstance(data, dict):
        return ["the settings file must be a JSON object"]
    if mode == "opt2d3":
        try:
            import d3_config
        except ImportError:
            return []
        conf = data.get("configuration", data)
        ok, errors = d3_config.validate_d3_config(conf) if isinstance(conf, dict) else (False, ["no configuration"])
        return [] if ok else list(errors)
    from . import decks
    return decks.validate(mode, data)


# ---------------------------------------------------------------- the transcript -> the form
_OPTION = re.compile(r"^\s*(\d+)\s*[:.)]\s*(\S.*?)\s*$", re.M)
_DEFAULT = re.compile(r"\(default:\s*([^)]*)\)|\[(?:1-\d+,\s*)?default=([^\]]*)\]|\[([^\]]*)\]\s*:?\s*$", re.I)


def question_options(text: str) -> List[Dict[str, str]]:
    """The menu lines of a question: the last block of numbered lines."""
    blocks, cur, last_end = [], [], -1
    for m in _OPTION.finditer(text):
        if cur and text.count("\n", last_end, m.start()) > 4:
            blocks.append(cur)
            cur = []
        cur.append(m)
        last_end = m.end()
    if cur:
        blocks.append(cur)
    if not blocks:
        return []
    # the block printed last is the one the answer picks from (a menu can be described first, then listed)
    best = blocks[-1]
    out, seen = [], set()
    for m in best:
        num, body = m.group(1), m.group(2)
        if num in seen:
            continue
        seen.add(num)
        head = re.split(r"\s+[-–]\s+", body, maxsplit=1)[0].strip()
        out.append({"num": num, "head": head, "text": body})
    return out


def question_default(text: str, prompt: str) -> Optional[str]:
    """The default the question shows: "(default: 7)", "[1-4, default=1]", "[800]", "[Y/n]";
    on the prompt line, else on the last line that shows one ("Select ... (default: 1):" then a
    menu, then "Enter your choice:")."""
    lines = [x.strip() for x in (text.strip().splitlines() + [prompt or ""]) if x.strip()]
    m = None
    for line in reversed(lines):
        for m2 in _DEFAULT.finditer(line):
            m = m2
        if m:
            tail = line
            break
    if not m:
        return None
    v = next((g for g in m.groups() if g is not None), "").strip()
    if v.upper() == "Y/N":
        return "y" if "[Y/n]" in tail else "n"
    return v


class Question:
    def __init__(self, raw: Dict[str, Any]):
        self.text = raw.get("question") or ""
        self.prompt = (raw.get("prompt") or "").strip()
        self.answer = raw.get("answer")
        self.options = question_options(self.text)
        self.default = question_default(self.text, self.prompt)
        lines = [x for x in self.text.strip().splitlines() if x.strip()]
        self.last = lines[-1].strip() if lines else self.prompt


def _matches(field: Field, q: Question) -> bool:
    return bool(field.re.search(q.text))


def _basis_dir_option(options: List[Dict[str, str]], want: str) -> Optional[Dict[str, str]]:
    """The menu line of an external basis directory the JSON names: the same
    path, or the same bundled family (a bare family name, or another
    checkout's path to it), so a path from another machine still reads as
    "TZVP-REV2" and not as its whole directory."""
    try:
        import basis_library as BL
    except ImportError:
        BL = None
    norm = want.rstrip("/")
    fam = BL.bundled_family_for(want) if BL else None
    for o in options:
        detail = o["text"].split(" - ", 1)[-1].strip()
        if detail.rstrip("/") == norm:
            return o
        if fam and BL.bundled_family_for(detail) == fam:
            return o
    return None


def _short_value(val: Any) -> str:
    """A value as a form row shows it: a directory path cut to its last folders (the row's tooltip has all of it)."""
    text = str(val)
    if "/" in text and len(text) > 40:
        try:
            import basis_library as BL
            return BL.display_name(text)
        except ImportError:
            return text
    return text


def _value_label(field: Field, q: Optional[Question], obj: Any) -> Tuple[Any, Optional[str]]:
    """The field's value now (from the JSON when it holds it) and the menu head it shows as."""
    val = None
    for k in field.keys:
        val = json_value(obj, k) if obj is not None else None
        if val is not None:
            break
    if field.id in ("doss_window", "doss_band_index") and obj is not None and (
            json_value(obj, "all_bands") or json_value(obj, "all_bands_confirmed")):
        return False, "No"                          # the window's opt-out (opt2d3's No, --doss-window all)
    if field.kind == "yesno":
        if isinstance(val, (list, tuple)):          # a record's values (LEVSHIFT's ISHIFT ILOCK): there is one
            val = bool(val)
        if isinstance(val, bool):
            return val, "Yes" if val else "No"
        if q is not None and q.default in ("y", "n"):
            return q.default == "y", "Yes" if q.default == "y" else "No"
        return val, None
    if field.kind == "value":
        if isinstance(val, (list, tuple)):          # a record's values, as the CLI asks for them ("5 1")
            val = " ".join(str(x) for x in val)
        if val is None and q is not None:
            val = q.default
        return val, None if val is None else str(val)
    # a menu: the option the JSON's value names, else the question's default
    if q is not None and q.options:
        want = str(val).strip() if val is not None else None
        hit = None
        if want:
            from .answer_driver import pick
            num = pick("\n".join(f"{o['num']}: {o['text']}" for o in q.options), want)
            hit = next((o for o in q.options if o["num"] == num), None)
        if hit is None and want and ("/" in want or "full.basis" in want or want == "stuttgart"):
            hit = _basis_dir_option(q.options, want)
        if hit is None and q.default:
            hit = next((o for o in q.options if o["num"] == q.default), None)
        return val, hit["head"] if hit else (_short_value(val) if val is not None else None)
    return val, None if val is None else str(val)


# Questions whose answer follows from another row's: the method from the functional, the type of the
# basis set (and keeping a follow-up's) from the basis set. Their answers are still accepted
# (the API's {"method": "HF"}); the form shows one row for each pair.
DERIVED_ROWS = ("method", "basis_type", "basis_keep")
# CLI questions inside the functional's pick: no row of their own (they are "Other questions" otherwise)
_FUNCTIONAL_STEPS = re.compile(r"Select functional category|Select Hartree-Fock method|Select method type", re.I)


def _basis_row(mode: str, qs: List[Question], survey_qs: List[Question], obj: Any, defaults_obj: Any,
               answers: Dict[str, Any], schema_info: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The one Basis set row: internal sets (the CLI's menu for these elements, else its whole table),
    the bundled external sets, and for a follow-up the parent's basis, kept unless another is picked."""
    try:
        import basis_library as BL
    except ImportError:
        return None
    asked = any(re.search(FIELD[k].match, x.text, re.I) for k in ("basis_set", "basis_type", "basis_keep")
                for x in qs + survey_qs)
    if not asked:
        return None
    internal = next((x.options for x in qs + survey_qs
                     if re.search(r"Select internal basis set", x.text, re.I) and x.options), None)
    if internal:
        opts = [{"head": o["head"], "text": o["text"], "num": o["num"], "group": "Internal sets (built into CRYSTAL)"}
                for o in internal]
    else:
        table = (schema_info or {}).get("internal_basis_sets") or []
        opts = [{"head": n, "text": n, "group": "Internal sets (built into CRYSTAL)"} for n in table]
    # the CLI's external menu numbers: 1 double-zeta, 2 triple-zeta, 3 your own directory, 4 the 2022 update
    for num, fam in (("1", "full.basis.doublezeta"), ("2", "full.basis.triplezeta"), ("4", "full.basis.triplezeta.rev2022")):
        opts.append({"head": BL.FAMILY_LABELS[fam], "text": f"{BL.FAMILY_LABELS[fam]} - MACE's bundled {fam} library",
                     "num": num, "group": "External sets (MACE's bundled libraries)"})

    def label(o: Any) -> Optional[str]:
        if o is None:
            return None
        name = json_value(o, "basis_set")
        if not name:
            return None
        kind = json_value(o, "basis_set_type")
        if kind == "EXTERNAL" or "/" in str(name) or str(name).startswith("full.basis") or str(name) == "stuttgart":
            return BL.display_name(str(name))
        return str(name)

    if mode == "opt2d12":
        parent = label(defaults_obj) or label(obj)
        if parent and parent.startswith("EXTERNAL (from"):      # the CLI's word for "the deck's own records"
            parent = None
        opts.insert(0, {"head": KEEP_BASIS, "group": "",
                        "text": KEEP_BASIS + (f" ({parent})" if parent else "")})
        default = KEEP_BASIS
        if "basis_set" in answers:
            value = str(answers["basis_set"])
        elif answers.get("basis_keep") in (False, "n", "no", "No", "false"):
            value = label(obj) or KEEP_BASIS                     # the API's old answer: not kept, the CLI's default set
        else:
            value = KEEP_BASIS
    else:
        value = str(answers["basis_set"]) if "basis_set" in answers else label(obj)
        default = label(defaults_obj)
    f = FIELD["basis_set"]
    full = json_value(obj, "basis_set") if obj is not None else None
    row = {"id": "basis_set", "label": f.label, "section": f.section, "kind": "menu", "unit": "", "hint": f.hint,
           "how": DERIVED_BY_CLI["basis_set"], "question": "Select basis set", "default": default, "value": value,
           "raw": full, "answered": "basis_set" in answers, "options": opts}
    if full and ("/" in str(full)):
        row["detail"] = str(full)               # the whole path, for the tooltip
    row["changed"] = "basis_set" in answers and str(answers["basis_set"]) != str(default)
    return row


# Menus whose levels are presets of d12_constants: SCF tolerances (TOLINTEG, TOLDEE) and the optimization's
# convergence (TOLDEG, TOLDEX, TOLDEE, MAXCYCLE); each option shows its values
PRESET_FIELDS = ("tolerance", "opt_convergence")


def preset_values(fid: str, head: str) -> Optional[str]:
    """'TOLINTEG: 8 8 8 9 24, TOLDEE: 9' for the SCF level named ``head`` (the optimization's for
    opt_convergence), from d12_constants' presets; None for Custom or an unknown name."""
    try:
        import d12_constants as C
    except ImportError:
        return None
    table, describe = ((C.SCF_TOLERANCE_PRESETS, C.describe_scf_preset) if fid == "tolerance"
                       else (C.OPT_CONVERGENCE_PRESETS, C.describe_opt_preset))
    want = re.sub(r"\s+", " ", str(head)).strip().lower()
    return next((describe(level) for level, p in table.items() if p["name"].lower() == want), None)


def option_values(fid: str, option: Dict[str, str]) -> Optional[str]:
    """The records a level of a preset menu sets: a preset's from d12_constants, the parent's own
    (the options opt2d12 adds when the parent's values are none of the presets) from its menu line."""
    vals = preset_values(fid, option["head"])
    if vals is None and option["head"] in parent_options():
        parts = re.split(r"\s+[-–]\s+", option["text"], maxsplit=1)
        vals = parts[1].strip() if len(parts) > 1 else None
    return vals


def parent_options() -> Tuple[str, ...]:
    """The heads of opt2d12's options built from the parent's values: keeping them
    (d12_constants.KEEP_PARENT_OPTION) and FREQ's Very tight with the parent's tighter values
    (FREQ_TIGHTER_OPTION)."""
    try:
        import d12_constants as C
    except ImportError:
        return ()
    return (C.KEEP_PARENT_OPTION, C.FREQ_TIGHTER_OPTION)


def tolerances_described(tolerances: Any) -> Optional[str]:
    """'TOLINTEG: 8 8 8 8 16, TOLDEE: 9', as opt2d12's menu prints them, or None."""
    if not isinstance(tolerances, dict):
        return None
    try:
        import d12_constants as C
    except ImportError:
        return None
    return C.describe_scf_values(tolerances) or None


def preset_named(tolerances: Any) -> Optional[str]:
    """The name of the SCF level whose TOLINTEG and TOLDEE these are, or None."""
    if not isinstance(tolerances, dict):
        return None
    try:
        import d12_constants as C
    except ImportError:
        return None
    ti = " ".join(str(tolerances.get("TOLINTEG", "")).split())
    for p in C.SCF_TOLERANCE_PRESETS.values():
        if ti == " ".join(p["TOLINTEG"].split()) and str(tolerances.get("TOLDEE")) == str(p["TOLDEE"]):
            return p["name"]
    return None


def build_form(mode: str, calc: str, survey: List[Dict[str, Any]], current: Optional[List[Dict[str, Any]]],
               obj: Any, defaults_obj: Any, answers: Dict[str, Any], patches: Dict[str, Any],
               schema_info: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """The Setup form: one row per field whose question the CLI asks for this
    mode and calculation (the survey run, or this run when it went further),
    with the CLI's own options and default, the value the JSON holds, and
    whether it differs from the defaults."""
    qs = [Question(x) for x in (current if current else survey)]
    survey_qs = [Question(x) for x in survey]
    rows: List[Dict[str, Any]] = []
    used = set()
    cats = (schema_info or {}).get("functional_categories") or []
    hf_names = hf_methods()
    forced_basis = {fn: b for c in cats for fn, b in (c.get("basis_requirements") or {}).items()}

    def fn_of(o: Any, default_json: bool = False) -> Optional[str]:
        """The functional a settings JSON (or the answers) names, Hartree-Fock variants included."""
        if default_json or "functional" not in answers:
            if o is None:
                return None
            v = json_value(o, "dft_functional") or json_value(o, "functional") or json_value(o, "hf_method")
            if (json_value(o, "method_type") or json_value(o, "method")) == "HF":
                v = json_value(o, "functional") or json_value(o, "hf_method") or v
            # the D3 suffix is the dispersion row's: the row names the functional itself
            return re.sub(r"-D3$", "", str(v), flags=re.I) if v else None
        return str(answers["functional"])
    now_fn = fn_of(obj)
    is_hf = (now_fn or "") in hf_names or (not now_fn and obj is not None and
                                          (json_value(obj, "method_type") or json_value(obj, "method")) == "HF")
    basis_forced = forced_basis.get(re.sub(r"-D3$", "", now_fn or "", flags=re.I)) if now_fn else None
    method_q = next((x for x in qs if re.search(METHOD_Q, x.text, re.I)), None) or \
        next((x for x in survey_qs if re.search(METHOD_Q, x.text, re.I)), None)
    for f in FIELDS:
        if mode not in f.modes or (f.calcs and calc not in f.calcs):
            continue
        if f.id in DERIVED_ROWS:       # worked out from the functional or the basis set: no row of its own
            for x in qs + survey_qs:
                if _matches(f, x):
                    used.add(id(x))
            continue
        q = next((x for x in qs if _matches(f, x)), None)
        if q is None:
            q = next((x for x in survey_qs if _matches(f, x)), None)
        if q is None and f.id == "functional":
            q = method_q               # a Hartree-Fock run never asks for a functional: its method question stands in
        if q is None:
            continue
        used.add(id(q))
        # a Hartree-Fock method, and a 3c one (which sets its own basis, D3 and grid), has no D3 question and no grid to choose
        if f.id in ("dispersion", "dft_grid") and (is_hf or basis_forced):
            continue
        if f.id == "basis_set":        # the merged basis row, below
            continue
        if f.id == "functional":
            def lab(o: Any, jd: bool):
                v = fn_of(o, jd)
                return v, v
            val, label = lab(obj, False)
            dval, dlabel = lab(defaults_obj, True)
            if label is None:
                val, label = _value_label(f, q, obj)
            if dlabel is None:
                dval, dlabel = _value_label(f, q, defaults_obj)
        else:
            val, label = _value_label(f, q, obj)
            dval, dlabel = _value_label(f, q, defaults_obj)
        row = {"id": f.id, "label": f.label, "section": f.section, "kind": f.kind, "unit": f.unit, "hint": f.hint,
               "how": DERIVED_BY_CLI[f.id], "question": q.last[-160:], "default": dlabel, "value": label,
               "raw": val, "answered": f.id in answers}
        if f.id == "functional":
            # numbered as the CLI lists them: the place in the category's list; Hartree-Fock last, as one more family
            row["options"] = [{"head": fn, "text": fn, "group": c["name"], "num": str(i)} for c in cats if c["key"] != "HF"
                              for i, fn in enumerate(c["functionals"], 1)]
            hf = next((c for c in cats if c["key"] == "HF"), None)
            if hf:
                row["options"] += [{"head": fn, "text": fn, "group": hf["name"], "num": str(i), "family": "hf"}
                                   for i, fn in enumerate(hf["functionals"], 1)]
            row["hint"] = "sets the method and the family too; Hartree-Fock variants are last"
            if forced_basis.get(re.sub(r"-D3$", "", str(label or ""), flags=re.I)):
                row["note"] = f"{label} sets its own basis set ({forced_basis[re.sub(r'-D3$', '', str(label), flags=re.I)]})"
        elif f.kind == "menu":
            row["options"] = [{"head": o["head"], "text": o["text"], "num": o["num"]} for o in q.options]
            if f.id in PRESET_FIELDS:
                # each level with the records it sets (d12_constants, the CLI's own table)
                for o in row["options"]:
                    vals = option_values(f.id, o)
                    if vals:
                        o["values"] = vals
                if f.id == "tolerance":
                    # the level the JSON's TOLINTEG and TOLDEE are (the menu's question default is the parent's):
                    # a preset, an option made from the parent's values (keep them; FREQ's tighter of the
                    # parent's and Very tight), else the values as they are
                    tol = json_value(obj, "tolerances") if obj is not None else None
                    named = preset_named(tol)
                    desc = tolerances_described(tol)
                    made = [o for o in row["options"] if o["head"] in parent_options()]
                    hit = next((o for o in made if desc and o.get("values") == desc), None)
                    if named and any(o["head"] == named for o in row["options"]):
                        row["value"] = label = named
                    elif hit:
                        row["value"] = label = hit["head"]
                    elif desc and any(label == o["head"] for o in made):
                        row["value"] = label = _short_value(val)
                cur = next((o.get("values") for o in row["options"] if o["head"] == label), None)
                if cur:
                    row["hint"] = f"{label}: {cur}"
            if f.id == "scf_method":
                # "DIIS (default - fastest convergence)", "Broyden (alternative ...)": the method's name
                for o in row["options"]:
                    o["head"] = re.sub(r"\s*\(.*$", "", o["head"]).strip() or o["head"]
                heads = {o["num"]: o["head"] for o in row["options"]}
                row["value"] = heads.get(next((o["num"] for o in q.options if o["head"] == label), None), label)
                row["default"] = heads.get(next((o["num"] for o in q.options if o["head"] == dlabel), None), dlabel)
                dlabel = row["default"]
        elif f.kind == "yesno":
            row["options"] = [{"head": "Yes", "text": "Yes"}, {"head": "No", "text": "No"}]
        row["changed"] = f.id in answers and (str(answers[f.id]) != str(dlabel) if f.kind != "yesno"
                                              else (answers[f.id] in (True, "y", "yes", "Yes")) != (dlabel == "Yes"))
        rows.append(row)
    # the basis set, one row: the type (internal or external) and, for a follow-up, keeping the parent's
    # basis follow from the set picked
    if mode in FIELD["basis_set"].modes and not basis_forced:
        brow = _basis_row(mode, qs, survey_qs, obj, defaults_obj, answers, schema_info)
        if brow is not None:
            at = max((i for i, r in enumerate(rows) if r["id"] in ("functional", "dispersion")), default=-1) + 1
            rows.insert(at, brow)
    for key, spec in PATCH_FIELDS.items():
        if mode not in spec["modes"] or spec.get("kind") in ("kpath", "soc"):
            continue
        cur = json_value(obj, key) if obj is not None else None
        rows.append({"id": key, "label": spec["label"], "section": spec["section"], "kind": "value", "how": "patch",
                     "hint": spec["hint"], "value": None if cur is None else (" ".join(map(str, cur)) if isinstance(cur, list) else str(cur)),
                     "default": None, "changed": key in patches, "answered": key in patches, "unit": ""})
    # the questions no field names: "Other questions" (answerable by their own prompt). A run no field
    # covers (opt2d3 TRANSPORT, CHARGE, POTENTIAL, CHARGE+POTENTIAL) lists them before any answer.
    # (TRANSPORT's one field is its k mesh: its other questions are still listed before any answer)
    named = any(mode in f.modes and (not f.calcs or calc in f.calcs) for f in FIELDS) and calc != "TRANSPORT"
    if current or not named:
        # a question asked more than once in a run (CHARGE+POTENTIAL's two blocks) is a row per ask,
        # "q:<text>#<n>", named after the block that asks it
        repeated = {t for run in (survey_qs, qs) for t in {x.last[-120:] for x in run}
                    if sum(1 for x in run if x.last[-120:] == t) > 1}
        seen: Dict[str, int] = {}
        block = None
        for q in qs:
            h = _BLOCK_RE.findall(q.text)
            if h:
                block = h[-1].strip().lower()
            if id(q) in used or not q.last or any(_matches(f, q) for f in FIELDS) or _is_gate(mode, q) \
                    or _EDITOR_QUESTION.search(q.last) or _FUNCTIONAL_STEPS.search(q.last):
                continue
            text = q.last[-120:]
            fid, label = "q:" + text, text.rstrip(": ")
            kind = "menu" if q.options else "yesno" if re.search(r"\[[yY]/[nN]\]", q.last) else "value"
            default, value = q.default, answers.get(fid, q.answer if q.answer else q.default)
            # a menu is named by its heading ("Frequency calculation type", "Charge density type", not
            # "Select type (1-2) [1]:") and shows its options' names, a yes/no Yes or No, as the named
            # fields do
            label = _menu_heading(q) or label
            if kind == "menu":
                heads = {o["num"]: o["head"] for o in q.options}
                default = heads.get(str(default), default)
                value = value if fid in answers else heads.get(str(value), value)
            elif kind == "yesno":
                yn = {"y": "Yes", "n": "No"}
                default = yn.get(str(default).lower(), default)
                value = value if fid in answers else yn.get(str(value).lower(), value)
            if text in repeated:
                seen[text] = seen.get(text, 0) + 1
                fid += f"#{seen[text]}"
                value = answers.get(fid, value)
                if not block:
                    label += f" (asked {seen[text]})"
                elif block.split()[-1] not in label.lower():
                    label += f" ({block})"
            row = {"id": fid, "label": label, "section": "Other questions", "kind": kind,
                   "how": "cli", "question": q.last[-160:], "default": default, "answered": fid in answers,
                   "value": value, "changed": fid in answers, "unit": ""}
            if kind == "menu":
                row["options"] = [{"head": o["head"], "text": o["text"], "num": o["num"]} for o in q.options]
            elif kind == "yesno":
                row["options"] = [{"head": "Yes", "text": "Yes"}, {"head": "No", "text": "No"}]
            rows.append(row)
    # the path format is the automatic path's: a template, labels or vectors typed by hand have none
    if mode == "opt2d3" and calc == "BAND" and "band_path" in answers \
            and not str(answers["band_path"]).lower().startswith("automatic"):
        rows = [r for r in rows if r["id"] != "band_format"]
    return rows


def _menu_heading(q: Question) -> Optional[str]:
    """A menu question's heading: the line before its first option ("Charge density type:")."""
    if not q.options:
        return None
    lines = [x.strip() for x in q.text.splitlines() if x.strip()]
    first = next((i for i, x in enumerate(lines) if x.startswith(f"{q.options[0]['num']}:")), None)
    if not first or lines[first - 1].startswith("==="):
        return None
    head = lines[first - 1].rstrip(":").strip()
    return head[:1].upper() + head[1:] if head else None


# opt2d3's block headers ("=== CHARGE DENSITY CONFIGURATION ==="): what a repeated question belongs to
_BLOCK_RE = re.compile(r"===\s*([A-Za-z][A-Za-z ]*?)\s+CONFIGURATION\s*===")
# opt2d3's "Edit the band k-path?" prompt: Setup has the k-path editor for it (answered No, its default)
_EDITOR_QUESTION = re.compile(r"Edit the band k-path", re.I)


def _is_gate(mode: str, q: Question) -> bool:
    return any(re.search(g["match"], q.text, re.I | re.S) for g in GATES[mode].values()) or \
        bool(re.search(r"Save these options|save.*configuration", q.last, re.I))


# ---------------------------------------------------------------- one Setup run
def _deck_stamp(p: Path):
    st = p.stat()
    return (st.st_mtime_ns, st.st_size, st.st_ino)


def _decks_in(folder: Path, before) -> List[Path]:
    """The decks in folder that this run wrote: new names, or names whose file changed since
    before (a {name: stamp} snapshot; a plain set of names compares by name only). A rerun in
    the same staged tree rewrites a deck under the name an earlier run used."""
    if not folder.is_dir():
        return []
    out = []
    for p in folder.iterdir():
        if not (p.is_file() and p.suffix.lower() in (".d12", ".d3")):
            continue
        if p.name in before and (not isinstance(before, dict) or before[p.name] == _deck_stamp(p)):
            continue
        out.append(p)
    return sorted(out)


def _shrink_of(deck: str) -> Optional[str]:
    lines = deck.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "SHRINK" and i + 1 < len(lines):
            nxt = lines[i + 1].strip()
            more = lines[i + 2].strip() if nxt.split()[:1] == ["0"] and i + 2 < len(lines) else ""
            return (nxt + (" / " + more if more else "")).strip()
    return None


def survey_rules(mode: str) -> List[Dict[str, Any]]:
    """Every gate opened, every question answered with Enter: the list of questions and their menus."""
    return list(GATES[mode].values())


def ask(st: Staged, work: Path, rules: List[Dict[str, Any]], tag: str, ctx=None, extra_args=()) -> Dict[str, Any]:
    """Run the interactive command with these answers: its saved JSON, deck and questions."""
    d = work / f"ask_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    rules_p, tr_p, json_p, out = d / "rules.json", d / "transcript.json", d / "answers.json", d / "out"
    out.mkdir(exist_ok=True)
    rules_p.write_text(json.dumps({"rules": rules, "known": known_questions(st.mode), "max_prompts": 400}))
    args = ask_args(st, json_p, out) + list(extra_args)
    res = clirun.run(clirun.driver_argv(rules_p, tr_p, "--no-banner", *args), st.tree, ctx, timeout=180,
                     env={**clirun.cli_env(), **st.env} if st.env else None)
    try:
        tr = json.loads(tr_p.read_text())
    except (OSError, json.JSONDecodeError):
        tr = {"questions": [], "error": "no transcript"}
    decks = _decks_in(out, set())
    log = res.log.replace(str(st.tree) + os.sep, "").replace(str(d) + os.sep, "")
    return {"code": res.returncode, "json_text": json_p.read_text() if json_p.is_file() else None,
            "deck": decks[0].read_text(errors="replace") if decks else None, "deck_name": decks[0].name if decks else None,
            "questions": tr.get("questions", []), "error": tr.get("error"), "log": log, "elapsed": res.elapsed}


def run_printed(st: Staged, json_text: str, ctx=None, extra_args=()) -> Dict[str, Any]:
    """The printed command, unchanged, in the mirror: the preview deck (P == G)."""
    jp = st.tree / st.json_rel
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json_text)
    out = st.tree / st.out_rel
    before = {p.name: _deck_stamp(p) for p in out.iterdir() if p.is_file()} if out.is_dir() else {}
    args = batch_args(st) + list(extra_args)
    # the printed command is the user's own: only the static k-path switch (a workflow step's stand-in output) reaches it
    env = {**clirun.cli_env(), **{k: "1" for k in ("MACE_STATIC_KPATH", "MACE_STAND_IN_KPATH") if st.env.get(k)}} \
        if st.env.get("MACE_STATIC_KPATH") else None
    res = clirun.run(clirun.mace_argv("--no-banner", *args), st.tree, ctx, timeout=180, env=env)
    decks = _decks_in(out, before)
    log = res.log.replace(str(st.tree) + os.sep, "")
    return {"code": res.returncode, "decks": [{"name": p.name, "text": p.read_text(errors="replace")} for p in decks],
            "log": log, "elapsed": res.elapsed, "args": args}


P1_NOTE_MAX_ATOMS = 2000


def p1_note(cif: Path) -> Optional[str]:
    """Plan T13: a CIF that declares P1 (one operation) while spglib finds a
    higher group at 1e-3 Å, e.g. a structure built in the GUI and saved with
    every atom: cif2d12 keeps P1, which CRYSTAL runs more slowly. Nothing is
    changed; the note offers the symmetrized copy. None when it does not apply
    (or the structure is over 2000 atoms, where spglib is not worth a preview)."""
    try:
        from ase.io.cif import parse_cif
        from mace.utils.cif_symbols import normalize_block
        blocks = [b for b in map(normalize_block, parse_cif(str(cif))) if b.has_structure()]
        if not blocks or len(blocks[-1].get_spacegroup(True).get_symop()) != 1:
            return None
        from . import structures
        p = structures.read_structure(cif)
        if len(p["numbers"]) > P1_NOTE_MAX_ATOMS:
            return None
        y = structures.symmetry(p["cell"], p["frac"], p["numbers"], p["pbc"], symprec=1e-3)
    except Exception:
        return None
    if not y.get("available") or int(y["number"]) <= 1:
        return None
    return (f"This CIF is P1; spglib finds {y['international']} ({y['number']}) at 1e-3 Å. "
            f"cif2d12 keeps P1 (a slower CRYSTAL run).")


_WARN_HEAD = re.compile(r"^(?:\[WARN\]|[^\w\s(\[]|\s)*(?:WARNING|Warning)?\s*:?\s*")


def _plain_warning(line: str) -> str:
    """A CLI warning line as a Problems message: the row's icon already says
    "warning", so the "[WARN]", the emoji and the "WARNING:" go."""
    return _WARN_HEAD.sub("", line).strip()


def problems_from(log: str, source: str = "Setup", limit: int = 400) -> List[Dict[str, Any]]:
    out = []
    for line in log.splitlines():
        s = line.strip()
        if s.startswith("[WARN]") or s.startswith("WARNING") or s.startswith("Warning:"):
            msg = _plain_warning(s)
            if msg:
                out.append({"severity": "warning", "source": source, "message": msg[:limit]})
    seen, uniq = set(), []
    for p in out:
        if p["message"] not in seen:
            seen.add(p["message"])
            uniq.append(p)
    return uniq[:20]


_DOSS_OFFER = re.compile(r"(This DOS needs \d+ points for a [\d.]+ eV step over all bands \(~\d+ eV, including core "
                         r"levels\))(?:\.\s*(No window around E_F (?:is offered|can be written): [^\n]*))?")


def doss_window_note(form: List[Dict[str, Any]], log: str, problems: List[Dict[str, Any]]) -> None:
    """opt2d3's warning that an all-bands DOS needs many points, on the Resolution row (its note) and in
    Problems pointing at it; the Window around E_F row below it (a 1D material's Bands by index) is the
    one-click choice."""
    m = _DOSS_OFFER.search(log or "")
    if not m:
        return
    banded = re.search(r"Use bands (\d+) to (\d+) by index", log or "")
    note = m.group(1) + (f". {m.group(2)}" if m.group(2) else
                         f": the deck covers bands {banded.group(1)} to {banded.group(2)} by index, a 1D material's "
                         f"default (Bands by index; No keeps every band)" if banded else
                         ": the deck covers a window around E_F (Window around E_F; No keeps every band)")
    for row in form:
        if row.get("id") == "doss_resolution":
            row["note"] = note
    if not any(p.get("field") == "doss_resolution" for p in problems):
        problems[:] = [p for p in problems if not str(p.get("message", "")).startswith("This DOS needs")]
        problems.append({"severity": "warning", "source": "Setup", "field": "doss_resolution", "message": note})


def parent_refusals(path: Optional[Path]) -> Dict[str, str]:
    """{calc type: why} for the follow-ups this parent cannot have, which
    Setup grays out: ORBITALS from a two-component (SOC) run, on which
    CRYSTAL23 crashes (mace.orbitals.kchoice.SOC_REFUSAL)."""
    if path is None or is_deck(path):
        return {}
    from mace.utils.calc_detection import is_two_component_run
    if not is_two_component_run(path):
        return {}
    from mace.orbitals.kchoice import SOC_REFUSAL
    return {"ORBITALS": SOC_REFUSAL}


def parent_info(mode: str, input_rel: str, problems: List[Dict[str, Any]],
                path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Where a follow-up's values come from: {kind: "out" | "d12", name, warning, refused}.
    ``warning``: what the CLI said about the parent itself, e.g. an unfinished
    optimization (opt2d12 prints "<name>: unfinished optimization ..., so the
    deck has its starting geometry"; the deck is the CLI's either way).
    ``refused``: parent_refusals of ``path`` (the staged parent), when given."""
    if mode == "cif2d12":
        return None
    name = Path(input_rel).name
    warning = next((p["message"][len(name) + 2:] for p in problems
                    if p.get("severity") == "warning" and p["message"].startswith(name + ": ")), None)
    return {"kind": "d12" if is_deck(name) else "out", "name": name, "warning": warning,
            "refused": parent_refusals(path)}


def refusal(log: str) -> str:
    """The CLI's own words when it wrote no deck: its error lines, else the log's end."""
    lines = [x.strip() for x in log.splitlines() if x.strip()]
    errs, follows = [], False
    for x in lines:
        if re.match(r"^(\[ERR\]|\[ERROR\]|ERROR|Error|\[X\])", x) or "Aborting" in x or "requires" in x.lower():
            errs.append(x)
            follows = True
        elif follows and x.startswith("- "):
            errs.append(x)          # a refusal's listed changes ("a SOC deck needs N changes:")
        else:
            follows = False
    # opt2d3's closing "No D3 file was generated." says less than the reason before it
    told = [x for x in errs if not re.search(r"No D3 file was generated\.?$", x)]
    errs = told or errs
    return "\n".join(errs[-6:]) if errs else "\n".join(lines[-12:])


# ---------------------------------------------------------------- charged systems: the CLI's switches
# convert (cif2d12) and opt2d12 take the same charged-system switches (their
# "charged systems" argument group); opt2d3 takes none. Setup passes them as
# switches only, on the command it runs and on the one it prints, and leaves
# every check of what they mean to the CLI (its refusal is shown as it is).
# (key, switch, kind, choices, label, hint); the order is the order on the command line.
CHARGE_FLAGS = (
    ("ion_charges", "--ion-charges", "choice", ("auto",), "Find ions",
     "Find known ions (Li+, FSI-, ...) by connectivity and give them their formal charges"),
    ("net_charge", "--net-charge", "number", None, "Net charge",
     "Net charge of the system (default: the sum of the ions found)"),
    ("site_states", "--site-states", "json", None, "Site states",
     'Per-atom shell charges as JSON, e.g. [{"atom": 1, "shell_charges": [2, 0, 0, 0, 0]}]; needs a net charge'),
    ("chemod", "--chemod", "choice", ("auto", "yes", "no"), "CHEMOD",
     "Write CHEMOD (the CLI's default: auto)"),
    ("fragment_bond_scale", "--fragment-bond-scale", "number", None, "Fragment bond scale",
     "With Find ions: two atoms bond when closer than scale x the sum of their covalent radii (default 1.15)"),
    ("spinlock", "--spinlock", "int", None, "SPINLOCK",
     "nalpha-nbeta for an odd electron count"),
    ("spinlock_cycles", "--spinlock-cycles", "int", None, "SPINLOCK cycles",
     "SCF cycles the SPINLOCK holds"),
)
CHARGE_MODES = ("cif2d12", "opt2d12")
# atom numbers belong to one structure: a batch of many takes every other switch
BATCH_CHARGE_FLAGS = tuple(f[0] for f in CHARGE_FLAGS if f[0] != "site_states")
SITE_STATES_MAX = 20000


def charge_flags(raw: Any, allowed: Optional[Tuple[str, ...]] = None) -> Dict[str, Any]:
    """The charged-system switches of a request, by key ({} for none). Only
    their shape is checked here (a number is a number, a choice is one of the
    CLI's, site states are the JSON itself, never a file name); what they mean
    is the CLI's to check."""
    if raw in (None, ""):
        return {}
    keys = allowed or tuple(f[0] for f in CHARGE_FLAGS)
    if not isinstance(raw, dict) or set(raw) - set(f[0] for f in CHARGE_FLAGS):
        raise SetupError("flags are the CLI's charged-system switches: "
                         + ", ".join(f[1] for f in CHARGE_FLAGS if f[0] in keys))
    out: Dict[str, Any] = {}
    for key, switch, kind, choices, _, _ in CHARGE_FLAGS:
        v = raw.get(key)
        if key not in keys and v not in (None, "", False):
            raise SetupError(f"{switch} names atoms of one structure: set it in Calculation Setup on that file")
        if v is None or v == "" or v is False:
            continue
        if kind == "choice":
            if v not in choices:
                raise SetupError(f"{switch} is one of {', '.join(choices)}")
        elif kind in ("number", "int"):
            if isinstance(v, bool):
                raise SetupError(f"{switch} is a number")
            try:
                num = float(v)
            except (TypeError, ValueError):
                raise SetupError(f"{switch} is a {'whole ' if kind == 'int' else ''}number")
            if num != num or num in (float("inf"), float("-inf")):
                raise SetupError(f"{switch} is a number")
            if kind == "int" and not num.is_integer():
                raise SetupError(f"{switch} is a whole number")
            v = int(num) if num.is_integer() else num
        else:                                   # site states: the JSON text itself
            if not isinstance(v, str) or len(v) > SITE_STATES_MAX:
                raise SetupError(f"{switch} is JSON text of at most {SITE_STATES_MAX} characters")
            v = v.strip()
            if not v.startswith("["):
                raise SetupError(f'{switch} is the JSON list itself ([{{"atom": 1, "shell_charges": [...]}}]); '
                                 "Setup does not read it from a file")
        out[key] = v
    return out


def flag_args(flags: Dict[str, Any], mode: str) -> List[str]:
    """The switches as the command line takes them (none for opt2d3)."""
    if mode not in CHARGE_MODES or not flags:
        return []
    out: List[str] = []
    for key, switch, _, _, _, _ in CHARGE_FLAGS:
        if key in flags:
            out += [switch, str(flags[key])]
    return out


# --static-kpath: opt2d3's BAND (and opt2d12's phonon dispersion) writes the static HPKOT table path of the
# space group where the seekpath package is missing (it stops there without the switch); with the package
# installed SeeK-path's path is computed either way
STATIC_KPATH_SWITCH = "--static-kpath"
STATIC_KPATH_MODES = ("opt2d3",)


def charge_schema() -> List[Dict[str, Any]]:
    return [{"id": k, "switch": s, "kind": kind, "choices": list(c) if c else None, "label": label, "hint": hint,
             "batch": k in BATCH_CHARGE_FLAGS} for k, s, kind, c, label, hint in CHARGE_FLAGS]


# ---------------------------------------------------------------- spin-orbit coupling (SOC)
def _soc():
    import soc_ecp
    return soc_ecp


_ELEMENTS_CACHE: "OrderedDict[Tuple[str, int, int], Tuple[int, ...]]" = OrderedDict()


def _atomic_numbers(p: Path):
    """The elements of a structure file (CRYSTAL's ECP numbering 2xx read as xx), cached by file stamp;
    a string (why) when the file cannot be read, so a caller never takes "unread" for "no elements"."""
    try:
        s = p.stat()
    except OSError as e:
        return f"cannot read {p.name}: {e.strerror or e}"
    key = (str(p), s.st_mtime_ns, s.st_size)
    hit = _ELEMENTS_CACHE.get(key)
    if hit is None:
        from . import structures
        try:
            hit = tuple(sorted({int(z) % 100 or int(z) for z in structures.read_structure(p)["numbers"]}))
        except ModuleNotFoundError as e:
            hit = (f"cannot read the elements of {p.name}: install {e.name or 'ase'} (pip install "
                   f"{e.name or 'ase'}); MACE needs it to read this file")
        except Exception as e:      # noqa: BLE001 - reported to the caller, never read as "no elements"
            hit = f"cannot read the elements of {p.name}: {e}"
        _ELEMENTS_CACHE[key] = hit
        while len(_ELEMENTS_CACHE) > 4096:
            _ELEMENTS_CACHE.popitem(last=False)
    return hit


def stutsc_refusals(paths: Iterable[Path], name=None) -> Dict[str, Any]:
    """Which of ``paths`` hold an element whose STUTSC the CLI refuses (soc_ecp.py's table and words):
    ``{"files": n, "refused": [{"z", "symbol", "why", "files": [names]}], "all": every file has one,
    "unread": [{"file", "why"}]}``. A file whose elements cannot be read (no ase for a CIF) is listed in
    ``unread``, never taken as refusing nothing: STUTSC cannot be checked for it.
    Batch Setup and the workflow builder gray STUTSC out with it, as Setup does for its structure."""
    E = _soc()
    from mace.utils.element_palettes import SYMBOLS
    paths = list(paths)
    name = name or (lambda p: p.name)
    by_z: Dict[int, List[str]] = {}
    hit_files = set()
    unread: List[Dict[str, str]] = []
    for p in paths:
        zs = _atomic_numbers(p)
        if isinstance(zs, str):
            unread.append({"file": name(p), "why": zs})
            continue
        for z in zs:
            if z in E.STUTSC_NOT_THE_PUBLISHED_ECP:
                by_z.setdefault(z, []).append(name(p))
                hit_files.add(str(p))
    refused = [{"z": z, "symbol": SYMBOLS[z - 1] if 0 < z <= len(SYMBOLS) else f"Z={z}",
                "why": E.STUTSC_NOT_THE_PUBLISHED_ECP[z], "files": names} for z, names in sorted(by_z.items())]
    return {"files": len(paths), "refused": refused, "all": bool(paths) and len(hit_files) == len(paths),
            "unread": unread}


def soc_schema() -> Dict[str, Any]:
    """The SOC section's tables, all the CLI's own (Crystal_d12/soc_ecp.py,
    CRYSTALOptToD3): the keys, the ECP routes, the elements whose STUTSC is
    refused and why, the SMEAR width the HPCC Au runs converged with, and the
    words the CLI refuses a SOC deck with for the calculations it is not for."""
    E = _soc()
    not_for = {"OPT": E.unsupported_message("OPTGEOM"), "FREQ": E.unsupported_message("FREQCALC")}
    try:
        import CRYSTALOptToD3 as D3
        two = [c for c in CALC_TYPES["opt2d3"] if c in D3.TWO_COMPONENT_D3_TYPES]
    except Exception:
        two = ["BAND", "DOSS", "CHARGE"]
    prop = ("A properties deck has no SOC setting of its own: it reads the wavefunction of its parent run. "
            f"From a two-component (SOC) run opt2d3 writes {', '.join(two)} and refuses the others "
            "(CRYSTAL23 manual sec. 6.4, p. 183).")
    for c in CALC_TYPES["opt2d3"]:
        not_for[c] = prop
    return {"keys": list(SOC_KEYS), "calcs": list(SOC_CALCS), "modes": list(SOC_MODES),
            "ecp_forms": list(E.ECP_FORMS), "soc_basis_families": list(E.SOC_BASIS_FAMILIES),
            "guesses": list(E.SOC_GUESSES),
            "stutsc_refused": {str(z): why for z, why in E.STUTSC_NOT_THE_PUBLISHED_ECP.items()},
            "metal_smear": E.SOC_METAL_SMEAR_EXAMPLE, "fmixing": E.SOC_FMIXING,
            "restart_fmixing": E.SOC_RESTART_FMIXING, "not_for": not_for,
            "properties_from_2c": two}


def check_soc_value(key: str, value: Any) -> None:
    """A SOC patch value, checked by the CLI's own checks (None removes the key)."""
    if value is None:
        return
    E = _soc()
    try:
        if key == "soc":
            if not isinstance(value, bool):
                raise SetupError("soc is true or false")
        elif key == "soc_ecp":
            E._checked_ecp_form(value)
        elif key == "soc_basis":
            E._checked_soc_basis(value)
        elif key == "soc_guess":
            E._checked_guess(value)
        elif key == "smear":
            if isinstance(value, bool):
                raise E.SocError(f"smear must be a width in hartree above 0, not {value!r}")
            E.checked_smear(value)
    except E.SocError as exc:
        raise SetupError(str(exc))


_NOT_WRITING = re.compile(r"Not writing \S+?\.d12: (.+)$")


def soc_refusal(log: str) -> Optional[str]:
    """The reason the CLI gave for not writing a deck ("Not writing <deck>: <reason>"), as it said it."""
    lines = log.splitlines()
    for k, line in enumerate(lines):
        m = _NOT_WRITING.search(line.strip())
        if m:
            said = [m.group(1).strip()]
            # a refusal that lists several changes goes on in "  - ..." lines
            for more in lines[k + 1:]:
                if not more.strip().startswith("- "):
                    break
                said.append(more.strip())
            return "\n".join(said)
    return None


def _soc_warning(message: str) -> bool:
    return bool(re.search(r"\bSOC\b|SMEAR|GUESSP|two-component", message))


def soc_guesses(mode: str, source_path: Optional[Path]) -> List[Dict[str, Any]]:
    """The starting guesses this parent allows: from scratch always; GUESSPNOSO
    (a spin-polarized 1c parent without symmetry) and GUESSPSO (a 2c parent),
    each checked by opt2d12's own soc_restart_plan, read-only, with its words
    when the parent does not allow it."""
    E = _soc()
    out: List[Dict[str, Any]] = [{"value": None, "ok": True, "reason": None}]
    if mode != "opt2d12" or source_path is None:
        return out + [{"value": g, "ok": False, "reason": E.NO_PARENT_FOR_GUESS} for g in E.SOC_GUESSES]
    import contextlib
    import io
    import CRYSTALOptToD12 as O
    from mace.utils.calc_detection import is_two_component_deck, is_two_component_output
    deck_parent = is_deck(source_path.name)
    d12 = source_path if deck_parent else source_path.with_suffix(".d12")
    two_c = False
    try:
        two_c = (is_two_component_deck(d12.read_text(errors="replace")) if d12.is_file() else False) or (
            not deck_parent and is_two_component_output(source_path.read_text(errors="replace")))
    except OSError:
        pass
    # where the deck would go: a name nobody has (soc_restart_plan only stats it)
    child = Path(tempfile.gettempdir()) / f"mace-soc-check-{os.getpid()}" / "child.d12"
    for g in E.SOC_GUESSES:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            problem = O.soc_restart_plan({"soc": True, "soc_guess": g, "two_component": two_c},
                                         None if deck_parent else str(source_path),
                                         str(d12) if d12.is_file() else None, str(child))[0]
        out.append({"value": g, "ok": problem is None, "reason": problem})
    return out


def soc_state(mode: str, calc: str, obj: Any, defaults_obj: Any, log: str, guesses: List[Dict[str, Any]],
              metal: Optional[str]) -> Dict[str, Any]:
    """The SOC section of one Setup run: whether it applies (a single point),
    the keys the settings file has and the parent's (or the defaults') file
    had, the starting guesses the parent allows, and from the printed
    command's run its "SOC deck:" lines and its refusal, in its words."""
    values = {k: obj[k] for k in SOC_KEYS if isinstance(obj, dict) and k in obj}
    base = {k: defaults_obj[k] for k in SOC_KEYS if isinstance(defaults_obj, dict) and k in defaults_obj}
    on = values.get("soc") is True
    return {"applies": mode in SOC_MODES and calc in SOC_CALCS, "values": values, "base": base,
            "guesses": guesses, "metal": metal,
            "changes": [ln.strip() for ln in log.splitlines() if ln.strip().startswith("SOC deck:")] if on else [],
            "refusal": soc_refusal(log) if on else None}


# ---------------------------------------------------------------- the whole run
class Request:
    """What one Setup run is asked for (from POST /api/setup/run)."""

    def __init__(self, body: Dict[str, Any]):
        src = body.get("source") or {}
        if not isinstance(src, dict):
            raise SetupError("source is {path} or {session, rev}")
        self.source = src
        self.calc_type = body.get("calc_type")
        self.mode = body.get("mode")
        answers = body.get("answers") or {}
        patches = body.get("patches") or {}
        if not isinstance(answers, dict) or not isinstance(patches, dict):
            raise SetupError("answers and patches are objects {field: value}")
        self.answers = OrderedDict((str(k), answers[k]) for k in answers)
        for k, v in self.answers.items():
            if not isinstance(v, (str, int, float, bool)) or (isinstance(v, str) and len(v) > 200):
                raise SetupError(f"the answer for {k} must be a short text, number or yes/no")
        self.patches = OrderedDict((str(k), patches[k]) for k in patches)
        for k in self.patches:
            if k not in PATCH_FIELDS:
                raise SetupError(f"{k} is not a setting written into the JSON by hand")
            if k in SOC_KEYS:
                check_soc_value(k, self.patches[k])
        hj = body.get("json_text")
        if hj is not None and (not isinstance(hj, str) or len(hj) > 1_000_000):
            raise SetupError("json_text must be the settings file's text")
        self.json_text = hj
        self.preview = body.get("preview", True) is not False
        # --static-kpath (opt2d3, opt2d12): the static table path of the space group instead of SeeK-path's
        self.static_kpath = body.get("static_kpath") is True
        # a workflow step's form on a stand-in output (no real structure to run SeeK-path on): the same path,
        # by the environment (the printed command is the user's own, without it)
        self.stand_in_kpath = body.get("stand_in_kpath") is True
        # the CLI's own charged-system switches (--net-charge, --chemod, ...; charge_flags above)
        self.flags = charge_flags(body.get("flags"))
        self.form = body.get("form", True) is not False
        # the GUI's Functional / Basis set settings (session > folder > user), the CLI's
        # defaults for a deck without a parent: {"functional": "", "basis_set": ""}
        sd = body.get("defaults") or {}
        if not isinstance(sd, dict) or set(sd) - {"functional", "basis_set"} or any(
                not isinstance(v, str) or len(v) > 60 for v in sd.values()):
            raise SetupError("defaults is {functional, basis_set}: short names")
        self.defaults = {k: sd[k].strip() for k in ("functional", "basis_set") if k in sd}


def source_key(mode: str, path: Path) -> Tuple:
    """What the staged input depends on: the file and the siblings the CLI reads, by stat."""
    def stamp(p: Path):
        st = os.stat(p)
        return (p.name, int(st.st_mtime_ns), int(st.st_size))
    parts = [stamp(path)]
    if mode == "opt2d12":
        d12 = path.with_suffix(".d12")
        if d12.is_file():
            parts.append(stamp(d12))
        f9 = path.with_suffix(".f9")
        if f9.is_file() and not is_deck(path):
            parts.append(stamp(f9))
    elif mode == "opt2d3":
        parts += [stamp(p) for p in _d12_globs(path)]
        parts += [("sibling", n) for n in _anatase_siblings(path)]
        f9 = find_wavefunction(path)
        parts.append(("f9", f9["name"], f9["bytes"]) if f9 else ("f9", None))
    return (str(path.resolve()),) + tuple(parts)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def setup_env(defaults: Dict[str, str]) -> Dict[str, str]:
    """The CLI's environment for these GUI defaults: MACE_SETUP_FUNCTIONAL and
    MACE_SETUP_BASIS_SET (blank = the CLI's own default), and no folder file of
    its own (the GUI's value already is the folder's, the user's or this
    session's). Empty when the page sent none (a page from before 1.1.6)."""
    if not defaults:
        return {}
    from mace.utils import folder_settings
    env = {folder_settings.ENV: "none"}
    for key, var in folder_settings.SETUP_ENV.items():
        env[var] = defaults.get(key.split(".", 1)[1], "")
    return env


def run_setup(req: Request, mode: str, calc: str, work: Path, source_path: Optional[Path], rel: Optional[str],
              src_key: Any, cache_get: Callable[[Any], Any], cache_put: Callable[[Any, Any, int], None], ctx=None,
              cif_bytes: Optional[bytes] = None, cif_name: Optional[str] = None,
              schema_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Stage, ask the CLI (its defaults, its questions, these answers), write
    the patches, then run the printed command on the result. Each CLI run's
    answer is cached by (input, mode, calc type, answers, CLI sources)."""
    t0 = time.monotonic()

    def progress(f, msg):
        if ctx is not None:
            ctx.progress(f, msg)

    progress(0.05, "Staging the input")
    st = stage(mode, calc, work, source_path, rel, cif_bytes, cif_name)
    st.env = setup_env(req.defaults)
    if req.stand_in_kpath:
        st.env = {**st.env, "MACE_STATIC_KPATH": "1", "MACE_STAND_IN_KPATH": "1"}
    base_key = (src_key, mode, calc, clirun.cli_hash(), sorted(st.env.items()))
    cmd_name = {"cif2d12": "mace convert", "opt2d12": "mace opt2d12", "opt2d3": "mace opt2d3"}[mode]

    def cached_ask(tag, rules, frac, msg, extra=()):
        key = ("setup-ask", base_key, json.dumps(rules, sort_keys=True), list(extra))
        hit = cache_get(key)
        if hit is None:
            progress(frac, msg)
            hit = ask(st, work, rules, tag, ctx, extra)
            cache_put(key, hit, len(json.dumps(hit)))
        if hit.get("error"):
            # a question no rule or known question names (a reworded prompt), a menu line not found:
            # stopped unanswered, never left at a default nobody chose
            raise SetupError(f"{cmd_name} did not ask what Setup expected: {hit['error']}. "
                             f"Set it through the CLI: {cmd_name} …")
        return hit

    if mode == "cif2d12" and calc != DEFAULT_CALC[mode] and "calc_type" not in req.answers:
        # convert asks for the calculation type: SP and FREQ are answers, like any other
        req.answers = OrderedDict([("calc_type", calc)] + list(req.answers.items()))
    if req.json_text is not None and not req.form:
        defaults = survey = {"json_text": None, "questions": [], "log": ""}        # a hand-written file: the deck only
    else:
        defaults = cached_ask("defaults", list(DEFAULT_RULES[mode]), 0.15, f"Asking {cmd_name} for its defaults")
        survey = defaults if mode == "opt2d3" else cached_ask("survey", survey_rules(mode), 0.25,
                                                              f"Reading {cmd_name}'s questions")
    problems: List[Dict[str, Any]] = []
    current = None
    asked = None
    charge = flag_args(req.flags, mode)          # the charged-system switches, on every command
    if req.static_kpath and mode in STATIC_KPATH_MODES:
        charge = charge + [STATIC_KPATH_SWITCH]
    if req.json_text is not None:
        final = req.json_text
        source_of_json = "hand"
    elif req.answers or req.flags or charge:
        asked = cached_ask("answers", answer_rules(mode, req.answers), 0.35, f"Answering {cmd_name}'s questions", charge)
        final = asked.get("json_text")
        current = asked.get("questions")
        source_of_json = "answers"
    else:
        final = defaults.get("json_text")
        source_of_json = "defaults"
    base_run = asked if asked is not None else defaults
    if final is None:
        why = refusal(base_run.get("log") or "")
        out = _refused(mode, calc, st, why, base_run, defaults, survey, schema_info, req, charge)
        out["elapsed"] = round(time.monotonic() - t0, 3)
        return out
    final, changed = apply_patch(final, req.patches)
    bad = validate(mode, final)
    if bad:
        raise SetupError("; ".join(bad))
    try:
        obj = json.loads(final)
    except json.JSONDecodeError:
        obj = None
    try:
        defaults_obj = json.loads(defaults["json_text"]) if defaults.get("json_text") else None
    except json.JSONDecodeError:
        defaults_obj = None
    form = build_form(mode, calc, survey.get("questions") or [], current, obj, defaults_obj, req.answers, req.patches,
                      schema_info) if req.form else []
    args = batch_args(st) + charge
    result: Dict[str, Any] = {
        "mode": mode, "calc_type": calc, "json_text": final, "json_sha256": _sha(final),
        "defaults_sha256": _sha(defaults["json_text"]) if defaults.get("json_text") else None,
        "defaults_json": defaults.get("json_text"),
        "json_source": source_of_json, "patched": changed, "form": form,
        "command": command_line(args), "argv": args, "json_rel": st.json_rel, "input_rel": st.input_rel,
        "f9": st.f9, "notes": st.notes, "problems": problems, "deck": None, "deck_name": None, "decks": [],
        "transcript": [{"q": Question(q).last[-160:], "a": q.get("answer")} for q in (current or [])][-60:],
    }
    problems += problems_from(base_run.get("log") or "")
    if mode == "opt2d3" and calc == "DOSS":
        doss_window_note(result.get("form") or [], base_run.get("log") or "", problems)
    result["parent"] = parent_info(mode, st.input_rel, problems, st.tree / st.input_rel)
    if mode == "opt2d3" and st.f9:
        problems.append({"severity": "info", "source": "Setup", "field": None,
                         "message": f"Preview used an empty stand-in for {st.f9['name']} ({_mb(st.f9['bytes'])}); the deck does not "
                                    f"read it. Running the command copies it to {d3_base(Path(st.input_rel).stem)}_"
                                    f"{calc.lower()}.f9."})
    for n in st.notes:
        problems.append({"severity": "info", "source": "Setup", "message": n})
    if mode == "cif2d12":
        note = cache_get(("setup-p1", base_key))
        if note is None:
            note = p1_note(st.tree / st.input_rel) or ""
            cache_put(("setup-p1", base_key), note, 256)
        if note:
            # the CLI's bare "Space group mismatch!" says the same thing, with less
            problems[:] = [p for p in problems if not p["message"].lower().startswith("space group mismatch")]
            problems.append({"severity": "info", "source": "Setup", "message": note, "action_id": "save_symmetrized"})
    if req.preview:
        key = ("setup-preview", base_key, _sha(final), charge)
        p = cache_get(key)
        if p is None:
            progress(0.6, f"Running {command_line(args)}")
            p = run_printed(st, final, ctx, charge)
            cache_put(key, p, len(json.dumps(p)))
        progress(0.9, "Reading the deck")
        if p["code"] != 0 or not p["decks"]:
            why = refusal(p["log"])
            result.update(deck=None, deck_name=None, log=p["log"][-6000:], refused=why)
            problems.insert(0, {"severity": "error", "source": "Setup", "message": f"The CLI wrote no deck: {why.splitlines()[-1] if why else ''}"})
        else:
            main = p["decks"][0]
            result.update(deck=main["text"], deck_name=main["name"], decks=p["decks"], log=p["log"][-6000:])
            if mode == "cif2d12":
                result["shrink"] = _shrink_of(main["text"])
            # the control route: the interactive run's own deck against the printed command's
            ctrl = base_run.get("deck")
            if source_of_json != "hand" and not req.patches and ctrl is not None:
                same = ctrl == main["text"]
                result["control_same"] = same
                if not same:
                    problems.append({"severity": "warning", "source": "Setup",
                                     "message": f"{cmd_name} writes a different deck when it asks than from its saved "
                                                "settings (a CLI finding: the saved-settings deck is shown)"})
    if mode in SOC_MODES:
        _soc_section(result, mode, calc, obj, defaults_obj, p if req.preview else None, source_path, st,
                     base_key, cache_get, cache_put, problems)
    if mode == "opt2d3" and calc == "BAND" and req.preview:
        kp_info = _kpath_info(st, final, req, result, base_key, cache_get, cache_put, ctx, source_path, charge)
        if kp_info is not None:
            result["kpath"] = kp_info
            for e in kp_info["errors"]:
                problems.insert(0, {"severity": "error", "source": "Setup", "field": "band_path", "message": f"k-path: {e}"})
            for w in kp_info["warnings"]:
                problems.append({"severity": "warning", "source": "Setup", "field": "band_path", "message": f"k-path: {w}"})
    result["elapsed"] = round(time.monotonic() - t0, 3)
    return result


def _soc_section(result, mode, calc, obj, defaults_obj, p, source_path, st, base_key, cache_get, cache_put,
                 problems) -> None:
    """result["soc"] (soc_state), and the CLI's SOC warnings and refusal as
    Problems on the SOC section (field "soc"): the preview run's own lines."""
    key = ("setup-soc", base_key)
    hit = cache_get(key)
    if hit is None:
        metal = None
        if mode == "opt2d12" and source_path is not None and not is_deck(source_path.name):
            import CRYSTALOptToD12 as O
            metal = O.soc_parent_metal(str(source_path))
        hit = {"guesses": soc_guesses(mode, source_path), "metal": metal}
        cache_put(key, hit, len(json.dumps(hit)))
    log = (p or {}).get("log") or ""
    soc = soc_state(mode, calc, obj, defaults_obj, log, hit["guesses"], hit["metal"])
    result["soc"] = soc
    if soc["values"].get("soc") is not True:
        return
    if soc["refusal"]:
        problems[:] = [x for x in problems if not (x["severity"] == "error" and x["message"].startswith("The CLI wrote no deck"))]
        problems.insert(0, {"severity": "error", "source": "Setup", "field": "soc",
                            "message": f"The CLI wrote no deck: {soc['refusal']}"})
    seen = {x["message"] for x in problems}
    # the CLI's SOC warnings whole: the measured numbers are at their end
    for w in problems_from(log, limit=4000):
        if _soc_warning(w["message"]) and w["message"] not in seen:
            problems.append({**w, "field": "soc"})
            seen.add(w["message"])


def _band_path_module():
    try:
        import d3_band_path
    except ImportError:
        _sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Crystal_d3"))
        import d3_band_path
    return d3_band_path


def _kpath_info(st, final: str, req: "Request", result: Dict[str, Any], base_key, cache_get, cache_put, ctx,
                source_path=None, charge=()):
    """The k-path editor's data: the segments the unedited settings compute (from that deck), the
    edit (band_path) and what is wrong with it. The unedited deck is the preview without band_path
    (cached like any preview: usually the one shown before the first edit)."""
    kp = _band_path_module()
    edit = req.patches.get("band_path")
    deck = result.get("deck")
    if edit is not None:
        plain_json, _ = apply_patch(final, {"band_path": None})
        # the main preview's key (charge included): usually the preview shown before the first edit
        key = ("setup-preview", base_key, _sha(plain_json), charge)
        p = cache_get(key)
        if p is None:
            p = run_printed(st, plain_json, ctx, charge)
            cache_put(key, p, len(json.dumps(p)))
        deck = p["decks"][0]["text"] if p.get("code") == 0 and p.get("decks") else None
    parsed = kp.parse_band_deck(deck) if deck else None
    if not parsed or not parsed["segments"]:
        return None
    # a slab or a polymer: added points keep their non-periodic k components 0 (manual p. 309, note 3)
    dim = _kpath_dimensionality(source_path)
    errors, warnings = (kp.validate(edit, parsed["segments"], need_k=not parsed["labels_mode"], dimensionality=dim)
                        if edit is not None else ([], []))
    return {"computed": parsed["segments"], "edited": edit, "errors": errors, "warnings": warnings,
            "labels_mode": parsed["labels_mode"], "dimensionality": dim}


def _kpath_dimensionality(source_path) -> int:
    """3, 2 (slab), 1 (polymer) or 0 of the parent .out, as opt2d3 reads it (3 when unknown)."""
    if source_path is None:
        return 3
    try:
        import d3_kpoints
    except ImportError:
        _sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Crystal_d3"))
        import d3_kpoints
    return d3_kpoints.output_dimensionality(str(source_path))


def _mb(n: int) -> str:
    return f"{n / 2**20:.0f} MB" if n >= 2**20 else f"{n / 1024:.0f} KB" if n >= 1024 else f"{n} B"


def _refused(mode, calc, st, why, base_run, defaults, survey, schema_info, req, charge=()) -> Dict[str, Any]:
    """The CLI wrote no settings: its refusal is the answer (no f9, no seekpath, ...)."""
    msg = why.splitlines()[-1] if why else "the CLI stopped"
    if mode == "opt2d3" and st.f9 is None and re.search(r"wavefunction|fort\.9", base_run.get("log") or "", re.I):
        name = Path(st.input_rel).name
        stem = Path(st.input_rel).stem
        msg = (f"opt2d3 writes no {calc} deck without {d3_base(stem)}_sp.f9 (or fort.9) beside {name} "
               f"(the wavefunction of the SCF run)")
    args = batch_args(st) + list(charge)
    return {"mode": mode, "calc_type": calc, "json_text": None, "json_sha256": None, "form": [], "refused": why,
            "deck": None, "deck_name": None, "decks": [], "log": (base_run.get("log") or "")[-6000:],
            "command": command_line(args), "argv": args, "json_rel": st.json_rel, "input_rel": st.input_rel,
            "f9": st.f9, "notes": st.notes, "save_blocked": msg, "parent": parent_info(mode, st.input_rel, [], st.tree / st.input_rel),
            "problems": [{"severity": "error", "source": "Setup", "message": msg}]}
