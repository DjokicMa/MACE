#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRYSTAL17/23 Optimization Output to D12 Converter
-------------------------------------------------
This script extracts optimized geometries from CRYSTAL17/23 output files
and creates new D12 input files for follow-up calculations.

DESCRIPTION:
    Takes the optimized geometry from CRYSTAL17/23 output files and creates
    new D12 input files with updated coordinates. The script attempts to
    preserve the original calculation settings and allows the user to
    modify them interactively.

USAGE:
    1. Single file processing:
       python CRYSTALOptToD12.py --out-file file.out --d12-file file.d12

    2. Process all files in a directory:
       python CRYSTALOptToD12.py --directory /path/to/files

    3. Batch processing with shared settings:
       python CRYSTALOptToD12.py --directory /path/to/files --shared-settings

    4. Specify output directory:
       python CRYSTALOptToD12.py --directory /path/to/files --output-dir /path/to/output

    5. Save/load settings:
       python CRYSTALOptToD12.py --save-options --options-file settings.json

    6. Apply saved settings to many files (asks once at a terminal; --yes
       or no terminal: no question):
       python CRYSTALOptToD12.py --directory /path/to/files --config-file settings.json
       python CRYSTALOptToD12.py --out-file opts/*.out --config-file settings.json --yes

AUTHOR:
    New entirely reworked script by Marcus Djokic
    Based on prior versions written by Wangwei Lan, Kevin Lucht, Danny Maldonado, Marcus Djokic
"""

import os
import sys
import re
import argparse
import json
from pathlib import Path

# Import from new modular structure
from d12_constants import (
    # Constants
    ELEMENT_SYMBOLS, SPACEGROUP_SYMBOLS, DEFAULT_SETTINGS,
    DEFAULT_OPT_SETTINGS, DEFAULT_TOLERANCES, DEFAULT_FREQ_SETTINGS,
    freq_default_tolerances, TIGHT_SCF_CALC_TYPES, tight_scf_maxcycle,
    SCF_MAXCYCLE_FALLBACK, TIGHT_SCF_MIN_MAXCYCLE, a_calc,
    check_scf_modifications, apply_scf_modifications, describe_scf_modifications,
    k_mesh_text, check_kpoint_modifications, describe_k_mesh, symmetric_k_points, AUTO_K_POINTS,
    spin_state_change, spin_change_title, SPIN_CHANGE_GUESS,
    is_spin_polarized, set_spin_polarized,
    normalize_config_functionals,
    FUNCTIONAL_CATEGORIES, COMMON_FUNCTIONALS, D3_FUNCTIONALS,
    DFT_GRID_OPTIONS, DISPERSION_OPTIONS, SMEARING_OPTIONS,
    PRINT_OPTIONS, MULTI_ORIGIN_SPACEGROUPS, ATOMIC_NUMBER_TO_SYMBOL,
    ECP_ELEMENTS_EXTERNAL,
    # Utility functions
    yes_no_prompt, get_valid_input, safe_float, safe_int,
    generate_unit_cell_line, read_basis_file, generate_k_points, slab_k_points,
    check_basis_set_compatibility, hf3c_metal_warning, basis_family_name,
    # Configuration functions (from merged d12_config_common)
    configure_tolerances, configure_scf_settings, select_basis_set,
    configure_dft_grid, configure_dispersion, configure_spin_polarization,
    configure_smearing, CUSTOM_FUNCTIONAL, ParentRefused,
)
from d12_parsers import (
    CrystalOutputParser, CrystalInputParser, DECK_GEOMETRY_KEYS, DECK_TEXT_KEYS,
    LOW_DIM_GROUPS,
)
from d12_config import unwrap_d12_config
from d12_calc_freq import (
    get_advanced_frequency_settings, write_frequency_section, INTENSITY_SETTINGS,
    freq_mode,
    phonon_dispersion_refusal, PhononDispersionRefused, gamma_only_phonon_dos,
)
from d12_calc_basic import write_optimization_section, configure_single_point
from d12_writer import (
    write_basis_block,
    write_optimization_block, write_frequency_block, write_properties_block,
    write_print_options, write_k_points, write_spin_settings,
    write_smearing_settings, write_minimal_raman_section,
    write_dft_section, write_basis_set_section
)
# Import write_scf_section from d12_writer
from d12_writer import write_scf_section, DEFAULT_SPINLOCK_CYCLES, atomic_deck
from charge_records import (ChargeCarryError, carry_charge_records, charge_from_parent_out, has_origin_shift,
                            origin_shift_record)
from basis_file_check import BasisFileError
import basis_library
from soc_ecp import (SocDeckBuffer, SocError, checked_smear, metal_parent_warning,
                     parent_metal_evidence, is_spin_orbit_deck)
import soc_ecp
from d12_interactive import (
    display_current_settings, interactive_d12_configuration,
    get_calculation_options_from_current, get_calculation_options,
    save_options_to_file, load_options_from_file, ensure_known_functional
)

# Shared UI layer. This file runs both via `mace opt2d12` (mace on sys.path)
# and standalone (`python CRYSTALOptToD12.py`, where mace may not import). The
# import is fully guarded with a plain-text shim so styling never breaks the
# standalone path or alters exit codes / stream routing.
try:
    from mace.utils import ui
except Exception:
    import sys as _sys, re as _re
    class _UIShim:
        # Mirror ui.py _MARKUP_TOKEN: also strip the bare [/] close tag (the old
        # r"\[/?[a-z#]..." required a char after /, so [/] leaked through).
        _TAG = _re.compile(r"\[/[^\[\]]*\]|\[[a-z#][^\[\]]*\]")
        def _p(self, m): return self._TAG.sub("", str(m))
        def ok(self, m): print(self._p(m))
        def info(self, m): print(self._p(m))
        def print(self, m): print(self._p(m))
        def warn(self, m): print(self._p(m), file=_sys.stderr)
        def err(self, m): print(self._p(m), file=_sys.stderr)
        def rule(self, t=""): print(self._p(t))
        def table(self, cols, rows, title=None):
            if title: print(self._p(title))
            for r in rows: print("  ".join(str(c) for c in r))
        def progress(self, it, **k): return it
        def badge(self, s): return str(s).upper()
    ui = _UIShim()


# What the last process_files() call wrote, or why it wrote nothing, and what
# the user should know about the deck it wrote. A batch reads it after each
# file for its status line and the closing summary.
LAST_RESULT = {"deck": None, "reason": None, "notes": []}


def _fail(reason, message=None):
    """Print an error and record it as the reason this file wrote nothing."""
    ui.err(message if message is not None else reason)
    LAST_RESULT["reason"] = reason


# The records an answer piped onto the wrong question changes without any
# question rejecting it: compared between the parent deck and the new one.
_DRIFT_RECORDS = ("BASISSET", "TOLINTEG", "TOLDEE", "LEVSHIFT", "HISTDIIS", "MAXTRADIUS")
_GRID_KEYWORDS = ("OLDGRID", "LGRID", "XLGRID", "XXLGRID", "XXXLGRID", "HUGEGRID")


# OPTGEOM's own convergence record of the same name as the SCF one: read and
# named apart ("OPTGEOM TOLDEE"), so an OPTGEOM TOLDEE is never compared with
# the SCF TOLDEE.
_OPTGEOM_DRIFT_RECORDS = ("TOLDEE",)


def _drift_records(text):
    """The settings of a deck that piped answers can move, as display strings.

    Records inside the OPTGEOM block (up to its ENDOPT or END) are its own:
    TOLDEE there is "OPTGEOM TOLDEE", and the SCF TOLDEE is the one outside it.
    """
    lines = [" ".join(line.split()) for line in text.splitlines()]
    upper = [line.upper() for line in lines]
    records = {}
    in_optgeom = False
    for i, key in enumerate(upper):
        if key == "OPTGEOM":
            in_optgeom = True
            records.setdefault("OPTGEOM", "")
            continue
        if in_optgeom and key in ("ENDOPT", "END"):
            in_optgeom = False
            continue
        name = f"OPTGEOM {key}" if in_optgeom and key in _OPTGEOM_DRIFT_RECORDS else key
        if key in _DRIFT_RECORDS and name not in records:
            records[name] = lines[i + 1] if i + 1 < len(lines) else ""
        elif key == "DFT" and "functional" not in records and i + 1 < len(lines):
            records["functional"] = lines[i + 1]
    if "DFT" in upper:
        records["DFT grid"] = next((k for k in upper if k in _GRID_KEYWORDS), "DEFAULT")
    return records


def _same_record(a, b):
    try:
        return [float(x) for x in a.split()] == [float(x) for x in b.split()]
    except (AttributeError, ValueError):
        return a == b


def settings_changed_from_parent(parent_text, child_text):
    """``(name, parent value, new value)`` for each drift-prone record that differs."""
    parent = _drift_records(parent_text)
    child = _drift_records(child_text)
    changed = []
    optgeom = tuple(f"OPTGEOM {key}" for key in _OPTGEOM_DRIFT_RECORDS)
    for name in ("functional", "BASISSET", "DFT grid") + _DRIFT_RECORDS[1:] + optgeom:
        before, after = parent.get(name), child.get(name)
        if name == "DFT grid" and (before is None or after is None):
            continue                     # HF on one side: no grid to compare
        if name in optgeom and ("OPTGEOM" not in parent or "OPTGEOM" not in child):
            continue                     # an OPTGEOM on one side only: a new or dropped optimization
        if not _same_record(before, after):
            changed.append((name, before or "(none)", after or "(none)"))
    return changed


def stdin_is_terminal() -> bool:
    """True when someone at a terminal can answer a question.

    With stdin closed (``<&-``) sys.stdin is None, and a closed file object
    raises on isatty(); neither is someone to ask.
    """
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except (ValueError, OSError):
        return False


# --non-interactive reads answers that are on stdin already (the workflow
# engine pipes them in) and waits this long for each one before it takes the
# answers as finished.
PIPED_ANSWER_WAIT_S = 5.0


class _PipedAnswers:
    """A stand-in for sys.stdin that never waits for an answer for long.

    readline() returns end of input ("") once nothing has arrived for
    ``wait`` seconds, so a pipe left open with nothing on it ends the
    answers as a closed one does. It reads the file descriptor itself
    (unbuffered) so that answers already read never sit in a buffer that
    select() cannot see.
    """

    def __init__(self, real, wait):
        self._real = real
        self._fd = real.fileno()
        self._wait = wait
        self._buf = b""
        self._ended = False
        self.encoding = getattr(real, "encoding", None) or "utf-8"

    def _fill(self):
        import select
        while b"\n" not in self._buf and not self._ended:
            ready, _, _ = select.select([self._fd], [], [], self._wait)
            chunk = os.read(self._fd, 65536) if ready else b""
            if not chunk:
                self._ended = True
            self._buf += chunk

    def answers_waiting(self):
        """True when an answer is there to read (the end of input is none)."""
        self._fill()
        return bool(self._buf)

    def readline(self, size=-1):
        self._fill()
        line, sep, rest = self._buf.partition(b"\n")
        self._buf = rest
        return (line + sep).decode(self.encoding, "replace")

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self._real, name)


def piped_answers_or_none(wait=PIPED_ANSWER_WAIT_S):
    """A reader for the answers on stdin, or None when there are none to read.

    For --non-interactive, which must never ask anyone: a terminal is not
    read at all, and a pipe with nothing on it within ``wait`` seconds holds
    no answers. Answers on a pipe or in a file (the workflow engine's) are
    read as before; end of input is left for the caller's EOFError fallback.
    """
    global _PIPED_READER
    if sys.stdin is None or stdin_is_terminal():
        return None
    # One reader for the whole run: with --directory each file takes the
    # next answers, and a fresh reader would lose those the last one read.
    reader = _PIPED_READER
    if reader is None or reader._real is not sys.stdin:
        try:
            reader = _PipedAnswers(sys.stdin, wait)
        except (AttributeError, ValueError, OSError):
            # no file descriptor (answers in memory, as tests give them):
            # nothing there can block, so it is read as it is
            return sys.stdin
        _PIPED_READER = reader
    try:
        waiting = reader.answers_waiting()
    except (OSError, ValueError):
        # select() cannot watch this stdin (not a pipe or file on this
        # platform): read it as before.
        _PIPED_READER = None
        return sys.stdin
    return reader if waiting else None


_PIPED_READER = None


# An optimization CRYSTAL never finished (killed, out of time, out of memory)
# has no "OPT END" line; its .out still parses, but only the starting geometry
# is there to convert.
UNFINISHED_OPT_NOTE = ("unfinished optimization (no OPT END in the .out), so the deck "
                       "has its starting geometry")
# One stopped at the cycle limit ends "OPT END - FAILED" and prints no final
# geometry (MSUCOF-4-FeCp_H2 on HPCC, 800 points), so the parser falls back to
# the first geometry too: the input cell, not the last point's.
FAILED_OPT_NOTE = ("optimization did not converge (OPT END - FAILED), so the deck has "
                   "its starting geometry, not the last point's")

try:
    from mace.utils.calc_detection import is_optimization_output
except Exception:  # standalone run without mace importable
    def is_optimization_output(content: str) -> bool:
        return bool(re.search(r"^[ \t]*(?:\*[ \t]+OPTIMIZATION STARTS"
                              r"|[A-Z]+(?: [A-Z]+)* OPTIMIZATION - POINT[ \t]+\d)",
                              content, re.MULTILINE))


# A SOC deck written without SMEAR from a parent whose .out looks metallic,
# where nobody could be asked for a SMEAR width (soc_ecp, "SMEAR for metals").
SOC_METAL_NOTE = ('SOC deck without SMEAR from a parent that looks metallic; it may '
                  'abort as fcc Au did at SHRINK 10 10 ("smear": <hartree> in the '
                  'template adds SMEAR)')


def soc_parent_metal(out_file):
    """Why the .out a SOC deck is made from looks metallic, or None
    (soc_ecp.parent_metal_evidence). None too for a deck made from a deck
    (opt2d12 --d12-file, no .out): there is no run to read."""
    if not out_file:
        return None
    try:
        with open(out_file, errors="replace") as f:
            return parent_metal_evidence(f.read())
    except OSError:
        return None


def ask_for_soc_smear(deck, evidence, ask=None):
    """Before a SOC deck without SMEAR from a metallic-looking parent is
    written: warn, then ask for a SMEAR width (Enter keeps the deck without
    SMEAR) when someone can answer, else note it for the run's summary. The
    deck is never given SMEAR without an answer or a "smear" setting."""
    ui.warn(metal_parent_warning(evidence))
    if ask is None:
        ask = stdin_is_terminal()
    if ask:
        while True:
            try:
                answer = input("SMEAR width in hartree for this SOC deck, e.g. 0.005 "
                               "(Enter = no SMEAR): ").strip()
            except EOFError:
                break
            if not answer:
                ui.info("  No SMEAR: the deck is written without it.")
                return
            try:
                deck.set_smear(checked_smear(answer))
                return
            except SocError as exc:
                ui.warn(f"  {exc}")
    LAST_RESULT["notes"].append(SOC_METAL_NOTE)


def print_soc_metal_summary(names):
    """The closing lines for SOC decks noted with SOC_METAL_NOTE, if any."""
    if names:
        ui.warn("SOC decks written without SMEAR from a parent that looks metallic - "
                "they may abort as fcc Au did at SHRINK 10 10; set \"smear\": <width "
                "in hartree> in the template, or use a denser mesh:")
        for name in names:
            ui.warn(f"  {name}")


# A "soc_guess" restart with nothing staged: CRYSTAL stops on GUESSPSO with
# nothing in fort.20 (measured on HPCC), and GUESSPNOSO would have no parent
# matrix to start from, so such a deck is never written.
try:
    from mace.utils.calc_detection import missing_2c_matrix_note
except Exception:  # standalone run without mace importable
    def missing_2c_matrix_note(record: str) -> str:
        if record.upper() == "GUESSPSO":
            return "CRYSTAL stops on GUESSPSO with nothing in fort.20 (measured)"
        return (f"{record.upper()} would have no 1c parent matrix in fort.20 to start from "
                f"(what CRYSTAL then does was not measured)")


def soc_restart_plan(options, output_file, input_file, new_filename):
    """For a deck asking for a 2c restart ("soc_guess", manual p. 173):
    (problem, parent .f9, child .f20, parent deck text). problem is None when
    the restart may be written; the frame and shell comparison with the deck
    as written comes later (SocDeckBuffer). Nothing is written here."""
    try:
        guess = soc_ecp._checked_guess(options.get("soc_guess"))
    except SocError as exc:
        return str(exc), None, None, None
    record = guess.upper()
    if not options.get("soc"):
        return (f"soc_guess ({record}) is a restart of a two-component SOC deck, and this "
                f"deck is scalar; set \"soc\": true"), None, None, None
    if not output_file:
        return (f"{record} restarts from the density matrix (.f9) of the parent's run, which "
                f"a deck does not have: set it up from the run's .out"), None, None, None
    if not (input_file and os.path.exists(input_file)):
        return (f"{record} needs the parent deck (.d12) to check that its density matrix "
                f"fits this deck"), None, None, None
    with open(input_file, errors="replace") as fh:
        parent = fh.read()
    if guess == "guesspso":
        if not options.get("two_component"):
            return (f"GUESSPSO restarts from the density matrix of a previous two-component "
                    f"SCF (manual p. 173), and the parent is a one-component run; "
                    f"\"guesspnoso\" starts from an unrestricted 1c run without symmetry"), \
                None, None, None
        ui.info("  GUESSPSO: the parent is a two-component run "
                + ("with SOC" if is_spin_orbit_deck(parent) else
                   "without SOC (a 2c reference run; manual p. 173 asks only for a "
                   "previous 2c-SCF)"))
    else:
        problem = guesspnoso_parent_problem(options, parent, output_file)
        if problem:
            return problem, None, None, None
    f9 = os.path.splitext(output_file)[0] + ".f9"
    if not os.path.isfile(f9) or os.path.getsize(f9) == 0:
        return (f"{record} restarts from the parent's density matrix, and "
                f"{os.path.basename(f9)} next to {os.path.basename(output_file)} is "
                f"{'empty' if os.path.isfile(f9) else 'missing'}; "
                f"{missing_2c_matrix_note(record)}"), \
            None, None, None
    f20 = os.path.splitext(new_filename)[0] + ".f20"
    if (os.path.realpath(os.path.dirname(os.path.abspath(f20)))
            == os.path.realpath(os.path.dirname(os.path.abspath(f9)))
            and os.path.splitext(os.path.basename(f20))[0]
            == os.path.splitext(os.path.basename(f9))[0]):
        return (f"the deck would have the parent's own name in the parent's directory, so "
                f"its job would overwrite the parent's {os.path.basename(f9)}; write it "
                f"elsewhere (--output-dir)"), None, None, None
    if os.path.exists(f20):
        import filecmp
        if not filecmp.cmp(f20, f9, shallow=False):
            return (f"{os.path.basename(f20)} already exists and is not "
                    f"{os.path.basename(f9)}; not overwriting it"), None, None, None
    return None, f9, f20, parent


# What CRYSTAL does with a GUESSPNOSO deck whose 1c parent does not fit:
# it aborts at once in READFG (measured on CRYSTAL23, Bi2 bilayer), never
# runs on from a wrong matrix.
READFG_SYMMETRY = "ERROR **** READFG **** RESTART NOT POSSIBLE - SYMMETRY INCOMPATIBILITY"
READFG_SHELLS = ("ERROR **** READFG **** RESTART NOT POSSIBLE - OPEN-CLOSED SHELL "
                 "INCOMPATIBILITY")

_SYMMETRY_OPERATORS = re.compile(r"^[ \t]*NUMBER OF SYMMETRY OPERATORS[ \t]*:[ \t]*(\d+)",
                                 re.MULTILINE)
GUESSPNOSO_HOWTO = (
    "Rerun the parent as an unrestricted 1c-SCF (UHF, manual p. 131, or SPIN in the "
    "DFT block, p. 140) without symmetry (space group 1, or SYMMREMO in its geometry "
    "block, p. 78), then make this deck from it; from a SOC deck that is the deck "
    "without its TWOCOMPON block, with SYMMREMO and SPIN added and its INPSOC basis "
    "kept (Crystal_d12/README.md). For the SOC energy from a scalar "
    "matrix in one cycle the manual offers 2NDVARIAT (p. 174), which MACE does not write.")


_ELECTRONS_PER_CELL = re.compile(r"^\s*N\. OF ELECTRONS PER CELL\s+(\d+)", re.MULTILINE)


def parent_electron_count(output_file):
    """The electrons per cell a parent .out reports (CRYSTAL's "N. OF
    ELECTRONS PER CELL"), or None when it does not say."""
    try:
        with open(output_file, errors="replace") as fh:
            m = _ELECTRONS_PER_CELL.search(fh.read())
    except (OSError, TypeError):
        return None
    return int(m.group(1)) if m else None


def guesspnoso_parent_problem(options, parent, output_file):
    """Why a parent cannot start a GUESSPNOSO deck, or None. GUESSP[NOSO]
    needs a previous 1c-SCF that was unrestricted (UHF, p. 131, or SPIN,
    p. 140) and ran without space-group or point-group symmetry (SYMMREMO,
    p. 78) - manual p. 173. Read from the parent deck, with the parent .out
    as the fallback for each."""
    records = {ln.strip().upper() for ln in parent.splitlines()[1:]}
    try:
        with open(output_file, errors="replace") as fh:
            out = fh.read()
    except OSError:
        out = ""
    if options.get("two_component") or "TWOCOMPON" in records:
        return ("GUESSPNOSO starts from a previous one-component SCF (manual p. 173), and "
                "the parent is a two-component run; \"guesspso\" restarts from it")
    problems, crystal_says = [], []
    unrestricted = ("UHF" in records or "SPIN" in records
                    or bool(re.search(r"SPIN POLARIZED|TYPE OF CALCULATION.*UNRESTRICTED", out)))
    if not unrestricted:
        problems.append("the parent is restricted (no UHF, p. 131, or SPIN, p. 140)")
        crystal_says.append(READFG_SHELLS)
    frame = soc_ecp.geometry_frame(parent)
    group_one = frame is not None and frame[1] and \
        frame[1][1 if frame[0] == "CRYSTAL" else 0] == (repr(1.0),)
    operators = _SYMMETRY_OPERATORS.findall(out)
    if "SYMMREMO" in records:
        if not group_one:
            ui.warn("  Warning: GUESSPNOSO from a SYMMREMO parent with a space group other "
                    "than 1: whether the child's expanded atom list is in the order of the "
                    "parent's matrix was measured only for a Bi2 bilayer (layer group 72, "
                    "where it was); CRYSTAL does not check it")
    elif not group_one and not (operators and operators[-1] == "1"):
        problems.append("the parent ran with symmetry (its group is not 1 and it has no "
                        "SYMMREMO, p. 78)")
        crystal_says.append(READFG_SYMMETRY)
    if problems:
        return ("GUESSPNOSO needs an unrestricted 1c-SCF run without symmetry (manual "
                "p. 173), and " + "; ".join(problems) + ". CRYSTAL would abort at once with "
                + " or ".join(f'"{words}"' for words in crystal_says)
                + " (measured on CRYSTAL23). " + GUESSPNOSO_HOWTO)
    ui.info("  GUESSPNOSO: the parent is an unrestricted one-component run without "
            "symmetry")
    return None


def optimization_problem(out_file):
    """What to tell the user about the optimization an .out holds: the note
    for one without CRYSTAL's OPT END (unfinished), for one that ended
    "OPT END - FAILED" (not converged), or None."""
    try:
        with open(out_file, errors="replace") as f:
            content = f.read()
    except OSError:
        return None
    if "OPT END - FAILED" in content:
        return FAILED_OPT_NOTE
    if "OPT END" not in content and is_optimization_output(content):
        return UNFINISHED_OPT_NOTE
    return None


# The marker a template saved from a parent with an EXTERNAL basis carries in
# place of a basis name: "the basis written in the parent deck".
PARENT_BASIS_MARKER = "EXTERNAL (from original D12)"

# The keys a config's basis choice is made of.
TEMPLATE_BASIS_KEYS = (
    "basis_set", "basis_set_type", "basis_set_path",
    "use_original_external_basis", "has_original_external_basis",
    "external_basis_data", "external_basis_info",
)

# Keys --save-options wrote that hold one structure's data, not a setting:
# its atoms, its optimization log, its basis records. They are never applied
# to another structure, so a template no longer stores them (the Ag1Br1
# template was 90 KB, almost all of it these).
TEMPLATE_STRUCTURE_DATA_KEYS = (
    "coordinates", "primitive_cell", "conventional_cell",
    "primitive_coordinates", "crystallographic_coordinates",
    "optimization_content", "symmetry_operations",
    "external_basis_data", "external_basis_info",
    # Whether this structure's parent was a two-component run (its "soc",
    # the setting, is kept).
    "two_component",
)


def template_uses_parent_basis(config_data) -> bool:
    """True when a config's basis means "each parent's own basis".

    --save-options run on a parent with an EXTERNAL basis stores the marker
    PARENT_BASIS_MARKER (and, before this release, that parent's basis
    records). That is not a basis another structure can use: each structure
    keeps the basis its own parent deck was written with, external or internal.
    A named internal basis or an external basis directory is a real choice and
    applies to every structure.
    """
    basis = str(config_data.get("basis_set") or "")
    if basis.startswith("EXTERNAL (from original"):
        return True
    return (config_data.get("basis_set_type") == "EXTERNAL"
            and bool(config_data.get("use_original_external_basis"))
            and not config_data.get("basis_set_path"))


def options_for_template(options) -> dict:
    """The settings --save-options writes: all but one structure's data.

    The k-point mesh is left out too. No opt2d12 question sets it, so the one
    in the options is always the parent's own, and in a template it made every
    other structure's mesh be regenerated from its cell instead of kept from
    its parent. A config's "k_points" is ignored when applied for the same
    reason; one mesh for every file is set by writing
    "k_points_for_all_files" into the config by hand.
    """
    from basis_library import portable_options
    # a bundled basis library is stored by name, not by this checkout's path
    return portable_options({k: v for k, v in options.items()
                             if k not in TEMPLATE_STRUCTURE_DATA_KEYS
                             and k not in ("k_points", "k_points_set", "spin_change")})


# The config key that sets one k-point mesh for every file. A config's plain
# "k_points" is ignored: --save-options before this release stored the saving
# structure's own mesh there, and each derived deck keeps its own parent's.
ALL_FILES_K_POINTS_KEY = "k_points_for_all_files"


def all_files_k_points(value):
    """The mesh of a config's k_points_for_all_files as the deck writes it
    ("IS ISP" or "ka kb kc", as in a parent deck), "auto" (each file's mesh
    from its own cell), or None if it is neither 1 to 3 whole numbers nor
    "auto". One number n means n n n (d12_constants.k_mesh_text)."""
    return k_mesh_text(value)


def set_k_mesh(options, mesh, source):
    """This deck's k-point mesh, set by a config (k_points_for_all_files, a
    workflow step's kpoint_modifications): the mesh the writer then keeps as
    a parent's own (the parent_k_points it compares against), or None for
    "auto", which the writer generates from the structure's cell."""
    if mesh == AUTO_K_POINTS:
        options["k_points"] = None
        options.pop("k_points_set", None)
        ui.info(f"  k-points: MACE's automatic mesh from this structure's cell ({source})")
        return None
    options["k_points"] = mesh
    # set here, not the parent's: the writer makes symmetry-equivalent directions agree
    options["k_points_set"] = True
    ui.info(f"  k-points: {mesh} ({source})")
    return mesh


def merge_optimization_settings(parent, override, replace_type=False):
    """The parent's OPTGEOM settings with ``override``'s values on top.

    Keys match case-insensitively (the d12 parser writes TOLDEG, the
    interactive flow toldeg), so an override always replaces the parent's
    value however either spells it. ``replace_type`` drops the parent's
    optimization type when the caller sets ``optimization_type`` separately:
    the writer prefers a ``type`` inside the settings over that argument.
    """
    merged = dict(parent) if isinstance(parent, dict) else {}
    if replace_type:
        merged.pop("type", None)
    for key, value in (override or {}).items():
        for existing in [k for k in merged if k.lower() == str(key).lower()]:
            del merged[existing]
        merged[key] = value
    return merged


def child_freq_settings(parent_freq: dict) -> dict:
    """The parent deck's FREQCALC settings a derived deck carries.

    Everything the parent asked for, except RESTART: that restarts the
    parent's own frequency run from the FREQINFO.DAT it wrote (manual sec.
    8.2, p. 219), which a new deck does not have.
    """
    return {k: v for k, v in (parent_freq or {}).items() if k != "restart"}


def prefer_deck_unrecognised_functional(settings: dict, in_data: dict) -> None:
    """Ask about a deck functional line nothing could map, over the output's guess.

    When the .d12 names a functional that is not a CRYSTAL23 keyword, the
    functional the output parser read from CRYSTAL's printed exchange and
    correlation names (a substring guess: the same line is printed for BLYP
    and B3LYP, PBE and PBE0-13, ...) used to be taken silently. Drop it, so
    the unrecognized-functional prompt (or its warning) handles the parent.
    """
    if (in_data.get("unrecognised_functional") and not in_data.get("functional")
            and settings.get("functional")):
        ui.warn(f"  The D12's functional '{in_data['unrecognised_functional']}' is not a "
                f"CRYSTAL23 keyword; not using the output's reading of it "
                f"({settings['functional']}).")
        settings["functional"] = None


def parent_dispersion_input(settings: dict):
    """The parent's DFTD3/GRIMME records to write, or None.

    They go with the functional they were parsed with: always for the
    parent's own EXCHANGE/CORRELAT definition, and for a functional keyword
    while dispersion is still on (the D3 question's answer) and the
    functional is still the parent's. Any other functional takes the D3
    question's usual "<name>-D3" form.
    """
    records = settings.get("custom_dftd3")
    if not records:
        return None
    functional = settings.get("functional") or ""
    parent = settings.get("custom_dftd3_functional", CUSTOM_FUNCTIONAL)
    if functional == CUSTOM_FUNCTIONAL:
        # Only the parent's own definition: not a replacement typed at the
        # unknown-functional prompt (LSRSH-PBE with its record).
        return records if parent == CUSTOM_FUNCTIONAL else None

    def base(name):
        name = str(name or "")
        return name[:-3] if name.upper().endswith("-D3") else name

    if settings.get("dispersion") and parent and base(functional).upper() == base(parent).upper():
        return records
    return None


def _with_opt_type_default(settings: dict, opt_type) -> dict:
    """Settings whose optimization type is --opt-type, so the prompts show it."""
    if not opt_type:
        return settings
    settings = dict(settings)
    settings["optimization_type"] = opt_type
    if settings.get("optimization_settings"):
        settings["optimization_settings"] = {**settings["optimization_settings"],
                                             "type": opt_type}
    return settings


def apply_opt_type_flag(options: dict, opt_type) -> None:
    """Make an explicit --opt-type the optimization type of an OPT deck.

    The writer takes the type inside optimization_settings over
    optimization_type, so both are set. Used after the settings prompts,
    which used to leave the parent's type (or the answer) in place of the
    flag whenever answers came from stdin.
    """
    if not opt_type or options.get("calculation_type") != "OPT":
        return
    opt_settings = options.get("optimization_settings")
    opt_settings = dict(opt_settings) if isinstance(opt_settings, dict) else {}
    opt_settings["type"] = opt_type
    options["optimization_settings"] = opt_settings
    options["optimization_type"] = opt_type


def dedupe_dispersion_suffix(functional: str) -> str:
    """Collapse any accidental repeated '-D3' in a functional name to one.

    Purely cosmetic: the SCF section writes the functional/dispersion correctly
    regardless, but the continuation filenames embed the functional string, so a
    doubled value produced names like ``..._B3LYP-D3-D3_optimized.d12``. This
    guarantees a generated filename never carries ``-D3-D3`` no matter how the
    functional was derived (parsed output, reused config, re-continuation).
    """
    if not functional:
        return functional
    return re.sub(r'(?:-D3){2,}', '-D3', functional)


def _get_phonon_band_path_title(band_settings, geometry_data, dimensionality="CRYSTAL"):
    """Generate path information string for phonon band calculations."""
    from d3_kpoints import get_band_path_from_symmetry, unicode_to_ascii_kpoint

    seekpath_source = None
    if str(dimensionality).upper() == "POLYMER":
        # the Gamma-X path along the chain write_frequency_section writes for a polymer
        return " - Phonon Band Structure - default - GAMMA-X"
    if str(dimensionality).upper() == "SLAB":
        # The in-plane path write_frequency_section writes for a slab
        from d12_calc_freq import slab_phonon_band_path
        slab_path = slab_phonon_band_path(dict(band_settings),
                                          geometry_data.get("optimization_content") or "")
        if slab_path is None:
            return None
        labels = slab_path[2]
        source = " - Literature" if band_settings.get("literature_path") else (
            " - SeeKPath" if band_settings.get("seekpath_full") else " - default")
        if not labels:
            return " - Phonon Band Structure" + source + " - in-plane path"
        return (" - Phonon Band Structure" + source + " - "
                + "-".join(l if l == "|" else unicode_to_ascii_kpoint(l) for l in labels))
    # For automatic paths, we need to determine the path based on space group
    if band_settings.get("auto_path", False) or band_settings.get("path") == "AUTO" or band_settings.get("path") == "auto":
        # Try to get path from space group
        space_group = geometry_data.get("spacegroup", 1)
        lattice_type = "P"  # Default
        
        # Try to extract lattice type from optimization content
        opt_content = geometry_data.get("optimization_content", "")
        if opt_content:
            import re
            sg_symbol_match = re.search(r'SPACE GROUP.*?:\s+([A-Z]\s*[\-/0-9\s]*[A-Z0-9]*)', opt_content)
            if sg_symbol_match:
                symbol = sg_symbol_match.group(1).strip()
                if symbol:
                    lattice_type = symbol[0]
        
        # Generate the path labels based on the format
        format_type = band_settings.get("format", "labels")
        path_method = band_settings.get("path_method", "labels")
        
        # For all formats, we get the appropriate labels
        if format_type == "seekpath" and band_settings.get("seekpath_full", False):
            # Try to get SeeK-path labels. write_frequency_section picks the
            # SeeK-path variant from the parent's output (lattice parameters,
            # inversion), so the labels have to be read from the same output
            # to name the path it writes.
            try:
                from d3_kpoints import get_seekpath_labels, get_seekpath_full_kpath
                from d12_calc_freq import _output_as_file
                with _output_as_file(opt_content or None) as out_file:
                    path_labels = get_seekpath_labels(space_group, lattice_type, out_file)
                    result = get_seekpath_full_kpath(space_group, lattice_type, out_file, cell="primitive")
                if result:
                    kpath_info = result[1]
                    if kpath_info.get("source") in ("literature", "default"):
                        seekpath_source = kpath_info["source"]
                        if seekpath_source == "default":
                            # No SeeK-path or literature path: the segments
                            # are the standard path's.
                            path_labels = get_band_path_from_symmetry(
                                space_group, lattice_type)
                    elif kpath_info.get("has_inversion"):
                        seekpath_source = "seekpath_inv"
                    else:
                        seekpath_source = "seekpath_noinv"
            except:
                # Fallback to standard labels
                path_labels = get_band_path_from_symmetry(space_group, lattice_type)
        else:
            # For all other formats (labels, vectors, literature), use standard labels
            path_labels = get_band_path_from_symmetry(space_group, lattice_type)
    elif "path_labels" in band_settings:
        path_labels = band_settings["path_labels"]
    else:
        # Try to extract from path if it's in label format
        path = band_settings.get("path", [])
        if path and isinstance(path[0], str) and " " in path[0]:
            # Extract labels from segments like "G X", "X M"
            path_labels = []
            for segment in path:
                parts = segment.split()
                if path_labels and parts[0] == path_labels[-1]:
                    # Continuous path
                    path_labels.append(parts[1])
                else:
                    # Discontinuous path
                    if path_labels:
                        path_labels.append("|")
                    path_labels.extend(parts)
        else:
            return None
    
    # Convert labels to ASCII and format
    path_str = []
    for label in path_labels:
        if label == "|":
            path_str.append("|")
        else:
            path_str.append(unicode_to_ascii_kpoint(label))
    
    # Determine the k-path source for the title
    kpath_source = seekpath_source or band_settings.get("kpath_source", "default")
    if kpath_source == "seekpath_inv":
        source_info = " - SeeKPath (w.I)"
    elif kpath_source == "seekpath_noinv":
        source_info = " - SeeKPath (no.I)"
    elif kpath_source == "seekpath":
        source_info = " - SeeKPath"
    elif kpath_source == "literature":
        source_info = " - Literature"
    elif kpath_source == "manual":
        source_info = " - Manual"
    elif kpath_source == "template":
        source_info = " - Template"
    elif kpath_source == "fractional":
        source_info = " - Fractional"
    else:
        source_info = " - default"
    
    # Create title string
    return " - Phonon Band Structure" + source_info + " - " + "-".join(path_str)


def use_low_dim_group(settings):
    """Make a SLAB/POLYMER deck's group record the parent's layer/rod group.

    Every consumer of settings["spacegroup"] (the group record, the cell
    record, write-only-unique-atoms) reads it as the deck's group. For a SLAB
    or POLYMER that is the layer group (1-80) or rod group (1-99) - taken from
    the parent deck, or from the .out's "TWO-SIDED PLANE GROUP N." / "POLYMER
    GROUP N." line - never a 3D space group number from the .out's symmetry
    analysis (its "CORRESPONDING SPACE GROUP"). With neither known,
    a 3D number is dropped and the deck falls back to group 1 (P1, all atoms).
    """
    dim = settings.get("dimensionality")
    if dim not in LOW_DIM_GROUPS:
        return
    key, _top = LOW_DIM_GROUPS[dim]
    label = key.replace("_", " ")
    group = settings.get(key)
    current = settings.get("spacegroup")
    if group is not None:
        if current is not None and current != group:
            ui.info(f"  Using the parent's {label} {group}, not space group {current}")
        settings["spacegroup"] = group
    elif current is not None:
        ui.warn(f"Warning: no {label} found for this {dim}; space group {current} "
                f"is a 3D group, not a {label}. Writing the structure in group 1 (P1).")
        settings["spacegroup"] = None


def write_supercell_in_p1(out_data, settings):
    """Describe a SUPERCEL/SUPERCON parent's cell in group 1 (P1). True if it did.

    After SUPERCEL or SUPERCON (manual sec. 4.21, pp.74-77) the geometry
    CRYSTAL prints - and computes - is the supercell, with only the symmetry
    operators the supercell keeps; ATOMSUBS/ATOMREMO/GHOSTS etc. after it
    reduce them further. The space group CRYSTAL printed (and the parent
    deck's group record) is still the primitive host's, so the child deck was
    that group with the supercell's lattice and its asymmetric atoms: SG 227
    with a = 5.01 A and 5 atoms for a 2x2x2 diamond, not diamond. The
    supercell's own operators cannot be written as a group record, so the
    child lists every atom of the supercell in P1: the same cell and atoms,
    in the same order (so atom labels such as GHOSTS still name the same
    atoms), at the cost of CRYSTAL's symmetry reduction.
    """
    content = out_data.get("optimization_content") or ""
    if "SUPERCELL GENERATED" not in content:
        return False
    group = settings.get("spacegroup")
    settings["spacegroup"] = out_data["spacegroup"] = 1
    settings["origin_setting"] = out_data["origin_setting"] = "0 0 0"
    for key in ("layer_group", "rod_group"):
        if settings.get(key) is not None:
            settings[key] = 1
    # The supercell is the primitive cell CRYSTAL computed; its lattice and
    # its full atom list are the ones the writer must use.
    if out_data.get("primitive_cell"):
        out_data["conventional_cell"] = out_data["primitive_cell"]
    out_data["coordinates"] = list(out_data.get("primitive_coordinates")
                                   or out_data.get("coordinates") or [])
    out_data["crystallographic_coordinates"] = list(out_data["coordinates"])
    ui.info(f"  The parent computed a supercell (SUPERCEL/SUPERCON): writing its "
            f"{len(out_data['coordinates'])} atoms in P1, not in the host's group {group}")
    return True


def low_dim_group_error(settings):
    """Why the SLAB/POLYMER group record would be invalid, or None if it is fine."""
    dim = settings.get("dimensionality")
    if dim not in LOW_DIM_GROUPS:
        return None
    key, top = LOW_DIM_GROUPS[dim]
    group = settings.get("spacegroup", 1)
    if isinstance(group, int) and not isinstance(group, bool) and 1 <= group <= top:
        return None
    return (f"{dim} group record {group!r} is not a {key.replace('_', ' ')} "
            f"(1-{top}); CRYSTAL would refuse or misread the deck.")


def conventional_atom_number(atom_number, settings) -> int:
    """The atomic number to write for an atom of the parent's geometry.

    A parent with an EXTERNAL basis numbers its atoms by that basis (Ag with
    an ECP is 247: CRYSTAL reads Z = NAT mod 100, NAT > 200 meaning an
    effective core potential). Those numbers name records of the parent's own
    basis. A deck that uses an internal basis (BASISSET) instead needs the
    plain atomic number, 47, or CRYSTAL looks for a basis for atom 247. An
    external-basis deck keeps the parent's numbers as they are.
    """
    number = int(atom_number)
    # The 3c methods always write BASISSET, whatever the basis type says.
    internal = (settings.get("basis_set_type") != "EXTERNAL"
                or settings.get("functional") in THREE_C_FUNCTIONALS)
    if internal and number > 100:
        return number % 100
    return number


def deck_atom_number(atom_number, settings) -> int:
    """The atomic number the deck writes for an atom of the parent's geometry
    (conventional_atom_number, plus 200 for an element that needs an ECP in
    an EXTERNAL basis; an internal basis handles its ECPs itself). A phonopy
    displaced supercell's .ext numbers its atoms the same way."""
    number = conventional_atom_number(atom_number, settings)
    if settings.get("basis_set_type") == "EXTERNAL" and number in ECP_ELEMENTS_EXTERNAL:
        number += 200
    return number


THREE_C_FUNCTIONALS = ("HF3C", "HFSOL3C", "PBEH3C", "HSE3C", "B973C", "PBESOL03C", "HSESOL3C")


def _write_library_records(f, basis_dir, atom_numbers):
    """Write each element's record from an EXTERNAL basis library.

    For ECP elements (Z >= 37) the records are named with a +200 offset. An
    element the library has no record for raises BasisFileError naming the
    element and the family: a deck without its basis is one CRYSTAL always
    rejects, so it is never written (atomic_deck discards the part written).

    Ends the basis input with the "99 0" line CRYSTAL reads it up to, unless
    the last record already ends with it (a hand-made library file may).
    """
    from d12_constants import missing_basis_record

    last = ""
    for atom_num in sorted(atom_numbers):
        file_num = atom_num + 200 if atom_num in ECP_ELEMENTS_EXTERNAL else atom_num
        if not basis_library.record_exists(basis_dir, file_num):
            raise missing_basis_record(basis_dir, atom_num, file_num)
        last = basis_library.read_record(basis_dir, file_num)
        if last and not last.endswith("\n"):
            last += "\n"
        f.write(last)
    tail = [line.split() for line in last.splitlines() if line.strip()]
    if not tail or tail[-1] != ["99", "0"]:
        f.write("99 0\n")


def _library_basis_dir(settings, external_basis_data=None):
    """The EXTERNAL basis library a deck's records are read from, or None
    when its records are the parent's own (use_original_external_basis) or it
    has no external basis. Mirrors the writers' choice of path exactly."""
    if settings.get("basis_set_type") != "EXTERNAL":
        return None
    if settings.get("functional") in THREE_C_FUNCTIONALS:
        return None  # the 3c methods always write BASISSET
    if settings.get("use_original_external_basis") and external_basis_data:
        return None
    if settings.get("basis_set_path"):
        return settings["basis_set_path"]
    legacy_path = settings.get("basis_set")
    if legacy_path and basis_library.path_exists(legacy_path):
        return legacy_path
    return None


def write_d12_file(output_file, geometry_data, settings, external_basis_data=None,
                   parent_k_points=None, ask=None, parent_metal=None, soc_restart_parent=None):
    """Write new D12 file with optimized geometry and settings (see
    _write_d12_file). A basis record that is missing, or one CRYSTAL could not
    read, refuses the deck: nothing is written, the reason is recorded in
    LAST_RESULT and False is returned."""
    try:
        return _write_d12_file(output_file, geometry_data, settings, external_basis_data,
                               parent_k_points=parent_k_points, ask=ask,
                               parent_metal=parent_metal,
                               soc_restart_parent=soc_restart_parent)
    except BasisFileError as exc:
        _fail(str(exc), f"Not writing {os.path.basename(output_file)}: {exc}")
        return False


def _write_d12_file(output_file, geometry_data, settings, external_basis_data=None,
                    parent_k_points=None, ask=None, parent_metal=None, soc_restart_parent=None):
    """Write new D12 file with optimized geometry and settings.

    parent_k_points is the raw k-point value parsed from the parent .d12 this
    deck is generated from. When settings still carry exactly that value, the
    parent's mesh is written back as-is (including an anisotropic
    ``0 ISP / ka kb kc`` mesh). It is deliberately a call argument, not a
    settings key, so it never reaches --save-options JSON.

    ask: kept for callers; a basis set that lacks an element is always
    refused now (CRYSTAL rejects such a deck), so nothing is asked.

    Returns True on success. Returns False when creation is aborted (a basis
    set without a record for one of the elements, or a
    phonon dispersion asked of a system it cannot be written for). A
    SLAB/POLYMER whose group record is not a layer/rod group is refused the
    same way, before anything is written. The deck is written beside
    output_file and renamed onto it only when complete, so an abort or an
    exception never leaves a 0-byte or truncated deck.
    """
    group_problem = low_dim_group_error(settings)
    if group_problem:
        _fail(group_problem)
        return False

    calc_type = settings.get("calculation_type", settings.get("calc_type", "OPT"))
    if calc_type == "FREQ":
        # a molecule's phonon DOS is the Gamma-only one (manual p.236), not a dispersion
        if settings.get("freq_settings") is not None:
            settings["freq_settings"] = gamma_only_phonon_dos(
                settings.get("dimensionality", "CRYSTAL"), settings.get("freq_settings"))
        refusal = phonon_dispersion_refusal(
            settings.get("dimensionality", "CRYSTAL"), settings.get("freq_settings"),
            geometry_data.get("optimization_content") or "")
        if refusal:
            _fail(refusal, f"\nNot writing {os.path.basename(output_file)}: {refusal}")
            return False

    # A parent that was a two-component run (its deck's TWOCOMPON block or its
    # .out's 2c headers, read by the parsers). Its "soc" carries over to this
    # deck unless the template says "soc": false; either way the user is told.
    if settings.get("two_component"):
        if settings.get("soc"):
            ui.info("  SOC: the parent is a two-component SOC run, so this deck is one too "
                    "(\"soc\": false in the template writes a scalar deck)")
        else:
            reason = ("the template sets \"soc\": false" if "soc" in settings else
                      "no SOC could be read from a parent .d12 - none, or no SOC in its "
                      "TWOCOMPON block; \"soc\": true in the template writes a SOC deck")
            ui.warn(f"  Warning: the parent is a two-component run, but this deck is "
                    f"scalar ({reason})")
            if settings.get("use_original_external_basis") and any(
                    str(line).strip().upper() in ("INPSOC", "STUTSC")
                    for line in external_basis_data or []):
                ui.warn("  Warning: its basis keeps the parent's spin-orbit ECPs (INPSOC "
                        "or STUTSC); what a one-component SCF does with a SOREP is not "
                        "in the CRYSTAL23 manual (untested)")

    with atomic_deck(output_file) as f:
        if settings.get("soc"):
            # Two-component SOC deck: written as usual, then rewritten by
            # soc_ecp.soc_deck before it reaches the file.
            f = SocDeckBuffer(f, options=settings, restart_from=soc_restart_parent)

        # Title
        # Title from the file NAME only: with --output-dir the path carries a
        # directory, which must not leak into the CRYSTAL title line.
        title = os.path.basename(output_file).replace(".d12", "")
        
        # Add phonon band path information if this is a FREQ calculation with bands
        calc_type = settings.get("calculation_type", settings.get("calc_type", "OPT"))
        if calc_type == "FREQ":
            freq_settings = settings.get("freq_settings", {})
            if freq_settings.get("dispersion", False) and "bands" in freq_settings:
                band_settings = freq_settings["bands"]
                # Debug output
                # print(f"DEBUG: band_settings = {band_settings}")
                path_info = _get_phonon_band_path_title(
                    band_settings, geometry_data, settings.get("dimensionality", "CRYSTAL"))
                if path_info:
                    title += path_info
        # A deck that turns spin polarization on or off says so: it starts
        # from a fresh SCF guess, not from the parent's density matrix.
        if settings.get("spin_change"):
            title += spin_change_title(settings["spin_change"])
        
        f.write(f"{title}\n")

        # Structure section. A phonopy displaced supercell (calculation type
        # PHONOPY) has its geometry in <job>.ext, which the job script stages
        # as fort.34 (EXTERNAL, manual p. 21); its atoms only number the basis.
        external = bool(geometry_data.get("external_geometry"))
        dimensionality = settings.get("dimensionality", "CRYSTAL")
        f.write("EXTERNAL\n" if external else f"{dimensionality}\n")

        if dimensionality == "CRYSTAL" and not external:
            # Handle space groups with multiple origins
            spacegroup = settings.get("spacegroup", 1)
            origin_setting = settings.get("origin_setting", "0 0 0")

            # Check if this space group has special origin settings
            if spacegroup in MULTI_ORIGIN_SPACEGROUPS:
                spg_info = MULTI_ORIGIN_SPACEGROUPS[spacegroup]
                # Preserve the original origin setting if it matches known alternatives
                if origin_setting == spg_info.get('alt_crystal_code', ''):
                    f.write(f"{origin_setting}\n")  # Use original alternate origin
                elif origin_setting == spg_info.get('crystal_code', '0 0 0'):
                    f.write(f"{origin_setting}\n")  # Use original default origin
                else:
                    # Fallback to extracted origin setting to preserve original
                    f.write(f"{origin_setting}\n")
            else:
                f.write(f"{origin_setting}\n")

            f.write(f"{spacegroup}\n")
            if has_origin_shift(str(origin_setting)):
                # IFSO > 1: CRYSTAL reads the origin shift (24ths of the
                # cell vectors) on the line after the space group
                f.write(origin_shift_record(settings.get("origin_shift")) + "\n")
        elif dimensionality in ["SLAB", "POLYMER"]:
            f.write(f"{settings.get('spacegroup', 1)}\n")
        elif dimensionality == "MOLECULE":
            f.write("1\n")  # C1 symmetry

        # Unit cell parameters (if not molecule)
        if dimensionality != "MOLECULE" and geometry_data.get("conventional_cell"):
            cell_line = generate_unit_cell_line(
                settings.get("spacegroup", 1),
                geometry_data["conventional_cell"],
                dimensionality,
            )
            if cell_line:
                f.write(f"{cell_line}\n")

        # Atomic coordinates - use crystallographic coordinates when conventional cell is used
        # For centered lattices (non-P space groups), CRYSTAL reports both primitive and
        # crystallographic (conventional) coordinates. When using conventional_cell for
        # lattice parameters, we MUST use crystallographic_coordinates to match.
        use_conventional = geometry_data.get("conventional_cell") is not None
        crystallographic_coords = geometry_data.get("crystallographic_coordinates", [])

        if use_conventional and crystallographic_coords:
            # Use crystallographic coordinates to match conventional cell
            coords = crystallographic_coords
            ui.info(f"  Using crystallographic coordinates ({len(coords)} atoms) with conventional cell")
        else:
            # Use primitive coordinates (default)
            coords = geometry_data["coordinates"]
            if use_conventional and not crystallographic_coords:
                ui.warn(f"  Warning: Conventional cell used but crystallographic coordinates not found")

        # Filter coordinates if requested
        if settings.get("write_only_unique", False):
            coords_to_write = [c for c in coords if c.get("is_unique", True)]
        else:
            coords_to_write = coords

        if not external:
            f.write(f"{len(coords_to_write)}\n")

        for atom in ([] if external else coords_to_write):
            # Store original atomic number for symbol lookup
            original_atom_num = conventional_atom_number(atom["atom_number"], settings)
            atom_num = deck_atom_number(atom["atom_number"], settings)

            # An EXTERNAL-basis parent keeps conventional numbers (282 = Pb
            # with an ECP); the element is the number modulo 100.
            symbol = ATOMIC_NUMBER_TO_SYMBOL.get(
                original_atom_num % 100 if original_atom_num > 100 else original_atom_num, "X")
            
            # Handle different coordinate systems based on dimensionality
            if dimensionality == "SLAB":
                # For SLAB: fractional a,b coordinates and Cartesian z coordinate
                # In CRYSTAL output files for SLAB, z-coordinates from "ATOMS IN THE ASYMMETRIC UNIT"
                # are already in Angstroms (shown as Z(ANGSTROM) in header), not fractional
                z_cart = float(atom['z'])
                f.write(
                    f"{atom_num} {atom['x']} {atom['y']} {z_cart:.6f} Biso 1.000000 {symbol}\n"
                )
            elif dimensionality == "POLYMER":
                # For POLYMER: fractional x, Cartesian y,z coordinates
                # In CRYSTAL output files for POLYMER, y,z coordinates should already be in Angstroms
                y_cart = float(atom['y'])
                z_cart = float(atom['z'])
                f.write(
                    f"{atom_num} {atom['x']} {y_cart:.6f} {z_cart:.6f} Biso 1.000000 {symbol}\n"
                )
            elif dimensionality == "MOLECULE":
                # For MOLECULE: all Cartesian coordinates
                # In CRYSTAL output files for MOLECULE, all coordinates should be in Angstroms
                x_cart = float(atom['x'])
                y_cart = float(atom['y'])
                z_cart = float(atom['z'])
                f.write(
                    f"{atom_num} {x_cart:.6f} {y_cart:.6f} {z_cart:.6f} Biso 1.000000 {symbol}\n"
                )
            else:
                # For CRYSTAL: all fractional coordinates
                f.write(
                    f"{atom_num} {atom['x']} {atom['y']} {atom['z']} Biso 1.000000 {symbol}\n"
                )

        # Calculation-specific section - MUST come before basis set
        if settings["calculation_type"] == "OPT":
            # For OPT: OPTGEOM follows directly after coordinates, no END needed
            write_optimization_section(
                f,
                settings.get("optimization_type", "FULLOPTG"),
                settings.get("optimization_settings", DEFAULT_OPT_SETTINGS),
                # A tolerance the parent's OPTGEOM never set stays unset, so
                # CRYSTAL's default applies to the child as it did to the parent.
                fill_missing_tolerances=False,
            )
        elif settings["calculation_type"] == "FREQ":
            # Check if this is ANHARM or FREQCALC
            if settings.get("freq_mode") == "ANHARM":
                # For ANHARM: Goes outside geometry block
                # First close the geometry section
                f.write("END\n")
                # Then write ANHARM section
                anharm_settings = settings.get("anharm_settings", {})
                # Convert format from UI to d12creation
                converted_anharm = {
                    "atom_label": anharm_settings.get("h_atom", 1),
                    "keepsymm": anharm_settings.get("keep_symmetry", False),
                    "points": 26 if anharm_settings.get("points26", False) else 7
                }
                # Handle isotopes format conversion
                if "isotopes" in anharm_settings:
                    isotope_dict = {}
                    for atom_label, mass in anharm_settings["isotopes"]:
                        isotope_dict[atom_label] = mass
                    converted_anharm["isotopes"] = isotope_dict
                    
                from d12_calc_freq import write_anharm_section
                write_anharm_section(f, converted_anharm)
            else:
                # For FREQCALC: FREQCALC follows directly after coordinates, no END needed
                # Determine crystal system from space group
                spacegroup = settings.get("spacegroup", 1)
                crystal_system = None
                if spacegroup:
                    if 1 <= spacegroup <= 2:
                        crystal_system = "triclinic"
                    elif 3 <= spacegroup <= 15:
                        crystal_system = "monoclinic"
                    elif 16 <= spacegroup <= 74:
                        crystal_system = "orthorhombic"
                    elif 75 <= spacegroup <= 142:
                        crystal_system = "tetragonal"
                    elif 143 <= spacegroup <= 167:
                        crystal_system = "trigonal"
                    elif 168 <= spacegroup <= 194:
                        crystal_system = "hexagonal"
                    elif 195 <= spacegroup <= 230:
                        crystal_system = "cubic"
                        
                # Get optimization section text if available
                optimization_section = geometry_data.get("optimization_content", None)
                    
                write_frequency_section(
                    f, settings.get("freq_settings", DEFAULT_FREQ_SETTINGS), 
                    crystal_system, spacegroup, optimization_section,
                    dimensionality=settings.get("dimensionality", "CRYSTAL")
                )
        elif settings["calculation_type"] in ("ELASTIC", "EOS", "QHA", "CPHF"):
            # ELASTCON / ELAPIEZO / PIEZOCON / PIEZOCP / PHOTOELA / EOS / QHA / CPHF: the
            # last keyword of the geometry input, with its own END
            # (CRYSTAL23 manual ch. 9, 10, 12-13; d12_calc_elastic)
            from d12_calc_elastic import write_elastic_section
            write_elastic_section(f, settings.get("elastic_settings"), settings["calculation_type"])
        else:
            # For SP: Plain SP calculations don't have OPTGEOM/FREQCALC blocks
            # Just continue to basis set section
            pass

        # Handle basis sets and method section
        functional = settings.get("functional", "")
        method = "HF" if functional in ["RHF", "UHF", "HF3C", "HFSOL3C"] else "DFT"

        # Handle HF 3C methods and regular HF methods
        if functional in ["HF3C", "HFSOL3C"]:
            # These are HF methods with corrections, write basis set but no DFT block
            # Write BASISSET keyword for internal basis sets
            f.write("BASISSET\n")
            f.write(f"{settings['basis_set']}\n")
            # A spin-polarized parent gives an unrestricted HF-3c run: UHF
            # (manual p. 131), the open-shell HF that SPIN is in the DFT 3c
            # methods. The HF-3c block keeps its own END (p. 158-159).
            if settings.get("spin_polarized"):
                f.write("UHF\n")

            # Add 3C corrections
            if functional == "HF3C":
                caution = hf3c_metal_warning(
                    conventional_atom_number(atom["atom_number"], settings)
                    for atom in coords_to_write)
                if caution:
                    ui.warn(f"  Warning: {caution}")
                f.write("HF3C\n")
                f.write("END\n")
            elif functional == "HFSOL3C":
                f.write("HFSOL3C\n")
                f.write("END\n")
        elif functional in ["RHF", "UHF"]:
            # Standard HF methods
            if settings.get("basis_set_type") == "EXTERNAL":
                # External basis set handling - need END to close geometry section
                f.write("END\n")
                # Handle external basis sets for HF methods
                if settings.get("use_original_external_basis") and external_basis_data:
                    # Use the external basis from the original file
                    for line in external_basis_data:
                        f.write(f"{line}\n")
                    f.write("99 0\n")
                    f.write("END\n")
                elif settings.get("basis_set_path"):
                    # Read basis sets from specified path
                    # Note: CRYSTAL doesn't support comments, so we just print the path info
                    ui.info(f"  External basis set from: {settings['basis_set_path']}")
                    unique_atoms = set()
                    for atom in coords_to_write:
                        unique_atoms.add(int(atom["atom_number"]))

                    # Read basis set files
                    # For ECP elements (Z >= 37), basis files are named with +200 offset
                    _write_library_records(f, settings["basis_set_path"], unique_atoms)
                    f.write("END\n")
                else:
                    # Fallback: basis_set_type is EXTERNAL but no path provided
                    legacy_path = settings.get("basis_set")
                    if legacy_path and basis_library.path_exists(legacy_path):
                        ui.info(f"  Note: Using legacy basis_set field as path: {legacy_path}")
                        unique_atoms = set()
                        for atom in coords_to_write:
                            unique_atoms.add(int(atom["atom_number"]))

                        # For ECP elements (Z >= 37), basis files are named with +200 offset
                        _write_library_records(f, legacy_path, unique_atoms)
                        f.write("END\n")
                    else:
                        ui.err("ERROR: External basis set type selected but no valid path provided!")
                        ui.err(f"  basis_set_path: {settings.get('basis_set_path')}")
                        ui.err(f"  basis_set: {settings.get('basis_set')}")
                        raise ValueError("External basis set path not configured properly")
            else:
                # Internal basis set
                f.write("BASISSET\n")
                f.write(f"{settings.get('basis_set', 'POB-TZVP-REV2')}\n")

            # For UHF, add the UHF keyword
            if functional == "UHF":
                f.write("UHF\n")
        elif functional in ["PBEH3C", "HSE3C", "B973C", "PBESOL03C", "HSESOL3C"]:
            # DFT 3C methods
            f.write("BASISSET\n")
            f.write(f"{settings['basis_set']}\n")

            # Use write_dft_section to properly handle XLGRID for HSESOL3C
            write_dft_section(
                f,
                functional,
                False,  # No dispersion for 3C methods
                settings.get("dft_grid", "XLGRID"),
                settings.get("spin_polarized"),
            )
        else:
            # Standard basis set and method handling
            if settings.get("basis_set_type") == "EXTERNAL":
                # External basis set handling - need END to close geometry section
                f.write("END\n")
                # Write external basis set data
                if settings.get("use_original_external_basis") and external_basis_data:
                    # Use the external basis from the original file
                    for line in external_basis_data:
                        f.write(f"{line}\n")
                    f.write("99 0\n")
                    f.write("END\n")
                elif settings.get("basis_set_path"):
                    # Read basis sets from specified path
                    # Note: CRYSTAL doesn't support comments, so we just print the path info
                    ui.info(f"  External basis set from: {settings['basis_set_path']}")
                    unique_atoms = set()
                    for atom in coords_to_write:
                        unique_atoms.add(int(atom["atom_number"]))

                    # Read basis set files
                    # For ECP elements (Z >= 37), basis files are named with +200 offset
                    _write_library_records(f, settings["basis_set_path"], unique_atoms)
                    f.write("END\n")
                else:
                    # Fallback: basis_set_type is EXTERNAL but no path provided
                    # This can happen if basis_set contains a path (legacy behavior)
                    legacy_path = settings.get("basis_set")
                    if legacy_path and basis_library.path_exists(legacy_path):
                        ui.info(f"  Note: Using legacy basis_set field as path: {legacy_path}")
                        unique_atoms = set()
                        for atom in coords_to_write:
                            unique_atoms.add(int(atom["atom_number"]))

                        # For ECP elements (Z >= 37), basis files are named with +200 offset
                        _write_library_records(f, legacy_path, unique_atoms)
                        f.write("END\n")
                    else:
                        ui.err("ERROR: External basis set type selected but no valid path provided!")
                        ui.err(f"  basis_set_path: {settings.get('basis_set_path')}")
                        ui.err(f"  basis_set: {settings.get('basis_set')}")
                        raise ValueError("External basis set path not configured properly")
            else:
                # Internal basis set
                f.write("BASISSET\n")
                f.write(f"{settings.get('basis_set', 'POB-TZVP-REV2')}\n")

            # Write method section
            if method == "HF":
                # Handle HF methods
                if functional == "UHF":
                    f.write("UHF\n")
                # RHF is default, no keyword needed
            else:
                # Write DFT section
                write_dft_section(
                    f,
                    functional,
                    settings.get("dispersion"),
                    settings.get("dft_grid", "XLGRID"),
                    settings.get("spin_polarized"),
                    custom_functional=settings.get("custom_functional"),
                    custom_dftd3=parent_dispersion_input(settings),
                    custom_dftd3_in_dft=bool(settings.get("custom_dftd3_in_dft")),
                )

        # SCF parameters section
        atomic_numbers = [conventional_atom_number(atom["atom_number"], settings)
                          for atom in coords_to_write]

        # Check basis set compatibility. External basis records carried
        # verbatim from the parent (use_original_external_basis) are the
        # parent's own and are not checked: checking the (default) internal
        # basis against them raised false alarms for elements like Pb/Ag. A
        # basis read from an EXTERNAL library path (basis_set_path, or the
        # legacy path in basis_set) is checked against that library: a family
        # can lack an element the parent's basis had (full.basis.triplezeta
        # has no Kr). A deck that switches an external-basis parent to an
        # internal basis is checked like any other internal-basis deck (by its
        # plain atomic numbers).
        library_dir = _library_basis_dir(settings, external_basis_data)
        if (settings.get("basis_set_type") == "EXTERNAL" and library_dir is None
                and settings.get("functional") not in THREE_C_FUNCTIONALS):
            is_compatible, missing_elements = True, []
        elif library_dir is not None:
            basis_label = basis_family_name(library_dir)
            is_compatible, missing_elements = check_basis_set_compatibility(
                library_dir,
                [z - 200 if z > 200 else z for z in atomic_numbers],
                "EXTERNAL",
            )
        else:
            basis_label = settings.get("basis_set")
            is_compatible, missing_elements = check_basis_set_compatibility(
                settings.get("basis_set", "POB-TZVP-REV2"),
                atomic_numbers,
                settings.get("basis_set_type", "INTERNAL")
            )

        if not is_compatible:
            # CRYSTAL rejects a deck with no basis for one of its elements, so
            # it is never written: there is no "continue anyway?" (a workflow
            # callback, a batch or --yes had no one to ask, and a user who
            # said yes got a deck that could not run). The deck is partially
            # written at this point - discard it so the caller can't submit a
            # truncated d12, and signal failure.
            no_basis = (f"basis set '{basis_label}' has no basis for "
                        + ", ".join(f"{ATOMIC_NUMBER_TO_SYMBOL.get(z, str(z))} (Z={z})"
                                    for z in missing_elements))
            _fail(no_basis, f"Not writing {os.path.basename(output_file)}: {no_basis}.")
            f.discard()
            return False

        # Prepare k-points with same logic as d12creation.py
        k_points_info = None
        shrink_isp = None
        # The mesh came from the parent deck itself (not a config file from
        # another material, not regenerated from the cell).
        k_from_parent = (
            parent_k_points is not None
            and settings.get("k_points") == parent_k_points
        )
        if settings.get("k_points"):
            k_points_raw = settings["k_points"]
            
            # Handle different k-points formats from extraction
            if isinstance(k_points_raw, tuple) and len(k_points_raw) == 3:
                # Tuple format (ka, kb, kc) from extraction
                k_points_info = k_points_raw
            elif isinstance(k_points_raw, str):
                # String format like "12 12 12" from extraction
                try:
                    parts = k_points_raw.split()
                    if len(parts) == 3:
                        k_points_info = (int(parts[0]), int(parts[1]), int(parts[2]))
                    elif len(parts) == 2 and k_from_parent and int(parts[0]) > 0:
                        # One-line 'SHRINK IS ISP' form (the most common one):
                        # IS subdivisions along every reciprocal vector. Without
                        # this branch the mesh was regenerated from the cell
                        # (e.g. a parent's 5 10 became 7 14). Only the parent's
                        # own mesh is kept this way: a two-number mesh from a
                        # saved config or --shared-settings belongs to another
                        # material, so it is still regenerated from this cell.
                        is_ = int(parts[0])
                        k_points_info = (is_, is_, is_)
                        isp = int(parts[1])
                        if isp != 2 * is_:
                            shrink_isp = isp
                    elif len(parts) == 1:
                        # Single value - apply to all directions
                        k = int(parts[0])
                        k_points_info = (k, k, k)
                except (ValueError, IndexError):
                    k_points_info = None
                    
        # A mesh a config set for this deck (k_points_for_all_files, a
        # workflow step's kpoint_modifications) is written as given, its
        # non-periodic factors 1 and the factors of symmetry-equivalent
        # directions made equal (CRYSTAL stops on SHRINK BREAKS SYMMETRY).
        if k_points_info and k_from_parent and settings.get("k_points_set"):
            given = tuple(k_points_info)
            periodic = {"SLAB": 2, "POLYMER": 1}.get(dimensionality, 3)
            fixed = symmetric_k_points(given[:periodic] + (1,) * (3 - periodic), dimensionality,
                                       settings.get("spacegroup", 1))
            if fixed[:periodic] != given[:periodic]:
                ui.info(f"Note: Using k-points {' '.join(map(str, fixed[:periodic]))} (was "
                        f"{' '.join(map(str, given[:periodic]))}) so the mesh keeps "
                        f"{'layer' if dimensionality == 'SLAB' else 'space'} group "
                        f"{settings.get('spacegroup')}'s symmetry")
            k_points_info = fixed

        # Fallback: Generate k-points based on cell size if not extracted
        if k_points_info is None and geometry_data.get("conventional_cell"):
            a, b, c = [float(x) for x in geometry_data["conventional_cell"][:3]]
            k_points_info = generate_k_points(
                a, b, c, dimensionality, settings.get("spacegroup", 1)
            )

        # Enhanced k-points handling with symmetry consideration
        enhanced_k_points = k_points_info
        if k_points_info and dimensionality == "CRYSTAL":
            ka, kb, kc = k_points_info
            spacegroup = settings.get("spacegroup", 1)
            
            # For symmetrized structures (non-P1), prefer uniform k-points
            # This ensures compatibility with simplified SHRINK format
            # A mesh taken from the parent deck is kept as the parent wrote it.
            if not k_from_parent and spacegroup != 1 and (ka != kb or kb != kc or ka != kc):
                # Use maximum k-point for uniform sampling in symmetrized structures
                k_max = max(ka, kb, kc)
                enhanced_k_points = (k_max, k_max, k_max)
                ui.info(f"Note: Using uniform k-points ({k_max},{k_max},{k_max}) for symmetrized structure (space group {spacegroup})")
        elif k_points_info and dimensionality == "SLAB" and not k_from_parent:
            # A generated (or config) mesh must give the directions the layer
            # group makes equivalent the same factor, or CRYSTAL stops with
            # SHRINK BREAKS SYMMETRY. The parent's own mesh is kept as written.
            enhanced_k_points = slab_k_points(
                tuple(k_points_info), settings.get("spacegroup", 1),
                geometry_data.get("conventional_cell"),
            )
            if enhanced_k_points != tuple(k_points_info):
                ui.info(f"Note: Using k-points {enhanced_k_points[0]} {enhanced_k_points[1]} "
                        f"(was {k_points_info[0]} {k_points_info[1]}) so the mesh keeps "
                        f"layer group {settings.get('spacegroup')}'s symmetry")

        scf = settings.get("scf_settings") or {}
        scf_method = scf.get("method", "DIIS")
        # Parent-deck SCF records the writer cannot infer: BROYDEN's parameter
        # line, LEVSHIFT, and BIPOSIZE/EXCHSIZE. Sequences may be lists after
        # a JSON round-trip of saved options.
        scf_extra = {}
        broyden = scf.get("broyden")
        if scf_method == "BROYDEN" and broyden and len(broyden) >= 3:
            scf_extra.update(broyden_w0=broyden[0], broyden_imix=int(broyden[1]),
                             broyden_istart=int(broyden[2]))
        levshift = scf.get("levshift")
        if levshift and len(levshift) >= 2:
            scf_extra["levshift"] = (int(levshift[0]), int(levshift[1]))
        for size_key in ("biposize", "exchsize"):
            if scf.get(size_key) is not None:
                scf_extra[size_key] = int(scf[size_key])
        # HISTDIIS as the parent deck (or the answer) has it; None = no record.
        # Settings that never saw a deck keep the writer's default.
        if "histdiis" in scf:
            scf_extra["histdiis"] = int(scf["histdiis"]) if scf["histdiis"] else None
        # The parent's GUESSP restart request. submitcrystal23.sh drops the
        # record when it has no density matrix to stage, so carrying it is safe.
        if scf.get("guessp"):
            scf_extra["guessp"] = True
        # SPINLOCK needs SPIN (DFT) or UHF in this deck; in a closed-shell run
        # CRYSTAL aborts on it. A UHF deck carries no SPIN keyword.
        spin_active = bool(settings.get("spin_polarized")) or settings.get("functional") == "UHF"

        # The SCF cycle limit. A FREQ (or ELASTIC/EOS/QHA/CPHF) deck runs at a
        # tighter TOLDEE than its parent, so it gets at least 200 cycles; a
        # parent's higher value is kept, and a parent with none gets 800.
        scf_maxcycle = scf.get("maxcycle", SCF_MAXCYCLE_FALLBACK)
        if calc_type in TIGHT_SCF_CALC_TYPES and settings.get("scf_maxcycle_explicit"):
            # MAXCYCLE set for this deck (a workflow step's scf_modifications):
            # written as given; the floor is for the parent's value only.
            if int(scf_maxcycle) < TIGHT_SCF_MIN_MAXCYCLE:
                ui.warn(f"  Warning: SCF MAXCYCLE {scf_maxcycle} as set, below the {TIGHT_SCF_MIN_MAXCYCLE} "
                        f"{a_calc(calc_type)} deck otherwise gets; its tighter TOLDEE may need more cycles")
        elif calc_type in TIGHT_SCF_CALC_TYPES:
            scf_maxcycle = tight_scf_maxcycle(scf.get("maxcycle"))
            if scf_maxcycle != scf.get("maxcycle"):
                ui.info(f"  SCF MAXCYCLE {scf_maxcycle} (the parent's "
                        f"{scf.get('maxcycle') or 'CRYSTAL default'} is too few for "
                        f"{calc_type}'s tighter TOLDEE)")

        write_scf_section(
            f,
            settings.get("tolerances", DEFAULT_TOLERANCES),
            enhanced_k_points,
            dimensionality,
            settings.get("smearing"),
            settings.get("smearing_width", 0.01),
            scf_method,
            scf_maxcycle,
            scf.get("fmixing", 30),
            len(atomic_numbers),
            settings.get("spacegroup", 1),
            # Carry a configured fixed spin state through on the OPT-continuation
            # path too, gated on spin polarization (0/None => unchanged output).
            spinlock=(settings.get("spinlock", 0) if spin_active else 0),
            # The parent's own 'SPINLOCK / 0 N' is kept as written.
            write_zero_spinlock=bool(spin_active and settings.get("spinlock_explicit")
                                     and not settings.get("spinlock")),
            # Preserve the parsed/configured SCF-cycle count for the lock; without
            # this the writer falls back to DEFAULT_SPINLOCK_CYCLES (50) and a deck
            # with e.g. 'SPINLOCK 2 30' is silently regenerated as '2 50'.
            spinlock_cycles=settings.get("spinlock_cycles", DEFAULT_SPINLOCK_CYCLES),
            preserve_directional=k_from_parent,
            shrink_isp=shrink_isp if k_from_parent else None,
            gradcal=calc_type == "PHONOPY",
            **scf_extra,
        )

        # Note: The single END at the very end is written by write_scf_section

        if settings.get("soc"):
            # A deck the 2c-SCF cannot run (OPT, FREQ, ...) is refused first,
            # with no SMEAR warning or question for a deck never written.
            refusal = f.refusal()
            if refusal:
                f.discard()
            else:
                # A parent that looks metallic (parent_metal: why) and no SMEAR:
                # warn, and ask for one or note it (ask_for_soc_smear).
                if parent_metal and not f.will_have_smear():
                    ask_for_soc_smear(f, parent_metal, ask)
                refusal = f.finish()
            if refusal:
                _fail(refusal, f"Not writing {os.path.basename(output_file)}: {refusal}")
                return False

    return True


def _keep_extracted_settings(settings, calc_type, opt_type, origin_setting):
    """Options for the truly non-interactive path.

    Keeps every setting extracted from the source calculation and changes only
    the calculation type. Shared by --non-interactive on its own and by
    --non-interactive --calc-type when no answers are available on stdin.
    """
    # True non-interactive mode (no config file, no calc type specified)
    options = settings.copy()
    # Default to SP if not specified
    options["calculation_type"] = calc_type
    if calc_type == "FREQ":
        # FREQ gets its own default SCF tolerances (Very tight, or the parent's
        # values where tighter)
        options["tolerances"] = freq_default_tolerances(
            settings.get("calculation_type"), settings.get("tolerances"))

    # Set optimization type if it's an OPT calculation
    if options["calculation_type"] == "OPT":
        parent_opt = dict(options.get("optimization_settings") or {})
        if opt_type:
            # The writer takes the type inside optimization_settings over
            # optimization_type, so the parent's own type has to go too, or
            # --opt-type was ignored for every OPT parent.
            parent_opt["type"] = opt_type
            options["optimization_settings"] = parent_opt
            options["optimization_type"] = opt_type
        else:
            # The parent's own optimization type, else FULLOPTG
            options["optimization_type"] = parent_opt.get("type") or "FULLOPTG"

    # Handle origin setting: "auto" preserves the origin extracted from
    # the source calculation. The old behavior guessed a directive from
    # a space-group table ("0 1 0" rhombohedral flag for sg 143-194,
    # "0 0 1" shifted origin for everything else), which rewrote correct
    # origins — e.g. Fd-3m "0 0 0" re-emitted as the origin-2 "0 0 1"
    # form while keeping origin-1 coordinates (wrong structure).
    if origin_setting == "auto":
        options["origin_setting"] = settings.get("origin_setting", "0 0 0")
    else:
        options["origin_setting"] = origin_setting

    # Keep all other settings from the extracted data
    if "write_only_unique" not in options:
        # Check if original input had space group > 1 (not P1)
        if settings.get("spacegroup", 1) > 1:
            # For symmetric structures, default to writing only unique atoms
            options["write_only_unique"] = True
        else:
            # For P1 structures, write all atoms
            options["write_only_unique"] = False

    # A parent functional MACE could not identify: warn and use HSE06.
    ensure_known_functional(options)

    ui.info("\nRunning in non-interactive mode with settings:")
    ui.info(f"  Calculation type: {options['calculation_type']}")
    if options['calculation_type'] == 'OPT':
        ui.info(f"  Optimization type: {options['optimization_type']}")
    ui.info(f"  Origin setting: {options['origin_setting']}")
    return options


# Line-2 geometry kinds whose atoms are not in the deck (CRYSTAL23 manual
# sec. 2.1: EXTERNAL reads fort.34, DLVINPUT the DLV file) and the one that
# opt2d12's writer has no records for.
DECK_GEOMETRY_ELSEWHERE = ("EXTERNAL", "DLVINPUT")
DECK_KINDS_WRITTEN = ("CRYSTAL", "SLAB", "POLYMER", "MOLECULE")


def deck_parent_refusal(deck_file):
    """Why the deck cannot be its own parent (None when it can).

    A deck is its own parent when its geometry is its records: not one read
    from another file (EXTERNAL, DLVINPUT), not one that block-1 records
    then edit (SUPERCEL, ATOMSUBS, SLABCUT, ... - mace.preopt.deckscan's
    table: the child, written from the records, would lose the edit).
    GHOSTS is basis-set input and goes with the charge records."""
    name = os.path.basename(deck_file)
    stem = os.path.splitext(name)[0]
    with open(deck_file, errors="replace") as f:
        text = f.read()
    lines = text.splitlines()
    kind = lines[1].strip().split()[0].upper() if len(lines) > 1 and lines[1].strip() else ""
    run_out = f"set up the new deck from its run's .out (mace opt2d12 --out-file {stem}.out)"
    if kind in DECK_GEOMETRY_ELSEWHERE:
        return (f"{name} has no geometry of its own: {kind} reads it from fort.34 when the job runs; "
                + run_out)
    if kind not in DECK_KINDS_WRITTEN:
        return f"{name} is a {kind or 'deck with no'} geometry opt2d12 does not write; " + run_out
    try:
        from mace.preopt.deckscan import scan
        # Only the records after line 2: scan() reports a line-2 kind other
        # than CRYSTAL as an edit, and lists each keyword once, so filtering
        # the kinds out of its list also dropped a real MOLECULE edit (the
        # 3D-to-0D extraction) of a CRYSTAL deck. With line 2 read as CRYSTAL,
        # every keyword it reports is an edit written after the atoms.
        after_line2 = "\n".join(lines[:1] + ["CRYSTAL"] + lines[2:])
        edits = [e for e in scan(after_line2)["geometry_edits"] if e != "GHOSTS"]
    except ImportError:                     # standalone, without the mace package
        edits = []
    if edits:
        return (f"{name} builds its structure with {', '.join(edits)} after the atoms it lists, so the "
                "new deck would have the structure before that edit; " + run_out
                + ", where CRYSTAL printed the structure it built")
    return None


def deck_geometry_data(in_data):
    """The geometry_data the deck writer takes, from the deck's own records
    (what CrystalOutputParser gives for an .out): its cell and its atoms as
    written, in the deck's own group, so the writer prints them back byte for
    byte (tests/test_deck_round_trip.py). An atom keeps the number it was
    written with unless its record ends in its element symbol (the writer
    adds 200 for an ECP element of an EXTERNAL basis itself)."""
    cp = in_data.get("cell_parameters")
    cell = None
    if cp:
        cell = [cp["a"], cp["b"] if cp["b"] is not None else cp["a"],
                cp["c"] if cp["c"] is not None else 500.0,
                cp["alpha"], cp["beta"], 90.0 if cp["gamma"] is None else cp["gamma"]]
    coords = []
    for atom, xyz, label in zip(in_data["atoms"], in_data["atom_coordinate_text"],
                                in_data["atom_labels"]):
        number = atom["atom_number"] if label in (None, "X") else atom["atomic_number"]
        x, y, z = xyz
        coords.append({"atom_number": str(number), "x": x, "y": y, "z": z})
    group = in_data.get("spacegroup")
    if in_data.get("dimensionality") in LOW_DIM_GROUPS:
        group = in_data.get(LOW_DIM_GROUPS[in_data["dimensionality"]][0]) or group
    return {"conventional_cell": cell, "coordinates": coords, "crystallographic_coordinates": list(coords),
            "spacegroup": group, "dimensionality": in_data.get("dimensionality", "CRYSTAL"),
            "origin_setting": in_data.get("origin_setting", "0 0 0"), "optimization_content": ""}


def read_deck_parent(deck_file):
    """(geometry_data, None) for a deck that is its own parent, or (None, why)."""
    why = deck_parent_refusal(deck_file)
    if why:
        return None, why
    try:
        in_data = CrystalInputParser(deck_file).parse()
    except Exception as e:
        return None, f"could not read {os.path.basename(deck_file)}: {e}"
    if in_data.get("geometry_unparsed") or not in_data.get("atoms"):
        return None, (f"could not read the geometry records of {os.path.basename(deck_file)}"
                      + (f": {in_data['geometry_unparsed']}" if in_data.get("geometry_unparsed") else ""))
    return deck_geometry_data(in_data), None


def ask_shared_elastic_settings(shared_settings, template_settings):
    """``shared_settings`` with "elastic_settings" for an ELASTIC, EOS, QHA or
    CPHF batch: the answers to the module's questions (d12_calc_elastic), asked
    once, with the template parent's own block as the defaults when it is the
    same module. Nothing is asked when --elastic-settings gave them."""
    calc_type = (shared_settings or {}).get("calculation_type")
    if calc_type not in ("ELASTIC", "EOS", "QHA", "CPHF"):
        return shared_settings
    import d12_calc_elastic as elastic
    if elastic.REQUESTED:
        return shared_settings
    current = template_settings.get("elastic_settings")
    mode = str((current or {}).get("mode") or "").upper()
    if not (mode == calc_type or (calc_type == "ELASTIC" and mode in elastic.ELASTIC_MODES)):
        current = None
    print()
    ui.rule(f"{calc_type} SETTINGS (all files)")
    shared_settings["elastic_settings"] = elastic.configure_elastic(
        calc_type, template_settings.get("dimensionality", "CRYSTAL"), current=current)
    return shared_settings


def process_files(*args, **kwargs):
    """Process CRYSTAL output and input files (see _process_files).

    A phonon dispersion refused as soon as it is chosen (a POLYMER, say)
    ends the file there, before the questions that follow it. So does a
    parent a PHONOPY set or an ELASTIC/EOS/QHA/CPHF deck cannot start from
    (ParentRefused: its lines go to stderr).
    """
    import d12_interactive
    try:
        return _process_files(*args, **kwargs)
    except PhononDispersionRefused as exc:
        _fail(str(exc), f"\nNot writing a deck: {exc}")
        return False, None
    except ParentRefused as exc:
        for line in exc.lines:
            print(line, file=sys.stderr)
        LAST_RESULT["reason"] = " ".join(exc.lines[-1].split())
        return False, None
    finally:
        d12_interactive.FOLLOW_UP_CHECK = None


def follow_up_check(kind, output_file, input_file, config_file=None):
    """Raise ParentRefused when this parent cannot seed a ``kind`` deck: a
    PHONOPY set always (d12_calc_phonopy.parent_refusal), an ELASTIC, EOS, QHA
    or CPHF deck asked for here (not a workflow step's config file, nor a
    RESTART of the parent's own run: d12_calc_elastic.parent_refusal)."""
    lines = None
    if kind == "PHONOPY":
        import d12_calc_phonopy
        lines = d12_calc_phonopy.parent_refusal(output_file, input_file)
    elif kind in ("ELASTIC", "EOS", "QHA", "CPHF") and output_file and not config_file:
        import d12_calc_elastic
        if not (d12_calc_elastic.REQUESTED or {}).get("restart"):
            lines = d12_calc_elastic.parent_refusal(output_file)
    if lines:
        raise ParentRefused(lines)


def _process_files(output_file, input_file=None, shared_settings=None, config_file=None, non_interactive=False, calc_type=None, opt_type=None, origin_setting="auto",
                  output_dir=None, confirm_config=None, unattended=False,
                  reserved_decks=None, charge_options=None):
    """Process CRYSTAL output and input files

    Args:
        output_file: Path to .out file
        input_file: Path to .d12 file (optional)
        shared_settings: Pre-defined settings to use (optional)
        config_file: Path to JSON config file (optional)
        non_interactive: Run in non-interactive mode (optional)
        calc_type: Calculation type for non-interactive mode (optional)
        opt_type: Optimization type for non-interactive mode (optional)
        origin_setting: Origin setting for non-interactive mode (optional)
        confirm_config: ask "Apply these settings from config file?" (True),
            apply without asking (False), or ask only when stdin is a terminal
            (None, the default). With nothing on stdin the question ended the
            run with "EOF when reading a line".
        unattended: nobody is there to answer (a batch, --yes,
            --non-interactive). Nothing is asked, whether or not stdin is a
            terminal: a question that would have been asked fails the file
            with its reason instead, so the result is the same either way.
        reserved_decks: {deck path: .out} of the decks this run has written;
            a deck that would overwrite one of them fails instead.
        charge_options: ion_charges / net_charge / site_states / chemod /
            fragment_bond_scale / spinlock / spinlock_cycles for a charged
            system, recomputed from the new geometry (ion_charges.py); the
            same keys in a config file apply too, these win.

    Returns:
        tuple: (success, settings_used). LAST_RESULT then holds the deck
        written, or the reason nothing was.
    """
    LAST_RESULT["deck"] = None
    LAST_RESULT["reason"] = None
    LAST_RESULT["notes"] = []
    answers_piped = False
    if unattended:
        confirm_config = False

    # A follow-up that needs a finished OPT or SP run refuses any other parent
    # before anything is asked: --calc-type here, menu choices 4-8 once chosen.
    import d12_interactive
    d12_interactive.FOLLOW_UP_CHECK = (
        lambda kind: follow_up_check(kind, output_file, input_file, config_file))
    if calc_type:
        follow_up_check(calc_type, output_file, input_file, config_file)

    if output_file is None:
        # --d12-file alone: the deck is its own parent (its geometry records
        # and its settings), as an .out's parent deck gives its settings
        out_data, why = read_deck_parent(input_file)
        if why:
            _fail(why, f"Not writing a deck: {why}")
            return False, None
        ui.info(f"\nGeometry from the deck itself: {input_file} (its records as written, "
                "not a geometry CRYSTAL optimized)")
    else:
        # Parse output file
        ui.info(f"\nParsing output file: {output_file}")
        out_parser = CrystalOutputParser(output_file)
        try:
            out_data = out_parser.parse()
        except Exception as e:
            _fail(f"could not parse the output file: {e}", f"Error parsing output file: {e}")
            return False, None

    opt_problem = optimization_problem(output_file) if output_file else None
    if opt_problem:
        ui.warn(f"Warning: {os.path.basename(output_file)}: {opt_problem}")
    has_parent_deck = bool(input_file and os.path.exists(input_file))

    # Parse input file if provided
    settings = out_data.copy()
    external_basis_data = []
    parent_k_points = None
    # The parent deck's SCELPHONO expansion, if it ran a phonon dispersion.
    parent_scelphono = None

    if input_file and os.path.exists(input_file):
        ui.info(f"Parsing input file: {input_file}")
        in_parser = CrystalInputParser(input_file)
        try:
            in_data = in_parser.parse()
            parent_k_points = in_data.get("k_points")
            parent_scelphono = (in_data.get("freq_settings") or {}).get("scelphono")

            # Merge data, with special handling for DFT settings
            for key, value in in_data.items():
                if key in DECK_GEOMETRY_KEYS or key in DECK_TEXT_KEYS:
                    # The parent's geometry input; the child's geometry is the .out's.
                    continue
                if key == "freq_settings":
                    value = child_freq_settings(value)
                if key not in settings or settings[key] is None:
                    settings[key] = value
                elif key in ["functional", "dispersion", "spin_polarized", "dft_grid", "method",
                           "is_3c_method", "smearing", "use_smearing", "smearing_width",
                           "k_points", "scf_method", "scf_maxcycle", "fmixing", "scf_direct",
                           "mulliken_analysis", "diis_history", "calculation_type",
                           "optimization_settings", "freq_settings", "origin_setting",
                           "spacegroup", "dimensionality", "tolerances",
                           "basis_set", "basis_set_type", "basis_set_path",
                           "unrecognised_functional", "unrecognised_functional_source",
                           "layer_group", "rod_group"]:
                    # For all calculation settings, prefer input file (.d12) over output file (.out)
                    # because .d12 contains the original user-specified settings
                    # INCLUDING tolerances and basis set - the output parser has issues extracting these correctly
                    if value is not None:
                        settings[key] = value
                        # Debug output for symmetry-related settings
                        if key in ["origin_setting", "spacegroup", "dimensionality"]:
                            ui.info(f"  Preserving {key} from D12 file: {value} (was {settings.get(key, 'not set')} from output)")
                        elif key in ["basis_set", "basis_set_type"]:
                            ui.info(f"  Preserving {key} from D12 file: {value}")
                elif key == "scf_settings":
                    # Merge SCF settings
                    if "scf_settings" not in settings:
                        settings["scf_settings"] = {}
                    settings["scf_settings"].update(value)

            prefer_deck_unrecognised_functional(settings, in_data)

            # Store external basis data from D12 file
            external_basis_data = in_data.get("external_basis_data", [])

            # If we have external basis data from D12, mark it for potential reuse
            if external_basis_data:
                settings["has_original_external_basis"] = True
                ui.info(f"  Found external basis set data in D12 file ({len(external_basis_data)} lines)")
        except Exception as e:
            ui.warn(f"Warning: Error parsing input file: {e}")
            ui.warn("Continuing with output file data only")
            has_parent_deck = False

    use_low_dim_group(settings)
    parent_supercell = write_supercell_in_p1(out_data, settings)

    # Set defaults if not found
    if not settings.get("spacegroup"):
        if settings["dimensionality"] == "MOLECULE":
            settings["spacegroup"] = 1
        else:
            ui.warn("Warning: Space group not found. Defaulting to P1")
            settings["spacegroup"] = 1

    # Handle external basis sets - this must be checked BEFORE setting defaults
    # to ensure external basis data is always used when available
    if external_basis_data and settings.get("has_original_external_basis"):
        # Always enable external basis reuse when data is available from D12
        settings["basis_set_type"] = "EXTERNAL"
        settings["use_original_external_basis"] = True
        if not settings.get("basis_set"):
            settings["basis_set"] = "EXTERNAL (from original D12)"
        # Said once the config or answers have chosen the method (below):
        # an HF-3c or DFT-3c child is written with its own basis instead.
    elif not settings.get("basis_set"):
        # No external basis and no basis set specified - use default internal
        settings["basis_set"] = "POB-TZVP-REV2"
        settings["basis_set_type"] = "INTERNAL"

    # Set default tolerances if not found
    if not settings.get("tolerances"):
        settings["tolerances"] = DEFAULT_TOLERANCES.copy()

    # Set default SCF settings if not found
    if not settings.get("scf_settings"):
        settings["scf_settings"] = {"method": "DIIS", "maxcycle": 800, "fmixing": 30}

    # Get user options or use shared settings
    if config_file:
        # Config file takes precedence - process it first
        # Load settings from config file
        ui.info(f"\nLoading settings from config file: {config_file}")
        try:
            with open(config_file, 'r') as f:
                # example_configs/*.json and save_d12_config wrap the
                # settings in {"version", "type", "configuration"}
                config_data = unwrap_d12_config(json.load(f))

            # Legacy functional spellings ("HSEsol-3C", "M06-2X", "HF") become
            # the CRYSTAL23 keyword, as a workflow plan's are when it is
            # loaded; an unknown name writes nothing.
            fn_msgs, fn_errs = normalize_config_functionals(config_data)
            if fn_errs:
                _fail(fn_errs[0], "The config file names a functional CRYSTAL23 does not "
                                  "accept; nothing written: " + "; ".join(fn_errs))
                return False, None
            for msg in fn_msgs:
                ui.warn(f"  Config: {msg}")

            # Show config summary
            print()
            ui.rule("CONFIG FILE SETTINGS")
            ui.info(f"Calculation type: {config_data.get('calculation_type', 'Not specified')}")

            # Check for method_modifications
            if 'method_modifications' in config_data:
                method_mods = config_data['method_modifications']
                if 'new_functional' in method_mods:
                    ui.info(f"Functional: {method_mods['new_functional']} (via method_modifications)")
                elif 'functional' in method_mods:
                    ui.info(f"Functional: {method_mods['functional']} (via method_modifications)")
                else:
                    ui.info(f"Functional: {config_data.get('functional', 'Not specified')}")
            else:
                ui.info(f"Method: {config_data.get('method', 'Not specified')}")
                ui.info(f"Functional: {config_data.get('functional', 'Not specified')}")

            if config_data.get('dispersion'):
                ui.info(f"Dispersion: Yes")
            ui.info(f"Basis set: {config_data.get('basis_set', 'Not specified')}")
            ui.info(f"DFT grid: {config_data.get('dft_grid', 'Not specified')}")

            # Show tolerance modifications if present
            if 'tolerance_modifications' in config_data:
                tol_mods = config_data['tolerance_modifications']
                if 'custom_tolerances' in tol_mods:
                    ui.info(f"Custom tolerances: {tol_mods['custom_tolerances']}")

            ui.rule()

            # Ask user if they want to apply these settings (skip in non-interactive mode)
            if confirm_config is None:
                confirm_config = stdin_is_terminal()
            if non_interactive:
                apply_config = True
                ui.info("\nApplying config file settings (non-interactive mode).")
            elif not confirm_config:
                # Passing --config-file asks for its settings. A batch confirms
                # once for all files, --yes skips the question, and with no
                # terminal there is nobody to ask: a piped "y" applied the
                # settings, and the same settings are applied without reading it.
                apply_config = True
                ui.info("\nApplying config file settings.")
            else:
                apply_config = yes_no_prompt("\nApply these settings from config file?", default="yes")
            
            if apply_config:
                # Preserve external basis settings before config override
                # These must be maintained unless config explicitly provides new basis settings
                had_external_basis = settings.get("use_original_external_basis", False)
                external_basis_type = settings.get("basis_set_type")

                # Use settings from config file
                options = settings.copy()
                # Override with config file settings — but never the target's
                # geometry identity: a config saved from one material carries
                # its spacegroup/dimensionality, and applying it to another
                # material wrote the wrong symmetry (e.g. sg-227 diamond
                # options applied to an sg-166 polymorph)
                geometry_identity_keys = [
                    "coordinates", "primitive_cell", "conventional_cell",
                    "spacegroup", "space_group", "dimensionality",
                    "origin_setting", "origin_shift", "cell_parameters", "lattice_parameters",
                    "layer_group", "rod_group",
                ]
                # A template that stores its own parent's EXTERNAL basis (the
                # marker, plus that parent's basis records in templates saved
                # before this release) means "each parent's own basis": this
                # structure keeps the basis its parent deck was written with.
                # Applied as a basis it failed on every internal-basis parent
                # ("External basis set path not configured properly").
                keep_parent_basis = template_uses_parent_basis(config_data)
                if keep_parent_basis and has_parent_deck:
                    ui.info("  Basis: keeping this structure's own parent basis "
                            "(the config stores its parent's external basis)")
                elif keep_parent_basis:
                    ui.warn(f"  Basis: {settings.get('basis_set')} - converted from the .out "
                            "alone (no parent .d12, so no parent basis to keep)")
                for key, value in config_data.items():
                    if key in geometry_identity_keys:
                        continue
                    if key in TEMPLATE_STRUCTURE_DATA_KEYS:
                        # Another structure's atoms or basis records.
                        continue
                    if keep_parent_basis and key in TEMPLATE_BASIS_KEYS:
                        continue
                    if key in ("k_points", ALL_FILES_K_POINTS_KEY):
                        # Handled below: a derived deck keeps its parent's mesh.
                        continue
                    options[key] = value
                if calc_type:
                    # An explicit --calc-type (and --opt-type) wins over the
                    # config and over the parent's own type, which a config
                    # without calculation_type left in place: --calc-type SP
                    # wrote an OPT deck from an OPT parent.
                    if options.get("calculation_type") != calc_type:
                        ui.info(f"  Calculation type: {calc_type} (--calc-type)")
                    options["calculation_type"] = calc_type
                    apply_opt_type_flag(options, opt_type)
                if "k_points" in config_data:
                    # --save-options before this release stored the saving
                    # structure's own mesh; applied here it made every mesh be
                    # regenerated from the cell (Ag1Cl3's 8 16 became 9 18).
                    ui.info(f"  k-points: this structure's own (the config's k_points "
                            f"{config_data['k_points']} is the mesh of the structure it was "
                            f"saved from; {ALL_FILES_K_POINTS_KEY} sets one for every file)")
                if ALL_FILES_K_POINTS_KEY in config_data:
                    mesh = all_files_k_points(config_data[ALL_FILES_K_POINTS_KEY])
                    if mesh is None:
                        _fail(f"{ALL_FILES_K_POINTS_KEY} in the config must be 1 to 3 "
                              f"whole numbers, not {config_data[ALL_FILES_K_POINTS_KEY]!r}")
                        return False, None
                    # Written as given, as a parent's own mesh is.
                    parent_k_points = set_k_mesh(options, mesh, f"the config's {ALL_FILES_K_POINTS_KEY}")
                # A workflow step's own mesh (kpoint_modifications) over the
                # parent's; GUESSP stays (d12_constants.KPOINT_MODIFICATION_KEYS).
                if config_data.get("kpoint_modifications") is not None:
                    try:
                        k_mods = check_kpoint_modifications(config_data["kpoint_modifications"])
                    except ValueError as exc:
                        _fail(str(exc), f"The config's {exc}; nothing written.")
                        return False, None
                    parent_k_points = set_k_mesh(options, k_mods["new_k_points"], "this step's kpoint_modifications")
                # An external basis directory named by the config is used for
                # this structure too, not the parent deck's own records.
                if (not keep_parent_basis and config_data.get("basis_set_type") == "EXTERNAL"
                        and config_data.get("basis_set_path")
                        and "use_original_external_basis" not in config_data):
                    options["use_original_external_basis"] = False

                # A plan step's optimization_settings override only the OPTGEOM
                # values it names: the parent's others (MAXTRADIUS, a TOLDEE
                # the step leaves out) stay, instead of the step replacing the
                # parent's OPTGEOM wholesale.
                if isinstance(config_data.get("optimization_settings"), dict):
                    options["optimization_settings"] = merge_optimization_settings(
                        settings.get("optimization_settings"),
                        config_data["optimization_settings"],
                        replace_type="optimization_type" in config_data,
                    )

                # Restore external basis settings if they were set and config didn't override them
                # This ensures workflow-generated configs (which only set functional/calc_type)
                # don't accidentally disable external basis handling
                if had_external_basis and "basis_set_type" not in config_data and "basis_set_path" not in config_data:
                    options["use_original_external_basis"] = True
                    options["basis_set_type"] = external_basis_type

                # Handle both "frequency_settings" (from workflow) and "freq_settings" (direct usage)
                freq_key = None
                if "frequency_settings" in options:
                    freq_key = "frequency_settings"
                elif "freq_settings" in options:
                    freq_key = "freq_settings"
                
                if freq_key and isinstance(options[freq_key], dict):
                    # Rename to freq_settings for consistency with rest of script
                    if freq_key == "frequency_settings":
                        options["freq_settings"] = options.pop("frequency_settings")
                    # A FREQ parent's NUMDERIV stays unless the config names
                    # one (None: no keyword), as a FREQ child of opt2d12 keeps
                    # the parent's FREQCALC block.
                    parent_numderiv = (settings.get("freq_settings") or {}).get("numderiv")
                    if "numderiv" not in options["freq_settings"] and parent_numderiv is not None:
                        options["freq_settings"] = dict(options["freq_settings"],
                                                        numderiv=parent_numderiv)
                        ui.info(f"  NUMDERIV {parent_numderiv}: kept from the parent")
                    # So do its IR/Raman intensity and spectrum settings
                    # when the config names none of them (the planner's
                    # default FREQ steps name none) and runs in the parent's
                    # mode: a phonon dispersion child of a Gamma IR parent
                    # gets none, as opt2d12's phonon templates write none.
                    parent_freq = settings.get("freq_settings") or {}
                    parent_intens = {k: v for k, v in parent_freq.items()
                                     if k in INTENSITY_SETTINGS}
                    if (parent_intens and not INTENSITY_SETTINGS & set(options["freq_settings"])
                            and freq_mode(options["freq_settings"]) == freq_mode(parent_freq)):
                        options["freq_settings"] = dict(options["freq_settings"], **parent_intens)
                        ui.info("  IR/Raman intensity settings: kept from the parent")
                    
                    # Convert temprange from dict to tuple if needed
                    if "temprange" in options["freq_settings"] and isinstance(options["freq_settings"]["temprange"], dict):
                        temprange_dict = options["freq_settings"]["temprange"]
                        options["freq_settings"]["temprange"] = (
                            temprange_dict.get("n_temps", 20),
                            temprange_dict.get("t_min", 0),
                            temprange_dict.get("t_max", 400)
                        )
                    # Also convert pressrange if it exists
                    if "pressrange" in options["freq_settings"] and isinstance(options["freq_settings"]["pressrange"], dict):
                        pressrange_dict = options["freq_settings"]["pressrange"]
                        options["freq_settings"]["pressrange"] = (
                            pressrange_dict.get("n_press", 20),
                            pressrange_dict.get("p_min", 0),
                            pressrange_dict.get("p_max", 10)
                        )
                
                # A structure with symmetry is written as its asymmetric unit,
                # whatever the config says: "write all atoms" in a template
                # saved from a P1 structure (a molecule) listed every atom of
                # another structure's cell under its space group record, so
                # CRYSTAL would generate each symmetry copy again.
                if (options.get("write_only_unique") is False
                        and (settings.get("spacegroup") or 1) > 1):
                    options["write_only_unique"] = True
                    ui.info("  Writing the asymmetric unit (this structure has "
                            f"space group {settings.get('spacegroup')})")

                # Set write_only_unique if not specified in config
                if "write_only_unique" not in options:
                    # Check if original input had space group > 1 (not P1)
                    if settings.get("spacegroup", 1) > 1:
                        # For symmetric structures, default to writing only unique atoms
                        options["write_only_unique"] = config_data.get("write_only_unique", True)
                    else:
                        # For P1 structures, write all atoms
                        options["write_only_unique"] = config_data.get("write_only_unique", False)
                
                # Handle method_modifications if present
                if "method_modifications" in config_data:
                    method_mods = config_data["method_modifications"]
                    if "new_functional" in method_mods:
                        options["functional"] = method_mods["new_functional"]
                        ui.info(f"  Functional changed to: {method_mods['new_functional']}")
                    elif "functional" in method_mods:
                        options["functional"] = method_mods["functional"]
                        ui.info(f"  Functional changed to: {method_mods['functional']}")

                # Check if the selected functional is a 3C method: named under
                # method_modifications, or as the config's own "functional"
                # when the config names no basis (it was left at the parent's,
                # written "BASISSET / EXTERNAL (from original D12)")
                method_mods = config_data.get("method_modifications") or {}
                functional_name = method_mods.get("new_functional") or method_mods.get("functional")
                if not functional_name and not config_data.get("basis_set"):
                    functional_name = config_data.get("functional")
                if functional_name:
                    # If it's a 3C method, update basis set
                    if functional_name in ["PBEH3C", "HSE3C", "B973C", "PBESOL03C", "HSESOL3C", "HF3C", "HFSOL3C"]:
                        options["is_3c_method"] = True
                        options["basis_set_type"] = "INTERNAL"
                        # Get the basis set for this 3C method
                        # FUNCTIONAL_CATEGORIES already imported from d12_constants
                        for category in FUNCTIONAL_CATEGORIES.values():
                            if "basis_requirements" in category:
                                if functional_name in category["basis_requirements"]:
                                    options["basis_set"] = category["basis_requirements"][functional_name]
                                    ui.info(f"  Basis set updated to: {options['basis_set']} (required for 3C method)")
                        # For HSESOL3C, ensure XLGRID is set
                        if functional_name == "HSESOL3C":
                            options["dft_grid"] = "XLGRID"
                            ui.info(f"  DFT grid set to: XLGRID (required for HSESOL3C)")
                if "keep_spin" in method_mods and not method_mods["keep_spin"]:
                    options["spin_polarized"] = False
                if "keep_grid" in method_mods and not method_mods["keep_grid"]:
                    options["dft_grid"] = None

                # A functional named by the config (a plan step, a saved
                # config) is used exactly as written: it says itself whether it
                # carries -D3. Without this the parent's dispersion flag stayed
                # on, and a plan's PBE0 on a B3LYP-D3 parent was written PBE0-D3.
                # An explicit "dispersion" in the config still wins.
                explicit_functional = ((config_data.get("method_modifications") or {}).get("new_functional")
                                       or (config_data.get("method_modifications") or {}).get("functional")
                                       or config_data.get("functional"))

                # "method": "HF" asks for a Hartree-Fock deck. Its flavor is
                # "hf_method" or "functional", RHF when neither names one - the
                # way cif2d12 reads the same file (d12_config.config_to_cif_options).
                # A null functional reached the filename code and stopped the
                # file with "argument of type 'NoneType' is not iterable", and a
                # config naming only the method kept the parent's DFT functional.
                if (str(config_data.get("method") or "").upper() == "HF"
                        and not (config_data.get("method_modifications") or {}).get("new_functional")
                        and not (config_data.get("method_modifications") or {}).get("functional")):
                    hf_method = str(config_data.get("hf_method") or config_data.get("functional")
                                    or "RHF").upper()
                    if hf_method not in ("RHF", "UHF", "HF3C", "HFSOL3C"):
                        _fail(f"the config's method is HF but its functional is {hf_method}",
                              f"The config's method is HF but its functional is {hf_method}; "
                              f"an HF config names RHF, UHF, HF3C or HFSOL3C.")
                        return False, None
                    options["functional"] = explicit_functional = hf_method
                    options["method"] = "HF"
                    # HF3C / HFSOL3C are defined for a pure HF calculation in
                    # the MINIX / SOLMINIX basis (manual 5.3.1 p.158, 5.4.1
                    # p.162), written BASISSET / MINIX / HF3C / END (p.159).
                    # The parent's basis, external or internal, was kept and
                    # an external one came out as "BASISSET / EXTERNAL ...".
                    required_basis = {"HF3C": "MINIX", "HFSOL3C": "SOLMINIX"}.get(hf_method)
                    if required_basis:
                        if options.get("basis_set") != required_basis:
                            ui.info(f"  Basis set: {required_basis} (required by {hf_method}; "
                                    f"the parent's basis is not used)")
                        options["basis_set"] = required_basis
                        options["basis_set_type"] = "INTERNAL"
                        options["use_original_external_basis"] = False
                        options["is_3c_method"] = True
                if explicit_functional and "dispersion" not in config_data:
                    options["dispersion"] = str(explicit_functional).upper().endswith("-D3")
                    # A plan's functional switch (method_modifications
                    # new_functional) with no "use_dispersion" keeps the
                    # parent's D3 when the new functional has D3 parameters:
                    # a follow-up preserves what the user did not change (the
                    # GUI's workflow builder does the same). "use_dispersion":
                    # false drops it; a custom_functional is written as is.
                    new_fn = str(method_mods.get("new_functional") or "")
                    if (new_fn and new_fn == str(explicit_functional)
                            and "use_dispersion" not in method_mods
                            and not options["dispersion"] and settings.get("dispersion")
                            and new_fn.upper() in {f.upper() for f in D3_FUNCTIONALS}):
                        options["dispersion"] = True
                        ui.info(f"  D3 dispersion: kept from the parent ({new_fn}-D3; "
                                f"\"use_dispersion\": false drops it)")
                    elif new_fn and new_fn == str(explicit_functional) and method_mods.get("use_dispersion"):
                        options["dispersion"] = new_fn.upper() in {f.upper() for f in D3_FUNCTIONALS}

                # A functional switch keeps the parent's spin treatment unless
                # the config says otherwise (scf_modifications SPIN, applied
                # below, "keep_spin": false, its own "spin_polarized", or RHF /
                # UHF named outright). A UHF parent's spin lives in its method
                # name, not the spin flag, so a B3LYP step off it came out
                # closed shell, titled "SPIN off", and an odd-electron UHF
                # parent was refused.
                named_spin = str(method_mods.get("new_functional") or method_mods.get("functional")
                                 or config_data.get("hf_method") or config_data.get("functional")
                                 or "").upper()
                if (explicit_functional and named_spin not in ("RHF", "UHF")
                        and method_mods.get("keep_spin") is not False
                        and not (isinstance(config_data.get("scf_modifications"), dict)
                                 and "SPIN" in config_data["scf_modifications"])
                        and "spin_polarized" not in config_data
                        and "is_spin_polarized" not in config_data):
                    parent_spin = is_spin_polarized(settings)
                    if parent_spin != is_spin_polarized(options):
                        set_spin_polarized(options, parent_spin)
                        ui.info(f"  Spin polarization: kept from the parent "
                                f"({'spin-polarized' if parent_spin else 'closed shell'}; "
                                f"scf_modifications \"SPIN\" changes it)")

                # Handle tolerance_modifications if present. The planner's
                # FREQ steps nest their tolerances in the frequency settings
                # instead; honor those when no explicit override is given.
                # Either way the override is laid over the parent's values,
                # so a plan that sets only TOLDEE keeps the parent's TOLINTEG.
                custom_tol = None
                if "tolerance_modifications" in config_data:
                    custom_tol = (config_data["tolerance_modifications"] or {}).get("custom_tolerances")
                elif isinstance(options.get("freq_settings"), dict):
                    custom_tol = options["freq_settings"].get("custom_tolerances")
                # A FREQ deck starts from the FREQ default (Very tight), not
                # the optimization's tolerances, so a config naming only
                # TOLDEE still gets the Very tight TOLINTEG.
                is_freq = options.get("calculation_type") == "FREQ"
                base_tol = (freq_default_tolerances(settings.get("calculation_type"),
                                                    settings.get("tolerances"))
                            if is_freq else (settings.get("tolerances") or {}))
                if isinstance(custom_tol, dict) and custom_tol:
                    options["tolerances"] = {**base_tol, **custom_tol}
                    ui.info(f"  Tolerances updated: {custom_tol}")
                elif is_freq:
                    options["tolerances"] = {**base_tol, **(config_data.get("tolerances") or {})}
                    ui.info(f"  FREQ SCF tolerances: {options['tolerances']}")

                # A workflow step's SCF records over the parent's (MAXCYCLE,
                # FMIXING, the mixing method, LEVSHIFT, SMEAR, SPINLOCK,
                # TOLDEE): what the interactive "modify settings" questions set.
                if config_data.get("scf_modifications") is not None:
                    try:
                        scf_mods = check_scf_modifications(config_data["scf_modifications"])
                    except ValueError as exc:
                        _fail(str(exc), f"The config's {exc}; nothing written.")
                        return False, None
                    if scf_mods:
                        apply_scf_modifications(options, scf_mods)
                        # SPINLOCK is written only in a spin-polarized deck
                        # (CRYSTAL stops on one in a closed-shell run): one set
                        # on a closed-shell step was dropped while the log said
                        # it was set.
                        if scf_mods.get("SPINLOCK") and not is_spin_polarized(options):
                            _fail(f"SPINLOCK {scf_mods['SPINLOCK']} needs a spin-polarized deck, and this "
                                  f"step runs closed shell",
                                  f"The config's scf_modifications.SPINLOCK {scf_mods['SPINLOCK']} needs a "
                                  f"spin-polarized deck, and this step runs closed shell "
                                  f"(add \"SPIN\": true to turn spin polarization on); nothing written.")
                            return False, None
                        ui.info(f"  SCF settings changed: {describe_scf_modifications(scf_mods)}")

                # The config named no functional and the parent's was not
                # identified: warn and use HSE06 rather than ask.
                ensure_known_functional(options)

                ui.ok("Config file settings applied.")
            else:
                # Fall back to interactive mode
                options = get_calculation_options_from_current(settings)

        except Exception as e:
            ui.err(f"Error loading config file: {e}")
            if unattended:
                # A batch or --yes asks nothing, per file least of all.
                _fail(f"the config file could not be applied ({e})")
                return False, None
            ui.warn("Falling back to interactive mode.")
            try:
                options = get_calculation_options_from_current(settings)
            except EOFError:
                # Nobody to answer the interactive questions (answers piped
                # after the config, as the workflow engine does, are still
                # read above).
                _fail(f"the config file could not be applied ({e})",
                      "No answers on stdin for the interactive settings; nothing written.")
                return False, None
    elif non_interactive and not calc_type:
        # True non-interactive mode (no config file, no calc type specified)
        options = _keep_extracted_settings(settings, "SP", opt_type, origin_setting)
    elif non_interactive and calc_type:
        # --non-interactive with --calc-type still walks the interactive settings
        # flow: the workflow engine drives it by piping scripted answers to stdin,
        # and its output depends on those answers.
        #
        # Run from a shell with nothing on stdin - the documented form,
        # `mace opt2d12 --out-file X --calc-type SP --non-interactive` - the first
        # prompt hit end-of-file and the run died with EOFError. When the answers
        # run out, keep the extracted settings instead, exactly as --non-interactive
        # does without --calc-type. The engine always supplies enough answers, so
        # its output is unchanged.
        #
        # Nothing is ever waited for: a terminal is not asked, and a pipe left
        # open with no answers on it (a caller that never closes stdin) counts
        # as no answers, where it used to block on the first question forever.
        answers = piped_answers_or_none()
        try:
            if answers is None:
                raise EOFError
            real_stdin, sys.stdin = sys.stdin, answers
            try:
                options = get_calculation_options_from_current(
                    _with_opt_type_default(settings, opt_type), calc_type=calc_type)
            finally:
                sys.stdin = real_stdin
            apply_opt_type_flag(options, opt_type)
        except EOFError:
            ui.warn("\nNo answers on stdin for the interactive settings prompts; "
                    "keeping the settings extracted from the source calculation.")
            options = _keep_extracted_settings(settings, calc_type, opt_type, origin_setting)
    elif shared_settings:
        # Preserve external basis settings before shared_settings override
        had_external_basis = settings.get("use_original_external_basis", False)
        external_basis_type = settings.get("basis_set_type")

        # Merge shared settings with current settings
        options = settings.copy()
        # Settings chosen on a first file whose parent had an EXTERNAL basis
        # carry that parent's basis: every other file keeps its own, as with
        # a --config-file template.
        keep_parent_basis = template_uses_parent_basis(shared_settings)
        # Override with shared settings (except geometry-specific data)
        for key, value in shared_settings.items():
            if keep_parent_basis and key in TEMPLATE_BASIS_KEYS:
                continue
            if key in TEMPLATE_STRUCTURE_DATA_KEYS:
                continue
            if key not in [
                "coordinates",
                "primitive_cell",
                "conventional_cell",
                "spacegroup",
                "dimensionality",
                "origin_setting",
                "layer_group",
                "rod_group",
            ]:
                options[key] = value

        # Restore external basis settings if they were set and shared_settings didn't override them
        if had_external_basis and "basis_set_type" not in shared_settings and "basis_set_path" not in shared_settings:
            options["use_original_external_basis"] = True
            options["basis_set_type"] = external_basis_type

        # Ensure consistency for 3C methods
        if options.get("functional") in [
            "HF3C",
            "HFSOL3C",
            "PBEH3C",
            "HSE3C",
            "B973C",
            "PBESOL03C",
            "HSESOL3C",
        ]:
            options["dispersion"] = False
            options["is_3c_method"] = True
            options["dft_grid"] = None
        ensure_known_functional(options)
    else:
        # Interactive mode (possibly with pre-selected calc_type)
        answers_piped = not stdin_is_terminal()
        options = get_calculation_options_from_current(
            _with_opt_type_default(settings, opt_type), calc_type=calc_type)
        # An explicit --opt-type wins over the parent's type and the answers.
        apply_opt_type_flag(options, opt_type)
        
        # Ensure write_only_unique is set based on space group
        if "write_only_unique" not in options:
            if settings.get("spacegroup", 1) > 1:
                options["write_only_unique"] = True
            else:
                options["write_only_unique"] = False

    # ELASTIC / EOS / QHA / CPHF decks: the module's settings (--elastic-settings, the
    # config's "elastic_settings", the questions or CRYSTAL's defaults), the
    # manual's limits, and an SCF TOLDEE no looser than the module's 8
    if options.get("calculation_type") in ("ELASTIC", "EOS", "QHA", "CPHF"):
        from d12_calc_elastic import resolve_for_opt2d12
        problem = resolve_for_opt2d12(
            options, settings, config_file=config_file,
            interactive=not (non_interactive or unattended or config_file or shared_settings),
            log=ui.info)
        if problem:
            _fail(problem, f"\nNot writing the {options['calculation_type']} deck: {problem}")
            return False, None
    # A PHONOPY set: phonopy's settings (--dim and the flags with it, the
    # config's "phonopy_settings", the questions or the suggested supercell)
    # and an SCF TOLDEE no looser than 10
    if options.get("calculation_type") == "PHONOPY":
        from d12_calc_phonopy import resolve_for_opt2d12 as resolve_phonopy
        problem = resolve_phonopy(
            options, settings, config_file=config_file,
            interactive=not (non_interactive or unattended or config_file or shared_settings),
            log=ui.info)
        if problem:
            _fail(problem, f"\nNot writing the phonopy set: {problem}")
            return False, None

    # Keep the parent deck's SCF records that no prompt or config sets: the
    # BROYDEN parameters, LEVSHIFT, BIPOSIZE/EXCHSIZE, HISTDIIS and GUESSP. The interactive flow
    # replaces scf_settings with only maxcycle/fmixing/method, which dropped them.
    parent_scf = settings.get("scf_settings") or {}
    if isinstance(options.get("scf_settings"), dict) or "scf_settings" not in options:
        merged_scf = dict(options.get("scf_settings") or {})
        for scf_key in ("broyden", "levshift", "biposize", "exchsize", "histdiis", "guessp"):
            if scf_key in parent_scf:
                merged_scf.setdefault(scf_key, parent_scf[scf_key])
        options["scf_settings"] = merged_scf

    # Spin polarization turned on or off for this deck (a workflow step's
    # scf_modifications SPIN, or the spin question): the SCF starts from a
    # fresh guess, never the parent's density matrix. A two-component deck
    # treats spin itself and restarts by its own rules (GUESSPSO/GUESSPNOSO).
    # (Worked out for this parent only: never carried in from saved or shared settings.)
    options.pop("spin_change", None)
    spin_change = None if options.get("soc") else spin_state_change(settings, options)
    if spin_change:
        electrons = parent_electron_count(output_file)
        if spin_change == "off" and electrons is not None and electrons % 2:
            _fail(f"a closed-shell deck needs an even electron count, and the parent has {electrons}",
                  f"\nNot writing a closed-shell deck: the parent has {electrons} electrons per cell, an odd "
                  f"count only a spin-polarized run can hold (keep SPIN for this step)")
            return False, None
        options["spin_change"] = spin_change
        scf_now = options.get("scf_settings")
        dropped = isinstance(scf_now, dict) and bool(scf_now.pop("guessp", None))
        ui.info(f"  Spin polarization {spin_change} (the parent ran "
                f"{'closed shell' if spin_change == 'on' else 'spin-polarized'}): the SCF starts from a fresh "
                f"guess{' (the parent deck' + chr(39) + 's GUESSP is dropped)' if dropped else ''}, since "
                f"{SPIN_CHANGE_GUESS}")

    # Ensure symmetry settings are preserved from original input
    if "spacegroup" not in options and "spacegroup" in settings:
        options["spacegroup"] = settings["spacegroup"]
    if "origin_setting" not in options and "origin_setting" in settings:
        options["origin_setting"] = settings["origin_setting"]
    if "dimensionality" not in options and "dimensionality" in settings:
        options["dimensionality"] = settings["dimensionality"]
    
    # Always ensure write_only_unique is set based on space group
    # This applies to all modes (interactive, shared, non-interactive)
    if "write_only_unique" not in options:
        spacegroup = options.get("spacegroup", settings.get("spacegroup", 1))
        if spacegroup > 1:
            options["write_only_unique"] = True
        else:
            options["write_only_unique"] = False
    if parent_supercell:
        # Every atom of the supercell, in P1, whatever a config or the
        # shared settings of another file say.
        options.update(spacegroup=1, origin_setting="0 0 0", write_only_unique=False)

    # The deck carries the parent's external basis only on this branch of
    # the writer; a 3c functional always writes BASISSET <its basis>.
    if (options.get("functional") not in THREE_C_FUNCTIONALS
            and options.get("basis_set_type") == "EXTERNAL"
            and options.get("use_original_external_basis")
            and external_basis_data):
        ui.info("  Using external basis set from original D12 file")

    # Create output filename (a deck's own child: named after the deck, and
    # not "_optimized", which says the geometry came from an optimization)
    base_name = os.path.splitext(output_file or input_file)[0]
    calc_type = options["calculation_type"]
    functional = options.get("functional", "RHF")

    # Don't add -D3 to 3C methods or HF methods or if dispersion is already included
    # in the name. The "-D3 not in functional" guard is essential here: this is the
    # tool that emits the "_optimized" continuation filenames, and d12_parsers bakes
    # "-D3" into the functional parsed from real CRYSTAL "DFT-D3" output, so without
    # it a B3LYP-D3 source doubled to "B3LYP-D3-D3" (and tripled on each continuation).
    if (
        options.get("dispersion")
        and "-3C" not in functional
        and "3C" not in functional
        and "-D3" not in functional
        and functional not in ["RHF", "UHF", "HF3C", "HFSOL3C"]
    ):
        functional += "-D3"

    # Belt-and-suspenders against the long-standing filename nuisance: never emit
    # a doubled '-D3-D3' in the continuation filename, regardless of how the
    # functional was derived. Cosmetic only — the SCF content is unaffected.
    functional = dedupe_dispersion_suffix(functional)

    # A two-component SOC deck is named apart from the scalar deck of the
    # same parent ("_soc"), so the two never overwrite each other.
    soc_tag = "_soc" if options.get("soc") else ""
    # an ELASTIC deck is named by its module (_elastcon, _elapiezo, ...), so
    # the modules of one parent never overwrite each other
    type_token = (str((options.get("elastic_settings") or {}).get("mode") or "elastcon").lower()
                  if calc_type == "ELASTIC" else calc_type.lower())
    if calc_type == "PHONOPY":
        # a phonopy set is named by its supercell (_ph222): the folder of its
        # displaced decks (d12_calc_phonopy)
        from d12_calc_phonopy import dim_tag
        type_token = "ph" + dim_tag((options.get("phonopy_settings") or {}).get("dim") or "2 2 2")
    new_filename = (f"{base_name}_{type_token}_{functional}{soc_tag}"
                    f"{'_optimized' if output_file else ''}.d12")

    # SCELPHONO builds the phonon supercell in the geometry input (manual
    # sec. 4.21 p.73; the geometry CRYSTAL prints after it is the supercell,
    # primitive-cell atoms first, p.74), and DISPERSION runs on that cell
    # (sec. 8.8 p.232). A FREQ deck from such a run starts from the supercell
    # and, carrying the parent's SCELPHONO, would expand it again. An SP or
    # OPT deck starts from that supercell too: measured on HPCC (MgO,
    # SCELPHONO 2x2x2 + DISPERSION), the output parser returns the 16-atom
    # supercell (a = 5.95 A, 60 deg), and the SP deck written from it was
    # space group 225 with a = 5.95 A and 5 asymmetric atoms - not MgO.
    if parent_scelphono:
        # Any SCELPHONO run, a Gamma-only FREQ as much as a DISPERSION one.
        if calc_type == "FREQ":
            reason = ("the parent built a phonon supercell (SCELPHONO): its geometry is the "
                      "SCELPHONO supercell, which a FREQ deck with SCELPHONO would expand again")
        else:
            reason = ("the parent built a phonon supercell (SCELPHONO): its output geometry is "
                      "the SCELPHONO supercell (manual sec. 4.21 p.73-74), so a "
                      f"{calc_type} deck would start from the supercell")
        _fail(reason, f"\nNot writing {os.path.basename(new_filename)}: {reason}. Make the "
                      f"{calc_type} deck from the OPT or SP output the SCELPHONO run started from.")
        return False, options

    # CUSTOM-XC names the parent's own EXCHANGE/CORRELAT/HYBRID records; a
    # parent without them has nothing to write, and the writer would stop
    # half way through the deck. Refuse before any file is opened.
    if options.get("functional") == CUSTOM_FUNCTIONAL and not options.get("custom_functional"):
        _fail(f"functional '{CUSTOM_FUNCTIONAL}' but the parent has no EXCHANGE/CORRELAT/HYBRID records",
              f"\nNot writing {os.path.basename(new_filename)}: the functional "
              f"'{CUSTOM_FUNCTIONAL}' means the parent's own EXCHANGE/CORRELAT/HYBRID "
              f"definition, and this parent's DFT block has none. Name a CRYSTAL23 "
              f"functional (e.g. PBE0, HSE06) instead.")
        return False, options
    # --output-dir was parsed, and the directory created, but never reached this
    # point, so every deck landed in the current directory regardless.
    # The name is joined on by its base name: base_name above carries the
    # --out-file's directory, and joining an absolute one onto output_dir
    # yields that absolute path, so the deck landed next to the parent.
    if output_dir:
        new_filename = os.path.join(output_dir, os.path.basename(new_filename))

    # Two inputs of one run whose decks have the same name: the second would
    # overwrite the first, and both would be reported as written.
    if reserved_decks is not None:
        earlier = reserved_decks.get(os.path.realpath(new_filename))
        if earlier:
            _fail(f"its deck {os.path.basename(new_filename)} was already written from "
                  f"{earlier} in this run; not overwriting it")
            return False, options

    if calc_type == "PHONOPY":
        # phonopy's displaced supercells: a folder named as this deck would be,
        # holding one deck per displacement, each written by write_d12_file
        # with an EXTERNAL geometry (d12_calc_phonopy)
        import d12_calc_phonopy
        setdir = d12_calc_phonopy.write_set(
            new_filename, output_file, input_file, options,
            write_deck=lambda path, geometry, deck_settings, k: write_d12_file(
                path, geometry, deck_settings, external_basis_data, parent_k_points=k, ask=False),
            deck_number=lambda z: deck_atom_number(z, options),
            parent_k_points=parent_k_points, unattended=unattended,
            confirm=lambda folder: yes_no_prompt(f"{folder} is not empty. Write the set into it?", "no"),
            log=ui.info, warn=ui.warn, fail=_fail)
        if not setdir:
            if not LAST_RESULT["reason"]:
                LAST_RESULT["reason"] = "phonopy set not written"
            return False, options
        ui.ok(f"\nSuccessfully created the phonopy set {setdir}")
        LAST_RESULT["deck"] = setdir
        if reserved_decks is not None:
            reserved_decks[os.path.realpath(new_filename)] = output_file or input_file
        return True, options

    # Write new D12 file
    if os.path.exists(new_filename):
        ui.warn(f"\n{os.path.basename(new_filename)} already exists; it is replaced by "
                f"the new deck")
    ui.info(f"\nWriting new D12 file: {new_filename}")

    # Debug output for symmetry settings
    ui.info(f"Symmetry settings:")
    ui.info(f"  Space group: {options.get('spacegroup', 'Not set')}")
    ui.info(f"  Origin setting: {options.get('origin_setting', 'Not set')}")
    ui.info(f"  Dimensionality: {options.get('dimensionality', 'Not set')}")
    ui.info(f"  Write only unique atoms: {options.get('write_only_unique', 'Not set')}")
    
    # Convert frequency settings if present
    if "freq_settings" in options:
        converted_options = options.copy()
        freq_settings = options["freq_settings"].copy()
        
        # Pass space group and Bravais lattice info for high-symmetry point generation
        if "spacegroup" in out_data:
            freq_settings["space_group"] = out_data["spacegroup"]
        
        # Try to determine Bravais lattice from the geometry
        # This is a simplified approach - could be enhanced with proper symmetry analysis
        if "spacegroup" in out_data and out_data["spacegroup"] is not None:
            sg = out_data["spacegroup"]
            # Determine Bravais lattice based on space group ranges
            # This is a simplified mapping - ideally would extract from symmetry operations
            if sg <= 2:
                bravais = "P"  # Triclinic
            elif sg <= 15:
                bravais = "P" if sg <= 9 else "C"  # Monoclinic
            elif sg <= 74:
                # Orthorhombic - needs more detailed analysis
                if sg in [20, 21, 35, 36, 37, 38, 39, 40, 41, 63, 64, 65, 66, 67, 68]:
                    bravais = "C"
                elif sg in [22, 42, 43, 69, 70]:
                    bravais = "F"
                elif sg in [23, 24, 44, 45, 46, 71, 72, 73, 74]:
                    bravais = "I"
                else:
                    bravais = "P"
            elif sg <= 142:
                # Tetragonal
                bravais = "I" if sg >= 79 else "P"
            elif sg <= 167:
                # Trigonal/Rhombohedral
                bravais = "R" if sg in [146, 148, 155, 160, 161, 166, 167] else "P"
            elif sg <= 194:
                # Hexagonal
                bravais = "P"
            else:
                # Cubic
                if sg in [196, 202, 203, 209, 210, 216, 219, 225, 226, 227, 228]:
                    bravais = "F"
                elif sg in [197, 199, 204, 206, 211, 214, 217, 220, 229, 230]:
                    bravais = "I"
                else:
                    bravais = "P"
            
            freq_settings["bravais_lattice"] = bravais
        
        # Handle phonon bands conversion
        if freq_settings.get("phonon_bands", False):
            # Remove the phonon_bands flag and replace with proper bands dict
            freq_settings.pop("phonon_bands", None)
            
            # Create bands dictionary
            bands_dict = {
                "shrink": freq_settings.get("shrink", 16),
                "npoints": freq_settings.get("n_points_per_segment", 100),
                "path": "AUTO" if freq_settings.get("auto_kpath", False) else []
            }
            
            # Handle custom path if provided
            if "band_path" in freq_settings:
                bands_dict["path"] = freq_settings["band_path"]
            
            freq_settings["bands"] = bands_dict
            
            # Mark as dispersion calculation
            freq_settings["dispersion"] = True
        
        # Handle phonon DOS conversion
        if freq_settings.get("phonon_dos", False):
            freq_settings.pop("phonon_dos", None)
            
            # Get DOS settings if provided
            dos_settings = freq_settings.get("dos_settings", {})
            freq_settings["pdos"] = {
                "max_freq": dos_settings.get("max_freq", 2000),
                "nbins": dos_settings.get("n_bins", 200),
                "projected": dos_settings.get("projected", True)
            }
            
            # Mark as dispersion calculation
            freq_settings["dispersion"] = True
        
        # Handle INS conversion
        if freq_settings.get("calculate_ins", False):
            freq_settings.pop("calculate_ins", None)
            
            # Get INS settings if provided
            ins_settings = freq_settings.get("ins_settings", {})
            freq_settings["ins"] = {
                "max_freq": ins_settings.get("max_freq", 3000),
                "nbins": ins_settings.get("n_bins", 300),
                "neutron_type": ins_settings.get("neutron_type", 2)
            }
            
            # INS requires dispersion
            freq_settings["dispersion"] = True
        
        converted_options["freq_settings"] = freq_settings
    else:
        converted_options = options
    
    # Use optimized geometry from output but with preserved settings
    # The geometry_data (out_data) contains the optimized coordinates with is_unique flags
    # The settings (options) contains the preserved symmetry and other settings from D12
    # A 2c restart ("soc_guess"): checked before the deck is written, and the
    # parent's .f9 staged as <deck stem>.f20 once it is.
    restart_f9 = restart_f20 = restart_parent = None
    if converted_options.get("soc_guess") is not None:
        problem, restart_f9, restart_f20, restart_parent = soc_restart_plan(
            converted_options, output_file, input_file, new_filename)
        if problem:
            _fail(problem, f"\nNot writing {os.path.basename(new_filename)}: {problem}")
            return False, options
    parent_metal = soc_parent_metal(output_file) if converted_options.get("soc") else None
    # A RESTART/RESTART2 elastic or EOS deck reads the parent run's
    # ELASINFO.DAT / EOSINFO.DAT, which the job script stages as
    # <JOB>.<file> next to the deck; the new deck has a new job name.
    info_src = info_dst = None
    if converted_options.get("calculation_type") in ("ELASTIC", "EOS", "QHA"):
        from d12_calc_elastic import restart_info_plan
        problem, info_src, info_dst = restart_info_plan(
            converted_options.get("elastic_settings"), output_file, new_filename)
        if problem:
            _fail(problem, f"\nNot writing {os.path.basename(new_filename)}: {problem}")
            return False, options
    # The new deck replaces an older one of the same name as it is written,
    # and some refusals come only after that (the charge-record checks read
    # the written deck). A copy of the older deck is put back on a refusal.
    old_deck = _set_aside_old_deck(new_filename)
    try:
        written = write_d12_file(new_filename, out_data, converted_options, external_basis_data,
                                 parent_k_points=parent_k_points,
                                 ask=False if unattended else None,
                                 parent_metal=parent_metal,
                                 soc_restart_parent=restart_parent)
        if written and restart_f9:
            import shutil
            try:
                if not os.path.exists(restart_f20):
                    shutil.copyfile(restart_f9, restart_f20)
            except OSError as exc:
                _drop_new_deck(new_filename, old_deck)
                _fail(f"could not stage {os.path.basename(restart_f9)} as "
                      f"{os.path.basename(restart_f20)} ({exc}); "
                      f"{missing_2c_matrix_note(converted_options['soc_guess'])}")
                return False, options
            ui.info(f"  {converted_options['soc_guess'].upper()}: staged "
                    f"{os.path.basename(restart_f9)} as {os.path.basename(restart_f20)} "
                    f"(the job script reads it as fort.20)")
        if written and info_src:
            import shutil
            try:
                shutil.copyfile(info_src, info_dst)
            except OSError as exc:
                _drop_new_deck(new_filename, old_deck)
                _fail(f"could not stage {info_src} as {os.path.basename(info_dst)} ({exc})")
                return False, options
            ui.info(f"  RESTART: staged {info_src} as {os.path.basename(info_dst)} "
                    f"(the job script puts it in the run's scratch folder)")
    except BasisFileError as exc:
        # A basis record CRYSTAL could not read (atomic_deck has removed the
        # part-written deck).
        _fail(str(exc), f"Not writing {os.path.basename(new_filename)}: {exc}")
        written = False
    if not written:
        # atomic_deck left any older deck as it was
        _drop_set_aside(old_deck)
        ui.err(f"\nFailed to create {new_filename}: D12 creation aborted.")
        if not LAST_RESULT["reason"]:
            LAST_RESULT["reason"] = "D12 creation aborted"
        return False, options

    # A charged or ionic parent (CHEMOD / CHARGED / DOPING) must give a child
    # with the same electron count; the settings model does not carry these
    # records, so copy them from the parent deck text - whenever the parent
    # deck exists, even if parsing it failed above. Without a parent deck the
    # .out's Mulliken electron count gives the charge.
    parent_deck_exists = bool(input_file and os.path.exists(input_file))
    try:
        with open(output_file, "r", errors="ignore") as of:
            parent_out_text = of.read()
    except (OSError, TypeError):            # no .out: a deck is its own parent
        parent_out_text = None
    with open(new_filename, "r") as cf:
        child_text = cf.read()
    parent_text = None
    soft_notes = []
    charge_notes = []
    carried = child_text
    requested = {k: v for k, v in (charge_options or {}).items() if v is not None}
    if config_file:
        # the same keys in the config (applied to options above) count too;
        # the command-line flags win
        requested = {**{k: options.get(k) for k in CHARGE_OPTION_KEYS
                        if (options or {}).get(k) is not None}, **requested}
    recompute = any(requested.get(k) not in (None, "", "none")
                    for k in ("ion_charges", "net_charge", "site_states"))
    # Without a charge change, a spin state given on the command line is
    # applied on its own (as cif2d12 does; a config's "spinlock" is an SCF
    # setting the writer already wrote); CHEMOD and the fragment bond scale
    # have nothing to act on, so asking for them is refused rather than
    # silently dropped.
    flags = {k: v for k, v in (charge_options or {}).items() if v is not None}
    spin_alone = not recompute and any(flags.get(k) is not None
                                       for k in ("spinlock", "spinlock_cycles"))
    idle = []
    if not recompute:
        if requested.get("chemod") not in (None, "", "auto"):
            idle.append(f"chemod {requested['chemod']} (--chemod)")
        fbs = requested.get("fragment_bond_scale")
        if fbs not in (None, "") and abs(float(fbs) - 1.15) > 1e-9:
            idle.append(f"fragment_bond_scale {fbs} (--fragment-bond-scale)")
    if idle:
        _drop_new_deck(new_filename, old_deck)
        reason = (", ".join(idle) + " only take effect with a charge change: give "
                  "--ion-charges auto, --net-charge Q or --site-states (ion_charges, "
                  "net_charge, site_states in a config)")
        _fail(f"charge state: {reason}", f"Not writing {new_filename}: {reason}")
        return False, options
    try:
        if parent_deck_exists:
            with open(input_file, "r", errors="ignore") as pf:
                parent_text = pf.read()
        if recompute and "TWOCOMPON" in (ln.strip().upper() for ln in child_text.splitlines()):
            raise ValueError(
                "the charged-system options (ion_charges, net_charge, site_states) are not "
                "written into a SOC deck: DOPING, CHARGED and SPINLOCK are not in the "
                "two-component chapter of the CRYSTAL23 manual (ch. 6, p. 166), and opt2d12 "
                "does not read a SOC deck's INPSOC records to place CHEMOD; write the SOC "
                "deck without them")
        if recompute:
            # asked for: the charge is worked out again from the new geometry
            # and replaces whatever the parent carried
            carried, charge_notes = recompute_ion_charges(
                child_text, parent_text, requested,
                interactive=not unattended and stdin_is_terminal())
        elif parent_deck_exists:
            carried, charge_notes = carry_charge_records(child_text, parent_text, parent_out_text)
        elif parent_out_text:
            carried, soft_notes = charge_from_parent_out(child_text, parent_out_text)
        if spin_alone:
            import ion_charges as ic
            try:
                carried, spin_notes = ic.set_spinlock(carried, flags.get("spinlock"),
                                                      flags.get("spinlock_cycles"),
                                                      parent_out_text)
            except ic.IonChargeError as e:
                raise ValueError(str(e))
            charge_notes = list(charge_notes) + spin_notes
    except (ChargeCarryError, ValueError) as e:
        _drop_new_deck(new_filename, old_deck)
        _fail(f"charge state: {e}", f"Not writing {new_filename}: {e}")
        return False, options
    # a SOC deck: the charge records go in after soc_deck wrote it, so they are
    # checked here as soc_deck checks a deck's own
    soc_refusal = (soc_ecp.charge_records_refusal(carried, (options or {}).get("soc_basis"))
                   if carried != child_text else None)
    if soc_refusal:
        _drop_new_deck(new_filename, old_deck)
        _fail(f"charge state: {soc_refusal}", f"Not writing {new_filename}: {soc_refusal}")
        return False, options
    if carried != child_text:
        with atomic_deck(new_filename) as cf:
            cf.write(carried)
    for note in soft_notes:
        if note.startswith("WARNING"):
            ui.warn(note)
        else:
            ui.info(f"Charge state: {note}")
    for note in charge_notes:
        if note.startswith("WARNING"):
            ui.warn(note)
            LAST_RESULT["notes"].append(note)
        else:
            ui.info(f"Charge state: {note}")

    _drop_set_aside(old_deck)
    ui.ok(f"\nSuccessfully created {new_filename}")
    if answers_piped and parent_text:
        # Answers read from a pipe: an answer file written for another MACE
        # version lands answers on other questions, and a valid answer on the
        # wrong question is taken without complaint. Show what moved away
        # from the parent so it cannot happen silently.
        with open(new_filename, "r") as cf:
            changed = settings_changed_from_parent(parent_text, cf.read())
        if changed:
            ui.warn("Answers were read from a pipe, and these settings differ from the "
                    "parent deck. Check they are the ones you meant: an answer file "
                    "written for an older MACE puts answers on other questions "
                    "(see Upgrading in CHANGELOG.md; --save-options and --config-file "
                    "do not drift).")
            for name, before, after in changed:
                ui.warn(f"  {name}: {before} -> {after}")
    LAST_RESULT["deck"] = new_filename
    if reserved_decks is not None:
        reserved_decks[os.path.realpath(new_filename)] = output_file or input_file
    if opt_problem:
        LAST_RESULT["notes"].append(opt_problem)

    return True, options


def _set_aside_old_deck(path):
    """A copy of the deck at ``path`` (None when there is none), beside it.

    Hidden (``.<name>.previous``) and without the .d12 ending, so nothing
    takes it for a deck. A symlinked deck is copied from the file it names.
    """
    import shutil
    if not os.path.isfile(path):
        return None
    real = os.path.realpath(path)
    directory, name = os.path.split(real)
    copy = os.path.join(directory, f".{name}.previous")
    try:
        shutil.copy2(real, copy)
    except OSError as exc:
        ui.warn(f"  Warning: could not keep a copy of the older {os.path.basename(path)} "
                f"({exc}); a refusal would lose it")
        return None
    return copy


def _drop_set_aside(copy):
    """Remove the copy _set_aside_old_deck made (the new deck stands)."""
    if copy:
        try:
            os.remove(copy)
        except OSError:
            pass


def _drop_new_deck(path, copy):
    """Remove a refused deck, putting back the older deck it replaced."""
    try:
        if copy:
            os.replace(copy, os.path.realpath(path))
            ui.info(f"  {os.path.basename(path)}: the older deck is kept")
        else:
            os.remove(path)
    except OSError:
        pass


CHARGE_OPTION_KEYS = ("ion_charges", "net_charge", "site_states", "chemod", "fragment_bond_scale",
                      "spinlock", "spinlock_cycles")


def recompute_ion_charges(child_text, parent_text, requested, interactive=False):
    """The child deck with its charge records worked out again from its own geometry.

    The carried records are replaced (ion_charges.plan_for_deck on the new
    deck). When the parent deck's fragments differ in composition from the
    child's, a warning says so (a proton or an anion moved between molecules
    during the optimization changes which ions there are).
    """
    import ion_charges as ic

    stripped = ic.strip_charge_records(child_text)
    parent_spin = ic.deck_method(parent_text).spinlock if parent_text else None
    # the net charge the carried SPINLOCK was set for: the parent's
    reference_q = None
    if parent_text:
        try:
            from charge_records import parent_electron_offset
            reference_q = -parent_electron_offset(parent_text)
        except Exception:
            reference_q = None
    try:
        setup = ic.plan_for_deck(
            stripped, ion_charges=requested.get("ion_charges"), net_charge=requested.get("net_charge"),
            site_states=ic.parse_site_states(requested.get("site_states")),
            chemod=requested.get("chemod") or "auto",
            fragment_bond_scale=requested.get("fragment_bond_scale") or 1.15,
            spinlock=requested.get("spinlock") if "spinlock" in requested else None,
            spinlock_cycles=requested.get("spinlock_cycles"), parent_spinlock=parent_spin,
            spinlock_reference_charge=reference_q,
            interactive=interactive)
    except ic.IonChargeError as e:
        raise ValueError(str(e))
    notes = [f"recomputed from the new geometry: {line}" for line in setup.report] + list(setup.warnings)
    if parent_text and requested.get("ion_charges") == "auto":
        try:
            def kinds(text):
                g = ic.read_deck_geometry(text)
                frags = ic.frag.classify(g.symbols, g.xyz, g.cell, g.periodic_dirs,
                                         requested.get("fragment_bond_scale") or 1.15, auto=False)
                return sorted((f["formula"], f["ion"] or "") for f in frags)
            before, after = kinds(parent_text), kinds(stripped)
            if before != after:
                notes.append("WARNING: the fragments changed during the optimization (parent: "
                             + ", ".join(f for f, _ in before) + "; now: "
                             + ", ".join(f for f, _ in after) + ") - check the charges")
        except Exception:
            pass
    return ic.apply_charge_setup(stripped, setup), notes


def find_file_pairs(directory):
    """Find matching .out and .d12 file pairs in a directory

    Returns:
        list: List of tuples (out_file, d12_file or None)
    """
    pairs = []

    # Find all .out files, excluding SLURM output files
    out_files = sorted(f for f in os.listdir(directory)
                       if f.endswith(".out") and not f.startswith("slurm-"))

    for out_file in out_files:
        base_name = out_file[:-4]  # Remove .out extension
        d12_file = f"{base_name}.d12"

        full_out_path = os.path.join(directory, out_file)
        full_d12_path = (
            os.path.join(directory, d12_file)
            if os.path.exists(os.path.join(directory, d12_file))
            else None
        )

        pairs.append((full_out_path, full_d12_path))

    return pairs


def pair_out_files(out_files):
    """(out, d12 or None) for each --out-file path, in the order given.

    Each .out is paired with the .d12 of the same name beside it; without
    one it is converted from the .out alone, as a single --out-file is.
    """
    pairs = []
    for out_file in out_files:
        d12_file = os.path.splitext(out_file)[0] + ".d12"
        pairs.append((out_file, d12_file if os.path.exists(d12_file) else None))
    return pairs


def same_deck_name_inputs(file_pairs, output_dir):
    """{.out: [the other .out files]} for inputs whose decks would be written
    under the same name in the same directory.

    A deck is named <.out name>_<calc>_<functional>_optimized.d12 and written
    in --output-dir or beside its .out, so two .out files of the same name
    (dupA/X.out, dupB/X.out) with one --output-dir collide. Neither is written:
    which structure the deck held would depend on the order of the files.
    """
    groups = {}
    for out_file, _ in file_pairs:
        target = output_dir or os.path.dirname(out_file) or "."
        stem = os.path.splitext(os.path.basename(out_file))[0]
        groups.setdefault((os.path.realpath(target), stem), []).append(out_file)
    return {out_file: [other for other in group if other != out_file]
            for group in groups.values() if len(group) > 1 for out_file in group}


def _template_basis_text(config_data):
    if template_uses_parent_basis(config_data):
        return "each structure keeps its own parent's basis"
    if config_data.get("basis_set_type") == "EXTERNAL" and config_data.get("basis_set_path"):
        return f"external basis files in {config_data['basis_set_path']}"
    return config_data.get("basis_set") or "each parent's own (the config names none)"


def print_template_plan(config_file, config_data, file_pairs, output_dir):
    """What a batch run of a --config-file will do, shown before any file."""
    mods = config_data.get("method_modifications") or {}
    functional = (mods.get("new_functional") or mods.get("functional")
                  or config_data.get("functional") or "each parent's own")
    tol_mods = (config_data.get("tolerance_modifications") or {}).get("custom_tolerances")
    tolerances = tol_mods or config_data.get("tolerances")
    k_mods = config_data.get("kpoint_modifications")
    k_mesh = k_mesh_text(k_mods.get("new_k_points")) if isinstance(k_mods, dict) else None
    if k_mods is not None:
        k_text = (f"{describe_k_mesh(k_mesh)} for every file (the config's kpoint_modifications)" if k_mesh
                  else f"invalid kpoint_modifications {k_mods!r}")
    elif ALL_FILES_K_POINTS_KEY in config_data:
        mesh = all_files_k_points(config_data[ALL_FILES_K_POINTS_KEY])
        k_text = (f"{describe_k_mesh(mesh)} for every file (the config's {ALL_FILES_K_POINTS_KEY})" if mesh
                  else f"invalid {ALL_FILES_K_POINTS_KEY} {config_data[ALL_FILES_K_POINTS_KEY]!r}")
    elif config_data.get("k_points"):
        k_text = (f"each parent's own mesh (the config's k_points {config_data['k_points']} "
                  f"is the mesh of the structure it was saved from, and is not used)")
    else:
        k_text = "each parent's own mesh"
    with_d12 = sum(1 for _, d12 in file_pairs if d12)

    print()
    ui.rule("APPLY CONFIG TO ALL FILES")
    ui.info(f"Config file:  {config_file}")
    ui.info(f"Files:        {len(file_pairs)} ({with_d12} with their .d12, "
            f"{len(file_pairs) - with_d12} from the .out alone)")
    ui.info(f"Decks go to:  {output_dir if output_dir else 'next to each .out file'}")
    ui.info(f"Calculation:  {config_data.get('calculation_type') or 'as each parent'}")
    ui.info(f"Functional:   {functional}")
    ui.info(f"Basis set:    {_template_basis_text(config_data)}")
    if config_data.get("dft_grid"):
        ui.info(f"DFT grid:     {config_data['dft_grid']}")
    if tolerances:
        ui.info(f"Tolerances:   {', '.join(f'{k} {v}' for k, v in tolerances.items())}")
    if "spin_polarized" in config_data:
        ui.info(f"Spin:         {'spin-polarized' if config_data['spin_polarized'] else 'closed shell'}")
    ui.info(f"k-points:     {k_text}")
    ui.info("Kept from each structure: its optimized geometry, atoms and symmetry.")
    ui.rule()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""\
Convert CRYSTAL17/23 optimization output to new D12 input files.

The output is given with --out-file (or every file in --directory). There is
no positional argument. The calculation type for --non-interactive comes
from --calc-type {SP,OPT,FREQ,ELASTIC,EOS,QHA,CPHF,PHONOPY} (the menu's
choices 1-8; 4-8 start from a finished OPT or SP run).

Mechanical-property and response modules (CRYSTAL23 manual ch. 9-10, 12-13),
their sub-keywords in --elastic-settings (keys in lower case):
  ELASTIC  mode ELASTCON (elastic constants, p. 290), ELAPIEZO (elastic +
           piezoelectric, p. 298), PIEZOCON (Berry phase, p. 296), PIEZOCP
           (CPHF/KS, p. 298) or PHOTOELA (photoelastic, p. 301)
  EOS      E(V) fitted to four equations of state (p. 284; 3D only)
  QHA      quasi-harmonic V(T), K(T), thermal expansion (p. 253; 1D-3D, p. 14)
  CPHF     polarizability, dielectric tensor, hyperpolarizabilities (p. 270)
Results: mace analyze elastic RUN.out.

Phonons: FREQ writes CRYSTAL's own FREQCALC deck. PHONOPY (choice 8) writes
phonopy's displaced supercells, a folder of EXTERNAL decks with the parent's
settings (--dim, --amplitude, --pm, --cry-sym, --keep-shrink); after the jobs,
mace analyze phonopy DIR writes the force constants, bands, DOS and thermal
properties (it runs by itself when the last job finishes).""",
        epilog="""\
Examples:
  %(prog)s --out-file optimized.out
  %(prog)s --out-file optimized.out --calc-type SP --non-interactive
  %(prog)s --directory . --calc-type FREQ --non-interactive
  %(prog)s --directory opts --config-file sp_template.json --yes
  %(prog)s --out-file opts/*.out --config-file sp_template.json --output-dir sp
  %(prog)s --out-file mat_opt.out --calc-type ELASTIC --non-interactive \\
      --elastic-settings '{"mode": "ELAPIEZO", "numderiv": 5, "print": true}'
  %(prog)s --out-file mat_opt.out --calc-type EOS --non-interactive \\
      --elastic-settings '{"range": "0.92 1.08 8", "prange": "-5 10 16"}'
  %(prog)s --out-file mat_opt.out --calc-type PHONOPY --non-interactive --dim "2 2 2"
  %(prog)s --d12-file deck.d12 --calc-type OPT    # a deck as the parent""",
    )
    parser.add_argument(
        "--out-file", nargs="+", metavar="OUT",
        help="CRYSTAL output file(s) (.out). Several, or a shell glob such as "
             "'opts/*.out', are processed as one batch, each with the .d12 of "
             "the same name beside it when there is one",
    )
    parser.add_argument(
        "--d12-file", type=str,
        help="Original CRYSTAL input file (.d12) for a single --out-file "
             "(default: the .d12 of the same name beside it, if there is one). "
             "Without --out-file the deck is the parent: the new deck has its "
             "geometry and its settings (not for EXTERNAL decks or decks that "
             "edit their geometry, e.g. SUPERCEL: use their run's .out)",
    )
    parser.add_argument(
        "--directory",
        type=str,
        default=".",
        help="Directory containing files (default: current directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for new D12 files (default: same as input)",
    )
    parser.add_argument(
        "--shared-settings",
        action="store_true",
        help="Apply the same calculation settings to all files",
    )
    parser.add_argument(
        "--save-options", action="store_true", help="Save options to file"
    )
    parser.add_argument(
        "--options-file",
        type=str,
        default="crystal_opt_settings.json",
        help="File to save/load options",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="JSON config file (e.g. one saved with --save-options) whose settings are "
             "applied to every file; each structure keeps its own geometry, symmetry "
             "and, for a config saved from an external-basis parent, its own basis",
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Apply --config-file without asking. Otherwise one file asks "
             "'Apply these settings?' and a batch asks once for all files, "
             "both only when run at a terminal",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run in non-interactive mode (skip all prompts, use defaults or provided options)",
    )
    parser.add_argument(
        "--calc-type",
        choices=["SP", "OPT", "FREQ", "ELASTIC", "EOS", "QHA", "CPHF", "PHONOPY"],
        help="Calculation type for non-interactive mode (ELASTIC: ELASTCON, ELAPIEZO, "
             "PIEZOCON, PIEZOCP or PHOTOELA; EOS: equation of state; QHA: quasi-harmonic; "
             "CPHF: dielectric response; PHONOPY: phonopy's displaced supercells)",
    )
    parser.add_argument(
        "--elastic-settings",
        metavar="JSON",
        help="ELASTIC/EOS/QHA/CPHF block settings: a JSON object or a JSON file, e.g. "
             "'{\"mode\": \"ELASTCON\", \"numderiv\": 5, \"nuchess\": true}' "
             "(keys: the CRYSTAL sub-keywords in lower case: mode, numderiv, stepsize, deform, "
             "clampion, nuchess, hessnum2, modepart, hessstep, preoptgeom, pressure, presseos, "
             "seismdir, awesome, tolalpha, dynamic, range, vrange, prange, qstep, points, temperat, "
             "scelphono, neglefre, third, fourth, cphf_dynamic, damping, restart, print)",
    )
    parser.add_argument(
        "--opt-type",
        choices=["FULLOPTG", "ATOMONLY", "CELLONLY", "ITATOCEL", "CVOLOPT", "INTREDUN"],
        help="Optimization type for OPT calculations in non-interactive mode",
    )
    parser.add_argument(
        "--static-kpath",
        action="store_true",
        help="Write the static HPKOT table path for a 3D SeeK-path phonon dispersion "
             "when seekpath is not installed (SeeK-path's standard cell, which can "
             "differ from CRYSTAL's); without it such a deck stops. Same as "
             "MACE_STATIC_KPATH=1",
    )
    parser.add_argument(
        "--origin-setting",
        default="auto",
        help="Origin setting: 'auto', '0 0 1', '0 1 0', etc. (default: auto-detect)",
    )

    charged = parser.add_argument_group(
        "charged systems (recomputed from the new geometry; see Crystal_d12/README.md)")
    charged.add_argument("--ion-charges", choices=["auto"],
                         help="find known ions (Li+, FSI-, ...) in the new geometry and set "
                              "CHEMOD/DOPING for them, replacing the parent's records")
    charged.add_argument("--net-charge", type=float, help="net charge (default: the ions found)")
    charged.add_argument("--site-states",
                         help='JSON or JSON file: [{"atom": 1, "shell_charges": [...]}]')
    charged.add_argument("--chemod", choices=["auto", "yes", "no"], help="write CHEMOD (default auto)")
    charged.add_argument("--fragment-bond-scale", type=float,
                         help="bond when distance < scale x covalent radii sum (default 1.15)")
    charged.add_argument("--spinlock", type=int, help="nalpha-nbeta for an odd electron count")
    charged.add_argument("--spinlock-cycles", type=int, help="SCF cycles the SPINLOCK holds")

    ph = parser.add_argument_group(
        "phonopy (--calc-type PHONOPY: displaced supercells; needs phonopy, pip install 'phonopy<5')")
    ph.add_argument("--dim", nargs="+", metavar="N",
                    help='supercell: "N N N" or 9 integers, row-major (default: each lattice vector '
                         'at least 10 A)')
    ph.add_argument("--amplitude", type=float, help="displacement in Angstrom (default: phonopy's 0.01)")
    ph.add_argument("--pm", choices=["auto", "yes", "no"], help="plus/minus displacements (default: auto)")
    ph.add_argument("--cry-sym", action="store_true",
                    help="write each displaced cell's symmetry into its .ext (phonopy's CRY_SYM): fewer "
                         "k-points, but phonopy warns some space groups (e.g. R-3m) do not work")
    ph.add_argument("--keep-shrink", action="store_true",
                    help="keep the parent's SHRINK (default: divide it by the supercell size, rounding "
                         "up; diagonal supercells only)")
    ph.add_argument("--force", action="store_true", help="write the set into a non-empty folder")

    args = parser.parse_args()
    phonopy_flags = {k: v for k, v in (("dim", " ".join(args.dim) if args.dim else None),
                                       ("amplitude", args.amplitude), ("pm", args.pm),
                                       ("cry_sym", args.cry_sym or None),
                                       ("keep_shrink", args.keep_shrink or None),
                                       ("force", args.force or None)) if v is not None}
    if phonopy_flags:
        if args.calc_type != "PHONOPY":
            parser.error(f"--{next(iter(phonopy_flags)).replace('_', '-')} goes with --calc-type PHONOPY")
        import d12_calc_phonopy
        try:
            d12_calc_phonopy.normalize(phonopy_flags)
        except ValueError as e:
            parser.error(str(e))
        d12_calc_phonopy.REQUESTED = phonopy_flags
    if getattr(args, "elastic_settings", None):
        import d12_calc_elastic
        try:
            d12_calc_elastic.REQUESTED = d12_calc_elastic.load_requested(args.elastic_settings)
        except (OSError, ValueError) as e:
            parser.error(f"--elastic-settings: {e}")
    if args.static_kpath:
        from d3_kpoints import request_static_kpath
        request_static_kpath()
    charge_options = {
        "ion_charges": args.ion_charges,
        "net_charge": (int(args.net_charge) if args.net_charge is not None
                       and float(args.net_charge).is_integer() else args.net_charge),
        "site_states": args.site_states, "chemod": args.chemod,
        "fragment_bond_scale": args.fragment_bond_scale,
        "spinlock": args.spinlock, "spinlock_cycles": args.spinlock_cycles,
    }
    repeated = []
    if args.out_file:
        # A path given twice (or as a.out and ./a.out) is converted once: a
        # second run over it wrote the same deck again and counted it twice.
        unique, seen = [], set()
        for path in args.out_file:
            key = os.path.realpath(path)
            if key in seen:
                repeated.append(path)
                continue
            seen.add(key)
            unique.append(path)
        args.out_file = unique
    if args.out_file and len(args.out_file) > 1 and args.d12_file:
        parser.error("--d12-file goes with a single --out-file; with several, each "
                     ".out uses the .d12 of the same name beside it")

    ui.rule("CRYSTAL17/23 Optimization Output to D12 Converter")
    ui.info("Enhanced version matching NewCifToD12.py configurations")
    ui.info("New entirely reworked script by Marcus Djokic")
    ui.info(
        "Based on old versions by Wangwei Lan, Kevin Lucht, Danny Maldonado, Marcus Djokic"
    )
    print("")
    for path in repeated:
        ui.warn(f"{path} is listed more than once; converting it once")

    # Create output directory if specified
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.d12_file and not args.out_file:
        # A deck as its own parent: its geometry records and its settings
        if not os.path.isfile(args.d12_file):
            ui.err(f"Error: Deck {args.d12_file} not found")
            sys.exit(1)
        success, options = process_files(
            None,
            args.d12_file,
            config_file=args.config_file,
            non_interactive=args.non_interactive,
            calc_type=args.calc_type,
            opt_type=args.opt_type,
            origin_setting=args.origin_setting,
            output_dir=args.output_dir,
            confirm_config=False if args.yes else None,
            unattended=bool(args.yes or args.non_interactive),
            charge_options=charge_options,
        )
        if success and args.save_options:
            with open(args.options_file, "w") as f:
                json.dump(options_for_template(options), f, indent=2)
            ui.ok(f"Settings saved to {args.options_file}")
        if not success:
            sys.exit(1)

    # Single file processing
    elif args.out_file and len(args.out_file) == 1:
        out_file = args.out_file[0]
        if not os.path.exists(out_file):
            ui.err(f"Error: Output file {out_file} not found")
            sys.exit(1)

        d12_file = args.d12_file
        if d12_file is None:
            # A lone --out-file (e.g. a glob that matched one file) is paired
            # with the .d12 of the same name beside it, as each file of a batch
            # is. Without it the parent deck's settings were silently replaced
            # by what the .out shows: an external ECP basis became
            # POB-TZVP-REV2, with or without a --config-file. Every workflow
            # caller passes --d12-file whenever there is a deck.
            d12_file = pair_out_files([out_file])[0][1]
            if d12_file:
                ui.info(f"With input: {d12_file} (the .d12 beside {os.path.basename(out_file)})")
            else:
                ui.info(f"No .d12 beside {os.path.basename(out_file)}: converting from the .out alone")

        success, options = process_files(
            out_file,
            d12_file,
            config_file=args.config_file,
            non_interactive=args.non_interactive,
            calc_type=args.calc_type,
            opt_type=args.opt_type,
            origin_setting=args.origin_setting,
            output_dir=args.output_dir,
            confirm_config=False if args.yes else None,
            unattended=bool(args.yes or args.non_interactive),
            charge_options=charge_options,
        )

        if success and SOC_METAL_NOTE in LAST_RESULT["notes"]:
            print()
            print_soc_metal_summary([out_file])

        if success and args.save_options:
            with open(args.options_file, "w") as f:
                # The settings only: no structure's atoms or basis records.
                json.dump(options_for_template(options), f, indent=2)
            ui.ok(f"Settings saved to {args.options_file}")

        if not success:
            # No deck was written: say so to the caller (the workflow engine
            # checks the exit status), not only on the terminal.
            sys.exit(1)

    else:
        # Several --out-file paths, or every .out in --directory
        if args.out_file:
            file_pairs = pair_out_files(args.out_file)
        else:
            file_pairs = find_file_pairs(args.directory)

        if not file_pairs and (args.config_file or not stdin_is_terminal()):
            # Nobody to ask for a path (or the run was meant to apply a
            # template): say so and fail rather than wait at a prompt.
            ui.err(f"No CRYSTAL output files (.out) found in {args.directory}")
            sys.exit(1)

        if not file_pairs:
            ui.warn(f"No .out files found in {args.directory}")
            # Ask for output file path
            print()
            input_file = input("Enter CRYSTAL output file path: ").strip()

            if not input_file:
                ui.warn("No file path provided. Exiting.")
                return

            input_path = Path(input_file)

            if not input_path.exists():
                ui.err(f"Error: File or directory not found: {input_file}")
                return
            
            # Check if it's a directory
            if input_path.is_dir():
                # Look for .out files in the directory
                file_pairs = find_file_pairs(input_path)
                if not file_pairs:
                    ui.err(f"\nError: No CRYSTAL output files (.out) found in directory: {input_path}")
                    ui.err("Please specify a CRYSTAL output file (e.g., material.out) or a directory containing .out files.")
                    return
                else:
                    ui.info(f"\nFound {len(file_pairs)} output file(s) in {input_path.name}:")
                    for pair in sorted(file_pairs)[:10]:  # Show first 10
                        # pair[0] is a string path, so we need to convert it to Path to get the name
                        ui.info(f"  - {Path(pair[0]).name}")
                    if len(file_pairs) > 10:
                        ui.info(f"  ... and {len(file_pairs) - 10} more")
            elif input_path.is_file():
                # Convert single file mode - process as single file
                # Create a single file pair for processing
                file_pairs = [(input_path, None)]
            else:
                ui.err(f"Error: {input_path} is not a valid file or directory.")
                return

        ui.info(f"Found {len(file_pairs)} output file(s) to process")

        config_data = None
        if args.config_file:
            # Read the template once, before any file, so a bad one stops the
            # batch instead of failing (or dropping to questions) per file.
            try:
                with open(args.config_file) as f:
                    config_data = unwrap_d12_config(json.load(f))
                if not isinstance(config_data, dict):
                    raise ValueError("not a JSON object of settings")
                fn_errs = normalize_config_functionals(config_data)[1]
                if fn_errs:
                    raise ValueError("; ".join(fn_errs))
            except Exception as e:
                ui.err(f"Cannot use config file {args.config_file}: {e}")
                sys.exit(1)
            if len(file_pairs) > 1:
                print_template_plan(args.config_file, config_data, file_pairs, args.output_dir)
                ask = not (args.yes or args.non_interactive) and stdin_is_terminal()
                if ask and not yes_no_prompt(f"\nApply to all {len(file_pairs)} files?", default="yes"):
                    ui.warn("Canceled: nothing written.")
                    sys.exit(1)

        # Ask about shared settings mode if multiple files and not specified.
        # A --config-file is the shared settings: nothing to ask.
        use_shared_settings = args.shared_settings and not args.config_file
        if not args.non_interactive and len(file_pairs) > 1 and not args.config_file:
            print()
            ui.rule("MULTIPLE FILE PROCESSING OPTIONS")
            ui.info("You can either:")
            ui.info("1. Use shared settings for all files (faster, applies same calculation settings)")
            ui.info("2. Configure each file individually (more control per file)")
            ui.info("\nNote: Geometry and symmetry are always preserved from each file.")

            use_shared = input("\nUse shared settings for all files? [Y/n]: ").strip().lower()
            use_shared_settings = use_shared != 'n'
        
        # If shared settings requested, get them once
        shared_settings = None
        if use_shared_settings and len(file_pairs) > 1:
            print()
            ui.rule("SHARED SETTINGS MODE")
            ui.info("Define calculation settings to apply to all files.")
            ui.info(
                "Note: Geometry, symmetry, and space group info will be preserved from each file."
            )

            # Use first file as template for getting settings
            first_out, first_d12 = file_pairs[0]
            ui.info(f"\nUsing {os.path.basename(first_out)} as template for settings...")

            # Parse first file to get baseline settings
            out_parser = CrystalOutputParser(first_out)
            try:
                out_data = out_parser.parse()
                settings = out_data.copy()

                if first_d12:
                    in_parser = CrystalInputParser(first_d12)
                    try:
                        in_data = in_parser.parse()
                        # Use the same merge logic as in process_files
                        for key, value in in_data.items():
                            if key in DECK_GEOMETRY_KEYS or key in DECK_TEXT_KEYS:
                                continue
                            if key == "freq_settings":
                                value = child_freq_settings(value)
                            if key not in settings or settings[key] is None:
                                settings[key] = value
                            elif key in ["functional", "dispersion", "spin_polarized", "dft_grid", "method",
                                       "is_3c_method", "smearing", "use_smearing", "smearing_width",
                                       "k_points", "scf_method", "scf_maxcycle", "fmixing", "scf_direct",
                                       "mulliken_analysis", "diis_history", "calculation_type",
                                       "optimization_settings", "freq_settings", "origin_setting",
                                       "spacegroup", "dimensionality", "tolerances",
                                       "basis_set", "basis_set_type", "basis_set_path",
                                       "unrecognised_functional", "unrecognised_functional_source",
                                       "layer_group", "rod_group"]:
                                # For all calculation settings, prefer input file (.d12) over output file (.out)
                                # INCLUDING tolerances and basis set - the output parser has issues extracting these correctly
                                if value is not None:
                                    settings[key] = value
                            elif key == "scf_settings":
                                # Merge SCF settings
                                if "scf_settings" not in settings:
                                    settings["scf_settings"] = {}
                                settings["scf_settings"].update(value)

                        prefer_deck_unrecognised_functional(settings, in_data)

                        # Check for external basis data from D12
                        template_external_basis = in_data.get("external_basis_data", [])
                        if template_external_basis:
                            settings["has_original_external_basis"] = True
                            ui.info(f"  Found external basis set data in D12 file ({len(template_external_basis)} lines)")
                    except:
                        pass

                # Get shared settings
                shared_settings = get_calculation_options_from_current(settings, shared_mode=True)
                # An ELASTIC/EOS/QHA/CPHF batch: the module's questions, once for
                # every file (each file's own run gives no answers in shared mode)
                shared_settings = ask_shared_elastic_settings(shared_settings, settings)

                print()
                ui.rule("Shared settings defined. These will be applied to all files.")

            except Exception as e:
                ui.err(f"Error getting shared settings: {e}")
                return

        # One file asks as a single --out-file does; a batch was confirmed
        # (or needs no confirming) above.
        if len(file_pairs) > 1 or args.yes:
            confirm_config = False
        else:
            confirm_config = None
        # Nobody answers per-file questions in a template batch, with --yes or
        # --non-interactive: a question fails that file instead, at a terminal
        # or not.
        unattended = bool(args.yes or args.non_interactive
                          or (args.config_file and len(file_pairs) > 1))

        # Inputs whose decks would have the same name in the same directory:
        # the later one overwrote the earlier and both counted as written.
        collisions = same_deck_name_inputs(file_pairs, args.output_dir)
        reserved_decks = {}

        # Process all file pairs
        written, failed = [], []
        total = len(file_pairs)
        for index, (out_file, d12_file) in enumerate(file_pairs, 1):
            # --out-file paths as given (dupA/X.out and dupB/X.out stay apart);
            # a --directory's files by name.
            name = out_file if args.out_file else os.path.basename(out_file)
            print()
            ui.rule(f"Processing: {name}")
            if d12_file:
                ui.info(f"With input: {os.path.basename(d12_file)}")
            else:
                ui.info(f"No {os.path.splitext(os.path.basename(out_file))[0]}.d12 beside it: "
                        "converting from the .out alone")

            LAST_RESULT["deck"] = LAST_RESULT["reason"] = None
            LAST_RESULT["notes"] = []
            if out_file in collisions:
                reason = (f"its deck would have the same name as the one from "
                          f"{', '.join(collisions[out_file])} (same file name, same output "
                          f"directory); neither is written")
                failed.append((name, reason))
                ui.err(f"({index}/{total}) {name}: FAILED: {reason}")
                continue
            try:
                if not os.path.exists(out_file):
                    raise FileNotFoundError("file not found")
                success, options = process_files(
                    out_file,
                    d12_file,
                    shared_settings,
                    config_file=args.config_file,
                    non_interactive=args.non_interactive,
                    calc_type=args.calc_type,
                    opt_type=args.opt_type,
                    origin_setting=args.origin_setting,
                    output_dir=args.output_dir,
                    confirm_config=confirm_config,
                    unattended=unattended,
                    reserved_decks=reserved_decks,
                    charge_options=charge_options,
                )
                reason = LAST_RESULT["reason"]
            except Exception as e:
                # Per-file isolation (same contract as NewCifToD12): one bad
                # structure — e.g. the deliberate monoclinic unique-axis
                # ValueError — must not abort the rest of the batch.
                ui.err(f"Error processing {name}: {e}")
                success, reason = False, str(e) or type(e).__name__
            if success:
                deck = LAST_RESULT["deck"]
                notes = list(LAST_RESULT["notes"])
                written.append((name, deck, notes))
                source = "" if d12_file else " (from the .out alone)"
                if notes:
                    ui.warn(f"({index}/{total}) {name}: wrote {deck}{source} - WARNING: "
                            + "; ".join(notes))
                else:
                    ui.ok(f"({index}/{total}) {name}: wrote {deck}{source}")
            else:
                reason = reason or "no deck written (see the messages above)"
                failed.append((name, reason))
                ui.err(f"({index}/{total}) {name}: FAILED: {reason}")

        print()
        ui.rule()
        unfinished = [(name, note) for name, _, notes in written for note in notes
                      if note in (UNFINISHED_OPT_NOTE, FAILED_OPT_NOTE)]
        summary = (f"{len(written)} written"
                   + (f" ({len(unfinished)} from unfinished or failed optimizations)"
                      if unfinished else "")
                   + f", {len(failed)} failed (of {total} files)")
        if failed:
            ui.err(summary)
            for name, reason in failed:
                ui.err(f"  {name}: {reason}")
        elif unfinished:
            ui.warn(summary)
        else:
            ui.ok(summary)
        if unfinished:
            ui.warn("Written from the starting geometry of an optimization that did not "
                    "finish or did not converge:")
            for name, note in unfinished:
                ui.warn(f"  {name}: {'no OPT END' if note == UNFINISHED_OPT_NOTE else 'OPT END - FAILED'}")
        print_soc_metal_summary([name for name, _, notes in written if SOC_METAL_NOTE in notes])

        # Save options if requested
        if args.save_options and shared_settings:
            with open(args.options_file, "w") as f:
                json.dump(options_for_template(shared_settings), f, indent=2)
            ui.ok(f"\nShared settings saved to {args.options_file}")

        # Exit status for scripts: a batch that failed on any file, or wrote
        # nothing, is not a success.
        if failed or not written:
            sys.exit(1)


if __name__ == "__main__":
    main()
