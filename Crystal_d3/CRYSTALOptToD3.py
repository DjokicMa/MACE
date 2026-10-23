#!/usr/bin/env python3
"""
CRYSTALOptToD3.py - Generate D3 property input files from CRYSTAL calculations

Author: Marcus Djokic
Institution: Michigan State University, Mendoza Group

This script generates CRYSTAL D3 input files for property calculations from completed
CRYSTAL calculations (optimization or single point). It handles:
- BAND: Electronic band structure
- DOSS: Density of states
- TRANSPORT: Boltzmann transport properties  
- CHARGE: Charge density (ECH3/ECHG)
- POTENTIAL: Electrostatic potential (POT3/POTC)
- ORBITALS: Orbitals as Molden files (NEWK + ORBITALS)

The script automatically:
- Extracts necessary information from output files
- Copies required binary files (fort.9/fort.98)
- Generates properly formatted D3 input files
- Handles all calculation types interactively
"""

import glob
import os
import shutil
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Single-sourced full-precision constant (fallback keeps this standalone-runnable).
try:
    from mace.constants import HARTREE_TO_EV, ANGSTROM_TO_BOHR
except ImportError:
    HARTREE_TO_EV = 27.211386245988
    ANGSTROM_TO_BOHR = 1.0 / 0.529177210903

# Import configuration modules
try:
    # Try absolute import first (for package usage)
    from Crystal_d3.d3_interactive import (configure_d3_calculation, get_band_info_from_output,
                               parse_basis_set_info, get_atoms_and_shells,
                               generate_orbital_indices, create_doss_projections,
                               extract_shrink_from_d12, extract_shrink_from_out,
                               ghost_atom_labels, GHOST_SUFFIX,
                               orbitals_kchoice, orbitals_file_warnings)
    from Crystal_d3.d3_kpoints import (get_band_path_from_symmetry, get_kpoint_coordinates_from_labels,
                           request_static_kpath,
                           extract_and_process_shrink, scale_kpoint_segments, get_seekpath_labels,
                           get_seekpath_full_kpath, get_literature_kpath_vectors, unicode_to_ascii_kpoint,
                           validate_kpoint_labels_for_crystal23, in_plane_path, in_plane_labels,
                           on_chain_path, chain_path_in_zone, CHAIN_PATH_SEGMENTS, CHAIN_PATH_SHRINK, CHAIN_PATH_LABELS,
                           get_crystal_system_from_space_group, CRYSTAL23_VALID_LABELS)
except ImportError:
    # Fall back to relative import (for script usage)
    from d3_interactive import (configure_d3_calculation, get_band_info_from_output,
                               parse_basis_set_info, get_atoms_and_shells,
                               generate_orbital_indices, create_doss_projections,
                               extract_shrink_from_d12, extract_shrink_from_out,
                               ghost_atom_labels, GHOST_SUFFIX,
                               orbitals_kchoice, orbitals_file_warnings)
    from d3_kpoints import (get_band_path_from_symmetry, get_kpoint_coordinates_from_labels,
                           request_static_kpath,
                           extract_and_process_shrink, scale_kpoint_segments, get_seekpath_labels,
                           get_seekpath_full_kpath, get_literature_kpath_vectors, unicode_to_ascii_kpoint,
                           validate_kpoint_labels_for_crystal23, in_plane_path, in_plane_labels,
                           on_chain_path, chain_path_in_zone, CHAIN_PATH_SEGMENTS, CHAIN_PATH_SHRINK, CHAIN_PATH_LABELS,
                           get_crystal_system_from_space_group, CRYSTAL23_VALID_LABELS)
from d3_config import (save_d3_config, load_d3_config, validate_d3_config,
                      print_d3_config_summary, save_d3_options_prompt,
                      list_available_d3_configs, select_d3_config_file, apply_doss_override, set_doss_override,
                      DOSS_STEP_PRESETS, DOSS_MAX_NPTS,
                      doss_atomic_numbers, doss_span, doss_step_ev, doss_explicit_points,
                      doss_npts, doss_grid_note, doss_window_offer, set_non_interactive,
                      DOSS_DEFAULT_WINDOW_EV, doss_window_ha, doss_keeps_all_bands, doss_span_estimate_ev,
                      doss_band_counts, doss_band_index_range, doss_conduction_bands, doss_snap_window,
                      DOSS_BAND_INDEX_SPAN_EV, DOSS_SNAP_MAX_EV, DOSS_REACH_EV,
                      apply_transport_override, set_transport_override, transport_mesh, reciprocal_lengths,
                      TRANSPORT_K_PRESETS)
import sys
from pathlib import Path
# Add Crystal_d12 to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "Crystal_d12"))
from d12_interactive import yes_no_prompt

# Opt-in "press b to go back" navigation (no-op if menu_nav is unavailable).
try:
    from menu_nav import run_with_back as _run_with_back
except Exception:  # pragma: no cover - defensive fallback
    def _run_with_back(flow_fn):
        return flow_fn()
from d12_constants import (SPACEGROUP_SYMBOLS, SPACEGROUP_ALTERNATIVES,
                          SPACEGROUP_SYMBOL_TO_NUMBER)

# Unified MACE visual layer. This file runs both via `mace opt2d3` (mace on
# sys.path) AND standalone (`python CRYSTALOptToD3.py`, where mace may be
# absent), so the import is fully guarded with a plain-text shim fallback.
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


# MAPNET unit keywords (manual 26810-26813): ANGSTROM (the documented default,
# so no keyword is written), BOHR and FRACTION. Only FRACTION and the Angstrom
# default are offered here; anything else is refused rather than guessed at,
# because the two differ by the lattice parameter and CRYSTAL cannot tell that
# a plane was given in the wrong unit.
_MAP_FRACTION_WORDS = {"F", "FRAC", "FRACTION", "FRACTIONAL"}
_MAP_ANGSTROM_WORDS = {"C", "CART", "CARTESIAN", "ANGSTROM"}

# ECH3/POT3 grids of a slab, polymer or molecule (manual p.322 and p.351): SCALE
# multiplies the extent of the atoms along each non-periodic direction, so a flat
# structure (a monolayer along z, a planar chain along z) has none to scale.
# CRYSTAL23 then stops ("GRID3D **** USE RANGE NOT SCALE - SYSTEM HAS ZERO EXTENT
# IN Z", monolayer graphene) or writes a cube one point thick with an infinite
# step (planar polyacetylene). An axis the atoms span less than FLAT_EXTENT_A
# makes the deck use RANGE instead: the atoms' extent plus FLAT_PAD_A on each
# side of every non-periodic direction (Cartesian, written in bohr).
FLAT_EXTENT_A = 0.5
FLAT_PAD_A = 5.0


# Two-component (SOC) parents. The .out markers and the deck's TWOCOMPON block
# are defined once, in mace/utils/calc_detection.py (standard library only);
# run standalone, it is loaded by path, and a copy of this script with neither
# beside it uses the same markers inline.
try:
    from mace.utils.calc_detection import (is_two_component_deck, is_two_component_output,
                                           orbitals_root)
except ImportError:
    try:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            "_mace_calc_detection",
            Path(__file__).resolve().parent.parent / "mace" / "utils" / "calc_detection.py")
        _calc_detection = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_calc_detection)
        is_two_component_deck = _calc_detection.is_two_component_deck
        is_two_component_output = _calc_detection.is_two_component_output
        orbitals_root = _calc_detection.orbitals_root
    except Exception:  # pragma: no cover - flat copy without the mace package
        def orbitals_root(stem: str) -> str:
            return f"{stem}_orb"

        def is_two_component_output(content: str) -> bool:
            return re.search(r"^[ \t]*(?:TOTAL [XY]-COMP MAGNETIZATION[ \t]"
                             r"|- NUMBER OF FULLY OCCUPIED/TOTAL SPINORS[ \t])",
                             content, re.MULTILINE) is not None

        def is_two_component_deck(content: str) -> bool:
            return any(line.strip().upper() == "TWOCOMPON"
                       for line in content.splitlines()[1:])

# What properties can do with a two-component (2c-SCF) wavefunction. The
# CRYSTAL23 manual, sec. 6.4 (p. 183), lists the properties keywords the 2c
# implementation supports: NEWK, BAND, ECHG, ECH3, PPAN, PROPS2COMP, ANBD,
# BWIDTH, DOSS and TOPO (and auxiliary ones), and p. 166 says "Any feature not
# explicitly mentionned in this chapter is not supported" in 2c. BOLTZTRA
# (TRANSPORT) and POT3/POTC (POTENTIAL) are not in the list.
TWO_COMPONENT_D3_TYPES = {"BAND", "DOSS", "CHARGE", "ECH3", "ECHG"}

# Measured on HPCC (a molecule, stock serial properties on a 2c fort.9):
# ECH3 ends with rc 0 but DENS_CUBE.DAT is 0 bytes. Refused for 0D 2c parents.
TWO_COMPONENT_0D_REFUSED = {"CHARGE", "ECH3", "ECHG"}
TWO_COMPONENT_0D_CUBE_REASON = (
    "Measured on HPCC: the stock properties runs ECH3 on a 2c molecule's fort.9 "
    "with exit code 0 but writes an empty (0-byte) DENS_CUBE.DAT, and ECHG was "
    "never run from one, so MACE writes no charge-density deck for a 0D "
    "two-component parent. Run the charge density from the scalar (1c) SCF.")

_TWO_COMPONENT_REFUSED = {
    "TRANSPORT": "BOLTZTRA", "POTENTIAL": "POT3/POTC", "POT3": "POT3", "POTC": "POTC",
    "CHARGE+POTENTIAL": "POT3",
}


def _resolve_map_coord_type(value: Any) -> Optional[str]:
    """Map an answer/config value onto "FRACTION" or "ANGSTROM", else None."""
    word = str(value).strip().upper()
    if word in _MAP_FRACTION_WORDS:
        return "FRACTION"
    if word in _MAP_ANGSTROM_WORDS:
        return "ANGSTROM"
    return None


def _strip_run_suffix(stem: str) -> str:
    """The name opt2d3 gives a run's files: its stem without _opt/_sp."""
    for suffix in ['_opt', '_sp', '_OPT', '_SP', '.out', '.log']:
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


# "Edit the band k-path?" is asked at a terminal, or on piped answers with
# --edit-kpath: 1.1.5 did not ask it, so piped 1.1.5 answers keep their order
# (the GUI's Calculation Setup has its own k-path editor).
_EDIT_KPATH = False


def request_kpath_question() -> None:
    """Ask "Edit the band k-path?" on piped answers too (--edit-kpath)."""
    global _EDIT_KPATH
    _EDIT_KPATH = True


def _kpath_question_wanted() -> bool:
    if _EDIT_KPATH:
        return True
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except Exception:
        return False


class D3Generator:
    """Handle D3 file generation from CRYSTAL output files."""
    
    def __init__(self, input_file: str, calc_type: str, output_dir: Optional[str] = None):
        self.input_file = Path(input_file).resolve()
        self.calc_type = calc_type.upper()
        self.base_name = self.input_file.stem
        
        # Remove common suffixes to get clean base name
        self.base_name = _strip_run_suffix(self.base_name)
        # anatase_sp.out and anatase_opt.out side by side would both write
        # anatase_band.d3 (and anatase_band.f9), the second over the first;
        # so would anatase.out and anatase_sp.out, and anatase_opt_sp.out
        # would take the anatase_opt_* name anatase_opt.out keeps when it
        # collides. With another output that has (or makes) the same name
        # beside it, each keeps its whole name.
        if self.base_name != self.input_file.stem:
            for other in self.input_file.parent.glob(f"{glob.escape(self.base_name)}*"):
                if (other.suffix in ('.out', '.log') and other.resolve() != self.input_file
                        and self.base_name in (other.stem, _strip_run_suffix(other.stem))):
                    ui.info(f"Note: {other.name} is beside {self.input_file.name}; naming the "
                            f"files {self.input_file.stem}_* so neither overwrites the other")
                    self.base_name = self.input_file.stem
                    break

        self.input_dir = self.input_file.parent
        self.output_dir = Path(output_dir).resolve() if output_dir else self.input_dir
        
        # Parse output file for structure info
        self.structure_info = self._parse_output_file()
        # True while generate_d3 gathers the settings by asking on stdin
        self._answers_on_stdin = False

    def _can_ask(self) -> bool:
        """May a writer ask a question here?

        At a terminal, or while the settings are being asked on stdin (a
        terminal or answers piped in, as 1.1.1 read them). A run given its
        settings (--config-file, shared settings) with no terminal may not.
        """
        if self._answers_on_stdin:
            return True
        try:
            return sys.stdin is not None and sys.stdin.isatty()
        except Exception:
            return False
    
    def parent_is_two_component(self) -> bool:
        """Whether the parent run was a two-component (TWOCOMPON) SCF: its .out
        has the lines CRYSTAL prints for a 2c-SCF, or its .d12 (same name,
        same folder) has the TWOCOMPON block."""
        try:
            if is_two_component_output(self.input_file.read_text(errors="replace")):
                return True
        except OSError:
            pass
        deck = self.input_file.with_suffix(".d12")
        try:
            return deck.exists() and is_two_component_deck(deck.read_text(errors="replace"))
        except OSError:
            return False

    def _parse_output_file(self) -> Dict[str, Any]:
        """Extract structure information from output file."""
        info = {
            'dimensionality': 3,
            'n_atoms': 0,
            'n_ao': 0,
            'n_electrons': 0,
            'space_group': 1,
            'lattice_type': 'P',
            'crystal_family': 'triclinic',
            'title': self.base_name,
            'atom_elements': []  # List of element symbols for each atom
        }
        
        if not self.input_file.exists():
            ui.warn(f"Warning: Output file not found: {self.input_file}")
            return info
        
        with open(self.input_file, 'r') as f:
            content = f.read()
        
        # Dimensionality
        if 'SLAB CALCULATION' in content or 'SLAB GROUP' in content:
            info['dimensionality'] = 2
        elif 'POLYMER CALCULATION' in content:
            info['dimensionality'] = 1
        elif 'MOLECULAR CALCULATION' in content:
            info['dimensionality'] = 0
        
        # Number of atoms - use total atoms in unit cell for DOSS projections
        atoms_match = re.search(r'ATOMS IN THE UNIT CELL:\s+(\d+)', content)
        if atoms_match:
            info['n_atoms'] = int(atoms_match.group(1))
        else:
            # Fallback to asymmetric unit if unit cell count not found
            atoms_match = re.search(r'ATOMS IN THE ASYMMETRIC UNIT\s+(\d+)', content)
            if atoms_match:
                info['n_atoms'] = int(atoms_match.group(1))
        
        # Number of AOs
        ao_match = re.search(r'NUMBER OF AO\s+(\d+)', content)
        if ao_match:
            info['n_ao'] = int(ao_match.group(1))
        
        # Number of electrons
        # CRYSTAL prints "N. OF ELECTRONS PER CELL   632" (not PER UNIT CELL)
        elec_match = re.search(r'N\. OF ELECTRONS PER (?:UNIT )?CELL\s+(\d+)', content)
        if elec_match:
            info['n_electrons'] = int(elec_match.group(1))
        
        # Space group symbol and lattice type
        # Updated regex to properly capture full space group symbols like "F D 3 M" or "P -4 M 2"
        # Capture everything after the colon that starts with a letter, then strip whitespace
        sg_symbol_match = re.search(r'SPACE GROUP[^:]*:\s*([A-Z][^$\n]+)', content)
        if sg_symbol_match:
            symbol = sg_symbol_match.group(1).strip()
            # Extract lattice type (first letter)
            if symbol:
                info['lattice_type'] = symbol[0]
                
                # Try to map symbol to space group number
                # First try the symbol as-is
                if symbol in SPACEGROUP_SYMBOL_TO_NUMBER:
                    info['space_group'] = SPACEGROUP_SYMBOL_TO_NUMBER[symbol]
                # Try alternatives mapping (includes spaced versions)
                elif symbol in SPACEGROUP_ALTERNATIVES:
                    info['space_group'] = SPACEGROUP_ALTERNATIVES[symbol]
                # Try removing spaces
                elif symbol.replace(' ', '') in SPACEGROUP_SYMBOL_TO_NUMBER:
                    info['space_group'] = SPACEGROUP_SYMBOL_TO_NUMBER[symbol.replace(' ', '')]
                # Try with dashes
                elif symbol.replace(' ', '-') in SPACEGROUP_SYMBOL_TO_NUMBER:
                    info['space_group'] = SPACEGROUP_SYMBOL_TO_NUMBER[symbol.replace(' ', '-')]
                
        # Space group number - try different patterns
        # Pattern 1: SPACE GROUP NUMBER: 123
        sg_match = re.search(r'SPACE GROUP.*?NUMBER:\s*(\d+)', content)
        if sg_match:
            info['space_group'] = int(sg_match.group(1))

        # A slab names its layer group and the corresponding space group
        # ("TWO-SIDED PLANE GROUP N. 80 : P 6/M M M", "CORRESPONDING SPACE
        # GROUP N. 191"); layer groups are identified by that space group
        # (manual App. A.2, p.421), and the plane-group symbol gives the
        # lattice letter (P or C).
        if info['dimensionality'] == 2:
            sg_match = re.search(r'CORRESPONDING SPACE GROUP N\.\s*(\d+)', content)
            if sg_match:
                info['space_group'] = int(sg_match.group(1))
            lattice_match = re.search(r'PLANE GROUP N\.\s*\d+\s*:\s*([A-Z])', content)
            if lattice_match:
                info['lattice_type'] = lattice_match.group(1)

        # Crystal family
        if 'CRYSTAL FAMILY' in content:
            family_match = re.search(r'CRYSTAL FAMILY\s+:\s+(\w+)', content)
            if family_match:
                info['crystal_family'] = family_match.group(1).lower()
        
        # Title from output
        title_match = re.search(r'^\s*\*+\s+(.+?)\s+\*+', content, re.MULTILINE)
        if title_match:
            info['title'] = title_match.group(1).strip()
        
        # Extract atom elements from ATOM listing
        # Look for pattern like: "1 T 6 C" or "1 T 22 TI" where the trailing token
        # is the element symbol. CRYSTAL prints symbols in ALL CAPS ("TI", "PB",
        # "SI"), so the second letter must also allow uppercase; the old
        # "[A-Z][a-z]?" captured only "T" for titanium and mislabeled every
        # two-letter element in the DOSS atom projections.
        # CRYSTAL reprints the full geometry once per optimization step, so a
        # findall over the whole file multiplied the atom list by the number of
        # steps (e.g. 34 atoms -> 1428 entries for Ti6O11). The leading atom
        # index runs 1..N within one listing and resets on the next block, so
        # collect only the first contiguous, strictly-increasing run. This makes
        # len(atom_elements) == n_atoms again, so the index-bounds checks in the
        # projection code reject out-of-range atoms instead of mislabeling them
        # from a later geometry block. The first n_atoms entries are unchanged,
        # so every valid projection is byte-identical to before.
        atom_pattern = re.compile(r'^\s*(\d+)\s+[TF]\s+\d+\s+([A-Z][A-Za-z]?)\s+', re.MULTILINE)
        atom_elements = []
        last_idx = 0
        for m in atom_pattern.finditer(content):
            idx = int(m.group(1))
            if idx <= last_idx:
                break
            atom_elements.append(m.group(2))
            last_idx = idx
        # GHOSTS atoms keep their basis but are not that element any more;
        # the geometry listing (printed before GHOSTS acts) still names them.
        for label in ghost_atom_labels(content):
            if 0 < label <= len(atom_elements):
                atom_elements[label - 1] += GHOST_SUFFIX
        if atom_elements:
            info['atom_elements'] = atom_elements

        return info
    
    def _copy_wavefunction(self) -> bool:
        """Find the fort.9/fort.98 to copy to the D3 file's name.

        The copy itself is made by _place_wavefunction once the deck is
        written: a run that ends without a deck (input ran out, a path it
        cannot write) used to leave a full <base>_<type>.f9 behind.
        """
        self._pending_wavefunction = None
        # Look for wavefunction files - try various naming patterns
        # The output's own wavefunction first: X_opt_sp.out is named X_opt_*,
        # and X_opt.f9 beside it is the optimization's, not this run's.
        input_stem = self.input_file.stem
        wf_files = [
            f'{input_stem}.f9', f'{input_stem}.f98',
            'fort.9', 'fort.98', 
            f'{self.base_name}.f9', f'{self.base_name}.f98',
        ]
        
        # Add numbered suffix patterns (sp, sp2, sp3...sp99, opt, opt2...opt99, etc.)
        for suffix in ['sp', 'opt', 'freq', 'soc']:
            # Base suffix without number
            wf_files.extend([
                f'{self.base_name}_{suffix}.f9', 
                f'{self.base_name}_{suffix}.f98'
            ])
            # Numbered suffixes from 2 to 99
            for i in range(2, 100):
                wf_files.extend([
                    f'{self.base_name}_{suffix}{i}.f9', 
                    f'{self.base_name}_{suffix}{i}.f98'
                ])
        
        source_wf = None
        for wf_file in wf_files:
            wf_path = self.input_dir / wf_file
            if wf_path.exists():
                source_wf = wf_path
                break
        
        if not source_wf:
            print()
            ui.warn("Warning: No wavefunction file (fort.9/fort.98) found!")
            ui.warn("The D3 calculation will fail without the wavefunction file.")
            # Non-interactive (batch/workflow callback): fail cleanly instead of
            # raising EOFError at the prompt. Nothing is on disk yet - the deck
            # is only written after this returns - so just signal failure and
            # let the caller report it instead of claiming success.
            if not self._can_ask():
                ui.err("Aborting D3 file creation (no wavefunction, no terminal to confirm).")
                return False
            cont = yes_no_prompt("Continue anyway?", "no")
            return cont
        
        # Determine target filename based on calculation type
        d3_name = f"{self.base_name}_{self.calc_type.lower()}"
        target_wf = self.output_dir / f"{d3_name}.f9"
        
        if source_wf != target_wf:
            self._pending_wavefunction = (source_wf, target_wf)
        return True

    def _place_wavefunction(self):
        """Copy the wavefunction _copy_wavefunction found, now the deck exists."""
        pending = getattr(self, '_pending_wavefunction', None)
        self._pending_wavefunction = None
        if not pending:
            return
        source_wf, target_wf = pending
        print()
        ui.info(f"Copying wavefunction: {source_wf.name} -> {target_wf.name}")
        shutil.copy2(source_wf, target_wf)
    
    def _parse_manual_projection(self, spec: str, total_shells: Dict, atom_elements: List[str]) -> Optional[str]:
        """Parse manual projection specification and convert to DOSS format.
        
        Handles formats like:
        - "1" or "1C" -> "-1 1  #1C"
        - "C all" or "C" -> element total projection
        - "C S" -> element S orbital projection
        """
        spec = spec.strip()
        if not spec:
            return None
        
        # Check if it's an atom projection (starts with a number)
        if spec[0].isdigit():
            # Extract atom number
            atom_num_match = re.match(r'^(\d+)', spec)
            if atom_num_match:
                atom_num = int(atom_num_match.group(1))
                if 0 < atom_num <= len(atom_elements):
                    element = atom_elements[atom_num - 1]
                    return f"-1 {atom_num}  #{atom_num}{element}"
                else:
                    return None
        
        # Otherwise, it's an element/orbital projection
        parts = spec.split()
        if not parts:
            return None
        
        element = parts[0].upper()
        
        # Check if element exists in the system
        if element not in total_shells:
            return None
        
        # If just element name or "element all", return element total
        if len(parts) == 1 or (len(parts) == 2 and parts[1].lower() == "all"):
            # Create element total projection - combine all orbitals
            all_indices = []
            for shell_type in ["S", "P", "SP", "D", "F"]:
                if shell_type in total_shells[element]:
                    all_indices.extend(total_shells[element][shell_type])
            
            if all_indices:
                return f"{len(all_indices)} {' '.join(map(str, all_indices))}  #{element} all"
            else:
                return None
        
        # Specific orbital type
        if len(parts) == 2:
            orbital_type = parts[1].upper()
            
            # Check if orbital type exists for this element
            if orbital_type in total_shells[element] and total_shells[element][orbital_type]:
                indices = total_shells[element][orbital_type]
                return f"{len(indices)} {' '.join(map(str, indices))}  #{element} {orbital_type}"
            else:
                # Check for combined SP
                if orbital_type == "SP" and "S" in total_shells[element] and "P" in total_shells[element]:
                    # Combine S and P
                    indices = total_shells[element]["S"] + total_shells[element]["P"]
                    return f"{len(indices)} {' '.join(map(str, indices))}  #{element} SP"
                else:
                    return None
        
        return None
    
    def _in_plane_band_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """The BAND settings of a slab, cut to the plane of the slab.

        A slab is periodic in a and b; BAND takes I3 = J3 = 0 in 2D (manual
        p.310, note 3). The path settings were made for the corresponding 3D
        space group, so the segments (or label edges) that leave the kz = 0
        plane are dropped. The title labels are cut the same way, or left out
        when they do not name the segments one to one.
        """
        config = dict(config)
        space_group = self.structure_info.get('space_group', 1)
        lattice_type = self.structure_info.get('lattice_type', 'P')
        method = config.get("path_method")
        if method == "labels":
            path = config.get("path")
            if not path or path == "auto":
                path = get_band_path_from_symmetry(space_group, lattice_type)
            config["path"] = in_plane_labels(path, space_group, lattice_type)
        elif method != "manual" and config.get("segments"):
            labels = config.get("path_labels")
            if labels is None and config.get("seekpath_full"):
                labels = get_seekpath_labels(space_group, lattice_type, str(self.input_file))
            segments, labels = in_plane_path(config["segments"], labels)
            if not segments:
                # Nothing of this path lies in the plane: take the in-plane
                # SeeK-path path of the corresponding space group.
                frac, _ = get_seekpath_full_kpath(space_group, lattice_type, str(self.input_file))
                frac, _ = in_plane_path(frac)
                labels = get_seekpath_labels(space_group, lattice_type, str(self.input_file))
                segments, config["shrink"] = scale_kpoint_segments(frac, config.get("shrink", 16))
                ui.warn("  No segment of this k-path lies in the slab plane; "
                        "using the in-plane SeeK-path path instead")
            config["segments"] = segments
            config["in_plane_labels"] = labels
        return config

    def _chain_band_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """The BAND settings of a polymer, cut to its chain axis.

        CRYSTAL23 takes no label records for a polymer (rod groups 1, 28 and
        51 stopped with "BANDE **** FORMAT ERROR IN INPUT DECK") and only the
        k1 component of each point (manual p.310, note 3). A label path, or
        a k-point path with no segment along the chain, becomes Gamma -> X
        whatever space group the polymer's symbol maps to; the along-chain
        segments of a k-point path are kept.
        """
        segments = []
        if config.get("path_method") != "labels" and config.get("segments"):
            segments = chain_path_in_zone(on_chain_path(config["segments"]), config.get("shrink") or 0)
        config = {k: v for k, v in config.items()
                  if k not in ("seekpath_full", "literature_path", "path", "path_labels")}
        config["path_method"] = "coordinates"
        if segments:
            config["segments"] = segments
            config["path_labels"] = ["along the chain"]
        else:
            config.update(segments=[list(s) for s in CHAIN_PATH_SEGMENTS],
                          shrink=CHAIN_PATH_SHRINK, path_labels=list(CHAIN_PATH_LABELS))
        return config

    def _label_path_as_coordinates(self, config: Dict[str, Any],
                                   path: List[str]) -> Optional[Dict[str, Any]]:
        """A label BAND configuration rewritten as opt2d3's k-point vector one.

        The labels' Table 14.1/14.2 coordinates are scaled to integers in
        units of 1/ISS, ISS from the parent's shrink -- the deck opt2d3's
        format 2 writes for the same parent. None when a label has no
        coordinates for this lattice.
        """
        space_group = self.structure_info.get('space_group', 1)
        lattice_type = self.structure_info.get('lattice_type', 'P')
        frac: List[List[float]] = []
        run: List[str] = []
        for label in list(path) + ["|"]:
            if label != "|":
                run.append(label)
                continue
            if len(run) >= 2:
                segments = get_kpoint_coordinates_from_labels(run, space_group, lattice_type)
                if not segments:
                    return None
                frac += segments
            run = []
        if not frac:
            return None
        out_file = str(self.input_file)
        shrink = extract_and_process_shrink(out_file, Path(out_file).stem.split('_')[0],
                                            Path(out_file).parent, {})
        segments, shrink = scale_kpoint_segments(frac, shrink)
        config = dict(config)
        config.update(path_method="coordinates", segments=segments, shrink=shrink,
                      path_labels=list(path))
        return config

    def _write_band_d3(self, config: Dict[str, Any]) -> Optional[str]:
        """Write BAND calculation D3 file (None when no readable deck can be
        written). An edited path (config["band_path"], d3_band_path) is written
        from the deck of the computed path; without one the computed deck is
        returned as it is."""
        deck = self._write_computed_band_d3(config)
        band_path = config.get("band_path")
        if deck is None or not band_path:
            return deck
        import d3_band_path as kpath
        parsed = kpath.parse_band_deck(deck)
        try:
            norm = kpath.normalize(band_path)
        except ValueError as exc:
            ui.err(f"BAND: the edited k-path (band_path) cannot be used: {exc}; no D3 file was written.")
            return None
        if parsed and kpath.same_path(norm, parsed["segments"]):
            return deck                     # the computed path as it is: the same deck
        if parsed and parsed["labels_mode"] and not any("k_from" in s or "k_to" in s for s in norm):
            norm = self._crystal_readable_label_edit(norm)
            if norm is None:
                return None
        if parsed and parsed["labels_mode"] and any("k_from" in s or "k_to" in s for s in norm):
            # a point given by coordinates: the whole path is written as coordinates
            converted = self._label_path_as_coordinates(config, kpath.path_tokens(parsed["segments"]))
            if converted is None:
                ui.err("BAND: the edited k-path adds coordinates, but this path's labels have no "
                       "coordinates for this lattice; no D3 file was written.")
                return None
            deck = self._write_computed_band_d3(converted)
            parsed = kpath.parse_band_deck(deck)
        _, warnings = kpath.validate(norm, parsed["segments"] if parsed else None)
        for w in warnings:
            ui.warn(f"  BAND: {w}")
        try:
            edited = kpath.write_band_deck(deck, norm, scale_kpoint_segments,
                                           dimensionality=self.structure_info.get('dimensionality', 3))
        except ValueError as exc:
            ui.err(f"BAND: the edited k-path (band_path) cannot be written: {exc}; no D3 file was written.")
            return None
        ui.info("  BAND: k-path as edited: " + " ".join(kpath.path_tokens(norm)))
        return edited

    def _crystal_readable_label_edit(self, norm: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """An edited labels-only path that CRYSTAL23 can read.

        CRYSTAL23 reads BAND labels only from its Tables 14.1/14.2 (manual
        p. 309, note 6); any other label stops the run with "K-POINTS NOT
        CODIFIED". When every label is in the table the path is returned as
        it is. Otherwise every point gets its Table 14.1/14.2 coordinates, so
        the deck is written as coordinates (CRYSTAL does not mix the two
        forms); a label MACE has no coordinates for is refused by name
        (None)."""
        import d3_band_path as kpath
        kpts = sys.modules[validate_kpoint_labels_for_crystal23.__module__]
        space_group = self.structure_info.get('space_group', 1)
        lattice_type = self.structure_info.get('lattice_type', 'P')
        system = kpts.get_crystal_system_from_space_group(space_group, lattice_type)
        if system not in kpts.KPOINT_COORDINATES:
            return norm
        valid = kpts.CRYSTAL23_VALID_LABELS.get(system, set())
        shown = kpath.visible(norm)
        labels = list(dict.fromkeys(kpath.canonical_label(s[e]) for s in shown for e in ("from", "to")))
        unread = [lab for lab in labels if lab not in ("GAMMA", "G") and lab not in valid]
        if not unread:
            return norm
        table = kpts.KPOINT_COORDINATES[system]
        missing = [lab for lab in labels if lab not in table]
        if missing:
            ui.err(f"BAND: the edited k-path names {', '.join(unread)}, which CRYSTAL23 does not "
                   f"read as a label for this lattice ({system}, space group {space_group}; "
                   f"manual p. 309, note 6: only the labels of Tables 14.1/14.2, else "
                   f"\"K-POINTS NOT CODIFIED\"), and MACE has no coordinates for "
                   f"{', '.join(missing)} to write the path as coordinates instead. Give the "
                   f"coordinates (k_from/k_to) in band_path. No D3 file was written.")
            return None
        ui.warn(f"  BAND: CRYSTAL23 does not read {', '.join(unread)} as a label for this lattice "
                f"({system}; manual p. 309, note 6), so the edited path is written as "
                f"coordinates (Tables 14.1/14.2).")
        out = []
        for s in norm:
            seg = dict(s)
            for end in ("from", "to"):
                lab = kpath.canonical_label(s[end])
                if lab in table:
                    seg[f"k_{end}"] = list(table[lab])
            out.append(seg)
        return out

    def _offer_band_path_edit(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Ask whether to edit the k-path; the edit goes into config["band_path"]
        (saved with --save-config, as the GUI's Calculation Setup writes it)."""
        import d3_band_path as kpath
        try:
            if not yes_no_prompt("\nEdit the band k-path (reorder, reverse, remove or add points)?", "no"):
                return config
        except EOFError:
            return config
        computed = self.computed_band_path(config)
        if not computed:
            ui.warn("  This path cannot be edited here (no BAND segments were computed)")
            return config
        try:
            edited = kpath.edit_interactively(computed, input, print,
                                              self.structure_info.get('dimensionality', 3))
        except EOFError:
            return config
        if edited is None:
            return config
        config = dict(config)
        # points of the computed path by label (so the edit carries over to other
        # materials, as an automatic path does); added points with coordinates
        known = kpath.points(computed)
        config["band_path"] = [{k: v for k, v in seg.items()
                                if not (k.startswith("k_") and kpath.canonical_label(seg[k[2:]]) in known
                                        and kpath._same_k(v, known[kpath.canonical_label(seg[k[2:]])]))}
                               for seg in edited]
        return config

    def computed_band_path(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """The segments (labels, fractional k) the deck of ``config`` computes, before any edit."""
        import d3_band_path as kpath
        deck = self._write_computed_band_d3({k: v for k, v in config.items() if k != "band_path"})
        parsed = kpath.parse_band_deck(deck) if deck else None
        return parsed["segments"] if parsed else []

    def _write_computed_band_d3(self, config: Dict[str, Any]) -> Optional[str]:
        """The BAND deck of the computed (unedited) path."""
        lines = ["BAND"]

        dimensionality = self.structure_info.get('dimensionality', 3)
        if dimensionality == 1 and config.get("path_method") != "manual":
            config = self._chain_band_config(config)
        if config.get("path_method") == "labels":
            space_group = self.structure_info.get('space_group', 1)
            lattice_type = self.structure_info.get('lattice_type', 'P')
            path = config.get("path")
            if not path or path == "auto":
                path = get_band_path_from_symmetry(space_group, lattice_type)
            # CRYSTAL23 reads BAND labels (ISS = 0) only from its Tables
            # 14.1/14.2 and only in 3D: a triclinic lattice has no table
            # entry ("BANDE **** K-POINTS NOT CODIFIED") and a slab takes no
            # label record at all ("BANDE **** FORMAT ERROR IN INPUT DECK"),
            # both measured with CRYSTAL23. Write the same path as integer
            # coordinates instead, as opt2d3's k-point vector format does.
            if (dimensionality == 2
                    or not validate_kpoint_labels_for_crystal23(path, space_group, lattice_type)[0]):
                converted = self._label_path_as_coordinates(config, path)
                if converted is None:
                    ui.err("BAND: the k-point labels " + " ".join(path) + " cannot be written "
                           "for this lattice (CRYSTAL23 does not recognize them as labels and "
                           "MACE has no coordinates for them); no D3 file was written. Use "
                           "k-point vectors or a manual path instead.")
                    return None
                config = converted

        in_plane = dimensionality == 2
        if in_plane:
            config = self._in_plane_band_config(config)

        # Determine the path info for the title
        path_info = ""
        if in_plane and "in_plane_labels" in config:
            labels = config["in_plane_labels"]
            path_info = (" - " + "-".join(l if l == "|" else unicode_to_ascii_kpoint(l) for l in labels)
                         if labels else " - in-plane path")
        elif config.get("path_method") == "labels":
            # Get the path labels
            if "path" in config and config["path"] != "auto":
                path = config["path"]
            else:
                # Auto-detect path
                space_group = self.structure_info.get('space_group', 1)
                lattice_type = self.structure_info.get('lattice_type', 'P')
                path = get_band_path_from_symmetry(space_group, lattice_type)
            # Check if path contains discontinuous segments (indicated by |)
            path_str = []
            for label in path:
                if label == "|":
                    path_str.append("|")
                else:
                    # Convert Unicode to ASCII
                    path_str.append(unicode_to_ascii_kpoint(label))
            path_info = " - " + "-".join(path_str)
        elif config.get("seekpath_full"):
            # Get SeeK-path labels if available
            space_group = self.structure_info.get('space_group', 1)
            lattice_type = self.structure_info.get('lattice_type', 'P')
            seekpath_labels = get_seekpath_labels(space_group, lattice_type, str(self.input_file))
            if len(seekpath_labels) > 1:
                # Join labels with hyphens, preserving | for discontinuities
                path_str = []
                for label in seekpath_labels:
                    if label == "|":
                        path_str.append("|")
                    else:
                        # Convert Unicode to ASCII (though these should already be ASCII)
                        path_str.append(unicode_to_ascii_kpoint(label))
                path_info = " - " + "-".join(path_str)
            else:
                path_info = " - SeeK-path"
        elif config.get("literature_path"):
            # Get literature path labels
            space_group = self.structure_info.get('space_group', 1)
            lattice_type = self.structure_info.get('lattice_type', 'P')
            # Literature paths use the same labels as standard paths for now
            path_labels = get_band_path_from_symmetry(space_group, lattice_type)
            path_str = []
            for label in path_labels:
                if label == "|":
                    path_str.append("|")
                else:
                    path_str.append(unicode_to_ascii_kpoint(label))
            path_info = " - " + "-".join(path_str)
        elif config.get("path_method") == "coordinates":
            # For vector paths, check if we have stored path labels
            if "path_labels" in config:
                path_str = []
                for label in config["path_labels"]:
                    if label == "|":
                        path_str.append("|")
                    else:
                        path_str.append(unicode_to_ascii_kpoint(label))
                path_info = " - " + "-".join(path_str)
            else:
                # Try to get standard path labels
                space_group = self.structure_info.get('space_group', 1)
                lattice_type = self.structure_info.get('lattice_type', 'P')
                path_labels = get_band_path_from_symmetry(space_group, lattice_type)
                path_str = []
                for label in path_labels:
                    if label == "|":
                        path_str.append("|")
                    else:
                        path_str.append(unicode_to_ascii_kpoint(label))
                path_info = " - " + "-".join(path_str)
        elif config.get("path_method") == "manual":
            path_info = " - Manual path"
        
        # Determine the k-path source for the title
        kpath_source = config.get("kpath_source", "default")
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
        
        # Always include material name, source, and path in title
        title = f"{self.base_name} - Band Structure{source_info}{path_info}"
        if config.get("title") and config["title"] != "Band Structure":
            # If there's a custom title, include it
            title = f"{self.base_name} - {config['title']}{source_info}{path_info}"
        lines.append(title)
        
        if config.get("path_method") == "labels":
            # High-symmetry labels - use appropriate path based on space group
            if "path" not in config or config.get("path") == "auto":
                # Auto-detect path based on space group and lattice type
                space_group = self.structure_info.get('space_group', 1)
                lattice_type = self.structure_info.get('lattice_type', 'P')
                path = get_band_path_from_symmetry(space_group, lattice_type)
            else:
                path = config.get("path", ["GAMMA", "X", "M", "GAMMA"])

            # Every label left here is one CRYSTAL23 reads (the others
            # were written as coordinates above)
            validated_path = path

            # Use original label format
            # Count segments exactly as they are emitted below —
            # consecutive pairs within runs between "|" discontinuities.
            # The old counter also paired labels ACROSS each "|", so the
            # NLINE header claimed one more segment than was written and
            # CRYSTAL read END as a segment.
            n_segments = 0
            run_len = 0
            for _label in validated_path:
                if _label == "|":
                    if run_len >= 2:
                        n_segments += run_len - 1
                    run_len = 0
                else:
                    run_len += 1
            if run_len >= 2:
                n_segments += run_len - 1

            shrink = 0
            n_points = config.get("n_points", 1000)

            # Use band range from config
            first_band = config.get("first_band", 1)
            last_band = config.get("last_band")
            if last_band is None:
                last_band = self.structure_info['n_ao'] if self.structure_info['n_ao'] > 0 else 100

            plot = 1 if config.get("plot", True) else 0
            print_eig = 1 if config.get("print_eigenvalues", False) else 0

            lines.append(f"{n_segments} {shrink} {n_points} {first_band} {last_band} {plot} {print_eig}")

            # Add path segments - handle discontinuous paths
            current_segment = []

            def _crystal_label(lbl):
                # CRYSTAL's BAND label records use the single-letter
                # names from its Tables 14.1/14.2 — "GAMMA" is not a
                # recognized label
                return 'G' if lbl.upper() in ('GAMMA', 'Γ') else lbl

            for label in validated_path:
                if label == "|":
                    # End current segment
                    if len(current_segment) >= 2:
                        # Create segments from consecutive points
                        for i in range(len(current_segment) - 1):
                            lines.append(f"{_crystal_label(current_segment[i])} {_crystal_label(current_segment[i+1])}")
                    current_segment = []
                else:
                    current_segment.append(label)

            # Handle final segment
            if len(current_segment) >= 2:
                for i in range(len(current_segment) - 1):
                    lines.append(f"{_crystal_label(current_segment[i])} {_crystal_label(current_segment[i+1])}")
        
        elif config.get("path_method") == "manual":
            # Manual mixed path
            segments = config.get("manual_segments", [])
            n_segments = len(segments)
            shrink = config.get("shrink", 0)
            n_points = config.get("n_points", 1000)
            
            first_band = config.get("first_band", 1) 
            last_band = config.get("last_band")
            if last_band is None:
                last_band = self.structure_info['n_ao'] if self.structure_info['n_ao'] > 0 else 100
            
            plot = 1 if config.get("plot", True) else 0
            print_eig = 1 if config.get("print_eigenvalues", False) else 0
            
            lines.append(f"{n_segments} {shrink} {n_points} {first_band} {last_band} {plot} {print_eig}")
            
            # Add manual segments as-is
            for segment in segments:
                lines.append(segment)
                
        else:
            # Fractional coordinates
            segments = config.get("segments", [])
            n_segments = len(segments)
            shrink = config.get("shrink", 16)
            n_points = config.get("n_points", 1000)
            
            first_band = config.get("first_band", 1) 
            last_band = config.get("last_band")
            if last_band is None:
                last_band = self.structure_info['n_ao'] if self.structure_info['n_ao'] > 0 else 100
            
            plot = 1 if config.get("plot", True) else 0
            print_eig = 1 if config.get("print_eigenvalues", False) else 0
            
            lines.append(f"{n_segments} {shrink} {n_points} {first_band} {last_band} {plot} {print_eig}")
            
            # Add coordinate segments
            for segment in segments:
                lines.append(f"{segment[0]} {segment[1]} {segment[2]}  {segment[3]} {segment[4]} {segment[5]}")
        
        lines.append("END")
        return '\n'.join(lines)
    
    def _newk_mesh(self, d12_file: Path) -> List[str]:
        """The NEWK mesh line(s) for DOSS/TRANSPORT: the parent's SHRINK.

        From the parent .d12 when it is there, else from the mesh its .out
        prints (the same lines), so an .out plus .f9 alone gives the deck the
        .d12 would. Only when neither has a mesh is the dimensionality default
        used, and then it is said.
        """
        dimensionality = self.structure_info.get('dimensionality', 3)
        if d12_file.exists():
            # for_doss=True ensures ISP >= 2*IS
            return [l.strip() for l in extract_shrink_from_d12(
                str(d12_file), for_doss=True, dimensionality=dimensionality)]
        from_out = extract_shrink_from_out(str(self.input_file), for_doss=True,
                                           dimensionality=dimensionality)
        if from_out:
            return [l.strip() for l in from_out]
        is_val = 8 if dimensionality == 3 else 12
        ui.warn(f"Found no .d12 and no SHRINK in {self.input_file.name}; "
                   f"NEWK uses {is_val} {2 * is_val}")
        return [f"{is_val} {2 * is_val}"]

    def _write_doss_d3(self, config: Dict[str, Any]) -> str:
        """Write DOSS calculation D3 file with NEWK and orbital projections."""
        lines = []
        
        # First, we need NEWK
        lines.append("NEWK")
        
        # Extract SHRINK parameters from the corresponding D12 file
        d12_file = self.input_file.with_suffix('.d12')
        if not d12_file.exists():
            # Try to find D12 file in same directory
            d12_files = list(self.input_dir.glob(f"{self.base_name}*.d12"))
            if d12_files:
                d12_file = d12_files[0]
        
        lines.extend(self._newk_mesh(d12_file))
        
        lines.append("1 0")  # IFE=1, IPLO=0
        
        # Now DOSS section
        lines.append("DOSS")
        
        # Debug output
        print()
        ui.info(f"DOSS configuration:")
        ui.info(f"  project_orbital_types: {config.get('project_orbital_types', True)}")
        ui.info(f"  n_atoms: {self.structure_info.get('n_atoms', 0)}")
        ui.info(f"  n_ao: {self.structure_info.get('n_ao', 0)}")

        # A saved or example config may carry only "projection_type" (the
        # menu number) without the explicit keys below. Without this, a
        # "1: Total DOS only" config fell through to the per-orbital default.
        # Explicit keys always win.
        proj_type = config.get("projection_type")
        if proj_type in (1, 2, 3, 4) and "project_orbital_types" not in config:
            config = dict(config)
            config["project_orbital_types"] = proj_type != 1
            if proj_type == 2:
                config.setdefault("element_only", True)
            elif proj_type in (3, 4):
                config.setdefault("element_only", False)
                config.setdefault("include_element_totals", proj_type == 3)

        # Determine projection type
        if config.get("manual_projections"):
            # Manual projections (option 6) - handle first
            manual_specs = config.get("manual_projection_specs", [])
            projections = []
            
            # Parse basis set info from output file for validation
            data_list, n_ao = parse_basis_set_info(str(self.input_file))
            total_shells = {}
            
            if data_list:
                # Get atoms and shells
                atoms, atoms_shells = get_atoms_and_shells(data_list)
                # Generate orbital indices
                total_shells = generate_orbital_indices(atoms, atoms_shells)
            
            # Get atom information
            atom_elements = self.structure_info.get('atom_elements', [])
            
            # Parse each manual specification
            for spec in manual_specs:
                parsed_proj = self._parse_manual_projection(spec, total_shells, atom_elements)
                if parsed_proj:
                    projections.append(parsed_proj)
                else:
                    ui.warn(f"Warning: Skipping invalid projection specification: '{spec}'")

            # Update npro to match actual valid projections
            npro = len(projections)
            if npro == 0:
                ui.err("Error: No valid projections found. Using total DOS only.")
                npro = 0
                projections = []
        elif config.get("project_orbital_types", True):
            # Generate orbital projections by element and type (default behavior)
            # Parse basis set info from output file
            data_list, n_ao = parse_basis_set_info(str(self.input_file))
            
            if data_list:
                # Get atoms and shells
                atoms, atoms_shells = get_atoms_and_shells(data_list)
                
                # Generate orbital indices
                total_shells = generate_orbital_indices(atoms, atoms_shells)
                
                # Create projection lines
                element_only = config.get("element_only", False)
                include_totals = config.get("include_element_totals", True)
                projections = create_doss_projections(total_shells, element_only, include_totals)
                npro = len(projections)
                ui.info(f"  Generated {npro} orbital projections")
            else:
                # Fallback when detailed basis set parsing fails
                ui.warn(f"Warning: Could not parse detailed basis set information from {self.input_file}")
                ui.warn("  Cannot create orbital-resolved projections without basis set info")
                ui.warn("  Using total DOS only")
                npro = 0
                projections = []
        elif config.get("project_all_atoms") or config.get("project_atoms"):
            # Manual atom projections
            atom_elements = self.structure_info.get('atom_elements', [])
            if config.get("project_all_atoms"):
                npro = self.structure_info['n_atoms']
                projections = []
                for i in range(npro):
                    if i < len(atom_elements):
                        element = atom_elements[i]
                        projections.append(f"-1 {i+1}  #{i+1}{element}")
                    else:
                        projections.append(f"-1 {i+1}  #Atom{i+1}")
            else:
                atoms = config.get("project_atoms", [])
                npro = len(atoms)
                projections = []
                for a in atoms:
                    atom_idx = a - 1  # atoms are 1-indexed
                    if atom_idx < len(atom_elements):
                        element = atom_elements[atom_idx]
                        projections.append(f"-1 {a}  #{a}{element}")
                    else:
                        projections.append(f"-1 {a}  #Atom{a}")
        else:
            # Total DOS only (option 1)
            npro = 0
            projections = []
        
        # NPT: a number of points as given, else the energy step's (d3_config.doss_npts)
        config = apply_doss_override(config)
        try:
            from Crystal_d3.d3_interactive import trusted_fermi_au, band_extents
        except ImportError:
            from d3_interactive import trusted_fermi_au, band_extents
        try:
            out_text = Path(self.input_file).read_text(errors="replace")
        except OSError:
            out_text = ""

        # Every band at a step that needs more than DOSS_WINDOW_OFFER_NPTS points: the default depends on
        # the periodic dimensionality, on every route (opt2d3's Enter, --non-interactive, a settings file,
        # a workflow step, Setup); all_bands (opt2d3's No, --doss-window all) keeps every band. 3D and 2D:
        # the window around E_F, its bounds moved into band gaps when a BAND run of the SCF gives the bands;
        # 1D: bands by index, since CRYSTAL needs a window's bounds in gaps (d3_config, DOSS_SNAP_MARGIN_EV)
        dim = self.structure_info.get('dimensionality', 3)
        conduction = doss_conduction_bands(config)
        extents = None
        band_index_default = snap_default = False
        if (not config.get("energy_window") and not config.get("band_range") and not doss_keeps_all_bands(config)
                and doss_explicit_points(config) is None):
            span_all = doss_span_estimate_ev(doss_atomic_numbers(out_text))
            step = doss_step_ev(config)
            offer = doss_window_offer(span_all, doss_npts(span_all, None, step), step)
            counts = doss_band_counts(out_text) if dim == 1 else None
            if dim == 1 and (offer or conduction is not None) and counts is None:
                ui.warn(f"  Warning: {(offer + '. ') if offer else ''}{self.input_file.name} prints no band counts "
                        f"(NUMBER OF AO, N. OF ELECTRONS and CORE ELECTRONS PER CELL): the deck covers all bands.")
            elif dim == 1 and (offer or conduction is not None):
                first, last = doss_band_index_range(counts, conduction)
                config = {**config, "band_range": (first, last)}
                band_index_default = True
                core, occ = counts["core"], counts["occupied"]
                ecp = f", {counts['semicore']} of them ECP semicore" if counts["semicore"] else ""
                ui.warn(f"  Warning: {(offer + ': ') if offer else ''}the deck covers bands {first} to {last} "
                        f"instead (a 1D material: by band index, never by energy, since CRYSTAL needs an energy "
                        f"window's bounds in band gaps): above the {core} core bands{ecp}, to {last - occ} bands "
                        f"above the highest occupied one (band {occ}). --doss-conduction-bands N for more "
                        f"conduction bands, --doss-window all for every band")
            elif offer:
                lo, hi = DOSS_DEFAULT_WINDOW_EV
                ef, why = trusted_fermi_au(str(self.input_file))
                if ef is None:
                    ui.warn(f"  Warning: {offer}. No window around E_F can be written: {why}. The deck "
                            f"covers all bands.")
                else:
                    ui.warn(f"  Warning: {offer}: the deck covers E_F {lo:g} to +{hi:g} eV instead, without the "
                            f"core levels and the high virtual bands (--doss-window all for every band, "
                            f"--doss-window LO HI for other bounds)")
                    config = {**config, "energy_window": doss_window_ha(lo, hi)}
                    snap_default = True
        if conduction is not None and not band_index_default:
            ui.info(f"  --doss-conduction-bands / conduction_bands applies to a 1D material's bands by index; "
                    f"this {dim}D deck does not use it")

        # A window relative to E_F needs a trustworthy E_F: the SCF's, else a BAND or DOSS run of it
        # (never CRYSTAL's -1.0 placeholder of a spin-locked SCF, nor E = 0); without one, all bands
        fermi = None
        if config.get("energy_window") and not config.get("energy_window_absolute"):
            fermi, why = trusted_fermi_au(str(self.input_file))
            if fermi is None:
                ui.warn(f"  Warning: no energy window around E_F: {why}. The deck covers all bands.")
                config = {k: v for k, v in config.items() if k != "energy_window"}
                snap_default = False

        # The window's bounds in band gaps: the default window's moved outward (never narrowed) by a BAND run
        # of this SCF; a given window is written as given, with a warning on a 1D material
        if config.get("energy_window") and fermi is not None and (snap_default or dim == 1):
            extents = band_extents(str(self.input_file))
            lo_ha, hi_ha = config["energy_window"]
            if extents is None and snap_default:
                ui.info("  No BAND run of this SCF beside it: the window's bounds are not checked against "
                        "the bands")
            elif extents is not None:
                new_lo, new_hi, moved = doss_snap_window(fermi + lo_ha, fermi + hi_ha, extents[2], fermi)
                if moved and snap_default:
                    config = {**config, "energy_window": (new_lo - fermi, new_hi - fermi)}
                    what = ("moved outward into band gaps" if (new_lo, new_hi) != (fermi + lo_ha, fermi + hi_ha)
                            else "are inside bands, with no gap near")
                    ui.warn(f"  The window's bounds {what} (a window bound inside a band stopped CRYSTAL23 on a 1D "
                            f"material): {'; '.join(moved)} (the bands from {extents[0].name})")
                elif moved:
                    inside = "; ".join(m.split(": moved")[0].replace(" was inside", " is inside") for m in moved)
                    ui.warn(f"  Warning: {inside} (the bands from {extents[0].name}): a 1D DOSS with a window "
                            f"bound inside a band stopped CRYSTAL23 (double free or corruption); the default (no "
                            f"--doss-window) takes the bands by index")
                elif snap_default:
                    ui.info(f"  The window's bounds are in band gaps (the bands from {extents[0].name})")
            elif dim == 1:
                ui.warn("  Warning: a 1D DOSS with a window bound inside a band stopped CRYSTAL23 (double free or "
                        "corruption); no BAND run of this SCF is beside it to check the bounds. The default "
                        "(no --doss-window) takes the bands by index")

        # the band-index default's span, for its NPT: the BAND run's bands when it is beside the SCF, else
        # a bound (DOSS_BAND_INDEX_SPAN_EV); the DOS reaches as high as its last band goes
        band_span = None
        if band_index_default:
            band_span = DOSS_BAND_INDEX_SPAN_EV
            extents = band_extents(str(self.input_file))
            first, last = config["band_range"]
            if extents is not None and extents[1] <= first and last - extents[1] < len(extents[2]):
                sel = extents[2][first - extents[1]:last - extents[1] + 1]
                band_span = (max(h for _, h in sel) - min(l for l, _ in sel)) * HARTREE_TO_EV + DOSS_SNAP_MAX_EV
                ef, _ = trusted_fermi_au(str(self.input_file))
                if ef is not None:
                    top = (max(h for _, h in sel) - ef) * HARTREE_TO_EV
                    ui.info(f"  Bands {first} to {last} span {band_span - DOSS_SNAP_MAX_EV:.1f} eV and reach "
                            f"E_F{top:+.1f} eV (the bands from {extents[0].name})")
                    if top < DOSS_REACH_EV:
                        ui.warn(f"  Warning: the DOS will stop at E_F{top:+.1f} eV, below E_F+{DOSS_REACH_EV:g} eV: "
                                f"--doss-conduction-bands N for more conduction bands")

        # Handle band/energy range
        if config.get("energy_window"):
            # Energy window specified - negative band indices trigger BMI/BMA input
            first_band = -1
            last_band = -1
            bmi, bma = config["energy_window"]
            # The configured window is relative to the Fermi level (the prompt collects eV
            # below/above E_F), but CRYSTAL's BMI/BMA record holds ABSOLUTE energies in Hartree
            # (manual, DOSS, p.320: INZB and IFNB negative)
            if fermi is not None:
                bmi += fermi
                bma += fermi
                ui.info(f"  Energy window centered on the Fermi level ({fermi:.6f} Ha"
                        + (f"; {why}" if why else "") + ")")
        elif config.get("band_range"):
            # Specific band range. CRYSTAL stops with "BAND RANGE NOT ALLOWED"
            # for band 0 or a band past the last one (measured on HPCC with
            # the example config's 0 999), so clamp to 1..n_ao when known.
            first_band, last_band = config["band_range"]
            first_band = max(int(first_band), 1)
            n_ao = self.structure_info.get('n_ao', 0)
            if n_ao > 0 and int(last_band) > n_ao:
                last_band = n_ao
            bmi, bma = None, None
        else:
            # All bands (default) - use actual band indices
            first_band = 1
            # Use n_ao as approximation for last band if available
            if self.structure_info.get('n_ao', 0) > 0:
                last_band = self.structure_info['n_ao']
            else:
                last_band = 200  # Safe default
            bmi, bma = None, None

        # the span: an energy window's width, else an estimate from the atoms (the run prints the real one)
        try:
            out_text = Path(self.input_file).read_text(errors="replace")
        except OSError:
            out_text = ""
        span_ev, estimated = ((band_span, True) if band_span else doss_span(config, doss_atomic_numbers(out_text)))
        explicit = doss_explicit_points(config)
        n_points = doss_npts(span_ev, explicit, doss_step_ev(config))
        line, coarse = doss_grid_note(span_ev, n_points, explicit is not None, estimated)
        ui.info(f"  Energy grid: {line}")
        if coarse:
            ui.warn(f"  Warning: {coarse}")

        # IPLO values: 0=no output, 1=fort.25, 2=DOSS.DAT
        # Default to 2 (DOSS.DAT) as it's more commonly used
        output_format = config.get("output_format", 2)
        npol = config.get("npol", 14)  # Default from alldos.py
        
        # nprint: 0=no integrated DOS, 1=print integrated DOS
        if config.get("print_integrated", False):
            nprint = 1
        else:
            nprint = 0
        
        # First line after DOSS
        lines.append(f"{npro} {n_points} {first_band} {last_band} {output_format} {npol} {nprint}")
        
        # Add BMI/BMA if using energy window (only when first_band and last_band are negative)
        if first_band < 0 and last_band < 0 and bmi is not None:
            lines.append(f"{bmi} {bma}")
        
        # Add projection specifications
        for proj in projections:
            lines.append(proj)

        # NPR is a count of printing options, and a nonzero NPR must be
        # followed by that many prtrec pairs (manual, DOSS; App. C: 105 =
        # density of states along energy points). Without the record
        # CRYSTAL read END as it and stopped with "FORMAT ERROR".
        if nprint:
            lines.append("105 -1")

        lines.append("END")
        return '\n'.join(lines)
    
    def _write_transport_d3(self, config: Dict[str, Any]) -> str:
        """Write BOLTZTRA transport calculation D3 file."""
        # CRYSTAL requires the wavefunction to be re-diagonalized first:
        # a bare BOLTZTRA deck aborts with "NEWK MUST BE CALLED BEFORE
        # BOLTZTRA" (real case: phase-5 QA, job 12185576). Same NEWK block
        # the DOSS writer emits.
        lines = ["NEWK"]
        d12_file = self.input_file.with_suffix('.d12')
        if not d12_file.exists():
            d12_files = list(self.input_dir.glob(f"{self.base_name}*.d12"))
            if d12_files:
                d12_file = d12_files[0]
        # the k mesh: by a spacing along the periodic reciprocal vectors (d3_config.transport_mesh), never
        # coarser than the parent SCF's, whose mesh converges energies, not the transport distribution
        config = apply_transport_override(config)
        parent = self._newk_mesh(d12_file)
        try:
            parent_is = int(parent[0].split()[0])
        except (IndexError, ValueError):
            parent_is = 0
        if parent_is <= 0:
            lines.extend(parent)
        else:
            n, isp, why = transport_mesh(config, parent_is, reciprocal_lengths(str(self.input_file)))
            lines.append(f"{n} {isp}")
            ui.info(f"  TRANSPORT k mesh: NEWK IS {n} ISP {isp} ({why})")
        lines.append("1 0")  # IFE=1 (Fermi recomputed), IPLO=0

        lines.append("BOLTZTRA")
        
        # Temperature range
        t_min, t_max, t_step = config.get("temperature_range", (100, 800, 50))
        lines.append("TRANGE")
        lines.append(f"{t_min} {t_max} {t_step}")
        
        # Chemical potential range
        mu_min, mu_max, mu_step = config.get("mu_range", (-2.0, 2.0, 0.01))
        lines.append("MURANGE")
        lines.append(f"{mu_min} {mu_max} {mu_step}")
        
        # Transport distribution function range
        tdf_min, tdf_max, tdf_step = config.get("tdf_range", (-5.0, 5.0, 0.01))
        lines.append("TDFRANGE")
        lines.append(f"{tdf_min} {tdf_max} {tdf_step}")
        
        # Optional parameters
        tau = config.get("relaxation_time", 10)
        if tau != 10:
            lines.append("RELAXTIM")
            lines.append(str(tau))
        
        smear = config.get("smearing", 0.0)
        if smear > 0:
            lines.append("SMEAR")
            lines.append(str(smear))
            
            smear_type = config.get("smearing_type", 0)
            if smear_type != 0:
                lines.append("SMEARTYP")
                lines.append(str(smear_type))

        # one END closes the BOLTZTRA block, the second terminates the
        # properties deck — with a single END the run computes everything,
        # then aborts with "END OF DATA IN INPUT DECK" while exiting
        lines.append("END")
        lines.append("END")
        return '\n'.join(lines)

    def _cartesian_atoms(self) -> List[Tuple[float, float, float]]:
        """The atoms of the run's last geometry (the one fort.9 holds) as (x, y, z):
        its last "CARTESIAN COORDINATES - PRIMITIVE CELL" table (Angstrom), else its
        last "ATOMS IN THE ASYMMETRIC UNIT" table, whose non-periodic columns are
        Cartesian Angstrom too (the periodic ones are fractional: not used here)."""
        try:
            text = self.input_file.read_text(errors="replace")
        except OSError:
            return []
        num = r"[-+]?\d+\.\d+(?:[EeDd][-+]?\d+)?"
        tables = (("CARTESIAN COORDINATES - PRIMITIVE CELL", rf"^\s*\d+\s+\d+\s+\S+\s+({num})\s+({num})\s+({num})\s*$"),
                  ("ATOMS IN THE ASYMMETRIC UNIT", rf"^\s*\d+\s+[TF]\s+\d+\s+\S+\s+({num})\s+({num})\s+({num})\s*$"))
        for head, pattern in tables:
            start = text.rfind(head)
            if start < 0:
                continue
            row, atoms = re.compile(pattern), []
            for line in text[start:].splitlines()[1:]:
                m = row.match(line)
                if m:
                    atoms.append(tuple(float(x.replace("D", "E").replace("d", "e")) for x in m.groups()))
                elif atoms:
                    break
            if atoms:
                return atoms
        return []

    def _flat_range_records(self, config: Dict[str, Any]) -> Optional[List[str]]:
        """RANGE records (bohr) for ECH3/POT3 when the atoms are flat along a
        non-periodic direction, where SCALE has no extent to scale (see
        FLAT_EXTENT_A); None when SCALE works (or the geometry cannot be read)."""
        dimensionality = self.structure_info['dimensionality']
        axes = {2: [2], 1: [1, 2], 0: [0, 1, 2]}[dimensionality]
        atoms = self._cartesian_atoms()
        if not atoms:
            return None
        lo = [min(a[k] for a in atoms) for k in axes]
        hi = [max(a[k] for a in atoms) for k in axes]
        flat = [k for k, a, b in zip(axes, lo, hi) if b - a < FLAT_EXTENT_A]
        if not flat:
            return None
        pad = float(config.get("range_pad", FLAT_PAD_A))
        ui.info(f"The atoms are flat along {', '.join('XYZ'[k] for k in flat)}: SCALE has no extent to scale "
                f"there, so the grid uses RANGE (the atoms + {pad:g} A on each side).")
        return [" ".join(f"{(a - pad) * ANGSTROM_TO_BOHR:.4f}" for a in lo),
                " ".join(f"{(b + pad) * ANGSTROM_TO_BOHR:.4f}" for b in hi)]

    def _scale_or_flat_range(self, config: Dict[str, Any]) -> List[str]:
        """The non-periodic grid of an ECH3/POT3 deck without use_range: SCALE
        (ZSCALE / YSCALE,ZSCALE / XSCALE,YSCALE,ZSCALE), or RANGE for flat atoms."""
        flat = self._flat_range_records(config)
        if flat is not None:
            return ["RANGE"] + flat
        scale = int(config.get("scale", 3))
        n = {2: 1, 1: 2, 0: 3}[self.structure_info['dimensionality']]
        return ["SCALE", " ".join([str(scale)] * n)]

    def _range_records(self, config: Dict[str, Any]) -> Optional[List[str]]:
        """The two RANGE records for the non-periodic directions (ECH3/POT3).

        Manual 25494-25500 (ECH3, page 322) and 27341-27347 (POT3, page 351):
        RANGE is followed by all the minima on one record and all the maxima on
        the next - ZMIN/ZMAX for a slab, YMIN,ZMIN / YMAX,ZMAX for a polymer,
        XMIN,YMIN,ZMIN / XMAX,YMAX,ZMAX for a molecule - as a "boundary for
        non-periodic dimensions (au)" (manual 25490), i.e. in bohr.

        The values come from the configuration ("range_min"/"range_max") when it
        carries them, otherwise they are prompted for, which needs a terminal.
        Returns None (after reporting) when neither is possible, so the caller
        can abandon the deck instead of writing a truncated one.
        """
        dimensionality = self.structure_info['dimensionality']
        axes = {2: ["Z"], 1: ["Y", "Z"], 0: ["X", "Y", "Z"]}[dimensionality]

        range_min = config.get("range_min")
        range_max = config.get("range_max")
        if range_min is not None or range_max is not None:
            if range_min is None or range_max is None:
                ui.err("RANGE needs both range_min and range_max in the configuration.")
                return None
            if len(range_min) != len(axes) or len(range_max) != len(axes):
                ui.err(f"RANGE for a {dimensionality}D system takes {len(axes)} value(s) "
                       f"({','.join(axes)}) per record, got {len(range_min)} and {len(range_max)}.")
                return None
            return [" ".join(str(v) for v in range_min),
                    " ".join(str(v) for v in range_max)]

        if not self._can_ask():
            # Non-interactive (batch/workflow callback): fail cleanly instead of
            # raising EOFError at the prompt.
            ui.err("RANGE boundaries requested but there is no terminal to ask for them.")
            ui.err('Supply "range_min" and "range_max" (bohr) in the configuration, or use SCALE.')
            return None

        print()
        ui.info("Define explicit ranges for non-periodic directions:")
        mins = [float(input(f"{axis} min (bohr): ")) for axis in axes]
        maxs = [float(input(f"{axis} max (bohr): ")) for axis in axes]
        return [" ".join(str(v) for v in mins), " ".join(str(v) for v in maxs)]

    def _write_charge_d3(self, config: Dict[str, Any]) -> Optional[str]:
        """Write charge density calculation D3 file.

        Returns None (after reporting) when a required piece of input can be
        neither read from the configuration nor prompted for; the caller must
        then write no deck at all.
        """
        lines = []
        
        if config.get("type") == "ECH3":
            lines.append("ECH3")
            lines.append(str(config.get("n_points", 100)))
            
            # Only for non-3D systems do we need SCALE or RANGE
            if self.structure_info['dimensionality'] < 3:
                if config.get("use_range", False):
                    lines.append("RANGE")
                    records = self._range_records(config)
                    if records is None:
                        return None
                    lines.extend(records)
                else:
                    lines.extend(self._scale_or_flat_range(config))
        
        else:  # ECHG
            lines.append("ECHG")
            lines.append(str(config.get("derivative_order", 0)))
            
            # Need MAPNET input. The records are unconditional for ECHG (manual
            # p. 324 lists "insert MAPNET input records" with no if-clause): a
            # deck of ECHG / IDER / END makes CRYSTAL read END as NPY. So the
            # plane is always asked at a terminal, and a configuration without
            # it (need_map_points false or no map_points) is refused in batch.
            points = config.get("map_points")
            npy = config.get("n_points")

            if points is not None:
                coord_type = _resolve_map_coord_type(config.get("map_coord_type", ""))
                if coord_type is None:
                    ui.err('map_points needs an explicit map_coord_type: "F" (FRACTION)')
                    ui.err('or "C" (ANGSTROM, the MAPNET default).')
                    ui.err("The two differ by the lattice parameter and CRYSTAL cannot")
                    ui.err("detect a plane given in the wrong unit.")
                    return None
                if len(points) != 3 or any(len(p) != 3 for p in points):
                    ui.err(f"MAPNET takes exactly three points A, B, C of three "
                           f"coordinates each; got {len(points)}.")
                    return None
                if npy is None:
                    npy = 50
                    ui.warn(f"No n_points (MAPNET NPY) in the configuration, using {npy}.")
            else:
                if not self._can_ask():
                    # Non-interactive (batch/workflow callback): fail cleanly
                    # instead of raising EOFError at the prompt.
                    ui.err("ECHG needs the MAPNET map plane (manual p. 324: the MAPNET records "
                           "are mandatory) and there is no terminal to ask; no D3 file was written.")
                    ui.err('Supply "map_points" (three points A, B, C) and "map_coord_type" '
                           '("F" or "C") in the configuration (and "n_points", default 50).')
                    return None
                print()
                ui.info("Define map plane by three points A, B, C")
                ui.info("Enter coordinates in fractional (crystal) or Cartesian (Angstrom) units")

                answer = input("Coordinate type (F)ractional or (C)artesian [F]: ").strip() or "F"
                # Up to 1.1.5 a "Define charge density map plane? [Y/n]" came
                # first; a piped answer meant for it lands here. Yes asks
                # again, no is the refusal it now is (there is no deck
                # without the plane).
                if answer.lower() in ("y", "yes"):
                    answer = input("Coordinate type (F)ractional or (C)artesian [F]: ").strip() or "F"
                elif answer.lower() in ("n", "no"):
                    ui.err("ECHG needs the MAPNET map plane (manual p. 324: the MAPNET records "
                           "are mandatory); no D3 file was written.")
                    return None
                coord_type = _resolve_map_coord_type(answer)
                if coord_type is None:
                    ui.err(f"'{answer}' is neither FRACTION nor ANGSTROM; not guessing "
                           "the unit of the map plane.")
                    return None

                # a point is three numbers; anything else (an answer meant for
                # another question, e.g. a 1.1.5-era stream answering the
                # removed plane question with Enter) is refused, never guessed
                points = []
                for point in ["A", "B", "C"]:
                    raw = input(f"Point {point} (x y z): ").strip()
                    try:
                        coords = [float(x) for x in raw.split()]
                    except ValueError:
                        coords = []
                    if len(coords) != 3:
                        ui.err(f"Point {point} is three numbers (x y z), not '{raw}'; no D3 file was written.")
                        return None
                    points.append(coords)

                if npy is None:
                    raw = input("Number of points along the B-A segment [50]: ").strip()
                    try:
                        npy = int(raw or 50)
                    except ValueError:
                        ui.err(f"The number of points is a whole number, not '{raw}'; no D3 file was written.")
                        return None

            # MAPNET records: NPY first, then the unit of measure (it
            # applies to subsequent input), then the keyword declaring how
            # the three points are given. "MAPNET" itself is a dummy
            # keyword and is never written into the deck. nBC is chosen by
            # CRYSTAL so the net is as equally spaced as possible.
            # ANGSTROM is the documented default, so it needs no keyword.
            lines.append(str(npy))
            if coord_type == "FRACTION":
                lines.append("FRACTION")
            lines.append("COORDINA")

            # Points
            for i, point in enumerate(points):
                lines.append(f"{point[0]} {point[1]} {point[2]}")

            # End of the MAPNET input block; the deck's own END follows
            lines.append("END")
        
        lines.append("END")
        return '\n'.join(lines)
    
    def _write_potential_d3(self, config: Dict[str, Any]) -> Optional[str]:
        """Write electrostatic potential calculation D3 file.

        Returns None (after reporting) when a required piece of input can be
        neither read from the configuration nor prompted for; the caller must
        then write no deck at all.
        """
        lines = []
        
        if config.get("type") == "POT3":
            lines.append("POT3")
            lines.append(str(config.get("n_points", 100)))
            lines.append(str(config.get("itol", 5)))
            
            # Only for non-3D systems do we need SCALE or RANGE
            if self.structure_info['dimensionality'] < 3:
                if config.get("use_range", False):
                    lines.append("RANGE")
                    records = self._range_records(config)
                    if records is None:
                        return None
                    lines.extend(records)
                else:
                    lines.extend(self._scale_or_flat_range(config))
        
        else:  # POTC
            lines.append("POTC")
            ica = config.get("ica", 0)
            
            if ica == 0:  # At points
                if config.get("at_atoms", False):
                    npu = 0
                    ipa = 0 if config.get("all_atoms", True) else 1
                else:
                    npu = config.get("n_points", 0)
                    if config.get("read_file", False):
                        # NPU < 0 is the flag that makes CRYSTAL read the point
                        # coordinates from POTC.INP instead of from the deck
                        npu = -abs(npu)
                    ipa = 0
                
                lines.append(f"{ica} {npu} {ipa}")
                
                # Add custom points if needed. A configuration that carries the
                # coordinates implies custom points: the header has already
                # promised NPU records (manual 27382, "if NPU > 0 insert NPU
                # records"), so if none follow CRYSTAL reads the deck's END as
                # an X,Y,Z triple.
                supplied = config.get("points")
                if (config.get("custom_points", False) or supplied) and npu > 0:
                    if supplied is not None:
                        if len(supplied) != npu:
                            ui.err(f"POTC declares NPU={npu} but {len(supplied)} point(s) "
                                   "were supplied.")
                            ui.err("CRYSTAL reads exactly NPU coordinate records, so the "
                                   "deck would be misread.")
                            return None
                        for point in supplied:
                            lines.append(" ".join(str(v) for v in point))
                    else:
                        if not self._can_ask():
                            # Non-interactive (batch/workflow callback): fail
                            # cleanly instead of raising EOFError at the prompt.
                            ui.err(f"POTC needs {npu} point(s) and there is no terminal to ask.")
                            ui.err('Supply "points" (Cartesian, bohr) in the configuration.')
                            return None
                        print()
                        ui.info("Enter point coordinates (Cartesian, bohr):")
                        for i in range(npu):
                            coords = input(f"Point {i+1} (x y z): ").strip()
                            lines.append(coords)
            
            else:  # Plane/volume averaged
                z_min, z_max = config.get("z_range", (0, 10))
                n_planes = config.get("n_planes", 100)
                
                lines.append(f"{ica} {n_planes} 0")
                lines.append(f"{z_min} {z_max}")
                
                if ica == 3:  # Volume averaged - ZD is read only for ICA=3
                    thickness = config.get("slice_thickness", 0.5)
                    lines.append(str(thickness))
        
        lines.append("END")
        return '\n'.join(lines)
    
    def _parent_d12_text(self) -> Optional[str]:
        """The parent's .d12 (same name, else the first <base>*.d12 beside it)."""
        d12_file = self.input_file.with_suffix('.d12')
        if not d12_file.exists():
            d12_files = sorted(self.input_dir.glob(f"{glob.escape(self.base_name)}*.d12"))
            if not d12_files:
                return None
            d12_file = d12_files[0]
        try:
            return d12_file.read_text(errors="replace")
        except OSError:
            return None

    def orbitals_kchoice(self, config: Dict[str, Any]):
        """The NEWK mesh of this output's ORBITALS deck (mace.orbitals.kchoice)."""
        kc = orbitals_kchoice()
        try:
            text = self.input_file.read_text(errors="replace")
        except OSError:
            text = ""
        return kc.choose_kmesh(
            text, config.get("kmesh", "band_edges"),
            dimensionality=self.structure_info.get('dimensionality', 3),
            custom_is=config.get("is"), d12_text=self._parent_d12_text())

    def _write_orbitals_d3(self, config: Dict[str, Any]) -> Optional[str]:
        """Write an ORBITALS deck: the orbitals as Molden files.

        NEWK first (ORBITALS needs it; its mesh is the k points written, no
        mesh line for a molecule), then ORBITALS with the root name
        <deck stem>_orb, ICAR and ILOC each on a line of its own (one line
        "ICAR ILOC" stops properties with FORMAT ERROR IN INPUT DECK, and the
        serial run still exits 0), and one END: ORBITALS has no END of its
        own (measured on CRYSTAL23, HPCC, 0D to 3D).
        """
        why = orbitals_config_refusal(config)
        if why:
            ui.err(why)
            return None
        kc = orbitals_kchoice()
        try:
            choice = self.orbitals_kchoice(config)
        except kc.KChoiceError as e:
            ui.err(f"No ORBITALS deck written: {e}")
            return None
        dimensionality = choice.dimensionality
        ui.info("ORBITALS configuration:")
        if choice.is_ is None:
            ui.info("  Molecule: one Molden file with every molecular orbital")
        else:
            ui.info(f"  k points: {kc.MODE_LABELS.get(choice.mode, choice.mode)}; "
                    f"NEWK IS {choice.is_} ISP {choice.isp}")
            for edge in choice.edges:
                ui.info(f"  {edge.describe()}")
        for note in choice.notes:
            (ui.warn if note.startswith("Warning") else ui.info)(f"  Note: {note}")
        try:
            text = self.input_file.read_text(errors="replace")
        except OSError:
            text = ""
        for warning in orbitals_file_warnings(kc, choice.is_, dimensionality, text):
            ui.warn(f"  Warning: {warning}")

        lines = ["NEWK"]
        lines.extend(choice.newk_lines())
        lines.append("1 0")  # IFE=1 (Fermi level recomputed), NPR=0
        lines.append("ORBITALS")
        lines.append(orbitals_root(f"{self.base_name}_orbitals"))
        lines.append("0" if dimensionality == 0 else "1")  # ICAR (no effect on the files)
        lines.append("0")                                    # ILOC: Bloch orbitals, always
        lines.append("END")
        return '\n'.join(lines)

    def generate_d3(self, shared_config: Optional[Dict[str, Any]] = None,
                    answers_on_stdin: Optional[bool] = None) -> Optional[Dict[str, Any]]:
        """Generate the D3 file based on calculation type.
        
        Returns:
            Configuration dictionary if successful, None otherwise
        """
        # With no settings given they are asked on stdin, and so are the
        # writers' own questions; answers piped in that run out end the
        # run with no deck instead of an EOFError traceback. Settings shared
        # by a multi-file run were themselves answered on stdin, so each
        # file's own questions are too (answers_on_stdin=True).
        self._answers_on_stdin = (not shared_config if answers_on_stdin is None
                                  else answers_on_stdin)
        try:
            return self._generate_d3(shared_config)
        except EOFError:
            print()
            ui.err("The input ended before every question was answered "
                   "(end of input); no D3 file was written.")
            return None
        except RuntimeError as e:
            # A 3D SeeK-path path without seekpath (d3_kpoints.SeekpathRequired,
            # matched by name: the module can be imported under two names)
            if type(e).__name__ != "SeekpathRequired":
                raise
            print()
            ui.err(str(e))
            return None
        finally:
            self._answers_on_stdin = False

    def _generate_d3(self, shared_config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        print()
        ui.rule(f"Generating {self.calc_type} D3 file")
        ui.info(f"Input file: {self.input_file}")
        ui.info(f"Base name: {self.base_name}")

        # A molecule first: BAND/DOSS/TRANSPORT need a periodic structure,
        # whatever else the parent is (so no 2c "supported" line precedes it).
        dimensionality = self.structure_info.get('dimensionality', 3)
        why = molecule_refusal(self.calc_type, dimensionality)
        if why:
            ui.err(why)
            return None

        # A two-component (SOC) parent: only what manual sec. 6.4 supports.
        if self.parent_is_two_component():
            if self.calc_type == "ORBITALS":
                # mace.orbitals.kchoice.SOC_REFUSAL: CRYSTAL23 crashes on it
                ui.err(f"Not writing an ORBITALS deck: {self.input_file.name} is a two-component "
                       f"run. {orbitals_kchoice().SOC_REFUSAL}")
                return None
            if self.calc_type not in TWO_COMPONENT_D3_TYPES:
                record = _TWO_COMPONENT_REFUSED.get(self.calc_type, self.calc_type)
                ui.err(f"Not writing a {self.calc_type} deck: {self.input_file.name} is a "
                       f"two-component (SOC) run, and {record} is not among the properties "
                       f"CRYSTAL23 supports from a 2c-SCF (manual sec. 6.4, p. 183: NEWK, "
                       f"BAND, ECHG, ECH3, PPAN, PROPS2COMP, ANBD, BWIDTH, DOSS, TOPO).")
                return None
            if dimensionality == 0 and self.calc_type in TWO_COMPONENT_0D_REFUSED:
                ui.err(f"Not writing a {self.calc_type} deck: {self.input_file.name} is a "
                       f"two-component (SOC) molecule. {TWO_COMPONENT_0D_CUBE_REASON}")
                return None
            ui.info(f"The parent is a two-component (SOC) run; {self.calc_type} is among "
                    f"the properties the manual supports from it (sec. 6.4, p. 183).")
            # Measured on HPCC with the stock serial properties on a 2c fort.9.
            if self.calc_type == "BAND":
                ui.info("Measured: the stock properties runs BAND on a 2c fort.9 and writes "
                        "all 2 x NAO spinor bands to BAND.DAT (manual p. 166); a deck "
                        "asking for bands 1-NAO gave all 2 x NAO.")
            elif self.calc_type == "DOSS":
                ui.warn("Note: the stock properties runs DOSS on a 2c fort.9 once the DOSS "
                        "record is valid (measured). A 2c run has 2 x NAO spinor bands "
                        "(manual p. 166); which of them a band range MACE takes from NAO "
                        "selects was not measured, so check the band range in the deck.")
            elif self.calc_type == "CHARGE":
                ui.info("Measured: the stock properties runs ECH3 on a periodic 2c fort.9 "
                        "(DENS_CUBE.DAT written); ECHG was not run from one.")

        # Check if wavefunction exists and copy it
        # ALL calculation types need the wavefunction file
        if not self._copy_wavefunction():
            return None
        
        # Configure calculation - use shared config if provided
        if shared_config:
            config = shared_config.copy()  # Make a copy to allow modifications
            
            # For BAND calculations with automatic path, recalculate path for each material
            if self.calc_type == "BAND" and config.get("auto_path", False):
                # Re-extract space group for this specific material
                space_group = self.structure_info.get('space_group', 1)
                lattice_type = self.structure_info.get('lattice_type', 'P')
                # The default path (Enter, the workflow's default) is SeeK-path for a
                # 3D crystal only; a slab or polymer takes the label path
                # (d3_kpoints.default_band_format). A chosen format 4 is kept.
                if (config.get("kpath_default") and config.get("seekpath_full")
                        and self.structure_info.get('dimensionality', 3) in (1, 2)):
                    from d3_kpoints import default_label_band_config
                    config = default_label_band_config(config)
                    ui.info("  A slab or polymer: the default band path is the high-symmetry label path")
                
                if config.get("path_method") == "labels":
                    # Get appropriate band path for THIS material's symmetry
                    from d3_kpoints import get_band_path_from_symmetry
                    config["path"] = get_band_path_from_symmetry(space_group, lattice_type)
                    config["kpath_source"] = "default"  # Label-based paths use default source
                    ui.info(f"  Using band path for space group {space_group}: {' → '.join(config['path'])}")
                elif config.get("path_method") == "coordinates":
                    # Convert labels to coordinates for THIS material's symmetry
                    from d3_kpoints import (get_band_path_from_symmetry, get_kpoint_coordinates_from_labels,
                                           get_seekpath_full_kpath, get_literature_kpath_vectors)
                    
                    if config.get("seekpath_full", False):
                        # Use SeeK-path full path
                        result = get_seekpath_full_kpath(space_group, lattice_type, str(self.input_file))
                        if result:
                            frac_segments, kpath_info = result

                            # For seekpath library results, use the library's calculated shrink factor
                            # This ensures exact integer representation of parametric k-points
                            if kpath_info.get('source') in ('seekpath_library', 'seekpath_data_parametric') and 'shrink_factor' in kpath_info:
                                shrink = kpath_info['shrink_factor']
                                ui.info(f"    Using seekpath library's shrink factor: {shrink}")
                            else:
                                # Fall back to D12's shrink factor
                                shrink = extract_and_process_shrink(str(self.input_file), self.base_name,
                                                                   self.input_dir, config)

                            # Scale fractional coordinates
                            coord_segments, adjusted_shrink = scale_kpoint_segments(frac_segments, shrink)
                            config["segments"] = coord_segments
                            config["shrink"] = adjusted_shrink
                            # Store SeeK-path labels
                            config["path_labels"] = get_seekpath_labels(space_group, lattice_type, str(self.input_file))
                            # Store k-path source info
                            if kpath_info.get("source") == "literature":
                                config["kpath_source"] = "literature"
                            elif kpath_info.get("source") == "default":
                                config["kpath_source"] = "default"
                            elif kpath_info.get("has_inversion"):
                                config["kpath_source"] = "seekpath_inv"
                            else:
                                config["kpath_source"] = "seekpath_noinv"
                            ui.info(f"  Using SeeK-path full k-path for space group {space_group} with shrink={adjusted_shrink}")
                        else:
                            # Fallback - try literature path first
                            frac_segments = get_literature_kpath_vectors(space_group, lattice_type)
                            if frac_segments:
                                # Use literature path
                                shrink = extract_and_process_shrink(str(self.input_file), self.base_name,
                                                                   self.input_dir, config)
                                coord_segments, adjusted_shrink = scale_kpoint_segments(frac_segments, shrink)
                                config["segments"] = coord_segments
                                config["shrink"] = adjusted_shrink
                                config["path_labels"] = get_band_path_from_symmetry(space_group, lattice_type)
                                config["kpath_source"] = "literature"
                                ui.info(f"  SeeK-path not available, using literature k-path for space group {space_group}")
                            else:
                                # Final fallback to standard path
                                path_labels = get_band_path_from_symmetry(space_group, lattice_type)
                                frac_segments = get_kpoint_coordinates_from_labels(path_labels, space_group, lattice_type)
                                # Segment extremes are expressed in units of 1/ISS, so the
                                # fractional vectors must be scaled by the shrink factor
                                shrink = extract_and_process_shrink(str(self.input_file), self.base_name,
                                                                   self.input_dir, config)
                                coord_segments, adjusted_shrink = scale_kpoint_segments(frac_segments, shrink)
                                config["segments"] = coord_segments
                                config["shrink"] = adjusted_shrink
                                config["path_labels"] = path_labels  # Store labels for title
                                config["kpath_source"] = "default"
                                ui.info(f"  SeeK-path not available, using standard k-point vectors for space group {space_group}")
                    
                    elif config.get("literature_path", False):
                        # Use literature path ("default" when there is none)
                        frac_segments = get_literature_kpath_vectors(space_group, lattice_type)
                        config["kpath_source"] = "literature" if frac_segments else "default"
                        if frac_segments:
                            # Extract and process shrink factor
                            shrink = extract_and_process_shrink(str(self.input_file), self.base_name, 
                                                               self.input_dir, config)
                            
                            # Scale fractional coordinates
                            coord_segments, adjusted_shrink = scale_kpoint_segments(frac_segments, shrink)
                            config["segments"] = coord_segments
                            config["shrink"] = adjusted_shrink
                            # Store path labels (literature paths use standard labels for now)
                            config["path_labels"] = get_band_path_from_symmetry(space_group, lattice_type)
                            ui.info(f"  Using literature k-path for space group {space_group} with shrink={adjusted_shrink}")
                        else:
                            # Fallback
                            path_labels = get_band_path_from_symmetry(space_group, lattice_type)
                            frac_segments = get_kpoint_coordinates_from_labels(path_labels, space_group, lattice_type)
                            # Segment extremes are expressed in units of 1/ISS, so the
                            # fractional vectors must be scaled by the shrink factor
                            shrink = extract_and_process_shrink(str(self.input_file), self.base_name,
                                                               self.input_dir, config)
                            coord_segments, adjusted_shrink = scale_kpoint_segments(frac_segments, shrink)
                            config["segments"] = coord_segments
                            config["shrink"] = adjusted_shrink
                            config["path_labels"] = path_labels  # Store labels for title
                            ui.info(f"  Literature path not available, using standard k-point vectors for space group {space_group}")
                    
                    else:
                        # Standard path - need to scale coordinates
                        config["kpath_source"] = "default"
                        path_labels = get_band_path_from_symmetry(space_group, lattice_type)
                        frac_segments = get_kpoint_coordinates_from_labels(path_labels, space_group, lattice_type)
                        
                        # Extract and process shrink factor
                        shrink = extract_and_process_shrink(str(self.input_file), self.base_name, 
                                                           self.input_dir, config)
                        
                        # Scale fractional coordinates by shrink factor
                        coord_segments, adjusted_shrink = scale_kpoint_segments(frac_segments, shrink)

                        config["segments"] = coord_segments
                        config["path_labels"] = path_labels  # Store labels for title
                        config["shrink"] = adjusted_shrink
                        ui.info(f"  Using k-point vectors for space group {space_group} with shrink={adjusted_shrink}")

            # For BAND calculations with bands around Fermi, recalculate for each material
            if self.calc_type == "BAND" and config.get("bands") == "fermi":
                # Extract band info for this material
                try:
                    from Crystal_d3.d3_interactive import get_band_info_from_output
                except ImportError:
                    from d3_interactive import get_band_info_from_output
                band_info = get_band_info_from_output(str(self.input_file))

                if band_info.get('valence_bands', 0) > 0:
                    # Get offsets from config
                    bands_below = config.get("bands_below_fermi", 500)
                    bands_above = config.get("bands_above_fermi", 600)

                    # Calculate actual band numbers for THIS material
                    first = max(1, band_info['valence_bands'] - bands_below)
                    last = min(band_info['n_bands'], band_info['valence_bands'] + bands_above)

                    config["first_band"] = first
                    config["last_band"] = last
                    ui.info(f"  Material has {band_info['valence_bands']} valence bands (Fermi level)")
                    ui.info(f"  Selected bands {first} to {last} ({bands_below} below + {bands_above} above Fermi)")
                else:
                    # Fallback to all bands
                    ui.warn("  Warning: Could not determine Fermi level, using all bands")
                    config["first_band"] = 1
                    config["last_band"] = None

            # TRANSPORT with μ relative to E_F (the default; "vbm": the top of the occupied bands, which
            # is the E_F an insulator's sources give): MURANGE holds absolute eV, so it needs a
            # trustworthy E_F for this material, never CRYSTAL's -1.0 placeholder or E = 0
            if self.calc_type == "TRANSPORT" and config.get("mu_reference") in ("fermi", "vbm"):
                try:
                    from Crystal_d3.d3_interactive import trusted_fermi_au
                except ImportError:
                    from d3_interactive import trusted_fermi_au
                fermi_au, why = trusted_fermi_au(str(self.input_file))
                if fermi_au is None:
                    ui.err(f"  Error: no TRANSPORT deck for {self.input_file.name}: its chemical potentials are "
                           f"relative to E_F, and there is {why[0].lower() + why[1:] if why else 'none'}. "
                           f"Give an absolute range (mu_reference \"absolute\", mu_range in eV) or run BAND or "
                           f"DOSS from this SCF first.")
                    return None
                fermi_energy_ev = fermi_au * HARTREE_TO_EV
                if config.get("mu_range_type") == "auto_fermi" or "mu_relative_range" not in config:
                    # a saved (cleaned) configuration holds the range relative to E_F
                    mu_min_rel, mu_max_rel, mu_step = config.get("mu_range", (-2.0, 2.0, 0.01))
                else:
                    mu_min_rel, mu_max_rel = config["mu_relative_range"]
                    mu_step = config.get("mu_range", (0, 0, 0.01))[2]
                mu_min_abs = fermi_energy_ev + mu_min_rel
                mu_max_abs = fermi_energy_ev + mu_max_rel
                config["mu_range"] = (mu_min_abs, mu_max_abs, mu_step)
                # TDFRANGE is absolute eV too (the TDF.DAT's energies): the transport distribution must
                # cover the chemical potentials, so its range is relative to E_F as theirs is
                if "tdf_relative_range" in config:
                    tdf_lo, tdf_hi = config["tdf_relative_range"]
                    tdf_step = config.get("tdf_range", (0, 0, 0.01))[2]
                else:
                    tdf_lo, tdf_hi, tdf_step = config.get("tdf_range", (-5.0, 5.0, 0.01))
                config["tdf_range"] = (fermi_energy_ev + tdf_lo, fermi_energy_ev + tdf_hi, tdf_step)
                ui.info(f"  Using Fermi energy {fermi_energy_ev:.3f} eV" + (f" ({why})" if why else ""))
                ui.info(f"  Chemical potential range: {mu_min_abs:.3f} to {mu_max_abs:.3f} eV; transport "
                        f"distribution {config['tdf_range'][0]:.3f} to {config['tdf_range'][1]:.3f} eV")
        else:
            config = configure_d3_calculation(self.calc_type, str(self.input_file))
        
        # Generate D3 content
        if self.calc_type == "BAND":
            if not shared_config and self._can_ask() and not config.get("band_path") and _kpath_question_wanted():
                config = self._offer_band_path_edit(config)
            d3_content = self._write_band_d3(config)
            if d3_content is None:
                return None
        elif self.calc_type == "DOSS":
            d3_content = self._write_doss_d3(config)
        elif self.calc_type == "TRANSPORT":
            d3_content = self._write_transport_d3(config)
        elif self.calc_type == "ORBITALS":
            d3_content = self._write_orbitals_d3(config)
        elif self.calc_type in ["CHARGE", "ECH3", "ECHG"]:
            d3_content = self._write_charge_d3(config)
        elif self.calc_type in ["POTENTIAL", "POT3", "POTC"]:
            d3_content = self._write_potential_d3(config)
        elif self.calc_type == "CHARGE+POTENTIAL":
            # Combined ECH3+POT3 calculation
            # The config already contains charge_config and potential_config from configure_d3_calculation
            if "charge_config" in config and "potential_config" in config:
                # Using config (either from shared_config or just configured)
                charge_config = config["charge_config"]
                potential_config = config["potential_config"]
            else:
                # This shouldn't happen if configure_d3_calculation worked correctly
                ui.err("Error: CHARGE+POTENTIAL configuration missing charge_config or potential_config")
                return None
            
            # Write combined file - drop the charge block's terminating END so
            # the combined deck carries exactly one. This must be a SUFFIX
            # strip: rstrip('\nEND') strips the character SET {\n,E,N,D} and
            # would eat into a data line ending in any of those characters.
            charge_block = self._write_charge_d3(charge_config)
            if charge_block is None:
                return None
            potential_block = self._write_potential_d3(potential_config)
            if potential_block is None:
                return None
            if charge_block.endswith("\nEND"):
                charge_block = charge_block[:-len("\nEND")]
            d3_content = charge_block
            d3_content += "\n" + potential_block

        else:
            ui.err(f"Error: Unknown calculation type {self.calc_type}")
            return False

        # A writer returns None when it could not obtain a required record
        # (map plane, POTC points, RANGE boundaries). Write nothing at all: a
        # truncated deck is worse than a missing one, because the caller checks
        # for the file.
        if d3_content is None:
            ui.err("No D3 file written (incomplete configuration).")
            return None

        # Write D3 file
        d3_filename = f"{self.base_name}_{self.calc_type.lower()}.d3"
        d3_path = self.output_dir / d3_filename
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(d3_path, 'w') as f:
            f.write(d3_content)
        
        print()
        ui.ok(f"D3 file written: {d3_path}")
        self._place_wavefunction()

        # Additional instructions
        print()
        ui.info("To run the calculation:")
        ui.info(f"1. Make sure the wavefunction file is present: {self.base_name}_{self.calc_type.lower()}.f9")
        ui.info(f"2. Submit the job with the D3 file: {d3_filename}")

        if self.calc_type == "BAND":
            print()
            ui.info("Output files:")
            ui.info("  - BAND.DAT: Band structure data")
            ui.info("  - fort.25: Band structure for plotting")
        elif self.calc_type == "DOSS":
            print()
            ui.info("Output files:")
            ui.info("  - DOSS.DAT: Density of states data")
            ui.info("  - fort.25: DOS for plotting (if requested)")
        elif self.calc_type == "TRANSPORT":
            print()
            ui.info("Output files:")
            ui.info("  - SIGMA.DAT: Electrical conductivity")
            ui.info("  - SEEBECK.DAT: Seebeck coefficient")
            ui.info("  - SIGMAS.DAT: σS product")
            ui.info("  - KAPPA.DAT: Thermal conductivity")
            ui.info("  - TDF.DAT: Transport distribution function")
        elif self.calc_type == "ORBITALS":
            root = orbitals_root(f"{self.base_name}_orbitals")
            print()
            ui.info("Output files:")
            if self.structure_info.get('dimensionality', 3) == 0:
                ui.info(f"  - {root}.molden: every molecular orbital")
            else:
                ui.info(f"  - {root}_<k>_K<ijk>_real.molden (and _complex.molden): "
                        f"every band at each k point")
        
        # Return the configuration for potential saving
        return config


# Properties CRYSTAL computes for periodic systems only. CRYSTAL23 manual: BAND "does not run
# for molecules!! (0D)" (p.309), DOSS "do not run for molecules!" (p.320); BOLTZTRA's transport
# coefficients come from the band structure. ECH3/POT3 (CHARGE, POTENTIAL) work for molecules.
PERIODIC_ONLY = {
    "BAND": "a molecule has no band structure (CRYSTAL23 manual, BAND: \"does not run for molecules\")",
    "DOSS": "CRYSTAL computes no density of states for a molecule (CRYSTAL23 manual, DOSS: \"do not run for molecules\")",
    "TRANSPORT": "BOLTZTRA's transport coefficients come from the band structure, which a molecule does not have",
}


def molecule_refusal(calc_type: str, dimensionality: int) -> Optional[str]:
    """Why ``calc_type`` cannot be set up from a molecule (dimensionality 0), or None."""
    if dimensionality != 0 or calc_type not in PERIODIC_ONLY:
        return None
    return (f"{calc_type} requires a periodic structure, and this output is a molecule (MOLECULE): "
            f"{PERIODIC_ONLY[calc_type]}. No D3 file was written; CHARGE and POTENTIAL work for molecules.")


def orbitals_config_refusal(config: Dict[str, Any]) -> Optional[str]:
    """Why an ORBITALS configuration cannot be written, or None.

    ILOC 1 (Wannier functions from LOCALWF) is refused: those files are not
    usable (the coefficients are normalized neither as home-cell nor as
    Γ-folded functions, measured on Si and polyyne), and Pproperties aborts on
    it. Wannier functions go through the Wannier90 / lcao2wannier route.
    """
    iloc = config.get("iloc", config.get("ILOC", 0))
    if str(iloc).strip() not in ("0", "", "None", "False") or config.get("wannier"):
        kc = orbitals_kchoice()
        return (f"ORBITALS with ILOC {iloc} is not written: {kc.WANNIER_ROUTE}. "
                f"MACE writes ILOC 0 (Bloch orbitals) only.")
    return None


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""\
Generate CRYSTAL D3 property calculation input files.

The CRYSTAL output is given with --input/-i (or every .out in the current
directory with --batch). There is no positional argument.""",
        epilog="""\
Examples:
  %(prog)s --input optimized.out --calc-type BAND
  %(prog)s -i optimized.out -t DOSS
  %(prog)s --batch --calc-type BAND""",
    )
    
    parser.add_argument(
        "--input", "-i",
        help="CRYSTAL output file (.out or .log)"
    )
    parser.add_argument(
        "--calc-type", "-t", "--calc_type",
        dest="calc_type",
        choices=["BAND", "DOSS", "TRANSPORT", "CHARGE", "POTENTIAL", "CHARGE+POTENTIAL", "ORBITALS"],
        help="Type of property calculation"
    )
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="Process all .out files in current directory"
    )
    parser.add_argument(
        "--shared-settings", "-s",
        action="store_true",
        help="Use shared settings for all files in batch mode"
    )
    parser.add_argument(
        "--config-file", "-c",
        help="Load D3 configuration from JSON file"
    )
    parser.add_argument(
        "--save-config",
        action="store_true",
        help="Save configuration to JSON file after interactive setup"
    )
    parser.add_argument(
        "--options-file",
        type=str,
        default=None,
        help="Filename for saving configuration (suppresses interactive prompt)"
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available D3 configuration files"
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="Output directory for generated D3 files (default: same as input file)"
    )
    parser.add_argument(
        "--static-kpath",
        action="store_true",
        help="Write the static HPKOT table path for a 3D SeeK-path BAND path when "
             "seekpath is not installed (SeeK-path's standard cell, which can differ "
             "from CRYSTAL's); without it such a deck stops. Same as MACE_STATIC_KPATH=1"
    )
    parser.add_argument(
        "--edit-kpath",
        action="store_true",
        help="BAND: ask \"Edit the band k-path?\" when the answers are piped in too "
             "(it is asked only at a terminal otherwise)"
    )
    parser.add_argument(
        "--doss-step",
        metavar="EV|PRESET",
        help="DOSS: the energy step of the DOS grid in eV, or screening (0.2), standard (0.02, the "
             "default) or fine (0.005); the deck gets ceil(span / step) points, 1000-20000"
    )
    parser.add_argument(
        "--doss-points",
        type=int,
        metavar="N",
        help="DOSS: this many energy points, as given (e.g. 1000 to screen many structures)"
    )
    parser.add_argument(
        "--doss-window",
        nargs="+",
        metavar="LO HI|all",
        help="DOSS: the energies from LO to HI eV around E_F (e.g. -20 15, the default of a 3D or 2D material "
             "when every band would need more than 5000 points, its bounds moved outward into band gaps when a "
             "BAND run of the SCF is beside it; needs a trustworthy E_F: the SCF's, else a BAND or DOSS run of "
             "it), or all: every band. A 1D material's default takes the bands by index instead"
    )
    parser.add_argument(
        "--doss-conduction-bands",
        type=int,
        metavar="N",
        help="DOSS: a 1D material's bands by index (its default) run to N bands above the highest occupied "
             "one (default: as many as the valence bands, at least the minimal valence shells' bands and 4)"
    )
    parser.add_argument(
        "--transport-k",
        metavar="SPACING|PRESET",
        help="TRANSPORT: the k-point spacing of BOLTZTRA's NEWK mesh in 1/Å (2*pi included), or screening (the "
             "parent SCF's mesh), standard (the default) or fine; never coarser than the parent's mesh"
    )
    parser.add_argument(
        "--transport-is",
        type=int,
        metavar="IS",
        help="TRANSPORT: this NEWK shrinking factor, as given (ISP = 2 IS)"
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Answer every question with its default (Enter)"
    )

    args = parser.parse_args()
    if args.doss_step is not None and args.doss_points is not None:
        parser.error("give --doss-step or --doss-points, not both")
    if args.doss_points is not None and not 2 <= args.doss_points <= DOSS_MAX_NPTS:
        parser.error(f"--doss-points takes 2 to {DOSS_MAX_NPTS} points (DOSS.DAT's header prints NPT + 2 "
                     "in five digits)")
    if args.doss_step is not None:
        try:
            if str(args.doss_step).strip().lower() not in DOSS_STEP_PRESETS and float(args.doss_step) <= 0:
                raise ValueError
        except ValueError:
            parser.error(f"--doss-step takes a step in eV larger than 0 or one of {', '.join(DOSS_STEP_PRESETS)}")
    if args.doss_window is not None:
        if [w.lower() for w in args.doss_window] == ["all"]:
            args.doss_window = ["all"]
        else:
            try:
                args.doss_window = [float(w) for w in args.doss_window]
            except ValueError:
                args.doss_window = None
            if not args.doss_window or len(args.doss_window) != 2 or not args.doss_window[0] < 0 < args.doss_window[1]:
                parser.error("--doss-window takes LO < 0 < HI, in eV around E_F (e.g. -20 15), or all")
    if args.doss_conduction_bands is not None and args.doss_conduction_bands < 1:
        parser.error("--doss-conduction-bands takes a number of bands, at least 1")
    set_doss_override(args.doss_step, args.doss_points, args.doss_window, args.doss_conduction_bands)
    if args.transport_k is not None and args.transport_is is not None:
        parser.error("give --transport-k or --transport-is, not both")
    if args.transport_is is not None and args.transport_is < 1:
        parser.error("--transport-is takes a shrinking factor of at least 1")
    if args.transport_k is not None:
        try:
            if str(args.transport_k).strip().lower() not in TRANSPORT_K_PRESETS and float(args.transport_k) <= 0:
                raise ValueError
        except ValueError:
            parser.error(f"--transport-k takes a spacing in 1/Å larger than 0 or one of "
                         f"{', '.join(TRANSPORT_K_PRESETS)}")
    set_transport_override(args.transport_k, args.transport_is)
    if args.non_interactive:
        set_non_interactive(True)
        # every question gets the empty answer (its default); a question without one stops the run
        import io
        sys.stdin = io.StringIO("\n" * 500)
    if args.static_kpath:
        request_static_kpath()
    if args.edit_kpath:
        request_kpath_question()
    
    # Handle --list-configs
    if args.list_configs:
        configs = list_available_d3_configs()
        if configs:
            print()
            ui.info("Available D3 configuration files:")
            for config in configs:
                ui.info(f"  - {config}")
        else:
            print()
            ui.info("No D3 configuration files found in current directory.")
        return
    
    # If no input file and not batch mode, check if there are .out files in current dir
    if not args.input and not args.batch:
        out_files = list(Path.cwd().glob("*.out"))
        if out_files:
            # Automatically switch to batch mode if .out files found
            args.batch = True
            print()
            ui.info(f"Found {len(out_files)} output file(s) in current directory:")
            for f in sorted(out_files)[:5]:  # Show first 5
                ui.info(f"  - {f.name}")
            if len(out_files) > 5:
                ui.info(f"  ... and {len(out_files) - 5} more")
    
    if args.batch:
        # Batch mode
        if hasattr(args, 'batch_dir') and args.batch_dir:
            # Use the directory specified by the user
            out_files = list(args.batch_dir.glob("*.out"))
            batch_dir = args.batch_dir
        else:
            # Use current working directory
            out_files = list(Path.cwd().glob("*.out"))
            batch_dir = Path.cwd()
            
        if not out_files:
            ui.info(f"No .out files found in {batch_dir}")
            return

        ui.info(f"Found {len(out_files)} output files")

        # A config file's calculation_type answers the type question: load it
        # first (it was loaded after the question, so a batch run with nothing
        # on stdin hit the menu and the config's type was never used).
        loaded_config = load_d3_config(args.config_file) if args.config_file else None
        config_calc_type = (loaded_config or {}).get("calculation_type")

        # Get calculation type
        if not args.calc_type and not config_calc_type:
            print()
            ui.info("Select calculation type for all files:")
            ui.info("1: BAND - Electronic band structure")
            ui.info("2: DOSS - Density of states")
            ui.info("3: TRANSPORT - Transport properties")
            ui.info("4: CHARGE - Charge density")
            ui.info("5: POTENTIAL - Electrostatic potential")
            ui.info("6: CHARGE+POTENTIAL - Combined calculation")
            ui.info("7: ORBITALS - Orbitals as Molden files")

            calc_types = {
                "1": "BAND", "2": "DOSS", "3": "TRANSPORT",
                "4": "CHARGE", "5": "POTENTIAL", "6": "CHARGE+POTENTIAL",
                "7": "ORBITALS"
            }
            try:
                choice = input("\nSelect type (1-7): ").strip()
                while choice not in calc_types:
                    # Re-prompt instead of silently defaulting a typo to BAND, which
                    # would generate the wrong property calculation with no warning
                    # (same contract as the single-file path).
                    ui.warn(f"  '{choice}' is not a valid choice; enter a number 1-7.")
                    choice = input("Select type (1-7): ").strip()
            except EOFError:
                print()
                ui.err("No calculation type: give --calc-type, or a config file with "
                       "calculation_type. No D3 file was written.")
                sys.exit(1)
            calc_type = calc_types[choice]
        else:
            calc_type = args.calc_type or config_calc_type
        
        # Ask about shared settings if not specified and more than one file
        # (a config file is shared by every file already)
        if not args.shared_settings and len(out_files) > 1 and not args.config_file:
            print()
            ui.info("Multiple files detected. Use shared settings for all files?")
            use_shared = yes_no_prompt("Use same configuration for all files?", "yes")
            args.shared_settings = use_shared
        
        # Configure shared settings if requested
        shared_config = None
        
        # Load from config file if specified
        if args.config_file:
            shared_config = loaded_config
            if shared_config:
                print_d3_config_summary(shared_config)
                # Validate configuration
                is_valid, errors = validate_d3_config(shared_config)
                if not is_valid:
                    ui.err("Configuration validation errors:")
                    for error in errors:
                        ui.err(f"  - {error}")
                    # exit non-zero: callers (workflow engine) detect failure
                    # by exit code / missing output, not by parsing stderr
                    sys.exit(1)
                # Override calc_type from config if present
                if "calculation_type" in shared_config:
                    calc_type = shared_config["calculation_type"]
            else:
                ui.err(f"Failed to load configuration from {args.config_file}")
                sys.exit(1)
        elif args.shared_settings:
            print()
            ui.rule("Configuring shared settings for all files")
            # Use first file as reference for getting structure info
            first_file = str(out_files[0])
            # Shared-settings config is gathered BEFORE any per-file mkdir/copy, so it is
            # safe to run under the back controller ("press b to go back"). Per-file mode
            # (generate_d3) is left unwrapped: its writers' questions come after
            # _copy_wavefunction's checks, which must not be replayed.
            try:
                shared_config = _run_with_back(lambda: configure_d3_calculation(calc_type, first_file))
            except RuntimeError as e:
                if type(e).__name__ != "SeekpathRequired":
                    raise
                print()
                ui.err(str(e))
                sys.exit(1)
            print()
            ui.ok("Shared settings configured")

            # Option to save configuration
            if args.save_config or yes_no_prompt("\nSave this configuration for future use?", "no"):
                if args.options_file:
                    # Use specified filename without prompting
                    save_d3_config(shared_config, args.options_file)
                    ui.ok(f"Configuration saved to {args.options_file}")
                else:
                    save_d3_options_prompt(shared_config, skip_prompt=True)
        
        # Process each file. generate_d3 returns None when no deck was written
        # (missing wavefunction, incomplete configuration); the loop must not
        # report success for those files.
        failed = []
        for out_file in out_files:
            print()
            ui.rule(f"Processing: {out_file}")
            generator = D3Generator(str(out_file), calc_type, args.output_dir)
            if generator.generate_d3(shared_config) is None:
                failed.append(out_file)

        if failed:
            print()
            ui.err(f"No D3 file generated for {len(failed)} of {len(out_files)} file(s):")
            for failed_file in failed:
                ui.err(f"  - {failed_file}")
            if len(failed) == len(out_files):
                # Nothing at all was produced: exit non-zero so callers that
                # gate on the return code see the failure.
                sys.exit(1)

    else:
        # Interactive mode
        if args.input:
            input_file = args.input
        else:
            input_file = input("Enter CRYSTAL output file path: ").strip()
        
        input_path = Path(input_file)
        
        # Check if the input exists
        if not input_path.exists():
            ui.err(f"Error: File or directory not found: {input_file}")
            return
        
        # Check if it's a directory
        if input_path.is_dir():
            # Look for .out files in the directory
            out_files = list(input_path.glob('*.out'))
            if not out_files:
                ui.err(f"Error: No CRYSTAL output files (.out) found in directory: {input_path}")
                ui.err("Please specify a CRYSTAL output file (e.g., material.out) or a directory containing .out files.")
                return
            else:
                # Process all files in the directory
                print()
                ui.info(f"Found {len(out_files)} output file(s) in {input_path.name}:")
                for f in sorted(out_files)[:10]:  # Show first 10
                    ui.info(f"  - {f.name}")
                if len(out_files) > 10:
                    ui.info(f"  ... and {len(out_files) - 10} more")
                
                # Get calculation type
                if not args.calc_type:
                    print()
                    ui.info("Select calculation type for all files:")
                    ui.info("1: BAND - Electronic band structure")
                    ui.info("2: DOSS - Density of states")
                    ui.info("3: TRANSPORT - Transport properties")
                    ui.info("4: CHARGE - Charge density")
                    ui.info("5: POTENTIAL - Electrostatic potential")
                    ui.info("6: CHARGE+POTENTIAL - Combined calculation")
                    ui.info("7: ORBITALS - Orbitals as Molden files")

                    choice = input("\nSelect type (1-7): ").strip()
                    calc_types = {
                        "1": "BAND", "2": "DOSS", "3": "TRANSPORT",
                        "4": "CHARGE", "5": "POTENTIAL", "6": "CHARGE+POTENTIAL",
                        "7": "ORBITALS"
                    }
                    while choice not in calc_types:
                        # Re-prompt instead of silently defaulting a typo to BAND
                        # (same contract as the single-file path).
                        ui.warn(f"  '{choice}' is not a valid choice; enter a number 1-7.")
                        choice = input("Select type (1-7): ").strip()
                    calc_type = calc_types[choice]
                else:
                    calc_type = args.calc_type
                
                # Ask about shared settings if more than one file
                shared_config = None
                if len(out_files) > 1:
                    print()
                    ui.info("Multiple files detected. Use shared settings for all files?")
                    use_shared = yes_no_prompt("Use same configuration for all files?", "yes")

                    if use_shared:
                        print()
                        ui.rule("Configuring shared settings for all files")
                        # Use first file as reference for getting structure info
                        first_file = str(out_files[0])
                        temp_generator = D3Generator(first_file, calc_type, args.output_dir)
                        shared_config = temp_generator.generate_d3()
                        if shared_config:
                            print()
                            ui.ok("Shared settings configured")

                            # Option to save configuration
                            if yes_no_prompt("\nSave this configuration for future use?", "no"):
                                save_d3_options_prompt(shared_config, skip_prompt=True)
                        else:
                            # No config came back (no wavefunction for the
                            # reference file, or the settings were abandoned).
                            # Say so instead of silently dropping into per-file
                            # configuration for every file.
                            print()
                            ui.warn("Shared settings were not configured; "
                                    "configuring each file individually.")

                # Process all files
                print()
                ui.info(f"Processing {len(out_files)} files...")
                failed = []
                for out_file in sorted(out_files):
                    print()
                    ui.rule(f"Processing: {out_file.name}")
                    generator = D3Generator(str(out_file), calc_type, args.output_dir)

                    if shared_config:
                        # the shared settings were answered on stdin: so are
                        # this file's own questions (map plane, points)
                        config = generator.generate_d3(shared_config, answers_on_stdin=True)
                    else:
                        # Interactive configuration for each file
                        config = generator.generate_d3()
                    if config is None:
                        failed.append(out_file.name)

                print()
                ui.rule()
                if failed:
                    ui.err(f"No D3 file generated for: {', '.join(failed)}")
                ui.ok(f"Completed processing {len(out_files) - len(failed)} of "
                      f"{len(out_files)} files")
                return
        elif not input_path.is_file():
            ui.err(f"Error: {input_path} is not a valid file.")
            ui.err("Please specify a CRYSTAL output file (e.g., material.out).")
            return
        elif not input_path.suffix.lower() in ['.out', '.log']:
            ui.warn(f"Warning: {input_path.name} may not be a CRYSTAL output file.")
            ui.warn("CRYSTAL output files typically have .out or .log extensions.")
        
        # If we're in batch mode (directory was provided), skip single-file processing
        if not (hasattr(args, 'batch') and args.batch and input_file is None):
            # Single file processing.
            # A config file carries its own calculation_type. Load it ONCE here and
            # reuse it for both calc-type resolution and generation -- it was
            # previously loaded twice (a 'peek' plus the load below), printing the
            # "Loaded D3 configuration..." banner twice. (The peek originally fixed
            # an EOFError when applying a saved config non-interactively.)
            config = None
            if args.config_file:
                config = load_d3_config(args.config_file)
                if not config:
                    ui.err(f"Failed to load configuration from {args.config_file}")
                    sys.exit(1)
                if not args.calc_type and config.get('calculation_type'):
                    args.calc_type = config['calculation_type']

            if not args.calc_type:
                print()
                ui.info("Select calculation type:")
                ui.info("1: BAND - Electronic band structure")
                ui.info("2: DOSS - Density of states")
                ui.info("3: TRANSPORT - Transport properties (Boltzmann)")
                ui.info("4: CHARGE - Charge density (3D or 2D)")
                ui.info("5: POTENTIAL - Electrostatic potential")
                ui.info("6: CHARGE+POTENTIAL - Combined calculation")
                ui.info("7: ORBITALS - Orbitals as Molden files")

                calc_types = {
                    "1": "BAND", "2": "DOSS", "3": "TRANSPORT",
                    "4": "CHARGE", "5": "POTENTIAL", "6": "CHARGE+POTENTIAL",
                    "7": "ORBITALS"
                }
                choice = input("\nSelect type (1-7): ").strip()
                while choice not in calc_types:
                    # Re-prompt instead of silently defaulting a typo to BAND, which
                    # would generate the wrong property calculation with no warning.
                    ui.warn(f"  '{choice}' is not a valid choice; enter a number 1-7.")
                    choice = input("Select type (1-7): ").strip()
                calc_type = calc_types[choice]
            else:
                calc_type = args.calc_type

            # Validate / summarize the already-loaded config.
            if config:
                print_d3_config_summary(config)
                is_valid, errors = validate_d3_config(config)
                if not is_valid:
                    ui.err("Configuration validation errors:")
                    for error in errors:
                        ui.err(f"  - {error}")
                    # exit non-zero: callers (workflow engine) detect failure
                    # by exit code / missing output, not by parsing stderr
                    sys.exit(1)
                # Override calc_type from config if present
                if "calculation_type" in config:
                    calc_type = config["calculation_type"]
            
            generator = D3Generator(input_file, calc_type, args.output_dir)
            
            # Generate D3 file with config if available
            if config:
                result = generator.generate_d3(config)
            else:
                # Interactive configuration
                config = generator.generate_d3()
                result = config

                # Option to save configuration
                if args.save_config and config:
                    if args.options_file:
                        # Use specified filename without prompting
                        save_d3_config(config, args.options_file)
                        ui.ok(f"Configuration saved to {args.options_file}")
                    else:
                        save_d3_options_prompt(config, skip_prompt=True)

            if result is None:
                # No D3 file was written. The workflow executor gates on the
                # exit code alone, so returning 0 here reports a success that
                # never happened.
                ui.err("No D3 file was generated.")
                sys.exit(1)


if __name__ == "__main__":
    main()