"""
D3 Interactive Configuration Module

Author: Marcus Djokic
Institution: Michigan State University, Mendoza Group

This module contains:
- Interactive configuration functions for all D3 calculation types
- Output file parsing utilities
- Basis set parsing and orbital projection generation
- SHRINK extraction and modification utilities
"""

import os
import re
from typing import Dict, List, Optional, Tuple, Any, Union
import sys
from pathlib import Path

# Single-sourced full-precision constants (fallback keeps this standalone-runnable).
try:
    from mace.constants import HARTREE_TO_EV, EV_TO_HARTREE
except ImportError:
    HARTREE_TO_EV = 27.211386245988
    EV_TO_HARTREE = 1.0 / HARTREE_TO_EV

# Import k-point utilities from the new module
from d3_kpoints import (
    get_crystal_system_from_space_group,
    get_band_path_from_symmetry,
    get_kpoint_coordinates_from_labels,
    get_literature_kpath_vectors,
    get_seekpath_full_kpath,
    extract_and_process_shrink,
    scale_kpoint_segments,
    default_band_format,
    BAND_TEMPLATES
)

# Add Crystal_d12 to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "Crystal_d12"))

# Opt-in "press b to go back" navigation + crash-safe back-aware readers
# (fall back to plain input() behavior if menu_nav is unavailable).
try:
    from menu_nav import (nav_read as _nav_read, nav_int as _nav_int,
                          nav_float as _nav_float)
except Exception:  # pragma: no cover - defensive fallback
    def _nav_read(prompt="", valid_set=None):
        return input(prompt)

    def _nav_int(prompt="", default=None, choices=None):
        while True:
            r = input(prompt).strip()
            if not r and default is not None:
                return int(default)
            try:
                return int(r)
            except ValueError:
                print("Please enter a whole number.")

    def _nav_float(prompt="", default=None):
        while True:
            r = input(prompt).strip()
            if not r and default is not None:
                return float(default)
            try:
                return float(r)
            except ValueError:
                print("Please enter a number.")


# Import yes_no_prompt function
def yes_no_prompt(prompt: str, default: str = "yes") -> bool:
    """Prompt for yes/no input with default value."""
    if default.lower() == "yes":
        prompt_str = f"{prompt} [Y/n]: "
        default_val = True
    else:
        prompt_str = f"{prompt} [y/N]: "
        default_val = False

    invalid = 0
    while True:
        response = _nav_read(prompt_str, valid_set={"y", "yes", "n", "no"}).strip().lower()
        if not response:
            return default_val
        if response in ['y', 'yes']:
            return True
        if response in ['n', 'no']:
            return False
        invalid += 1
        from d12_constants import stop_if_out_of_step
        stop_if_out_of_step(invalid, prompt, "yes or no")
        print("Please respond with 'yes' or 'no' (or 'y' or 'n').")
# Import generate_k_points function
def generate_k_points(system: str, n_points: int = 30) -> List[Tuple[str, List[float]]]:
    """Generate high-symmetry k-points for band structure calculations."""
    # For now, return a simple path
    return [
        ("G", [0.0, 0.0, 0.0]),
        ("X", [0.5, 0.0, 0.0]),
        ("M", [0.5, 0.5, 0.0]),
        ("G", [0.0, 0.0, 0.0])
    ]


def crystal_family_and_centring(out_file: str) -> Tuple[str, str]:
    """(crystal family, centering letter) of a CRYSTAL output's space group,
    e.g. ("TRIGONAL", "R") for R -3 C, as get_crystal_supported_labels takes
    them; ("CUBIC", "P") when the group cannot be read."""
    try:
        from d12_constants import SPACEGROUP_SYMBOL_TO_NUMBER, SPACEGROUP_ALTERNATIVES
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent.parent / "Crystal_d12"))
        from d12_constants import SPACEGROUP_SYMBOL_TO_NUMBER, SPACEGROUP_ALTERNATIVES
    with open(out_file, 'r', errors='ignore') as f:
        content = f.read()
    match = re.search(r'^\s*SPACE GROUP \((?:NON)?CENTROSYMMETRIC\)\s*:\s*(\S.*?)\s*$',
                      content, re.M)
    if not match:
        return "CUBIC", "P"
    symbol = match.group(1)
    number = (SPACEGROUP_SYMBOL_TO_NUMBER.get(symbol) or SPACEGROUP_ALTERNATIVES.get(symbol)
              or SPACEGROUP_SYMBOL_TO_NUMBER.get(symbol.replace(' ', ''))
              or SPACEGROUP_ALTERNATIVES.get(symbol.replace(' ', '')))
    if not number:
        return "CUBIC", "P"
    for top, family in ((2, "TRICLINIC"), (15, "MONOCLINIC"), (74, "ORTHORHOMBIC"),
                        (142, "TETRAGONAL"), (167, "TRIGONAL"), (194, "HEXAGONAL"),
                        (230, "CUBIC")):
        if number <= top:
            return family, symbol[0].upper()
    return "CUBIC", "P"


def _ask_doss_resolution(out_file: Optional[str], doss_config: Dict[str, Any]) -> Dict[str, Any]:
    """DOSS's energy resolution: a preset step (Standard by default), another step, or a number of
    points honored as given. Each choice shows the points and step it gives for this output. When a
    step over all bands needs more than DOSS_WINDOW_OFFER_NPTS points, a window around E_F is offered."""
    try:
        import d3_config as C
    except ImportError:
        from Crystal_d3 import d3_config as C
    text = ""
    if out_file and os.path.exists(out_file):
        with open(out_file, errors="replace") as fh:
            text = fh.read()
    span, estimated = C.doss_span(doss_config, C.doss_atomic_numbers(text))
    chosen = {k: v for k, v in C.doss_override().items() if k not in ("energy_window", "all_bands")}
    if not chosen:
        chosen = _doss_resolution_menu(C, span, estimated)
    # a step over every band (core levels to the highest virtual bands): offer a window around E_F
    if (estimated and C.doss_explicit_points(chosen) is None and not doss_config.get("band_range")
            and not set(C.doss_override()) & {"energy_window", "all_bands"} and not C.non_interactive()):
        step = C.doss_step_ev(chosen)
        offer = C.doss_window_offer(span, C.doss_npts(span, None, step), step)
        if offer:
            chosen.update(_offer_doss_window(C, out_file, offer))
    return chosen


def _offer_doss_window(C, out_file: Optional[str], offer: str) -> Dict[str, Any]:
    """The window around E_F, the default for a fine all-bands grid. {} is the default window (the deck
    writer applies it, as on every route), an energy_window other bounds, all_bands every band. A 1D
    material's default is its bands by index instead (CRYSTAL needs a window's bounds in band gaps):
    {} again, conduction_bands another number of conduction bands."""
    banded = _offer_band_index(C, out_file, offer)
    if banded is not None:
        return banded
    fermi, why = trusted_fermi_au(out_file)
    print(f"\n{offer}.")
    if fermi is None:
        print(f"No window around E_F can be written: {why}. The deck covers all bands.")
        return {}
    lo, hi = C.DOSS_DEFAULT_WINDOW_EV
    raw = _nav_read(f"Use a window around E_F ({lo:g} to +{hi:g} eV) instead? [Y/n]: ",
                    valid_set={"y", "n", "yes", "no"}).strip().lower()
    if raw in ("n", "no"):
        return {"all_bands": True}
    while True:
        lo_ev = _nav_float(f"Window below E_F (eV) [{lo:g}]: ", default=lo)
        hi_ev = _nav_float(f"Window above E_F (eV) [{hi:g}]: ", default=hi)
        if lo_ev < 0 < hi_ev:
            if (lo_ev, hi_ev) == (lo, hi):
                return {}
            return {"energy_window": C.doss_window_ha(lo_ev, hi_ev)}
        print("The window must run from below E_F (a negative number) to above it (a positive one).")


def _offer_band_index(C, out_file: Optional[str], offer: str) -> Optional[Dict[str, Any]]:
    """A 1D material's default (d3_config.doss_band_index_range): its valence and low conduction bands by
    index. None for another dimensionality (or an output without band counts)."""
    if not out_file or not os.path.exists(str(out_file)):
        return None
    try:
        from d3_kpoints import output_dimensionality
        if output_dimensionality(str(out_file)) != 1:
            return None
    except Exception:
        return None
    with open(out_file, errors="replace") as fh:
        counts = C.doss_band_counts(fh.read())
    if counts is None:
        return None
    first, last = C.doss_band_index_range(counts)
    occ = counts["occupied"]
    print(f"\n{offer}.")
    print(f"A 1D material's DOS takes its bands by index (CRYSTAL needs an energy window's bounds in band gaps): "
          f"bands {first} to {last}, from the first band above the {counts['core']} core bands to {last - occ} "
          f"bands above the highest occupied one (band {occ}).")
    raw = _nav_read(f"Use bands {first} to {last} by index instead? [Y/n]: ",
                    valid_set={"y", "n", "yes", "no"}).strip().lower()
    if raw in ("n", "no"):
        return {"all_bands": True}
    while True:
        n = _nav_int(f"Conduction bands above the highest occupied band [{last - occ}]: ", default=last - occ)
        if n >= 1:
            return {} if n == last - occ else {"conduction_bands": n}
        print("Please enter at least 1 band.")


def _doss_resolution_menu(C, span, estimated) -> Dict[str, Any]:
    menu = C.doss_resolution_menu(span, estimated)
    where = (f"the {span:.0f} eV window" if not estimated else
             f"all bands, ~{span:.0f} eV (estimated from the atoms; the run prints the real span)") if span else None
    print("\nEnergy resolution (the energy step of the DOS grid" + (f"; {where}" if where else "") + "):")
    for k, (_, line) in enumerate(menu, 1):
        print(f"{k}: {line}")
    default = str(C.DOSS_RESOLUTIONS.index("standard") + 1)
    invalid = 0
    while True:
        raw = _nav_read(f"Select energy resolution (1-{len(menu)}) [{default}]: ", valid_set={"_choice_"}).strip() or default
        if raw.isdigit() and 1 <= int(raw) <= len(menu):
            break
        invalid += 1
        try:
            from menu_nav import stop_if_out_of_step
            stop_if_out_of_step(invalid, "Select energy resolution", f"1-{len(menu)}")
        except ImportError:
            pass
        print(f"Please choose 1-{len(menu)}.")
    name = menu[int(raw) - 1][0]
    if name in C.DOSS_STEP_PRESETS:
        return {"energy_resolution": name, "n_points": "auto"}
    if name == "step":
        while True:
            step = _nav_float(f"DOS energy step (eV) [{C.DOSS_DEFAULT_STEP_EV}]: ", default=C.DOSS_DEFAULT_STEP_EV)
            if step > 0:
                return {"energy_resolution": "step", "energy_step_ev": step, "n_points": "auto"}
            print("Please enter a step larger than 0.")
    while True:
        n = _nav_int(f"Number of energy points [{C.DOSS_NPTS_RANGE[0]}]: ", default=C.DOSS_NPTS_RANGE[0])
        if 2 <= n <= C.DOSS_MAX_NPTS:
            return {"energy_resolution": "points", "n_points": n}
        print(f"Please enter 2 to {C.DOSS_MAX_NPTS} points.")


def parent_newk_is(out_file: Optional[str]) -> Optional[int]:
    """The parent SCF's IS as a DOSS/TRANSPORT NEWK takes it (its .d12, else its .out)."""
    if not out_file or not os.path.exists(str(out_file)):
        return None
    try:
        from d3_kpoints import output_dimensionality
        dim = output_dimensionality(str(out_file))
    except Exception:
        dim = None
    d12 = Path(out_file).with_suffix(".d12")
    try:
        lines = (extract_shrink_from_d12(str(d12), for_doss=True, dimensionality=dim) if d12.exists()
                 else extract_shrink_from_out(str(out_file), for_doss=True, dimensionality=dim))
        n = int(lines[0].split()[0]) if lines else 0
    except Exception:
        return None
    return n if n > 0 else None


def _ask_transport_k(out_file: Optional[str]) -> Dict[str, Any]:
    """TRANSPORT's k mesh: a preset spacing (Standard by default), the parent's mesh, another spacing, or an IS."""
    try:
        import d3_config as C
    except ImportError:
        from Crystal_d3 import d3_config as C
    if C.transport_override():
        return C.transport_override()
    parent = parent_newk_is(out_file)
    recip = C.reciprocal_lengths(out_file) if out_file else None
    menu = C.transport_k_menu(parent, recip)
    print("\nk-point mesh for the transport distribution (BOLTZTRA's NEWK; the SCF's mesh converges energies, "
          "not transport):")
    for k, (_, line) in enumerate(menu, 1):
        print(f"{k}: {line}")
    default = str(C.TRANSPORT_K_RESOLUTIONS.index("standard") + 1)
    while True:
        raw = _nav_read(f"Select k-point mesh (1-{len(menu)}) [{default}]: ", valid_set={"_choice_"}).strip() or default
        if raw.isdigit() and 1 <= int(raw) <= len(menu):
            break
        print(f"Please choose 1-{len(menu)}.")
    name = menu[int(raw) - 1][0]
    if name in C.TRANSPORT_K_PRESETS:
        return {"k_resolution": name}
    if name == "spacing":
        while True:
            std = C.transport_k_spacing({"k_resolution": "standard"}, len(recip) if recip else 3)
            sp = _nav_float(f"k-point spacing (1/Å) [{std:g}]: ", default=std)
            if sp > 0:
                return {"k_resolution": "spacing", "k_spacing": sp}
            print("Please enter a spacing larger than 0.")
    while True:
        n = _nav_int(f"NEWK shrinking factor IS [{parent or 12}]: ", default=parent or 12)
        if n >= 1:
            return {"k_resolution": "is", "newk_is": n}
        print("Please enter at least 1.")


def trusted_fermi_au(out_file: Optional[str]) -> Tuple[Optional[float], str]:
    """(E_F in Hartree or None, why) for E_F-relative ranges of a deck written from ``out_file``
    (a DOSS window, TRANSPORT's chemical potentials): the SCF's printed one unless it is missing or
    CRYSTAL's -1.0 placeholder (spins locked, SPINLOCK), else a BAND or DOSS run of the same SCF
    beside it (mace.utils.fermi_level.scf_fermi). None means: no window may be put around it."""
    if not out_file or not os.path.exists(str(out_file)):
        return None, "no output to read a Fermi energy from"
    printed = get_band_info_from_output(str(out_file)).get('fermi_energy')
    try:
        from mace.utils.fermi_level import scf_fermi
    except ImportError:
        if printed is not None and abs(float(printed) + 1.0) > 5e-6:
            return float(printed), ""
        return None, "no trustworthy Fermi energy in the output (CRYSTAL's -1.0 placeholder or none)"
    f = scf_fermi(str(out_file), printed)
    return f.value_au, f.note


def band_extents(out_file: Optional[str]):
    """(BAND.DAT, first band, each band's (lowest, highest) energy, Hartree, absolute) of a BAND run of the
    SCF ``out_file`` beside it (mace.utils.fermi_level.sibling_band_extents), else None."""
    if not out_file or not os.path.exists(str(out_file)):
        return None
    try:
        from mace.utils.fermi_level import sibling_band_extents
    except ImportError:
        return None
    try:
        return sibling_band_extents(str(out_file))
    except Exception:
        return None


def get_band_info_from_output(out_file: str) -> Dict[str, Any]:
    """Extract band structure information from CRYSTAL output file."""
    info = {
        'n_bands': 0,
        'valence_bands': 0,
        'conduction_bands': 0,
        'band_gap': None,
        'gap_type': None,
        'fermi_energy': None,
        'n_k_points': 0,
        'space_group': 1,
        'lattice_type': 'P',
        'lattice_params': None
    }
    
    if not os.path.exists(out_file):
        return info
    
    with open(out_file, 'r') as f:
        content = f.read()
    
    # Number of bands (AOs)
    ao_match = re.search(r'NUMBER OF AO\s+(\d+)', content)
    if ao_match:
        info['n_bands'] = int(ao_match.group(1))
    
    # Band gap information
    gap_match = re.search(r'(DIRECT|INDIRECT) ENERGY BAND GAP\s*:\s*([\d.]+)', content)
    if gap_match:
        info['gap_type'] = gap_match.group(1).lower()
        info['band_gap'] = float(gap_match.group(2))
    
    # Fermi energy - try multiple patterns
    fermi_energy = None
    
    # Pattern 1: Standard FERMI ENERGY line
    fermi_match = re.search(r'FERMI ENERGY\s*\[.*?\]\s*:\s*([-\d.]+)', content)
    if fermi_match:
        fermi_energy = float(fermi_match.group(1))
    
    # Pattern 2: EFERMI from final SCF cycle (metallic systems)
    if fermi_energy is None:
        # Look for SCF convergence marker and then find the last EFERMI before it
        scf_ended_match = re.search(r'== SCF ENDED[^\n]*\n', content)
        if scf_ended_match:
            # Get content up to SCF convergence
            content_before_convergence = content[:scf_ended_match.start()]
            # Find all EFERMI occurrences before convergence
            efermi_matches = re.findall(r'EFERMI\(AU\)\s*([-\d.E+]+)', content_before_convergence)
            if efermi_matches:
                # Take the last one (from final SCF cycle)
                fermi_energy = float(efermi_matches[-1])
        else:
            # Fallback: if no SCF ENDED marker, still try to find last EFERMI
            efermi_matches = re.findall(r'EFERMI\(AU\)\s*([-\d.E+]+)', content)
            if efermi_matches:
                fermi_energy = float(efermi_matches[-1])
    
    # Pattern 3: Use TOP OF VALENCE BANDS as approximation for insulators
    if fermi_energy is None:
        # Look for band structure info after SCF convergence
        scf_ended_match = re.search(r'== SCF ENDED[^\n]*\n', content)
        if scf_ended_match:
            # Get content after SCF convergence (where band info is printed)
            content_after_convergence = content[scf_ended_match.end():]
            # Find VBM energies after convergence
            vbm_matches = re.findall(r'TOP OF VALENCE BANDS.*?EIG\s*([-\d.E+]+)\s*AU', content_after_convergence)
            if vbm_matches:
                vbm_energies = [float(e) for e in vbm_matches]
                fermi_energy = max(vbm_energies)
        
        # Fallback: search entire content if no SCF marker found
        if fermi_energy is None:
            vbm_matches = re.findall(r'TOP OF VALENCE BANDS.*?EIG\s*([-\d.E+]+)\s*AU', content)
            if vbm_matches:
                vbm_energies = [float(e) for e in vbm_matches]
                fermi_energy = max(vbm_energies)
    
    if fermi_energy is not None:
        info['fermi_energy'] = fermi_energy
    
    # Valence and conduction bands
    vb_match = re.search(r'TOP OF VALENCE BANDS.*?BAND\s+(\d+)', content, re.DOTALL)
    if vb_match:
        info['valence_bands'] = int(vb_match.group(1))
    
    cb_match = re.search(r'BOTTOM OF CONDUCTION BANDS.*?BAND\s+(\d+)', content, re.DOTALL)
    if cb_match:
        info['conduction_bands'] = int(cb_match.group(1)) - info['valence_bands']
    
    # Space group
    sg_match = re.search(r'SPACE GROUP.*?NUMBER:\s*(\d+)', content)
    if sg_match:
        info['space_group'] = int(sg_match.group(1))
    
    # Lattice type from space group symbol
    sg_symbol_match = re.search(r'SPACE GROUP.*?:\s+([A-Z]\s*[\-/0-9\s]*[A-Z0-9]*)', content)
    if sg_symbol_match:
        symbol = sg_symbol_match.group(1).strip()
        if symbol:
            info['lattice_type'] = symbol[0]
    
    # Lattice parameters (A, B, C, ALPHA, BETA, GAMMA)
    lattice_match = re.search(r'LATTICE PARAMETERS.*?\n\s*A\s+B\s+C\s+ALPHA\s+BETA\s+GAMMA.*?\n\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', content, re.DOTALL)
    if lattice_match:
        info['lattice_params'] = [
            float(lattice_match.group(1)),  # A
            float(lattice_match.group(2)),  # B
            float(lattice_match.group(3)),  # C
            float(lattice_match.group(4)),  # ALPHA
            float(lattice_match.group(5)),  # BETA
            float(lattice_match.group(6))   # GAMMA
        ]
    
    return info


def get_recommended_doss_shrink(dimensionality: int, original_is: int, warn_only: bool = True) -> int:
    """Get recommended IS value for DOSS based on CRYSTAL manual guidelines.
    
    Args:
        dimensionality: System dimensionality (0, 1, 2, or 3)
        original_is: Original IS value from calculation
        warn_only: If True, warn but don't cap values outside recommendations
        
    Returns:
        Recommended IS value for DOSS
    """
    if dimensionality == 3:
        # 3D systems: 4-12 suggested in manual
        if warn_only and original_is > 12:
            print(f"    Note: IS={original_is} is higher than typical (4-12) for 3D DOSS")
            print(f"    Using original value. Ensure NPOL is appropriate (10-18).")
            return original_is
        return max(4, original_is) if warn_only else min(max(4, original_is), 12)
    elif dimensionality in [1, 2]:
        # 1D/2D systems: 6-18 suggested
        if warn_only and original_is > 18:
            print(f"    Note: IS={original_is} is higher than typical (6-18) for {dimensionality}D DOSS")
            print(f"    Using original value. Ensure NPOL is appropriate (10-18).")
            return original_is
        return max(6, original_is) if warn_only else min(max(6, original_is), 18)
    else:
        # 0D systems: use conservative values
        return max(4, original_is) if warn_only else min(max(4, original_is), 8)


def extract_shrink_from_d12(d12_file: str, for_doss: bool = False, dimensionality: Optional[int] = None) -> List[str]:
    """Extract SHRINK lines from D12 file and format for NEWK.
    
    Args:
        d12_file: Path to D12 file
        for_doss: If True, format for DOSS NEWK (ISP >= 2*IS)
        dimensionality: System dimensionality for DOSS recommendations
        
    Returns:
        List of strings representing the SHRINK/NEWK section
    """
    
    if not os.path.exists(d12_file):
        if for_doss:
            # Conservative default for DOSS
            is_val = 8 if dimensionality == 3 else 12
            isp = 2 * is_val
            return [f"{is_val} {isp}"]
        else:
            return ["48 48"]  # Default fallback
        
    shrink_lines = []
    found_shrink = False
    line_count = 0
    
    with open(d12_file, 'r') as f:
        for line in f:
            if found_shrink:
                shrink_lines.append(line.strip())
                line_count += 1
                if line_count >= 2:  # We need at most 2 lines after SHRINK
                    break
            elif 'SHRINK' in line:
                found_shrink = True
                
    if not shrink_lines:
        return ["48 48"]  # Default fallback

    return _newk_from_shrink_lines(shrink_lines, for_doss, dimensionality)


def extract_shrink_from_out(out_file: str, for_doss: bool = False,
                            dimensionality: Optional[int] = None) -> Optional[List[str]]:
    """The NEWK lines extract_shrink_from_d12 would give, read from a .out.

    For a parent whose .d12 is not beside its .out. CRYSTAL prints the mesh as
    "SHRINK. FACT.(MONKH.)   10 10 10" and, when the Gilat net differs from
    it, "SHRINKING FACTOR(GILAT NET)   20". An equal mesh in the periodic
    directions reads as "SHRINK IS IPMG", an unequal one as "SHRINK 0 IPMG"
    plus the three factors, as the .d12 would have written it. None when the
    .out prints no mesh (a molecule, or no .out).
    """
    try:
        with open(out_file, 'r', errors='replace') as f:
            text = f.read()
    except OSError:
        return None
    m = re.search(r'SHRINK\.\s*FACT\.\(MONKH\.\)\s*(\d+)\s+(\d+)\s+(\d+)', text)
    if not m:
        return None
    k = [int(m.group(i)) for i in (1, 2, 3)]
    g = re.search(r'SHRINKING FACTOR\(GILAT NET\)\s*(\d+)', text[m.start():])
    periodic = k[:dimensionality] if dimensionality in (1, 2, 3) else [x for x in k if x > 1] or k
    is_val = max(k)
    gilat = int(g.group(1)) if g else is_val
    if len(set(periodic)) == 1:
        shrink_lines = [f"{is_val} {gilat}"]
    else:
        shrink_lines = [f"0 {gilat}", f"{k[0]} {k[1]} {k[2]}"]
    return _newk_from_shrink_lines(shrink_lines, for_doss, dimensionality)


def _newk_from_shrink_lines(shrink_lines: List[str], for_doss: bool,
                            dimensionality: Optional[int]) -> List[str]:
    """NEWK lines from the lines that follow a SHRINK record."""
    # Parse the SHRINK format
    first_line = shrink_lines[0].split()
    
    if len(first_line) == 2:
        # Could be IS IPMG or "0 IPMG" format
        is_val = int(first_line[0])
        ipmg = int(first_line[1])
        
        # Check for SHRINK 0 format
        if is_val == 0 and len(shrink_lines) > 1:
            # SHRINK 0 format with k-points on next line
            k_line = shrink_lines[1].split()
            if len(k_line) >= 3:
                k1, k2, k3 = int(k_line[0]), int(k_line[1]), int(k_line[2])
                
                if for_doss:
                    # For DOSS, ensure ISP >= 2*IS as per CRYSTAL manual
                    is_val = max(k1, k2, k3)
                    if dimensionality is not None:
                        is_val = get_recommended_doss_shrink(dimensionality, is_val)
                    isp = 2 * is_val
                    return [f"{is_val} {isp}"]
                else:
                    # Keep original format
                    return shrink_lines[:2]
        else:
            # Standard IS IPMG format
            if for_doss:
                # For DOSS, ensure ISP >= 2*IS as per CRYSTAL manual
                if dimensionality is not None:
                    is_val = get_recommended_doss_shrink(dimensionality, is_val)
                isp = max(ipmg, 2 * is_val)
                return [f"{is_val} {isp}"]
            else:
                # For other calculations, keep original
                return [f"{is_val} {ipmg}"]
    
    elif len(first_line) == 1:
        # Single value SHRINK format
        is_val = int(first_line[0])
        if for_doss:
            if dimensionality is not None:
                is_val = get_recommended_doss_shrink(dimensionality, is_val)
            isp = 2 * is_val
            return [f"{is_val} {isp}"]
        else:
            return [first_line[0]]
                
    # Return first line if we couldn't parse
    return [shrink_lines[0]]


GHOST_SUFFIX = "_GHOST"


def ghost_atom_labels(text: str) -> set:
    """Labels of the atoms a run turned into ghosts (GHOSTS, manual p.82).

    CRYSTAL lists them after " * ATOMS TRANSFORMED INTO GHOSTS :   N" as
    "label(Z)" entries, e.g. "   1(  8)".
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = re.search(r"ATOMS TRANSFORMED INTO GHOSTS\s*:\s*(\d+)", line)
        if not m:
            continue
        n, labels = int(m.group(1)), set()
        for row in lines[i + 1:]:
            found = re.findall(r"(\d+)\(\s*\d+\)", row)
            if not found:
                break
            labels.update(int(x) for x in found)
            if len(labels) >= n:
                break
        return labels
    return set()


def parse_basis_set_info(out_file: str) -> Tuple[List[List], int]:
    """Parse basis set information from CRYSTAL output file using alldos.py logic."""
    
    data_list = []
    n_ao = 0
    
    if not os.path.exists(out_file):
        return data_list, n_ao
        
    with open(out_file, 'r') as f:
        lines = f.readlines()
    
    # Extract number of AOs
    for line in lines:
        if "NUMBER OF AO" in line:
            clean_line = [x for x in line.split(" ") if x != ""]
            if len(clean_line) > 3:
                n_ao = int(clean_line[3])
            break
    
    # Find LOCAL ATOMIC FUNCTIONS BASIS SET section
    index_shells = 0
    for i, line in enumerate(lines):
        if "LOCAL ATOMIC FUNCTIONS BASIS SET" in line:
            index_shells = i
            break
    
    if index_shells == 0:
        return data_list, n_ao
    
    # Extract shell data starting from 4 lines after the header (like alldos.py)
    unclean_shells = []
    for index in range(index_shells + 4, len(lines)):
        if "PROCESS" in lines[index]:
            continue
        if (
            "*******" in lines[index]
            or " INFORMATION " in lines[index]
            or lines[index] == "\n"
        ):
            break
        else:
            unclean_shells.append(lines[index].rstrip('\n'))
    
    # Convert to data_list format
    for line in unclean_shells:
        data_list.append([x for x in line.split(" ") if x != ""])

    # A ghost atom's functions are listed under its element; projected with
    # that element they would add the ghost's (empty) states to it.
    ghosts = ghost_atom_labels("".join(lines))
    for entry in data_list:
        if len(entry) == 5 and entry[0].isdigit() and int(entry[0]) in ghosts:
            entry[1] += GHOST_SUFFIX

    return data_list, n_ao


def get_unique_atoms(atoms: List[str]) -> List[Tuple[str, int]]:
    """Get unique atoms and their first occurrence index."""
    seen = {}
    result = []
    for i, atom in enumerate(atoms):
        if atom not in seen:
            seen[atom] = i
            result.append((atom, i))
    return result

def get_atoms_and_shells(data_list: List[List]) -> Tuple[List[str], List[Dict]]:
    """Extract atoms and shell information from alldos.py style data_list."""
    
    # Find all atoms in the data (lines with 5 elements)
    atoms = []
    atom_list = []  # Unique atom types
    atoms_index = []
    
    for i, line in enumerate(data_list):
        if len(line) == 5:
            atoms.append(line[1])  # Element symbol
            # A ghost has its element's functions: CRYSTAL lists them under
            # the first atom of that element, ghost or not.
            element = line[1].removesuffix(GHOST_SUFFIX)
            if element not in atom_list:
                atom_and_index = [element, i]
                atom_list.append(element)
                atoms_index.append(atom_and_index)
    
    # Count shells for each atom type (similar to alldos.py's number_shells)
    atoms_shells = {}
    for atom in atoms_index:
        S, P, SP, D, F = 0, 0, 0, 0, 0
        for i in range(int(atom[1]) + 1, len(data_list)):
            if len(data_list[i]) >= 5:
                break
            elif len(data_list[i]) > 0 and "S" in data_list[i][-1]:
                S += 1
            elif len(data_list[i]) > 0 and "SP" in data_list[i][-1]:
                if len(data_list[i]) == 2:
                    SP_shells = data_list[i][0].split("-")
                    for j in range(int(SP_shells[0]), int(SP_shells[1]) + 1):
                        SP += 1
                elif len(data_list[i]) == 3:
                    for j in range(
                        int(data_list[i][0].split("-")[0]), int(data_list[i][1]) + 1
                    ):
                        SP += 1
            elif len(data_list[i]) > 0 and "P" in data_list[i][-1]:
                if len(data_list[i]) == 2:
                    P_shells = data_list[i][0].split("-")
                    for j in range(int(P_shells[0]), int(P_shells[1]) + 1):
                        P += 1
                elif len(data_list[i]) == 3:
                    for j in range(
                        int(data_list[i][0].split("-")[0]), int(data_list[i][1]) + 1
                    ):
                        P += 1
            elif len(data_list[i]) > 0 and "D" in data_list[i][-1]:
                if len(data_list[i]) == 2:
                    D_shells = data_list[i][0].split("-")
                    for j in range(int(D_shells[0]), int(D_shells[1]) + 1):
                        D += 1
                elif len(data_list[i]) == 3:
                    for j in range(
                        int(data_list[i][0].split("-")[0]), int(data_list[i][1]) + 1
                    ):
                        D += 1
            elif len(data_list[i]) > 0 and "F" in data_list[i][-1]:
                if len(data_list[i]) == 2:
                    F_shells = data_list[i][0].split("-")
                    for j in range(int(F_shells[0]), int(F_shells[1]) + 1):
                        F += 1
                elif len(data_list[i]) == 3:
                    for j in range(
                        int(data_list[i][0].split("-")[0]), int(data_list[i][1]) + 1
                    ):
                        F += 1
        atoms_shells[atom[0]] = {"S": S, "SP": SP, "P": P, "D": D, "F": F}
        S, P, SP, D, F = 0, 0, 0, 0, 0
    for atom in atoms:
        if atom.endswith(GHOST_SUFFIX) and atom not in atoms_shells:
            atoms_shells[atom] = dict(atoms_shells[atom.removesuffix(GHOST_SUFFIX)])

    return atoms, atoms_shells


def get_shells(atoms: List[str], atoms_shells: Dict) -> Dict[str, Dict[str, List[int]]]:
    """Generate shell information for each unique element (alldos.py style)."""
    # In this context, atoms_shells is already the shell count dictionary
    # We need to use generate_orbital_indices instead
    return generate_orbital_indices(atoms, atoms_shells)

def generate_orbital_indices(atoms: List[str], atoms_shells: List[Dict]) -> Dict[str, Dict[str, List[int]]]:
    """Generate orbital indices similar to alldos.py get_shells function."""
    
    AOindex = 1
    total_shells = {}
    
    for element in atoms_shells:
        total_shells[element] = {"S": [], "P": [], "D": [], "F": [], "SP": []}
    
    # Process all atoms (not just unique elements)
    for atom in atoms:
        for shell in range(0, int(atoms_shells[atom]["S"])):
            total_shells[atom]["S"].append(AOindex)
            AOindex += 1
        for shell in range(0, int(atoms_shells[atom]["SP"])):
            total_shells[atom]["SP"].append(AOindex)
            AOindex += 1
        for shell in range(0, int(atoms_shells[atom]["P"])):
            total_shells[atom]["P"].append(AOindex)
            AOindex += 1
        for shell in range(0, int(atoms_shells[atom]["D"])):
            total_shells[atom]["D"].append(AOindex)
            AOindex += 1
        for shell in range(0, int(atoms_shells[atom]["F"])):
            total_shells[atom]["F"].append(AOindex)
            AOindex += 1
    
    return total_shells


def create_doss_projections(total_shells: Dict, element_only: bool = False, include_totals: bool = True) -> List[str]:
    """Create DOSS projection specifications from orbital data (alldos.py style)."""
    
    projections = []
    
    if element_only:
        # Project by element only (option 2)
        for atom in total_shells:
            atom_total_indices = []
            
            # Collect all orbital indices for this element
            for shell_type in ['S', 'SP', 'P', 'D', 'F']:
                atom_total_indices.extend(total_shells[atom][shell_type])
            
            if atom_total_indices:
                proj_line = f"{len(atom_total_indices)}  {' '.join(map(str, atom_total_indices))}  #{atom} all"
                projections.append(proj_line)
    else:
        # Project by element and orbital type (options 3 and 4)
        for atom in total_shells:
            # Add each shell type separately
            for shell_type in ['S', 'SP', 'P', 'D', 'F']:
                if len(total_shells[atom][shell_type]) > 0:
                    indices = total_shells[atom][shell_type]
                    proj_line = f"{len(indices)}  {' '.join(map(str, indices))}  #{atom} {shell_type}"
                    projections.append(proj_line)
            
            # Add total for this element only if requested (option 3)
            if include_totals:
                atom_total_indices = []
                for shell_type in ['S', 'SP', 'P', 'D', 'F']:
                    atom_total_indices.extend(total_shells[atom][shell_type])
                
                if atom_total_indices:
                    proj_line = f"{len(atom_total_indices)}  {' '.join(map(str, atom_total_indices))}  #{atom} all"
                    projections.append(proj_line)
    
    return projections


def configure_band_calculation(out_file: Optional[str] = None) -> Dict[str, Any]:
    """Configure BAND calculation settings interactively."""
    print("\n=== BAND STRUCTURE CONFIGURATION ===")
    
    band_config = {}
    
    # Don't ask for title in shared mode - it will be set per material
    # Title will be automatically set in CRYSTALOptToD3.py to include material name
    
    # Path definition method
    print("\nBand path definition:")
    print("1: Automatic - Use standard path based on crystal symmetry from output file")
    print("2: Template selection - Choose from common band paths")
    print("3: Custom labels - Specify path using labels (G, X, M, etc.)")
    print("4: Fractional coordinates - Specify path using k-point vectors")
    
    path_method = _nav_read("Select method (1-4) [1]: ", valid_set={"_choice_"}).strip() or "1"
    
    if path_method == "1":
        # True automatic path from output file
        if out_file:
            # Extract space group from output file using robust detection
            import re
            import sys

            # Import space group symbol mappings
            sys.path.insert(0, str(Path(__file__).parent.parent / "Crystal_d12"))
            try:
                from d12_constants import SPACEGROUP_SYMBOL_TO_NUMBER, SPACEGROUP_ALTERNATIVES
            except ImportError:
                SPACEGROUP_SYMBOL_TO_NUMBER = {}
                SPACEGROUP_ALTERNATIVES = {}

            space_group = 1
            lattice_type = 'P'

            try:
                with open(out_file, 'r') as f:
                    content = f.read()

                # Find space group symbol
                sg_symbol_match = re.search(r'SPACE GROUP[^:]*:\s*([A-Z][^$\n]+)', content)
                if sg_symbol_match:
                    symbol = sg_symbol_match.group(1).strip()
                    if symbol:
                        # Extract lattice type (first letter)
                        lattice_type = symbol[0]

                        # Try to map symbol to space group number using comprehensive dictionaries
                        if symbol in SPACEGROUP_SYMBOL_TO_NUMBER:
                            space_group = SPACEGROUP_SYMBOL_TO_NUMBER[symbol]
                        elif symbol in SPACEGROUP_ALTERNATIVES:
                            space_group = SPACEGROUP_ALTERNATIVES[symbol]
                        elif symbol.replace(' ', '') in SPACEGROUP_SYMBOL_TO_NUMBER:
                            space_group = SPACEGROUP_SYMBOL_TO_NUMBER[symbol.replace(' ', '')]
                        elif symbol.replace(' ', '-') in SPACEGROUP_SYMBOL_TO_NUMBER:
                            space_group = SPACEGROUP_SYMBOL_TO_NUMBER[symbol.replace(' ', '-')]

                # Space group number - direct number pattern (fallback)
                sg_match = re.search(r'SPACE GROUP.*?NUMBER:\s*(\d+)', content)
                if sg_match:
                    space_group = int(sg_match.group(1))

                # A slab names its layer group and the corresponding space
                # group ("CORRESPONDING SPACE GROUP N. 191", manual App. A.2,
                # p.421), as CRYSTALOptToD3's structure parser reads it; the
                # symbol search above finds nothing there, so a slab's path
                # was the P1 one.
                if 'SLAB CALCULATION' in content or 'SLAB GROUP' in content:
                    slab_sg = re.search(r'CORRESPONDING SPACE GROUP N\.\s*(\d+)', content)
                    if slab_sg:
                        space_group = int(slab_sg.group(1))
                    slab_lat = re.search(r'PLANE GROUP N\.\s*\d+\s*:\s*([A-Z])', content)
                    if slab_lat:
                        lattice_type = slab_lat.group(1)

                # crystal_system is the descriptive form (e.g. "cubic_fc") for
                # display only. Keep lattice_type as the single-letter centering:
                # get_band_path_from_symmetry / get_kpoint_coordinates_from_labels
                # call get_crystal_system_from_space_group themselves and that
                # function matches on the LETTER ('F', 'I', ...). The old code
                # overwrote lattice_type with the descriptive form, so every
                # centered lattice fell through to the simple/primitive k-path
                # (FCC treated as simple cubic, rhombohedral as hexagonal, ...).
                from d3_kpoints import get_crystal_system_from_space_group
                crystal_system = get_crystal_system_from_space_group(space_group, lattice_type)

                print(f"\nDetected space group: {space_group} "
                      f"({crystal_system}, {lattice_type}-centered)")
            except Exception as e:
                print(f"\nCould not read space group from output file: {e}")
                print("Using default P1")

        # Get appropriate path based on symmetry
        path_labels = get_band_path_from_symmetry(space_group, lattice_type)
        
        # Now ask which format to use for the path
        print("\nBand path format:")
        print("1: High-symmetry labels (CRYSTAL-compatible subset)")
        print("2: K-point vectors (fractional coordinates)")
        print("3: Literature path with vectors (comprehensive, includes non-CRYSTAL points)")
        print("4: SeeK-path full paths (extended Bravais lattice notation)")
        # SeeK-path for a 3D crystal, the label path for a slab or polymer
        # (d3_kpoints.default_band_format)
        default_format = default_band_format(out_file)
        format_choice = _nav_read(f"Select format (1-4) [{default_format}]: ", valid_set={"_choice_"}).strip()
        if not format_choice:
            format_choice = default_format
            # the default, not a choice: a slab or polymer of a batch gets the label path
            band_config["kpath_default"] = True
            if default_format == "1":
                print("  A slab or polymer: the high-symmetry label path (SeeK-path's is format 4)")
        
        if format_choice == "1":
            # Use labels - CRYSTAL-compatible subset
            band_config["path_method"] = "labels"
            band_config["shrink"] = 0
            band_config["path"] = path_labels
            band_config["auto_path"] = True  # Mark that this was auto-detected
        elif format_choice == "2":
            # Convert labels to k-point vectors
            band_config["path_method"] = "coordinates"
            band_config["path"] = "auto"  # Ensure auto-detection is triggered
            
            # Extract and process shrink factor (from the parent's SHRINK,
            # else from the cell fort.9 holds)
            base_name = Path(out_file).stem.split('_')[0] if out_file else "material"
            input_dir = Path(out_file).parent if out_file else Path.cwd()
            shrink = extract_and_process_shrink(out_file, base_name, input_dir, band_config)
            band_config["shrink"] = shrink
            band_config["auto_path"] = True  # Mark that this was auto-detected
            
            # Get k-point coordinates for the labels (returns fractional coordinates)
            frac_segments = get_kpoint_coordinates_from_labels(path_labels, space_group, lattice_type)

            # Scale fractional coordinates by shrink factor to get integers
            coord_segments, adjusted_shrink = scale_kpoint_segments(frac_segments, shrink)
            band_config["segments"] = coord_segments
            band_config["shrink"] = adjusted_shrink
            # Names the path in the title, as the config route does
            band_config["path_labels"] = path_labels
            print(f"\n✓ Converted path to k-point vectors (shrink={adjusted_shrink})")
        else:
            # SeeK-path full paths with comprehensive k-points
            band_config["path_method"] = "coordinates"
            
            # Extract and process shrink factor
            base_name = Path(out_file).stem.split('_')[0] if out_file else "material"
            input_dir = Path(out_file).parent if out_file else Path.cwd()
            shrink = extract_and_process_shrink(out_file, base_name, input_dir, band_config)
            band_config["shrink"] = shrink
            band_config["auto_path"] = True
            
            if format_choice == "3":
                # Literature path with comprehensive k-points
                band_config["literature_path"] = True
                band_config["path"] = "auto"  # Ensure auto-detection is triggered
                # Names the path in the title, as the config route does
                band_config["path_labels"] = path_labels
                
                # Get literature path based on space group and lattice type
                frac_segments = get_literature_kpath_vectors(space_group, lattice_type)
                if frac_segments:
                    # Scale fractional coordinates by shrink factor
                    coord_segments, adjusted_shrink = scale_kpoint_segments(frac_segments, shrink)
                    band_config["segments"] = coord_segments
                    band_config["shrink"] = adjusted_shrink
                    band_config["kpath_source"] = "literature"
                    print(f"✓ Using comprehensive literature k-path with {len(coord_segments)} segments (shrink={adjusted_shrink})")
                else:
                    # Fallback to standard path
                    print("\nLiterature path not available, using standard path")
                    frac_segments = get_kpoint_coordinates_from_labels(path_labels, space_group, lattice_type)
                    # Segment extremes are expressed in units of 1/ISS, so the
                    # fractional vectors must be scaled by the shrink factor
                    coord_segments, adjusted_shrink = scale_kpoint_segments(frac_segments, shrink)
                    band_config["segments"] = coord_segments
                    band_config["shrink"] = adjusted_shrink
            else:
                # format_choice == "4" - SeeK-path
                band_config["seekpath_full"] = True
                band_config["path"] = "auto"  # Ensure auto-detection is triggered

                # Get SeeK-path full path based on extended Bravais lattice (in fractional coordinates)
                result = get_seekpath_full_kpath(space_group, lattice_type, out_file)
                if result:
                    frac_segments, kpath_info = result
                    # Store k-path source info
                    if kpath_info.get("source") == "literature":
                        band_config["kpath_source"] = "literature"
                    elif kpath_info.get("source") == "default":
                        band_config["kpath_source"] = "default"
                    elif kpath_info.get("has_inversion"):
                        band_config["kpath_source"] = "seekpath_inv"
                    else:
                        band_config["kpath_source"] = "seekpath_noinv"

                    # For seekpath library results, use the library's calculated shrink factor
                    # This ensures exact integer representation of parametric k-points
                    if kpath_info.get('source') in ('seekpath_library', 'seekpath_data_parametric') and 'shrink_factor' in kpath_info:
                        shrink = kpath_info['shrink_factor']
                        print(f"  Using seekpath library's shrink factor: {shrink}")

                    # Scale fractional coordinates by shrink factor
                    coord_segments, adjusted_shrink = scale_kpoint_segments(frac_segments, shrink)
                    band_config["segments"] = coord_segments
                    band_config["shrink"] = adjusted_shrink
                    print(f"✓ Using SeeK-path full k-path with {len(coord_segments)} segments (shrink={adjusted_shrink})")
                else:
                    # Fallback to standard path
                    print("\nSeeK-path full path not available, using standard path")
                    frac_segments = get_kpoint_coordinates_from_labels(path_labels, space_group, lattice_type)
                    # Segment extremes are expressed in units of 1/ISS, so the
                    # fractional vectors must be scaled by the shrink factor
                    coord_segments, adjusted_shrink = scale_kpoint_segments(frac_segments, shrink)
                    band_config["segments"] = coord_segments
                    band_config["shrink"] = adjusted_shrink
    
    elif path_method == "2":
        # Template selection
        band_config["path_method"] = "labels"
        band_config["shrink"] = 0
        
        print("\nSelect crystal system:")
        print("1: Cubic")
        print("2: Hexagonal")
        print("3: Tetragonal")
        print("4: Orthorhombic")
        print("5: Monoclinic")
        print("6: Triclinic")
        
        system_choice = _nav_read("Select system (1-6) [1]: ", valid_set={"_choice_"}).strip() or "1"
        
        template_map = {
            "1": ["cubic", "cubic2", "cubic3"],
            "2": ["hexagonal", "hexagonal2"],
            "3": ["tetragonal", "tetragonal2"],
            "4": ["orthorhombic", "orthorhombic2", "orthorhombic3", "orthorhombic4"],
            "5": ["monoclinic", "monoclinic2"],
            "6": ["triclinic"]
        }
        
        templates = template_map.get(system_choice, ["cubic"])
        
        if len(templates) > 1:
            print("\nAvailable templates:")
            for i, template in enumerate(templates):
                t = BAND_TEMPLATES[template]
                print(f"{i+1}: {t['description']}")
                print(f"   Path: {' → '.join(t['path'])}")
            
            template_idx = _nav_int(f"\nSelect template (1-{len(templates)}) [1]: ", default=1) - 1
            template_name = templates[template_idx]
        else:
            template_name = templates[0]
        
        template = BAND_TEMPLATES[template_name]
        band_config["path"] = template["path"]
        band_config["template"] = template_name
        band_config["kpath_source"] = "template"
        print(f"\n✓ Using {template['description']} template")
    
    elif path_method == "3":
        # Custom labels
        band_config["path_method"] = "labels"
        band_config["shrink"] = 0
        band_config["kpath_source"] = "manual"
        
        # Display CRYSTAL-supported labels for detected crystal system
        from d3_kpoints import get_crystal_supported_labels, extract_lattice_parameters_from_output
        
        # Try to get crystal system info
        crystal_system = "CUBIC"  # Default
        lattice_type = "P"  # Default
        
        if out_file:
            try:
                # extract_lattice_parameters_from_output returns the cell
                # only (no crystal_system / lattice_type), so every output
                # was shown the cubic P labels; read the space group instead
                crystal_system, lattice_type = crystal_family_and_centring(out_file)
            except Exception:
                pass
        
        # Get supported labels
        supported_labels = get_crystal_supported_labels(crystal_system, lattice_type)
        
        print(f"\nCRYSTAL-supported k-point labels for {crystal_system} ({lattice_type} centering):")
        print("=" * 60)
        for label, coords in sorted(supported_labels.items()):
            print(f"  {label}: {coords}")
        print("=" * 60)
        
        print("\nEnter band path as space-separated labels (e.g., GAMMA X M GAMMA):")
        print("Note: Use GAMMA for the zone center. Labels are case-sensitive.")
        print("      Use | for discontinuous paths (e.g., GAMMA X M GAMMA | R X)")
        print("      This creates separate segments without connecting GAMMA to R.")
        path_str = input("Path: ").strip()
        band_config["path"] = path_str.split() if path_str else ["GAMMA", "X", "M", "GAMMA"]
    
    else:
        # Manual k-point entry
        band_config["path_method"] = "manual"
        band_config["kpath_source"] = "fractional"
        
        # Ask about mixed segments
        use_mixed = yes_no_prompt("\nUse mixed label/coordinate segments?", "no")
        
        if use_mixed:
            # Mixed path - each segment can be labels or coordinates
            n_segments = _nav_int("Number of path segments: ")
            segments = []
            band_config["shrink"] = 16  # Default, will be adjusted if needed
            
            for i in range(n_segments):
                print(f"\nSegment {i+1}:")
                seg_type = _nav_read("Type (L)abels or (C)oordinates [L]: ", valid_set={"_choice_"}).strip().upper() or "L"
                
                if seg_type == "L":
                    labels = input("Enter two labels (e.g., G X): ").strip().upper()
                    segments.append(labels)
                    band_config["shrink"] = 0  # Labels mode
                else:
                    print("Enter coordinates (fractional or integer):")
                    coords = input("Six numbers (x1 y1 z1 x2 y2 z2): ").strip()
                    segments.append(coords)
                    # Check if we need shrink
                    try:
                        vals = [float(x) for x in coords.split()]
                        if all(v == int(v) for v in vals):
                            # Integer coordinates, need appropriate shrink
                            max_val = max(abs(int(v)) for v in vals)
                            if max_val > band_config.get("shrink", 16):
                                band_config["shrink"] = max_val * 2
                    except:
                        pass
            
            band_config["manual_segments"] = segments
            
        else:
            # All segments same type
            print("\nSegment format:")
            print("1: All segments use labels")
            print("2: All segments use fractional coordinates")
            format_choice = _nav_read("Select format (1-2) [1]: ", valid_set={"_choice_"}).strip() or "1"
            
            n_segments = _nav_int("Number of path segments: ")
            segments = []
            
            if format_choice == "1":
                # All segments use labels
                band_config["shrink"] = 0  # Use labels mode
                for i in range(n_segments):
                    segment = input(f"Segment {i+1} (e.g., 'G X'): ").strip().upper()
                    segments.append(segment)
            else:
                # All segments use coordinates
                # Try to extract shrink from D12 file if available
                base_name = Path(out_file).stem.split('_')[0] if out_file else "material"
                input_dir = Path(out_file).parent if out_file else Path.cwd()
                default_shrink = extract_and_process_shrink(out_file, base_name, input_dir, {})
                
                shrink_input = _nav_read(f"Shrink factor for k-points [{default_shrink}]: ", valid_set={"_choice_"}).strip()
                shrink = int(shrink_input) if shrink_input else default_shrink
                
                # Ensure it's even for cleaner paths
                if shrink % 2 == 1:
                    shrink = shrink + 1
                    print(f"✓ Using shrink factor {shrink} (rounded to even number)")
                
                band_config["shrink"] = shrink
                
                print("\nEnter segments as fractional coordinates (will be scaled by shrink factor):")
                frac_segments = []
                for i in range(n_segments):
                    segment_str = input(f"Segment {i+1}: ").strip()
                    try:
                        coords = [float(x) for x in segment_str.split()]
                        if len(coords) == 6:
                            frac_segments.append(coords)
                        else:
                            print("Warning: Expected 6 numbers, skipping segment")
                    except ValueError:
                        print("Warning: Invalid input, skipping segment")
                # Scaled together: a shrink the scaling has to raise for one
                # point is the shrink of every segment (and of the header).
                if frac_segments:
                    scaled, band_config["shrink"] = scale_kpoint_segments(frac_segments, shrink)
                    segments = [" ".join(str(v) for v in seg) for seg in scaled]

            # The BAND writer reads a manual path's segments as lines, as the
            # mixed route stores them.
            band_config["manual_segments"] = segments
    
    # Number of k-points
    n_points = _nav_int("\nTotal number of k-points along path [1000]: ", default=1000)
    band_config["n_points"] = n_points
    
    # Band range
    print("\nBand range selection:")
    print("1: All bands")
    print("2: Around Fermi level (adaptive per material)")
    print("3: Custom range")

    band_range = _nav_read("Select option (1-3) [1]: ", valid_set={"_choice_"}).strip() or "1"

    if band_range == "1":
        # This will be set per material in CRYSTALOptToD3.py based on n_ao
        band_config["bands"] = "auto"
        band_config["auto_bands"] = True
        band_config["first_band"] = 1
        band_config["last_band"] = None  # Will be set to n_ao
    elif band_range == "2":
        # Around Fermi: store the OFFSETS, not specific band numbers
        # This allows each material to calculate its own range based on its Fermi level
        band_config["bands"] = "fermi"
        band_config["bands_below_fermi"] = _nav_int("Number of bands below Fermi [500]: ", default=500)
        band_config["bands_above_fermi"] = _nav_int("Number of bands above Fermi [600]: ", default=600)
        print(f"\n✓ Will select {band_config['bands_below_fermi']} bands below and {band_config['bands_above_fermi']} bands above Fermi for each material")
    else:
        # Custom range
        first = _nav_int("First band [1]: ", default=1)
        last = _nav_int("Last band [all]: ", default=0)
        band_config["first_band"] = first
        band_config["last_band"] = last if last > 0 else None
    
    # Output options
    band_config["plot"] = True  # Always generate plot
    band_config["print_eigenvalues"] = yes_no_prompt("\nPrint eigenvalues to output?", "no")
    
    return band_config


def configure_doss_calculation(out_file: Optional[str] = None) -> Dict[str, Any]:
    """Configure DOSS calculation settings interactively."""
    print("\n=== DENSITY OF STATES CONFIGURATION ===")
    
    doss_config = {}
    
    # Projection type
    print("\nDOS projection type:")
    print("1: Total DOS only")
    print("2: Projected DOS by element (all orbitals)")
    print("3: Projected DOS by element and orbital type (with totals)")
    print("4: Projected DOS by orbital type only (no element totals)")
    print("5: Projected DOS on specific atoms")
    print("6: Manual projections (custom combinations)")
    
    proj_type = _nav_read("Select projection type (1-6) [3]: ", valid_set={"_choice_"}).strip() or "3"
    
    doss_config["projection_type"] = int(proj_type)
    
    if proj_type == "1":
        # Total DOS only
        doss_config["npro"] = 0
        doss_config["project_orbital_types"] = False
    elif proj_type == "2":
        # By element only
        doss_config["project_orbital_types"] = True
        doss_config["element_only"] = True
    elif proj_type == "3":
        # By element and orbital type with totals (default)
        doss_config["project_orbital_types"] = True
        doss_config["element_only"] = False
        doss_config["include_element_totals"] = True
    elif proj_type == "4":
        # By orbital type only (no element totals)
        doss_config["project_orbital_types"] = True
        doss_config["element_only"] = False
        doss_config["include_element_totals"] = False
    elif proj_type == "5":
        # Specific atoms
        doss_config["project_orbital_types"] = False
        print("\nAtom selection:")
        print("1: Project on all atoms")
        print("2: Project on specific atoms")
        
        atom_choice = _nav_read("Select (1-2) [1]: ", valid_set={"_choice_"}).strip() or "1"
        
        if atom_choice == "1":
            doss_config["project_all_atoms"] = True
        else:
            atom_list = input("Enter atom numbers (e.g., 1 3 5): ").strip()
            if atom_list:
                doss_config["project_atoms"] = [int(a) for a in atom_list.split()]
            else:
                doss_config["project_atoms"] = [1]
    else:
        # Manual projections (option 6)
        doss_config["project_orbital_types"] = False
        doss_config["manual_projections"] = True
        
        # Get atom information from output file if available
        if out_file and os.path.exists(out_file):
            # Parse the output file for atom information
            n_atoms = 0
            atom_elements = []
            
            with open(out_file, 'r') as f:
                content = f.read()
            
            # Extract number of atoms
            atoms_match = re.search(r'ATOMS IN THE UNIT CELL:\s+(\d+)', content)
            if atoms_match:
                n_atoms = int(atoms_match.group(1))
            else:
                atoms_match = re.search(r'ATOMS IN THE ASYMMETRIC UNIT\s+(\d+)', content)
                if atoms_match:
                    n_atoms = int(atoms_match.group(1))
            
            # Extract atom elements. CRYSTAL prints symbols ALL CAPS ("TI", "PB"),
            # so the second letter must allow uppercase; "[A-Z][a-z]?" matched
            # nothing for two-letter elements (the trailing \s+ hit the second
            # element letter), leaving atom_elements empty.
            # CRYSTAL reprints the geometry once per optimization step, so a
            # findall over the whole file repeats the atom list N times. The
            # leading atom index runs 1..N within one listing and resets on the
            # next block; collect only the first contiguous run so the count
            # matches n_atoms (the [:n_atoms] consumers below were already
            # masking this, but len(atom_elements) is now correct too).
            atom_pattern = re.compile(r'^\s*(\d+)\s+[TF]\s+\d+\s+([A-Z][A-Za-z]?)\s+', re.MULTILINE)
            atom_elements = []
            _last_idx = 0
            for _m in atom_pattern.finditer(content):
                _idx = int(_m.group(1))
                if _idx <= _last_idx:
                    break
                atom_elements.append(_m.group(2))
                _last_idx = _idx
            
            # Parse basis set to show available orbitals
            data_list, n_ao = parse_basis_set_info(out_file)
            total_shells = {}
            
            if data_list:
                atoms, atoms_shells = get_atoms_and_shells(data_list)
                atoms_index = get_unique_atoms(atoms)
                total_shells = get_shells(atoms, atoms_shells)
            
            if n_atoms > 0:
                print(f"\nSystem contains {n_atoms} atoms:")
                if atom_elements:
                    for i, elem in enumerate(atom_elements[:n_atoms]):
                        print(f"  Atom {i+1}: {elem}")
                else:
                    print(f"  {n_atoms} atoms (element types not determined)")
                
                # Show available orbitals for each element
                if total_shells:
                    print("\nAvailable orbital projections by element:")
                    for element in sorted(set(atom_elements[:n_atoms])):
                        if element in total_shells:
                            available = []
                            for shell_type in ["S", "P", "SP", "D", "F"]:
                                if total_shells[element].get(shell_type):
                                    available.append(shell_type)
                            if available:
                                print(f"  {element}: {', '.join(available)}, all")
                
                print("\nInput format examples:")
                print("  - Atom projection: '1' or '1C' (for atom 1)")
                print("  - Element total: 'C all' or 'C' (all orbitals of element C)")
                print("  - Orbital-specific: 'C S', 'O P', etc.")
                print("\nNote: The script will convert your input to the proper DOSS format.")
        
        # Helper function to validate projection
        def validate_projection(spec: str, total_shells: Dict, atom_elements: List[str], n_atoms: int) -> Tuple[bool, str]:
            """Validate a projection specification and return (is_valid, error_message)."""
            spec = spec.strip()
            if not spec:
                return False, "Empty specification"
            
            # Check if it's an atom projection (starts with a number)
            if spec[0].isdigit():
                # Extract atom number
                atom_num_match = re.match(r'^(\d+)', spec)
                if atom_num_match:
                    atom_num = int(atom_num_match.group(1))
                    if atom_num < 1 or atom_num > n_atoms:
                        return False, f"Atom number {atom_num} out of range (1-{n_atoms})"
                    return True, ""
            
            # Otherwise, it's an element/orbital projection
            parts = spec.split()
            if not parts:
                return False, "Empty specification"
            
            element = parts[0].upper()
            
            # Check if element exists in the system
            if element not in total_shells:
                available_elements = sorted(total_shells.keys())
                return False, f"Element {element} not found in system. Available: {', '.join(available_elements)}"
            
            # If just element name or "element all", it's valid
            if len(parts) == 1 or (len(parts) == 2 and parts[1].lower() == "all"):
                return True, ""
            
            # Specific orbital type
            if len(parts) == 2:
                orbital_type = parts[1].upper()
                
                # Check if orbital type exists for this element
                if orbital_type in total_shells[element] and total_shells[element][orbital_type]:
                    return True, ""
                else:
                    # Check for combined SP
                    if orbital_type == "SP" and "S" in total_shells[element] and "P" in total_shells[element]:
                        return True, ""
                    else:
                        available = [o for o in ["S", "P", "SP", "D", "F"] if o in total_shells[element] and total_shells[element][o]]
                        return False, f"Orbital type {orbital_type} not available for element {element}. Available: {', '.join(available)}"
            
            return False, f"Invalid projection format: '{spec}'"
        
        doss_config["manual_projection_specs"] = []
        valid_projections = []
        
        print("\nEnter projections (press Enter without input to stop):")
        i = 1
        while True:
            print(f"\nProjection {i}:")
            proj_spec = input("Enter projection specification: ").strip()
            
            if not proj_spec:
                if i == 1:
                    print("At least one projection is required!")
                    continue
                else:
                    break
            
            # Validate the projection
            is_valid, error_msg = validate_projection(proj_spec, total_shells, atom_elements, n_atoms)
            
            if is_valid:
                valid_projections.append(proj_spec)
                i += 1
            else:
                print(f"  ERROR: {error_msg}")
                print("  Please enter a valid projection.")
        
        doss_config["manual_projection_specs"] = valid_projections
        doss_config["npro"] = len(valid_projections)
        print(f"\nTotal valid projections: {len(valid_projections)}")
    
    # Energy window or band range
    print("\nEnergy/band range:")
    print("1: All bands")
    print("2: Energy window around Fermi level")
    print("3: Specific band range")
    
    range_choice = _nav_read("Select option (1-3) [1]: ", valid_set={"_choice_"}).strip() or "1"
    
    if range_choice == "2":
        # Energy window - collect in eV but convert to Ha
        EV_TO_HA = EV_TO_HARTREE
        try:
            from d3_config import DOSS_DEFAULT_WINDOW_EV as _win
        except ImportError:
            from Crystal_d3.d3_config import DOSS_DEFAULT_WINDOW_EV as _win
        bmi_ev = _nav_float(f"Lower energy limit (eV below Fermi) [{_win[0]:g}]: ", default=_win[0])
        bma_ev = _nav_float(f"Upper energy limit (eV above Fermi) [{_win[1]:g}]: ", default=_win[1])
        # Convert to Hartree
        bmi_ha = bmi_ev * EV_TO_HA
        bma_ha = bma_ev * EV_TO_HA
        doss_config["energy_window"] = (bmi_ha, bma_ha)
    elif range_choice == "3":
        # Band range
        first = _nav_int("First band [1]: ", default=1)
        last = _nav_int("Last band [all]: ", default=-1)
        if last > 0:
            doss_config["band_range"] = (first, last)
    
    # Energy resolution: the step of the energy grid, or a number of points (d3_config.doss_npts)
    doss_config.update(_ask_doss_resolution(out_file, doss_config))
    
    # Output format
    print("\nOutput format:")
    print("1: CRYSTAL fort.25 format")
    print("2: DOSS.DAT format")
    
    out_format = _nav_read("Select format (1-2) [2]: ", valid_set={"_choice_"}).strip() or "2"
    doss_config["output_format"] = int(out_format) if out_format in ["1", "2"] else 2
    
    # Polynomial order for DOS smoothing
    print("\nPolynomial order (NPOL) for DOS smoothing:")
    print("  Suggested: 10-18 (higher NPOL = smoother DOS)")
    print("  Note: Higher IS values support higher NPOL values")
    npol = _nav_int("NPOL value [14]: ", default=14)
    doss_config["npol"] = npol
    
    doss_config["print_integrated"] = yes_no_prompt("\nPrint integrated DOS to output?", "no")
    
    return doss_config


def configure_transport_calculation(out_file: Optional[str] = None) -> Dict[str, Any]:
    """Configure BOLTZTRA transport calculation settings."""
    print("\n=== TRANSPORT PROPERTIES CONFIGURATION ===")
    
    transport_config = {}
    
    # Temperature range
    print("\nTemperature range:")
    t_min = _nav_float("Minimum temperature (K) [100]: ", default=100)
    t_max = _nav_float("Maximum temperature (K) [800]: ", default=800)
    t_step = _nav_float("Temperature step (K) [50]: ", default=50)
    transport_config["temperature_range"] = (t_min, t_max, t_step)
    
    # Chemical potential range
    print("\nChemical potential range:")
    
    # E_F: the SCF's, else a BAND or DOSS run of it (never CRYSTAL's -1.0 placeholder): MURANGE and
    # TDFRANGE hold absolute eV, so a range relative to E_F (or the VBM, the top of the occupied bands)
    # needs it. Without an output (settings for many outputs) a relative range is kept relative and
    # placed per material when its deck is written.
    fermi_energy_ev = None
    fermi_au, why = trusted_fermi_au(out_file) if out_file else (None, "")
    if fermi_au is not None:
        fermi_energy_ev = fermi_au * HARTREE_TO_EV
        print(f"Detected Fermi energy: {fermi_energy_ev:.3f} eV" + (f" ({why})" if why else ""))

    if out_file and fermi_energy_ev is None:
        print(f"\nWARNING: {why[0].upper() + why[1:] if why else 'No Fermi energy'}.")
        print("The chemical potentials cannot be placed relative to E_F: give them as absolute energies "
              "(eV, CRYSTAL's scale), e.g. around the VBM of a BAND run.")
        mu_min = _nav_float("Minimum μ (eV, absolute): ")
        mu_max = _nav_float("Maximum μ (eV, absolute): ")
        mu_step = _nav_float("μ step (eV) [0.01]: ", default=0.01)
        transport_config["mu_range"] = (mu_min, mu_max, mu_step)
        transport_config["mu_reference"] = "absolute"
        mu_option = None
    else:
        print("\nChemical potential options:")
        print("1: Relative to Fermi energy (automatic)")
        print("2: Relative to VBM (manual)")
        print("3: Absolute values")
        mu_option = _nav_read("Select option (1-3) [1]: ", valid_set={"_choice_"}).strip() or "1"

    if mu_option in ("1", "2"):
        # Relative to E_F (an insulator's E_F from these sources is its VBM)
        print("\nRange relative to Fermi energy:" if mu_option == "1" else "\nRange relative to the VBM:")
        mu_min_rel = _nav_float("Minimum μ (eV below Fermi) [-2.0]: ", default=-2.0)
        mu_max_rel = _nav_float("Maximum μ (eV above Fermi) [2.0]: ", default=2.0)
        mu_step = _nav_float("μ step (eV) [0.01]: ", default=0.01)
        if fermi_energy_ev is not None:
            # Store as absolute values
            transport_config["mu_range"] = (fermi_energy_ev + mu_min_rel,
                                          fermi_energy_ev + mu_max_rel,
                                          mu_step)
            transport_config["mu_reference"] = "fermi"
            transport_config["mu_relative_range"] = (mu_min_rel, mu_max_rel)
        else:
            # no output: kept relative, placed around each material's E_F by the deck writer
            transport_config["mu_range"] = (mu_min_rel, mu_max_rel, mu_step)
            transport_config["mu_reference"] = "fermi" if mu_option == "1" else "vbm"
            if mu_option == "1":
                transport_config["mu_range_type"] = "auto_fermi"
    elif mu_option == "3":
        print("\nAbsolute chemical potential range:")
        mu_min = _nav_float("Minimum μ (eV) [-2.0]: ", default=-2.0)
        mu_max = _nav_float("Maximum μ (eV) [2.0]: ", default=2.0)
        mu_step = _nav_float("μ step (eV) [0.01]: ", default=0.01)
        transport_config["mu_range"] = (mu_min, mu_max, mu_step)
        transport_config["mu_reference"] = "absolute"

    # Transport distribution function range: TDFRANGE holds absolute eV, and the distribution must
    # cover the chemical potentials, so with μ relative to E_F it is relative to E_F too
    relative = transport_config.get("mu_reference") in ("fermi", "vbm")
    print("\nTransport distribution function range" + (" (eV around E_F):" if relative else " (eV, absolute):"))
    tdf_min = _nav_float("Minimum energy (eV) [-5.0]: ", default=-5.0)
    tdf_max = _nav_float("Maximum energy (eV) [5.0]: ", default=5.0)
    tdf_step = _nav_float("Energy step (eV) [0.01]: ", default=0.01)
    if relative and fermi_energy_ev is not None:
        transport_config["tdf_range"] = (fermi_energy_ev + tdf_min, fermi_energy_ev + tdf_max, tdf_step)
        transport_config["tdf_relative_range"] = (tdf_min, tdf_max)
    else:
        transport_config["tdf_range"] = (tdf_min, tdf_max, tdf_step)
    
    # The k mesh of BOLTZTRA's NEWK (d3_config.transport_mesh)
    transport_config.update(_ask_transport_k(out_file))

    # Relaxation time
    tau = _nav_float("\nRelaxation time (fs) [10]: ", default=10)
    transport_config["relaxation_time"] = tau
    
    # Smearing
    use_smear = yes_no_prompt("\nUse smearing for metallic systems?", "no")
    if use_smear:
        smear = _nav_float("Smearing width (eV) [0.025]: ", default=0.025)
        transport_config["smearing"] = smear
        
        print("\nSmearing type:")
        print("0: Fermi-Dirac")
        print("1: Gaussian")
        print("2: Methfessel-Paxton")
        smear_type = _nav_int("Select type (0-2) [0]: ", default=0)
        transport_config["smearing_type"] = smear_type
    
    return transport_config


def configure_charge_density_calculation() -> Dict[str, Any]:
    """Configure charge density calculation (ECH3/ECHG)."""
    print("\n=== CHARGE DENSITY CONFIGURATION ===")
    
    charge_config = {}
    
    # Calculation type
    print("\nCharge density type:")
    print("1: ECH3 - 3D charge density grid")
    print("2: ECHG - 2D charge density map")
    
    calc_type = _nav_read("Select type (1-2) [1]: ", valid_set={"_choice_"}).strip() or "1"
    
    if calc_type == "1":
        charge_config["type"] = "ECH3"
        
        # Number of grid points
        n_points = _nav_int("\nNumber of grid points per direction [100]: ", default=100)
        charge_config["n_points"] = n_points
        
        # For lower dimensional systems
        print("\nFor slabs/polymers/molecules, define non-periodic directions:")
        print("1: Use automatic SCALE")
        print("2: Define explicit RANGE")
        
        range_choice = _nav_read("Select option (1-2) [1]: ", valid_set={"_choice_"}).strip() or "1"
        
        if range_choice == "2":
            charge_config["use_range"] = True
        else:
            charge_config["use_range"] = False
            scale = _nav_float("Scale factor for non-periodic directions [3.0]: ", default=3.0)
            charge_config["scale"] = scale
    
    else:
        charge_config["type"] = "ECHG"
        
        # Derivative order - CRYSTAL defines IDER 0 and 1 only
        print("\nDerivative order:")
        print("0: Charge density")
        print("1: Charge density and its gradient")
        
        deriv = _nav_int("Select order (0-1) [0]: ", default=0)
        if deriv not in (0, 1):
            print("Warning: only IDER=0 and IDER=1 are defined for ECHG, using 0")
            deriv = 0
        charge_config["derivative_order"] = deriv
        
        # MAPNET's NPY, "number of points on the B-A segment" (manual 26772).
        # Collected here rather than only in the writer so a saved ECHG config
        # can be loaded again: validate_d3_config requires n_points for CHARGE,
        # so a config without it was rejected on reload.
        charge_config["n_points"] = _nav_int(
            "\nNumber of points along the B-A segment [50]: ", default=50)

        # The map plane (MAPNET records, mandatory for ECHG, manual p. 324) is
        # asked when the deck is written: its points A, B, C belong to each file.
        charge_config["need_map_points"] = True
    
    return charge_config


def configure_potential_calculation() -> Dict[str, Any]:
    """Configure electrostatic potential calculation (POT3/POTC)."""
    print("\n=== ELECTROSTATIC POTENTIAL CONFIGURATION ===")
    
    potential_config = {}
    
    # Calculation type
    print("\nPotential calculation type:")
    print("1: POT3 - 3D potential grid")
    print("2: POTC - Potential at specific points")
    
    calc_type = _nav_read("Select type (1-2) [1]: ", valid_set={"_choice_"}).strip() or "1"
    if calc_type not in ("1", "2"):
        # an answer meant for another question (a 1.1.5-era stream's "y" to
        # the removed map-plane question) would shift every later one
        print(f"Error: the potential type is 1 (POT3) or 2 (POTC), not '{calc_type}'; "
              "no D3 file was written.", file=sys.stderr)
        raise SystemExit(1)
    
    if calc_type == "1":
        potential_config["type"] = "POT3"
        
        # Grid settings
        n_points = _nav_int("\nNumber of grid points per direction [100]: ", default=100)
        potential_config["n_points"] = n_points
        
        itol = _nav_int("Bipolar expansion tolerance (5-8) [5]: ", default=5)
        potential_config["itol"] = itol
        
        # For lower dimensional systems
        print("\nFor slabs/polymers/molecules, define non-periodic directions:")
        print("1: Use automatic SCALE")
        print("2: Define explicit RANGE")
        
        range_choice = _nav_read("Select option (1-2) [1]: ", valid_set={"_choice_"}).strip() or "1"
        
        if range_choice == "2":
            potential_config["use_range"] = True
        else:
            potential_config["use_range"] = False
            scale = _nav_float("Scale factor for non-periodic directions [3.0]: ", default=3.0)
            potential_config["scale"] = scale
    
    else:
        potential_config["type"] = "POTC"
        
        # ICA parameter - ICA=1 is not implemented in CRYSTAL, the plane and
        # volume averages are ICA=2 and ICA=3 (2D systems only)
        print("\nCalculation mode:")
        print("0: Potential at specific points")
        print("2: Plane-averaged potential (2D only)")
        print("3: Volume-averaged potential (2D only)")
        
        ica = _nav_int("Select mode (0, 2 or 3) [0]: ", default=0)
        if ica not in (0, 2, 3):
            print("Warning: ICA=1 is not implemented in CRYSTAL, using 0")
            ica = 0
        potential_config["ica"] = ica
        
        if ica == 0:
            # Points specification
            print("\nDefine points:")
            print("1: At atomic positions")
            print("2: At custom points")
            print("3: Read from file")
            
            point_choice = _nav_read("Select option (1-3) [1]: ", valid_set={"_choice_"}).strip() or "1"
            
            if point_choice == "1":
                potential_config["at_atoms"] = True
                potential_config["all_atoms"] = yes_no_prompt("Calculate at all atoms?", "yes")
                if not potential_config["all_atoms"]:
                    atom_list = input("Enter atom numbers: ").strip()
                    potential_config["atoms"] = [int(a) for a in atom_list.split()]
            elif point_choice == "2":
                n_points = _nav_int("Number of points: ", default=1)
                potential_config["n_points"] = n_points
                potential_config["custom_points"] = True
            else:
                # Store the plain count; the writer applies the NPU < 0 flag
                # that makes CRYSTAL read the coordinates from POTC.INP
                potential_config["n_points"] = _nav_int("Number of points in file: ", default=1)
                potential_config["read_file"] = True
        
        else:
            # Plane/volume averaging
            z_min = _nav_float("Z minimum (bohr) [0]: ", default=0)
            z_max = _nav_float("Z maximum (bohr) [10]: ", default=10)
            potential_config["z_range"] = (z_min, z_max)
            
            n_planes = _nav_int("Number of planes [100]: ", default=100)
            potential_config["n_planes"] = n_planes
            
            if ica == 3:
                # ZD, half thickness of the averaging volume (ICA=3 only)
                thickness = _nav_float("Half thickness of the volume (bohr) [0.5]: ", default=0.5)
                potential_config["slice_thickness"] = thickness
    
    return potential_config


_KCHOICE = None


def orbitals_kchoice():
    """mace/orbitals/kchoice.py (standard library only): from the mace package,
    else loaded by path, for a run of these scripts without it installed."""
    global _KCHOICE
    if _KCHOICE is None:
        try:
            from mace.orbitals import kchoice as mod
        except ImportError:
            import importlib.util as ilu
            path = Path(__file__).resolve().parent.parent / "mace" / "orbitals" / "kchoice.py"
            spec = ilu.spec_from_file_location("_mace_orbitals_kchoice", path)
            mod = ilu.module_from_spec(spec)
            sys.modules[spec.name] = mod      # its dataclasses look their module up
            spec.loader.exec_module(mod)
        _KCHOICE = mod
    return _KCHOICE


def orbitals_file_warnings(kc, is_: Optional[int], dimensionality: int, out_text: str) -> List[str]:
    """The file count/size warnings for an ORBITALS mesh (HPCC 1M-file quota)."""
    nbf = re.search(r'NUMBER OF AO\s+(\d+)', out_text or "")
    spin = "UNRESTRICTED OPEN SHELL" in (out_text or "")
    est = kc.estimate_orbital_files(is_, dimensionality, nbf=int(nbf.group(1)) if nbf else None,
                                    spin=spin, scf_text=out_text)
    return est.warnings


def configure_orbitals_calculation(out_file: Optional[str] = None) -> Dict[str, Any]:
    """Configure an ORBITALS (Molden export) calculation: which k points.

    CRYSTAL writes every orbital at every k point of the NEWK mesh; the
    question is the mesh. ICAR (no effect on the files) and ILOC (always 0:
    Wannier functions go through the Wannier90 / lcao2wannier route) are not
    asked.
    """
    kc = orbitals_kchoice()
    print("\n=== ORBITALS (MOLDEN FILES) CONFIGURATION ===")
    text = ""
    if out_file:
        try:
            text = Path(out_file).read_text(errors="replace")
        except OSError:
            text = ""
    dim = kc.dimensionality_of(text) if text else None
    dim = 3 if dim is None else dim
    config: Dict[str, Any] = {"calculation_type": "ORBITALS", "kmesh": "band_edges"}
    if dim == 0:
        print("Molecule: ORBITALS writes one Molden file with every molecular orbital "
              "(no k points to choose).")
        return config

    if text:
        preview = kc.choose_kmesh(text, "band_edges", dimensionality=dim)
        for edge in preview.edges:
            print(f"  {edge.describe()}")
        print(f"  Band edges: IS {preview.is_}")
        for note in preview.notes:
            print(f"  Note: {note}")

    print("\nk points to write:")
    print("1: Γ and the real k points of the band edges (default)")
    print("2: Γ only (IS 1, one file)")
    print("3: All real k points (IS 2)")
    print("4: Custom shrinking factor IS")
    choice = _nav_read("Select k points (1-4) [1]: ", valid_set={"_choice_"}).strip() or "1"
    while choice not in ("1", "2", "3", "4"):
        print(f"  '{choice}' is not a valid choice; enter a number 1-4.")
        choice = _nav_read("Select k points (1-4) [1]: ", valid_set={"_choice_"}).strip() or "1"
    config["kmesh"] = {"1": "band_edges", "2": "gamma", "3": "all_real", "4": "custom"}[choice]

    is_ = {"gamma": 1, "all_real": 2}.get(config["kmesh"])
    if config["kmesh"] == "band_edges" and text:
        is_ = preview.is_
    if config["kmesh"] == "custom":
        if dim >= 2:
            print(f"  A {dim}D structure: only IS 1 and IS 2 are usable (IS 3 and up has complex "
                  f"k points, whose {dim}D Molden coefficients fail validation).")
        else:
            print("  1D: complex k points (IS 3 and up) were validated and can be used.")
        while True:
            is_ = _nav_int("Shrinking factor IS [2]: ", default=2)
            problem = kc.custom_is_problem(is_, dim)
            if not problem:
                break
            print(f"  {problem}")
        config["is"] = is_

    if is_ is not None:
        for warning in orbitals_file_warnings(kc, is_, dim, text):
            print(f"  Warning: {warning}")
    return config


def configure_d3_calculation(calc_type: str, out_file: Optional[str] = None, 
                           save_prompt: bool = False) -> Dict[str, Any]:
    """Main function to configure any D3 calculation type.
    
    Args:
        calc_type: Type of calculation (BAND, DOSS, etc.)
        out_file: Path to output file for context
        save_prompt: Whether to prompt user to save configuration
        
    Returns:
        Configuration dictionary
    """
    
    calc_type = calc_type.upper()
    
    if calc_type == "BAND":
        config = configure_band_calculation(out_file)
    elif calc_type == "DOSS":
        config = configure_doss_calculation(out_file)
    elif calc_type == "TRANSPORT":
        config = configure_transport_calculation(out_file)
    elif calc_type == "ORBITALS":
        config = configure_orbitals_calculation(out_file)
    elif calc_type in ["CHARGE", "ECH3", "ECHG"]:
        config = configure_charge_density_calculation()
    elif calc_type in ["POTENTIAL", "POT3", "POTC"]:
        config = configure_potential_calculation()
    elif calc_type == "CHARGE+POTENTIAL":
        # Configure both charge and potential settings
        print("\n--- Charge Density Settings ---")
        charge_config = configure_charge_density_calculation()
        
        print("\n--- Electrostatic Potential Settings ---")
        potential_config = configure_potential_calculation()
        
        config = {
            "charge_config": charge_config,
            "potential_config": potential_config
        }
    else:
        raise ValueError(f"Unknown calculation type: {calc_type}")
    
    # Add calculation type to config if not present
    if "calculation_type" not in config:
        config["calculation_type"] = calc_type
    
    # Optional save prompt
    if save_prompt:
        from d3_config import save_d3_options_prompt
        save_d3_options_prompt(config)
    
    return config