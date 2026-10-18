# CRYSTAL Input File Generation Suite (D12 Creation Scripts)

A suite of Python scripts for generating and managing CRYSTAL quantum chemistry input files (.d12) from CIF files and optimized geometries. This modular system supports the entire CRYSTAL calculation workflow including geometry optimization, single point calculations, and advanced frequency/phonon analyses.

---

## Overview

This suite represents a complete refactoring of CRYSTAL input generation tools, designed with:
- **Modularity**: Each calculation type has its own configuration module
- **Consistency**: Unified behavior across all scripts
- **Extensibility**: Easy to add new calculation types and features
- **Integration**: Full compatibility with the workflow automation system

---

## Key Features

- **CIF to D12 Conversion**: Automated conversion with interactive configuration
- **Multiple Calculation Types**: OPT, SP, FREQ with extensive customization
- **JSON Configuration Support**: Save and reuse settings across multiple calculations
- **Advanced Frequency Analysis**: IR/Raman spectra, phonon dispersion, anharmonic corrections
- **Ghost Atom Support**: Automated insertion for surface calculations
- **Basis Set Management**: Internal and external basis sets with ECP support
- **DFT Functional Library**: Wide range of functionals including dispersion corrections
- **Database Integration**: Automatic extraction and storage of all calculation parameters
- **Workflow Automation**: Integrates with the enhanced queue manager

---

## Architecture

### Modular Design

The refactored system separates concerns into focused modules:

```
Core Scripts:
├── NewCifToD12.py         # CIF → D12 conversion (main entry point)
├── CRYSTALOptToD12.py     # Extract optimized geometry → new D12
├── Ghosts/create_d12_w-ghosts.py # Automatic ghost atom insertion
└── Ghosts/manual_create_d12_w-ghosts.py # Manual ghost atom insertion

Configuration Modules:
├── d12_calc_basic.py      # Basic calculation settings
├── d12_calc_freq.py       # Frequency calculation configuration
├── d12_calc_opt.py        # Optimization settings (future)
└── d12_calc_sp.py         # Single point settings (future)

Support Modules:
├── d12_constants.py       # Constants, basis sets, functionals
├── d12_parsers.py         # Output/input file parsers
├── d12_writer.py          # D12 file writing utilities
└── d12_interactive.py     # Interactive prompts and utilities
```

### Benefits of Refactoring

- **Code Reduction**: ~400 lines of duplicated code removed
- **Single Source of Truth**: Each configuration exists in one place
- **Easier Maintenance**: Updates propagate to all scripts automatically
- **Better Testing**: Modular components can be tested independently
- **Enhanced Consistency**: Identical behavior across all entry points

---

## Main Scripts

### `NewCifToD12.py`

**Purpose:** Primary entry point for converting CIF files to CRYSTAL D12 input files.

**Features:**

- **Input Modes**: SLAB (2D), MOLECULE (0D), CRYSTAL (3D), and POLYMER (1D)
- **Interactive Configuration**: Step-by-step guided setup with sensible defaults
- **Batch Processing**: Convert entire directories of CIF files
- **Calculation Types**: 
  - Single Point (SP)
  - Geometry Optimization (OPT) with multiple convergence options
  - Frequency/Phonon calculations (FREQ) with extensive options
- **DFT Support**: Full functional library including hybrids, meta-GGA, dispersion
- **Basis Sets**: Internal (POB, Peintinger) and external with automatic ECP handling
- **Advanced Features**:
  - Spin polarization and initial spin states
  - K-point mesh optimization
  - SCF convergence control
  - Symmetry handling and tolerance settings
  - Mulliken population analysis

**Usage Examples:**

```bash
# Interactive mode (recommended)
python NewCifToD12.py

# Batch conversion with directory
python NewCifToD12.py --cif_dir /path/to/cifs

# Specify output directory
python NewCifToD12.py --cif_dir ./cifs --output_dir ./d12s

# Save configuration for reuse
python NewCifToD12.py --save_options --options_file my_settings.json

# Batch processing with configuration
python NewCifToD12.py --batch --options_file high_accuracy_sp.json --cif_dir ./cifs
```

**Frequency Calculation Features:**
- **Calculation Modes**:
  - Gamma point only
  - Phonon dispersion
  - Custom k-point sampling
- **Intensity Calculations**:
  - IR: Berry phase, Wannier functions, or CPHF methods
  - Raman: CPHF with experimental conditions (RAMANEXP)
  - Polarizability and hyperpolarizability
- **Spectral Generation**:
  - Broadened IR/Raman spectra
  - Customizable peak shapes (Gaussian, Lorentzian, Voigt)
  - Multiple output formats (DAT, CSV, JCAMP-DX)
- **Advanced Options**:
  - Anharmonic corrections (ANHARM, VSCF, VCI)
  - Elastic constants
  - Thermodynamic properties
  - Mode-following and isotope effects

**Integration:** 
- Fully integrated with workflow automation system
- Settings automatically extracted and stored in materials database
- Compatible with enhanced queue manager for job submission  

---

### JSON Configuration Support

**Purpose:** Standardize and reuse calculation settings across multiple structures.

**Available Example Configurations:**
- `standard_dft_opt.json` - B3LYP-D3 optimization for most materials
- `high_accuracy_sp.json` - HSE06-D3 single point with tight convergence
- `3c_composite.json` - Fast PBEH3C composite method
- `freq_analysis.json` - Frequency calculations with tight tolerances
- `surface_slab.json` - 2D slab calculations with appropriate settings
- `metallic_system.json` - Settings for metallic/magnetic systems
- `quick_screen.json` - Fast HF/STO-3G for initial screening
- `phonon_bands.json` - Phonon band structure and DOS

**Usage with NewCifToD12.py:**
```bash
# Use predefined configuration (batch mode loads the options file)
python NewCifToD12.py --batch --options_file example_configs/standard_dft_opt.json --cif_dir ./cifs

# Create and save custom configuration
python NewCifToD12.py --save_options --options_file my_config.json

# Batch processing
python NewCifToD12.py --batch --options_file my_config.json --cif_dir ./cifs
```

**Usage with CRYSTALOptToD12.py:**
```bash
# Apply configuration to optimized structure
python CRYSTALOptToD12.py --config-file example_configs/high_accuracy_sp.json --out-file optimized.out

# Save current settings
python CRYSTALOptToD12.py --out-file optimized.out --save-options --options-file sp_settings.json

# Apply them to every optimized structure: one question for all files at a
# terminal (none with --yes, or when nothing is on stdin)
python CRYSTALOptToD12.py --directory ./opts --config-file sp_settings.json --output-dir ./sp
python CRYSTALOptToD12.py --out-file opts/*.out --config-file sp_settings.json --yes
```

A saved template holds calculation settings only. Each structure keeps its own
optimized geometry, symmetry and k-point mesh; a template saved from a parent
with an EXTERNAL basis means "each structure's own parent basis", so the other
structures keep theirs (external or internal). A template that names an
internal basis (e.g. POB-TZVP-REV2) or an external basis directory applies it
to every file. Each `.out` is read with the `.d12` of the same name beside it
(a single `--out-file` too, unless `--d12-file` names another).

A `k_points` entry in a config is not applied (templates saved by earlier
versions stored the saving structure's own mesh there). To give every file the
same mesh on purpose, add it by hand: `"k_points_for_all_files": "8 16"` (or
`"8 8 8"`, or one number); it is written exactly as given.

Nothing is asked per file: a file that would need an answer (a basis set that
lacks one of its elements) fails with the reason instead, at a terminal or
not. The run ends with "N written, M failed" and each failure's reason, and
exits non-zero if any file failed or none was written. A parent optimization
without CRYSTAL's OPT END (killed, out of time), or one that ended "OPT END -
FAILED" at the cycle limit (CRYSTAL then prints no final geometry), is still
converted, from its starting geometry, and counted in the summary as "K from
unfinished or failed optimizations". Two `.out` files of the same name sent to one `--output-dir`
would write the same deck; neither is written.

**Unified Interface:**
```bash
# Automatically detect file type and apply configuration
python d12_from_config.py --config standard_dft_opt.json structure.cif
python d12_from_config.py --config freq_analysis.json optimized.out

# List available configurations
python d12_from_config.py --list-configs

# Show configuration details
python d12_from_config.py --show-config metallic_system.json
```

**Configuration Structure:**
```json
{
  "version": "1.0",
  "type": "d12_configuration",
  "configuration": {
    "name": "standard_dft_opt",
    "description": "Standard DFT optimization",
    "calculation_type": "OPT",
    "method": "DFT",
    "functional": "B3LYP",
    "dispersion": true,
    "basis_set": "POB-TZVP-REV2",
    "tolerances": {
      "TOLINTEG": "7 7 7 7 14",
      "TOLDEE": 7
    },
    ...
  }
}
```

See `example_configs/README.md` for detailed documentation of each configuration.

---

### `create_d12_w-ghosts.py`

**Purpose:** Automatically adds ghost atoms using geometry from `_bulk.out` and `_slab.out`.

**Features:**

- Parses coordinates  
- Computes spacing from slab/bulk output  
- Injects ghost atoms top and bottom  

**Usage:**

```bash
python Ghosts/create_d12_w-ghosts.py
```

**Inputs:** `*_slab.out`, `*_bulk.out`, `*_slab.d12`  
**Outputs:** `*_ghostatoms_slab.d12`  
**Libraries:** `numpy`, `math`, `matplotlib`, `glob`, `os`, `csv`, `warnings`  

---

### `manual_create_d12_w-ghosts.py`

**Purpose:** Manually inserts ghost atoms with user-defined spacing.

**Features:**

- Prompts for spacing value  
- Duplicates atoms symmetrically  

**Usage:**

```bash
python Ghosts/manual_create_d12_w-ghosts.py
```

**Input:** `*_slab.d12`  
**Output:** `_<spacing>A_ghosts_slab.d12`  
**Libraries:** `numpy`, `math`, `matplotlib`, `glob`, `os`, `csv`, `warnings`  

---

### `CRYSTALOptToD12.py`

**Purpose:** Extract optimized geometries from CRYSTAL output files and generate new input files for subsequent calculations.

**Features:**

- **Smart Parsing**: 
  - Extracts final optimized geometry
  - Preserves calculation settings from original input
  - Handles both .out and .gui files
  - Fallback to original D12 for missing parameters
- **Calculation Types**:
  - Single Point (SP) - energy and properties at optimized geometry
  - Frequency (FREQ) - vibrational analysis with all options from NewCifToD12
  - Custom - user-defined calculation sequences
- **Setting Preservation**:
  - Maintains functional, basis set, and method settings
  - Intelligently adjusts tolerances for different calculations
  - Preserves k-points and SCF parameters
- **Batch Processing**:
  - Process entire directories
  - Automatic file discovery and pairing
  - Shared settings mode for consistent parameters

**Usage Examples:**

```bash
# Interactive mode
python CRYSTALOptToD12.py

# Process single file (non-interactive)
python CRYSTALOptToD12.py --out-file structure_opt.out --calc-type SP --non-interactive

# Batch processing
python CRYSTALOptToD12.py --directory ./optimized --calc-type FREQ --non-interactive

# Shared settings for batch
python CRYSTALOptToD12.py --directory ./optimized --shared-settings

# One saved template for every structure (--out-file takes several paths or a glob)
python CRYSTALOptToD12.py --directory ./optimized --config-file sp_settings.json --yes
python CRYSTALOptToD12.py --out-file optimized/*.out --config-file sp_settings.json
```

**Advanced Features:**
- **Intelligent Defaults**: Suggests appropriate settings based on previous calculation
- **Tolerance Optimization**: Recommends tighter tolerances for frequency calculations
- **Error Handling**: Graceful fallback when output parsing fails
- **Setting Merge Logic**: Prioritizes D12 file for critical parameters like TOLINTEG

**Integration:**
- Works with workflow automation
- Outputs compatible with enhanced queue manager
- Preserves material ID and calculation history  

## Charged systems and ion clusters

Ion/solvent clusters (e.g. Li+ with FSI- and ethylene carbonate cut from MD) and
charged cells need two things in the deck: the net electron count, and
usually a starting guess with the ions already charged. MACE writes both, opt-in:

```bash
# cif2d12: ions found by connectivity, net charge = sum of the ions found
mace convert --batch --options_file example_configs/ion_cluster_molecule.json --cif_dir clusters/
mace convert --batch --options_file opts.json --ion-charges auto --net-charge 0 --spinlock 1 --spinlock-cycles 30

# existing decks (any atom order): -o DIR or --in-place; --replace drops their own records first
mace ions cluster.d12 --ion-charges auto -o charged/
mace ions cell.d12 --net-charge -1 --in-place

# derived decks: the parent's records are carried (see below), or worked out
# again from the optimized geometry
mace opt2d12 --out-file cluster_opt.out --calc-type SP --non-interactive --ion-charges auto
```

The same settings as config/options keys: `"ion_charges": "auto"`,
`"net_charge": q`, `"site_states": [{"atom": i, "shell_charges": [...]}]`
(1-based atom, one charge per shell of that atom's basis; needs `net_charge`),
`"chemod": "auto" | "yes" | "no"`, `"fragment_bond_scale"` (default 1.15) and
the existing `"spinlock"` / `"spinlock_cycles"`. Without them every deck is
byte-identical to before (pinned by `tests/test_default_decks_pinned.py`).

### The records (CRYSTAL23 manual, and what was measured)

Tags: **[manual]** CRYSTAL23 user's manual; **[measured]** run on HPCC with
CRYSTAL/23-intel-2023a (2026-10-01); **[untested]** not run.

- `CHEMOD` (basis-set input, p.80): `NC`, then `LA CH(1..NS)` per atom; NS must
  equal the atom's shell count [manual]. CRYSTAL checks the symmetry with the
  new configuration and stops rather than lowering it [manual]; on one Li of a
  symmetry orbit (Li2O, Fm-3m) it stops with `ERROR **** CHEMOD **** SHELL
  CHARGES NOT COMPATIBLE WITH SYMMETRY` (Pcrystal prints only an MPI_Abort)
  [measured]. Both Li of the orbit were accepted, as was one Li in the same cell
  written in P1 [measured]; MACE writes `CHEMOD` only for decks without
  symmetry (MOLECULE from cif2d12, P1 cells) and otherwise refuses it.
- `CHARGED` (p.80): a non-neutral cell with a compensating background; "do not
  use for total energy comparison" [manual].
- `DOPING` (p.103): adds (positive) or removes (negative) electrons [manual]. It
  prints nothing, and the SCF header's "N. OF ELECTRONS PER CELL" keeps the
  count before doping; the Mulliken "NO. OF ELECTRONS" line has the real count
  [measured: Li+(EC)2, header 95, Mulliken 94]. The properties (BAND) header of
  a doped run does show the doped count [measured: MgO DOPING -2, SCF header
  20, properties header 18; that SCF had not converged, the count is the point].
- With an internal BASISSET, `CHEMOD` (+ `CHARGED`) does not change the count:
  the guess is renormalized ("CHARGE NORMALIZATION FACTOR") [measured].
- After `BASISSET <library>` there is no basis-set block: the lines after the
  library name are already block 3, and `CHEMOD` is accepted there ("ELECTRONIC
  CONFIGURATION MODIFIED IN n ATOMS") [measured]. Do not close it with `END`:
  that `END` ends block 3, and the DFT block, `TOLINTEG` and everything after
  it are silently ignored - the run falls back to HF without an error
  [measured]. MACE never writes that `END`; `mace ions` refuses a deck that has
  one, and opt2d12 warns when a parent had one (the parent ran HF).
- With an explicit basis (after `99 0`), a non-neutral `CHEMOD` sets the count
  (with `CHARGED` it matches `DOPING` to 1e-9 Ha) [measured]. For Li+/EC with
  the POB-TZVP-REV2 file: `CHEMOD` + `CHARGED` 48 electrons, `CHEMOD` alone 48
  (a MOLECULE needs no `CHARGED`; same energy), `DOPING -1` alone 48 (energy
  within 4e-7 Ha), but `CHEMOD` + `DOPING -1` 47 - the charge counted twice
  [measured]. A periodic cell with a non-neutral explicit `CHEMOD` and no
  `CHARGED` stops: `ERROR **** INPBAS **** UNIT CELL NOT NEUTRAL` [measured].
  Under Pcrystal (8 ranks) that text is not printed at all - the .out ends in
  `MPI_Abort` only; a serial run of the same deck shows it [measured]. MACE
  never counts such an output as finished (`mace completion` sorts it with the
  runtime failures), and the queue manager's failure message says to rerun
  serially or read fort.87 rather than "unknown error".
  NO3- split 1/3 over its O under `CHARGED` gave 32.000000 electrons, and so
  did 4.33 x 3 (0.01 short): CRYSTAL rounds the count [measured]. MACE still
  writes such splits as 4.333333 / 4.333333 / 4.333334 so the sum is exact.

So MACE writes:

| Basis | Net charge q | CHEMOD asked for | Records |
|---|---|---|---|
| internal BASISSET | q != 0 | either | `DOPING -q`; `CHEMOD` only shapes the guess; never `CHARGED` |
| explicit | q != 0 | yes (MOLECULE default) | `CHEMOD` + `CHARGED`, no `DOPING`; the shell charges must remove exactly q |
| explicit | q != 0 | no (periodic default) | `DOPING -q` |
| either | 0 | yes | `CHEMOD` only, never `DOPING 0` |

A periodic explicit-basis cell gets `DOPING` alone unless `chemod: yes` (then
`CHARGED` adds a background and MACE warns). Measured counts from decks MACE
wrote (SOLDEF2MSVP, HSEsol-3c): Li+(EC)2 q=+1 94, Li+FSI-EC q=0 138, Li+(FSI-)2
q=-1 182, Li+(FSI-)3 q=-2 272; explicit POB Li+(NO3-)2 q=-1 66 - each the
neutral count minus q.

### Ions and their starting guess

Fragments come from bonds (distance < 1.15 x the sum of Cordero covalent
radii; periodic images along the periodic directions, so an ion across a cell
face is one fragment and a framework bonded to its own image is "extended").
Known ions, matched by formula, coordination and terminal atoms:

| Ion | Guess |
|---|---|
| Li+, Na+, K+, Mg2+, Ca2+, Zn2+ | charge removed from the valence s shell |
| FSI-, TFSI- (4 sulfonyl O), OTf- (3 sulfonyl O), ClO4- (4 O), NO3- (3 O), PF6- (6 F), BF4- (4 F) | the extra electron spread over the terminal atoms' valence p shell |

FSI- gives the group's recipe exactly: Li `2.0 0.0 0.0 0.0 0.0`, each sulfonyl
O `2.0 2.0 0.0 0.0 4.25 0.0 0.0`. Anything else (HNO3, SO2F2, solvents) is
neutral. The table cations are never bonded to anything, so a Li+ 1.95 A from a
solvent O stays its own ion; that is also why auto mode refuses what looks
like an ionic solid (lone O/F/Cl/... atoms, or a table metal bonded to an
extended framework): there it would invent ions. Use `net_charge` (DOPING) for
those. In a cell with a framework the framework is counted neutral (warning).

An anion outside the table would also be counted neutral, so auto mode
refuses a structure with table ions and a molecule that cannot be a
closed-shell neutral one: each bond gets order 1-3 and every atom's bond
orders must add up to one of its usual valences (H 1, C 4, N 3/5, O 2, S
2/4/6, P 3/5, halogens 1/3/5/7, ...). CO3 and SO4 (2 unpaired valences), OH,
carboxylates and alkoxides (1) fail that test; EC, water, SO2F2, HNO3, DMSO and
benzene pass. Li2CO3 in auto mode is therefore refused rather than written
with `DOPING -2`; give `net_charge` (the guess of the anion stays neutral, with
a warning; `site_states` can shape it). A molecule MACE cannot judge (an
element without a valence entry) that touches a table cation gets a warning
when the auto net charge is not 0.

Shell charges: for an explicit basis they are the file's own CHE values. For
internal libraries MACE knows SOLDEF2MSVP, POB-TZVP-REV2 and POB-DZVP-REV2 for
H B C N O F Na Mg P S Cl K Ca Zn Li [measured: CRYSTAL's LOCAL ATOMIC FUNCTIONS
listing and atomic-guess shell occupations, filled in listing order, which
reproduces the explicit files' CHE values]. Any other library or element needs
`site_states`, checked against the shell listing of a parent .out
(`mace ions --parent-out`).

### Odd electron counts

The count is the sum of atomic numbers (valence charges for an explicit ECP;
pseudopotential cores are even) minus q. An odd count needs `SPIN` (DFT) +
`SPINLOCK n cycles` or UHF + `SPINLOCK`: the value comes from the options, then
the deck's or parent's own `SPINLOCK`, then a question at a terminal (default
`1` and 50 cycles), otherwise a molecule's deck is refused [measured: `SPIN` +
`SPINLOCK 1 30` works for ATOMREMO + DOPING cells]. RHF and HF-3c/HFsol-3c are refused, and
so is `GHOSTS` + `DOPING` with an odd count ("NALPHA-NBETA NOT COMPATIBLE WITH
NUMBER OF ELECTRONS": the parity is checked before the doping) [measured]; the
working ghost recipe is an explicit basis with `CHEMOD` + `CHARGED`.
A periodic cell may also keep an odd count closed shell (a half-filled band);
MACE then only warns. The screened hybrids HSEsol-3c, HSE06 and HSEsol (HJS
screened exchange) crash on any odd-electron open-shell system on CRYSTAL23,
charged or not, as a MOLECULE (0D) and in a periodic 3D box alike: "DIIS TEST:
NaN" at SCF cycle 1-3, then a segfault or malloc assert; `NODIIS` +
`LEVSHIFT`, `SPINLOCK 1 30` and dropping `BIPOSIZE`/`EXCHSIZE` do not help
[measured]. The same systems converge with PBE0 and PBEsol0 (175-531 cycles),
and even counts with `SPIN` are fine [measured]. MACE keeps the functional
asked for - it never changes the method - and warns for every dimensionality
(cif2d12, opt2d12 and `mace ions`), naming PBE0/PBEsol0 as what converged.

### Derived decks (opt2d12)

`CHEMOD`, `CHARGED` and `DOPING` of the parent are carried into SP/FREQ/...
children (also when the parent deck could not be parsed). `CHEMOD` only when
the child lists the same atoms and loads the same basis set (same BASISSET name
or identical explicit blocks); otherwise it is dropped with a warning, and a
charge it carried (an explicit basis) becomes `DOPING`, from the parent's
Mulliken count when the .out is there. Without a parent deck the .out's
Mulliken count minus the valence charges gives `DOPING`; without a Mulliken
print (a FREQ parent, often) MACE warns only when the .out shows a sign of a
charged run (CHEMOD, CHARGED, "UNIT CELL NOT NEUTRAL", a renormalized guess),
since `DOPING` itself leaves no trace in the .out [measured]. An odd count gets the
parent's `SPIN`/`SPINLOCK`, or a molecule's deck is refused (a periodic one is
written with a warning). Measured through a
POB-TZVP-REV2 template: a SOLDEF2MSVP Li+(EC)2 parent (`CHEMOD` + `DOPING -1`)
gave `DOPING -1` without `CHEMOD`, 94 electrons; an explicit-basis Li+(NO3-)2
parent (`CHEMOD` + `CHARGED`) gave `DOPING 1`, 66 electrons; with
`--ion-charges auto` the guess was rebuilt for the new basis (94 electrons,
energy within 3e-7 Ha) [measured].

### Output side

`mace analyze --extract-properties` adds `electron_count` and
`fragment_charges` (per fragment: summed Mulliken charge = valence charge -
population, spin, formal charge, deviation) for charged runs, and prints them as
a Fragments section; an ion more than 0.5 e from its formal charge is flagged
as having lost or gained its electron. This is a heuristic: Mulliken charges
depend on the basis set and are never close to formal charges.
`mace analyze --extract-properties <dir> --fragments` computes them for
neutral runs too.

### Not run on CRYSTAL yet [untested]

SLAB and POLYMER decks with charges; `chemod: yes` (CHEMOD + CHARGED) in a
periodic explicit-basis cell; fractional net charges (written as fractional
`DOPING`, no spin state); auto mode in a cell with a framework (counted
neutral); internal-library ECP elements (the parity assumes the even cores
pseudopotentials have); the fragment report on periodic outputs.

---

## Installation

```bash
pip install numpy matplotlib ase spglib
```

---

## Outputs

All scripts generate CRYSTAL23-compatible `.d12` files:

- From `.cif` via `NewCifToD12.py`  
- With ghost atoms via `create_d12_w-ghosts.py` or `manual_create_d12_w-ghosts.py`  
- Post-optimization via `CRYSTALOptToD12.py`  

---

## Configuration Modules

### `d12_calc_freq.py`

**Purpose:** Frequency calculation configuration module.

**Features:**

- **Interactive Configuration**: Guided setup with templates and custom options
- **Calculation Templates**:
  - Basic frequencies: Gamma point vibrations
  - IR spectrum: With intensity calculations
  - Raman spectrum: With CPHF and experimental conditions
  - IR + Raman: Combined spectroscopy
  - Phonon bands: Dispersion and DOS
  - Elastic + Phonon: Mechanical properties
  - Custom: Build your own configuration

- **Intensity Methods**:
  - Berry Phase: Fast, good for periodic systems
  - Wannier Functions: Better for ionic systems  
  - CPHF: Most accurate, required for Raman

- **Advanced Features**:
  - RAMANEXP: Experimental conditions (temperature, laser wavelength)
  - Anharmonic corrections: ANHARM, VSCF, VCI methods
  - Mode following and coupling analysis
  - Isotope substitution effects
  - Temperature-dependent properties
  - Pressure effects (quasi-harmonic approximation)

- **Phonon Band Structure Enhancements**:
  - **Extended Bravais Lattice Detection**: `d3_kpoints.get_extended_bravais` picks the lattice variant (e.g., aP2 vs aP3, oF1 vs oF2 vs oF3) from cell parameters when it gets them; without them it returns the first variant (aP2, oF1, tI1, ...). A phonon path from `CRYSTALOptToD12.py` gets them from the parent output (`d3_kpoints.extract_lattice_parameters_from_output`); one from `NewCifToD12.py` has no output and uses the first variant
  - **Inversion Symmetry Detection** (SeeK-path format only): `d3_kpoints.get_seekpath_full_kpath` uses the `<variant>_noinv` path with primed k-points for a non-centrosymmetric structure, falling back to the centrosymmetric path when no `_noinv` entry exists. Inversion is read from the parent's CRYSTAL output: `d12_calc_freq._output_as_file` writes the output text that `CRYSTALOptToD12.py` passes (`optimization_content`) to a temporary file, and `d3_kpoints.detect_inversion_from_crystal_output` reads the centrosymmetric flag, the SYMMOPS table or the printed space-group number from it (with the seekpath library installed, SeeK-path decides from the structure it parses from the same file). The space-group number decides (`d3_kpoints.has_inversion_symmetry`) only when there is no output (`NewCifToD12.py`) or the output shows none of these. The label, vector and literature formats do not use inversion
  - **Parent output**: `d12_calc_freq.write_frequency_section` and `d12_calc_freq.get_auto_phonon_path` receive the parsed output as text (`optimization_content`) and hand it to the SeeK-path helpers (`get_seekpath_full_kpath`, and `get_seekpath_labels` for the title) as a temporary `.out` file (`d12_calc_freq._output_as_file`), since those helpers read a file. From `NewCifToD12.py` there is no output, and the space-group number is used
  - **seekpath required**: with a parent output of a 3D cell, the SeeK-path format computes the path with the seekpath library (a required dependency). Without it, `CRYSTALOptToD12.py` refuses that deck (`d12_calc_freq.phonon_dispersion_refusal`) with the install command, unless `--static-kpath` or `MACE_STATIC_KPATH=1` asks for the static HPKOT table path. Slabs, and the label, vector and literature formats, do not need it
  - **K-path Source Tracking**: Labels phonon band titles with path source:
    - `SeeKPath (w.I)` - SeeK-path with inversion symmetry
    - `SeeKPath (no.I)` - SeeK-path without inversion symmetry  
    - `Literature` - Literature-based paths
    - `default` - Default/standard paths
  - **Discontinuous Path Support**: With custom labels in the "Phonon band structure" template (template 6, path option 3 in `d12_calc_freq.get_advanced_frequency_settings`), "|" splits the path and no segment is written between the labels on either side of it. The custom-labels prompt of the "Custom settings" template (template 8) does not handle "|": it pairs every consecutive token, so a "|" would be written into a segment. The coordinate formats (vectors, literature, SeeK-path) write each segment as its own start/end pair, so a break is a segment that does not start where the previous one ended. The automatic label format uses `d3_kpoints.get_band_path_from_symmetry`, whose paths contain no "|"

### `d12_calc_basic.py`

**Purpose:** Basic calculation settings shared across all calculation types.

**Features:**
- DFT functional selection with categories (LDA, GGA, Hybrid, meta-GGA)
- Basis set configuration (internal/external)
- K-point mesh optimization
- SCF convergence parameters
- Tolerance settings with calculation-specific recommendations
- Symmetry and optimization controls

### `d12_constants.py`

**Purpose:** Central repository of constants and data structures.

**Contents:**
- Atomic symbols and numbers
- Element properties and ECP requirements
- Basis set libraries (POB-TZVP, Peintinger, Stuttgart)
- DFT functional definitions with exchange/correlation percentages
- Space group data for multi-origin systems
- Default calculation parameters

### `d12_parsers.py`

**Purpose:** Parse CRYSTAL output and input files.

**Features:**
- Extract optimized geometries from .out files
- Parse calculation settings from .out files
- Read existing D12 files for setting preservation
- Handle fort.34 (GUI) files
- Extract properties: energy, forces, band gaps
- Error handling for incomplete files

### `d12_writer.py`

**Purpose:** Write properly formatted D12 files.

**Features:**
- Format geometries with correct precision
- Write calculation keywords in proper order
- Handle special formatting requirements
- Validate settings before writing
- Support all CRYSTAL input sections

### `soc_ecp.py`

**Purpose:** Two-component spin-orbit (SOC) single-point decks, only when asked
for with `"soc": true` in a cif2d12 options file or an opt2d12 template.

- Reads the Stuttgart spin-orbit ECPs in `Crystal_d3/Archived/basis/sopseud/2NN.mol`
  (MOLPRO format) and writes them as CRYSTAL23 `INPSOC` records, INTERNAL
  convention (manual pp. 84-87; the manual's Eu example is reproduced exactly).
- An atom's ECP is replaced only if it is exactly the scalar part of the
  spin-orbit ECP, so its valence basis stays the one published with that
  potential. The `full.basis.doublezeta`/`triplezeta` ECPs of Rb-Bi (not Yb)
  qualify, and those of `full.basis.triplezeta.rev2022` (not La or Yb).
  All-electron atoms are left as they are and get no spin-orbit term (manual
  p. 175): the log names them, with a warning from Ga (Z = 31) on, where the
  4p splitting lost is about 0.3-0.5 eV (Ga-Br are all-electron in the
  `full.basis` families; the stuttgart library's ECP10 records 231-234 for
  Ga-Se carry the term, but no writer picks them: enter them by hand).
  An ECP label above 300 (382 for a second Pb) is its element, Z = label
  modulo 100. CHEMOD after `99 0` is kept (manual p. 172); GHOSTS, CHARGED
  and other basis-input records are refused by name.
- Adds `TWOCOMPON` / `SOC` / `END` before `SCFDIR` (manual sec. 6.2).
- Settings measured on the stock CRYSTAL/23 build: DIIS and HISTDIIS are left
  out (a DIIS record stops a 2c run), SHRINK gets its Gilat net equal to the
  Monkhorst net (`IS IS`, or `0 max(IS1,IS2,IS3)` in the directional form), and
  the deck gets FMIXING 50 and at least 200 cycles. FMIXING 50 converged
  fastest of 30/50/60/70/85 in every system measured (Bi2 bilayer 12 cycles,
  PbTe 9, fcc Au at SHRINK 12 12 25) with the same energies as 85; FMIXING 30
  aborted Bi2 in cycle 1. Any other FMIXING in the deck (MACE's 30, or the
  parent's or template's) is replaced, and the log names the value replaced.
  `"soc_fmixing": N` (0-100) in the options file or template writes FMIXING N
  instead. A MAXCYCLE below 200 is kept, with a warning.
- Never adds SMEAR by itself. `"smear": <width in hartree>` in the options
  file or template writes SMEAR (manual p. 126) after SHRINK. opt2d12 warns
  when the parent's .out looks metallic (its last SCF cycle reports "POSSIBLY
  CONDUCTING STATE" or a band gap below 0.1 eV) and the SOC deck has no SMEAR,
  and asks for a width at a terminal (Enter = none); in a batch it lists the
  deck in the closing summary. cif2d12 prints one such warning per run.
- Accepts SPIN and the `-D3` forms of BLYP, PBE, B3LYP, PBE0 and PW1PW. SPIN
  has no effect in a 2c deck: the same energy with and without it (HPCC).
- Measured on the stock CRYSTAL/23 build (HPCC): PbTe (SHRINK 7 7) and a Bi2
  bilayer (SHRINK 0 10 / 10 10 1) converge. fcc Au converges on a 12 12 mesh
  but aborts in the first SCF cycles on MACE's default 10 10, and on 11 11
  ("ERROR **** ZERO **** FERMI ENERGY NOT IN INTERVAL"); SMEAR 0.005 lets
  10 10 converge.
- A parent deck with a TWOCOMPON block holding SOC makes the `opt2d12` child a
  SOC deck too (with a note) unless the template sets `"soc": false`; the
  parent's INPSOC records are kept when they are exactly MACE's. A scalar child
  of a SOC parent gets a warning; an OPT or FREQ child of a SOC parent is
  refused (manual p. 166).
- Refuses, writing no deck: OPT/FREQ and other non-single-point runs,
  functionals and keywords manual chapter 6 does not list (including a DFTD3
  block), `BASISSET`, and ECPs that are not a spin-orbit ECP's scalar part.

Options for SOC decks (keys of a cif2d12 options file or an opt2d12 template;
none is on by default, and without them every deck is as above):

| Key | Values | What it does |
| --- | --- | --- |
| `soc_fmixing` | 0-100 | FMIXING of the SOC deck (default 50). |
| `smear` | width in hartree | SMEAR after SHRINK (see above). |
| `soc_ecp` | `"inpsoc"` (default), `"stutsc"` | `"stutsc"` writes CRYSTAL's internal `STUTSC` record (manual p. 84) in place of the INPSOC records, for the elements where it was measured to be the same potential with the same core (`soc_ecp.STUTSC_CORES`: Rb-Mo, Ru-I, Cs, Ba, Ta-Bi; not Hf, whose STUTSC gives the r² local term the exponent of the first term, 10.2488560 instead of the published ECP60MDF's 10.7291270, +9.67e-5 Ha measured, nor La-Lu, where CRYSTAL stops). The ECP must still pass the pairing check, and SOSCALE must be 1 (STUTSC has no scale factor). A parent deck's STUTSC is kept when its valence basis is MACE's library basis for such an element, or the official pob-TZVP-rev2-SOC block; anything else is refused. |
| `soc_guess` | `"guesspso"`, `"guesspnoso"` | opt2d12 and workflow plan steps only (refused by cif2d12: no parent). Writes the 2c restart record in the TWOCOMPON block (manual p. 173) and stages the parent's `.f9` (beside its `.out`) as `<deck stem>.f20`, which the job script copies to fort.20. `guesspso`: the parent must be a 2c run (with or without SOC). `guesspnoso`: the parent must be a 1c run that was unrestricted (UHF or SPIN) and ran without symmetry (group 1 or SYMMREMO); the child gets no SYMMREMO (TWOCOMPON deactivates symmetry, p. 169). The parent deck and the deck as written must have the same symmetry frame and per-atom shells (ECP records aside), or nothing is written. FMIXING 80 for both (measured on a Bi2 bilayer only: at 50 both restarts diverged; at 80 GUESSPSO converged in 3-5 cycles and GUESSPNOSO in 29, where a cold SOC start takes 12), unless `soc_fmixing` is set. `mace preflight` and `mace submit` refuse a restart deck with nothing to stage (CRYSTAL stops on GUESSPSO with nothing in fort.20, measured; GUESSPNOSO without its parent's matrix was not run). A `guesspnoso` parent that is restricted or kept its symmetry is refused before anything is written: CRYSTAL23 aborts such a restart at once in READFG (measured; see "A 1c parent for GUESSPNOSO" below). A GUESSPSO or GUESSPNOSO outside the TWOCOMPON block is refused everywhere (`soc_deck`, `mace preflight`, `mace submit` with or without `--nosubmit`): CRYSTAL stops on it at input ("KEYWORD GUESSPSO NOT ALLOWED", measured). |
| `soc_basis` | `"pob-tzvp-rev2-soc"` | Replaces every ECP atom's basis block whole by the official X_pob_TZVP_rev2_SOC block for its element (CRYSTAL's STUTSC with the valence basis published for it, "to be used with CRYSTAL23"; J. Laun, T. Bredow, J. Comput. Chem. 43 (2022) 839, doi 10.1002/jcc.26839; J. K. Desmarais, J. P. Flament, A. Erba, Phys. Rev. B 101 (2020) 235142, doi 10.1103/PhysRevB.101.235142). The set covers Rb-I except Tc, Cs, Ba and Hf-Bi; a deck with any other ECP element (La-Lu, actinides, Po onwards) is refused, naming it. All-electron atoms (H-Br, and MACE's all-electron Tc) keep their block, since there is no SOC operator on an all-electron atom (manual p. 175); atom numbers never change. Refused with `"soc_ecp": "inpsoc"`, with SOSCALE other than 1, and for BASISSET decks. The data, with source URL, retrieval date and references, is `basis_sets/pob-tzvp-rev2-soc.lib`, built by `basis_sets/build_library.py` and read through `basis_library.soc_basis_record`. Its valence shells are pob-TZVP-rev2's (for Rb-I those of `full.basis.triplezeta.rev2022`, not the 2018 ones of `full.basis.triplezeta`), and the ECP is CRYSTAL's STUTSC - except for Hf, which gets the official valence shells on MACE's INPSOC (the published ECP60MDF; `soc_ecp.STUTSC_NOT_THE_PUBLISHED_ECP`) - so the energy differs from a scalar parent's: the SOC contribution needs a 2c run without SOC in the same basis (manual p. 174), which MACE does not write - copy the deck and delete the `SOC` line of its TWOCOMPON block. |

**A 1c parent for GUESSPNOSO.** `"soc_guess": "guesspnoso"` needs a parent
that ran as an unrestricted one-component SCF without symmetry (manual p. 173).
MACE does not write that parent; make it by hand from the SOC deck you want to
start (or from any deck in the same frame and basis):

1. Delete the `TWOCOMPON` ... `END` block (the `TWOCOMPON`, `SOC` and `END`
   lines before `SCFDIR`, and a `GUESSPSO` or `GUESSPNOSO` line between them).
2. Add `SYMMREMO` (p. 78) as the last record of the geometry block, before its
   `END`.
3. Add `SPIN` (p. 140) as the first record of the `DFT` block, unless it is
   there already. A Hartree-Fock deck has no `DFT` block: add `UHF` (p. 131)
   to its SCF records instead.
4. Keep the basis as it is, `INPSOC` (or `STUTSC`) records included, so the
   parent's matrix has the shells of the SOC deck.

Run it, then make the SOC deck from its `.out` with `opt2d12` and a template
holding `"soc": true, "soc_guess": "guesspnoso"`; the parent's `.f9` is staged
as the child's `.f20`, and the child gets FMIXING 80. Measured on CRYSTAL23
(Bi2 bilayer): the restart from such a parent converged in 29 cycles to the
cold-start SOC energy (5.6e-8 Ha apart; FMIXING 50 diverged); `SYMMREMO` in the
2c deck changes nothing, so opt2d12 does not write it there. A parent that kept
its symmetry makes CRYSTAL abort at once with `ERROR **** READFG **** RESTART
NOT POSSIBLE - SYMMETRY INCOMPATIBILITY`, a restricted one with `... - OPEN-CLOSED
SHELL INCOMPATIBILITY`; opt2d12 refuses both before writing anything.

### `d12_interactive.py`

**Purpose:** Interactive prompts and user interface utilities.

**Features:**
- Consistent yes/no prompts
- Clear section headers
- Input validation
- Default value handling
- User-friendly messages

---

## Recent Improvements

### Parser Fixes
- **Atomic Symbols**: Fixed ECP handling to preserve correct element symbols
- **TOLINTEG Values**: Corrected negative value conversion from output files
- **K-points**: Fixed extraction to read correct line after SHRINK keyword
- **Shared Settings**: Unified merge logic between batch and individual processing

### Frequency Enhancements  
- **RAMANEXP Support**: Added experimental condition settings for Raman
- **Template System**: Pre-configured templates for common calculations
- **Advanced Options**: Separated basic and advanced settings for clarity
- **Tolerance Recommendations**: Automatic suggestions for frequency calculations
- **Phonon Path Generation**: SeeK-path format via `d3_kpoints.get_seekpath_full_kpath` (inversion and cell parameters from the parent output when there is one; see `d12_calc_freq.py` above)

### Phonon Band Structure Updates
- **Extended Bravais Support**: `d3_kpoints.get_extended_bravais` uses cell parameters when given them (read from the parent output for phonon paths from `CRYSTALOptToD12.py`)
- **Inversion Symmetry**: SeeK-path format only; read from the parent output (`d3_kpoints.detect_inversion_from_crystal_output`, reached through `d12_calc_freq._output_as_file`), and from the space-group number (`d3_kpoints.has_inversion_symmetry`) when there is no output
- **Source Labeling**: Clear indication of k-path source (SeeK-path w/wo inversion, literature, default) in titles
- **Non-centrosymmetric Paths**: Support for primed k-points (X', Y', Z', etc.) when needed
- **Discontinuous Paths**: "|" in custom label paths splits segments; coordinate paths list separate segments (details under `d12_calc_freq.py`)

### Code Quality
- **Modular Refactoring**: Separated calculation configs into dedicated modules
- **Removed Duplication**: Eliminated ~400 lines of repeated code
- **Consistent Behavior**: Unified configuration between all scripts
- **Better Documentation**: Inline help and clear prompts

---

## Usage Tips

### For New Users
1. Start with `NewCifToD12.py` in interactive mode
2. Use calculation templates for common tasks
3. Accept default values unless you have specific requirements
4. Use "Basic" customization level for CIF conversion

### For Frequency Calculations
1. Use tighter tolerances (TOLINTEG: 9 9 9 11 38)
2. For IR spectra, Berry phase is usually sufficient
3. For Raman, CPHF is required
4. Consider RAMANEXP for accurate intensities
5. Use templates to avoid manual configuration
6. Phonon band structures automatically use optimal k-paths with source tracking:
   - Title format: `filename - Phonon Band Structure - <source> - <k-path>`
   - Sources: SeeKPath (w.I), SeeKPath (no.I), Literature, or default

### For Batch Processing
1. Test with one structure first
2. Use shared settings mode for consistency
3. Organize inputs in clean directories
4. Check output files for convergence

### For Workflow Integration
1. Use run_workflow.py for complex sequences
2. Let the system handle file dependencies
3. Monitor progress with enhanced queue manager
4. Check material database for results

---

## Requirements

### Python Libraries
```bash
pip install numpy matplotlib ase spglib
```

### Environment
- Python 3.11 or newer (CI tests 3.11 and 3.12)
- CRYSTAL17/23 installation
- Access to basis set files
- For workflows: SLURM cluster environment

---

## File Formats

### Input Files
- **CIF**: Crystallographic Information Files
- **D12**: CRYSTAL input files
- **OUT**: CRYSTAL output files
- **GUI**: CRYSTAL fort.34 geometry files

### Output Files  
- **D12**: Generated CRYSTAL input
- **DAT**: Spectral data files
- **CSV**: Tabulated results
- **JCAMP-DX**: Spectroscopy standard format

---

## Support

For issues or questions:
1. Check inline help in scripts
2. Review this README
3. Examine example calculations
4. Consult CRYSTAL manual for keyword details

---

## UX Improvements Summary (July 2025)

### General D12 UX Fixes

#### MAXCYCLE for Optimization
- **Fixed**: MAXCYCLE now correctly shows 800 (not 200) for optimization
- **Updated**: All prompts, displays, and fallback values use 800 as default

#### Redundant Optimization Prompts
- **Removed**: Eliminated redundant Y/n prompt before optimization menu
- **Improved**: Direct access to optimization configuration (Default/TIGHTOPT/Custom)

#### SCF Settings Organization  
- **Added**: SMEAR option now included in SCF settings section
- **Improved**: Logical grouping of all SCF-related options
- **Enhanced**: SMEAR prompts include guidance (0.005-0.02 Hartree for metals)

### CRYSTALOptToD12 UX Improvements

#### SP Calculation Flow
- **Fixed**: Removed redundant configuration prompt for SP calculations
- **Enhanced**: Direct convergence settings with detailed tolerance displays
- **Added**: Dedicated SP convergence menu with clear recommendations

#### SCF Settings Streamlining
- **Removed**: Double prompt for SCF settings
- **Fixed**: FMIXING now uses direct value input (not Y/N)
- **Added**: Helpful descriptions for FMIXING parameter

#### Tolerance Level Corrections
- **Standardized**: Consistent tolerance naming across all menus:
  - Standard: 7 7 7 7 14, TOLDEE: 7
  - Tight: 8 8 8 9 24, TOLDEE: 9
  - Very tight: 9 9 9 11 38, TOLDEE: 11
  - Ultra tight: 10 10 10 12 40, TOLDEE: 12

### Frequency Calculation UX Enhancements

#### SCF Tolerance Defaults
- **Fixed**: FREQ calculations now recommend tight tolerances by default
- **Added**: Dedicated SCF convergence menu with frequency-specific guidance
- **Default**: Very Tight (option 3) for accurate force constants

#### Phonon DOS Type Selection
- **Added**: Clear choice between PDOS and INS before parameter input
- **Enhanced**: Type-specific parameter prompts and defaults
- **Improved**: INS includes neutron weighting options

#### Temperature Range Display
- **Fixed**: Default temperature values shown before customization prompt
- **Enhanced**: Clear confirmation of selected temperature range

### Optimization Convergence Updates

#### Enhanced Menu System
- **Added**: Convergence menu with detailed explanations
- **Improved**: Clear descriptions of TOLDEG, TOLDEX, TOLDEE, MAXCYCLE
- **Enhanced**: Context-appropriate recommendations for each level

#### Geometry Tolerance Adjustments
- **Refined**: Better progression between convergence levels:
  - Standard: TOLDEG=0.0003, TOLDEX=0.0012, TOLDEE=7
  - Tight: TOLDEG=0.0001, TOLDEX=0.0004, TOLDEE=8 (3x tighter than default)
  - Very Tight: TOLDEG=0.00003, TOLDEX=0.00012, TOLDEE=9 (old TIGHTOPT values)
- **Improved**: Intermediate "Tight" option provides better stepping stone

#### Default Changes
- **Updated**: Default changed from TIGHTOPT to Standard convergence
- **Rationale**: Standard convergence sufficient for most calculations
- **Benefit**: Faster routine calculations, with easy access to tighter options

### User Flow Improvements

#### Before
```
Use default optimization settings? [Y/n] n
Select convergence (1-3) [1]: 2
Change SCF settings? [y/N] y  
Use default SCF settings? [Y/n] n
(SMEAR option missing)
```

#### After
```
=== OPTIMIZATION CONVERGENCE CRITERIA ===
Select convergence option:
1. Default - MAXCYCLE=800
2. TIGHTOPT - MAXCYCLE=800
3. Custom
Choice [1]: 2

=== SCF SETTINGS ===
MAXCYCLE [800]: 
FMIXING [30]: 
Use SMEAR? [y/N]: 
Use LEVSHIFT? [y/N]: 
```

---
