# MACE Technical Documentation

## MACE - Mendoza Automated CRYSTAL Engine

**Developed by**: Marcus Djokic (Primary Developer)  
**Contributors**: Daniel Maldonado Lopez, Brandon Lewis, William Comaskey  
**Advisor**: Prof. Jose Luis Mendoza-Cortes  
**Institution**: Michigan State University, Mendoza Group  

## Repository Overview

MACE (Mendoza Automated CRYSTAL Engine) is a scientific computing toolkit for **CRYSTAL** quantum chemistry software workflows. CRYSTAL is used for periodic and molecular calculations of electronic structure, properties, and materials science applications. The codebase provides automation scripts, utilities, and analysis tools for the entire CRYSTAL computational workflow on SLURM HPC clusters.

## Installation and Setup

### Initial Setup
1. Clone or download the MACE repository
2. Navigate to the repository root directory
3. Run the setup script:

```bash
# Interactive setup (recommended)
python setup_mace.py --add-to-path

# Specify shell directly
python setup_mace.py --shell zsh --add-to-path
python setup_mace.py --shell both --add-to-path  # Configure both bash and zsh

# Setup without PATH modifications
python setup_mace.py --shell zsh
```

### Environment Variables
The setup script configures:
- `MACE_HOME`: Points to the repository root
- `PATH`: Adds all script directories (with `--add-to-path`)

### Activation
After setup:
```bash
# Apply changes to current session
source ~/.zshrc  # or ~/.bashrc

# Or use the activation script for temporary use
source activate_mace.sh
```

### New Directory Structure (Phase 3)
```
reorganization/
├── mace_cli                   # Main MACE command-line interface
├── setup_mace.py              # Installation and setup script
├── mace/                      # Core MACE package
│   ├── run_mace.py           # Main workflow runner
│   ├── workflow/             # Workflow management
│   ├── queue/                # Queue and job management  
│   ├── database/             # Material tracking database
│   ├── submission/           # Job submission scripts
│   ├── recovery/             # Error detection and recovery
│   ├── utils/                # Utilities and helpers
│   └── config/               # Configuration files
├── Crystal_d12/              # D12 input generation
├── Crystal_d3/               # D3 properties generation
└── code/                     # Legacy directory structure (preserved)
```

## Common Commands

### Running Scripts
All scripts are Python-based and run directly:
```bash
python script_name.py
```

### MACE Commands (Primary Interface)
```bash
# Interactive workflow planning
mace workflow --interactive
python mace_cli workflow --interactive  # Or with full path

# Submit calculations (runs exactly what you submit; no workflow is inferred)
mace submit my_calculation.d12
mace submit property.d3
mace submit --track my_calculation.d12           # also record in materials.db

# Submit and plan what follows (opt-in; implies --track)
mace submit --progress full_electronic my_calculation.d12
mace submit --progress interactive my_calculation.d12

# Monitor calculations
mace monitor
mace manager --status

# Analyze results
mace analyze --extract-properties .

# Convert CIF files
mace convert --cif_dir ./cifs

# Plot outputs (band/DOS/structure/cube/FREQ/IR/Raman)
mace plotting

# Quick workflow execution
mace workflow --quick-start --cif-dir ./cifs --workflow full_electronic

# Check calculation completion status
mace completion                           # Check current directory
mace completion --recursive               # Scan subdirectories (workflow structures)
mace completion --detailed                # Show detailed file listings
mace completion --remove-zombie-jobs      # Detect and cancel failed jobs still running in SLURM
mace completion --recursive --remove-zombie-jobs  # Full workflow zombie detection
mace completion --organize                # Sort files by status into folders
```

### One D12 Template for Many Structures (`mace opt2d12`)
```bash
# Save the settings chosen for one structure as a template
mace opt2d12 --out-file optimized.out --save-options --options-file sp_template.json

# Apply the template to every .out in a directory, or to the files a glob names
mace opt2d12 --directory opts --config-file sp_template.json --output-dir sp
mace opt2d12 --out-file opts/*.out --config-file sp_template.json --output-dir sp --yes
```
- `--out-file` takes one or more paths; with `--out-file`, `--directory` is ignored. Without either, the current directory is used. Each `.out` is read with the `.d12` of the same name beside it, if there is one
- `--config-file` is a settings JSON: one saved with `--save-options`, or a file from `Crystal_d12/example_configs/`. Each structure keeps its own geometry and symmetry. It is read once before any file, and an unreadable one stops the run
- With several files, the plan is printed and asked about once (`Apply to all N files?`), only when run at a terminal; `--yes` (`-y`) skips that question. No per-file questions are asked: a file that would need an answer fails with the reason and the others carry on
- `--output-dir` is created if missing (default: beside each `.out`). The run exits non-zero if any file failed or nothing was written

### Phonons with phonopy (`mace phonopy`)
Finite-displacement phonon dispersion, DOS and thermal properties for a 3D
crystal, with CRYSTAL computing the forces and phonopy (optional dependency,
`pip install 'phonopy<5'`) doing the rest. phonopy runs on the machine where you
type the commands; the cluster only runs CRYSTAL.
```bash
# 1. Displaced supercells from a finished OPT (or SP) parent
mace phonopy generate mat_opt.out --dim "2 2 2"          # reads mat_opt.d12 beside it
# 2. Run them - a plain submission, NOT --track or --progress, with the walltime step 1 printed
mace submit phonopy_mat_opt_222 --walltime 2:00:00
# 3. Check the outputs, then forces -> band.yaml, mesh.yaml, total_dos.dat, thermal_properties.yaml
mace phonopy collect phonopy_mat_opt_222 [--nac-from mat_freq.out] [--plot]
```
- **generate** writes one deck per displaced supercell, `<stem>_ph<dim>_disp-NNN.d12`,
  with its geometry in `<stem>_ph<dim>_disp-NNN.ext`. The deck starts with `EXTERNAL`
  (manual p.21) and the job script copies the `.ext` to `fort.34` before CRYSTAL starts
  (format: manual pp.450-451). Job names carry the parent's name and the supercell, so
  two sets never share a scratch directory. It also writes `phonopy_disp.yaml` (the
  displacements) and `mace_phonopy.json` (parent paths and checksums, every deck edit,
  versions).
- **The parent's method is kept.** The deck has the layout phonopy's own CRYSTAL
  writer gives it: `EXTERNAL`, the parent's basis set and SCF block byte for byte
  (`BASISSET` replaces the END of the geometry block, manual p.28), then `TOLDEE`,
  `GRADCAL` (manual p.111) and `END`. The part in between is phonopy's TEMPLATE and
  must not contain an END, TOLDEE or GRADCAL: measured on CRYSTAL23, a TEMPLATE that
  ends with END makes CRYSTAL stop before GRADCAL, with return code 0 and no forces.
  For an explicit basis it is also written to `TEMPLATE` in the set directory. The
  edits, each printed: the final END removed; TOLDEE moved before GRADCAL and raised
  to at least 10 (`--toldee N` raises it further, never lowers it); GUESSP dropped;
  `BIPOSIZE`/`EXCHSIZE 110000000` added when the parent had neither and the supercell
  has more than 5 atoms (d12_writer's rule); a parent GRADCAL moved to the end.
  SHRINK is divided by the supercell size in each direction, so the supercell is
  sampled as densely as the parent (diamond, parent `SHRINK 12 24`: `6 12` for 2x2x2,
  `4 8` for 3x3x3); where the size does not divide the parent's value it is rounded
  up, never coarser than the parent, and a note says so. A non-diagonal `--dim`
  keeps the parent's SHRINK with a warning; `--keep-shrink` keeps it always.
- **Refused parents** (each with its reason): not 3D (use CRYSTAL's FREQCALC +
  SCELPHONO + DISPERSI, manual pp.73, 232); not terminated normally; SCF not converged;
  an optimisation without `OPT END - CONVERGED`; a FREQCALC output as the geometry
  source; CHEMOD/CHARGED/DOPING, ATOMSPIN/SPINLOCK/BETALOCK or GHOSTS (atom labels and
  electron counts do not carry over to a supercell yet); a magnetic parent (Mulliken
  spin above 0.1 on an atom) with no spin lock; an SCF block with a second END before
  the last record (CRYSTAL stops at the first, so the displaced decks would stop before
  GRADCAL); and any SCF-block keyword not known to be safe in a P1 supercell, which is
  named. An SP parent is accepted with a warning
  that its geometry is assumed relaxed; a parent cell that is not primitive (P1,
  SUPERCEL) gives a folded-bands warning.
- `--dim` takes 3 integers or a 9-integer matrix; `--amplitude` (default 0.01 A) and
  `--pm auto|yes|no` are phonopy's. An `--output-dir` must be empty (or `--force`) and
  must not be inside `workflow_outputs/`.
- `--cry-sym` (opt-in) writes the symmetry operators of each displaced cell into its
  `.ext`, as phonopy does when a `CRY_SYM` file exists (one is written to the set
  directory). The displaced cell keeps more symmetry than P1, so CRYSTAL needs fewer
  k-points: measured on CRYSTAL23 for diamond (Fd-3m, 2x2x2), the same frequencies as
  P1 to 0.04 cm-1 at about a third of the core hours. phonopy warns that some space
  groups do not work this way (e.g. R-3m); a rhombohedral parent gets a louder
  warning, and `collect` still checks that each output's printed geometry is its
  displaced cell. Compare with a P1 set before trusting a new structure type.
- **Walltime.** A displaced supercell costs far more than its parent, and a job that
  runs out of walltime leaves no forces. `generate` prints a walltime per job, at
  `mace submit`'s 32 tasks, for `mace submit DIR --walltime ...`: k-points times
  atoms cubed, calibrated on the heaviest case measured on CRYSTAL23 (diamond,
  B3LYP-D3 with SPIN and XLGRID, 2x2x2 in P1, SHRINK 6: 3160 s on 64 tasks), never
  below 2 h and at most 7 days (with a warning). For that diamond it gives 2 h at
  2x2x2 and 20 h at 3x3x3, where the measured 3x3x3 job had not finished its SCF
  after 2 h on 64 tasks. Lighter methods run faster than the estimate. It is also
  recorded as `walltime` in `mace_phonopy.json`.
- **Do not use `--track` or `--progress`** with these decks: the completion callback
  would record the displaced supercells as calculations of the parent's material.
- **collect** first checks every output: present, terminated normally, SCF converged,
  one force row per supercell atom, and the geometry CRYSTAL printed equal (to 1e-4 A)
  to that deck's displaced cell - a converged run of the wrong geometry (a stale job
  script that never staged the `.ext`; resubmit with `--overwrite-sh`) would otherwise
  give silently wrong force constants. A run that ended normally with no forces
  usually means an END before GRADCAL in its deck, and is reported as such. Any
  problem is listed per file and nothing is written; `--check` stops after the
  checks. Forces are then read one output at a time, so an output phonopy cannot
  parse is named, and `FORCE_SETS` is checked to exist before anything else is
  written (phonopy's own `phonopy-init -f` exits 0 when it fails). Then it writes `FORCE_SETS`,
  `phonopy_params.yaml`, `band.yaml` (`--band auto` uses SeeK-path on the parent's
  primitive cell, converted to CRYSTAL's reciprocal basis; or give phonopy BAND points
  with `--band "0 0 0  1/2 0 1/2, ..."` and `--band-labels`; `--band none` skips it),
  `mesh.yaml` (`--mesh 100.0` or `"a b c"`), `total_dos.dat` and
  `thermal_properties.yaml` (`--tmin/--tmax/--tstep`), and `band_dos.pdf` with `--plot`.
  Frequencies are in THz (x 33.35641 = cm-1). Modes below -0.1 THz away from Gamma are
  reported; thermal properties skip imaginary modes.
- **phonopy's own commands** (phonopy 4.7.2 syntax) work on the same set, run in its
  directory, e.g. to plot with a `band.conf` of your own:
  ```bash
  phonopy-init --crystal -f mat_opt_ph222_disp-*.out   # -> FORCE_SETS (collect writes it too)
  phonopy --factor=521.47083 --config band.conf        # band.yaml in cm-1 (default THz)
  ```
  `phonopy-init -f` exits 0 even when it cannot read an output: check that
  `FORCE_SETS` exists. Since phonopy 4 the setup options (`-d`, `--dim`, `-c`, `-f`)
  belong to `phonopy-init`; `phonopy` itself takes no `--crystal` (the calculator is
  in `phonopy_disp.yaml`); a configuration file needs `--config` (a bare
  `phonopy` with a .conf as its argument computes nothing); and there is no `--nac`:
  a `BORN` file in the directory (as `collect --nac-from` writes) switches the
  non-analytical correction on, `--nonac` off. With an explicit basis the set
  directory also holds `TEMPLATE` (and `CRY_SYM` with `--cry-sym`), so
  `phonopy-init --crystal -d --dim 2 2 2 -c ../mat_opt.out` there writes the same
  decks as `supercell-NNN.d12` for the default `--amplitude` and `--pm`, apart from
  the title and TOLDEE: phonopy always writes 10, MACE the parent's value when it is
  higher (or `--toldee`).
- **NAC** (`--nac-from`): a Gamma FREQCALC of the same material with `INTENS` and
  `INTCPHF`, same functional and basis (its `.d12` must sit beside the output). The Born
  charges and dielectric tensor are read with phonopy-crystal-born and written to
  `BORN`; the run is refused unless its cell has the same cartesian orientation, atoms
  and positions as the phonopy cell and CRYSTAL's asymmetric unit is phonopy's.

### Job Management (Direct Script Access)
```bash
# Enhanced queue management with material tracking
python mace/enhanced_queue_manager.py --d12-dir /path/to/materials --max-jobs 200

# Submit all .d12 files in directory
./mace/submission/submitcrystal23.sh

# Submit properties calculations (.d3 files)
./mace/submission/submit_prop.sh
```

### Material Tracking System
```bash
# Error recovery and automated fixes
mace recover --action recover --max-recoveries 10

# View recovery statistics
mace recover --action stats

# Workflow automation and progression
python mace/workflow/engine.py --action process

# Check workflow status for specific material
python mace/workflow/engine.py --action status --material-id material_name

# Material monitoring dashboard
mace monitor --action dashboard --interval 30

# Database health and statistics
mace database --action stats

# File organization and management
python mace/utils/file_manager.py --action organize --material-id material_name
```

### Workflow Manager
```bash
# Interactive workflow planning (recommended for first use)
python mace/run_workflow.py --interactive

# Quick start with predefined templates
python mace/run_workflow.py --quick-start --cif-dir ./cifs --workflow full_electronic

# Execute a saved workflow plan
python mace/run_workflow.py --execute workflow_plan_20250618_145837.json

# Check workflow status
python mace/run_workflow.py --status

# Show available templates
python mace/run_workflow.py --show-templates
```

### Dependency Installation

#### Core Dependencies (Required)
```bash
# Scientific computing stack
pip install numpy>=1.21.0 matplotlib>=3.5.0 ase>=3.22.0 spglib>=1.16.0

# Material tracking system dependencies (Phase 2)
pip install pyyaml>=6.0 pandas>=1.3.0
```

#### One-Line Installation
```bash
pip install numpy matplotlib ase spglib pyyaml pandas
```

#### Package Purposes
- **numpy**: Numerical computing and array operations
- **matplotlib**: Plotting, visualization, and multi-page PDFs
- **ase**: Atomic Simulation Environment for structure storage and manipulation
- **spglib**: Space group operations and symmetry analysis
- **pyyaml**: YAML configuration file handling for workflow automation
- **pandas**: Data analysis and CSV processing

#### Optional Dependencies
```bash
# Themed terminal UI (MACE falls back to plain text without it)
pip install rich

# Interactive cube/FREQ plotting engines (mace plotting --cube / --freq)
pip install plotly scipy kaleido==0.2.1 scikit-image

# For accurate band structure k-paths (highly recommended)
pip install seekpath>=2.0.0

# Finite-displacement phonons (mace phonopy)
pip install 'phonopy<5'
```

#### Package Purposes (Optional)
- **rich**: Themed terminal UI (status lines, tables, progress bars, live dashboards)
- **plotly / scipy / kaleido**: Cube and FREQ plot rendering — HTML output needs only plotly; kaleido adds static PNG/SVG/PDF export; scipy interpolates non-orthogonal cube grids
- **scikit-image**: Isosurface extraction for cube plots
- **phonopy**: `mace phonopy generate/collect` (finite-displacement phonons, NAC); 4.7.2 tested
- **seekpath**: Accurate band structure k-path generation using the HPKOT methodology (Hinuma et al., Comp. Mat. Sci. 128, 140 (2017)). Provides correct parametric k-points for non-cubic lattices and proper discontinuity detection. Without seekpath, band paths fall back to static dictionaries which are only accurate for cubic systems.

#### Installation Verification
```bash
python -c "import numpy, matplotlib, ase, spglib, yaml, pandas; print('All dependencies installed successfully')"

# Verify seekpath (optional but recommended)
python -c "import seekpath; print('seekpath available for accurate band paths')"
```

## Architecture Overview

### Workflow Categories

The codebase is organized into distinct workflow stages:

1. **Input Preparation** (`Crystal_d12/`)
   - Convert CIF files to CRYSTAL .d12 format
   - Add ghost atoms for surface calculations
   - Extract optimized geometries from completed calculations

2. **Job Management** (`mace/queue/`, `mace/submission/`)
   - SLURM-based queue management with fault tolerance
   - Automated job submission, monitoring, and recovery
   - Resource allocation and scratch space management
   - **Material tracking database** (Phase 2) - SQLite + ASE integration
   - **Automated error recovery** (Phase 2) - Configurable fixes for common errors
   - **Workflow automation** (Phase 2) - OPT → SP → BAND/DOSS/TRANSPORT/CHARGE+POTENTIAL progression

3. **Status Monitoring** (`Check_Scripts/`)
   - Categorize job completion status based on CRYSTAL error patterns
   - Automatically sort completed/errored jobs into organized folders
   - Fix common input errors (e.g., SHRINK parameters)

4. **Input Generation** (`Crystal_d3/`)
   - Generate DOS, band structure, transport, and charge density/potential input files
   - Template-based creation using crystal symmetry information
   - Basis set management for different calculation types
   - RCSR database integration for 2P and 3P periodic structures
   - Band plotting templates for different crystal systems
   - CRYSTALOptToD3.py: Unified D3 generation with basic/advanced/expert configuration modes

5. **Analysis & Visualization** (`mace plotting` / `Plotting/`, `code/Band_Alignment/`, `code/Post_Processing_Scripts/`)
   - Extract electronic properties (band gaps, work functions)
   - Generate publication-quality plots and multi-page PDFs
   - Material property analysis and comparison
   - Automated phonon band structure plotting from f25 files
   - Crystal symmetry analysis and labeling utilities

6. **Workflow Manager** (`mace/run_workflow.py`)
   - Complete end-to-end workflow planning and execution system
   - Interactive configuration with three customization levels
   - Integration with all existing tools (NewCifToD12.py, CRYSTALOptToD12.py, etc.)
   - Automated workflow templates and custom sequence design
   - SLURM resource management with intelligent defaults
   - JSON-based configuration persistence and reproducibility

### Key Design Patterns

- **File-based workflows**: Scripts process batches of files in directories
- **CSV output**: Results stored in CSV format for further analysis
- **Error-tolerant parsing**: Handles variations in CRYSTAL output files
- **Modular design**: Each script focuses on specific workflow stage
- **HPC integration**: Sophisticated SLURM job management with resource optimization

### Common File Types

- **.d12**: CRYSTAL input files for SCF calculations
- **.d3**: CRYSTAL input files for properties calculations
- **.out**: Main CRYSTAL calculation output files
- **.f9**: Binary wavefunction files (fort.9)
- **.f25**: CRYSTAL phonon calculation output files
- **.cif**: Crystallographic Information Files
- **.band**: Band structure template files for different crystal systems
- **.param**: Parameter files for different calculation types (1c, 2c, opt, soc)
- **CSV files**: Analysis results, job status tracking, and RCSR structure data
- **SQLite database**: Material tracking, calculation history, and workflow state (Phase 2)
- **YAML files**: Configuration for error recovery and workflow automation (Phase 2)

### CRYSTAL-Specific Keywords

The scripts recognize specific CRYSTAL output patterns:
- `INDIRECT ENERGY BAND GAP:` / `DIRECT ENERGY BAND GAP:` - Band gap extraction
- `TOP OF VALENCE BANDS` / `BOTTOM OF CONDUCTION BANDS` - Band edge positions
- `FERMI ENERGY` - Electronic structure analysis
- `LATTICE PARAMETERS` - Structural information
- `ETOT(AU)` - Total energy extraction
- `POSSIBLY CONDUCTING STATE` - Electronic state classification

### Bundled Basis Sets

MACE's external basis sets live in `Crystal_d12/basis_sets/`, one library file
per family, read by `Crystal_d12/basis_library.py`:

| family | menu (external basis) | what it is |
|---|---|---|
| `full.basis.doublezeta` | 1: DZVP-REV2 | pob-DZVP-rev2 (H-Br), pob-DZVP 2018 (Rb-I), shared ERD / rev2 records for Tc, Cs-Bi and Ac-Es |
| `full.basis.triplezeta` | 2: TZVP-REV2 (default) | pob-TZVP-rev2 (H-Br, Cs, Ba, Hf-Bi), pob-TZVP 2018 (Rb-I), ERD 2017 (Tc, La-Lu small-core with 4f in the valence, Ac-Es) |
| `full.basis.triplezeta.rev2022` | 4: TZVP-REV2 (2022 update: Rb-I, La-Lu) | triplezeta with X_POB_TZVP_rev2 for Rb-I (not Tc) and X_pob_TZVP_rev2 for La-Lu; La has a 46-electron core here |
| `stuttgart` | - | legacy Stuttgart-type records used by the SOC tooling |
| `pob-tzvp-rev2-soc` | - | X_pob_TZVP_rev2_SOC (STUTSC) records for two-component SOC decks; data only, `basis_library.soc_basis_record(z)` |

- **Paths.** A path such as `mace_config.DEFAULT_TZ_PATH`
  (`.../basis_sets/full.basis.triplezeta/`) still names its family, also when
  typed at menu option 3 (custom path). Only these exact paths are served from
  the library; any other directory is read from disk as numbered files.
- **Configuration files.** `"external_basis_family": "tz"`, `"dz"` or
  `"tz-rev2022"` selects a family without writing a path.
- **Provenance.** Each record carries its source set, reference, DOI and how it
  was identified. `basis_sets/PROVENANCE.md` is the generated summary.
- **Rebuilding.** `basis_sets/build_library.py` rebuilds the files: `--check`
  compares them with the committed files, `--fetch CACHE` rebuilds the families
  fetched from crystal.unito.it, and `--verify CACHE` compares the site's
  records with the bundled ones.
- **Format.** In a `.lib` file each record is stored as `#%record <n>
  bytes=<N>`, then `#%  field value` lines, then exactly N bytes of the record
  as CRYSTAL reads it.
- **Record names.** A record's name is the old file name: Z for all-electron
  records and 200 + Z for ECP records, which MACE also writes in the geometry.

### HPC Environment Requirements

- **SLURM** workload manager with `squeue`, `sbatch`, `scancel`
- **CRYSTAL17/23** quantum chemistry software
- **Intel MPI** runtime environment
- **Module system** for software environment management
- High-performance scratch storage with InfiniBand interconnect

### RCSR Database Integration

The codebase integrates with the Reticular Chemistry Structure Resource (RCSR):
- **2P Structures**: 2-periodic (layered) materials with space group and vertex coordination data
- **3P Structures**: 3-periodic (framework) materials
- **Structure Data**: Each topology includes lattice parameters, atomic positions, vertex symbols, and coordination information
- **CSV Format**: All RCSR data organized in structured CSV files for automated processing
- **Template Generation**: Automatic CRYSTAL input file generation from RCSR topologies

### Error Handling Strategies

Scripts implement multiple layers of fault tolerance:
- CRYSTAL-specific error pattern recognition
- Automatic job recovery and resubmission
- Graceful degradation for file system issues
- Atomic file operations for data integrity
- Multiple backup locations for critical status files

## Workflow Manager (Phase 3)

The workflow manager (`mace/run_workflow.py`) provides a unified interface for planning and executing complex CRYSTAL calculation workflows. It integrates all existing tools into a cohesive, user-friendly system.

### Core Components

#### **1. Interactive Workflow Planner (`mace/workflow/planner.py`)**
- **Purpose**: Plan complete calculation sequences with full configuration
- **Features**:
  - Input type detection (CIF files, existing D12s, or mixed)
  - Three-level CIF customization system
  - Workflow template selection and modification
  - SLURM resource planning with intelligent defaults
  - JSON configuration persistence

#### **2. Workflow Executor (`mace/workflow/executor.py`)**
- **Purpose**: Execute planned workflows with error handling and progress tracking
- **Features**:
  - Batch CIF conversion with timeout protection
  - Dependency-aware job submission
  - Progress monitoring and error recovery
  - File organization and cleanup

#### **3. Main Interface (`run_workflow.py`)**
- **Purpose**: User-friendly entry point for all workflow operations
- **Modes**:
  - `--interactive`: Full interactive planning
  - `--quick-start`: Rapid deployment with templates
  - `--execute`: Run saved workflow plans
  - `--status`: Monitor active workflows
  - `--show-templates`: Display available templates

### Workflow Templates

#### **Pre-defined Templates**
1. **`basic_opt`**: OPT only
   - Single geometry optimization
   - Walltime: 7 days
   
2. **`opt_sp`**: OPT → SP
   - Optimization followed by single point
   - Walltimes: 7 days (OPT), 3 days (SP)
   
3. **`full_electronic`**: OPT → SP → BAND → DOSS
   - Complete electronic structure characterization
   - Walltimes: 7d → 3d → 1d → 1d
   
4. **`double_opt`**: OPT → OPT2 → SP
   - Two-stage optimization for difficult systems
   - Enhanced convergence criteria for OPT2
   
5. **`complete`**: OPT → SP → BAND → DOSS → FREQ
   - Full characterization including vibrational analysis
   - Property extraction for every step

6. **`transport_analysis`**: OPT → SP → TRANSPORT
   - Transport properties calculation (conductivity, Seebeck coefficient)
   - Walltimes: 7d → 3d → 1d

7. **`charge_analysis`**: OPT → SP → CHARGE+POTENTIAL
   - Charge density and electrostatic potential analysis
   - Walltimes: 7d → 3d → 1d

8. **`combined_analysis`**: OPT → SP → BAND → DOSS → TRANSPORT
   - Electronic structure with transport properties
   - Walltimes: 7d → 3d → 1d → 1d → 1d

#### **Custom Workflows**
- Interactive workflow designer
- Dependency validation
- Custom calculation sequences
- Resource optimization per step

### CIF Customization Levels

#### **Level 1: Basic**
- DFT functional selection
- Basis set choice
- Uses sensible defaults for all other settings

#### **Level 2: Advanced**
- Method selection (DFT/HF)
- Functional categories (LDA, GGA, Hybrid, meta-GGA)
- Basis set types (internal/external)
- Dispersion correction options
- Spin polarization settings
- Optimization type selection

#### **Level 3: Expert**
- Full NewCifToD12.py integration
- Complete interactive configuration
- Custom tolerances and convergence criteria
- Advanced symmetry handling options

### SLURM Integration

#### **Intelligent Resource Defaults**
```bash
# Optimization calculations
Cores: 32, Memory: 5G, Walltime: 7-00:00:00, Account: mendoza_q

# Single point calculations  
Cores: 32, Memory: 4G, Walltime: 3-00:00:00, Account: mendoza_q

# Properties calculations (BAND/DOSS/TRANSPORT/CHARGE+POTENTIAL)
Cores: 28, Memory: 80G, Walltime: 2:00:00, Account: mendoza_q

# Frequency calculations
Cores: 32, Memory: 5G, Walltime: 7-00:00:00, Account: mendoza_q
```

#### **Dynamic Script Generation**
- **Individual SLURM scripts**: Each material gets its own script (e.g., `mat_1_dia.sh`)
- **Template-based generation**: Uses workflow-generated templates as base
- **Custom resource allocation**: Per-step resource configuration
- **Scratch directory management**: `$SCRATCH/{workflow_id}/step_{num}_{type}/{material}/`
- **Automatic job submission**: Direct `sbatch` execution with job ID tracking

#### **Enhanced Execution Features**
```bash
# Automatic material name extraction and cleanup
1_dia_opt_BULK_OPTGEOM_symm_CRYSTAL_OPT_symm_B3LYP-D3_POB-TZVP-REV2.d12
  ↓ Cleaned to: mat_1_dia

# Individual folder structure creation
workflow_outputs/workflow_20250618_160317/step_001_OPT/mat_1_dia/
├── mat_1_dia.d12     # Material input file
├── mat_1_dia.sh      # Individual SLURM script
├── mat_1_dia.out     # CRYSTAL output (after job completion)
└── mat_1_dia.o12345  # SLURM output file

# Customized SLURM script generation
--job-name=mat_1_dia_opt
--output=mat_1_dia_opt.o%j
export scratch=$SCRATCH/workflow_20250618_160317/step_001_OPT/mat_1_dia
```

### Configuration Management

#### **JSON Persistence**
All workflow configurations are saved as JSON files for:
- **Reproducibility**: Exact recreation of workflows
- **Sharing**: Transfer configurations between users/systems
- **Version Control**: Track changes in calculation setups
- **Batch Processing**: Automated execution of saved plans

#### **Configuration Structure**
```json
{
  "created": "2025-06-18T14:58:37",
  "input_type": "cif",
  "input_directory": "/path/to/cifs",
  "workflow_sequence": ["OPT", "SP", "BAND", "DOSS"],
  "step_configurations": {
    "OPT_1": { "source": "cif_conversion", ... },
    "SP_2": { "source": "CRYSTALOptToD12.py", ... }
  },
  "cif_conversion_config": { ... },
  "execution_settings": { ... }
}
```

`execution_settings.guessp_restart` (default `false`; only `true` turns it on):
SP, FREQ and OPTn steps built from a completed OPT or SP start their SCF from
that step's density matrix (GUESSP, with its `.f9` staged as `<job>.f20`), when
both decks have the same symmetry, atom list, basis set and spin treatment.
The engine reads the plan each time a step completes, so the setting can be
added to `workflow_configs/workflow_plan_<id>.json` of a running workflow.
Only the engine applies it: the executor's older step generator
(`WorkflowExecutor.generate_inputs_with_crystal_opt`, which normal workflows
no longer reach) writes its decks without GUESSP and says so.

### Integration with Existing Tools

#### **Tool Integration**
- **NewCifToD12.py**: CIF → D12 conversion with full configuration
- **CRYSTALOptToD12.py**: OPT → SP/FREQ generation
- **CRYSTALOptToD3.py**: Band structure / DOS / transport input generation (`--calc-type BAND|DOSS|TRANSPORT|CHARGE|POTENTIAL|CHARGE+POTENTIAL`)
- **Enhanced Queue Manager**: Job submission and monitoring
- **Material Database**: Calculation tracking and provenance

#### **File Dependencies**
The workflow manager automatically handles file dependencies:
```
CIF → OPT.d12 → OPT.out/.gui → SP.d12 → SP.out/.f9 → BAND.d3/DOSS.d3
```

### Usage Examples

#### **Interactive Planning**
```bash
python mace/run_workflow.py --interactive
# 1. Select input type (CIF/D12/Mixed)
# 2. Choose CIF customization level (Basic/Advanced/Expert)  
# 3. Select workflow template or design custom
# 4. Configure SLURM resources
# 5. Save configuration and execute
```

#### **Quick Start**
```bash
# Process CIFs with full electronic workflow
python mace/run_workflow.py --quick-start --cif-dir ./cifs --workflow full_electronic

# Use existing D12s for optimization only
python mace/run_workflow.py --quick-start --d12-dir ./d12s --workflow basic_opt
```

#### **Batch Execution**
```bash
# Execute saved workflow plan
python mace/run_workflow.py --execute workflow_plan_20250618_145837.json

# Monitor progress
python mace/run_workflow.py --status
```

### Error Handling and Recovery

#### **Error Management**
- **CIF Conversion**: Timeout protection, file existence checks
- **Job Submission**: SLURM integration with retry logic
- **File Operations**: Atomic operations with rollback capability
- **Configuration**: Validation and fallback to defaults

#### **Progress Tracking**
- Real-time workflow status monitoring
- Integration with material database for provenance
- Detailed logging and error reporting
- Recovery from partial failures

### Directory Structure

The workflow manager creates organized directory structures with individual calculation folders:

```
working_directory/
├── workflow_configs/          # JSON configuration files
│   ├── cif_conversion_config.json
│   └── workflow_plan_*.json
├── workflow_scripts/          # Generated SLURM script templates
│   ├── submitcrystal23_opt_1.sh
│   └── submit_prop_band_3.sh
├── workflow_inputs/           # Initial input files
│   └── step_001_OPT/
├── workflow_outputs/          # Individual calculation folders
│   └── workflow_20250618_160317/
│       └── step_001_OPT/
│           ├── mat_1_dia/      # Individual material folder
│           │   ├── mat_1_dia.d12
│           │   ├── mat_1_dia.sh  # Individual SLURM script
│           │   └── mat_1_dia.out
│           ├── mat_2_dia2/     # Individual material folder
│           │   ├── mat_2_dia2.d12
│           │   ├── mat_2_dia2.sh
│           │   └── mat_2_dia2.out
│           └── ...
└── temp/                      # Temporary files
```

#### Individual Calculation Management
Each material gets its own isolated calculation environment:
- **Unique folders**: `mat_1_dia/`, `mat_2_dia2/`, etc.
- **Individual SLURM scripts**: Generated from templates with material-specific settings
- **Isolated scratch space**: `$SCRATCH/workflow_ID/step_001_OPT/material_name/`
- **Independent job submission**: Each material submitted as separate SLURM job

### Best Practices

#### **Workflow Planning**
1. **Start with Templates**: Use predefined templates as starting points
2. **Resource Planning**: Consider walltime requirements for your systems
3. **Test with Small Sets**: Validate workflows with 1-2 structures first
4. **Save Configurations**: Persist successful setups for reuse

#### **Execution Management**
1. **Monitor Progress**: Use `--status` to track workflow execution
2. **Error Recovery**: Check logs and use error recovery tools
3. **Resource Optimization**: Adjust SLURM settings based on system performance
4. **Database Integration**: Use material tracking for large studies

### Advanced Features

#### **Custom Workflow Design**
- Interactive workflow builder
- Dependency validation
- Resource optimization suggestions
- Custom calculation parameters

#### **Batch Processing**
- Multiple workflow execution
- Resource sharing and optimization
- Progress aggregation
- Parallel execution management

The workflow manager ties the CRYSTAL automation tools together behind one interface for complex calculation workflows while keeping the flexibility of the underlying components.

### Material Tracking System (Phase 2 - IMPLEMENTED)

The enhanced system provides material lifecycle tracking:

#### **Core Components**
- **`mace/database/`**: SQLite database with ASE integration for structure storage
- **`mace/queue/manager.py`**: Enhanced queue manager with material tracking and callback system
- **`mace/recovery/`**: Automated error detection and recovery with YAML configuration
- **`mace/workflow/engine.py`**: Orchestrates OPT → SP → BAND/DOSS/TRANSPORT/CHARGE+POTENTIAL workflow progression
- **`mace/utils/file_manager.py`**: Organized file management by material ID and calculation type
- **`mace/material_monitor.py`**: Real-time monitoring dashboard and health checks
- **`mace/utils/property_extractor.py`**: Complete property extraction from CRYSTAL output files
- **`mace/utils/formula_extractor.py`**: Chemical formula and space group extraction from input/output files

#### **Key Features**
- **Material ID Consistency**: Handles complex file naming from NewCifToD12.py and CRYSTALOptToD12.py
- **Isolated Script Execution**: Creates clean directories for CRYSTALOptToD3.py (BAND/DOSS/TRANSPORT) requirements
- **Automated Workflow Progression**: OPT completion triggers SP generation, SP completion triggers BAND/DOSS/TRANSPORT/CHARGE+POTENTIAL
- **Error Recovery**: Configurable fixes for SHRINK errors, memory issues, convergence problems
- **Enhanced Callback System**: Multi-location queue manager detection checking both local and parent directories
- **Directory Organization**: `base_dir/calc_type/` structure for efficient file management
- **Database Integration**: Complete calculation history and provenance tracking

#### **Database Schema**
```sql
materials:        # Core material information and metadata
calculations:     # Individual calculation tracking with SLURM integration  
properties:       # Extracted material properties from completed calculations
files:           # File association and integrity tracking
workflow_templates: # Reusable calculation sequences
workflow_instances: # Active workflow state tracking
```

#### **Configuration Files**
- **`recovery_config.yaml`**: Error recovery strategies and retry policies
- **`workflows.yaml`**: Workflow definitions and resource requirements
- **`materials.db`**: SQLite database for material tracking
- **`structures.db`**: ASE database for atomic structure storage

#### **Enhanced Callback System**
All SLURM job scripts now include an enhanced callback mechanism that automatically detects and calls the appropriate queue manager upon job completion:

**Multi-Location Detection:**
- Checks for queue manager in multiple locations
- Falls back through parent directories
- Also checks for legacy `crystal_queue_manager.py`
- Ensures workflows continue regardless of execution context

**Callback Logic** (abridged from the generated script; see `mace/submission/submitcrystal23.sh`):
```bash
# Callback appended to all SLURM scripts: resolve the queue manager, then
# run it in completion mode. Search order:
#   1. $MACE_HOME/mace/queue/manager.py   (or $MACE_HOME/enhanced_queue_manager.py)
#   2. mace/queue/manager.py relative to the job dir (walking up parents)
#   3. legacy enhanced_queue_manager.py / crystal_queue_manager.py fallbacks
if [ ! -z "$QUEUE_MANAGER" ]; then
    python "$QUEUE_MANAGER" --max-jobs 250 --reserve 30 --max-submit 5 --callback-mode completion
fi
```

**Benefits:**
- Works with various directory structures and workflow execution contexts
- No manual intervention required for job progression
- Supports both current and legacy queue manager locations

### Complete File Storage and Provenance System

The file storage system provides calculation provenance tracking, addressing the need to store D12/D3 input files with all settings and maintain complete calculation history.

#### **File Storage**

**Purpose**: Store and manage all calculation files with complete settings extraction and integrity verification.

**Key Features:**
- **Complete File Preservation**: Stores all calculation files (D12/D3 inputs, outputs, binary files, scripts)
- **Settings Extraction**: Automatically extracts and parses all CRYSTAL calculation parameters from D12/D3 files
- **Integrity Verification**: SHA256 checksums ensure file integrity over time
- **Organized Storage**: Files organized by calculation ID with metadata preservation
- **Database Integration**: File records and extracted settings stored in materials database

#### **Supported File Types**
```
Input Files:     .d12, .d3, .input files with complete settings extraction
Output Files:    .out, .output, .log files with property extraction coordination
Binary Files:    .f9, .f25, fort.9, fort.25, .wf, .prop (wavefunction, phonon data)
Property Files:  .BAND, .DOSS, .OPTC, .ELPH calculation results
Script Files:    .sh, .slurm, .job SLURM submission scripts
Plot Files:      .png, .pdf, .eps, .svg generated visualizations
Data Files:      .csv, .json, .yaml, .xml analysis results
Config Files:    .conf, .cfg, .ini, .param configuration files
```

#### **Settings Extraction Capabilities**
The system automatically extracts settings from D12/D3 input files:

**CRYSTAL Keywords**: OPTGEOM, DFT, EXCHANGE, CORRELAT, NONLOCAL, SHRINK, TOLINTEG, etc.
**Calculation Parameters**: SHRINK factors, TOLINTEG values, TOLDEE, MAXCYCLE, FMIXING
**Basis Set Information**: Internal vs external basis sets, basis set file references
**Geometry Settings**: Optimization parameters, convergence criteria
**Exchange-Correlation**: Functional types, dispersion corrections
**SCF Parameters**: Convergence thresholds, mixing algorithms

#### **Usage**
File storage runs automatically when the queue manager registers a completed
job — no manual step is needed. Stored files, extracted settings, and
properties are queried through the database command:

```bash
# Database health and statistics
mace database --action stats

# Query calculations for a material
mace database --action query --material-id diamond

# Show extracted properties
mace database --action properties --material-id diamond

# Export results (CSV / Excel)
mace database --action export --format excel --output results.xlsx
```

#### **Automatic Integration**
The file storage system is automatically integrated into the enhanced queue manager:

**On Job Completion:**
1. **File Storage**: All calculation files automatically stored with settings extraction
2. **Property Extraction**: Properties extracted from output files and stored in database
3. **Material Updates**: Formula and space group information updated
4. **Workflow Progression**: Next calculation steps planned and submitted

**Storage Organization:**
```
calculation_storage/
├── calculations/
│   ├── calc_diamond_opt_001/
│   │   ├── diamond.d12          # Input file
│   │   ├── diamond.out          # Output file
│   │   ├── diamond.sh           # SLURM script
│   │   ├── fort.9               # Wavefunction
│   │   ├── fort.25              # Phonon data
│   │   └── calc_diamond_opt_001_metadata.json
│   └── calc_diamond_sp_002/
│       ├── diamond_sp.d12
│       ├── diamond_sp.out
│       └── ...
└── materials/
    └── diamond/
        ├── material_metadata.json
        └── ...
```

#### **Database Schema Extensions**
The file storage system extends the existing database schema:

**Files Table**: Complete file tracking with checksums and metadata
**Settings Storage**: Extracted D12/D3 settings stored in calculations.settings_json
**File Categories**: Automatic categorization by importance and type
**Integrity Tracking**: Checksum verification and corruption detection

#### **Benefits**
- **Complete Provenance**: Every calculation can be exactly reproduced
- **Settings Analysis**: Parameter analysis across materials and calculation types
- **Error Debugging**: Access to all files for failed calculation analysis
- **Data Mining**: Systematic analysis of successful parameter combinations
- **Backup and Recovery**: Complete calculation reconstruction capability
- **Research Reproducibility**: Full documentation of all calculation parameters

## Development Notes

- Scripts use **Python 3.x** with scientific computing stack (numpy, matplotlib, ase)
- Plotting pins a headless matplotlib backend (`Agg`) so it runs on compute nodes
- All file paths should be absolute, not relative
- Scripts designed for batch processing of hundreds of calculations
- Job queue manager implements intelligent throttling and resource management
- External basis sets are one library file per family in `Crystal_d12/basis_sets/` (see Bundled Basis Sets); records are keyed by atomic number (200 + Z for ECP records)
- RCSR database provides extensive 2P and 3P periodic structure templates with symmetry data
- Phonon calculations supported with automatic band structure plotting from f25 files
- Crystal symmetry analysis integrated for automated structure classification