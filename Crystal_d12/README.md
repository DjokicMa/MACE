# CRYSTAL Input File Generation Suite (D12 Creation Scripts)

A comprehensive suite of Python scripts for generating and managing CRYSTAL quantum chemistry input files (.d12) from CIF files and optimized geometries. This modular system supports the entire CRYSTAL calculation workflow including geometry optimization, single point calculations, and advanced frequency/phonon analyses.

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
- **DFT Functional Library**: Comprehensive support including dispersion corrections
- **Database Integration**: Automatic extraction and storage of all calculation parameters
- **Workflow Automation**: Seamless integration with enhanced queue manager

---

## Architecture

### Modular Design

The refactored system separates concerns into focused modules:

```
Core Scripts:
├── NewCifToD12.py         # CIF → D12 conversion (main entry point)
├── CRYSTALOptToD12.py     # Extract optimized geometry → new D12
├── create_d12_w-ghosts.py # Automatic ghost atom insertion
└── manual_create_d12_w-ghosts.py # Manual ghost atom insertion

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

# Single file conversion
python NewCifToD12.py --cif_file structure.cif

# Specify calculation type
python NewCifToD12.py --calc_type OPT --cif_dir ./cifs

# Using JSON configuration
python NewCifToD12.py --options_file standard_dft_opt.json structure.cif

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
# Use predefined configuration
python NewCifToD12.py --options_file example_configs/standard_dft_opt.json structure.cif

# Create and save custom configuration
python NewCifToD12.py structure.cif --save_options --options_file my_config.json

# Batch processing
python NewCifToD12.py --batch --options_file my_config.json --cif_dir ./cifs
```

**Usage with CRYSTALOptToD12.py:**
```bash
# Apply configuration to optimized structure
python CRYSTALOptToD12.py --config-file example_configs/high_accuracy_sp.json optimized.out

# Save current settings
python CRYSTALOptToD12.py optimized.out --save-options --options-file sp_settings.json
```

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
python create_d12_w-ghosts.py
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
python manual_create_d12_w-ghosts.py
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

# Process single file
python CRYSTALOptToD12.py --input structure_opt.out --calc_type SP

# Batch processing
python CRYSTALOptToD12.py --input_dir ./optimized --calc_type FREQ

# Shared settings for batch
python CRYSTALOptToD12.py --input_dir ./optimized --shared_settings
```

**Advanced Features:**
- **Intelligent Defaults**: Suggests appropriate settings based on previous calculation
- **Tolerance Optimization**: Recommends tighter tolerances for frequency calculations
- **Error Handling**: Graceful fallback when output parsing fails
- **Setting Merge Logic**: Prioritizes D12 file for critical parameters like TOLINTEG

**Integration:**
- Works seamlessly with workflow automation
- Outputs compatible with enhanced queue manager
- Preserves material ID and calculation history  

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

**Purpose:** Comprehensive frequency calculation configuration module.

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
  - **Extended Bravais Lattice Detection**: Automatically determines correct lattice variant (e.g., aP2 vs aP3, oF1 vs oF2 vs oF3)
  - **Inversion Symmetry Detection**: Selects appropriate k-paths for centrosymmetric vs non-centrosymmetric structures
  - **K-path Source Tracking**: Labels phonon band titles with path source:
    - `SeeKPath (w.I)` - SeeK-path with inversion symmetry
    - `SeeKPath (no.I)` - SeeK-path without inversion symmetry  
    - `Literature` - Literature-based paths
    - `default` - Default/standard paths
  - **Discontinuous Path Support**: Proper handling of "|" separators in k-paths

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
- Robust error handling for incomplete files

### `d12_writer.py`

**Purpose:** Write properly formatted D12 files.

**Features:**
- Format geometries with correct precision
- Write calculation keywords in proper order
- Handle special formatting requirements
- Validate settings before writing
- Support all CRYSTAL input sections

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
- **Phonon Path Generation**: Full integration with SeeK-path methodology including inversion symmetry detection

### Phonon Band Structure Updates
- **Extended Bravais Support**: Cell parameter-aware extended Bravais lattice determination for accurate k-paths
- **Inversion Symmetry**: Automatic detection from CRYSTAL output for proper k-point selection
- **Source Labeling**: Clear indication of k-path source (SeeK-path w/wo inversion, literature, default) in titles
- **Non-centrosymmetric Paths**: Support for primed k-points (X', Y', Z', etc.) when needed
- **Discontinuous Paths**: Correct handling of "|" separators for non-connected k-path segments

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
- Python 3.7+
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
- **Added**: Comprehensive convergence menu with detailed explanations
- **Improved**: Clear descriptions of TOLDEG, TOLDEX, TOLDEE, MAXCYCLE
- **Enhanced**: Context-appropriate recommendations for each level

#### Geometry Tolerance Adjustments
- **Refined**: Better progression between convergence levels:
  - Standard: TOLDEG=0.0003, TOLDEX=0.0012, TOLDEE=7
  - Tight: TOLDEG=0.0001, TOLDEX=0.0004, TOLDEE=8 (NEW - 3x tighter)
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
