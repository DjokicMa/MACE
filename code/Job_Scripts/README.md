# CRYSTAL Workflow System Documentation

This is the comprehensive documentation for the CRYSTAL quantum chemistry workflow system, providing automated calculation management on SLURM HPC clusters.

## Table of Contents

1. [System Overview](#system-overview)
2. [Quick Start](#quick-start)
3. [Workflow Manager](#workflow-manager)
4. [Material Tracking System](#material-tracking-system)
5. [Property Extraction](#property-extraction)
6. [Error Recovery](#error-recovery)
7. [Advanced Features](#advanced-features)
8. [Directory Structure](#directory-structure)
9. [Recent Updates](#recent-updates)
10. [Troubleshooting](#troubleshooting)

---

## System Overview

The CRYSTAL workflow system provides comprehensive automation for quantum chemistry calculations using CRYSTAL17/23 software on SLURM HPC clusters. It manages the complete calculation pipeline from CIF files to final property extraction.

### Key Components

- **Workflow Manager** (`run_workflow.py`): Unified interface for planning and executing complex calculation sequences
- **Enhanced Queue Manager**: SLURM job management with material tracking and automated workflow progression
- **Material Database**: SQLite + ASE integration for complete structure and calculation tracking
- **Property Extraction**: Automated extraction of 100+ electronic, structural, and thermodynamic properties
- **Error Recovery**: Automated detection and recovery from common calculation errors

### Supported Workflows

The system supports arbitrary workflow sequences including:
- **Basic**: `OPT` only
- **Standard**: `OPT → SP → BAND → DOSS`
- **Complete**: `OPT → SP → BAND → DOSS → FREQ`
- **Multi-stage**: `OPT → OPT2 → SP → OPT3 → SP2 → BAND → DOSS → FREQ`
- **Complex**: `SP → OPT → SP2 → BAND → DOSS → OPT2 → SP3 → FREQ → OPT3`

---

## Quick Start

### Basic Workflow Execution

```bash
# Process CIFs with full electronic characterization
python run_workflow.py --quick-start \
  --cif-dir ./cifs \
  --workflow full_electronic

# Interactive planning for complex workflows
python run_workflow.py --interactive

# Execute saved workflow
python run_workflow.py --execute workflow_plan_20250618_145837.json

# Monitor progress
python enhanced_queue_manager.py --status
```

### Installation

```bash
# Core dependencies
pip install numpy matplotlib ase spglib PyPDF2 pyyaml pandas

# Verify installation
python -c "import numpy, matplotlib, ase, spglib, PyPDF2, yaml, pandas; print('All dependencies installed successfully')"
```

---

## Workflow Manager

### Operating Modes

#### 1. Interactive Mode
Full workflow customization with guided configuration:
```bash
python run_workflow.py --interactive
```

Features:
- Input type selection (CIF/D12/Mixed)
- Three-level CIF customization (Basic/Advanced/Expert)
- Custom workflow sequence design
- SLURM resource planning
- Configuration persistence

#### 2. Quick Start Mode
Rapid deployment with templates:
```bash
python run_workflow.py --quick-start --cif-dir ./cifs --workflow full_electronic
```

#### 3. Execute Mode
Run saved configurations:
```bash
python run_workflow.py --execute workflow_plan_20250623_101909.json
```

### CIF Customization Levels

1. **Basic**: DFT functional and basis set selection
2. **Advanced**: Full method configuration including spin, dispersion, convergence
3. **Expert**: Complete NewCifToD12.py integration with all features

### Workflow Templates

| Template | Sequence | Typical Walltime |
|----------|----------|------------------|
| `basic_opt` | OPT | 7 days |
| `opt_sp` | OPT → SP | 7d + 3d |
| `full_electronic` | OPT → SP → BAND → DOSS | 7d → 3d → 1d → 1d |
| `double_opt` | OPT → OPT2 → SP | 7d → 7d → 3d |
| `complete` | OPT → SP → BAND → DOSS → FREQ | Full characterization |

### Workflow Features

#### Generalized Numbered Calculations
- Supports unlimited numbered calculations (OPT1-OPT99, SP1-SP99, etc.)
- Dynamic dependency resolution based on workflow sequence
- Smart input source selection (CIF vs previous calculation)

#### Intelligent CIF Generation
The system automatically generates from CIF when:
- SP or OPT is the first calculation in workflow
- OPT appears but no prior OPT exists
- OPT follows non-geometry calculations (FREQ, BAND, DOSS)

#### Dependency Management
- **Timing**: Workflow sequence determines when calculations run
- **Input Sources**: Smart selection based on calculation requirements
  - FREQ always uses highest completed OPT geometry
  - OPT after FREQ/BAND/DOSS uses highest completed OPT
  - BAND/DOSS use most recent wavefunction (SP or OPT)

### Directory Structure

```
working_directory/
├── workflow_configs/          # JSON configuration files
│   ├── workflow_plan_*.json   # Saved workflow plans
│   └── cif_conversion_config.json
├── workflow_scripts/          # Generated SLURM script templates
├── workflow_outputs/          # Individual calculation folders
│   └── workflow_YYYYMMDD_HHMMSS/
│       └── step_NNN_TYPE/
│           └── material_name/
│               ├── material.d12    # Input file
│               ├── material.sh     # Individual SLURM script
│               ├── material.out    # CRYSTAL output
│               └── .workflow_metadata.json
└── temp/                      # Temporary processing
```

---

## Material Tracking System

### Database Schema

```sql
materials:          # Core material information with formula, space group
calculations:       # Individual calculation tracking with SLURM integration
properties:         # Extracted material properties (100+ properties)
files:             # File associations with SHA256 checksums
workflow_templates: # Reusable calculation sequences
workflow_instances: # Active workflow state tracking
```

### Key Features

- **Material ID Consistency**: Handles complex file naming from NewCifToD12.py and CRYSTALOptToD12.py
- **Complete Provenance**: Every file stored with settings extraction and checksums
- **Automated Progression**: Callbacks trigger next workflow steps automatically
- **Enhanced Callbacks**: Multi-location queue manager detection for flexible deployment

### File Storage System

The system automatically stores all calculation files with complete settings extraction:

**Supported File Types**:
- Input files: `.d12`, `.d3` with complete CRYSTAL settings extraction
- Output files: `.out`, `.log` with property extraction
- Binary files: `.f9`, `.f25`, `fort.9`, `fort.25` (wavefunctions, phonons)
- Property files: `.BAND`, `.DOSS`, `.OPTC` calculation results
- Scripts: `.sh`, `.slurm` SLURM submission scripts
- Visualizations: `.png`, `.pdf`, `.eps` plots
- Data files: `.csv`, `.json`, `.yaml` analysis results

**Usage**:
```bash
# Store calculation files
python file_storage_manager.py --store /path/to/calc --calc-id calc_001

# Query stored files
python query_stored_files.py --material-id diamond

# Verify integrity
python file_storage_manager.py --verify --calc-id calc_001
```

---

## Property Extraction

### Comprehensive Property Set (100+ Properties)

#### Electronic Properties
- Band gaps: direct, indirect, fundamental (with k-points)
- Band edges: VBM/CBM positions and k-points
- Work functions and electron affinity
- Fermi energy and DOS at E_F
- Electronic classification (metal/semiconductor/insulator)
- Spin states and magnetic moments
- Effective masses (electron/hole) from band curvature
- Charge distribution and Mulliken charges

#### Structural Properties
- Lattice parameters (a, b, c, α, β, γ)
- Cell volume and density
- Space group and symmetry operations
- Optimized atomic positions
- Bond lengths and angles
- Coordination numbers

#### Thermodynamic Properties
- Total energy (Hartree, eV, kJ/mol)
- Formation energy
- Optimization convergence metrics
- Vibrational frequencies and modes
- Zero-point energy
- Thermodynamic functions (if FREQ available)

#### Computational Details
- DFT functional and basis set
- k-point grid (SHRINK factors)
- SCF convergence (TOLDEE)
- Calculation time and resources
- CRYSTAL version and build

### Advanced Electronic Analysis

The system includes sophisticated analysis for:

#### Effective Mass Calculations
- Real effective mass from band structure: m* = ℏ²/|d²E/dk²|
- Finite difference method for derivatives
- Separate electron and hole masses
- Sanity checks (0.01 to 10 m_e)

#### Metal/Semiconductor Classification
```
Classification thresholds:
- Metal: Band gap < 0.027 eV AND DOS(E_F) > 0.05 × average
- Semimetal: Band gap < 0.027 eV AND DOS(E_F) < 0.05 × average
- Semiconductor: 0.027 eV < gap < 3.0 eV
- Insulator: gap > 3.0 eV
```

---

## Error Recovery

### Automated Error Detection and Recovery

The system handles common CRYSTAL errors automatically:

1. **SHRINK Errors**: Automatic k-point grid adjustment
2. **Memory Issues**: Resource allocation optimization
3. **Convergence Problems**: SCF parameter adjustment
4. **Symmetry Errors**: Tolerance relaxation
5. **Basis Set Issues**: Automatic fixes for common problems

### Recovery Configuration

`recovery_config.yaml` defines strategies:
```yaml
error_patterns:
  shrink_error:
    pattern: "SHRINK FACTORS"
    fix: adjust_shrink
    max_attempts: 3
    
  memory_error:
    pattern: "INSUFFICIENT MEMORY"
    fix: increase_memory
    resource_adjustments:
      memory: "2x"
```

### Usage

```bash
# Enable automatic recovery
python error_recovery.py --action recover --max-recoveries 10

# Check recovery statistics
python error_recovery.py --action stats

# Manual recovery for specific material
python error_recovery.py --material-id failing_material --fix shrink_error
```

---

## Advanced Features

### Race Condition Prevention

The system includes comprehensive race condition fixes:

1. **Distributed Locking**: File-based locks prevent multiple queue managers
2. **Database WAL Mode**: SQLite Write-Ahead Logging for concurrent access
3. **Unique Directory Generation**: UUID-based naming prevents collisions
4. **Randomized Callbacks**: Reduces simultaneous callback conflicts

### Expert Configuration

For OPT2, OPT3, etc., expert configuration files allow complete control:
```json
{
  "functional": "B3LYP",
  "dispersion": true,
  "basis_set": "POB-TZVP-REV2",
  "tolerances": {
    "TOLDEE": 9,
    "TOLINTEG": [8, 8, 8, 8, 16]
  }
}
```

### Workflow Modification

The system supports complex workflow patterns:
- Sequential optimizations with different functionals
- Parallel property calculations
- Conditional workflow branches
- Custom calculation sequences

---

## Directory Structure

### Current Organization

```
Job_Scripts/
├── README.md                          # This file
├── core/                              # Main production files
│   ├── enhanced_queue_manager.py      # Primary queue manager
│   ├── material_database.py           # Database engine
│   ├── workflow_engine.py             # Workflow orchestration
│   ├── workflow_planner.py            # Interactive planning
│   ├── workflow_executor.py           # Execution engine
│   └── run_workflow.py                # Main entry point
├── utils/                             # Utility scripts
│   ├── populate_completed_jobs.py     # Database population
│   ├── error_detector.py              # Error detection
│   └── file_storage_manager.py        # File management
├── config/                            # Configuration files
│   ├── recovery_config.yaml           # Error recovery config
│   └── workflows.yaml                 # Workflow definitions
└── working/                           # Active data
    ├── materials.db                   # Material database
    └── workflow_outputs/              # Calculation outputs
```

---

## Recent Updates

### Workflow Engine Enhancements (Phase 3)

1. **Generalized Calculations**: Support for unlimited numbered calculations (OPT1-99, SP1-99)
2. **Smart CIF Generation**: Automatic detection when CIF source is needed
3. **Expert Config Support**: CRYSTALOptToD12.py now accepts --config-file parameter
4. **Input Validation**: Comprehensive validation prevents crashes from invalid user input
5. **Dependency Intelligence**: Workflows follow exact sequence with smart source selection

### Key Fixes Applied

- ✅ Workflow progression follows planned sequence exactly
- ✅ Expert configurations properly applied to OPT2/OPT3
- ✅ FREQ calculations use correct OPT source
- ✅ OPT after FREQ/BAND/DOSS handled correctly
- ✅ Input validation prevents workflow crashes
- ✅ Parallel calculations (BAND+DOSS) execute properly

---

## Troubleshooting

### Common Issues and Solutions

#### Module Loading
```bash
# Ensure CRYSTAL modules are loaded
module load crystal23
module list  # Verify loaded modules
```

#### Database Locks
```bash
# If database is locked
python material_database.py --unlock
# Or wait for timeout (60 seconds)
```

#### Workflow Not Progressing
```bash
# Check workflow metadata
cat workflow_outputs/*/step_*/*/.workflow_metadata.json

# Verify callback is working
tail -n 50 *.o*  # Check SLURM output files
```

#### Memory Issues
```bash
# Increase memory in workflow planning
# Or edit recovery_config.yaml for automatic adjustment
```

### Debug Commands

```bash
# Comprehensive status check
python workflow_status.py --debug

# Test without submission
python run_workflow.py --dry-run

# Check specific calculation
python show_properties.py --calc-id calc_001

# Database integrity check
python material_database.py --check-integrity
```

### Log Files

Important logs to check:
- SLURM output: `*.o*` files
- Workflow logs: `workflow_outputs/*/workflow.log`
- Error logs: `workflow_outputs/*/step_*/*/error.log`
- Database log: `materials.db.log`

---

## External Dependencies and Integration

### External Scripts Called

The workflow system integrates with several external scripts:

#### From Crystal_To_CIF Directory:
- **NewCifToD12.py** - Converts CIF files to CRYSTAL D12 format
- **CRYSTALOptToD12.py** - Generates SP/FREQ inputs from optimized structures  
- **d12creation.py** - Shared utilities and constants

#### From Creation_Scripts Directory:
- **create_band_d3.py** - Generates band structure input files
- **alldos.py** - Generates density of states input files

### Script Location Resolution

The system uses multiple fallback paths to locate scripts:
1. Current working directory
2. Parent directories (up to 5 levels)
3. Hardcoded paths relative to script location
4. Dynamic sys.path modifications

### Configuration Files

#### Active Configuration:
- **recovery_config.yaml** - Actively used by error_recovery.py for automated error recovery
  - Defines recovery strategies for 8+ error types
  - Customizable retry limits and parameters
  - Falls back to defaults if not found

#### Note on workflows.yaml:
- The file `workflows.yaml` exists but is not currently loaded by the system
- All workflow definitions are hardcoded in the Python scripts
- This may be implemented in future versions for external workflow configuration

---

## Best Practices

1. **Start Small**: Test workflows with 1-2 structures first
2. **Use Templates**: Build on predefined workflow templates
3. **Save Configurations**: Keep successful workflow JSONs for reuse
4. **Monitor Progress**: Use `enhanced_queue_manager.py --status` regularly
5. **Enable Recovery**: Use automatic error recovery for production runs
6. **Regular Backups**: Backup `materials.db` and workflow configurations
7. **Resource Planning**: Consider material complexity when setting walltimes

---

## Support and Contributing

For issues or questions:
1. Check this README and troubleshooting section
2. Review log files for error messages
3. Check the [CLAUDE.md](../../CLAUDE.md) file for codebase guidance

When reporting issues, include:
- Workflow configuration JSON
- Error messages from logs
- SLURM job IDs
- Material IDs affected

---

*This documentation represents the current state of the CRYSTAL workflow system with all recent enhancements and fixes applied.*