# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a scientific computing toolkit for **CRYSTAL** quantum chemistry software workflows. CRYSTAL is used for periodic and molecular calculations of electronic structure, properties, and materials science applications. The codebase provides automation scripts, utilities, and analysis tools for the entire CRYSTAL computational workflow on SLURM HPC clusters.

## Common Commands

### Running Scripts
All scripts are Python-based and run directly:
```bash
python script_name.py
```

### Job Management (SLURM Environment)
```bash
# Enhanced queue management with material tracking (NEW - Phase 2)
python enhanced_queue_manager.py --base-dir /path/to/materials --max-jobs 200

# Legacy queue management (still supported)
./crystal_queue_manager.py --max-jobs 200 --reserve 20

# Check job status
./crystal_queue_manager.py --status

# Submit all .d12 files in directory
./submitcrystal23.py

# Submit properties calculations (.d3 files)
./submit_prop.py

# Cancel jobs above certain ID
./cancel-jobs.sh 150000

# Navigate to job scratch directory
source cd_job.sh job_output.o
```

### Material Tracking System (NEW - Phase 2)
```bash
# Error recovery and automated fixes
python error_recovery.py --action recover --max-recoveries 10

# View recovery statistics
python error_recovery.py --action stats

# Workflow automation and progression
python workflow_engine.py --action process

# Check workflow status for specific material
python workflow_engine.py --action status --material-id material_name

# Material monitoring dashboard
python material_monitor.py --action dashboard --interval 30

# Database health and statistics
python material_monitor.py --action stats

# File organization and management
python crystal_file_manager.py --action organize --material-id material_name
```

### Dependency Installation

#### Core Dependencies (Required)
```bash
# Scientific computing stack
pip install numpy>=1.21.0 matplotlib>=3.5.0 ase>=3.22.0 spglib>=1.16.0 PyPDF2>=2.0.0

# Material tracking system dependencies (Phase 2)
pip install pyyaml>=6.0 pandas>=1.3.0
```

#### One-Line Installation
```bash
pip install numpy matplotlib ase spglib PyPDF2 pyyaml pandas
```

#### Package Purposes
- **numpy**: Numerical computing and array operations
- **matplotlib**: Plotting, visualization, and multi-page PDFs
- **ase**: Atomic Simulation Environment for structure storage and manipulation
- **spglib**: Space group operations and symmetry analysis
- **PyPDF2**: PDF processing and generation
- **pyyaml**: YAML configuration file handling for workflow automation
- **pandas**: Data analysis and CSV processing

#### Optional Dependencies
```bash
# For extended functionality
pip install scipy>=1.7.0 scikit-learn>=1.0.0
```

#### Installation Verification
```bash
python -c "import numpy, matplotlib, ase, spglib, PyPDF2, yaml, pandas; print('All dependencies installed successfully')"
```

## Architecture Overview

### Workflow Categories

The codebase is organized into distinct workflow stages:

1. **Input Preparation** (`Crystal_To_CIF/`)
   - Convert CIF files to CRYSTAL .d12 format
   - Add ghost atoms for surface calculations
   - Extract optimized geometries from completed calculations

2. **Job Management** (`Job_Scripts/`)
   - SLURM-based queue management with fault tolerance
   - Automated job submission, monitoring, and recovery
   - Resource allocation and scratch space management
   - **Material tracking database** (Phase 2) - SQLite + ASE integration
   - **Automated error recovery** (Phase 2) - Configurable fixes for common errors
   - **Workflow automation** (Phase 2) - OPT → SP → BAND/DOSS progression

3. **Status Monitoring** (`Check_Scripts/`)
   - Categorize job completion status based on CRYSTAL error patterns
   - Automatically sort completed/errored jobs into organized folders
   - Fix common input errors (e.g., SHRINK parameters)

4. **Input Generation** (`Creation_Scripts/`)
   - Generate DOS, band structure, and transport property input files
   - Template-based creation using crystal symmetry information
   - Basis set management for different calculation types
   - RCSR database integration for 2P and 3P periodic structures
   - Band plotting templates for different crystal systems

5. **Analysis & Visualization** (`Plotting_Scripts/`, `Band_Alignment/`, `Post_Processing_Scripts/`)
   - Extract electronic properties (band gaps, work functions)
   - Generate publication-quality plots and multi-page PDFs
   - Comprehensive material property analysis and comparison
   - Automated phonon band structure plotting from f25 files
   - Crystal symmetry analysis and labeling utilities

### Key Design Patterns

- **File-based workflows**: Scripts process batches of files in directories
- **CSV output**: Results stored in CSV format for further analysis
- **Error-tolerant parsing**: Robust handling of CRYSTAL output file variations
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

### HPC Environment Requirements

- **SLURM** workload manager with `squeue`, `sbatch`, `scancel`
- **CRYSTAL17/23** quantum chemistry software
- **Intel MPI** runtime environment
- **Module system** for software environment management
- High-performance scratch storage with InfiniBand interconnect

### RCSR Database Integration

The codebase includes comprehensive integration with the Reticular Chemistry Structure Resource (RCSR):
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

### Material Tracking System (Phase 2 - IMPLEMENTED)

The enhanced system provides comprehensive material lifecycle tracking:

#### **Core Components**
- **`material_database.py`**: SQLite database with ASE integration for structure storage
- **`enhanced_queue_manager.py`**: Extended queue manager with material tracking
- **`error_recovery.py`**: Automated error detection and recovery with YAML configuration
- **`workflow_engine.py`**: Orchestrates OPT → SP → BAND/DOSS workflow progression
- **`crystal_file_manager.py`**: Organized file management by material ID and calculation type
- **`material_monitor.py`**: Real-time monitoring dashboard and health checks

#### **Key Features**
- **Material ID Consistency**: Handles complex file naming from NewCifToD12.py and CRYSTALOptToD12.py
- **Isolated Script Execution**: Creates clean directories for alldos.py and create_band_d3.py requirements
- **Automated Workflow Progression**: OPT completion triggers SP generation, SP completion triggers BAND/DOSS
- **Error Recovery**: Configurable fixes for SHRINK errors, memory issues, convergence problems
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

## Development Notes

- Scripts use **Python 3.x** with scientific computing stack (numpy, matplotlib, ase)
- **TkAgg backend** configured for matplotlib compatibility
- All file paths should be absolute, not relative
- Scripts designed for batch processing of hundreds of calculations
- Job queue manager implements intelligent throttling and resource management
- Basis sets organized by atomic number in `full.basis.doublezeta/`, `full.basis.triplezeta/`, and `stuttgart/`
- RCSR database provides extensive 2P and 3P periodic structure templates with symmetry data
- Phonon calculations supported with automatic band structure plotting from f25 files
- Crystal symmetry analysis integrated for automated structure classification