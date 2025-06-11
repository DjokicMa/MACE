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
# Automated queue management with intelligent job control
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

### Dependency Installation
```bash
pip install numpy matplotlib ase spglib PyPDF2
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

3. **Status Monitoring** (`Check_Scripts/`)
   - Categorize job completion status based on CRYSTAL error patterns
   - Automatically sort completed/errored jobs into organized folders
   - Fix common input errors (e.g., SHRINK parameters)

4. **Input Generation** (`Creation_Scripts/`)
   - Generate DOS, band structure, and transport property input files
   - Template-based creation using crystal symmetry information
   - Basis set management for different calculation types

5. **Analysis & Visualization** (`Plotting_Scripts/`, `Band_Alignment/`, `Post_Processing_Scripts/`)
   - Extract electronic properties (band gaps, work functions)
   - Generate publication-quality plots and multi-page PDFs
   - Comprehensive material property analysis and comparison

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
- **.cif**: Crystallographic Information Files
- **CSV files**: Analysis results and job status tracking

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

### Error Handling Strategies

Scripts implement multiple layers of fault tolerance:
- CRYSTAL-specific error pattern recognition
- Automatic job recovery and resubmission
- Graceful degradation for file system issues
- Atomic file operations for data integrity
- Multiple backup locations for critical status files

## Development Notes

- Scripts use **Python 3.x** with scientific computing stack (numpy, matplotlib, ase)
- **TkAgg backend** configured for matplotlib compatibility
- All file paths should be absolute, not relative
- Scripts designed for batch processing of hundreds of calculations
- Job queue manager implements intelligent throttling and resource management
- Basis sets organized by atomic number in `full.basis.doublezeta/` and `full.basis.triplezeta/`