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

### Comprehensive Workflow Manager (NEW - Phase 3)
```bash
# Interactive workflow planning (recommended for first use)
python run_workflow.py --interactive

# Quick start with predefined templates
python run_workflow.py --quick-start --cif-dir ./cifs --workflow full_electronic

# Execute a saved workflow plan
python run_workflow.py --execute workflow_plan_20250618_145837.json

# Check workflow status
python run_workflow.py --status

# Show available templates
python run_workflow.py --show-templates
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

1. **Input Preparation** (`Crystal_d12/`)
   - Convert CIF files to CRYSTAL .d12 format
   - Add ghost atoms for surface calculations
   - Extract optimized geometries from completed calculations

2. **Job Management** (`Job_Scripts/`)
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

5. **Analysis & Visualization** (`Plotting_Scripts/`, `Band_Alignment/`, `Post_Processing_Scripts/`)
   - Extract electronic properties (band gaps, work functions)
   - Generate publication-quality plots and multi-page PDFs
   - Comprehensive material property analysis and comparison
   - Automated phonon band structure plotting from f25 files
   - Crystal symmetry analysis and labeling utilities

6. **Comprehensive Workflow Manager** (`Job_Scripts/run_workflow.py`) **(NEW - Phase 3)**
   - Complete end-to-end workflow planning and execution system
   - Interactive configuration with three customization levels
   - Integration with all existing tools (NewCifToD12.py, CRYSTALOptToD12.py, etc.)
   - Automated workflow templates and custom sequence design
   - SLURM resource management with intelligent defaults
   - JSON-based configuration persistence and reproducibility

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

## Comprehensive Workflow Manager (Phase 3)

The workflow manager (`run_workflow.py`) provides a unified interface for planning and executing complex CRYSTAL calculation workflows. It integrates all existing tools into a cohesive, user-friendly system.

### Core Components

#### **1. Interactive Workflow Planner (`workflow_planner.py`)**
- **Purpose**: Plan complete calculation sequences with full configuration
- **Features**:
  - Input type detection (CIF files, existing D12s, or mixed)
  - Three-level CIF customization system
  - Workflow template selection and modification
  - SLURM resource planning with intelligent defaults
  - JSON configuration persistence

#### **2. Workflow Executor (`workflow_executor.py`)**
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
   - Comprehensive property extraction

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
- Access to all d12creation.py features
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

### Integration with Existing Tools

#### **Seamless Integration**
- **NewCifToD12.py**: CIF → D12 conversion with full configuration
- **CRYSTALOptToD12.py**: OPT → SP/FREQ generation
- **create_band_d3.py**: Band structure input generation
- **alldos.py**: Density of states input generation
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
python run_workflow.py --interactive
# 1. Select input type (CIF/D12/Mixed)
# 2. Choose CIF customization level (Basic/Advanced/Expert)  
# 3. Select workflow template or design custom
# 4. Configure SLURM resources
# 5. Save configuration and execute
```

#### **Quick Start**
```bash
# Process CIFs with full electronic workflow
python run_workflow.py --quick-start --cif-dir ./cifs --workflow full_electronic

# Use existing D12s for optimization only
python run_workflow.py --quick-start --d12-dir ./d12s --workflow basic_opt
```

#### **Batch Execution**
```bash
# Execute saved workflow plan
python run_workflow.py --execute workflow_plan_20250618_145837.json

# Monitor progress
python run_workflow.py --status
```

### Error Handling and Recovery

#### **Robust Error Management**
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
4. **Database Integration**: Leverage material tracking for large studies

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

The workflow manager represents the culmination of all CRYSTAL automation tools, providing a comprehensive, user-friendly interface for complex calculation workflows while maintaining the flexibility and power of the underlying components.

### Material Tracking System (Phase 2 - IMPLEMENTED)

The enhanced system provides comprehensive material lifecycle tracking:

#### **Core Components**
- **`material_database.py`**: SQLite database with ASE integration for structure storage
- **`enhanced_queue_manager.py`**: Extended queue manager with material tracking and enhanced callback system
- **`error_recovery.py`**: Automated error detection and recovery with YAML configuration
- **`workflow_engine.py`**: Orchestrates OPT → SP → BAND/DOSS/TRANSPORT/CHARGE+POTENTIAL workflow progression
- **`crystal_file_manager.py`**: Organized file management by material ID and calculation type
- **`material_monitor.py`**: Real-time monitoring dashboard and health checks
- **`file_storage_manager.py`**: Comprehensive file storage with settings extraction and provenance tracking
- **`crystal_property_extractor.py`**: Complete property extraction from CRYSTAL output files
- **`formula_extractor.py`**: Chemical formula and space group extraction from input/output files

#### **Key Features**
- **Material ID Consistency**: Handles complex file naming from NewCifToD12.py and CRYSTALOptToD12.py
- **Isolated Script Execution**: Creates clean directories for alldos.py and create_band_d3.py requirements
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
- Checks `$DIR/enhanced_queue_manager.py` first (local directory)
- Falls back to `$DIR/../../../../enhanced_queue_manager.py` (parent directories)
- Also checks for `crystal_queue_manager.py` in the same locations
- Ensures workflows continue regardless of execution context

**Callback Logic:**
```bash
# Enhanced callback automatically added to all SLURM scripts
if [ -f $DIR/enhanced_queue_manager.py ]; then
    cd $DIR
    python enhanced_queue_manager.py --max-jobs 250 --reserve 30 --max-submit 5 --callback-mode completion
elif [ -f $DIR/../../../../enhanced_queue_manager.py ]; then
    cd $DIR/../../../../
    python enhanced_queue_manager.py --max-jobs 250 --reserve 30 --max-submit 5 --callback-mode completion
elif [ -f $DIR/crystal_queue_manager.py ]; then
    cd $DIR
    ./crystal_queue_manager.py --max-jobs 250 --reserve 30 --max-submit 5
elif [ -f $DIR/../../../../crystal_queue_manager.py ]; then
    cd $DIR/../../../../
    ./crystal_queue_manager.py --max-jobs 250 --reserve 30 --max-submit 5
fi
```

**Benefits:**
- **Flexible Deployment**: Works with various directory structures and workflow execution contexts
- **Automatic Queue Management**: No manual intervention required for job progression
- **Dual Compatibility**: Supports both enhanced and legacy queue managers
- **Context Awareness**: Adapts to execution environment automatically

### Complete File Storage and Provenance System (NEW - Enhanced Phase 2)

The comprehensive file storage system provides complete calculation provenance tracking, addressing the need to store D12/D3 input files with all settings and maintain complete calculation history.

#### **File Storage Manager (`file_storage_manager.py`)**

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
The system automatically extracts comprehensive settings from D12/D3 input files:

**CRYSTAL Keywords**: OPTGEOM, DFT, EXCHANGE, CORRELAT, NONLOCAL, SHRINK, TOLINTEG, etc.
**Calculation Parameters**: SHRINK factors, TOLINTEG values, TOLDEE, MAXCYCLE, FMIXING
**Basis Set Information**: Internal vs external basis sets, basis set file references
**Geometry Settings**: Optimization parameters, convergence criteria
**Exchange-Correlation**: Functional types, dispersion corrections
**SCF Parameters**: Convergence thresholds, mixing algorithms

#### **Usage Examples**
```bash
# Store files for a completed calculation
python file_storage_manager.py --store /path/to/calculation --calc-id calc_diamond_opt_001 --material-id diamond --calc-type OPT

# Retrieve all files for a calculation
python file_storage_manager.py --retrieve /path/to/destination --calc-id calc_diamond_opt_001

# List stored files
python file_storage_manager.py --list-files --calc-id calc_diamond_opt_001

# Verify file integrity
python file_storage_manager.py --verify --calc-id calc_diamond_opt_001

# Show extracted settings
python file_storage_manager.py --settings --calc-id calc_diamond_opt_001
```

#### **Query Stored Files (`query_stored_files.py`)**
Comprehensive querying system for stored files and settings:

```bash
# Query files for specific calculation
python query_stored_files.py --calc-id calc_diamond_opt_001

# Query all files for a material
python query_stored_files.py --material-id diamond

# List all stored files in database
python query_stored_files.py --list-all

# Show settings summary across all calculations
python query_stored_files.py --settings-summary
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
- **TkAgg backend** configured for matplotlib compatibility
- All file paths should be absolute, not relative
- Scripts designed for batch processing of hundreds of calculations
- Job queue manager implements intelligent throttling and resource management
- Basis sets organized by atomic number in `full.basis.doublezeta/`, `full.basis.triplezeta/`, and `stuttgart/`
- RCSR database provides extensive 2P and 3P periodic structure templates with symmetry data
- Phonon calculations supported with automatic band structure plotting from f25 files
- Crystal symmetry analysis integrated for automated structure classification