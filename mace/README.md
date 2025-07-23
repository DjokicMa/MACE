# MACE - Mendoza Automated CRYSTAL Engine

<p align="center">
  <pre>
      ███╗   ███╗ █████╗  ██████╗███████╗
      ████╗ ████║██╔══██╗██╔════╝██╔════╝
      ██╔████╔██║███████║██║     █████╗  
      ██║╚██╔╝██║██╔══██║██║     ██╔══╝  
      ██║ ╚═╝ ██║██║  ██║╚██████╗███████╗
      ╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝╚══════╝
  </pre>
</p>

<p align="center">
  <strong>Comprehensive automation toolkit for CRYSTAL quantum chemistry workflows</strong>
</p>

---

## Overview

MACE (Mendoza Automated CRYSTAL Engine) is a powerful automation framework designed to streamline quantum chemistry calculations using the CRYSTAL software package. It provides end-to-end workflow management, from structure preparation to property analysis, with robust error handling and HPC integration.

**Developed by**: Marcus Djokic (Primary Developer)  
**Contributors**: Daniel Maldonado Lopez, Brandon Lewis, William Comaskey  
**PI**: Prof. Jose Luis Mendoza-Cortes  
**Institution**: Michigan State University, Mendoza Group

## Key Features

### 🚀 Workflow Automation
- **Interactive workflow planning** with customizable calculation sequences
- **Automated progression** from geometry optimization to property calculations
- **Material tracking database** for comprehensive calculation history
- **Error recovery** with configurable retry strategies

### 💎 CRYSTAL Integration
- **CIF to D12 conversion** with full customization options
- **Property calculations** including band structures, DOS, transport, and charge analysis
- **Phonon calculations** with automated band structure plotting
- **Basis set management** for different calculation types

### 🖥️ HPC Optimization
- **SLURM integration** with intelligent resource allocation
- **Queue management** with concurrent job limiting
- **Scratch space optimization** for large calculations
- **Multi-node parallelization** support

### 📊 Analysis Tools
- **Automated property extraction** from output files
- **Publication-quality plots** for band structures and DOS
- **Comprehensive CSV reports** for material properties
- **Electronic structure classification** (metal/semiconductor/insulator)

## Quick Start

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd reorganization

# Run interactive setup
python setup_mace.py --add-to-path

# Apply configuration
source ~/.zshrc  # or ~/.bashrc
```

See [INSTALLATION.md](../INSTALLATION.md) for detailed instructions.

### Basic Usage

#### Getting Help
```bash
# General help
mace --help

# Command-specific help (shows MACE wrapper options)
mace workflow --help
mace submit --help
mace monitor --help
mace analyze --help

# For conversion/generation commands, these pass through to the underlying scripts
mace convert --help    # Shows NewCifToD12.py help
mace opt2d12 --help   # Shows CRYSTALOptToD12.py help
mace opt2d3 --help    # Shows CRYSTALOptToD3.py help
```

#### 1. Interactive Workflow Planning
```bash
mace workflow --interactive
# Or with full path:
python ../mace_cli workflow --interactive
```
This launches an interactive session to:
- Select input files (CIFs or existing D12s)
- Choose calculation workflow (e.g., OPT → SP → BAND → DOS)
- Configure SLURM resources
- Set up material tracking

#### 2. Quick Workflow Execution
```bash
# Process CIF files with full electronic structure workflow
mace workflow --quick-start --cif-dir ./cifs --workflow full_electronic

# Run optimization only
mace workflow --quick-start --d12-dir ./d12s --workflow basic_opt
```

#### 3. Submit Individual Calculations
```bash
# Submit CRYSTAL calculations
mace submit calculation.d12
mace submit property.d3

# Or use direct scripts (if in PATH)
submitcrystal23.sh calculation.d12
submit_prop.sh property.d3
```

#### 4. Monitor Progress
```bash
# Real-time monitoring dashboard
mace monitor --dashboard

# Check queue status
mace monitor --status

# View material properties
python mace/utils/show_properties.py material_id
```

## Workflow Templates

MACE includes predefined workflow templates:

- **basic_opt**: Geometry optimization only
- **opt_sp**: OPT → Single point calculation
- **full_electronic**: OPT → SP → BAND → DOS (recommended)
- **transport_analysis**: OPT → SP → TRANSPORT
- **charge_analysis**: OPT → SP → CHARGE+POTENTIAL
- **complete**: OPT → SP → BAND → DOS → FREQ

## Comprehensive Directory Structure

### Overview
```
mace/
├── run_mace.py               # Main workflow manager interface
├── run_workflow.py           # Primary workflow execution script
├── enhanced_queue_manager.py # Enhanced SLURM queue management
├── material_monitor.py       # Real-time monitoring dashboard
│
├── config/                   # Configuration management
├── database/                 # Material tracking database
├── queue/                    # Job queue management
├── recovery/                 # Error detection and recovery
├── submission/               # Job submission scripts
├── utils/                    # Utility functions and tools
└── workflow/                 # Workflow planning and execution
```

### Detailed Component Breakdown

#### **Root Level Scripts**

- **`run_mace.py`** - Main entry point for MACE workflow system
  - Interactive workflow planning mode
  - Quick-start templates for common workflows
  - Workflow execution and monitoring
  - Example: `python run_mace.py --interactive`

- **`run_workflow.py`** - Direct workflow execution wrapper
  - Simplified interface to run_mace.py
  - Used by mace_cli for workflow command

- **`enhanced_queue_manager.py`** - Advanced queue management
  - Material tracking integration
  - Early failure detection
  - Automated workflow progression
  - Resource optimization

- **`material_monitor.py`** - Real-time monitoring dashboard
  - Live calculation status
  - Property extraction monitoring
  - Database health checks

#### **1. config/** - Configuration Management

Stores system configuration files for error recovery and workflow settings.

**Files:**
- **`recovery_config.yaml`** - Error recovery strategies
  ```yaml
  SHRINK_ERROR:
    detection: "SHRINK FACTOR"
    fix: increase_shrink
    max_attempts: 3
  MEMORY_ERROR:
    detection: "MEMORY ALLOCATION"
    fix: increase_memory
    escalation: reduce_cores
  ```

#### **2. database/** - Material Tracking System

SQLite-based tracking system with ASE integration for complete calculation provenance.

**Core Components:**
- **`materials.py`** - Main database interface
  - Thread-safe material and calculation tracking
  - Property storage and retrieval
  - ASE structure integration
  - Example usage:
    ```python
    db = MaterialDatabase()
    db.add_material("diamond", structure)
    db.update_calculation_status(calc_id, "completed")
    ```

- **`queries.py`** - Database query utilities
  - Complex query builders
  - Property aggregation functions
  - Statistical analysis helpers

- **`create_fresh_database.py`** - Database initialization
  - Creates schema with proper indices
  - Sets up trigger functions
  - Initializes workflow templates

- **`database_status_report.py`** - Generate reports
  - Material statistics
  - Calculation success rates
  - Performance metrics

#### **3. queue/** - Job Queue Management

Sophisticated SLURM integration with intelligent job scheduling.

**Key Scripts:**
- **`manager.py`** - Enhanced queue manager class
  - Job submission with throttling
  - Resource allocation optimization
  - Callback handling for job completion
  - Integration with material database

- **`monitor.py`** - Queue monitoring utilities
  - Real-time job status tracking
  - Resource utilization analysis
  - Failure pattern detection

- **`queue_lock_manager.py`** - Concurrency control
  - Prevents race conditions
  - Manages callback throttling
  - Ensures atomic operations

#### **4. recovery/** - Error Detection and Recovery

Automated error handling with configurable recovery strategies.

**Components:**
- **`detector.py`** - Error pattern detection
  - Parses CRYSTAL output files
  - Identifies common error patterns
  - Classifies error severity

- **`recovery.py`** - Recovery engine
  - Applies fixes based on error type
  - Integrates with fixk.py and updatelists2.py
  - Automatic job resubmission
  - Example recovery flow:
    ```python
    error = detector.detect_error(output_file)
    fix = recovery.get_fix_strategy(error)
    recovery.apply_fix(input_file, fix)
    recovery.resubmit_job(job_id)
    ```

#### **5. submission/** - Job Submission

SLURM script generation and job submission utilities.

**Scripts:**
- **`crystal.py`** - Submit CRYSTAL calculations
  - Handles .d12 input files
  - Resource allocation
  - Scratch directory setup

- **`properties.py`** - Submit property calculations
  - Handles .d3 property files
  - Manages wavefunction dependencies
  - Optimized for memory-intensive calculations

- **`submitcrystal23.sh`** - CRYSTAL23 submission script
  - Module loading
  - Environment setup
  - Callback integration

- **`submit_prop.sh`** - Property submission script
  - Specialized for BAND/DOSS/TRANSPORT
  - Higher memory allocation
  - Wavefunction handling

#### **6. utils/** - Utility Functions

Comprehensive toolkit for property extraction and analysis.

**Property Extraction:**
- **`property_extractor.py`** - Extract all properties
  - Band gaps (direct/indirect)
  - Total energies
  - Structural parameters
  - Electronic properties

- **`formula_extractor.py`** - Chemical information
  - Extract molecular formula
  - Determine space group
  - Count atoms and species

- **`settings_extractor.py`** - Settings extraction
  - Parse D12/D3 input files
  - Extract calculation parameters
  - Store configuration history

**Analysis Tools:**
- **`advanced_electronic_analyzer.py`** - Electronic analysis
  - Band structure analysis
  - DOS integration
  - Fermi level determination

- **`population_analysis_processor.py`** - Population analysis
  - Mulliken charges
  - Orbital populations
  - Charge density analysis

**Display and Visualization:**
- **`banner.py`** - MACE banner display
- **`animation.py`** - Progress animations
- **`show_properties.py`** - Display extracted properties

#### **7. workflow/** - Workflow Management

Complete workflow planning and execution system.

**Core Components:**
- **`planner.py`** - Interactive workflow planner
  - CIF/D12 input selection
  - Template-based workflow design
  - Resource planning
  - Configuration persistence

- **`executor.py`** - Workflow execution engine
  - Dependency management
  - Error handling
  - Progress tracking
  - File organization

- **`engine.py`** - Workflow automation
  - Automatic progression (OPT→SP→Properties)
  - State management
  - Recovery integration

- **`monitor_workflow.py`** - Workflow monitoring
  - Real-time status updates
  - Progress visualization
  - Performance metrics

**Subdirectory:**
- **`common/`** - Shared components
  - **`constants.py`** - Workflow constants
    - Calculation type definitions
    - Resource defaults
    - Template configurations

### Integration Points

1. **Material Database** ↔ **Queue Manager**
   - Automatic calculation tracking
   - Status updates on job completion

2. **Error Recovery** ↔ **Submission**
   - Automatic resubmission of fixed jobs
   - Resource adjustment based on errors

3. **Workflow Engine** ↔ **All Components**
   - Orchestrates entire calculation pipeline
   - Manages dependencies and data flow

4. **Property Extractor** ↔ **Database**
   - Stores extracted properties
   - Enables property queries

### Usage Patterns

**Simple Calculation:**
```bash
mace submit calculation.d12
```

**Complete Workflow:**
```bash
mace workflow --interactive
# Select CIFs → Choose workflow → Configure resources → Execute
```

**Monitoring:**
```bash
mace monitor --dashboard
# Real-time view of all calculations
```

**Analysis:**
```bash
mace analyze --extract-properties output_directory/
# Extract and store all properties
```

## Documentation

- [INSTALLATION.md](INSTALLATION.md) - Installation and setup guide
- [CLAUDE.md](../CLAUDE.md) - Comprehensive technical documentation
- [Examples](examples/) - Tutorial notebooks and example workflows

## Advanced Features

### Material Database
Track all calculations with full provenance:
```python
from mace.database.materials import MaterialDatabase
db = MaterialDatabase()
properties = db.get_material_properties("diamond")
```

### Custom Workflows
Create complex calculation sequences:
```python
from mace.workflow.planner import WorkflowPlanner
planner = WorkflowPlanner()
planner.add_calculation("OPT", resources={"cores": 32, "walltime": "7-00:00:00"})
planner.add_calculation("BAND", dependencies=["OPT"])
```

### Error Recovery
Automatic error detection and fixing:
```yaml
# recovery_config.yaml
SHRINK_ERROR:
  detection: "SHRINK FACTOR"
  fix: increase_shrink
  max_attempts: 3
```

## Contributing

We welcome contributions! Please see our contributing guidelines for:
- Code style conventions
- Testing requirements
- Pull request process

## Support

- **Issues**: Report bugs via GitHub Issues
- **Questions**: Contact the development team
- **Documentation**: See [CLAUDE.md](../CLAUDE.md) for comprehensive docs

## License

[License information to be added]

## Citation

If you use MACE in your research, please cite:
```
[Citation to be added]
```

---

<p align="center">
  <strong>MACE - Making CRYSTAL calculations accessible and automated</strong>
</p>