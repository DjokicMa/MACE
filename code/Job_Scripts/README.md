# CRYSTAL SLURM Job Management System

This repository contains a comprehensive suite of scripts for managing CRYSTAL quantum chemistry calculations on SLURM HPC clusters. The system provides automated job submission, queue management, and job monitoring capabilities for both CRYSTAL17 and CRYSTAL23 versions.

## Scripts Overview

### SLURM Script Generation System

This system has two main approaches for generating SLURM job scripts:

#### **Manual Job Submission (Direct Use)**
**Scripts:** `submitcrystal23.sh`, `submit_prop.sh`, `submitcrystal17.sh`

These are script generators that create customized SLURM scripts for immediate submission:

- **`submitcrystal23.sh`** - Generates OPT/SP calculation scripts
- **`submit_prop.sh`** - Generates BAND/DOSS/properties calculation scripts  
- **`submitcrystal17.sh`** - Legacy CRYSTAL17 support

**Usage:**
```bash
./submitcrystal23.sh job_name    # Creates job_name.sh and submits via sbatch
./submit_prop.sh job_name        # Creates job_name.sh and submits via sbatch
```

**Features:**
- Dynamic SLURM script generation with resource customization
- Automatic callback integration for queue management
- Full path resolution for CRYSTAL executables
- Multi-location queue manager detection (prefers base directory)

#### **Workflow System (Automated)**
**Components:** `workflow_planner.py`, `workflow_executor.py`, `run_workflow.py`

The workflow system uses the same base scripts but applies additional customizations:

1. **Planning Phase** (`workflow_planner.py`):
   - Reads base scripts (`submitcrystal23.sh`, `submit_prop.sh`)
   - Applies resource customizations (cores, memory, walltime, account)
   - Creates workflow-specific templates in `workflow_scripts/`

2. **Execution Phase** (`workflow_executor.py`):
   - Reads workflow templates from `workflow_scripts/`
   - Applies material-specific customizations (names, directories, scratch paths)
   - Generates individual scripts for each material calculation

**Template Files in `workflow_scripts/`:**
- `submitcrystal23_opt_1.sh` - OPT calculations (step 1)
- `submitcrystal23_sp_2.sh` - SP calculations (step 2)  
- `submit_prop_band_3.sh` - BAND calculations (step 3)
- `submit_prop_doss_4.sh` - DOSS calculations (step 4)
- `submitcrystal23_freq_5.sh` - FREQ calculations (step 5)

**Important Notes:**
- Workflow templates are **generated from base scripts**, not manually edited
- Templates include workflow-specific directory structures and callback logic
- Each material gets an individual SLURM script (e.g., `material_name.sh`)
- All scripts use multi-location queue manager detection for reliability

#### **Callback Integration**
All generated scripts include automatic callback logic:

```bash
# Check multiple possible locations for queue managers (prefer base directory)
if [ -f $DIR/../../../../enhanced_queue_manager.py ]; then
    cd $DIR/../../../../
    python enhanced_queue_manager.py --callback-mode completion --max-recovery-attempts 3
elif [ -f $DIR/enhanced_queue_manager.py ]; then
    cd $DIR  
    python enhanced_queue_manager.py --callback-mode completion --max-recovery-attempts 3
fi
```

This ensures:
- Jobs automatically trigger next workflow steps upon completion
- Queue manager runs from directory with all dependencies
- Fallback logic handles different execution environments
- Error recovery integration for failed calculations

---

### Core Job Management

#### `enhanced_queue_manager.py` ⭐ **Phase 2 - Enhanced**

**Advanced Features:**
- **Material Tracking Database**: SQLite-based tracking of materials, calculations, and file records
- **Enhanced Callback Architecture**: Multi-location queue manager detection with flexible script path resolution
- **Early Failure Detection**: Automatic detection and cancellation of failing jobs
- **Automated Workflow Progression**: Intelligent OPT → SP → BAND/DOSS job chaining
- **Organized Calculation Folders**: Efficient calc_type/materials directory layout (quota-friendly)
- **Enhanced Error Analysis**: Detailed error classification and recovery suggestions
- **Material ID Consistency**: Handles complex naming from NewCifToD12.py and CRYSTALOptToD12.py
- **Isolated Script Execution**: Creates clean directories for alldos.py and create_band_d3.py requirements

**Key Capabilities:**
- Comprehensive material lifecycle tracking with unique material IDs
- Property extraction and storage from completed calculations
- Intelligent job prioritization and resource management
- Integration with existing SLURM scripts via callback mechanism
- Backward compatibility with legacy queue manager

**Usage:**
```bash
# Callback mode (triggered automatically by SLURM jobs)
python enhanced_queue_manager.py --callback-mode completion --max-jobs 250 --reserve 30

# Manual operations
python enhanced_queue_manager.py --status                    # Database status report
python enhanced_queue_manager.py --submit-file structure.d12 # Submit specific file
python enhanced_queue_manager.py --callback-mode submit_new  # Submit new jobs if capacity available
```

**Callback Modes:**
- `completion`: Handle job completion, check queue, submit new jobs (default for SLURM integration)
- `status_check`: Update job statuses from SLURM queue
- `submit_new`: Submit new jobs if under capacity limits
- `early_failure`: Check for and cancel early-failing jobs
- `full_check`: Complete monitoring cycle

**Requirements:**
- Python 3.x with sqlite3 support
- SLURM environment with `squeue`, `sbatch` commands  
- Material database dependencies (included: `material_database.py`)
- Enhanced submission scripts with callback integration

#### `crystal_queue_manager.py` **(Legacy)**

**Primary Features:**
- Basic SLURM job queue management with JSON persistence
- Simple job status tracking (pending, running, completed)
- Continuous monitoring approach with basic throttling
- Fallback storage for status files

**Usage:**
```bash
./crystal_queue_manager.py --d12-dir /path/to/jobs --max-jobs 250 --reserve 30
./crystal_queue_manager.py --status  # View current status only
```

**Note:** Legacy system maintained for backward compatibility. New deployments should use `enhanced_queue_manager.py`.

---

### Phase 2 Material Tracking Components ⭐ **NEW**

#### `material_database.py`

**Core database engine for material tracking system.**

**Features:**
- **SQLite + ASE Integration**: Combines structured data with atomic structure storage
- **Thread-Safe Operations**: Supports concurrent access from multiple queue managers
- **Material ID Generation**: Consistent material identifiers across complex file naming
- **Calculation Tracking**: Complete history with prerequisites and workflow relationships
- **Backup and Recovery**: Automated database maintenance and integrity checking

**Key Methods:**
```python
create_material(material_id, formula, source_type, source_file)
create_calculation(material_id, calc_type, input_file, settings)
update_calculation_status(calc_id, status, output_file=None)
get_calculations_by_status(status, calc_type=None, material_id=None)
get_next_calculation_in_workflow(material_id)
```

#### `error_recovery.py`

**Automated error detection and recovery system with YAML configuration.**

**Features:**
- **YAML Configuration**: `recovery_config.yaml` defines error recovery strategies
- **Integration with fixk.py**: Automatic SHRINK parameter fixes
- **Memory and Timeout Handling**: Increases resources for failed calculations
- **Convergence Fixes**: Adjusts SCF parameters for difficult systems
- **Recovery Tracking**: Prevents infinite retry loops with configurable limits

**Usage:**
```bash
# Attempt recovery for all failed calculations
python error_recovery.py --action recover --max-recoveries 10

# View recovery statistics
python error_recovery.py --action stats

# Create default configuration
python error_recovery.py --create-config
```

#### `workflow_engine.py`

**Orchestrates complete CRYSTAL workflows with file generation and isolation.**

**Features:**
- **Automated Workflow Progression**: OPT → SP → BAND/DOSS with real script integration
- **Isolated Directory Management**: Creates clean environments for alldos.py and create_band_d3.py
- **Script Integration**: Handles CRYSTALOptToD12.py, alldos.py, create_band_d3.py requirements
- **File Naming Consistency**: Maintains material ID consistency across complex naming
- **Workflow Configuration**: `workflows.yaml` defines calculation sequences

**Usage:**
```bash
# Process completed calculations and trigger next steps
python workflow_engine.py --action process

# Check workflow status for specific material
python workflow_engine.py --action status --material-id material_name

# View all workflow progress
python workflow_engine.py --action workflow
```

#### `crystal_file_manager.py`

**Organized file management system with material-based organization.**

**Features:**
- **Directory Organization**: Efficient `calc_type/` structure for file quotas
- **File Discovery**: Automatic detection and cataloging of calculation files
- **Integrity Checking**: Checksum validation and file verification
- **Cleanup Operations**: Archival and removal of old files
- **Integration with Check Scripts**: Works with existing file organization tools

#### `material_monitor.py`

**Real-time monitoring dashboard and system health checks.**

**Features:**
- **CLI Dashboard**: Real-time status updates with color-coded indicators
- **Health Monitoring**: Database, SLURM, and file system health checks
- **Performance Metrics**: Throughput analysis and bottleneck identification
- **Alert System**: Configurable alerts for critical issues
- **Statistics Reporting**: Comprehensive system usage and efficiency metrics

**Usage:**
```bash
# Start interactive monitoring dashboard
python material_monitor.py --action dashboard --interval 30

# Quick system statistics
python material_monitor.py --action stats

# Database health check
python material_monitor.py --action health
```

#### `input_settings_extractor.py` ⭐ **NEW**

**Comprehensive D12/D3 input settings extraction and storage.**

**Features:**
- **Complete D12 Analysis**: Extracts CRYSTAL keywords, parameters, functionals, basis sets
- **D3 Property Parameters**: Captures band structure paths, DOS projections, k-point grids
- **Database Integration**: Stores settings directly in materials.db calculations.settings_json
- **Automatic Operation**: Integrated into enhanced_queue_manager for job completion processing
- **Calculation Provenance**: Preserves complete input parameter history

**Extracted D12 Settings:**
- CRYSTAL Keywords: OPTGEOM, DFT, EXCHANGE, CORRELAT, SHRINK, etc.
- Calculation Parameters: SHRINK factors, TOLINTEG, TOLDEE, MAXCYCLE
- Functional Information: Exchange/correlation functionals, dispersion corrections
- Basis Set Details: Internal vs external, basis function counts
- Optimization Settings: Convergence criteria, geometry constraints

**Extracted D3 Settings:**
- Property Types: BAND, DOSS, NEWK calculations
- Band Structure: K-point paths and sampling density
- DOS Parameters: Projection settings and energy ranges
- Transport Properties: Calculation-specific parameters

**Usage:**
```bash
# Manual extraction (testing)
python input_settings_extractor.py --input-file structure.d12 --extract-only

# Database storage (automatic via queue manager)
python input_settings_extractor.py --input-file structure.d12 --calc-id calc_001 --db-path materials.db
```

#### `query_input_settings.py` ⭐ **NEW**

**Query and display stored input settings from materials database.**

**Features:**
- **Calculation-Specific Queries**: Show complete settings for individual calculations
- **Material Overview**: Display all settings across calculation sequence
- **Global Statistics**: Analyze parameter usage across all materials
- **Settings Comparison**: Compare parameters between different calculations

**Usage:**
```bash
# Show settings for specific calculation
python query_input_settings.py --calc-id calc_diamond_opt_001

# Show all settings for a material
python query_input_settings.py --material-id diamond

# List all calculations with stored settings
python query_input_settings.py --list-all

# Global settings summary and statistics
python query_input_settings.py --settings-summary
```

#### Configuration Files

**`recovery_config.yaml`**: Error recovery strategies
```yaml
error_recovery:
  shrink_error:
    handler: "fixk_handler"
    max_retries: 3
    resubmit_delay: 300
  memory_error:
    handler: "memory_handler"
    memory_factor: 1.5
    max_memory: "200GB"
```

**`workflows.yaml`**: Workflow definitions and resource requirements
```yaml
workflows:
  full_characterization:
    steps:
      - name: "geometry_optimization"
        calc_type: "OPT"
        next_steps: ["single_point"]
      - name: "single_point"
        calc_type: "SP"
        prerequisites: ["geometry_optimization"]
        next_steps: ["band_structure", "density_of_states"]
```

---

### Job Submission Scripts

#### `submitcrystal23.sh`

SLURM batch script generator for CRYSTAL23 calculations with integrated queue management.

**Features:**
- Creates dynamic SLURM job scripts with proper resource allocation
- Configures Intel MPI environment for parallel execution
- Implements scratch directory management for I/O optimization
- **Enhanced Integration**: Auto-triggers enhanced_queue_manager upon job completion for continuous workflow
- Supports 32-core parallel execution with 5GB memory per core
- **Enhanced Callback System**: Multi-location queue manager detection checking both local ($DIR) and parent ($DIR/../../../../) directories

**Generated Job Parameters:**
- 32 MPI tasks on single node
- 7-hour wall time limit
- Intel Skylake AVX512 architecture targeting
- Automatic file transfer to/from scratch space

#### `submitcrystal17.sh`

Legacy SLURM submission script for CRYSTAL17 calculations.

**Key Differences from CRYSTAL23:**
- Uses CRYSTAL/17-intel-2023a module
- No integrated queue management
- Similar resource allocation and scratch management

#### `submit_prop.sh` / `submit_prop_17.sh`

Specialized submission scripts for CRYSTAL properties calculations.

**Features:**
- Optimized for post-SCF property calculations
- Transfers multiple output files (band structure, DOS, transport properties)
- Reduced wall time (2 hours) and adjusted memory allocation (80GB total)
- Handles `.d3` input files and requires `.f9` wavefunction files
- **Enhanced Integration**: Multi-location callback system supporting both enhanced_queue_manager.py and crystal_queue_manager.py

---

### Batch Submission Utilities

#### `submitcrystal23.py` / `submitcrystal17.py`

Simple batch submission scripts that process all `.d12` files in the current directory.

**Functionality:**
- Automatically discovers all CRYSTAL input files
- Submits jobs using corresponding shell scripts
- Minimal error handling - designed for simple bulk submission

#### `submit_prop.py` / `submit_prop_17.py`

Batch submission utilities for properties calculations processing `.d3` files.

---

### Job Management Utilities

#### `cancel-jobs.sh`

Selective job cancellation utility for SLURM environments.

**Usage:**
```bash
./cancel-jobs.sh 12345  # Cancels all user jobs with ID > 12345
```

**Features:**
- User-specific job filtering
- Numerical job ID comparison
- Safety checks for minimum job number

#### `cd_job.sh`

Navigation utility for accessing job scratch directories.

**Features:**
- Automatic detection of job output files
- Extracts scratch directory path from job logs
- Handles multiple job output scenarios
- Interactive file selection when ambiguous

**Usage:**
```bash
source cd_job.sh [job_output_file]  # Changes to job's scratch directory
```

---

## System Requirements

### Software Dependencies
- **Python 3.x** with standard libraries (`os`, `sys`, `subprocess`, `json`, `pathlib`)
- **SLURM Workload Manager** (commands: `squeue`, `sbatch`, `scancel`)
- **CRYSTAL17/23** quantum chemistry software
- **Intel MPI** runtime environment
- **Module system** for software environment management

### Hardware Requirements
- Intel Skylake AVX512 or compatible architecture (configurable)
- Minimum 160GB RAM per node for standard jobs
- High-performance scratch storage system
- InfiniBand or equivalent high-speed interconnect

### File System Requirements
- Shared file system accessible from compute nodes
- Scratch storage with high I/O throughput
- Write permissions for status file management

---

## Installation and Setup

1. **Clone/Download Scripts:**
   ```bash
   # Place all scripts in your working directory
   chmod +x *.sh *.py
   ```

2. **Configure Environment:**
   - Ensure SLURM environment is properly configured
   - Verify CRYSTAL module availability
   - Test scratch directory access

3. **Customize Settings:**
   - Edit account information in submission scripts (`#SBATCH -A`)
   - Adjust resource requirements based on your cluster
   - Modify queue limits in `crystal_queue_manager.py`

---

## Usage Workflows

### Enhanced Workflow: Material Tracking & Automated Management ⭐ **Recommended**

1. **Setup Job Directory:**
   ```bash
   # Place all .d12 input files in directory
   # Ensure enhanced_queue_manager.py and material_database.py are present
   ```

2. **Start Initial Jobs:**
   ```bash
   # Submit a few initial jobs to start the workflow
   python enhanced_queue_manager.py --callback-mode submit_new --max-jobs 200 --reserve 20
   ```

3. **Monitor Progress:**
   ```bash
   # Check database status and job progress
   python enhanced_queue_manager.py --status
   
   # View material tracking details
   python -c "from material_database import MaterialDatabase; db = MaterialDatabase(); print(db.get_database_stats())"
   ```

4. **Automatic Operation:**
   - Jobs automatically trigger enhanced_queue_manager upon completion
   - System maintains organized calc_type/materials folder structure (efficient for file quotas)
   - Database tracks all materials, calculations, and file records
   - Early failure detection prevents resource waste

### Legacy Workflow: Basic Queue Management

1. **Setup Job Directory:**
   ```bash
   # Place all .d12 input files in directory
   # Ensure submitcrystal23.sh is present
   ```

2. **Start Queue Manager:**
   ```bash
   ./crystal_queue_manager.py --max-jobs 200 --reserve 20
   ```

3. **Monitor Progress:**
   ```bash
   ./crystal_queue_manager.py --status
   ```

### Advanced Workflow: Batch Processing

1. **Bulk Submission:**
   ```bash
   ./submitcrystal23.py  # Submits all .d12 files
   ```

2. **Properties Calculations:**
   ```bash
   # After SCF completion, run properties
   ./submit_prop.py  # Submits all .d3 files
   ```

3. **Job Management:**
   ```bash
   # Cancel recent jobs if needed
   ./cancel-jobs.sh 150000
   
   # Navigate to job directory
   source cd_job.sh job_output.o
   ```

---

## Configuration Options

### Enhanced Queue Manager Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-jobs` | 250 | Maximum concurrent jobs in queue |
| `--reserve` | 30 | Reserved slots for critical jobs |
| `--max-submit` | 5 | Limit jobs submitted per callback (prevents overwhelming cluster) |
| `--d12-dir` | `.` | Directory containing input files |
| `--db-path` | `materials.db` | SQLite database file for material tracking |
| `--callback-mode` | `completion` | Callback trigger mode |
| `--disable-tracking` | False | Disable material tracking (legacy mode) |

### Legacy Queue Manager Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-jobs` | 250 | Maximum concurrent jobs in queue |
| `--reserve` | 30 | Reserved slots for critical jobs |
| `--max-submit` | None | Limit jobs submitted per run |
| `--d12-dir` | `.` | Directory containing input files |
| `--status-file` | `crystal_job_status.json` | Job tracking file |

### Resource Allocation

**Standard CRYSTAL Jobs:**
- 32 MPI tasks, 1 node
- 5GB memory per CPU
- 7-hour wall time

**Properties Jobs:**
- 28 MPI tasks, 1 node  
- 80GB total memory
- 2-hour wall time

---

## File Formats and Conventions

### Input Files
- **`.d12`**: CRYSTAL input files for SCF calculations
- **`.d3`**: CRYSTAL input files for properties calculations
- **`.f9`**: Binary wavefunction files (fort.9)

### Output Files
- **`.out`**: Main calculation output
- **`.f9`**: Updated wavefunction file
- **Job outputs**: `jobname-JOBID.o` SLURM standard output

### Status Tracking

**Enhanced System:**
- **`materials.db`**: SQLite database with comprehensive material tracking
- **Organized folders**: `calc_type/` directory structure (quota-efficient)
- **File records**: Complete tracking of all input/output files per calculation

**Legacy System:**
- **`crystal_job_status.json`**: Basic job state tracking
- **Backup locations**: `~/`, `/tmp/` for failover

---

## Troubleshooting

### Common Issues

**Queue Manager Won't Start:**
- Check file permissions on status file
- Verify SLURM commands are available
- Ensure Python 3.x is installed

**Jobs Fail to Submit:**
- Verify `submitcrystal23.sh` is executable
- Check SLURM account settings
- Confirm module availability

**Status File Corruption:**
- Queue manager automatically recovers to backup locations
- Manually delete corrupted status file to reset

**Memory Issues:**
- Adjust `--mem-per-cpu` in submission scripts
- Consider node memory limits
- Monitor job resource usage

### Performance Optimization

**For Large Job Sets:**
- Use `--max-submit` to throttle submission rate
- Increase `--reserve` slots for important jobs
- Monitor cluster load and adjust `--max-jobs`

**For I/O Intensive Jobs:**
- Ensure scratch storage is properly configured
- Monitor scratch space usage
- Consider file transfer optimization

---

## Advanced Features

### Enhanced Material Tracking System
The enhanced queue manager provides comprehensive material lifecycle management:
- **Unique Material IDs**: Generated from file content for consistent tracking
- **Calculation Relationships**: Links between OPT → SP → BAND/DOSS workflows  
- **Property Storage**: Extracted band gaps, energies, and other calculated properties
- **File Management**: Complete tracking of all input, output, and intermediate files

### Enhanced Callback Architecture
**Key Advantages:**
- **No Runtime Limits**: Avoids 2-hour dev node restrictions by using event-driven callbacks
- **Multi-Location Detection**: Checks both local ($DIR) and parent ($DIR/../../../../) directories for queue managers
- **Dual Queue Manager Support**: Compatible with both enhanced_queue_manager.py and crystal_queue_manager.py
- **Resource Efficient**: Runs only when needed (job completions) rather than continuous monitoring
- **SLURM Integration**: Seamlessly integrates with existing SLURM job scripts
- **Scalable**: Handles hundreds of concurrent calculations without performance degradation

### Automatic Job Chaining
Enhanced system provides intelligent workflow progression:
- **Dependency Tracking**: Automatically submits follow-up calculations when prerequisites complete
- **Early Failure Detection**: Cancels failing jobs early to conserve resources
- **Load Balancing**: Respects cluster limits with configurable job submission throttling
- **Error Recovery**: Provides detailed error analysis and recovery suggestions

### Fault Tolerance
Multi-layered error handling and recovery:
- **Database Transactions**: Atomic operations ensure data consistency
- **Graceful Degradation**: Falls back to legacy mode if database unavailable
- **Error Classification**: Detailed analysis of CRYSTAL-specific error patterns
- **Automatic Recovery**: Self-healing capabilities for common failure scenarios

### Integration Capabilities
Scripts are designed for integration with:
- Workflow management systems
- Monitoring and alerting tools
- Custom analysis pipelines
- Batch processing frameworks

---

## License and Support

These scripts are provided as-is for academic and research use. Users should adapt configurations to match their specific cluster environments and requirements.

For CRYSTAL software support, consult the official CRYSTAL documentation and user community.
