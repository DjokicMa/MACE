# CRYSTAL SLURM Job Management System

This repository contains a comprehensive suite of scripts for managing CRYSTAL quantum chemistry calculations on SLURM HPC clusters. The system provides automated job submission, queue management, and job monitoring capabilities for both CRYSTAL17 and CRYSTAL23 versions.

## Scripts Overview

### Core Job Management

#### `crystal_queue_manager.py`

**Primary Features:**
- Automated SLURM job queue management with robust error handling
- Tracks job status (pending, running, completed) using JSON persistence
- Intelligent job submission respecting cluster limits and resource constraints
- Automatic recovery from file system issues with fallback storage locations
- Built-in job throttling and reservation system for critical jobs

**Key Capabilities:**
- Scans directories for `.d12` input files and manages their submission
- Maintains job state across script executions
- Implements atomic file operations for data integrity
- Provides detailed status reporting and logging
- Self-healing status file management

**Usage:**
```bash
./crystal_queue_manager.py --d12-dir /path/to/jobs --max-jobs 250 --reserve 30
./crystal_queue_manager.py --status  # View current status only
```

**Requirements:**
- Python 3.x
- SLURM environment with `squeue`, `sbatch` commands
- Write access to working directory or fallback locations
- Compatible submission scripts (`submitcrystal23.sh`)

---

### Job Submission Scripts

#### `submitcrystal23.sh`

SLURM batch script generator for CRYSTAL23 calculations with integrated queue management.

**Features:**
- Creates dynamic SLURM job scripts with proper resource allocation
- Configures Intel MPI environment for parallel execution
- Implements scratch directory management for I/O optimization
- Auto-triggers queue manager upon job completion for continuous workflow
- Supports 32-core parallel execution with 5GB memory per core

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

### Basic Workflow: Automated Queue Management

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

### Queue Manager Settings

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
- **`crystal_job_status.json`**: Persistent job state tracking
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

### Automatic Job Chaining
The CRYSTAL23 submission script includes automatic queue management triggering, enabling continuous job processing without manual intervention.

### Fault Tolerance
The queue manager implements multiple layers of error handling:
- Atomic file operations for status persistence
- Multiple backup locations for critical data
- Graceful degradation when file systems are unavailable
- Automatic recovery from temporary failures

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
