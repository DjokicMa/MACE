# Consolidated CRYSTAL Workflow Documentation

This document consolidates all documentation for the CRYSTAL workflow system, combining information from multiple README and documentation files.

## Table of Contents

1. [System Overview](#system-overview)
2. [Workflow Manager](#workflow-manager)
3. [Material Tracking System](#material-tracking-system)
4. [Property Extraction](#property-extraction)
5. [Error Recovery](#error-recovery)
6. [Effective Mass Calculations](#effective-mass-calculations)
7. [Recent Fixes and Improvements](#recent-fixes-and-improvements)
8. [Usage Examples](#usage-examples)

---

## System Overview

The CRYSTAL workflow system provides comprehensive automation for quantum chemistry calculations using CRYSTAL17/23 software on SLURM HPC clusters. It manages the complete calculation pipeline from CIF files to final property extraction.

### Key Components

- **Workflow Manager** (`run_workflow.py`): Unified interface for planning and executing complex calculation sequences
- **Material Database**: SQLite + ASE integration for structure and calculation tracking
- **Enhanced Queue Manager**: SLURM job management with material tracking and callbacks
- **Property Extraction**: Automated extraction of electronic, structural, and thermodynamic properties
- **Error Recovery**: Automated detection and recovery from common calculation errors

### Workflow Sequences

Common workflow patterns:
- **Basic**: OPT only
- **Standard**: OPT → SP → BAND → DOSS
- **Complete**: OPT → SP → BAND → DOSS → FREQ
- **Multi-stage**: OPT → OPT2 → SP → OPT3 → SP2 → BAND → DOSS → FREQ

---

## Workflow Manager

### Overview

The workflow manager (`run_workflow.py`) provides a unified interface for planning and executing CRYSTAL calculation workflows. It integrates all existing tools into a cohesive system with three operating modes.

### Operating Modes

#### 1. Interactive Mode
```bash
python run_workflow.py --interactive
```
- Full interactive planning with customization options
- Three-level CIF customization (Basic/Advanced/Expert)
- Custom workflow sequence design
- SLURM resource configuration

#### 2. Quick Start Mode
```bash
python run_workflow.py --quick-start --cif-dir ./cifs --workflow full_electronic
```
- Rapid deployment with predefined templates
- Minimal user interaction required
- Suitable for standard workflows

#### 3. Execute Mode
```bash
python run_workflow.py --execute workflow_plan_20250618_145837.json
```
- Execute saved workflow configurations
- Reproducible calculations
- Batch processing capability

### CIF Customization Levels

1. **Basic**: DFT functional and basis set selection
2. **Advanced**: Full method configuration including spin, dispersion, convergence
3. **Expert**: Complete NewCifToD12.py integration with all features

### Workflow Templates

- `basic_opt`: Single optimization (7 days)
- `opt_sp`: Optimization + single point (7d + 3d)
- `full_electronic`: OPT → SP → BAND → DOSS (7d → 3d → 1d → 1d)
- `double_opt`: OPT → OPT2 → SP with enhanced convergence
- `complete`: Full characterization including vibrational analysis

### Directory Structure

```
working_directory/
├── workflow_configs/          # JSON configuration files
├── workflow_scripts/          # Generated SLURM scripts
├── workflow_inputs/           # Initial input files
├── workflow_outputs/          # Calculation outputs
│   └── workflow_ID/
│       └── step_001_OPT/
│           └── material_name/
│               ├── material.d12
│               ├── material.sh   # Individual SLURM script
│               └── material.out
└── workflow_temp/             # Temporary files
```

---

## Material Tracking System

### Database Schema

The material tracking system uses SQLite with ASE integration:

```sql
materials:        # Core material information
calculations:     # Individual calculation tracking
properties:       # Extracted material properties  
files:           # File associations and checksums
workflow_templates: # Reusable calculation sequences
workflow_instances: # Active workflow state tracking
```

### Key Features

- **Material ID Consistency**: Handles complex file naming conventions
- **Isolated Script Execution**: Clean environments for each calculation
- **Automated Progression**: OPT → SP → BAND/DOSS workflow automation
- **File Storage**: Complete provenance tracking with SHA256 checksums
- **Enhanced Callbacks**: Multi-location queue manager detection

### File Storage System

Automatically stores and manages calculation files:
- D12/D3 input files with complete settings extraction
- Output files with property extraction
- Binary files (fort.9, fort.25)
- SLURM scripts and logs
- Visualization outputs

---

## Property Extraction

### Comprehensive Property Extraction

The system extracts a wide range of properties:

#### Electronic Properties
- Band gaps (direct/indirect/fundamental)
- Band edges (VBM/CBM positions)
- Work functions
- Fermi energy
- Electronic classification (metal/semiconductor/insulator)
- Spin states and magnetic moments
- Effective masses (from band curvature)
- Electron/hole mobilities

#### Structural Properties
- Lattice parameters (a, b, c, α, β, γ)
- Cell volume
- Space group and symmetry
- Optimized atomic positions
- Density

#### Thermodynamic Properties
- Total energy
- Formation energy
- Optimization convergence
- Vibrational frequencies
- Elastic constants

### Advanced Electronic Analysis

The `advanced_electronic_analyzer.py` provides:
- Real effective mass from band structure derivatives (m* = ℏ²/|d²E/dk²|)
- Sophisticated metal/semimetal classification
- Transport property calculations
- DOS analysis at Fermi level

### Semimetal Classification Thresholds

Current implementation uses a two-level approach:
1. **Band Gap**: < 0.027 eV (≈ kT at room temperature)
2. **DOS at E_F**: > 0.05 × average DOS

Classification:
- **Metal**: Small gap AND high DOS(E_F)
- **Semimetal**: Small gap AND low DOS(E_F)
- **Semiconductor**: 0.027 eV < gap < 3.0 eV
- **Insulator**: gap > 3.0 eV

---

## Error Recovery

### Automated Error Detection and Recovery

The system includes comprehensive error handling:

#### Common Errors Handled
1. **SHRINK Errors**: Automatic k-point grid adjustment
2. **Memory Issues**: Resource allocation optimization
3. **Convergence Problems**: SCF parameter adjustment
4. **Symmetry Errors**: Tolerance relaxation
5. **File System Issues**: Retry with exponential backoff

#### Recovery Configuration

`recovery_config.yaml` defines recovery strategies:
```yaml
error_patterns:
  shrink_error:
    pattern: "SHRINK FACTORS"
    fix: adjust_shrink
    max_attempts: 3
```

### Error Recovery Workflow

1. Job fails with error
2. Error detector identifies issue
3. Recovery strategy applied
4. New input generated
5. Job resubmitted automatically

---

## Effective Mass Calculations

### Implementation Status

✅ **Fully Implemented** in `advanced_electronic_analyzer.py`
- Calculates from band structure curvature
- Finite difference method for d²E/dk²
- Separate electron and hole masses
- Sanity checks (0.01 to 10 m_e)

### Usage

1. **Automatic**: Included in workflow when BAND/DOSS calculations complete
2. **Manual Update**: 
   ```bash
   python update_effective_masses.py
   ```

### Accuracy Considerations

**Good Accuracy For:**
- Direct gap semiconductors
- Materials with parabolic bands
- Well-converged calculations

**Limited Accuracy For:**
- Highly anisotropic materials
- Flat band systems
- Band crossings near E_F

---

## Recent Fixes and Improvements

### Workflow Sequence Fix (Current Issue)

**Problem**: Workflow engine ignores planned sequence, uses hardcoded dependencies
- OPT completion triggers SP instead of OPT2
- Material names corrupted (test2_sp2, test2_sp3 instead of test3_sp, test4_sp)

**Solution Required**: Update `_get_next_steps_from_sequence` to follow exact sequence

### Expert Mode Fix

**Problem**: Expert mode deferred configuration to execution phase
**Solution**: Run CRYSTALOptToD12.py interactively during planning phase

### Numbered Calculations Support

**Added**: Support for OPT2, OPT3, SP2, BAND2, etc.
- Regex parsing for calculation types
- Proper dependency tracking
- Sequential numbering preservation

### Parallel Execution Logic

**Implemented**: Proper dependency-aware parallel execution
- OPT → SP + FREQ (parallel)
- SP → BAND + DOSS + next OPT (parallel)
- FREQ depends only on corresponding OPT

### Enhanced Callback System

**Multi-location Detection**: Checks multiple paths for queue manager
- Local directory first
- Parent directories fallback
- Legacy queue manager support

---

## Usage Examples

### Complete Workflow Example

```bash
# 1. Interactive planning
python run_workflow.py --interactive

# 2. Select input type (CIF/D12)
# 3. Choose customization level
# 4. Design workflow sequence
# 5. Configure SLURM resources

# Monitor progress
python workflow_status.py

# Check specific material
python enhanced_queue_manager.py --status --material-id diamond

# Update properties after completion
python update_effective_masses.py
```

### Quick CIF Processing

```bash
# Process CIFs with full electronic characterization
python run_workflow.py --quick-start \
  --cif-dir ./cifs \
  --workflow full_electronic \
  --functional B3LYP-D3 \
  --basis-set POB-TZVP-REV2
```

### Database Queries

```bash
# Check material properties
python show_properties.py --material-id diamond

# Query specific properties
sqlite3 materials.db "SELECT * FROM properties WHERE property_name LIKE '%band_gap%'"

# Export results
python database_status_report.py --export results.csv
```

### Error Recovery

```bash
# Check for errors
python error_detector.py --scan

# Manual recovery trigger
python error_recovery.py --action recover --material-id failing_material

# View recovery statistics
python error_recovery.py --action stats
```

---

## Best Practices

1. **Start Small**: Test workflows with 1-2 structures first
2. **Use Templates**: Build on predefined workflow templates
3. **Monitor Progress**: Regular status checks during execution
4. **Save Configurations**: Keep successful workflow JSONs for reuse
5. **Resource Planning**: Consider walltime for your material types
6. **Error Handling**: Enable automatic recovery for long runs
7. **Database Backups**: Regular backups of materials.db

---

## Troubleshooting

### Common Issues

1. **Module Loading**: Ensure CRYSTAL modules are available
2. **Scratch Space**: Verify $SCRATCH is accessible
3. **Database Locks**: Use timeout settings for concurrent access
4. **Memory Limits**: Adjust for large systems
5. **Convergence**: Use appropriate tolerances for material type

### Debug Commands

```bash
# Check workflow status
python workflow_status.py --debug

# Verify file integrity
python file_storage_manager.py --verify --calc-id calc_001

# Test scripts without submission
python run_workflow.py --dry-run

# Check error logs
grep ERROR *.log | less
```

---

## Future Enhancements

### Planned Features
- ML-based parameter optimization
- Automated basis set selection
- Cloud/distributed execution support
- Web interface for monitoring
- Integration with materials databases

### Performance Optimizations
- Parallel CIF conversion
- Smart job scheduling
- Checkpoint/restart capability
- Resource prediction models

---

This consolidated documentation represents the current state of the CRYSTAL workflow system as of the latest updates.