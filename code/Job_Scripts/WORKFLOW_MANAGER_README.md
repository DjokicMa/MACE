# CRYSTAL Workflow Manager

A comprehensive, user-friendly system for planning and executing complex CRYSTAL quantum chemistry calculation workflows on SLURM HPC clusters.

## Quick Start

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

## Installation Requirements

```bash
# Install required Python packages
pip install numpy matplotlib ase spglib PyPDF2 pyyaml pandas

# Verify installation
python -c "import numpy, matplotlib, ase, spglib, PyPDF2, yaml, pandas; print('All dependencies installed successfully')"
```

## Core Features

### 🎯 **Interactive Workflow Planning**
- Complete end-to-end workflow design
- Three-level CIF customization system
- Intelligent SLURM resource defaults
- JSON configuration persistence

### 🔧 **Predefined Workflow Templates**
- **`basic_opt`**: OPT only
- **`opt_sp`**: OPT → SP
- **`full_electronic`**: OPT → SP → BAND → DOSS
- **`double_opt`**: OPT → OPT2 → SP
- **`complete`**: OPT → SP → BAND → DOSS → FREQ

### ⚡ **Smart Resource Management**
- **OPT/FREQ**: 7 days, 32 cores, 5G memory per CPU
- **SP**: 3 days, 32 cores, 4G memory per CPU  
- **BAND/DOSS**: 1 day, 28 cores, 80G total memory
- Automatic SLURM script generation

### 🔗 **Seamless Tool Integration**
- NewCifToD12.py (CIF → D12 conversion)
- CRYSTALOptToD12.py (OPT → SP generation)
- Enhanced Queue Manager (job submission)
- Material Database (tracking & provenance)

## Usage Guide

### Interactive Mode (Recommended)

```bash
python run_workflow.py --interactive
```

**Step-by-Step Process:**
1. **Input Selection**: Choose CIF files, existing D12s, or mixed
2. **CIF Customization**: Select from 3 levels:
   - **Basic**: Functional + basis set
   - **Advanced**: Method, dispersion, spin settings
   - **Expert**: Full NewCifToD12.py integration
3. **Workflow Planning**: Choose or customize calculation sequence
4. **Resource Configuration**: Set SLURM parameters
5. **Execution**: Save configuration and run workflow

### Quick Start Mode

```bash
# Process CIFs with full electronic characterization
python run_workflow.py --quick-start --cif-dir ./my_cifs --workflow full_electronic

# Use existing D12s for basic optimization
python run_workflow.py --quick-start --d12-dir ./my_d12s --workflow basic_opt

# Double optimization for difficult convergence
python run_workflow.py --quick-start --cif-dir ./cifs --workflow double_opt
```

### Batch Execution

```bash
# Execute a saved workflow plan
python run_workflow.py --execute workflow_plan_20250618_145837.json

# Monitor active workflows
python run_workflow.py --status
```

## CIF Customization Levels

### Level 1: Basic
Perfect for routine calculations with standard settings.
- DFT functional selection (HSE06, B3LYP, PBE0, etc.)
- Basis set choice (POB-TZVP-REV2, 6-31G*, etc.)
- All other settings use sensible defaults

### Level 2: Advanced
Ideal for specialized calculations requiring specific settings.
- **Method**: DFT vs Hartree-Fock
- **Functional Categories**: LDA, GGA, Hybrid, meta-GGA
- **Basis Sets**: Internal vs external basis sets
- **Dispersion**: D3 correction options
- **Spin**: Polarization settings
- **Optimization**: FULLOPTG, CELLONLY, ATOMONLY

### Level 3: Expert
Full access to all NewCifToD12.py features and d12creation.py integration.
- Complete interactive configuration interface
- Custom tolerance and convergence settings
- Advanced symmetry handling options
- Specialized calculation parameters
- Expert-level fine-tuning

## Workflow Templates

### 1. Basic Optimization (`basic_opt`)
```
CIF → OPT (7 days)
```
- Single geometry optimization
- Ideal for structure validation

### 2. Optimization + Single Point (`opt_sp`)
```
CIF → OPT (7 days) → SP (3 days)
```
- Optimized geometry + accurate electronic properties
- Good for basic electronic structure analysis

### 3. Full Electronic (`full_electronic`)
```
CIF → OPT (7 days) → SP (3 days) → BAND (1 day) → DOSS (1 day)
```
- Complete electronic characterization
- Band structure and density of states
- Recommended for most electronic property studies

### 4. Double Optimization (`double_opt`)
```
CIF → OPT (7 days) → OPT2 (7 days) → SP (3 days)
```
- Two-stage optimization with tighter convergence
- Essential for difficult systems or high-accuracy requirements

### 5. Complete Analysis (`complete`)
```
CIF → OPT (7 days) → SP (3 days) → BAND (1 day) → DOSS (1 day) → FREQ (7 days)
```
- Full structural, electronic, and vibrational analysis
- Comprehensive material characterization

## Directory Structure

The workflow manager creates organized directory structures with **individual calculation folders** for each material:

```
working_directory/
├── workflow_configs/          # JSON configuration files
│   ├── cif_conversion_config.json
│   └── workflow_plan_*.json
├── workflow_scripts/          # SLURM script templates
│   ├── submitcrystal23_opt_1.sh
│   ├── submitcrystal23_sp_2.sh
│   ├── submit_prop_band_3.sh
│   └── submit_prop_doss_4.sh
├── workflow_inputs/           # Initial input files
│   └── step_001_OPT/
├── workflow_outputs/          # Individual calculation execution
│   └── workflow_20250618_160317/
│       └── step_001_OPT/
│           ├── mat_1_dia/      # Individual material folder
│           │   ├── mat_1_dia.d12     # Input file
│           │   ├── mat_1_dia.sh      # Individual SLURM script
│           │   ├── mat_1_dia.out     # CRYSTAL output
│           │   └── mat_1_dia.o12345  # SLURM job output
│           ├── mat_2_dia2/     # Another material
│           │   ├── mat_2_dia2.d12
│           │   ├── mat_2_dia2.sh
│           │   └── mat_2_dia2.out
│           └── mat_3_4_2T1_CA/ # Complex naming handled
│               ├── mat_3_4_2T1_CA.d12
│               ├── mat_3_4_2T1_CA.sh
│               └── mat_3_4_2T1_CA.out
├── temp/                      # Temporary files
└── materials.db               # Material tracking database
```

### 🔧 Individual Calculation Management

Each material gets its own **completely isolated** calculation environment:

#### **Automatic Material Name Extraction**
```bash
# Complex filenames are automatically cleaned
1_dia_opt_BULK_OPTGEOM_symm_CRYSTAL_OPT_symm_B3LYP-D3_POB-TZVP-REV2.d12
  ↓ Extracted as: mat_1_dia

3,4^2T1-CA_BULK_OPTGEOM_TZ_symm_CRYSTAL_OPT_symm_B3LYP-D3_POB-TZVP-REV2.d12  
  ↓ Extracted as: mat_3_4_2T1_CA
```

#### **Individual SLURM Script Generation**
Each material gets a customized SLURM script:
```bash
#!/bin/bash
# Generated by CRYSTAL Workflow Manager
# Workflow ID: workflow_20250618_160317
# Calculation Type: OPT
# Step: 1
# Material: mat_1_dia
# Generated: 2025-06-18 16:03:17

#SBATCH -J mat_1_dia_opt
#SBATCH -o mat_1_dia_opt-%J.o
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=32
#SBATCH --mem-per-cpu=5G
#SBATCH --time=7-00:00:00
#SBATCH -A mendoza_q

# Individual scratch directory
export scratch=$SCRATCH/workflow_20250618_160317/step_001_OPT/mat_1_dia

# Run calculation  
mpirun -n $SLURM_NTASKS /opt/software-current/2023.06/x86_64/intel/skylake_avx512/software/CRYSTAL/23-intel-2023a/bin/Pcrystal 2>&1 >& $DIR/mat_1_dia.out
```

#### **Isolated Scratch Space Management**
```bash
# Each material gets its own scratch directory
$SCRATCH/workflow_20250618_160317/step_001_OPT/mat_1_dia/
$SCRATCH/workflow_20250618_160317/step_001_OPT/mat_2_dia2/
$SCRATCH/workflow_20250618_160317/step_001_OPT/mat_3_4_2T1_CA/
```

## SLURM Script Generation Architecture

The workflow system uses a two-stage script generation process that ensures reliable, customized job scripts.

### Stage 1: Workflow Planning (`workflow_planner.py`)

**Source Scripts Used:**
- **`submitcrystal23.sh`** - Base generator for OPT/SP/FREQ calculations
- **`submit_prop.sh`** - Base generator for BAND/DOSS/properties calculations

**Process:**
1. **Read Base Scripts**: Imports the complete script logic from source generators
2. **Apply Resource Customizations**: Modifies SLURM directives (cores, memory, walltime, account)
3. **Create Workflow Templates**: Saves customized templates to `workflow_scripts/`

**Generated Templates:**
```
workflow_scripts/
├── submitcrystal23_opt_1.sh    # OPT calculations (step 1)
├── submitcrystal23_sp_2.sh     # SP calculations (step 2)  
├── submit_prop_band_3.sh       # BAND calculations (step 3)
├── submit_prop_doss_4.sh       # DOSS calculations (step 4)
└── submitcrystal23_freq_5.sh   # FREQ calculations (step 5)
```

### Stage 2: Workflow Execution (`workflow_executor.py`)

**Process:**
1. **Read Workflow Templates**: Loads the customized templates from `workflow_scripts/`
2. **Apply Material-Specific Settings**: Customizes for each individual material
   - Material names and job IDs
   - Scratch directory paths
   - Working directory paths
   - File references

**Individual Script Generation:**
Each material gets its own SLURM script:
```
workflow_outputs/workflow_20250618_160317/step_001_OPT/
├── mat_1_dia/
│   ├── mat_1_dia.d12
│   └── mat_1_dia.sh          # Individual SLURM script
├── mat_2_dia2/
│   ├── mat_2_dia2.d12
│   └── mat_2_dia2.sh         # Individual SLURM script
└── ...
```

### Callback Integration

All generated scripts include robust callback logic for workflow progression:

```bash
# Multi-location queue manager detection (prefers base directory)
if [ -f $DIR/../../../../enhanced_queue_manager.py ]; then
    echo "Using enhanced_queue_manager.py from base directory"
    cd $DIR/../../../../
    python enhanced_queue_manager.py --callback-mode completion --max-recovery-attempts 3
elif [ -f $DIR/enhanced_queue_manager.py ]; then
    echo "Using enhanced_queue_manager.py from local directory"
    cd $DIR
    python enhanced_queue_manager.py --callback-mode completion --max-recovery-attempts 3
else
    echo "Warning: No queue manager found"
fi
```

**Key Features:**
- **Base Directory Preference**: Ensures all dependencies are available
- **Fallback Logic**: Handles different execution environments
- **Debug Output**: Shows which queue manager path is used
- **Error Recovery**: Automatic retry and escalation for failed jobs
- **Workflow Progression**: Triggers next calculation steps upon completion

### Important Notes

- **Do NOT manually edit** `workflow_scripts/` templates - they are auto-generated
- **To modify defaults**: Edit the source scripts (`submitcrystal23.sh`, `submit_prop.sh`)
- **For workflow-specific changes**: Use the interactive planning interface
- **Templates are regenerated** each time you run `python run_workflow.py --interactive`

## Configuration Management

### JSON Persistence
All workflow configurations are automatically saved as JSON files:

```json
{
  "created": "2025-06-18T14:58:37",
  "input_type": "cif",
  "input_directory": "/path/to/cifs",
  "workflow_sequence": ["OPT", "SP", "BAND", "DOSS"],
  "step_configurations": {
    "OPT_1": {
      "source": "cif_conversion",
      "calculation_type": "OPT",
      "method": "DFT",
      "functional": "B3LYP-D3",
      "basis_set": "POB-TZVP-REV2"
    },
    "SP_2": {
      "source": "CRYSTALOptToD12.py",
      "calculation_type": "SP",
      "inherit_settings": true
    }
  },
  "cif_conversion_config": { ... },
  "execution_settings": {
    "max_concurrent_jobs": 200,
    "enable_material_tracking": true
  }
}
```

### Reproducibility Features
- **Exact Recreation**: Saved configurations can recreate identical workflows
- **Sharing**: Transfer configurations between users and systems
- **Version Control**: Track changes in calculation setups
- **Batch Processing**: Automated execution of saved plans

## Error Handling & Recovery

### Robust Error Management
- **CIF Conversion**: Timeout protection and file existence checks
- **Job Submission**: SLURM integration with retry logic
- **File Operations**: Atomic operations with rollback capability
- **Configuration**: Validation and fallback to defaults

### Progress Tracking
- Real-time workflow status monitoring
- Integration with material database for provenance
- Detailed logging and error reporting
- Recovery from partial failures

### Common Issues & Solutions

#### Issue: CIF conversion hangs
**Solution**: The workflow manager includes timeout protection (5 minutes) and will automatically fallback to default settings.

#### Issue: SLURM jobs fail
**Solution**: Check resource requirements and queue limits. Use `--status` to monitor job states.

#### Issue: File dependencies missing
**Solution**: The workflow manager automatically handles file dependencies. Check that previous steps completed successfully.

## Advanced Features

### Custom Workflow Design
```bash
# Interactive workflow builder
python run_workflow.py --interactive
# Choose option 6: Custom workflow
# Design your own calculation sequence
```

### Batch Processing
```bash
# Process multiple CIF directories
for dir in cif_dir_*; do
    python run_workflow.py --quick-start --cif-dir "$dir" --workflow full_electronic
done

# Execute multiple saved plans
for plan in workflow_configs/workflow_plan_*.json; do
    python run_workflow.py --execute "$plan"
done
```

### Integration with Existing Workflows
```bash
# Use with enhanced queue manager
python enhanced_queue_manager.py --max-jobs 200 --base-dir ./

# Monitor with material database
python material_monitor.py --action dashboard

# Error recovery
python error_recovery.py --action recover --max-recoveries 10
```

## Best Practices

### 1. Workflow Planning
- **Start Small**: Test with 1-2 structures before large batches
- **Use Templates**: Start with predefined templates
- **Resource Planning**: Consider walltime requirements for your systems
- **Save Configurations**: Persist successful setups for reuse

### 2. Execution Management
- **Monitor Progress**: Use `--status` to track execution
- **Check Resources**: Ensure adequate SLURM allocations
- **Organize Files**: Use the automatic directory structure
- **Database Integration**: Leverage material tracking for large studies

### 3. Performance Optimization
- **Parallel Execution**: Run multiple workflows simultaneously
- **Resource Balancing**: Adjust core/memory ratios based on system performance
- **Queue Management**: Use appropriate SLURM accounts and partitions
- **Error Recovery**: Implement automated restart strategies

## Troubleshooting

### Getting Help
```bash
# Show all available options
python run_workflow.py --help

# Display workflow templates
python run_workflow.py --show-templates

# Check current status
python run_workflow.py --status
```

### Debug Mode
For detailed debugging, check the generated configuration files:
```bash
# View workflow configuration
cat workflow_configs/workflow_plan_*.json

# Check CIF conversion settings
cat workflow_configs/cif_conversion_config.json

# Examine generated SLURM scripts
ls workflow_scripts/
```

### Common Workflows

#### High-Throughput Screening
```bash
# Process large numbers of CIFs efficiently
python run_workflow.py --quick-start \
    --cif-dir ./high_throughput_cifs \
    --workflow opt_sp \
    --max-jobs 500
```

#### Difficult Systems
```bash
# Use double optimization for challenging systems
python run_workflow.py --interactive
# Select: CIF input → Expert customization → double_opt template
# Customize convergence criteria and optimization settings
```

#### Property Analysis
```bash
# Complete electronic and vibrational characterization
python run_workflow.py --quick-start \
    --cif-dir ./materials \
    --workflow complete
```

## Integration with HPC Systems

### SLURM Configuration
The workflow manager automatically generates appropriate SLURM scripts:

```bash
#!/bin/bash
#SBATCH --job-name=crystal_opt
#SBATCH --ntasks=32
#SBATCH --mem=5G
#SBATCH --time=7-00:00:00
#SBATCH --account=mendoza_q
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# Automatically generated by CRYSTAL Workflow Manager
module load crystal/23
mpirun -np 32 crystal < input.d12 > output.out
```

### Queue Management
Integrates with the enhanced queue manager for:
- Intelligent job throttling
- Resource optimization
- Dependency management
- Progress monitoring

## Support & Development

### Documentation
- Complete documentation in `CLAUDE.md`
- Inline help: `python run_workflow.py --help`
- Examples in `workflow_configs/` directory

### Contributing
- Report issues and suggestions
- Submit workflow templates
- Enhance error recovery strategies
- Improve resource optimization

---

**The CRYSTAL Workflow Manager represents the culmination of years of CRYSTAL automation development, providing a comprehensive, user-friendly interface for complex quantum chemistry workflows while maintaining the flexibility and power of the underlying computational tools.**