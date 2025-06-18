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
- **OPT/FREQ**: 7 days, 32 cores, 5G memory
- **SP**: 3 days, 32 cores, 4G memory
- **BAND/DOSS**: 1 day, 28 cores, 48G memory
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

The workflow manager creates organized directory structures:

```
working_directory/
├── workflow_configs/          # JSON configuration files
│   ├── cif_conversion_config.json
│   └── workflow_plan_*.json
├── workflow_scripts/          # Generated SLURM scripts
│   ├── submitcrystal23_opt_1.sh
│   ├── submitcrystal23_sp_2.sh
│   ├── submit_prop_band_3.sh
│   └── submit_prop_doss_4.sh
├── workflow_inputs/           # Organized input files by step
│   ├── step_001_OPT/
│   ├── step_002_SP/
│   └── step_003_BAND/
├── workflow_outputs/          # Calculation results
│   └── workflow_20250618_145837/
├── temp/                      # Temporary files
└── materials.db               # Material tracking database
```

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