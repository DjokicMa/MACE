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

## Directory Structure

```
mace/
├── workflow/          # Workflow planning and execution
├── queue/             # Job queue management
├── database/          # Material tracking database
├── submission/        # SLURM job submission
├── recovery/          # Error detection and recovery
├── utils/             # Property extraction and analysis
└── config/            # Configuration files

Crystal_d12/           # D12 input file generation
Crystal_d3/            # D3 property file generation
code/                  # Legacy scripts (preserved for compatibility)
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