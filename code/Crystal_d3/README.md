# Crystal D3 Property Calculation Scripts

This directory contains comprehensive Python scripts for generating CRYSTAL D3 property calculation input files. These tools support band structure, density of states (DOS), charge density, electrostatic potential, and transport property calculations with multiple usage modes: interactive, command-line interface (CLI), and JSON configuration files.

## Main Scripts

### `CRYSTALOptToD3.py` - Primary D3 Generation Tool

**Purpose**: Generate CRYSTAL D3 property calculation input files from completed CRYSTAL calculations (optimization or single point).

**Supported Calculation Types**:
- **BAND**: Electronic band structure
- **DOSS**: Density of states (total, projected, orbital-resolved)
- **CHARGE**: Charge density (ECH3/ECHG)
- **POTENTIAL**: Electrostatic potential (POT3/POTC)
- **CHARGE+POTENTIAL**: Combined calculation
- **TRANSPORT**: Boltzmann transport properties

**Usage Modes**:

1. **Interactive Mode** (default):
```bash
python CRYSTALOptToD3.py --input material.out --calc-type BAND
# Or simply:
python CRYSTALOptToD3.py  # Will prompt for all options
```

2. **CLI Mode** (with arguments):
```bash
# Single file with specific calculation type
python CRYSTALOptToD3.py --input diamond.out --calc-type DOSS

# Batch processing all .out files
python CRYSTALOptToD3.py --batch --calc-type BAND

# Batch with shared settings
python CRYSTALOptToD3.py --batch --shared-settings
```

3. **JSON Configuration Mode**:
```bash
# Use saved configuration
python CRYSTALOptToD3.py --input material.out --config-file my_doss_config.json

# Save configuration during interactive setup
python CRYSTALOptToD3.py --input material.out --calc-type DOSS --save-config

# Batch processing with configuration
python CRYSTALOptToD3.py --batch --config-file doss_orbital_projections.json

# List available configurations
python CRYSTALOptToD3.py --list-configs
```

**Key Features**:
- Automatic wavefunction file (fort.9/fort.98) detection and copying
- Space group and symmetry-aware band path generation
- Basis set parsing for orbital projections
- Support for all CRYSTAL dimensionalities (0D, 1D, 2D, 3D)
- Material-specific path recalculation in batch mode

### `d3_config.py` - JSON Configuration Management

**Purpose**: Save, load, and validate D3 calculation settings in JSON format for reproducibility and batch processing.

**Key Functions**:
- `save_d3_config()`: Save configuration to JSON file
- `load_d3_config()`: Load configuration from JSON file
- `validate_d3_config()`: Validate configuration completeness
- `get_default_d3_config()`: Get default settings for each calculation type
- `print_d3_config_summary()`: Display configuration summary

**JSON Configuration Structure**:
```json
{
  "version": "1.0",
  "type": "d3_configuration",
  "calculation_type": "DOSS",
  "configuration": {
    "calculation_type": "DOSS",
    "projection_type": 3,
    "energy_range": "window",
    "energy_window": [-0.3677, 0.7354],
    "n_points": 2000,
    "print_integrated": true,
    "output_format": 0,
    "projections": []
  }
}
```

### `d3_interactive.py` - Interactive Configuration Module

**Purpose**: Provides interactive prompts for configuring all D3 calculation types with sensible defaults and validation.

**Features**:
- Guided configuration for each calculation type
- Automatic basis set parsing for orbital projections
- Energy unit conversion (eV ↔ Hartree)
- Validation of user inputs
- Integration with JSON configuration saving

### `d3_kpoints.py` - K-point Path Generation

**Purpose**: Generate high-symmetry k-point paths for band structure calculations based on space group symmetry.

**Features**:
- Literature-standard k-point paths for all space groups
- SeeK-path integration for comprehensive paths
- Automatic SHRINK factor extraction and scaling
- Support for both label-based and coordinate-based paths

## Supporting Scripts

### `alldos.py` - Legacy DOS Generation

**Purpose**: Original script for generating DOSS input files with orbital projections.

**Usage**:
```bash
python alldos.py
```

**Note**: Functionality fully integrated into `CRYSTALOptToD3.py` with enhanced features.

### `create_band_d3.py` - Legacy Band Structure Generation

**Purpose**: Original script for creating BAND.d3 files.

**Usage**:
```bash
python create_band_d3.py
```

**Note**: Functionality fully integrated into `CRYSTALOptToD3.py` with automatic symmetry detection.

### `create_Transportd3.py` - Legacy Transport Properties

**Purpose**: Generate TRANSPORT.d3 files for Boltzmann transport calculations.

**Note**: Functionality integrated into `CRYSTALOptToD3.py`.

## Example Configurations

The `example_configs/` directory contains ready-to-use JSON configuration files:

- `band_high_symmetry.json` - Band structure with automatic path detection
- `doss_total_only.json` - Total DOS calculation
- `doss_orbital_projections.json` - DOS with element/orbital projections
- `charge_density_3d.json` - 3D charge density calculation

## Workflow Integration

These scripts are fully integrated with the CRYSTAL workflow management system:

- **`run_workflow.py`**: Automatically generates D3 files as part of workflow sequences
- **`enhanced_queue_manager.py`**: Triggers D3 generation upon successful completion of calculations
- **Material Database**: All D3 settings are extracted and stored for provenance tracking

## Requirements

- Python 3.6+
- NumPy (for numerical operations)
- Standard Python libraries: `os`, `sys`, `re`, `pathlib`, `json`

## Best Practices

1. **For Single Calculations**: Use interactive mode to explore options
2. **For Multiple Materials**: Save configuration once, then use JSON mode
3. **For Workflows**: Let the workflow manager handle D3 generation automatically
4. **For Reproducibility**: Always save and version control your JSON configurations

## Tips

- Energy windows in interactive mode are entered in eV but stored in Hartree
- NEWK values for DOSS are automatically extracted from the parent calculation
- Band paths are automatically determined from space group symmetry
- Use `--list-configs` to see available configuration files
- Configuration files can be shared between users for consistent settings

## Integration with Material Database

When D3 files are generated, their settings are automatically:
- Extracted and stored in the materials database
- Linked to the parent calculation
- Available for querying and analysis
- Used for workflow progression decisions

This ensures complete calculation provenance and enables systematic analysis across materials.