# Crystal D3 Property Calculation Scripts

This directory contains Python scripts for generating CRYSTAL D3 property calculation input files. These tools support band structure, density of states (DOS), charge density, electrostatic potential, and transport property calculations with multiple usage modes: interactive, command-line interface (CLI), and JSON configuration files.

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

# Batch processing with configuration (run in the directory holding the .out
# files; --config-file is a path, not looked up in example_configs/)
python CRYSTALOptToD3.py --batch --config-file $MACE_HOME/Crystal_d3/example_configs/doss_orbital_projections.json

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
    "output_format": 2,
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
- Literature-standard k-point paths for all space groups (Setyawan & Curtarolo 2010)
- **NEW**: SeeK-path library integration for accurate parametric k-points
- **NEW**: Cell parameter-aware extended Bravais lattice determination
- Automatic SHRINK factor extraction and scaling
- Support for both label-based and coordinate-based paths
- K-point coordinate dictionaries for all crystal systems

**SeeKPath Library Integration**:

For accurate band structure k-paths, install the `seekpath` library:
```bash
pip install seekpath
```

When seekpath is installed and you select "SeeK-path full paths" (option 4) during BAND configuration:
- K-point coordinates are calculated from actual lattice parameters (not static tables)
- Parametric k-points (monoclinic, orthorhombic, rhombohedral, etc.) are handled correctly
- SHRINK factor is automatically adjusted for exact integer representation
- Discontinuities in the path are detected and marked with "|" (`seekpath_interface.convert_to_mace_format` inserts one wherever a segment does not start at the previous segment's end)
- Inversion symmetry comes from seekpath's own symmetry analysis of the structure (`has_inversion_symmetry`). The path is requested with time-reversal symmetry (`seekpath_interface.get_accurate_bandpath`, `with_time_reversal=True`), so it has no primed k-points even for a non-centrosymmetric structure

Without seekpath, the code falls back to static dictionaries which are only accurate for cubic systems.

**SHRINK Factor Extraction Logic**:
The system uses a hierarchical approach to obtain valid SHRINK values:

1. **D12 File (Highest Priority)**: Searches for original input file
   - Standard format: `SHRINK\n16 16` → uses 16
   - K-point format: `SHRINK\n0 24\n12 12 12` → uses max(12,12,12)
   - Single value: `SHRINK\n16` → uses 16

2. **Output File (Second Priority)**: Extracts from calculation output
   - Pattern: `SHRINK. FACT.(MONKH.) 12 12 12` → uses max(12,12,12)

3. **Lattice Parameters (Third Priority)**: Calculates using "a*k > 60" rule
   - For lattice parameter a=3.567 Å: shrink = max(2, int(60/3.567)) = 16
   - Ensures adequate k-point sampling for band structures

4. **Default Fallback**: Uses 16 if all other methods fail

All SHRINK values are rounded up to even numbers for cleaner k-paths.

**Recent Enhancements**:
- Added all major SeeK-path extended Bravais lattice entries (aP2, aP3, mP1, oF1-3, tI1-2, hR1-2, cF1-2, cI1, etc.)
- Implemented cell parameter analysis functions to distinguish between variants:
  - Triclinic: aP2 vs aP3 based on angle relationships
  - Orthorhombic F: oF1/oF2/oF3 based on shortest axis
  - Tetragonal I: tI1 vs tI2 based on c/a ratio
  - Hexagonal R: hR1 vs hR2 as SeeK-path (hR1 for c/a > sqrt(3/2), rhombohedral angle < 90)
  - Cubic: cF1 vs cF2 by space group (below 207 / from 207), as SeeK-path; one cI
- Automatic lattice parameter extraction from CRYSTAL output files
- Enhanced `get_extended_bravais()` function that uses cell parameters when available

## Legacy Scripts (removed)

The legacy D3 generators (`alldos.py`, `create_band_d3.py`, `create_Transportd3.py`)
have been **removed** — their functionality is fully integrated into `CRYSTALOptToD3.py`.
They remain recoverable from git history if ever needed.

- DOS generation → `CRYSTALOptToD3.py --calc-type DOSS`
- Band structure generation → `CRYSTALOptToD3.py --calc-type BAND`
- Transport properties → `CRYSTALOptToD3.py --calc-type TRANSPORT`

## Example Configurations

The `example_configs/` directory contains ready-to-use JSON configuration files:

- `band_high_symmetry.json` - Band structure with automatic path detection
- `band_auto_everything.json` - Band structure with auto path, bands, and shrink factor
- `doss_total_only.json` - Total DOS calculation
- `doss_orbital_projections.json` - DOS with element/orbital projections
- `doss_element_orbital_auto.json` - DOS with automatic element/orbital projections from the basis set
- `charge_density_3d.json` - 3D charge density calculation
- `transport_auto_fermi.json` - Transport with chemical potential range relative to the Fermi energy

## Workflow Integration

These scripts are fully integrated with the CRYSTAL workflow management system:

- **`mace/run_workflow.py`**: Automatically generates D3 files as part of workflow sequences
- **`mace/enhanced_queue_manager.py`**: Triggers D3 generation upon successful completion of calculations
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

## Output Formats

### Band Structure Titles
The band structure titles now include information about the source of the k-path used:

```
<material_name> - Band Structure - <source> - <k-path>
```

**Source Types**:
- **SeeKPath (w.I)**: SeeK-path with inversion symmetry (centrosymmetric structures)
- **SeeKPath (no.I)**: SeeK-path without inversion symmetry (includes primed k-points only when the static fallback data is used; see Inversion Symmetry below)
- **Literature**: From Setyawan & Curtarolo (2010) standard paths
- **Manual**: Custom labels entered by user
- **Template**: Pre-defined template paths
- **Fractional**: Fractional coordinate paths
- **default**: Basic k-paths based on crystal system

### CRYSTAL K-point Labels
When using custom paths, the system displays CRYSTAL-supported k-point labels specific to your crystal system. Labels are case-sensitive and include fractional coordinates for reference.

## Known Limitations

### Discontinuous Paths in CRYSTAL
CRYSTAL's BAND input has no discontinuity marker: in both label mode (SHRINK = 0) and coordinate mode, each line is one segment (`START END`). MACE expresses a break by leaving out the segment across it:
- **Label mode**: `CRYSTALOptToD3._write_band_d3` splits the path at each `|` and writes segments only between consecutive labels inside each part, so no segment joins the labels on either side of a `|`, and the segment count on the header line matches. If any label is not valid for CRYSTAL23 (`d3_kpoints.validate_kpoint_labels_for_crystal23`), the whole path is converted to coordinates and split the same way. The automatic label paths (`d3_kpoints.get_band_path_from_symmetry`) contain no `|`; one only appears in custom labels (path method 3) or a JSON config `path`.
- **Coordinate mode** (vectors, literature, SeeK-path): each segment is written as its own start/end pair, so a break is a segment that does not start where the previous one ended. The `|` markers in the SeeK-path labels are used only for the title.
- **Plot**: CRYSTAL still places the segments one after another along the band plot, so the two points on either side of a break share one position on the k-axis.

### Inversion Symmetry
Inversion is used only by the SeeK-path format (automatic path, format 4); the label, vector and literature formats ignore it.
- With the `seekpath` library: seekpath's own symmetry analysis, with time-reversal symmetry, so no primed k-points (see SeeKPath Library Integration).
- Without it (static `seekpath_data` fallback in `d3_kpoints.get_seekpath_full_kpath`): `d3_kpoints.detect_inversion_from_crystal_output` reads the output (the `SPACE GROUP (CENTROSYMMETRIC)` line, an inversion operator, or the space-group number), falling back to `d3_kpoints.has_inversion_symmetry`. A non-centrosymmetric structure gets the `<variant>_noinv` path with primed k-points; if no `_noinv` entry exists, the centrosymmetric path is used with a warning.

### Current Technical Limitations
1. **2D Materials**: A SLAB output's BAND path is the path of its corresponding space group cut to the slab plane (segments with I3 = J3 = 0, manual p.310 note 3); the seekpath library is not used for slabs, so lattice-dependent points of centred-rectangular slabs take the static representative values. POLYMER (1D) outputs still get 3D paths.
2. **Extended Bravais Variants**: Some variants (oS2, oI2-3) need full implementation
3. **Layer Groups**: No support for 2D layer group k-paths (groups 1-80)
4. **Inversion Symmetry**: Used only by the SeeK-path format (see Inversion Symmetry above)