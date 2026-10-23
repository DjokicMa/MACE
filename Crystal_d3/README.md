# Crystal D3 Property Calculation Scripts

This directory contains Python scripts for generating CRYSTAL D3 property calculation input files. These tools support band structure, density of states (DOS), charge density, electrostatic potential, transport property and orbital (Molden) calculations with multiple usage modes: interactive, command-line interface (CLI), and JSON configuration files.

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
- **ORBITALS**: the parent SCF's orbitals as Molden files, for `mace plotting --orbitals` and the GUI (see [Orbitals (ORBITALS)](../docs/cli.md#orbitals-orbitals))

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
    "energy_window": [-0.735, 0.5512],
    "energy_resolution": "standard",
    "n_points": "auto",
    "print_integrated": true,
    "output_format": 2,
    "projections": []
  }
}
```

DOSS energy grid (`d3_config.doss_npts`, the one rule for opt2d3, the GUI and workflows):
`energy_resolution` is `screening` (0.2 eV step), `standard` (0.02 eV, `DOSS_DEFAULT_STEP_EV`),
`fine` (0.005 eV), `step` (with `energy_step_ev`) or `points`; with a step the deck gets
ceil(span / step) points within 1000-99997 (CRYSTAL's DOSS.DAT limit; the span estimated from
the atoms for a band range); a number in `n_points` is written as given (2-99997). Above 5000
points over all bands the deck covers a window around E_F instead, on every route (`energy_window`,
Hartree around E_F, default -20..+15 eV, `DOSS_DEFAULT_WINDOW_EV`, for a 3D or 2D material, its bounds
moved outward into band gaps by a BAND run of the SCF, `doss_snap_window`; a 1D material's bands by index,
`doss_band_index_range`, to `conduction_bands` above the highest occupied band; `"all_bands": true`, opt2d3's
No or `--doss-window all`, keeps every band);
E_F-relative ranges (the DOSS window, TRANSPORT's MURANGE and TDFRANGE) take E_F from
`d3_interactive.trusted_fermi_au` (`mace.utils.fermi_level.scf_fermi`). TRANSPORT's NEWK mesh comes from
`d3_config.transport_mesh` (`k_resolution`: `screening`, `standard`, `fine`, `spacing` with
`k_spacing` in 1/Å, or `is` with `newk_is`).

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

The `seekpath` library is a required dependency (`requirements.txt`): it computes the HPKOT path of a 3D cell from its structure, and MACE writes it in CRYSTAL's reciprocal basis. Install it with `pip install seekpath` (on MSU HPCC, into the module Python the jobs use: INSTALLATION.md, "HPC / SLURM clusters").

SeeK-path full paths (format 4) are the default BAND path since 1.1.6: Enter at "Select format (1-4) [4]" picks them, and workflow BAND steps (`opt2d3_default_band_config()`), quick-start and `mace submit --progress` write the same deck. The high-symmetry label path (format 1, the 1.1.5 default) is answer `1`.

When you select "SeeK-path full paths" (option 4) during BAND configuration for a 3D cell:
- K-point coordinates are calculated from actual lattice parameters (not static tables)
- Parametric k-points (monoclinic, orthorhombic, rhombohedral, etc.) are handled correctly
- SHRINK factor is automatically adjusted for exact integer representation
- Discontinuities in the path are detected and marked with "|" (`seekpath_interface.convert_to_mace_format` inserts one wherever a segment does not start at the previous segment's end)
- Inversion symmetry comes from seekpath's own symmetry analysis of the structure (`has_inversion_symmetry`). The path is requested with time-reversal symmetry (`seekpath_interface.get_accurate_bandpath`, `with_time_reversal=True`), so it has no primed k-points even for a non-centrosymmetric structure

Without seekpath, a 3D SeeK-path path stops that deck (`d3_kpoints.SeekpathRequired`, printed with the install command), unless the static path is asked for: `--static-kpath` on `mace opt2d3` / `mace opt2d12`, or `MACE_STATIC_KPATH=1`. Slabs and polymers do not need it (a slab never uses the library, a polymer's path is cut to its chain axis) and get no message, and the label, k-point vector, literature and custom paths do not use it. The static path comes from dictionaries (`d3_kpoints.seekpath_data`): SeeK-path's numbers in SeeK-path's standard cell, which for many lattices is not CRYSTAL's. Their lattice-dependent points are computed for the output's cell for tI1, tI2, hR1, hR2, oF1 and C-centered monoclinic (mC1) lattices (`hpkot_static.py`, SeeK-path's HPKOT formulas, in CRYSTAL's reciprocal basis and with the library's ISS rule), when CRYSTAL's conventional cell is in SeeK-path's setting; other non-cubic lattices (and cells outside that setting, with the reason) get a warning.

Both routes take the computed geometry of the output: the final optimized geometry after an optimization, otherwise the first computed geometry, so a SUPERCEL output's path is in the supercell's reciprocal basis (f_super = E f_prim).

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

### `d3_band_path.py` - Editing a BAND k-path

The computed path (any of the four BAND path methods) can be edited before the deck is written: reorder, reverse or leave out segments, remove points, move the parts after a jump (`|`), keep the continuous part only, or add points by fractional coordinates. The edit is the `band_path` key of the configuration:

```json
"band_path": [
  {"from": "GAMMA", "to": "X"},
  {"from": "GAMMA", "to": "W"},
  {"from": "W", "to": "GAMMA"},
  {"from": "GAMMA", "to": "P", "k_to": [0.25, 0.25, 0.25]}
]
```

Segments are in order; two that do not meet (here `X` then `GAMMA`) are a discontinuity. Points of the computed path are named by label only, so the file carries over to other materials of the same lattice type as an automatic path does; points you add carry their coordinates (in the basis of the deck's segment records, CRYSTAL's primitive reciprocal cell). `"hidden": true` keeps a segment in the list without computing it.

- `mace opt2d3` at a terminal asks "Edit the band k-path?" after the BAND settings (default No; with answers piped in it is asked only with `--edit-kpath`, so 1.1.5 answer scripts keep their order) and opens a small editor (`r N` reverse, `d N` remove, `m N P` move, `p LABEL` remove a point, `a LABEL x y z` add a point, `c` continuous only, `jf`/`jl` jumps first/last, `u` undo, `reset`); `--save-config` saves the result.
- Calculation Setup in `mace gui` has the same editor under BAND and writes the same key, with a view of the first Brillouin zone (click a point there, or its button, to add it to the path).
- `mace plotting --band ... --band-path FILE` draws a band structure along an edited path (the band plot's editor in `mace gui` passes its edit the same way for the publication figure), and `mace plotting --bz` draws the zone with the path (`d3_bz.py`: the Wigner-Seitz cell of the reciprocal lattice, 2D for a slab, a segment for a polymer).
- The checks: at least one segment, at most 20 (BAND's NLINE), labels of letters, digits, `_` and `'`, no zero-length segment, coordinates for any point not on the computed path, and for a slab (polymer) a zero k_z (k_y and k_z) on every point: CRYSTAL23 manual p. 309, BAND note 3 (the opt2d3 prompt, a `band_path` and the GUI editor refuse such a point). A path that no longer visits a high-symmetry point of the computed one gets a warning.
- The deck: a label path stays labels unless a point was added by coordinates, or unless the edit names a label CRYSTAL23 does not read for that lattice (only Tables 14.1/14.2, manual p. 309 note 6, else "K-POINTS NOT CODIFIED"): then the path is written as coordinates when MACE has the label's coordinates, and refused naming the label when it has not; otherwise the segments are written in units of 1/ISS with the ISS that makes every point exact. The title names the edited path and says `(edited)`. With no `band_path` (or one equal to the computed path) the deck is byte-for-byte the one written before.

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

- Python 3.11 or newer (CI tests 3.11 and 3.12)
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
- Without it (static `seekpath_data` fallback in `d3_kpoints.get_seekpath_full_kpath`): `d3_kpoints.detect_inversion_from_crystal_output` reads the output (the `SPACE GROUP (CENTROSYMMETRIC)` line, an inversion operator, or the space-group number), falling back to `d3_kpoints.has_inversion_symmetry`. A non-centrosymmetric structure gets the `<variant>_noinv` path with primed k-points; if no `_noinv` entry exists, the centrosymmetric path is used with a warning. A slab (always the static path) drops the primed time-reversal copies when time reversal holds, that is unless its parent is a two-component SCF (`d3_kpoints.time_reversal_half`, `time_reversal_holds`).

### Current Technical Limitations
1. **2D Materials**: A SLAB output's BAND path is the path of its corresponding space group cut to the slab plane (segments with I3 = J3 = 0, manual p.310 note 3); the seekpath library is not used for slabs, so lattice-dependent points of centered-rectangular slabs take the static representative values. A POLYMER (1D) output's path runs along its chain only (Gamma to X, I2 = I3 = J2 = J3 = 0); a SeeK-path path that leaves the chain's zone (k1 beyond 1/2, as when SeeK-path's primitive cell of the 3D box is shorter than the chain's) becomes Gamma to X (`d3_kpoints.chain_path_in_zone`).
2. **Extended Bravais Variants**: Some variants (oS2, oI2-3) need full implementation
3. **Layer Groups**: No support for 2D layer group k-paths (groups 1-80)
4. **Inversion Symmetry**: Used only by the SeeK-path format (see Inversion Symmetry above)