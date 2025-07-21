# D3 Configuration Examples

This directory contains example JSON configuration files for different D3 calculation types in CRYSTAL.

## Usage

### Loading a configuration file:
```bash
python CRYSTALOptToD3.py --input material.out --config-file band_high_symmetry.json
```

### Batch processing with configuration:
```bash
python CRYSTALOptToD3.py --batch --config-file doss_orbital_projections.json
```

### Saving your own configuration:
```bash
python CRYSTALOptToD3.py --input material.out --calc-type BAND --save-config
```

## Available Examples

### 1. `band_high_symmetry.json`
- Electronic band structure calculation
- Automatic high-symmetry path detection
- 200 points per segment
- Uses label-based k-points

### 2. `band_auto_everything.json`
- **Fully automatic** band structure
- Auto-detects: path, bands, shrink factor
- Uses SeeK-path for comprehensive k-point coverage
- Ideal for batch processing different materials

### 3. `doss_total_only.json`
- Total density of states only
- Uses all bands
- 1000 energy points

### 4. `doss_orbital_projections.json`
- DOS with element and orbital projections (hardcoded example)
- Energy window: -10 to 20 eV around Fermi
- 2000 energy points

### 5. `doss_element_orbital_auto.json`
- **Automatic** element and orbital projections
- Projections calculated based on basis set
- Energy window: -10 to 20 eV
- Works with any material

### 6. `charge_density_3d.json`
- 3D charge density calculation
- 100×100×100 grid
- Standard ECH3 format

### 7. `transport_auto_fermi.json`
- **Automatic** chemical potential range relative to Fermi
- Chemical potential: -2 to +2 eV around Fermi energy
- Temperature range: 100-800 K
- Fermi energy extracted automatically for each material

## Configuration Structure

Each configuration file contains:
- `version`: Configuration format version
- `type`: Always "d3_configuration"
- `calculation_type`: Type of D3 calculation
- `configuration`: The actual settings for the calculation

## "Auto" Settings

Many settings can be set to "auto" for dynamic calculation based on each material:

### BAND Calculations:
- **`path: "auto"`**: Automatically determines high-symmetry k-point path based on space group
- **`bands: "auto"`**: Uses all available bands from the calculation
- **`shrink: "auto"`**: Extracts appropriate SHRINK factor from the parent calculation
- **`labels: "auto"`**: Automatically assigns k-point labels based on symmetry

### DOSS Calculations:
- **`projection_type: 2, 3, or 4`** with `project_orbital_types: true`: Automatically generates projections based on the basis set
- Projections are recalculated for each material's specific orbitals

### TRANSPORT Calculations:
- **`mu_reference: "fermi"`**: Automatically sets chemical potential range relative to the Fermi energy
- Fermi energy is extracted from each material's output file
- Chemical potential range is recalculated for each material

## Creating Custom Configurations

1. Run the interactive setup:
   ```bash
   python CRYSTALOptToD3.py --input your_file.out --calc-type DOSS
   ```

2. Configure your settings interactively

3. When prompted, save the configuration to a JSON file

4. Reuse the configuration for other materials:
   ```bash
   python CRYSTALOptToD3.py --batch --config-file your_config.json
   ```

## Best Practices

- Use "auto" settings when processing multiple materials with different symmetries
- Save material-specific settings only when you need exact control
- Test configurations on a single material before batch processing
- Keep configurations in version control for reproducibility