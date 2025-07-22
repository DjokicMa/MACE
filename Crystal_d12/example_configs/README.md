# D12 Configuration Examples

This directory contains example JSON configuration files for different D12 calculation types in CRYSTAL. These configurations can be used with NewCifToD12.py and CRYSTALOptToD12.py to quickly set up calculations with tested parameters.

## Usage

### Loading a configuration file:
```bash
# With NewCifToD12.py
python NewCifToD12.py --config-file standard_dft_opt.json structure.cif

# With CRYSTALOptToD12.py
python CRYSTALOptToD12.py --config-file high_accuracy_sp.json optimized.out

# List available configurations
python NewCifToD12.py --list-configs
```

### Creating custom configurations:
```bash
# Run interactively and save settings
python NewCifToD12.py structure.cif --save-config my_custom_config.json

# Use the saved config for other structures
python NewCifToD12.py --config-file my_custom_config.json *.cif
```

## Available Examples

### 1. `standard_dft_opt.json`
- **Purpose**: Standard geometry optimization for most materials
- **Method**: B3LYP-D3/POB-TZVP-REV2
- **Features**: 
  - Balanced accuracy and speed
  - Dispersion correction included
  - Standard convergence criteria
  - Full geometry optimization (atoms + cell)

### 2. `high_accuracy_sp.json`
- **Purpose**: High-accuracy single point calculations
- **Method**: HSE06-D3/POB-TZVP-REV2
- **Features**:
  - Hybrid functional for accurate band gaps
  - Extra-fine DFT grid
  - Tight SCF convergence
  - Ideal for final electronic structure

### 3. `3c_composite.json`
- **Purpose**: Fast screening with composite methods
- **Method**: PBEH3C/MINIX
- **Features**:
  - 3-component composite method
  - Minimal basis set (built into method)
  - Good accuracy at low cost
  - Excellent for initial screening

### 4. `freq_analysis.json`
- **Purpose**: Vibrational frequency calculations
- **Method**: B3LYP-D3/POB-TZVP-REV2
- **Features**:
  - Very tight convergence (required for frequencies)
  - IR intensities included
  - Thermodynamic properties calculated
  - No Raman by default (expensive)

### 5. `surface_slab.json`
- **Purpose**: 2D surface calculations
- **Method**: PBE-D3/POB-DZVP-REV2
- **Features**:
  - SLAB dimensionality
  - Atom-only optimization (fix cell)
  - Fermi smearing for metallic surfaces
  - Lighter basis for efficiency

### 6. `metallic_system.json`
- **Purpose**: Metallic and magnetic systems
- **Method**: PBE/POB-TZVP-REV2
- **Features**:
  - Spin polarization enabled
  - Fermi smearing (0.02 Ha)
  - Extended SCF cycles
  - Lower mixing for stability

### 7. `quick_screen.json`
- **Purpose**: Very fast initial calculations
- **Method**: HF/STO-3G
- **Features**:
  - Minimal basis set
  - Hartree-Fock (no DFT)
  - Loose convergence criteria
  - For structure validation only

### 8. `phonon_bands.json`
- **Purpose**: Phonon band structure and DOS
- **Method**: B3LYP-D3/POB-TZVP-REV2
- **Features**:
  - Phonon dispersion calculation
  - Automatic k-path detection
  - Phonon DOS with projections
  - Thermodynamic properties

## Configuration Structure

Each JSON file contains:
```json
{
  "version": "1.0",
  "type": "d12_configuration",
  "configuration": {
    "name": "config_name",
    "description": "What this config is for",
    "calculation_type": "OPT|SP|FREQ",
    "method": "HF|DFT",
    "functional": "B3LYP|PBE|HSE06|...",
    "basis_set": "POB-TZVP-REV2|STO-3G|...",
    "tolerances": {...},
    "optimization_settings": {...},
    "scf_settings": {...},
    ...
  }
}
```

## Key Settings Explained

### Tolerances
- **TOLINTEG**: Integration grid accuracy (higher = more accurate)
  - Standard: "7 7 7 7 14"
  - Tight: "8 8 8 9 30"
  - Very tight: "9 9 9 11 38"
- **TOLDEE**: SCF energy convergence (higher = tighter)
  - Loose: 6
  - Standard: 7-8
  - Tight: 9-11

### SCF Settings
- **method**: "DIIS" (default), "BROYDEN", etc.
- **maxcycle**: Maximum SCF iterations (800-1200)
- **fmixing**: Mixing percentage (15-30, lower for difficult systems)

### DFT-Specific
- **functional**: Exchange-correlation functional
- **dispersion**: D3 dispersion correction (true/false)
- **dft_grid**: Integration grid quality
  - LGRID: Light grid (fast)
  - XLGRID: Standard grid
  - XXLGRID: Extra fine (accurate)

### Optimization Settings
- **optimization_type**: 
  - FULLOPTG: Optimize atoms and cell
  - ATOMONLY: Fix cell, optimize atoms
  - CELLONLY: Fix atoms, optimize cell
- **toldeg**: Gradient convergence (0.0003)
- **toldex**: Displacement convergence (0.0012)
- **maxcycle**: Maximum optimization steps (800)

## Creating Custom Configurations

1. **Start with an example**: Choose the closest match to your needs
2. **Modify key parameters**:
   - Change functional/basis set
   - Adjust convergence criteria
   - Enable/disable features
3. **Test on one structure**: Validate settings work
4. **Save and reuse**: Apply to multiple structures

## Best Practices

1. **For new materials**: Start with `standard_dft_opt.json`
2. **For metals**: Use `metallic_system.json` as base
3. **For surfaces**: Start with `surface_slab.json`
4. **For screening**: Use `quick_screen.json` or `3c_composite.json`
5. **For final results**: Use `high_accuracy_sp.json` after optimization

## Tips

- Tighter tolerances increase accuracy but also computation time
- 3C methods (PBEH3C) provide good accuracy at low cost
- Always validate frequencies have no imaginary modes
- For difficult SCF convergence, reduce fmixing and increase maxcycle
- Test configurations on small systems before large calculations

## Workflow Integration

These configurations work seamlessly with MACE workflows:
```bash
# Use in workflow planning
mace workflow --interactive --d12-config standard_dft_opt.json

# Batch processing with config
mace convert --from-cif --config-file 3c_composite.json *.cif
```