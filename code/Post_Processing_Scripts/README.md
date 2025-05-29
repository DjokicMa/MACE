# Electronic Structure Properties Extractor

This script analyzes CRYSTAL output files to extract comprehensive material properties from both slab and bulk calculations, generating a comparative analysis of electronic and structural properties.

## Script Overview

### `grab_properties.py`

Processes paired `.out` simulation files (slab and bulk) to extract:
- **Electronic Properties**: Band gaps, electronic states, Fermi energies
- **Structural Properties**: Lattice parameters, atomic coordinates, densities
- **Comparative Analysis**: Binding energies, interatomic distances
- **Material Classification**: Conductor/Semiconductor/Insulator categorization

Generates:
- `HSE06_PROPERTIES.csv` containing comprehensive material property comparisons

**Requirements**:
- Python 3.x
- Libraries: `numpy`, `matplotlib`, `csv`, `os`, `glob`, `math`, `pathlib`, `warnings`
- Input Files: Paired `*_slab.out` and `*_bulk.out` files from CRYSTAL calculations
- Output File: `HSE06_PROPERTIES.csv`

---

## Requirements

- Python 3.x
- Required libraries:
  - `numpy`
  - `matplotlib` (with TkAgg backend)
  - `csv`, `os`, `glob`, `math`, `pathlib`, `warnings` (standard library)

Install dependencies:

```bash
pip install numpy matplotlib
```

---

## Configuration

**Important**: Update the `DIR` variable in the script to point to your data directory:

```python
DIR = "/path/to/your/crystal/output/files/"
```

The script expects paired files with naming convention:
- `MaterialName_slab.out`
- `MaterialName_bulk.out`

---

## Usage

Place all paired `*_slab.out` and `*_bulk.out` files in the specified directory.

Run the script:

```bash
python grab_properties.py
```

---

## Output Format

### `HSE06_PROPERTIES.csv`

Contains comprehensive material analysis with the following columns:

#### Electronic Properties
- **Eg (slab/bulk)**: Band gap values in eV for slab and bulk structures
- **Type**: Gap type classification (DIRECT/INDIRECT)
- **State (slab/bulk)**: Electronic state classification:
  - `COND`: Conducting (Eg = 0 eV)
  - `SEMI`: Semiconductor (0 < Eg < 9 eV)
  - `INSU`: Insulator (Eg > 9 eV)

#### Energetic Properties
- **E (slab/bulk)**: Total energies in atomic units
- **Eb**: Binding energy (bulk - slab) in eV

#### Structural Properties
- **IAD (slab/bulk)**: Interatomic distances in Angstroms
- **Dens (bulk)**: Bulk density (mass/volume ratio)
- **Mass**: Total atomic mass
- **Atoms**: Number of irreducible atoms
- **Vacuum**: Vacuum space calculation for layered structures

---

## Key Features

### Automated Material Classification
The script automatically categorizes materials based on band gap values:
- Conductors: Band gap = 0 eV
- Semiconductors: 0 < Band gap < 9 eV  
- Insulators: Band gap > 9 eV

### Structural Analysis
- Extracts lattice parameters (a, b, c, α, β, γ) and unit cell volume
- Calculates interatomic distances using combinatorial approach
- Determines vacuum spacing in layered structures

### Energy Analysis
- Computes binding energies between slab and bulk phases
- Converts energies from atomic units to electron volts (Hartree conversion factor: 27.2114)

### Error Handling
- Validates file existence before processing
- Handles missing or malformed data gracefully
- Continues processing remaining files if individual files fail

---

## Data Extraction Keywords

The script searches for specific patterns in CRYSTAL output files:

```python
# Electronic structure keywords
INDGAP = " INDIRECT ENERGY BAND GAP:"
DIRGAP = " DIRECT ENERGY BAND GAP:"
COND = " POSSIBLY CONDUCTING STATE"

# Structural keywords  
LAT = " LATTICE PARAMETERS  (ANGSTROMS AND DEGREES) - PRIMITIVE CELL"
CART = " CARTESIAN COORDINATES - PRIMITIVE CELL"

# Energy keywords
"ETOT(AU)" # Total energy extraction
```

---

## Mathematical Calculations

### Interatomic Distance Calculation
Uses combinatorial analysis to compute average interatomic distances:

```python
N = nCr(atoms, 2)  # Number of unique atom pairs
IAD = sum_of_distances / N  # Average interatomic distance
```

### Vacuum Space Calculation
For layered structures, calculates interlayer spacing using crystallographic coordinates and unit cell parameters.

---

## Notes

- The script is optimized for HSE06 functional calculations but works with other DFT methods
- TkAgg backend is used for matplotlib to ensure compatibility across different systems
- Processing continues even if individual material calculations fail
- Output CSV uses comma separation for easy import into analysis software

---

## Troubleshooting

**Common Issues**:
- Ensure input files follow the `*_slab.out` and `*_bulk.out` naming convention
- Verify the `DIR` path is correctly set and accessible
- Check that CRYSTAL output files contain the required keywords
- Ensure both slab and bulk files exist for each material

**File Requirements**:
- Input files must be valid CRYSTAL output files
- Files should contain completed calculations (not interrupted runs)
- Both slab and bulk calculations should use consistent parameters
