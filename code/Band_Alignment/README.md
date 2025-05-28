# Electronic Structure Analysis Scripts

This folder contains Python scripts for analyzing electronic structure data from simulation output files, specifically:
- **Band Gap Extraction** (`CBM_VBM.csv`)
- **Work Function Computation** (`WF.csv`)

## Scripts Overview

### `CBM_VBM.py`

Processes `.out` simulation files to extract:
- Alpha/Beta band gaps
- VBM (Valence Band Maximum)
- CBM (Conduction Band Minimum)
- Gap type (DIRECT / INDIRECT / CONDUCTING)

Generates:
- `CBM_VBM.csv` summarizing material-wise band edge properties.

**Requirements**:
- Python 3.x
- Libraries: `numpy`, `matplotlib`, `os`, `glob`, `csv`, `math`, `warnings`
- Input Files: `.out` files with keywords like `INDIRECT ENERGY BAND GAP:`, `DIRECT ENERGY BAND GAP:`, `TOP OF VALENCE BANDS`
- Output File: `CBM_VBM.csv`

[See reference guide for specifics] 

---

### `getWF.py`

Parses files of the form:
- `*_POTC.POTC.dat` and `*_POTC.out`

Extracts:
- Electrostatic potential top/bottom
- Work functions (from both ends of slab)
- Fermi energy

Generates:
- `WF.csv` summarizing potential and work function data.

**Requirements**:
- Python 3.x
- Libraries: `numpy`, `csv`, `os`, `glob`
- Input Files: `*_POTC.POTC.dat` (potential data) and `*_POTC.out` (contains `FERMI ENERGY`)
- Output File: `WF.csv`

[See reference guide for specifics] 

---

## Requirements

- Python 3.x
- `numpy`
- `matplotlib` (TkAgg backend enabled)

Install with:

```bash
pip install numpy matplotlib
```

---

## Usage

Place all `.out`, `*_POTC.POTC.dat`, and `*_POTC.out` files in the same directory.

Run the scripts:

```bash
python CBM_VBM.py
python getWF.py
```

---

## Outputs

### `CBM_VBM.csv`

Contains:
- Material name
- Eg_alpha / Eg_beta: Band gap values in eV
- Gap type (DIRECT / INDIRECT / CONDUCTING)
- VBM/CBM (alpha & beta channels)
- Total VBM/CBM: final values used for band alignment

### `WF.csv`

Contains:
- EPOT top/bot: Electrostatic potential at slab surfaces
- WF top/bot: Work function from top and bottom
- WFmax/min: Max and min work functions
- EFermi: Fermi level in eV
