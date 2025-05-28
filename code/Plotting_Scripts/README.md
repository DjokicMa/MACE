# D12 Generation and Visualization Scripts

This folder contains Python scripts designed for CRYSTAL23 simulations. The tools automate the generation of `.d12` input files, insertion of ghost atoms, and provide band structure and DOS plotting utilities.

## Scripts Overview

### `OverviewPDF.py`

Generates a multi-page PDF summarizing band structure, density of states (DOS), and atomic structure for a given material.

**Features**:

* Parses `_BAND.d3`, `_DOSS.d3`, and `.cif` files
* Produces publication-quality plots
* Outputs a complete PDF with structure, band, and DOS plots

**Requirements**:

* Python 3.x
* Libraries: `numpy`, `matplotlib`, `ase`, `os`, `glob`, `PyPDF2`

**Usage**:

```bash
python OverviewPDF.py
```

---

### `plottingCIFs.py`

Visualizes atomic structures directly from `.cif` files.

**Features**:

* Plots unit cells using `ase.visualize.plot`
* Color-coded atoms and cell boundaries

**Requirements**:

* Python 3.x
* Libraries: `ase`, `matplotlib`

**Usage**:

```bash
python plottingCIFs.py --path /path/to/cif_file.cif
```

---

### `ipDOS_V2.py`

Parses `_DOSS.d3` and `.out` files to create total and projected density of states plots.

**Features**:

* Extracts spin-polarized DOS
* Annotates Fermi level
* Saves plots as PNG or PDF

**Requirements**:

* Python 3.x
* Libraries: `numpy`, `matplotlib`, `os`, `glob`

**Usage**:

```bash
python ipDOS_V2.py
```

---

### `ipBANDS_V2.py`

Parses `_BAND.d3` and `.out` files to plot electronic band structures.

**Features**:

* Plots spin-up and spin-down bands
* Highlights Fermi level
* Outputs a styled PNG/PDF

**Requirements**:

* Python 3.x
* Libraries: `numpy`, `matplotlib`, `os`, `glob`

**Usage**:

```bash
python ipBANDS_V2.py
```

---

## Requirements

* Python 3.x
* `numpy`
* `matplotlib`
* `ase`
* `glob`
* `PyPDF2`

Install with:

```bash
pip install numpy matplotlib ase PyPDF2
```

---

## Outputs

### Visualization Outputs

* `*.png` and `*.pdf` for bands and DOS
* `*.pdf` combining structure, band, and DOS views

These scripts enhance post-processing for CRYSTAL23 outputs by automating the visualization pipeline for publication and presentation use.

---

## Notes

* Ensure `_BAND.d3`, `_DOSS.d3`, and corresponding `.out` and `.cif` files exist in the same working directory.
* Customize paths and styling inside the scripts if needed.
