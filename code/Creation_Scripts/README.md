# D3 Generation Scripts

This folder contains Python scripts used for processing quantum chemistry output files from CRYSTAL simulations. These tools extract basis shell data, compute density of states (DOS), band structure indices, and prepare input files for transport calculations. The scripts automate parsing, indexing, and formatting required to feed data into visualization or analysis workflows.

---

## Scripts Overview

### `alldos.py`

**Purpose**: Parses quantum chemistry `.out` and `.d12` files to generate `_DOSS.d3` files for projected DOS plotting.

* Extracts and categorizes atomic orbital shells (S, P, D, F, SP) by element
* Maps orbital indices used in CRYSTAL basis sets
* Constructs the DOSS input sections needed for plotting with CRYSTAL tools

**Requirements**:

* Python 3.x
* Libraries: `os`, `sys`, `math`, `re`, `linecache`, `collections`

**I/O**:

* **Input**: Matching `.d12` and `.out` file pairs in working directory
* **Output**: `_DOSS.d3` file per job, formatted for CRYSTAL plotting

**Integration**: Fully integrated with `run_workflow.py` and `enhanced_queue_manager.py`. Generated D3 files have their settings automatically extracted and stored in the materials database, including DOS parameters, k-point grids, and projection settings.

---

### `create_band_d3.py`

**Purpose**: Prepares a `BAND.d3` file with appropriate parameters and indexing.

* Defines band energy paths and formatting for plotting band structures
* Organizes relevant shell indices if needed for band projection

**Requirements**:

* Python 3.x
* Libraries: `os`, `numpy`

**I/O**:

* **Input**: Extracted band structure or CRYSTAL output with AO index data
* **Output**: `BAND.d3` ready for plotting tools

**Integration**: Fully integrated with `run_workflow.py` and `enhanced_queue_manager.py`. Generated BAND.d3 files have their settings automatically extracted and stored in the materials database, including k-point paths, band structure parameters, and calculation settings.

---

### `create_Transportd3.py`

**Purpose**: Builds a `TRANSPORT.d3` input file based on atomic orbital shell indexing.

* Creates formatted shell groupings by element and orbital type (S, P, D, F, SP)
* Aligns output with structure of `.f9` data files from CRYSTAL transport simulations

**Requirements**:

* Python 3.x
* Libraries: `os`, `sys`, `re`, `linecache`

**I/O**:

* **Input**: `.out` files and pre-labeled shell information
* **Output**: `TRANSPORT.d3` file for transport property calculations

---

## Usage

Ensure all scripts and job `.out` and `.d12` files are in the working directory. Then run:

```bash
python alldos.py
python create_band_d3.py
python create_Transportd3.py
```

These scripts will generate `.d3` files required for DOS, band, and transport post-processing workflows.

