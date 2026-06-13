# Technology Stack

**Analysis Date:** 2026-06-13

## Languages

**Primary:**
- Python 3 (3.11+ on HPC cluster, 3.12.2 on local dev via Anaconda) - all application logic, workflow engine, parsers, database, plotting

**Secondary:**
- Bash - SLURM job submission scripts (`mace/submission/submitcrystal23.sh`, `mace/submission/submit_prop.sh`), environment setup (`activate_mace.sh`), portable generator (`mace/submission/portable_slurm_generator.py` generates bash)

## Runtime

**Environment:**
- Conda (Anaconda): `/home/marcus/anaconda3/bin/python` — local development; required for `ase`, `spglib`, `seekpath`
- HPC cluster: environment modules system (`module load CRYSTAL/23-intel-2023a`, `module load Python/3.11.3-GCCcore-12.3.0`, `module load Python-bundle-PyPI/2023.06-GCCcore-12.3.0`)

**Package Manager:**
- pip via requirements.txt
- Lockfile: Not present (no `requirements.lock` or `pip freeze` snapshot)

## Frameworks

**Core:**
- No web framework — pure CLI Python application with `argparse`-based entry points

**Workflow Engine:**
- Custom-built in `mace/workflow/engine.py` — orchestrates multi-step DFT calculation chains (optimization → single-point → band structure → DOS → phonons)

**Queue Management:**
- Custom threading-based queue manager (`mace/queue/manager.py`) — polls SLURM `squeue`, submits jobs via `sbatch`, auto-advances workflow on job completion
- Uses `threading.Thread` and `threading.Lock` throughout for concurrent job monitoring

**Testing:**
- No formal test framework (pytest/unittest not used)
- Single manual test file: `Crystal_d3/test_seekpath_interface.py` — standalone integration test for seekpath k-path generation

**Build/Dev:**
- `setup_mace.py` — custom setup script (not distutils/setuptools); writes shell rc aliases, validates dependencies
- `mace_cli` — shebang Python script (`#!/usr/bin/env python3`) serving as the unified CLI entry point

## Key Dependencies

**Critical:**
- `numpy>=1.21.0` — array math for band structure, charge density, geometry parsing throughout plotting scripts
- `ase>=3.22.0` — Atomic Simulation Environment; CIF file I/O (`ase.io.read`), neighbor lists (`ase.neighborlist`), structure manipulation, optional ASE DB adapter in `mace/database/materials.py`
- `spglib>=1.16.0` — space group symmetry analysis; used conditionally in `Crystal_d12/CrystalOutToCif.py` and `Crystal_d12/NewCifToD12.py`
- `pyyaml>=6.0` — YAML parsing; used in `mace/workflow/planner.py` and `mace/recovery/recovery.py`; recovery config at `mace/config/recovery_config.yaml`
- `pandas>=1.3.0` — tabular data; used in check scripts (`code/Check_Scripts/check_completedV2.py`, `updatelists2.py`), recovery pandas utils (`mace/recovery/pandas_utils.py`), database analysis modules

**Plotting:**
- `matplotlib>=3.5.0` — all band structure, DOS, phonon, charge density plots; used across `code/Plotting_Scripts/`, `code/NewPlotting_Scripts/`
- `fpdf` (fpdf2) — PDF generation in `code/Plotting_Scripts/OverviewPDF.py`
- `Pillow` — image manipulation in `code/Plotting_Scripts/OverviewPDF.py`

**Parsing/Export:**
- `PyPDF2>=2.0.0` — PDF reading capability; listed in requirements and validated by `setup_mace.py`

**Optional:**
- `seekpath` — HPKOT k-path generation (Hinuma et al. 2017); optional import guarded in `Crystal_d3/seekpath_interface.py` and `Crystal_d3/d3_kpoints.py`; must be installed manually (commented out in `requirements.txt`)

**Infrastructure:**
- `sqlite3` (stdlib) — embedded SQL database backing `mace/database/materials.py`; no ORM, raw SQL with thread-safe connection management
- `threading` (stdlib) — concurrent job monitoring across queue manager, workflow engine, recovery detector, file manager
- `subprocess` (stdlib) — invoking `sbatch`, `squeue`, `scancel` SLURM commands
- `json` (stdlib) — workflow state persistence, config serialization

## Configuration

**Environment:**
- `MACE_HOME` — env var pointing to repo root; used by `mace_config.py` and SLURM scripts to locate queue manager
- `SCRATCH` — HPC env var pointing to cluster scratch filesystem; used in SLURM scripts
- No `.env` file; no secrets required — purely local HPC workflow tool

**Runtime Config Files:**
- `mace_config.py` — module-level path constants (basis set dirs, input dirs); respects `MACE_HOME` override
- `mace/config/recovery_config.yaml` — error recovery strategies, retry limits, memory/walltime scaling rules
- `Crystal_d12/example_configs/*.json` — example d12 calculation config JSON files
- `Crystal_d3/example_configs/*.json` — example d3 calculation config JSON files
- Per-calculation: `materials.db` (SQLite file in working directory) tracks all calculation states

**Build:**
- No `setup.cfg`, `pyproject.toml`, or `Makefile`; setup is manual via `setup_mace.py` and `activate_mace.sh`

## Platform Requirements

**Development:**
- Linux (Anaconda Python environment at `/home/marcus/anaconda3/`)
- Anaconda must have `ase`, `spglib`, `seekpath` installed (only available there)
- `MACE_HOME` set; repo root added to `PATH` and `PYTHONPATH`

**Production (HPC Cluster):**
- SLURM-managed HPC cluster (Michigan State University HPCC)
- Cluster account: `mendoza_q` (hardcoded in default SLURM scripts)
- Modules: `CRYSTAL/23-intel-2023a`, `Python/3.11.3-GCCcore-12.3.0`, `Python-bundle-PyPI/2023.06-GCCcore-12.3.0`
- MPI runtime: Intel MPI via `I_MPI_HYDRA_BOOTSTRAP="ssh" mpirun`
- CRYSTAL23 binaries: `$EBROOTCRYSTAL/bin/Pcrystal` (parallel crystal), `$EBROOTCRYSTAL/bin/Pproperties` (parallel properties)
- Scratch storage: `$SCRATCH/crys23/<jobname>/` — each job copies input to scratch

---

*Stack analysis: 2026-06-13*
