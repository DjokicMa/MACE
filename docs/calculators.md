# Calculators: force fields, charges, tight binding and ML potentials

`mace calc` (and **Calculation › Calculators…** in the GUI) runs a single
point, a geometry optimization, finite-difference frequencies, molecular
dynamics or partial charges on any structure MACE reads: a CIF, a `.d12`
deck, a CRYSTAL output (any frame), xyz/extxyz, POSCAR, fort.34 and the other
formats the GUI opens.

Slabs, polymers and molecules keep their periodicity (a group-1 SLAB,
POLYMER or MOLECULE deck, an extended XYZ with `pbc="T T F"`, an XYZ, or a
GUI document of one): nothing wraps or sees an image along a non-periodic
axis, `--relax-cell` relaxes only the periodic directions, and a molecule
keeps its box. xtb, DFTB+ and the ML potentials, which want a 3D cell, get a
copy with 20 Å of vacuum along the open axes. The optimized structure is
written as an extended XYZ (exact, pbc kept) and as a P1 CIF in a box with
15 Å of vacuum. `mace preopt` behaves the same.

These are not CRYSTAL results. An optimized structure is a force-field or ML
geometry: the CIF it writes carries the `# mace-preopt:` marker, so
`mace convert` warns before it builds anything but an OPT from it. Run a
CRYSTAL OPT before SP, FREQ, BAND or DOSS.

## Backends

| Backend | Methods | What it needs | License |
|---|---|---|---|
| `uff` | `uff4mof` (default), `uff` | nothing (built in) | parameters from the papers |
| `eqeq` | `eqeq` (charges only) | nothing (built in) | method from the paper |
| `xtb` | `gfn2` (default), `gfn1`, `gfnff` | the `xtb` program, 6.7 or newer | LGPL-3.0, not bundled |
| `dftb` | `skf` (default), `gfn2`, `gfn1` | the `dftb+` program; a Slater-Koster set for `skf` | LGPL-3.0, not bundled; each set has its own |
| `mlip` | the ML models of `mace mlip list --all` | an ML Python environment and the weights | per model (below) |

`mace calc list` shows what is installed here and, for what is not, the
command that sets it up; `mace calc setup BACKEND` prints the steps.

### UFF and UFF4MOF

MACE carries its own UFF: numpy only, energies, forces and the stress tensor
for molecules, slabs and triclinic crystals, so cells can be relaxed.

- The 127 UFF atom types of Rappé et al. (J. Am. Chem. Soc. 114, 10024, 1992)
  and the UFF4MOF types of Addicoat et al. (J. Chem. Theory Comput. 10, 880,
  2014) and Coupry et al. (J. Chem. Theory Comput. 12, 5215, 2016): Zn4O
  (`Zn3f2`, `O_3_f`), paddlewheels (`Cu4+2`, `Zn4+2`), Zr6 clusters (`Zr8f4`),
  the mu3 oxygens (`O_2_z`) and the rest of the published list.
- Typing follows RDKit's conventions (resonance types `C_R`, `N_R`, `O_R` for
  conjugated atoms): energies and forces equal RDKit's UFF to 1e-4 kcal/mol
  on a 30-molecule test set. In UFF4MOF mode, carboxylates are `C_R` with two
  C-O bonds of order 1.5 and their oxygens `O_2`; metal-oxygen bonds have
  order 0.5 and metal-metal bonds 0.25.
- Van der Waals: Lennard-Jones, 1-2 and 1-3 pairs excluded, switched smoothly
  to zero between 11 and 12.5 Å (`--cutoff`). UFF has no charges; with
  `--uff-charges eqeq` the EQeq charges of the starting geometry are added as
  fixed point charges (damped shifted-force Coulomb).
- The UFF typing is reported with every result (atom-type counts and any atom
  that had no type for its coordination).

### EQeq

The extended charge equilibration of Wilmer, Kim and Snurr (J. Phys. Chem.
Lett. 3, 2506, 2012), with the published defaults (dielectric 1.2, hydrogen
electron affinity -2.0 eV, the paper's charge centers such as Zn2+ and Zr4+,
Ewald sums as published). On IRMOF-1, MACE's charges equal the authors'
reference program to 1e-5 e. Slabs and wires use direct sums along their
periodic axes, molecules plain sums. `--charge-centers Cu=1` changes a center.
The charges CIF (`_atom_site_charge`) is what RASPA and Zeo++ read.

### xTB

GFN2-xTB, GFN1-xTB and GFN-FF through the `xtb` program, found as Clean's
GFN-FF tier finds it (`MACE_XTB`, PATH, or this machine's module system; see
[preopt.md](preopt.md)). Molecules (no periodic axis without vacuum) run as
molecules; periodic cells run GFN1/GFN2 through xtb's tblite library, which
in some xtb builds crashes for periodic GFN2 (MACE says so; GFN1 or the tblite
Python package, `--engine tblite`, are the alternatives). Charges from GFN1
and GFN2 come with every result.

### DFTB+

DFTB+ as the force engine: `skf` is SCC-DFTB with your Slater-Koster set
(`--skf-dir` or `MACE_DFTB_SKF`); a set whose README lists Hubbard
derivatives (3ob) runs as DFTB3 with them, as the set prescribes. MACE never
ships or downloads Slater-Koster files: get them from dftb.org (each set has
its own license and the citation its README asks for). `gfn2`/`gfn1` use
DFTB+'s own xTB Hamiltonian (DFTB+ 22 or newer). k-points: `--kpts 4,4,4`,
or from `--kpoint-density` (n = ceil(20 Å / |a|) by default). Open shells
(`--uhf`) are not set up for DFTB+; use xTB.

### ML potentials

The models are third-party, run in their own Python environment (`mace mlip
install` or `setup`), never in MACE's, and their weights are downloaded only by
`mace mlip fetch` after the license is shown.

| Model id | Family (package) | Code / weights license |
|---|---|---|
| `acesuit-mace-mpa-0-medium` (default) | MACE-MP (ACEsuit mace-torch) | MIT / MIT |
| `acesuit-mace-mp-0b3-medium` | MACE-MP (ACEsuit mace-torch) | MIT / MIT |
| `orb-v3-conservative-inf-mpa`, `orb-v3-conservative-inf-omat` | ORB-v3 (orb-models) | Apache-2.0 / Apache-2.0 |
| `mattersim-v1.0.0-1m`, `mattersim-v1.0.0-5m` | MatterSim (mattersim) | MIT / MIT |
| `chgnet-0.3.0` | CHGNet (chgnet) | BSD-3-Clause-LBNL |
| `orbmol-v2` | OrbMol-v2 (orb-models) | Apache-2.0 / Apache-2.0 |
| `meta-uma-s-1p1`, `meta-uma-s-1p2` | UMA (Meta fairchem) | MIT / FAIR Chemistry License v1, gated |
| `acesuit-mace-off24-medium`, `acesuit-mace-off23-medium` | MACE-OFF (ACEsuit mace-torch) | MIT / Academic Software License: academic use only, opt-in |

MACE-MP is ACEsuit's `mace-torch` package, which is unrelated to this MACE
tool (the two share a name only). Commercial-safe by default: every model
listed by default allows commercial use, and none of the others is ever a
default. MACE-OFF is academic, non-commercial use only: it is offered only
after you acknowledge that (`mace mlip fetch acesuit-mace-off24-medium
--accept-academic-license`; the date is recorded beside the weights), it shows
in `mace calc list` and the Calculators only then, and every result is labeled
"(academic license)".

One environment holds MACE-MP, CHGNet, ORB-v3, OrbMol-v2 and MatterSim
(ORB-v3 and OrbMol-v2 need orb-models 0.7, Python 3.12 or newer and torch 2.8
or newer, MatterSim numpy 2); `mace mlip install --backend ...` makes it, and
`mace mlip setup` prints the commands.

**OrbMol-v2** (Orbital Materials, Apache-2.0; trained on OMol25 and OPoly26 at
the ωB97M-V/def2-TZVPD level) is a molecular-chemistry potential: molecules,
ions, metal complexes and electrolytes. It takes the total charge and the spin
multiplicity: `--charge Q` and `--multiplicity M` (or `--uhf N`, M = N + 1), else
read from the file the structure came from (a .d12: neutral unless it has
CHARGED, CHEMOD or DOPING, and SPINLOCK n means M = n + 1; a CRYSTAL .out: the
charge from the electron count when no effective core potential is used, the
spin from SPINLOCK or the last Mulliken spins), else charge 0 and the lowest
multiplicity the electron count allows. The result says which values were
used and where they came from, and warns for an odd electron count or an open
shell. It runs molecules (0D) and periodic 3D cells (Particle Mesh Ewald
through nvalchemiops; on a CPU the first periodic call compiles kernels for
about 15 s), not slabs or polymers (orb-models does not support 1D and 2D
periodicity), and it predicts no stress, so there is no cell relaxation or
NPT. It is never chosen for you.

**UMA** (Meta's fairchem; code MIT, weights under the FAIR Chemistry License
v1 of May 14, 2025, which allows commercial use under its Acceptable Use
Policy) needs its own environment, because fairchem-core pins torch 2.13 and
e3nn 0.5 against mace-torch's e3nn 0.4.4: `mace mlip install --backend
fairchem-core` (into `venv-fairchem`). Its weights are gated on Hugging Face:
accept the terms at https://huggingface.co/facebook/UMA with your own account,
log in where MACE runs (`hf auth login` in that environment, or set
`HF_TOKEN`), then `mace mlip fetch meta-uma-s-1p1`, which shows the license and
the Acceptable Use Policy, asks, records your accept and downloads with your
access into fairchem's cache. MACE never accepts the gate for you and never
stores your token; without access it says so with the page to ask on. UMA has
task heads: `omat` (inorganic materials, the default for 3D, 2D and 1D),
`omol` (molecules, the default for 0D; it takes the charge and multiplicity
as OrbMol-v2 does), `oc20` (catalysis), `odac` (MOFs) and `omc` (molecular
crystals); `--task-head` picks another. The result names the model and the
head.

**Your own model file.** A model file you already have (your own MACE-OFF23
download, a fine-tuned MACE model, a MatterSim checkpoint), on your drive or
the cluster's, runs with `--model-file`:

```bash
mace calc opt EC.xyz -b mlip --model-file ~/models/MACE-OFF23_medium.model \
    --model-backend mace-torch --model-license-ok
```

`--model-backend` names the package in the ML environment that loads the
file: `mace-torch` (any MACE model file) or `mattersim`. `--model-license-ok`
is required: it is your statement that you hold the model's license and that
it allows this use, which MACE cannot check for a model it does not offer.
MACE prints that the model is user-supplied and that its license terms are
your responsibility, reads the file where it is (it never downloads, copies
or lists it, and never makes it a default) and records its path and sha256 in
the report. The GUI's Calculators panel offers only the models above.

**GPU.** `--device auto` (the default) gives the ML worker one GPU, GPU 0,
when there is one, and the CPU otherwise. A second GPU is never used unless
you ask: `--device cuda:1`, or `CUDA_VISIBLE_DEVICES` set before MACE starts
(SLURM sets it inside a GPU job; `auto` then takes the first GPU of the
job). GPU numbers are nvidia-smi's: the worker runs with
`CUDA_DEVICE_ORDER=PCI_BUS_ID` unless you set another order. The worker may
use half of that GPU's memory (`--gpu-mem-fraction`).

## Tasks

| Task | What it does | Files (next to INPUT, or `-o DIR`) |
|---|---|---|
| `sp` | energy, forces, stress (periodic), charges (xTB, DFTB+) | `<stem>_<backend>-<method>_sp.json` |
| `opt` | FIRE, BFGS or L-BFGS; `--relax-cell` (not for a cell with vacuum), `--keep-symmetry` | the report, `_opt.cif` (in the input's space group when it kept it, else P1; `.xyz` for a molecule), `_opt.traj` |
| `freq` | central finite differences (`--displacement` 0.01 Å); Gamma-point phonons of a crystal cell, or a molecule's vibrations; harmonic ZPE, free energy, entropy at `--temperature` | the report, `_freq_modes.extxyz` (each mode a frame with its displacement vectors) |
| `md` | NVE (Velocity Verlet), NVT (Langevin, `--friction`), NPT (Berendsen, `--pressure` bar); Maxwell-Boltzmann start at `--temperature`, `--seed` | the report, `_md.traj`, `_md.extxyz`, `_md_log.csv` (energies, temperature, volume, pressure) |
| `charges` | EQeq, or the xTB/DFTB+ charges | the report, `<stem>_<backend>-<method>_charges.cif` |

Exit codes: 0 done; 3 done but not converged, or stopped by `--time-limit`
(the partial result is written); 1 could not run or failed; 2 usage error.
When the backend itself fails (xtb's SCF does not converge, DFTB+ stops, the
ML worker fails), `mace calc` prints what failed, the backend's message and
the folder its log files were copied to, and exits 1; `--debug` adds the
Python traceback.

## In the GUI

**Calculation › Calculators…** (also in the command palette: "uff", "xtb",
"mace-mp", "md", ...) opens a panel beside the viewport for the active
structure: backend, method, task and its settings, Run and Cancel. The job
runs in the GUI's process pool with progress in the Tasks panel. The answer
shows the energy, maximum force, pressure, charges by element, frequencies
and thermochemistry or the MD summary, warnings and the citation (Copy
citation). An optimized or MD structure opens as a new document. With *Save
files next to the document* the CLI's files are written into the document's
folder (never over existing files unless *Overwrite* is ticked). Backends that
are not installed list the commands that set them up; the GUI never installs
or downloads anything. An ML model is loaded once in the server's ML worker
(the one Clean uses), on GPU 0 by default.

## Citations

Every result's report lists what to cite (`provenance.citation`): Rappé et
al. 1992 for UFF, Addicoat 2014 and Coupry 2016 for UFF4MOF, Wilmer 2012 for
EQeq, Bannwarth 2019 / Grimme 2017 / Spicher 2020 for GFN2/GFN1/GFN-FF,
Hourahine 2020 for DFTB+ (plus your Slater-Koster set's papers), and each ML
model's papers and training-data citations.
