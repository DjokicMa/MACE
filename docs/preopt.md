# Pre-optimization: springs, GFN-FF and ML potentials

`mace preopt` and the GUI's **Build › Clean Geometry…** move the atoms of a
structure to a better **starting geometry** for CRYSTAL. The result is never a
final structure:

- **CRYSTAL still optimizes it. The next CRYSTAL step must be an OPT.** An SP,
  FREQ, BAND or DOSS on a force-field or ML geometry is not meaningful (FREQ
  shows spurious imaginary modes).
- The method is never downgraded. Pre-optimization writes geometry only; it
  never reads or changes OPTGEOM, the functional, the basis, the tolerances,
  SPIN or the charge and spin keywords. Those stay in your options, and
  `mace convert` builds the deck from them as always.
- An FF/ML result never silently changes the space group CRYSTAL will keep
  (see [Space groups](#space-groups)).

Inputs: CIF, .d12, XYZ/extended XYZ, POSCAR and a CRYSTAL .out (its last
geometry, read as the GUI reads it). A SLAB, POLYMER or MOLECULE deck is read
in group 1, where it lists every atom; one in another layer, rod or point group
needs CRYSTAL's group tables, so MACE refuses it and names the .out of its run
to use instead. A slab, polymer or molecule keeps its pbc: nothing wraps along
a non-periodic axis and `--relax-cell` moves only the periodic directions.

```
mace preopt hand_built.cif                                  # springs (the default)
mace preopt LiFSI_box.cif --method gfnff --charge 0
mace preopt MgO.cif --method mlip:acesuit-mace-mpa-0-medium --relax-cell
mace preopt rattled.cif --method auto --symmetrize 0.1 -o preopt/
mace preopt graphene_opt.out --method mlip:acesuit-mace-mpa-0-medium --relax-cell   # a slab run's last geometry
mace convert --cif_dir preopt/ --batch --options_file my_opt_options.json   # an OPT
```

## The methods

| Method | What it is | Works for | Default fmax / steps / time |
|---|---|---|---|
| `springs` | MACE's spring model: bonds pull toward the sum of the covalent radii, other near pairs repel. Always available. | Tidying hand-placed atoms; not physics | 1e-3 eV/Å, 2000, 10 s |
| `gfnff` | GFN-FF, a generic force field, through the external `xtb` program (6.7 or newer) | Molecules, ions and ion pairs in a box, covalent solids | 0.01 eV/Å, 500, 60 s |
| `mlip:<id>` | A universal ML potential, run in its own Python environment | Periodic inorganic solids, oxides and halides | 0.02 eV/Å, 300, 120 s |
| `auto` | The best available for the structure (see [Auto](#auto)) | Anything | per method |

Positions only by default; `--relax-cell` (the GUI's **Relax the cell**)
relaxes the cell too. Every run first makes the spring result, so there is
always something to fall back to.

The springs hold the bonds `--bonds` (the GUI's **Bonds held**) names:
`shown` (the default) the bonds the GUI draws, 1.15 × the sum of the bond
radii with metal-metal pairs in a structure of metals only; `lenient` the same
rule at 1.2 ×, which also holds bonds stretched a little past what the GUI
draws (a hand-dragged atom); `none` no bond, only the repulsion that pushes
close atoms apart. A structure with no bond to hold (a lone atom) has nothing
to clean, except with `none`.

### Where GFN-FF works and fails

GFN-FF in xtb 6.7.1 was checked on synthetic cells (atoms only, fixed cell, and
with the cell relaxed where it ran):

| Works | Fails ("GFN-FF can't type this structure") |
|---|---|
| NaCl (cell -6 %), Si (cell -2 %), diamond, GaAs, rutile (c -8 %), CsPbI3 (cell -10 %), Al2O3, quartz, graphite, hBN, Cu, Al; molecules, Li+ and FSI- in a box | MgO (at any lattice constant), CaO, LiF, KBr, ZnO, SrTiO3, bcc Fe |

Cell relaxation of ionic solids often crashes even where atoms-only runs work,
and the cells GFN-FF gives are off by several percent. When xtb fails, the run
keeps the spring result and says why. MACE rebuilds GFN-FF's topology for every
force call: xtb 6.7.1 reuses a topology file that is wrong once an atom crosses
a cell face, and relaxations then "converge" uphill.

### ML potentials

| Model (`--method`) | Code | Weights | Size | Role |
|---|---|---|---|---|
| `mlip:acesuit-mace-mpa-0-medium`: MACE-MPA-0 (ACEsuit mace-torch) | mace-torch 0.3.16, MIT | MIT | 79 MB, fetched on request | the default |
| `mlip:chgnet-0.3.0`: CHGNet 0.3.0 | chgnet 0.4.2, BSD-3-Clause (LBNL) | BSD-3-Clause (LBNL), inside the chgnet package | 0 | the offline fallback |

**ACEsuit's `mace-torch` package is unrelated to this MACE.** Its Python
package is also called `mace`, which is why the ML potential always runs out of
process, in its own interpreter (`python -I`, no PYTHONPATH, a private working
directory), and why the worker checks that `mace` belongs to the installed
mace-torch before it loads anything.

Commercial-safe by default: every model listed, defaulted to, chosen by Auto
or in a preset allows commercial use (MIT, BSD, Apache, and the FAIR Chemistry
License of Meta's UMA, which is `mace calc` only); the allow-list is part of
MACE's code, not of a data file. Academic-only models are an explicit opt-in:
MACE-OFF24 and MACE-OFF23 medium (organic molecules, Academic Software
License) are never a default or chosen by Auto, are left out of `mace mlip
list` unless you pass `--all` or `--academic`, and are fetched from ACEsuit's
repository and run only after you acknowledge academic, non-commercial use
(`mace mlip fetch ID --accept-academic-license`, and at a terminal you type
ACADEMIC; in the GUI a checkbox under the license). The acknowledgment is
recorded with its date beside the weights (`ACADEMIC.json`); every result made
with one is labeled "(academic license)", in its summary, report, CIF header
and marker (`license=academic`) and the GUI. Then `--method
mlip:acesuit-mace-off24-medium` runs it here. Other Academic Software License
models (MACE-OMAT, MACE-MATPES, MACE-OMOL, MACE-MH, GRACE) are not offered.

### Auto

Auto looks at the structure with ASE's dimensionality analysis (RDA, its own
bonding scan, which also works for metals) and for C-H bonds. The scan stops
at bonds of 3 x the sum of the covalent radii: past that ASE's score has
almost nothing left to give, and going on across the vacuum of a slab took
minutes. A sparse cell such as an ion pair in a large box, which never joins
up within that range, counts as 0D:

- a charged system, molecules or ions in a box (0D components), or organic
  content: GFN-FF, then the default ML model (only when the charge is 0);
- otherwise (a neutral periodic inorganic network, metals included): the
  default ML model, then CHGNet, then GFN-FF.

Methods that are not installed are skipped. The GUI keeps one ML model loaded
at a time, so its Auto tries only the default model; when that one is
rejected, it goes on to GFN-FF without CHGNet. Pick CHGNet 0.3.0 in the list,
or run `mace preopt --method auto`, for the whole chain. The springs are always the last
resort, and the result says what ran, for example "Auto: ML potential:
MACE-MPA-0 (ACEsuit mace-torch), converged in 13 steps" or "Auto: GFN-FF
refused (Z > 86 (U): GFN-FF gives wrong energies there); kept the spring
result".

## What a run checks

Before a GFN-FF or ML run (refused, the spring result is kept):

- more than 2000 atoms; GFN-FF: more than 1000 atoms, or any element beyond Rn
  (U and Pu give E = 0 silently);
- vacuum: every cell is periodic in 3D, so a slab or a molecule is a cell with
  vacuum, found geometrically. GFN-FF needs at least 10 Å between periodic
  images (atom center to center); a cell relaxation of a cell with vacuum is
  refused, and the spring result written in its place keeps the cell fixed
  too (the springs alone, asked for with `--relax-cell`, still relax it and
  warn that the vacuum shrinks);
- ML: more than 300 atoms on the CPU (the CLI allows up to 1500 with an
  explicit `--time-limit`, or with `--device cuda:N`), elements the model has
  no parameters for, less than 2 GB of free memory; under Auto, any charge.

After the run, measured against the starting structure (rejected, the spring
result is kept):

- the energy rose;
- the volume changed by more than 10 %;
- a new close contact (a pair closer than 0.75 × the sum of the covalent radii);
- the space group changed (below);
- not converged (the step or time limit was hit: Run Longer in the GUI, or a
  larger `--steps`/`--time-limit`);
- non-finite numbers.

`mace preopt` exits 3 when an explicit `gfnff` or `mlip:` run was refused,
failed or rejected and the spring result was written instead.

### Space groups

CRYSTAL OPT keeps the space group it is given. A pre-optimization that raised
the group (pulled a tilted or Jahn-Teller-distorted structure into a higher
symmetry) would therefore remove the distortion from the final CRYSTAL result
for good. So:

- with **keep the space group** (the default), any change is rejected;
- without it, a gain is still rejected, and a loss is reported;
- **symmetrize first** (`--symmetrize TOL`, the GUI checkbox; off by default)
  is the only allowed change. It runs before anything else, the raised group
  becomes the reference, and the CLI, the readout and the provenance say so:
  "raised from P2_1/c to Fm-3m by symmetrize-first (0.1 Å); CRYSTAL OPT will
  keep Fm-3m and cannot restore a lower-symmetry distortion."

A rattled P1 cell that relaxes back toward a symmetric structure is a gain, too:
symmetrize it first if the higher symmetry is what you want.

The output CIF is written in the input's own space group and setting when the
run kept it (MACE reads it back and requires an atom-for-atom match); otherwise
it is P1, with a warning, because cif2d12 then builds a P1 deck. A P1 input
symmetrized first is the one case where the result has a group the CIF cannot
carry: the warning then says to set `symmetry_handling` to `SPGLIB` in your
options, so that cif2d12 finds the group again and the deck is not P1.

## Charge and spin

- `--charge Q` (the GUI's **Charge**, prefilled from the document) goes to
  GFN-FF. A wrong charge is accepted silently by xtb: check it.
- A .d12 with CHARGED, CHEMOD or DOPING needs `--charge` for gfnff, mlip and
  auto (the GUI asks for it); the springs ignore the charge.
- `--uhf N` is passed to xtb and recorded, but **GFN-FF ignores it**.
- The ML models take no charge or spin input. They were trained on Materials
  Project GGA/GGA+U data, so magnetic order and +U are implicit and averaged;
  an explicit `mlip:` run with a charge warns that the model sees a neutral
  cell.

## Decks that edit their geometry

MACE reads a .d12's geometry as its asymmetric unit expanded by the space
group. It does not apply the records that make CRYSTAL build a different
structure: SUPERCEL, SUPERCON, SCELCONF, SCELPHONO, ATOMREMO, ATOMSUBS,
ATOMINSE, ATOMDISP, ATOMROT, ATOMORDE, ELASTIC, BREAKELAS, USESAED, POINTCHG,
ROTCRY, SLABCUT, SLABINFO, NANOTUBE, SWCNT, NANOMULTI, CLUSTER, CLUSTSIZE,
FULLE, HYDROSUB, MOLECULE, MOLEXP, MOLSPLIT, MOLEBSSE, ATOMBSSE, NANORODS,
NANOCRYSTAL, WULFF (CRYSTAL23 manual pp. 35-37), the basis-set GHOSTS, and a
line-2 geometry other than CRYSTAL. `mace preopt` refuses such a deck (exit 1):
pre-optimize the expanded structure instead, by opening the .out of a run of the
deck or exporting a CIF from it. In the GUI, such a document is the host
geometry: Clean offers the springs only, with a warning that they act on the
host, not on the structure CRYSTAL builds.

## Output and provenance

- `<stem>_preopt.cif`: the geometry, with a comment ("Pre-optimized with ...
  Starting geometry only, not a CRYSTAL result. Run a CRYSTAL OPT from this
  geometry before SP/FREQ/BAND/DOSS.") and one marker line,
  `# mace-preopt: method=... model=... version=1`.
- `<stem>_preopt.json`: MACE's version, the input's file name and sha256, the
  method asked for and the one that ran, every attempt (status, reason, steps,
  energy and volume change, space group before and after, largest shift,
  seconds), the code, model and their licenses, the weights' sha256, training
  data and citations, the device, charge, uhf, symmetry options, fmax, steps
  and the output CIF's mode. No absolute paths and no host names.
- `--json` prints the same record on stdout. Nothing is overwritten without
  `--force`.

`mace convert` prints a warning (and changes nothing else) when a CIF carrying
the marker of a GFN-FF or ML run is turned into a deck that is not an OPT. The
GUI's Setup panel does the same for a document accepted from such a Clean, and
Save CIF of that document writes the marker line. Accepting a GFN-FF/ML Clean
records the method, model, license, training data, charge and citation in the
document's provenance (never in a deck title).

Exit codes: 0 written (the requested method, or whatever Auto ran); 1 nothing
written (bad input, a refused deck, a charged deck without `--charge`,
`--no-fallback` after a failure, nothing to clean); 2 usage error; 3 the spring
result written instead of an explicit gfnff/mlip run.

## Installing the optional tools

Neither tool is needed for the springs, and MACE never installs either.

**xtb** (GFN-FF; LGPL-3.0, run as a separate program, never bundled):

```
conda install -c conda-forge "xtb>=6.7"              # about 0.65 GB
export MACE_XTB=/path/to/xtb                         # a specific build
```

On a cluster with environment modules (Lmod or Tcl), MACE finds and loads an
xtb module itself: the newest 6.7 or later that `module -t avail xtb` lists,
loaded in a separate shell for xtb's own process only (your shell and MACE's
Python keep their modules). The search order is `MACE_XTB`, then `xtb` on PATH
(6.7 or newer), then the module. Pin one module with
`export MACE_XTB_MODULE=<exact module name>`, turn the search off with
`MACE_XTB_MODULE=none`, or point `MACE_XTB` at a binary. The module found is
remembered for 7 days in `~/.cache/mace/modules.json`, and the run's
provenance names it ("xtb 6.7.1 (module ..., loaded automatically)"). See
[HPCC notes](#hpcc-notes) for a fixed wrapper instead.

xtb before 6.7 has no periodic GFN-FF; MACE says "Found xtb 6.6.1; GFN-FF for
periodic cells needs 6.7+". Without coreutils' `timeout`, an xtb whose parent
was killed runs until it finishes on its own (MACE still stops it on a cancel).

**An ML environment.** `mace mlip install` makes it, or the GUI's
**Settings › ML potentials › Set Up ML Environment…** (the same installer):
it shows the folder, the Python, each package's license, the size and the
space and file quota, then runs the steps below and records the result. By
hand (`mace mlip setup` prints these steps): Make it in the default
place, `DIR=${XDG_DATA_HOME:-~/.local/share}/mace-tool/mlip/venv` beside the
weights, and MACE finds it with nothing set or recorded (the
GUI too). An environment elsewhere works when recorded with
`mace mlip setup --python`; `MACE_MLIP_PYTHON`, then the record, win over the
default place. `mace mlip list` says which one it uses.

```
python -m venv --system-site-packages DIR            # a Python that already has torch
DIR/bin/python -m pip install --upgrade pip            # an old pip (22.x) fails on torch's dependencies
DIR/bin/pip install mace-torch==0.3.16 chgnet==0.4.2   # about 80 MB
# no torch yet: a CPU-only torch first (several hundred MB; a plain
# `pip install torch` pulls a CUDA build of several GB); with mace-torch
# and chgnet about 1.5 GB and 46,000 files in all
DIR/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
df -h DIR                                            # check there is room
mace mlip setup --python DIR/bin/python               # probe and record it (optional in the default place)
mace mlip fetch acesuit-mace-mpa-0-medium            # shows the license, then downloads 79 MB
mace mlip list                                       # what is ready
```

`mace mlip fetch` downloads model weights (`mace mlip install` also installs packages and calls it). It shows
the model, the licenses with their links, the size, URL, sha256, training data
and citation, asks (pass `--yes` in a script), verifies size and sha256, and
keeps `LICENSE.txt` and `ACCEPTED.json` next to the weights in
`$XDG_DATA_HOME/mace-tool/mlip/<id>/` (or `MACE_MLIP_DIR`). That is kept apart
from `~/.cache/mace`, which mace-torch uses for its own downloads. The worker
always runs offline.

In the GUI an unavailable method is grayed out with these commands,
**Recheck** looks again, and Settings › ML potentials installs the environment
and downloads weights, each only when you ask (the same installer and fetch).

## Environment variables

| Variable | Meaning |
|---|---|
| `MACE_XTB` | the xtb program to use (else `xtb` on PATH) |
| `MACE_MLIP_PYTHON` | the ML environment's interpreter (else the one `mace mlip setup --python` recorded, else `$XDG_DATA_HOME/mace-tool/mlip/venv/bin/python` when it exists) |
| `MACE_MLIP_DIR` | where the weights are (else `$XDG_DATA_HOME/mace-tool/mlip`) |
| `MACE_MLIP_NO_DOWNLOAD=1` | `mace mlip fetch` refuses to download (compute nodes) |

## Resources

- CPU: xtb runs with one thread; the ML worker with `--threads K` (default a
  quarter of the CPUs this process may use, 1 to 4, or SLURM's allocation). In
  the GUI everything runs in the niced process pool, the ML model is loaded
  once in one niced worker that stops after 5 minutes idle, and an ML clean
  holds as many CPU slots as the worker's threads.
- GPU: never by default. `--device cuda:N` (CLI only) shows only GPU N to the
  worker, where it is called `cuda:0`, and caps it to 15 % of that GPU's
  memory. On shared workstations, pick the GPU your admins allow. Inside a job
  whose GPUs are already restricted (`CUDA_VISIBLE_DEVICES` set by SLURM or the
  site), N counts within that list (`3,5` and `cuda:1` give GPU 5), and an N
  outside it, or a cuda device in a job without GPUs, is refused.
- Time: about 0.13 s per GFN-FF call; 25-400 ms per ML step for small cells and
  2-4 s at 216 atoms on the CPU, which is why the GUI refuses ML above 300
  atoms and uses a Run button for these methods.
- Temporary files live in `<tmp>/mace-preopt-<uid>/` and are removed when a run
  ends, is canceled or fails; directories of killed processes are swept later.

## HPCC notes

Fetch the ML weights on a dev node (compute nodes may have no network), then run
jobs with `MACE_MLIP_NO_DOWNLOAD=1`.

**An ML environment on a module Python (MSU HPCC as an example).** Make it on a
dev node, where the default `Python/3.11.3-GCCcore-12.3.0` module is loaded, in
the default place (`~/.local/share/mace-tool/mlip/venv`; home is shared by every
node, so the GUI and jobs find it too), and run the pip lines after `unset PYTHONPATH`: the module's PYTHONPATH (its
Python-bundle-PyPI site-packages) makes pip skip packages the module already
has, and MACE runs the environment without PYTHONPATH, so those would be
missing. Upgrade pip first; the module's pip 22.3 discards current wheels and
fails. `mace mlip install` does all of this for you.

The environment's python finds its `libpython3.x.so` only through the
`LD_LIBRARY_PATH` its module sets; without it, it stops with "error while
loading shared libraries". Its standard library may need more of the
module's toolchain: the 3.11 module's `_ssl` loads OpenSSL 1.1's
`libssl.so.1.1`, which the system does not have. MACE therefore records those
folders with the interpreter: `mace mlip install` and `mace mlip setup
--python` (the GUI's Use an Existing Python…) start the environment's python
once with no `LD_LIBRARY_PATH`, import its standard library's extension
modules, find each missing library in `lib` beside its real path (the
module's prefix), beside the `home` in its `pyvenv.cfg` or in a folder of the
current `LD_LIBRARY_PATH` (the module's, when it is loaded), and save the
folders in `~/.config/mace/mlip.json` under `library_path`. Every time MACE
starts that interpreter (the probe, the ML worker in `mace calc`, `mace
preopt`, the GUI and a job's callback, pip in a repair) the folder goes in
front of that child's `LD_LIBRARY_PATH` only, so the environment runs on a
dev node, a compute node, the gateway or under a shell with another Python
module loaded, without loading its module by hand. This holds for every
environment: the default `venv`, and `venv-<name>` (for example
`venv-fairchem`) or one elsewhere. An environment recorded before MACE did
this records, at the first probe that needs it, what it finds beside the
interpreter; record it once more where its module is loaded (`mace mlip
setup --python DIR/bin/python`) so that the toolchain's folders are recorded
too. `mace mlip list` names a library MACE could not find
("cannot open shared object file") and says so.

On MSU HPCC, for example, orb-models 0.7 (ORB-v3, OrbMol-v2) needs Python
3.12, which Lmod does not load beside the default 3.11 module. Give `mace
mlip install` the 3.12 module's interpreter from the shell MACE runs in; it
needs no `module load`, because MACE finds that interpreter's library folder
the same way:

```
mace mlip install --backend orb-models --fetch orbmol-v2 \
    --python /opt/software-current/2023.06/x86_64/generic/software/Python/3.12.3-GCCcore-13.3.0/bin/python3.12
```

(`( module purge; module load Python/3.12.3-GCCcore-13.3.0; command -v
python3.12 )` prints that path.) MACE itself keeps running on the 3.11
module, and `mace calc -m orbmol-v2` then runs with the 3.11 module loaded,
with no module loaded, and in jobs.

**xtb on a cluster (MSU HPCC as an example).** MACE loads the newest suitable
xtb module by itself (see above), also when that module cannot be loaded beside
the Python module MACE runs on: the load happens in a separate shell after
`module purge`, for xtb only. A fixed wrapper is the alternative when you
prefer not to rely on the module search. On MSU HPCC, for example, the module
`xtb/6.7.1-gfbf-2024a` (6.6.1 is too old) cannot be loaded beside the Python
module MACE runs on
(`Python/3.11.3-GCCcore-12.3.0`): they are built with different toolchains
(gfbf/2024a and gfbf/2023a), and Lmod refuses to load both. The module's `xtb`
binary does not run without the module's libraries either. xtb only needs its
own library path, so make a small wrapper once, in a subshell (your shell keeps
its modules), and point `MACE_XTB` at it (in `~/.bashrc` to keep it):

```
mkdir -p ~/bin
( module purge; module load xtb/6.7.1-gfbf-2024a &&
  printf '#!/bin/sh\nexport LD_LIBRARY_PATH="%s"\nexec "%s" "$@"\n' "$LD_LIBRARY_PATH" "$(command -v xtb)" > ~/bin/xtb-gfnff &&
  chmod +x ~/bin/xtb-gfnff )
export MACE_XTB=~/bin/xtb-gfnff
```

The wrapper does not load the module on each call: Lmod takes about 1.5 s, and
GFN-FF runs xtb once per force call, so a run would mostly wait for Lmod. Then
`mace preopt --method gfnff`, or Recheck in the GUI's Clean dialog, finds xtb
6.7.1. The wrapper works in SLURM jobs too, since its paths are absolute.

## Limits

- ML lattices come out about 1 % too large (PBE-like training data).
- The ML models take no charge or spin input; magnetic order and +U are implicit
  and averaged (MP GGA/GGA+U training data).
- GFN-FF ignores `--uhf` (recorded only) and accepts a wrong `--charge`
  silently; its cells are often several percent off, and it fails on dense
  ionic oxides and halides.
- ML is refused above 300 atoms on the CPU in the GUI.
- Not supported yet: UFF/UFF4MOF, GULP's GFN-FF, tblite GFN1/GFN2, ASL and Meta
  models, a GUI install or download, a .d12 with only the geometry replaced
  (`mace preopt` writes a CIF), and pre-optimization as a workflow step.

## Citations

Cite the method you used (the CLI prints it, the GUI's **Copy citation** copies
it, and the provenance JSON records it):

- GFN-FF: S. Spicher and S. Grimme, Robust atomistic modeling of materials,
  organometallic, and biochemical systems, Angew. Chem. Int. Ed. 59, 15665
  (2020), doi:10.1002/anie.202004239.
- The xtb program: C. Bannwarth et al., Extended tight-binding quantum
  chemistry methods, WIREs Comput. Mol. Sci. 11, e1493 (2021),
  doi:10.1002/wcms.1493.
- MACE architecture: I. Batatia et al., MACE: Higher order equivariant message
  passing neural networks for fast and accurate force fields, NeurIPS 2022,
  doi:10.48550/arXiv.2206.07697.
- MACE-MP foundation models: I. Batatia et al., A foundation model for
  atomistic materials chemistry, arXiv:2401.00096,
  doi:10.48550/arXiv.2401.00096.
- CHGNet: B. Deng et al., Nat. Mach. Intell. 5, 1031 (2023),
  doi:10.1038/s42256-023-00716-3.

Training data (attribution):

- MPtrj, the Materials Project trajectories (CC BY 4.0; Materials Project terms
  of use): both models.
- sAlex, a subset of Alexandria (CC BY 4.0): L. Barroso-Luque et al., Open
  Materials 2024 (OMat24) Inorganic Materials Dataset and Models,
  arXiv:2410.12771. Alexandria itself: Th. Cavignac, J. Schmidt et al.,
  AI-Driven expansion and application of the Alexandria database, J. Phys.
  Mater. 9, 025014 (2026), doi:10.1088/2515-7639/ae6620 (the citation the
  Alexandria site asks for): MACE-MPA-0.
