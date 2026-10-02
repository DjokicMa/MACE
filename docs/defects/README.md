# Point defects: `mace defects`

`mace defects` builds the decks for a point defect in a periodic cell (a
pristine supercell plus one single-point deck per charge state), and reads the
results back into formation energies and charge-transition levels.

Every deck is made by **editing the text of your host deck**. The method,
basis set, grids, tolerances and every other host line are kept byte for byte;
only the supercell, defect, charge and spin records are added. A defect set
therefore always runs exactly the host's method.

Where this page says *measured*, the behaviour was run with CRYSTAL23
(23-intel-2023a) on HPCC; *from the manual* cites the CRYSTAL23 user's manual;
*untested* means neither.

## Workflow

```bash
# 1. a single-point host deck (an OPT deck holds the initial geometry: make an SP
#    of the optimised structure first)
mace opt2d12 --calc-type SP ...            # -> mgo_sp.d12

# 2. the deck set
mace defects build mgo_sp.d12 --vacancy O --supercon 2x2x2 --charges 0,1,2 --calibrate -o mgo_VO

# 3. without a site label only the pristine deck is written (status
#    "awaiting_pristine"): CRYSTAL numbers the supercell atoms, so the label comes
#    from its output. Submit, then resume as soon as the pristine .out shows its
#    geometry tables (seconds after the job starts - no need to wait for the SCF):
mace submit $PWD/mgo_VO
mace defects build --resume mgo_VO
for j in mgo_sp_q0 mgo_sp_qp1 ...; do mace submit $PWD/mgo_VO/$j.d12; done   # as printed

# 4. optional: electrostatic potential at the atoms, for potential alignment
mace defects potc mgo_VO                    # -> mgo_VO/potc/<job>_potc.d3 + .f9
mace submit $PWD/mgo_VO/potc

# 5. results
mace defects analyze mgo_VO --mu O=-75.0Ha --json report.json
```

`mace submit` takes one target, so after a resume submit only the new decks
(the command line is printed). Give `mace submit` absolute paths: with a
relative path into another directory (`mace submit set/x.d12`) it does not find
the job script it has just generated (an existing quirk of `mace submit`; the
printed commands use absolute paths). `potc` writes into `DIR/potc/` so that a later
`mace submit DIR` never picks up property decks without their wave function.

### build options

| option | meaning |
|---|---|
| `--vacancy EL` | remove one atom (`ATOMREMO`, p.42) |
| `--ghost EL` | vacancy keeping the basis functions (`GHOSTS`, p.82) |
| `--substitute OLD=NEW` | replace one atom (`ATOMSUBS`, p.43) |
| `--insert EL@x,y,z [--cart]` | add one atom (`ATOMINSE`, p.41): fractional in the host deck's cell, or with `--cart` Cartesian Angstrom in the supercell frame. **Untested on HPCC.** |
| `--site-label N` | the supercell atom, in CRYSTAL's numbering |
| `--site-frac x,y,z` | the site in the host deck's cell (matched within 0.3 A, under PBC) |
| `--supercell NxNxN`, `--supercell-matrix M` | `SUPERCEL` (expands the primitive cell, pp.75-77) |
| `--supercon NxNxN` | `SUPERCON` (expands the crystallographic cell; 3D only) |
| `--sizes 3,4,5,6` | 2D: one n x n set per size, for the A + B/L fit |
| `--charges 0,1,2` | charge states q (`--charges=-1,0,1` when the list starts with a minus) |
| `--spin-state q=NSPIN` | n_alpha - n_beta for one charge state (repeatable) |
| `--spinlock-cycles 30` | SPINLOCK length (30 as measured; MACE's writer default of 50 is not used here) |
| `--host-spin keep` | accept SPINLOCK / ATOMSPIN / BETALOCK in the host; then `--spin-state` is required for every q |
| `--calibrate [0.5,1.0,1.5]` | 3D: fractional-DOPING calibration decks (below) |
| `--calibrate-on pristine` | build them on the pristine cell instead of the q=0 defect cell (**untested as a calibration source**) |
| `--calibrate-endpoints` | also restricted DOPING 0 / +-2 decks |
| `--pristine-out FILE` | the pristine supercell output, when it is somewhere else |
| `--shrink 'IS [ISP]'` / `'ka kb kc'` | replace the host's SHRINK in every deck (default: kept, with the equivalent-density value printed) |
| `--name`, `-o`, `--force`, `--resume DIR` | job prefix, output directory, overwrite / GHOSTS on an ECP atom, finish a waiting set |

## What goes where

| record | where | why |
|---|---|---|
| `SUPERCEL` / `SUPERCON` | block 1, after the geometry and any other block-1 keyword (e.g. `SLABCUT`), before `END` / `BASISSET` | pp.74-77 |
| `ATOMREMO` / `ATOMSUBS` / `ATOMINSE` | right after the supercell records, so labels and coordinates refer to the supercell | pp.41-43 |
| `GHOSTS` | after the `BASISSET` name line, or after the `99 0` of an explicit basis | p.82; measured |
| `DOPING`, `SPINLOCK` | SCF block, just before a trailing `PPAN` (else before the final `END`) | p.103, p.128; reproduces the measured decks byte for byte |
| `SPIN` (DFT) / `UHF` (HF; `RHF` becomes `UHF`) | after the `DFT` line / start of block 3 | only when a state needs a spin lock |
| `PPAN` | before the final `END`, in **every** deck, when the host has none (printed) | the electron count is read from the Mulliken line, printed only with PPAN |

## Charge and spin

* `DOPING = -q` (p.103): electrons added; `-1.0` removes one electron. No
  record for q = 0.
* The electron count of the defect cell is `N_def = N_super + dZ - q`. N_super
  comes from the pristine output when there is one; otherwise only its parity
  is known: the supercell holds m primitive cells with the same count, so an
  even m gives an even N_super, and a P1 host gives the parity of the sum of Z
  (ECP cores and default shell charges are even or neutral). An odd m of a
  symmetric host waits for the pristine output.
* Default spin: n_alpha - n_beta = 0 for even N, 1 for odd N. A different
  state (e.g. a triplet) is asked for with `--spin-state q=NSPIN`; NSPIN must
  have the parity of N (p.128), otherwise the build is refused.
* Odd N, or NSPIN != 0: `SPIN` (or `UHF`) plus `SPINLOCK NSPIN 30`. Measured:
  "SPIN + SPINLOCK 1 30 works" for odd ATOMREMO + DOPING states.
* A host with `SPIN` (MACE's default) is kept as it is: even states only get
  `DOPING`, odd states only `SPINLOCK`.
* Every choice is printed per state, e.g. `q=+1: N=631 odd -> SPINLOCK 1 30`.

## Ghost vacancies

* Even N: `GHOSTS` + `DOPING` (measured: MgO V_O q=0 and q=+2).
* Odd N with an internal basis: **refused**. Measured: `GHOSTS + DOPING +
  SPINLOCK` aborts (`GHOST NALPHA-NBETA NOT COMPATIBLE WITH NUMBER OF
  ELECTRONS` - the parity is checked before doping; under Pcrystal only a bare
  MPI_Abort is printed); without SPINLOCK the state collapses to spin 0; and
  `CHEMOD + CHARGED` is silently ignored by internal libraries (the run stays
  neutral). Use `--vacancy`, or an explicit-basis host.
* Odd N with an explicit basis: the measured recipe `GHOSTS / CHEMOD 1 <far
  atom> <shell charges> / CHARGED` after `99 0`, no DOPING, plus SPIN +
  SPINLOCK. The far atom is the atom of the host's anion element farthest from
  the ghost (PBC; ties: lowest label), from the pristine output. q is taken
  from its last occupied shell (added electrons go to the first shell with
  room; capacities s 2, sp 8, p 6, d 10, f 14).
* `GHOSTS` on an atom with an effective core potential (explicit Z >= 200, or
  an internal library with Z > 36) is refused unless `--force`: "Do not use
  with ECP" (p.82).
* `GHOSTS` with -D3, gCP or a 3c method: **untested** (warned).

## 3D: fractional-DOPING calibration

Measured on MgO V_O (2x2x2, 63 atoms): raw E(q) - E(0) of a charged supercell
carries a constant offset C per electron. With raw energies eps(+/0) comes out
15.7 eV above the VBM inside a 7.25 eV gap, which is impossible. Fractional
DOPING runs give dE/dN = eps_F + C with C = 0.444 Ha, constant to 0.001-0.004
Ha along the ladder and tracking 2 x (EXT EL-SPHEROPOLE) / N_e. **That relation
is a hypothesis, not proven**; MACE reports it as a cross-check only.

`--calibrate` writes restricted decks with DOPING -0.5, -1.0, -1.5 (on the side
of the requested charges: +0.5, +1.0, +1.5 for acceptors; both when q has both
signs). They are restricted (SPIN, SPINLOCK, UHF removed, recorded in the
manifest as a deliberate difference from the host) and placed on the q = 0
defect cell by default. Measured round 2: the pristine-cell C agrees with the
defect-cell C within 0.004 Ha for MgO (conventional 2x2x2 and primitive 3x3x3),
so `--calibrate-on pristine` is offered, but untested as a source for a defect
set. Fractional DOPING needs a converged k-mesh: "the appropriate value of the
shrinking factor ... is in some cases several times larger than for the
neutral cell" (p.103). SMEAR hosts are refused (free energies, smeared
occupations); LEVSHIFT hosts are warned.

At each interior point i of the ladder, with its neighbours i-1 and i+1:

    C_i = (E_{i+1} - E_{i-1}) / (N_{i+1} - N_{i-1}) - EFERMI_i

The endpoints are the q = 0 and q = 2 defect decks when restricted or ending
with |Mulliken spin| < 0.01, else the `--calibrate-endpoints` decks. Without
usable endpoints only the central value at DOPING -1.0 exists, and the
analysis says so. C is the mean; a spread above 0.005 Ha is a warning.

    E_corr(q) = E_raw(q) + q C        only for q inside the calibrated range

A charge outside the range is "not calibrated" and not reported (never
extrapolated). Without a calibration only q = 0 is reported, with the reason.
Every 3D result carries `"image_charge_correction": "none"` and the label
**"uncorrected for image charges"**: no dielectric correction exists yet.

## 2D: size extrapolation

Measured on hBN V_N (3x3 to 6x6): charged slabs are accepted silently, and
fractional slopes match eps_F directly (no offset). E(q) - E(0) converges as

    dE_q(L) = A + B / L,   L = sqrt(|a x b|)  (Angstrom, of the supercell)

fitted by least squares (at least 2 sizes, a warning below 3; residuals are
reported). Then eps(+/0) = -A_{+1} - E_VBM and eps(0/-) = A_{-1} - E_VBM, with
E_VBM from the largest pristine cell (the spread across sizes is reported).
Measured: eps(+/0) about 3.8 eV and eps(0/-) about 6.6 eV above the VBM (gap
6.91 eV), residuals <= 0.02 eV. The A + B/L form is empirical. A charged sheet
is not compensated in the vacuum (POTC planar average: V linear in |z|), so
there is **no vacuum level** in a charged 2D cell.

## Formation energies and levels

    E_f(q; E_F) = E_corr(q) - E_pristine - sum_i n_i mu_i + q (E_VBM + E_F)

* n_i > 0 for atoms added (vacancy: n = -1).
* mu_i are **absolute** energies with units (`--mu O=-75.0Ha`, `--mu O=-2040.8eV`
  or `--mu-file`), from the **same method and basis**. Without mu, E_f is
  reported relative to q = 0.
* E_VBM is the top of the valence bands on the pristine cell's SCF k-mesh;
  its band and k index are printed; `--vbm` overrides it.
* eps(q/q') = (E_f(q'; 0) - E_f(q; 0)) / (q - q'), flagged when outside [0, gap].
* All results are **unrelaxed** (ions at the pristine positions).

## Potential alignment (POTC)

`mace defects potc DIR` writes `POTC / 0 0 0 / END` decks (V at every atom;
POTC.DAT columns: index, x y z in bohr, V0 in au) and copies `<job>.f9` to
`DIR/potc/<job>_potc.f9`, refusing a missing or empty `.f9`; `--planar ZM ZP`
(2D) adds `POTC / 2 41 0` planar averages (bohr). The analysis matches atoms of
the defect and pristine cells by their bohr coordinates under PBC and reports
dV = <V_def - V_pris> over the farthest quarter of the atoms from the defect,
with its spread (a warning above 0.05 eV). It is **reported always and applied
only with `--apply-alignment`**, as q x (-dV). Measured: a 2x2x2 MgO cell is too
small for a plateau. Alignment may double-count with C; it is not applied by
default for that reason.

## Refusals

| host or request | reason |
|---|---|
| `MOLECULE` | a cluster has no defect supercell; write the charge with DOPING (charged-systems records) |
| `POLYMER`, `HELIX` | only 3D and 2D hosts |
| `EXTERNAL` / `DLVINPUT` | the geometry is in another file and cannot be checked |
| `OPTGEOM` | an OPT deck holds the initial geometry and a charged supercell must not relax its cell: `mace opt2d12 --calc-type SP` |
| `FREQCALC`, `ELASTCON`, `EOS`, `SCELPHONO` | build from an SP deck |
| existing `SUPERCEL`, `SUPERCON`, `ATOMREMO`, `ATOMSUBS`, `ATOMINSE`, `GHOSTS` | build from the pristine primitive host |
| `KEEPSYMM` in block 1 | ATOMREMO / ATOMSUBS / ATOMINSE / GHOSTS would act on every symmetry-equivalent atom (pp.41-44, 54, 82) |
| `CHEMOD`, `CHARGED`, `DOPING` in the host | already charged |
| `BOHR` / `FRACTION` with `--insert` | they change the units of the ATOMINSE coordinates (p.41) |
| `SPINLOCK`, `ATOMSPIN`, `BETALOCK` without `--host-spin keep` | a host spin lock is not the spin state of each charged supercell |
| `TWOCOMPON` | one-component sets only |
| `SUPERCON` on a 2D host (SLAB, CRYSTAL + SLABCUT) | SUPERCON refers to the 3D crystallographic cell; use a 2x2 SUPERCEL |
| a singular or non-integer supercell matrix | pp.74-77 |
| NSPIN of the wrong parity | p.128 |
| a new element missing from an explicit basis | add its basis block to the host (no appending in this version) |
| a new element missing from the internal library | measured library coverage (d12_constants) |
| an element given alone that has several sites in the host | choose with `--site-frac` or `--site-label` (the sites are listed) |
| `--site-frac` with SUPERCEL on a centred host | SUPERCEL expands the primitive cell; use SUPERCON or a label |
| an inserted atom closer than 0.8 A to an atom | |
| odd ghost states with an internal basis; GHOSTS on an ECP atom | see Ghost vacancies |
| `--calibrate` on a SMEAR host or a 2D host | see above |

## Analysis checks

* The electron count is the Mulliken `NO. OF ELECTRONS` line, never the header
  `N. OF ELECTRONS PER CELL`, which shows the count **before** DOPING
  (measured). A job whose count differs from N_pristine + dZ - q by more than
  1e-4 is excluded ("DOPING not applied").
* |Mulliken spin - NSPIN| > 0.1 is a warning: the state drifted after the lock
  was released.
* Unconverged SCFs (`TOO MANY CYCLES`) are excluded with the reason.
* Site report: Mulliken charges (and spins) of the defect's neighbours (within
  1.2 x the shortest distance), the ghost's XX row, or the substituted /
  inserted atom, against the same atoms of the pristine cell (matched by
  position after ATOMREMO renumbering).

## Example: MgO V_O

`MgO_host.d12` is the PBE0 / POB-TZVP-REV2 MgO deck of the measured runs and
`MgO_VO/` the set built from it with

    mace defects build docs/defects/MgO_host.d12 --vacancy O --site-label 1 \
        --supercon 2x2x2 --charges 0,1,2 --calibrate --name mgo -o docs/defects/MgO_VO

Label 1 (an O at the origin) was read from the measured probe output of the
same supercell. The decks are byte-identical (titles aside) to the ones run on
HPCC: q = 0, +1, +2 and the calibration decks (DOPING -0.5, -1.0, -1.5). The
set is not run here; its measured results were C = 0.444 Ha, eps(2+/+) about
1.9 eV and eps(+/0) about 3.6 eV above the VBM (unrelaxed, uncorrected for image
charges). `mu.example.json` shows the `--mu-file` format; its value is a
placeholder.

## Known limits

* Labels come from CRYSTAL's output or from you, which costs one round trip
  (shortened by reading the running pristine output).
* The C hypothesis is unproven; calibration covers only the measured direction
  (electron removal on MgO) and the range of the ladder.
* SPINLOCK release can let a state drift; the final spin is checked.
* No relaxation, no dielectric (image-charge) correction.
* Not handled: decks derived later from a ghost parent (opt2d12 / opt2d3) do
  not carry GHOSTS.
