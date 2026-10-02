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

# 2. the deck set (--calibrate adds the optional fractional-DOPING cross-check decks)
mace defects build mgo_sp.d12 --vacancy O --supercon 2x2x2 --charges 0,1,2 -o mgo_VO

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

# 5. 3D charged states: the dielectric constant for the image-charge term, from a
#    CPHF run of the host (CPHF / END as the last geometry keyword), or --eps VALUE
mace submit $PWD/mgo_cphf.d12

# 6. results
mace defects analyze mgo_VO --mu O=-75.0Ha --eps-from mgo_cphf.out --json report.json
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

## 3D: charged-cell offset and image charge

Measured on CRYSTAL23 (MgO, LiF and Si, pristine and defect cells of 53-249
atoms): the raw energy of a charged supercell carries an offset C per electron.
With raw energies eps(+/0) of MgO V_O comes out 15.7 eV above the VBM inside a
7.25 eV gap. C does not shrink with the cell and is material dependent (MgO
0.449, LiF 0.307, Si 0.445 Ha); it equals

    C = 2 E_sph / N          (to 0.2-1 % in all 10 measured series)

with E_sph the last `::: EXT EL-SPHEROPOLE` term of the output and N the
electron count of the cell: the header `N. OF ELECTRONS PER CELL` (printed
before DOPING) plus the DOPING value. The analysis applies it as the trapezoid
between the charge state and the neutral defect of the same cell:

    E_corr(q) = E_raw(q) + q (C(q) + C(0)) / 2

(E_raw + q C(q) alone leaves 0.13 eV at q = 2). Janak residuals after this
correction are at most 0.03 eV. No extra runs are needed; a state whose output
has no spheropole line, or a set without a converged q = 0 defect, is not
reported, with the reason. The correction was validated on electron removal
(q > 0); negative states get a note.

The image-charge term is the Makov-Payne monopole term with the Lany-Zunger
scaling, added to E_f(q):

    E_MP(q) = -f q^2 v_M / (2 eps),    f = 1 - (1 - 1/eps) / 3

v_M is the Ewald Madelung potential of the actual supercell lattice (read
from the pristine output; simple cubic: -2.837297 / L, so E_MP > 0). A cell
elongated or flattened by a factor of about 4 or more has v_M > 0 and a
negative E_MP. eps is

* `--eps VALUE`: used as given;
* `--eps-from FILE`: eps_inf from the `DIELECTRIC TENSOR` of a CPHF output
  (`CPHF` / `END` as the last geometry keyword of the host), or of a FREQCALC
  INTENS INTCPHF output; with `--eps-static`, eps_0 = eps_inf + the `SUM TENSOR
  OF THE VIBRATIONAL CONTRIBUTIONS TO THE STATIC DIELECTRIC TENSOR` of a
  FREQCALC INTCPHF output. Tensors are averaged (a warning above 5 %
  anisotropy).

eps_inf is the choice for these unrelaxed (vertical) defects; eps_0 is for
relaxed ones and gets a note. Measured MgO: eps_inf 2.865, eps_0 9.34. Without
a dielectric constant the charged states are reported with the offset only and
labelled **"uncorrected for image charges"**; their levels then still depend on
the cell size.

Measured, MgO V_O (unrelaxed, PBE0/POB-TZVP-REV2, eps_inf 2.865):

| cell | atoms | eps(+/0) | eps(2+/+) |
|---|---|---|---|
| primitive 3x3x3 | 53 | 2.851 | -0.193 |
| conventional 2x2x2 | 63 | 2.883 | -0.092 |
| primitive 4x4x4 | 127 | 2.878 | -0.137 |
| primitive 5x5x5 | 249 | 2.875 | -0.146 |

Every report says which corrections were applied: the labels name the offset
(spheropole or calibration), the image-charge term with its eps and v_M (or
"uncorrected for image charges") and whether potential alignment was applied.

### Optional cross-check: fractional-DOPING calibration

`--calibrate` writes restricted decks with DOPING -0.5, -1.0, -1.5 (on the side
of the requested charges: +0.5, +1.0, +1.5 for acceptors; both when q has both
signs). They are restricted (SPIN, SPINLOCK, UHF removed, recorded in the
manifest as a deliberate difference from the host) and placed on the q = 0
defect cell by default. Measured: the pristine-cell C agrees with the
defect-cell C within 0.004 Ha for MgO (conventional 2x2x2 and primitive 3x3x3),
so `--calibrate-on pristine` is offered as well. Fractional DOPING needs a
converged k-mesh: "the appropriate value of the shrinking factor ... is in
some cases several times larger than for the neutral cell" (p.103). SMEAR
hosts are refused (free energies, smeared occupations); LEVSHIFT hosts are
warned.

At each interior point i of the ladder, with its neighbours i-1 and i+1:

    C_i = (E_{i+1} - E_{i-1}) / (N_{i+1} - N_{i-1}) - EFERMI_i

The endpoints are the q = 0 and q = 2 defect decks when restricted or ending
with |Mulliken spin| < 0.01, else the `--calibrate-endpoints` decks. Without
usable endpoints only the central value at DOPING -1.0 exists, and the
analysis says so. C is the mean; a spread above 0.005 Ha is a warning.

When the calibration decks exist, the analysis prints C next to the
spheropole values and the levels it would give (`levels_calibration_crosscheck`
in the JSON). `analyze --offset calibration` uses it instead:

    E_corr(q) = E_raw(q) + q C        only for q inside the calibrated range

A charge outside the range is then "not calibrated" and not reported (never
extrapolated). Measured on MgO V_O (conventional 2x2x2) the calibration C is
0.444 Ha and the levels differ from the spheropole ones by 0.02-0.05 eV.

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

    E_f(q; E_F) = E_corr(q) - E_pristine - sum_i n_i mu_i + q (E_VBM + E_F) + E_MP(q)

* n_i > 0 for atoms added (vacancy: n = -1).
* mu_i are **absolute** energies with units (`--mu O=-75.0Ha`, `--mu O=-2040.8eV`
  or `--mu-file`), from the **same method and basis**. Without mu, E_f is
  reported relative to q = 0.
* Every E is the cell's total energy **including D3 dispersion and gCP**
  where the method has them (-D3 functionals, 3c methods such as HSEsol-3c):
  SCF + D3 + gCP, which is CRYSTAL's `TOTAL ENERGY + DISP (+ GCP) (AU)` line,
  not the bare `== SCF ENDED ... E(AU)` value. The terms do not cancel when
  the defect cell has an atom fewer or more than the pristine cell, so mu_i
  must be total energies on the same footing. They do cancel between charge
  states at fixed geometry: levels and the offsets are unchanged.
  A job whose terms differ from the pristine's (e.g. D3 missing) is excluded.
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
only with `--apply-alignment`**, as q x (-dV). Measured on MgO V_O: a 2x2x2
cell is too small for a plateau, and with the offset and image-charge terms
applied, aligning the far-atom potentials made the agreement between cell
sizes **worse** for charged cells (53 and 127 atom cells). It is off by
default for that reason.

**Sign convention (assumed, not verified).** The term added is q x (-dV), with
dV the far-atom mean of V0(defect) - V0(pristine) from POTC.DAT, converted from
au to eV. That sign is right if V0 is the electrostatic potential in the usual
convention - the energy of a unit *positive* test charge, so that the defect
cell's one-electron levels (its VBM) sit -dV relative to the pristine cell's.
The manual (p.352) calls V "the exact electrostatic potential" and relates it to
the charge density by Poisson's law, but MACE has not checked the sign against a
reference calculation. Before relying on `--apply-alignment`, check the sign of
dV on a case with a known answer.

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

* The electron count checked is the Mulliken `NO. OF ELECTRONS` line (PPAN),
  not the header `N. OF ELECTRONS PER CELL`, which shows the count **before**
  DOPING (measured); the spheropole offset uses header + DOPING. A job whose
  count differs from N_pristine + dZ - q by more than 1e-4 is excluded
  ("DOPING not applied").
* |Mulliken spin - NSPIN| > 0.1 is a warning, saying whether the SCF ended
  with the lock still on or after it was released. Measured: MgO N_O q=0 with
  SPINLOCK 1 30 converged in 15 cycles, still locked, and its final Mulliken
  spin was 0.84, not 1.
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
set is not run here; with eps_inf = 2.865 its measured levels are eps(+/0) =
2.883 eV and eps(2+/+) = -0.092 eV (unrelaxed; table above). Without the
image-charge term they are 3.546 and 1.896 eV. `mu.example.json` shows the `--mu-file` format; its value is a
placeholder.

## Known limits

* Labels come from CRYSTAL's output or from you, which costs one round trip
  (shortened by reading the running pristine output).
* The spheropole offset was validated on electron removal (q > 0); the
  calibration cross-check covers only the range of its ladder.
* SPINLOCK release can let a state drift; the final spin is checked.
* No relaxation. The image-charge term is the isotropic Makov-Payne monopole
  term (Lany-Zunger scaled); anisotropic dielectrics use the average.
* Not handled: decks derived later from a ghost parent (opt2d12 / opt2d3) do
  not carry GHOSTS.
