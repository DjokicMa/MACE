# MACE example suite

```
mace examples                 # the table below
mace examples hydrate         # unpack the compressed files
mace gui --dir examples/suite --open
```

Inputs and CRYSTAL23 outputs for five material archetypes across the calculation types, so the GUI's viewer and plots can be tried without running anything first. Folders: `<archetype>/<system>/<calculation type>/`.

Four entries cover every calculation type, one per dimensionality: diamond (3D), h-BN (2D), polyacetylene (1D) and EC (0D). Each starts from an OPT run; SP runs at the optimized geometry and every other calculation starts from that SP (its output for the d12 decks, its wavefunction for the properties runs), with the decks MACE writes (`mace opt2d12`, `mace opt2d3`). n/a marks only what CRYSTAL23 does not run for that dimensionality, with the manual page below. The other entries are extra examples.

| Archetype | System | OPT | SP | FREQ | BAND | DOSS | TRANSPORT | CHARGE+POTENTIAL | ORBITALS | PHONON_BAND | PHONON_DOS | ELASTIC | EOS | QHA | CPHF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Porous framework (zeolite/MOF/COF-like) | SOD | output | output | output | output | output | output | output | - | - | - | - | - | - | - |
| Dense 3D crystal | diamond | output | output | output | output | output | output | output | output | output | output | output | output | output | output |
| Dense 3D crystal | C1-RCSR-bcu-f | - | - | - | - | - | output | - | - | - | - | - | - | - | - |
| 2D material / slab | hBN | output | output | output | output | output | output | output | output | output | output | output | n/a | output | output |
| 2D material / slab | graphene | output | output | n/a | output | output | output | output | - | - | - | - | - | - | - |
| 2D material / slab | 4LG_AA | output | - | - | - | output | - | - | - | - | - | - | - | - | - |
| 1D polymer | polyacetylene | output | output | output | output | output | output | output | output | output | output | output | n/a | output | output |
| 0D molecule | EC | output | output | output | n/a | n/a | n/a | output | output | n/a | output | n/a | n/a | n/a | output |

output = a real CRYSTAL23 output is included. failed run = the run's real output is included, but CRYSTAL stopped before finishing (reason below). deck only = the input deck is included and its output is added when it has been run (`mace examples` lists them; the manifest names what each one needs). n/a = CRYSTAL does not run it for this system, or gives no usable result (reason below).

## n/a

- hBN EOS, polyacetylene EOS: EOS scans the cell volume with volume-constrained optimizations (CRYSTAL23 manual, ch. 12, p.282), which CRYSTAL23 runs for 3D systems only (p.12: "Constant volume or pressure constrained geometry optimizations (3D only)"); probe runs of EOS on h-BN and polyacetylene stopped with "FOR EOS CALCULATION THE SYSTEM MUST BE 3D"
- graphene FREQ: no usable FREQ for this semimetal: without Fermi smearing its SCF oscillates (the FREQ run stopped with SCF ENDED - TOO MANY CYCLES at 200 cycles), and with smearing CRYSTAL23's analytic gradients, from which FREQCALC builds the Hessian, are not reliable for a conducting system (CRYSTAL23 manual, SMEAR, p.127). FREQ runs from smeared SPs (Broyden mixing) gave the E2g (G) mode at 235 cm-1 with SMEAR 0.005 and at -719 cm-1 with SMEAR 0.001, where graphene's is near 1580 cm-1, so no FREQ deck or output is shipped
- EC BAND: a molecule has no band structure (CRYSTAL23 manual, BAND: "does not run for molecules", p.309)
- EC DOSS: CRYSTAL computes no density of states for a molecule (CRYSTAL23 manual, DOSS: "do not run for molecules", p.320)
- EC TRANSPORT: BOLTZTRA's transport distribution function sums band velocities over the k points of the Brillouin zone (CRYSTAL23 manual, eq. 14.4, p.314), and a molecule has no bands (BAND: "does not run for molecules", p.309)
- EC PHONON_BAND: a phonon dispersion samples k points defined by a SCELPHONO supercell of a 1D, 2D or 3D periodic system (CRYSTAL23 manual, pp.232-233; SCELPHONO takes a 3D, 2D or 1D expansion matrix, p.73); CRYSTAL23 stops a MOLECULE at SCELPHONO (SUPERCELL OPTION NOT ALLOWED FOR MOLECULES)
- EC ELASTIC: elastic constants are strain derivatives along the periodic directions, and the elements of a non-periodic direction are null by definition (CRYSTAL23 manual, ch. 13, p.288): a molecule has none; a probe run of ELASTCON on EC stopped with "SYSTEM MUST BE PERIODIC FOR ELAPIEZO"
- EC EOS: EOS scans the cell volume with volume-constrained optimizations (CRYSTAL23 manual, ch. 12, p.282), which CRYSTAL23 runs for 3D systems only (p.12: "Constant volume or pressure constrained geometry optimizations (3D only)"); a molecule has no cell volume
- EC QHA: QHA fits phonon frequencies against the cell volume, relaxing the structure at each volume (CRYSTAL23 manual, sec. 9.2, p.253); it is generalized to 1D and 2D systems (p.14), not to molecules: a probe run of QHA on EC stopped with "ELASTIC OPTION NOT ALLOWED FOR MOLECULES"

## Provenance

Output files are unmodified copies of CRYSTAL23 runs by the Mendoza group (the group's own data, shared for use with MACE), of files in MACE's tests/data, or of CRYSTAL23 runs made for this suite on the MSU HPCC from the decks next to them; each file's origin is in `source`. Wavefunction (fort.9) and scratch files are not included. Input decks marked "written by mace" were generated from those outputs by MACE itself. Cube grids are decimated by 2 along each axis (by 4 for the polyacetylene grids, 100 x 518 x 404 points in CRYSTAL's run). The diamond decks embed a published pob-TZVP-type carbon basis set.

## Size

Text files over 32 KiB are stored `.xz`-compressed and unpacked in place the first time the GUI lists this folder or `mace examples hydrate` runs; `MANIFEST.json` carries each file's SHA-256. On a read-only install use `mace examples copy DEST`. A DOSS run's fort.25 repeats its DOSS.DAT in another format and is left out (4LG_AA's DOSS keeps one).

Rebuild with `python examples/gui/build_examples.py --suite --corpus /path/to/test --results /path/to/runs`: the runs folder holds the finished runs (`<archetype>/<system>/<calculation type>/`), and every deck the build writes must be the one that ran.
