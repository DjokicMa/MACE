# What to cite

MACE prepares, runs and analyzes calculations that other programs and
methods carry out. Cite MACE itself ([README](../README.md#citation)), and
then the work your results depend on. The list below names the references
for everything MACE runs or bundles; cite the ones you used.

## Always

- **CRYSTAL23**, which does every calculation: A. Erba, J. K. Desmarais, S.
  Casassa, B. Civalleri, L. Donà, I. J. Bush, B. Searle, L. Maschio, L.
  Edith-Daga, A. Cossard, C. Ribaldone, E. Ascrizzi, N. L. Marana, J.-P.
  Flament and B. Kirtman, CRYSTAL23: A Program for Computational Solid State
  Physics and Chemistry, J. Chem. Theory Comput. 19, 6891 (2023),
  doi:10.1021/acs.jctc.2c00958.
- **Basis sets and functionals**: the papers of the basis set family and the
  functional you chose (the pob-TZVP and pob-TZVP-rev2 references are in
  [Crystal_d12/README.md](../Crystal_d12/README.md) and
  [Crystal_d12/basis_sets/PROVENANCE.md](../Crystal_d12/basis_sets/PROVENANCE.md)).

## Band paths, symmetry and structures

- **SeeK-path** (the band and phonon k-paths of 3D cells): Y. Hinuma, G.
  Pizzi, Y. Kumagai, F. Oba and I. Tanaka, Band structure diagram paths based
  on crystallography, Comput. Mater. Sci. 128, 140 (2017),
  doi:10.1016/j.commatsci.2016.10.015.
- **spglib** (space groups in the deck generators and the GUI): A. Togo, K.
  Shinohara and I. Tanaka, Spglib: a software library for crystal symmetry
  search, Sci. Technol. Adv. Mater.: Methods 4, 2384822 (2024),
  doi:10.1080/27660400.2024.2384822.
- **ASE** (structure reading and writing): A. H. Larsen et al., The atomic
  simulation environment - a Python library for working with atoms, J. Phys.:
  Condens. Matter 29, 273002 (2017), doi:10.1088/1361-648X/aa680e.

## Phonons (`mace phonopy`)

- **phonopy**: A. Togo, L. Chaput, T. Tadano and I. Tanaka, Implementation
  strategies in phonopy and phono3py, J. Phys.: Condens. Matter 35, 353001
  (2023), doi:10.1088/1361-648X/acd831; and A. Togo and I. Tanaka, First
  principles phonon calculations in materials science, Scr. Mater. 108, 1
  (2015), doi:10.1016/j.scriptamat.2015.07.021.

## Point defects (`mace defects`)

- **Image-charge correction**: G. Makov and M. C. Payne, Periodic boundary
  conditions in ab initio calculations, Phys. Rev. B 51, 4014 (1995),
  doi:10.1103/PhysRevB.51.4014.
- **Its scaling**: S. Lany and A. Zunger, Assessment of correction methods
  for the band-gap problem and for finite-size effects in supercell defect
  calculations: Case studies for ZnO and GaAs, Phys. Rev. B 78, 235104
  (2008), doi:10.1103/PhysRevB.78.235104.

## Starting geometries (`mace preopt`, the GUI's Clean)

A pre-optimized geometry is only a starting point that CRYSTAL optimizes
again, but cite the method that made it: GFN-FF and the xtb program, or the
ML potential (MACE-MPA-0 or CHGNet) and its training data. The references
are in [preopt.md](preopt.md#citations); the CLI prints the one for the
method it ran, and the GUI's **Copy citation** copies it.

## The GUI's 3D views

- **3Dmol.js**: N. Rego and D. Koes, 3Dmol.js: molecular visualization with
  WebGL, Bioinformatics 31, 1322 (2015), doi:10.1093/bioinformatics/btu829.

The licenses of the code MACE ships (3Dmol.js, the Lucide icons, the element
colors) are listed in the [README](../README.md#license).
