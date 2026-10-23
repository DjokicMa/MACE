# Structure analysis

MACE 1.1.6 adds structure analysis tools for any structure MACE opens (CIF,
D12, CRYSTAL output, POSCAR, XYZ, fort.34, ...). Each tool works three ways:

- **Command line:** `mace analyze <tool> FILE` (add `--json` for machine-readable output).
  The structure tools take several files, `mace analyze porosity *.cif --csv summary.csv`:
  one line per structure, and one CSV row per structure (or `--json`, a list). A file that
  fails is reported and the others go on (exit code 1).
- **GUI:** Analysis › Structure Analysis, a panel with one tab per tool. Results
  for pores, crystal shapes, Miller planes and nets are drawn in the 3D view.
  Explorer › Analyze Structures… runs a tool on many files, with the same
  summary in the bottom panel's Analysis tab.
- **Python:** `mace.analysis.<module>` (the functions below).

Only numpy, scipy, spglib and ase are needed. Zeo++ is used if it is installed.

| Tool | Command | GUI tab |
|---|---|---|
| Pore sizes, volumes, surface areas | `mace analyze porosity FILE` | Porosity |
| Simulated powder pattern | `mace analyze pxrd FILE` | Powder Pattern |
| Peaks of a measured pattern | `mace analyze peaks PATTERN` | Peaks & Fit |
| Indexing | `mace analyze index PATTERN` | Peaks & Fit |
| Pawley / Rietveld fit | `mace analyze refine PATTERN --structure FILE` | Peaks & Fit |
| Underlying net, RCSR symbol, interpenetration | `mace analyze topology FILE` | Topology |
| Reference nets (RCSR, bundled) | `mace analyze rcsr status` / `import 3dall.txt` | Topology |
| Coordination polyhedra | `mace analyze polyhedra FILE` | Polyhedra |
| Bond and angle distributions, g(r) | `mace analyze bonds FILE [--rdf]` | Bonds & Angles |
| Crystal morphology (BFDH, Wulff) | `mace analyze morphology FILE` | Morphology |
| Miller planes, cleavage ranking | `mace analyze planes FILE --hkl h k l` / `--cleavage` | Planes |
| Is Zeo++ installed? | `mace analyze zeopp` | Porosity › Advanced |

## Porosity

```
mace analyze porosity HKUST-1.cif                 # N2 probe, CCDC radii
mace analyze porosity HKUST-1.cif --probe CO2
mace analyze porosity HKUST-1.cif --probe all     # every preset adsorbate
mace analyze porosity HKUST-1.cif --probe 1.5 --radii uff
mace analyze porosity HKUST-1.cif --engine both   # native and Zeo++ side by side
mace analyze porosity HKUST-1.cif --cube pores    # pores_accessible.cube, pores_inaccessible.cube
```

**Adsorbate presets** (probe radius = half the kinetic diameter, Breck 1974):
He 1.30 Å, H2 1.445 Å, H2O 1.325 Å, CO2 1.65 Å, Ar 1.70 Å, N2 1.82 Å, CH4 1.90 Å.
Any radius from 0 to 10 Å can be given instead.

**Atomic radii:** `ccdc` (default; Bondi's van der Waals radii as the CCDC uses
them, which are also Zeo++'s defaults) or `uff` (σ/2 of the Universal Force
Field). `--radius Zn=1.39` overrides one element.

**What is reported** (the Zeo++ names):

| Quantity | Meaning |
|---|---|
| Di | Largest included sphere |
| Df | Largest free sphere: the largest sphere that can travel through the crystal (also per axis a, b, c) |
| Dif | Largest included sphere on the channels a sphere of size Df can travel (dead ends left out) |
| AV / NAV | Accessible / non-accessible volume of the probe *center* (Zeo++ `-vol`) |
| POAV / PONAV | Volume the probe *sphere* can occupy, accessible / not (Zeo++ `-volpo`) |
| ASA / NASA | Accessible / non-accessible surface area (Zeo++ `-sa`), Å², m²/cm³, m²/g |
| Geometric void fraction | Space outside every atom (no probe) |
| Density | From the cell and its atoms |
| Channels / pockets | Channel systems and their dimensionality; closed pockets |
| Per channel and pocket (`features` in `--json`) | Each one: accessible or closed, dimensionality, probe-center and probe-occupiable volume, surface area, its own largest included sphere and where it sits, centroid |

**How it works (native engine).** On a grid over the cell (0.15 Å by default;
coarser for very large cells), MACE computes the exact distance from each grid
point to the nearest atom *surface*, periodic images included. A Voronoi
decomposition of the atom centers gives the network of paths with the most
clearance. The largest free sphere and the channels a probe can travel come
from that network. A union-find search tracks the lattice vector each
connection crosses, so a pore that connects to its own periodic image is a
channel. One that does not is a closed pocket, which the probe cannot reach.
Volumes are counted on the grid. Surface areas sample each atom's sphere of
radius R + r_probe. The probe-occupiable volume also samples where two spheres
meet (crevices), so it converges as the grid gets finer.
Along a non-periodic direction (a slab or a molecule) a pore that reaches the
open boundary is accessible; for those systems the volumes, fractions, Di and
Df include the vacuum. The box along those directions holds the atoms with at
least 5 Å of vacuum on each side: a CRYSTAL slab, polymer or molecule (a
0.5 Å dummy vector there) gets a box built around its atoms.

**Validation** (CCDC radii, N2 probe; the SOD and HKUST-1 rows are in
tests/test_analysis_structure_tools.py, the others were checked during
development against Zeo++ 0.3 built from source):

| Structure | Quantity | MACE | Zeo++ 0.3 (`-ha`) |
|---|---|---|---|
| Sodalite (SOD) | Di / Df | 5.823 / 2.132 Å | 5.821 / 2.132 Å |
| LTA | Di / Df / AV fraction | 10.266 / 3.814 / 0.110 | 10.238 / 3.814 / 0.111 |
| FAU | Di / Df / AV fraction | 10.702 / 6.951 / 0.175 | 10.696 / 6.951 / 0.175 |
| CHA | Di / Df | 6.738 / 3.319 Å | 6.741 / 3.319 Å |
| MFI | Di / Df | 5.938 / 4.264 Å | 5.942 / 4.299 Å |
| HKUST-1 (COD 4002052) | Di / Df / AV fraction / ASA | 13.287 / 6.654 / 0.268 / 1909 m²/cm³ | 13.281 / 6.701 / 0.266 / 1894 m²/cm³ |

Dense crystals (Si, NaCl, MgO) give zero accessible volume for N2. Surface
areas agree with Zeo++ within about 5% (LTA, N2: 1119 against 1073 m²/cm³) and
with a dense sampling of the exposed sphere area within 1% (MFI: 735 against
730 m²/cm³; Zeo++ 720). The accessible and closed cages agree with Zeo++ cage
by cage: LTA's β cages are closed to He and N2 and open from a probe radius
between 0.6 and 0.7 Å in both. With the IZA database's convention (oxygen
only, r = 1.32 Å) the native engine gives the database's included and free
spheres to 0.01 Å (LTA 11.05 / 4.21, FAU 11.24 / 7.35, CHA 7.37 / 3.72, MFI
6.36 / 4.70, 4.46, 4.46 Å). Two quantities differ by design:

- POAV/PONAV are 2-11% larger than Zeo++'s `-volpo`. MACE counts every
  point a probe sphere can reach, crevices included. A Monte Carlo check of
  that definition, each point tested exactly against the union of spheres,
  matches MACE to 1% (MFI, N2: 0.286 ± 0.003 exact, MACE 0.284, Zeo++
  0.257).
- Dif is the largest sphere anywhere on the channels a sphere of size Df can
  travel. Zeo++ reports it along one path it picks, so its value can be
  smaller (HKUST-1: 13.29 against 11.13 Å).

For unequal radii, MACE moves each window's narrowest point to the center of
the window. Zeo++ subdivides spheres instead (`-ha`). The two Df values then
agree within about 0.05 Å.

**Zeo++** (optional; Willems et al., Microporous Mesoporous Mater. 149, 134
(2012); BSD-style license from LBNL). MACE never installs or bundles it.
Build it from <http://www.zeoplusplus.org> (`make` in `voro++/src`, then
`make`). Then put `network` on PATH or set `MACE_ZEOPP=/path/to/network`.
MACE writes a P1 `.cssr` file and passes its own radii with `-r`, so both
engines use the same atoms. `--engine zeopp` runs Zeo++ only, and
`--engine both` shows the two side by side.

**Per channel and pocket.** Every center grid point belongs to one channel
system or one pocket (a grid piece no network node reached joins the nearest
one of its class), and so does every point of the occupiable shell and of
the surface sampling. The channels' volumes and areas therefore add up to AV,
POAV and ASA exactly, and the pockets' to NAV, PONAV and NASA. Each one's
largest included sphere is refined off the grid like Di; the largest of them
is Di when Di's sphere lies in a pore the probe fits.

**In the GUI**, Compute Pores draws the probe-occupiable volume as two
isosurfaces: bluish green for pores the adsorbate can reach and orange for
closed pockets. You can hide either one, set the opacity, draw them as a
wireframe, with their edges or as points, and pick their mesh quality (Draft
for screening to Publication for a figure; GUI guide, Mesh quality). The
surfaces stay through display changes, belong to the document they
were computed for, and go away with Clear 3D Overlays. The bottom panel's
Analysis tab lists every channel and pocket with totals and averages (a click
highlights and frames one), and Inspector › 3D Overlays changes one
structure's surfaces (each channel or pocket on its own too).

## Powder diffraction

```
mace analyze pxrd Si.cif                          # Cu K-alpha, 5-90 degrees
mace analyze pxrd Si.cif --source MoKa --range 5 40 --out si.xy --plot si.png
mace analyze pxrd Si.cif --b-iso 0.5 --W 0.004 --size-nm 50
```

The pattern is kinematic, for Bragg-Brentano geometry. It uses Waasmaier-Kirfel
atomic form factors and the Lorentz-polarization factor (with a monochromator
angle if set), plus an isotropic Debye-Waller factor. Every reflection is
enumerated, so multiplicities are implicit, and lines at the same 2θ merge.
Miller indices refer to the standard conventional cell. Profiles are
pseudo-Voigt with Caglioti widths and optional Scherrer size broadening; the
Kα2 line is added for sources that have one (Cu, Mo, Co, Cr, Fe, Ag; or give
any wavelength in Å). Anomalous dispersion is not included.
Validation: Si and NaCl peak positions match pymatgen's XRDCalculator to
0.001°, and relative intensities to about 1%.

**Measured patterns:** two- or three-column text files (`.xy`, `.xye`, `.dat`,
`.txt`, `.csv`: 2θ, intensity, optional σ; comment and header lines are
skipped) and PANalytical `.xrdml`. In the GUI, pick one from this computer or
give a path in the served folder.

```
mace analyze peaks sample.xy                       # background (SNIP), peak search, pseudo-Voigt fits
mace analyze index sample.xy                       # trial cells, ranked by M20
mace analyze index sample.xy --structure model.cif # index with a known cell, refine it
mace analyze refine sample.xy --structure model.cif               # Pawley
mace analyze refine sample.xy --structure model.cif --method rietveld --positions --po 0 0 1
```

**Indexing from scratch** tries the cubic, tetragonal, hexagonal and
orthorhombic systems with every lattice centering. It ranks the trial cells by
de Wolff's M20. Kα2 satellites are removed from the peak list first.
Monoclinic and triclinic cells are not searched. For those, index against a
structure model's cell.

**Pawley and Rietveld.** The lattice is refined as a symmetry-allowed strain of
the starting cell. A cubic cell has 1 free parameter, tetragonal and
hexagonal 2, orthorhombic 3, monoclinic 4 and triclinic 6. Use `--triclinic`
to free them all. Also refined: the zero shift, the Caglioti U, V, W, the
pseudo-Voigt η and a Chebyshev background. Pawley gives each reflection family
its own intensity and leaves out the space group's systematic absences.
Rietveld takes intensities from the structure: a scale, an overall B, atomic
coordinates along the directions their site symmetry allows (`--positions`),
and March-Dollase preferred orientation (`--po h k l`). The linear parameters
are solved exactly at each step, so only the cell, profile and structure
parameters are searched. The fit reports Rp, Rwp, Rexp and GOF, with standard
uncertainties on the lattice parameters.
Validation: synthetic patterns with Poisson noise recover Si a = 5.431 Å and
a 0.03° zero shift (Pawley), and rutile a, c, B and the oxygen x = 0.3048 from
a start of 0.295 (Rietveld), with GOF ≈ 1.03.
Not modeled: peak asymmetry, sample displacement, absorption, anisotropic
broadening, occupancies and anisotropic displacement parameters. A dedicated
program (GSAS-II, FullProf, TOPAS) remains the tool for publication-grade
refinements.

## Topology and interpenetration

```
mace analyze topology HKUST-1.cif                  # tbo
mace analyze topology zeolite.cif --mode atoms
mace analyze rcsr status
mace analyze rcsr import 3dall.txt                 # only for a newer RCSR list
```

**Simplification:**

- `atoms`: terminal atoms are removed and 2-connected atoms become edges,
  repeatedly. A zeolite becomes its T-atom net.
- `sbu`, for metal-organic frameworks: metals plus the oxo, hydroxo and halide
  atoms bridging them form the inorganic building units. Organic linkers
  bonded to three or more units become vertices; linkers bonded to two become
  edges. An infinite (rod) unit falls back to one vertex per metal.
- `auto` picks `sbu` when the structure has metals and carbon, else `atoms`.

Molecules not bonded to the framework (solvent, guests) are left out.
Metal-metal contacts count as bonds in metals and building units, but not in
an inorganic compound, so CsCl is the bcu net of Cs-Cl bonds.

**Identification** compares each vertex's coordination sequence (10 shells)
with the reference nets. Topological density td10 and point symbols are
reported too. Equal coordination sequences are necessary for two nets to be
the same and nearly always sufficient, but not proof. When two reference nets
share them (lta and rho, for example), both are listed.

**Interpenetration:** each connected piece of the net repeats along the lattice
spanned by its cycles. The index of that lattice is the number of disjoint
copies it splits into, so Cu2O reads as two interpenetrating dia nets. A piece
of lower dimensionality (layers, chains) is reported as such, without
classifying how layers interweave.

**Reference nets.** MACE ships RCSR's list of 3645 3-periodic nets
(`mace/analysis/rcsr_data/3dall.txt.xz`: <http://rcsr.net/data/3dall.txt>,
downloaded 2026-10-22, unmodified), so FAU, LTA and CHA read as fau, lta and
cha with no setup. RCSR states no license or terms of use for its downloads;
MACE redistributes the file with credit, and it is RCSR's work, not covered
by MACE's MIT License. Source, checksum and credits are in
`mace/analysis/rcsr_data/NOTICE.txt`. Built in besides are 21 common nets
whose coordination sequences MACE computes itself from textbook embeddings
(the diamond, NaCl, CsCl, Cu, NbO, PtS, SrSi2, sodalite, quartz positions,
...), which also give the 2-periodic layers:

`pcu bcu fcu dia srs nbo sod pts lon qtz ftw she scu ths tbo pto acs hcb sql kgm hxl`

To use a newer RCSR list, download <http://rcsr.net/data/3dall.txt> and run
`mace analyze rcsr import 3dall.txt`, or use Import on the GUI's Topology tab
with a copy in the served folder. A Systre `.cgd` file such as RCSRnets.cgd
works too; MACE computes its coordination sequences, which takes longer. The
imported list is stored in `$XDG_CONFIG_HOME/mace/rcsr_nets.json`
(`MACE_CONFIG_DIR` overrides the folder) and is tried before the bundled one,
so a net it renames or adds wins. `mace analyze rcsr status` lists what is
loaded.

**Citing RCSR.** The net symbols and reference data are RCSR's. When you
report a net from MACE, cite M. O'Keeffe, M. A. Peskov, S. J. Ramsden and
O. M. Yaghi, The Reticular Chemistry Structure Resource (RCSR) Database of,
and Symbols for, Crystal Nets, Acc. Chem. Res. 41, 1782-1789 (2008),
[doi:10.1021/ar800124u](https://doi.org/10.1021/ar800124u). The
`mace analyze topology` output and the GUI's Topology result print this
citation, and the `--json` result carries it as `citation`.

## VESTA and Materials Studio tools

**Coordination polyhedra** (`mace analyze polyhedra FILE`): for each polyhedron
center (the viewer's Polyhedral style rule: a cation with 4-12 anion ligands,
a perovskite's A site left out), the coordination number and bond lengths to
those ligands are reported, with
the polyhedral volume and Baur's distortion index. Quadratic elongation and
bond-angle variance are given for tetrahedra and octahedra. Hoppe's ECoN and
the closest ideal shape by continuous shape measure (CShM) are reported too.
The GUI tab also toggles the polyhedra in the 3D view.

**Bond and angle distributions** (`mace analyze bonds FILE`): every bond by
element pair and every X-A-Y angle, with statistics and histograms. The bonds
are the ones the GUI draws at its default criteria: closer than `--tolerance`
(1.15) × the sum of the bond radii, metal-metal pairs only in a structure of
metals only; topology, polyhedra and planes use the same rule.
`--rdf [--pair Si-O]` gives g(r) instead.

**Morphology** (`mace analyze morphology FILE`): BFDH from the structure (a
face's distance is proportional to 1/d of the first allowed reflection), or
Wulff from surface energies you give (`--energies "100:1.2, 111:0.9"`). Each
family is expanded by the Laue group. The table gives each family's share of
the surface; `--obj` writes the shape, and the GUI draws it at the cell
center. Attachment-energy morphology (which needs a force field) is not
included.

**Miller planes and cleavage** (`mace analyze planes FILE --hkl 1 1 1 --offset 0.5`):
the plane's spacing, the polygon where it cuts the cell (drawn in the GUI, with
a slider for its position), and the atoms on it. `--cleavage` ranks the
low-index families by the fewest bonds broken per nm² at the best cut. This is
a bond-counting estimate with no relaxation; it puts Si {111} and NaCl {100}
first.

## Python

```python
from mace.analysis import structure, porosity, pxrd, refine, indexing, topology, polyhedra, morphology, planes
st = structure.load("HKUST-1.cif")              # or structure.from_atoms(ase_atoms)
porosity.analyze(st, probe="N2", render=False)  # dict as in the tables above
pxrd.simulate(st, "CuKa", (5, 50))
refine.fit(pxrd.read_pattern("sample.xy"), "rietveld", structure=st, refine_positions=True)
indexing.index_ab_initio([28.44, 47.30, 56.12, 69.13, 76.38], "CuKa1")
topology.analyze(st)["summary"]                  # {'topologies': ['tbo'], 'interpenetration': 1, ...}
```
