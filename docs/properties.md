# Properties: what the database stores and what the GUI shows

MACE reads properties from CRYSTAL outputs into `materials.db`
(`mace analyze --extract-properties`, the completion callback of a tracked
job, the GUI's **Create or Update Database…**), and the GUI computes more
from the same runs. This page lists both: every key the database stores, by
calculation type, with its unit, where it comes from and its database key,
and what the GUI computes or shows that the database does not have yet.

```bash
mace analyze --list-properties                    # every key, its unit, calculation types and source
mace analyze --list-properties --calc-type FREQ   # one calculation type
mace analyze --list-properties --derived          # the derived properties only
mace analyze --list-properties --gui              # the GUI's values, and the gaps
mace analyze --list-properties --json             # the same as JSON
```

## Two kinds of property

- **Stored by the extractor**: read off the run's `.out` and the files beside
  it (BAND.DAT, DOSS.DAT, SEEBECK.dat, ...) by `mace/utils/property_extractor.py`.
  Each key, its category, unit and value type are in
  `mace/utils/property_keys.py`.
- **Derived**: computed from what the extractor read, the run's sibling files
  and other runs, by a function registered in `mace/properties/`
  (one module per area in `mace/properties/defs/`). The extractor runs every
  derived property that applies after its own extraction and stores the
  values with the run's other rows. The run's `extraction_provenance` row says
  which derived properties ran, at which version, from which files, and why
  the others were skipped. A derived property that fails (status `error`)
  keeps the values an earlier extraction stored.

### Turning derived properties off

- `--no-derived` (or `MACE_DERIVED=0`): no derived property is computed;
  the stored ones stay.
- `--skip-property NAME`, repeatable (or `MACE_SKIP_PROPERTIES=NAME,NAME`):
  that property is not computed; its stored rows stay. `mace analyze
  --list-properties --derived` lists the names.
- A job's completion callback (`mace queue`) and the MCP results tool skip
  the slow ones, `bader_charges` (a Bader partition of the density cube) and
  `packing_fraction` (a distance grid); `mace analyze --extract-properties`
  computes them, and `MACE_DERIVED_EXPENSIVE=1` turns them on there too.
- Derived properties use at most `MACE_DERIVED_THREADS` BLAS/OpenMP threads
  (default: half the cores, at most 4).

A derived property marked **interactive** is not computed on extraction: it
depends on choices made in the GUI (a fit, a strain series). The GUI computes
it on request, and its **Save to Database** stores it. Those rows are written
by `mace-gui` (the `extractor_script` column) and a re-extraction of the run
keeps them; the extractor replaces only its own rows.

## The Properties tab (GUI)

The bottom panel's **Properties** tab shows, for the run of the document in
focus (an output, or a plot, spectrum, volume or deck a run wrote),
everything the database has for that run: grouped by calculation type and
group, with units, and where each value came from (the file, the
extractor or `mace-gui`, the derived property's version, when it was
extracted).

- **Which database**: the open Database document's, when it has the run;
  else the nearest `materials.db` from the run's folder up to the served
  folder; else the folder's workflow database (`.mace_context_*`).
- **Not in the database yet: Extract** extracts that one run into the
  database (`mace analyze --extract-properties --output-file RUN --db-path DB
  --register --register-output-only --backup`, the command shown): it
  registers that run only (not the rest of its folder), backs up the database
  first, and replaces only this run's rows.
- **No database**: the tab computes the same values from the run on the fly
  (the extractor and the registry, nothing written) and says so;
  **Create Database…** opens Create or Update Database from a Folder.
- **Copy**, **CSV** and **JSON**, as in the Analysis tab: the rows shown, with
  labels in plain text (`m_c`, not markup), units, values and provenance.
  Each value is exported in the unit shown (`value`, `unit`, `normalization`),
  with the stored value and unit beside it (`stored_value`, `stored_unit`) and
  the other keys a merged row stands for (`also`).

### Units

The tab can show every value with a physical unit in another unit of its
dimension. This is display only: the database keeps each key, value and
unit as the extraction wrote them.

- **Units** (top of the tab) sets every row at once: **As stored**,
  **Hartree atomic units** (Hartree, Bohr, Hartree/Bohr³, e·Bohr, ...),
  **eV/Å (solid state)** (eV, Å, GPa, e·Å, THz, J/m²) or **kJ/mol
  (chemistry)** (kJ/mol, Å, GPa, D, cm⁻¹, J/(mol·K); energy levels such as a
  gap, a band edge or E<sub>F</sub> stay in eV). A dimension a preset does not
  name keeps its stored unit.
- **Normalize** sets the per-cell (extensive) values: **Total (per cell)**,
  **Per atom** (atoms_in_unit_cell) or **Per formula unit** (the reduced
  formula of the cell's atoms and its Z; diamond's two-atom cell is C, Z = 2).
  An intensive value (a gap, an energy per atom, a formation energy per atom,
  an adsorption or reaction energy) is never divided. Which keys are
  extensive is declared, not guessed: `property_keys.EXTENSIVE` for the
  extractor's keys, `Output(extensive=True)` for a derived property's.
- **A row's Unit cell** is a drop-down of the units that row can take (with
  per atom and per formula unit for an extensive one). Picking one changes
  that row's value and unit only, and wins over the top until **Units**
  changes again, which sets every row afresh. **Normalize** rescales every
  extensive row, those with a unit of their own too, and keeps that unit
  (Hartree becomes Hartree/atom). The choice is kept in this browser
  (`localStorage` key `mace.gui.propunits.v1`).
- **Digits.** Energies are shown to CRYSTAL's printed precision: 10 decimals
  in Hartree, the same resolution (10⁻¹⁰ Hartree) in any other unit (9
  decimals in eV, 7 in kJ/mol, 5 in cm⁻¹). This covers the energies of the
  system (total energies and their parts, the thermodynamics, the reference
  energies, the energy per atom) and every energy stored in Hartree
  (E<sub>F</sub>, the orbital energies); a level CRYSTAL prints in eV (a gap,
  a band edge) and every other value keep their significant digits. The
  server's value text and the GUI's converted text are formatted the same way.
- **`energy_per_atom_ev`** (E<sub>total</sub>/atom, the run card's) is the SCF
  energy per atom *without* the D3 and gCP corrections; the corrected energy
  per atom is `total_energy_corrected_au` with **Normalize › Per atom**.
- **One row for one quantity.** A quantity stored under several keys that
  differ only by unit (E<sub>D3</sub> as `d3_dispersion_energy_au` and
  `d3_dispersion_energy_ev`, the FREQ thermodynamics in Hartree, eV and
  kJ/mol, a surface energy in J/m² and eV/Å²) is one row, under its most
  precise key (the Hartree value CRYSTAL prints); the other keys are shown
  as `+1`/`+2` beside it and the filter finds them. Its value in another unit
  is converted from that key. Every key stays in the database. The groups are
  declared (`property_keys.SAME_QUANTITY`, `Output(same_as=...)`), and a test
  finds any key that looks like another one in a different unit and is not
  declared.

The conversion factors live in `mace/properties/units.py`, derived from
`mace/constants.py` (the extractor's Hartree-to-eV and Bohr-to-Å values and the
exact SI 2019 constants of CODATA 2018); the GUI receives them with the rows,
and `mace analyze --list-properties --units` prints the same table (`--json`
for the factors). The units per dimension are in [Display units](#display-units)
below.

## Adding a derived property

One module in `mace/properties/defs/` (every module there is loaded; no other
file needs editing), then `mace analyze --list-properties --update-doc
docs/properties.md`:

```python
from mace.properties.registry import Output, Sibling, Reference, Skip, derived

@derived("pugh_ratio", calc_types=("ELASTIC",), category="mechanical", group="Mechanical",
         requires=("bulk_modulus_hill_gpa", "shear_modulus_hill_gpa"), version="1",
         outputs=(Output("pugh_ratio", "dimensionless", label="B/G", precision=4,
                         description="Pugh's ratio B_H/G_H (ductile above 1.75)"),),
         source="bulk_modulus_hill_gpa / shear_modulus_hill_gpa")
def _pugh(ctx):
    """Pugh's ratio."""
    g = ctx.value("shear_modulus_hill_gpa")
    if not g:
        raise Skip("no shear modulus")
    return {"pugh_ratio": ctx.value("bulk_modulus_hill_gpa") / g}
```

- `outputs`: the database keys it writes. `unit` and `type` follow
  `property_keys.py` (`float`, `int`, `bool` with unit `boolean`, `text`
  with unit `category`, `json`); a curve or array is `json`, and `blob=True`
  keeps it in the `property_blobs` table. `label` may carry plotly markup
  (`m<sub>c</sub>`, `Å<sup>3</sup>`); CSV and the terminal show it plain
  (`m_c`). A key the extractor itself writes cannot be an output.
- `calc_types`: the calculation types it applies to (`"*"`: every run).
- `requires`: keys the extractor must have read (else it is skipped, and
  the provenance row says which were missing).
- `siblings`: files beside the run, `Sibling("BAND")` and `Sibling("DOSS")`
  (the run's `<stem>.BAND.DAT`, or the BAND/DOSS run of the same parent),
  `Sibling("d12")`, or any file with `find=lambda output_file: path`;
  `required=False` makes one optional. `ctx.sibling("BAND")` is its path.
- `references`: other runs. `Reference("bulk", calc_types=("OPT", "SP"),
  find=lookup)`: the run the user gives (`--reference bulk=FILE|CALC_ID`;
  the GUI's picker), else `lookup(ctx, conn)` in the database (`conn` a
  read-only sqlite3 connection, or None without one; return a list of
  `RunRef(calc_id=..., via="database")`, for example
  `registry.db_runs(conn, calc_types=("OPT",), formula="MgO")`, the newest
  first); `many=True` for several runs (the
  elements of a formation energy). `ctx.reference("bulk").value("total_energy_au")`
  reads its stored rows (or, for a file, the extractor's values of it). A
  required reference that is not found skips the property ("needs the
  reference run bulk").
- `interactive=True`: computed by the GUI (`registry.compute_interactive`)
  and stored by its Save to Database (`registry.store_interactive`), not by
  the extractor.
- Raise `Skip("why")` when it does not apply; any other error is reported in
  the provenance row and never stops an extraction.
- Bump `version` when the computation changes: the provenance row records it.

The parity check (`mace/properties/parity.py`, tests/test_property_parity.py)
compares, for every run of the example suite, what the GUI shows with what
the database holds after extraction; a property the GUI computes is added to
its producers there.

## Electronic properties of BAND, DOSS and ORBITALS runs

Derived from the files the GUI's plots and viewers read, by the same readers
(`mace/properties/defs/band_edges.py`, `dos_analysis.py`,
`orbital_composition.py`; each module's docstring has the definitions):

- **BAND** (the BAND.DAT through `mace.transport.mass.read_band`, mace tau's
  reader; values along the computed path): direct or indirect gap with the k
  points of the VBM, CBM and smallest vertical gap (path labels such as `Γ`,
  else `0.71 of Γ–X`), the gap of each spin channel, the valence band width
  and the dispersion of the edge bands, the alpha − beta splitting of the band
  edges, and the Fermi velocity at every E_F crossing of a metal or semimetal.
  A gapless run is a metal or a semimetal by its DOS at E_F (below).
  A spin-polarized SCF that ends with its spins still locked (SPINLOCK, MACE's
  default with SPIN for a closed-shell system) computes no Fermi level: CRYSTAL
  keeps -1.0 Hartree in the wavefunction, and a BAND run without NEWK writes
  its BAND.DAT and fort.25 as E − (−1.0). Every reader of band energies then
  takes the Fermi level `mace.utils.fermi_level` derives, the top of the
  occupied bands (the electron count split by the locked α − β), and
  `fermi_energy_note` (the band plot's note, `mace plotting`'s line) says so;
  a run whose occupied and empty bands overlap gets no number.
- **DOSS** (the DOSS.DAT through `mace/gui/results.dos_data`, the DOS plot's
  reader, projection labels from the deck): Sommerfeld γ from N(E_F), the
  d-band center (two conventions, below), filling and width of transition-metal
  d projections, the orbital/atom character of the band edges (of E_F in a
  metal), carrier concentrations against the Fermi level at 300 K with the
  intrinsic level and n_i, and rough density-of-states effective masses
  (`mace.transport.mass.dos_mass`).
- **ORBITALS** (the Molden files through `mace/orbitals`, the orbital
  viewer's reader): the Mulliken composition of the HOMO and LUMO by atom
  shell and element, per k point and spin channel.

Spin-polarized runs: the details (`band_edges`, `d_band`,
`band_edge_composition`, `orbital_composition`) keep each spin channel; the
scalars are of both spins, plus the per-channel gaps and splittings.

### Effective masses: which one to use

Three masses are stored, and they give different numbers. Use the first.

| Shown as | Keys | Run | What it is |
|---|---|---|---|
| Effective mass m_c, m_d (accurate) | `effective_mass_{electron,hole}_{mc,md}` | BAND | **Accurate, the one to use.** mace tau's fit (`mace/transport/mass.py`): the conductivity mass m_c and the density-of-states mass m_d, over every degenerate band and equivalent valley at the edge, along several directions, inside an energy window, with an R² check |
| Effective mass, quick estimate (one band along the path) | `real_{electron,hole,average}_effective_mass` | BAND | **Quick estimate.** The extractor's curvature of one band along the k-path direction at the edge (`AdvancedElectronicAnalyzer`); misses the other directions, degenerate bands and valleys |
| DOS mass m_d (rough, from the smoothed DOS) | `dos_effective_mass_{electron,hole}` | DOSS | **Rough.** A fit to CRYSTAL's DOS near the edge, which is smoothed (NPOL Legendre terms) and sampled on the DOSS energy grid: both move it |

The Properties tab shows the first two in one group, "Effective masses".

### d-band center: two conventions

Both are the first moment of the transition metal's d projected DOS, from
E_F (Hammer–Nørskov); the literature uses either window:

- `d_band_center_ev`, "d-band center (to E_F + 10 eV)": occupied and empty d
  states, up to E_F + 10 eV. The cutoff keeps the basis set's diffuse and
  polarization d functions out: over the full DOSS range Ag₄Cl₂S's center is
  +3.8 eV (17 of its 49 Ag d states lie above E_F + 10 eV). The filling and
  width are over this window too. Cu(I) in PbCuSeCl: −2.46 eV, filling 0.98;
  Ti(IV) in Ti₂O₄Pb₃Se₃: +2.59 eV, filling 0.16.
- `d_band_center_occupied_ev`, "d-band center (occupied)": the occupied
  part only, up to E_F (PbCuSeCl −2.65 eV; Ti₂O₄Pb₃Se₃ −4.3 eV, the Ti d share
  of the O/Se p valence band of a d⁰ ion).

### Metal or semimetal

A BAND run whose gap along the path is below 0.01 eV, or whose bands
overlap, is gapless. Its density of states at E_F decides what it is:
N(E_F), the total DOS of both spins averaged over E_F ± 0.05 eV per atom of
the cell (states/eV/atom), from the DOSS run of the same parent (its
`<base>_doss.DOSS.DAT` in the run's folder or in a folder beside it, such as
`step_006_DOSS/` or `DOSS/`).

- N(E_F) < 0.05 states/eV/atom: **semimetal**; otherwise **metal**.
- No DOSS run: **metal or semimetal (no DOS to tell)**; MACE does not guess.

`band_path_gap_type` holds the class, `band_path_fermi_dos_per_atom` the
N(E_F) used and `band_path_semimetal_threshold` the threshold. The extractor's
`electronic_classification` of the run takes the same value and threshold
(one code path: `property_keys.classify_gapless`,
`advanced_electronic_analyzer.fermi_dos_per_atom`); without a DOSS run it
stays "metal", its rule from the gap alone.

Why 0.05: experiment (Sommerfeld γ / 2.357) puts the semimetals Bi, graphite
and Sb at 0.003, 0.006 and 0.05 states/eV/atom and the simple metals Cu, Al
and Na at 0.29, 0.57 and 0.59. In the test corpus, graphene and four-layer
graphene give 0.0002–0.006, FSI-doped four-layer graphene 0.013–0.031 and the
3,4²T10-CA carbon net 0.023 (all semimetal: graphene bands with E_F moved
off the Dirac point), the 3,4²T1-CA net 0.085 and bcc Fe 0.89 (metal). The
threshold sits in the widest gap of these values. The band analysis used
to compare the DOS at one grid point with 5% of the mean DOS of the
whole DOSS window, a scale set by the window and the basis set rather than the
material, and so split two nearly identical FSI-doped graphenes into a metal
and a semimetal.

Not derived, and why:

- **N(E_F)**: already stored by the extractor (`dos_at_fermi_total`, both
  spins, states/eV/cell; `dos_at_fermi_alpha`, `dos_at_fermi_beta`).
- **Rashba-type splitting**: no two-component (SOC) BAND output exists in the
  test corpus or the example suite to validate a fit on.
- **Conduction band width**: the empty bands of a Gaussian basis run on into
  high virtual states without a gap (159 eV for 3,4^2T11-CA), so the
  conduction manifold has no basis-independent top; the CBM band's dispersion
  is stored, the manifold stays in `band_edges`.
- **Fermi velocity of a gapped run, spin splitting of a closed-shell run,
  carriers and DOS masses of a metal**: not defined (skipped, with the reason
  in the provenance row).

## Reference energies

Five derived properties combine the run with other runs
(`mace/properties/defs/reference_energies.py`, the shared code in
`mace/properties/reference.py`). Each applies to OPT, SP, FREQ and CPHF runs.

| Property | Formula | ΔE keys (ΔH, ΔG: `enthalpy`, `gibbs_energy` for `energy`) | References |
|---|---|---|---|
| Formation energy | ΔE_f = [E(run) − Σ_i n_i E(ref_i)/N(ref_i)] / N(run) | `formation_energy_ev_per_atom` (eV/atom) | `elements`: one run per element |
| Cohesive energy | E_coh = [Σ_i n_i E(atom_i) − E(run)] / N(run), positive when bound | `cohesive_energy_ev_per_atom` (eV/atom) | `atoms`: one isolated-atom run per element |
| Surface energy | γ = [E(slab) − n E(bulk)] / 2A; n bulk cells in the slab, A the slab's cell area | `surface_energy_j_m2` (J/m²), `surface_energy_ev_ang2` (eV/Å²) | `bulk`: a 3D run of the slab's composition |
| Adsorption energy | E_ads = [E(complex) − E(surface) − k E(adsorbate)] / k, negative when it binds | `adsorption_energy_ev` (eV), `adsorption_energy_kj_mol` (kJ/mol) | `surface` and `adsorbate` |
| Reaction energy | ΔE_r = Σ_products c E − Σ_reactants c E; the run is a product | `reaction_energy_ev` (eV), `reaction_energy_kj_mol` (kJ/mol) | `reactants` (and `products`) |

- **Energies**: each run's `total_energy_corrected_au` (with D3 and gCP),
  else `total_energy_au` (a run with neither correction). E is per cell
  (a molecule: per molecule), so n_i, N, n, k and the reaction's
  coefficients count the atoms and cells of the runs as computed;
  1 Hartree = 27.211386 eV, 1 eV = 96.485 kJ/mol.
- **ΔH and ΔG** are stored when every run has FREQ thermodynamics: its own
  (a FREQ run) or a FREQ run of the same material in the database that may
  be combined with it. They use each run's H = G + TS and G (CRYSTAL's
  harmonic thermodynamics at the FREQ temperature and pressure, which must be
  the same for all). A run without its own takes only the thermal correction
  of the other FREQ run (G − EL and H − EL, EL its electronic energy), scaled
  by the ratio of their atoms and added to its own energy, so a supercell's
  FREQ run serves its primitive cell; a FREQ run whose composition is not a
  multiple of the run's is not used. A phonon dispersion or DOS run on a
  supercell (SCELPHONO) prints its thermodynamics per input cell: they are
  scaled to the supercell its energy and atoms are of, as its cell is (the
  input cell times the expansion's determinant).
- **Normalization**: formation and cohesive energies are per atom, the
  surface energy per area and the adsorption energy per adsorbate; the
  reaction energy is per reaction as written (the run's coefficient 1). A
  reference run replaced by the same material in another cell size gives the
  same values (its coefficient or count scales), and so does the run itself
  for the per-atom, per-area and per-adsorbate values (a reaction's run in an
  n-times cell is n reactions). Runs that printed several temperatures (TEMPERAT) in
  common also get ΔH(T) and ΔG(T) in the property's JSON (`vs_temperature`).
  Otherwise only ΔE is stored, and the JSON's `thermo_note` says which run
  has no thermodynamics.
- Each property's JSON (`formation_energy`, `cohesive_energy`,
  `surface_energy`, `adsorption_energy`, `reaction_energy`) lists every run
  with its composition, energy, functional and basis set, the warnings, the
  coefficients (the reaction's equation, k, n and A) and the thermodynamics
  used.
- **Composition**: the run's atomic positions, else its Mulliken table,
  else the output's ATOMS IN THE UNIT CELL table. The surface energy needs
  the slab's composition to be n times the bulk's (a non-stoichiometric
  slab needs chemical potentials, which are not covered) and assumes two
  equivalent surfaces (a symmetric slab). The adsorption energy needs the
  complex to be the surface plus a whole number k of adsorbates; a slab or
  polymer surface must be in the complex's cell (area or length within 1%),
  and an adsorbate is a molecule or a layer in that cell. A reaction without
  coefficients is balanced from the compositions when it balances in exactly
  one way (the run's coefficient is 1); given coefficients must balance.

### Which runs are the references

In this order: the runs given on the command line or in the GUI, else the
run's saved choice, else a lookup in the database.

```bash
mace analyze --extract-properties --output-file slab.out --reference bulk=bulk.out
mace analyze --extract-properties --output-file AB.out --reference reactants=A.out --reference reactants=2*B_OPT_20260101_120000
mace analyze --extract-properties --output-file AB.out --reference reactants=auto   # forget the saved choice
```

- `--reference NAME=FILE|CALC_ID`, repeatable (several elements, reactants);
  NAME is one of `elements`, `atoms`, `bulk`, `surface`, `adsorbate`,
  `reactants`, `products` (another name is an error);
  `N*FILE` or `N*CALC_ID` gives a reaction coefficient. A choice given for a
  property that is then stored is **saved with the run** (the
  `reference_runs` row, written by `mace-reference`; a file the database
  tracks is saved as its calc_id), so every later extraction (a folder
  rebuilt, Extract Again) uses it. `NAME=auto` forgets it. Given with a
  folder, a reference applies to every run of it (the elements of a set of
  compounds); it is saved only with the runs whose property was stored,
  and never for an output without a calculation record (a warning says so;
  `--register` records it).
- **The database's lookup** (nothing given or saved) uses finished OPT, SP,
  FREQ and CPHF runs that may be combined with the run (below). Formation: for
  each element, the bulk (3D) or molecule (0D) run of that element alone
  with the lowest energy per atom (its most stable phase), never another run
  of the run's own material or of its structure (the same composition and
  cell per input cell: its phonon supercells, its CPHF run); an element
  itself is measured against it, and the lowest one is skipped (zero by
  definition). Cohesive: the
  isolated-atom (one-atom molecule) run of each element with the lowest
  energy (its ground spin state). Surface: the bulk (3D) run of the slab's
  composition with the lowest energy per atom. Adsorption, for a slab
  complex only: the clean surface and adsorbate that add up to it in the same
  cell (a surface sharing no element with the adsorbate first, then the
  closest cell; a phonon supercell's cell is the supercell's, so a 2x2
  supercell of a sheet is no sheet with three adsorbates). A reaction's runs are never looked up. A folder
  (`--extract-properties DIR`) computes the properties that read other runs
  in a second pass, after every run's own rows are stored, for every run of
  the folder (unchanged ones under `--skip-unchanged` too): the values do
  not depend on the order the runs are read in, and a lookup whose answer
  changed (a new or lower-energy reference run) is updated. A reference
  run that cannot be read (a locked database) is an `error`, never "needs
  the reference run", and the stored value stays.
- **Add Reference…** (the Properties tab, for a run in the database):
  choose the reference energy, then the run(s) of each reference among the
  database's runs (those that fit first; Show All Runs lists every run), each
  marked **ok**, **warning** or **refused** with its reasons. Compute and Save
  runs the run's extraction with `--reference NAME=CALC_ID` (the command is
  shown): the property is computed and stored, and the choice saved. Use the
  Database's Choice forgets it. The tab's footer then says whether the
  property was stored, or why not.

### Runs that may not be combined

Checked for every pair of runs, given ones included; a refusal is the skip
reason in the run's `extraction_provenance` (and the picker's mark).

| Refused (nothing stored) | Warned (stored; the JSON's `warnings`) |
|---|---|
| another functional (both named by their deck keyword, or both by CRYSTAL's energy expression); a deck's `EXCHANGE`/`CORRELAT` pair is named `X/C` (`PBE/PBE`, `LDA/VWN`, with ` HYBRID n` for n % exact exchange) | a functional named by its keyword in one run and by the energy expression in the other (no deck beside one output) |
| another named basis set (BASISSET) | a basis set written in the deck (`custom (in the deck)`): not compared |
| another CRYSTAL version (CRYSTAL17 against CRYSTAL23) | another release of the same version |
| D3 or gCP in one run only (both printing their energy terms) | a functional, basis set or version a run does not say |
| a run with no energy | a spin-polarized run combined with a closed-shell one (valid only when the closed-shell run has no unpaired electrons) |

**Not covered**: basis-set superposition error (BSSE, counterpoise)
corrections of adsorption and binding energies, charged-defect corrections
(image charges, potential alignment) and chemical potentials
(non-stoichiometric slabs, defect formation energies). Compute those
separately.

## Not computed, and why

Properties a derived property could be expected to give but does not, each with its reason:

- **Charge-density difference** (CHARGE+POTENTIAL): every difference needs a second density (a reference
  run, or the atoms' densities from another run); it belongs with the reference-run properties. The
  Volume document's combination computes one interactively.
- **Dipole correction of a molecule or polymer**: CRYSTAL's 0D and 1D systems have no periodic images
  along their open directions, so no correction exists (a slab's is `cell_dipole_correction_ev`).
- **Work function of a bulk, polymer or molecule**: no vacuum along a periodic axis (bulk), and a
  molecule's or polymer's E<sub>vac</sub> − E<sub>F</sub> is not a work function.
- **Planar average of a polymer or molecule**: no lattice planes to average over.
- **Dipole from the density of a slab**: the grid sum over a slab's many atoms is not accurate enough;
  its dipole is read from the exact potential step between its vacuum regions instead.
- **Bader charges and the density dipole on a thinned cube** (the example suite's cubes keep every
  second point, N points over N − ½ steps of a period): the grid sums need CRYSTAL's own even grid;
  the potential profile still works.
- **Density (g/cm³)**: the extractor's own key `density`.
- **Debye temperature from an ELASTIC run's sound velocities**: the extractor's own key
  `debye_temperature_elastic_k`.
- **Thermodynamics versus T**: `thermodynamics_vs_temperature` holds every TEMPERAT/PRESSURE point
  of a FREQ run; nothing is missing.
- **Mean-square displacement of a periodic cell**: its printed modes are Γ's only; the acoustic and
  dispersive modes that carry most of ⟨u<sup>2</sup>⟩ are missing (a molecule's is computed).
- **Phonon Debye temperatures and acoustic velocities of a Γ-only FREQ run**: they need the phonon
  DOS (FREQCALC PDOS) or bands through Γ (DISPERSI + BANDS) in the run's fort.25.
- **Packing fraction of a slab, polymer or molecule**: it depends on the vacuum the model adds.

## Inventory

<!-- BEGIN GENERATED: mace analyze --list-properties --update-doc docs/properties.md -->

The database stores **336 keys** from the extractor's own parsing and **232 keys** of **31 derived properties** (19 of these keys belong to interactive properties, saved from the GUI). Of the 37 things the GUI computes or shows from a run, 27 are in the database and **9 are gaps** (marked **gap** below).

SCF in the calculation-type column means OPT, SP, FREQ, ELASTIC, EOS, QHA and CPHF (an ELASTIC, EOS or QHA run's structure and energies are its reference cell's).

### Derived properties

| Property | Keys | Unit | Calculation types | Needs | Version | Source |
|---|---|---|---|---|---|---|
| adsorption_energy | 7 keys: `adsorption_energy_ev`, `adsorption_energy_kj_mol`, `adsorption_enthalpy_ev`, ... | JSON, eV, kJ/mol | OPT, SP, FREQ, CPHF | reference run surface, reference run adsorbate | 1 | the complex's energy and composition, a clean-surface run and an adsorbate run |
| bader_charges | 7 keys: `bader_charge_min`, `bader_charge_max`, `bader_spin_total`, ... | JSON, dimensionless, e, electrons, μB | CHARGE+POTENTIAL | DENS, SPIN (optional) | 1 | <stem>_DENS.CUBE (+ _SPIN.CUBE): MACE's grid Bader partition (on-grid ascent with near-grid refinement), mace.properties.volumetric.bader |
| band_edge_character | 7 keys: `vbm_character`, `vbm_character_fraction`, `cbm_character`, ... | JSON, category, dimensionless | DOSS | DOSS | 1 | DOSS.DAT via mace/gui/results.dos_data (the GUI's DOS plot reader, projection labels from the .d3); the edges of dos_structure |
| band_gap_character | 9 keys: `band_path_gap_type`, `band_path_vbm_k`, `band_path_cbm_k`, ... | JSON, category, eV, states/(eV·atom) | BAND | BAND, DOSS (optional) | 2 | BAND.DAT (or BAND fort.25) via mace.transport.mass.read_band, the reader of mace tau's masses; N(E_F) of the DOSS.DAT via advanced_electronic_analyzer.fermi_dos_per_atom |
| band_spin_splitting | `vbm_spin_splitting_mev`, `cbm_spin_splitting_mev`, `band_max_spin_splitting_mev` | meV | BAND | BAND | 1 | BAND.DAT (or BAND fort.25) via mace.transport.mass.read_band, the reader of mace tau's masses |
| band_widths | `valence_band_width_ev`, `vbm_band_dispersion_ev`, `cbm_band_dispersion_ev` | eV | BAND | BAND | 1 | BAND.DAT (or BAND fort.25) via mace.transport.mass.read_band, the reader of mace tau's masses |
| carrier_concentration | `intrinsic_fermi_level_300k_ev`, `intrinsic_carrier_concentration_300k_cm3`, `intrinsic_carrier_concentration_300k_cm2`, `intrinsic_carrier_concentration_300k_cm1`, `carrier_concentration_300k` | JSON, cm⁻², cm⁻³, cm⁻¹, eV | DOSS | DOSS | 1 | DOSS.DAT via mace/gui/results.dos_data (the GUI's DOS plot reader, projection labels from the .d3); the gap of dos_structure; the cell of the .out |
| cell_dipole | `cell_dipole_debye`, `cell_dipole_x_debye`, `cell_dipole_y_debye`, `cell_dipole_z_debye`, `cell_dipole_correction_ev`, `cell_dipole` | D, JSON, eV | CHARGE+POTENTIAL | POT (optional), DENS (optional), SCF (optional) | 1 | a slab: <stem>_POT.CUBE's vacuum step; a molecule or polymer: the parent SCF's DIPOLE MOMENT line, else <stem>_DENS.CUBE and the cube's nuclei on a fine grid (mace.properties.volumetric) |
| cohesive_energy | `cohesive_energy_ev_per_atom`, `cohesive_enthalpy_ev_per_atom`, `cohesive_gibbs_energy_ev_per_atom`, `cohesive_energy` | JSON, eV/atom | OPT, SP, FREQ, CPHF | reference run atoms | 1 | the run's energy and composition, and isolated-atom runs |
| d_band_center | `d_band_center_ev`, `d_band_center_occupied_ev`, `d_band_filling`, `d_band_width_ev`, `d_band_element`, `d_band` | JSON, category, dimensionless, eV | DOSS | DOSS | 2 | DOSS.DAT via mace/gui/results.dos_data (the GUI's DOS plot reader, projection labels from the .d3) |
| dipole_moment | `dipole_moment_debye`, `dipole_moment_x_debye`, `dipole_moment_y_debye`, `dipole_moment_z_debye`, `dipole_potential_step_ev` | D, eV | SP, OPT, FREQ, CPHF | - | 1 | .out: DIPOLE MOMENT ALONG NON PERIODIC DIRECTIONS (the last SCF; the Debye line) and, for a slab, primitive_cell_area |
| dos_effective_mass | `dos_effective_mass_electron`, `dos_effective_mass_hole`, `dos_effective_masses` | JSON, m_e | DOSS | DOSS | 1 | DOSS.DAT via mace.transport.mass.dos_mass (mace tau's DOS fit, the integrated DOS near the edge) |
| effective_masses | `effective_mass_electron_mc`, `effective_mass_electron_md`, `effective_mass_hole_mc`, `effective_mass_hole_md`, `effective_masses` | JSON, m_e | BAND | BAND, DOSS (optional) | 1 | BAND.DAT (+ DOSS.DAT) via mace.transport.mass.auto_masses (mace tau --m-star auto) |
| energy_per_atom | `energy_per_atom_ev` | eV | every run | total_energy_au, atoms_in_unit_cell | 1 | total_energy_au and atoms_in_unit_cell of the .out |
| fermi_velocity | `fermi_velocity_mean_m_s`, `fermi_velocity_min_m_s`, `fermi_velocity_max_m_s`, `fermi_level_crossings`, `fermi_velocity_crossings` | JSON, dimensionless, m/s | BAND | BAND | 1 | BAND.DAT (or BAND fort.25) via mace.transport.mass.read_band, the reader of mace tau's masses |
| formation_energy | `formation_energy_ev_per_atom`, `formation_enthalpy_ev_per_atom`, `formation_gibbs_energy_ev_per_atom`, `formation_energy` | JSON, eV/atom | OPT, SP, FREQ, CPHF | reference run elements | 1 | the run's energy and composition, and its elemental reference runs |
| mean_square_displacement | `msd_mean_300k_ang2`, `msd_max_300k_ang2`, `msd_mean_0k_ang2`, `mean_square_displacements` | JSON, Å² | FREQ | - | 1 | .out: NORMAL MODES NORMALIZED TO CLASSICAL AMPLITUDES and the isotopic masses (the GUI's mode viewer's parser) |
| mechanics_derived | 79 keys: `youngs_modulus_min`, `youngs_modulus_min_2d`, `youngs_modulus_max`, ... | C/m², GPa, J/(mol·K), JSON, N/m, TPa⁻¹, W/(m·K), boolean, cm⁻¹, degrees, dimensionless, e, km/s, m/N, meV/cell, pC/N, pm/V | ELASTIC, EOS, QHA, CPHF, FREQ | - | 1 | mace.elastic.properties.all_properties(mace.elastic.analysis.analyze(.out)): the Mechanical Properties Results' numbers |
| mulliken_summary | `mulliken_charge_min`, `mulliken_charge_max`, `mulliken_spin_total`, `mulliken_spin_abs_total`, `mulliken_spin_max_abs`, `mulliken_by_element` | JSON, e, μB | SP, OPT, FREQ, CPHF | processed_atomic_charges | 1 | processed_atomic_charges (the .out's Mulliken tables, population_analysis_processor) |
| orbital_composition | `homo_character`, `homo_character_fraction`, `lumo_character`, `lumo_character_fraction`, `orbital_composition` | JSON, category, dimensionless | ORBITALS | orbitals_homo_band | 1 | the .molden files via mace/orbitals (OrbitalRun: coefficients and the overlap S(k) of its self-check) |
| packing_fraction | `packing_fraction_vdw`, `packing_fraction_covalent` | dimensionless | SP, OPT, FREQ, CPHF | - | 1 | the final geometry of the .out; mace.analysis.porosity's distance field on a grid of 0.15 Å (at most 10⁶ points) |
| phonon_debye_temperature | `debye_temperature_phonon_k`, `debye_temperature_phonon_log_k`, `phonon_debye_moments` | JSON, K | FREQ | f25 | 1 | the PDOS blocks of the run's fort.25 (FREQCALC PDOS), via mace.plotting.f25 |
| phonon_sound_velocities | `sound_velocity_phonon_ta_min_km_s`, `sound_velocity_phonon_la_max_km_s`, `sound_velocity_phonon_mean_km_s`, `debye_temperature_sound_k`, `phonon_sound_velocities` | JSON, K, km/s | FREQ | f25 | 1 | the BAND blocks of the run's fort.25 (FREQCALC DISPERSI + BANDS), via mace.plotting.f25; the atom count and volume of the computed cell (a SCELPHONO run's supercell) the .out prints |
| potential_profile | 7 keys: `potential_planar_average`, `vacuum_level_top_ev`, `vacuum_level_bottom_ev`, ... | JSON, eV | CHARGE+POTENTIAL | POT | 1 | <stem>_POT.CUBE (POT3) and the .out's FERMI ENERGY, via mace.properties.volumetric |
| reaction_energy | 7 keys: `reaction_energy_ev`, `reaction_energy_kj_mol`, `reaction_enthalpy_ev`, ... | JSON, eV, kJ/mol | OPT, SP, FREQ, CPHF | reference run reactants, reference run products | 1 | the run (a product) and the reactant and product runs the user gives |
| reference_runs (interactive) | `reference_runs` | JSON | OPT, SP, FREQ, CPHF | - | 1 | --reference ROLE=FILE\|CALC_ID, or Add Reference… in the Properties tab |
| relaxation_time_dpt (interactive) | 18 keys: `relaxation_time_dpt_electron_x_fs`, `mobility_dpt_electron_x_cm2_vs`, `relaxation_time_dpt_electron_x`, ... | JSON, cm²/(V·s), fs | SP | reference run strained_runs | 1 | mace tau fit on a strain series (GUI: Relaxation Time > Fit > Save to Database) |
| run_info | `dft_functional`, `basis_set`, `crystal_version`, `mpi_processes` | category, dimensionless | every run | d12 (optional) | 1 | the .out banner and the .d12 beside it (the GUI's run card) |
| sommerfeld_coefficient | `sommerfeld_gamma_mj_mol_k2`, `sommerfeld_gamma_per_atom_mj_mol_k2` | mJ/(mol·K²) | DOSS | dos_at_fermi_total | 1 | dos_at_fermi_total (N(E_F), both spins, states/eV/cell) of the extractor |
| surface_energy | 7 keys: `surface_energy_j_m2`, `surface_energy_ev_ang2`, `surface_enthalpy_j_m2`, ... | J/m², JSON, eV/Å² | OPT, SP, FREQ, CPHF | reference run bulk | 1 | the slab's energy, composition and area, and a bulk run |
| thermodynamics_vs_temperature | `thermodynamics_vs_temperature`, `thermodynamic_n_points` | JSON, dimensionless | FREQ | - | 1 | the .out's THERMODYNAMIC FUNCTIONS ... AT (T = .., P = ..) blocks |

### Display units

The units the Properties tab can show a value in (display only: the database keeps each key in its stored unit). A preset sets every dimension listed for it; a dimension it does not name keeps the stored unit.

| Dimension | Units | Hartree atomic units | eV/Å (solid state) | kJ/mol (chemistry) |
|---|---|---|---|---|
| Energy | eV, meV, Hartree, mHartree, Ry, kJ/mol, kcal/mol, cm⁻¹ | Hartree | eV | kJ/mol |
| Energy levels (a gap, a band edge, E<sub>F</sub>, a work function) | the same | Hartree | eV | eV |
| Length | Å, Bohr, nm, pm | Bohr | Å | Å |
| Area | Å², Bohr², nm² | Bohr² | Å² | Å² |
| Volume | Å³, Bohr³, nm³ | Bohr³ | Å³ | Å³ |
| Inverse length | Å⁻¹, bohr⁻¹, nm⁻¹ | bohr⁻¹ | Å⁻¹ | Å⁻¹ |
| Pressure | GPa, MPa, kbar, bar, Mbar, atm, Hartree/Bohr³ | Hartree/Bohr³ | GPa | GPa |
| Compressibility | TPa⁻¹, GPa⁻¹ | as stored | as stored | as stored |
| Force | eV/Å, Hartree/Bohr | Hartree/Bohr | eV/Å | eV/Å |
| Dipole moment | D, e·Å, e·Bohr | e·Bohr | e·Å | D |
| Frequency | cm⁻¹, THz, meV | as stored | THz | cm⁻¹ |
| Characteristic temperature | K, meV | as stored | K | as stored |
| Density | g/cm³, kg/m³ | as stored | as stored | as stored |
| Time | fs, ps | as stored | as stored | as stored |
| Duration | seconds, min, h | as stored | as stored | as stored |
| Velocity | km/s, m/s, Å/fs | as stored | as stored | as stored |
| Angle | degrees, rad | as stored | as stored | as stored |
| Entropy | meV/K, mHartree/K, J/(mol·K) | mHartree/K | meV/K | J/(mol·K) |
| Density of states | states/eV, states/Hartree | states/Hartree | states/eV | states/eV |
| Surface energy | J/m², mJ/m², eV/Å², meV/Å², Hartree/Bohr² | Hartree/Bohr² | J/m² | J/m² |
| Power factor | W/(m·K²), mW/(m·K²), μW/(cm·K²) | as stored | as stored | as stored |
| Seebeck coefficient | μV/K, mV/K, V/K | as stored | as stored | as stored |
| Mobility | cm²/(V·s), m²/(V·s) | as stored | as stored | as stored |

**One row for one quantity**: 30 quantities are stored under several keys that differ only by unit; the tab shows each once, under its first key below (the most precise), and converts it to the unit picked:

`total_energy_au` (`total_energy_ev`), `total_energy_plus_d3_au` (`total_energy_plus_d3_ev`), `total_energy_corrected_au` (`total_energy_corrected_ev`), `d3_dispersion_energy_au` (`d3_dispersion_energy_ev`), `gcp_energy_au` (`gcp_energy_ev`), `kinetic_energy_au` (`kinetic_energy_ev`), `electron_electron_energy_au` (`electron_electron_energy_ev`), `electron_nuclear_energy_au` (`electron_nuclear_energy_ev`), `nuclear_nuclear_energy_au` (`nuclear_nuclear_energy_ev`), `exchange_energy_au` (`exchange_energy_ev`), `correlation_energy_au` (`correlation_energy_ev`), `electronic_energy_au` (`electronic_energy_ev`, `electronic_energy_kj_mol`), `zero_point_energy_au` (`zero_point_energy_ev`, `zero_point_energy_kj_mol`), `thermal_energy_au` (`thermal_energy_ev`, `thermal_energy_kj_mol`), `pv_term_au` (`pv_term_ev`, `pv_term_kj_mol`), `entropy_term_au` (`entropy_term_ev`, `entropy_term_kj_mol`), `free_energy_correction_au` (`free_energy_correction_ev`, `free_energy_correction_kj_mol`), `gibbs_free_energy_au` (`gibbs_free_energy_ev`, `gibbs_free_energy_kj_mol`), `enthalpy_au` (`enthalpy_ev`, `enthalpy_kj_mol`), `entropy_mhartree_cell_k` (`entropy_mev_cell_k`, `entropy_j_mol_k`), `heat_capacity_mhartree_cell_k` (`heat_capacity_mev_cell_k`, `heat_capacity_j_mol_k`), `surface_energy_j_m2` (`surface_energy_ev_ang2`), `surface_enthalpy_j_m2` (`surface_enthalpy_ev_ang2`), `surface_gibbs_energy_j_m2` (`surface_gibbs_energy_ev_ang2`), `adsorption_energy_ev` (`adsorption_energy_kj_mol`), `adsorption_enthalpy_ev` (`adsorption_enthalpy_kj_mol`), `adsorption_gibbs_energy_ev` (`adsorption_gibbs_energy_kj_mol`), `reaction_energy_ev` (`reaction_energy_kj_mol`), `reaction_enthalpy_ev` (`reaction_enthalpy_kj_mol`), `reaction_gibbs_energy_ev` (`reaction_gibbs_energy_kj_mol`).

**Per cell (extensive)**, also shown per atom and per formula unit (with the keys of the same quantity in other units): `bader_spin_total`, `chargepot_scf_total_energy_au`, `correlation_energy_au`, `d3_dispersion_energy_au`, `dos_at_fermi_alpha`, `dos_at_fermi_beta`, `dos_at_fermi_total`, `electron_electron_energy_au`, `electron_nuclear_energy_au`, `electronic_energy_au`, `enthalpy_au`, `entropy_mhartree_cell_k`, `entropy_term_au`, `exchange_energy_au`, `free_energy_correction_au`, `gcp_energy_au`, `gibbs_free_energy_au`, `heat_capacity_mhartree_cell_k`, `kinetic_energy_au`, `mulliken_spin_abs_total`, `mulliken_spin_total`, `nuclear_nuclear_energy_au`, `primitive_cell_volume`, `pv_term_au`, `thermal_energy_au`, `total_energy_au`, `total_energy_corrected_au`, `total_energy_plus_d3_au`, `two_component_magnetization_norm`, `zero_point_energy_au`.

### What the GUI computes or shows

| What | Where | Calculation types | Database keys | Status | Note |
|---|---|---|---|---|---|
| Run details card: energy, energy per atom, functional, basis set, CRYSTAL version, MPI processes | Run Analysis › card (runinfo.meta_job) | every .out | total_energy_au, energy_per_atom_ev, dft_functional, basis_set, crystal_version, mpi_processes | derived | energy_per_atom and run_info: the card reads the same functions |
| Run details card: start and end dates, wall and CPU time, shrink, TOLINTEG, spin | Run Analysis › card | every .out | wall_time_s, cpu_time_total_s, scf_shrink_k*, scf_tolinteg*, spin_polarized | stored | the dates are the calculations table's started_at/completed_at for tracked runs |
| Band-edge effective masses m_c, m_d (electrons, holes): the accurate ones, to use | Relaxation Time dialog (m* auto), mace tau | BAND | effective_mass_{electron,hole}_{mc,md}, effective_masses | derived | stored with the BAND run (no transport axis): every degenerate band and valley, several directions, R² checked. The extractor's real_{electron,hole,average}_effective_mass is a quick estimate (one band along the path, AdvancedElectronicAnalyzer) and gives other numbers |
| Deformation-potential relaxation time and mobility of a strain series | Relaxation Time dialog › Fit | SP (the unstrained run) | relaxation_time_dpt_{carrier}_{axis}_fs, mobility_dpt_{carrier}_{axis}_cm2_vs, relaxation_time_dpt_{carrier}_{axis} | interactive | Save to Database in the dialog |
| Thermodynamics at every TEMPERAT temperature (G, H, S, Cv vs T) | FREQ output (the first block only was stored) | FREQ | thermodynamics_vs_temperature, thermodynamic_n_points | derived | - |
| Reference energies: formation, cohesive, surface, adsorption and reaction energies (ΔE; ΔH and ΔG with FREQ thermodynamics for every run) | Properties tab › Add Reference… (and the database's own lookup) | OPT, SP, FREQ | formation_*, cohesive_*, surface_*, adsorption_*, reaction_*, reference_runs | derived | the references chosen are saved with the run (reference_runs, kept on re-extraction) |
| Transport peaks: \|S\|max, PF max and ZTe max (average and per axis), carrier type | Transport plot | TRANSPORT | transport_* | stored | - |
| Transport curves S, σ, κ, PF, ZTe, κe, L vs μ and T | Transport plot (results.transport_data) | TRANSPORT | - | **gap** | curves; the peaks are stored |
| Total ZT with a lattice κL (and τ) | Transport plot › ZT with κL | TRANSPORT | transport_zt_total* | **gap** | stored by the CLI (--kappa-lattice); the GUI's value is not saved |
| Elastic: Voigt/Reuss/Hill moduli, Young's modulus, Poisson's ratio, A^U, Debye T, Born stability | Mechanical Properties Results | ELASTIC | bulk_modulus_*, shear_modulus_*, ... | stored | - |
| Elastic: directional extremes of E, G, ν and the linear compressibility, sound and group velocity extremes, anisotropy indices, Pugh, Cauchy, Chen/Tian hardness, Clarke/Cahill κmin, piezoelectric extremes, the 2D anisotropy | Mechanical Properties Results (elastic/views.py) | ELASTIC | youngs_modulus_{min,max}, shear_modulus_*, poisson_ratio_*, sound_velocity_q*_*, pugh_ratio, cauchy_*, vickers_hardness_*, min_thermal_conductivity_*, piezo_*, mechanics_derived | derived | mechanics_derived: mace.elastic.properties.all_properties, the document's numbers |
| Elastic: the directions of the extremes; each Born criterion | Mechanical Properties Results › surfaces (elastic/views.py) | ELASTIC | - | **gap** | the extremes are stored, not their directions; elastic_born_stable is |
| EOS fit residuals; QHA S, Cv, Cp, γ near 300 K; CPHF principal n, birefringence, 2V, SHG d_eff; Born charges, ε0, LO-TO of FREQ with INTCPHF | Mechanical Properties Results | EOS, QHA, CPHF, FREQ | eos_*_residual_*, formula_units, qha_*_300k, refractive_index_principal_*, birefringence*, optic_axial_angle_deg, shg_d_*, born_charge_*, dielectric_constant_static, lo_to_splitting_max | derived | mechanics_derived |
| QHA harmonic heat capacity per volume; CPHF dielectric spectrum | Mechanical Properties Results | QHA, CPHF | - | **gap** | curves (qha_thermal_properties and dielectric_cphf are stored) |
| Planar and macroscopic average of the potential; vacuum levels, potential step and work functions of a slab | Properties tab | CHARGE+POTENTIAL | potential_planar_average, vacuum_level_{top,bottom}_ev, potential_vacuum_step_ev, work_function*_ev | derived | potential_profile (POT cube) |
| Cell dipole (slab: from the vacuum step, with its dipole correction; molecule, polymer: from the density) | Properties tab | CHARGE+POTENTIAL | cell_dipole_*_debye, cell_dipole_correction_ev, cell_dipole | derived | cell_dipole |
| Bader charges, basin volumes and spin moments | Properties tab | CHARGE+POTENTIAL | bader_charge_{min,max}, bader_spin_*, bader_charges | derived | bader_charges: MACE's own grid algorithm (DENS + SPIN cubes) |
| Phonon Debye temperatures (DOS moments), acoustic velocities and their Debye temperature; mean-square displacements of a molecule | Properties tab | FREQ | debye_temperature_phonon*_k, sound_velocity_phonon_*, debye_temperature_sound_k, msd_* | derived | phonon_debye_temperature, phonon_sound_velocities (fort.25), mean_square_displacement (0D) |
| Mulliken charge and spin per element, total magnetization; printed dipole moment; packing fractions | Properties tab | SP, OPT, FREQ | mulliken_charge_*, mulliken_spin_*, mulliken_by_element, dipole_moment_*, packing_fraction_* | derived | mulliken_summary, dipole_moment, packing_fraction |
| IR and Raman spectra (broadened) | IR/Raman plot (spectra.py) | FREQ | vibrational_frequencies, raman_intensities | **gap** | the modes and intensities are stored, the broadened curves are not |
| Band and DOS plots: gaps, band edges, DOS at E_F | Band and DOS plots | BAND, DOSS | band_*, dos_* | stored | - |
| Band gap character: direct or indirect, the k points of the VBM, CBM and smallest direct gap, the gap of each spin channel; metal or semimetal of a gapless run | Properties tab | BAND | band_path_gap_type, band_path_{vbm,cbm,direct_gap}_k, band_path_gap_{alpha,beta}_ev, band_path_fermi_dos_per_atom, band_path_semimetal_threshold, band_edges | derived | along the band path (BAND.DAT via mace.transport.mass.read_band); a gapless run is a semimetal when N(E_F) of the DOSS run is below 0.05 states/eV/atom |
| Valence band width and the dispersion of the VBM and CBM bands | Properties tab | BAND | valence_band_width_ev, vbm_band_dispersion_ev, cbm_band_dispersion_ev | derived | the conduction manifold's width is in band_edges only (an LCAO basis gives it no top) |
| Spin splitting of the band edges (spin-polarized runs) | Properties tab | BAND | vbm_spin_splitting_mev, cbm_spin_splitting_mev, band_max_spin_splitting_mev | derived | - |
| Fermi velocities of a metal or semimetal at each E_F crossing of the path | Properties tab | BAND | fermi_velocity_{mean,min,max}_m_s, fermi_level_crossings, fermi_velocity_crossings | derived | - |
| Sommerfeld coefficient γ from N(E_F) | Properties tab | DOSS | sommerfeld_gamma_mj_mol_k2, sommerfeld_gamma_per_atom_mj_mol_k2 | derived | N(E_F) itself is the extractor's dos_at_fermi_total |
| d-band center (to E_F + 10 eV, and occupied only), filling and width of transition-metal d projections | Properties tab | DOSS | d_band_center_ev, d_band_center_occupied_ev, d_band_filling, d_band_width_ev, d_band_element, d_band | derived | the projections as the DOS plot reads and labels them (results.dos_data) |
| Orbital and atom character of the band edges (of E_F in a metal) from the projected DOS | Properties tab | DOSS | {vbm,cbm,fermi_level}_character(_fraction), band_edge_composition | derived | - |
| Carrier concentrations against the Fermi level at 300 K; intrinsic level and n_i | Properties tab | DOSS | carrier_concentration_300k, intrinsic_fermi_level_300k_ev, intrinsic_carrier_concentration_300k_cm{3,2,1} | derived | - |
| Density-of-states effective masses from the DOS (rough) | Properties tab | DOSS | dos_effective_mass_{electron,hole}, dos_effective_masses | derived | mace.transport.mass.dos_mass, the DOS fit of mace tau, on CRYSTAL's smoothed DOS (NPOL, grid); the BAND run's effective_mass_*_md is the accurate m_d |
| Mulliken composition of the HOMO and LUMO (per k point and spin) | Properties tab | ORBITALS | {homo,lumo}_character(_fraction), orbital_composition | derived | the coefficients and overlap of the orbital viewer (mace/orbitals) |
| Volume statistics (min, max, integral of a cube) | Volume view (volume.stats) | CHARGE+POTENTIAL | - | **gap** | - |
| Convergence: SCF cycles per point, gradients and displacements vs thresholds | Run Analysis › Convergence | OPT | scf_cycles, optimization_* | **gap** | per-point curves; the final values are stored |
| Problems (imaginary modes, warnings, tolerances) | Problems tab | every .out | num_imaginary_frequencies | not a property | - |
| Structure Analysis: porosity, powder pattern, topology, polyhedra, morphology, planes | Analysis tab | structures (and a run's geometry) | - | **gap** | computed per structure for the session; not stored |
| Coordination: CN, ECoN, neighbor shells | Coordination tab | structures, runs | neighbor_analysis, final_neighbor_analysis | stored | ECoN is not stored |
| Phonopy bands, DOS and thermal properties | Phonopy results | phonopy folders | - | **gap** | not a CRYSTAL output |

### Stored keys by calculation type

Each key once, under the first calculation type it applies to; the last column lists all of them.

#### Every run

| Key | Unit | Source | Calculation types | Meaning |
|---|---|---|---|---|
| `calculation_type` | category | .out: timings and the run's type | every run | the run's type, as calculations.calc_type names it: OPT, SP, FREQ, BAND, DOSS, TRANSPORT, CHARGE+POTENTIAL or ORBITALS |
| `cpu_time_total_s` | seconds | .out: timings and the run's type | every run | TOTAL CPU TIME: the CPU time of all processes |
| `extraction_provenance` | JSON | the registry: which derived properties ran, their versions and files | every run | MACE version, registry version and, per derived property, its version, status (stored or skipped, and why) and the files it read |
| `scf_cycles` | cycles | .out: timings and the run's type | every run | cycles of the last SCF |
| `spin_polarized` | boolean | .out: UNRESTRICTED OPEN SHELL, alpha/beta tables, SPIN cube | every run | an unrestricted (spin-polarized) run, whatever its net spin |
| `wall_time_s` | seconds | .out: timings and the run's type | every run | wall time: the last TELAPSE of the output |

#### SCF runs (OPT, SP, FREQ, ELASTIC, EOS, QHA, CPHF)

| Key | Unit | Source | Calculation types | Meaning |
|---|---|---|---|---|
| `alpha_band_gap` | eV | .out: the last SCF | SCF | alpha-channel gap of CRYSTAL's summary |
| `atomic_position_axes` | category | .out: geometry and NEIGHBORS tables | SCF | the axes of atomic_positions when they are not all fractional, as CRYSTAL prints them (e.g. "X/A Y/B Z(ANGSTROM)") |
| `atomic_positions` | coordinates | .out: geometry and NEIGHBORS tables | SCF | atoms of the final geometry (see atomic_position_axes for the axes) |
| `atoms_in_unit_cell` | dimensionless | .out: geometry and NEIGHBORS tables | SCF | atoms in the primitive cell |
| `beta_band_gap` | eV | .out: the last SCF | SCF | beta-channel gap of CRYSTAL's summary |
| `centering_code` | code | .out: SPACE GROUP / layer group lines | SCF | centering code (CENTRING CODE) |
| `correlation_energy_au` | Hartree | .out: the last SCF (TOTAL ENERGY, D3, gCP, ENERGIES IN A.U.) | SCF | correlation energy of the last SCF ("ENERGIES IN A.U.") |
| `correlation_energy_ev` | eV | .out: the last SCF (TOTAL ENERGY, D3, gCP, ENERGIES IN A.U.) | SCF | correlation energy of the last SCF ("ENERGIES IN A.U.") |
| `crystal_system` | category | .out: SPACE GROUP / layer group lines | SCF | crystal system |
| `crystallographic_a` | Å | .out: the final geometry (an OPT's last point; the input geometry otherwise) | SCF | conventional cell a of the final geometry (the input geometry when the run does not optimize); none along a non-periodic direction |
| `crystallographic_alpha` | degrees | .out: the final geometry (an OPT's last point; the input geometry otherwise) | SCF | conventional cell alpha of the final geometry (the input geometry when the run does not optimize); none along a non-periodic direction |
| `crystallographic_b` | Å | .out: the final geometry (an OPT's last point; the input geometry otherwise) | SCF | conventional cell b of the final geometry (the input geometry when the run does not optimize); none along a non-periodic direction |
| `crystallographic_beta` | degrees | .out: the final geometry (an OPT's last point; the input geometry otherwise) | SCF | conventional cell beta of the final geometry (the input geometry when the run does not optimize); none along a non-periodic direction |
| `crystallographic_c` | Å | .out: the final geometry (an OPT's last point; the input geometry otherwise) | SCF | conventional cell c of the final geometry (the input geometry when the run does not optimize); none along a non-periodic direction |
| `crystallographic_cell_volume` | Å³ | .out: the final geometry (an OPT's last point; the input geometry otherwise) | SCF | conventional cell volume |
| `crystallographic_gamma` | degrees | .out: the final geometry (an OPT's last point; the input geometry otherwise) | SCF | conventional cell gamma of the final geometry (the input geometry when the run does not optimize); none along a non-periodic direction |
| `d3_dispersion_energy_au` | Hartree | .out: the last SCF (TOTAL ENERGY, D3, gCP, ENERGIES IN A.U.) | SCF | D3 dispersion correction |
| `d3_dispersion_energy_ev` | eV | .out: the last SCF (TOTAL ENERGY, D3, gCP, ENERGIES IN A.U.) | SCF | D3 dispersion correction |
| `density` | g/cm³ | .out: geometry and NEIGHBORS tables | SCF | density of the final geometry |
| `electron_count` | JSON | computed: electrons of a charged run (ion_fragments) | SCF | electron count of a charged run |
| `electron_electron_energy_au` | Hartree | .out: the last SCF (TOTAL ENERGY, D3, gCP, ENERGIES IN A.U.) | SCF | total E-E of the last SCF ("ENERGIES IN A.U.") |
| `electron_electron_energy_ev` | eV | .out: the last SCF (TOTAL ENERGY, D3, gCP, ENERGIES IN A.U.) | SCF | total E-E of the last SCF ("ENERGIES IN A.U.") |
| `electron_nuclear_energy_au` | Hartree | .out: the last SCF (TOTAL ENERGY, D3, gCP, ENERGIES IN A.U.) | SCF | total E-N + N-E of the last SCF ("ENERGIES IN A.U.") |
| `electron_nuclear_energy_ev` | eV | .out: the last SCF (TOTAL ENERGY, D3, gCP, ENERGIES IN A.U.) | SCF | total E-N + N-E of the last SCF ("ENERGIES IN A.U.") |
| `exchange_energy_au` | Hartree | .out: the last SCF (TOTAL ENERGY, D3, gCP, ENERGIES IN A.U.) | SCF | exchange energy of the last SCF ("ENERGIES IN A.U.") |
| `exchange_energy_ev` | eV | .out: the last SCF (TOTAL ENERGY, D3, gCP, ENERGIES IN A.U.) | SCF | exchange energy of the last SCF ("ENERGIES IN A.U.") |
| `final_max_bond_distance_ang` | Å | .out: geometry and NEIGHBORS tables | SCF | longest neighbor distance (the last table) |
| `final_max_coordination_number` | dimensionless | .out: geometry and NEIGHBORS tables | SCF | largest neighbor count of a shell (the last table) |
| `final_min_bond_distance_ang` | Å | .out: geometry and NEIGHBORS tables | SCF | shortest neighbor distance (the last table) |
| `final_neighbor_analysis` | JSON | .out: geometry and NEIGHBORS tables | SCF | neighbors of each atom, from the last NEIGHBORS table |
| `final_total_coordination_shells` | dimensionless | .out: geometry and NEIGHBORS tables | SCF | neighbor shells listed (the last table) |
| `fragment_charges` | JSON | computed: Mulliken charge and spin per molecular fragment (ion_fragments) | SCF | Mulliken charge and spin of each molecular fragment |
| `gcp_energy_au` | Hartree | .out: the last SCF (TOTAL ENERGY, D3, gCP, ENERGIES IN A.U.) | SCF | gCP correction (3c methods) |
| `gcp_energy_ev` | eV | .out: the last SCF (TOTAL ENERGY, D3, gCP, ENERGIES IN A.U.) | SCF | gCP correction (3c methods) |
| `initial_atomic_positions` | coordinates | .out: geometry and NEIGHBORS tables | SCF | atoms of the input geometry |
| `initial_atoms_count` | dimensionless | .out: geometry and NEIGHBORS tables | SCF | atoms of the input geometry |
| `initial_crystallographic_a` | Å | .out: the final geometry (an OPT's last point; the input geometry otherwise) | SCF | conventional cell a of the input geometry |
| `initial_crystallographic_alpha` | degrees | .out: the final geometry (an OPT's last point; the input geometry otherwise) | SCF | conventional cell alpha of the input geometry |
| `initial_crystallographic_b` | Å | .out: the final geometry (an OPT's last point; the input geometry otherwise) | SCF | conventional cell b of the input geometry |
| `initial_crystallographic_beta` | degrees | .out: the final geometry (an OPT's last point; the input geometry otherwise) | SCF | conventional cell beta of the input geometry |
| `initial_crystallographic_c` | Å | .out: the final geometry (an OPT's last point; the input geometry otherwise) | SCF | conventional cell c of the input geometry |
| `initial_crystallographic_gamma` | degrees | .out: the final geometry (an OPT's last point; the input geometry otherwise) | SCF | conventional cell gamma of the input geometry |
| `initial_density` | g/cm³ | .out: geometry and NEIGHBORS tables | SCF | density of the input geometry |
| `initial_primitive_a` | Å | .out: the final geometry (an OPT's last point; the input geometry otherwise) | SCF | primitive cell a of the input geometry |
| `initial_primitive_alpha` | degrees | .out: the final geometry (an OPT's last point; the input geometry otherwise) | SCF | primitive cell alpha of the input geometry |
| `initial_primitive_b` | Å | .out: the final geometry (an OPT's last point; the input geometry otherwise) | SCF | primitive cell b of the input geometry |
| `initial_primitive_beta` | degrees | .out: the final geometry (an OPT's last point; the input geometry otherwise) | SCF | primitive cell beta of the input geometry |
| `initial_primitive_c` | Å | .out: the final geometry (an OPT's last point; the input geometry otherwise) | SCF | primitive cell c of the input geometry |
| `initial_primitive_cell_area` | Å² | .out: the final geometry (an OPT's last point; the input geometry otherwise) | SCF | slab cell area of the input geometry |
| `initial_primitive_cell_length` | Å | .out: the final geometry (an OPT's last point; the input geometry otherwise) | SCF | polymer cell length of the input geometry |
| `initial_primitive_cell_volume` | Å³ | .out: the final geometry (an OPT's last point; the input geometry otherwise) | SCF | primitive cell volume of the input geometry |
| `initial_primitive_gamma` | degrees | .out: the final geometry (an OPT's last point; the input geometry otherwise) | SCF | primitive cell gamma of the input geometry |
| `kinetic_energy_au` | Hartree | .out: the last SCF (TOTAL ENERGY, D3, gCP, ENERGIES IN A.U.) | SCF | kinetic energy of the last SCF ("ENERGIES IN A.U.") |
| `kinetic_energy_ev` | eV | .out: the last SCF (TOTAL ENERGY, D3, gCP, ENERGIES IN A.U.) | SCF | kinetic energy of the last SCF ("ENERGIES IN A.U.") |
| `layer_group_number` | number | .out: SPACE GROUP / layer group lines | SCF | layer group of a slab ("TWO-SIDED PLANE GROUP N." of the output) |
| `layer_group_symbol` | category | .out: SPACE GROUP / layer group lines | SCF | layer group symbol of a slab |
| `max_bond_distance_ang` | Å | .out: geometry and NEIGHBORS tables | SCF | longest neighbor distance (the first table) |
| `max_coordination_number` | dimensionless | .out: geometry and NEIGHBORS tables | SCF | largest neighbor count of a shell (the first table) |
| `min_bond_distance_ang` | Å | .out: geometry and NEIGHBORS tables | SCF | shortest neighbor distance (the first table) |
| `mulliken_alpha_minus_beta` | electrons | .out: Mulliken population tables (PPAN) | SCF | Mulliken alpha-beta (spin) populations per atom and orbital (mulliken_population per atom: the electron count, not the net charge) |
| `mulliken_alpha_plus_beta` | electrons | .out: Mulliken population tables (PPAN) | SCF | Mulliken alpha+beta (charge) populations per atom and orbital (mulliken_population per atom: the electron count, not the net charge) |
| `mulliken_population` | electrons | .out: Mulliken population tables (PPAN) | SCF | Mulliken populations of a closed-shell run |
| `mulliken_valence_charges` | electrons | .out: Mulliken population tables (PPAN) | SCF | valence charge of each pseudopotential element, by Z |
| `neighbor_analysis` | JSON | .out: geometry and NEIGHBORS tables | SCF | neighbors of each atom, from the first NEIGHBORS table |
| `nuclear_nuclear_energy_au` | Hartree | .out: the last SCF (TOTAL ENERGY, D3, gCP, ENERGIES IN A.U.) | SCF | total N-N of the last SCF ("ENERGIES IN A.U.") |
| `nuclear_nuclear_energy_ev` | eV | .out: the last SCF (TOTAL ENERGY, D3, gCP, ENERGIES IN A.U.) | SCF | total N-N of the last SCF ("ENERGIES IN A.U.") |
| `overlap_population` | JSON | .out: Mulliken population tables (PPAN) | SCF | overlap populations of a closed-shell run |
| `overlap_population_alpha_minus_beta` | JSON | .out: Mulliken population tables (PPAN) | SCF | Mulliken alpha-beta (spin) overlap populations |
| `overlap_population_alpha_plus_beta` | JSON | .out: Mulliken population tables (PPAN) | SCF | Mulliken alpha+beta (charge) overlap populations |
| `primitive_a` | Å | .out: the final geometry (an OPT's last point; the input geometry otherwise) | SCF | primitive cell a of the final geometry (the input geometry when the run does not optimize); none along a non-periodic direction |
| `primitive_alpha` | degrees | .out: the final geometry (an OPT's last point; the input geometry otherwise) | SCF | primitive cell alpha of the final geometry (the input geometry when the run does not optimize); none along a non-periodic direction |
| `primitive_b` | Å | .out: the final geometry (an OPT's last point; the input geometry otherwise) | SCF | primitive cell b of the final geometry (the input geometry when the run does not optimize); none along a non-periodic direction |
| `primitive_beta` | degrees | .out: the final geometry (an OPT's last point; the input geometry otherwise) | SCF | primitive cell beta of the final geometry (the input geometry when the run does not optimize); none along a non-periodic direction |
| `primitive_c` | Å | .out: the final geometry (an OPT's last point; the input geometry otherwise) | SCF | primitive cell c of the final geometry (the input geometry when the run does not optimize); none along a non-periodic direction |
| `primitive_cell_area` | Å² | .out: the final geometry (an OPT's last point; the input geometry otherwise) | SCF | area of the 2D cell (slab) |
| `primitive_cell_length` | Å | .out: the final geometry (an OPT's last point; the input geometry otherwise) | SCF | length of the 1D cell (polymer) |
| `primitive_cell_volume` | Å³ | .out: the final geometry (an OPT's last point; the input geometry otherwise) | SCF | primitive cell volume (3D) |
| `primitive_gamma` | degrees | .out: the final geometry (an OPT's last point; the input geometry otherwise) | SCF | primitive cell gamma of the final geometry (the input geometry when the run does not optimize); none along a non-periodic direction |
| `processed_atomic_charges` | JSON | computed: population_analysis_processor on the Mulliken tables | SCF | net atomic charges (population analysis) |
| `processed_bonding_analysis` | JSON | computed: population_analysis_processor on the Mulliken tables | SCF | bonds from overlap populations (population analysis) |
| `processed_chemical_summary` | JSON | computed: population_analysis_processor on the Mulliken tables | SCF | charge and bond summary (population analysis) |
| `processed_coordination_environments` | JSON | computed: population_analysis_processor on the Mulliken tables | SCF | coordination of each atom (population analysis) |
| `processed_magnetic_info` | JSON | computed: population_analysis_processor on the Mulliken tables | SCF | spin moments (population analysis) |
| `scf_anderson_enabled` | boolean | .out: the SCF and DFT settings echoed | SCF | Anderson mixing |
| `scf_anderson_factor` | dimensionless | .out: the SCF and DFT settings echoed | SCF | Anderson mixing factor |
| `scf_bipolarization` | boolean | .out: the SCF and DFT settings echoed | SCF | BIPOLARIZ |
| `scf_biposize` | dimensionless | .out: the SCF and DFT settings echoed | SCF | BIPOSIZE |
| `scf_broyden_enabled` | boolean | .out: the SCF and DFT settings echoed | SCF | Broyden mixing |
| `scf_broyden_factor` | dimensionless | .out: the SCF and DFT settings echoed | SCF | Broyden mixing factor |
| `scf_dft_grid_points` | dimensionless | .out: the SCF and DFT settings echoed | SCF | DFT grid points |
| `scf_diis` | boolean | .out: the SCF and DFT settings echoed | SCF | DIIS for SCF active |
| `scf_diis_start_cycle` | cycles | .out: the SCF and DFT settings echoed | SCF | first DIIS cycle |
| `scf_diis_subspace_size` | dimensionless | .out: the SCF and DFT settings echoed | SCF | DIIS subspace |
| `scf_exchange_permutation` | boolean | .out: the SCF and DFT settings echoed | SCF | EXCHPERM |
| `scf_exchsize` | dimensionless | .out: the SCF and DFT settings echoed | SCF | EXCHSIZE |
| `scf_fmixing_percentage` | dimensionless | .out: the SCF and DFT settings echoed | SCF | FMIXING, % |
| `scf_grid_pruning` | boolean | .out: the SCF and DFT settings echoed | SCF | grid pruning |
| `scf_ilasize` | dimensionless | .out: the SCF and DFT settings echoed | SCF | ILASIZE |
| `scf_intgpack` | dimensionless | .out: the SCF and DFT settings echoed | SCF | INTGPACK |
| `scf_levshift_cycles` | cycles | .out: the SCF and DFT settings echoed | SCF | level shift cycles |
| `scf_levshift_enabled` | boolean | .out: the SCF and DFT settings echoed | SCF | level shifter on |
| `scf_levshift_factor` | dimensionless | .out: the SCF and DFT settings echoed | SCF | level shift |
| `scf_madelimit` | dimensionless | .out: the SCF and DFT settings echoed | SCF | MADELIMIT |
| `scf_maxcycle` | cycles | .out: the SCF and DFT settings echoed | SCF | MAXCYCLE |
| `scf_poleordr` | dimensionless | .out: the SCF and DFT settings echoed | SCF | POLEORDR |
| `scf_ppan` | boolean | .out: the SCF and DFT settings echoed | SCF | PPAN |
| `scf_savepred` | boolean | .out: the SCF and DFT settings echoed | SCF | SAVEPRED |
| `scf_savewf` | boolean | .out: the SCF and DFT settings echoed | SCF | SAVEWF |
| `scf_scfdir` | boolean | .out: the SCF and DFT settings echoed | SCF | SCFDIR |
| `scf_toldee` | dimensionless | .out: the SCF and DFT settings echoed | SCF | TOLDEE exponent |
| `scf_tolinteg` | code | .out: the SCF and DFT settings echoed | SCF | TOLINTEG T1-T5 exponents when the deck sets TOLINTEG |
| `scf_two_component` | boolean | .out: the SCF and DFT settings echoed | SCF | a two-component (TWOCOMPON) SCF |
| `space_group_number` | number | .out: SPACE GROUP / layer group lines | SCF | space group (for a slab: the corresponding space group CRYSTAL prints); none for a molecule |
| `total_coordination_shells` | dimensionless | .out: geometry and NEIGHBORS tables | SCF | neighbor shells listed (the first table) |
| `total_energy_au` | Hartree | .out: the last SCF (TOTAL ENERGY, D3, gCP, ENERGIES IN A.U.) | SCF | TOTAL ENERGY(DFT): the SCF energy without D3 and gCP |
| `total_energy_corrected_au` | Hartree | .out: the last SCF (TOTAL ENERGY, D3, gCP, ENERGIES IN A.U.) | SCF | total energy + D3 + gCP: the energy to compare |
| `total_energy_corrected_ev` | eV | .out: the last SCF (TOTAL ENERGY, D3, gCP, ENERGIES IN A.U.) | SCF | total energy + D3 + gCP: the energy to compare |
| `total_energy_ev` | eV | .out: the last SCF (TOTAL ENERGY, D3, gCP, ENERGIES IN A.U.) | SCF | TOTAL ENERGY(DFT): the SCF energy without D3 and gCP |
| `total_energy_plus_d3_au` | Hartree | .out: the last SCF (TOTAL ENERGY, D3, gCP, ENERGIES IN A.U.) | SCF | total energy + D3 (without gCP) |
| `total_energy_plus_d3_ev` | eV | .out: the last SCF (TOTAL ENERGY, D3, gCP, ENERGIES IN A.U.) | SCF | total energy + D3 (without gCP) |
| `two_component_atomic_magnetization` | μB | .out: TWOCOMPON SCF magnetization | SCF | final (Mx, My, Mz) of each atom |
| `two_component_magnetization` | μB | .out: TWOCOMPON SCF magnetization | SCF | final total magnetization vector (Mx, My, Mz) |
| `two_component_magnetization_norm` | μB | .out: TWOCOMPON SCF magnetization | SCF | \|M\| of the final total magnetization |

#### OPT

| Key | Unit | Source | Calculation types | Meaning |
|---|---|---|---|---|
| `band_gap` | eV | .out: the last SCF's gap (BAND.DAT/DOSS.DAT gaps are band_fundamental_gap_ev, dos_band_gap_from_dos_ev) | OPT, SP, FREQ, ELASTIC, EOS, QHA, CPHF, DOSS, TRANSPORT | the gap of the last SCF: the fundamental gap across both spin channels of an unrestricted run; 0 for a run that ends conducting |
| `band_gap_type` | category | .out: the last SCF | OPT, SP, FREQ, ELASTIC, EOS, QHA, CPHF, DOSS, TRANSPORT | direct or indirect (of band_gap) |
| `classification_gap_source` | category | .out: the last SCF | OPT, SP, FREQ, ELASTIC, EOS, QHA, CPHF, BAND, DOSS, TRANSPORT, CHARGE+POTENTIAL | the gap electronic_classification is from: scf (band_gap) or band_structure (band_fundamental_gap_ev) |
| `direct_band_gap` | eV | .out: the last SCF | OPT, SP, FREQ, ELASTIC, EOS, QHA, CPHF, DOSS, TRANSPORT | band_gap, when it is direct |
| `electronic_classification` | category | computed: metal / semiconductor / insulator from the gap (property_keys.classify_gap); a gapless BAND run with a DOSS run beside it: metal or semimetal by N(E_F) per atom (property_keys.classify_gapless) | OPT, SP, FREQ, ELASTIC, EOS, QHA, CPHF, BAND, DOSS, TRANSPORT | metal (gap < 0.01 eV), semiconductor (up to 4 eV) or insulator (above 4 eV), from band_gap, else from the band structure; a gapless band structure with a DOS beside it is a semimetal when N(E_F) < 0.05 states/eV/atom (band_path_gap_type's rule) |
| `fermi_energy_au` | Hartree | .out: the last SCF's EFERMI, else the FERMI ENERGY a BAND or DOSS run prints | OPT, SP, FREQ, ELASTIC, EOS, QHA, CPHF, BAND, DOSS, TRANSPORT | Fermi energy: the last SCF's, else the one a BAND or DOSS output prints; never CRYSTAL's -1.0 placeholder of a spin-locked SCF (fermi_energy_note) |
| `final_gradient_norm` | Hartree/Bohr | .out: OPTGEOM convergence | OPT | last gradient norm |
| `indirect_band_gap` | eV | .out: the last SCF | OPT, SP, FREQ, ELASTIC, EOS, QHA, CPHF, DOSS, TRANSPORT | band_gap, when it is indirect |
| `magnetic_classification` | category | .out: the last SCF | OPT, SP, FREQ, ELASTIC, EOS, QHA, CPHF, BAND, DOSS, TRANSPORT, CHARGE+POTENTIAL | spin_polarized (an unrestricted run with a net or atomic spin) or non_magnetic |
| `optimization_converged` | boolean | .out: OPTGEOM convergence | OPT | OPT END - CONVERGED |
| `optimization_cycles` | cycles | .out: OPTGEOM convergence | OPT | energy and gradient calculations |
| `scf_gilat_shrink` | dimensionless | .out: the SCF and DFT settings echoed | OPT, SP, FREQ, ELASTIC, EOS, QHA, CPHF, BAND, DOSS, TRANSPORT, CHARGE+POTENTIAL | Gilat net shrink |
| `scf_shrink_k1` | dimensionless | .out: the SCF and DFT settings echoed | OPT, SP, FREQ, ELASTIC, EOS, QHA, CPHF, BAND, DOSS, TRANSPORT, CHARGE+POTENTIAL | Monkhorst-Pack shrink, 1st |
| `scf_shrink_k2` | dimensionless | .out: the SCF and DFT settings echoed | OPT, SP, FREQ, ELASTIC, EOS, QHA, CPHF, BAND, DOSS, TRANSPORT, CHARGE+POTENTIAL | Monkhorst-Pack shrink, 2nd |
| `scf_shrink_k3` | dimensionless | .out: the SCF and DFT settings echoed | OPT, SP, FREQ, ELASTIC, EOS, QHA, CPHF, BAND, DOSS, TRANSPORT, CHARGE+POTENTIAL | Monkhorst-Pack shrink, 3rd |

#### FREQ

| Key | Unit | Source | Calculation types | Meaning |
|---|---|---|---|---|
| `dielectric_constant_electronic` | dimensionless | .out: CPHF/CPKS tensors | CPHF, ELASTIC, FREQ | ε∞: mean of the principal CPHF/KS dielectric constants |
| `dielectric_cphf` | JSON | .out: CPHF/CPKS tensors | CPHF, ELASTIC, FREQ | CPHF/KS polarizability α, dielectric ε∞ and χ(1) components, principal ε, refractive indices |
| `electronic_energy_au` | Hartree | .out: the first THERMODYNAMIC FUNCTIONS block (AT T, P) | FREQ | EL, the electronic energy of the FREQ run |
| `electronic_energy_ev` | eV | .out: the first THERMODYNAMIC FUNCTIONS block (AT T, P) | FREQ | EL, the electronic energy of the FREQ run |
| `electronic_energy_kj_mol` | kJ/mol | .out: the first THERMODYNAMIC FUNCTIONS block (AT T, P) | FREQ | EL, the electronic energy of the FREQ run |
| `enthalpy_au` | Hartree | computed: H = G + TS from the printed columns | FREQ | H = G + TS |
| `enthalpy_ev` | eV | computed: H = G + TS from the printed columns | FREQ | H = G + TS |
| `enthalpy_kj_mol` | kJ/mol | computed: H = G + TS from the printed columns | FREQ | H = G + TS |
| `entropy_j_mol_k` | J/(mol·K) | .out: the first THERMODYNAMIC FUNCTIONS block (AT T, P) | FREQ | entropy S |
| `entropy_mev_cell_k` | meV/(cell·K) | .out: the first THERMODYNAMIC FUNCTIONS block (AT T, P) | FREQ | entropy S |
| `entropy_mhartree_cell_k` | mHartree/(cell·K) | .out: the first THERMODYNAMIC FUNCTIONS block (AT T, P) | FREQ | entropy S |
| `entropy_term_au` | Hartree | .out: the first THERMODYNAMIC FUNCTIONS block (AT T, P) | FREQ | TS |
| `entropy_term_ev` | eV | .out: the first THERMODYNAMIC FUNCTIONS block (AT T, P) | FREQ | TS |
| `entropy_term_kj_mol` | kJ/mol | .out: the first THERMODYNAMIC FUNCTIONS block (AT T, P) | FREQ | TS |
| `free_energy_correction_au` | Hartree | .out: the first THERMODYNAMIC FUNCTIONS block (AT T, P) | FREQ | ET+PV-TS |
| `free_energy_correction_ev` | eV | .out: the first THERMODYNAMIC FUNCTIONS block (AT T, P) | FREQ | ET+PV-TS |
| `free_energy_correction_kj_mol` | kJ/mol | .out: the first THERMODYNAMIC FUNCTIONS block (AT T, P) | FREQ | ET+PV-TS |
| `gibbs_free_energy_au` | Hartree | .out: the first THERMODYNAMIC FUNCTIONS block (AT T, P) | FREQ | G = EL+E0+ET+PV-TS |
| `gibbs_free_energy_ev` | eV | .out: the first THERMODYNAMIC FUNCTIONS block (AT T, P) | FREQ | G = EL+E0+ET+PV-TS |
| `gibbs_free_energy_kj_mol` | kJ/mol | .out: the first THERMODYNAMIC FUNCTIONS block (AT T, P) | FREQ | G = EL+E0+ET+PV-TS |
| `has_force_constants` | boolean | .out: FREQCALC mode table | FREQ | a force-constant matrix is printed |
| `has_frequency_data` | boolean | .out: FREQCALC mode table | FREQ | a FREQ output |
| `has_normal_mode_displacements` | boolean | .out: FREQCALC mode table | FREQ | normal modes are printed |
| `heat_capacity_j_mol_k` | J/(mol·K) | .out: the first THERMODYNAMIC FUNCTIONS block (AT T, P) | FREQ | heat capacity C |
| `heat_capacity_mev_cell_k` | meV/(cell·K) | .out: the first THERMODYNAMIC FUNCTIONS block (AT T, P) | FREQ | heat capacity C |
| `heat_capacity_mhartree_cell_k` | mHartree/(cell·K) | .out: the first THERMODYNAMIC FUNCTIONS block (AT T, P) | FREQ | heat capacity C |
| `highest_frequency_cm` | cm⁻¹ | .out: FREQCALC mode table | FREQ | highest vibration |
| `hyperpolarizability_beta` | JSON | .out: CPHF2 tensors | CPHF, FREQ | first hyperpolarizability β and χ(2) components |
| `lowest_frequency_cm` | cm⁻¹ | .out: FREQCALC mode table | FREQ | lowest vibration, the 3 (molecule: 6, linear: 5) rigid-body modes closest to zero left out; negative for an imaginary mode |
| `num_displaced_calculations` | dimensionless | .out: FREQCALC mode table | FREQ | displaced geometries |
| `num_imaginary_frequencies` | dimensionless | .out: FREQCALC mode table | FREQ | imaginary modes beyond 20 cm⁻¹, rigid-body modes left out |
| `num_raman_modes` | dimensionless | .out: FREQCALC mode table | FREQ | modes in the Raman table |
| `num_vibrational_modes` | dimensionless | .out: FREQCALC mode table | FREQ | printed mode rows |
| `num_vibrational_modes_total` | dimensionless | .out: FREQCALC mode table | FREQ | modes, degenerate ones counted |
| `pv_term_au` | Hartree | .out: the first THERMODYNAMIC FUNCTIONS block (AT T, P) | FREQ | PV |
| `pv_term_ev` | eV | .out: the first THERMODYNAMIC FUNCTIONS block (AT T, P) | FREQ | PV |
| `pv_term_kj_mol` | kJ/mol | .out: the first THERMODYNAMIC FUNCTIONS block (AT T, P) | FREQ | PV |
| `raman_intensities` | JSON | .out: FREQCALC mode table | FREQ | averaged isotropic Raman intensities |
| `refractive_index_electronic` | dimensionless | .out: CPHF/CPKS tensors | CPHF, ELASTIC, FREQ | mean principal refractive index |
| `symmetry_allowed_dof` | dimensionless | .out: FREQCALC mode table | FREQ | symmetry-allowed internal degrees of freedom |
| `thermal_energy_au` | Hartree | .out: the first THERMODYNAMIC FUNCTIONS block (AT T, P) | FREQ | ET, the thermal energy |
| `thermal_energy_ev` | eV | .out: the first THERMODYNAMIC FUNCTIONS block (AT T, P) | FREQ | ET, the thermal energy |
| `thermal_energy_kj_mol` | kJ/mol | .out: the first THERMODYNAMIC FUNCTIONS block (AT T, P) | FREQ | ET, the thermal energy |
| `thermodynamic_pressure_mpa` | MPa | .out: the first THERMODYNAMIC FUNCTIONS block (AT T, P) | FREQ | pressure of the thermodynamics |
| `thermodynamic_temperature_k` | K | .out: the first THERMODYNAMIC FUNCTIONS block (AT T, P) | FREQ | temperature of the thermodynamics |
| `vibrational_frequencies` | cm⁻¹ | .out: FREQCALC mode table | FREQ | modes: frequency, irrep, IR/Raman activity |
| `vibrational_temperatures` | K | .out: FREQCALC mode table | FREQ | vibrational temperature of each mode |
| `zero_point_energy_au` | Hartree | .out: E0 of the harmonic thermodynamics | FREQ | E0, the zero-point energy |
| `zero_point_energy_ev` | eV | .out: E0 of the harmonic thermodynamics | FREQ | E0, the zero-point energy |
| `zero_point_energy_kj_mol` | kJ/mol | .out: E0 of the harmonic thermodynamics | FREQ | E0, the zero-point energy |

#### BAND

| Key | Unit | Source | Calculation types | Meaning |
|---|---|---|---|---|
| `advanced_analysis_error` | category | .out: timings and the run's type | BAND, DOSS | the band analysis failed |
| `advanced_analysis_note` | category | why a BAND.DAT/DOSS.DAT was not analyzed | BAND, DOSS | why a BAND.DAT/DOSS.DAT was not analyzed |
| `band_cbm_kpoint_index` | dimensionless | <run>.BAND.DAT (+ the .out and .d3: k path, Fermi level) | BAND | CBM k point (0-based, per spin) |
| `band_dat_cbm_ev` | eV | <run>.BAND.DAT (+ the .out and .d3: k path, Fermi level) | BAND | conduction band minimum, from E_F (BAND.DAT) |
| `band_dat_exists` | boolean | <run>.BAND.DAT (+ the .out and .d3: k path, Fermi level) | BAND | the BAND.DAT file is there |
| `band_dat_kpath_labels` | JSON | <run>.BAND.DAT (+ the .out and .d3: k path, Fermi level) | BAND | BAND.DAT path labels |
| `band_dat_metallic` | boolean | <run>.BAND.DAT (+ the .out and .d3: k path, Fermi level) | BAND | bands cross E_F (BAND.DAT) |
| `band_dat_num_bands` | dimensionless | <run>.BAND.DAT (+ the .out and .d3: k path, Fermi level) | BAND | bands in BAND.DAT |
| `band_dat_num_kpoints` | dimensionless | <run>.BAND.DAT (+ the .out and .d3: k path, Fermi level) | BAND | k points per spin in BAND.DAT |
| `band_dat_num_panels` | dimensionless | <run>.BAND.DAT (+ the .out and .d3: k path, Fermi level) | BAND | path segments in BAND.DAT |
| `band_dat_num_spins` | dimensionless | <run>.BAND.DAT (+ the .out and .d3: k path, Fermi level) | BAND | spin blocks in BAND.DAT |
| `band_dat_size` | bytes | <run>.BAND.DAT (+ the .out and .d3: k path, Fermi level) | BAND | BAND.DAT size |
| `band_dat_vbm_ev` | eV | <run>.BAND.DAT (+ the .out and .d3: k path, Fermi level) | BAND | valence band maximum, from E_F (BAND.DAT) |
| `band_direct_gap_ev` | eV | <run>.BAND.DAT (+ the .out and .d3: k path, Fermi level) | BAND | smallest vertical gap along the path |
| `band_end` | dimensionless | <run>.BAND.DAT (+ the .out and .d3: k path, Fermi level) | BAND | last band written |
| `band_file_analyzed` | category | <run>.BAND.DAT (+ the .out and .d3: k path, Fermi level) | BAND | the BAND.DAT the band analysis read |
| `band_fundamental_gap_ev` | eV | <run>.BAND.DAT (+ the .out and .d3: k path, Fermi level) | BAND | the gap along the band path: CBM - VBM over every k point and spin |
| `band_gap_is_direct` | boolean | <run>.BAND.DAT (+ the .out and .d3: k path, Fermi level) | BAND | VBM and CBM at one k point |
| `band_kpath_label_string` | category | <run>.BAND.DAT (+ the .out and .d3: k path, Fermi level) | BAND | path labels, joined by - |
| `band_start` | dimensionless | <run>.BAND.DAT (+ the .out and .d3: k path, Fermi level) | BAND | first band written, when it is not band 1 |
| `band_structure` | JSON | <run>.BAND.DAT (+ the .out and .d3: k path, Fermi level) | BAND | k path, labels, band energies at the high-symmetry points, band edges |
| `band_structure_k_points` | dimensionless | <run>.BAND.DAT (+ the .out and .d3: k path, Fermi level) | BAND | BAND.DAT rows read by the band analysis (all spins) |
| `band_structure_k_range_max` | bohr⁻¹ | <run>.BAND.DAT (+ the .out and .d3: k path, Fermi level) | BAND | length of the k path |
| `band_vbm_kpoint_index` | dimensionless | <run>.BAND.DAT (+ the .out and .d3: k path, Fermi level) | BAND | VBM k point (0-based, per spin) |
| `dos_mean` | states/(Hartree·cell) | <run>.DOSS.DAT (+ the .out: projections, Fermi level) | BAND, DOSS | mean of the nonzero DOS |
| `dos_threshold` | states/(Hartree·cell) | <run>.DOSS.DAT (+ the .out: projections, Fermi level) | BAND, DOSS | the DOS at E_F above which the band analysis calls a gapless system a metal (0.05 states/eV/atom, per cell) |
| `doss_file_analyzed` | category | <run>.DOSS.DAT (+ the .out: projections, Fermi level) | BAND, DOSS | the DOSS.DAT the band analysis read |
| `effective_mass_calculation_method` | category | <run>.BAND.DAT (+ the .out and .d3: k path, Fermi level) | BAND | how the effective masses were found |
| `fermi_energy_note` | category | mace.utils.fermi_level, when the .out prints CRYSTAL's -1.0 placeholder: the band data's derived E_F, else the DOSS.DAT's | BAND, DOSS | where fermi_energy_au is from when the .out prints CRYSTAL's -1.0 placeholder (an SCF that ended with its spins locked, SPINLOCK): derived from the bands (the top of the occupied ones), the DOSS.DAT's, or why there is none |
| `has_band_structure` | boolean | <run>.BAND.DAT (+ the .out and .d3: k path, Fermi level) | BAND | a BAND output |
| `real_average_effective_mass` | m_e | <run>.BAND.DAT (+ the .out and .d3: k path, Fermi level) | BAND | quick estimate of the average effective mass: the curvature of one band along the k-path direction at the band edge (geometric mean of electron and hole). Use effective_mass_*_mc / _md (accurate: every degenerate band and valley, several directions, R² checked) |
| `real_electron_effective_mass` | m_e | <run>.BAND.DAT (+ the .out and .d3: k path, Fermi level) | BAND | quick estimate of the electron effective mass: the curvature of one band along the k-path direction at the band edge. Use effective_mass_*_mc / _md (accurate: every degenerate band and valley, several directions, R² checked) |
| `real_hole_effective_mass` | m_e | <run>.BAND.DAT (+ the .out and .d3: k path, Fermi level) | BAND | quick estimate of the hole effective mass: the curvature of one band along the k-path direction at the band edge. Use effective_mass_*_mc / _md (accurate: every degenerate band and valley, several directions, R² checked) |
| `total_bands` | dimensionless | <run>.BAND.DAT (+ the .out and .d3: k path, Fermi level) | BAND | bands written |
| `total_kpoints` | dimensionless | <run>.BAND.DAT (+ the .out and .d3: k path, Fermi level) | BAND | k points along the path (.out) |

#### DOSS

| Key | Unit | Source | Calculation types | Meaning |
|---|---|---|---|---|
| `dos_at_fermi_alpha` | states/eV | <run>.DOSS.DAT (+ the .out: projections, Fermi level) | DOSS | DOS at E_F, alpha channel |
| `dos_at_fermi_beta` | states/eV | <run>.DOSS.DAT (+ the .out: projections, Fermi level) | DOSS | DOS at E_F, beta channel |
| `dos_at_fermi_total` | states/eV | <run>.DOSS.DAT (+ the .out: projections, Fermi level) | DOSS | DOS at E_F, both spins (per cell) |
| `dos_band_gap_from_dos_ev` | eV | <run>.DOSS.DAT (+ the .out: projections, Fermi level) | DOSS | the empty DOS window around E_F (an estimate, within the grid step) |
| `dos_integrated_states_to_fermi` | electrons | <run>.DOSS.DAT (+ the .out: projections, Fermi level) | DOSS | states per cell integrated up to E_F |
| `dos_is_metallic` | boolean | <run>.DOSS.DAT (+ the .out: projections, Fermi level) | DOSS | states at E_F (DOSS.DAT) |
| `dos_num_projections` | dimensionless | <run>.DOSS.DAT (+ the .out: projections, Fermi level) | DOSS | projected DOS columns |
| `dos_structure` | JSON | <run>.DOSS.DAT (+ the .out: projections, Fermi level) | DOSS | downsampled total DOS per spin, band edges, projection weights |
| `doss_dat_energy_max` | Hartree | <run>.DOSS.DAT (+ the .out: projections, Fermi level) | DOSS | highest DOSS.DAT energy, from E_F |
| `doss_dat_energy_min` | Hartree | <run>.DOSS.DAT (+ the .out: projections, Fermi level) | DOSS | lowest DOSS.DAT energy, from E_F |
| `doss_dat_exists` | boolean | <run>.DOSS.DAT (+ the .out: projections, Fermi level) | DOSS | the DOSS.DAT file is there |
| `doss_dat_num_energy_points` | dimensionless | <run>.DOSS.DAT (+ the .out: projections, Fermi level) | DOSS | energy points per spin |
| `doss_dat_num_spins` | dimensionless | <run>.DOSS.DAT (+ the .out: projections, Fermi level) | DOSS | spin blocks in DOSS.DAT |
| `doss_dat_size` | bytes | <run>.DOSS.DAT (+ the .out: projections, Fermi level) | DOSS | DOSS.DAT size |
| `doss_reach_note` | category | <run>.DOSS.DAT (+ the .out: projections, Fermi level) | DOSS | why a DOS with its bands by index (a 1D material's default) stops too low: its DOSS.DAT ends below E_F + 15 eV, and --doss-conduction-bands adds conduction bands, mace.utils.doss_reach; absent when it does not |
| `has_dos` | boolean | <run>.DOSS.DAT (+ the .out: projections, Fermi level) | DOSS | a DOSS output |

#### TRANSPORT

| Key | Unit | Source | Calculation types | Meaning |
|---|---|---|---|---|
| `has_transport` | boolean | SEEBECK.dat, SIGMA.dat, KAPPA.dat (+ the .out: RELAXTIM) | TRANSPORT | a TRANSPORT (BOLTZTRA) output |
| `transport` | JSON | SEEBECK.dat, SIGMA.dat, KAPPA.dat (+ the .out: RELAXTIM) | TRANSPORT | transport summary: peaks, their T and mu, units, files |
| `transport_carrier_type` | category | SEEBECK.dat, SIGMA.dat, KAPPA.dat (+ the .out: RELAXTIM) | TRANSPORT | p-type or n-type at the \|S\| peak |
| `transport_data_available` | boolean | SEEBECK.dat, SIGMA.dat, KAPPA.dat (+ the .out: RELAXTIM) | TRANSPORT | usable SEEBECK.dat data |
| `transport_kappa0_at_zt_max` | W/(m·K) | SEEBECK.dat, SIGMA.dat, KAPPA.dat (+ the .out: RELAXTIM) | TRANSPORT | κ₀ (KAPPA.dat K coefficient) at the transport_zt_electronic_max_avg peak |
| `transport_kappa_electronic_at_zt_max` | W/(m·K) | SEEBECK.dat, SIGMA.dat, KAPPA.dat (+ the .out: RELAXTIM) | TRANSPORT | κₑ = κ₀ − TσS² at the transport_zt_electronic_max_avg peak |
| `transport_mesh_note` | category | SEEBECK.dat, SIGMA.dat, KAPPA.dat (+ the .out: RELAXTIM) | TRANSPORT | why the run's k mesh looks too coarse for transport (a mesh well below the Standard one for its cell, a band Lorenz number far from Sommerfeld's), mace.utils.transport_mesh; absent when it does not |
| `transport_n_data_points` | dimensionless | SEEBECK.dat, SIGMA.dat, KAPPA.dat (+ the .out: RELAXTIM) | TRANSPORT | rows in SEEBECK.dat |
| `transport_n_temperatures` | dimensionless | SEEBECK.dat, SIGMA.dat, KAPPA.dat (+ the .out: RELAXTIM) | TRANSPORT | temperatures in SEEBECK.dat |
| `transport_power_factor_best_axis` | category | SEEBECK.dat, SIGMA.dat, KAPPA.dat (+ the .out: RELAXTIM) | TRANSPORT | the axis of the highest PF (in plane for a slab) |
| `transport_power_factor_max_avg` | W/(m·K²) | SEEBECK.dat, SIGMA.dat, KAPPA.dat (+ the .out: RELAXTIM) | TRANSPORT | peak PF = S²σ of the averaged tensors (an average, not an axis) |
| `transport_power_factor_max_xx` | W/(m·K²) | SEEBECK.dat, SIGMA.dat, KAPPA.dat (+ the .out: RELAXTIM) | TRANSPORT | peak PF = S_xx²σ_xx along xx |
| `transport_power_factor_max_yy` | W/(m·K²) | SEEBECK.dat, SIGMA.dat, KAPPA.dat (+ the .out: RELAXTIM) | TRANSPORT | peak PF = S_yy²σ_yy along yy |
| `transport_power_factor_max_zz` | W/(m·K²) | SEEBECK.dat, SIGMA.dat, KAPPA.dat (+ the .out: RELAXTIM) | TRANSPORT | peak PF = S_zz²σ_zz along zz |
| `transport_seebeck_max_uv_per_k` | μV/K | SEEBECK.dat, SIGMA.dat, KAPPA.dat (+ the .out: RELAXTIM) | TRANSPORT | peak \|S\| of the averaged tensor diagonal |
| `transport_tensor_average` | category | SEEBECK.dat, SIGMA.dat, KAPPA.dat (+ the .out: RELAXTIM) | TRANSPORT | how the *_avg values average the tensors: (xx+yy+zz)/3 of each of S, sigma and kappa0, or (xx+yy)/2 in plane for a slab |
| `transport_zt_electronic_best_axis` | category | SEEBECK.dat, SIGMA.dat, KAPPA.dat (+ the .out: RELAXTIM) | TRANSPORT | the axis of the highest ZTₑ (in plane for a slab) |
| `transport_zt_electronic_max_avg` | dimensionless | SEEBECK.dat, SIGMA.dat, KAPPA.dat (+ the .out: RELAXTIM) | TRANSPORT | peak ZTₑ of the averaged tensors (an average, not an axis; electrons only, no lattice κ) |
| `transport_zt_electronic_max_xx` | dimensionless | SEEBECK.dat, SIGMA.dat, KAPPA.dat (+ the .out: RELAXTIM) | TRANSPORT | peak ZTₑ along xx (electrons only, no lattice κ) |
| `transport_zt_electronic_max_yy` | dimensionless | SEEBECK.dat, SIGMA.dat, KAPPA.dat (+ the .out: RELAXTIM) | TRANSPORT | peak ZTₑ along yy (electrons only, no lattice κ) |
| `transport_zt_electronic_max_zz` | dimensionless | SEEBECK.dat, SIGMA.dat, KAPPA.dat (+ the .out: RELAXTIM) | TRANSPORT | peak ZTₑ along zz (electrons only, no lattice κ) |
| `transport_zt_total` | JSON | computed with --kappa-lattice (and --tau-fs): S²σT/(κe + κL) | TRANSPORT | total ZT details: formula, κ_L, τ, peaks per axis |
| `transport_zt_total_best_axis` | category | computed with --kappa-lattice (and --tau-fs): S²σT/(κe + κL) | TRANSPORT | the axis of the highest total ZT (in plane for a slab) |
| `transport_zt_total_inputs` | category | computed with --kappa-lattice (and --tau-fs): S²σT/(κe + κL) | TRANSPORT | fingerprint of the .out and .dat files the total ZT is from |
| `transport_zt_total_kappa_electronic_at_max` | W/(m·K) | computed with --kappa-lattice (and --tau-fs): S²σT/(κe + κL) | TRANSPORT | κₑ at the transport_zt_total_max_avg peak (at τ) |
| `transport_zt_total_kappa_lattice_at_max` | W/(m·K) | computed with --kappa-lattice (and --tau-fs): S²σT/(κe + κL) | TRANSPORT | κ_L at the transport_zt_total_max_avg peak |
| `transport_zt_total_lattice_source` | category | computed with --kappa-lattice (and --tau-fs): S²σT/(κe + κL) | TRANSPORT | where κ_L came from |
| `transport_zt_total_max_avg` | dimensionless | computed with --kappa-lattice (and --tau-fs): S²σT/(κe + κL) | TRANSPORT | peak total ZT = S²σT/(κₑ + κ_L) of the averaged tensors, with the κ_L given to --kappa-lattice |
| `transport_zt_total_max_mu_ev` | eV | computed with --kappa-lattice (and --tau-fs): S²σT/(κe + κL) | TRANSPORT | mu of transport_zt_total_max_avg |
| `transport_zt_total_max_temperature_k` | K | computed with --kappa-lattice (and --tau-fs): S²σT/(κe + κL) | TRANSPORT | T of transport_zt_total_max_avg |
| `transport_zt_total_max_xx` | dimensionless | computed with --kappa-lattice (and --tau-fs): S²σT/(κe + κL) | TRANSPORT | peak total ZT along xx with the κ_L given |
| `transport_zt_total_max_yy` | dimensionless | computed with --kappa-lattice (and --tau-fs): S²σT/(κe + κL) | TRANSPORT | peak total ZT along yy with the κ_L given |
| `transport_zt_total_max_zz` | dimensionless | computed with --kappa-lattice (and --tau-fs): S²σT/(κe + κL) | TRANSPORT | peak total ZT along zz with the κ_L given |
| `transport_zt_total_tau_fs` | fs | computed with --kappa-lattice (and --tau-fs): S²σT/(κe + κL) | TRANSPORT | the relaxation time the total ZT is at |
| `transport_zt_total_tau_source` | category | computed with --kappa-lattice (and --tau-fs): S²σT/(κe + κL) | TRANSPORT | how τ was chosen |

#### CHARGE+POTENTIAL

| Key | Unit | Source | Calculation types | Meaning |
|---|---|---|---|---|
| `charge_potential` | JSON | .out and the ECH3/POT3 .CUBE files | CHARGE+POTENTIAL | ECH3/POT3 grid metadata and files |
| `chargepot_computed_properties` | category | .out and the ECH3/POT3 .CUBE files | CHARGE+POTENTIAL | what the grids hold |
| `chargepot_grid_dims` | category | .out and the ECH3/POT3 .CUBE files | CHARGE+POTENTIAL | grid points along each axis |
| `chargepot_grid_files` | category | .out and the ECH3/POT3 .CUBE files | CHARGE+POTENTIAL | the .CUBE files |
| `chargepot_grid_points` | dimensionless | .out and the ECH3/POT3 .CUBE files | CHARGE+POTENTIAL | grid points |
| `chargepot_has_ech3` | boolean | .out and the ECH3/POT3 .CUBE files | CHARGE+POTENTIAL | a charge density grid (ECH3) |
| `chargepot_has_pot3` | boolean | .out and the ECH3/POT3 .CUBE files | CHARGE+POTENTIAL | a potential grid (POT3) |
| `chargepot_num_grid_files` | dimensionless | .out and the ECH3/POT3 .CUBE files | CHARGE+POTENTIAL | number of .CUBE files |
| `chargepot_scf_total_energy_au` | Hartree | .out and the ECH3/POT3 .CUBE files | CHARGE+POTENTIAL | SCF energy the grids are of |

#### ORBITALS

| Key | Unit | Source | Calculation types | Meaning |
|---|---|---|---|---|
| `orbitals` | JSON | the ORBITALS .molden files (+ the .out) | ORBITALS | the ORBITALS index: files, k points, levels, HOMO/LUMO |
| `orbitals_dimensionality` | dimensionless | the ORBITALS .molden files (+ the .out) | ORBITALS | dimensionality of the run |
| `orbitals_files` | category | the ORBITALS .molden files (+ the .out) | ORBITALS | the Molden files |
| `orbitals_gap_ev` | eV | the ORBITALS .molden files (+ the .out) | ORBITALS | gap of the .out's band edges |
| `orbitals_homo_band` | dimensionless | the ORBITALS .molden files (+ the .out) | ORBITALS | HOMO level |
| `orbitals_homo_energy_au` | Hartree | the ORBITALS .molden files (+ the .out) | ORBITALS | HOMO energy |
| `orbitals_kpoints` | category | the ORBITALS .molden files (+ the .out) | ORBITALS | k points, their coordinates and kind |
| `orbitals_lumo_band` | dimensionless | the ORBITALS .molden files (+ the .out) | ORBITALS | level above the HOMO at its k point and spin |
| `orbitals_lumo_energy_au` | Hartree | the ORBITALS .molden files (+ the .out) | ORBITALS | energy of orbitals_lumo_band |
| `orbitals_nbf` | dimensionless | the ORBITALS .molden files (+ the .out) | ORBITALS | basis functions |
| `orbitals_num_files` | dimensionless | the ORBITALS .molden files (+ the .out) | ORBITALS | Molden files |
| `orbitals_num_kpoints` | dimensionless | the ORBITALS .molden files (+ the .out) | ORBITALS | k points written |
| `orbitals_self_check` | category | the ORBITALS .molden files (+ the .out) | ORBITALS | the worst Molden self-check |
| `orbitals_spin_polarized` | boolean | the ORBITALS .molden files (+ the .out) | ORBITALS | the Molden files hold alpha and beta orbitals |
| `orbitals_total_bytes` | bytes | the ORBITALS .molden files (+ the .out) | ORBITALS | size of the Molden files |
| `orbitals_usable_kpoints` | dimensionless | the ORBITALS .molden files (+ the .out) | ORBITALS | k points MACE can draw |

#### ELASTIC

| Key | Unit | Source | Calculation types | Meaning |
|---|---|---|---|---|
| `bulk_modulus_hill_gpa` | GPa | .out (+ ELASINFO/EOSINFO.DAT) via mace.elastic.analysis | ELASTIC, EOS, QHA | bulk modulus, Hill average |
| `bulk_modulus_reuss_gpa` | GPa | .out (+ ELASINFO/EOSINFO.DAT) via mace.elastic.analysis | ELASTIC, EOS, QHA | bulk modulus, Reuss bound |
| `bulk_modulus_voigt_gpa` | GPa | .out (+ ELASINFO/EOSINFO.DAT) via mace.elastic.analysis | ELASTIC, EOS, QHA | bulk modulus, Voigt bound |
| `debye_temperature_elastic_k` | K | .out (+ ELASINFO/EOSINFO.DAT) via mace.elastic.analysis | ELASTIC, EOS, QHA | Debye temperature from the mean sound velocity |
| `elastic_2d_moduli` | JSON | .out (+ ELASINFO/EOSINFO.DAT) via mace.elastic.analysis | ELASTIC, EOS, QHA | in-plane moduli of a slab (E_x, E_y, ν_xy, G_xy) |
| `elastic_anisotropy_universal` | dimensionless | .out (+ ELASINFO/EOSINFO.DAT) via mace.elastic.analysis | ELASTIC, EOS, QHA | universal anisotropy index A^U (0 = isotropic) |
| `elastic_born_stable` | boolean | .out (+ ELASINFO/EOSINFO.DAT) via mace.elastic.analysis | ELASTIC, EOS, QHA | C is positive definite (mechanically stable) |
| `elastic_compliance_tensor` | JSON | .out (+ ELASINFO/EOSINFO.DAT) via mace.elastic.analysis | ELASTIC, EOS, QHA | compliance tensor S = C⁻¹ (TPa⁻¹) |
| `elastic_min_eigenvalue_gpa` | GPa | .out (+ ELASINFO/EOSINFO.DAT) via mace.elastic.analysis | ELASTIC, EOS, QHA | smallest eigenvalue of C (Born stability) |
| `elastic_seismic_velocities` | JSON | .out (+ ELASINFO/EOSINFO.DAT) via mace.elastic.analysis | ELASTIC, EOS, QHA | Christoffel wave velocities Vp, Vs1, Vs2 (km/s) per direction |
| `elastic_tensor` | JSON | .out (+ ELASINFO/EOSINFO.DAT) via mace.elastic.analysis | ELASTIC, EOS, QHA | stiffness tensor C (Voigt 6x6) as printed: matrix, unit, system |
| `photoelastic_tensor` | JSON | .out (+ ELASINFO/EOSINFO.DAT) via mace.elastic.analysis | ELASTIC, EOS, QHA | elasto-optic (Pockels) tensor p (6x6) |
| `piezo_optic_tensor` | JSON | .out (+ ELASINFO/EOSINFO.DAT) via mace.elastic.analysis | ELASTIC, EOS, QHA | piezo-optic tensor π (6x6, Brewster) |
| `piezoelectric_d_tensor` | JSON | .out (+ ELASINFO/EOSINFO.DAT) via mace.elastic.analysis | ELASTIC, EOS, QHA | converse piezoelectric tensor d (3x6, pC/N) |
| `piezoelectric_e_tensor` | JSON | .out (+ ELASINFO/EOSINFO.DAT) via mace.elastic.analysis | ELASTIC, EOS, QHA | direct piezoelectric tensor e (3x6, C/m²) |
| `poisson_ratio` | dimensionless | .out (+ ELASINFO/EOSINFO.DAT) via mace.elastic.analysis | ELASTIC, EOS, QHA | Poisson's ratio from the Hill moduli |
| `shear_modulus_hill_gpa` | GPa | .out (+ ELASINFO/EOSINFO.DAT) via mace.elastic.analysis | ELASTIC, EOS, QHA | shear modulus, Hill average |
| `shear_modulus_reuss_gpa` | GPa | .out (+ ELASINFO/EOSINFO.DAT) via mace.elastic.analysis | ELASTIC, EOS, QHA | shear modulus, Reuss bound |
| `shear_modulus_voigt_gpa` | GPa | .out (+ ELASINFO/EOSINFO.DAT) via mace.elastic.analysis | ELASTIC, EOS, QHA | shear modulus, Voigt bound |
| `sound_velocity_mean_km_s` | km/s | .out (+ ELASINFO/EOSINFO.DAT) via mace.elastic.analysis | ELASTIC, EOS, QHA | mean sound velocity from the Hill moduli |
| `youngs_modulus_gpa` | GPa | .out (+ ELASINFO/EOSINFO.DAT) via mace.elastic.analysis | ELASTIC, EOS, QHA | Young's modulus from the Hill moduli |

#### EOS

| Key | Unit | Source | Calculation types | Meaning |
|---|---|---|---|---|
| `eos_bulk_modulus_gpa` | GPa | .out (+ ELASINFO/EOSINFO.DAT) via mace.elastic.analysis | EOS, QHA | B0 of the Birch-Murnaghan EOS fit |
| `eos_bulk_modulus_prime` | dimensionless | .out (+ ELASINFO/EOSINFO.DAT) via mace.elastic.analysis | EOS, QHA | B0′ of the Birch-Murnaghan EOS fit |
| `eos_equilibrium_volume_ang3` | Å³ | .out (+ ELASINFO/EOSINFO.DAT) via mace.elastic.analysis | EOS, QHA | V0 of the Birch-Murnaghan EOS fit |
| `eos_fits` | JSON | .out (+ ELASINFO/EOSINFO.DAT) via mace.elastic.analysis | EOS, QHA | V0, E0, B0, B0′ of each equation of state |
| `eos_points` | JSON | .out (+ ELASINFO/EOSINFO.DAT) via mace.elastic.analysis | EOS, QHA | the E(V) points the EOS fits use |

#### QHA

| Key | Unit | Source | Calculation types | Meaning |
|---|---|---|---|---|
| `qha_bulk_modulus_300k_gpa` | GPa | .out (+ ELASINFO/EOSINFO.DAT) via mace.elastic.analysis | QHA | QHA isothermal bulk modulus near 300 K |
| `qha_thermal_expansion_300k_per_k` | 1/K | .out (+ ELASINFO/EOSINFO.DAT) via mace.elastic.analysis | QHA | QHA volumetric thermal expansion near 300 K |
| `qha_thermal_properties` | JSON | .out (+ ELASINFO/EOSINFO.DAT) via mace.elastic.analysis | QHA | QHA V(T), K_T(T), α_V(T), Cp−Cv, K_S and lattice expansion |
| `qha_volume_300k_ang3` | Å³ | .out (+ ELASINFO/EOSINFO.DAT) via mace.elastic.analysis | QHA | QHA cell volume at the temperature nearest 300 K |

#### CPHF

| Key | Unit | Source | Calculation types | Meaning |
|---|---|---|---|---|
| `hyperpolarizability_gamma` | JSON | .out: CPHF2/CPHF4 tensors | CPHF | second hyperpolarizability γ and χ(3) components |

<!-- END GENERATED -->
