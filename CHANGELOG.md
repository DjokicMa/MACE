# Changelog

All notable changes to MACE (Mendoza Automated CRYSTAL Engine) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.5] - 2026-10-02

Everything since 1.1.4. New: charged systems and ion clusters (fragments found
by connectivity, CHEMOD/DOPING/spin planned for you, per-fragment Mulliken
charges), `mace defects` (defect deck sets and formation energies, with the 3D
charged-cell offset and image-charge corrections measured on CRYSTAL23),
`mace phonopy` (finite-displacement phonons through phonopy 4.x), and one basis
file per family with an opt-in TZVP-REV2 2022 update and the SOC basis records.
**Band paths from an OPT parent now use the optimised cell** (they used the
initial cell), static band paths compute their lattice-dependent points for the
cell, and long-axis cells keep the band-path ISS at most 999; regenerate BAND
decks made from OPT parents. Also fixed: the bundled Pu basis record, the BAND
gap of doped runs, ECP Mulliken net charges, odd-electron and charged-parent
decks, 1.1.1 plans' functional names, piped answers to interactive runs, and
several menu and job-script fixes. Workflow BAND steps now write the deck
opt2d3 writes for the same choices (NSUB 1000; a default step is opt2d3's
Enter-through label path), opt2d12 children of a
SUPERCEL/SUPERCON parent are the supercell in P1, and the bundled La-Lu and
Ac-Es basis records are neutral. Several of these change generated decks -
see Fixed below. Coming from 1.1.1-1.1.4, read Upgrading first.

### Upgrading from 1.1.1-1.1.4

- **Workflows planned before 1.1.2 produce mixed decks.** The plan's SP
  basis, tolerance and functional overrides and its FREQ settings
  (tolerances, NUMDERIV, IR/Raman) now apply, but only to steps generated
  after the upgrade. Finish such workflows on the old version or re-plan
  them.
- **1.1.1 plans with a custom functional** stored prose names (HSEsol-3C,
  HF-3C, M06-2X, HF, ...). They are now rewritten to the CRYSTAL23 keyword
  when the plan is loaded, with a warning. A -D3 suffix CRYSTAL23 has no
  parameters for is dropped, and an unknown name stops that step. Nothing to
  do unless a step is refused; then re-plan it.
- **`workflow_scripts/` is a snapshot** taken when the workflow is planned.
  A template copied before 1.1.3 that the plan names is now rebuilt from the
  plan (current generator, the plan's resources and directives) when the
  engine uses it; the old copy is kept as `<name>.pre-1.1.3`. A template the
  plan does not name is left alone, with a warning: re-plan, or copy the
  current `mace/submission/submitcrystal23.sh` and `submit_prop.sh` into it,
  to get the empty-`$SCRATCH` guard and `.f20` staging.
- **`mace submit` reuses existing `<job>.sh` files.** Scripts generated
  before 1.1.3 lack the scratch guard and GUESSP staging; regenerate them
  with `mace submit --overwrite-sh` (MACE now warns when it reuses one).
- **Job scripts and `workflow_scripts/` copies written by 1.1.3-1.1.4 are
  reused without a warning** and lack this release's job-script changes:
  two-component GUESSP handling (a 2c job's own `.f9` is no longer staged as
  fort.20), keeping a hand-staged `fort.20`, staging `<job>.ext` as fort.34
  (`mace phonopy` decks) and the `.mace_scratch` cleanup. For TWOCOMPON
  GUESSP restarts, hand-staged densities and phonopy decks, regenerate them
  with `mace submit --overwrite-sh`, or re-plan the workflow.
- **GUESSP from a hand-staged density** can go next to the deck as
  `<job>.f20`, or into the scratch directory as `fort.20`.
- **`<submit dir>/.mace_scratch` is now removed** after a run that ended
  normally. Do not keep files of your own there.
- **Piped answer files must be regenerated.** opt2d12 now asks for the
  integration grid, asks "Keep the basis set from the original D12? [Y/n]"
  before the basis menus when the parent deck has an external (EXTERNAL)
  basis and the functional is not a 3c method (the menus follow only on
  "n"; an internal-basis parent, a 3c functional and --shared-settings get
  the basis menus as before, with no new question), takes LEVSHIFT as one "LEVSHIFT ISHIFT
  ILOCK" answer (was value, then cycles; a single value is now asked again),
  and asks for the MAXTRADIUS value and the HISTDIIS size. opt2d3 ECHG
  offers order 0-1 (was 0-2) and adds "Number of points along the B-A
  segment"; a `--config-file` run without a terminal needs
  `map_points`/`map_coord_type` in the config. The planner's SP "Use tight
  convergence?" and OPT "Use tighter convergence?" questions are 0-3 level
  menus (y/yes and n/no are still taken, with a note). Shifted answers can
  land in other questions. Three invalid answers in a row to any question
  now stop a piped run, and a piped opt2d12 run lists every setting that
  differs from the parent deck, but a shifted answer that is valid for the
  question it lands on cannot be caught; use `--save-options` and
  `--config-file` instead of piping.
- **Planner and quick-start FREQ steps no longer write NUMDERIV 2.** New
  plans use CRYSTAL's default; plans saved before 1.1.2 keep 2. Set NUMDERIV
  2 to compare with older runs.
- **FREQ children of a FREQ parent with IR/Raman intensities.** A piped
  opt2d12 answer file for such a child (the parent's FREQCALC block has
  intensity or spectrum settings) now meets "Select template (0-8) [0]": a
  blank answer keeps the parent's settings (was Basic), "1" still picks
  Basic, and the questions that followed the template are not asked after
  "0". A parent with only NUMDERIV gets the menu as before. In the planner's
  Advanced FREQ level, a blank "Select calculation type" now keeps the
  parent's intensities ("1" is an explicit none).
- **Planner phonon dispersion steps.** "Use Fourier interpolation
  (INTERPHESS)?" now defaults to no (blank was yes), and a band structure
  step stores opt2d12's automatic BANDS path. Plans saved before 1.1.5 keep
  the INTERPHESS 2 2 2 and the deck with no BANDS path they stored; re-plan
  such steps.
- **opt2d12 templates saved by 1.1.1** do not record a "no" to MAXTRADIUS or
  SPINLOCK, so they now pick up the parent's values. Re-save the template
  if that is not wanted.
- **Re-organising sorts previously misfiled SP/FREQ outputs** into
  `completesp/` and `completefreq/` (1.1.1 put them in `complete/`).
  The same scan corrects the `calc_type` of a record an older MACE typed
  from its file name (an SP named `*_OPT_*` recorded as OPT); workflow types
  (OPT2, SP2, ...) are left as they are, and `calc_id` keeps its old name.
- **`mace/database/utils/populate_completed_jobs.py` was removed.** Use
  `mace/database/populate_completed_jobs.py`.
- **Stale property rows.** Re-extracting a calculation now removes the rows
  it no longer produces (1.1.1's band fields on DOSS runs, optimisation rows
  on OPTGEOM-titled SP runs, doped BAND gaps). Rows stored without a
  calc_id cannot be attributed to one output: `mace analyze
  --extract-properties` removes those of a material when it re-extracted
  every output of that material it can see. With `--material-id`, or when
  `--filter-type`/`--calc-type` left one of the material's outputs out, they
  are kept; rebuild the database to clean those.
- **Workflow BAND steps planned before 1.1.5.** Plans store the BAND
  settings they were made with: a planner BAND step keeps the NSUB 10000 it
  stored (new plans and the engine default use opt2d3's 1000), and a step set
  in the planner's Advanced menu keeps keys opt2d3 does not read ("labels"
  writes k-point vectors; literature, SeeK-path and custom paths write a deck
  with no path, NLINE 0). Basic and quick-start BAND steps keep the k-point
  vector or SeeK-path format they stored (new plans and the engine default
  write opt2d3's Enter-through label path), and Basic, Advanced and
  quick-start FREQ steps keep the Very tight tolerances they stored, which
  replace a FREQ parent's own. Re-plan such BAND and FREQ steps.
- **Regenerate opt2d12 children of SUPERCEL/SUPERCON parents** (e.g. defect
  supercells). They were the host's space group with the supercell's cell
  and asymmetric atoms, not the parent's crystal.
- **Regenerate decks with La-Lu on the bundled triplezeta/doublezeta records,
  or Ac-Es on any bundled family.** Their shell charges did not add up to a
  neutral atom, and CRYSTAL stopped with INPBAS "UNIT CELL NOT NEUTRAL".
- **Basis sets are one file per family.** Old `basis_sets/<family>/` paths
  still work, including at the custom-path prompt. Outside tools that read
  the per-element files must use `basis_library.read_record`.

### Added

- **Opt-in TZVP-REV2 2022 update: `full.basis.triplezeta.rev2022`.** The
  bundled triplezeta set carries pob-TZVP 2018 for Rb-I and ERD small-core
  2017 for La-Lu. The new family replaces those with the newer sets from
  crystal.unito.it: X_POB_TZVP_rev2 for Rb-Mo and Ru-I (Laun & Bredow, JCC
  2022, DOI 10.1002/jcc.26839) and X_pob_TZVP_rev2 for La-Lu (Seidler, Laun &
  Bredow, JCC 2023, DOI 10.1002/jcc.27097). Tc keeps the ERD 2017 all-electron
  record, because the site has no Tc_POB_TZVP_rev2. Every other record is
  byte-identical to `full.basis.triplezeta`.
  - **Core treatment.** Rb-I keep the ECPs of pob-TZVP 2018 (the same ZNUC).
    Ce-Lu keep a 28-electron core with 4f in the valence, as ERD has.
    **La changes:** its rev2 ECP has ZNUC 11, a 46-electron core, where ERD
    has ZNUC 29. With this family each La atom carries 18 fewer electrons,
    which shows in the Mulliken `NO. OF ELECTRONS`.
  - **How to choose it.** Pick external basis menu option `4: TZVP-REV2 (2022
    update: Rb-I, La-Lu)`, or put `"external_basis_family": "tz-rev2022"` in a
    configuration file. In code, use `mace_config.DEFAULT_TZ_REV2022_PATH` or
    `get_basis_path("tz-rev2022")`. Options 1-3 are unchanged and 2 is still
    the default. Decks are named `..._full.basis.triplezeta.rev2022.d12`, and
    the workflow executor keeps such a child on this family even when
    `MACE_TZ_BASIS_PATH` or `MACE_BASIS_DIR` is set.
  - **Provenance.** Each record keeps its URL, retrieval date, page checksum,
    DOI as printed and core. `basis_sets/fetch_unito.py` and `build_library.py
    --fetch` rebuild the family.
  - **Status.** Untested in CRYSTAL. The Rb-I records should reproduce
    CRYSTAL23's internal `BASISSET POB-TZVP-REV2` energies exactly.
- **POB-TZVP-REV2 SOC basis records (data only).** X_pob_TZVP_rev2_SOC for
  Rb-Mo, Ru-I, Cs, Ba and Hf-Bi, from crystal.unito.it. Each is the rev2
  valence shells (checked numerically) on CRYSTAL's internal STUTSC library
  (manual p.84), for two-component SOC runs (pp.169-175).
  - Read them with `basis_library.soc_basis_record(z)`. They are never offered
    as an ordinary external basis: there is no menu entry, path or
    configuration key for them.
  - Untested in CRYSTAL.
- **Charged systems and ion clusters** (opt-in; see "Charged systems and ion
  clusters" in `Crystal_d12/README.md`). `cif2d12` (`--ion-charges auto`,
  `--net-charge`, `--site-states`, `--chemod`, `--fragment-bond-scale`,
  `--spinlock`, `--spinlock-cycles`, or the same keys in an options/config
  file), `opt2d12` (the same flags, recomputed from the optimised geometry)
  and a new `mace ions` command for existing decks write `CHEMOD` (starting
  shell charges), `DOPING` (the net charge), `CHARGED` and, for an odd
  electron count, `SPIN` + `SPINLOCK`. Ions are found by connectivity
  (covalent radii), so atom order does not matter: Li+, Na+, K+, Mg2+, Ca2+,
  Zn2+, FSI-, TFSI-, OTf-, ClO4-, NO3-, PF6-, BF4-; anything else is neutral,
  except that a molecule that cannot be closed-shell neutral by its bonds
  (CO3, SO4, OH, a carboxylate: an anion outside the table) stops auto mode
  until `net_charge` is given, so Li2CO3 is not written with `DOPING -2`.
  On the 65 Li/FSI/solvent clusters of the test corpus it finds the same
  ions in any atom order (the group's script assumed a fixed order and left
  two clusters without records). The records follow what was measured on
  CRYSTAL23: with an internal BASISSET the count comes from `DOPING` and
  `CHEMOD` only shapes the guess; with an explicit basis a non-neutral
  `CHEMOD` + `CHARGED` sets the count and `DOPING` is not added (both
  together counted the charge twice). An odd count in a molecule is never
  written closed shell (a periodic cell may be, with a warning: a half-filled
  band); RHF/HF-3c and `GHOSTS` + `DOPING` with an odd count are refused; an
  odd-electron open-shell deck with an HSE-type screened hybrid gets a
  warning in every dimensionality, also in opt2d12 children (measured:
  HSEsol-3c, HSE06 and HSEsol crash at SCF cycle 1-3 on any odd-electron
  open-shell system, 0D and 3D, while PBE0/PBEsol0 converge; the functional
  is never changed); and
  `CHEMOD` needs a deck without symmetry (CRYSTAL stops: "SHELL CHARGES NOT
  COMPATIBLE WITH SYMMETRY"). Shell charges for SOLDEF2MSVP, POB-TZVP-REV2
  and POB-DZVP-REV2 (H B C N O F Na Mg P S Cl K Ca Zn Li) were measured on
  HPCC; other internal libraries need explicit `site_states`. New example
  config `ion_cluster_molecule.json` (P1, every atom written, so a cluster
  CIF that declares a space group is not reduced).
- **Per-fragment Mulliken charges of charged runs.** Property extraction adds
  `electron_count` and `fragment_charges` (each molecule/ion's charge, spin,
  formal charge and deviation, an ion more than 0.5 e from its formal charge
  flagged) when a run was charged (Mulliken count different from the sum of
  valence charges, a renormalised guess other than after a GUESSP restart,
  or "UNIT CELL NOT NEUTRAL"), and the
  extractor prints them as a Fragments section;
  `mace analyze --extract-properties <dir> --fragments` (or
  `MACE_FRAGMENT_CHARGES=1`) computes them for every run. Neutral runs get
  no new properties otherwise.
- **`mace defects`: point-defect deck sets and their analysis (new command;
  nothing existing changes).** `mace defects build HOST.d12` writes a pristine
  supercell deck (SUPERCEL/SUPERCON, 3D or 2D including CRYSTAL + SLABCUT
  hosts) and one single-point deck per charge state for a vacancy (ATOMREMO),
  ghost vacancy (GHOSTS), substitution (ATOMSUBS) or interstitial (ATOMINSE,
  untested on HPCC), by editing the host deck's text: method, basis, grids and
  tolerances stay byte for byte. Charge is `DOPING = -q`; odd electron counts
  (or `--spin-state q=NSPIN`, parity-checked, p.128) get SPIN/UHF and
  `SPINLOCK NSPIN 30`, and every choice is printed. The records sit where the
  decks run on HPCC had them, and those decks (MgO V_O q=0/+1/+2, ghost q=0/+2,
  the explicit-basis odd-ghost recipe, fractional-DOPING decks, hBN V_N 3x3 to
  6x6) are reproduced byte for byte. Refused with the reason: OPT/FREQ hosts
  (build from an SP), KEEPSYMM in block 1, charged hosts, host spin locks
  (unless `--host-spin keep`), odd ghost states with an internal basis
  (measured to abort or collapse), GHOSTS on ECP atoms (p.82), and more (see
  `docs/defects/README.md`). Supercell labels come from CRYSTAL's output: a set
  without one writes the pristine deck and finishes with `--resume DIR` as soon
  as the running pristine output shows its geometry. `mace defects potc DIR`
  stages POTC runs (with copies of the `.f9`) into `DIR/potc`. `mace defects
  analyze DIR` checks the Mulliken electron count (DOPING applied) and the
  final spin, and reports formation energies and transition levels, all
  unrelaxed. 3D charged states get the offset measured on CRYSTAL23 (MgO, LiF,
  Si): `E(q) + q*(C(q)+C(0))/2` with `C = 2*E_sph/N` from the output's
  `EXT EL-SPHEROPOLE` term and N = header electron count + DOPING, and, with
  `--eps VALUE` or `--eps-from` a CPHF (eps_inf) or FREQCALC INTCPHF output
  (`--eps-static`: eps_0), the Lany-Zunger-scaled Makov-Payne image-charge
  term with the Ewald Madelung potential of the actual supercell. On the
  measured MgO V_O runs (53 to 249 atoms, eps_inf 2.865) this gives eps(+/0)
  2.85-2.88 eV and eps(2+/+) -0.09 to -0.19 eV. Without a dielectric constant
  charged states are labelled "uncorrected for image charges"; every report
  names the corrections it applied. `--calibrate` adds restricted
  fractional-DOPING decks as an optional cross-check (`analyze --offset
  calibration` uses them instead). 2D sets use an A + B/L fit over the sizes.
  Energies are SCF + D3 + gCP for -D3 and 3c methods (CRYSTAL's
  `TOTAL ENERGY + DISP + GCP` line, not the bare `SCF ENDED` value): those
  terms do not cancel between a defect cell and the pristine cell, so chemical
  potentials must include them too, and a job whose terms differ from the
  pristine's is excluded. Potential alignment is reported, and applied only
  with `--apply-alignment` (measured to worsen the agreement between charged
  cells of different size). A worked MgO V_O deck set is in
  `docs/defects/MgO_VO` (not run).
- **`mace phonopy generate` / `mace phonopy collect`: finite-displacement phonons with
  phonopy** (optional dependency, `phonopy<5`). `generate` reads a finished 3D OPT or
  SP parent and writes one EXTERNAL deck per displaced supercell, named after the
  parent and the supercell, with its geometry in `<job>.ext`. The deck has the layout
  phonopy's own CRYSTAL writer gives it, measured to give forces on CRYSTAL23: the
  parent's basis and SCF block (phonopy's TEMPLATE, also written to `TEMPLATE` for an
  explicit basis; never with an END, which stops CRYSTAL before GRADCAL with no
  forces), then TOLDEE (at least 10), GRADCAL and END. The basis and SCF block are
  kept byte for byte apart from the printed edits (GUESSP dropped, buffers for large
  cells, SHRINK divided by the supercell size - rounded up where it does not divide -
  unless `--keep-shrink` or a non-diagonal `--dim`). `--cry-sym` (opt-in) writes each
  displaced cell's symmetry into its `.ext` as phonopy's CRY_SYM does - the same
  frequencies for diamond at about a third of the core hours, measured on CRYSTAL23 -
  with phonopy's caveat that some space groups (e.g. R-3m) do not work. Parents it
  cannot reproduce in a P1 supercell (non-3D, charged, spin-locked, magnetic without a
  lock, ghosts, unknown SCF keywords, unconverged) are refused with the reason.
  `collect` checks every output - including that CRYSTAL ran the displaced geometry it
  was given - names any output whose forces phonopy cannot read, and stops unless
  FORCE_SETS was really written (phonopy's `phonopy-init -f` exits 0 when it fails)
  before writing band, mesh, DOS and thermal properties, with optional non-analytical
  correction from a Gamma FREQCALC with INTENS + INTCPHF. Submit the decks with a
  plain `mace submit DIR`, never `--track`, with the walltime `generate` prints: it
  grows with k-points x atoms^3 and carries a 1.5x margin (3 h for a 2x2x2 hybrid-DFT
  diamond, 30 h for 3x3x3, which measured on CRYSTAL23 did not finish its SCF in 2 h).
- The CRYSTAL job script copies `<job>.ext` to `fort.34` when that file exists next to
  the deck (EXTERNAL geometry). No existing deck has one, so their jobs are unchanged.
- `mace preflight` checks an EXTERNAL deck when its geometry sits next to it as
  `<stem>.ext`, staging it as fort.34 the way the job script does.

### Changed

- **The bundled external basis sets are one file per family.**
  - `Crystal_d12/basis_sets/full.basis.triplezeta/`, `full.basis.doublezeta/`
    and `stuttgart/` (about 300 per-element files) are now
    `full.basis.triplezeta.lib`, `full.basis.doublezeta.lib` and
    `stuttgart.lib`, read by `Crystal_d12/basis_library.py`.
  - The old directory paths still work everywhere MACE takes a basis path,
    including `mace_config.DEFAULT_TZ_PATH`, configuration files that name
    them and the custom-path prompt (menu option 3). Any other directory is
    still read from disk.
  - Every record is returned byte for byte: tests pin all records and the
    decks written from them. No deck changes, apart from the Pu, La-Lu and
    Ac-Es fixes below; five stuttgart records CRYSTAL cannot read are refused.
  - **Custom tooling** that opened `basis_sets/<family>/<n>` directly should
    call `basis_library.read_record(path, n)` instead, or
    `basis_library.load_family(name)`.
  - Each record now states where it came from (`basis_sets/PROVENANCE.md`).
    The records were matched exactly against crystal.unito.it and checked
    against basissetexchange.org: H-Br, Cs, Ba and Hf-Bi are pob-TZVP-rev2,
    Rb-I are pob-TZVP 2018, and DZ H-Br and Rb-I are pob-DZVP-rev2 and
    pob-DZVP 2018. Unmatched records say "unverified".
  - `python mace_config.py` (`ensure_paths_exist`) reports the three `.lib`
    files under the same names. This is diagnostic output only.
- **Workflow BAND steps use NSUB 1000, as opt2d3** (engine default, planner
  Basic and Advanced, quick-start), instead of 10000. DOSS keeps its 10000
  energy points. Deck change: NSUB only.
- **Default workflow BAND steps write opt2d3's Enter-through deck**: the
  automatic path as high-symmetry labels (format 1, ISS 0). The engine
  default and the planner's Basic BAND wrote k-point vectors, and
  quick-start / `--progress` the SeeK-path vectors, so the same parent got a
  different path, title and NLINE record by route; the decks now match opt2d3
  byte for byte (checked on 13 lattices and a slab). Deck change: path
  records, NLINE, ISS and title. Vectors, literature and SeeK-path paths are
  formats 2-4 in the planner's Advanced BAND menu.
- opt2d3's k-point-vector and literature BAND titles name the path
  ("V-Y-GAMMA-X" instead of "in-plane path" on slabs) and the "Literature"
  source, as the same configuration replayed from a file (a workflow step or
  a saved Expert config) titles it. Only the title line changes; a
  literature request with no literature path is titled "default".
- opt2d12 runs answered through a pipe list every setting that differs from
  the parent deck (functional, basis, grid, TOLINTEG, TOLDEE, LEVSHIFT,
  HISTDIIS, MAXTRADIUS), so an answer file written for an older MACE cannot
  change a deck silently. The deck is unchanged.
- The phonopy walltime suggestion carries a 1.5x margin (diamond 2x2x2
  hybrid 2 h -> 3 h, 3x3x3 20 h -> 30 h); lighter methods and `--cry-sym`
  are not shortened, since a job killed at its walltime leaves no forces.
- The workflow engine no longer prints raw DEBUG dumps (FREQ settings,
  workflow id/settings/sequence, sequence position).
- The defects README states the POTC sign convention the alignment assumes
  (unverified), and the phonopy docs explain the phonopy/CRYSTAL mass offset
  (about 1 cm-1 at 540 cm-1 for Si) and why an internal-basis set has no
  TEMPLATE (phonopy's writer puts ENDGEOM before it).
- INSTALLATION.md says to install the module-Python dependencies on a
  development or compute node: an MSU HPCC gateway node refuses
  `module load Python/3.11.3-GCCcore-12.3.0` (the module the job scripts
  load), while the same load works on dev-amd20. Job scripts are unchanged;
  they run on compute nodes.
- The stuttgart library's description says which records are unverified (117
  of 119) and which CRYSTAL cannot read.
- Tests: the bundled MgO defect outputs
  `tests/data/supercell_parents/rg_q0.out.gz` and `rs_qm1.out.gz` are now
  trimmed excerpts (about 500 lines each) with no MPI start-up lines,
  process counts, run dates, timings or per-node CPU times;
  `tests/test_supercell_parent_excerpts.py` keeps such lines out. So is
  the rod group 4 polymer parent `tests/data/low_dim_groups/polyyne_rg4.out`
  (now without its SCF cycles and timings, about 410 lines).

### Fixed

- **A workflow FREQ step keeps a FREQ parent's SCF tolerances.** The
  planner's Basic, Advanced and Expert-fallback FREQ steps and the
  quick-start FREQ step wrote Very tight (9 9 9 11 38 / 11) into the plan as
  if the user had chosen it, replacing a FREQ parent's tolerances. They now
  leave the choice to opt2d12: Very tight after an OPT parent (as before, and
  as the manual opt2d12 route gives), the parent's own after a FREQ parent.
  Tolerances set in the plan still win.
- **A FREQ child of a FREQ parent keeps the parent's IR/Raman
  intensities** (INTENS, INTCPHF, INTRAMAN, IRSPEC/RAMSPEC) on every route,
  when it runs in the parent's Gamma-point mode. opt2d12's interactive FREQ
  template menu now defaults to "0: Keep the parent's FREQCALC settings"
  when the parent has intensity or spectrum settings (it defaulted to "Basic
  frequencies only", writing NOINTENS); a parent with only NUMDERIV gets the
  usual questions, its NUMDERIV the default answer. The workflow's Basic and
  quick-start FREQ steps no longer write "no intensities" into the plan, the
  Advanced level's blank answer keeps the parent's, and a config that names
  no intensity setting keeps the parent's when its FREQ mode is the
  parent's. A config that names NUMDERIV alone therefore no longer drops the
  parent's intensities, and a phonon dispersion config gets none (NOINTENS),
  as opt2d12's phonon templates write.
- **The workflow's phonon band structure step writes opt2d12's deck.** The
  planner's Advanced FREQ phonon route (full dispersion, band structure) on
  default answers wrote INTERPHESS 2 2 2 and no BANDS path. It now writes
  what opt2d12's "6: Phonon band structure" template writes on default
  answers: SCELPHONO 2x2x2, NOINTENS, DISPERSION and the automatic
  high-symmetry BANDS label path, with no INTERPHESS (checked byte for byte
  on the corpus AgCl FREQ parent). Deck change: BANDS records instead of
  INTERPHESS, and the title names the path.
- **The workflow's phonon DOS steps write opt2d12's deck.** The planner's
  Advanced FREQ phonon route with output type "2: Phonon density of
  states" or "3: Both band structure and DOS" stored only a label nothing
  reads, so the deck was `FREQCALC / NOINTENS / DISPERSION / ENDFREQ`: no
  PDOS, and no BANDS for "both". On default answers DOS now writes what
  opt2d12's "7: Phonon density of states" writes (PDOS `2000.0 200`,
  atom-projected), and "both" what opt2d12's custom dispersion route
  without INTERPHESS writes: template 6's BANDS path, then that PDOS. An
  output type not on the menu, which says it defaults to the band
  structure, now writes its BANDS path too. Deck change: re-plan phonon
  DOS steps.
- The planner's Expert FREQ fallback (used when the interactive opt2d12 run
  fails) wrote IR + Raman intensities through CPHF while saying "using
  advanced settings". It now writes what opt2d12's interactive FREQ route
  gives on default answers (Gamma-point frequencies, no intensities, or a
  FREQ parent's own) and says so.
- **The default BAND deck for a triclinic parent or any slab** (opt2d3
  Enter-through, and so the workflow default) is now integer k-point
  coordinates, with ISS taken from the parent's shrink. This is the deck
  opt2d3's k-point vector format writes. CRYSTAL23 rejects label input
  there: triclinic stopped with "K-POINTS NOT CODIFIED" and slabs with
  "FORMAT ERROR IN INPUT DECK". A label path that MACE cannot express for
  the lattice now writes no deck, where it used to write placeholder
  records.
- **The phonon band structure deck for a triclinic (P-1/P1) parent**
  (opt2d12's "6: Phonon band structure" on default answers, and so the
  planner's phonon band structure step) writes its BANDS path as integer
  coordinates (`BANDS / 16 100 / 9 / 8 8 0 0 8 0 ...`) instead of the
  ISS 0 label records (`V Y`, `Y GAMMA`, ...) CRYSTAL23 does not codify for
  a triclinic lattice. Deck change: regenerate triclinic phonon band decks.
- **A polymer's BAND deck runs along the chain** (opt2d3 Enter-through, and
  so the workflow default): Gamma -> X, `1 2 NSUB ... / 0 0 0  1 0 0`,
  whatever space group the polymer's symbol maps to. A polymer whose
  corresponding space group MACE maps (rod groups 28, 51) got an ISS = 0
  label deck, which CRYSTAL23 stops with "BANDE **** FORMAT ERROR IN INPUT
  DECK" (a polymer takes no label records). One whose symbol it does not map
  (rod groups 1 and 4) got the triclinic 3D path with I2/I3 != 0, against
  manual p.310 note 3: CRYSTAL ran it as five copies of Gamma-X under
  meaningless labels. The k-point vector, literature and SeeK-path formats
  keep their segments along the chain (I2 = I3 = J2 = J3 = 0), or fall back
  to Gamma -> X. Measured on CRYSTAL23 for rod groups 1, 4 and 28. Deck
  change: regenerate polymer BAND decks.
- **opt2d12 writes the phonon band structure (template 6) for a slab.** It
  asked every question and then refused, saying a slab has no space group
  to take the path from. A slab now gets SCELPHONO as a 2x2 matrix (`2 0 /
  0 2`, manual p.73; the 3x3 matrix aborts) and BANDS records of 4 integers
  (I1 I2 J1 J2), the in-plane part of the corresponding space group's path
  as opt2d3 cuts it. CRYSTAL23 takes no BANDS labels for a slab (INPFREQ
  FORMAT ERROR) and reads only 4 integers of each record, so 6-integer
  records went to other k points. Measured on an hBN monolayer: `12 100 / 2
  / 6 0 0 0 / 0 0 4 4` ran as M-Gamma-K and the P1 slab path as
  V-Y-Gamma-X. WANG (3D only, manual p.234), INTERPHESS, PDOS and INS are
  still refused for a slab (not run on CRYSTAL23), as is a slab from
  cif2d12 (no output to read the corresponding space group from) and a
  polymer dispersion (not run). opt2d12 now refuses a dispersion as soon as
  the FREQ template is chosen, before the method, basis and SCF questions.
- opt2d12's "6: Phonon band structure" FREQ template no longer carries
  one run's answers into the next run in the same process: the
  template's BANDS settings were shared, so after a default label-path run
  a later run lost the k-point density question and wrote the earlier
  run's segment count. Each run now starts from the template itself.
- The advanced electronic analysis in property extraction reads only the
  output's own BAND.DAT / DOSS.DAT: `<stem>.BAND.DAT`, else the same
  parent's `_band` / `_doss` run. It took the first such file in the
  output's directory, which could belong to another calculation. Two equally
  good matches are now skipped and named in `advanced_analysis_note`, not
  guessed between.
- **opt2d12 `--config-file` reads legacy functional spellings** ("HSEsol-3C",
  "PBEh-3C", "M06-2X", "HF", ...) as workflow plans do: the CRYSTAL23 keyword
  is written and a 3c method gets its basis. This covers the planner's Expert
  configs under `workflow_configs/expert_*_configs`, which the engine hands to
  opt2d12. "M06-2X" reached the deck, which CRYSTAL23 rejects, and
  "HSEsol-3C" kept the parent's explicit basis as "BASISSET / EXTERNAL". An
  unknown name now writes nothing and says why.
- The workflow planner's free-text questions (the sequence builder's "Next
  calculation", the Advanced BAND custom path and "Use all bands?", and the
  SLURM directives) refuse answers that cannot be meant for them (an unknown
  calculation type, a path that is not k-point labels, a directive that is
  not an sbatch option, anything but yes/no) and stop a piped run after
  three in a row. These and the planner's shared yes/no, menu and number
  questions stop with a message, not an EOFError traceback, when piped
  answers run out.
- opt2d12's FREQ menus ask again on an answer that is not on the menu. The
  template menu took any other answer (e.g. "9") as "Custom settings", the
  FREQCALC/ANHARM, mode and supercell questions took it as their first or
  last entry, and the phonon band-path crystal-system and template menus fell
  back to their first entry without saying so.
- The workflow planner's number prompts (queue limits, BAND/DOSS points,
  DOSS projection type and energy range, TRANSPORT chemical potentials,
  temperatures and relaxation time, CHARGE+POTENTIAL grid, FREQ supercell,
  k-point count, CPHF settings, peak widths, dielectric constant and the
  temperature list) ask again on an answer that is not a number instead of
  ending `mace workflow` with a ValueError. Counts must now be at least 1
  (0 allowed for reserve slots, recovery attempts and DOSS projection type
  0-4); absolute chemical potentials have no default and are asked again
  when left blank.
- Band and phonon paths for a slab no longer print "Install 'seekpath'":
  the library is not used for a slab on purpose (it would standardise the
  vacuum axis; manual p.310 note 3), so the advice changed nothing. A slab
  now gets one line saying the static in-plane path is used; a 3D cell
  without the library still gets the install hint. The path is unchanged.
- opt2d12's log names the basis the deck is written with. From an
  external-basis parent it printed "Using external basis set from original
  D12 file" (and "Preserving external basis settings") before the config
  chose the method, so HF-3c children logged that and then "Basis set:
  MINIX", and DFT-3c children (PBEH3C, HSESOL3C, ...) logged it although the
  deck says BASISSET <3c basis>. The line is now printed once, only when the
  deck carries the parent's external basis. The legacy workflow executor
  likewise logs "Basis set: <3c basis> (required by <functional>; the
  parent's basis is not used)" for a 3c step. Log only: decks are unchanged.
- **opt2d12 no longer writes SP or OPT decks from a phonon-supercell
  (SCELPHONO) parent** (a dispersion or a Gamma-only SCELPHONO FREQ; the
  refusal says "phonon supercell (SCELPHONO)"). The parent's output geometry
  is the SCELPHONO supercell (manual sec. 4.21 p.73-74); measured on HPCC
  (MgO, SCELPHONO 2x2x2 + DISPERSION) the SP deck written from it was a
  space-group-225 cell with a = 5.95 A and 5 asymmetric atoms instead of
  MgO. Such children are now refused like FREQ children, with the advice to
  make them from the OPT or SP output the dispersion run started from. Deck
  change: these decks are no longer written.
- The job script no longer stages a job's own two-component `.f9` for a
  GUESSP inside a TWOCOMPON block. There GUESSP means GUESSPNOSO (manual
  p.173): the 2c-SCF starts from a one-component matrix of an unrestricted
  run without symmetry. GUESSP, GUESSPNOSO and 2NDVARIAT inside TWOCOMPON now
  use only an explicitly staged `$JOB.f20` (with a note of the p.173
  requirement); with none, a bare GUESSP is dropped from the scratch copy
  (cold start), and GUESSPNOSO/2NDVARIAT are left as written with a warning
  (CRYSTAL stops without fort.20, measured). A GUESSP at the top level of a
  deck with a TWOCOMPON block gets the same handling, as CRYSTAL reads it like
  GUESSPNOSO too (measured on HPCC, Bi2 slab, PBE0+SOC: the deck's own 2c
  `.f9` as fort.20 stops with "io error Read_real_1d", a 1c matrix computed
  with symmetry with "SYMMETRY INCOMPATIBILITY"); the OPTGEOM RESTART of such
  a deck no longer keeps its own 2c fort.20 either. A one-component
  `$JOB.f20` staged for a top-level GUESSP is used with a warning: that
  measured run read it, then diverged where the cold start converged in 12
  cycles. A GUESSP in a one-component deck is handled as before.
- `opt2d3 --config-file` finds the shipped example configs by bare name
  (`band_auto_everything.json`) from any directory, and a relative
  `.../Crystal_d3/example_configs/<name>` written against another working
  directory, and prints the file it loaded. A configuration that cannot be
  loaded now ends opt2d3 with exit status 1 instead of 0 with no deck.
- **SeeK-path band paths use the computed geometry: the optimised cell after
  an optimisation, the supercell after SUPERCEL.** With the seekpath library,
  `opt2d3` (and the workflow BAND step, and `opt2d12` phonon-dispersion paths)
  read the cell from the first "PRIMITIVE CELL" lattice parameters and the
  atoms from the first atom list of the parent output. For an OPT parent that
  is the initial geometry, so the path's lattice-dependent points (and, when
  the cell changed enough, the lattice variant, tI1 or tI2 say) were those of
  the starting cell. After SUPERCEL the cell was the input primitive cell
  while the atoms were the supercell's (FREQ `..._supercel222.out`: 16 atoms
  in a 2.507 A cell instead of the 5.013 A supercell). Cell and atoms now come
  from one computed geometry block: the one after the last FINAL OPTIMIZED
  GEOMETRY, otherwise the first (GEOMETRY FOR WAVE FUNCTION, the cell fort.9
  holds). **K-paths are in the basis of the computed cell**: a BAND deck from
  a SUPERCEL output is in the supercell's reciprocal basis (folded bands).
  Outputs whose geometry input edits the cell (SUPERCEL, ATOMSUBS, ATOMREMO,
  ELASTIC, ...) change the same way. An atom count that does not match the
  one CRYSTAL prints, or an output without such a block, is read as before
  with a warning. An optimisation that ended unconverged (OPT END - FAILED)
  prints no final geometry; its last printed point is used, the geometry its
  fort.9 holds (a killed run keeps the first).
- **Static band paths (without the seekpath library) use the computed cell
  too.** The static route took the conventional cell of the input echo: an
  optimisation's starting cell chose the lattice variant (tI1 or tI2, oF1 or
  oF3, hR1 or hR2) and the oA parameter. It now takes the CRYSTALLOGRAPHIC
  CELL printed with the final optimised geometry. A SUPERCEL output's static
  path is mapped to the supercell's reciprocal basis, f_super = E f_prim with
  E the printed EXPANSION MATRIX OF PRIMITIVE CELL (rows), as the library
  route gives it; before, it was written in the primitive cell's basis while
  BAND ran on the supercell. An R cell given in rhombohedral axes (CRYSTAL
  echoes it as a = b = c, alpha = beta = gamma) is read in hexagonal axes.
  `opt2d12` now writes the child of a SUPERCEL parent as the supercell in P1
  (below), and its phonon-dispersion path is that cell's own. Single-point
  outputs without SUPERCEL or geometry editing are unchanged.
- **Static band paths compute their lattice-dependent points for the cell.**
  Without the seekpath library the static paths of body-centred tetragonal
  (tI1, tI2), rhombohedral (hR1, hR2), face-centred orthorhombic (oF1) and
  C-centred monoclinic (mS1, SeeK-path's mC1) lattices held the
  lattice-dependent points of one particular cell (tI1's Z at 0.341718 for
  every crystal), so their segments ended off the special points of any
  other cell. Those points are now evaluated for the cell CRYSTAL prints
  with SeeK-path's HPKOT formulas (new `Crystal_d3/hpkot_static.py`, numpy
  only; transcribed from SeeK-path, MIT, Hinuma et al. 2017) and given in
  CRYSTAL's reciprocal basis, with the ISS the library route chooses: the
  static and library routes write the same BAND path for the 30 centred
  corpus outputs; for about 1% of random cells the ISS chosen can differ.
  The warning and the "install seekpath" note are gone for these paths.
  Cells not in SeeK-path's setting (oF axes not in a < b < c, a C-centred
  monoclinic cell whose a-c mesh SeeK-path reduces) or of another variant
  (oF2/oF3, mC2/mC3) keep the old entry with a warning that gives the
  reason; aP, mP, oP, oI, oS and oC paths still warn as before. Paths
  written without an output file are unchanged.
- **Band-path ISS stays at most 999 for long-axis cells.** When no ISS up to
  999 placed the lattice-dependent points within half a k-point spacing (at
  10000 points), both the seekpath library route and the new static
  computed-point route wrote the points exactly, with ISS up to ~10**6 (a
  C2/m cell of 9.7 x 3.7 x 20.8 A: ISS 959108 with the library, 588382
  statically). fort.25 writes the segment ends as fixed-width integers (6I3
  in App. D p.447; 6I5 measured), so such ends overflowed there. ISS is now
  the one up to 999 that moves the points least (990 for that cell, a move
  of 8.6e-4 1/A, 0.13 of the k-point spacing at NSUB 1000), with a warning
  giving the move. Deck change: the ISS and segment-end integers of BAND
  paths for such cells only; cells that fitted before are unchanged.
  Untested on CRYSTAL.
- **The bundled Pu basis record (`294`, triplezeta and doublezeta) declared
  11 shells but holds 9.** CRYSTAL would have read the next atom's records,
  or `99 0`, as Pu's missing shells. It now says `294 9`, as the source set
  does. That alone did not make Pu decks run: its f-shell charge was also 3
  short of neutral (see the Ac-Es fix below). **Deck change:** any deck with
  Pu and a bundled external basis. Untested in CRYSTAL.
- **The BAND gap of a doped run used the wrong band.** Without CRYSTAL's
  "TOP OF VALENCE BANDS" line the occupied-band count was the header's
  "N. OF ELECTRONS PER CELL" // 2. It now comes from the Mulliken
  "NO. OF ELECTRONS" count (never the ALPHA-BETA print, which is the spin) of
  the BAND .out or its SCF parent beside it; an odd or fractional count
  leaves the BAND gap unset instead of wrong.
- **Mulliken net charges of pseudopotential (ECP) atoms.** The net charge
  was the Mulliken Z column minus the population; for an ECP atom that
  column holds CRYSTAL's label (282 for Pb) and the population counts valence
  electrons only, so Pb came out near +261. It is now measured against the
  pseudopotential's NUCLEAR CHARGE. An ECP atom (Mulliken Z 200 + Z) whose
  valence charge the .out does not give is flagged with no charge, rather
  than measured against Z (the 21 ECP outputs of the test corpus all give
  it). Newly extracted rows change for ECP atoms; rows already in a database
  are not migrated (re-extract to update).
- **Derived decks keep a charged parent's electron count when the basis set
  changes.** An explicit-basis parent carries its charge in `CHEMOD` +
  `CHARGED`; a child with an internal BASISSET got the same records, which
  there only reshape the guess, so it ran neutral. The charge is now written
  as `DOPING`. `CHEMOD` is copied only when the child loads the same basis
  set (it was copied whenever the atom counts matched, although its shell
  charges are per shell of the parent's basis). When the basis changes and
  the parent's records are exactly what MACE's ion charges write for it, the
  equivalent `CHEMOD` is worked out again for the child's basis (net charge
  as `DOPING`, or `CHARGED` in an explicit basis); any other `CHEMOD` is
  dropped with a warning that gives the reason. `DOPING` is always kept,
  with a warning when the atom count changed. Deck change for such children.
- **A derived deck with an odd electron count is no longer written closed
  shell.** It gets the parent's `SPIN`/`SPINLOCK`, or opt2d12 refuses it
  with the reason.
- **opt2d12 kept no charge records when the parent deck failed to parse,**
  although the file was there; they are now carried from its text. **With
  no parent deck at all**, a charged parent's `DOPING` is recovered from the
  .out's Mulliken count; without a Mulliken print a warning says the charge
  state cannot be recovered when the .out shows a sign of a charged run
  (CHEMOD, CHARGED, "UNIT CELL NOT NEUTRAL", a renormalised guess), and an
  informational note says so otherwise (a neutral FREQ parent is not warned
  about). Deck change for such children.
- **A failed run's message says what CRYSTAL printed.** The queue manager
  reported "Calculation failed with unknown error" for any error it had no
  pattern for; it now quotes CRYSTAL's `ERROR ****` line, and for an .out
  that ends in `MPI_Abort` with no error text says that Pcrystal can hide an
  input error such as INPBAS "UNIT CELL NOT NEUTRAL" (measured on CRYSTAL23:
  only a serial run prints it). The error type, and so recovery, is
  unchanged; such outputs were and are never counted as finished (pinned on
  the real outputs).
- **An END after `CHEMOD` in a library-basis deck is caught.** After
  `BASISSET <library>` the lines are already block 3, so an `END` closing
  `CHEMOD` ends the input there and CRYSTAL23 silently drops the DFT block
  (the run falls back to HF; measured on CRYSTAL23). MACE never wrote it;
  `mace ions` now refuses a deck that has one (also with `--replace`, which
  used to strip `CHEMOD` and keep the stray `END`), and opt2d12 warns that
  such a parent ran HF. A `CHEMOD` record whose atom labels stand on lines of
  their own, the shell charges on the next line (as CRYSTAL also reads it),
  is now read, carried and stripped whole; it used to be cut after one line
  per atom, which hid that `END` from the check.
- `mace submit DIR` with a relative directory (or `sub/deck.d12`) submitted nothing:
  every deck failed with "Failed to generate/find script". Absolute paths were fine.
- **cif2d12 / `mace convert` custom setup shows the functional menus again.**
  Since 1.1.2 every CIF was treated as a parent whose functional could not be
  identified. The category and functional menus were replaced by a free-text
  prompt that rejected menu numbers, and blank answers wrote HSE06 without
  D3. Blank answers write HSE06-D3 again, as in 1.1.1 and as the defaults
  banner says. opt2d12 parents keep the replacement prompt.
- **Workflow planner convergence menus accept yes/no again.** The SP and OPT
  convergence questions became 0-3 menus, and any other answer was quietly
  taken as 0 (keep the previous step's values). y/yes and n/no now mean
  what they did in 1.1.1. SP: y/yes is Very tight (TOLINTEG 9 9 9 11 38,
  TOLDEE 11), n/no keeps the previous step's tolerances. OPT: y/yes is
  TOLDEG 1.5e-5, TOLDEX 6e-5, TOLDEE 8, MAXCYCLE 1000; n/no is TOLDEG 3e-5,
  TOLDEX 1.2e-4, TOLDEE 7, MAXCYCLE 800. A note is printed for both, and
  any other answer is asked again.
- **Answers piped into interactive cif2d12 and opt2d3 runs are read again.**
  cif2d12 ignored a piped answer to the space-group mismatch question, and
  opt2d3 ECHG refused with "no terminal to ask". If the pipe runs out, the
  question takes the unattended choice (cif2d12) or the run writes no deck
  and exits 1 (opt2d3), with no traceback. `--batch` and `--config-file` runs
  are unchanged.
- **Out-of-step piped answers stop the run.** With no terminal, three invalid
  answers in a row to any question (yes/no, numbered menus, numbers,
  opt2d3, the planner and plotting) now stop the run with an error; at a
  terminal the question still asks until answered.
  Before, the next blank answer took the default and the run carried on with
  answers out of step. In the planner this covers its number, memory,
  walltime and level menus too, and an off-menu Advanced BAND path option or
  path format is asked again (it was taken as a custom path, or as labels).
- **A fort.20 staged by hand for GUESSP is kept.** Copying a density matrix
  into the job's scratch directory as fort.20 was how GUESSP restarts were
  done in 1.1.1. Since 1.1.2 the job script stripped GUESSP when there was
  no `<job>.f20`/`.f9` next to the deck, or copied the job's own old `.f9`
  over it. A non-empty fort.20 in scratch is now used unless the job's `.f9`
  is newer (then it was left by an earlier run of the same job). The same
  applies in the TWOCOMPON branch.
- **The `.mace_scratch` fallback is cleaned up.** When `$SCRATCH` is
  unusable, CRYSTAL runs in `<submit dir>/.mace_scratch`, which was never
  removed. Its job directory is now removed after a run that reached
  CRYSTAL's normal end; a run that did not keeps it for an OPTGEOM restart.
  Any scratch fallback is now also noted in `<job>.out`.
- **A job script from before 1.1.3 is no longer reused silently.** `mace
  submit` warns when a MACE-written `<job>.sh` has no scratch guard, with the
  `--overwrite-sh` hint; the script is not changed. A workflow's
  `workflow_scripts/` template from before 1.1.3 that the plan names is
  rebuilt from the plan, the way a fresh execution of the plan writes it
  (the old copy kept as `<name>.pre-1.1.3`), so later steps get the scratch
  guard and RESTART/GUESSP staging; one the plan does not name is warned
  about, with the re-plan hint.
- **Decks are written through a symlink and keep their permissions.** The
  atomic deck write replaced a symlinked deck with a regular file (the link
  target kept the old deck) and gave every deck the umask's mode. The link
  target is now written and the deck's mode kept, as in 1.1.1; `mace ions`
  uses the same writer.
- `mace <command> -h` (and a help flag after other arguments, or after
  `--no-banner`) shows that command's help instead of the general usage.
- `mace submit DIR` on a mixed .d12/.d3 directory stops with a message (exit
  1) instead of an EOFError traceback when stdin has no answer.
- `mace convert` without `--batch` stops with a message pointing at
  `--batch`, instead of an EOFError traceback, when stdin ends before the
  setup questions are answered.
- opt2d12's free-text LEVSHIFT ("ISHIFT ILOCK") and HISTDIIS-size answers
  are asked again on bad input instead of quietly taking the default (which
  pushed the rest of a piped answer file one more question out of step).
  Deck change: an ILOCK other than 0/1, a single LEVSHIFT value or a
  non-positive HISTDIIS size is no longer written or replaced.
- The planner's optimization-type menu asks again on an off-menu answer
  instead of silently keeping the previous type, and its CHARGE+POTENTIAL
  output-type question asks again instead of crashing with a KeyError.
- `mace submit --interactive` accepts walltimes like 12:00:00, 2:00:00 and
  12:00, which its own prompt gives as examples (they were refused and the
  default used).
- opt2d3 runs answered through a pipe read their later questions there too
  (missing-wavefunction "Continue anyway?", each file's ECHG map plane and
  POTC points with shared settings), as in 1.1.1, instead of refusing for
  want of a terminal. Runs given a `--config-file` still never ask.
- `opt2d3 --batch --config-file` takes the calculation type from the config
  instead of asking for it first; with none anywhere it exits 1 naming
  `--calc-type`.
- opt2d12's grid menu marks the parent's grid as the default instead of
  always labelling XLGRID. The default itself is unchanged.
- **The HSE-type odd-electron crash is no longer resubmitted with more
  memory.** A DIIS NaN at SCF cycle 1-3, then ranks killed by signal 11/6/9,
  on an odd-electron open-shell run with HSEsol-3c, HSE06 or HSEsol is
  reported as `screened_hybrid_odd_error`, which names PBE0/PBEsol0 as what
  converged and keeps the functional. A closed-shell out-of-memory kill is
  still `memory_error`.
- A job squeue still lists as COMPLETED is classified from its .out; a run
  whose ranks were killed (SLURM records OOM and segfault kills as
  COMPLETED) is no longer marked completed.
- Re-extracting untracked outputs (no calculation record) also removes the
  rows an older extractor wrote and the current one no longer produces (e.g.
  1.1.1's band fields of DOSS runs), when the run re-extracts every output
  of that material it can see; a database rebuild is no longer needed for
  that.
- Re-organising (the populate scan) corrects the calc type of a record an
  older MACE typed from its file name (e.g. an SP named `*_OPT_*` recorded
  as OPT); workflow-numbered types are left as they are.
- **Workflow BAND steps write the deck opt2d3 writes for the same choices.**
  The planner's Advanced BAND menu stored keys opt2d3 does not read:
  "labels" gave k-point vectors, and literature, SeeK-path and custom paths
  gave a deck with no path (NLINE 0). It now writes the keys opt2d3 saves,
  asks for a custom path at planning time instead of promising it "during
  execution", and stores a band range as first and last band. Checked on the
  59 periodic SP parents of the test corpus, with and without seekpath: the
  engine default, planner Basic, quick-start and Advanced formats 1-4 are
  byte-identical to opt2d3's decks for the same choices.
- Interactive opt2d3 reads a slab's "CORRESPONDING SPACE GROUP N." line. It
  wrote the P1 (triclinic) path for slab outputs that print only that line;
  the config route already read it.
- Workflow D3 steps use a plan's per-material Expert configurations; the
  engine ignored them and ran its own default deck. A config file planned on
  another machine is found again under this run's `workflow_configs`.
- Workflow FREQ steps keep a FREQ parent's NUMDERIV when the plan names
  none, as opt2d12 does. At the planner prompt a blank answer means "the
  parent's, else none" and 0 means no keyword.
- The legacy WorkflowExecutor step generator passes its step config (and an
  Expert plan's saved config) as `--config-file` with `--non-interactive`;
  it used `--options-file`, which opt2d12 only writes to, so nothing was
  applied and no deck was written.
- **opt2d12 writes the child of a SUPERCEL/SUPERCON parent as the supercell
  it computed**, every atom in P1 in CRYSTAL's order, instead of the host's
  space group with the supercell's cell and asymmetric atoms (the diamond
  2x2x2 SP child was SG 227, a = 5.01 A, 5 atoms). The parent's DOPING,
  CHEMOD and GHOSTS are kept, and SHRINK is written in the P1 form (same
  mesh). The workflow engine goes through the same code.
- Decks derived from a GHOSTS parent keep its ghost atoms: opt2d12 copies
  GHOSTS when the child lists the same atoms (ghosts are not counted in the
  electron parity), and opt2d3 DOSS projects the ghosts' basis functions on
  their own (`<El>_GHOST`) instead of under their element.
- opt2d12 `--calc-type` (and `--opt-type`) win over a `--config-file`: a
  config without `calculation_type` no longer turns `--calc-type SP` into an
  OPT deck.
- Without the seekpath library, centrosymmetric C-centred groups (C2/m,
  C2/c, Cmcm, ...) get SeeK-path's path (9 and 11 segments) instead of the
  primed Cc / Cmc2_1 path (18 and 22); the static k-path table has no
  duplicate keys, and aP1's labels name its segments. An orthorhombic F cell
  gets the oF variant SeeK-path takes (1/a^2 vs 1/b^2 + 1/c^2), not the
  shortest-axis guess, and a centrosymmetric oF2 cell the time-reversal
  path.
- **The bundled La-Lu (ERD, full.basis.triplezeta/doublezeta) and Ac-Es (all
  three families) basis records have neutral formal shell charges;** every
  La-Lu or Ac-Es deck on them stopped with INPBAS "UNIT CELL NOT NEUTRAL".
  La-Lu: +2 on the first empty s/sp shell and +1 on the first empty d,
  keeping the published 4f count (the site prints an Ln3+ guess); Ac-Es: the
  site's f charge restored. Exponents and coefficients are unchanged.
  Untested in CRYSTAL.
- Basis records CRYSTAL cannot read (stuttgart 204, 219, 220, 227, 294) are
  refused on every deck path, naming the record, instead of being copied
  into decks.
- A walltime-killed two-component OPT whose TWOCOMPON block asks for
  GUESSP/GUESSPNOSO/2NDVARIAT is no longer restarted from its own
  two-component fort.20 (io error Read_real_1d), and the job script drops
  GUESSROTM (and its angles) with the TWOCOMPON GUESSP it drops for want of
  a one-component matrix. Refreshed old job scripts get the same staging.
- CHEMOD records with the label on its own line and integer charges are read
  correctly (labels, and the charge an explicit-basis CHEMOD sets).
- The static k-path of an optimised SUPERCEL cell is the host's path in the
  supercell basis, as for its SP.
- The custom-labels BAND menu lists the labels for the output's own lattice
  (it always showed cubic P), and the shrink fallback (no SHRINK in deck or
  output) uses the computed cell.
- Computed geometry blocks of SLAB and POLYMER outputs are recognised.
- opt2d3 no longer overwrites X_sp's deck with X_opt's when both are in one
  directory: they are written as `X_sp_<type>.d3` and `X_opt_<type>.d3`; a
  lone output keeps the short name.
- The seekpath-missing note is no longer printed on every import.
- An opt2d12 `--config-file` whose top-level `functional` is a 3c method
  (HSESOL3C, PBEH3C, ...) and that names no basis gets the method's own
  basis. On a parent with an explicit basis the deck said "BASISSET /
  EXTERNAL (from original D12)", which CRYSTAL cannot read. A functional
  under `method_modifications`, and a config that names its basis, already
  worked.
- Charged-system setup refuses when the parent .out lists other shells than
  MACE's table for an internal library, and the CHEMOD-with-symmetry refusal
  names the orbit-wide CHEMOD and MODISYMM alternatives.
- opt2d3 reads the electron count CRYSTAL prints.
- `mace defects analyze` says when a `--site-label` sits on the wrong
  element instead of reporting DOPING not applied; an ECP site's electron
  count is given as its parity only; and a warning is printed when the
  image-charge term is applied to a far-from-cubic supercell.
- A Mulliken atom with a second all-electron basis (label Z + 1000) is
  measured against Z instead of being flagged as an ECP of unknown valence;
  CRYSTAL takes an ECP only on Z + 200 or Z + 300 (manual p.25).
- A SPINLOCK, ATOMSPIN, BETALOCK or DOPING phonopy parent is refused once,
  not twice.
- `basis_library` loads a family from `MACE_HOME` in a `copy_dependencies`
  job directory.
- The test suite writes no bytecode, so it never writes into the test
  corpus.
- The opt2d12 tests that answer prompts through a runner import the
  checkout under test; they imported whichever `mace` PYTHONPATH named.
- **A job's completion callback runs the MACE that submitted it.** Job
  scripts run under `bash --login`, which re-reads `~/.bashrc`; an
  `export MACE_HOME=...` there (with PYTHONPATH, as `setup_mace.py
  --add-to-path` writes it) decided which install's queue manager and
  `mace` package ran the job's follow-up and recovery steps, so a second
  install (a side-by-side test, a shared group copy) had its callbacks run
  by the one the dotfile names, silently. Job scripts now set MACE_HOME and
  PYTHONPATH to the generating install after the login shell has started,
  and say so in the job log when the login shell named another; `mace` and
  the queue manager export their own install to the workflow's generator
  copies. Job scripts and `workflow_scripts/` copies written before 1.1.5
  are not pinned; regenerate them with `mace submit --overwrite-sh`, or
  re-plan the workflow.
- The input-settings queries (`python -m mace.database.query.queries`) and
  `settings_extractor.query_calculation_settings` read the extracted deck
  settings from `input_settings_json`, where they are stored; they read
  `settings_json` (the job's workflow metadata), so a stored extraction was
  never shown and calculations with only workflow metadata were listed.
  Databases from before that column still have theirs read from
  `settings_json`.
- Importing `mace.utils.show_properties` no longer opens (and creates) a
  `materials.db` in the current directory; the query runs only when the
  script is run.
- `setup_mace.py --update-templates` checks the job-script templates in
  `mace/submission/` and reports each one; it looked in the removed
  `code/Job_Scripts/` and silently did nothing. Its `--help` examples name
  `setup_mace.py` (they named `setup_crystal_tools.py`).
- The install docs (README, INSTALLATION, the Crystal_d12 and Crystal_d3
  READMEs) require Python 3.11 or newer, the oldest version CI tests; they
  said 3.6+ or 3.7+, which nothing tests.
- `pytest.ini` no longer says the Crystal_d12 menu tests run separately;
  every test is under `tests/`.
- **Automatic workflow steps after the first ran nothing.** The engine kept
  a step's script generator as `<job>.sh` and ran `bash <job>.sh <job>`,
  so the generator overwrote the file bash was reading. The submitted
  script stopped after the module loads (no CRYSTAL run, no completion
  callback) and each follow-up ended in seconds with no `.out`; the first
  step of `mace workflow --execute` and quick-start did the same. The
  generator now runs from a separate copy (broken since 1.1.2).
- A job that left the queue without writing its `.out` (cancelled while
  pending, killed before CRYSTAL started) stayed `submitted` for good, so
  its workflow could neither finish nor recover. Once sacct reports the
  job ended, the record is marked failed with that reason (`no_output`).
  `mace monitor --status` also counts submitted calculations, which it
  left out of the totals.
- **`mace submit --progress` split a workflow over two databases.** It
  recorded the submissions in `./materials.db`, but the plan's job scripts
  export the workflow context, so the completion callback recorded the step
  again in `.mace_context_<id>/materials.db`, filed its properties in
  `./materials.db` under the second record's id, and left the first record
  `submitted`. The submissions are now recorded in the context database,
  as `mace workflow --execute` records its first step, and the queue
  manager gives property extraction, settings extraction, the workflow
  engine and error recovery the database it opened rather than the
  `materials.db` default.
- `mace manager --d12-dir DIR` submitted nothing: with no `--callback-mode`
  it ran the job-script completion callback, which outside a workflow only
  records status. It now runs a full cycle (update the records, catch early
  failures, submit the decks in `DIR`), and the manager is built once.
- Jobs the queue manager submitted outside its database's directory
  (`mace manager --organize`, or `--d12-dir` elsewhere) had completion
  callbacks that opened a new `materials.db` in the job directory, so the
  manager's records stayed `submitted` with no properties. The job script
  now exports `MACE_DB_PATH`, and the callback passes it as `--db-path`.
  Job scripts written before this release do not carry it; regenerate
  them, or run `mace manager --callback-mode status_check` from the
  database's directory to bring the records up to date.
- `mace submit DIR` and `mace submit DIR/x.d3` with a relative path
  failed for D3 decks ("Failed to generate/find script") and submitted
  nothing, as the D12 path did before 1.1.5; the D3 path now resolves the
  directory first too.
- `mace analyze --extract-properties` filed every property a second time
  with no calculation id: it matched an output to its calculation by the
  exact path string, and a scan passes relative paths where the database
  holds resolved ones. It now compares the resolved paths and, in an
  in-place directory with several calculations, picks the one whose deck
  has the output's name. In a workflow directory it stores into the
  context database `mace database` and `mace monitor` read (not
  `./materials.db`), and "Materials processed" counts the materials (it
  always printed 0).
- `mace database --action export` (CSV, JSON) left out every property
  unless `--include-property` named some; it now includes them all by
  default, as intended.

## [1.1.4] - 2026-10-02

Everything since 1.1.3. **Band and phonon-dispersion paths made with the SeeK-path
library were written in SeeK-path's cell, not CRYSTAL's, since 1.0.0:**
C-centred (oC) and A-centred (oA) orthorhombic paths went through wrong
k-points, and triclinic, some monoclinic and non-standard orthorhombic paths had
wrong labels or segments. They are now converted to CRYSTAL's reciprocal basis;
regenerate affected BAND and phonon paths. Decks derived from a charged or
ionic parent now keep its `CHEMOD`, `CHARGED` and `DOPING`; before, SP and FREQ
children ran a different charge state. Also: SeeK-path-consistent static paths
for more lattices, in-plane band paths for slabs, smaller band-path ISS, HF-3c
and HFsol-3c configs write their MINIX/SOLMINIX basis, and several menu,
workflow-state and job-script fixes. Several of these change generated decks -
see Fixed below.

### Fixed

- **SeeK-path band and phonon-dispersion paths are written in CRYSTAL's
  reciprocal basis.** Since 1.0.0 the seekpath-library route (BAND `.d3`
  decks from `opt2d3`, phonon-dispersion paths in FREQ `.d12` decks from
  `opt2d12`, and the workflow steps that run them) copied SeeK-path's point
  coordinates unchanged. Those are fractions of the reciprocal vectors of
  SeeK-path's standardised primitive cell; BAND reads them as fractions of
  the reciprocal vectors of CRYSTAL's primitive cell (manual p.309), which is
  often a different cell. The points are now converted (f = N^-1 f_SeeK-path,
  with SeeK-path's cell = N times CRYSTAL's), keeping their labels and
  Cartesian positions; SeeK-path's own numbers are kept where every segment
  is already a symmetry image of the right one, so those decks are
  unchanged. Affected: **C-centred orthorhombic (oC)** and **A-centred
  orthorhombic (oA, groups 38-41)** cells, whose points and segments were
  wrong; **triclinic (aP)** cells, whose labels named other points;
  **primitive orthorhombic (oP)** cells of groups whose setting does not fix
  the axes (Pmmm, for one) when not given in a < b < c order; and any other
  cell CRYSTAL holds in a setting SeeK-path standardises differently (in the
  test corpus a cubic cF1 cell given in a non-standard primitive cell, and
  two P2/m and P2_1/m cells whose Gamma-A and E-Z segments ran to another
  copy of A and E). Without the library, the static oA paths (added in this
  release cycle) are converted the same way and take the cell's own
  lattice-dependent parameter; the static paths of other lattices whose
  standard cell CRYSTAL's can differ from (aP, mP, mC, oP, oC, oF, oI) now
  warn. Band-structure and phonon-dispersion decks made for such cells
  should be regenerated.

- **A deck derived from a charged or ionic parent keeps its charge state.**
  `opt2d12` wrote SP, FREQ and other children without the parent's `CHEMOD`
  (per-atom starting shell charges, e.g. Li+ and FSI-), `CHARGED` and `DOPING`
  (net electrons added or removed) records, so the child of a +1 Li/FSI/EC
  cluster ran as a neutral one with a different electron count. They are now
  copied from the parent deck: `CHEMOD`/`CHARGED` after the basis set and
  `DOPING` in the SCF block, as in decks CRYSTAL23 runs. `CHEMOD` names atoms by
  position, so it is only copied when the child lists the same number of atoms;
  otherwise the deck is written with a warning to add it by hand.

- **An opt2d12 config with `"method": "HF"` and HF3C or HFSOL3C writes MINIX
  or SOLMINIX.** It kept the parent's basis: an external-basis parent gave
  `BASISSET / EXTERNAL (from original D12) / HF3C / END`, which is not input,
  and an internal one `BASISSET / POB-TZVP-REV2 / HF3C`. The manual defines
  HF-3c for a pure HF calculation in MINIX and HFSOL-3c in SOLMINIX (sec.
  5.3.1 p. 158, 5.4.1 p. 162); such decks now read `BASISSET / MINIX / HF3C /
  END` (layout p. 159, run on CRYSTAL23), with plain atomic numbers (47, not
  the external basis's 247).
- **HF-3c on a structure with metal atoms prints a caution** (opt2d12 and
  cif2d12). The manual (p. 158) says HF-3c was designed for organic systems,
  is carefully tested for molecular crystals, and that inorganic crystals and
  metals are less well tested; an AgCl HF3C/MINIX run diverged on CRYSTAL23.
  The deck is still written.
- **opt2d12 refuses a FREQ deck from a phonon-dispersion (SCELPHONO) FREQ
  run.** The child repeated the parent's SCELPHONO on the parent's geometry,
  which is already the SCELPHONO supercell (manual pp. 73-74), so the
  supercell would be expanded again.
- **Out-of-range answers are asked again** at the OPT convergence-level menu
  (it fell through to Custom), the FREQ IR-method menu (KeyError) and the FREQ
  NUMDERIV menu (ValueError on a non-number; 3 was written although NUMDERIV
  takes 1 or 2, manual p. 218).
- **The planner's BAND prompt asks for the total number of k-points along
  the path** (NSUB, manual p. 309), not "points per k-path segment". The
  default (10000) is unchanged.
- **The workflow state records the step that completed.** After an OPT whose
  planned next step was FREQ it could record another calculation's type.
- **The job script does not stage an empty `<job>.f20` for GUESSP;** it
  falls back to the job's own non-empty `.f9`, or drops GUESSP as before.
- **`guessp_restart` is documented as applied by the workflow engine only;**
  the executor's older step generator, which normal workflows no longer
  reach, now says it writes its decks without GUESSP.
- **Primitive hexagonal groups get SeeK-path's hP1/hP2 split.** Without the
  seekpath library every primitive trigonal and hexagonal group got the hP1
  path, whose last segment K-H_2 only hP1 groups need. SeeK-path (2.2.2,
  HPKOT) uses hP2 for 150, 152, 154, 156, 158, 164, 165 and 168-194; those
  groups (P321, P-3m1, P6_3/mmc, P6/mmm, ...) now get the hP2 path, one
  segment shorter. 143-145, 147, 149, 151, 153, 157, 159, 162 and 163 keep
  hP1.
- **The static face-centred cubic SeeK-path paths are SeeK-path's.** Without
  the seekpath library, the cF1 entry held Setyawan-Curtarolo's 9-segment
  path (Gamma-X-W-K-Gamma-L-U-W|L-K|U-X) and the cF2 entry a 9-segment path
  with a primed K' in a centrosymmetric group, both titled "SeeKPath". They
  now hold SeeK-path's cF1 (Gamma-X-U|K-Gamma-L-W-X-W_2, 7 segments) and cF2
  (Gamma-X-U|K-Gamma-L-W-X, 6 segments), and the cF1_noinv title gets the
  "|" between U and K that its segments already had.
- **Face-centred cubic groups get SeeK-path's cF1/cF2 split.** Without the
  seekpath library the split was reversed for 202, 203 and 225-228: Fm-3m,
  Fm-3c, Fd-3m (diamond, Si) and Fd-3c got cF1, and Fm-3 and Fd-3 got cF2.
  SeeK-path uses cF1 below 207 (196, 202, 203) and cF2 from 207 (209, 210,
  216, 219, 225-228), so Fm-3m and Fd-3m decks lose the X-W_2 segment and
  F23 decks gain it.
- **Body-centred cubic groups are all cI1.** The static lookup called 199,
  204, 206, 211, 214, 217 and 220 "cI2", a lattice SeeK-path (HPKOT) does not
  have, backed by table entries copied from cI1's. Those entries are removed
  and every body-centred cubic group is cI1. The decks are unchanged (the
  paths were identical).
- **The static rhombohedral SeeK-path paths are SeeK-path's.** Without the
  seekpath library, R-3, R-3m and R-3c (148, 166, 167) got Setyawan-
  Curtarolo's rhombohedral paths under a "SeeKPath" title, with points
  (B, B1, Q, P1, X, Z) SeeK-path does not have: 11 segments for hR1, 10 for
  hR2. They now get SeeK-path's hR1 path (Gamma-T-H_2|H_0-L-Gamma-S_0|S_2-F-
  Gamma, 7 segments) and hR2 path (Gamma-L-T-P_0|P_2-Gamma-F, 5 segments),
  with the same representative values for the lattice-dependent points
  H, S and P as the non-centrosymmetric entries.
- **Rhombohedral groups get SeeK-path's hR1/hR2 choice.** Without the
  seekpath library the static lookup took hR1 for c/a < sqrt(6) (hexagonal
  axes), so typical rhombohedral crystals (c/a 2.5-7; AgBr R-3c, Bi2Se3,
  corundum) got the hR2 path and flat cells got hR1. SeeK-path (HPKOT) takes
  hR1 for c/a > sqrt(3/2), i.e. a rhombohedral angle below 90 degrees, and
  hR2 otherwise; a cell given in rhombohedral axes is now read by its angle.
  Rhombohedral decks from outputs with c/a above sqrt(6) or below sqrt(3/2)
  change path.
- **A-centred orthorhombic groups get SeeK-path's oA paths.** Without the
  seekpath library, Amm2, Aem2, Ama2 and Aea2 (38-41) fell through to the
  primitive orthorhombic oP1 path. They now get SeeK-path's oA1 (b < c) or
  oA2 (b > c) path with its primed copy, and the title marks the jumps
  C_0|SIGMA_0 and A_0|E_0 (F_0|DELTA_0, B_0|G_0) that the segments make.
- **`opt2d3` reads a slab's symmetry.** A SLAB output names its symmetry as
  a plane group and the corresponding space group ("TWO-SIDED PLANE GROUP
  N. 80 : P 6/M M M", "CORRESPONDING SPACE GROUP N. 191"; manual App. A.2,
  p.421), not on the "SPACE GROUP ... :" line `opt2d3` reads, so every slab
  was taken as space group 1 (P1) and, without the seekpath library, got the
  triclinic band path. It now takes the corresponding space group and the
  plane group's lattice letter.
- **A slab's band path stays in the plane of the slab.** `opt2d3` BAND wrote
  a 3D path for a SLAB parent: with the seekpath library, SeeK-path's path
  for the slab's 500 A box, e.g. graphene's hP2 path with nine segments, six
  of them at or towards kz = 1/2. A slab is periodic in a and b only, and
  BAND takes I3 = J3 = 0 in 2D (manual p.310, note 3). The path of the
  corresponding space group is now cut to its segments in the kz = 0 plane
  (graphene: Gamma-M-K-Gamma), in coordinate and label mode, and the title
  names the segments kept. For a slab the seekpath library is not used:
  it standardises the 3D box and can turn the vacuum axis into a or b (an
  oblique slab becomes mP with b along the normal), so its coordinates would
  not be in CRYSTAL's slab cell; the static path is. If no segment of a path
  lies in the plane (the literature path of a C-centred rectangular slab),
  the in-plane SeeK-path path is written instead. The `opt2d12` SeeK-path
  phonon band path of a slab is cut the same way. Polymers (1D) are not
  changed. Untested on CRYSTAL.
- **SeeK-path band paths keep ISS within three digits.** With the seekpath
  library, ISS was the largest denominator of every SeeK-path point - also
  points not on the path - after rounding lattice-dependent coordinates to a
  denominator below 10**6, so decks carried ISS up to ~10**6 (median about
  6*10**4 over representative cells of all 230 groups). CRYSTAL runs them
  (the k-point density depends only on NSUB, manual p.309 note 4), but
  fort.25 writes the segment ends with format 6I3 (App. D, p.447), so ends
  of 1000 and more printed as asterisks. Now the points SeeK-path fixes by
  fractions (its points.txt) set ISS to their lowest common denominator,
  from 4 (cubic P 4, cI 4, cF 8, hexagonal 6, previously 16 or 18), and are
  written exactly; lattice-dependent points are written as k/ISS with ISS at
  most 999, a multiple of that denominator, choosing the ISS that moves them
  least, provided no point moves by more than half the k-point spacing of
  the path at 10000 points (the workflow's NSUB). Over all 230 groups and
  about 4800 representative cells the largest move is 1.2e-4 of a reciprocal
  lattice vector (3.7e-4 1/A with 2*pi, at most 0.42 of the k-point spacing
  at 10000 points); MACE's old round trip of these decks already moved points
  by up to 1.2e-4. Whether a point is lattice-dependent now comes from
  SeeK-path's definitions, not from its value being within 1e-6 of a small
  fraction. NSUB is unchanged. The `opt2d12` SeeK-path phonon path now
  starts from the library's ISS, as `opt2d3` does, instead of rescaling the
  library's points to the band shrink. Untested on CRYSTAL.

## [1.1.3] - 2026-10-01

Everything since 1.1.2. Band paths are written with exact high-symmetry points
and follow SeeK-path for more lattices (cubic, hexagonal, monoclinic, centred);
`opt2d12` FREQ decks repeat their FREQ parent's FREQCALC settings and
Hartree-Fock parents stay Hartree-Fock; the DOSS example configs now give decks
CRYSTAL runs; a P1 deck from a CIF without symmetry operators holds the right
atoms; and workflow steps can optionally restart their SCF from the previous
step's density matrix (off by default). Several of these change the content of
generated decks - see Fixed below.

### Added

- **Workflow steps can restart their SCF from the previous step's density
  matrix (opt-in).** With `"guessp_restart": true` under `execution_settings`
  in the workflow plan JSON, an SP, FREQ or OPTn step the engine builds from a
  completed OPT or SP gets the predecessor's `.f9` copied to `<job>.f20` in its
  directory (the job script stages that as fort.20) and a GUESSP record before
  SCFDIR. The manual (GUESSP, pp. 114-115) requires the same symmetry and the
  same atoms, basis functions and shells in the same order, and CRYSTAL does
  not check this, so the engine only does it when both decks have the same
  symmetry records, atom list, basis set and spin treatment (UHF/ROHF/SPIN);
  the lattice, coordinates and functional may differ. It is not done when the
  follow-up sets ATOMSPIN (p. 99: "does not work with GUESSP") or either deck
  is two-component, when the predecessor left no non-empty `.f9`, or when the
  step's job script predates the `.f20` staging; the engine prints why. The
  setting is off by default, and without it the engine writes exactly what it
  wrote before. Untested on CRYSTAL.

### Fixed

- **The Crystal_d12 README says where the phonon SeeK-path path gets its
  inversion.** It still said the phonon path had only the space-group number,
  and kept a known issue about the output text being taken for a file name.
  Since that fix the parent output reaches the SeeK-path helpers as a file
  (`d12_calc_freq._output_as_file`), and inversion and cell parameters come
  from it (`d3_kpoints.detect_inversion_from_crystal_output`); the number is
  the fallback when there is no output.
- **`opt2d12 --config-file` writes Hartree-Fock decks from an HF config.** A
  config with `"method": "HF"` and a null functional (the form
  `quick_screen.json` had) stopped every file with `argument of type
  'NoneType' is not iterable`, and one naming only the method, or its flavour
  as `hf_method`, kept the parent's DFT functional. opt2d12 now reads the
  flavour from `hf_method` or `functional`, RHF when neither names one, as
  cif2d12 does, and refuses an HF config that names a DFT functional.
- **"Proceed with spglib space group" writes spglib's cell with spglib's
  group.** When a CIF's group and spglib's disagree, option 2 declared
  spglib's group number but kept the CIF's cell and atoms, so CRYSTAL applied
  the group's standard operators to a cell in another setting: rock salt given
  as its 60-degree primitive cell became an Fm-3m cube with a = 3.99 A instead
  of 5.64 A. The deck now carries spglib's standardised conventional cell and
  its asymmetric unit, in the origin the deck declares for two-origin groups.
- **A P1 deck from an Fd-3m CIF holds the structure the CIF describes.** A
  CIF with no symmetry-operator loop is expanded by ASE in origin choice 1, but
  the converter reads Fd-3m atoms at (1/8, 1/8, 1/8) as origin choice 2 - as
  the symmetrised deck ("0 0 0") and CRYSTAL's own diamond example do. The
  P1 deck of such a diamond CIF therefore held 16 atoms 1.26 A apart instead
  of diamond's 8. The P1 expansion now uses the origin the symmetrised deck
  declares for Fd-3m, and for any two-origin group when origin_setting is
  STANDARD. CIFs that list their operators are expanded as before.
- **P2/c and P2_1/c keep the primitive monoclinic SeeK-path.** Without the
  seekpath library, a band or phonon path built from a CRYSTAL output reads
  the cell parameters, and with them space groups 13 and 14 were sent to the
  C-centred monoclinic path (mS1), whose points belong to another Brillouin
  zone; the split between P and C was taken at group 11. They now get mP1,
  as SeeK-path does and as they already did without cell parameters.
- **Simple cubic groups 207-230 get SeeK-path's cP2 path.** Without the
  seekpath library every primitive cubic group took the cP1 path, whose last
  segment M-X_1 belongs to groups 195-206 only. P432, P-43m, Pm-3m and the
  other groups 207-230 now take cP2 (cP2_noinv when non-centrosymmetric), as
  SeeK-path does.
- **Static SeeK-path titles name the segments the deck writes.** Without the
  seekpath library, the title labels of the P-1 (aP3), C-centred monoclinic
  (mS1: C2, Cm, C2/m, ...), F-orthorhombic (oF1-oF3: Fmmm, Fddd), I-tetragonal
  (tI1/tI2: I4/m, I4/mmm, ...) and non-centrosymmetric F-cubic (F-43m, F23, ...)
  paths ran on where SeeK-path's path jumps, so the title had more edges than
  the deck had segments and band-plot nodes were misnamed. They now follow
  SeeK-path's path, with a "|" at each jump. The P-1 aP2 path also called
  (1/2, 1/2, 0) and (1/2, 0, 1/2) N and M; SeeK-path names them V and U. The
  coordinates are unchanged.
- **Static SeeK-path titles of the other centred lattices name their
  segments too.** The same run-on labels were in the paths used for the
  non-centrosymmetric rhombohedral groups (R3, R32, R3m, R3c: hR1/hR2),
  body-centred tetragonal groups (I4, I-4, I422, I4mm, I-42d, ...: tI1/tI2)
  and face-centred orthorhombic groups (F222, Fmm2, Fdd2: oF1/oF3), and for
  every body-centred orthorhombic group (Immm, Imm2, I222, ...: oI1) and
  every C-centred orthorhombic group (Cmcm, Cmc2_1, Cmmm, ...: oS1). They
  now follow SeeK-path's path with a "|" at each jump; the tI2 path without
  inversion also named its point G as Gamma. The coordinates are unchanged.
- **Hexagonal K and H points are written exactly in BAND and phonon paths.**
  Coordinate paths are written in units of 1/shrink, and K = (1/3, 1/3, 0) is
  only a whole number of steps when the shrink is a multiple of 3; at the usual
  16 it was rounded to 5/16. Quarter points had the same problem at a shrink
  such as 6 or 990. The shrink is now raised to the next common multiple of the
  points' denominators (16 becomes 18 for hexagonal paths), the way a shrink
  too small for the path was already raised.
- **The DOSS example configs give decks CRYSTAL runs.** Decks written from
  `doss_total_only.json`, `doss_orbital_projections.json` and
  `doss_element_orbital_auto.json` (and any DOSS deck with "print integrated
  DOS" switched on) stopped in CRYSTAL with "DOSS FORMAT ERROR IN INPUT DECK":
  the printing-option count was 1 but no printing-option record followed. That
  record (`105 -1`) is now written. `doss_total_only.json` also asked for bands
  0 to 999, which CRYSTAL refuses ("BAND RANGE NOT ALLOWED"); a band range is
  now kept within 1 to the number of AOs. The examples now write `DOSS.DAT`
  (they asked for no file at all), and "Total DOS only" gives a total DOS
  instead of per-orbital projections. All three examples were run on HPCC.
- **An `opt2d12` phonon deck with a SeeK-path band path is written.** The
  SeeK-path helpers read a `.out` file, but `opt2d12` passed them the parent's
  output text, so every such deck stopped with `OSError: File name too long`.
  They now read the same text from a temporary file, and the path matches what
  the `.out` itself gives.
- **The DAT-file and population-analysis processors open the materials
  database.** `process_calculation_dat_files` and
  `process_material_population_analysis` (`mace/utils`) imported
  `MaterialDatabase` from `material_database`, a module that no longer exists,
  and stopped with ModuleNotFoundError; they now use `mace.database.materials`.
- **Space-group tables match the International Tables and CRYSTAL.** Pbcn
  (60) was listed with two origin choices, and its "alternate origin" code
  `0 1 0` is CRYSTAL's rhombohedral-axes flag: a cif2d12 deck asked for that
  origin carried it. It now writes `0 0 0` for Pbcn with any origin option;
  every other group's origin handling is unchanged, and the list is now
  exactly the ITA two-origin groups. The spelling table maps group 90 as
  CRYSTAL prints it ("P 4 21 2", not "P 42 1 2"), maps P-42c to 112 (not
  114), and has no duplicate key; every lookup already reached the right
  number through another table, so no other deck changes.
- **`copy_dependencies.py` copies the files it lists.** Its list still used
  the names from before the scripts moved into the `mace` package, and it
  looked for `Crystal_d12/` and `Crystal_d3/` inside `mace/`, so it reported
  most files missing and copied none of the converters. It now copies each
  file from where it lives, under its current name, together with the
  modules the copied converters import (`d12_config.py`, `menu_nav.py`,
  `spglib_compat.py`, `seekpath_interface.py`). It no longer tries to copy
  `recovery_config.yaml`, which the recovery engine reads from `mace/config/`.
- **Documentation examples and paths that did not work.** The D3 batch
  example passed `--config-file doss_orbital_projections.json`, which is
  opened as given and not looked up in `example_configs/`; it now gives the
  full `$MACE_HOME/Crystal_d3/example_configs/` path. DOCUMENTATION.md listed
  three of the six `--calc-type` values of `opt2d3`, pointed at
  `Band_Alignment/` and `Post_Processing_Scripts/` without their `code/`
  prefix, and referred to a `d12creation.py` that no longer exists. It now
  also documents applying one `opt2d12` template to many structures.
- **The standard-library `queue` module is no longer replaced by
  `mace/queue`.** `mace_cli` puts `mace/` itself on `sys.path`, where the
  `mace/queue` package shadows the standard module, so the first bare
  `import queue` in the process registered `mace/queue` under that name.
  Removing an unused `import queue` from `mace/workflow/executor.py` did not
  stop this, because `pyarrow`, which `mace.database` imports, runs its own
  `import queue`. Importing `mace` now loads the standard `queue` with
  `mace/` hidden from `sys.path`, before anything else can, and `mace_cli`
  imports the job-queue package as `mace.queue` rather than `queue`.
- **A cif2d12 SLAB deck lists each atom once, at its own height.** A SLAB
  record takes the symmetry-unique atoms, with z in Angstrom from the layer
  group's origin (CRYSTAL23 manual p. 21). Handed a whole cell - which batch
  mode keeps when spglib and the CIF disagree, or with `write_only_unique`
  off - the converter wrote every atom with z = c times fractional z: a Bi2
  bilayer (P-3m1, layer group 72) got its inversion image as a second Bi at
  z = 19.10 A instead of -0.90 A, and CRYSTAL built a four-atom slab 38 A
  thick. Atoms the layer group generates from an earlier one are now left
  out, using the CIF's own operators when they match the named group's in
  number and kind (otherwise the atoms are written as given, with a
  warning). For a layer group with an operation that reverses z, the whole
  layer is moved by one offset so that the operation's plane - at
  fractional z = 0 or 1/2 in the cell - is at z = 0; the layer stays in one
  piece, even when it is centred on z = 1/2 or straddles z = 0/1, and every
  z distance in it is the cell's own. A polar-layer-group slab keeps its
  c times z heights; when it straddles z = 0/1, the atoms just below z = 1
  are written one cell lower, next to the rest of the layer.
- **The static orthorhombic-P path without inversion names its segments
  in order.** Without the seekpath library, `oP1_noinv` listed the labels
  Y-T before X-U (and Y'-T' before X'-U') while its coordinates, like
  SeeK-path's path, run X-U first, so those segments carried each other's
  labels.
- **Non-centrosymmetric structures with a 2-fold axis get the
  non-centrosymmetric SeeK-path path.** Without the seekpath library, the
  band and phonon paths read inversion from the CRYSTAL output, and the
  SYMMOPS check matched a run of six matrix elements that a 2-fold axis
  along z (and, in the hexagonal frame, an in-plane 2-fold or -6) shares
  with the inversion. Every such group - in the
  primitive lattices 16-19, 25-34, 75-78, 81, 89-96, 99-106, 111-118,
  149-154, 168-174, 177-190, 195, 198, 207, 208, 212, 213, 215 and 218 - was
  called centrosymmetric and got the path without primed points. An operator
  now counts as an inversion only when its whole rotation part is -I, and a
  SYMMOPS table without one means no inversion. The space-group-number check
  no longer reads a digit out of a symbol ("P N N 2" as group 2).
- **The static SeeK-path band paths go through the zone-boundary points.**
  Without the seekpath library, band and phonon paths come from a table in
  `d3_kpoints` that defines several lattices twice; the definitions that
  win had every coordinate doubled (body-centred cubic x4, hexagonal x6), so
  simple cubic X was (0, 1, 0) - a reciprocal lattice vector, the same point
  as Gamma - where SeeK-path and CRYSTAL23 (manual Tables 14.1-14.2,
  pp. 311-312) have (0, 1/2, 0). The paths for aP, cP, cI, hP, mP, oP, tP,
  tI and mC2 (and the aP and mC2 non-centrosymmetric ones) now use SeeK-path's
  coordinates: e.g. every centrosymmetric primitive cubic, tetragonal,
  orthorhombic, monoclinic and hexagonal group, Im-3m, Ia-3d and I4/mmm.
- **`get_auto_phonon_path` returns a path in the labels and vectors
  formats.** Both stopped with NameError, because `d12_calc_freq` checked
  `get_crystal_system_from_space_group` without importing it from
  `d3_kpoints`. Nothing calls the function today, so no deck changes.
- **An `opt2d12` SeeK-path phonon deck's title names the path it writes.**
  The title took its labels from the space-group number alone, while the
  BANDS path is chosen from the parent's output, whose lattice parameters
  and symmetry operators pick the SeeK-path variant. For AgBr in R-3c the
  title listed an 11-segment path over the 10-segment one written, and
  every such title called its path "default". The title now reads the same
  output, so it lists the written path's labels and says "SeeKPath (w.I)",
  "SeeKPath (no.I)" or "Literature" as the path was chosen.
- **The workflow planner loads without `d12_constants`.** When
  `d12_constants` or `DummyFileCreator` could not be imported, the planner
  reported it with `ui.warn` before `ui` was defined, so the import stopped
  with NameError instead of using its built-in fallbacks. `ui` is now set up
  first.
- **A mistyped answer to a D3 yes/no question is asked again.** In the
  interactive `opt2d3` setup, an answer other than y/yes/n/no (or Enter)
  counted as "no", even where the default was "yes", and the question moved
  on. It now prints "Please respond with 'yes' or 'no' (or 'y' or 'n')." and
  asks again, as the D12 tools do. Every recognised answer, and end of input,
  behave as before.
- **An `opt2d12` FREQ deck repeats its FREQ parent's FREQCALC settings.** Only
  NUMDERIV was read from the parent's FREQCALC block, so the child of a parent
  that asked for IR intensities (`INTENS / INTCPHF / ENDCPHF`), Raman
  intensities or IR/Raman spectra was written with `NOINTENS`. The whole block
  is read now, and a FREQ child written without questions (`--non-interactive`,
  or a `--config-file` without FREQ settings of its own) repeats it; RESTART
  is not repeated, as it restarts the parent's own run from the FREQINFO.DAT
  that run wrote (manual sec. 8.2, p. 219). A record the writer never writes
  (e.g. NOUSESYMM) is reported in the parse as `freq_unparsed`. The
  interactive questions still default only NUMDERIV to the parent's.
- **A Hartree-Fock parent deck reads back as Hartree-Fock.** `CrystalInputParser`
  gave no functional for a deck with no Hamiltonian keyword (CRYSTAL's default
  RHF, manual p. 123) or with `HF3C` / `HFSOL3C` (manual sec. 5.3.1, pp. 158,
  162), so such a deck was written back with a DFT block. It now reads them as
  the functionals `RHF`, `HF3C` and `HFSOL3C` (method HF), as the deck writer
  takes them, and reads SMEAR under the writer's `smearing` key as well. For
  `opt2d12`: an RHF parent's `.out` already names RHF, and SMEAR still comes
  from the `.out`, so those children are unchanged; an HF-3c or HFsol-3c
  parent's child now gets that functional, where the `.out` could give RHF and
  drop the correction (untested: no HF-3c run to hand).
- **A deck MACE wrote reads back as the same deck.** `CrystalInputParser`,
  which reads the parent deck for `opt2d12`, kept coordinates only as floats
  (`1.250000000000E-01` came back as `0.125`) and stripped the spacing of an
  EXTERNAL basis' records. It now also keeps the title, each atom record's
  coordinates and last word, and the EXTERNAL basis records as written.
  Parsing a deck and writing it back with `opt2d12`'s writer now gives the
  same deck byte for byte for all 811 decks `cif2d12` writes in the new
  round-trip test (CRYSTAL, SLAB, POLYMER and MOLECULE; internal and EXTERNAL
  basis; SP, OPT and FREQ; 80 before), for `opt2d12`'s own children of the
  real runs in `tests/data`, and for 324 of the 401 real decks of the test
  corpus apart from their title, which is `./<name>` (an older MACE took it
  from the path). The other 77 hold records the writer never writes, or in
  another order: they were edited after MACE wrote them (SUPERCEL, a RESTART
  that recovery adds to OPTGEOM, GUESSP below MAXCYCLE, RAMSPEC ahead of
  IRSPEC) or written by hand. The new keys never reach `opt2d12`'s settings,
  so its decks are unchanged.

### Removed

- Five standalone scripts that nothing runs, imports or documents:
  `mace/workflow/status.py`, `mace/workflow/callback.py` and
  `mace/workflow/check_workflows.py` (superseded by `mace status` and the
  queue manager's completion callback), `mace/utils/scf_settings_extractor.py`
  and `mace/utils/analyze_script_dependencies.py`.
- The phonon fallback k-path tables `SPACEGROUP_TO_PATH` and
  `HIGH_SYMMETRY_PATHS` (`Crystal_d12/d12_constants.py`). They gave one path
  per crystal system whatever the lattice centring (C2/m the primitive
  monoclinic path, Fmmm and Immm the primitive orthorhombic one, I4/mmm the
  primitive tetragonal one, R-3m the hexagonal one), and their only reader was
  a fallback in `d12_calc_freq.get_auto_phonon_path`, a function nothing in
  MACE calls. Its coordinate paths now all come from the centring-aware
  band-path code that its "vectors" format already used. No deck changes.

### Testing

- Tests for paths that had none: the workflow plan JSON from planner to
  executor to a job's callback (save, load, validation, queue settings),
  `mace status` and `mace completion` on real outputs (categories, zombie
  jobs, moving files, the workflow's isolated database), 'b' to go back
  through the real OPT/SP/FREQ and opt2d3 questionnaires, and the timeout
  RESTART recovery on the committed PbTiO3 and Ag2Br3 runs, end to end with
  sbatch/squeue/sacct/scontrol stand-ins on PATH.
- The corpus sweeps that skip without `test/` (deck geometry rebuild,
  EXTERNAL basis blocks, TESTPDIM insertion, outputs typed by their deck,
  settings extraction, aggregation keys, timeout RESTART) also run over the
  real decks and outputs committed under `tests/data`, which now include the
  two original ECP OPT decks (`tests/data/samples/ecp_decks`).
- Three known bugs are pinned as expected failures: at the OPT
  convergence-level menu an answer outside 1-4 is taken as Custom, and in
  the FREQ menu an answer outside 1-3 for the IR method (KeyError) or a
  non-number for NUMDERIV (ValueError) stops the questionnaire.
- CI has a second job with ase, spglib and seekpath installed
  (`requirements-optional.txt`), so the tests that need them run there.

## [1.1.2] - 2026-09-29

Everything since 1.1.1. Decks MACE derives from a finished calculation now keep
the parent's method and settings; `opt2d12` prompts default to the parent's
values and a saved template can be applied to many structures at once; OPT jobs
that run out of walltime continue from their last step (`OPTGEOM RESTART`, with
a fallback when the restart itself aborts); calculation types are read from a
file's records rather than its name; and a series of CRYSTAL23-conformance,
job-script and recovery fixes. Several of these change the content of generated
decks - see Fixed and Changed below.

Conformance work traced from a user report: every hybrid lead perovskite died
with `ERROR **** LoadBa **** UNIT CELL NOT NEUTRAL` under HSE-3c, on cells that
are neutral. Auditing that one bug against the CRYSTAL23 manual turned up a
family of others. Every behavioural change below was run through
CRYSTAL/23-intel-2023a on real hardware, not just reasoned from the manual.

### Fixed

- **A POLYMER or SLAB deck made from a `.out` alone keeps the parent's rod or
  layer group.** `opt2d12` wrote whatever space group it had as the group
  record. With the parent `.d12` present that was the parent's own record, so
  every existing child was right (checked: all 162 SP/OPT/FREQ children of the
  27 slab decks in `test/`). From the `.out` alone it was not: a POLYMER
  output's "CORRESPONDING SPACE GROUP : P 65" line was read as 3D space group
  170 and written as the rod group, which CRYSTAL refuses (rod groups run
  1-99, layer groups 1-80), and one inside the range names a different group.
  A SLAB fell back to P1 instead of its layer group. The child now takes the
  group from the parent deck or the `.out`'s own "TWO-SIDED PLANE GROUP N." /
  "POLYMER GROUP N." line, never from a 3D analysis, and a group outside the
  range is refused before any deck is written.
- **A SLAB deck's k-point mesh keeps its layer group's symmetry.** A mesh
  `opt2d12` generates (a child made from the `.out` alone) or `cif2d12` writes
  comes from the conventional a and b, and was only equalised for 3D decks. In
  a centred-rectangular layer group the two primitive directions are
  equivalent, so graphene in layer group 47 got `SHRINK 0 36 / 18 10 1` and
  CRYSTAL stopped with "SHRINK BREAKS SYMMETRY". Square, hexagonal and
  centred-rectangular layer groups (CRYSTAL numbers the last 10, 13, 16, 22,
  26, 35, 36, 47, 48) now get one factor for both in-plane directions; oblique
  and primitive rectangular groups and POLYMER meshes are unchanged, and a
  child made with the parent `.d12` still takes the parent's own SHRINK.
  Checked with real SCF runs of CRYSTAL/23 on HPCC: layer group 47 from
  `opt2d12` and from `cif2d12` (refused before, converged now), 37 (18 10 1
  kept) and 80, and rod groups 28 and 51.
- **The example configs load in `mace convert --batch --options_file`.** None
  of the eight `Crystal_d12/example_configs/*.json` did: they wrap their
  settings in `{"version", "type", "configuration"}` and spell them the
  opt2d12 way (`functional`, `dispersion`, `scf_settings`), and every run
  stopped with `KeyError: 'dimensionality'`. The loader now unwraps the file
  and maps those names onto cif2d12's. What a config leaves out gets the
  interactive defaults (3D `CRYSTAL`, the CIF's space group written as its
  asymmetric unit, INTERNAL basis), and a file that cannot describe a deck
  (no functional, an unknown method, calculation type or symmetry setting)
  is refused with the reason and a non-zero exit. Each config was converted
  from real CIFs and checked by CRYSTAL23 (`mace preflight`).
  A flat options file saved by `--save_options` or the workflow planner
  writes the same decks as before - checked byte for byte on all 168 corpus
  CIFs with 18 such files - with one deliberate exception: a lowercase
  `"calculation_type": "opt"` used to fail the comparison with `"OPT"` and
  silently write a single point with `_opt_` in its name; it is now read as
  OPT, so the deck gets its OPTGEOM block and the name reads `_OPT_`. The
  basis set is written exactly as the file spells it (`DEF2-MSVP` stays
  `DEF2-MSVP`), and a 3c method paired with another basis is written as
  asked with a warning: CRYSTAL23 runs HSE-3c on POB-TZVP-REV2, which the
  lead perovskites need because its internal def2-mSVP has no Pb.
  `phonon_bands.json` now names the 2x2x2 `SCELPHONO` supercell that
  `FREQCALC DISPERSION` needs (without it CRYSTAL23 stops with
  `MAKE SUPERCELL WITH SCELPHONO`) and writes its phonon band path as
  k-point coordinates, since CRYSTAL23 rejects the label form there
  (`FORMAT ERROR IN FREQCALC INPUT DECK`).
- **`mace opt2d12 --config-file` reads the example configs too.** It also
  took the wrapper for the settings and crashed. `quick_screen.json` now names
  its HF flavour (`"functional": "RHF"`) so both converters read it the same
  way.
- **`d12_from_config.py` runs.** It passed the input file as an argument
  neither converter accepts, and did not recognise a CRYSTAL output that
  starts with MPI start-up lines. It also reports a file for which no deck
  was written as a failure (and exits non-zero) instead of "Successfully".
- **`mace convert --batch` exits non-zero when it writes nothing.** A refused
  options file, or a batch in which every CIF is refused (e.g. HF-3c's MINIX
  on a Pb structure), printed the reason and exited 0, so the workflow
  executor - which only checks the exit status - carried on with no decks.
- **The built-in `3c_composite` template uses def2-mSVP**, the basis PBEh-3c
  is defined on, instead of MINIX (HF-3c's). The basis now comes from the
  `basis_requirements` table rather than a second copy; a configuration that
  names a 3c method and no basis gets that basis.
- **cif2d12 FREQ decks follow the lattice centring for the phonon path.** The
  automatic phonon-dispersion path read the centring only from a CRYSTAL
  output, so every deck made from a CIF got the primitive path: Fd-3m
  diamond got the simple-cubic M-G-R-X-G where opt2d12 writes X-G-L-W-G for
  the same structure. cif2d12 now takes the letter from the space group
  symbol and writes the same path as opt2d12.
- **Phonon dispersion is refused off a 3D crystal, and no partial deck is
  left.** `opt2d12 --config-file phonon_bands.json` on a SLAB parent crashed
  in the title (no space group to take the path from) and left a 0-byte
  deck; on a MOLECULE it wrote `SCELPHONO` + `DISPERSION`, which CRYSTAL23
  stops with `SUPERCELL OPTION NOT ALLOWED FOR MOLECULES`. The 3x3
  `SCELPHONO` matrix both converters write is also rejected for a SLAB
  (`THE SUPERCELL IS OF ZERO VOLUME`; a slab takes 2x2), and the automatic
  path needs a 3D space group. Both converters now refuse dispersion for a
  MOLECULE, SLAB or POLYMER with the reason, and
  write every deck to a hidden part file that is renamed into place only when
  complete, so an abort or a crash never leaves a 0-byte or truncated deck.
- **`opt2d12 --output-dir` holds the deck when `--out-file` is an absolute
  path.** The new name was built from the parent's full path, and joining an
  absolute path onto the output directory discards the directory, so the
  deck was written next to the parent (into the test corpus, for one run).
- **`mace convert --batch` never reads the terminal.** On a structure whose
  atoms are all symmetry-unique (every P1 CIF), the symmetry step asked "Use
  the 'reduced' structure anyway?", hit end of input and reported "Error
  during symmetry analysis" before carrying on - 85 of the 168 corpus CIFs.
  Batch mode now takes symmetry from the options file (`symmetry_handling`,
  `write_only_unique`, `symmetry_tolerance`, now documented) and gives every
  question an interactive run would ask the prompt's own default, so the
  decks are byte-for-byte the ones written before, without the error. A CIF
  with no space group is skipped with a message instead of prompting, and a
  batch run without spglib says so once at the start.
- **A walltime-killed geometry optimization continues where it stopped.**
  When an OPT runs out of time after at least one optimization step, the
  recovery adds `RESTART` to the OPTGEOM block of the same deck, so the new job
  picks up the earlier steps from OPTINFO.DAT in the same scratch directory
  (CRYSTAL23 manual 7.4.3) and uses the killed run's last density matrix as
  the SCF guess. A job killed in its first SCF has no OPTINFO.DAT and starts
  over as before. The job script checks for OPTINFO.DAT again on the compute
  node and falls back to a fresh start if it is missing. The killed run's
  output is kept as `<job>.out.timeout1`, `.timeout2`, ...
- **A RESTART rerun that aborts continues as a fresh optimization from the
  best geometry reached.** Measured on HPCC: the rerun re-evaluates the
  lowest-energy point, sees almost no energy change, shrinks the trust radius
  to zero and dies in `MPI_Abort` (`ERROR **** BFGS_ **** PXK TOO SMALL` only
  in the scratch `fort.87`; SLURM says COMPLETED). This is now recognised
  (error type `opt_trust_radius_error`, up to `max_retries` 2) and the same
  deck is resubmitted without `RESTART`, starting from the lowest-energy
  point reached by the runs of that deck (energies from runs that started
  from another geometry are not comparable: CRYSTAL fixes its integral
  screening at the starting geometry). Only the cell parameters and atom coordinates
  change - read from the geometry CRYSTAL prints at every optimization point,
  in the deck's own frame - and the deck as it was is kept first as
  `<job>.d12.orig` (then `.orig2`, ...; a backup is never overwritten). The
  aborted output is kept as `<job>.out.optabort1`, ... A timed-out OPT whose
  scratch directory has lost OPTINFO.DAT starts from its best point the same
  way instead of from the beginning. Only 3D decks given by space group are
  rewritten; for anything else the job is not resubmitted.
- **CRYSTAL's own error is read from the scratch `fort.87`** when the `.out`
  ends without CRYSTAL's normal end, so such aborts get a real error type (or
  `crystal_error` with CRYSTAL's message) instead of `unknown_error`.
- **Per-error `max_retries` is enforced** across the job's recovery lineage,
  on top of the overall limit of recovery attempts (the built-in default for
  timeouts is now 2, as in `recovery_config.yaml`).
- **Job scripts written before the RESTART staging existed get it on
  recovery.** A workflow copies the job-script generator into
  `workflow_scripts/` when it is planned, so the job scripts of a workflow
  planned before this release run a RESTART deck without checking for
  OPTINFO.DAT. The timeout recovery adds the current staging to the copy of
  the script it resubmits (the original is left alone), unless the script
  handles `fort.20` in some other way. Such a workflow's later steps are still
  first submitted with the old scripts; re-plan it to update them.
- **An empty `<job>.f9` is no longer staged as the GUESSP guess.** A run
  CRYSTAL aborts during an optimization still copies its (empty) fort.9 back.
- **The doubled walltime stays within the queue's limit.** It is checked with
  SLURM when the job is resubmitted (`sbatch --test-only`, plus the MaxTime of
  the partition the job runs in): 7 days on the general partitions and
  mendoza_q, 14 days for jobs submitted with `-A mendoza_q_long`. A job is
  never moved to another partition or account. A job already at the limit is
  resubmitted at the same walltime only when it can get further than last
  time - an OPT continuing with RESTART or from its best point. Anything else
  (an SP or FREQ, an OPT killed in its first SCF) would time out again, so it
  is not resubmitted and the log says so. `max_walltime` is used only when
  SLURM cannot be reached.
- **Timeouts are recognised at all.** SLURM writes the time-limit notice to
  the job's `-o` log and leaves the `.out` cut off mid-line, so a timed-out job
  was classified as an unknown error and never recovered. SLURM's TIMEOUT
  state, or that notice in the job's log, now marks it as a timeout.

- **The calculation type is read from a file's records, not its name or
  title.** `_opt` matched `_optimized`, so the queue manager typed every
  chained deck (`X_opt_..._optimized_sp_..._optimized`) submitted without an
  explicit type as an OPT; on the corpus that was every SP and FREQ deck and
  most BAND, DOSS, CHARGE+POTENTIAL and TRANSPORT decks. Keywords in a deck or
  output title (`..._BULK_OPTGEOM_...`) no longer make a run look like an OPT,
  CHARGE+POTENTIAL decks (ECH3/POT3) are no longer typed BAND, the recovery
  detector no longer reports FREQ and properties outputs as SP, DOSS outputs
  are no longer stored with band-structure fields, and `mace analyze
  --filter-type` matches outputs again. A deck's own records decide (OPTGEOM,
  FREQCALC, the properties keywords); a name is only a tie-break.
- **Generated SP, FREQ and OPT2 decks keep the parent's settings.** Decks the
  workflow engine derives from a finished calculation were quietly changing
  the method. Found by regenerating decks from real parents through every
  engine path and comparing them keyword by keyword with the parent:
  - the functional was guessed by substring, so PBE0, PBEsol, PBESOL0, PBEh-3c
    and LC-wPBE parents became PBE, UHF became RHF, and the no-plan SP path
    turned HF parents into HSE06-D3;
  - an HSEsol-3c SP parent produced FREQ/SP/OPT2 decks labelled `HSESOL3C-D3`
    (3c methods already contain D3 and gCP), or crashed before writing one;
  - the two-number `SHRINK IS ISP` form, the most common in practice, was
    regenerated from the cell (`5 10` became `7 14`, some meshes coarser), and
    anisotropic meshes such as `30 30 10` were flattened;
  - FMIXING, MAXCYCLE, BROYDEN, LEVSHIFT, SPINLOCK and BIPOSIZE/EXCHSIZE were
    reset to defaults, and OPT2 decks lost MAXTRADIUS or gained tolerances the
    parent never set;
  - a plan's FREQ tolerances and SP overrides were never applied;
  - with no plan, SP decks reset tolerances to Standard, added or dropped D3,
    and replaced the parent's own EXTERNAL basis with the library copy.

  The tolerance, basis, D3, SCF and SPINLOCK prompts now default to the
  parent's value; an explicit answer or plan setting still wins.
  A saved `--config-file` applied to a different material still gets a
  k-mesh from its own cell.

- **`opt2d12` prompts default to the parent's settings.** Pressing Enter now
  keeps what the parent had: OPT type and convergence (including tolerances
  that match no preset), MAXTRADIUS, level shifting, HISTDIIS, GUESSP, the
  DFT grid (now actually asked), CVOLOPT, and ITATOCEL/INTREDUN types.
  Answering "yes" to level shifting on a parent without it now takes effect,
  and a "no" to MAXTRADIUS survives `--save-options`/`--config-file`.
- **Parent functionals are kept exactly or flagged.** Functionals written as
  EXCHANGE/CORRELAT/HYBRID records were reduced to the exchange name, so a
  PBE0 written that way became PBE. Those records, range-separation records
  (SR-OMEGA, SR-HYB, ...), LSRSH-PBE's value line and a separate DFTD3 or
  GRIMME block are now carried over verbatim. A parent functional CRYSTAL23
  does not accept (e.g. B2PLYP, SCAN-D3) now prompts for a replacement, with
  HSE06 as the default; paths that cannot ask print a warning.
- **Deck titles are no longer read as keywords.** A title such as
  `..._BULK_OPTGEOM_...` made an SP parent look like an OPT and moved its SCF
  settings into the child's OPTGEOM block.
- **An OPT in an SP-first plan is built from the SP.** The engine's CIF
  re-conversion for this case passed flags NewCifToD12 has never accepted
  and always failed; it has been removed. An OPT planned after BAND, DOSS, TRANSPORT or
  CHARGE+POTENTIAL is built from the last OPT or SP; after TRANSPORT and
  CHARGE+POTENTIAL the workflow used to stop.
- **A FREQ parent's NUMDERIV is kept** in a follow-up FREQ deck, and is the
  default at the NUMDERIV prompt.

- **Basis-set coverage is measured, not assumed.** CRYSTAL counts neutrality as
  basis-set shell charges against nuclear charge, and its internal def2-mSVP
  library is broken for Pb. MACE emitted those decks silently because the
  compatibility check guarded on `basis_set in INTERNAL_BASIS_SETS` and the 3c
  sets have no entry there, so the element loop never ran. Coverage for every
  element 1-99 across all 12 internal basis sets was measured by running
  CRYSTAL23 on one-atom cells; raw results and the scanner live in
  `tests/basis_coverage/`. The hand-written ranges were wrong in the dangerous
  direction - POB-TZVP-REV2 claimed He, Ne, Ar and Tc. External sets are now
  checked for the file the run would actually read, since `read_basis_file`
  returns `""` for a missing one and silently drops that basis block.
- **An EXTERNAL basis is no longer truncated.** The basis block is terminated by
  its own `99 0` record, but the parser matched that as a *substring*. Basis and
  ECP data are columns of numbers and `99 0` occurs inside them constantly:
  lead's ECP line `12.296303 281.285499 0` ends `...499 0`, so the parser stopped
  mid-ECP and dropped Pb and every element after it. CRYSTAL rejected the result
  with `ERROR **** INPBAS **** FORMAT ERROR IN INPUT DECK`. Measured over the
  corpus, 13 of 106 EXTERNAL decks truncated this way, every one Pb-bearing;
  all 106 now extract byte-for-byte. This is why OPT steps succeeded while every
  SP failed - OPT decks come from a CIF, SP decks are regenerated from the OPT's
  D12 through this parser.
- **Deck-breaking keyword errors.** `functional_keyword_map` rewrote three
  correct CRYSTAL keywords into strings that appear nowhere in the manual: VBH,
  PWGGA and WCGGA became VBHLYP, PW91GGA and WCGGAPBE. BROYDEN was written bare,
  without the W0/IMIX/ISTART record it requires. `OPT_TYPES` offered INTONLY and
  ITATOCELL, neither a CRYSTAL keyword; both are now ATOMONLY. The planner
  emitted prose spellings (HF-3C, M06-2X) where the input keywords are HF3C and
  M062X, offered "HF" which is not a Hamiltonian, and appended `-D3` to any
  functional rather than the ten the manual parametrizes.
- **Silently wrong geometry.** Rhombohedral cells written with IFHR=1 emitted
  `a c` where the manual requires `a alpha`, so alpha was read as 5.13 degrees
  instead of 55.29 - a collapsed cell, no error raised. SLAB decks always wrote
  three cell parameters instead of the minimal set, so CRYSTAL consumed `b` as
  the atom count.
- **SLAB and POLYMER group numbers.** The record after SLAB is a *layer* group
  (Appendix A.2, IGR 1-80) and after POLYMER a *rod* group (A.3, IGR 1-99), but
  the converter wrote the 3D space-group number into both. A hexagonal slab
  (space group 191) made CRYSTAL MPI_Abort having written no `fort.87` at all,
  so MACE's own error classification saw nothing. POLYMER failed more quietly,
  since a wrong number is usually inside 1-99 and simply builds a different
  chain. The reverse map is deliberately partial - 45 of 230 space groups for
  layer groups, 75 for rod groups - and refuses rather than guesses where the
  appendix row is not in the International Tables first setting or where several
  candidate orientations exist.
- **Slabs whose symmetry axis is off the in-plane origin are refused.** Every
  layer group the map can produce except P1 carries a rotation axis along z that
  CRYSTAL places at the in-plane origin before expanding the asymmetric unit
  about it. A structure whose axis sits elsewhere expands into a different slab
  and runs to completion looking correct. This is a necessary condition, not a
  sufficient one; `mace preflight` remains the authoritative check.
- **Silently wrong properties.** Four BAND fallback paths wrote fractional
  k-points where CRYSTAL expects integers in units of 1/ISS, aborting the read or
  collapsing the path to GAMMA. IRSPEC was suppressed whenever no dielectric
  tensor was supplied, so IR spectra were never generated at all. SCELPHONO was
  written inside FREQCALC rather than the geometry block, and with three values
  where a nine-real expansion matrix is required. POTC's ICA was off by one.
- **HF plus an external basis never closed the geometry block**, so every
  RHF/UHF deck built from a CIF aborted immediately.
- **The workflow dropped a configured node exclusion.** The planner collects one
  and writes it into the scripts it generates, but Phase 0 regenerates every
  script from the plan at execution time and discarded it - nothing in the
  resource mapping matched `--exclude`. Workflow jobs therefore ran with no
  exclusion while `mace submit`, which honours the same setting, kept it. Only an
  explicitly configured exclusion is written; with none, the template is
  unchanged.
- **Unattended runs no longer die on a prompt.** Three `stdin.isatty()` sites
  were unguarded, including the missing-element prompt - which, now that the
  compatibility check actually fires, would have hit `EOFError` mid-sweep. The
  batch symmetry-mismatch path likewise reaches an explicit logged decision
  instead of surfacing as "Error during symmetry analysis".
- **Documentation version drift.** The README banner still said 1.1.0. The
  single-source version test only scanned `.py` files and `mace_cli`, so a
  version written in prose had nothing holding it to the package; it is now
  pinned. Historical "NEW in v1.1.0" notes and the v1.0.0 Zenodo citation are
  deliberately left alone.
- **A job no longer runs CRYSTAL on an empty INPUT when `$SCRATCH` is empty.**
  On some nodes (agx-000) `$SCRATCH` is empty inside the job even under
  `bash --login`, so the scratch directory became `/crys23`, nothing could be
  staged there and CRYSTAL stopped with `END OF DATA IN INPUT DECK` (reproduced
  on HPCC with the old script). The OPT/SP/FREQ and properties job scripts now
  check the scratch directory before staging into it and otherwise use, in
  order, `/mnt/scratch/$USER` (what HPCC sets `$SCRATCH` to, so the same
  directory), `.mace_scratch/` in the submit directory, then `$TMPDIR`, saying
  which in the job log. With none writable the job stops before CRYSTAL runs,
  and `<job>.out` then holds only that error (the previous output is kept as
  `<job>.out.prev<N>`, or overwritten where it cannot be moved), so an earlier
  run's results or error are never read as this one's. The stop is classified
  as `scratch_error` (`scratch` in `mace check`) and is not recovered
  automatically: every fallback, the shared submit directory included, was
  unwritable, so a resubmission would stop the same way. The directory used is
  recorded as `.<job>.scratch` in the submit directory; the recovery reads it
  to find OPTINFO.DAT and fort.87, and a RESTART rerun that lands in a
  different directory takes OPTINFO.DAT, fort.20 and fort.9 together from
  whichever directory has the newer OPTINFO.DAT.
  When `$SCRATCH` is empty where the recovery runs, it now rebuilds it the same
  way instead of treating the scratch directory as unknown.
- **`recovery_config.yaml` is read.** The recovery engine only looked for the
  file in the directory it ran in, so the one shipped in `mace/config` never
  applied. It now uses an explicit `--config` file, else `recovery_config.yaml`
  in the current directory, else the shipped one. A file is merged into the
  built-in defaults key by key, where it used to replace the whole
  `error_recovery` section (so a partial file silently dropped every recovery it
  did not mention). An entry naming a handler that does not exist is reported;
  for an error type MACE recovers, the built-in handler is used with the
  entry's other settings, so its `max_retries` still counts and
  `manual_escalation` means `max_retries: 0` - a file can never switch on a
  recovery it switched off. The shipped file named five handlers that do not
  exist; it now lists exactly the recoveries MACE performs, with the built-in
  values. With no `recovery_config.yaml` in the job directory (the usual case)
  every recovery runs as before. An older copy of the shipped file in a job
  directory (the legacy `copy_dependencies` put one there) still keeps
  disk-space clean-up off (`max_retries: 0`), but is now merged instead of
  replacing the section, so the recoveries it does not mention - the newer
  optimization fresh start for a collapsed step size - are no longer dropped.
- **`opt2d12 --directory DIR --config-file T` no longer asks, or stops, per
  file.** It asked "Apply these settings from config file? [Y/n]" for every
  file; with nothing on stdin each file logged "EOF when reading a line", no
  deck was written, and the run exited 0. A single `--out-file` with nothing on
  stdin (or a closed one) failed the same way. See Added for the batch run.
- **A saved `opt2d12` template applies to other structures.** Settings saved
  with `--save-options` from a parent with an EXTERNAL basis stored the marker
  "EXTERNAL (from original D12)" as the basis; applied with `--config-file` to
  any internal-basis parent the deck failed with "External basis set path not
  configured properly" (78 of the 118 `test/OPT` parents with the report's
  Ag1Br1 template). That marker now means "each structure's own parent basis":
  an external-basis parent keeps its own basis records, an internal one its
  named basis. A template that names an internal basis (POB-TZVP-REV2) or an
  external basis directory still applies it to every file. Templates saved by
  earlier versions load and follow the same rule.
- **A deck that moves an external-basis parent to an internal basis writes
  plain atomic numbers.** The parent numbers ECP atoms by its basis (silver
  247); under BASISSET the deck now writes 47, and the basis-coverage check,
  which skipped every such deck, runs on it (an HSESOL3C template on the five
  lead parents in `test/OPT` is now refused: SOLDEF2MSVP has no Pb).
- **A template from a P1 structure no longer writes every atom of a symmetric
  one.** Its "write all atoms" was applied as is, so an HSESOL3C template saved
  from a molecule wrote diamond's two carbons under space group 227 (31 of 118
  `test/OPT` decks). A structure with symmetry is written as its asymmetric
  unit.
- **A lone `opt2d12 --out-file X.out` uses the `X.d12` beside it.** Without
  `--d12-file` it ran from the `.out` alone, with or without `--config-file`
  (a shell glob that matched one file did the same), so the parent deck's
  settings were silently replaced by what the `.out` shows: Ag1Br1's external
  ECP basis became POB-TZVP-REV2, while the log said the parent's basis was
  kept. It is now paired like each file of a batch; a file with no `.d12`
  beside it says it is converted from the `.out` alone.
- **`opt2d12` never answers the basis-coverage question for you.** With
  `--yes`, `--non-interactive` or a `--config-file` batch, a basis that lacks
  an element (SOLDEF2MSVP and Pb) still asked "Do you want to continue
  anyway?" at a terminal, and Enter wrote a deck CRYSTAL rejects. That file
  now fails with the reason, the same with a terminal as without. With stdin
  closed (`<&-`) opt2d12 crashed with AttributeError where it checked for a
  terminal; it is treated as nobody to ask.

### Added

- **The `.d12` parser reads the whole geometry input**: the space, layer, rod
  or point group record, CRYSTAL's IFHR (rhombohedral axes) flag, the cell
  record expanded to a, b, c, alpha, beta, gamma, and every atom (atomic
  number and coordinates). It used to stop at the space group. Every CRYSTAL
  and SLAB deck in `test/` (137) is rebuilt from the parse, record for record,
  by MACE's own deck writer. `opt2d12` still takes a child's geometry from the
  `.out`: its decks are byte-identical to before for all 1458 SP/OPT/FREQ
  children of the `test/OPT` and `test/SP` parents. A deck that names its
  space group by symbol (IFLAG = 1) is read too, in the spaced form CRYSTAL
  takes ("F M 3 M", "P 21/C", "P -4 21 M", "R -3 C"); a spelling CRYSTAL23
  refuses or misreads ("FM3M", "F M -3 M", "P -4 2 1 M") is reported in the
  parse as `geometry_unparsed` instead of silently leaving the geometry out.
- **`mace preflight`** - runs CRYSTAL over a copy of a deck with a TESTPDIM
  record inserted. TESTPDIM stops after the whole input is read and symmetry
  analysed, which is late enough to catch a bad group, a bad lattice record, or a
  basis set with no functions for one of the elements. Measured at about a second
  per deck, against a queue wait to learn the same thing. A run that produced no
  `fort.87` *and* no success marker counts as a failure, because that is exactly
  how the hexagonal slab died.
- **One `opt2d12` template for many structures.** `--config-file` applies to
  every file of `--directory DIR` or of several `--out-file` paths (a shell
  glob such as `opts/*.out`; each `.out` is paired with the `.d12` of the same
  name beside it). Before any file it prints the plan (config, file count,
  output directory, the settings it applies) and, at a terminal only, asks
  once "Apply to all N files? [Y/n]"; `-y`/`--yes` skips that, and with no
  terminal nothing is asked. A single file still asks at a terminal, and a
  piped "y" works as before. Each file gets a status line and the run ends
  with "N written, M failed" and each failure's reason; it exits non-zero if
  any file failed or none was written (as does any `--directory` run now,
  with or without a config; files are taken in name order). Nothing is asked
  per file: a question a file would need (a basis without one of its
  elements) fails that file instead. A path given twice is converted once,
  and two inputs whose decks would have the same name in the same directory
  (`dupA/X.out` and `dupB/X.out` with one `--output-dir`) both fail rather
  than one overwriting the other. A parent optimisation CRYSTAL never finished
  (no OPT END: killed, out of time) or stopped unconverged at the cycle limit
  ("OPT END - FAILED", which prints no final geometry) is still converted, from
  its starting geometry, and flagged on its line and in the summary: "N
  written (K from unfinished or failed optimisations), M failed". On the 118 `test/OPT` parents the batch
  decks are byte-identical to the single-file config path run once per file,
  for templates saved from an internal-basis, an external-basis and an
  HSESOL3C parent.
- **SLABCUT**, opt-in via `options['slabcut']`. The manual builds a slab from the
  3D structure and derives the layer group itself, sidestepping the orientation
  ambiguity that forces a refusal above. It needs two runs - a SLABINFO probe to
  number the atomic layers, then the real cut.
- **GUESSP SCF restart.** Every run already saved its converged density matrix as
  `$JOB.f9` and nothing ever read one back. CRYSTAL reads the guess from
  `fort.20`, so the job script now stages one. Measured on the same MgO cell,
  functional and basis: cold start -275.14904353964 AU in 11 cycles against
  -275.14904354956 AU in 5. It is opt-in - a staged matrix is only a valid guess
  for the same geometry *and* basis set. Where no matrix exists the script strips
  the GUESSP record from the scratch copy, because CRYSTAL does not fall back to
  the atomic guess; it stops with
  `ERROR **** GUESSP **** COPY OF WAVEFUNCTION FILE fort.20 CAN NOT BE FOUND`.
- **A stale node-exclusion warning.** `MENDOZA_NODES` is a hardcoded list and the
  partition is not static. A retired node leaves a harmless dead entry, but a
  *renamed* node silently stops being excluded while the menu still reports that
  it is. The exclusion menu now checks the list against `sinfo` and says
  something only when there is drift, staying silent off-cluster or when the
  check fails.
- **ECHG map planes can come from a config** instead of only interactive
  prompts, so an ECHG deck is reachable from a workflow at all.

### Changed

- **One set of convergence presets.** The planner, `opt2d12`, `cif2d12` and
  the quick-start plan share `opt2d12`'s Standard / Tight / Very tight tiers
  for OPT convergence and SCF tolerances. The planner's default OPT is now
  Standard (TOLDEG 0.0003, TOLDEX 0.0012), which it previously set 10 times
  tighter; planner steps with no settings of their own inherit the parent's.
  The planner's ATOMSONLY is corrected to ATOMONLY.
- **FREQ decks default to Very tight SCF tolerances** (TOLINTEG 9 9 9 11 38,
  TOLDEE 11) on every path unless a value is given.
- **NUMDERIV is written only when asked for.**
- **The basis-set menu asks the measured tables.** For any structure with max
  Z > 36 it previously offered only POB-DZVP and POB-TZVP, neither of which
  carries Pb, while filtering out POB-TZVP-REV2, which does - the menu was itself
  a cause of the reported failure. It now falls back to an external set when
  nothing internal covers the structure.
- **VBH, PWGGA and WCGGA are emitted as EXCHANGE/CORRELAT pairs**, which is how
  CRYSTAL exposes them; there is no standalone keyword. The pairings are the
  manual's own. The input parser reads such pairs back to the single name MACE
  uses - the same gap already existed for PBESOL and SOGGA, whose decks never
  round-tripped either.
- **The SPINLOCK prompt no longer claims -1 gives an antiferromagnetic guess.**
  It does not: SPINLOCK locks the cell total n_alpha - n_beta. The manual's AFM
  recipe needs ATOMSPIN, which MACE does not implement.
- **`FUNCTIONAL_KEYWORD_MAP` is deleted** rather than corrected. It was imported
  in one place and never read there, and half its entries were wrong in ways that
  would have been silent - LC-wPBE and LC-BLYP, both range-separated hybrids,
  were mapped onto Wu-Cohen exchange records.
- **`opt2d12 --save-options` stores settings, not a structure.** The template
  no longer holds the parent's atoms, optimisation log, external basis records
  or k-point mesh (the report's template drops from 90 KB to about 1 KB). No
  opt2d12 question sets the mesh, so the stored one was always the parent's,
  and it made every other structure's mesh be regenerated from its cell
  (Ag1Cl3's SHRINK 8 16 became 9 18). Each structure now keeps its own
  parent's SHRINK, also with a template saved earlier: a config's `k_points`
  is not applied. To give every file one mesh on purpose, write
  `"k_points_for_all_files": "8 16"` (or `"8 8 8"`, or one number) into the
  config by hand; it is written exactly as given.
- The interactive CIF-converter banner credits its author only; the trailing
  tool-attribution clause is gone, and a test keeps it that way.

### Testing

771 tests at the start of this work, 1074 now, with the regression tests
asserting against real decks under `test/` rather than fixtures wherever one
exists.

## [1.1.1] - 2026-08-18

Verified end-to-end on MSU HPCC (SLURM + CRYSTAL23): a bare submission runs only
what was submitted, and a `--progress full_electronic` run drove
OPT → SP → BAND + DOSS to completion inside its own plan directory.

### Changed

- **A manual submission no longer starts a workflow.** Every job script MACE
  writes ends with the queue-manager completion callback; outside a workflow
  that callback used to adopt every untracked `.d12`/`.d3` in the directory
  tree and then progress the completed job from built-in defaults (OPT → SP,
  SP → BAND + DOSS) inside a synthesized `workflow_outputs/workflow_<time>/`
  directory. A hand-run `mace submit` now runs exactly what was submitted;
  progression requires a plan. `MACE_PLANLESS_PROGRESSION=1` restores the old
  behavior. `mace manager` and `--callback-mode submit_new` are unchanged —
  keeping the queue fed is what they are for.

### Added

- `mace submit --progress TEMPLATE|interactive` — plan the steps that should
  follow decks you built by hand, then run them as each job completes. Writes
  a real workflow plan (no CIF conversion: the existing decks are the starting
  point) and stamps each submission with its workflow ID, so progression
  follows the plan rather than defaults. `interactive` uses the planner's own
  step prompts. Implies `--track`. A deck may enter the sequence at any step.
- `mace submit --walltime TIME` — override the SLURM time limit for a
  submission instead of taking the per-calc-type defaults (OPT 7 days, SP 3
  days). Short jobs queue sooner, which matters for test and QA runs. Applies
  to the manual D12/D3 submitters, the tracked path, and every step of a
  `--progress` plan.
- Short flags for `mace workflow`: `-i/--interactive`, `-e/--execute`,
  `-q/--quick-start`, `-s/--status`, `-T/--show-templates`, `-c/--cif-dir`,
  `-d/--d12-dir`, `-w/--workflow`, `-W/--work-dir`, `-D/--db-path`,
  `-j/--max-jobs`.

### Fixed

- **`mace opt2d12 --calc-type X --non-interactive` no longer dies with EOFError**
  when nothing is on stdin - the form MACE's own help documents. It walked the
  interactive settings flow (the workflow engine drives that flow with scripted
  answers) and crashed at the first prompt. When no answers are available it
  now keeps the settings extracted from the source calculation. The engine's
  generated decks are byte-identical before and after.
- **`mace opt2d12 --output-dir` is honoured.** It was parsed and the directory
  created, but the path never reached the writer, so decks always landed in the
  current directory. The deck's CRYSTAL title line still carries only the file
  name.

- **Follow-up steps could be orphaned from their workflow.** In an isolated
  context the completion callback re-registers a finished job from its output
  file, and that scan recorded no workflow metadata. Progression still found
  the plan (it falls back to `$MACE_WORKFLOW_ID`), but the output-directory
  resolver read the same empty settings and minted a fresh
  `workflow_<timestamp>` directory with no plan file — so the step it generated
  was stamped with a dead id, and when that step finished no plan could be
  found and the chain stopped. The resolver now uses the same environment
  fallback, and the scan stamps the workflow id onto the records it creates.
  Previously masked: the old no-plan branch emitted default BAND + DOSS
  regardless, so the chain looked correct while the id was already wrong.
- `WORKFLOW_TEMPLATES` was missing `opt_sp_freq`, which the CLI already
  accepted.
- **Template BAND steps used a lower-quality k-path than the interactive
  planner.** `seekpath_full` is the only flag that makes the D3 generator take
  the SeeK-path (HPKOT) branch, and the quick-start / `--progress <template>`
  BAND config never set it — so those BANDs fell back to the built-in
  extended-Bravais tables while the planner's expert config got the full path.
  Wrong paths only for non-cubic lattices, and silent either way. The flag is
  now requested unconditionally: plans are built on a login node while the D3
  is generated later on a compute node, so probing for the library at plan time
  would test the wrong machine, and `get_seekpath_full_kpath` already degrades
  seekpath → literature → static on its own. Verified on HPCC (diamond):
  `default - X-GAMMA-L-W-GAMMA` (4 segments) became
  `SeeKPath (w.I) - GAMMA-X-U-|-K-GAMMA-L-W-X` (6 segments).

## [1.1.0] - 2026-07-12

Changes since 1.0.5.

### Added

#### Themed terminal UI (visual layer)
- `mace/utils/ui.py` rich-based facade: status lines, tables, progress bars,
  spinners, live dashboards, themed startup banner
- Selectable color themes (`--theme <name>`, `--save-theme`, `MACE_THEME`)
- Fully optional: degrades to plain text without `rich`; honors `NO_COLOR` and
  `TERM=dumb`; user text is never interpreted as markup (injection-safe)

### Fixed
- **d12/d3 generation** — an aborted D12 creation no longer leaves a truncated deck reported as success; TOLINTEG extraction preserves custom tolerances on pure-DFT outputs; batch mode re-prompts on invalid input.
- **Database** — TRANSPORT and CHARGE+POTENTIAL outputs use the canonical material ID (no duplicate rows), and transport statistics count Seebeck entries only.
- **Plotting/UX polish** — missing/unreadable spectra files give clean errors instead of tracebacks; `mace plotting` propagates its exit status; malformed `--iso` is a usage error; plain-mode (no-rich) output preserves bracketed text verbatim.
- **HPC QA campaign (release wave)** — ~18 workflow/queue/recovery fixes from an end-to-end SLURM test campaign. Highlights: one workflow mints ONE workflow id (fixes the nested-DB split-brain between engine and queue manager); job-state checks confirm via `sacct` before failing jobs missing from `squeue`; recovery attempt caps count the whole lineage, so resubmission chains stay bounded; TRANSPORT d3 decks emit `NEWK` before `BOLTZTRA` and get their properties terminator `END`; CIF conversion without spglib writes the CIF's asymmetric unit instead of the expanded cell; plotting pins a headless matplotlib backend (`Agg`) for compute nodes.

### Changed
- **Repo hygiene** — internal planning/audit docs untracked (kept on disk), generated artifacts gitignored, unused legacy modules removed (legacy queue manager, portable SLURM generator, contextual executor/planner variants, installer/env-helper utilities); PyPDF2 dropped as a dependency (nothing imports it).
- **Formula ordering** for newly extracted formulas follows a revised element convention (e.g. `TiPbO3` vs the older `PbO3Ti`); previously stored rows are unaffected. `fermi_energy` rows written by v1.1.0 carry the correct `Hartree` unit label (older rows said `eV` while storing Hartree values).

## [1.0.5] - 2026-06-14

Changes since 1.0.0. Tagged retroactively: 1.0.5 was the version `mace --version`
reported at this point, but it was never separately announced, so its changes
were first described in the 1.1.0 notes. 1.0.1 through 1.0.4 were never released.

### Added

#### Plotting subsystem (`mace plotting`)
One command for publication-ready plots from CRYSTAL outputs, with content-based
file detection, per-kind flags (`--band --dos --structure --cube --freq --ir
--raman --all`), an interactive menu, and `-o` output routing:

#### Deep property extraction
The materials database now stores the full scientific results of each calculation
type, not just scalar summaries — as compact JSON plus flat, queryable scalar rows
in the existing `properties` table (no schema change):

#### Queue / submission
- In-place submission for manual `mace submit` (no forced reorganization);
  `--organize` restores the copy-into-folders layout
- `completion` command surfaced in `mace --help`; all command help audited
  against the real parsers (fabricated flags removed)

### Fixed
- **FREQ extractor** previously parsed vibrational data into a discarded local variable; frequencies/IR/Raman now actually persist (fixed the units-anchor `(CM**-1)` and mode-range parse bugs).
- **Error-recovery chain** — previously-dead paths called nonexistent DB/manager APIs (errors swallowed): max-recovery-attempts, recovered-job resubmission, and workflow-engine step submission now work end-to-end; the timeout handler parses the `-t 7-00:00:00` day form and never shrinks walltime; the memory handler preserves `--mem-per-cpu` vs `--mem`; recovered resubmissions record the bumped script so repeated failures escalate cumulatively.
- **d12/d3 generation correctness** — SPINLOCK parse + round-trip (a configured spin lock survives OPT continuation and JSON-config reuse); origin-setting preservation; k-point table fixes (C-/I-centered orthorhombic assignments, duplicate table key); DOSS Fermi-window unit consistency.
- **JSON config save/apply round-trips** for `opt2d12` / `opt2d3` (settings no longer drift through a save→load cycle); invalid interactive calc-type choices re-prompt instead of silently defaulting to BAND.
- **Database correctness** — canonical material-ID derivation, NULL-safe dedup on re-extraction, pressure unit-conversion table (kbar/Mbar swap, atm factor), enthalpy H = G + TS, full-precision Hartree↔eV constants (single source: `mace/constants.py`), pyarrow import-order crash guard.

### Changed
- **Repo hygiene** — development and internal artifacts kept out of the repository; ad-hoc validation scripts centralized under `tests/`.
- **CI** — GitHub Actions runs the self-contained test suite on a fresh clone (data-dependent tests skip without the local `test/` corpus).

## [1.0.0] - 2026-02-12

### Added

#### Core Framework
- **MACE CLI** (`mace_cli`) - Unified command-line interface for all MACE functionality
- **Workflow Manager** - Complete end-to-end workflow planning and execution system
- **Material Tracking Database** - SQLite + ASE integration for calculation history and provenance
- **Enhanced Queue Manager** - Intelligent SLURM job scheduling with material tracking
- **Error Recovery System** - Automated detection and fixing of common CRYSTAL errors

#### Input Generation
- **NewCifToD12.py** - CIF to CRYSTAL D12 input file conversion
- **CRYSTALOptToD12.py** - Generate inputs from optimized structures
- **CRYSTALOptToD3.py** - Unified D3 generation with basic/advanced/expert modes

#### Band Structure
- **Seekpath Integration** - Accurate k-path generation using the seekpath library (HPKOT methodology)
- Support for all 26 extended Bravais lattice types (cF1, cF2, hR1, hR2, mC1, etc.)
- Proper handling of parametric k-points for non-cubic lattices
- Automatic SHRINK factor calculation for exact integer k-point coordinates

#### Property Calculations
- Band structure (BAND) input generation with automatic k-path detection
- Density of states (DOSS) with orbital resolution
- Transport properties (Boltzmann transport calculations)
- Charge density and electrostatic potential analysis

#### Workflow Features
- Interactive workflow planning with three customization levels (Basic/Advanced/Expert)
- Pre-defined workflow templates (basic_opt, opt_sp, full_electronic, double_opt, complete)
- Workflow isolation for running multiple workflows in the same directory
- JSON-based configuration persistence for reproducibility

#### File Management
- Complete file storage with settings extraction from D12/D3 files
- SHA256 checksums for file integrity verification
- Organized storage by calculation ID with metadata preservation

#### Monitoring
- Real-time calculation monitoring dashboard
- Completion status checking with `mace completion`
- Zombie job detection and cleanup

### Dependencies
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- ase >= 3.22.0
- spglib >= 1.16.0
- PyPDF2 >= 2.0.0
- pyyaml >= 6.0
- pandas >= 1.3.0
- seekpath (optional, recommended for accurate band structure k-paths)

---

## Roadmap (not yet scheduled)

### Planned
- PyPI package distribution
- Additional workflow templates
- Enhanced visualization tools
