# Changelog

All notable changes to MACE (Mendoza Automated CRYSTAL Engine) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Conformance work traced from a user report: every hybrid lead perovskite died
with `ERROR **** LoadBa **** UNIT CELL NOT NEUTRAL` under HSE-3c, on cells that
are neutral. Auditing that one bug against the CRYSTAL23 manual turned up a
family of others. Every behavioural change below was run through
CRYSTAL/23-intel-2023a on real hardware, not just reasoned from the manual.

### Fixed

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

### Added

- **`mace preflight`** - runs CRYSTAL over a copy of a deck with a TESTPDIM
  record inserted. TESTPDIM stops after the whole input is read and symmetry
  analysed, which is late enough to catch a bad group, a bad lattice record, or a
  basis set with no functions for one of the elements. Measured at about a second
  per deck, against a queue wait to learn the same thing. A run that produced no
  `fort.87` *and* no success marker counts as a failure, because that is exactly
  how the hexagonal slab died.
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
