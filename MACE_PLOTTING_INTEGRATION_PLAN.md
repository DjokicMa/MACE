# MACE Plotting Integration Plan — ECH3/POT3 Cube, FREQ Vibrational Modes, IR/Raman Spectra (FINAL)

## 0. Implementation Status & Corrections — updated 2026-06-13

**The correctness fixes (§2) are implemented and regression-tested in place** in
`test/AddedPlottingFunctionalty/` (gitignored; Phase 3 relocates these engines
into `mace/plotting/`, carrying the fixes). Tests live in tracked `tests/`:

| Fix | Where | Test | Status |
|-----|-------|------|--------|
| 2.1 freq tail-block + imaginary modes | `vibmode_viewer.py` | `tests/test_vibmode_parser.py` | ✅ done |
| 2.2 degenerate E/T modes (positional cursor) | `vibmode_viewer.py` | `tests/test_vibmode_parser.py` | ✅ done |
| 2.5 element/ECP heavy-element coverage (scoped) | `vibmode_viewer.py` | `tests/test_vibmode_parser.py` | ✅ done |
| render guard: displacement/atom-count mismatch | `vibmode_viewer.py` | `tests/test_vibmode_parser.py` | ✅ done |
| 2.3 non-orthogonal cube subtraction (full-affine interp) | `subtract_cubes.py` | `tests/test_cube_subtract_nonortho.py` | ✅ done |
| 2.3 slice fidelity — single / `--slice-all` / `--slice-browse` (go.Surface for skew) | `crystal_cubeviz_plotly.py` | `tests/test_cube_slice_skew.py` | ✅ done |
| 2.6 negative-natoms multi-dataset cubes | `crystal_cubeviz_plotly.py`, `subtract_cubes.py` | `tests/test_cube_negative_natoms.py` | ✅ done |

**Empirical coverage (validated, not just unit-tested):** vibmode renders across
0D/2D/3D × P1/cubic/tetragonal/monoclinic/rocksalt × organics/ECP-heavy/halides;
cube viz across 3D cubic/tetragonal/hexagonal/triclinic + heavy elements + slabs
of all orientations + all 10 real graphite-slab variants (synthetic cubes used
for the 3D-crystal/alt-slab geometries the corpus lacks); all 5 IR/Raman plotters
across the real molecular SPEC.DAT set. Full suite: **455 passed**.

**Corrections to the analysis below (which predates the data being complete):**
- **§1 / §7 "zero SPEC.DAT" is OUTDATED.** `test/FREQ` contains **264** spectra
  files (132 `IRSPEC.DAT` @ 3 cols + 132 `RAMSPEC.DAT` @ 10 cols), all 0D
  molecular. The spectra leg is **validated, not blocked**. No crystalline/slab
  SPEC exists, so the 2.4 reduced-column risk is real but untriggered by current
  data; every real file matches the scripts' 3/10-column gates.
- **`--slice-all` hardened:** large cubes (e.g. a 500-slice slab) are now
  subsampled to a 36-panel grid (previously a hard plotly spacing `ValueError`).

### Phase 3 wiring — DONE (2026-06-13, branch `plotting-phase0-wiring`)

Cube + FREQ are now first-class `mace plotting` kinds end-to-end:

| TODO | What | Commit |
|------|------|--------|
| #1 | Relocate `crystal_cubeviz_plotly.py` + `vibmode_viewer.py` into tracked `mace/plotting/engines/` (the gitignored `test/` copies never shipped); re-point regression tests at the package (they previously skipped on a clone) | `52620a41` |
| #2 | `detect.py` content-sniff: `is_freq_output` (C1 phonon-banner + NORMAL-MODES gate); discover/classify made sniff-aware (cube/spectra stay glob-only) | `1c04f8c8` |
| #3 | `handlers/cube.py` + `handlers/freq.py` — argv-driving the engines headlessly (`--save`/`--gif`, never `fig.show()`); registry entries (FREQ sniff = `is_freq_output`); engine-args escape hatch screened (L4); representative-mode default (H5) | `b2ff3581` |
| #4 | CLI: `--cube`/`--freq`(`--vibmodes`) + option groups + positional FILE(s); `--diff` 2-operand rule (H4); `--all-modes` outside mode group (H5); classify-by-content dispatch | `4168ed55` |
| #6 | `requirements.txt` declares plotly/scipy/kaleido; help synced (epilog + `mace_cli`) | `19e2b7ab` |

E2E verified through the real CLI (`--freq … --list-modes`; `--cube … --iso` →
HTML). Full suite: **484 passed**.

**Deviations from the plan, with rationale:**
- **Engine relocation pulled forward from Phase 6 → Phase 3 (TODO #1).** The plan
  assumed engines could be consumed in place via `sys.path-insert`, but the whole
  `test/` tree is gitignored (`.gitignore:30`), so they would not ship. Relocating
  was a hard prerequisite, not optional. Only the two consumed engines moved; the
  standalone diagnostics (`subtract_cubes`/`diagnose`/`slice_browser`) stayed —
  the cube engine's own `interpolate_cube_at_vertices` diff path is already
  non-orthogonal-safe, so they are redundant.
- **No richer `Detected` dataclass / cube-subtype PlotKinds.** Phase 0 had already
  folded cube sub-type into engine-resolved metadata (one `PlotKind.CUBE`), which
  also sidesteps the C2 two-sources-of-truth risk: detect never subtypes cubes;
  the engine remains the sole authority.

### Phase 4 spectra (IR/Raman) — DONE (2026-06-14, branch `plotting-phase0-wiring`)

| Step | What | Commit |
|------|------|--------|
| H3 + lift | `handlers/spectra_api.py` — pure read/render/average lifted from `plotIRRAM.py` (verified superset; the 5 scripts' `read_*` are byte-identical → one `read_spectrum`). 2.4 fixes applied: column gate relaxed `!=N`→"≥ needed", no `os.getcwd()`, no `input()`, format parametrized, ragged-grid-safe averaging | `cbc3a2de` |
| handlers | `handlers/spectra.py` — SPECTRA_IR (`*IRSPEC.DAT`) + SPECTRA_RAMAN (`*RAMSPEC.DAT`) registry entries; conf-style per-material averaging; raman modes total/par_perp/all; `--format html`→png fallback | `36f5a048` |
| CLI | `--ir`/`--raman`/`--spectra` (umbrella via `_dispatch_kinds`) + `--raman-mode`/`--average`/`--ir-column`; help synced | `77ad8a42` |

Discovery is glob-only (suffix-anchored `*IRSPEC.DAT`/`*RAMSPEC.DAT`, C3-safe — never
column-sniffs arbitrary `.dat`). E2E verified on real data (`--spectra --raman-mode
all` → IR absorbance png + Raman all png). Validated against 132 IRSPEC (3-col) +
132 RAMSPEC (10-col). Full suite: **520 passed**.

**Scoped out of Phase 4 (deferred, low value / no inputs):**
- `--raman-component {xx..zz}` single-direction selector (the `all` mode already
  plots all six; no validated single-direction plotter exists to lift).
- `--from-freq` / `SPECTRA_FROM_FREQ` derivation (synthesize spectra from FREQ runs
  with `can_spectra`). Real `*SPEC.DAT` exist for every FREQ run in the corpus, so
  this adds nothing now; needs the FREQ `can_spectra` sniff first.

**Still outstanding:**
- **TODO #5 — `grid_geometry.py` shared affine: DEFERRED** (Phase-6-style). The
  validated engines carry their own inline affine (tested), and no new consumer
  needs the shared helper, so extraction is pure DRY refactor with risk to tested
  code — deliberately not done. Revisit with the Phase 6 library migration.
- **Phase 6 — library migration** of the two big engines is already effectively
  done (TODO #1 relocated them); only the optional `grid_geometry` extraction
  remains.

Plotting integration is now functionally complete for all five kinds: band, DOS,
structure, cube, FREQ, IR, Raman.

---

## 1. Executive Summary

This plan adds three plotting capabilities to the existing `mace plotting` command: (1) ECH3/POT3 cube visualization (charge density / electrostatic potential / spin / difference), (2) FREQ vibrational normal-mode 3D viewing, and (3) IR/Raman spectra. The literal goal — *auto-pick the correct visualization from the input, with explicit overrides* — is a **classification problem**, so the architecture is built around a content-sniffing classifier + registry spine (`detect.py` → `PlotKind`), not a flat extension of the existing filename-glob discovery, which provably cannot distinguish the dataset's sub-types (228 `.CUBE` all share extension; 96 FREQ-signature `.out` are indistinguishable by extension from ~540 ordinary `.out`).

**Chosen approach (verified against the codebase):**
- **Registry + content-sniff classifier** is the structural spine. It collapses the four hand-synced integration points the current code requires (`discover_plottable_files` L90, `print_discovered_files` L125, `run_interactive` option_map ~L605, `--all` aggregation) into a single registration call.
- **sys.path-insert invocation** (the convention `main.py` already uses three times for `Plotting/`, lines 263/430/551) for the two large, import-safe engines (`crystal_cubeviz_plotly.py`, `vibmode_viewer.py`). **Verified:** both have `__main__` guards and import with no module-level side effects — consumed in place, not relocated.
- **Exactly one forced refactor-lift** — the 5 spectra scripts execute `os.getcwd()` at module top level (**verified:** `irspec_plot.py:60`, `plotIRRAM.py:494`, `ramspec_plot.py:121`, `irspec_average.py:107`, `ramspec_average.py:173`) and call `input()` interactively, so they are *unimportable as-is*. The pure `read_*`/`plot_*` bodies are lifted from `plotIRRAM.py` (the verified superset) into a guarded `handlers/spectra_api.py`.
- **One small shared module** — `grid_geometry.py` (the full-affine `grid_to_cartesian`) — extracted only because consolidation *reduces* fix surface for the cube-nonortho fix (one fix, three consumers).
- **No new `periodic_data.py` fork.** `mace/utils/formula_extractor.py` already implements the `Z>200` ECP decode; reuse it.

**Dependency reality (verified):** `requirements.txt` declares numpy, matplotlib, ase, spglib, PyPDF2, pyyaml, pandas (seekpath commented). It declares **neither plotly, scipy, nor kaleido** — those three are present only in `/home/marcus/anaconda3` (plotly 6.0.1, scipy 1.15.2, kaleido 0.2.1) and must be declared. *Note: the earlier critique claim that ase/spglib are also undeclared is **rejected** — both are present in `requirements.txt` lines 9 and 12.*

**Standing rule honored:** validated CRYSTAL parsers (`CubeFile._determine_data_type`, `Crystal23FreqParser` regexes, the spectra `read_*` tokenizers, matplotlib style blocks) are preserved. New behavior is layered on top; the six fixes below are surgical and validated in place *before* any relocation.

**Critical caveat — SUPERSEDED 2026-06-13 (see §0):** this originally read "the
spectra leg has zero test inputs." That is no longer true — `test/FREQ` now
contains **264** real spectra files (132 `IRSPEC.DAT` @ 3 cols + 132 `RAMSPEC.DAT`
@ 10 cols, all 0D molecular) and all five IR/Raman plotters are validated against
them. The remaining caveat is narrower: there is still **no crystalline/slab
SPEC** data, so the 2.4 reduced-column path stays unexercised by real inputs.

---

## 2. Generalization Fixes to Apply (prioritized by severity)

Each fix was adversarially verified. All fixes preserve validated parsing logic.

### 2.1 `freq-format` — fixed-6-column eigenvector regex; imaginary modes undetected — **VERDICT: SOUND** (apply, with the co-fix refinements)
**Severity: correctness (HIGH).**
**Root cause (verified at runtime on the FREQ corpus):** Three defects in `Crystal23FreqParser` (`vibmode_viewer.py`). (a) X/Y/Z eigenvector regexes require *exactly* 6 value groups; the final block of any mode count not divisible by 6 has 1–5 columns and is silently dropped — highest-frequency modes of most molecules become un-animatable. (b) Value subpattern rejects E-notation (latent). (c) The modes-table freq capture `([\d.]+)` has no leading minus, so imaginary rows never enter `self.modes`; `list_modes` further filters `freq > 1.0`.
**Fix:** Replace fixed-arity regexes with header-driven tokenize-and-validate. One shared `_NUM = r'[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?'`. Match X/Y/Z structural prefix only, then `re.findall(_NUM)` sliced to `n_cols = len(current_displacements)` (live header column count). Widen the modes-table freq and THz captures to `_NUM` and tag `mode['imaginary']`. Surface imaginary modes in `list_modes`.
**Refinements (mandatory co-fixes):**
- **Imaginary threshold:** tag `imaginary = freq < -1.0` (not `< 0.0`). Numerically-negative acoustic residuals (−0.01, −0.02 cm⁻¹) are the translational/rotational modes the `>1.0` cutoff suppresses. **See M2 for the threshold-validation requirement** — the −1.0 cutoff must be confirmed against the corpus negative-frequency distribution before it is hard-coded.
- **Amend the acceptance test:** the smoke test ("every listed mode has shape-(n_atoms,3) displacement") fails on the real corpus (pre-existing near-degenerate freq-collision + `n_atoms` from asymmetric unit ≠ eigenvector atom count on high-symmetry crystals). Scope the test to the molecular subset, OR co-fix per 2.2. Add a `coords.shape[0] == disp.shape[0]` guard in `VibModeAnimator` that refuses to render (rather than numpy-broadcasting garbage).
- **Encoding hardening:** `open(filename, encoding='utf-8', errors='replace')`; strict `*.out` glob so binary `.f9`/`.f25` siblings are never read as text.
**Validates against:** `test/FREQ/Suflolane_*temp.out`, `Ag1Cl2_*optimized.out`, `TiPbO3_mp-19845_*optimized.out` (tail-block recovery); `1LiFSI-1Sulfolane-conf3_*temp.out` (imaginary −12.69/−24.24); `1_dia_opt_rev1_freq_*supercel222.out` (multiple-of-6 regression, byte-identical). Whole-corpus loop over all 95 `test/FREQ/*.out` + the 1 mislocated `test/BAND/3,4^2T13-CA_rev1_freq_*optimized.out` (see C1).

### 2.2 `vib-degenerate` — degenerate (E/T) normal modes collapse to first frequency — **VERDICT: FLAWED** (apply only the refined form; merge with 2.1)
**Severity: correctness (HIGH).**
**Root cause (verified at runtime on TiPbO3 cubic PM3M, 15 modes):** The MODES table prints degenerate sets as ranges (`7- 9 ... 116.07 (F1u)`). `_parse_modes_table` stored only the range *start* and discarded the range end + irrep; `_parse_normal_modes` then matched each of N identical FREQ columns to `self.modes` by `abs(Δfreq)<0.1` with first-match-break, so all N columns resolved to the same index and N−1 eigenvectors were silently overwritten.
**Fix (refined):**
- Expand each table range into one entry per index `range(start, end+1)`, each carrying its own mode number + shared freq + irrep + `degeneracy`/`degenerate_with`.
- Replace freq-proximity matching with **positional cursor** consumption (column k → `self.modes[cursor+k]`). Table and displacement block are emitted in the same CRYSTAL order — 1-to-1 mapping immune to degeneracy. This also resolves the near-degenerate freq-collision drop.
- **Refinement 1 (verified required):** relax **both** the CM⁻¹ (group 4) **and** THz (group 5) captures to `_NUM`; imaginary rows have a negative THz (`-164.23 -4.92`) and otherwise misalign the cursor. (Unifies with 2.1's `_NUM`.)
- **Refinement 2 (verified required):** widen the irrep capture to `\(([\w'"+-]+)\s*\)` — Cs/C2v outputs print `(A' )`, `(B" )`; `Ag1Cl5` has rows that otherwise parse to **zero** modes.
- **Refinement 3 (verified required):** handle 1–6 column displacement blocks (same fix as 2.1's tokenizer; the "preserve X/Y/Z regexes exactly" directive is the direct cause of the trailing-block failure and is overridden here).
- Carry `irrep`/`degeneracy` through to `list_modes()` and viewer titles.
**Validates against:** `test/FREQ/TiPbO3_mp-19845_sg221_*optimized.out` (primary: 15 distinct modes, F1u/F2u preserved, −164.23 triplet present, degenerate eigenvectors pairwise distinct); `3_dia3_opt_rev1_freq_*optimized.out`, `Ag1Cl1_*optimized.out`, `Ag1Br1_*optimized.out` (rocksalt cubic); `TiPbO3_mp-20459_sg99_*optimized.out` (tetragonal, mixed degeneracy); `Ag1Cl5_*optimized.out` (primed irreps); molecular `1LiFSI-*conf*_freq_HSESOL3C_*temp.out` (regression — classify each diff correction-vs-regression; do **not** assert no-op).

> **Note:** 2.1 and 2.2 touch the same three regex sites in `vibmode_viewer.py` (`_parse_modes_table` L146, `_parse_normal_modes` L193, `list_modes`) and must be implemented as **one combined change**.

### 2.3 `cube-nonortho` — diagonal-only grid in subtract/diagnose; c-perpendicular assumption in slice — **VERDICT: FLAWED** (apply core fix + refinements)
**Severity: correctness (HIGH).**
**Root cause (verified against code + real header):** `subtract_cubes.py:create_coordinate_grids` builds 1D axes from only the diagonal of the voxel matrix; `trilinear_interpolate` treats them as separable rectilinear. **Verified real header** of `4LG_FSI_TopMiddle_2x2_ABAB_FSI_opt_charge+potential_DENS.CUBE`: voxel row 0 = `(0.140339, -0.081018, 0.0)`, cell angles `90.0 90.0 119.997974`. The off-diagonal `-0.081018` is silently dropped; query points land up to `99 × 0.081018 ≈ 8.02 Bohr` off at the grid edge. The canonical correct path already exists: `CubeFile.get_cartesian_grid` (`crystal_cubeviz_plotly.py:462`) uses the full matrix.
**Fix (core verified correct at runtime):** Extract a shared `grid_to_cartesian(origin, V, nvox)` (mirroring `get_cartesian_grid`) into `grid_geometry.py`. Interpolate in fractional index space: `frac = inv(Ms) @ (P − origin_S)` (`Ms` = voxel vectors as columns), then `scipy.ndimage.map_coordinates(order=1)`.
**Refinements (mandatory):**
- **nonortho threshold (H1, verified):** real values are noisy (off-diag `-0.081018`, angle `119.997974°`, NOT round numbers). Use a **relative tolerance**: `nonortho = abs(offdiag) > 1e-6 * max(abs(diag))` — NOT exact `≠ 0` (a `1e-15` rounding off-diagonal would needlessly route an orthogonal cell through the slower full-affine path). Keep the `>0.01°` angle deviation as a secondary signal. Validate the threshold is **True** on the slab and **False** on a pure-diagonal molecular POT cube.
- **Mask tolerance (verified bug):** `frac <= (ns-1)` is too strict — float round-trip overshoots by ~1e-14, flagging self-grid boundary points OUTSIDE and zero-filling. Use `inside = all((frac >= -eps) & (frac <= (ns-1)+eps))` with `eps=1e-6` and `np.clip` before `map_coordinates`.
- **Slice rendering (verified bug):** `go.Heatmap(x=2D, y=2D)` does **not** render a skewed parallelogram (Heatmap/Contour are rectilinear-only in plotly 6.0.1). Use `go.Surface(x=X2, y=Y2, z=zeros, surfacecolor=data)` (or `go.Carpet`/`go.Contourcarpet`) for true skew.
- **Atom-overlay frame (verified bug):** keep the slice surface in the **same** Cartesian frame as the atoms (`go.Surface`); if falling back to index-axis Heatmap, convert atoms to fractional via `inv(Ms)`.
- **Singular-matrix guard:** wrap `np.linalg.inv(Ms)` in try/except `LinAlgError`.
- Update `diagnose_cube_subtraction.py`: call `is_nonorthogonal(V)` and amend the "voxel sizes match" narrative to state interpolation is now full-matrix/triclinic-safe.
**Validates against:** `test/ECH3POT3/4LG_FSI_*_charge+potential_*.CUBE` (30 hex slabs — assert new point cloud matches `get_cartesian_grid` <1e-6 Bohr; slice renders skewed parallelogram); regression `test/ECH3POT3/1LiFSI-1Sulfolane-conf1_*POT.CUBE` (pure-diagonal — <1e-9 identical to pre-fix). Run with `/home/marcus/anaconda3/bin/python`.

### 2.4 `spectra-robustness` — `!=N`-col gate, cwd-only, interactive, brittle `-confN_` naming — **VERDICT: FLAWED** (apply with refinements; this is the forced lift)
**Severity: correctness + integration-blocking (HIGH).**
**Root cause (verified):** Four coupled defects across all 5 spectra scripts: (1) hard `data.shape[1] != 10` / `!= 3` gates reject reduced-column runs; (2) module-level `os.getcwd()` (all 5, line numbers in §1) + interactive `input()` → unimportable/headless-hang; (3) `np.mean` over ragged grids; (4) brittle `^(.+?)-conf\d+_` grouping. **Verified:** `read_irspec` and `read_ramspec` bodies are **byte-identical** between `plotIRRAM.py` and the standalones, so the read path is a confirmed superset.
**Fix:** Lift pure `read_irspec`/`read_ramspec` + `plot_irspec`/`plot_raman_*` from `plotIRRAM.py` into guarded `mace/plotting/handlers/spectra_api.py`. Min-column **downgrade** (not reject): Raman ≥2→total, ≥4→par_perp, ≥10→all. Thread `input_path`/`output_dir`, return written PNG paths. Replace `input()` with `kind`/`raman_mode`/`average` params. Grid-safe averaging via `np.interp`. Tolerant `_material_key`.
**Refinements (mandatory):**
- **H3 prerequisite — prove the superset before lifting:** `read_*` bodies are confirmed identical; before lifting, **diff the `plot_*`/`average_*` bodies** across all 5 scripts (`plotIRRAM.py` vs `ramspec_plot.py:plot_total/plot_total_par_perp/plot_all` and vs the two `*_average.py`) and prove `plotIRRAM` contains every branch, or **merge the union**. Do not assert "superset" of the plot/average paths without recording the diff.
- **IR absorbance column (verified bug):** absorbance = `data[:, 2]` when `ncols≥3`, else `data[:, 1]` — **never** "last column." 3D-periodic IRSPEC appends reflectance/refractive-index/dielectric columns after col 2; "last-of-N" would mislabel a dielectric column as Absorbance.
- **Grid-safe averaging (verified bug):** `np.interp` clamps to endpoints, fabricating flat tails; reference grid hard-wired to `all_w[0]` is order-dependent. Pick the **widest common-overlap range** (or densest grid); set `np.interp(left=nan, right=nan)` and drop NaN rows; assert source `w` is sorted-ascending.
- **Auto-pick wiring (verified gap):** add `SPECTRA_IR`/`SPECTRA_RAMAN` to the classifier (Section 4) so spectra participate in auto-detect.
- **Matching (verified fragility):** anchor case-insensitively to suffix `re.search(r'(?i)(IR|RAM)SPEC\.DAT$', f)`; strip via regex sub anchored at `$`, not global `str.replace`.
**Validates against:** **Grouping (runtime now):** real conf-style stems in `test/FREQ` and `test/ECH3POT3` (e.g. `1LiFSI-1DEC-conf{N}`, `1LiFSI-2EC-2DEC-1FEC-conf{N}`) — assert each family collapses to one key, non-conf names fall back to stem. **Column/averaging:** synthetic fixtures only — including a >3-column "3D-crystal" IRSPEC fixture (assert Absorbance trace == col index 2) and a narrowest-RANGE-first conf fixture (guard order-dependence). See Section 7.

### 2.5 `element-ecp` — sparse element tables + ECP heavy-element handling — **VERDICT: OVERREACH** (apply scoped form only)
**Severity: correctness (MEDIUM).**
**Root cause (verified at runtime):** The cube coloring claim is **wrong** — `get_atom_properties()` (`crystal_cubeviz_plotly.py:794`) is a complete Z=1..109 table; only `_COVALENT_RADII` (opt-in behind `--bonds`) is sparse. The cube `Z+200` ECP claim is **unsupported**: all 228 cube col0 values are `{1,3,6,7,8,9,16}` — zero ECP signatures. The **actual** vibmode bug, confirmed by running the parser on `TiPbO3_mp-20459_sg99_*optimized.out`: it yields `elements == ['TI','PB','O',...]` and `PB`/`TI` **miss all four element dicts** (`ELEMENT_COLORS`/`ELEMENT_OUTLINE_COLORS`/`COVALENT_RADII`/`DISPLAY_SIZES`), falling to DEFAULT. 31 heavy-element crystalline FREQ files are affected.
**Scoped fix (apply only this):**
- **ADD uppercase symbol keys** `PB, AG, SE, TI` (plus `AL, TE`) to vibmode's four element dicts with **per-element** covalent radii (Pb 1.46, Ag 1.45, Se 1.20, Ti 1.60, Br 1.20) — **not** a flat `>20 → 1.5 Å` fallback (which creates spurious metal-metal bonds).
- **H2 casing — REFINED/CORRECTED:** the critique's specific "aluminum is broken today" example is **rejected as stated**: there is **no `'Al'` key** in any of the four element dicts (only in `ATOMIC_NUM_TO_ELEMENT`, a separate Z→symbol table), and the **primary** coordinate path uppercases at `vibmode_viewer.py:120` (`element = match.group(3).upper()`), so `_calculate_bonds`' raw `COVALENT_RADII.get(elem_i,...)` lookups succeed in practice. The casing mismatch is real **only on the dead fallback path** (`ATOMIC_NUM_TO_ELEMENT` at L139, which returns mixed-case `He/Be/Al/Ne/Ar`). **Apply the cheap hardening anyway:** add `.upper()` to the two `_calculate_bonds` lookups (L306-309) and keep all dict keys uppercase, so the rare fallback path is correct too. Do not claim this fixes a live bug — it closes a latent one.
- **Reuse** `formula_extractor._symbol()` (already does `Z>200`) as a hardening shim for the numeric-fallback branch; **do not** fork `periodic_data.py`.
- **Do NOT** apply `Z+200` to cube col0 — no dataset evidence; if retained defensively, apply `normalize_z` only at color/symbol lookup, **never** mutate `cube.atomic_numbers` (breaks the subtract→write→reread round-trip).
**Validates against:** `test/FREQ/TiPbO3_mp-20459_sg99_*optimized.out` (assert atom element == `'PB'`/`'TI'` via the **text** path, `_get_color('PB')`/`_get_size('PB')` non-DEFAULT after the fix — confirmed DEFAULT pre-fix); `Ag2Cl1Se1_*optimized.out` (AG/SE/CL non-DEFAULT); regression `1_dia_*supercel222.out`, `1LiFSI-1DEC-conf1_*DENS.CUBE` unchanged.

### 2.6 `cube-orbital-dims` — negative-natoms multi-DSET cubes unsupported — **VERDICT: SOUND** (apply with round-trip + select_dataset refinements)
**Severity: robustness (LOW — latent, foreign-cube interop).**
**Root cause (verified):** Negative `natoms` (Gaussian-cube multi-DSET convention) means a trailing `m id1..idm` line + interleaved data; `read_cube` uses `abs(natoms)` for the atom loop but never skips the DSET line. **Honest reframe:** CRYSTAL23 emits *separate positive-natoms* DENS/POT/SPIN files — all 228 cubes have positive natoms. This is **foreign-cube interoperability hardening, not a CRYSTAL gap.**
**Fix:** After the atom loop, if `natoms < 0` read one extra line for `n_datasets` + `dset_ids` (keep natoms signed). If `m>1`, reshape to `(nx,ny,nz,m)`, set `data = [...,0]`, keep `data_all`. Add length-guard `len(data_list) == nx*ny*nz*m` with observed-vs-expected message. The `m==1` branch is byte-identical to today.
**Refinements:**
- **Round-trip (verified broken in original):** `write_cube_file` emits no DSET line and flattens only the active field. Either write the DSET line + flatten `data_all`, OR force `natoms` positive + `n_datasets=1` on write; never emit a negative header with 1/m of the data.
- **`select_dataset` must re-run** `_determine_data_type` + `_detect_and_crop_vacuum` + `_analyze_structure` + isovalues (not just isovalues).
- Document that the interleaved reshape assumes dataset-index is fastest-varying.
**Validates against:** Regression `test/ECH3POT3/1LiFSI-1DEC-conf1_*DENS.CUBE` (natoms positive, `data shape == nvoxels`, `data_all is None`) + hex slabs. New capability: a generated negative-natoms 2-DSET cube (assert `n_datasets==2`, `data_all` trailing axis 2, `select_dataset(1)` switches field); assert subtract→write→reread round-trips natoms unchanged. Mirror in `subtract_cubes.py:read_cube_file`.

**Priority order:** 2.1+2.2 (combined, HIGH, FREQ) → 2.3 (HIGH, cube) → 2.4 (HIGH, spectra lift) → 2.5 (MEDIUM, scoped) → 2.6 (LOW, defensive).

---

## 3. Chosen Integration Architecture + Target Module Layout

**Architecture:** Registry + content-sniff classifier spine. `main.py` shrinks to: `parse → (override pins PlotKind | detect.discover()) → loop registry[kind].handler(files, config, out_dir)`. Engines consumed via the established `sys.path-insert + import-by-bare-name` pattern (`main.py` plot_bands L263–315). All interactivity stays in `main.py`'s `configure_*` functions; handlers are 100% headless and never call `input()`.

```
mace/plotting/
  main.py            # SHRINKS: parse -> override-or-discover -> registry dispatch loop.
                     #   Existing create_parser (~L693), run_interactive (~L605),
                     #   configure_output_formats (~L168), prompt helpers (yes_no_prompt L22,
                     #   select_option L40, get_float_input L63, get_string_input L77) RETAINED.
  detect.py          # NEW. PlotKind enum + Detected dataclass; classify_file(path)->Detected;
                     #   discover(dir)->Dict[PlotKind, List[Detected]]. Reuses an EXTRACTED header-only
                     #   classifier from CubeFile (see C2) + the Crystal23FreqParser signature + eigenvector
                     #   gate (see C1); NO re-implementation of subtype rules.
  registry.py        # NEW. REGISTRY: Dict[PlotKind, PlotterEntry(handler, configure, label,
                     #   override_flag)]. Single registration point collapses the 4 sync sites.
  handlers/
    __init__.py      # imports each handler module so its register() runs (populates REGISTRY).
    legacy.py        # registers EXISTING plot_bands/plot_dos/plot_structures UNCHANGED.
    cube.py          # configure_cube_plot(interactive) + plot_cube(files, config, out_dir)->List[str].
                     #   sys.path-inserts test/AddedPlottingFunctionalty, imports crystal_cubeviz_plotly,
                     #   builds CubeFile, passes the detect-resolved subtype IN (so the engine does not
                     #   re-derive — see C2), calls plot_*_plotly, writes via explicit out=.
    freq.py          # configure_freq_plot + plot_freq(...). Imports vibmode_viewer
                     #   (Crystal23FreqParser / VibModeAnimator). Writes HTML/GIF via explicit out=.
    spectra.py       # configure_spectra_plot + plot_spectra(...) -> calls spectra_api.
    spectra_api.py   # THE ONE FORCED LIFT. Pure read_irspec/read_ramspec + plot_irspec/
                     #   plot_raman_* lifted from plotIRRAM.py behind a guard; data_folder/output_dir
                     #   threaded as params. Holds the 2.4 spectra-robustness fixes.
  grid_geometry.py   # NEW, small. grid_to_cartesian(origin, V, nvox) + inverse + is_nonorthogonal
                     #   (relative-tolerance per H1). Single source for the 2.3 affine (3 consumers).
```

**Deliberately NOT created:** no `periodic_data.py` fork (reuse `formula_extractor._symbol`); no package copies of the cube/vibmode engines (consumed via path-insert, edited in place for fixes only).

**Registry entry shape:**
```python
@dataclass
class PlotterEntry:
    handler: Callable[[List[Detected], dict, str], List[str]]  # returns written paths
    configure: Callable[[bool], dict]                          # interactive flag -> config
    label: str                                                 # menu + summary text
    override_flag: str                                         # e.g. '--cube'
```

---

## 4. Auto-Detection Logic (`detect.py`)

**`PlotKind` enum:** `BAND, DOS, STRUCTURE` (existing) + `CUBE_DENSITY, CUBE_POTENTIAL, CUBE_SPIN, CUBE_DIFF, CUBE_GENERIC, FREQ_MODES, SPECTRA_IR, SPECTRA_RAMAN, SPECTRA_FROM_FREQ, TRANSPORT_UNSUPPORTED, UNKNOWN`.

**`Detected` dataclass:** `(path, kind, subtype, dim, nonortho: bool, can_spectra: bool, confidence, source, note)`.

**Per-input sniffing (all bounded, header-only, `encoding='utf-8', errors='replace'`):**

1. **`*.CUBE` / `*.cube`** — read 6 header lines once.
   - **Subtype — single source of truth (C2, verified divergence):** the engine's `_determine_data_type` (L198-235) consults the line-1/line-2 **comment ONLY for the difference branch** (L201-216), then falls to **filename tokens** (`'spin'`, `'ech3'`/`'dens'`/`'density'`, `'pot'`/`'potential'`, L218-229), then to a data-sign fallback requiring the full grid (L232-235). A detect.py that subtyped ordinary cubes by line-1 comment would **contradict the engine** (two sources of truth). **Resolution: extract a header-only classifier from the engine** — refactor `_determine_data_type` to a free function `classify_cube_subtype(comment1, comment2, filename, data=None)` that detect.py and the engine both call with `data=None` (header-only). detect.py passes the resolved subtype **into** the engine (cube.py wires it through) so the engine does not re-derive. This keeps one rule, honors "don't fix what works" (the rule body is preserved, only its call site/signature change), and removes the divergence.
   - *Geometry from line 2 + voxel rows 4–6:* line 2 all-zeros → 0D molecule; cell lengths+angles → periodic. `nonortho` per the **H1 relative-tolerance rule** (`abs(offdiag) > 1e-6·max(abs(diag))`, secondary `>0.01°` angle test). **`nonortho` and `dim` are metadata** (camera/iso defaults), NOT separate PlotKinds.
   - **M3 (verified gap):** a cube with **blank comment AND no filename token** can only be subtyped by loading the grid. detect labels it `CUBE_GENERIC`; the summary says "subtype determined at load." (All 228 corpus filenames carry `DENS`/`POT`/`SPIN`, so this is defensive.)
   - **M1 (1D, verified absent in data):** compute `dim` = count of nonzero cell vectors. 1D cubes (one nonzero vector) are handled by the periodic fall-through; documented as "handled-by-fallthrough, **untested** (no 1D systems in corpus)."

2. **`*.out` / `*.OUT`** — strict glob (never open `.f9`/`.f25`/`.o`). Bounded scan (≤8000 lines, early-exit) for `CALCULATION OF PHONON FREQUENCIES AT THE GAMMA POINT`.
   - **C1 (verified — corpus-wide rescan done):** the signature hits **96 `.out` corpus-wide**, not 95: **95 in `test/FREQ/` + 1 in `test/BAND/`** (`3,4^2T13-CA_rev1_freq_B3LYP-D3-D3_optimized.out`, a genuine freq run mislocated in BAND/). FREQ `.out` are NOT confined to `test/FREQ/`. **Two mandatory gates:**
     - **(a) Eigenvector-block gate:** require the signature **AND** a real modes block. **Verified:** all 96 signature hits carry exactly one `NORMAL MODES NORMALIZED` marker — gate on its presence so aborted freq runs that printed the header but produced no modes classify UNKNOWN, not FREQ_MODES.
     - **(b) `--all` intent guard:** in `--all` auto-detect over a mixed directory, FREQ_MODES must emit an explicit per-file line: `"treating X as FREQ normal-mode viewer — override with --band/--dos if this is a band/DOS run"`. A freq run that is also a band-path run is surfaced, not silently double-rendered.
   - `can_spectra=True` if an IR/Raman intensity/CPKS marker is present on the same open file. No signature → `UNKNOWN`, skipped silently (avoids 540+ ordinary `.out` false hits).

3. **`*IRSPEC.DAT` / `*RAMSPEC.DAT`** — case-insensitive suffix-anchored match → `SPECTRA_IR` / `SPECTRA_RAMAN` (the dataset has **264** such files in `test/FREQ`; the absent-DAT/derive-from-FREQ pass below still applies to FREQ-only directories).

4. **`.dat` routing (C3, verified):** corpus has **176 `.dat`**: 76 `*BAND.DAT` + 80 `*DOSS.DAT` (handled by existing BAND/DOS globs) + **20 TRANSPORT** (`KAPPA/SEEBECK/SIGMA/SIGMAS/TDF`, 4 each). Explicit rules: BAND/DOS `.dat` → existing globs; `*SPEC.DAT` → spectra; **TRANSPORT `.dat` → `TRANSPORT_UNSUPPORTED`**, which prints a one-line `"transport .dat detected — not yet supported (N files)"` (so 20 files are not silently invisible) and is excluded from rendering. **Hard rule:** the spectra matcher is suffix-anchored `(?i)(IR|RAM)SPEC\.DAT$` and must **never** fall back to column-count sniffing on arbitrary `.dat` (a multi-column `SEEBECK.dat` would mis-plot as a spectrum). All other `.dat` → UNKNOWN, dropped.

5. **Existing BAND/DOS/CIF globs** kept verbatim for back-compat.

**`discover(directory)`** — non-recursive default (`--recursive` opt-in); strict extension allowlist `{.cube, .out, .cif, .dat}`. Bucket into `Dict[PlotKind, List[Detected]]`; drop UNKNOWN silently. **Spectra-derivation pass:** if both `SPECTRA_IR`/`SPECTRA_RAMAN` buckets are empty but `FREQ_MODES` entries have `can_spectra=True`, synthesize a `SPECTRA_FROM_FREQ` bucket with an actionable note ("no *SPEC.DAT found; N FREQ run(s) can generate IR/Raman — run CRYSTAL CPKS step or `mace plotting --spectra --from-freq`"). Group conf families via `^(.+?)-conf\d+_` for the *summary only*.

**Coexistence:** detection is **per-file keyed on (kind, subtype)** — the same stem yields STRUCTURE, CUBE_DENSITY, CUBE_POTENTIAL, CUBE_SPIN, FREQ_MODES in separate buckets; no collision.

**Precedence rules:**
1. Explicit override flag wins absolutely — pins the PlotKind family; detection still runs to *find* files within that family. For cubes, a sub-flag (`--esp`/`--spin`) overrides even the resolved subtype, and a warning is emitted if it contradicts the comment/filename.
2. Single explicit file, no flag → classify directly; UNKNOWN → print "detected as UNKNOWN; force with `--cube`/`--freq`/`--spectra`".
3. Cube subtype confidence: comment-difference > filename token > data-sign (engine's own order, preserved via the extracted classifier).
4. Real `SPECTRA_IR`/`RAMAN` DAT always precedes `SPECTRA_FROM_FREQ`.
5. Binary/input siblings (`.f9/.f25/.o/.d3/.d12/.sh`) are **never** classified (strict allowlist).

---

## 5. CLI Flag/Subcommand Surface

**Design:** type-pin flags added to the existing mutually-exclusive `mode_group` (`main.py:712`) — **not** argparse subparsers. Consistent with `--band/--dos/--structure`, no top-level dispatch change (`command_args('plotting')` forwards post-`plotting` tokens verbatim; `mace_cli:1135`). Each flag *pins* the PlotKind the classifier would infer; absence → auto-detect.

**Type-pin flags (mode_group):** `--cube`, `--freq` (`--vibmodes` alias), `--ir`, `--raman`, `--spectra` (umbrella). Existing `--band/--dos/--structure/--all/-i` unchanged.

**Cube quantity sub-selectors (imply `--cube`):** `--density`/`--charge`, `--esp`/`--potential`, `--spin`, `--diff A B`.

**`cube visualization options` group:** `--iso V[,V2…]`, `--view {iso,slice,both}`, `--slice AXIS POS`, `--slice-all AXIS`, `--alpha F`, `--colorscale NAME`, `--log`/`--linear`, `--clip P`, `--no-atoms`, `--bonds`, `--publication`, `--engine-args "…"` (escape hatch forwarding all ~66 engine flags verbatim, shlex-split).

**`vibrational mode options` group:** `--mode N`, `--modes N-M|N,M`, `--all-modes` (NOT in mode_group — avoids clash with top-level `--all`), `--list-modes`, `--amplitude F`, `--gif` (+`--gif-fps`), `--static`, `--compare`, `--engine-args "…"`.

**`spectra options` group:** `--raman-mode {total,powder,single}`, `--raman-component {xx,xy,xz,yy,yz,zz}` (requires `single`), `--average`/`--no-average`, `--ir-column N` (default index 2 — the 2.4 fix), `--from-freq`.

**Shared:** `-d/--directory` (existing), positional `FILE(s)`, `-o/--output` (existing), `--format {html,png,svg,pdf}` (default html; png/svg/pdf via kaleido, fall back to html with warning if kaleido import fails). **Discipline:** handlers always pass an explicit `out=*.html`/`*.png` path so engines *write* rather than `fig.show()`.

**H4 — argparse nargs vs positional `FILE(s)` (verified ambiguity):** A greedy positional `FILE(s) nargs='*'` interleaved with `--diff A B` (nargs=2), `--slice AXIS POS` (nargs=2), `--slice-all AXIS`, and the existing `--supercell NX NY NZ` (nargs=3, `main.py:794-800`) is a classic argparse ambiguity (`mace plotting --diff a.CUBE b.CUBE c.CUBE` cannot reliably tell whether `c.CUBE` is a third positional or an error). **Resolution:** make the positional `FILE(s)` **`nargs='*'` only when no nargs-consuming option is present**; specifically — `--diff` consumes exactly its two operands and **forbids extra positionals** (error if any remain); `--slice`/`--slice-all`/`--supercell` likewise consume their fixed operands first. Add an **argparse-level test matrix** covering every nargs option × {0,1,2,3} positional files (Section 7).

**H5 — `--all-modes` / `--list-modes` precedence (verified under-specified):**
- `--list-modes` **short-circuits** — prints the mode table (including imaginary, with irreps) and **ignores all render/format flags** (`--format gif` with `--list-modes` is a no-op on format, not an error).
- Top-level `--all` (auto-detect-everything) must **NOT** imply `--all-modes`. Per-file FREQ default under `--all` = a single representative mode (lowest real mode), **not** every mode (which could emit hundreds of HTML files). `--all-modes` is an explicit opt-in sub-option.
- Both rows added to the usage table below.

**Help sync:** update the one hand-maintained string in `mace_cli show_command_help` (`mace_cli:898-940`) and the `create_parser` epilog (`main.py:699-708`).

### Concrete usage table
| Command | Effect |
|---|---|
| `mace plotting --cube -d ./ECH3POT3 -o ./plots` | Auto-detect every cube quantity per file; suffixed outputs |
| `mace plotting --cube foo_DENS.CUBE --iso 1e-4,1e-3 --colorscale Viridis` | Density isosurface, explicit isovalues |
| `mace plotting --esp foo_POT.CUBE --view slice --slice z 0.5` | ESP slice (diverging preset, skewed-safe for hex) |
| `mace plotting --spin foo_SPIN.CUBE` | Spin density, signed iso pairs |
| `mace plotting --diff after_DENS.CUBE before_DENS.CUBE -o ./plots` | Δρ difference (engine subtract path, non-ortho safe); extra positionals rejected |
| `mace plotting --freq -d ./FREQ --list-modes` | List mode table (incl. imaginary + irreps); ignores render/format flags |
| `mace plotting --freq mol_FREQ.out --mode 7 --format gif` | Single mode → GIF |
| `mace plotting --freq mol_FREQ.out --all-modes` | All modes → one HTML (explicit opt-in) |
| `mace plotting --freq a.out b.out --compare` | Side-by-side comparison |
| `mace plotting --ir -d ./spectra` | Auto-find `*IRSPEC.DAT`; "unvalidated" notice |
| `mace plotting --ir m-conf1_IRSPEC.DAT m-conf2_IRSPEC.DAT --average` | Conformer-averaged IR (widest common-overlap grid) |
| `mace plotting --raman foo_RAMSPEC.DAT --raman-mode single --raman-component zz` | Single-crystal Raman zz |
| `mace plotting --all -d ./run -o ./plots` | Auto-detect everything; FREQ emits per-file override notice; one representative mode each; spectra only if DAT present |
| `mace plotting --cube foo_DENS.CUBE --engine-args "--camera-preset top"` | Advanced escape hatch (66 flags); output-controlling flags rejected (L4) |

**Key edge cases:** `--vibmodes` on a non-FREQ `.out` (no `NORMAL MODES` block) fails loudly ("no normal modes found; did this run do FREQCALC?"); `--diff` requires exactly 2 cubes (non-conformal grids → forward `--align-grids`); `--ir`/`--raman` with no DAT prints "spectra plotting is unvalidated; no files found" rather than a traceback and is excluded from `--all`; **L4 — `--engine-args` is shlex-split and screened: output-controlling flags (`--save-png`, `--save-html`, `--show`) are rejected with a warning so mace's `out=` writer discipline always wins.**

---

## 6. Migration / Phasing Plan (ordered, with dependencies)

**Phase 0 — Scaffolding (no behavior change).**
Create `detect.py`, `registry.py`, `handlers/__init__.py`, `handlers/legacy.py`. Move existing `plot_bands`/`plot_dos`/`plot_structures` registration into `legacy.py` **unchanged**; wire `main.py` to dispatch BAND/DOS/STRUCTURE via the registry. **L3 (incorporated): make `run_interactive` registry-driven in Phase 0**, not deferred — so every later phase's registration auto-appears in the interactive menu and `print_discovered_files` (no silent menu omissions). **Gate:** all existing band/dos/structure tests pass identically. *No engine touched.* — *Dependency: none.*

**Phase 1 — FREQ fixes in place (2.1 + 2.2 + scoped 2.5).**
Edit `vibmode_viewer.py` parsers (combined regex change), add the four uppercase element keys, apply the H2 casing hardening. **Validate against all 96 FREQ-signature `.out` (95 in `test/FREQ` + 1 in `test/BAND`) BEFORE wiring** — fixes proven in place, honoring "don't fix what works." — *Dependency: Phase 0.*

**Phase 2 — Cube fixes in place (2.3 + defensive 2.6) + extract header classifier (C2).**
Extract `grid_geometry.py`; refactor `_determine_data_type` → `classify_cube_subtype(...)` free function (C2 single-source); route `crystal_cubeviz_plotly.py` (and `subtract_cubes.py`) through `grid_geometry`; apply H1 relative-tolerance nonortho, mask-tolerance, `go.Surface` slice, atom-frame, singular guard, DSET round-trip. **Validate against `test/ECH3POT3` (228 cubes) BEFORE wiring.** — *Dependency: Phase 0.*

**L1 (incorporated) — combined engine-edit regression gate:** Phases 1 and 2 both edit engines Phase 3 imports. They may be developed in parallel, but **must NOT be validated only in isolation** — run a **combined regression gate** (full FREQ + cube corpus) on the merged working copy *before* Phase 3 begins.

**Phase 3 — Wire cube + FREQ handlers + classifier.**
Implement `handlers/cube.py` (passes detect-resolved subtype into the engine), `handlers/freq.py` (sys.path-insert, `out=` writer discipline, render guard). Implement `detect.py` cube + `.out` sniffing (C1 eigenvector gate + `--all` intent guard; C2 shared classifier; C3 `.dat` routing) and registry entries. Wire CLI flags (`--cube`/`--freq` + sub-flags, H4 nargs resolution, H5 precedence, L4 engine-args screen) into `mode_group`. **Gate:** `mace plotting --cube`/`--freq` produce files for the corpus; auto-detect picks correctly; argparse test matrix passes. — *Dependency: Phases 1 & 2 validated + combined gate.*

**Phase 4 — Spectra lift + handler (2.4) — isolated, marked unvalidated.**
**H3 prerequisite first:** diff the `plot_*`/`average_*` bodies across all 5 spectra scripts and record the result (the `read_*` bodies are already proven identical). Then lift the union into `handlers/spectra_api.py` with all 2.4 refinements. Wire `detect.py` spectra suffix detection + `SPECTRA_FROM_FREQ` derivation pass + `--ir`/`--raman`/`--spectra` flags. Ship with an "unvalidated — no test inputs" notice; exclude from `--all` unless DAT present. **L2 (incorporated):** Phase 4's detect wiring (`SPECTRA_FROM_FREQ` reads `can_spectra` from FREQ entries) **depends on Phase 3's `.out` sniffer** — only the `spectra_api` lift + grouping/column logic is independent of Phases 1–3. **Gate:** grouping validated against real conf filenames now; column/averaging validated against synthetic fixtures (Section 7).

**Phase 5 — Dependency declaration + help/docs sync.**
Declare **plotly 6.0.1, scipy 1.15.2, kaleido 0.2.1** in `requirements.txt` (the three genuinely undeclared deps; ase/spglib already present). **M5 (scoped/corrected):** audit the *full* runtime import set of the plotting package and pin versions, but do **not** re-add ase/spglib as "missing" — they are already declared. **M4 (incorporated):** add a kaleido smoke test that actually writes one PNG via `fig.write_image` **wrapped in a timeout** (kaleido 0.2.1 is the fragile pre-v1 Chromium-subprocess release that imports yet can hang headless) — the import-failure fallback does not cover a hang. Sync `show_command_help` + epilog. Deprecate the redundant scripts (below).

**Phase 6 (follow-up, deferred) — Library migration of the two big engines.**
Relocate `crystal_cubeviz_plotly.py`/`vibmode_viewer.py` into the package as a **pure relocation with no parser changes riding along.** *Not in scope for initial integration.*

**Script migration disposition:**
- *Library-consumed, edited in place for fixes only:* `crystal_cubeviz_plotly.py`, `vibmode_viewer.py`.
- *Forced lift (verified superset for read paths; plot/average diff pending per H3):* `plotIRRAM.py` → `handlers/spectra_api.py`.
- *Deprecated / NOT wired:* `irspec_plot.py`, `ramspec_plot.py`, `irspec_average.py`, `ramspec_average.py` (subsumed); `subtract_cubes.py`, `diagnose_cube_subtraction.py`, `slice_browser_standalone.py` (duplicate cubeviz arithmetic/slice + carry the orthogonal-only limitation 2.3 kills). Keep as standalone diagnostic CLIs at most; mark deprecated; exclude from the mace import path.

---

## 7. Testing Strategy Against `/mnt/iscsi/UsefulScripts/Codebase/reorganization/test`

**Harness:** `/home/marcus/anaconda3/bin/python` (only env with plotly/scipy/kaleido/ase/spglib). Use real `test/` outputs, never synthetic fixtures, except where no real input exists (spectra).

**FREQ (2.1 + 2.2 + 2.5) — fully validatable now, 96 signature files:**
- **C1 corpus-wide:** run the signature scan over all **637 `.out`**; assert exactly **96 hits (95 `test/FREQ` + 1 `test/BAND/3,4^2T13-CA_rev1_freq_*`)**, each with one `NORMAL MODES NORMALIZED` block; assert the eigenvector gate classifies any header-only-no-modes `.out` as UNKNOWN.
- Per-file: `len(p.displacements) == len([m for m in p.modes])` (modulo intentional acoustic skips); assert `self._mode_cursor == len(self.modes)`.
- Tail-block recovery: `get_displacement(last_mode)` non-None for `Suflolane`, `Ag1Cl2`, `TiPbO3_mp-19845`.
- Degeneracy: TiPbO3 cubic → 15 distinct modes; F1u triplets pairwise `np.allclose == False`; −164.23 imaginary triplet present; F1u/F2u irreps surfaced.
- Primed irreps: `Ag1Cl5` parses >0 modes.
- **M2 (incorporated) — imaginary-threshold validation:** scan all 96 files' negative-frequency distribution; **confirm a clean gap around −1.0 cm⁻¹** (no real soft mode clusters in (−1.0, 0)). If the gap is not clean, replace the fixed −1.0 with a structural criterion (acoustic = the 3 lowest-|freq| translational modes by symmetry). Assert −0.01/−0.02 residuals NOT tagged imaginary; −12.69/−24.24 ARE.
- Element: `TiPbO3_mp-20459_sg99` elements == `['TI','PB','O',...]` (confirmed at runtime); `_get_color('PB')`/`_get_size('PB')` non-DEFAULT **after** fix (confirmed DEFAULT pre-fix); `Ag2Cl1Se1` AG/SE/CL non-DEFAULT. H2 hardening: assert `_calculate_bonds` and `_get_color` agree for a mixed-case fallback symbol.
- Regression: `1_dia_*supercel222.out` displacements byte-identical pre/post; molecular `1LiFSI-*conf*` diffs classified correction-vs-regression (not no-op).
- Render guard: `VibModeAnimator` refuses (not broadcasts) when `coords.shape[0] != disp.shape[0]`.

**Cube (2.3 + 2.6 + C2) — fully validatable now, 228 cubes:**
- **H1 nonortho threshold:** assert `is_nonorthogonal` is **True** for `4LG_FSI_*_charge+potential_*DENS.CUBE` (off-diag −0.081018, angle 119.997974°) and **False** for a pure-diagonal molecular POT cube.
- Non-ortho point cloud: new full-matrix grid matches `CubeFile.get_cartesian_grid` <1e-6 Bohr; `subtract_cubes` output differs from old diagonal path by ~`|v01|·index` Bohr.
- Mask boundary: no spurious zero-fill on self-grid boundary voxels (eps fix).
- Slice render: `--slice z/y/x` on a hex DENS cube → skewed parallelogram (`go.Surface`), atoms aligned.
- **C2 single-source:** assert `classify_cube_subtype(...)` (header-only) and `CubeFile`'s loaded `data_type` agree on all 228 cubes — no detect/engine divergence.
- **M3:** confirm none of the 228 cubes is comment-blank + token-blank (all carry `DENS`/`POT`/`SPIN`); add one synthetic token-blank cube → assert `CUBE_GENERIC`, subtype deferred to load.
- Regression: orthogonal molecular cube (`1LiFSI-1Sulfolane-conf1_*POT.CUBE`) → subtract/slice <1e-9 identical to pre-fix.
- DSET (2.6): regression `1LiFSI-1DEC-conf1_*DENS.CUBE` (`data.shape == nvoxels`, `data_all is None`); generated negative-natoms 2-DSET cube → `n_datasets==2`, `data_all` trailing axis 2, `select_dataset(1)` re-runs analysis + switches field; subtract→write→reread round-trips natoms unchanged.

**Spectra (2.4) — partial; grouping now, plotting via synthetic fixtures:**
- **H3:** record the cross-script `plot_*`/`average_*` body diff as a test artifact before the lift.
- *Grouping (runtime now):* feed real conf-style stems from `test/FREQ` + `test/ECH3POT3` to `_material_key`/`group_files_by_material`; assert each `-confN_` family collapses to one key (`1LiFSI-1DEC`, `1LiFSI-2EC-2DEC-1FEC`); non-conf names fall back to stem.
- *Headless:* `import handlers.spectra_api; plot_spectrum(dir, kind='raman', raman_mode='all')` returns a non-empty list and never calls `input()`.

**C3 — `.dat` routing:** assert the 20 `test/TRANSPORT/*.dat` classify `TRANSPORT_UNSUPPORTED` (visible note, not rendered, not column-sniffed); 76 BAND + 80 DOSS `.dat` route to existing globs; no `.dat` reaches the spectra matcher except `*SPEC.DAT`.

**H4 — argparse matrix:** every nargs option (`--diff`, `--slice`, `--slice-all`, `--supercell`) × {0,1,2,3} trailing positional files; assert `--diff` with a 3rd positional errors cleanly (not silent absorb).

**M4 — kaleido:** write one PNG via `fig.write_image` under a timeout; if it hangs/fails, fall back to HTML with warning. Confirms `--format png/svg/pdf` before claiming support.

**`*SPEC.DAT` inputs — RESOLVED 2026-06-13 (this subsection is superseded):**
`test/FREQ` now has **264** real spectra files (132 IR @ 3 cols + 132 Raman @ 10
cols, all 0D molecular) and all five plotters are validated against them
(`test/_spectra_coverage.py`). The original "generate the missing inputs" plan is
moot for molecular spectra. Still genuinely missing: **crystalline/slab SPEC**
(needed to exercise the 2.4 reduced-column path) — generate via the CPKS/intensity
step (`INTENS`/`INTRAMAN`/`INTCPHF` + `IRSPEC`/`RAMSPEC`) on a periodic FREQ run,
or with a synthetic reduced-column `RAMSPEC.DAT` fixture.
2. **Synthetic fixtures (interim), derived from the real CRYSTAL legend** in `test/FREQ/1LiFSI-1EC-conf5_*temp.out` (Raman col1=freq, col2-4=powder Total/Par/Perp, col5-10=single-crystal xx..zz; IR col1=freq, col2=wavelen, col3=absorb):
   - One 10-col Raman, one 4-col, one 2-col → assert 4-col/2-col now produce a PNG (downgrade), not skipped.
   - Two same-material `-conf1_`/`-conf2_` files with **different-length** wavenumber grids → assert grid-safe averaging interpolates onto common-overlap range; a third with narrowest-RANGE first → guard order-dependence.
   - A **>3-column IRSPEC** (freq, wavelen, absorb, reflectance, n, dielectric) → **assert plotted Absorbance == column index 2, not last-of-N** (guards the 2.4 regression).
   - Place fixtures under `test/FREQ/` so grouping + plotting tests share one directory.

**Whole-corpus smoke test:** loop `Crystal23FreqParser` over all 96 FREQ-signature `.out` and `CubeFile` over all 228 `test/ECH3POT3/*.CUBE`; assert no constructor raises and (scoped to the molecular subset where `n_atoms` is reliable) every listed mode has a shape-`(n_atoms,3)` displacement. The high-symmetry-crystal atom-count files are a documented known-issue, not a red gate.

---

### Critique points explicitly REJECTED or CORRECTED (with one-line reason)
- **H2 "aluminum is broken today":** rejected as stated — there is no `'Al'` key in any of the four element dicts, and the primary coordinate path uppercases at `vibmode_viewer.py:120`, so the live lookup succeeds; the casing fix is applied as latent-path hardening (the dead `ATOMIC_NUM_TO_ELEMENT` fallback), not a live-bug fix.
- **M5 "ASE/spglib/seekpath also undeclared":** rejected — `requirements.txt` declares ase (L9) and spglib (L12); seekpath is intentionally commented. Only plotly/scipy/kaleido are genuinely undeclared.