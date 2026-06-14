# Pre-Push Review — `origin/main..main` (90 commits)

**Date:** 2026-06-13
**Method:** 17-agent multi-subsystem review (Workflow tool), each finding adversarially
re-verified by an independent skeptic against the real code, the pinned pytest suite, and
real `test/*.out` data. `mace/plotting/*`, `tests/test_plotting_*`, `tests/test_vibmode_parser.py`
were excluded (separate effort).
**Agents:** 59 · **Raw findings:** 41 · **Confirmed:** 35 · **Refuted/overstated:** 6

## RESOLUTION (2026-06-13, all tiers fixed)

After this report, a **cloud ultrareview** was also run and reconciled. Net new from
the cloud: 2 confirmed correctness fixes (timeout-handler regex; scan material_id
divergence) + 1 flagged plotting issue (below). **All actionable findings are now
fixed and committed** (each behind a regression test where applicable); suite **457
passing** (was 424). Commit map:
- Blockers: `c05a3043` (B1/B2), `f840e946` (SPINLOCK).
- Cloud correctness: `d9cb77b5` (timeout `-t`/days), `a86c2d01` (canonical material_id).
- Low/UX waves: `b7d694e7` (DOSS parser), `e3e235a3` (db), `4217a8e3` (d3),
  `aa9cb596` (CLI/UX), `dbeddb31` (d12 back-nav), `c248bb00` (disk recovery),
  doc/CI hygiene commits.
- ⚠️ **Cloud bug_001 (plotting) — NOT actioned (your separate instance).** The
  committed `HEAD:mace/plotting/main.py` has no broken imports, so the current push
  is safe; but your uncommitted refactor imports `registry/detect/prompts/handlers`
  which exist only as untracked files. Commit `main.py` **together** with those four
  submodules or `mace plotting` breaks on a clean checkout.

---

## Verdict (original): ⛔ DO NOT PUSH YET

Two **HIGH** correctness bugs in the **error-recovery resubmission path** (both *newly
activated* by this diff — they were latent/dead on `origin/main`). One **MEDIUM** (SPINLOCK
cycle count dropped on regenerate). The rest are low/nit polish, several of them *pre-existing*
(not introduced by this branch). No security issues. The recovery path is exactly the area the
standing caveat warned about (no failed CRYSTAL outputs in `test/`, so it was mock-only verified
— and the mocks hid these bugs).

| Severity | Count |
|---|---|
| critical | 0 |
| high | 2 |
| medium | 1 |
| low | 17 |
| nit | 15 |

Category mix: ui-ux 10 · maintainability 8 · latent-bug 7 · optimization 4 · regression-vs-main 3 · correctness-bug 3.

---

## BLOCKERS (fix before push) — both in the recovery resubmission feature

These two are intertwined. The new `submit_script_override` plumbing (intended to honor a
recovery-bumped SLURM script) means a memory/timeout failure now *actually resubmits* a bumped
script — but (a) it can't be submitted, and (b) if it could, it would be catastrophically wrong.
**One regression test exercising the real path covers both.**

### B1 — `recovery#F1` · HIGH · correctness-bug
**`mace/recovery/recovery.py` — `memory_handler`, L377-423**
The memory regex matches both `--mem` and `--mem-per-cpu`, reads the bare number, ×1.5, then
**always** rewrites to **total** `--mem={n}GB`. The shipped template `submitcrystal23.sh:19`
uses `--mem-per-cpu=5G` with `ntasks=32` → real allocation **160 GB**. After "recovery" it
becomes `--mem=7GB` total — a **~23× reduction**, guaranteeing an immediate worse OOM. Reproduced
live. The pinning test only used the `--mem=80GB` total form, never the per-cpu form the real
template emits.
**Fix:** capture the directive name in the regex; preserve `--mem-per-cpu` form (scale the
per-cpu value); only emit total `--mem` when the source was total. Add a regression test using the
actual `submitcrystal23.sh` per-cpu form.

### B2 — `submission-queue#F1` · HIGH · regression-vs-main
**`mace/queue/manager.py` — `submit_to_slurm` override + else branch, L697-700 / L755-772**
The bumped script is a *generated SLURM batch file* (literal `#SBATCH` lines), so it misses the
generator-marker check (L725) and falls into the else branch that tries to **directly execute**
the `.sh`. The recovery handlers write it `0o644` (never `chmod`), so execution raises
`PermissionError` — which is **swallowed** by `resubmit_fixed_calculation`'s broad except → recovery
silently reported failed, job never queued. Even if executable, it would run `Pcrystal` on the login
node (no `sbatch`) and never print `Submitted batch job N`. This else-branch was effectively dead on
`origin/main`; the override param makes it live. `tests/test_recovery.py` mocks `submit_to_slurm`, so
the real path is never exercised.
**Fix:** when `submit_script_override` is set, detect a ready-made batch file and submit via
`['sbatch', str(script_path)]`, then parse the job id (mirror the already-fixed `mace submit` path).
Add a non-mocked regression test driving `submit_to_slurm` against a fake `sbatch` on PATH.

> Net today: **B2 masks B1** — the bumped script never reaches SLURM because it can't be submitted.
> Fixing B2 alone would expose B1 (the 23× cut). Fix both together.

---

## MEDIUM

### `d12-interactive#F1` / `d12-generation#F1` (same bug, two call sites) · correctness-bug
**`Crystal_d12/CRYSTALOptToD12.py:653` and `Crystal_d12/NewCifToD12.py:~1139`; `d12_config.py:462`**
The new parser captures `spinlock_cycles`, but the writer call sites pass only `spinlock=` and omit
`spinlock_cycles=`, so `write_scf_section` falls back to `DEFAULT_SPINLOCK_CYCLES=50`. A source deck
with a non-default cycle count (e.g. `SPINLOCK / 2 30`) is silently regenerated as `2 50` on OPT
continuation or JSON-config reuse. Masked on every shipped reference deck (all use `0 50`).
Reproduced live. (Verifier note: the reviewer claimed a `test_spinlock_roundtrip.py` exists and
masks this — **it does not; there are zero spinlock tests** — but the underlying bug is real.)
**Fix:** pass `spinlock_cycles=settings.get("spinlock_cycles", DEFAULT_SPINLOCK_CYCLES)` at both call
sites; add `'spinlock_cycles'` to `d12_config.py` `direct_mappings`; add a real round-trip test.

---

## LOW (17)

**Parsers / extraction**
- `dat-formula#F1` (regression) — DOSS.DAT with **no NEPTS/NPROJ header** infers width 0 → parses to
  zero points (old line-by-line parser recovered these). All 80 real `test/` DOSS files carry the
  header, so edge-case. Add a width fallback like the BAND parser.
- `dat-formula#F2` (correctness) — `dos_at_fermi` takes `max(|DOS|)` over ±0.0037 Ha, so for an
  insulator it reports the valence-band tail (diamond → 0.195) instead of ~0 at E=0. Report DOS at the
  point closest to E=0. Metallic/insulator *classification* is unaffected (uses separate logic).
- `dat-formula#F3` (maint) — `process_band_dat_file`/`process_doss_dat_file` don't coerce `str`→`Path`
  (re-introduces the trap the formula-extractor fix removed). Add `file_path = Path(file_path)`.

**Database / analysis**
- `db-core#F1` (latent) — populate dedup `same_workdir` fallback compares full calc_type, so numbered
  engine steps (`SP2`/`OPT2`/`BAND3`) miss the base scan type (`SP`/...). Still a strict improvement
  over main. Normalize with `rstrip('0123456789')`.
- `db-analysis-export#F1` (latent) — `suggested_calculations` prefix-matching (`dep_name.split('_')[0]`)
  over-suggests, incl. **already-completed** calc types and an irrelevant TRANSPORT for a metal.
  Advisory output only. Match full key + exclude completed.

**Recovery / submission**
- `recovery#F2` (latent, low-confidence) — disk-full is classified `io_error` but the recoverable gate
  excludes it, so `cleanup_handler` is unreachable on the live manager path. Pre-existing; no real
  failed `.out` to confirm.
- `submission-queue#F2` (regression) — untracked `mace submit` raises unhandled `FileNotFoundError`
  when `sbatch` is absent (old `os.system` warned instead). Wrap in `try/except FileNotFoundError`.

**Crystal_d3**
- `crystal-d3#F1` (ui-ux) — config file loaded twice (banner printed twice) when calc_type is peeked
  from config. Reuse the peeked dict.
- `crystal-d3#F2` (latent) — duplicate dict key `'orthorhombic_bc'` in `normalize_lattice_type` maps
  I-centered → 'C'. Pre-existing; not hit on normal paths, but the new code routes I-centered ortho to
  that name. Remove the duplicate line.

**UI/UX**
- `x-ui-ux#F1` — `get_user_choice` reads first keystroke with plain `input()`, so `b` (back) only works
  on the *second* press. Use `_nav_read` on the first read too.
- `x-ui-ux#F3` — shipped `mace_examples.sh` / `setup_mace.py` help is half-migrated: still advertises
  bare legacy scripts (`enhanced_queue_manager.py`, `material_monitor.py`) the CLI itself deprecates.
- `x-ui-ux#F4` — `--track` submissions lack the single copy-pasteable job-id summary the default path
  now prints. Accumulate ids and print one summary line.

**Deletion hygiene / tests / perf**
- `x-deletion-safety#F1` — orphaned `code/Post_Processing_Scripts/README.md` still documents the
  deleted `grab_properties.py` as runnable. Delete/rewrite it.
- `x-optimization#F1` — DOSS/BAND `.DAT` parsed twice per material (per-type + advanced analyzer).
  Pre-existing; share the parsed arrays.
- `x-optimization#F2` — `_extract_scf_settings` runs 10 DOTALL `.*?` regexes over the whole `.out`
  (~657 ms on a 5 MB file). Pre-existing; add cheap substring guards before each.
- `x-test-quality#F2` — CI pins Python 3.11 but the suite only ever runs on local 3.12. Add a matrix.

## NIT (15) — brief
`db-core#F2` extracted_at stored as raw datetime (deprecation warning) · `db-core#F3` misleading
`output_file=NULL` comment · `db-utils#F1` `validate_all_materials` silently ignores unknown ids ·
`db-analysis-export#F2` LaTeX `key_columns` references non-existent bare `total_energy` ·
`extraction#F2` `ipDOS_V2` IndexError on truncated final DOS record · `cli-entry#F1` `submit --help`
still says target required · `cli-entry#F2` `analyze` branch inlines logic instead of `command_args()`
helper · `x-deletion-safety#F2` `AUTHORSHIP.md` still lists deleted modules · `x-optimization#F3`
`analyze_missing_data` double `get_material()` + N+1 queries · `x-test-quality#F1` `test_aggregation_keys`
uses fixed `/tmp` DB path not `tmp_path` · `x-test-quality#F3` CI install unpinned/uncached ·
`x-ui-ux#F5` back-nav prompt cosmetics (`[b=back] (previously: X)` ordering/double-space) ·
`x-ui-ux#F6` d3 single-file calc-type silently defaults invalid input to BAND · `recovery#F3` &
`workflow-engine#F2` — *positive confirmations* (clean `opt_sp_freq` template; safer `--action status`
default), no action.

---

## Refuted / overstated (filtered out by adversarial verification)

The skeptic layer rejected 6 — useful signal that the review didn't over-claim (the recurring failure
mode this campaign):
- `db-utils#F2` — pressure-table "float-representation drift": values are correct (`math.isclose` true), no real fix proposed.
- `extraction#F1` — "corrected-total fallback adopts stale initial-geometry line": primary path returns the correct final-geometry value on real data; fallback only triggers on a state that doesn't occur.
- `workflow-engine#F1` — "D3 workflow_id fallback mints a non-matching id": only in a doubly-degenerate state that the normal flow never reaches.
- `d12-interactive#F2` & `x-ui-ux#F2` — "back-nav can't cross helper boundaries / top-level menus": mechanics described correctly but the cited failing path is by-design and doesn't actually break.
- `crystal-d3#F3` — "GAMMA→G label conversion unverified": all 144 real BAND `.d3` fixtures use coordinate/SeeK-path segments, not label format; the path isn't exercised.

---

## Subsystem health (confirmed-finding count)

`x-security` 0 · `db-utils` 1 · `extraction` 1 · `workflow-engine` 1 (nit) · `db-analysis-export` 2 ·
`crystal-d3` 2 · `d12-generation`/`d12-interactive` 1 (shared) · `cli-entry` 2 (nits) · `db-core` 3 ·
`dat-formula` 3 · `recovery` 3 · `submission-queue` 2 · `x-ui-ux` 5 · `x-optimization` 3 · `x-test-quality` 3.

The headline fixes from the campaign verified **correct against real data**: full-precision
HARTREE_TO_EV consolidation (no mixed-precision remnants, eV==au·H to 1e-12), gCP/corrected-total/
molecular-FREQ/enthalpy extraction (matches CRYSTAL's printed lines), BAND/DOSS gap-by-index parsing,
pressure-table fix (old kbar/Mbar were swapped, atm off ~1000×), `cleanup_old_records` timedelta fix
(old code raised `ValueError` on most calls), the contextual-DB trio, settings-extractor import, and
the `node_exclusion` shell→argv hardening. SQL is parameterized throughout; no injection found.
