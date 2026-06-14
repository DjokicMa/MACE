# MACE CLI Help-Text Audit

**Date:** 2026-06-14  ·  **Method:** 18-agent validation (one per subcommand + top-level mace --help), each comparing the *rendered* `mace <cmd> --help` against the real dispatch + underlying module, every finding citing file:line.

**Findings:** 48 total — 13 high · 8 medium · 18 low · 9 nit

## RESOLUTION (2026-06-14) — all doc-only fixes applied

Every non-plotting finding is fixed in `mace_cli` (help text only, no behavior changes):
- Deleted the **fabricated `--action … --help`** content for `workflow` (it has no `--action` → now redirects to the real flag-driven interface), and stripped non-existent flags from `engine`/`recover`/`monitor` action-help; added the real `monitor --action report`.
- Rewrote the broken **passthrough** curated blocks (`convert`, `opt2d12`, `opt2d3`) to the real interfaces, and fixed `opt2cif`'s trailing note. (Passthrough `--help` still shows each tool's own accurate argparse output.)
- Added missing **`status` / `version` / `credits`** help blocks (+ routed `version`/`credits`); fixed `manager --dry-run` (labeled not-implemented) and added `--organize`; fixed `recover --max-recoveries` default (3→10) and the recoverable-errors list; added `analyze --calc-id`, `recover --db`, `monitor` pass-through flags; added `opt_sp_freq` to the workflow template list; fixed the top-level epilog (now lists `credits`/`version` + an Aliases section for `status`/`queue`).
- **Plotting findings (3) left as-is** — advisory; that area is under active development elsewhere.

Verified: `mace_cli` compiles; every `<cmd> --help` renders; the previously fabricated invocations no longer advertise non-existent flags; full suite **529 passing**.

## Overall

Help-text accuracy across the MACE CLI is uneven. Pass-through-only utilities (submit, queue, database) and the auto-generated argparse help they expose are accurate, but the hand-maintained curated help blocks in mace_cli have drifted badly from the underlying tools' real interfaces. Three commands are outright inaccurate (workflow, convert, engine): they advertise positional arguments, flags, and entire --action sub-modes that do not exist and that error out (exit 2) when a user copies them verbatim. The recover and monitor commands document several non-existent flags/actions alongside real, undocumented ones. A recurring structural problem is that every passthrough command (convert, opt2d12, opt2d3, opt2cif, completion) intercepts --help BEFORE show_command_help, so the curated help_text blocks for those commands are dead code that no user ever sees yet still drift; conversely status and version reach show_command_help but have no help_text key, so their --help prints "No help available" or the generic top-level help. Several deprecated/alias commands (status, queue, credits, version) are inconsistently represented between the usage brace list, the epilog Commands block, and the argparse choices. Net: bare --help renders for nearly everything, but the per-command guidance is trustworthy mainly for the passthrough tools' own auto-generated argparse output and for submit/database; the curated MACE-level help should not be trusted without verification for workflow, convert, engine, recover, and monitor.

## Per-command accuracy

| Command | Verdict | Findings |
|---|---|---|
| `mace --help (and bare mace)` | minor-issues | 3 |
| `workflow` | inaccurate | 4 |
| `submit` | accurate | 0 |
| `monitor` | minor-issues | 6 |
| `analyze` | minor-issues | 1 |
| `convert` | inaccurate | 4 |
| `opt2d12` | minor-issues | 1 |
| `opt2d3` | minor-issues | 2 |
| `opt2cif` | minor-issues | 2 |
| `status` | missing-help | 2 |
| `queue` | accurate | 1 |
| `manager` | minor-issues | 4 |
| `recover` | minor-issues | 5 |
| `database` | accurate | 3 |
| `engine` | inaccurate | 4 |
| `plotting` | minor-issues | 3 |
| `credits` | minor-issues | 2 |
| `version` | missing-help | 1 |

## Top priorities

1. **workflow** — The footer says 'use mace workflow --action <ACTION> --help' and an entire fabricated --action plan/execute/status/templates help block describes options (--config, --template, --plan, --resume, --workflow-id, etc.) that do not exist; `mace workflow --action plan` errors with exit code 2. The real command is flag-driven.
   - *Fix:* Delete show_workflow_action_help (mace_cli:453-531) and the workflow branch of show_action_specific_help, remove the --action footer (mace_cli:617), and rewrite the help to document the real flags: --interactive, --execute PLAN_FILE, --quick-start, --status, --show-templates plus --cif-dir/--d12-dir/--workflow/--work-dir/--db-path/--max-jobs.
2. **convert** — Every example is broken: it shows a positional CIF arg and --functional/--basis flags, none of which exist (real interface is --cif_dir/--output_dir/--batch/--save_options/--options_file). It also wrongly claims running with no args shows help, when it actually launches the interactive converter.
   - *Fix:* Rewrite usage to 'mace convert [--cif_dir DIR] [--output_dir DIR] [options]' and replace examples with --cif_dir-based ones; state that functional/basis are chosen interactively and that no-args starts the interactive converter. Note the curated block is currently never rendered (passthrough), so also wire it in or keep it synced with NewCifToD12.py.
3. **engine** — Action-help fabricates --dry-run and --max-submit (process), --list/--show/--create (workflow), and --pending-only (status); engine.py only defines --action/--material-id/--db/--work-dir, so all of these error with 'unrecognized arguments'. The --dry-run claim even contradicts the top-level help's own 'not implemented' note.
   - *Fix:* Strip the fabricated flags from action_help (process mace_cli:427-428, workflow mace_cli:433-444, status mace_cli:412). Rewrite the workflow action description (it prints a per-material completion summary, it does not show/modify templates) and reconcile the documented OPT->SP->BAND/DOSS/TRANSPORT/CHARGE+POTENTIAL progression with the code's 4-step set.
4. **recover** — Action-help advertises --error-type, --material-id, and config --show/--edit/--validate, none of which exist (exit 2). The recoverable-errors list is also wrong: it lists 'Basis set issues' (which are manual-only) and omits timeout and disk-space (which ARE auto-recovered). --max-recoveries default is misstated as 3 vs real 10.
   - *Fix:* Remove the four non-existent flags from the recover action-help; correct the recoverable-errors list to SHRINK/memory/SCF convergence/timeout/disk-space and re-label basis as manual; change action-help --max-recoveries default to 10; document the real --db flag.
5. **monitor** — Three documented entry points are broken commands: --dashboard (dashboard is the no-flag default; passing the flag errors), --max-materials (never implemented; errors), and --action health (not a valid choice; errors). Additionally --material-id is silently ignored in --status mode.
   - *Fix:* Remove --dashboard, --max-materials, and the health action from the help (or implement them). Document the real --action report and the useful pass-through flags (--output, --workflow-id). Either implement --material-id filtering in the status branch or note that it is currently not honored.
6. **status** — `mace status --help` prints 'No help available for command: status' because it is in the routing list (mace_cli:1086) and argparse choices but has no help_text key. The command works as a backward-compat alias to 'mace monitor --status', so the alias is effectively undocumented in help.
   - *Fix:* Add a 'status' key to the help_text dict describing it as a backward-compat alias for 'mace monitor --status' that forwards all args; this matches the runtime note already printed at dispatch (mace_cli:1625).
7. **manager** — --dry-run is documented as 'show what would be submitted without actually submitting' but is NOT implemented: the CLI prints 'Note: --dry-run is not supported... ignoring' and then performs a REAL submission. A user expecting a preview gets actual job submission.
   - *Fix:* Remove the --dry-run line or relabel it as not-implemented/ignored (pointing to 'mace manager --status' for a non-submitting view); long-term, implement real dry-run in queue/manager.py. Also document the real --organize flag and the valid --callback-mode values.

## All findings (by command, severity-sorted)

### `analyze`

- **[low] option-in-code-not-in-help**
  - help says: Options block lists --material-id, --calc-type, --filter-material, --filter-type, --output-file, --db-path (no --calc-id)
  - code: The underlying extractor defines `--calc-id` ("Override calculation ID"), and the analyze dispatch forwards all args verbatim to it, so `mace analyze --extract-properties . --calc-id <X>` is a working-but-undocumented flag.
  - evidence: Rendered help Options block (mace_cli:697-704) omits --calc-id; code defines it at mace/utils/property_extractor.py:2229 `parser.add_argument("--calc-id", help="Override calculation ID")`; dispatch forwards args verbatim at mace_cli:1597-1600.
  - fix: Add a line to the analyze Options block: `  --calc-id ID               Override calculation ID` (mirrors --material-id).

### `convert`

- **[high] bad-example**
  - help says: Examples:   mace convert structure.cif   mace convert *.cif   mace convert structures/ --functional B3LYP --basis POB-TZVP
  - code: NewCifToD12.py's argparse defines NO positional argument and NO --functional/--basis flags; only --cif_dir/--output_dir/--batch/--save_options/--options_file exist. Passing a positional CIF path or --functional/--basis causes argparse to exit with 'unrecognized arguments'.
  - evidence: Rendered help_text block (mace_cli:723-726). Verified: `mace convert structure.cif` -> 'NewCifToD12.py: error: unrecognized arguments: structure.cif'; `mace convert structures/ --functional B3LYP --basis POB-TZVP` -> 'error: unrecognized arguments: structures/ --functional B3LYP --basis POB-TZVP'. Code: NewCifToD12.py:
  - fix: Replace examples with the real interface, e.g.:   mace convert --cif_dir structures/   mace convert --cif_dir structures/ --output_dir d12s/   mace convert --cif_dir . --save_options --options_file myopts.json (and note functional/basis are chosen interactively, not via flags).
- **[high] stale-description**
  - help says: Usage: mace convert <cif_files> [options]
  - code: There is no <cif_files> positional. Input is selected via --cif_dir DIR (default ./). NewCifToD12.py:1352 calls parser.parse_args() with only the five optional flags; process_cifs(args.cif_dir, ...) at NewCifToD12.py:1388 reads CIFs from the directory.
  - evidence: Rendered help_text block (mace_cli:718): 'Usage: mace convert <cif_files> [options]'. Code: NewCifToD12.py:1329-1330 (--cif_dir default './'), 1388 (process_cifs(args.cif_dir, ...)). Real usage line per argparse: 'usage: NewCifToD12.py [-h] [--cif_dir CIF_DIR] [--output_dir OUTPUT_DIR] [--batch] [--save_options] [--opt
  - fix: Change usage to: 'Usage: mace convert [--cif_dir DIR] [--output_dir DIR] [options]' and document that input is a directory (default current dir), not individual file arguments.
- **[medium] stale-description**
  - help says: For full options, the command will show NewCifToD12.py help when run without arguments.
  - code: Running `mace convert` with no arguments does NOT show help — it launches the interactive converter (prints 'CIF to D12 Converter for CRYSTAL23' and prompts for calculation options). Help is only shown via --help/-h.
  - evidence: Rendered help_text block (mace_cli:728). Verified: `mace convert` (no args) -> 'CIF to D12 Converter for CRYSTAL23 ... Enhanced by Marcus Djokic with AI assistance' then interactive prompts. Code: NewCifToD12.py:1365-1372 (else branch: get_calculation_options_new() interactive flow when not --batch).
  - fix: Change to: 'For full options, run mace convert --help. Running with no arguments starts the interactive converter (reads CIFs from --cif_dir, default ./).'
- **[low] missing-help-block**
  - help says: (help_text['convert'] block at mace_cli:717-729 is the documented help, but it is never shown by `mace convert --help`)
  - code: convert is in passthrough_commands (mace_cli:1029) and intercepted at mace_cli:1040 before the show_command_help path; --help is forwarded to NewCifToD12.main() (mace_cli:1058-1062), so NewCifToD12.py argparse help is shown instead of the help_text['convert'] block. The block is reachable only via show_command_help('co
  - evidence: mace_cli:1040 'if ... sys.argv[1] in passthrough_commands'; mace_cli:1058-1062 dispatch to cif_main(); mace_cli:1086 command list for show_command_help does NOT include 'convert'. Rendered `mace convert --help` output is the NewCifToD12.py argparse usage, not the lines from mace_cli:717-729.
  - fix: Either remove/repurpose the unused help_text['convert'] block to avoid drift, or wire convert --help to print it (e.g. show_command_help before passthrough) so the curated examples are what users see. Until then, keep the block in sync with NewCifToD12.py since it remains the maintained doc.

### `credits`

- **[low] missing-help-block**
  - help says: Running `mace credits --help` prints the generic top-level help (usage: mace_cli [-h] [--no-banner] [--version] [--credits] ...) instead of any credits-specific help.
  - code: `credits` is a valid command (choices at mace_cli:1123) that dispatches at mace_cli:1611 to print get_credits(). But it has no entry in the help_text dict (mace_cli:589-1009, last key is 'plotting' at :898; dict closes at :1009) and it is not in the command-help routing list at mace_cli:1086. So `mace credits --help` n
  - evidence: Rendered `mace credits --help` -> 'usage: mace_cli [-h] [--no-banner] [--version] [--credits]' (top-level help, no credits section; grep for 'No help available' returns 0). Code: help_text dict mace_cli:589-1009 (no 'credits' key); routing list mace_cli:1086 lists workflow,submit,monitor,analyze,status,queue,manager,re
  - fix: Either add a short 'credits' entry to the help_text dict (e.g. describing that `mace credits` / `mace --credits` shows the development team) and add 'credits' to the routing list at mace_cli:1086, or accept that credits has no options and leave the top-level help fallback (documenting this intent in a comment).
- **[nit] stale-description**
  - help says: The 'Commands:' listing shown by `mace credits --help` (top-level epilog) lists: workflow, submit, monitor, analyze, completion, convert, opt2d12, opt2d3, opt2cif, manager, recover, database, engine, plotting. It does not list `credits` (nor version/status/queue), and it lists `completion`.
  - code: `credits` IS a valid command (argparse choices at mace_cli:1123) yet is omitted from that descriptive listing; conversely `completion` appears in the listing but is NOT among the argparse `choices` at mace_cli:1122-1123 (it is handled earlier as a passthrough command, mace_cli:1029).
  - evidence: Rendered `mace credits --help` 'Commands:' block includes 'completion  Check calculation status ...' but has no 'credits' line; choices line mace_cli:1123 = [...,'plotting','credits','version'] (includes credits/version, excludes completion); passthrough handling at mace_cli:1029.
  - fix: Note: this is the top-level epilog text (not the credits block specifically). For credits-discoverability, add a 'credits' line to the Commands listing in the top-level help epilog so users can find it; this is a top-level-help issue rather than a per-command credits issue.

### `database`

- **[low] required-vs-optional**
  - help says: history action help Options block: "--reason TEXT         Reason for change/rollback" with Operations listing "rollback  Rollback property to previous version" and example "mace database --action history --operation compare --material-id 1_dia --version 1 --version2 3" (no rollback example showing required args)
  - code: The rollback operation hard-requires four flags together; without all of them it errors out and does nothing: `if not material_id or not property_name or not version or not reason: print("Error: --material-id, --property, --version, and --reason required for rollback"); return`
  - evidence: Rendered: `mace database --action history --help` shows "--reason TEXT         Reason for change/rollback" (mace_cli:268) and Operations list (mace_cli:270-275). Code: mace_cli:2834-2837 requires --material-id, --property, --version, AND --reason for rollback.
  - fix: In the history action-help, note that --material-id, --property, --version and --reason are all REQUIRED for --operation rollback (and add a rollback usage example), rather than describing --reason as a generic optional field.
- **[low] option-in-code-not-in-help**
  - help says: Top-level ACTIONS list documents 13 actions ending with "aggregate     Aggregate properties by material groups"; no 'clean' action is listed.
  - code: Dispatch has an additional `elif action == 'clean':` branch that only prints "Database cleanup not yet implemented" (a no-op stub).
  - evidence: Rendered: `mace database --help` ACTIONS block (mace_cli:822-835) lists 13 actions, no 'clean'. Code: mace_cli:2925-2927 `elif action == 'clean': print("Database cleanup not yet implemented")`.
  - fix: Either remove the dead 'clean' stub from the dispatch, or leave it undocumented (acceptable since it is a harmless no-op). No user-facing help change strictly required; flagged for completeness.
- **[nit] stale-description**
  - help says: Top-level OPTIONS block: "--material-ids LIST   Comma-separated IDs (missing only)"
  - code: --material-ids is parsed and used by THREE actions: missing (mace_cli:2549), workflow (mace_cli:2671), and validate (mace_cli:2712), not just missing.
  - evidence: Rendered: `mace database --help` line "--material-ids LIST   Comma-separated IDs (missing only)" (mace_cli:845). Code: mace_cli:2549 (missing), 2671 (workflow), 2712 (validate) all parse --material-ids.
  - fix: Change the parenthetical from "(missing only)" to "(missing/workflow/validate)" or drop the action restriction, since the flag is honored by multiple actions (and each per-action help already documents it correctly).

### `engine`

- **[high] option-in-help-not-in-code**
  - help says: mace engine --action process --help lists: '--dry-run    Show what would be done without executing' and '--max-submit N    Maximum jobs to submit (default: 10)'
  - code: The engine argparse (engine.py main()) defines only --action, --material-id, --db, --work-dir. There is no --dry-run or --max-submit. The 'process' branch simply calls workflow_engine.process_completed_calculations() with no submit-limit or dry-run support. Running `mace engine --action process --dry-run` fails with 'e
  - evidence: Rendered: 'mace engine --action process --help' -> 'Options:\n  --material-id ID      Process specific material only\n  --dry-run             Show what would be done without executing\n  --max-submit N        Maximum jobs to submit (default: 10)'. Help source: mace_cli:425-428 (action_help['process']). Code: mace/workf
  - fix: Remove the --dry-run and --max-submit lines from action_help['process'] (mace_cli:427-428). The process action takes no options beyond the global --material-id/--db/--work-dir; --material-id isn't even read in the process branch. Note this also directly contradicts the top-level help's own line 'Note: The --dry-run opt
- **[high] option-in-help-not-in-code**
  - help says: mace engine --action workflow --help lists: '--list    List available workflow templates', '--show TEMPLATE    Show details of a specific template', '--create NAME    Create new workflow template'
  - code: The 'workflow' action takes no options; it just iterates db.get_all_materials() and prints a per-material progress summary. None of --list/--show/--create are defined in the argparse parser, and there is no template list/show/create logic in this action. Running `mace engine --action workflow --list` fails with 'engine
  - evidence: Rendered: 'mace engine --action workflow --help' -> 'Options:\n  --list                List available workflow templates\n  --show TEMPLATE       Show details of a specific template\n  --create NAME         Create new workflow template'. Help source: mace_cli:438-441 (action_help['workflow']). Code: mace/workflow/engin
  - fix: Rewrite action_help['workflow'] (mace_cli:433-444) to describe what the action actually does (print a workflow-completion summary across all materials in the database) and remove the fabricated --list/--show/--create options, or implement those flags in engine.py if intended. Also the help title 'Show or modify workflo
- **[medium] option-in-help-not-in-code**
  - help says: mace engine --action status --help lists: '--pending-only    Show only materials with pending workflows'
  - code: The status action has no --pending-only flag. The parser defines only --action/--material-id/--db/--work-dir. The status branch requires --material-id (prints 'Please specify --material-id for status checking' if absent) and has no pending-only filtering. `mace engine --action status --pending-only` would error with un
  - evidence: Rendered: 'mace engine --action status --help' -> 'Options:\n  --material-id ID      Show status for specific material\n  --pending-only        Show only materials with pending workflows'. Help source: mace_cli:411-412 (action_help['status']). Code: mace/workflow/engine.py:3565-3580 (status branch only uses args.materi
  - fix: Remove the '--pending-only' line from action_help['status'] (mace_cli:412), or implement it in engine.py. Only --material-id is real for this action.
- **[low] stale-description**
  - help says: mace engine --action process --help: 'Workflow progression:\n  OPT → SP → BAND/DOSS/TRANSPORT/CHARGE+POTENTIAL'
  - code: The 'workflow' action's own summary code assumes a 4-step workflow (OPT, SP, BAND, DOSS) at engine.py:3595 ('total_steps = 4  # OPT, SP, BAND, DOSS'), which is narrower than the TRANSPORT/CHARGE+POTENTIAL progression the process help advertises. This is a minor descriptive mismatch about the supported progression rathe
  - evidence: Rendered: 'mace engine --action process --help' -> 'Workflow progression:\n  OPT → SP → BAND/DOSS/TRANSPORT/CHARGE+POTENTIAL'. Code: mace/workflow/engine.py:3595 'total_steps = 4  # OPT, SP, BAND, DOSS'.
  - fix: Reconcile the documented progression with the code's actual step set (or vice versa). Low priority since process_completed_calculations may handle more types elsewhere; verify before changing wording.

### `mace --help (and bare mace)`

- **[low] option-in-help-not-in-code**
  - help says: Commands:\n  ...\n  completion  Check calculation status - categorize completed/errored jobs, organize files
  - code: 'completion' is listed in the epilog 'Commands:' block (mace_cli:1100) and is a working command, BUT it is NOT present in the argparse positional `choices` list (mace_cli:1122-1123). It works only because an early passthrough block intercepts argv before argparse runs (passthrough_commands at mace_cli:1029; dispatch at
  - evidence: Rendered help Commands block: 'completion  Check calculation status - categorize completed/errored jobs, organize files'. Rendered usage choices brace list: '{workflow,submit,monitor,analyze,convert,opt2d12,opt2d3,opt2cif,status,queue,manager,recover,database,engine,plotting,credits,version}' (no 'completion'). Code: e
  - fix: Add 'completion' to the argparse `choices` list at mace_cli:1122-1123 so the usage brace list and the descriptive Commands block agree (it already works via passthrough, so this is purely for consistency/discoverability).
- **[nit] stale-description**
  - help says: usage: mace_cli [-h] [--no-banner] [--version] [--credits] [{...,status,queue,...}]  — and the 'Commands:' descriptive block lists workflow/submit/monitor/.../plotting but does NOT list status or queue.
  - code: `status` and `queue` ARE accepted (in choices at mace_cli:1123) and dispatch to backward-compat redirects: status -> 'monitor --status' (mace_cli:1623-1631, prints "Note: 'mace status' has been merged into 'mace monitor --status'"), queue -> 'manager' (mace_cli:1727-1736, prints "Note: 'mace queue' is deprecated"). So 
  - evidence: Rendered usage shows '...,status,queue,...' in the choices; rendered Commands block has no 'status' or 'queue' entry. Code: choices mace_cli:1123; status redirect mace_cli:1623-1631; queue deprecation mace_cli:1727-1736.
  - fix: Either add a brief '(deprecated)' note line for status/queue in the Commands block, or remove them from the visible help while keeping them functional — at minimum make the usage brace list and the Commands descriptions consistent.
- **[nit] stale-description**
  - help says: options:\n  --version  show program's version number and exit\n  --credits  Show developer credits  — while the usage brace list ALSO contains positional choices 'credits' and 'version'.
  - code: 'credits' and 'version' are duplicated as both positional choices (mace_cli:1123; dispatch at credits mace_cli:1611-1621, version mace_cli:2954-2963) and as the --credits/--version flags (mace_cli:1127-1128). Both forms work but only the flags are described in the 'options:' section; the positional 'credits'/'version' 
  - evidence: Rendered usage choices include 'credits,version'; rendered Commands block omits both; options section documents '--version' and '--credits'. Code: choices mace_cli:1123; --version/--credits mace_cli:1127-1128; positional credits mace_cli:1611-1621; positional version mace_cli:2954-2963.
  - fix: Optional: note in the Commands block that 'credits' and 'version' are also available as positional commands (equivalent to the --credits/--version flags), or leave as-is since both forms function correctly.

### `manager`

- **[high] bad-example**
  - help says: --dry-run             Show what would be submitted without actually submitting
  - code: The CLI accepts --dry-run but does NOT implement dry-run semantics. It sets dry_run=True, then in the non-status branch prints 'Note: --dry-run is not supported by the queue manager; ignoring.' and proceeds to call queue.manager.main() which performs the REAL submission flow. So a user passing --dry-run gets actual job
  - evidence: Rendered help: '  --dry-run             Show what would be submitted without actually submitting'. Code: mace_cli:1665-1667 sets dry_run=True; mace_cli:1719-1722: 'if dry_run:' -> print('Note: --dry-run is not supported by the queue manager; ignoring.'); queue_main() is then called unconditionally at mace_cli:1725. Con
  - fix: Either remove the --dry-run line from the help text, or change its description to reflect reality, e.g. '--dry-run  (Not implemented; ignored. Use mace manager --status for a non-submitting view.)'. Long-term, implement actual dry-run support in queue/manager.py if the previewing behavior is desired.
- **[medium] option-in-code-not-in-help**
  - help says: (no mention of --organize in the Options list)
  - code: The CLI accepts --organize (sets organize=True), passes organize_outputs=organize to EnhancedCrystalQueueManager, and forwards --organize to queue.manager.main(). It is a real behavioral toggle: default is in-place submission (no copies); --organize copies each input into a <calc_type>/<material_id>/ folder.
  - evidence: Rendered help Options block lists only --status, --d12-dir, --max-jobs, --reserve, --callback-mode, --dry-run; no --organize. Code: mace_cli:1646 (organize=False default with comment 'submit in place'); mace_cli:1653-1655 (parses --organize); mace_cli:1676-1681 (organize_outputs=organize); mace_cli:1711-1714 (forwards 
  - fix: Add a line to the Options block: '  --organize            Copy inputs into <calc_type>/<material_id>/ folders (default: submit in place)' and optionally an example showing organized vs in-place layout.
- **[low] stale-description**
  - help says: --callback-mode MODE  Callback mode for completion handling
  - code: The underlying argparse restricts --callback-mode to choices ['completion', 'early_failure', 'status_check', 'submit_new', 'full_check'] with default 'completion'. The help text gives a vague description and lists no valid values or default, so a user cannot tell which MODE strings are accepted (an invalid value causes
  - evidence: Rendered help: '  --callback-mode MODE  Callback mode for completion handling'. Code: mace/queue/manager.py:1838-1843 add_argument('--callback-mode', choices=['completion','early_failure','status_check','submit_new','full_check'], default='completion'). The mace_cli example also only shows 'completion' (mace_cli:788).
  - fix: Document the valid values and default, e.g. '--callback-mode MODE  One of: completion (default), early_failure, status_check, submit_new, full_check'.
- **[nit] stale-description**
  - help says: Note: This runs the Enhanced Queue Manager (queue/manager.py) which requires SLURM.
  - code: The module is located at mace/queue/manager.py; the dispatch imports it as 'from queue.manager import ...' (a sys.path-relative path). The help cites the bare path 'queue/manager.py' which does not match the on-disk location under the mace/ package.
  - evidence: Rendered help: 'Note: This runs the Enhanced Queue Manager (queue/manager.py) which requires SLURM.' Actual file: /mnt/iscsi/UsefulScripts/Codebase/reorganization/mace/queue/manager.py (found via `find ... -name manager.py -path '*queue*'`); import at mace_cli:1634 'from queue.manager import EnhancedCrystalQueueManager
  - fix: Update the path reference to mace/queue/manager.py for accuracy (cosmetic).

### `monitor`

- **[high] option-in-help-not-in-code**
  - help says: Options:   --dashboard           Launch real-time monitoring dashboard (default)
  - code: The dashboard runs by DEFAULT with no flag (mace_cli:1577-1579 only injects '--action dashboard' when no --action is present). The literal flag '--dashboard' is passed straight through to the backend argparse, which does not define it, so `mace monitor --dashboard` errors with: 'material_monitor.py: error: unrecognized
  - evidence: Rendered: top-level `mace monitor --help` lists '--dashboard           Launch real-time monitoring dashboard (default)' (mace_cli:677). Backend argparse defines no --dashboard option (mace/queue/monitor.py:718-728). Verified at runtime: `mace monitor --dashboard --no-banner` -> 'material_monitor.py: error: unrecognized
  - fix: Remove the `--dashboard` option from the help (or change it to a note like 'dashboard is the default when no other flag is given'), since passing --dashboard literally is a broken command.
- **[high] option-in-help-not-in-code**
  - help says: Options:   --interval SECONDS    Update interval (default: 30)   --max-materials N     Limit display to N materials (default: unlimited)
  - code: The backend argparse (mace/queue/monitor.py:719-728) defines --base-dir, --db-path, --action, --interval, --output, --no-context, --workflow-id. There is NO --max-materials anywhere (grep finds it only in the help string mace_cli:361). `mace monitor --action dashboard --max-materials 5` errors: 'material_monitor.py: er
  - evidence: Rendered: `mace monitor --action dashboard --help` shows '--max-materials N     Limit display to N materials (default: unlimited)' (show_monitor_action_help, mace_cli:361). grep '--max-materials\|max_materials' over mace/queue/monitor.py: no match (only mace_cli:361). Verified at runtime: `mace monitor --action dashboa
  - fix: Remove `--max-materials` from the dashboard action-help, or implement it in mace/queue/monitor.py. (`--interval` is correct.)
- **[high] wrong-action-choice**
  - help says: show_monitor_action_help documents an action 'health': `mace monitor --action health` -> 'Check database health and integrity... Database connection tests, Schema validation, Index optimization, ...'
  - code: The backend's --action choices are exactly ['dashboard','status','report','stats'] (mace/queue/monitor.py:721). 'health' is not a valid choice, so `mace monitor --action health` errors: "material_monitor.py: error: argument --action: invalid choice: 'health' (choose from 'dashboard', 'status', 'report', 'stats')".
  - evidence: Rendered: `mace monitor --action bogus --help` lists 'Available actions: dashboard, health, stats' (show_monitor_action_help dict, mace_cli:382-393). Backend: parser.add_argument('--action', choices=['dashboard','status','report','stats'], default='dashboard') at mace/queue/monitor.py:721. Verified at runtime: `mace mo
  - fix: Remove the 'health' action from show_monitor_action_help (or add 'health' to the backend --action choices and implement it). Align the documented action set with the backend choices.
- **[medium] option-in-code-not-in-help**
  - help says: Available actions advertised by show_monitor_action_help: dashboard, health, stats. The 'report' action is never documented.
  - code: The backend supports --action report (mace/queue/monitor.py:721, handled at :793-803) which generates a detailed JSON system report (using --output). This valid, working action is absent from both the action-help dict and the action listing.
  - evidence: Backend choices include 'report' (mace/queue/monitor.py:721) with a real handler 'elif args.action == "report":' (mace/queue/monitor.py:793). Rendered action-help dict keys are only dashboard/stats/health (mace_cli:354-393); `mace monitor --action bogus --help` -> 'Available actions: dashboard, health, stats' (no 'repo
  - fix: Add a 'report' entry to show_monitor_action_help documenting `mace monitor --action report [--output FILE]`.
- **[medium] stale-description**
  - help says: Top-level `mace monitor --help` Options:   --material-id ID      Show status for specific material
  - code: In the --status path, mace_cli parses --material-id into a local variable (mace_cli:1512-1513) but never uses it to filter the displayed status; the only later references to material_id are dict keys (calc['material_id']). Running `mace monitor --status --material-id foo` returns the full database status (76 materials)
  - evidence: mace_cli:1512-1513 assigns material_id; no subsequent use as a filter in the --status block (1499-1573); get_database_stats()/get_recent_calculations() are called without material_id. Verified at runtime: `mace monitor --status --material-id foo` printed 'Materials: 76 / Calculations: 109' (whole DB), not filtered.
  - fix: Either implement material-specific filtering in the --status branch, or change the help text to note that --material-id is currently not honored in status mode.
- **[low] option-in-code-not-in-help**
  - help says: Neither the top-level monitor help nor the action-help mentions --output, --base-dir, --db-path, --no-context, or --workflow-id.
  - code: The backend defines real, working flags: --base-dir (default '.'), --db-path (default 'materials.db'), --output (report output file), --no-context (disable workflow context detection), --workflow-id (specific workflow to monitor). These pass through the dispatch (mace_cli:1582 sends all_args to the backend) and are usa
  - evidence: mace/queue/monitor.py:719-728 (--base-dir:719, --db-path:720, --output:725, --no-context:726, --workflow-id:728). Dispatch passes args through at mace_cli:1582. None appear in the rendered `mace monitor --help` (mace_cli:671-691) or the action-help (mace_cli:354-393).
  - fix: Document the useful pass-through flags (at minimum --output for reports and --workflow-id for context selection) in the monitor help.

### `opt2cif`

- **[low] missing-help-block**
  - help says: help_text dict has 'opt2cif': block at mace_cli:757-770 (e.g. 'For full options, the command will show CrystalOutToCif.py help when run without arguments.')
  - code: The opt2cif help_text block is never displayed. opt2cif is in passthrough_commands (mace_cli:1029); for any invocation containing --help/-h the passthrough branch (mace_cli:1040-1083) re-points sys.argv to CrystalOutToCif.py and calls its main(), so argparse there prints help and sys.exit(0) runs (mace_cli:1083). opt2c
  - evidence: Rendered: first line is 'usage: CrystalOutToCif.py [-h] ...' (argparse, not the help_text block). Code: passthrough list mace_cli:1029; opt2cif dispatch mace_cli:1073-1077; help-eligible list excluding opt2cif mace_cli:1086; help_text block mace_cli:757-770.
  - fix: Either remove the dead help_text['opt2cif'] block, or make it the source of truth by adding opt2cif to the show_command_help-eligible list and intercepting --help before the passthrough. Minimally, update the block so it matches the real options if a maintainer ever surfaces it.
- **[nit] bad-example**
  - help says: usage: CrystalOutToCif.py [-h] ... and Examples: 'CrystalOutToCif.py material.out', 'CrystalOutToCif.py . --output-dir cifs/', etc.
  - code: The user invokes the command as `mace opt2cif`, but the passthrough sets sys.argv[0] = 'CrystalOutToCif.py' (mace_cli:1076), so argparse prints the raw script name in the usage line and %(prog)s examples. The shown commands are not what the user typed and would not be directly copy-pasteable as `mace` invocations.
  - evidence: Rendered usage line: 'usage: CrystalOutToCif.py [-h] [--output-dir OUTPUT_DIR] ...' and examples like 'CrystalOutToCif.py material.out  # Convert single file'. Code sets the prog name at mace_cli:1076 (sys.argv = ['CrystalOutToCif.py'] + sys.argv[2:]); examples use %(prog)s at Crystal_d12/CrystalOutToCif.py:998-1005.
  - fix: Set sys.argv[0] to 'mace opt2cif' in the passthrough (mace_cli:1076) so the usage line and %(prog)s examples render as the actual user-facing command.

### `opt2d12`

- **[medium] bad-example**
  - help says: help_text['opt2d12'] block (mace_cli:730-743): 'Usage: mace opt2d12 <output_file> [options]' and Examples 'mace opt2d12 calculation.out --sp', '--freq', '--both'
  - code: CRYSTALOptToD12.py main() defines NO positional argument and NO --sp/--freq/--both flags. The output file is supplied via the OPTIONAL flag --out-file; calc type is --calc-type {SP,OPT,FREQ}. A bare positional or --sp/--freq/--both is rejected by argparse as 'unrecognized arguments' (exit 2).
  - evidence: Rendered help_text block source quote 'Usage: mace opt2d12 <output_file> [options]' / 'mace opt2d12 calculation.out --sp' at mace_cli:731,738-740. Real argparse: CRYSTALOptToD12.py:1228 (--out-file flag, no positional), 1267-1271 (--calc-type {SP,OPT,FREQ}); no --sp/--freq/--both anywhere. Verified at runtime: `mace --
  - fix: Update the help_text['opt2d12'] block to match the real CLI: usage 'mace opt2d12 --out-file FILE [--calc-type SP|OPT|FREQ] [options]' with examples like 'mace opt2d12 --out-file optimized.out --calc-type SP --non-interactive'. Remove the <output_file> positional and the --sp/--freq/--both examples. (Note: this block is

### `opt2d3`

- **[low] stale-description**
  - help says: Usage: mace opt2d3 <output_file> --calc-type TYPE [options]  Examples:   mace opt2d3 optimized.out --calc-type BAND   mace opt2d3 optimized.out --calc-type DOSS   mace opt2d3 optimized.out --calc-type ALL --mode advanced
  - code: The actual command takes the output file via the --input/-i FLAG, not a positional argument, and has no positional at all. `mace opt2d3 optimized.out --calc-type BAND` fails with `error: unrecognized arguments: optimized.out`. There is also no `ALL` calc-type (choices are BAND/DOSS/TRANSPORT/CHARGE/POTENTIAL/CHARGE+POT
  - evidence: help_text dict block: mace_cli:744-756 (Usage line mace_cli:745; examples mace_cli:751-753). Code: argparse defines `--input`,`-i` with no positional (Crystal_d3/CRYSTALOptToD3.py:1259-1262); calc-type choices (CRYSTALOptToD3.py:1266) lack ALL; no --mode anywhere. Verified at runtime: `mace --no-banner opt2d3 optimized
  - fix: Either delete the unreachable opt2d3 dict block (mace_cli:744-756) since the passthrough already shows the underlying tool's accurate argparse help, or correct it to: `Usage: mace opt2d3 --input <output_file> --calc-type TYPE [options]` with examples `mace opt2d3 --input optimized.out --calc-type BAND` / `... --calc-ty
- **[nit] other**
  - help says: usage: CRYSTALOptToD3.py [-h] [--input INPUT] ...
  - code: The rendered `mace opt2d3 --help` usage line shows the internal script name `CRYSTALOptToD3.py` instead of `mace opt2d3`, because the passthrough rewrites sys.argv[0] to 'CRYSTALOptToD3.py' before calling the underlying main. Content is otherwise fully accurate and auto-generated from the real argparse.
  - evidence: Rendered: `usage: CRYSTALOptToD3.py [-h] [--input INPUT] [--calc-type {BAND,DOSS,...}] ...` (from `mace_cli --no-banner opt2d3 --help`). Code: sys.argv set to ['CRYSTALOptToD3.py'] + args at mace_cli:1071 then opt2d3_main() at mace_cli:1072; argparse with no prog= at CRYSTALOptToD3.py:1255-1257.
  - fix: Cosmetic only; if desired, set prog='mace opt2d3' in the ArgumentParser (CRYSTALOptToD3.py:1255) or pass it when invoked from mace_cli so the usage line reads `mace opt2d3` instead of the internal script name. Affects all passthrough commands consistently.

### `plotting`

- **[low] option-in-code-not-in-help**
  - help says: Cube Options block lists: --density/--esp/--spin, --diff, --iso, --view {iso,slice,slice-all}, --slice AXIS POS, --colorscale, --no-atoms/--bonds/--publication, --engine-args (mace_cli:936-943)
  - code: create_parser also defines four real cube flags not in the help: --slice-all AXIS (a standalone convenience flag, distinct from --view slice-all), --log, --linear, and --clip (type=float, default=99.5).
  - evidence: Help (mace_cli:936-943) cube block omits these. Code: mace/plotting/main.py:329 (--slice-all metavar AXIS), :335 (--log), :336 (--linear), :337-338 (--clip default 99.5).
  - fix: Add to the Cube Options block: '--slice-all AXIS  Grid of slices along AXIS (x/y/z)', '--log / --linear  Cube color scale (log vs linear)', and '--clip PCT  Color clip percentile (default: 99.5)'.
- **[low] option-in-code-not-in-help**
  - help says: Vibrational Mode Options block lists: --list-modes, --mode N, --all-modes, --gif, and '--amplitude F / --normalize / --static' (mace_cli:945-950)
  - code: The FREQ argument group also defines --gif-fps (type=int, default=20) and --frames (type=int, downstream default 30) which are absent from the help text.
  - evidence: Help (mace_cli:945-950) lists no --gif-fps or --frames. Code: mace/plotting/main.py:355-356 (--gif-fps default 20), :361-362 (--frames). Downstream defaults confirmed at mace/plotting/handlers/freq.py:74 (frames=30) and :76 (gif_fps=20).
  - fix: Add to the Vibrational Mode Options block: '--gif-fps N  GIF frames per second (default: 20)' and '--frames N  Animation frame count (default: 30)'.
- **[nit] option-in-code-not-in-help**
  - help says: Modes block documents the default interactive mode only as '(default)  Interactive mode - guided configuration' (mace_cli:905)
  - code: There is an explicit -i/--interactive flag (store_true) to force interactive mode; the help never names the flag, only the implicit default behavior.
  - evidence: Help (mace_cli:905) shows '(default)' with no flag. Code: mace/plotting/main.py:174-178 defines '-i', '--interactive'.
  - fix: Optionally note the explicit flag, e.g. '-i, --interactive       Interactive mode - guided configuration (default if no other mode)'.

### `queue`

- **[nit] stale-description**
  - help says: Usage: mace queue [options]\n\nDeprecated: Use 'mace manager' instead.\nThis command will be removed in a future version.
  - code: The dispatch forwards ALL arguments to `manager` (sys.argv = [sys.argv[0], 'manager'] + command_args('queue'); main()), so `mace queue` transparently accepts every `mace manager` option (e.g. --status, --d12-dir, --max-jobs, --reserve, --callback-mode). The help's '[options]' is correct but unspecific; it does not expl
  - evidence: Rendered help: 'Usage: mace queue [options]\n\nDeprecated: Use 'mace manager' instead.\nThis command will be removed in a future version.' (mace_cli:792-797). Dispatch forwards everything to manager: mace_cli:1734 `sys.argv = [sys.argv[0], 'manager'] + command_args('queue')` and mace_cli:1735 `main()`; command_args ret
  - fix: Optional: append a single line such as 'All options are forwarded to `mace manager` (see `mace manager --help`).' This is cosmetic; the existing stub is already accurate and intentional for a deprecated shim.

### `recover`

- **[high] option-in-help-not-in-code**
  - help says: Options:   --error-type TYPE     Focus on specific error type   --material-id ID      Recover specific material only
  - code: recovery.py main() only adds --action, --config, --db, --max-recoveries, --create-config. There is no --error-type or --material-id argument, so `mace recover --error-type X` / `--material-id Y` triggers argparse 'unrecognized arguments' and exits with error 2.
  - evidence: Rendered `mace recover --action recover --help` lists '--error-type TYPE' and '--material-id ID' (source mace_cli:543-544). Code: mace/recovery/recovery.py:819-829 add_argument calls contain no such options (only --action, --config, --db, --max-recoveries, --create-config).
  - fix: Remove --error-type and --material-id from the recover action-help, or implement them in recovery.py main(). As written they are broken commands.
- **[high] option-in-help-not-in-code**
  - help says: Usage: mace recover --action config [options] Options:   --show                Display current configuration   --edit                Open configuration in editor   --validate            Check configuration validity
  - code: --action config simply calls recovery_engine.save_default_config() and prints 'Configuration saved to ...' (it overwrites/creates the config). There is no --show, --edit, or --validate flag; passing them causes argparse 'unrecognized arguments' (exit 2).
  - evidence: Rendered `mace recover --action config --help` lists '--show', '--edit', '--validate' (source mace_cli:569-571). Code: mace/recovery/recovery.py:836-839 `if args.create_config or args.action == 'config': recovery_engine.save_default_config(); print(...)`. No such flags exist in the argparse (recovery.py:819-829).
  - fix: Remove --show/--edit/--validate from the config action-help. Describe the real behavior: '--action config' (equivalent to --create-config) writes the default recovery configuration file.
- **[medium] wrong-default**
  - help says: --max-recoveries N    Maximum recovery attempts per material (default: 3)
  - code: argparse defines --max-recoveries with default=10; the top-level help block correctly says default 10. Only the --action recover action-help states 3.
  - evidence: Rendered `mace recover --action recover --help`: '--max-recoveries N    Maximum recovery attempts per material (default: 3)' (source mace_cli:542). Code: mace/recovery/recovery.py:826-827 `parser.add_argument("--max-recoveries", type=int, default=10, ...)`. Also note detect_and_recover_errors caps at the config max_con
  - fix: Change the action-help to '(default: 10)' to match the argparse default and the top-level help; remove the inaccurate 'per material' framing (it is a total cap).
- **[low] option-in-code-not-in-help**
  - help says: Neither the top-level help (mace_cli:803-807) nor any action-help mentions a --db option.
  - code: recovery.py main() accepts --db (path to materials database, default 'materials.db'), which is forwarded verbatim by the mace dispatch.
  - evidence: Code: mace/recovery/recovery.py:824-825 `parser.add_argument("--db", default="materials.db", help="Path to materials database")`. Dispatch passes all args through: mace_cli:1742-1744. The flag is absent from the rendered top-level and action help.
  - fix: Add '--db FILE   Path to materials database (default: materials.db)' to the top-level recover help options list.
- **[low] stale-description**
  - help says: Recoverable errors:   - SHRINK factor issues   - Memory allocation failures   - SCF convergence problems   - Basis set issues
  - code: The default config auto-recovers shrink_error (fixk_handler), memory_error, convergence_error (SCF), timeout_error, and disk_space_error. Basis-set errors are explicitly NOT auto-recovered: the shipped config marks basis_set_error / basis_linear_dependence as 'MANUAL INTERVENTION REQUIRED' / 'manual_escalation'. So the
  - evidence: Rendered `mace recover --action recover --help` lists 'Basis set issues' as recoverable (source mace_cli:547-550). Code: default_config handlers at mace/recovery/recovery.py:90-122 (shrink_error/memory_error/convergence_error/timeout_error/disk_space_error); no basis handler in recovery.py. Shipped mace/config/recovery
  - fix: Update the recoverable-errors list to match real auto-recovery handlers (SHRINK, memory, SCF convergence, timeout/walltime, disk space) and drop or re-label 'Basis set issues' as requiring manual intervention.

### `status`

- **[high] missing-help-block**
  - help says: Rendered output of `mace status --help`: "No help available for command: status"
  - code: `status` is in the argparse choices (mace_cli:1123) and in the per-command --help interception list (mace_cli:1086), which routes `mace status --help` to show_command_help('status'). The help_text dict in show_command_help (mace_cli:589-1009) contains keys for workflow, submit, monitor, analyze, convert, opt2cif, manag
  - evidence: Rendered: `$ mace_cli status --help` -> "No help available for command: status". Code: mace_cli:1086 (status in --help interception list), mace_cli:1123 (status in choices), mace_cli:589 `help_text = {` with keys at lines 590-1009 and NO 'status' key (verified by grep of lines 589-1009 for "status':" -> no match), mace
  - fix: Add a 'status' key to the help_text dict in show_command_help (mace_cli ~589) documenting that the command is a backward-compat alias, e.g.: "Usage: mace status [options]\n\n'mace status' is a backward-compatibility alias for 'mace monitor --status'.\nAll arguments are forwarded to monitor. See: mace monitor --help". T
- **[low] stale-description**
  - help says: Epilog 'Commands:' list shown by `mace --help` (mace_cli:1096-1118) and the top-level choices (mace_cli:1123) do not mention `status` at all, even though it is an accepted, working command.
  - code: `status` is a valid choice (mace_cli:1123) and dispatches to a working redirect (mace_cli:1623-1633), but it is intentionally omitted from the epilog Commands list (mace_cli:1096-1118) presumably because it is deprecated/merged into monitor. This is consistent with deprecation but means the only way a user learns of th
  - evidence: mace_cli:1096-1118 (epilog Commands block lists workflow..plotting but not status), mace_cli:1123 (status present in choices), mace_cli:1623-1633 (working dispatch). Runtime note at mace_cli:1625: "Note: 'mace status' has been merged into 'mace monitor --status'".
  - fix: Acceptable to leave out of the main Commands list if deprecation is intentional, but pair with fixing status-1 so `mace status --help` at least explains the redirect. Optionally add a one-line deprecation note under monitor in the epilog.

### `version`

- **[low] missing-help-block**
  - help says: `mace version --help` renders the full top-level help (usage: mace_cli [-h] ... {workflow,submit,...,version} ... / 'Commands:' listing / 'Quick Start' / 'More Help') — i.e. global help, not anything describing the `version` command.
  - code: There is no 'version' entry in the help_text dict in show_command_help() (mace_cli:589-1009), and `version` is omitted from the command list at mace_cli:1086 that routes `<cmd> --help` to show_command_help(). So `mace version --help` is never handled as command help; it reaches argparse (mace_cli:1091+) where `--help` 
  - evidence: Rendered: `/home/marcus/anaconda3/bin/python mace_cli version --help` prints 'usage: mace_cli [-h] [--no-banner] [--version] [--credits] [{workflow,...,version}] [args ...]' followed by the global 'Commands:' / 'Quick Start' epilog (top-level help), exit 0 — no version-specific text. Code: mace_cli:1086 allow-list = ['
  - fix: Add a 'version' entry to the help_text dict and add 'version' to the allow-list at mace_cli:1086, so `mace version --help` prints a short description: e.g. 'version - Show MACE version and supported CRYSTAL releases (CRYSTAL17, CRYSTAL23). Equivalent flag: `mace --version` (prints `MACE v<x.y.z>`).' Alternatively, rout

### `workflow`

- **[high] stale-description**
  - help says: For action-specific options, use: mace workflow --action <ACTION> --help
  - code: The workflow command delegates entirely to run_mace.py's main(), which defines NO --action argument. mace_cli reconstructs sys.argv from everything after 'workflow' (mace_cli:1190-1195) and calls run_mace.main(). run_mace.py's argparse (mace/run_mace.py:90-113) has only the mutually-exclusive mode group and a few optio
  - evidence: Rendered: 'For action-specific options, use: mace workflow --action <ACTION> --help' (mace_cli:617). Code: run_mace.py:90-113 defines no --action; dispatch mace_cli:1185-1195 passes args straight to run_mace.main(); verified runtime: `mace workflow --action plan` -> 'run_mace.py: error: one of the arguments --interacti
  - fix: Remove the '--action <ACTION>' footer from the workflow help block (mace_cli:617). The workflow command is flag-driven (--interactive/--execute/--quick-start/--status/--show-templates), not action-driven.
- **[high] option-in-help-not-in-code**
  - help says: mace workflow --action plan/execute/status/templates with options --config FILE, --template NAME, --output FILE, --plan FILE, --resume, --skip-validation, --workflow-id ID, --active-only, --detailed (entire show_workflow_action_help block, mace_cli:453-517)
  - code: None of these actions or options exist. There is no --action dispatch for workflow and run_mace.py's argparse (mace/run_mace.py:90-113) defines none of --config, --template, --output, --plan, --resume, --skip-validation, --workflow-id, --active-only, or --detailed. The action-help text is printed only by mace_cli's pre
  - evidence: Rendered `mace workflow --action execute --help` shows '--plan FILE  Workflow plan JSON file to execute', '--resume', '--skip-validation' (mace_cli:472-487). Real flag is '--execute', metavar=PLAN_FILE (run_mace.py:93-94); no --plan/--resume/--skip-validation in run_mace.py:90-113. Intercept at mace_cli:1022-1026.
  - fix: Delete show_workflow_action_help (mace_cli:453-531) and the workflow branch in show_action_specific_help, OR rewrite it to describe the real flag-driven modes (--interactive, --execute PLAN_FILE, --quick-start, --status, --show-templates) and their actual options (--cif-dir, --d12-dir, --workflow, --work-dir, --db-path
- **[medium] stale-description**
  - help says: --workflow TYPE   Workflow template: basic_opt, opt_sp, full_electronic, double_opt, complete, transport_analysis, charge_analysis, combined_analysis (default: full_electronic)
  - code: run_mace.py's --workflow choices include 9 templates; the help lists only 8 and omits 'opt_sp_freq' (OPT -> SP -> FREQ). A user copying the help would not know opt_sp_freq is selectable, though the default (full_electronic) is correct.
  - evidence: Rendered: '--workflow TYPE ... basic_opt, opt_sp, full_electronic, double_opt, complete, transport_analysis, charge_analysis, combined_analysis (default: full_electronic)' (mace_cli:603-605). Code: choices=["basic_opt", "opt_sp", "opt_sp_freq", "full_electronic", "double_opt", "complete", "transport_analysis", "charge_
  - fix: Add 'opt_sp_freq' to the --workflow template list in the help block (mace_cli:603-605) so all 9 valid choices are documented.
- **[low] stale-description**
  - help says: Templates include: basic_opt, opt_sp, full_electronic, transport_analysis, complete (mace workflow --action templates)
  - code: The action-help templates list (mace_cli:510-514) shows only 5 of the 9 real templates; it omits opt_sp_freq, double_opt, charge_analysis, and combined_analysis. (This is part of the wholly-fabricated action-help block, but the template subset is independently incomplete versus run_mace.py:77-85 / choices at run_mace.p
  - evidence: Rendered `mace workflow --action templates`: 'basic_opt / opt_sp / full_electronic / transport_analysis / complete' (mace_cli:510-514). Code lists 9 templates (mace/run_mace.py:77-85 epilog and choices at 106-107), including the 4 missing ones.
  - fix: If the action-help block is kept, sync its template list with run_mace.py's 9 templates; ideally remove this block (see wf-2) since `mace workflow --show-templates` is the working path.
