"""`mace recover`: fix failed tracked calculations and resubmit them.

The pieces are the ones the queue manager's job callback already uses; this
module only finds the failed rows, shows what would happen and, once
confirmed, hands each calculation to that same path:

1. scan the tracked database(s) for rows with status 'failed' (scoped by
   --calc-id, --dir, --errors);
2. classify each output as the queue manager does
   (EnhancedCrystalQueueManager.analyze_calculation_error, which reads the
   shared table in mace.utils.crystal_errors, plus SLURM's walltime verdict);
   every class has a handler or a "no automatic recovery: why";
3. show a table: calculation, error class, proposed fix, retries used of the
   limit (per error type from recovery_config.yaml, and per lineage); a step
   of a paused or canceled workflow is skipped (resume it first), as the job
   callback skips it;
4. --dry-run: run the fix on copies of the deck, job script and output and
   show the diff of every file it would write and the sbatch command;
5. otherwise, after confirmation (or --yes), run one worker per calculation
   the way the job callback runs: a fresh Python in the calculation's folder
   with the job script's workflow exports, which takes the shared OS queue
   lock (QUEUE_STATE_LOCK), checks again that the row is still failed, that
   its workflow is not paused or canceled (``mace workflow --pause/--cancel``)
   and that no job of it is queued or running (squeue and the database), checks
   the max-jobs throttle and then calls the queue manager's
   attempt_error_recovery -> ErrorRecoveryEngine.attempt_recovery ->
   resubmit_fixed_calculation. Tracking, workflow_id/workflow_step, the
   lineage retry limits and the job-script reuse rule all hold as in the
   callback.

A job that ended more than MACE_RECOVER_MAX_AGE_DAYS (default 3) days ago is
never resubmitted by the callbacks. An explicit `mace recover` may resubmit
it, after its own question (or --yes --include-stale).
"""

import argparse
import contextlib
import difflib
import io
import json
import os
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PREFIX = "MACE_RECOVER_RESULT "
STALE_DAYS_DEFAULT = 3.0
WORKER_TIMEOUT_S = 3600
# What the job scripts export for the job callback (the workflow context).
_EXPORT_RE = re.compile(r'^export (MACE_WORKFLOW_ID|MACE_CONTEXT_DIR|MACE_ISOLATION_MODE)="([^"]*)"', re.M)
_CONTEXT_VARS = ("MACE_WORKFLOW_ID", "MACE_CONTEXT_DIR", "MACE_ISOLATION_MODE")
ACTIVE_STATUSES = ("submitted", "running")
QUEUE_ACTIVE = ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "REQUEUED", "RESIZING")


# ------------------------------------------------------------------ helpers
@contextlib.contextmanager
def _quiet():
    """Swallow the constructors' chatter (lock directory, context database)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf


def stale_days() -> float:
    try:
        return float(os.environ.get('MACE_RECOVER_MAX_AGE_DAYS') or STALE_DAYS_DEFAULT)
    except ValueError:
        return STALE_DAYS_DEFAULT


def _class_key(error_type: str) -> str:
    """The shared classifier's class for an error type (memory_error -> oom),
    or the error type itself when it is decided before the table (finalrun_loop)."""
    from mace.utils import crystal_errors
    for c in crystal_errors.ERROR_CLASSES:
        if c.error_type == error_type:
            return c.key
    return error_type


def _no_recovery_reason(error_type: str, message: str) -> str:
    from mace.utils import crystal_errors
    from mace.recovery import screened_hybrid
    why = next((c.no_recovery for c in crystal_errors.ERROR_CLASSES
                if c.error_type == error_type and c.no_recovery), None)
    if why:
        return why
    if error_type == screened_hybrid.ERROR_TYPE:
        return "more memory cannot help; the functional is the user's to change"
    if error_type == 'no_output':
        return "the job wrote no output; read its SLURM log (<job>-<id>.o)"
    if error_type in ('unknown_error', 'crystal_error', 'file_error'):
        return f"MACE has no fix for this ({message})" if message else "MACE has no fix for this"
    return f"no recovery handler for {error_type}"


def describe_fix(handler: str, cfg: Dict) -> str:
    """What a recovery handler will do, in the words of its configuration."""
    if handler == 'fixk_handler':
        return "new deck with the SHRINK factors fixed (fixk.py)"
    if handler == 'memory_handler':
        return (f"new job script: memory x{cfg.get('memory_factor', 1.5)} "
                f"(at most {cfg.get('max_memory', '200GB')})")
    if handler == 'convergence_handler':
        return (f"new deck: SCF MAXCYCLE +{cfg.get('max_cycles_increase', 1000)}, "
                f"FMIXING -{cfg.get('fmixing_adjustment', 10)} (not below 10)")
    if handler == 'timeout_handler':
        text = (f"new job script: walltime x{cfg.get('walltime_factor', 2.0)} "
                f"(at most {cfg.get('max_walltime', '7-00:00:00')} and the queue's limit)")
        if cfg.get('enable_restart', True):
            text += "; an optimization goes on from where it stopped"
        return text
    if handler == 'opt_fresh_start_handler':
        return "deck rewritten as a fresh optimization from the best geometry reached"
    if handler == 'finalrun_loop_handler':
        return "deck rewritten as a fresh optimization from the best geometry, TOLINTEG one level up"
    if handler == 'cleanup_handler':
        return "large temporary files (over 100 MB) removed from the work folder; deck unchanged"
    return f"{handler}"


def _resolve_script(calc: Dict) -> Optional[Path]:
    from mace.recovery.recovery import _resolve_job_script
    return _resolve_job_script(calc)


def _output_of(calc: Dict) -> Optional[Path]:
    """The calculation's .out: the one on record, else <input stem>.out."""
    if calc.get('output_file') and Path(calc['output_file']).is_file():
        return Path(calc['output_file'])
    if calc.get('input_file'):
        cand = Path(calc['input_file']).with_suffix('.out')
        if cand.is_file():
            return cand
    return None


def script_exports(calc: Dict) -> Dict[str, str]:
    """The workflow-context variables the calculation's job script exports
    (what its job callback runs with)."""
    script = _resolve_script(calc)
    if script is None:
        return {}
    try:
        return dict(_EXPORT_RE.findall(script.read_text(errors='ignore')))
    except OSError:
        return {}


def _settings(calc: Dict) -> Dict:
    try:
        return json.loads(calc.get('settings_json') or '{}')
    except (json.JSONDecodeError, TypeError):
        return {}


def queue_rows(env: Optional[Dict[str, str]] = None) -> Optional[List[Dict[str, str]]]:
    """The user's queued jobs ({id, name, state, workdir}), None without squeue."""
    from mace.queue.slurm_ids import slurm_env
    user = os.environ.get('USER') or os.environ.get('LOGNAME') or ''
    try:
        r = subprocess.run(['squeue', '-h', '-u', user, '-o', '%i|%j|%T|%Z'], capture_output=True,
                           text=True, timeout=60, env=slurm_env(env))
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    rows = []
    for line in r.stdout.splitlines():
        parts = line.strip().split('|')
        if len(parts) < 4:
            continue
        rows.append({'id': parts[0], 'name': '|'.join(parts[1:-2]), 'state': parts[-2],
                     'workdir': parts[-1]})
    return rows


def _same_dir(a: str, b: str) -> bool:
    try:
        return os.path.realpath(a) == os.path.realpath(b)
    except (OSError, ValueError):
        return False


def queued_twin(calc: Dict, queue: List[Dict[str, str]]) -> Optional[str]:
    """Why this calculation is already in the queue, or None: its own job is
    still there, or a job of its deck (the deck name, or a recovery of it)
    runs or waits in its folder."""
    from mace.queue.slurm_ids import expand_queue_id, Truncated
    own = str(calc.get('slurm_job_id') or '')
    stem = Path(calc['input_file']).stem if calc.get('input_file') else None
    for q in queue:
        if q['state'] and q['state'] not in QUEUE_ACTIVE:
            continue
        ids = expand_queue_id(q['id'])
        ids = [q['id']] if isinstance(ids, Truncated) or not ids else ids
        if own and (own in ids or own == q['id']):
            return f"its job {own} is still in the queue ({q['state']})"
        if stem and (q['name'] == stem or q['name'].startswith(stem + "_recovery_")):
            if not q['workdir'] or not calc.get('work_dir') or _same_dir(q['workdir'], calc['work_dir']):
                return f"job {q['id']} ({q['name']}) of this deck is already in the queue ({q['state']})"
    return None


def workflow_of(calc: Dict) -> Optional[str]:
    """The workflow a calculation belongs to: its settings, else what its job
    script exports (the job callback's own fallback)."""
    return _settings(calc).get('workflow_id') or script_exports(calc).get('MACE_WORKFLOW_ID') or None


def workflow_hold(calc: Dict, extra_dirs=()) -> Optional[Tuple[str, str]]:
    """(workflow id, 'paused'|'canceled') when the calculation's workflow is
    held by its control file (``mace workflow --pause/--cancel``), else None.
    The control file is looked for as the job callback looks for it: from
    the calculation's folder, its deck's folder and ``extra_dirs`` upward."""
    wid = workflow_of(calc)
    if not wid:
        return None
    from mace.workflow.control import held_state
    starts = [calc.get('work_dir'),
              str(Path(calc['input_file']).parent) if calc.get('input_file') else None, *extra_dirs]
    try:
        state = held_state(wid, starts)
    except Exception:
        return None
    return (wid, state) if state else None


def hold_reason(held: Tuple[str, str]) -> str:
    wid, state = held
    return (f"workflow {wid} is {state}: resume it first (mace workflow --resume {wid}); a {state} "
            "workflow submits no recovery job")


def active_successor(db, calc: Dict) -> Optional[str]:
    """Why a newer run of this calculation is already under way, or None: a
    later row of the same material and calculation type that is submitted
    or running."""
    try:
        with db._get_connection() as conn:
            row = conn.execute(
                "SELECT calc_id, slurm_job_id, status FROM calculations"
                " WHERE material_id = ? AND calc_type = ? AND calc_id != ?"
                "   AND status IN ('submitted', 'running') AND created_at >= ?"
                " ORDER BY created_at DESC LIMIT 1",
                (calc['material_id'], calc['calc_type'], calc['calc_id'],
                 calc.get('created_at') or '')).fetchone()
    except Exception:
        return None
    if row:
        return f"a newer run is {row[2]} ({row[0]}, job {row[1] or '?'})"
    return None


# ------------------------------------------------------------------ databases
def discover_dbs(db_arg: Optional[str], dir_arg: Optional[str]) -> List[Path]:
    """--db, else materials.db in --dir (or here) and the workflow context
    databases there (.mace_context_*/materials.db)."""
    if db_arg:
        p = Path(db_arg).expanduser()
        return [p.resolve()] if p.is_file() else []
    base = Path(dir_arg).expanduser() if dir_arg else Path.cwd()
    found = []
    for cand in [base / "materials.db", *sorted(base.glob(".mace_context_*/materials.db"))]:
        if cand.is_file() and cand.resolve() not in found:
            found.append(cand.resolve())
    if dir_arg and not found and (Path.cwd() / "materials.db").is_file():
        found.append((Path.cwd() / "materials.db").resolve())
    return found


def _inside(path: Optional[str], folder: Path) -> bool:
    if not path:
        return False
    try:
        p = os.path.realpath(path)
        f = os.path.realpath(str(folder))
        return p == f or p.startswith(f.rstrip(os.sep) + os.sep)
    except (OSError, ValueError):
        return False


# ------------------------------------------------------------------ the plan
class Planner:
    """Classification, fix and retry state of failed rows, read-only."""

    def __init__(self, db_path: Path, args):
        from mace.queue.manager import EnhancedCrystalQueueManager
        self.db_path = Path(db_path)
        self.args = args
        max_jobs, reserve = queue_limits(args.max_jobs, args.reserve)
        with _quiet():
            self.mgr = EnhancedCrystalQueueManager(
                d12_dir=str(self.db_path.parent), db_path=str(self.db_path),
                max_jobs=max_jobs, reserve_slots=reserve,
                max_recovery_attempts=args.max_recovery_attempts, organize_outputs=False)
        self.db = self.mgr.db
        self._engines: Dict[str, object] = {}

    def engine_for(self, calc: Dict):
        """The recovery engine the job callback would use for this calc: its
        --config, else recovery_config.yaml in its folder, else the shipped one."""
        from mace.recovery.recovery import ErrorRecoveryEngine, RECOVERY_CONFIG_NAME, PACKAGED_RECOVERY_CONFIG
        if self.args.config:
            cfg = str(Path(self.args.config).expanduser())
        else:
            local = Path(calc.get('work_dir') or '.') / RECOVERY_CONFIG_NAME
            cfg = str(local) if local.is_file() else str(PACKAGED_RECOVERY_CONFIG)
        if cfg not in self._engines:
            with _quiet():
                self._engines[cfg] = ErrorRecoveryEngine(str(self.db_path), cfg)
        return self._engines[cfg]

    def failed_rows(self) -> List[Dict]:
        rows = self.db.get_calculations_by_status('failed')
        rows.sort(key=lambda r: r.get('created_at') or '')
        want = {c for v in self.args.calc_id for c in v.split(',') if c.strip()}
        if want:
            rows = [r for r in rows if r['calc_id'] in want]
        if self.args.dir:
            folder = Path(self.args.dir).expanduser()
            rows = [r for r in rows if _inside(r.get('work_dir') or (
                str(Path(r['input_file']).parent) if r.get('input_file') else None), folder)]
        return rows

    def classify(self, calc: Dict) -> Tuple[str, str]:
        """The error as the queue manager's handle_failed_calculation decides
        it: the output (and the scratch fort.87) through the shared classifier,
        SLURM's walltime verdict over it, the recorded type when the output
        says nothing more."""
        from mace.recovery.recovery import canonical_error_type
        view = dict(calc)
        out = _output_of(calc)
        if out is not None:
            view['output_file'] = str(out)
        with _quiet():
            et, msg = self.mgr.analyze_calculation_error(view)
            timed = self.mgr._walltime_kill_evidence(view, calc.get('slurm_state') or '')
        if timed and et != 'finalrun_loop':
            et, msg = 'timeout_error', timed
        stored = calc.get('error_type')
        vague = ('no_output', 'unknown_error', 'file_error', None, '')
        if et in vague and stored not in vague:
            et, msg = stored, calc.get('error_message') or msg
        return canonical_error_type(et), msg

    def ended_at(self, calc: Dict) -> Optional[datetime]:
        """When the job ended: sacct End, else the output's last write, else
        when MACE recorded the failure."""
        jid = calc.get('slurm_job_id')
        if jid:
            with _quiet():
                self.mgr._sacct_job_state(jid)
            end = self.mgr._sacct_end.get(str(jid))
            if end:
                return end
        out = _output_of(calc)
        if out is not None:
            try:
                return datetime.fromtimestamp(out.stat().st_mtime)
            except OSError:
                pass
        try:
            return datetime.fromisoformat(calc['completed_at']) if calc.get('completed_at') else None
        except ValueError:
            return None

    def plan(self, calc: Dict, queue: Optional[List[Dict[str, str]]]) -> Dict:
        error_type, message = self.classify(calc)
        engine = self.engine_for(calc)
        cfg = engine.config.get('error_recovery', {}).get(error_type, {})
        handler = cfg.get('handler')
        recoverable = (error_type in self.mgr.RECOVERABLE_ERRORS and bool(handler)
                       and hasattr(engine, handler))
        lineage = self.mgr.get_recovery_attempt_count(calc['calc_id'])
        spent = self.mgr._error_retry_count(calc['calc_id'], error_type)
        per_max = cfg.get('max_retries')
        entry = {
            'calc_id': calc['calc_id'], 'db': str(self.db_path), 'calc_type': calc.get('calc_type'),
            'material_id': calc.get('material_id'), 'deck': calc.get('input_file'),
            'work_dir': calc.get('work_dir'), 'job_id': calc.get('slurm_job_id'),
            'workflow_id': _settings(calc).get('workflow_id'),
            'error_type': error_type, 'error_class': _class_key(error_type), 'message': message,
            'handler': handler if recoverable else None,
            'fix': (describe_fix(handler, cfg) if recoverable
                    else "no automatic recovery: " + _no_recovery_reason(error_type, message)),
            'retries': {'error': [spent, per_max], 'total': [lineage, self.args.max_recovery_attempts]},
            'ended_at': None, 'stale': False, 'blocked': None, 'held': None,
        }
        held = workflow_hold(calc, [self.db_path.parent])
        if held:
            entry['workflow_id'] = held[0]
            entry['held'] = held[1]
        if not recoverable:
            entry['state'] = 'manual'
            return entry
        if lineage >= self.args.max_recovery_attempts or (per_max is not None and spent >= int(per_max)):
            entry['state'] = 'limit'
            return entry
        blocked = (hold_reason(held) if held else None) or active_successor(self.db, calc) \
            or (queued_twin(calc, queue) if queue else None)
        if blocked:
            entry['state'], entry['blocked'] = 'blocked', blocked
            return entry
        ended = self.ended_at(calc)
        entry['ended_at'] = ended.isoformat(timespec='minutes') if ended else None
        if ended and datetime.now() - ended > timedelta(days=stale_days()):
            entry['state'], entry['stale'] = 'stale', True
        else:
            entry['state'] = 'ready'
        return entry

    # ---------------------------------------------------------- dry run
    def preview(self, calc: Dict, entry: Dict) -> Dict:
        """The fix applied to copies: the files it would write or change
        (with diffs) and the sbatch command, nothing touched."""
        engine = self.engine_for(calc)
        cfg = engine.config['error_recovery'][entry['error_type']]
        deck = Path(calc['input_file']) if calc.get('input_file') else None
        work = Path(calc.get('work_dir') or (deck.parent if deck else '.'))
        script = _resolve_script(calc)
        out = _output_of(calc)
        changes, notes, commands = [], [], []
        if entry['handler'] == 'cleanup_handler':
            for f in engine.cleanup_candidates(calc):
                changes.append({'path': str(f), 'kind': 'deleted', 'diff': ''})
        with tempfile.TemporaryDirectory(prefix=".mace-recover-preview-", dir=self._preview_parent()) as sb:
            sb = Path(sb)
            originals = {}
            for src in (deck, script, out):
                if src is not None and src.is_file():
                    dst = sb / src.name
                    shutil.copy2(src, dst)
                    originals[src.name] = src
            # the job script's record of the scratch it really ran in
            # (<submit dir>/.<JOB>.scratch): the OPT-timeout fix reads it to
            # decide on RESTART, as the real run does
            try:
                for rec in work.glob(".*.scratch"):
                    if rec.is_file():
                        shutil.copy2(rec, sb / rec.name)
            except OSError:
                pass
            before = {p.name: p.read_bytes() for p in sb.iterdir()}
            view = dict(calc, input_file=str(sb / deck.name) if deck else None,
                        job_script=str(sb / script.name) if script else None,
                        output_file=str(sb / out.name) if out else None, work_dir=str(sb),
                        error_type=entry['error_type'], error_message=entry['message'])
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                try:
                    handler = getattr(engine, entry['handler'])
                    result = handler(view, cfg) if entry['handler'] != 'cleanup_handler' else (
                        Path(view['input_file']), None)
                except Exception as e:          # the run would fail the same way
                    print(f"The fix failed: {e}")
                    result = None
            home = str(deck.parent if deck else work)
            notes = [l.replace(str(sb), home) for l in buf.getvalue().splitlines() if l.strip()]
            if isinstance(result, tuple):
                fixed_input, fixed_script = result
            else:
                fixed_input, fixed_script = result, None
            def real(name):
                name = Path(name).name
                if name in originals:
                    return originals[name]
                if name.endswith('.sh'):
                    return (script.parent if script else work) / name
                return (deck.parent if deck else work) / name
            for p in sorted(sb.iterdir()):
                text_now = p.read_bytes()
                if p.name in before and text_now == before[p.name]:
                    continue
                origin = originals.get(p.name)
                if origin is None:                     # a new file: diff it against its source
                    src = deck if p.suffix in ('.d12', '.d3') else script
                    kind = 'new'
                else:
                    src, kind = origin, 'modified'
                old = src.read_text(errors='ignore').splitlines(keepends=True) if src and src.is_file() else []
                diff = ''.join(difflib.unified_diff(
                    old, p.read_text(errors='ignore').splitlines(keepends=True),
                    fromfile=str(src) if src else '/dev/null', tofile=str(real(p.name))))
                changes.append({'path': str(real(p.name)), 'kind': kind, 'from': str(src) if src else None,
                                'diff': diff})
            if not fixed_input:
                return {'ok': False, 'changes': changes, 'notes': notes, 'commands': []}
            fixed_input = real(Path(fixed_input).name)
            fixed_script = real(Path(fixed_script).name) if fixed_script else None
        target_dir, target, template, wf = self._target(calc, fixed_input)
        if Path(target) != Path(fixed_input).resolve() and Path(target) != Path(fixed_input):
            kind = 'replaced' if Path(target).exists() else 'new'
            changes.append({'path': str(target), 'kind': kind, 'from': str(fixed_input),
                            'diff': f"(a copy of {Path(fixed_input).name}: the workflow folder's deck name)\n"})
        existing = Path(target).with_suffix('.sh')
        if fixed_script:
            commands.append(f"cd {shlex.quote(str(target_dir))} && sbatch {shlex.quote(str(fixed_script))}")
        elif existing.is_file() and not self.args.overwrite_sh:
            commands.append(f"cd {shlex.quote(str(target_dir))} && sbatch {shlex.quote(str(existing))}"
                            "   (the existing job script, reused)")
        else:
            if existing.is_file():
                changes.append({'path': str(existing), 'kind': 'replaced', 'from': template,
                                'diff': "(written again from the template: --overwrite-sh)\n"})
            commands.append(f"cd {shlex.quote(str(target_dir))} && bash {shlex.quote(str(template))} "
                            f"{shlex.quote(Path(target).stem)}   (writes {Path(target).stem}.sh)")
            commands.append(f"cd {shlex.quote(str(target_dir))} && sbatch {shlex.quote(str(existing))}")
        return {'ok': True, 'changes': changes, 'notes': notes, 'commands': commands,
                'workflow_context': wf}

    def _preview_parent(self) -> Optional[str]:
        """Where the dry run copies the files it fixes: $TMPDIR when it is
        set, else next to the database (the system /tmp may be memory, and
        an output can be large); the system default only when that folder
        cannot be written."""
        if os.environ.get('TMPDIR'):
            return None                    # tempfile's own choice is $TMPDIR
        folder = self.db_path.parent
        return str(folder) if os.access(folder, os.W_OK) else None

    def _target(self, calc: Dict, fixed_input: Path):
        """(folder, submitted deck, template, workflow?) as the worker's queue
        manager would decide them in the calculation's folder."""
        from types import SimpleNamespace
        from mace.queue.manager import EnhancedCrystalQueueManager as M
        work = Path(calc.get('work_dir') or Path(fixed_input).parent)
        exports = script_exports(calc)
        saved_env = {k: os.environ.get(k) for k in _CONTEXT_VARS}
        cwd = os.getcwd()
        try:
            for k in _CONTEXT_VARS:
                os.environ.pop(k, None)
            os.environ.update(exports)
            os.chdir(work)
            ns = SimpleNamespace(organize_outputs=False, d12_dir=work)
            ns.is_workflow_context = M._detect_workflow_context(ns)
            ns.script_paths = M._setup_script_paths(ns)
            ns.create_calculation_folder = lambda *a, **k: M.create_calculation_folder(ns, *a, **k)
            calc_dir, target = M.submission_target(ns, Path(fixed_input), calc['calc_type'],
                                                   calc['material_id'], work_dir=work)
            template = M._get_submit_script_for_calc_type(ns, calc['calc_type'])
            return calc_dir, target, template, bool(ns.is_workflow_context)
        finally:
            os.chdir(cwd)
            for k, v in saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


# ------------------------------------------------------------------ text
STATE_ACTION = {
    'ready': "resubmit",
    'stale': "ask: ended {ended} (over {days:g} days ago)",
    'limit': "skip: retry limit reached",
    'manual': "skip: needs a person",
    'blocked': "skip: {blocked}",
}


def _retries(e: Dict) -> str:
    spent, per_max = e['retries']['error']
    lin, lin_max = e['retries']['total']
    per = f"{spent}/{per_max}" if per_max is not None else f"{spent}/-"
    return f"{per}, total {lin}/{lin_max}"


def render_table(entries: List[Dict]) -> str:
    if not entries:
        return "No failed calculations to recover."
    rows = [("#", "Calculation", "Error", "Retries", "Proposed fix", "Action")]
    for i, e in enumerate(entries, 1):
        action = STATE_ACTION[e['state']].format(ended=(e.get('ended_at') or '?').replace('T', ' '),
                                                 days=stale_days(), blocked=e.get('blocked'))
        rows.append((str(i), e['calc_id'], f"{e['error_class']} ({e['error_type']})"
                     if e['error_class'] != e['error_type'] else e['error_type'],
                     _retries(e) if e['handler'] else "-", e['fix'], action))
    widths = [max(len(r[c]) for r in rows) for c in range(4)]
    lines = ["  ".join(r[c].ljust(widths[c]) for c in range(4)) + "  " + r[4] + "  [" + r[5] + "]"
             for r in rows]
    db_names = sorted({e['db'] for e in entries})
    head = "Failed calculations in " + ", ".join(db_names) + ":"
    return head + "\n" + "\n".join("  " + l for l in lines)


def render_summary(entries: List[Dict]) -> str:
    count = lambda s: sum(1 for e in entries if e['state'] == s)
    parts = [f"{count('ready')} to resubmit"]
    if count('stale'):
        parts.append(f"{count('stale')} ended over {stale_days():g} days ago (asked separately)")
    if count('limit'):
        parts.append(f"{count('limit')} at the retry limit")
    if count('manual'):
        parts.append(f"{count('manual')} without automatic recovery")
    held = sum(1 for e in entries if e['state'] == 'blocked' and e.get('held'))
    if count('blocked') - held:
        parts.append(f"{count('blocked') - held} already queued or running")
    if held:
        parts.append(f"{held} in a paused or canceled workflow (resume it first)")
    return "; ".join(parts) + "."


def render_preview(i: int, e: Dict) -> str:
    p = e.get('preview') or {}
    lines = [f"[{i}] {e['calc_id']} - {e['fix']}"]
    if e.get('message'):
        lines.append(f"    error: {e['message']}")
    for n in p.get('notes', []):
        lines.append(f"    fix: {n}")
    if not p.get('ok'):
        lines.append("    The fix produced nothing; this calculation would not be resubmitted.")
    for c in p.get('changes', []):
        verb = {'new': 'would write', 'modified': 'would change', 'replaced': 'would replace',
                'deleted': 'would delete'}[c['kind']]
        lines.append(f"    {verb} {c['path']}")
        for d in (c.get('diff') or '').splitlines():
            lines.append(f"      {d}")
    for cmd in p.get('commands', []):
        lines.append(f"    would run: {cmd}")
    if e.get('workflow_id') and e.get('held'):
        lines.append(f"    workflow {e['workflow_id']} is {e['held']}: nothing is resubmitted until it is resumed")
    elif e.get('workflow_id'):
        lines.append(f"    keeps workflow {e['workflow_id']}: the workflow goes on when the new run completes")
    if any('_recovery_' in c['path'] for c in p.get('changes', [])):
        lines.append("    (a _recovery_<date>_<time> name gets the time of the real run)")
    return "\n".join(lines)


# ------------------------------------------------------------------ asking
def ask(question: str, assume: Optional[bool]) -> bool:
    """y/N; the end of the input is No. ``assume`` answers without asking."""
    from mace.utils.piped_answers import stop_if_out_of_step
    if assume is not None:
        print(f"{question} [y/N] {'y' if assume else 'n'} (--yes)")
        return assume
    invalid = 0
    while True:
        try:
            answer = input(f"{question} [y/N] ")
        except EOFError:
            print()
            return False
        a = answer.strip().lower()
        if a in ('y', 'yes'):
            return True
        if a in ('', 'n', 'no'):
            return False
        invalid += 1
        stop_if_out_of_step(invalid, question, "yes or no")


# ------------------------------------------------------------------ the run
def _worker_env(calc: Dict) -> Dict[str, str]:
    env = dict(os.environ)
    for k in _CONTEXT_VARS:
        env.pop(k, None)
    env.update(script_exports(calc))
    env['PYTHONPATH'] = str(REPO_ROOT) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env.setdefault('MACE_NO_BANNER', '1')
    return env


def worker_argv(e: Dict, args) -> List[str]:
    argv = [sys.executable, "-m", "mace.recovery.resubmit", "--worker", "--db", e['db'],
            "--calc-id", e['calc_id'], "--error-type", e['error_type'],
            "--error-message", e['message'] or '', "--max-recovery-attempts", str(args.max_recovery_attempts)]
    if args.max_jobs is not None:            # else the worker reads the queue budget
        argv += ["--max-jobs", str(args.max_jobs), "--reserve", str(args.reserve or 0)]
    if args.config:
        argv += ["--config", str(Path(args.config).expanduser().resolve())]
    if args.overwrite_sh:
        argv.append("--overwrite-sh")
    return argv


def run_worker(e: Dict, calc: Dict, args) -> Dict:
    work = calc.get('work_dir') or (str(Path(calc['input_file']).parent) if calc.get('input_file') else '.')
    try:
        p = subprocess.run(worker_argv(e, args), cwd=work, env=_worker_env(calc), stdin=subprocess.DEVNULL,
                           capture_output=True, text=True, timeout=WORKER_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return {'status': 'failed', 'why': f"the resubmission did not finish in {WORKER_TIMEOUT_S} s", 'log': ''}
    except OSError as exc:
        return {'status': 'failed', 'why': str(exc), 'log': ''}
    result, log = None, []
    for line in p.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            try:
                result = json.loads(line[len(RESULT_PREFIX):])
            except ValueError:
                pass
        else:
            log.append(line)
    if result is None:
        tail = (p.stderr or '').strip().splitlines()[-5:]
        result = {'status': 'failed', 'why': "the resubmission stopped: " + (" | ".join(tail) or f"exit {p.returncode}")}
    result['log'] = "\n".join(log)
    return result


JSON_NEEDS_ANSWER = ("--json asks no questions: add --yes (apply the fixes and resubmit) or --dry-run (show "
                     "the changes only)")


def run_recover(args) -> int:
    json_mode = bool(getattr(args, 'json', False))
    if json_mode and not args.yes and not args.dry_run:
        # the y/N question would vanish into the JSON (the command looks hung)
        print(f"mace recover: {JSON_NEEDS_ANSWER}", file=sys.stderr)
        print(json.dumps({'ok': False, 'error': JSON_NEEDS_ANSWER}, indent=1))
        return 2
    real_stdout = sys.stdout
    buf = io.StringIO()
    if json_mode:
        sys.stdout = buf
    try:
        code, payload = _run(args, json_mode)
    finally:
        sys.stdout = real_stdout
    if json_mode:
        payload['text'] = buf.getvalue()
        print(json.dumps(payload, indent=1, default=str))
    return code


def _run(args, json_mode: bool) -> Tuple[int, Dict]:
    dbs = discover_dbs(args.db, args.dir)
    payload = {'entries': [], 'results': [], 'dry_run': bool(args.dry_run), 'stale_days': stale_days()}
    if not dbs:
        where = args.db or (f"{args.dir}" if args.dir else "this folder")
        print(f"No tracked database found ({where}); nothing to recover.")
        return 0, payload
    errors = {x.strip() for x in (args.errors or '').split(',') if x.strip()}
    queue = queue_rows()
    if queue is None:
        print("squeue is not available: the queue is checked again (and the database) before each "
              "resubmission")
    entries, calcs = [], {}
    unreadable = 0
    for db in dbs:
        try:
            planner = Planner(db, args)
            rows_ = planner.failed_rows()
        except sqlite3.OperationalError as exc:
            # a read-only database or folder: MaterialDatabase writes as it opens
            unreadable += 1
            ro = "readonly" in str(exc) or not os.access(str(db), os.W_OK) or \
                not os.access(str(Path(db).parent), os.W_OK)
            why = "it is read-only" if ro else str(exc)
            print(f"Cannot open {db} for writing ({why}); nothing to recover there.")
            payload.setdefault('unreadable', []).append({'db': str(db), 'why': why})
            continue
        for calc in rows_:
            e = planner.plan(calc, queue)
            if errors and e['error_type'] not in errors and e['error_class'] not in errors:
                continue
            e['_planner'] = planner
            calcs[(e['db'], e['calc_id'])] = calc
            entries.append(e)
    clean = lambda e: {k: v for k, v in e.items() if not k.startswith('_')}
    if unreadable and unreadable == len(dbs):
        return 1, payload
    print(render_table(entries))
    if not entries:
        return (1 if unreadable else 0), payload
    print(render_summary(entries))
    if args.dry_run:
        print()
        print("Dry run: nothing is changed or submitted.")
        for i, e in enumerate(entries, 1):
            if e['state'] in ('ready', 'stale'):
                e['preview'] = e['_planner'].preview(calcs[(e['db'], e['calc_id'])], e)
                print(render_preview(i, e))
        payload['entries'] = [clean(e) for e in entries]
        return 0, payload
    payload['entries'] = [clean(e) for e in entries]
    ready = [e for e in entries if e['state'] == 'ready']
    stale = [e for e in entries if e['state'] == 'stale']
    chosen = []
    assume = True if args.yes else None
    if ready and ask(f"Apply the fixes and resubmit {len(ready)} calculation(s)?", assume):
        chosen += ready
    if stale:
        days = stale_days()
        names = ", ".join(e['calc_id'] for e in stale)
        print(f"{len(stale)} calculation(s) ended more than {days:g} days ago ({names}); the queue manager "
              "does not resubmit those by itself.")
        if ask("Resubmit them too?", (True if args.include_stale else False) if args.yes else None):
            chosen += stale
        elif args.yes and not args.include_stale:
            print("  (left alone: add --include-stale to resubmit them with --yes)")
    if not chosen:
        if (ready or stale) and not args.yes:
            print("Nothing submitted. Run again with --yes to apply the fixes and resubmit, or with "
                  "--dry-run to see the changes first.")
        return 0, payload
    if len(chosen) > args.max_recoveries:
        print(f"Resubmitting the first {args.max_recoveries} (--max-recoveries); "
              f"{len(chosen) - args.max_recoveries} left for the next run.")
        chosen = chosen[:args.max_recoveries]
    failures = 0
    for i, e in enumerate(chosen, 1):
        calc = calcs[(e['db'], e['calc_id'])]
        print(f"[{i}/{len(chosen)}] {e['calc_id']}: {e['fix']}")
        res = run_worker(e, calc, args)
        for line in (res.get('log') or '').splitlines():
            if line.strip():
                print(f"    {line}")
        status = res.get('status')
        if status == 'submitted':
            print(f"  Resubmitted: job {res.get('job_id')} ({res.get('new_calc_id')}), "
                  f"script {res.get('job_script')}")
        elif status == 'throttled':
            left = len(chosen) - i
            print(f"  Not submitted: {res.get('why')}; {left} more left for a later run.")
            payload['results'].append(dict(res, calc_id=e['calc_id'], log=None))
            break
        else:
            if status in ('failed', 'busy'):     # busy: the queue lock timed out; run again
                failures += 1
            print(f"  Not submitted: {res.get('why')}")
        payload['results'].append(dict(res, calc_id=e['calc_id'], log=None))
    done = sum(1 for r in payload['results'] if r.get('status') == 'submitted')
    print(f"Resubmitted {done} of {len(chosen)} calculation(s).")
    return (1 if failures else 0), payload


def queue_limits(max_jobs: Optional[int], reserve: Optional[int]) -> Tuple[int, int]:
    """(max_jobs, reserve) as mace manager reads them: without --max-jobs the
    user-wide queue budget (budget + reserve, reserve); with it, N and
    --reserve (default 0), the budget capping a larger N."""
    from mace.queue import budget
    if max_jobs is not None:
        budget.set_legacy_cap(max_jobs, reserve or 0)
        return int(max_jobs), int(reserve or 0)
    if reserve is not None:
        print("Warning: --reserve is ignored without --max-jobs: the queue budget applies. Set the reserve "
              "with [queue] reserve in site.toml (mace queue status shows it), or pass --max-jobs N too.",
              file=sys.stderr)
    cfg = budget.resolve(quiet=True)
    return cfg.budget + cfg.reserve, cfg.reserve


# ------------------------------------------------------------------ worker
def _worker(argv: List[str]) -> int:
    p = argparse.ArgumentParser(prog="mace recover (worker)")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--db", required=True)
    p.add_argument("--calc-id", required=True)
    p.add_argument("--error-type", required=True)
    p.add_argument("--error-message", default="")
    p.add_argument("--config", default=None)
    p.add_argument("--overwrite-sh", action="store_true")
    p.add_argument("--max-jobs", type=int, default=None)
    p.add_argument("--reserve", type=int, default=None)
    p.add_argument("--max-recovery-attempts", type=int, default=3)
    a = p.parse_args(argv)
    from mace.queue.manager import EnhancedCrystalQueueManager
    from mace.queue.queue_lock_manager import QUEUE_STATE_LOCK
    max_jobs, reserve = queue_limits(a.max_jobs, a.reserve)
    with _quiet():
        mgr = EnhancedCrystalQueueManager(
            d12_dir=os.getcwd(), max_jobs=max_jobs, reserve_slots=reserve, db_path=a.db,
            enable_tracking=True, enable_error_recovery=True,
            max_recovery_attempts=a.max_recovery_attempts, organize_outputs=False)
        if a.config:
            from mace.recovery.recovery import ErrorRecoveryEngine
            mgr.error_recovery_engine = ErrorRecoveryEngine(mgr.db_path, a.config)
    if mgr.error_recovery_engine is None:
        return _say({'status': 'failed', 'why': "the error recovery engine did not start"})

    def locked():
        return _resubmit_locked(mgr, a)

    try:
        if mgr.lock_manager:
            res = mgr.lock_manager.with_lock(QUEUE_STATE_LOCK, locked, timeout=120)
        else:
            res = locked()
    except TimeoutError:
        res = {'status': 'busy', 'why': "another MACE process held the queue lock for 2 minutes; run "
                                        "mace recover again"}
    return _say(res)


def _say(res: Dict) -> int:
    sys.stdout.flush()
    print(RESULT_PREFIX + json.dumps(res, default=str))
    return 0


def _resubmit_locked(mgr, a) -> Dict:
    calc = mgr.db.get_calculation(a.calc_id)
    if not calc:
        return {'status': 'skipped', 'why': f"{a.calc_id} is not in {a.db}"}
    if calc.get('status') != 'failed':
        return {'status': 'skipped', 'why': f"its status is now '{calc.get('status')}' (handled meanwhile)"}
    # the workflow may have been paused or canceled since the plan was shown
    held = workflow_hold(calc, [Path(a.db).resolve().parent, os.getcwd()])
    if held:
        return {'status': 'skipped', 'why': hold_reason(held)}
    why = active_successor(mgr.db, calc)
    if why:
        return {'status': 'skipped', 'why': why}
    queue = queue_rows()
    if queue is not None:
        why = queued_twin(calc, queue)
        if why:
            return {'status': 'skipped', 'why': why}
    queued = mgr._queued_job_count()
    if queued >= mgr.max_jobs - mgr.reserve_slots:
        return {'status': 'throttled', 'why': f"Queue nearly full ({queued}/{mgr.max_jobs})"}
    out = _output_of(calc)
    if out is not None and not calc.get('output_file'):
        calc['output_file'] = str(out)
    # The class on the row, as handle_failed_calculation records it (the
    # lineage retry count reads it later); its end time stays.
    with mgr.db._get_connection() as conn:
        conn.execute("UPDATE calculations SET error_type = ?, error_message = ? WHERE calc_id = ?",
                     (a.error_type, a.error_message or None, calc['calc_id']))
    before = {r['calc_id'] for r in mgr.db.get_calculations_by_status(material_id=calc['material_id'])}
    ok = mgr.attempt_error_recovery(calc, a.error_type, a.error_message,
                                    reuse_job_script=not a.overwrite_sh)
    new = [r for r in mgr.db.get_calculations_by_status(material_id=calc['material_id'])
           if r['calc_id'] not in before]
    if not ok:
        return {'status': 'failed', 'why': "the fix or the submission failed (see above)"}
    new = new[0] if new else {}
    return {'status': 'submitted', 'new_calc_id': new.get('calc_id'), 'job_id': new.get('slurm_job_id'),
            'job_script': new.get('job_script'), 'input_file': new.get('input_file'),
            'workflow_id': _settings(new).get('workflow_id')}


if __name__ == "__main__":
    if "--worker" in sys.argv[1:]:
        sys.exit(_worker(sys.argv[1:]))
    print("mace.recovery.resubmit is run by `mace recover`.", file=sys.stderr)
    sys.exit(2)
