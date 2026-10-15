"""Job scripts written by a MACE older than 1.1.3.

`mace submit` reuses an existing <job>.sh unless --overwrite-sh is given, and
a workflow builds every step's script from the workflow_scripts/ copy taken
when it was planned. Scripts from before 1.1.3 have no scratch guard - on a
node where $SCRATCH is empty CRYSTAL then reads no INPUT and stops with
"END OF DATA IN INPUT DECK" - and none of the later staging (GUESSP .f20).
They are not rewritten here, since a user may have edited them; reusing one
is only reported.
"""
import functools
import re

#: Every MACE-generated job script copies the deck to INPUT in scratch.
_INPUT_COPY = re.compile(
    r'^\s*cp\s+"?\$\{?DIR\}?/\$\{?JOB\}?\.(?:d12|d3)"?\s+"?\$\{?scratch\}?/\$\{?JOB\}?/INPUT',
    re.M)
GUARD_MARKER = "SCRATCH GUARD"


def predates_scratch_guard(text: str) -> bool:
    """True for a MACE-generated job script (or generator) without the guard."""
    return bool(text) and GUARD_MARKER not in text and bool(_INPUT_COPY.search(text))


def reused_script_warning(name: str, properties: bool = False) -> str:
    also = "" if properties else " and no GUESSP restart staging"
    return (f"  WARNING: {name} was written by an older MACE (before 1.1.3) and is reused as it is.\n"
            f"    It has no scratch guard (on a node where $SCRATCH is empty CRYSTAL stops with\n"
            f"    END OF DATA IN INPUT DECK){also}. Regenerate it with\n"
            f"    `mace submit --overwrite-sh ...` (this replaces {name}, including any edits).")


def workflow_template_warning(name: str) -> str:
    return (f"  WARNING: workflow_scripts/{name} was copied from an older MACE (before 1.1.3):\n"
            f"    it has no scratch guard (on a node where $SCRATCH is empty CRYSTAL stops with\n"
            f"    END OF DATA IN INPUT DECK) and no GUESSP restart staging. workflow_scripts/ is a\n"
            f"    snapshot taken when the workflow was planned: re-plan the workflow to pick up the\n"
            f"    job-script fixes.")


#: What a deck name cannot hold: it becomes the SLURM job name (-J, -o), the
#: job script's file names and its `export JOB=` line. The generators
#: (submitcrystal23.sh, submit_prop.sh) refuse such a name as well.
_UNUSABLE_NAME = re.compile(r"""[\s'"`$\\()<>|&;*?\[\]{}]""")


def unusable_job_name(name: str) -> bool:
    """True for a deck name (without .d12/.d3) a job script cannot use."""
    return not name or bool(_UNUSABLE_NAME.search(name))


def unusable_name_message(file_name: str) -> str:
    return (f"  Error: {file_name}: a job script cannot use this deck name (it becomes the SLURM job "
            f"name and the JOB variable of the script; no spaces or any of ' \" ` $ \\ ( ) < > | & ; * ? "
            f"[ ] {{ }}). Rename the deck; skipped.")


# --------------------------------------------- scripts MACE wrote, unchanged
#
# Since 1.1.5 the generators end every job script with a line holding the
# POSIX `cksum` (CRC and byte count) of everything above it. A script whose
# line still matches is one MACE wrote and nobody edited since: `mace submit`
# may write it again (with the resources asked for now); any other script is
# the user's and is never replaced without --overwrite-sh.

MARKER_PREFIX = "# MACE-generated; cksum of the lines above: "
_MARKER_RE = re.compile(r"\A# MACE-generated; cksum of the lines above: (\d+) (\d+)\n?\Z")
# the line the tracking queue manager adds after `export JOB=` (manager.py
# _export_db_path): MACE's own edit, not the user's
_DB_PATH_LINE = re.compile(rb'^export MACE_DB_PATH="[^"\n]*"\n', re.M)


def _crc_table():
    table = []
    for i in range(256):
        c = i << 24
        for _ in range(8):
            c = ((c << 1) ^ 0x04C11DB7) if c & 0x80000000 else (c << 1)
        table.append(c & 0xFFFFFFFF)
    return table


_CRC = _crc_table()


def posix_cksum(data: bytes) -> int:
    """The CRC `cksum` prints for ``data`` (POSIX.2)."""
    crc = 0
    for b in data:
        crc = ((crc << 8) & 0xFFFFFFFF) ^ _CRC[((crc >> 24) ^ b) & 0xFF]
    n = len(data)
    while n:
        crc = ((crc << 8) & 0xFFFFFFFF) ^ _CRC[((crc >> 24) ^ (n & 0xFF)) & 0xFF]
        n >>= 8
    return (~crc) & 0xFFFFFFFF


def generated_unmodified(path) -> bool:
    """True when ``path`` is a job script a MACE generator wrote (1.1.5 or
    later) and nobody has changed since."""
    from pathlib import Path
    try:
        data = Path(path).read_bytes()
    except OSError:
        return False
    body, sep, last = data.rstrip(b"\n").rpartition(b"\n")
    if not sep:
        return False
    m = _MARKER_RE.match(last.decode("utf-8", "replace"))
    if not m:
        return False
    body += b"\n"
    for candidate in (body, _DB_PATH_LINE.sub(b"", body, count=1)):
        if posix_cksum(candidate) == int(m.group(1)) and len(candidate) == int(m.group(2)):
            return True
    return False


# ------------------------------------------- scripts an older MACE wrote
#
# MACE 1.0.0 - the 1.1.5 preview wrote no marker. A script one of them wrote
# with its default resources, and nobody changed since, is the header below
# (the deck's name in -J, -o and `export JOB=`) and then, word for word, the
# rest of one of those generators (cksum, byte count). Such a script carries
# none of the later job-script fixes, so `mace submit` writes it again; any
# other script without the marker is treated as the user's.
_OLDER_HEADER = {
    "scf": ("#!/bin/bash --login\n#SBATCH -J {name}\n#SBATCH -o {name}-%J.o\n"
            "#SBATCH --cpus-per-task=1\n#SBATCH --ntasks=32\n#SBATCH -A mendoza_q # or general\n"
            "#SBATCH -N 1\n#SBATCH -t 7-00:00:00\n#SBATCH --mem-per-cpu=5G\n"),
    "prop": ("#!/bin/bash --login\n#SBATCH -J {name}\n#SBATCH -o {name}-%J.o\n"
             "#SBATCH --cpus-per-task=1\n#SBATCH --ntasks=28\n#SBATCH -A mendoza_q\n"
             "#SBATCH -N 1\n#SBATCH -t 2:00:00\n#SBATCH --mem=80G\n"),
}
_OLDER_BODIES = {
    (1242864924, 3147): "scf",    # 1.0.0, 1.0.5, 1.1.0, 1.1.1
    (2576790788, 11630): "scf",   # 1.1.2, 1.1.3
    (2596851560, 11713): "scf",   # 1.1.4
    (4281232069, 21974): "scf",   # 1.1.5 preview
    (3461320374, 3762): "prop",   # 1.0.0, 1.0.5, 1.1.0, 1.1.1
    (245046113, 7896): "prop",    # 1.1.2, 1.1.3, 1.1.4
    (4292517004, 11402): "prop",  # 1.1.5 preview
    (3121515522, 21926): "scf",   # 1.1.5 preview, pinned to its install
    (2722617413, 11354): "prop",  # 1.1.5 preview, pinned to its install
}
# the install the 1.1.5 preview pinned a script to (read as '')
_PIN_LINE = re.compile(rb"^MACE_SUBMIT_HOME='[^'\n]*'\n", re.M)


def older_generated_unmodified(path):
    """The kind ('scf' or 'prop') of a job script an older MACE (before
    1.1.5) wrote with its default resources and nobody changed since, else
    None."""
    from pathlib import Path
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if MARKER_PREFIX.encode() in data:
        return None
    name = path.stem
    head, sep, body = data.partition(f"export JOB={name}\n".encode())
    if not sep:
        return None
    # the line the tracking queue manager adds after `export JOB=`
    if _DB_PATH_LINE.match(body):
        body = _DB_PATH_LINE.sub(b"", body, count=1)
    body = _PIN_LINE.sub(b"MACE_SUBMIT_HOME=''\n", body, count=1)
    kind = _OLDER_BODIES.get((posix_cksum(body), len(body)))
    if kind and head == _OLDER_HEADER[kind].format(name=name).encode():
        return kind
    return None


def older_regenerated_message(name: str) -> str:
    return f"  {name}.sh was written by an older MACE; regenerating it with 1.1.5's fixes."


# ------------------------------- generated, but from a customized generator
#
# A workflow step's script comes from its workflow_scripts/ copy (the plan's
# cores and memory) and an -i script from a copy with the typed resources:
# both carry a valid marker. Written again from the plain template they
# would lose those sizes, so --walltime / node exclusion is set in them as
# they are instead.

_SBATCH = re.compile(r"^#SBATCH\b.*$", re.M)


def sbatch_lines(text: str):
    return _SBATCH.findall(text)


_PLAIN_NAME = "MACEPLAINJOBNAME"


def plain_sbatch_lines(kind: str, name: str):
    """The #SBATCH lines the shipped template (with this site's settings)
    writes for job ``name``, or None when that cannot be told."""
    lines = _plain_sbatch_lines(kind)
    return None if lines is None else [l.replace(_PLAIN_NAME, name) for l in lines]


@functools.lru_cache(maxsize=None)
def _plain_sbatch_lines(kind: str):
    name = _PLAIN_NAME
    import os
    import subprocess
    import tempfile
    from pathlib import Path
    template = Path(__file__).with_name("submitcrystal23.sh" if kind == "scf" else "submit_prop.sh")
    try:
        try:
            from mace.submission import site
        except ImportError:  # run as a script from mace/submission/
            import importlib.util
            spec = importlib.util.spec_from_file_location("mace_site", template.with_name("site.py"))
            site = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(site)
        with tempfile.TemporaryDirectory(prefix="mace_plain_") as tmp:
            gen, cleanup, _ = site.generator_for(template, kind, tmp, out=lambda *a, **k: None)
            try:
                r = subprocess.run(["bash", str(gen), name], cwd=tmp, capture_output=True,
                                   text=True, timeout=60, env=dict(os.environ))
            finally:
                cleanup()
            out = Path(tmp) / f"{name}.sh"
            if r.returncode != 0 or not out.exists():
                return None
            return tuple(sbatch_lines(out.read_text(errors="replace")))
    except Exception:
        return None


def generated_from_custom_copy(path, kind: str) -> bool:
    """True for a script MACE wrote, unchanged since, whose #SBATCH lines are
    not the ones the plain template (and site) writes for it."""
    from pathlib import Path
    if not generated_unmodified(path):
        return False
    plain = plain_sbatch_lines(kind, Path(path).stem)
    if plain is None:
        return False
    return sbatch_lines(Path(path).read_text(errors="replace")) != plain


def set_in_place(path, walltime=None, exclude=None):
    """Set the walltime and/or node exclusion of an unchanged MACE script,
    keeping everything else (its cores and memory), and stamp it again."""
    from pathlib import Path
    path = Path(path)
    data = path.read_text()
    body = data.rstrip("\n").rpartition("\n")[0] + "\n"   # without the marker
    lines = body.split("\n")
    sb = [i for i, l in enumerate(lines) if l.startswith("#SBATCH")]
    if walltime:
        t = [i for i in sb if re.match(r"#SBATCH\s+(-t\b|--time\b)", lines[i])]
        if t:
            lines[t[0]] = f"#SBATCH -t {walltime}"
        elif sb:
            lines.insert(sb[-1] + 1, f"#SBATCH -t {walltime}")
        sb = [i for i, l in enumerate(lines) if l.startswith("#SBATCH")]
    if exclude:
        x = [i for i in sb if lines[i].startswith("#SBATCH --exclude")]
        if x:
            lines[x[0]] = f"#SBATCH --exclude={exclude}"
        elif sb:
            lines.insert(sb[-1] + 1, f"#SBATCH --exclude={exclude}")
    body = "\n".join(lines)
    raw = body.encode()
    path.write_text(body + f"{MARKER_PREFIX}{posix_cksum(raw)} {len(raw)}\n")


def reuse_conflicts(data_folder, names, overwrite_sh: bool, resources_given: bool,
                    kind=None, walltime=None, exclude=None, interactive=False, out=print):
    """(names to regenerate, names whose script must not be replaced).

    An existing <name>.sh an older MACE wrote with its default resources is
    regenerated (it lacks the later job-script fixes). With resource options
    (-i, --walltime, --exclude-*): one MACE wrote and nobody changed is
    written again with the options - unless it came from a customized
    generator (a workflow plan's copy, -i): then only the walltime / node
    exclusion asked for is set in it, so its cores and memory are kept; one
    edited by hand is a conflict, since reusing it would silently ignore
    the options."""
    from pathlib import Path
    regen, conflicts, in_place = set(), [], []
    if overwrite_sh:
        return regen, conflicts
    for name in names:
        sh = Path(data_folder) / f"{name}.sh"
        if not sh.exists():
            continue
        if older_generated_unmodified(sh):
            regen.add(name)
            out(older_regenerated_message(name))
            continue
        if not resources_given:
            continue
        if generated_unmodified(sh):
            if (kind and not interactive and (walltime or exclude)
                    and generated_from_custom_copy(sh, kind)):
                in_place.append(sh)
            else:
                regen.add(name)
        else:
            conflicts.append(name)
    if not conflicts:
        for sh in in_place:
            set_in_place(sh, walltime, exclude)
            out(in_place_message(sh.stem, walltime, exclude))
    return regen, conflicts


def in_place_message(name: str, walltime=None, exclude=None) -> str:
    what = " and ".join(w for w, on in (("walltime", walltime), ("node exclusion", exclude)) if on)
    return (f"  {name}.sh keeps its cores and memory (it was written from a customized template, "
            f"e.g. a workflow plan's); only its {what} was set.")


def conflict_message(conflicts, options: str) -> str:
    shown = ", ".join(f"{n}.sh" for n in conflicts[:8]) + (" ..." if len(conflicts) > 8 else "")
    return (f"Error: {options} would not reach {len(conflicts)} job script(s) that already exist and were "
            f"edited or customized (by hand, or with other resources by an older MACE): {shown}.\n"
            f"  Nothing was submitted. Add --overwrite-sh to write them again with these options "
            f"(your edits are lost), or {_keep_advice(options)}.")


def _keep_advice(options: str) -> str:
    """How to keep the edited scripts. --progress and --array cannot use
    them at all (the plan's scripts carry the workflow context; an array's
    tasks are generated together), so dropping options is no advice there."""
    words = options.split(" / ")
    needs_new = [o for o in words if o in ("--progress", "--array")]
    resources = [o for o in words if o not in needs_new]
    if not needs_new:
        return "drop the options to submit the scripts as they are"
    keep = (f"rename the edited scripts (or submit them without {' / '.join(needs_new)}"
            + (f" and {' / '.join(resources)}" if resources else "") + ") to keep them")
    return keep
