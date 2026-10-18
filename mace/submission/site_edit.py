"""``mace site set KEY VALUE`` / ``mace site unset KEY``: edit your own site.toml.

Only the user's site.toml (``~/.config/mace/site.toml``, or the file
``MACE_SITE_CONFIG`` names) is edited; group files are edited by hand by
whoever owns them. The edit is a small text change, so every comment, blank
line and other value stays as it was:

* set: the key's line in its table gets the new value (its trailing comment
  kept); a commented-out example of the key in that table (``# ntasks = 32``,
  as ``mace site --init`` writes them) is uncommented; otherwise the line is
  added at the end of the table, and a missing table is added after its
  neighbors. ``mem`` and ``mem_per_cpu`` of one table replace each other.
* unset: the key's line is removed, or commented out when it carries a
  comment of its own; a table that a set added and that is empty again is
  removed with it.

Every edit is checked before anything is written: the key against the
settings MACE knows, the value with the same checks ``mace submit`` makes,
and the edited file is read back to make sure that exactly this key changed
(a file this editor cannot change safely, such as an inline table, is left
alone with a message to edit it by hand).
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mace.submission import site

INT_KEYS = {"ntasks", "nodes", "cpus_per_task", "max_jobs", "reserve"}
BOOL_KEYS = {("modules", "purge"), ("resources", "auto")}
LIST_KEYS = {("slurm", "extra"), ("mpi", "env"), ("modules", "python")}
# a key -> (Site attribute, sources key) for the "now" line
ATTRS = {("slurm", "account"): ("account", "account"), ("slurm", "partition"): ("partition", "partition"),
         ("slurm", "qos"): ("qos", "qos"), ("slurm", "constraint"): ("constraint", "constraint"),
         ("slurm", "extra"): ("extra", "extra"), ("modules", "crystal"): ("crystal", "crystal"),
         ("modules", "crystal_minimum"): ("crystal_minimum", "crystal_minimum"),
         ("modules", "crystal_maximum"): ("crystal_maximum", "crystal_maximum"),
         ("modules", "python"): ("python", "python"), ("modules", "purge"): ("module_purge", "module_purge"),
         ("queue", "max_jobs"): ("max_jobs", "max_jobs"), ("queue", "reserve"): ("reserve", "reserve"),
         ("local_run", "max_time"): ("run_max_time", "run_max_time"),
         ("local_run", "gateway_message"): ("gateway_message", "gateway_message"),
         ("local_run", "module_message"): ("module_message", "module_message"),
         ("mpi", "launcher"): ("launcher", "launcher"), ("mpi", "srun_mpi"): ("srun_mpi", "srun_mpi"),
         ("mpi", "env"): ("mpi_env", "mpi_env"), ("paths", "scratch"): ("scratch", "scratch"),
         ("crystal", "bin_dir"): ("crystal_bin_dir", "crystal_bin_dir")}
ENV_OF = {"ntasks": "MACE_SLURM_NTASKS", "nodes": "MACE_SLURM_NODES", "cpus_per_task": "MACE_SLURM_CPUS_PER_TASK",
          "mem": "MACE_SLURM_MEM", "mem_per_cpu": "MACE_SLURM_MEM_PER_CPU", "walltime": "MACE_SLURM_WALLTIME",
          ("slurm", "account"): "MACE_SLURM_ACCOUNT", ("slurm", "partition"): "MACE_SLURM_PARTITION",
          ("slurm", "qos"): "MACE_SLURM_QOS", ("slurm", "constraint"): "MACE_SLURM_CONSTRAINT",
          ("slurm", "extra"): "MACE_SLURM_EXTRA", ("mpi", "launcher"): "MACE_MPI_LAUNCHER",
          ("paths", "scratch"): "MACE_SCRATCH_BASE", ("crystal", "bin_dir"): "MACE_CRYSTAL_BIN_DIR",
          ("modules", "purge"): "MACE_MODULE_PURGE", ("local_run", "max_time"): "MACE_RUN_MAX_TIME",
          ("modules", "crystal"): "MACE_CRYSTAL_MODULE"}

HEADER_RE = re.compile(r"^\s*\[(?!\[)\s*([^\]#]+?)\s*\]\s*(#.*)?$")
PART_RE = re.compile(r'\s*(?:"([^"]*)"|\'([^\']*)\'|([A-Za-z0-9_+-]+))\s*')


class EditError(ValueError):
    """The key, the value or the file cannot be edited: nothing was written."""


# ------------------------------------------------------------------ keys and values
def valid_keys() -> List[str]:
    out = []
    for section in sorted(site.KNOWN):
        for k in sorted(site.KNOWN[section]):
            if section == "resources" and (k == "properties" or k in site.RESOURCE_TYPES):
                continue
            out.append(f"{section}.{k}")
    out += [f"resources.<properties|{'|'.join(site.RESOURCE_TYPES)}>.<{'|'.join(site.TABLE_KEYS)}>"]
    return out


def parse_key(spec: str) -> Tuple[Tuple[str, ...], str]:
    """(table path, key) of 'resources.OPT.ntasks', 'slurm.account', ...;
    calculation types in any case. Raises EditError for an unknown key."""
    parts = [p.strip() for p in (spec or "").split(".")]
    hint = "keys: " + ", ".join(valid_keys())
    if len(parts) == 3 and parts[0] == "resources":
        level = parts[1].upper() if parts[1].upper() in site.RESOURCE_TYPES else parts[1]
        if level in ("properties",) + site.RESOURCE_TYPES and parts[2] in site.TABLE_KEYS:
            return ("resources", level), parts[2]
    elif len(parts) == 2 and parts[0] in site.KNOWN and parts[0] not in site.FOREIGN_TABLES:
        sect, key = parts
        if key in site.KNOWN[sect] and not (sect == "resources" and (key == "properties" or key in site.RESOURCE_TYPES)):
            return (sect,), key
    raise EditError(f"{spec!r} is not a site setting ({hint})")


def parse_value(table: Tuple[str, ...], key: str, raw: str) -> Any:
    """The TOML value of the text typed for ``key``."""
    text = raw.strip()
    if key in INT_KEYS:
        if not re.fullmatch(r"[0-9]{1,6}", text):
            raise EditError(f"{key} is a whole number, not {raw!r}")
        return int(text)
    if (table[0], key) in BOOL_KEYS:
        if text.lower() not in ("true", "false"):
            raise EditError(f"{key} is true or false, not {raw!r}")
        return text.lower() == "true"
    if text.startswith("["):
        try:
            import tomllib
            value = tomllib.loads(f"v = {text}")["v"]
        except Exception as exc:          # noqa: BLE001 - any parse failure is the user's typo
            raise EditError(f"{raw!r} is not a list (write it as [\"a\", \"b\"]): {exc}") from exc
        if not all(isinstance(v, str) for v in value):
            raise EditError(f"{raw!r}: the list holds text only")
        return value
    if (table[0], key) == ("slurm", "extra"):
        return [i.strip() for i in re.split(r",(?=\s*--)", text) if i.strip()]
    if (table[0], key) == ("modules", "python"):
        return [i.strip() for i in text.split(",") if i.strip()]
    if (table[0], key) == ("mpi", "env"):
        return [text]
    return text


def toml_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(toml_literal(v) for v in value) + "]"
    return json.dumps(str(value), ensure_ascii=False)


def table_header(table: Tuple[str, ...]) -> str:
    return "[" + ".".join(p if re.fullmatch(r"[A-Za-z0-9_-]+", p) else json.dumps(p) for p in table) + "]"


def label(table: Tuple[str, ...], key: str) -> str:
    return f"{table_header(table)} {key}"


# ------------------------------------------------------------------ the text edit
def _header_parts(line: str) -> Optional[Tuple[str, ...]]:
    m = HEADER_RE.match(line)
    if not m:
        return None
    name, parts, pos = m.group(1), [], 0
    while pos < len(name):
        pm = PART_RE.match(name, pos)
        if not pm or pm.end() == pos:
            return None
        parts.append(next(g for g in pm.groups() if g is not None))
        pos = pm.end()
        if pos < len(name):
            if name[pos] != ".":
                return None
            pos += 1
    return tuple(parts)


def _sections(lines: List[str]) -> List[Tuple[Optional[Tuple[str, ...]], int, int]]:
    """(table, header index, end index exclusive) of each table; the root
    table (before the first header) has header index -1."""
    heads = [(i, _header_parts(l)) for i, l in enumerate(lines)]
    heads = [(i, p) for i, p in heads if p is not None]
    out = []
    prev: Tuple[Optional[Tuple[str, ...]], int] = ((), -1)
    for i, parts in heads:
        out.append((prev[0], prev[1], i))
        prev = (parts, i)
    out.append((prev[0], prev[1], len(lines)))
    return out


def _key_re(key: str, commented: bool) -> "re.Pattern":
    k = rf'(?:{re.escape(key)}|"{re.escape(key)}"|\'{re.escape(key)}\')'
    return re.compile(rf"^(\s*){'#[ ]?' if commented else ''}({k})(\s*=\s*)(.*)$")


def _split_value(rest: str) -> Optional[Tuple[str, str, str]]:
    """(value, gap, comment) of the text after ``key =``; None when the value
    does not end on this line."""
    import tomllib
    for m in [*re.finditer("#", rest), None]:       # the first # after a whole value starts the comment
        end = len(rest) if m is None else m.start()
        cand = rest[:end].rstrip()
        try:
            tomllib.loads(f"v = {cand}")
        except Exception:                 # noqa: BLE001
            continue
        if m is None:
            return cand, "", ""
        return cand, rest[len(cand):end], rest[end:]
    return None


def _line_with(indent: str, key: str, eq: str, value: str, old_value: str, gap: str, comment: str) -> str:
    if not comment:
        return f"{indent}{key}{eq}{value}"
    # keep the comment's column when the new value fits
    gap = " " * max(1, len(old_value) + len(gap) - len(value)) if gap.strip() == "" else gap
    return f"{indent}{key}{eq}{value}{gap}{comment}"


def _find(lines, sections, table, key, commented=False):
    rx = _key_re(key, commented)
    for t, h, e in sections:
        if t == table:
            for i in range(h + 1, e):
                m = rx.match(lines[i])
                # a commented-out line counts only when it is an example of the key (# ntasks = 32)
                if m and (not commented or _split_value(m.group(4)) is not None):
                    return i, m
    return None, None


def _remove_line(lines: List[str], i: int, m) -> Optional[str]:
    """Remove (or comment out) the key line ``i``; the old value text."""
    split = _split_value(m.group(4))
    if split and split[2]:
        lines[i] = f"{m.group(1)}# {lines[i][len(m.group(1)):]}"
    else:
        del lines[i]
    return split[0] if split else m.group(4)


def edit_text(text: str, table: Tuple[str, ...], key: str, value: Any = None, unset: bool = False
              ) -> Tuple[str, List[str]]:
    """(new text, notes): ``text`` with ``key`` of ``table`` set to ``value``
    (or removed). Raises EditError when the line cannot be found or edited."""
    lines = text.split("\n")
    notes: List[str] = []
    sections = _sections(lines)
    if unset:
        i, m = _find(lines, sections, table, key)
        if i is None:
            raise EditError(f"{label(table, key)} is not set in this file")
        _remove_line(lines, i, m)
        _drop_empty_table(lines, table)
        return "\n".join(lines), notes
    # one memory form per table: the other one goes
    other = {"mem": "mem_per_cpu", "mem_per_cpu": "mem"}.get(key) if table[0] == "resources" else None
    if other:
        i, m = _find(lines, sections, table, other)
        if i is not None:
            _remove_line(lines, i, m)
            notes.append(f"{label(table, other)} removed (a table sets mem or mem_per_cpu, not both)")
            sections = _sections(lines)
    lit = toml_literal(value)
    i, m = _find(lines, sections, table, key)
    if i is None:
        i, m = _find(lines, sections, table, key, commented=True)
    if i is not None:
        split = _split_value(m.group(4))
        if split is None:
            raise EditError(f"the {label(table, key)} line goes on past its line; edit it by hand")
        old, gap, comment = split
        lines[i] = _line_with(m.group(1), m.group(2), m.group(3), lit, old, gap, comment)
        return "\n".join(lines), notes
    new_line = f"{key} = {lit}"
    for t, h, e in sections:
        if t == table and h >= 0:
            last = max([j for j in range(h, e) if lines[j].strip()] or [h])
            lines.insert(last + 1, new_line)
            return "\n".join(lines), notes
    # a new table: before its first sub-table, after its parent's group of tables, or at the end
    group = [(t, h, e) for t, h, e in sections if h >= 0 and len(table) > 1 and t[:1] == table[:1]]
    subs = [h for t, h, e in sections if h >= 0 and len(t) > len(table) and t[:len(table)] == table]
    if subs:
        at = min(subs)
        lines[at:at] = [table_header(table), new_line, ""]
        return "\n".join(lines), notes
    end = max(e for _, _, e in group) if group else len(lines)
    last = max([j for j in range(0, end) if lines[j].strip()] or [-1])
    block = ([""] if last >= 0 else []) + [table_header(table), new_line]
    lines[last + 1:last + 1] = block
    return "\n".join(lines), notes


def _drop_empty_table(lines: List[str], table: Tuple[str, ...]) -> None:
    """Remove ``table``'s header when nothing but blank lines is left in it
    (a table a set added), with the blank line before it."""
    for t, h, e in _sections(lines):
        if t == table and h >= 0 and all(not lines[j].strip() for j in range(h + 1, e)):
            if any(len(o) > len(table) and o[:len(table)] == table for o, oh, _ in _sections(lines) if oh >= 0):
                return
            start = h - 1 if h > 0 and not lines[h - 1].strip() else h
            del lines[start:h + 1]
            return


# ------------------------------------------------------------------ checks and the write
def _get(data: Dict[str, Any], path: Tuple[str, ...]) -> Any:
    for p in path:
        if not isinstance(data, dict) or p not in data:
            return None
        data = data[p]
    return data


def _expected(old: Dict[str, Any], table, key, value, unset) -> Dict[str, Any]:
    want = json.loads(json.dumps(old, default=str))
    node = want
    for p in table:
        node = node.setdefault(p, {})
    if unset:
        node.pop(key, None)
    else:
        node[key] = value
        other = {"mem": "mem_per_cpu", "mem_per_cpu": "mem"}.get(key) if table[0] == "resources" else None
        if other:
            node.pop(other, None)
    # an empty table a set added and an unset emptied is gone; an empty one kept is fine either way
    return want


def _prune(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: _prune(v) for k, v in d.items() if not (isinstance(v, dict) and not _prune(v))}
    return d


def _read(text: str, path: Path) -> Dict[str, Any]:
    import tomllib
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise EditError(f"{path} is not valid TOML ({exc}); fix it by hand first") from exc


def _check_value(table, key, value, path: Path) -> None:
    data: Dict[str, Any] = {}
    node = data
    for p in table:
        node = node.setdefault(p, {})
    node[key] = value
    try:
        site._apply_toml(site.Site(), data, path)
    except site.SiteConfigError as exc:
        raise EditError(str(exc).replace("; nothing was generated", "; nothing was written")) from exc


def _write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically; a symbolic link (a site.toml kept
    in a dotfiles repository) is written through and stays a link."""
    if path.is_symlink():
        path = Path(os.path.realpath(path))
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
    try:
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    except OSError as exc:
        raise _write_error(path, exc) from exc
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException as exc:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        if isinstance(exc, OSError):
            raise _write_error(path, exc) from exc
        raise


def _write_error(path: Path, exc: OSError) -> "EditError":
    """``exc`` (from writing the temporary copy beside ``path``) said about
    ``path`` itself, not the temporary file's name."""
    import errno
    if exc.errno in (errno.EACCES, errno.EPERM, errno.EROFS):
        return EditError(f"cannot write {path}: its folder is read-only for you; nothing was written")
    return EditError(f"cannot write {path}: {exc.strerror or exc}; nothing was written")


def edit(spec: str, raw: Optional[str] = None, unset: bool = False, env: Optional[Dict[str, str]] = None
         ) -> Tuple[Path, List[str]]:
    """Set (or unset) one key of the user's site.toml: (path, lines to print).
    Raises EditError (nothing written) on an unknown key, an invalid value,
    or a file this editor cannot change safely."""
    env = dict(os.environ if env is None else env)
    table, key = parse_key(spec)
    value = None if unset else parse_value(table, key, raw if raw is not None else "")
    path = site.config_path(env)
    if path is None:
        raise EditError("MACE_SITE_CONFIG=none: there is no site.toml to edit")
    out: List[str] = []
    if not path.exists():
        if unset:
            raise EditError(f"{path} does not exist: nothing to unset")
        _check_value(table, key, value, path)
        site.init_file(env)
        out.append(f"Created {path} (mace site --init: every other setting commented out)")
    elif not unset:
        _check_value(table, key, value, path)
    text = path.read_text()
    before = _read(text, path)
    if unset and _get(before, table + (key,)) is None:
        raise EditError(f"{label(table, key)} is not set in {path}")
    new_text, notes = edit_text(text, table, key, value, unset)
    try:
        after = _read(new_text, path)
    except EditError:
        after = None
    if after is None or _prune(after) != _prune(_expected(before, table, key, value, unset)):
        raise EditError(f"{path}: {label(table, key)} is written in a form this command does not edit "
                        "(an inline table or a dotted key); edit the file by hand")
    # the whole file as mace submit reads it: an error this edit makes is refused
    try:
        site._apply_toml(site.Site(), after, path)
    except site.SiteConfigError as exc:
        try:
            site._apply_toml(site.Site(), before, path)
        except site.SiteConfigError:
            out.append(f"Warning: {path} has another invalid value: {exc}")
        else:
            raise EditError(str(exc).replace("; nothing was generated", "; nothing was written")) from exc
    _write(path, new_text)
    out.append(f"{'Removed' if unset else 'Set'} {label(table, key)}"
               + ("" if unset else f" = {toml_literal(value)}") + f" in {path}")
    out += notes
    out += now_lines(table, key, env)
    return path, out


# ------------------------------------------------------------------ the effective value
def _res_line(s: "site.Site", kind: str, calc_type: Optional[str]) -> str:
    v = s.effective_resources(kind, calc_type)
    flag = "--mem" if v["mem_form"] == "mem" else "--mem-per-cpu"
    wall = s.walltime(kind, calc_type) or site.TEMPLATE_WALLTIME[kind]
    return f"--ntasks={v['ntasks']} -N {v['nodes']} --cpus-per-task={v['cpus_per_task']} {flag}={v['mem']} -t {wall}"


def now_lines(table: Tuple[str, ...], key: str, env: Dict[str, str]) -> List[str]:
    """What the job scripts get now, with every layer (environment, group files)."""
    try:
        s = site.load(env)
    except site.SiteConfigError as exc:
        return [f"Error: {exc}"]
    out = []
    if table[0] == "resources" and key == "auto":
        on = [t for t in site.RESOURCE_TYPES if s.auto(site.type_kind(t), t)]
        out.append(f"Now: size-aware suggestions on for {', '.join(on)}" if on else
                   "Now: size-aware suggestions off (the job sizes set, else the built-in ones)")
        if on:
            out.append("     (values you set win; `mace site suggest <deck>` shows a deck's suggestion)")
        return out
    if table[0] == "resources":
        level = table[1] if len(table) > 1 else ""
        if level in site.RESOURCE_TYPES:
            out.append(f"Now: {level} jobs {_res_line(s, site.type_kind(level), level)}")
        elif level == "properties":
            out.append(f"Now: properties jobs {_res_line(s, 'prop', None)}")
        else:
            out.append(f"Now: CRYSTAL jobs {_res_line(s, 'scf', None)}")
            out.append(f"     properties jobs {_res_line(s, 'prop', None)}")
        types = [t for t in site.RESOURCE_TYPES if t in s.typed(site.type_kind(t)) and t != level]
        if types and level not in site.RESOURCE_TYPES:
            out.append(f"     ([resources.<TYPE>] tables of their own win for {', '.join(types)}: mace site shows them)")
        env_name = ENV_OF.get(key)
    else:
        attr, src_key = ATTRS.get((table[0], key), (None, None))
        if (table[0], key) in (("crystal", "scf"), ("crystal", "properties")):
            kind = "scf" if key == "scf" else "prop"
            out.append(f"Now: {s.crystal_exe[kind]}  ({s.sources.get('crystal_exe.' + kind, 'built-in')})")
        elif attr:
            out.append(f"Now: {getattr(s, attr)}  ({s.sources.get(src_key, 'built-in')})")
        env_name = ENV_OF.get((table[0], key))
    if env_name and env.get(env_name):
        out.append(f"Note: {env_name}={env[env_name]} is set in this shell and wins over site.toml")
    return out
