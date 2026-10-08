"""Storage and file-count quotas, from whatever the cluster offers.

Adapter chain (cached as one key, ``quota``, 10 min / 2 min):

1. ``[quota] command`` in site.toml (an argv list; ``format`` table, lfs,
   quota or json);
2. the site's own ``quota`` alias, found once a day with ``bash -lc 'alias
   quota'`` (20 s timeout; output discarded except the alias line) and then
   run directly from the cached argv, never through a shell. It is used only
   when its target is an absolute path to a file outside ``$HOME`` that the
   user cannot write, with no shell syntax; its leading ``NAME=value``
   assignments are applied only for ``LANG``, ``LC_*``, ``TERM``,
   ``COLUMNS``, ``NO_COLOR``, ``FORCE_COLOR`` and the names in ``[quota]
   alias_env_allow``. Names that change which code runs (``LD_*``,
   ``DYLD_*``, ``PYTHON*``, ``PERL5*``, ``RUBY*``, ``PATH``, ``BASH_ENV``,
   ``ENV``, ``IFS``, ``SHELLOPTS``) are refused, except an assignment of the
   empty string to ``LD_*``/``DYLD_*``/``PYTHON*`` (clearing a variable can
   only remove injected code; one site's alias clears ``LD_LIBRARY_PATH``);
3. ``lfs quota -u $USER <mount>`` for each mount of the paths whose type is
   the lfs one (NETWORK_FS_LFS);
4. ``mmlsquota -u $USER --block-size 1K <fs>`` for GPFS mounts;
5. ``quota -s -w`` (Linux disk quotas, NFS rpc.rquotad);
6. ``df -P -k`` / ``df -P -i`` on the paths: file-system wide, labeled
   "Filesystem (not your quota)", never leveled while a quota source answered.

When 1 or 2 answers, 3-5 are skipped (they report the same spaces). Levels:
warning from ``[quota] warn_percent`` (80), critical from
``critical_percent`` (90), for space and for files. Byte units are base
1024; file counts are base 1000.
"""
from __future__ import annotations

import json
import os
import re
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from mace.cluster import config as config_mod
from mace.cluster import query

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
DISCOVERY_TIMEOUT = 20.0
DISCOVERY_MAX_AGE = 86400.0
BASE_ENV_ALLOW = ("LANG", "TERM", "COLUMNS", "NO_COLOR", "FORCE_COLOR")
CLEARABLE_PREFIXES = ("LD_", "DYLD_", "PYTHON")
NETWORK_FS_LFS = "lustre"  # british-ok: "lustre" is a file system type
SHELL_META = set(";|&`$()<>\\\n*?[]{}!")


# ------------------------------------------------------------------ units
def to_bytes(text: str) -> Optional[float]:
    """``87.1G`` -> bytes (base 1024); a bare number is KiB (quota/df -k)."""
    t = (text or "").strip().rstrip("*")
    m = re.match(r"^([\d.]+)\s*([KMGTPE]?)(i?B)?$", t, re.I)
    if not m:
        return None
    unit = m.group(2).upper()
    n = float(m.group(1))
    if not unit:
        return n * 1024.0
    return n * 1024 ** ("KMGTPE".index(unit) + 1)


def to_count(text: str) -> Optional[float]:
    """``342.0K`` / ``342k`` / ``1.0M`` -> files (base 1000)."""
    t = (text or "").strip().rstrip("*")
    m = re.match(r"^([\d.]+)\s*([KMGTkmgt]?)$", t)
    if not m:
        return None
    unit = m.group(2).upper()
    return float(m.group(1)) * (1000 ** ("KMGT".index(unit) + 1) if unit else 1)


def _pct(used: Optional[float], limit: Optional[float]) -> Optional[float]:
    if used is None or not limit:
        return None
    return round(100.0 * used / limit, 1)


def _row(label, source, space_used=None, space_limit=None, files_used=None, files_limit=None,
         space_pct=None, files_pct=None, path=None, kind="quota", grace=None):
    return {"label": label, "source": source, "path": path, "kind": kind,
            "space_used_b": space_used, "space_limit_b": space_limit,
            "space_pct": space_pct if space_pct is not None else _pct(space_used, space_limit),
            "files_used": files_used, "files_limit": files_limit,
            "files_pct": files_pct if files_pct is not None else _pct(files_used, files_limit),
            "grace": grace}


# ------------------------------------------------------------------ parsers
def table_parser(text: str, source: str = "site tool") -> List[Dict[str, Any]]:
    """A ``|``- or box-drawn table with a header naming Directory and
    Space/Files Quota/Used/Left/% columns (ANSI colors stripped)."""
    lines = [ANSI.sub("", l) for l in (text or "").splitlines()]
    header, rows = None, []
    for line in lines:
        s = line.strip()
        if not s or set(s) <= set("+-=|─━┼┬┴├┤┌┐└┘│ "):
            continue
        cells = [c.strip() for c in re.split(r"[|│┃]", s)]
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]
        if header is None:
            low = [c.lower() for c in cells]
            if any("space" in c or "block" in c for c in low) and any("file" in c or "inode" in c for c in low):
                header = low
            continue
        if len(cells) != len(header):
            continue
        d = dict(zip(header, cells))

        def col(kind, what):
            for h, v in d.items():
                if kind in h and what in h:
                    return v
            return None
        sq, su = col("space", "quota") or col("block", "quota"), col("space", "used") or col("block", "used")
        sp = col("space", "%")
        fq, fu = col("file", "quota") or col("inode", "quota"), col("file", "used") or col("inode", "used")
        fp = col("file", "%")
        label = cells[0]

        def pct(v):
            try:
                return float((v or "").rstrip("%"))
            except ValueError:
                return None
        rows.append(_row(label, source, to_bytes(su) if su else None, to_bytes(sq) if sq else None,
                         to_count(fu) if fu else None, to_count(fq) if fq else None,
                         space_pct=pct(sp), files_pct=pct(fp)))
    return rows


def quota_parser(text: str) -> List[Dict[str, Any]]:
    """``quota -s -w``: one line per file system: fs space quota limit grace
    files quota limit grace (``-w`` keeps each on one line; ``*`` marks over
    quota)."""
    rows = []
    for line in (text or "").splitlines():
        f = line.split()
        if len(f) < 7 or f[0].lower() in ("filesystem", "disk"):
            continue
        fs = f[0]
        rest = f[1:]
        # grace columns may be empty: space used, quota, limit, [grace], files used, quota, limit, [grace]
        nums = [x for x in rest if re.match(r"^[\d.]+[KMGTkmgt]?\*?$", x)]
        if len(nums) < 6:
            continue
        su, sq, sl, fu, fq, fl = nums[:6]
        limit_b = to_bytes(sq) or to_bytes(sl)
        limit_f = to_count(fq) or to_count(fl)
        rows.append(_row(fs, "quota", to_bytes(su), limit_b,
                         to_count(fu), limit_f))
    return rows


def lfs_parser(text: str, mount: str = "") -> List[Dict[str, Any]]:
    """``lfs quota -u U <mount>``: kbytes quota limit grace files quota
    limit grace; ``*`` marks over quota."""
    rows = []
    lines = (text or "").splitlines()
    for i, line in enumerate(lines):
        f = line.split()
        if not f or f[0].lower() in ("disk", "filesystem"):
            continue
        if len(f) == 1 and i + 1 < len(lines):         # a long mount name wraps the values to the next line
            f = f + lines[i + 1].split()
        if len(f) < 8:
            continue
        try:
            kb, kq, kl = (float(x.rstrip("*")) for x in f[1:4])
            files, fq, fl = (float(x.rstrip("*")) for x in f[5:8])
        except ValueError:
            continue
        rows.append(_row(f[0], "lfs", kb * 1024, (kq or kl) * 1024 or None, files, (fq or fl) or None,
                         path=mount or f[0], grace=f[4] if f[4] not in ("-", "0") else None))
    return rows


def mmlsquota_parser(text: str) -> List[Dict[str, Any]]:
    """``mmlsquota -u U --block-size 1K fs``: Filesystem Fileset type KB
    quota limit in_doubt grace | files quota limit in_doubt grace."""
    rows = []
    for line in (text or "").splitlines():
        f = line.split()
        if "|" not in f or len(f) < 12 or not f[3].isdigit():
            continue
        bar = f.index("|")
        try:
            kb, kq, kl = float(f[3]), float(f[4]), float(f[5])
            files, fq, fl = float(f[bar + 1]), float(f[bar + 2]), float(f[bar + 3])
        except (ValueError, IndexError):
            continue
        rows.append(_row(f"{f[0]} {f[1]}", "mmlsquota", kb * 1024, (kq or kl) * 1024 or None, files,
                         (fq or fl) or None))
    return rows


def df_parser(text_k: str, text_i: str = "") -> List[Dict[str, Any]]:
    """``df -P -k`` and ``df -P -i`` on the same paths -> one row per mount."""
    inodes = {}
    for line in (text_i or "").splitlines()[1:]:
        f = line.split()
        if len(f) >= 6 and f[1].isdigit():
            inodes[f[-1]] = (float(f[2]), float(f[1]))
    rows, seen = [], set()
    for line in (text_k or "").splitlines()[1:]:
        f = line.split()
        if len(f) < 6 or not f[1].isdigit() or f[-1] in seen:
            continue
        seen.add(f[-1])
        used_i, total_i = inodes.get(f[-1], (None, None))
        rows.append(_row(f"Filesystem {f[-1]} (not your quota)", "filesystem", float(f[2]) * 1024,
                         float(f[1]) * 1024, used_i, total_i or None, path=f[-1], kind="filesystem"))
    return rows


def json_parser(text: str) -> List[Dict[str, Any]]:
    rows = []
    for r in json.loads(text or "[]"):
        rows.append(_row(str(r.get("label") or r.get("path") or "?"), "site command", r.get("space_used_b"),
                         r.get("space_limit_b"), r.get("files_used"), r.get("files_limit"), path=r.get("path")))
    return rows


def levels(row: Dict[str, Any], warn: int, critical: int) -> str:
    worst = "ok"
    for k in ("space_pct", "files_pct"):
        v = row.get(k)
        if v is None:
            continue
        if v >= critical:
            return "critical"
        if v >= warn:
            worst = "warning"
    return worst


# ------------------------------------------------------------------ mounts and paths
def mounts(proc_mounts: str = "/proc/mounts") -> List[Tuple[str, str]]:
    out = []
    try:
        for line in Path(proc_mounts).read_text().splitlines():
            f = line.split()
            if len(f) >= 3:
                out.append((f[1].replace("\\040", " "), f[2]))
    except OSError:
        pass
    return out


def mount_of(path: str, table: Sequence[Tuple[str, str]]) -> Optional[Tuple[str, str]]:
    best = None
    p = os.path.realpath(path)
    for mp, fstype in table:
        if p == mp or p.startswith(mp.rstrip("/") + "/") or mp == "/":
            if best is None or len(mp) > len(best[0]):
                best = (mp, fstype)
    return best


def scratch_roots(env=None) -> List[str]:
    """The user's scratch folders: the shared roots helper when present,
    else $SCRATCH, $WORK, $PSCRATCH when set and existing."""
    try:
        from mace.gui import roots          # PLAN_connect D12, when merged
        return [str(p) for p in roots.scratch_roots(env or os.environ)]
    except Exception:
        pass
    env = os.environ if env is None else env
    out = []
    for k in ("SCRATCH", "WORK", "PSCRATCH"):
        v = (env.get(k) or "").strip()
        if v and os.path.isdir(v) and v not in out:
            out.append(v)
    return out


def paths(cfg=None, served_root: Optional[str] = None) -> List[str]:
    cfg = cfg or config_mod.load()
    out = [os.path.expanduser("~")] + scratch_roots() + [os.path.expanduser(p) for p in cfg.get("quota", "paths")]
    if served_root:
        out.append(str(served_root))
    seen, res = set(), []
    for p in out:
        if p and os.path.isdir(p) and os.path.realpath(p) not in seen:
            seen.add(os.path.realpath(p))
            res.append(p)
    return res


# ------------------------------------------------------------------ alias discovery
def env_allowed(name: str, extra: Sequence[str]) -> bool:
    return name in BASE_ENV_ALLOW or name.startswith("LC_") or name in extra


def check_alias(value: str, home: Optional[str] = None, extra_allow: Sequence[str] = ()
                ) -> Tuple[Optional[List[str]], Dict[str, str], Optional[str]]:
    """(argv, env, None) for an alias MACE may run directly, else (None, {},
    why). See the module doc for the rule."""
    home = os.path.realpath(home or os.path.expanduser("~"))
    stripped = re.sub(r'="(?:[^"\\]*)"|=\'[^\']*\'', "=", value)
    if any(c in SHELL_META for c in stripped):
        return None, {}, "the quota alias uses shell syntax"
    try:
        words = shlex.split(value)
    except ValueError:
        return None, {}, "the quota alias cannot be read"
    env: Dict[str, str] = {}
    while words and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", words[0]):
        name, val = words.pop(0).split("=", 1)
        if config_mod.refused_env_name(name):
            if val == "" and name.startswith(CLEARABLE_PREFIXES):
                env[name] = ""
                continue
            return None, {}, f"the quota alias sets {name}, which may change which code runs"
        if not env_allowed(name, extra_allow):
            return None, {}, (f"The quota alias sets {name}; add it to [quota] alias_env_allow in site.toml "
                              f"if it is safe")
        env[name] = val
    if not words:
        return None, {}, "the quota alias names no program"
    target = words[0]
    if not os.path.isabs(target):
        return None, {}, "the quota alias target is not an absolute path"
    real = os.path.realpath(target)
    if not os.path.isfile(real):
        return None, {}, "the quota alias target is not a file"
    if real == home or real.startswith(home.rstrip("/") + "/"):
        return None, {}, "the quota alias target is inside your home folder"
    if os.access(real, os.W_OK):
        return None, {}, "the quota alias target is writable by you"
    return [target] + words[1:], env, None


def parse_alias_line(text: str) -> Optional[str]:
    for line in (text or "").splitlines():
        m = re.match(r"^\s*alias\s+quota=(.*)$", line.strip())
        if m:
            v = m.group(1).strip()
            try:
                parts = shlex.split(v)
                return parts[0] if len(parts) == 1 else v
            except ValueError:
                return v.strip("'")
    return None


def discover_alias(rediscover: bool = False, cfg=None) -> Dict[str, Any]:
    """The site quota alias, discovered at most once a day (cached)."""
    cfg = cfg or config_mod.load()
    f = query.cache_dir() / "quota-alias.json"
    try:
        rec = json.loads(f.read_text())
    except (OSError, ValueError):
        rec = {}
    if not rediscover and rec.get("at") and query.now() - float(rec["at"]) < DISCOVERY_MAX_AGE:
        return rec
    rec = {"at": query.now(), "argv": None, "env": {}, "note": None}
    bash = query.available("bash")
    if not bash:
        rec["note"] = "bash not found; the site quota alias was not looked for"
    else:
        proc = query.execute([bash, "-lc", "alias quota 2>/dev/null"], timeout=DISCOVERY_TIMEOUT,
                             cwd=os.path.expanduser("~"))
        if proc.returncode == 124:
            rec["note"] = f"Looking up the site quota alias timed out after {int(DISCOVERY_TIMEOUT)} s (login profile)"
        else:
            value = parse_alias_line(proc.stdout)
            if value:
                argv, env, why = check_alias(value, extra_allow=cfg.get("quota", "alias_env_allow"))
                rec.update(argv=argv, env=env, note=why, alias=value)
    try:
        query._atomic_write(f, rec)
    except OSError:
        pass
    return rec


# ------------------------------------------------------------------ the chain
def _run_rows(argv, parser, env_extra=None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    proc = query.execute(argv, env=env, timeout=30)
    if proc.returncode not in (0,) and not proc.stdout.strip():
        return [], query._first_line(proc.stderr) if proc.stderr else f"{argv[0]} failed"
    try:
        return parser(proc.stdout), None
    except Exception as exc:
        return [], f"could not read {os.path.basename(argv[0])}: {exc}"


def collect(rediscover: bool = False, served_root: Optional[str] = None, cfg=None) -> Dict[str, Any]:
    cfg = cfg or config_mod.load()
    warn, crit = cfg.get("quota", "warn_percent"), cfg.get("quota", "critical_percent")
    rows: List[Dict[str, Any]] = []
    sources: List[str] = []
    notes: List[str] = []
    cmd = cfg.get("quota", "command")
    if cmd:
        fmt = cfg.get("quota", "format")
        parser = {"table": table_parser, "lfs": lfs_parser, "quota": quota_parser, "json": json_parser}[fmt]
        r, err = _run_rows(cmd, parser)
        if r:
            rows += r
            sources.append("site command")
        elif err:
            notes.append(f"[quota] command: {err}")
    if not rows:
        alias = discover_alias(rediscover=rediscover, cfg=cfg)
        if alias.get("argv"):
            r, err = _run_rows(alias["argv"], table_parser, alias.get("env"))
            if r:
                rows += r
                sources.append("site tool")
            elif err:
                notes.append(f"Site quota tool: {err}")
        elif alias.get("note") and alias.get("alias"):
            notes.append(alias["note"])
        elif alias.get("note"):
            notes.append(alias["note"])
    ps = paths(cfg, served_root)
    table = mounts()
    if not rows:
        user = os.environ.get("USER") or ""
        done = set()
        for p in ps:
            m = mount_of(p, table)
            if not m or m[0] in done:
                continue
            done.add(m[0])
            if m[1] == NETWORK_FS_LFS and query.available("lfs"):
                r, err = _run_rows(["lfs", "quota", "-u", user, m[0]], lambda t, mp=m[0]: lfs_parser(t, mp))
                if r:
                    rows += r
                    sources.append("lfs")
            elif m[1] == "gpfs" and query.available("mmlsquota"):
                r, err = _run_rows(["mmlsquota", "-u", user, "--block-size", "1K", os.path.basename(m[0])],
                                   mmlsquota_parser)
                if r:
                    rows += r
                    sources.append("mmlsquota")
        if query.available("quota"):
            r, err = _run_rows(["quota", "-s", "-w"], quota_parser)
            if r:
                rows += r
                sources.append("quota")
    have_quota = bool(rows)
    if ps and query.available("df"):
        k = query.execute(["df", "-P", "-k"] + ps, timeout=30)
        i = query.execute(["df", "-P", "-i"] + ps, timeout=30)
        df_rows = df_parser(k.stdout, i.stdout)
        for r in df_rows:
            if have_quota:
                r["informational"] = True
        rows += df_rows
        if df_rows:
            sources.append("filesystem")
    if not have_quota:
        notes.append("No quota information found; showing filesystem usage instead.")
    for r in rows:
        if r.get("informational"):
            r["level"] = "info"
        else:
            r["level"] = levels(r, warn, crit)
    return {"rows": rows, "sources": sources, "notes": notes, "warn_percent": warn,
            "critical_percent": crit, "have_quota": have_quota, "paths": ps}


def read(refresh: bool = False, tag: Optional[str] = None, rediscover: bool = False,
         served_root: Optional[str] = None, background: bool = False) -> query.Result:
    """The cached chain result (key ``quota``, 10 min / 2 min)."""
    cfg = config_mod.load()

    def fetch():
        data = collect(rediscover=rediscover, served_root=served_root, cfg=cfg)
        return query.Proc(0, json.dumps(data), "")
    return query.run("quota", fetch, ttl=600, refresh_floor=120, refresh=refresh or rediscover, tag=tag,
                     background=background, parse=lambda p: json.loads(p.stdout))


def warnings(data: Dict[str, Any]) -> List[str]:
    """The quota warning sentences (CLI overview, status chip)."""
    out = []
    for r in (data or {}).get("rows", []):
        if r.get("level") not in ("warning", "critical"):
            continue
        if r.get("files_pct") is not None and r["files_pct"] >= data["warn_percent"]:
            used = f"{int(r['files_used']):,}" if r.get("files_used") is not None else "?"
            lim = f"{int(r['files_limit']):,}" if r.get("files_limit") else "?"
            out.append(f"{r['label']}: {r['files_pct']:g} % of your file quota is used ({used} of {lim} files). "
                       f"New jobs may fail to write output. Find what uses them: mace cluster files")
        if r.get("space_pct") is not None and r["space_pct"] >= data["warn_percent"]:
            out.append(f"{r['label']}: {r['space_pct']:g} % of your space quota is used "
                       f"({human_bytes(r.get('space_used_b'))} of {human_bytes(r.get('space_limit_b'))}).")
    return out


def human_bytes(b: Optional[float]) -> str:
    if b is None:
        return "?"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if abs(b) < 1024 or unit == "PiB":
            return f"{b:.1f} {unit}" if unit != "B" else f"{int(b)} B"
        b /= 1024.0
    return f"{b:.1f} PiB"
