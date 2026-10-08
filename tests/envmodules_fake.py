"""A fake environment-module system for tests: a ``module`` shell function in
a BASH_ENV file, a MODULEPATH marker, and per-module load scripts that put a
fake program on PATH. Every ``module`` call is logged, so a test can count the
subprocesses MACE started. Nothing here touches the real module system."""
import os
import stat
from pathlib import Path

MODULE_FN = r'''
module() {
  echo "$*" >> "$FAKE_MODS/calls.log"
  case "$1" in
    -t)
      case "$2" in
        avail) echo "$MODULEPATH:" >&2; [ -f "$FAKE_MODS/avail.txt" ] && cat "$FAKE_MODS/avail.txt" >&2 ;;
        spider) [ -f "$FAKE_MODS/spider.txt" ] && cat "$FAKE_MODS/spider.txt" >&2 ;;
      esac ;;
    load)
      f="$FAKE_MODS/load/$(printf '%s' "$2" | tr '/' '_')"
      [ -f "$f" ] || { echo "Lmod has detected the following error: unknown module $2" >&2; return 1; }
      . "$f" ;;
    purge) export FAKE_PURGED=1 ;;
  esac
}
'''


def _exe(path: Path, body: str) -> Path:
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def make(tmp: Path, avail=(), spider=(), programs=None, conflicts=()):
    """The fake system under tmp. ``programs``: {module: (binary name, script
    body)}; a module in ``conflicts`` loads only after ``module purge``.
    Returns the environment overrides that turn it on."""
    root = tmp / "fakemods"
    (root / "load").mkdir(parents=True, exist_ok=True)
    (root / "avail.txt").write_text("".join(f"{m}\n" for m in avail))
    if spider:
        (root / "spider.txt").write_text("".join(f"{m}\n" for m in spider))
    for mod, (binary, body) in (programs or {}).items():
        bindir = root / "opt" / mod.replace("/", "_") / "bin"
        libdir = root / "opt" / mod.replace("/", "_") / "lib"
        bindir.mkdir(parents=True, exist_ok=True)
        libdir.mkdir(parents=True, exist_ok=True)
        _exe(bindir / binary, body)
        guard = 'if [ -z "$FAKE_PURGED" ]; then echo "Lmod: conflict" >&2; return 1; fi\n' if mod in conflicts else ""
        (root / "load" / mod.replace("/", "_")).write_text(
            guard + f'export PATH="{bindir}:$PATH"\nexport LD_LIBRARY_PATH="{libdir}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"\n')
    init = root / "init.sh"
    init.write_text(MODULE_FN)
    home = tmp / "home"
    home.mkdir(exist_ok=True)
    return {"BASH_ENV": str(init), "MODULEPATH": str(root / "modulefiles"), "FAKE_MODS": str(root),
            "HOME": str(home), "XDG_CACHE_HOME": str(tmp / "cache")}


def calls(env) -> list:
    p = Path(env["FAKE_MODS"]) / "calls.log"
    return p.read_text().splitlines() if p.exists() else []


XTB_671 = 'echo "      * xtb version 6.7.1 (edcfbbe) compiled by test"\n'
XTB_661 = 'echo "      * xtb version 6.6.1 (8d0f1dd) compiled by test"\n'


def apply(monkeypatch, env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    for k in ("LMOD_CMD", "LMOD_VERSION", "MODULESHOME", "MACE_XTB", "MACE_XTB_MODULE"):
        monkeypatch.delenv(k, raising=False)
    for k in list(os.environ):
        if k.startswith("BASH_FUNC_module"):
            monkeypatch.delenv(k, raising=False)
