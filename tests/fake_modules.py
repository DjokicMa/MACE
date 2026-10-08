"""A fake environment-module system for tests: a ``module()`` bash function in
a file that ``BASH_ENV`` names, so every ``bash -c`` (and ``bash -lc``) child
has it, plus stub CRYSTAL binaries.

``FakeModules(root, spec)`` with ``spec = {name: {...}}``:
  root:  True  -> the load sets EBROOTCRYSTAL to the module's own folder
  path:  True  -> the load puts <folder>/bin on PATH (Spack-style modules)
  bins:  binaries created executable under <folder>/bin (default both)
  needs: a module that must be loaded first (an Lmod hierarchy)
  hidden: listed only by ``module -t spider``, not by ``avail``

``python=True`` (the default) also provides the job scripts' built-in Python
modules (no binaries), as on the cluster they come from; ``python=False`` is
a cluster without them.

``env(base)`` gives the environment to run MACE with: BASH_ENV, a MODULEPATH
marker, and MACE_MODULES_SPY (one line per module subprocess MACE starts).
"""
import os
from pathlib import Path

BINARIES = ("Pcrystal", "Pproperties")


BUILTIN_PYTHON = ("Python/3.11.3-GCCcore-12.3.0", "Python-bundle-PyPI/2023.06-GCCcore-12.3.0")


class FakeModules:
    def __init__(self, root, spec, python=True):
        self.root = Path(root)
        self.spec = dict(spec)
        if python:
            for name in BUILTIN_PYTHON:
                self.spec.setdefault(name, {"bins": ()})
        self.root.mkdir(parents=True, exist_ok=True)
        self.spy = self.root / "module_calls.log"
        self.spy.write_text("")
        self.folders = {}
        cases = []
        for name, s in self.spec.items():
            folder = self.root / "mods" / name.replace("/", "_")
            (folder / "bin").mkdir(parents=True, exist_ok=True)
            for b in s.get("bins", BINARIES):
                p = folder / "bin" / b
                p.write_text("#!/bin/bash\necho stub\n")
                p.chmod(0o755)
            self.folders[name] = folder
            body = []
            if s.get("needs"):
                body.append(f'case ":$LOADEDMODULES:" in *":{s["needs"]}:"*) ;; *) return 1 ;; esac')
            if s.get("root"):
                body.append(f'export EBROOTCRYSTAL="{folder}"')
            if s.get("path"):
                body.append(f'export PATH="{folder}/bin:$PATH"')
            body.append('export LOADEDMODULES="${LOADEDMODULES:+$LOADEDMODULES:}' + name + '"')
            cases.append(f'      "{name}") ' + "; ".join(body) + " ;;")
        avail = [n for n, s in self.spec.items() if not s.get("hidden")]
        allmods = list(self.spec)

        def listing(names):
            return "; ".join(f'case "{n}" in "$3"/*) echo "{n}" ;; esac' for n in names) or ":"
        self.script = self.root / "module.sh"
        self.script.write_text(
            "module() {\n"
            '  case "$1" in\n'
            "    -t)\n"
            '      case "$2" in\n'
            f"        avail) {{ echo '/fake/modulefiles:'; {listing(avail)}; }} >&2 ;;\n"
            f"        spider) {{ {listing(allmods)}; }} >&2 ;;\n"
            "      esac ;;\n"
            '    purge) unset EBROOTCRYSTAL; export LOADEDMODULES="" ;;\n'
            "    load)\n"
            "      shift\n"
            '      for m in "$@"; do\n'
            '        case "$m" in\n'
            + "\n".join(cases) + "\n"
            "          *) return 1 ;;\n"
            "        esac\n"
            "      done ;;\n"
            "  esac\n"
            "}\n")

    def env(self, base=None, **extra):
        e = dict(os.environ if base is None else base)
        e.update(BASH_ENV=str(self.script), MODULEPATH="/fake/modulefiles", MACE_MODULES_SPY=str(self.spy))
        e.pop("LMOD_CMD", None)
        e.pop("LOADEDMODULES", None)
        e.update(extra)
        return e

    def calls(self):
        return [l for l in self.spy.read_text().splitlines() if l.strip()]

    def reset_calls(self):
        self.spy.write_text("")


# fixture A of PLAN_submit D29: the preferred module is missing
FIXTURE_A = {
    "CRYSTAL/23-intel-2023b": {"root": True},
    "CRYSTAL/23-MPP-intel-2023a": {"root": True},
    "CRYSTAL/23_v1_binary": {"path": True},
    "CRYSTAL/24-intel-x": {"root": True},
    "CRYSTAL/17-x": {"root": True},
}
