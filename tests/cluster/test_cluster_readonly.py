"""F17 and F14-b: the cluster code is read-only and names no site.

* No command list in mace/cluster/ or the GUI's cluster route runs scancel,
  sbatch without --test-only, scontrol update/hold/release/requeue or a
  sacctmgr change, and no code deletes, moves or renames a file.
* No site literal (a university, its storage or node names, a user name)
  appears in the cluster code, the GUI's cluster files or these tests.
"""
import ast
import re
from pathlib import Path

from cluster_fixtures import own_cache  # noqa: F401  (autouse)
from cluster_fixtures import REPO_ROOT

PKG = REPO_ROOT / "mace" / "cluster"
GUI_FILES = [REPO_ROOT / "mace" / "gui" / "cluster_api.py"] + \
    sorted((REPO_ROOT / "mace" / "gui" / "static").rglob("cluster*.js")) + \
    sorted((REPO_ROOT / "mace" / "gui" / "static").rglob("cluster*.css"))
SITE = re.compile(r"mendoza|msu|icer|djokicma|amd20|dev-amd|dev-intel|gs21|ufs18|ffs24", re.I)
CHANGES = {"scontrol": {"update", "hold", "release", "requeue", "suspend", "resume", "create", "delete"},
           "sacctmgr": {"add", "modify", "delete", "remove", "create", "update", "archive", "load"}}


def _py_files():
    out = sorted(PKG.glob("*.py"))
    out += [p for p in GUI_FILES if p.suffix == ".py" and p.exists()]
    return out


def _lists(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.List, ast.Tuple)) and node.elts:
            words = [e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
            if words:
                yield node, words


def test_no_state_changing_command():
    bad = []
    for f in _py_files():
        tree = ast.parse(f.read_text())
        for node, words in _lists(tree):
            head = words[0]
            if head == "scancel":
                bad.append(f"{f.name}:{node.lineno}: scancel")
            if head == "sbatch" and "--test-only" not in words:
                bad.append(f"{f.name}:{node.lineno}: sbatch without --test-only")
            if head in CHANGES and set(words[1:]) & CHANGES[head]:
                bad.append(f"{f.name}:{node.lineno}: {head} {sorted(set(words[1:]) & CHANGES[head])}")
        for m in re.finditer(r"['\"]scancel['\"]", f.read_text()):
            bad.append(f"{f.name}: names scancel")
    assert not bad, bad


def test_no_file_changes():
    bad = []
    for f in _py_files():
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                if name in ("unlink", "rmtree", "remove", "rename", "rmdir", "removedirs"):
                    # the cache's own temporary file and a scan's temporary result are the only exceptions
                    if f.name in ("query.py", "cli.py", "cluster_api.py") and name in ("unlink",):
                        continue
                    bad.append(f"{f.name}:{node.lineno}: {name}()")
    assert not bad, bad


def test_no_site_literals():
    files = sorted(PKG.glob("*.py")) + [p for p in GUI_FILES if p.exists()] + \
        sorted((REPO_ROOT / "tests" / "cluster").glob("*.py"))
    files = [f for f in files if f.name != "test_cluster_readonly.py"]
    hits = []
    for f in files:
        for n, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
            if SITE.search(line):
                hits.append(f"{f.relative_to(REPO_ROOT)}:{n}: {line.strip()[:80]}")
    assert not hits, hits


def test_stdlib_only_at_module_level():
    """Every module of the package imports only the standard library and
    mace's stdlib-only helpers at module level."""
    allowed_mace = {"mace.cluster", "mace.queue.slurm_ids", "mace.recovery.slurm_limits",
                    "mace.submission.job_array", "mace.utils.calc_detection", "mace.completion_checker"}
    import sys
    std = set(sys.stdlib_module_names)
    bad = []
    for f in sorted(PKG.glob("*.py")):
        tree = ast.parse(f.read_text())
        for node in tree.body:
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module]
            for n in names:
                top = n.split(".")[0]
                if top == "__future__" or top in std:
                    continue
                if any(n == a or n.startswith(a + ".") for a in allowed_mace):
                    continue
                bad.append(f"{f.name}: {n}")
    assert not bad, bad
