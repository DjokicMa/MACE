"""A browser test that waits for the structure viewer to draw atoms needs ase
on the server (it reads the structures with it). Without ase (CI's 3.11
packages) such a test must skip, not time out: each one is guarded by a
module-level ``pytest.importorskip("ase")``, ``@H.needs_ase``, or an
``importorskip("ase")`` in its body."""
import ast
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRAWS = "viewport.displayed()"


def _is_ase_skip(node):
    return (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "importorskip"
            and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "ase")


def _guarded_module(tree):
    return any(isinstance(st, ast.Expr) and _is_ase_skip(st.value) for st in tree.body)


def _guarded_function(fn):
    if any("needs_ase" in ast.unparse(d) for d in fn.decorator_list):
        return True
    return any(_is_ase_skip(n) for n in ast.walk(fn))


def test_every_test_that_waits_for_drawn_atoms_skips_without_ase():
    unguarded = []
    for path in sorted(HERE.glob("test_*.py")):
        if path.name == Path(__file__).name:
            continue
        src = path.read_text()
        if DRAWS not in src:
            continue
        tree = ast.parse(src)
        if _guarded_module(tree):
            continue
        for fn in tree.body:
            if isinstance(fn, ast.FunctionDef) and fn.name.startswith("test_") \
                    and DRAWS in ast.get_source_segment(src, fn) and not _guarded_function(fn):
                unguarded.append(f"{path.name}::{fn.name}")
    assert not unguarded, unguarded


def test_the_guard_sees_an_unguarded_test():
    tree = ast.parse("import pytest\n\ndef test_x(app):\n    wait('viewport.displayed() > 0')\n")
    fn = tree.body[1]
    assert not _guarded_module(tree) and not _guarded_function(fn)
    tree = ast.parse("import pytest\n\n@H.needs_ase\ndef test_x(app):\n    pass\n")
    assert _guarded_function(tree.body[1])
