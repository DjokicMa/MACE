"""Every question the CLIs behind Calculation Setup can ask, read from their source.

Setup drives cif2d12, opt2d12 and opt2d3 through answer_driver, which stops at a
question no rule or known-question names. The route sweep in test_setup_prompts.py
catches that at run time for the routes it runs; this module lists every prompt call
in the source (input(), yes_no_prompt(), get_user_input(), _nav_read/_nav_int/_nav_float)
so a test can check each one has an answer rule or is named as not reachable from Setup.

Each prompt is rendered as the driver would see it: the strings printed since the
previous prompt in the same function, then the prompt itself (an f-string's fields
become "X"; yes_no_prompt adds its "[Y/n]" suffix, get_user_input its menu tail).
"""
import ast
from pathlib import Path
from typing import List, NamedTuple, Optional

REPO = Path(__file__).resolve().parents[2]

# The CLIs each Setup mode runs, with the modules their questions come from.
MODE_FILES = {
    "cif2d12": ["Crystal_d12/NewCifToD12.py", "Crystal_d12/d12_interactive.py",
                "Crystal_d12/d12_calc_basic.py", "Crystal_d12/d12_calc_freq.py", "Crystal_d12/d12_constants.py"],
    "opt2d12": ["Crystal_d12/CRYSTALOptToD12.py", "Crystal_d12/d12_interactive.py",
                "Crystal_d12/d12_calc_basic.py", "Crystal_d12/d12_calc_freq.py", "Crystal_d12/d12_constants.py"],
    "opt2d3": ["Crystal_d3/CRYSTALOptToD3.py", "Crystal_d3/d3_interactive.py"],
}
PROMPT_FUNCS = {"input", "_nav_read", "_nav_int", "_nav_float", "yes_no_prompt", "get_user_input"}
# what a helper adds after the prompt text it is given
SUFFIX = {"get_user_choice": " (default: X): "}


def _helpers(rel: str) -> set:
    """The modules' own prompt helpers: functions whose first parameter is the prompt."""
    tree = ast.parse((REPO / rel).read_text(), rel)
    return {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
            and n.args.args and n.args.args[0].arg == "prompt"}


def prompt_funcs() -> set:
    return PROMPT_FUNCS | {h for files in MODE_FILES.values() for rel in files for h in _helpers(rel)}
# print() and the CLIs' ui.* printers (ui.err writes to stderr, which the driver reads too)
PRINT_FUNCS = {"print", "warn", "info", "note", "err", "ok", "rule"}


class Prompt(NamedTuple):
    file: str
    line: int
    func: str           # the prompt helper called
    prompt: Optional[str]  # the prompt's text (None: built at run time)
    text: str           # what the driver sees: printed context + prompt


def _render(node) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(v.value if isinstance(v, ast.Constant) else "X" for v in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        a, b = _render(node.left), _render(node.right)
        return None if a is None or b is None else a + b
    return None


def _name(call: ast.Call) -> str:
    f = call.func
    return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""


def _assigned(fn) -> dict:
    """A function's string variables: name -> [(line, text)] for `name = "..."` (a prompt
    passed by name, e.g. prompt = f"LEVSHIFT ... [{a} {b}]: "; input(prompt))."""
    out = {}
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            s = _render(n.value)
            if s is not None:
                out.setdefault(n.targets[0].id, []).append((n.lineno, s))
    return out


def _arg_text(node, fn, line: int) -> Optional[str]:
    s = _render(node)
    if s is None and isinstance(node, ast.Name) and fn is not None:
        before = [t for ln, t in _assigned(fn).get(node.id, []) if ln <= line]
        s = before[-1] if before else None
    return s


def _dead_ends(tree) -> dict:
    """Call id -> the blocks it is in that end the function's flow (return, raise, continue,
    break, sys.exit): text printed there is never followed by a question outside the block."""
    out = {}
    for n in ast.walk(tree):
        for field in ("body", "orelse", "finalbody", "handlers"):
            block = getattr(n, field, None)
            if not isinstance(block, list) or not block or not isinstance(block[0], ast.AST):
                continue
            last = block[-1]
            ends = isinstance(last, (ast.Return, ast.Raise, ast.Continue, ast.Break)) or (
                isinstance(last, ast.Expr) and isinstance(last.value, ast.Call) and _name(last.value) == "exit")
            if not ends or isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
                continue
            for stmt in block:
                for c in ast.walk(stmt):
                    if isinstance(c, ast.Call):
                        out.setdefault(id(c), set()).add((id(n), field))
    return out


def prompts(rel: str) -> List[Prompt]:
    tree = ast.parse((REPO / rel).read_text(), rel)
    names = prompt_funcs()
    out: List[Prompt] = []
    funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    seen = set()
    # each call belongs to its innermost function
    owner = {}
    for fn in sorted(funcs, key=lambda n: (n.end_lineno - n.lineno)):
        for n in ast.walk(fn):
            if isinstance(n, ast.Call) and id(n) not in owner:
                owner[id(n)] = fn
    dead = _dead_ends(tree)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    by_fn = {}
    for c in calls:
        by_fn.setdefault(id(owner.get(id(c))), []).append(c)
    for group in by_fn.values():
        group.sort(key=lambda c: (c.lineno, c.col_offset))
        printed: List[tuple] = []
        for c in group:
            name = _name(c)
            if name in PRINT_FUNCS and c.args:
                s = _arg_text(c.args[0], owner.get(id(c)), c.lineno)
                if s is not None:
                    printed.append((s + "\n", dead.get(id(c), set())))
                continue
            if name not in names or (c.lineno, c.col_offset) in seen:
                continue
            seen.add((c.lineno, c.col_offset))
            fn = owner.get(id(c))
            # the helpers' own calls pass their parameter through
            if fn is not None and fn.name in names:
                printed = []
                continue
            p = _arg_text(c.args[0], fn, c.lineno) if c.args else ""
            if p is not None and name == "yes_no_prompt":
                kw = {k.arg: k.value for k in c.keywords}
                default = _render(c.args[1]) if len(c.args) > 1 else _render(kw["default"]) if "default" in kw else "yes"
                p += " [y/N] " if default == "no" else " [Y/n] "
            if p is not None and name in SUFFIX:
                p += SUFFIX[name]
            if p is not None and name == "get_user_input":
                p = "\n" + p + " (default: X):\n1: X\nEnter your choice: "
            here = dead.get(id(c), set())
            text = "".join(t for t, blocks in printed if blocks <= here) + (p or "")
            out.append(Prompt(rel, c.lineno, name, p, text))
            printed = []
    return sorted(out, key=lambda p: p.line)


def all_prompts(mode: str) -> List[Prompt]:
    return [p for rel in MODE_FILES[mode] for p in prompts(rel)]
