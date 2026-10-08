"""Every localStorage use in the GUI has a decision (polish P0): it is the
profile's (profile.js, settings.js, docprefs.js, the drafts module), or its
key is on the LOCAL_ONLY list in profile.js (things that belong to one
browser: drafts, caches, the overwrite question). A new localStorage key
without a decision fails here, including keys passed through a wrapper
function (app.js `store`, builder_util's `load`/`store`), whose call sites
must name a classified key (a literal, or `// ls-key: <prefix>` on the line).
"""
import re
from pathlib import Path

from conftest import REPO_ROOT

STATIC = REPO_ROOT / "mace" / "gui" / "static"
OWNERS = {"js/profile.js", "js/settings.js", "js/docprefs.js", "js/editor/drafts.js"}
# functions that take the key as a parameter: (file, name)
WRAPPERS = {("app.js", "store"), ("js/lazy/builder_util.js", "load"), ("js/lazy/builder_util.js", "store")}

LS_CALL = re.compile(r"localStorage\.(?:getItem|setItem|removeItem)\(\s*([^,)]+)")
LS_ANY = re.compile(r"localStorage\.(?:getItem|setItem|removeItem|key|clear)\b")


def local_only():
    text = (STATIC / "js" / "profile.js").read_text()
    block = text[text.index("export const LOCAL_ONLY"):text.index("];", text.index("export const LOCAL_ONLY"))]
    return re.findall(r'"([^"]+)"', block)


def sources():
    out = {}
    for p in sorted(STATIC.rglob("*.js")):
        rel = p.relative_to(STATIC).as_posix()
        if rel.startswith("vendor/"):
            continue
        out[rel] = p.read_text()
    return out


def prefix_of(expr, text):
    """The fixed start of a key expression: a literal, a template literal's
    head, or a constant / key-builder defined in the same file; None if unknown."""
    expr = expr.strip()
    m = re.match(r'^["\']([^"\']*)["\']', expr)
    if m:
        return m.group(1)
    m = re.match(r"^`([^`$]*)", expr)
    if m:
        return m.group(1)
    name = re.match(r"^([A-Za-z_$][\w$]*)", expr)
    if not name:
        return None
    n = name.group(1)
    # const NAME = "..." | `...` ; or const NAME = (...) => `...` ; or function NAME(...) { return `...` }
    for pat in (rf"(?:const|let|var)\s+{re.escape(n)}\s*=\s*[\"'`]([^\"'`$]*)",
                rf"(?:const|let|var)\s+{re.escape(n)}\s*=\s*\([^)]*\)\s*=>\s*[\"'`]([^\"'`$]*)",
                rf"(?:const|let|var)\s+{re.escape(n)}\s*=\s*\([^)]*\)\s*=>\s*\(?\s*[\"'`]([^\"'`$]*)",
                rf"function\s+{re.escape(n)}\s*\([^)]*\)\s*\{{\s*return\s+[\"'`]([^\"'`$]*)"):
        m = re.search(pat, text)
        if m:
            return m.group(1)
    return None


def wrappers_in(rel, text):
    """Functions whose localStorage call takes one of their own parameters as the key."""
    found = set()
    for m in re.finditer(r"(?:function\s+([A-Za-z_$][\w$]*)\s*\(([^)]*)\)|(?:const|let)\s+([A-Za-z_$][\w$]*)\s*=\s*\(([^)]*)\)\s*=>)", text):
        name = m.group(1) or m.group(3)
        params = [p.strip().split("=")[0].strip() for p in (m.group(2) or m.group(4) or "").split(",") if p.strip()]
        body = text[m.end():m.end() + 400]
        for c in LS_CALL.finditer(body):
            if c.group(1).strip() in params:
                found.add((rel, name))
    return found


def check(files, classified):
    """[problems] for a {rel: text} set."""
    def ok(pre):
        return bool(pre) and any(pre.startswith(c) or (c.startswith(pre) and len(pre) >= 9) for c in classified)

    problems = []
    for rel, text in files.items():
        lines = text.splitlines()
        wr = wrappers_in(rel, text)
        if rel in OWNERS:
            continue
        for w in sorted(wr - WRAPPERS):
            problems.append(f"{rel}: {w[1]}() passes its parameter to localStorage: list it in WRAPPERS and classify its callers")
        for i, line in enumerate(lines, 1):
            if LS_CALL.search(line) and (rel, _enclosing(lines, i)) in wr:
                continue                                   # a wrapper's own body: its callers are checked below
            for c in LS_CALL.finditer(line):
                ann = re.search(r"//\s*ls-key:\s*(\S+)", line)
                pre = ann.group(1) if ann else prefix_of(c.group(1), text)
                if not ok(pre):
                    problems.append(f"{rel}:{i}: localStorage key {c.group(1).strip()!r} has no decision (profile.js LOCAL_ONLY)")
        # the call sites of every wrapper (in its own file, or through window.MACE)
        for (wf, name) in WRAPPERS:
            if wf == rel:
                pat = rf"(?<![\w$.]){re.escape(name)}\(\s*([^,)]+)"
            else:
                pat = rf"(?:M\(\)\.|MACE\.){re.escape(name)}\(\s*([^,)]+)"
            for m in re.finditer(pat, text):
                ln = text[:m.start()].count("\n")
                src = lines[ln]
                if re.search(rf"function\s+{re.escape(name)}\s*\(|(?:const|let)\s+{re.escape(name)}\s*=", src):
                    continue
                ann = re.search(r"//\s*ls-key:\s*(\S+)", src)
                pre = ann.group(1) if ann else prefix_of(m.group(1), text)
                if not ok(pre):
                    problems.append(f"{rel}:{ln + 1}: {name}({m.group(1).strip()}) stores a key with no decision")
    return problems


def _enclosing(lines, i):
    """The function (declaration or arrow) that the 1-based line i is in, looking back a few lines."""
    for k in range(i - 1, max(-1, i - 15), -1):
        m = re.search(r"(?:function\s+([A-Za-z_$][\w$]*)\s*\(|(?:const|let)\s+([A-Za-z_$][\w$]*)\s*=\s*\([^)]*\)\s*=>)", lines[k])
        if m:
            return m.group(1) or m.group(2)
    return ""


def test_every_localstorage_use_has_a_decision():
    problems = check(sources(), local_only())
    assert not problems, "\n".join(problems)


def test_the_guard_catches_a_new_key_and_a_new_wrapper():
    fixture = {
        "js/new_feature.js": "function keep(k, v) { localStorage.setItem(k, v); }\n"
                             "const X = () => localStorage.getItem('mace.gui.brandNew.v1');\n"
                             "keep(someVar, 1);\n",
    }
    problems = check(fixture, local_only())
    assert any("keep() passes its parameter" in p for p in problems), problems
    assert any("mace.gui.brandNew.v1" in p for p in problems), problems
    ok = {"js/x.js": "const Y = () => localStorage.getItem('mace.gui.draft.abc');\n"}
    assert not check(ok, local_only())
