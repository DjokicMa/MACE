"""Every relative link in MACE's Markdown files points at something real.

A link `[text](path)`, `[text](path#anchor)`, `[text](#anchor)` or an HTML
`href="path"` / `src`/`srcset` in a tracked .md file must name a file or
folder of the repository, and its anchor must be a heading (GitHub's slug,
with -1, -2 for repeated headings) or an `<a id>` of the target. Web links are
not fetched. Code blocks and inline code are skipped: they show commands,
not links.
"""
import re
import subprocess
from pathlib import Path

from conftest import REPO_ROOT

LINK = re.compile(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
HTML = re.compile(r"""(?:href|src|srcset)="([^"]+)\"""")
FENCE = re.compile(r"^(```|~~~).*?^\1", re.M | re.S)


def _md_files():
    try:
        out = subprocess.run(["git", "ls-files", "*.md"], cwd=REPO_ROOT, capture_output=True, text=True,
                             timeout=60, check=True).stdout.split()
        return [REPO_ROOT / f for f in out]
    except (OSError, subprocess.SubprocessError):
        skip = {".git", ".planning", "node_modules", "__pycache__"}
        return [p for p in REPO_ROOT.rglob("*.md") if not skip & set(p.relative_to(REPO_ROOT).parts)]


def _slug(heading):
    h = re.sub(r"<[^>]+>", "", heading)
    h = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", h)
    h = re.sub(r"[`*]", "", h.strip().lower())
    return re.sub(r"[^\w\- ]", "", h).replace(" ", "-")


_ANCHORS = {}


def _anchors(path):
    if path not in _ANCHORS:
        text = FENCE.sub("", path.read_text(errors="replace"))
        seen, out = {}, set()
        for h in re.findall(r"^#{1,6} (.+?)\s*#*\s*$", text, re.M):
            s = _slug(h)
            n = seen.get(s, 0)
            out.add(s if n == 0 else f"{s}-{n}")
            seen[s] = n + 1
        out |= set(re.findall(r'<a (?:name|id)="([^"]+)"', text))
        _ANCHORS[path] = out
    return _ANCHORS[path]


def _broken(md):
    text = FENCE.sub("", md.read_text(errors="replace"))
    text = re.sub(r"`[^`\n]*`", "", text)
    targets = LINK.findall(text) + [t for v in HTML.findall(text) for t in v.split(",")]
    bad = []
    for target in (t.strip().split(" ")[0] for t in targets):
        if not target or re.match(r"^[a-z]+:", target, re.I):
            continue
        path, _, anchor = target.partition("#")
        dest = md if not path else (md.parent / path)
        if not dest.exists():
            bad.append(f"{target} (no such file)")
        elif anchor and dest.suffix == ".md" and anchor not in _anchors(dest.resolve()):
            bad.append(f"{target} (no such heading)")
    return bad


def test_every_markdown_link_resolves():
    files = _md_files()
    assert any(f.name == "README.md" for f in files)
    broken = {str(f.relative_to(REPO_ROOT)): b for f in files if f.exists() for b in [_broken(f)] if b}
    assert not broken, broken


def test_the_checker_sees_a_broken_link(tmp_path):
    (tmp_path / "b.md").write_text("# B\n\n## Two words\n")
    a = tmp_path / "a.md"
    a.write_text("# A\n[ok](b.md#two-words) [x](b.md#nope) [y](missing.md) [z](#a) [w](#gone)\n"
                 "```\n[in code](nowhere.md)\n```\n<img src=\"nope.png\">\n")
    assert _broken(a) == ["b.md#nope (no such heading)", "missing.md (no such file)", "#gone (no such heading)",
                          "nope.png (no such file)"]


def test_doc_links_printed_by_the_code_resolve():
    """Messages, help texts and the GUI name docs/<file>.md#<anchor>; each must exist."""
    pat = re.compile(r"(docs/[\w/.-]+\.md|[A-Z]+\.md)#([a-z0-9-]+)")
    files = [REPO_ROOT / "mace_cli", *(p for p in REPO_ROOT.joinpath("mace").rglob("*")
                                       if p.suffix in (".py", ".js", ".html", ".sh") and "vendor" not in p.parts)]
    found, bad = 0, []
    for f in files:
        for doc, anchor in pat.findall(f.read_text(errors="ignore")):
            found += 1
            target = REPO_ROOT / doc
            if not target.is_file() or anchor not in _anchors(target.resolve()):
                bad.append(f"{f.relative_to(REPO_ROOT)}: {doc}#{anchor}")
    assert found >= 5
    assert not bad, bad
