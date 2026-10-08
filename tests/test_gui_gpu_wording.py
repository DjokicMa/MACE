"""GPU 0 is the default everywhere MACE talks about the GPU (polish A7).

The GUI first shipped with "GPU1 by default, never GPU0" (one workstation's
preference: its GPU 1 trains models). The product default is GPU 0; a
second GPU is an example (`--gpu 1`, MACE_GUI_GPU=1). This guard keeps the
old wording out of the code, the docs and the 1.1.5 release notes. Allowed:
tests, and the bench paragraph of docs/gui.md (it describes the test rig)."""
import re

from conftest import REPO_ROOT

OLD = [r"never GPU ?0", r"GPU ?1 by default", r"GPU 1, never GPU 0", r"GPU ?0 drives the display",
       r"N >= 1; GPU0"]


def _texts():
    for p in sorted((REPO_ROOT / "mace").rglob("*")):
        if p.suffix in (".py", ".js", ".html", ".css", ".md") and "vendor" not in p.parts and "__pycache__" not in p.parts:
            yield p, p.read_text(errors="ignore")
    for p in sorted((REPO_ROOT / "docs").rglob("*.md")):
        text = p.read_text(errors="ignore")
        if p.name == "gui.md":
            # the bench section describes the opt-in GPU1 test rig
            text = re.sub(r"MACE_GUI_BENCH_GPU1.*?(?=\n#)", "", text, flags=re.S)
        yield p, text
    for name in ("README.md", "INSTALLATION.md", "mace_cli"):
        yield REPO_ROOT / name, (REPO_ROOT / name).read_text(errors="ignore")
    cl = (REPO_ROOT / "CHANGELOG.md").read_text(errors="ignore")
    m = re.search(r"^## \[1\.1\.5\].*?(?=^## \[)", cl, flags=re.S | re.M)
    if m:
        yield REPO_ROOT / "CHANGELOG.md", m.group(0)


def test_no_old_gpu_default_wording():
    hits = []
    for p, text in _texts():
        for pat in OLD:
            for m in re.finditer(pat, text):
                line = text[:m.start()].count("\n") + 1
                hits.append(f"{p.relative_to(REPO_ROOT)}:{line}: {m.group(0)}")
    assert not hits, hits
