"""phonopy commands MACE documents or prints are phonopy 4 syntax.

Measured with phonopy 4.7.2: setup options (-d, --dim, -f) moved to
`phonopy-init` (`phonopy --crystal -d` is an error); `phonopy` itself rejects
`--crystal`; a bare `phonopy band.conf` silently computes nothing (it needs
`--config`); `--nac` is gone (a BORN file switches NAC on, `--nonac` off).
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT

FILES = [REPO_ROOT / "README.md", REPO_ROOT / "DOCUMENTATION.md", REPO_ROOT / "mace_cli",
         REPO_ROOT / "CHANGELOG.md", *sorted((REPO_ROOT / "mace" / "phonopy_bridge").glob("*.py")),
         *sorted(p for p in (REPO_ROOT / "tests").glob("test_phonopy_*.py")
                 if p.name != Path(__file__).name)]   # this file quotes v3 lines on purpose

# `phonopy` as a command: not `mace phonopy`, not `phonopy-init`/`phonopy_bridge`.
_CMD = re.compile(r"(?<![\w./-])(?<!mace )phonopy(?![\w-])\s+(?P<args>-[^\n`]*|\S+\.conf\b)")
_V3 = [(re.compile(r"(^|\s)--crystal\b"), "--crystal belongs to phonopy-init"),
       (re.compile(r"(^|\s)(-d|--dim|-f|-c)(\s|=|$)"), "setup option belongs to phonopy-init"),
       (re.compile(r"(^|\s)--nac(?![\w-])"), "--nac is gone (BORN present = NAC)"),
       (re.compile(r"^\S+\.conf\b"), "a .conf needs --config")]


def _v3_uses(text):
    found = []
    for m in _CMD.finditer(text):
        args = m.group("args")
        if args.startswith("--config") or args.startswith("-q --config"):
            pass
        for pat, why in _V3:
            if pat.search(args):
                found.append((m.group(0).strip(), why))
    return found


def test_the_checker_catches_v3_command_lines():
    v3 = ("phonopy --crystal -d --dim 2 2 2\n`phonopy band.conf`\n"
          "phonopy --nac --config band.conf\nphonopy -f supercell-001.o\n")
    assert len(_v3_uses(v3)) >= 4
    ok = ("phonopy-init --crystal -d --dim 2 2 2\nphonopy --config band.conf\n"
          "phonopy --factor=521.47083 --config band.conf\nmace phonopy collect d --nac-from x\n"
          "phonopy -q --qpoints 0 0 0\n")
    assert _v3_uses(ok) == []


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_no_phonopy_3_command_lines(path):
    assert _v3_uses(path.read_text(errors="replace")) == []


def test_documentation_shows_the_4_7_commands():
    doc = (REPO_ROOT / "DOCUMENTATION.md").read_text()
    for cmd in ("phonopy-init --crystal -f", "phonopy --factor=521.47083 --config band.conf",
                "phonopy-init --crystal -d --dim"):
        assert cmd in doc, cmd


def _tool(name):
    p = Path(sys.executable).with_name(name)
    return str(p) if p.is_file() else shutil.which(name)


def test_documented_phonopy_init_route_works_on_a_mace_set(tmp_path):
    """phonopy-init --crystal -f and phonopy --config, as documented, on the
    set mace phonopy generate wrote (phonopy's NaCl-CRYSTAL example outputs)."""
    pytest.importorskip("phonopy")
    init, ph = _tool("phonopy-init"), _tool("phonopy")
    if not init or not ph:
        pytest.skip("phonopy command-line tools not installed next to this python")
    fix = REPO_ROOT / "tests" / "fixtures" / "phonopy_crystal" / "NaCl"
    for f in ("parent.out", "parent.d12"):
        shutil.copy(fix / f, tmp_path / f)
    env = dict(os.environ, MACE_NO_BANNER="1")
    p = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "phonopy", "generate",
                        "parent.out", "--dim", "4 4 4"], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    d = tmp_path / "phonopy_parent_444"
    outs = []
    for i in (1, 2):
        dst = d / f"parent_ph444_disp-{i:03d}.out"
        shutil.copy(fix / f"supercell-{i:03d}.o", dst)
        outs.append(dst.name)
    p = subprocess.run([init, "--crystal", "-f", *outs], cwd=d, capture_output=True,
                       text=True, timeout=600)
    assert p.returncode == 0 and (d / "FORCE_SETS").is_file(), p.stdout + p.stderr
    (d / "band.conf").write_text("BAND = 0 0 0  1/2 0 1/2\nBAND_POINTS = 3\n")
    p = subprocess.run([ph, "-q", "--factor=521.47083", "--config", "band.conf"], cwd=d,
                       capture_output=True, text=True, timeout=600)
    assert p.returncode == 0 and (d / "band.yaml").is_file(), p.stdout + p.stderr
