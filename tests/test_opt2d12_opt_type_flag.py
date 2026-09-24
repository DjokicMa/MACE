"""An explicit --opt-type is the optimization type, whatever stdin answers.

--opt-type was honoured only by --non-interactive without --calc-type. When
the settings prompts were answered from stdin (the workflow engine's form,
and any scripted run), the parent's type or the prompt's default was
written instead.
"""
import subprocess
import sys

import pytest

from conftest import REPO_ROOT, TEST_DATA

MACE_CLI = REPO_ROOT / "mace_cli"


def _parent(tmp_path, dft_lines, after=(), grid=True):
    src = TEST_DATA / "OPT" / "1_dia_opt_rev1.out"
    if not src.exists():
        pytest.skip("test/ corpus not present (gitignored, ~12GB)")
    deck = []
    for line in src.with_suffix(".d12").read_text().splitlines():
        if line.strip() == "B3LYP-D3":
            deck += dft_lines
        elif line.strip() == "XLGRID" and not grid:
            continue
        elif line.strip() == "ENDDFT":
            deck += [line, *after]
        else:
            deck.append(line)
    (tmp_path / "parent.d12").write_text("\n".join(deck) + "\n")
    (tmp_path / "parent.out").write_text(src.read_text(errors="replace"))
    return tmp_path


def _run(cwd, *extra, stdin="", ok=True):
    cmd = [sys.executable, str(MACE_CLI), "opt2d12", "--out-file", "parent.out",
           "--d12-file", "parent.d12", *extra]
    result = subprocess.run(cmd, cwd=cwd, input=stdin, capture_output=True,
                            text=True, timeout=300)
    log = result.stdout + result.stderr
    decks = [p for p in cwd.glob("*.d12") if p.name != "parent.d12"]
    if not ok:
        return result.returncode, decks, log
    assert result.returncode == 0, log[-1500:]
    assert len(decks) == 1, (sorted(p.name for p in cwd.iterdir()), log[-1500:])
    return decks[0].read_text().splitlines(), log


def test_opt_type_flag_wins_over_answers_on_stdin(tmp_path):
    lines, _ = _run(_parent(tmp_path, ["PBE0"]), "--non-interactive", "--calc-type", "OPT",
                    "--opt-type", "ATOMONLY", stdin="\n" * 80)
    assert lines[lines.index("OPTGEOM") + 1] == "ATOMONLY"


def test_opt_type_flag_wins_in_the_interactive_flow(tmp_path):
    # "n" to the exact-settings question, 2 = OPT, then every default
    lines, _ = _run(_parent(tmp_path, ["PBE0"]), "--opt-type", "CELLONLY",
                    stdin="n\n2\n" + "\n" * 80)
    assert lines[lines.index("OPTGEOM") + 1] == "CELLONLY"
