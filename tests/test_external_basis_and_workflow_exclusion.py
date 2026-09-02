"""Two production failures reported against a 576-structure campaign.

Both were found by reading real artefacts, not by reasoning about the code:
the basis truncation was measured over the 106 EXTERNAL decks in ``test/``,
and the missing ``--exclude`` was reproduced by running the executor's own
script regeneration over the shipped template.
"""
import re
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT, TEST_DATA

sys.path.insert(0, str(REPO_ROOT / "Crystal_d12"))

from d12_parsers import CrystalInputParser  # noqa: E402


# ============================================================
# EXTERNAL basis: the "99 0" terminator is a record, not a substring
# ============================================================

# A Pb block with its ECP. The first ECP term line is the real one from the
# corpus - "12.296303 281.285499 0" contains the substring "99 0" (...499 0),
# which a substring match reads as the end of the whole basis section.
_PB_DECK = """\
CRYSTAL
0 0 0
221
4.0
1
82 0.0 0.0 0.0
END
282 12
INPUT
22. 0 2 4 4 2 2
 12.296303 281.285499 0
  8.632634  62.520217 0
0 0 3 2. 1.
  1.0 1.0
8 4
0 0 2 2. 1.
  2.0 1.0
99 0
END
DFT
EXCHANGE
BECKE
END
"""


def _basis_of(text, tmp_path):
    deck = tmp_path / "t.d12"
    deck.write_text(text)
    return CrystalInputParser(str(deck)).parse()


def test_ecp_coefficients_are_not_mistaken_for_the_terminator(tmp_path):
    """The exact failure behind ERROR **** INPBAS **** FORMAT ERROR."""
    data = _basis_of(_PB_DECK, tmp_path)

    assert data["basis_set_type"] == "EXTERNAL"
    basis = data["external_basis_data"]

    # Truncation used to stop here, keeping only the three lines above it.
    assert " 12.296303 281.285499 0".strip() in basis, (
        "the Pb ECP line must be part of the basis, not read as its end")
    # Everything after Pb has to survive too - oxygen followed lead.
    assert "8 4" in basis, "elements after Pb were dropped with the terminator"
    # And the terminator itself is never basis data.
    assert "99 0" not in basis


def test_terminator_still_recognised_when_it_is_the_whole_record(tmp_path):
    """The fix must not stop finding a legitimate terminator."""
    data = _basis_of(_PB_DECK, tmp_path)
    assert data["basis_set_type"] == "EXTERNAL"
    assert data["external_basis_data"], "basis block came back empty"


@pytest.mark.parametrize("spaced", ["99 0", "  99   0  ", "99  0"])
def test_terminator_tolerates_whitespace(tmp_path, spaced):
    data = _basis_of(_PB_DECK.replace("99 0\nEND", f"{spaced}\nEND"), tmp_path)
    assert data["basis_set_type"] == "EXTERNAL"
    assert "8 4" in data["external_basis_data"]


def test_every_real_external_deck_round_trips_its_basis_block():
    """Measured over the real corpus: 13 of 106 truncated before the fix."""
    if not TEST_DATA.exists():
        pytest.skip("test/ data corpus not present (gitignored, ~12GB)")

    decks = []
    for f in TEST_DATA.rglob("*.d12"):
        txt = f.read_text(errors="ignore")
        if re.search(r"^\s*99\s+0\s*$", txt, re.M) and "BASISSET" not in txt:
            decks.append(f)
    if not decks:
        pytest.skip("no EXTERNAL-basis decks in the corpus")

    bad = []
    for f in decks:
        lines = f.read_text(errors="ignore").split("\n")
        end = next(i for i, l in enumerate(lines) if re.match(r"^\s*99\s+0\s*$", l))
        geo = max(j for j in range(end) if lines[j].strip() == "END")
        want = [l.strip() for l in lines[geo + 1:end] if l.strip()]
        got = CrystalInputParser(str(f)).parse().get("external_basis_data", [])
        if got != want:
            bad.append((f.name, len(got), len(want)))

    assert not bad, f"{len(bad)} of {len(decks)} EXTERNAL decks mis-parsed: {bad[:3]}"


# ============================================================
# The workflow executor must honour a configured node exclusion
# ============================================================

TEMPLATE = REPO_ROOT / "mace" / "submission" / "submitcrystal23.sh"


def _regenerate(script_config, tmp_path):
    from mace.workflow.executor import WorkflowExecutor

    ex = WorkflowExecutor.__new__(WorkflowExecutor)   # skip __init__ side effects
    ex.work_dir = tmp_path
    return ex.apply_script_customizations(TEMPLATE.read_text(), script_config)


_RESOURCES = {"ntasks": 32, "nodes": 1, "walltime": "7-00:00:00",
              "account": "mendoza_q", "memory_per_cpu": "5G"}


def test_configured_exclusion_reaches_the_generated_script(tmp_path):
    """Phase 0 regenerates every script and used to drop node_exclusion."""
    out = _regenerate(
        {"workflow_id": "w", "node_exclusion": "agg-[011-012],amr-[163,178-179]",
         "resources": dict(_RESOURCES)}, tmp_path)

    live = [l for l in out.split("\n")
            if l.strip().startswith("echo '#SBATCH --exclude=")]
    assert live, "the configured exclusion never reached the generated script"
    assert "agg-[011-012],amr-[163,178-179]" in live[0]


def test_no_exclusion_configured_leaves_the_template_alone(tmp_path):
    """Absent an explicit choice, nothing is added - no new default."""
    out = _regenerate({"workflow_id": "w", "resources": dict(_RESOURCES)}, tmp_path)
    assert not [l for l in out.split("\n")
                if l.strip().startswith("echo '#SBATCH --exclude=")]


def test_generated_script_emits_a_real_sbatch_directive(tmp_path, monkeypatch):
    """The template is a GENERATOR - assert on what it writes, not on itself."""
    import subprocess

    out = _regenerate(
        {"workflow_id": "w", "node_exclusion": "amr-[163,178-179]",
         "resources": dict(_RESOURCES)}, tmp_path)
    gen = tmp_path / "gen.sh"
    gen.write_text(out)
    subprocess.run(["bash", str(gen), "myjob"], cwd=tmp_path,
                   capture_output=True, timeout=60)

    job = (tmp_path / "myjob.sh").read_text()
    directives = [l for l in job.split("\n") if l.startswith("#SBATCH --exclude=")]
    assert directives == ["#SBATCH --exclude=amr-[163,178-179]"], job
