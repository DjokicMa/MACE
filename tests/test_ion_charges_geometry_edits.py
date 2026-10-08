"""Which geometry records `mace ions` (and the CHEMOD carry path) refuse.

CHEMOD names atoms by their position in the list CRYSTAL builds. A record
after the atom list that adds, removes or reorders atoms (ATOMORDE, manual
p.41, reorders them after the geometry is read) makes the deck's labels point
at other atoms, so the deck is not written. ion_charges used a private list of
17 records that missed ATOMORDE and the cluster/nanorod/Wulff builders and
refused PRIMITIV (a properties keyword); it now reads the one table in
mace.preopt.deckscan, minus the records that keep every atom in place.
"""
import os
import subprocess
import sys

import pytest

from conftest import REPO_ROOT
import ion_charges as ic
from ion_charges import strip_charge_records
from mace.preopt import deckscan

MACE_CLI = REPO_ROOT / "mace_cli"

ATOMS = """8   0.5693 0.5693 0.5693
3   0.1000 0.1000 0.1000
17  0.5000 0.5000 0.5000
8   0.4307 0.4307 0.5693
8   0.4307 0.5693 0.4307
8   0.5693 0.4307 0.4307
"""


def _deck(extra=""):
    return ("LiClO4 interleaved P1\nCRYSTAL\n0 0 0\n1\n12.0 12.0 12.0 90.0 90.0 90.0\n6\n" + ATOMS + extra
            + "END\nBASISSET\nPOB-TZVP-REV2\nDFT\nPBE0\nENDDFT\nSHRINK\n2 2\nEND\n")


def _run(args, cwd):
    env = dict(os.environ)
    for k in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH"):
        env.pop(k, None)
    return subprocess.run([sys.executable, str(MACE_CLI)] + args, cwd=cwd, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=300, env=env)


def _ions(tmp_path, name, text):
    (tmp_path / f"{name}.d12").write_text(text)
    run = _run(["ions", f"{name}.d12", "--ion-charges", "auto", "--chemod", "yes", "--non-interactive",
                "-o", "out"], tmp_path)
    out = tmp_path / "out" / f"{name}.d12"
    return run, (out.read_text() if out.exists() else None)


def test_atomorde_deck_is_not_written(tmp_path):
    run, written = _ions(tmp_path, "atomorde", _deck("ATOMORDE\n"))
    assert written is None
    assert run.returncode != 0
    msg = " ".join((run.stdout + run.stderr).split())
    assert "atomorde.d12: NOT written" in msg
    assert "edits the structure (ATOMORDE) after the atom list" in msg


def test_elastic_deck_is_still_written_with_only_the_charge_records(tmp_path):
    text = _deck("ELASTIC\n-1\n0.01 0 0\n0 0 0\n0 0 0\n")
    run, written = _ions(tmp_path, "elastic", text)
    assert run.returncode == 0, run.stderr
    assert "CHEMOD" in written
    assert strip_charge_records(written) == text


def test_plain_deck_bytes_are_the_deck_plus_the_charge_records(tmp_path):
    text = _deck()
    run, written = _ions(tmp_path, "plain", text)
    assert run.returncode == 0, run.stderr
    assert "CHEMOD\n5\n" in written
    assert strip_charge_records(written) == text


@pytest.mark.parametrize("key", ["ATOMORDE", "FULLE", "NANOMULTI", "HYDROSUB", "CLUSTSIZE", "NANORODS",
                                 "NANOCRYSTAL", "WULFF", "SUPERCEL", "ATOMDISP"])
def test_atom_changing_records_are_refused(key):
    with pytest.raises(ic.IonChargeError, match=rf"\({key}\)"):
        ic.read_deck_geometry(_deck(f"{key}\n1\n"))


@pytest.mark.parametrize("key", ["ELASTIC", "BREAKELAS", "ATOMROT", "MOLEXP", "MOLSPLIT", "USESAED",
                                 "POINTCHG", "PRIMITIV"])
def test_label_preserving_records_are_accepted(key):
    g = ic.read_deck_geometry(_deck(f"{key}\n1\n"))
    assert g.symbols == ["O", "Li", "Cl", "O", "O", "O"]


def test_a_record_is_matched_on_its_first_token():
    with pytest.raises(ic.IonChargeError, match=r"\(ATOMORDE\)"):
        ic.read_deck_geometry(_deck("atomorde   reorder after reading\n"))


def test_one_table():
    assert set(ic.GEOMETRY_EDITS) | ic.LABEL_PRESERVING_EDITS == set(deckscan.GEOMETRY_EDITS)
    assert "PRIMITIV" not in ic.GEOMETRY_EDITS


_PROBE = ("import sys; sys.modules['mace'] = None; import ion_charges as ic; "
          "print(ic._deckscan.__name__, sorted(ic.GEOMETRY_EDITS) == {expected!r})")


def test_flat_job_directory_copy_reads_the_same_table(tmp_path, capsys):
    from mace.utils.copy_dependencies import copy_dependencies
    copy_dependencies(str(tmp_path))
    capsys.readouterr()
    assert (tmp_path / "deckscan.py").is_file()
    code = _PROBE.format(expected=sorted(ic.GEOMETRY_EDITS))
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True,
                       timeout=120, env=env)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["deckscan", "True"]


def test_repo_layout_without_the_mace_package_reads_the_same_table(tmp_path):
    code = _PROBE.format(expected=sorted(ic.GEOMETRY_EDITS))
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT / "Crystal_d12", capture_output=True,
                       text=True, timeout=120, env=env)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["_mace_deckscan", "True"]
