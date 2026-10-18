"""opt2d3 writes ORBITALS decks: NEWK, ORBITALS, root, ICAR, ILOC, one END.

Measured on CRYSTAL23 (HPCC, 2026-10-16): ICAR and ILOC have to be records of
their own (on one line properties stops with FORMAT ERROR IN INPUT DECK, and
the serial program still exits 0), ORBITALS has no END of its own (one END
closes the deck), and a molecule's NEWK has no mesh line. The root is
<deck stem>_orb, so the job script can copy the files back by name.

Refused: a two-component (SOC) parent (CRYSTAL23 crashes in NEWK/ORBITALS,
spec sec. 6), ILOC 1 (Wannier functions through ORBITALS are not usable) and
a 2D/3D custom IS with complex k points.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT

_D3 = str(REPO_ROOT / "Crystal_d3")
if _D3 not in sys.path:
    sys.path.insert(0, _D3)

import CRYSTALOptToD3 as d3gen       # noqa: E402
import d3_config                      # noqa: E402

DATA = REPO_ROOT / "tests" / "data" / "orbitals" / "scf"

SI_DECK = "NEWK\n2 12\n1 0\nORBITALS\nsi_orbitals_orb\n1\n0\nEND"
H2O_DECK = "NEWK\n1 0\nORBITALS\nh2o_orbitals_orb\n0\n0\nEND"


def _stage(tmp_path, name, with_d12=True):
    d = tmp_path / "run"
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(DATA / f"{name}.out", d / f"{name}.out")
    if with_d12 and (DATA / f"{name}.d12.txt").is_file():     # .txt: not a committed sample deck
        shutil.copy(DATA / f"{name}.d12.txt", d / f"{name}.d12")
    (d / f"{name}.f9").write_text("wavefunction placeholder\n")
    return d


def _generate(tmp_path, monkeypatch, name, config):
    d = _stage(tmp_path, name)
    monkeypatch.setattr(d3gen.D3Generator, "_copy_wavefunction", lambda self: True)
    monkeypatch.setattr(d3gen.D3Generator, "_place_wavefunction", lambda self: None)
    g = d3gen.D3Generator(str(d / f"{name}.out"), "ORBITALS", output_dir=str(d))
    result = g.generate_d3(dict(config, calculation_type="ORBITALS"))
    deck = d / f"{name}_orbitals.d3"
    return result, (deck.read_text() if deck.is_file() else None)


# ------------------------------------------------------------------ goldens
def test_periodic_deck_golden(tmp_path, monkeypatch):
    _, deck = _generate(tmp_path, monkeypatch, "si", {"kmesh": "band_edges"})
    assert deck == SI_DECK


def test_molecule_deck_golden(tmp_path, monkeypatch):
    _, deck = _generate(tmp_path, monkeypatch, "h2o", {"kmesh": "band_edges"})
    assert deck == H2O_DECK


@pytest.mark.parametrize("name,config,mesh", [
    ("si", {"kmesh": "gamma"}, "1 12"),
    ("si", {"kmesh": "all_real"}, "2 12"),
    ("mgoslab", {"kmesh": "band_edges"}, "1 12"),
    ("polyyne", {"kmesh": "band_edges"}, "2 16"),
    ("polyyne", {"kmesh": "custom", "is": 4}, "4 16"),
    ("fe", {"kmesh": "band_edges"}, "1 16"),
])
def test_mesh_per_mode(tmp_path, monkeypatch, name, config, mesh):
    _, deck = _generate(tmp_path, monkeypatch, name, config)
    lines = deck.split("\n")
    assert lines[:3] == ["NEWK", mesh, "1 0"]
    assert lines[3:] == ["ORBITALS", f"{name}_orbitals_orb", "1", "0", "END"]


def test_icar_and_iloc_are_records_of_their_own_and_one_end(tmp_path, monkeypatch):
    for name in ("si", "h2o"):
        _, deck = _generate(tmp_path / name, monkeypatch, name, {})
        lines = deck.split("\n")
        i = lines.index("ORBITALS")
        assert lines[i + 2] in ("0", "1") and lines[i + 3] == "0"     # ICAR, ILOC: one number each
        assert lines.count("END") == 1 and lines[-1] == "END"


# ------------------------------------------------------------------ refusals
def test_two_component_parent_is_refused(tmp_path, monkeypatch):
    from mace.orbitals.kchoice import SOC_REFUSAL
    result, deck = _generate(tmp_path, monkeypatch, "hi2c", {"kmesh": "band_edges"})
    assert result is None and deck is None
    # the message itself, through the CLI as the user runs it
    d, r = _cli(tmp_path / "cli", "hi2c", "ORBITALS")
    text = " ".join((r.stdout + r.stderr).split())
    assert r.returncode != 0
    assert SOC_REFUSAL in text, text[-1500:]
    assert not list(d.glob("*.d3"))


def test_iloc_1_is_refused(tmp_path, monkeypatch):
    result, deck = _generate(tmp_path, monkeypatch, "si", {"kmesh": "gamma", "iloc": 1})
    assert result is None and deck is None
    assert "Wannier90" in d3gen.orbitals_config_refusal({"iloc": 1})
    assert d3gen.orbitals_config_refusal({"iloc": 0}) is None
    ok, errors = d3_config.validate_d3_config({"calculation_type": "ORBITALS", "iloc": 1})
    assert not ok and any("ILOC 1" in e for e in errors)


def test_complex_custom_is_is_refused_in_3d(tmp_path, monkeypatch):
    result, deck = _generate(tmp_path, monkeypatch, "si", {"kmesh": "custom", "is": 4})
    assert result is None and deck is None


def test_molecules_are_not_refused():
    assert d3gen.molecule_refusal("ORBITALS", 0) is None


# ------------------------------------------------------------------ d3_config
def test_config_validation_and_saving():
    assert d3_config.validate_d3_config({"calculation_type": "ORBITALS"}) == (True, [])
    assert d3_config.validate_d3_config(d3_config.get_default_d3_config("ORBITALS"))[0]
    ok, errors = d3_config.validate_d3_config({"calculation_type": "ORBITALS", "kmesh": "custom"})
    assert not ok and "custom needs" in errors[0]
    ok, errors = d3_config.validate_d3_config({"calculation_type": "ORBITALS", "kmesh": "dense"})
    assert not ok
    cleaned = d3_config.clean_config_for_saving({"calculation_type": "ORBITALS", "kmesh": "custom", "is": 2,
                                                 "edges": [1], "notes": ["x"]})
    assert cleaned == {"calculation_type": "ORBITALS", "kmesh": "custom", "is": 2}


# ------------------------------------------------------------------ the CLI
def _cli(d, name, calc_type, stdin="\n" * 40, config=None):
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(DATA / f"{name}.out", d / f"{name}.out")
    (d / f"{name}.f9").write_text("wavefunction placeholder\n")
    env = dict(os.environ)
    for k in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH"):
        env.pop(k, None)
    env["MACE_NO_BANNER"] = "1"
    args = [sys.executable, str(REPO_ROOT / "mace_cli"), "opt2d3", "--input", str(d / f"{name}.out")]
    if calc_type:
        args += ["--calc-type", calc_type]
    if config:
        (d / "orb.json").write_text(json.dumps({"version": "1.0", "type": "d3_configuration",
                                                "calculation_type": "ORBITALS", "configuration": config}))
        args += ["--config-file", str(d / "orb.json")]
    r = subprocess.run(args, cwd=str(d), capture_output=True, text=True, env=env, timeout=180, input=stdin)
    return d, r


def test_cli_menu_and_default_answers_write_the_band_edge_deck(tmp_path):
    # "7" picks ORBITALS from the calculation menu; Enter takes the band-edge k points
    d, r = _cli(tmp_path, "si", None, stdin="7\n\n")
    assert r.returncode == 0, (r.stdout + r.stderr)[-2000:]
    assert (d / "si_orbitals.d3").read_text() == SI_DECK
    assert (d / "si_orbitals.f9").is_file()
    assert "K 10" in r.stdout and "nearest real points" in " ".join(r.stdout.split())


def test_cli_k_menu_choices(tmp_path):
    d, r = _cli(tmp_path / "g", "si", "ORBITALS", stdin="2\n")
    assert r.returncode == 0, (r.stdout + r.stderr)[-2000:]
    assert (d / "si_orbitals.d3").read_text().split("\n")[1] == "1 12"
    # custom IS on a 3D structure: 4 is refused at the question, 2 is taken
    d, r = _cli(tmp_path / "c", "si", "ORBITALS", stdin="4\n4\n2\n")
    assert r.returncode == 0, (r.stdout + r.stderr)[-2000:]
    assert "complex k points" in r.stdout
    assert (d / "si_orbitals.d3").read_text().split("\n")[1] == "2 12"


def test_cli_molecule_asks_nothing(tmp_path):
    d, r = _cli(tmp_path, "h2o", "ORBITALS", stdin="")
    assert r.returncode == 0, (r.stdout + r.stderr)[-2000:]
    assert (d / "h2o_orbitals.d3").read_text() == H2O_DECK


def test_cli_config_file(tmp_path):
    d, r = _cli(tmp_path, "polyyne", "ORBITALS", stdin="", config={"calculation_type": "ORBITALS",
                                                                  "kmesh": "all_real"})
    assert r.returncode == 0, (r.stdout + r.stderr)[-2000:]
    assert (d / "polyyne_orbitals.d3").read_text().split("\n")[:2] == ["NEWK", "2 16"]


def test_cli_file_count_warning(tmp_path):
    # IS 12 on the 1D chain: complex points are allowed in 1D, and the files are counted
    d, r = _cli(tmp_path, "polyyne", "ORBITALS", stdin="4\n200\n")
    assert r.returncode == 0, (r.stdout + r.stderr)[-2000:]
    assert "Molden files" in r.stdout and "1M-file quota" in r.stdout
