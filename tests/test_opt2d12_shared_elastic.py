"""opt2d12 batch with shared settings: ELASTIC/EOS/QHA/CPHF ask their questions once.

In shared mode the settings come from the first file's answers and every
file is written without questions, so menu choices 4-7 never asked the
module's (which module, NUMDERIV, NUCHESS, ...): every deck was a bare
"ELASTCON / END" whatever the user wanted. The questions are now asked once,
after the shared settings, and the answers go to every file and to the
--save-options template.

Two real parents (the repo's Si PBE0 and diamond B3LYP-D3 optimizations),
answers piped as a user's answer file would be, through mace_cli.
"""
import json
import os
import shutil
import subprocess
import sys

from conftest import REPO_ROOT

MACE_CLI = REPO_ROOT / "mace_cli"
SI = REPO_ROOT / "tests" / "fixtures" / "phonopy_crystal" / "Si"
DIA = REPO_ROOT / "tests" / "data" / "existing_calcs"

# shared [Y], baseline [Y], calculation type 4 (ELASTIC), the five "Change ...?"
# [N], then the module's questions: ELASTCON, NUMDERIV 5, STEPSIZE [0.01],
# nuclear relaxation 2 (NUCHESS, with its HESSNUM2 default), and the rest at their defaults
ANSWERS = "\n\n4\n" + "\n" * 5 + "1\n5\n\n2\n" + "\n" * 40


def _batch(tmp_path, answers, *extra):
    shutil.copy(SI / "parent.out", tmp_path / "si.out")
    shutil.copy(SI / "parent.d12", tmp_path / "si.d12")
    for ext in ("out", "d12"):
        shutil.copy(DIA / f"dia_opt.{ext}", tmp_path / f"dia_opt.{ext}")
    env = dict(os.environ, MACE_NO_BANNER="1")
    env.pop("PYTHONPATH", None)
    return subprocess.run([sys.executable, str(MACE_CLI), "opt2d12", "--directory", ".", "--output-dir", "o",
                           *extra], cwd=tmp_path, env=env, input=answers, capture_output=True, text=True,
                          timeout=300)


def _block(path, key):
    lines = path.read_text().splitlines()
    i = lines.index(key)
    return lines[i:lines.index("END", i) + 1]


def test_the_module_questions_are_asked_once_and_applied_to_every_file(tmp_path):
    r = _batch(tmp_path, ANSWERS, "--save-options", "--options-file", "shared.json")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    assert r.stdout.count("Mechanical property module:") == 1
    decks = sorted((tmp_path / "o").glob("*.d12"))
    assert [d.name for d in decks] == ["dia_opt_elastcon_B3LYP-D3_optimized.d12",
                                       "si_elastcon_B3LYP-D3_optimized.d12"]
    for d in decks:
        assert _block(d, "ELASTCON") == ["ELASTCON", "NUMDERIV", "5", "NUCHESS", "HESSNUM2", "PRINT", "END"], d.name
    saved = json.loads((tmp_path / "shared.json").read_text())
    saved = saved.get("configuration", saved)
    assert saved["elastic_settings"]["mode"] == "ELASTCON"
    assert saved["elastic_settings"]["numderiv"] == 5 and saved["elastic_settings"]["nuchess"] is True


def test_the_saved_template_writes_the_same_decks(tmp_path):
    first = tmp_path / "first"
    first.mkdir()
    assert _batch(first, ANSWERS, "--save-options", "--options-file", "shared.json").returncode == 0
    again = tmp_path / "again"
    again.mkdir()
    r = _batch(again, "", "--config-file", str(first / "shared.json"))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    for d in sorted((first / "o").glob("*.d12")):
        assert _block(again / "o" / d.name, "ELASTCON") == _block(d, "ELASTCON")


def test_an_sp_batch_asks_nothing_more(tmp_path):
    r = _batch(tmp_path, "\n\n1\n" + "\n" * 40)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    assert "Mechanical property module:" not in r.stdout
    assert len(list((tmp_path / "o").glob("*_sp_*.d12"))) == 2
