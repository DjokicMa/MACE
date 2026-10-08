"""[paths] scratch: an opt-in scratch candidate right after $SCRATCH (PLAN_G
G3, D2, D3, A8, A11). Never guessed: with no setting every script keeps its
bytes on a workstation, on the owner's cluster and on another cluster;
`mace site check` only suggests a directory it finds, as a comment."""
import json
import os

import pytest

from test_scratch_guard import DECK, _run, _stub
from test_site_generic_cluster import _cluster
from test_site_jobscript import PROP, SCF, _deck, _env, _mace, _template_output, _toml

from mace.submission import site


def _gen(tmp, scratch, kind="scf"):
    s = site.Site()
    s.scratch = scratch
    text = site.customize_generator_text((SCF if kind == "scf" else PROP).read_text(), kind, s, None)[0]
    g = tmp / f"gen_{kind}.sh"
    g.write_text(text)
    return g


@pytest.mark.parametrize("where", ["workstation", "owner", "other"])
def test_no_setting_no_change(tmp_path, where):
    if where == "workstation":
        env = _env(tmp_path)
    else:
        env = _cluster(tmp_path, owner=(where == "owner"))[0]
    (tmp_path / "scratch" / "tester").mkdir(parents=True)          # a /scratch/$USER-like folder exists
    env["MACE_SCRATCH_PROBE_ROOT"] = str(tmp_path)
    d = tmp_path / "decks"
    _deck(d, "m.d12")
    p = _mace(d, env, "submit", "--nosubmit", "m.d12")
    assert p.returncode == 0, p.stdout + p.stderr
    text = (d / "m.sh").read_text()
    assert "/scratch/tester" not in text and str(tmp_path / "scratch") not in text
    if where == "workstation":
        assert (d / "m.sh").read_bytes() == _template_output(tmp_path, SCF, "m")


def test_candidate_goes_before_the_msu_lines(tmp_path):
    g = _gen(tmp_path, "/scratch/$USER")
    lines = g.read_text().splitlines()
    i = lines.index(site.SCRATCH_LINES[0])
    assert lines[i - 3:i] == ['if [ -d "/scratch/$MACE_USER" ]; then',
                              '  MACE_SCRATCH_TRY+=("$(readlink -f "/scratch/$MACE_USER")$MACE_SCRATCH_SUFFIX")', "fi"]
    # idempotent
    s = site.Site()
    s.scratch = "/scratch/$USER"
    assert site.customize_generator_text(g.read_text(), "scf", s, None)[0] == g.read_text()


@pytest.mark.parametrize("kind,job,suffix", [("scf", "diamond", "/crys23")])
def test_the_job_picks_the_site_scratch(tmp_path, kind, job, suffix):
    base = tmp_path / "sc" / "tester"
    base.mkdir(parents=True)
    submit = tmp_path / "submit"
    submit.mkdir()
    (submit / f"{job}.d12").write_text(DECK)
    g = _gen(tmp_path, f"{tmp_path}/sc/$USER", kind)
    import subprocess
    subprocess.run(["bash", str(g), job], cwd=submit, check=True, capture_output=True)
    script = submit / f"{job}.sh"
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
    r = _run(tmp_path, script, {"SCRATCH": "", "USER": "tester"})
    want = f"{os.path.realpath(base)}{suffix}/{job}"
    assert (submit / f".{job}.scratch").read_text().strip() == want, r.stdout + r.stderr
    assert _stub(submit, job)["CWD"] == want


@pytest.mark.parametrize("bad", ["/scratch/../etc", "/scratch/'x'", "/scratch/$(id)", "/scratch/${PATH}",
                                 "/scratch;id", "/scratch/a b", "relative/dir", "/scratch/`id`", "/scratch/$PWD"])
def test_invalid_scratch_values(tmp_path, bad):
    import json as _j
    _toml(tmp_path, "[paths]\nscratch = " + _j.dumps(bad) + "\n")
    d = tmp_path / "decks"
    _deck(d, "m.d12")
    p = _mace(d, _env(tmp_path), "submit", "--nosubmit", "m.d12")
    assert p.returncode == 2 and "[paths] scratch" in p.stdout and not (d / "m.sh").exists()


def test_env_scratch_base(tmp_path):
    d = tmp_path / "decks"
    _deck(d, "m.d12")
    p = _mace(d, _env(tmp_path, MACE_SCRATCH_BASE="/work/$USER/s"), "submit", "--nosubmit", "m.d12")
    assert p.returncode == 0 and 'if [ -d "/work/$MACE_USER/s" ]; then' in (d / "m.sh").read_text()


def test_site_check_suggests_but_changes_nothing(tmp_path):
    env, fake = _cluster(tmp_path, owner=False)
    (tmp_path / "scratch" / "tester").mkdir(parents=True)
    env["MACE_SCRATCH_PROBE_ROOT"] = str(tmp_path)
    r = _mace(tmp_path, env, "site", "check", "--json", "--no-sbatch")
    rep = json.loads(r.stdout)
    import shutil
    if rep.get("kept"):
        shutil.rmtree(rep["kept"])
    item = next(i for i in rep["items"] if i["item"] == "Scratch")
    assert f'# scratch = "{tmp_path}/scratch/$USER"' in item["fix"] and "shared by all nodes" in item["fix"]
    d = tmp_path / "decks"
    _deck(d, "m.d12")
    assert _mace(d, env, "submit", "--nosubmit", "m.d12").returncode == 0
    assert "/scratch/tester" not in (d / "m.sh").read_text()
