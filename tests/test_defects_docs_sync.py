"""The MgO V_O example in docs/defects is what the builder writes today."""
import json
import subprocess
import sys

from _defects_fixtures import deck, same_but_title
from conftest import REPO_ROOT

DOCS = REPO_ROOT / "docs" / "defects"


def test_docs_decks_equal_fresh_builder_output(tmp_path):
    out = tmp_path / "MgO_VO"
    r = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "defects", "build",
                        "docs/defects/MgO_host.d12", "--vacancy", "O", "--site-label", "1",
                        "--supercon", "2x2x2", "--charges", "0,1,2", "--calibrate", "--name", "mgo",
                        "-o", str(out)], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
                       stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stdout + r.stderr
    fresh = sorted(p.name for p in out.iterdir())
    shipped = sorted(p.name for p in (DOCS / "MgO_VO").iterdir() if p.name != "mu.example.json")
    assert fresh == shipped
    for name in fresh:
        assert (out / name).read_text() == (DOCS / "MgO_VO" / name).read_text(), name
    assert not list((DOCS / "MgO_VO").glob("*.d3"))


def test_docs_decks_are_the_measured_ones():
    d = DOCS / "MgO_VO"
    for tag, ref in (("pris", "mgo_pris"), ("q0", "mgo_v0"), ("qp1", "mgo_v1"), ("qp2", "mgo_v2"),
                     ("calm05", "mgo_j05"), ("calm10", "mgo_j10"), ("calm15", "mgo_j15")):
        assert same_but_title((d / f"mgo_{tag}.d12").read_text(), deck(ref)), tag
    mu = json.loads((d / "mu.example.json").read_text())
    assert mu["O"].endswith("Ha")
