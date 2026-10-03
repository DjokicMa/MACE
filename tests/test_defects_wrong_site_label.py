"""A --site-label on the wrong element is reported as such by the analysis.

Before a pristine output exists, `mace defects build --site-label N` is
accepted with a warning only. The analysis then took the jobs' electron
counts against the pristine one, so a V_O built on a Mg site was excluded as
"DOPING not applied" - the count of the wrong atom - instead of saying the
site is Mg. Pristine geometry: the HPCC MgO 2x2x2 (SUPERCON) supercell
excerpt in tests/defects_data (atoms 1-32 O, 33-64 Mg).
"""
import gzip

from conftest import REPO_ROOT

from mace.defects import analysis

PRIS = (REPO_ROOT / "tests" / "defects_data" / "mgo_pris_geometry.out").read_text()
DEFECT = gzip.open(REPO_ROOT / "tests" / "data" / "supercell_parents" / "rs_qm1.out.gz", "rt").read()


def _manifest(label):
    return {"defect": {"kind": "vacancy", "element": "O", "delta_electrons": -8},
            "site": {"label": label, "source": "label (unchecked)"},
            "supercells": [{"pristine_job": "p", "jobs": [
                {"name": "p", "role": "pristine", "q": 0, "doping": 0.0},
                {"name": "d", "role": "defect", "q": 0, "doping": 0.0}]}]}


def test_a_label_on_the_wrong_element_is_named():
    results = {r.name: r for r in analysis.check_jobs(_manifest(33), {"p": PRIS, "d": DEFECT})}
    reason = results["d"].excluded
    assert reason and "site 33 of the pristine supercell is Mg, not O" in reason
    assert "DOPING not applied" not in reason
    assert results["p"].excluded != reason


def test_a_label_on_the_right_element_is_not_flagged():
    assert analysis._wrong_site(_manifest(1), PRIS) is None
