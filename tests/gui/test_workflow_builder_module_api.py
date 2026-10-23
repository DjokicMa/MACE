"""The workflow builder's ELASTIC, EOS, QHA and CPHF steps (1.1.6).

"Add a step" offers them after an OPT or SP. Their form is the module block
(Calculation Setup's module fields, mace opt2d12's --elastic-settings keys)
with the deck mace opt2d12 writes from a stand-in of the OPT before them;
the block becomes the spec's elastic_settings override, and the preview's
projection shows each module deck as the engine writes it.
"""
import json

import pytest

import submit_util as SU
from test_workflow_builder_api import CIF, _cifs, _draft, _preview, _task

pytestmark = pytest.mark.skipif(not CIF.exists(), reason="test/ data corpus not present")

SEQ = ["OPT", "SP", "ELASTIC", "EOS", "QHA", "CPHF"]


@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    s = SU.Served(tmp_path_factory.mktemp("wfmod"), MACE_SITE_CONFIG="none")
    yield s
    s.stop()


def _block(text, key):
    lines = text.splitlines()
    i = lines.index(key)
    return lines[i:lines.index("END", i) + 1]


def test_templates_list_the_module_steps_after_opt_or_sp(srv):
    st, _, r = srv.req("api/workflow/templates")
    assert st == 200
    for t in ("ELASTIC", "EOS", "QHA", "CPHF"):
        ct = r["calc_types"][t]
        assert ct["depends_on"] == ["OPT", "SP"] and ct["kind"] == "d12" and not ct["repeatable"], ct
        assert ct["help"] and t in ct["name"]


def test_elastic_form_is_the_module_block_and_its_deck(srv):
    _cifs(srv, "m1", names=("A",))
    draft = _draft("m1", names=("A",), template=None, sequence=SEQ, steps={})
    st, r = _task(srv, "workflow/step-form", {"draft": draft, "step": "ELASTIC_3"})
    assert st == 200, r
    assert r["mode"] == "module" and r["parent"] == "OPT_1" and r["module"] == "ELASTCON"
    assert [m["mode"] for m in r["modes"]] == ["ELASTCON", "ELAPIEZO", "PIEZOCON", "PIEZOCP", "PHOTOELA"]
    assert r["settings"] == {} and r["toldee"] == 8 and "refusal" not in r
    assert _block(r["deck"], "ELASTCON") == ["ELASTCON", "END"] and "OPTGEOM" not in r["deck"]
    draft["steps"] = {"ELASTIC_3": {"module": {"mode": "ELAPIEZO", "numderiv": 5, "print": True}}}
    st, r = _task(srv, "workflow/step-form", {"draft": draft, "step": "ELASTIC_3"})
    assert st == 200 and r["module"] == "ELAPIEZO", r
    assert _block(r["deck"], "ELAPIEZO") == ["ELAPIEZO", "NUMDERIV", "5", "PRINT", "END"]
    assert r["describe"] == "ELAPIEZO: NUMDERIV 5 PRINT"


def test_own_type_forms(srv):
    _cifs(srv, "m2", names=("A",))
    draft = _draft("m2", names=("A",), template=None, sequence=SEQ,
                   steps={"QHA_5": {"module": {"temperat": ["60", "10", "600"]}}})
    st, r = _task(srv, "workflow/step-form", {"draft": draft, "step": "QHA_5"})
    assert st == 200 and r["mode"] == "module" and [m["mode"] for m in r["modes"]] == ["QHA"], r
    assert r["toldee"] == 10 and _block(r["deck"], "QHA") == ["QHA", "TEMPERAT", "60 10.0 600.0", "END"]
    st, r = _task(srv, "workflow/step-form", {"draft": draft, "step": "CPHF_6"})
    assert st == 200 and r["module"] == "CPHF" and _block(r["deck"], "CPHF") == ["CPHF", "END"]


def test_bad_module_settings_are_refused(srv):
    _cifs(srv, "m3", names=("A",))
    draft = _draft("m3", names=("A",), template=None, sequence=SEQ,
                   steps={"EOS_4": {"module": {"mode": "ELASTCON"}}})
    st, r = _task(srv, "workflow/step-form", {"draft": draft, "step": "EOS_4"})
    assert st == 400 and "not an EOS step's" in r["error"]
    draft["steps"] = {"ELASTIC_3": {"module": {"numderiv": 1}}}
    st, r = _task(srv, "workflow/step-form", {"draft": draft, "step": "ELASTIC_3"})
    assert st == 200 and "at least 2" in r["refusal"] and r["deck"] is None, r
    st, prev = _preview(srv, _draft("m3", names=("A",), steps={"BAND_3": {"module": {"mode": "EOS"}}}))
    assert st == 400 and "ELASTIC, EOS, QHA or CPHF step's settings" in prev["error"]


def test_preview_projects_each_module_deck(srv):
    _cifs(srv, "m4", names=("A",))
    steps = {"ELASTIC_3": {"module": {"mode": "ELASTCON", "nuchess": True, "hessnum2": True}},
             "EOS_4": {"module": {"range": ["0.94", "1.06", "6"]}, "resources": {"walltime": "2-00:00:00"}}}
    st, prev = _preview(srv, _draft("m4", names=("A",), template=None, sequence=SEQ, steps=steps))
    assert st == 200 and prev["code"] == 0, (prev.get("refused"), (prev.get("cli_log") or "")[-2000:])
    spec = json.loads(prev["spec"]["text"])
    assert spec["sequence"] == SEQ
    assert spec["steps"]["ELASTIC_3"]["overrides"] == {"elastic_settings": steps["ELASTIC_3"]["module"]}
    assert spec["steps"]["EOS_4"]["overrides"] == {"elastic_settings": {"range": ["0.94", "1.06", "6"]}}
    plan = json.loads(prev["plan"]["text"])
    assert plan["step_configurations"]["ELASTIC_3"]["elastic_settings"]["nuchess"] is True
    for t, n in (("elastic", 3), ("eos", 4), ("qha", 5), ("cphf", 6)):
        assert any(f"submitcrystal23_{t}_{n}.sh" in w["rel"] for w in prev["workflow_scripts"]), t
    got = {p["step"]: p for p in prev["projected"]}
    assert {"ELASTIC_3", "EOS_4", "QHA_5", "CPHF_6"} <= set(got), (list(got), prev.get("warnings"))
    assert all(got[k]["parent"] == "OPT_1" for k in ("ELASTIC_3", "EOS_4", "QHA_5", "CPHF_6"))
    assert _block(got["ELASTIC_3"]["deck_text"], "ELASTCON") == ["ELASTCON", "NUCHESS", "HESSNUM2", "END"]
    assert _block(got["EOS_4"]["deck_text"], "EOS") == ["EOS", "RANGE", "0.94 1.06 6", "END"]
    assert _block(got["QHA_5"]["deck_text"], "QHA") == ["QHA", "END"]
    assert got["EOS_4"]["resources"].get("walltime") == "2-00:00:00", got["EOS_4"]["resources"]
    assert "Pproperties" not in (got["CPHF_6"]["script_text"] or "")
