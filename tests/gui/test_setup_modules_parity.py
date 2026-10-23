"""Calculation Setup's calculation types and the Calculation menu's Regular /
Properties Calculations carry the CLI's numbers (1.1.6): one table,
mace/workflow/common/constants.py SETUP_CALC_GROUPS, served as setup/schema
calc_groups. Its numbers and words are the ones opt2d12's "Select calculation
type" menu prints (Crystal_d12/d12_interactive.py, get_calculation_type: 1 SP
... 7 CPHF, 8 PHONOPY) and opt2d3's (Crystal_d3/CRYSTALOptToD3.py: 1 BAND ...
7 ORBITALS). What each type needs from the
file comes from the workflow's prerequisites (CALC_TYPES depends_on). The
ELASTIC modules come in the order of the CLI's "Mechanical property module"
menu (d12_calc_elastic.ELASTIC_MODES, which the elastic/modes route keeps).
Static: no browser, no server."""
import re
import sys

from guiharness import REPO_ROOT

MODULES_JS = REPO_ROOT / "mace" / "gui" / "static" / "js" / "setup" / "modules.js"
INTERACTIVE = REPO_ROOT / "Crystal_d12" / "d12_interactive.py"
OPT2D3 = REPO_ROOT / "Crystal_d3" / "CRYSTALOptToD3.py"


def _cli_numbers():
    src = INTERACTIVE.read_text()
    body = src[src.index("def get_calculation_type"):]
    body = body[:body.index("\ndef ", 1)]
    pairs = re.findall(r'"(\d)":\s*"([A-Z]+)"', body)
    return {name: int(n) for n, name in pairs}


def _opt2d3_menu():
    """{type: (number, words)} of opt2d3's "Select calculation type" lines ("1: BAND - Electronic band structure")."""
    src = OPT2D3.read_text()
    body = src[src.index('ui.info("Select calculation type:")'):]
    return {m.group(2): (int(m.group(1)), m.group(3)) for m in re.finditer(r'ui\.info\("(\d): ([A-Z+]+) - ([^"]+)"\)', body[:2000])}


def _groups():
    from mace.gui import setup
    return {g["key"]: g for g in setup.calc_groups()["groups"]}


def test_the_calculation_types_carry_the_cli_numbers():
    cli = _cli_numbers()
    assert cli == {"SP": 1, "OPT": 2, "FREQ": 3, "ELASTIC": 4, "EOS": 5, "QHA": 6, "CPHF": 7, "PHONOPY": 8}, cli
    g = _groups()
    regular = {t["calc"]: t["n"] for t in g["regular"]["types"]}
    assert regular == cli
    # the words opt2d12's menu prints for each
    menu = INTERACTIVE.read_text()
    for t in g["regular"]["types"]:
        if t["n"]:
            assert f'("{t["n"]}", "{t["label"]}")' in menu, t
    d3 = _opt2d3_menu()
    assert {k: n for k, (n, _) in d3.items()} == {t["calc"]: t["n"] for t in g["properties"]["types"]}, d3
    assert [t["calc"] for t in g["properties"]["types"]] == sorted(d3, key=lambda k: d3[k][0])
    # the GUI keeps no numbers of its own
    js = MODULES_JS.read_text()
    assert "SCF_NUMBERS" not in js and not re.search(r"\{ n: ", js)


def test_what_each_type_needs_is_the_workflow_s_prerequisites():
    from mace.gui import setup
    from mace.workflow.common import constants as WC
    g = _groups()
    needs = {t["calc"]: t["needs"] for grp in g.values() for t in grp["types"]}
    # a structure (a CIF, a deck, an output) is enough for what mace convert and opt2d12 --d12-file write
    assert sorted(c for c, n in needs.items() if n == "structure") == sorted(setup.CALC_TYPES["cif2d12"]) == sorted(setup.CALC_TYPES["opt2d12"])
    # everything that reads a computed run needs a finished OPT or SP output: the modules, phonopy, opt2d3's types
    assert {c for c, n in needs.items() if n == "run"} == set(WC.MODULE_STEP_TYPES) | {"PHONOPY"} | set(setup.CALC_TYPES["opt2d3"])
    for c in WC.CALC_TYPES:
        if c in needs:
            assert (needs[c] == "run") == ("SP" in WC.CALC_TYPES[c]["depends_on"]), c
    assert setup.calc_groups()["needs"]["run"] == "a finished OPT or SP output (.out)"


def test_the_elastic_modules_keep_the_cli_menu_order():
    sys.path.insert(0, str(REPO_ROOT / "Crystal_d12"))
    import d12_calc_elastic as E
    from mace.gui import mechanics_routes as MR
    modes = [m["mode"] for m in MR.modes()["modes"]]
    assert [m for m in modes if m not in E.OWN_TYPES] == list(E.ELASTIC_MODES)
    # the Setup form numbers them 1 - ... in that order (what the CLI asks for with "Module [1]")
    assert '`<option value="${esc(m.mode)}">${i + 1} - ${esc(m.title)}</option>`' in MODULES_JS.read_text()
