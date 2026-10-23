"""The Rotate tool's Pivot (1.1.6): the selection's centroid or center of mass (made whole across the cell's faces),
the cell's center (a slab's: the middle of its drawing box along the open axis), the origin, or a picked atom; the
browser's rule (viewport/gizmo_math.js pivotPoint, run in node) and `mace transform --about`'s
(structure_transform.pivot_point) on the same structures, against values worked out by hand. Also: the masses are
ase's, and Settings (settings.js, settings.py, the schema, docprefs) agree on the stored values."""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
JS = REPO / "mace" / "gui" / "static" / "js"
sys.path.insert(0, str(REPO))

ase = pytest.importorskip("ase")
from ase.data import atomic_masses  # noqa: E402

from mace.structure_transform import TransformError, pivot_point, whole  # noqa: E402

needs_node = pytest.mark.skipif(not shutil.which("node"), reason="node not installed")

# H and C across the a face (frac 0.05 and 0.95: made whole they sit at x = +0.2 and -0.2 Å), an O inside
PAIR = {"cell": [[4, 0, 0], [0, 4, 0], [0, 0, 6]], "pbc": [True, True, True], "numbers": [1, 6, 8], "symbols": ["H", "C", "O"],
        "frac": [[0.05, 0.5, 0.5], [0.95, 0.5, 0.5], [0.6, 0.5, 0.5]]}
PAIR["cart"] = (np.array(PAIR["frac"]) @ np.array(PAIR["cell"], float)).tolist()
# a slab (c open, tilted): two atoms at z = 1 and 2 Å in a drawing box whose c spans them
SLAB = {"cell": [[3, 0, 0], [0, 3, 0], [0.5, 0, 4]], "pbc": [True, True, False], "numbers": [6, 6], "symbols": ["C", "C"],
        "cart": [[0.2, 0.3, 1.0], [1.1, 1.4, 2.0]], "origin": [0, 0, 0]}
SLAB["frac"] = (np.array(SLAB["cart"]) @ np.linalg.inv(np.array(SLAB["cell"], float))).tolist()

M_H, M_C = 1.008, 12.011
EXPECT_PAIR = {
    "centroid": [0.0, 2.0, 3.0],
    "com": [(M_H * 0.2 + M_C * -0.2) / (M_H + M_C), 2.0, 3.0],     # pulled toward C, not mid-cell
    "cell": [2.0, 2.0, 3.0],
    "origin": [0.0, 0.0, 0.0],
    "atom:3": [-1.6, 2.0, 3.0],     # O (not selected): its image nearest the pair's centroid (x = 2.4 - 4)
    "atom:2": [-0.2, 2.0, 3.0],     # C (selected): where the made-whole pair has it
}
# the slab's cell center: the middle of a and b, and along the tilted open c the atoms' mid-height (1.5 Å),
# moved along c itself: 1.5 * (0.5, 0, 4) / 4
EXPECT_SLAB_CELL = [1.5 + 0.1875, 1.5, 1.5]

SCRIPT = r"""
import * as M from %(math)s;
const out = {};
const pair = %(pair)s, slab = %(slab)s;
const w = M.wholeSelection(pair, new Set([0, 1]));
for (const p of ["centroid", "com", "cell", "origin", "atom:3", "atom:2", "atom:9", "nonsense"]) out[p] = M.pivotPoint(pair, w, p);
out.slabCell = M.pivotPoint(slab, M.wholeSelection(slab, new Set([0])), "cell", { mode: "bbox" });
out.parse = ["atom:12", "com", "atom:0", "", "cell"].map(M.parsePivot);
out.re = M.PIVOT_RE.source;
out.choices = M.PIVOTS.map((x) => x[0]);
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def js(tmp_path_factory):
    if not shutil.which("node"):
        pytest.skip("node not installed")
    script = tmp_path_factory.mktemp("pivot") / "run.mjs"
    script.write_text(SCRIPT % {"math": json.dumps((JS / "viewport" / "gizmo_math.js").as_uri()),
                                "pair": json.dumps(PAIR), "slab": json.dumps(SLAB)})
    r = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def py_pivot(p, idx, about):
    cell = np.array(p["cell"], float)
    fw = whole(np.array(p["frac"], float), cell, p["pbc"], idx)
    return pivot_point(about, p, idx, fw @ cell, fw)


@needs_node
@pytest.mark.parametrize("about", list(EXPECT_PAIR))
def test_browser_pivot_of_a_pair_split_across_the_cell_face(js, about):
    assert js[about]["point"] == pytest.approx(EXPECT_PAIR[about], abs=1e-9)
    assert js[about]["mode"] == about.split(":")[0]


@pytest.mark.parametrize("about", list(EXPECT_PAIR))
def test_cli_pivot_of_a_pair_split_across_the_cell_face(about):
    point, name = py_pivot(PAIR, [0, 1], about)
    assert point == pytest.approx(EXPECT_PAIR[about], abs=1e-9)
    if about.startswith("atom:"):
        assert name == {"atom:3": "O3", "atom:2": "C2"}[about]


@needs_node
def test_browser_names_and_bad_atoms(js):
    assert js["atom:3"]["name"] == "O3" and js["atom:2"]["name"] == "C2"
    assert js["com"]["name"] == "Center of Mass" and js["centroid"]["name"] == "Centroid"
    # an atom number past the end, or junk: the centroid (the panel says so)
    assert js["atom:9"]["bad"] and js["atom:9"]["point"] == pytest.approx([0, 2, 3])
    assert js["nonsense"]["mode"] == "centroid"
    assert js["parse"] == [{"mode": "atom", "atom": 11}, {"mode": "com", "atom": -1}, {"mode": "centroid", "atom": -1},
                           {"mode": "centroid", "atom": -1}, {"mode": "cell", "atom": -1}]


def test_center_of_mass_weighs_h_and_c():
    """H and C 2 Å apart: the center of mass is 2·m_H/(m_H + m_C) from C (0.155 Å), the centroid 1 Å."""
    p = {"cell": [[10, 0, 0], [0, 10, 0], [0, 0, 10]], "pbc": [True] * 3, "numbers": [1, 6], "symbols": ["H", "C"],
         "frac": [[0.5, 0.5, 0.5], [0.7, 0.5, 0.5]]}
    p["cart"] = (np.array(p["frac"]) @ np.array(p["cell"], float)).tolist()
    com, _ = py_pivot(p, [0, 1], "com")
    cen, _ = py_pivot(p, [0, 1], "centroid")
    assert com[0] == pytest.approx(7.0 - 2 * M_H / (M_H + M_C), abs=1e-9)
    assert cen[0] == pytest.approx(6.0)


@needs_node
def test_browser_cell_center_of_a_slab_is_its_drawing_box_middle(js):
    assert js["slabCell"]["point"] == pytest.approx(EXPECT_SLAB_CELL, abs=1e-9)


def test_cli_cell_center_of_a_slab_is_its_drawing_box_middle():
    cell = np.array(SLAB["cell"], float)
    fw = np.array(SLAB["frac"], float)[[0]]
    point, name = pivot_point("cell", SLAB, [0], fw @ cell, fw)
    assert point == pytest.approx(EXPECT_SLAB_CELL, abs=1e-9) and name == "the cell center"


def test_cli_atom_out_of_range_and_bad_about():
    with pytest.raises(TransformError, match="atoms 1 to 3"):
        py_pivot(PAIR, [0, 1], "atom:4")
    with pytest.raises(TransformError, match="atom:N"):
        py_pivot(PAIR, [0, 1], "mass")
    assert py_pivot(PAIR, [0, 1], "1, 2, 3")[0] == pytest.approx([1, 2, 3])


def test_browser_masses_are_ases():
    text = (JS / "geom" / "core.js").read_text()
    m = re.search(r"export const ATOMIC_MASS = \[([^\]]+)\]", text)
    table = [float(x) for x in m.group(1).replace("\n", " ").split(",")]
    assert len(table) == len(atomic_masses)
    assert np.allclose(table, atomic_masses, rtol=0, atol=1e-6)


@needs_node
def test_settings_validation_schema_and_browser_agree(js):
    from mace.gui import docprefs, settings as S
    text = (JS / "settings.js").read_text()
    js_re = re.search(r"export const ROTATE_PIVOT = /\^\((.+)\)\$/;", text).group(1)
    assert js_re == S.ROTATE_PIVOT.pattern
    assert js["re"] == f"^({S.ROTATE_PIVOT.pattern})$"
    assert '"viewport.rotatePivot": "centroid"' in text
    # the schema offers the four fixed choices (a picked atom is per document), each one the server takes
    schema = (JS / "lazy" / "settings_schema.js").read_text()
    row = re.search(r'\["viewport", "viewport\.rotatePivot", "Rotate pivot", "select", \{ options: opts\((.+?)\),\n', schema).group(1)
    offered = re.findall(r'\["(\w+)", "', row)
    assert offered == js["choices"] == ["centroid", "com", "cell", "origin"]
    for v in offered + ["atom:1", "atom:120"]:
        assert S.check("viewport.rotatePivot", v) is None, v
    for v in ["atom:0", "atom:", "atom:-1", "mass", "", 3, None, "Centroid"]:
        assert S.check("viewport.rotatePivot", v), v
    # a document keeps its own (docprefs share the Settings checks)
    assert docprefs.check_values({"viewport.rotatePivot": "atom:7"}) == {}
    assert "viewport.rotatePivot" in docprefs.check_values({"viewport.rotatePivot": "atom:0"})
