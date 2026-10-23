"""Bonds in metals only (1.1.6): JARVIS-DFT cells drew no bond at all.

Metal-metal pairs are not bonded, as VESTA draws ionic solids (no Mg-Mg in
MgO). The viewer applied that to every structure, so an alloy or an
intermetallic, which has nothing but metals, drew none: 139 of 300 randomly
picked JARVIS-DFT CIFs (~/dft3d_cifs on the HPCC). A structure of metals only
now keeps its metal-metal bonds, as mace analyze bonds and topology count them
(analysis/topology.py bonds); a structure with a nonmetal draws exactly what it
drew before. The fixtures are JARVIS-DFT CIFs (CC0), read through the GUI's own
reader and bonded by geom/core.js, the code the viewer draws with."""
import json
import shutil
import subprocess

import pytest

from conftest import REPO_ROOT

pytest.importorskip("ase")
pytest.importorskip("spglib")

from mace.gui import structures as S  # noqa: E402

JARVIS = REPO_ROOT / "tests" / "fixtures" / "jarvis"
CORE_JS = REPO_ROOT / "mace" / "gui" / "static" / "js" / "geom" / "core.js"
ALLOYS = {"JVASP-100318.cif": "PdZn", "JVASP-76747.cif": "YZnGa", "JVASP-106626.cif": "YbAg"}
COMPOUND = "JVASP-100029.cif"          # BiSbPd2: Sb is a nonmetal, so Pd-Pd and Bi-Pd stay unbonded


def _drawn(name, setting=False):
    """What the viewer computes for the 1x1x1 cell with boundary atoms: metal-metal on or off, and the bonds."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed (the GUI's own tests check core.js in Firefox)")
    p = S.read_cif(JARVIS / name)
    s = {k: p[k] for k in ("cell", "frac", "pbc", "radii", "metal", "symbols")}
    code = (f"import('{CORE_JS.as_uri()}').then((G) => {{ const s = JSON.parse(process.argv[1]);"
            "const d = G.displayList(s, [1, 1, 1], true), radii = new Float32Array(d.n), metal = new Uint8Array(d.n);"
            "for (let k = 0; k < d.n; k++) { radii[k] = s.radii[d.base[k]]; metal[k] = s.metal[d.base[k]] ? 1 : 0; }"
            f"const mm = G.metalMetalBonds(s, {json.dumps(setting)}), p = G.cellListBonds(d.cart, radii, metal, 1.15, mm), kinds = {{}};"
            "for (let q = 0; q < p.length; q += 2) { const k = [s.symbols[d.base[p[q]]], s.symbols[d.base[p[q + 1]]]].sort().join('-');"
            " kinds[k] = (kinds[k] || 0) + 1; }"
            "console.log(JSON.stringify({mm, n: p.length / 2, kinds})); });")
    out = subprocess.run([node, "--input-type=module", "-e", code, json.dumps(s)], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


@pytest.mark.parametrize("name", sorted(ALLOYS))
def test_an_alloy_draws_its_metal_metal_bonds(name):
    r = _drawn(name)
    assert r["mm"] is True and r["n"] > 0, (ALLOYS[name], r)


def test_a_compound_with_a_nonmetal_draws_what_it_drew():
    r = _drawn(COMPOUND)
    assert r["mm"] is False
    assert r["n"] == 6 and set(r["kinds"]) == {"Pd-Sb"}, r
    # Settings' Bonds between metals still adds them
    on = _drawn(COMPOUND, True)
    assert on["mm"] is True and on["n"] > r["n"] and "Pd-Pd" in on["kinds"], on


def test_ionic_solids_keep_no_cation_cation_bond():
    from ase.build import bulk
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    mgo = bulk("MgO", "rocksalt", a=4.212)
    p = S._payload(mgo.cell.array, mgo.get_scaled_positions(), mgo.positions, mgo.numbers, (True,) * 3,
                   {"kind": "test"}, symmetry_data={"available": False})
    code = (f"import('{CORE_JS.as_uri()}').then((G) => console.log(JSON.stringify("
            "[G.metalMetalBonds(JSON.parse(process.argv[1]), false), G.metalMetalBonds({metal: [true, true]}), "
            "G.metalMetalBonds({metal: []}), G.metalMetalBonds(null)])));")
    out = subprocess.run([node, "--input-type=module", "-e", code, json.dumps({"metal": p["metal"]})],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == [False, True, False, False]
