"""The bond rule (1.1.6, Build › Calculate Bonds…): one function in
geom/core.js that every bond search in the browser uses (the drawn bonds on the
main thread and in the worker, Bonded atoms outside, the editor's neighbor and
fragment selection, coordination and the polyhedra).

Two atoms are bonded when min < d < tol x (r_i + r_j); a pair rule replaces
that for one pair of elements (a length window, or never). Without pair rules,
at the default tolerance and minimum, every search gives exactly what it gave
before (the tolerance and the 0.1 Å minimum were fixed in the code)."""
import json
import shutil
import subprocess

import pytest

from conftest import REPO_ROOT

pytest.importorskip("ase")

from ase.build import bulk  # noqa: E402

from mace.gui import structures as S  # noqa: E402

CORE_JS = REPO_ROOT / "mace" / "gui" / "static" / "js" / "geom" / "core.js"

# the searches of one structure under settings `cfg`: periodic bonds per cell (with the pair of each), the drawn
# bonds of the 1x1x1 cell with boundary atoms, the bonded shells (coordination, polyhedra) and the rule itself
SCRIPT = """import(process.argv[1]).then((G) => {
  const {s, cfg} = JSON.parse(process.argv[2]);
  const defaults = {bondTolerance: 1.15, bondMin: 0.1, bondMetalMetal: false, bondPairs: {}};
  const rule = G.bondRule(s, (k) => (k in cfg ? cfg[k] : defaults[k]));
  const frac = Float64Array.from(s.frac.flat()), radii = Float32Array.from(s.radii), metal = Uint8Array.from(s.metal.map(Number));
  const per = G.periodicBonds(s.cell, frac, radii, metal, rule.tol, rule.metalMetal, s.pbc, rule);
  const kinds = {};
  for (let k = 0; k < per.i.length; k++) { const n = [s.symbols[per.i[k]], s.symbols[per.j[k]]].sort().join('-'); kinds[n] = (kinds[n] || 0) + 1; }
  const d = G.displayList(s, [1, 1, 1], true), dr = new Float32Array(d.n), dm = new Uint8Array(d.n);
  for (let k = 0; k < d.n; k++) { dr[k] = s.radii[d.base[k]]; dm[k] = s.metal[d.base[k]] ? 1 : 0; }
  const drawn = G.cellListBonds(d.cart, dr, dm, rule.tol, rule.metalMetal, G.ruleFor(rule, d.base)).length / 2;
  const plain = G.cellListBonds(d.cart, dr, dm, rule.tol, rule.metalMetal).length / 2;
  const shells = G.bondedShells(s.cell, frac, radii, metal, rule.tol, s.pbc, rule).map((a) => a.length);
  console.log(JSON.stringify({periodic: per.i.length, kinds, drawn, plain, shells, tol: rule.tol, min: rule.min, mm: rule.metalMetal,
    pairs: [...rule.pairs.entries()], names: [G.pairName('si', ' O'), G.pairName('O-Si'), G.pairName('Xx', 'O')]}));
});"""


def _node():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed (the GUI's own tests check core.js in Firefox)")
    return node


def run(atoms, **cfg):
    p = S._payload(atoms.cell.array, atoms.get_scaled_positions(), atoms.positions, atoms.numbers, (True,) * 3,
                   {"kind": "test"}, symmetry_data={"available": False})
    s = {k: p[k] for k in ("cell", "frac", "pbc", "radii", "metal", "symbols", "numbers")}
    out = subprocess.run([_node(), "--input-type=module", "-e", SCRIPT, CORE_JS.as_uri(), json.dumps({"s": s, "cfg": cfg})],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


MGO = bulk("MgO", "rocksalt", a=4.212)          # Mg-O 2.106 Å, Mg-Mg 2.978 Å
SI = bulk("Si", "diamond", a=5.431)             # Si-Si 2.352 Å, second shell 3.840 Å
GESE = bulk("GeSe", "rocksalt", a=5.60)         # Ge-Se 2.80 Å: 1.17 x the covalent sum


def test_the_defaults_are_the_search_before_the_rule():
    r = run(MGO)
    assert r["tol"] == 1.15 and r["min"] == 0.1 and r["mm"] is False and r["pairs"] == []
    assert r["kinds"] == {"Mg-O": 6} and r["drawn"] == r["plain"]
    assert r["names"] == ["O-Si", "O-Si", ""]


def test_tolerance_reaches_longer_bonds():
    assert run(GESE)["periodic"] == 0                          # 2.80 Å > 1.15 x 2.40 Å
    r = run(GESE, bondTolerance=1.2)
    assert r["kinds"] == {"Ge-Se": 6} and r["drawn"] > 0


def test_a_pair_never_bonded():
    r = run(MGO, bondPairs={"Mg-O": {"never": True}})
    assert r["periodic"] == 0 and r["drawn"] == 0
    # coordination's shells: Mg without an O falls back to its metal shell (as copper's twelve Cu), O has none
    assert r["shells"] == [12, 0]


def test_a_pair_window_bonds_a_metal_pair_and_its_min_cuts_short_ones():
    # a max of its own bonds Mg-Mg, which the metal rule leaves out of a compound
    r = run(MGO, bondPairs={"Mg-Mg": {"max": 3.1}})
    assert r["kinds"] == {"Mg-O": 6, "Mg-Mg": 6}, r["kinds"]
    # a min above Mg-O drops it; the other pairs keep the tolerance
    r = run(MGO, bondPairs={"Mg-O": {"min": 2.2}})
    assert r["periodic"] == 0
    # a min alone keeps the tolerance's max: Mg-O 2.106 Å stays above 2.0
    assert run(MGO, bondPairs={"Mg-O": {"min": 2.0}})["kinds"] == {"Mg-O": 6}


def test_periodic_images_and_the_second_shell():
    r = run(SI)
    assert r["kinds"] == {"Si-Si": 4} and r["shells"] == [4, 4]      # 4 bonds per atom, each once per cell
    r = run(SI, bondPairs={"Si-Si": {"max": 4.0}})                    # out to the 12 second neighbors
    assert r["kinds"] == {"Si-Si": 16} and r["shells"] == [16, 16]
    assert r["drawn"] > run(SI)["drawn"]


def test_the_shortest_bond():
    assert run(MGO, bondMin=2.2)["periodic"] == 0
    assert run(MGO, bondMin=2.0)["kinds"] == {"Mg-O": 6}


def test_malformed_pair_rules_are_skipped():
    r = run(MGO, bondPairs={"Mg-Xx": {"max": 3}, "Mg-O": "no", "O-O": {"max": -1}, "Mg-Mg": {"max": 3.1}})
    assert len(r["pairs"]) == 1 and r["kinds"] == {"Mg-O": 6, "Mg-Mg": 6}


def test_settings_check_the_pair_rules():
    from mace.gui import settings as G
    assert G.check("bondPairs", {}) is None
    assert G.check("bondPairs", {"O-Si": {"min": 1.2, "max": 1.9}, "Mg-O": {"never": True}, "Mg-Mg": {"max": 3.1}}) is None
    for bad in ({"o-si": {"max": 2}}, {"O-Si": {}}, {"O-Si": {"max": 11}}, {"O-Si": {"min": 2, "max": 1}},
                {"O-Si": {"never": "yes"}}, {"O-Si": {"max": 2, "x": 1}}, [], {"O-Si": {"max": True}}):
        assert G.check("bondPairs", bad), bad
    assert G.check("bondMin", 0.1) is None and G.check("bondMin", 3.5)
    assert G.check("bondTolerance", 1.15) is None and G.check("bondTolerance", 2.0)
    assert G.check("bondMetalMetal", True) is None and G.check("bondMetalMetal", "yes")


# ---------------------------------------------------------------- the server's twin (structures.bond_pairs)
# Clean, mace analyze (bonds, topology, polyhedra, planes) and Compare's bond table search bonds on the server:
# structures.bond_pairs under structures.bond_rule of the same settings must list exactly the bonds core.js's
# periodicBonds lists (each pair once, i <= j, an atom's own image on the positive side).
PERIODIC = """import(process.argv[1]).then((G) => {
  const {s, cfg} = JSON.parse(process.argv[2]);
  const defaults = {bondTolerance: 1.15, bondMin: 0.1, bondMetalMetal: false, bondPairs: {}};
  const rule = G.bondRule(s, (k) => (k in cfg ? cfg[k] : defaults[k]));
  const per = G.periodicBonds(s.cell, Float64Array.from(s.frac.flat()), Float32Array.from(s.radii),
    Uint8Array.from(s.metal.map(Number)), rule.tol, rule.metalMetal, s.pbc, rule);
  console.log(JSON.stringify([...per.i].map((i, k) => [i, per.j[k], [...per.img.slice(3 * k, 3 * k + 3)]])));
});"""


def js_bonds(atoms, **cfg):
    pbc = tuple(bool(p) for p in atoms.pbc)
    p = S._payload(atoms.cell.array, atoms.get_scaled_positions(wrap=False), atoms.positions, atoms.numbers, pbc,
                   {"kind": "test"}, symmetry_data={"available": False})
    s = {k: p[k] for k in ("cell", "frac", "pbc", "radii", "metal", "symbols", "numbers")}
    out = subprocess.run([_node(), "--input-type=module", "-e", PERIODIC, CORE_JS.as_uri(), json.dumps({"s": s, "cfg": cfg})],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return sorted((i, j, tuple(t)) for i, j, t in json.loads(out.stdout))


def py_bonds(atoms, **cfg):
    return sorted(S.bond_pairs(atoms.cell.array, atoms.positions, atoms.numbers, atoms.pbc, **S.bond_rule(cfg)))


def _twin_cases():
    from ase import Atoms
    import numpy as np
    alloy = bulk("Au", "fcc", a=4.0, cubic=True)
    alloy.numbers[:2] = 29                                   # Cu2Au2: metals only, its metal-metal bonds drawn
    agcl = bulk("AgCl", "rocksalt", a=5.78)                  # a heavy cation's bond radius
    slab = Atoms("C2", cell=[[2.46, 0, 0], [-1.23, 2.1304, 0], [0, 0, 20]], pbc=[True, True, False],
                 scaled_positions=[[0, 0, 0.5], [1 / 3, 2 / 3, 0.5]])
    water = Atoms("OH2", positions=[[3, 4, 5], [3.9572, 4, 5], [2.76, 4.927, 5]], cell=np.eye(3) * 10, pbc=False)
    shifted = MGO.copy()
    shifted.positions += [5.3, -2.2, 9.1]                    # unwrapped positions: the images follow them
    return [("MgO", MGO, {}), ("MgO tol", MGO, {"bondTolerance": 1.3}), ("MgO mm", MGO, {"bondMetalMetal": True}),
            ("MgO never", MGO, {"bondPairs": {"Mg-O": {"never": True}}}),
            ("MgO window", MGO, {"bondPairs": {"Mg-Mg": {"max": 3.1}, "Mg-O": {"min": 2.0}}}),
            ("MgO min", MGO, {"bondMin": 2.2}), ("MgO shifted", shifted, {}),
            ("Si", SI, {}), ("Si second shell", SI, {"bondPairs": {"Si-Si": {"max": 4.0}}}),
            ("GeSe", GESE, {"bondTolerance": 1.2}), ("Cu", bulk("Cu", "fcc", a=3.61), {}), ("CuAu", alloy, {}),
            ("AgCl", agcl, {}), ("graphene slab", slab, {}), ("water", water, {})]


@pytest.mark.parametrize("name", [c[0] for c in _twin_cases()])
def test_the_server_finds_the_bonds_the_viewer_draws(name):
    _, atoms, cfg = next(c for c in _twin_cases() if c[0] == name)
    want = js_bonds(atoms, **cfg)
    assert py_bonds(atoms, **cfg) == want, name
    if name in ("Cu", "CuAu"):
        assert want                                          # an alloy: metal-metal bonds, drawn and cleaned


def test_clean_and_analyze_use_the_rule():
    from mace.analysis import topology
    from mace.analysis.structure import from_atoms
    from mace.gui import clean as C
    cu = bulk("Cu", "fcc", a=3.61)
    assert sorted(C.bonded_pairs(cu)) == py_bonds(cu) and len(C.bonded_pairs(cu)) == 6        # Cu-Cu: a metal's bonds
    rule = S.bond_rule({"bondPairs": {"Mg-O": {"never": True}}})
    assert C.bonded_pairs(MGO, rule) == [] and len(C.bonded_pairs(MGO)) == 6           # the document's own rule
    assert len(topology.bonds(from_atoms(cu))) == 6 and len(topology.bonds(from_atoms(MGO))) == 6
