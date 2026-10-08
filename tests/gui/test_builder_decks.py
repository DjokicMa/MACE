"""GUI -> JSON -> CLI on built structures, against an independent manual
route (plan T13).

decks.py already runs the real CLI, so comparing its preview with cif2d12 on
the same CIF would check the CLI against itself. The risk is the built
structure, so the reference here is built with ase only (no MACE GUI code),
written as a CIF and converted by the real `mace convert --batch` in a
subprocess. The GUI route builds the same structure through /api/build,
saves it with Save As's default CIF mode (P1), and previews the deck of the
saved file. The decks must agree line for line, the title line and the
order of the coordinate lines aside (compared as sets, to 1e-6 in fractional
units). The space-group line is pinned to the manual route's value (S5: 1,
since cif2d12 keeps the P1 that the CIF declares), and SPIN follows the JSON
(standard_dft_opt.json writes none; with spin_polarized: true it is there
on both routes)."""
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

import guiharness as H
import p4util as U

np = pytest.importorskip("numpy")
ase_io = pytest.importorskip("ase.io")
pytest.importorskip("spglib")

CONFIG = H.REPO_ROOT / "Crystal_d12" / "example_configs" / "standard_dft_opt.json"


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("decks")
    root, xdg = H.examples_copy(tmp), tmp / "xdg"
    proc, base, _ = H.start_server(root, xdg, MACE_GUI_NO_WARMUP="1")
    yield base, root, tmp
    proc.terminate()
    proc.wait(timeout=15)


def mgo():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ase_io.read(str(H.EXAMPLES / "structures" / "MgO.cif"))


# ---------------------------------------------------------------- the manual route (ase only)
def manual_slab(layers=5, vacuum=15.0):
    """ase.build.surface + S2's plane trimming, written out here."""
    from ase.build import surface
    s = surface(mgo(), (0, 0, 1), layers + 2, vacuum=None)
    z = s.positions[:, 2]
    order = np.argsort(z)
    planes, cur = [], [order[0]]
    for i in order[1:]:
        if z[i] - z[cur[-1]] <= 0.15:
            cur.append(i)
        else:
            planes.append(cur)
            cur = [i]
    planes.append(cur)
    keep = sorted(i for p in planes[:layers] for i in p)
    s = s[keep]
    top, bottom = s.positions[:, 2].max(), s.positions[:, 2].min()
    cell = s.cell.array.copy()
    cell[2] = [0, 0, (top - bottom) + vacuum]
    s.set_cell(cell, scale_atoms=False)
    s.center(axis=2)
    return s


def manual(name):
    from ase.build import make_supercell
    a = mgo()
    if name == "MgO_2x2x2":
        return make_supercell(a, 2 * np.eye(3))
    if name == "MgO_r2xr2x1":
        return make_supercell(a, [[1, -1, 0], [1, 1, 0], [0, 0, 1]])
    if name == "MgO_001_5L":
        return manual_slab()
    if name == "MgO_2x2x2_V_O_4b":
        s = make_supercell(a, 2 * np.eye(3))
        first_o = [i for i, x in enumerate(s.get_chemical_symbols()) if x == "O"][0]
        del s[first_o]
        return s
    raise KeyError(name)


def cli_deck(atoms, name, cfg: Path, work: Path) -> str:
    d = work / f"manual_{name}_{cfg.stem}"
    (d / "cif").mkdir(parents=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ase_io.write(str(d / "cif" / f"{name}.cif"), atoms, format="cif")
    env = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "PYTHONPATH", "MACE_HOME")}
    env.update(NO_COLOR="1", MACE_NO_BANNER="1", MPLBACKEND="Agg")
    r = subprocess.run([sys.executable, str(H.REPO_ROOT / "mace_cli"), "convert", "--no-banner", "--batch", "--options_file", str(cfg),
                        "--cif_dir", str(d / "cif"), "--output_dir", str(d / "out")], stdin=subprocess.DEVNULL,
                       capture_output=True, text=True, timeout=300, env=env, cwd=str(d))
    decks = sorted((d / "out").glob("*.d12"))
    assert r.returncode == 0 and decks, r.stdout[-2000:] + r.stderr[-2000:]
    return decks[0].read_text()


# ---------------------------------------------------------------- the GUI route
GUI = {
    "MgO_2x2x2": ("supercell", {"n": [2, 2, 2]}),
    "MgO_r2xr2x1": ("supercell", {"matrix": [[1, -1, 0], [1, 1, 0], [0, 0, 1]]}),
    "MgO_001_5L": ("slab", {"hkl": [0, 0, 1], "layers": 5, "vacuum": 15}),
    "MgO_2x2x2_V_O_4b": ("defect", {"kind": "vacancy", "site": 4, "matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 2]]}),
}


def gui_saved(base, root, name, cif_mode=None):
    kind, params = GUI[name]
    st, _, r = U.post(base, f"api/build/{kind}", {"source": {"path": "structures/MgO.cif"}, "params": params,
                                                  "open_as": {"name": name}})
    if st == 202:
        st, _, r = wait_task(base, r)
    assert st == 200 and r.get("session"), r
    body = {"session": r["session"]["session"], "path": f"built/{name}{'_sym' if cif_mode else ''}.cif", "format": "cif",
            "overwrite": True}
    if cif_mode:
        body["cif_mode"] = cif_mode
    (root / "built").mkdir(exist_ok=True)
    st, _, out = U.post(base, "api/structure/save", body)
    assert st == 200, out
    return body["path"], r["session"]["session"]


def wait_task(base, r, timeout=120):
    import time
    tid = r["task"]["id"]
    end = time.time() + timeout
    while time.time() < end:
        st, _, res = U.req(base, f"api/tasks/{tid}/result")
        if st != 409:
            return st, _, res
        time.sleep(0.2)
    raise AssertionError("the build task did not finish")


def gui_deck(base, path, cfg: Path) -> str:
    st, _, r = U.post(base, "api/deck/preview", {"mode": "cif2d12", "json": json.loads(cfg.read_text()), "structure": path})
    assert st == 200, r
    return r["deck"]


# ---------------------------------------------------------------- comparing decks
def split_deck(text):
    """(title, keyword lines in order, the space-group line, coordinate rows as a set)."""
    lines = text.splitlines()
    title = lines[0]
    rest = lines[1:]
    # CRYSTAL geometry: CRYSTAL, 0 0 0, space group, cell, natoms, then natoms rows
    k = rest.index("CRYSTAL")
    sg = rest[k + 2].strip()
    cell = rest[k + 3].split()
    n = int(rest[k + 4].split()[0])
    rows = rest[k + 5:k + 5 + n]
    atoms = sorted((int(r.split()[0]), *(round(float(x), 6) % 1.0 for x in r.split()[1:4])) for r in rows)
    keywords = rest[:k + 2] + rest[k + 5 + n:]
    return title, keywords, sg, [float(x) for x in cell], atoms


def same_atoms(a, b, tol=1e-6, translate=False):
    """The same atoms (number, fractional position mod 1) as sets; with
    `translate`, up to one rigid shift (a slab's in-plane origin is a free
    choice: the GUI and ase pick different ones for the same slab)."""
    if len(a) != len(b):
        return False
    pa, pb = np.array(a, float), np.array(b, float)
    if sorted(pa[:, 0]) != sorted(pb[:, 0]):
        return False

    def match(shift):
        for row in pa:
            d = pb[:, 1:] - (row[1:] + shift)
            d -= np.round(d)
            if not ((pb[:, 0] == row[0]) & (np.abs(d).max(1) <= tol)).any():
                return False
        return True
    if match(np.zeros(3)):
        return True
    if translate:
        z0 = pa[0, 0]
        for row in pb[pb[:, 0] == z0]:
            if match(row[1:] - pa[0, 1:]):
                return True
    return False


SIZE_KEYS = {"SHRINK", "BIPOSIZE", "EXCHSIZE"}


def symmetry_free(keywords):
    """The keyword lines without the ones cif2d12 sizes from the symmetry:
    SHRINK and the integral buffers, each with its number lines."""
    out, skip = [], False
    for line in keywords:
        word = line.strip()
        if word in SIZE_KEYS:
            skip = True
            continue
        if skip and (not word or word[0].isdigit() or word[0] == "-"):
            continue
        skip = False
        out.append(line)
    return out


def spin_cfg(tmp: Path) -> Path:
    data = json.loads(CONFIG.read_text())
    data["configuration"]["spin_polarized"] = True
    p = tmp / "standard_dft_opt_spin.json"
    p.write_text(json.dumps(data, indent=2))
    return p


@pytest.mark.parametrize("name", list(GUI))
def test_gui_deck_equals_manual_route(served, name):
    base, root, tmp = served
    path, _ = gui_saved(base, root, name)
    for cfg in (CONFIG, spin_cfg(tmp)):
        g = split_deck(gui_deck(base, path, cfg))
        m = split_deck(cli_deck(manual(name), name, cfg, tmp))
        assert g[1] == m[1], (name, cfg.name, g[1], m[1])                 # every keyword line, in order
        assert g[2] == m[2] == "1", (name, g[2], m[2])                     # the space group (S5: P1 kept)
        assert np.allclose(g[3], m[3], atol=1e-6), (name, g[3], m[3])
        assert same_atoms(g[4], m[4], translate=name.endswith("_5L")), (name, len(g[4]), len(m[4]))
        spin = any(l.strip() == "SPIN" for l in g[1])
        assert spin == bool(json.loads(cfg.read_text())["configuration"].get("spin_polarized")), (name, cfg.name)


def test_symmetrized_save_refused_for_a_supercell(served):
    """A symmetrized CIF writes spglib's refined cell; for a supercell that is
    the space group's smaller conventional cell (MgO 2x2x2 would come back as
    the 8-atom cell, a different calculation), so Save As refuses it and says
    to save P1. A cell that is not a supercell symmetrizes as before, and its
    deck keeps the P1 route's keyword lines."""
    base, root, tmp = served
    (root / "built").mkdir(exist_ok=True)
    kind, params = GUI["MgO_2x2x2"]
    st, _, r = U.post(base, f"api/build/{kind}", {"source": {"path": "structures/MgO.cif"}, "params": params, "open_as": {"name": "x"}})
    st, _, out = U.post(base, "api/structure/save", {"session": r["session"]["session"], "path": "built/MgO_222_sym.cif",
                                                     "format": "cif", "cif_mode": "symmetrized", "overwrite": True})
    assert st == 400 and "Save as CIF (P1)" in out["error"] and "64 atoms, 8 in that cell" in out["error"], out
    assert not (root / "built" / "MgO_222_sym.cif").exists()
    s = U.req(base, "api/structure?path=structures/MgO.cif")[2]
    paths = {}
    for mode in ("p1", "symmetrized"):
        st, _, out = U.post(base, "api/structure/save", {"open": {"path": "structures/MgO.cif", "expected": s["file"]},
                                                         "path": f"MgO_{mode}.cif", "format": "cif", "cif_mode": mode, "overwrite": True})
        assert st == 200, out
        paths[mode] = f"MgO_{mode}.cif"
    a, b = split_deck(gui_deck(base, paths["p1"], CONFIG)), split_deck(gui_deck(base, paths["symmetrized"], CONFIG))
    # cif2d12 sizes the k-mesh and the integral buffers from the symmetry it
    # is given (P1: an anisotropic SHRINK and bigger buffers), so those lines
    # differ by design; the method, basis, SCF and OPT lines may not
    assert symmetry_free(a[1]) == symmetry_free(b[1]), (a[1], b[1])
    assert b[2] == "225" and a[2] == "1"


def test_saved_cif_equals_export(served):
    """The saved CIF equals Export Structure > CIF (P1) of the same document, byte for byte."""
    base, root, tmp = served
    path, sid = gui_saved(base, root, "MgO_r2xr2x1")
    st, h, body = U.post(base, "api/structure/export", {"source": {"session": sid}, "format": "cif", "cif_mode": "p1"})
    assert st == 200
    raw = body if isinstance(body, (bytes, bytearray)) else str(body).encode()
    assert (root / path).read_bytes() == bytes(raw)
