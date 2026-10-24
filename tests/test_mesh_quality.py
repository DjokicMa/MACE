"""Mesh quality presets (1.1.6, mace/analysis/mesh_quality.py and
static/js/viewport/mesh_quality.js): one set of numbers for the server and the
page, Standard identical to the surfaces before presets, finer presets give
more triangles (pores) and denser grids (cubes), knobs stay inside their hard
limits, and a porosity result is re-meshed without its analysis."""
import base64
import json
import re

import numpy as np
import pytest

from conftest import REPO_ROOT
from mace.analysis import mesh_quality as MQ

JS = (REPO_ROOT / "mace" / "gui" / "static" / "js" / "viewport" / "mesh_quality.js").read_text()


def _js_table(name):
    body = re.search(rf"export const {name} = (\{{.*?\n\}});", JS, re.S).group(1)
    body = re.sub(r"(\w+):", r'"\1":', body).replace("null", "null").replace("undefined", "null")
    return json.loads(re.sub(r",(\s*[}\]])", r"\1", body))


def test_page_and_server_presets_agree():
    assert re.search(r'export const PRESETS = \["draft", "standard", "high", "publication"\]', JS)
    assert MQ.PRESETS == ("draft", "standard", "high", "publication") and MQ.DEFAULT == "standard"
    assert _js_table("PORES") == {k: {kk: (int(vv) if float(vv).is_integer() else vv) for kk, vv in v.items()} for k, v in MQ.PORES.items()}
    js_vol = _js_table("VOLUME")
    for q, v in MQ.VOLUME.items():
        assert {k: js_vol[q][k] for k in ("points", "upsample")} == v, q
    lim = re.search(r"export const PORE_LIMITS = (\{.*?\});", JS).group(1)
    assert json.loads(re.sub(r"(\w+):", r'"\1":', lim)) == {k: list(v) for k, v in MQ.PORE_LIMITS.items()}
    orb = re.search(r"export const ORBITAL = (\{.*?\});", JS).group(1)
    assert json.loads(re.sub(r"(\w+):", r'"\1":', orb)) == MQ.ORBITAL


def test_knobs_take_the_preset_and_clamp_hand_values():
    assert MQ.pores(None) == {"quality": "standard", "custom": False, **MQ.PORES["standard"]}
    assert MQ.pores({"quality": "nonsense"})["quality"] == "standard"
    k = MQ.pores({"quality": "draft", "grid": 10_000, "max_triangles": 5, "smooth": "1.5"})
    assert k["grid"] == 160 and k["max_triangles"] == 20_000 and k["smooth"] == 1.5 and k["custom"]
    with pytest.raises(ValueError):
        MQ.pores({"grid": "lots"})
    assert MQ.volume({"quality": "standard"}, legacy="low")["points"] == 250_000
    assert MQ.volume({"quality": "publication", "points": 9e9})["points"] == 8_000_000
    # finer presets never ask for less
    for a, b in zip(MQ.PRESETS, MQ.PRESETS[1:]):
        for key in ("grid", "upsample", "max_triangles"):
            assert MQ.PORES[a][key] <= MQ.PORES[b][key]


def test_pore_surfaces_have_a_preview_tier_and_their_own_default():
    # Preview: below Draft, the pore surfaces only (a volume asked for it gets Standard)
    assert MQ.PORE_PRESETS == ("preview",) + MQ.PRESETS
    assert re.search(r'export const PORE_PRESETS = \["preview", \.\.\.PRESETS\]', JS)
    assert MQ.pores({"quality": "preview"}) == {"quality": "preview", "custom": False, **MQ.PORES["preview"]}
    assert MQ.volume({"quality": "preview"})["quality"] == "standard" and "preview" not in MQ.VOLUME
    for key in ("grid", "upsample", "max_triangles"):
        assert MQ.PORES["preview"][key] <= MQ.PORES["draft"][key]
    # the GUI draws them at their own setting (pores.quality), not the shared mesh.quality
    default = re.search(r'export const PORE_DEFAULT = "(\w+)";', JS).group(1)
    assert default in MQ.PORE_PRESETS and default != MQ.DEFAULT
    settings_js = (REPO_ROOT / "mace" / "gui" / "static" / "js" / "settings.js").read_text()
    assert re.search(rf'"pores\.quality": "{default}"', settings_js)
    assert re.search(r'"pores\.quality": \["preview", "draft", "standard", "high", "publication"\]', settings_js)
    schema = (REPO_ROOT / "mace" / "gui" / "static" / "js" / "lazy" / "settings_schema.js").read_text()
    assert '"pores.quality", "Pore surface mesh quality"' in schema


@pytest.fixture(scope="module")
def lta():
    pytest.importorskip("scipy")
    ase = pytest.importorskip("ase.spacegroup")
    from mace.analysis.structure import from_atoms
    at = ase.crystal(["O", "O", "O", "Si"], basis=[(0, 0.2122, 0.5), (0.1103, 0.1103, 0.3384), (0, 0.2967, 0.2967), (0, 0.1823, 0.3684)],
                     spacegroup=221, cellpar=[11.919] * 3 + [90] * 3)
    return from_atoms(at, "LTA")


@pytest.fixture(scope="module")
def analysed(lta):
    from mace.analysis import porosity
    return porosity.analyze(lta, probe="N2", spacing=0.4, render=True, keep_remesh=True)


def test_standard_is_the_surface_before_presets(lta, analysed):
    from mace.analysis import porosity
    old = porosity.analyze(lta, probe="N2", spacing=0.4, render=True)["render"]["meshes"]
    new = analysed["render"]["meshes"]
    for kind in ("accessible", "inaccessible"):
        if old.get(kind):
            assert new[kind]["positions"] == old[kind]["positions"] and new[kind]["faces"] == old[kind]["faces"]
    assert new["quality"]["quality"] == "standard" and new["quality"]["triangles"] > 0
    again = analysed["_remesh"]({"quality": "standard"})["meshes"]
    assert again["accessible"]["positions"] == new["accessible"]["positions"]


def test_finer_presets_give_more_triangles(analysed):
    tri = {q: analysed["_remesh"]({"quality": q})["meshes"]["quality"]["triangles"] for q in MQ.PORE_PRESETS}
    assert tri["preview"] < tri["draft"] <= tri["standard"] < tri["high"] < tri["publication"], tri
    # a hand knob wins over the preset and is reported as custom
    m = analysed["_remesh"]({"quality": "draft", "max_triangles": 20_000})["meshes"]
    assert m["quality"]["custom"] and m["quality"]["triangles"] <= tri["draft"]


def test_the_gui_route_remeshes_without_the_analysis(lta, monkeypatch):
    from mace.gui import analysis_routes as R
    from mace.analysis import porosity
    params = {"probe": "N2", "spacing": "0.4"}
    R._porosity(lta, params, None, None, keep=True)
    monkeypatch.setattr(porosity, "analyze", lambda *a, **k: pytest.fail("the analysis ran again"))
    out = R.remesh(lta, params, {"quality": "draft"})
    assert out["meshes"]["quality"]["quality"] == "draft"
    P = np.frombuffer(base64.b64decode(out["meshes"]["accessible"]["positions"]), "<f4")
    assert P.size and np.isfinite(P).all()


def _cube(path, n=16):
    r = (np.indices((n, n, n)).transpose(1, 2, 3, 0) - n / 2) * 0.3
    data = np.exp(-(r ** 2).sum(-1))
    with open(path, "w") as f:
        f.write(" Charge density - 3D GRID - GAUSSIAN CUBE FORMAT ./mol\n")
        f.write("     0.00000000     0.00000000     0.00000000     0.000000   0.000000   0.000000\n")
        f.write(f"    1 {-n * 0.15:12.6f} {-n * 0.15:12.6f} {-n * 0.15:12.6f}\n")
        for i in range(3):
            v = [0.0, 0.0, 0.0]
            v[i] = 0.3
            f.write(f"{n:5d} {v[0]:12.6f} {v[1]:12.6f} {v[2]:12.6f}\n")
        f.write("    8     8.000000     0.000000     0.000000     0.000000\n")
        flat = data.ravel()
        for i in range(0, len(flat), 6):
            f.write("".join(f"{v:13.5E}" for v in flat[i:i + 6]) + "\n")
    return path


def test_cube_grids_per_preset(tmp_path):
    pytest.importorskip("scipy")
    from mace.gui import volume as V
    c = V.read(str(_cube(tmp_path / "mol_DENS.CUBE")))
    std, _ = V.decimate(c, "medium")
    g, meta = V.decimate(c, "publication")
    # Publication refines a grid under its budget twice by cubic interpolation, corners kept
    assert g.shape == tuple((n - 1) * 2 + 1 for n in std.shape) and meta["upsample"] == 2
    assert np.allclose(g[::2, ::2, ::2], std, atol=1e-5)
    assert np.allclose(np.asarray(meta["voxels"]) * 2, V.decimate(c, "medium")[1]["voxels"])
    # a hand budget is held inside the hard limits
    assert V.decimate(c, "low", points=1000)[1]["points_budget"] == MQ.VOLUME_LIMITS["points"][0]
    assert V.mesh_knobs("medium") == {"points": 1_000_000, "upsample": 1}
    with pytest.raises(ValueError):
        V.mesh_knobs("ultra")
