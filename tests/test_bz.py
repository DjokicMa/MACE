"""The first Brillouin zone (Crystal_d3/d3_bz.py) for mace plotting --bz and the mace gui view,
and band plots along an edited k-path (mace plotting --band-path).

Each zone is checked against what it must be: its volume is the reciprocal cell's, its
vertices are where the perpendicular bisectors of the nearest lattice vectors meet, the counts
are the textbook polyhedra (cube, truncated octahedron, rhombic dodecahedron, hexagonal
prism, hexagon), and the high-symmetry points lie in it. The path's points are seekpath's."""
import json
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "Crystal_d3"))
sys.path.insert(0, str(REPO_ROOT / "Plotting"))
import d3_band_path as kp  # noqa: E402
import d3_bz as bz  # noqa: E402

pytest.importorskip("scipy")

A = 3.0
CELLS = {
    # name: (cell rows in Angstrom, dim, vertices, faces, edges)
    "cubic": (np.eye(3) * A, 3, 8, 6, 12),
    "fcc": (A / 2 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]), 3, 24, 14, 36),
    "bcc": (A / 2 * np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]), 3, 14, 12, 24),
    "hexagonal": ([[2.46, 0, 0], [-1.23, 2.130422, 0], [0, 0, 6.7]], 3, 12, 8, 18),
    # graphiteABslip's base-centered orthorhombic (oC2) primitive cell, as CRYSTAL prints it
    "oC2 (graphite AB slip)": ([[3.67981, 3.11238, 0], [-3.67981, 3.11238, 0], [0, 0, 4.24648]], 3, 12, 8, 18),
    "hexagonal slab (2D)": ([[2.46, 0, 0], [-1.23, 2.130422, 0], [0, 0, 500]], 2, 6, 1, 6),
}


def _area(poly):
    return 0.5 * np.linalg.norm(sum(np.cross(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))))


@pytest.mark.parametrize("name", list(CELLS))
def test_zone_shape_volume_and_bisectors(name):
    from scipy.spatial import ConvexHull
    cell, dim, nv, nf, ne = CELLS[name]
    z = bz.brillouin_zone(cell, dim)
    V, rec = np.array(z["vertices"]), np.array(z["recip"])
    assert (len(V), len(z["faces"]), len(z["edges"])) == (nv, nf, ne)
    if dim == 3:
        assert ConvexHull(V).volume == pytest.approx(abs(np.linalg.det(rec)), rel=1e-9)
    else:
        assert _area(V) == pytest.approx(np.linalg.norm(np.cross(rec[0], rec[1])), rel=1e-9)
        assert np.allclose(V[:, 2], 0)
    # every vertex is in the zone and on at least `dim` bisector planes
    for v in V:
        assert bz.inside(v, rec, dim)
        on = sum(1 for n in np.ndindex(*(5,) * dim) if any(np.array(n) - 2)
                 and abs(v @ (g := (np.array(n) - 2) @ rec[:dim]) - g @ g / 2) < 1e-6)
        assert on >= dim
    assert np.allclose(np.array(z["recip"]) @ np.array(cell).T, 2 * np.pi * np.eye(3))


def test_polymer_zone_is_a_segment():
    z = bz.brillouin_zone([[2.5, 0, 0], [0, 20, 0], [0, 0, 20]], 1)
    assert np.allclose(z["vertices"], [[-np.pi / 2.5, 0, 0], [np.pi / 2.5, 0, 0]]) and z["edges"] == [[0, 1]]


def test_oc2_graphite_slip_points_against_its_zone():
    """graphiteABslip's deck (SeeKPath, oC2, zeta = 389677/851984): DELTA_0, B_0, Z, R, S lie in the
    zone of the cell CRYSTAL printed; Y and F_0 as that deck writes them, (1/2, 1/2, 0) and
    (zeta, 1-zeta, 0), lie outside it (equivalent states, by b1, but the segments Gamma-Y-F_0 leave
    the zone), which the views report point by point. (Y - b1 and F_0 - b2 are on it.)"""
    cell = CELLS["oC2 (graphite AB slip)"][0]
    rec = bz.reciprocal(cell)
    zeta = 389677 / 851984
    inside = {"DELTA_0": [-zeta, zeta, 0], "B_0": [-zeta, zeta, 0.5], "Z": [0, 0, 0.5], "S": [0, 0.5, 0],
              "R": [0, 0.5, 0.5], "GAMMA": [0, 0, 0]}
    outside = {"Y": [0.5, 0.5, 0], "F_0": [zeta, 1 - zeta, 0]}
    for lab, k in inside.items():
        assert bz.inside(np.array(k) @ rec, rec), lab
    for lab, k in outside.items():
        c = np.array(k) @ rec
        assert not bz.inside(c, rec), lab
        assert any(bz.inside(c - g, rec) for g in (rec[0], rec[1], -rec[0], -rec[1])), lab   # one reciprocal vector over


def test_path_points_are_seekpaths_and_in_the_zone():
    """Si: the SeeK-path deck's labels and coordinates are seekpath's own points (CRYSTAL basis)."""
    pytest.importorskip("seekpath")
    out = REPO_ROOT / "examples" / "gui" / "outputs" / "Si_sp.out"
    deck = kp.parse_band_deck((REPO_ROOT / "tests" / "fixtures" / "band_path_golden" / "si_seekpath.d3").read_text())
    sp = bz.seekpath_points(str(out))
    view = bz.bz_view(str(out), deck["segments"])
    rec = view["bz"]["recip"]
    for s in deck["segments"]:
        for end in ("from", "to"):
            lab = kp.canonical_label(s[end])
            assert lab in sp and np.allclose(sp[lab], s[f"k_{end}"]), lab
            assert view["points"][lab]["on_path"] and view["points"][lab]["inside"] and bz.inside(view["points"][lab]["cart"], rec)
    assert {lab for lab, p in view["points"].items() if not p["on_path"]} == set(sp) - {kp.canonical_label(x) for s in deck["segments"] for x in (s["from"], s["to"])}
    assert len(view["path"]) == len(deck["segments"])


def test_slab_view_is_two_dimensional():
    out = REPO_ROOT / "examples" / "gui" / "outputs" / "graphene_slab_sp.out"
    deck = kp.parse_band_deck((REPO_ROOT / "tests" / "fixtures" / "band_path_golden" / "graphene_slab_seekpath.d3").read_text())
    view = bz.bz_view(str(out), deck["segments"])
    assert view["bz"]["dim"] == 2 and len(view["bz"]["vertices"]) == 6
    assert [p for p in view["points"]] == ["GAMMA", "M", "K"]
    for p in view["points"].values():
        assert bz.inside(p["cart"], view["bz"]["recip"], 2)


def test_label_tex():
    assert bz.kp_label_tex("GAMMA") == r"$\Gamma$" and bz.kp_label_tex("F_0") == "$F_{0}$"
    assert bz.kp_label_tex("DELTA_0") == r"$\Delta_{0}$"


# ---------------------------------------------------------------- the CLI
def test_mace_plotting_bz_writes_the_figure(tmp_path):
    shutil.copy(REPO_ROOT / "examples" / "gui" / "outputs" / "Si_sp.out", tmp_path / "Si_sp.out")
    shutil.copy(REPO_ROOT / "tests" / "fixtures" / "band_path_golden" / "si_seekpath.d3", tmp_path / "Si_band.d3")
    edit = kp.reverse(kp.parse_band_deck((tmp_path / "Si_band.d3").read_text())["segments"], 0)
    (tmp_path / "edit.json").write_text(json.dumps({"band_path": [{"from": s["from"], "to": s["to"]} for s in edit]}))
    r = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "--no-banner", "plotting", "--bz", "Si_band.d3",
                        "--band-path", "edit.json", "-o", "figs"], cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp_path / "figs" / "Si.BZ.png").stat().st_size > 1000 and (tmp_path / "figs" / "Si.BZ.svg").is_file()


def _edited_axes(monkeypatch, tmp_path, edit):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    import ipBANDS_V2 as ipb
    from test_gui_results import write_band_path
    path = write_band_path(tmp_path, stem="g_band", path="GAMMA-M-K-GAMMA-A-|-L-M",
                           ticks=(0.0, 0.78265, 1.20908, 2.05632, 2.32554, 2.6))
    (tmp_path / "edit.json").write_text(json.dumps({"band_path": edit}))
    saved = []
    monkeypatch.setattr(Figure, "savefig", lambda self, *a, **k: saved.append(self))
    monkeypatch.setattr(ipb.plt, "close", lambda *a, **k: None)
    args = SimpleNamespace(list_segments=False, segments=None, path=None, no_gaps=False, gap_width=0.02,
                           E_lower=-10, E_upper=10, fixed_width=None, fixed_height=None, alpha=1.0,
                           spin_up_color=ipb.DEFAULT_SPIN_UP_COLOR, spin_down_color=ipb.DEFAULT_SPIN_DOWN_COLOR,
                           auto_width=True, formats="png", band_path=str(tmp_path / "edit.json"))
    ipb.ipBANDS(str(path), "g", {"d3": str(tmp_path / "g_band.d3")}, args, str(tmp_path))
    fig = saved[-1]
    fig.canvas.draw()
    return fig.axes[0], path


def test_band_plot_along_an_edited_path(monkeypatch, tmp_path):
    """--band-path: the order, a reversal (the data flipped), a hidden segment (a jump)."""
    import ipBANDS_V2 as ipb
    edit = [{"from": "M", "to": "GAMMA"}, {"from": "M", "to": "K"}, {"from": "K", "to": "GAMMA", "hidden": True},
            {"from": "L", "to": "M"}, {"from": "GAMMA", "to": "A"}]
    ax, path = _edited_axes(monkeypatch, tmp_path, edit)
    labels = [t.get_text().strip() for t in ax.get_xticklabels()]
    # each jump's two labels share one tick ("Γ|M", the GUI's, 1.1.6)
    assert labels == ["M", "Γ|M", "K|L", "M|Γ", "A"]
    # the first band's first point is the data at M (point index of the M tick)
    E, B, *_ = ipb.read_band_data(str(path))
    m = int(np.argmin(np.abs(E - 0.78265)))
    first = next(ln for ln in ax.get_lines() if len(ln.get_xdata()) > 5)
    assert first.get_ydata()[0] == pytest.approx(B[m, 0])


def test_band_path_must_match_the_data(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="not a segment"):
        _edited_axes(monkeypatch, tmp_path, [{"from": "X", "to": "W"}])
