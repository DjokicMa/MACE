"""Animation export (1.1.6): mace/gui/animexport.py and `mace animate` (run through mace_cli).

* GIF: one palette for every frame, the frame rate, the loop count (forever, once, n plays) and a
  transparent background, read back with Pillow.
* WebM and MP4 through ffmpeg when it is on PATH; without it the error names ffmpeg and GIF still works.
* Frames from real CRYSTAL23 outputs (test/ corpus): a FREQ run's mode (one cycle, the GUI's amplitude
  rule: the physical zero-point amplitude CRYSTAL prints by default, the largest displacement 0.5 A with
  --amplitude-scale normalized), an optimization's steps; an MD/NEB .extxyz through ASE.
* The print patterns: the same assignment rule as the GUI (js/viewport/print_patterns.js), and a
  patterned frame differs from a plain one.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import REPO_ROOT
from mace.gui import animexport as A

Image = pytest.importorskip("PIL.Image")
CORPUS = REPO_ROOT / "test"
FREQ = CORPUS / "FREQ" / "EC_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized_freq_HSESOL3C_optimized.out"
OPT = CORPUS / "OPT" / "2_dia2_opt_rev1.out"


def frames(n=6, size=(64, 48), alpha=False):
    out = []
    for k in range(n):
        im = Image.new("RGBA", size, (255, 255, 255, 0 if alpha else 255))
        from PIL import ImageDraw
        ImageDraw.Draw(im).ellipse([4 + 6 * k, 10, 24 + 6 * k, 30], fill=(200, 40, 40, 255))
        out.append(im)
    return out


@pytest.mark.parametrize("loop, want", [(0, 0), (1, None), (3, 2)])
def test_gif_frames_rate_and_loop_count(tmp_path, loop, want):
    r = A.encode(frames(), tmp_path / "a.gif", "gif", fps=10, loop=loop)
    im = Image.open(tmp_path / "a.gif")
    assert im.n_frames == 6 and r["frames"] == 6 and r["size"] == [64, 48]
    assert im.info["duration"] == 100
    assert im.info.get("loop") == want


def test_gif_transparent_background(tmp_path):
    A.encode(frames(alpha=True), tmp_path / "t.gif", "gif", fps=20, transparent=True)
    im = Image.open(tmp_path / "t.gif")
    assert im.info.get("transparency") == 255
    rgba = im.convert("RGBA")
    assert rgba.getpixel((60, 2))[3] == 0            # the background
    assert rgba.getpixel((14, 20))[3] == 255         # the disc


def test_bad_requests_say_why(tmp_path):
    with pytest.raises(A.AnimError, match="format"):
        A.encode(frames(), tmp_path / "x.avi", "avi")
    with pytest.raises(A.AnimError, match="frame rate"):
        A.encode(frames(), tmp_path / "x.gif", "gif", fps=500)
    with pytest.raises(A.AnimError, match="no frames"):
        A.encode([], tmp_path / "x.gif", "gif")


def test_video_without_ffmpeg_names_it(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "ffmpeg_path", lambda: None)
    caps = A.capabilities()
    assert caps["gif"] and not caps["webm"] and not caps["mp4"] and "ffmpeg" in caps["why"]
    with pytest.raises(A.AnimError, match="ffmpeg"):
        A.encode(frames(), tmp_path / "x.mp4", "mp4")


@pytest.mark.skipif(not A.ffmpeg_path(), reason="ffmpeg not on PATH")
@pytest.mark.parametrize("fmt, magic", [("mp4", b"ftyp"), ("webm", b"\x1aE\xdf\xa3")])
def test_video_with_ffmpeg(tmp_path, fmt, magic):
    if not A.capabilities()[fmt]:
        pytest.skip(f"this ffmpeg has no {fmt} encoder")
    r = A.encode(frames(size=(65, 49)), tmp_path / f"v.{fmt}", fmt, fps=12, transparent=fmt == "webm")
    head = (tmp_path / f"v.{fmt}").read_bytes()[:16]
    assert magic in head and r["bytes"] > 0


def test_pattern_rule_matches_the_gui():
    col = {"A": (128, 128, 128), "B": (130, 130, 130), "C": (32, 32, 32), "D": (133, 131, 129)}
    assert A.assign_patterns(["A", "A", "B", "C"], col.get, "auto") == {"B": "slash"}
    assert A.assign_patterns(["A", "A", "B", "D", "C"], col.get, "auto") == {"B": "slash", "D": "cross"}
    assert A.assign_patterns(["A", "C"], col.get, "all") == {"A": "slash", "C": "cross"}
    assert A.assign_patterns(["A", "B"], col.get, "off") == {}
    # the same pattern order as the page
    js = (REPO_ROOT / "mace/gui/static/js/viewport/print_patterns.js").read_text()
    assert 'PATTERNS = ["' + '", "'.join(A.PATTERNS) + '"]' in js


def test_element_legend_swatches_carry_the_atoms_patterns():
    """The element legend (made server-side, mace/utils/element_legend.py) draws the patterns the GUI puts
    on the atoms: an SVG <pattern> tile per element over its swatch, inside the legend's own group (the
    GUI moves that group into an exported structure SVG), and the same marks in the PNG."""
    import io
    import xml.etree.ElementTree as ET
    from mace.utils import element_legend as EL
    assert EL.PATTERNS == A.PATTERNS
    S = "{http://www.w3.org/2000/svg}"
    pats = {"Mg": "slash", "O": "cross"}
    for shape in ("circle", "square"):
        root = ET.fromstring(EL.legend(symbols=["Mg", "O"], fmt="svg", patterns=pats, options={"shape": shape}).content)
        group = root.find(f"{S}g[@id='element-legend']")
        tiles = {p.get("id") for p in group.iter(f"{S}pattern")}
        assert {t.rsplit("-", 1)[0] for t in tiles} == {"lg-pp-slash", "lg-pp-cross"}, tiles
        for sym, pid in pats.items():
            fills = [e.get("fill") for e in root.find(f".//{S}g[@id='legend-{sym}']") if e.tag != f"{S}text"]
            assert fills[0].startswith("#") and len(fills) == 2 and fills[1].startswith(f"url(#lg-pp-{pid}-"), fills
            assert fills[1][5:-1] in tiles
    # the ink as on the atoms: dark on a light swatch, light on a dark one
    light = EL.legend(symbols=["H"], fmt="svg", patterns={"H": "dots"}, colors={"H": "#ffffff"}).content.decode()
    dark = EL.legend(symbols=["H"], fmt="svg", patterns={"H": "dots"}, colors={"H": "#202020"}).content.decode()
    assert 'id="lg-pp-dots-d"' in light and 'fill="rgb(0,0,0)" fill-opacity="0.62"' in light
    assert 'id="lg-pp-dots-l"' in dark and 'fill="rgb(255,255,255)" fill-opacity="0.78"' in dark
    # no patterns: the legend as before
    plain = EL.legend(symbols=["Mg", "O"], fmt="svg").content.decode()
    assert "<pattern" not in plain and "<defs>" not in plain
    # the PNG: the marks drawn inside the swatch, nothing outside it
    on = Image.open(io.BytesIO(EL.legend(symbols=["O"], fmt="png", patterns={"O": "grid"}, options={"dpi": 96}).content))
    off = Image.open(io.BytesIO(EL.legend(symbols=["O"], fmt="png", options={"dpi": 96}).content))
    a, b = np.asarray(on.convert("RGBA"), float), np.asarray(off.convert("RGBA"), float)
    diff = np.argwhere(np.abs(a - b).sum(axis=2) > 30)
    assert len(diff) > 10
    s = 18 * 1.0                                   # the swatch (px at 96 dpi), from x = y = pad = 9
    assert diff[:, 0].min() >= 9 - 1 and diff[:, 0].max() <= 9 + s and diff[:, 1].max() <= 9 + s, diff
    with pytest.raises(ValueError, match="patterns"):
        EL.legend(symbols=["O"], fmt="svg", patterns={"O": "zigzag"})
    with pytest.raises(ValueError, match="patterns"):
        EL.legend(symbols=["O"], fmt="svg", patterns={"Xx": "slash"})


def _xyz(path, n=4):
    lines = []
    for k in range(n):
        lines += ["3", 'Lattice="6 0 0 0 6 0 0 0 6" Properties=species:S:1:pos:R:3 pbc="T T T"']
        lines += [f"O 1.0 1.0 {1.0 + 0.1 * k:.3f}", "H 1.8 1.0 1.0", "H 1.0 1.8 1.0"]
    path.write_text("\n".join(lines) + "\n")


def test_trajectory_file_through_ase_renders_every_frame(tmp_path):
    pytest.importorskip("ase")
    p = tmp_path / "neb.extxyz"
    _xyz(p, 4)
    src = A.frames_from_path(p)
    assert src["kind"] == "trajectory" and len(src["frames"]) == 4
    imgs = A.render_frames(src, size=(120, 90), patterns="all", grayscale=True)
    plain = A.render_frames(src, size=(120, 90), patterns="off", grayscale=True)
    assert [i.size for i in imgs] == [(120, 90)] * 4
    assert np.asarray(imgs[0]).tobytes() != np.asarray(plain[0]).tobytes()
    # the frames differ (the O atom moves) and the camera is one for all of them
    assert np.asarray(imgs[0]).tobytes() != np.asarray(imgs[-1]).tobytes()


def _corpus(p):
    if not p.is_file():
        pytest.skip("test/ corpus not present (gitignored)")


def _turning_point(src):
    # sin(2 pi f / 24): frame 0 the rest geometry, frame 6 the turning point
    X = np.array([f["cart"] for f in src["frames"]])
    assert np.allclose(X[12], X[0], atol=1e-9)
    return float(np.linalg.norm(X[6] - X[0], axis=1).max())


def test_freq_mode_frames_follow_the_gui_rule():
    _corpus(FREQ)
    src = A.frames_from_path(FREQ, mode=10, frames=24, amplitude=1.0, amplitude_scale="normalized")
    assert src["kind"] == "freq" and src["mode"]["mode"] == 10 and len(src["frames"]) == 24
    # Fit 0.5 A: the largest displacement 0.5 A
    d6 = _turning_point(src)
    assert abs(d6 - 0.5) < 1e-6, d6
    with pytest.raises(A.AnimError, match="no mode"):
        A.frames_from_path(FREQ, mode=9999)


def test_freq_mode_frames_default_to_the_physical_amplitude():
    """As the GUI (freq.amplitudeScale "physical"): the largest atom moves its printed amplitude."""
    _corpus(FREQ)
    from mace.gui import outindex, freq as gui_freq
    idx = outindex.build(FREQ)
    parsed = gui_freq.parse_job(None, path=str(FREQ), start=idx.freq[0], end=idx.freq[1])
    printed = np.asarray(parsed["displacements"][10], float) * gui_freq.BOHR
    top = float(np.linalg.norm(printed, axis=1).max())
    assert 0.001 < top < 0.3, top                       # a zero-point amplitude: well under 0.5 A
    d6 = _turning_point(A.frames_from_path(FREQ, mode=10, frames=24))
    assert abs(d6 - top) < 1e-4, (d6, top)
    d6 = _turning_point(A.frames_from_path(FREQ, mode=10, frames=24, amplitude=10))
    assert abs(d6 - 10 * top) < 1e-3, (d6, top)


def test_render_route_takes_the_views_scale():
    """File > Export Animation sends the FREQ view's scale; Physical runs to x100."""
    assert A._render_opts({})["amplitude_scale"] == "physical"
    assert A._render_opts({"amplitude": 50})["amplitude"] == 50
    assert A._render_opts({"amplitude": 2, "amplitude_scale": "normalized"})["amplitude_scale"] == "normalized"
    with pytest.raises(A.AnimError, match="amplitude runs"):
        A._render_opts({"amplitude": 50, "amplitude_scale": "normalized"})
    with pytest.raises(A.AnimError, match="amplitude_scale is one of"):
        A._render_opts({"amplitude_scale": "fit"})


def _mace(*args, cwd):
    return subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), *args], capture_output=True, text=True,
                          timeout=600, cwd=str(cwd), env={**os.environ, "MPLBACKEND": "Agg"})


def test_mace_animate_cli(tmp_path):
    r = _mace("animate", "--check", cwd=tmp_path)
    assert r.returncode == 0 and "GIF:  yes" in r.stdout
    r = _mace("animate", "--help", cwd=tmp_path)
    assert r.returncode == 0 and "mace animate" in r.stdout
    flat = " ".join(r.stdout.split())        # help lines wrap at the terminal width
    assert "--amplitude-scale {physical,normalized}" in flat and "physical zero-point amplitude" in flat
    _corpus(FREQ)
    r = _mace("animate", str(FREQ), "--mode", "10", "--size", "200x150", "-o", str(tmp_path / "m.gif"), cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    im = Image.open(tmp_path / "m.gif")
    assert im.n_frames == 24 and im.size == (200, 150)
    # the physical amplitude by default (as the GUI); --amplitude-scale normalized is the 0.5 A fit
    r = _mace("animate", str(FREQ), "--mode", "10", "--size", "200x150", "--amplitude-scale", "physical",
              "-o", str(tmp_path / "p.gif"), cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    r = _mace("animate", str(FREQ), "--mode", "10", "--size", "200x150", "--amplitude-scale", "normalized",
              "-o", str(tmp_path / "n.gif"), cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "m.gif").read_bytes() == (tmp_path / "p.gif").read_bytes()
    assert (tmp_path / "m.gif").read_bytes() != (tmp_path / "n.gif").read_bytes()
    r = _mace("animate", str(FREQ), "--amplitude-scale", "fit", cwd=tmp_path)
    assert r.returncode == 2 and "physical" in r.stderr and "normalized" in r.stderr
    r = _mace("animate", str(FREQ), "--list-modes", cwd=tmp_path)
    assert r.returncode == 0 and "Mode" in r.stdout and "\n    10 " in r.stdout
    _corpus(OPT)
    r = _mace("animate", str(OPT), "--size", "160x120", "--grayscale", "-o", str(tmp_path / "o.gif"), cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert Image.open(tmp_path / "o.gif").n_frames == 7
    r = _mace("animate", str(OPT), "--source", "freq", cwd=tmp_path)
    assert r.returncode == 1 and "no normal modes" in r.stderr


def test_an_animation_too_many_pixels_is_refused_before_it_is_drawn(tmp_path, monkeypatch):
    """Every frame is held as RGBA until it is encoded: frames x width x height is capped (the server ran out
    of memory), with words that point at Every and Size."""
    with pytest.raises(A.AnimError, match="every n-th frame.*smaller"):
        A._render_opts({"mode": 7, "frames": 240, "width": 4096, "height": 4096})
    assert A._render_opts({"mode": 7, "frames": 24, "width": 1920, "height": 1080})["frames"] == 24
    pytest.importorskip("ase")
    p = tmp_path / "md.extxyz"
    _xyz(p, 8)
    src = A.frames_from_path(p)
    monkeypatch.setattr(A, "MAX_PIXELS", 8 * 120 * 90 - 1)
    with pytest.raises(A.AnimError, match="--every"):
        A.render_frames(src, size=(120, 90))
    assert len(A.frames_from_path(p, every=2)["frames"]) == 4
    assert len(A.render_frames(A.frames_from_path(p, every=2), size=(120, 90))) == 4


def test_captured_frames_are_capped_in_pixels(monkeypatch):
    """anim/frame: the GUI's captured frames add up to at most MAX_PIXELS (a frame sent again replaces its own)."""
    import base64
    import io
    from types import SimpleNamespace
    buf = io.BytesIO()
    Image.new("RGBA", (40, 30)).save(buf, "PNG")
    data = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    monkeypatch.setattr(A, "MAX_PIXELS", 3 * 40 * 30)

    class H:
        state = SimpleNamespace(cache=None)
        headers = {}

        def _json(self, obj, status=200):
            return obj
    h = H()
    key = A.post_route(h, "anim/start", {})["key"]
    try:
        for k in range(3):
            assert A.post_route(h, "anim/frame", {"key": key, "index": k, "data": data})["n"] == k + 1
        assert A.post_route(h, "anim/frame", {"key": key, "index": 1, "data": data})["n"] == 3
        with pytest.raises(A.AnimError, match="billion pixels"):
            A.post_route(h, "anim/frame", {"key": key, "index": 3, "data": data})
    finally:
        A.post_route(h, "anim/cancel", {"key": key})


# ---------------------------------------------------------------- the bonds drawn: the viewer's rule (1.1.6)
CORE_JS = REPO_ROOT / "mace" / "gui" / "static" / "js" / "geom" / "core.js"
DRAWN = """import(process.argv[1]).then((G) => {
  const {s, cfg} = JSON.parse(process.argv[2]);
  const defaults = {bondTolerance: 1.15, bondMin: 0.1, bondMetalMetal: false, bondPairs: {}};
  const rule = G.bondRule(s, (k) => (k in cfg ? cfg[k] : defaults[k]));
  const b = G.cellListBonds(Float64Array.from(s.cart.flat()), Float32Array.from(s.radii), Uint8Array.from(s.metal.map(Number)),
    rule.tol, rule.metalMetal, G.ruleFor(rule));
  const out = [];
  for (let k = 0; k < b.length; k += 2) out.push([Math.min(b[k], b[k + 1]), Math.max(b[k], b[k + 1])]);
  console.log(JSON.stringify(out));
});"""


def _viewer_bonds(atoms, **cfg):
    """The bonds geom/core.js draws between these atoms (the viewer's cellListBonds under its bond rule)."""
    import json
    import shutil
    from mace.gui import structures as S
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    p = S._payload(atoms.cell.array, atoms.get_scaled_positions(wrap=False), atoms.positions, atoms.numbers,
                   (True,) * 3, {"kind": "test"}, symmetry_data={"available": False})
    s = {k: p[k] for k in ("cart", "radii", "metal", "numbers", "symbols", "pbc")}
    out = subprocess.run([node, "--input-type=module", "-e", DRAWN, CORE_JS.as_uri(), json.dumps({"s": s, "cfg": cfg})],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return sorted(tuple(x) for x in json.loads(out.stdout))


@pytest.mark.parametrize("name", ["MgO", "AgCl", "MgO never Mg-O"])
def test_exported_bonds_are_the_viewers(name):
    """Animation export draws the viewer's bonds between its atoms: MgO without Mg-Mg (2.98 Å, inside
    1.15 x the covalent sum, which the export drew before), AgCl's Ag-Cl at 2.89 Å (a heavy cation's
    bond radius; the covalent sum drew none) and the document's own criteria."""
    pytest.importorskip("ase")
    from ase.build import bulk
    from mace.gui import structures as S
    atoms = bulk("AgCl" if name == "AgCl" else "MgO", "rocksalt", a=5.78 if name == "AgCl" else 4.212, cubic=True)
    cfg = {"bondPairs": {"Mg-O": {"never": True}}} if "never" in name else {}
    got = sorted(A.frame_bonds(atoms.positions, atoms.numbers, S.bond_rule(cfg) if cfg else None))
    assert got == _viewer_bonds(atoms, **cfg)
    sym = atoms.get_chemical_symbols()
    kinds = {"-".join(sorted((sym[i], sym[j]))) for i, j in got}
    assert kinds == {"MgO": {"Mg-O"}, "AgCl": {"Ag-Cl"}, "MgO never Mg-O": set()}[name], kinds


def test_render_draws_the_frame_bonds(monkeypatch):
    """render_frames draws exactly frame_bonds' pairs, with the document's criteria it is given."""
    pytest.importorskip("ase")
    from ase.build import bulk
    from mace.gui import structures as S
    atoms = bulk("MgO", "rocksalt", a=4.212, cubic=True)
    src = {"kind": "structure", "frames": [A._frame(atoms.numbers, atoms.positions, atoms.cell.array, (True,) * 3)]}
    seen = []
    real = A.frame_bonds
    monkeypatch.setattr(A, "frame_bonds", lambda cart, numbers, rule=None: seen.append(rule) or real(cart, numbers, rule))
    rule = S.bond_rule({"bondTolerance": 1.3})
    with_bonds = A.render_frames(src, size=(120, 90), bond_rule=rule)
    assert seen == [rule]
    plain = A.render_frames(src, size=(120, 90), bonds=False)
    assert np.asarray(with_bonds[0]).tobytes() != np.asarray(plain[0]).tobytes()
    assert A._render_opts({"bonds": {"bondTolerance": 1.3}})["bond_rule"] == rule
    assert A._render_opts({})["bond_rule"] is None
