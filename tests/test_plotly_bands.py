"""mace plotting's plotly band/DOS figure is the GUI's (1.1.6).

* parity: node runs the GUI's figure builder (mace/gui/static/js/docs/bandfig.js, styled by
  plotpresets.js styleFigure) and mace/plotting/plotly_bands.py builds its figure from the same
  data, options and style: the two figure JSONs are equal (traces, colors, dashes, widths, fills,
  axis ranges and titles, k labels, shapes, legends), for every preset, spin color scheme, DOS
  color mode and spin layout, a Fermi shift, a unit, Log Y, an edited k-path;
* the option names: a style chosen in the GUI (a plot's own options over Settings) and the same
  style passed to mace plotting as flags (the GUI's Copy CLI command: plotpresets.js styleFlags,
  presets.py style_flags) give the same figure;
* plotly_bands.TOKENS are tokens.css's light plot tokens.
"""
import copy
import json
import re
import shutil
import subprocess

import pytest

from conftest import REPO_ROOT
from mace.gui import results
from mace.plotting import plotly_bands as PB
from mace.plotting import presets as P

JS = REPO_ROOT / "mace" / "gui" / "static" / "js"
FE = REPO_ROOT / "tests" / "data" / "spin_fe"
CORPUS = REPO_ROOT / "test"
pytestmark = pytest.mark.skipif(not shutil.which("node"), reason="node not installed")

NODE = """
import * as BF from %(bf)s;
import * as PP from %(pp)s;
import { readFileSync } from "node:fs";
const cases = JSON.parse(readFileSync(%(data)s, "utf8"));
const out = cases.map((c) => {
  const css = (n) => (c.tokens[n] ?? "");
  const look = BF.lookOf(c.o, c.base, PP);
  const env = { css, look, tone: "light", paper: c.paper, presets: PP, elementColors: c.ec, width: c.width, marginR: 14, fermiKey: c.fermiKey };
  const band = c.edit ? BF.editedBand(c.band, c.edit) : c.band;
  const fig = BF.bandFigure(band, c.dos, c.o, env);
  PP.styleFigure(fig.traces, fig.layout, "band", PP.resolveId(look.style, "band"), css);
  return { traces: fig.traces, layout: fig.layout, flags: PP.styleFlags(look) };
});
console.log(JSON.stringify(out));
"""


def _norm(x):
    return json.loads(json.dumps(x, allow_nan=False))


def _close(a, b, path="$"):
    """a == b, numbers within 1e-9 (relative)."""
    if isinstance(a, dict) and isinstance(b, dict):
        assert set(a) == set(b), f"{path}: keys {sorted(set(a) ^ set(b))}"
        for k in a:
            _close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            _close(x, y, f"{path}[{i}]")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        assert abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b)), f"{path}: {a} != {b}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


@pytest.fixture(scope="module")
def fe():
    return results.band_data(FE / "fe_band.BAND.DAT"), results.dos_data(FE / "fe_doss.DOSS.DAT")


def _case(band, dos, o=None, look=None, base=None, edit=None, width=960, fermi_key=False):
    opts = PB.opts_for()
    opts.update(o or {})
    opts.update(look or {})          # a plot's own style is part of its options (plot.js DEFAULTS.bands)
    pid = P.resolve_id(PB.look_of(opts, base).get("style"), "band")
    return {"band": band, "dos": dos, "o": opts, "base": base or {}, "edit": edit, "width": width, "fermiKey": fermi_key,
            "tokens": PB.tokens(pid), "paper": {"paper_bgcolor": "#ffffff", "plot_bgcolor": "#ffffff"},
            "ec": P.element_colors("vesta")}


def _python(c):
    tk = c["tokens"]
    css = lambda n: tk.get(n, "")  # noqa: E731
    look = PB.look_of(c["o"], c["base"])
    env = {"css": css, "look": look, "tone": "light", "paper": c["paper"], "element_colors": c["ec"],
           "width": c["width"], "marginR": 14, "fermiKey": c["fermiKey"]}
    band = PB.edited_band(copy.deepcopy(c["band"]), c["edit"]) if c["edit"] else c["band"]
    fig = PB.band_figure(band, c["dos"], c["o"], env)
    traces, layout = PB.style_figure(fig["traces"], fig["layout"], "band", P.resolve_id(look["style"], "band"), css)
    return {"traces": traces, "layout": layout, "flags": P.style_flags(look)}


def _node(cases, tmp_path):
    data = tmp_path / "cases.json"
    data.write_text(json.dumps(cases))
    script = tmp_path / "run.mjs"
    script.write_text(NODE % {"bf": json.dumps((JS / "docs" / "bandfig.js").as_uri()),
                              "pp": json.dumps((JS / "plotpresets.js").as_uri()), "data": json.dumps(str(data))})
    r = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout)


def _compare(cases, tmp_path):
    got = _node(cases, tmp_path)
    for k, (c, g) in enumerate(zip(cases, got)):
        _close(_norm(_python(copy.deepcopy(c))), g, f"case{k}")


STYLES = [{}] + [{"style": pid} for pid in ("publication", "presentation", "grayscale", "colorblind", "vesta")]
SPIN = [{"spin_colors": s} for s in ("mace", "blue-red", "black-vermilion")] + [
    {"spin_colors": "custom", "spin_up_color": "#112233", "spin_down_color": "#AABBCC"},
    {"spin_up_color": "#123456"}, {"spin_down_style": "solid"}, {"spin_down_style": "dot"}]
DOS = [{"dos_colors": m} for m in P.DOS_MODES] + [{"dos_spin": "overlaid"}, {"dos_spin": "mirrored"}, {"dos_total": False},
                                                  {"dos_smooth": 0.05}, {"dos_smooth": 0.1, "dos_spin": "overlaid"}]


def test_figures_match_the_gui_fe(fe, tmp_path):
    band, dos = fe
    cases = []
    for look in STYLES + SPIN + DOS:
        cases.append(_case(band, dos, look=look))
    cases += [_case(band, None), _case(None, dos, width=520), _case(band, dos, fermi_key=True),
              _case(band, dos, o={"shift": 0.5}, fermi_key=True), _case(band, None, o={"unit": "meV"}),
              # the vacuum scale with a Fermi shift (F4-2): E - (E_vac - 0.25 eV), the line E_F - 0.25 eV
              _case({**band, "reference": "vacuum"}, {**dos, "reference": "vacuum"}, o={"shift": -0.25}, fermi_key=True),
              _case(band, dos, o={"shift": 0.125, "unit": "meV"}, fermi_key=True),
              _case(band, dos, o={"ylog": True}), _case(band, None, o={"ylog": True}),
              _case(band, dos, o={"spin": "up"}), _case(band, dos, o={"spin": "down"}),
              _case(band, dos, o={"emin": -3, "emax": 2, "hidden": ["Fe (d)"]}), _case(band, dos, width=400),
              # Settings (the user's defaults) under a plot's own choice
              _case(band, dos, base={"style": "publication", "dos_spin": "overlaid"}, look={"style": "colorblind"}),
              _case(band, None, edit=[{"from": "P", "to": "N"}, {"from": "H", "to": "P"}, {"from": "GAMMA", "to": "H"},
                                      {"from": "N", "to": "GAMMA", "hidden": True}])]
    _compare(cases, tmp_path)


@pytest.mark.skipif(not (CORPUS / "DOSS").is_dir(), reason="test/ data corpus not present")
def test_figures_match_the_gui_corpus(tmp_path):
    cases = []
    ag = next((CORPUS / "DOSS").glob("Ag4Br2S*_doss.DOSS.DAT"))
    dos = results.dos_data(ag)
    for look in [{}, {"style": "grayscale"}] + [{"dos_colors": m} for m in P.DOS_MODES] + [{"dos_spin": "mirrored"}]:
        cases.append(_case(None, dos, look=look, width=520))
    dia = sorted(CORPUS.rglob("*diamond*BAND.DAT")) + sorted(CORPUS.rglob("C_*BAND.DAT"))
    for b in dia[:1]:
        band = results.band_data(b)
        cases += [_case(band, None), _case(band, None, look={"style": "publication"}),
                  _case(band, None, look={"band_color": "#2E7D32"}), _case(band, None, look={"style": "grayscale"})]
    _compare(cases, tmp_path)


LOOKS = STYLES + SPIN + DOS + [{"style": "publication", "spin_colors": "blue-red", "spin_down_style": "solid",
                                "dos_colors": "element-orbital", "dos_spin": "overlaid", "dos_total": False},
                               {"band_color": "#2e7d32"}]


@pytest.mark.parametrize("k", range(len(LOOKS)))
def test_gui_choice_and_cli_flags_give_one_figure(fe, tmp_path, k):
    """A style chosen in the GUI, and mace plotting with the flags the GUI's Copy CLI command writes
    for it (its styleFlags), draw the same figure."""
    from mace.plotting.main import create_parser
    from mace.plotting.handlers.legacy import _band_config_from_args
    from mace.plotting.bandos_cli import look_of_config
    band, dos = fe
    look = LOOKS[k]
    gui = _case(band, dos, look=look)
    js = _node([gui], tmp_path)[0]
    assert js["flags"] == P.style_flags(PB.look_of(gui["o"], {}))
    args = create_parser().parse_args(["--band-dos", "x.BAND.DAT", *js["flags"]])
    cli_look = look_of_config(_band_config_from_args(args))
    cli = _case(band, dos)
    cli["o"] = {**PB.opts_for(), **cli_look}
    cli["tokens"] = PB.tokens(cli_look["style"])
    py = _norm(_python(cli))
    js.pop("flags")
    py.pop("flags")
    _close(py, js)


def test_tokens_are_tokens_css():
    css = (REPO_ROOT / "mace" / "gui" / "static" / "css" / "tokens.css").read_text()
    root = css[css.index(":root {"):css.index("}", css.index(":root {"))]
    val = lambda name, block: re.search(re.escape(name) + r":\s*([^;]+);", block).group(1).strip()  # noqa: E731
    for name in ("--text", "--text-2", "--text-3", "--border-strong", "--accent", "--plot-bg", "--surface", "--font-sans"):
        assert PB.TOKENS[name] == val(name, root), name
    assert PB.TOKENS["--line"] == val("--border", root)
    for k in range(1, 11):
        assert PB.tokens("default")[f"--plot-{k}"] == val(f"--plot-{k}", root)
    for pid in ("publication", "colorblind", "grayscale"):
        sel = f':root[data-plot-tone="light"] :is(.plot-probe, .fm-spec)[data-plot-preset="{pid}"] {{'
        block = css[css.index(sel):css.index("}", css.index(sel))]
        tk = PB.tokens(pid)
        for k in range(1, 11):
            assert tk[f"--plot-{k}"] == val(f"--plot-{k}", block), (pid, k)
        if "--plot-ink" in block:
            assert tk["--plot-ink"] == val("--plot-ink", block)


def test_style_flags_round_trip():
    from mace.plotting.main import create_parser
    from mace.plotting.handlers.legacy import _band_config_from_args
    from mace.plotting.bandos_cli import look_of_config
    for look in LOOKS:
        full = PB.look_of(look, {})
        args = create_parser().parse_args(["--band", "x", *P.style_flags(full)])
        want = {k: v.lower() if k.endswith("_color") else v for k, v in full.items()}     # flags write #rrggbb in lowercase
        assert look_of_config(_band_config_from_args(args)) == want


def test_fermi_shift_names_the_axis_and_the_line(fe):
    """F4-2: the data move by -s, so the axis is E - (E_F + s) (E - (E_vac + s) on the vacuum scale) and
    the dashed line, which stays put, is E_F + s: the matplotlib figures' energy_axis_label."""
    band, dos = fe
    for ref, zero in (("fermi", "E<sub>F</sub>"), ("vacuum", "E<sub>vac</sub>")):
        c = _case({**band, "reference": ref}, {**dos, "reference": ref}, o={"shift": 0.5}, fermi_key=True)
        fig = _python(c)
        assert fig["layout"]["yaxis"]["title"]["text"] == f"E − ({zero} + 0.50 eV) (eV)"
        key = next(t for t in fig["traces"] if t.get("meta") == "key-fermi")
        assert key["name"] == "E<sub>F</sub> + 0.50 eV"
        c = _case({**band, "reference": ref}, None, fermi_key=True)
        fig = _python(c)
        assert fig["layout"]["yaxis"]["title"]["text"] == f"E − {zero} (eV)"
        assert next(t for t in fig["traces"] if t.get("meta") == "key-fermi")["name"] == "E<sub>F</sub>"


def test_band_and_dos_lines_keep_every_point(fe):
    """M3: plotly's line.simplify (on by default) drops nearly collinear points at the 1x layout size, so
    the 300 dpi PNG showed straight facets along curved bands; every band and DOS line turns it off
    (bandfig.js too: the parity tests compare the traces)."""
    band, dos = fe
    fig = _python(_case(band, dos, look={"dos_spin": "mirrored"}))
    drawn = [t for t in fig["traces"] if t.get("meta") in ("up", "down", "dos")]
    assert drawn and all(t["line"]["simplify"] is False for t in drawn)


@pytest.mark.skipif(not (CORPUS / "DOSS").is_dir(), reason="test/ data corpus not present")
def test_closed_shell_run_is_drawn_in_a_neutral_color(tmp_path):
    """Marcus 2026-10-16: a run that is not spin-polarized (diamond: alpha = beta) draws its bands and
    its one-sided DOS total in the plot's ink, not spin up's pink, with no spin legend; --band-color /
    the style's "bands" color set it; a spin-polarized run (Fe) keeps the spin pair. The GUI's
    bandfig.js draws the same (the cases go through node too)."""
    dia = "1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized"
    band = results.band_data(next((CORPUS / "BAND").glob(f"{dia}_band.BAND.DAT")))
    dos = results.dos_data(CORPUS / "DOSS" / f"{dia}_doss.DOSS.DAT")
    assert not band["spin_polarized"] and not dos["spin_polarized"]
    cases = [_case(band, dos), _case(band, dos, look={"band_color": "#2E7D32"}), _case(band, dos, look={"style": "publication"})]
    _compare(cases, tmp_path)
    for c, want in zip(cases, [PB.tokens("default")["--plot-ink"] or PB.tokens("default")["--text"], "#2e7d32", "#000000"]):
        fig = _python(c)
        bands = [t for t in fig["traces"] if t.get("meta") == "up"]
        total = next(t for t in fig["traces"] if t.get("meta") == "dos" and t["name"] == "Total")
        assert bands and all(t["line"]["color"] == want and t["showlegend"] is False for t in bands)
        assert total["line"]["color"] == want
        # 1.1.6 legend: the Spin group holds one closed-shell key in the band color, no spin pair
        spin_keys = [t for t in fig["traces"] if t.get("meta") == "key-spin"]
        assert [(t["name"], t["line"]["color"]) for t in spin_keys] == [("α = β (closed shell)", want)]
        assert not any("spin up" in str(t.get("name")) for t in fig["traces"] if t.get("showlegend", True))
    fe = results.band_data(FE / "fe_band.BAND.DAT")
    fig = _python(_case(fe, None, look={"band_color": "#2e7d32"}))
    up = next(t for t in fig["traces"] if t.get("meta") == "up")
    assert up["line"]["color"] == P.spin_colors("default")["up"] and up["showlegend"] is False
    assert [t["name"] for t in fig["traces"] if t.get("meta") == "key-spin"] == ["spin up (α)", "spin down (β)"]


# ---------------------------------------------------------------- the 1.1.6 DOS / legend design
def _closed(band, dos):
    """Fe as a closed-shell run (one set of bands, one DOS block): the N1 layout without the corpus."""
    b = {**band, "spin_polarized": False, "spin_identical": False, "down": None}
    d = {**dos, "spin_polarized": False, "spin_summed": False,
         "projections": [{**p, "down": None} for p in dos["projections"]]}
    return b, d


def test_dos_layout_matches_the_gui_spin_and_closed_shell(fe, tmp_path):
    """S1 (mirrored spin DOS), N1 (closed shell), the smoothing and the legend placement: bandfig.js and
    plotly_bands.py draw the same figure for every preset, DOS alone, band alone and band + DOS."""
    band, dos = fe
    cb, cd = _closed(band, dos)
    cases = []
    for look in STYLES + [{"dos_smooth": 0.05}, {"spin_colors": "mace"}, {"spin_colors": "black-vermilion"}]:
        cases += [_case(None, dos, look=look, width=520), _case(cb, cd, look=look), _case(None, cd, look=look, width=520),
                  _case(cb, None, look=look)]
    cases += [_case(None, dos, o={"hidden": []}, width=520), _case(None, cd, o={"hidden": []}, look={"dos_smooth": 0.05}, width=520),
              _case(band, dos, o={"hidden": [], "spin": "up"}, look={"dos_smooth": 0.05}),
              _case(None, dos, o={"emin": -12, "emax": 25}, width=520), _case(band, dos, o={"emin": -12, "emax": 25}),
              _case(None, dos, o={"ylog": True}, width=520), _case(cb, cd, fermi_key=True, look={"dos_smooth": 0.05})]
    _compare(cases, tmp_path)


def test_spin_dos_is_mirrored_with_neutral_spin_keys(fe):
    band, dos = fe
    fig = _python(_case(band, dos))
    lay, tr = fig["layout"], fig["traces"]
    spin = P.spin_colors("default")
    # S1: magnitudes both sides of a symmetric range, the words over each half
    ax = lay["xaxis2"]
    assert ax["range"][0] == -ax["range"][1] and all(not t.startswith("-") for t in ax["ticktext"])
    assert ax["zerolinecolor"] == PB.tokens("default")["--text"]
    words = {a["text"]: a["font"]["color"] for a in lay["annotations"]}
    assert words["<b>↑ up</b>"] == spin["up"] and words["<b>down ↓</b>"] == spin["down"]
    # the totals filled in their spin's color; projections keep their color, dashed on the down side
    up_t = next(t for t in tr if t.get("meta") == "dos" and t["name"] == "Total ↑")
    dn_t = next(t for t in tr if t.get("meta") == "dos" and t["name"] == "Total ↓")
    assert up_t["fillcolor"] == P.spin_fills(spin)["up"] and dn_t["fillcolor"] == P.spin_fills(spin)["down"]
    assert "fillpattern" not in dn_t
    d_up = next(t for t in tr if t.get("meta") == "dos" and t["name"] == "Fe d ↑")
    d_dn = next(t for t in tr if t.get("meta") == "dos" and t["name"] == "Fe d ↓")
    assert d_up["line"]["color"] == d_dn["line"]["color"] and d_dn["line"]["dash"] == "dash" and d_up["line"]["dash"] == "solid"
    # projection colors keep clear of the spin pair (blue/red: no blue)
    assert d_up["line"]["color"] not in (spin["up"], spin["down"], "#2f5fd0")
    # the 2-4 projections that carry the DOS: Fe d and Fe s (the design notes)
    assert [t["name"] for t in tr if t.get("meta") == "key-proj"] == ["Total", "Fe d", "Fe s"]
    # two titled legend groups, neutral spin keys, one legend right of the plot
    keys = [t for t in tr if t.get("meta") == "key-spin"]
    assert [(t["name"], t["line"]["dash"], t["line"]["color"]) for t in keys] == [
        ("spin up (α)", "solid", PB.tokens("default")["--text"]), ("spin down (β)", "dash", PB.tokens("default")["--text"])]
    assert keys[0]["legendgrouptitle"]["text"] == "Spin"
    assert next(t for t in tr if t.get("meta") == "key-proj")["legendgrouptitle"]["text"] == "Projections"
    assert lay["legend"]["x"] >= 1 and "legend2" not in lay
    # E_F dotted (dashed is spin down)
    assert all(s["line"]["dash"] == "dot" for s in lay["shapes"] if s.get("name") in ("fermi", "fermi2"))


def test_band_alone_legend_is_one_line_above_the_frame(fe):
    band, _dos = fe
    lay = _python(_case(band, None))["layout"]
    assert lay["legend"]["orientation"] == "h" and lay["legend"]["y"] >= 1 and lay["legend"]["title"]["text"] == "Spin"


def test_dos_alone_legend_places_itself(fe):
    """Inside the plot when the box covers no curve, else outside on the right."""
    _band, dos = fe
    lay = _python(_case(None, dos, width=520))["layout"]
    leg = lay["legend"]
    inside = leg["x"] < 1
    if inside:
        assert leg["xanchor"] == "right" and leg["borderwidth"] == 1
    spot = P.legend_spot([([0, 10], [-0.9, -0.9])], [0, 10], [-1, 1], 0.3, 0.5)
    assert spot == "top"
    assert P.legend_spot([([0, 10], [0.9, 0.9])], [0, 10], [-1, 1], 0.3, 0.5) == "bottom"
    assert P.legend_spot([([0, 10, 10], [0, 0.9, -0.9])], [0, 10], [-1, 1], 0.3, 0.5) == "outside"


def test_closed_shell_dos_and_smoothing(fe):
    band, dos = fe
    cb, cd = _closed(band, dos)
    fig = _python(_case(None, cd, look={"dos_smooth": 0.05}, width=520))
    tr, lay = fig["traces"], fig["layout"]
    ink = PB.tokens("default")["--text"]
    total = next(t for t in tr if t.get("meta") == "dos" and t["name"] == "Total")
    assert total["line"]["color"] == ink and total["fillcolor"] == P.neutral_fill(ink)
    assert lay["xaxis"]["range"][0] == 0 and "tickvals" not in lay["xaxis"]
    assert [t["name"] for t in tr if t.get("meta") == "key-spin"] == ["α = β (closed shell)"]
    assert any(a["text"] == "Gaussian smoothing σ = 0.05 eV (display only)" for a in lay["annotations"])
    raw = next(p for p in cd["projections"] if p["label"] == "Total")["up"]
    assert total["x"] == P.gauss_smooth(raw, cd["energy"], 0.05) and total["x"] != raw
    # smoothing off by default; the area under the curve is kept
    plain = next(t for t in _python(_case(None, cd, width=520))["traces"] if t.get("meta") == "dos" and t["name"] == "Total")
    assert plain["x"] == raw
    sm = P.gauss_smooth(raw, cd["energy"], 0.05)
    assert abs(sum(sm) - sum(raw)) < 1e-6 * max(1.0, sum(raw)) + 0.02 * max(raw)


def test_outside_the_dos_window_is_shaded(fe):
    """The energies the DOSS run did not compute are no data, not zero states."""
    _band, dos = fe
    lay = _python(_case(None, dos, o={"emin": -14, "emax": 6}, width=520))["layout"]
    gaps = [s for s in lay["shapes"] if s.get("name") == "no-data"]
    assert gaps and gaps[0]["y1"] == pytest.approx(dos["window"][0])
    assert any(a["text"] == "outside the DOS window (no data)" for a in lay["annotations"])
    assert not [s for s in _python(_case(None, dos, width=520))["layout"]["shapes"] if s.get("name") == "no-data"]


def test_dos_smooth_names_everywhere():
    """One name for the option: presets STYLE_OPTIONS, the flag, the folder key and the GUI setting."""
    from mace.plotting.main import create_parser
    from mace.plotting.handlers.legacy import _band_config_from_args
    from mace.plotting.bandos_cli import look_of_config
    from mace.utils import folder_settings as FS
    opt = next(o for o in P.STYLE_OPTIONS if o["name"] == "dos_smooth")
    assert (opt["cli"], opt["gui"], opt["default"]) == ("--dos-smooth", "plots.dosSmooth", 0)
    assert FS.KEYS["plots.dos_smooth"]["gui"] == "plots.dosSmooth"
    p = create_parser()
    assert look_of_config(_band_config_from_args(p.parse_args(["--dos", "x", "--dos-smooth", "0.05"])))["dos_smooth"] == 0.05
    assert look_of_config(_band_config_from_args(p.parse_args(["--dos", "x", "--dos-smooth", "0.1"])))["dos_smooth"] == 0.1
    assert look_of_config(_band_config_from_args(p.parse_args(["--dos", "x"])))["dos_smooth"] == 0
    with pytest.raises(SystemExit):
        p.parse_args(["--dos", "x", "--dos-smooth", "-1"])


def test_dos_layout_helpers_are_one_rule(fe, tmp_path):
    """The exported DOS layout helpers (presets.py / plotpresets.js) give the same answers: the gallery
    and the matplotlib figures reuse them."""
    _band, dos = fe
    tot = next(p for p in dos["projections"] if p["label"] == "Total")
    lines = [{"label": p["label"], "show": True, "total": p["label"] == "Total"} for p in dos["projections"]]
    weights = P.dos_weights(dos["energy"], dos["projections"], -6, 6)
    args = {"y": tot["up"], "e": dos["energy"], "lines": lines, "w": weights, "win": dos["window"]}
    data = tmp_path / "h.json"
    data.write_text(json.dumps(args))
    script = tmp_path / "h.mjs"
    script.write_text(f"""import * as PP from {json.dumps((JS / 'plotpresets.js').as_uri())};
import {{ readFileSync }} from "node:fs";
const a = JSON.parse(readFileSync({json.dumps(str(data))}, "utf8"));
console.log(JSON.stringify({{ smooth: PP.gaussSmooth(a.y, a.e, 0.05), pick: PP.pickProjections(a.lines, a.w),
  pal: [PP.projectionPalette("default", ["#1f4e9c", "#c0392b"]), PP.projectionPalette("colorblind", ["#0072b2", "#d55e00"], "#111111"),
        PP.projectionPalette("grayscale", null)],
  fills: [PP.spinFills({{up: "#1f4e9c", down: "#c0392b"}}, "grayscale"), PP.neutralFill("#15181e")],
  gaps: PP.dosWindowGaps(a.win, -12, 25), ticks: [PP.mirroredTicks(-7.3, 7.3), PP.mirroredTicks(-0.43, 0.43)],
  spot: PP.legendSpot([[[0, 10], [-0.9, -0.9]]], [0, 10], [-1, 1], 0.3, 0.5) }}));""")
    r = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-2000:]
    py = {"smooth": P.gauss_smooth(tot["up"], dos["energy"], 0.05), "pick": P.pick_projections(lines, weights),
          "pal": [P.projection_palette("default", ["#1f4e9c", "#c0392b"]), P.projection_palette("colorblind", ["#0072b2", "#d55e00"], "#111111"),
                  P.projection_palette("grayscale", None)],
          "fills": [P.spin_fills({"up": "#1f4e9c", "down": "#c0392b"}, "grayscale"), P.neutral_fill("#15181e")],
          "gaps": P.dos_window_gaps(dos["window"], -12, 25), "ticks": [P.mirrored_ticks(-7.3, 7.3), P.mirrored_ticks(-0.43, 0.43)],
          "spot": P.legend_spot([([0, 10], [-0.9, -0.9])], [0, 10], [-1, 1], 0.3, 0.5)}
    _close(_norm(py), json.loads(r.stdout))
