"""Plot presets (1.1.6): one definition for mace plotting and mace gui.

* plotpresets_data.js is the current output of ``python -m mace.plotting.presets --js``;
* the GUI's functions (plotpresets.js) give what presets.py gives, on the DOS
  projections of real runs (node runs the module);
* the DOS coloring modes: Auto picks by orbital for one element, element x orbital
  for a few, by element for many; every line reads on the plot (2:1);
* the one spin color pair is a single constant, used by both sides.
"""
import json
import shutil
import subprocess

import pytest

from conftest import REPO_ROOT
from mace.plotting import presets as P

JS = REPO_ROOT / "mace" / "gui" / "static" / "js"

LABELS = {
    "diamond": ["C (s)", "C (p)", "C (d)", "C (all)", "Total"],
    "ag4br2s": ["Ag (s)", "Ag (p)", "Ag (d)", "Ag (f)", "Ag (all)", "Br (s)", "Br (p)", "Br (d)", "Br (all)",
                "S (s)", "S (p)", "S (d)", "S (all)", "Total"],
    "pbcusecl": ["Pb (s)", "Pb (p)", "Pb (d)", "Pb (all)", "Cu (s)", "Cu (p)", "Cu (d)", "Cu (f)", "Cu (all)",
                 "Se (s)", "Se (p)", "Se (d)", "Se (all)", "Cl (s)", "Cl (p)", "Cl (d)", "Cl (all)", "Total"],
    "unnamed": ["projection 1", "projection 2", "Total"],
}


def test_generated_js_data_is_current():
    assert (REPO_ROOT / P.JS_DATA).read_text() == P.js_module(), "run: python -m mace.plotting.presets --js"


def test_spin_color_schemes():
    """Marcus 2026-10-16: named schemes shared by the GUI and the CLI; Blue/red is the default (MACE
    pink/cyan a named scheme),
    Publication keeps black/vermilion, Print grayscale and Colorblind safe their own pairs; beta dashed."""
    S = P.SPIN_SCHEMES
    assert S["mace"]["light"] == {"up": "#e0218a", "down": "#00a0a4"} and S["mace"]["dark"] == {"up": "#fa26a0", "down": "#2ff3e0"}
    assert S["blue-red"]["light"] == {"up": "#1f4e9c", "down": "#c0392b"} and S["blue-red"]["dark"] == {"up": "#7aa7f0", "down": "#f08a7c"}
    assert S["black-vermilion"]["light"] == {"up": "#000000", "down": "#d55e00"}
    assert S["black-vermilion"]["dark"] == {"up": "#ffffff", "down": "#e8803a"}
    assert P.DEFAULT_SPIN_SCHEME == "blue-red" and P.SPIN_COLORS["light"] == S["blue-red"]["light"] and P.SPIN_DOWN_DASH == "dash"
    for pid in ("default", "presentation"):
        assert P.spin_colors(pid) == S["blue-red"]["light"] and P.spin_colors(pid, tone="dark") == S["blue-red"]["dark"]
    assert P.spin_colors("default", scheme="mace") == S["mace"]["light"]
    assert P.spin_colors("publication") == S["black-vermilion"]["light"]
    assert P.spin_colors("grayscale") == {"up": "#000000", "down": "#767676"}
    assert P.spin_colors("colorblind", scheme="blue-red") == S["blue-red"]["light"]
    # custom colors: with "custom", or alone (auto); never over a named scheme
    assert P.spin_colors("default", scheme="custom", up="#123456") == {"up": "#123456", "down": "#c0392b"}
    assert P.spin_colors("default", up="#ABCDEF")["up"] == "#abcdef"
    assert P.spin_colors("default", scheme="mace", up="#123456") == S["mace"]["light"]


def test_spin_shown_dos_lines_are_solid_and_dominant_is_strongest():
    """Bug (e): with both spins drawn the dash means spin down only (every projection solid, told
    apart by color); the dominant orbital (Cr d of CrO2) gets the strongest shade, never the
    faintest or dotted one."""
    labels = ["Cr (s)", "Cr (p)", "Cr (d)", "Cr (f)", "Cr (all)", "O (s)", "O (p)", "O (d)", "O (all)", "Total"]
    w = {"Cr (s)": 1, "Cr (p)": 2, "Cr (d)": 50, "Cr (f)": 0.1, "O (s)": 1, "O (p)": 20, "O (d)": 0.5}
    colors = P.element_colors("vesta")
    for mode in ("element-orbital", "orbital"):
        r = P.dos_styles(labels, mode, colors, spin=True, weights=w)
        assert {x["dash"] for x in r["lines"] if x["show"]} == {"solid"}, mode
        shown = [x for x in r["lines"] if x["show"] and not x["total"]]
        assert len({x["color"] for x in shown}) == len(shown), mode
    r = P.dos_styles(labels, "element-orbital", colors, weights=w)
    by = {x["label"]: x for x in r["lines"]}
    assert by["Cr (d)"]["dash"] == "solid" and by["Cr (f)"]["dash"] == "dashdot"
    # the strongest shade is the darkest on a light plot
    assert P.luminance(by["Cr (d)"]["color"]) < min(P.luminance(by[k]["color"]) for k in ("Cr (s)", "Cr (p)", "Cr (f)"))
    rs = P.dos_styles(labels, "element-orbital", colors, spin=True, weights=w)
    assert {x["label"]: x["color"] for x in rs["lines"]} == {x["label"]: x["color"] for x in r["lines"]}


def test_auto_dos_mode():
    mode = lambda k: P.dos_styles(LABELS[k], "auto")["mode"]   # noqa: E731
    assert (mode("diamond"), mode("ag4br2s"), mode("pbcusecl"), mode("unnamed")) == \
        ("orbital", "element-orbital", "element", "total")


@pytest.mark.parametrize("mode", P.DOS_MODES)
@pytest.mark.parametrize("bg", ["#ffffff", "#16191f"])
def test_every_dos_line_reads(mode, bg):
    colors = P.element_colors("vesta")
    for k, labels in LABELS.items():
        r = P.dos_styles(labels, mode, colors, bg=bg)
        shown = [x for x in r["lines"] if x["show"]]
        assert shown, (k, mode)
        for x in shown:
            assert P.contrast(x["color"], bg) >= 2 - 1e-9, (k, mode, x)
        if r["mode"] == "total":
            assert [x["name"] for x in shown] == ["Total"]


def test_legend_names():
    assert [P.legend_name(x) for x in ["Mg (s)", "O (p)", "Mg (all)", "Mg ", "Total", "projection 2"]] == \
        ["Mg s", "O p", "Mg", "Mg", "Total", "projection 2"]


def test_grayscale_tells_lines_apart_by_style():
    r = P.dos_styles(LABELS["ag4br2s"], "element-orbital", P.element_colors("vesta"), preset="grayscale")
    shown = [x for x in r["lines"] if x["show"] and not x["total"]]
    assert len({(x["color"], x["dash"]) for x in shown}) == len(shown)
    assert all(P._rgb(x["color"])[0] == P._rgb(x["color"])[1] == P._rgb(x["color"])[2] for x in shown)


def test_vesta_is_for_maps_only():
    assert P.resolve_id("vesta", "maps") == "vesta" and P.resolve_id("vesta", "band") == "default"
    assert P.get("nope", "band") is P.PRESETS["default"]


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_gui_functions_match_python(tmp_path):
    colors = P.element_colors("vesta")
    cases = [(k, m, pid, bg, spin) for k in LABELS for m in P.DOS_MODES for pid in ("default", "grayscale", "colorblind")
             for bg in ("#ffffff", "#16191f") for spin in (False, True)]
    weights = {lab: (7 * i) % 11 + 0.5 for k in LABELS for i, lab in enumerate(LABELS[k])}
    script = tmp_path / "run.mjs"
    script.write_text(
        f"import * as PP from {json.dumps((JS / 'plotpresets.js').as_uri())};\n"
        f"const L = {json.dumps(LABELS)}, C = {json.dumps(colors)}, cases = {json.dumps(cases)}, W = {json.dumps(weights)};\n"
        "const out = cases.map(([k, m, pid, bg, spin]) => PP.dosStyles(L[k], {mode: m, elementColors: C, preset: pid, bg, spin, weights: spin ? W : null}));\n"
        "out.push(['#ffe600', '#8fff00', '#ffffff'].map((c) => [PP.readable(c), PP.readable(c, '#16191f'), PP.shade(c, 'p'), PP.shade(c, 'f', '#16191f')]));\n"
        "console.log(JSON.stringify(out));\n")
    got = json.loads(subprocess.run(["node", str(script)], capture_output=True, text=True, check=True, timeout=60).stdout)
    want = [P.dos_styles(LABELS[k], m, colors, preset=pid, bg=bg, spin=spin, weights=weights if spin else None)
            for k, m, pid, bg, spin in cases]
    want.append([[P.readable(c), P.readable(c, "#16191f"), P.shade(c, "p"), P.shade(c, "f", "#16191f")]
                 for c in ("#ffe600", "#8fff00", "#ffffff")])
    assert got == want
