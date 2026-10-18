"""WCAG 2.1 contrast of the GUI's design tokens (tokens.css), every theme.

No browser: the tokens are parsed from the stylesheet. Every theme, the
base ones and the editor themes of 1.1.6 alike, must keep every text token
legible on every surface it is drawn on:

* body, muted and disabled text (--text, --text-2, --text-3), links and
  accents, and the status colors (ok/warn/error/info) need 4.5:1 on the page,
  the panels, the hovered and selected rows (--surface-3, --accent-soft: a
  menu's hover, a picked list row) and a selected table row (--sel-soft);
* text on a filled control or chip: the primary button (--on-accent on
  --accent and its hover), the danger button, the warning banners and chips,
  the done steps (--on-accent on --success), the tooltip and the classic
  toast (the surface colors on --text), the file-type badges (--k-on on each
  --k-*): 4.5:1;
* the focus ring and the borders that outline controls (WCAG 1.4.11): 3:1 on
  every surface;
* the plots: their text 4.5:1 on --plot-bg, the axis frame 3:1, the grid
  visible (1.15:1, it is decoration), every series 2:1 and the first eight
  distinct from one another (CIE76 ΔE at least 12);
* the 3D view: the axis triad, cell and hover color 3:1 on --viewport-bg,
  the selection 2:1, and a background that hides at most one common element
  of the VESTA palette (luminance contrast under 1.25:1).

The two high-contrast themes meet AAA: 7:1 for text, 4.5:1 for the focus
ring and control borders, and 3:1 for plot series and the 3D marks.

The forced surfaces (Settings › Appearance › Plot background, a White or
Black 3D background in a theme of the other lightness) are token sets of
their own, checked the same way for every theme they can show in.

Settings › Appearance › Plot style › Publication (data-plot-style) swaps the
plot series for Okabe–Ito after a first one in ink, on a light or a dark plot
(data-plot-tone): checked the same way on every theme's own plot background
and on both forced ones, AAA in the high-contrast themes.

A failure is fixed in tokens.css (in the theme's own spirit), never in a
component. The themes are the :root[data-theme="..."] blocks; every one is
a full token set (every color the dark theme sets, and the plot series), and
the list matches js/themes.js, boot.js, server.py and diagnostics.py.
"""
import json
import math
import re

import pytest

from conftest import REPO_ROOT

STATIC = REPO_ROOT / "mace" / "gui" / "static"
TOKENS = (STATIC / "css" / "tokens.css").read_text()
THEMES_JS = (STATIC / "js" / "themes.js").read_text()


def _block(start):
    i = TOKENS.index(start)
    return TOKENS[i:TOKENS.index("}", i)]


def _vars(text):
    return dict(re.findall(r"--([\w-]+):\s*(#[0-9a-fA-F]{6})\b", text))


def _all_vars(text):
    return dict(re.findall(r"--([\w-]+):\s*([^;]+);", text))


LIGHT = _vars(_block(":root {"))
# the OS-dark block applies with System (no data-theme); it is the same as the explicit dark one
OS_DARK = {**LIGHT, **_vars(_block(":root:not([data-theme]) {"))}
THEME_IDS = re.findall(r':root\[data-theme="([\w-]+)"\] \{', TOKENS)
BLOCKS = {t: _block(f':root[data-theme="{t}"] {{') for t in THEME_IDS}
THEMES = {"light": LIGHT, **{t: {**LIGHT, **_vars(b)} for t, b in BLOCKS.items()}}
DARK = THEMES["dark"]
HIGH_CONTRAST = ("hc-light", "hc-dark")
EDITOR = ("catppuccin-mocha", "catppuccin-latte", "dracula", "gruvbox-dark", "nord", "one-dark",
          "solarized-dark", "solarized-light", "tokyo-night", "tokyo-night-storm")

# the forced surfaces: a generic light and dark set, which the high-contrast and colorblind themes refine
FORCED = {v: _vars(_block(f':root[data-plot-surface="{v}"] :is(')) for v in ("light", "dark")}
FORCED_HC = {v: _vars(_block(f':root:is([data-theme="hc-light"], [data-theme="hc-dark"])[data-plot-surface="{v}"]'))
             for v in ("light", "dark")}
FORCED_CVD = {v: _vars(_block(f':root[data-theme="cvd"][data-plot-surface="{v}"]')) for v in ("light", "dark")}
# Plot style › Publication, by the plot surface's lightness
# the plot presets with series of their own (1.1.6: tokens.css [data-plot-preset] blocks, by plot tone)
PRESET_BLOCKS = {pid: {v: _vars(_block(f':root[data-plot-tone="{v}"] :is(.plot-probe, .fm-spec)[data-plot-preset="{pid}"]'))
                       for v in ("light", "dark")} for pid in ("publication", "colorblind", "grayscale")}
PUBLICATION = PRESET_BLOCKS["publication"]


def forced(theme, v):
    """The tokens a plot (or a 3D view's overlays) draws with on the forced light or dark surface."""
    T = {**THEMES[theme], **FORCED[v]}
    if theme in HIGH_CONTRAST:
        T.update(FORCED_HC[v])
    if theme == "cvd":
        T.update(FORCED_CVD[v])
    return T


def _lum(h):
    c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def ratio(a, b):
    la, lb = _lum(a), _lum(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def mix(a, b, p):
    """color-mix(in srgb, a p, b): a over b at opacity p."""
    ca, cb = ([int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in (a, b))
    return "#" + "".join(f"{round(x * p + y * (1 - p)):02x}" for x, y in zip(ca, cb))


def _lab(h):
    c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    xyz = (0.4124 * c[0] + 0.3576 * c[1] + 0.1805 * c[2]) / 0.95047, 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2], \
        (0.0193 * c[0] + 0.1192 * c[1] + 0.9505 * c[2]) / 1.08883
    f = [t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116 for t in xyz]
    return 116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])


def delta_e(a, b):
    return math.dist(_lab(a), _lab(b))


SURFACES = ("bg", "surface", "surface-2", "surface-3", "accent-soft")
TEXT = ("text", "text-2", "text-3", "accent", "success", "danger", "warning", "info")
# a filled control's text on its fill (fg, bg): buttons, banners, chips, done steps, tooltip, toast
FILLED = [("on-accent", "accent"), ("on-accent", "accent-hover"), ("on-danger", "danger"), ("on-accent", "warning"),
          ("surface", "warning"), ("on-accent", "success"), ("surface", "text"), ("surface-3", "text"), ("bg", "text")]
SERIES = [f"plot-{k}" for k in range(1, 11)]
ELEMENTS = json.loads((STATIC / "element_colors.json").read_text())["schemes"]["vesta"]
COMMON = ("H C N O F S P Si Na Cl Mg Al Fe Ti Zn Cu Ca K Li Br I Pb Ba Sr Ni Co Mn Cr Ag Au Ga Ge Se Sn Te Cs "
          "Rb Zr Mo W Bi La").split()


def need(theme):
    return 7 if theme in HIGH_CONTRAST else 4.5


def ui(theme):
    return 4.5 if theme in HIGH_CONTRAST else 3


def mark(theme):
    return 3 if theme in HIGH_CONTRAST else 2


# ---------------------------------------------------------------- the lists agree

def test_the_themes_are_the_ones_the_pickers_list():
    listed = re.findall(r'\{ id: "([\w-]+)", group:', THEMES_JS)
    assert listed[0] == "system" and listed[1] == "light", listed
    assert sorted(listed[2:]) == sorted(THEME_IDS), (listed, THEME_IDS)
    assert len(THEME_IDS) == 16          # dark, the five of 1.1.6 and the ten editor themes
    assert "publication" not in THEME_IDS and 'id: "publication"' not in THEMES_JS   # Plot style › Publication now
    boot = (STATIC / "js" / "boot.js").read_text()
    server = (REPO_ROOT / "mace" / "gui" / "server.py").read_text()
    diag = (REPO_ROOT / "mace" / "gui" / "diagnostics.py").read_text()
    for t in ["light"] + THEME_IDS:
        assert f'"{t}"' in boot and f'"{t}"' in server and f'"{t}"' in diag, t


def test_the_themes_come_in_groups():
    """Light & Dark, Accessibility, Editor themes: each theme in one, in that order, the
    editor themes by name with a family's flavors side by side; the ids of 1.1.6's first themes stay."""
    groups = re.findall(r'\{ id: "(\w+)", label: "([^"]+)" \}', THEMES_JS)
    assert [g[1] for g in groups] == ["Light & Dark", "Accessibility", "Editor themes"], groups
    rows = re.findall(r'\{ id: "([\w-]+)", group: "(\w+)", label: "([^"]+)"', THEMES_JS)
    order = [g[0] for g in groups]
    assert [order.index(g) for _, g, _ in rows] == sorted(order.index(g) for _, g, _ in rows), rows   # contiguous
    by = {g: [t for t, gg, _ in rows if gg == g] for g in order}
    assert by["base"] == ["system", "light", "dark", "sepia", "black"]
    assert by["access"] == ["hc-light", "hc-dark", "cvd"]
    assert tuple(by["editor"]) == EDITOR
    labels = {t: lab for t, _, lab in rows}
    assert labels["sepia"] == "Sepia"
    assert re.search(r'id: "sepia"[^}]*aliases: \["paper"\]', THEMES_JS)        # the old name still finds it


def test_every_editor_theme_names_its_source_and_license():
    for t in EDITOR:
        i = TOKENS.index(f':root[data-theme="{t}"] {{')
        note = TOKENS[TOKENS.rindex("/*", 0, i):i]
        assert "github.com/" in note and "MIT" in note, (t, note)


def test_os_dark_and_chosen_dark_are_the_same():
    assert OS_DARK == DARK


@pytest.mark.parametrize("theme", THEME_IDS)
def test_every_theme_is_a_full_token_set(theme):
    """Each block sets every token the dark theme sets (and the plot series),
    so no theme inherits a color made for another."""
    series = set(SERIES)
    want = set(_all_vars(_block(':root[data-theme="dark"] {')))
    if theme != "dark":                  # the dark theme keeps the light series (they read on both)
        want |= series
    have = set(_all_vars(BLOCKS[theme]))
    assert not want - have, (theme, sorted(want - have))
    assert re.search(r"color-scheme: (light|dark);", BLOCKS[theme])


def test_dark_looking_themes_swap_the_logo():
    """The light and dark logo files: every theme with color-scheme: dark shows the dark one."""
    dark = [t for t, b in BLOCKS.items() if "color-scheme: dark" in b]
    assert set(dark) == {"dark", "hc-dark", "black"} | {t for t in EDITOR if "latte" not in t and "light" not in t}, dark
    shell = (STATIC / "css" / "shell.css").read_text()
    for t in dark:
        assert f'[data-theme="{t}"]' in shell, t
        assert re.search(rf'id: "{t}"[^}}]*dark: true', THEMES_JS), t
    for t in set(THEME_IDS) - set(dark):
        assert not re.search(rf'id: "{t}"[^}}]*dark: true', THEMES_JS), t


def test_every_theme_has_a_swatch():
    for t in ["system"] + THEME_IDS:
        assert f'.theme-swatch[data-swatch="{t}"]' in TOKENS, t


# ---------------------------------------------------------------- the interface

@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("fg", TEXT)
def test_text_on_every_surface(theme, fg):
    T = THEMES[theme]
    low = {bg: round(ratio(T[fg], T[bg]), 2) for bg in SURFACES if ratio(T[fg], T[bg]) < need(theme)}
    assert not low, (theme, fg, T[fg], low)


@pytest.mark.parametrize("theme", THEMES)
def test_text_on_a_selected_row(theme):
    """A selected table row (--sel-soft, 22% of the selection color) keeps its text legible."""
    T = THEMES[theme]
    low = {(fg, s): round(ratio(T[fg], mix(T["sel"], T[s], .22)), 2) for fg in ("text", "text-2") for s in ("surface", "surface-2")
           if ratio(T[fg], mix(T["sel"], T[s], .22)) < need(theme)}
    assert not low, (theme, low)


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("pair", FILLED, ids=lambda p: f"{p[0]}-on-{p[1]}")
def test_text_on_filled_controls(theme, pair):
    T = THEMES[theme]
    assert ratio(T[pair[0]], T[pair[1]]) >= need(theme), (theme, pair, round(ratio(T[pair[0]], T[pair[1]]), 2))


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("what", ["focus", "border-strong"])
def test_focus_ring_and_control_borders(theme, what):
    T = THEMES[theme]
    on = ("bg", "surface", "surface-2", "surface-3") + (("accent-soft",) if what == "focus" else ())
    low = {bg: round(ratio(T[what], T[bg]), 2) for bg in on if ratio(T[what], T[bg]) < ui(theme)}
    assert not low, (theme, what, T[what], low)


@pytest.mark.parametrize("theme", THEMES)
def test_file_kind_chips(theme):
    T = THEMES[theme]
    low = {k: round(ratio(T["k-on"], v), 2) for k, v in T.items() if k.startswith("k-") and k != "k-on"
           and ratio(T["k-on"], v) < need(theme)}
    assert not low, (theme, low)


# ---------------------------------------------------------------- plots and the 3D view

def _plot_problems(theme, T):
    bad = {}
    for fg in ("text", "text-2", "text-3"):
        if ratio(T[fg], T["plot-bg"]) < need(theme):
            bad[fg] = round(ratio(T[fg], T["plot-bg"]), 2)
    if ratio(T["border-strong"], T["plot-bg"]) < ui(theme):
        bad["frame"] = round(ratio(T["border-strong"], T["plot-bg"]), 2)
    if ratio(T["border"], T["plot-bg"]) < 1.15:
        bad["grid"] = round(ratio(T["border"], T["plot-bg"]), 2)
    for s in SERIES:
        if ratio(T[s], T["plot-bg"]) < mark(theme):
            bad[s] = round(ratio(T[s], T["plot-bg"]), 2)
    first = SERIES[:8]
    close = [(a, b, round(delta_e(T[a], T[b]), 1)) for i, a in enumerate(first) for b in first[i + 1:] if delta_e(T[a], T[b]) < 12]
    if close:
        bad["alike"] = close
    return bad


def _view_problems(theme, T, bg):
    bad = {}
    for m in ("axis-a", "axis-b", "axis-c", "cell", "hover-atom"):
        if ratio(T[m], bg) < ui(theme):
            bad[m] = round(ratio(T[m], bg), 2)
    if ratio(T["sel"], bg) < mark(theme):
        bad["sel"] = round(ratio(T["sel"], bg), 2)
    if ratio(T["text-2"], bg) < ui(theme):                     # the triad's origin
        bad["text-2"] = round(ratio(T["text-2"], bg), 2)
    if ratio(T["text"], T["surface"]) < need(theme):           # a label on its own chip
        bad["label"] = round(ratio(T["text"], T["surface"]), 2)
    return bad


@pytest.mark.parametrize("theme", THEMES)
def test_plots_read_on_the_plot_background(theme):
    bad = _plot_problems(theme, THEMES[theme])
    assert not bad, (theme, bad)


@pytest.mark.parametrize("theme", THEMES)
def test_3d_marks_read_on_the_viewport(theme):
    T = THEMES[theme]
    bad = _view_problems(theme, T, T["viewport-bg"])
    assert not bad, (theme, bad)


@pytest.mark.parametrize("theme", THEMES)
def test_the_3d_background_keeps_common_elements(theme):
    """At most one common VESTA element (yellow S on white, navy Cr on near-black) under 1.25:1."""
    bg = THEMES[theme]["viewport-bg"]
    lost = [e for e in COMMON if ratio(ELEMENTS[e], bg) < 1.25]
    assert len(lost) <= 1, (theme, bg, lost)


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("surface", ["light", "dark"])
def test_forced_plot_surfaces(theme, surface):
    """Settings › Appearance › Plot background: a light or dark plot reads in every theme."""
    T = forced(theme, surface)
    assert (_lum(T["plot-bg"]) > 0.8) == (surface == "light"), (theme, surface, T["plot-bg"])
    bad = _plot_problems(theme, T)
    assert not bad, (theme, surface, bad)


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("surface", ["light", "dark"])
def test_3d_overlays_follow_a_white_or_black_background(theme, surface):
    """A White (Black) 3D background in a theme of the other lightness draws its overlays with the
    forced light (dark) set; one of the theme's own lightness keeps the theme's tokens. Either way the
    triad, cell, hover, selection and labels read on it."""
    bg = "#ffffff" if surface == "light" else "#000000"
    theme_dark = "color-scheme: dark" in BLOCKS.get(theme, "color-scheme: light")
    own = (surface == "dark") == theme_dark
    T = THEMES[theme] if own else forced(theme, surface)
    assert own or T["viewport-bg"] == bg
    bad = _view_problems(theme, T, bg)
    assert not bad, (theme, surface, "theme's own" if own else "forced", bad)


@pytest.mark.parametrize("theme", HIGH_CONTRAST)
def test_high_contrast_plots_and_axes_are_3(theme):
    T = THEMES[theme]
    marks = SERIES + ["axis-a", "axis-b", "axis-c", "cell"]
    low = {(m, bg): round(ratio(T[m], T[bg]), 2) for m in marks for bg in ("plot-bg", "viewport-bg", "surface")
           if ratio(T[m], T[bg]) < 3}
    assert not low, (theme, low)


def test_true_black_backgrounds():
    """True black: pure black surfaces (OLED)."""
    B = THEMES["black"]
    assert {B[k] for k in ("bg", "surface", "viewport-bg", "plot-bg")} == {"#000000"}


def _hue(h):
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))
    mx, mn = max(r, g, b), min(r, g, b)
    if mx - mn < 0.08:
        return None                                   # a gray or brown-gray: no hue to speak of
    if mx == r:
        return (60 * (g - b) / (mx - mn)) % 360
    if mx == g:
        return 60 * (b - r) / (mx - mn) + 120
    return 60 * (r - g) / (mx - mn) + 240


def test_sepia_is_warm_and_not_solarized_light():
    """Sepia is an e-reader page: brown ink on a darker tan than Solarized Light's cream, text softer
    than black on white yet over 4.5:1 everywhere (the tests above), and no blue in its accent, links,
    focus, status, plot series, 3D axes or file badges (warm hues only: no hue from 170° to 290°)."""
    S, L = THEMES["sepia"], THEMES["solarized-light"]
    assert _lum(S["bg"]) < _lum(L["bg"]) and _lum(S["surface"]) < _lum(L["surface"]), (S["bg"], L["bg"])
    assert delta_e(S["surface"], L["surface"]) > 8 and delta_e(S["text"], L["text"]) > 15
    assert 6 < ratio(S["text"], S["surface"]) < 12                        # softer than black on white
    warm = ["accent", "accent-hover", "focus", "hover-atom", "info", "success", "warning", "danger", "measure", "sel",
            "axis-a", "axis-b", "axis-c", "cell", "vol-plus", "vol-minus", "cmp-a", "cmp-b", "text", "text-2", "text-3"]
    warm += SERIES + [k for k in S if k.startswith("k-") and k != "k-on"]
    blue = {k: (S[k], round(_hue(S[k]))) for k in warm if _hue(S[k]) is not None and 170 <= _hue(S[k]) <= 290}
    assert not blue, blue
    # Solarized Light stays faithful to the published palette: its blue accent, base3 page, base02 text
    assert (L["surface"], L["text"], L["axis-c"], L["plot-10"]) == ("#fdf6e3", "#073642", "#268bd2", "#268bd2")


def published(theme, surface):
    """The tokens a plot draws with in Plot style › Publication: on the theme's own plot background
    (surface None, Plot background Follow) or a forced light or dark one."""
    T = THEMES[theme] if surface is None else forced(theme, surface)
    tone = surface or ("dark" if "color-scheme: dark" in BLOCKS.get(theme, "color-scheme: light") else "light")
    return {**T, **PUBLICATION[tone]}, tone


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("surface", [None, "light", "dark"], ids=["follow", "light", "dark"])
def test_publication_plot_style(theme, surface):
    """Okabe–Ito after a first series in ink, black (white) axes and text on a light (dark) plot: every
    series 2:1 on the plot (3:1 in the high-contrast themes), the first eight apart, the ink at the
    theme's text contrast, in every theme on every plot background."""
    T, tone = published(theme, surface)
    assert T["plot-ink"] == T["plot-1"] == ("#000000" if tone == "light" else "#ffffff")
    bad = _plot_problems(theme, T)
    if ratio(T["plot-ink"], T["plot-bg"]) < need(theme):
        bad["ink"] = round(ratio(T["plot-ink"], T["plot-bg"]), 2)
    assert not bad, (theme, surface, tone, bad)


def test_publication_series_are_okabe_ito():
    """The publication series are Okabe–Ito's hues: exact where they read, the orange and sky blue a shade
    deeper on a light plot and Paul Tol's olive for the yellow there (1.3:1 on white)."""
    okabe_ito = {"#000000", "#e69f00", "#56b4e9", "#009e73", "#f0e442", "#0072b2", "#d55e00", "#cc79a7"}
    light = [PUBLICATION["light"][f"plot-{k}"] for k in range(1, 9)]
    assert light[:5] == ["#000000", "#0072b2", "#d55e00", "#009e73", "#cc79a7"] and light[7] == "#999933", light
    dark = [PUBLICATION["dark"][f"plot-{k}"] for k in range(1, 9)]
    assert dark[0] == "#ffffff" and len(set(dark) & okabe_ito) >= 4, dark


def preset_tokens(pid, theme, surface):
    T = THEMES[theme] if surface is None else forced(theme, surface)
    tone = surface or ("dark" if "color-scheme: dark" in BLOCKS.get(theme, "color-scheme: light") else "light")
    return {**T, **PRESET_BLOCKS[pid][tone]}, tone


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("surface", [None, "light", "dark"], ids=["follow", "light", "dark"])
@pytest.mark.parametrize("pid", ["colorblind", "grayscale"])
def test_preset_series_read_on_every_plot(pid, theme, surface):
    """Plot style Colorblind safe and Print grayscale (1.1.6): every series 2:1 on the plot (3:1 in the
    high-contrast themes), the text tokens as the theme's; Colorblind safe's first eight apart; Print
    grayscale's grays tell series apart by line style and markers (styleFigure), not hue: they are gray."""
    T, tone = preset_tokens(pid, theme, surface)
    bad = _plot_problems(theme, T)
    if pid == "grayscale":
        bad.pop("alike", None)
        assert all(T[s][1:3] == T[s][3:5] == T[s][5:7] for s in SERIES), [T[s] for s in SERIES]
        assert ratio(T["plot-ink"], T["plot-bg"]) >= need(theme)
    assert not bad, (pid, theme, surface, tone, bad)


def test_preset_palettes_match_mace_plotting():
    """The light series of each preset are mace plotting's (mace/plotting/presets.py): screen and file agree."""
    from mace.plotting import presets
    for pid in ("publication", "colorblind", "grayscale"):
        assert [PRESET_BLOCKS[pid]["light"][f"plot-{k}"] for k in range(1, 11)] == presets.PRESETS[pid]["palette"], pid


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("surface", [None, "light", "dark"], ids=["follow", "light", "dark"])
def test_spin_pairs_read_on_every_plot(theme, surface):
    """Every preset's spin pair (presets.py; the interim blue / red of SPIN_COLORS by default) reads on
    every plot background of its tone: 2:1, 3:1 in the high-contrast themes."""
    from mace.plotting import presets
    T = THEMES[theme] if surface is None else forced(theme, surface)
    tone = surface or ("dark" if "color-scheme: dark" in BLOCKS.get(theme, "color-scheme: light") else "light")
    bad = {}
    for pid in presets.ORDER:
        pair = presets.spin_colors(pid, "band", tone)
        for k, c in pair.items():
            if ratio(c, T["plot-bg"]) < mark(theme):
                bad[f"{pid}.{k}"] = (c, round(ratio(c, T["plot-bg"]), 2))
    assert not bad, (theme, surface, bad)


def test_colorblind_theme_uses_okabe_ito():
    """The colorblind-safe theme's axes and first seven series are Okabe-Ito hues; its yellow
    (#f0e442, 1.3:1 on white) gives way to Paul Tol's olive, which reads on the white plot."""
    okabe_ito = {"#000000", "#e69f00", "#56b4e9", "#009e73", "#f0e442", "#0072b2", "#d55e00", "#cc79a7"}
    C = THEMES["cvd"]
    assert {C["axis-a"], C["axis-b"], C["axis-c"]} <= okabe_ito
    assert len({C[f"plot-{k}"] for k in range(1, 9)} & okabe_ito) == 7 and C["plot-8"] == "#999933"
    for v in ("light", "dark"):
        F = forced("cvd", v)
        assert len({F[f"plot-{k}"] for k in range(1, 9)} & (okabe_ito | {"#ffffff"})) >= 7, v
