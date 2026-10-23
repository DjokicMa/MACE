"""The 3D overlays of 1.1.6 never vanish on their background (tokens.css, every theme).

The volume view's slice plane (outline --cell, fill --accent), the Brillouin zone view (edges --text-3,
labels --text, path --accent) and the isosurfaces' + and − colors draw in the 3D view's tokens
(app.js vpVar), which are the theme's on its own background and a forced light or dark set on a White
or Black background of the other lightness. Each must stand out on the background it is drawn on: 3:1
(the labels 4.5:1) on --viewport-bg, in every theme, for the theme's own set and both forced sets.
The slice preview draws on the inspector's --surface-2 (js/viewport/contrast.js moves a token that is
short of 3:1 there, which three editor themes' axis colors are: checked here so a new one is seen).
"""
import importlib.util

import pytest

from conftest import REPO_ROOT

spec = importlib.util.spec_from_file_location("tokens_contrast", REPO_ROOT / "tests" / "gui" / "test_tokens_contrast.py")
TC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TC)

MARKS = {"cell": 3, "accent": 3, "text-3": 3, "text": 4.5, "vol-plus": 3, "vol-minus": 3}
# the axis letters of the preview that js/viewport/contrast.js adjusts on --surface-2 (known, kept short)
PREVIEW_ADJUSTED = {("catppuccin-latte", "axis-b"), ("nord", "axis-a"), ("solarized-light", "axis-b")}


def sets(theme):
    yield "own", TC.THEMES[theme]
    for v in ("light", "dark"):
        yield v, TC.forced(theme, v)


@pytest.mark.parametrize("theme", sorted(TC.THEMES))
def test_overlay_marks_stand_out_on_the_viewport_background(theme):
    bad = []
    for name, T in sets(theme):
        bg = T["viewport-bg"]
        for tok, need in MARKS.items():
            r = TC.ratio(T[tok], bg)
            if r < need:
                bad.append((name, tok, round(r, 2)))
    assert not bad, bad


@pytest.mark.parametrize("theme", sorted(TC.THEMES))
def test_preview_marks_on_the_inspector_surface(theme):
    T = TC.THEMES[theme]
    short = {(theme, t) for t in ("text-2", "accent", "axis-a", "axis-b", "axis-c") if TC.ratio(T[t], T["surface-2"]) < 3}
    assert short <= PREVIEW_ADJUSTED, short - PREVIEW_ADJUSTED
