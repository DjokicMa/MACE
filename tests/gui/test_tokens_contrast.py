"""WCAG 2.1 AA contrast of the GUI's design tokens (tokens.css), both themes.

No browser: the tokens are parsed from the stylesheet. Text and status
colors need 4.5:1 on every surface they are drawn on; the focus ring and the
borders that outline controls need 3:1 (WCAG 1.4.11); the file-kind chips
and the accent buttons need 4.5:1 for their own text. A failure is fixed in
tokens.css, never in a component.
"""
import re

import pytest

from conftest import REPO_ROOT

TOKENS = (REPO_ROOT / "mace" / "gui" / "static" / "css" / "tokens.css").read_text()


def _block(start):
    i = TOKENS.index(start)
    return TOKENS[i:TOKENS.index("}", i)]


def _vars(text):
    return dict(re.findall(r"--([\w-]+):\s*(#[0-9a-fA-F]{6})\b", text))


LIGHT = _vars(_block(":root {"))
DARK = {**LIGHT, **_vars(_block(':root[data-theme="dark"] {'))}
# the OS-dark block is the same as the explicit dark one
OS_DARK = {**LIGHT, **_vars(_block(':root:not([data-theme="light"]) {'))}


def _lum(h):
    c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def ratio(a, b):
    la, lb = _lum(a), _lum(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


SURFACES = ("bg", "surface", "surface-2", "surface-3", "accent-soft")
TEXT = ("text", "text-2", "text-3", "accent", "success", "danger", "warning", "info")
THEMES = {"light": LIGHT, "dark": DARK}


def test_os_dark_and_chosen_dark_are_the_same():
    assert OS_DARK == DARK


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("fg", TEXT)
def test_text_on_every_surface_is_4_5(theme, fg):
    T = THEMES[theme]
    low = {bg: round(ratio(T[fg], T[bg]), 2) for bg in SURFACES if ratio(T[fg], T[bg]) < 4.5}
    assert not low, (theme, fg, T[fg], low)


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("pair", [("on-accent", "accent"), ("on-accent", "accent-hover"), ("on-danger", "danger")])
def test_text_on_filled_buttons_is_4_5(theme, pair):
    T = THEMES[theme]
    assert ratio(T[pair[0]], T[pair[1]]) >= 4.5, (theme, pair)


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("ui", ["focus", "border-strong"])
def test_focus_ring_and_control_borders_are_3(theme, ui):
    T = THEMES[theme]
    low = {bg: round(ratio(T[ui], T[bg]), 2) for bg in ("bg", "surface", "surface-2") if ratio(T[ui], T[bg]) < 3}
    assert not low, (theme, ui, T[ui], low)


@pytest.mark.parametrize("theme", THEMES)
def test_file_kind_chips_are_4_5(theme):
    T = THEMES[theme]
    low = {k: round(ratio(T["k-on"], v), 2) for k, v in T.items() if k.startswith("k-") and k != "k-on"
           and ratio(T["k-on"], v) < 4.5}
    assert not low, (theme, low)
