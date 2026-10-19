"""Inspector › Rendering › Style: the Atom Size and Bond Radius sliders change
what the structure view draws, live, under every style they apply to; picking
a style shows its values in the sliders; editing a slider afterwards makes the
dropdown Custom. Real browser, software GL; Firefox or Chromium (MACE_GUI_BROWSER)."""
import time

import pytest

import guiharness as H

pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")

from test_presets_e2e import app, choose, expand, fresh, open_doc, values  # noqa: E402,F401


def slide(drv, key, value):
    """Move a Display slider as a user does (input, then change), and let the view redraw."""
    el = expand(drv).find_element("id", f"dp-{key}")
    drv.execute_script("""const e = arguments[0]; e.value = arguments[1];
      e.dispatchEvent(new Event('input', {bubbles: true})); e.dispatchEvent(new Event('change', {bubbles: true}));""", el, value)
    H.wait(drv, "return Math.abs(window.__mace.display.of(window.__mace.activeDoc())[arguments[0]] - arguments[1]) < 1e-9", 5, key, value)
    time.sleep(0.6)


def drawn(drv):
    """The first atom's sphere {scale, radius} and stick radius as 3Dmol draws them."""
    return drv.execute_script("""const a = window.__mace.viewport.scene().layers.structure.selectedAtoms({})[0].style;
      return {sphere: a.sphere ? (a.sphere.radius || a.sphere.scale) : null, stick: a.stick ? a.stick.radius : null}""")


def slider_value(drv, key):
    return float(expand(drv).find_element("id", f"dp-{key}").get_attribute("value"))


@pytest.mark.parametrize("style", ["default", "vesta", "grayscale"])
def test_atom_size_and_bond_radius_change_the_drawing(fresh, style):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    choose(drv, "style", style)
    before = drawn(drv)
    slide(drv, "atomScale", 0.8)
    after = drawn(drv)
    assert after["sphere"] == pytest.approx(before["sphere"] * 2, rel=1e-3), (before, after)
    slide(drv, "bondRadius", 0.3)
    assert drawn(drv)["stick"] == pytest.approx(0.3), drawn(drv)
    assert values(drv)["stylePreset"] == "custom"
    assert not H.errors(drv)


def test_wireframe_stick_radius_and_spacefill_scale_follow_the_sliders(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    choose(drv, "style", "wireframe")
    slide(drv, "bondRadius", 0.2)
    assert drawn(drv)["stick"] == pytest.approx(0.2)
    choose(drv, "style", "spacefill")
    before = drawn(drv)["sphere"]
    slide(drv, "atomScale", 0.2)
    assert drawn(drv)["sphere"] == pytest.approx(before / 2, rel=1e-3)
    # the Display section's own Stick style: 0.16 Å at the default bond radius, following the slider
    choose(drv, "style", "default")
    set_atom_style(drv, "stick")
    assert drawn(drv)["stick"] == pytest.approx(0.16)
    slide(drv, "bondRadius", 0.24)
    assert drawn(drv)["stick"] == pytest.approx(0.32)
    set_atom_style(drv, "ballstick")
    assert not H.errors(drv)


def set_atom_style(drv, value):
    drv.execute_script("""const e = document.getElementById('opt-style'); e.value = arguments[0];
      e.dispatchEvent(new Event('change', {bubbles: true}))""", value)
    time.sleep(0.8)


def atom_style_box(drv):
    return drv.execute_script("return document.getElementById('opt-style').value")


def test_picking_a_style_shows_its_values_then_an_edit_makes_it_custom(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    for style in ["polyhedral", "wireframe", "default"]:
        choose(drv, "style", style)
        want = drv.execute_script("return window.__mace.display.STYLE_PRESETS[arguments[0]]", style)
        assert slider_value(drv, "atomScale") == pytest.approx(want["atomScale"]), style
        assert slider_value(drv, "bondRadius") == pytest.approx(want["bondRadius"]), style
        assert expand(drv).find_element("id", "dp-stylePreset").get_attribute("value") == style
    # the Display section's Atom style shows what a style draws, and comes back after it
    assert atom_style_box(drv) == "ballstick"
    choose(drv, "style", "spacefill")
    assert atom_style_box(drv) == "spacefill"
    choose(drv, "style", "wireframe")
    assert atom_style_box(drv) == "stick"
    set_atom_style(drv, "ballstick")              # edited afterwards: the style is Custom, drawn as picked
    assert values(drv)["atomDraw"] == "auto" and values(drv)["stylePreset"] == "custom"
    assert expand(drv).find_element("id", "dp-stylePreset").get_attribute("value") == "custom"
    choose(drv, "style", "default")
    assert atom_style_box(drv) == "ballstick"
    choose(drv, "light", "publication")
    assert slider_value(drv, "ambientOcclusion") == pytest.approx(0.5)
    slide(drv, "atomScale", 0.6)
    assert expand(drv).find_element("id", "dp-stylePreset").get_attribute("value") == "custom"
    assert expand(drv).find_element("id", "dp-renderPreset").get_attribute("value") == "publication"
    assert not H.errors(drv)
