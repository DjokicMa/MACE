"""The Rendering group's 3Dmol patch (polish A6): every shader text it
replaces is in the vendored 3Dmol (a 3Dmol upgrade fails here, loudly), the
patched programs carry the ambient uniform, and ambient light, the light
direction and the default look do what they say on real pixels (software
GL, headless Firefox)."""
import json
import re
import time

import pytest

import guiharness as H

STATIC = H.REPO_ROOT / "mace" / "gui" / "static"


def guards():
    src = (STATIC / "js" / "viewport" / "rendering.js").read_text()
    block = src[src.index("export const GUARDS"):src.index("];", src.index("export const GUARDS"))]
    return re.findall(r'\["([^"]+)",\s*\n?\s*"([^"]+)"\]', block)


def test_every_guard_is_in_the_vendored_3dmol():
    vendored = (STATIC / "vendor" / "3dmol" / "3Dmol-min.js").read_text()
    g = guards()
    assert len(g) == 4, g
    for old, new in g:
        assert old in vendored, f"3Dmol changed: {old!r} is gone (update viewport/rendering.js GUARDS)"
        assert "maceAmbient" in new


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    pytest.importorskip("selenium.webdriver")
    pytest.importorskip("ase")             # the server reads the structures with it
    tmp = tmp_path_factory.mktemp("render")
    proc, base, _ = H.start_server(H.examples_copy(tmp), tmp / "xdg")
    drv = H.firefox(900, 700)
    try:
        H.load(drv, base)
        drv.execute_async_script("const d = arguments[0]; window.__mace.open('structures/MgO.cif').then(() => d(1))")
        time.sleep(1.2)
        yield drv
    finally:
        drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


# luminance of the drawn atoms' pixels in the canvas: [mean of the darkest quarter, max, min]
LUM = """
const v = window.MACE.state.viewers['viewer-main']; v.paintNow();
const cv = v.getCanvas(), url = cv.toDataURL('image/png');
return new Promise((ok) => { const img = new Image(); img.onload = () => {
  const c = document.createElement('canvas'); c.width = cv.width; c.height = cv.height;
  const g = c.getContext('2d'); g.drawImage(img, 0, 0);
  const d = g.getImageData(0, 0, c.width, c.height).data, bg = [d[0], d[1], d[2]], L = [];
  for (let k = 0; k < d.length; k += 16) {
    if (Math.abs(d[k] - bg[0]) + Math.abs(d[k + 1] - bg[1]) + Math.abs(d[k + 2] - bg[2]) < 40) continue;
    L.push(arguments[0] === 'max' ? Math.max(d[k], d[k + 1], d[k + 2]) : 0.2126 * d[k] + 0.7152 * d[k + 1] + 0.0722 * d[k + 2]);
  }
  L.sort((a, b) => a - b);
  const q = L.slice(0, Math.max(1, L.length >> 2));
  ok([q.reduce((a, b) => a + b, 0) / q.length, L[L.length - 1], L[Math.floor(L.length / 100)], L.length]); }; img.src = url; });"""


@pytest.mark.gui
def test_patched_programs_carry_the_uniform(page):
    r = page.execute_script("""const v = window.MACE.state.viewers['viewer-main'], r = v.renderer;
      return {patched: r.maceAmbientPatch, programs: r._programs.filter((p) => p.code.includes('directionalLightColor')).map((p) => p.code.includes('maceAmbient'))};""")
    assert r["patched"] and r["programs"] and all(r["programs"]), r


@pytest.mark.gui
def test_default_light_is_3dmols(page):
    r = page.execute_script("""const v = window.MACE.state.viewers['viewer-main'];
      const L = v.scene.children.find((c) => c.position && 'intensity' in c);
      return [L.position.x, L.position.y, L.position.z, L.intensity, v.renderer.maceAmbient];""")
    n = (0.2 ** 2 + 0.2 ** 2 + 1) ** 0.5
    assert r[:3] == pytest.approx([0.2 / n, 0.2 / n, 1 / n], abs=1e-9) and r[3] == 1 and r[4] == 0


@pytest.mark.gui
@pytest.mark.parametrize("style", ["ballstick", "spacefill"])
def test_ambient_lifts_the_dark_side(page, style):
    page.execute_script("""const st = document.getElementById('opt-style'); st.value = arguments[0]; st.dispatchEvent(new Event('change'));""", style)
    time.sleep(0.8)
    means = []
    for a in (0, 0.2, 0.4):
        page.execute_script("window.__mace.settings.set('ambient', arguments[0])", a)
        time.sleep(0.5)
        means.append(page.execute_script(LUM, "lum")[0])
    page.execute_script("window.__mace.settings.set('ambient', 0)")
    assert means[0] < means[1] < means[2], means


@pytest.mark.gui
@pytest.mark.parametrize("direction", ["upper-left", "top"])
def test_off_axis_light_keeps_a_floor(page, direction):
    """Upper Left and Top raise ambient to 15% (the Inspector does it); the dark side stays above 10% of the brightest."""
    page.execute_script("window.__mace.settings.set('showCell', false); window.__mace.settings.set('lightDirection', arguments[0]);"
                        "window.__mace.settings.set('ambient', 0.15)", direction)
    time.sleep(0.6)
    # brightness as the largest channel: a red O and an orange Mg each keep their dark side above the floor
    mean_dark, mx, mn, n = page.execute_script(LUM, "max")
    page.execute_script("window.__mace.settings.set('lightDirection', 'front'); window.__mace.settings.set('ambient', 0);"
                        "window.__mace.settings.set('showCell', true)")
    # the 1st percentile, not the single darkest pixel (antialiased edges against bonds)
    assert n > 100 and mn > 0.1 * mx, (mn, mx)


@pytest.mark.gui
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_theme_outline_stands_out_on_the_background(page, theme):
    """An outline with no color chosen is drawn black on a white background
    (Publication) and white on a black one, in either theme; on the theme's
    background it is the theme's text color."""
    page.execute_script("const s = window.__mace.settings; s.set('theme', arguments[0]); s.set('outlineColor', ''); s.set('outline', true);", theme)
    drawn = {}
    for bg in ("white", "black", "theme"):
        page.execute_script("window.__mace.settings.set('background', arguments[0])", bg)
        time.sleep(0.4)
        drawn[bg] = page.execute_script("const v = window.MACE.state.viewers['viewer-main']; v.paintNow(); return JSON.parse(v.maceLook)[5]")
    text = page.execute_script("return getComputedStyle(document.documentElement).getPropertyValue('--text').trim()")
    page.execute_script("const s = window.__mace.settings; s.set('outline', false); s.set('background', 'theme'); s.set('theme', 'system');")
    assert drawn == {"white": "#000000", "black": "#ffffff", "theme": text}, drawn
