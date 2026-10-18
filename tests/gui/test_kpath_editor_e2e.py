"""The k-path editor (mace/gui/static/js/kpath_editor.js) in the band plot's options and in
Calculation Setup, in Firefox, by pointer and keyboard (no screenshots asserted).

The usability script, each step found from what the screen says (a tooltip or a label on
the control, no docs): on graphite's path Γ-M-K-Γ-A-L-H-A|L-M|H-K
  1. hide K→Γ        the eye on the K→Γ arrow ("Hide K → Γ")
  2. reverse Γ→M     the Γ→M arrow ("Reverse Γ → M")
  3. move A|L first  the jump chip ("Jump A|L: drag to move the part after it"); arrow keys
  4. remove H        the H chip's × ("Remove H ...")
  5. reset           "Reset to computed path"
and the plot (ticks, gaps, the band data flipped) follows each step; Mod+Z undoes one."""
import json
import shutil
import sys
import time

import pytest

import guiharness as H
import p4util as U
import setup_routes as R

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("plotly")
sys.path.insert(0, str(H.REPO_ROOT / "tests"))
from test_gui_results import write_band_path  # noqa: E402
from selenium.webdriver.common.action_chains import ActionChains  # noqa: E402
from selenium.webdriver.common.keys import Keys  # noqa: E402

GRAPHITE = "GAMMA-M-K-GAMMA-A-L-H-A-|-L-M-|-H-K"
TICKS = (0.0, 0.78265, 1.20908, 2.05632, 2.32554, 3.10818, 3.53462, 4.38186, 4.65597, 4.93007)
HEX_POINTS = {"GAMMA": (0, 0, 0), "M": (0.5, 0, 0), "K": (1 / 3, 1 / 3, 0), "A": (0, 0, 0.5), "L": (0.5, 0, 0.5),
              "H": (1 / 3, 1 / 3, 0.5)}
GRAPHITE_OUT = """ CRYSTAL - PROPERTIES
 DIRECT LATTICE VECTOR COMPONENTS (ANGSTROM)
         2.12415   -1.22638    0.00000
        -0.00000    2.45276    0.00000
         0.00000    0.00000    6.17522
"""


@pytest.fixture(scope="module", params=["firefox", "chromium"])
def app(request, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("kpath")
    root = R.served_root(tmp)
    for sub in ("gr", "gr2"):
        (root / sub).mkdir()
        write_band_path(root / sub, stem="g_band", path=GRAPHITE, ticks=TICKS, points=HEX_POINTS)
        # the cell, as a properties run's output prints it (the zone view reads it)
        (root / sub / "g_band.out").write_text(GRAPHITE_OUT)
    (root / "slip").mkdir()
    write_band_path(root / "slip")
    real = H.firefox
    if request.param == "chromium":     # start_app opens Firefox; the same app in Chromium
        H.firefox = lambda w, h, download_dir=None, **kw: H.chromium(w, h, download_dir=download_dir)
    try:
        a = U.start_app(tmp, root=root, width=1500, height=1000)
    finally:
        H.firefox = real
    try:
        yield a
    finally:
        U.stop_app(a)


@pytest.fixture
def fresh(app):
    drv = app[0]
    U.reset(drv, 1500, 1000)
    drv.execute_script("Object.keys(localStorage).filter((k) => k.startsWith('mace.gui.plot.') || k.startsWith('mace.gui.kpath')"
                       " || k.startsWith('mace.gui.setup.')).forEach((k) => localStorage.removeItem(k))")
    if drv.execute_script("return !!(window.__mace.ctx && window.__mace.ctx().setupOpen)"):
        drv.execute_script("window.__mace.run('calc.setup')")          # back to the plot's options
        time.sleep(0.3)
    return app


def open_plot(drv, path):
    U.open_doc(drv, path, wait_atoms=False)
    H.wait(drv, "const v = window.__mace.plotView, s = v && v.state(); return s && s.drawn && v.el().dataset.drawn === arguments[0]", 60, path)
    H.wait(drv, "return !!document.querySelector('.po-kpath .kp-strip')", 20)
    time.sleep(0.3)


FIG = """const p = document.querySelector('#plot-host .pv-plot'), t = p.data.find((x) => x.meta === 'up');
  return {ticks: p.layout.xaxis.ticktext.map((s) => s.replace(/<br>|\\u00a0|<\\/?sub>/g, '')), vals: p.layout.xaxis.tickvals,
          x: t.x, y: t.y, kpath: window.__mace.plotView.state().opts.kpath}"""


def fig(drv):
    return drv.execute_script(FIG)


def seg(drv, name):
    els = [e for e in drv.find_elements("css selector", ".po-kpath .kp-seg") if f": {name}" in e.get_attribute("aria-label")]
    assert els, name
    return els[0]


def see(drv, el):
    """The inspector scrolls: bring a control into view before the pointer goes to it."""
    drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", el)
    time.sleep(0.15)
    return el


def hover_click(drv, chip, sel):
    ActionChains(drv).move_to_element(see(drv, chip)).perform()
    b = chip.find_element("css selector", sel)
    assert b.get_attribute("data-tip"), "every control says what it does"
    b.click()


def wait_kpath(drv, test):
    H.wait(drv, f"const k = window.__mace.plotView.state().opts.kpath; return {test}", 10)
    time.sleep(0.3)


def first_band_value_at(drv, k_label):
    """The first band's energy where the plot puts the tick k_label (for the data-flip check)."""
    f = fig(drv)
    i = f["ticks"].index(k_label)
    x = f["vals"][i]
    best = min((j for j, v in enumerate(f["x"]) if v is not None), key=lambda j: abs(f["x"][j] - x))
    return f["y"][best]


def test_five_step_usability_script(fresh):
    drv = fresh[0]
    open_plot(drv, "gr/g_band.BAND.DAT")
    help_text = drv.find_element("css selector", ".po-kpath .kp-help").text
    assert "reverse" in help_text and "eye" in help_text and "Drag" in help_text
    f0 = fig(drv)
    assert f0["ticks"] == ["Γ", "M", "K", "Γ", "A", "L", "H", "A|L", "M|H", "K"] and f0["kpath"] is None
    m_energy = first_band_value_at(drv, "M")

    # 1. hide K→Γ: a jump K|Γ appears, the plot leaves the segment out
    hover_click(drv, seg(drv, "K → Γ"), ".kp-eye")
    wait_kpath(drv, "!!k && k.some((s) => s.hidden)")
    f1 = fig(drv)
    assert "K|Γ" in f1["ticks"] and f1["vals"][-1] < f0["vals"][-1]
    assert drv.find_element("css selector", ".po-kpath .kp-seg.kp-off")

    # 2. reverse Γ→M: the plot now starts at M, with the data of M
    hover_click(drv, seg(drv, "Γ → M"), ".kp-rev")
    wait_kpath(drv, "k[0].from === 'M'")
    f2 = fig(drv)
    # M→Γ then M→K: a jump at Γ|M
    assert f2["ticks"][:2] == ["M", "Γ|M"] and f2["y"][0] == pytest.approx(m_energy, abs=1e-6)

    # 3. move A|L first: the jump chip, then ← until its part leads
    gap = next(e for e in drv.find_elements("css selector", ".po-kpath .kp-gap") if "Jump A | L" in e.get_attribute("aria-label"))
    assert "drag" in gap.get_attribute("data-tip").lower()
    drv.execute_script("arguments[0].focus()", gap)                 # Tab or a click on its label selects it
    for _ in range(4):
        if fig(drv)["ticks"][0] == "L":
            break
        drv.switch_to.active_element.send_keys(Keys.ARROW_LEFT)
        time.sleep(0.4)
    assert fig(drv)["ticks"][0] == "L"

    # 4. remove H: its chip's × says what it does and shows it first (no data runs from L to A:
    # both segments are hidden, a jump L|... is left)
    h = next(e for e in drv.find_elements("css selector", ".po-kpath .kp-pt") if e.text.strip().startswith("H"))
    x = h.find_element("css selector", ".kp-xpt")
    assert x.get_attribute("data-tip") == "Remove H: hides L → H and H → A" == x.get_attribute("aria-label")
    ActionChains(drv).move_to_element(see(drv, x)).perform()
    H.wait(drv, "return document.querySelectorAll('.po-kpath .kp-seg.kp-prev').length === 2", 5)
    H.wait(drv, "return document.querySelector('#plot-host .pv-plot').layout.shapes.filter((s) => s.name === 'kp-affected').length === 2", 5)
    x.click()
    wait_kpath(drv, "k.some((s) => s.from === 'L' && s.to === 'H' && s.hidden) && k.some((s) => s.from === 'H' && s.to === 'A' && s.hidden)")
    assert not any(t == "H" for t in fig(drv)["ticks"])

    # Mod+Z on the editor undoes the last step, Shift+Mod+Z does it again
    drv.execute_script("document.querySelector('.po-kpath .kp-seg').focus()")
    U.keys(drv, Keys.CONTROL, "z")
    wait_kpath(drv, "!k.some((s) => s.from === 'L' && s.to === 'H' && s.hidden)")
    U.keys(drv, Keys.CONTROL, Keys.SHIFT, "z")
    wait_kpath(drv, "k.some((s) => s.from === 'L' && s.to === 'H' && s.hidden)")

    # 5. reset
    b = next(e for e in drv.find_elements("css selector", ".po-kpath [data-kp-act=reset]"))
    assert b.text == "Reset to computed path"
    b.click()
    wait_kpath(drv, "k === null")
    assert fig(drv)["ticks"] == f0["ticks"]
    U.no_errors(drv)


def test_keyboard_and_the_empty_state(fresh):
    drv = fresh[0]
    open_plot(drv, "gr/g_band.BAND.DAT")
    first = drv.find_elements("css selector", ".po-kpath .kp-seg")[0]
    assert "Arrow keys move it, R reverses, H hides, Delete removes" in first.get_attribute("aria-label")
    drv.execute_script("arguments[0].focus()", first)               # Tab lands on it the same way
    drv.switch_to.active_element.send_keys("r")                     # R reverses
    wait_kpath(drv, "k && k[0].from === 'M'")
    drv.switch_to.active_element.send_keys(Keys.ARROW_RIGHT)        # → moves it one place on
    wait_kpath(drv, "k[1].from === 'M' && k[1].to === 'GAMMA'")
    drv.switch_to.active_element.send_keys(Keys.DELETE)             # Delete removes it
    wait_kpath(drv, "k.length === 8")
    # hiding every segment is refused, and says why
    for _ in range(12):                                             # the eye of every segment but one
        n = drv.execute_script("const v = document.querySelectorAll('.po-kpath .kp-seg:not(.kp-off)');"
                               "if (v.length > 1) v[v.length - 1].querySelector('.kp-eye').click(); return v.length")
        if n <= 1:
            break
        time.sleep(0.3)
    last = drv.find_elements("css selector", ".po-kpath .kp-seg:not(.kp-off)")
    assert len(last) == 1
    drv.execute_script("arguments[0].focus()", last[0])
    drv.switch_to.active_element.send_keys("h")
    H.wait(drv, "const m = document.querySelector('.po-kpath .kp-flash'); return m && m.textContent.includes('At least one segment must stay visible')", 5)
    assert len(drv.find_elements("css selector", ".po-kpath .kp-seg:not(.kp-off)")) == 1
    U.no_errors(drv)


def test_jumps_continuous_presets_and_export(fresh):
    drv, downloads = fresh[0], fresh[5]
    open_plot(drv, "gr/g_band.BAND.DAT")
    drv.find_element("css selector", ".po-kpath [data-kp-act=jumps_first]").click()
    wait_kpath(drv, "k && k[0].from === 'L'")
    assert fig(drv)["ticks"][:2] == ["L", "M|H"]
    drv.find_element("css selector", ".po-kpath [data-kp-act=continuous_only]").click()
    wait_kpath(drv, "k.length === 7")
    assert all("|" not in t for t in fig(drv)["ticks"])
    # the plotted data as CSV follows the edited path
    for f in downloads.iterdir():
        f.unlink()
    drv.find_element("css selector", "#plot-host .pk-export").click()
    drv.find_element("css selector", ".pk-menu [data-v=csv]").click()
    end = time.time() + 20
    while time.time() < end and not any(f.name.endswith("_data.csv") for f in downloads.iterdir()):
        time.sleep(0.2)
    csv = []
    while time.time() < end and len(csv) < 2:          # a download can be seen before it is written out
        csv = next(f for f in downloads.iterdir() if f.name.endswith("_data.csv")).read_text().splitlines()
        time.sleep(0.2)
    path_x = [float(r.split(",")[0]) for r in csv[1:]]
    assert max(path_x) == pytest.approx(fig(drv)["vals"][-1], abs=1e-6)
    # a preset: saved here, offered on another plot of the same path, not on another lattice's
    drv.find_element("css selector", ".po-kpath [data-kp-preset-name]").send_keys("hex continuous")
    drv.find_element("css selector", ".po-kpath [data-kp-act=preset_save]").click()
    H.wait(drv, "return [...document.querySelectorAll('.po-kpath [data-kp-preset] option')].some((o) => o.value === 'hex continuous')", 5)
    open_plot(drv, "slip/slip_band.BAND.DAT")
    opt = drv.execute_script("return [...document.querySelectorAll('.po-kpath [data-kp-preset] option')].map((o) => [o.value, o.disabled])")
    assert ["hex continuous", True] in opt
    open_plot(drv, "gr2/g_band.BAND.DAT")
    assert fig(drv)["kpath"] is None
    drv.find_element("css selector", ".po-kpath [data-kp-act=preset_apply]").click()
    wait_kpath(drv, "k && k.length === 7")
    U.no_errors(drv)


def test_drag_and_drop_reorders(fresh):
    """HTML5 drag events on the strip: a segment onto another takes its place; a jump moves its part."""
    drv = fresh[0]
    open_plot(drv, "gr/g_band.BAND.DAT")
    drv.execute_script("""const segs = document.querySelectorAll('.po-kpath .kp-seg');
      const dt = new DataTransfer(), fire = (el, t) => el.dispatchEvent(new DragEvent(t, {bubbles: true, cancelable: true, dataTransfer: dt}));
      fire(segs[2], 'dragstart'); fire(segs[0], 'dragover'); fire(segs[0], 'drop'); fire(segs[2], 'dragend');""")
    wait_kpath(drv, "k && k[0].from === 'K' && k[0].to === 'GAMMA'")
    drv.execute_script("""const g = document.querySelectorAll('.po-kpath .kp-gap'), s = document.querySelectorAll('.po-kpath .kp-seg');
      const dt = new DataTransfer(), fire = (el, t) => el.dispatchEvent(new DragEvent(t, {bubbles: true, cancelable: true, dataTransfer: dt}));
      fire(g[g.length - 1], 'dragstart'); fire(s[0], 'dragover'); fire(s[0], 'drop');""")
    wait_kpath(drv, "k[0].from === 'H' && k[0].to === 'K'")
    U.no_errors(drv)


# ---------------------------------------------------------------- Calculation Setup (a path to compute)
def open_band_setup(drv):
    from test_setup_e2e import open_setup
    open_setup(drv, "outputs/Si_sp.out")
    drv.execute_script("document.querySelector(`#inspector .su-calc [data-calc='BAND']`).click()")
    H.wait(drv, "const s = window.__mace.setup.state(); return s.calc === 'BAND' && s.result && !s.busy && s.result.kpath", 120)
    H.wait(drv, "return !!document.querySelector('#inspector .su-kpath .kp-strip')", 20)


def setup_state(drv):
    return drv.execute_script("""const s = window.__mace.setup.state();
      return {deck: s.result && s.result.deck, patches: s.patches, json: s.jsonText, busy: !!s.busy || !!s.pendingRun,
              kpath: s.result && s.result.kpath}""")


def wait_setup(drv, test, timeout=120):
    H.wait(drv, f"const s = window.__mace.setup.state(); return !s.busy && !s.pendingRun && s.result && ({test})", timeout)
    time.sleep(0.3)


def test_setup_edit_add_point_warning_and_cli_parity(fresh, tmp_path):
    drv, base, root = fresh[0], fresh[1], fresh[2]
    open_band_setup(drv)
    st = setup_state(drv)
    assert [(s["from"], s["to"]) for s in st["kpath"]["computed"]] == [("X", "GAMMA"), ("GAMMA", "L"), ("L", "W"), ("W", "GAMMA")]
    host = drv.find_element("css selector", "#inspector .su-kpath")
    # reverse X→Γ: the deck's first label record follows
    ActionChains(drv).move_to_element(see(drv, host.find_elements("css selector", ".kp-seg")[0])).perform()
    host.find_elements("css selector", ".kp-seg .kp-rev")[0].click()
    wait_setup(drv, "s.result.deck && s.result.deck.split('\\n')[3] === 'G X'")
    st = setup_state(drv)
    assert st["patches"]["band_path"][0] == {"from": "GAMMA", "to": "X"}
    assert '"band_path"' in st["json"] and "(edited)" in st["deck"].splitlines()[1]
    # remove L: its two segments become Γ→W, and the inline warning says L is no longer visited
    # (the panel draws itself again when its run ends: the chip is looked up as it is clicked)
    tip = drv.execute_script("""const p = [...document.querySelectorAll('#inspector .su-kpath .kp-pt')].find((e) => e.textContent.startsWith('L'));
      const b = p.querySelector('.kp-xpt'); const t = b.dataset.tip; b.click(); return t""")
    assert tip == "Remove L: joins Γ → W"
    wait_setup(drv, "s.patches.band_path && s.patches.band_path.length === 3")
    H.wait(drv, "const w = document.querySelector('#inspector .su-kpath .kp-msg.warn'); return w && w.textContent.includes('no longer visits L')", 20)
    # add a point: a bad coordinate is refused in place; a good one is computed (the deck goes to coordinates)
    form = drv.find_element("css selector", "#inspector .su-kpath [data-kp-add]")
    form.find_element("name", "label").send_keys("P")
    for n, v in zip(("kx", "ky", "kz"), ("1/4", "abc", "0.25")):
        form.find_element("name", n).send_keys(v)
    form.find_element("css selector", "button[type=submit]").click()
    msg = drv.find_element("css selector", "#inspector .su-kpath .kp-addmsg").text
    assert "three numbers" in msg and form.find_element("name", "ky").get_attribute("aria-invalid") == "true"
    form.find_element("name", "ky").clear()
    form.find_element("name", "ky").send_keys("0.25")
    form.find_element("css selector", "button[type=submit]").click()
    wait_setup(drv, "s.patches.band_path.length === 4 && s.result.deck && s.result.deck.split('\\n')[2].split(' ')[1] !== '0'")
    st = setup_state(drv)
    assert st["patches"]["band_path"][-1] == {"from": "GAMMA", "to": "P", "k_to": [0.25, 0.25, 0.25]}
    assert st["deck"].splitlines()[1].endswith("GAMMA-X-|-GAMMA-W-GAMMA-P")
    # the same JSON through the CLI writes the same deck (the GUI writes opt2d3's own key)
    d = tmp_path / "cli"
    shutil.copytree(root / "outputs", d / "in")
    (d / "cfg.json").write_text(st["json"])
    R.run_cli_manual(["opt2d3", "-i", "in/Si_sp.out", "-t", "BAND", "--config-file", "cfg.json", "--output-dir", "out"], R.ENTER, d)
    assert next((d / "out").glob("*.d3")).read_text() == st["deck"]
    # undo in Setup takes the added point back out
    drv.execute_script("document.querySelector('#inspector .su-kpath .kp-seg').focus()")
    U.keys(drv, Keys.CONTROL, "z")
    wait_setup(drv, "s.patches.band_path && s.patches.band_path.length === 3")
    # reset: no band_path, the deck as computed
    drv.find_element("css selector", "#inspector .su-kpath [data-kp-act=reset]").click()
    wait_setup(drv, "!('band_path' in s.patches)")
    st = setup_state(drv)
    assert "band_path" not in st["json"] and "(edited)" not in st["deck"]
    U.no_errors(drv)


def test_cli_prompt_writes_the_same_key(tmp_path):
    """opt2d3's own prompt: the band_path it saves is what Setup writes for the same edit."""
    d = tmp_path / "p"
    shutil.copytree(H.REPO_ROOT / "examples" / "gui" / "outputs", d / "in")
    (d / "in" / "Si_sp.f9").touch()
    answers = [("Select method", ""), ("Select format", ""), ("Total number of k-points", ""), ("Select option", ""),
               ("Print eigenvalues", ""), ("Edit the band k-path", "y"), ("k-path edit", "r 1"),
               ("k-path edit", "p L"), ("k-path edit", "a P 0.25 0.25 0.25"), ("k-path edit", "")]
    R.run_cli_manual(["opt2d3", "-i", "in/Si_sp.out", "-t", "BAND", "--edit-kpath", "--save-config", "--options-file", "cfg.json",
                      "--output-dir", "out"], answers, d)
    cfg = json.loads((d / "cfg.json").read_text())
    conf = cfg.get("configuration", cfg)
    assert conf["band_path"] == [{"from": "GAMMA", "to": "X"}, {"from": "GAMMA", "to": "W"}, {"from": "W", "to": "GAMMA"},
                                 {"from": "GAMMA", "to": "P", "k_to": [0.25, 0.25, 0.25]}]
    deck = next((d / "out").glob("*.d3")).read_text()
    assert deck.splitlines()[1].endswith("GAMMA-X-|-GAMMA-W-GAMMA-P")


def test_continuous_only_says_and_shows_what_it_keeps(fresh):
    drv = fresh[0]
    open_plot(drv, "gr/g_band.BAND.DAT")
    b = drv.find_element("css selector", ".po-kpath [data-kp-act=continuous_only]")
    assert b.get_attribute("data-tip").startswith("Keeps the longest continuous part: Γ → M → K → Γ → A → L → H → A")
    ActionChains(drv).move_to_element(see(drv, b)).perform()
    H.wait(drv, "return document.querySelectorAll('.po-kpath .kp-seg.kp-dim').length === 2", 5)
    H.wait(drv, "return document.querySelector('#plot-host .pv-plot').layout.shapes.filter((s) => s.name === 'kp-keep').length === 2", 5)
    ActionChains(drv).move_to_element(see(drv, drv.find_element("css selector", ".po-kpath .kp-help"))).perform()
    H.wait(drv, "return !document.querySelector('.po-kpath .kp-seg.kp-dim')", 5)
    U.no_errors(drv)


BZ = "const e = document.querySelector('.po-kpath .kp-bz'); return e && e.__bz"


def bz_state(drv, sel=".po-kpath .kp-bz"):
    return drv.execute_script(f"const e = document.querySelector('{sel}'); return e && e.__bz.state()")


def test_brillouin_zone_view_is_linked_to_the_strip(fresh):
    drv = fresh[0]
    open_plot(drv, "gr/g_band.BAND.DAT")
    H.wait(drv, "const e = document.querySelector('.po-kpath .kp-bz'); return e && e.__bz.state().loaded && e.__bz.state().viewer", 60)
    st = bz_state(drv)
    assert set(st["points"]) == {"GAMMA", "M", "K", "A", "L", "H"} and st["segments"] == 9 and not st["failed"]
    assert drv.find_element("css selector", ".po-kpath .kp-bz canvas")
    # a chip under the pointer is marked in the zone view
    ActionChains(drv).move_to_element(see(drv, seg(drv, "M → K"))).perform()
    H.wait(drv, "const h = document.querySelector('.po-kpath .kp-bz').__bz.state().hl; return h && h.indices[0] === 1 && h.kind === 'hover'", 5)
    # a segment under the pointer in the zone view is marked in the strip and on the plot
    ActionChains(drv).move_to_element(see(drv, drv.find_element("css selector", ".po-kpath .kp-help"))).perform()
    drv.execute_script("document.querySelector('.po-kpath .kp-bz').__bz.hoverSegment(4)")
    H.wait(drv, "return document.querySelector('.po-kpath .kp-seg[data-i=\"4\"]').classList.contains('kp-hl')", 5)
    H.wait(drv, "return document.querySelector('#plot-host .pv-plot').layout.shapes.some((s) => s.name === 'kp-hover')", 5)
    drv.execute_script("document.querySelector('.po-kpath .kp-bz').__bz.hoverSegment(null)")
    # a click on a segment there selects its chip; the edits redraw the zone view's path
    drv.execute_script("document.querySelector('.po-kpath .kp-bz').__bz.selectSegment(2)")
    H.wait(drv, "return document.activeElement && document.activeElement.matches('.po-kpath .kp-seg[data-i=\"2\"]')", 5)
    drv.switch_to.active_element.send_keys("h")
    wait_kpath(drv, "k && k[2].hidden")
    assert bz_state(drv)["segments"] == 9
    U.no_errors(drv)


BZ_LIVE = "const d = arguments[0]; d(window.__maceBz ? window.__maceBz.listening() : 0)"


def test_closed_zone_views_stop_following_the_theme(fresh):
    """A Brillouin zone view listens for theme changes while it lives: closing the band plot
    (and leaving Setup's k-path) lets go of it, so old views do not redraw on every theme switch."""
    drv = fresh[0]
    drv.execute_script("window.__mace.closeAll()")
    time.sleep(0.3)
    n0 = drv.execute_async_script(BZ_LIVE)
    for _ in range(3):
        open_plot(drv, "gr/g_band.BAND.DAT")
        H.wait(drv, "const e = document.querySelector('.po-kpath .kp-bz'); return e && e.__bz.state().loaded", 60)
        assert drv.execute_async_script(BZ_LIVE) == n0 + 1
        drv.execute_script("window.__mace.closeAll()")
        time.sleep(0.3)
        assert drv.execute_async_script(BZ_LIVE) == n0
    drv.execute_script("document.dispatchEvent(new Event('mace-theme'))")
    U.no_errors(drv)


def test_publication_export_follows_the_edit_like_the_cli(fresh, tmp_path):
    """The band figure mace plotting makes for the GUI's edit is byte-for-byte the CLI's
    `mace plotting --band ... --band-path FILE` with the same edit."""
    drv, base, root = fresh[0], fresh[1], fresh[2]
    open_plot(drv, "gr/g_band.BAND.DAT")
    hover_click(drv, seg(drv, "Γ → M"), ".kp-rev")
    wait_kpath(drv, "k && k[0].from === 'M'")
    drv.execute_script("arguments[0].focus()", seg(drv, "K → Γ"))
    drv.switch_to.active_element.send_keys("h")
    wait_kpath(drv, "k && k[0].from === 'M' && k[2].hidden")
    edit = drv.execute_script("return window.__mace.plotView.state().opts.kpath")
    r = drv.execute_async_script("""const done = arguments[arguments.length - 1];
      (async () => {
        const q = new URLSearchParams({path: 'gr/g_band.BAND.DAT', format: 'png', band_path: JSON.stringify(arguments[0])});
        const res = await fetch('api/plot?' + q, {credentials: 'same-origin'});
        const j = await res.json();
        const png = j.images.find((n) => n.endsWith('.png'));
        const img = await fetch('api/plot-image?' + new URLSearchParams({key: j.key, name: png}), {credentials: 'same-origin'});
        const b = new Uint8Array(await img.arrayBuffer());
        let s = ''; for (const x of b) s += String.fromCharCode(x);
        return {name: png, b64: btoa(s), command: j.command, log: j.log};
      })().then(done, (e) => done({error: String(e)}));""", edit)
    assert "error" not in r, r
    assert r["command"].endswith("--band-path band_path.json"), r["command"]
    d = tmp_path / "cli"
    d.mkdir()
    for f in ("g_band.BAND.DAT", "g_band.d3"):
        shutil.copy(root / "gr" / f, d / f)
    (d / "edit.json").write_text(json.dumps({"band_path": edit}))
    p = R.subprocess.run([sys.executable, str(H.MACE_CLI), "--no-banner", "plotting", "--band", "g_band.BAND.DAT",
                          "--band-path", "edit.json"], cwd=str(d), capture_output=True, text=True, timeout=300, env=R.env())
    assert p.returncode == 0, p.stdout + p.stderr
    import base64
    assert (d / r["name"]).read_bytes() == base64.b64decode(r["b64"]), r["log"]


def test_setup_zone_view_adds_a_point(fresh):
    """Calculation Setup: the zone's points are buttons too (the keyboard's way to a click on the
    point in the view); a point of the computed path is added by label, another with its k."""
    drv = fresh[0]
    open_band_setup(drv)
    H.wait(drv, "const e = document.querySelector('#inspector .su-kpath .kp-bz'); return e && e.__bz.state().loaded", 60)
    st = bz_state(drv, "#inspector .su-kpath .kp-bz")
    assert {"GAMMA", "X", "L", "W", "K", "U"} <= set(st["points"])
    b = drv.find_element("css selector", "#inspector .su-kpath [data-bz-pick=U]")
    assert "Add a segment to U" in b.get_attribute("data-tip")
    b.click()
    wait_setup(drv, "s.patches.band_path && s.patches.band_path.slice(-1)[0].to === 'U'")
    last = setup_state(drv)["patches"]["band_path"][-1]
    assert last["from"] == "GAMMA" and last["k_to"] == [0.625, 0.25, 0.625]
    drv.execute_script("document.querySelector('#inspector .su-kpath .kp-bz').__bz.pickPoint('X')")
    wait_setup(drv, "s.patches.band_path.slice(-1)[0].to === 'X'")
    assert "k_to" not in setup_state(drv)["patches"]["band_path"][-1]
    U.no_errors(drv)
