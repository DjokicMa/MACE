"""The run view's ENERGY ALONG THE RUN card across FINALRUN restarts, in a real
browser (MACE_GUI_BROWSER=firefox, the default, or chromium).

1_dia_opt_rev1.out (the corpus) has two FINALRUN segments, points 1-3 and 4.
Segments: Continuous (the default) is one curve relative to the run's point;
Separate re-references each segment. Copy TSV is the same as `mace analyze
--convergence FILE --energy` (mace.utils.out_convergence.energy_along_run),
Open as Plot opens Run Analysis on the card's data, and the choice is kept
per document across reopening."""
import os
import time

import pytest

import guiharness as H
import p4util as U
import runcorpus as RC

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")

from mace.utils import out_convergence as oc  # noqa: E402

REL = "runs/1_dia_opt_rev1.out"


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    RC.need(RC.DIA_OPT)
    tmp = tmp_path_factory.mktemp("esegs")
    root = H.examples_copy(tmp)
    RC.copy_into(root / "runs", RC.DIA_OPT, RC.DIA_OPT.with_suffix(".d12"))
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_TEST_HOOKS="1")
    drv = None
    try:
        drv = H.browser(os.environ.get("MACE_GUI_BROWSER", "firefox"), width=1280, height=800)
        H.load(drv, base)
        drv.execute_script("window.__mace.settings.set('reduceMotion', 'on')")
        yield drv, root
    finally:
        if drv is not None:
            drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


def open_run(drv):
    drv.execute_script("window.__mace.closeAll()")
    from selenium.common.exceptions import StaleElementReferenceException
    for _ in range(5):
        try:
            row = drv.find_element("css selector", f'#file-list .file[data-path="{REL}"]')
            drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", row)
            row.click()
            break
        except StaleElementReferenceException:
            time.sleep(0.5)
    # the convergence (points, segments) has arrived: the dotted FINALRUN line is drawn
    H.wait(drv, "return document.querySelectorAll('#o-energy .seg-line').length === 1 && "
                "document.querySelectorAll('#o-energy circle.pt').length >= 2", 60)


def card(drv):
    return drv.execute_script("""
        const lab = document.querySelector('#o-e-seg').closest('label');
        return { label: document.querySelector('#o-energy-label').textContent,
          seg: document.querySelector('#o-e-seg').value, segShown: !lab.hidden && lab.offsetParent !== null,
          lines: document.querySelectorAll('#o-energy polyline.line').length,
          pts: document.querySelectorAll('#o-energy .pt').length, ring: document.querySelectorAll('#o-energy .ring').length,
          ticks: [...document.querySelectorAll('#o-energy text.tick')].map((t) => t.textContent),
          tip: document.querySelector('#o-energy .seg-line title').textContent, tsv: window.MACE.energyTSV() }""")


def pick(drv, sel, value):
    from selenium.webdriver.support.ui import Select
    Select(drv.find_element("css selector", sel)).select_by_value(value)


def tsv_rows(text):
    head, *rows = text.strip().split("\n")
    return head, [r.split("\t") for r in rows]


def same_tsv(gui, cli):
    """The GUI's TSV and the CLI's: the same header, points, segments, frames and values."""
    gh, gr = tsv_rows(gui)
    ch, cr = tsv_rows(cli)
    assert gh == ch
    assert [r[:3] for r in gr] == [r[:3] for r in cr]
    for a, b in zip(gr, cr):
        assert abs(float(a[3]) - float(b[3])) < 1e-12
        assert abs(float(a[4]) - float(b[4])) < 1e-9, (a, b)


def test_continuous_default_separate_toggle_persistence_and_tsv(app):
    drv, root = app
    res = oc.parse(root / REL)
    assert len(res["segments"]) == 2
    open_run(drv)
    pick(drv, "#o-e-rel", "final")
    from selenium.webdriver.support.ui import Select
    Select(drv.find_element("css selector", "#o-e-unit")).select_by_visible_text("meV")
    H.wait(drv, "return /relative to the final point/.test(document.querySelector('#o-energy-label').textContent)", 10)
    c = card(drv)
    # Continuous by default: one connected curve, relative to the run's final point
    assert c["seg"] == "continuous" and c["segShown"]
    assert c["label"].endswith("meV, relative to the final point of the run"), c["label"]
    assert c["lines"] == 1
    assert c["tip"] == "FINALRUN restart before POINT 4"
    same_tsv(c["tsv"], oc.energy_tsv(oc.energy_along_run(res, "final", "continuous"), "meV"))
    assert c["tsv"].split("\n")[0] == "point\tsegment\tframe\tE_Ha\tdE_meV"

    # Separate: the per-segment view, the TSV follows
    pick(drv, "#o-e-seg", "separate")
    H.wait(drv, "return document.querySelectorAll('#o-energy polyline.line').length === 2", 10)
    c = card(drv)
    assert c["label"].endswith("relative to the final point of each FINALRUN segment"), c["label"]
    assert c["tip"].endswith("(re-referenced)")
    same_tsv(c["tsv"], oc.energy_tsv(oc.energy_along_run(res, "final", "separate"), "meV"))

    # remembered for this document: closed and opened again, still Separate
    open_run(drv)
    H.wait(drv, "return document.querySelector('#o-e-seg').value === 'separate' && "
                "document.querySelectorAll('#o-energy polyline.line').length === 2", 10)
    pick(drv, "#o-e-seg", "continuous")
    H.wait(drv, "return document.querySelectorAll('#o-energy polyline.line').length === 1", 10)
    open_run(drv)
    assert card(drv)["seg"] == "continuous"
    U.no_errors(drv)


def test_log_view_handles_the_zero_at_the_final_point(app):
    drv, root = app
    res = oc.parse(root / REL)
    open_run(drv)
    pick(drv, "#o-e-rel", "final")
    pick(drv, "#o-e-seg", "continuous")
    if drv.execute_script("return document.querySelector('#o-e-log').getAttribute('aria-pressed')") != "true":
        drv.find_element("css selector", "#o-e-log").click()
    H.wait(drv, "return /log/.test(document.querySelector('#o-energy-label').textContent)", 10)
    c = card(drv)
    rows = oc.energy_along_run(res, "final", "continuous")
    nonzero = [r for r in rows if not r["is_ref"]]
    # the reference (point 4 and the final geometry) has no log value: left out, nothing NaN or Infinity
    assert c["pts"] == len(nonzero) and c["ring"] == 0
    assert not any(("NaN" in t or "Infinity" in t) for t in c["ticks"]), c["ticks"]
    assert "NaN" not in drv.execute_script("return document.querySelector('#o-energy svg').outerHTML")
    # Separate in the log view: segment 2 is a single point (its own reference): only segment 1 is left
    pick(drv, "#o-e-seg", "separate")
    H.wait(drv, "return document.querySelectorAll('#o-energy polyline.line').length === 1", 10)
    sep = [r for r in oc.energy_along_run(res, "final", "separate") if not r["is_ref"]]
    assert card(drv)["pts"] == len(sep)
    drv.find_element("css selector", "#o-e-log").click()
    pick(drv, "#o-e-seg", "continuous")
    U.no_errors(drv)


def test_open_as_plot_uses_the_cards_choice(app):
    drv, _ = app
    for mode, groups in (("continuous", 1), ("separate", 2)):
        open_run(drv)
        pick(drv, "#o-e-rel", "final")
        pick(drv, "#o-e-seg", mode)
        H.wait(drv, f"return document.querySelectorAll('#o-energy polyline.line').length === {groups}", 10)
        H.wait(drv, "const b = document.querySelector('#o-e-plot'); return b && b.offsetParent !== null", 30)
        b = drv.find_element("css selector", "#o-e-plot")
        drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", b)
        b.click()
        H.wait(drv, "const s = window.__mace.runView && window.__mace.runView.state(); "
                    "return s && s.drawn && s.drawn.endsWith('|energy') && s.opts.energy.segments === arguments[0]", 60, mode)
        st = drv.execute_script("return window.__mace.runView.state().opts.energy")
        assert st["relTo"] == "final" and st["unit"] == "meV" and st["per"] == "cell" and not st["ylog"] and "single" not in st
        tr = drv.execute_script("return window.__mace.runView.figure().traces")
        lines = [t for t in tr if str(t["meta"]).startswith("e0_")]
        assert len(lines) == groups
        ann = drv.execute_script("return window.__mace.runView.figure().traces.filter((t) => t.meta === 'key').map((t) => t.name.replace('<br>', ' '))")      # the restart's legend entry
        assert ("FINALRUN restart (re-referenced)" in ann) == (mode == "separate")
        assert any(a.startswith("FINALRUN restart") for a in ann)
        # Plot Options: the Segments choice is there and switches the plot
        drv.find_element("css selector", "#runplot-host .pk-more").click()
        sel = drv.find_element("css selector", ".pk-pop [data-k=segments]")
        assert sel.get_attribute("value") == mode
        other = "separate" if mode == "continuous" else "continuous"
        pick(drv, ".pk-pop [data-k=segments]", other)
        H.wait(drv, "return window.__mace.runView.state().opts.energy.segments === arguments[0] && "
                    "window.__mace.runView.figure().traces.filter((t) => String(t.meta).startsWith('e0_')).length === arguments[1]",
               10, other, 3 - groups)
        from selenium.webdriver.common.keys import Keys
        U.keys(drv, Keys.ESCAPE)
    U.no_errors(drv)
