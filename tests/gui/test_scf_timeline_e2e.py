"""The SCF convergence timeline in a real browser (MACE_GUI_BROWSER=firefox, the
default, or chromium): the run view's SCF card on a spin-locked OPT (log |ΔE|
against TOLDEE, the mixing and DIIS cycles marked, Copy TSV = `mace analyze
--convergence FILE --scf`, the options kept, Open as Plot), Run Analysis › SCF
on a spin-polarized SP and on a FREQ with two SCFs (the step picker), and a
growing SP output followed while it is written."""
import os
import time

import pytest

import guiharness as H
import runcorpus as RC

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("plotly")

from mace.utils import out_convergence as oc  # noqa: E402

SP = RC.CORPUS / "SP" / "1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.out"
FREQ = RC.CORPUS / "FREQ" / ("Ag2Cl1_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis.triplezeta_opt_B3LYP-D3-D3_optimized"
                             "_freq_B3LYP-D3-D3_optimized.out")


def _opt_with_scfout(d):
    """1_dia_opt_rev1.out with a <job>.SCFOUT.LOG for points 2-4 (as the job script copies it back),
    in CRYSTAL's SCFOUT.LOG layout (tests/test_scf_timeline.scfout_block)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("scf_unit", str(H.REPO_ROOT / "tests" / "test_scf_timeline.py"))
    unit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(unit)
    d.mkdir(exist_ok=True)
    RC.copy_into(d, RC.DIA_OPT)
    p = d / RC.DIA_OPT.name
    pts = {q["point"]: q for q in oc.parse(p)["points"]}
    text = ""
    for n in (2, 3, 4):
        e = pts[n]["energy_scf_au"]
        text += unit.scfout_block([e + 10 ** -(k + 2) for k in range(pts[n]["scf_cycles"])] + [e], e)
    (d / (p.stem + ".SCFOUT.LOG")).write_text(text)


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    for p in (RC.DIA_OPT, SP, FREQ):
        RC.need(p)
    tmp = tmp_path_factory.mktemp("scf")
    root = H.examples_copy(tmp)
    RC.copy_into(root / "runs", RC.DIA_OPT, SP, FREQ)
    _opt_with_scfout(root / "runs2")
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_TEST_HOOKS="1")
    drv = None
    try:
        drv = H.browser(os.environ.get("MACE_GUI_BROWSER", "firefox"), width=1400, height=1000)
        drv.set_script_timeout(120)
        H.load(drv, base)
        drv.execute_script("window.__mace.settings.set('reduceMotion', 'on')")
        yield drv, root
    finally:
        if drv is not None:
            drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


def open_doc(drv, rel):
    drv.execute_script("window.__mace.closeAll()")
    drv.execute_async_script("const done = arguments[1]; window.__mace.open(arguments[0]).then(() => done())", rel)


def card(drv):
    return drv.execute_script("""
        const el = document.querySelector('#o-scf');
        return { label: document.querySelector('#o-scf-label').textContent, pts: el.querySelectorAll('circle.pt').length,
          up: el.querySelectorAll('circle.pt.up').length, thr: el.querySelectorAll('.thr-line').length,
          ev: [...el.querySelectorAll('.ev-line')].map((l) => l.dataset.kind), log: document.querySelector('#o-s-log').getAttribute('aria-pressed'),
          unit: document.querySelector('#o-s-unit').value, stepShown: !document.querySelector('#o-s-step').closest('label').hidden,
          tsv: window.MACE.scfTSV(), html: el.innerHTML }""")


def pick(drv, sel, text):
    from selenium.webdriver.support.ui import Select
    Select(drv.find_element("css selector", sel)).select_by_visible_text(text)


def same_tsv(gui, cli):
    g = [r.split("\t") for r in gui.rstrip("\n").split("\n")]
    c = [r.split("\t") for r in cli.rstrip("\n").split("\n")]
    assert g[0] == c[0] and len(g) == len(c)
    for a, b in zip(g[1:], c[1:]):
        assert a[0] == b[0] and a[4] == b[4], (a, b)
        for k in (1, 2, 3):
            assert (a[k] == "" and b[k] == "") or abs(float(a[k]) - float(b[k])) <= 1e-9 * max(1.0, abs(float(b[k]))), (a, b)


def rp_wait(drv, tab="scf"):
    H.wait(drv, "const s = window.__mace.runView && window.__mace.runView.state(); return s && s.drawn && s.drawn.endsWith('|' + arguments[0])", 60, tab)


def test_run_view_scf_card(app):
    drv, root = app
    drv.execute_script("Object.keys(localStorage).filter((k) => k.startsWith('mace.gui.scfcard.')).forEach((k) => localStorage.removeItem(k))")
    open_doc(drv, "runs/1_dia_opt_rev1.out")
    H.wait(drv, "return document.querySelectorAll('#o-scf circle.pt').length >= 3", 60)
    c = card(drv)
    tl = oc.scf_timeline(oc.parse(root / "runs" / "1_dia_opt_rev1.out"))
    blk = oc.scf_block(tl)
    # log |ΔE| by default, every cycle of POINT 1, the TOLDEE line, mixing and DIIS marked
    assert c["log"] == "true" and c["unit"] == "Ha" and not c["stepShown"]
    assert c["label"].startswith("SCF convergence · POINT 1, 7 cycles, log |ΔE| in Ha · TOLDEE 1e-7 Ha"), c["label"]
    assert c["pts"] == len(blk["cycles"]) and c["thr"] == 1
    assert c["up"] == sum(1 for x in blk["cycles"] if x["cyc"] > 0 and x["detot"] > 0)
    assert c["ev"] == [e["kind"] for e in blk["events"]] == ["fmix", "diis"]
    assert "NaN" not in c["html"] and "Infinity" not in c["html"]
    same_tsv(c["tsv"], oc.scf_tsv(blk, "Ha"))
    # meV and the linear view, kept for this document
    pick(drv, "#o-s-unit", "meV")
    drv.find_element("css selector", "#o-s-log").click()
    H.wait(drv, "return /E − E\\(last cycle\\) in meV/.test(document.querySelector('#o-scf-label').textContent)", 10)
    c = card(drv)
    assert c["thr"] == 0 and c["pts"] == len(blk["cycles"])
    same_tsv(c["tsv"], oc.scf_tsv(blk, "meV"))
    open_doc(drv, "runs/1_dia_opt_rev1.out")
    H.wait(drv, "return document.querySelectorAll('#o-scf circle.pt').length >= 3", 60)
    c = card(drv)
    assert c["unit"] == "meV" and c["log"] == "false"
    # Open as Plot: Run Analysis › SCF on the card's choices
    b = drv.find_element("css selector", "#o-s-plot")
    drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", b)
    b.click()
    rp_wait(drv)
    st = drv.execute_script("return window.__mace.runView.state().opts.scf")
    assert st["unit"] == "meV" and st["ylog"] is False and st["block"] == "latest"
    tr = {t["meta"]: t for t in drv.execute_script("return window.__mace.runView.figure().traces")}
    assert tr["cycles"]["y"] == [6, 5, 5, 3] and len(tr["detail"]["x"]) == len(blk["cycles"])
    lay = drv.execute_script("return window.__mace.runView.figure().layout")
    assert sorted(s["name"] for s in lay["shapes"] if str(s.get("name", "")).startswith("scf-event")) == ["scf-event-diis", "scf-event-fmix"]
    assert drv.execute_script("return window.__mace.runView.table().columns") == ["cycle", "E_Ha", "dE_meV", "spin", "events"]
    assert not H.errors(drv)


def test_run_analysis_scf_sp_and_freq_steps(app):
    drv, root = app
    open_doc(drv, "runs/" + SP.name)
    drv.execute_script("window.__maceOpenRunPlot(arguments[0], 'scf')", "runs/" + SP.name)
    rp_wait(drv)
    tr = {t["meta"]: t for t in drv.execute_script("return window.__mace.runView.figure().traces")}
    assert len(tr["detail"]["x"]) == 9 and "cycles" not in tr                 # a single point: no cycles per point
    ann = drv.execute_script("return window.__mace.runView.figure().layout.annotations.map((a) => a.text)")
    # the TOLDEE line's and the notable cycles' labels are legend entries (1.1.5: no boxes over the data)
    keys = drv.execute_script("return window.__mace.runView.figure().traces.filter((t) => t.meta === 'key').map((t) => t.name.replace('<br>', ' '))")
    assert any(k.startswith("TOLDEE 1e-9 Ha") for k in keys), keys
    assert any(k.startswith("Fock/KS mixing 30% (cycle ") for k in keys) and any(k.startswith("DIIS on (cycle ") for k in keys), keys
    assert not any(a in ("Fock/KS mixing 30%", "DIIS on") for a in ann), ann
    rows = drv.execute_script("return window.__mace.runView.table().rows")
    assert [r[0] for r in rows] == list(range(9)) and rows[1][3] == 0       # the spin density column
    # a FREQ with two SCFs: Latest (the second) by default, then the first from the picker
    tl = oc.scf_timeline(oc.parse(root / "runs" / FREQ.name))
    drv.execute_script("window.__maceOpenRunPlot(arguments[0], 'scf')", "runs/" + FREQ.name)
    rp_wait(drv)
    H.wait(drv, "return !!document.querySelector('#runplot-host [data-scfstep]')", 20)
    n = drv.execute_script("return window.__mace.runView.figure().traces.find((t) => t.meta === 'detail').x.length")
    assert n == len(tl["blocks"][1]["cycles"])
    from selenium.webdriver.support.ui import Select
    Select(drv.find_element("css selector", "#runplot-host [data-scfstep]")).select_by_value("1")
    H.wait(drv, "return window.__mace.runView.figure().traces.find((t) => t.meta === 'detail').x.length === arguments[0]", 20,
           len(tl["blocks"][0]["cycles"]))
    assert not H.errors(drv)


def test_growing_sp_is_followed(app):
    """An SP still being written (no end marker): Run Analysis › SCF reads it again while it
    grows (window.__maceLiveScale shortens the 15 s period) and the running SCF gains cycles."""
    drv, root = app
    full = SP.read_bytes()
    cut = full.find(b" CYC   4 ETOT")
    live = root / "runs" / "live_sp.out"
    live.write_bytes(full[:cut])
    drv.execute_script("window.__maceLiveScale = 0.02")
    try:
        drv.execute_script("window.__mace.closeAll(); window.__maceOpenRunPlot('runs/live_sp.out', 'scf')")
        rp_wait(drv)
        H.wait(drv, "const t = window.__mace.runView.figure().traces.find((x) => x.meta === 'detail'); return t && t.x.length === 4", 30)
        assert "running" in drv.execute_script("return document.querySelector('#runplot-host .pv-note').textContent")
        with open(live, "ab") as f:
            f.write(full[cut:full.find(b" CYC   7 ETOT")])
        H.wait(drv, "const t = window.__mace.runView.figure().traces.find((x) => x.meta === 'detail'); return t && t.x.length === 7", 30)
        with open(live, "ab") as f:
            f.write(full[full.find(b" CYC   7 ETOT"):])
        H.wait(drv, "const t = window.__mace.runView.figure().traces.find((x) => x.meta === 'detail'); return t && t.x.length === 9", 30)
    finally:
        drv.execute_script("window.__maceLiveScale = 1")
    assert not H.errors(drv)


def _section(drv):
    return drv.execute_script("""
        const r = document.querySelector('#inspector .insp-section[data-key="scf"] .scf-card');
        if (!r) return null;
        const el = r.querySelector('[data-s="chart"]');
        return { shown: !r.classList.contains('hidden') && r.offsetParent !== null, pts: el.querySelectorAll('circle.pt').length,
          thr: el.querySelectorAll('.thr-line').length, ev: [...el.querySelectorAll('.ev-line')].map((l) => l.dataset.kind),
          label: r.querySelector('[data-s="label"]').textContent, step: !r.querySelector('[data-s="step"]').closest('label').hidden }""")


def test_sp_opens_with_its_scf(app):
    """An SP output opens as a structure (no run view): the inspector shows its SCF first, right away."""
    drv, root = app
    drv.execute_script("Object.keys(localStorage).filter((k) => k.startsWith('mace.gui.scfcard.')).forEach((k) => localStorage.removeItem(k))")
    open_doc(drv, "runs/" + SP.name)
    H.wait(drv, "const r = document.querySelector('#inspector .insp-section[data-key=\"scf\"] .scf-card circle.pt'); return !!r", 60)
    s = _section(drv)
    blk = oc.scf_block(oc.scf_timeline(oc.parse(root / "runs" / SP.name)))
    # the first inspector section, in view without scrolling
    keys = drv.execute_script("return [...document.querySelectorAll('#inspector .insp-section')].map((x) => x.dataset.key)")
    assert keys[:2] == ["scf", "run"], keys
    assert drv.execute_script("""const c = document.querySelector('#inspector .scf-card [data-s="chart"] svg').getBoundingClientRect();
        const h = document.querySelector('#inspector').getBoundingClientRect(); return c.top >= h.top && c.bottom <= h.bottom""")
    assert s["shown"] and s["pts"] == len(blk["cycles"]) and s["thr"] == 1 and not s["step"]
    assert s["ev"] == ["fmix", "diis"] and s["label"].startswith("SCF convergence · SCF, 9 cycles, log |ΔE| in Ha · TOLDEE 1e-9 Ha")
    # the run view's card stays with optimizations: an SP has none
    assert drv.execute_script("return window.__mace.activeDoc().kind") == "structure"
    # its own controls: the linear view
    drv.find_element("css selector", '#inspector .scf-card [data-s="log"]').click()
    H.wait(drv, "return /E − E\\(last cycle\\)/.test(document.querySelector('#inspector .scf-card [data-s=\"label\"]').textContent)", 10)
    drv.find_element("css selector", '#inspector .scf-card [data-s="log"]').click()
    assert not H.errors(drv)


def test_opt_scfout_log_gives_every_point_a_step(app):
    """An OPT with its <job>.SCFOUT.LOG next to it (what the job script copies back): the
    run view's SCF card lists every point, Latest is the last one."""
    drv, root = app
    p = root / "runs2" / RC.DIA_OPT.name
    drv.execute_script("Object.keys(localStorage).filter((k) => k.startsWith('mace.gui.scfcard.')).forEach((k) => localStorage.removeItem(k))")
    open_doc(drv, f"runs2/{p.name}")
    H.wait(drv, "const a = window.__mace.activeDoc(); return a && a.path === arguments[0] && /POINT 4/.test(document.querySelector('#o-scf-label').textContent)", 60, f"runs2/{p.name}")
    c = card(drv)
    assert c["stepShown"] and c["label"].startswith("SCF convergence · POINT 4, 4 cycles"), c["label"]
    opts = drv.execute_script("return [...document.querySelectorAll('#o-s-step option')].map((o) => o.textContent)")
    assert opts == ["Latest", "POINT 1 · 7 cycles", "POINT 2 · 6 cycles", "POINT 3 · 6 cycles", "POINT 4 · 4 cycles"]
    pick(drv, "#o-s-step", "POINT 2 · 6 cycles")
    H.wait(drv, "return /POINT 2, 6 cycles/.test(document.querySelector('#o-scf-label').textContent)", 10)
    assert card(drv)["pts"] == 6
    assert not H.errors(drv)
