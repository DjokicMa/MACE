"""A single-point output in the structure view (1.1.6 preview reports): an SP
opens as one geometry, never as an optimization (also when CRYSTAL printed its
geometry twice), its Run Analysis has only the SCF tab and the optimization
commands say why they are off; Build › Supercell works on its long workflow
name; a 2×2×2 display range refits the camera so all three directions show,
and a SLAB says it repeats a and b only. Uses the shipped examples; runs in CI."""
import json
import shutil
import time

import pytest

import guiharness as H

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("numpy")

OUT = H.EXAMPLES / "outputs"
# a workflow SP's name: past the 61 characters a builder's name may have, with "opt" in it
LONG = "sp/Si_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis.triplezeta_opt_B3LYP-D3-D3_optimized_sp_B3LYP-D3-D3_optimized.out"
TWICE = "sp/Si_twice_sp.out"
SLAB = "outputs/graphene_slab_sp.out"


def _twice(src, dst):
    """Si_sp.out with its geometry block printed twice (as after a geometry edit of the input)."""
    lines = src.read_text().splitlines(True)
    start = next(i for i, ln in enumerate(lines) if "GEOMETRY FOR WAVE FUNCTION" in ln)
    end = next(i for i, ln in enumerate(lines) if "T = ATOM BELONGING" in ln)
    dst.write_text("".join(lines[:start] + lines[start:end + 1] + ["\n"] + lines[start:]))


@pytest.fixture(scope="module", params=["chromium", "firefox"])
def app(request, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("spstruct")
    root = H.examples_copy(tmp)
    (root / "sp").mkdir()
    shutil.copyfile(OUT / "Si_sp.out", root / LONG)
    _twice(OUT / "Si_sp.out", root / TWICE)
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_TEST_HOOKS="1", MACE_GUI_MAX_WORKERS="2")
    try:
        drv = H.browser(request.param, width=1280, height=800)
    except BaseException:
        proc.terminate()
        proc.wait(timeout=15)
        raise
    try:
        H.load(drv, base)
        drv.execute_script("window.__mace.settings.set('reduceMotion', 'on'); window.__mace.settings.set('showBoundary', false)")
        yield drv, base
    finally:
        drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


@pytest.fixture
def drv(app, request):
    failed = request.session.testsfailed
    yield app[0]
    if request.session.testsfailed > failed:
        H.save_failure(app[0], request.node.name)
    app[0].execute_script("window.__mace.closeAll(); window.__mace.settings.set('displayLo', [0, 0, 0]); "
                          "window.__mace.settings.set('displayHi', [1, 1, 1])")
    time.sleep(0.3)


def open_doc(drv, path):
    kind = drv.execute_async_script("const d = arguments[1]; window.__mace.open(arguments[0]).then((x) => d(x && x.kind), "
                                    "(e) => d('error: ' + e.message))", path)
    H.wait(drv, "const v = window.MACE.state.viewers['viewer-main']; return !!(v && v.getModel())", 30)
    time.sleep(0.8)
    return kind


def reason(drv, cmd):
    return drv.execute_script("return window.__mace.menuReason(arguments[0])", cmd)


def run(drv, cmd):
    drv.execute_script("window.__mace.run(arguments[0])", cmd)
    time.sleep(0.8)


def test_an_sp_is_one_geometry_not_an_optimization(drv):
    for path in (LONG, TWICE):
        assert open_doc(drv, path) == "structure", path
        assert drv.execute_script("return JSON.stringify(window.__mace.docs().map((d) => d.kind))") == '["structure"]'
        badges = drv.execute_script("return [...document.querySelectorAll('#tab-structure .badge, .badge')].map((b) => b.textContent)")
        assert "single-point geometry" in badges and "initial geometry" not in badges, badges
        for cmd in ("analysis.trajectory", "analysis.convergence", "analysis.cellRun", "analysis.trackMeasurement"):
            assert reason(drv, cmd).startswith("A single-point run has one geometry"), (cmd, reason(drv, cmd))
        assert reason(drv, "analysis.runPlot") == "" and reason(drv, "analysis.scf") == ""
        drv.execute_script("window.__mace.closeAll()")
        time.sleep(0.3)


def test_the_twice_printed_sp_summary_says_no_optimization(app):
    drv = app[0]
    tok = drv.execute_script("return window.MACE.q({ path: arguments[0] })", TWICE)
    r = drv.execute_async_script("const d = arguments[1]; window.MACE.api('out/summary?' + arguments[0]).then((x) => d(JSON.stringify(x)), (e) => d(String(e)))", tok)
    s = json.loads(r)
    assert s["n_steps"] == 2 and s["is_opt"] is False


def test_run_analysis_of_an_sp_has_only_its_scf(drv):
    open_doc(drv, LONG)
    run(drv, "analysis.runPlot")
    H.wait(drv, "const b = document.querySelector('.rp-tabs [data-tab=\"energy\"]'); return !!b && b.hidden", 20)
    tabs = drv.execute_script("return [...document.querySelectorAll('.rp-tabs [data-tab]')].filter((b) => !b.hidden).map((b) => b.dataset.tab)")
    assert tabs == ["scf"]
    assert drv.execute_script("return document.querySelector('.rp-tabs [data-tab=\"scf\"]').getAttribute('aria-selected')") == "true"


def test_supercell_of_an_sp_with_a_long_workflow_name(drv):
    open_doc(drv, LONG)
    assert reason(drv, "build.supercell") == ""
    run(drv, "build.supercell")
    H.wait(drv, "const i = [...document.querySelectorAll('input[aria-invalid]')].find((x) => x.offsetParent); return !!i && i.value.endsWith('_2x2x2')", 20)
    name, invalid = drv.execute_script("const i = [...document.querySelectorAll('input[aria-invalid]')].find((x) => x.offsetParent); return [i.value, i.getAttribute('aria-invalid')]")
    assert invalid == "false" and len(name) <= 61 and name.startswith("Si_sym_CRYSTAL"), name
    go = drv.execute_script("return [...document.querySelectorAll('button')].find((b) => b.offsetParent && b.textContent.trim() === 'Create')")
    assert go is not None and go.is_enabled()
    go.click()
    H.wait(drv, "return window.__mace.docs().length === 2", 30)
    H.wait(drv, "return window.MACE.state.structure && window.MACE.state.structure.numbers.length === 16", 30)


IN_VIEW = """const v = window.MACE.state.viewers['viewer-main'], m = v.getModel(), r = v.getCanvas().getBoundingClientRect();
  const at = m.selectedAtoms({}); let out = 0;
  for (const a of at) { const p = v.modelToScreen({ x: a.x, y: a.y, z: a.z }), x = p.x - r.left, y = p.y - r.top;
    if (x < 0 || y < 0 || x > r.width || y > r.height) out++; }
  return [at.length, out];"""


def test_a_3d_range_refits_so_c_repeats_too(drv):
    open_doc(drv, "outputs/Si_sp.out")
    n0 = drv.execute_script("return window.__mace.viewState().displayedAtoms")
    for n in (2, 3):
        run(drv, f"view.range{n}")
        H.wait(drv, f"return window.__mace.viewState().displayedAtoms === {n0 * n ** 3}", 20)
        time.sleep(0.5)
        drawn, outside = drv.execute_script(IN_VIEW)
        assert drawn >= n0 * n ** 3 and outside == 0, (n, drawn, outside)
        assert drv.execute_script("return document.getElementById('tb-range-label').textContent") == f"{n}×{n}×{n}"


def test_a_slab_range_repeats_a_and_b_and_says_so(drv):
    open_doc(drv, SLAB)
    n0 = drv.execute_script("return window.__mace.viewState().displayedAtoms")
    run(drv, "view.range3")
    H.wait(drv, f"return window.__mace.viewState().displayedAtoms === {n0 * 9}", 20)
    assert drv.execute_script("return document.getElementById('tb-range-label').textContent") == "3×3×1"
    assert "not periodic along c" in drv.execute_script("return document.getElementById('tb-range').dataset.tip")
    H.wait(drv, "return [...document.querySelectorAll('.toast, [role=status], [role=alert]')].some((t) => /SLAB is not periodic along c.*3×3×1/.test(t.textContent))", 10)
    assert not H.errors(drv), H.errors(drv)
