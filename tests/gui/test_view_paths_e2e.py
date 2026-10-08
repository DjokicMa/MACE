"""Phase 4 carry-overs (a), (b) and (d) in headless Firefox: the default
view (c up, a toward the lower left, b right) with a tight fit of what is
drawn; long folder paths shortened with "~" and "/…/" (the full path in the
tooltip); and unfinished commands that name what they arrive with instead
of "Coming soon"."""
import math
import re
import shutil
import time
from pathlib import Path

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")

TRICLINIC_CIF = """data_tri
_cell_length_a 5.10
_cell_length_b 6.30
_cell_length_c 7.40
_cell_angle_alpha 78.0
_cell_angle_beta 104.0
_cell_angle_gamma 96.0
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si1 Si 0.10 0.20 0.30
O1 O 0.35 0.15 0.42
O2 O 0.62 0.71 0.18
Si2 Si 0.80 0.55 0.66
"""


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("p4view")
    root = H.examples_copy(tmp)
    (root / "structures" / "Triclinic.cif").write_text(TRICLINIC_CIF)
    a = U.start_app(tmp, root=root)
    try:
        yield a
    finally:
        U.stop_app(a)


@pytest.fixture
def fresh(app):
    U.reset(app[0])
    return app


FIT_JS = """
const sc = window.__mace.viewport.scene(), cam = window.__mace.viewport.camera();
const disp = sc.display, R = sc.drawnRadii, s = window.__mace.doc().structure;
const v = window.MACE.state.viewers['viewer-main'], cv = v.getCanvas();
cam.invalidate();
const c = cam.calibrate(), P = c.P;
const pr = (x, y, z) => { const w = P[8] * x + P[9] * y + P[10] * z + 1;
  return [(P[0] * x + P[1] * y + P[2] * z + P[3]) / w, (P[4] * x + P[5] * y + P[6] * z + P[7]) / w]; };
let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
for (let k = 0; k < disp.n; k++) {
  const [x, y] = pr(disp.cart[3 * k], disp.cart[3 * k + 1], disp.cart[3 * k + 2]), r = R[k] * c.s;
  x0 = Math.min(x0, x - r); x1 = Math.max(x1, x + r); y0 = Math.min(y0, y - r); y1 = Math.max(y1, y + r);
}
const vs = window.__mace.viewState(), range = vs.range || 1, rr = Array.isArray(range) ? range : [range, range, range];
const pbc = s.pbc || [true, true, true], C = s.cell.map((r, k) => r.map((x) => x * (pbc[k] ? (rr[k] || 1) : 1)));
if (vs.cell) for (const i of [0, 1]) for (const j of [0, 1]) for (const k of [0, 1]) {
  const p = [0, 1, 2].map((m) => i * C[0][m] + j * C[1][m] + k * C[2][m]);
  const [x, y] = pr(...p);
  x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
}
return {x0, x1, y0, y1, W: cv.clientWidth, H: cv.clientHeight, triad: window.__mace.triad(), n: disp.n};
"""


def fit_box(drv):
    return drv.execute_script(FIT_JS)


def check_fit(b, label):
    W, Hh = b["W"], b["H"]
    bw, bh = b["x1"] - b["x0"], b["y1"] - b["y0"]
    fill = max(bw / W, bh / Hh)          # the limiting dimension
    assert 0.80 <= fill <= 0.94, (label, fill, b)
    cx, cy = (b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2
    assert abs(cx - W / 2) <= 0.02 * W and abs(cy - Hh / 2) <= 0.02 * Hh, (label, cx, cy, W, Hh)
    ax = {t["axis"]: t for t in b["triad"]}
    c = ax["c"]
    # c up: the screen arrow within 5 degrees of vertical (screen y grows downward)
    ang = math.degrees(math.atan2(abs(c["x"]), -c["y"]))
    assert ang <= 5.0, (label, "c tilt", ang, c)
    lens = {k: math.hypot(t["x"], t["y"]) for k, t in ax.items()}
    longest = max(lens.values())
    assert lens["a"] >= 0.35 * longest and lens["b"] >= 0.35 * longest, (label, lens)


CASES = [("structures/MgO.cif", {}), ("structures/MgO.cif", {"showBoundary": False}),
         ("structures/MgO.cif", {"range": 2}), ("structures/Si.cif", {}), ("structures/hBN.cif", {}),
         ("outputs/Ag2Br3_opt_unfinished.d12", {}), ("structures/Triclinic.cif", {})]


@pytest.mark.parametrize("size", [(1280, 800), (1920, 1080)])
def test_default_fit_fills_viewport(fresh, size):
    drv = fresh[0]
    H.set_viewport(drv, *size)
    time.sleep(0.3)
    for path, opts in CASES:
        drv.execute_script("window.__mace.closeAll()")
        if "showBoundary" in opts:
            drv.execute_script("window.__mace.settings.set('showBoundary', arguments[0])", opts["showBoundary"])
        U.open_doc(drv, path)
        if "range" in opts:
            drv.execute_script("window.__mace.run('view.range2')")
            time.sleep(0.4)
            drv.execute_script("window.__mace.run('view.reset')")
        time.sleep(0.5)
        b = fit_box(drv)
        label = f"{path} {opts} {size}"
        check_fit(b, label)
        U.shot(drv, f"fit_{Path(path).stem}_{'_'.join(f'{k}{v}' for k, v in opts.items())}_{size[0]}")
        drv.execute_script("window.__mace.settings.set('showBoundary', true)")
        if "range" in opts:
            drv.execute_script("window.__mace.run('view.range1')")
    H.set_viewport(drv, 1280, 800)
    U.no_errors(drv)


def test_reset_view_equals_default(fresh):
    drv = fresh[0]
    U.open_doc(drv, "structures/MgO.cif")
    first = drv.execute_script("return window.MACE.state.viewers['viewer-main'].getView()")
    drv.execute_script("window.__mace.viewport.camera().orbit(80, 40); window.__mace.viewport.camera().zoom(1.7)")
    time.sleep(0.2)
    moved = drv.execute_script("return window.MACE.state.viewers['viewer-main'].getView()")
    assert max(abs(a - b) for a, b in zip(first, moved)) > 1e-3
    U.keys(drv, "")       # Home
    time.sleep(0.6)
    again = drv.execute_script("return window.MACE.state.viewers['viewer-main'].getView()")
    assert max(abs(a - b) for a, b in zip(first, again)) < 1e-4, (first, again)
    # the setting: Along c looks straight down c
    drv.execute_script("window.__mace.settings.set('viewport.defaultView', 'c')")
    time.sleep(0.6)
    t = {x["axis"]: x for x in drv.execute_script("return window.__mace.triad()")}
    assert t["c"]["z"] > 0.99, t
    drv.execute_script("window.__mace.settings.set('viewport.defaultView', 'standard')")
    U.no_errors(drv)


def test_side_by_side_fit_keeps_atoms_inside(fresh):
    """The run's side-by-side view (Phase 3, app.js fitView, unchanged): every
    atom of both panes inside its canvas."""
    drv = fresh[0]
    U.open_doc(drv, "outputs/Ag2Br3_opt_unfinished.out", wait_atoms=False)
    H.wait(drv, "return !!(window.MACE.state.traj && window.MACE.state.viewers['viewer-before'])")
    drv.execute_script("window.MACE.setLook('side')")
    time.sleep(1.0)
    out = drv.execute_script("""const res = [];
        for (const id of ['viewer-before', 'viewer-after']) {
          const v = window.MACE.state.viewers[id]; if (!v) continue;
          const cv = v.getCanvas(), r = cv.getBoundingClientRect(), m = v.getModel();
          const atoms = m ? m.selectedAtoms({}) : [];
          const bad = atoms.filter((a) => { const p = v.modelToScreen({x: a.x, y: a.y, z: a.z});
            return p.x < r.left || p.x > r.right || p.y < r.top || p.y > r.bottom; }).length;
          res.push({id, n: atoms.length, bad, w: r.width});
        }
        return res;""")
    assert len(out) == 2 and all(o["n"] > 0 and o["bad"] == 0 for o in out), out
    drv.execute_script("window.MACE.setLook('overlay')")
    U.no_errors(drv)


def test_paths_abbrev(fresh):
    drv = fresh[0]
    r = drv.execute_async_script("""const done = arguments[0];
      import(document.querySelector('script[src*="/js/shell.js"]').src.replace('shell.js', 'paths.js')).then((P) => done([
        P.abbrev('/home/u/x', {home: '/home/u', max: 60}),
        P.abbrev('/home/u', {home: '/home/u', max: 60}),
        P.abbrev('/home/user/projects/mace/runs/final_review/demo', {home: '/home/user', max: 24}),
        P.abbrev('/home/user/projects/mace/runs/final_review/demo', {home: '/home/user', max: 12}),
        P.abbrev('/srv/group/ProjectFiles/Codebase/workspace0001/test', {home: '/home/user', max: 30}),
        P.abbrev('/data/a/' + 'x'.repeat(50), {home: '', max: 20}),
        P.abbrev('/home/user2/x', {home: '/home/user', max: 60}),
        P.abbrev('/srv/data', {max: 60}),
        P.middle('abcdefghij', 5),
        // polish A9: named folders, the longest match first; "~" only for HOME
        P.abbrev('/scratch/u/p/project', {aliases: [{name: 'HOME', path: '/home/u'}, {name: 'SCRATCH', path: '/scratch/u'}], max: 60}),
        P.abbrev('/scratch/u/work/p', {aliases: [{name: 'SCRATCH', path: '/scratch/u'}, {name: 'WORK', path: '/scratch/u/work'}], max: 60}),
        P.abbrev('/real/home/u/x', {aliases: [{name: 'HOME', path: '/home/u'}, {name: 'HOME', path: '/real/home/u'}], max: 60}),
        P.abbrev('/scratch/u', {aliases: [{name: 'SCRATCH', path: '/scratch/u/'}], max: 60}),
        P.abbrev('/scratch/user2/x', {aliases: [{name: 'SCRATCH', path: '/scratch/user'}], max: 60}),
        P.abbrev('/home/u', {aliases: [{name: 'HOME', path: '/home/u'}], max: 60}),
        P.abbrev('/', {aliases: [{name: 'HOME', path: '/home/u'}], max: 60}),
        P.abbrev('/scratch/u/' + 'a_very_long_project_name_for_review', {aliases: [{name: 'SCRATCH', path: '/scratch/u'}], max: 24}),
      ]), (e) => done(['ERR ' + e.message]));""")
    assert r[0] == "~/x" and r[1] == "~"
    assert r[2] == "~/…/final_review/demo"
    assert r[3] == "~/…/demo"
    assert r[4] == "/srv/…/workspace0001/test"
    assert len(r[5]) == 20 and "…" in r[5] and r[5].startswith("xxx")
    assert r[6] == "/home/user2/x"            # a sibling of the home folder is not ~
    assert r[7] == "/srv/data"
    assert r[8] == "ab…ij"
    assert r[9] == "$SCRATCH/p/project"
    assert r[10] == "$WORK/p"                 # nested: the longest match wins
    assert r[11] == "~/x"                     # home reached through its resolved path
    assert r[12] == "$SCRATCH"
    assert r[13] == "/scratch/user2/x"        # a sibling is not the alias
    assert r[14] == "~" and r[15] == "/"
    assert len(r[16]) <= 24 and "…" in r[16], r[16]


def test_window_title_names_the_folder_and_scratch(tmp_path):
    """Polish A9: HOME is a symlink, the folder is under $SCRATCH (as on a
    cluster): the browser tab says "project — MACE", the menu bar starts with
    $SCRATCH/ and fits at 1600 px, its tooltip is the full path. The same
    under $WORK."""
    real_home = tmp_path / "real_home"
    real_home.mkdir()
    (tmp_path / "home-link").symlink_to(real_home)
    for var in ("SCRATCH", "WORK"):
        area = tmp_path / var.lower()
        root = area / "p" / "project"
        root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(H.EXAMPLES, root)
        xdg = tmp_path / ("xdg_" + var)
        env = {"HOME": str(tmp_path / "home-link"), var: str(area)}
        proc, base, _ = H.start_server(root, xdg, **env)
        drv = H.firefox(1600, 900)
        try:
            H.load(drv, base)
            st = drv.execute_script("const t = document.getElementById('win-title');"
                                    "return [document.title, t.textContent, t.title, t.scrollWidth > t.clientWidth + 1]")
            assert st[0] == "project — MACE", st
            assert f"${var}/p/project" in st[1] and "…" not in st[1], st
            assert st[2] == str(root.resolve()) and not st[3], st
        finally:
            drv.quit()
            proc.terminate()
            proc.wait(timeout=15)


def test_welcome_and_root_chip_abbreviated(tmp_path):
    """Served from a long folder: the welcome sentence, the Explorer's folder
    chip and the window title show it shortened with /…/, nothing clips, and
    the tooltip has the full path; the chip's menu copies it."""
    deep = tmp_path / ("a_rather_long_folder_name_" * 2) / "projects_for_the_review" / "final_review" / "demo"
    deep.parent.mkdir(parents=True)
    root = H.examples_copy(tmp_path)
    root.rename(deep)
    a = U.start_app(tmp_path, root=deep)
    drv = a[0]
    try:
        full = str(deep.resolve())
        st = drv.execute_script("""const chip = document.querySelector('#root-chip'), w = document.querySelector('#welcome-root'),
            t = document.querySelector('#win-title');
            const clip = (el) => el.scrollWidth > el.clientWidth + 1;
            return {chip: chip.textContent, chipTitle: chip.title, welcome: w.textContent, welcomeTitle: w.title,
              title: t.textContent, titleTip: t.title, chipClip: clip(chip), titleClip: clip(t),
              welcomeRight: w.getBoundingClientRect().right, pRight: w.closest('p').getBoundingClientRect().right}""")
        assert st["chipTitle"] == full and st["welcomeTitle"] == full and st["titleTip"] == full
        assert "/…/" in st["chip"] and st["chip"].endswith("demo"), st
        assert "/…/" in st["welcome"] and st["welcome"].endswith("demo"), st
        assert not st["chipClip"] and not st["titleClip"], st
        assert st["welcomeRight"] <= st["pRight"] + 1, st
        # the chip's context menu: Copy Folder Path, from the registry
        chip = drv.find_element("id", "root-chip")
        from selenium.webdriver.common.action_chains import ActionChains
        ActionChains(drv).context_click(chip).perform()
        H.wait(drv, "return document.querySelector('.menu [data-cmd-id=\"file.copyRootPath\"]')", 5)
        U.keys(drv, "")
        U.no_errors(drv)
    finally:
        U.stop_app(a)


def test_no_coming_soon_text(fresh):
    drv = fresh[0]
    static = Path(__file__).resolve().parents[2] / "mace" / "gui" / "static"
    hits = []
    for p in static.rglob("*"):
        if p.is_file() and p.suffix in (".js", ".css", ".html", ".json", ".svg") and "vendor" not in p.parts:
            if re.search(r"coming soon", p.read_text(errors="replace"), re.I):
                hits.append(p.name)
    assert not hits, hits
    U.open_doc(drv, "structures/MgO.cif")
    texts = drv.execute_script("""const out = [];
        const tops = [...document.querySelectorAll('#menus .menu-top')];
        for (const b of tops) {
          b.dispatchEvent(new PointerEvent('pointerdown', {bubbles: true, button: 0}));
          document.querySelectorAll('.menu button').forEach((x) => out.push((x.title || '') + ' ' + x.textContent));
          document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}));
          document.querySelectorAll('.menu').forEach((m) => m.remove());
        }
        window.__mace.palette.results('', window.__mace.ctx()).forEach((r) => out.push(r.label + ' ' + (r.why || '')));
        return out;""")
    assert texts and not [t for t in texts if "coming soon" in t.lower()]
    # an unfinished command names what it arrives with, in the menu, the palette and the toast
    # (Documentation and Open to the Side are real in 1.1.5: the Quantum ESPRESSO export stands in)
    why = drv.execute_script("return window.__mace.menuReason('file.export.QE')")
    assert why == "Needs the Quantum ESPRESSO backend"
    assert drv.execute_script("return window.__mace.palette.results('quantum espresso')[0].why") == why
    drv.execute_script("window.__mace.run('file.export.QE')")
    time.sleep(0.2)
    assert any(t.endswith(": Needs the Quantum ESPRESSO backend") for t in U.toasts(drv)), U.toasts(drv)
    U.no_errors(drv)
