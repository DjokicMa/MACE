"""The Jobs panel's sticky header covers the rows scrolled under it (Firefox
and Chromium, light and dark): it has an opaque background from the theme's
tokens, it paints above every row cell (the sticky actions column too), and
it stays below the menus' layer. Fake SLURM, demo data only."""
import pytest

import guiharness as H
import submit_util as SU

pytestmark = [pytest.mark.gui, pytest.mark.guismoke]
pytest.importorskip("selenium.webdriver")

N_JOBS = 40


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("jobshead")
    root = tmp / "proj"
    d = root / "decks"
    d.mkdir(parents=True)
    fake = SU.FakeSlurm(tmp / "fs", {"sacct": {}})
    for i in range(N_JOBS):
        (d / f"mat{i}.d12").write_text(SU.SP_D12)
        fake.add_record(id=str(5000 + i), name=f"mat{i}", state="RUNNING" if i % 2 else "PENDING", user=SU.USER,
                        workdir=str(d), elapsed="1:02", limit="2:00:00", cpus=4, nodelist="n007", reason="Priority")
    env = SU.fake_env(fake, tmp)
    env.pop("XDG_CACHE_HOME")
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_TEST_HOOKS="1", MACE_GUI_NO_WARMUP="1", **env)
    yield base
    proc.terminate()
    proc.wait(timeout=30)


PROBE = """
const table = document.querySelector('.bottom-pane.jobs table.jobs-table');
let sc = table.parentElement;
while (sc && !(sc.scrollHeight > sc.clientHeight + 4 && /(auto|scroll)/.test(getComputedStyle(sc).overflowY))) sc = sc.parentElement;
if (!sc) return {error: 'no scrolling container'};
sc.scrollTop = 0;
const rowH = table.tBodies[0].rows[0].getBoundingClientRect().height;
sc.scrollTop = rowH * 6.5;            // rows now pass under the header
const ths = [...table.tHead.rows[0].cells];
const out = {scrolled: sc.scrollTop, cells: []};
for (const th of ths) {
  const r = th.getBoundingClientRect(), cs = getComputedStyle(th);
  const x = r.left + r.width / 2, y = r.top + r.height / 2;
  const hit = document.elementFromPoint(x, y);
  // the scroller's top edge (inside its border): no strip above the header where rows show
  const edge = sc.getBoundingClientRect().top + sc.clientTop;
  const above = document.elementFromPoint(x, edge + 1);
  out.cells.push({bg: cs.backgroundColor, z: cs.zIndex, top: r.top, scTop: edge,
                  hitInHeader: !!(hit && hit.closest('thead') === table.tHead), hit: hit ? hit.tagName + '.' + hit.className : null,
                  edgeInHeader: !!(above && above.closest('thead') === table.tHead),
                  edgeHit: above ? above.tagName + '.' + above.className : null});
}
out.zMenu = getComputedStyle(document.documentElement).getPropertyValue('--z-menu').trim();
return out;
"""


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("browser", ["firefox", "chromium"])
def test_jobs_header_covers_the_rows_under_it(served, browser, theme):
    try:
        drv = H.browser(browser, width=1280, height=800)
    except Exception as e:                       # noqa: BLE001
        if browser == "chromium":
            pytest.skip(f"no Chromium: {e}")
        raise
    try:
        H.load(drv, served)
        drv.execute_script("document.documentElement.setAttribute('data-theme', arguments[0])", theme)
        drv.execute_script("window.__mace.run('view.jobs')")
        H.wait(drv, f"return document.querySelectorAll('.bottom-pane.jobs tr[data-id]').length === {N_JOBS}", 40)
        res = drv.execute_script(PROBE)
        assert "error" not in res, res
        assert res["scrolled"] > 0, res
        for c in res["cells"]:
            assert c["bg"] not in ("transparent", "rgba(0, 0, 0, 0)"), c
            assert c["hitInHeader"], c                       # the header, not a row, is on top
            assert abs(c["top"] - c["scTop"]) <= 1, c        # stuck at the very top of the scroller
            assert c["edgeInHeader"], c                      # no row in a strip above it
            assert c["z"] != "auto" and int(c["z"]) < int(res["zMenu"]), (c, res["zMenu"])
        # the header's color is the theme's surface token, so it follows the theme
        surface = drv.execute_script("""const p = document.createElement('div'); p.style.background = 'var(--surface)';
            document.body.appendChild(p); const c = getComputedStyle(p).backgroundColor; p.remove(); return c;""")
        assert {c["bg"] for c in res["cells"]} == {surface}, (res["cells"][0], surface)
    except BaseException:
        H.save_failure(drv, f"jobs_header_{browser}_{theme}")
        raise
    finally:
        drv.quit()
