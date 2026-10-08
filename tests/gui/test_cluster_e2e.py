"""F13 in headless Firefox (software GL, fresh profile, real clicks) with
FakeSlurm on the server's PATH: the Cluster button and document, Will It
Start?, Storage with the status chip and a canceled scan, a Job document that
follows a growing real OPT output, finished-job notifications, and the
absence of the view where nothing cluster-like exists."""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

import guiharness as H
import p4util as U

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cluster"))
from cluster_fixtures import SITE_QUOTA_TABLE, TEST_DATA, fake  # noqa: E402

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
from selenium.webdriver.common.by import By  # noqa: E402


def _opt(d: Path, name: str, point: int):
    src = sorted((TEST_DATA / "OPT").glob("Ag1Cl5_sym*.out")) if (TEST_DATA / "OPT").exists() else []
    if not src:
        pytest.skip("test/ data corpus not present")
    d.mkdir(parents=True, exist_ok=True)
    lines = src[0].read_text(errors="replace").splitlines(True)
    heads = [i for i, l in enumerate(lines) if "OPTIMIZATION - POINT" in l]
    (d / f"{name}.out").write_text("".join(lines[:heads[point]]))
    shutil.copy(src[0].with_suffix(".d12"), d / f"{name}.d12")
    return lines, heads


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("clustere2e")
    root = H.examples_copy(tmp)
    home = tmp / "home"
    home.mkdir()
    tool_dir = tmp / "site_bin"
    tool_dir.mkdir()
    tool = tool_dir / "userquota.py"
    tool.write_text(f"#!{sys.executable}\nimport sys\nsys.stdout.write({SITE_QUOTA_TABLE!r} + '\\n')\n")
    tool.chmod(0o555)
    tool_dir.chmod(0o555)
    (home / ".bash_profile").write_text(f"alias quota='NO_COLOR=1 {tool}'\n")
    fs = fake(tmp)
    job_dir = root / "runs" / "mat1"
    lines, heads = _opt(job_dir, "mat1_opt", 5)
    fs.add_record(id="4001", state="RUNNING", name="mat1_opt", workdir=str(job_dir), cpus=32,
                  time_limit="0:30:00", elapsed="0:28:00")
    fs.update(job_text={"4001": f"JobId=4001 JobName=mat1_opt UserId=user1(1001) JobState=RUNNING Reason=None "
                                f"RunTime=00:28:00 TimeLimit=00:30:00 Partition=general-long NodeList=cpu-001 "
                                f"NumNodes=1 NumCPUs=32 WorkDir={job_dir} StdOut={job_dir}/mat1_opt-4001.o "
                                f"Account=acct1 QOS=normal"},
              sacct_rows={"4001": ["4001|mat1_opt|COMPLETED|00:29:00|00:30:00|32|4G||00:00:00|cpu=32|cpu-001|0:0|"
                                   "2030-01-01T00:00:00|2030-01-01T00:29:00|" + str(job_dir)]})
    env = {k: v for k, v in fs.env.items() if k in ("PATH", "FAKE_SLURM_STATE", "FAKE_SLURM_LOG",
                                                    "FAKE_SLURM_HOOK_LOG", "USER", "LOGNAME")}
    env.update(HOME=str(home), MACE_GUI_NO_WARMUP="1", MACE_CLUSTER_TEST_POLL_SECONDS="2", MACE_SITE_CONFIG="none")
    a = U.start_app(tmp, root=root, **env)
    try:
        yield {"app": a, "drv": a[0], "fs": fs, "root": root, "job_dir": job_dir, "lines": lines, "heads": heads,
               "tmp": tmp}
    finally:
        U.stop_app(a)
        tool_dir.chmod(0o755)


def click(drv, css, timeout=15):
    el = H.wait(drv, f"return document.querySelector({css!r})", timeout)
    drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", el)
    el.click()
    return el


def pane_text(drv):
    return drv.execute_script("const p = document.querySelector('#cluster-host .clv-pane'); return p ? p.innerText : ''")


def test_cluster_button_and_tabs(app):
    """F13-a: the activity button opens the document; every tab renders its
    table; Refresh inside the floor counts down and calls nothing."""
    drv, fs = app["drv"], app["fs"]
    H.wait(drv, "return document.querySelector('#activity [data-cmd=\"view.cluster\"]')")
    click(drv, '#activity [data-cmd="view.cluster"]')
    H.wait(drv, "return document.querySelector('#tab-cluster.active .clv-cards')")
    assert "Your jobs" in pane_text(drv) and "mat1_opt" in pane_text(drv)
    assert drv.execute_script("return document.querySelector('.clv-title').textContent") == "Cluster: cpu-cluster"
    assert drv.execute_script("return document.querySelector('#cluster-host [role=tab][data-tab=software]').hidden")
    for tab, want in (("nodes", "amd24"), ("fit", "Find Free Nodes"), ("storage", "home (NFS)"),
                      ("limits", "Max running jobs"), ("overview", "Your jobs")):
        click(drv, f'#cluster-host [role=tab][data-tab="{tab}"]')
        H.wait(drv, f"return (document.querySelector('#cluster-host .clv-pane').innerText || '').includes({want!r})", 20)
        U.shot(drv, f"cluster-{tab}")
    n_nodes = len([c for c in fs.calls("scontrol") if "nodes" in c["argv"]])
    n_squeue = len(fs.calls("squeue"))
    click(drv, '#cluster-host [role=tab][data-tab="nodes"]')
    H.wait(drv, "return (document.querySelector('#cluster-host .clv-pane').innerText || '').includes('amd24')")
    click(drv, "#cluster-host .clv-refresh")
    H.wait(drv, "return /available in \\d+ s/.test(document.querySelector('.clv-strip').textContent) || "
                "document.querySelector('#toasts').innerText.includes('next refresh is allowed')", 10)
    time.sleep(0.5)
    assert len([c for c in fs.calls("scontrol") if "nodes" in c["argv"]]) == n_nodes
    U.no_errors(drv)


def test_will_it_start(app):
    """F13-b: Find Free Nodes and Predict Start show the fixture's numbers
    and the predicted time; exactly one sbatch --test-only."""
    drv, fs = app["drv"], app["fs"]
    drv.execute_script("window.__mace.openCluster('fit')")
    H.wait(drv, "return document.querySelector('#cluster-host .clv-fit')")
    inp = drv.find_element(By.CSS_SELECTOR, "#cluster-host .clv-fit input[name=cpus]")
    inp.clear()
    inp.send_keys("24")
    click(drv, '#cluster-host [data-act="find"]')
    H.wait(drv, "return (document.querySelector('.clv-fit-out').innerText || '').includes('Fits now on')")
    out = drv.execute_script("return document.querySelector('.clv-fit-out').innerText")
    assert "general-long" in out
    before = len([c for c in fs.calls("sbatch") if "--test-only" in c["argv"]])
    click(drv, '#cluster-host [data-act="predict"]')
    H.wait(drv, "return (document.querySelector('.clv-fit-out').innerText || '').includes('SLURM predicts a start')")
    out = drv.execute_script("return document.querySelector('.clv-fit-out').innerText")
    assert "cpu-[001-002]" in out and "general-long" in out and "UTC" in out
    calls = fs.calls("sbatch")
    assert len([c for c in calls if "--test-only" in c["argv"]]) - before == 1
    assert not [c for c in calls if "--test-only" not in c["argv"]]


def test_storage_chip_and_scan_cancel(app):
    """F13-c: warning at 87 %, critical at 91 %, the status-bar chip; a scan
    of a 30 000-file tree starts and Cancel stops its process."""
    drv = app["drv"]
    H.wait(drv, "return document.querySelector('#st-storage')", 20)
    chip = drv.execute_script("return document.querySelector('#st-storage').textContent")
    assert "91" in chip and drv.execute_script("return document.querySelector('#st-storage').classList.contains('crit')")
    click(drv, "#st-storage")
    H.wait(drv, "return document.querySelector('#cluster-host .clv-space')")
    U.shot(drv, "cluster-storage-chip")
    lv = drv.execute_script("return [...document.querySelectorAll('#cluster-host .clv-bar')].map((b) => b.className)")
    assert any("warning" in c for c in lv) and any("critical" in c for c in lv)
    if shutil.disk_usage("/tmp").free < 2 * 1024 ** 3:
        pytest.skip("less than 2 GB free in /tmp")
    tree = app["root"] / "scantree"
    for i in range(30000):
        d = tree / f"d{i // 500:03d}"
        if i % 500 == 0:
            d.mkdir(parents=True, exist_ok=True)
        (d / f"f{i}").write_bytes(b"")
    try:
        click(drv, '#cluster-host [data-act="scan"]')
        root_in = drv.find_element(By.CSS_SELECTOR, "#cluster-host .clv-scanform input[name=root]")
        root_in.clear()
        root_in.send_keys("scantree")
        click(drv, '#cluster-host [data-act="start"]')
        H.wait(drv, "return document.querySelector('#cluster-host [data-act=\"cancel\"]')", 20)
        time.sleep(1.0)
        assert subprocess.run(["pgrep", "-f", f"mace.cluster.filescan {tree}"], capture_output=True).returncode == 0
        click(drv, '#cluster-host [data-act="cancel"]')
        H.wait(drv, "return (document.querySelector('#cluster-host .clv-scan').innerText || '').includes('canceled')", 20)
        time.sleep(1.0)
        assert subprocess.run(["pgrep", "-f", f"mace.cluster.filescan {tree}"], capture_output=True).returncode != 0
    finally:
        shutil.rmtree(tree, ignore_errors=True)


def test_job_document_follows_the_output(app):
    """F13-d: Details on a RUNNING job -> "Point 5 ..."; the next real POINT
    block appended -> the view updates within two polls (2 s hook); the
    walltime banner shows for a short limit."""
    drv = app["drv"]
    drv.execute_script("window.__mace.openCluster('overview')")
    click(drv, '#cluster-host [data-job="4001"]')
    H.wait(drv, "return (document.querySelector('#job-host .jbv-summary') || {}).textContent", 20)
    assert drv.execute_script("return document.querySelector('#job-host .jbv-summary').textContent").startswith("Point 5")
    assert drv.execute_script("return !!document.querySelector('#job-host .jbv-risk')")
    lines, heads = app["lines"], app["heads"]
    with open(app["job_dir"] / "mat1_opt.out", "a") as fh:
        fh.write("".join(lines[heads[5]:heads[6]]))
    t0 = time.time()
    H.wait(drv, "return ((document.querySelector('#job-host .jbv-summary') || {}).textContent || '').startsWith('Point 6')", 8)
    assert time.time() - t0 <= 6.5
    # the ETA as levels: the total first, then the optimization's points (polish I)
    H.wait(drv, "return [...document.querySelectorAll('#job-host .jbv-levels li')].some((li) => "
                "li.textContent.startsWith('Optimization point 6'))", 8)
    items = _levels(drv)
    assert len(items) >= 2 and items[0].startswith("Total"), items
    U.shot(drv, "cluster-job")
    U.no_errors(drv)


def _levels(drv):
    return drv.execute_script("return [...document.querySelectorAll('#job-host .jbv-levels li')]"
                              ".map((li) => li.textContent.trim())")


_CONTRAST_JS = """
function lum(c) { const m = c.match(/[\\d.]+/g).map(Number);
  const f = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
  return 0.2126 * f(m[0]) + 0.7152 * f(m[1]) + 0.0722 * f(m[2]); }
function bg(el) { for (; el; el = el.parentElement) { const c = getComputedStyle(el).backgroundColor;
  if (c && !/rgba\\(.*, 0\\)$/.test(c) && c !== 'transparent') return c; } return 'rgb(255, 255, 255)'; }
return [...document.querySelectorAll(arguments[0])].map((el) => {
  const a = lum(getComputedStyle(el).color), b = lum(bg(el));
  return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05); });
"""


def test_job_document_levels_grow_live(app):
    """The Job document's levels follow an output that keeps growing: real
    POINT blocks appended to the .out one after another (as CRYSTAL writes
    them) move "Optimization point N" on without a click; the "Why this
    estimate" disclosure opens on a real click, holds the total's basis and
    stays open across the refreshes; the list is labeled and its finished
    and pending lines keep AA contrast."""
    drv = app["drv"]
    lines, heads, out = app["lines"], app["heads"], app["job_dir"] / "mat1_opt.out"
    drv.execute_script("window.__mace.openCluster('overview')")
    click(drv, '#cluster-host [data-job="4001"]')
    H.wait(drv, "return document.querySelector('#job-host .jbv-levels li')", 20)
    assert drv.execute_script("return document.querySelector('#job-host .jbv-levels').getAttribute('aria-label')") \
        == "Progress and time left"
    click(drv, "#job-host .jbv-why summary")
    H.wait(drv, "return document.querySelector('#job-host .jbv-why').open")
    why = drv.execute_script("return document.querySelector('#job-host .jbv-why').innerText")
    assert "left in total" in why and "Thresholds:" in why
    done_points = drv.execute_script("return document.querySelector('#job-host .jbv-summary').textContent")
    start = int(done_points.split()[1])
    for point in range(start + 1, min(start + 3, len(heads) - 1) + 1):
        block = lines[heads[point - 1]:heads[point]]
        half = len(block) // 2
        with open(out, "a") as fh:                          # CRYSTAL writes a point in pieces
            fh.write("".join(block[:half]))
        time.sleep(1.0)
        with open(out, "a") as fh:
            fh.write("".join(block[half:]))
        H.wait(drv, "return [...document.querySelectorAll('#job-host .jbv-levels li')].some((li) => "
                    f"li.textContent.startsWith('Optimization point {point} '))", 10)
        items = _levels(drv)
        assert items[0].startswith("Total"), items
        assert drv.execute_script("return document.querySelector('#job-host .jbv-why').open"), point
    assert any(t.startswith("Next point's SCF:") for t in _levels(drv))
    ratios = drv.execute_script(_CONTRAST_JS, "#job-host .jbv-levels .jbv-lt, #job-host .jbv-why li")
    assert ratios and min(ratios) >= 4.5, ratios
    U.shot(drv, "cluster-job-levels")
    U.no_errors(drv)


def test_job_document_freq_levels(app):
    """A frequency job mid-displacement (real lines of a corpus CPHF/Raman
    run, its clock set from the output's own timings): the stages read in
    CRYSTAL's order with the finished ones first, the displacement line
    appears once in the levels (not again as a separate paragraph), and
    "Why this estimate" gives no CPHF sentence as the displacements' reason."""
    drv, fs = app["drv"], app["fs"]
    src = TEST_DATA / "FREQ" / ("1LiFSI-8Sulfolane-conf1_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_"
                                "optimized_freq_HSESOL3C_optimized.out")
    if not src.exists():
        pytest.skip("test/ data corpus not present")
    from mace.cluster import progress as P
    lines = src.read_text(errors="replace").splitlines(True)
    rows = [i for i, l in enumerate(lines) if P._DISP_ROW.match(l)]
    d = app["root"] / "runs" / "freq1"
    d.mkdir(parents=True, exist_ok=True)
    text = "".join(lines[:rows[40] + 1])
    (d / "freq1.out").write_text(text)
    el = int(P.timings(text.splitlines())[-1][2] + 600)
    e = f"{el // 3600}:{el % 3600 // 60:02d}:{el % 60:02d}"
    fs.add_record(id="4102", state="RUNNING", name="freq1", workdir=str(d), cpus=32, time_limit="2-00:00:00", elapsed=e)
    fs.update(job_text={"4102": f"JobId=4102 JobName=freq1 UserId=user1(1001) JobState=RUNNING Reason=None "
                                f"RunTime={e} TimeLimit=2-00:00:00 Partition=general-long NodeList=cpu-001 "
                                f"NumNodes=1 NumCPUs=32 WorkDir={d} StdOut={d}/freq1-4102.o Account=acct1"})
    drv.execute_script("window.__mace.openCluster('overview')")
    H.wait(drv, "return document.querySelector('#cluster-host [data-job=\"4102\"]')", 20)
    click(drv, '#cluster-host [data-job="4102"]')
    H.wait(drv, "return [...document.querySelectorAll('#job-host .jbv-levels li')].some((li) => "
                "li.textContent.startsWith('Displacement 41 of 387'))", 20)
    items = _levels(drv)
    assert items[0].startswith("Total about"), items
    order = [t.split(":")[0].split(" (")[0] for t in items[1:]]
    assert order[:5] == ["Reference SCF for IR/Raman", "IR/Raman", "Raman", "IR/Raman tensors", "Central point"], items
    assert items[-1].startswith("Frequencies and output")
    prog = drv.execute_script("return document.querySelector('#job-host .jbv-body').innerText")
    assert prog.count("Displacement 41 of 387") == 2, prog       # the heading and its level, no third line
    click(drv, "#job-host .jbv-why summary")
    why = drv.execute_script("return document.querySelector('#job-host .jbv-why').innerText")
    assert "Displacements: rows left" in why and "CPHF/CPKS) running" not in why, why
    U.shot(drv, "cluster-job-freq-levels")
    U.no_errors(drv)


def test_notifications_once(app):
    """F13-e: with a stubbed Notification (granted) and the setting on, a
    finished job notifies once; a reload does not repeat it."""
    drv, fs = app["drv"], app["fs"]
    stub = ("window.__notes = []; window.Notification = class { constructor(t, o) { window.__notes.push(t); }"
            " static get permission() { return 'granted'; } static requestPermission() { return Promise.resolve('granted'); } };")
    drv.execute_script(stub)
    drv.execute_script("window.__mace.settings.set('cluster.notify', true)")
    fs.remove_record("4001")
    H.wait(drv, "return window.__notes.length >= 1", 20)
    time.sleep(5)
    assert drv.execute_script("return window.__notes.length") == 1
    # the interim toast's sink ran once for the one event (info toasts dismiss themselves)
    assert drv.execute_script("return (window.__maceJobEvents || []).filter((e) => e.id === '4001').length") == 1
    H.load(drv, app["app"][1])
    drv.execute_script(stub)
    time.sleep(5)
    assert drv.execute_script("return window.__notes.length") == 0
    drv.execute_script("window.__mace.settings.set('cluster.notify', false)")


def test_keyboard_tabs(app):
    """F13-g (part): the tab list moves with the arrow keys and Home/End."""
    drv = app["drv"]
    drv.execute_script("window.__mace.openCluster('overview')")
    H.wait(drv, "return document.querySelector('#cluster-host [role=tab][aria-selected=true]')")
    tab = drv.find_element(By.CSS_SELECTOR, '#cluster-host [role=tab][data-tab="overview"]')
    tab.click()
    from selenium.webdriver.common.keys import Keys
    tab.send_keys(Keys.ARROW_RIGHT)
    assert drv.execute_script("return document.activeElement.dataset.tab") == "nodes"
    drv.switch_to.active_element.send_keys(Keys.END)
    assert drv.execute_script("return document.activeElement.dataset.tab") == "limits"
    labels = drv.execute_script("return [...document.querySelectorAll('#cluster-host input, #cluster-host select')]"
                                ".filter((e) => !e.closest('label') && !e.getAttribute('aria-label')).length")
    assert labels == 0


def _retry(fn, tries=5):
    """A timed refresh may re-render the tab between finding an element and
    using it: try again on a stale element."""
    from selenium.common.exceptions import StaleElementReferenceException
    for i in range(tries):
        try:
            return fn()
        except StaleElementReferenceException:
            if i == tries - 1:
                raise
            time.sleep(0.2)


def _type(drv, css, text, clear=False):
    def go():
        el = drv.find_element(By.CSS_SELECTOR, css)
        el.click()
        if clear:
            el.clear()
        el.send_keys(text)
    _retry(go)


def _blur(drv):
    """Move the focus out of the field (to the selected tab)."""
    _retry(lambda: drv.find_element(By.CSS_SELECTOR, "#cluster-host .clv-tabs [aria-selected=true]").click())


def _value(drv, css):
    return drv.execute_script(f"const i = document.querySelector({css!r}); return i ? i.value : null")


def test_timed_refresh_keeps_what_was_typed(app):
    """The timed refresh (2 s with the hook) re-renders the tab: a typed
    Limits question, its answer, the Will It Start? fields and an open scan
    form all survive it."""
    drv = app["drv"]
    drv.execute_script("window.__mace.openCluster('limits')")
    H.wait(drv, "return document.querySelector('#cluster-host input[name=n]')")
    _type(drv, "#cluster-host input[name=n]", "600")
    _blur(drv)
    time.sleep(5)
    assert _value(drv, "#cluster-host input[name=n]") == "600"
    _type(drv, "#cluster-host input[name=c]", "4")
    _retry(lambda: click(drv, '#cluster-host [data-act="check"]'))
    H.wait(drv, "return document.querySelector('#cluster-host .clv-head-out').innerText.trim().length > 0", 10)
    answer = drv.execute_script("return document.querySelector('#cluster-host .clv-head-out').innerText")
    time.sleep(5)
    assert drv.execute_script("return document.querySelector('#cluster-host .clv-head-out').innerText") == answer
    assert _value(drv, "#cluster-host input[name=n]") == "600"
    _retry(lambda: click(drv, '#cluster-host [role=tab][data-tab="fit"]'))
    H.wait(drv, "return document.querySelector('#cluster-host .clv-fit input[name=constraint]')")
    _type(drv, "#cluster-host .clv-fit input[name=constraint]", "amd24")
    _blur(drv)
    time.sleep(5)
    assert _value(drv, "#cluster-host .clv-fit input[name=constraint]") == "amd24"
    _retry(lambda: click(drv, '#cluster-host [role=tab][data-tab="storage"]'))
    H.wait(drv, "return document.querySelector('#cluster-host [data-act=scan]')")
    _retry(lambda: click(drv, '#cluster-host [data-act="scan"]'))
    _type(drv, "#cluster-host .clv-scanform input[name=root]", "runs", clear=True)
    _blur(drv)
    time.sleep(5)
    assert _value(drv, "#cluster-host .clv-scanform input[name=root]") == "runs"
    U.no_errors(drv)


def test_cluster_css_stays_in_its_views(app):
    """cluster.css is global once loaded: a .warn or .chip of another view
    keeps its own look."""
    drv = app["drv"]
    drv.execute_script("window.__mace.openCluster('overview')")
    H.wait(drv, "return document.querySelector('#tab-cluster.active .clv-cards')")
    got = drv.execute_script(
        "const p = document.createElement('p'); p.className = 'warn'; document.body.appendChild(p);"
        "const b = document.createElement('span'); b.className = 'chip'; document.body.appendChild(b);"
        "const r = [getComputedStyle(p).display, getComputedStyle(b).paddingTop];"
        "p.remove(); b.remove(); return r;")
    assert got[0] == "block", got
    assert got[1] == "3px", got                    # app.css's .chip, not the Cluster view's
    got = drv.execute_script(
        "const o = document.createElement('ol'); o.innerHTML = '<li class=\"done d1\">x</li>';"
        "document.body.appendChild(o); const li = o.firstChild;"
        "const r = [getComputedStyle(o).listStyleType, getComputedStyle(li).paddingLeft, getComputedStyle(li).fontStyle];"
        "o.remove(); return r;")
    assert got == ["decimal", "0px", "normal"], got     # the Job document's level styles stay in .jbv-levels


def test_software_tab(tmp_path):
    """F13 Software: with a module system, the tab shows the CRYSTAL row of
    Submit's site.crystal_module() and the xtb module; Refresh asks again;
    without one the tab is hidden (the other tests' server). The two helpers
    are stand-ins injected into the server process."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cluster"))
    from test_cluster_software import FAKE_ENVMODULES, FAKE_SITE
    inject = tmp_path / "inject"
    inject.mkdir()
    (inject / "sitecustomize.py").write_text(
        "import sys, types\n"
        f"_e = types.ModuleType('mace.utils.envmodules'); exec({FAKE_ENVMODULES!r}, _e.__dict__)\n"
        f"_s = types.ModuleType('mace.submission.site'); exec({FAKE_SITE!r}, _s.__dict__)\n"
        "sys.modules['mace.utils.envmodules'] = _e\nsys.modules['mace.submission.site'] = _s\n")
    fs = fake(tmp_path)
    env = {k: v for k, v in fs.env.items() if k in ("PATH", "FAKE_SLURM_STATE", "FAKE_SLURM_LOG",
                                                    "FAKE_SLURM_HOOK_LOG", "USER", "LOGNAME")}
    env.update(HOME=str(tmp_path), MACE_GUI_NO_WARMUP="1", MACE_SITE_CONFIG="none", PYTHONPATH=str(inject))
    a = U.start_app(tmp_path, root=H.examples_copy(tmp_path), **env)
    try:
        drv = a[0]
        H.wait(drv, "return document.querySelector('#activity [data-cmd=\"view.cluster\"]')")
        click(drv, '#activity [data-cmd="view.cluster"]')
        H.wait(drv, "const b = document.querySelector('#cluster-host [role=tab][data-tab=software]'); return b && !b.hidden")
        click(drv, '#cluster-host [role=tab][data-tab="software"]')
        try:
            H.wait(drv, "return document.querySelector('#cluster-host .clv-soft')", 20)
        except AssertionError:
            raise AssertionError(pane_text(drv) + " | " + str(H.errors(drv)))
        text = pane_text(drv)
        assert "CRYSTAL/23-intel-2023a" in text and "not used: CRYSTAL/23: no Pcrystal" in text
        assert "xtb/6.7.1-gfbf-2024a" in text
        U.shot(drv, "cluster-software")
        click(drv, "#cluster-host .clv-refresh")
        H.wait(drv, "return document.querySelector('#cluster-host .clv-soft')", 20)
        U.no_errors(drv)
    finally:
        U.stop_app(a)


def test_no_view_without_cluster_tools(tmp_path):
    """F13-f: no SLURM and no quota tool on PATH: no Cluster button, no chip."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    for tool in ("bash", "df", "nice"):
        p = shutil.which(tool)
        if p:
            (bin_ / tool).symlink_to(p)
    root = H.examples_copy(tmp_path)
    a = U.start_app(tmp_path, root=root, PATH=str(bin_), MACE_GUI_NO_WARMUP="1", HOME=str(tmp_path),
                    MACE_SITE_CONFIG="none")
    try:
        drv = a[0]
        time.sleep(2)
        assert not drv.execute_script("return document.querySelector('#activity [data-cmd=\"view.cluster\"]')")
        assert not drv.execute_script("return document.querySelector('#st-storage')")
    finally:
        U.stop_app(a)


def test_job_document_array_tasks(app):
    """A Job document of an array task lists the array's tasks; Details on
    another task (a real click) opens that task's document."""
    drv, fs = app["drv"], app["fs"]
    d = app["root"] / "runs" / "arr"
    d.mkdir(parents=True, exist_ok=True)
    for t, st in ((1, "RUNNING"), (2, "RUNNING"), (3, "PENDING")):
        fs.add_record(id=f"5000_{t}", array="5000", task=t, state=st, name="arr_sp", workdir=str(d), cpus=4)
    try:
        st = fs.load()
        jt = dict(st.get("job_text") or {})
        for t in (1, 2, 3):
            jt[f"5000_{t}"] = (f"JobId={5000 + t} ArrayJobId=5000 ArrayTaskId={t} JobName=arr_sp UserId=user1(1001) "
                               f"JobState=RUNNING Reason=None RunTime=00:01:00 TimeLimit=01:00:00 WorkDir={d} "
                               "NumNodes=1 NumCPUs=4 Partition=general-short Account=acct1")
        fs.update(job_text=jt)
        time.sleep(2.5)                                   # the next poll reads the new rows (2 s hook)
        drv.execute_script("window.__mace.openJob('5000_2')")
        H.wait(drv, "return document.querySelector('#job-host .jbv-array')", 20)
        txt = drv.execute_script("return document.querySelector('#job-host .jbv-array').innerText")
        assert "5000_1" in txt and "5000_3" in txt and "this task" in txt
        click(drv, '#job-host .jbv-array [data-job="5000_3"]')
        H.wait(drv, "return ((document.querySelector('#job-host h2') || {}).textContent || '').startsWith('Job 5000_3')", 20)
        U.shot(drv, "cluster-job-array")
        U.no_errors(drv)
    finally:
        for t in (1, 2, 3):
            fs.remove_record(f"5000_{t}")
