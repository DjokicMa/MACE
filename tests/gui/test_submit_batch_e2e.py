"""Submit, Jobs and Batch Setup in headless Firefox, real clicks, through the
real `mace gui` with the fake SLURM on its PATH (PLAN_submit D3-D7, D14,
D19, D20, D25).

The decks are copies of real diamond decks from the corpus (test/), or the
public diamond deck of the default-path goldens without it."""
import json
import shutil
import time

import pytest

import guiharness as H
import p4util as U
import submit_util as SU

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")
from selenium.webdriver.common.action_chains import ActionChains  # noqa: E402
from selenium.webdriver.common.keys import Keys  # noqa: E402

from conftest import TEST_DATA  # noqa: E402


def _decks(root):
    d = root / "proj" / "decks"
    d.mkdir(parents=True)
    src = [TEST_DATA / "OPT" / f"{n}.d12" for n in ("1_dia_opt_rev1", "2_dia2_opt_rev1", "3_dia3_opt_rev1")]
    for i, s in enumerate(src, 1):
        if s.is_file():
            shutil.copy2(s, d / f"mat{i}_sp.d12")
        else:
            (d / f"mat{i}_sp.d12").write_text(SU.SP_D12)
    cifs = root / "cifs"
    cifs.mkdir()
    for n in ("MgO", "NaCl"):
        shutil.copy2(H.EXAMPLES / "structures" / f"{n}.cif", cifs)
    return d


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sube2e")
    root = H.examples_copy(tmp)
    _decks(root)
    fake = SU.FakeSlurm(tmp / "fs", {"sacct": {}})
    env = SU.fake_env(fake, tmp)
    env.pop("XDG_CACHE_HOME")
    a = U.start_app(tmp, root=root, **env)
    try:
        yield a, fake
    finally:
        U.stop_app(a)


@pytest.fixture
def fresh(app):
    (a, fake) = app
    U.reset(a[0])
    a[0].execute_script("window.__mace.profile.remember('submitLast', {})")    # each test starts from the defaults
    fake.log_file.write_text("")
    return a, fake


def _menu_rows(drv, label):
    H.wait(drv, f"return !!document.querySelector('.menu[aria-label=\"{label}\"]')", 5)
    return drv.execute_script(f"""return [...document.querySelector('.menu[aria-label="{label}"]').children].map((el) =>
        el.classList.contains('menu-sep') ? '-' : el.querySelector('.lbl').textContent)""")


def _click_item(drv, label):
    drv.execute_script("[...document.querySelectorAll('.menu .menu-item')].find((b) => b.querySelector('.lbl').textContent === arguments[0]).click()", label)


def _right_click(drv, css):
    el = drv.find_element("css selector", css)
    drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", el)
    ActionChains(drv).context_click(el).perform()


def _dialog(drv, timeout=60):
    H.wait(drv, "return !!document.querySelector('#submit-dialog[open]')", 15)
    try:
        H.wait(drv, "const d = document.querySelector('#submit-dialog'); return d && d.querySelector('[data-part=command]') && !d.querySelector('[data-part=previewing]')", timeout)
    except AssertionError:
        tasks = drv.execute_async_script("const done = arguments[0]; fetch('api/tasks').then((r) => r.json())"
                                         ".then((j) => done(j.tasks.map((x) => [x.kind, x.state, x.title, x.error])))")
        raise AssertionError("the dialog shows no preview: " + drv.execute_script(
            "return document.querySelector('#submit-dialog .dlg-body').innerText") + f"\ntasks: {tasks}\nerrors: {H.errors(drv)}")


def _part(drv, name):
    return drv.execute_script("const e = document.querySelector(`#submit-dialog [data-part='${arguments[0]}']`); return e ? e.textContent : null", name)


def _sbatch(fake):
    return [c for c in fake.calls("sbatch") if "--test-only" not in c["argv"]]


def _api(base, path, data=None):
    return U.req(base, path, data)


# ---------------------------------------------------------------- D3: never on its own
def test_no_auto_submit(fresh):
    (drv, base, root, xdg, proc, dl), fake = fresh
    keys = drv.execute_script("return window.__mace.commands.filter((c) => ['calc.submit', 'calc.runHere'].includes(c.id)).map((c) => [c.id, c.keys || []])")
    assert all(k == [] for _, k in keys), keys
    _right_click(drv, '#file-list .file[data-path="proj/decks/mat1_sp.d12"]')
    _menu_rows(drv, "Explorer menu")
    _click_item(drv, "Submit…")
    _dialog(drv)
    assert drv.execute_script("return document.activeElement.dataset.act") == "cancel"
    # Enter on the dialog never submits
    drv.execute_script("document.querySelector('#submit-dialog input, #submit-dialog select') && document.querySelector('#submit-dialog input').focus()")
    U.keys(drv, Keys.ENTER)
    time.sleep(1.0)
    assert drv.execute_script("return !!document.querySelector('#submit-dialog[open]')")
    assert _sbatch(fake) == []
    U.keys(drv, Keys.ESCAPE)
    H.wait(drv, "return !document.querySelector('#submit-dialog[open]')", 5)
    assert _sbatch(fake) == []
    U.no_errors(drv)


# ---------------------------------------------------------------- D4, D5, D6: the dialog says what is submitted
def test_dialog_shows_exact_command_and_jobs_cancel_one_task(fresh):
    (drv, base, root, xdg, proc, dl), fake = fresh
    _right_click(drv, '#file-list .folder[data-dir="proj/decks"]')
    assert _menu_rows(drv, "Folder menu")[:2] == ["Batch Setup…", "Submit Folder…"]
    _click_item(drv, "Submit Folder…")
    _dialog(drv)
    assert _part(drv, "command") == "cd proj && mace submit decks"
    res = _part(drv, "resources")
    assert "mat1_sp" in res and "cores" in res
    writes = _part(drv, "writes")
    assert "proj/decks/mat1_sp.sh" in writes
    assert "proj/decks/mat1_sp.out" in _part(drv, "job-writes")
    assert _sbatch(fake) == []
    # click every option: each changes the command it shows
    drv.find_element("css selector", '#submit-dialog [data-o="array"]').click()
    H.wait(drv, "return (document.querySelector('#submit-dialog [data-part=command]') || {}).textContent === 'cd proj && mace submit --array decks'", 30)
    mr = drv.find_element("css selector", '#submit-dialog [data-o="max_running"]')
    mr.send_keys("2")
    H.wait(drv, "return (document.querySelector('#submit-dialog [data-part=command]') || {}).textContent === 'cd proj && mace submit --array --max-running 2 decks'", 30)
    drv.find_element("css selector", '#submit-dialog [data-o="track"]').click()
    H.wait(drv, "return ((document.querySelector('#submit-dialog [data-part=command]') || {}).textContent || '').includes('--track')", 30)
    wt = drv.find_element("css selector", '#submit-dialog [data-o="walltime"]')
    wt.send_keys("0:30:00")
    H.wait(drv, "return ((document.querySelector('#submit-dialog [data-part=command]') || {}).textContent || '').includes('--walltime 0:30:00')", 30)
    drv.find_element("css selector", '#submit-dialog input[name="sub-ex"][value="nodes"]').click()
    ex = drv.find_element("css selector", '#submit-dialog [data-o="exclude-value"]')
    ex.send_keys("n001")
    H.wait(drv, "return ((document.querySelector('#submit-dialog [data-part=command]') || {}).textContent || '').includes('--exclude-nodes n001')", 30)
    drv.find_element("css selector", '#submit-dialog input[name="sub-ex"][value="none"]').click()
    H.wait(drv, "return !(document.querySelector('#submit-dialog [data-part=command]') || {textContent: '-'}).textContent.includes('--exclude')", 30)
    sel = drv.find_element("css selector", '#submit-dialog [data-o="progress"]')
    opts = drv.execute_script("return [...arguments[0].options].map((o) => [o.value, o.disabled])", sel)
    assert ("custom", False) not in [tuple(o) for o in opts] and all(o[0] != "interactive" for o in opts)
    assert _sbatch(fake) == []
    shown = _part(drv, "command")
    _dialog(drv)
    # the preview the dialog shows is what the server previewed
    btn = drv.find_element("css selector", '#submit-dialog [data-act="submit"]')
    H.wait(drv, "return !document.querySelector('#submit-dialog [data-act=submit]').disabled", 30)
    assert btn.text == "Submit 3 jobs"
    btn.click()
    H.wait(drv, "return !document.querySelector('#submit-dialog[open]')", 60)
    H.wait(drv, "return [...document.querySelectorAll('#toasts > *')].some((t) => t.textContent.includes('Submitted 3 jobs'))", 30)
    toast = [t for t in U.toasts(drv) if "Submitted 3 jobs" in t][0]
    sb = _sbatch(fake)
    assert len(sb) == 1 and sb[0]["argv"][:2] == ["sbatch", "--hold"]
    arr = [c for c in fake.calls("scontrol") if c["argv"][1] == "release"][0]["argv"][2]
    for t in range(3):
        assert f"{arr}_{t}" in toast
    # the scripts carry what the dialog said
    assert "#SBATCH --array=0-2%2" in sb[0]["script_text"]
    assert shown == "cd proj && mace submit --array --max-running 2 --track --walltime 0:30:00 decks"
    # the options are remembered for the next Submit (Settings › Remember Submit options), in the GUI profile
    last = drv.execute_script("return window.__mace.profile.memory('submitLast').slurm")
    assert last["array"] and last["max_running"] == "2" and last["walltime"] == "0:30:00"
    # Jobs: the tasks grouped under their array, cancel one task
    for t in range(3):
        fake.set_record(f"{arr}_{t}", workdir=str(root / "proj" / "decks"))
    drv.execute_script("[...document.querySelectorAll('#toasts .acts button')].find((b) => b.textContent === 'Open Jobs').click()")
    H.wait(drv, "return document.querySelector('.bottom-pane.jobs.active') !== null", 10)
    time.sleep(15.5)            # past the shared reader's 15 s refresh floor
    drv.find_element("css selector", '.bottom-pane.jobs [data-act="refresh"]').click()
    H.wait(drv, f"return document.querySelectorAll('.bottom-pane.jobs tr[data-id^=\"{arr}_\"]').length === 3", 20)
    assert f"Array {arr}" in drv.execute_script("return document.querySelector('.bottom-pane.jobs tr.group').textContent")
    fake.log_file.write_text("")
    drv.find_element("css selector", f'.bottom-pane.jobs [data-act="cancel"][data-id="{arr}_1"]').click()
    H.wait(drv, "return !!document.querySelector('#setup-confirm[open]')", 5)
    body = drv.execute_script("return document.querySelector('#setup-confirm .dlg-body').textContent")
    assert f"scancel {arr}_1" in body
    drv.find_element("css selector", '#setup-confirm [data-r="yes"]').click()
    H.wait(drv, f"return [...document.querySelectorAll('#toasts > *')].some((t) => t.textContent.includes('Canceled {arr}_1'))", 20)
    assert [c["argv"] for c in fake.calls("scancel")] == [["scancel", f"{arr}_1"]]
    H.wait(drv, "return /Jobs: \\d+ running · \\d+ pending/.test(document.querySelector('#statusbar').textContent)", 20)
    U.no_errors(drv)


# ---------------------------------------------------------------- D25: finished toast once
def test_job_finished_toast(fresh):
    (drv, base, root, xdg, proc, dl), fake = fresh
    d = root / "proj" / "decks"
    fake.add_record(id="9100", name="mat2_sp", state="RUNNING", user=SU.USER, workdir=str(d))
    fake.update(sacct={"9100": "FAILED"})
    drv.execute_script("window.__mace.run('view.jobs')")
    time.sleep(15.5)
    drv.find_element("css selector", '.bottom-pane.jobs [data-act="refresh"]').click()
    H.wait(drv, "return !!document.querySelector('.bottom-pane.jobs tr[data-id=\"9100\"]')", 20)
    fake.remove_record("9100")
    time.sleep(15.5)
    drv.find_element("css selector", '.bottom-pane.jobs [data-act="refresh"]').click()
    H.wait(drv, "return [...document.querySelectorAll('#toasts > *')].some((t) => t.textContent.includes('mat2_sp (9100) finished: FAILED'))", 20)
    time.sleep(1)
    n = drv.execute_script("return [...document.querySelectorAll('#toasts > *')].filter((t) => t.textContent.includes('(9100)')).length")
    assert n == 1
    drv.execute_script("document.querySelector('.bottom-tab[data-bottom=problems]').click()")
    assert "mat2_sp (9100) finished: FAILED" in drv.execute_script("return document.querySelector('.bottom-pane.problems').textContent")
    # one sacct for this job (an earlier test's canceled task may be confirmed now too: the
    # own-jobs answer is cached for up to a minute, shared with the Cluster view)
    assert len([c for c in fake.calls("sacct") if "9100" in " ".join(c["argv"])]) == 1


# ---------------------------------------------------------------- Jobs row -> Job document
def test_jobs_row_details_and_double_click_open_the_job_document(fresh):
    """PLAN_F: the Job document (progress, ETA) opens from a Jobs row's Details
    and from a double-click on the row (both were missing on HPCC)."""
    (drv, base, root, xdg, proc, dl), fake = fresh
    d = root / "proj" / "decks"
    fake.add_record(id="9200", name="mat3_sp", state="RUNNING", user=SU.USER, workdir=str(d))
    try:
        drv.execute_script("window.__mace.run('view.jobs')")
        time.sleep(15.5)
        drv.find_element("css selector", '.bottom-pane.jobs [data-act="refresh"]').click()
        H.wait(drv, "return !!document.querySelector('.bottom-pane.jobs tr[data-id=\"9200\"]')", 20)
        btn = drv.find_element("css selector", '.bottom-pane.jobs [data-act="details"][data-id="9200"]')
        ActionChains(drv).move_to_element(btn).click().perform()
        H.wait(drv, "const d = window.__mace.doc(); return !!d && d.kind === 'job' && d.jobId === '9200'", 10)
        n = drv.execute_script("return window.__mace.docs().length")
        drv.execute_script("window.__mace.run('view.jobs')")
        drv.execute_script("window.__mace.activate(0)")
        row = drv.find_element("css selector", '.bottom-pane.jobs tr[data-id="9200"] td.name')
        ActionChains(drv).double_click(row).perform()
        H.wait(drv, "const d = window.__mace.doc(); return !!d && d.kind === 'job' && d.jobId === '9200'", 10)
        assert drv.execute_script("return window.__mace.docs().length") == n      # the same document again
        U.no_errors(drv)
    finally:
        fake.remove_record("9200")


# ---------------------------------------------------------------- D14, D3 for batches
def test_batch_setup_never_submits_and_diffs(fresh):
    (drv, base, root, xdg, proc, dl), fake = fresh
    _right_click(drv, '#file-list .folder[data-dir="cifs"]')
    _menu_rows(drv, "Folder menu")
    _click_item(drv, "Batch Setup…")
    H.wait(drv, "return document.querySelectorAll('#batch-host [data-part=rows] tbody tr').length === 2", 60)
    assert drv.execute_script("return window.__mace.doc().title") == "Batch: cifs"
    drv.find_element("css selector", '#batch-host [data-go="settings"]').click()
    H.wait(drv, "return !!document.querySelector('#batch-host [data-part=fields]')", 120)
    time.sleep(0.5)                       # the form settles (it renders once more when the CLI's answer lands)
    mc = drv.find_elements("css selector", '#batch-host [data-ans="maxcycle"]')
    if mc:
        from selenium.common.exceptions import StaleElementReferenceException
        for _ in range(5):
            try:
                el = drv.find_element("css selector", '#batch-host [data-ans="maxcycle"]')
                drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", el)
                el.clear()
                el.send_keys("444")
                el.send_keys(Keys.TAB)
                break
            except StaleElementReferenceException:
                time.sleep(0.3)
        H.wait(drv, "return ((document.querySelector('#batch-host [data-part=changed]') || {}).textContent || '').includes('maxcycle = 444')", 10)
    drv.find_element("css selector", '#batch-host [data-go="preview"]').click()
    H.wait(drv, "return document.querySelectorAll('#batch-host [data-part=plan] tbody tr').length === 2", 300)
    drv.find_element("css selector", '#batch-host [data-diff="cifs/MgO.cif"]').click()
    H.wait(drv, "return window.__mace.doc() && window.__mace.doc().kind === 'text'", 20)
    if mc:
        assert "+444" in drv.execute_script("return document.querySelector('#text-body').textContent")
    idx = drv.execute_script("return window.__mace.docs().findIndex((x) => x.kind === 'batch')")
    drv.execute_script("window.__mace.activate(arguments[0])", idx)
    H.wait(drv, "return window.__mace.doc().kind === 'batch'", 10)
    drv.find_element("css selector", '#batch-host [data-go="write"]').click()
    name = drv.find_element("css selector", '#batch-host [data-b="name"]')
    name.clear()
    name.send_keys("../x")
    assert drv.find_element("css selector", '#batch-host [data-act="write"]').get_attribute("disabled")
    name.clear()
    name.send_keys("batch_e2e")
    drv.find_element("css selector", '#batch-host [data-act="write"]').click()
    H.wait(drv, "return !!document.querySelector('#batch-host [data-part=commands]')", 300)
    cmds = drv.execute_script("return document.querySelector('#batch-host [data-part=commands]').textContent")
    assert cmds.startswith("mace convert --batch --options_file batch_e2e/options/")
    assert (root / "cifs" / "batch_e2e" / "commands.sh").is_file()
    assert _sbatch(fake) == []                                  # writing a batch never submits
    drv.find_element("css selector", '#batch-host [data-act="submit"]').click()
    _dialog(drv)
    assert _part(drv, "command") == "cd cifs && mace submit --array batch_e2e/decks"
    drv.find_element("css selector", '#submit-dialog [data-act="cancel"]').click()
    assert _sbatch(fake) == []
    U.no_errors(drv)


# ---------------------------------------------------------------- D20: accessibility
def test_submit_dialog_and_jobs_accessibility(fresh):
    (drv, base, root, xdg, proc, dl), fake = fresh
    drv.execute_script("document.getElementById('theme-btn').focus()")
    drv.execute_script("window.__mace.run('calc.submit', {paths: ['proj/decks/mat2_sp.d12']})")
    _dialog(drv)
    unnamed = drv.execute_script("""return [...document.querySelectorAll('#submit-dialog input, #submit-dialog select, #submit-dialog button')]
        .filter((e) => !(e.labels && e.labels.length) && !e.getAttribute('aria-label') && !e.textContent.trim()).map((e) => e.outerHTML.slice(0, 80))""")
    assert not unnamed, unnamed
    try:
        axe_mod = pytest.importorskip("axe_selenium_python")
        axe = axe_mod.Axe(drv)
        axe.inject()
        res = drv.execute_async_script("const done = arguments[arguments.length - 1]; axe.run('#submit-dialog', "
                                       "{runOnly: {type: 'tag', values: ['wcag2a', 'wcag2aa']}}).then(done, (e) => done({error: String(e)}))")
        bad = [v["id"] for v in res.get("violations", []) if v.get("impact") in ("serious", "critical")]
        assert not bad, bad
    finally:
        U.keys(drv, Keys.ESCAPE)
    H.wait(drv, "return !document.querySelector('#submit-dialog[open]')", 5)
    assert drv.execute_script("return document.activeElement.id") == "theme-btn"
    drv.execute_script("window.__mace.run('view.jobs')")
    H.wait(drv, "return !!document.querySelector('.bottom-pane.jobs .jobs-bar')", 15)
    rows = drv.execute_script("return [...document.querySelectorAll('.bottom-pane.jobs tr[data-id]')].map((r) => r.tabIndex)")
    assert all(t == 0 for t in rows)


# ---------------------------------------------------------------- D7: no SLURM here
def test_without_slurm(tmp_path):
    root = H.examples_copy(tmp_path)
    _decks(root)
    a = U.start_app(tmp_path, root=root, PATH="/usr/bin:/bin")
    try:
        drv = a[0]
        drv.execute_script("window.__mace.run('calc.submit', {paths: ['proj/decks/mat1_sp.d12']})")
        H.wait(drv, "return !!document.querySelector('#submit-dialog[open]')", 15)
        assert drv.execute_script("return document.querySelector('#submit-dialog input[name=sub-where][value=slurm]').disabled")
        assert drv.execute_script("return document.querySelector('#submit-dialog input[name=sub-where][value=local]').checked")
        assert drv.execute_script("return document.querySelector('#submit-title').textContent") == "Run on This Node"
        U.keys(drv, Keys.ESCAPE)
        drv.execute_script("window.__mace.run('view.jobs')")
        H.wait(drv, "return (document.querySelector('#jobs-note') || {}).textContent === 'No SLURM on this machine. Jobs you run here with Run on This Node appear in Tasks.'", 15)
        U.no_errors(drv)
    finally:
        U.stop_app(a)


# ---------------------------------------------------------------- D26: above 20 decks the preview is explicit
def test_many_decks_need_update_preview(fresh):
    (drv, base, root, xdg, proc, dl), fake = fresh
    many = root / "many"
    many.mkdir(exist_ok=True)
    for i in range(21):
        (many / f"m{i:02d}.d12").write_text(SU.SP_D12)
    drv.execute_script("window.__mace.run('calc.submit', {paths: ['many']})")
    _dialog(drv)
    assert drv.find_element("css selector", '#submit-dialog [data-act="submit"]').text == "Submit 21 jobs"
    drv.find_element("css selector", '#submit-dialog [data-o="track"]').click()
    H.wait(drv, "return !!document.querySelector('#submit-dialog [data-act=update]')", 10)
    assert drv.execute_script("return document.querySelector('#submit-dialog [data-act=submit]').disabled")
    assert "--track" not in _part(drv, "command")                    # no preview ran on its own
    drv.find_element("css selector", '#submit-dialog [data-act="update"]').click()
    H.wait(drv, "return (document.querySelector('#submit-dialog [data-part=command]') || {}).textContent === 'cd . && mace submit --track many' || (document.querySelector('#submit-dialog [data-part=command]') || {}).textContent === 'mace submit --track many'", 60)
    H.wait(drv, "return !document.querySelector('#submit-dialog [data-act=submit]').disabled", 30)
    drv.find_element("css selector", '#submit-dialog [data-act="cancel"]').click()
    assert _sbatch(fake) == []


# ---------------------------------------------------------------- D2 in the dialog: a refused Submit offers a new preview
def test_refused_submit_offers_update_preview(fresh):
    (drv, base, root, xdg, proc, dl), fake = fresh
    deck = root / "proj" / "decks" / "mat3_sp.d12"
    drv.execute_script("window.__mace.run('calc.submit', {paths: ['proj/decks/mat3_sp.d12']})")
    _dialog(drv)
    H.wait(drv, "return !document.querySelector('#submit-dialog [data-act=submit]').disabled", 30)
    text = deck.read_text()
    try:
        deck.write_text(text + "\n")                     # the deck changes after the preview
        drv.find_element("css selector", '#submit-dialog [data-act="submit"]').click()
        H.wait(drv, "return ((document.querySelector('#submit-dialog [data-part=error]') || {}).textContent || '').includes('preview again')", 30)
        assert drv.execute_script("return document.querySelector('#submit-dialog [data-act=submit]').disabled")
        assert not drv.execute_script("return document.querySelector('#submit-dialog [data-act=cancel]').disabled")
        drv.find_element("css selector", '#submit-dialog [data-act="update"]').click()
        _dialog(drv)
        H.wait(drv, "return !document.querySelector('#submit-dialog [data-act=submit]').disabled", 30)
        assert not drv.execute_script("return document.querySelector('#submit-dialog [data-part=error]')")
        assert _sbatch(fake) == []
        drv.find_element("css selector", '#submit-dialog [data-act="cancel"]').click()
        H.wait(drv, "return !document.querySelector('#submit-dialog[open]')", 5)
        assert _sbatch(fake) == []
        U.no_errors(drv)
    finally:
        deck.write_text(text)


# ---------------------------------------------------------------- G5/G6: the Site row and its Check button
def test_site_row_shows_resolved_values_and_check_runs_mace_site_check(fresh):
    (drv, base, root, xdg, proc, dl), fake = fresh
    _right_click(drv, '#file-list .file[data-path="proj/decks/mat1_sp.d12"]')
    _menu_rows(drv, "Explorer menu")
    _click_item(drv, "Submit…")
    _dialog(drv)
    H.wait(drv, "return !!document.querySelector('#submit-dialog [data-part=site-details]')", 15)
    shown = drv.execute_script("""return Object.fromEntries([...document.querySelectorAll('#submit-dialog [data-part=site-details] > div')]
        .map((d) => [d.dataset.site, d.querySelector('dd').textContent.replace(/\\s+/g, ' ').trim()]))""")
    # the same describe() the CLI prints with `mace site --json`
    site = _api(base, "api/submit/capabilities")[2]["site"]
    res = site["resources"]["scf"]["value"]
    assert {"resources", "mpi-launch", "scratch", "qos", "constraint", "executables"} <= set(shown), shown
    assert f"{res['ntasks']} tasks" in shown["resources"] and res["mem"] in shown["resources"]
    assert shown["mpi-launch"].startswith(site["mpi"]["launcher"])
    assert site["scratch"]["order"][0] in shown["scratch"]
    assert site["crystal_install"]["scf"] in shown["executables"]
    # Check: `mace site check --json` through the task system; sbatch --test-only only, never a job
    drv.find_element("css selector", '#submit-dialog [data-act="site-check"]').click()
    H.wait(drv, "return !!document.querySelector('#submit-dialog [data-part=site-check]')", 240)
    items = drv.execute_script("return [...document.querySelectorAll('#submit-dialog [data-part=site-check] li')]"
                               ".map((li) => [li.dataset.status, li.textContent.replace(/\\s+/g, ' ').trim()])")
    assert len(items) >= 8, items
    assert items[0][1].startswith("OK Settings:") and any(t.split(" ", 1)[1].startswith("Job script:") for _, t in items), items
    assert any("--test-only" in c["argv"] for c in fake.calls("sbatch"))      # asked SLURM, as the CLI does
    assert _sbatch(fake) == []
    assert not list(root.rglob("mace-site-check-*"))
    U.keys(drv, Keys.ESCAPE)
    H.wait(drv, "return !document.querySelector('#submit-dialog[open]')", 5)
    U.no_errors(drv)
