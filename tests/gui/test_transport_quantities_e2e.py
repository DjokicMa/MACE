"""The transport plot's Quantity choice in a real browser (1.1.6), on the example
suite's diamond TRANSPORT run (spin-polarized, written by MACE as .DAT, with
RELAXTIM from its .out):

* opening any of the run's files offers every quantity: the five files beside
  it and PF, κₑ, ZT, L, σ/τ and PF/τ derived from them, in the same document (as
  a density volume offers its run's potential); one the files do not allow is
  grayed, with the reason, and stays unchosen;
* SIGMAS and TDF draw (an open-shell SIGMAS.dat gave no rows; an open-shell
  TDF.dat drew its beta block after its alpha block on one line);
* the curves are mace.utils.transport_plot.traces's, the ones mace plotting
  --transport draws, for ZT with and without κ_L, all temperatures, a
  component, the carrier axis and the Lorenz number's L₀ line; ZT's peak is the
  database's (results.transport_zt);
* Copy CLI Command is a command mace plotting's parser takes, and Publication
  Figure runs mace plotting --transport with the plot's options.
With MACE_GUI_SHOTS the figures are saved there."""
import lzma
import shlex
import shutil
import time
from pathlib import Path

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("plotly")
from selenium.webdriver.common.by import By  # noqa: E402

SUITE = H.REPO_ROOT / "examples" / "suite" / "2-dense-3d" / "diamond" / "TRANSPORT"
STEM = "diamond_transport"


def _unpack(dst: Path, skip=()):
    dst.mkdir(parents=True)
    for f in sorted(SUITE.iterdir()):
        name = f.name[:-3] if f.name.endswith(".xz") else f.name
        if any(name.endswith(s) for s in skip):
            continue
        (dst / name).write_bytes(lzma.decompress(f.read_bytes()) if f.name.endswith(".xz") else f.read_bytes())


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    if not SUITE.is_dir():
        pytest.skip("examples/suite not present")
    tmp = tmp_path_factory.mktemp("transport_quantities")
    root = tmp / "root"
    _unpack(root / "dia")
    _unpack(root / "nokappa", skip=(".KAPPA.DAT",))
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_TEST_HOOKS="1", MACE_GUI_MAX_WORKERS="2")
    drv = None
    try:
        drv = H.firefox(1366, 820)
        H.load(drv, base)
        yield drv, root
    finally:
        if drv is not None:
            drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


def state(drv):
    return drv.execute_script("return window.__mace.plotView.state()")


def figure(drv):
    return drv.execute_script("const el = window.__mace.plotView.el(); return {names: el.data.map((t) => t.name),"
                              " x: el.data.map((t) => Array.from(t.x)), y: el.data.map((t) => Array.from(t.y)),"
                              " xtitle: el.layout.xaxis.title.text, ytitle: el.layout.yaxis.title.text, xtype: el.layout.xaxis.type}")


def buttons(drv):
    return drv.execute_script("return [...document.querySelectorAll('#inspector .po-tq button')].map((b) => ({"
                              "k: b.dataset.tquantity, label: b.textContent, checked: b.getAttribute('aria-checked') === 'true',"
                              " off: b.getAttribute('aria-disabled') === 'true', title: b.title}))")


def drawn(drv, cond="true", timeout=60):
    H.wait(drv, "const v = window.__mace.plotView; const s = v && v.state(); return s && s.kind === 'transport' && s.drawn"
                f" && s.transport && ({cond})", timeout)
    time.sleep(0.2)


def pick(drv, key):
    drv.execute_script("document.querySelector(`#inspector .po-tq [data-tquantity='${arguments[0]}']`).click()", key)
    drawn(drv, f"s.transport.quantity === '{key}'")


def set_opt(drv, patch, cond="true"):
    drv.execute_script("window.__mace.plotView.setOpt(arguments[0], 'test')", patch)
    time.sleep(0.3)
    drawn(drv, cond)


def same(a, b):
    """The plotted values are the computed ones (to the server's JSON float encoding); gaps where they are."""
    return len(a) == len(b) and all((x is None) == (y is None) and (x is None or x == pytest.approx(y, rel=1e-12))
                                    for x, y in zip(a, b))


def python_traces(path, quantity, component="avg", temperature=None, x="mu", kappa_l=None):
    from mace.utils import transport_plot as T
    d = T.quantity_data(path, quantity, kappa_l)
    return d, T.traces(d, component, temperature, x)


def test_any_file_offers_every_quantity_and_each_draws(app):
    drv, root = app
    drv.execute_script(U.RESET_JS)
    dia = root / "dia"
    assert U.open_doc(drv, f"dia/{STEM}.SIGMAS.DAT") == "plot"
    drawn(drv, "s.transport.quantity === 'SIGMAS'")
    b = buttons(drv)
    # κ<sub>0</sub>, κ<sub>e</sub> drawn as subscripts (labels.js): the buttons' text
    assert [x["label"] for x in b] == ["S", "σ", "σS", "κ0", "TDF", "PF", "κe", "ZT", "L", "σ/τ", "PF/τ"]
    assert not [x for x in b if x["off"]], b
    assert [x["k"] for x in b if x["checked"]] == ["SIGMAS"]
    f = figure(drv)
    assert "SIGMAS" in f["ytitle"] and sum(v is not None for v in f["y"][0]) > 100          # it draws (it gave no rows)

    # the TDF: alpha + beta summed point by point, so energy runs one way
    pick(drv, "TDF")
    f = figure(drv)
    assert f["xtitle"] == "Energy (eV)" and len(f["names"]) == 1
    assert all(a < b for a, b in zip(f["x"][0], f["x"][0][1:]))
    U.shot(drv, "transport_tdf")

    # ZT without κ_L: ZTₑ, labeled an upper bound; then with κ_L the total, its bound dashed, the database's peak
    pick(drv, "zt")
    _, want = python_traces(dia / f"{STEM}.SIGMAS.DAT", "zt")
    f = figure(drv)
    assert f["ytitle"] == "ZT<sub>e</sub> (κ<sub>L</sub> = 0, an upper bound)" and f["names"] == [t["name"] for t in want]
    # at the Standard k mesh (NEWK IS 48) no κₑ is below the files' precision (the SCF mesh gave 16-18 such
    # points, drawn as gaps with a note), and the mesh is not flagged as too coarse
    notes = drv.execute_script("return [...document.querySelectorAll('.po-note')].map((e) => e.textContent).join(' | ')")
    assert "below the files' precision" not in notes and "too coarse" not in notes, notes
    set_opt(drv, {"T": 300})
    U.shot(drv, "transport_zt_e")
    set_opt(drv, {"T": None})
    set_opt(drv, {"kl": "1.5"}, "s.zt && s.zt.zt_total_max")
    d, want = python_traces(dia / f"{STEM}.SIGMAS.DAT", "zt", kappa_l="1.5")
    f = figure(drv)
    assert f["names"] == [t["name"] for t in want] == ["ZT (κ<sub>L</sub> = 1.5 W m<sup>−1</sup> K<sup>−1</sup>, τ = 10 fs)",
                                                         "ZT<sub>e</sub> (κ<sub>L</sub> = 0)"]
    assert same(f["y"][0], want[0]["y"])
    # the y axis spans the total (the bound runs up to ~1e9 in the gap at 100 K), and the subtitle names ZT
    from mace.utils import transport_plot as T
    top = drv.execute_script("return window.__mace.plotView.el().layout.yaxis.range")
    assert top == pytest.approx(list(T.y_range(d, want)), rel=1e-12) and top[1] < 100
    assert "Figure of merit ZT" in drv.execute_script("return document.querySelector('.pv-note-line').textContent")
    from mace.gui import results
    z = results.transport_zt(dia / f"{STEM}.SEEBECK.DAT", "1.5", None)
    assert state(drv)["zt"]["zt_total_max"]["value"] == z["zt_total_max"]["value"] == d["zt_total_max"]["value"]
    U.shot(drv, "transport_zt")
    set_opt(drv, {"T": 300})
    U.shot(drv, "transport_zt_300K")

    # all temperatures of one component, and the carrier axis
    set_opt(drv, {"T": "all", "component": "xx"})
    _, want = python_traces(dia / f"{STEM}.SIGMAS.DAT", "zt", "xx", "all", kappa_l="1.5")
    assert figure(drv)["names"] == [t["name"] for t in want] and len(want) > 15
    set_opt(drv, {"T": 300, "component": "avg", "x": "carriers"})
    _, want = python_traces(dia / f"{STEM}.SIGMAS.DAT", "zt", "avg", 300, "carriers", kappa_l="1.5")
    f = figure(drv)
    assert f["names"] == [t["name"] for t in want] and f["xtype"] == "log" and "cm<sup>−3</sup>" in f["xtitle"]

    # the power factor in µW cm⁻¹ K⁻² (plot markup), and Copy CLI Command / Publication Figure with the plot's options
    set_opt(drv, {"x": "mu"})
    pick(drv, "pf")
    _, want = python_traces(dia / f"{STEM}.SIGMAS.DAT", "pf", "avg", 300)
    f = figure(drv)
    assert f["names"] == [t["name"] for t in want] and "µW cm<sup>−1</sup> K<sup>−2</sup>" in f["ytitle"]
    assert same(f["y"][0], want[0]["y"])
    U.shot(drv, "transport_pf")
    pick(drv, "lorenz")
    _, want = python_traces(dia / f"{STEM}.SIGMAS.DAT", "lorenz", "avg", 300)
    assert figure(drv)["names"] == [t["name"] for t in want] and want[-1]["role"] == "reference"
    U.shot(drv, "transport_lorenz")
    text = drv.execute_script("return window.__mace.plotView.cliText()")
    words = shlex.split(text)
    assert words[:4] == ["mace", "plotting", "--transport", f"{STEM}.SIGMAS.DAT"]
    from mace.plotting.main import create_parser
    args = create_parser().parse_args(words[2:])
    assert (args.quantity, args.temperature, args.x_axis) == ("lorenz", "300", "mu")
    r = drv.execute_async_script("const done = arguments[0]; window.__mace.plotView.publication().then(done, (e) => done({error: String(e)}))")
    assert "error" not in r, r
    assert "--transport" in r["command"] and "--quantity=lorenz" in r["command"] and "--temperature=300" in r["command"], r["command"]
    assert any(n.endswith(".svg") for n in r["images"]), r
    errs = [e for e in H.errors(drv) if "ResizeObserver loop" not in e]
    assert not errs, errs


def test_a_quantity_the_files_do_not_allow_is_grayed_with_the_reason(app):
    drv, root = app
    drv.execute_script(U.RESET_JS)
    assert U.open_doc(drv, f"nokappa/{STEM}.SEEBECK.DAT") == "plot"
    drawn(drv, "s.transport.quantity === 'SEEBECK'")
    b = {x["k"]: x for x in buttons(drv)}
    for k in ("KAPPA", "zt", "lorenz", "kappa_e"):
        assert b[k]["off"] and "KAPPA.dat" in b[k]["title"], b[k]
    for k in ("SEEBECK", "SIGMA", "SIGMAS", "TDF", "pf", "sigma_tau", "pf_tau"):
        assert not b[k]["off"], b[k]
    drv.execute_script("document.querySelector(\"#inspector .po-tq [data-tquantity='zt']\").click()")
    time.sleep(0.5)
    assert state(drv)["transport"]["quantity"] == "SEEBECK"
    toast = drv.execute_script("return [...document.querySelectorAll('#toasts .toast2')].map((t) => t.textContent).join(' ')")
    assert "KAPPA.dat" in toast, toast
    pick(drv, "pf_tau")
    assert "10<sup>11</sup> W m<sup>−1</sup> K<sup>−2</sup> s<sup>−1</sup>" in figure(drv)["ytitle"]
    notes = drv.execute_script("return [...document.querySelectorAll('.po-note')].map((e) => e.textContent).join(' | ')")
    assert "τ = 10 fs" in notes and "diamond_transport.out" in notes, notes
