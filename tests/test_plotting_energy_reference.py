"""The band/DOS energy zero of ``mace plotting`` (1.1.6): ``--energy-reference
{auto,fermi,vacuum}`` and ``--fermi-shift EV``, and the ±6 eV default window
(the GUI's).

* the real ``mace_cli`` as a subprocess, on synthetic BAND.DAT / DOSS.DAT /
  POTC files (the writers of test_gui_results and test_ipdos_readers):
  without the new options the figure is the one before them, the shift and
  the reference name themselves in the axis title and the legend, "fermi"
  takes a POTC run off the vacuum scale, "vacuum" without POTC files is an
  error, and no --e-lower/--e-upper draws -6 to 6 eV;
* the plotters' own arithmetic: the shift moves the data, not the dashed
  line or the window;
* the gallery's band/DOS/band + DOS panes and its CLI line take the same
  options;
* the GUI's plot route (mace/gui/plots.py): the shift and reference it is
  given, in the command it shows and runs, and the POTC files it copies."""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [p for p in (str(REPO), str(REPO / "tests"), str(REPO / "Plotting")) if p not in sys.path]

from test_gui_results import write_band  # noqa: E402
from test_ipdos_readers import write_doss  # noqa: E402

BAND, DOS = "mgo_band.BAND.DAT", "mgo_doss.DOSS.DAT"


def _cli(*args, cwd):
    return subprocess.run([sys.executable, str(REPO / "mace_cli"), "--no-banner", "plotting", *args], cwd=str(cwd),
                          capture_output=True, text=True, timeout=600, env=_env())


def _env():
    import os
    # these check the matplotlib plotters' figures (tests/test_plotting_backends.py: plotly's)
    return {**os.environ, "MACE_FOLDER_SETTINGS": "none", "MACE_PLOT_BACKEND": "matplotlib"}


def _run(tmp_path, potc=False, spin=False):
    write_band(tmp_path, spin=spin)
    write_doss(tmp_path, material="mgo")
    if potc:
        (tmp_path / "mgo_POTC.POTC.DAT").write_text("# z V\n0.0 0.1\n1.0 0.1\n")
        (tmp_path / "mgo_POTC.out").write_text(" FERMI ENERGY      -0.2\n")
    return tmp_path


def _svg(tmp_path, kind, *args, rc=0):
    """The plot's SVG with matplotlib's run-to-run noise (date, clip/marker ids) taken out."""
    out = tmp_path / ("mgo.BANDS.svg" if kind == "--band" else "mgo.DOSS.svg")
    out.unlink(missing_ok=True)
    p = _cli(kind, BAND if kind == "--band" else DOS, "--format", "svg", *args, cwd=tmp_path)
    assert p.returncode == rc, p.stdout[-2000:] + p.stderr[-2000:]
    if not out.exists():
        return None, p.stdout + p.stderr
    t = re.sub(r"<dc:date>.*?</dc:date>", "", out.read_text())
    ids = {}
    t = re.sub(r"\b[pm][0-9a-f]{10}\b", lambda m: ids.setdefault(m.group(0), f"ID{len(ids)}"), t)
    return t, p.stdout + p.stderr


@pytest.mark.parametrize("kind", ["--band", "--dos"])
def test_no_new_options_draw_what_the_defaults_draw(tmp_path, kind):
    _run(tmp_path)
    plain, _ = _svg(tmp_path, kind)
    assert "E - E_F" in plain and "E_{vac}" not in plain
    explicit, _ = _svg(tmp_path, kind, "--fermi-shift", "0", "--energy-reference", "auto")
    assert explicit == plain
    # the default window is -6 to 6 eV (as the GUI's), no longer -8 to 8
    six, _ = _svg(tmp_path, kind, "--e-lower", "-6", "--e-upper", "6")
    eight, _ = _svg(tmp_path, kind, "--e-lower", "-8", "--e-upper", "8")
    assert six == plain and eight != plain


@pytest.mark.parametrize("kind", ["--band", "--dos"])
def test_a_fermi_shift_names_itself(tmp_path, kind):
    _run(tmp_path)
    plain, _ = _svg(tmp_path, kind)
    shifted, _ = _svg(tmp_path, kind, "--fermi-shift", "0.5")
    assert shifted != plain
    assert r"$E - (E_F+0.5)$ (eV)" in shifted          # the axis title (matplotlib keeps it as a comment)
    assert "$E_F+0.5$ eV" in shifted                            # the dashed line's legend entry
    down, _ = _svg(tmp_path, kind, "--fermi-shift=-0.25")
    assert r"$E - (E_F-0.25)$ (eV)" in down


@pytest.mark.parametrize("kind", ["--band", "--dos"])
def test_the_reference_with_a_potc_run(tmp_path, kind):
    plain_dir = tmp_path / "plain"
    plain_dir.mkdir()
    no_potc, _ = _svg(_run(plain_dir), kind)
    potc_dir = tmp_path / "potc"
    potc_dir.mkdir()
    _run(potc_dir, potc=True)
    auto, _ = _svg(potc_dir, kind)
    assert "E - E_{vac}" in auto                     # auto: the vacuum scale, as before
    assert _svg(potc_dir, kind, "--energy-reference", "vacuum")[0] == auto
    # fermi: off the vacuum scale, the same figure as without the POTC run
    assert _svg(potc_dir, kind, "--energy-reference", "fermi")[0] == no_potc
    shifted, _ = _svg(potc_dir, kind, "--fermi-shift", "0.5")
    assert r"$E - (E_{vac}+0.5)$ (eV)" in shifted


@pytest.mark.parametrize("kind", ["--band", "--dos"])
def test_vacuum_without_potc_files_is_an_error(tmp_path, kind):
    _run(tmp_path)
    svg, log = _svg(tmp_path, kind, "--energy-reference", "vacuum", rc=1)    # the run fails (F4-3)
    assert svg is None
    assert "--energy-reference vacuum needs the run's POTC files" in log and "Traceback" not in log


def test_bad_reference_is_refused_by_the_parser(tmp_path):
    p = _cli("--band", "--energy-reference", "vac", cwd=tmp_path)
    assert p.returncode == 2 and "invalid choice" in p.stderr


def test_help_names_the_options():
    p = subprocess.run([sys.executable, str(REPO / "mace_cli"), "--no-banner", "plotting", "--help"],
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 0
    flat = " ".join(p.stdout.split())
    assert "--energy-reference {auto,fermi,vacuum}" in flat and "--fermi-shift EV" in flat
    assert "(default: -6, as the GUI)" in flat and "(default: 6, as the GUI)" in flat
    # and argparse's own help (mace plotting's parser) says the same
    from mace.plotting.main import create_parser
    own = " ".join(create_parser().format_help().split())
    assert "--energy-reference {auto,fermi,vacuum}" in own and "(default: -6, as the GUI)" in own


# ---------------------------------------------------------------- the plotters' arithmetic

def test_the_band_shift_moves_the_bands_only():
    import ipBANDS_V2 as ipb
    a = np.array([[1.0, 2.0]])
    up, closed = ipb.apply_fermi_shift(a, np.zeros_like(a), 0.5)
    assert up.tolist() == [[0.5, 1.5]] and not closed.any()          # a closed shell's zero beta block stays zero
    up, down = ipb.apply_fermi_shift(a, a + 0.1, 0.5)
    assert down.ravel().tolist() == pytest.approx([0.6, 1.6])
    assert ipb.apply_fermi_shift(a, a, 0.0)[0] is a
    assert ipb.energy_axis_label(False) == r"$E - E_F$ (eV)"   # 1.1.6: plain, as the GUI
    assert ipb.energy_axis_label(True) == r"$E - E_{vac}$ (eV)"
    assert ipb.fermi_legend_label(0) == "$E_F$" and ipb.fermi_legend_label(-1.5) == "$E_F-1.5$ eV"


def test_the_band_reference_choices(tmp_path):
    import ipBANDS_V2 as ipb
    dat, out = tmp_path / "m_POTC.POTC.DAT", tmp_path / "m_POTC.out"
    assert ipb.energy_reference("auto", None, None) == (None, None, False)
    with pytest.raises(ValueError, match="POTC"):
        ipb.energy_reference("vacuum", str(dat), str(out))
    dat.write_text("0 0.1\n")
    out.write_text(" FERMI ENERGY -0.2\n")
    assert ipb.energy_reference("vacuum", str(dat), str(out)) == (str(dat), str(out), True)
    assert ipb.energy_reference("auto", str(dat), str(out)) == (str(dat), str(out), True)
    assert ipb.energy_reference("fermi", str(dat), str(out)) == (None, None, False)


# ---------------------------------------------------------------- the gallery

def test_gallery_panes_take_the_shift_and_reference(tmp_path):
    from mace.plotting import gallery as G
    _run(tmp_path, potc=True)
    band = tmp_path / BAND
    base = G.pane(band, "bandos")
    assert base["error"] is None and base["panels"][0]["ylabel"] == "E vs vacuum (eV)"
    f = base["panels"][0]["fermi"]
    assert abs(f) > 1
    moved = G.pane(band, "bandos", shift=0.5)
    b0, b1 = base["panels"][0], moved["panels"][0]
    assert b1["fermi"] == f and b1["ylim"] == b0["ylim"]                 # the line and the window stay
    y0 = [v for v in b0["series"][0]["y"] if v is not None]
    y1 = [v for v in b1["series"][0]["y"] if v is not None]
    assert y1 == pytest.approx([v - 0.5 for v in y0], abs=1e-4)          # the data moves
    assert b1["ylabel"] == "E − (E<sub>vac</sub> +0.5) (eV)"
    # the DOS panel: its energies are the reader's less the shift (the kept range follows the window)
    from mace.gui import results
    energies = np.array(results.dos_data(tmp_path / DOS)["energy"]) - 0.5
    d1 = moved["panels"][1]
    assert d1["fermi"] == f
    assert all(np.abs(energies - v).min() < 1e-4 for v in d1["series"][0]["y"])
    fermi = G.pane(band, "band", reference="fermi")["panels"][0]
    assert fermi["fermi"] == 0 and fermi["ylabel"] == "E (eV)" and fermi["ylim"] == list(G.EWINDOW)
    plain = tmp_path / "plain"
    plain.mkdir()
    _run(plain)
    assert "POTC" in G.pane(plain / BAND, "band", reference="vacuum")["error"]
    with pytest.raises(G.GalleryError):
        G.pane(band, "band", reference="vac")


def test_gallery_command_round_trips_the_options():
    from mace.plotting import gallery as G
    from mace.plotting.main import create_parser
    assert G.EWINDOW == (-6.0, 6.0)
    a = G.command("bandos", ["a"], 3, "o.svg", shift=0.25, reference="fermi")
    assert "--fermi-shift=0.25" in a and "--energy-reference=fermi" in a
    args = create_parser().parse_args(a[1:])
    assert args.fermi_shift == 0.25 and args.energy_reference == "fermi"
    assert not any("fermi" in x or "reference" in x for x in G.command("band", ["a"], 3, "o.svg"))


def test_gallery_cli_takes_the_shift(tmp_path):
    _run(tmp_path)
    for name, extra in (("a.svg", []), ("b.svg", ["--fermi-shift", "0.5"])):
        p = _cli("--gallery", "band", str(tmp_path / BAND), "-o", str(tmp_path / name), *extra, cwd=tmp_path)
        assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    assert (tmp_path / "a.svg").read_text() != (tmp_path / "b.svg").read_text()


# ---------------------------------------------------------------- the GUI's plot route

def test_gui_plot_route_reads_the_shift_and_reference():
    from mace.gui import plots
    q = {"shift": "0.25", "reference": "vacuum"}
    assert plots.energy_reference(q.get) == (0.25, "vacuum")
    assert plots.energy_reference({}.get) == (0.0, "auto")
    for bad in ({"shift": "x"}, {"shift": "nan"}, {"shift": "51"}, {"reference": "vac"}):
        with pytest.raises(ValueError):
            plots.energy_reference(bad.get)
    cmd = plots.command(Path("x.BAND.DAT"), "band", "svg", None, (-6, 6), 0.25, "vacuum")
    assert cmd == "mace plotting --band x.BAND.DAT --format svg --e-lower=-6 --e-upper=6 --fermi-shift=0.25 --energy-reference=vacuum"
    # the defaults add nothing (the commands of before), and the spectra take none
    assert plots.command(Path("x.BAND.DAT"), "band", "", None, (-1, 1)) == "mace plotting --band x.BAND.DAT --e-lower=-1 --e-upper=1"
    assert plots.command(Path("x_IRSPEC.DAT"), "ir", "", None, None, 0.5, "fermi") == "mace plotting --ir x_IRSPEC.DAT"


def test_gui_plot_route_copies_the_potc_files(tmp_path):
    from mace.gui import plots
    _run(tmp_path)
    assert plots._potc_files(tmp_path / BAND, "band") == []
    (tmp_path / "mgo_POTC.POTC.DAT").write_text("0 0.1\n")
    assert plots._potc_files(tmp_path / BAND, "band") == []                 # both or none, as the readers
    (tmp_path / "mgo_POTC.out").write_text(" FERMI ENERGY -0.2\n")
    names = [p.name for p in plots._potc_files(tmp_path / BAND, "band")]
    assert names == ["mgo_POTC.POTC.DAT", "mgo_POTC.out"]
    assert [p.name for p in plots._potc_files(tmp_path / DOS, "dos")] == names
    assert plots._potc_files(tmp_path / BAND, "ir") == []
