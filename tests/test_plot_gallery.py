"""The plot gallery's shared core (mace/plotting/gallery.py) and its CLI,
``mace plotting --gallery KIND FILES/FOLDER``.

* the set: folders searched recursively (hidden folders and backups left
  out), a file of another kind brings its run's file of this kind, the cap
  and its message, the order;
* the panes: every kind's description from the repository's demo files
  (structures, SCF, optimization), the energy window, an unreadable file
  as an error in its pane;
* the captions: material id, formula, gap and energy (Ha converted to eV)
  from a materials.db beside the files, matched by the recorded output, and
  the CSV;
* the CLI: the real ``mace_cli`` as a subprocess writes the grid (SVG, PNG)
  and the caption CSV, refuses a missing input, and draws the same panes the
  module does.
The corpus adds band, DOS and band + DOS (skipped without it)."""
import csv
import io
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DEMO = REPO / "examples" / "gui"
sys.path[:0] = [p for p in (str(REPO),) if p not in sys.path]

from mace.plotting import gallery as G  # noqa: E402


def _cli(*args, cwd=REPO, timeout=600):
    return subprocess.run([sys.executable, str(REPO / "mace_cli"), "--no-banner", "plotting", *args], cwd=str(cwd),
                          capture_output=True, text=True, timeout=timeout)


@pytest.fixture()
def demo(tmp_path):
    root = tmp_path / "demo"
    shutil.copytree(DEMO, root)
    return root


# ---------------------------------------------------------------- the set

def test_collect_folder_recursive_sorted_and_skips_hidden(demo):
    (demo / ".hidden").mkdir()
    shutil.copyfile(demo / "structures" / "Si.cif", demo / ".hidden" / "Si.cif")
    (demo / "backups").mkdir()
    shutil.copyfile(demo / "structures" / "Si.cif", demo / "backups" / "Si.cif")
    got = G.collect([demo], "structure")
    names = [p.name for p in got["items"]]
    # .d12 decks and CIFs both carry a structure; hidden folders and backups do not count
    assert "Si.cif" in names and names.count("Si.cif") == 1
    assert all(".hidden" not in str(p) and "backups" not in str(p) for p in got["items"])
    cifs = [p.name for p in got["items"] if p.suffix == ".cif"]
    assert cifs == ["diamond.cif", "hBN.cif", "LiF.cif", "MgO.cif", "NaCl.cif", "Si.cif"]
    assert got["message"] is None and not got["capped"]


def test_collect_cap_and_message(tmp_path):
    for k in range(12):
        shutil.copyfile(DEMO / "structures" / "Si.cif", tmp_path / f"Si{k:02d}.cif")
    got = G.collect([tmp_path], "structure", cap=5)
    assert len(got["items"]) == 5 and got["total"] == 12 and got["capped"]
    assert got["message"].startswith("Showing the first 5 of 12 files")
    assert [p.name for p in got["items"]] == [f"Si{k:02d}.cif" for k in range(5)]


def test_collect_empty_and_bad_kind(tmp_path):
    got = G.collect([tmp_path], "ir")
    assert got["items"] == [] and got["message"] == "No ir spectrum files in this set."
    with pytest.raises(G.GalleryError):
        G.collect([tmp_path], "phonons")


def test_collect_outputs_for_scf_and_opt(demo):
    scf = [p.name for p in G.collect([demo / "outputs"], "scf")["items"]]
    assert scf == ["Ag2Br3_opt_unfinished.out", "graphene_slab_sp.out", "Si_sp.out"]
    opt = [p.name for p in G.collect([demo / "outputs"], "opt")["items"]]
    assert opt == ["Ag2Br3_opt_unfinished.out"]                       # only the optimization


def test_related_file_brings_its_runs_file(demo):
    # an .out picked for its structure is itself a structure file; a deck picked for SCF brings its output
    got = G.collect([demo / "outputs" / "Si_sp.d12"], "scf")
    assert [p.name for p in got["items"]] == ["Si_sp.out"]


def test_titles_disambiguate_shared_material_ids(tmp_path):
    a, b = tmp_path / "1_dia_opt_rev1.out", tmp_path / "1_dia_opt_BULK_OPTGEOM.out"
    c = tmp_path / "MgO.cif"
    t = G.titles([a, b, c])
    assert t[0] == a.name and t[1] == b.name and t[2] == "MgO"


# ---------------------------------------------------------------- panes

def test_structure_pane(demo):
    pytest.importorskip("ase")                                              # the CIF is read with ase
    pn = G.pane(demo / "structures" / "MgO.cif", "structure")
    assert pn["error"] is None and pn["title"] == "MgO"
    pl = pn["panels"][0]
    assert len(pl["atoms"]["x"]) == 8 and pl["equal"] and pl["cell"] and len(pl["cell"]["x"]) == 5
    assert all(c.startswith("#") for c in pl["atoms"]["color"])


def test_scf_and_opt_panes(demo):
    pn = G.pane(demo / "outputs" / "graphene_slab_sp.out", "scf")
    pl = pn["panels"][0]
    assert pn["error"] is None and pl["ylog"] and pl["xlabel"] == "SCF cycle"
    ys = pl["series"][0]["y"]
    assert len(ys) > 3 and all(y > 0 for y in ys) and ys[-1] < ys[0]       # |ΔE| falls along the SCF
    op = G.pane(demo / "outputs" / "Ag2Br3_opt_unfinished.out", "opt")
    assert op["error"] is None and op["panels"][0]["ylabel"] == "E − E(final) (meV)"


def test_unreadable_file_is_an_error_in_its_pane(tmp_path):
    bad = tmp_path / "x.BAND.DAT"
    bad.write_text("not a band file\n")
    pn = G.pane(bad, "band")
    assert pn["panels"] == [] and pn["error"]
    with pytest.raises(G.GalleryError):
        G.pane(bad, "band", (2.0, 1.0))


def test_shared_limits_union():
    panes = [{"panels": [{"series": [{"x": [0, 1], "y": [1, 5]}], "energy": None}]},
             {"panels": [{"series": [{"x": [0, 3], "y": [-2, 4]}], "energy": None}]},
             {"panels": [{"series": [{"x": [0, 1], "y": [None, 2]}], "energy": "y"}]}]
    lim = G.shared_limits(panes)
    assert lim[0]["x"] == [0, 3] and lim[0]["y"] == [-2, 5]           # the energy axis is the window's, not shared


# ---------------------------------------------------------------- captions

def _db(path: Path, rows, props, calcs=()):
    c = sqlite3.connect(str(path))
    c.executescript("CREATE TABLE materials (material_id TEXT PRIMARY KEY, formula TEXT NOT NULL);"
                    "CREATE TABLE calculations (calc_id TEXT PRIMARY KEY, material_id TEXT, output_file TEXT, input_file TEXT,"
                    " work_dir TEXT, created_at TEXT);"
                    "CREATE TABLE properties (material_id TEXT, calc_id TEXT, property_name TEXT, property_value REAL,"
                    " property_unit TEXT, extracted_at TEXT);")
    c.executemany("INSERT INTO materials VALUES (?, ?)", rows)
    c.executemany("INSERT INTO properties VALUES (?, ?, ?, ?, ?, ?)", props)
    c.executemany("INSERT INTO calculations VALUES (?, ?, ?, ?, ?, ?)", calcs)
    c.commit()
    c.close()


def test_captions_from_the_db_beside_the_files(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    shutil.copyfile(DEMO / "structures" / "MgO.cif", runs / "MgO.cif")
    (runs / "odd_name_sp.out").write_text("x\n")
    (runs / "Unknown.cif").write_text("x\n")
    _db(tmp_path / "materials.db", [("MgO", "Mg4O4"), ("NaCl_bulk", "Na4Cl4")],
        [("MgO", None, "band_gap_ev", 7.5, "eV", "2026-01-01"),
         ("MgO", None, "total_energy", -1.0, "Ha", "2026-01-01"),
         ("NaCl_bulk", "c1", "band_gap", 5.0, "eV", "2026-01-01")],
        [("c1", "NaCl_bulk", "runs/odd_name_sp.out", None, None, "2026-01-01")])
    items = [runs / "MgO.cif", runs / "odd_name_sp.out", runs / "Unknown.cif"]
    caps = G.captions(items)
    by = {r["n"]: r for r in caps["rows"]}
    assert by[1]["material_id"] == "MgO" and by[1]["formula"] == "Mg4O4" and by[1]["band_gap_eV"] == 7.5
    assert by[1]["energy_eV"] == pytest.approx(-G.HARTREE_EV, abs=1e-6)          # Ha in the database, eV in the table
    # an output a calculation records (a relative path, under the database's folder) gives that material
    assert by[2]["material_id"] == "NaCl_bulk" and by[2]["formula"] == "Na4Cl4" and by[2]["band_gap_eV"] == 5.0
    assert by[3]["formula"] is None and by[3]["band_gap_eV"] is None
    assert caps["databases"] == [str(tmp_path / "materials.db")]
    rows = list(csv.reader(io.StringIO(G.captions_csv(caps["rows"]))))
    assert rows[0] == list(G.CAPTION_COLUMNS) and rows[1][:4] == ["1", "MgO", "Mg4O4", "7.5"] and rows[3][2] == ""


def test_captions_without_a_db(tmp_path):
    f = tmp_path / "LiF.cif"
    shutil.copyfile(DEMO / "structures" / "LiF.cif", f)
    caps = G.captions([f], stop=tmp_path)
    assert caps["rows"][0]["material_id"] == "LiF" and caps["rows"][0]["formula"] is None and caps["databases"] == []


# ---------------------------------------------------------------- the figure and the CLI

def test_render_figure_formats(demo, tmp_path):
    panes = [G.pane(p, "structure") for p in G.collect([demo / "structures"], "structure")["items"][:3]]
    panes.append({"name": "bad", "title": "bad", "panels": [], "error": "unreadable"})
    out = G.render_figure(panes, tmp_path / "g.svg", cols=2)
    text = out.read_text()
    assert "1. diamond" in text and "4. bad" in text and "unreadable" in text
    assert G.render_figure(panes, tmp_path / "g.png").read_bytes()[:4] == b"\x89PNG"
    with pytest.raises(G.GalleryError):
        G.render_figure(panes, tmp_path / "g.jpg")


def test_cli_gallery_writes_grid_and_captions(demo, tmp_path):
    out = tmp_path / "grid.svg"
    csv_p = tmp_path / "caps.csv"
    p = _cli("--gallery", "structure", str(demo / "structures"), "--cols", "4", "-o", str(out), "--caption-csv", str(csv_p))
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    assert "Traceback" not in p.stdout + p.stderr
    assert out.is_file() and "6. Si" in out.read_text()
    assert "Gallery: " in p.stdout and "(6 panes, 4 columns)" in p.stdout
    lines = csv_p.read_text().splitlines()
    assert lines[0] == "n,material_id,formula,band_gap_eV,energy_eV,file" and len(lines) == 7
    assert lines[4].startswith("4,MgO,Mg4O4,")                       # examples/gui/materials.db, one folder up
    # a folder as -o: gallery_<kind>.<format>
    p = _cli("--gallery", "scf", str(demo / "outputs"), "-o", str(tmp_path), "--format", "png")
    assert p.returncode == 0, p.stderr[-2000:]
    assert (tmp_path / "gallery_scf.png").read_bytes()[:4] == b"\x89PNG"


def test_cli_gallery_refusals(tmp_path):
    p = _cli("--gallery", "structure", str(tmp_path / "nope"))
    assert p.returncode == 2 and "Not found" in p.stderr
    p = _cli("--gallery", "phonons", str(tmp_path))
    assert p.returncode == 2 and "invalid choice" in p.stderr
    p = _cli("--gallery", "ir", str(tmp_path), "-o", str(tmp_path / "x.svg"))
    assert p.returncode == 1 and "No ir spectrum files" in p.stdout and not (tmp_path / "x.svg").exists()


def test_command_round_trips_through_the_parser():
    from mace.plotting.main import create_parser
    a = G.command("bandos", ["a", "b c"], 5, "o.svg", shared_axes=False, ewin=(-3.0, 2.0), caption_csv="c.csv")
    args = create_parser().parse_args(a[1:])
    assert args.gallery == "bandos" and args.files == ["a", "b c"] and args.cols == 5 and args.output == "o.svg"
    assert args.independent_axes and (args.e_lower, args.e_upper) == (-3.0, 2.0) and args.caption_csv == "c.csv"
    assert "--e-lower=-3" in a and "--independent-axes" in a
    # the window is always spelled out (1.1.6 review: a folder's mace-folder.toml must not change what the line draws)
    assert [x for x in G.command("band", ["a"], 3, "o.svg") if x.startswith("--e-")] == ["--e-lower=-6", "--e-upper=6"]
    assert not any(x.startswith("--e-") for x in G.command("structure", ["a"], 3, "o.svg"))


# ---------------------------------------------------------------- the corpus: band, DOS, band + DOS

@pytest.fixture()
def bands(tmp_path):
    from conftest import find_data
    band = find_data("BAND/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_band.BAND.DAT")
    dos = find_data("DOSS/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_doss.DOSS.DAT")
    d = tmp_path / "bd"
    d.mkdir()
    for f in (band, band.with_name(band.name.replace(".BAND.DAT", ".d3")), dos,
              dos.with_name(dos.name.replace(".DOSS.DAT", ".d3"))):
        shutil.copyfile(f, d / f.name)
    return d


def test_band_dos_and_bandos_panes(bands):
    band = next(bands.glob("*.BAND.DAT"))
    pn = G.pane(band, "band", (-4.0, 3.0))
    pl = pn["panels"][0]
    assert pn["error"] is None and pl["energy"] == "y" and pl["ylim"] == [pl["fermi"] - 4.0, pl["fermi"] + 3.0]
    assert pl["ticklabels"][0] == "Γ" and len(pl["ticks"]) == len(pl["ticklabels"])
    assert None in pl["series"][0]["x"]                                 # the bands joined with breaks: one line object
    bd = G.pane(band, "bandos")
    assert len(bd["panels"]) == 2 and bd["panels"][1]["energy"] == "y" and bd["partner"].endswith(".DOSS.DAT")
    dos = G.pane(next(bands.glob("*.DOSS.DAT")), "dos")
    assert dos["panels"][0]["energy"] == "x" and dos["panels"][0]["series"][0]["name"] == "Total"
    # a DOSS file picked for band + DOS brings its band file
    assert [p.name for p in G.collect([next(bands.glob("*.DOSS.DAT"))], "bandos")["items"]] == [band.name]


def test_cli_bandos_png(bands, tmp_path):
    p = _cli("--gallery", "bandos", str(bands), "--e-lower", "-5", "--e-upper", "5", "-o", str(tmp_path / "b.png"))
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    assert (tmp_path / "b.png").read_bytes()[:4] == b"\x89PNG"


# ---------------------------------------------------------------- the spin look and the DOS plot's rule (queue 14)

SPIN_FE = REPO / "tests" / "data" / "spin_fe"


def test_command_names_the_spin_look():
    from mace.plotting.main import create_parser
    a = G.command("bandos", ["spin_fe"], 3, "o.svg", spin_colors="custom", spin_up_color="#AA0011",
                  spin_down_color="#0011aa", spin_down_style="dot")
    args = create_parser().parse_args(a[1:])
    assert (args.spin_colors, args.spin_up_color, args.spin_down_color, args.spin_style) == \
        ("custom", "#aa0011", "#0011aa", "dot")
    plain = G.command("bandos", ["spin_fe"], 3, "o.svg")
    assert not any(x.startswith("--spin") for x in plain), plain
    assert not any(x.startswith("--spin") for x in G.command("structure", ["a"], 3, "o.svg", spin_colors="blue-red"))
    bc = G.command("bandos", ["dia"], 3, "o.svg", band_color="#2E8B57")
    assert create_parser().parse_args(bc[1:]).band_color == "#2e8b57"
    assert not any(x.startswith("--band-color") for x in plain)


def test_gallery_dos_follows_the_dos_plot_rule_and_spin_look():
    """A spin DOS pane: both spins drawn, so every projection is solid (spin down in spin down's line
    style) and an element's orbitals are ranked by their weight in the window, as the DOS plot; the
    band panel takes the spin scheme. The CLI gallery used the closed-shell rule and the style's pair."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex
    from mace.gui import results
    from mace.plotting import presets as P
    pn = G.pane(SPIN_FE / "fe_band.BAND.DAT", "bandos", (-6.0, 6.0))
    assert pn["error"] is None, pn["error"]
    band, dos = pn["panels"]
    assert dos["dos_spin"] is True and dos["dos_weights"]
    d = results.dos_data(SPIN_FE / "fe_doss.DOSS.DAT")
    f = dos["fermi"]
    want = P.dos_weights(d["energy"], d["projections"], f - 6.0, f + 6.0)
    assert dos["dos_weights"] == pytest.approx(want, abs=1e-6)
    look = {"spin_colors": "blue-red", "spin_down_style": "dot"}
    pair = P.spin_colors("default", "band", scheme="blue-red")
    fig, (ax1, ax2) = plt.subplots(1, 2)
    drawn = {ax1: [], ax2: []}
    for ax in (ax1, ax2):
        def plot(*a, _ax=ax, _orig=ax.plot, **k):
            drawn[_ax].append((to_hex(k["color"]), k["linestyle"]))
            return _orig(*a, **k)
        ax.plot = plot
    dot, solid = P.mpl_dash("dot"), P.mpl_dash("solid")
    try:
        G._draw_panel(ax1, band, None, False, "default", look)
        G._draw_panel(ax2, dos, None, False, "default", look)
        for (color, ls), s in zip(drawn[ax1], band["series"]):
            if s.get("role") in ("up", "down"):
                assert color == pair[s["role"]]
                assert ls == (dot if s["role"] == "down" else solid), s
        for (color, ls), s in zip(drawn[ax2], dos["series"]):
            if s["role"] == "projection":
                assert ls == (solid if s["spin"] == "up" else dot), s["name"]
    finally:
        plt.close(fig)


def test_cli_gallery_takes_the_spin_flags(tmp_path):
    d = tmp_path / "spin_fe"
    shutil.copytree(SPIN_FE, d)
    p = _cli("--gallery", "bandos", str(d), "--spin-colors", "blue-red", "--spin-down-style", "dot",
             "-o", str(tmp_path / "g.svg"))
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    svg = (tmp_path / "g.svg").read_text()
    from mace.plotting import presets as P
    pair = P.spin_colors("default", "band", scheme="blue-red")
    assert pair["up"] in svg.lower() and pair["down"] in svg.lower()


# ---------------------------------------------------------------- the 1.1.6 DOS layout in the gallery

def test_gallery_dos_panel_carries_the_dos_layout():
    """A spin DOS pane: mirrored, the 2-4 projections the DOS plot draws by default, the largest |DOS|
    for a symmetric axis, E_F and the DOS = 0 baseline; a closed-shell one one-sided."""
    from mace.plotting import presets as P
    from mace.gui import results
    pn = G.pane(SPIN_FE / "fe_doss.DOSS.DAT", "dos", (-6.0, 6.0))
    pl = pn["panels"][0]
    assert pl["dos_side"] == "mirrored" and pl["eflines"] == [pl["fermi"]] and pl["baseline"] == 0.0
    d = results.dos_data(SPIN_FE / "fe_doss.DOSS.DAT")
    look = P.dos_styles([p["label"] for p in d["projections"]], "auto", show_total=True, spin=True,
                        weights=pl["dos_weights"])
    pk = P.pick_projections(look["lines"], pl["dos_weights"])
    shown = [s["proj"] for s in pl["series"] if s["role"] == "projection" and s["spin"] == "up"]
    assert shown == pk["shown"] and 2 <= len(shown) <= 4 and pl["dos_order"] == pk["order"]
    downs = [v for s in pl["series"] if s["spin"] == "down" for v in s["y"] if v is not None]
    assert min(downs) < 0                                                     # spin down: the mirrored half
    assert G.dos_axis(pl) == pytest.approx([-pl["dos_max"] * 1.08, pl["dos_max"] * 1.08])
    assert G.shared_limits([pn])[0]["y"] == [0.0, pl["dos_max"]]
    assert G.dos_axis(pl, [0.0, 10.0]) == pytest.approx([-10.8, 10.8])
    assert G.dos_axis({**pl, "dos_side": "one"}) == pytest.approx([0.0, pl["dos_max"] * 1.08])


def test_gallery_dos_smoothing_and_no_data(bands):
    from mace.plotting import presets as P
    dos = next(bands.glob("*.DOSS.DAT"))
    plain = G.pane(dos, "dos", (-30.0, 30.0))["panels"][0]
    assert plain["dos_side"] == "one" and plain["dos_smooth"] == 0
    assert plain["nodata"], "a ±30 eV window reaches past the DOSS run's energies"
    sm = G.pane(dos, "dos", (-30.0, 30.0), dos_smooth=0.05)["panels"][0]
    assert sm["dos_smooth"] == 0.05 and sm["series"][0]["y"] != plain["series"][0]["y"]
    a = G.command("dos", ["x"], 3, "o.svg", dos_smooth=0.05)
    assert "--dos-smooth=0.05" in a and not any(x.startswith("--dos-smooth") for x in G.command("dos", ["x"], 3, "o.svg"))
    from mace.plotting.main import create_parser
    assert create_parser().parse_args(a[1:]).dos_smooth == 0.05
    assert P.smooth_sigma(sm["dos_smooth"]) == 0.05


def test_cli_gallery_draws_the_mirrored_dos(tmp_path):
    """The matplotlib gallery: |DOS| tick labels on a symmetric axis, the up/down labels, the spin fills."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mace.plotting import presets as P
    pl = G.pane(SPIN_FE / "fe_doss.DOSS.DAT", "dos", (-6.0, 6.0))["panels"][0]
    fig, ax = plt.subplots()
    try:
        G._draw_panel(ax, pl, None, False, "default", {"spin_colors": "blue-red"})
        lo, hi = ax.get_ylim()
        assert lo == pytest.approx(-hi)
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert labels and not any(t.startswith(("-", "−")) for t in labels)
        words = [t.get_text() for t in ax.texts]
        assert "↑ up" in words and "down ↓" in words
        pair = P.spin_colors("default", "band", scheme="blue-red")
        fills = {G._mpl_rgba(P.spin_fills(pair, "default")[k]) for k in ("up", "down")}
        got = {tuple(round(c, 3) for c in col.get_facecolor()[0]) for col in ax.collections}
        assert {tuple(round(c, 3) for c in f) for f in fills} <= got
    finally:
        plt.close(fig)


def test_closed_shell_band_pane_is_neutral_and_the_legend_is_grouped(bands, tmp_path):
    """One set of bands in the neutral band color, not spin up's; render_figure(legend=True) gives the
    two titled groups (Spin with the closed-shell key, Projections)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex
    from mace.plotting import presets as P
    pn = G.pane(next(bands.glob("*.BAND.DAT")), "bandos", (-6.0, 6.0))
    band, dos = pn["panels"]
    assert band["closed"] is True and dos["dos_side"] == "one"
    fig, ax = plt.subplots()
    try:
        info = G._draw_panel(ax, band, None, False, "default", {"spin_colors": "blue-red"})
        assert {to_hex(ln.get_color()) for ln in ax.get_lines() if len(ln.get_xdata()) > 2} == {info["bands"]}
        assert info["bands"] != P.spin_colors("default", "band", scheme="blue-red")["up"]
    finally:
        plt.close(fig)
    seen = {}
    real = G._legend

    def spy(fig, ax, infos, band_dos, small):
        real(fig, ax, infos, band_dos, small)
        seen["texts"] = [t.get_text() for t in ax.get_legend().get_texts()]
    G._legend = spy
    try:
        G.render_figure([pn], tmp_path / "g.png", cols=1, legend=True)
    finally:
        G._legend = real
    t = seen["texts"]
    assert t[:2] == ["Spin", "α = β (closed shell)"] and "Projections" in t and "Total" in t, t


def test_closed_shell_panes_take_the_band_color(bands):
    """--band-color (settings' band color) colors a closed-shell run's bands and its DOS total line;
    without one the preset's band color (or ink) is kept."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex
    from mace.plotting import presets as P
    band, dos = G.pane(next(bands.glob("*.BAND.DAT")), "bandos", (-6.0, 6.0))["panels"]
    for style in ("default", "publication"):
        fig, (a1, a2) = plt.subplots(1, 2)
        try:
            info = G._draw_panel(a1, band, None, False, style, {"band_color": "#2E8B57"})
            assert info["bands"] == "#2e8b57"
            assert {to_hex(ln.get_color()) for ln in a1.get_lines() if len(ln.get_xdata()) > 2} == {"#2e8b57"}
            G._draw_panel(a2, dos, None, False, style, {"band_color": "#2E8B57"})
            assert "#2e8b57" in {to_hex(ln.get_color()) for ln in a2.get_lines()}
            plain = G._draw_panel(a1, band, None, False, style, {"band_color": None})
            ink = "#000000" if style == "publication" else "#222222"
            assert plain["bands"] == P.band_color(style, "band", "light", None, ink)
        finally:
            plt.close(fig)


def test_matplotlib_band_dos_figure_has_the_grouped_legend_and_band_color(bands, tmp_path, monkeypatch):
    """mace plotting --band-dos with matplotlib: the gallery's band + DOS pane with its grouped legend,
    and --band-color reaching the closed-shell bands."""
    from mace.plotting.handlers import legacy
    seen = {}
    real = G.render_figure

    def spy(panes, out, **kw):
        seen.update(kw)
        return real(panes, out, **kw)
    monkeypatch.setattr(G, "render_figure", spy)
    config = {**legacy.configure_band_dos_plot(False), "backend": "matplotlib", "formats": ["png"],
              "band_color": "#2e8b57"}
    made = legacy.plot_band_dos([str(next(bands.glob("*.BAND.DAT")))], config, str(tmp_path / "o"))
    assert made and Path(made[0]).read_bytes()[:4] == b"\x89PNG"
    assert seen["legend"] is True and seen["spin_look"]["band_color"] == "#2e8b57"


def test_command_explicit_names_every_default():
    """Export with a mace-folder.toml: the child reads the file, so every value
    the grid draws with is passed, the defaults too."""
    a = G.command("bandos", ["x"], 3, "o.svg", explicit=True)
    for flag in ("--style", "--dos-colors", "--dos-smooth=0", "--spin-colors=auto", "--spin-down-style=dash",
                 "--fermi-shift=0", "--energy-reference=auto"):
        assert any(x == flag or x.startswith(flag + "=") or x == flag.split("=")[0] for x in a), (flag, a)
    assert "default" in a and "auto" in a
    plain = G.command("bandos", ["x"], 3, "o.svg")
    assert not any(x.startswith(("--style", "--fermi-shift", "--energy-reference")) for x in plain)
    assert "--fermi-shift=0.3" in G.command("band", ["x"], 3, "o.svg", shift=0.3)


def test_gui_gallery_pane_applies_the_fermi_shift(bands):
    """The grid draws with Settings' Fermi shift, as Export and Copy CLI Command do."""
    from mace.gui import gallery as GG

    class St:
        def resolve(self, p):
            return Path(p)

        def rel(self, p):
            return Path(p).name

    band = next(bands.glob("*.BAND.DAT"))
    q0 = {"kind": "band", "path": str(band), "emin": "-4", "emax": "3"}
    p0 = GG.pane(St(), q0.get)["panels"][0]
    p1 = GG.pane(St(), {**q0, "shift": "0.5"}.get)["panels"][0]
    assert p1 == G.pane(band, "band", (-4.0, 3.0), 0.5)["panels"][0]
    assert p1 != p0
    with pytest.raises(ValueError):
        GG.pane(St(), {**q0, "shift": "x"}.get)
