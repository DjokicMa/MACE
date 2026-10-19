"""The GUI's Copy CLI Command and Publication Figure name the plot's look the way mace plotting reads it.

* js/docs/clicmd.js (node) writes `mace plotting --band|--dos|--band-dos FILE` with the window, Fermi
  shift, energy zero, unit and style flags; mace plotting's own parser takes every flag, and the look it
  reads back (bandos_cli.look_of_config) is the look the GUI drew;
* an edited k-path is written to band_path.json first;
* plots.look_args (the Publication Figure task) takes the presets.STYLE_OPTIONS names and writes
  exactly presets.style_flags; plots.KINDS has the band + DOS figure (--band-dos).
"""
import json
import shlex
import shutil
import subprocess

import pytest

from conftest import REPO_ROOT
from mace.gui import plots
from mace.plotting import presets as P

JS = REPO_ROOT / "mace" / "gui" / "static" / "js"
node = pytest.mark.skipif(not shutil.which("node"), reason="node not installed")

LOOKS = [
    {},
    {"style": "publication"},
    {"style": "presentation", "spin_colors": "blue-red", "spin_down_style": "solid"},
    {"spin_colors": "custom", "spin_up_color": "#AA0011", "spin_down_color": "#0011aa"},
    {"spin_up_color": "#123456"},
    {"spin_down_style": "dot", "dos_colors": "element-orbital", "dos_spin": "overlaid", "dos_total": False},
    {"style": "grayscale", "spin_colors": "black-vermilion", "dos_colors": "total", "dos_spin": "mirrored"},
]
CASES = [dict(kind="bandos", file="fe_band.BAND.DAT", emin=-6, emax=6, look=lk) for lk in LOOKS] + [
    dict(kind="band", file="my run_band.BAND.DAT", emin=-3.5, emax=2, shift=0.25, unit="meV", look={"style": "colorblind"}),
    dict(kind="dos", file="fe_doss.DOSS.DAT", emin=-8, emax=4, reference="vacuum", unit="kJ/mol", look={"dos_total": False}),
    dict(kind="band", file="fe_band.BAND.DAT", emin=-6, emax=6, look={},
         kpath=[{"from": "GAMMA", "to": "H"}, {"from": "P", "to": "N"}]),
]


def _node(cases, tmp_path):
    script = tmp_path / "run.mjs"
    script.write_text(f"import {{ cliCommand }} from {json.dumps((JS / 'docs' / 'clicmd.js').as_uri())};\n"
                      f"const cases = {json.dumps(cases)};\n"
                      "console.log(JSON.stringify(cases.map((c) => cliCommand(c))));\n")
    r = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


@node
def test_copied_command_is_what_mace_plotting_reads(tmp_path):
    from mace.plotting.main import create_parser
    from mace.plotting.handlers.legacy import _band_config_from_args
    from mace.plotting.bandos_cli import look_of_config
    for case, text in zip(CASES, _node(CASES, tmp_path)):
        lines = text.split("\n")
        if case.get("kpath"):
            assert lines[0] == "cat > band_path.json <<'EOF'" and lines[2] == "EOF"
            assert json.loads(lines[1]) == {"band_path": case["kpath"]}
        words = shlex.split(lines[-1])
        assert words[:2] == ["mace", "plotting"]
        mode = {"band": "--band", "dos": "--dos", "bandos": "--band-dos"}[case["kind"]]
        assert words[2] == mode and words[3] == case["file"]
        args = create_parser().parse_args(words[2:])
        assert (args.e_lower, args.e_upper) == (case["emin"], case["emax"])
        assert args.fermi_shift == case.get("shift", 0)
        assert (args.energy_reference == "vacuum") == (case.get("reference") == "vacuum")
        assert (args.units or "eV") == case.get("unit", "eV")
        assert bool(args.band_path) == bool(case.get("kpath"))
        want = {k: v.lower() if k.endswith("_color") else v for k, v in {**P.STYLE_DEFAULTS, **case["look"]}.items()}
        assert look_of_config(_band_config_from_args(args)) == want, text


FOLDER_TOML = """[units]
energy = "meV"

[plots]
energy_window = [-2.0, 1.0]
fermi_shift = 0.5
energy_reference = "vacuum"
style = "publication"
spin_colors = "blue-red"
spin_down_style = "dot"
dos_colors = "element"
dos_spin = "overlaid"
dos_total = false
"""


@node
def test_copied_command_beats_a_folder_file(tmp_path, monkeypatch):
    """With a mace-folder.toml (explicit), the copied command names the values equal to the defaults
    too, so the folder file's [plots] and [units] values do not replace the figure on screen."""
    import importlib
    M = importlib.import_module("mace.plotting.main")
    from mace.plotting.handlers.legacy import _band_config_from_args
    from mace.plotting.bandos_cli import look_of_config
    from mace.utils import folder_settings as FS
    (tmp_path / "mace-folder.toml").write_text(FOLDER_TOML)
    monkeypatch.setenv(FS.ENV, str(tmp_path / "mace-folder.toml"))
    cases = [dict(c, explicit=True) for c in CASES if c["kind"] in ("band", "dos", "bandos")]
    cases.append(dict(kind="bandos", file="fe_band.BAND.DAT", emin=-6, emax=6, look={"dos_total": True}, explicit=True))
    for case, text in zip(cases, _node(cases, tmp_path)):
        words = shlex.split(text.split("\n")[-1])
        parser = M.create_parser()
        assert M._folder_defaults(parser, words[2:]), "the folder file was not read"
        args = parser.parse_args(words[2:])
        assert (args.e_lower, args.e_upper) == (case["emin"], case["emax"]), text
        assert args.fermi_shift == case.get("shift", 0), text
        assert args.energy_reference == case.get("reference", "fermi"), text
        assert args.units == case.get("unit", "eV"), text
        want = {k: v.lower() if k.endswith("_color") else v for k, v in {**P.STYLE_DEFAULTS, **case["look"]}.items()}
        assert look_of_config(_band_config_from_args(args)) == want, text
    # without a folder file the command stays short: the defaults are left out
    plain = _node([dict(kind="bandos", file="x.BAND.DAT", emin=-6, emax=6)], tmp_path)[0]
    assert plain == "mace plotting --band-dos x.BAND.DAT --e-lower=-6 --e-upper=6", plain


def test_explicit_style_flags_name_every_default():
    assert P.style_flags({}) == []
    assert P.style_flags({}, explicit=True) == ["--style=default", "--spin-colors=auto", "--band-style=auto", "--spin-down-style=dash",
                                                "--dos-colors=auto", "--dos-spin=auto", "--dos-total", "--dos-smooth=0"]


def test_publication_look_args_are_style_flags():
    for look in LOOKS:
        q = {k: ("0" if v is False else v) for k, v in look.items()}
        assert plots.look_args("bandos", q.get) == P.style_flags(look)
        assert plots.look_args("dos", q.get) == P.style_flags(look)
    # a spectrum: the plot style only
    assert plots.look_args("ir", {"style": "publication", "spin_colors": "mace"}.get) == ["--style=publication"]
    # the unit of a band/DOS plot; the preview page's old names still read
    assert plots.look_args("band", {"unit": "meV"}.get) == ["--units=meV"]
    assert plots.look_args("band", {"spin_style": "dot", "spin_up": "#000000"}.get) == \
        ["--spin-up-color=#000000", "--spin-down-style=dot"]
    for bad in ({"style": "nope"}, {"spin_colors": "rainbow"}, {"spin_up_color": "red"}, {"unit": "K"}, {"dos_spin": "x"}):
        with pytest.raises(ValueError):
            plots.look_args("band", bad.get)


def test_band_dos_is_a_publication_kind():
    assert plots.KINDS["bandos"][0] == "--band-dos" and "bandos" in plots.ENERGY_KINDS
    cmd = plots.command(REPO_ROOT / "tests" / "data" / "spin_fe" / "fe_band.BAND.DAT", "bandos", "svg", None, (-6, 6))
    assert cmd.startswith("mace plotting --band-dos fe_band.BAND.DAT --format svg --e-lower=-6 --e-upper=6")


def test_publication_figure_is_explicit_with_a_folder_file():
    """The Publication Figure's child reads the served folder's mace-folder.toml:
    with one, every screen value is passed, the defaults too (as Copy CLI
    Command), so the file's [plots] values cannot replace them."""
    from pathlib import Path
    assert plots.look_args("bandos", {}.get, explicit=True) == P.style_flags({}, explicit=True) + ["--units=eV"]
    assert plots.look_args("ir", {}.get, explicit=True) == ["--style=default"]
    assert plots.look_args("bandos", {}.get) == []
    cmd = plots.command(Path("x.BAND.DAT"), "bandos", "svg", None, (-6, 6), 0.0, "auto",
                        plots.look_args("bandos", {"dos_total": "1"}.get, explicit=True), explicit=True)
    for flag in ("--fermi-shift=0", "--energy-reference=auto", "--style=default", "--dos-total",
                 "--spin-colors=auto", "--units=eV"):
        assert flag in cmd.split(), cmd
    assert plots.command(Path("x.BAND.DAT"), "band", "", None, (-1, 1), explicit=False) == \
        "mace plotting --band x.BAND.DAT --e-lower=-1 --e-upper=1"


def test_publication_figure_route_is_explicit_only_with_a_folder_file(tmp_path, monkeypatch):
    from mace.gui import folderprefs
    monkeypatch.delenv("MACE_FOLDER_SETTINGS", raising=False)
    assert folderprefs.cli_env(tmp_path)[folderprefs.F.ENV] == "none"
    (tmp_path / "mace-folder.toml").write_text("[plots]\nfermi_shift = 0.3\n")
    assert folderprefs.cli_env(tmp_path)[folderprefs.F.ENV] != "none"
    src = (REPO_ROOT / "mace" / "gui" / "server.py").read_text()
    assert 'explicit = env.get(gui_folderprefs.F.ENV, "none") != "none"' in src


def test_publication_figure_never_reads_a_partner_out_of_the_served_folder(tmp_path):
    """A band file whose DOS partner is a symlink out of the served folder:
    the band + DOS Publication Figure copied and plotted the outside file."""
    from pathlib import Path
    fe = REPO_ROOT / "tests" / "data" / "spin_fe"
    root, outside = tmp_path / "root", tmp_path / "secret"
    root.mkdir()
    outside.mkdir()
    shutil.copy(fe / "fe_band.BAND.DAT", root)
    shutil.copy(fe / "fe_doss.DOSS.DAT", outside)
    (root / "fe_doss.DOSS.DAT").symlink_to(outside / "fe_doss.DOSS.DAT")

    class Disk:
        def acquire(self, key):
            d = tmp_path / "cache" / key
            d.mkdir(parents=True, exist_ok=True)
            return d

        def release(self, key, remove=False):
            pass

    def inside(f):
        r = Path(f).resolve()
        return root.resolve() in r.parents
    with pytest.raises(ValueError, match="no band structure and DOS of the same run"):
        plots.render(root / "fe_band.BAND.DAT", "bandos", Disk(), inside=inside)
    assert not list((tmp_path / "cache").rglob("fe_doss.DOSS.DAT"))
    src = (REPO_ROOT / "mace" / "gui" / "server.py").read_text()
    assert 'data["partner"] = filekind.rel_inside(st, other)' in src


def test_plot_task_command_quotes_the_file_name():
    from pathlib import Path
    cmd = plots.command(Path("my fe's_band.BAND.DAT"), "band", "", None, (-1, 1))
    assert shlex.split(cmd)[3] == "my fe's_band.BAND.DAT", cmd
