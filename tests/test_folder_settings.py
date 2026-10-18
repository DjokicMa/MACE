"""Per-folder settings (mace-folder.toml, MACE 1.1.6): the shared loader
(finding the file by walking up, precedence, validation, the security
rules, editing that keeps comments), and the CLI commands that honor it
through the real mace_cli: mace plotting (energy window), mace opt2d12 /
opt2d3 / convert (presets), mace submit (job size and walltime per job
kind within site.toml's [folders] rules) and mace site. Without a folder
file nothing changes (byte for byte)."""
import os
import stat
import subprocess
import sys

import pytest

from test_array_default_path_golden import BAND_D3, SP_D12
from test_site_jobscript import _deck, _env, _mace, _toml

from mace.submission import site
from mace.utils import folder_settings as F


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A home folder under tmp_path, so the walk up stops there."""
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.delenv(F.ENV, raising=False)
    monkeypatch.setattr(F, "CWD_DISCOVERY", True)
    return h


def _write(d, text):
    d.mkdir(parents=True, exist_ok=True)
    (d / F.FILENAME).write_text(text)
    return d / F.FILENAME


# ---------------------------------------------------------------- finding the file
def test_walks_up_and_the_nearest_file_wins(home):
    proj = home / "proj"
    outer = _write(proj, '[viewer]\natom_scale = 0.5\n')
    deep = proj / "a" / "b"
    deep.mkdir(parents=True)
    assert F.find(deep) == outer
    fs = F.load(deep)
    assert fs.values == {"viewer.atom_scale": 0.5} and fs.folder == proj and not fs.warnings
    inner = _write(proj / "a", '[viewer]\nshow_cell = false\n')
    assert F.load(deep).values == {"viewer.show_cell": False}          # nearest only, never merged
    assert F.find(home / "elsewhere") is None


def test_the_walk_stops_below_home_and_root(home):
    _write(home, '[viewer]\natom_scale = 0.5\n')                        # in home itself: never read
    (home / "proj").mkdir()
    assert F.find(home / "proj") is None and F.load(home / "proj").values == {}


def test_env_turns_it_off_or_names_a_file(home, monkeypatch):
    f = _write(home / "proj", '[viewer]\natom_scale = 0.5\n')
    monkeypatch.setenv(F.ENV, "none")
    assert F.find(home / "proj") is None
    other = _write(home / "shared", '[viewer]\natom_scale = 0.7\n')
    monkeypatch.setenv(F.ENV, str(other))
    assert F.load(home / "proj").values == {"viewer.atom_scale": 0.7} and f.exists()


def test_cwd_default_and_the_servers_switch(home, monkeypatch):
    _write(home / "proj", '[viewer]\natom_scale = 0.5\n')
    monkeypatch.chdir(home / "proj")
    assert F.load().values == {"viewer.atom_scale": 0.5}
    monkeypatch.setattr(F, "CWD_DISCOVERY", False)                       # mace gui's server
    assert F.load().values == {} and F.load(home / "proj").values == {"viewer.atom_scale": 0.5}


# ---------------------------------------------------------------- validation
def test_unknown_keys_and_bad_values_are_ignored_with_warnings(home):
    f = _write(home / "p", '[viewer]\natom_scale = 9\ncolor_scheme = "vesta"\nshiny = 1\n[plots]\n'
                           'energy_window = [5, -5]\nlog_delta_e = true\n[weird]\nx = 1\n'
                           '[database]\ncolumns = ["formula", "nope"]\nrefresh_seconds = 15\n')
    fs = F.load(home / "p")
    assert fs.values == {"viewer.color_scheme": "vesta", "plots.log_delta_e": True, "database.refresh_seconds": 15}
    text = "\n".join(fs.warnings)
    assert "viewer.atom_scale must be a number from 0.05 to 1.5" in text
    assert "plots.energy_window must be [low, high]" in text and "database.columns must be a list" in text
    assert "unknown keys ignored: viewer.shiny, weird" in text and str(f) in text


def test_invalid_toml_ignores_the_file(home):
    fs = F.load(_write(home / "p", "[viewer\natom_scale = 1\n").parent)
    assert fs.refused and fs.values == {} and "not valid TOML" in fs.warnings[0]


def test_paths_must_stay_inside_the_folder(home, tmp_path):
    p = home / "p"
    (p / "presets").mkdir(parents=True)
    (p / "presets" / "sp.json").write_text("{}")
    outside = tmp_path / "evil.json"
    outside.write_text("{}")
    os.symlink(outside, p / "presets" / "link.json")
    _write(p, '[setup]\nopt2d12_config = "presets/sp.json"\nopt2d3_config = "../evil.json"\n'
              f'cif2d12_options = "{outside}"\n[database]\nfile = "presets/link.json"\n')
    fs = F.load(p)
    assert fs.values == {"setup.opt2d12_config": "presets/sp.json"}
    text = "\n".join(fs.warnings)
    assert "opt2d3_config must not leave this folder" in text
    assert "cif2d12_options must be a relative path" in text and "database.file leads out of this folder" in text


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores the mode bits")
def test_files_others_can_change_are_ignored(home):
    f = _write(home / "p", '[viewer]\natom_scale = 0.5\n')
    os.chmod(f, 0o666)
    fs = F.load(home / "p")
    assert fs.refused and fs.values == {} and "can be changed by everyone" in fs.warnings[0]
    os.chmod(f, 0o644)
    os.chmod(home / "p", 0o777)
    fs = F.load(home / "p")
    assert fs.refused and "folder everyone can write to" in fs.warnings[0]
    os.chmod(home / "p", 0o755)
    assert F.load(home / "p").values == {"viewer.atom_scale": 0.5}


def test_a_symlinked_file_is_ignored(home):
    real = _write(home / "real", '[viewer]\natom_scale = 0.5\n')
    (home / "p").mkdir()
    os.symlink(real, home / "p" / F.FILENAME)
    fs = F.load(home / "p")
    assert fs.refused and "symbolic link" in fs.warnings[0]


def test_gui_values_and_paths_relative_to_the_served_folder(home):
    p = home / "p"
    (p / "dbs").mkdir(parents=True)
    _write(p, '[viewer]\nprojection = "perspective"\n[units]\nenergy = "meV"\n[database]\nfile = "dbs/x.db"\n'
              '[submit]\nntasks = 16\n')
    fs = F.load(p / "dbs")
    assert fs.gui(p) == {"projection": "perspective", "energyUnit": "meV", "defaultDb": "dbs/x.db"}
    assert fs.gui(p / "dbs") == {"projection": "perspective", "energyUnit": "meV", "defaultDb": "x.db"}
    assert fs.cli_only() == {"submit.ntasks": 16}
    # every GUI key a folder can hold is one GUI setting, once
    assert len(F.GUI_KEYS) == len([k for k in F.KEYS.values() if k["gui"]])


# ---------------------------------------------------------------- writing
def test_write_never_follows_a_planted_temp_file(home):
    """The save's temp file is a fresh one (O_EXCL): a link planted at the old guessable name
    (.mace-folder.toml.<pid>.tmp) is not followed, so the file it points at is not overwritten."""
    p = home / "p"
    p.mkdir()
    victim = home / "victim.txt"
    victim.write_text("keep me\n")
    planted = p / f".{F.FILENAME}.{os.getpid()}.tmp"
    planted.symlink_to(victim)
    F.write(p / F.FILENAME, {"viewer.atom_scale": 0.5})
    assert victim.read_text() == "keep me\n"
    assert "atom_scale = 0.5" in (p / F.FILENAME).read_text()
    assert stat.S_IMODE((p / F.FILENAME).stat().st_mode) == 0o644
    assert sorted(x.name for x in p.iterdir()) == sorted([F.FILENAME, planted.name])


def test_write_keeps_comments_and_creates_from_the_template(home):
    p = home / "p"
    p.mkdir()
    f = p / F.FILENAME
    fs = F.write(f, {"viewer.atom_scale": 0.5, "plots.energy_window": [-4.0, 4.0], "submit.properties.ntasks": 8})
    text = f.read_text()
    assert text.startswith("# MACE folder settings") and "# color_scheme" in text
    assert "atom_scale = 0.5" in text and "energy_window = [-4.0, 4.0]" in text
    assert text.index("[submit.properties]") < text.index("ntasks = 8")
    assert fs.values == {"viewer.atom_scale": 0.5, "plots.energy_window": [-4.0, 4.0], "submit.properties.ntasks": 8}
    assert stat.S_IMODE(f.stat().st_mode) & 0o777 == 0o644
    # a hand-written comment and value survive; a value's own trailing comment too
    f.write_text(f.read_text().replace("atom_scale = 0.5", "atom_scale = 0.5   # agreed in group meeting")
                 + "# the end\n")
    F.write(f, {"viewer.atom_scale": 0.6, "units.energy": "Ha"}, ["plots.energy_window"])
    text = f.read_text()
    assert "atom_scale = 0.6  # agreed in group meeting" in text and text.endswith("# the end\n")
    assert "energy_window = [" not in text.replace("# energy_window", "") and 'energy = "Ha"' in text
    assert F.read(f).values == {"viewer.atom_scale": 0.6, "units.energy": "Ha", "submit.properties.ntasks": 8}


def test_write_refuses_bad_values_and_hand_written_forms(home):
    p = home / "p"
    f = _write(p, 'viewer = { atom_scale = 0.5 }\n')
    before = f.read_text()
    with pytest.raises(ValueError, match="atom_scale must be a number"):
        F.write(f, {"viewer.atom_scale": 7})
    with pytest.raises(ValueError, match="is not a folder setting"):
        F.write(f, {"viewer.nope": 1})
    with pytest.raises(ValueError):
        F.write(f, {"viewer.atom_scale": 0.6})                         # an inline table: edit by hand
    assert f.read_text() == before
    with pytest.raises(ValueError, match="must not leave this folder"):
        F.write(f, {"setup.opt2d12_config": "../x.json"})


def test_edit_text_handles_multiline_arrays():
    text = '[database]\ncolumns = [\n  "formula",\n  "status",\n]  # mine\n[viewer]\nlabels = "off"\n'
    out = F.edit_text(text, {"database.columns": ["status"]}, [])
    assert out == '[database]\ncolumns = ["status"]  # mine\n[viewer]\nlabels = "off"\n'
    assert F.edit_text(out, {}, ["database.columns"]) == '[database]\n[viewer]\nlabels = "off"\n'


# ---------------------------------------------------------------- the CLI's uses
def test_preset_args_only_when_none_is_given(home):
    p = home / "p"
    (p / "presets").mkdir(parents=True)
    for n in ("sp.json", "band.json", "cif.json"):
        (p / "presets" / n).write_text("{}")
    _write(p, '[setup]\nopt2d12_config = "presets/sp.json"\nopt2d3_config = "presets/band.json"\n'
              'cif2d12_options = "presets/cif.json"\n')
    fs = F.load(p)
    a, note = F.preset_args("opt2d12", ["--out-file", "x.out"], fs)
    assert a == ["--out-file", "x.out", "--config-file", str(p / "presets" / "sp.json")] and "opt2d12_config" in note
    for argv in (["--config-file", "mine.json"], ["--config-file=mine.json"], ["--save-options"], ["-h"]):
        assert F.preset_args("opt2d12", argv, fs) == (argv, None)
    assert F.preset_args("opt2d3", ["-i", "x.out"], fs)[0][-2:] == ["--config-file", str(p / "presets" / "band.json")]
    for argv in (["-c", "x"], ["--save-config"], ["--list-configs"]):
        assert F.preset_args("opt2d3", argv, fs) == (argv, None)
    # cif2d12's options file matters for --batch only, and is never the one --save_options writes
    assert F.preset_args("convert", ["--cif_dir", "."], fs) == (["--cif_dir", "."], None)
    assert F.preset_args("convert", ["--batch"], fs)[0] == ["--batch", "--options_file", str(p / "presets" / "cif.json")]
    assert F.preset_args("convert", ["--batch", "--save_options"], fs)[1] is None
    (p / "presets" / "sp.json").unlink()
    a, note = F.preset_args("opt2d12", ["--out-file", "x.out"], fs)
    assert a == ["--out-file", "x.out"] and "does not exist; not used" in note
    assert F.preset_args("opt2d12", ["x"], F.FolderSettings()) == (["x"], None)


def test_plotting_takes_the_folder_window_unless_given(home, monkeypatch):
    import importlib
    pm = importlib.import_module("mace.plotting.main")
    from mace.plotting.registry import PlotKind, get
    seen = []
    monkeypatch.setattr(get(PlotKind.BAND), "handler", lambda files, config, out: seen.append(dict(config)))
    p = home / "p"
    p.mkdir()
    (p / "x.BAND.DAT").write_text("#\n")
    monkeypatch.chdir(p)
    assert pm.main(["--band", "x.BAND.DAT"]) == 0
    base = seen.pop()
    assert (base["e_lower"], base["e_upper"]) == (-6.0, 6.0)          # no folder file: the CLI's own defaults
    _write(p, '[plots]\nenergy_window = [-3, 2.5]\n')
    assert pm.main(["--band", "x.BAND.DAT"]) == 0
    got = seen.pop()
    assert (got["e_lower"], got["e_upper"]) == (-3.0, 2.5)
    assert {k: v for k, v in got.items() if k not in ("e_lower", "e_upper")} == \
        {k: v for k, v in base.items() if k not in ("e_lower", "e_upper")}
    assert pm.main(["--band", "x.BAND.DAT", "--e-lower", "-1"]) == 0
    got = seen.pop()
    assert (got["e_lower"], got["e_upper"]) == (-1.0, 2.5)            # a flag wins


def _cli(cwd, env, *args):
    return _mace(cwd, env, *args)


def test_cli_notes_through_mace_cli(tmp_path):
    env = _env(tmp_path)
    p = tmp_path / "home" / "proj"
    (p / "presets").mkdir(parents=True)
    (p / "presets" / "sp.json").write_text("{}")
    r = _cli(p, env, "plotting", "--band")
    assert r.returncode == 1 and "MACE:" not in r.stderr                 # no file: no note
    _write(p, '[plots]\nenergy_window = [-3, 3]\nlog_delta_e = 4\n[setup]\nopt2d12_config = "presets/sp.json"\n')
    r = _cli(p, env, "plotting", "--band")
    assert "using --e-lower -3 --e-upper 3 from" in r.stderr and "plots.log_delta_e must be true or false" in r.stderr
    r = _cli(p, env, "opt2d12", "--out-file", "missing.out", "--calc-type", "SP", "--non-interactive")
    assert f"using --config-file {p / 'presets' / 'sp.json'}" in r.stderr
    r = _cli(p, env, "opt2d12", "--out-file", "missing.out", "--config-file", "own.json", "--non-interactive")
    assert "using --config-file" not in r.stderr


# ---------------------------------------------------------------- mace submit (the site layer)
def _proj(tmp, text):
    d = tmp / "home" / "proj"
    _write(d, text)
    return d


def _submit_in(d, env, deck, *args):
    _deck(d, deck, SP_D12 if deck.endswith(".d12") else BAND_D3)
    if deck.endswith(".d3"):
        (d / deck.replace(".d3", ".f9")).write_text("fort9\n")
    (d / (deck.rsplit(".", 1)[0] + ".sh")).unlink(missing_ok=True)
    p = _mace(d, env, "submit", "--nosubmit", *args, deck)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    return (d / (deck.rsplit(".", 1)[0] + ".sh")).read_text(), p


def _time(text):
    return [l for l in text.splitlines() if l.startswith("#SBATCH -t ")]


def test_submit_without_a_folder_file_keeps_its_bytes(tmp_path):
    env = _env(tmp_path)
    d = tmp_path / "home" / "proj"
    a, _ = _submit_in(d, env, "m.d12")
    b, _ = _submit_in(d, dict(env, MACE_FOLDER_SETTINGS="none"), "m.d12")
    assert a == b


@pytest.mark.parametrize("deck", ["m.d12", "m_band.d3"])
@pytest.mark.parametrize("target", ["deck", "folder"])
def test_submit_from_outside_reads_the_decks_folder_file(tmp_path, deck, target):
    # review fix: `mace submit proj/m.d12` from the parent folder read the
    # parent's (no) folder file, while the callback in proj reads proj's
    env = _env(tmp_path)
    d = _proj(tmp_path, '[submit]\nntasks = 12\nwalltime = "3:00:00"\n[submit.properties]\nntasks = 4\n')
    _deck(d, deck, SP_D12 if deck.endswith(".d12") else BAND_D3)
    if deck.endswith(".d3"):
        (d / deck.replace(".d3", ".f9")).write_text("fort9\n")
    arg = f"proj/{deck}" if target == "deck" else "proj"
    p = _mace(d.parent, env, "submit", "--nosubmit", arg)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    text = (d / (deck.rsplit(".", 1)[0] + ".sh")).read_text()
    want = "#SBATCH --ntasks=12" if deck.endswith(".d12") else "#SBATCH --ntasks=4"
    assert want in text
    assert f"job settings from {d / F.FILENAME}" in p.stdout
    # the same bytes as a submit from inside the folder (what the callback sees)
    inside, _ = _submit_in(d, env, deck)
    assert inside == text


def test_submit_takes_job_size_and_walltime_per_kind(tmp_path):
    env = _env(tmp_path)
    d = _proj(tmp_path, '[submit]\nntasks = 48\nwalltime = "1-12:00:00"\n[submit.properties]\nntasks = 4\n'
                        'walltime = "3:00:00"\n')
    scf, p = _submit_in(d, env, "m.d12")
    assert "#SBATCH --ntasks=48" in scf and _time(scf) == ["#SBATCH -t 1-12:00:00"]
    assert f"job settings from {d / F.FILENAME}" in p.stdout
    prop, _ = _submit_in(d, env, "m_band.d3")
    assert "#SBATCH --ntasks=4" in prop and _time(prop) == ["#SBATCH -t 0-03:00:00"]
    # an explicit --walltime and the environment still win
    scf, _ = _submit_in(d, env, "m.d12", "--walltime", "0:30:00")
    assert _time(scf) == ["#SBATCH -t 0:30:00"]
    scf, _ = _submit_in(d, dict(env, MACE_SLURM_NTASKS="8"), "m.d12")
    assert "#SBATCH --ntasks=8" in scf
    # mace site says where they come from
    r = _mace(d, env, "site")
    assert f"Folder:      {d / F.FILENAME}" in r.stdout and "CRYSTAL 1-12:00:00, properties 0-03:00:00" in r.stdout
    assert f"folder: {d / F.FILENAME}" in r.stdout


def test_site_rules_bound_what_a_folder_sets(tmp_path):
    _toml(tmp_path, '[folders]\nallow = ["ntasks", "partition"]\npartitions = ["amd24", "intel18"]\n'
                    'max_ntasks = 64\n')
    env = _env(tmp_path)
    d = _proj(tmp_path, '[submit]\nntasks = 128\nwalltime = "1:00:00"\npartition = "amd24"\n')
    s = site.load(dict(env, MACE_FOLDER_SETTINGS=str(d / F.FILENAME)))
    assert s.partition == "amd24" and s.sources["partition"] == "folder"
    assert "ntasks" not in s.res_raw[""] and s.folder_walltime == {}
    text = "\n".join(s.warnings)
    assert "[submit] ntasks is ignored: the site allows at most 64" in text
    assert "[submit] walltime is ignored: the site does not let a folder set the walltime" in text
    _write(d, '[submit]\npartition = "gpu"\n')
    s = site.load(dict(env, MACE_FOLDER_SETTINGS=str(d / F.FILENAME)))
    assert s.partition is None and "not one of the site's partitions for folders: amd24, intel18" in "\n".join(s.warnings)
    # by default a folder may set ntasks and walltime, and choose no partition
    _toml(tmp_path, "")
    _write(d, '[submit]\nntasks = 16\npartition = "amd24"\n')
    s = site.load(dict(env, MACE_FOLDER_SETTINGS=str(d / F.FILENAME)))
    assert s.res_raw[""]["ntasks"] == 16 and s.partition is None
    assert "the site lists no partitions a folder may choose" in "\n".join(s.warnings)
    # the environment stays above the folder
    _toml(tmp_path, '[folders]\npartitions = ["amd24"]\n')
    _write(d, '[submit]\npartition = "amd24"\n')
    s = site.load(dict(env, MACE_FOLDER_SETTINGS=str(d / F.FILENAME), MACE_SLURM_PARTITION="debug"))
    assert s.partition == "debug"


def test_site_folders_table_is_checked(tmp_path):
    for text, msg in (('[folders]\nallow = ["account"]\n', "[folders] allow"),
                      ('[folders]\nmax_walltime = "soon"\n', "[folders] max_walltime"),
                      ('[folders]\npartitions = ["a b"]\n', "[folders] partitions")):
        _toml(tmp_path, text)
        with pytest.raises(site.SiteConfigError, match=msg.replace("[", r"\[").replace("]", r"\]")):
            site.load(_env(tmp_path))


def test_folder_walltime_is_capped_by_the_partition(tmp_path):
    from test_walltime_partition_maxtime import slurm_maxtime
    d = _proj(tmp_path, '[submit]\nwalltime = "5-00:00:00"\n')
    text = site.TEMPLATES["scf"].read_text()
    for maxtime, want in (("2-00:00:00", "2-00:00:00"), ("14-00:00:00", "5-00:00:00")):
        env = slurm_maxtime(tmp_path, dict(_env(tmp_path), MACE_FOLDER_SETTINGS=str(d / F.FILENAME)), maxtime)
        s = site.load_resolved(env)
        new, changes, _ = site.customize_generator_text(text, "scf", s, None)
        assert f"walltime {want}" in changes and f"time={want.split('-')[0]}" in new


# ---------------------------------------------------------------- 1.1.6 follow-ups
def test_interactive_plotting_offers_the_folder_window(home, monkeypatch):
    from mace.plotting.handlers import legacy
    asked = []
    monkeypatch.setattr(legacy, "yes_no_prompt", lambda q, d: asked.append(q) or q.startswith("Use default"))
    monkeypatch.setattr(legacy, "configure_output_formats", lambda interactive=True: ["png"])
    monkeypatch.setattr(legacy, "select_option", lambda *a, **k: 1)
    p = home / "p"
    p.mkdir()
    monkeypatch.chdir(p)
    c = legacy.configure_band_plot(True)
    assert (c["e_lower"], c["e_upper"]) == (-6.0, 6.0) and asked[0] == "Use default energy range (-6 to +6 eV)?"
    _write(p, '[plots]\nenergy_window = [-4, 3.5]\n')
    asked.clear()
    for conf in (legacy.configure_band_plot, legacy.configure_dos_plot):
        c = conf(True)
        assert (c["e_lower"], c["e_upper"]) == (-4.0, 3.5)
    assert asked[0] == "Use default energy range (-4 to +3.5 eV)?"
    assert legacy.configure_band_plot(False)["e_lower"] == -6.0          # --all and flags: their own defaults


def test_workflow_plans_take_the_folder_walltime(tmp_path, monkeypatch):
    env = _env(tmp_path)
    d = _proj(tmp_path, '[submit]\nntasks = 48\nwalltime = "1-00:00:00"\n[submit.properties]\nwalltime = "3:00:00"\n')
    for k in list(os.environ):
        if k.startswith(("MACE_SLURM_", "MACE_SITE")):
            monkeypatch.delenv(k)
    monkeypatch.setenv("XDG_CONFIG_HOME", env["XDG_CONFIG_HOME"])
    monkeypatch.setenv("MACE_SITE_GROUP_CONFIG", "none")
    monkeypatch.setenv(F.ENV, str(d / F.FILENAME))
    s = site.load()
    assert site.overlay_resources({"ntasks": 32, "walltime": "7-00:00:00"}, s, "scf") == \
        {"ntasks": 48, "walltime": "1-00:00:00"}
    assert site.overlay_resources({"walltime": "2:00:00"}, s, "prop")["walltime"] == "0-03:00:00"
    assert s.resource_defaults()["SP"]["walltime"] == "1-00:00:00"
    from mace import run_mace
    q = run_mace._get_quick_slurm_config("SP", 1)["resources"]
    assert (q["ntasks"], q["walltime"]) == (48, "1-00:00:00")
    from mace.workflow import planner
    assert planner._folder_walltime("prop") == "0-03:00:00"
    monkeypatch.setenv(F.ENV, "none")
    s = site.load()
    assert site.overlay_resources({"ntasks": 32, "walltime": "7-00:00:00"}, s, "scf") == \
        {"ntasks": 32, "walltime": "7-00:00:00"}


def test_an_unusable_parent_functional_falls_back_to_the_folders(monkeypatch):
    import d12_interactive as DI
    monkeypatch.setattr(DI, "_SCRATCH_CACHE", {})
    monkeypatch.setenv("MACE_SETUP_FUNCTIONAL", "PBE0")
    opts = DI.ensure_known_functional({"method_type": "DFT", "functional": None, "dispersion": True,
                                       "unrecognised_functional": "B2PLYP"})
    assert opts["functional"] == "PBE0-D3"
    monkeypatch.setattr(DI, "_SCRATCH_CACHE", {})
    monkeypatch.setenv("MACE_SETUP_FUNCTIONAL", "")
    opts = DI.ensure_known_functional({"method_type": "DFT", "functional": None, "dispersion": False})
    assert opts["functional"] == "HSE06"
    # a parent whose functional is known keeps it
    monkeypatch.setattr(DI, "_SCRATCH_CACHE", {})
    monkeypatch.setenv("MACE_SETUP_FUNCTIONAL", "PBE0")
    assert DI.ensure_known_functional({"method_type": "DFT", "functional": "B3LYP"})["functional"] == "B3LYP"


# ---------------------------------------------------------------- walltime: site.toml's rules (1.1.6 review)
@pytest.mark.parametrize("value", ["12:00", "30", "90:00", "1-25:00:00", "400-00:00:00", "soon"])
def test_folder_walltime_refuses_what_site_toml_refuses(value):
    assert F.check_value("submit.walltime", value)
    assert F.walltime_seconds(value) is None
    with pytest.raises(site.SiteConfigError):
        site._valid_walltime(value, "x")


@pytest.mark.parametrize("value,secs", [("2:00:00", 7200), ("1-00", 86400), ("2-00:00:00", 172800),
                                        ("1-12:30", 131400), ("100:00:00", 360000)])
def test_folder_walltime_takes_what_site_toml_takes(value, secs):
    assert F.check_value("submit.walltime", value) is None
    assert F.walltime_seconds(value) == secs


def test_an_ambiguous_folder_walltime_is_ignored_with_a_warning(tmp_path):
    env = _env(tmp_path)
    d = _proj(tmp_path, '[submit]\nwalltime = "12:00"\n')
    s = site.load(dict(env, MACE_FOLDER_SETTINGS=str(d / F.FILENAME)))
    assert s.walltime("scf") is None and not s.folder_walltime


def test_site_folders_max_walltime_uses_the_same_rules(tmp_path):
    _toml(tmp_path, '[folders]\nmax_walltime = "12:00"\n')
    with pytest.raises(site.SiteConfigError, match="max_walltime"):
        site.load(_env(tmp_path))


def test_color_scheme_takes_every_palette_preset():
    """1.1.6 review: Save to this folder must work for every preset the GUI offers (cvd, legacy too)."""
    from mace.utils import element_palettes as EP
    assert set(F.KEYS["viewer.color_scheme"]["choices"]) == set(EP.preset_ids())
