"""`mace site set --folder [DIR] KEY VALUE` / `mace site unset --folder [DIR]
KEY`: a project folder's mace-folder.toml from the shell (CLI/GUI parity,
1.1.6). The GUI's Settings > Save to this folder and these commands use one
writer and one set of refusals (mace/utils/folder_settings.py): a value is
checked before anything is written, comments are kept, a symbolic link, a
file or folder everyone can change, and a file above the folder are never
written, and a [submit] value the site's [folders] rules would ignore is
refused. Real CLI subprocesses."""
import os
import stat
import tomllib

import pytest

from test_site_jobscript import _env, _mace

from mace.utils import folder_settings as F


def _proj(tmp_path):
    d = tmp_path / "proj"
    d.mkdir(exist_ok=True)
    return d


def _set(tmp_path, env, *args):
    return _mace(tmp_path, env, "site", *args)


def test_set_creates_the_file_and_unset_removes_the_key(tmp_path):
    env = _env(tmp_path)
    d = _proj(tmp_path)
    r = _set(tmp_path, env, "set", "--folder", str(d), "plots.energy_window", "-3,5")
    assert r.returncode == 0, r.stdout + r.stderr
    f = d / F.FILENAME
    text = f.read_text()
    assert text.startswith(F.TEMPLATE.splitlines()[0])             # created from the GUI's template
    assert tomllib.loads(text)["plots"]["energy_window"] == [-3.0, 5.0]
    assert f"Created {f}" in r.stdout and "Set [plots] energy_window = [-3.0, 5.0]" in r.stdout
    # the same file the GUI writes: its reader sees the value
    assert F.read(f).get("plots.energy_window") == [-3.0, 5.0]
    for key, value, want in (("viewer.show_axes", "false", False), ("database.refresh_seconds", "30", 30),
                             ("units.energy", "meV", "meV"), ("viewer.display_to", "[1, 1, 1.5]", [1.0, 1.0, 1.5]),
                             ("database.columns", "material_id,formula", ["material_id", "formula"]),
                             ("submit.walltime", "1-00", "1-00")):
        r = _set(tmp_path, env, "set", "--folder", str(d), key, value)
        assert r.returncode == 0, (key, r.stdout + r.stderr)
        assert F.read(f).get(key) == want, key
    r = _set(tmp_path, env, "unset", "--folder", str(d), "plots.energy_window")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "plots.energy_window" not in F.read(f).values and "Removed [plots] energy_window" in r.stdout
    r = _set(tmp_path, env, "unset", "--folder", str(d), "plots.energy_window")
    assert r.returncode == 1 and "is not set" in r.stderr


def test_the_folder_defaults_to_the_working_folder_and_comments_are_kept(tmp_path):
    env = _env(tmp_path)
    d = _proj(tmp_path)
    (d / F.FILENAME).write_text("# our group's settings\n[units]\nenergy = \"eV\"   # keep eV\n")
    r = _mace(d, env, "site", "set", "--folder", "units.energy", "Ha")
    assert r.returncode == 0, r.stdout + r.stderr
    text = (d / F.FILENAME).read_text()
    assert text.startswith("# our group's settings\n") and 'energy = "Ha"  # keep eV' in text


def test_bad_keys_and_values_are_refused_before_writing(tmp_path):
    env = _env(tmp_path)
    d = _proj(tmp_path)
    for key, value, words in (("plots.energy_window", "5,-3", "high above low"),
                              ("viewer.atom_scale", "9", "must be a number from"),
                              ("units.energy", "furlongs", "must be one of"),
                              ("no.such", "1", "is not a folder setting"),
                              ("setup.opt2d12_config", "../x.json", "must not leave this folder"),
                              ("submit.walltime", "12:00", "not \"12:00\"")):
        r = _set(tmp_path, env, "set", "--folder", str(d), key, value)
        assert r.returncode == 2, (key, r.stdout)
        assert words in r.stderr, (key, r.stderr)
    assert not (d / F.FILENAME).exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores the permission bits")
def test_the_guis_refusals(tmp_path):
    env = _env(tmp_path)
    d = _proj(tmp_path)
    # a symbolic link
    real = tmp_path / "elsewhere.toml"
    real.write_text("[units]\nenergy = \"eV\"\n")
    (d / F.FILENAME).symlink_to(real)
    r = _set(tmp_path, env, "set", "--folder", str(d), "units.energy", "Ha")
    assert r.returncode == 2 and "symbolic link" in r.stderr, r.stderr
    assert real.read_text() == "[units]\nenergy = \"eV\"\n"
    (d / F.FILENAME).unlink()
    # a folder everyone can write to: MACE would ignore a file there
    open_dir = tmp_path / "scratchroot"
    open_dir.mkdir()
    open_dir.chmod(0o777)
    r = _set(tmp_path, env, "set", "--folder", str(open_dir), "units.energy", "Ha")
    assert r.returncode == 2 and "everyone can write to" in r.stderr, r.stderr
    assert not (open_dir / F.FILENAME).exists()
    # a file above the folder is read there, never written from below
    sub = d / "runs"
    sub.mkdir()
    (d / F.FILENAME).write_text("[units]\nenergy = \"eV\"\n")
    r = _set(tmp_path, env, "set", "--folder", str(sub), "units.energy", "Ha")
    assert r.returncode == 2 and "above this folder" in r.stderr, r.stderr
    assert "Ha" not in (d / F.FILENAME).read_text() and not (sub / F.FILENAME).exists()
    # a file everyone can change is not used, so not edited either
    (d / F.FILENAME).chmod(0o666)
    os.chmod(d / F.FILENAME, stat.S_IMODE(os.stat(d / F.FILENAME).st_mode) | stat.S_IWOTH)
    r = _set(tmp_path, env, "set", "--folder", str(d), "units.energy", "Ha")
    assert r.returncode == 2 and "everyone" in r.stderr, r.stderr


def test_a_submit_value_the_site_would_ignore_is_refused(tmp_path):
    env = _env(tmp_path)
    d = _proj(tmp_path)
    site = tmp_path / "xdg" / "mace"
    site.mkdir(parents=True)
    (site / "site.toml").write_text('[folders]\nallow = ["walltime"]\nmax_walltime = "2-00:00:00"\n')
    r = _set(tmp_path, env, "set", "--folder", str(d), "submit.ntasks", "8")
    assert r.returncode == 2 and "does not let a folder set ntasks" in r.stderr, r.stderr
    r = _set(tmp_path, env, "set", "--folder", str(d), "submit.walltime", "3-00")
    assert r.returncode == 2 and "at most 2-00:00:00" in r.stderr, r.stderr
    r = _set(tmp_path, env, "set", "--folder", str(d), "submit.partition", "general")
    assert r.returncode == 2 and "partition" in r.stderr, r.stderr
    assert not (d / F.FILENAME).exists()
    r = _set(tmp_path, env, "set", "--folder", str(d), "submit.walltime", "1-00")
    assert r.returncode == 0, r.stdout + r.stderr
    assert F.read(d / F.FILENAME).get("submit.walltime") == "1-00"


def test_plain_site_set_still_edits_site_toml(tmp_path):
    env = _env(tmp_path)
    r = _set(tmp_path, env, "set", "resources.ntasks", "16")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp_path / "xdg" / "mace" / "site.toml").is_file()
    assert not (tmp_path / F.FILENAME).exists()


def test_the_gui_route_uses_the_shared_target_and_writer():
    """folderprefs keeps no writer or refusal of its own."""
    from conftest import REPO_ROOT
    text = (REPO_ROOT / "mace" / "gui" / "folderprefs.py").read_text()
    assert "F.save_target(" in text and "F.write(" in text
    assert "is read-only for you" not in text and "above this folder" not in text
