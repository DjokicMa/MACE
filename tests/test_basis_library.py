"""The bundled basis sets are one library file per family (basis_library.py).

The records themselves are pinned byte for byte by test_basis_library_pins.py;
these tests cover the library format, the loader, which paths it serves, and
that a user's own basis directory is still read from disk.
"""
import contextlib
import io
import os
import subprocess
import sys

import pytest

import basis_library as BL
from conftest import REPO_ROOT

BASIS_SETS = REPO_ROOT / "Crystal_d12" / "basis_sets"


def test_the_old_per_element_directories_are_gone_but_the_paths_still_work():
    import mace_config
    from d12_constants import read_basis_file

    for family in BL.FAMILIES:
        assert not (BASIS_SETS / family).exists()
        assert (BASIS_SETS / f"{family}.lib").is_file()
    assert BL.path_exists(mace_config.DEFAULT_TZ_PATH)
    assert BL.path_exists(mace_config.DEFAULT_DZ_PATH)
    assert read_basis_file(mace_config.DEFAULT_TZ_PATH, 82).startswith("282 ")


@pytest.mark.parametrize("family", BL.FAMILIES)
def test_every_record_says_where_it_came_from(family):
    lib = BL.load_family(family)
    assert lib.description
    for key in lib.keys():
        prov = lib.field(key, "provenance")
        assert len(prov) == 1 and prov[0], (family, key)
        if family == "full.basis.triplezeta":
            assert "unverified" not in prov[0]
            assert lib.field(key, "source"), key


def test_doi_only_where_copied_from_a_page():
    """Every DOI in the library was printed on crystal.unito.it: either in the
    fetched record's own reference text, or - for the legacy families, whose
    records were matched against the site - one of the three set families'
    papers (pob-TZVP-rev2 2019, pob-TZVP 2018, sixth-period rev2 2021)."""
    legacy = {"10.1002/jcc.26013", "10.1002/jcc.25195", "10.1002/jcc.26521"}
    for family in BL.FAMILIES:
        lib = BL.load_family(family)
        for key in lib.keys():
            for doi in lib.field(key, "doi"):
                ref = " ".join(lib.field(key, "reference"))
                assert doi in legacy or doi in ref, (family, key, doi)


def test_copies_of_triplezeta_records_inherit_its_label():
    tz = BL.load_family("full.basis.triplezeta")
    dz = BL.load_family("full.basis.doublezeta")
    st = BL.load_family("stuttgart")
    same = [k for k in dz.keys() if dz.record_bytes(k) == tz.record_bytes(k)]
    assert len(same) == 41
    for k in same:
        assert dz.field(k, "source") == tz.field(k, "source")
        assert "byte-identical to full.basis.triplezeta" in dz.field(k, "provenance")[0]
    assert st.field("250", "source") == tz.field("250", "source")
    assert st.field("231", "provenance") == ["unverified (legacy MACE file)"]


def test_doublezeta_provenance_is_an_exact_match_or_unverified():
    """DZ records not copied from TZ were matched exactly (same lines, same
    numbers at printed precision) against X_pob_DZVP_rev2 (H-Br) and
    X_POB_DZVP_2018 (Rb-I); Si has no such set on the site."""
    dz = BL.load_family("full.basis.doublezeta")
    unverified = [k for k in dz.keys()
                  if dz.field(k, "provenance") == ["unverified (legacy MACE file)"]]
    assert unverified == ["14"]
    for key in dz.keys():
        prov = dz.field(key, "provenance")[0]
        if key == "14":
            continue
        assert prov.startswith(("exact numeric match to ", "byte-identical to ")), key
        source = dz.field(key, "source")[0]
        if int(key) <= 35:
            assert source.endswith("_pob_DZVP_rev2"), key
        elif 237 <= int(key) <= 253:
            assert source.endswith("_POB_DZVP_2018"), key


def test_library_files_and_provenance_table_are_what_the_builder_makes():
    result = subprocess.run(
        [sys.executable, str(BASIS_SETS / "build_library.py"), "--check"],
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("family", ["full.basis.triplezeta", "full.basis.doublezeta"])
def test_every_external_record_follows_the_basis_grammar(family):
    """Each record is exactly one basis record by the CRYSTAL23 grammar
    (manual pp.25, 84): NSHL shells, each with NG primitives when ITYB = 0,
    after the ECP when NAT > 200 - nothing missing, nothing left over."""
    lib = BL.load_family(family)
    for key in lib.keys():
        text = lib.record(key)
        try:
            record, trailing = BL.cut_record(text)
        except BL.GrammarError as exc:
            pytest.fail(f"{family} record {key}: {exc}")
        assert (record, trailing) == (text, ""), key


def test_shell_count_reads_nshl():
    assert BL.shell_count("full.basis.triplezeta", "282") == 12
    assert BL.shell_count("full.basis.triplezeta", "1") == 4


# --- which paths the library serves ----------------------------------------

def test_only_the_exact_bundled_paths_are_served(tmp_path, monkeypatch):
    monkeypatch.delenv("MACE_HOME", raising=False)
    tz = str(BASIS_SETS / "full.basis.triplezeta")
    assert BL.bundled_family_for(tz) == "full.basis.triplezeta"
    assert BL.bundled_family_for(tz + "/") == "full.basis.triplezeta"
    assert BL.bundled_family_for(tz + "/sub") is None
    assert BL.bundled_family_for(tz + ".lib") is None
    assert BL.bundled_family_for(str(tmp_path / "full.basis.triplezeta")) is None
    monkeypatch.chdir(tmp_path)
    # a bare library name is the bundled library (E-M6: saved settings store
    # it by name, so they work on another machine) - unless a directory of
    # that name is in the way
    assert BL.bundled_family_for("full.basis.triplezeta") == "full.basis.triplezeta"
    (tmp_path / "full.basis.triplezeta").mkdir()
    assert BL.bundled_family_for("full.basis.triplezeta") is None
    assert BL.bundled_family_for("./basis_sets/full.basis.triplezeta/") is None


def test_a_users_own_directory_is_read_from_disk(tmp_path):
    from d12_constants import check_basis_set_compatibility, read_basis_file

    mine = tmp_path / "full.basis.triplezeta"
    mine.mkdir()
    (mine / "6").write_text("6 1\n0 0 1 2.0 1.0\n 1.0 1.0\n")
    (mine / "282").write_text("282 0\nHAYWSC\n")
    assert read_basis_file(str(mine), 6) == "6 1\n0 0 1 2.0 1.0\n 1.0 1.0\n"
    assert read_basis_file(str(mine) + "/", 82) == "282 0\nHAYWSC\n"
    assert check_basis_set_compatibility(str(mine), [6, 82, 8], "EXTERNAL") == (False, [8])
    assert BL.record_exists(str(mine), 6) and not BL.record_exists(str(mine), 8)
    from basis_file_check import BasisFileError
    with pytest.raises(BasisFileError, match=r"has no basis for O \(Z=8\)"):
        read_basis_file(str(mine), 8)


def test_a_copied_loader_still_serves_the_checkouts_paths(tmp_path, monkeypatch):
    """copy_dependencies copies basis_library.py into a job directory; the
    absolute bundled paths the decks and configs carry must still resolve
    there, from the <family>.lib beside the (absent) directory."""
    import importlib.util
    import mace_config

    monkeypatch.delenv("MACE_HOME", raising=False)
    copy = tmp_path / "basis_library.py"
    copy.write_bytes((REPO_ROOT / "Crystal_d12" / "basis_library.py").read_bytes())
    spec = importlib.util.spec_from_file_location("copied_basis_library", copy)
    copied = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(copied)
    tz = str(BASIS_SETS / "full.basis.triplezeta") + "/"
    assert copied.bundled_family_for(tz) == "full.basis.triplezeta"
    assert copied.read_record(tz, 282) == BL.read_record(mace_config.DEFAULT_TZ_PATH, 282)
    # A real directory of that name is the user's own and stays on disk.
    own = tmp_path / "full.basis.triplezeta"
    own.mkdir()
    (tmp_path / "full.basis.triplezeta.lib").write_bytes(b"not a library")
    assert copied.bundled_family_for(str(own)) is None


def test_copied_scripts_with_mace_home_set_still_read_the_bundled_basis(tmp_path):
    """The real job-directory setup: copy_dependencies copies the Crystal_d12
    scripts flat into a directory, and activate_mace.sh exports MACE_HOME, so
    mace_config's DEFAULT_TZ_PATH lies under $MACE_HOME. That path must be
    served from $MACE_HOME's <family>.lib - not from a basis_sets/ beside the
    copied basis_library.py, which does not exist - as main read the directory
    there. Run as a subprocess so nothing of this test process leaks in."""
    from mace.utils.copy_dependencies import copy_dependencies

    job = tmp_path / "job"
    job.mkdir()
    with contextlib.redirect_stdout(io.StringIO()):
        copy_dependencies(str(job))
    assert (job / "basis_library.py").is_file()
    assert not (job / "basis_sets").exists()
    script = (
        "import sys\n"
        f"sys.path[:0] = [{str(job)!r}]\n"
        f"sys.path.append({str(REPO_ROOT)!r})\n"
        "import basis_library, mace_config\n"
        f"assert basis_library.__file__.startswith({str(job)!r}), basis_library.__file__\n"
        "from d12_constants import read_basis_file\n"
        "for path in (mace_config.DEFAULT_TZ_PATH, mace_config.DEFAULT_DZ_PATH,\n"
        "             mace_config.DEFAULT_TZ_REV2022_PATH):\n"
        "    assert basis_library.record_exists(path, 6), path\n"
        "    print(repr(read_basis_file(path, 6)))\n"
    )
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
    env["MACE_HOME"] = str(REPO_ROOT)
    result = subprocess.run([sys.executable, "-c", script], cwd=job, env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "not found" not in result.stdout
    expected = [repr(BL.read_record(str(BASIS_SETS / f), 6)) for f in
                ("full.basis.triplezeta", "full.basis.doublezeta",
                 "full.basis.triplezeta.rev2022")]
    assert result.stdout.splitlines() == expected


def test_a_root_match_prefers_the_library_beside_that_root(tmp_path, monkeypatch):
    """A path under $MACE_HOME is served from $MACE_HOME's library file, even
    when this module's own basis_sets/ has one too."""
    home = tmp_path / "home"
    (home / "Crystal_d12" / "basis_sets").mkdir(parents=True)
    lib = home / "Crystal_d12" / "basis_sets" / "stuttgart.lib"
    lib.write_bytes(b"#%MACE-BASIS-LIBRARY 1\n#%family stuttgart\n"
                    b"#%record 6 bytes=4\n#%  provenance made up\n6 0\n\n")
    monkeypatch.setenv("MACE_HOME", str(home))
    path = str(home / "Crystal_d12" / "basis_sets" / "stuttgart") + "/"
    assert BL.bundled_family_for(path) == "stuttgart"
    assert BL.read_record(path, 6) == "6 0\n"
    assert not BL.record_exists(path, 8)


def test_mace_config_reports_the_library_files():
    """`python mace_config.py` lists the bundled basis sets as present."""
    env = {k: v for k, v in os.environ.items() if k != "MACE_HOME"}
    result = subprocess.run([sys.executable, str(REPO_ROOT / "mace_config.py")],
                            capture_output=True, text=True, timeout=60, env=env)
    assert result.returncode == 0, result.stderr
    for name, lib in (("Basis DZ", "full.basis.doublezeta.lib"),
                      ("Basis TZ", "full.basis.triplezeta.lib"),
                      ("Stuttgart", "stuttgart.lib")):
        line = next(l for l in result.stdout.splitlines() if f" {name} " in l)
        assert line.startswith("\u2713") and line.endswith(lib), line


# --- the file format -------------------------------------------------------

GOOD = (b"#%MACE-BASIS-LIBRARY 1\n#%family demo\n#%description test\n"
        b"#%record 6 bytes=8\n#%  provenance made up\n6 1\nabc\n\n"
        b"#%record 8 bytes=3\n#%  provenance made up\n8 0\n"
        b"#%missing 9 not fitted\n")


def _write(tmp_path, data):
    path = tmp_path / "demo.lib"
    path.write_bytes(data)
    return path


def test_a_wellformed_library_parses(tmp_path):
    lib = BL.load_file(_write(tmp_path, GOOD))
    assert lib.name == "demo" and lib.keys() == ["6", "8"]
    assert lib.record("6") == "6 1\nabc\n"
    assert lib.record("8") == "8 0"          # no final newline, kept as stored
    assert lib.provenance("6") == [("provenance", "made up")]
    with pytest.raises(BL.MissingRecord, match="not fitted"):
        lib.record("9")


@pytest.mark.parametrize("bad, message", [
    (GOOD.replace(b"bytes=8", b"bytes=9"), "not 9 bytes"),
    (GOOD.replace(b"bytes=8", b"bytes=x"), "bad length"),
    (GOOD.replace(b"#%record 8 ", b"#%record 6 "), "duplicate record 6"),
    (GOOD.replace(b"LIBRARY 1", b"LIBRARY 2"), "unknown library version"),
    (GOOD.replace(b"#%family demo\n", b""), "no #%family"),
    (GOOD.replace(b"#%MACE-BASIS-LIBRARY 1\n", b""), "not a MACE basis library"),
    (GOOD + b"stray line\n", "unexpected line"),
    (GOOD + b"#%missing 8 also\n", "also listed as missing"),
])
def test_a_corrupt_library_is_refused(tmp_path, bad, message):
    with pytest.raises(BL.LibraryFormatError, match=message):
        BL.load_file(_write(tmp_path, bad))


def test_a_corrupt_bundled_library_fails_loudly(tmp_path, monkeypatch):
    broken = tmp_path / "basis_sets"
    broken.mkdir()
    data = (BASIS_SETS / "full.basis.triplezeta.lib").read_bytes()
    (broken / "full.basis.triplezeta.lib").write_bytes(data.replace(b"bytes=", b"bytes=1", 1))
    monkeypatch.setattr(BL, "BASIS_SETS_DIR", str(broken))
    BL.load_family.cache_clear()
    try:
        with pytest.raises(BL.LibraryFormatError):
            BL.load_family("full.basis.triplezeta")
    finally:
        BL.load_family.cache_clear()


def test_unknown_family_is_refused():
    with pytest.raises(KeyError):
        BL.load_family("full.basis.quadruplezeta")
    assert os.path.basename(BL.library_file("stuttgart")) == "stuttgart.lib"
