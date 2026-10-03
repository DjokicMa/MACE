"""basis_library reads the bundled families from a job directory too.

copy_dependencies copies basis_library.py into a job directory without the
basis_sets/ beside it. read_record already finds the library through the
basis path (the checkout's or $MACE_HOME's), but load_family - and with it
shell_count, soc_basis_record and iter_records - read only the module's own
basis_sets/ and failed there. They now fall back to $MACE_HOME's.
"""
import importlib.util
import shutil

import pytest

from conftest import REPO_ROOT


def _job_dir_copy(tmp_path):
    for name in ("basis_library.py", "basis_file_check.py"):
        shutil.copy(REPO_ROOT / "Crystal_d12" / name, tmp_path / name)
    spec = importlib.util.spec_from_file_location("job_basis_library", tmp_path / "basis_library.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_family_uses_mace_home_from_a_job_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MACE_HOME", str(REPO_ROOT))
    lib = _job_dir_copy(tmp_path)
    assert not (tmp_path / "basis_sets").exists()
    assert lib.shell_count("full.basis.triplezeta", "8") > 0
    assert "282" in lib.load_family("full.basis.triplezeta").keys()


def test_without_mace_home_it_still_says_what_is_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("MACE_HOME", raising=False)
    lib = _job_dir_copy(tmp_path)
    with pytest.raises(FileNotFoundError):
        lib.load_family("full.basis.triplezeta")
