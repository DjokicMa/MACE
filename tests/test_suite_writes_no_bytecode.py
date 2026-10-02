"""The suite writes no bytecode, so it never writes into the read-only corpus.

tests/test_cube_subtract_nonortho.py and test_cube_negative_natoms.py import
test/AddedPlottingFunctionalty/subtract_cubes.py, and the import left
__pycache__/subtract_cubes.cpython-311.pyc in the corpus (found on disk).
"""
import importlib
import subprocess
import sys


def test_importing_a_module_from_another_directory_writes_no_pyc(tmp_path, monkeypatch):
    (tmp_path / "corpus_mod.py").write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert importlib.import_module("corpus_mod").VALUE == 1
    assert not (tmp_path / "__pycache__").exists()


def test_subprocesses_write_no_pyc_either(tmp_path):
    (tmp_path / "corpus_mod2.py").write_text("VALUE = 2\n")
    r = subprocess.run([sys.executable, "-c", "import corpus_mod2"], cwd=tmp_path,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert not (tmp_path / "__pycache__").exists()
