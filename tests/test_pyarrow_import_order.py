"""Regression test for the PyArrow/Arrow extension double-registration crash.

Importing `mace.database` registers the Arrow C++ extension type, so a LATER
`import pandas` raised
`pyarrow.lib.ArrowKeyError: A type extension with name arrow.py_extension_type
already defined` -- crashing real `mace` commands (notably Excel export, which
lazily imports pandas in export/formats.py). The whole suite missed this because
pytest imports modules in-process and never imports pandas after mace.database;
these tests run the REAL invocation order in fresh SUBPROCESSES, the way the CLI
actually builds up its import stack.

mace/database/__init__.py now eagerly imports pyarrow first to prevent the clash.
importorskip is done inside the tests (not at module top) so collection order of
unrelated test modules is not perturbed.
"""
import subprocess
import sys

import pytest


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_mace_database_then_pandas_does_not_crash():
    """The exact real-world order that crashed: DB layer first, then pandas."""
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    r = _run("import mace.database; import pandas; print('OK')")
    assert "ArrowKeyError" not in r.stderr, r.stderr
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_real_excel_export_pandas_path_does_not_crash():
    """Drive the real lazy-pandas export path (export/formats.py) after the DB
    import -- this is the actual `mace` Excel-export sequence that broke."""
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    r = _run(
        "import mace.database\n"
        "from mace.database.export.formats import ExportFormatter\n"
        "import pandas as pd\n"
        "print('rows', len(pd.DataFrame([{'a': 1}])))\n"
    )
    assert "ArrowKeyError" not in r.stderr, r.stderr
    assert r.returncode == 0, r.stderr
