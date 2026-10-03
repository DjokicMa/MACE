"""The queue manager hands every helper the database it opened.

HPCC QA (2026-10-03): in a job whose script exports the workflow context, the
completion callback's manager opened the context database (self.db) but kept
self.db_path == 'materials.db', which it then gave the property extractor,
the input-settings extractor, the workflow engine and the error recovery
engine. The OPT's 115 properties were filed in ./materials.db under a calc_id
that exists only in the context database.
"""
import json
import os
import subprocess
import sys

from conftest import REPO_ROOT

PROBE = r"""
import json, sys
from mace.queue.manager import EnhancedCrystalQueueManager
m = EnhancedCrystalQueueManager(d12_dir='.', enable_error_recovery=True)
print(json.dumps({
    'db': str(m.db.db_path),
    'db_path': str(m.db_path),
    'recovery': str(m.error_recovery_engine.db.db_path) if m.error_recovery_engine else None,
}))
"""


def test_callback_manager_uses_the_context_database_throughout(tmp_path):
    wid = "workflow_20261003_162306"
    context_dir = tmp_path / f".mace_context_{wid}"
    env = dict(os.environ)
    env.update(MACE_WORKFLOW_ID=wid, MACE_CONTEXT_DIR=str(context_dir),
               MACE_ISOLATION_MODE="isolated", PYTHONPATH=str(REPO_ROOT))
    env.pop("MACE_HOME", None)
    proc = subprocess.run([sys.executable, "-c", PROBE], cwd=str(tmp_path), env=env,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    paths = json.loads(proc.stdout.strip().splitlines()[-1])
    expected = str(context_dir / "materials.db")
    where = lambda p: os.path.normpath(os.path.join(str(tmp_path), p))
    assert where(paths["db"]) == expected
    assert where(paths["db_path"]) == expected, paths
    assert where(paths["recovery"]) == expected, paths
    assert not (tmp_path / "materials.db").exists()
