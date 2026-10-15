"""A recovered workflow step is resubmitted from its own folder (1.1.5).

In a workflow the queue manager found a step's folder by searching
workflow_outputs/workflow_*/step_*/<material_id>. A later step's folder is
named after its calculation (step_002_SP/A_sp), not the material, so the
search missed it: it found step_001_OPT/A (the OPT step's folder) when that
existed, and otherwise fell back to the current directory with "Could not
find workflow directory for A, using current directory". The recovered SP
deck was then copied into, and run from, the wrong folder.
The recorded work_dir of the failed calculation decides now.
"""
import os
import stat

import pytest

SP = """TITLE
CRYSTAL
0 0 0
1
2.0
1
6 0.0 0.0 0.0
END
BASISSET
POB-TZVP-REV2
DFT
PBE
ENDDFT
TOLDEE
9
END
"""
MSG = "Could not find workflow directory"


@pytest.fixture
def wf(tmp_path, monkeypatch):
    from mace.queue.manager import EnhancedCrystalQueueManager

    root = tmp_path / "proj"
    wid = "workflow_20261015_000000_abcd"
    out = root / "workflow_outputs" / wid
    opt_dir = out / "step_001_OPT" / "A"
    sp_dir = out / "step_002_SP" / "A_sp"
    b_dir = out / "step_001_OPT" / "B"
    for d in (opt_dir, sp_dir, b_dir, root / "workflow_configs", root / "workflow_scripts"):
        d.mkdir(parents=True)
    import shutil
    from conftest import REPO_ROOT
    shutil.copy(REPO_ROOT / "mace" / "submission" / "submitcrystal23.sh",
                root / "workflow_scripts" / "submitcrystal23_sp_2.sh")
    (opt_dir / "A.d12").write_text(SP)
    (sp_dir / "A_sp.d12").write_text(SP)
    cap = tmp_path / "sbatched"
    cap.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "sbatch"
    fake.write_text(f'#!/bin/bash\necho "$PWD $1" >> "{cap}/calls"\necho "Submitted batch job 4242"\n')
    fake.chmod(fake.stat().st_mode | stat.S_IRWXU)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.delenv("MACE_WORKFLOW_ID", raising=False)
    monkeypatch.chdir(root)
    mgr = EnhancedCrystalQueueManager(d12_dir=str(root), db_path=str(tmp_path / "mgr.db"),
                                      enable_tracking=True, enable_error_recovery=False)
    assert mgr.is_workflow_context and mgr.workflow_root == root
    mgr.db.create_material(material_id="A", formula="C", source_type="d12", source_file=str(sp_dir / "A_sp.d12"))
    calc_id = mgr.db.create_calculation(material_id="A", calc_type="SP", input_file=str(sp_dir / "A_sp.d12"),
                                        work_dir=str(sp_dir))
    mgr.db.update_calculation_status(calc_id, "failed")
    return mgr, mgr.db.get_calculation(calc_id), root, opt_dir, sp_dir, b_dir, cap


def _check(mgr, calc, opt_dir, sp_dir, cap, capsys):
    assert mgr.resubmit_fixed_calculation(calc, fixed_input=sp_dir / "A_sp.d12")
    out = capsys.readouterr().out
    assert MSG not in out
    assert sorted(p.name for p in opt_dir.iterdir()) == ["A.d12"]        # the OPT step untouched
    rows = [r for r in mgr.db.get_calculations_by_status("submitted") if r["calc_type"] == "SP"]
    assert [r["work_dir"] for r in rows] == [str(sp_dir)]
    calls = (cap / "calls").read_text().split()
    assert calls[0] == str(sp_dir)                                     # sbatch ran in the SP step's folder


def test_recovered_sp_step_resubmits_from_its_own_folder(wf, capsys):
    mgr, calc, root, opt_dir, sp_dir, b_dir, cap = wf
    _check(mgr, calc, opt_dir, sp_dir, cap, capsys)


def test_from_another_materials_folder(wf, capsys, monkeypatch):
    """A completion callback runs in its own job's folder; a recovery it does
    for another calculation still uses that calculation's folder."""
    mgr, calc, root, opt_dir, sp_dir, b_dir, cap = wf
    monkeypatch.chdir(b_dir)
    _check(mgr, calc, opt_dir, sp_dir, cap, capsys)


def test_no_recorded_folder_says_where_it_submits(wf, capsys):
    """A workflow folder that cannot be found: the warning names the folder
    actually used."""
    mgr, calc, root, opt_dir, sp_dir, b_dir, cap = wf
    d = mgr.create_calculation_folder("Z", "SP")
    out = capsys.readouterr().out
    assert d == root
    assert f"no folder of Z under {root / 'workflow_outputs'}" in out and str(root) in out
