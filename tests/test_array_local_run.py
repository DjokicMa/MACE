"""`mace run` and SLURM job arrays: --array is refused with a pointer to
`mace submit --array`, and a deck that is a task of a MACE job array (queued
under the array's name) is refused like a same-named job."""
from fake_slurm import FakeSlurm
from test_mace_run_cli import box  # noqa: F401  (the stub tree fixture)


def _fake(box, records):
    fake = FakeSlurm(box.root / "slurm", {"records": records})
    env = {"PATH": f"{fake.bindir}:{box.env['PATH']}",
           "FAKE_SLURM_STATE": fake.env["FAKE_SLURM_STATE"],
           "FAKE_SLURM_LOG": fake.env["FAKE_SLURM_LOG"]}
    return fake, env


def test_array_flag_is_refused(box):
    box.deck("x.d12")
    for flags in (["--array"], ["--max-running", "2"]):
        r = box.run(*flags, "x.d12")
        assert r.returncode == 2, r.text
        assert "job arrays are for `mace submit --array`" in r.text
        assert "unrecognized arguments" not in r.text
    assert box.binary_runs() == []


def test_deck_queued_as_an_array_task_is_refused(box):
    box.deck("x.d12")
    arr = box.root / "decks" / ".mace_arrays" / "arr_t_0"
    arr.mkdir(parents=True)
    (arr / "arr_t_0.sh").write_text("#!/bin/bash\n")
    (arr / "manifest.tsv").write_text(
        "# MACE job array manifest v1: task\tdir\tjob\tscript\tcalc_id\n"
        f"0\t{box.work}\tother\t{box.work}/other.sh\t-\n"
        f"1\t{box.work}\tx\t{box.work}/x.sh\t-\n")
    rec = lambda t, st: {"id": f"4242_{t}", "array": "4242", "task": t, "state": st,
                         "name": "mace_arr_d12_0", "command": str(arr / "arr_t_0.sh")}
    fake, env = _fake(box, [rec(0, "RUNNING"), rec(1, "PENDING")])
    r = box.run("-n", "1", "x.d12", **env)
    assert r.returncode == 2, r.text
    assert "x is SLURM array task 4242_1 (PENDING)" in r.text
    assert box.binary_runs() == []
    squeues = [c["argv"] for c in fake.calls("squeue")]
    assert squeues[0] == ["squeue", "-h", "-u", "mace_no_such_user_zz", "-o", "%j %i %T"]
    assert squeues[1] == ["squeue", "-h", "-u", "mace_no_such_user_zz", "-r", "-o",
                          "%j|%i|%T|%o"]


def test_no_second_squeue_without_array_jobs(box):
    box.deck("x.d12")
    fake, env = _fake(box, [{"id": "777", "state": "RUNNING", "name": "other"}])
    r = box.run("-n", "1", "x.d12", **env)
    assert r.returncode == 0, r.text
    assert len(fake.calls("squeue")) == 1
