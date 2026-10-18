"""Write decks now for a SOC restart (1.1.6): opt2d12 with "soc_guess" copies
the parent's .f9 as <deck>.f20 (the CLI's own copy, in its staging folder);
moving it into the folder never replaces a different .f20 (refused in the
CLI's words) and needs room for the copy, as opt2d3's .f9 copy does."""
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import guiharness as H
import p4util as U

pytest.importorskip("numpy")
sys.path.insert(0, str(H.REPO_ROOT))
from mace.gui import saving  # noqa: E402
from mace.gui import setup as S  # noqa: E402

CORPUS = H.REPO_ROOT / "test"
TIPBO3_SP = "SP/TiPbO3_mp-20459_sg99_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis.triplezeta_opt_B3LYP-D3-D3_optimized_sp_B3LYP-D3-D3_optimized"
corpus = pytest.mark.skipif(not (CORPUS / (TIPBO3_SP + ".out")).is_file(), reason="needs the test/ corpus")
NAME = Path(TIPBO3_SP).name


def _parent(d: Path) -> Path:
    """A spin-polarized parent without symmetry (SYMMREMO), so GUESSPNOSO is allowed, with a stand-in .f9."""
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CORPUS / f"{TIPBO3_SP}.out", d / f"{NAME}.out")
    deck = (CORPUS / f"{TIPBO3_SP}.d12").read_text()
    (d / f"{NAME}.d12").write_text(deck.replace("\nEND\n", "\nSYMMREMO\nEND\n", 1))
    (d / f"{NAME}.f9").write_bytes(bytes(range(256)) * 4)
    return d / f"{NAME}.out"


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    if not (CORPUS / (TIPBO3_SP + ".out")).is_file():
        pytest.skip("needs the test/ corpus")
    tmp = tmp_path_factory.mktemp("soc_save")
    root = tmp / "proj"
    _parent(root / "parent")
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_NO_WARMUP="1", MACE_GUI_TEST_HOOKS="1")
    yield root, base
    proc.terminate()
    proc.wait(timeout=15)


def _result(base, r, timeout=300):
    st, _, body = r
    if st != 202:
        return st, body
    tid = body["task"]["id"]
    end = time.time() + timeout
    while time.time() < end:
        s, _, res = U.req(base, f"api/tasks/{tid}/result")
        if s == 200:
            return s, res
        if s != 409 or (isinstance(res, dict) and res.get("task", {}).get("state") in ("failed", "canceled")):
            return s, res
        time.sleep(0.2)
    raise AssertionError("the task did not end")


SRC = {"path": f"parent/{NAME}.out"}
KEYS = {"soc": True, "soc_guess": "guesspnoso"}


def _save(base, json_text):
    return _result(base, U.post(base, "api/setup/save", {"source": SRC, "calc_type": "SP", "json_text": json_text,
                                                          "name": "soc_restart.json", "overwrite": True,
                                                          "write_decks": True}))


@corpus
def test_write_decks_copies_the_parents_f9_as_f20_and_never_over_a_different_one(served):
    root, base = served
    st, _, r = U.post(base, "api/setup/run", {"source": SRC, "calc_type": "SP", "patches": KEYS, "wait": 120})
    assert st == 200 and r["deck"] and "GUESSPNOSO" in r["deck"].splitlines(), r.get("refused")
    st, w = _save(base, r["json_text"])
    assert st == 200, w
    assert w["soc_f9"]["name"] == f"{NAME}.f9" and w["soc_f9"]["bytes"] == 1024
    deck = root / w["written"][0]
    assert deck.read_text() == r["deck"]
    f20 = deck.with_suffix(".f20")
    f9 = root / "parent" / f"{NAME}.f9"
    assert f20.read_bytes() == f9.read_bytes()
    # the same copy again: allowed (it is the parent's .f9)
    st, w2 = _save(base, r["json_text"])
    assert st == 200, w2
    # a different .f20 already there: refused in the CLI's words; nothing moved
    f20.write_bytes(b"another run's matrix")
    deck.write_text("the deck before\n")
    st, w3 = _save(base, r["json_text"])
    assert st != 200
    msg = str(w3)
    assert f"{f20.name} already exists and is not {f9.name}; not overwriting it" in msg, w3
    assert f20.read_bytes() == b"another run's matrix" and deck.read_text() == "the deck before\n"


@corpus
def test_a_scalar_or_scratch_soc_deck_copies_nothing(served):
    root, base = served
    st, _, r = U.post(base, "api/setup/run", {"source": SRC, "calc_type": "SP", "patches": {"soc": True}, "wait": 120})
    assert st == 200 and r["deck"], r.get("refused")
    st, s = _result(base, U.post(base, "api/setup/save", {"source": SRC, "calc_type": "SP", "json_text": r["json_text"],
                                                           "name": "soc_scratch.json", "overwrite": True}))
    assert st == 200 and "soc_f9" not in s


def test_soc_restart_matrix_reads_the_settings_file(tmp_path):
    out = tmp_path / "a.out"
    out.write_text("x")
    (tmp_path / "a.f9").write_bytes(b"m")
    assert saving.soc_restart_matrix('{"soc": true, "soc_guess": "guesspso"}', out) == tmp_path / "a.f9"
    assert saving.soc_restart_matrix('{"soc": true}', out) is None
    assert saving.soc_restart_matrix('{"soc": false, "soc_guess": "guesspso"}', out) is None
    assert saving.soc_restart_matrix('{"soc": true, "soc_guess": "guesspso"}', tmp_path / "a.d12") is None
    assert saving.soc_restart_matrix("not json", out) is None


def test_write_decks_needs_room_for_the_f20_copy(tmp_path, monkeypatch):
    """The free-space rule of opt2d3's .f9 copy: its size plus 1 GB, checked before anything runs."""
    from mace.gui import server
    out = tmp_path / "parent" / "a.out"
    out.parent.mkdir()
    out.write_text("x")
    monkeypatch.setattr(server.shutil, "disk_usage", lambda p: SimpleNamespace(free=10 << 20))
    handler = SimpleNamespace(state=SimpleNamespace(root=tmp_path, rel_new=lambda p: str(Path(p).relative_to(tmp_path))))
    saved = {"argv": ["opt2d12", "--out-file", "parent/a.out", "--config-file", "parent/c.json", "--yes"],
             "soc_f9": {"name": "a.f9", "path": "parent/a.f9", "bytes": 5 << 20}, "command": "mace opt2d12"}
    with pytest.raises(S.SetupError, match=r"a\.f9 is .*the \.f20 copy needs its size plus 1 GB"):
        server._write_decks(handler, saved, "opt2d12", "SP", out)
