"""Atomic deck writes keep a symlinked deck a symlink and keep its mode.

A deck is written to a hidden .<name>.part file and renamed over the target,
so a job never picks up half a deck. os.replace on the link itself replaced a
symlinked deck with a regular file (the shared target kept the old deck) and
gave it the umask's mode instead of the deck's. 1.1.1 wrote with open(path,
'w'), which goes through the link and keeps the mode; so does this now.
"""
import os
import subprocess
import sys

from conftest import REPO_ROOT
from d12_writer import atomic_deck
from ion_fixtures import ec, li_near, molecule_deck, shuffled

MACE_CLI = REPO_ROOT / "mace_cli"


def _shared_deck(tmp_path, text):
    shared = tmp_path / "shared"
    shared.mkdir()
    target = shared / "d.d12"
    target.write_text(text)
    target.chmod(0o664)
    link = tmp_path / "d.d12"
    link.symlink_to(target)
    return target, link


def test_atomic_deck_writes_through_a_symlink_and_keeps_the_mode(tmp_path):
    target, link = _shared_deck(tmp_path, "old\n")
    old_umask = os.umask(0o022)
    try:
        with atomic_deck(str(link)) as f:
            f.write("new\n")
    finally:
        os.umask(old_umask)
    assert link.is_symlink()
    assert target.read_text() == "new\n"
    assert (target.stat().st_mode & 0o777) == 0o664
    assert sorted(p.name for p in target.parent.iterdir()) == ["d.d12"]   # no .part left
    assert sorted(p.name for p in tmp_path.iterdir()) == ["d.d12", "shared"]


def test_atomic_deck_keeps_the_mode_of_a_plain_deck(tmp_path):
    deck = tmp_path / "x.d12"
    deck.write_text("old\n")
    deck.chmod(0o640)
    with atomic_deck(str(deck)) as f:
        f.write("new\n")
    assert deck.read_text() == "new\n"
    assert (deck.stat().st_mode & 0o777) == 0o640


def test_a_discarded_deck_leaves_the_symlink_target_alone(tmp_path):
    target, link = _shared_deck(tmp_path, "old\n")
    with atomic_deck(str(link)) as f:
        f.write("half")
        f.discard()
    assert link.is_symlink() and target.read_text() == "old\n"
    assert sorted(p.name for p in target.parent.iterdir()) == ["d.d12"]


def test_mace_ions_in_place_on_a_symlinked_deck(tmp_path):
    solvent = ec()
    target, link = _shared_deck(tmp_path, molecule_deck(shuffled(solvent + [li_near(solvent, 1)], seed=3)))
    env = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR")}
    r = subprocess.run([sys.executable, str(MACE_CLI), "ions", str(link), "--net-charge", "1",
                        "--in-place"], cwd=tmp_path, stdin=subprocess.DEVNULL,
                       capture_output=True, text=True, timeout=300, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert link.is_symlink()
    assert "DOPING" in target.read_text()
    assert (target.stat().st_mode & 0o777) == 0o664
