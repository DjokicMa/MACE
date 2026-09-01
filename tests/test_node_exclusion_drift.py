"""Checking the hardcoded buy-in exclusion against the live partition.

MENDOZA_NODES names nodes to AVOID, which SLURM cannot tell us - partition
membership is a different question - so the list has to stay hardcoded. What
CAN be checked is whether those nodes are still in the partition at all: a
node that is retired leaves a stale entry, and one that is renamed silently
stops being excluded.

Verified against the real cluster: mendoza_q currently holds 206 nodes
(acm 37, agg 16, agx 2, amr 60, ncc 1, skl 87, vim 3) and all five excluded
nodes are still present, so the shipped list is not stale.
"""
import subprocess
import sys

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT))

from mace.utils.node_exclusion import NodeExclusionManager  # noqa: E402


class _Result:
    def __init__(self, returncode=0, stdout=""):
        self.returncode, self.stdout = returncode, stdout


def _stub(monkeypatch, **kw):
    """Replace subprocess.run so the tests never need SLURM."""
    calls = []

    def fake_run(cmd, **_):
        calls.append(cmd)
        if "raises" in kw:
            raise kw["raises"]
        return _Result(kw.get("returncode", 0), kw.get("stdout", ""))

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def test_partition_query_asks_slurm_for_one_node_per_line(monkeypatch):
    """-N with %N gives one node per line, so the reply needs no range
    expansion - parsing "agg-[011-012]" ourselves would be a second, redundant
    implementation of a format SLURM already flattens on request."""
    calls = _stub(monkeypatch, stdout="agg-011\nagg-012\namr-163\n")
    got = NodeExclusionManager().query_partition_nodes("mendoza_q")
    assert got == ["agg-011", "agg-012", "amr-163"]
    cmd = calls[0]
    assert cmd[0] == "sinfo" and "-N" in cmd
    assert "mendoza_q" in cmd, "the partition must actually be passed through"


def test_duplicate_node_lines_are_collapsed(monkeypatch):
    """sinfo -N repeats a node once per partition it belongs to."""
    _stub(monkeypatch, stdout="amr-163\namr-163\nagg-011\n")
    assert NodeExclusionManager().query_partition_nodes() == ["agg-011", "amr-163"]


def test_no_slurm_is_reported_as_unknown_not_as_no_drift(monkeypatch):
    """The distinction that matters. On a machine with no sinfo the partition
    reads as empty, and treating that as "nothing is stale" would turn a failed
    check into a silent all-clear."""
    _stub(monkeypatch, raises=FileNotFoundError("sinfo"))
    drift = NodeExclusionManager().check_exclusion_drift()
    assert drift["available"] is False
    assert drift["stale"] == [] and drift["still_present"] == []
    assert drift["excluded"], "the list under test must still be reported"


def test_nonzero_exit_is_also_unknown(monkeypatch):
    _stub(monkeypatch, returncode=1, stdout="")
    assert NodeExclusionManager().check_exclusion_drift()["available"] is False


def test_retired_node_is_reported_stale(monkeypatch):
    """amr-179 has left the partition; the entry excluding it is now dead
    weight and the reason it existed is no longer visible."""
    _stub(monkeypatch, stdout="agg-011\nagg-012\namr-163\namr-178\nskl-051\n")
    drift = NodeExclusionManager().check_exclusion_drift()
    assert drift["available"] is True
    assert drift["stale"] == ["amr-179"]
    assert "amr-163" in drift["still_present"]


def test_all_present_reports_no_drift(monkeypatch):
    """The state measured on the real cluster."""
    m = NodeExclusionManager()
    _stub(monkeypatch, stdout="\n".join(m.MENDOZA_NODES + ["skl-051", "acm-018"]))
    drift = m.check_exclusion_drift()
    assert drift["stale"] == []
    assert sorted(drift["still_present"]) == sorted(m.MENDOZA_NODES)


def test_an_explicit_list_can_be_checked(monkeypatch):
    """Callers may check a list other than the shipped default."""
    _stub(monkeypatch, stdout="agg-011\n")
    drift = NodeExclusionManager().check_exclusion_drift(exclude=["agg-011", "gone-001"])
    assert drift["still_present"] == ["agg-011"] and drift["stale"] == ["gone-001"]


# --- the one place the check is wired: the "exclude Mendoza nodes" choice ---
#
# Drift only matters to whoever maintains the hardcoded list, and only while
# they are choosing to rely on it - which is also the only moment SLURM is
# reachable, since that menu runs on the login node. The tests below drive the
# real menu with stdin and a stubbed sinfo; what they pin hardest is that the
# exclusion returned is the same string whatever the check says.

MENDOZA_EXCLUDE_STRING = "agg-[011-012],amr-[163,178-179]"


def _choose_mendoza(monkeypatch, menu_choice="5"):
    monkeypatch.setattr("builtins.input", lambda *_: menu_choice)


def test_menu_option_five_warns_about_a_retired_node(monkeypatch, capsys):
    _choose_mendoza(monkeypatch)
    _stub(monkeypatch, stdout="agg-011\nagg-012\namr-163\namr-178\nskl-051\n")
    exclude_str = NodeExclusionManager().interactive_node_exclusion()
    out = capsys.readouterr().out
    assert exclude_str == MENDOZA_EXCLUDE_STRING
    assert "amr-179 no longer in mendoza_q" in out


def test_menu_option_five_is_silent_when_the_list_is_current(monkeypatch, capsys):
    _choose_mendoza(monkeypatch)
    _stub(monkeypatch, stdout="\n".join(NodeExclusionManager.MENDOZA_NODES))
    exclude_str = NodeExclusionManager().interactive_node_exclusion()
    assert exclude_str == MENDOZA_EXCLUDE_STRING
    assert "no longer in" not in capsys.readouterr().out


def test_menu_option_five_says_nothing_off_cluster(monkeypatch, capsys):
    """On a laptop the check cannot run, and inventing a warning from that
    would be worse than staying quiet."""
    _choose_mendoza(monkeypatch)
    _stub(monkeypatch, raises=FileNotFoundError("sinfo"))
    exclude_str = NodeExclusionManager().interactive_node_exclusion()
    assert exclude_str == MENDOZA_EXCLUDE_STRING
    assert "no longer in" not in capsys.readouterr().out


def test_a_broken_check_never_reaches_the_user(monkeypatch):
    """The notice is advisory, so nothing it does may escape into a submission."""
    m = NodeExclusionManager()
    monkeypatch.setattr(m, "check_exclusion_drift",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert m.stale_exclusion_notice() is None


def test_the_other_menu_choices_do_not_query_the_partition(monkeypatch):
    """Only the hardcoded-list choice pays for the check; the AMD20 default
    must not gain a second SLURM round trip."""
    calls = _stub(monkeypatch, stdout="")
    _choose_mendoza(monkeypatch, menu_choice="2")
    NodeExclusionManager().interactive_node_exclusion()
    assert all(cmd[0] != "sinfo" for cmd in calls)


def test_planner_menu_warns_through_its_own_ui(monkeypatch, tmp_path):
    """The planner keeps a second copy of this menu (Mendoza is choice 3), so
    it needs the notice wired too - routed through ui.warn, not print."""
    from mace.workflow import planner as planner_mod

    warned = []

    class _RecordingUI:
        def info(self, m): pass
        def warn(self, m): warned.append(m)
        def err(self, m): pass

    monkeypatch.setattr(planner_mod, "ui", _RecordingUI())
    _choose_mendoza(monkeypatch, menu_choice="3")
    _stub(monkeypatch, stdout="agg-011\nagg-012\namr-163\namr-178\n")

    p = planner_mod.WorkflowPlanner(work_dir=str(tmp_path))
    assert p.prompt_node_exclusion() == MENDOZA_EXCLUDE_STRING
    assert any("amr-179 no longer in mendoza_q" in m for m in warned)
