"""Pin what every node-exclusion menu returns for every choice.

The "exclude nodes" menu is reached from four places: the manager's own
interactive menu, the two submission scripts (which delegate to it) and the
workflow planner (which numbers its choices differently - Mendoza is 5 in the
manager menu, 3 in the planner). Users answer these menus by number, and
scripts pipe the answers in, so what is pinned here is the exact --exclude
string each answer sequence produces, the default choice, and how many answers
each sequence reads - a menu that asks one question more or less would shift
every piped answer after it.

SLURM is replaced by a fake scontrol listing a small cluster, so these run
anywhere.
"""
import builtins
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT))

from mace.utils.node_exclusion import NodeExclusionManager  # noqa: E402

# Listed out of order on purpose: the menus must sort, not trust scontrol.
FAKE_NODES = [
    "amr-178", "amr-000", "nvf-001", "amr-001", "agg-012", "agg-011", "amr-002",
    "amr-163", "amr-179", "nvf-000", "agx-001", "nfh-001", "neh-001", "nel-001",
    "acm-018", "acm-019", "nal-001", "nif-001", "skl-051", "skl-052", "skl-054",
    "nvl-001", "dev-001",
]

AMD20 = "amr-[000-002,163,178-179],nvf-[000-001]"
ALL_AMD = ("amr-[000-002,163,178-179],nvf-[000-001],agx-001,nfh-001,neh-001,"
           "nel-001,nal-001,agg-[011-012],acm-[018-019]")
ALL_INTEL = "nif-001,nvl-001,skl-[051-052,054]"
MENDOZA = "agg-[011-012],amr-[163,178-179]"


class _Result:
    def __init__(self, returncode, stdout):
        self.returncode, self.stdout = returncode, stdout


@pytest.fixture(autouse=True)
def fake_slurm(monkeypatch):
    """scontrol lists FAKE_NODES; sinfo is unreachable, so the Mendoza drift
    notice stays quiet and cannot change the result."""
    def fake_run(cmd, **_):
        if cmd[0] == "scontrol":
            return _Result(0, "".join(f"NodeName={n} Arch=x86_64 State=IDLE\n"
                                      for n in FAKE_NODES))
        return _Result(1, "")
    monkeypatch.setattr(subprocess, "run", fake_run)


def _drive(monkeypatch, menu, answers):
    """Feed ``answers`` to ``menu`` and return (result, prompts read).

    Fails if the menu asks for more answers than the sequence has, so a menu
    that grows a question is caught rather than fed a blank."""
    queue = list(answers)
    prompts = []

    def fake_input(prompt=""):
        prompts.append(prompt)
        if not queue:
            raise AssertionError(f"menu asked for an extra answer: {prompt!r}")
        return queue.pop(0)

    monkeypatch.setattr(builtins, "input", fake_input)
    result = menu()
    assert not queue, f"menu left answers unread: {queue}"
    return result, prompts


# --- the manager menu, used as-is by mace/submission/crystal.py and properties.py

MANAGER_CASES = [
    # (answers, expected --exclude string)
    (["", ""], AMD20),                      # default choice is 2, confirm defaults to yes
    (["1"], None),
    (["2", ""], AMD20),
    (["2", "y"], AMD20),
    (["2", "n"], None),
    (["3", ""], ALL_AMD),
    (["3", "n"], None),
    (["4", ""], ALL_INTEL),
    (["4", "n"], None),
    (["5"], MENDOZA),
    # 6 -> 1: by cluster generation
    (["6", "1", "1", ""], "agx-001,nfh-001,neh-001,nel-001,agg-[011-012]"),
    (["6", "1", "2", ""], "acm-[018-019]"),
    (["6", "1", "3", ""], "nal-001"),
    (["6", "1", "4", ""], "amr-[000-002,163,178-179]"),
    (["6", "1", "5", ""], "nvf-[000-001]"),
    (["6", "1", "6", ""], "nif-001"),
    (["6", "1", "7", ""], "skl-[051-052,054]"),
    (["6", "1", "8", ""], "nvl-001"),
    (["6", "1", "9", "amd20, intel18", ""], "amr-[000-002,163,178-179],skl-[051-052,054]"),
    (["6", "1", "9", "amd20 bogus", ""], "amr-[000-002,163,178-179]"),
    (["6", "1", "4", "n"], None),
    (["6", "1", "10"], None),
    # 6 -> 2: by node type
    (["6", "2", "1", ""], "amr-[000-002,163,178-179]"),
    (["6", "2", "2", ""], "nvf-[000-001]"),
    (["6", "2", "3", ""], "nvl-001"),
    (["6", "2", "13", ""], "dev-001"),
    (["6", "2", "2", "n"], None),
    (["6", "2", "14", "agg nvf", ""], "agg-[011-012],nvf-[000-001]"),
    (["6", "2", "14", "agg,skl", ""], "agg-[011-012],skl-[051-052,054]"),
    (["6", "2", "99"], None),
    (["6", "2", "x"], None),
    # 6 -> 3: manual list
    (["6", "3", "amr-042,amr-043,amr-044,nvf-100"], "amr-[042-044],nvf-100"),
    (["6", "3", "amr-[042-050]"], "amr-[042-050]"),
    (["6", "3", "bad,amr-001"], "amr-001"),
    (["6", "3", ""], None),
    (["6", "4"], None),
    # invalid top-level choices
    (["7"], None),
    (["x"], None),
    (["02"], None),
    (["0"], None),
]


def _manager_menu():
    return NodeExclusionManager().interactive_node_exclusion()


def _crystal_menu():
    from mace.submission import crystal
    return crystal.prompt_node_exclusion()


def _properties_menu():
    from mace.submission import properties
    return properties.prompt_node_exclusion()


@pytest.mark.parametrize("menu", [_manager_menu, _crystal_menu, _properties_menu],
                         ids=["manager", "submission.crystal", "submission.properties"])
@pytest.mark.parametrize("answers,expected", MANAGER_CASES,
                         ids=["/".join(a) or "default" for a, _ in MANAGER_CASES])
def test_manager_menu_exclude_string(monkeypatch, menu, answers, expected):
    result, _ = _drive(monkeypatch, menu, answers)
    assert result == expected


@pytest.mark.parametrize("menu", [_manager_menu, _crystal_menu, _properties_menu],
                         ids=["manager", "submission.crystal", "submission.properties"])
def test_manager_menu_default_is_choice_2(monkeypatch, menu):
    _, prompts = _drive(monkeypatch, menu, ["", ""])
    assert "[1-6]" in prompts[0] and "(default: 2)" in prompts[0]


# --- the workflow planner's menu

PLANNER_CASES = [
    ([""], AMD20),                          # default choice is 2, and it does not confirm
    (["1"], None),
    (["2"], AMD20),
    (["3"], MENDOZA),
    # 4: one node type, confirm defaults to no
    (["4", "1", "y"], "amr-[000-002,163,178-179]"),
    (["4", "2", "Y"], "nvf-[000-001]"),
    (["4", "5", "y"], "agx-001"),
    (["4", "1", ""], None),
    (["4", "1", "yes"], None),
    (["4", "14", "agg", "y"], "agg-[011-012]"),
    (["4", "14", "zzz"], None),
    (["4", "99"], None),
    (["4", "x"], None),
    # 5: manual list
    (["5", "amr-042,amr-043,nvf-100"], "amr-[042-043],nvf-100"),
    (["5", "amr-[042-050]"], "amr-[042-050]"),
    (["5", "bad,amr-001"], "amr-001"),
    (["5", ""], None),
    # invalid top-level choices
    (["6"], None),
    (["x"], None),
    (["02"], None),
    (["0"], None),
]


class _QuietUI:
    def info(self, m): pass
    def warn(self, m): pass
    def err(self, m): pass


@pytest.fixture
def planner(monkeypatch, tmp_path):
    from mace.workflow import planner as planner_mod
    monkeypatch.setattr(planner_mod, "ui", _QuietUI())
    return planner_mod.WorkflowPlanner(work_dir=str(tmp_path))


@pytest.mark.parametrize("answers,expected", PLANNER_CASES,
                         ids=["/".join(a) or "default" for a, _ in PLANNER_CASES])
def test_planner_menu_exclude_string(monkeypatch, planner, answers, expected):
    result, _ = _drive(monkeypatch, planner.prompt_node_exclusion, answers)
    assert result == expected


def test_planner_menu_default_is_choice_2(monkeypatch, planner):
    _, prompts = _drive(monkeypatch, planner.prompt_node_exclusion, [""])
    assert "[1-5]" in prompts[0] and "(default: 2)" in prompts[0]


def test_planner_without_slurm_module_asks_nothing(monkeypatch, planner):
    planner.node_manager = None
    assert _drive(monkeypatch, planner.prompt_node_exclusion, [])[0] is None
