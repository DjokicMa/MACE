"""The exclusion menus share one list of choices and one Mendoza node list.

tests/test_node_exclusion_menus.py pins what each answer returns; this pins
the shared definitions those menus are now built from, so a choice cannot be
renumbered or relabelled in one menu without the change being deliberate.
"""
import sys

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT))

from mace.utils.node_exclusion import (  # noqa: E402
    DEFAULT_EXCLUSION_CHOICE, EXCLUSION_CHOICES, MANAGER_MENU, PLANNER_MENU,
    NodeExclusionManager, menu_choice, menu_default_answer,
)


def test_option_numbers_users_rely_on():
    assert [MANAGER_MENU.index(k) + 1 for k in ("none", "amd20", "mendoza", "custom")] == [1, 2, 5, 6]
    assert [PLANNER_MENU.index(k) + 1 for k in ("none", "amd20", "mendoza", "by_type", "manual")] == [1, 2, 3, 4, 5]


def test_every_menu_choice_has_a_label():
    for menu in (MANAGER_MENU, PLANNER_MENU):
        assert set(menu) <= set(EXCLUSION_CHOICES)


def test_default_is_amd20_in_both_menus():
    assert DEFAULT_EXCLUSION_CHOICE == "amd20"
    assert menu_default_answer(MANAGER_MENU) == menu_default_answer(PLANNER_MENU) == "2"
    assert menu_choice(MANAGER_MENU, "") == menu_choice(PLANNER_MENU, "  ") == "amd20"


def test_only_exact_option_numbers_select_a_choice():
    assert menu_choice(PLANNER_MENU, "3") == "mendoza"
    assert menu_choice(MANAGER_MENU, " 5 ") == "mendoza"
    for answer in ("02", "0", "6", "x", "2.0", "²"):
        assert menu_choice(PLANNER_MENU, answer) is None


def test_mendoza_label_is_built_from_the_node_list():
    m = NodeExclusionManager()
    assert m.mendoza_exclude_string() == "agg-[011-012],amr-[163,178-179]"
    planner_line = m.menu_lines(PLANNER_MENU)[2]
    assert planner_line.startswith("3) Exclude Mendoza group nodes (agg-[011-012], amr-[163,178-179])")


def test_node_list_parser_warns_and_skips_bad_names():
    warned = []
    got = NodeExclusionManager().exclude_string_from_node_list(
        "amr-001, amr-002,bad,nvf-010", warn=warned.append)
    assert got == "amr-[001-002],nvf-010"
    assert warned == ["Warning: Invalid node format 'bad', skipping"]
    assert NodeExclusionManager().exclude_string_from_node_list("amr-[1-3]") == "amr-[1-3]"
