"""Submit's node exclusions are the CLI's four (1.1.6): these nodes, a node type, and on MSU HPCC the AMD20
nodes and the Mendoza group's nodes (`mace submit --exclude-amd20`, `--exclude-mendoza`)."""
import pytest

import submit_util as SU
from submit_util import SP_D12

from mace.gui import submit as sub


def test_msu_groups_are_switches_of_the_command(tmp_path):
    d = tmp_path / "decks"
    d.mkdir()
    (d / "a.d12").write_text(SP_D12)
    sel = sub.selection(tmp_path, [d])
    for kind, flag in (("amd20", "--exclude-amd20"), ("mendoza", "--exclude-mendoza")):
        (c,) = sub.build_commands(tmp_path, "slurm", sel, {"exclude": {"kind": kind, "value": "ignored"}})
        assert c.argv == ["submit", flag, "decks"] and c.display.endswith(f"mace submit {flag} decks")
    with pytest.raises(sub.SubmitError):
        sub.build_commands(tmp_path, "slurm", sel, {"exclude": {"kind": "everything"}})


def test_the_groups_are_offered_where_the_clis_choices_apply(monkeypatch):
    from mace.utils import node_exclusion
    monkeypatch.setattr(node_exclusion, "msu_menus", lambda env=None: True)
    assert [g["kind"] for g in sub.capabilities(False)["exclude_groups"]] == ["amd20", "mendoza"]
    assert all(g["switch"].startswith("--exclude-") and g["label"] for g in node_exclusion.submit_groups())
    monkeypatch.setattr(node_exclusion, "msu_menus", lambda env=None: False)
    assert sub.capabilities(False)["exclude_groups"] == []
